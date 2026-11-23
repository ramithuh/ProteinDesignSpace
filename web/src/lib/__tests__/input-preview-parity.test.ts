import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import {
  resolveGenie3CompiledInput,
  resolveRfd4CompiledInput,
} from '$lib/engine/conditioning';
import {
  clearInputPreviewCache,
  compileNodeInputPreview,
} from '$lib/engine/input-preview';
import {
  applyBackboneNoise,
  buildGapAwareResidueIndices,
  compileMpnnInputSpec,
  parseFixedPositionsAndChains,
  parseTiedPositions,
} from '$lib/engine/mpnn-spec';
import {
  compactResidueTokens,
  formatMultiChainFasta,
  formatOrderReadyAa,
  toggleResidueTokens,
} from '$lib/engine/residue-tokens';
import type {
  CanonicalArtifact,
  FoldedStructure,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

// ---------------------------------------------------------------------------
// Synthetic multi-chain PDB fixture builder (supports negative resnums,
// insertion codes like H100A, missing-density gaps, HETATM ligands & buffers,
// and custom B-factors).
// ---------------------------------------------------------------------------

interface FixtureResidue {
  chain: string;
  resNum: number;
  insCode?: string;
  resName: string;
  bfactor?: number;
  xOffset: number;
}

function padL(s: string, n: number): string {
  return s.length >= n ? s : ' '.repeat(n - s.length) + s;
}

function fmtAtomLine(
  serial: number,
  atomName: string,
  resName: string,
  chain: string,
  resNum: number,
  insCode: string,
  x: number,
  y: number,
  z: number,
  bfactor = 25.0,
  record: 'ATOM  ' | 'HETATM' = 'ATOM  ',
  elem?: string,
): string {
  const atomField = atomName.length >= 4 ? atomName.slice(0, 4) : ` ${atomName.padEnd(3, ' ')}`;
  const el = elem ?? atomName.replace(/[0-9]/g, '').slice(0, 1);
  return (
    `${record}${padL(String(serial), 5)} ${atomField} ${padL(resName, 3)} ${chain.slice(0, 1)}` +
    `${padL(String(resNum), 4)}${(insCode || ' ').slice(0, 1)}   ` +
    `${padL(x.toFixed(3), 8)}${padL(y.toFixed(3), 8)}${padL(z.toFixed(3), 8)}` +
    `  1.00${padL(bfactor.toFixed(2), 6)}          ${padL(el, 2)}`
  );
}

function buildRichMultiChainPdb(opts: { includeLigandsAndBuffers?: boolean } = {}): string {
  const residues: FixtureResidue[] = [
    // Chain A: starts at negative resnum A-2, has gap between A2 and A6 (missing 3,4,5 -> gapBefore=3)
    { chain: 'A', resNum: -2, resName: 'MET', bfactor: 82.5, xOffset: 0 },
    { chain: 'A', resNum: -1, resName: 'ALA', bfactor: 84.0, xOffset: 3.8 },
    { chain: 'A', resNum: 0, resName: 'GLY', bfactor: 85.0, xOffset: 7.6 },
    { chain: 'A', resNum: 1, resName: 'VAL', bfactor: 88.0, xOffset: 11.4 },
    { chain: 'A', resNum: 2, resName: 'LEU', bfactor: 90.0, xOffset: 15.2 },
    // Missing density gap: resNums 3, 4, 5 absent -> A6 has gapBefore = 3
    { chain: 'A', resNum: 6, resName: 'ILE', bfactor: 91.5, xOffset: 19.0 },
    { chain: 'A', resNum: 7, resName: 'LYS', bfactor: 89.0, xOffset: 22.8 },
    { chain: 'A', resNum: 8, resName: 'ASP', bfactor: 87.5, xOffset: 26.6 },
    // Chain H: antibody-style insertion code H100A, H100B
    { chain: 'H', resNum: 99, resName: 'TYR', bfactor: 78.0, xOffset: 40.0 },
    { chain: 'H', resNum: 100, resName: 'TRP', bfactor: 76.5, xOffset: 43.8 },
    { chain: 'H', resNum: 100, insCode: 'A', resName: 'PHE', bfactor: 74.0, xOffset: 47.6 },
    { chain: 'H', resNum: 100, insCode: 'B', resName: 'SER', bfactor: 75.0, xOffset: 51.4 },
    { chain: 'H', resNum: 101, resName: 'THR', bfactor: 79.0, xOffset: 55.2 },
  ];

  const lines: string[] = [];
  let serial = 1;
  for (const r of residues) {
    const x = r.xOffset;
    const bf = r.bfactor ?? 25.0;
    const ins = r.insCode ?? '';
    lines.push(fmtAtomLine(serial++, 'N', r.resName, r.chain, r.resNum, ins, x - 1.2, 0.5, 0.0, bf));
    lines.push(fmtAtomLine(serial++, 'CA', r.resName, r.chain, r.resNum, ins, x, 0.0, 0.0, bf));
    lines.push(fmtAtomLine(serial++, 'C', r.resName, r.chain, r.resNum, ins, x + 1.2, -0.5, 0.2, bf));
    lines.push(fmtAtomLine(serial++, 'O', r.resName, r.chain, r.resNum, ins, x + 1.5, -1.6, 0.4, bf));
  }

  if (opts.includeLigandsAndBuffers) {
    // Biological cofactor SAM + crystallization buffers SO4 and GOL
    lines.push(fmtAtomLine(serial++, 'C1', 'SAM', 'A', 501, '', 10.0, 8.0, 2.0, 30.0, 'HETATM', 'C'));
    lines.push(fmtAtomLine(serial++, 'N1', 'SAM', 'A', 501, '', 11.2, 8.0, 2.0, 30.0, 'HETATM', 'N'));
    lines.push(fmtAtomLine(serial++, 'S', 'SO4', 'A', 601, '', 20.0, 12.0, 0.0, 45.0, 'HETATM', 'S'));
    lines.push(fmtAtomLine(serial++, 'O1', 'SO4', 'A', 601, '', 21.4, 12.0, 0.0, 45.0, 'HETATM', 'O'));
    lines.push(fmtAtomLine(serial++, 'C1', 'GOL', 'H', 701, '', 45.0, 10.0, 1.0, 50.0, 'HETATM', 'C'));
    lines.push(fmtAtomLine(serial++, 'O1', 'GOL', 'H', 701, '', 46.2, 10.0, 1.0, 50.0, 'HETATM', 'O'));
  }

  lines.push('END');
  return lines.join('\n') + '\n';
}

function makeStructureArtifact(
  pdbText: string,
  overrides: Partial<StructureArtifact> = {},
): StructureArtifact {
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id: overrides.id ?? 'struct-rich-1',
    label: overrides.label ?? 'rich_complex.pdb',
    sourceModelId: overrides.sourceModelId ?? 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    chainIdMap: parsed.chainIdMap,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    atomGranularity: 'all-atom',
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// 1. Static Worker-Import Guards (Zero Duplicated Spec Logic)
// ---------------------------------------------------------------------------

describe('Single-source-of-truth worker import guards', () => {
  it('mpnn.worker.ts imports directly from mpnn-spec and has zero duplicated parser/lock functions', () => {
    const src = readFileSync(resolve(process.cwd(), 'src/lib/workers/mpnn.worker.ts'), 'utf8');
    expect(src).toMatch(/from\s+['"]\$lib\/engine\/mpnn-spec['"]/);
    expect(src).toContain('compileMpnnInputSpec');
    expect(src).toContain('applyBackboneNoise');
    expect(src).toContain('buildGapAwareResidueIndices');
    expect(src).not.toMatch(/function\s+parseFixedPositionsAndChains\b/);
    expect(src).not.toMatch(/function\s+parseTiedPositions\b/);
    expect(src).not.toMatch(/function\s+buildGapAwareResidueIndices\b/);
    expect(src).not.toMatch(/function\s+applyBackboneNoise\b/);
  });

  it('rfd4.worker.ts and genie3.worker.ts use resolveRfd4CompiledInput / resolveGenie3CompiledInput', () => {
    const rfd4Src = readFileSync(resolve(process.cwd(), 'src/lib/workers/rfd4.worker.ts'), 'utf8');
    const g3Src = readFileSync(resolve(process.cwd(), 'src/lib/workers/genie3.worker.ts'), 'utf8');
    expect(rfd4Src).toContain('resolveRfd4CompiledInput');
    expect(g3Src).toContain('resolveGenie3CompiledInput');
  });
});

// ---------------------------------------------------------------------------
// 2. ProteinMPNN spec & preview parity
// ---------------------------------------------------------------------------

describe('ProteinMPNN compiled input spec & preview parity', () => {
  const richPdb = buildRichMultiChainPdb({ includeLigandsAndBuffers: true });

  it('parses negative resnums, insertion codes, gaps, and applies deterministic backbone noise', () => {
    const parsedFixed = parseFixedPositionsAndChains('H100A, A-2-0, H', 'A');
    expect(parsedFixed.lockedChains).toEqual(['H']);
    expect(parsedFixed.fixedPositions).toEqual([
      { chain: 'H', number: 100, insertionCode: 'A' },
      { chain: 'A', number: -2 },
      { chain: 'A', number: -1 },
      { chain: 'A', number: 0 },
    ]);

    const tied = parseTiedPositions('A1=H100A; A2=H100B', 'A');
    expect(tied).toEqual([
      [{ chain: 'A', number: 1 }, { chain: 'H', number: 100, insertionCode: 'A' }],
      [{ chain: 'A', number: 2 }, { chain: 'H', number: 100, insertionCode: 'B' }],
    ]);

    const coords = new Float32Array([1, 2, 3, Number.NaN, 5, 6]);
    const noisy1 = applyBackboneNoise(coords, 0.2, 42);
    const noisy2 = applyBackboneNoise(coords, 0.2, 42);
    expect(Array.from(noisy1)).toEqual(Array.from(noisy2));
    expect(Number.isNaN(noisy1[3])).toBe(true);
    expect(noisy1[0]).not.toBe(1);
    expect(applyBackboneNoise(coords, 0, 42)).toBe(coords);
  });

  it('computes exact post-precedence masks for inherited locks, unlockInherited, designChains, ties, and gaps', () => {
    const struct = makeStructureArtifact(richPdb, {
      id: 'mpnn-struct-1',
      lockedChains: ['H'],
      fixedResidueIndices: [4], // 1-based -> flatIndex 3 (A1)
      chainRoles: { A: 'binder', H: 'target' },
    });

    // Case 1: default inherits lockedChains=['H'] and fixedResidueIndices=[4] + explicit user fixed A-2
    const specDefault = compileMpnnInputSpec(struct, {
      fixedPositions: 'A-2',
      omitAminoAcids: 'C, M',
      backboneNoise: 0.15,
    });
    expect(specDefault.lockedChains).toEqual(['H']);
    expect(specDefault.designChains).toEqual(['A']);
    expect(specDefault.omitAminoAcids).toBe('CMX');
    expect(specDefault.backboneNoise).toBe(0.15);

    const resA_m2 = specDefault.residues.find((r) => r.chain === 'A' && r.resNum === -2)!;
    const resA1 = specDefault.residues.find((r) => r.chain === 'A' && r.resNum === 1)!;
    const resA6 = specDefault.residues.find((r) => r.chain === 'A' && r.resNum === 6)!;
    const resH100A = specDefault.residues.find(
      (r) => r.chain === 'H' && r.resNum === 100 && r.insCode === 'A',
    )!;

    expect(resA_m2.willDesign).toBe(false);
    expect(resA_m2.inheritedLock).toBe(false); // explicit user lock
    expect(resA1.willDesign).toBe(false);
    expect(resA1.inheritedLock).toBe(true); // inherited fixedResidueIndices
    expect(resA6.willDesign).toBe(true);
    expect(resA6.gapBefore).toBe(3); // missing resNums 3,4,5
    expect(resH100A.willDesign).toBe(false);
    expect(resH100A.inheritedLock).toBe(true); // inherited lockedChains

    // Gap-aware residue indices preserve true sequence separation across A2 -> A6
    const gapIdx = buildGapAwareResidueIndices(specDefault.structure);
    expect(gapIdx[5] - gapIdx[4]).toBe(4); // A6 - A2 = 4

    // Case 2: unlockInherited=true unlocks both inherited lockedChains and inherited fixedResidueIndices
    const specUnlocked = compileMpnnInputSpec(struct, {
      unlockInherited: true,
      tiedPositions: 'A1=H100A',
    });
    expect(specUnlocked.lockedChains).toEqual([]);
    expect(specUnlocked.designChains).toBeUndefined();
    expect(specUnlocked.designedCount).toBe(struct.length);
    expect(specUnlocked.residues.every((r) => r.willDesign && !r.inheritedLock)).toBe(true);
    const tiedA1 = specUnlocked.residues.find((r) => r.chain === 'A' && r.resNum === 1)!;
    const tiedH100A = specUnlocked.residues.find(
      (r) => r.chain === 'H' && r.resNum === 100 && r.insCode === 'A',
    )!;
    expect(tiedA1.tieGroupId).toBe(1);
    expect(tiedH100A.tieGroupId).toBe(1);

    // Case 3: explicit designChains overrides inherited lockedChains=['H']
    const specExplicitChains = compileMpnnInputSpec(struct, {
      designChains: 'H',
    });
    expect(specExplicitChains.designChains).toEqual(['H']);
    expect(
      specExplicitChains.residues.filter((r) => r.chain === 'H').every((r) => r.willDesign),
    ).toBe(true);
    expect(
      specExplicitChains.residues.filter((r) => r.chain === 'A').every((r) => !r.willDesign),
    ).toBe(true);
    // Excluded by explicit designChains is NOT an inheritedLock
    expect(
      specExplicitChains.residues.find((r) => r.chain === 'A' && r.resNum === -2)!.inheritedLock,
    ).toBe(false);

    // Verify 1:1 parity with compileNodeInputPreview
    const graph: WorkflowGraph = {
      id: 'g-mpnn',
      name: 'MPNN Parity',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: richPdb } },
        {
          id: 'mpnn',
          adapterId: 'web-protein-mpnn',
          params: {
            family: 'solublempnn',
            modelName: 'v_48_020',
            fixedPositions: 'A-2',
            tiedPositions: 'A2=H100B',
            omitAminoAcids: 'C',
            backboneNoise: 0.2,
          },
        },
      ],
      edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' }],
    };

    const preview = compileNodeInputPreview(graph, 'mpnn', { src: { structure: struct } });
    expect(preview.availability).toBe('ready');
    expect(preview.payload.format).toBe('mpnn-masks');
    expect(preview.entities.includedLigands).toContain('SAM');
    expect(preview.entities.filteredBuffers).toEqual(expect.arrayContaining(['SO4', 'GOL']));

    // Because H100B is locked (inherited chain H) and tied to A2, A2 is also pinned!
    const pA2 = preview.chains[0].residues.find((r) => r.token === 'A2')!;
    expect(pA2.tieGroupId).toBe(1);
    expect(pA2.roles).toContain('tied');
    expect(pA2.roles).toContain('fixed-sequence');
    expect(preview.chains[0].residues.find((r) => r.token === 'A6')!.gapBefore).toBe(3);
    expect(preview.chains[1].residues.find((r) => r.token === 'H100A')!.inheritedLock).toBe(true);
    expect(preview.chains[0].residues[0].bfactor).toBeCloseTo(82.5, 1);
    expect(preview.chains[0].residues[0].backboneAtoms).toBe('N/CA/C/O');

    // Highlight layers include fixed-sequence (#eab308) and tied (#a855f7)
    const fixedLayer = preview.highlightLayers.find((l) => l.role === 'fixed-sequence')!;
    const tiedLayer = preview.highlightLayers.find((l) => l.role === 'tied')!;
    expect(fixedLayer.colorHex).toBe(0xeab308);
    expect(fixedLayer.tokens).toContain('H100A');
    expect(fixedLayer.tokens).toContain('A-2');
    expect(tiedLayer.colorHex).toBe(0xa855f7);
    expect(tiedLayer.tokens).toEqual(expect.arrayContaining(['A2', 'H100B']));
  });
});

// ---------------------------------------------------------------------------
// 3. All 7 Adapters Parity Tests + Pending-Upstream & Determinism
// ---------------------------------------------------------------------------

describe('compileNodeInputPreview across all 7 adapters', () => {
  const richPdb = buildRichMultiChainPdb({ includeLigandsAndBuffers: true });
  const proteinOnlyPdb = buildRichMultiChainPdb({ includeLigandsAndBuffers: false });

  it('pdb-input: handles demo-bundle, insertion codes, buffers, and cropSegments ghosting', () => {
    clearInputPreviewCache();
    const demoGraph: WorkflowGraph = {
      id: 'g-pdb-demo',
      name: 'PDB Demo',
      nodes: [{ id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } }],
      edges: [],
    };
    const demoPrev = compileNodeInputPreview(demoGraph, 'src', {});
    expect(demoPrev.availability).toBe('ready');
    expect(demoPrev.budget.totalResidues).toBe(53);
    expect(demoPrev.chains[0].residues[0].backboneAtoms).toBe('N/CA/C/O');

    const fullGraph: WorkflowGraph = {
      id: 'g-pdb-full',
      name: 'PDB Full',
      nodes: [
        {
          id: 'src',
          adapterId: 'pdb-input',
          params: { mode: 'custom', filename: 'rich.pdb', pdbText: richPdb },
        },
      ],
      edges: [],
    };
    const fullPrev = compileNodeInputPreview(fullGraph, 'src', {});
    expect(fullPrev.availability).toBe('ready');
    expect(fullPrev.budget.totalResidues).toBe(13);
    expect(fullPrev.entities.includedLigands).toContain('SAM');
    expect(fullPrev.entities.filteredBuffers).toEqual(expect.arrayContaining(['SO4', 'GOL']));

    const cropGraph: WorkflowGraph = {
      id: 'g-pdb-crop',
      name: 'PDB Crop',
      nodes: [
        {
          id: 'src',
          adapterId: 'pdb-input',
          params: {
            mode: 'custom',
            filename: 'rich.pdb',
            pdbText: richPdb,
            cropSegments: 'A-2-8, A:SAM',
          },
        },
      ],
      edges: [],
    };
    const cropPrev = compileNodeInputPreview(cropGraph, 'src', {});
    expect(cropPrev.availability).toBe('ready');
    expect(cropPrev.budget.totalResidues).toBe(8);
    expect(cropPrev.entities.includedLigands).toContain('SAM');
    expect(cropPrev.entities.filteredBuffers).toEqual([]);

    // Chain H residues are marked cropped-out with flatIndex=-1 and alpha=0.22 highlight layer
    const chainH = cropPrev.chains.find((c) => c.chainId === 'H')!;
    expect(chainH.residues.every((r) => r.flatIndex === -1 && r.roles.includes('cropped-out'))).toBe(true);
    const cropLayer = cropPrev.highlightLayers.find((l) => l.role === 'cropped-out')!;
    expect(cropLayer.colorHex).toBe(0x94a3b8);
    expect(cropLayer.alpha).toBe(0.22);
    expect(cropLayer.tokens).toContain('H100A');
  });

  it('sequence-input: splits multi-chain FASTA and flags empty invalid input', () => {
    const graph: WorkflowGraph = {
      id: 'g-seq',
      name: 'Seq',
      nodes: [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          params: {
            sequence: '>binder_complex\nMKTAYIAKQR:GVLIKDTRWY\n>cand2\nACDEFGHIKL:MNPQRSTVWY',
          },
        },
      ],
      edges: [],
    };
    const prev = compileNodeInputPreview(graph, 'seq', {});
    expect(prev.availability).toBe('ready');
    expect(prev.chains).toHaveLength(2);
    expect(prev.chains[0].chainId).toBe('A');
    expect(prev.chains[1].chainId).toBe('B');
    expect(prev.fastaEntries).toHaveLength(2);
    expect(prev.budget.totalResidues).toBe(20);

    const badGraph: WorkflowGraph = {
      ...graph,
      nodes: [{ id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MK' } }],
    };
    const badPrev = compileNodeInputPreview(badGraph, 'seq', {});
    expect(badPrev.availability).toBe('error');
    expect(badPrev.issues.some((i) => i.severity === 'error')).toBe(true);
  });

  it('rfd4-proteina: matches resolveRfd4CompiledInput across binder (buffers/hotspots/cropped), motif, and partial encoder VRAM', () => {
    const struct = makeStructureArtifact(richPdb, { id: 'rfd4-tpl-1' });
    const binderParams = {
      mode: 'binder',
      length: 50,
      lockChain: 'A',
      hotspots: 'A-2, A6',
      nsteps: 60,
      recycleSchedule: 'low',
      seed: 42,
    };
    const direct = resolveRfd4CompiledInput(binderParams, struct);
    expect(direct.contig).toContain('A1-8,/0,50-50');
    expect(direct.conditions).toHaveProperty('C_HOT');
    expect(direct.conditions).toHaveProperty('C_CRD');
    expect(direct.conditions).toHaveProperty('C_SEQ');

    const graph: WorkflowGraph = {
      id: 'g-rfd4-binder',
      name: 'RFD4 Binder',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: richPdb } },
        { id: 'rfd4', adapterId: 'rfd4-proteina', params: binderParams },
      ],
      edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'rfd4', toPort: 'structure' }],
    };
    const prev = compileNodeInputPreview(graph, 'rfd4', { src: { structure: struct } });
    expect(prev.availability).toBe('ready');
    expect(prev.payload.format).toBe('rfd4-spec');
    expect(prev.payload.compiledSummary.contig).toBe(direct.contig);
    expect(prev.entities.includedLigands).toContain('SAM');
    expect(prev.entities.filteredBuffers).toEqual(expect.arrayContaining(['SO4', 'GOL']));

    // Hotspots layer (#f97316 + accentSticks), fixed-backbone (#64748b), fixed-sequence (#eab308), cropped-out (#94a3b8 alpha 0.22)
    const hotLayer = prev.highlightLayers.find((l) => l.role === 'hotspot')!;
    expect(hotLayer.colorHex).toBe(0xf97316);
    expect(hotLayer.accentSticks).toBe(true);
    expect(hotLayer.tokens).toEqual(expect.arrayContaining(['A-2', 'A6']));
    const cropLayer = prev.highlightLayers.find((l) => l.role === 'cropped-out')!;
    expect(cropLayer.colorHex).toBe(0x94a3b8);
    expect(cropLayer.alpha).toBe(0.22);
    expect(cropLayer.tokens).toContain('H100A');
    expect(prev.budget.designedResidues).toBe(50);
    expect(prev.budget.fixedResidues).toBe(8);

    // Partial diffusion adds +138 MB encoder weights and +400 MB co-resident VRAM (900 + 400 = 1300)
    const partialGraph: WorkflowGraph = {
      ...graph,
      nodes: [
        graph.nodes[0],
        { id: 'rfd4', adapterId: 'rfd4-proteina', params: { mode: 'partial', partialFraction: 0.4 } },
      ],
    };
    const partialPrev = compileNodeInputPreview(partialGraph, 'rfd4', { src: { structure: struct } });
    expect(partialPrev.availability).toBe('ready');
    expect(partialPrev.budget.nodeWeightsMB).toBe(330 + 138);
    expect(partialPrev.budget.nodeVramMB).toBe(900 + 400);

    // Unindexed motif mode highlights motif with #14b8a6 + accentSticks
    const motifGraph: WorkflowGraph = {
      ...graph,
      nodes: [
        graph.nodes[0],
        {
          id: 'rfd4',
          adapterId: 'rfd4-proteina',
          params: { mode: 'motif', length: 60, motifSpan: 'H100A, H100B', unindexedMotif: true },
        },
      ],
    };
    const motifPrev = compileNodeInputPreview(motifGraph, 'rfd4', { src: { structure: struct } });
    expect(motifPrev.availability).toBe('ready');
    const motifLayer = motifPrev.highlightLayers.find((l) => l.role === 'motif')!;
    expect(motifLayer.colorHex).toBe(0x14b8a6);
    expect(motifLayer.accentSticks).toBe(true);
    expect(motifLayer.tokens).toEqual(expect.arrayContaining(['H100A', 'H100B']));
  });

  it('genie3: enforces midpoint contig resolution and binder-first Chain A / Chain B labeling', () => {
    const struct = makeStructureArtifact(proteinOnlyPdb, { id: 'g3-tpl-1' });

    // Binder mode: Chain A = binder (40 aa), Chain B = fixed target (Chain A of template, 8 aa)
    const binderParams = {
      taskMode: 'binder',
      length: 40,
      lockChain: 'A',
      hotspots: 'A-2, A6',
      nSampleStep: 100,
    };
    const directBinder = resolveGenie3CompiledInput(binderParams, struct);
    expect(directBinder.mode).toBe('binder');
    expect(directBinder.chainLengths).toEqual([40, 8]);

    const binderGraph: WorkflowGraph = {
      id: 'g-g3-binder',
      name: 'Genie3 Binder',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: proteinOnlyPdb } },
        { id: 'g3', adapterId: 'genie3', params: binderParams },
      ],
      edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    };
    const binderPrev = compileNodeInputPreview(binderGraph, 'g3', { src: { structure: struct } });
    expect(binderPrev.availability).toBe('ready');
    expect(binderPrev.payload.format).toBe('genie3-spec');
    expect(binderPrev.chains[0].chainId).toBe('A');
    expect(binderPrev.chains[0].roleTag).toBe('binder');
    expect(binderPrev.chains[0].residues).toHaveLength(40);
    const targetTrack = binderPrev.chains.find((c) => c.roleTag === 'target')!;
    expect(targetTrack.chainId).toBe('A');
    expect(targetTrack.labelChainId).toBe('B');
    expect(targetTrack.residues[0].labelChain).toBe('B');
    expect(binderPrev.budget.totalResidues).toBe(48);
    expect(binderPrev.budget.designedResidues).toBe(40);
    expect(binderPrev.budget.fixedResidues).toBe(8);
    const hotLayer = binderPrev.highlightLayers.find((l) => l.role === 'hotspot')!;
    expect(hotLayer.colorHex).toBe(0xf97316);
    expect(hotLayer.accentSticks).toBe(true);
    expect(hotLayer.tokens).toEqual(expect.arrayContaining(['A-2', 'A6']));

    // Motif mode with range contig "10-20,H100-100B,10-20" -> midpoint 15 + 3 + 15 = 33 aa
    const motifParams = {
      taskMode: 'motif',
      contig: '10-20,H100-100B,10-20',
      motifSpan: 'H100-100B',
      nSampleStep: 100,
    };
    const directMotif = resolveGenie3CompiledInput(motifParams, struct);
    expect(directMotif.mode).toBe('motif');
    expect(directMotif.totalLength).toBe(33);

    const motifGraph: WorkflowGraph = {
      ...binderGraph,
      nodes: [binderGraph.nodes[0], { id: 'g3', adapterId: 'genie3', params: motifParams }],
    };
    const motifPrev = compileNodeInputPreview(motifGraph, 'g3', { src: { structure: struct } });
    expect(motifPrev.availability).toBe('ready');
    expect(motifPrev.budget.totalResidues).toBe(33);
    expect(motifPrev.budget.fixedResidues).toBe(3);
    expect(motifPrev.budget.designedResidues).toBe(30);
    const motifLayer = motifPrev.highlightLayers.find((l) => l.role === 'motif')!;
    expect(motifLayer.colorHex).toBe(0x14b8a6);
    expect(motifLayer.accentSticks).toBe(true);
    expect(motifLayer.tokens).toEqual(['H100', 'H100A', 'H100B']);
  });

  it.each(['ef2-fast-300m', 'ef2-fast-600m'])('previews ESMFold2 chains and its own schedule for %s', (model) => {
    const graph: WorkflowGraph = {
      id: 'ef2-preview', name: 'EF2 preview',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'ACD:EFG' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model, esmfold2Steps: 32 } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const sequences: SequenceSet = {
      kind: 'SequenceSet', id: 'seqs', parentStructureId: 'none', sourceModelId: 'sequence-input',
      candidates: [{ id: 'c', sequence: 'ACD:EFG', sourceModelId: 'sequence-input', score: null, globalScore: null }],
    };
    const preview = compileNodeInputPreview(graph, 'fold', { seq: { sequences } });
    expect(preview.availability).toBe('ready');
    expect(preview.chains.map((c) => c.sequence)).toEqual(['ACD', 'EFG']);
    expect(preview.payload.compiledSummary).toMatchObject({ model, esmfold2Steps: 32, chainLengths: ['3', '3'] });
    for (const field of ['tolerance', 'recycles', 'diffusionSteps', 'maxMsaSequences']) {
      expect(preview.payload.compiledSummary).not.toHaveProperty(field);
    }
  });

  it('localfold-af2: validates target-only MSA masking, custom .a3m parsing, maxSequences, and AF3 ligands', () => {
    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'seqset-binder-1',
      parentStructureId: 'gen-1',
      sourceModelId: 'solublempnn/v_48_020',
      lockedChains: ['B'],
      chainRoles: { A: 'binder', B: 'target' },
      candidates: [
        { id: 'mpnn-1', sequence: 'MKTAYIAKQRGVLIKDTRWY', chainLengths: [10, 10], score: 1.12, globalScore: 1.2, sourceModelId: 'solublempnn' },
        { id: 'mpnn-2', sequence: 'ACDEFGHIKLGVLIKDTRWY', chainLengths: [10, 10], score: 1.35, globalScore: 1.4, sourceModelId: 'solublempnn' },
        { id: 'mpnn-3', sequence: 'MNPQRSTVWYGVLIKDTRWY', chainLengths: [10, 10], score: 1.65, globalScore: 1.7, sourceModelId: 'solublempnn' },
      ],
    };

    const graph: WorkflowGraph = {
      id: 'g-lf',
      name: 'LocalFold Target-Only',
      nodes: [
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: {} },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            model: 'multimer',
            msaMode: 'target-only',
            recycles: 3,
            maxSequences: 2,
          },
        },
      ],
      edges: [{ fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };

    const prev = compileNodeInputPreview(graph, 'fold', { mpnn: { sequences: seqSet } });
    expect(prev.availability).toBe('ready');
    expect(prev.payload.format).toBe('localfold-batch');
    expect(prev.chains[0].chainId).toBe('A');
    expect(prev.chains[0].msaStatus).toBe('target-only-masked');
    expect(prev.chains[0].msaDepth).toBe(1);
    expect(prev.chains[1].chainId).toBe('B');
    expect(prev.chains[1].msaStatus).toBe('mmseqs2-search');
    expect(prev.fastaEntries).toHaveLength(3);
    expect(prev.fastaEntries.filter((f) => f.selected)).toHaveLength(2);
    expect(prev.fastaEntries[2].selected).toBe(false);

    // Valid custom .a3m alignment
    const a3mGraph: WorkflowGraph = {
      ...graph,
      nodes: [
        graph.nodes[0],
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            model: 'boltz2',
            msaMode: 'a3m',
            msaA3mText: '>query\nMKTAYIAKQRGVLIKDTRWY\n>hit1\nMKTAYIAKQRGVLIKDTRWF\n>hit2\nMKTAYIAKQRGVLIKDTRWY\n',
            ligandCcdCodes: 'SAM, ZN',
            dnaSequences: 'ACGTACGT',
          },
        },
      ],
    };
    const a3mPrev = compileNodeInputPreview(a3mGraph, 'fold', { mpnn: { sequences: seqSet } });
    expect(a3mPrev.availability).toBe('ready');
    expect(a3mPrev.chains[0].msaStatus).toBe('a3m-provided');
    expect(a3mPrev.chains[0].msaDepth).toBe(3);
    expect(a3mPrev.entities.includedLigands).toEqual(['SAM', 'ZN']);
    expect(a3mPrev.entities.nucleicChains).toHaveLength(1);

    // Mismatched .a3m surfaces compiler error
    const badA3mGraph: WorkflowGraph = {
      ...graph,
      nodes: [
        graph.nodes[0],
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: { model: 'monomer', msaMode: 'a3m', msaA3mText: '>query\nAAAAA\n' },
        },
      ],
    };
    const badA3mPrev = compileNodeInputPreview(badA3mGraph, 'fold', { mpnn: { sequences: seqSet } });
    expect(badA3mPrev.availability).toBe('error');
    expect(badA3mPrev.issues.some((i) => i.source === 'compiler' && i.severity === 'error')).toBe(true);
  });

  it('designability-scorer: verifies self-consistency, explicit residueMapping, and pending-upstream state', () => {
    const struct = makeStructureArtifact(proteinOnlyPdb, { id: 'gen-struct-1' });
    const folds: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'folds-1',
      parentStructureId: 'gen-struct-1',
      sourceStructureId: 'gen-struct-1',
      folds: [
        {
          sequenceId: 'mpnn-1',
          sequence: struct.sequence,
          foldModelId: 'localfold-af2-monomer-int5',
          sourceStructureId: 'gen-struct-1',
          pdbText: struct.pdbText,
          caCoords: struct.caCoords,
          plddt: new Float32Array(struct.length).fill(85),
          meanPlddt: 85,
          ptm: 0.82,
          elapsedMs: 120,
        },
      ],
    };

    const graph: WorkflowGraph = {
      id: 'g-score',
      name: 'Scorer',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 70 } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: {} },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer' } },
        { id: 'score', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };

    // Pending upstream when GPU nodes haven't run yet
    const pendingScore = compileNodeInputPreview(graph, 'score', {});
    expect(pendingScore.availability).toBe('pending-upstream');
    expect(pendingScore.missingUpstreamNodeIds).toEqual(expect.arrayContaining(['gen', 'fold']));
    expect(pendingScore.budget.subgraphDownloadMB).toBeGreaterThanOrEqual(330 + 10 + 97);

    // Pending upstream on MPNN before RFD4 runs
    const pendingMpnn = compileNodeInputPreview(graph, 'mpnn', {});
    expect(pendingMpnn.availability).toBe('pending-upstream');
    expect(pendingMpnn.missingUpstreamNodeIds).toEqual(['gen']);
    expect(pendingMpnn.budget.nodeWeightsMB).toBe(10);
    expect(pendingMpnn.budget.subgraphDownloadMB).toBe(330 + 10);

    // Ready scorer with self-consistency
    const outputs: Record<string, Record<string, CanonicalArtifact>> = {
      gen: { structure: struct },
      fold: { folds },
    };
    const readyScore = compileNodeInputPreview(graph, 'score', outputs);
    expect(readyScore.availability).toBe('ready');
    expect(readyScore.payload.format).toBe('scorer-mapping');
    expect(readyScore.payload.compiledSummary.referenceRelation).toBe('self-consistency');

    // Explicit residueMapping highlights mapped residues and ghosts unmapped residues
    const mappedGraph: WorkflowGraph = {
      ...graph,
      nodes: [
        ...graph.nodes.slice(0, 3),
        { id: 'score', adapterId: 'designability-scorer', params: { residueMapping: '0:0, 1:1, 2:2, 3:3' } },
      ],
    };
    const mappedScore = compileNodeInputPreview(mappedGraph, 'score', outputs);
    expect(mappedScore.availability).toBe('ready');
    expect(mappedScore.payload.compiledSummary.referenceRelation).toBe('external-mapped');
    expect(mappedScore.highlightLayers.find((l) => l.role === 'motif')!.tokens).toHaveLength(4);
    expect(mappedScore.highlightLayers.find((l) => l.role === 'cropped-out')!.alpha).toBe(0.22);

    // Deterministic previewDigest
    const again = compileNodeInputPreview(mappedGraph, 'score', outputs);
    expect(again.previewDigest).toBe(mappedScore.previewDigest);
    expect(again.previewDigest).not.toBe(readyScore.previewDigest);
  });
});

// ---------------------------------------------------------------------------
// 4. Round-trip residue-tokens.ts tests
// ---------------------------------------------------------------------------

describe('residue-tokens utilities round-trip', () => {
  it('compactResidueTokens & toggleResidueTokens preserve insertion codes, negative resnums, and ranges', () => {
    const compacted = compactResidueTokens([
      { chain: 'A', seqId: -2 },
      { chain: 'A', seqId: -1 },
      { chain: 'A', seqId: 0 },
      { chain: 'A', seqId: 10 },
      { chain: 'A', seqId: 11 },
      { chain: 'H', seqId: 100, insCode: 'A' },
      { chain: 'H', seqId: 100, insCode: 'B' },
    ]);
    expect(compacted).toBe('A-2-0, A10, A11, H100A, H100B');

    // Toggle adds new residues including H100A and collapses contiguous runs
    const added = toggleResidueTokens(
      'A10, A11',
      [
        { authChain: 'A', authSeqId: 12, insCode: '' },
        { authChain: 'H', authSeqId: 100, insCode: 'A' },
      ],
      'add',
    );
    expect(added).toBe('A10-12, H100A');

    // Toggle on an already-present selection removes it cleanly
    const toggledOff = toggleResidueTokens(
      added,
      [{ authChain: 'H', authSeqId: 100, insCode: 'A' }],
      'toggle',
    );
    expect(toggledOff).toBe('A10-12');

    // Removing middle residue splits the range
    const split = toggleResidueTokens(
      'A10-14',
      [{ authChain: 'A', authSeqId: 12, insCode: '' }],
      'remove',
    );
    expect(split).toBe('A10, A11, A13, A14');
  });

  it('formatMultiChainFasta and formatOrderReadyAa produce clean wet-lab exports', () => {
    const fasta = formatMultiChainFasta('design1', [
      { chainId: 'A', roleTag: 'binder', sequence: 'MKTAYIAKQR' },
      { chainId: 'B', roleTag: 'target', sequence: 'GVLIKDTRWY' },
    ]);
    expect(fasta).toBe('>design1|chain_A|binder\nMKTAYIAKQR\n>design1|chain_B|target\nGVLIKDTRWY');

    const colonFasta = formatMultiChainFasta(
      'design1',
      [
        { chainId: 'A', sequence: 'MKTAY' },
        { chainId: 'B', sequence: 'GVLIK' },
      ],
      { colonMultimer: true },
    );
    expect(colonFasta).toBe('>design1\nMKTAY:GVLIK');

    expect(formatOrderReadyAa('ak-x123:def')).toBe('AKDEF');
    expect(formatOrderReadyAa('ak-x123:def', true)).toBe('MAKDEF');
    expect(formatOrderReadyAa('MKDEF', true)).toBe('MKDEF');
  });

  it('cifText-only StructureArtifact previews, structure-compare multi-port merge, and pdb-input benchmarkId partial preview', () => {
    clearInputPreviewCache();
    const cifHolo = [
      'data_test_holo',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 N MET A 1 0.000 0.000 0.000',
      'ATOM 2 CA MET A 1 1.458 0.000 0.000',
      'ATOM 3 C MET A 1 2.009 1.420 0.000',
      'ATOM 4 O MET A 1 1.251 2.390 0.000',
      'ATOM 5 N ALA A 2 3.320 1.520 0.000',
      'ATOM 6 CA ALA A 2 3.980 2.820 0.000',
      'ATOM 7 C ALA A 2 5.490 2.680 0.000',
      'ATOM 8 O ALA A 2 6.050 1.580 0.000',
      'ATOM 9 N VAL A 3 7.320 1.520 0.000',
      'ATOM 10 CA VAL A 3 7.980 2.820 0.000',
      'ATOM 11 C VAL A 3 9.490 2.680 0.000',
      'ATOM 12 O VAL A 3 10.050 1.580 0.000',
      'HETATM 13 C1 DHZ A 300 12.000 1.000 0.000',
      'HETATM 14 C2 DHZ A 300 13.400 1.000 0.000',
    ].join('\n');
    const parsedCif = parseStructureText(cifHolo);
    const cifOnlyArtifact: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'cif-only-holo-1',
      label: 'CIF Only Holo',
      sourceModelId: 'pdb-input',
      length: parsedCif.nResidues,
      sequence: parsedCif.sequence,
      caCoords: parsedCif.caCoords,
      backbone4Coords: parsedCif.backbone4Coords,
      pdbText: '',
      cifText: cifHolo,
      chainIds: parsedCif.chainIds,
      chainLengths: parsedCif.diagnostics.chainLengths,
      insCodes: parsedCif.insCodes,
      sourceResidues: parsedCif.sourceResidues,
      chainIdMap: parsedCif.chainIdMap,
      atomPresence: parsedCif.atomPresence,
      observedBackboneMask: parsedCif.observedBackboneMask,
      diagnostics: parsedCif.diagnostics,
      atomGranularity: 'all-atom',
      geometry: computeGeometryStats(parsedCif.caCoords),
    };
    const cifOnlyNoDiag: StructureArtifact = {
      ...cifOnlyArtifact,
      id: 'cif-only-nodiag-1',
      diagnostics: undefined,
    };

    const apoPdb = [
      'ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  ',
      'ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00 20.00           C  ',
      'ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00 20.00           C  ',
      'ATOM      4  O   MET A   1       1.251   2.390   0.000  1.00 20.00           O  ',
      'ATOM      5  N   ALA A   2       3.320   1.520   0.000  1.00 20.00           N  ',
      'ATOM      6  CA  ALA A   2       3.980   2.820   0.000  1.00 20.00           C  ',
      'ATOM      7  C   ALA A   2       5.490   2.680   0.000  1.00 20.00           C  ',
      'ATOM      8  O   ALA A   2       6.050   1.580   0.000  1.00 20.00           O  ',
      'ATOM      9  N   VAL A   3       7.320   1.520   0.000  1.00 20.00           N  ',
      'ATOM     10  CA  VAL A   3       7.980   2.820   0.000  1.00 20.00           C  ',
      'ATOM     11  C   VAL A   3       9.490   2.680   0.000  1.00 20.00           C  ',
      'ATOM     12  O   VAL A   3      10.050   1.580   0.000  1.00 20.00           O  ',
      'END',
    ].join('\n');
    const apoArtifact = makeStructureArtifact(apoPdb, { id: 'apo-ref-1', label: 'Apo Ref' });

    const folds: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'folds-cif-1',
      parentStructureId: cifOnlyNoDiag.id,
      sourceStructureId: cifOnlyNoDiag.id,
      folds: [
        {
          sequenceId: 'cand-1',
          sequence: cifOnlyNoDiag.sequence,
          foldModelId: 'af3',
          sourceStructureId: cifOnlyNoDiag.id,
          pdbText: apoPdb,
          caCoords: cifOnlyNoDiag.caCoords,
          plddt: new Float32Array(cifOnlyNoDiag.length).fill(90),
          meanPlddt: 90,
          ptm: 0.9,
          elapsedMs: 50,
        },
      ],
    };

    // 1. designability-scorer with cifText-only StructureArtifact (including missing diagnostics)
    const scorerGraph: WorkflowGraph = {
      id: 'g-cif-scorer',
      name: 'CIF Scorer',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1CTT' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'af3' } },
        { id: 'score', adapterId: 'designability-scorer', params: {} },
      ],
      edges: [
        { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };
    const scorerPrev = compileNodeInputPreview(scorerGraph, 'score', {
      src: { structure: cifOnlyNoDiag },
      fold: { folds },
    });
    expect(scorerPrev.availability).toBe('ready');
    expect(scorerPrev.chains.length).toBeGreaterThan(0);
    expect(scorerPrev.previewStructurePdb).toBe(cifHolo);
    expect(scorerPrev.entities.includedLigands).toEqual(['DHZ']);

    // 2. structure-compare with cifText-only reference + multiple structures
    const cmpGraph: WorkflowGraph = {
      id: 'g-cif-cmp',
      name: 'CIF Compare',
      nodes: [
        { id: 'ref', adapterId: 'pdb-input', params: {} },
        { id: 'm1', adapterId: 'localfold-af2', params: { model: 'af3' } },
        { id: 'cmp', adapterId: 'structure-compare', params: {} },
      ],
      edges: [
        { fromNode: 'ref', fromPort: 'structure', toNode: 'cmp', toPort: 'structures' },
        { fromNode: 'm1', fromPort: 'structure', toNode: 'cmp', toPort: 'structures' },
      ],
    };
    const cmpPrev = compileNodeInputPreview(cmpGraph, 'cmp', {
      ref: { structure: apoArtifact },
      m1: { structure: cifOnlyArtifact },
    });
    expect(cmpPrev.availability).toBe('ready');
    expect(cmpPrev.chains.length).toBeGreaterThan(0);
    expect(cmpPrev.previewStructurePdb).toBeTruthy();
    expect(cmpPrev.fastaEntries).toHaveLength(2);
    expect(cmpPrev.entities.includedLigands).toEqual(['DHZ']);

    const cmpSingleCifPrev = compileNodeInputPreview(
      {
        ...cmpGraph,
        edges: [{ fromNode: 'm1', fromPort: 'structure', toNode: 'cmp', toPort: 'structures' }],
      },
      'cmp',
      { m1: { structure: cifOnlyArtifact } },
    );
    expect(cmpSingleCifPrev.chains.length).toBeGreaterThan(0);
    expect(cmpSingleCifPrev.previewStructurePdb).toBe(cifHolo);

    // 3. thermompnn, antifold-esmif1, genie2, genie3, protpardelle with cifText-only StructureArtifact
    const singleStructAdapters: Array<{ adapterId: string; params: Record<string, unknown>; port: string }> = [
      { adapterId: 'thermompnn', params: {}, port: 'structure' },
      { adapterId: 'antifold-esmif1', params: { variant: 'esmif1' }, port: 'structure' },
      { adapterId: 'genie2', params: { motifRange: 'A1-2', length: 60 }, port: 'template' },
      {
        adapterId: 'genie3',
        params: { taskMode: 'motif', contig: '10-20,A1-2,10-20', motifSpan: 'A1-2' },
        port: 'structure',
      },
      { adapterId: 'protpardelle', params: { taskMode: 'partial' }, port: 'structure' },
      { adapterId: 'protpardelle', params: { taskMode: 'motif', motifSelection: 'A1-2', length: 60 }, port: 'structure' },
    ];
    for (const cfg of singleStructAdapters) {
      clearInputPreviewCache();
      const g: WorkflowGraph = {
        id: `g-cif-${cfg.adapterId}`,
        name: cfg.adapterId,
        nodes: [
          { id: 'src', adapterId: 'pdb-input', params: {} },
          { id: 'dst', adapterId: cfg.adapterId, params: cfg.params },
        ],
        edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'dst', toPort: cfg.port }],
      };
      const p = compileNodeInputPreview(g, 'dst', { src: { structure: cifOnlyArtifact } });
      expect(p.previewStructurePdb, `${cfg.adapterId} should populate previewStructurePdb from cifText`).toBe(cifHolo);
    }

    // 4. pdb-input partial preview falls back to benchmarkId cropSegments when cropSegments param is empty
    const benchPartialGraph: WorkflowGraph = {
      id: 'g-pdb-bench-partial',
      name: 'PDB Bench Partial',
      nodes: [
        {
          id: 'src',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda', cropSegments: '' },
        },
      ],
      edges: [],
    };
    const benchPartial = compileNodeInputPreview(benchPartialGraph, 'src', {});
    expect(benchPartial.availability).toBe('partial');
    expect(benchPartial.entities.includedLigands).toEqual(['ZN', 'DHZ']);
  });
});
