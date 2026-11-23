import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  buildTrajectoryMultiModelPdb,
  findContextStructure,
  inferTrajectoryResidueCount,
} from '$lib/engine/peek';
import { buildComparisonState } from '$lib/engine/compare';
import { makeDesignKey, makeFoldKey, makeSequenceKey, normalizeToCollection } from '$lib/engine/collections';
import { exportCollectionCsv } from '$lib/engine/export-collection';
import { selectFoldWorkItems } from '$lib/engine/bio-capabilities';
import { extractRfd4TemplateContext, resolveRfd4CompiledInput } from '$lib/engine/conditioning';
import { computeUpstreamLockSummary } from '$lib/ui/node-display-helpers';
import { formatRfd4AuthorRunLabel } from '$lib/plugins/rfd4-proteina/rfd4.worker';
import { applyTransformToPdb, computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parsePdbBackbone, type SourceResidueId } from '$lib/protein-ops/pdb';
import type {
  CanonicalArtifact,
  DesignabilityMetrics,
  FoldedStructure,
  SequenceSet,
  StructureArtifact,
  TrajectoryFramesArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

function makeHelix(n: number, ox: number, oy: number, oz: number, radius = 2.3, pitch = 1.5): Float32Array {
  const c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const theta = i * 1.745;
    c[i * 3 + 0] = ox + radius * Math.cos(theta);
    c[i * 3 + 1] = oy + radius * Math.sin(theta);
    c[i * 3 + 2] = oz + i * pitch;
  }
  return c;
}

function makeBetaHairpin(n: number, ox: number, oy: number, oz: number): Float32Array {
  const c = new Float32Array(n * 3);
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    if (i < half) {
      c[i * 3 + 0] = ox + i * 3.3;
      c[i * 3 + 1] = oy + (i % 2 === 0 ? 0.6 : -0.6);
      c[i * 3 + 2] = oz;
    } else {
      const k = i - half;
      c[i * 3 + 0] = ox + (half - 1 - k) * 3.3;
      c[i * 3 + 1] = oy + 5.2 + (k % 2 === 0 ? 0.6 : -0.6);
      c[i * 3 + 2] = oz + 1.4;
    }
  }
  return c;
}

const AA1_TO_3: Record<string, string> = {
  A: 'ALA', C: 'CYS', D: 'ASP', E: 'GLU', F: 'PHE',
  G: 'GLY', H: 'HIS', I: 'ILE', K: 'LYS', L: 'LEU',
  M: 'MET', N: 'ASN', P: 'PRO', Q: 'GLN', R: 'ARG',
  S: 'SER', T: 'THR', V: 'VAL', W: 'TRP', Y: 'TYR',
};

function buildTwoChainPdb(
  firstCa: Float32Array,
  firstSeq: string,
  secondCa: Float32Array,
  secondSeq: string,
): string {
  const lines: string[] = [];
  let serial = 1;
  const emit = (ca: Float32Array, seq: string, chain: string) => {
    for (let i = 0; i < seq.length; i++) {
      const x = ca[i * 3].toFixed(3).padStart(8);
      const y = ca[i * 3 + 1].toFixed(3).padStart(8);
      const z = ca[i * 3 + 2].toFixed(3).padStart(8);
      lines.push(
        `ATOM  ${String(serial++).padStart(5)}  CA  ${AA1_TO_3[seq[i]] ?? 'ALA'} ${chain}${String(i + 1).padStart(4)}    ${x}${y}${z}  1.00 90.00           C  `,
      );
    }
    lines.push('TER');
  };
  emit(firstCa, firstSeq, 'A');
  emit(secondCa, secondSeq, 'B');
  lines.push('END');
  return lines.join('\n');
}

function makeDummyStruct(id: string, sequence: string, sourceModelId = 'pdb-input'): StructureArtifact {
  const ca = makeHelix(sequence.length, 0, 0, 0);
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId,
    length: sequence.length,
    sequence,
    caCoords: ca,
    backbone4Coords: new Float32Array(sequence.length * 12),
    pdbText: 'ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n',
    geometry: computeGeometryStats(ca),
  };
}

describe('Overnight Pass A — Issue #83 (LASErMPNN 100% self-recovery)', () => {
  it('findContextStructure({ excludeSelf: true }) skips LASErMPNN own packed output and returns upstream WT structure', () => {
    const wtStruct = makeDummyStruct('pdb-wt-1', 'MKTAYIAKQRQISFVKSHFSRQ', 'pdb-input');
    const packedStruct: StructureArtifact = {
      ...makeDummyStruct('lasermpnn-packed-1', 'GGLAYIAKRRQLSFVKSHLAAQ', 'lasermpnn'),
      parentStructureId: wtStruct.id,
    };
    const graph: WorkflowGraph = {
      id: 'laser-graph',
      name: 'LASErMPNN Graph',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: 'WT PDB', params: {} },
        { id: 'laser', adapterId: 'lasermpnn', label: 'LASErMPNN', params: {} },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'laser', toPort: 'structure' }],
    };
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      pdb: { structure: wtStruct },
      laser: {
        structure: packedStruct,
        sequences: {
          kind: 'SequenceSet',
          id: 'laser-seq-1',
          parentStructureId: wtStruct.id,
          sourceStructureId: wtStruct.id,
          sourceModelId: 'lasermpnn',
          candidates: [
            {
              id: 'lasermpnn-1',
              sequence: packedStruct.sequence,
              score: 1.1,
              globalScore: 1.2,
              sourceModelId: 'lasermpnn',
              sourceStructureId: wtStruct.id,
            },
          ],
        },
      },
    };

    // Default behavior returns self output when present
    expect(findContextStructure(graph, 'laser', outputsByNode)).toBe(packedStruct);
    // excludeSelf: true skips self output and resolves upstream WT structure
    expect(findContextStructure(graph, 'laser', outputsByNode, { excludeSelf: true })).toBe(wtStruct);
    // Node without upstream ancestors returns null under excludeSelf: true
    expect(findContextStructure(graph, 'pdb', outputsByNode, { excludeSelf: true })).toBeNull();
  });

  it('ArtifactPeekPanel guards SequenceSet recovery against self-packed output structures', () => {
    const src = readFileSync(resolve(process.cwd(), 'src/lib/ui/ArtifactPeekPanel.svelte'), 'utf8');
    expect(src).toContain('upstreamStructure?: StructureArtifact | null');
    expect(src).toContain('selfOutputStructIds');
    expect(src).toContain('recoveryRefStructure');
    expect(src).toContain('isSelfPackedContext');
  });
});

describe('Overnight Pass A — Issue #84 (3D Compare heterodimer target-chain Kabsch)', () => {
  it('aligns opposite-order equal-length heterodimers (RFD4 A=target,B=binder vs BoltzGen/Genie3 A=binder,B=target) on matching target chain', () => {
    // Realistic 5O45-like dimensions: 116-aa target (PD-L1) + 70-aa de novo binders = 186 aa total on BOTH branches
    const nTarget = 116;
    const nBinder = 70;
    const targetSeq = 'FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPY';
    expect(targetSeq).toHaveLength(nTarget);

    // Two completely different de novo binders (helix bundle vs beta hairpin, totally different sequences & poses)
    const binder1Seq = 'LEEAARLLKKAEELAKRLEEAARLLKKAEELAKRLEEAARLLKKAEELAKRLEEAARLLKKAEELAKRLE';
    const binder2Seq = 'GSVTVTIKDGQTVTIKDGSVTVTIKDGQTVTIKDGSVTVTIKDGQTVTIKDGSVTVTIKDGQTVTIKDGS';
    expect(binder1Seq).toHaveLength(nBinder);
    expect(binder2Seq).toHaveLength(nBinder);

    const targetCa = makeHelix(nTarget, 0, 0, 0);
    const binder1Ca = makeHelix(nBinder, 25, 10, 5, 3.1, 1.4);
    const binder2Ca = makeBetaHairpin(nBinder, -30, -18, 40);

    // RFD4: Chain A = target (0..115), Chain B = binder1 (116..185)
    const rfd4Pdb = buildTwoChainPdb(targetCa, targetSeq, binder1Ca, binder1Seq);
    const rfd4Parsed = parsePdbBackbone(rfd4Pdb);

    // BoltzGen / Genie 3: Chain A = binder2 (0..69), Chain B = target (70..185), rigidly rotated + translated
    const rawBgPdb = buildTwoChainPdb(binder2Ca, binder2Seq, targetCa, targetSeq);
    const R90 = new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 1]);
    const tShift = new Float64Array([42, -28, 19]);
    const bgPdb = applyTransformToPdb(rawBgPdb, R90, tShift);
    const bgParsed = parsePdbBackbone(bgPdb);

    const graph: WorkflowGraph = {
      id: 'pdl1-cmp',
      name: '5O45 Compare',
      nodes: [
        { id: 'rfd4', adapterId: 'rfd4-proteina', label: 'RFD4', params: {} },
        { id: 'bg', adapterId: 'boltzgen', label: 'BoltzGen-1', params: {} },
      ],
      edges: [],
    };
    const makeStruct = (
      id: string,
      parsed: ReturnType<typeof parsePdbBackbone>,
      pdbText: string,
      chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>,
    ): StructureArtifact => ({
      kind: 'StructureArtifact',
      id,
      label: id,
      sourceModelId: id,
      length: parsed.nResidues,
      sequence: parsed.sequence,
      caCoords: parsed.caCoords,
      backbone4Coords: parsed.backbone4Coords,
      pdbText,
      chainIds: parsed.chainIds,
      chainLengths: parsed.diagnostics.chainLengths,
      chainRoles,
      geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums),
    });

    // 1. With explicit chainRoles: must align on the 116-aa target chain, NOT flat 0..185!
    const cmp = buildComparisonState(
      graph,
      ['rfd4', 'bg'],
      {
        rfd4: { structure: makeStruct('rfd4', rfd4Parsed, rfd4Pdb, { A: 'target', B: 'binder' }) },
        bg: { structure: makeStruct('bg', bgParsed, bgPdb, { A: 'binder', B: 'target' }) },
      },
      { layout: 'overlay', colorMode: 'by-model', alignKabsch: true },
    );

    expect(cmp.pairwise).not.toBeNull();
    expect(cmp.pairwise!.alignmentLabel).toBe('Target-chain Kabsch');
    expect(cmp.pairwise!.alignedResidues).toBe(nTarget);
    expect(cmp.pairwise!.isSubstructure).toBe(true);
    expect(cmp.pairwise!.rmsd).toBeLessThan(0.01);
    expect(cmp.pairwise!.tmScore).toBeGreaterThan(0.999);

    // Verify BoltzGen's target chain B (residues 70..185) is superimposed onto RFD4's target chain A (0..115)
    const alignedBg = parsePdbBackbone(cmp.models[1].pdbText);
    for (let i = 0; i < nTarget; i++) {
      const mIdx = nBinder + i;
      expect(Math.abs(alignedBg.caCoords[mIdx * 3 + 0] - targetCa[i * 3 + 0])).toBeLessThan(0.02);
      expect(Math.abs(alignedBg.caCoords[mIdx * 3 + 1] - targetCa[i * 3 + 1])).toBeLessThan(0.02);
      expect(Math.abs(alignedBg.caCoords[mIdx * 3 + 2] - targetCa[i * 3 + 2])).toBeLessThan(0.02);
    }

    // 2. Even WITHOUT chainRoles, sequence-identity fallback detects PD-L1 target chain at different offset
    const cmpNoRoles = buildComparisonState(
      graph,
      ['rfd4', 'bg'],
      {
        rfd4: { structure: makeStruct('rfd4', rfd4Parsed, rfd4Pdb, undefined) },
        bg: { structure: makeStruct('bg', bgParsed, bgPdb, undefined) },
      },
      { layout: 'overlay', colorMode: 'by-model', alignKabsch: true },
    );
    expect(cmpNoRoles.pairwise!.alignmentLabel).toBe('Target-chain Kabsch');
    expect(cmpNoRoles.pairwise!.alignedResidues).toBe(nTarget);
    expect(cmpNoRoles.pairwise!.rmsd).toBeLessThan(0.01);

    // 3. Same-order complexes keep existing 'Inter-model Kabsch' behavior
    const rfd4Shifted = applyTransformToPdb(rfd4Pdb, R90, tShift);
    const rfd4ShiftedParsed = parsePdbBackbone(rfd4Shifted);
    const cmpSameOrder = buildComparisonState(
      graph,
      ['rfd4', 'bg'],
      {
        rfd4: { structure: makeStruct('rfd4-a', rfd4Parsed, rfd4Pdb, { A: 'target', B: 'binder' }) },
        bg: { structure: makeStruct('rfd4-b', rfd4ShiftedParsed, rfd4Shifted, { A: 'target', B: 'binder' }) },
      },
      { layout: 'overlay', colorMode: 'by-model', alignKabsch: true },
    );
    expect(cmpSameOrder.pairwise!.alignmentLabel).toBe('Inter-model Kabsch');
    expect(cmpSameOrder.pairwise!.alignedResidues).toBe(nTarget + nBinder);
    expect(cmpSameOrder.pairwise!.rmsd).toBeLessThan(0.01);
  });
});

describe('Overnight Pass A — Issue #85 (EvoDiff designKey & EF2/Scorer Campaign CSV consistency)', () => {
  it('evodiff.worker.ts shares one designKey across candidates so LocalFold strictly honors maxSequences=1', () => {
    const evoSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/evodiff/evodiff.worker.ts'), 'utf8');
    expect(evoSrc).toContain("const designKey = 'evodiff';");
    expect(evoSrc).not.toContain('const designKey = `evodiff-${s + 1}`');

    const k0 = makeSequenceKey('evodiff-oa-dm-38m', 'evodiff', 0);
    const k1 = makeSequenceKey('evodiff-oa-dm-38m', 'evodiff', 1);
    expect(k0).not.toBe(k1);

    const evoSeqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'evodiff-seqs-test',
      parentStructureId: 'evodiff-de-novo',
      sourceModelId: 'evodiff',
      collectionKind: 'multi-sample',
      candidates: [
        {
          id: 'evodiff-s1-abc',
          key: k0,
          sequenceKey: k0,
          sequence: 'MKTAYIAKQRQISFVKSHFS',
          score: null,
          globalScore: null,
          sourceModelId: 'evodiff',
          designKey: 'evodiff',
          designIndex: 0,
          sampleIndex: 0,
          seed: 42,
        },
        {
          id: 'evodiff-s2-abc',
          key: k1,
          sequenceKey: k1,
          sequence: 'GSVTVTIKDGQTVTIKDGSV',
          score: null,
          globalScore: null,
          sourceModelId: 'evodiff',
          designKey: 'evodiff',
          designIndex: 0,
          sampleIndex: 1,
          seed: 43,
        },
      ],
    };

    const work = selectFoldWorkItems(evoSeqSet, { maxSequences: 1, maxTotalFolds: 16 });
    expect(work.selected).toHaveLength(1);
    expect(work.selected[0].id).toBe('evodiff-s1-abc');
    expect(work.excludedCandidates).toHaveLength(1);
    expect(work.excludedCandidates[0].id).toBe('evodiff-s2-abc');
  });

  it('EF2 FoldedStructure and Scorer DesignabilityMetrics Campaign CSV omit phantom fold:...:s0 rows and preserve identical keys', () => {
    const fam = 'test-campaign-fam';
    const d0 = makeDesignKey(fam, 0);
    const d1 = makeDesignKey(fam, 1);
    const s0 = makeSequenceKey(fam, d0, 0);
    const s1 = makeSequenceKey(fam, d1, 0);
    const sExcl = makeSequenceKey(fam, d1, 1);
    const f0 = makeFoldKey(fam, s0, 0, 'ef2-fast-300m');
    const f1 = makeFoldKey(fam, s1, 0, 'ef2-fast-300m');

    const ca = makeHelix(12, 0, 0, 0);
    const geom = computeGeometryStats(ca);
    const dummyPdb = 'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 88.00           C\nEND\n';

    const folded: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'folds-ef2-fast-300m-test',
      parentStructureId: 'bb-0',
      sourceStructureId: 'bb-0',
      collectionKind: 'independent-candidates',
      folds: [
        {
          sequenceId: 'mpnn-d1-s1',
          key: s0,
          designKey: d0,
          designIndex: 0,
          sequenceKey: s0,
          foldKey: f0,
          seed: 1001,
          sourceStructureId: 'bb-0',
          sampleIndex: 0,
          sequence: 'ACDEFGHIKLMN',
          foldModelId: 'localfold-ef2-fast-300m-int5',
          pdbText: dummyPdb,
          caCoords: ca,
          plddt: new Float32Array(12).fill(89.5),
          meanPlddt: 89.5,
          ptm: 0.85,
          elapsedMs: 320,
        },
        {
          sequenceId: 'mpnn-d2-s1',
          key: s1,
          designKey: d1,
          designIndex: 1,
          sequenceKey: s1,
          foldKey: f1,
          seed: 1002,
          sourceStructureId: 'bb-1',
          sampleIndex: 0,
          sequence: 'NPQRSTVWYACD',
          foldModelId: 'localfold-ef2-fast-300m-int5',
          pdbText: dummyPdb,
          caCoords: ca,
          plddt: new Float32Array(12).fill(82.1),
          meanPlddt: 82.1,
          ptm: 0.79,
          elapsedMs: 310,
        },
      ],
      excludedCandidates: [
        { key: sExcl, id: 'mpnn-d2-s2', index: 2, reason: 'Excluded by perDesignTopK=1' },
      ],
      // Mirrors localfold.worker.ts: itemOutcomes carries completed foldKeys (`fold:...:s0`) + excluded keys
      itemOutcomes: [
        { key: f0, status: 'completed' },
        { key: f1, status: 'completed' },
        { key: sExcl, status: 'excluded', reason: 'Excluded by perDesignTopK=1' },
      ],
    };

    const metrics: DesignabilityMetrics = {
      kind: 'DesignabilityMetrics',
      id: 'metrics-ef2-test',
      referenceStructureId: 'bb-0',
      passRate: 1,
      best: null,
      rows: [
        {
          structureId: 'bb-0',
          sourceStructureId: 'bb-0',
          sequenceId: 'mpnn-d1-s1',
          candidateKey: s0,
          designKey: d0,
          designIndex: 0,
          sequenceKey: s0,
          foldKey: f0,
          sequence: 'ACDEFGHIKLMN',
          scRmsd: 0.88,
          scTm: 0.94,
          meanPlddt: 89.5,
          confidenceAvailable: true,
          ptm: 0.85,
          designable: true,
          referenceMode: 'self-consistency',
          alignedFoldPdb: dummyPdb,
          designedPdb: dummyPdb,
          plddt: new Float32Array(12).fill(89.5),
          geometry: geom,
          foldElapsedMs: 320,
        },
        {
          structureId: 'bb-1',
          sourceStructureId: 'bb-1',
          sequenceId: 'mpnn-d2-s1',
          candidateKey: s1,
          designKey: d1,
          designIndex: 1,
          sequenceKey: s1,
          foldKey: f1,
          sequence: 'NPQRSTVWYACD',
          scRmsd: 1.42,
          scTm: 0.87,
          meanPlddt: 82.1,
          confidenceAvailable: true,
          ptm: 0.79,
          designable: true,
          referenceMode: 'self-consistency',
          alignedFoldPdb: dummyPdb,
          designedPdb: dummyPdb,
          plddt: new Float32Array(12).fill(82.1),
          geometry: geom,
          foldElapsedMs: 310,
        },
      ],
    };

    const foldCsv = exportCollectionCsv(normalizeToCollection(folded), { snapshotId: folded.id });
    const scoreCsv = exportCollectionCsv(normalizeToCollection(metrics), { snapshotId: metrics.id });

    const foldLines = foldCsv.trim().split('\n');
    const scoreLines = scoreCsv.trim().split('\n');

    // Header + 2 completed folds + 1 genuine exclusion = 4 lines (ZERO trailing blank phantom fold:...:s0 rows!)
    expect(foldLines).toHaveLength(4);
    expect(scoreLines).toHaveLength(3);

    const parseRow = (line: string) => line.split(',');
    const fRow0 = parseRow(foldLines[1]);
    const fRow1 = parseRow(foldLines[2]);
    const fExcl = parseRow(foldLines[3]);
    const sRow0 = parseRow(scoreLines[1]);
    const sRow1 = parseRow(scoreLines[2]);

    // Columns: 0=snapshot_id, 1=design_key, 2=sequence_key, 3=fold_key, 4=reference_structure_id, 7=status
    expect([fRow0[1], fRow0[2], fRow0[3]]).toEqual([d0, s0, f0]);
    expect([fRow1[1], fRow1[2], fRow1[3]]).toEqual([d1, s1, f1]);
    expect([sRow0[1], sRow0[2], sRow0[3]]).toEqual([d0, s0, f0]);
    expect([sRow1[1], sRow1[2], sRow1[3]]).toEqual([d1, s1, f1]);
    // Exact parity between EF2 and Scorer CSV identity columns
    expect(fRow0.slice(1, 4)).toEqual(sRow0.slice(1, 4));
    expect(fRow1.slice(1, 4)).toEqual(sRow1.slice(1, 4));
    // Genuine exclusion row preserved with clean sequence_key & reason
    expect(fExcl[2]).toBe(sExcl);
    expect(fExcl[7]).toBe('excluded');
    expect(fExcl[8]).toContain('perDesignTopK=1');
  });
});

describe('Overnight Pass A — Issue #87 (TrajectoryFrames viewer, tiedPositions badge, Genie 3 pre-run lock, RFD4 author motif badge)', () => {
  it('synthesizes clean multi-MODEL PDB & residue count for RFdiffusion caCoords-only TrajectoryFrames', () => {
    const parentStruct = makeDummyStruct('rfd-sym-1', 'MKTAYIAK', 'rfdiffusion');
    const traj: TrajectoryFramesArtifact = {
      kind: 'TrajectoryFrames',
      id: 'rfd-traj-abc',
      label: 'RFdiffusion trajectory (3 steps)',
      parentStructureId: parentStruct.id,
      sourceModelId: 'rfdiffusion',
      frameCount: 3,
      ordering: 'ordered-frames',
      collectionKind: 'ordered-frames',
      frames: [
        { key: 'frame-0', frameIndex: 0, caCoords: makeHelix(8, 0, 0, 0), timeStep: 50 },
        { key: 'frame-1', frameIndex: 1, caCoords: makeHelix(8, 1, 0, 0), timeStep: 25 },
        { key: 'frame-2', frameIndex: 2, caCoords: makeHelix(8, 2, 0, 0), timeStep: 1 },
      ],
    };

    expect(inferTrajectoryResidueCount(traj, parentStruct)).toBe(8);
    const pdb = buildTrajectoryMultiModelPdb(traj, parentStruct);
    expect(pdb).not.toBeNull();
    expect(pdb!).toContain('MODEL        1');
    expect(pdb!).toContain('MODEL        2');
    expect(pdb!).toContain('MODEL        3');
    expect(pdb!).toContain('ENDMDL');
    expect(pdb!).toContain('MET A   1');
    expect(pdb!.trim().endsWith('END')).toBe(true);

    // ArtifactPeekPanel must render TrajectoryFrames cleanly with download button
    const peekSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/ArtifactPeekPanel.svelte'), 'utf8');
    expect(peekSrc).toContain("artifact?.kind === 'TrajectoryFrames'");
    expect(peekSrc).toContain('{:else if trajArt}');
    expect(peekSrc).toContain('↓ Trajectory PDB');
  });

  it('computeUpstreamLockSummary surfaces tiedPositions (post-run & pre-run symmetry) and predicts Genie 3 Lock B · Design A even with lockChain=A', () => {
    // 1. Pre-run RFdiffusion symmetry (C3) -> ProteinMPNN
    const symGraph: WorkflowGraph = {
      id: 'rfd-sym-c3',
      name: 'RFdiffusion C3 Symmetry',
      nodes: [
        { id: 'rfd', adapterId: 'rfdiffusion', label: 'RFdiffusion', params: { taskMode: 'symmetry', symmetry: 'c3', length: 72 } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN', params: {} },
      ],
      edges: [{ fromNode: 'rfd', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' }],
    };

    const preSym = computeUpstreamLockSummary(symGraph, 'mpnn', {});
    expect(preSym).not.toBeNull();
    expect(preSym!.status).toBe('active');
    expect(preSym!.symmetryLabel).toBe('C3');
    expect(preSym!.tiedPositionGroupCount).toBe(24);
    expect(preSym!.nodeBadge).toContain('C3 tied');

    // 2. Post-run materialized structure carrying tiedPositions
    const tiedStr = Array.from({ length: 24 }, (_, i) => `A${i + 1}=B${i + 25}=C${i + 49}`).join('; ');
    const symStruct: StructureArtifact = {
      ...makeDummyStruct('rfd-c3-out', 'A'.repeat(72), 'rfdiffusion'),
      label: 'RFdiffusion (C3) (72 aa)',
      chainIds: [...Array(24).fill('A'), ...Array(24).fill('B'), ...Array(24).fill('C')],
      tiedPositions: tiedStr,
      chainRoles: { A: 'scaffold', B: 'scaffold', C: 'scaffold' },
    };
    const postSym = computeUpstreamLockSummary(symGraph, 'mpnn', { rfd: { structure: symStruct } });
    expect(postSym).not.toBeNull();
    expect(postSym!.symmetryLabel).toBe('C3');
    expect(postSym!.tiedPositionGroupCount).toBe(24);
    expect(postSym!.nodeBadge).toContain('C3 tied');
    expect(postSym!.detailText).toContain('24 tied residue groups');

    // 3. Pre-run Genie 3 binder with explicit lockChain='A' (like pdl1_5o45) MUST still predict Lock B · Design A!
    const geniePdl1Graph: WorkflowGraph = {
      id: 'genie-pdl1',
      name: 'Genie 3 PD-L1',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: '5O45', params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' } },
        { id: 'g3', adapterId: 'genie3', label: 'Genie 3', params: { taskMode: 'binder', lockChain: 'A', length: 70 } },
        { id: 'campnn', adapterId: 'web-protein-mpnn', label: 'CaMPNN (G3)', params: { family: 'ca' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'template' },
        { fromNode: 'g3', fromPort: 'structure', toNode: 'campnn', toPort: 'structure' },
      ],
    };
    const g3Pre = computeUpstreamLockSummary(geniePdl1Graph, 'campnn', {});
    expect(g3Pre).not.toBeNull();
    expect(g3Pre!.lockedChains).toEqual(['B']);
    expect(g3Pre!.designingChains).toEqual(['A']);
    expect(g3Pre!.chainRoles).toEqual({ A: 'binder', B: 'target' });
    expect(g3Pre!.nodeBadge).toContain('Lock B · Design A');
  });

  it('formatRfd4AuthorRunLabel formats RFD4 motif badges using author PDB numbering for negative, offset, and insertion-code residues', () => {
    // Build a PDB with negative author residue numbers (-6..9, so 1-based label_seq_id 5..12 == author -2..5)
    const pdbLines: string[] = [];
    for (let i = 0; i < 16; i++) {
      const authRes = i - 6;
      const x = (i * 3.8).toFixed(3).padStart(8);
      pdbLines.push(
        `ATOM  ${String(i + 1).padStart(5)}  CA  ALA A${String(authRes).padStart(4)}    ${x}   0.000   0.000  1.00 90.00           C  `,
      );
    }
    pdbLines.push('END');
    const negPdb = pdbLines.join('\n');
    const parsedNeg = parsePdbBackbone(negPdb);
    const tplCtx = extractRfd4TemplateContext(negPdb);

    const tplStruct: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'neg-tpl',
      label: 'Negative ResNum PDB',
      sourceModelId: 'pdb-input',
      length: parsedNeg.nResidues,
      sequence: parsedNeg.sequence,
      caCoords: parsedNeg.caCoords,
      backbone4Coords: parsedNeg.backbone4Coords,
      pdbText: negPdb,
      chainIds: parsedNeg.chainIds,
      sourceResidues: parsedNeg.sourceResidues,
      geometry: computeGeometryStats(parsedNeg.caCoords),
    };

    const resolved = resolveRfd4CompiledInput(
      { mode: 'motif', length: 60, motifSpan: 'A-2-5', nsteps: 60 },
      tplStruct,
      undefined,
      tplCtx,
    );
    // Internal compiled contig uses label_seq_id A5-12 for RFD4 engine
    expect(resolved.contig).toContain('A5-12');
    expect(resolved.runLabel).toContain('motif:A5-12');

    // Author-numbered runLabel restores motif:A-2-5 and contig A-2-5!
    const formatted = formatRfd4AuthorRunLabel(resolved.runLabel, {
      motifSpans: resolved.intent?.motifSpans,
      templateContext: tplCtx,
      sourceResidues: tplStruct.sourceResidues,
    });
    expect(formatted).toContain('motif:A-2-5');
    expect(formatted).toContain(',A-2-5,');
    expect(formatted).not.toContain('motif:A5-12');

    // Also works via sourceResidues alone (without templateContext), including insertion codes
    const insResidues: SourceResidueId[] = [
      { index: 0, authChainId: 'H', resNum: 99, insCode: '', resName: 'ALA', projectedChainId: 'H' },
      { index: 1, authChainId: 'H', resNum: 100, insCode: 'A', resName: 'GLY', projectedChainId: 'H' },
      { index: 2, authChainId: 'H', resNum: 100, insCode: 'B', resName: 'SER', projectedChainId: 'H' },
      { index: 3, authChainId: 'H', resNum: 100, insCode: 'C', resName: 'TYR', projectedChainId: 'H' },
      { index: 4, authChainId: 'H', resNum: 101, insCode: '', resName: 'ASP', projectedChainId: 'H' },
    ];
    const insFormatted = formatRfd4AuthorRunLabel('RFD4 · 50-50,H2-4 · unidx-motif:H2-4', {
      sourceResidues: insResidues,
    });
    expect(insFormatted).toBe('RFD4 · 50-50,H100A-100C · unidx-motif:H100A-100C');
  });
});
