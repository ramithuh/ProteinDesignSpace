import { describe, expect, it } from 'vitest';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { autoLayoutGraph, estimateNodeCardHeight } from '$lib/engine/graph-utils';
import { clearInputPreviewCache, compileNodeInputPreview } from '$lib/engine/input-preview';
import {
  compileWorkflowPlanToGraph,
  reconcilePlannedWorkflow,
  type BlockGraphWorkflowPlan,
} from '$lib/engine/workflow-planner';
import type { SequenceSet, StructureArtifact, WorkflowGraph } from '$lib/engine/types';
import { generateSampleHelixBundlePdb, parseStructureText } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';

function buildSyntheticPdbWithHetatms(nResidues: number, hetatms: Array<{ resName: string; atomName: string; chain: string; resSeq: number }>): string {
  const lines: string[] = [];
  let serial = 1;
  for (let i = 1; i <= nResidues; i++) {
    const x = (i * 3.8).toFixed(3).padStart(8);
    const y = '   0.000';
    const z = '   0.000';
    const seq = String(i).padStart(4);
    lines.push(`ATOM  ${String(serial++).padStart(5)}  N   ALA A${seq}    ${x}${y}${z}  1.00 20.00           N  `);
    lines.push(`ATOM  ${String(serial++).padStart(5)}  CA  ALA A${seq}    ${x}${y}${z}  1.00 20.00           C  `);
    lines.push(`ATOM  ${String(serial++).padStart(5)}  C   ALA A${seq}    ${x}${y}${z}  1.00 20.00           C  `);
    lines.push(`ATOM  ${String(serial++).padStart(5)}  O   ALA A${seq}    ${x}${y}${z}  1.00 20.00           O  `);
  }
  for (const h of hetatms) {
    lines.push(
      `HETATM${String(serial++).padStart(5)} ${h.atomName.padEnd(4)} ${h.resName.padStart(3)} ${h.chain}${String(h.resSeq).padStart(4)}      10.000  10.000  10.000  1.00 20.00           C  `,
    );
  }
  lines.push('END');
  return lines.join('\n');
}

describe('Overnight Pass B — #63 Generic Wired-Template Unconditional Guard & AI Describe Repairs', () => {
  it('warns with 1-click fix when any multi-mode template generator has a wired structure in unconditional mode', () => {
    const cases: Array<{
      adapterId: string;
      modeKey: 'mode' | 'taskMode';
      expectedCode: string;
      expectedTargetModes: string[];
    }> = [
      { adapterId: 'rfd4-proteina', modeKey: 'mode', expectedCode: 'rfd4-unused-template', expectedTargetModes: ['partial'] },
      { adapterId: 'genie3', modeKey: 'taskMode', expectedCode: 'genie3-unused-template', expectedTargetModes: ['binder', 'motif'] },
      { adapterId: 'protpardelle', modeKey: 'taskMode', expectedCode: 'protpardelle-unused-template', expectedTargetModes: ['partial', 'motif'] },
      { adapterId: 'boltzgen', modeKey: 'taskMode', expectedCode: 'boltzgen-unused-template', expectedTargetModes: ['motif', 'binder'] },
      { adapterId: 'rfdiffusion', modeKey: 'taskMode', expectedCode: 'rfdiffusion-unused-template', expectedTargetModes: ['motif'] },
    ];

    for (const c of cases) {
      const g: WorkflowGraph = {
        id: `uncond-${c.adapterId}`,
        name: `Uncond ${c.adapterId}`,
        nodes: [
          { id: 'pdb', adapterId: 'pdb-input', label: 'PDB Input', params: { mode: 'demo-bundle' } },
          { id: 'gen', adapterId: c.adapterId, label: c.adapterId, params: { [c.modeKey]: 'unconditional' } },
        ],
        edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'gen', toPort: 'structure' }],
      };

      const issues = validateBioCapabilities(g);
      const warn = issues.find((i) => i.code === c.expectedCode && i.nodeId === 'gen');
      expect(warn, `Expected ${c.expectedCode} for ${c.adapterId}`).toBeDefined();
      expect(warn!.severity).toBe('warning');
      expect(warn!.fix).toBeDefined();
      expect(warn!.fix!.nodeId).toBe('gen');
      expect(c.expectedTargetModes).toContain(String(warn!.fix!.paramsPatch[c.modeKey]));

      // Applying the 1-click fix clears the unused-template warning
      Object.assign(g.nodes[1].params, warn!.fix!.paramsPatch);
      const after = validateBioCapabilities(g);
      expect(after.some((i) => i.code === c.expectedCode)).toBe(false);
    }
  });

  it('does not emit rfd4-unused-template when customSpecYaml is provided', () => {
    const g: WorkflowGraph = {
      id: 'rfd4-yaml',
      name: 'RFD4 YAML',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          params: { mode: 'unconditional', customSpecYaml: 'contig: A1-20/30-30' },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' }],
    };
    expect(validateBioCapabilities(g).some((i) => i.code === 'rfd4-unused-template')).toBe(false);
  });

  it('offers Switch to Motif Scaffolding on protpardelle when motifSelection is already set', () => {
    const g: WorkflowGraph = {
      id: 'pp-motif-sel',
      name: 'PP Motif Sel',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'pp', adapterId: 'protpardelle', params: { taskMode: 'unconditional', motifSelection: 'A10-25' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'pp', toPort: 'structure' }],
    };
    const warn = validateBioCapabilities(g).find((i) => i.code === 'protpardelle-unused-template');
    expect(warn?.fix?.paramsPatch).toEqual({ taskMode: 'motif' });
  });

  it('repairs ProtPardelle taskMode to partial or motif when wired to PDB and Gemma-4 omits taskMode', () => {
    const partialPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: '1UBQ', params: { pdbId: '1UBQ' } },
        { id: 'pp', adapter: 'protpardelle', label: 'ProtPardelle-1c', params: { rewindSteps: 30 } },
      ],
      edges: [['pdb', 'pp']],
      summary: 'Refine 1UBQ with ProtPardelle partial diffusion',
    };

    const reconciledPartial = reconcilePlannedWorkflow(
      partialPlan,
      'Refine ubiquitin 1UBQ using ProtPardelle partial diffusion rewind 30 steps',
    );
    const compiledPartial = compileWorkflowPlanToGraph(reconciledPartial).graph;
    const ppPartial = compiledPartial.nodes.find((n) => n.id === 'pp')!;
    expect(ppPartial.params.taskMode).toBe('partial');

    const motifPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: '1UBQ', params: { pdbId: '1UBQ' } },
        { id: 'pp', adapter: 'protpardelle', label: 'ProtPardelle-1c', params: { length: 90 } },
      ],
      edges: [['pdb', 'pp']],
      summary: 'Scaffold motif from 1UBQ with ProtPardelle',
    };
    const reconciledMotif = reconcilePlannedWorkflow(
      motifPlan,
      'Scaffold a motif from 1UBQ with ProtPardelle',
    );
    const compiledMotif = compileWorkflowPlanToGraph(reconciledMotif).graph;
    expect(compiledMotif.nodes.find((n) => n.id === 'pp')!.params.taskMode).toBe('motif');
  });

  it('extracts explicit residue length for RFdiffusion symmetry when Gemma-4 omits length or defaults to 80', () => {
    const omittedLenPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'rfd', adapter: 'rfdiffusion', label: 'RFdiffusion', params: { taskMode: 'symmetry', symmetry: 'c3' } },
      ],
      edges: [],
      summary: 'C3 symmetric 72aa oligomer with RFdiffusion',
    };

    const rec1 = reconcilePlannedWorkflow(
      omittedLenPlan,
      'Generate a C3 symmetric 72aa oligomer with RFdiffusion',
    );
    const g1 = compileWorkflowPlanToGraph(rec1).graph;
    const rfd1 = g1.nodes.find((n) => n.id === 'rfd')!;
    expect(rfd1.params.taskMode).toBe('symmetry');
    expect(rfd1.params.symmetry).toBe('c3');
    expect(rfd1.params.length).toBe(72);
    // 72 is divisible by 3 -> no rounding warning!
    expect(validateBioCapabilities(g1).some((i) => i.code === 'rfdiffusion-symmetry-length-adjust')).toBe(false);

    // Also works when prompt says "72 residues" and Gemma-4 left default length: 80
    const default80Plan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'rfd', adapter: 'rfdiffusion', label: 'RFdiffusion', params: { taskMode: 'symmetry', symmetry: 'c3', length: 80 } },
      ],
      edges: [],
      summary: 'C3 symmetric oligomer',
    };
    const rec2 = reconcilePlannedWorkflow(
      default80Plan,
      'Generate a C3 symmetric 72 residues oligomer with RFdiffusion',
    );
    const g2 = compileWorkflowPlanToGraph(rec2).graph;
    expect(g2.nodes.find((n) => n.id === 'rfd')!.params.length).toBe(72);
  });
});

describe('Overnight Pass B — #86 Honest Node Studio Input Previews (Genie 2, ProtPardelle, EvoDiff, FoldFlow 2)', () => {
  it('Genie 2 strips crystal HETATMs (M3L, CA) and highlights only motifRange with true scaffold length', () => {
    clearInputPreviewCache();
    const pdb1prwLike = buildSyntheticPdbWithHetatms(147, [
      { resName: 'M3L', atomName: 'CA', chain: 'A', resSeq: 115 },
      { resName: 'CA', atomName: 'CA', chain: 'A', resSeq: 201 },
    ]);
    const g: WorkflowGraph = {
      id: 'g2-preview',
      name: 'Genie 2 Preview',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1PRW.cif',
          params: { mode: 'custom', filename: '1PRW.pdb', pdbText: pdb1prwLike, benchmarkId: 'rfd_1prw_efhand' },
        },
        {
          id: 'g2',
          adapterId: 'genie2',
          label: 'Genie 2',
          params: { length: 80, motifRange: 'A16-35', steps: 1000, scale: 0.4 },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g2', toPort: 'template' }],
    };

    const preview = compileNodeInputPreview(g, 'g2', {});
    expect(preview.availability).toBe('ready');
    // Crystal HETATMs M3L and CA must NEVER appear in includedLigands
    expect(preview.entities.includedLigands).toEqual([]);
    expect(preview.entities.filteredBuffers).toContain('M3L');
    expect(preview.entities.filteredBuffers).toContain('CA');
    // True scaffold length = 80, motif A16-35 = 20 residues fixed, 60 designed
    expect(preview.budget.totalResidues).toBe(80);
    expect(preview.budget.fixedResidues).toBe(20);
    expect(preview.budget.designedResidues).toBe(60);
    expect(preview.chains).toHaveLength(1);
    expect(preview.chains[0].residues.filter((r) => r.roles.includes('motif'))).toHaveLength(20);
    expect(preview.chains[0].residues.filter((r) => r.roles.includes('designable'))).toHaveLength(60);
    expect(preview.highlightLayers.some((l) => l.role === 'motif' && l.tokens.length === 20)).toBe(true);
  });

  it('ProtPardelle marks all residues designable in partial mode (with MiniMPNN note) and highlights motif in motif mode', () => {
    clearInputPreviewCache();
    const pdb76 = buildSyntheticPdbWithHetatms(76, []);
    const gPartial: WorkflowGraph = {
      id: 'pp-partial-preview',
      name: 'PP Partial',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: '1UBQ.pdb', params: { mode: 'custom', filename: '1UBQ.pdb', pdbText: pdb76 } },
        {
          id: 'pp',
          adapterId: 'protpardelle',
          label: 'ProtPardelle-1c',
          params: { taskMode: 'partial', checkpoint: 'cc91_epoch383', sidechainMode: true, numSteps: 100, rewindSteps: 40 },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'pp', toPort: 'structure' }],
    };

    const partialPrev = compileNodeInputPreview(gPartial, 'pp', {});
    expect(partialPrev.availability).toBe('ready');
    expect(partialPrev.budget.totalResidues).toBe(76);
    expect(partialPrev.budget.designedResidues).toBe(76);
    expect(partialPrev.budget.fixedResidues).toBe(0);
    expect(partialPrev.chains[0].residues.every((r) => r.roles.includes('designable'))).toBe(true);
    expect(String(partialPrev.payload.compiledSummary.miniMpnnNote ?? '')).toMatch(/MiniMPNN/i);

    // Motif mode with explicit motifSelection
    const gMotif: WorkflowGraph = {
      id: 'pp-motif-preview',
      name: 'PP Motif',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: '1UBQ.pdb', params: { mode: 'custom', filename: '1UBQ.pdb', pdbText: pdb76 } },
        {
          id: 'pp',
          adapterId: 'protpardelle',
          label: 'ProtPardelle-1c',
          params: { taskMode: 'motif', length: 80, motifSelection: 'A16-35', sidechainMode: true },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'pp', toPort: 'structure' }],
    };
    const motifPrev = compileNodeInputPreview(gMotif, 'pp', {});
    expect(motifPrev.availability).toBe('ready');
    expect(motifPrev.budget.totalResidues).toBe(80);
    expect(motifPrev.budget.fixedResidues).toBe(20);
    expect(motifPrev.budget.designedResidues).toBe(60);
    expect(motifPrev.chains[0].residues.filter((r) => r.roles.includes('motif'))).toHaveLength(20);
    expect(motifPrev.highlightLayers.some((l) => l.role === 'motif' && l.tokens.length === 20)).toBe(true);
  });

  it('EvoDiff infill populates chains/fastaEntries from Sequence Input or PDB Input and marks maskPositions designable', () => {
    clearInputPreviewCache();
    // 1. From Sequence Input
    const seq40 = 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY';
    const gSeq: WorkflowGraph = {
      id: 'evo-seq',
      name: 'EvoDiff Seq',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', label: 'Seq Input', params: { sequence: `>wt40\n${seq40}` } },
        { id: 'evo', adapterId: 'evodiff', label: 'EvoDiff-38M', params: { taskMode: 'infill', maskPositions: '15-30' } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'evo', toPort: 'sequences' }],
    };

    const prevSeq = compileNodeInputPreview(gSeq, 'evo', {});
    expect(prevSeq.availability).toBe('ready');
    expect(prevSeq.budget.totalResidues).toBe(40);
    expect(prevSeq.budget.designedResidues).toBe(16); // 15..30 inclusive
    expect(prevSeq.budget.fixedResidues).toBe(24);
    expect(prevSeq.chains).toHaveLength(1);
    expect(prevSeq.chains[0].residues.filter((r) => r.roles.includes('designable'))).toHaveLength(16);
    expect(prevSeq.chains[0].residues.filter((r) => r.roles.includes('fixed-sequence'))).toHaveLength(24);
    expect(prevSeq.fastaEntries).toHaveLength(1);

    // 2. From PDB Input (live materialized sequences port without prior Run)
    const gPdb: WorkflowGraph = {
      id: 'evo-pdb',
      name: 'EvoDiff PDB',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: 'Demo Bundle', params: { mode: 'demo-bundle' } },
        { id: 'evo', adapterId: 'evodiff', label: 'EvoDiff-38M', params: { taskMode: 'infill', maskPositions: '10-20' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'sequences', toNode: 'evo', toPort: 'sequences' }],
    };
    const prevPdb = compileNodeInputPreview(gPdb, 'evo', {});
    expect(prevPdb.availability).toBe('ready');
    expect(prevPdb.budget.totalResidues).toBe(53);
    expect(prevPdb.budget.designedResidues).toBe(11);
    expect(prevPdb.budget.fixedResidues).toBe(42);
    expect(prevPdb.chains).toHaveLength(1);
    expect(prevPdb.fastaEntries).toHaveLength(1);
  });

  it('FoldFlow 2 populates chains/fastaEntries from candidate 0 and flags stripped ":" chain breaks', () => {
    clearInputPreviewCache();
    const g: WorkflowGraph = {
      id: 'ff2-preview',
      name: 'FoldFlow 2 Preview',
      nodes: [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          label: 'Dimer Seq',
          params: { sequence: '>dimer1\nACDEFGHIKLMNPQ:RSTVWYACDEFG\n>dimer2\nGGGGGGGGGG' },
        },
        { id: 'ff2', adapterId: 'foldflow2', label: 'FoldFlow 2', params: { length: 80, steps: 50 } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'ff2', toPort: 'sequences' }],
    };

    const prev = compileNodeInputPreview(g, 'ff2', {});
    expect(prev.availability).toBe('ready');
    expect(prev.budget.totalResidues).toBe(26); // 14 + 12 collapsed
    expect(prev.chains).toHaveLength(1);
    expect(prev.chains[0].sequence).toBe('ACDEFGHIKLMNPQRSTVWYACDEFG');
    expect(prev.fastaEntries).toHaveLength(2);
    expect(prev.fastaEntries[0].selected).toBe(true);
    expect(prev.fastaEntries[1].selected).toBe(false);
    expect(prev.payload.compiledSummary.strippedChainBreaks).toBe(true);
    expect(prev.payload.headline).toMatch(/chain breaks stripped/i);
    expect(prev.issues.some((i) => /chain breaks/i.test(i.message))).toBe(true);

    // Also verify with a direct pre-run SequenceSet containing raw ":" in candidate 0
    const rawSeqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'raw-seq-1',
      parentStructureId: 'none',
      sourceModelId: 'web-protein-mpnn',
      candidates: [
        {
          id: 'mpnn-1',
          key: 'mpnn-1',
          sequence: 'ACDEFGHIKL:MNPQRSTVWY',
          score: 1.2,
          globalScore: 1.2,
          sourceModelId: 'web-protein-mpnn',
          sourceStructureId: 'none',
        },
      ],
    };
    const prevRaw = compileNodeInputPreview(g, 'ff2', { seq: { sequences: rawSeqSet } });
    expect(prevRaw.budget.totalResidues).toBe(20);
    expect(prevRaw.payload.compiledSummary.strippedChainBreaks).toBe(true);
  });
});

describe('Overnight Pass B — #88 Dynamic ROW_GAP in autoLayoutGraph', () => {
  it('gives 4-port cards like ESM-C 300M larger vertical clearance than 2-port cards with zero overlap', () => {
    const esmcHeight = estimateNodeCardHeight({ id: 'ec', adapterId: 'esmc', params: {} });
    const mpnnHeight = estimateNodeCardHeight({ id: 'mp', adapterId: 'web-protein-mpnn', params: {} });
    expect(esmcHeight).toBe(132 + 4 * 26); // 236 px
    expect(mpnnHeight).toBeLessThan(esmcHeight);

    const g: WorkflowGraph = {
      id: 'esmc-stack',
      name: 'ESM-C Stacked Column',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: 'PDB', params: {} },
        { id: 'ec', adapterId: 'esmc', label: 'ESM-C 300M', params: {} },
        { id: 'thermo', adapterId: 'thermompnn', label: 'ThermoMPNN', params: {} },
        { id: 'dyna', adapterId: 'dyna1', label: 'Dyna-1', params: {} },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'ec', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'thermo', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'dyna', toPort: 'structure' },
      ],
    };

    const laid = autoLayoutGraph(g);
    const ecPos = laid.nodes.find((n) => n.id === 'ec')!.position!;
    const thermoPos = laid.nodes.find((n) => n.id === 'thermo')!.position!;
    const dynaPos = laid.nodes.find((n) => n.id === 'dyna')!.position!;

    // All three sit in column 1 (same X)
    expect(ecPos.x).toBe(thermoPos.x);
    expect(thermoPos.x).toBe(dynaPos.x);

    // Clearance below ESM-C 300M must be at least esmcHeight + 24 (260px), never the old 188px!
    expect(thermoPos.y - ecPos.y).toBeGreaterThanOrEqual(esmcHeight + 24);
    expect(thermoPos.y - ecPos.y).toBeGreaterThan(188);

    // Every adjacent pair in the column clears the upper card's estimated height
    const col1 = [
      { n: laid.nodes.find((x) => x.id === 'ec')!, pos: ecPos },
      { n: laid.nodes.find((x) => x.id === 'thermo')!, pos: thermoPos },
      { n: laid.nodes.find((x) => x.id === 'dyna')!, pos: dynaPos },
    ].sort((a, b) => a.pos.y - b.pos.y);
    for (let i = 0; i < col1.length - 1; i++) {
      expect(col1[i + 1].pos.y - col1[i].pos.y).toBeGreaterThanOrEqual(
        estimateNodeCardHeight(col1[i].n) + 24,
      );
    }
  });

  it('keeps helper parseStructureText sanity intact for demo bundle in ProtPardelle partial preview', () => {
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parseStructureText(pdb);
    const structArt: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'demo-art',
      label: 'demo_3helix_53aa.pdb',
      sourceModelId: 'pdb-input',
      length: parsed.nResidues,
      sequence: parsed.sequence,
      caCoords: parsed.caCoords,
      backbone4Coords: parsed.backbone4Coords,
      pdbText: parsed.pdbText,
      chainIds: parsed.chainIds,
      chainLengths: parsed.diagnostics.chainLengths,
      diagnostics: parsed.diagnostics,
      geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes),
    };
    const g: WorkflowGraph = {
      id: 'pp-demo',
      name: 'PP Demo',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'pp', adapterId: 'protpardelle', params: { taskMode: 'partial', sidechainMode: false } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'pp', toPort: 'structure' }],
    };
    const prev = compileNodeInputPreview(g, 'pp', { pdb: { structure: structArt } });
    expect(prev.budget.totalResidues).toBe(53);
    expect(prev.budget.designedResidues).toBe(53);
    expect(prev.budget.fixedResidues).toBe(0);
    expect(prev.payload.compiledSummary.miniMpnnNote).toBeUndefined();
  });
});
