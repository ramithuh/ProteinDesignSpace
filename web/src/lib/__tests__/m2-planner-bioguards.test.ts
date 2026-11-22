import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  PLANNER_TIMEOUT_MS,
  compileWorkflowPlanToGraph,
  reconcilePlannedWorkflow,
  repairDraftedWorkflowFromBioFeedback,
  type BlockGraphWorkflowPlan,
  type PdbWorkflowPlan,
} from '$lib/engine/workflow-planner';
import {
  hasAntibodyChainHallmarks,
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from '$lib/engine/bio-capabilities';
import {
  ExecutionScheduler,
  computeInvalidatedNodeIds,
  deriveSequenceSetFromStructure,
} from '$lib/engine/scheduler';
import {
  CRYSTALLIZATION_BUFFERS,
  compileConditionToRfd4,
  extractRfd4TemplateContext,
  isCrystallizationBuffer,
} from '$lib/engine/conditioning';
import { validateGraph } from '$lib/engine/graph-utils';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import type {
  CanonicalArtifact,
  FoldedStructure,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

function makePdb(opts: {
  nRes?: number;
  chain?: string;
  startNum?: number;
  caOnly?: boolean;
  header?: string;
  resNums?: number[];
  resNames?: string[];
  insCodes?: string[];
  hetatms?: string[];
}): string {
  const nRes = opts.nRes ?? 10;
  const chain = opts.chain ?? 'A';
  const startNum = opts.startNum ?? 1;
  const lines: string[] = [];
  if (opts.header) lines.push(opts.header);
  let serial = 1;
  for (let i = 0; i < nRes; i++) {
    const num = opts.resNums?.[i] ?? startNum + i;
    const rn = String(num).padStart(4);
    const ins = opts.insCodes?.[i] ? opts.insCodes[i] : ' ';
    const resName = (opts.resNames?.[i] ?? 'ALA').padEnd(3).slice(0, 3);
    const x = (i * 3.8).toFixed(3).padStart(8);
    if (!opts.caOnly) {
      lines.push(
        `ATOM  ${String(serial++).padStart(5)}  N   ${resName} ${chain}${rn}${ins}   ${x}   0.000   0.000  1.00 20.00           N  `,
      );
    }
    lines.push(
      `ATOM  ${String(serial++).padStart(5)}  CA  ${resName} ${chain}${rn}${ins}   ${x}   1.000   0.000  1.00 20.00           C  `,
    );
    if (!opts.caOnly) {
      lines.push(
        `ATOM  ${String(serial++).padStart(5)}  C   ${resName} ${chain}${rn}${ins}   ${x}   2.000   0.000  1.00 20.00           C  `,
      );
      lines.push(
        `ATOM  ${String(serial++).padStart(5)}  O   ${resName} ${chain}${rn}${ins}   ${x}   2.000   1.000  1.00 20.00           O  `,
      );
    }
  }
  for (const h of opts.hetatms ?? []) lines.push(h);
  lines.push('END');
  return lines.join('\n');
}

function makeStructArtifact(opts: {
  id: string;
  label?: string;
  length?: number;
  sequence?: string;
  caOnly?: boolean;
  pdbText?: string;
  cifText?: string;
  chainIds?: string[];
  diagnostics?: StructureArtifact['diagnostics'];
}): StructureArtifact {
  const L = opts.length ?? opts.sequence?.length ?? 12;
  const seq = opts.sequence ?? 'A'.repeat(L);
  return {
    kind: 'StructureArtifact',
    id: opts.id,
    label: opts.label ?? opts.id,
    sourceModelId: opts.caOnly ? 'genie3' : 'pdb-input',
    length: L,
    sequence: seq,
    caCoords: new Float32Array(L * 3),
    backbone4Coords: new Float32Array(L * 4 * 3),
    pdbText: opts.pdbText ?? makePdb({ nRes: L, caOnly: opts.caOnly }),
    ...(opts.cifText ? { cifText: opts.cifText } : {}),
    ...(opts.chainIds ? { chainIds: opts.chainIds } : {}),
    ...(opts.caOnly ? { atomGranularity: 'ca-only' as const } : { atomGranularity: 'backbone4' as const }),
    ...(opts.diagnostics ? { diagnostics: opts.diagnostics } : {}),
    geometry: computeGeometryStats(new Float32Array(L * 3)),
  };
}

class StubWorker {
  private listeners = new Map<string, Set<(ev: MessageEvent) => void>>();
  addEventListener(type: string, cb: (ev: MessageEvent) => void) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(cb);
  }
  removeEventListener(type: string, cb: (ev: MessageEvent) => void) {
    this.listeners.get(type)?.delete(cb);
  }
  postMessage(msg: { id: number; type: string }) {
    if (msg.type === 'dispose') {
      this.emit('message', { data: { id: msg.id, type: 'disposed' } } as MessageEvent);
      return;
    }
    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'stub-seq',
      parentStructureId: 'stub',
      sourceModelId: 'stub',
      candidates: [
        {
          id: 'cand-1',
          sequence: 'ACDEFGHIKL',
          score: 1.0,
          globalScore: 1.0,
          sourceModelId: 'stub',
        },
      ],
    };
    const folded: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'stub-fold',
      parentStructureId: 'stub',
      folds: [],
    };
    this.emit('message', {
      data: {
        id: msg.id,
        type: 'result',
        outputs: {
          sequences: seqSet,
          folds: folded,
          structure: makeStructArtifact({ id: 'stub-gen', length: 10, caOnly: true }),
        },
        elapsedMs: 1,
      },
    } as MessageEvent);
  }
  terminate() {}
  private emit(type: string, ev: MessageEvent) {
    for (const cb of this.listeners.get(type) ?? []) cb(ev);
  }
}

describe('M2: Planner validation, bio-capabilities & scheduler integration', () => {
  beforeEach(() => {
    vi.stubGlobal('Worker', StubWorker);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('Follow-up: PLANNER_TIMEOUT_MS is bumped to 30_000 ms', () => {
    expect(PLANNER_TIMEOUT_MS).toBe(30_000);
  });

  it('§2.3 (#62): filters supported blocks from plan.unsupported while preserving truly unsupported/versioned models', () => {
    const plan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'evo1', adapter: 'evodiff', label: 'EvoDiff', params: { taskMode: 'unconditional', length: 64 } },
      ],
      edges: [],
      unsupported: [
        'EvoDiff',
        'evodiff',
        'La-Proteina',
        'SiD-Protein',
        'FrameFlow',
        'FoldFlow 2',
        'BoltzGen-1',
        'Chroma',
        'Chai-1',
        'RFdiffusion1',
        'RFdiffusion2',
        'RFdiffusion3',
        'ESM3',
      ],
      summary: 'Test unsupported filtering',
    };
    const compiled = compileWorkflowPlanToGraph(plan);
    expect(compiled.unsupported).toEqual([
      'Chroma',
      'Chai-1',
      'RFdiffusion1',
      'RFdiffusion2',
      'RFdiffusion3',
      'ESM3',
    ]);
  });

  it('§2.3 (#62): per-adapter sanitizeBlockNodeParams strips leaked keys, maps aliases, normalizes EvoDiff #MASK#, and preserves UI tracking on edit', () => {
    const plan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb6', adapter: 'pdb-input', label: 'PDB Input', params: { pdbId: '6XEY' } },
        {
          id: 'rfd1',
          adapter: 'rfd4-proteina',
          label: 'RFD4-Proteina',
          params: { mode: 'unconditional', length: 70, numSequences: 99, topK: 5, autoSeededBenchmarkId: 'injected' },
        },
        {
          id: 'pp1',
          adapter: 'protpardelle',
          label: 'ProtPardelle-1c',
          params: { mode: 'motif', length: 80, topK: 3, motifSelection: 'A1-10' },
        },
        {
          id: 'af1',
          adapter: 'antifold-esmif1',
          label: 'AntiFold',
          params: { model: 'antifold', maxSequences: 6, topK: 2, fixedPositions: '' },
        },
        {
          id: 'evo1',
          adapter: 'evodiff',
          label: 'EvoDiff',
          params: {
            taskMode: 'infill',
            fixedSequence: 'm k t [MASK] #mask# <MASK> a c',
          },
        },
      ],
      edges: [
        ['pdb6', 'af1'],
        ['pp1', 'evo1'],
      ],
      summary: 'Sanitization test',
    };

    const compiled = compileWorkflowPlanToGraph(plan, undefined, { lenient: true }).graph;
    const rfd = compiled.nodes.find((n) => n.id === 'rfd1')!;
    const pp = compiled.nodes.find((n) => n.id === 'pp1')!;
    const af = compiled.nodes.find((n) => n.id === 'af1')!;
    const evo = compiled.nodes.find((n) => n.id === 'evo1')!;

    expect(rfd.params).not.toHaveProperty('numSequences');
    expect(rfd.params).not.toHaveProperty('topK');
    expect(rfd.params).not.toHaveProperty('autoSeededBenchmarkId');
    expect(pp.params.taskMode).toBe('motif');
    expect(pp.params).not.toHaveProperty('mode');
    expect(pp.params).not.toHaveProperty('topK');
    expect(af.params.variant).toBe('antifold');
    expect(af.params.numSequences).toBe(6);
    expect(af.params).not.toHaveProperty('maxSequences');
    expect(af.params).not.toHaveProperty('topK');
    expect(evo.params.fixedSequence).toBe('MKT###AC');

    // UI tracking keys on rfd4-proteina / genie3 are preserved across block_graph edit update_params
    const currentGraph: WorkflowGraph = {
      id: 'g-ui',
      name: 'UI Tracking',
      nodes: [
        {
          id: 'rfd_ui',
          adapterId: 'rfd4-proteina',
          params: {
            mode: 'unconditional',
            length: 70,
            autoSeededBenchmarkId: 'rfd_1prw',
            autoSeededLength: 70,
            userEditedLength: true,
            userEditedTask: false,
          },
        },
      ],
      edges: [],
    };
    const editPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'edit',
      editOps: [{ action: 'update_params', nodeId: 'rfd_ui', params: { length: 90 } }],
      summary: 'Update length',
    };
    const edited = compileWorkflowPlanToGraph(editPlan, currentGraph).graph;
    const rfdAfter = edited.nodes.find((n) => n.id === 'rfd_ui')!;
    expect(rfdAfter.params.length).toBe(90);
    expect(rfdAfter.params.autoSeededBenchmarkId).toBe('rfd_1prw');
    expect(rfdAfter.params.autoSeededLength).toBe(70);
    expect(rfdAfter.params.userEditedLength).toBe(true);
    expect(rfdAfter.params.userEditedTask).toBe(false);
  });

  it('§2.4 (#63): resolveBlockEdgePorts wires sequences for LocalFold/EvoDiff/FoldFlow2, structure for MPNN/RFD4/ESMC, and template for Genie2', () => {
    const plan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: { pdbId: '1UBQ' } },
        { id: 'af', adapter: 'localfold-af2', label: 'LocalFold', params: { model: 'monomer' } },
        { id: 'evo', adapter: 'evodiff', label: 'EvoDiff', params: { taskMode: 'infill' } },
        { id: 'ff2', adapter: 'foldflow2', label: 'FoldFlow 2', params: { taskMode: 'unconditional' } },
        { id: 'esmc', adapter: 'esmc', label: 'ESM-C 300M', params: { mode: 'wildtype' } },
        { id: 'mpnn', adapter: 'web-protein-mpnn', label: 'ProteinMPNN', params: {} },
        { id: 'rfd', adapter: 'rfd4-proteina', label: 'RFD4-Proteina', params: { mode: 'partial' } },
        { id: 'g2', adapter: 'genie2', label: 'Genie 2', params: { length: 60 } },
      ],
      edges: [
        ['pdb', 'af'],
        ['pdb', 'evo'],
        ['pdb', 'ff2'],
        ['pdb', 'esmc'],
        ['pdb', 'mpnn'],
        ['pdb', 'rfd'],
        ['pdb', 'g2'],
      ],
      summary: 'Port resolution matrix',
    };
    const g = compileWorkflowPlanToGraph(plan).graph;
    expect(validateGraph(g)).toEqual([]);
    const edgeTo = (id: string) => g.edges.find((e) => e.toNode === id)!;
    expect(edgeTo('af')).toMatchObject({ fromPort: 'sequences', toPort: 'sequences' });
    expect(edgeTo('evo')).toMatchObject({ fromPort: 'sequences', toPort: 'sequences' });
    expect(edgeTo('ff2')).toMatchObject({ fromPort: 'sequences', toPort: 'sequences' });
    expect(edgeTo('esmc')).toMatchObject({ fromPort: 'structure', toPort: 'structure' });
    expect(edgeTo('mpnn')).toMatchObject({ fromPort: 'structure', toPort: 'structure' });
    expect(edgeTo('rfd')).toMatchObject({ fromPort: 'structure', toPort: 'structure' });
    expect(edgeTo('g2')).toMatchObject({ fromPort: 'structure', toPort: 'template' });
  });

  it('§2.4 (#63): ExecutionScheduler materializes pdb-input.outputs.sequences for protein PDBs and omits for pure DNA', async () => {
    const protStruct = makeStructArtifact({ id: 'prot-1', length: 8, sequence: 'ACDEFGHI' });
    const derived = deriveSequenceSetFromStructure(protStruct);
    expect(derived?.candidates[0].sequence).toBe('ACDEFGHI');
    expect(derived?.sourceStructureId).toBe('prot-1');

    const emptyDnaStruct = makeStructArtifact({ id: 'dna-1', length: 0, sequence: '' });
    expect(deriveSequenceSetFromStructure(emptyDnaStruct)).toBeUndefined();

    const sched = new ExecutionScheduler();
    const graph: WorkflowGraph = {
      id: 'pdb-to-fold',
      name: 'PDB to LocalFold',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: makePdb({ nRes: 10 }) } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const res = await sched.run(graph, () => {});
    const pdbSeqs = res.outputsByNode.pdb.sequences as SequenceSet | undefined;
    expect(pdbSeqs?.kind).toBe('SequenceSet');
    expect(pdbSeqs?.candidates[0].sequence).toBe('A'.repeat(10));
    expect(res.outputsByNode.fold.folds.kind).toBe('FoldedStructure');
  });

  it('§2.4 (#63): 1LVM partial redesign repairs empty/inverted fixedPositions & cropSegments, and preserves fixedPositions across MPNN -> LigandMPNN', () => {
    const rawPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: { pdbId: '1LVM', cropSegments: 'A1-10' } },
        { id: 'mpnn', adapter: 'web-protein-mpnn', label: 'ProteinMPNN', params: { fixedPositions: 'A1-10' } },
      ],
      edges: [['pdb', 'mpnn']],
      summary: 'Redesign first 10 residues of 1LVM chain A',
    };
    const reconciled = reconcilePlannedWorkflow(
      rawPlan,
      'Redesign first 10 residues of 1LVM chain A with ProteinMPNN',
    );
    const compiled = compileWorkflowPlanToGraph(reconciled).graph;
    const pdb = compiled.nodes.find((n) => n.id === 'pdb')!;
    const mpnn = compiled.nodes.find((n) => n.id === 'mpnn')!;
    expect(pdb.params.cropSegments).toBe('A');
    expect(mpnn.params.fixedPositions).toBe('A11-9999');

    // Upgrading web-protein-mpnn -> ligandmpnn on ligand-containing PDB preserves fixedPositions
    const ligPdbGraph: WorkflowGraph = {
      id: '1lvm-lig',
      name: '1LVM Ligand',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          params: {
            mode: 'custom',
            pdbText: makePdb({
              nRes: 15,
              chain: 'A',
              hetatms: ['HETATM  999  ZN   ZN A 200       1.000   1.000   1.000  1.00 20.00          ZN  '],
            }),
          },
        },
        {
          id: 'mpnn',
          adapterId: 'web-protein-mpnn',
          params: { family: 'solublempnn', fixedPositions: 'A11-9999', seed: 42 },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' }],
    };
    const repaired = repairDraftedWorkflowFromBioFeedback(ligPdbGraph).graph;
    const upgraded = repaired.nodes.find((n) => n.id === 'mpnn')!;
    expect(upgraded.adapterId).toBe('ligandmpnn');
    expect(upgraded.params.fixedPositions).toBe('A11-9999');
    expect(upgraded.params.seed).toBe(42);
  });

  it('§2.4 (#63): 6XEY antibody AntiFold defaults/repairs targetChain to H while respecting explicit chain L and 1N8Z/1MEL', () => {
    const hallucinatedPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: { pdbId: '6XEY' } },
        { id: 'af', adapter: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold', targetChain: 'A' } },
      ],
      edges: [['pdb', 'af']],
      summary: 'Antibody CDR3 design on 6XEY',
    };
    const rec = reconcilePlannedWorkflow(hallucinatedPlan, 'Design CDR3 of antibody 6XEY with AntiFold');
    const g = compileWorkflowPlanToGraph(rec).graph;
    const af = g.nodes.find((n) => n.id === 'af')!;
    expect(af.params.targetChain).toBe('H');
    expect(af.params.regions).toBe('CDR3');

    // Explicit chain L is preserved
    const explicitL: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: { pdbId: '6XEY' } },
        { id: 'af', adapter: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold' } },
      ],
      edges: [['pdb', 'af']],
      summary: 'Light chain CDR1',
    };
    const gL = compileWorkflowPlanToGraph(
      reconcilePlannedWorkflow(explicitL, 'Design CDR1 of chain L of antibody 6XEY with AntiFold'),
    ).graph;
    expect(gL.nodes.find((n) => n.id === 'af')!.params.targetChain).toBe('L');

    // 1N8Z must stay '' so antifold.worker.ts auto-detects VH chain B
    const n8zPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: { pdbId: '1N8Z' } },
        { id: 'af', adapter: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold', regions: 'CDR3' } },
      ],
      edges: [['pdb', 'af']],
      summary: '1N8Z CDR3',
    };
    const gN8z = compileWorkflowPlanToGraph(
      reconcilePlannedWorkflow(n8zPlan, 'Design CDR3 of antibody 1N8Z with AntiFold'),
    ).graph;
    expect(gN8z.nodes.find((n) => n.id === 'af')!.params.targetChain).toBe('');

    // Legacy kind: 'pdb' plan also repairs targetChain for 6XEY
    const pdbRec = reconcilePlannedWorkflow(
      {
        kind: 'pdb',
        pdbId: '6XEY',
        crop: null,
        tasks: ['antifold'],
        targetChain: 'A',
        regions: null,
        numSequences: 2,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: '6XEY AntiFold',
      } satisfies PdbWorkflowPlan,
      'Design CDR1 and CDR3 of antibody 6XEY with AntiFold',
    ) as PdbWorkflowPlan;
    expect(pdbRec.targetChain).toBe('H');
  });

  it('§2.4 (#63): 1YCR disambiguates binder -> mdm2_1ycr vs motif -> rfd_1ycr_p53 and seeds Genie 3 / ProtPardelle', () => {
    const binderPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: { pdbId: '1YCR' } },
        { id: 'g3', adapter: 'genie3', label: 'Genie 3', params: { taskMode: 'binder', length: 70 } },
      ],
      edges: [['pdb', 'g3']],
      summary: 'MDM2 binder on 1YCR',
    };
    const gBinder = compileWorkflowPlanToGraph(binderPlan).graph;
    const pdbBinder = gBinder.nodes.find((n) => n.id === 'pdb')!;
    const g3Binder = gBinder.nodes.find((n) => n.id === 'g3')!;
    expect(pdbBinder.params.benchmarkId).toBe('mdm2_1ycr');
    expect(pdbBinder.params.cropSegments).toBe('A25-109');
    expect(g3Binder.params.hotspots).toBe('A54, A61, A62, A67, A72, A93');

    const motifPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: { pdbId: '1YCR' } },
        { id: 'pp', adapter: 'protpardelle', label: 'ProtPardelle-1c', params: { length: 80 } },
      ],
      edges: [['pdb', 'pp']],
      summary: 'p53 motif scaffold on 1YCR',
    };
    const gMotif = compileWorkflowPlanToGraph(motifPlan).graph;
    const pdbMotif = gMotif.nodes.find((n) => n.id === 'pdb')!;
    const ppMotif = gMotif.nodes.find((n) => n.id === 'pp')!;
    expect(pdbMotif.params.benchmarkId).toBe('rfd_1ycr_p53');
    expect(pdbMotif.params.cropSegments).toBe('B19-27');
    expect(ppMotif.params.taskMode).toBe('motif');
    expect(ppMotif.params.motifSelection).toBe('B19-27');
  });

  it('§2.4 (#63): 5O45 overrides overly broad cropSegments "A" with curated A17-132', () => {
    const plan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: { pdbId: '5O45', cropSegments: 'A' } },
        { id: 'rfd', adapter: 'rfd4-proteina', label: 'RFD4-Proteina', params: { mode: 'binder', length: 70 } },
      ],
      edges: [['pdb', 'rfd']],
      summary: 'PD-L1 binder on 5O45',
    };
    const g = compileWorkflowPlanToGraph(plan).graph;
    const pdb = g.nodes.find((n) => n.id === 'pdb')!;
    expect(pdb.params.benchmarkId).toBe('pdl1_5o45');
    expect(pdb.params.cropSegments).toBe('A17-132');
  });

  it('§2.4 (#63): designability-scorer BFS auto-wires all 11 generators (new & edit) and CA_ONLY_GENS sets Cα-ProteinMPNN when family omitted', () => {
    const all11Generators = [
      'rfd4-proteina',
      'genie3',
      'genie2',
      'protpardelle',
      'rfdiffusion',
      'boltzgen',
      'la-proteina',
      'proteina',
      'sid-protein',
      'frameflow',
      'foldflow2',
    ] as const;

    for (const genId of all11Generators) {
      const nodes: BlockGraphWorkflowPlan['nodes'] =
        genId === 'foldflow2'
          ? [
              { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: {} },
              { id: 'gen', adapter: genId, label: genId, params: {} },
              { id: 'mpnn', adapter: 'web-protein-mpnn', label: 'ProteinMPNN', params: {} },
              { id: 'fold', adapter: 'localfold-af2', label: 'LocalFold', params: {} },
              { id: 'sc', adapter: 'designability-scorer', label: 'Scorer', params: {} },
            ]
          : [
              { id: 'gen', adapter: genId, label: genId, params: {} },
              { id: 'mpnn', adapter: 'web-protein-mpnn', label: 'ProteinMPNN', params: {} },
              { id: 'fold', adapter: 'localfold-af2', label: 'LocalFold', params: {} },
              { id: 'sc', adapter: 'designability-scorer', label: 'Scorer', params: {} },
            ];
      const edges: Array<[string, string]> =
        genId === 'foldflow2'
          ? [
              ['pdb', 'gen'],
              ['gen', 'mpnn'],
              ['mpnn', 'fold'],
              ['fold', 'sc'],
            ]
          : [
              ['gen', 'mpnn'],
              ['mpnn', 'fold'],
              ['fold', 'sc'],
            ];
      const g = compileWorkflowPlanToGraph({
        kind: 'block_graph',
        mode: 'new',
        nodes,
        edges,
        summary: `Scorer BFS ${genId}`,
      }).graph;
      const scStructEdge = g.edges.find((e) => e.toNode === 'sc' && e.toPort === 'structure');
      expect(scStructEdge?.fromNode).toBe('gen');

      if (genId === 'genie3' || genId === 'genie2' || genId === 'proteina' || genId === 'sid-protein') {
        const mpnn = g.nodes.find((n) => n.id === 'mpnn')!;
        expect(mpnn.params.family).toBe('ca');
        expect(mpnn.label).toBe('Cα-ProteinMPNN');
      }
    }

    // Explicit family override is respected, and mode: 'edit' auto-wires scorer structure
    const baseGraph: WorkflowGraph = {
      id: 'edit-scorer',
      name: 'Edit Scorer',
      nodes: [
        { id: 'lap', adapterId: 'la-proteina', params: { length: 100 } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'proteinmpnn' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer' } },
      ],
      edges: [
        { fromNode: 'lap', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const edited = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [
          { action: 'add_node', nodeId: 'sc_new', adapter: 'designability-scorer', params: {} },
          { action: 'add_edge', fromId: 'fold', toId: 'sc_new' },
        ],
        summary: 'Add scorer',
      },
      baseGraph,
    ).graph;
    expect(edited.edges.some((e) => e.fromNode === 'lap' && e.toNode === 'sc_new' && e.toPort === 'structure')).toBe(true);
  });

  it('§3.5a (#72): antifold-non-antibody warns on 1LYZ & >=90aa non-Ig structures without false positives on 1N8Z, 1MEL, 6W41, 6XEY, 1IGT', () => {
    // 1. 1LYZ pre-fetch, live, custom, and even with invalid targetChain='H' (absent from 1LYZ chain A)
    const lyzPreFetch: WorkflowGraph = {
      id: '1lyz-pre',
      name: '1LYZ Pre',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1LYZ' } },
        { id: 'af', adapterId: 'antifold-esmif1', params: { variant: 'antifold' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    expect(validateBioCapabilities(lyzPreFetch).some((i) => i.code === 'antifold-non-antibody')).toBe(true);

    const lyz129Pdb = makePdb({ nRes: 129, chain: 'A', header: 'HEADER    HYDROLASE                               1LYZ' });
    const lyzCustom: WorkflowGraph = {
      id: '1lyz-custom',
      name: '1LYZ Custom',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: lyz129Pdb } },
        { id: 'af', adapterId: 'antifold-esmif1', params: { variant: 'antifold', targetChain: 'H' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    expect(validateBioCapabilities(lyzCustom).some((i) => i.code === 'antifold-non-antibody')).toBe(true);

    // 2. Intermediate non-Ig LocalFold/LASErMPNN >=90 aa live output warns
    const nonIgLive = makeStructArtifact({ id: 'fold-150', length: 150, chainIds: Array(150).fill('A') });
    const nonIgLiveGraph: WorkflowGraph = {
      id: 'fold-antifold',
      name: 'Fold AntiFold',
      nodes: [
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer' } },
        { id: 'af', adapterId: 'antifold-esmif1', params: { variant: 'antifold' } },
      ],
      edges: [{ fromNode: 'fold', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(nonIgLiveGraph, { fold: { structure: nonIgLive } }).some(
        (i) => i.code === 'antifold-non-antibody',
      ),
    ).toBe(true);

    // 3. Negative controls: 1N8Z (A/B/C Kabat hallmarks), 1MEL (mmCIF header & IMGT VHH on chain A), 6W41, 6XEY, 1IGT
    const n8zResNums = [22, 36, 92, 103, 104, 105];
    const n8zResNames = ['CYS', 'TRP', 'CYS', 'TRP', 'ALA', 'ALA'];
    const n8zPdb = makePdb({
      nRes: 6,
      chain: 'B',
      resNums: n8zResNums,
      resNames: n8zResNames,
      insCodes: ['', '', '', '', '', ''],
    });
    const n8zGraph: WorkflowGraph = {
      id: '1n8z',
      name: '1N8Z',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1N8Z', pdbText: n8zPdb } },
        { id: 'af', adapterId: 'antifold-esmif1', params: { variant: 'antifold' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    expect(validateBioCapabilities(n8zGraph).some((i) => i.code === 'antifold-non-antibody')).toBe(false);

    // IMGT VHH hallmarks on chain A (even with C23 -> kabatVL=1, imgt=4)
    expect(
      hasAntibodyChainHallmarks([
        {
          id: 'A',
          residues: [
            { number: 23, insertion: '', aa1: 'C' },
            { number: 41, insertion: '', aa1: 'W' },
            { number: 104, insertion: '', aa1: 'C' },
            { number: 118, insertion: '', aa1: 'W' },
          ],
        },
      ]),
    ).toBe(true);

    const melCif = 'data_1MEL\n_struct.title "CAMELID VHH NANOBODY COMPLEX"\n_atom_site.id 1\n';
    const melStruct = makeStructArtifact({
      id: '1mel-live',
      length: 120,
      cifText: melCif,
      pdbText: makePdb({ nRes: 120, chain: 'A' }),
      chainIds: Array(120).fill('A'),
    });
    const melGraph: WorkflowGraph = {
      id: '1mel',
      name: '1MEL',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1MEL' } },
        { id: 'af', adapterId: 'antifold-esmif1', params: { variant: 'antifold' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(melGraph, { pdb: { structure: melStruct } }).some(
        (i) => i.code === 'antifold-non-antibody',
      ),
    ).toBe(false);

    for (const abPdb of ['6W41', '6XEY', '1IGT']) {
      const abGraph: WorkflowGraph = {
        id: `ab-${abPdb}`,
        name: abPdb,
        nodes: [
          {
            id: 'pdb',
            adapterId: 'pdb-input',
            params: { mode: 'pdb-id', pdbId: abPdb, pdbText: makePdb({ nRes: 10, chain: 'H' }) },
          },
          { id: 'af', adapterId: 'antifold-esmif1', params: { variant: 'antifold' } },
        ],
        edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
      };
      expect(validateBioCapabilities(abGraph).some((i) => i.code === 'antifold-non-antibody')).toBe(false);
    }
  });

  it('§3.5b (#73): Cα-only sources block all 5 full-backbone consumers in Layer 1, Layer 2, and ExecutionScheduler.run', async () => {
    const caSources: Array<{ adapterId: string; params: Record<string, unknown> }> = [
      { adapterId: 'genie3', params: { taskMode: 'unconditional', length: 60 } },
      { adapterId: 'genie2', params: { length: 60 } },
      { adapterId: 'proteina', params: { length: 60 } },
      { adapterId: 'sid-protein', params: { length: 60 } },
      { adapterId: 'pdb-input', params: { mode: 'custom', pdbText: makePdb({ nRes: 10, caOnly: true }) } },
    ];
    const consumers: Array<{ adapterId: string; params: Record<string, unknown> }> = [
      { adapterId: 'web-protein-mpnn', params: { family: 'solublempnn' } },
      { adapterId: 'ligandmpnn', params: {} },
      { adapterId: 'lasermpnn', params: {} },
      { adapterId: 'antifold-esmif1', params: { variant: 'antifold' } },
      { adapterId: 'antifold-esmif1', params: { variant: 'esmif1' } },
      { adapterId: 'thermompnn', params: {} },
    ];

    for (const src of caSources) {
      for (const dst of consumers) {
        const g: WorkflowGraph = {
          id: `ca-${src.adapterId}-${dst.adapterId}`,
          name: 'CA Matrix',
          nodes: [
            { id: 'src', adapterId: src.adapterId, params: src.params },
            { id: 'dst', adapterId: dst.adapterId, params: dst.params },
          ],
          edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'dst', toPort: 'structure' }],
        };
        const issues = validateBioCapabilities(g);
        const err = issues.find((i) => i.code === 'mpnn-ca-only-backbone');
        expect(err?.severity).toBe('error');
        if (dst.adapterId === 'web-protein-mpnn') {
          expect(err?.fix?.label).toBe('Switch to Cα-ProteinMPNN');
        }
        await expect(new ExecutionScheduler().run(g, () => {})).rejects.toThrow(/Cα-only/i);
      }
    }

    // Layer 2 validateRuntimeNodeInputs throws on Cα-only live artifact for all 5 consumers
    const caArt = makeStructArtifact({ id: 'ca-live', length: 10, caOnly: true });
    for (const dst of consumers) {
      expect(() =>
        validateRuntimeNodeInputs(
          { id: 'dst', adapterId: dst.adapterId, params: dst.params },
          { structure: caArt },
        ),
      ).toThrow(/Cα-only/i);
    }

    // Safe Cα consumers & partial missing-backbone crystal PDB do NOT block
    for (const safe of [
      { adapterId: 'web-protein-mpnn', params: { family: 'ca' } },
      { adapterId: 'esmc', params: {} },
      { adapterId: 'dyna1', params: {} },
      { adapterId: 'designability-scorer', params: {} },
    ]) {
      const safeGraph: WorkflowGraph = {
        id: `safe-${safe.adapterId}`,
        name: 'Safe CA',
        nodes: [
          { id: 'g3', adapterId: 'genie3', params: { length: 60 } },
          { id: 'dst', adapterId: safe.adapterId, params: safe.params },
        ],
        edges: [{ fromNode: 'g3', fromPort: 'structure', toNode: 'dst', toPort: 'structure' }],
      };
      expect(
        validateBioCapabilities(safeGraph).filter((i) => i.severity === 'error'),
      ).toEqual([]);
    }

    const partialBackbonePdb = [
      makePdb({ nRes: 9, chain: 'A', startNum: 1 }).replace(/\nEND$/, ''),
      'ATOM     99  CA  ALA A  10      38.000   1.000   0.000  1.00 20.00           C  ',
      'END',
    ].join('\n');
    const partialGraph: WorkflowGraph = {
      id: 'partial-bb',
      name: 'Partial Backbone',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: partialBackbonePdb } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'solublempnn' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' }],
    };
    const partialIssue = validateBioCapabilities(partialGraph).find(
      (i) => i.code === 'mpnn-ca-only-backbone',
    );
    expect(partialIssue?.severity).toBe('warning');
  });

  it('Bio P0-2b (#73): inferNodeOutputProfile does not leak 1LVM uncropped template tokens in RFD4/Genie3 motif mode and retains 1CTT non-buffer ligands', () => {
    const live1Lvm: Record<string, Record<string, CanonicalArtifact>> = {
      pdb: {
        structure: makeStructArtifact({
          id: '1lvm-live',
          length: 467,
          sequence: 'A'.repeat(467),
          diagnostics: {
            chainCount: 5,
            chainLengths: [95, 95, 95, 95, 87],
            hasLigands: true,
            ligandNames: ['SO4', 'GOL'],
            hasNucleicAcids: false,
            nucleicAcidChains: [],
          },
        }),
      },
    };
    const rfdMotifGraph: WorkflowGraph = {
      id: '1lvm-rfd4-motif',
      name: '1LVM RFD4 Motif',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1LVM' } },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          params: { mode: 'motif', length: 80, motifSpan: 'A1-10', contig: '35-35,A1-10,35-35' },
        },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'solublempnn' } },
        { id: 'fold_b', adapterId: 'localfold-af2', params: { model: 'boltz2' } },
        { id: 'fold_m', adapterId: 'localfold-af2', params: { model: 'monomer' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' },
        { fromNode: 'rfd', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold_b', toPort: 'sequences' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold_m', toPort: 'sequences' },
      ],
    };
    const rfdIssues = validateBioCapabilities(rfdMotifGraph, live1Lvm);
    expect(rfdIssues.filter((i) => i.code === 'multimer-into-af2-monomer')).toEqual([]);
    expect(rfdIssues.filter((i) => i.code === 'webgpu-token-ceiling')).toEqual([]);
    expect(rfdIssues.filter((i) => i.code === 'af3-apo-refold-note')).toEqual([]);

    const genieMotifGraph: WorkflowGraph = {
      id: '1lvm-g3-motif',
      name: '1LVM Genie3 Motif',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1LVM' } },
        { id: 'g3', adapterId: 'genie3', params: { taskMode: 'motif', length: 80, motifSpan: 'A1-10' } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'ca' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'boltz2' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' },
        { fromNode: 'g3', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const g3Issues = validateBioCapabilities(genieMotifGraph, live1Lvm);
    expect(g3Issues.filter((i) => i.code === 'webgpu-token-ceiling')).toEqual([]);

    // 1CTT retains ZN + DHZ ligands in RFD4 motif mode without multimer false positive
    const live1Ctt: Record<string, Record<string, CanonicalArtifact>> = {
      pdb: {
        structure: makeStructArtifact({
          id: '1ctt-live',
          length: 294,
          sequence: 'A'.repeat(294),
          diagnostics: {
            chainCount: 2,
            chainLengths: [147, 147],
            hasLigands: true,
            ligandNames: ['ZN', 'DHZ'],
            hasNucleicAcids: false,
            nucleicAcidChains: [],
          },
        }),
      },
    };
    const cttMotifGraph: WorkflowGraph = {
      id: '1ctt-rfd4-motif',
      name: '1CTT RFD4 Motif',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1CTT' } },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          params: {
            mode: 'motif',
            length: 150,
            contig: '150-150',
            motifSpan: 'A102, A104, A129, A132',
            unindexedMotif: true,
          },
        },
        { id: 'mpnn', adapterId: 'ligandmpnn', params: { useLigandContext: true } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' },
        { fromNode: 'rfd', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const cttIssues = validateBioCapabilities(cttMotifGraph, live1Ctt);
    expect(cttIssues.some((i) => i.code === 'af2-non-protein-entity')).toBe(true);
    expect(cttIssues.some((i) => i.code === 'multimer-into-af2-monomer')).toBe(false);
  });

  it('§2.2 & §2.5c (#61, #65): structure-compare preserves reference + all structures and exempts layout/colorMode/alignKabsch from invalidation & cache keys', async () => {
    const graph1: WorkflowGraph = {
      id: 'cmp-cache',
      name: 'Compare Cache',
      nodes: [
        { id: 'ref', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: makePdb({ nRes: 6, chain: 'A' }) } },
        { id: 's1', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: makePdb({ nRes: 7, chain: 'A' }) } },
        { id: 's2', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: makePdb({ nRes: 8, chain: 'A' }) } },
        {
          id: 'cmp',
          adapterId: 'structure-compare',
          params: { layout: 'overlay', colorMode: 'by-model', alignKabsch: true },
        },
      ],
      edges: [
        { fromNode: 'ref', fromPort: 'structure', toNode: 'cmp', toPort: 'reference' },
        { fromNode: 's1', fromPort: 'structure', toNode: 'cmp', toPort: 'structures' },
        { fromNode: 's2', fromPort: 'structure', toNode: 'cmp', toPort: 'structures' },
      ],
    };

    const graph2: WorkflowGraph = {
      ...graph1,
      nodes: graph1.nodes.map((n) =>
        n.id === 'cmp'
          ? { ...n, params: { layout: 'side-by-side', colorMode: 'plddt', alignKabsch: false } }
          : n,
      ),
    };
    const inv = computeInvalidatedNodeIds(graph1, graph2);
    expect(inv.staleNodeIds).toEqual([]);
    expect(inv.validNodeIds).toEqual(['ref', 's1', 's2', 'cmp']);

    const scheduler = new ExecutionScheduler();
    const res1 = await scheduler.run(graph1, () => {});
    const cmpOut = res1.outputsByNode.cmp.structure as StructureArtifact;
    const refId = (res1.outputsByNode.ref.structure as StructureArtifact).id;
    const s1Id = (res1.outputsByNode.s1.structure as StructureArtifact).id;
    const s2Id = (res1.outputsByNode.s2.structure as StructureArtifact).id;
    expect(cmpOut.parentStructureId).toBe(refId);
    expect(cmpOut.collection?.items.map((item) => item.id)).toEqual([refId, s1Id, s2Id]);

    const cachedNodes: string[] = [];
    await scheduler.run(graph2, (ev) => {
      if (ev.status === 'cached') cachedNodes.push(ev.nodeId);
    });
    expect(cachedNodes).toContain('cmp');
  });

  it('Bio-02a (#64): conditioning CRYSTALLIZATION_BUFFERS filters peptide caps ACE, NME, NH2, FOR', () => {
    for (const cap of ['ACE', 'NME', 'NH2', 'FOR']) {
      expect(CRYSTALLIZATION_BUFFERS.has(cap)).toBe(true);
      expect(isCrystallizationBuffer(cap)).toBe(true);
    }
    const pdbWithAce = makePdb({
      nRes: 10,
      chain: 'A',
      hetatms: ['HETATM  900  CH3 ACE A   0       0.000   0.000   0.000  1.00 20.00           C  '],
    });
    const ctx = extractRfd4TemplateContext(pdbWithAce);
    const compiled = compileConditionToRfd4({ contig: '80-80', motifSpans: ['A1-5'] }, ctx);
    expect(compiled.contig).not.toContain('ACE');
  });

  it('M2-R2 Finding 1: block_graph edit update_params preserves custom pdbText/mode/tracking keys and rejects op.params tracking injection', () => {
    const validPdb = makePdb({ nRes: 12, chain: 'A' });
    const currentGraph: WorkflowGraph = {
      id: 'custom-pdb-edit',
      name: 'Custom PDB Edit',
      nodes: [
        {
          id: 'pdb_c',
          adapterId: 'pdb-input',
          params: {
            mode: 'custom',
            pdbText: validPdb,
            filename: 'my_upload.pdb',
            fetchedPdbId: 'CUSTOM',
          },
        },
        {
          id: 'rfd_c',
          adapterId: 'rfd4-proteina',
          params: {
            mode: 'unconditional',
            length: 70,
            customSpecYaml: 'task: custom_spec',
            autoSeededBenchmarkId: 'rfd_1prw',
            userEditedLength: false,
          },
        },
      ],
      edges: [],
    };

    const edited = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [
          { action: 'update_params', nodeId: 'pdb_c', params: { cropSegments: 'A1-6' } },
          {
            action: 'update_params',
            nodeId: 'rfd_c',
            params: {
              length: 90,
              autoSeededBenchmarkId: 'injected_by_llm',
              autoSeededLength: 999,
              userEditedLength: true,
              userEditedTask: true,
              customSpecYaml: 'injected_yaml',
            },
          },
        ],
        summary: 'Crop custom PDB and update RFD4 length',
      },
      currentGraph,
    ).graph;

    const pdbAfter = edited.nodes.find((n) => n.id === 'pdb_c')!;
    expect(pdbAfter.params.mode).toBe('custom');
    expect(pdbAfter.params.pdbText).toBe(validPdb);
    expect(pdbAfter.params.filename).toBe('my_upload.pdb');
    expect(pdbAfter.params.fetchedPdbId).toBe('CUSTOM');
    expect(pdbAfter.params.cropSegments).toBe('A1-6');

    const rfdAfter = edited.nodes.find((n) => n.id === 'rfd_c')!;
    expect(rfdAfter.params.length).toBe(90);
    expect(rfdAfter.params.autoSeededBenchmarkId).toBe('rfd_1prw');
    expect(rfdAfter.params.userEditedLength).toBe(false);
    expect(rfdAfter.params).not.toHaveProperty('autoSeededLength');
    expect(rfdAfter.params).not.toHaveProperty('userEditedTask');
    expect(rfdAfter.params.customSpecYaml).toBe('task: custom_spec');
  });

  it('M2-R2 Finding 2: 6XEY explicit chain L overrides hallucinated A, H1-95 keeps H, CDR3 narrows default regions, and 1N8Z/1MEL leave targetChain empty', () => {
    // 1. 6XEY explicit chain L repairs hallucinated targetChain: 'A' in block_graph and kind: 'pdb'
    const bgChainL = compileWorkflowPlanToGraph(
      reconcilePlannedWorkflow(
        {
          kind: 'block_graph',
          mode: 'new',
          nodes: [
            { id: 'pdb', adapter: 'pdb-input', label: 'PDB', params: { pdbId: '6XEY' } },
            { id: 'af', adapter: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold', targetChain: 'A' } },
          ],
          edges: [['pdb', 'af']],
          summary: '6XEY chain L',
        },
        'Design CDR1 of chain L of antibody 6XEY with AntiFold',
      ),
    ).graph;
    const afBgL = bgChainL.nodes.find((n) => n.id === 'af')!;
    expect(afBgL.params.targetChain).toBe('L');
    expect(afBgL.params.regions).toBe('CDR1');

    const pdbChainL = reconcilePlannedWorkflow(
      {
        kind: 'pdb',
        pdbId: '6XEY',
        crop: null,
        tasks: ['antifold'],
        targetChain: 'A',
        regions: 'CDR1 CDR2 CDR3',
        numSequences: 2,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: '6XEY chain L',
      } satisfies PdbWorkflowPlan,
      'Design CDR1 of chain L of antibody 6XEY with AntiFold',
    ) as PdbWorkflowPlan;
    expect(pdbChainL.targetChain).toBe('L');
    expect(pdbChainL.regions).toBe('CDR1');

    // 2. 6XEY with framework lock H1-95 still assigns targetChain: 'H', whereas standalone H leaves ''
    const bgFrameworkLock = compileWorkflowPlanToGraph(
      reconcilePlannedWorkflow(
        {
          kind: 'block_graph',
          mode: 'new',
          nodes: [
            { id: 'pdb', adapter: 'pdb-input', label: 'PDB', params: { pdbId: '6XEY' } },
            {
              id: 'af',
              adapter: 'antifold-esmif1',
              label: 'AntiFold',
              params: { variant: 'antifold', fixedPositions: 'H1-95', regions: 'CDR1 CDR2 CDR3' },
            },
          ],
          edges: [['pdb', 'af']],
          summary: '6XEY CDR3 with H1-95 lock',
        },
        'Design only CDR3 of antibody 6XEY with AntiFold',
      ),
    ).graph;
    const afFw = bgFrameworkLock.nodes.find((n) => n.id === 'af')!;
    expect(afFw.params.targetChain).toBe('H');
    expect(afFw.params.regions).toBe('CDR3');

    const bgWholeChainLock = compileWorkflowPlanToGraph(
      reconcilePlannedWorkflow(
        {
          kind: 'block_graph',
          mode: 'new',
          nodes: [
            { id: 'pdb', adapter: 'pdb-input', label: 'PDB', params: { pdbId: '6XEY' } },
            {
              id: 'af',
              adapter: 'antifold-esmif1',
              label: 'AntiFold',
              params: { variant: 'antifold', fixedPositions: 'H' },
            },
          ],
          edges: [['pdb', 'af']],
          summary: '6XEY whole H locked',
        },
        'Design CDR3 of antibody 6XEY with AntiFold',
      ),
    ).graph;
    expect(bgWholeChainLock.nodes.find((n) => n.id === 'af')!.params.targetChain).toBe('');

    // 3. 1N8Z & 1MEL with CDR-H3 / heavy chain leave targetChain: '' (even if LLM hallucinated 'A')
    for (const [pdbId, prompt] of [
      ['1N8Z', 'Design CDR-H3 of antibody 1N8Z with AntiFold'],
      ['1N8Z', 'Design heavy chain CDR3 of 1N8Z with AntiFold'],
      ['1MEL', 'Design CDR-H3 of nanobody 1MEL with AntiFold'],
    ] as const) {
      const gExcl = compileWorkflowPlanToGraph(
        reconcilePlannedWorkflow(
          {
            kind: 'block_graph',
            mode: 'new',
            nodes: [
              { id: 'pdb', adapter: 'pdb-input', label: 'PDB', params: { pdbId } },
              {
                id: 'af',
                adapter: 'antifold-esmif1',
                label: 'AntiFold',
                params: { variant: 'antifold', targetChain: 'A', regions: 'CDR1 CDR2 CDR3' },
              },
            ],
            edges: [['pdb', 'af']],
            summary: `${pdbId} CDR-H3`,
          },
          prompt,
        ),
      ).graph;
      const afExcl = gExcl.nodes.find((n) => n.id === 'af')!;
      expect(afExcl.params.targetChain).toBe('');
      expect(afExcl.params.regions).toBe('CDR3');

      const pdbExcl = reconcilePlannedWorkflow(
        {
          kind: 'pdb',
          pdbId,
          crop: null,
          tasks: ['antifold'],
          targetChain: 'A',
          regions: 'CDR1 CDR2 CDR3',
          numSequences: 2,
          topK: null,
          refold: false,
          foldModels: [],
          stabilityScan: false,
          summary: `${pdbId} CDR-H3`,
        } satisfies PdbWorkflowPlan,
        prompt,
      ) as PdbWorkflowPlan;
      expect(pdbExcl.targetChain).toBe('');
      expect(pdbExcl.regions).toBe('CDR3');
    }
  });

  it('M2-R2 Finding 3: 1YCR motif prompt mentioning MDM2-p53 complex resolves to rfd_1ycr_p53 motif instead of mdm2_1ycr binder', () => {
    const motifWithMdm2Mention = compileWorkflowPlanToGraph(
      reconcilePlannedWorkflow(
        {
          kind: 'block_graph',
          mode: 'new',
          nodes: [
            { id: 'pdb', adapter: 'pdb-input', label: 'PDB', params: { pdbId: '1YCR' } },
            { id: 'rfd', adapter: 'rfd4-proteina', label: 'RFD4', params: { mode: 'motif', length: 70 } },
          ],
          edges: [['pdb', 'rfd']],
          summary: '1YCR p53 motif',
        },
        'Scaffold the p53 helix motif from 1YCR (MDM2-p53 complex) with RFD4',
      ),
    ).graph;

    const pdbNode = motifWithMdm2Mention.nodes.find((n) => n.id === 'pdb')!;
    const rfdNode = motifWithMdm2Mention.nodes.find((n) => n.id === 'rfd')!;
    expect(pdbNode.params.benchmarkId).toBe('rfd_1ycr_p53');
    expect(pdbNode.params.cropSegments).toBe('B19-27');
    expect(rfdNode.params.mode).toBe('motif');
    expect(String(rfdNode.params.contig)).toContain('B19-27');
    expect(String(rfdNode.params.contig)).not.toContain('/0');
  });

  it('M2-R2 Finding 4: filters Cα-ProteinMPNN from unsupported and normalizes null/invalid mode/taskMode/variant params', () => {
    const unsupRes = compileWorkflowPlanToGraph({
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'evo1', adapter: 'evodiff', label: 'EvoDiff', params: { taskMode: 'unconditional', length: 64 } },
      ],
      edges: [],
      unsupported: ['Cα-ProteinMPNN', 'Chroma'],
      summary: 'Unsupported filter',
    });
    expect(unsupRes.unsupported).toEqual(['Chroma']);

    const sanitizedGraph = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'new',
        nodes: [
          { id: 'pp', adapter: 'protpardelle', label: 'PP', params: { taskMode: null, mode: 'motif' } },
          { id: 'rfd', adapter: 'rfd4-proteina', label: 'RFD4', params: { mode: 'bogus_mode' } },
          { id: 'g3', adapter: 'genie3', label: 'Genie 3', params: { taskMode: 'bogus_mode' } },
          { id: 'esmc', adapter: 'esmc', label: 'ESM-C', params: { mode: 'bogus_mode' } },
          { id: 'af_mode', adapter: 'antifold-esmif1', label: 'ESM-IF1', params: { mode: 'esmif1' } },
          {
            id: 'af_prio',
            adapter: 'antifold-esmif1',
            label: 'ESM-IF1',
            params: { model: 'antifold', variant: 'esmif1' },
          },
        ],
        edges: [],
        summary: 'Mode/alias sanitization',
      },
      undefined,
      { lenient: true },
    ).graph;

    expect(sanitizedGraph.nodes.find((n) => n.id === 'pp')!.params.taskMode).toBe('motif');
    expect(sanitizedGraph.nodes.find((n) => n.id === 'rfd')!.params.mode).toBe('unconditional');
    expect(sanitizedGraph.nodes.find((n) => n.id === 'g3')!.params.taskMode).toBe('unconditional');
    expect(sanitizedGraph.nodes.find((n) => n.id === 'esmc')!.params.mode).toBe('wildtype');
    expect(sanitizedGraph.nodes.find((n) => n.id === 'af_mode')!.params.variant).toBe('esmif1');
    expect(sanitizedGraph.nodes.find((n) => n.id === 'af_prio')!.params.variant).toBe('esmif1');
  });

  it('M2-R2 Finding 5: pre-fetch 1LYZ, 1UBQ, 5O45, demo-bundle, and rfd4-proteina with targetChain H still emit antifold-non-antibody', () => {
    const nonAbSources: Array<{ adapterId: string; params: Record<string, unknown> }> = [
      { adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1LYZ' } },
      { adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1UBQ' } },
      { adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' } },
      { adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
      { adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 80 } },
    ];

    for (const src of nonAbSources) {
      const g: WorkflowGraph = {
        id: `prefetch-nonab-${src.adapterId}`,
        name: 'Pre-fetch non-antibody with targetChain H',
        nodes: [
          { id: 'src', adapterId: src.adapterId, params: src.params },
          { id: 'af', adapterId: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold', targetChain: 'H' } },
        ],
        edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
      };
      expect(validateBioCapabilities(g).some((i) => i.code === 'antifold-non-antibody')).toBe(true);
    }
  });

  it('M2-R2 Finding 6: estimateContigProteinMetrics supports rfdiffusion "/" delimiters & multi-char chains, and 5O45 binder pre-fetch does not double-count approxTokens', () => {
    // 1. rfdiffusion slash-delimited contig exceeding AF2 512 ceiling (200+20+200+150 = 570)
    const rfdSlashGraph: WorkflowGraph = {
      id: 'rfdiff-slash-ceiling',
      name: 'RFdiffusion slash ceiling',
      nodes: [
        {
          id: 'rfdiff',
          adapterId: 'rfdiffusion',
          params: { taskMode: 'motif', contig: '200-200/A1-20/200-200/150-150', length: 80 },
        },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'proteinmpnn' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer' } },
      ],
      edges: [
        { fromNode: 'rfdiff', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    expect(validateBioCapabilities(rfdSlashGraph).some((i) => i.code === 'webgpu-token-ceiling')).toBe(true);

    // 2. Multi-char chain IDs AA1-320 and AA:1-320 in contig (200 + 320 = 520 > 512)
    for (const contig of ['200-200,AA1-320', '200-200,AA:1-320']) {
      const multiChainContigGraph: WorkflowGraph = {
        id: 'multi-char-contig',
        name: 'Multi-char contig',
        nodes: [
          { id: 'rfd', adapterId: 'rfd4-proteina', params: { mode: 'motif', contig, length: 70 } },
          { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'proteinmpnn' } },
          { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer' } },
        ],
        edges: [
          { fromNode: 'rfd', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
          { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        ],
      };
      expect(validateBioCapabilities(multiChainContigGraph).some((i) => i.code === 'webgpu-token-ceiling')).toBe(true);
    }

    // 3. Pre-fetch 5O45 RFD4 binder L=220 (116 + 220 = 336 <= 384 Boltz-2 ceiling -> no false ceiling warning)
    const pdl1PreFetch220: WorkflowGraph = {
      id: '5o45-prefetch-220',
      name: '5O45 pre-fetch L=220',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' } },
        { id: 'rfd', adapterId: 'rfd4-proteina', params: { mode: 'binder', lockChain: 'A', length: 220 } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'solublempnn' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'boltz2' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' },
        { fromNode: 'rfd', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    expect(validateBioCapabilities(pdl1PreFetch220).some((i) => i.code === 'webgpu-token-ceiling')).toBe(false);

    // Pre-fetch 5O45 RFD4 binder L=300 (116 + 300 = 416 > 384 -> triggers ceiling warning)
    const pdl1PreFetch300: WorkflowGraph = {
      ...pdl1PreFetch220,
      nodes: pdl1PreFetch220.nodes.map((n) =>
        n.id === 'rfd' ? { ...n, params: { ...n.params, length: 300 } } : n,
      ),
    };
    expect(validateBioCapabilities(pdl1PreFetch300).some((i) => i.code === 'webgpu-token-ceiling')).toBe(true);
  });

  it('M2-R3 Finding 1: update_params switching custom pdb-input to pdbId 1LYZ clears stale pdbText/filename and emits antifold-non-antibody', () => {
    const custom6xeyPdb = makePdb({ nRes: 15, chain: 'H' });
    const currentGraph: WorkflowGraph = {
      id: 'custom-to-1lyz',
      name: 'Custom 6XEY to 1LYZ',
      nodes: [
        {
          id: 'pdb_c',
          adapterId: 'pdb-input',
          params: {
            mode: 'custom',
            pdbText: custom6xeyPdb,
            filename: '6xey.pdb',
            fetchedPdbId: 'CUSTOM',
          },
        },
        {
          id: 'af',
          adapterId: 'antifold-esmif1',
          params: { variant: 'antifold', targetChain: 'H' },
        },
      ],
      edges: [{ fromNode: 'pdb_c', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };

    const compiled = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [{ action: 'update_params', nodeId: 'pdb_c', params: { pdbId: '1LYZ' } }],
        summary: 'Switch custom PDB to 1LYZ',
      },
      currentGraph,
    );
    const edited = compiled.graph;

    const pdbAfter = edited.nodes.find((n) => n.id === 'pdb_c')!;
    const afAfter = edited.nodes.find((n) => n.id === 'af')!;
    expect(pdbAfter.params.mode).toBe('pdb-id');
    expect(pdbAfter.params.pdbId).toBe('1LYZ');
    expect(pdbAfter.params.pdbText).toBeUndefined();
    expect(pdbAfter.params.fetchedPdbId).toBeUndefined();
    expect(pdbAfter.params.filename).toBeUndefined();
    // Because stale 6XEY pdbText was cleared, repairDraftedWorkflowFromBioFeedback detects
    // antifold-non-antibody on 1LYZ and auto-switches AntiFold -> ESM-IF1.
    expect(afAfter.params.variant).toBe('esmif1');
    expect(compiled.notes.some((n) => n.includes('Switch to ESM-IF1'))).toBe(true);
    expect(
      validateBioCapabilities({
        ...edited,
        nodes: edited.nodes.map((n) =>
          n.id === 'af' ? { ...n, params: { variant: 'antifold', targetChain: 'H' } } : n,
        ),
      }).some((i) => i.code === 'antifold-non-antibody'),
    ).toBe(true);
  });

  it('M2-R3 Finding 2: update_params switching 1YCR to pdbId 5O45 or benchmarkId pdl1_5o45 clears stale 1YCR state and seeds 5O45 cleanly without B19-27 leak', () => {
    const stub1ycrPdb = makePdb({ nRes: 20, chain: 'A' });
    const baseGraph: WorkflowGraph = {
      id: 'ycr-to-5o45',
      name: '1YCR to 5O45',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          params: {
            mode: 'pdb-id',
            pdbId: '1YCR',
            fetchedPdbId: '1YCR',
            pdbText: stub1ycrPdb,
            benchmarkId: 'rfd_1ycr_p53',
            cropSegments: 'B19-27',
          },
        },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          params: {
            mode: 'motif',
            length: 70,
            motifSpan: 'B19-27',
            contig: '28-28,B19-27,28-28',
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' }],
    };

    // Case A: update_params { pdbId: '5O45' }
    const editedByPdbId = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [{ action: 'update_params', nodeId: 'pdb', params: { pdbId: '5O45' } }],
        summary: 'Switch 1YCR to 5O45',
      },
      baseGraph,
    ).graph;

    const pdbA = editedByPdbId.nodes.find((n) => n.id === 'pdb')!;
    const rfdA = editedByPdbId.nodes.find((n) => n.id === 'rfd')!;
    expect(pdbA.params.mode).toBe('pdb-id');
    expect(pdbA.params.pdbId).toBe('5O45');
    expect(pdbA.params.pdbText).toBeUndefined();
    expect(pdbA.params.fetchedPdbId).toBeUndefined();
    expect(pdbA.params.benchmarkId).toBe('pdl1_5o45');
    expect(pdbA.params.cropSegments).toBe('A17-132');
    expect(rfdA.params.mode).toBe('binder');
    expect(rfdA.params.motifSpan).toBeUndefined();
    expect(rfdA.params.hotspots).toBe('A54, A56, A66, A115');
    expect(String(rfdA.params.contig)).toBe('A17-132,/0,70-70');

    // Case B: update_params { benchmarkId: 'pdl1_5o45' }
    const editedByBmId = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [{ action: 'update_params', nodeId: 'pdb', params: { benchmarkId: 'pdl1_5o45' } }],
        summary: 'Switch benchmarkId to pdl1_5o45',
      },
      baseGraph,
    ).graph;

    const pdbB = editedByBmId.nodes.find((n) => n.id === 'pdb')!;
    const rfdB = editedByBmId.nodes.find((n) => n.id === 'rfd')!;
    expect(pdbB.params.mode).toBe('pdb-id');
    expect(pdbB.params.pdbId).toBe('5O45');
    expect(pdbB.params.pdbText).toBeUndefined();
    expect(pdbB.params.fetchedPdbId).toBeUndefined();
    expect(pdbB.params.benchmarkId).toBe('pdl1_5o45');
    expect(pdbB.params.cropSegments).toBe('A17-132');
    expect(rfdB.params.mode).toBe('binder');
    expect(rfdB.params.motifSpan).toBeUndefined();
    expect(rfdB.params.hotspots).toBe('A54, A56, A66, A115');
    expect(String(rfdB.params.contig)).toBe('A17-132,/0,70-70');

    // Case C: update_params { mode: 'demo-bundle' } clears stale pdbText/fetchedPdbId/benchmarkId/cropSegments/pdbId
    const editedToDemo = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [{ action: 'update_params', nodeId: 'pdb', params: { mode: 'demo-bundle' } }],
        summary: 'Switch 1YCR to demo-bundle',
      },
      baseGraph,
    ).graph;

    const pdbDemo = editedToDemo.nodes.find((n) => n.id === 'pdb')!;
    const rfdDemo = editedToDemo.nodes.find((n) => n.id === 'rfd')!;
    expect(pdbDemo.params.mode).toBe('demo-bundle');
    expect(pdbDemo.params.pdbId).toBeUndefined();
    expect(pdbDemo.params.pdbText).toBeUndefined();
    expect(pdbDemo.params.fetchedPdbId).toBeUndefined();
    expect(pdbDemo.params.benchmarkId).toBeUndefined();
    expect(pdbDemo.params.cropSegments).toBeUndefined();
    expect(rfdDemo.params.motifSpan).toBeUndefined();
    expect(rfdDemo.params.contig).toBeUndefined();
  });

  it('M2-R3 Finding 3: 1N8Z clears hallucinated targetChain H without explicit chain H, comma-separated CDR1, CDR2, CDR3 narrows to CDR3, and antifold mode=esmif1 skips AntiFold CDR/chain injection', () => {
    // 1. 1N8Z with hallucinated targetChain: 'H' (no explicit "chain H") clears to '' and comma-separated CDR1, CDR2, CDR3 narrows to CDR3
    const bg1n8z = compileWorkflowPlanToGraph(
      reconcilePlannedWorkflow(
        {
          kind: 'block_graph',
          mode: 'new',
          nodes: [
            { id: 'pdb', adapter: 'pdb-input', label: 'PDB', params: { pdbId: '1N8Z' } },
            {
              id: 'af',
              adapter: 'antifold-esmif1',
              label: 'AntiFold',
              params: { variant: 'antifold', targetChain: 'H', regions: 'CDR1, CDR2, CDR3' },
            },
          ],
          edges: [['pdb', 'af']],
          summary: '1N8Z CDR3',
        },
        'Design only CDR3 of antibody 1N8Z with AntiFold',
      ),
    ).graph;
    const afBg = bg1n8z.nodes.find((n) => n.id === 'af')!;
    expect(afBg.params.targetChain).toBe('');
    expect(afBg.params.regions).toBe('CDR3');

    const pdb1n8z = reconcilePlannedWorkflow(
      {
        kind: 'pdb',
        pdbId: '1N8Z',
        crop: null,
        tasks: ['antifold'],
        targetChain: 'H',
        regions: 'CDR1, CDR2, CDR3',
        numSequences: 2,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: '1N8Z CDR3',
      } satisfies PdbWorkflowPlan,
      'Design only CDR3 of antibody 1N8Z with AntiFold',
    ) as PdbWorkflowPlan;
    expect(pdb1n8z.targetChain).toBe('');
    expect(pdb1n8z.regions).toBe('CDR3');

    // 2. 6XEY with comma-separated default 'CDR1, CDR2, CDR3' also narrows to 'CDR3'
    const bg6xey = compileWorkflowPlanToGraph(
      reconcilePlannedWorkflow(
        {
          kind: 'block_graph',
          mode: 'new',
          nodes: [
            { id: 'pdb', adapter: 'pdb-input', label: 'PDB', params: { pdbId: '6XEY' } },
            {
              id: 'af',
              adapter: 'antifold-esmif1',
              label: 'AntiFold',
              params: { variant: 'antifold', regions: 'CDR1, CDR2, CDR3' },
            },
          ],
          edges: [['pdb', 'af']],
          summary: '6XEY CDR3',
        },
        'Design only CDR3 of antibody 6XEY with AntiFold',
      ),
    ).graph;
    const af6xey = bg6xey.nodes.find((n) => n.id === 'af')!;
    expect(af6xey.params.targetChain).toBe('H');
    expect(af6xey.params.regions).toBe('CDR3');

    // 3. antifold-esmif1 with mode: 'esmif1' (or model: 'esmif1') before variant alias mapping does not receive targetChain: 'H' or regions: 'CDR3'
    for (const aliasParams of [{ mode: 'esmif1' }, { model: 'esmif1' }] as Array<Record<string, unknown>>) {
      const bgEsmif = compileWorkflowPlanToGraph(
        reconcilePlannedWorkflow(
          {
            kind: 'block_graph',
            mode: 'new',
            nodes: [
              { id: 'pdb', adapter: 'pdb-input', label: 'PDB', params: { pdbId: '6XEY' } },
              { id: 'af', adapter: 'antifold-esmif1', label: 'ESM-IF1', params: aliasParams },
            ],
            edges: [['pdb', 'af']],
            summary: '6XEY ESM-IF1 alias',
          },
          'Design only CDR3 of antibody 6XEY with ESM-IF1',
        ),
      ).graph;
      const afEsm = bgEsmif.nodes.find((n) => n.id === 'af')!;
      expect(afEsm.params.variant).toBe('esmif1');
      expect(afEsm.params.targetChain).toBe('');
      expect(afEsm.params.regions).toBe('CDR1 CDR2 CDR3');
    }
  });

  it('M2-R4 Finding 1: update_params switching custom pdb-input (prevBm undefined) to 5O45/pdl1_5o45 or 1BCF aligns downstream generator mode and preserves explicit hotspots and non-motif/non-binder modes', () => {
    const customPdb = makePdb({ nRes: 20, chain: 'A' });
    const motifGraph: WorkflowGraph = {
      id: 'custom-motif-base',
      name: 'Custom Motif',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          params: { mode: 'custom', pdbText: customPdb, filename: 'custom.pdb', fetchedPdbId: 'CUSTOM' },
        },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          params: { mode: 'motif', length: 70, motifSpan: 'A5-15', contig: '25-25,A5-15,25-25' },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' }],
    };

    // Case A: custom (motif) -> { pdbId: '5O45' }
    const byPdbId = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [{ action: 'update_params', nodeId: 'pdb', params: { pdbId: '5O45' } }],
        summary: 'Switch custom PDB to 5O45',
      },
      motifGraph,
    ).graph;
    const pdbA = byPdbId.nodes.find((n) => n.id === 'pdb')!;
    const rfdA = byPdbId.nodes.find((n) => n.id === 'rfd')!;
    expect(pdbA.params.mode).toBe('pdb-id');
    expect(pdbA.params.pdbId).toBe('5O45');
    expect(pdbA.params.benchmarkId).toBe('pdl1_5o45');
    expect(pdbA.params.cropSegments).toBe('A17-132');
    expect(pdbA.params.pdbText).toBeUndefined();
    expect(pdbA.params.filename).toBeUndefined();
    expect(rfdA.params.mode).toBe('binder');
    expect(rfdA.params.hotspots).toBe('A54, A56, A66, A115');
    expect(String(rfdA.params.contig)).toBe('A17-132,/0,70-70');
    expect(rfdA.params.motifSpan).toBeUndefined();
    expect(rfdA.params.unindexedMotif).toBeUndefined();

    // Case B: custom (motif) -> { benchmarkId: 'pdl1_5o45' }
    const byBmId = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [{ action: 'update_params', nodeId: 'pdb', params: { benchmarkId: 'pdl1_5o45' } }],
        summary: 'Switch custom PDB to pdl1_5o45',
      },
      motifGraph,
    ).graph;
    const rfdB = byBmId.nodes.find((n) => n.id === 'rfd')!;
    expect(rfdB.params.mode).toBe('binder');
    expect(rfdB.params.hotspots).toBe('A54, A56, A66, A115');
    expect(String(rfdB.params.contig)).toBe('A17-132,/0,70-70');
    expect(rfdB.params.motifSpan).toBeUndefined();
    expect(rfdB.params.unindexedMotif).toBeUndefined();

    // Case C: custom (motif) -> { benchmarkId: 'pdl1_5o45' } with explicit hotspots: 'A115'
    const withExplicitHotspots = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [
          { action: 'update_params', nodeId: 'pdb', params: { benchmarkId: 'pdl1_5o45' } },
          { action: 'update_params', nodeId: 'rfd', params: { hotspots: 'A115' } },
        ],
        summary: 'Switch custom PDB to pdl1_5o45 with explicit hotspot A115',
      },
      motifGraph,
    ).graph;
    const rfdC = withExplicitHotspots.nodes.find((n) => n.id === 'rfd')!;
    expect(rfdC.params.mode).toBe('binder');
    expect(rfdC.params.hotspots).toBe('A115');
    expect(String(rfdC.params.contig)).toBe('A17-132,/0,70-70');
    expect(rfdC.params.motifSpan).toBeUndefined();
    expect(rfdC.params.unindexedMotif).toBeUndefined();

    // Case D: Reverse custom (binder) -> { pdbId: '1BCF' }
    const binderGraph: WorkflowGraph = {
      ...motifGraph,
      nodes: [
        motifGraph.nodes[0],
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          params: { mode: 'binder', length: 70, hotspots: 'A1, A2', contig: 'A1-3,/0,70-70' },
        },
      ],
    };
    const reverseTo1Bcf = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [{ action: 'update_params', nodeId: 'pdb', params: { pdbId: '1BCF' } }],
        summary: 'Switch custom binder PDB to 1BCF motif',
      },
      binderGraph,
    ).graph;
    const pdbD = reverseTo1Bcf.nodes.find((n) => n.id === 'pdb')!;
    const rfdD = reverseTo1Bcf.nodes.find((n) => n.id === 'rfd')!;
    expect(pdbD.params.benchmarkId).toBe('rfd_1bcf_diiron');
    expect(rfdD.params.mode).toBe('motif');
    expect(rfdD.params.hotspots).toBeUndefined();
    expect(rfdD.params.motifSpan).toBe('A18-25, A47-54, A92-99, A123-130');
    expect(String(rfdD.params.contig)).toContain('A18-25');
    expect(String(rfdD.params.contig)).not.toContain('/0');

    // Case E: Preserves non-motif/non-binder modes (e.g. 'partial') when prevBm is undefined
    const partialGraph: WorkflowGraph = {
      ...motifGraph,
      nodes: [
        motifGraph.nodes[0],
        { id: 'rfd', adapterId: 'rfd4-proteina', params: { mode: 'partial', length: 70 } },
      ],
    };
    const partialEdited = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        editOps: [{ action: 'update_params', nodeId: 'pdb', params: { pdbId: '5O45' } }],
        summary: 'Switch custom partial PDB to 5O45',
      },
      partialGraph,
    ).graph;
    expect(partialEdited.nodes.find((n) => n.id === 'rfd')!.params.mode).toBe('partial');
  });

  it('M2-R4 Finding 2: estimateContigProteinMetrics parses [PRDN]-suffixed bare and named expandable contig segments for WebGPU token ceiling checks', () => {
    const chainAPdb = makePdb({ nRes: 205, chain: 'A' });
    const makeSuffixedGraph = (contig: string, model = 'monomer', extra: Record<string, unknown> = {}): WorkflowGraph => ({
      id: `prdn-contig-${contig}`,
      name: 'PRDN contig ceiling',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: chainAPdb } },
        { id: 'rfd', adapterId: 'rfd4-proteina', params: { mode: 'motif', contig, length: 70, ...extra } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'solublempnn' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' },
        { fromNode: 'rfd', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    });

    // Both named (h1:200-200P) and bare (200-200P) suffixed segments sum to 550 > 512 (AF2 monomer ceiling)
    for (const contig of ['h1:200-200P,A1-200,150-150', '200-200P,A1-200,150-150']) {
      const issues = validateBioCapabilities(makeSuffixedGraph(contig));
      expect(issues.some((i) => i.code === 'webgpu-token-ceiling')).toBe(true);
    }

    // Control below 512 ceiling (100 + 200 + 150 = 450 <= 512) does not emit webgpu-token-ceiling
    const belowIssues = validateBioCapabilities(makeSuffixedGraph('h1:100-100P,A1-200,150-150'));
    expect(belowIssues.some((i) => i.code === 'webgpu-token-ceiling')).toBe(false);

    // Bare suffixed contig (370-370P) + unindexed motif (20 aa) = 390 > 384 (boltz2 ceiling)
    const bareUnindexedAbove = validateBioCapabilities(
      makeSuffixedGraph('370-370P', 'boltz2', { motifSpan: 'A1-20', unindexedMotif: true }),
    );
    expect(bareUnindexedAbove.some((i) => i.code === 'webgpu-token-ceiling')).toBe(true);
  });
});
