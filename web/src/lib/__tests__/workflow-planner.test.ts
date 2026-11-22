import { describe, expect, it } from 'vitest';
import {
  compileLocalStarterPlan,
  compileWorkflowPlanToGraph,
  DEFAULT_PLANNER_ENDPOINT,
  LOCAL_EXAMPLE_PLANS,
  parseRetryAfterSec,
  type ConditionedDesignWorkflowPlan,
  type DeNovoWorkflowPlan,
  type EditCurrentWorkflowPlan,
  type PdbWorkflowPlan,
  type WorkflowPlan,
} from '$lib/engine/workflow-planner';
import { validateGraph } from '$lib/engine/graph-utils';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';

describe('WorkflowPlan deterministic compiler', () => {
  it('compiles Di-Iron 4-Helix Bundle motif + Genie 3 & RFD4 + AF2/RF3/Protenix-v2 into a valid 16-block multi-oracle graph', () => {
    const plan: ConditionedDesignWorkflowPlan = {
      kind: 'conditioned_design',
      designTask: 'motif',
      benchmarkId: 'rfd_1bcf_diiron',
      pdbId: null,
      crop: null,
      generators: ['genie3', 'rfd4-proteina'],
      length: null,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer', 'rosettafold3', 'protenix2'],
      score: true,
      stabilityScan: false,
      summary: 'Di-Iron 4-Helix (1BCF): Genie 3 & RFD4 → AF2, RF3, & Protenix2 eval',
    };

    const { graph, notes } = compileWorkflowPlanToGraph(plan);
    expect(notes).toEqual([]);
    expect(validateGraph(graph)).toEqual([]);
    expect(validateBioCapabilities(graph, {}).filter((i) => i.severity === 'error')).toEqual([]);

    // 1 shared PDB (1BCF) + Genie3 + Cα-MPNN + RFD4 + 6 folds (3 per branch) + 6 scorers = 16 nodes
    expect(graph.nodes).toHaveLength(16);
    expect(graph.edges).toHaveLength(21);

    const pdb = graph.nodes.find((n) => n.adapterId === 'pdb-input')!;
    expect(pdb.params.pdbId).toBe('1BCF');
    expect(pdb.params.benchmarkId).toBe('rfd_1bcf_diiron');
    expect(String(pdb.params.cropSegments)).toContain('A18-25');

    const folds = graph.nodes.filter((n) => n.adapterId === 'localfold-af2');
    expect(folds).toHaveLength(6);
    expect(folds.map((f) => f.params.model).sort()).toEqual([
      'monomer',
      'monomer',
      'protenix2',
      'protenix2',
      'rosettafold3',
      'rosettafold3',
    ]);
  });

  it('auto-drops Genie 3 on all-atom enzyme/ligand benchmarks where supportsGenie3 is false', () => {
    const plan: ConditionedDesignWorkflowPlan = {
      kind: 'conditioned_design',
      designTask: 'enzyme-ligand',
      benchmarkId: 'ame_1ctt_cda',
      pdbId: null,
      crop: null,
      generators: ['genie3', 'rfd4-proteina'],
      length: null,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['rosettafold3'],
      score: true,
      stabilityScan: false,
      summary: 'AME CDA Zn enzyme scaffolding',
    };

    const { graph, notes } = compileWorkflowPlanToGraph(plan);
    expect(validateGraph(graph)).toEqual([]);
    expect(graph.nodes.some((n) => n.adapterId === 'genie3')).toBe(false);
    expect(graph.nodes.some((n) => n.adapterId === 'rfd4-proteina')).toBe(true);
    expect(notes.length).toBe(1);
  });

  it('compiles dual-generator de novo workflow with Top-K and ThermoMPNN stability scan', () => {
    const plan: DeNovoWorkflowPlan = {
      kind: 'de_novo',
      generators: ['rfd4-proteina', 'genie3'],
      length: 100,
      numDesigns: 2,
      numSequences: 4,
      topK: 2,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: true,
      summary: 'RFD4 & Genie 3 (100aa) → Top-2 → AF2 → ThermoMPNN',
    };

    const { graph } = compileWorkflowPlanToGraph(plan);
    expect(validateGraph(graph)).toEqual([]);
    expect(graph.nodes.filter((n) => n.adapterId === 'top-k-filter')).toHaveLength(2);
    expect(graph.nodes.filter((n) => n.adapterId === 'thermompnn')).toHaveLength(2);
    expect(graph.nodes.find((n) => n.adapterId === 'genie3')?.params.length).toBe(100);
    expect(graph.nodes.find((n) => n.adapterId === 'rfd4-proteina')?.params.length).toBe(100);
  });

  it('compiles PDB AntiFold CDR3 and holo LigandMPNN workflows cleanly', () => {
    const antiPlan: PdbWorkflowPlan = {
      kind: 'pdb',
      pdbId: '1FDL',
      crop: null,
      task: 'antifold',
      targetChain: 'H',
      regions: 'CDR3',
      numSequences: 3,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: false,
      summary: '1FDL H CDR3 → AntiFold',
    };
    const { graph: antiGraph } = compileWorkflowPlanToGraph(antiPlan);
    expect(validateGraph(antiGraph)).toEqual([]);
    const af = antiGraph.nodes.find((n) => n.adapterId === 'antifold-esmif1')!;
    expect(af.params.variant).toBe('antifold');
    expect(af.params.targetChain).toBe('H');
    expect(af.params.regions).toBe('CDR3');

    const ligPlan: PdbWorkflowPlan = {
      kind: 'pdb',
      pdbId: '3PTB',
      crop: 'A, BEN, CA',
      task: 'ligandmpnn',
      targetChain: null,
      regions: null,
      numSequences: 2,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: false,
      summary: '3PTB [A, BEN, CA] → LigandMPNN',
    };
    const { graph: ligGraph } = compileWorkflowPlanToGraph(ligPlan);
    expect(validateGraph(ligGraph)).toEqual([]);
    expect(ligGraph.nodes.find((n) => n.adapterId === 'pdb-input')?.params.cropSegments).toBe(
      'A, BEN, CA',
    );
  });

  it('compiles multi-task PDB comparison (LigandMPNN vs LASErMPNN) and incremental edit_current operations (ThermoMPNN, RF3, Top-K splice)', () => {
    const cmpPlan: PdbWorkflowPlan = {
      kind: 'pdb',
      pdbId: '3PTB',
      crop: 'A, BEN, CA',
      tasks: ['ligandmpnn', 'lasermpnn'],
      targetChain: null,
      regions: null,
      numSequences: 2,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: false,
      summary: '3PTB [A, BEN, CA]: Compare LigandMPNN vs LASErMPNN',
    };
    const { graph: cmpGraph } = compileWorkflowPlanToGraph(cmpPlan);
    expect(validateGraph(cmpGraph)).toEqual([]);
    expect(cmpGraph.nodes.filter((n) => n.adapterId === 'pdb-input')).toHaveLength(1);
    expect(cmpGraph.nodes.some((n) => n.adapterId === 'ligandmpnn')).toBe(true);
    expect(cmpGraph.nodes.some((n) => n.adapterId === 'lasermpnn')).toBe(true);

    // Start with a Genie 3 -> Cα-MPNN -> AF2 -> scRMSD graph and incrementally edit it
    const base = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['genie3'],
      length: 80,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      summary: 'Genie 3 base',
    }).graph;

    const foldNode = base.nodes.find((n) => n.adapterId === 'localfold-af2')!;
    const mpnnNode = base.nodes.find((n) => n.adapterId === 'web-protein-mpnn')!;

    const editPlan: EditCurrentWorkflowPlan = {
      kind: 'edit_current',
      operations: [
        {
          action: 'insert_top_k',
          attachToNodeId: foldNode.id,
          model: null,
          foldModels: [],
          topK: 2,
          numSequences: null,
          targetChain: null,
          regions: null,
          score: true,
        },
        {
          action: 'add_stability_scan',
          attachToNodeId: foldNode.id,
          model: null,
          foldModels: [],
          topK: null,
          numSequences: null,
          targetChain: null,
          regions: null,
          score: true,
        },
        {
          action: 'add_fold_oracles',
          attachToNodeId: mpnnNode.id,
          model: null,
          foldModels: ['rosettafold3'],
          topK: null,
          numSequences: null,
          targetChain: null,
          regions: null,
          score: true,
        },
      ],
      summary: 'Splice Top-2 before AF2, add ThermoMPNN, and add RF3 refold branch',
    };

    const edited = compileWorkflowPlanToGraph(editPlan, base);
    expect(edited.isIncrementalEdit).toBe(true);
    expect(edited.addedNodeCount).toBe(4); // Top-K + ThermoMPNN + RF3 + scRMSD
    expect(validateGraph(edited.graph)).toEqual([]);
    expect(edited.graph.nodes.some((n) => n.adapterId === 'top-k-filter')).toBe(true);
    expect(edited.graph.nodes.some((n) => n.adapterId === 'thermompnn')).toBe(true);
    expect(
      edited.graph.nodes.some(
        (n) => n.adapterId === 'localfold-af2' && n.params.model === 'rosettafold3',
      ),
    ).toBe(true);
    // Original MPNN -> AF2 direct wire must be spliced through Top-K
    expect(
      edited.graph.edges.some((e) => e.fromNode === mpnnNode.id && e.toNode === foldNode.id),
    ).toBe(false);
  });

  it('propagates unsupported model notices on partial matches and throws UnsupportedWorkflowError on fully unsupported plans', () => {
    const partial = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['genie3'],
      length: 100,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['rosettafold3'],
      score: true,
      stabilityScan: false,
      unsupported: ['Chroma', 'Chai-1'],
      summary: '100aa de novo: Genie 3 → RoseTTAFold 3 (partial match)',
    });
    expect(validateGraph(partial.graph)).toEqual([]);
    expect(partial.unsupported).toEqual(['Chroma', 'Chai-1']);

    expect(() =>
      compileWorkflowPlanToGraph({
        kind: 'unsupported',
        unsupported: ['BindCraft', 'Chai-1'],
        reason: "BindCraft and Chai-1 aren't available in denovo.run yet.",
        suggestions: ['Use RFD4-Proteina or Genie 3'],
        summary: 'BindCraft and Chai-1 are not in denovo.run',
      }),
    ).toThrow(/BindCraft and Chai-1/);
  });

  it('points DEFAULT_PLANNER_ENDPOINT to the custom domain planner.denovo.run and compiles all 4 local starter chips at 0 Neurons', () => {
    expect(DEFAULT_PLANNER_ENDPOINT).toBe('https://planner.denovo.run/api/plan');

    const base = compileLocalStarterPlan(
      LOCAL_EXAMPLE_PLANS.find((e) => e.id === 'denovo-100aa')!.plan,
    );
    expect(base.model).toBe('local-preset');
    expect(base.latencyMs).toBe(0);
    expect(validateGraph(base.graph)).toEqual([]);

    for (const ex of LOCAL_EXAMPLE_PLANS) {
      const compiled = compileLocalStarterPlan(ex.plan, ex.requiresExisting ? base.graph : undefined);
      expect(compiled.model).toBe('local-preset');
      expect(compiled.latencyMs).toBe(0);
      expect(validateGraph(compiled.graph)).toEqual([]);
    }
  });

  it('rejects unknown plan.kind and invalid PDB crop syntax, and parses Retry-After headers', () => {
    expect(() =>
      compileWorkflowPlanToGraph({ kind: 'mystery_plan' } as unknown as WorkflowPlan),
    ).toThrow(/Unsupported workflow plan kind/);

    expect(() =>
      compileWorkflowPlanToGraph({
        kind: 'pdb',
        pdbId: '3PTB',
        crop: 'A; <script>alert(1)</script>',
        task: 'ligandmpnn',
        targetChain: null,
        regions: null,
        numSequences: 2,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: 'Invalid crop test',
      }),
    ).toThrow(/invalid PDB crop/i);

    expect(parseRetryAfterSec('42')).toBe(42);
    expect(parseRetryAfterSec('0')).toBeNull();
    expect(parseRetryAfterSec(null)).toBeNull();
  });

  it('compiles RFD4 native seq vs ProteinMPNN side-by-side refolding off the same RFD4 backbone', () => {
    const { graph, notes } = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['rfd4-proteina'],
      length: 100,
      numDesigns: 1,
      sequenceDesigners: ['native', 'web-protein-mpnn'],
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      summary: '100aa RFD4-Proteina: compare RFD4 native seq vs ProteinMPNN → AF2 refold',
    });

    expect(notes).toEqual([]);
    expect(validateGraph(graph)).toEqual([]);
    // 1 RFD4 + 1 ProteinMPNN + 2 AF2 folds + 2 scRMSD scorers = 6 nodes
    expect(graph.nodes).toHaveLength(6);
    const rfd4 = graph.nodes.find((n) => n.adapterId === 'rfd4-proteina')!;
    const mpnn = graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn')!;
    const folds = graph.nodes.filter((n) => n.adapterId === 'localfold-af2');
    expect(folds).toHaveLength(2);
    // One fold directly wired to rfd4.sequences, one wired to ProteinMPNN.sequences
    expect(graph.edges.some((e) => e.fromNode === rfd4.id && e.fromPort === 'sequences')).toBe(true);
    expect(graph.edges.some((e) => e.fromNode === mpnn.id && e.fromPort === 'sequences')).toBe(true);
  });

  it('compiles ProtPardelle-1c de novo + ESM-C + Dyna-1 and PDB ESM-C / Dyna-1 evaluation graphs', () => {
    const { graph: deNovoGraph } = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['protpardelle'],
      length: 90,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      esmcScore: true,
      dynamicsScan: true,
      summary: '90aa ProtPardelle-1c → AF2, ESM-C 300M & Dyna-1',
    });
    expect(validateGraph(deNovoGraph)).toEqual([]);
    expect(deNovoGraph.nodes.some((n) => n.adapterId === 'protpardelle')).toBe(true);
    expect(deNovoGraph.nodes.some((n) => n.adapterId === 'esmc')).toBe(true);
    expect(deNovoGraph.nodes.some((n) => n.adapterId === 'dyna1')).toBe(true);

    const { graph: pdbGraph } = compileWorkflowPlanToGraph({
      kind: 'pdb',
      pdbId: '1UBQ',
      crop: null,
      tasks: ['esmc', 'dyna1'],
      targetChain: null,
      regions: null,
      numSequences: null,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: false,
      summary: '1UBQ → ESM-C 300M & Dyna-1 dynamics',
    });
    expect(validateGraph(pdbGraph)).toEqual([]);
    expect(pdbGraph.nodes.map((n) => n.adapterId).sort()).toEqual(['dyna1', 'esmc', 'pdb-input']);
  });
});
