import { describe, expect, it } from 'vitest';
import {
  compileWorkflowPlanToGraph,
  type ConditionedDesignWorkflowPlan,
  type DeNovoWorkflowPlan,
  type PdbWorkflowPlan,
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
});
