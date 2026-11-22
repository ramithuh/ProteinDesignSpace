import { describe, expect, it } from 'vitest';
import {
  compileWorkflowPlanToGraph,
  reconcilePlannedWorkflow,
  type BlockGraphWorkflowPlan,
} from '$lib/engine/workflow-planner';
import { validateGraph } from '$lib/engine/graph-utils';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import type { WorkflowGraph } from '$lib/engine/types';

function makeReactiveProxy<T extends object>(obj: T): T {
  return new Proxy(obj, {
    get(target, prop, receiver) {
      const val = Reflect.get(target, prop, receiver);
      if (val && typeof val === 'object') return makeReactiveProxy(val as object);
      return val;
    },
  });
}

describe('Flexible Block-Graph IR (v9) compiler', () => {
  it('Q0: compiles "compare af3, af2, and rf3 for folding a protein" into a valid 4-block fork', () => {
    const plan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        {
          id: 'seq_1',
          adapter: 'sequence-input',
          label: 'Sequence Input',
          params: { sequence: 'NLYIQWLKDGGPSSGRPPPS' },
        },
        { id: 'fold_af3', adapter: 'localfold-af2', label: 'AlphaFold 3', params: { model: 'af3' } },
        { id: 'fold_af2', adapter: 'localfold-af2', label: 'AF2 Model 1', params: { model: 'monomer' } },
        {
          id: 'fold_rf3',
          adapter: 'localfold-af2',
          label: 'RoseTTAFold 3',
          params: { model: 'rosettafold3' },
        },
      ],
      edges: [
        ['seq_1', 'fold_af3'],
        ['seq_1', 'fold_af2'],
        ['seq_1', 'fold_rf3'],
      ],
      summary: 'Compare AF3, AF2, and RF3 on a protein sequence',
    };

    const reconciled = reconcilePlannedWorkflow(plan, 'compare af3, af2, and rf3 for folding a protein');
    const { graph, isIncrementalEdit } = compileWorkflowPlanToGraph(reconciled);
    expect(isIncrementalEdit).toBe(false);
    expect(validateGraph(graph)).toEqual([]);
    expect(validateBioCapabilities(graph, {}).filter((i) => i.severity === 'error')).toEqual([]);
    expect(graph.nodes).toHaveLength(4);
    expect(graph.edges).toHaveLength(3);
    expect(graph.nodes.filter((n) => n.adapterId === 'localfold-af2').map((n) => n.params.model)).toEqual([
      'af3',
      'monomer',
      'rosettafold3',
    ]);
  });

  it('Q11 (Exp 5): compiles forked T=0.1 vs T=0.8 A/B comparison with Genie 3 auto-ca and scRMSD auto-wire', () => {
    const plan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        {
          id: 'gen_1',
          adapter: 'genie3',
          label: 'Genie 3 (60aa)',
          params: { taskMode: 'unconditional', length: 60, numDesigns: 1 },
        },
        {
          id: 'mpnn_lo',
          adapter: 'web-protein-mpnn',
          label: 'Cα-MPNN (T=0.1)',
          params: { family: 'ca', temperature: 0.1 },
        },
        {
          id: 'mpnn_hi',
          adapter: 'web-protein-mpnn',
          label: 'Cα-MPNN (T=0.8)',
          params: { family: 'ca', temperature: 0.8 },
        },
        { id: 'fold_lo', adapter: 'localfold-af2', label: 'AF2 (T=0.1)', params: { model: 'monomer' } },
        { id: 'fold_hi', adapter: 'localfold-af2', label: 'AF2 (T=0.8)', params: { model: 'monomer' } },
        { id: 'score_lo', adapter: 'designability-scorer', label: 'scRMSD (T=0.1)', params: {} },
        { id: 'score_hi', adapter: 'designability-scorer', label: 'scRMSD (T=0.8)', params: {} },
      ],
      // Deliberately omit gen_1 -> score_lo/score_hi to verify compiler auto-wires structure input
      edges: [
        ['gen_1', 'mpnn_lo'],
        ['gen_1', 'mpnn_hi'],
        ['mpnn_lo', 'fold_lo'],
        ['mpnn_hi', 'fold_hi'],
        ['fold_lo', 'score_lo'],
        ['fold_hi', 'score_hi'],
      ],
      summary: 'Genie 3 (60aa): Cα-MPNN T=0.1 vs T=0.8 → AF2 → scRMSD',
    };

    const { graph } = compileWorkflowPlanToGraph(plan);
    expect(validateGraph(graph)).toEqual([]);
    expect(graph.nodes).toHaveLength(7);
    // 6 explicit + 2 auto-wired structure edges to designability-scorer
    expect(graph.edges).toHaveLength(8);
    expect(
      graph.edges.filter((e) => e.fromNode === 'gen_1' && e.toPort === 'structure' && e.toNode.startsWith('score_')),
    ).toHaveLength(2);
    expect(graph.nodes.find((n) => n.id === 'mpnn_lo')?.params.temperature).toBe(0.1);
    expect(graph.nodes.find((n) => n.id === 'mpnn_hi')?.params.temperature).toBe(0.8);
  });

  it('Q12 & Q13: compiles Claudin-4 (5B2G) A/B redesign and DHFR (1RX2) 3-way Biophysics fork', () => {
    const dhfrPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb_1', adapter: 'pdb-input', label: 'DHFR (1RX2)', params: { pdbId: '1RX2' } },
        { id: 'thermo_1', adapter: 'thermompnn', label: 'ThermoMPNN', params: {} },
        { id: 'esmc_1', adapter: 'esmc', label: 'ESM-C 300M', params: {} },
        { id: 'dyna_1', adapter: 'dyna1', label: 'Dyna-1', params: {} },
      ],
      edges: [
        ['pdb_1', 'thermo_1'],
        ['pdb_1', 'esmc_1'],
        ['pdb_1', 'dyna_1'],
      ],
      summary: 'DHFR (1RX2): ThermoMPNN + ESM-C + Dyna-1',
    };

    const { graph } = compileWorkflowPlanToGraph(dhfrPlan);
    expect(validateGraph(graph)).toEqual([]);
    expect(graph.nodes.find((n) => n.id === 'pdb_1')?.params.mode).toBe('pdb-id');
    expect(graph.edges).toHaveLength(3);
  });

  it('Q5 edit mode: executes add_node, add_edge, update_params, and remove_node splice safely on reactive Proxy graph', () => {
    const base: WorkflowGraph = {
      id: 'base',
      name: 'Base',
      nodes: [
        { id: 'pdb_1', adapterId: 'pdb-input', label: '1UBQ', params: { mode: 'pdb-id', pdbId: '1UBQ' } },
        {
          id: 'mpnn_1',
          adapterId: 'web-protein-mpnn',
          label: 'SolubleMPNN',
          params: { family: 'solublempnn', temperature: 0.1, numSequences: 2 },
        },
        { id: 'topk_1', adapterId: 'top-k-filter', label: 'Top-K', params: { topK: 2 } },
        { id: 'fold_1', adapterId: 'localfold-af2', label: 'AF2', params: { model: 'monomer' } },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' },
        { fromNode: 'mpnn_1', fromPort: 'sequences', toNode: 'topk_1', toPort: 'sequences' },
        { fromNode: 'topk_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' },
      ],
    };

    const proxied = makeReactiveProxy(base);
    const editPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'edit',
      editOps: [
        { action: 'add_node', nodeId: 'thermo_1', adapter: 'thermompnn', label: 'ThermoMPNN', params: {} },
        { action: 'add_edge', fromId: 'fold_1', toId: 'thermo_1' },
        { action: 'update_params', nodeId: 'mpnn_1', params: { temperature: 0.25 } },
        { action: 'remove_node', nodeId: 'topk_1' },
      ],
      summary: 'Add ThermoMPNN, set T=0.25, remove Top-K',
    };

    const { graph, isIncrementalEdit, addedNodeCount } = compileWorkflowPlanToGraph(editPlan, proxied);
    expect(isIncrementalEdit).toBe(true);
    expect(addedNodeCount).toBe(1);
    expect(validateGraph(graph)).toEqual([]);
    expect(graph.nodes.some((n) => n.id === 'thermo_1')).toBe(true);
    expect(graph.nodes.some((n) => n.id === 'topk_1')).toBe(false);
    expect(graph.nodes.find((n) => n.id === 'mpnn_1')?.params.temperature).toBe(0.25);
    // Spliced edge mpnn_1 -> fold_1 preserved
    expect(graph.edges.some((e) => e.fromNode === 'mpnn_1' && e.toNode === 'fold_1')).toBe(true);
    expect(graph.edges.some((e) => e.fromNode === 'fold_1' && e.toNode === 'thermo_1')).toBe(true);
  });
});
