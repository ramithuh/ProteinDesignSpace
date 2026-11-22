import { describe, expect, it } from 'vitest';
import { validateGraph } from '$lib/engine/graph-utils';
import { encodeGraphToHash, decodeHashToGraph } from '$lib/engine/share';
import { compileWorkflowPlanToGraph, type BlockGraphWorkflowPlan } from '$lib/engine/workflow-planner';
import { COMPARE_PALETTE } from '$lib/engine/compare';
import type { WorkflowGraph } from '$lib/engine/types';

describe('3D Superpose / Compare (structure-compare) block & multi-wire fan-in', () => {
  it('supports up to 6 incoming wires on structure-compare.structures and round-trips via #g= share link', async () => {
    const graph: WorkflowGraph = {
      id: 'fold-cmp-test',
      name: 'Compare AF3, AF2, and ESMFold2',
      nodes: [
        { id: 'seq_1', adapterId: 'sequence-input', label: 'Seq', params: { sequence: 'NLYIQWLKDGGPSSGRPPPS' } },
        { id: 'fold_af3', adapterId: 'localfold-af2', label: 'AlphaFold 3', params: { model: 'af3' } },
        { id: 'fold_af2', adapterId: 'localfold-af2', label: 'AF2 Model', params: { model: 'monomer' } },
        { id: 'fold_esm', adapterId: 'localfold-af2', label: 'ESMFold2 300M', params: { model: 'ef2-fast-300m' } },
        {
          id: 'cmp_1',
          adapterId: 'structure-compare',
          label: '3D Compare',
          params: { layout: 'overlay', colorMode: 'by-model', alignKabsch: true },
        },
      ],
      edges: [
        { fromNode: 'seq_1', fromPort: 'sequences', toNode: 'fold_af3', toPort: 'sequences' },
        { fromNode: 'seq_1', fromPort: 'sequences', toNode: 'fold_af2', toPort: 'sequences' },
        { fromNode: 'seq_1', fromPort: 'sequences', toNode: 'fold_esm', toPort: 'sequences' },
        { fromNode: 'fold_af3', fromPort: 'structure', toNode: 'cmp_1', toPort: 'structures' },
        { fromNode: 'fold_af2', fromPort: 'structure', toNode: 'cmp_1', toPort: 'structures' },
        { fromNode: 'fold_esm', fromPort: 'structure', toNode: 'cmp_1', toPort: 'structures' },
      ],
    };

    expect(validateGraph(graph)).toEqual([]);
    expect(COMPARE_PALETTE.length).toBeGreaterThanOrEqual(6);

    const encoded = await encodeGraphToHash(graph);
    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    expect(validateGraph(decoded!)).toEqual([]);
    expect(decoded!.edges.filter((e) => e.toNode === 'cmp_1' && e.toPort === 'structures')).toHaveLength(3);
  });

  it('compiles v9 block_graph motif vs design superposition wiring pdb-input -> reference and genie3 -> structures', () => {
    const plan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        {
          id: 'pdb_1',
          adapter: 'pdb-input',
          label: '1LVM Motif',
          params: { mode: 'pdb-id', pdbId: '1LVM', cropSegments: 'A1-10' },
        },
        {
          id: 'gen_1',
          adapter: 'genie3',
          label: 'Genie 3 Redesign',
          params: { taskMode: 'motif', length: 100, motifSpan: '1-10' },
        },
        {
          id: 'cmp_1',
          adapter: 'structure-compare',
          label: 'Motif vs Design',
          params: { layout: 'overlay', colorMode: 'by-model', alignKabsch: true },
        },
      ],
      edges: [
        ['pdb_1', 'gen_1'],
        ['pdb_1', 'cmp_1'],
        ['gen_1', 'cmp_1'],
      ],
      summary: 'Superpose 1LVM motif on Genie 3 design',
    };

    const { graph } = compileWorkflowPlanToGraph(plan);
    expect(validateGraph(graph)).toEqual([]);
    expect(graph.edges.some((e) => e.fromNode === 'pdb_1' && e.toNode === 'cmp_1' && e.toPort === 'reference')).toBe(true);
    expect(graph.edges.some((e) => e.fromNode === 'gen_1' && e.toNode === 'cmp_1' && e.toPort === 'structures')).toBe(true);
  });
});
