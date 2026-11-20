import { describe, it, expect } from 'vitest';
import {
  ADAPTER_REGISTRY,
  RFD4_BENCHMARK_OPTIONS,
  WORKFLOW_PRESETS,
} from '$lib/engine/registry';
import {
  PORT_COLORS,
  STAGE_COLORS,
  autoLayoutGraph,
  defaultParamsForAdapter,
  validateConnection,
  validateGraph,
  wouldCreateCycle,
} from '$lib/engine/graph-utils';
import type { WorkflowGraph } from '$lib/engine/types';

describe('engine/registry & engine/graph-utils', () => {
  it('all WORKFLOW_PRESETS pass validateGraph with 0 errors and have valid typed acyclic edges', () => {
    expect(WORKFLOW_PRESETS.length).toBeGreaterThanOrEqual(5);

    for (const preset of WORKFLOW_PRESETS) {
      const g = preset.graph;
      const errors = validateGraph(g);
      expect(errors, `Preset "${preset.id}" had validation errors: ${errors.join('; ')}`).toEqual([]);

      // No duplicate (toNode, toPort) target wires in any built-in preset
      const seenTargets = new Set<string>();
      for (const e of g.edges) {
        const key = `${e.toNode}.${e.toPort}`;
        expect(seenTargets.has(key), `Duplicate target port ${key} in preset ${preset.id}`).toBe(false);
        seenTargets.add(key);

        // Removing edge e and validating re-connection must succeed
        const withoutE: WorkflowGraph = {
          ...g,
          edges: g.edges.filter((x) => x !== e),
        };
        const check = validateConnection(withoutE, e.fromNode, e.fromPort, e.toNode, e.toPort);
        expect(check.valid, `Invalid edge in preset ${preset.id}: ${check.reason}`).toBe(true);
      }
    }
  });

  it('every ADAPTER_REGISTRY entry has valid ports, stage colors, and default params', () => {
    for (const [id, manifest] of Object.entries(ADAPTER_REGISTRY)) {
      expect(manifest.id).toBe(id);
      expect(STAGE_COLORS[manifest.stage]).toBeDefined();
      for (const spec of [...Object.values(manifest.inputs), ...Object.values(manifest.outputs)]) {
        expect(PORT_COLORS[spec.type]).toBeDefined();
      }
      const params = defaultParamsForAdapter(id);
      expect(typeof params).toBe('object');
      expect(Object.keys(params).length).toBeGreaterThan(0);
    }
    expect(RFD4_BENCHMARK_OPTIONS.length).toBeGreaterThan(0);
  });

  it('validateGraph catches empty graphs, unknown adapters, and missing required inputs', () => {
    expect(validateGraph({ id: 'empty', name: 'Empty', nodes: [], edges: [] }).length).toBe(1);

    const badAdapter: WorkflowGraph = {
      id: 'bad',
      name: 'Bad',
      nodes: [{ id: 'n1', adapterId: 'does-not-exist', params: {} }],
      edges: [],
    };
    expect(validateGraph(badAdapter)[0]).toContain('unknown adapter');

    const missingInputs: WorkflowGraph = {
      id: 'missing',
      name: 'Missing',
      nodes: [
        { id: 'score', adapterId: 'designability-scorer', params: {} },
      ],
      edges: [],
    };
    const errs = validateGraph(missingInputs);
    expect(errs.length).toBe(2); // missing structure + folds

    // Optional inputs (rfd4-proteina template & condition) are NOT required
    const optionalOnly: WorkflowGraph = {
      id: 'opt',
      name: 'Opt',
      nodes: [{ id: 'gen', adapterId: 'rfd4-proteina', params: defaultParamsForAdapter('rfd4-proteina') }],
      edges: [],
    };
    expect(validateGraph(optionalOnly)).toEqual([]);
  });

  it('validateConnection enforces exact typed port compatibility and rejects unknown ports/nodes', () => {
    const graph: WorkflowGraph = {
      id: 'types',
      name: 'Types',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: {} },
        { id: 'cond', adapterId: 'rfd4-conditioner', params: {} },
        { id: 'rfd4', adapterId: 'rfd4-proteina', params: {} },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: {} },
        { id: 'fold', adapterId: 'localfold-af2', params: {} },
        { id: 'score', adapterId: 'designability-scorer', params: {} },
      ],
      edges: [],
    };

    // Valid compatible connections
    expect(validateConnection(graph, 'src', 'structure', 'mpnn', 'structure').valid).toBe(true);
    expect(validateConnection(graph, 'cond', 'condition', 'rfd4', 'condition').valid).toBe(true);
    expect(validateConnection(graph, 'rfd4', 'sequences', 'fold', 'sequences').valid).toBe(true);
    expect(validateConnection(graph, 'fold', 'folds', 'score', 'folds').valid).toBe(true);

    // Type mismatches
    const mismatch1 = validateConnection(graph, 'src', 'structure', 'fold', 'sequences');
    expect(mismatch1.valid).toBe(false);
    expect(mismatch1.reason).toContain('Type mismatch');

    const mismatch2 = validateConnection(graph, 'cond', 'condition', 'mpnn', 'structure');
    expect(mismatch2.valid).toBe(false);
    expect(mismatch2.reason).toContain('Type mismatch');

    const mismatch3 = validateConnection(graph, 'mpnn', 'sequences', 'score', 'folds');
    expect(mismatch3.valid).toBe(false);
    expect(mismatch3.reason).toContain('Type mismatch');

    // Missing node / unknown port
    expect(validateConnection(graph, 'ghost', 'structure', 'mpnn', 'structure').valid).toBe(false);
    expect(validateConnection(graph, 'src', 'nonexistent', 'mpnn', 'structure').valid).toBe(false);
    expect(validateConnection(graph, 'src', 'structure', 'mpnn', 'nonexistent').valid).toBe(false);
  });

  it('rejects self-loops, 2-node cycles, and transitive cycles while allowing diamonds', () => {
    const nodes = [
      { id: 'a', adapterId: 'rfd4-proteina', params: {} },
      { id: 'b', adapterId: 'rfd4-proteina', params: {} },
      { id: 'c', adapterId: 'rfd4-proteina', params: {} },
      { id: 'd', adapterId: 'designability-scorer', params: {} },
    ];
    const edges = [
      { fromNode: 'a', fromPort: 'structure', toNode: 'b', toPort: 'template' },
      { fromNode: 'b', fromPort: 'structure', toNode: 'c', toPort: 'template' },
    ];
    const g: WorkflowGraph = { id: 'cyc', name: 'Cycle', nodes, edges };

    // Self-loop
    expect(wouldCreateCycle(nodes, edges, 'a', 'a')).toBe(true);
    expect(validateConnection(g, 'a', 'structure', 'a', 'template').valid).toBe(false);

    // 2-node cycle b -> a
    expect(wouldCreateCycle(nodes, edges, 'b', 'a')).toBe(true);
    const cyc2 = validateConnection(g, 'b', 'structure', 'a', 'template');
    expect(cyc2.valid).toBe(false);
    expect(cyc2.reason).toContain('cycle');

    // Transitive 3-node cycle c -> a
    expect(wouldCreateCycle(nodes, edges, 'c', 'a')).toBe(true);
    const cyc3 = validateConnection(g, 'c', 'structure', 'a', 'template');
    expect(cyc3.valid).toBe(false);
    expect(cyc3.reason).toContain('cycle');

    // Diamond (a -> c while a -> b -> c exists) is NOT a cycle
    expect(wouldCreateCycle(nodes, edges, 'a', 'c')).toBe(false);
    expect(validateConnection(g, 'a', 'structure', 'c', 'template').valid).toBe(true);
  });

  it('autoLayoutGraph is deterministic, left-to-right monotonic, and non-overlapping for all presets', () => {
    expect(autoLayoutGraph({ id: 'e', name: 'e', nodes: [], edges: [] }).nodes).toEqual([]);

    for (const preset of WORKFLOW_PRESETS) {
      const stripped: WorkflowGraph = {
        ...structuredClone(preset.graph),
        nodes: preset.graph.nodes.map((n) => ({ ...n, position: undefined })),
      };
      const laid1 = autoLayoutGraph(stripped);
      const laid2 = autoLayoutGraph(stripped);

      // Exact determinism
      expect(laid1.nodes.map((n) => n.position)).toEqual(laid2.nodes.map((n) => n.position));

      const posById = new Map(laid1.nodes.map((n) => [n.id, n.position!]));

      // Every node has finite non-negative coordinates
      for (const n of laid1.nodes) {
        expect(n.position).toBeDefined();
        expect(Number.isFinite(n.position!.x)).toBe(true);
        expect(Number.isFinite(n.position!.y)).toBe(true);
        expect(n.position!.x).toBeGreaterThanOrEqual(0);
        expect(n.position!.y).toBeGreaterThanOrEqual(0);
      }

      // Every edge flows strictly left-to-right (fromNode.x < toNode.x)
      for (const e of laid1.edges) {
        const pFrom = posById.get(e.fromNode)!;
        const pTo = posById.get(e.toNode)!;
        expect(pFrom.x, `Edge ${e.fromNode}->${e.toNode} in ${preset.id}`).toBeLessThan(pTo.x);
      }

      // Pairwise non-overlapping bounding boxes (node cards are ~220x122 px)
      const CARD_W = 220;
      const CARD_H = 122;
      for (let i = 0; i < laid1.nodes.length; i++) {
        for (let j = i + 1; j < laid1.nodes.length; j++) {
          const a = laid1.nodes[i].position!;
          const b = laid1.nodes[j].position!;
          const overlapX = Math.abs(a.x - b.x) < CARD_W;
          const overlapY = Math.abs(a.y - b.y) < CARD_H;
          expect(
            overlapX && overlapY,
            `Overlap in preset ${preset.id} between ${laid1.nodes[i].id} (${a.x},${a.y}) and ${laid1.nodes[j].id} (${b.x},${b.y})`,
          ).toBe(false);
        }
      }
    }
  });
});
