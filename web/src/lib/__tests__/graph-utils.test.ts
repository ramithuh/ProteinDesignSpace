import { describe, it, expect } from 'vitest';
import {
  ADAPTER_REGISTRY,
  RFD4_BENCHMARK_OPTIONS,
  WORKFLOW_PRESETS,
} from '$lib/engine/registry';
import {
  PORT_COLORS,
  PORT_STEP,
  STAGE_COLORS,
  autoLayoutGraph,
  defaultParamsForAdapter,
  insertBlockSmart,
  snapNodePositionToPorts,
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

  it('validateGraph rejects duplicate node IDs, cycles, self-loops, port mismatches, unknown ports, and duplicate target ports', () => {
    // 1. Duplicate node IDs
    const dupIds: WorkflowGraph = {
      id: 'dup-ids',
      name: 'Dup IDs',
      nodes: [
        { id: 'n1', adapterId: 'pdb-input', params: {} },
        { id: 'n1', adapterId: 'pdb-input', params: {} },
      ],
      edges: [],
    };
    expect(validateGraph(dupIds).some((e) => /Duplicate node ID/i.test(e))).toBe(true);

    // 2. Cycle (a -> b -> a)
    const cyclic: WorkflowGraph = {
      id: 'cyc',
      name: 'Cyclic',
      nodes: [
        { id: 'a', adapterId: 'rfd4-proteina', params: {} },
        { id: 'b', adapterId: 'rfd4-proteina', params: {} },
      ],
      edges: [
        { fromNode: 'a', fromPort: 'structure', toNode: 'b', toPort: 'template' },
        { fromNode: 'b', fromPort: 'structure', toNode: 'a', toPort: 'template' },
      ],
    };
    expect(validateGraph(cyclic).some((e) => /cycle/i.test(e))).toBe(true);

    // 3. Self-loop
    const selfLoop: WorkflowGraph = {
      id: 'self',
      name: 'Self',
      nodes: [{ id: 'a', adapterId: 'rfd4-proteina', params: {} }],
      edges: [{ fromNode: 'a', fromPort: 'structure', toNode: 'a', toPort: 'template' }],
    };
    const selfErrs = validateGraph(selfLoop);
    expect(selfErrs.some((e) => /Self-loop/i.test(e))).toBe(true);

    // 4. Port type mismatch + unknown ports
    const badPorts: WorkflowGraph = {
      id: 'bad-ports',
      name: 'Bad Ports',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: {} },
        { id: 'fold', adapterId: 'localfold-af2', params: {} },
      ],
      edges: [
        { fromNode: 'src', fromPort: 'structure', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    expect(validateGraph(badPorts).some((e) => /Type mismatch/i.test(e))).toBe(true);

    const unknownPorts: WorkflowGraph = {
      id: 'unk-ports',
      name: 'Unknown Ports',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: {} },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: {} },
      ],
      edges: [
        { fromNode: 'src', fromPort: 'badOut', toNode: 'mpnn', toPort: 'badIn' },
      ],
    };
    const unkErrs = validateGraph(unknownPorts);
    expect(unkErrs.some((e) => /unknown output port/i.test(e))).toBe(true);
    expect(unkErrs.some((e) => /unknown input port/i.test(e))).toBe(true);

    // 5. Duplicate incoming edges to same target port
    const dupTarget: WorkflowGraph = {
      id: 'dup-target',
      name: 'Dup Target',
      nodes: [
        { id: 'src1', adapterId: 'pdb-input', params: {} },
        { id: 'src2', adapterId: 'pdb-input', params: {} },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: {} },
      ],
      edges: [
        { fromNode: 'src1', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'src2', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
      ],
    };
    expect(validateGraph(dupTarget).some((e) => /multiple incoming edges/i.test(e))).toBe(true);
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

  it('insertBlockSmart places at {60,60} on empty graph, respects dropPos, and nudges vertically on overlap', () => {
    const empty: WorkflowGraph = { id: 'g', name: 'G', nodes: [], edges: [] };
    const r1 = insertBlockSmart(empty, 'pdb-input');
    expect(r1.wiredEdge).toBeUndefined();
    expect(r1.graph.nodes).toHaveLength(1);
    expect(r1.graph.nodes[0].id).toBe(r1.nodeId);
    expect(r1.graph.nodes[0].position).toEqual({ x: 60, y: 60 });
    expect(r1.graph.nodes[0].params).toEqual(defaultParamsForAdapter('pdb-input'));
    expect(empty.nodes).toHaveLength(0); // pure, no mutation

    // Explicit dropPos
    const rDrop = insertBlockSmart(r1.graph, 'web-protein-mpnn', r1.nodeId, { x: 412.4, y: 198.7 });
    const droppedNode = rDrop.graph.nodes.find((n) => n.id === rDrop.nodeId)!;
    expect(droppedNode.position).toEqual({ x: 412, y: 199 });

    // 280px to the right of selectedNode + vertical nudge when occupied
    const r2 = insertBlockSmart(r1.graph, 'web-protein-mpnn', r1.nodeId);
    const n2 = r2.graph.nodes.find((n) => n.id === r2.nodeId)!;
    expect(n2.position).toEqual({ x: 340, y: 60 });

    // Insert another node with r1 still selected -> same x=340 is occupied by n2, so y must nudge vertically
    const r3 = insertBlockSmart(r2.graph, 'rfd4-conditioner', r1.nodeId);
    const n3 = r3.graph.nodes.find((n) => n.id === r3.nodeId)!;
    expect(n3.position!.x).toBe(340);
    expect(n3.position!.y).toBeGreaterThan(60);
    expect(Math.abs(n3.position!.y - n2.position!.y)).toBeGreaterThanOrEqual(130);
  });

  it('insertBlockSmart auto-wires downstream and upstream (source/conditioner) without overwriting occupied target ports', () => {
    const base: WorkflowGraph = {
      id: 'wire-test',
      name: 'Wire Test',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', label: 'PDB Input', params: {}, position: { x: 60, y: 60 } },
      ],
      edges: [],
    };

    // Downstream auto-wire: pdb-input -> web-protein-mpnn
    const rMpnn = insertBlockSmart(base, 'web-protein-mpnn', 'src');
    expect(rMpnn.wiredEdge).toEqual({
      fromNode: 'src',
      fromPort: 'structure',
      toNode: rMpnn.nodeId,
      toPort: 'structure',
    });
    expect(rMpnn.graph.edges).toHaveLength(1);

    // Inserting another pdb-input while mpnn is selected MUST NOT overwrite mpnn.structure (already occupied)
    const rSrcDup = insertBlockSmart(rMpnn.graph, 'pdb-input', rMpnn.nodeId);
    expect(rSrcDup.wiredEdge).toBeUndefined();
    expect(rSrcDup.graph.edges).toEqual(rMpnn.graph.edges);

    // Upstream auto-wire: inserting pdb-input when an UNWIRED mpnn is selected -> new pdb-input -> mpnn
    const unwiredMpnn: WorkflowGraph = {
      id: 'unwired',
      name: 'Unwired',
      nodes: [
        { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN', params: {}, position: { x: 300, y: 60 } },
      ],
      edges: [],
    };
    const rUpstreamSrc = insertBlockSmart(unwiredMpnn, 'pdb-input', 'mpnn');
    expect(rUpstreamSrc.wiredEdge).toEqual({
      fromNode: rUpstreamSrc.nodeId,
      fromPort: 'structure',
      toNode: 'mpnn',
      toPort: 'structure',
    });

    // Upstream conditioner auto-wire: inserting rfd4-conditioner when rfd4-proteina is selected
    const rfd4Only: WorkflowGraph = {
      id: 'rfd4-only',
      name: 'RFD4',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4-Proteina', params: {}, position: { x: 300, y: 60 } },
      ],
      edges: [],
    };
    const rCond = insertBlockSmart(rfd4Only, 'rfd4-conditioner', 'gen');
    expect(rCond.wiredEdge).toEqual({
      fromNode: rCond.nodeId,
      fromPort: 'condition',
      toNode: 'gen',
      toPort: 'condition',
    });

    // Inserting a second conditioner while gen is selected must NOT overwrite gen.condition
    const rCond2 = insertBlockSmart(rCond.graph, 'rfd4-conditioner', 'gen');
    expect(rCond2.wiredEdge).toBeUndefined();
    expect(rCond2.graph.edges).toHaveLength(1);
  });

  it('insertBlockSmart generates collision-free IDs and numbered labels', () => {
    const g: WorkflowGraph = {
      id: 'ids',
      name: 'IDs',
      nodes: [
        { id: 'web_protein_mpnn_1', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN', params: {}, position: { x: 60, y: 60 } },
        { id: 'web_protein_mpnn_2', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN #2', params: {}, position: { x: 60, y: 220 } },
      ],
      edges: [],
    };
    const r = insertBlockSmart(g, 'web-protein-mpnn');
    expect(r.nodeId).toBe('web_protein_mpnn_3');
    const added = r.graph.nodes.find((n) => n.id === r.nodeId)!;
    expect(added.label).toBe('ProteinMPNN #3');
    expect(new Set(r.graph.nodes.map((n) => n.id)).size).toBe(3);
  });

  it('snaps placed and dragged blocks so connected port handles align on straight horizontal lines', () => {
    // Classic 4-block Genie 3 pipeline: Genie 3 -> SolubleMPNN -> LocalFold -> scRMSD / scTM
    const start: WorkflowGraph = {
      id: 'straight-wires',
      name: 'Straight Wires',
      nodes: [
        { id: 'g3', adapterId: 'genie3', params: {}, position: { x: 40, y: 100 } },
      ],
      edges: [],
    };
    const rMpnn = insertBlockSmart(start, 'web-protein-mpnn', 'g3');
    const rFold = insertBlockSmart(rMpnn.graph, 'localfold-af2', rMpnn.nodeId);
    const rScore = insertBlockSmart(rFold.graph, 'designability-scorer', rFold.nodeId);

    const g3Pos = rScore.graph.nodes.find((n) => n.id === 'g3')!.position!;
    const mpnnPos = rScore.graph.nodes.find((n) => n.id === rMpnn.nodeId)!.position!;
    const foldPos = rScore.graph.nodes.find((n) => n.id === rFold.nodeId)!.position!;
    const scorePos = rScore.graph.nodes.find((n) => n.id === rScore.nodeId)!.position!;

    // Row 0 -> Row 0 stays at exact same Y
    expect(mpnnPos.y).toBe(g3Pos.y);
    expect(foldPos.y).toBe(mpnnPos.y);
    // LocalFold.folds (row 0) -> scRMSD.folds (row 1) shifts up by exact PORT_STEP (26px)
    expect(scorePos.y).toBe(foldPos.y - PORT_STEP);

    // Magnetic snap on manual drag within ±18px snaps straight back to foldPos.y - PORT_STEP
    const snapped = snapNodePositionToPorts(rScore.graph, rScore.nodeId, { x: 960, y: scorePos.y + 11 }, 18);
    expect(snapped).toEqual({ x: 960, y: scorePos.y });

    // Dragging far away (>18px) does not fight the user
    const free = snapNodePositionToPorts(rScore.graph, rScore.nodeId, { x: 960, y: scorePos.y + 60 }, 18);
    expect(free).toEqual({ x: 960, y: scorePos.y + 60 });
  });
});
