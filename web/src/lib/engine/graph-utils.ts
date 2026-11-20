// Typed DAG validation, port colors, default node factories, and auto-layout
// for the model-agnostic denovo.run node-graph canvas.

import { ADAPTER_REGISTRY } from './registry';
import type { PortType, WorkflowEdge, WorkflowGraph, WorkflowNode } from './types';

export const PORT_COLORS: Record<PortType, { bg: string; label: string; short: string }> = {
  StructureArtifact: { bg: '#3b82f6', label: 'Structure (3D)', short: 'Structure' },
  TrajectoryFrames: { bg: '#06b6d4', label: 'Trajectory', short: 'Trajectory' },
  SequenceSet: { bg: '#a855f7', label: 'Sequences (AA)', short: 'Sequences' },
  FoldedStructure: { bg: '#10b981', label: 'Refold (AF2)', short: 'Folds' },
  DesignabilityMetrics: { bg: '#f59e0b', label: 'Metrics (scRMSD)', short: 'Metrics' },
};

export const STAGE_COLORS: Record<string, string> = {
  source: '#64748b',
  generate: '#3b82f6',
  'inverse-fold': '#a855f7',
  fold: '#10b981',
  score: '#f59e0b',
};

export function defaultParamsForAdapter(adapterId: string): Record<string, unknown> {
  switch (adapterId) {
    case 'pdb-input':
      return { mode: 'demo-bundle', pdbText: '', filename: 'demo_3helix_53aa.pdb' };
    case 'sequence-input':
      return {
        label: 'custom-seq-1',
        sequence: 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALE',
      };
    case 'rfd4-proteina':
      return { length: 70, nsteps: 30, recycleSchedule: 'low', seed: 42, customSpecYaml: '' };
    case 'legacy-teacher':
      return {
        modelUrl: '/models/legacy-teacher/v1/model.onnx',
        length: 80,
        nfe: 32,
        sampler: 'sde-proteina',
        schedule: 'log',
        scheduleP: 2.0,
        seed: 42,
        sdeGamma: 0.1,
        sdeGtMax: 3.0,
        sdeGtMode: 'one-minus-t-over-t',
        sdeTLimOde: 0.99,
        sdeNoiseSeed: 7,
      };
    case 'web-protein-mpnn':
      return {
        family: 'solublempnn',
        modelName: 'v_48_020',
        numSequences: 2,
        temperature: 0.1,
        seed: 42,
        fixedPositions: '',
      };
    case 'localfold-af2':
      return { recycles: 1, maxSequences: 2, seed: 0 };
    case 'designability-scorer':
      return { scRmsdThreshold: 2.0, plddtThreshold: 70 };
    default:
      return {};
  }
}

export interface ConnectionCheck {
  valid: boolean;
  reason?: string;
}

/** Check if adding edge (fromNode -> toNode) would create a cycle in the graph. */
export function wouldCreateCycle(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  fromNode: string,
  toNode: string,
): boolean {
  if (fromNode === toNode) return true;
  // BFS/DFS starting from `toNode` following existing edges; if we can reach `fromNode`, cycle!
  const adj = new Map<string, string[]>();
  for (const n of nodes) adj.set(n.id, []);
  for (const e of edges) adj.get(e.fromNode)?.push(e.toNode);

  const visited = new Set<string>([toNode]);
  const stack = [toNode];
  while (stack.length > 0) {
    const cur = stack.pop()!;
    if (cur === fromNode) return true;
    for (const nxt of adj.get(cur) ?? []) {
      if (!visited.has(nxt)) {
        visited.add(nxt);
        stack.push(nxt);
      }
    }
  }
  return false;
}

/**
 * Validate whether connecting (fromNode:fromPort) -> (toNode:toPort) is allowed:
 * 1. Both nodes & declared ports must exist in ADAPTER_REGISTRY.
 * 2. Output port type must exactly match Input port type.
 * 3. Edge must not introduce a cycle.
 */
export function validateConnection(
  graph: WorkflowGraph,
  fromNode: string,
  fromPort: string,
  toNode: string,
  toPort: string,
): ConnectionCheck {
  if (fromNode === toNode) return { valid: false, reason: 'Cannot connect a node to itself' };
  const src = graph.nodes.find((n) => n.id === fromNode);
  const dst = graph.nodes.find((n) => n.id === toNode);
  if (!src || !dst) return { valid: false, reason: 'Missing source or target node' };

  const srcManifest = ADAPTER_REGISTRY[src.adapterId];
  const dstManifest = ADAPTER_REGISTRY[dst.adapterId];
  if (!srcManifest || !dstManifest) return { valid: false, reason: 'Unknown adapter' };

  const outSpec = srcManifest.outputs[fromPort];
  const inSpec = dstManifest.inputs[toPort];
  if (!outSpec) return { valid: false, reason: `Unknown output port "${fromPort}"` };
  if (!inSpec) return { valid: false, reason: `Unknown input port "${toPort}"` };

  if (outSpec.type !== inSpec.type) {
    return {
      valid: false,
      reason: `Type mismatch: ${outSpec.type} cannot wire into ${inSpec.type}`,
    };
  }
  if (wouldCreateCycle(graph.nodes, graph.edges, fromNode, toNode)) {
    return { valid: false, reason: 'Connection would introduce a cycle' };
  }
  return { valid: true };
}

/** Validate the full graph before execution (missing required inputs, cycles, unknown adapters). */
export function validateGraph(graph: WorkflowGraph): string[] {
  const errors: string[] = [];
  if (graph.nodes.length === 0) {
    errors.push('Canvas is empty — drag or add at least one block from the palette.');
    return errors;
  }
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  for (const node of graph.nodes) {
    const manifest = ADAPTER_REGISTRY[node.adapterId];
    if (!manifest) {
      errors.push(`Node "${node.id}" references unknown adapter "${node.adapterId}".`);
      continue;
    }
    for (const [portName, spec] of Object.entries(manifest.inputs)) {
      if (spec.required === false) continue;
      const connected = graph.edges.some(
        (e) => e.toNode === node.id && e.toPort === portName && byId.has(e.fromNode),
      );
      if (!connected) {
        errors.push(
          `"${node.label ?? manifest.shortName}" is missing required input "${portName}" (${spec.type}).`,
        );
      }
    }
  }
  return errors;
}

/** Compute clean left-to-right topological positions for nodes lacking coordinates. */
export function autoLayoutGraph(graph: WorkflowGraph): WorkflowGraph {
  const depth = new Map<string, number>();
  for (const n of graph.nodes) depth.set(n.id, 0);

  // Relax longest path from sources (handles branching DAGs cleanly)
  for (let iter = 0; iter < graph.nodes.length; iter++) {
    let changed = false;
    for (const e of graph.edges) {
      const dSrc = depth.get(e.fromNode) ?? 0;
      const dDst = depth.get(e.toNode) ?? 0;
      if (dSrc + 1 > dDst) {
        depth.set(e.toNode, dSrc + 1);
        changed = true;
      }
    }
    if (!changed) break;
  }

  const byCol = new Map<number, WorkflowNode[]>();
  for (const n of graph.nodes) {
    const col = depth.get(n.id) ?? 0;
    const list = byCol.get(col) ?? [];
    list.push(n);
    byCol.set(col, list);
  }

  const X_GAP = 290;
  const Y_GAP = 165;
  const posMap = new Map<string, { x: number; y: number }>();

  for (const [col, list] of byCol.entries()) {
    const totalH = (list.length - 1) * Y_GAP;
    list.forEach((n, idx) => {
      posMap.set(n.id, {
        x: 40 + col * X_GAP,
        y: 60 + idx * Y_GAP - totalH / 2 + 100,
      });
    });
  }

  return {
    ...graph,
    nodes: graph.nodes.map((n) => ({
      ...n,
      position: posMap.get(n.id) ?? { x: 40, y: 80 },
    })),
  };
}
