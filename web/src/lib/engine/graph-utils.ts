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
  ConditionSpec: { bg: '#f43f5e', label: 'Conditioning (RFD4)', short: 'Condition' },
};

export const STAGE_COLORS: Record<string, string> = {
  source: '#64748b',
  condition: '#f43f5e',
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
      return {
        mode: 'unconditional',
        length: 70,
        nsteps: 60,
        recycleSchedule: 'low',
        earlyStop: true,
        seed: 42,
        presetId: 'custom_cath_monomer',
        partialFraction: 0.5,
        customSpecYaml: '',
      };
    case 'rfd4-conditioner':
      return {
        taskPreset: 'custom',
        contig: '70-70',
        lockChain: '',
        hotspots: '',
        motifSpan: '',
        unindexedMotif: false,
        hbondMode: 'none',
        hbondSelection: '',
        cathCode: '',
        paintBrush: 'hotspots',
        rawJsonOverride: '',
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
      return { model: 'monomer', recycles: 1, diffusionSteps: 25, maxSequences: 2, seed: 0 };
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

/** Validate the full graph before execution (unique IDs, valid typed edges, single edge per target port, required inputs, cycle-free). */
export function validateGraph(graph: WorkflowGraph): string[] {
  const errors: string[] = [];
  if (graph.nodes.length === 0) {
    errors.push('Canvas is empty — click + Add Block to place a block.');
    return errors;
  }

  const byId = new Map<string, WorkflowNode>();
  for (const node of graph.nodes) {
    if (byId.has(node.id)) {
      errors.push(`Duplicate node ID "${node.id}".`);
    } else {
      byId.set(node.id, node);
    }
    if (!ADAPTER_REGISTRY[node.adapterId]) {
      errors.push(`Node "${node.id}" references unknown adapter "${node.adapterId}".`);
    }
  }

  // Validate edges: endpoints, ports, matching PortType, single edge per target port, self-loops
  const targetPorts = new Set<string>();
  const validConnectedTargets = new Set<string>();
  const indeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const id of byId.keys()) {
    indeg.set(id, 0);
    adj.set(id, []);
  }

  for (const e of graph.edges) {
    const src = byId.get(e.fromNode);
    const dst = byId.get(e.toNode);
    if (!src) {
      errors.push(`Edge references missing source node "${e.fromNode}".`);
      continue;
    }
    if (!dst) {
      errors.push(`Edge references missing target node "${e.toNode}".`);
      continue;
    }
    if (e.fromNode === e.toNode) {
      errors.push(`Self-loop detected on node "${e.fromNode}".`);
    }

    const srcManifest = ADAPTER_REGISTRY[src.adapterId];
    const dstManifest = ADAPTER_REGISTRY[dst.adapterId];
    let edgePortsOk = true;
    if (srcManifest && dstManifest) {
      const outSpec = srcManifest.outputs[e.fromPort];
      const inSpec = dstManifest.inputs[e.toPort];
      if (!outSpec) {
        errors.push(`Node "${src.id}" has unknown output port "${e.fromPort}".`);
        edgePortsOk = false;
      }
      if (!inSpec) {
        errors.push(`Node "${dst.id}" has unknown input port "${e.toPort}".`);
        edgePortsOk = false;
      }
      if (outSpec && inSpec && outSpec.type !== inSpec.type) {
        errors.push(
          `Type mismatch on wire ${src.id}.${e.fromPort} -> ${dst.id}.${e.toPort} (${outSpec.type} vs ${inSpec.type}).`,
        );
        edgePortsOk = false;
      }
    }

    const targetKey = `${e.toNode}:${e.toPort}`;
    if (targetPorts.has(targetKey)) {
      errors.push(`Input port "${e.toNode}.${e.toPort}" has multiple incoming edges.`);
    } else {
      targetPorts.add(targetKey);
    }
    if (edgePortsOk) {
      validConnectedTargets.add(targetKey);
    }

    adj.get(e.fromNode)!.push(e.toNode);
    indeg.set(e.toNode, (indeg.get(e.toNode) ?? 0) + 1);
  }

  // Check required inputs
  for (const node of byId.values()) {
    const manifest = ADAPTER_REGISTRY[node.adapterId];
    if (!manifest) continue;
    for (const [portName, spec] of Object.entries(manifest.inputs)) {
      if (spec.required === false) continue;
      if (!validConnectedTargets.has(`${node.id}:${portName}`)) {
        errors.push(
          `"${node.label ?? manifest.shortName}" is missing required input "${portName}" (${spec.type}).`,
        );
      }
    }
  }

  // Cycle detection via Kahn's topological sort
  const queue: string[] = [];
  for (const [id, d] of indeg) {
    if (d === 0) queue.push(id);
  }
  let visited = 0;
  while (queue.length > 0) {
    const cur = queue.shift()!;
    visited++;
    for (const nxt of adj.get(cur) ?? []) {
      const nd = (indeg.get(nxt) ?? 1) - 1;
      indeg.set(nxt, nd);
      if (nd === 0) queue.push(nxt);
    }
  }
  if (visited !== byId.size) {
    errors.push('Workflow graph contains a cycle (must be a directed acyclic graph).');
  }

  return errors;
}

// Pixel geometry of ProteinNode.svelte (used for exact handle-to-handle alignment)
const NODE_HEIGHT_1PORT = 122;
const PORT0_Y = 70; // vertical offset of 1st port handle center from node top
const PORT_STEP = 24; // vertical spacing between consecutive port handles

/**
 * Compute clean, crossing-minimized, handle-aligned left-to-right layout:
 * 1. Longest-path column assignment.
 * 2. Barycenter + output-port-index ordering inside each column (no branch crossings).
 * 3. Clear corridor (`NODE_HEIGHT + 44px`) below long skip-edges (`colSpan >= 2`) so
 *    skip wires run as dead-straight horizontal tracks completely above intermediate cards.
 * 4. Exact port-Y alignment for 1-to-1 horizontal chains so wires stay perfectly straight.
 */
export function autoLayoutGraph(graph: WorkflowGraph): WorkflowGraph {
  if (graph.nodes.length === 0) return graph;

  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const depth = new Map<string, number>();
  for (const n of graph.nodes) depth.set(n.id, 0);

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
  let maxCol = 0;
  for (const n of graph.nodes) {
    const col = depth.get(n.id) ?? 0;
    if (col > maxCol) maxCol = col;
    const list = byCol.get(col) ?? [];
    list.push(n);
    byCol.set(col, list);
  }

  const portIdx = (nodeId: string, dir: 'inputs' | 'outputs', portName: string): number => {
    const node = byId.get(nodeId);
    if (!node) return 0;
    const keys = Object.keys(ADAPTER_REGISTRY[node.adapterId]?.[dir] ?? {});
    const idx = keys.indexOf(portName);
    return idx >= 0 ? idx : 0;
  };

  // Columns bridged by a long skip-edge (colSpan >= 2) need a clear top corridor
  const skipTargetCols = new Set<number>();
  const spannedCols = new Set<number>();
  for (const e of graph.edges) {
    const c1 = depth.get(e.fromNode) ?? 0;
    const c2 = depth.get(e.toNode) ?? 0;
    if (c2 - c1 >= 2) {
      skipTargetCols.add(c2);
      for (let c = c1 + 1; c < c2; c++) spannedCols.add(c);
    }
  }

  const X_START = 40;
  const X_GAP = 300;
  const TOP_Y = 24;
  const CORRIDOR_Y = TOP_Y + NODE_HEIGHT_1PORT + 48; // 194px — completely below top skip wire!
  const ROW_GAP = 188;

  const posMap = new Map<string, { x: number; y: number }>();

  for (let col = 0; col <= maxCol; col++) {
    const list = byCol.get(col);
    if (!list || list.length === 0) continue;

    // Sort multi-node columns by upstream handle Y so branches never cross
    if (col > 0 && list.length > 1) {
      const barycenter = (node: WorkflowNode): number => {
        const inc = graph.edges.filter((e) => e.toNode === node.id);
        if (inc.length === 0) return 0;
        let sum = 0;
        for (const e of inc) {
          const upY = posMap.get(e.fromNode)?.y ?? TOP_Y;
          const outI = portIdx(e.fromNode, 'outputs', e.fromPort);
          sum += upY + PORT0_Y + outI * PORT_STEP;
        }
        return sum / inc.length;
      };
      list.sort((a, b) => barycenter(a) - barycenter(b));
    }

    const x = X_START + col * X_GAP;

    if (list.length === 1) {
      const n = list[0];
      const inc = graph.edges.filter((e) => e.toNode === n.id);
      let y = TOP_Y;

      if (spannedCols.has(col)) {
        // Intermediate node sitting beneath a top skip-edge: drop cleanly into the lower lane
        const prevSingle = inc.find(
          (e) => (depth.get(e.fromNode) ?? 0) === col - 1 && spannedCols.has(col - 1),
        );
        if (prevSingle) {
          // Align exact handle Y with previous lower-lane node for a dead-straight wire!
          const upY = posMap.get(prevSingle.fromNode)?.y ?? CORRIDOR_Y;
          const outI = portIdx(prevSingle.fromNode, 'outputs', prevSingle.fromPort);
          const inI = portIdx(n.id, 'inputs', prevSingle.toPort);
          y = upY + (outI - inI) * PORT_STEP;
        } else {
          y = CORRIDOR_Y;
        }
      } else if (skipTargetCols.has(col)) {
        // Sink receiving a top skip-edge: align exact port Y with the skip-edge source!
        const skipEdge = inc.find((e) => col - (depth.get(e.fromNode) ?? 0) >= 2);
        if (skipEdge) {
          const upY = posMap.get(skipEdge.fromNode)?.y ?? TOP_Y;
          const outI = portIdx(skipEdge.fromNode, 'outputs', skipEdge.fromPort);
          const inI = portIdx(n.id, 'inputs', skipEdge.toPort);
          y = Math.max(TOP_Y, upY + (outI - inI) * PORT_STEP);
        }
      } else if (inc.length === 1) {
        // Pure 1-to-1 connection: align target handle exactly with source handle!
        const e = inc[0];
        const upY = posMap.get(e.fromNode)?.y ?? TOP_Y;
        const outI = portIdx(e.fromNode, 'outputs', e.fromPort);
        const inI = portIdx(n.id, 'inputs', e.toPort);
        y = Math.max(TOP_Y, upY + (outI - inI) * PORT_STEP);
      }

      posMap.set(n.id, { x, y: Math.round(y) });
    } else {
      // Multi-row column (branched DAG): place rows with generous non-overlapping spacing
      let cursorY = spannedCols.has(col) ? CORRIDOR_Y - 30 : TOP_Y;
      for (const n of list) {
        const inc = graph.edges.filter((e) => e.toNode === n.id);
        let idealY = cursorY;
        if (inc.length === 1) {
          const e = inc[0];
          const upY = posMap.get(e.fromNode)?.y;
          if (upY !== undefined) {
            const outI = portIdx(e.fromNode, 'outputs', e.fromPort);
            const inI = portIdx(n.id, 'inputs', e.toPort);
            idealY = Math.max(cursorY, upY + (outI - inI) * PORT_STEP);
          }
        }
        posMap.set(n.id, { x, y: Math.round(idealY) });
        cursorY = idealY + ROW_GAP;
      }
    }
  }

  return {
    ...graph,
    nodes: graph.nodes.map((n) => ({
      ...n,
      position: posMap.get(n.id) ?? { x: X_START, y: TOP_Y },
    })),
  };
}

export interface InsertBlockResult {
  graph: WorkflowGraph;
  nodeId: string;
  wiredEdge?: WorkflowEdge;
}

/**
 * Pure helper to insert a new block into a WorkflowGraph with:
 * - collision-free deterministic node ID + readable label + default params
 * - smart placement (dropPos if given, else 280px right of selectedNode with vertical nudge if occupied, else { x: 60, y: 60 })
 * - auto-wiring of the first validateConnection-valid port pair without overwriting an already-occupied target input port.
 */
export function insertBlockSmart(
  graph: WorkflowGraph,
  adapterId: string,
  selectedNodeId?: string | null,
  dropPos?: { x: number; y: number },
): InsertBlockResult {
  const manifest = ADAPTER_REGISTRY[adapterId];
  if (!manifest) {
    throw new Error(`Unknown adapter "${adapterId}"`);
  }

  // Collision-free ID
  const existingIds = new Set(graph.nodes.map((n) => n.id));
  const prefix = adapterId.replace(/[^a-z0-9]/gi, '_');
  let seq = 1;
  let nodeId = `${prefix}_${seq}`;
  while (existingIds.has(nodeId)) {
    seq++;
    nodeId = `${prefix}_${seq}`;
  }

  const sameCount = graph.nodes.filter((n) => n.adapterId === adapterId).length;
  const label = sameCount > 0 ? `${manifest.shortName} #${sameCount + 1}` : manifest.shortName;

  const selectedNode = selectedNodeId
    ? graph.nodes.find((n) => n.id === selectedNodeId)
    : undefined;

  const isOccupied = (px: number, py: number): boolean =>
    graph.nodes.some((n) => {
      if (!n.position) return false;
      return Math.abs(n.position.x - px) < 220 && Math.abs(n.position.y - py) < 130;
    });

  let position: { x: number; y: number };
  if (dropPos) {
    position = { x: Math.round(dropPos.x), y: Math.round(dropPos.y) };
  } else if (selectedNode) {
    const px = (selectedNode.position?.x ?? 60) + 280;
    let py = selectedNode.position?.y ?? 60;
    let guard = 0;
    while (isOccupied(px, py) && guard < 24) {
      py += 160;
      guard++;
    }
    position = { x: Math.round(px), y: Math.round(py) };
  } else {
    const px = 60;
    let py = 60;
    let guard = 0;
    while (isOccupied(px, py) && guard < 24) {
      py += 160;
      guard++;
    }
    position = { x: px, y: py };
  }

  const newNode: WorkflowNode = {
    id: nodeId,
    adapterId,
    label,
    params: defaultParamsForAdapter(adapterId),
    position,
  };

  const graphWithNode: WorkflowGraph = {
    ...graph,
    nodes: [...graph.nodes, newNode],
    edges: [...graph.edges],
  };

  let wiredEdge: WorkflowEdge | undefined;
  if (selectedNode) {
    const occupiedTargets = new Set(graph.edges.map((e) => `${e.toNode}:${e.toPort}`));

    const findFirstValidEdge = (fromId: string, toId: string): WorkflowEdge | undefined => {
      const fromNode = graphWithNode.nodes.find((n) => n.id === fromId);
      const toNode = graphWithNode.nodes.find((n) => n.id === toId);
      if (!fromNode || !toNode) return undefined;
      const fromManifest = ADAPTER_REGISTRY[fromNode.adapterId];
      const toManifest = ADAPTER_REGISTRY[toNode.adapterId];
      if (!fromManifest || !toManifest) return undefined;

      for (const fromPort of Object.keys(fromManifest.outputs)) {
        for (const toPort of Object.keys(toManifest.inputs)) {
          if (occupiedTargets.has(`${toId}:${toPort}`)) continue;
          if (validateConnection(graphWithNode, fromId, fromPort, toId, toPort).valid) {
            return { fromNode: fromId, fromPort, toNode: toId, toPort };
          }
        }
      }
      return undefined;
    };

    const selManifest = ADAPTER_REGISTRY[selectedNode.adapterId];
    const insertUpstream =
      manifest.stage === 'source' ||
      (manifest.stage === 'condition' && selManifest?.stage !== 'source');

    if (insertUpstream) {
      wiredEdge = findFirstValidEdge(nodeId, selectedNode.id);
    } else {
      wiredEdge = findFirstValidEdge(selectedNode.id, nodeId);
    }
  }

  const nextGraph: WorkflowGraph = wiredEdge
    ? { ...graphWithNode, edges: [...graphWithNode.edges, wiredEdge] }
    : graphWithNode;

  return {
    graph: nextGraph,
    nodeId,
    ...(wiredEdge ? { wiredEdge } : {}),
  };
}
