// Typed DAG validation, port colors, default node factories, and auto-layout
// for the model-agnostic denovo.run node-graph canvas.

import { ADAPTER_REGISTRY } from './registry';
import { validateNodeParams } from './param-validation';
import type { PortType, WorkflowEdge, WorkflowGraph, WorkflowNode } from './types';

export const PORT_COLORS: Record<PortType, { bg: string; label: string; short: string }> = {
  StructureArtifact: { bg: '#3b82f6', label: 'Structure (3D)', short: 'Structure' },
  TrajectoryFrames: { bg: '#06b6d4', label: 'Trajectory', short: 'Trajectory' },
  SequenceSet: { bg: '#a855f7', label: 'Sequences (AA)', short: 'Sequences' },
  FoldedStructure: { bg: '#10b981', label: 'Predicted Structure', short: 'Predicted' },
  DesignabilityMetrics: { bg: '#f59e0b', label: 'Designability Scores', short: 'Scores' },
  ConditionSpec: { bg: '#f43f5e', label: 'Design Constraints', short: 'Constraints' },
};

export const STAGE_COLORS: Record<string, string> = {
  source: '#64748b',
  condition: '#f43f5e',
  generate: '#3b82f6',
  'inverse-fold': '#a855f7',
  fold: '#10b981',
  score: '#f59e0b',
};

export const STAGE_LABELS: Record<string, string> = {
  source: 'Input',
  condition: 'Conditioning',
  generate: 'Generation',
  'inverse-fold': 'Sequence Design',
  fold: 'Prediction',
  score: 'Evaluation',
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
    case 'genie3':
      return {
        length: 80,
        nSampleStep: 100,
        directionScale: 0.8,
        noiseScale: 1.0,
        predictSequence: true,
        seed: 42,
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
    const outLabel = PORT_COLORS[outSpec.type]?.label ?? outSpec.type;
    const inLabel = PORT_COLORS[inSpec.type]?.label ?? inSpec.type;
    return {
      valid: false,
      reason: `Type mismatch: ${outLabel} cannot connect to ${inLabel}`,
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
        const outLabel = PORT_COLORS[outSpec.type]?.label ?? outSpec.type;
        const inLabel = PORT_COLORS[inSpec.type]?.label ?? inSpec.type;
        errors.push(
          `Type mismatch on connection ${src.label ?? src.id}.${e.fromPort} → ${dst.label ?? dst.id}.${e.toPort} (${outLabel} vs ${inLabel}).`,
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

  // Check required inputs and node parameter validity
  for (const node of byId.values()) {
    const manifest = ADAPTER_REGISTRY[node.adapterId];
    if (!manifest) continue;
    for (const [portName, spec] of Object.entries(manifest.inputs)) {
      if (spec.required === false) continue;
      if (!validConnectedTargets.has(`${node.id}:${portName}`)) {
        const portLabel = PORT_COLORS[spec.type]?.label ?? spec.type;
        errors.push(
          `"${node.label ?? manifest.shortName}" is missing required input "${portName}" (${portLabel}).`,
        );
      }
    }
    errors.push(...validateNodeParams(node));
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
  proposedEdges: WorkflowEdge[];
}

export interface InsertBlockOptions {
  /** If true (default for backwards-compat), immediately commit proposed edges to graph.edges. Set false for ghost preview UX. */
  autoCommit?: boolean;
  /** Optional parameter overrides merged onto defaultParamsForAdapter(adapterId). */
  params?: Record<string, unknown>;
  /** Optional display label override (deduplicated with #2, #3, ... if already used). */
  label?: string;
}

/**
 * Compute up to N non-conflicting proposed wires for a newly inserted node:
 * - When inserted via + Add Block (no dropPos): anchors on `selectedNodeId`, and if the new node
 *   still has unconnected required inputs (e.g. scRMSD/scTM needs `structure` + `folds`), walks
 *   upstream from `selectedNodeId` to propose the nearest matching ancestor output too.
 * - When inserted via drag-and-drop (`dropPos` given): uses spatial proximity to nearby upstream
 *   nodes only; dropping into open space on a complex graph (>= 4 nodes) proposes 0 wires.
 */
export function proposeConnectionsForNode(
  graphWithNode: WorkflowGraph,
  newNodeId: string,
  selectedNodeId?: string | null,
  dropPos?: { x: number; y: number },
): WorkflowEdge[] {
  const newNode = graphWithNode.nodes.find((n) => n.id === newNodeId);
  if (!newNode) return [];
  const manifest = ADAPTER_REGISTRY[newNode.adapterId];
  if (!manifest) return [];

  const occupiedTargets = new Set(graphWithNode.edges.map((e) => `${e.toNode}:${e.toPort}`));
  const proposed: WorkflowEdge[] = [];

  const tryAddEdge = (fromId: string, fromPort: string, toId: string, toPort: string): boolean => {
    const targetKey = `${toId}:${toPort}`;
    if (occupiedTargets.has(targetKey)) return false;
    const trialGraph: WorkflowGraph = {
      ...graphWithNode,
      edges: [...graphWithNode.edges, ...proposed],
    };
    if (!validateConnection(trialGraph, fromId, fromPort, toId, toPort).valid) return false;
    occupiedTargets.add(targetKey);
    proposed.push({ fromNode: fromId, fromPort, toNode: toId, toPort });
    return true;
  };

  const findFirstValidEdge = (fromId: string, toId: string, allowOptional = true): WorkflowEdge | undefined => {
    const fromNode = graphWithNode.nodes.find((n) => n.id === fromId);
    const toNode = graphWithNode.nodes.find((n) => n.id === toId);
    if (!fromNode || !toNode) return undefined;
    const fromManifest = ADAPTER_REGISTRY[fromNode.adapterId];
    const toManifest = ADAPTER_REGISTRY[toNode.adapterId];
    if (!fromManifest || !toManifest) return undefined;

    for (const fromPort of Object.keys(fromManifest.outputs)) {
      for (const [toPort, inSpec] of Object.entries(toManifest.inputs)) {
        if (!allowOptional && inSpec.required === false) continue;
        if (occupiedTargets.has(`${toId}:${toPort}`)) continue;
        const trialGraph: WorkflowGraph = {
          ...graphWithNode,
          edges: [...graphWithNode.edges, ...proposed],
        };
        if (validateConnection(trialGraph, fromId, fromPort, toId, toPort).valid) {
          return { fromNode: fromId, fromPort, toNode: toId, toPort };
        }
      }
    }
    return undefined;
  };

  // Determine anchor node
  let anchorNode: WorkflowNode | undefined;
  if (dropPos) {
    // Spatial proximity overrides stale selection when dragging onto canvas
    const existingCount = graphWithNode.nodes.length - 1;
    let bestDist = Infinity;
    for (const n of graphWithNode.nodes) {
      if (n.id === newNodeId || !n.position) continue;
      const dx = dropPos.x - n.position.x;
      const dy = Math.abs(dropPos.y - n.position.y);
      // Upstream neighbor sits to the left (dx > 0) within ~420px horizontally and 220px vertically
      const inUpstreamZone = dx >= -60 && dx <= 420 && dy <= 220;
      const inDownstreamZone =
        (manifest.stage === 'source' || manifest.stage === 'condition') &&
        dx <= 60 &&
        dx >= -420 &&
        dy <= 220;
      if (!inUpstreamZone && !inDownstreamZone) continue;
      const d = Math.hypot(dx, dy);
      if (d < bestDist) {
        bestDist = d;
        anchorNode = n;
      }
    }
    // On complex graphs (>= 4 existing nodes), dropping in open space proposes nothing
    if (!anchorNode && existingCount >= 4) {
      return [];
    }
  } else if (selectedNodeId) {
    anchorNode = graphWithNode.nodes.find((n) => n.id === selectedNodeId && n.id !== newNodeId);
  }

  if (!anchorNode) return [];

  const anchorManifest = ADAPTER_REGISTRY[anchorNode.adapterId];
  const insertUpstream =
    manifest.stage === 'source' ||
    (manifest.stage === 'condition' && anchorManifest?.stage !== 'source');

  if (insertUpstream) {
    const e = findFirstValidEdge(newNodeId, anchorNode.id, true);
    if (e) tryAddEdge(e.fromNode, e.fromPort, e.toNode, e.toPort);
    return proposed;
  }

  // Match compatible output(s) from anchorNode into newNode
  const directEdge = findFirstValidEdge(anchorNode.id, newNodeId, true);
  if (directEdge) {
    tryAddEdge(directEdge.fromNode, directEdge.fromPort, directEdge.toNode, directEdge.toPort);
  }

  // Multi-port ancestor completion: if newNode still has unconnected REQUIRED inputs
  // (e.g., adding scRMSD/scTM while LocalFold is selected -> also propose ancestor StructureArtifact),
  // BFS upstream from anchorNode to find the nearest matching producer.
  const missingRequiredPorts = Object.entries(manifest.inputs)
    .filter(([port, spec]) => spec.required !== false && !occupiedTargets.has(`${newNodeId}:${port}`))
    .map(([port]) => port);

  if (missingRequiredPorts.length > 0) {
    const visited = new Set<string>([anchorNode.id, newNodeId]);
    const queue: string[] = [];
    for (const e of graphWithNode.edges) {
      if (e.toNode === anchorNode.id && !visited.has(e.fromNode)) {
        visited.add(e.fromNode);
        queue.push(e.fromNode);
      }
    }

    while (queue.length > 0 && proposed.length < 2) {
      const ancId = queue.shift()!;
      const ancNode = graphWithNode.nodes.find((n) => n.id === ancId);
      const ancManifest = ancNode ? ADAPTER_REGISTRY[ancNode.adapterId] : undefined;
      if (ancNode && ancManifest) {
        for (const toPort of missingRequiredPorts) {
          if (occupiedTargets.has(`${newNodeId}:${toPort}`)) continue;
          const reqType = manifest.inputs[toPort]?.type;
          const matchingOut = Object.entries(ancManifest.outputs).find(([, s]) => s.type === reqType);
          if (matchingOut) {
            tryAddEdge(ancId, matchingOut[0], newNodeId, toPort);
          }
        }
      }
      for (const e of graphWithNode.edges) {
        if (e.toNode === ancId && !visited.has(e.fromNode)) {
          visited.add(e.fromNode);
          queue.push(e.fromNode);
        }
      }
    }
  }

  return proposed;
}

/**
 * Pure helper to insert a new block into a WorkflowGraph with:
 * - collision-free deterministic node ID + readable label + default params
 * - smart placement (dropPos if given, else 280px right of selectedNode with vertical nudge if occupied, else { x: 60, y: 60 })
 * - proposed compatible wire(s) (committed immediately if `options.autoCommit !== false`, or returned as `proposedEdges` only when `autoCommit: false`).
 */
export function insertBlockSmart(
  graph: WorkflowGraph,
  adapterId: string,
  selectedNodeId?: string | null,
  dropPos?: { x: number; y: number },
  options?: InsertBlockOptions,
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

  const customLabel = options?.label?.trim();
  const baseLabel = customLabel || manifest.shortName;
  let label: string;
  if (customLabel) {
    const existingLabels = new Set(graph.nodes.map((n) => n.label).filter(Boolean));
    if (!existingLabels.has(baseLabel)) {
      label = baseLabel;
    } else {
      let idx = 2;
      while (existingLabels.has(`${baseLabel} #${idx}`)) idx++;
      label = `${baseLabel} #${idx}`;
    }
  } else {
    const sameCount = graph.nodes.filter((n) => n.adapterId === adapterId).length;
    label = sameCount > 0 ? `${baseLabel} #${sameCount + 1}` : baseLabel;
  }

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

  const baseParams = defaultParamsForAdapter(adapterId);
  const mergedParams = options?.params ? { ...baseParams, ...options.params } : baseParams;

  const newNode: WorkflowNode = {
    id: nodeId,
    adapterId,
    label,
    params: mergedParams,
    position,
  };

  const graphWithNode: WorkflowGraph = {
    ...graph,
    nodes: [...graph.nodes, newNode],
    edges: [...graph.edges],
  };

  const proposedEdges = proposeConnectionsForNode(graphWithNode, nodeId, selectedNodeId, dropPos);
  const autoCommit = options?.autoCommit !== false;
  const wiredEdge = autoCommit ? proposedEdges[0] : undefined;

  const nextGraph: WorkflowGraph =
    autoCommit && proposedEdges.length > 0
      ? { ...graphWithNode, edges: [...graphWithNode.edges, ...proposedEdges] }
      : graphWithNode;

  return {
    graph: nextGraph,
    nodeId,
    proposedEdges,
    ...(wiredEdge ? { wiredEdge } : {}),
  };
}
