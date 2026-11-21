// Pure-TS helpers for inspecting ("peeking") intermediate node inputs/outputs,
// extracting ancestor subgraphs for partial runs, and analyzing intermediate modalities.

import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from './registry';
import type {
  CanonicalArtifact,
  PortType,
  StructureArtifact,
  WorkflowEdge,
  WorkflowGraph,
} from './types';

export interface PeekSlot {
  id: string; // 'in:<port>' | 'out:<port>'
  direction: 'in' | 'out';
  portName: string;
  portType: PortType;
  required: boolean;
  wired: boolean;
  edgeId?: string;
  peerNodeId?: string;
  peerNodeLabel?: string;
  peerPortName?: string;
  artifact: CanonicalArtifact | null;
}

export interface SequenceAnalysis {
  length: number;
  uniqueAaCount: number;
  unknownCount: number;
  topAa: { aa: string; count: number; pct: number } | null;
  hydrophobicPct: number;
  polarPct: number;
  chargedPct: number;
  recoveryPct: number | null;
  warnings: string[];
}

const HYDROPHOBIC = new Set(['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']);
const CHARGED = new Set(['D', 'E', 'K', 'R', 'H']);
const POLAR = new Set(['S', 'T', 'N', 'Q', 'Y', 'C', 'G']);

export function makeEdgeId(fromNode: string, fromPort: string, toNode: string, toPort: string): string {
  return `${fromNode}:${fromPort}->${toNode}:${toPort}`;
}

/**
 * List all incoming and outgoing port slots for a given node, resolving live artifacts
 * from `outputsByNode` and annotating upstream wire metadata.
 */
export function getNodePeekSlots(
  graph: WorkflowGraph,
  nodeId: string,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
): PeekSlot[] {
  const node = graph.nodes.find((n) => n.id === nodeId);
  if (!node) return [];
  const manifest = ADAPTER_REGISTRY[node.adapterId];
  if (!manifest) return [];

  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
  const slots: PeekSlot[] = [];

  for (const [portName, spec] of Object.entries(manifest.inputs)) {
    const edge = graph.edges.find((e) => e.toNode === nodeId && e.toPort === portName);
    const peer = edge ? nodeById.get(edge.fromNode) : undefined;
    const peerLabel = peer
      ? peer.label || ADAPTER_REGISTRY[peer.adapterId]?.shortName || peer.id
      : undefined;
    const artifact = edge ? (outputsByNode[edge.fromNode]?.[edge.fromPort] ?? null) : null;
    slots.push({
      id: `in:${portName}`,
      direction: 'in',
      portName,
      portType: spec.type,
      required: spec.required !== false,
      wired: Boolean(edge),
      edgeId: edge ? makeEdgeId(edge.fromNode, edge.fromPort, edge.toNode, edge.toPort) : undefined,
      peerNodeId: edge?.fromNode,
      peerNodeLabel: peerLabel,
      peerPortName: edge?.fromPort,
      artifact,
    });
  }

  for (const [portName, spec] of Object.entries(manifest.outputs)) {
    const artifact = outputsByNode[nodeId]?.[portName] ?? null;
    slots.push({
      id: `out:${portName}`,
      direction: 'out',
      portName,
      portType: spec.type,
      required: true,
      wired: true,
      artifact,
    });
  }

  return slots;
}

/**
 * Choose the most informative default peek slot for a node:
 * 1) First outgoing slot with live data
 * 2) Otherwise first incoming slot with live data (useful while this node is still running!)
 * 3) Otherwise first outgoing slot, or first incoming slot.
 */
export function pickDefaultPeekSlot(slots: PeekSlot[]): string | null {
  const outWithData = slots.find((s) => s.direction === 'out' && s.artifact !== null);
  if (outWithData) return outWithData.id;
  const inWithData = slots.find((s) => s.direction === 'in' && s.artifact !== null);
  if (inWithData) return inWithData.id;
  const firstOut = slots.find((s) => s.direction === 'out');
  if (firstOut) return firstOut.id;
  return slots[0]?.id ?? null;
}

/**
 * Extract the minimal ancestor subgraph required to execute `targetNodeId`
 * (preserving original node ordering and all edges within that ancestor set).
 */
export function extractSubgraphUpTo(graph: WorkflowGraph, targetNodeId: string): WorkflowGraph {
  const nodeIds = new Set(graph.nodes.map((n) => n.id));
  if (!nodeIds.has(targetNodeId)) {
    return { ...graph, nodes: [], edges: [] };
  }

  const needed = new Set<string>([targetNodeId]);
  const queue: string[] = [targetNodeId];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    for (const e of graph.edges) {
      if (e.toNode === cur && nodeIds.has(e.fromNode) && !needed.has(e.fromNode)) {
        needed.add(e.fromNode);
        queue.push(e.fromNode);
      }
    }
  }

  return {
    ...graph,
    nodes: graph.nodes.filter((n) => needed.has(n.id)),
    edges: graph.edges.filter((e) => needed.has(e.fromNode) && needed.has(e.toNode)),
  };
}

/**
 * Walk upstream from `nodeId` (breadth-first) to find the closest available
 * StructureArtifact for 3D context or sequence-recovery comparison.
 */
export function findContextStructure(
  graph: WorkflowGraph,
  nodeId: string,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
): StructureArtifact | null {
  const visited = new Set<string>([nodeId]);
  const queue: string[] = [nodeId];

  while (queue.length > 0) {
    const cur = queue.shift()!;
    const outs = outputsByNode[cur];
    if (outs) {
      for (const art of Object.values(outs)) {
        if (art.kind === 'StructureArtifact') return art;
      }
    }
    const incoming = graph.edges.filter((e) => e.toNode === cur && !visited.has(e.fromNode));
    // Prioritize incoming `structure` edges over `folds` edges so auto-wired Scorers
    // (where `fold -> score` may precede `gen -> score`) resolve the designed backbone
    // (preserving `trajectoryPdb`) rather than LocalFold's refolded StructureArtifact.
    const edgePriority = (e: WorkflowEdge): number => {
      if (e.toPort === 'structure') return 0;
      if (e.fromPort === 'structure') return 1;
      if (e.toPort === 'folds' || e.fromPort === 'folds') return 3;
      return 2;
    };
    incoming.sort((a, b) => edgePriority(a) - edgePriority(b));
    for (const e of incoming) {
      if (!visited.has(e.fromNode)) {
        visited.add(e.fromNode);
        queue.push(e.fromNode);
      }
    }
  }
  return null;
}

/**
 * Compute lightweight composition & sanity statistics for an amino-acid sequence,
 * plus optional sequence identity/recovery against a same-length reference sequence.
 */
export function analyzeSequence(rawSeq: string, referenceSeq?: string): SequenceAnalysis {
  const seq = rawSeq.trim().toUpperCase();
  const length = seq.length;
  if (length === 0) {
    return {
      length: 0,
      uniqueAaCount: 0,
      unknownCount: 0,
      topAa: null,
      hydrophobicPct: 0,
      polarPct: 0,
      chargedPct: 0,
      recoveryPct: null,
      warnings: ['Empty sequence'],
    };
  }

  const counts = new Map<string, number>();
  let unknownCount = 0;
  let hyd = 0;
  let pol = 0;
  let chg = 0;

  for (let i = 0; i < length; i++) {
    const ch = seq[i];
    if (ch === 'X' || ch === '?' || ch === '-') {
      unknownCount++;
      continue;
    }
    counts.set(ch, (counts.get(ch) ?? 0) + 1);
    if (HYDROPHOBIC.has(ch)) hyd++;
    else if (CHARGED.has(ch)) chg++;
    else if (POLAR.has(ch)) pol++;
  }

  let topAa: { aa: string; count: number; pct: number } | null = null;
  for (const [aa, count] of counts.entries()) {
    if (!topAa || count > topAa.count) {
      topAa = { aa, count, pct: (count / length) * 100 };
    }
  }

  let recoveryPct: number | null = null;
  if (referenceSeq) {
    const ref = referenceSeq.trim().toUpperCase();
    if (ref.length === length && length > 0) {
      let matches = 0;
      let comparable = 0;
      for (let i = 0; i < length; i++) {
        if (ref[i] !== 'X' && seq[i] !== 'X') {
          comparable++;
          if (ref[i] === seq[i]) matches++;
        }
      }
      if (comparable > 0) recoveryPct = (matches / comparable) * 100;
    }
  }

  const warnings: string[] = [];
  if (unknownCount > 0) {
    const pct = ((unknownCount / length) * 100).toFixed(0);
    warnings.push(`${unknownCount} unknown ('X') residue${unknownCount > 1 ? 's' : ''} (${pct}%)`);
  }
  if (topAa && length >= 10 && topAa.pct >= 50) {
    warnings.push(`Low complexity: ${topAa.pct.toFixed(0)}% ${topAa.aa}`);
  } else if (length >= 20 && counts.size > 0 && counts.size <= 3) {
    warnings.push(`Very low diversity (${counts.size} unique AAs)`);
  }

  return {
    length,
    uniqueAaCount: counts.size,
    unknownCount,
    topAa,
    hydrophobicPct: (hyd / length) * 100,
    polarPct: (pol / length) * 100,
    chargedPct: (chg / length) * 100,
    recoveryPct,
    warnings,
  };
}

/** Return a bounded preview of PDB/mmCIF text for raw-text inspection. */
export function previewTextLines(text: string, maxLines = 40): { preview: string; totalLines: number; truncated: boolean } {
  const lines = text.split(/\r?\n/);
  const totalLines = lines.length;
  if (totalLines <= maxLines) {
    return { preview: text, totalLines, truncated: false };
  }
  return {
    preview: lines.slice(0, maxLines).join('\n'),
    totalLines,
    truncated: true,
  };
}

/** Normalize worker foldModelId values (e.g. `localfold-af2-monomer-int5`, `localfold-boltz2-int5`, `localfold-af2`) into a model slug. */
export function foldModelSlug(foldModelId: string): string {
  const clean = foldModelId
    .trim()
    .replace(/^localfold-(?:af2-)?/i, '')
    .replace(/-int5$/i, '')
    .trim();
  return clean || 'folded';
}

/** Format a worker foldModelId into a human-readable folding model name. */
export function formatFoldModelName(foldModelId: string): string {
  const slug = foldModelSlug(foldModelId);
  if (slug.toLowerCase() === 'af2') return 'AlphaFold2';
  const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === slug);
  return meta ? meta.label : slug;
}
