// Pure-TS helpers for inspecting ("peeking") intermediate node inputs/outputs,
// extracting ancestor subgraphs for partial runs, and analyzing intermediate modalities.

import { computeIpsae, sliceProteinPae } from '$lib/protein-ops/ipsae';
import { AA1_TO_AA3 } from '$lib/protein-ops/pdb';
import { getActivePluginRegistry } from '$lib/kernel/active-registry';
import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from './registry';
import type {
  CanonicalArtifact,
  DesignabilityRow,
  PortType,
  StructureArtifact,
  TrajectoryFramesArtifact,
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
  /** Optional plugin-declared slot availability in the node's active mode. */
  availability?: 'active' | 'inactive-in-mode';
  /** Human-readable explanation when `availability === 'inactive-in-mode'`. */
  inactiveReason?: string;
  /** Optional semantic role for the 3D stage header (e.g. `'receptor'`, `'designed'`, `'folded'`). */
  viewportRole?: string;
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
  const registry = getActivePluginRegistry();
  const plugin = registry?.getPlugin(node.adapterId);
  const manifest = plugin?.manifest ?? ADAPTER_REGISTRY[node.adapterId];
  if (!manifest) return [];

  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
  const slots: PeekSlot[] = [];
  const connectedInputs = new Set<string>();

  for (const [portName, spec] of Object.entries(manifest.inputs)) {
    const edge = graph.edges.find((e) => e.toNode === nodeId && e.toPort === portName);
    if (edge) connectedInputs.add(portName);
    const peer = edge ? nodeById.get(edge.fromNode) : undefined;
    const peerLabel = peer
      ? peer.label ||
        registry?.getPlugin(peer.adapterId)?.manifest.shortName ||
        ADAPTER_REGISTRY[peer.adapterId]?.shortName ||
        peer.id
      : undefined;
    const artifact = edge ? (outputsByNode[edge.fromNode]?.[edge.fromPort] ?? null) : null;
    const primaryType = (Array.isArray(spec.type) ? spec.type[0] : spec.type) as PortType;
    slots.push({
      id: `in:${portName}`,
      direction: 'in',
      portName,
      portType: primaryType,
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
    const primaryType = (Array.isArray(spec.type) ? spec.type[0] : spec.type) as PortType;
    slots.push({
      id: `out:${portName}`,
      direction: 'out',
      portName,
      portType: primaryType,
      required: true,
      wired: true,
      artifact,
    });
  }

  // Apply plugin-owned slot availability / viewport-role metadata (e.g. GNINA inactive `track` in score/dock modes)
  const resolvedSlots = plugin?.display.resolveSlots?.(node.params, connectedInputs);
  if (resolvedSlots && resolvedSlots.length > 0) {
    const slotMap = new Map(resolvedSlots.map((d) => [`${d.direction}:${d.portName}`, d]));
    for (const slot of slots) {
      const desc = slotMap.get(slot.id);
      if (!desc) continue;
      slot.availability = desc.availability;
      if (desc.inactiveReason) slot.inactiveReason = desc.inactiveReason;
      if (desc.viewportRole) slot.viewportRole = desc.viewportRole;
    }
  }

  return slots;
}

const PORT_TYPE_PRIORITY: Record<PortType, number> = {
  DesignabilityMetrics: 80,
  MutationMatrix: 70,
  ResidueTrack: 65,
  FoldedStructure: 60,
  StructureArtifact: 50,
  SequenceSet: 40,
  SequenceEmbedding: 30,
  TrajectoryFrames: 20,
  ConditionSpec: 10,
};

export function getPortTypePriority(portType: PortType): number {
  const registry = getActivePluginRegistry();
  if (registry?.portTypes.has(portType)) {
    return registry.portTypes.resolvePeekPriority(portType);
  }
  return PORT_TYPE_PRIORITY[portType] ?? 0;
}

/**
 * Choose the most informative default peek slot for a node using explicit output modality priority:
 * 1) Active outgoing slot with live data, sorted by PORT_TYPE_PRIORITY
 * 2) Otherwise active incoming slot with live data (useful while this node is still running)
 * 3) Otherwise highest-priority active outgoing slot, or first slot.
 */
export function pickDefaultPeekSlot(slots: PeekSlot[]): string | null {
  const isActive = (s: PeekSlot) => s.availability !== 'inactive-in-mode';
  const outWithData = slots.filter((s) => s.direction === 'out' && s.artifact !== null && isActive(s));
  if (outWithData.length > 0) {
    outWithData.sort((a, b) => getPortTypePriority(b.portType) - getPortTypePriority(a.portType));
    return outWithData[0].id;
  }
  const inWithData = slots.filter((s) => s.direction === 'in' && s.artifact !== null && isActive(s));
  if (inWithData.length > 0) {
    inWithData.sort((a, b) => getPortTypePriority(b.portType) - getPortTypePriority(a.portType));
    return inWithData[0].id;
  }
  const activeOuts = slots.filter((s) => s.direction === 'out' && isActive(s));
  if (activeOuts.length > 0) {
    activeOuts.sort((a, b) => getPortTypePriority(b.portType) - getPortTypePriority(a.portType));
    return activeOuts[0].id;
  }
  const outs = slots.filter((s) => s.direction === 'out');
  if (outs.length > 0) {
    outs.sort((a, b) => getPortTypePriority(b.portType) - getPortTypePriority(a.portType));
    return outs[0].id;
  }
  return slots[0]?.id ?? null;
}

export function selectPrimaryOutputPort(
  outs: Record<string, CanonicalArtifact> | undefined,
): { portName: string; artifact: CanonicalArtifact; is3D: boolean } | null {
  if (!outs) return null;
  let best: { portName: string; artifact: CanonicalArtifact; priority: number } | null = null;
  for (const [portName, art] of Object.entries(outs)) {
    const priority = getPortTypePriority(art.kind as PortType);
    if (!best || priority > best.priority) {
      best = { portName, artifact: art, priority };
    }
  }
  if (!best) return null;
  const is3D =
    best.artifact.kind === 'DesignabilityMetrics' ||
    best.artifact.kind === 'FoldedStructure' ||
    best.artifact.kind === 'StructureArtifact';
  return { portName: best.portName, artifact: best.artifact, is3D };
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
 *
 * When `opts.excludeSelf` is `true`, skip reading `outputsByNode[nodeId]` and only inspect
 * upstream ancestors (so co-packing inverse folders like `LASErMPNN` resolve their upstream
 * WT input structure rather than their own packed output).
 */
export function findContextStructure(
  graph: WorkflowGraph,
  nodeId: string,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
  opts?: { excludeSelf?: boolean },
): StructureArtifact | null {
  const visited = new Set<string>([nodeId]);
  const queue: string[] = [nodeId];
  const excludeSelf = Boolean(opts?.excludeSelf);

  while (queue.length > 0) {
    const cur = queue.shift()!;
    if (!excludeSelf || cur !== nodeId) {
      const outs = outputsByNode[cur];
      if (outs) {
        for (const art of Object.values(outs)) {
          if (art.kind === 'StructureArtifact') return art;
        }
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

/** Return a bounded preview of PDB/mmCIF text for raw-text inspection without splitting the full file. */
export function previewTextLines(text: string, maxLines = 40): { preview: string; totalLines: number; truncated: boolean } {
  const limit = Math.max(0, Math.floor(maxLines));
  let nlCount = 0;
  let previewEnd = -1;
  let idx = text.indexOf('\n');
  while (idx !== -1) {
    nlCount++;
    if (nlCount === limit && previewEnd === -1) {
      previewEnd = idx;
    }
    idx = text.indexOf('\n', idx + 1);
  }
  const totalLines = nlCount + 1;
  if (totalLines <= limit) {
    return { preview: text, totalLines, truncated: false };
  }
  if (limit === 0) {
    return { preview: '', totalLines, truncated: true };
  }
  let rawPreview = previewEnd >= 0 ? text.slice(0, previewEnd) : text;
  if (rawPreview.endsWith('\r')) {
    rawPreview = rawPreview.slice(0, -1);
  }
  const preview = rawPreview.includes('\r') ? rawPreview.replace(/\r\n/g, '\n') : rawPreview;
  return {
    preview,
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

/** Infer per-chain residue counts from a PDB string's CA records when more than one chain is present. */
export function inferPdbChainLengths(pdbText?: string): number[] | undefined {
  if (!pdbText) return undefined;
  const counts: number[] = [];
  const seen = new Set<string>();
  let lastChain = '';
  for (const line of pdbText.split('\n')) {
    if (!line.startsWith('ATOM  ') || line.length < 27) continue;
    if (line.slice(12, 16).trim() !== 'CA') continue;
    const chain = line[21] || 'A';
    const resKey = `${chain}:${line.slice(22, 27)}`;
    if (seen.has(resKey)) continue;
    seen.add(resKey);
    if (counts.length === 0 || chain !== lastChain) {
      counts.push(1);
      lastChain = chain;
    } else {
      counts[counts.length - 1] += 1;
    }
  }
  return counts.length > 1 ? counts : undefined;
}

/** Insert `:` chain delimiters into a concatenated multimer amino-acid sequence when chainLengths sum matches. */
export function formatCandidateSequence(sequence: string, chainLengths?: readonly number[]): string {
  if (!chainLengths || chainLengths.length <= 1) return sequence;
  const sum = chainLengths.reduce((acc, n) => acc + n, 0);
  if (sum !== sequence.length) return sequence;
  const parts: string[] = [];
  let offset = 0;
  for (const len of chainLengths) {
    parts.push(sequence.slice(offset, offset + len));
    offset += len;
  }
  return parts.join(':');
}

/** Resolve chain lengths for a DesignabilityRow, prioritizing row.chainLengths before PDB/context fallback. */
export function resolveActiveRowChainLengths(
  row: Pick<DesignabilityRow, 'chainLengths' | 'alignedFoldPdb'> | null | undefined,
  contextStructure?: Pick<StructureArtifact, 'chainLengths' | 'diagnostics'> | null,
): number[] | undefined {
  if (!row) return undefined;
  if (row.chainLengths && row.chainLengths.length > 1) {
    return [...row.chainLengths];
  }
  const fromPdb = inferPdbChainLengths(row.alignedFoldPdb);
  if (fromPdb && fromPdb.length > 1) return fromPdb;
  const ctxLens = contextStructure?.chainLengths ?? contextStructure?.diagnostics?.chainLengths;
  return ctxLens && ctxLens.length > 1 ? [...ctxLens] : undefined;
}

/** Resolve ipSAE / ipSAE_min from explicit values or fallback computation on the candidate PAE matrix. */
export function resolveFoldIpsae(fold: {
  ipsae?: number;
  ipsaeMin?: number;
  pae?: Float32Array;
  chainLengths?: readonly number[];
  chainIds?: readonly string[];
}): { ipsae: number; ipsaeMin?: number } | null {
  if (typeof fold.ipsae === 'number' && Number.isFinite(fold.ipsae)) {
    return {
      ipsae: fold.ipsae,
      ...(typeof fold.ipsaeMin === 'number' && Number.isFinite(fold.ipsaeMin)
        ? { ipsaeMin: fold.ipsaeMin }
        : {}),
    };
  }
  if (!fold.pae || fold.pae.length === 0) return null;
  const chainIds = fold.chainIds;
  const chainLengths = fold.chainLengths;
  const L =
    chainIds && chainIds.length > 0
      ? chainIds.length
      : chainLengths && chainLengths.length > 0
        ? chainLengths.reduce((a, b) => a + b, 0)
        : 0;
  if (L <= 0) return null;
  let perResChains: string[];
  if (chainIds && chainIds.length === L && new Set(chainIds).size >= 2) {
    perResChains = [...chainIds];
  } else if (
    chainLengths &&
    chainLengths.length > 1 &&
    chainLengths.reduce((a, b) => a + b, 0) === L
  ) {
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    perResChains = [];
    for (let c = 0; c < chainLengths.length; c++) {
      const cid = c < alphabet.length ? alphabet[c] : `C${c}`;
      for (let k = 0; k < chainLengths[c]; k++) perResChains.push(cid);
    }
  } else {
    return null;
  }
  const sliced = sliceProteinPae(fold.pae, L);
  if (!sliced) return null;
  const res = computeIpsae(sliced, perResChains, 10);
  return res ? { ipsae: res.ipsae, ipsaeMin: res.ipsaeMin } : null;
}

/** Infer residue count for a TrajectoryFramesArtifact from frame coordinates, PDB text, or parent structure. */
export function inferTrajectoryResidueCount(
  traj: TrajectoryFramesArtifact,
  contextStruct?: Pick<StructureArtifact, 'id' | 'length'> | null,
): number | null {
  const firstFrameWithCoords = traj.frames?.find((f) => f.caCoords && f.caCoords.length >= 3);
  if (firstFrameWithCoords?.caCoords) {
    return Math.floor(firstFrameWithCoords.caCoords.length / 3);
  }
  const samplePdb = traj.frames?.find((f) => f.pdbText?.trim())?.pdbText ?? traj.pdbText;
  if (samplePdb) {
    const seen = new Set<string>();
    for (const line of samplePdb.split('\n')) {
      if (line.startsWith('ENDMDL')) break;
      if (!line.startsWith('ATOM  ') || line.length < 27) continue;
      if (line.slice(12, 16).trim() !== 'CA') continue;
      seen.add(`${line[21] || 'A'}:${line.slice(22, 27)}`);
    }
    if (seen.size > 0) return seen.size;
  }
  if (contextStruct && contextStruct.length > 0 && contextStruct.id === traj.parentStructureId) {
    return contextStruct.length;
  }
  return null;
}

function stripPdbModelEnvelope(pdbText: string): string {
  const out: string[] = [];
  for (const line of pdbText.split(/\r?\n/)) {
    const tr = line.trim();
    if (!tr) continue;
    if (line.startsWith('MODEL ') || tr === 'MODEL' || tr === 'ENDMDL' || tr === 'END') continue;
    out.push(line);
  }
  return out.join('\n');
}

/**
 * Build a downloadable multi-MODEL PDB string from a TrajectoryFramesArtifact:
 * uses pre-serialized `pdbText` when present, wraps per-frame `pdbText` entries, or
 * synthesizes clean Cα multi-MODEL PDB records directly from per-frame `caCoords`.
 */
export function buildTrajectoryMultiModelPdb(
  traj: TrajectoryFramesArtifact,
  contextStruct?: Pick<StructureArtifact, 'id' | 'length' | 'sequence' | 'chainIds' | 'sourceResidues'> | null,
): string | null {
  if (traj.pdbText && traj.pdbText.trim().length > 0) {
    return traj.pdbText;
  }
  const frames = traj.frames ?? [];
  if (frames.length === 0) return null;

  const chunks: string[] = [
    `REMARK   1 DENOVO.RUN TRAJECTORY EXPORT (${frames.length} FRAMES)`,
  ];
  let emittedModels = 0;

  for (let idx = 0; idx < frames.length; idx++) {
    const frame = frames[idx];
    const modelNum = idx + 1;
    if (frame.pdbText && frame.pdbText.trim().length > 0) {
      const body = stripPdbModelEnvelope(frame.pdbText);
      if (!body) continue;
      emittedModels++;
      chunks.push(`MODEL     ${String(modelNum).padStart(4, ' ')}`);
      chunks.push(body);
      chunks.push('ENDMDL');
      continue;
    }
    if (frame.caCoords && frame.caCoords.length >= 3) {
      const nRes = Math.floor(frame.caCoords.length / 3);
      const useCtx = Boolean(contextStruct && contextStruct.length === nRes);
      emittedModels++;
      const stepNote =
        typeof frame.timeStep === 'number' && Number.isFinite(frame.timeStep)
          ? ` T=${frame.timeStep}`
          : '';
      chunks.push(`REMARK   2 FRAME ${modelNum}/${frames.length} KEY=${frame.key}${stepNote}`);
      chunks.push(`MODEL     ${String(modelNum).padStart(4, ' ')}`);
      for (let r = 0; r < nRes; r++) {
        const x = frame.caCoords[r * 3];
        const y = frame.caCoords[r * 3 + 1];
        const z = frame.caCoords[r * 3 + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
        const aa1 = useCtx ? (contextStruct!.sequence[r]?.toUpperCase() ?? 'A') : 'A';
        const resName = (AA1_TO_AA3[aa1] ?? 'ALA').padStart(3, ' ');
        const rawChain = useCtx ? (contextStruct!.chainIds?.[r] ?? 'A') : 'A';
        const chain = (rawChain.trim()[0] || 'A');
        const resNum = useCtx ? (contextStruct!.sourceResidues?.[r]?.resNum ?? r + 1) : r + 1;
        const serialStr = String((r + 1) % 100000).padStart(5, ' ');
        const resSeqStr = String(resNum).padStart(4, ' ');
        const xStr = x.toFixed(3).padStart(8, ' ');
        const yStr = y.toFixed(3).padStart(8, ' ');
        const zStr = z.toFixed(3).padStart(8, ' ');
        chunks.push(
          `ATOM  ${serialStr}  CA  ${resName} ${chain}${resSeqStr}    ${xStr}${yStr}${zStr}  1.00  0.00           C  `,
        );
      }
      chunks.push('ENDMDL');
    }
  }

  if (emittedModels === 0) return null;
  chunks.push('END');
  return `${chunks.join('\n')}\n`;
}
