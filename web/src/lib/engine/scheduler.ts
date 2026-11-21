// Headless DAG ExecutionScheduler (zero Svelte/DOM dependencies).
// Handles topological execution, content-addressed memoization, Single-Active-GPU-Worker
// VRAM mutex (dispose -> terminate), CPU nodes, and WebMCP / window.__denovo hooks.

import MpnnWorker from '$lib/workers/mpnn.worker?worker';
import LocalFoldWorker from '$lib/workers/localfold.worker?worker';
import Rfd4Worker from '$lib/workers/rfd4.worker?worker';
import Genie3Worker from '$lib/workers/genie3.worker?worker';
import ThermoMpnnWorker from '$lib/workers/thermompnn.worker?worker';
import LigandMpnnWorker from '$lib/workers/ligandmpnn.worker?worker';
import LaserMpnnWorker from '$lib/workers/lasermpnn.worker?worker';
import AntiFoldWorker from '$lib/workers/antifold.worker?worker';
import {
  cropRawAllAtomStructure,
  generateSampleHelixBundlePdb,
  parseStructureText,
} from '$lib/protein-ops/pdb';
import { fetchStructureByPdbId, normalizePdbId } from '$lib/protein-ops/rcsb';
import { applyTransformToPdb, computeGeometryStats, kabschAlign } from '$lib/protein-ops/kabsch';
import { computeIpsae, sliceProteinPae } from '$lib/protein-ops/ipsae';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from './registry';
import { validateGraph } from './graph-utils';
import { normalizeWorkflowGraph } from './share';
import {
  resolveFoldConfidence,
  splitMultiChainSequence,
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from './bio-capabilities';
import {
  clearAllModelCaches,
  type CheckpointStorageAdapter,
} from './opfs-cache';
import {
  exportJoinedCollectionCsv,
  joinBranchArtifacts,
  joinKeyedCollections,
} from './compare';
import { sha256Hex, type CollectionMember } from './collections';
import { planTopKFilter } from './work-plan';
import type {
  ArtifactRef,
  CanonicalArtifact,
  CollectionItemOutcome,
  DesignabilityMetrics,
  DesignabilityRow,
  ExcludedCandidateRecord,
  FoldedCandidate,
  FoldedStructure,
  NodeProgressEvent,
  ReferenceRelation,
  ReferenceResidueMapping,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
  WorkflowNode,
} from './types';

export { exportJoinedCollectionCsv, joinBranchArtifacts, joinKeyedCollections };
export type { CheckpointStorageAdapter };

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

/** Deterministic key-sorted JSON serializer so reordered param keys never bust memoization. */
function canonicalJsonStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((v) => (v === undefined ? 'null' : canonicalJsonStringify(v))).join(',')}]`;
  }
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj).sort();
  const parts: string[] = [];
  for (const k of keys) {
    const v = obj[k];
    if (v === undefined) continue;
    parts.push(`${JSON.stringify(k)}:${canonicalJsonStringify(v)}`);
  }
  return `{${parts.join(',')}}`;
}

/**
 * Pure CPU/fetch materializer for a `pdb-input` node.
 * Shared by both ExecutionScheduler and instant live 3D preview (no Run needed).
 */
export async function materializePdbInputNode(
  node: WorkflowNode,
  onProgress?: (ev: NodeProgressEvent) => void,
): Promise<StructureArtifact> {
  const mode = String(node.params.mode ?? 'demo-bundle');
  let rawText: string;
  let filename: string;
  if (mode === 'pdb-id') {
    const normId = normalizePdbId(String(node.params.pdbId ?? ''));
    if (!normId) {
      throw new Error('PDB Input is in PDB ID mode, but no PDB ID was entered.');
    }
    const cachedText = String(node.params.pdbText ?? '').trim();
    const fetchedFor = normalizePdbId(String(node.params.fetchedPdbId ?? ''));
    if (cachedText.length > 0 && fetchedFor === normId) {
      rawText = cachedText;
      const rawFn = String(node.params.filename ?? '').trim();
      filename = rawFn && rawFn !== 'demo_3helix_53aa.pdb' ? rawFn : `${normId}.cif`;
    } else {
      onProgress?.({
        nodeId: node.id,
        status: 'loading',
        phase: 'fetch',
        progress: 0.2,
        detail: `Fetching ${normId} from RCSB PDB...`,
      });
      const fetched = await fetchStructureByPdbId(normId);
      rawText = fetched.text;
      filename = fetched.filename;
    }
  } else if (mode === 'custom' && String(node.params.pdbText ?? '').trim().length > 0) {
    rawText = String(node.params.pdbText);
    filename = String(node.params.filename || 'uploaded.pdb');
  } else {
    rawText = generateSampleHelixBundlePdb();
    filename = 'demo_3helix_53aa.pdb';
  }
  const cropSpec = mode === 'demo-bundle' ? '' : String(node.params.cropSegments ?? '').trim();
  const isCropped = cropSpec.length > 0;
  let effectiveRawText = rawText;
  let omittedLigands: string[] = [];
  if (isCropped) {
    effectiveRawText = cropRawAllAtomStructure(rawText, cropSpec).text;
  }
  const parsed = parseStructureText(effectiveRawText);
  if (isCropped) {
    try {
      const fullLigands = parseStructureText(rawText).diagnostics.ligandNames;
      if (fullLigands.length > 0) {
        const keptSet = new Set(parsed.diagnostics.ligandNames.map((l) => l.toUpperCase()));
        omittedLigands = fullLigands.filter((l) => !keptSet.has(l.toUpperCase()));
      }
    } catch {
      omittedLigands = [];
    }
  }
  const diagnostics =
    omittedLigands.length > 0
      ? { ...parsed.diagnostics, omittedLigands }
      : parsed.diagnostics;
  const hasNonProteinContext = parsed.diagnostics.hasLigands || parsed.diagnostics.hasNucleicAcids;
  if (parsed.nResidues < 3 && !hasNonProteinContext) {
    throw new Error('Input PDB/mmCIF must contain at least 3 residues with Cα atoms');
  }
  const geometry = computeGeometryStats(
    parsed.caCoords,
    parsed.chainIds,
    parsed.resNums,
    parsed.insCodes,
  );
  const benchmarkId = String(node.params.benchmarkId ?? '').trim() || undefined;

  const summaryParts: string[] = [];
  if (parsed.nResidues > 0) summaryParts.push(`L=${parsed.nResidues}`);
  if (parsed.diagnostics.ligandNames.length > 0) summaryParts.push(parsed.diagnostics.ligandNames.join('+'));
  if (parsed.diagnostics.nucleicAcidChains.length > 0) {
    summaryParts.push(`DNA(${parsed.diagnostics.nucleicAcidChains.join(',')})`);
  }
  const summaryTag = summaryParts.join(' · ') || 'context';

  return {
    kind: 'StructureArtifact',
    id: `pdb-${fnv1a(`${rawText}::${cropSpec}${benchmarkId ? `::${benchmarkId}` : ''}`)}`,
    label: isCropped
      ? `${filename} [${cropSpec}] (${summaryTag})`
      : `${filename} (${summaryTag})`,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.nResidues > 0 ? parsed.pdbText : parsed.format === 'mmcif' ? '' : effectiveRawText,
    cifText: parsed.format === 'mmcif' ? effectiveRawText : undefined,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    chainIdMap: parsed.chainIdMap,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics,
    ...(benchmarkId ? { benchmarkId } : {}),
    geometry,
  };
}

/**
 * Pure CPU materializer for a `sequence-input` node.
 * Shared by both ExecutionScheduler and instant live preview (no Run needed).
 */
export function materializeSequenceInputNode(node: WorkflowNode): SequenceSet {
  const raw = String(node.params.sequence ?? '').trim();
  if (!raw) throw new Error('Sequence Input is empty');
  const candidates: Array<{
    id: string;
    key: string;
    sequence: string;
    chainLengths?: number[];
    score: null;
    globalScore: null;
    sourceModelId: string;
    sourceStructureId: string;
  }> = [];
  if (raw.includes('>')) {
    const blocks = raw.split(/^\s*>/m).filter((b) => b.trim().length > 0);
    const seenIds = new Set<string>();
    blocks.forEach((b, idx) => {
      const lines = b.split('\n');
      const baseHeader = lines[0].trim().split(/\s+/)[0] || `seq-${idx + 1}`;
      const split = splitMultiChainSequence(lines.slice(1).join(''));
      if (split.concatenated.length >= 5) {
        let id = baseHeader;
        let suffix = 2;
        while (seenIds.has(id)) id = `${baseHeader}_${suffix++}`;
        seenIds.add(id);
        candidates.push({
          id,
          key: id,
          sequence: split.concatenated,
          ...(split.chains.length > 1 ? { chainLengths: split.chainLengths } : {}),
          score: null,
          globalScore: null,
          sourceModelId: 'sequence-input',
          sourceStructureId: 'none',
        });
      }
    });
  } else {
    const split = splitMultiChainSequence(raw);
    if (split.concatenated.length >= 5) {
      const id = String(node.params.label || 'custom-seq-1');
      candidates.push({
        id,
        key: id,
        sequence: split.concatenated,
        ...(split.chains.length > 1 ? { chainLengths: split.chainLengths } : {}),
        score: null,
        globalScore: null,
        sourceModelId: 'sequence-input',
        sourceStructureId: 'none',
      });
    }
  }
  if (candidates.length === 0) throw new Error('No valid amino acid sequences (>= 5 aa) found');
  return {
    kind: 'SequenceSet',
    id: `seqin-${fnv1a(JSON.stringify(candidates))}`,
    parentStructureId: 'none',
    sourceStructureId: 'none',
    sourceModelId: 'sequence-input',
    collectionKind: 'independent-candidates',
    candidates,
  };
}

function topoSort(graph: WorkflowGraph): WorkflowNode[] {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const indeg = new Map<string, number>(graph.nodes.map((n) => [n.id, 0]));
  const adj = new Map<string, string[]>(graph.nodes.map((n) => [n.id, []]));

  for (const e of graph.edges) {
    if (!byId.has(e.fromNode) || !byId.has(e.toNode)) continue;
    adj.get(e.fromNode)!.push(e.toNode);
    indeg.set(e.toNode, (indeg.get(e.toNode) ?? 0) + 1);
  }

  const queue: string[] = [];
  for (const [id, d] of indeg) if (d === 0) queue.push(id);

  const ordered: WorkflowNode[] = [];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    ordered.push(byId.get(cur)!);
    for (const nxt of adj.get(cur) ?? []) {
      const nd = (indeg.get(nxt) ?? 1) - 1;
      indeg.set(nxt, nd);
      if (nd === 0) queue.push(nxt);
    }
  }
  if (ordered.length !== graph.nodes.length) {
    throw new Error('WorkflowGraph contains a cycle or invalid edge');
  }
  return ordered;
}

export type RunSource = 'canvas' | 'api';
export type RunLifecycleStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface ActiveRunState {
  runId: string | null;
  source: RunSource | null;
  status: RunLifecycleStatus;
  graphId: string | null;
  activeNodeId: string | null;
  leaseHeld: boolean;
  completedNodeIds: string[];
  unfinishedNodeIds: string[];
  startedAt: string | null;
  finishedAt: string | null;
  error?: string | null;
}

export interface RunHistoryEntry {
  runId: string;
  source: RunSource;
  graphId: string;
  status: Exclude<RunLifecycleStatus, 'idle' | 'running'>;
  startedAt: string;
  finishedAt: string;
  elapsedMs: number;
  completedNodeIds: string[];
  unfinishedNodeIds: string[];
  refsByNode: Record<string, Record<string, ArtifactRef>>;
  error?: string;
}

export interface CheckpointNodeEntry {
  nodeId: string;
  adapterId: string;
  cacheKey: string;
  completedAt: string;
  outputs: Record<string, CanonicalArtifact>;
  refs: Record<string, ArtifactRef>;
  /** Deterministic integrity digest binding `(nodeId, adapterId, cacheKey, outputs)` against tampering or swaps. */
  integrity?: string;
}

export interface SchedulerCheckpoint {
  schema: 'denovo.checkpoint/v1';
  runId: string;
  graphId: string;
  createdAt: string;
  status: RunLifecycleStatus;
  completedNodeIds: string[];
  unfinishedNodeIds: string[];
  nodes: Record<string, CheckpointNodeEntry>;
}

export interface MemoStats {
  entryCount: number;
  usedBytes: number;
  maxBytes: number;
  evictionCount: number;
}

export interface ExecutionSchedulerOptions {
  /** Byte budget for in-memory node result memoization (independent of OPFS/CacheStorage weight cache). */
  maxMemoBytes?: number;
  /** Maximum number of in-memory memoized node results retained (default 64). */
  maxMemoEntries?: number;
  /** Maximum number of completed/failed/cancelled runs retained in run history. */
  maxRunHistory?: number;
  /** Optional durable checkpoint storage adapter (e.g. IndexedDB or in-memory store). */
  checkpointStore?: CheckpointStorageAdapter;
  /** Idle delay (ms) after a run completes before auto-releasing the active GPU worker (default 60_000; 0 disables). */
  gpuIdleReleaseMs?: number;
}

export interface RunResult {
  runId?: string;
  source?: RunSource;
  status?: 'completed' | 'cancelled';
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>;
  refsByNode: Record<string, Record<string, ArtifactRef>>;
  completedNodeIds?: string[];
  unfinishedNodeIds?: string[];
  totalElapsedMs: number;
}

export interface RunOptions {
  bypassCache?: boolean;
  onNodeOutput?: (nodeId: string, outputs: Record<string, CanonicalArtifact>) => void;
  source?: RunSource;
  signal?: AbortSignal;
  runId?: string;
  resumeCheckpoint?: SchedulerCheckpoint | string;
  checkpointKey?: string;
  /** When true, return a partial RunResult on cancellation instead of throwing RunCancellationError. */
  returnPartialOnCancel?: boolean;
}

export class RunLeaseError extends Error {
  readonly activeRunId: string | null;
  readonly activeSource: RunSource | null;

  constructor(message: string, activeRunId: string | null, activeSource: RunSource | null) {
    super(message);
    this.name = 'RunLeaseError';
    this.activeRunId = activeRunId;
    this.activeSource = activeSource;
  }
}

export class RunCancellationError extends Error {
  readonly runId: string;
  readonly completedNodeIds: string[];
  readonly unfinishedNodeIds: string[];
  readonly outputsByNode: Record<string, Record<string, CanonicalArtifact>>;
  readonly refsByNode: Record<string, Record<string, ArtifactRef>>;
  readonly checkpoint: SchedulerCheckpoint;

  constructor(
    message: string,
    info: {
      runId: string;
      completedNodeIds: string[];
      unfinishedNodeIds: string[];
      outputsByNode: Record<string, Record<string, CanonicalArtifact>>;
      refsByNode: Record<string, Record<string, ArtifactRef>>;
      checkpoint: SchedulerCheckpoint;
    },
  ) {
    super(message);
    this.name = 'RunCancellationError';
    this.runId = info.runId;
    this.completedNodeIds = info.completedNodeIds;
    this.unfinishedNodeIds = info.unfinishedNodeIds;
    this.outputsByNode = info.outputsByNode;
    this.refsByNode = info.refsByNode;
    this.checkpoint = info.checkpoint;
  }
}

const DEFAULT_MAX_MEMO_BYTES = 64 * 1024 * 1024; // 64 MiB
const DEFAULT_MAX_RUN_HISTORY = 20;
const DEFAULT_GPU_IDLE_RELEASE_MS = 60_000; // 60 s idle auto-release for Safari/Metal VRAM hygiene

/**
 * Validate a worker or CPU adapter output map against `ADAPTER_REGISTRY[node.adapterId].outputs`
 * BEFORE caching or forwarding outputs downstream (Issue #4).
 */
export function validateAdapterOutputs(
  node: WorkflowNode,
  outputs: unknown,
): Record<string, CanonicalArtifact> {
  const manifest = ADAPTER_REGISTRY[node.adapterId];
  if (!manifest) {
    throw new Error(`Unknown adapter "${node.adapterId}" during output contract validation`);
  }
  if (!outputs || typeof outputs !== 'object' || Array.isArray(outputs)) {
    throw new Error(
      `Output contract violation on node "${node.id}" (${node.adapterId}): adapter returned a non-object outputs payload.`,
    );
  }
  const record = outputs as Record<string, unknown>;
  for (const [portName, portSpec] of Object.entries(manifest.outputs)) {
    const required = portSpec.required !== false;
    const value = record[portName];
    if (value === undefined || value === null) {
      if (required) {
        throw new Error(
          `Output contract violation on node "${node.id}" (${node.adapterId}): missing required output port "${portName}" (expected ${portSpec.type}).`,
        );
      }
      continue;
    }
    if (typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(
        `Output contract violation on node "${node.id}" (${node.adapterId}): output port "${portName}" is not a valid artifact object (expected ${portSpec.type}).`,
      );
    }
    const art = value as Partial<CanonicalArtifact>;
    if (art.kind !== portSpec.type) {
      throw new Error(
        `Output contract violation on node "${node.id}" (${node.adapterId}): output port "${portName}" returned kind "${String(art.kind)}", expected "${portSpec.type}".`,
      );
    }
    if (typeof art.id !== 'string' || art.id.trim().length === 0) {
      throw new Error(
        `Output contract violation on node "${node.id}" (${node.adapterId}): output port "${portName}" (${portSpec.type}) is missing a non-empty artifact id.`,
      );
    }
  }
  return record as Record<string, CanonicalArtifact>;
}

/**
 * Estimate retained JS heap bytes for a node's output artifact map so `ExecutionScheduler`
 * enforces a deterministic result-cache byte budget independent of OPFS weight caches.
 */
export function estimateArtifactOutputsBytes(outputs: Record<string, CanonicalArtifact>): number {
  const seen = new Set<unknown>();
  const walk = (val: unknown): number => {
    if (val === null || val === undefined) return 8;
    if (typeof val === 'string') return 24 + val.length * 2;
    if (typeof val === 'number' || typeof val === 'boolean') return 8;
    if (typeof val !== 'object') return 8;
    if (seen.has(val)) return 0;
    seen.add(val);
    if (ArrayBuffer.isView(val)) {
      return 32 + val.byteLength;
    }
    if (val instanceof ArrayBuffer) {
      return 32 + val.byteLength;
    }
    if (Array.isArray(val)) {
      let sum = 32;
      for (let i = 0; i < val.length; i++) sum += walk(val[i]);
      return sum;
    }
    let sum = 48;
    for (const [k, v] of Object.entries(val as Record<string, unknown>)) {
      sum += 16 + k.length * 2 + walk(v);
    }
    return sum;
  };
  return walk(outputs);
}

function encodeTypedJson(value: unknown): string {
  return JSON.stringify(value, (_key, v) => {
    if (typeof v === 'number') {
      if (Number.isNaN(v)) return { __num: 'NaN' };
      if (v === Infinity) return { __num: 'Infinity' };
      if (v === -Infinity) return { __num: '-Infinity' };
      return v;
    }
    if (v instanceof Float32Array) {
      return {
        __typedArray: 'Float32Array',
        data: Array.from(v, (x) => (Number.isNaN(x) ? 'NaN' : x === Infinity ? 'Inf' : x === -Infinity ? '-Inf' : x)),
      };
    }
    if (v instanceof Float64Array) {
      return {
        __typedArray: 'Float64Array',
        data: Array.from(v, (x) => (Number.isNaN(x) ? 'NaN' : x === Infinity ? 'Inf' : x === -Infinity ? '-Inf' : x)),
      };
    }
    if (v instanceof Int32Array) {
      return { __typedArray: 'Int32Array', data: Array.from(v) };
    }
    if (v instanceof Uint8Array) {
      return { __typedArray: 'Uint8Array', data: Array.from(v) };
    }
    return v;
  });
}

/** Compute a deterministic integrity digest binding a checkpoint node entry's metadata and outputs. */
export function computeCheckpointEntryIntegrity(
  entry: Pick<CheckpointNodeEntry, 'nodeId' | 'adapterId' | 'cacheKey' | 'outputs'>,
): string {
  const payload = encodeTypedJson({
    nodeId: entry.nodeId,
    adapterId: entry.adapterId,
    cacheKey: entry.cacheKey,
    outputs: entry.outputs,
  });
  // Two independent salted FNV-1a hashes (64 bits total) for fast deterministic integrity verification
  return `${fnv1a(`h1:${payload}`)}-${fnv1a(`h2:${payload.length}:${payload}`)}`;
}

/** Losslessly serialize a `SchedulerCheckpoint` (preserving Float32Array / Float64Array / Int32Array / Uint8Array / NaN). */
export function serializeCheckpoint(checkpoint: SchedulerCheckpoint): string {
  const stampedNodes: Record<string, CheckpointNodeEntry> = {};
  for (const [k, entry] of Object.entries(checkpoint.nodes)) {
    stampedNodes[k] = {
      ...entry,
      integrity: entry.integrity ?? computeCheckpointEntryIntegrity(entry),
    };
  }
  return encodeTypedJson({ ...checkpoint, nodes: stampedNodes });
}

/** Losslessly deserialize a `SchedulerCheckpoint` JSON string back into live typed-array artifacts. */
export function deserializeCheckpoint(raw: string): SchedulerCheckpoint {
  const decodeNum = (x: unknown): number => {
    if (x === 'NaN') return Number.NaN;
    if (x === 'Inf') return Number.POSITIVE_INFINITY;
    if (x === '-Inf') return Number.NEGATIVE_INFINITY;
    return Number(x);
  };
  const parsed = JSON.parse(raw, (_key, value) => {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      const rec = value as Record<string, unknown>;
      if (typeof rec.__num === 'string' && Object.keys(rec).length === 1) {
        if (rec.__num === 'NaN') return Number.NaN;
        if (rec.__num === 'Infinity') return Number.POSITIVE_INFINITY;
        if (rec.__num === '-Infinity') return Number.NEGATIVE_INFINITY;
      }
      if (typeof rec.__typedArray === 'string' && Array.isArray(rec.data)) {
        const nums = rec.data.map(decodeNum);
        switch (rec.__typedArray) {
          case 'Float32Array':
            return new Float32Array(nums);
          case 'Float64Array':
            return new Float64Array(nums);
          case 'Int32Array':
            return new Int32Array(nums);
          case 'Uint8Array':
            return new Uint8Array(nums);
        }
      }
    }
    return value;
  }) as Partial<SchedulerCheckpoint>;

  if (!parsed || parsed.schema !== 'denovo.checkpoint/v1' || typeof parsed.nodes !== 'object') {
    throw new Error('Invalid checkpoint format: expected schema "denovo.checkpoint/v1"');
  }
  return parsed as SchedulerCheckpoint;
}

/**
 * Compute which node IDs in `nextGraph` are invalidated (stale) relative to `prevGraph`
 * after editing upstream node parameters, adapters, or incoming wires.
 */
export function computeInvalidatedNodeIds(
  prevGraph: WorkflowGraph,
  nextGraph: WorkflowGraph,
): { staleNodeIds: string[]; validNodeIds: string[] } {
  const prevById = new Map(prevGraph.nodes.map((n) => [n.id, n]));
  const prevIncoming = new Map<string, string>();
  for (const n of prevGraph.nodes) {
    const inc = prevGraph.edges
      .filter((e) => e.toNode === n.id)
      .map((e) => `${e.fromNode}:${e.fromPort}->${e.toPort}`)
      .sort()
      .join('|');
    prevIncoming.set(n.id, inc);
  }

  const nextIncoming = new Map<string, string>();
  const children = new Map<string, string[]>();
  for (const n of nextGraph.nodes) {
    children.set(n.id, []);
    const inc = nextGraph.edges
      .filter((e) => e.toNode === n.id)
      .map((e) => `${e.fromNode}:${e.fromPort}->${e.toPort}`)
      .sort()
      .join('|');
    nextIncoming.set(n.id, inc);
  }
  for (const e of nextGraph.edges) {
    children.get(e.fromNode)?.push(e.toNode);
  }

  const stale = new Set<string>();
  const queue: string[] = [];

  for (const n of nextGraph.nodes) {
    const prev = prevById.get(n.id);
    const changed =
      !prev ||
      prev.adapterId !== n.adapterId ||
      canonicalJsonStringify(prev.params ?? {}) !== canonicalJsonStringify(n.params ?? {}) ||
      prevIncoming.get(n.id) !== nextIncoming.get(n.id);
    if (changed) {
      stale.add(n.id);
      queue.push(n.id);
    }
  }

  // Propagate staleness transitively to all downstream dependents
  while (queue.length > 0) {
    const cur = queue.shift()!;
    for (const childId of children.get(cur) ?? []) {
      if (!stale.has(childId)) {
        stale.add(childId);
        queue.push(childId);
      }
    }
  }

  const staleNodeIds: string[] = [];
  const validNodeIds: string[] = [];
  for (const n of nextGraph.nodes) {
    if (stale.has(n.id)) staleNodeIds.push(n.id);
    else validNodeIds.push(n.id);
  }
  return { staleNodeIds, validNodeIds };
}

function parseReferenceResidueMapping(
  raw: unknown,
  foldLen: number,
  refLen: number,
): { pairs: Array<[number, number]>; spec: ReferenceResidueMapping } | null {
  if (raw === undefined || raw === null || raw === '') return null;
  const pairs: Array<[number, number]> = [];
  let label: string | undefined;

  const assertStrictInt = (v: unknown, role: string): number => {
    if (typeof v !== 'number' || !Number.isInteger(v)) {
      throw new Error(
        `Invalid residueMapping: ${role} must be a strict finite integer (got ${String(v)}).`,
      );
    }
    return v;
  };

  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    for (const tok of trimmed.split(/[,\s;]+/).filter(Boolean)) {
      const m = /^(\d+)\s*(?::|->|=)\s*(\d+)$/.exec(tok);
      if (!m) {
        throw new Error(
          `Invalid residueMapping entry "${tok}". Expected 0-based "foldIdx:refIdx" pairs (e.g. "0:0,1:1,2:2").`,
        );
      }
      pairs.push([Number(m[1]), Number(m[2])]);
    }
  } else if (Array.isArray(raw)) {
    for (const entry of raw) {
      if (!Array.isArray(entry) || entry.length < 2) {
        throw new Error('Invalid residueMapping array: every entry must be a [foldIdx, refIdx] pair.');
      }
      pairs.push([
        assertStrictInt(entry[0], 'fold index'),
        assertStrictInt(entry[1], 'reference index'),
      ]);
    }
  } else if (typeof raw === 'object') {
    const obj = raw as ReferenceResidueMapping;
    label = obj.label;
    if (Array.isArray(obj.pairs)) {
      for (const entry of obj.pairs) {
        if (!Array.isArray(entry) || entry.length < 2) {
          throw new Error('Invalid residueMapping.pairs: every entry must be a [foldIdx, refIdx] pair.');
        }
        pairs.push([
          assertStrictInt(entry[0], 'fold index'),
          assertStrictInt(entry[1], 'reference index'),
        ]);
      }
    } else if (Array.isArray(obj.foldIndices) && Array.isArray(obj.referenceIndices)) {
      if (obj.foldIndices.length !== obj.referenceIndices.length) {
        throw new Error(
          `Invalid residueMapping: foldIndices (${obj.foldIndices.length}) and referenceIndices (${obj.referenceIndices.length}) must have equal length.`,
        );
      }
      for (let i = 0; i < obj.foldIndices.length; i++) {
        pairs.push([
          assertStrictInt(obj.foldIndices[i], 'fold index'),
          assertStrictInt(obj.referenceIndices[i], 'reference index'),
        ]);
      }
    } else {
      throw new Error(
        'Invalid residueMapping object: provide either "pairs" or both "foldIndices" and "referenceIndices".',
      );
    }
  } else {
    throw new Error('Invalid residueMapping parameter type.');
  }

  if (pairs.length < 3) {
    throw new Error(
      `Invalid residueMapping: at least 3 mapped residue pairs are required for Kabsch superposition (got ${pairs.length}).`,
    );
  }
  const seenFold = new Set<number>();
  const seenRef = new Set<number>();
  for (const [fi, ri] of pairs) {
    if (fi < 0 || fi >= foldLen) {
      throw new Error(
        `Invalid residueMapping: fold index ${fi} is out of bounds for fold length ${foldLen}.`,
      );
    }
    if (ri < 0 || ri >= refLen) {
      throw new Error(
        `Invalid residueMapping: reference index ${ri} is out of bounds for reference length ${refLen}.`,
      );
    }
    if (seenFold.has(fi)) {
      throw new Error(
        `Invalid residueMapping: duplicate fold index ${fi} (mapping must be 1-to-1 with >= 3 distinct pairs).`,
      );
    }
    if (seenRef.has(ri)) {
      throw new Error(
        `Invalid residueMapping: duplicate reference index ${ri} (mapping must be 1-to-1 with >= 3 distinct pairs).`,
      );
    }
    seenFold.add(fi);
    seenRef.add(ri);
  }

  return {
    pairs,
    spec: {
      pairs,
      foldIndices: pairs.map(([fi]) => fi),
      referenceIndices: pairs.map(([, ri]) => ri),
      ...(label ? { label } : {}),
    },
  };
}

interface ActiveGpuSlot {
  adapterId: string;
  worker: Worker;
  abortActive?: (err: Error) => void;
}

export class ExecutionScheduler {
  private activeGpu: ActiveGpuSlot | null = null;
  private gpuDisposalPromise: Promise<void> | null = null;
  private gpuIdleTimer: ReturnType<typeof setTimeout> | null = null;
  private gpuIdleReleaseMs: number;
  private memo = new Map<string, Record<string, CanonicalArtifact>>();
  private memoEntryBytes = new Map<string, number>();
  private memoKeysByNode = new Map<string, Set<string>>();
  private memoUsedBytes = 0;
  private maxMemoBytes: number;
  private maxMemoEntries: number;
  private memoEvictionCount = 0;
  private reqCounter = 1;
  private runSeq = 0;
  private running = false;
  private activeAbortController: AbortController | null = null;
  private runStateListeners = new Set<(state: ActiveRunState) => void>();
  private runHistory: RunHistoryEntry[] = [];
  private maxRunHistory: number;
  private checkpointStore?: CheckpointStorageAdapter;
  private latestCheckpoint: SchedulerCheckpoint | null = null;
  private activeRun: ActiveRunState = {
    runId: null,
    source: null,
    status: 'idle',
    graphId: null,
    activeNodeId: null,
    leaseHeld: false,
    completedNodeIds: [],
    unfinishedNodeIds: [],
    startedAt: null,
    finishedAt: null,
    error: null,
  };

  constructor(opts: ExecutionSchedulerOptions = {}) {
    this.maxMemoBytes =
      typeof opts.maxMemoBytes === 'number' && Number.isFinite(opts.maxMemoBytes) && opts.maxMemoBytes >= 0
        ? Math.floor(opts.maxMemoBytes)
        : DEFAULT_MAX_MEMO_BYTES;
    this.maxMemoEntries =
      typeof opts.maxMemoEntries === 'number' && Number.isFinite(opts.maxMemoEntries) && opts.maxMemoEntries >= 0
        ? Math.floor(opts.maxMemoEntries)
        : 64;
    this.maxRunHistory =
      typeof opts.maxRunHistory === 'number' && opts.maxRunHistory >= 1
        ? Math.floor(opts.maxRunHistory)
        : DEFAULT_MAX_RUN_HISTORY;
    this.checkpointStore = opts.checkpointStore;
    this.gpuIdleReleaseMs =
      typeof opts.gpuIdleReleaseMs === 'number' && opts.gpuIdleReleaseMs >= 0
        ? opts.gpuIdleReleaseMs
        : DEFAULT_GPU_IDLE_RELEASE_MS;
  }

  private clearGpuIdleTimer(): void {
    if (this.gpuIdleTimer !== null) {
      clearTimeout(this.gpuIdleTimer);
      this.gpuIdleTimer = null;
    }
  }

  private scheduleGpuIdleRelease(): void {
    this.clearGpuIdleTimer();
    if (!this.activeGpu || this.running) return;
    if (!Number.isFinite(this.gpuIdleReleaseMs) || this.gpuIdleReleaseMs <= 0) return;
    const timer = setTimeout(() => {
      this.gpuIdleTimer = null;
      if (!this.running && this.activeGpu) {
        void this.releaseActiveGpuInternal().catch(() => {});
      }
    }, this.gpuIdleReleaseMs);
    // Avoid pinning Node/Vitest event loops when scheduler sits idle
    if (typeof timer === 'object' && timer !== null && 'unref' in timer && typeof timer.unref === 'function') {
      timer.unref();
    }
    this.gpuIdleTimer = timer;
  }

  /** Clean up idle timers and terminate any active GPU worker (e.g. on page unmount). */
  async destroy(): Promise<void> {
    this.clearGpuIdleTimer();
    await this.releaseActiveGpuInternal('Scheduler destroyed');
  }

  getActiveGpuAdapter(): string | null {
    return this.activeGpu?.adapterId ?? null;
  }

  /** Return a snapshot of the canonical reactive run ownership state (shared by canvas and API runs). */
  getActiveRunState(): ActiveRunState {
    return {
      ...this.activeRun,
      completedNodeIds: [...this.activeRun.completedNodeIds],
      unfinishedNodeIds: [...this.activeRun.unfinishedNodeIds],
    };
  }

  /** Subscribe to canonical run ownership and lifecycle updates; returns an unsubscribe handle. */
  subscribeRunState(cb: (state: ActiveRunState) => void): () => void {
    this.runStateListeners.add(cb);
    cb(this.getActiveRunState());
    return () => {
      this.runStateListeners.delete(cb);
    };
  }

  private emitRunState(): void {
    const snap = this.getActiveRunState();
    for (const cb of this.runStateListeners) {
      try {
        cb(snap);
      } catch {
        /* ignore subscriber errors */
      }
    }
  }

  /** Return bounded history of prior completed/failed/cancelled runs. */
  getRunHistory(): RunHistoryEntry[] {
    return this.runHistory.map((h) => ({
      ...h,
      completedNodeIds: [...h.completedNodeIds],
      unfinishedNodeIds: [...h.unfinishedNodeIds],
      refsByNode: { ...h.refsByNode },
    }));
  }

  private pushRunHistory(entry: RunHistoryEntry): void {
    this.runHistory.push(entry);
    if (this.runHistory.length > this.maxRunHistory) {
      this.runHistory.splice(0, this.runHistory.length - this.maxRunHistory);
    }
  }

  /** Inspect in-memory result-cache byte budget stats (independent of OPFS weight cache). */
  getMemoStats(): MemoStats {
    return {
      entryCount: this.memo.size,
      usedBytes: this.memoUsedBytes,
      maxBytes: this.maxMemoBytes,
      evictionCount: this.memoEvictionCount,
    };
  }

  /** Update the in-memory result-cache byte budget, immediately evicting LRU entries if needed. */
  setMaxMemoBytes(maxBytes: number): void {
    this.maxMemoBytes = Math.max(0, Math.floor(maxBytes));
    this.evictMemoToBudget(0);
  }

  private getMemoEntry(key: string): Record<string, CanonicalArtifact> | undefined {
    const hit = this.memo.get(key);
    if (!hit) return undefined;
    // Refresh LRU recency by moving key to the end of Map insertion order
    this.memo.delete(key);
    this.memo.set(key, hit);
    const bytes = this.memoEntryBytes.get(key) ?? 0;
    this.memoEntryBytes.delete(key);
    this.memoEntryBytes.set(key, bytes);
    return hit;
  }

  private evictMemoToBudget(incomingBytes: number): void {
    while (
      (this.memoUsedBytes + incomingBytes > this.maxMemoBytes ||
        (incomingBytes > 0 && this.memo.size >= this.maxMemoEntries)) &&
      this.memo.size > 0
    ) {
      const oldestKey = this.memo.keys().next().value as string | undefined;
      if (oldestKey === undefined) break;
      const oldBytes = this.memoEntryBytes.get(oldestKey) ?? 0;
      this.memo.delete(oldestKey);
      this.memoEntryBytes.delete(oldestKey);
      this.memoUsedBytes = Math.max(0, this.memoUsedBytes - oldBytes);
      this.memoEvictionCount++;
    }
  }

  private setMemoEntry(key: string, outputs: Record<string, CanonicalArtifact>): void {
    const entryBytes = estimateArtifactOutputsBytes(outputs);
    if (this.memo.has(key)) {
      const prevBytes = this.memoEntryBytes.get(key) ?? 0;
      this.memo.delete(key);
      this.memoEntryBytes.delete(key);
      this.memoUsedBytes = Math.max(0, this.memoUsedBytes - prevBytes);
    }
    if (this.maxMemoBytes <= 0 || this.maxMemoEntries < 1 || entryBytes > this.maxMemoBytes) {
      // Never violate the configured byte budget with an oversized single entry
      this.evictMemoToBudget(0);
      return;
    }
    this.evictMemoToBudget(entryBytes);
    this.memo.set(key, outputs);
    this.memoEntryBytes.set(key, entryBytes);
    this.memoUsedBytes += entryBytes;
  }

  private recordNodeMemoKey(nodeId: string, cacheKey: string): void {
    let set = this.memoKeysByNode.get(nodeId);
    if (!set) {
      set = new Set();
      this.memoKeysByNode.set(nodeId, set);
    }
    set.add(cacheKey);
  }

  private deleteMemoKey(key: string): void {
    if (!this.memo.has(key)) return;
    const prevBytes = this.memoEntryBytes.get(key) ?? 0;
    this.memo.delete(key);
    this.memoEntryBytes.delete(key);
    this.memoUsedBytes = Math.max(0, this.memoUsedBytes - prevBytes);
  }

  private assertMaintenanceLeaseFree(action: string, force?: boolean): void {
    if (this.activeRun.leaseHeld && !force) {
      throw new RunLeaseError(
        `Refusing to ${action}: run "${this.activeRun.runId}" (source: ${this.activeRun.source ?? 'unknown'}) holds the active run lease. Cancel the active run first or pass { force: true }.`,
        this.activeRun.runId,
        this.activeRun.source,
      );
    }
  }

  clearMemo(opts?: { force?: boolean }): void {
    this.assertMaintenanceLeaseFree('clear result cache', opts?.force);
    this.memo.clear();
    this.memoEntryBytes.clear();
    this.memoKeysByNode.clear();
    this.memoUsedBytes = 0;
    this.latestCheckpoint = null;
  }

  /** Guarded maintenance action to release GPU VRAM and clear result cache (and optionally model weight caches). */
  async clearCache(opts?: { force?: boolean; clearModelWeights?: boolean }): Promise<void> {
    this.assertMaintenanceLeaseFree('clear cache', opts?.force);
    await this.releaseActiveGpu({ force: opts?.force });
    this.clearMemo({ force: opts?.force });
    if (opts?.clearModelWeights) {
      await clearAllModelCaches();
    }
  }

  /** Alias for `releaseActiveGpu()` guarded by the active run lease. */
  async releaseGpu(opts?: { force?: boolean }): Promise<void> {
    return this.releaseActiveGpu(opts);
  }

  /** Request explicit cancellation of the currently active run (if any). */
  cancelActiveRun(reason = 'Run cancelled by caller'): boolean {
    if (!this.running || !this.activeAbortController) return false;
    this.activeAbortController.abort(new Error(reason));
    return true;
  }

  /**
   * Mark nodes invalidated by an upstream graph edit and evict their stale entries from
   * BOTH `this.memo` and the active checkpoint (without erasing prior completed runs in `runHistory`).
   */
  invalidateEditedGraph(
    prevGraph: WorkflowGraph,
    nextGraph: WorkflowGraph,
  ): { staleNodeIds: string[]; validNodeIds: string[] } {
    const result = computeInvalidatedNodeIds(prevGraph, nextGraph);
    if (result.staleNodeIds.length > 0) {
      const staleSet = new Set(result.staleNodeIds);
      for (const nodeId of staleSet) {
        const keys = this.memoKeysByNode.get(nodeId);
        if (keys) {
          for (const k of keys) this.deleteMemoKey(k);
          this.memoKeysByNode.delete(nodeId);
        }
      }
      if (this.latestCheckpoint) {
        const nextNodes: Record<string, CheckpointNodeEntry> = {};
        for (const [nid, entry] of Object.entries(this.latestCheckpoint.nodes)) {
          if (staleSet.has(nid)) {
            if (entry.cacheKey) this.deleteMemoKey(entry.cacheKey);
          } else {
            nextNodes[nid] = entry;
          }
        }
        this.latestCheckpoint = {
          ...this.latestCheckpoint,
          completedNodeIds: this.latestCheckpoint.completedNodeIds.filter((id) => !staleSet.has(id)),
          unfinishedNodeIds: [
            ...new Set([
              ...this.latestCheckpoint.unfinishedNodeIds,
              ...result.staleNodeIds,
            ]),
          ],
          nodes: nextNodes,
        };
      }
    }
    return result;
  }

  /** Export the latest run checkpoint as a structured object (or `null` if no run has executed). */
  exportCheckpointObject(): SchedulerCheckpoint | null {
    if (!this.latestCheckpoint) return null;
    return deserializeCheckpoint(serializeCheckpoint(this.latestCheckpoint));
  }

  /** Export the latest run checkpoint as a durable JSON string (preserving typed arrays). */
  exportCheckpoint(): string {
    if (!this.latestCheckpoint) {
      throw new Error('No run checkpoint available to export');
    }
    return serializeCheckpoint(this.latestCheckpoint);
  }

  /**
   * Import a serialized or structured checkpoint into this scheduler so a subsequent `run()`
   * resumes ONLY unfinished or invalidated nodes (reusing verified checkpointed outputs).
   */
  importCheckpoint(raw: string | SchedulerCheckpoint): SchedulerCheckpoint {
    const ckpt = typeof raw === 'string' ? deserializeCheckpoint(raw) : raw;
    if (!ckpt || ckpt.schema !== 'denovo.checkpoint/v1' || typeof ckpt.nodes !== 'object') {
      throw new Error('Invalid checkpoint format: expected schema "denovo.checkpoint/v1"');
    }
    // Validate every imported checkpoint entry's output contract AND entry integrity BEFORE trusting or memoizing it
    const validatedEntries: CheckpointNodeEntry[] = [];
    for (const [nodeKey, entry] of Object.entries(ckpt.nodes)) {
      if (!entry || typeof entry !== 'object') {
        throw new Error(`Invalid checkpoint entry for node "${nodeKey}"`);
      }
      const nodeId = entry.nodeId || nodeKey;
      validateAdapterOutputs(
        { id: nodeId, adapterId: entry.adapterId, params: {} },
        entry.outputs,
      );
      if (!entry.integrity || typeof entry.integrity !== 'string') {
        throw new Error(
          `Checkpoint integrity violation for node "${nodeId}": missing integrity digest.`,
        );
      }
      const expectedDigest = computeCheckpointEntryIntegrity({
        nodeId,
        adapterId: entry.adapterId,
        cacheKey: entry.cacheKey,
        outputs: entry.outputs,
      });
      if (entry.integrity !== expectedDigest) {
        throw new Error(
          `Checkpoint integrity violation for node "${nodeId}": entry digest mismatch (tampered or swapped outputs).`,
        );
      }
      validatedEntries.push(entry);
    }
    this.latestCheckpoint = ckpt;
    for (const entry of validatedEntries) {
      if (entry.cacheKey && entry.outputs) {
        this.setMemoEntry(entry.cacheKey, entry.outputs);
        this.recordNodeMemoKey(entry.nodeId, entry.cacheKey);
      }
    }
    return ckpt;
  }

  /** Persist the latest checkpoint via the configured `checkpointStore` adapter. */
  async saveCheckpointToStore(key: string): Promise<void> {
    if (!this.checkpointStore) {
      throw new Error('No checkpointStore configured on ExecutionScheduler');
    }
    await this.checkpointStore.save(key, this.exportCheckpoint());
  }

  /** Load and import a checkpoint via the configured `checkpointStore` adapter. */
  async loadCheckpointFromStore(key: string): Promise<SchedulerCheckpoint | null> {
    if (!this.checkpointStore) {
      throw new Error('No checkpointStore configured on ExecutionScheduler');
    }
    const raw = await this.checkpointStore.load(key);
    if (!raw) return null;
    return this.importCheckpoint(raw);
  }

  /** Deterministically dispose & terminate the active GPU worker to reclaim WebGPU VRAM.
   *  Respects the active run lease unless `{ force: true }` is explicitly passed. */
  async releaseActiveGpu(opts?: { force?: boolean }): Promise<void> {
    this.assertMaintenanceLeaseFree('release active GPU worker', opts?.force);
    return this.releaseActiveGpuInternal();
  }

  private async waitForPendingGpuDisposal(): Promise<void> {
    if (this.gpuDisposalPromise) {
      await this.gpuDisposalPromise;
    }
  }

  private async releaseActiveGpuInternal(abortMessage?: string): Promise<void> {
    this.clearGpuIdleTimer();
    const slot = this.activeGpu;
    if (!slot) {
      await this.waitForPendingGpuDisposal();
      return;
    }
    this.activeGpu = null;

    if (slot.abortActive) {
      const abort = slot.abortActive;
      slot.abortActive = undefined;
      abort(
        new Error(
          abortMessage ?? `GPU worker "${slot.adapterId}" was released while running`,
        ),
      );
    }

    const prevPending = this.gpuDisposalPromise;
    const disposal = (async () => {
      if (prevPending) {
        await prevPending.catch(() => {});
      }
      await new Promise<void>((resolve) => {
        const id = this.reqCounter++;
        const finish = () => {
          clearTimeout(timer);
          slot.worker.removeEventListener('message', onMsg);
          try {
            slot.worker.terminate();
          } catch {
            /* ignore */
          }
          resolve();
        };
        const timer = setTimeout(finish, 500);
        const onMsg = (ev: MessageEvent) => {
          if (ev.data?.id === id && ev.data?.type === 'disposed') {
            finish();
          }
        };
        slot.worker.addEventListener('message', onMsg);
        try {
          slot.worker.postMessage({ id, type: 'dispose' });
        } catch {
          finish();
        }
      });
    })();

    this.gpuDisposalPromise = disposal.finally(() => {
      if (this.gpuDisposalPromise === disposal) {
        this.gpuDisposalPromise = null;
      }
    });
    await this.gpuDisposalPromise;
  }

  private spawnWorker(adapterId: string): Worker {
    switch (adapterId) {
      case 'web-protein-mpnn':
        return new MpnnWorker();
      case 'localfold-af2':
        return new LocalFoldWorker();
      case 'rfd4-proteina':
        return new Rfd4Worker();
      case 'genie3':
        return new Genie3Worker();
      case 'thermompnn':
        return new ThermoMpnnWorker();
      case 'ligandmpnn':
        return new LigandMpnnWorker();
      case 'lasermpnn':
        return new LaserMpnnWorker();
      case 'antifold-esmif1':
        return new AntiFoldWorker();
      default:
        throw new Error(`No GPU worker registered for adapter "${adapterId}"`);
    }
  }

  private async runGpuNode(
    node: WorkflowNode,
    inputs: Record<string, CanonicalArtifact>,
    onProgress: (ev: NodeProgressEvent) => void,
    signal?: AbortSignal,
  ): Promise<{ outputs: Record<string, CanonicalArtifact>; elapsedMs: number }> {
    if (signal?.aborted) {
      throw new Error('Workflow run cancelled');
    }

    // Never spawn or reuse a GPU worker while a prior worker disposal is still completing
    await this.waitForPendingGpuDisposal();

    // Enforce Single-Active-GPU-Worker Mutex (warm-1-LRU)
    if (this.activeGpu && this.activeGpu.adapterId !== node.adapterId) {
      onProgress({
        nodeId: node.id,
        status: 'loading',
        phase: 'vram',
        progress: 0,
        detail: `Releasing ${this.activeGpu.adapterId} VRAM before starting ${node.label ?? node.adapterId}...`,
      });
      await this.releaseActiveGpuInternal();
    }

    if (signal?.aborted) {
      throw new Error('Workflow run cancelled');
    }

    onProgress({
      nodeId: node.id,
      status: 'loading',
      phase: 'init',
      progress: 0.02,
      detail: `Starting ${node.label ?? node.adapterId} worker...`,
    });

    if (!this.activeGpu) {
      const worker = this.spawnWorker(node.adapterId);
      this.activeGpu = { adapterId: node.adapterId, worker };
    }

    const slot = this.activeGpu;
    const worker = slot.worker;
    const id = this.reqCounter++;

    return await new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        worker.removeEventListener('message', onMessage);
        worker.removeEventListener('error', onError);
        signal?.removeEventListener('abort', onAbortSignal);
        if (slot.abortActive === abort) {
          slot.abortActive = undefined;
        }
      };
      const abort = (err: Error) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(err);
      };
      const onAbortSignal = () => {
        void this.releaseActiveGpuInternal('Workflow run cancelled');
        abort(new Error('Workflow run cancelled'));
      };
      const onMessage = (ev: MessageEvent) => {
        const m = ev.data;
        if (m?.id !== id) return;
        if (m.type === 'progress') {
          onProgress({
            nodeId: node.id,
            status: m.phase === 'weights' || m.phase === 'init' ? 'loading' : 'running',
            phase: m.phase,
            progress: m.progress,
            detail: m.detail,
          });
        } else if (m.type === 'result') {
          if (settled) return;
          settled = true;
          cleanup();
          resolve({ outputs: m.outputs, elapsedMs: m.elapsedMs ?? 0 });
        } else if (m.type === 'error') {
          if (settled) return;
          settled = true;
          cleanup();
          reject(new Error(m.message || `Worker ${node.adapterId} failed`));
        }
      };
      const onError = (err: ErrorEvent) => {
        if (settled) return;
        settled = true;
        cleanup();
        if (this.activeGpu === slot) {
          this.activeGpu = null;
        }
        try {
          worker.terminate();
        } catch {
          /* ignore */
        }
        reject(new Error(err.message || `Worker ${node.adapterId} crashed`));
      };

      slot.abortActive = abort;
      worker.addEventListener('message', onMessage);
      worker.addEventListener('error', onError);
      if (signal) {
        if (signal.aborted) {
          onAbortSignal();
          return;
        }
        signal.addEventListener('abort', onAbortSignal, { once: true });
      }
      try {
        // Plain-object clone strips any reactive Proxy (e.g. Svelte 5 $state) before postMessage.
        const plainParams = JSON.parse(JSON.stringify(node.params ?? {}));
        worker.postMessage({ id, type: 'run', params: plainParams, inputs });
      } catch (err) {
        abort(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  private async runCpuNode(
    node: WorkflowNode,
    inputs: Record<string, CanonicalArtifact>,
    onProgress?: (ev: NodeProgressEvent) => void,
    upstreamLineageHint?: string,
  ): Promise<Record<string, CanonicalArtifact>> {
    if (node.adapterId === 'pdb-input') {
      const art = await materializePdbInputNode(node, onProgress);
      return { structure: art };
    }

    if (node.adapterId === 'sequence-input') {
      const seqSet = materializeSequenceInputNode(node);
      return { sequences: seqSet };
    }

    if (node.adapterId === 'top-k-filter') {
      const seqIn = inputs.sequences as SequenceSet | undefined;
      if (!seqIn || seqIn.kind !== 'SequenceSet') {
        throw new Error('top-k-filter requires an upstream SequenceSet input');
      }
      const modeRaw = String(node.params.mode ?? 'per-backbone');
      const mode: 'per-backbone' | 'global' = modeRaw === 'global' ? 'global' : 'per-backbone';
      const rawTopK = Number(node.params.topK ?? 2);
      const topK = Number.isFinite(rawTopK) ? Math.max(1, Math.min(16, Math.round(rawTopK))) : 2;
      const rawMaxTotal = Number(node.params.maxTotal ?? 8);
      const maxTotal = Number.isFinite(rawMaxTotal)
        ? Math.max(1, Math.min(64, Math.round(rawMaxTotal)))
        : 8;
      const metricRaw = String(node.params.metric ?? 'score');
      const metric: 'score' | 'globalScore' = metricRaw === 'globalScore' ? 'globalScore' : 'score';
      const directionRaw = String(node.params.direction ?? 'min');
      const direction: 'asc' | 'desc' =
        directionRaw === 'max' || directionRaw === 'desc' ? 'desc' : 'asc';
      const directionParam: 'min' | 'max' = direction === 'desc' ? 'max' : 'min';

      const setParent = (seqIn.sourceStructureId ?? seqIn.parentStructureId ?? '').trim();
      const hasSetParent = setParent.length > 0 && setParent.toLowerCase() !== 'none';
      const groupOrder = new Map<string, number>();
      const origIndexByKey = new Map<string, number>();
      const origCandByKey = new Map<string, SequenceCandidate>();
      const usedKeys = new Set<string>();

      const members: CollectionMember<SequenceCandidate>[] = seqIn.candidates.map((cand, idx) => {
        const rawDesignAnchor = (
          cand.designKey ??
          cand.sourceStructureId ??
          (hasSetParent ? setParent : '')
        ).trim();
        const hasDesign = rawDesignAnchor.length > 0 && rawDesignAnchor.toLowerCase() !== 'none';
        const designKey = hasDesign ? rawDesignAnchor : undefined;
        let designIndex = cand.designIndex;
        if (designKey !== undefined && designIndex === undefined) {
          if (!groupOrder.has(designKey)) groupOrder.set(designKey, groupOrder.size);
          designIndex = groupOrder.get(designKey);
        }
        let memberKey = (cand.sequenceKey ?? cand.key ?? cand.id).trim() || `seq-${idx}`;
        if (usedKeys.has(memberKey)) {
          memberKey = `${memberKey}::${idx}`;
        }
        usedKeys.add(memberKey);
        origIndexByKey.set(memberKey, idx);
        origCandByKey.set(memberKey, cand);
        const ordinal =
          typeof cand.sampleIndex === 'number' && Number.isInteger(cand.sampleIndex) && cand.sampleIndex >= 0
            ? cand.sampleIndex
            : idx;
        const candSourceId = (cand.sourceStructureId ?? (hasSetParent ? setParent : '')).trim();
        const hasSourceId = candSourceId.length > 0 && candSourceId.toLowerCase() !== 'none';
        return {
          key: memberKey,
          ordinal,
          memberKind: 'SequenceCandidate',
          lineage: {
            ...(designKey ? { designKey, designIndex } : {}),
            sequenceKey: memberKey,
            sequenceIndex: ordinal,
            ...(hasSourceId
              ? {
                  sourceStructure: {
                    artifactId: candSourceId,
                    payloadDigest: '',
                    ...(designKey ? { designKey } : {}),
                  },
                }
              : {}),
            parentKeys: designKey ? [designKey] : [],
          },
          payload: cand,
          summary: {
            label: cand.id,
            residueCount: cand.sequence.length,
            sequenceDigest: sha256Hex(cand.sequence),
            score: cand.score,
            globalScore: cand.globalScore,
          },
        };
      });

      const plan = planTopKFilter(members, {
        mode,
        topK,
        maxTotal,
        metric,
        direction,
      });

      const selectedCandidates: SequenceCandidate[] = plan.selected.map((s) => s.member.payload!);
      const excludedCandidates: ExcludedCandidateRecord[] = plan.excluded.map((ex, exIdx) => {
        const orig = origCandByKey.get(ex.key);
        return {
          key: ex.key,
          id: orig?.id ?? ex.key,
          index: origIndexByKey.get(ex.key) ?? selectedCandidates.length + exIdx,
          reason: ex.reason ?? `Excluded by top-k-filter (${mode}, topK=${topK}, maxTotal=${maxTotal})`,
        };
      });
      const itemOutcomes: CollectionItemOutcome[] = [
        ...plan.selected.map((s): CollectionItemOutcome => ({
          key: s.key,
          status: 'completed',
        })),
        ...excludedCandidates.map((ex): CollectionItemOutcome => ({
          key: ex.key,
          status: 'excluded',
          reason: ex.reason,
        })),
      ];

      const distinctSelectedSources = [
        ...new Set(
          selectedCandidates
            .map((c) => c.sourceStructureId?.trim() ?? '')
            .filter((id) => id.length > 0 && id.toLowerCase() !== 'none'),
        ),
      ];

      const filteredSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `topk-${seqIn.id}-${fnv1a(canonicalJsonStringify({ mode, topK, maxTotal, metric, direction, planDigest: plan.planDigest }))}`,
        parentStructureId: seqIn.parentStructureId,
        ...(seqIn.sourceStructureId ? { sourceStructureId: seqIn.sourceStructureId } : {}),
        ...(distinctSelectedSources.length > 1
          ? { sourceStructureIds: distinctSelectedSources }
          : seqIn.sourceStructureIds
            ? { sourceStructureIds: [...seqIn.sourceStructureIds] }
            : {}),
        sourceModelId: seqIn.sourceModelId,
        ...(seqIn.collectionKind ? { collectionKind: seqIn.collectionKind } : {}),
        ...(seqIn.lockedChains ? { lockedChains: [...seqIn.lockedChains] } : {}),
        ...(seqIn.chainRoles ? { chainRoles: { ...seqIn.chainRoles } } : {}),
        ...(excludedCandidates.length > 0 ? { excludedCandidates } : {}),
        itemOutcomes,
        filterMetadata: {
          mode,
          topK,
          maxTotal: plan.policy.maxTotal,
          metric,
          direction: directionParam,
          totalInput: plan.summary.totalInput,
          selectedCount: plan.summary.selectedCount,
          excludedCount: plan.summary.excludedCount,
          description: plan.summary.description,
        },
        candidates: selectedCandidates,
      };
      return { sequences: filteredSet };
    }

    if (node.adapterId === 'designability-scorer') {
      const structure = inputs.structure as StructureArtifact | undefined;
      const folds = inputs.folds as FoldedStructure | undefined;
      if (!structure || !folds) {
        throw new Error('designability-scorer requires both structure and folds inputs');
      }
      const scThresh = Number(node.params.scRmsdThreshold ?? 2.0);
      const plddtThresh = Number(node.params.plddtThreshold ?? 70);
      const motifThresh = Number(node.params.motifRmsdThreshold ?? 1.0);
      const ipaeThresh = Number(node.params.ipaeThreshold ?? 10.0);
      const ipsaeThresh = Number(node.params.ipsaeThreshold ?? 0);
      const targetAlignedBinderThresh = Number(
        node.params.targetAlignedBinderRmsdThreshold ?? Math.max(scThresh * 2, 4.0),
      );

      const allowExternalReference = Boolean(node.params.allowExternalReference);
      const strictLineage = Boolean(node.params.strictLineage);
      const rawMappingParam = node.params.residueMapping ?? node.params.referenceMapping;

      const normalizeLineageId = (val: string | undefined | null): string => {
        const s = String(val ?? '').trim();
        return s && s.toLowerCase() !== 'none' ? s : '';
      };

      // Helper: precompute motif pairs, genMotifRmsd, and designedStructureText for a specific reference backbone
      const prepareRefScoringContext = (ref: StructureArtifact) => {
        const refLen = Math.floor(ref.caCoords.length / 3);
        const motifPairs: Array<{ idx: number; coord: [number, number, number] }> = [];
        if (
          Array.isArray(ref.motifIndices) &&
          Array.isArray(ref.nativeMotifCoords) &&
          ref.motifIndices.length > 0 &&
          ref.nativeMotifCoords.length > 0
        ) {
          const limit = Math.min(ref.motifIndices.length, ref.nativeMotifCoords.length);
          for (let k = 0; k < limit; k++) {
            const idx = ref.motifIndices[k];
            const nc = ref.nativeMotifCoords[k];
            if (
              Number.isInteger(idx) &&
              idx >= 0 &&
              idx < refLen &&
              Array.isArray(nc) &&
              nc.length >= 3 &&
              Number.isFinite(nc[0]) &&
              Number.isFinite(nc[1]) &&
              Number.isFinite(nc[2])
            ) {
              motifPairs.push({ idx, coord: [nc[0], nc[1], nc[2]] });
            }
          }
        }
        let nativeMotifCa: Float32Array | null = null;
        let genMotifRmsd: number | undefined;
        if (motifPairs.length >= 3) {
          nativeMotifCa = new Float32Array(motifPairs.length * 3);
          const genMotifCa = new Float32Array(motifPairs.length * 3);
          for (let k = 0; k < motifPairs.length; k++) {
            const { idx, coord } = motifPairs[k];
            nativeMotifCa[k * 3 + 0] = coord[0];
            nativeMotifCa[k * 3 + 1] = coord[1];
            nativeMotifCa[k * 3 + 2] = coord[2];
            genMotifCa[k * 3 + 0] = ref.caCoords[idx * 3 + 0];
            genMotifCa[k * 3 + 1] = ref.caCoords[idx * 3 + 1];
            genMotifCa[k * 3 + 2] = ref.caCoords[idx * 3 + 2];
          }
          const genAlign = kabschAlign(genMotifCa, nativeMotifCa);
          if (Number.isFinite(genAlign.rmsd)) {
            genMotifRmsd = genAlign.rmsd;
          }
        }
        const hasNonProtein = Boolean(
          ref.diagnostics?.hasLigands || ref.diagnostics?.hasNucleicAcids,
        );
        const designedStructureText =
          ref.cifText && (hasNonProtein || !ref.pdbText) ? ref.cifText : ref.pdbText;
        return {
          ref,
          refLen,
          refDisplayName: ref.label || ref.id,
          motifPairs,
          nativeMotifCa,
          genMotifRmsd,
          designedStructureText,
        };
      };

      // Helper: score a single FoldedCandidate against an already-resolved StructureArtifact reference
      const scoreFoldAgainstRef = (
        f: FoldedCandidate,
        ctx: ReturnType<typeof prepareRefScoringContext>,
        lineageOpts: {
          rowLineage: string;
          rowSelfConsistencyValid: boolean;
          rowLineageMismatch: boolean;
          rowReferenceRelation: ReferenceRelation;
          rowReferenceLabel: string;
          rowReferenceMode: 'self-consistency' | 'external-reference' | 'lineage-mismatch';
          parsedMapping: ReturnType<typeof parseReferenceResidueMapping>;
        },
      ): DesignabilityRow => {
        const { ref, refLen, motifPairs, nativeMotifCa, genMotifRmsd, designedStructureText } = ctx;
        const {
          rowLineage,
          rowSelfConsistencyValid,
          rowLineageMismatch,
          rowReferenceRelation,
          rowReferenceLabel,
          rowReferenceMode,
          parsedMapping,
        } = lineageOpts;
        const foldLen = Math.floor(f.caCoords.length / 3);

        let align;
        let alignedResidueCount: number;
        if (parsedMapping) {
          const nMapped = parsedMapping.pairs.length;
          const foldMappedCa = new Float32Array(nMapped * 3);
          const refMappedCa = new Float32Array(nMapped * 3);
          for (let k = 0; k < nMapped; k++) {
            const [fi, ri] = parsedMapping.pairs[k];
            foldMappedCa[k * 3 + 0] = f.caCoords[fi * 3 + 0];
            foldMappedCa[k * 3 + 1] = f.caCoords[fi * 3 + 1];
            foldMappedCa[k * 3 + 2] = f.caCoords[fi * 3 + 2];
            refMappedCa[k * 3 + 0] = ref.caCoords[ri * 3 + 0];
            refMappedCa[k * 3 + 1] = ref.caCoords[ri * 3 + 1];
            refMappedCa[k * 3 + 2] = ref.caCoords[ri * 3 + 2];
          }
          align = kabschAlign(foldMappedCa, refMappedCa);
          alignedResidueCount = nMapped;
        } else {
          if (f.caCoords.length !== ref.caCoords.length) {
            throw new Error(
              `Length mismatch in designability-scorer: refolded "${f.sequenceId}" has ${foldLen} residues, ` +
                `but designed structure "${ref.label}" has ${refLen} residues.`,
            );
          }
          align = kabschAlign(f.caCoords, ref.caCoords);
          alignedResidueCount = refLen;
        }
        const alignedFoldPdb = applyTransformToPdb(f.pdbText, align.rotation, align.translation);

        let foldResNums: number[] | undefined =
          f.resNums && f.resNums.length === foldLen ? f.resNums : undefined;
        let foldChainIds = f.chainIds && f.chainIds.length === foldLen ? f.chainIds : undefined;
        let foldInsCodes: string[] | undefined;
        if ((!foldResNums || !foldChainIds) && f.pdbText) {
          try {
            const parsedFold = parseStructureText(f.pdbText);
            if (parsedFold.nResidues === foldLen) {
              if (!foldResNums) foldResNums = parsedFold.resNums;
              if (!foldChainIds) foldChainIds = parsedFold.chainIds;
              foldInsCodes = parsedFold.insCodes;
            }
          } catch {
            /* keep f.chainIds without resNums if pdbText is synthetic */
          }
        }
        const geom = computeGeometryStats(f.caCoords, foldChainIds, foldResNums, foldInsCodes);

        // Multi-chain binder & interface evaluation (#8) — only applicable for full 1:1 equal-length complexes
        const isFullOneToOne = !parsedMapping && foldLen === refLen;
        if (isFullOneToOne) {
          const hasStructChainInfo =
            (ref.chainLengths && ref.chainLengths.length > 0) ||
            (ref.chainIds && ref.chainIds.length === refLen);
          const hasFoldChainInfo =
            (f.chainLengths && f.chainLengths.length > 0) ||
            (foldChainIds && foldChainIds.length === foldLen);
          if (hasStructChainInfo && hasFoldChainInfo) {
            const structRanges = resolveScorerChainRanges(
              refLen,
              ref.chainLengths,
              ref.chainIds,
            );
            const foldRanges = resolveScorerChainRanges(foldLen, f.chainLengths, foldChainIds);
            const stoichMatches =
              structRanges.length === foldRanges.length &&
              structRanges.every((sr, idx) => sr.length === foldRanges[idx].length);
            if (!stoichMatches) {
              throw new Error(
                `Chain stoichiometry mismatch in designability-scorer: refolded "${f.sequenceId}" has ${foldRanges.length} chain(s) [${foldRanges.map((r) => r.length).join(', ')}], ` +
                  `but reference structure "${ref.label}" has ${structRanges.length} chain(s) [${structRanges.map((r) => r.length).join(', ')}].`,
              );
            }
          }
        }
        const chainRanges = isFullOneToOne
          ? resolveScorerChainRanges(
              refLen,
              ref.chainLengths ?? f.chainLengths,
              ref.chainIds ?? foldChainIds,
            )
          : [];
        let ipae: number | undefined;
        let ipsae: number | undefined;
        let ipsaeMin: number | undefined;
        let binderPlddt: number | undefined;
        let targetPlddt: number | undefined;
        let binderScRmsd: number | undefined;
        let targetAlignedBinderRmsd: number | undefined;

        if (isFullOneToOne && chainRanges.length > 1) {
          const split = splitBinderAndTargetIndices(
            chainRanges,
            ref.chainRoles,
            ref.lockedChains,
          );
          const { binderIndices, targetIndices, explicitRoleSplit } = split;

          if (f.pae && f.pae.length >= refLen * refLen) {
            const paeDim = Math.round(Math.sqrt(f.pae.length));
            if (paeDim >= refLen && paeDim * paeDim === f.pae.length) {
              ipae = computeInterfacePae(
                f.pae,
                paeDim,
                refLen,
                chainRanges,
                explicitRoleSplit ? binderIndices : undefined,
                explicitRoleSplit ? targetIndices : undefined,
              );
            }
          }

          if (explicitRoleSplit) {
            if (f.plddt.length === refLen) {
              if (binderIndices.length > 0) {
                let sum = 0;
                for (const idx of binderIndices) sum += f.plddt[idx];
                const mean = sum / binderIndices.length;
                if (Number.isFinite(mean)) binderPlddt = mean;
              }
              if (targetIndices.length > 0) {
                let sum = 0;
                for (const idx of targetIndices) sum += f.plddt[idx];
                const mean = sum / targetIndices.length;
                if (Number.isFinite(mean)) targetPlddt = mean;
              }
            }

            if (binderIndices.length >= 3) {
              const foldBinderCa = gatherCoordsByIndices(f.caCoords, binderIndices);
              const structBinderCa = gatherCoordsByIndices(ref.caCoords, binderIndices);
              const bAlign = kabschAlign(foldBinderCa, structBinderCa);
              if (Number.isFinite(bAlign.rmsd)) {
                binderScRmsd = bAlign.rmsd;
              }
            }

            if (targetIndices.length >= 3 && binderIndices.length >= 1) {
              const foldTargetCa = gatherCoordsByIndices(f.caCoords, targetIndices);
              const structTargetCa = gatherCoordsByIndices(ref.caCoords, targetIndices);
              const tAlign = kabschAlign(foldTargetCa, structTargetCa);
              if (Number.isFinite(tAlign.rmsd)) {
                const { rotation: R, translation: t } = tAlign;
                let sqSum = 0;
                for (let k = 0; k < binderIndices.length; k++) {
                  const off = binderIndices[k] * 3;
                  const x = f.caCoords[off + 0];
                  const y = f.caCoords[off + 1];
                  const z = f.caCoords[off + 2];
                  const dx = R[0] * x + R[1] * y + R[2] * z + t[0] - ref.caCoords[off + 0];
                  const dy = R[3] * x + R[4] * y + R[5] * z + t[1] - ref.caCoords[off + 1];
                  const dz = R[6] * x + R[7] * y + R[8] * z + t[2] - ref.caCoords[off + 2];
                  sqSum += dx * dx + dy * dy + dz * dz;
                }
                const tabRmsd = Math.sqrt(sqSum / binderIndices.length);
                if (Number.isFinite(tabRmsd)) {
                  targetAlignedBinderRmsd = tabRmsd;
                }
              }
            }
          }
        }

        const ipsaeChainRanges =
          chainRanges.length > 1
            ? chainRanges
            : resolveScorerChainRanges(foldLen, f.chainLengths, foldChainIds);
        if (ipsaeChainRanges.length > 1 && f.pae && f.pae.length >= foldLen * foldLen) {
          const proteinPae = sliceProteinPae(f.pae, foldLen);
          if (proteinPae) {
            const perResChains = expandChainRangesToIds(ipsaeChainRanges);
            const ipsaeRes = computeIpsae(proteinPae, perResChains, 10);
            if (ipsaeRes) {
              ipsae = ipsaeRes.ipsae;
              ipsaeMin = ipsaeRes.ipsaeMin;
            }
          }
        }
        if (ipsae === undefined && typeof f.ipsae === 'number' && Number.isFinite(f.ipsae)) {
          ipsae = f.ipsae;
          if (typeof f.ipsaeMin === 'number' && Number.isFinite(f.ipsaeMin)) {
            ipsaeMin = f.ipsaeMin;
          }
        }

        let refoldMotifRmsd: number | undefined;
        if (isFullOneToOne && nativeMotifCa && motifPairs.length >= 3) {
          const refoldMotifCa = new Float32Array(motifPairs.length * 3);
          for (let k = 0; k < motifPairs.length; k++) {
            const off = motifPairs[k].idx * 3;
            refoldMotifCa[k * 3 + 0] = f.caCoords[off + 0];
            refoldMotifCa[k * 3 + 1] = f.caCoords[off + 1];
            refoldMotifCa[k * 3 + 2] = f.caCoords[off + 2];
          }
          const refoldAlign = kabschAlign(refoldMotifCa, nativeMotifCa);
          if (Number.isFinite(refoldAlign.rmsd)) {
            refoldMotifRmsd = refoldAlign.rmsd;
          }
        }

        const confVal = resolveFoldConfidence(f);
        const confidenceAvailable = confVal !== null;
        const effectiveMeanPlddt = confVal !== null ? confVal : Number.NaN;

        const failedGates: string[] = [];
        if (!rowSelfConsistencyValid || rowLineageMismatch) {
          failedGates.push('External reference (not self-consistency)');
        }
        if (!confidenceAvailable) {
          failedGates.push('Confidence unavailable (no pLDDT)');
        } else if (effectiveMeanPlddt < plddtThresh) {
          failedGates.push(`pLDDT ${effectiveMeanPlddt.toFixed(1)} < ${plddtThresh}`);
        }
        if (!Number.isFinite(align.rmsd)) {
          failedGates.push('scRMSD unavailable');
        } else if (align.rmsd >= scThresh) {
          failedGates.push(`scRMSD ${align.rmsd.toFixed(2)} Å ≥ ${scThresh.toFixed(1)} Å`);
        }
        if (binderScRmsd !== undefined && binderScRmsd >= scThresh) {
          failedGates.push(`Binder scRMSD ${binderScRmsd.toFixed(2)} Å ≥ ${scThresh.toFixed(1)} Å`);
        }
        if (binderPlddt !== undefined && binderPlddt < plddtThresh) {
          failedGates.push(`Binder pLDDT ${binderPlddt.toFixed(1)} < ${plddtThresh}`);
        }
        if (targetAlignedBinderRmsd !== undefined && targetAlignedBinderRmsd >= targetAlignedBinderThresh) {
          failedGates.push(
            `Binder RMSD (target-aligned) ${targetAlignedBinderRmsd.toFixed(2)} Å ≥ ${targetAlignedBinderThresh.toFixed(1)} Å`,
          );
        }
        if (ipae !== undefined && ipae >= ipaeThresh) {
          failedGates.push(`iPAE ${ipae.toFixed(1)} Å ≥ ${ipaeThresh.toFixed(1)} Å`);
        }
        if (
          Number.isFinite(ipsaeThresh) &&
          ipsaeThresh > 0 &&
          ipsae !== undefined &&
          ipsae < ipsaeThresh
        ) {
          failedGates.push(`ipSAE ${ipsae.toFixed(2)} < ${ipsaeThresh.toFixed(2)}`);
        }
        if (refoldMotifRmsd !== undefined && refoldMotifRmsd >= motifThresh) {
          failedGates.push(`Motif RMSD ${refoldMotifRmsd.toFixed(2)} Å ≥ ${motifThresh.toFixed(1)} Å`);
        }
        const designable = failedGates.length === 0;

        const designKey = ref.designKey ?? f.designKey;
        const designIndex = ref.designIndex ?? f.designIndex;

        return {
          structureId: ref.id,
          sequenceId: f.sequenceId,
          candidateKey: f.key ?? f.sequenceId,
          ...(designKey ? { designKey } : {}),
          ...(designIndex !== undefined ? { designIndex } : {}),
          ...(f.sequenceKey ? { sequenceKey: f.sequenceKey } : {}),
          ...(f.foldKey ? { foldKey: f.foldKey } : {}),
          ...(rowLineage ? { sourceStructureId: rowLineage } : {}),
          sequence: f.sequence,
          scRmsd: align.rmsd,
          scTm: align.tmScore,
          meanPlddt: effectiveMeanPlddt,
          confidenceAvailable,
          ptm: f.ptm,
          iptm: f.iptm ?? null,
          ...(ipae !== undefined ? { ipae } : {}),
          ...(ipsae !== undefined ? { ipsae } : {}),
          ...(ipsaeMin !== undefined ? { ipsaeMin } : {}),
          ...(binderPlddt !== undefined ? { binderPlddt } : {}),
          ...(targetPlddt !== undefined ? { targetPlddt } : {}),
          ...(binderScRmsd !== undefined ? { binderScRmsd } : {}),
          ...(targetAlignedBinderRmsd !== undefined ? { targetAlignedBinderRmsd } : {}),
          ...(genMotifRmsd !== undefined ? { genMotifRmsd } : {}),
          ...(refoldMotifRmsd !== undefined ? { refoldMotifRmsd } : {}),
          referenceMode: rowReferenceMode,
          referenceRelation: rowReferenceRelation,
          referenceLabel: rowReferenceLabel,
          selfConsistencyValid: rowSelfConsistencyValid,
          lineageMismatch: rowLineageMismatch,
          ...(parsedMapping ? { referenceMappingApplied: true } : {}),
          alignedResidueCount,
          designable,
          failedGates,
          ...(f.msaMode ? { msaMode: f.msaMode } : {}),
          ...(typeof f.msaDepth === 'number' && Number.isFinite(f.msaDepth) ? { msaDepth: f.msaDepth } : {}),
          alignedFoldPdb,
          designedPdb: designedStructureText,
          pae: f.pae,
          plddt: f.plddt,
          geometry: geom,
          foldElapsedMs: f.elapsedMs,
        };
      };

      const collectionItems =
        structure.collection?.items && structure.collection.items.length > 1
          ? structure.collection.items
          : null;

      if (collectionItems) {
        // Multi-backbone collection scoring: resolve EACH fold strictly against its exact declared parent backbone!
        // NEVER fall back to collection.items[0], representative, or length matching.
        const ctxById = new Map<string, ReturnType<typeof prepareRefScoringContext>>();
        const ctxByDesignKey = new Map<string, ReturnType<typeof prepareRefScoringContext>>();
        for (const item of collectionItems) {
          const ctx = prepareRefScoringContext(item);
          ctxById.set(item.id, ctx);
          if (item.designKey) ctxByDesignKey.set(item.designKey, ctx);
        }

        let anyLineageMismatch = false;
        let mismatchedDetail: string | null = null;
        let appliedMappingSpec: ReferenceResidueMapping | undefined;
        const evaluatedDesignIds = new Set<string>();
        const passingDesignIds = new Set<string>();

        const rows: DesignabilityRow[] = folds.folds.map((f) => {
          const foldLen = Math.floor(f.caCoords.length / 3);
          const foldSourceId = normalizeLineageId(f.sourceStructureId);
          const foldDesignKey = normalizeLineageId(f.designKey);
          const byIdCtx = foldSourceId ? ctxById.get(foldSourceId) : undefined;
          const byKeyCtx = foldDesignKey ? ctxByDesignKey.get(foldDesignKey) : undefined;

          let parentError: string | null = null;
          let resolvedCtx: ReturnType<typeof prepareRefScoringContext> | undefined;

          if (!foldSourceId && !foldDesignKey) {
            parentError = `Unlinked parent reference for "${f.sequenceId}": fold declares neither sourceStructureId nor designKey (refusing implicit fallback).`;
          } else if (foldSourceId && foldDesignKey) {
            if (byKeyCtx && byKeyCtx.ref.id !== foldSourceId) {
              parentError = `Cross-parent conflict for "${f.sequenceId}": designKey "${foldDesignKey}" belongs to "${byKeyCtx.ref.id}", conflicting with sourceStructureId "${foldSourceId}".`;
            } else if (byIdCtx && byIdCtx.ref.designKey && byIdCtx.ref.designKey !== foldDesignKey) {
              parentError = `Cross-parent conflict for "${f.sequenceId}": sourceStructureId "${foldSourceId}" belongs to designKey "${byIdCtx.ref.designKey}", conflicting with declared designKey "${foldDesignKey}".`;
            } else if (byIdCtx && byKeyCtx && byIdCtx.ref.id !== byKeyCtx.ref.id) {
              parentError = `Cross-parent conflict for "${f.sequenceId}": sourceStructureId "${foldSourceId}" and designKey "${foldDesignKey}" resolved to different backbones.`;
            } else if (!byIdCtx && !byKeyCtx) {
              parentError = `Missing parent backbone for "${f.sequenceId}": neither sourceStructureId "${foldSourceId}" nor designKey "${foldDesignKey}" exists in the structure collection.`;
            } else if (!byKeyCtx && byIdCtx && byIdCtx.ref.designKey) {
              parentError = `Cross-parent conflict for "${f.sequenceId}": declared designKey "${foldDesignKey}" is absent; refusing to alias to "${byIdCtx.ref.designKey}".`;
            } else {
              resolvedCtx = byKeyCtx ?? byIdCtx;
            }
          } else {
            resolvedCtx = byKeyCtx ?? byIdCtx;
            if (!resolvedCtx) {
              parentError = `Missing parent backbone for "${f.sequenceId}": declared parent "${foldSourceId || foldDesignKey}" was not found in the structure collection.`;
            }
          }

          if (parentError || !resolvedCtx) {
            const errMsg = parentError ?? `Missing parent backbone for "${f.sequenceId}".`;
            anyLineageMismatch = true;
            if (!mismatchedDetail) mismatchedDetail = foldSourceId || foldDesignKey || 'unlinked';
            if (strictLineage) {
              throw new Error(errMsg);
            }
            const confVal = resolveFoldConfidence(f);
            const confidenceAvailable = confVal !== null;
            const effectiveMeanPlddt = confVal !== null ? confVal : Number.NaN;
            const geom = computeGeometryStats(
              f.caCoords,
              f.chainIds && f.chainIds.length === foldLen ? f.chainIds : undefined,
              f.resNums && f.resNums.length === foldLen ? f.resNums : undefined,
            );
            const failedGates: string[] = [errMsg];
            if (!confidenceAvailable) {
              failedGates.push('Confidence unavailable (no pLDDT)');
            } else if (effectiveMeanPlddt < plddtThresh) {
              failedGates.push(`pLDDT ${effectiveMeanPlddt.toFixed(1)} < ${plddtThresh}`);
            }
            return {
              structureId: foldSourceId || structure.id,
              sequenceId: f.sequenceId,
              candidateKey: f.key ?? f.sequenceId,
              ...(foldDesignKey ? { designKey: foldDesignKey } : {}),
              ...(f.designIndex !== undefined ? { designIndex: f.designIndex } : {}),
              ...(f.sequenceKey ? { sequenceKey: f.sequenceKey } : {}),
              ...(f.foldKey ? { foldKey: f.foldKey } : {}),
              ...(foldSourceId ? { sourceStructureId: foldSourceId } : {}),
              sequence: f.sequence,
              scRmsd: Number.NaN,
              scTm: Number.NaN,
              meanPlddt: effectiveMeanPlddt,
              confidenceAvailable,
              ptm: f.ptm,
              iptm: f.iptm ?? null,
              referenceMode: 'lineage-mismatch',
              referenceRelation: foldSourceId || foldDesignKey ? 'external-mismatch' : 'external-unlinked',
              referenceLabel: `Invalid parent reference · ${errMsg}`,
              selfConsistencyValid: false,
              lineageMismatch: true,
              alignedResidueCount: 0,
              designable: false,
              failedGates,
              ...(f.msaMode ? { msaMode: f.msaMode } : {}),
              ...(typeof f.msaDepth === 'number' && Number.isFinite(f.msaDepth) ? { msaDepth: f.msaDepth } : {}),
              alignedFoldPdb: f.pdbText,
              designedPdb: '',
              pae: f.pae,
              plddt: f.plddt,
              geometry: geom,
              foldElapsedMs: f.elapsedMs,
            };
          }

          const parsedMapping = parseReferenceResidueMapping(rawMappingParam, foldLen, resolvedCtx.refLen);
          if (parsedMapping && !appliedMappingSpec) {
            appliedMappingSpec = parsedMapping.spec;
          }
          const rowSelfConsistencyValid = !parsedMapping && !allowExternalReference;
          const rowReferenceRelation: ReferenceRelation = parsedMapping
            ? 'external-mapped'
            : rowSelfConsistencyValid
              ? 'self-consistency'
              : 'external-unlinked';
          const rowReferenceMode: 'self-consistency' | 'external-reference' | 'lineage-mismatch' =
            rowSelfConsistencyValid ? 'self-consistency' : 'external-reference';
          const rowReferenceLabel =
            rowReferenceRelation === 'self-consistency'
              ? `Self-consistency against ${resolvedCtx.refDisplayName}`
              : parsedMapping
                ? `Comparison against ${resolvedCtx.refDisplayName} · Mapped ${parsedMapping.pairs.length} residues`
                : `Comparison against ${resolvedCtx.refDisplayName} · External reference mode enabled`;

          const row = scoreFoldAgainstRef(f, resolvedCtx, {
            rowLineage: resolvedCtx.ref.id,
            rowSelfConsistencyValid,
            rowLineageMismatch: false,
            rowReferenceRelation,
            rowReferenceLabel,
            rowReferenceMode,
            parsedMapping,
          });
          evaluatedDesignIds.add(resolvedCtx.ref.id);
          if (row.designable) passingDesignIds.add(resolvedCtx.ref.id);
          return row;
        });

        rows.sort((a, b) => {
          const af = Number.isFinite(a.scRmsd);
          const bf = Number.isFinite(b.scRmsd);
          if (af && bf) return a.scRmsd - b.scRmsd;
          if (af) return -1;
          if (bf) return 1;
          return 0;
        });

        const passed = rows.filter((r) => r.designable).length;
        const topSelfConsistencyValid =
          !appliedMappingSpec &&
          !allowExternalReference &&
          !anyLineageMismatch &&
          rows.length > 0 &&
          rows.every((r) => r.selfConsistencyValid === true);

        let lineageStatus: DesignabilityMetrics['lineageStatus'];
        let referenceMode: DesignabilityMetrics['referenceMode'];
        let referenceRelation: ReferenceRelation;
        let referenceLabel: string;
        let lineageWarning: string | undefined;

        if (appliedMappingSpec) {
          lineageStatus = 'external-mapped';
          referenceMode = 'external-reference';
          referenceRelation = 'external-mapped';
          referenceLabel = `Comparison across ${evaluatedDesignIds.size} backbones · Mapped ${appliedMappingSpec.pairs?.length ?? 0} residues`;
        } else if (anyLineageMismatch) {
          lineageStatus = 'mismatch';
          referenceMode = 'lineage-mismatch';
          referenceRelation = 'external-mismatch';
          referenceLabel = `Multi-design parent mismatch (${mismatchedDetail ?? 'unknown'})`;
          lineageWarning =
            `Multi-design parent mismatch: one or more folds referenced a missing, unlinked, or conflicting parent backbone (${mismatchedDetail ?? 'unknown'}). ` +
            `Self-consistency pass status is disabled on mismatched rows.`;
        } else if (allowExternalReference) {
          lineageStatus = 'external-unmapped';
          referenceMode = 'external-reference';
          referenceRelation = 'external-unlinked';
          referenceLabel = `External reference comparison across ${evaluatedDesignIds.size} backbones`;
        } else {
          lineageStatus = 'matched';
          referenceMode = 'self-consistency';
          referenceRelation = 'self-consistency';
          referenceLabel = `Self-consistency across ${evaluatedDesignIds.size} backbones`;
        }

        const threshHash = fnv1a(
          canonicalJsonStringify({
            scThresh,
            plddtThresh,
            motifThresh,
            ipaeThresh,
            ipsaeThresh,
            targetAlignedBinderThresh,
            allowExternalReference,
            strictLineage,
            mapping: appliedMappingSpec ?? null,
            collectionId: structure.collection?.id ?? null,
          }),
        );
        const metrics: DesignabilityMetrics = {
          kind: 'DesignabilityMetrics',
          id: `metrics-${structure.id}-${folds.id}-${threshHash}`,
          rows,
          best: rows[0] ?? null,
          passRate: rows.length > 0 ? passed / rows.length : 0,
          referenceStructureId: structure.id,
          foldParentStructureId: folds.parentStructureId || 'multi-design',
          referenceMode,
          referenceRelation,
          referenceLabel,
          selfConsistencyValid: topSelfConsistencyValid,
          lineageStatus,
          lineageMismatch: anyLineageMismatch,
          multiDesign: true,
          evaluatedDesignCount: evaluatedDesignIds.size,
          designHitRate:
            evaluatedDesignIds.size > 0
              ? passingDesignIds.size / evaluatedDesignIds.size
              : null,
          ...(lineageWarning ? { lineageWarning } : {}),
          ...(appliedMappingSpec ? { referenceMapping: appliedMappingSpec } : {}),
          ...(folds.collectionKind ? { collectionKind: folds.collectionKind } : {}),
        };
        return { metrics };
      }

      // Single-backbone legacy path (100% preserved semantics)
      const refCtx = prepareRefScoringContext(structure);
      const structLen = refCtx.refLen;

      const foldSetLineages = new Set<string>(
        [
          normalizeLineageId(folds.parentStructureId),
          normalizeLineageId(folds.sourceStructureId),
          normalizeLineageId(upstreamLineageHint),
        ].filter(Boolean),
      );
      const foldSetHasConflict = foldSetLineages.size > 1;
      const foldSetNonRef = [...foldSetLineages].find((id) => id !== structure.id);
      const foldSetLineage = foldSetNonRef ?? [...foldSetLineages][0] ?? '';

      let anyLineageMismatch = false;
      let mismatchedSourceId: string | null = null;
      let appliedMappingSpec: ReferenceResidueMapping | undefined;
      const refDisplayName = refCtx.refDisplayName;

      const rows: DesignabilityRow[] = folds.folds.map((f) => {
        const foldLen = Math.floor(f.caCoords.length / 3);
        const parsedMapping = parseReferenceResidueMapping(rawMappingParam, foldLen, structLen);
        if (parsedMapping && !appliedMappingSpec) {
          appliedMappingSpec = parsedMapping.spec;
        }

        const rowLineageSet = new Set<string>(foldSetLineages);
        const perFoldLineage = normalizeLineageId(f.sourceStructureId);
        if (perFoldLineage) rowLineageSet.add(perFoldLineage);
        const rowHasConflict = rowLineageSet.size > 1;
        const rowNonRef = [...rowLineageSet].find((id) => id !== structure.id);
        const rowLineage = rowNonRef ?? [...rowLineageSet][0] ?? '';

        const hasExplicitExternal = Boolean(parsedMapping) || allowExternalReference;
        const rowSelfConsistencyValid =
          !parsedMapping &&
          !allowExternalReference &&
          !rowHasConflict &&
          rowLineageSet.size === 1 &&
          rowLineage === structure.id;
        const rowHasLineageDisagreement =
          rowHasConflict || (Boolean(rowLineage) && rowLineage !== structure.id);
        const rowLineageMismatch = !hasExplicitExternal && rowHasLineageDisagreement;
        if (rowLineageMismatch) {
          anyLineageMismatch = true;
          if (!mismatchedSourceId) {
            mismatchedSourceId = rowHasConflict ? [...rowLineageSet].join(' vs ') : rowLineage;
          }
        }

        const rowReferenceRelation: ReferenceRelation = parsedMapping
          ? 'external-mapped'
          : rowSelfConsistencyValid
            ? 'self-consistency'
            : rowHasLineageDisagreement
              ? 'external-mismatch'
              : 'external-unlinked';

        const rowOriginLabel = rowHasConflict ? [...rowLineageSet].join(' / ') : rowLineage;
        const rowReferenceLabel =
          rowReferenceRelation === 'self-consistency'
            ? `Self-consistency against ${refDisplayName}`
            : rowReferenceRelation === 'external-mapped'
              ? `Comparison against ${refDisplayName} · Mapped ${parsedMapping!.pairs.length} residues${rowOriginLabel ? ` · Fold originated from ${rowOriginLabel}` : ''}`
              : rowReferenceRelation === 'external-mismatch'
                ? `Comparison against ${refDisplayName} · Fold originated from ${rowOriginLabel}`
                : allowExternalReference && rowOriginLabel
                  ? `Comparison against ${refDisplayName} · External reference mode enabled`
                  : `Comparison against ${refDisplayName} · No parent structure recorded`;

        const rowReferenceMode: 'self-consistency' | 'external-reference' | 'lineage-mismatch' =
          rowLineageMismatch
            ? 'lineage-mismatch'
            : rowSelfConsistencyValid
              ? 'self-consistency'
              : 'external-reference';

        return scoreFoldAgainstRef(f, refCtx, {
          rowLineage,
          rowSelfConsistencyValid,
          rowLineageMismatch,
          rowReferenceRelation,
          rowReferenceLabel,
          rowReferenceMode,
          parsedMapping,
        });
      });

      // Also detect top-level lineage mismatch even if `folds.folds` is empty
      const topHasLineageDisagreement =
        foldSetHasConflict || (Boolean(foldSetLineage) && foldSetLineage !== structure.id);
      if (
        !anyLineageMismatch &&
        !appliedMappingSpec &&
        !allowExternalReference &&
        topHasLineageDisagreement
      ) {
        anyLineageMismatch = true;
        mismatchedSourceId = foldSetHasConflict
          ? [...foldSetLineages].join(' vs ')
          : foldSetLineage;
      }

      const topSelfConsistencyValid =
        !appliedMappingSpec &&
        !allowExternalReference &&
        !anyLineageMismatch &&
        !foldSetHasConflict &&
        foldSetLineages.size === 1 &&
        foldSetLineage === structure.id &&
        rows.every((r) => r.selfConsistencyValid === true);

      let lineageStatus: DesignabilityMetrics['lineageStatus'];
      let referenceMode: DesignabilityMetrics['referenceMode'];
      let referenceRelation: ReferenceRelation;
      let referenceLabel: string;
      let lineageWarning: string | undefined;

      if (appliedMappingSpec) {
        lineageStatus = 'external-mapped';
        referenceMode = 'external-reference';
        referenceRelation = 'external-mapped';
        referenceLabel = `Comparison against ${refDisplayName} · Mapped ${appliedMappingSpec.pairs?.length ?? 0} residues${foldSetLineage ? ` · Fold originated from ${foldSetLineage}` : ''}`;
      } else if (anyLineageMismatch) {
        lineageStatus = 'mismatch';
        referenceMode = 'lineage-mismatch';
        referenceRelation = 'external-mismatch';
        referenceLabel = `Comparison against ${refDisplayName} · Fold originated from ${mismatchedSourceId}`;
        lineageWarning =
          `Reference lineage mismatch: folds descended from structure "${mismatchedSourceId}" ` +
          `were wired to different reference structure "${structure.id}" ("${structure.label}"). ` +
          `Self-consistency pass status is disabled; wire the originating structure or configure allowExternalReference / residueMapping for external-reference comparison.`;
        if (strictLineage) {
          throw new Error(lineageWarning);
        }
      } else if (allowExternalReference) {
        lineageStatus = 'external-unmapped';
        referenceMode = 'external-reference';
        referenceRelation = topHasLineageDisagreement
          ? 'external-mismatch'
          : 'external-unlinked';
        referenceLabel = topHasLineageDisagreement
          ? `Comparison against ${refDisplayName} · Fold originated from ${foldSetHasConflict ? [...foldSetLineages].join(' / ') : foldSetLineage}`
          : foldSetLineage
            ? `Comparison against ${refDisplayName} · External reference mode enabled`
            : `Comparison against ${refDisplayName} · No parent structure recorded`;
        if (foldSetLineage && foldSetLineage !== structure.id) {
          lineageWarning = `External-reference comparison enabled without custom residueMapping: comparing folds from "${foldSetLineage}" against "${structure.id}" by positional index.`;
        }
      } else if (!foldSetLineage) {
        lineageStatus = 'untracked';
        referenceMode = 'external-reference';
        referenceRelation = 'external-unlinked';
        referenceLabel = `Comparison against ${refDisplayName} · No parent structure recorded`;
        lineageWarning =
          `Unverified reference correspondence: incoming folds have no recorded parent structure (sequence-only source). ` +
          `Self-consistency pass status is disabled; supply residueMapping or allowExternalReference to make external-reference correspondence explicit.`;
      } else {
        lineageStatus = 'matched';
        referenceMode = 'self-consistency';
        referenceRelation = 'self-consistency';
        referenceLabel = `Self-consistency against ${refDisplayName}`;
      }

      rows.sort((a, b) => {
        const af = Number.isFinite(a.scRmsd);
        const bf = Number.isFinite(b.scRmsd);
        if (af && bf) return a.scRmsd - b.scRmsd;
        if (af) return -1;
        if (bf) return 1;
        return 0;
      });
      const passed = rows.filter((r) => r.designable).length;
      const threshHash = fnv1a(
        canonicalJsonStringify({
          scThresh,
          plddtThresh,
          motifThresh,
          ipaeThresh,
          ipsaeThresh,
          targetAlignedBinderThresh,
          allowExternalReference,
          strictLineage,
          mapping: appliedMappingSpec ?? null,
        }),
      );
      const metrics: DesignabilityMetrics = {
        kind: 'DesignabilityMetrics',
        id: `metrics-${structure.id}-${folds.id}-${threshHash}`,
        rows,
        best: rows[0] ?? null,
        passRate: rows.length > 0 ? passed / rows.length : 0,
        referenceStructureId: structure.id,
        foldParentStructureId: foldSetLineage || folds.parentStructureId || 'none',
        referenceMode,
        referenceRelation,
        referenceLabel,
        selfConsistencyValid: topSelfConsistencyValid,
        lineageStatus,
        lineageMismatch: anyLineageMismatch,
        ...(lineageWarning ? { lineageWarning } : {}),
        ...(appliedMappingSpec ? { referenceMapping: appliedMappingSpec } : {}),
        ...(folds.collectionKind ? { collectionKind: folds.collectionKind } : {}),
      };
      return { metrics };
    }

    throw new Error(`Unknown CPU adapter "${node.adapterId}"`);
  }

  /**
   * Programmatic/WebMCP entry point sharing the exact same run ownership, runId,
   * reactive state, lease, checkpointing, and memo budget as canvas `run()`.
   */
  async runWorkflowApi(
    graph: WorkflowGraph,
    onProgress: (ev: NodeProgressEvent) => void = () => {},
    opts: Omit<RunOptions, 'source'> = {},
  ): Promise<RunResult> {
    return this.run(graph, onProgress, { ...opts, source: 'api' });
  }

  async run(
    graph: WorkflowGraph,
    onProgress: (ev: NodeProgressEvent) => void,
    opts: RunOptions = {},
  ): Promise<RunResult> {
    if (this.running) {
      throw new RunLeaseError(
        `A workflow run ("${this.activeRun.runId}", source: ${this.activeRun.source ?? 'unknown'}) is already in progress`,
        this.activeRun.runId,
        this.activeRun.source,
      );
    }

    // Validate & normalize graph BEFORE acquiring lease or emitting progress (preserves pre-run error contract)
    graph = normalizeWorkflowGraph(graph, graph?.id, graph?.name, false);
    const errors = validateGraph(graph);
    for (const issue of validateBioCapabilities(graph)) {
      if (issue.severity === 'error' && !errors.includes(issue.message)) {
        errors.push(issue.message);
      }
    }
    if (errors.length) throw new Error(errors.join('\n'));
    const order = topoSort(graph);
    this.clearGpuIdleTimer();

    if (opts.resumeCheckpoint) {
      this.importCheckpoint(opts.resumeCheckpoint);
    }
    // Keep an immutable snapshot of prior checkpoint entries for this run so mid-run
    // progress updates to `this.latestCheckpoint` never discard later completed nodes
    // even when memo is disabled (`maxMemoBytes: 0`) or evicts entries under pressure.
    const resumeSnapshotNodes: Record<string, CheckpointNodeEntry> = {
      ...(this.latestCheckpoint?.nodes ?? {}),
    };

    const source: RunSource = opts.source ?? 'canvas';
    const runId = opts.runId ?? `run-${++this.runSeq}-${Date.now().toString(36)}`;
    const startedAt = new Date().toISOString();
    const tStart = performance.now();

    const internalAbort = new AbortController();
    const onExternalAbort = () => {
      internalAbort.abort(opts.signal?.reason ?? new Error('Run cancelled'));
    };
    if (opts.signal) {
      if (opts.signal.aborted) {
        internalAbort.abort(opts.signal.reason ?? new Error('Run cancelled'));
      } else {
        opts.signal.addEventListener('abort', onExternalAbort, { once: true });
      }
    }

    this.running = true;
    this.activeAbortController = internalAbort;
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {};
    const refsByNode: Record<string, Record<string, ArtifactRef>> = {};
    const completedNodeIds: string[] = [];
    const checkpointNodes: Record<string, CheckpointNodeEntry> = {};

    const allNodeIds = order.map((n) => n.id);
    const unfinishedOf = (): string[] => {
      const doneSet = new Set(completedNodeIds);
      return allNodeIds.filter((id) => !doneSet.has(id));
    };

    const buildCheckpoint = (status: RunLifecycleStatus): SchedulerCheckpoint => ({
      schema: 'denovo.checkpoint/v1',
      runId,
      graphId: graph.id,
      createdAt: new Date().toISOString(),
      status,
      completedNodeIds: [...completedNodeIds],
      unfinishedNodeIds: unfinishedOf(),
      nodes: { ...checkpointNodes },
    });

    this.activeRun = {
      runId,
      source,
      status: 'running',
      graphId: graph.id,
      activeNodeId: null,
      leaseHeld: true,
      completedNodeIds: [],
      unfinishedNodeIds: [...allNodeIds],
      startedAt,
      finishedAt: null,
      error: null,
    };
    this.emitRunState();

    let finalized = false;

    const persistCheckpointSafe = async (ckpt: SchedulerCheckpoint, swallowError: boolean) => {
      if (!opts.checkpointKey || !this.checkpointStore) return;
      try {
        await this.checkpointStore.save(opts.checkpointKey, serializeCheckpoint(ckpt));
      } catch (saveErr) {
        if (!swallowError) throw saveErr;
        console.warn('[denovo.run] Failed to persist terminal checkpoint:', saveErr);
      }
    };

    const finishCancelled = async (): Promise<RunResult> => {
      await this.waitForPendingGpuDisposal();
      const finishedAt = new Date().toISOString();
      const unfinishedNodeIds = unfinishedOf();
      const ckpt = buildCheckpoint('cancelled');
      this.latestCheckpoint = ckpt;
      if (!finalized) {
        finalized = true;
        this.activeRun = {
          runId,
          source,
          status: 'cancelled',
          graphId: graph.id,
          activeNodeId: null,
          leaseHeld: false,
          completedNodeIds: [...completedNodeIds],
          unfinishedNodeIds,
          startedAt,
          finishedAt,
          error: 'Run cancelled',
        };
        this.pushRunHistory({
          runId,
          source,
          graphId: graph.id,
          status: 'cancelled',
          startedAt,
          finishedAt,
          elapsedMs: performance.now() - tStart,
          completedNodeIds: [...completedNodeIds],
          unfinishedNodeIds,
          refsByNode: { ...refsByNode },
          error: 'Run cancelled',
        });
        this.emitRunState();
        await persistCheckpointSafe(ckpt, true);
      }
      if (opts.returnPartialOnCancel) {
        return {
          runId,
          source,
          status: 'cancelled',
          outputsByNode,
          refsByNode,
          completedNodeIds: [...completedNodeIds],
          unfinishedNodeIds,
          totalElapsedMs: performance.now() - tStart,
        };
      }
      throw new RunCancellationError(
        `Workflow run "${runId}" cancelled (${completedNodeIds.length} completed, ${unfinishedNodeIds.length} unfinished)`,
        {
          runId,
          completedNodeIds: [...completedNodeIds],
          unfinishedNodeIds,
          outputsByNode,
          refsByNode,
          checkpoint: ckpt,
        },
      );
    };

    try {
      for (const node of order) {
        if (internalAbort.signal.aborted) {
          return await finishCancelled();
        }

        const manifest = ADAPTER_REGISTRY[node.adapterId];
        if (!manifest) throw new Error(`Unknown adapter "${node.adapterId}"`);

        this.activeRun = {
          ...this.activeRun,
          activeNodeId: node.id,
          completedNodeIds: [...completedNodeIds],
          unfinishedNodeIds: unfinishedOf(),
        };
        this.emitRunState();

        // Gather upstream inputs
        const inputs: Record<string, CanonicalArtifact> = {};
        const inputIds: Record<string, string> = {};
        let upstreamLineageHint: string | undefined;
        for (const edge of graph.edges) {
          if (edge.toNode !== node.id) continue;
          const up = outputsByNode[edge.fromNode]?.[edge.fromPort];
          if (up) {
            inputs[edge.toPort] = up;
            inputIds[edge.toPort] = up.id;
          }
          // Trace SequenceSet lineage feeding the upstream LocalFold node for designability-scorer
          if (node.adapterId === 'designability-scorer' && edge.toPort === 'folds') {
            const seqEdge = graph.edges.find(
              (e) => e.toNode === edge.fromNode && e.toPort === 'sequences',
            );
            if (seqEdge) {
              const seqArt = outputsByNode[seqEdge.fromNode]?.[seqEdge.fromPort];
              if (seqArt && seqArt.kind === 'SequenceSet') {
                const candidateLineage = seqArt.sourceStructureId ?? seqArt.parentStructureId;
                if (candidateLineage && candidateLineage.toLowerCase() !== 'none') {
                  upstreamLineageHint = candidateLineage;
                }
              }
            }
          }
        }

        // Content-addressed memoization key (canonicalized key order prevents spurious misses)
        const cacheKey = fnv1a(
          canonicalJsonStringify({ adapter: node.adapterId, params: node.params, inputIds }),
        );

        let nodeOutputs: Record<string, CanonicalArtifact>;
        let elapsedMs = 0;

        // Check verified memo or immutable resume checkpoint snapshot entry with matching cacheKey
        const memoHit = !opts.bypassCache ? this.getMemoEntry(cacheKey) : undefined;
        const ckptEntry =
          !opts.bypassCache && !memoHit && resumeSnapshotNodes[node.id]?.cacheKey === cacheKey
            ? resumeSnapshotNodes[node.id]
            : undefined;
        const cachedOutputs = memoHit ?? ckptEntry?.outputs;

        if (cachedOutputs) {
          // Always re-verify output contract even on cached/checkpointed entries before passing downstream
          nodeOutputs = validateAdapterOutputs(node, cachedOutputs);
          if (!memoHit) {
            this.setMemoEntry(cacheKey, nodeOutputs);
          }
          this.recordNodeMemoKey(node.id, cacheKey);
          onProgress({
            nodeId: node.id,
            status: 'cached',
            progress: 1,
            detail: ckptEntry ? 'Checkpoint hit (skipped)' : 'Memoized hit (skipped)',
            elapsedMs: 0,
          });
        } else {
          const nStart = performance.now();
          onProgress({ nodeId: node.id, status: 'running', progress: 0, detail: 'Starting...' });
          try {
            let rawOutputs: unknown;
            if (manifest.runtime === 'cpu-ts') {
              rawOutputs = await this.runCpuNode(node, inputs, onProgress, upstreamLineageHint);
              elapsedMs = performance.now() - nStart;
            } else {
              // Layer 2 biological capability guard before launching GPU worker / downloading weights
              const guard = validateRuntimeNodeInputs(node, inputs);
              if (guard.warnings.length > 0) {
                console.warn(`[denovo.run] ${node.label ?? node.adapterId}:`, guard.warnings.join(' | '));
              }
              const res = await this.runGpuNode(node, inputs, onProgress, internalAbort.signal);
              rawOutputs = res.outputs;
              elapsedMs = res.elapsedMs || performance.now() - nStart;
            }

            if (internalAbort.signal.aborted) {
              return await finishCancelled();
            }

            // Issue #4: Validate worker/CPU output contract BEFORE memoizing or passing downstream
            nodeOutputs = validateAdapterOutputs(node, rawOutputs);

            // Propagate SequenceSet sourceStructureId onto FoldedStructure if missing
            if (node.adapterId === 'localfold-af2') {
              const seqIn = inputs.sequences as SequenceSet | undefined;
              const foldOut = nodeOutputs.folds as FoldedStructure | undefined;
              const seqLineage = seqIn?.sourceStructureId ?? seqIn?.parentStructureId;
              if (
                foldOut &&
                !foldOut.sourceStructureId &&
                seqLineage &&
                seqLineage.toLowerCase() !== 'none' &&
                (!foldOut.parentStructureId || foldOut.parentStructureId.toLowerCase() === 'none')
              ) {
                foldOut.sourceStructureId = seqLineage;
              }
            }

            this.setMemoEntry(cacheKey, nodeOutputs);
            this.recordNodeMemoKey(node.id, cacheKey);
            onProgress({
              nodeId: node.id,
              status: 'done',
              progress: 1,
              detail: `Done in ${(elapsedMs / 1000).toFixed(1)}s`,
              elapsedMs,
            });
          } catch (err) {
            if (err instanceof RunCancellationError) {
              throw err;
            }
            if (internalAbort.signal.aborted) {
              return await finishCancelled();
            }
            const message = err instanceof Error ? err.message : String(err);
            onProgress({ nodeId: node.id, status: 'error', error: message, detail: message });
            throw err;
          }
        }

        outputsByNode[node.id] = nodeOutputs;
        opts.onNodeOutput?.(node.id, nodeOutputs);
        refsByNode[node.id] = {};
        for (const [port, art] of Object.entries(nodeOutputs)) {
          refsByNode[node.id][port] = summarizeArtifact(art);
        }

        completedNodeIds.push(node.id);
        const ckptNodeBase: CheckpointNodeEntry = {
          nodeId: node.id,
          adapterId: node.adapterId,
          cacheKey,
          completedAt: new Date().toISOString(),
          outputs: nodeOutputs,
          refs: { ...refsByNode[node.id] },
        };
        checkpointNodes[node.id] = {
          ...ckptNodeBase,
          integrity: computeCheckpointEntryIntegrity(ckptNodeBase),
        };
        const progressCkpt = buildCheckpoint('running');
        this.latestCheckpoint = progressCkpt;
        await persistCheckpointSafe(progressCkpt, false);

        this.activeRun = {
          ...this.activeRun,
          activeNodeId: node.id,
          completedNodeIds: [...completedNodeIds],
          unfinishedNodeIds: unfinishedOf(),
        };
        this.emitRunState();
      }

      const finishedAt = new Date().toISOString();
      const totalElapsedMs = performance.now() - tStart;
      const doneCkpt = buildCheckpoint('completed');
      this.latestCheckpoint = doneCkpt;
      await persistCheckpointSafe(doneCkpt, false);

      finalized = true;
      this.activeRun = {
        runId,
        source,
        status: 'completed',
        graphId: graph.id,
        activeNodeId: null,
        leaseHeld: false,
        completedNodeIds: [...completedNodeIds],
        unfinishedNodeIds: [],
        startedAt,
        finishedAt,
        error: null,
      };
      this.pushRunHistory({
        runId,
        source,
        graphId: graph.id,
        status: 'completed',
        startedAt,
        finishedAt,
        elapsedMs: totalElapsedMs,
        completedNodeIds: [...completedNodeIds],
        unfinishedNodeIds: [],
        refsByNode: { ...refsByNode },
      });
      this.emitRunState();

      return {
        runId,
        source,
        status: 'completed',
        outputsByNode,
        refsByNode,
        completedNodeIds: [...completedNodeIds],
        unfinishedNodeIds: [],
        totalElapsedMs,
      };
    } catch (err) {
      if (err instanceof RunCancellationError) {
        throw err;
      }
      if (!finalized) {
        finalized = true;
        await this.waitForPendingGpuDisposal();
        const finishedAt = new Date().toISOString();
        const message = err instanceof Error ? err.message : String(err);
        const unfinishedNodeIds = unfinishedOf();
        const failCkpt = buildCheckpoint('failed');
        this.latestCheckpoint = failCkpt;
        this.activeRun = {
          runId,
          source,
          status: 'failed',
          graphId: graph.id,
          activeNodeId: null,
          leaseHeld: false,
          completedNodeIds: [...completedNodeIds],
          unfinishedNodeIds,
          startedAt,
          finishedAt,
          error: message,
        };
        this.pushRunHistory({
          runId,
          source,
          graphId: graph.id,
          status: 'failed',
          startedAt,
          finishedAt,
          elapsedMs: performance.now() - tStart,
          completedNodeIds: [...completedNodeIds],
          unfinishedNodeIds,
          refsByNode: { ...refsByNode },
          error: message,
        });
        this.emitRunState();
        // Never allow a checkpoint-store failure during error cleanup to corrupt terminal state/history
        await persistCheckpointSafe(failCkpt, true);
      }
      throw err;
    } finally {
      opts.signal?.removeEventListener('abort', onExternalAbort);
      await this.waitForPendingGpuDisposal();
      this.running = false;
      this.activeAbortController = null;
      if (this.activeRun.leaseHeld) {
        this.activeRun = { ...this.activeRun, leaseHeld: false, activeNodeId: null };
        this.emitRunState();
      }
      this.scheduleGpuIdleRelease();
    }
  }
}

function summarizeArtifact(art: CanonicalArtifact): ArtifactRef {
  if (art.kind === 'StructureArtifact') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        label: art.label,
        length: art.length,
        sourceModelId: art.sourceModelId,
        rg: Number(art.geometry.rg.toFixed(2)),
        helixPct: Number(art.geometry.helixPct.toFixed(1)),
        ...(art.representativeSelection ? { representativeSelection: art.representativeSelection } : {}),
      },
    };
  }
  if (art.kind === 'TrajectoryFrames') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        frameCount: art.frameCount,
        ordering: art.ordering,
        parentStructureId: art.parentStructureId,
        ...(art.label ? { label: art.label } : {}),
      },
    };
  }
  if (art.kind === 'SequenceSet') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        count: art.candidates.length,
        bestScore: art.candidates[0]?.score ?? null,
        topSequence: art.candidates[0]?.sequence ?? '',
        ...(art.collectionKind ? { collectionKind: art.collectionKind } : {}),
      },
    };
  }
  if (art.kind === 'FoldedStructure') {
    const confVals = art.folds
      .map((f) => resolveFoldConfidence(f))
      .filter((v): v is number => v !== null);
    const iptmVals = art.folds
      .map((f) => f.iptm)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    const ipsaeVals = art.folds
      .map((f) => f.ipsae)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        count: art.folds.length,
        bestPlddt: confVals.length > 0 ? Number(Math.max(...confVals).toFixed(1)) : null,
        confidenceAvailable: confVals.length > 0,
        ...(iptmVals.length > 0 ? { bestIptm: Number(Math.max(...iptmVals).toFixed(3)) } : {}),
        ...(ipsaeVals.length > 0 ? { bestIpsae: Number(Math.max(...ipsaeVals).toFixed(3)) } : {}),
        ...(art.collectionKind ? { collectionKind: art.collectionKind } : {}),
        ...(art.representativeSelection ? { representativeSelection: art.representativeSelection } : {}),
      },
    };
  }
  if (art.kind === 'MutationMatrix') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        metric: art.metric,
        direction: art.direction,
        length: art.residues.length,
        alphabetSize: art.alphabet.length,
        targetedCount: art.targetedMutations?.length ?? 0,
      },
    };
  }
  if (art.kind === 'ResidueTrack') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        quantity: art.quantity,
        length: art.residues.length,
        ...(art.summary ? { mean: art.summary.mean, max: art.summary.max } : {}),
      },
    };
  }
  if (art.kind === 'SequenceEmbedding') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        length: art.sequence.length,
        dim: art.dim,
        ...(art.pseudoPerplexity !== undefined ? { pseudoPerplexity: art.pseudoPerplexity } : {}),
      },
    };
  }
  if (art.kind === 'ConditionSpec') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        contig: art.contig,
        summary: art.summary,
      },
    };
  }
  const best = art.best;
  return {
    id: art.id,
    kind: art.kind,
    summary: {
      passRate: art.passRate,
      bestScRmsd: best && Number.isFinite(best.scRmsd) ? Number(best.scRmsd.toFixed(3)) : null,
      bestScTm: best && Number.isFinite(best.scTm) ? Number(best.scTm.toFixed(3)) : null,
      bestPlddt: best && Number.isFinite(best.meanPlddt) ? Number(best.meanPlddt.toFixed(1)) : null,
      ...(art.lineageMismatch !== undefined ? { lineageMismatch: art.lineageMismatch } : {}),
      ...(art.lineageStatus ? { lineageStatus: art.lineageStatus } : {}),
      ...(art.referenceMode ? { referenceMode: art.referenceMode } : {}),
      ...(art.referenceRelation ? { referenceRelation: art.referenceRelation } : {}),
      ...(art.referenceLabel ? { referenceLabel: art.referenceLabel } : {}),
      ...(art.selfConsistencyValid !== undefined
        ? { selfConsistencyValid: art.selfConsistencyValid }
        : {}),
      ...(best && typeof best.iptm === 'number' && Number.isFinite(best.iptm)
        ? { bestIptm: Number(best.iptm.toFixed(3)) }
        : {}),
      ...(best && typeof best.ipae === 'number' && Number.isFinite(best.ipae)
        ? { bestIpae: Number(best.ipae.toFixed(2)) }
        : {}),
      ...(best && typeof best.ipsae === 'number' && Number.isFinite(best.ipsae)
        ? { bestIpsae: Number(best.ipsae.toFixed(3)) }
        : {}),
      ...(best && typeof best.binderScRmsd === 'number' && Number.isFinite(best.binderScRmsd)
        ? { bestBinderScRmsd: Number(best.binderScRmsd.toFixed(3)) }
        : {}),
      ...(best &&
      typeof best.targetAlignedBinderRmsd === 'number' &&
      Number.isFinite(best.targetAlignedBinderRmsd)
        ? { bestTargetAlignedBinderRmsd: Number(best.targetAlignedBinderRmsd.toFixed(3)) }
        : {}),
      ...(best && typeof best.genMotifRmsd === 'number' && Number.isFinite(best.genMotifRmsd)
        ? { bestGenMotifRmsd: Number(best.genMotifRmsd.toFixed(3)) }
        : {}),
      ...(best && typeof best.refoldMotifRmsd === 'number' && Number.isFinite(best.refoldMotifRmsd)
        ? { bestRefoldMotifRmsd: Number(best.refoldMotifRmsd.toFixed(3)) }
        : {}),
    },
  };
}

interface ScorerChainRange {
  chainId: string;
  start: number;
  length: number;
}

function resolveScorerChainRanges(
  L: number,
  chainLengths?: readonly number[],
  chainIds?: readonly string[],
): ScorerChainRange[] {
  if (L <= 0) return [];
  if (chainIds && chainIds.length === L) {
    const ranges: ScorerChainRange[] = [];
    let start = 0;
    for (let i = 1; i <= L; i++) {
      if (i === L || chainIds[i] !== chainIds[start]) {
        ranges.push({ chainId: chainIds[start], start, length: i - start });
        start = i;
      }
    }
    if (ranges.length > 0) return ranges;
  }
  if (chainLengths && chainLengths.length > 0 && chainLengths.reduce((a, b) => a + b, 0) === L) {
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let start = 0;
    return chainLengths.map((len, idx) => {
      const r = { chainId: alphabet[idx % alphabet.length], start, length: len };
      start += len;
      return r;
    });
  }
  return [{ chainId: 'A', start: 0, length: L }];
}

function expandChainRangesToIds(ranges: readonly ScorerChainRange[]): string[] {
  const out: string[] = [];
  for (const r of ranges) {
    for (let i = 0; i < r.length; i++) out.push(r.chainId);
  }
  return out;
}

function splitBinderAndTargetIndices(
  ranges: readonly ScorerChainRange[],
  chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>,
  lockedChains?: readonly string[],
): { binderIndices: number[]; targetIndices: number[]; explicitRoleSplit: boolean } {
  let binderRanges: ScorerChainRange[] = [];
  let targetRanges: ScorerChainRange[] = [];
  let explicitRoleSplit = false;

  if (chainRoles) {
    binderRanges = ranges.filter((r) => chainRoles[r.chainId] === 'binder');
    targetRanges = ranges.filter((r) => chainRoles[r.chainId] === 'target');
    if (binderRanges.length > 0 && targetRanges.length > 0) {
      explicitRoleSplit = true;
    }
  }

  if (!explicitRoleSplit && lockedChains && lockedChains.length > 0) {
    const locked = new Set(lockedChains.map((c) => c.trim().toUpperCase()));
    targetRanges = ranges.filter((r) => locked.has(r.chainId.toUpperCase()));
    binderRanges = ranges.filter((r) => !locked.has(r.chainId.toUpperCase()));
    if (binderRanges.length > 0 && targetRanges.length > 0) {
      explicitRoleSplit = true;
    }
  }

  if (binderRanges.length === 0 || targetRanges.length === 0) {
    // Fallback heuristic: shorter chain is the designed binder (tie-breaker: later chain, matching Target=A, Binder=B)
    let binderIdx = ranges.length - 1;
    let minLen = ranges[binderIdx].length;
    for (let i = ranges.length - 2; i >= 0; i--) {
      if (ranges[i].length < minLen) {
        minLen = ranges[i].length;
        binderIdx = i;
      }
    }
    binderRanges = [ranges[binderIdx]];
    targetRanges = ranges.filter((_, idx) => idx !== binderIdx);
  }

  const expand = (rs: readonly ScorerChainRange[]): number[] => {
    const out: number[] = [];
    for (const r of rs) {
      for (let i = r.start; i < r.start + r.length; i++) out.push(i);
    }
    return out;
  };

  return {
    binderIndices: expand(binderRanges),
    targetIndices: expand(targetRanges),
    explicitRoleSplit,
  };
}

function gatherCoordsByIndices(coords: Float32Array, indices: readonly number[]): Float32Array {
  const out = new Float32Array(indices.length * 3);
  for (let i = 0; i < indices.length; i++) {
    const src = indices[i] * 3;
    out[i * 3 + 0] = coords[src + 0];
    out[i * 3 + 1] = coords[src + 1];
    out[i * 3 + 2] = coords[src + 2];
  }
  return out;
}

function computeInterfacePae(
  pae: Float32Array,
  paeDim: number,
  proteinLen: number,
  ranges: readonly ScorerChainRange[],
  binderIndices?: readonly number[],
  targetIndices?: readonly number[],
): number | undefined {
  let sum = 0;
  let count = 0;
  if (binderIndices && targetIndices && binderIndices.length > 0 && targetIndices.length > 0) {
    for (const bi of binderIndices) {
      for (const ti of targetIndices) {
        const v1 = pae[bi * paeDim + ti];
        if (Number.isFinite(v1)) {
          sum += v1;
          count++;
        }
        const v2 = pae[ti * paeDim + bi];
        if (Number.isFinite(v2)) {
          sum += v2;
          count++;
        }
      }
    }
  } else {
    const chainOf = new Int32Array(proteinLen);
    for (let c = 0; c < ranges.length; c++) {
      const r = ranges[c];
      for (let i = r.start; i < r.start + r.length; i++) chainOf[i] = c;
    }
    for (let i = 0; i < proteinLen; i++) {
      const ci = chainOf[i];
      for (let j = 0; j < proteinLen; j++) {
        if (ci === chainOf[j]) continue;
        const v = pae[i * paeDim + j];
        if (Number.isFinite(v)) {
          sum += v;
          count++;
        }
      }
    }
  }
  return count > 0 ? sum / count : undefined;
}

/**
 * Expose programmatic engine API on window.__denovo AND register progressive WebMCP
 * tools (document.modelContext / navigator.modelContext) when the browser supports it.
 */
export function attachProgrammaticApi(
  scheduler: ExecutionScheduler,
  onProgress?: (ev: NodeProgressEvent) => void,
): { webMcpRegistered: boolean } {
  if (typeof window === 'undefined') return { webMcpRegistered: false };

  const api = {
    listAdapters: () => Object.values(ADAPTER_REGISTRY),
    listPresets: () => WORKFLOW_PRESETS.map((p) => ({ id: p.id, name: p.name, summary: p.summary })),
    getActiveRunState: () => scheduler.getActiveRunState(),
    getRunHistory: () => scheduler.getRunHistory(),
    getMemoStats: () => scheduler.getMemoStats(),
    cancelActiveRun: (reason?: string) => scheduler.cancelActiveRun(reason),
    runWorkflow: async (graph: WorkflowGraph, opts?: Omit<RunOptions, 'source'>) => {
      const res = await scheduler.runWorkflowApi(graph, (ev) => onProgress?.(ev), opts);
      return {
        runId: res.runId,
        status: res.status,
        refs: res.refsByNode,
        completedNodeIds: res.completedNodeIds,
        unfinishedNodeIds: res.unfinishedNodeIds,
        totalElapsedMs: res.totalElapsedMs,
      };
    },
    runPreset: async (presetId: string) => {
      const preset = WORKFLOW_PRESETS.find((p) => p.id === presetId);
      if (!preset) throw new Error(`Unknown preset "${presetId}"`);
      const res = await scheduler.runWorkflowApi(
        structuredClone(preset.graph),
        (ev) => onProgress?.(ev),
      );
      return {
        runId: res.runId,
        status: res.status,
        refs: res.refsByNode,
        completedNodeIds: res.completedNodeIds,
        unfinishedNodeIds: res.unfinishedNodeIds,
        totalElapsedMs: res.totalElapsedMs,
      };
    },
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (window as any).__denovo = api;

  // Progressive WebMCP registration (Chrome/Edge Origin Trial)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mc = (document as any).modelContext ?? (navigator as any).modelContext;
  if (mc && typeof mc.registerTool === 'function') {
    try {
      mc.registerTool({
        name: 'denovo_list_presets',
        description: 'List available browser-native WebGPU protein design and refolding workflows on denovo.run.',
        inputSchema: { type: 'object', properties: {} },
        execute: async () => api.listPresets(),
      });
      mc.registerTool({
        name: 'denovo_run_preset',
        description: 'Execute a preset WebGPU protein design / refolding workflow locally in the browser and return scRMSD, scTM, and pLDDT metrics.',
        inputSchema: {
          type: 'object',
          properties: {
            presetId: {
              type: 'string',
              enum: WORKFLOW_PRESETS.map((p) => p.id),
            },
          },
          required: ['presetId'],
        },
        execute: async (args: { presetId: string }) => api.runPreset(args.presetId),
      });
      return { webMcpRegistered: true };
    } catch {
      return { webMcpRegistered: false };
    }
  }
  return { webMcpRegistered: false };
}
