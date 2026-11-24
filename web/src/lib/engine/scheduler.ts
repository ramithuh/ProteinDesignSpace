// Headless DAG ExecutionScheduler (zero Svelte/DOM dependencies).
// Handles topological execution, content-addressed memoization, Single-Active-GPU-Worker
// VRAM mutex (dispose -> terminate), CPU nodes, and WebMCP / window.__denovo hooks.
// All 21 GPU workers are dynamically lazy-loaded via PluginRegistry (zero static `?worker` imports).

import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS, WORKFLOW_PRESETS } from './registry';
import { validateGraph } from './graph-utils';
import { validateNodeParams } from './param-validation';
import { normalizeWorkflowGraph } from './share';
import {
  resolveFoldConfidence,
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
import { fnv1a, canonicalJsonStringify } from '$lib/kernel/hash-utils';
import type { PluginRegistry } from '$lib/kernel/plugin-registry';
import type { CpuExecutionContext, ModelPlugin, UpstreamGraphQuery } from '$lib/kernel/plugin-types';
import { getActivePluginRegistry } from '$lib/kernel/active-registry';
import { hydrateLegacyArtifactToEnvelope, type ArtifactEnvelope } from '$lib/kernel/artifacts';
import {
  materializePdbInputNode,
  deriveSequenceSetFromStructure,
  pdbInputPlugin,
} from '$lib/plugins/pdb-input';
import {
  materializeSequenceInputNode,
  sequenceInputPlugin,
} from '$lib/plugins/sequence-input';
import { topKFilterPlugin } from '$lib/plugins/top-k-filter';
import { structureComparePlugin } from '$lib/plugins/structure-compare';
import { designabilityScorerPlugin } from '$lib/plugins/designability-scorer';
import type {
  AdapterManifest,
  ArtifactRef,
  CanonicalArtifact,
  FoldedStructure,
  NodeProgressEvent,
  SequenceSet,
  WorkflowGraph,
  WorkflowNode,
} from './types';

export {
  exportJoinedCollectionCsv,
  joinBranchArtifacts,
  joinKeyedCollections,
  materializePdbInputNode,
  deriveSequenceSetFromStructure,
  materializeSequenceInputNode,
};
export type { CheckpointStorageAdapter };

const BUILTIN_CPU_PLUGINS: Readonly<Record<string, ModelPlugin>> = {
  'pdb-input': pdbInputPlugin,
  'sequence-input': sequenceInputPlugin,
  'top-k-filter': topKFilterPlugin,
  'structure-compare': structureComparePlugin,
  'designability-scorer': designabilityScorerPlugin,
};

function resolveManifestForAdapter(
  adapterId: string,
  registry?: PluginRegistry | null,
): AdapterManifest | undefined {
  const fromReg = registry?.getPlugin(adapterId) ?? getActivePluginRegistry()?.getPlugin(adapterId);
  if (fromReg) return fromReg.manifest as unknown as AdapterManifest;
  return Object.hasOwn(ADAPTER_REGISTRY, adapterId) ? ADAPTER_REGISTRY[adapterId] : undefined;
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
  /** Optional Layer 1 PluginRegistry for plugin-driven worker spawning, isolation keys, and cooperative abort. */
  pluginRegistry?: PluginRegistry;
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
  registry?: PluginRegistry | null,
): Record<string, CanonicalArtifact> {
  const manifest = resolveManifestForAdapter(node.adapterId, registry);
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
    const expectedTypes = Array.isArray(portSpec.type) ? portSpec.type : [portSpec.type];
    const expectedLabel = expectedTypes.join(' | ');
    const value = record[portName];
    if (value === undefined || value === null) {
      if (required) {
        throw new Error(
          `Output contract violation on node "${node.id}" (${node.adapterId}): missing required output port "${portName}" (expected ${expectedLabel}).`,
        );
      }
      continue;
    }
    if (typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(
        `Output contract violation on node "${node.id}" (${node.adapterId}): output port "${portName}" is not a valid artifact object (expected ${expectedLabel}).`,
      );
    }
    const art = value as Partial<CanonicalArtifact>;
    if (!expectedTypes.includes(art.kind as never)) {
      throw new Error(
        `Output contract violation on node "${node.id}" (${node.adapterId}): output port "${portName}" returned kind "${String(art.kind)}", expected "${expectedLabel}".`,
      );
    }
    if (typeof art.id !== 'string' || art.id.trim().length === 0) {
      throw new Error(
        `Output contract violation on node "${node.id}" (${node.adapterId}): output port "${portName}" (${expectedLabel}) is missing a non-empty artifact id.`,
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

function normalizeParamsForComputation(
  adapterId: string,
  params: Record<string, unknown> | undefined,
  registry?: PluginRegistry | null,
): Record<string, unknown> {
  if (!params) return {};
  const plugin =
    registry?.getPlugin(adapterId) ??
    getActivePluginRegistry()?.getPlugin(adapterId) ??
    BUILTIN_CPU_PLUGINS[adapterId];
  const displayOnlyKeys = plugin?.display.displayOnlyParamKeys;
  if (!displayOnlyKeys || displayOnlyKeys.length === 0) return params;
  const skipSet = new Set(displayOnlyKeys);
  const filtered: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (!skipSet.has(key)) {
      filtered[key] = value;
    }
  }
  return filtered;
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
      canonicalJsonStringify(normalizeParamsForComputation(prev.adapterId, prev.params)) !==
        canonicalJsonStringify(normalizeParamsForComputation(n.adapterId, n.params)) ||
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

interface ActiveGpuSlot {
  adapterId: string;
  isolatedCheckpoint?: string;
  worker: Worker;
  abortActive?: (err: Error) => void;
}

export type RemoteGpuExecutor = (
  node: WorkflowNode,
  inputs: Record<string, CanonicalArtifact>,
  onProgress: (ev: NodeProgressEvent) => void,
  signal?: AbortSignal,
) => Promise<{ outputs: Record<string, CanonicalArtifact>; elapsedMs: number }> | null;

export type FoldLimitResolver = (
  node: WorkflowNode,
) => { maxFoldResidues?: number | null; isHostGpu?: boolean } | undefined;

export type ExecutionProvenanceResolver = (node: WorkflowNode) => string;

export class ExecutionScheduler {
  private activeGpu: ActiveGpuSlot | null = null;
  private pluginRegistry: PluginRegistry | null = null;
  private remoteGpuExecutor: RemoteGpuExecutor | null = null;
  private foldLimitResolver: FoldLimitResolver | null = null;
  private executionProvenanceResolver: ExecutionProvenanceResolver | null = null;
  private abortPendingPromise: Promise<void> | null = null;
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
    this.pluginRegistry = opts.pluginRegistry ?? null;
    this.gpuIdleReleaseMs =
      typeof opts.gpuIdleReleaseMs === 'number' && opts.gpuIdleReleaseMs >= 0
        ? opts.gpuIdleReleaseMs
        : DEFAULT_GPU_IDLE_RELEASE_MS;
  }

  /** Attach or replace the Layer 1 PluginRegistry used for dynamic worker spawning and cooperative abort. */
  setPluginRegistry(registry: PluginRegistry | null): void {
    this.pluginRegistry = registry;
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

  /**
   * Register an optional remote GPU executor (e.g. self-hosted host-GPU WebGPU server).
   * Returning `null` falls through to the local browser WebGPU worker.
   */
  setRemoteGpuExecutor(executor: RemoteGpuExecutor | null): void {
    this.remoteGpuExecutor = executor;
  }

  /**
   * Register an optional per-node folding residue ceiling resolver (e.g. from GPU Settings or Host GPU).
   */
  setFoldLimitResolver(resolver: FoldLimitResolver | null): void {
    this.foldLimitResolver = resolver;
  }

  /**
   * Register an execution provenance tag resolver (M6: partitions memo & checkpoint keys
   * across compute trust boundaries, e.g. 'local' vs 'remote:https://host:8790').
   */
  setExecutionProvenanceResolver(resolver: ExecutionProvenanceResolver | null): void {
    this.executionProvenanceResolver = resolver;
  }

  /**
   * Execute a single GPU node directly on local WebGPU workers (used by the hidden /runner
   * host harness inside server/selfhost-server.mjs so all 22 workers & VRAM lifecycle are reused).
   * H4: Enforces manifest allowlist, validateNodeParams, and validateRuntimeNodeInputs with server ceiling.
   */
  async runSingleGpuNode(
    node: WorkflowNode,
    inputs: Record<string, CanonicalArtifact>,
    onProgress: (ev: NodeProgressEvent) => void,
    signal?: AbortSignal,
    maxFoldResidues?: number,
  ): Promise<{ outputs: Record<string, CanonicalArtifact>; elapsedMs: number }> {
    const manifest = ADAPTER_REGISTRY[node.adapterId];
    if (!manifest || manifest.runtime === 'cpu-ts') {
      throw new Error(`Disallowed or unknown GPU adapter "${node.adapterId}".`);
    }
    const connectedInputPorts = Object.keys(inputs ?? {});
    const paramErrors = validateNodeParams(node, { connectedInputs: connectedInputPorts });
    if (paramErrors.length > 0) {
      throw new Error(`Invalid parameters for ${node.adapterId}: ${paramErrors.join('; ')}`);
    }
    validateRuntimeNodeInputs(node, inputs, {
      ...(typeof maxFoldResidues === 'number' && Number.isFinite(maxFoldResidues)
        ? { maxFoldResidues: Math.round(maxFoldResidues) }
        : {}),
      isHostGpu: true,
    });
    this.clearGpuIdleTimer();
    try {
      return await this.runLocalGpuNode(node, inputs, onProgress, signal, maxFoldResidues);
    } finally {
      this.scheduleGpuIdleRelease();
    }
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
    if (this.abortPendingPromise) {
      await this.abortPendingPromise.catch(() => {});
    }
    if (this.gpuDisposalPromise) {
      await this.gpuDisposalPromise;
    }
  }

  /**
   * Two-phase cooperative abort with run-lease barrier:
   * sends `{ id, type: 'abort' }`, awaits `{ id, type: 'aborted' }` for `graceMs` (>= 1500 ms),
   * keeping the warm GPU worker alive on timely settlement or escalating to `releaseActiveGpuInternal()` on timeout.
   */
  private requestCooperativeAbort(slot: ActiveGpuSlot, graceMs: number): Promise<void> {
    const effectiveGraceMs = Math.max(1500, Math.round(graceMs));
    const prevAbort = this.abortPendingPromise;
    const abortFlow = (async () => {
      if (prevAbort) {
        await prevAbort.catch(() => {});
      }
      const settled = await new Promise<boolean>((resolve) => {
        const abortId = this.reqCounter++;
        let done = false;
        const finish = (ok: boolean) => {
          if (done) return;
          done = true;
          clearTimeout(timer);
          slot.worker.removeEventListener('message', onMsg);
          slot.worker.removeEventListener('error', onErr);
          resolve(ok);
        };
        const timer = setTimeout(() => finish(false), effectiveGraceMs);
        const onMsg = (ev: MessageEvent) => {
          if (ev.data?.id === abortId && ev.data?.type === 'aborted') {
            finish(true);
          }
        };
        const onErr = () => finish(false);
        slot.worker.addEventListener('message', onMsg);
        slot.worker.addEventListener('error', onErr);
        try {
          slot.worker.postMessage({ id: abortId, type: 'abort' });
        } catch {
          finish(false);
        }
      });

      if (!settled && this.activeGpu === slot) {
        await this.releaseActiveGpuInternal('Workflow run cancelled');
      }
    })();

    const tracked = abortFlow.finally(() => {
      if (this.abortPendingPromise === tracked) {
        this.abortPendingPromise = null;
      }
    });
    this.abortPendingPromise = tracked;
    return tracked;
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

  private async spawnWorker(adapterId: string): Promise<Worker> {
    let registry = this.pluginRegistry ?? getActivePluginRegistry();
    if (!registry) {
      const { defaultPluginRegistry } = await import('$lib/plugins');
      registry = defaultPluginRegistry;
    }
    const plugin = registry.getPlugin(adapterId);
    if (plugin?.execution.createWorker) {
      return await plugin.execution.createWorker();
    }
    throw new Error(`No GPU worker registered for adapter "${adapterId}"`);
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

    if (this.remoteGpuExecutor) {
      const remotePromise = this.remoteGpuExecutor(node, inputs, onProgress, signal);
      if (remotePromise) {
        return await remotePromise;
      }
    }

    return await this.runLocalGpuNode(node, inputs, onProgress, signal);
  }

  private async runLocalGpuNode(
    node: WorkflowNode,
    inputs: Record<string, CanonicalArtifact>,
    onProgress: (ev: NodeProgressEvent) => void,
    signal?: AbortSignal,
    explicitMaxFoldResidues?: number,
  ): Promise<{ outputs: Record<string, CanonicalArtifact>; elapsedMs: number }> {
    if (signal?.aborted) {
      throw new Error('Workflow run cancelled');
    }

    // Never spawn or reuse a GPU worker while a prior cooperative abort or disposal is settling
    await this.waitForPendingGpuDisposal();

    const plugin = (this.pluginRegistry ?? getActivePluginRegistry())?.getPlugin(node.adapterId);
    // Worker isolation key: plugin hook first, legacy LocalFold ESMFold2 fallback otherwise
    const isolatedCheckpoint = plugin?.execution.getWorkerIsolationKey
      ? plugin.execution.getWorkerIsolationKey(node.params)
      : node.adapterId === 'localfold-af2'
        ? LOCALFOLD_MODEL_OPTIONS.find(
            (m) => m.id === String(node.params.model ?? '').trim() && m.family === 'esmfold2',
          )?.id
        : undefined;
    if (this.activeGpu && (this.activeGpu.adapterId !== node.adapterId ||
        this.activeGpu.isolatedCheckpoint !== isolatedCheckpoint)) {
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
      const worker = await new Promise<Worker>((resolve, reject) => {
        if (signal?.aborted) {
          reject(new Error('Workflow run cancelled'));
          return;
        }
        let settled = false;
        const onAbort = () => {
          if (settled) return;
          settled = true;
          reject(new Error('Workflow run cancelled'));
        };
        signal?.addEventListener('abort', onAbort, { once: true });
        this.spawnWorker(node.adapterId).then(
          (w) => {
            signal?.removeEventListener('abort', onAbort);
            if (settled || signal?.aborted) {
              try {
                w.terminate();
              } catch {
                /* ignore */
              }
              if (!settled) {
                settled = true;
                reject(new Error('Workflow run cancelled'));
              }
              return;
            }
            settled = true;
            resolve(w);
          },
          (err) => {
            signal?.removeEventListener('abort', onAbort);
            if (!settled) {
              settled = true;
              reject(err);
            }
          },
        );
      });
      this.activeGpu = { adapterId: node.adapterId, isolatedCheckpoint, worker };
    }

    const slot = this.activeGpu;
    const worker = slot.worker;
    const id = this.reqCounter++;
    const supportsCooperativeAbort = Boolean(plugin?.execution.supportsCooperativeAbort);
    const abortGraceMs = Math.max(1500, plugin?.execution.abortGraceMs ?? 1500);

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
        if (supportsCooperativeAbort) {
          void this.requestCooperativeAbort(slot, abortGraceMs);
        } else {
          void this.releaseActiveGpuInternal('Workflow run cancelled');
        }
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
        const resolvedLimit = explicitMaxFoldResidues ?? this.foldLimitResolver?.(node)?.maxFoldResidues ?? undefined;
        worker.postMessage({
          id,
          type: 'run',
          params: plainParams,
          inputs,
          ...(typeof resolvedLimit === 'number' && Number.isFinite(resolvedLimit)
            ? { maxFoldResidues: Math.round(resolvedLimit) }
            : {}),
        });
      } catch (err) {
        abort(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  private runCpuNode(
    node: WorkflowNode,
    inputs: Record<string, CanonicalArtifact>,
    onProgress?: (ev: NodeProgressEvent) => void,
    multiInputs?: Record<string, CanonicalArtifact[]>,
    upstreamGraph?: UpstreamGraphQuery,
    upstreamArtifacts?: Readonly<Record<string, Readonly<Record<string, CanonicalArtifact>>>>,
    abortSignal?: AbortSignal,
  ): Promise<Record<string, CanonicalArtifact>> {
    const plugin =
      this.pluginRegistry?.getPlugin(node.adapterId) ??
      getActivePluginRegistry()?.getPlugin(node.adapterId) ??
      BUILTIN_CPU_PLUGINS[node.adapterId];
    if (plugin?.execution.executeCpu) {
      const ctx: CpuExecutionContext = {
        node,
        inputs: inputs as unknown as Record<string, ArtifactEnvelope>,
        multiInputs: (multiInputs ?? {}) as unknown as CpuExecutionContext['multiInputs'],
        upstreamGraph: upstreamGraph ?? {
          getIncomingEdges: () => [],
          getNode: () => undefined,
          getNodeOutputs: () => undefined,
        },
        upstreamArtifacts: (upstreamArtifacts ??
          {}) as unknown as CpuExecutionContext['upstreamArtifacts'],
        onProgress: (ev) => onProgress?.({ ...ev, nodeId: node.id }),
        abortSignal: abortSignal ?? new AbortController().signal,
      };
      return plugin.execution.executeCpu(ctx) as unknown as Promise<
        Record<string, CanonicalArtifact>
      >;
    }

    return Promise.reject(new Error(`Unknown CPU adapter "${node.adapterId}"`));
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
    let order: WorkflowNode[] | null = null;
    let resumeOutputs: Record<string, Record<string, CanonicalArtifact>> | undefined;
    if (errors.length === 0) {
      order = topoSort(graph);
      if (opts.resumeCheckpoint && !opts.bypassCache) {
        try {
          const ckpt =
            typeof opts.resumeCheckpoint === 'string'
              ? deserializeCheckpoint(opts.resumeCheckpoint)
              : opts.resumeCheckpoint;
          if (
            ckpt &&
            ckpt.schema === 'denovo.checkpoint/v1' &&
            ckpt.nodes &&
            typeof ckpt.nodes === 'object'
          ) {
            const matched: Record<string, Record<string, CanonicalArtifact>> = {};
            for (const node of order) {
              const entry = ckpt.nodes[node.id];
              if (!entry || !entry.outputs) continue;
              const manifest = ADAPTER_REGISTRY[node.adapterId];
              if (!manifest) continue;
              const inputIds: Record<string, string | string[]> = {};
              let missingUpstreamMatch = false;
              for (const edge of graph.edges) {
                if (edge.toNode !== node.id) continue;
                const upNodeOutputs = matched[edge.fromNode];
                if (!upNodeOutputs) {
                  missingUpstreamMatch = true;
                  break;
                }
                const up = upNodeOutputs[edge.fromPort];
                if (!up) continue;
                if (manifest.inputs[edge.toPort]?.multiple) {
                  const prev = inputIds[edge.toPort];
                  inputIds[edge.toPort] = Array.isArray(prev) ? [...prev, up.id] : [up.id];
                } else {
                  inputIds[edge.toPort] = up.id;
                }
              }
              if (missingUpstreamMatch) continue;
              const ckptProv = this.executionProvenanceResolver?.(node);
              const cacheKey = fnv1a(
                canonicalJsonStringify({
                  adapter: node.adapterId,
                  params: normalizeParamsForComputation(node.adapterId, node.params),
                  inputIds,
                  ...(ckptProv && ckptProv !== 'local' ? { backend: ckptProv } : {}),
                }),
              );
              if (entry.cacheKey === cacheKey) {
                matched[node.id] = entry.outputs;
              }
            }
            if (Object.keys(matched).length > 0) {
              resumeOutputs = matched;
            }
          }
        } catch {
          // Malformed checkpoint string/object will be reported by importCheckpoint below
          resumeOutputs = undefined;
        }
      }
    }
    const bioFoldLimitOpts = this.foldLimitResolver
      ? { resolveNodeFoldLimits: (n: WorkflowNode) => this.foldLimitResolver?.(n) }
      : undefined;
    for (const issue of validateBioCapabilities(graph, resumeOutputs, bioFoldLimitOpts)) {
      if (issue.severity === 'error' && !errors.includes(issue.message)) {
        errors.push(issue.message);
      }
    }
    if (errors.length) throw new Error(errors.join('\n'));
    order ??= topoSort(graph);
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

    const nodeProvenanceById: Record<string, string> = {};

    try {
      for (const node of order) {
        if (internalAbort.signal.aborted) {
          return await finishCancelled();
        }

        const manifest = resolveManifestForAdapter(node.adapterId, this.pluginRegistry);
        if (!manifest) throw new Error(`Unknown adapter "${node.adapterId}"`);

        this.activeRun = {
          ...this.activeRun,
          activeNodeId: node.id,
          completedNodeIds: [...completedNodeIds],
          unfinishedNodeIds: unfinishedOf(),
        };
        this.emitRunState();

        // Gather upstream inputs (preserving all upstream IDs on multi-input fan-in ports for cache keys)
        const inputs: Record<string, CanonicalArtifact> = {};
        const multiInputs: Record<string, CanonicalArtifact[]> = {};
        const inputIds: Record<string, string | string[]> = {};
        const upstreamProv: Record<string, string | string[]> = {};
        for (const edge of graph.edges) {
          if (edge.toNode !== node.id) continue;
          const up = outputsByNode[edge.fromNode]?.[edge.fromPort];
          if (up) {
            const upProv = nodeProvenanceById[edge.fromNode] ?? 'local';
            const isMulti = Boolean(manifest.inputs[edge.toPort]?.multiple);
            if (!isMulti || !inputs[edge.toPort]) {
              inputs[edge.toPort] = up;
            }
            if (isMulti) {
              const prevList = multiInputs[edge.toPort];
              multiInputs[edge.toPort] = prevList ? [...prevList, up] : [up];
              const prev = inputIds[edge.toPort];
              inputIds[edge.toPort] = Array.isArray(prev) ? [...prev, up.id] : [up.id];
              const prevP = upstreamProv[edge.toPort];
              upstreamProv[edge.toPort] = Array.isArray(prevP) ? [...prevP, upProv] : [upProv];
            } else {
              inputIds[edge.toPort] = up.id;
              upstreamProv[edge.toPort] = upProv;
            }
          }
        }

        // Content-addressed memoization key (canonicalized key order + propagated compute provenance boundary, M6)
        const ownProv = this.executionProvenanceResolver?.(node) ?? 'local';
        const hasNonLocalUpstream = Object.values(upstreamProv).some((p) =>
          Array.isArray(p) ? p.some((x) => x !== 'local') : p !== 'local',
        );
        const effectiveProv =
          ownProv !== 'local' || hasNonLocalUpstream
            ? hasNonLocalUpstream
              ? `${ownProv}|${fnv1a(canonicalJsonStringify(upstreamProv))}`
              : ownProv
            : 'local';
        nodeProvenanceById[node.id] = effectiveProv;

        const cacheKey = fnv1a(
          canonicalJsonStringify({
            adapter: node.adapterId,
            params: normalizeParamsForComputation(node.adapterId, node.params, this.pluginRegistry),
            inputIds,
            ...(effectiveProv !== 'local' ? { backend: effectiveProv } : {}),
          }),
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
          nodeOutputs = validateAdapterOutputs(node, cachedOutputs, this.pluginRegistry);
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
              const upstreamGraph: UpstreamGraphQuery = {
                getIncomingEdges: (nid, toPort) =>
                  graph.edges.filter(
                    (e) => e.toNode === nid && (toPort === undefined || e.toPort === toPort),
                  ),
                getNode: (nid) => graph.nodes.find((gn) => gn.id === nid),
                getNodeOutputs: (nid) =>
                  outputsByNode[nid] as unknown as Record<string, ArtifactEnvelope> | undefined,
              };
              rawOutputs = await this.runCpuNode(
                node,
                inputs,
                onProgress,
                multiInputs,
                upstreamGraph,
                outputsByNode,
                internalAbort.signal,
              );
              elapsedMs = performance.now() - nStart;
            } else {
              // Layer 2 biological capability guard before launching GPU worker / downloading weights
              const foldOpts = this.foldLimitResolver?.(node);
              const guard = validateRuntimeNodeInputs(node, inputs, foldOpts);
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
            nodeOutputs = validateAdapterOutputs(node, rawOutputs, this.pluginRegistry);

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
  if (art.kind !== 'DesignabilityMetrics') {
    // Generic Layer 1 / multi-domain PortType fallback (e.g. future PropertyTable / SmallMolecule)
    const rawArt = art as unknown as { id: string; kind: string };
    const registry = getActivePluginRegistry();
    const desc = registry?.portTypes.get(rawArt.kind);
    const summary = desc
      ? desc.summarize(hydrateLegacyArtifactToEnvelope(rawArt, desc.domainId))
      : { kind: rawArt.kind };
    return {
      id: rawArt.id,
      kind: rawArt.kind as ArtifactRef['kind'],
      summary,
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
