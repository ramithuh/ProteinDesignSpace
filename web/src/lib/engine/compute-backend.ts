// Pluggable compute backend routing & SSE client for self-hosted host-GPU WebGPU execution.
// Supports three modes:
//   - 'local'  : all GPU models run in local browser WebGPU workers (default on denovo.run)
//   - 'remote' : all GPU models run on the self-hosted machine's GPU (zero Python deps)
//   - 'hybrid' : lightweight GPU models (<= hybridThresholdMB) run locally when hardware WebGPU
//                is available; heavy models (or devices without WebGPU) route to the host GPU.

import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS, RFD4_BENCHMARK_OPTIONS } from './registry';
import { decodeWireValue, encodeWireValue } from './remote-codec';
import type { CanonicalArtifact, NodeProgressEvent, WorkflowNode } from './types';

export type ComputeRoutingMode = 'local' | 'remote' | 'hybrid';

export interface ComputeBackendConfig {
  mode: ComputeRoutingMode;
  /** Base URL of self-hosted server (empty string = same origin when served by selfhost-server.mjs). */
  serverUrl: string;
  /** Optional Bearer API token when server sets DENOVO_API_TOKEN. */
  apiToken: string;
  /** In 'hybrid' mode, models whose active weight download exceeds this MB threshold run on host GPU. */
  hybridThresholdMB: number;
  /**
   * Optional user override for maximum LocalFold residue ceiling (512–4096).
   * `null` = Auto (512 AF3 / 768 AF2 in local browser; server ceiling / 2048 on Host GPU).
   */
  maxFoldResidues: number | null;
}

export interface RemoteServerHealth {
  ok: boolean;
  service?: string;
  version?: string;
  gpu?: {
    vendor: string;
    architecture: string;
    description: string;
    name?: string;
    shortName?: string;
    vramMB?: number | null;
    fallback: boolean;
  } | null;
  activeAdapter?: string | null;
  queueDepth?: number;
  runningJobId?: string | null;
  maxFoldResidues?: number;
  error?: string;
}

export const COMPUTE_BACKEND_STORAGE_KEY = 'denovo.computeBackend.v1';
export const DEFAULT_HYBRID_THRESHOLD_MB = 40;
export const DEFAULT_HOST_MAX_FOLD_RESIDUES = 2048;
export const MIN_CUSTOM_FOLD_RESIDUES = 512;
export const MAX_CUSTOM_FOLD_RESIDUES = 4096;

export function sanitizeMaxFoldResidues(raw: unknown): number | null {
  if (raw === null || raw === undefined || raw === '' || raw === 'auto') return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  return Math.max(MIN_CUSTOM_FOLD_RESIDUES, Math.min(MAX_CUSTOM_FOLD_RESIDUES, Math.round(n)));
}

export function defaultComputeBackendConfig(): ComputeBackendConfig {
  return {
    mode: 'local',
    serverUrl: '',
    apiToken: '',
    hybridThresholdMB: DEFAULT_HYBRID_THRESHOLD_MB,
    maxFoldResidues: null,
  };
}

export function normalizeServerBaseUrl(raw: string): string {
  const trimmed = raw.trim().replace(/\/+$/, '');
  return trimmed;
}

export function loadComputeBackendConfig(): ComputeBackendConfig {
  const base = defaultComputeBackendConfig();
  if (typeof localStorage === 'undefined') return base;
  try {
    const raw = localStorage.getItem(COMPUTE_BACKEND_STORAGE_KEY);
    if (!raw) return base;
    const parsed = JSON.parse(raw) as Partial<ComputeBackendConfig>;
    const mode: ComputeRoutingMode =
      parsed.mode === 'remote' || parsed.mode === 'hybrid' ? parsed.mode : 'local';
    const threshold = Number(parsed.hybridThresholdMB);
    return {
      mode,
      serverUrl: typeof parsed.serverUrl === 'string' ? normalizeServerBaseUrl(parsed.serverUrl) : '',
      apiToken: typeof parsed.apiToken === 'string' ? parsed.apiToken.trim() : '',
      hybridThresholdMB:
        Number.isFinite(threshold) && threshold >= 0 && threshold <= 2000
          ? Math.round(threshold)
          : DEFAULT_HYBRID_THRESHOLD_MB,
      maxFoldResidues: sanitizeMaxFoldResidues(parsed.maxFoldResidues),
    };
  } catch {
    return base;
  }
}

export function saveComputeBackendConfig(cfg: ComputeBackendConfig): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(
      COMPUTE_BACKEND_STORAGE_KEY,
      JSON.stringify({
        mode: cfg.mode,
        serverUrl: normalizeServerBaseUrl(cfg.serverUrl),
        apiToken: cfg.apiToken.trim(),
        hybridThresholdMB: cfg.hybridThresholdMB,
        maxFoldResidues: sanitizeMaxFoldResidues(cfg.maxFoldResidues),
      }),
    );
  } catch {
    /* ignore storage quota / private mode */
  }
}

/** Format a compact GPU label for the header status pill (e.g. "Host · RTX 5090"). */
export function formatHostGpuPillLabel(
  health: RemoteServerHealth | null,
  mode: ComputeRoutingMode,
): string {
  const prefix = mode === 'hybrid' ? 'Hybrid' : 'Host';
  if (!health?.ok) return `${prefix} GPU (!)`;
  const short =
    health.gpu?.shortName?.trim() ||
    health.gpu?.name?.trim() ||
    '';
  if (short) return `${prefix} · ${short}`;
  return `${prefix} GPU`;
}

/** Format a rich GPU summary for the popover status line (e.g. "NVIDIA GeForce RTX 5090 (32 GB)"). */
export function formatHostGpuDetail(health: RemoteServerHealth | null): string {
  if (!health?.ok) return health?.error ? `Offline: ${health.error}` : 'Not tested yet';
  const g = health.gpu;
  if (!g) return 'Host GPU ready';
  if (g.name) {
    const gb = g.vramMB && g.vramMB >= 1024 ? ` · ${Math.round(g.vramMB / 1024)} GB` : '';
    return `${g.name}${gb}`;
  }
  return g.description || g.vendor || 'Host GPU ready';
}

/** Resolve the effective folding residue ceiling for a node given Settings + Host GPU routing. */
export function resolveNodeFoldLimitOptions(
  node: WorkflowNode,
  config: ComputeBackendConfig,
  remoteHealth?: RemoteServerHealth | null,
  opts?: { localWebGpuAvailable?: boolean; localWebGpuFallback?: boolean },
): { maxFoldResidues?: number; isHostGpu: boolean } {
  const isHostGpu = shouldRouteNodeToRemote(node, config, opts);
  const userOverride = sanitizeMaxFoldResidues(config.maxFoldResidues);
  if (isHostGpu) {
    const hostDefault =
      sanitizeMaxFoldResidues(remoteHealth?.maxFoldResidues) ?? DEFAULT_HOST_MAX_FOLD_RESIDUES;
    return {
      maxFoldResidues: userOverride ?? hostDefault,
      isHostGpu: true,
    };
  }
  return {
    ...(userOverride !== null ? { maxFoldResidues: userOverride } : {}),
    isHostGpu: false,
  };
}

/** Estimate active weight size (MB) for routing decisions in hybrid mode. */
export function estimateNodeActiveWeightsMB(node: WorkflowNode): number {
  const manifest = ADAPTER_REGISTRY[node.adapterId];
  if (!manifest) return 0;
  if (node.adapterId === 'localfold-af2') {
    const modelId = String(node.params?.model ?? 'monomer');
    const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
    return meta.approxMB;
  }
  if (node.adapterId === 'rfd4-proteina') {
    const mode = String(node.params?.mode ?? 'unconditional');
    const needsEnc =
      mode === 'partial' ||
      mode === 'inverse-folding' ||
      (mode === 'preset' &&
        Boolean(
          RFD4_BENCHMARK_OPTIONS.find((o) => o.id === String(node.params?.presetId ?? ''))
            ?.needsEncoder,
        ));
    return (manifest.approxWeightsMB ?? 330) + (needsEnc ? 138 : 0);
  }
  const prec = String(node.params?.precision ?? '').trim().toLowerCase();
  if (prec && manifest.precisionWeightsMB && prec in manifest.precisionWeightsMB) {
    return manifest.precisionWeightsMB[prec];
  }
  return manifest.approxWeightsMB ?? 0;
}

/** Decide whether a GPU node should execute on the self-hosted remote server. */
export function shouldRouteNodeToRemote(
  node: WorkflowNode,
  config: ComputeBackendConfig,
  opts?: { localWebGpuAvailable?: boolean; localWebGpuFallback?: boolean },
): boolean {
  const manifest = ADAPTER_REGISTRY[node.adapterId];
  if (!manifest || manifest.runtime === 'cpu-ts') return false;
  if (config.mode === 'local') return false;
  if (config.mode === 'remote') return true;
  // Hybrid mode: route to host GPU if client lacks hardware WebGPU OR model exceeds weight threshold
  if (opts?.localWebGpuAvailable === false || opts?.localWebGpuFallback === true) {
    return true;
  }
  return estimateNodeActiveWeightsMB(node) > config.hybridThresholdMB;
}

function buildApiUrl(serverUrl: string, path: string): string {
  const base = normalizeServerBaseUrl(serverUrl);
  return base ? `${base}${path}` : path;
}

function buildAuthHeaders(apiToken: string, extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...(extra ?? {}) };
  if (apiToken.trim()) {
    headers.Authorization = `Bearer ${apiToken.trim()}`;
  }
  return headers;
}

export async function checkRemoteServerHealth(
  serverUrl: string,
  apiToken = '',
  fetchImpl: typeof fetch = fetch,
  timeoutMs = 3500,
): Promise<RemoteServerHealth> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetchImpl(buildApiUrl(serverUrl, '/api/health'), {
      method: 'GET',
      headers: buildAuthHeaders(apiToken),
      signal: ctrl.signal,
    });
    const text = await resp.text();
    let json: Record<string, unknown> | null = null;
    try {
      json = JSON.parse(text) as Record<string, unknown>;
    } catch {
      json = null;
    }
    if (!resp.ok || !json || json.service !== 'denovo-selfhost') {
      return {
        ok: false,
        error:
          (typeof json?.error === 'string' && json.error) ||
          `HTTP ${resp.status} from ${buildApiUrl(serverUrl, '/api/health')}`,
      };
    }
    return {
      ok: Boolean(json.ok),
      service: String(json.service),
      version: typeof json.version === 'string' ? json.version : undefined,
      gpu: (json.gpu as RemoteServerHealth['gpu']) ?? null,
      activeAdapter: typeof json.activeAdapter === 'string' ? json.activeAdapter : null,
      queueDepth: typeof json.queueDepth === 'number' ? json.queueDepth : 0,
      runningJobId: typeof json.runningJobId === 'string' ? json.runningJobId : null,
      maxFoldResidues: sanitizeMaxFoldResidues(json.maxFoldResidues) ?? undefined,
      error: typeof json.error === 'string' ? json.error : undefined,
    };
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    };
  } finally {
    clearTimeout(timer);
  }
}

export async function releaseRemoteServerGpu(
  serverUrl: string,
  apiToken = '',
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  await fetchImpl(buildApiUrl(serverUrl, '/api/release-gpu'), {
    method: 'POST',
    headers: buildAuthHeaders(apiToken, { 'Content-Type': 'application/json' }),
    body: '{}',
  });
}

function makeJobId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `job_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Execute one GPU workflow node on the self-hosted server over HTTP + SSE,
 * streaming progress events back to `onProgress` and decoding TypedArrays transparently.
 */
export async function runNodeOnRemoteServer(opts: {
  serverUrl: string;
  apiToken?: string;
  node: WorkflowNode;
  inputs: Record<string, CanonicalArtifact>;
  onProgress: (ev: NodeProgressEvent) => void;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
}): Promise<{ outputs: Record<string, CanonicalArtifact>; elapsedMs: number }> {
  const { serverUrl, apiToken = '', node, inputs, onProgress, signal } = opts;
  const fetchImpl = opts.fetchImpl ?? fetch;
  if (signal?.aborted) {
    throw new Error('Workflow run cancelled');
  }

  const jobId = makeJobId();
  const plainNode: WorkflowNode = {
    id: node.id,
    adapterId: node.adapterId,
    label: node.label,
    params: JSON.parse(JSON.stringify(node.params ?? {})),
  };

  onProgress({
    nodeId: node.id,
    status: 'loading',
    phase: 'remote',
    progress: 0.01,
    detail: `Sending ${node.label ?? node.adapterId} to host GPU...`,
  });

  const cancelRemote = () => {
    void fetchImpl(buildApiUrl(serverUrl, '/api/cancel'), {
      method: 'POST',
      headers: buildAuthHeaders(apiToken, { 'Content-Type': 'application/json' }),
      body: JSON.stringify({ jobId }),
    }).catch(() => {});
  };

  const onAbort = () => cancelRemote();
  signal?.addEventListener('abort', onAbort, { once: true });

  try {
    const resp = await fetchImpl(buildApiUrl(serverUrl, '/api/run-node'), {
      method: 'POST',
      headers: buildAuthHeaders(apiToken, {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      }),
      body: JSON.stringify({
        jobId,
        node: plainNode,
        inputs: encodeWireValue(inputs),
      }),
      signal,
    });

    if (!resp.ok) {
      let msg = `Host GPU server returned HTTP ${resp.status}`;
      try {
        const errJson = (await resp.json()) as { error?: string };
        if (errJson?.error) msg = errJson.error;
      } catch {
        /* ignore */
      }
      throw new Error(msg);
    }

    if (!resp.body) {
      throw new Error('Host GPU server returned an empty response stream.');
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let resultPayload: { outputs: Record<string, CanonicalArtifact>; elapsedMs: number } | null = null;
    let streamError: string | null = null;

    const handleFrame = (rawFrame: string) => {
      const lines = rawFrame.split(/\r?\n/);
      let eventName = 'message';
      const dataLines: string[] = [];
      for (const line of lines) {
        if (!line || line.startsWith(':')) continue;
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      if (dataLines.length === 0) return;
      const dataText = dataLines.join('\n');
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(dataText) as Record<string, unknown>;
      } catch {
        return;
      }
      if (eventName === 'progress') {
        const detailRaw = typeof data.detail === 'string' ? data.detail : '';
        onProgress({
          nodeId: node.id,
          status:
            data.status === 'loading' || data.status === 'running'
              ? data.status
              : data.phase === 'weights' || data.phase === 'init' || data.phase === 'queue'
                ? 'loading'
                : 'running',
          phase: typeof data.phase === 'string' ? data.phase : 'remote',
          progress: typeof data.progress === 'number' ? data.progress : undefined,
          detail: detailRaw ? `[Host GPU] ${detailRaw}` : '[Host GPU] Running...',
        });
      } else if (eventName === 'result') {
        const decodedOutputs = decodeWireValue<Record<string, CanonicalArtifact>>(data.outputs ?? {});
        const elapsedMs = typeof data.elapsedMs === 'number' ? data.elapsedMs : 0;
        resultPayload = { outputs: decodedOutputs, elapsedMs };
      } else if (eventName === 'error') {
        streamError = typeof data.message === 'string' ? data.message : 'Host GPU execution failed';
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sepIdx: number;
      while ((sepIdx = buffer.search(/\r?\n\r?\n/)) !== -1) {
        const frame = buffer.slice(0, sepIdx);
        const matchLen = buffer.slice(sepIdx).startsWith('\r\n\r\n') ? 4 : 2;
        buffer = buffer.slice(sepIdx + matchLen);
        handleFrame(frame);
      }
    }
    if (buffer.trim()) {
      handleFrame(buffer);
    }

    if (streamError) {
      throw new Error(streamError);
    }
    if (!resultPayload) {
      throw new Error(
        signal?.aborted
          ? 'Workflow run cancelled'
          : 'Host GPU stream ended before returning a result.',
      );
    }
    return resultPayload;
  } catch (err) {
    if (signal?.aborted) {
      throw new Error('Workflow run cancelled');
    }
    throw err;
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
}
