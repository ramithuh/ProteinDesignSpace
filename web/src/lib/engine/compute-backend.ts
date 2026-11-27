// Pluggable compute backend routing & SSE client for self-hosted host-GPU WebGPU execution.
// Hardened:
//   - Never persists apiToken in localStorage/sessionStorage (in memory only, purges legacy stored tokens)
//   - Enforces HTTPS outside loopback, rejects URL credentials/redirects
//   - Clamps Host GPU folding residue ceiling to server's advertised maxFoldResidues
//   - Uses server-issued { jobId, cancelToken } capability tokens for cancellation
//   - Enforces Content-Type, response byte budgets, and immediate reader cancel on terminal SSE events

import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS, RFD4_BENCHMARK_OPTIONS } from './registry';
import { decodeWireValue, encodeWireValue } from './remote-codec';
import type { CanonicalArtifact, NodeProgressEvent, WorkflowNode } from './types';
import type { WebGpuHardwareCapabilities } from './webgpu-device';

export type ComputeRoutingMode = 'local' | 'remote' | 'hybrid';

export interface ComputeBackendConfig {
  mode: ComputeRoutingMode;
  /** Base URL of self-hosted server (empty string = same origin when served by selfhost-server.mjs). */
  serverUrl: string;
  /** Optional Bearer API token (kept in memory only; never saved to localStorage or sessionStorage). */
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
  busy?: boolean;
  maxFoldResidues?: number;
  authRequired?: boolean;
  authenticatedBy?: 'cookie' | 'bearer' | 'none';
  error?: string;
}

export const COMPUTE_BACKEND_STORAGE_KEY = 'denovo.computeBackend.v1';
const LEGACY_COMPUTE_SESSION_TOKEN_KEY = 'denovo.computeBackend.sessionToken.v1';
export const DEFAULT_HYBRID_THRESHOLD_MB = 40;
export const DEFAULT_HOST_MAX_FOLD_RESIDUES = 2048;
export const MIN_CUSTOM_FOLD_RESIDUES = 512;
export const MAX_CUSTOM_FOLD_RESIDUES = 4096;

const MAX_HEALTH_RESPONSE_BYTES = 64 * 1024;
const MAX_SSE_BUFFER_CHARS = 128 * 1024 * 1024;

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
  return raw.trim().replace(/\/+$/, '');
}

function isLoopbackHostname(hostname: string): boolean {
  const h = hostname.toLowerCase();
  return h === 'localhost' || h === '127.0.0.1' || h === '[::1]' || h === '::1';
}

/**
 * Validate and canonicalize a remote backend URL (H7).
 * Empty string means same-origin. Non-loopback URLs must use HTTPS and contain no credentials/query/hash.
 */
export function validateRemoteServerUrl(raw: string): {
  ok: boolean;
  baseUrl: string;
  origin: string;
  error?: string;
} {
  const trimmed = normalizeServerBaseUrl(raw);
  if (!trimmed) {
    if (typeof window !== 'undefined' && window.location) {
      const { protocol, hostname, origin } = window.location;
      if (protocol === 'http:' && !isLoopbackHostname(hostname)) {
        return {
          ok: false,
          baseUrl: '',
          origin,
          error:
            'Same-origin Self-Hosted server over plain http:// is only allowed on localhost/127.0.0.1. Use https:// (e.g. Tailscale Serve) or SSH port forwarding.',
        };
      }
      return { ok: true, baseUrl: '', origin: origin || 'same-origin' };
    }
    return { ok: true, baseUrl: '', origin: 'same-origin' };
  }
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return { ok: false, baseUrl: trimmed, origin: '', error: 'Invalid Host Server URL.' };
  }
  if (parsed.username || parsed.password) {
    return {
      ok: false,
      baseUrl: trimmed,
      origin: '',
      error: 'Host Server URL must not contain embedded credentials.',
    };
  }
  if (parsed.search || parsed.hash) {
    return {
      ok: false,
      baseUrl: trimmed,
      origin: '',
      error: 'Host Server URL must not contain query parameters or fragments.',
    };
  }
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
    return {
      ok: false,
      baseUrl: trimmed,
      origin: '',
      error: 'Host Server URL must use https:// (or http:// for localhost only).',
    };
  }
  if (parsed.protocol === 'http:' && !isLoopbackHostname(parsed.hostname)) {
    return {
      ok: false,
      baseUrl: trimmed,
      origin: '',
      error: 'Non-loopback Host Server URLs must use https:// (e.g. Tailscale Serve or reverse proxy TLS).',
    };
  }
  return {
    ok: true,
    baseUrl: parsed.origin,
    origin: parsed.origin.toLowerCase(),
  };
}

/** Return the canonical backend origin string for trust-boundary & token-isolation checks. */
export function resolveRemoteBackendOrigin(serverUrl: string): string {
  const check = validateRemoteServerUrl(serverUrl);
  return check.ok ? check.origin : normalizeServerBaseUrl(serverUrl).toLowerCase();
}

/** Remove the tab-scoped Bearer token that earlier builds kept in sessionStorage. */
function purgeLegacySessionToken(): void {
  if (typeof sessionStorage === 'undefined') return;
  try {
    sessionStorage.removeItem(LEGACY_COMPUTE_SESSION_TOKEN_KEY);
  } catch {
    /* ignore storage access errors (private mode / disabled storage) */
  }
}

export function loadComputeBackendConfig(): ComputeBackendConfig {
  purgeLegacySessionToken();
  const base = defaultComputeBackendConfig();
  if (typeof localStorage === 'undefined') return base;
  try {
    const raw = localStorage.getItem(COMPUTE_BACKEND_STORAGE_KEY);
    if (!raw) return base;
    const parsed = JSON.parse(raw) as Partial<ComputeBackendConfig>;
    const mode: ComputeRoutingMode =
      parsed.mode === 'remote' || parsed.mode === 'hybrid' ? parsed.mode : 'local';
    const threshold = Number(parsed.hybridThresholdMB);
    const serverUrl =
      typeof parsed.serverUrl === 'string' ? normalizeServerBaseUrl(parsed.serverUrl) : '';
    const cfg: ComputeBackendConfig = {
      mode,
      serverUrl,
      // M5: Bearer tokens live in memory only and are never hydrated from Web Storage
      // (same-origin hosts keep sessions via the HttpOnly cookie; cross-origin hosts re-prompt).
      apiToken: '',
      hybridThresholdMB:
        Number.isFinite(threshold) && threshold >= 0 && threshold <= 2000
          ? Math.round(threshold)
          : DEFAULT_HYBRID_THRESHOLD_MB,
      maxFoldResidues: sanitizeMaxFoldResidues(parsed.maxFoldResidues),
    };
    // Migrate legacy storage entries immediately to purge any previously persisted token
    if (typeof parsed.apiToken === 'string' && parsed.apiToken.length > 0) {
      saveComputeBackendConfig(cfg);
    }
    return cfg;
  } catch {
    return base;
  }
}

export function saveComputeBackendConfig(cfg: ComputeBackendConfig): void {
  if (typeof localStorage === 'undefined') return;
  try {
    // M5: Persist non-secret routing preferences only — apiToken is never saved to Web Storage
    localStorage.setItem(
      COMPUTE_BACKEND_STORAGE_KEY,
      JSON.stringify({
        mode: cfg.mode,
        serverUrl: normalizeServerBaseUrl(cfg.serverUrl),
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
  const short = health.gpu?.shortName?.trim() || health.gpu?.name?.trim() || '';
  if (short) return `${prefix} · ${short}`;
  return `${prefix} GPU`;
}

/** Format a rich GPU summary for the popover status line (e.g. "NVIDIA GeForce RTX 5090 · 32 GB"). */
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

const MIB = 1024 * 1024;
const GIB = 1024 * 1024 * 1024;

/**
 * Compute a safe Browser `Auto` residue ceiling for a LocalFold model from probed
 * WebGPU adapter limits and system memory capabilities.
 */
export function computeSafeBrowserFoldCeiling(
  modelId: string,
  hardwareCaps?: WebGpuHardwareCapabilities | null,
): number {
  const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
  const familyBaseline = meta.family === 'af2' || meta.family === 'af2-multimer' ? 768 : 512;
  const maxStorageBindingBytes = hardwareCaps?.limits?.maxStorageBufferBindingSize;
  if (!hardwareCaps || typeof maxStorageBindingBytes !== 'number' || !Number.isFinite(maxStorageBindingBytes) || maxStorageBindingBytes <= 0) {
    return familyBaseline;
  }

  const pairChannels = meta.pairChannels ?? 128;
  const tokenExpansionFactor = meta.tokenExpansionFactor ?? 1;
  const rawBindingTokens = Math.floor(Math.sqrt(maxStorageBindingBytes / (pairChannels * 4)));
  const bindingCeiling = Math.floor(rawBindingTokens / tokenExpansionFactor);

  if (meta.family === 'esmfold2') {
    return 512;
  }

  const hasCapableRam =
    (hardwareCaps.deviceMemoryGiB !== null && hardwareCaps.deviceMemoryGiB >= 8) ||
    (hardwareCaps.estimatedMemoryBudgetBytes !== null && hardwareCaps.estimatedMemoryBudgetBytes >= 2 * GIB);
  const isCapableDesktop =
    !hardwareCaps.isSoftwareFallback &&
    maxStorageBindingBytes >= 256 * MIB &&
    hasCapableRam;

  if (isCapableDesktop) {
    if (meta.family === 'af2' || meta.family === 'af2-multimer') {
      return Math.max(64, Math.min(768, bindingCeiling));
    }
    if (pairChannels <= 128 && tokenExpansionFactor === 1) {
      return Math.max(64, Math.min(768, bindingCeiling));
    }
    return Math.max(64, Math.min(512, bindingCeiling));
  }

  if (meta.family === 'af2' || meta.family === 'af2-multimer') {
    const isLowRam = hardwareCaps.deviceMemoryGiB !== null && hardwareCaps.deviceMemoryGiB <= 4;
    const cap = hardwareCaps.isSoftwareFallback || isLowRam ? 512 : 768;
    return Math.max(64, Math.min(cap, bindingCeiling));
  }

  return Math.max(64, Math.min(512, bindingCeiling));
}

/**
 * Resolve the effective folding residue ceiling for a node given Settings + Host GPU routing.
 * H4: On Host GPU, never exceed the operator's advertised `remoteHealth.maxFoldResidues`.
 */
export function resolveNodeFoldLimitOptions(
  node: WorkflowNode,
  config: ComputeBackendConfig,
  remoteHealth?: RemoteServerHealth | null,
  opts?: {
    localWebGpuAvailable?: boolean;
    localWebGpuFallback?: boolean;
    localHardwareCaps?: WebGpuHardwareCapabilities | null;
  },
): { maxFoldResidues?: number; isHostGpu: boolean } {
  const isHostGpu = shouldRouteNodeToRemote(node, config, opts);
  const userOverride = sanitizeMaxFoldResidues(config.maxFoldResidues);
  if (isHostGpu) {
    const hostDefault =
      sanitizeMaxFoldResidues(remoteHealth?.maxFoldResidues) ?? DEFAULT_HOST_MAX_FOLD_RESIDUES;
    return {
      maxFoldResidues: userOverride !== null ? Math.min(userOverride, hostDefault) : hostDefault,
      isHostGpu: true,
    };
  }
  if (userOverride !== null) {
    return { maxFoldResidues: userOverride, isHostGpu: false };
  }
  if (opts?.localHardwareCaps && node.adapterId === 'localfold-af2') {
    const modelId = String(node.params?.model ?? 'monomer').trim();
    return {
      maxFoldResidues: computeSafeBrowserFoldCeiling(modelId, opts.localHardwareCaps),
      isHostGpu: false,
    };
  }
  return { isHostGpu: false };
}

/** Estimate active weight size (MB) for routing decisions in hybrid mode. */
export function estimateNodeActiveWeightsMB(node: WorkflowNode): number {
  const manifest = ADAPTER_REGISTRY[node.adapterId];
  if (!manifest) return 0;
  if (node.adapterId === 'localfold-af2') {
    const modelId = String(node.params?.model ?? 'monomer').trim();
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
  if (opts?.localWebGpuAvailable === false || opts?.localWebGpuFallback === true) {
    return true;
  }
  return estimateNodeActiveWeightsMB(node) > config.hybridThresholdMB;
}

function buildApiUrl(serverUrl: string, path: string): string {
  const check = validateRemoteServerUrl(serverUrl);
  if (!check.ok) {
    throw new Error(check.error || 'Invalid Host Server URL.');
  }
  return check.baseUrl ? `${check.baseUrl}${path}` : path;
}

function buildAuthHeaders(apiToken: string, extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...(extra ?? {}) };
  if (apiToken.trim()) {
    headers.Authorization = `Bearer ${apiToken.trim()}`;
  }
  return headers;
}

async function readBoundedText(resp: Response, maxBytes: number): Promise<string> {
  if (!resp.body) {
    const t = await resp.text();
    if (t.length > maxBytes) throw new Error('Response exceeded maximum allowed size.');
    return t;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let total = 0;
  let out = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        void reader.cancel().catch(() => {});
        throw new Error('Response exceeded maximum allowed size.');
      }
      out += decoder.decode(value, { stream: true });
    }
    out += decoder.decode();
    return out;
  } finally {
    reader.releaseLock();
  }
}

function fetchCredentialsMode(serverUrl: string): RequestCredentials {
  return normalizeServerBaseUrl(serverUrl) ? 'omit' : 'same-origin';
}

export async function authenticateRemoteServerSession(
  serverUrl: string,
  token: string,
  fetchImpl: typeof fetch = fetch,
): Promise<{ ok: boolean; authMode?: 'cookie' | 'none'; error?: string }> {
  const urlCheck = validateRemoteServerUrl(serverUrl);
  if (!urlCheck.ok) return { ok: false, error: urlCheck.error };
  try {
    const resp = await fetchImpl(buildApiUrl(serverUrl, '/api/auth'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: fetchCredentialsMode(serverUrl),
      redirect: 'error',
      body: JSON.stringify({ token: token.trim() }),
    });
    const text = await readBoundedText(resp, MAX_HEALTH_RESPONSE_BYTES);
    let json: Record<string, unknown> | null = null;
    try {
      json = JSON.parse(text) as Record<string, unknown>;
    } catch {
      json = null;
    }
    if (!resp.ok || !json?.ok) {
      return {
        ok: false,
        error:
          (typeof json?.error === 'string' && json.error) ||
          `Authentication failed (HTTP ${resp.status}).`,
      };
    }
    return {
      ok: true,
      authMode: json.authMode === 'cookie' ? 'cookie' : 'none',
    };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

export async function logoutRemoteServerSession(
  serverUrl: string,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  const urlCheck = validateRemoteServerUrl(serverUrl);
  if (!urlCheck.ok) return;
  try {
    await fetchImpl(buildApiUrl(serverUrl, '/api/logout'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: fetchCredentialsMode(serverUrl),
      redirect: 'error',
      body: '{}',
    });
  } catch {
    /* ignore */
  }
}

export async function checkRemoteServerHealth(
  serverUrl: string,
  apiToken = '',
  fetchImpl: typeof fetch = fetch,
  timeoutMs = 3500,
): Promise<RemoteServerHealth> {
  const urlCheck = validateRemoteServerUrl(serverUrl);
  if (!urlCheck.ok) {
    return { ok: false, error: urlCheck.error };
  }
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetchImpl(buildApiUrl(serverUrl, '/api/health'), {
      method: 'GET',
      headers: buildAuthHeaders(apiToken),
      credentials: fetchCredentialsMode(serverUrl),
      redirect: 'error',
      signal: ctrl.signal,
    });
    const text = await readBoundedText(resp, MAX_HEALTH_RESPONSE_BYTES);
    let json: Record<string, unknown> | null = null;
    try {
      json = JSON.parse(text) as Record<string, unknown>;
    } catch {
      json = null;
    }
    if (!resp.ok || !json || json.service !== 'denovo-selfhost') {
      const looksHtml = resp.ok && !json && /^\s*<!doctype|^\s*<html/i.test(text);
      return {
        ok: false,
        error:
          (typeof json?.error === 'string' && json.error) ||
          (looksHtml
            ? 'No self-hosted GPU server at this URL'
            : `HTTP ${resp.status} from ${buildApiUrl(serverUrl, '/api/health')}`),
      };
    }
    return {
      ok: Boolean(json.ok),
      service: String(json.service),
      version: typeof json.version === 'string' ? json.version : undefined,
      gpu: (json.gpu as RemoteServerHealth['gpu']) ?? null,
      activeAdapter: typeof json.activeAdapter === 'string' ? json.activeAdapter : null,
      queueDepth: typeof json.queueDepth === 'number' ? json.queueDepth : 0,
      busy: Boolean(json.busy),
      maxFoldResidues: sanitizeMaxFoldResidues(json.maxFoldResidues) ?? undefined,
      authRequired: typeof json.authRequired === 'boolean' ? json.authRequired : undefined,
      authenticatedBy:
        json.authenticatedBy === 'cookie' ||
        json.authenticatedBy === 'bearer' ||
        json.authenticatedBy === 'none'
          ? json.authenticatedBy
          : undefined,
      error: typeof json.error === 'string' ? json.error : undefined,
    };
  } catch (err) {
    const rawMsg = err instanceof Error ? err.message : String(err);
    const isGenericNetworkErr =
      /networkerror|failed to fetch|load failed/i.test(rawMsg);
    if (isGenericNetworkErr && normalizeServerBaseUrl(serverUrl)) {
      const pageOrigin =
        typeof window !== 'undefined' && window.location?.origin
          ? window.location.origin
          : 'https://denovo.run';
      return {
        ok: false,
        error: `Unreachable or blocked by CORS — open the server URL directly, or start the server with DENOVO_ALLOWED_ORIGINS=${pageOrigin}`,
      };
    }
    return {
      ok: false,
      error: rawMsg,
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
    credentials: fetchCredentialsMode(serverUrl),
    redirect: 'error',
    body: '{}',
  });
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

  // Server issues { jobId, cancelToken } via the opening 'started' SSE frame (M1)
  let serverJobId = '';
  let serverCancelToken = '';
  const cancelRemote = () => {
    if (!serverJobId || !serverCancelToken) return;
    void fetchImpl(buildApiUrl(serverUrl, '/api/cancel'), {
      method: 'POST',
      headers: buildAuthHeaders(apiToken, { 'Content-Type': 'application/json' }),
      credentials: fetchCredentialsMode(serverUrl),
      redirect: 'error',
      body: JSON.stringify({ jobId: serverJobId, cancelToken: serverCancelToken }),
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
      credentials: fetchCredentialsMode(serverUrl),
      redirect: 'error',
      body: JSON.stringify({
        node: plainNode,
        inputs: encodeWireValue(inputs),
      }),
      signal,
    });

    if (!resp.ok) {
      let msg = `Host GPU server returned HTTP ${resp.status}`;
      try {
        const text = await readBoundedText(resp, MAX_HEALTH_RESPONSE_BYTES);
        const errJson = JSON.parse(text) as { error?: string };
        if (errJson?.error) msg = errJson.error;
      } catch {
        /* ignore */
      }
      throw new Error(msg);
    }

    const ct = (resp.headers.get('content-type') || '').toLowerCase();
    if (!ct.includes('text/event-stream')) {
      throw new Error(`Unexpected response Content-Type from host GPU server ("${ct || 'missing'}").`);
    }

    if (!resp.body) {
      throw new Error('Host GPU server returned an empty response stream.');
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalResult: { outputs: Record<string, CanonicalArtifact>; elapsedMs: number } | null = null;
    let finalError: string | null = null;

    const processEventBlock = (block: string) => {
      let eventType = 'message';
      const dataLines: string[] = [];
      for (const rawLine of block.split(/\r?\n/)) {
        if (!rawLine || rawLine.startsWith(':')) continue;
        if (rawLine.startsWith('event:')) {
          eventType = rawLine.slice(6).trim();
        } else if (rawLine.startsWith('data:')) {
          dataLines.push(rawLine.slice(5).trimStart());
        }
      }
      if (dataLines.length === 0) return;
      let payload: Record<string, unknown>;
      try {
        payload = JSON.parse(dataLines.join('\n')) as Record<string, unknown>;
      } catch {
        return;
      }

      if (eventType === 'started') {
        if (typeof payload.jobId === 'string') serverJobId = payload.jobId;
        if (typeof payload.cancelToken === 'string') serverCancelToken = payload.cancelToken;
        if (signal?.aborted) cancelRemote();
      } else if (eventType === 'progress') {
        const detail = typeof payload.detail === 'string' ? payload.detail : '';
        onProgress({
          nodeId: node.id,
          status:
            payload.status === 'loading' || payload.status === 'running'
              ? payload.status
              : 'running',
          phase: typeof payload.phase === 'string' ? payload.phase : 'running',
          progress: typeof payload.progress === 'number' ? payload.progress : undefined,
          detail: detail ? `[Host GPU] ${detail}` : '[Host GPU] Running...',
        });
      } else if (eventType === 'result') {
        const decodedOutputs = decodeWireValue<Record<string, CanonicalArtifact>>(payload.outputs);
        finalResult = {
          outputs: decodedOutputs,
          elapsedMs: typeof payload.elapsedMs === 'number' ? payload.elapsedMs : 0,
        };
      } else if (eventType === 'error') {
        finalError = typeof payload.message === 'string' ? payload.message : 'Host GPU node execution failed';
      }
    };

    try {
      while (!finalResult && !finalError) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        if (buffer.length > MAX_SSE_BUFFER_CHARS) {
          throw new Error('Host GPU stream exceeded maximum allowed buffer size.');
        }
        let sepIdx = buffer.indexOf('\n\n');
        while (sepIdx !== -1 && !finalResult && !finalError) {
          const block = buffer.slice(0, sepIdx);
          buffer = buffer.slice(sepIdx + 2);
          processEventBlock(block);
          sepIdx = buffer.indexOf('\n\n');
        }
      }
      if (!finalResult && !finalError && buffer.trim()) {
        processEventBlock(buffer);
      }
    } finally {
      // M4: Immediately cancel the underlying stream once terminal event or error arrives
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }

    if (finalError) {
      throw new Error(finalError);
    }
    if (!finalResult) {
      throw new Error('Host GPU server closed the stream before returning a result.');
    }
    return finalResult;
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
}
