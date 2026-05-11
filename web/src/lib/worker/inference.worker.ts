/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/webgpu';
import type { WorkerRequest, WorkerResponse } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

// ORT-Web fetches its WASM artifacts at runtime. Pin to CDN matching the
// installed npm version. Single-threaded so we don't need COOP+COEP isolation.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.wasm.numThreads = 1;

// Frozen constants from legacy-model/data (see tools/exports/legacy-teacher/sample_onnx.py).
const BACKBONE_PATTERN = [18, 2, 1]; // N, CA, C atom IDs (RF2AA encoding)
const ALA_TOKEN = 1;

let session: ort.InferenceSession | null = null;
let activeProvider = 'unknown';

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

async function fetchWithProgress(
  url: string,
  onProgress: (loaded: number, total: number) => void,
): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  const total = Number(resp.headers.get('content-length') ?? 0);
  if (!resp.body) throw new Error('no response body (no streaming reader)');

  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

async function load(id: number, modelUrl: string) {
  const t0 = performance.now();
  const bytes = await fetchWithProgress(modelUrl, (loaded, total) =>
    post({ id, type: 'progress', loaded, total }),
  );
  const tFetched = performance.now();

  // Detect WebGPU support and pick provider list. Manual fallback only —
  // ORT-Web does NOT automatically retry on a different EP if the first fails.
  let providers: ('webgpu' | 'wasm')[] = ['wasm'];
  if ('gpu' in navigator) {
    try {
      // Lightweight probe: just ask for an adapter.
      // Cast through `any` — we don't depend on @webgpu/types just for this probe.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (adapter) providers = ['webgpu', 'wasm'];
    } catch (_) {
      /* fall through to wasm-only */
    }
  }

  try {
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    });
    activeProvider = providers[0];
  } catch (err) {
    if (providers[0] === 'webgpu') {
      // Retry without WebGPU
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      activeProvider = `wasm (webgpu failed: ${err instanceof Error ? err.message : String(err)})`;
    } else {
      throw err;
    }
  }
  const tInit = performance.now();

  post({
    id,
    type: 'loaded',
    provider: activeProvider,
    inputNames: [...session.inputNames],
    outputNames: [...session.outputNames],
    modelSizeBytes: bytes.byteLength,
    initLatencyMs: tInit - tFetched,
  });
  // Don't hold the model bytes after the session has consumed them.
  // (No-op here because `bytes` falls out of scope at function return,
  // but worth being deliberate as model sizes grow.)
}

// Mulberry32 — deterministic, cheap, good-enough for visual reproducibility.
function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rng: () => number) {
  const u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v);
}

function buildStaticFeatures(batch: number, length: number) {
  const o = length * 3;
  const atomIds = new BigInt64Array(batch * o);
  const aaSeq = new BigInt64Array(batch * o);
  const mask = new Uint8Array(batch * o);
  for (let b = 0; b < batch; b++) {
    for (let r = 0; r < length; r++) {
      for (let a = 0; a < 3; a++) {
        const idx = b * o + r * 3 + a;
        atomIds[idx] = BigInt(BACKBONE_PATTERN[a]);
        aaSeq[idx] = BigInt(ALA_TOKEN);
        mask[idx] = 1;
      }
    }
  }
  return {
    backbone_atom_ids: new ort.Tensor('int64', atomIds, [batch, o]),
    backbone_aa_seq: new ort.Tensor('int64', aaSeq, [batch, o]),
    backbone_mask: new ort.Tensor('bool', mask, [batch, o]),
  };
}

function zeroCenter(coords: Float32Array, o: number) {
  let cx = 0,
    cy = 0,
    cz = 0;
  for (let i = 0; i < o; i++) {
    cx += coords[i * 3 + 0];
    cy += coords[i * 3 + 1];
    cz += coords[i * 3 + 2];
  }
  cx /= o;
  cy /= o;
  cz /= o;
  for (let i = 0; i < o; i++) {
    coords[i * 3 + 0] -= cx;
    coords[i * 3 + 1] -= cy;
    coords[i * 3 + 2] -= cz;
  }
}

async function forward(id: number, length: number, seed: number, t: number) {
  if (!session) throw new Error('model not loaded');

  const batch = 1;
  const o = length * 3;
  const rng = mulberry32(seed);
  const coords = new Float32Array(batch * o * 3);
  for (let i = 0; i < coords.length; i++) coords[i] = randn(rng);
  zeroCenter(coords, o);

  const staticFeats = buildStaticFeatures(batch, length);
  const feeds: Record<string, ort.Tensor> = {
    backbone_coords: new ort.Tensor('float32', coords, [batch, o, 3]),
    t: new ort.Tensor('float32', new Float32Array([t]), [batch]),
    ...staticFeats,
  };
  // Filter to declared inputs (legacy exporter DCE'd backbone_aa_idx).
  const expected = new Set(session.inputNames);
  const filtered: Record<string, ort.Tensor> = {};
  for (const k of Object.keys(feeds)) if (expected.has(k)) filtered[k] = feeds[k];

  const t0 = performance.now();
  const out = await session.run(filtered);
  const latencyMs = performance.now() - t0;

  const x1Hat = out.x1_hat as ort.Tensor;
  const data = x1Hat.data as Float32Array;
  post({
    id,
    type: 'forwardDone',
    outputShape: x1Hat.dims,
    latencyMs,
    sample: Array.from(data.slice(0, 9)),
  });
}

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  try {
    switch (req.type) {
      case 'load':
        await load(req.id, req.modelUrl);
        break;
      case 'forward':
        await forward(req.id, req.length, req.seed, req.t);
        break;
      default: {
        // exhaustive
        const _exhaustive: never = req;
        void _exhaustive;
      }
    }
  } catch (err) {
    post({
      id: req.id,
      type: 'error',
      message: err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err),
    });
  }
};

export {};
