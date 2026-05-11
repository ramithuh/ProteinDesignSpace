/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/webgpu';
import { sampleCenteredNoise } from '$lib/protein-ops/coords';
import { makeSchedule, type ScheduleKind } from '$lib/protein-ops/schedules';
import { sampleEulerFMEndpoint } from '$lib/samplers/euler-fm-endpoint';
import { sampleSDEProteinaEndpoint } from '$lib/samplers/sde-proteina-endpoint';
import type { SamplerKind, SDEParams, WorkerRequest, WorkerResponse } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

// ORT-Web fetches its WASM artifacts at runtime. Pin to CDN matching the
// installed npm version. Single-threaded so we don't need COOP+COEP isolation.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.wasm.numThreads = 1;

// Frozen constants from legacy-model/data — see tools/exports/legacy-teacher/sample_onnx.py.
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
  const tStart = performance.now();
  const bytes = await fetchWithProgress(modelUrl, (loaded, total) =>
    post({ id, type: 'progress', loaded, total }),
  );
  const tFetched = performance.now();

  let providers: ('webgpu' | 'wasm')[] = ['wasm'];
  if ('gpu' in navigator) {
    try {
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
  void tStart;

  post({
    id,
    type: 'loaded',
    provider: activeProvider,
    inputNames: [...session.inputNames],
    outputNames: [...session.outputNames],
    modelSizeBytes: bytes.byteLength,
    initLatencyMs: tInit - tFetched,
  });
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

/** Reshape a [batch, o*3] Float32Array trajectory frame into per-sample [o*3] frames. */
function unbatchFrame(frame: Float32Array, batch: number, oTimes3Times3: number): Float32Array[] {
  const per = oTimes3Times3;
  const out: Float32Array[] = [];
  for (let b = 0; b < batch; b++) {
    out.push(frame.slice(b * per, (b + 1) * per));
  }
  return out;
}

async function runSample(
  id: number,
  length: number,
  nSamples: number,
  nfe: number,
  scheduleKind: ScheduleKind,
  schedulP: number,
  seed: number,
  sampler: SamplerKind,
  sde: SDEParams | undefined,
) {
  if (!session) throw new Error('model not loaded');

  const batch = nSamples;
  const o = length * 3;
  const perSampleLen = o * 3;

  const staticFeats = buildStaticFeatures(batch, length);
  const expected = new Set(session.inputNames);

  const forward = async (x: Float32Array, tArr: Float32Array): Promise<Float32Array> => {
    const feeds: Record<string, ort.Tensor> = {
      backbone_coords: new ort.Tensor('float32', x, [batch, o, 3]),
      t: new ort.Tensor('float32', tArr, [batch]),
      ...staticFeats,
    };
    for (const k of Object.keys(feeds)) {
      if (!expected.has(k)) delete feeds[k];
    }
    const out = await session!.run(feeds);
    const x1Hat = out.x1_hat as ort.Tensor;
    return new Float32Array(x1Hat.data as Float32Array);
  };

  const xInit = sampleCenteredNoise(batch, o, seed);
  const schedule = makeSchedule(nfe, scheduleKind, schedulP);

  const tStart = performance.now();
  let lastStepEnd = tStart;

  const trajectories: Float32Array[][] = Array.from({ length: nSamples }, () => []);
  const onFrame = (step: number, t: number, x: Float32Array) => {
    const perSample = unbatchFrame(x, batch, perSampleLen);
    for (let b = 0; b < nSamples; b++) trajectories[b].push(perSample[b]);
    const now = performance.now();
    post({ id, type: 'sampleStep', step, t, latencyMs: now - lastStepEnd });
    lastStepEnd = now;
  };

  let result: { final: Float32Array; trajectory: Float32Array[] };
  if (sampler === 'euler-fm') {
    result = await sampleEulerFMEndpoint(forward, xInit, batch, o, {
      numSteps: nfe,
      schedule,
      onFrame,
    });
  } else if (sampler === 'sde-proteina') {
    if (!sde) throw new Error('sde params required for sampler=sde-proteina');
    result = await sampleSDEProteinaEndpoint(forward, xInit, batch, o, {
      numSteps: nfe,
      schedule,
      gamma: sde.gamma,
      gtMode: sde.gtMode,
      gtEps: sde.gtEps,
      gtMax: sde.gtMax,
      tLimOde: sde.tLimOde,
      seed: sde.noiseSeed,
      onFrame,
    });
  } else {
    const _exhaustive: never = sampler;
    throw new Error(`unknown sampler: ${_exhaustive}`);
  }

  const finalSamples = unbatchFrame(result.final, batch, perSampleLen);
  post({
    id,
    type: 'sampleDone',
    finalSamples,
    trajectories,
    length,
    totalLatencyMs: performance.now() - tStart,
    sampler,
  });
}

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  try {
    switch (req.type) {
      case 'load':
        await load(req.id, req.modelUrl);
        break;
      case 'sample':
        await runSample(
          req.id,
          req.length,
          req.nSamples,
          req.nfe,
          req.schedule,
          req.schedulP,
          req.seed,
          req.sampler,
          req.sde,
        );
        break;
      default: {
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
