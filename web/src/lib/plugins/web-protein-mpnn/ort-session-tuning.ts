// ONNX Runtime Web 1.31.1 session tuning for web-protein-mpnn, checked against
// `onnxruntime-common@1.31.1` (`lib/inference-session.ts`) and `onnxruntime-web@1.31.1`
// (`lib/wasm/session-options.ts`, `lib/backend-wasm.ts`).

import type { InferenceSession } from 'onnxruntime-web';

/**
 * Native WebGPU EP options. `storageBufferCacheMode: 'simple'` reuses exact-size storage buffers
 * (decoderStep runs L times at one shape) instead of rounding to buckets; `validationMode: 'wgpuOnly'`
 * skips the EP's own checks on top of WebGPU's. The MPNN worker is exclusive-lease, so the exact-size
 * cache never outlives the run that filled it.
 */
export const MPNN_WEBGPU_EP_OPTIONS = {
  storageBufferCacheMode: 'simple',
  validationMode: 'wgpuOnly',
} as const satisfies Omit<InferenceSession.WebGpuExecutionProviderOption, 'name'>;

/**
 * Limits the WebGPU EP raises to the adapter maximum when it creates its own device; an injected
 * device has to offer the same, or kernels sized from device limits (e.g. Concat's per-pass input
 * count from `maxStorageBuffersPerShaderStage`) fall back to the WebGPU defaults.
 */
export const ORT_WEBGPU_ADAPTER_LIMITS = [
  'maxBindGroups',
  'maxComputeWorkgroupStorageSize',
  'maxComputeWorkgroupsPerDimension',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxStorageBuffersPerShaderStage',
] as const;

/**
 * WASM thread count: ORT's own default (`backend-wasm.ts` `initializeFlags`), made explicit because
 * proteinmpnn-web pins `env.wasm.numThreads = 1` whenever `numThreads` is not passed. Threads need
 * SharedArrayBuffer, so anything but a cross-origin-isolated page stays single-threaded.
 */
export function resolveOrtWasmThreads(
  crossOriginIsolated: boolean,
  hardwareConcurrency: number | undefined,
): number {
  if (!crossOriginIsolated) return 1;
  return Math.min(4, Math.ceil((hardwareConcurrency || 1) / 2));
}

type ExecutionProviderConfig = NonNullable<InferenceSession.SessionOptions['executionProviders']>[number];

/** Replace the bare `'webgpu'` EP name with a configured WebGPU EP option object. */
export function upgradeWebGpuExecutionProviders(
  executionProviders: readonly ExecutionProviderConfig[],
  epOptions: Omit<InferenceSession.WebGpuExecutionProviderOption, 'name'>,
): ExecutionProviderConfig[] {
  return executionProviders.map((ep) =>
    ep === 'webgpu' ? ({ ...epOptions, name: 'webgpu' } as InferenceSession.WebGpuExecutionProviderOption) : ep,
  );
}

function isSessionOptionsArg(value: unknown): value is InferenceSession.SessionOptions {
  return (
    typeof value === 'object' &&
    value !== null &&
    !ArrayBuffer.isView(value) &&
    !(value instanceof ArrayBuffer) &&
    Array.isArray((value as InferenceSession.SessionOptions).executionProviders)
  );
}

interface InferenceSessionFactoryLike {
  create: (...args: unknown[]) => Promise<unknown>;
}

/**
 * Run `fn` while `ort.InferenceSession.create` turns `'webgpu'` EP names into
 * `{ name: 'webgpu', ...epOptions }`. proteinmpnn-web's `OnnxBackend.create` always overwrites
 * `executionProviders` with bare names, and ORT 1.31.1's native WebGPU EP reads its device and
 * tuning only from that per-EP object (`env.webgpu.device` is written back by ORT, never read).
 * WASM-only sessions, including proteinmpnn-web's own WebGPU-to-WASM fallback, pass through untouched.
 */
export async function withWebGpuExecutionProviderOptions<T>(
  ort: { InferenceSession: unknown },
  epOptions: Omit<InferenceSession.WebGpuExecutionProviderOption, 'name'>,
  fn: () => Promise<T>,
): Promise<T> {
  const factory = ort.InferenceSession as InferenceSessionFactoryLike;
  const original = factory.create;
  factory.create = function (this: unknown, ...args: unknown[]) {
    const last = args.length - 1;
    const options = args[last];
    if (last >= 1 && isSessionOptionsArg(options)) {
      args[last] = {
        ...options,
        executionProviders: upgradeWebGpuExecutionProviders(options.executionProviders!, epOptions),
      };
    }
    return original.apply(this, args);
  };
  try {
    return await fn();
  } finally {
    factory.create = original;
  }
}
