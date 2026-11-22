export interface WebGpuAdapterInfoSnapshot {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean;
}

export interface ResilientAdapterResult {
  adapter: GPUAdapter;
  info: WebGpuAdapterInfoSnapshot;
  isSoftwareFallback: boolean;
  hasF16: boolean;
}

export interface ResilientDeviceResult extends ResilientAdapterResult {
  device: GPUDevice;
}

const SOFTWARE_ADAPTER_RE =
  /swiftshader|llvmpipe|lavapipe|software|\bwarp\b|basic render|microsoft basic/i;

const DEVICE_LOST_ERROR_RE =
  /device\s*lost|mapAsync|valid external Instance|BufferManager|WebGPU.*(lost|error|failed)|OrtRun|createBuffer.*lost/i;

export function extractAdapterInfo(adapter: GPUAdapter): WebGpuAdapterInfoSnapshot {
  const raw = (
    adapter as unknown as {
      info?: {
        vendor?: string;
        architecture?: string;
        device?: string;
        description?: string;
        isFallbackAdapter?: boolean;
      };
      isFallbackAdapter?: boolean;
    }
  ).info;
  const fallbackFlag = Boolean(
    raw?.isFallbackAdapter ??
      (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter,
  );
  return {
    vendor: String(raw?.vendor ?? '').trim(),
    architecture: String(raw?.architecture ?? '').trim(),
    device: String(raw?.device ?? '').trim(),
    description: String(raw?.description ?? '').trim(),
    isFallbackAdapter: fallbackFlag,
  };
}

export function isSoftwareWebGpuAdapter(info: WebGpuAdapterInfoSnapshot): boolean {
  if (info.isFallbackAdapter) return true;
  const combined = [info.vendor, info.architecture, info.device, info.description].join(' ');
  return SOFTWARE_ADAPTER_RE.test(combined);
}

export function formatWebGpuAdapterSummary(
  info: WebGpuAdapterInfoSnapshot,
  hasF16: boolean,
): string {
  const tag = info.description || info.device || info.vendor || info.architecture || 'ready';
  const prec = hasF16 ? 'f16' : 'f32';
  return isSoftwareWebGpuAdapter(info) ? `${tag} (software, ${prec})` : `${tag} (${prec})`;
}

export function isWebGpuDeviceLostError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err ?? '');
  return DEVICE_LOST_ERROR_RE.test(msg);
}

export async function requestResilientWebGpuAdapter(opts?: {
  retries?: number;
  retryDelayMs?: number;
  allowSoftwareFallback?: boolean;
}): Promise<ResilientAdapterResult> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu) {
    throw new Error('WebGPU is unavailable in this browser or context.');
  }
  const retries = Math.max(1, opts?.retries ?? 3);
  const retryDelayMs = opts?.retryDelayMs ?? 150;
  const allowSoftwareFallback = opts?.allowSoftwareFallback ?? true;

  let adapter: GPUAdapter | null = null;
  for (let attempt = 0; attempt < retries && !adapter; attempt++) {
    if (attempt > 0) {
      await new Promise((resolve) => setTimeout(resolve, attempt * retryDelayMs));
    }
    adapter =
      (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
      (await navigator.gpu.requestAdapter());
  }

  if (!adapter) {
    throw new Error('No WebGPU adapter available.');
  }

  const info = extractAdapterInfo(adapter);
  const isSoftwareFallback = isSoftwareWebGpuAdapter(info);
  if (isSoftwareFallback && !allowSoftwareFallback) {
    throw new Error('Hardware WebGPU adapter required; software fallback rejected.');
  }

  return {
    adapter,
    info,
    isSoftwareFallback,
    hasF16: adapter.features.has('shader-f16'),
  };
}

export async function requestResilientWebGpuDevice(opts?: {
  useF16?: boolean;
  retries?: number;
  retryDelayMs?: number;
  allowSoftwareFallback?: boolean;
}): Promise<ResilientDeviceResult> {
  const res = await requestResilientWebGpuAdapter(opts);
  const wantF16 = (opts?.useF16 ?? true) && res.hasF16;
  const device = await res.adapter.requestDevice({
    requiredFeatures: wantF16 ? ['shader-f16'] : [],
    requiredLimits: {
      maxBufferSize: res.adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: res.adapter.limits.maxStorageBufferBindingSize,
    },
  });
  return { ...res, device };
}
