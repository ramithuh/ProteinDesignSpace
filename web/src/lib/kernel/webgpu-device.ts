// Layer 1 Core Kernel — Resilient WebGPU Adapter/Device Acquisition & Safari Memory Budget Helper.
// Consolidates Linux Dawn/Vulkan powerPreference retries, experimental subgroup exclusion fallback,
// software-adapter detection, device.lost matching, and Safari macOS desktop deviceMemory polyfill.

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

const EXPERIMENTAL_SUBGROUP_FEATURES = new Set<string>([
  'chromium-experimental-subgroup-matrix',
  'subgroup-size-control',
]);

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

/**
 * Safari omits `navigator.deviceMemory`, which would otherwise clamp macOS Apple Silicon desktops
 * to a 4 GiB mobile assumption (~1.33 GiB budget). Returns a safe desktop macOS memory budget
 * in bytes when `deviceMemory` is absent on non-mobile macOS, or `null` otherwise.
 */
export function estimateSafeBrowserMemoryBudgetBytes(
  navOverride?: { deviceMemory?: number; userAgent?: string; hardwareConcurrency?: number } | null,
): number | null {
  const nav =
    navOverride !== undefined
      ? navOverride
      : typeof navigator === 'object' && navigator !== null
        ? (navigator as Navigator & { deviceMemory?: number })
        : null;
  if (
    nav &&
    typeof nav.deviceMemory === 'number' &&
    Number.isFinite(nav.deviceMemory) &&
    nav.deviceMemory > 0
  ) {
    return null;
  }
  const ua = nav?.userAgent ?? '';
  const isMobile = /iPhone|iPad|iPod|Android|Mobile/i.test(ua);
  const isMacDesktop = !isMobile && /Macintosh|Mac OS X/i.test(ua);
  if (isMacDesktop) {
    const cores = typeof nav?.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : 8;
    const estGiB = cores >= 10 ? 16 : 8;
    return Math.round((estGiB * 1024 * 1024 * 1024) / 3);
  }
  return null;
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
  optionalFeatures?: readonly string[];
}): Promise<ResilientDeviceResult> {
  const res = await requestResilientWebGpuAdapter(opts);
  const wantF16 = (opts?.useF16 ?? true) && res.hasF16;
  const requestedFeatures: GPUFeatureName[] = [];
  if (wantF16) requestedFeatures.push('shader-f16');
  for (const feat of opts?.optionalFeatures ?? []) {
    if (res.adapter.features.has(feat) && !requestedFeatures.includes(feat as GPUFeatureName)) {
      requestedFeatures.push(feat as GPUFeatureName);
    }
  }

  const requiredLimits = {
    maxBufferSize: res.adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: res.adapter.limits.maxStorageBufferBindingSize,
  };

  let device: GPUDevice;
  try {
    device = await res.adapter.requestDevice({
      requiredFeatures: requestedFeatures,
      requiredLimits,
    });
  } catch (firstErr) {
    // Linux Dawn/Vulkan fallback: retry excluding experimental subgroup features if requested
    const safeFeatures = requestedFeatures.filter(
      (f) => !EXPERIMENTAL_SUBGROUP_FEATURES.has(f),
    );
    if (safeFeatures.length === requestedFeatures.length) {
      throw firstErr;
    }
    device = await res.adapter.requestDevice({
      requiredFeatures: safeFeatures,
      requiredLimits,
    });
  }
  return { ...res, device };
}
