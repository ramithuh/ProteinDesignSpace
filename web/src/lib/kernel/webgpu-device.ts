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

export interface WebGpuAdapterLimitsSnapshot {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeWorkgroupStorageSize: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupSizeX: number;
}

export interface WebGpuHardwareCapabilities {
  isSoftwareFallback: boolean;
  limits: WebGpuAdapterLimitsSnapshot | null;
  deviceMemoryGiB: number | null;
  estimatedMemoryBudgetBytes: number | null;
}

export interface ResilientAdapterResult {
  adapter: GPUAdapter;
  info: WebGpuAdapterInfoSnapshot;
  isSoftwareFallback: boolean;
  hasF16: boolean;
  limits: WebGpuAdapterLimitsSnapshot | null;
  hardwareCapabilities: WebGpuHardwareCapabilities;
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

export function extractAdapterLimits(
  adapter: GPUAdapter | null | undefined,
): WebGpuAdapterLimitsSnapshot | null {
  const raw = (adapter as unknown as { limits?: Record<string, unknown> } | null | undefined)?.limits;
  if (!raw || typeof raw !== 'object') return null;
  const numOr = (value: unknown, fallback: number): number => {
    const n = Number(value);
    return Number.isFinite(n) && n >= 0 ? n : fallback;
  };
  const maxStorageBufferBindingSize = numOr(raw.maxStorageBufferBindingSize, 0);
  const maxBufferSize = numOr(raw.maxBufferSize, 0);
  if (maxStorageBufferBindingSize <= 0 && maxBufferSize <= 0) return null;
  return {
    maxStorageBufferBindingSize,
    maxBufferSize,
    maxComputeWorkgroupStorageSize: numOr(raw.maxComputeWorkgroupStorageSize, 0),
    maxComputeInvocationsPerWorkgroup: numOr(raw.maxComputeInvocationsPerWorkgroup, 0),
    maxComputeWorkgroupSizeX: numOr(raw.maxComputeWorkgroupSizeX, 0),
  };
}

export function probeBrowserHardwareCapabilities(
  res:
    | (Pick<ResilientAdapterResult, 'isSoftwareFallback'> & {
        limits?: WebGpuAdapterLimitsSnapshot | null;
        adapter?: GPUAdapter | null;
      })
    | null
    | undefined,
  navOverride?: { deviceMemory?: number; userAgent?: string; hardwareConcurrency?: number } | null,
): WebGpuHardwareCapabilities {
  const nav =
    navOverride !== undefined
      ? navOverride
      : typeof navigator === 'object' && navigator !== null
        ? (navigator as Navigator & { deviceMemory?: number })
        : null;
  const rawDeviceMem = nav?.deviceMemory;
  const deviceMemoryGiB =
    typeof rawDeviceMem === 'number' && Number.isFinite(rawDeviceMem) && rawDeviceMem > 0
      ? rawDeviceMem
      : null;
  const estimatedMemoryBudgetBytes = estimateSafeBrowserMemoryBudgetBytes(navOverride);
  const limits = res?.limits ?? extractAdapterLimits(res?.adapter);
  return {
    isSoftwareFallback: Boolean(res?.isSoftwareFallback),
    limits,
    deviceMemoryGiB,
    estimatedMemoryBudgetBytes,
  };
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

  const limits = extractAdapterLimits(adapter);
  const hardwareCapabilities = probeBrowserHardwareCapabilities({
    isSoftwareFallback,
    limits,
    adapter,
  });

  return {
    adapter,
    info,
    isSoftwareFallback,
    hasF16: adapter.features.has('shader-f16'),
    limits,
    hardwareCapabilities,
  };
}

export async function requestResilientWebGpuDevice(opts?: {
  useF16?: boolean;
  retries?: number;
  retryDelayMs?: number;
  allowSoftwareFallback?: boolean;
  optionalFeatures?: readonly string[];
  /** Extra adapter limits to raise to the adapter's maximum (e.g. the set ONNX Runtime requests for its own device). */
  adapterLimits?: readonly string[];
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

  const requiredLimits: Record<string, number> = {
    maxBufferSize: res.adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: res.adapter.limits.maxStorageBufferBindingSize,
  };
  const adapterLimitTable = res.adapter.limits as unknown as Record<string, unknown>;
  for (const key of opts?.adapterLimits ?? []) {
    const value = adapterLimitTable[key];
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
      requiredLimits[key] = value;
    }
  }

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

/** A `createBuffer` the device refused for lack of memory (see `tryCreateBuffer`). */
export class WebGpuOutOfMemoryError extends Error {
  readonly size: number;
  readonly label: string;

  constructor(desc: GPUBufferDescriptor, cause: unknown) {
    const label = desc.label ? ` "${desc.label}"` : '';
    const detail = (cause as { message?: string } | null)?.message;
    super(`WebGPU out of memory allocating ${desc.size} bytes for buffer${label}${detail ? `: ${detail}` : ''}`);
    this.name = 'WebGpuOutOfMemoryError';
    this.size = desc.size;
    this.label = String(desc.label ?? '');
  }
}

/**
 * `device.createBuffer(desc)` inside an `out-of-memory` error scope (TVM `webgpu.ts` `tryCreateBuffer`).
 * WebGPU reports a refused allocation asynchronously and hands back an invalid buffer, so without the
 * scope the failure only surfaces at a later submit, or as a device loss. Resolves to the buffer, or
 * destroys the invalid buffer and rejects with `WebGpuOutOfMemoryError` so callers can shrink and retry.
 */
export async function tryCreateBuffer(
  device: GPUDevice,
  desc: GPUBufferDescriptor,
): Promise<GPUBuffer> {
  device.pushErrorScope('out-of-memory');
  let buffer: GPUBuffer;
  try {
    buffer = device.createBuffer(desc);
  } catch (err) {
    // Keep the error-scope stack balanced even when createBuffer throws synchronously.
    await device.popErrorScope().catch(() => null);
    throw err;
  }
  const oom = await device.popErrorScope();
  if (oom) {
    try {
      buffer.destroy();
    } catch {
      /* invalid buffer */
    }
    throw new WebGpuOutOfMemoryError(desc, oom);
  }
  return buffer;
}

export interface WebGpuDeviceLostGuard {
  /** False only while `destroy()` is tearing a device down (WebLLM `engine.ts` `deviceLostIsError`). */
  readonly deviceLostIsError: boolean;
  /** Run `onUnexpectedLoss` when `device` is lost for any reason other than this guard's own `destroy()`. */
  watch(device: GPUDevice, onUnexpectedLoss: (info: GPUDeviceLostInfo) => void): void;
  /** Destroy `device` intentionally: its `lost` promise then resolves without calling `onUnexpectedLoss`. */
  destroy(device: GPUDevice | null | undefined): void;
}

/**
 * Separates intentional `device.destroy()` from real device loss. `device.lost` resolves
 * asynchronously after `destroy()`, so a plain boolean flipped around the call would already
 * be reset by the time the handler runs; the guard also remembers each device it destroyed.
 */
export function createDeviceLostGuard(): WebGpuDeviceLostGuard {
  const destroyedIntentionally = new WeakSet<GPUDevice>();
  let deviceLostIsError = true;
  return {
    get deviceLostIsError() {
      return deviceLostIsError;
    },
    watch(device, onUnexpectedLoss) {
      void device.lost.then(
        (info) => {
          if (!deviceLostIsError || destroyedIntentionally.has(device)) return;
          onUnexpectedLoss(info);
        },
        () => {},
      );
    },
    destroy(device) {
      if (!device) return;
      destroyedIntentionally.add(device);
      deviceLostIsError = false;
      try {
        device.destroy();
      } catch {
        /* already lost */
      } finally {
        deviceLostIsError = true;
      }
    },
  };
}
