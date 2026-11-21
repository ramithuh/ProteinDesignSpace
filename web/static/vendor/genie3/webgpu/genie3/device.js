/**
 * WebGPU device bootstrap for Genie 3.
 *
 * Adapted from LocalFold's `webgpu/runtime/device.js` and `wgsl-compat.js`:
 * - Explicitly requests raised adapter limits (WebGPU defaults cap storage bindings at 128 MiB).
 * - Requests optional `shader-f16`, `subgroups`, and `timestamp-query` when available.
 * - Probes `bitcast<vec2<f16>>(u32)` compiler support so browsers whose WGSL compiler rejects f16
 *   bitcasts (e.g. older Firefox Naga) cleanly fall back to core WGSL `unpack2x16float`.
 */

const RAISED_LIMITS = [
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxStorageBuffersPerShaderStage',
  'maxComputeWorkgroupsPerDimension',
  'maxComputeWorkgroupStorageSize',
  'maxComputeWorkgroupSizeX',
  'maxComputeInvocationsPerWorkgroup',
];

export const SPEC_FLOOR_CEILINGS = Object.freeze({
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
});

export const PORTABLE_CEILINGS = Object.freeze({
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
});

const deviceCapabilities = new WeakMap();

const F16_BITCAST_PROBE = `
enable f16;
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let v = bitcast<vec2<f16>>(src[0]);
  dst[0] = f32(v.x) + f32(v.y);
}
`;

/**
 * Probe whether `device` actually compiles `enable f16; bitcast<vec2<f16>>(u32)`.
 * @param {GPUDevice} device
 * @returns {Promise<boolean>}
 */
export async function probeF16BitcastSupport(device) {
  if (!device?.features?.has?.('shader-f16')) return false;
  if (typeof device.createShaderModule !== 'function' || typeof device.createComputePipelineAsync !== 'function') {
    return true;
  }
  try {
    if (typeof device.pushErrorScope === 'function') device.pushErrorScope('validation');
    const module = device.createShaderModule({ label: 'genie3.probe.f16-bitcast', code: F16_BITCAST_PROBE });
    await device.createComputePipelineAsync({
      label: 'genie3.probe.f16-bitcast',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const err = typeof device.popErrorScope === 'function' ? await device.popErrorScope() : null;
    return err === null;
  } catch {
    try {
      if (typeof device.popErrorScope === 'function') await device.popErrorScope();
    } catch {
      // Ignore error scope mismatch if pushErrorScope threw.
    }
    return false;
  }
}

/**
 * Inspect recorded/detected capabilities for a GPUDevice.
 * @param {GPUDevice} device
 * @returns {{hasF16: boolean, f16Bitcast: boolean, hasSubgroups: boolean, hasTimestampQuery: boolean, limits: object, adapterInfo: object|null}}
 */
export function getDeviceCapabilities(device) {
  const cached = device ? deviceCapabilities.get(device) : null;
  if (cached) return cached;
  const hasF16 = Boolean(device?.features?.has?.('shader-f16'));
  return {
    hasF16,
    f16Bitcast: hasF16,
    hasSubgroups: Boolean(device?.features?.has?.('subgroups')),
    hasTimestampQuery: Boolean(device?.features?.has?.('timestamp-query')),
    limits: device?.limits ?? {},
    adapterInfo: device?.adapterInfo ?? null,
  };
}

/**
 * Request a WebGPU adapter & device with raised compute/storage limits and optional `shader-f16`.
 *
 * @param {{
 *   exclude?: string[],
 *   portableLimits?: boolean | 'spec',
 *   powerPreference?: 'high-performance' | 'low-power',
 *   forceFallbackAdapter?: boolean,
 * }} [opts]
 * @returns {Promise<{adapter: GPUAdapter, device: GPUDevice, capabilities: object} | null>}
 */
export async function requestWebGpuDevice({
  exclude = [],
  portableLimits = false,
  powerPreference = 'high-performance',
  forceFallbackAdapter = false,
} = {}) {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return null;
  }
  let adapter = await navigator.gpu.requestAdapter({ powerPreference, forceFallbackAdapter });
  if (!adapter && !forceFallbackAdapter) {
    adapter = await navigator.gpu.requestAdapter();
  }
  if (!adapter) return null;

  const withheld = new Set(exclude);
  const optional = ['shader-f16', 'subgroups', 'timestamp-query'];
  const requiredFeatures = optional.filter(
    (feature) => adapter.features?.has?.(feature) && !withheld.has(feature),
  );

  const ceiling = portableLimits === 'spec'
    ? SPEC_FLOOR_CEILINGS
    : portableLimits === true
      ? PORTABLE_CEILINGS
      : {};

  const requiredLimits = {};
  for (const name of RAISED_LIMITS) {
    const available = adapter.limits?.[name];
    if (typeof available === 'number' && Number.isFinite(available)) {
      requiredLimits[name] = ceiling[name] === undefined ? available : Math.min(available, ceiling[name]);
    }
  }

  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  if (!device.adapterInfo && adapter.info) {
    try {
      device.adapterInfo = adapter.info;
    } catch {
      // Some runtimes freeze GPUDevice properties.
    }
  }

  const hasF16 = Boolean(device.features?.has?.('shader-f16'));
  const f16Bitcast = hasF16 ? await probeF16BitcastSupport(device) : false;
  const capabilities = {
    hasF16,
    f16Bitcast,
    hasSubgroups: Boolean(device.features?.has?.('subgroups')),
    hasTimestampQuery: Boolean(device.features?.has?.('timestamp-query')),
    limits: device.limits ?? {},
    adapterInfo: adapter.info ?? device.adapterInfo ?? null,
  };
  deviceCapabilities.set(device, capabilities);

  return { adapter, device, capabilities };
}
