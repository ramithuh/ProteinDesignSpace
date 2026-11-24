// Compatibility facade re-exporting Layer 1 Core Kernel WebGPU utilities.
export {
  estimateSafeBrowserMemoryBudgetBytes,
  extractAdapterInfo,
  extractAdapterLimits,
  formatWebGpuAdapterSummary,
  isSoftwareWebGpuAdapter,
  isWebGpuDeviceLostError,
  probeBrowserHardwareCapabilities,
  requestResilientWebGpuAdapter,
  requestResilientWebGpuDevice,
  type ResilientAdapterResult,
  type ResilientDeviceResult,
  type WebGpuAdapterInfoSnapshot,
  type WebGpuAdapterLimitsSnapshot,
  type WebGpuHardwareCapabilities,
} from '$lib/kernel/webgpu-device';
