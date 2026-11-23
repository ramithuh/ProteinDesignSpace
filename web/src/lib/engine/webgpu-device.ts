// Compatibility facade re-exporting Layer 1 Core Kernel WebGPU utilities.
export {
  estimateSafeBrowserMemoryBudgetBytes,
  extractAdapterInfo,
  formatWebGpuAdapterSummary,
  isSoftwareWebGpuAdapter,
  isWebGpuDeviceLostError,
  requestResilientWebGpuAdapter,
  requestResilientWebGpuDevice,
  type ResilientAdapterResult,
  type ResilientDeviceResult,
  type WebGpuAdapterInfoSnapshot,
} from '$lib/kernel/webgpu-device';
