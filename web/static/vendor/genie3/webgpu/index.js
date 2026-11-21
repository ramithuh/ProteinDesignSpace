/**
 * Public browser & Web Worker entry point for Genie 3 WebGPU inference.
 *
 * Mirrors the ergonomic pattern of `rfd4-proteina-webgpu/webgpu/index.js` so web apps
 * (e.g. `denovo.run` workers) can import everything from a single entry point:
 *
 * ```js
 * import {
 *   requestWebGpuDevice,
 *   fetchGenie3Weights,
 *   Genie3GeneratorGpu,
 * } from 'genie3-webgpu';
 *
 * const { device } = await requestWebGpuDevice();
 * const weights = await fetchGenie3Weights('/weights/genie3/fp16', { cache: true });
 * const gen = await Genie3GeneratorGpu.create(device, weights, weights.config, { half: true });
 * const result = await gen.generate({ length: 100, nSampleStep: 100, predictSequence: true });
 * // result.pdb, result.cif, result.sequence, result.ca, result.backbone4
 * gen.destroy();
 * ```
 */

export * from '../shared/genie3/index.js';

export {
  PORTABLE_CEILINGS,
  SPEC_FLOOR_CEILINGS,
  getDeviceCapabilities,
  probeF16BitcastSupport,
  requestWebGpuDevice,
} from './genie3/device.js';

export {
  COPY_DST,
  COPY_SRC,
  Genie3Ops,
  LAYER_NORM_INPLACE_SOURCE,
  LAYER_NORM_SOURCE,
  MAP_READ,
  MAX_X,
  SOFTMAX_ROWS_SOURCE,
  STORAGE,
  TRANSPOSE_LN_SOURCE,
  UNIFORM,
  elementwiseSource,
  float32ToFloat16Bits,
  fold2,
  gemmSource,
  packFloat32ToHalfWords,
} from './genie3/ops.js';

export {
  BACKBONE_UPDATE_COMPOSE_SOURCE,
  FULL_IPA_LOGITS_SOURCE,
  IPA_PAIR_OUT_SOURCE,
  IPA_POINT_APPLY_SOURCE,
  IPA_POINT_OUT_SOURCE,
  IPA_SCALAR_OUT_SOURCE,
  REDUCED_IPA_LOGITS_SOURCE,
  TRI_GATE_SCATTER_SOURCE,
  recordBackboneUpdate,
  recordFullIpa,
  recordReducedIpa,
} from './genie3/kernels_ipa.js';

export {
  DEFAULT_PAIR_CHUNK_BYTES,
  Genie3DenoiserGpu,
  PAIR_INIT_EXT_SOURCE,
  buildSingleInputFeatures,
  estimateVramBudget,
  expectedWeightKeys,
} from './genie3/denoiser.js';

export {
  Genie3GeneratorGpu,
  generateGenie3,
  yieldToBrowser,
} from './genie3/generate.js';
