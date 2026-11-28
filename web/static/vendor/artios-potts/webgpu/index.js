// Zero-dependency WebGPU backend for the ArtiosPotts Potts-parameter head
// (artios/model/artios_potts.py). Pairs the MPNN featurizer + encoder/decoder
// with the factorised GraphPotts head and returns the fields h and couplings J
// a Potts sampler consumes.

export {
  createArtiosPottsGpu,
  GpuRuntime,
  prepareFeatures,
  pottsLinearWeights,
  rbfParameters,
  NUM_BACKBONE_ATOMS,
  NUM_NON_PROTEIN_RESTYPES,
  DEFAULT_PACING,
} from './engine.js';

export {
  DEFAULT_CONFIG,
  normalizeConfig,
  normalizePrecision,
  weightSpec,
  packLinearWeight,
  parseSafetensors,
  loadSafetensorsManifest,
  createWeightSource,
  resolveWeights,
  f32ToF16Bits,
  f16BitsToF32,
} from './weights.js';

export * as shaders from './shaders.js';

import { pottsLinearWeights } from './engine.js';
import { normalizeConfig, normalizePrecision, packLinearWeight, resolveWeights } from './weights.js';

/** True when this runtime exposes a WebGPU entry point. */
export function pottsGpuSupported() {
  return !!globalThis.navigator?.gpu;
}

/**
 * Resolve a weight source and replace every tiled-matmul weight with the exact
 * FP32 matrix the GPU multiplies by at `precision` (identity for 'f32', the
 * round-tripped halves for 'f16', the per-output-row dequantised values for
 * 'int8'). Lets a CPU reference reproduce a quantised GPU run bit-for-bit up
 * to accumulation order.
 *
 * @returns {Promise<Map<string, Float32Array>>} every resolved tensor, with the
 *          linear weights dequantised in place.
 */
export async function dequantizePottsWeights(weights, config = {}, precision = 'f32', fetchFn) {
  const cfg = normalizeConfig(config);
  const prec = normalizePrecision(precision);
  const tensors = await resolveWeights(weights, cfg, fetchFn);
  const out = new Map(tensors);
  if (prec === 'f32') return out;
  for (const spec of pottsLinearWeights(cfg)) {
    const w = tensors.get(spec.name);
    if (!w) {
      if (spec.optional) continue;
      throw Error(`ArtiosPotts WebGPU: missing tensor '${spec.name}'`);
    }
    out.set(spec.name, packLinearWeight(w, spec.rows, spec.cols, prec).dequantized);
  }
  return out;
}
