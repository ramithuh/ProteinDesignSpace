// Zero-dependency WebGPU backend for the ArtiosBackbone diffusion module.

export {
  createArtiosBackboneGpu,
  LocalAttnTransformerGpu,
  GpuRuntime,
  kernelOps,
  prepareLinear,
  weightGetter,
  normalizeConfig,
  atomTokenTables,
  edmScalars,
  SIGMA_DATA,
  N_REF_FEATURES,
} from './engine.js';
export * as shaders from './shaders.js';
export { linearFormat, dequantizeLinear, f32ToF16Bits, f16BitsToF32 } from './quant.js';

import { SIGMA_DATA } from './engine.js';

/** EDM sampler constants of EDMSampler.sample (generator.py). */
export const EDM_SAMPLER_DEFAULTS = Object.freeze({
  gamma0: 0.8,
  gammaMin: 1.0,
  noiseScaleLambda: 1.003,
  stepScaleEta: 1.5,
});

/** EDMInferenceNoiseScheduler.get_schedule: Karras rho = 7 schedule, last = 0. */
export function edmNoiseSchedule(nSteps, sigmaData = SIGMA_DATA) {
  const sMax = 160.0;
  const sMin = 4e-4;
  const rho = 7.0;
  const out = new Float32Array(nSteps + 1);
  for (let i = 0; i <= nSteps; i++) {
    const v = sMax ** (1 / rho) + (i / nSteps) * (sMin ** (1 / rho) - sMax ** (1 / rho));
    out[i] = sigmaData * v ** rho;
  }
  out[nSteps] = 0;
  return out;
}

/** Noise level t_hat and added-noise std for one step (c_tau_last -> c_tau). */
export function edmStepNoise(cTauLast, cTau, { gamma0, gammaMin, noiseScaleLambda } = EDM_SAMPLER_DEFAULTS) {
  const gamma = cTau > gammaMin ? gamma0 : 0;
  const tHat = cTauLast * (1 + gamma);
  return { tHat, noiseStd: noiseScaleLambda * Math.sqrt(tHat * tHat - cTauLast * cTauLast) };
}

/** x_next = x_noisy + eta * (c_tau - t_hat) * (x_noisy - x_denoised) / t_hat */
export function edmSamplerUpdate(xNoisy, xDenoised, tHat, cTau, stepScaleEta = EDM_SAMPLER_DEFAULTS.stepScaleEta) {
  const out = new Float32Array(xNoisy.length);
  const k = (stepScaleEta * (cTau - tHat)) / tHat;
  for (let i = 0; i < out.length; i++) out[i] = xNoisy[i] + k * (xNoisy[i] - xDenoised[i]);
  return out;
}
