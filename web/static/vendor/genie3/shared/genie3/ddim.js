/**
 * Diffusion schedule, deterministic RNG, and DDIM sampling math for Genie 3.
 *
 * Mirrors:
 * - `genie3/generation/diffusion/schedule.py` (`cosine_beta_schedule`, `get_betas`)
 * - `genie3/generation/diffusion/sampler/sampler.py` (`Sampler.sample`)
 * - `genie3/generation/diffusion/sampler/ddim.py` (`DDIMSampler._setup_schedule`,
 *   `DDIMSampler._step`, `DDIMSampler._sample_sequence`)
 */

import { NUM_RESTYPES } from './constants.js';
import { cloneBatchForSequenceDecoding } from './features.js';

/**
 * Deterministic seeded PRNG (Mulberry32 + Box-Muller Gaussian) with zero dependencies.
 *
 * @param {number} [seed=42]
 * @returns {{ next: () => number, randn: () => number, randnArray: (n: number) => Float32Array }}
 */
export function createRng(seed = 42) {
  let state = (seed >>> 0) || 0x9e3779b9;
  let hasSpare = false;
  let spare = 0;

  function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function randn() {
    if (hasSpare) {
      hasSpare = false;
      return spare;
    }
    let u = 0;
    let v = 0;
    while (u <= 1e-12) u = next();
    v = next();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    const angle = 2.0 * Math.PI * v;
    spare = mag * Math.sin(angle);
    hasSpare = true;
    return mag * Math.cos(angle);
  }

  function randnArray(n) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = randn();
    return out;
  }

  return { next, randn, randnArray };
}

/**
 * Cosine variance schedule matching `cosine_beta_schedule(n_timestep)` in `schedule.py`.
 *
 * Returns a `Float32Array` of length `nTimestep + 1` where index `0` is `0.0` (clean)
 * and index `s` (`1..nTimestep`) is `beta_s`.
 *
 * @param {number} [nTimestep=1000]
 * @returns {Float32Array}
 */
export function cosineBetaSchedule(nTimestep = 1000) {
  const steps = nTimestep + 1;
  // Match PyTorch default float32 precision in `cosine_beta_schedule` (`torch.linspace` -> `cos**2`)
  const alphasCumprod = new Float32Array(steps);
  const halfPi = Math.fround(Math.PI * 0.5);
  for (let i = 0; i < steps; i += 1) {
    const x = Math.fround(i);
    const angle = Math.fround(Math.fround(x / steps) * halfPi);
    const c = Math.fround(Math.cos(angle));
    alphasCumprod[i] = Math.fround(c * c);
  }
  const a0 = alphasCumprod[0];
  for (let i = 0; i < steps; i += 1) {
    alphasCumprod[i] = Math.fround(alphasCumprod[i] / a0);
  }

  const betas = new Float32Array(steps);
  betas[0] = 0.0;
  for (let i = 1; i < steps; i += 1) {
    const b = Math.fround(1.0 - Math.fround(alphasCumprod[i] / alphasCumprod[i - 1]));
    betas[i] = Math.min(Math.max(b, 0.0), 0.999);
  }
  return betas;
}

/**
 * Precompute DDIM/DDPM diffusion schedule buffers matching `DDIMSampler._setup_schedule`.
 *
 * @param {number} [nTimestep=1000]
 * @param {string} [schedule='cosine']
 */
export function createDdimSchedule(nTimestep = 1000, schedule = 'cosine') {
  if (schedule !== 'cosine') {
    throw new Error(`Unsupported diffusion schedule "${schedule}" (expected "cosine")`);
  }
  const betas = cosineBetaSchedule(nTimestep);
  const len = betas.length;
  const alphas = new Float32Array(len);
  const alphasCumprod = new Float32Array(len);
  const oneMinusAlphasCumprod = new Float32Array(len);
  const sqrtAlphasCumprod = new Float32Array(len);
  const sqrtOneMinusAlphasCumprod = new Float32Array(len);

  // Match PyTorch FP32 cumprod precision closely
  let prod = 1.0;
  for (let i = 0; i < len; i += 1) {
    const a = 1.0 - betas[i];
    alphas[i] = a;
    prod = Math.fround(prod * a);
    alphasCumprod[i] = prod;
    const om = Math.fround(1.0 - prod);
    oneMinusAlphasCumprod[i] = om;
    sqrtAlphasCumprod[i] = Math.sqrt(prod);
    sqrtOneMinusAlphasCumprod[i] = Math.sqrt(Math.max(0.0, om));
  }

  return {
    nTimestep,
    schedule,
    betas,
    alphas,
    alphasCumprod,
    oneMinusAlphasCumprod,
    sqrtAlphasCumprod,
    sqrtOneMinusAlphasCumprod,
  };
}

/**
 * Compute the descending DDIM sampling timestep sequence matching `Sampler.sample`:
 *   `step_size = Math.floor(nTimestep / nSampleStep)`
 *   `steps = reversed(arange(1, nTimestep + 1, step_size))`
 *
 * Example (`nTimestep=1000, nSampleStep=100`): `stepSize = 10`, `steps = [991, 981, ..., 11, 1]`.
 *
 * @param {number} [nTimestep=1000]
 * @param {number} [nSampleStep=100]
 * @returns {{ steps: Int32Array, stepSize: number }}
 */
export function getSamplingSteps(nTimestep = 1000, nSampleStep = 100) {
  const stepSize = Math.floor(nTimestep / nSampleStep);
  if (stepSize < 1) {
    throw new Error(`Invalid nSampleStep=${nSampleStep} for nTimestep=${nTimestep}`);
  }
  const asc = [];
  for (let s = 1; s <= nTimestep; s += stepSize) asc.push(s);
  asc.reverse();
  return {
    steps: Int32Array.from(asc),
    stepSize,
  };
}

/**
 * Execute the pure DDIM update formula for a single reverse step `s -> s - stepSize`.
 *
 * Mirrors `DDIMSampler._step` in `genie3/generation/diffusion/sampler/ddim.py`:
 *   `zs_out = xs - xl_pred`
 *   `xs_out = ((xs - sqrt(1 - alpha_bar[s]) * zs_out) / sqrt(alpha_bar[s])) * gt_atom_mask`
 *   If `s - stepSize <= 0`: `xt = xs_out`
 *   Else (`t = s - stepSize`):
 *     `sigma = sqrt((one_minus_abar[t] / one_minus_abar[s]) * (1 - abar[s] / abar[t])) * eta`
 *     `direction = sqrt(1 - abar[t] - sigma^2) * zs_out`
 *     `xt = sqrt_abar[t] * xs_out + direction_scale * direction + noise_scale * sigma * z`
 *
 * @param {{
 *   xs: Float32Array,
 *   xlPred: Float32Array,
 *   s: number,
 *   stepSize: number,
 *   schedule: ReturnType<typeof createDdimSchedule>,
 *   gtAtomMask: ArrayLike<number>,
 *   directionScale?: number,
 *   noiseScale?: number,
 *   eta?: number,
 *   noise?: Float32Array | null,
 *   rng?: { randnArray: (n: number) => Float32Array } | null,
 * }} params
 * @returns {{ xt: Float32Array, xsOut: Float32Array, zsOut: Float32Array, sigma: number }}
 */
export function ddimStepUpdate({
  xs,
  xlPred,
  s,
  stepSize,
  schedule,
  gtAtomMask,
  directionScale = 0.8,
  noiseScale = 1.0,
  eta = 1.0,
  noise = null,
  rng = null,
}) {
  const n = xs.length;
  const L = gtAtomMask.length;
  const zsOut = new Float32Array(n);
  const xsOut = new Float32Array(n);

  const sqrtAbarS = schedule.sqrtAlphasCumprod[s];
  const sqrtOneMinusAbarS = schedule.sqrtOneMinusAlphasCumprod[s];

  for (let i = 0; i < L; i += 1) {
    const m = gtAtomMask[i];
    const base = i * 3;
    for (let d = 0; d < 3; d += 1) {
      const idx = base + d;
      const z = xs[idx] - xlPred[idx];
      zsOut[idx] = z;
      xsOut[idx] = ((xs[idx] - sqrtOneMinusAbarS * z) / sqrtAbarS) * m;
    }
  }

  if (s - stepSize <= 0) {
    return {
      xt: new Float32Array(xsOut),
      xsOut,
      zsOut,
      sigma: 0.0,
    };
  }

  const t = s - stepSize;
  const abarS = schedule.alphasCumprod[s];
  const abarT = schedule.alphasCumprod[t];
  const omS = schedule.oneMinusAlphasCumprod[s];
  const omT = schedule.oneMinusAlphasCumprod[t];

  const sigma = Math.sqrt((omT / omS) * (1.0 - abarS / abarT)) * eta;
  const dirCoef = Math.sqrt(Math.max(0.0, 1.0 - abarT - sigma * sigma)) * directionScale;
  const xCoef = schedule.sqrtAlphasCumprod[t];
  const nCoef = noiseScale * sigma;

  const zDraw = noise ?? (nCoef !== 0 ? (rng ?? createRng()).randnArray(n) : null);
  const xt = new Float32Array(n);
  for (let idx = 0; idx < n; idx += 1) {
    const noiseTerm = zDraw ? nCoef * zDraw[idx] : 0.0;
    xt[idx] = xCoef * xsOut[idx] + dirCoef * zsOut[idx] + noiseTerm;
  }

  return { xt, xsOut, zsOut, sigma };
}

/**
 * One-shot sequence decoding (`sequence_decoding_order == "all"`) matching
 * `DDIMSampler._sample_sequence` in `ddim.py`:
 *
 *   1. `t = 1`
 *   2. `xl_noisy = (sqrt_abar[1] * xl + sqrt_one_minus_abar[1] * (z * gt_atom_mask)) * gt_atom_mask`
 *   3. `out = denoiseStep(batch, xl_noisy, 1 / nTimestep)`
 *   4. `prob = softmax(out.ai / temperature, dim=-1)`
 *   5. `restype = one_hot(argmax(prob)) * (1 - cond_seq_mask) + gt_restype * cond_seq_mask` (masked by `gt_atom_mask`)
 *
 * @param {{
 *   denoiseStep: (batch: object, xl: Float32Array, tNorm: number) => Promise<{ ai: Float32Array, xl: Float32Array }> | { ai: Float32Array, xl: Float32Array },
 *   batch: object,
 *   xl: Float32Array,
 *   schedule: ReturnType<typeof createDdimSchedule>,
 *   temperature?: number,
 *   noise?: Float32Array | null,
 *   rng?: { randnArray: (n: number) => Float32Array } | null,
 * }} params
 * @returns {Promise<{ restypeOneHot: Float32Array, logits: Float32Array, probs: Float32Array, indices: Int32Array }>}
 */
export async function sampleSequenceOneShot({
  denoiseStep,
  batch,
  xl,
  schedule,
  temperature = 0.1,
  noise = null,
  rng = null,
}) {
  const L = batch.token_mask.length;
  const t = 1;
  const z = noise ?? (rng ?? createRng()).randnArray(L * 3);
  const sqrtAbar1 = schedule.sqrtAlphasCumprod[t];
  const sqrtOmAbar1 = schedule.sqrtOneMinusAlphasCumprod[t];

  const xlNoisy = new Float32Array(L * 3);
  for (let i = 0; i < L; i += 1) {
    const m = batch.gt_atom_mask[i];
    const base = i * 3;
    for (let d = 0; d < 3; d += 1) {
      const idx = base + d;
      const zl = z[idx] * m;
      xlNoisy[idx] = (sqrtAbar1 * xl[idx] + sqrtOmAbar1 * zl) * m;
    }
  }

  const seqBatch = cloneBatchForSequenceDecoding(batch);
  const out = await denoiseStep(seqBatch, xlNoisy, t / schedule.nTimestep);
  const logits = out.ai;

  const probs = new Float32Array(L * NUM_RESTYPES);
  const indices = new Int32Array(L);
  const restypeOneHot = new Float32Array(L * NUM_RESTYPES);

  for (let i = 0; i < L; i += 1) {
    const base = i * NUM_RESTYPES;
    let maxVal = -Infinity;
    for (let c = 0; c < NUM_RESTYPES; c += 1) {
      const v = logits[base + c] / temperature;
      if (v > maxVal) maxVal = v;
    }
    let sum = 0;
    for (let c = 0; c < NUM_RESTYPES; c += 1) {
      const e = Math.exp(logits[base + c] / temperature - maxVal);
      probs[base + c] = e;
      sum += e;
    }
    let bestIdx = 0;
    let bestProb = -1;
    for (let c = 0; c < NUM_RESTYPES; c += 1) {
      const p = probs[base + c] / sum;
      probs[base + c] = p;
      if (p > bestProb) {
        bestProb = p;
        bestIdx = c;
      }
    }
    indices[i] = bestIdx;

    const condMask = batch.cond_seq_mask[i];
    const atomMask = batch.gt_atom_mask[i];
    for (let c = 0; c < NUM_RESTYPES; c += 1) {
      const predOh = c === bestIdx ? 1.0 : 0.0;
      const gtOh = batch.gt_restype[base + c];
      restypeOneHot[base + c] = (predOh * (1.0 - condMask) + gtOh * condMask) * atomMask;
    }
  }

  return { restypeOneHot, logits, probs, indices };
}

/**
 * Run a complete DDIM sampling trajectory (reusable by both CPU and WebGPU engines).
 *
 * @param {{
 *   denoiseStep: (batch: object, xl: Float32Array, tNorm: number) => Promise<{ xl: Float32Array, ai?: Float32Array }> | { xl: Float32Array, ai?: Float32Array },
 *   batch: object,
 *   nTimestep?: number,
 *   nSampleStep?: number,
 *   directionScale?: number,
 *   noiseScale?: number,
 *   eta?: number,
 *   predictSequence?: boolean,
 *   sequenceTemperature?: number,
 *   seed?: number,
 *   rng?: { randnArray: (n: number) => Float32Array } | null,
 *   initNoise?: Float32Array | null,
 *   stepNoises?: Array<Float32Array | null> | null,
 *   seqNoise?: Float32Array | null,
 *   steps?: ArrayLike<number> | null,
 *   stepSize?: number | null,
 *   onStep?: ((info: { stepIndex: number, totalSteps: number, timestep: number, xl: Float32Array }) => void) | null,
 *   signal?: AbortSignal | null,
 * }} opts
 */
export async function runDdimSampling(opts) {
  const {
    denoiseStep,
    batch,
    nTimestep = 1000,
    nSampleStep = 100,
    directionScale = 0.8,
    noiseScale = 1.0,
    eta = 1.0,
    predictSequence = false,
    sequenceTemperature = 0.1,
    seed = 42,
    rng: providedRng = null,
    initNoise = null,
    stepNoises = null,
    seqNoise = null,
    steps: customSteps = null,
    stepSize: customStepSize = null,
    onStep = null,
    signal = null,
  } = opts;

  const L = batch.token_mask.length;
  const rng = providedRng ?? createRng(seed);
  const schedule = createDdimSchedule(nTimestep, 'cosine');
  const computed = getSamplingSteps(nTimestep, nSampleStep);
  const steps = customSteps ?? computed.steps;
  const stepSize = customStepSize ?? computed.stepSize;

  let xl = initNoise ? new Float32Array(initNoise) : rng.randnArray(L * 3);
  let lastAi = null;

  for (let idx = 0; idx < steps.length; idx += 1) {
    if (signal?.aborted) {
      throw new DOMException('Sampling aborted', 'AbortError');
    }
    const s = steps[idx];
    const tNorm = Math.fround(s / nTimestep);
    const out = await denoiseStep(batch, xl, tNorm);
    lastAi = out.ai ?? lastAi;

    const noiseForStep = stepNoises ? (stepNoises[idx] ?? null) : null;
    const stepRes = ddimStepUpdate({
      xs: xl,
      xlPred: out.xl,
      s,
      stepSize,
      schedule,
      gtAtomMask: batch.gt_atom_mask,
      directionScale,
      noiseScale,
      eta,
      noise: noiseForStep,
      rng,
    });
    xl = stepRes.xt;

    if (onStep) {
      onStep({
        stepIndex: idx,
        totalSteps: steps.length,
        timestep: s,
        xl,
      });
    }
  }

  let seqResult = null;
  if (predictSequence) {
    if (signal?.aborted) {
      throw new DOMException('Sampling aborted', 'AbortError');
    }
    seqResult = await sampleSequenceOneShot({
      denoiseStep,
      batch,
      xl,
      schedule,
      temperature: sequenceTemperature,
      noise: seqNoise,
      rng,
    });
  }

  return {
    xl,
    ai: seqResult ? seqResult.restypeOneHot : null,
    seqLogits: seqResult ? seqResult.logits : lastAi,
    seqProbs: seqResult ? seqResult.probs : null,
    seqIndices: seqResult ? seqResult.indices : null,
    schedule,
  };
}
