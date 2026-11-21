/**
 * End-to-end Genie 3 DDIM protein design generator on WebGPU (`Genie3GeneratorGpu`).
 *
 * Reuses `shared/genie3/` directly for:
 * - Unconditional & spec-based feature construction (`features.js`)
 * - Cosine DDIM schedule, step updates, and one-shot sequence decoding (`ddim.js`)
 * - Ideal trans-peptide backbone (`N, CA, C, O`) reconstruction (`backbone.js`)
 * - Mol* / ProteinMPNN / LocalFold-compatible PDB & mmCIF export (`pdb.js`)
 */

import { centerCaCoords, reconstructBackboneFromCa } from '../../shared/genie3/backbone.js';
import { resolveGenie3Config } from '../../shared/genie3/constants.js';
import { runDdimSampling } from '../../shared/genie3/ddim.js';
import { createFeaturesFromLength } from '../../shared/genie3/features.js';
import {
  backboneToMmcif,
  backboneToPdb,
  sequenceFromRestype,
} from '../../shared/genie3/pdb.js';
import { Genie3DenoiserGpu } from './denoiser.js';

let yieldChannel;

/** Yield to the browser event loop via MessageChannel (avoids >=1s background-tab timer clamping). */
export function yieldToBrowser() {
  if (typeof process !== 'undefined' && process.versions?.node) {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }
  if (typeof MessageChannel === 'undefined') {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }
  if (yieldChannel === undefined) yieldChannel = new MessageChannel();
  return new Promise((resolve) => {
    yieldChannel.port1.onmessage = () => resolve();
    yieldChannel.port2.postMessage(undefined);
  });
}

export class Genie3GeneratorGpu {
  /**
   * @param {GPUDevice} device
   * @param {object} weights Normalized or raw weights (`{ tensors, config }` or `{ [key]: { data, shape } }`)
   * @param {object} [cfg] Optional model/diffusion/sampler config override
   * @param {{
   *   half?: boolean,
   *   pairChunkBytes?: number,
   *   canonicalQuat?: boolean,
   *   deferUpload?: boolean,
   * }} [opts]
   */
  constructor(device, weights, cfg = null, opts = {}) {
    this.device = device;
    this.opts = opts;
    this.half = opts.half ?? false;
    this.denoiser = new Genie3DenoiserGpu(device, weights, cfg, opts);
    this.cfg = this.denoiser.cfg;
  }

  /**
   * Async factory that uploads weights layer-by-layer, yielding between blocks so a browser page
   * or worker stays responsive during preload.
   */
  static async create(device, weights, cfg = null, opts = {}) {
    const gen = new Genie3GeneratorGpu(device, weights, cfg, { ...opts, deferUpload: true });
    gen.opts = opts;
    for (const _ of gen.denoiser.uploadSteps()) {
      await yieldToBrowser();
    }
    return gen;
  }

  /**
   * Run a single V1Denoiser forward pass on WebGPU.
   *
   * @param {object} batch Feature batch
   * @param {Float32Array} xl Noisy CA coordinates `[L * 3]`
   * @param {number} t Normalized diffusion timestep (`s / n_timestep`)
   * @param {object} [opts]
   * @returns {Promise<{ ai: Float32Array, xl: Float32Array, intermediates?: object }>}
   */
  async denoiseStep(batch, xl, t, opts = {}) {
    return this.denoiser.forward(batch, xl, t, opts);
  }

  /**
   * Generate a de novo protein structure (and optional one-shot sequence) using DDIM sampling on WebGPU.
   *
   * @param {{
   *   length?: number,
   *   batch?: object,
   *   nSampleStep?: number,
   *   nTimestep?: number,
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
   *   center?: boolean,
   *   caOnly?: boolean,
   *   chainId?: string,
   *   entryId?: string,
   *   yieldBetweenSteps?: boolean,
   *   onStep?: ((info: { stepIndex: number, totalSteps: number, timestep: number, xl: Float32Array }) => void | Promise<void>) | null,
   *   signal?: AbortSignal | null,
   * }} [opts]
   */
  async generate(opts = {}) {
    const startMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const cfg = resolveGenie3Config(this.cfg);
    const batch = opts.batch ?? createFeaturesFromLength(opts.length ?? 100);
    const L = batch.token_mask.length;

    const nTimestep = opts.nTimestep ?? cfg.diffusion?.n_timestep ?? cfg.n_timestep ?? 1000;
    const nSampleStep = opts.nSampleStep ?? cfg.sampler?.n_sample_step ?? 100;
    const directionScale = opts.directionScale ?? cfg.sampler?.direction_scale ?? 0.8;
    const noiseScale = opts.noiseScale ?? cfg.sampler?.noise_scale ?? 1.0;
    const eta = opts.eta ?? cfg.sampler?.eta ?? 1.0;
    const predictSequence = opts.predictSequence ?? cfg.sampler?.predict_sequence ?? true;
    const sequenceTemperature = opts.sequenceTemperature ?? cfg.sampler?.sequence_temperature ?? 0.1;
    const yieldBetweenSteps = opts.yieldBetweenSteps ?? Boolean(opts.onStep);

    let stepsRun = 0;
    const onStepWrapped = opts.onStep || yieldBetweenSteps
      ? async (info) => {
          stepsRun = info.stepIndex + 1;
          if (opts.onStep) await opts.onStep(info);
          if (yieldBetweenSteps) await yieldToBrowser();
        }
      : (info) => {
          stepsRun = info.stepIndex + 1;
        };

    // Note: runDdimSampling invokes onStep synchronously; handle async callbacks cleanly.
    let pendingOnStep = Promise.resolve();
    const ddimRes = await runDdimSampling({
      denoiseStep: async (b, xl, tNorm) => {
        await pendingOnStep;
        return this.denoiseStep(b, xl, tNorm);
      },
      batch,
      nTimestep,
      nSampleStep,
      directionScale,
      noiseScale,
      eta,
      predictSequence,
      sequenceTemperature,
      seed: opts.seed ?? 42,
      rng: opts.rng ?? null,
      initNoise: opts.initNoise ?? null,
      stepNoises: opts.stepNoises ?? null,
      seqNoise: opts.seqNoise ?? null,
      steps: opts.steps ?? null,
      stepSize: opts.stepSize ?? null,
      onStep: (info) => {
        pendingOnStep = Promise.resolve(onStepWrapped(info));
      },
      signal: opts.signal ?? null,
    });
    await pendingOnStep;

    const shouldCenter = opts.center !== false;
    const ca = shouldCenter ? centerCaCoords(ddimRes.xl, batch.gt_atom_mask) : new Float32Array(ddimRes.xl);
    const backbone4 = reconstructBackboneFromCa(ca, {
      chainIds: batch.asym_id,
      center: false,
    });

    const seqSource = ddimRes.ai ?? ddimRes.seqLogits;
    const sequence = seqSource ? sequenceFromRestype(seqSource, L) : 'A'.repeat(L);
    const pdb = backboneToPdb(backbone4, L, {
      sequence,
      chainId: opts.chainId ?? 'A',
      chainIds: batch.asym_id,
      residueIndices: batch.residue_index,
      caOnly: opts.caOnly === true,
    });
    const cif = backboneToMmcif(backbone4, L, {
      entryId: opts.entryId ?? 'GENIE3',
      sequence,
      chainId: opts.chainId ?? 'A',
      chainIds: batch.asym_id,
      residueIndices: batch.residue_index,
      caOnly: opts.caOnly === true,
    });

    const endMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    return {
      length: L,
      stepsRun,
      elapsedMs: endMs - startMs,
      xl: ddimRes.xl,
      ca,
      backbone4,
      sequence,
      ai: ddimRes.ai,
      seqLogits: ddimRes.seqLogits,
      seqProbs: ddimRes.seqProbs,
      seqIndices: ddimRes.seqIndices,
      pdb,
      cif,
    };
  }

  /** Memory accounting snapshot in bytes. */
  memoryUsage() {
    return this.denoiser.memoryUsage();
  }

  destroy() {
    this.denoiser.destroy();
  }
}

/**
 * One-shot helper that creates a `Genie3GeneratorGpu`, runs `generate(opts)`, and destroys GPU buffers.
 */
export async function generateGenie3(device, weights, cfg = null, opts = {}) {
  const gen = await Genie3GeneratorGpu.create(device, weights, cfg, {
    half: opts.half ?? false,
    pairChunkBytes: opts.pairChunkBytes,
    canonicalQuat: opts.canonicalQuat,
  });
  try {
    return await gen.generate(opts);
  } finally {
    gen.destroy();
  }
}
