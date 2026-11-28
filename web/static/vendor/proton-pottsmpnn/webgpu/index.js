/**
 * Proton-PottsMPNN — WebGPU backend.
 *
 * The MPNN encoder and the Potts `h`/`J` head (and the teacher-forced decoder)
 * run as WGSL compute kernels over GPU-resident weights; the featurizer, the
 * Potts energies and the pH-selective block-descent optimizer are the same
 * `shared/` code the CPU backend uses, because they are index-heavy and
 * sequential rather than dense arithmetic.
 *
 *     const model = await createProtonPottsWebGPU({ baseUrl: '.cache/weights' });
 *     const out   = await model.run(model.structureFromPDB(pdbText));
 *     console.log(out.h, out.J, out.timings.gpuMs);
 *     model.destroy();
 */

import { loadBundle } from '../cpu/index.js';
import {
  blockDescent,
  buildCausalMasks,
  buildReverseIndex,
  canonicalSequence,
  centreNeighbourhood,
  featurize,
  FEATURIZER_CONSTANTS,
  greedyCoordinateDescent,
  halfToFloat,
  logSoftmaxInPlace,
  makePottsTables,
  pottsCandidateEnergies,
  pottsEnergy,
  pottsMutationDelta,
  resolveCentres,
  selectiveReport,
  structureFromPDB,
  tokenNames,
  validTokenMask,
  V6_DEP_MAP,
  V6_TOKENS,
} from '../shared/index.js';
import { createEngine, requestDevice, tensorToFloat32 } from './engine.js';

export { createEngine, requestDevice, tensorToFloat32 } from './engine.js';
export * as shaders from './shaders.js';

/**
 * Create a WebGPU inference handle.
 *
 * @param {object} opts
 * @param {string} [opts.baseUrl='.cache/weights'] where `manifest.json` and the
 *   safetensors bundles live.
 * @param {'fp32'|'fp16'|'int8'} [opts.precision='fp32'] which bundle to upload.
 *   FP16 and INT8 stay packed on the device and are unpacked inside the GEMM,
 *   always accumulating in FP32.
 * @param {GPUDevice} [opts.device] an existing device to adopt (not destroyed by
 *   `destroy()`; a device this function creates is).
 * @param {(url: string) => Promise<ArrayBuffer>} [opts.read] byte reader override.
 */
export async function createProtonPottsWebGPU(opts = {}) {
  const { manifest, bundle, arch, precision } = await loadBundle(opts);

  let device = opts.device;
  let ownsDevice = false;
  let adapterInfo = opts.adapterInfo ?? null;
  if (!device) {
    const got = await requestDevice(opts.navigatorGpu, opts);
    device = got.device;
    adapterInfo = got.info;
    ownsDevice = true;
  }

  const engine = createEngine(device, { bundle, arch, precision, halfToFloat });
  const numNeighbors = opts.numNeighbors ?? arch.numNeighbors ?? FEATURIZER_CONSTANTS.numNeighbors;
  // Dequantized FP32 copies of the positional-embedding Linear, the only weight
  // the host-side featurizer needs. It must go through the same widening as the
  // GPU weights: in the INT8 bundle this is a quantized 2-D tensor, and using
  // its raw bytes would feed the featurizer integers in place of weights.
  const hostWeights = new Map();
  for (const name of [
    'graph_featurization_module.positional_embedding.embed_positional_features.weight',
    'graph_featurization_module.positional_embedding.embed_positional_features.bias',
  ]) {
    hostWeights.set(name, {
      data: tensorToFloat32(bundle, name, halfToFloat),
      shape: bundle.tensor(name).shape,
    });
  }

  let destroyed = false;
  const check = () => {
    if (destroyed) throw new Error('this ProtonPottsWebGPU handle has been destroyed');
  };

  let lastShape = null;
  let plan = null;

  const uploadShape = (structure, features) => {
    const { L } = structure;
    const { K, edgeIdx, maskE, Eraw } = features;
    if (!plan || lastShape?.L !== L || lastShape?.K !== K) {
      plan = engine.compile({ L, K });
      lastShape = { L, K };
    }
    const reverse = buildReverseIndex(edgeIdx, L, K);
    const mergeValid = new Float32Array(L * K);
    for (let e = 0; e < L * K; e += 1) {
      const j = edgeIdx[e];
      const ok = reverse.hasReverse[e] && maskE[e] && maskE[j * K + reverse.reverseK[e]];
      mergeValid[e] = ok ? 1 : 0;
    }
    engine.write(plan.Eraw, Eraw);
    engine.write(plan.edgeIdx, Uint32Array.from(edgeIdx));
    engine.write(plan.maskE, Float32Array.from(maskE));
    engine.write(plan.residueMask, Float32Array.from(structure.residueMask));
    engine.write(plan.reverseK, Uint32Array.from(reverse.reverseK));
    engine.write(plan.mergeValid, mergeValid);
    return { plan, reverse, planShape: lastShape };
  };

  /**
   * Featurize on the host, then run the encoder and Potts head on the GPU.
   * `timings.gpuMs` is the submit-to-`onSubmittedWorkDone` wall time, so a warm
   * call measures the kernels rather than the upload.
   */
  const run = async (structure, options = {}) => {
    check();
    const t0 = performance.now();
    const features = featurize(structure, hostWeights, numNeighbors);
    const tFeat = performance.now();
    const { reverse, planShape } = uploadShape(structure, features);
    const tUpload = performance.now();

    const encoder = device.createCommandEncoder({ label: 'proton-potts-run' });
    plan.encodeAndHead(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const tGpu = performance.now();

    const L = structure.L;
    const { K } = features;
    const V = arch.pottsVocabSize;
    const J = await engine.readFloats(plan.Jmerged, L * K * V * V);
    const h = await engine.readFloats(plan.h, L * V);
    const tRead = performance.now();

    const tables = makePottsTables(J, features.edgeIdx, L, K, V);
    const seq = options.seq ?? structure.tokens;
    return {
      structure,
      features,
      reverse,
      planShape,
      J,
      h,
      tables,
      edgeIdx: features.edgeIdx,
      K,
      V,
      L,
      energy: pottsEnergy(J, features.edgeIdx, L, K, V, seq),
      timings: {
        featurizeMs: tFeat - t0,
        uploadMs: tUpload - tFeat,
        gpuMs: tGpu - tUpload,
        readbackMs: tRead - tGpu,
        totalMs: tRead - t0,
      },
    };
  };

  /** Teacher-forced decoder on the GPU: logits and log-probabilities. */
  const decode = async (out, options = {}) => {
    check();
    if (!arch.hasDecoder) throw new Error('this checkpoint has no decoder');
    // The decoder reuses the encoder's own `h_E` / `h_V` buffers, so it is only
    // meaningful for the run that last filled them. A later `run()` on a
    // different shape recompiles the plan and overwrites them.
    if (out.planShape !== lastShape) {
      throw new Error(
        'decode() must follow the run() it is given: another run has since '
        + 'overwritten the GPU-resident encoder state',
      );
    }
    const {
      causalityPattern = 'conditional_minus_self',
      initializeWithGroundTruth = true,
      decodingOrder = null,
      seq = out.structure.tokens,
    } = options;
    const { L } = out.structure;
    const H = arch.hiddenDim;
    const Vv = arch.vocabSize;
    const { causal, anti } = buildCausalMasks(
      out.structure, out.features, causalityPattern, decodingOrder,
    );
    const cmask = new Float32Array(L * out.K * 2);
    for (let e = 0; e < L * out.K; e += 1) {
      cmask[e * 2] = causal[e];
      cmask[e * 2 + 1] = anti[e];
    }
    const hS = new Float32Array(L * H);
    if (initializeWithGroundTruth) {
      for (let i = 0; i < L; i += 1) {
        hS.set(engine.WsHost.subarray(seq[i] * H, seq[i] * H + H), i * H);
      }
    }
    engine.write(plan.cmask, cmask);
    engine.write(plan.hS, hS);

    const t0 = performance.now();
    const encoder = device.createCommandEncoder({ label: 'proton-potts-decode' });
    plan.decode(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const gpuMs = performance.now() - t0;

    const logits = await engine.readFloats(plan.logits, L * Vv);
    const logProbs = Float32Array.from(logits);
    logSoftmaxInPlace(logProbs, L, Vv);
    return { logits, logProbs, timings: { gpuMs } };
  };

  return {
    backend: 'webgpu',
    precision,
    manifest,
    arch,
    vocab: V6_TOKENS,
    device,
    adapterInfo,
    weightBytes: engine.weightBytes,
    weightDtype: engine.weightDtype,

    structureFromPDB,
    run,
    decode,

    pottsEnergy: (out, seq = out.structure.tokens) =>
      pottsEnergy(out.J, out.edgeIdx, out.L, out.K, out.V, seq),
    candidateEnergies: (out, seq = out.structure.tokens) =>
      pottsCandidateEnergies(out.tables, seq),
    mutationDelta: (out, seq = out.structure.tokens) => pottsMutationDelta(out.tables, seq),
    resolveCentres: (structure, centres, depMap = V6_DEP_MAP) =>
      resolveCentres(structure, centres, depMap),
    selectiveGap: (out, centres, seq = out.structure.tokens) =>
      selectiveReport(out.tables, seq, centres),
    centreNeighbourhood: (out, centres, o) => centreNeighbourhood(out.tables, centres, o),

    optimize: (out, options) => {
      check();
      const res = blockDescent(out.tables, {
        seq: options.seq ?? out.structure.tokens,
        resSeq: out.structure.resSeq,
        validMask: options.validMask ?? validTokenMask(options.forbiddenTokens),
        ...options,
      });
      return { ...res, tokenNames: tokenNames(res.tokens), sequence: canonicalSequence(res.tokens) };
    },

    optimizeGreedy: (out, options) => {
      check();
      const res = greedyCoordinateDescent(out.tables, {
        seq: options.seq ?? out.structure.tokens,
        resSeq: out.structure.resSeq,
        validMask: options.validMask ?? validTokenMask(options.forbiddenTokens),
        ...options,
      });
      return { ...res, tokenNames: tokenNames(res.tokens), sequence: canonicalSequence(res.tokens) };
    },

    /**
     * Release the GPU buffers, and the device too if this handle created it.
     * Idempotent.
     */
    destroy() {
      if (destroyed) return;
      destroyed = true;
      engine.destroy();
      plan = null;
      hostWeights.clear();
      if (ownsDevice) device.destroy();
    },
  };
}

export default createProtonPottsWebGPU;
