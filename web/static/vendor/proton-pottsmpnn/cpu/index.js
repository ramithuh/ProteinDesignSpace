/**
 * Proton-PottsMPNN — pure-JavaScript CPU backend.
 *
 * The reference implementation of the port: no WebGPU, no npm dependencies, and
 * numerically matched to the PyTorch FP32 oracle to `relRMS < 1e-4`. It is both
 * the fallback when no adapter is available and the thing the GPU backend is
 * checked against.
 *
 *     const model = await createProtonPottsCPU({ baseUrl: '.cache/weights' });
 *     const out   = model.run(structureFromPDB(pdbText));
 *     model.destroy();
 */

import {
  archFromManifest,
  blockDescent,
  canonicalSequence,
  centreNeighbourhood,
  featurize,
  FEATURIZER_CONSTANTS,
  greedyCoordinateDescent,
  linearForward,
  makePottsTables,
  materializeWeights,
  parseSafetensors,
  pottsCandidateEnergies,
  pottsEnergy,
  pottsMutationDelta,
  resolveCentres,
  runDecoderAutoregressive,
  runDecoderTeacherForcing,
  runEncoder,
  runPottsHead,
  selectiveReport,
  structureFromPDB,
  tokenNames,
  validTokenMask,
  V6_DEP_MAP,
  V6_TOKENS,
} from '../shared/index.js';

export const PRECISIONS = Object.freeze(['fp32', 'fp16', 'int8']);

/**
 * Default reader: Node's `fs` when available, otherwise `fetch`.
 * Resolved lazily so the module stays importable in a browser.
 */
async function defaultRead(url) {
  if (typeof process !== 'undefined' && process.versions?.node && !/^https?:/.test(url)) {
    const { readFile } = await import('node:fs/promises');
    const buf = await readFile(url);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status} ${res.statusText}`);
  return res.arrayBuffer();
}

function joinUrl(base, name) {
  if (!base) return name;
  return base.endsWith('/') ? `${base}${name}` : `${base}/${name}`;
}

/**
 * Load the manifest and one precision's safetensors bundle.
 *
 * Shared by both backends: the GPU engine needs the same manifest and the same
 * raw bundle, it just keeps the tensors packed instead of widening them.
 */
export async function loadBundle({ baseUrl = '.cache/weights', precision = 'fp32', read = defaultRead } = {}) {
  if (!PRECISIONS.includes(precision)) {
    throw new Error(`precision must be one of ${PRECISIONS.join(', ')}, got "${precision}"`);
  }
  const manifest = JSON.parse(
    new TextDecoder().decode(new Uint8Array(await read(joinUrl(baseUrl, 'manifest.json')))),
  );
  const entry = manifest.precisions[precision];
  if (!entry) throw new Error(`manifest has no "${precision}" bundle`);
  const buffer = await read(joinUrl(baseUrl, entry.file));
  const bundle = parseSafetensors(buffer);
  return { manifest, bundle, arch: archFromManifest(manifest), precision, file: entry.file };
}

/**
 * Create a CPU inference handle.
 *
 * @param {object} opts
 * @param {string} [opts.baseUrl='.cache/weights'] directory or URL prefix holding
 *   `manifest.json` and the safetensors bundles written by `tools/export_weights.py`.
 * @param {'fp32'|'fp16'|'int8'} [opts.precision='fp32'] which bundle to load. FP16
 *   and INT8 are widened to FP32 on load, so the arithmetic is always FP32 (plus
 *   double-precision accumulators) and only the weights lose precision.
 * @param {(url: string) => Promise<ArrayBuffer>} [opts.read] byte reader override.
 */
export async function createProtonPottsCPU(opts = {}) {
  const { manifest, bundle, arch, precision } = await loadBundle(opts);
  const weights = materializeWeights(bundle);
  const numNeighbors = opts.numNeighbors ?? arch.numNeighbors ?? FEATURIZER_CONSTANTS.numNeighbors;
  let destroyed = false;

  const check = () => {
    if (destroyed) throw new Error('this ProtonPottsCPU handle has been destroyed');
  };

  /**
   * Featurize a structure and run the encoder + Potts head.
   *
   * @returns {object} `{ structure, features, encoder, J, h, tables, energy, ... }`
   */
  const run = (structure, options = {}) => {
    check();
    const t0 = performance.now();
    const features = featurize(structure, weights, numNeighbors);
    const tFeat = performance.now();
    const encoder = runEncoder(structure, features, weights, arch, linearForward);
    const tEnc = performance.now();
    const { J, h, reverse } = runPottsHead(structure, features, encoder, weights, arch, linearForward);
    const tHead = performance.now();
    const tables = makePottsTables(J, features.edgeIdx, structure.L, features.K, arch.pottsVocabSize);
    const seq = options.seq ?? structure.tokens;
    const energy = pottsEnergy(J, features.edgeIdx, structure.L, features.K, arch.pottsVocabSize, seq);
    return {
      structure,
      features,
      encoder,
      J,
      h,
      reverse,
      tables,
      edgeIdx: features.edgeIdx,
      K: features.K,
      V: arch.pottsVocabSize,
      L: structure.L,
      energy,
      timings: {
        featurizeMs: tFeat - t0,
        encoderMs: tEnc - tFeat,
        pottsHeadMs: tHead - tEnc,
        totalMs: tHead - t0,
      },
    };
  };

  return {
    backend: 'cpu',
    precision,
    manifest,
    arch,
    vocab: V6_TOKENS,
    weights,

    /** Parse a PDB string into the model's input tensors. */
    structureFromPDB,

    run,

    /** `E_global = H(s)` for any sequence against already-computed tables. */
    pottsEnergy: (out, seq = out.structure.tokens) =>
      pottsEnergy(out.J, out.edgeIdx, out.L, out.K, out.V, seq),

    /** Per-position conditional energies `[L, V]`. */
    candidateEnergies: (out, seq = out.structure.tokens) =>
      pottsCandidateEnergies(out.tables, seq),

    /** Exact `dH` for every single substitution, `[L, V]`. */
    mutationDelta: (out, seq = out.structure.tokens) => pottsMutationDelta(out.tables, seq),

    /** Resolve centre specs (`{ position | chain+resSeq, protonationType }`). */
    resolveCentres: (structure, centres, depMap = V6_DEP_MAP) =>
      resolveCentres(structure, centres, depMap),

    /** Per-centre `E_P - E_D` and their sum. */
    selectiveGap: (out, centres, seq = out.structure.tokens) =>
      selectiveReport(out.tables, seq, centres),

    /** The designable neighbourhood of a set of centres. */
    centreNeighbourhood: (out, centres, o) => centreNeighbourhood(out.tables, centres, o),

    /**
     * pH-selective block descent:
     * `minimise (1 - lambda) * z(E_global) + lambda * z(sum (E_P - E_D))`.
     */
    optimize: (out, options) => {
      check();
      const res = blockDescent(out.tables, {
        seq: options.seq ?? out.structure.tokens,
        resSeq: out.structure.resSeq,
        validMask: options.validMask ?? validTokenMask(options.forbiddenTokens),
        ...options,
      });
      return {
        ...res,
        tokenNames: tokenNames(res.tokens),
        sequence: canonicalSequence(res.tokens),
      };
    },

    /** `blockSize = 1` greedy coordinate descent on the same objective. */
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

    /** Teacher-forced decoder logits / log-probabilities. */
    decode: (out, options = {}) => {
      check();
      return runDecoderTeacherForcing(
        out.structure, out.features, out.encoder, weights, arch,
        { matmul: linearForward, ...options },
      );
    },

    /** Autoregressive sampling from the decoder (seeded, reproducible). */
    sample: (out, options = {}) => {
      check();
      return runDecoderAutoregressive(
        out.structure, out.features, out.encoder, weights, arch,
        { matmul: linearForward, ...options },
      );
    },

    /** Release the materialised weights. Idempotent. */
    destroy() {
      if (destroyed) return;
      destroyed = true;
      weights.clear();
    },
  };
}

export default createProtonPottsCPU;
