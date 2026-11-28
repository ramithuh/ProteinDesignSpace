/**
 * Float32 CPU reference implementation of the ArtiosBackbone single-step denoiser.
 *
 * A direct, dependency-free transcription of
 * `artios/model/modules/diffusion_backbone.py::DiffusionModule`, which is the
 * whole of ArtiosBackbone's network: the released checkpoint contains only
 * `diffusion_module.*` tensors.
 *
 * Pipeline, in the AF3 Algorithm 20 vocabulary:
 *
 *   atom attention encoder
 *     `ref_pos`/`r_noisy` -> per-atom embedding -> `average_atom_to_token`
 *     (ArtiosBackbone replaces AF3's local atom-attention tower with this
 *     linear embed + masked token average; there is no `LocalAttnTransformer`
 *     in the checkpoint)
 *   diffusion conditioning
 *     atom conformer pair/single features, motif restype + pair distances,
 *     motif depth, ligand RASA, relative position encoding, Fourier timestep
 *   diffusion transformer
 *     16 AdaLN-modulated DiT blocks, 16 heads, pair-bias attention
 *   atom attention decoder
 *     `broadcast_token_to_atom` + skip, atom-level symmetry-breaking features,
 *     transition + readout to coordinates
 *
 * The EDM preconditioning (`c_in`/`c_out`/`c_skip`) wraps the raw network, so
 * `denoise()` returns the denoised coordinate prediction the sampler consumes.
 *
 * All tensors are flat `Float32Array`s in row-major order; shapes are carried
 * separately. There is no batch dimension and `n_sample` is 1 — a caller that
 * wants several samples runs the denoiser once per sample, sharing the
 * conditioning cache (which is timestep- and sample-independent).
 */

import {
  CONDITIONING_BINS,
  N_ELEMENT,
  N_RESTYPE,
  RELPOS,
  SIGMA_DATA,
  centreRandomAugmentation,
  edmPreconditioning,
  kabschAlign,
  motifDistanceBins,
} from '../shared/index.js';

const ATOM_NAME_CHARS = 256; // 4 chars x 64 slots
const LAYERNORM_EPS = 1e-5;
const ATTENTION_MASK_INF = 1e9;
const AVERAGE_EPS = 1e-4; // average_atom_to_token denominator epsilon

/** Architecture of the released checkpoint (`DiffusionModuleConfig` defaults). */
export const DEFAULT_CONFIG = Object.freeze({
  cToken: 768,
  cAtom: 256,
  cS: 384,
  cZ: 128,
  numBlocks: 16,
  numHeads: 16,
  cTimestepEmbedding: 256,
  sigmaData: SIGMA_DATA,
  transitionExpansion: 2,
});

// ===========================================================================
// kernels
// ===========================================================================

/** `out[r, o] = sum_i x[r, i] * w[o, i] + (bias ? bias[o] : 0)` */
function linear(x, nRows, inDim, w, outDim, bias = null, out = null) {
  const dst = out ?? new Float32Array(nRows * outDim);
  for (let r = 0; r < nRows; r += 1) {
    const xo = r * inDim;
    const do_ = r * outDim;
    for (let o = 0; o < outDim; o += 1) {
      const wo = o * inDim;
      let acc = bias ? bias[o] : 0;
      for (let i = 0; i < inDim; i += 1) acc += x[xo + i] * w[wo + i];
      dst[do_ + o] = acc;
    }
  }
  return dst;
}

/** LayerNorm over the last axis, with optional affine weight/bias. */
function layerNorm(x, nRows, dim, weight = null, bias = null, eps = LAYERNORM_EPS, out = null) {
  const dst = out ?? new Float32Array(nRows * dim);
  for (let r = 0; r < nRows; r += 1) {
    const base = r * dim;
    let mean = 0;
    for (let i = 0; i < dim; i += 1) mean += x[base + i];
    mean /= dim;
    let variance = 0;
    for (let i = 0; i < dim; i += 1) {
      const d = x[base + i] - mean;
      variance += d * d;
    }
    variance /= dim;
    const inv = 1 / Math.sqrt(variance + eps);
    for (let i = 0; i < dim; i += 1) {
      const normed = (x[base + i] - mean) * inv;
      dst[base + i] = (weight ? weight[i] : 1) * normed + (bias ? bias[i] : 0);
    }
  }
  return dst;
}

const silu = (v) => v / (1 + Math.exp(-v));

/**
 * `SwiGLU`: `silu(a) * b` where `[a, b] = Linear(x)` split in half.
 * `w` has shape `[2 * outDim, inDim]`.
 */
function swiglu(x, nRows, inDim, w, outDim) {
  const wide = linear(x, nRows, inDim, w, 2 * outDim);
  const out = new Float32Array(nRows * outDim);
  for (let r = 0; r < nRows; r += 1) {
    const src = r * 2 * outDim;
    const dst = r * outDim;
    for (let i = 0; i < outDim; i += 1) {
      out[dst + i] = silu(wide[src + i]) * wide[src + outDim + i];
    }
  }
  return out;
}

/** Softmax in place over `[nRows, dim]`. */
function softmaxRows(x, nRows, dim) {
  for (let r = 0; r < nRows; r += 1) {
    const base = r * dim;
    let max = -Infinity;
    for (let i = 0; i < dim; i += 1) if (x[base + i] > max) max = x[base + i];
    let sum = 0;
    for (let i = 0; i < dim; i += 1) {
      const e = Math.exp(x[base + i] - max);
      x[base + i] = e;
      sum += e;
    }
    const inv = sum > 0 ? 1 / sum : 0;
    for (let i = 0; i < dim; i += 1) x[base + i] *= inv;
  }
  return x;
}

/**
 * `average_atom_to_token`: masked scatter-sum into tokens, divided by the
 * token's atom count plus `1e-4`.
 *
 * The count is **boolean**, not an integer, and that is not a typo here. The
 * reference implementation
 * (`artios.model.utils.average_atom_to_token`) computes it as
 *
 *     out_count = scatter(src=atom_mask[..., None], ..., reduce="sum")
 *
 * where `atom_mask` is a `torch.bool` tensor, so `scatter_sum` allocates a
 * `bool` accumulator and `scatter_add_` saturates: a token with three atoms
 * gets `True`, i.e. 1, not 3. The divisor is therefore `1.0001` for any token
 * with at least one unmasked atom and `1e-4` for an empty one, which makes this
 * a masked *sum* scaled by `1/1.0001` rather than a mean.
 *
 * The released weights were trained with that behaviour, so reproducing it is
 * required for parity — a true mean changes multi-atom token embeddings by the
 * token's atom count. Verified against the PyTorch oracle: the per-token
 * `s_atom` ratio is exactly `n_atoms_in_token / 1.0001`.
 */
export function averageAtomToToken(xAtom, nAtom, dim, atomToTokenIdx, atomMask, nToken) {
  const out = new Float32Array(nToken * dim);
  const occupied = new Uint8Array(nToken);
  for (let a = 0; a < nAtom; a += 1) {
    if (!atomMask[a]) continue;
    const t = atomToTokenIdx[a];
    occupied[t] = 1;
    const src = a * dim;
    const dst = t * dim;
    for (let i = 0; i < dim; i += 1) out[dst + i] += xAtom[src + i];
  }
  for (let t = 0; t < nToken; t += 1) {
    const inv = 1 / (occupied[t] + AVERAGE_EPS);
    const base = t * dim;
    for (let i = 0; i < dim; i += 1) out[base + i] *= inv;
  }
  return out;
}

/** `broadcast_token_to_atom`: gather each atom's token row. */
export function broadcastTokenToAtom(xToken, dim, atomToTokenIdx) {
  const nAtom = atomToTokenIdx.length;
  const out = new Float32Array(nAtom * dim);
  for (let a = 0; a < nAtom; a += 1) {
    const src = atomToTokenIdx[a] * dim;
    const dst = a * dim;
    for (let i = 0; i < dim; i += 1) out[dst + i] = xToken[src + i];
  }
  return out;
}

// ===========================================================================
// weight access
// ===========================================================================

function createWeightStore(weights) {
  const get = typeof weights?.get === 'function' ? (n) => weights.get(n) : (n) => weights[n];
  const has = typeof weights?.has === 'function' ? (n) => weights.has(n) : (n) => n in weights;
  const seen = new Set();

  /** Fetch a tensor by name, tolerating the `diffusion_module.` prefix. */
  const tensor = (name, expectedShape = null) => {
    let found = name;
    if (!has(found)) {
      const prefixed = `diffusion_module.${name}`;
      if (!has(prefixed)) {
        throw new Error(`createArtiosBackboneCPU: missing weight ${JSON.stringify(name)}`);
      }
      found = prefixed;
    }
    const value = get(found);
    const data = value?.data ?? value;
    const shape = value?.shape ?? null;
    if (!(data instanceof Float32Array)) {
      throw new Error(`createArtiosBackboneCPU: ${name} is not a Float32Array tensor`);
    }
    if (expectedShape && shape) {
      const matches =
        shape.length === expectedShape.length && shape.every((v, i) => v === expectedShape[i]);
      if (!matches) {
        throw new Error(
          `createArtiosBackboneCPU: ${name} has shape [${shape}], expected [${expectedShape}]`,
        );
      }
    }
    seen.add(found);
    return data;
  };

  return { tensor, seen };
}

// ===========================================================================
// model
// ===========================================================================

/**
 * Build a CPU denoiser bound to a set of weights.
 *
 * @param {object} options
 * @param {object} options.weights  a `loadWeightBundle` result, or a plain
 *   object mapping tensor name -> `{shape, data}` / `Float32Array`
 * @param {object} [options.config] architecture overrides (see `DEFAULT_CONFIG`)
 * @returns {{
 *   config: object,
 *   conditioningCache: (features: object, options?: object) => object,
 *   conditioning: (options: object) => {s: Float32Array, z: Float32Array},
 *   fForward: (options: object) => Float32Array,
 *   denoise: (options: object) => Float32Array,
 *   createPredictor: (options: object) => (xNoisy: Float32Array, noiseLevel: number) => Float32Array,
 *   touchedWeights: () => string[]
 * }}
 */
export function createArtiosBackboneCPU({ weights, config = {} }) {
  const cfg = { ...DEFAULT_CONFIG, ...normaliseConfig(config) };
  const { cToken, cAtom, cS, cZ, numBlocks, numHeads, cTimestepEmbedding, sigmaData } = cfg;
  const headDim = cToken / numHeads;
  if (!Number.isInteger(headDim)) {
    throw new Error(`cToken (${cToken}) must be divisible by numHeads (${numHeads})`);
  }

  const { tensor, seen } = createWeightStore(weights);
  const { lower: motifLower, upper: motifUpper, nBins: nMotifDistBins } = motifDistanceBins();

  const atomConditionDim = 1 + 1 + N_ELEMENT + ATOM_NAME_CHARS; // charge, mask, element, name
  const atomReadoutDim = 1 + N_ELEMENT + ATOM_NAME_CHARS; // no ref_mask in the decoder

  // ----- transition blocks (LN -> SwiGLU -> Linear) -----
  const transition = (prefix, x, nRows, dim) => {
    const n = cfg.transitionExpansion;
    const normed = layerNorm(
      x,
      nRows,
      dim,
      tensor(`${prefix}.layernorm.weight`, [dim]),
      tensor(`${prefix}.layernorm.bias`, [dim]),
    );
    const hidden = swiglu(
      normed,
      nRows,
      dim,
      tensor(`${prefix}.swiglu.linear.weight`, [2 * n * dim, dim]),
      n * dim,
    );
    return linear(hidden, nRows, n * dim, tensor(`${prefix}.linear_no_bias.weight`, [dim, n * dim]), dim);
  };

  // =======================================================================
  // conditioning
  // =======================================================================

  /** `DiffusionConditioning.get_atom_condition` */
  function atomCondition(f, capture) {
    const { nAtom, nToken } = f;

    // Token-level atom conditioning only covers "atom tokens": ligands and
    // motif-related conformers.
    const isAtom = new Float32Array(nAtom);
    for (let a = 0; a < nAtom; a += 1) {
      isAtom[a] = f.is_ligand[a] || f.is_motif_related[a] ? 1 : 0;
    }

    const refPosMasked = new Float32Array(nAtom * 3);
    for (let a = 0; a < nAtom; a += 1) {
      for (let c = 0; c < 3; c += 1) refPosMasked[a * 3 + c] = f.ref_pos[a * 3 + c] * isAtom[a];
    }
    const tokenRefPos = averageAtomToToken(
      refPosMasked, nAtom, 3, f.atom_to_token_idx, f.atom_mask, nToken,
    );

    const uidAtom = new Float32Array(nAtom);
    for (let a = 0; a < nAtom; a += 1) uidAtom[a] = f.ref_space_uid[a];
    const tokenUid = averageAtomToToken(
      uidAtom, nAtom, 1, f.atom_to_token_idx, f.atom_mask, nToken,
    );

    // v_lm: 1 where two tokens belong to the same reference conformer.
    const vlm = new Float32Array(nToken * nToken);
    for (let i = 0; i < nToken; i += 1) {
      for (let j = 0; j < nToken; j += 1) {
        vlm[i * nToken + j] = Math.abs(tokenUid[j] - tokenUid[i]) < 1e-2 ? 1 : 0;
      }
    }

    // Pair embedding from reference-position offsets and inverse square distance.
    const wOffsets = tensor('diffusion_conditioning.embed_atom_pair_offsets.weight', [cZ, 3]);
    const wValid = tensor('diffusion_conditioning.embed_atom_pair_offsets_valid.weight', [cZ, 1]);
    const wInvd = tensor('diffusion_conditioning.embed_atom_pair_invd.weight', [cZ, 1]);

    const z = new Float32Array(nToken * nToken * cZ);
    const dlm = new Float32Array(3);
    for (let i = 0; i < nToken; i += 1) {
      for (let j = 0; j < nToken; j += 1) {
        // d_lm[i, j] = ref_pos[i] - ref_pos[j]  (note the orientation)
        for (let c = 0; c < 3; c += 1) dlm[c] = tokenRefPos[i * 3 + c] - tokenRefPos[j * 3 + c];
        const invd = 1 / (1 + dlm[0] * dlm[0] + dlm[1] * dlm[1] + dlm[2] * dlm[2]);
        const valid = vlm[i * nToken + j];
        const base = (i * nToken + j) * cZ;
        for (let c = 0; c < cZ; c += 1) {
          const wo = c * 3;
          const value =
            wOffsets[wo] * dlm[0] +
            wOffsets[wo + 1] * dlm[1] +
            wOffsets[wo + 2] * dlm[2] +
            wValid[c] * valid +
            wInvd[c] * invd;
          z[base + c] = value * valid; // mask out inter-conformer pairs
        }
      }
    }

    // Single embedding from per-atom conformer features.
    const atomFeatures = new Float32Array(nAtom * atomConditionDim);
    for (let a = 0; a < nAtom; a += 1) {
      const base = a * atomConditionDim;
      atomFeatures[base] = f.ref_charge[a];
      atomFeatures[base + 1] = f.ref_mask[a];
      for (let c = 0; c < N_ELEMENT; c += 1) {
        atomFeatures[base + 2 + c] = f.ref_element[a * N_ELEMENT + c];
      }
      for (let c = 0; c < ATOM_NAME_CHARS; c += 1) {
        atomFeatures[base + 2 + N_ELEMENT + c] = f.ref_atom_name_chars[a * ATOM_NAME_CHARS + c];
      }
    }
    const sAtomWise = linear(
      atomFeatures,
      nAtom,
      atomConditionDim,
      tensor('diffusion_conditioning.atom_condition_embed.weight', [cS, atomConditionDim]),
      cS,
    );
    for (let a = 0; a < nAtom; a += 1) {
      if (isAtom[a]) continue;
      sAtomWise.fill(0, a * cS, (a + 1) * cS);
    }
    const s = averageAtomToToken(sAtomWise, nAtom, cS, f.atom_to_token_idx, f.atom_mask, nToken);

    const posEmbed = linear(
      tokenRefPos,
      nToken,
      3,
      tensor('diffusion_conditioning.embed_atom_ref_pos.weight', [cS, 3]),
      cS,
    );
    for (let i = 0; i < s.length; i += 1) s[i] += posEmbed[i];

    if (capture) {
      capture.s_atom = s;
      capture.z_atom = z;
      capture.v_lm = vlm;
      capture.token_ref_pos = tokenRefPos;
    }
    return { s, z };
  }

  /** `DiffusionConditioning.get_motif_condition` */
  function motifCondition(f, capture, augmentation) {
    const { nToken } = f;
    const { coords: motifCoord, rotation, translation } = centreRandomAugmentation(
      f.motif_coordinate,
      {
        mask: f.is_motif_token,
        sTrans: augmentation?.sTrans ?? 1.0,
        rotation: augmentation?.rotation ?? null,
        translation: augmentation?.translation ?? null,
        rng: augmentation?.rng ?? null,
      },
    );

    const s = linear(
      f.motif_restype,
      nToken,
      N_RESTYPE,
      tensor('diffusion_conditioning.embed_motif_restype.weight', [cS, N_RESTYPE]),
      cS,
    );
    const coordEmbed = linear(
      motifCoord,
      nToken,
      3,
      tensor('diffusion_conditioning.embed_motif_coord.weight', [cS, 3]),
      cS,
    );
    for (let t = 0; t < nToken; t += 1) {
      const live = f.is_motif_token[t] ? 1 : 0;
      const base = t * cS;
      for (let c = 0; c < cS; c += 1) s[base + c] = (s[base + c] + coordEmbed[base + c]) * live;
    }

    // Pair distances, binned with the strict `(x > lower) & (x < upper)` rule.
    const wDist = tensor('diffusion_conditioning.embed_motif_pair_dist.weight', [cZ, nMotifDistBins]);
    const z = new Float32Array(nToken * nToken * cZ);
    for (let i = 0; i < nToken; i += 1) {
      for (let j = 0; j < nToken; j += 1) {
        const bothMotif = f.is_motif_token[i] && f.is_motif_token[j];
        if (!bothMotif) continue;
        const dx = motifCoord[j * 3] - motifCoord[i * 3];
        const dy = motifCoord[j * 3 + 1] - motifCoord[i * 3 + 1];
        const dz = motifCoord[j * 3 + 2] - motifCoord[i * 3 + 2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const base = (i * nToken + j) * cZ;
        for (let b = 0; b < nMotifDistBins; b += 1) {
          if (!(dist > motifLower[b] && dist < motifUpper[b])) continue;
          for (let c = 0; c < cZ; c += 1) z[base + c] += wDist[c * nMotifDistBins + b];
        }
      }
    }

    if (capture) {
      capture.s_motif = s;
      capture.z_motif = z;
      capture.motif_coord_augmented = motifCoord;
      capture.motif_augmentation = { rotation, translation };
    }
    return { s, z };
  }

  /** Shared binning for `get_motif_depth_condition` / `get_rasa_condition`. */
  function binnedTokenCondition(values, tokenMask, nToken, binSize, nFineBins, nBins, weightName) {
    const onehot = new Float32Array(nToken * nBins);
    for (let t = 0; t < nToken; t += 1) {
      const value = values[t];
      let bin;
      if (value < 0) {
        bin = nFineBins + 1; // null / unconditioned bin
      } else {
        bin = Math.min(Math.max(Math.floor(value / binSize), 0), nFineBins);
      }
      onehot[t * nBins + bin] = 1;
    }
    const out = linear(onehot, nToken, nBins, tensor(weightName, [cS, nBins]), cS);
    for (let t = 0; t < nToken; t += 1) {
      if (tokenMask[t]) continue;
      out.fill(0, t * cS, (t + 1) * cS);
    }
    return out;
  }

  /** `RelativePositionEncoding.forward` (AF3 Algorithm 3). */
  function relativePositionEncoding(f) {
    const { nToken } = f;
    const { rMax, sMax, nFeatures } = RELPOS;
    const nResidueBins = 2 * rMax + 2;
    const nChainBins = 2 * sMax + 2;
    const differentChainIdx = 2 * rMax + 1;
    const differentEntityIdx = 2 * sMax + 1;

    const features = new Float32Array(nToken * nToken * nFeatures);
    for (let i = 0; i < nToken; i += 1) {
      for (let j = 0; j < nToken; j += 1) {
        const sameChain = f.asym_id[i] === f.asym_id[j];
        const sameResidue = f.residue_index[i] === f.residue_index[j];
        const sameEntity = f.entity_id[i] === f.entity_id[j];
        const base = (i * nToken + j) * nFeatures;

        const dResidue = sameChain
          ? clamp(f.residue_index[i] - f.residue_index[j] + rMax, 0, 2 * rMax)
          : differentChainIdx;
        features[base + dResidue] = 1;

        const dToken = sameChain && sameResidue
          ? clamp(f.token_index[i] - f.token_index[j] + rMax, 0, 2 * rMax)
          : differentChainIdx;
        features[base + nResidueBins + dToken] = 1;

        features[base + 2 * nResidueBins] = sameEntity ? 1 : 0;

        const dChain = sameEntity
          ? clamp(f.sym_id[i] - f.sym_id[j] + sMax, 0, 2 * sMax)
          : differentEntityIdx;
        features[base + 2 * nResidueBins + 1 + dChain] = 1;
      }
    }
    if (2 * nResidueBins + 1 + nChainBins !== nFeatures) {
      throw new Error('relpos feature layout does not sum to RELPOS.nFeatures');
    }
    return linear(
      features,
      nToken * nToken,
      nFeatures,
      tensor('diffusion_conditioning.relpos.embed_relpos_features.weight', [cZ, nFeatures]),
      cZ,
    );
  }

  /**
   * `DiffusionConditioning.compute_cache`: everything that does not depend on
   * the timestep or the noisy coordinates, so it can be reused across all
   * sampling steps and all samples.
   */
  function conditioningCache(features, { capture = null, motifAugmentation = null } = {}) {
    const f = prepareFeatures(features);
    const { nToken } = f;

    const atom = atomCondition(f, capture);
    const motif = motifCondition(f, capture, motifAugmentation);
    const depth = binnedTokenCondition(
      f.motif_depth_norm, f.token_mask, nToken,
      CONDITIONING_BINS.motifDepth.binSize,
      CONDITIONING_BINS.motifDepth.nFineBins,
      CONDITIONING_BINS.motifDepth.nBins,
      'diffusion_conditioning.embed_motif_depth.weight',
    );
    const rasa = binnedTokenCondition(
      f.rasa, f.token_mask, nToken,
      CONDITIONING_BINS.rasa.binSize,
      CONDITIONING_BINS.rasa.nFineBins,
      CONDITIONING_BINS.rasa.nBins,
      'diffusion_conditioning.embed_rasa.weight',
    );
    const relpos = relativePositionEncoding(f);

    // Pair: concat([z_atom + z_motif, relpos]) -> LN -> Linear -> + transition
    const nPair = nToken * nToken;
    const concat = new Float32Array(nPair * 2 * cZ);
    for (let p = 0; p < nPair; p += 1) {
      for (let c = 0; c < cZ; c += 1) {
        concat[p * 2 * cZ + c] = atom.z[p * cZ + c] + motif.z[p * cZ + c];
        concat[p * 2 * cZ + cZ + c] = relpos[p * cZ + c];
      }
    }
    const normed = layerNorm(
      concat, nPair, 2 * cZ,
      tensor('diffusion_conditioning.layernorm_z.weight', [2 * cZ]),
      tensor('diffusion_conditioning.layernorm_z.bias', [2 * cZ]),
    );
    const z = linear(
      normed, nPair, 2 * cZ,
      tensor('diffusion_conditioning.linear_no_bias_z.weight', [cZ, 2 * cZ]),
      cZ,
    );
    const zTransition = transition('diffusion_conditioning.transition_z', z, nPair, cZ);
    for (let i = 0; i < z.length; i += 1) z[i] += zTransition[i];

    // Single: atom + motif + depth + rasa + entity types
    const sBase = new Float32Array(nToken * cS);
    for (let i = 0; i < sBase.length; i += 1) {
      sBase[i] = atom.s[i] + motif.s[i] + depth[i] + rasa[i];
    }

    const entityFeatures = new Float32Array(nToken * 4);
    const motifEntityFeatures = new Float32Array(nToken * 2);
    for (let t = 0; t < nToken; t += 1) {
      entityFeatures[t * 4] = f.is_protein_token[t] ? 1 : 0;
      entityFeatures[t * 4 + 1] = f.is_ligand_token[t] ? 1 : 0;
      entityFeatures[t * 4 + 2] = f.is_dna_token[t] ? 1 : 0;
      entityFeatures[t * 4 + 3] = f.is_rna_token[t] ? 1 : 0;
      motifEntityFeatures[t * 2] = f.is_motif_token[t] ? 1 : 0;
      motifEntityFeatures[t * 2 + 1] = f.is_motif_related_token[t] ? 1 : 0;
    }
    const sEntity = linear(
      entityFeatures, nToken, 4,
      tensor('diffusion_conditioning.entity_type_embed.weight', [cS, 4]), cS,
    );
    const sMotifEntity = linear(
      motifEntityFeatures, nToken, 2,
      tensor('diffusion_conditioning.motif_entity_type_embed.weight', [cS, 2]), cS,
    );
    for (let i = 0; i < sBase.length; i += 1) sBase[i] += sEntity[i] + sMotifEntity[i];

    if (capture) {
      capture.s_motif_depth = depth;
      capture.s_rasa = rasa;
      capture.relpos = relpos;
      capture.s_base = sBase;
      capture.z_cache = z;
    }
    return { sBase, z, nToken, features: f };
  }

  /**
   * `DiffusionConditioning.forward`: add the Fourier timestep embedding to the
   * cached single conditioning and run the two transitions.
   */
  function conditioning({ features, noiseLevel, cache = null, capture = null, motifAugmentation = null }) {
    const resolved = cache ?? conditioningCache(features, { capture, motifAugmentation });
    const { sBase, z } = resolved;
    const nToken = resolved.nToken;

    const tHat = Math.log(noiseLevel / sigmaData) / 4;
    const w = tensor('diffusion_conditioning.fourier_embedding.w', [cTimestepEmbedding]);
    const b = tensor('diffusion_conditioning.fourier_embedding.b', [cTimestepEmbedding]);
    const noiseEmbed = new Float32Array(cTimestepEmbedding);
    for (let c = 0; c < cTimestepEmbedding; c += 1) {
      noiseEmbed[c] = Math.cos(2 * Math.PI * (tHat * w[c] + b[c]));
    }
    const normed = layerNorm(
      noiseEmbed, 1, cTimestepEmbedding,
      tensor('diffusion_conditioning.layernorm_t.weight', [cTimestepEmbedding]),
      tensor('diffusion_conditioning.layernorm_t.bias', [cTimestepEmbedding]),
    );
    const tEmbed = linear(
      normed, 1, cTimestepEmbedding,
      tensor('diffusion_conditioning.linear_t.weight', [cS, cTimestepEmbedding]), cS,
    );

    const s = new Float32Array(nToken * cS);
    for (let t = 0; t < nToken; t += 1) {
      for (let c = 0; c < cS; c += 1) s[t * cS + c] = sBase[t * cS + c] + tEmbed[c];
    }
    const t1 = transition('diffusion_conditioning.transition_s1', s, nToken, cS);
    for (let i = 0; i < s.length; i += 1) s[i] += t1[i];
    const t2 = transition('diffusion_conditioning.transition_s2', s, nToken, cS);
    for (let i = 0; i < s.length; i += 1) s[i] += t2[i];

    if (capture) {
      capture.noise_embed = noiseEmbed;
      capture.s_cond = s;
      capture.z_cond = z;
    }
    return { s, z, nToken, cache: resolved };
  }

  // =======================================================================
  // diffusion transformer
  // =======================================================================

  /** `DiffusionTransformer.forward`: AdaLN parameters, pair bias, 16 DiT blocks. */
  function diffusionTransformer(a, s, z, pairMask, nToken, capture) {
    const fused = 2 * numBlocks * cToken;
    const sNorm = layerNorm(
      s, nToken, cS,
      tensor('diffusion_transformer.condition_layernorm.weight', [cS]),
      tensor('diffusion_transformer.condition_layernorm.bias', [cS]),
    );

    const scale = linear(
      sNorm, nToken, cS,
      tensor('diffusion_transformer.scale_proj.0.weight', [fused, cS]), fused,
      tensor('diffusion_transformer.scale_proj.0.bias', [fused]),
    );
    const shift = linear(
      sNorm, nToken, cS,
      tensor('diffusion_transformer.shift_proj.weight', [fused, cS]), fused,
    );
    const gate = linear(
      sNorm, nToken, cS,
      tensor('diffusion_transformer.gate_proj.0.weight', [fused, cS]), fused,
      tensor('diffusion_transformer.gate_proj.0.bias', [fused]),
    );
    for (let i = 0; i < fused * nToken; i += 1) {
      scale[i] = 1 / (1 + Math.exp(-scale[i]));
      gate[i] = 1 / (1 + Math.exp(-gate[i]));
    }

    const nPair = nToken * nToken;
    const nBias = numHeads * numBlocks;
    const zNorm = layerNorm(
      z, nPair, cZ,
      tensor('diffusion_transformer.z_layernorm.weight', [cZ]),
      tensor('diffusion_transformer.z_layernorm.bias', [cZ]),
    );
    // [nPair, numBlocks * numHeads]; block b owns channels [b*numHeads, ...).
    const pairBias = linear(
      zNorm, nPair, cZ, tensor('diffusion_transformer.z_proj.weight', [nBias, cZ]), nBias,
    );

    let state = Float32Array.from(a);
    for (let block = 0; block < numBlocks; block += 1) {
      // chunk(2) splits attention (first half) from MLP (second half); each
      // half then chunks into numBlocks slices of cToken.
      const attnOffset = block * cToken;
      const mlpOffset = numBlocks * cToken + block * cToken;
      state = ditBlock({
        a: state,
        nToken,
        prefix: `diffusion_transformer.blocks.${block}`,
        scale, shift, gate, fused,
        attnOffset, mlpOffset,
        pairBias, nBias, biasOffset: block * numHeads,
        pairMask,
      });
      if (capture) capture[`dit_block_${block}`] = state;
    }
    return state;
  }

  /** `DiTBlock.forward`: AdaLN + gated pair-bias self-attention, AdaLN + gated SwiGLU MLP. */
  function ditBlock({
    a, nToken, prefix, scale, shift, gate, fused, attnOffset, mlpOffset,
    pairBias, nBias, biasOffset, pairMask,
  }) {
    const out = Float32Array.from(a);

    // --- attention ---
    const attnNorm = layerNorm(
      out, nToken, cToken,
      tensor(`${prefix}.attn_layernorm.weight`, [cToken]),
      tensor(`${prefix}.attn_layernorm.bias`, [cToken]),
    );
    const attnInput = new Float32Array(nToken * cToken);
    for (let t = 0; t < nToken; t += 1) {
      for (let c = 0; c < cToken; c += 1) {
        const i = t * cToken + c;
        attnInput[i] = shift[t * fused + attnOffset + c] + scale[t * fused + attnOffset + c] * attnNorm[i];
      }
    }

    const q = linear(
      attnInput, nToken, cToken,
      tensor(`${prefix}.attention.linear_q.weight`, [cToken, cToken]), cToken,
      tensor(`${prefix}.attention.linear_q.bias`, [cToken]),
    );
    const k = linear(
      attnInput, nToken, cToken,
      tensor(`${prefix}.attention.linear_k.weight`, [cToken, cToken]), cToken,
    );
    const v = linear(
      attnInput, nToken, cToken,
      tensor(`${prefix}.attention.linear_v.weight`, [cToken, cToken]), cToken,
    );

    const invSqrt = 1 / Math.sqrt(headDim);
    const context = new Float32Array(nToken * cToken);
    const logits = new Float32Array(nToken);
    for (let h = 0; h < numHeads; h += 1) {
      const headBase = h * headDim;
      for (let i = 0; i < nToken; i += 1) {
        for (let j = 0; j < nToken; j += 1) {
          let dot = 0;
          const qo = i * cToken + headBase;
          const ko = j * cToken + headBase;
          for (let d = 0; d < headDim; d += 1) dot += q[qo + d] * k[ko + d];
          const bias = pairBias[(i * nToken + j) * nBias + biasOffset + h];
          logits[j] = pairMask[i * nToken + j]
            ? bias + dot * invSqrt
            : -ATTENTION_MASK_INF;
        }
        softmaxRows(logits, 1, nToken);
        const co = i * cToken + headBase;
        for (let j = 0; j < nToken; j += 1) {
          const weight = logits[j];
          if (weight === 0) continue;
          const vo = j * cToken + headBase;
          for (let d = 0; d < headDim; d += 1) context[co + d] += weight * v[vo + d];
        }
      }
    }

    const gating = linear(
      attnInput, nToken, cToken,
      tensor(`${prefix}.attention.gate_linear.0.weight`, [cToken, cToken]), cToken,
      tensor(`${prefix}.attention.gate_linear.0.bias`, [cToken]),
    );
    for (let i = 0; i < context.length; i += 1) {
      context[i] *= 1 / (1 + Math.exp(-gating[i]));
    }
    const attnOut = linear(
      context, nToken, cToken,
      tensor(`${prefix}.attention.linear_output.weight`, [cToken, cToken]), cToken,
    );
    for (let t = 0; t < nToken; t += 1) {
      for (let c = 0; c < cToken; c += 1) {
        out[t * cToken + c] += gate[t * fused + attnOffset + c] * attnOut[t * cToken + c];
      }
    }

    // --- MLP ---
    const mlpNorm = layerNorm(
      out, nToken, cToken,
      tensor(`${prefix}.mlp_layernorm.weight`, [cToken]),
      tensor(`${prefix}.mlp_layernorm.bias`, [cToken]),
    );
    const mlpInput = new Float32Array(nToken * cToken);
    for (let t = 0; t < nToken; t += 1) {
      for (let c = 0; c < cToken; c += 1) {
        const i = t * cToken + c;
        mlpInput[i] = shift[t * fused + mlpOffset + c] + scale[t * fused + mlpOffset + c] * mlpNorm[i];
      }
    }
    const hidden = swiglu(
      mlpInput, nToken, cToken,
      tensor(`${prefix}.mlp.0.linear.weight`, [4 * cToken, cToken]), 2 * cToken,
    );
    const mlpOut = linear(
      hidden, nToken, 2 * cToken,
      tensor(`${prefix}.mlp.1.weight`, [cToken, 2 * cToken]), cToken,
    );
    for (let t = 0; t < nToken; t += 1) {
      for (let c = 0; c < cToken; c += 1) {
        out[t * cToken + c] += gate[t * fused + mlpOffset + c] * mlpOut[t * cToken + c];
      }
    }
    return out;
  }

  // =======================================================================
  // raw network and EDM wrapper
  // =======================================================================

  /**
   * `DiffusionModule.f_forward`: the raw network `F_theta(c_in * x, c_noise)`.
   *
   * @param {object} options
   * @param {Float32Array} options.rNoisyUnit input coordinates already scaled by `c_in`
   * @param {number} options.noiseLevel EDM sigma
   * @param {Float32Array} [options.prevPredUnit] previous prediction divided by `SIGMA_DATA`
   */
  function fForward({
    features, rNoisyUnit, noiseLevel, cache = null, prevPredUnit = null,
    capture = null, motifAugmentation = null,
  }) {
    const cond = conditioning({ features, noiseLevel, cache, capture, motifAugmentation });
    const f = cond.cache.features;
    const { nAtom, nToken } = f;
    if (rNoisyUnit.length !== nAtom * 3) {
      throw new Error(`fForward: rNoisyUnit has ${rNoisyUnit.length} values, expected ${nAtom * 3}`);
    }

    // --- atom attention encoder ---
    const qR = linear(
      rNoisyUnit, nAtom, 3, tensor('linear_no_bias_r_embed.weight', [cToken, 3]), cToken,
    );
    let qSkip = qR;
    if (prevPredUnit) {
      if (prevPredUnit.length !== nAtom * 3) {
        throw new Error(`fForward: prevPredUnit has ${prevPredUnit.length} values, expected ${nAtom * 3}`);
      }
      const qPrev = linear(
        prevPredUnit, nAtom, 3,
        tensor('linear_no_bias_prev_pred_embed.weight', [cToken, 3]), cToken,
      );
      const merged = new Float32Array(nAtom * 2 * cToken);
      for (let a = 0; a < nAtom; a += 1) {
        merged.set(qR.subarray(a * cToken, (a + 1) * cToken), a * 2 * cToken);
        merged.set(qPrev.subarray(a * cToken, (a + 1) * cToken), a * 2 * cToken + cToken);
      }
      qSkip = linear(
        merged, nAtom, 2 * cToken,
        tensor('self_cond_merge.weight', [cToken, 2 * cToken]), cToken,
      );
    }

    const aToken = averageAtomToToken(
      qSkip, nAtom, cToken, f.atom_to_token_idx, f.atom_mask, nToken,
    );

    // --- mix single conditioning into the token representation ---
    const sNorm = layerNorm(
      cond.s, nToken, cS,
      tensor('layernorm_s.weight', [cS]), tensor('layernorm_s.bias', [cS]),
    );
    const sMix = linear(
      sNorm, nToken, cS, tensor('linear_no_bias_s.weight', [cToken, cS]), cToken,
    );
    for (let t = 0; t < nToken; t += 1) {
      const live = f.token_mask[t] ? 1 : 0;
      for (let c = 0; c < cToken; c += 1) {
        const i = t * cToken + c;
        aToken[i] = (aToken[i] + sMix[i]) * live;
      }
    }

    // --- pair mask and masked pair representation ---
    const pairMask = new Uint8Array(nToken * nToken);
    for (let i = 0; i < nToken; i += 1) {
      for (let j = 0; j < nToken; j += 1) {
        pairMask[i * nToken + j] = f.token_mask[i] && f.token_mask[j] ? 1 : 0;
      }
    }
    const zMasked = Float32Array.from(cond.z);
    for (let p = 0; p < nToken * nToken; p += 1) {
      if (pairMask[p]) continue;
      zMasked.fill(0, p * cZ, (p + 1) * cZ);
    }

    if (capture) capture.a_token_pre_dit = Float32Array.from(aToken);

    // --- diffusion transformer ---
    let aOut = diffusionTransformer(aToken, cond.s, zMasked, pairMask, nToken, capture);
    for (let t = 0; t < nToken; t += 1) {
      if (f.token_mask[t]) continue;
      aOut.fill(0, t * cToken, (t + 1) * cToken);
    }
    if (capture) capture.a_token_post_dit = Float32Array.from(aOut);
    aOut = layerNorm(
      aOut, nToken, cToken,
      tensor('layernorm_a.weight', [cToken]), tensor('layernorm_a.bias', [cToken]),
    );

    // --- atom attention decoder ---
    const broadcast = broadcastTokenToAtom(aOut, cToken, f.atom_to_token_idx);
    for (let i = 0; i < broadcast.length; i += 1) broadcast[i] += qSkip[i];
    const qAtom = linear(
      broadcast, nAtom, cToken, tensor('linear_no_bias_a.weight', [cAtom, cToken]), cAtom,
    );

    // Break symmetry between backbone atoms with atom-level identity features.
    const readoutFeatures = new Float32Array(nAtom * atomReadoutDim);
    for (let a = 0; a < nAtom; a += 1) {
      const base = a * atomReadoutDim;
      readoutFeatures[base] = f.ref_charge[a];
      for (let c = 0; c < N_ELEMENT; c += 1) {
        readoutFeatures[base + 1 + c] = f.ref_element[a * N_ELEMENT + c];
      }
      for (let c = 0; c < ATOM_NAME_CHARS; c += 1) {
        readoutFeatures[base + 1 + N_ELEMENT + c] = f.ref_atom_name_chars[a * ATOM_NAME_CHARS + c];
      }
    }
    const atomIdentity = linear(
      readoutFeatures, nAtom, atomReadoutDim,
      tensor('linear_no_bias_f.weight', [cAtom, atomReadoutDim]), cAtom,
    );
    for (let i = 0; i < qAtom.length; i += 1) qAtom[i] += atomIdentity[i];
    if (capture) capture.q_atom_pre_readout = Float32Array.from(qAtom);

    // --- readout ---
    const readoutTransition = transition('readout_transition', qAtom, nAtom, cAtom);
    for (let i = 0; i < qAtom.length; i += 1) qAtom[i] += readoutTransition[i];
    const readoutNorm = layerNorm(
      qAtom, nAtom, cAtom,
      tensor('readout_layernorm.weight', [cAtom]), tensor('readout_layernorm.bias', [cAtom]),
    );
    const out = linear(
      readoutNorm, nAtom, cAtom, tensor('readout_linear.weight', [3, cAtom]), 3,
    );
    if (capture) capture.f_out = out;
    return out;
  }

  /**
   * `DiffusionModule.forward`: EDM-preconditioned denoising prediction.
   *
   * @param {object} options
   * @param {object} options.features feature block (see `prepareFeatures`)
   * @param {Float32Array} options.rNoisy noisy coordinates `[nAtom, 3]`
   * @param {number} options.noiseLevel EDM sigma
   * @param {object} [options.cache] result of `conditioningCache`
   * @param {Float32Array} [options.prevPred] previous denoised prediction (self-conditioning)
   * @returns {Float32Array} denoised coordinates `[nAtom, 3]`
   */
  function denoise({
    features, rNoisy, noiseLevel, cache = null, prevPred = null,
    capture = null, motifAugmentation = null,
  }) {
    const { cIn, cOut, cSkip } = edmPreconditioning(noiseLevel, sigmaData);
    const rNoisyUnit = new Float32Array(rNoisy.length);
    for (let i = 0; i < rNoisy.length; i += 1) rNoisyUnit[i] = rNoisy[i] * cIn;

    let prevPredUnit = null;
    if (prevPred) {
      prevPredUnit = new Float32Array(prevPred.length);
      // Previous predictions are already denoised coordinates, so they are
      // scaled by the fixed data scale rather than the step's c_in.
      for (let i = 0; i < prevPred.length; i += 1) prevPredUnit[i] = prevPred[i] / sigmaData;
    }

    const raw = fForward({
      features, rNoisyUnit, noiseLevel, cache, prevPredUnit, capture, motifAugmentation,
    });
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) out[i] = rNoisy[i] * cSkip + raw[i] * cOut;
    if (capture) {
      capture.r_noisy_unit = rNoisyUnit;
      capture.coords_pred = out;
      capture.edm = { cIn, cOut, cSkip };
    }
    return out;
  }

  /**
   * `DiffusionModule.as_predictor`: a closure for `edmSample`/`edmSamplerStep`
   * that computes the conditioning cache once and chains self-conditioning
   * across steps.
   *
   * Between steps the previous prediction is Kabsch-aligned onto the current
   * noisy coordinates over `is_resolved` (falling back to `atom_mask`), which
   * is what `align_pred_to_true` does in the reference. Pass
   * `alignPrevPred: null` to skip the alignment, or your own
   * `(prev, xNoisy, features) => Float32Array` to replace it.
   */
  function createPredictor({
    features, selfConditioning = true, motifAugmentation = null, alignPrevPred = defaultAlignPrevPred,
  }) {
    const cache = conditioningCache(features, { motifAugmentation });
    let prevPred = null;
    return (xNoisy, noiseLevel) => {
      let prev = prevPred;
      if (prev && alignPrevPred) prev = alignPrevPred(prev, xNoisy, cache.features);
      const coords = denoise({
        features, rNoisy: xNoisy, noiseLevel, cache, prevPred: selfConditioning ? prev : null,
      });
      if (selfConditioning) prevPred = coords;
      return coords;
    };
  }

  return {
    config: { ...cfg, headDim },
    conditioningCache,
    conditioning,
    fForward,
    denoise,
    createPredictor,
    touchedWeights: () => [...seen].sort(),
  };
}

// ===========================================================================
// helpers
// ===========================================================================

const clamp = (value, lo, hi) => (value < lo ? lo : value > hi ? hi : value);

/**
 * `align_pred_to_true`: rigidly align the previous prediction onto the current
 * noisy coordinates before feeding it back as self-conditioning. The reference
 * masks on `is_resolved`.
 */
function defaultAlignPrevPred(prev, xNoisy, features) {
  return kabschAlign(prev, xNoisy, { mask: features.is_resolved ?? features.atom_mask }).coords;
}

/** Accept both `tensor_schema.json`-style and camelCase config keys. */
function normaliseConfig(config) {
  const map = {
    c_token: 'cToken',
    c_atom: 'cAtom',
    c_s: 'cS',
    c_z: 'cZ',
    num_blocks: 'numBlocks',
    num_heads: 'numHeads',
    c_timestep_embedding: 'cTimestepEmbedding',
    sigma_data: 'sigmaData',
    transition_expansion: 'transitionExpansion',
  };
  const out = {};
  for (const [key, value] of Object.entries(config ?? {})) {
    const target = map[key] ?? key;
    if (target in DEFAULT_CONFIG) out[target] = value;
  }
  return out;
}

const REQUIRED_FEATURES = [
  'atom_to_token_idx', 'atom_mask', 'token_mask',
  'token_index', 'residue_index', 'asym_id', 'entity_id', 'sym_id',
  'ref_pos', 'ref_mask', 'ref_element', 'ref_charge', 'ref_atom_name_chars', 'ref_space_uid',
  'is_protein', 'is_ligand', 'is_dna', 'is_rna',
  'is_protein_token', 'is_ligand_token', 'is_dna_token', 'is_rna_token',
  'is_motif', 'is_motif_related', 'is_motif_token', 'is_motif_related_token',
  'motif_restype', 'motif_coordinate', 'motif_depth_norm', 'rasa',
];

const ATOM_FEATURE_WIDTHS = {
  atom_to_token_idx: 1, atom_mask: 1, ref_mask: 1, ref_charge: 1, ref_space_uid: 1,
  is_protein: 1, is_ligand: 1, is_dna: 1, is_rna: 1,
  is_motif: 1, is_motif_related: 1,
  ref_pos: 3, ref_element: N_ELEMENT, ref_atom_name_chars: ATOM_NAME_CHARS,
};

const TOKEN_FEATURE_WIDTHS = {
  token_mask: 1, token_index: 1, residue_index: 1, asym_id: 1, entity_id: 1, sym_id: 1,
  is_protein_token: 1, is_ligand_token: 1, is_dna_token: 1, is_rna_token: 1,
  is_motif_token: 1, is_motif_related_token: 1,
  motif_depth_norm: 1, rasa: 1,
  motif_coordinate: 3, motif_restype: N_RESTYPE,
};

/**
 * Validate a feature block and infer `nAtom` / `nToken`.
 *
 * Accepts typed arrays, plain arrays and boolean arrays interchangeably; the
 * kernels only ever read values and test truthiness.
 */
export function prepareFeatures(features) {
  if (features?.__prepared) return features;
  for (const name of REQUIRED_FEATURES) {
    if (!features?.[name]) throw new Error(`prepareFeatures: missing feature ${name}`);
  }
  const nAtom = features.nAtom ?? features.atom_to_token_idx.length;
  let nToken = features.nToken;
  if (nToken === undefined) {
    nToken = features.token_mask.length;
  }
  if (nAtom !== features.atom_to_token_idx.length) {
    throw new Error(
      `prepareFeatures: nAtom=${nAtom} but atom_to_token_idx has ` +
        `${features.atom_to_token_idx.length} entries`,
    );
  }

  for (const [name, width] of Object.entries(ATOM_FEATURE_WIDTHS)) {
    const got = features[name].length;
    if (got !== nAtom * width) {
      throw new Error(`prepareFeatures: ${name} has ${got} values, expected ${nAtom * width}`);
    }
  }
  for (const [name, width] of Object.entries(TOKEN_FEATURE_WIDTHS)) {
    const got = features[name].length;
    if (got !== nToken * width) {
      throw new Error(`prepareFeatures: ${name} has ${got} values, expected ${nToken * width}`);
    }
  }
  for (let a = 0; a < nAtom; a += 1) {
    const t = features.atom_to_token_idx[a];
    if (!(t >= 0 && t < nToken)) {
      throw new Error(`prepareFeatures: atom_to_token_idx[${a}] = ${t} is out of range [0, ${nToken})`);
    }
  }

  return { ...features, nAtom, nToken, __prepared: true };
}
