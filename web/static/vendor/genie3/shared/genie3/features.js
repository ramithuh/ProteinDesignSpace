/**
 * Protein feature construction and sinusoidal positional encodings for Genie 3.
 *
 * Mirrors:
 * - `genie3/generation/utils/feat_utils.py` (`create_np_features_from_length`,
 *   `_update_np_features_on_token_frame_index`, `_update_np_features_on_token_struct_frame_mask`,
 *   `_update_np_features_on_cond_struct_frame_mask`)
 * - `genie3/generation/utils/encode_utils.py` (`sinusoidal_encoding`)
 */

import {
  CA_ATOM_TYPE_INDEX,
  C_ATOM_SYMBOL_INDEX,
  NUM_ATOM_SYMBOLS,
  NUM_ATOM_TYPES,
  NUM_RESTYPES,
} from './constants.js';

/**
 * Exact Genie 3 sinusoidal encoding matching `genie3/generation/utils/encode_utils.py`.
 *
 * For each element `v` in `values` (length `n`):
 *   For even `j` (0, 2, ..., D-2): `cos(v * pi / N^(2 * j / D))`
 *   For odd  `j` (1, 3, ..., D-1): `sin(v * pi / N^(2 * (j + 1) / D))`
 *
 * NOTE: In upstream `encode_utils.py`:
 *   `k = arange(1, D + 1)`
 *   `sin_div_term = N ** (2 * k / D)` -> at odd index j, k = j + 1 -> `2 * (j + 1) / D`
 *   `cos_div_term = N ** (2 * (k - 1) / D)` -> at even index j, k - 1 = j -> `2 * j / D`
 *
 * @param {ArrayLike<number>} values Length `n`
 * @param {number} N Normalization base (`max_n_res`, `max_n_chain`, or `n_timestep`)
 * @param {number} D Output embedding dimension (must be even)
 * @param {Float32Array} [out] Optional preallocated `[n * D]` output buffer
 * @returns {Float32Array} Shape `[n, D]` flattened
 */
export function sinusoidalEncoding(values, N, D, out = null) {
  const n = values.length;
  const result = out ?? new Float32Array(n * D);

  // Precompute division terms matching PyTorch FP32
  const invDiv = new Float64Array(D);
  for (let j = 0; j < D; j += 2) {
    // Even index j: cos_div_term = N ** (2 * j / D)
    invDiv[j] = Math.PI / Math.pow(N, (2 * j) / D);
    // Odd index j + 1: sin_div_term = N ** (2 * (j + 2) / D)
    invDiv[j + 1] = Math.PI / Math.pow(N, (2 * (j + 2)) / D);
  }

  for (let i = 0; i < n; i += 1) {
    const v = values[i];
    const base = i * D;
    for (let j = 0; j < D; j += 2) {
      result[base + j] = Math.cos(v * invDiv[j]);
      result[base + j + 1] = Math.sin(v * invDiv[j + 1]);
    }
  }
  return result;
}

/**
 * Compute Frenet frame neighbor indices (`token_frame_lindex`, `token_frame_rindex`,
 * `token_frame_cindex_adj`, `token_frame_lindex_adj`, `token_frame_rindex_adj`)
 * per chain (handling both monomers and multi-chain complexes cleanly).
 *
 * Mirrors `_update_np_features_on_token_frame_index` in `feat_utils.py`.
 *
 * @param {object} feat Feature dict with `token_mask`, `atom_type`, `residue_index`, `asym_id`
 * @returns {object} Mutated feature dict with frame index arrays
 */
export function updateTokenFrameIndices(feat) {
  const L = feat.token_mask.length;
  const lindex = new Int32Array(L).fill(-1);
  const rindex = new Int32Array(L).fill(-1);
  const cindexAdj = new Int32Array(L);
  const lindexAdj = new Int32Array(L).fill(-1);
  const rindexAdj = new Int32Array(L).fill(-1);

  // Group token indices by chain (asym_id)
  const chains = new Map();
  for (let i = 0; i < L; i += 1) {
    cindexAdj[i] = i;
    const cid = feat.asym_id[i];
    if (!chains.has(cid)) chains.set(cid, []);
    chains.get(cid).push(i);
  }

  for (const chainTokens of chains.values()) {
    // Collect CA token indices in this chain
    const caIndices = [];
    for (const idx of chainTokens) {
      if (feat.atom_type[idx * NUM_ATOM_TYPES + CA_ATOM_TYPE_INDEX] > 0.5) {
        caIndices.push(idx);
      }
    }
    const caPosMap = new Map();
    for (let j = 0; j < caIndices.length; j += 1) {
      caPosMap.set(caIndices[j], j);
    }

    // Raw left/right indices
    for (let t = 0; t < chainTokens.length; t += 1) {
      const i = chainTokens[t];
      if (caPosMap.has(i)) {
        const j = caPosMap.get(i);
        if (j > 0) lindex[i] = caIndices[j - 1];
        if (j < caIndices.length - 1) rindex[i] = caIndices[j + 1];
      } else {
        if (t > 0) {
          const prev = chainTokens[t - 1];
          if (feat.residue_index[prev] === feat.residue_index[i]) lindex[i] = prev;
        }
        if (t < chainTokens.length - 1) {
          const next = chainTokens[t + 1];
          if (feat.residue_index[i] === feat.residue_index[next]) rindex[i] = next;
        }
      }
    }

    // Adjusted indices adopting adjacent frames for terminal/sidechain atoms
    for (let t = 0; t < chainTokens.length; t += 1) {
      const i = chainTokens[t];
      lindexAdj[i] = lindex[i];
      rindexAdj[i] = rindex[i];
      cindexAdj[i] = i;
    }

    for (let t = 0; t < chainTokens.length; t += 1) {
      const i = chainTokens[t];
      const hasRawFrame = lindex[i] !== -1 && rindex[i] !== -1;
      if (!hasRawFrame) {
        if (caPosMap.has(i)) {
          const j = caPosMap.get(i);
          if (caIndices.length >= 3) {
            if (j === 0) {
              const r = rindex[i];
              lindexAdj[i] = lindex[r];
              rindexAdj[i] = rindex[r];
              cindexAdj[i] = r;
            } else if (j === caIndices.length - 1) {
              const l = lindex[i];
              lindexAdj[i] = lindex[l];
              rindexAdj[i] = rindex[l];
              cindexAdj[i] = l;
            }
          }
        } else if (t > 0) {
          const prev = chainTokens[t - 1];
          lindexAdj[i] = lindexAdj[prev];
          rindexAdj[i] = rindexAdj[prev];
          cindexAdj[i] = cindexAdj[prev];
        }
      }
    }
  }

  feat.token_frame_lindex = lindex;
  feat.token_frame_rindex = rindex;
  feat.token_frame_cindex_adj = cindexAdj;
  feat.token_frame_lindex_adj = lindexAdj;
  feat.token_frame_rindex_adj = rindexAdj;
  return feat;
}

/**
 * Compute `token_struct_mask` and `token_struct_frame_mask` from `gt_atom_mask` and adjusted frame indices.
 * Mirrors `_update_np_features_on_token_struct_frame_mask`.
 *
 * @param {object} feat
 * @returns {object}
 */
export function updateTokenStructFrameMask(feat) {
  const L = feat.gt_atom_mask.length;
  const mask = new Float32Array(L);
  const structMask = new Float32Array(feat.gt_atom_mask);

  for (let i = 0; i < L; i += 1) {
    const ci = feat.token_frame_cindex_adj[i];
    const li = feat.token_frame_lindex_adj[i];
    const ri = feat.token_frame_rindex_adj[i];
    if (
      ci >= 0 && li >= 0 && ri >= 0 &&
      feat.gt_atom_mask[ci] > 0.5 &&
      feat.gt_atom_mask[li] > 0.5 &&
      feat.gt_atom_mask[ri] > 0.5
    ) {
      mask[i] = 1;
    } else {
      mask[i] = 0;
    }
  }

  feat.token_struct_mask = structMask;
  feat.token_struct_frame_mask = mask;
  return feat;
}

/**
 * Compute `cond_struct_frame_mask` from `cond_struct_mask` and raw frame indices.
 * Mirrors `_update_np_features_on_cond_struct_frame_mask`.
 *
 * @param {object} feat
 * @returns {object}
 */
export function updateCondStructFrameMask(feat) {
  const L = feat.cond_struct_mask.length;
  const mask = new Float32Array(L);

  for (let i = 0; i < L; i += 1) {
    const li = feat.token_frame_lindex[i];
    const ri = feat.token_frame_rindex[i];
    if (
      li >= 0 && ri >= 0 &&
      feat.cond_struct_mask[i] > 0.5 &&
      feat.cond_struct_mask[li] > 0.5 &&
      feat.cond_struct_mask[ri] > 0.5
    ) {
      mask[i] = 1;
    } else {
      mask[i] = 0;
    }
  }

  feat.cond_struct_frame_mask = mask;
  return feat;
}

/**
 * Create the exact unconditional Genie 3 feature batch for a protein of length `L` (>= 3).
 *
 * Mirrors `create_np_features_from_length(length)` in `genie3/generation/utils/feat_utils.py`.
 * All per-token 1D masks/indices are flat `Int32Array` or `Float32Array` of length `L`;
 * 2D features are row-major flat `Float32Array` (`gt_restype: [L, 20]`, `atom_type: [L, 37]`,
 * `atom_symbol: [L, 4]`, `gt_atom_positions: [L, 3]`).
 *
 * @param {number} length Number of residues (must be >= 3 for Frenet frames)
 * @param {{ tokenIndexOffset?: number }} [opts]
 * @returns {object} Feature batch object
 */
export function createFeaturesFromLength(length, opts = {}) {
  if (!Number.isInteger(length) || length < 3) {
    throw new Error(`Genie 3 requires length >= 3 for Frenet frames, got ${length}`);
  }
  const offset = opts.tokenIndexOffset ?? 0;

  const token_mask = new Float32Array(length).fill(1);
  const token_index = new Int32Array(length);
  const residue_index = new Int32Array(length);
  const asym_id = new Int32Array(length);
  const sym_id = new Int32Array(length);
  const entity_id = new Int32Array(length);
  const is_atomized = new Float32Array(length);
  const gt_restype = new Float32Array(length * NUM_RESTYPES);
  const atom_type = new Float32Array(length * NUM_ATOM_TYPES);
  const atom_symbol = new Float32Array(length * NUM_ATOM_SYMBOLS);
  const gt_atom_mask = new Float32Array(length).fill(1);
  const gt_atom_positions = new Float32Array(length * 3);
  const cond_group = new Int32Array(length);
  const cond_seq_mask = new Float32Array(length);
  const cond_struct_mask = new Float32Array(length);
  const cond_struct_frame_mask = new Float32Array(length);
  const cond_interface_mask = new Float32Array(length);

  for (let i = 0; i < length; i += 1) {
    token_index[i] = i + offset;
    residue_index[i] = i + 1;
    atom_type[i * NUM_ATOM_TYPES + CA_ATOM_TYPE_INDEX] = 1;
    atom_symbol[i * NUM_ATOM_SYMBOLS + C_ATOM_SYMBOL_INDEX] = 1;
  }

  const batch = {
    length,
    nToken: length,
    token_mask,
    token_index,
    residue_index,
    asym_id,
    sym_id,
    entity_id,
    is_atomized,
    gt_restype,
    atom_type,
    atom_symbol,
    gt_atom_mask,
    gt_atom_positions,
    cond_group,
    cond_seq_mask,
    cond_struct_mask,
    cond_struct_frame_mask,
    cond_interface_mask,
  };

  updateTokenFrameIndices(batch);
  updateTokenStructFrameMask(batch);

  if (offset !== 0) {
    for (const key of [
      'token_frame_lindex',
      'token_frame_rindex',
      'token_frame_lindex_adj',
      'token_frame_rindex_adj',
      'token_frame_cindex_adj',
    ]) {
      const arr = batch[key];
      for (let i = 0; i < length; i += 1) {
        if (arr[i] !== -1) arr[i] += offset;
      }
    }
  }

  return batch;
}

/**
 * Extensible feature builder for unconditional, motif-scaffolding, or binder tasks.
 * Starts from `createFeaturesFromLength(L)` and applies optional conditioning overrides,
 * then recomputes frame masks consistently.
 *
 * @param {{
 *   length: number,
 *   asymId?: ArrayLike<number>,
 *   symId?: ArrayLike<number>,
 *   entityId?: ArrayLike<number>,
 *   residueIndex?: ArrayLike<number>,
 *   condGroup?: ArrayLike<number>,
 *   condSeqMask?: ArrayLike<number>,
 *   condStructMask?: ArrayLike<number>,
 *   condInterfaceMask?: ArrayLike<number>,
 *   gtRestype?: ArrayLike<number>,
 *   gtAtomPositions?: ArrayLike<number>,
 * }} spec
 * @returns {object}
 */
export function createFeaturesFromSpec(spec) {
  const batch = createFeaturesFromLength(spec.length);
  if (spec.asymId) batch.asym_id.set(spec.asymId);
  if (spec.symId) batch.sym_id.set(spec.symId);
  if (spec.entityId) batch.entity_id.set(spec.entityId);
  if (spec.residueIndex) batch.residue_index.set(spec.residueIndex);
  if (spec.condGroup) batch.cond_group.set(spec.condGroup);
  if (spec.condSeqMask) batch.cond_seq_mask.set(spec.condSeqMask);
  if (spec.condStructMask) batch.cond_struct_mask.set(spec.condStructMask);
  if (spec.condInterfaceMask) batch.cond_interface_mask.set(spec.condInterfaceMask);
  if (spec.gtRestype) batch.gt_restype.set(spec.gtRestype);
  if (spec.gtAtomPositions) batch.gt_atom_positions.set(spec.gtAtomPositions);

  updateTokenFrameIndices(batch);
  updateTokenStructFrameMask(batch);
  updateCondStructFrameMask(batch);
  return batch;
}

/**
 * Shallow-clone a feature batch with independent mutable `gt_restype` and `cond_seq_mask` buffers
 * (useful before autoregressive or one-shot sequence decoding).
 *
 * @param {object} batch
 * @returns {object}
 */
export function cloneBatchForSequenceDecoding(batch) {
  return {
    ...batch,
    gt_restype: new Float32Array(batch.gt_restype),
    cond_seq_mask: new Float32Array(batch.cond_seq_mask),
  };
}
