/**
 * Potts factorized energy, conditional logits, pseudo-log-likelihood scoring and
 * the alphabet/bias masking applied before sampling.
 *
 * Mirrors `artios.model.modules.potts` (`compute_potts_energy`,
 * `compute_potts_single_logits`, `build_mutable_mask`, `build_alphabet_mask`,
 * `build_energy_bias_vector`, `build_group_energy_bias_vector`) and the mask/bias
 * block of `ArtiosPotts.sample_from_params`.
 *
 * The ports work on one structure at a time, so there is no batch dimension:
 * `h` is `[nToken, R]`, `J` is `[nToken, k, R, R]`, both flat and row major.
 *
 * @typedef {object} PottsParams
 * @property {Float32Array} h fields, `[nToken, R]`
 * @property {Float32Array} J couplings, `[nToken, k, R, R]`
 * @property {Int32Array} edgeIndex neighbour indices, `[nToken, k]`
 * @property {Uint8Array} edgeMask valid-edge mask, `[nToken, k]`
 * @property {number} nToken
 * @property {number} k neighbours per node
 * @property {number} numResidueTypes
 */

import {
  ALPHABET_BIAS,
  NUM_STANDARD_AA,
  CYS_INDEX,
  MET_INDEX,
  ONE_LETTER_TO_INDEX,
  RESIDUE_GROUPS,
} from './constants.js';

/**
 * Conditional logits `logit_i(a) = -(h_i(a) + sum_j J_ij(a, s_j))`, i.e.
 * `log p(s_i = a | s_{N(i)})` up to a constant. Self-edges and masked edges are
 * excluded, matching `compute_potts_single_logits`.
 *
 * @param {PottsParams} params
 * @param {ArrayLike<number>} sequence restype index per token
 * @param {Float32Array} [out]
 * @returns {Float32Array} `[nToken, R]`
 */
export function pottsSingleLogits(params, sequence, out) {
  const { h, J, edgeIndex, edgeMask, nToken, k, numResidueTypes: R } = params;
  const logits = out ?? new Float32Array(nToken * R);
  const acc = new Float64Array(R);

  for (let i = 0; i < nToken; i += 1) {
    for (let a = 0; a < R; a += 1) acc[a] = h[i * R + a];
    for (let s = 0; s < k; s += 1) {
      const edge = i * k + s;
      const j = edgeIndex[edge];
      if (j === i || !edgeMask[edge]) continue; // not_self & edge_mask
      const sj = sequence[j];
      const base = edge * R * R;
      for (let a = 0; a < R; a += 1) acc[a] += J[base + a * R + sj];
    }
    for (let a = 0; a < R; a += 1) logits[i * R + a] = -acc[a];
  }
  return logits;
}

/**
 * Per-node Potts energy `E_i(s) = h_i(s_i) + 0.5 * sum_j J_ij(s_i, s_j)`, so the
 * node sum is the total energy with every undirected edge counted once
 * (`compute_potts_energy`).
 *
 * @param {PottsParams} params
 * @param {ArrayLike<number>} sequence
 * @param {Float32Array} [out]
 * @returns {Float32Array} `[nToken]`
 */
export function pottsEnergyPerNode(params, sequence, out) {
  const { h, J, edgeIndex, edgeMask, nToken, k, numResidueTypes: R } = params;
  const energy = out ?? new Float32Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    const si = sequence[i];
    let coupling = 0;
    for (let s = 0; s < k; s += 1) {
      const edge = i * k + s;
      const j = edgeIndex[edge];
      if (j === i || !edgeMask[edge]) continue;
      coupling += J[edge * R * R + si * R + sequence[j]];
    }
    energy[i] = h[i * R + si] + 0.5 * coupling;
  }
  return energy;
}

/** Total Potts energy of `sequence`, with every edge counted once. */
export function pottsEnergy(params, sequence) {
  const perNode = pottsEnergyPerNode(params, sequence);
  let total = 0;
  for (let i = 0; i < perNode.length; i += 1) total += perNode[i];
  return total;
}

/** In-place log-softmax of `values[offset .. offset+count)`. */
export function logSoftmaxInto(values, offset, count) {
  let max = -Infinity;
  for (let i = 0; i < count; i += 1) {
    if (values[offset + i] > max) max = values[offset + i];
  }
  let sum = 0;
  for (let i = 0; i < count; i += 1) sum += Math.exp(values[offset + i] - max);
  const logSum = max + Math.log(sum);
  for (let i = 0; i < count; i += 1) values[offset + i] -= logSum;
  return values;
}

/**
 * Score a sequence on precomputed Potts parameters, as `PottsDesigner.score`.
 *
 * `energy` is summed over protein positions; the pseudo-log-likelihood is the
 * mean of `log p(s_i | rest)` restricted to the 20 standard amino acids, over
 * protein positions that actually hold one.
 *
 * @param {PottsParams} params
 * @param {ArrayLike<number>} sequence restype index per token
 * @param {object} [options]
 * @param {ArrayLike<number>} [options.isProteinToken] defaults to every token
 * @returns {{perNodeEnergy: Float32Array, totalEnergy: number, proteinEnergy: number,
 *   logits: Float32Array, logLikelihood: Float32Array, pseudoLogLikelihood: number,
 *   scoredPositions: number[]}}
 */
export function scoreSequence(params, sequence, options = {}) {
  const { nToken, numResidueTypes: R } = params;
  const isProteinToken = options.isProteinToken ?? new Uint8Array(nToken).fill(1);

  const perNodeEnergy = pottsEnergyPerNode(params, sequence);
  const logits = pottsSingleLogits(params, sequence);

  // log_softmax over the 20 standard amino acid channels only (clamped to the
  // row width, so a reduced alphabet cannot read past the end of a row).
  const numAa = Math.min(NUM_STANDARD_AA, R);
  const logLikelihood = new Float32Array(nToken).fill(NaN);
  const row = new Float64Array(numAa);
  const scoredPositions = [];
  let llSum = 0;
  for (let i = 0; i < nToken; i += 1) {
    for (let a = 0; a < numAa; a += 1) row[a] = logits[i * R + a];
    logSoftmaxInto(row, 0, numAa);
    const si = sequence[i];
    if (si < numAa) logLikelihood[i] = row[si];
    if (isProteinToken[i] && si < numAa) {
      scoredPositions.push(i);
      llSum += row[si];
    }
  }

  let totalEnergy = 0;
  let proteinEnergy = 0;
  for (let i = 0; i < nToken; i += 1) {
    totalEnergy += perNodeEnergy[i];
    if (isProteinToken[i]) proteinEnergy += perNodeEnergy[i];
  }

  return {
    perNodeEnergy,
    totalEnergy,
    proteinEnergy,
    logits,
    logLikelihood,
    pseudoLogLikelihood: scoredPositions.length ? llSum / scoredPositions.length : NaN,
    scoredPositions,
  };
}

/** `build_mutable_mask`: the valid tokens that are not identity-fixed. */
export function buildMutableMask(tokenMask, fixedMask = null) {
  const out = new Uint8Array(tokenMask.length);
  for (let i = 0; i < tokenMask.length; i += 1) {
    out[i] = tokenMask[i] && !(fixedMask && fixedMask[i]) ? 1 : 0;
  }
  return out;
}

/**
 * `build_alphabet_mask`: allowed residue types per position, `[nToken, R]`.
 *
 * Mutable positions may only take the 20 standard amino acids; non-mutable
 * positions allow every state so a fixed or non-protein token keeps its identity
 * without an alphabet-wide energy bias. The CYS/MET exclusions apply only where
 * we are free to mutate.
 */
export function buildAlphabetMask(mutableMask, numResidueTypes, excludeCys = false, excludeMet = false) {
  const n = mutableMask.length;
  const R = numResidueTypes;
  const mask = new Uint8Array(n * R);
  for (let i = 0; i < n; i += 1) {
    if (mutableMask[i]) {
      for (let a = 0; a < NUM_STANDARD_AA; a += 1) mask[i * R + a] = 1;
      if (excludeCys) mask[i * R + CYS_INDEX] = 0;
      if (excludeMet) mask[i * R + MET_INDEX] = 0;
    } else {
      mask.fill(1, i * R, i * R + R);
    }
  }
  return mask;
}

/** `build_energy_bias_vector`: one-letter-code -> value, or `null` if all zero. */
export function buildEnergyBiasVector(energyBias, numResidueTypes) {
  const bias = new Float32Array(numResidueTypes);
  let any = false;
  for (const [code, value] of Object.entries(energyBias ?? {})) {
    if (!value) continue;
    const index = ONE_LETTER_TO_INDEX[code.toUpperCase()];
    if (index === undefined) throw new Error(`unknown residue code ${JSON.stringify(code)}`);
    bias[index] = value;
    any = true;
  }
  return any ? bias : null;
}

/**
 * `build_group_energy_bias_vector`: group name -> value, accumulating for
 * residues in more than one group (HIS is both aromatic and cationic).
 */
export function buildGroupEnergyBiasVector(groupBias, numResidueTypes) {
  const bias = new Float32Array(numResidueTypes);
  let any = false;
  for (const [group, value] of Object.entries(groupBias ?? {})) {
    if (!value) continue;
    const members = RESIDUE_GROUPS[group];
    if (!members) {
      throw new Error(
        `unknown residue group ${JSON.stringify(group)}; expected one of ` +
        `${Object.keys(RESIDUE_GROUPS).sort().join(', ')}`,
      );
    }
    for (const index of members) bias[index] += value;
    any = true;
  }
  return any ? bias : null;
}

/**
 * Apply the alphabet mask and the per-residue energy biases to `h`, as the
 * mask/bias block of `ArtiosPotts.sample_from_params` does:
 * `h = h + ~alphabet_mask * ALPHABET_BIAS + energy_bias + group_energy_bias`.
 *
 * @returns {Float32Array} a new `[nToken, R]` field array
 */
export function applyFieldBias(h, numResidueTypes, options = {}) {
  const {
    alphabetMask = null,
    energyBias = null,
    groupEnergyBias = null,
    alphabetBias = ALPHABET_BIAS,
  } = options;
  const R = numResidueTypes;
  const out = Float32Array.from(h);
  for (let i = 0; i < out.length / R; i += 1) {
    for (let a = 0; a < R; a += 1) {
      let value = out[i * R + a];
      if (alphabetMask && !alphabetMask[i * R + a]) value += alphabetBias;
      if (energyBias) value += energyBias[a];
      if (groupEnergyBias) value += groupEnergyBias[a];
      out[i * R + a] = value;
    }
  }
  return out;
}

/**
 * Residue states a row-restricted sampler may drop without changing its output.
 *
 * State `a` is dead when, at every mutable row, its field exceeds the row
 * minimum by more than `2 k max|J| + penaltyScale * penaltyMax + 746 Tmax + 1`.
 * Then `exp(-(E_a - E_min) / T)` underflows to exactly 0 for every reachable
 * neighbourhood, penalty and temperature: the state adds +0 to every sum and can
 * never be drawn, so leaving it out is bit-identical to keeping it. (`exp(x)` is
 * exactly 0 below -745.14; the +1 absorbs float32 rounding of the energies.)
 *
 * `columns` is 20 when every non-amino-acid state is dead — the usual case,
 * since mutable rows carry `ALPHABET_BIAS` there — and `R` otherwise, e.g. for a
 * high `temperatureInit` or a small `alphabetBias`. `liveLetters` marks which of
 * the 20 amino acids are live, so a penalty need not be evaluated for the rest.
 *
 * @param {PottsParams[]} paramsList every parameter set the rows are read from
 *   (conditional, and unconditional under CFG), with biased fields
 * @param {object} options
 * @param {ArrayLike<number>} [options.mutableMask] rows that are sampled
 * @param {number} options.temperatureMax
 * @param {number} options.temperatureMin
 * @param {number} [options.penaltyMax] upper bound on a penalty gradient entry
 * @param {number} [options.penaltyScale] weight of the gradient in the energy
 * @returns {{columns: number, liveLetters: Uint8Array}}
 */
export function pottsLiveStates(paramsList, options) {
  const {
    mutableMask = null,
    temperatureMax,
    temperatureMin = temperatureMax,
    penaltyMax = 0,
    penaltyScale = 0.5,
  } = options;
  const { nToken, k, numResidueTypes: R } = paramsList[0];
  const Q = NUM_STANDARD_AA;
  const allLive = { columns: R, liveLetters: new Uint8Array(Math.min(Q, R)).fill(1) };
  // A temperature this small could overflow -E / T, where the argument fails.
  if (R <= Q || !(temperatureMin > 1e-30) || !(temperatureMax >= temperatureMin)) return allLive;

  const dead = new Uint8Array(R).fill(1);
  let anyRow = false;
  for (const { h, J } of paramsList) {
    let jMax = 0;
    for (let x = 0; x < J.length; x += 1) {
      const v = J[x] < 0 ? -J[x] : J[x];
      if (!(v <= jMax)) jMax = v; // a NaN coupling poisons the bound, as it must
    }
    // E_a >= h_a - k jMax, and E_best <= h_min + k jMax + penaltyScale * penaltyMax.
    const spread = 2 * k * jMax + penaltyScale * penaltyMax + 746 * temperatureMax + 1;
    if (!Number.isFinite(spread)) return allLive;
    for (let i = 0; i < nToken; i += 1) {
      if (mutableMask && !mutableMask[i]) continue;
      anyRow = true;
      let rowMin = Infinity;
      for (let a = 0; a < R; a += 1) if (h[i * R + a] < rowMin) rowMin = h[i * R + a];
      if (!Number.isFinite(rowMin)) return allLive;
      for (let a = 0; a < R; a += 1) if (!(h[i * R + a] - rowMin > spread)) dead[a] = 0;
    }
  }
  if (!anyRow) return allLive;

  let tailDead = true;
  for (let a = Q; a < R; a += 1) if (!dead[a]) tailDead = false;
  const liveLetters = new Uint8Array(Q);
  for (let a = 0; a < Q; a += 1) liveLetters[a] = dead[a] ? 0 : 1;
  return { columns: tailDead ? Q : R, liveLetters };
}

/**
 * Row-at-a-time access to the conditional energies, for samplers that update a
 * few positions at a time and need `-pottsSingleLogits` for those rows only.
 *
 * Built once per parameter set:
 *
 * * a neighbour list per sampled row with self and masked edges dropped, in the
 *   edge order `pottsSingleLogits` walks;
 * * `J` transposed to `[edge, s_j, a]` and cropped to `columns` states, so one
 *   neighbour's contribution is a contiguous run rather than a stride-`R` gather.
 *   This copies the sampled rows' couplings (`4 * edges * R * columns` bytes);
 *   with `transpose: false`, or when that allocation fails, rows are read from
 *   `J` in place instead — slower, same result.
 *
 * @param {PottsParams} params
 * @param {object} [options]
 * @param {ArrayLike<number>} [options.rowMask] rows that will be read; default all
 * @param {number} [options.columns] leading states to keep; default `R`
 * @param {boolean} [options.transpose]
 */
export function preparePotts(params, options = {}) {
  const { h, J, edgeIndex, edgeMask, nToken, k, numResidueTypes: R } = params;
  const { rowMask = null, columns = R, transpose = true } = options;
  const C = columns;
  const rowStart = new Int32Array(nToken + 1);
  let capacity = 0;
  for (let i = 0; i < nToken; i += 1) if (!rowMask || rowMask[i]) capacity += k;
  const edgeOf = new Int32Array(capacity);
  const neighbour = new Int32Array(capacity);
  let n = 0;
  for (let i = 0; i < nToken; i += 1) {
    if (!rowMask || rowMask[i]) {
      for (let s = 0; s < k; s += 1) {
        const edge = i * k + s;
        const j = edgeIndex[edge];
        if (j === i || !edgeMask[edge]) continue; // not_self & edge_mask
        edgeOf[n] = edge;
        neighbour[n] = j;
        n += 1;
      }
    }
    rowStart[i + 1] = n;
  }

  let Jt = null;
  if (transpose) {
    try {
      Jt = new Float32Array(n * R * C);
    } catch {
      Jt = null; // too large to copy: read J in place
    }
  }
  if (Jt) {
    const RC = R * C;
    for (let t = 0; t < n; t += 1) {
      const src = edgeOf[t] * R * R;
      const dst = t * RC;
      for (let a = 0; a < C; a += 1) {
        for (let b = 0; b < R; b += 1) Jt[dst + b * C + a] = J[src + a * R + b];
      }
    }
  }
  return { h, J, R, columns: C, rowStart, edgeOf, neighbour, Jt, rowMask };
}

// Written by the prefetch pass below so the engine cannot drop its loads.
let prefetchSink = 0;

/**
 * Conditional energies `U_i(a) = h_i(a) + sum_j J_ij(a, s_j)` of one prepared
 * row into `acc[0 .. columns)`. `acc` must be a Float64Array: the sum runs in
 * double over the edges in `pottsSingleLogits` order, so storing the result to
 * float32 gives exactly `-pottsSingleLogits` for that row.
 *
 * @param {ReturnType<typeof preparePotts>} prepared
 * @param {ArrayLike<number>} sequence restype index per token
 * @param {number} i a row inside `rowMask`
 * @param {Float64Array} acc
 */
export function pottsRowEnergies(prepared, sequence, i, acc) {
  const { h, J, R, columns: C, rowStart, edgeOf, neighbour, Jt } = prepared;
  const hBase = i * R;
  for (let a = 0; a < C; a += 1) acc[a] = h[hBase + a];
  const begin = rowStart[i];
  const end = rowStart[i + 1];
  if (Jt) {
    // Touch both ends of every run first: the loads are independent, so their
    // cache misses overlap instead of queueing behind the additions below.
    let x = 0;
    for (let t = begin; t < end; t += 1) {
      const base = (t * R + sequence[neighbour[t]]) * C;
      x += Jt[base] + Jt[base + C - 1];
    }
    prefetchSink = x;
    for (let t = begin; t < end; t += 1) {
      const base = (t * R + sequence[neighbour[t]]) * C;
      for (let a = 0; a < C; a += 1) acc[a] += Jt[base + a];
    }
  } else {
    for (let t = begin; t < end; t += 1) {
      const base = edgeOf[t] * R * R + sequence[neighbour[t]];
      for (let a = 0; a < C; a += 1) acc[a] += J[base + a * R];
    }
  }
  return acc;
}
