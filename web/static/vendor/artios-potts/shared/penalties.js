/**
 * Steering penalties for Potts sampling — the JS mirror of
 * `artios.model.modules.penalty`.
 *
 * Each penalty returns `{ value, grad }` where `value` is the scalar energy for
 * the whole structure and `grad` is `dvalue/dS` as a `[nToken, R]` array. The
 * reference gets that gradient from autograd; here it is written out by hand:
 *
 * * `complexityLcp` reproduces the reference's straight-through estimator
 *   exactly — the gradient is the discrete counterfactual `sum_w U_ij[i, w, a]`,
 *   i.e. "what would this window's penalty be if position i were amino acid a",
 *   which is what `U.detach() + U_diff - U_diff.detach()` differentiates to.
 * * every other penalty is a smooth function of `S`, so its gradient is the
 *   ordinary analytic one, matching what autograd computes there.
 *
 * `S` is a one-hot (or soft) sequence `[nToken, R]`, flat and row major.
 *
 * `complexityLcp` is by far the most expensive of these, so for a one-hot `S` it
 * runs on integer window counts (`createLcpStatic` / `createLcpState`) and
 * returns bit-identical results to the brute-force `complexityLcpLegacy`.
 */

import {
  AA_FREQUENCIES,
  AROMATIC_INDICES,
  CHARGED_INDICES,
  NEG_CHARGE_INDICES,
  NUM_STANDARD_AA,
  POS_CHARGE_INDICES,
} from './constants.js';

const EPS = 1e-8;

/** `build_natural_log_pi`: log of an organism's AA marginal, clamped at 1e-8. */
export function buildNaturalLogPi(organism = 'swissprot') {
  const table = AA_FREQUENCIES[organism];
  if (!table) {
    throw new Error(
      `unknown organism ${JSON.stringify(organism)}; expected one of ` +
      `${Object.keys(AA_FREQUENCIES).join(', ')}`,
    );
  }
  return Float32Array.from(table.mean, (p) => Math.log(Math.max(p, 1e-8)));
}

/** Sliding-window neighbour indices and validity, shared by both LCP penalties. */
function windowGeometry(nToken, chainId, tokenMask, windowSize) {
  const w = Math.min(windowSize, nToken);
  const halfW = Math.floor(w / 2);
  const index = new Int32Array(nToken * w);
  const valid = new Uint8Array(nToken * w);
  for (let i = 0; i < nToken; i += 1) {
    for (let o = 0; o < w; o += 1) {
      const raw = i + o - halfW;
      const inBounds = raw >= 0 && raw < nToken;
      const j = inBounds ? raw : Math.min(Math.max(raw, 0), nToken - 1);
      index[i * w + o] = j;
      valid[i * w + o] =
        inBounds && chainId[i] === chainId[j] && tokenMask[j] && tokenMask[i] ? 1 : 0;
    }
  }
  return { w, index, valid };
}

/**
 * `complexity_lcp` — local composition perplexity floor with the reference's
 * straight-through counterfactual gradient, written as the reference writes it:
 * every (window, position, letter) counterfactual is brute-forced.
 *
 * Cost is `O(nToken * windowSize * 400)` with a `Math.log` per term. It is kept
 * as the oracle for `complexityLcp` (pass `legacyPath: true` to reach it) and as
 * the path for a soft, non-one-hot `S`.
 */
export function complexityLcpLegacy(S, options) {
  const {
    nToken,
    numResidueTypes: R,
    chainId,
    tokenMask,
    windowSize = 30,
    entropyMin = 2.32,
    minCoverage = 0.9,
    differentiable = true,
  } = options;
  const Q = NUM_STANDARD_AA;
  const target = Math.exp(entropyMin);
  const { w, index, valid } = windowGeometry(nToken, chainId, tokenMask, windowSize);
  const coverageThreshold = Math.trunc(minCoverage * w);

  // Local composition counts per window centre: [nToken, Q].
  const counts = new Float64Array(nToken * Q);
  const totals = new Float64Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    for (let o = 0; o < w; o += 1) {
      if (!valid[i * w + o]) continue;
      const j = index[i * w + o];
      if (!tokenMask[j]) continue; // S_aa_masked zeroes invalid positions
      for (let q = 0; q < Q; q += 1) counts[i * Q + q] += S[j * R + q];
    }
    let total = 0;
    for (let q = 0; q < Q; q += 1) total += counts[i * Q + q];
    totals[i] = total;
  }

  const hasCoverage = new Uint8Array(nToken);
  const isValidCentre = new Uint8Array(nToken);
  const U = new Float64Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    hasCoverage[i] = totals[i] > coverageThreshold ? 1 : 0;
    isValidCentre[i] = totals[i] > 0 && tokenMask[i] ? 1 : 0;
    let entropy = 0;
    for (let q = 0; q < Q; q += 1) {
      const p = counts[i * Q + q] / (totals[i] + EPS);
      entropy -= p * Math.log(p + EPS);
    }
    const deficit = Math.min(Math.exp(entropy) - target, 0);
    U[i] = hasCoverage[i] * deficit * deficit;
  }

  let value = 0;
  for (let i = 0; i < nToken; i += 1) value += isValidCentre[i] * U[i];

  const grad = new Float32Array(nToken * R);
  if (!differentiable) return { value, grad };

  // Straight-through: grad[i, a] = mask_i_valid[i] * sum_w U_ij[i, w, a], where
  // U_ij is the penalty of neighbour window w with position i mutated to a.
  const counterfactual = new Float64Array(Q);
  for (let i = 0; i < nToken; i += 1) {
    if (!isValidCentre[i]) continue;
    for (let o = 0; o < w; o += 1) {
      if (!valid[i * w + o]) continue;
      const j = index[i * w + o];
      if (!hasCoverage[j]) continue;
      // counts of window j with position i's own contribution removed
      for (let q = 0; q < Q; q += 1) {
        counterfactual[q] = counts[j * Q + q] - S[i * R + q];
      }
      for (let a = 0; a < Q; a += 1) {
        let total = 0;
        for (let q = 0; q < Q; q += 1) {
          total += Math.max(counterfactual[q] + (q === a ? 1 : 0), 0);
        }
        let entropy = 0;
        for (let q = 0; q < Q; q += 1) {
          const n = Math.max(counterfactual[q] + (q === a ? 1 : 0), 0);
          const p = n / (total + EPS);
          entropy -= p * Math.log(p + EPS);
        }
        const deficit = Math.min(Math.exp(entropy) - target, 0);
        grad[i * R + a] += deficit * deficit;
      }
    }
  }
  return { value, grad };
}

/**
 * Token index per row of a one-hot `S`, or `null` when any row is not exactly
 * one-hot (soft, empty or multi-hot) and the count-based LCP does not apply.
 */
export function oneHotSequence(S, nToken, numResidueTypes) {
  const R = numResidueTypes;
  if (S.length !== nToken * R) return null;
  const sequence = new Int32Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    let hot = -1;
    for (let a = 0; a < R; a += 1) {
      const v = S[i * R + a];
      if (v === 1) {
        if (hot >= 0) return null;
        hot = a;
      } else if (v !== 0) return null;
    }
    if (hot < 0) return null;
    sequence[i] = hot;
  }
  return sequence;
}

/**
 * Sequence-independent tables of the count-based LCP: window membership in both
 * directions, the entropy-term table and the per-total "provably zero" bound.
 *
 * `lut[tot * M + c]` holds `p * log(p + EPS)` with `p = c / (tot + EPS)`, i.e. the
 * exact expression `complexityLcpLegacy` evaluates, so a table lookup reproduces
 * its terms bit for bit. (A 1-D `c * log(c)` table would not.)
 */
export function createLcpStatic(options) {
  const {
    nToken: N,
    chainId,
    tokenMask,
    windowSize = 30,
    entropyMin = 2.32,
    minCoverage = 0.9,
  } = options;
  const Q = NUM_STANDARD_AA;
  const w = Math.min(windowSize, N);
  const halfW = Math.floor(w / 2);
  // Members of window i, ascending offset — the order the legacy loops use.
  const start = new Int32Array(N + 1);
  const members = new Int32Array(N * w);
  let n = 0;
  for (let i = 0; i < N; i += 1) {
    for (let o = 0; o < w; o += 1) {
      const raw = i + o - halfW;
      if (raw >= 0 && raw < N && chainId[i] === chainId[raw] && tokenMask[raw] && tokenMask[i]) {
        members[n] = raw;
        n += 1;
      }
    }
    start[i + 1] = n;
  }
  // Inverse: the windows whose member list contains position p.
  const invStart = new Int32Array(N + 1);
  for (let t = 0; t < n; t += 1) invStart[members[t] + 1] += 1;
  for (let p = 0; p < N; p += 1) invStart[p + 1] += invStart[p];
  const invList = new Int32Array(n);
  const cursor = invStart.slice(0, N);
  for (let j = 0; j < N; j += 1) {
    for (let t = start[j]; t < start[j + 1]; t += 1) {
      invList[cursor[members[t]]] = j;
      cursor[members[t]] += 1;
    }
  }

  const M = w + 2;
  const lut = new Float64Array(M * M);
  for (let tot = 0; tot < M; tot += 1) {
    for (let c = 0; c < M; c += 1) {
      const p = c / (tot + EPS);
      lut[tot * M + c] = p * Math.log(p + EPS);
    }
  }
  // zeroBound[tot]: a window whose entropy exceeds this keeps entropy > entropyMin
  // after removing any present residue and adding any residue, so every
  // counterfactual deficit is exactly 0. Brute-forced from the table.
  const zeroBound = new Float64Array(M);
  for (let tot = 0; tot < M; tot += 1) {
    let dRemove = 0;
    let dAdd = 0;
    for (let c = 1; c <= tot; c += 1) {
      dRemove = Math.max(dRemove, lut[tot * M + c - 1] - lut[tot * M + c]);
    }
    for (let c = 0; c < tot; c += 1) {
      dAdd = Math.min(dAdd, lut[tot * M + c] - lut[tot * M + c + 1]);
    }
    zeroBound[tot] = entropyMin + 1e-7 + dRemove - dAdd;
  }
  const validToken = new Uint8Array(N);
  for (let i = 0; i < N; i += 1) validToken[i] = tokenMask[i] ? 1 : 0;

  return {
    N,
    Q,
    w,
    M,
    lut,
    zeroBound,
    coverageThreshold: Math.trunc(minCoverage * w),
    target: Math.exp(entropyMin),
    // Entries whose estimated entropy clears this are exactly 0 without a fold.
    foldThreshold: entropyMin + 1e-9,
    start,
    members: members.subarray(0, n),
    invStart,
    invList,
    validToken,
  };
}

/**
 * Mutable LCP state for one sequence: integer window counts kept up to date by
 * `flip`, plus lazily cached counterfactual rows.
 *
 * `D[j][b][a]` is the penalty of window j with one `b` removed and one `a` added.
 * It depends on the window's counts only, not on which member is mutated, so it
 * is cached per (window, removed letter) under a version stamp and shared by all
 * the window's members: `grad[i][a] = sum_{j covering i} D[j][seq[i]][a]`.
 *
 * Every shortcut skips terms that are provably exactly zero, and whatever is
 * computed is summed in the legacy order, so `rowGrad` and `value` return the
 * same bits as `complexityLcpLegacy` on the matching one-hot `S`.
 *
 * @param {ReturnType<typeof createLcpStatic>} tables
 * @param {ArrayLike<number> | null} [liveLetters] letters whose gradient entry
 *   the caller consumes; the rest are left at 0. Defaults to all 20.
 */
export function createLcpState(tables, liveLetters = null) {
  const {
    N, Q, M, lut, zeroBound, coverageThreshold, target, foldThreshold,
    start, members, invStart, invList, validToken,
  } = tables;
  const QB = Q + 1; // removed-letter slots: 20 amino acids + "not an amino acid"
  const letters = [];
  for (let a = 0; a < Q; a += 1) if (!liveLetters || liveLetters[a]) letters.push(a);
  const nLetters = letters.length;

  const counts = new Int32Array(N * Q);
  const totals = new Int32Array(N);
  // Window entropy, updated in O(1) per flip. A FILTER only: it drifts by
  // rounding, which the 1e-7 margin in zeroBound and the periodic refold cover.
  const entropyEstimate = new Float64Array(N);
  let flipsSinceRefold = 0;
  const windowStamp = new Int32Array(N).fill(1);
  const rowStamp = new Int32Array(N * QB);
  const rowIsZero = new Uint8Array(N * QB);
  const D = new Float64Array(N * QB * Q);
  const prefix = new Float64Array(Q + 1);
  const cf = new Int32Array(Q);
  const nonZero = new Int32Array(Q);

  function foldEntropy(j) {
    const row = totals[j] * M;
    let e = 0;
    for (let q = 0; q < Q; q += 1) e -= lut[row + counts[j * Q + q]];
    entropyEstimate[j] = e;
  }

  function refold() {
    for (let j = 0; j < N; j += 1) foldEntropy(j);
    flipsSinceRefold = 0;
  }

  /** Reset to `sequence` (restype index per token). */
  function init(sequence) {
    counts.fill(0);
    totals.fill(0);
    rowStamp.fill(0);
    windowStamp.fill(1);
    for (let j = 0; j < N; j += 1) {
      for (let t = start[j]; t < start[j + 1]; t += 1) {
        const s = sequence[members[t]];
        if (s < Q) {
          counts[j * Q + s] += 1;
          totals[j] += 1;
        }
      }
    }
    refold();
  }

  /** Position p changed from `oldState` to `newState`. */
  function flip(p, oldState, newState) {
    for (let t = invStart[p]; t < invStart[p + 1]; t += 1) {
      const j = invList[t];
      if (oldState < Q && newState < Q) {
        const row = totals[j] * M;
        const co = counts[j * Q + oldState];
        const cn = counts[j * Q + newState];
        entropyEstimate[j] += (lut[row + co] - lut[row + co - 1]) + (lut[row + cn] - lut[row + cn + 1]);
        counts[j * Q + oldState] = co - 1;
        counts[j * Q + newState] = cn + 1;
      } else {
        if (oldState < Q) {
          counts[j * Q + oldState] -= 1;
          totals[j] -= 1;
        }
        if (newState < Q) {
          counts[j * Q + newState] += 1;
          totals[j] += 1;
        }
        foldEntropy(j);
      }
      windowStamp[j] += 1;
    }
    flipsSinceRefold += 1;
    if (flipsSinceRefold >= 65536) refold();
  }

  function computeRow(j, b, key) {
    const c0 = j * Q;
    for (let q = 0; q < Q; q += 1) cf[q] = counts[c0 + q];
    // `clamped`: the removed letter is absent from window j, where the legacy
    // max(count - 1 + [q === a], 0) leaves the counts unchanged for a === b.
    let clamped = -1;
    let tot = totals[j] + 1;
    if (b < Q) {
      if (cf[b] > 0) {
        cf[b] -= 1;
        tot -= 1;
      } else clamped = b;
    }
    const row = tot * M;
    // Fold over non-zero counts only: a zero count contributes `e -= -0`, an
    // exact no-op, so every partial sum equals the legacy q = 0..19 fold.
    let nnz = 0;
    for (let q = 0; q < Q; q += 1) {
      if (cf[q] > 0) {
        nonZero[nnz] = q;
        nnz += 1;
      }
    }
    let e = 0;
    for (let t = 0; t < nnz; t += 1) {
      prefix[t] = e;
      e -= lut[row + cf[nonZero[t]]];
    }
    prefix[nnz] = e;

    const off = key * Q;
    let any = 0;
    let pos = 0;
    for (let u = 0; u < nLetters; u += 1) {
      const a = letters[u];
      while (pos < nnz && nonZero[pos] < a) pos += 1;
      const na = cf[a];
      // O(1) estimate (error ~1e-14), used only to prove the deficit is 0.
      const estimate = e + lut[row + na] - lut[row + na + 1];
      let val = 0;
      if (!(estimate > foldThreshold)) {
        let x = prefix[pos] - lut[row + na + 1]; // exact legacy summation order
        for (let t = na > 0 ? pos + 1 : pos; t < nnz; t += 1) x -= lut[row + cf[nonZero[t]]];
        const deficit = Math.min(Math.exp(x) - target, 0);
        val = deficit * deficit;
      }
      D[off + a] = val;
      if (val !== 0) any = 1;
    }
    if (clamped >= 0 && (!liveLetters || liveLetters[clamped])) {
      const row2 = totals[j] * M;
      let x = 0;
      for (let q = 0; q < Q; q += 1) x -= lut[row2 + counts[c0 + q]];
      const deficit = Math.min(Math.exp(x) - target, 0);
      D[off + clamped] = deficit * deficit;
      any = 0;
      for (let a = 0; a < Q; a += 1) if (D[off + a] !== 0) any = 1;
    }
    rowIsZero[key] = any ? 0 : 1;
    rowStamp[key] = windowStamp[j];
  }

  /**
   * Unscaled gradient row of position i (holding `state`) into
   * `out[offset .. offset + 20)`. `out` must be a Float32Array: the legacy code
   * accumulates into one, rounding after every window.
   */
  function rowGrad(i, state, out, offset = 0) {
    for (let a = 0; a < Q; a += 1) out[offset + a] = 0;
    if (!(totals[i] > 0 && validToken[i])) return;
    const b = state < Q ? state : Q;
    for (let t = start[i]; t < start[i + 1]; t += 1) {
      const j = members[t];
      if (!(totals[j] > coverageThreshold)) continue;
      if (b < Q && counts[j * Q + b] > 0 && entropyEstimate[j] > zeroBound[totals[j]]) continue;
      const key = j * QB + b;
      if (rowStamp[key] !== windowStamp[j]) computeRow(j, b, key);
      if (rowIsZero[key]) continue;
      const off = key * Q;
      for (let a = 0; a < Q; a += 1) out[offset + a] += D[off + a];
    }
  }

  /** Scalar penalty, summed in the legacy order. */
  function value() {
    let v = 0;
    for (let i = 0; i < N; i += 1) {
      const row = totals[i] * M;
      let entropy = 0;
      for (let q = 0; q < Q; q += 1) entropy -= lut[row + counts[i * Q + q]];
      const deficit = Math.min(Math.exp(entropy) - target, 0);
      const U = (totals[i] > coverageThreshold ? 1 : 0) * deficit * deficit;
      v += (totals[i] > 0 && validToken[i] ? 1 : 0) * U;
    }
    return v;
  }

  return { init, flip, rowGrad, value };
}

/**
 * A reusable `complexityLcp` evaluator: the window tables are built once and
 * every call costs integer counting and table lookups, with no `Math.log`.
 *
 * @returns {(S: Float32Array) => {value: number, grad: Float32Array}}
 */
export function createComplexityLcp(options) {
  const { nToken, numResidueTypes: R, differentiable = true, legacyPath = false } = options;
  const Q = NUM_STANDARD_AA;
  if (legacyPath || R < Q) return (S) => complexityLcpLegacy(S, options);
  let state = null;
  return (S) => {
    const sequence = oneHotSequence(S, nToken, R);
    if (!sequence) return complexityLcpLegacy(S, options);
    state ??= createLcpState(createLcpStatic(options));
    state.init(sequence);
    const grad = new Float32Array(nToken * R);
    if (differentiable) {
      for (let i = 0; i < nToken; i += 1) state.rowGrad(i, sequence[i], grad, i * R);
    }
    return { value: state.value(), grad };
  };
}

/**
 * `complexity_lcp` — local composition perplexity floor with the reference's
 * straight-through counterfactual gradient.
 *
 * For a one-hot `S` this runs on integer window counts and returns the same bits
 * as `complexityLcpLegacy`; a soft `S`, or `legacyPath: true`, takes the
 * brute-force path. Callers evaluating many states should hold on to a
 * `createComplexityLcp` evaluator instead, which keeps the window tables.
 */
export function complexityLcp(S, options) {
  return createComplexityLcp(options)(S);
}

/**
 * `complexity_lcp_natural_prior` — cross-entropy perplexity floor against a
 * natural AA marginal. Linear in `S`, so the gradient is exact.
 */
export function complexityLcpNaturalPrior(S, options) {
  const {
    nToken,
    numResidueTypes: R,
    logPi,
    chainId,
    tokenMask,
    windowSize = 30,
    natPerpMin = 16.1,
    minCoverage = 0.9,
  } = options;
  const Q = NUM_STANDARD_AA;
  const { w, index, valid } = windowGeometry(nToken, chainId, tokenMask, windowSize);
  const coverageThreshold = Math.trunc(minCoverage * w);

  const surprisal = new Float64Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    let s = 0;
    for (let q = 0; q < Q; q += 1) s -= S[i * R + q] * logPi[q];
    surprisal[i] = s * (tokenMask[i] ? 1 : 0);
  }

  const windowSum = new Float64Array(nToken);
  const windowCount = new Float64Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    let sum = 0;
    let count = 0;
    for (let o = 0; o < w; o += 1) {
      if (!valid[i * w + o]) continue;
      sum += surprisal[index[i * w + o]];
      count += 1;
    }
    windowSum[i] = sum;
    windowCount[i] = count;
  }

  let value = 0;
  // dvalue/dsurprisal[i], folded into the per-centre coefficient.
  const coefficient = new Float64Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    const hasCoverage = windowCount[i] > coverageThreshold;
    const isValid = windowCount[i] > 0 && tokenMask[i] && hasCoverage;
    if (!isValid) continue;
    const ce = windowSum[i] / (windowCount[i] + EPS);
    const perplexity = Math.exp(ce);
    const excess = Math.max(natPerpMin - perplexity, 0);
    value += excess * excess;
    if (excess > 0) {
      // d/dce of (natPerpMin - exp(ce))^2 = -2 * excess * exp(ce)
      coefficient[i] = (-2 * excess * perplexity) / (windowCount[i] + EPS);
    }
  }

  const grad = new Float32Array(nToken * R);
  for (let i = 0; i < nToken; i += 1) {
    if (coefficient[i] === 0) continue;
    for (let o = 0; o < w; o += 1) {
      if (!valid[i * w + o]) continue;
      const j = index[i * w + o];
      if (!tokenMask[j]) continue;
      for (let q = 0; q < Q; q += 1) grad[j * R + q] += coefficient[i] * -logPi[q];
    }
  }
  return { value, grad };
}

/** Shared body of the aromatic and charged active-site enrichment penalties. */
function enrichmentPenalty(S, { nToken, numResidueTypes: R, mask, indices, targetFraction }) {
  let share = 0;
  let nActive = 0;
  for (let i = 0; i < nToken; i += 1) {
    if (!mask[i]) continue;
    nActive += 1;
    for (const a of indices) share += S[i * R + a];
  }
  const grad = new Float32Array(nToken * R);
  if (nActive === 0) return { value: 0, grad };

  const fraction = share / (nActive + EPS);
  const deficit = Math.min(fraction - targetFraction, 0);
  const value = deficit * deficit;
  if (deficit < 0) {
    const d = (2 * deficit) / (nActive + EPS);
    for (let i = 0; i < nToken; i += 1) {
      if (!mask[i]) continue;
      for (const a of indices) grad[i * R + a] += d;
    }
  }
  return { value, grad };
}

/** `aromatic_enrichment_penalty` — one-sided quadratic on PHE/TRP/TYR share. */
export function aromaticEnrichmentPenalty(S, options) {
  return enrichmentPenalty(S, {
    ...options,
    mask: options.activeSiteMask,
    indices: AROMATIC_INDICES,
    targetFraction: options.targetFraction ?? 0.25,
  });
}

/** `charged_enrichment_penalty` — the same on ASP/GLU/ARG/LYS/HIS. */
export function chargedEnrichmentPenalty(S, options) {
  return enrichmentPenalty(S, {
    ...options,
    mask: options.activeSiteMask,
    indices: CHARGED_INDICES,
    targetFraction: options.targetFraction ?? 0.25,
  });
}

/** `net_charge_penalty` — symmetric quadratic on per-residue net charge. */
export function netChargePenalty(S, options) {
  const { nToken, numResidueTypes: R, tokenMask, targetPerResidue = -0.013 } = options;
  let pos = 0;
  let neg = 0;
  let nResidues = 0;
  for (let i = 0; i < nToken; i += 1) {
    if (!tokenMask[i]) continue;
    nResidues += 1;
    for (const a of POS_CHARGE_INDICES) pos += S[i * R + a];
    for (const a of NEG_CHARGE_INDICES) neg += S[i * R + a];
  }
  const grad = new Float32Array(nToken * R);
  if (nResidues === 0) return { value: 0, grad };

  const netPerResidue = (pos - neg) / (nResidues + EPS);
  const delta = netPerResidue - targetPerResidue;
  const d = (2 * delta) / (nResidues + EPS);
  for (let i = 0; i < nToken; i += 1) {
    if (!tokenMask[i]) continue;
    for (const a of POS_CHARGE_INDICES) grad[i * R + a] += d;
    for (const a of NEG_CHARGE_INDICES) grad[i * R + a] -= d;
  }
  return { value: delta * delta, grad };
}

/**
 * `composition_chi_squared_penalty` — dead-zone chi-squared against an organism's
 * AA frequencies, in units of excess z-scores squared.
 */
export function compositionChiSquaredPenalty(S, options) {
  const {
    nToken,
    numResidueTypes: R,
    proteinMutableMask,
    organism = 'swissprot',
    toleranceK = 1.0,
  } = options;
  const table = AA_FREQUENCIES[organism];
  if (!table) throw new Error(`unknown organism ${JSON.stringify(organism)}`);
  const Q = NUM_STANDARD_AA;

  const counts = new Float64Array(Q);
  let nEff = 0;
  for (let i = 0; i < nToken; i += 1) {
    if (!proteinMutableMask[i]) continue;
    nEff += 1;
    for (let q = 0; q < Q; q += 1) counts[q] += S[i * R + q];
  }
  const grad = new Float32Array(nToken * R);
  if (nEff === 0) return { value: 0, grad };

  let value = 0;
  const dByAa = new Float64Array(Q);
  for (let q = 0; q < Q; q += 1) {
    const p = counts[q] / (nEff + EPS);
    const mu = table.mean[q];
    const sigma = table.std[q];
    const excess = Math.max(Math.abs(p - mu) - toleranceK * sigma, 0);
    const scale = sigma * sigma + EPS;
    value += (excess * excess) / scale;
    if (excess > 0) {
      const sign = p - mu >= 0 ? 1 : -1;
      dByAa[q] = ((2 * excess * sign) / scale) / (nEff + EPS);
    }
  }
  for (let i = 0; i < nToken; i += 1) {
    if (!proteinMutableMask[i]) continue;
    for (let q = 0; q < Q; q += 1) grad[i * R + q] = dByAa[q];
  }
  return { value, grad };
}

/**
 * A named collection of penalties, as `PenaltyCollection`: calling it sums every
 * value and every gradient.
 *
 * @param {Record<string, (S: Float32Array) => {value: number, grad: Float32Array}>} penalties
 */
export function createPenaltyCollection(penalties) {
  const names = Object.keys(penalties);
  if (names.length === 0) return null;
  return {
    names,
    penalties,
    evaluate(S) {
      let value = 0;
      let grad = null;
      const byName = {};
      for (const name of names) {
        const result = penalties[name](S);
        byName[name] = result;
        value += result.value;
        if (grad === null) grad = Float32Array.from(result.grad);
        else for (let i = 0; i < grad.length; i += 1) grad[i] += result.grad[i];
      }
      return { value, grad, byName };
    },
  };
}

/**
 * Build the penalty collection the released sampling config asks for.
 *
 * @param {object} options weights and shapes, named as in the sampling config
 * @returns {ReturnType<typeof createPenaltyCollection>}
 */
export function buildPenalties(options = {}) {
  const {
    nToken,
    numResidueTypes,
    chainId,
    tokenMask,
    mutableMask,
    isProteinToken,
    activeSiteMask,
    complexityPenaltyWeight = 0,
    complexityWindowSize = 30,
    complexityEntropyMin = 2.32,
    complexityNatPriorPenaltyWeight = 0,
    complexityNatPriorWindowSize = 30,
    complexityNatPriorPerpMin = 16.1,
    complexityNatPriorOrganism = 'swissprot',
    aromaticPenaltyWeight = 0,
    aromaticTargetFraction = 0.25,
    chargedPenaltyWeight = 0,
    chargedTargetFraction = 0.25,
    compositionPenaltyWeight = 0,
    compositionOrganism = 'swissprot',
    compositionToleranceK = 1.0,
    netChargePenaltyWeight = 0,
    netChargeTargetPerResidue = -0.013,
    legacyPath = false,
  } = options;

  const scale = (weight, fn) => (S) => {
    const { value, grad } = fn(S);
    if (weight === 1) return { value, grad };
    const scaled = new Float32Array(grad.length);
    for (let i = 0; i < grad.length; i += 1) scaled[i] = grad[i] * weight;
    return { value: value * weight, grad: scaled };
  };
  const shape = { nToken, numResidueTypes };
  const penalties = {};

  let lcpOptions = null;
  if (complexityPenaltyWeight > 0) {
    lcpOptions = {
      ...shape,
      chainId,
      tokenMask,
      windowSize: complexityWindowSize,
      entropyMin: complexityEntropyMin,
    };
    penalties.lcp = scale(complexityPenaltyWeight, createComplexityLcp({ ...lcpOptions, legacyPath }));
  }
  if (complexityNatPriorPenaltyWeight > 0) {
    const logPi = buildNaturalLogPi(complexityNatPriorOrganism);
    penalties.nat_prior_lcp = scale(complexityNatPriorPenaltyWeight, (S) =>
      complexityLcpNaturalPrior(S, {
        ...shape,
        logPi,
        chainId,
        tokenMask,
        windowSize: complexityNatPriorWindowSize,
        natPerpMin: complexityNatPriorPerpMin,
      }));
  }
  if (aromaticPenaltyWeight > 0 && activeSiteMask) {
    penalties.aromatic = scale(aromaticPenaltyWeight, (S) =>
      aromaticEnrichmentPenalty(S, {
        ...shape,
        activeSiteMask,
        targetFraction: aromaticTargetFraction,
      }));
  }
  if (chargedPenaltyWeight > 0 && activeSiteMask) {
    penalties.charged = scale(chargedPenaltyWeight, (S) =>
      chargedEnrichmentPenalty(S, {
        ...shape,
        activeSiteMask,
        targetFraction: chargedTargetFraction,
      }));
  }
  if (compositionPenaltyWeight > 0) {
    const proteinMutableMask = andMask(mutableMask, isProteinToken);
    penalties.composition = scale(compositionPenaltyWeight, (S) =>
      compositionChiSquaredPenalty(S, {
        ...shape,
        proteinMutableMask,
        organism: compositionOrganism,
        toleranceK: compositionToleranceK,
      }));
  }
  if (netChargePenaltyWeight > 0) {
    const chargeMask = andMask(mutableMask, isProteinToken);
    penalties.net_charge = scale(netChargePenaltyWeight, (S) =>
      netChargePenalty(S, {
        ...shape,
        tokenMask: chargeMask,
        targetPerResidue: netChargeTargetPerResidue,
      }));
  }
  const collection = createPenaltyCollection(penalties);
  // When LCP is the only penalty the Gibbs sampler can evaluate it row by row
  // from incremental window counts; this is the recipe it needs for that.
  if (collection && lcpOptions && collection.names.length === 1) {
    collection.lcpOnly = { ...lcpOptions, weight: complexityPenaltyWeight };
  }
  return collection;
}

function andMask(a, b) {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] && b[i] ? 1 : 0;
  return out;
}
