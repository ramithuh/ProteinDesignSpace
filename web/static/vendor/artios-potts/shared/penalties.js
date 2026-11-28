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
 * straight-through counterfactual gradient.
 *
 * Cost is `O(nToken * windowSize * 400)`; with the released defaults
 * (`windowSize = 30`) that is ~360k operations per node, so a long structure
 * makes this the dominant cost of a Gibbs sweep.
 */
export function complexityLcp(S, options) {
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

  if (complexityPenaltyWeight > 0) {
    penalties.lcp = scale(complexityPenaltyWeight, (S) =>
      complexityLcp(S, {
        ...shape,
        chainId,
        tokenMask,
        windowSize: complexityWindowSize,
        entropyMin: complexityEntropyMin,
      }));
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
  return createPenaltyCollection(penalties);
}

function andMask(a, b) {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] && b[i] ? 1 : 0;
  return out;
}
