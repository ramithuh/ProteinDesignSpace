/**
 * Steerable MCMC sampling from Potts parameters — the JS mirror of
 * `artios.model.generator.PottsSampler` plus the mask/bias/penalty pipeline of
 * `ArtiosPotts.sample_from_params`.
 *
 * Both proposals of the reference are here:
 *
 * * `gibbs` — parallel colour sweeps over an independent set at a time, the
 *   released inference default.
 * * `dlmc` — discrete Langevin Monte Carlo with an optional Metropolis-Hastings
 *   correction, which is what the training config used.
 *
 * Steering knobs all behave as in the reference: annealed `temperature`, fixed
 * positions, an excluded alphabet (C/M), classifier-free guidance against a
 * ligand-free `h`/`J`, per-residue and per-group energy biases, and penalty
 * gradients added as a bias on the per-position energies.
 *
 * A Gibbs sweep only needs the conditional of the positions it is about to
 * resample, so by default it is evaluated row by row (`prepareGibbs`): Potts
 * energies through a per-row neighbour list, the complexity penalty from
 * incrementally maintained window counts. That is bit-identical to the
 * reference-shaped sweep — which evaluates all `nToken` rows for every colour
 * group and stays reachable with `legacyPath: true` — and is used whenever the
 * state is one-hot and no penalty other than `complexityLcp` is active.
 *
 * Random numbers come from a seeded 32-bit generator, so a given `seed` is
 * reproducible — but it is *not* PyTorch's stream, so sampled sequences do not
 * match a PyTorch run element for element even though the distribution does.
 */

import { ALPHABET_BIAS, NUM_STANDARD_AA } from './constants.js';
import {
  applyFieldBias,
  buildAlphabetMask,
  buildEnergyBiasVector,
  buildGroupEnergyBiasVector,
  buildMutableMask,
  pottsEnergyPerNode,
  pottsLiveStates,
  pottsRowEnergies,
  pottsSingleLogits,
  preparePotts,
} from './potts.js';
import { buildLigandProximityMask } from './featurize.js';
import { buildPenalties, createLcpState, createLcpStatic, oneHotSequence } from './penalties.js';

/** Seeded `sfc32` generator; `rng()` yields a float in [0, 1). */
export function createRng(seed = 0) {
  let a = 0x9e3779b9;
  let b = seed >>> 0 || 0x243f6a88;
  let c = 0xb7e15162;
  let d = 1;
  const next = () => {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 12; i += 1) next(); // warm up the state
  return next;
}

/** Sample one index from unnormalised non-negative weights. */
function sampleCategorical(weights, offset, count, rng) {
  let total = 0;
  for (let a = 0; a < count; a += 1) total += weights[offset + a];
  if (!(total > 0)) {
    // Degenerate row (every state forbidden): fall back to the arg-max, as a
    // uniform draw would pick a state the mask forbids.
    let best = 0;
    for (let a = 1; a < count; a += 1) {
      if (weights[offset + a] > weights[offset + best]) best = a;
    }
    return best;
  }
  const threshold = rng() * total;
  let cumulative = 0;
  for (let a = 0; a < count; a += 1) {
    cumulative += weights[offset + a];
    if (cumulative > threshold) return a;
  }
  return count - 1;
}

/** In-place `softmax(-energy / temperature)` over one row. */
function softmaxNegInto(out, energy, offset, count, temperature) {
  let max = -Infinity;
  for (let a = 0; a < count; a += 1) {
    const v = -energy[offset + a] / temperature;
    out[a] = v;
    if (v > max) max = v;
  }
  let sum = 0;
  for (let a = 0; a < count; a += 1) {
    out[a] = Math.exp(out[a] - max);
    sum += out[a];
  }
  for (let a = 0; a < count; a += 1) out[a] /= sum;
  return out;
}

/**
 * Greedy colouring for parallel Gibbs updates — `_greedy_coloring`.
 *
 * A node takes the lowest colour not used by an already-coloured neighbour in
 * its own list. Because the kNN graph is directed, a pair listed in one
 * direction only can still share a colour, so a sweep is approximately, not
 * exactly, Gibbs — same caveat as the reference.
 */
export function greedyColoring(edgeIndex, edgeMask, nToken, k, reserved = []) {
  const colors = new Int32Array(nToken).fill(-1);
  reserved.forEach((node, c) => { colors[node] = c; });
  const taken = new Set();
  for (let i = 0; i < nToken; i += 1) {
    if (colors[i] >= 0) continue;
    taken.clear();
    for (let s = 0; s < k; s += 1) {
      const j = edgeIndex[i * k + s];
      if (edgeMask[i * k + s] && j !== i) taken.add(colors[j]);
    }
    let c = 0;
    while (taken.has(c)) c += 1;
    colors[i] = c;
  }
  return colors;
}

/** Colour-group masks, one per colour — `_color_graph`. */
export function colorGroups(params, { activeSiteMask = null, granular = false } = {}) {
  const { edgeIndex, edgeMask, nToken, k } = params;
  const reserved = [];
  if (granular && activeSiteMask) {
    for (let i = 0; i < nToken; i += 1) if (activeSiteMask[i]) reserved.push(i);
  }
  const colors = greedyColoring(edgeIndex, edgeMask, nToken, k, reserved);
  let numColors = 0;
  for (let i = 0; i < nToken; i += 1) numColors = Math.max(numColors, colors[i] + 1);
  const groups = [];
  for (let c = 0; c < numColors; c += 1) {
    const mask = new Uint8Array(nToken);
    for (let i = 0; i < nToken; i += 1) mask[i] = colors[i] === c ? 1 : 0;
    groups.push(mask);
  }
  return groups;
}

function oneHotFromIndices(indices, numResidueTypes) {
  const out = new Float32Array(indices.length * numResidueTypes);
  for (let i = 0; i < indices.length; i += 1) out[i * numResidueTypes + indices[i]] = 1;
  return out;
}

function argmaxRows(S, nToken, R, out) {
  const indices = out ?? new Int32Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    let best = 0;
    for (let a = 1; a < R; a += 1) if (S[i * R + a] > S[i * R + best]) best = a;
    indices[i] = best;
  }
  return indices;
}

/**
 * Per-position conditional energies `U_i(a)`, optionally blended with an
 * unconditional pathway via classifier-free guidance in probability space.
 *
 * Returns the probabilities directly because the CFG blend is defined on
 * probabilities, not energies (`p_cfg = p_uncond + scale * (p_cond - p_uncond)`,
 * clamped at zero and renormalised).
 */
function conditionalProbs(params, sequence, temperature, options) {
  const {
    penaltyGrad = null,
    penaltyGradScale = 0.5,
    uncond = null,
    cfgScale = 1.0,
    activeSiteMask = null,
    activeSiteOnlyCfg = false,
  } = options;
  const { nToken, numResidueTypes: R } = params;

  const logits = pottsSingleLogits(params, sequence);
  const energy = logits; // reuse the buffer: U_i = -logits
  for (let i = 0; i < energy.length; i += 1) energy[i] = -energy[i];
  if (penaltyGrad) {
    for (let i = 0; i < energy.length; i += 1) energy[i] += penaltyGradScale * penaltyGrad[i];
  }

  const probs = new Float32Array(nToken * R);
  const row = new Float64Array(R);
  for (let i = 0; i < nToken; i += 1) {
    softmaxNegInto(row, energy, i * R, R, temperature);
    for (let a = 0; a < R; a += 1) probs[i * R + a] = row[a];
  }
  const useCfg = uncond && cfgScale !== 1.0;
  if (!useCfg) return probs;

  const uncondLogits = pottsSingleLogits(uncond, sequence);
  for (let i = 0; i < uncondLogits.length; i += 1) uncondLogits[i] = -uncondLogits[i];
  if (penaltyGrad) {
    for (let i = 0; i < uncondLogits.length; i += 1) {
      uncondLogits[i] += penaltyGradScale * penaltyGrad[i];
    }
  }
  const rowU = new Float64Array(R);
  for (let i = 0; i < nToken; i += 1) {
    if (activeSiteOnlyCfg && activeSiteMask && !activeSiteMask[i]) continue;
    softmaxNegInto(rowU, uncondLogits, i * R, R, temperature);
    let sum = 0;
    for (let a = 0; a < R; a += 1) {
      const blended = Math.max(rowU[a] + cfgScale * (probs[i * R + a] - rowU[a]), 0);
      probs[i * R + a] = blended;
      sum += blended;
    }
    const denominator = Math.max(sum, 1e-30);
    for (let a = 0; a < R; a += 1) probs[i * R + a] /= denominator;
  }
  return probs;
}

/** One Gibbs sweep over every colour group — `_potts_proposal_gibbs`. */
function gibbsSweep(S, params, temperature, groups, rng, options) {
  const { nToken, numResidueTypes: R } = params;
  const { tokenMask = null, penalty = null } = options;
  const sequence = new Int32Array(nToken);

  for (const group of groups) {
    argmaxRows(S, nToken, R, sequence);
    let penaltyGrad = null;
    if (penalty) penaltyGrad = penalty.evaluate(S).grad;
    const probs = conditionalProbs(params, sequence, temperature, {
      ...options,
      penaltyGrad,
      penaltyGradScale: 0.5, // the Gibbs path weights the penalty bias by 0.5
    });
    for (let i = 0; i < nToken; i += 1) {
      if (!group[i]) continue;
      if (tokenMask && !tokenMask[i]) continue;
      const pick = sampleCategorical(probs, i * R, R, rng);
      S.fill(0, i * R, i * R + R);
      S[i * R + pick] = 1;
    }
  }
  return S;
}

/**
 * Chain-invariant tables for the row-restricted Gibbs sweep. Build once per
 * backbone and pass as `gibbsPrepared` to every chain's `runPottsMcmc`.
 *
 * Returns `null` when the row-restricted sweep does not apply and the caller
 * must keep the legacy one: a penalty other than `complexityLcp` is active (only
 * the collection `buildPenalties` returns for LCP alone can be evaluated per
 * row), or the alphabet is narrower than the 20 amino acids.
 *
 * Takes the options of `runPottsMcmc`; `temperature` and `temperatureInit` bound
 * the schedule for the dead-state guard (`pottsLiveStates`).
 *
 * @param {import('./potts.js').PottsParams} params
 * @param {object} [options]
 * @param {boolean} [options.transposeJ] copy `J` into the contiguous row layout
 */
export function prepareGibbs(params, options = {}) {
  const {
    tokenMask = null,
    penalty = null,
    uncond = null,
    cfgScale = 1.0,
    activeSiteMask = null,
    granularActiveSiteColoring = false,
    temperature = 0.1,
    temperatureInit = 1.0,
    transposeJ = true,
  } = options;
  const { nToken, numResidueTypes: R } = params;
  const lcp = penalty ? penalty.lcpOnly ?? null : null;
  if (penalty && !(lcp && penalty.names.length === 1 && lcp.nToken === nToken)) return null;
  if (R < NUM_STANDARD_AA) return null;

  // Members to resample per colour, in index order. A group with no mutable
  // member draws nothing and changes nothing, so it is dropped.
  const groups = colorGroups(params, { activeSiteMask, granular: granularActiveSiteColoring });
  const members = [];
  for (const group of groups) {
    const list = [];
    for (let i = 0; i < nToken; i += 1) {
      if (group[i] && (!tokenMask || tokenMask[i])) list.push(i);
    }
    if (list.length > 0) members.push(Int32Array.from(list));
  }
  const rowMask = new Uint8Array(nToken);
  for (const list of members) for (const i of list) rowMask[i] = 1;

  const useCfg = Boolean(uncond) && cfgScale !== 1.0;
  const temperatureMax = Math.max(temperature, temperatureInit);
  const temperatureMin = Math.min(temperature, temperatureInit);
  let penaltyMax = 0;
  if (lcp) {
    // Each of at most `window` covering windows contributes a squared deficit,
    // and a deficit is at most exp(entropyMin) - 1 since perplexity is >= 1.
    const target = Math.exp(lcp.entropyMin ?? 2.32);
    penaltyMax = lcp.weight * Math.min(lcp.windowSize ?? 30, nToken) * (target - 1) * (target - 1);
  }
  const { columns, liveLetters } = pottsLiveStates(useCfg ? [params, uncond] : [params], {
    mutableMask: rowMask,
    temperatureMax,
    temperatureMin,
    penaltyMax,
    penaltyScale: 0.5,
  });

  return {
    params,
    tokenMask,
    penalty,
    uncond: useCfg ? uncond : null,
    activeSiteMask,
    granularActiveSiteColoring,
    temperatureMax,
    temperatureMin,
    numColors: groups.length,
    members,
    columns,
    liveLetters,
    potts: preparePotts(params, { rowMask, columns, transpose: transposeJ }),
    pottsUncond: useCfg ? preparePotts(uncond, { rowMask, columns, transpose: transposeJ }) : null,
    lcpTables: lcp ? createLcpStatic(lcp) : null,
    lcpWeight: lcp ? lcp.weight : 0,
  };
}

/** Whether tables from `prepareGibbs` were built for exactly this run. */
function gibbsPreparedMatches(prepared, params, options) {
  const {
    tokenMask = null,
    penalty = null,
    uncond = null,
    cfgScale = 1.0,
    activeSiteMask = null,
    granularActiveSiteColoring = false,
    temperature = 0.1,
    temperatureInit = 1.0,
  } = options;
  const useCfg = Boolean(uncond) && cfgScale !== 1.0;
  return prepared.params === params
    && prepared.tokenMask === tokenMask
    && prepared.penalty === penalty
    && prepared.uncond === (useCfg ? uncond : null)
    && prepared.activeSiteMask === activeSiteMask
    && prepared.granularActiveSiteColoring === granularActiveSiteColoring
    && Math.max(temperature, temperatureInit) <= prepared.temperatureMax
    && Math.min(temperature, temperatureInit) >= prepared.temperatureMin;
}

/**
 * One chain of row-restricted Gibbs over tables from `prepareGibbs`.
 *
 * `sweep` does what `gibbsSweep` does, for the rows that matter: per colour
 * group, the conditional of every mutable member is computed from the pre-group
 * state, a state is drawn, and only then are the picks applied. Each row repeats
 * the float steps of `conditionalProbs` — double sum over the edges stored to
 * float32, `+ 0.5 * penaltyGrad` in float32, softmax in double stored to float32,
 * the CFG blend — and the generator is consumed once per mutable member in the
 * same order, so the trajectory is bit-identical.
 */
function createGibbsChain(prepared, sequence, rng, options) {
  const { cfgScale = 1.0, activeSiteMask = null, activeSiteOnlyCfg = false } = options;
  const { members, columns: C, potts, pottsUncond } = prepared;
  const Q = NUM_STANDARD_AA;
  const lcp = prepared.lcpTables ? createLcpState(prepared.lcpTables, prepared.liveLetters) : null;
  const lcpWeight = prepared.lcpWeight;
  if (lcp) lcp.init(sequence);

  const R = potts.R;
  const acc = new Float64Array(R);
  const energy = new Float32Array(R);
  const penaltyGrad = new Float32Array(R);
  const row = new Float64Array(R);
  const rowU = new Float64Array(R);
  const probs = new Float32Array(R);
  let largest = 0;
  for (const list of members) largest = Math.max(largest, list.length);
  const picks = new Int32Array(largest);

  function sweep(temperature) {
    for (let g = 0; g < members.length; g += 1) {
      const list = members[g];
      for (let x = 0; x < list.length; x += 1) {
        const i = list[x];
        pottsRowEnergies(potts, sequence, i, acc);
        for (let a = 0; a < C; a += 1) energy[a] = acc[a];
        if (lcp) {
          lcp.rowGrad(i, sequence[i], penaltyGrad);
          if (lcpWeight !== 1) {
            for (let a = 0; a < Q; a += 1) penaltyGrad[a] = penaltyGrad[a] * lcpWeight;
          }
          for (let a = 0; a < C; a += 1) energy[a] += 0.5 * penaltyGrad[a];
        }
        softmaxNegInto(row, energy, 0, C, temperature);
        for (let a = 0; a < C; a += 1) probs[a] = row[a];
        if (pottsUncond && !(activeSiteOnlyCfg && activeSiteMask && !activeSiteMask[i])) {
          pottsRowEnergies(pottsUncond, sequence, i, acc);
          for (let a = 0; a < C; a += 1) energy[a] = acc[a];
          if (lcp) for (let a = 0; a < C; a += 1) energy[a] += 0.5 * penaltyGrad[a];
          softmaxNegInto(rowU, energy, 0, C, temperature);
          let sum = 0;
          for (let a = 0; a < C; a += 1) {
            const blended = Math.max(rowU[a] + cfgScale * (probs[a] - rowU[a]), 0);
            probs[a] = blended;
            sum += blended;
          }
          const denominator = Math.max(sum, 1e-30);
          for (let a = 0; a < C; a += 1) probs[a] /= denominator;
        }
        picks[x] = sampleCategorical(probs, 0, C, rng);
      }
      for (let x = 0; x < list.length; x += 1) {
        const i = list[x];
        const pick = picks[x];
        if (pick === sequence[i]) continue;
        if (lcp) lcp.flip(i, sequence[i], pick);
        sequence[i] = pick;
      }
    }
  }

  return { sequence, sweep };
}

/**
 * DLMC transition log-probabilities and total energy at state `S`
 * — `_dlmc_energy_and_log_probs`.
 */
function dlmcEnergyAndLogProbs(S, params, temperature, dt, options) {
  const { nToken, numResidueTypes: R } = params;
  const { penalty = null } = options;
  const sequence = argmaxRows(S, nToken, R);

  let energy = 0;
  const perNode = pottsEnergyPerNode(params, sequence);
  for (let i = 0; i < nToken; i += 1) energy += perNode[i];

  let penaltyGrad = null;
  if (penalty) {
    const result = penalty.evaluate(S);
    penaltyGrad = Float32Array.from(result.grad);
    // The DLMC path centres the bias on the current state before adding it.
    for (let i = 0; i < nToken; i += 1) {
      const atCurrent = result.grad[i * R + sequence[i]];
      for (let a = 0; a < R; a += 1) penaltyGrad[i * R + a] -= atCurrent;
    }
    energy += result.value;
  }

  const probs = conditionalProbs(params, sequence, temperature, {
    ...options,
    penaltyGrad,
    penaltyGradScale: 1.0, // the DLMC path adds the centred bias at full weight
  });

  const logProbs = new Float32Array(nToken * R);
  for (let i = 0; i < nToken; i += 1) {
    const si = sequence[i];
    const logPi = Math.log(Math.max(probs[i * R + si], 1e-30));
    let pFlip = 0;
    for (let a = 0; a < R; a += 1) {
      if (a === si) continue;
      const logPj = Math.log(Math.max(probs[i * R + a], 1e-30));
      // log sigmoid(logP_j - logP_i), then rate = exp(logQ - logP_j)
      const delta = logPj - logPi;
      const logQ = delta < 0 ? delta - Math.log1p(Math.exp(delta)) : -Math.log1p(Math.exp(-delta));
      const rate = Math.exp(logQ - logPj);
      const logPij = logPj + Math.log(-Math.expm1(-dt * rate));
      logProbs[i * R + a] = logPij;
      pFlip += Math.exp(logPij);
    }
    logProbs[i * R + si] = Math.log(Math.max(1 - pFlip, 1e-5));
  }
  return { energy, logProbs, sequence };
}

/** One DLMC proposal — `_potts_proposal_dlmc`. */
function dlmcProposal(S, params, temperature, dt, rng, options) {
  const { nToken, numResidueTypes: R } = params;
  const { tokenMask = null } = options;
  const { energy, logProbs, sequence } = dlmcEnergyAndLogProbs(S, params, temperature, dt, options);

  const probs = new Float32Array(nToken * R);
  for (let i = 0; i < probs.length; i += 1) probs[i] = Math.exp(logProbs[i]);
  const next = Float32Array.from(S);
  for (let i = 0; i < nToken; i += 1) {
    if (tokenMask && !tokenMask[i]) continue;
    const pick = sampleCategorical(probs, i * R, R, rng);
    next.fill(0, i * R, i * R + R);
    next[i * R + pick] = 1;
  }
  return { S: next, energy, logProbs, sequence };
}

/** Metropolis-Hastings correction for a DLMC proposal — `_mh_rejection_step`. */
function mhRejection(S, forward, params, temperature, dt, rng, options) {
  const { nToken, numResidueTypes: R } = params;
  const reverse = dlmcEnergyAndLogProbs(forward.S, params, temperature, dt, options);
  const proposed = argmaxRows(forward.S, nToken, R);

  let fluxForward = -forward.energy / temperature;
  let fluxReverse = -reverse.energy / temperature;
  for (let i = 0; i < nToken; i += 1) {
    fluxForward += forward.logProbs[i * R + proposed[i]];
    fluxReverse += reverse.logProbs[i * R + forward.sequence[i]];
  }
  const acceptance = Math.min(Math.exp(fluxReverse - fluxForward), 1);
  const accepted = rng() < acceptance;
  return { S: accepted ? forward.S : S, accepted, acceptance };
}

/**
 * Run MCMC on fully prepared Potts parameters — `PottsSampler.sample`.
 *
 * `tokenMask` here is the *mutable* mask: positions outside it keep their
 * `initialSequence` identity for the whole run.
 *
 * @param {import('./potts.js').PottsParams} params
 * @param {object} [options]
 */
export function runPottsMcmc(params, options = {}) {
  const {
    proposal = 'gibbs',
    nSweeps = 150,
    temperature = 0.1,
    temperatureInit = 1.0,
    annealingFraction = 0.8,
    dt = 0.1,
    rejectionStep = false,
    tokenMask = null,
    initialSequence = null,
    penalty = null,
    uncond = null,
    cfgScale = 1.0,
    activeSiteMask = null,
    activeSiteOnlyCfg = false,
    granularActiveSiteColoring = false,
    trackTrajectory = false,
    trajectoryInterval = 1,
    seed = 0,
    rng = createRng(seed),
    legacyPath = false,
    gibbsPrepared = null,
    transposeJ = true,
  } = options;
  const { nToken, numResidueTypes: R } = params;

  let S;
  if (initialSequence) {
    S = initialSequence.length === nToken * R
      ? Float32Array.from(initialSequence)
      : oneHotFromIndices(initialSequence, R);
  } else {
    // Initialise from the field: softmax(-h / temperature_init).
    S = new Float32Array(nToken * R);
    const row = new Float64Array(R);
    for (let i = 0; i < nToken; i += 1) {
      softmaxNegInto(row, params.h, i * R, R, temperatureInit);
      const pick = sampleCategorical(row, 0, R, rng);
      S[i * R + pick] = 1;
    }
  }

  const sweepOptions = {
    tokenMask,
    penalty,
    uncond,
    cfgScale,
    activeSiteMask,
    activeSiteOnlyCfg,
  };
  // Row-restricted Gibbs when it applies (one-hot state, no penalty beyond
  // LCP); otherwise, or on request, the reference-shaped full-grid sweep.
  let chain = null;
  if (proposal === 'gibbs' && !legacyPath) {
    const start = oneHotSequence(S, nToken, R);
    if (start) {
      const prepareOptions = {
        ...sweepOptions,
        granularActiveSiteColoring,
        temperature,
        temperatureInit,
        transposeJ,
      };
      const prepared = gibbsPrepared && gibbsPreparedMatches(gibbsPrepared, params, prepareOptions)
        ? gibbsPrepared
        : prepareGibbs(params, prepareOptions);
      if (prepared) chain = { ...createGibbsChain(prepared, start, rng, sweepOptions), prepared };
    }
  }
  const groups = proposal === 'gibbs' && !chain
    ? colorGroups(params, { activeSiteMask, granular: granularActiveSiteColoring })
    : null;
  const snapshot = () => (chain ? Int32Array.from(chain.sequence) : argmaxRows(S, nToken, R));

  const annealSteps = Math.trunc(annealingFraction * nSweeps);
  const schedule = new Float32Array(nSweeps);
  const trajectory = trackTrajectory
    ? { sweepIndices: [0], sequences: [snapshot()], energies: [] }
    : null;
  // Only the DLMC path can reject, so Gibbs reports null rather than a count.
  let accepted = proposal === 'gibbs' ? null : 0;

  for (let t = 0; t < nSweeps; t += 1) {
    const T = annealSteps > 0 && t < annealSteps
      ? temperatureInit + (t / annealSteps) * (temperature - temperatureInit)
      : temperature;
    schedule[t] = T;

    if (chain) {
      chain.sweep(T);
    } else if (proposal === 'gibbs') {
      S = gibbsSweep(S, params, T, groups, rng, sweepOptions);
    } else if (proposal === 'dlmc') {
      const forward = dlmcProposal(S, params, T, dt, rng, sweepOptions);
      if (rejectionStep) {
        const step = mhRejection(S, forward, params, T, dt, rng, sweepOptions);
        S = step.S;
        if (step.accepted) accepted += 1;
      } else {
        S = forward.S;
        accepted += 1;
      }
    } else {
      throw new Error(`unknown proposal ${JSON.stringify(proposal)}; use "gibbs" or "dlmc"`);
    }

    if (trajectory && ((t + 1) % trajectoryInterval === 0 || t === nSweeps - 1)) {
      trajectory.sweepIndices.push(t + 1);
      trajectory.sequences.push(snapshot());
    }
  }

  const sequence = snapshot();
  if (chain) S = oneHotFromIndices(sequence, R);
  const energyPerNode = pottsEnergyPerNode(params, sequence);
  let totalEnergy = 0;
  for (let i = 0; i < nToken; i += 1) totalEnergy += energyPerNode[i];
  if (trajectory) {
    trajectory.energies = trajectory.sequences.map((seq) => {
      const per = pottsEnergyPerNode(params, seq);
      let sum = 0;
      for (let i = 0; i < nToken; i += 1) sum += per[i];
      return sum;
    });
  }

  return {
    sequence,
    oneHot: S,
    energyPerNode,
    totalEnergy,
    temperatureSchedule: schedule,
    acceptedSweeps: accepted,
    trajectory,
    // Which Gibbs sweep ran, and how many states per row it evaluated.
    gibbsPath: proposal === 'gibbs' ? (chain ? 'row-restricted' : 'legacy') : null,
    gibbsColumns: chain ? chain.prepared.columns : null,
  };
}

/**
 * Everything `sampleFromParams` derives that does not depend on the seed: the
 * masks, the biased fields, the penalty collection and (for Gibbs) the tables of
 * `prepareGibbs`.
 *
 * Sampling several designs of one backbone repeats all of this per chain unless
 * it is built once here and handed back as `prepared`:
 *
 * ```js
 * const prepared = prepareSampleFromParams(params, options);
 * for (let i = 0; i < n; i += 1) sampleFromParams(params, { ...options, seed: seed + i, prepared });
 * ```
 *
 * Takes the options of `sampleFromParams`; `seed` and `rng` are ignored.
 */
export function prepareSampleFromParams(params, options = {}) {
  const {
    context,
    fixedMask = null,
    fixedSequence = null,
    designMask = null,
    excludeCys = true,
    excludeMet = true,
    energyBias = null,
    groupEnergyBias = null,
    alphabetBias = ALPHABET_BIAS,
    uncond = null,
    cfgScale = 1.0,
    activeSiteDistanceCutoff = 10.0,
    activeSiteMask: providedActiveSite = null,
    seed: _seed,
    rng: _rng,
    prepared: _prepared,
    temperatureInit = 1.0,
    ...samplerOptions
  } = options;

  if (!context) throw new Error('sampleFromParams needs the featurized context');
  const { nToken, numResidueTypes: R } = params;
  if (context.nToken !== nToken) {
    throw new Error(`context has ${context.nToken} tokens, params have ${nToken}`);
  }

  const activeSiteMask = providedActiveSite
    ?? buildLigandProximityMask(context, activeSiteDistanceCutoff);

  // design_mask = token_mask & is_protein_token (this port has no motif tokens)
  const baseDesign = designMask ?? (() => {
    const mask = new Uint8Array(nToken);
    for (let i = 0; i < nToken; i += 1) {
      mask[i] = context.tokenMask[i] && context.isProteinToken[i] ? 1 : 0;
    }
    return mask;
  })();
  const mutableMask = buildMutableMask(baseDesign, fixedMask);
  const alphabetMask = buildAlphabetMask(mutableMask, R, excludeCys, excludeMet);

  const biasedH = applyFieldBias(params.h, R, {
    alphabetMask,
    energyBias: energyBias ? buildEnergyBiasVector(energyBias, R) : null,
    groupEnergyBias: groupEnergyBias ? buildGroupEnergyBiasVector(groupEnergyBias, R) : null,
    alphabetBias,
  });
  const biasedParams = { ...params, h: biasedH };

  let biasedUncond = null;
  if (uncond) {
    biasedUncond = {
      ...uncond,
      h: applyFieldBias(uncond.h, R, {
        alphabetMask,
        energyBias: energyBias ? buildEnergyBiasVector(energyBias, R) : null,
        groupEnergyBias: groupEnergyBias ? buildGroupEnergyBiasVector(groupEnergyBias, R) : null,
        alphabetBias,
      }),
    };
  }

  // build_s_init keeps these positions and samples the rest from the field.
  const preserved = new Uint8Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    preserved[i] = context.tokenMask[i] && !mutableMask[i] ? 1 : 0;
  }
  const preservedSequence = fixedSequence
    ? Int32Array.from(fixedSequence)
    : Int32Array.from(context.restypeIndex);

  const penalty = buildPenalties({
    nToken,
    numResidueTypes: R,
    chainId: context.asymId,
    tokenMask: context.tokenMask,
    mutableMask,
    isProteinToken: context.isProteinToken,
    activeSiteMask,
    ...samplerOptions,
  });

  const { proposal = 'gibbs', legacyPath = false } = samplerOptions;
  const gibbsPrepared = proposal === 'gibbs' && !legacyPath
    ? prepareGibbs(biasedParams, {
      ...samplerOptions,
      temperatureInit,
      tokenMask: mutableMask,
      penalty,
      uncond: biasedUncond,
      cfgScale,
      activeSiteMask,
    })
    : null;

  return {
    params,
    context,
    biasedParams,
    biasedUncond,
    biasedH,
    mutableMask,
    alphabetMask,
    activeSiteMask,
    preserved,
    preservedSequence,
    penalty,
    gibbsPrepared,
  };
}

/**
 * Apply masks, biases and penalties to `params`, then sample — the JS mirror of
 * `ArtiosPotts.sample_from_params`.
 *
 * Design positions are the valid protein tokens that are not identity-fixed;
 * every other token (ligand, fixed, non-protein) keeps its own identity and is
 * left out of the alphabet restriction so it carries no artificial bias.
 *
 * @param {import('./potts.js').PottsParams} params raw `h`/`J` from the model
 * @param {object} options
 * @param {object} options.context the featurized context
 * @param {ArrayLike<number>} [options.fixedMask] positions to keep fixed
 * @param {ArrayLike<number>} [options.fixedSequence] restype index per token,
 *   defaulting to `context.restypeIndex`
 * @param {boolean} [options.excludeCys]
 * @param {boolean} [options.excludeMet]
 * @param {Record<string, number>} [options.energyBias] one-letter code -> value
 * @param {Record<string, number>} [options.groupEnergyBias] group name -> value
 * @param {boolean} [options.legacyPath] run the reference-shaped Gibbs sweep and
 *   the brute-force complexity penalty; same output, far slower
 * @param {ReturnType<typeof prepareSampleFromParams>} [options.prepared] the
 *   seed-independent setup, built from these same `params` and options. Only
 *   `seed` / `rng` may differ between the two calls.
 */
export function sampleFromParams(params, options = {}) {
  const {
    context,
    fixedMask: _fixedMask,
    fixedSequence: _fixedSequence,
    designMask: _designMask,
    excludeCys: _excludeCys,
    excludeMet: _excludeMet,
    energyBias: _energyBias,
    groupEnergyBias: _groupEnergyBias,
    alphabetBias: _alphabetBias,
    uncond: _uncond,
    cfgScale = 1.0,
    activeSiteDistanceCutoff: _activeSiteDistanceCutoff,
    activeSiteMask: _activeSiteMask,
    seed = 0,
    rng = createRng(seed),
    temperatureInit = 1.0,
    prepared: providedPrepared = null,
    ...samplerOptions
  } = options;

  if (providedPrepared && (providedPrepared.params !== params || providedPrepared.context !== context)) {
    throw new Error('sampleFromParams: `prepared` was built for different params or context');
  }
  const prepared = providedPrepared ?? prepareSampleFromParams(params, options);
  const {
    biasedParams, biasedUncond, biasedH, mutableMask, alphabetMask, activeSiteMask,
    preserved, preservedSequence, penalty, gibbsPrepared,
  } = prepared;
  const { nToken, numResidueTypes: R } = params;

  // build_s_init: sample the mutable positions from the field, keep the rest.
  const initial = new Float32Array(nToken * R);
  const row = new Float64Array(R);
  for (let i = 0; i < nToken; i += 1) {
    if (preserved[i]) {
      initial[i * R + preservedSequence[i]] = 1;
      continue;
    }
    softmaxNegInto(row, biasedH, i * R, R, temperatureInit);
    initial[i * R + sampleCategorical(row, 0, R, rng)] = 1;
  }

  const result = runPottsMcmc(biasedParams, {
    ...samplerOptions,
    temperatureInit,
    tokenMask: mutableMask,
    initialSequence: initial,
    penalty,
    uncond: biasedUncond,
    cfgScale,
    activeSiteMask,
    rng,
    gibbsPrepared,
  });

  return {
    ...result,
    mutableMask,
    alphabetMask,
    activeSiteMask,
    penaltyNames: penalty ? penalty.names : [],
    biasedH,
    numStandardAa: NUM_STANDARD_AA,
  };
}
