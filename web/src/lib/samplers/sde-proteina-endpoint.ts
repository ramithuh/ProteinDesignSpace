import { makeNormalRng, zeroCenter } from '$lib/protein-ops/coords';

export type GtMode = 'one-minus-t-over-t' | 'one-over-t';

export interface SDEProteinaEndpointOpts {
  numSteps: number;
  schedule: Float32Array; // length numSteps + 1
  /** Proteina γ — noise scale. Default 0.1 in legacy-model. */
  gamma: number;
  /** g(t) shape. "one-minus-t-over-t" = (1-t)/(t+ε); "one-over-t" = 1/(t+ε). */
  gtMode: GtMode;
  /** ε in the g(t) denominator. Default 1e-2. */
  gtEps: number;
  /** Optional clip on g(t). Production uses 3.0 to tame low-NFE blowup; null = no cap. */
  gtMax: number | null;
  /**
   * Above this t the per-step score-correction would dominate; fall back to a
   * deterministic FLM step. Default 0.99.
   */
  tLimOde: number;
  /** Seed for the per-step noise. Distinct from initial-noise seed. */
  seed: number;
  onFrame?: (step: number, t: number, x: Float32Array) => void | Promise<void>;
}

/**
 * Compute Proteina's g(t). Mirrors FlowMapSampler._compute_gt
 * (legacy-model-flow-map/src/legacy-model/training/flow_map.py:618-643).
 */
function computeGt(t: number, mode: GtMode, eps: number, gtMax: number | null): number {
  const tc = Math.max(Math.min(t, 1 - 1e-5), 0);
  let val: number;
  if (mode === 'one-minus-t-over-t') val = (1 - tc) / (tc + eps);
  else val = 1 / (tc + eps);
  if (gtMax !== null && val > gtMax) val = gtMax;
  return val;
}

/**
 * FLM-style deterministic endpoint step (used near t=1 instead of the SDE step).
 * Equivalent to forward Euler in velocity form: x + ((t_next-t_curr)/(1-t_curr)) · (x̂_1 - x).
 * Mirrors FlowMapSampler._flow_map_step_deterministic at flow_map.py:603-615.
 */
function flmStep(
  x: Float32Array,
  x1Hat: Float32Array,
  tCurr: number,
  tNext: number,
  out: Float32Array,
): void {
  const w = (tNext - tCurr) / (1 - tCurr + 1e-8);
  for (let j = 0; j < x.length; j++) out[j] = x[j] + w * (x1Hat[j] - x[j]);
}

/**
 * Proteina score-corrected Euler SDE over an x̂_1 endpoint-prediction model.
 * Per-step update (when t ≤ t_lim_ode):
 *   v       = (x̂_1 − x) / (1 − t + ε)
 *   score   = (t · v − x) / (1 − t + ε)
 *   g(t)    = (1−t)/(t+ε)  [or 1/(t+ε)], optionally clipped at gt_max
 *   drift   = v + g(t) · score
 *   σ       = sqrt(max(2 · g(t) · γ · dt, 0))
 *   x_next  = x + drift · dt + σ · randn_like(x)
 *
 * Near the endpoint (t > t_lim_ode) and on the final step, fall back to the
 * deterministic FLM step or snap to x̂_1 respectively. Mirrors
 * flow_map.py:766-832 exactly. CoM is zeroed per step (translation-equivariant).
 */
export async function sampleSDEProteinaEndpoint(
  forward: (x: Float32Array, t: Float32Array) => Promise<Float32Array>,
  xInit: Float32Array,
  batch: number,
  o: number,
  opts: SDEProteinaEndpointOpts,
): Promise<{ final: Float32Array; trajectory: Float32Array[] }> {
  let x = new Float32Array(xInit);
  zeroCenter(x, batch, o);

  const ts = opts.schedule;
  if (ts.length !== opts.numSteps + 1) {
    throw new Error(`schedule length ${ts.length} != numSteps+1 = ${opts.numSteps + 1}`);
  }

  const trajectory: Float32Array[] = [x.slice()];
  if (opts.onFrame) await opts.onFrame(0, ts[0], x);

  const tBatch = new Float32Array(batch);
  const noise = makeNormalRng(opts.seed);

  const next = new Float32Array(x.length);

  for (let i = 0; i < opts.numSteps; i++) {
    const tCurr = ts[i];
    const tNext = ts[i + 1];
    const isLast = i === opts.numSteps - 1;
    tBatch.fill(tCurr);

    const x1Hat = await forward(x, tBatch);

    if (isLast) {
      // Final step: snap to endpoint.
      for (let j = 0; j < x.length; j++) next[j] = x1Hat[j];
    } else if (tCurr > opts.tLimOde) {
      // Near endpoint: deterministic FLM step.
      flmStep(x, x1Hat, tCurr, tNext, next);
    } else {
      // Proteina SDE step.
      const denom = 1 - tCurr + 1e-5;
      const gt = computeGt(tCurr, opts.gtMode, opts.gtEps, opts.gtMax);
      const dt = tNext - tCurr;
      const noiseStd = Math.sqrt(Math.max(2 * gt * opts.gamma * dt, 0));
      for (let j = 0; j < x.length; j++) {
        const v = (x1Hat[j] - x[j]) / denom;
        const score = (tCurr * v - x[j]) / denom;
        const drift = v + gt * score;
        next[j] = x[j] + drift * dt + noiseStd * noise();
      }
    }

    // x_next was held in `next`; swap into x for the next iteration without realloc.
    const tmp = x;
    x = new Float32Array(next); // copy so the trajectory stash stays intact
    void tmp;

    zeroCenter(x, batch, o);
    trajectory.push(x.slice());
    if (opts.onFrame) await opts.onFrame(i + 1, tNext, x);
  }

  return { final: x, trajectory };
}
