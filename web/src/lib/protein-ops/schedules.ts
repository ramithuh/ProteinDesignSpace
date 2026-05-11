// Time schedules for flow-matching / diffusion samplers.
// Pure functions. Returned array has length numSteps + 1.

export type ScheduleKind = 'uniform' | 'linear' | 'power' | 'quadratic' | 'log';

/**
 * Mirror of FlowMapSampler._make_schedule in legacy-model.training.flow_map:862.
 *
 * - uniform / linear: equal Δt.
 * - power / quadratic: t^p — more NFE near t=0 (noise end).
 * - log: 1 − flip(logspace(−p, 0)), normalized — more NFE near t=1 (refinement end).
 *   This is the production schedule for the legacy-model teacher.
 */
export function makeSchedule(numSteps: number, kind: ScheduleKind, p = 2.0): Float32Array {
  const n = numSteps + 1;
  const ts = new Float32Array(n);

  if (kind === 'uniform' || kind === 'linear') {
    for (let i = 0; i < n; i++) ts[i] = i / numSteps;
    return ts;
  }
  if (kind === 'power' || kind === 'quadratic') {
    for (let i = 0; i < n; i++) {
      const t = i / numSteps;
      ts[i] = Math.pow(t, p);
    }
    return ts;
  }
  if (kind === 'log') {
    const raw = new Float64Array(n);
    // logspace(-p, 0, n) = 10^linspace(-p, 0, n)
    for (let i = 0; i < n; i++) {
      const u = i / (n - 1); // [0, 1]
      raw[i] = Math.pow(10, -p + u * p);
    }
    // Flip and complement: t = 1 - flipped
    for (let i = 0; i < n; i++) ts[i] = 1 - raw[n - 1 - i];
    // Renormalize to span exactly [0, 1].
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < n; i++) {
      if (ts[i] < mn) mn = ts[i];
      if (ts[i] > mx) mx = ts[i];
    }
    const range = mx - mn || 1;
    for (let i = 0; i < n; i++) ts[i] = (ts[i] - mn) / range;
    return ts;
  }
  // exhaustive
  const _exhaustive: never = kind;
  throw new Error(`unknown schedule: ${_exhaustive}`);
}
