import { zeroCenter } from '$lib/protein-ops/coords';

export interface EulerFMEndpointOpts {
  numSteps: number;
  schedule: Float32Array; // length numSteps + 1
  /** Called once per step with the *post-update* state (step 0 = initial noise). */
  onFrame?: (step: number, t: number, x: Float32Array) => void | Promise<void>;
}

/**
 * Forward Euler integration of an x̂_1 endpoint-prediction flow-matching model.
 *
 * Mirrors sample_teacher_euler in legacy-model-flow-map/scripts/sample_teacher.py:115.
 *   v        = (x̂_1 − x) / (1 − t + ε)
 *   x_{i+1}  = x + (t_{i+1} − t_i) · v
 *   last step snaps to x̂_1 directly (avoids 1/(1−t) blowup as t → 1).
 *   CoM is zeroed after every step (translation-equivariant target).
 *
 * @param forward  closure: takes (x [B·O·3], t [B]) and returns x̂_1 of the same shape.
 *                 The adapter constructs this; the sampler stays model-blind.
 * @param xInit    initial noise [batch, o, 3] flat.
 * @returns        { final, trajectory } where trajectory has numSteps + 1 frames
 *                 (frame 0 = initial noise post-zero-center, last = x̂_1 endpoint).
 */
export async function sampleEulerFMEndpoint(
  forward: (x: Float32Array, t: Float32Array) => Promise<Float32Array>,
  xInit: Float32Array,
  batch: number,
  o: number,
  opts: EulerFMEndpointOpts,
): Promise<{ final: Float32Array; trajectory: Float32Array[] }> {
  let x = new Float32Array(xInit); // defensive copy
  zeroCenter(x, batch, o);

  const ts = opts.schedule;
  if (ts.length !== opts.numSteps + 1) {
    throw new Error(
      `schedule length ${ts.length} != numSteps + 1 = ${opts.numSteps + 1}`,
    );
  }

  const trajectory: Float32Array[] = [x.slice()];
  if (opts.onFrame) await opts.onFrame(0, ts[0], x);

  const tBatch = new Float32Array(batch);

  for (let i = 0; i < opts.numSteps; i++) {
    const tCurr = ts[i];
    const tNext = ts[i + 1];
    tBatch.fill(tCurr);

    const x1Hat = await forward(x, tBatch);

    if (i === opts.numSteps - 1) {
      x = new Float32Array(x1Hat); // copy out of any GPU-backed buffer
    } else {
      const denom = 1.0 - tCurr + 1e-5;
      const dt = tNext - tCurr;
      const scale = dt / denom;
      const next = new Float32Array(x.length);
      for (let j = 0; j < x.length; j++) next[j] = x[j] + scale * (x1Hat[j] - x[j]);
      x = next;
    }
    zeroCenter(x, batch, o);
    trajectory.push(x.slice());
    if (opts.onFrame) await opts.onFrame(i + 1, tNext, x);
  }

  return { final: x, trajectory };
}
