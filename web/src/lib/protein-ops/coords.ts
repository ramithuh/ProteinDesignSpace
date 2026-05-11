// Coordinate utilities used by samplers + viewers.
// All shapes are flat Float32Array buffers; we pass batch + o explicitly.

/** Mulberry32: deterministic, fast PRNG. */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller standard normal sample from a uniform rng. */
export function randn(rng: () => number): number {
  const u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v);
}

/**
 * Cached normal generator: yields one ~N(0,1) per call. Uses Box-Muller in
 * polar form, caching the second sample to avoid wasting half the uniforms.
 */
export function makeNormalRng(seed: number): () => number {
  const u = mulberry32(seed);
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    const r1 = u();
    const r2 = u();
    const mag = Math.sqrt(-2 * Math.log(r1 + 1e-9));
    cached = mag * Math.sin(2 * Math.PI * r2);
    return mag * Math.cos(2 * Math.PI * r2);
  };
}

/**
 * Subtract per-batch centroid in place. x has shape [batch, o, 3] flat.
 */
export function zeroCenter(x: Float32Array, batch: number, o: number): void {
  for (let b = 0; b < batch; b++) {
    let cx = 0;
    let cy = 0;
    let cz = 0;
    const off = b * o * 3;
    for (let i = 0; i < o; i++) {
      cx += x[off + i * 3 + 0];
      cy += x[off + i * 3 + 1];
      cz += x[off + i * 3 + 2];
    }
    cx /= o;
    cy /= o;
    cz /= o;
    for (let i = 0; i < o; i++) {
      x[off + i * 3 + 0] -= cx;
      x[off + i * 3 + 1] -= cy;
      x[off + i * 3 + 2] -= cz;
    }
  }
}

/** Allocate a Gaussian noise buffer of shape [batch, o, 3] and zero-center it. */
export function sampleCenteredNoise(batch: number, o: number, seed: number): Float32Array {
  const rng = mulberry32(seed);
  const x = new Float32Array(batch * o * 3);
  for (let i = 0; i < x.length; i++) x[i] = randn(rng);
  zeroCenter(x, batch, o);
  return x;
}

/** Slice out the b-th sample (length o*3) from a [batch, o, 3] buffer. */
export function sliceSample(x: Float32Array, batch: number, o: number, b: number): Float32Array {
  const start = b * o * 3;
  return x.slice(start, start + o * 3);
}
