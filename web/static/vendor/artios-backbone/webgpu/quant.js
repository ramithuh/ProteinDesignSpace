// Weight packing for the WebGPU backbone matmuls.
//
// Linear weights keep PyTorch's [out, in] (row = output channel) layout.
// Packed formats pad K up to a multiple of 4 so every matmul K-quad maps to
// whole u32 words:
//   f16 : 2 halves per u32 (low half = even k), decoded with unpack2x16float
//   i8  : 4 signed bytes per u32 (byte j = k % 4), one f32 scale per row

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/** Round-to-nearest-even float32 -> IEEE binary16 bits (subnormals kept). */
export function f32ToF16Bits(value) {
  f32Scratch[0] = value;
  const x = u32Scratch[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h++;
    return sign | h;
  }
  let h = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
  return sign | h;
}

/** IEEE binary16 bits -> float32 value. */
export function f16BitsToF32(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

export const roundUp = (x, m) => Math.ceil(x / m) * m;

/**
 * Storage format chosen for a linear layer. Tiny-K projections (coordinate
 * embeddings, K = 3) stay FP32: they are a few KB and quantising 3-element
 * rows only costs accuracy.
 */
export function linearFormat(precision, K) {
  if (precision === 'fp32' || K < 32) return 'f32';
  if (precision === 'fp16') return 'f16';
  if (precision === 'int8') return 'i8';
  throw new Error(`unknown precision "${precision}" (expected fp32 | fp16 | int8)`);
}

export function packLinearF16(W, N, K) {
  const ldw = roundUp(K, 4) / 2;
  const data = new Uint32Array(N * ldw);
  for (let n = 0; n < N; n++) {
    for (let k = 0; k < K; k += 2) {
      const lo = f32ToF16Bits(W[n * K + k]);
      const hi = k + 1 < K ? f32ToF16Bits(W[n * K + k + 1]) : 0;
      data[n * ldw + (k >> 1)] = (lo | (hi << 16)) >>> 0;
    }
  }
  return { data, ldw };
}

/** Symmetric per-output-row INT8: w ~= q * scale[n], q in [-127, 127]. */
export function quantizeLinearInt8(W, N, K) {
  const ldw = roundUp(K, 4) / 4;
  const data = new Uint32Array(N * ldw);
  const scales = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    let amax = 0;
    for (let k = 0; k < K; k++) amax = Math.max(amax, Math.abs(W[n * K + k]));
    const scale = Math.fround(amax / 127);
    scales[n] = scale;
    for (let k = 0; k < K; k++) {
      const q = scale > 0 ? Math.max(-127, Math.min(127, Math.round(W[n * K + k] / scale))) : 0;
      const word = n * ldw + (k >> 2);
      data[word] = (data[word] | ((q & 0xff) << (8 * (k & 3)))) >>> 0;
    }
  }
  return { data, scales, ldw };
}

/** Exact FP32 values the GPU sees for a linear layer stored as `fmt`. */
export function dequantizeLinear(W, N, K, fmt) {
  const out = new Float32Array(N * K);
  if (fmt === 'f32') {
    out.set(W.subarray ? W.subarray(0, N * K) : W);
  } else if (fmt === 'f16') {
    for (let i = 0; i < N * K; i++) out[i] = f16BitsToF32(f32ToF16Bits(W[i]));
  } else if (fmt === 'i8') {
    const { data, scales, ldw } = quantizeLinearInt8(W, N, K);
    for (let n = 0; n < N; n++) {
      for (let k = 0; k < K; k++) {
        const byte = (data[n * ldw + (k >> 2)] >>> (8 * (k & 3))) & 0xff;
        out[n * K + k] = Math.fround(((byte << 24) >> 24) * scales[n]);
      }
    }
  } else {
    throw new Error(`unknown linear format ${fmt}`);
  }
  return out;
}
