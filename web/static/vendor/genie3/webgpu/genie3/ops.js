/**
 * Recorded-dispatch WebGPU compute engine for Genie 3.
 *
 * Modeled on LocalFold's BioEmu2 recorder (`bioemu2/ops.js`) and RFD4-Proteina's `rfd4p/engine.js`:
 * - Every op is recorded ONCE (pipeline, bind group, uniform block) into a flat program list.
 * - Replaying a denoiser step allocates ZERO GPU buffers, bind groups, or pipelines.
 * - Activations are float32 (matching PyTorch FP32 reference).
 * - Weights can be float32 or packed float16 (two f16s per u32 along K). Packed half weights work
 *   on EVERY WebGPU device via core WGSL `unpack2x16float`, and automatically stage in `f16` workgroup
 *   memory when `shader-f16` + `bitcast<vec2<f16>>` is supported.
 * - Every kernel stays within WebGPU spec minimums (`workgroup_size <= 256`, shared memory <= 16 KiB)
 *   and strictly guards folded 2D dispatch indices (`wid.x + wid.y * 65535u`) before buffer access.
 */

import { getDeviceCapabilities } from './device.js';

export const MAX_X = 65535;
export const STORAGE = 0x80;   // GPUBufferUsage.STORAGE
export const COPY_SRC = 0x4;   // GPUBufferUsage.COPY_SRC
export const COPY_DST = 0x8;   // GPUBufferUsage.COPY_DST
export const UNIFORM = 0x40;   // GPUBufferUsage.UNIFORM
export const MAP_READ = 0x1;   // GPUBufferUsage.MAP_READ
const MAP_MODE_READ = 0x1;     // GPUMapMode.READ

/** Fold `groups` workgroups into a 2D (x, y) dispatch grid under WebGPU's 65535 per-dimension cap. */
export function fold2(groups) {
  if (groups <= MAX_X) return [Math.max(groups, 1), 1];
  return [MAX_X, Math.ceil(groups / MAX_X)];
}

/**
 * Tiled 64x128 GEMM: C = alpha * A . B^T (+ bias) [act] [gate / + C], over (b0, b1) batches.
 *
 * A (m, k) at offA + b0*sA0 + b1*sA1 + m*sAm + k*sAk
 * B (n, k) at offB + b0*sB0 + b1*sB1 + n*sBn + k*sBk
 * C (m, n) at offC + b0*sC0 + b1*sC1 + m*sCm + n*sCn
 *
 * Workgroup: 256 threads, 64x128 output tile (4x8 per thread), BK = 16.
 * Shared memory: ta = 16*17 vec4<f32> (4352 B) + tb = 16*33 vec4<EB> (4224 B in f16, 8448 B in f32).
 * Max shared memory = 12800 B <= 16384 B (WebGPU guaranteed floor).
 *
 * `wordB`: B is float16 packed two to a u32 along k (`sBk == 1`, even `offB`, `sB0`, `sB1`, `sBn`).
 * `b16`: requires `wordB` and `shader-f16`; stages `tb` as `vec4<f16>` using `bitcast<vec2<f16>>`.
 * `hasGate`: enables binding 5 (`gsc`) with `gMode`:
 *   1 = v * g, 2 = c + v * g, 3 = v * sigmoid(g), 4 = c + v * sigmoid(g).
 */
export function gemmSource({ wordB = false, b16 = false, hasGate = false } = {}) {
  const useF16 = Boolean(wordB && b16);
  const EB = useF16 ? 'f16' : 'f32';
  const rows = [0, 1, 2, 3];
  const loadHalfB = useF16
    ? `if (rowB0 < p.n) { let p0 = bitcast<vec2<f16>>(b[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
        if (rowB0 + 1u < p.n) { let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
        if (rowB0 + 2u < p.n) { let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
        if (rowB0 + 3u < p.n) { let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }`
    : `if (rowB0 < p.n) { let p0 = unpack2x16float(b[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
        if (rowB0 + 1u < p.n) { let p1 = unpack2x16float(b[bOff + sBnHalf]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
        if (rowB0 + 2u < p.n) { let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
        if (rowB0 + 3u < p.n) { let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }`;
  const loadHalfBFull = useF16
    ? `let p0 = bitcast<vec2<f16>>(b[bOff]);
        let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]);
        let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]);
        let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]);
        lo = vec4<f16>(p0.x, p1.x, p2.x, p3.x);
        if (hasHi) { hi = vec4<f16>(p0.y, p1.y, p2.y, p3.y); }`
    : `let p0 = unpack2x16float(b[bOff]);
        let p1 = unpack2x16float(b[bOff + sBnHalf]);
        let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]);
        let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]);
        lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
        if (hasHi) { hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y); }`;

  return `${useF16 ? 'enable f16;\n' : ''}
struct P {
  m: u32, n: u32, k: u32, b1: u32,
  offA: u32, sA0: u32, sA1: u32, sAm: u32, sAk: u32,
  offB: u32, sB0: u32, sB1: u32, sBn: u32, sBk: u32,
  offC: u32, sC0: u32, sC1: u32, sCm: u32, sCn: u32,
  bias: u32, act: u32, accumulate: u32, alpha: f32, batches: u32,
  gOff: u32, gStride: u32, gMode: u32, pad0: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<${wordB ? 'u32' : 'f32'}>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
${hasGate ? '@group(0) @binding(5) var<storage, read> gsc: array<f32>;' : ''}
var<workgroup> ta: array<vec4<f32>, ${16 * 17}>;
var<workgroup> tb: array<vec4<${EB}>, ${16 * 33}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.n + 127u) / 128u;
  let tilesM = (p.m + 63u) / 64u;
  let flat = wid.x + wid.y * 65535u;
  let batch = flat / (tilesN * tilesM);
  let inBatch = batch < p.batches;
  let bat = min(batch, p.batches - 1u);
  let tile = flat % (tilesN * tilesM);
  let m0 = (tile / tilesN) * 64u;
  let n0 = (tile % tilesN) * 128u;
  let b0 = bat / p.b1;
  let b1 = bat % p.b1;
  let baseA = p.offA + b0 * p.sA0 + b1 * p.sA1;
  let baseB = p.offB + b0 * p.sB0 + b1 * p.sB1;

  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * 4u;
  let tn0 = tx * 4u;
  let tn1 = tn0 + 64u;

  var acc_0_0 = vec4<f32>(0.0); var acc_0_1 = vec4<f32>(0.0);
  var acc_1_0 = vec4<f32>(0.0); var acc_1_1 = vec4<f32>(0.0);
  var acc_2_0 = vec4<f32>(0.0); var acc_2_1 = vec4<f32>(0.0);
  var acc_3_0 = vec4<f32>(0.0); var acc_3_1 = vec4<f32>(0.0);

  let kkA = li % 16u;
  let r4A = li / 16u;
  let rowA0 = m0 + r4A * 4u;
  let aBase0 = baseA + rowA0 * p.sAm + kkA * p.sAk;
  let fullA = rowA0 + 3u < p.m;
  let taWrite = kkA * 17u + r4A;
  ${wordB ? `let kkB = (li % 8u) * 2u;
  let r4B = li / 8u;
  let rowB0 = n0 + r4B * 4u;
  let sBnHalf = p.sBn / 2u;
  let bBase0 = (baseB + rowB0 * p.sBn + kkB) / 2u;
  let fullB = rowB0 + 3u < p.n;
  let tbWrite0 = kkB * 33u + r4B;
  let tbWrite1 = (kkB + 1u) * 33u + r4B;` : ''}

  for (var k0 = 0u; k0 < p.k; k0 += 16u) {
    {
      var v = vec4<f32>(0.0);
      if (inBatch && k0 + kkA < p.k) {
        let aOff = aBase0 + k0 * p.sAk;
        if (fullA) {
          v = vec4<f32>(a[aOff], a[aOff + p.sAm], a[aOff + 2u * p.sAm], a[aOff + 3u * p.sAm]);
        } else {
          if (rowA0 < p.m) { v.x = a[aOff]; }
          if (rowA0 + 1u < p.m) { v.y = a[aOff + p.sAm]; }
          if (rowA0 + 2u < p.m) { v.z = a[aOff + 2u * p.sAm]; }
          if (rowA0 + 3u < p.m) { v.w = a[aOff + 3u * p.sAm]; }
        }
      }
      ta[taWrite] = v;
    }
    ${wordB ? `{
      var lo = vec4<${EB}>(0.0);
      var hi = vec4<${EB}>(0.0);
      if (inBatch && k0 + kkB < p.k) {
        let hasHi = k0 + kkB + 1u < p.k;
        let bOff = bBase0 + (k0 / 2u);
        if (fullB) {
          ${loadHalfBFull}
        } else {
          ${loadHalfB}
        }
      }
      tb[tbWrite0] = lo;
      tb[tbWrite1] = hi;
    }` : `for (var e = 0u; e < 2u; e += 1u) {
      let slot = li + e * 256u;
      let kk = slot % 16u;
      let r4 = slot / 16u;
      let k = k0 + kk;
      var v = vec4<f32>(0.0);
      if (inBatch && k < p.k) {
        let row0 = n0 + r4 * 4u;
        let bAddr0 = baseB + row0 * p.sBn + k * p.sBk;
        if (row0 + 3u < p.n) {
          v = vec4<f32>(b[bAddr0], b[bAddr0 + p.sBn], b[bAddr0 + 2u * p.sBn], b[bAddr0 + 3u * p.sBn]);
        } else {
          if (row0 < p.n) { v.x = b[bAddr0]; }
          if (row0 + 1u < p.n) { v.y = b[bAddr0 + p.sBn]; }
          if (row0 + 2u < p.n) { v.z = b[bAddr0 + 2u * p.sBn]; }
          if (row0 + 3u < p.n) { v.w = b[bAddr0 + 3u * p.sBn]; }
        }
      }
      tb[kk * 33u + r4] = v;
    }`}
    workgroupBarrier();
    ${Array.from({ length: 16 }, (_, kk) => `{
      let av = ta[${kk * 17}u + tm4];
      let bv0 = vec4<f32>(tb[${kk * 33}u + tx]);
      let bv1 = vec4<f32>(tb[${kk * 33 + 16}u + tx]);
      acc_0_0 = fma(vec4<f32>(av.x), bv0, acc_0_0); acc_0_1 = fma(vec4<f32>(av.x), bv1, acc_0_1);
      acc_1_0 = fma(vec4<f32>(av.y), bv0, acc_1_0); acc_1_1 = fma(vec4<f32>(av.y), bv1, acc_1_1);
      acc_2_0 = fma(vec4<f32>(av.z), bv0, acc_2_0); acc_2_1 = fma(vec4<f32>(av.z), bv1, acc_2_1);
      acc_3_0 = fma(vec4<f32>(av.w), bv0, acc_3_0); acc_3_1 = fma(vec4<f32>(av.w), bv1, acc_3_1);
    }`).join('\n    ')}
    workgroupBarrier();
  }

  if (!inBatch) { return; }
  let baseC = p.offC + b0 * p.sC0 + b1 * p.sC1;
  ${rows.map((i) => `{
    let m = m0 + tm + ${i}u;
    if (m < p.m) {
      let r0 = acc_${i}_0 * p.alpha;
      let r1 = acc_${i}_1 * p.alpha;
      let rowC = baseC + m * p.sCm;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((j) => {
        const nExpr = j < 4 ? `n0 + tn0 + ${j}u` : `n0 + tn1 + ${j - 4}u`;
        const valExpr = j < 4 ? `r0.${'xyzw'[j]}` : `r1.${'xyzw'[j - 4]}`;
        return `{
        let n = ${nExpr};
        if (n < p.n) {
          var v = ${valExpr};
          if (p.bias != 0u) { v += bias[n]; }
          if (p.act == 1u) { v = max(v, 0.0); }
          if (p.act == 2u) { v = 1.0 / (1.0 + exp(-v)); }
          let at = rowC + n * p.sCn;
          ${hasGate ? `let gv = gsc[p.gOff + m * p.gStride + n];
          let gval = select(gv, 1.0 / (1.0 + exp(-gv)), p.gMode == 3u || p.gMode == 4u);
          var outV = v * gval;
          if (p.gMode == 2u || p.gMode == 4u || p.accumulate != 0u) { outV += c[at]; }
          c[at] = outV;` : `if (p.accumulate != 0u) { v += c[at]; }
          c[at] = v;`}
        }
      }`;
      }).join('\n      ')}
    }
  }`).join('\n  ')}
}`;
}

/**
 * PyTorch-exact two-pass LayerNorm: y = (x - mean) * inverseSqrt(var + eps) [* w] [+ b].
 * One 64-lane workgroup per row; folded 2D dispatch with uniform control flow across all barriers.
 */
export const LAYER_NORM_SOURCE = `
struct P {
  rows: u32, width: u32, xStride: u32, yStride: u32,
  xOff: u32, yOff: u32, hasW: u32, hasB: u32,
  eps: f32, pad0: u32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> bv: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let xb = p.xOff + row * p.xStride;
  var s = 0.0;
  for (var c = li; c < p.width; c += 64u) { s += x[xb + c]; }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
      + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let mean = red[0] / f32(p.width);
  workgroupBarrier();
  var q = 0.0;
  for (var c = li; c < p.width; c += 64u) {
    let d = x[xb + c] - mean;
    q += d * d;
  }
  red[li] = q;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
      + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let inv = inverseSqrt(red[0] / f32(p.width) + p.eps);
  if (!live) { return; }
  let yb = p.yOff + row * p.yStride;
  for (var c = li; c < p.width; c += 64u) {
    var v = (x[xb + c] - mean) * inv;
    if (p.hasW != 0u) { v *= w[c]; }
    if (p.hasB != 0u) { v += bv[c]; }
    y[yb + c] = v;
  }
}
`;

/**
 * In-place LayerNorm on a single read_write buffer (avoids aliasing a buffer as both read & read_write).
 */
export const LAYER_NORM_INPLACE_SOURCE = `
struct P {
  rows: u32, width: u32, stride: u32, off: u32,
  hasW: u32, hasB: u32, eps: f32, pad0: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bv: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let base = p.off + row * p.stride;
  var s = 0.0;
  for (var c = li; c < p.width; c += 64u) { s += y[base + c]; }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
      + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let mean = red[0] / f32(p.width);
  workgroupBarrier();
  var q = 0.0;
  for (var c = li; c < p.width; c += 64u) {
    let d = y[base + c] - mean;
    q += d * d;
  }
  red[li] = q;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
      + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let inv = inverseSqrt(red[0] / f32(p.width) + p.eps);
  if (!live) { return; }
  for (var c = li; c < p.width; c += 64u) {
    var v = (y[base + c] - mean) * inv;
    if (p.hasW != 0u) { v *= w[c]; }
    if (p.hasB != 0u) { v += bv[c]; }
    y[base + c] = v;
  }
}
`;

/**
 * In-place row-wise numerically stable Softmax over `rows` rows of length `len`.
 * One 64-lane workgroup per row; folded 2D dispatch.
 */
export const SOFTMAX_ROWS_SOURCE = `
struct P { rows: u32, len: u32, off: u32, stride: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let base = p.off + row * p.stride;

  var mx = -3.0e38;
  for (var j = li; j < p.len; j += 64u) {
    mx = max(mx, data[base + j]);
  }
  red[li] = mx;
  workgroupBarrier();
  if (li < 32u) { red[li] = max(red[li], red[li + 32u]); } workgroupBarrier();
  if (li < 16u) { red[li] = max(red[li], red[li + 16u]); } workgroupBarrier();
  if (li < 8u) { red[li] = max(red[li], red[li + 8u]); } workgroupBarrier();
  if (li == 0u) {
    red[0] = max(max(max(red[0], red[4]), max(red[2], red[6])),
                 max(max(red[1], red[5]), max(red[3], red[7])));
  }
  workgroupBarrier();
  let top = red[0];
  workgroupBarrier();

  var s = 0.0;
  for (var j = li; j < p.len; j += 64u) {
    s += exp(data[base + j] - top);
  }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
      + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let invSum = 1.0 / red[0];
  if (!live) { return; }
  for (var j = li; j < p.len; j += 64u) {
    data[base + j] = exp(data[base + j] - top) * invSum;
  }
}
`;

/**
 * Fused channel-major transpose + LayerNorm for TriangleMult output:
 * Reads `x[c * nn + (r0 + row)]` for `c in 0..width-1`, applies LayerNorm over `c`,
 * writes `y[row * width + c]` (contiguous row-major chunk).
 */
export const TRANSPOSE_LN_SOURCE = `
struct P {
  rows: u32, width: u32, nn: u32, r0: u32,
  hasW: u32, hasB: u32, eps: f32, pad0: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> bv: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let globalRow = p.r0 + row;

  var s = 0.0;
  for (var c = li; c < p.width; c += 64u) {
    s += x[c * p.nn + globalRow];
  }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
      + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let mean = red[0] / f32(p.width);
  workgroupBarrier();

  var q = 0.0;
  for (var c = li; c < p.width; c += 64u) {
    let d = x[c * p.nn + globalRow] - mean;
    q += d * d;
  }
  red[li] = q;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
      + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let inv = inverseSqrt(red[0] / f32(p.width) + p.eps);
  if (!live) { return; }
  let yb = row * p.width;
  for (var c = li; c < p.width; c += 64u) {
    var v = (x[c * p.nn + globalRow] - mean) * inv;
    if (p.hasW != 0u) { v *= w[c]; }
    if (p.hasB != 0u) { v += bv[c]; }
    y[yb + c] = v;
  }
}
`;

/**
 * Generic elementwise kernel over `count` items with folded 2D dispatch guard.
 */
export function elementwiseSource(bindings, body) {
  const decls = bindings
    .map((b, at) => (b === null ? '' : `@group(0) @binding(${at + 1}) var<storage, ${b.write ? 'read_write' : 'read'}> b${at}: array<${b.type ?? 'f32'}>;`))
    .join('\n');
  return `
struct P { count: u32, pad0: u32, pad1: u32, pad2: u32, u: array<vec4<u32>, 4>, f: array<vec4<f32>, 4> };
@group(0) @binding(0) var<uniform> p: P;
${decls}
fn U(k: u32) -> u32 { return p.u[k / 4u][k % 4u]; }
fn F(k: u32) -> f32 { return p.f[k / 4u][k % 4u]; }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
${body}
}
`;
}

/** Pack a Float32Array into IEEE-754 float16 pairs inside a Uint32Array (2 f16s per u32). */
const _f32Scratch = new Float32Array(1);
const _u32Scratch = new Uint32Array(_f32Scratch.buffer);

export function float32ToFloat16Bits(val) {
  _f32Scratch[0] = val;
  const x = _u32Scratch[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 112;
  let mant = x & 0x007fffff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x00800000) >>> (1 - exp);
    if (mant & 0x00001000) mant += 0x00002000;
    return sign | (mant >>> 13);
  }
  if (exp === 143) {
    return mant ? (sign | 0x7e00 | (mant >>> 13)) : (sign | 0x7c00);
  }
  if (mant & 0x00001000) {
    mant += 0x00002000;
    if (mant & 0x00800000) {
      mant = 0;
      exp += 1;
    }
  }
  if (exp > 30) return sign | 0x7c00;
  return sign | (exp << 10) | (mant >>> 13);
}

export function packFloat32ToHalfWords(floats) {
  const words = new Uint32Array(Math.ceil(floats.length / 2));
  const len = floats.length;
  for (let i = 0, w = 0; i < len; i += 2, w += 1) {
    const lo = float32ToFloat16Bits(floats[i]);
    const hi = i + 1 < len ? float32ToFloat16Bits(floats[i + 1]) : 0;
    words[w] = lo | (hi << 16);
  }
  return words;
}

const devicePipelineCache = new WeakMap();
function pipelineCacheFor(device) {
  let cache = devicePipelineCache.get(device);
  if (!cache) {
    cache = new Map();
    devicePipelineCache.set(device, cache);
  }
  return cache;
}

export class Genie3Ops {
  /**
   * @param {GPUDevice} device
   * @param {{b16Panels?: boolean}} [opts]
   */
  constructor(device, opts = {}) {
    this.device = device;
    this.caps = getDeviceCapabilities(device);
    this.hasF16 = Boolean(this.caps.hasF16);
    this.b16Panels = opts.b16Panels ?? Boolean(this.caps.hasF16 && this.caps.f16Bitcast);
    this.pipelines = pipelineCacheFor(device);
    this.dummy = device.createBuffer({ label: 'genie3.dummy', size: 16, usage: STORAGE | COPY_DST });
    this.allocated = [];
    this.bytes = 0;
    this.weightBuffers = [];
    this.weightBytes = 0;
    this.readbackBuf = null;
    this.readbackSize = 0;
  }

  async pipeline(key, source) {
    const fullKey = `genie3:${key}`;
    if (!this.pipelines.has(fullKey)) {
      const module = this.device.createShaderModule({ label: fullKey, code: source });
      const promise = this.device.createComputePipelineAsync({
        label: fullKey,
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      this.pipelines.set(fullKey, promise);
    }
    return this.pipelines.get(fullKey);
  }

  /** Allocate a persistent activation storage buffer of `floats` 32-bit elements. */
  buffer(label, floats, data) {
    const size = Math.max(16, Math.ceil((floats * 4) / 16) * 16);
    const buffer = this.device.createBuffer({
      label: `genie3.${label}`,
      size,
      usage: STORAGE | COPY_SRC | COPY_DST,
    });
    if (data !== undefined) this.write(buffer, data);
    this.allocated.push(buffer);
    this.bytes += size;
    buffer.floats = floats;
    return buffer;
  }

  /** Allocate a weight storage buffer (lives across sequence length re-recordings). */
  weight(label, values, half = false, kDim = null) {
    const floats = values instanceof Float32Array ? values : Float32Array.from(values);
    const canHalf = Boolean(half && (kDim != null ? kDim % 2 === 0 : floats.length % 2 === 0));
    if (!canHalf) {
      const size = Math.max(16, Math.ceil((floats.length * 4) / 16) * 16);
      const buffer = this.device.createBuffer({
        label: `genie3.w.${label}`,
        size,
        usage: STORAGE | COPY_DST,
      });
      this.write(buffer, floats);
      this.weightBuffers.push(buffer);
      this.weightBytes += size;
      buffer.floats = floats.length;
      buffer.half = false;
      return buffer;
    }
    const packed = packFloat32ToHalfWords(floats);
    const size = Math.max(16, Math.ceil((packed.byteLength) / 16) * 16);
    const buffer = this.device.createBuffer({
      label: `genie3.w16.${label}`,
      size,
      usage: STORAGE | COPY_DST,
    });
    this.write(buffer, packed);
    this.weightBuffers.push(buffer);
    this.weightBytes += size;
    buffer.floats = floats.length;
    buffer.half = true;
    return buffer;
  }

  write(buffer, data, offsetFloats = 0) {
    const view = data instanceof ArrayBuffer
      ? new Uint8Array(data)
      : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const padded = view.byteLength % 4 === 0 ? view : (() => {
      const out = new Uint8Array(Math.ceil(view.byteLength / 4) * 4);
      out.set(view);
      return out;
    })();
    this.device.queue.writeBuffer(buffer, offsetFloats * 4, padded);
  }

  uniform(words) {
    const size = Math.max(16, Math.ceil(words.byteLength / 16) * 16);
    const buffer = this.device.createBuffer({ label: 'genie3.uniform', size, usage: UNIFORM | COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, words.buffer, words.byteOffset, words.byteLength);
    this.allocated.push(buffer);
    this.bytes += size;
    return buffer;
  }

  async step(key, source, groups, buffers, uniformWords, label = key) {
    const pipeline = await this.pipeline(key, source);
    const entries = [
      { binding: 0, resource: { buffer: this.uniform(uniformWords) } },
      ...buffers
        .map((b, at) => (b == null ? null : {
          binding: at + 1,
          resource: b.view ? { buffer: b.buffer, offset: b.offset, size: b.size } : { buffer: b },
        }))
        .filter(Boolean),
    ];
    const bindGroup = this.device.createBindGroup({
      label: `genie3.bg.${label}`,
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    const [x, y] = fold2(groups);
    return { pipeline, bindGroup, x, y, key, label };
  }

  /** Flatten nested step lists / programs into a single executable program. */
  program(steps) {
    const flat = [];
    const push = (item) => {
      if (!item) return;
      if (Array.isArray(item)) {
        for (const sub of item) push(sub);
      } else if (item.steps) {
        for (const sub of item.steps) push(sub);
      } else {
        flat.push(item);
      }
    };
    push(steps);
    return { steps: flat };
  }

  /** Record a tiled GEMM step. */
  async gemm(spec) {
    const { m, n, k } = spec;
    const [B0, B1] = spec.batches ?? [1, 1];
    const A = { off: 0, s0: 0, s1: 0, m: k, k: 1, ...spec.A };
    const B = { off: 0, s0: 0, s1: 0, n: k, k: 1, ...spec.B };
    const C = { off: 0, s0: 0, s1: 0, m: n, n: 1, ...spec.C };
    const wordB = spec.b.half === true;
    if (wordB && (B.k !== 1 || B.off % 2 || B.s0 % 2 || B.s1 % 2 || B.n % 2)) {
      throw new Error(`gemm(${spec.label ?? ''}): half weight requires unit k-stride and even offsets/strides`);
    }
    const b16 = wordB && this.b16Panels;
    const hasGate = Boolean(spec.gate);
    const g = spec.gate ?? null;
    const gMode = !g
      ? 0
      : g.sigmoid
        ? (spec.accumulate ? 4 : 3)
        : (spec.accumulate ? 2 : 1);

    const u = new Uint32Array(28);
    const f = new Float32Array(u.buffer);
    u.set([
      m, n, k, B1,
      A.off, A.s0, A.s1, A.m, A.k,
      B.off, B.s0, B.s1, B.n, B.k,
      C.off, C.s0, C.s1, C.m, C.n,
      spec.bias ? 1 : 0,
      { relu: 1, sigmoid: 2 }[spec.act] ?? 0,
      spec.accumulate ? 1 : 0,
    ]);
    f[22] = spec.alpha ?? 1.0;
    u[23] = B0 * B1;
    u[24] = g?.off ?? 0;
    u[25] = g?.stride ?? n;
    u[26] = gMode;
    u[27] = 0;

    const groups = Math.ceil(m / 64) * Math.ceil(n / 128) * B0 * B1;
    const key = `gemm${wordB ? '-h' : ''}${b16 ? '-b16' : ''}${hasGate ? '-g' : ''}`;
    const buffers = [spec.a, spec.b, spec.c, spec.bias ?? this.dummy, ...(hasGate ? [g.sc] : [])];
    return this.step(key, gemmSource({ wordB, b16, hasGate }), groups, buffers, u, spec.label ?? key);
  }

  /** Convenience wrapper for y = x @ w^T (+ bias) [act] [gate / + y]. */
  linear({ x, w, y, rows, inC, outC, bias = null, act = null, accumulate = false, alpha = 1.0, xOff = 0, yOff = 0, gate = null, label = null }) {
    return this.gemm({
      m: rows,
      n: outC,
      k: inC,
      a: x,
      b: w,
      c: y,
      bias,
      act,
      accumulate,
      alpha,
      gate,
      A: { off: xOff, m: inC, k: 1 },
      B: { off: 0, n: inC, k: 1 },
      C: { off: yOff, m: outC, n: 1 },
      label: label ?? `linear_${rows}x${outC}x${inC}`,
    });
  }

  /** Record LayerNorm step (out-of-place or in-place when x === y). */
  layerNorm({ x, y, rows, width, w = null, b = null, xOff = 0, yOff = 0, xStride = width, yStride = width, eps = 1e-5, label = 'ln' }) {
    if (x === y) {
      const u = new Uint32Array(8);
      const f = new Float32Array(u.buffer);
      u.set([rows, width, xStride, xOff, w ? 1 : 0, b ? 1 : 0]);
      f[6] = eps;
      return this.step('ln-inplace', LAYER_NORM_INPLACE_SOURCE, rows, [y, w ?? this.dummy, b ?? this.dummy], u, label);
    }
    const u = new Uint32Array(12);
    const f = new Float32Array(u.buffer);
    u.set([rows, width, xStride, yStride, xOff, yOff, w ? 1 : 0, b ? 1 : 0]);
    f[8] = eps;
    return this.step('ln', LAYER_NORM_SOURCE, rows, [x, y, w ?? this.dummy, b ?? this.dummy], u, label);
  }

  /** Record in-place row-wise softmax over `rows` rows of length `len`. */
  softmaxRows({ data, rows, len, off = 0, stride = len, label = 'softmax' }) {
    const u = new Uint32Array([rows, len, off, stride]);
    return this.step('softmax-rows', SOFTMAX_ROWS_SOURCE, rows, [data], u, label);
  }

  /** Record fused channel-major transpose + LayerNorm for TriangleMult output. */
  transposeLayerNorm({ x, y, rows, width, nn, r0 = 0, w = null, b = null, eps = 1e-5, label = 'tri-ln-out' }) {
    const u = new Uint32Array(8);
    const f = new Float32Array(u.buffer);
    u.set([rows, width, nn, r0, w ? 1 : 0, b ? 1 : 0]);
    f[6] = eps;
    return this.step('transpose-ln', TRANSPOSE_LN_SOURCE, rows, [x, y, w ?? this.dummy, b ?? this.dummy], u, label);
  }

  /** Record a custom elementwise map kernel over `count` items. */
  map(key, count, bindings, body, uVals = [], fVals = []) {
    const u = new Uint32Array(36);
    const f = new Float32Array(u.buffer);
    u[0] = count;
    for (let i = 0; i < uVals.length; i += 1) u[4 + i] = uVals[i];
    for (let i = 0; i < fVals.length; i += 1) f[20 + i] = fVals[i];
    const groups = Math.ceil(count / 256);
    const buffers = bindings.map((b) => (b ? b.buffer : null));
    return this.step(`map:${key}`, elementwiseSource(bindings, body), groups, buffers, u, key);
  }

  /** Encode and submit one or more recorded programs (chunks submits to stay well under GPU watchdogs). */
  run(programs) {
    const list = Array.isArray(programs) ? programs : [programs];
    const encoder = this.device.createCommandEncoder({ label: 'genie3.run' });
    const pass = encoder.beginComputePass({ label: 'genie3.compute' });
    for (const prog of list) {
      if (!prog) continue;
      const steps = prog.steps ?? [prog];
      for (const s of steps) {
        pass.setPipeline(s.pipeline);
        pass.setBindGroup(0, s.bindGroup);
        pass.dispatchWorkgroups(s.x, s.y);
      }
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  _ensureReadback(bytes) {
    const aligned = Math.max(16, Math.ceil(bytes / 16) * 16);
    if (!this.readbackBuf || this.readbackSize < aligned) {
      if (this.readbackBuf) this.readbackBuf.destroy();
      this.readbackBuf = this.device.createBuffer({
        label: 'genie3.readback',
        size: aligned,
        usage: MAP_READ | COPY_DST,
      });
      this.readbackSize = aligned;
    }
    return this.readbackBuf;
  }

  /** Run programs and read back `floats` Float32 values from `outBuf` (single submit + single mapAsync). */
  async runAndRead(programs, outBuf, floats, offsetFloats = 0) {
    const bytes = floats * 4;
    const rb = this._ensureReadback(bytes);
    const list = Array.isArray(programs) ? programs : [programs];
    const encoder = this.device.createCommandEncoder({ label: 'genie3.runAndRead' });
    const pass = encoder.beginComputePass({ label: 'genie3.compute' });
    for (const prog of list) {
      if (!prog) continue;
      const steps = prog.steps ?? [prog];
      for (const s of steps) {
        pass.setPipeline(s.pipeline);
        pass.setBindGroup(0, s.bindGroup);
        pass.dispatchWorkgroups(s.x, s.y);
      }
    }
    pass.end();
    encoder.copyBufferToBuffer(outBuf, offsetFloats * 4, rb, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await rb.mapAsync(MAP_MODE_READ, 0, bytes);
    const copy = new Float32Array(rb.getMappedRange(0, bytes).slice(0));
    rb.unmap();
    return copy;
  }

  /** Run programs and read back multiple named buffers in a single submit. */
  async runAndReadMany(programs, targets) {
    const list = Array.isArray(programs) ? programs : [programs];
    const encoder = this.device.createCommandEncoder({ label: 'genie3.runAndReadMany' });
    const pass = encoder.beginComputePass({ label: 'genie3.compute' });
    for (const prog of list) {
      if (!prog) continue;
      const steps = prog.steps ?? [prog];
      for (const s of steps) {
        pass.setPipeline(s.pipeline);
        pass.setBindGroup(0, s.bindGroup);
        pass.dispatchWorkgroups(s.x, s.y);
      }
    }
    pass.end();
    const totalBytes = targets.reduce((acc, t) => acc + Math.ceil((t.floats * 4) / 4) * 4, 0);
    const rb = this._ensureReadback(totalBytes);
    const slices = [];
    let off = 0;
    for (const t of targets) {
      const bytes = t.floats * 4;
      encoder.copyBufferToBuffer(t.buffer, (t.offsetFloats ?? 0) * 4, rb, off, bytes);
      slices.push({ key: t.key, off, bytes });
      off += Math.ceil(bytes / 4) * 4;
    }
    this.device.queue.submit([encoder.finish()]);
    await rb.mapAsync(MAP_MODE_READ, 0, totalBytes);
    const mapped = rb.getMappedRange(0, totalBytes);
    const result = {};
    for (const sl of slices) {
      result[sl.key] = new Float32Array(mapped.slice(sl.off, sl.off + sl.bytes));
    }
    rb.unmap();
    return result;
  }

  /** Read back a single buffer directly (e.g., for debug inspection). */
  async read(buf, floats, offsetFloats = 0) {
    return this.runAndRead([], buf, floats, offsetFloats);
  }

  /** Release activation and uniform buffers (called when resizing to a new sequence length). */
  releaseActivations() {
    for (const buf of this.allocated) buf.destroy();
    this.allocated.length = 0;
    this.bytes = 0;
  }

  /** Release ALL GPU resources owned by this engine (activations, weights, readback, dummy). */
  destroy() {
    this.releaseActivations();
    for (const buf of this.weightBuffers) buf.destroy();
    this.weightBuffers.length = 0;
    this.weightBytes = 0;
    if (this.readbackBuf) {
      this.readbackBuf.destroy();
      this.readbackBuf = null;
      this.readbackSize = 0;
    }
    if (this.dummy) {
      this.dummy.destroy();
      this.dummy = null;
    }
  }
}
