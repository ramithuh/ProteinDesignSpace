import { formatGpuLossDetail } from './resources.js';
import { QueuePacer } from '../../localfold/webgpu/runtime/queue-pacing.js';
import { yieldToBrowser } from '../../localfold/webgpu/runtime/yield.js';

// Eager WGSL tensor operations. All activation arithmetic stays on the GPU.
// Scratch buffers are reused only after their last encoded use; host uploads are
// never pooled, so queue.writeBuffer cannot overwrite earlier pending commands.
const DEFAULT_MAX_POOL_BYTES = 256 * 1024 * 1024;
const UNIFORM_SLAB_BYTES = 262144;
const UNIFORM_SLOT_BYTES = 256;
const SLOTS_PER_SLAB = UNIFORM_SLAB_BYTES / UNIFORM_SLOT_BYTES;

const UNPACK_I8_FN = `fn unpack4i8(word: u32) -> vec4<f32> {
  let s = bitcast<i32>(word);
  return vec4<f32>(
    f32((s << 24u) >> 24u),
    f32((s << 16u) >> 24u),
    f32((s << 8u) >> 24u),
    f32(s >> 24u)
  );
}`;

function foldGrid2D(groups) {
  const g = Math.max(1, groups);
  return g <= 65535 ? [g, 1] : [65535, Math.ceil(g / 65535)];
}

/**
 * Uniform-parameterized register-tiled 64x128 / 32x64 GEMM for RFD3 linear layers.
 * Eliminates per-(m, n, k) synchronous shader compilation, avoids >8 KiB workgroup
 * array init penalties on Firefox/Naga (every panel <= 4,352 B), and stages F16 weights
 * in half-precision workgroup panels when shader-f16 is active while accumulating in FP32.
 */
export function buildRfd3LinearGemmShader({
  large = false,
  dtype = 0,
  hasBias = false,
  reluInput = false,
  alignedB = false,
  useF16Panel = false,
} = {}) {
  const enableF16 = Boolean(dtype === 1 && alignedB && useF16Panel);
  const EB = enableF16 ? 'f16' : 'f32';
  const wElemType = dtype === 0 ? 'f32' : 'u32';
  let bindIdx = 2;
  const scalesBind = dtype === 2 ? bindIdx++ : -1;
  const biasBind = hasBias ? bindIdx++ : -1;
  const outBind = bindIdx++;
  const uniBind = bindIdx;

  const readUnalignedW =
    dtype === 0
      ? 'fn readW(row: u32, gk: u32, K: u32) -> f32 { return w[row * K + gk]; }'
      : dtype === 1
        ? 'fn readW(row: u32, gk: u32, K: u32) -> f32 { let idx = row * K + gk; let p = unpack2x16float(w[idx >> 1u]); return select(p.x, p.y, (idx & 1u) != 0u); }'
        : 'fn readW(row: u32, gk: u32, K: u32) -> f32 { let idx = row * K + gk; let b = (w[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u; return f32(bitcast<i32>(b << 24u) >> 24u) * scales[row]; }';

  const header = [
    enableF16 ? 'enable f16;' : '',
    dtype === 2 && alignedB ? UNPACK_I8_FN : '',
    !alignedB ? readUnalignedW : '',
    'struct GemmDims { m: u32, n: u32, k: u32, tilesN: u32, totalTiles: u32, pad0: u32, pad1: u32, pad2: u32 };',
    '@group(0) @binding(0) var<storage, read> x: array<f32>;',
    `@group(0) @binding(1) var<storage, read> w: array<${wElemType}>;`,
    scalesBind >= 0 ? `@group(0) @binding(${scalesBind}) var<storage, read> scales: array<f32>;` : '',
    biasBind >= 0 ? `@group(0) @binding(${biasBind}) var<storage, read> bias: array<f32>;` : '',
    `@group(0) @binding(${outBind}) var<storage, read_write> out: array<f32>;`,
    `@group(0) @binding(${uniBind}) var<uniform> d: GemmDims;`,
  ]
    .filter(Boolean)
    .join('\n');

  if (large) {
    // 64x128 register-tiled GEMM (256 threads, 4x8 per thread, BK=16)
    // Split B into tb0/tb1 (272 vec4s = 4,352 B each < 8,192 B Naga threshold; total <= 13,056 B < 16 KiB)
    const loadBSetup =
      dtype === 1 && alignedB
        ? `let kkB = (li % 8u) * 2u;
  let r4B = li / 8u;
  let rowB0 = n0 + r4B * 4u;
  let sBnHalf = d.k >> 1u;
  let bBase0 = (rowB0 * d.k + kkB) >> 1u;
  let fullB = rowB0 + 3u < d.n;`
        : dtype === 2 && alignedB
          ? `let kkB = (li % 4u) * 4u;
  let r4B = li / 4u;
  let rowB0 = n0 + r4B * 4u;
  let sBnQuad = d.k >> 2u;
  let bBase0 = (rowB0 * d.k + kkB) >> 2u;
  let fullB = rowB0 + 3u < d.n;
  var scB = vec4<f32>(0.0);
  if (li < 128u) {
    if (rowB0 < d.n) { scB.x = scales[rowB0]; }
    if (rowB0 + 1u < d.n) { scB.y = scales[rowB0 + 1u]; }
    if (rowB0 + 2u < d.n) { scB.z = scales[rowB0 + 2u]; }
    if (rowB0 + 3u < d.n) { scB.w = scales[rowB0 + 3u]; }
  }`
          : '';

    const loadBLoop =
      dtype === 1 && alignedB
        ? `{
      var lo = vec4<${EB}>(0.0);
      var hi = vec4<${EB}>(0.0);
      if (inTile && k0 + kkB < d.k) {
        let bOff = bBase0 + (k0 >> 1u);
        if (fullB) {
          let p0 = unpack2x16float(w[bOff]);
          let p1 = unpack2x16float(w[bOff + sBnHalf]);
          let p2 = unpack2x16float(w[bOff + 2u * sBnHalf]);
          let p3 = unpack2x16float(w[bOff + 3u * sBnHalf]);
          lo = vec4<${EB}>(${EB}(p0.x), ${EB}(p1.x), ${EB}(p2.x), ${EB}(p3.x));
          hi = vec4<${EB}>(${EB}(p0.y), ${EB}(p1.y), ${EB}(p2.y), ${EB}(p3.y));
        } else {
          if (rowB0 < d.n) { let p0 = unpack2x16float(w[bOff]); lo.x = ${EB}(p0.x); hi.x = ${EB}(p0.y); }
          if (rowB0 + 1u < d.n) { let p1 = unpack2x16float(w[bOff + sBnHalf]); lo.y = ${EB}(p1.x); hi.y = ${EB}(p1.y); }
          if (rowB0 + 2u < d.n) { let p2 = unpack2x16float(w[bOff + 2u * sBnHalf]); lo.z = ${EB}(p2.x); hi.z = ${EB}(p2.y); }
          if (rowB0 + 3u < d.n) { let p3 = unpack2x16float(w[bOff + 3u * sBnHalf]); lo.w = ${EB}(p3.x); hi.w = ${EB}(p3.y); }
        }
      }
      if (r4B < 16u) {
        tb0[kkB * 17u + r4B] = lo;
        tb0[(kkB + 1u) * 17u + r4B] = hi;
      } else {
        let r4H = r4B - 16u;
        tb1[kkB * 17u + r4H] = lo;
        tb1[(kkB + 1u) * 17u + r4H] = hi;
      }
    }`
        : dtype === 2 && alignedB
          ? `if (li < 128u) {
      var q0 = vec4<f32>(0.0); var q1 = vec4<f32>(0.0);
      var q2 = vec4<f32>(0.0); var q3 = vec4<f32>(0.0);
      if (inTile && k0 + kkB < d.k) {
        let bOff = bBase0 + (k0 >> 2u);
        if (fullB) {
          q0 = unpack4i8(w[bOff]) * scB.x;
          q1 = unpack4i8(w[bOff + sBnQuad]) * scB.y;
          q2 = unpack4i8(w[bOff + 2u * sBnQuad]) * scB.z;
          q3 = unpack4i8(w[bOff + 3u * sBnQuad]) * scB.w;
        } else {
          if (rowB0 < d.n) { q0 = unpack4i8(w[bOff]) * scB.x; }
          if (rowB0 + 1u < d.n) { q1 = unpack4i8(w[bOff + sBnQuad]) * scB.y; }
          if (rowB0 + 2u < d.n) { q2 = unpack4i8(w[bOff + 2u * sBnQuad]) * scB.z; }
          if (rowB0 + 3u < d.n) { q3 = unpack4i8(w[bOff + 3u * sBnQuad]) * scB.w; }
        }
      }
      let c0 = vec4<f32>(q0.x, q1.x, q2.x, q3.x);
      let c1 = vec4<f32>(q0.y, q1.y, q2.y, q3.y);
      let c2 = vec4<f32>(q0.z, q1.z, q2.z, q3.z);
      let c3 = vec4<f32>(q0.w, q1.w, q2.w, q3.w);
      if (r4B < 16u) {
        tb0[(kkB + 0u) * 17u + r4B] = c0;
        tb0[(kkB + 1u) * 17u + r4B] = c1;
        tb0[(kkB + 2u) * 17u + r4B] = c2;
        tb0[(kkB + 3u) * 17u + r4B] = c3;
      } else {
        let r4H = r4B - 16u;
        tb1[(kkB + 0u) * 17u + r4H] = c0;
        tb1[(kkB + 1u) * 17u + r4H] = c1;
        tb1[(kkB + 2u) * 17u + r4H] = c2;
        tb1[(kkB + 3u) * 17u + r4H] = c3;
      }
    }`
          : `{
      let kk = li % 16u;
      let r4 = li / 16u;
      let gk = k0 + kk;
      var v0 = vec4<f32>(0.0);
      var v1 = vec4<f32>(0.0);
      if (inTile && gk < d.k) {
        let row0 = n0 + r4 * 4u;
        if (row0 + 3u < d.n) {
          v0 = vec4<f32>(readW(row0, gk, d.k), readW(row0 + 1u, gk, d.k), readW(row0 + 2u, gk, d.k), readW(row0 + 3u, gk, d.k));
        } else {
          if (row0 < d.n) { v0.x = readW(row0, gk, d.k); }
          if (row0 + 1u < d.n) { v0.y = readW(row0 + 1u, gk, d.k); }
          if (row0 + 2u < d.n) { v0.z = readW(row0 + 2u, gk, d.k); }
          if (row0 + 3u < d.n) { v0.w = readW(row0 + 3u, gk, d.k); }
        }
        let row1 = row0 + 64u;
        if (row1 + 3u < d.n) {
          v1 = vec4<f32>(readW(row1, gk, d.k), readW(row1 + 1u, gk, d.k), readW(row1 + 2u, gk, d.k), readW(row1 + 3u, gk, d.k));
        } else {
          if (row1 < d.n) { v1.x = readW(row1, gk, d.k); }
          if (row1 + 1u < d.n) { v1.y = readW(row1 + 1u, gk, d.k); }
          if (row1 + 2u < d.n) { v1.z = readW(row1 + 2u, gk, d.k); }
          if (row1 + 3u < d.n) { v1.w = readW(row1 + 3u, gk, d.k); }
        }
      }
      tb0[kk * 17u + r4] = v0;
      tb1[kk * 17u + r4] = v1;
    }`;

    const fmaBody = Array.from(
      { length: 16 },
      (_, kk) => `{
      let av = ta[${kk * 17}u + tm4];
      let bv0 = vec4<f32>(tb0[${kk * 17}u + tx]);
      let bv1 = vec4<f32>(tb1[${kk * 17}u + tx]);
      acc_0_0 = fma(vec4<f32>(av.x), bv0, acc_0_0); acc_0_1 = fma(vec4<f32>(av.x), bv1, acc_0_1);
      acc_1_0 = fma(vec4<f32>(av.y), bv0, acc_1_0); acc_1_1 = fma(vec4<f32>(av.y), bv1, acc_1_1);
      acc_2_0 = fma(vec4<f32>(av.z), bv0, acc_2_0); acc_2_1 = fma(vec4<f32>(av.z), bv1, acc_2_1);
      acc_3_0 = fma(vec4<f32>(av.w), bv0, acc_3_0); acc_3_1 = fma(vec4<f32>(av.w), bv1, acc_3_1);
    }`,
    ).join('\n    ');

    const storeRows = [0, 1, 2, 3]
      .map(
        (i) => `{
    let m = m0 + tm + ${i}u;
    if (m < d.m) {
      let r0 = acc_${i}_0${hasBias ? ' + bVec0' : ''};
      let r1 = acc_${i}_1${hasBias ? ' + bVec1' : ''};
      let rowOut = m * d.n;
      if (col0 < d.n) { out[rowOut + col0] = r0.x; }
      if (col0 + 1u < d.n) { out[rowOut + col0 + 1u] = r0.y; }
      if (col0 + 2u < d.n) { out[rowOut + col0 + 2u] = r0.z; }
      if (col0 + 3u < d.n) { out[rowOut + col0 + 3u] = r0.w; }
      if (col1 < d.n) { out[rowOut + col1] = r1.x; }
      if (col1 + 1u < d.n) { out[rowOut + col1 + 1u] = r1.y; }
      if (col1 + 2u < d.n) { out[rowOut + col1 + 2u] = r1.z; }
      if (col1 + 3u < d.n) { out[rowOut + col1 + 3u] = r1.w; }
    }
  }`,
      )
      .join('\n  ');

    return {
      uniBind,
      code: `${header}
var<workgroup> ta: array<vec4<f32>, 272>;
var<workgroup> tb0: array<vec4<${EB}>, 272>;
var<workgroup> tb1: array<vec4<${EB}>, 272>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let flat = wid.x + wid.y * 65535u;
  let inTile = flat < d.totalTiles;
  let tile = min(flat, d.totalTiles - 1u);
  let m0 = (tile / d.tilesN) * 64u;
  let n0 = (tile % d.tilesN) * 128u;
  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * 4u;
  var acc_0_0 = vec4<f32>(0.0); var acc_0_1 = vec4<f32>(0.0);
  var acc_1_0 = vec4<f32>(0.0); var acc_1_1 = vec4<f32>(0.0);
  var acc_2_0 = vec4<f32>(0.0); var acc_2_1 = vec4<f32>(0.0);
  var acc_3_0 = vec4<f32>(0.0); var acc_3_1 = vec4<f32>(0.0);
  let kkA = li % 16u;
  let r4A = li / 16u;
  let rowA0 = m0 + r4A * 4u;
  let fullA = rowA0 + 3u < d.m;
  let taWrite = kkA * 17u + r4A;
  ${loadBSetup}
  for (var k0 = 0u; k0 < d.k; k0 += 16u) {
    {
      var av = vec4<f32>(0.0);
      if (inTile && k0 + kkA < d.k) {
        let aOff = rowA0 * d.k + k0 + kkA;
        if (fullA) {
          av = vec4<f32>(x[aOff], x[aOff + d.k], x[aOff + 2u * d.k], x[aOff + 3u * d.k]);
        } else {
          if (rowA0 < d.m) { av.x = x[aOff]; }
          if (rowA0 + 1u < d.m) { av.y = x[aOff + d.k]; }
          if (rowA0 + 2u < d.m) { av.z = x[aOff + 2u * d.k]; }
          if (rowA0 + 3u < d.m) { av.w = x[aOff + 3u * d.k]; }
        }
        ${reluInput ? 'av = max(av, vec4<f32>(0.0));' : ''}
      }
      ta[taWrite] = av;
    }
    ${loadBLoop}
    workgroupBarrier();
    ${fmaBody}
    workgroupBarrier();
  }
  if (!inTile) { return; }
  let col0 = n0 + tx * 4u;
  let col1 = col0 + 64u;
  ${hasBias ? `var bVec0 = vec4<f32>(0.0); var bVec1 = vec4<f32>(0.0);
  if (col0 < d.n) { bVec0.x = bias[col0]; }
  if (col0 + 1u < d.n) { bVec0.y = bias[col0 + 1u]; }
  if (col0 + 2u < d.n) { bVec0.z = bias[col0 + 2u]; }
  if (col0 + 3u < d.n) { bVec0.w = bias[col0 + 3u]; }
  if (col1 < d.n) { bVec1.x = bias[col1]; }
  if (col1 + 1u < d.n) { bVec1.y = bias[col1 + 1u]; }
  if (col1 + 2u < d.n) { bVec1.z = bias[col1 + 2u]; }
  if (col1 + 3u < d.n) { bVec1.w = bias[col1 + 3u]; }` : ''}
  ${storeRows}
}`,
    };
  }

  // 32x64 register-tiled GEMM for smaller shapes / high occupancy (256 threads, 2x4 per thread, BK=16)
  const loadBSetupSmall =
    dtype === 1 && alignedB
      ? `let kkB = (li % 8u) * 2u;
  let r4B = li / 8u;
  let rowB0 = n0 + r4B * 4u;
  let sBnHalf = d.k >> 1u;
  let bBase0 = (rowB0 * d.k + kkB) >> 1u;
  let fullB = rowB0 + 3u < d.n;
  let tbWrite0 = kkB * 17u + r4B;
  let tbWrite1 = (kkB + 1u) * 17u + r4B;`
      : dtype === 2 && alignedB
        ? `let kkB = (li % 4u) * 4u;
  let r4B = li / 4u;
  let rowB0 = n0 + r4B * 4u;
  let sBnQuad = d.k >> 2u;
  let bBase0 = (rowB0 * d.k + kkB) >> 2u;
  let fullB = rowB0 + 3u < d.n;
  var scB = vec4<f32>(0.0);
  if (li < 64u) {
    if (rowB0 < d.n) { scB.x = scales[rowB0]; }
    if (rowB0 + 1u < d.n) { scB.y = scales[rowB0 + 1u]; }
    if (rowB0 + 2u < d.n) { scB.z = scales[rowB0 + 2u]; }
    if (rowB0 + 3u < d.n) { scB.w = scales[rowB0 + 3u]; }
  }`
        : `let kkB = li % 16u;
  let r4B = li / 16u;
  let rowB0 = n0 + r4B * 4u;
  let fullB = rowB0 + 3u < d.n;
  let tbWrite = kkB * 17u + r4B;`;

  const loadBLoopSmall =
    dtype === 1 && alignedB
      ? `if (li < 128u) {
      var lo = vec4<${EB}>(0.0);
      var hi = vec4<${EB}>(0.0);
      if (inTile && k0 + kkB < d.k) {
        let bOff = bBase0 + (k0 >> 1u);
        if (fullB) {
          let p0 = unpack2x16float(w[bOff]);
          let p1 = unpack2x16float(w[bOff + sBnHalf]);
          let p2 = unpack2x16float(w[bOff + 2u * sBnHalf]);
          let p3 = unpack2x16float(w[bOff + 3u * sBnHalf]);
          lo = vec4<${EB}>(${EB}(p0.x), ${EB}(p1.x), ${EB}(p2.x), ${EB}(p3.x));
          hi = vec4<${EB}>(${EB}(p0.y), ${EB}(p1.y), ${EB}(p2.y), ${EB}(p3.y));
        } else {
          if (rowB0 < d.n) { let p0 = unpack2x16float(w[bOff]); lo.x = ${EB}(p0.x); hi.x = ${EB}(p0.y); }
          if (rowB0 + 1u < d.n) { let p1 = unpack2x16float(w[bOff + sBnHalf]); lo.y = ${EB}(p1.x); hi.y = ${EB}(p1.y); }
          if (rowB0 + 2u < d.n) { let p2 = unpack2x16float(w[bOff + 2u * sBnHalf]); lo.z = ${EB}(p2.x); hi.z = ${EB}(p2.y); }
          if (rowB0 + 3u < d.n) { let p3 = unpack2x16float(w[bOff + 3u * sBnHalf]); lo.w = ${EB}(p3.x); hi.w = ${EB}(p3.y); }
        }
      }
      tb[tbWrite0] = lo;
      tb[tbWrite1] = hi;
    }`
      : dtype === 2 && alignedB
        ? `if (li < 64u) {
      var q0 = vec4<f32>(0.0); var q1 = vec4<f32>(0.0);
      var q2 = vec4<f32>(0.0); var q3 = vec4<f32>(0.0);
      if (inTile && k0 + kkB < d.k) {
        let bOff = bBase0 + (k0 >> 2u);
        if (fullB) {
          q0 = unpack4i8(w[bOff]) * scB.x;
          q1 = unpack4i8(w[bOff + sBnQuad]) * scB.y;
          q2 = unpack4i8(w[bOff + 2u * sBnQuad]) * scB.z;
          q3 = unpack4i8(w[bOff + 3u * sBnQuad]) * scB.w;
        } else {
          if (rowB0 < d.n) { q0 = unpack4i8(w[bOff]) * scB.x; }
          if (rowB0 + 1u < d.n) { q1 = unpack4i8(w[bOff + sBnQuad]) * scB.y; }
          if (rowB0 + 2u < d.n) { q2 = unpack4i8(w[bOff + 2u * sBnQuad]) * scB.z; }
          if (rowB0 + 3u < d.n) { q3 = unpack4i8(w[bOff + 3u * sBnQuad]) * scB.w; }
        }
      }
      tb[(kkB + 0u) * 17u + r4B] = vec4<f32>(q0.x, q1.x, q2.x, q3.x);
      tb[(kkB + 1u) * 17u + r4B] = vec4<f32>(q0.y, q1.y, q2.y, q3.y);
      tb[(kkB + 2u) * 17u + r4B] = vec4<f32>(q0.z, q1.z, q2.z, q3.z);
      tb[(kkB + 3u) * 17u + r4B] = vec4<f32>(q0.w, q1.w, q2.w, q3.w);
    }`
        : `{
      var bv = vec4<f32>(0.0);
      let gk = k0 + kkB;
      if (inTile && gk < d.k) {
        if (fullB) {
          bv = vec4<f32>(readW(rowB0, gk, d.k), readW(rowB0 + 1u, gk, d.k), readW(rowB0 + 2u, gk, d.k), readW(rowB0 + 3u, gk, d.k));
        } else {
          if (rowB0 < d.n) { bv.x = readW(rowB0, gk, d.k); }
          if (rowB0 + 1u < d.n) { bv.y = readW(rowB0 + 1u, gk, d.k); }
          if (rowB0 + 2u < d.n) { bv.z = readW(rowB0 + 2u, gk, d.k); }
          if (rowB0 + 3u < d.n) { bv.w = readW(rowB0 + 3u, gk, d.k); }
        }
      }
      tb[tbWrite] = bv;
    }`;

  const fmaSmall = Array.from(
    { length: 16 },
    (_, kk) => `{
      let av4 = ta[${kk * 9}u + r4Read];
      let av = select(av4.xy, av4.zw, hiPair);
      let bv = vec4<f32>(tb[${kk * 17}u + tx]);
      acc0 = fma(vec4<f32>(av.x), bv, acc0);
      acc1 = fma(vec4<f32>(av.y), bv, acc1);
    }`,
  ).join('\n    ');

  return {
    uniBind,
    code: `${header}
var<workgroup> ta: array<vec4<f32>, 144>;
var<workgroup> tb: array<vec4<${EB}>, 272>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let flat = wid.x + wid.y * 65535u;
  let inTile = flat < d.totalTiles;
  let tile = min(flat, d.totalTiles - 1u);
  let m0 = (tile / d.tilesN) * 32u;
  let n0 = (tile % d.tilesN) * 64u;
  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * 2u;
  let r4Read = tm4 >> 1u;
  let hiPair = (tm4 & 1u) != 0u;
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  let kkA = li % 16u;
  let r4A = li / 16u;
  let rowA0 = m0 + r4A * 4u;
  let fullA = rowA0 + 3u < d.m;
  let taWrite = kkA * 9u + r4A;
  ${loadBSetupSmall}
  for (var k0 = 0u; k0 < d.k; k0 += 16u) {
    if (li < 128u) {
      var av = vec4<f32>(0.0);
      if (inTile && k0 + kkA < d.k) {
        let aOff = rowA0 * d.k + k0 + kkA;
        if (fullA) {
          av = vec4<f32>(x[aOff], x[aOff + d.k], x[aOff + 2u * d.k], x[aOff + 3u * d.k]);
        } else {
          if (rowA0 < d.m) { av.x = x[aOff]; }
          if (rowA0 + 1u < d.m) { av.y = x[aOff + d.k]; }
          if (rowA0 + 2u < d.m) { av.z = x[aOff + 2u * d.k]; }
          if (rowA0 + 3u < d.m) { av.w = x[aOff + 3u * d.k]; }
        }
        ${reluInput ? 'av = max(av, vec4<f32>(0.0));' : ''}
      }
      ta[taWrite] = av;
    }
    ${loadBLoopSmall}
    workgroupBarrier();
    ${fmaSmall}
    workgroupBarrier();
  }
  if (!inTile) { return; }
  let col0 = n0 + tx * 4u;
  ${hasBias ? `var bVec = vec4<f32>(0.0);
  if (col0 < d.n) { bVec.x = bias[col0]; }
  if (col0 + 1u < d.n) { bVec.y = bias[col0 + 1u]; }
  if (col0 + 2u < d.n) { bVec.z = bias[col0 + 2u]; }
  if (col0 + 3u < d.n) { bVec.w = bias[col0 + 3u]; }` : ''}
  let r0 = m0 + tm;
  if (r0 < d.m) {
    let res0 = acc0${hasBias ? ' + bVec' : ''};
    let rowOut0 = r0 * d.n;
    if (col0 < d.n) { out[rowOut0 + col0] = res0.x; }
    if (col0 + 1u < d.n) { out[rowOut0 + col0 + 1u] = res0.y; }
    if (col0 + 2u < d.n) { out[rowOut0 + col0 + 2u] = res0.z; }
    if (col0 + 3u < d.n) { out[rowOut0 + col0 + 3u] = res0.w; }
  }
  let r1 = r0 + 1u;
  if (r1 < d.m) {
    let res1 = acc1${hasBias ? ' + bVec' : ''};
    let rowOut1 = r1 * d.n;
    if (col0 < d.n) { out[rowOut1 + col0] = res1.x; }
    if (col0 + 1u < d.n) { out[rowOut1 + col0 + 1u] = res1.y; }
    if (col0 + 2u < d.n) { out[rowOut1 + col0 + 2u] = res1.z; }
    if (col0 + 3u < d.n) { out[rowOut1 + col0 + 3u] = res1.w; }
  }
}`,
  };
}

export class GPUOps {
  constructor(resources, { maxPoolBytes = DEFAULT_MAX_POOL_BYTES } = {}) {
    this.owner = resources;
    this.device = resources.device;
    this.hasF16 = Boolean(resources.hasF16 || resources.device?.features?.has?.('shader-f16'));
    this.pacers = new Map();
    this.pacerPhase = 'default';
    this.pacer = new QueuePacer(this.device);
    this.pacers.set('default', this.pacer);
    this.pieceUsed = false;
    this.pendingDispatches = 0;
    this.pipelines = new Map();
    this.pool = new Map();
    this.poolBytes = 0;
    this.maxPoolBytes = maxPoolBytes;
    this.deferredReleases = [];
    this.uniformSlabs = [];
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
    this.tensors = new Set();
    this.scopes = [];
    this.encoder = null;
    this.pass = null;
  }
  begin() {
    if (this.encoder) throw new Error('GPU command transaction already active');
    this.encoder = this.device.createCommandEncoder();
    this.pass = null;
    this.pieceUsed = false;
    this.pendingDispatches = 0;
  }
  allocUniform(words) {
    if (this.uniformSlotIdx >= SLOTS_PER_SLAB) {
      this.uniformSlabIdx++;
      this.uniformSlotIdx = 0;
    }
    while (this.uniformSlabs.length <= this.uniformSlabIdx) {
      const buffer = this.owner.allocate(UNIFORM_SLAB_BYTES, 64 | 8, 'RFD3 uniforms');
      this.uniformSlabs.push({
        buffer,
        cpu: new Uint32Array(UNIFORM_SLAB_BYTES / 4),
        usedWords: 0,
      });
    }
    const slab = this.uniformSlabs[this.uniformSlabIdx];
    const byteOffset = this.uniformSlotIdx * UNIFORM_SLOT_BYTES;
    const wordOffset = byteOffset >>> 2;
    slab.cpu.set(words, wordOffset);
    slab.usedWords = wordOffset + words.length;
    this.uniformSlotIdx++;
    return { buffer: slab.buffer, offset: byteOffset, size: words.byteLength };
  }
  flushUniforms() {
    const limit = Math.min(this.uniformSlabIdx + 1, this.uniformSlabs.length);
    for (let i = 0; i < limit; i++) {
      const slab = this.uniformSlabs[i];
      if (slab.usedWords > 0) {
        this.device.queue.writeBuffer(slab.buffer, 0, slab.cpu.subarray(0, slab.usedWords));
        slab.usedWords = 0;
      }
    }
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
  }
  releaseUniformSlabs() {
    for (const slab of this.uniformSlabs) {
      this.owner.release(slab.buffer);
    }
    this.uniformSlabs.length = 0;
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
  }
  endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }
  releaseBuffer(buffer) {
    if (this.encoder) this.deferredReleases.push(buffer);
    else this.owner.release(buffer);
  }
  drainDeferredReleases() {
    while (this.deferredReleases.length) {
      this.owner.release(this.deferredReleases.pop());
    }
  }
  evictOnePooled() {
    for (const [size, bucket] of this.pool.entries()) {
      if (bucket.length) {
        const buf = bucket.pop();
        this.poolBytes -= buf.size;
        if (!bucket.length) this.pool.delete(size);
        this.releaseBuffer(buf);
        return true;
      }
      this.pool.delete(size);
    }
    return false;
  }
  tensor(length, host = false) {
    if (!Number.isSafeInteger(length) || length < 1 || length * 4 > this.device.limits.maxStorageBufferBindingSize)
      throw new RangeError('Tensor exceeds the negotiated binding limit');
    const size = length * 4;
    const bucket = !host ? this.pool.get(size) : undefined;
    let buffer;
    if (bucket?.length) {
      buffer = bucket.pop();
      this.poolBytes -= buffer.size;
      if (!bucket.length) this.pool.delete(size);
    } else {
      if (!host && size <= this.maxPoolBytes) {
        while (this.poolBytes + size > this.maxPoolBytes) {
          if (!this.evictOnePooled()) break;
        }
      }
      buffer = this.owner.allocate(size, 128 | 8 | 4, 'RFD3 activation');
    }
    const t = { buffer, length, host, released: false };
    this.tensors.add(t);
    for (const scope of this.scopes) scope.push(t);
    return t;
  }
  upload(data) {
    const t = this.tensor(data.length, true);
    this.device.queue.writeBuffer(t.buffer, 0, Float32Array.from(data));
    return t;
  }
  free(t) {
    if (!t || t.released || !this.tensors.delete(t)) return;
    t.released = true;
    if (t.host || t.buffer.size > this.maxPoolBytes || this.maxPoolBytes <= 0) {
      this.releaseBuffer(t.buffer);
      return;
    }
    while (this.poolBytes + t.buffer.size > this.maxPoolBytes) {
      if (!this.evictOnePooled()) break;
    }
    if (this.poolBytes + t.buffer.size <= this.maxPoolBytes) {
      let b = this.pool.get(t.buffer.size);
      if (!b) this.pool.set(t.buffer.size, b = []);
      b.push(t.buffer);
      this.poolBytes += t.buffer.size;
    } else {
      this.releaseBuffer(t.buffer);
    }
  }
  scope(fn) {
    const created = [];
    this.scopes.push(created);
    try {
      const result = fn(), keep = new Set();
      const walk = v => {
        if (!v || typeof v !== 'object') return;
        if (v.buffer && v.length !== undefined) { keep.add(v); return; }
        for (const x of Object.values(v)) walk(x);
      };
      walk(result);
      for (const t of created) if (!keep.has(t)) this.free(t);
      return result;
    } catch (error) {
      for (const t of created) this.free(t);
      throw error;
    } finally {
      this.scopes.pop();
    }
  }
  dispatch(code, buffers, groups, label = 'RFD3 op') {
    if (!this.encoder) throw new Error('No GPU command transaction');
    let pipeline = this.pipelines.get(code);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code, label });
      pipeline = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
      this.pipelines.set(code, pipeline);
    }
    const group = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((t, binding) => ({
        binding,
        resource: t && t.buffer && t.offset !== undefined ? t : { buffer: t.buffer ?? t },
      })),
    });
    if (!this.pass) this.pass = this.encoder.beginComputePass({ label });
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, group);
    this.pass.dispatchWorkgroups(...groups);
    this.pendingDispatches++;
  }
  setPacerPhase(phase = 'default') {
    this.pacerPhase = phase;
    let pacer = this.pacers.get(phase);
    if (!pacer) {
      pacer = new QueuePacer(this.device);
      this.pacers.set(phase, pacer);
    }
    this.pacer = pacer;
  }
  piece() {
    if (!this.encoder) return;
    this.pieceUsed = true;
    this.pacer.pieceEnded(() => {
      if (this.pendingDispatches > 0) this.flush();
    });
  }
  flush() {
    if (!this.encoder) return;
    this.endPass();
    this.flushUniforms();
    const encoder = this.encoder;
    this.encoder = null;
    this.pendingDispatches = 0;
    this.device.queue.submit([encoder.finish()]);
    this.drainDeferredReleases();
    this.encoder = this.device.createCommandEncoder();
  }
  abort() {
    try { this.endPass(); } catch { /* ignore */ }
    this.encoder = null;
    this.pass = null;
    this.pendingDispatches = 0;
    this.pieceUsed = false;
    this.pacers.clear();
    this.pacer = new QueuePacer(this.device);
    this.pacers.set(this.pacerPhase, this.pacer);
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
    for (const slab of this.uniformSlabs) slab.usedWords = 0;
    this.drainDeferredReleases();
  }
  kernel(inputs, length, body, label = 'RFD3 elementwise') {
    const out = this.tensor(length), num = Math.ceil(length / 64), x = Math.min(65535, num), y = Math.ceil(num / x);
    const code = inputs.map((_, i) => `@group(0) @binding(${i}) var<storage,read> a${i}:array<f32>;`).join('\n') + `
@group(0) @binding(${inputs.length}) var<storage,read_write> out:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let idx=gid.x+gid.y*${x * 64}u;if(idx>=${length}u){return;}${body}}`;
    this.dispatch(code, [...inputs, out], [x, y], label);
    return out;
  }
  fill(length, value = 0) {
    return this.kernel([], length, `out[idx]=${Number(value).toFixed(12)};`);
  }
  add(a, b) {
    if (a.length !== b.length) throw new Error('GPU add shape mismatch');
    return this.kernel([a, b], a.length, 'out[idx]=a0[idx]+a1[idx];');
  }
  mul(a, b) {
    return this.kernel([a, b], a.length, 'out[idx]=a0[idx]*a1[idx];');
  }
  activation(a, kind) {
    const expression = { relu: 'max(v,0.0)', sigmoid: '1.0/(1.0+exp(-v))', silu: 'v/(1.0+exp(-v))' }[kind];
    if (!expression) throw new Error('Unknown activation');
    return this.kernel([a], a.length, `let v=a0[idx];out[idx]=${expression};`);
  }
  linear(a, w, bias, { reluInput = false } = {}) {
    const [n, k] = w.shape, m = a.length / k;
    if (!Number.isInteger(m)) throw new Error('GPU linear shape mismatch');
    const out = this.tensor(m * n), dtype = { F32: 0, F16: 1, I8: 2 }[w.dtype];
    if (dtype === undefined) throw new Error('Unsupported GPU linear weight dtype: ' + w.dtype);
    const alignedB = dtype === 1 ? k % 2 === 0 : dtype === 2 ? k % 4 === 0 : false;
    const useF16Panel = Boolean(dtype === 1 && alignedB && this.hasF16);
    const large = m >= 48 && n >= 96 && Math.ceil(m / 64) * Math.ceil(n / 128) >= 16;
    const bm = large ? 64 : 32;
    const bn = large ? 128 : 64;
    const tilesM = Math.ceil(m / bm);
    const tilesN = Math.ceil(n / bn);
    const totalTiles = tilesM * tilesN;
    const { code } = buildRfd3LinearGemmShader({
      large,
      dtype,
      hasBias: Boolean(bias),
      reluInput: Boolean(reluInput),
      alignedB,
      useF16Panel,
    });
    const uniformSlice = this.allocUniform(new Uint32Array([m, n, k, tilesN, totalTiles, 0, 0, 0]));
    const inputs = [a, w];
    if (dtype === 2) inputs.push(w.scales);
    if (bias) inputs.push(bias);
    inputs.push(out, uniformSlice);
    this.dispatch(code, inputs, foldGrid2D(totalTiles), 'RFD3 packed linear');
    return out;
  }
  fusedPosition(pos, mask, wA, bA, wB, bB, { sinusoidal = false, addTo } = {}) {
    const n = pos.length / 3, channels = wA.shape[0], total = n * n * channels;
    const dtypeA = { F32: 0, F16: 1, I8: 2 }[wA.dtype], dtypeB = { F32: 0, F16: 1, I8: 2 }[wB.dtype];
    const inputs = [pos, mask, wA, wB], decls = [
      '@group(0) @binding(0) var<storage,read> pos:array<f32>;',
      '@group(0) @binding(1) var<storage,read> mask:array<f32>;',
      '@group(0) @binding(2) var<storage,read> wA:array<u32>;',
      '@group(0) @binding(3) var<storage,read> wB:array<u32>;',
    ];
    const addBuf = (buf, name) => {
      const i = inputs.length;
      inputs.push(buf);
      decls.push(`@group(0) @binding(${i}) var<storage,read> ${name}:array<f32>;`);
    };
    if (dtypeA === 2) addBuf(wA.scales, 'sA');
    if (bA) addBuf(bA, 'bA');
    if (dtypeB === 2) addBuf(wB.scales, 'sB');
    if (bB) addBuf(bB, 'bB');
    if (addTo) addBuf(addTo, 'addIn');
    const out = this.tensor(total), num = Math.ceil(total / 64), x = Math.min(65535, num), y = Math.ceil(num / x);
    decls.push(`@group(0) @binding(${inputs.length}) var<storage,read_write> out:array<f32>;`);
    const unpack = (wName, sName, dt) =>
      dt === 0 ? `return bitcast<f32>(${wName}[i]);`
      : dt === 1 ? `return unpack2x16float(${wName}[i/2u])[i%2u];`
      : `return f32(bitcast<i32>(((${wName}[i/4u]>>(8u*(i%4u)))&255u)<<24u)>>24)*${sName}[row];`;
    const code = `${decls.join('\n')}
fn weightA(i:u32,row:u32)->f32 {${unpack('wA', 'sA', dtypeA)}}
fn weightB(i:u32,row:u32)->f32 {${unpack('wB', 'sB', dtypeB)}}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let idx=gid.x+gid.y*${x * 64}u;if(idx>=${total}u){return;}
 let pair=idx/${channels}u;let m=mask[pair];
 if(m==0.0){out[idx]=${addTo ? 'addIn[idx]' : '0.0'};return;}
 let i=pair/${n}u;let j=pair%${n}u;let col=idx%${channels}u;
 var square=0.0;for(var c=0u;c<3u;c++){let d=pos[i*3u+c]-pos[j*3u+c];square+=d*d;}
 ${sinusoidal
   ? `let dist=sqrt(square);var sumA=0.0;for(var k=0u;k<64u;k++){let angle=dist*exp(-log(10000.0)*f32(k%32u)/32.0);sumA+=select(sin(angle),cos(angle),k>=32u)*weightA(col*64u+k,col);}`
   : `var sumA=(1.0/(1.0+max(square,1e-6)))*weightA(col,col);`}
 ${bA ? 'sumA+=bA[col];' : ''}
 var sumB=m*weightB(col,col);
 ${bB ? 'sumB+=bB[col];' : ''}
 let v=(sumA+sumB)*m;
 out[idx]=${addTo ? 'addIn[idx]+v' : 'v'};
}`;
    this.dispatch(code, [...inputs, out], [x, y], 'RFD3 fused position embed');
    return out;
  }
  norm(a, channels, gain) {
    const rows = a.length / channels, out = this.tensor(a.length);
    if (!Number.isInteger(rows)) throw new Error('GPU RMSNorm shape mismatch');
    const code = `@group(0) @binding(0) var<storage,read> a:array<f32>;
${gain ? '@group(0) @binding(1) var<storage,read> gain:array<f32>;' : ''}
@group(0) @binding(${gain ? 2 : 1}) var<storage,read_write> out:array<f32>;
var<workgroup> sums:array<f32,64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){let row=g.x+g.y*65535u;if(row>=${rows}u){return;}var s=0.0;
 for(var c=id;c<${channels}u;c+=64u){let v=a[row*${channels}u+c];s+=v*v;}sums[id]=s;workgroupBarrier();
 for(var step=32u;step>0u;step/=2u){if(id<step){sums[id]+=sums[id+step];}workgroupBarrier();}
 let scale=inverseSqrt(sums[0]/${channels}.0+0.00000011920928955078125);
 for(var c=id;c<${channels}u;c+=64u){out[row*${channels}u+c]=a[row*${channels}u+c]*scale${gain ? '*gain[c]' : ''};}}`;
    this.dispatch(code, gain ? [a, gain, out] : [a, out], [Math.min(rows, 65535), Math.ceil(rows / 65535)], 'RFD3 RMSNorm');
    return out;
  }
  concat(arrays, widths, rows) {
    let offset = 0;
    const width = widths.reduce((a, b) => a + b, 0);
    const clauses = widths.map((w, i) => {
      const start = offset;
      offset += w;
      return `if(c>=${start}u&&c<${offset}u){out[idx]=a${i}[row*${w}u+c-${start}u];}`;
    });
    return this.kernel(arrays, rows * width, `let row=idx/${width}u;let c=idx%${width}u;${clauses.join('')}`);
  }
  knn(coords, forced, rows, k) {
    const out = this.tensor(rows * k);
    const code = `@group(0) @binding(0) var<storage,read> coords:array<f32>;
@group(0) @binding(1) var<storage,read> forced:array<f32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
var<workgroup> wgDistA:array<f32,128>;
var<workgroup> wgIdxA:array<u32,128>;
var<workgroup> wgDistB:array<f32,128>;
var<workgroup> wgIdxB:array<u32,128>;
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){
 let row=g.x+g.y*65535u;if(row>=${rows}u){return;}
 wgDistA[id]=3.402823e38;wgIdxA[id]=0x7fffff00u+id;
 let rx=coords[row*3u];let ry=coords[row*3u+1u];let rz=coords[row*3u+2u];
 for(var base=0u;base<${rows}u;base+=128u){
  let j=base+id;
  if(j<${rows}u){
   if(forced[row*${rows}u+j]!=0.0){wgDistB[id]= -1.0;}
   else{
    let dx=rx-coords[j*3u];let dy=ry-coords[j*3u+1u];let dz=rz-coords[j*3u+2u];
    wgDistB[id]=dx*dx+dy*dy+dz*dz;
   }
   wgIdxB[id]=j;
  }else{wgDistB[id]=3.402823e38;wgIdxB[id]=j;}
  workgroupBarrier();
  for(var span=2u;span<=128u;span*=2u){
   for(var stride=span/2u;stride>0u;stride/=2u){
    let other=id^stride;
    if(other>id){
     let desc=(id&span)!=0u;
     let di=wgDistB[id];let dj=wgDistB[other];let ii=wgIdxB[id];let ij=wgIdxB[other];
     let less=(di<dj)||(di==dj&&ii<ij);
     if(less==desc){wgDistB[id]=dj;wgDistB[other]=di;wgIdxB[id]=ij;wgIdxB[other]=ii;}
    }
    workgroupBarrier();
   }
  }
  let bd=wgDistB[127u-id];let bi=wgIdxB[127u-id];
  let ad=wgDistA[id];let ai=wgIdxA[id];
  if((bd<ad)||(bd==ad&&bi<ai)){wgDistA[id]=bd;wgIdxA[id]=bi;}
  workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){
   let other=id^stride;
   if(other>id){
    let di=wgDistA[id];let dj=wgDistA[other];let ii=wgIdxA[id];let ij=wgIdxA[other];
    if((dj<di)||(dj==di&&ij<ii)){wgDistA[id]=dj;wgDistA[other]=di;wgIdxA[id]=ij;wgIdxA[other]=ii;}
   }
   workgroupBarrier();
  }
 }
 if(id>=${k}u){wgIdxA[id]=0xffffffffu;}
 workgroupBarrier();
 for(var span=2u;span<=128u;span*=2u){
  for(var stride=span/2u;stride>0u;stride/=2u){
   let other=id^stride;
   if(other>id){
    let desc=(id&span)!=0u;
    let ii=wgIdxA[id];let ij=wgIdxA[other];
    if(ii!=ij&&(ii<ij)==desc){wgIdxA[id]=ij;wgIdxA[other]=ii;}
   }
   workgroupBarrier();
  }
 }
 if(id<${k}u){out[row*${k}u+id]=f32(wgIdxA[id]);}
}`;
    this.dispatch(code, [coords, forced, out], [Math.min(rows, 65535), Math.ceil(rows / 65535)], 'RFD3 KNN128');
    return { idx: out, k };
  }
  multiNeighboursSelect(intra, reserved, rows, k, inter) {
    const out = this.tensor(rows * rows);
    const code = `@group(0) @binding(0) var<storage,read> intra:array<f32>;
@group(0) @binding(1) var<storage,read> reserved:array<f32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
var<workgroup> wgDistA:array<f32,128>;
var<workgroup> wgIdxA:array<u32,128>;
var<workgroup> wgDistB:array<f32,128>;
var<workgroup> wgIdxB:array<u32,128>;
var<workgroup> wgResSum:array<u32,128>;
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){
 let row=g.x+g.y*65535u;if(row>=${rows}u){return;}
 wgDistA[id]=3.402823e38;wgIdxA[id]=0x7fffff00u+id;
 var rCount=0u;
 for(var base=0u;base<${rows}u;base+=128u){
  let j=base+id;
  if(j<${rows}u){
   let rv=reserved[row*${rows}u+j];
   out[row*${rows}u+j]=rv;
   rCount+=u32(rv);
   wgDistB[id]=intra[row*${rows}u+j];
   wgIdxB[id]=j;
  }else{wgDistB[id]=3.402823e38;wgIdxB[id]=j;}
  workgroupBarrier();
  for(var span=2u;span<=128u;span*=2u){
   for(var stride=span/2u;stride>0u;stride/=2u){
    let other=id^stride;
    if(other>id){
     let desc=(id&span)!=0u;
     let di=wgDistB[id];let dj=wgDistB[other];let ii=wgIdxB[id];let ij=wgIdxB[other];
     let less=(di<dj)||(di==dj&&ii<ij);
     if(less==desc){wgDistB[id]=dj;wgDistB[other]=di;wgIdxB[id]=ij;wgIdxB[other]=ii;}
    }
    workgroupBarrier();
   }
  }
  let bd=wgDistB[127u-id];let bi=wgIdxB[127u-id];
  let ad=wgDistA[id];let ai=wgIdxA[id];
  if((bd<ad)||(bd==ad&&bi<ai)){wgDistA[id]=bd;wgIdxA[id]=bi;}
  workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){
   let other=id^stride;
   if(other>id){
    let di=wgDistA[id];let dj=wgDistA[other];let ii=wgIdxA[id];let ij=wgIdxA[other];
    if((dj<di)||(dj==di&&ij<ii)){wgDistA[id]=dj;wgDistA[other]=di;wgIdxA[id]=ij;wgIdxA[other]=ii;}
   }
   workgroupBarrier();
  }
 }
 wgResSum[id]=rCount;workgroupBarrier();
 for(var s=64u;s>0u;s/=2u){if(id<s){wgResSum[id]+=wgResSum[id+s];}workgroupBarrier();}
 let resTotal=wgResSum[0];
 let col=wgIdxA[id];
 if(col<${rows}u&&(id<${k - inter}u||(id>=${k - inter}u+resTotal&&id<${k}u))){
  out[row*${rows}u+col]+=1.0;
 }
}`;
    this.dispatch(code, [intra, reserved, out], [Math.min(rows, 65535), Math.ceil(rows / 65535)], 'RFD3 multi neighbour select');
    return out;
  }
  maskToKnn(mask, rows, k, clampOne = false) {
    const idx = this.tensor(rows * k), logw = this.tensor(rows * k);
    const num = Math.ceil(rows / 64), x = Math.min(65535, num), y = Math.ceil(num / x);
    const code = `@group(0) @binding(0) var<storage,read> mask:array<f32>;
@group(0) @binding(1) var<storage,read_write> outIdx:array<f32>;
@group(0) @binding(2) var<storage,read_write> outLogw:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let row=gid.x+gid.y*${x * 64}u;if(row>=${rows}u){return;}
 var slot=0u;
 for(var j=0u;j<${rows}u;j++){
  let m=mask[row*${rows}u+j];
  if(m!=0.0&&slot<${k}u){
   outIdx[row*${k}u+slot]=f32(j);
   outLogw[row*${k}u+slot]=${clampOne ? '0.0' : 'log(m)'};
   slot++;
  }
 }
 for(;slot<${k}u;slot++){
  outIdx[row*${k}u+slot]= -1.0;
  outLogw[row*${k}u+slot]= -1e30;
 }
}`;
    this.dispatch(code, [mask, idx, logw], [x, y], 'RFD3 mask to KNN');
    return { idx, logw, k };
  }
  gatherPairs(pair, knnIdx, rows, k, channels) {
    return this.kernel([pair, knnIdx], rows * k * channels, `
 let slotIdx=idx/${channels}u;let c=idx%${channels}u;let row=slotIdx/${k}u;
 let col=i32(a1[slotIdx]);
 if(col<0){out[idx]=0.0;return;}
 out[idx]=a0[(row*${rows}u+u32(col))*${channels}u+c];`, 'RFD3 gather KNN pairs');
  }
  attention(q, k, v, { queries, keys, channels, heads, bias, mask, knn, gate, qPreScale = false }) {
    return this.scope(() => {
      const dim = channels / heads;
      if (knn) {
        const width = knn.k, inputs = [q, k, knn.idx];
        let biasIndex, logwIndex;
        if (bias) { biasIndex = inputs.length; inputs.push(bias); }
        if (knn.logw) { logwIndex = inputs.length; inputs.push(knn.logw); }
        const scores = this.kernel(inputs, queries * heads * width, `let slot=idx%${width}u;let head=(idx/${width}u)%${heads}u;let query=idx/${width * heads}u;
 let colSigned=i32(a2[query*${width}u+slot]);
 if(colSigned<0){out[idx]= -1e30;return;}
 let key=u32(colSigned);
 var sum=0.0;for(var c=0u;c<${dim}u;c++){sum+=${qPreScale ? `(a0[query*${channels}u+head*${dim}u+c]/sqrt(${dim}.0))` : `a0[query*${channels}u+head*${dim}u+c]`}*a1[key*${channels}u+head*${dim}u+c];}
 out[idx]=${qPreScale ? 'sum' : `sum/sqrt(${dim}.0)`}${bias ? `+a${biasIndex}[(query*${width}u+slot)*${heads}u+head]` : ''}${knn.logw ? `+a${logwIndex}[query*${width}u+slot]` : ''};`, 'RFD3 sparse attention scores');
        const prob = this.softmax(scores, width);
        const values = [prob, v, knn.idx];
        let gateIndex;
        if (gate) { gateIndex = values.length; values.push(gate); }
        return this.kernel(values, queries * channels, `let query=idx/${channels}u;let head=(idx%${channels}u)/${dim}u;let c=idx%${dim}u;var sum=0.0;
 for(var slot=0u;slot<${width}u;slot++){let colSigned=i32(a2[query*${width}u+slot]);if(colSigned>=0){let p=a0[(query*${heads}u+head)*${width}u+slot];if(p!=0.0){sum+=p*a1[u32(colSigned)*${channels}u+head*${dim}u+c];}}}
 out[idx]=sum${gate ? `*a${gateIndex}[idx]` : ''};`, 'RFD3 sparse attention values');
      }
      const inputs = [q, k];
      let biasIndex, maskIndex;
      if (bias) { biasIndex = inputs.length; inputs.push(bias); }
      if (mask) { maskIndex = inputs.length; inputs.push(mask); }
      const scores = this.kernel(inputs, queries * heads * keys, `let key=idx%${keys}u;let head=(idx/${keys}u)%${heads}u;let query=idx/${keys * heads}u;
${mask ? `if(a${maskIndex}[query*${keys}u+key]==0.0){out[idx]=-1e30;return;}` : ''}
 var sum=0.0;for(var c=0u;c<${dim}u;c++){sum+=${qPreScale ? `(a0[query*${channels}u+head*${dim}u+c]/sqrt(${dim}.0))` : `a0[query*${channels}u+head*${dim}u+c]`}*a1[key*${channels}u+head*${dim}u+c];}
 out[idx]=${qPreScale ? 'sum' : `sum/sqrt(${dim}.0)`}${bias ? `+a${biasIndex}[(query*${keys}u+key)*${heads}u+head]` : ''}${mask ? `+log(a${maskIndex}[query*${keys}u+key])` : ''};`, 'RFD3 attention scores');
      const prob = this.softmax(scores, keys);
      const values = [prob, v];
      if (gate) values.push(gate);
      return this.kernel(values, queries * channels, `let query=idx/${channels}u;let head=(idx%${channels}u)/${dim}u;let c=idx%${dim}u;var sum=0.0;
 for(var key=0u;key<${keys}u;key++){let p=a0[(query*${heads}u+head)*${keys}u+key];if(p!=0.0){sum+=p*a1[key*${channels}u+head*${dim}u+c];}}out[idx]=sum${gate ? '*a2[idx]' : ''};`, 'RFD3 attention values');
    });
  }
  sortRanks(a, width) {
    const size = 2 ** Math.ceil(Math.log2(width)), rows = a.length / width, out = this.tensor(a.length);
    const code = `@group(0) @binding(0) var<storage,read> a:array<f32>;@group(0) @binding(1) var<storage,read_write> out:array<f32>;
var<workgroup> value:array<f32,${size}>;var<workgroup> index:array<u32,${size}>;
@compute @workgroup_size(${size}) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){value[id]=3.402823e38;index[id]=id;workgroupBarrier();if(id<${width}u){var maximum=-3.4e38;for(var j=0u;j<${width}u;j++){maximum=max(maximum,a[g.x*${width}u+j]);}let v=a[g.x*${width}u+id];var before=0u;var lower=0u;for(var j=0u;j<${width}u;j++){let other=a[g.x*${width}u+j];if(other<maximum){lower++;if(j<id&&v<maximum){before++;}}else if(j<id&&v==maximum){before++;}}let position=before+select(0u,lower,v==maximum);value[position]=v;index[position]=id;}workgroupBarrier();
for(var span=2u;span<=${size}u;span*=2u){for(var stride=span/2u;stride>0u;stride/=2u){let other=id^stride;if(other>id){let descending=(id&span)!=0u;if((value[id]<value[other])==descending){let v=value[id];value[id]=value[other];value[other]=v;let i=index[id];index[id]=index[other];index[other]=i;}}workgroupBarrier();}}
if(index[id]<${width}u){out[g.x*${width}u+index[id]]=f32(id);}}`;
    this.dispatch(code, [a, out], [rows], 'RFD3 small neighbour rank');
    return out;
  }
  softmax(a, channels) {
    const rows = a.length / channels, out = this.tensor(a.length);
    const code = `@group(0) @binding(0) var<storage,read> a:array<f32>;@group(0) @binding(1) var<storage,read_write> out:array<f32>;
var<workgroup> tmp:array<f32,64>;var<workgroup> maximum:f32;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){let row=g.x+g.y*65535u;if(row>=${rows}u){return;}var m=-1e30;
for(var c=id;c<${channels}u;c+=64u){m=max(m,a[row*${channels}u+c]);}tmp[id]=m;workgroupBarrier();
for(var s=32u;s>0u;s/=2u){if(id<s){tmp[id]=max(tmp[id],tmp[id+s]);}workgroupBarrier();}if(id==0u){maximum=tmp[0];}workgroupBarrier();
var sum=0.0;for(var c=id;c<${channels}u;c+=64u){let p=exp(a[row*${channels}u+c]-maximum);out[row*${channels}u+c]=p;sum+=p;}tmp[id]=sum;workgroupBarrier();
for(var s=32u;s>0u;s/=2u){if(id<s){tmp[id]+=tmp[id+s];}workgroupBarrier();}for(var c=id;c<${channels}u;c+=64u){out[row*${channels}u+c]/=tmp[0];}}`;
    this.dispatch(code, [a, out], [Math.min(rows, 65535), Math.ceil(rows / 65535)], 'RFD3 softmax');
    return out;
  }
  async submit() {
    if (!this.encoder) {
      this.drainDeferredReleases();
      return;
    }
    this.endPass();
    this.flushUniforms();
    const encoder = this.encoder;
    const usedPacer = this.pieceUsed;
    this.encoder = null;
    this.pendingDispatches = 0;
    this.pieceUsed = false;
    this.device.queue.submit([encoder.finish()]);
    this.drainDeferredReleases();
    if (usedPacer) {
      await this.pacer.blockSubmitted(true);
      await yieldToBrowser();
    } else {
      await this.device.queue.onSubmittedWorkDone();
    }
  }
  async read(t) {
    const [data] = await this.readMany([t]);
    return data;
  }
  async readMany(tensors) {
    if (!tensors.length) return [];
    let totalBytes = 0;
    const offsets = [];
    for (const t of tensors) {
      offsets.push(totalBytes);
      totalBytes += t.length * 4;
    }
    const buffer = this.owner.allocate(totalBytes, 1 | 8, 'RFD3 readback');
    try {
      const enc = this.device.createCommandEncoder();
      for (let i = 0; i < tensors.length; i++) {
        enc.copyBufferToBuffer(tensors[i].buffer, 0, buffer, offsets[i], tensors[i].length * 4);
      }
      this.device.queue.submit([enc.finish()]);
      await buffer.mapAsync(1);
      const mapped = buffer.getMappedRange();
      const out = tensors.map((t, i) => new Float32Array(mapped.slice(offsets[i], offsets[i] + t.length * 4)));
      buffer.unmap();
      return out;
    } catch (err) {
      if (this.owner.closed && this.owner.loss) {
        throw new Error('GPU resources destroyed or device lost' + formatGpuLossDetail(this.owner.loss));
      }
      throw err;
    } finally {
      if (!this.owner.closed) this.owner.release(buffer);
    }
  }
  trim() {
    if (this.encoder) throw new Error('Cannot trim pending GPU commands');
    this.drainDeferredReleases();
    this.releaseUniformSlabs();
    for (const bucket of this.pool.values()) for (const b of bucket) this.owner.release(b);
    this.pool.clear();
    this.poolBytes = 0;
  }
  destroy() {
    try { this.endPass(); } catch { /* ignore */ }
    this.encoder = null;
    this.pass = null;
    this.pendingDispatches = 0;
    this.drainDeferredReleases();
    this.releaseUniformSlabs();
    for (const t of [...this.tensors]) {
      this.owner.release(t.buffer);
      t.released = true;
    }
    this.tensors.clear();
    for (const bucket of this.pool.values()) for (const b of bucket) this.owner.release(b);
    this.pool.clear();
    this.poolBytes = 0;
    this.pipelines.clear();
  }
}
