/**
 * WGSL compute kernels for Proton-PottsMPNN.
 *
 * Every shader here is plain WGSL with no extensions requested: FP16 and INT8
 * weights are stored *packed* in `array<u32>` storage buffers and unpacked in
 * the shader (`unpack2x16float` / byte extraction), and all arithmetic and
 * accumulation is FP32. That keeps one code path across adapters — a device
 * without `shader-f16` still runs the FP16 bundle, it just widens on read.
 *
 * Shapes are passed in uniforms rather than specialised into the source so one
 * pipeline per dtype serves every layer of the network.
 */

export const TILE = 16;
export const LN_WORKGROUP = 256;
export const ELEMENTWISE_WORKGROUP = 256;

/**
 * Shared uniform block for the elementwise kernels.
 *
 * `strideX` is `numWorkgroupsX * ELEMENTWISE_WORKGROUP`, which lets a kernel
 * flatten a 2-D dispatch back into one linear thread index. The Potts tensor is
 * `L * K * V * V` elements — 9.9M at L=229 — and a 1-D dispatch would need
 * 38,644 workgroups here but blows past `maxComputeWorkgroupsPerDimension`
 * (65,535) the moment the protein gets larger, so every elementwise kernel is
 * dispatched 2-D.
 */
const EDIMS_WGSL = /* wgsl */ `
struct EDims {
  a: u32,
  b: u32,
  c: u32,
  strideX: u32,
  scale: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

fn flatThread(gid: vec3<u32>, strideX: u32) -> u32 {
  return gid.y * strideX + gid.x;
}
`;

/** Flags packed into the GEMM uniform's `flags` field. */
export const GEMM_FLAG_BIAS = 1;
export const GEMM_FLAG_GELU = 2;

/**
 * `erf` and the exact (erf-based) GELU in FP32.
 *
 * `torch.nn.GELU()` is the exact erf form, so the tanh approximation would sit
 * ~1e-3 away from the oracle — far above the parity we check. This is W. J.
 * Cody's three-interval rational approximation evaluated in f32, which lands at
 * the f32 noise floor instead.
 */
const GELU_WGSL = /* wgsl */ `
fn cody_erf(x: f32) -> f32 {
  let y = abs(x);
  if (y <= 0.46875) {
    let a0 = 3.16112374387056560e0;
    let a1 = 1.13864154151050156e2;
    let a2 = 3.77485237685302021e2;
    let a3 = 3.20937758913846947e3;
    let a4 = 1.85777706184603153e-1;
    let b0 = 2.36012909523441209e1;
    let b1 = 2.44024637934444173e2;
    let b2 = 1.28261652607737228e3;
    let b3 = 2.84423683343917062e3;
    let z = y * y;
    var num = a4 * z;
    var den = z;
    num = (num + a0) * z; den = (den + b0) * z;
    num = (num + a1) * z; den = (den + b1) * z;
    num = (num + a2) * z; den = (den + b2) * z;
    return x * (num + a3) / (den + b3);
  }
  var res: f32;
  if (y <= 4.0) {
    let c = array<f32, 9>(
      5.64188496988670089e-1, 8.88314979438837594e0, 6.61191906371416295e1,
      2.98635138197400131e2, 8.81952221241769090e2, 1.71204761263407058e3,
      2.05107837782607147e3, 1.23033935479799725e3, 2.15311535474403846e-8);
    let d = array<f32, 8>(
      1.57449261107098347e1, 1.17693950891312499e2, 5.37181101862009858e2,
      1.62138957456669019e3, 3.29079923573345963e3, 4.36261909014324716e3,
      3.43936767414372164e3, 1.23033935480374942e3);
    var num = c[8] * y;
    var den = y;
    for (var i = 0u; i < 7u; i = i + 1u) {
      num = (num + c[i]) * y;
      den = (den + d[i]) * y;
    }
    res = (num + c[7]) / (den + d[7]);
  } else {
    let p = array<f32, 6>(
      3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1,
      1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2);
    let q = array<f32, 5>(
      2.56852019228982242e0, 1.87295284992346047e0, 5.27905102951428412e-1,
      6.05183413124413191e-2, 2.33520497626869185e-3);
    let z = 1.0 / (y * y);
    var num = p[5] * z;
    var den = z;
    for (var i = 0u; i < 4u; i = i + 1u) {
      num = (num + p[i]) * z;
      den = (den + q[i]) * z;
    }
    res = z * (num + p[4]) / (den + q[4]);
    res = (0.5641895835477563 - res) / y;
  }
  // Cody's split of exp(-y*y) keeps the exponential from losing bits at large y.
  let zt = floor(y * 16.0) / 16.0;
  let del = (y - zt) * (y + zt);
  let erfc = exp(-zt * zt) * exp(-del) * res;
  if (x > 0.0) { return 1.0 - erfc; }
  return erfc - 1.0;
}

fn gelu(x: f32) -> f32 {
  return 0.5 * x * (1.0 + cody_erf(x * 0.70710678118654752));
}
`;

/**
 * Tiled GEMM implementing a PyTorch `Linear`:
 * `C[m, n] = bias[n] + dot(A[m, :], W[n, :])`, with `W` row-major `[N, Kin]`.
 *
 * `dtype` selects how a weight element is fetched:
 *   - `'f32'` straight from an `array<f32>`
 *   - `'f16'` two halves per `u32`, widened with `unpack2x16float`
 *   - `'i8'`  four bytes per `u32`, sign-extended and scaled by the row's
 *     quantization scale (per-output-channel, so it is a single multiply)
 */
export function gemmSource(dtype) {
  let weightDecl;
  let loadW;
  if (dtype === 'f32') {
    weightDecl = '@group(0) @binding(2) var<storage, read> W: array<f32>;';
    loadW = 'return W[n * dims.kin + k];';
  } else if (dtype === 'f16') {
    weightDecl = '@group(0) @binding(2) var<storage, read> W: array<u32>;';
    loadW = `
  let i = n * dims.kin + k;
  let pair = unpack2x16float(W[i >> 1u]);
  if ((i & 1u) == 0u) { return pair.x; }
  return pair.y;`;
  } else if (dtype === 'i8') {
    weightDecl = '@group(0) @binding(2) var<storage, read> W: array<u32>;';
    loadW = `
  let i = n * dims.kin + k;
  let byte = (W[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
  var q = i32(byte);
  if (q > 127) { q = q - 256; }
  return f32(q) * wscale[n];`;
  } else {
    throw new Error(`unknown weight dtype "${dtype}"`);
  }

  return /* wgsl */ `
struct Dims { m: u32, kin: u32, n: u32, flags: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
${weightDecl}
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> C: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;

${GELU_WGSL}

fn loadW(n: u32, k: u32) -> f32 {${loadW}
}

const TILE: u32 = ${TILE}u;
var<workgroup> As: array<f32, ${TILE * TILE}>;
var<workgroup> Ws: array<f32, ${TILE * TILE}>;

@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wg.y * TILE + lid.y;
  let n = wg.x * TILE + lid.x;
  var acc = 0.0;
  let nTiles = (dims.kin + TILE - 1u) / TILE;
  for (var t = 0u; t < nTiles; t = t + 1u) {
    let k0 = t * TILE;

    // Stage A[m, k0 .. k0+TILE) — thread (lid.y = m-local, lid.x = k-local).
    let ka = k0 + lid.x;
    var av = 0.0;
    if (m < dims.m && ka < dims.kin) { av = A[m * dims.kin + ka]; }
    As[lid.y * TILE + lid.x] = av;

    // Stage W[n0 .. n0+TILE, k0 .. k0+TILE) transposed into [k-local][n-local],
    // so the inner product below reads both tiles without bank conflicts.
    let nw = wg.x * TILE + lid.x;
    let kw = k0 + lid.y;
    var wv = 0.0;
    if (nw < dims.n && kw < dims.kin) { wv = loadW(nw, kw); }
    Ws[lid.y * TILE + lid.x] = wv;

    workgroupBarrier();
    for (var kk = 0u; kk < TILE; kk = kk + 1u) {
      acc = acc + As[lid.y * TILE + kk] * Ws[kk * TILE + lid.x];
    }
    workgroupBarrier();
  }
  if (m >= dims.m || n >= dims.n) { return; }
  var v = acc;
  if ((dims.flags & ${GEMM_FLAG_BIAS}u) != 0u) { v = v + bias[n]; }
  if ((dims.flags & ${GEMM_FLAG_GELU}u) != 0u) { v = gelu(v); }
  C[m * dims.n + n] = v;
}
`;
}

/**
 * Row-wise LayerNorm over the last dimension (biased variance, `eps = 1e-5`),
 * one workgroup per row with a shared-memory tree reduction.
 */
export const LAYERNORM_WGSL = /* wgsl */ `
struct Dims { rows: u32, dim: u32, eps: f32, pad: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;

const WG: u32 = ${LN_WORKGROUP}u;
var<workgroup> partial: array<f32, ${LN_WORKGROUP}>;

@compute @workgroup_size(${LN_WORKGROUP}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= dims.rows) { return; }
  let off = row * dims.dim;
  let tid = lid.x;

  var s = 0.0;
  for (var i = tid; i < dims.dim; i = i + WG) { s = s + X[off + i]; }
  partial[tid] = s;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    workgroupBarrier();
  }
  let mean = partial[0] / f32(dims.dim);
  workgroupBarrier();

  var v = 0.0;
  for (var i = tid; i < dims.dim; i = i + WG) {
    let d = X[off + i] - mean;
    v = v + d * d;
  }
  partial[tid] = v;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    workgroupBarrier();
  }
  let inv = 1.0 / sqrt(partial[0] / f32(dims.dim) + dims.eps);
  workgroupBarrier();

  for (var i = tid; i < dims.dim; i = i + WG) {
    X[off + i] = (X[off + i] - mean) * inv * gamma[i] + beta[i];
  }
}
`;

/** Build `[h_V_i, h_E_ik, h_V_j]` `[L, K, 3H]` — the EncLayer message input. */
export const ENC_MSG_INPUT_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> dims: EDims;
@group(0) @binding(1) var<storage, read> hV: array<f32>;
@group(0) @binding(2) var<storage, read> hE: array<f32>;
@group(0) @binding(3) var<storage, read> edgeIdx: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

${EDIMS_WGSL}

@compute @workgroup_size(${ELEMENTWISE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = dims.a; let K = dims.b; let H = dims.c;
  let t = flatThread(gid, dims.strideX);
  if (t >= L * K * H) { return; }
  let h = t % H;
  let e = t / H;
  let i = e / K;
  let j = edgeIdx[e];
  let base = e * 3u * H;
  out[base + h] = hV[i * H + h];
  out[base + H + h] = hE[e * H + h];
  out[base + 2u * H + h] = hV[j * H + h];
}
`;

/**
 * Masked, scaled neighbour aggregation plus the residual:
 * `out[i] = hV[i] + sum_k (msg[i, k] * maskE[i, k]) / scale`.
 */
export const ENC_AGGREGATE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> dims: EDims;
@group(0) @binding(1) var<storage, read> hV: array<f32>;
@group(0) @binding(2) var<storage, read> msg: array<f32>;
@group(0) @binding(3) var<storage, read> maskE: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

${EDIMS_WGSL}

@compute @workgroup_size(${ELEMENTWISE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = dims.a; let K = dims.b; let H = dims.c;
  let t = flatThread(gid, dims.strideX);
  if (t >= L * H) { return; }
  let h = t % H;
  let i = t / H;
  var acc = 0.0;
  for (var k = 0u; k < K; k = k + 1u) {
    let e = i * K + k;
    acc = acc + msg[e * H + h] * maskE[e];
  }
  out[t] = hV[t] + acc / dims.scale;
}
`;

/** `A += B` over `count` elements. */
export const ADD_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> dims: EDims;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;

${EDIMS_WGSL}

@compute @workgroup_size(${ELEMENTWISE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = flatThread(gid, dims.strideX);
  if (t >= dims.a) { return; }
  A[t] = A[t] + B[t];
}
`;

/** Zero the rows of `[L, H]` whose residue is masked out. */
export const MASK_NODES_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> dims: EDims;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> residueMask: array<f32>;

${EDIMS_WGSL}

@compute @workgroup_size(${ELEMENTWISE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = dims.a; let H = dims.b;
  let t = flatThread(gid, dims.strideX);
  if (t >= L * H) { return; }
  X[t] = X[t] * residueMask[t / H];
}
`;

/**
 * Build the decoder message input `[L, K, 4H]`:
 * `[h_Vdec_i, h_E_ik, causal * h_S_j, causal * h_Vdec_j + anti * h_V_j]`.
 *
 * `cmask` holds `(causal, anti)` per edge as a `vec2<f32>`, which keeps the
 * binding count inside the default per-stage storage-buffer limit.
 */
export const DEC_MSG_INPUT_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> dims: EDims;
@group(0) @binding(1) var<storage, read> hVdec: array<f32>;
@group(0) @binding(2) var<storage, read> hE: array<f32>;
@group(0) @binding(3) var<storage, read> hS: array<f32>;
@group(0) @binding(4) var<storage, read> hVenc: array<f32>;
@group(0) @binding(5) var<storage, read> edgeIdx: array<u32>;
@group(0) @binding(6) var<storage, read> cmask: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

${EDIMS_WGSL}

@compute @workgroup_size(${ELEMENTWISE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = dims.a; let K = dims.b; let H = dims.c;
  let t = flatThread(gid, dims.strideX);
  if (t >= L * K * H) { return; }
  let h = t % H;
  let e = t / H;
  let i = e / K;
  let j = edgeIdx[e];
  let cm = cmask[e];
  let base = e * 4u * H;
  out[base + h] = hVdec[i * H + h];
  out[base + H + h] = hE[e * H + h];
  out[base + 2u * H + h] = cm.x * hS[j * H + h];
  out[base + 3u * H + h] = cm.x * hVdec[j * H + h] + cm.y * hVenc[j * H + h];
}
`;

/**
 * Potts head, pass 1: apply the edge mask and force the self edge (slot 0) to be
 * diagonal, so `diag(J[i, 0])` is exactly the single-body field `h_i`.
 */
export const POTTS_MASK_DIAG_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> dims: EDims;
@group(0) @binding(1) var<storage, read_write> J: array<f32>;
@group(0) @binding(2) var<storage, read> maskE: array<f32>;

${EDIMS_WGSL}

@compute @workgroup_size(${ELEMENTWISE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = dims.a; let K = dims.b; let V = dims.c;
  let vv = V * V;
  let t = flatThread(gid, dims.strideX);
  if (t >= L * K * vv) { return; }
  let e = t / vv;
  let rest = t % vv;
  if (maskE[e] == 0.0) { J[t] = 0.0; return; }
  if ((e % K) == 0u) {
    let a = rest / V;
    let b = rest % V;
    if (a != b) { J[t] = 0.0; }
  }
}
`;

/**
 * Potts head, pass 2: symmetrise reciprocal edges,
 * `out[i, k] = 0.5 * (J[i, k] + J[j, rev]^T)` where `j = edgeIdx[i, k]` and
 * `rev` is the slot in row `j` that points back at `i`.
 *
 * The kNN graph is asymmetric, so an edge without a reciprocal partner is left
 * untouched — hence a separate output buffer rather than an in-place update.
 */
export const POTTS_MERGE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> dims: EDims;
@group(0) @binding(1) var<storage, read> J: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> edgeIdx: array<u32>;
@group(0) @binding(4) var<storage, read> reverseK: array<u32>;
@group(0) @binding(5) var<storage, read> mergeValid: array<f32>;

${EDIMS_WGSL}

@compute @workgroup_size(${ELEMENTWISE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = dims.a; let K = dims.b; let V = dims.c;
  let vv = V * V;
  let t = flatThread(gid, dims.strideX);
  if (t >= L * K * vv) { return; }
  let e = t / vv;
  let rest = t % vv;
  if (mergeValid[e] == 0.0) { out[t] = J[t]; return; }
  let a = rest / V;
  let b = rest % V;
  let j = edgeIdx[e];
  let other = (j * K + reverseK[e]) * vv;
  out[t] = 0.5 * (J[t] + J[other + b * V + a]);
}
`;

/** Extract the single-body field `h[i, a] = J[i, 0, a, a]`. */
export const EXTRACT_FIELD_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> dims: EDims;
@group(0) @binding(1) var<storage, read> J: array<f32>;
@group(0) @binding(2) var<storage, read_write> h: array<f32>;

${EDIMS_WGSL}

@compute @workgroup_size(${ELEMENTWISE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = dims.a; let K = dims.b; let V = dims.c;
  let t = flatThread(gid, dims.strideX);
  if (t >= L * V) { return; }
  let a = t % V;
  let i = t / V;
  h[t] = J[(i * K) * (V * V) + a * V + a];
}
`;

/** Every shader in one place, for cache-keying and introspection. */
export const SHADERS = Object.freeze({
  gemmF32: () => gemmSource('f32'),
  gemmF16: () => gemmSource('f16'),
  gemmI8: () => gemmSource('i8'),
  layernorm: LAYERNORM_WGSL,
  encMsgInput: ENC_MSG_INPUT_WGSL,
  encAggregate: ENC_AGGREGATE_WGSL,
  add: ADD_WGSL,
  maskNodes: MASK_NODES_WGSL,
  decMsgInput: DEC_MSG_INPUT_WGSL,
  pottsMaskDiag: POTTS_MASK_DIAG_WGSL,
  pottsMerge: POTTS_MERGE_WGSL,
  extractField: EXTRACT_FIELD_WGSL,
});
