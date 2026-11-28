// WGSL compute kernels for the ArtiosBackbone diffusion module.
//
// Every generator returns { key, code, bindings, fields }:
//   key      - pipeline cache key (one pipeline per specialisation)
//   code     - WGSL source
//   bindings - ordered [{ name, type }] for an explicit bind group layout;
//              type is 'uniform' | 'read-only-storage' | 'storage'
//   fields   - ordered [[name, 'u32' | 'f32']] of the uniform Params struct
//
// Shapes that size register / workgroup arrays (attention head dim, tile
// sizes) are baked into the source; everything else is a uniform so a single
// pipeline serves every layer of the same kind.

export const ELEMWISE_WG = 256;
export const LN_WG = 128;
export const MATMUL_TILE = 64;

function paramsStruct(fields) {
  return `struct Params {\n${fields.map(([n, t]) => `  ${n}: ${t},`).join('\n')}\n}`;
}

function bindingDecls(bindings, types) {
  return bindings
    .map(({ name, type }, i) => {
      if (type === 'uniform') return `@group(0) @binding(${i}) var<uniform> ${name}: Params;`;
      const access = type === 'storage' ? 'read_write' : 'read';
      return `@group(0) @binding(${i}) var<storage, ${access}> ${name}: array<${types[name] ?? 'f32'}>;`;
    })
    .join('\n');
}

function kernel(key, fields, bindings, types, body) {
  const code = `${paramsStruct(fields)}\n${bindingDecls(bindings, types)}\n${body}`;
  return { key, code, bindings, fields };
}

const u32Fields = (names) => names.map((n) => [n, 'u32']);

// ---------------------------------------------------------------------------
// Matmul: Y[M, N] = X[M, K] . W[N, K]^T with FP32 accumulation.
//
// 64x64 output tile per workgroup, 16x16 threads, 4x4 register tile each,
// K staged through workgroup memory 16 at a time. Weights may be FP32,
// packed FP16 (unpack2x16float) or packed INT8 with a per-row scale.
//
// Fused epilogue, in order:
//   * int8 row scale, + bias
//   * act 'sigmoid' on columns [sigFrom, sigTo), or 'swiglu': weight rows are
//     interleaved (a0, b0, a1, b1, ...) so output col p = silu(y[2p]) * y[2p+1]
//   * * gate[r, goff + c]                   (AdaLN-Zero gated residual)
//   * + out[r, c] ('inplace') or + res[(r % resMod), roff + c] ('buffer')
//   * * rowScale[r]                         (token mask)
//   * store row-major, or transposed out[c * ldo + ooff + r]
// ---------------------------------------------------------------------------
export function matmulKernel({
  wfmt = 'f32',
  bias = false,
  act = 'none',
  gate = false,
  residual = 'none',
  rowScale = false,
  transposeOut = false,
} = {}) {
  if (!['f32', 'f16', 'i8'].includes(wfmt)) throw new Error(`matmul: bad wfmt ${wfmt}`);
  if (!['none', 'sigmoid', 'swiglu'].includes(act)) throw new Error(`matmul: bad act ${act}`);
  if (!['none', 'inplace', 'buffer'].includes(residual)) throw new Error(`matmul: bad residual ${residual}`);
  if (transposeOut && residual === 'inplace') throw new Error('matmul: transposed in-place residual unsupported');
  const key = `matmul:${wfmt}:${bias ? 'b' : ''}:${act}:${gate ? 'g' : ''}:${residual}:${rowScale ? 'rs' : ''}:${transposeOut ? 't' : ''}`;
  const fields = u32Fields([
    'M', 'N', 'K', 'ldx', 'xoff', 'ldw', 'wrow', 'ldo', 'ooff',
    'ldg', 'goff', 'ldr', 'roff', 'resMod', 'sigFrom', 'sigTo', 'scaleOff', 'biasOff',
  ]);
  const useAux = bias || wfmt === 'i8';
  const bindings = [
    { name: 'P', type: 'uniform' },
    { name: 'X', type: 'read-only-storage' },
    { name: 'W', type: 'read-only-storage' },
    ...(useAux ? [{ name: 'AUX', type: 'read-only-storage' }] : []),
    { name: 'OUT', type: 'storage' },
    ...(gate ? [{ name: 'G', type: 'read-only-storage' }] : []),
    ...(residual === 'buffer' ? [{ name: 'RES', type: 'read-only-storage' }] : []),
    ...(rowScale ? [{ name: 'RS', type: 'read-only-storage' }] : []),
  ];
  const types = { W: wfmt === 'f32' ? 'f32' : 'u32' };

  let loadW;
  if (wfmt === 'f32') {
    loadW = `
fn loadW(n: u32, k: u32) -> vec4<f32> {
  var w = vec4<f32>(0.0);
  if (n >= P.N) { return w; }
  let base = (P.wrow + n) * P.ldw;
  for (var u = 0u; u < 4u; u++) {
    if (k + u < P.K) { w[u] = W[base + k + u]; }
  }
  return w;
}`;
  } else if (wfmt === 'f16') {
    loadW = `
fn loadW(n: u32, k: u32) -> vec4<f32> {
  if (n >= P.N || k >= 2u * P.ldw) { return vec4<f32>(0.0); }
  let base = (P.wrow + n) * P.ldw + (k >> 1u);
  return vec4<f32>(unpack2x16float(W[base]), unpack2x16float(W[base + 1u]));
}`;
  } else {
    loadW = `
fn loadW(n: u32, k: u32) -> vec4<f32> {
  if (n >= P.N || k >= 4u * P.ldw) { return vec4<f32>(0.0); }
  let word = i32(W[(P.wrow + n) * P.ldw + (k >> 2u)]);
  return vec4<f32>(f32(extractBits(word, 0u, 8u)), f32(extractBits(word, 8u, 8u)),
                   f32(extractBits(word, 16u, 8u)), f32(extractBits(word, 24u, 8u)));
}`;
  }

  const pre = [];
  if (wfmt === 'i8') pre.push('v = v * AUX[P.scaleOff + P.wrow + n];');
  if (bias) pre.push('v = v + AUX[P.biasOff + P.wrow + n];');
  if (act === 'sigmoid') pre.push('if (n >= P.sigFrom && n < P.sigTo) { v = 1.0 / (1.0 + exp(-v)); }');

  const post = [];
  if (gate) post.push('v = v * G[r * P.ldg + P.goff + c];');
  if (residual === 'inplace') post.push('v = v + OUT[r * P.ldo + P.ooff + c];');
  if (residual === 'buffer') post.push('v = v + RES[(r % P.resMod) * P.ldr + P.roff + c];');
  if (rowScale) post.push('v = v * RS[r];');
  const outIdx = transposeOut ? 'c * P.ldo + P.ooff + r' : 'r * P.ldo + P.ooff + c';

  const epilogue =
    act === 'swiglu'
      ? `
    for (var jp = 0u; jp < 2u; jp++) {
      let n = n0 + 2u * jp;
      if (n >= P.N) { continue; }
      let a = epiPre(acc[i][2u * jp], n);
      let b = epiPre(acc[i][2u * jp + 1u], n + 1u);
      epiStore(r, n >> 1u, a / (1.0 + exp(-a)) * b);
    }`
      : `
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + j;
      if (n >= P.N) { continue; }
      epiStore(r, n, epiPre(acc[i][j], n));
    }`;

  const body = `
const BM = 64u;
const BN = 64u;
const BK = 16u;
var<workgroup> As: array<vec4<f32>, 256>;
var<workgroup> Bs: array<vec4<f32>, 256>;
${loadW}

fn epiPre(acc: f32, n: u32) -> f32 {
  var v = acc;
  ${pre.join('\n  ')}
  return v;
}

fn epiStore(r: u32, c: u32, val: f32) {
  var v = val;
  ${post.join('\n  ')}
  OUT[${outIdx}] = v;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let tx = lid.x;
  let ty = lid.y;
  let bm0 = wid.y * BM;
  let bn0 = wid.x * BN;
  let lr = tid >> 2u;
  let lk = (tid & 3u) * 4u;
  var acc: array<vec4<f32>, 4>;
  let nTiles = (P.K + BK - 1u) / BK;
  for (var t = 0u; t < nTiles; t++) {
    let k0 = t * BK;
    let gr = bm0 + lr;
    for (var u = 0u; u < 4u; u++) {
      let gk = k0 + lk + u;
      var xv = 0.0;
      if (gr < P.M && gk < P.K) { xv = X[gr * P.ldx + P.xoff + gk]; }
      As[(lk + u) * 16u + (lr >> 2u)][lr & 3u] = xv;
    }
    let wv = loadW(bn0 + lr, k0 + lk);
    for (var u = 0u; u < 4u; u++) {
      Bs[(lk + u) * 16u + (lr >> 2u)][lr & 3u] = wv[u];
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk++) {
      let a = As[kk * 16u + ty];
      let b = Bs[kk * 16u + tx];
      acc[0] = fma(vec4<f32>(a.x), b, acc[0]);
      acc[1] = fma(vec4<f32>(a.y), b, acc[1]);
      acc[2] = fma(vec4<f32>(a.z), b, acc[2]);
      acc[3] = fma(vec4<f32>(a.w), b, acc[3]);
    }
    workgroupBarrier();
  }
  let n0 = bn0 + tx * 4u;
  for (var i = 0u; i < 4u; i++) {
    let r = bm0 + ty * 4u + i;
    if (r >= P.M) { continue; }
    ${epilogue}
  }
}`;
  return kernel(key, fields, bindings, types, body);
}

// ---------------------------------------------------------------------------
// LayerNorm over the last dim (biased variance, eps = 1e-5), one workgroup
// per row. mode 'ln'   : y = LN(x * rowScale[r]) * gamma + beta
//            'adaln': y = shift[r, soff + c] + scale[r, coff + c] * LN(x)
// The 'adaln' form covers both AdaLN-Zero modulation (DiT blocks, scale/shift
// precomputed by the fused modulation matmul) and AdaptiveLayerNorm
// (scale = sigmoid(linear_s(s)), shift = linear_nobias_s(s)).
// ---------------------------------------------------------------------------
export function layerNormKernel({ mode = 'ln', affine = true, bias = false, rowScale = false } = {}) {
  if (mode !== 'ln' && mode !== 'adaln') throw new Error(`layerNorm: bad mode ${mode}`);
  const key = `layernorm:${mode}:${affine ? 'a' : ''}:${bias ? 'b' : ''}:${rowScale ? 'rs' : ''}`;
  const fields = [
    ...u32Fields(['M', 'D', 'ldx', 'ldo', 'ooff', 'lds', 'soff', 'ldc', 'coff', 'gridX']),
    ['eps', 'f32'],
  ];
  const isLn = mode === 'ln';
  const bindings = [
    { name: 'P', type: 'uniform' },
    { name: 'X', type: 'read-only-storage' },
    ...(isLn && affine ? [{ name: 'GAMMA', type: 'read-only-storage' }] : []),
    ...(isLn && bias ? [{ name: 'BETA', type: 'read-only-storage' }] : []),
    ...(!isLn ? [{ name: 'SHIFT', type: 'read-only-storage' }, { name: 'SCALE', type: 'read-only-storage' }] : []),
    ...(rowScale ? [{ name: 'RS', type: 'read-only-storage' }] : []),
    { name: 'OUT', type: 'storage' },
  ];
  let finish = 'var y = (x - mean) * rstd;';
  if (isLn && affine) finish += ' y = y * GAMMA[c];';
  if (isLn && bias) finish += ' y = y + BETA[c];';
  if (!isLn) finish += ' y = SHIFT[r * P.lds + P.soff + c] + SCALE[r * P.ldc + P.coff + c] * y;';
  const body = `
const WG = ${LN_WG}u;
var<workgroup> red: array<f32, WG>;

fn reduceSum(v: f32, lid: u32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) { red[lid] = red[lid] + red[lid + s]; }
    workgroupBarrier();
  }
  let total = red[0];
  workgroupBarrier();
  return total;
}

@compute @workgroup_size(${LN_WG})
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let r = wid.x + wid.y * P.gridX;
  if (r >= P.M) { return; }
  let base = r * P.ldx;
  let rs = ${rowScale ? 'RS[r]' : '1.0'};
  var s = 0.0;
  for (var c = lid; c < P.D; c += WG) { s += X[base + c] * rs; }
  let mean = reduceSum(s, lid) / f32(P.D);
  var s2 = 0.0;
  for (var c = lid; c < P.D; c += WG) { let d = X[base + c] * rs - mean; s2 += d * d; }
  let rstd = 1.0 / sqrt(reduceSum(s2, lid) / f32(P.D) + P.eps);
  for (var c = lid; c < P.D; c += WG) {
    let x = X[base + c] * rs;
    ${finish}
    OUT[r * P.ldo + P.ooff + c] = y;
  }
}`;
  return kernel(key, fields, bindings, {}, body);
}

// ---------------------------------------------------------------------------
// Multi-head attention with additive per-head pair bias, flash-style online
// softmax over key tiles. Q/K/V/G are column ranges of one fused projection
// buffer (row stride ld). Output is sigmoid-gated: out = G * softmax(.) V.
//
// local = false (token DiT):  workgroup = BQ queries x (head, sample);
//   logits = q.k / sqrt(D) + bias[h, i, j] where tokenMask[i] * tokenMask[j]
//   else -1e9 (masked_fill), exactly as attention_with_bias.
// local = true (atom windows): workgroup = one query block of NQ atoms;
//   keys are the NK-wide window starting at block * NQ - pad, zero-padded
//   outside [0, A) like rearrange_to_key_subset; bias / mask are trunked
//   [n_subsets, NQ, NK] tensors.
// ---------------------------------------------------------------------------
export function attentionKernel({ headDim, bq, bkt, tpr, local = false, gating = true }) {
  const D = headDim;
  if (D % tpr !== 0) throw new Error(`attention: headDim ${D} not divisible by tpr ${tpr}`);
  const key = `attn:${local ? 'local' : 'global'}:${D}:${bq}:${bkt}:${tpr}:${gating ? 'g' : ''}`;
  const fields = [
    ...u32Fields(['nRows', 'nKeys', 'ld', 'offQ', 'offK', 'offV', 'offG', 'ldo', 'nq', 'pad']),
    ['scale', 'f32'],
  ];
  const bindings = [
    { name: 'P', type: 'uniform' },
    { name: 'QKV', type: 'read-only-storage' },
    { name: 'BIAS', type: 'read-only-storage' },
    { name: 'MASK', type: 'read-only-storage' },
    { name: 'OUT', type: 'storage' },
  ];
  const rowSetup = local
    ? `let i = wid.x * P.nq + qi;
  let rowValid = qi < P.nq && i < P.nRows;
  let nKeys = P.nKeys;`
    : `let i = wid.x * BQ + qi;
  let rowValid = i < P.nRows;
  let nKeys = P.nRows;`;
  const keyRow = local
    ? `let ka = i32(wid.x * P.nq + j) - i32(P.pad);
      let valid = j < nKeys && ka >= 0 && ka < i32(P.nRows);
      let krow = s * P.nRows + u32(max(ka, 0));`
    : `let valid = j < nKeys;
      let krow = s * P.nRows + j;`;
  const score = local
    ? `if (j < nKeys && rowValid) {
        let pidx = (wid.x * P.nq + qi) * nKeys + j;
        if (MASK[pidx] > 0.5) {
          sc = dotK(jj) + BIAS[h * P.nRows * nKeys + pidx];
        } else {
          sc = -1e9;
        }
      }`
    : `if (j < nKeys && rowValid) {
        if (MASK[i] * MASK[j] > 0.5) {
          sc = dotK(jj) + BIAS[(h * P.nRows + i) * P.nRows + j];
        } else {
          sc = -1e9;
        }
      }`;
  const body = `
const D = ${D}u;
const BQ = ${bq}u;
const BKT = ${bkt}u;
const TPR = ${tpr}u;
const DPT = ${D / tpr}u;
const NT = ${bq * tpr}u;
const NEG = -3.0e38;
var<workgroup> Ks: array<f32, ${bkt * D}>;
var<workgroup> Vs: array<f32, ${bkt * D}>;
var<workgroup> Ss: array<f32, ${bq * bkt}>;
var<private> q: array<f32, ${D}>;

fn dotK(jj: u32) -> f32 {
  var acc = 0.0;
  for (var d = 0u; d < D; d++) { acc = fma(q[d], Ks[jj * D + d], acc); }
  return acc;
}

@compute @workgroup_size(${bq * tpr})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let qi = t / TPR;
  let lane = t % TPR;
  let h = wid.y;
  let s = wid.z;
  ${rowSetup}
  let qrow = s * P.nRows + i;
  if (rowValid) {
    for (var d = 0u; d < D; d++) { q[d] = QKV[qrow * P.ld + P.offQ + h * D + d] * P.scale; }
  }
  var m = NEG;
  var l = 0.0;
  var acc: array<f32, DPT>;
  let nTiles = (nKeys + BKT - 1u) / BKT;
  for (var kt = 0u; kt < nTiles; kt++) {
    for (var e = t; e < BKT * D; e += NT) {
      let jj = e / D;
      let d = e % D;
      let j = kt * BKT + jj;
      ${keyRow}
      var kv = 0.0;
      var vv = 0.0;
      if (valid) {
        kv = QKV[krow * P.ld + P.offK + h * D + d];
        vv = QKV[krow * P.ld + P.offV + h * D + d];
      }
      Ks[e] = kv;
      Vs[e] = vv;
    }
    workgroupBarrier();
    for (var jj = lane; jj < BKT; jj += TPR) {
      let j = kt * BKT + jj;
      var sc = NEG;
      ${score}
      Ss[qi * BKT + jj] = sc;
    }
    workgroupBarrier();
    var mt = m;
    for (var jj = 0u; jj < BKT; jj++) { mt = max(mt, Ss[qi * BKT + jj]); }
    let alpha = exp(m - mt);
    l = l * alpha;
    for (var u = 0u; u < DPT; u++) { acc[u] = acc[u] * alpha; }
    for (var jj = 0u; jj < BKT; jj++) {
      let p = exp(Ss[qi * BKT + jj] - mt);
      l += p;
      for (var u = 0u; u < DPT; u++) { acc[u] = fma(p, Vs[jj * D + lane + u * TPR], acc[u]); }
    }
    m = mt;
    workgroupBarrier();
  }
  if (rowValid) {
    let inv = 1.0 / l;
    for (var u = 0u; u < DPT; u++) {
      let d = lane + u * TPR;
      var o = acc[u] * inv;
      ${gating ? 'o = o * QKV[qrow * P.ld + P.offG + h * D + d];' : ''}
      OUT[qrow * P.ldo + h * D + d] = o;
    }
  }
}`;
  return kernel(key, fields, bindings, {}, body);
}

// ---------------------------------------------------------------------------
// average_atom_to_token: out[s, t] = sum_{a in token t, mask[a]} x[s, a] /
// ((count_t > 0 ? 1 : 0) + 1e-4) — matches PyTorch bool scatter_add_ saturation.
// Deterministic CSR gather (no float atomics):
// META = [offsets (nTok + 1) | atom ids], DENOM[t] = (count_t > 0 ? 1 : 0) + 1e-4.
// ---------------------------------------------------------------------------
export function scatterMeanKernel() {
  const fields = u32Fields(['nTok', 'nAtom', 'C', 'ldx', 'ldo', 'ooff']);
  const bindings = [
    { name: 'P', type: 'uniform' },
    { name: 'X', type: 'read-only-storage' },
    { name: 'META', type: 'read-only-storage' },
    { name: 'DENOM', type: 'read-only-storage' },
    { name: 'OUT', type: 'storage' },
  ];
  const body = `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let c = gid.x;
  let t = wid.y;
  let s = wid.z;
  if (c >= P.C) { return; }
  let start = META[t];
  let end = META[t + 1u];
  var acc = 0.0;
  for (var e = start; e < end; e++) {
    let a = META[P.nTok + 1u + e];
    acc += X[(s * P.nAtom + a) * P.ldx + c];
  }
  OUT[(s * P.nTok + t) * P.ldo + P.ooff + c] = acc / DENOM[t];
}`;
  return kernel('scatter_mean', fields, bindings, { META: 'u32' }, body);
}

const linearIndex = `let e = gid.x + gid.y * P.gridX * ${ELEMWISE_WG}u;
  if (e >= P.total) { return; }`;

// broadcast_token_to_atom + skip: out[s, a] = tok[s, idx[a]] + q[s, a]
export function gatherAddKernel() {
  const fields = u32Fields(['total', 'nTok', 'nAtom', 'C', 'ldq', 'gridX']);
  const bindings = [
    { name: 'P', type: 'uniform' },
    { name: 'TOK', type: 'read-only-storage' },
    { name: 'IDX', type: 'read-only-storage' },
    { name: 'Q', type: 'read-only-storage' },
    { name: 'OUT', type: 'storage' },
  ];
  const body = `
@compute @workgroup_size(${ELEMWISE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${linearIndex}
  let c = e % P.C;
  let sa = e / P.C;
  let a = sa % P.nAtom;
  let s = sa / P.nAtom;
  let t = min(IDX[a], P.nTok - 1u);
  OUT[e] = TOK[(s * P.nTok + t) * P.C + c] + Q[sa * P.ldq + c];
}`;
  return kernel('gather_add', fields, bindings, { IDX: 'u32' }, body);
}

// out[s, n, c] = base[n, c] + vec[s, c]   (s_base + linear_t(LN(fourier(t))))
export function broadcastAddKernel() {
  const fields = u32Fields(['total', 'nRows', 'C', 'gridX']);
  const bindings = [
    { name: 'P', type: 'uniform' },
    { name: 'BASE', type: 'read-only-storage' },
    { name: 'VEC', type: 'read-only-storage' },
    { name: 'OUT', type: 'storage' },
  ];
  const body = `
@compute @workgroup_size(${ELEMWISE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${linearIndex}
  let c = e % P.C;
  let n = (e / P.C) % P.nRows;
  let s = e / (P.C * P.nRows);
  OUT[e] = BASE[n * P.C + c] + VEC[s * P.C + c];
}`;
  return kernel('broadcast_add', fields, bindings, {}, body);
}

// FourierEmbedding: cos(2 pi (t' w + b)), t' = log(t_hat / sigma_data) / 4
// read from EDM[s * 4 + 3]. The argument is range-reduced to [-pi, pi].
export function fourierKernel() {
  const fields = u32Fields(['total', 'C', 'gridX']);
  const bindings = [
    { name: 'P', type: 'uniform' },
    { name: 'EDM', type: 'read-only-storage' },
    { name: 'FW', type: 'read-only-storage' },
    { name: 'FB', type: 'read-only-storage' },
    { name: 'OUT', type: 'storage' },
  ];
  const body = `
@compute @workgroup_size(${ELEMWISE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${linearIndex}
  let c = e % P.C;
  let s = e / P.C;
  let x = EDM[s * 4u + 3u] * FW[c] + FB[c];
  OUT[e] = cos(6.283185307179586 * (x - round(x)));
}`;
  return kernel('fourier', fields, bindings, {}, body);
}

// EDM preconditioning: r_unit = x * c_in(t); prev_unit = prev / sigma_data.
// EDM[s * 4 + {0, 1, 2, 3}] = c_in, c_skip, c_out, t'.
export function edmPrecondKernel({ prev = false } = {}) {
  const fields = [...u32Fields(['total', 'perSample', 'gridX']), ['invSigma', 'f32']];
  const bindings = [
    { name: 'P', type: 'uniform' },
    { name: 'XN', type: 'read-only-storage' },
    { name: 'EDM', type: 'read-only-storage' },
    { name: 'RU', type: 'storage' },
    ...(prev ? [{ name: 'PREV', type: 'read-only-storage' }, { name: 'PU', type: 'storage' }] : []),
  ];
  const body = `
@compute @workgroup_size(${ELEMWISE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${linearIndex}
  let s = e / P.perSample;
  RU[e] = XN[e] * EDM[s * 4u];
  ${prev ? 'PU[e] = PREV[e] * P.invSigma;' : ''}
}`;
  return kernel(`edm_precond:${prev ? 'p' : ''}`, fields, bindings, {}, body);
}

// EDM output: r_pred = x * c_skip(t) + F * c_out(t)
export function edmCombineKernel() {
  const fields = u32Fields(['total', 'perSample', 'gridX']);
  const bindings = [
    { name: 'P', type: 'uniform' },
    { name: 'XN', type: 'read-only-storage' },
    { name: 'F', type: 'read-only-storage' },
    { name: 'EDM', type: 'read-only-storage' },
    { name: 'OUT', type: 'storage' },
  ];
  const body = `
@compute @workgroup_size(${ELEMWISE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${linearIndex}
  let s = e / P.perSample;
  OUT[e] = XN[e] * EDM[s * 4u + 1u] + F[e] * EDM[s * 4u + 2u];
}`;
  return kernel('edm_combine', fields, bindings, {}, body);
}
