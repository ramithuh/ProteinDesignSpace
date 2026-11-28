// WGSL compute kernels for the ArtiosPotts MPNN + factorised Potts head.
//
// Every generator returns a kernel spec:
//   { label, code, bindings: [{ name, kind: 'uniform'|'read'|'rw' }], params: [[name, type]] }
// Binding i of the spec is @binding(i) of @group(0); the uniform (always binding 0) is a
// `Params` struct whose fields are listed in `params` (all 4-byte scalars, padded to 16 B).
// Dispatches are folded onto a 2D grid: workgroup id = wid.x + wid.y * 65535u.

export const MAX_GRID = 65535;
export const ELEMWISE_WG = 256;
export const LINEAR_TILE = { M: 64, N: 64, K: 16, threads: 256 };

const FOLD = 'wid.x + wid.y * 65535u';

function withPadding(fields) {
  const out = [...fields];
  while (out.length % 4) out.push([`_pad${out.length}`, 'u32']);
  return out;
}

function header(params, bindings) {
  const struct = `struct Params {\n${params.map(([n, t]) => `  ${n}: ${t},`).join('\n')}\n}`;
  const decls = bindings.map((b, i) => {
    if (b.kind === 'uniform') return `@group(0) @binding(${i}) var<uniform> P: Params;`;
    const access = b.kind === 'rw' ? 'read_write' : 'read';
    return `@group(0) @binding(${i}) var<storage, ${access}> ${b.name}: array<${b.type ?? 'f32'}>;`;
  });
  return `${struct}\n${decls.join('\n')}\n`;
}

function spec(label, params, bindings, body) {
  const p = withPadding(params);
  const b = [{ name: 'P', kind: 'uniform' }, ...bindings];
  return { label, params: p, bindings: b, code: header(p, b) + body };
}

const f = (x) => {
  const s = String(x);
  return /[.eE]/.test(s) ? s : `${s}.0`;
};

// exact-erf GELU (torch.nn.GELU default) via Abramowitz–Stegun 7.1.26 (|err| < 1.5e-7)
const ACT_FNS = /* wgsl */ `
fn erf_approx(x: f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476)); }
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
`;

function applyAct(act, v) {
  if (act === 'none') return v;
  if (act === 'gelu' || act === 'silu') return `${act}(${v})`;
  throw Error(`unknown activation ${act}`);
}

// ---------------------------------------------------------------------------
// Segmented row loader shared by the linear and gather/concat kernels.
//
// A logical input row r (global row = P.inBase + local row) is the concatenation of
// segments; each segment reads a [*, dim] f32 buffer at a row chosen by `index`:
//   'row'  -> r                 (dense rows, e.g. edge features [N*K, E])
//   'self' -> r / P.Kn          (the edge's own node i, e.g. h_nodes_self)
//   'nbr'  -> idx[r]            (the neighbour node j = edge_index[i, k])
// This fuses gather_nodes / expand_at_dim / torch.cat into the consumer's loads.
// ---------------------------------------------------------------------------
function segmentLoader(segments) {
  const total = segments.reduce((s, g) => s + g.dim, 0);
  let off = 0;
  const lines = segments.map((s, i) => {
    const row = s.index === 'row' ? 'r' : s.index === 'self' ? '(r / P.Kn)' : 'u32(idx[r])';
    if (!['row', 'self', 'nbr'].includes(s.index)) throw Error(`bad segment index ${s.index}`);
    const c = off === 0 ? 'c' : `(c - ${off}u)`;
    const expr = `x${i}[${row} * ${s.dim}u + ${c}]`;
    off += s.dim;
    return i === segments.length - 1 ? `  return ${expr};` : `  if (c < ${off}u) { return ${expr}; }`;
  });
  return {
    total,
    usesIdx: segments.some((s) => s.index === 'nbr'),
    code: `fn loadX(m: u32, c: u32) -> f32 {\n  let r = P.inBase + m;\n${lines.join('\n')}\n}\n`,
  };
}

function segmentBindings(segments, usesIdx) {
  const b = segments.map((_, i) => ({ name: `x${i}`, kind: 'read', type: 'f32' }));
  if (usesIdx) b.push({ name: 'idx', kind: 'read', type: 'i32' });
  return b;
}

// ---------------------------------------------------------------------------
// Tiled linear / matmul:  y[m, n] = act( sum_k X[m, k] * W[n, k] * s[n] + b[n] )
//
// W is PyTorch [out, in] row-major. precision:
//   'f32'  W: array<f32>, ldw = in
//   'f16'  W: array<u32>, two halves per word (unpack2x16float), ldw = ceil(in / 2)
//   'int8' W: array<u32>, four signed bytes per word, per-output-row scale s[n], ldw = ceil(in / 4)
// Accumulation is always FP32. aux = [scale (int8 only, rows) | bias (rows)].
// swiglu: weight rows [0, H) are `a`, [H, 2H) are `b`; y = a * silu(b) with H = P.N
// (SwiGLU_MLP's chunk(2) fused into the epilogue).
// 64x64 output tile, 16-wide K tile, 16x16 threads each owning a strided 4x4 block.
// ---------------------------------------------------------------------------
export function linearShader({ precision = 'f32', segments, bias = false, act = 'none', swiglu = false, label } = {}) {
  if (!['f32', 'f16', 'int8'].includes(precision)) throw Error(`unknown precision ${precision}`);
  const loader = segmentLoader(segments);
  const hasAux = precision === 'int8' || bias;
  const bindings = [
    ...segmentBindings(segments, loader.usesIdx),
    { name: 'w', kind: 'read', type: precision === 'f32' ? 'f32' : 'u32' },
    ...(hasAux ? [{ name: 'aux', kind: 'read', type: 'f32' }] : []),
    { name: 'y', kind: 'rw', type: 'f32' },
  ];
  const params = [['M', 'u32'], ['N', 'u32'], ['inBase', 'u32'], ['outBase', 'u32'], ['Kn', 'u32'], ['ldw', 'u32'], ['biasOff', 'u32']];

  const loadW = (target, rowOff) => {
    const row = rowOff ? `(n + ${rowOff})` : 'n';
    if (precision === 'f32') {
      return `
    for (var s = 0u; s < 4u; s++) {
      let e = li + s * 256u;
      let nn = e / TK;
      let kk = e % TK;
      let n = n0 + nn;
      let c = k0 + kk;
      var v = 0.0;
      if (n < P.N && c < KDIM) { v = w[${row} * P.ldw + c]; }
      ${target}[kk * WS + nn] = v;
    }`;
    }
    if (precision === 'f16') {
      return `
    for (var s = 0u; s < 2u; s++) {
      let e = li + s * 256u;
      let nn = e / 8u;
      let kw = e % 8u;
      let n = n0 + nn;
      let word = k0 / 2u + kw;
      var v = vec2<f32>(0.0, 0.0);
      if (n < P.N && word < P.ldw) { v = unpack2x16float(w[${row} * P.ldw + word]); }
      ${target}[(2u * kw) * WS + nn] = v.x;
      ${target}[(2u * kw + 1u) * WS + nn] = v.y;
    }`;
    }
    return `
    {
      let nn = li / 4u;
      let kw = li % 4u;
      let n = n0 + nn;
      let word = k0 / 4u + kw;
      var q = 0i;
      if (n < P.N && word < P.ldw) { q = bitcast<i32>(w[${row} * P.ldw + word]); }
      ${target}[(4u * kw) * WS + nn] = f32(extractBits(q, 0u, 8u));
      ${target}[(4u * kw + 1u) * WS + nn] = f32(extractBits(q, 8u, 8u));
      ${target}[(4u * kw + 2u) * WS + nn] = f32(extractBits(q, 16u, 8u));
      ${target}[(4u * kw + 3u) * WS + nn] = f32(extractBits(q, 24u, 8u));
    }`;
  };

  const R4 = [0, 1, 2, 3];
  const accDecl = R4.flatMap((i) => R4.map((j) => `  var a${i}${j} = 0.0;${swiglu ? ` var g${i}${j} = 0.0;` : ''}`)).join('\n');
  const inner = [
    ...R4.map((i) => `      let xa${i} = xs[kk * XS + ty + ${16 * i}u];`),
    ...R4.map((j) => `      let wa${j} = wsa[kk * WS + tx + ${16 * j}u];`),
    ...(swiglu ? R4.map((j) => `      let wb${j} = wsb[kk * WS + tx + ${16 * j}u];`) : []),
    ...R4.flatMap((i) => R4.map((j) => `      a${i}${j} = fma(xa${i}, wa${j}, a${i}${j});${swiglu ? ` g${i}${j} = fma(xa${i}, wb${j}, g${i}${j});` : ''}`)),
  ].join('\n');
  const epilogue = R4.flatMap((i) =>
    R4.map((j) => {
      const lines = [`    var v = a${i}${j};`];
      if (precision === 'int8') lines.push('    v = v * aux[n];');
      if (bias) lines.push('    v = v + aux[P.biasOff + n];');
      if (swiglu) {
        lines.push(`    var g = g${i}${j};`);
        if (precision === 'int8') lines.push('    g = g * aux[P.N + n];');
        if (bias) lines.push('    g = g + aux[P.biasOff + P.N + n];');
        lines.push('    v = v * silu(g);');
      }
      lines.push(`    v = ${applyAct(act, 'v')};`);
      return `  {
    let m = m0 + ty + ${16 * i}u;
    let n = n0 + tx + ${16 * j}u;
    if (m < P.M && n < P.N) {
${lines.map((l) => '  ' + l).join('\n')}
      y[(P.outBase + m) * P.N + n] = v;
    }
  }`;
    }),
  ).join('\n');

  const body = /* wgsl */ `
const KDIM: u32 = ${loader.total}u;
const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 16u;
const XS: u32 = 65u;
const WS: u32 = 65u;
var<workgroup> xs: array<f32, ${16 * 65}>;
var<workgroup> wsa: array<f32, ${16 * 65}>;
${swiglu ? `var<workgroup> wsb: array<f32, ${16 * 65}>;` : ''}
${ACT_FNS}
${loader.code}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let tilesN = (P.N + TN - 1u) / TN;
  let tile = ${FOLD};
  let m0 = (tile / tilesN) * TM;
  let n0 = (tile % tilesN) * TN;
  if (m0 >= P.M) { return; }
  let tx = lid.x;
  let ty = lid.y;
${accDecl}
  for (var k0 = 0u; k0 < KDIM; k0 += TK) {
    for (var s = 0u; s < 4u; s++) {
      let e = li + s * 256u;
      let mm = e / TK;
      let kk = e % TK;
      let m = m0 + mm;
      let c = k0 + kk;
      var v = 0.0;
      if (m < P.M && c < KDIM) { v = loadX(m, c); }
      xs[kk * XS + mm] = v;
    }
${loadW('wsa', '')}
${swiglu ? loadW('wsb', 'P.N') : ''}
    workgroupBarrier();
    for (var kk = 0u; kk < TK; kk++) {
${inner}
    }
    workgroupBarrier();
  }
${epilogue}
}
`;
  return spec(label ?? `linear_${precision}${swiglu ? '_swiglu' : ''}`, params, bindings, body);
}

// ---------------------------------------------------------------------------
// Gather / concat: materialise [rows, sum(dims)] from segments (gather_nodes,
// gather_and_concat_neighbourhood, augment_edges_with_nb_feats).
// ---------------------------------------------------------------------------
export function gatherConcatShader({ segments, label } = {}) {
  const loader = segmentLoader(segments);
  const bindings = [...segmentBindings(segments, loader.usesIdx), { name: 'y', kind: 'rw', type: 'f32' }];
  const params = [['rows', 'u32'], ['inBase', 'u32'], ['Kn', 'u32'], ['outBase', 'u32']];
  const body = /* wgsl */ `
const KDIM: u32 = ${loader.total}u;
${loader.code}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = (${FOLD}) * 256u + lid.x;
  if (t >= P.rows * KDIM) { return; }
  let m = t / KDIM;
  let c = t % KDIM;
  y[(P.outBase + m) * KDIM + c] = loadX(m, c);
}
`;
  return spec(label ?? 'gather_concat', params, bindings, body);
}

// ---------------------------------------------------------------------------
// LayerNorm over the last dim D (biased variance, eps inside the sqrt, like
// F.layer_norm), optionally fused with a residual add and a per-row mask:
//   x = (LN(x + res) * gamma + beta) * mask[row]       (in place on x)
// gb = [gamma (D) | beta (D)]. One 64-thread workgroup per row.
// ---------------------------------------------------------------------------
export function layerNormShader({ D, residual = false, rowMask = false, label } = {}) {
  const T = 64;
  const per = Math.ceil(D / T);
  const bindings = [
    { name: 'x', kind: 'rw', type: 'f32' },
    { name: 'gb', kind: 'read', type: 'f32' },
    ...(residual ? [{ name: 'res', kind: 'read', type: 'f32' }] : []),
    ...(rowMask ? [{ name: 'mask', kind: 'read', type: 'f32' }] : []),
  ];
  const params = [['rows', 'u32'], ['eps', 'f32']];
  const idx = (t) => `lid.x + ${t * T}u`;
  const load = Array.from({ length: per }, (_, t) =>
    `  var v${t} = 0.0;\n  if (${idx(t)} < D) { v${t} = x[base + ${idx(t)}]${residual ? ` + res[base + ${idx(t)}]` : ''}; s = s + v${t}; }`).join('\n');
  const sq = Array.from({ length: per }, (_, t) =>
    `  if (${idx(t)} < D) { let d = v${t} - mean; q = q + d * d; }`).join('\n');
  const store = Array.from({ length: per }, (_, t) =>
    `  if (${idx(t)} < D) { let c = ${idx(t)}; x[base + c] = ((v${t} - mean) * inv * gb[c] + gb[D + c]) * rm; }`).join('\n');
  const body = /* wgsl */ `
const D: u32 = ${D}u;
var<workgroup> red: array<f32, ${T}>;
fn reduce(v: f32, i: u32) -> f32 {
  red[i] = v;
  workgroupBarrier();
  for (var o = ${T / 2}u; o > 0u; o = o >> 1u) {
    if (i < o) { red[i] = red[i] + red[i + o]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = ${FOLD};
  if (row >= P.rows) { return; }
  let base = row * D;
  var s = 0.0;
${load}
  let mean = reduce(s, lid.x) / f32(D);
  var q = 0.0;
${sq}
  let variance = reduce(q, lid.x) / f32(D);
  let inv = 1.0 / sqrt(variance + P.eps);
  let rm = ${rowMask ? 'mask[row]' : '1.0'};
${store}
}
`;
  return spec(label ?? `layernorm_${D}`, params, bindings, body);
}

// ---------------------------------------------------------------------------
// Element-wise kernels.
// ---------------------------------------------------------------------------
function elementwise(label, bindings, params, expr) {
  return spec(label, [['n', 'u32'], ...params], bindings, /* wgsl */ `
${ACT_FNS}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = (${FOLD}) * 256u + lid.x;
  if (i >= P.n) { return; }
  ${expr}
}
`);
}

/** y = act(x), act in {'gelu', 'silu'} */
export function activationShader(act) {
  return elementwise(`act_${act}`, [{ name: 'x', kind: 'read' }, { name: 'y', kind: 'rw' }], [], `y[i] = ${applyAct(act, 'x[i]')};`);
}

/** y = a + b (residual add) */
export function residualAddShader() {
  return elementwise('residual_add', [{ name: 'a', kind: 'read' }, { name: 'b', kind: 'read' }, { name: 'y', kind: 'rw' }], [], 'y[i] = a[i] + b[i];');
}

/** y[r, c] = x[r, c] * silu(x[r, H + c])  (standalone SwiGLU gate; P.n = rows * H) */
export function swigluShader() {
  return elementwise('swiglu', [{ name: 'x', kind: 'read' }, { name: 'y', kind: 'rw' }], [['H', 'u32']],
    'let r = i / P.H; let c = i % P.H; y[i] = x[r * 2u * P.H + c] * silu(x[r * 2u * P.H + P.H + c]);');
}

// ---------------------------------------------------------------------------
// Neighbour aggregation over the K slots of each node (the upstream "scatter"):
//   out[i, d] = sum_k w[i, k] * msg[i, k, d] * invScale          (mode 'sum')
//   out[i, d] = sum_k w[i, k] * msg[i, k, d] / sum_k w[i, k]     (mode 'mean'; 0 if empty)
// The MPNN layers use mode 'sum' with invScale = 1 / num_neighbours.
// ---------------------------------------------------------------------------
export function aggregateShader({ D, mode = 'sum' } = {}) {
  if (!['sum', 'mean'].includes(mode)) throw Error(`unknown aggregate mode ${mode}`);
  const bindings = [{ name: 'msg', kind: 'read' }, { name: 'w', kind: 'read' }, { name: 'y', kind: 'rw' }];
  const finish = mode === 'sum' ? 'acc * P.invScale' : 'select(0.0, acc / cnt, cnt > 0.0)';
  return spec(`aggregate_${mode}`, [['nodes', 'u32'], ['Kn', 'u32'], ['invScale', 'f32']], bindings, /* wgsl */ `
const D: u32 = ${D}u;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = (${FOLD}) * 256u + lid.x;
  if (t >= P.nodes * D) { return; }
  let i = t / D;
  let d = t % D;
  var acc = 0.0;
  var cnt = 0.0;
  for (var k = 0u; k < P.Kn; k++) {
    let e = i * P.Kn + k;
    let we = w[e];
    acc = acc + we * msg[e * D + d];
    cnt = cnt + we;
  }
  y[t] = ${finish};
}
`);
}

// ---------------------------------------------------------------------------
// k-nearest-neighbour graph (MPNNFeaturizer._compute_neighbour_graph):
// d_adjust = dist * m_ij + (1 - m_ij) * max_j(dist * m_ij); ascending top-k with
// ties broken by the lower index. One thread per row; Keff = min(KMAX, N).
// ---------------------------------------------------------------------------
export function knnShader({ KMAX = 32 } = {}) {
  const bindings = [{ name: 'xyz', kind: 'read' }, { name: 'mask', kind: 'read' }, { name: 'idx', kind: 'rw', type: 'i32' }];
  return spec('knn', [['N', 'u32'], ['Keff', 'u32']], bindings, /* wgsl */ `
const KMAX: u32 = ${KMAX}u;
fn dist(i: u32, j: u32) -> f32 {
  let d = vec3<f32>(xyz[3u * i] - xyz[3u * j], xyz[3u * i + 1u] - xyz[3u * j + 1u], xyz[3u * i + 2u] - xyz[3u * j + 2u]);
  return sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = (${FOLD}) * 64u + lid.x;
  if (i >= P.N) { return; }
  let mi = mask[i];
  var dmax = 0.0;
  for (var j = 0u; j < P.N; j++) {
    dmax = max(dmax, dist(i, j) * (mi * mask[j]));
  }
  var bd: array<f32, KMAX>;
  var bi: array<u32, KMAX>;
  var cnt = 0u;
  for (var j = 0u; j < P.N; j++) {
    let m2 = mi * mask[j];
    let d = dist(i, j) * m2 + (1.0 - m2) * dmax;
    var pos: u32;
    if (cnt < P.Keff) {
      pos = cnt;
      cnt = cnt + 1u;
    } else if (d < bd[P.Keff - 1u]) {
      pos = P.Keff - 1u;
    } else {
      continue;
    }
    while (pos > 0u && bd[pos - 1u] > d) {
      bd[pos] = bd[pos - 1u];
      bi[pos] = bi[pos - 1u];
      pos = pos - 1u;
    }
    bd[pos] = d;
    bi[pos] = j;
  }
  for (var k = 0u; k < P.Keff; k++) { idx[i * P.Keff + k] = i32(bi[k]); }
}
`);
}

/** maskJ[e] = m[j];  edgeMask[e] = m[i] * m[j]  for e = i*Kn + k, j = idx[e] */
export function edgeMasksShader() {
  const bindings = [{ name: 'mask', kind: 'read' }, { name: 'idx', kind: 'read', type: 'i32' },
    { name: 'maskJ', kind: 'rw' }, { name: 'edgeMask', kind: 'rw' }];
  return spec('edge_masks', [['NK', 'u32'], ['Kn', 'u32']], bindings, /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let e = (${FOLD}) * 256u + lid.x;
  if (e >= P.NK) { return; }
  let mj = mask[u32(idx[e])];
  maskJ[e] = mj;
  edgeMask[e] = mask[e / P.Kn] * mj;
}
`);
}

// ---------------------------------------------------------------------------
// Edge input features (pre embed_edge) for a chunk of edge rows:
//   cols [0, 25*nr):  GaussianRBF of the 5x5 backbone atom distances (masked), pair-major
//   cols [25*nr, +pd): EdgeRelativePositionalEncoding gathered at (i, j): the one-hot
//                      relpos features times embed_relpos_features.weight = a sum of 4 columns
// bb = [N, 5, 4] (xyz, atom mask); tok = [N, 5] i32 (asym, residue, entity, sym, token index);
// aux = [rbf centers (nr) | relpos weight (pd x (4 rMax + 2 sMax + 7))].
// ---------------------------------------------------------------------------
export function edgeInputShader({ numRbf = 16, posDim = 16, rMax = 32, sMax = 2 } = {}) {
  const width = 25 * numRbf + posDim;
  const relIn = 4 * rMax + 2 * sMax + 7;
  const bindings = [{ name: 'bb', kind: 'read' }, { name: 'tok', kind: 'read', type: 'i32' },
    { name: 'idx', kind: 'read', type: 'i32' }, { name: 'aux', kind: 'read' }, { name: 'y', kind: 'rw' }];
  return spec('edge_input', [['rows', 'u32'], ['inBase', 'u32'], ['Kn', 'u32'], ['gamma', 'f32']], bindings, /* wgsl */ `
const NR: u32 = ${numRbf}u;
const WIDTH: u32 = ${width}u;
const RBF_W: u32 = ${25 * numRbf}u;
const REL_IN: u32 = ${relIn}u;
const RMAX: i32 = ${rMax}i;
const SMAX: i32 = ${sMax}i;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = (${FOLD}) * 256u + lid.x;
  if (t >= P.rows * WIDTH) { return; }
  let m = t / WIDTH;
  let c = t % WIDTH;
  let r = P.inBase + m;
  let i = r / P.Kn;
  let j = u32(idx[r]);
  var v = 0.0;
  if (c < RBF_W) {
    let pair = c / NR;
    let kern = c % NR;
    let a = (i * 5u + pair / 5u) * 4u;
    let b = (j * 5u + pair % 5u) * 4u;
    let dx = bb[a] - bb[b];
    let dy = bb[a + 1u] - bb[b + 1u];
    let dz = bb[a + 2u] - bb[b + 2u];
    let mm = bb[a + 3u] * bb[b + 3u];
    let dd = sqrt(dx * dx + dy * dy + dz * dz) * mm - aux[kern];
    v = exp(-P.gamma * dd * dd) * mm;
  } else {
    let o = c - RBF_W;
    let ti = i * 5u;
    let tj = j * 5u;
    let sameChain = tok[ti] == tok[tj];
    let sameRes = tok[ti + 1u] == tok[tj + 1u];
    let sameEnt = tok[ti + 2u] == tok[tj + 2u];
    let dRes = select(2 * RMAX + 1, clamp(tok[ti + 1u] - tok[tj + 1u] + RMAX, 0, 2 * RMAX), sameChain);
    let dTok = select(2 * RMAX + 1, clamp(tok[ti + 4u] - tok[tj + 4u] + RMAX, 0, 2 * RMAX), sameChain && sameRes);
    let dChain = select(2 * SMAX + 1, clamp(tok[ti + 3u] - tok[tj + 3u] + SMAX, 0, 2 * SMAX), sameEnt);
    let base = NR + o * REL_IN;
    v = aux[base + u32(dRes)] + aux[base + u32(2 * RMAX + 2 + dTok)];
    if (sameEnt) { v = v + aux[base + u32(4 * RMAX + 4)]; }
    v = v + aux[base + u32(4 * RMAX + 5 + dChain)];
  }
  y[m * WIDTH + c] = v;
}
`);
}

// ---------------------------------------------------------------------------
// Node input features (MPNNFeaturizer._compute_node_features, pre node_norm):
//   ligand_condition_embed(atom feats) * is_atomic, averaged atom -> token (sum / (count + 1e-4))
//   + entity_type_embed(entity flags) + non_protein_restype_embed(restype * (1 - is_protein))
// tokf = [N, 4 + nRes] (is_protein, is_ligand, is_dna, is_rna, restype one-hot)
// atoms = [A, 4] (ref_charge, ref_mask, atom_mask, is_atomic); elem = [A, nElem] (ref_element)
// csr = [N + 1 token starts | A atom ids grouped by token]; w = [D, 2 + nElem + 4 + nRes]
// ---------------------------------------------------------------------------
export function nodeInputShader({ D = 128, nElem = 128, nRes = 32 } = {}) {
  const lig = 2 + nElem;
  const wRow = lig + 4 + nRes;
  const bindings = [{ name: 'tokf', kind: 'read' }, { name: 'atoms', kind: 'read' }, { name: 'elem', kind: 'read' },
    { name: 'csr', kind: 'read', type: 'i32' }, { name: 'w', kind: 'read' }, { name: 'y', kind: 'rw' }];
  return spec('node_input', [['N', 'u32']], bindings, /* wgsl */ `
const D: u32 = ${D}u;
const NELEM: u32 = ${nElem}u;
const NRES: u32 = ${nRes}u;
const LIG: u32 = ${lig}u;
const WROW: u32 = ${wRow}u;
const TF: u32 = ${4 + nRes}u;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = (${FOLD}) * 256u + lid.x;
  if (t >= P.N * D) { return; }
  let n = t / D;
  let d = t % D;
  let wr = d * WROW;
  var lig = 0.0;
  var cnt = 0.0;
  let a0 = u32(csr[n]);
  let a1 = u32(csr[n + 1u]);
  for (var q = a0; q < a1; q++) {
    let a = u32(csr[P.N + 1u + q]);
    let am = atoms[a * 4u + 2u];
    cnt = cnt + am;
    let wa = am * atoms[a * 4u + 3u];
    if (wa != 0.0) {
      var s = atoms[a * 4u] * w[wr] + atoms[a * 4u + 1u] * w[wr + 1u];
      for (var e = 0u; e < NELEM; e++) { s = s + elem[a * NELEM + e] * w[wr + 2u + e]; }
      lig = lig + wa * s;
    }
  }
  var v = lig / (select(0.0, 1.0, cnt > 0.0) + 1e-4);
  let tb = n * TF;
  var ent = 0.0;
  for (var e = 0u; e < 4u; e++) { ent = ent + tokf[tb + e] * w[wr + LIG + e]; }
  let notProtein = 1.0 - tokf[tb];
  var res = 0.0;
  for (var c = 0u; c < NRES; c++) { res = res + tokf[tb + 4u + c] * notProtein * w[wr + LIG + 4u + c]; }
  y[t] = v + ent + res;
}
`);
}

// ---------------------------------------------------------------------------
// Potts fields: h = scale * token_mask * hpre, then zero-sum gauge h -= mean_a h.
// ---------------------------------------------------------------------------
export function pottsFieldsShader({ R = 32, gauge = true } = {}) {
  const bindings = [{ name: 'hpre', kind: 'read' }, { name: 'mask', kind: 'read' }, { name: 'h', kind: 'rw' }];
  return spec('potts_fields', [['N', 'u32'], ['scale', 'f32']], bindings, /* wgsl */ `
const R: u32 = ${R}u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = (${FOLD}) * 64u + lid.x;
  if (n >= P.N) { return; }
  let s = P.scale * mask[n];
  var mean = 0.0;
  ${gauge ? 'for (var a = 0u; a < R; a++) { mean = mean + s * hpre[n * R + a]; }\n  mean = mean / f32(R);' : ''}
  for (var a = 0u; a < R; a++) { h[n * R + a] = s * hpre[n * R + a] - mean; }
}
`);
}

// ---------------------------------------------------------------------------
// Factorised couplings for a chunk of edges (GraphPotts FACTOR):
//   L[a, f] = s * lr[e, a*F + f],  Rt[f, b] = s * lr[e, R*F + f*R + b],  s = scale * edge_mask
//   J[a, b] = sum_f L[a, f] Rt[f, b]
// with the zero-sum gauge applied exactly in factor space (double-centring a product equals
// the product of the centred factors). One 256-thread workgroup per edge.
// ---------------------------------------------------------------------------
export function pottsCouplingShader({ R = 32, F = 32, gauge = true } = {}) {
  const bindings = [{ name: 'lr', kind: 'read' }, { name: 'edgeMask', kind: 'read' }, { name: 'J', kind: 'rw' }];
  return spec('potts_coupling', [['rows', 'u32'], ['inBase', 'u32'], ['scale', 'f32']], bindings, /* wgsl */ `
const R: u32 = ${R}u;
const F: u32 = ${F}u;
const RF: u32 = ${R * F}u;
const RR: u32 = ${R * R}u;
var<workgroup> Ls: array<f32, ${R * F}>;
var<workgroup> Rs: array<f32, ${F * R}>;
var<workgroup> mL: array<f32, ${F}>;
var<workgroup> mR: array<f32, ${F}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = ${FOLD};
  if (m >= P.rows) { return; }
  let e = P.inBase + m;
  let s = P.scale * edgeMask[e];
  for (var q = li; q < RF; q += 256u) {
    Ls[q] = s * lr[m * 2u * RF + q];
    Rs[q] = s * lr[m * 2u * RF + RF + q];
  }
  workgroupBarrier();
${gauge ? `  for (var ff = li; ff < F; ff += 256u) {
    var accL = 0.0;
    var accR = 0.0;
    for (var a = 0u; a < R; a++) {
      accL = accL + Ls[a * F + ff];
      accR = accR + Rs[ff * R + a];
    }
    mL[ff] = accL / f32(R);
    mR[ff] = accR / f32(R);
  }
  workgroupBarrier();
  for (var q = li; q < RF; q += 256u) {
    Ls[q] = Ls[q] - mL[q % F];
    Rs[q] = Rs[q] - mR[q / R];
  }
  workgroupBarrier();` : ''}
  for (var q = li; q < RR; q += 256u) {
    let a = q / R;
    let b = q % R;
    var acc = 0.0;
    for (var ff = 0u; ff < F; ff++) { acc = fma(Ls[a * F + ff], Rs[ff * R + b], acc); }
    J[e * RR + q] = acc;
  }
}
`);
}

// ---------------------------------------------------------------------------
// _symmetrize_J: J_sym[i,k,a,b] = 0.5 * edge_mask[i,k] * has_reverse * (J[i,k,a,b] + J[j,k',b,a])
// where k' is the first slot of j's neighbour list pointing back to i.
// ---------------------------------------------------------------------------
export function pottsSymmetrizeShader({ R = 32 } = {}) {
  const bindings = [{ name: 'Jin', kind: 'read' }, { name: 'idx', kind: 'read', type: 'i32' },
    { name: 'edgeMask', kind: 'read' }, { name: 'J', kind: 'rw' }];
  return spec('potts_symmetrize', [['NK', 'u32'], ['Kn', 'u32']], bindings, /* wgsl */ `
const R: u32 = ${R}u;
const RR: u32 = ${R * R}u;
const TS: u32 = ${R + 1}u;
var<workgroup> T: array<f32, ${R * (R + 1)}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let e = ${FOLD};
  if (e >= P.NK) { return; }
  let i = i32(e / P.Kn);
  let j = u32(idx[e]);
  var rev = P.Kn;
  for (var k = P.Kn; k > 0u; k--) {
    if (idx[j * P.Kn + k - 1u] == i) { rev = k - 1u; }
  }
  let m = select(0.0, 0.5 * edgeMask[e], rev < P.Kn);
  let er = j * P.Kn + select(0u, rev, rev < P.Kn);
  for (var q = li; q < RR; q += 256u) { T[(q / R) * TS + q % R] = Jin[er * RR + q]; }
  workgroupBarrier();
  for (var q = li; q < RR; q += 256u) {
    let a = q / R;
    let b = q % R;
    J[e * RR + q] = m * (Jin[e * RR + q] + T[b * TS + a]);
  }
}
`);
}
