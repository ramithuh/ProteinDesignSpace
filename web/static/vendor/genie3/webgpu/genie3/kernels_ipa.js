/**
 * Specialized WGSL compute kernels for Genie 3:
 * - ReducedInvariantPointAttention (used in LatentTransformer blocks)
 * - Full AF2 InvariantPointAttention with 3D rigid-frame query/key/value points + pair bias
 * - BackboneUpdate (quaternion normalization -> rotation matrix -> SE(3) frame composition)
 * - TriangleMultiplicativeUpdate gate-and-scatter helper
 * - Global token pad / unpad helpers
 *
 * Every kernel uses folded 2D dispatch guards (`wid.x + wid.y * 65535u`) and stays within
 * WebGPU spec minimums (`workgroup_size <= 256`, zero dynamic allocations).
 */

/**
 * ReducedIPA attention logits:
 *   attn[h, i, j] = scalarFactor * dot(q[i, h], k[j, h])
 *                 + pairFactor * b[(i * M + j) * H + h]
 *                 + inf * (mask[i] * mask[j] - 1)
 *
 * Layouts:
 *   q:    [M, H, c_h]
 *   kv:   [M, H, 2 * c_h] (interleaved per head: k = [0..c_h-1], v = [c_h..2*c_h-1])
 *   b:    [M, M, H]
 *   mask: [M]
 *   attn: [H, M, M]
 */
export const REDUCED_IPA_LOGITS_SOURCE = `
struct P {
  M: u32, H: u32, cH: u32, pad0: u32,
  scalarFactor: f32, pairFactor: f32, infVal: f32, pad1: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> attn: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let total = p.H * p.M * p.M;
  if (idx >= total) { return; }
  let j = idx % p.M;
  let rem = idx / p.M;
  let i = rem % p.M;
  let h = rem / p.M;

  let qBase = (i * p.H + h) * p.cH;
  let kBase = (j * p.H + h) * (2u * p.cH);
  var dotQK = 0.0;
  for (var d = 0u; d < p.cH; d += 1u) {
    dotQK += q[qBase + d] * kv[kBase + d];
  }
  let pairBias = b[(i * p.M + j) * p.H + h];
  let sqMask = mask[i] * mask[j];
  attn[idx] = dotQK * p.scalarFactor + pairBias * p.pairFactor + p.infVal * (sqMask - 1.0);
}
`;

/**
 * Transform local IPA points from 3-plane layout `[N, 3, H * P]` into global coordinates `[N, H, P, 3]`
 * using residue rigid frames `fi = (rots [N, 9], trans [N, 3])`:
 *   global = R_i * local + t_i
 */
export const IPA_POINT_APPLY_SOURCE = `
struct P { N: u32, H: u32, pts: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ptsLocal: array<f32>;
@group(0) @binding(2) var<storage, read> rots: array<f32>;
@group(0) @binding(3) var<storage, read> trans: array<f32>;
@group(0) @binding(4) var<storage, read_write> ptsGlobal: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let plane = p.H * p.pts;
  let total = p.N * plane;
  if (idx >= total) { return; }
  let hp = idx % plane;
  let i = idx / plane;

  let base = i * 3u * plane + hp;
  let lx = ptsLocal[base];
  let ly = ptsLocal[base + plane];
  let lz = ptsLocal[base + 2u * plane];

  let r = i * 9u;
  let t = i * 3u;
  let gx = rots[r + 0u] * lx + rots[r + 1u] * ly + rots[r + 2u] * lz + trans[t + 0u];
  let gy = rots[r + 3u] * lx + rots[r + 4u] * ly + rots[r + 5u] * lz + trans[t + 1u];
  let gz = rots[r + 6u] * lx + rots[r + 7u] * ly + rots[r + 8u] * lz + trans[t + 2u];

  let outBase = idx * 3u;
  ptsGlobal[outBase + 0u] = gx;
  ptsGlobal[outBase + 1u] = gy;
  ptsGlobal[outBase + 2u] = gz;
}
`;

/**
 * Full AF2 IPA attention logits:
 *   attn[h, i, j] = scalarFactor * dot(q[i, h], k[j, h])
 *                 + pairFactor * b[(i * N + j) * H + h]
 *                 - 0.5 * softplus(headWeights[h]) * pointFactor * sum_{p=0}^{nQkPts-1} ||qPts[i,h,p] - kPts[j,h,p]||^2
 *                 + inf * (mask[i] * mask[j] - 1)
 */
export const FULL_IPA_LOGITS_SOURCE = `
struct P {
  N: u32, H: u32, cH: u32, nQkPts: u32,
  nKvPts: u32, pad0: u32, pad1: u32, pad2: u32,
  scalarFactor: f32, pairFactor: f32, pointFactor: f32, infVal: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read> qPts: array<f32>;
@group(0) @binding(4) var<storage, read> kvPts: array<f32>;
@group(0) @binding(5) var<storage, read> b: array<f32>;
@group(0) @binding(6) var<storage, read> headWeights: array<f32>;
@group(0) @binding(7) var<storage, read> mask: array<f32>;
@group(0) @binding(8) var<storage, read_write> attn: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let total = p.H * p.N * p.N;
  if (idx >= total) { return; }
  let j = idx % p.N;
  let rem = idx / p.N;
  let i = rem % p.N;
  let h = rem / p.N;

  let qBase = (i * p.H + h) * p.cH;
  let kBase = (j * p.H + h) * (2u * p.cH);
  var dotQK = 0.0;
  for (var d = 0u; d < p.cH; d += 1u) {
    dotQK += q[qBase + d] * kv[kBase + d];
  }

  let qpBase = (i * p.H + h) * p.nQkPts * 3u;
  let kpBase = (j * p.H + h) * p.nKvPts * 3u;
  var dist2 = 0.0;
  for (var pt = 0u; pt < p.nQkPts; pt += 1u) {
    let dx = qPts[qpBase + pt * 3u + 0u] - kvPts[kpBase + pt * 3u + 0u];
    let dy = qPts[qpBase + pt * 3u + 1u] - kvPts[kpBase + pt * 3u + 1u];
    let dz = qPts[qpBase + pt * 3u + 2u] - kvPts[kpBase + pt * 3u + 2u];
    dist2 += dx * dx + dy * dy + dz * dz;
  }

  let hw = headWeights[h];
  let sp = select(log(1.0 + exp(hw)), hw, hw > 20.0);
  let ptTerm = -0.5 * (sp * p.pointFactor) * dist2;
  let pairBias = b[(i * p.N + j) * p.H + h];
  let sqMask = mask[i] * mask[j];
  attn[idx] = dotQK * p.scalarFactor + pairBias * p.pairFactor + ptTerm + p.infVal * (sqMask - 1.0);
}
`;

/**
 * Scalar attention output written into the leading `[0 .. H*cH)` slice of `concat[i, :]`.
 */
export const IPA_SCALAR_OUT_SOURCE = `
struct P { M: u32, H: u32, cH: u32, concatDim: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> attn: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read_write> concat: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let hc = p.H * p.cH;
  if (idx >= p.M * hc) { return; }
  let d = idx % p.cH;
  let rem = idx / p.cH;
  let h = rem % p.H;
  let i = rem / p.H;

  let attnBase = (h * p.M + i) * p.M;
  var acc = 0.0;
  for (var j = 0u; j < p.M; j += 1u) {
    let vIdx = (j * p.H + h) * (2u * p.cH) + p.cH + d;
    acc += attn[attnBase + j] * kv[vIdx];
  }
  concat[i * p.concatDim + h * p.cH + d] = acc;
}
`;

/**
 * Point attention output for Full AF2 IPA:
 *   g[i, h, pt] = sum_j attn[h, i, j] * vPts[j, h, pt]
 *   local = R_i^T * (g - t_i)
 *   norm  = sqrt(dot(local, local) + eps)
 *
 * Writes into `concat[i, :]` matching PyTorch `torch.cat((o, *unbind(o_pt, -1), o_pt_norm, o_pair), -1)`:
 *   x plane at `ptOff + hp`
 *   y plane at `ptOff + HP + hp`
 *   z plane at `ptOff + 2*HP + hp`
 *   norm    at `ptOff + 3*HP + hp`
 */
export const IPA_POINT_OUT_SOURCE = `
struct P {
  N: u32, H: u32, nQkPts: u32, nVPts: u32,
  ptOff: u32, concatDim: u32, eps: f32, pad0: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> attn: array<f32>;
@group(0) @binding(2) var<storage, read> kvPts: array<f32>;
@group(0) @binding(3) var<storage, read> rots: array<f32>;
@group(0) @binding(4) var<storage, read> trans: array<f32>;
@group(0) @binding(5) var<storage, read_write> concat: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let hpCount = p.H * p.nVPts;
  if (idx >= p.N * hpCount) { return; }
  let pt = idx % p.nVPts;
  let rem = idx / p.nVPts;
  let h = rem % p.H;
  let i = rem / p.H;

  let attnBase = (h * p.N + i) * p.N;
  let nKvPts = p.nQkPts + p.nVPts;
  var gx = 0.0;
  var gy = 0.0;
  var gz = 0.0;
  for (var j = 0u; j < p.N; j += 1u) {
    let a = attn[attnBase + j];
    let vBase = ((j * p.H + h) * nKvPts + p.nQkPts + pt) * 3u;
    gx += a * kvPts[vBase + 0u];
    gy += a * kvPts[vBase + 1u];
    gz += a * kvPts[vBase + 2u];
  }

  let t = i * 3u;
  let dx = gx - trans[t + 0u];
  let dy = gy - trans[t + 1u];
  let dz = gz - trans[t + 2u];

  let r = i * 9u;
  // R_i^T * d
  let lx = rots[r + 0u] * dx + rots[r + 3u] * dy + rots[r + 6u] * dz;
  let ly = rots[r + 1u] * dx + rots[r + 4u] * dy + rots[r + 7u] * dz;
  let lz = rots[r + 2u] * dx + rots[r + 5u] * dy + rots[r + 8u] * dz;
  let nrm = sqrt(lx * lx + ly * ly + lz * lz + p.eps);

  let hp = h * p.nVPts + pt;
  let base = i * p.concatDim + p.ptOff;
  concat[base + hp] = lx;
  concat[base + hpCount + hp] = ly;
  concat[base + 2u * hpCount + hp] = lz;
  concat[base + 3u * hpCount + hp] = nrm;
}
`;

/**
 * Pair attention output written into `concat[i, pairOff + h * cZ + c]`:
 *   o_pair[i, h, c] = sum_j attn[h, i, j] * z[(i * M + j) * cZ + c]
 */
export const IPA_PAIR_OUT_SOURCE = `
struct P { M: u32, H: u32, cZ: u32, pairOff: u32, concatDim: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> attn: array<f32>;
@group(0) @binding(2) var<storage, read> z: array<f32>;
@group(0) @binding(3) var<storage, read_write> concat: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let hz = p.H * p.cZ;
  if (idx >= p.M * hz) { return; }
  let c = idx % p.cZ;
  let rem = idx / p.cZ;
  let h = rem % p.H;
  let i = rem / p.H;

  let attnBase = (h * p.M + i) * p.M;
  let zRowBase = i * p.M * p.cZ + c;
  var acc = 0.0;
  for (var j = 0u; j < p.M; j += 1u) {
    acc += attn[attnBase + j] * z[zRowBase + j * p.cZ];
  }
  concat[i * p.concatDim + p.pairOff + h * p.cZ + c] = acc;
}
`;

/**
 * BackboneUpdate + SE(3) frame composition in-place:
 *   params[i] = [b, c, d, tx, ty, tz]
 *   q = [1, b, c, d] / sqrt(1 + b^2 + c^2 + d^2)
 *   delta_R = quat_to_rot(q)
 *   R_new = R_old @ delta_R
 *   t_new = R_old @ [tx, ty, tz] + t_old
 */
export const BACKBONE_UPDATE_COMPOSE_SOURCE = `
struct P { N: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> rots: array<f32>;
@group(0) @binding(3) var<storage, read_write> trans: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 64u + li;
  if (i >= p.N) { return; }

  let pb = i * 6u;
  let b = params[pb + 0u];
  let c = params[pb + 1u];
  let d = params[pb + 2u];
  let tx = params[pb + 3u];
  let ty = params[pb + 4u];
  let tz = params[pb + 5u];

  let invNorm = inverseSqrt(1.0 + b * b + c * c + d * d);
  let qa = invNorm;
  let qb = b * invNorm;
  let qc = c * invNorm;
  let qd = d * invNorm;

  // Exact match to affine_utils.py quat_to_rot
  let d00 = qa * qa + qb * qb - qc * qc - qd * qd;
  let d01 = 2.0 * (qb * qc - qa * qd);
  let d02 = 2.0 * (qb * qd + qa * qc);
  let d10 = 2.0 * (qb * qc + qa * qd);
  let d11 = qa * qa - qb * qb + qc * qc - qd * qd;
  let d12 = 2.0 * (qc * qd - qa * qb);
  let d20 = 2.0 * (qb * qd - qa * qc);
  let d21 = 2.0 * (qc * qd + qa * qb);
  let d22 = qa * qa - qb * qb - qc * qc + qd * qd;

  let r = i * 9u;
  let a00 = rots[r + 0u]; let a01 = rots[r + 1u]; let a02 = rots[r + 2u];
  let a10 = rots[r + 3u]; let a11 = rots[r + 4u]; let a12 = rots[r + 5u];
  let a20 = rots[r + 6u]; let a21 = rots[r + 7u]; let a22 = rots[r + 8u];

  rots[r + 0u] = a00 * d00 + a01 * d10 + a02 * d20;
  rots[r + 1u] = a00 * d01 + a01 * d11 + a02 * d21;
  rots[r + 2u] = a00 * d02 + a01 * d12 + a02 * d22;
  rots[r + 3u] = a10 * d00 + a11 * d10 + a12 * d20;
  rots[r + 4u] = a10 * d01 + a11 * d11 + a12 * d21;
  rots[r + 5u] = a10 * d02 + a11 * d12 + a12 * d22;
  rots[r + 6u] = a20 * d00 + a21 * d10 + a22 * d20;
  rots[r + 7u] = a20 * d01 + a21 * d11 + a22 * d21;
  rots[r + 8u] = a20 * d02 + a21 * d12 + a22 * d22;

  let t = i * 3u;
  trans[t + 0u] = a00 * tx + a01 * ty + a02 * tz + trans[t + 0u];
  trans[t + 1u] = a10 * tx + a11 * ty + a12 * tz + trans[t + 1u];
  trans[t + 2u] = a20 * tx + a21 * ty + a22 * tz + trans[t + 2u];
}
`;

/**
 * TriangleMult gate + mask + channel-major scatter:
 * Reads `ab[rowLocal, 0..4*cH)` = [a_p, a_g, b_p, b_g] for a chunk starting at `r0`,
 * multiplies by `pairMask[r0 + rowLocal]`, and writes:
 *   aT[c * nn + r0 + rowLocal] = a_p * sigmoid(a_g) * mask
 *   bT[c * nn + r0 + rowLocal] = b_p * sigmoid(b_g) * mask
 */
export const TRI_GATE_SCATTER_SOURCE = `
struct P { rows: u32, cH: u32, nn: u32, r0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ab: array<f32>;
@group(0) @binding(2) var<storage, read> pairMask: array<f32>;
@group(0) @binding(3) var<storage, read_write> aT: array<f32>;
@group(0) @binding(4) var<storage, read_write> bT: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  if (idx >= p.rows * p.cH) { return; }
  let c = idx % p.cH;
  let rLocal = idx / p.cH;
  let rGlobal = p.r0 + rLocal;
  let m = pairMask[rGlobal];
  let base = rLocal * (4u * p.cH);
  let ap = ab[base + c];
  let ag = ab[base + p.cH + c];
  let bp = ab[base + 2u * p.cH + c];
  let bg = ab[base + 3u * p.cH + c];
  let outIdx = c * p.nn + rGlobal;
  aT[outIdx] = (ap / (1.0 + exp(-ag))) * m;
  bT[outIdx] = (bp / (1.0 + exp(-bg))) * m;
}
`;

/**
 * Record ReducedIPA steps into a program list (accumulates directly into `s` in-place!).
 */
export async function recordReducedIpa(ops, {
  s, p, mask, w, B, M, cS, cZ, cH, H, inf = 1e5,
}) {
  const concatDim = H * (cH + cZ);
  const uLogits = new Uint32Array(8);
  const fLogits = new Float32Array(uLogits.buffer);
  uLogits.set([M, H, cH, 0]);
  fLogits[4] = Math.sqrt(1.0 / (2 * cH));
  fLogits[5] = Math.sqrt(0.5);
  fLogits[6] = inf;

  const uScalar = new Uint32Array([M, H, cH, concatDim]);
  const uPair = new Uint32Array([M, H, cZ, H * cH, concatDim, 0, 0, 0]);

  return [
    await ops.linear({ x: s, w: w.qW, bias: w.qB, y: B.ipaQ, rows: M, inC: cS, outC: H * cH, label: 'ripa.q' }),
    await ops.linear({ x: s, w: w.kvW, bias: w.kvB, y: B.ipaKv, rows: M, inC: cS, outC: 2 * H * cH, label: 'ripa.kv' }),
    await ops.linear({ x: p, w: w.bW, bias: w.bB, y: B.ipaB, rows: M * M, inC: cZ, outC: H, label: 'ripa.b' }),
    await ops.step(
      'reduced-ipa-logits',
      REDUCED_IPA_LOGITS_SOURCE,
      Math.ceil((H * M * M) / 256),
      [B.ipaQ, B.ipaKv, B.ipaB, mask, B.ipaAttn],
      uLogits,
      'ripa.logits',
    ),
    await ops.softmaxRows({ data: B.ipaAttn, rows: H * M, len: M, label: 'ripa.softmax' }),
    await ops.step(
      'ipa-scalar-out',
      IPA_SCALAR_OUT_SOURCE,
      Math.ceil((M * H * cH) / 256),
      [B.ipaAttn, B.ipaKv, B.ipaConcat],
      uScalar,
      'ripa.scalarOut',
    ),
    await ops.step(
      'ipa-pair-out',
      IPA_PAIR_OUT_SOURCE,
      Math.ceil((M * H * cZ) / 256),
      [B.ipaAttn, p, B.ipaConcat],
      uPair,
      'ripa.pairOut',
    ),
    await ops.linear({
      x: B.ipaConcat, w: w.outW, bias: w.outB, y: s,
      rows: M, inC: concatDim, outC: cS, accumulate: true, label: 'ripa.out',
    }),
  ];
}

/**
 * Record Full AF2 IPA steps into a program list (accumulates directly into `s` in-place!).
 */
export async function recordFullIpa(ops, {
  s, z, rots, trans, mask, w, B, N, cS, cZ, cH, H, nQkPts, nVPts, inf = 1e5, eps = 1e-8,
}) {
  const nKvPts = nQkPts + nVPts;
  const concatDim = H * (cH + 4 * nVPts + cZ);

  const uQPtApply = new Uint32Array([N, H, nQkPts, 0]);
  const uKvPtApply = new Uint32Array([N, H, nKvPts, 0]);

  const uLogits = new Uint32Array(12);
  const fLogits = new Float32Array(uLogits.buffer);
  uLogits.set([N, H, cH, nQkPts, nKvPts, 0, 0, 0]);
  fLogits[8] = Math.sqrt(1.0 / (3 * cH));
  fLogits[9] = Math.sqrt(1.0 / 3.0);
  fLogits[10] = Math.sqrt(1.0 / (3 * nQkPts * 4.5));
  fLogits[11] = inf;

  const uScalar = new Uint32Array([N, H, cH, concatDim]);
  const uPtOut = new Uint32Array(8);
  const fPtOut = new Float32Array(uPtOut.buffer);
  uPtOut.set([N, H, nQkPts, nVPts, H * cH, concatDim]);
  fPtOut[6] = eps;

  const uPair = new Uint32Array([N, H, cZ, H * cH + 4 * H * nVPts, concatDim, 0, 0, 0]);

  return [
    await ops.linear({ x: s, w: w.qW, bias: w.qB, y: B.ipaQ, rows: N, inC: cS, outC: H * cH, label: 'ipa.q' }),
    await ops.linear({ x: s, w: w.kvW, bias: w.kvB, y: B.ipaKv, rows: N, inC: cS, outC: 2 * H * cH, label: 'ipa.kv' }),
    await ops.linear({ x: s, w: w.qPtsW, bias: w.qPtsB, y: B.ipaPtsLocal, rows: N, inC: cS, outC: 3 * H * nQkPts, label: 'ipa.qPts' }),
    await ops.step(
      'ipa-point-apply',
      IPA_POINT_APPLY_SOURCE,
      Math.ceil((N * H * nQkPts) / 256),
      [B.ipaPtsLocal, rots, trans, B.ipaQPts],
      uQPtApply,
      'ipa.qPtsApply',
    ),
    await ops.linear({ x: s, w: w.kvPtsW, bias: w.kvPtsB, y: B.ipaPtsLocal, rows: N, inC: cS, outC: 3 * H * nKvPts, label: 'ipa.kvPts' }),
    await ops.step(
      'ipa-point-apply',
      IPA_POINT_APPLY_SOURCE,
      Math.ceil((N * H * nKvPts) / 256),
      [B.ipaPtsLocal, rots, trans, B.ipaKvPts],
      uKvPtApply,
      'ipa.kvPtsApply',
    ),
    await ops.linear({ x: z, w: w.bW, bias: w.bB, y: B.ipaB, rows: N * N, inC: cZ, outC: H, label: 'ipa.b' }),
    await ops.step(
      'full-ipa-logits',
      FULL_IPA_LOGITS_SOURCE,
      Math.ceil((H * N * N) / 256),
      [B.ipaQ, B.ipaKv, B.ipaQPts, B.ipaKvPts, B.ipaB, w.headWeights, mask, B.ipaAttn],
      uLogits,
      'ipa.logits',
    ),
    await ops.softmaxRows({ data: B.ipaAttn, rows: H * N, len: N, label: 'ipa.softmax' }),
    await ops.step(
      'ipa-scalar-out',
      IPA_SCALAR_OUT_SOURCE,
      Math.ceil((N * H * cH) / 256),
      [B.ipaAttn, B.ipaKv, B.ipaConcat],
      uScalar,
      'ipa.scalarOut',
    ),
    await ops.step(
      'ipa-point-out',
      IPA_POINT_OUT_SOURCE,
      Math.ceil((N * H * nVPts) / 256),
      [B.ipaAttn, B.ipaKvPts, rots, trans, B.ipaConcat],
      uPtOut,
      'ipa.ptOut',
    ),
    await ops.step(
      'ipa-pair-out',
      IPA_PAIR_OUT_SOURCE,
      Math.ceil((N * H * cZ) / 256),
      [B.ipaAttn, z, B.ipaConcat],
      uPair,
      'ipa.pairOut',
    ),
    await ops.linear({
      x: B.ipaConcat, w: w.outW, bias: w.outB, y: s,
      rows: N, inC: concatDim, outC: cS, accumulate: true, label: 'ipa.out',
    }),
  ];
}

/**
 * Record BackboneUpdate + SE(3) frame compose in-place on `(rots, trans)`.
 */
export async function recordBackboneUpdate(ops, { s, rots, trans, w, B, N, cS }) {
  const u = new Uint32Array([N, 0, 0, 0]);
  return [
    await ops.linear({ x: s, w: w.w, bias: w.b, y: B.bbParams, rows: N, inC: cS, outC: 6, label: 'bb.linear' }),
    await ops.step(
      'bb-update-compose',
      BACKBONE_UPDATE_COMPOSE_SOURCE,
      Math.ceil(N / 64),
      [B.bbParams, rots, trans],
      u,
      'bb.compose',
    ),
  ];
}
