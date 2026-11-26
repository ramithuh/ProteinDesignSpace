// WGSL kernels for the cg2all SE(3)-Transformer. Every kernel reads a 256-byte uniform slot and addresses one
// activation arena `A`, so the six pipelines are shape-independent and created once per device.
// Bindings (shared layout): 0 uniform params · 1 W f32 weights · 2 A f32 activations (rw)
//                           3 G u32 graph (src, csr offsets, csr edges) · 4 B f32 basis + edge features
//                           5 W4 = W viewed as vec4 (radial output rows are 32 floats, 16-byte aligned)

const HEADER = /* wgsl */ `
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
fn elu(y: f32) -> f32 { return select(exp(y) - 1.0, y, y > 0.0); }
fn flatIndex(gid: vec3<u32>, nwg: vec3<u32>, size: u32) -> u32 { return gid.x + gid.y * nwg.x * size; }
`;

/** LinearSE3 for one degree on the channel concat [a | b]: out[n][co][m] = Σ W[co][ci] · in[n][ci][m]. */
export const LINEAR_WGSL = /* wgsl */ `
struct Params { total: u32, cOut: u32, cInA: u32, cInB: u32, M: u32, offA: u32, offB: u32, offOut: u32, offW: u32 };
@group(0) @binding(0) var<uniform> P: Params;
${HEADER}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = flatIndex(gid, nwg, 64u);
  if (idx >= P.total) { return; }
  let m = idx % P.M;
  let t = idx / P.M;
  let co = t % P.cOut;
  let n = t / P.cOut;
  let wb = P.offW + co * (P.cInA + P.cInB);
  var s = 0.0;
  for (var ci = 0u; ci < P.cInA; ci++) { s += W[wb + ci] * A[P.offA + (n * P.cInA + ci) * P.M + m]; }
  for (var ci = 0u; ci < P.cInB; ci++) { s += W[wb + P.cInA + ci] * A[P.offB + (n * P.cInB + ci) * P.M + m]; }
  A[P.offOut + idx] = s;
}
`;

/** NormSE3 for one degree: LayerNorm over channel norms (clamped at 2^-24), ELU, rescale. */
export const NORM_WGSL = /* wgsl */ `
struct Params { n: u32, C: u32, M: u32, offIn: u32, offOut: u32, offG: u32, offB: u32 };
@group(0) @binding(0) var<uniform> P: Params;
${HEADER}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let n = flatIndex(gid, nwg, 64u);
  if (n >= P.n) { return; }
  var nr: array<f32, 64>;
  let base = n * P.C * P.M;
  var mean = 0.0;
  for (var c = 0u; c < P.C; c++) {
    var s = 0.0;
    for (var m = 0u; m < P.M; m++) { let v = A[P.offIn + base + c * P.M + m]; s += v * v; }
    nr[c] = max(sqrt(s), 5.9604644775390625e-8);
    mean += nr[c];
  }
  mean /= f32(P.C);
  var vs = 0.0;
  for (var c = 0u; c < P.C; c++) { let d = nr[c] - mean; vs += d * d; }
  let inv = inverseSqrt(vs / f32(P.C) + 1e-5);
  for (var c = 0u; c < P.C; c++) {
    let f = elu((nr[c] - mean) * inv * W[P.offG + c] + W[P.offB + c]) / nr[c];
    for (var m = 0u; m < P.M; m++) { A[P.offOut + base + c * P.M + m] = A[P.offIn + base + c * P.M + m] * f; }
  }
}
`;

/**
 * Fused TFN convolution for one output degree over a tile of 16 edges (edge-level output, pool=False):
 * radial MLP (4→32 LN ELU →32 LN ELU) per block, tmp = x_src ⊗ basis staged in workgroup memory, radial
 * weights generated on the fly from the 32-float rows of the last linear layer and contracted immediately.
 * Each of the 256 invocations owns (c_out lane, two edges); W2 rows are read once per pair of edges.
 */
export const CONV_WGSL = /* wgsl */ `
const TE: u32 = 16u;
const CC: u32 = 4u;
const NONE: u32 = 0xffffffffu;
struct Block { dIn: u32, cIn: u32, L: u32, F: u32, fOff: u32, fCount: u32, offX: u32, basisOff: u32, basisSize: u32, mlpOff: u32, w2Off: u32, pad: u32 };
struct Params { E: u32, edgeBase: u32, cOut: u32, K: u32, offOut: u32, nBlocks: u32, featOff: u32, pad: u32, blocks: array<Block, 3> };
@group(0) @binding(0) var<uniform> P: Params;
${HEADER}
@group(0) @binding(3) var<storage, read> G: array<u32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read> W4: array<vec4<f32>>;

var<workgroup> h0: array<f32, 512>;
var<workgroup> h1: array<f32, 512>;
var<workgroup> stat: array<f32, 32>;
var<workgroup> tmp: array<f32, 1600>;

fn layerStats(lid: u32, buf: u32) {
  if (lid < TE) {
    var mean = 0.0;
    for (var j = 0u; j < 32u; j++) { mean += select(h1[lid * 32u + j], h0[lid * 32u + j], buf == 0u); }
    mean /= 32.0;
    var vs = 0.0;
    for (var j = 0u; j < 32u; j++) {
      let d = select(h1[lid * 32u + j], h0[lid * 32u + j], buf == 0u) - mean;
      vs += d * d;
    }
    stat[2u * lid] = mean;
    stat[2u * lid + 1u] = inverseSqrt(vs / 32.0 + 1e-5);
  }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let e0 = P.edgeBase + wid.x * TE;
  let lane = lid % 32u;
  let eg = lid / 32u;
  let me = lid / 16u;
  let mj = (lid % 16u) * 2u;
  var acc: array<f32, 20>;
  for (var i = 0u; i < 20u; i++) { acc[i] = 0.0; }
  var prevMlp = NONE;
  for (var bi = 0u; bi < P.nBlocks; bi++) {
    let b = P.blocks[bi];
    if (b.mlpOff != prevMlp) {
      prevMlp = b.mlpOff;
      let e = e0 + me;
      let fo = P.featOff + min(e, P.E - 1u) * 4u;
      for (var q = 0u; q < 2u; q++) {
        let j = mj + q;
        var s = W[b.mlpOff + 128u + j];
        for (var i = 0u; i < 4u; i++) { s += W[b.mlpOff + j * 4u + i] * B[fo + i]; }
        h0[me * 32u + j] = s;
      }
      workgroupBarrier();
      layerStats(lid, 0u);
      workgroupBarrier();
      for (var q = 0u; q < 2u; q++) {
        let j = mj + q;
        let y = (h0[me * 32u + j] - stat[2u * me]) * stat[2u * me + 1u];
        h0[me * 32u + j] = elu(y * W[b.mlpOff + 160u + j] + W[b.mlpOff + 192u + j]);
      }
      workgroupBarrier();
      for (var q = 0u; q < 2u; q++) {
        let j = mj + q;
        var s = W[b.mlpOff + 1248u + j];
        for (var i = 0u; i < 32u; i++) { s += W[b.mlpOff + 224u + j * 32u + i] * h0[me * 32u + i]; }
        h1[me * 32u + j] = s;
      }
      workgroupBarrier();
      layerStats(lid, 1u);
      workgroupBarrier();
      for (var q = 0u; q < 2u; q++) {
        let j = mj + q;
        let y = (h1[me * 32u + j] - stat[2u * me]) * stat[2u * me + 1u];
        h1[me * 32u + j] = elu(y * W[b.mlpOff + 1280u + j] + W[b.mlpOff + 1312u + j]);
      }
      workgroupBarrier();
    }
    let fc = b.fCount;
    let cnt = TE * CC * fc * P.K;
    for (var ci0 = 0u; ci0 < b.cIn; ci0 += CC) {
      for (var t = lid; t < cnt; t += 256u) {
        let k = t % P.K;
        var r = t / P.K;
        let f = r % fc;
        r = r / fc;
        let cc = r % CC;
        let el = r / CC;
        let e = e0 + el;
        let ci = ci0 + cc;
        var v = 0.0;
        if (e < P.E && ci < b.cIn) {
          let xo = b.offX + (G[e] * b.cIn + ci) * b.L;
          if (b.basisOff == NONE) {
            v = A[xo];
          } else {
            let bo = b.basisOff + e * b.basisSize + f * P.K + k;
            for (var l = 0u; l < b.L; l++) { v += A[xo + l] * B[bo + l * fc * P.K]; }
          }
        }
        tmp[t] = v;
      }
      workgroupBarrier();
      for (var rr = 0u; rr < 2u; rr++) {
        let co = lane + 32u * rr;
        if (co < P.cOut) {
          for (var cc = 0u; cc < CC; cc++) {
            let ci = ci0 + cc;
            if (ci < b.cIn) {
              for (var f = 0u; f < fc; f++) {
                let r4 = (b.w2Off + ((co * b.cIn + ci) * b.F + b.fOff + f) * 32u) / 4u;
                var s0 = 0.0;
                var s1 = 0.0;
                for (var jj = 0u; jj < 8u; jj++) {
                  let w = W4[r4 + jj];
                  let ha = eg * 32u + jj * 4u;
                  let hb = (eg + 8u) * 32u + jj * 4u;
                  s0 += w.x * h1[ha] + w.y * h1[ha + 1u] + w.z * h1[ha + 2u] + w.w * h1[ha + 3u];
                  s1 += w.x * h1[hb] + w.y * h1[hb + 1u] + w.z * h1[hb + 2u] + w.w * h1[hb + 3u];
                }
                let ta = ((eg * CC + cc) * fc + f) * P.K;
                let tb = (((eg + 8u) * CC + cc) * fc + f) * P.K;
                for (var k = 0u; k < P.K; k++) {
                  acc[rr * 10u + k] += s0 * tmp[ta + k];
                  acc[rr * 10u + 5u + k] += s1 * tmp[tb + k];
                }
              }
            }
          }
        }
      }
      workgroupBarrier();
    }
  }
  for (var rr = 0u; rr < 2u; rr++) {
    let co = lane + 32u * rr;
    for (var q = 0u; q < 2u; q++) {
      let e = e0 + eg + 8u * q;
      if (co < P.cOut && e < P.E) {
        for (var k = 0u; k < P.K; k++) { A[P.offOut + (e * P.cOut + co) * P.K + k] = acc[rr * 10u + q * 5u + k]; }
      }
    }
  }
}
`;

/** copy_e_sum of edge outputs onto destination nodes plus in-degree × self-interaction (final ConvSE3). */
export const POOL_WGSL = /* wgsl */ `
struct Params { total: u32, cOut: u32, K: u32, cIn: u32, offEdge: u32, offX: u32, offOut: u32, offK: u32, hasSelf: u32, csrOff: u32, csrEdges: u32 };
@group(0) @binding(0) var<uniform> P: Params;
${HEADER}
@group(0) @binding(3) var<storage, read> G: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = flatIndex(gid, nwg, 64u);
  if (idx >= P.total) { return; }
  let k = idx % P.K;
  let t = idx / P.K;
  let co = t % P.cOut;
  let n = t / P.cOut;
  let a = G[P.csrOff + n];
  let b = G[P.csrOff + n + 1u];
  var s = 0.0;
  for (var p = a; p < b; p++) { s += A[P.offEdge + (G[P.csrEdges + p] * P.cOut + co) * P.K + k]; }
  if (P.hasSelf == 1u && b > a) {
    var ks = 0.0;
    for (var ci = 0u; ci < P.cIn; ci++) { ks += W[P.offK + co * P.cIn + ci] * A[P.offX + (n * P.cIn + ci) * P.K + k]; }
    s += f32(b - a) * ks;
  }
  A[P.offOut + idx] = s;
}
`;

/** Multi-head graph attention (AttentionSE3): one invocation per (node, head), softmax over incoming edges. */
export const ATTN_WGSL = /* wgsl */ `
struct Params { n: u32, heads: u32, csrOff: u32, csrEdges: u32, cv: vec4<u32>, ckv: vec4<u32>, hasKey: vec4<u32>,
  offKV: vec4<u32>, offQ: vec4<u32>, offZ: vec4<u32>, scale: f32, nDeg: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
${HEADER}
@group(0) @binding(3) var<storage, read> G: array<u32>;

fn logit(e: u32, n: u32, h: u32) -> f32 {
  var s = 0.0;
  for (var d = 0u; d < P.nDeg; d++) {
    if (P.hasKey[d] == 1u) {
      let M = 2u * d + 1u;
      let per = P.cv[d] / P.heads;
      for (var c = h * per; c < (h + 1u) * per; c++) {
        for (var m = 0u; m < M; m++) {
          s += A[P.offKV[d] + (e * P.ckv[d] + P.cv[d] + c) * M + m] * A[P.offQ[d] + (n * P.cv[d] + c) * M + m];
        }
      }
    }
  }
  return s * P.scale;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = flatIndex(gid, nwg, 64u);
  if (idx >= P.n * P.heads) { return; }
  let n = idx / P.heads;
  let h = idx % P.heads;
  let a = G[P.csrOff + n];
  let b = G[P.csrOff + n + 1u];
  var mx = -3.0e38;
  for (var p = a; p < b; p++) { mx = max(mx, logit(G[P.csrEdges + p], n, h)); }
  var sum = 0.0;
  for (var p = a; p < b; p++) { sum += exp(logit(G[P.csrEdges + p], n, h) - mx); }
  for (var d = 0u; d < P.nDeg; d++) {
    let M = 2u * d + 1u;
    let per = P.cv[d] / P.heads;
    for (var c = h * per; c < (h + 1u) * per; c++) {
      for (var m = 0u; m < M; m++) { A[P.offZ[d] + (n * P.cv[d] + c) * M + m] = 0.0; }
    }
  }
  for (var p = a; p < b; p++) {
    let e = G[P.csrEdges + p];
    let w = exp(logit(e, n, h) - mx) / sum;
    for (var d = 0u; d < P.nDeg; d++) {
      let M = 2u * d + 1u;
      let per = P.cv[d] / P.heads;
      for (var c = h * per; c < (h + 1u) * per; c++) {
        for (var m = 0u; m < M; m++) {
          let zi = P.offZ[d] + (n * P.cv[d] + c) * M + m;
          A[zi] = A[zi] + w * A[P.offKV[d] + (e * P.ckv[d] + c) * M + m];
        }
      }
    }
  }
}
`;

/** Residual add: A[offA + i] += A[offB + i]. */
export const ADD_WGSL = /* wgsl */ `
struct Params { total: u32, offA: u32, offB: u32 };
@group(0) @binding(0) var<uniform> P: Params;
${HEADER}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = flatIndex(gid, nwg, 64u);
  if (idx >= P.total) { return; }
  A[P.offA + idx] = A[P.offA + idx] + A[P.offB + idx];
}
`;

export const KERNELS = {
  linear: LINEAR_WGSL,
  norm: NORM_WGSL,
  conv: CONV_WGSL,
  pool: POOL_WGSL,
  attn: ATTN_WGSL,
  add: ADD_WGSL,
};
