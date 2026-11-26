// cg2all graph & feature construction (libcg.BaseClass.get_geometry / geom_to_feature, libdata.PredictionData)
// plus the SE(3)-Transformer edge basis (se3_transformer/basis.py with e3nn 0.5.1 spherical harmonics).
// All lengths are in nm, as in cg2all. Arrays are flat, row-major, Float64.

import { WIGNER_3J } from './constants.js';

export const EPS = 1e-6;
/** Neighbour radius (nm) of the CalphaBasedModel graph and of the n_neigh feature. */
export const GRAPH_RADIUS = 1.0;
/** Cα–Cα distance (nm) above which consecutive residues of one chain are treated as a chain break. */
export const CHAIN_BREAK_CUTOFF = 1.0;
/** Maximum feature degree of the SE(3)-Transformer (fiber_hidden has degrees 0, 1, 2). */
export const MAX_DEGREE = 2;
export const N_NODE_SCALAR = 17;
export const N_NODE_VECTOR = 4;
/** Invariant edge features: bonded / ssbond / space one-hot plus |r_ij|. */
export const N_EDGE_SCALAR = 4;

const norm3 = (x, y, z) => Math.sqrt(x * x + y * y + z * z);

/**
 * Backbone continuity (libcg.get_continuity_cg): `prev[i]` is true when residue i is bonded to i-1
 * (same chain and Cα–Cα ≤ cutoff), `next[i]` likewise for i+1.
 */
export function computeContinuity(pos, chainIndex, cutoff = CHAIN_BREAK_CUTOFF) {
  const n = chainIndex.length;
  const prev = new Uint8Array(n);
  const next = new Uint8Array(n);
  for (let i = 1; i < n; i++) {
    const d = norm3(pos[3 * i] - pos[3 * i - 3], pos[3 * i + 1] - pos[3 * i - 2], pos[3 * i + 2] - pos[3 * i - 1]);
    const cont = chainIndex[i] === chainIndex[i - 1] && !(d > cutoff);
    prev[i] = cont ? 1 : 0;
    next[i - 1] = cont ? 1 : 0;
  }
  return { prev, next };
}

/** DGL radius_graph(x, r, self_loop=False): edges (i -> j) for dist <= r, src-major order. */
export function radiusGraph(pos, radius = GRAPH_RADIUS) {
  const n = pos.length / 3;
  const src = [];
  const dst = [];
  const r2 = radius * radius;
  for (let i = 0; i < n; i++) {
    const xi = pos[3 * i];
    const yi = pos[3 * i + 1];
    const zi = pos[3 * i + 2];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = xi - pos[3 * j];
      const dy = yi - pos[3 * j + 1];
      const dz = zi - pos[3 * j + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      // Compare squared distances away from the boundary; take the exact sqrt only near it.
      if (d2 < r2 * 0.999 || (d2 <= r2 * 1.001 && Math.sqrt(d2) <= radius)) {
        src.push(i);
        dst.push(j);
      }
    }
  }
  return { src: Int32Array.from(src), dst: Int32Array.from(dst) };
}

function vNonzeroNorm(x, y, z, index = 0) {
  const s = norm3(x, y, z);
  if (s > EPS) return [x / s, y / s, z / s];
  return index === 0 ? [1, 0, 0] : index === 1 ? [0, 1, 0] : [0, 0, 1];
}

/** torch_basics.torsion_angle on four points (atan2 form with v_norm_safe / v_nonzero guards). */
function torsionAngle(p0, p1, p2, p3) {
  const b1 = vNonzeroNorm(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
  const b2 = vNonzeroNorm(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
  const b3 = vNonzeroNorm(p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]);
  let c1 = [b2[1] * b3[2] - b2[2] * b3[1], b2[2] * b3[0] - b2[0] * b3[2], b2[0] * b3[1] - b2[1] * b3[0]];
  if (!(norm3(c1[0], c1[1], c1[2]) > EPS)) c1 = [1, 0, 0];
  const c2 = [b1[1] * b2[2] - b1[2] * b2[1], b1[2] * b2[0] - b1[0] * b2[2], b1[0] * b2[1] - b1[1] * b2[0]];
  const p1v = b1[0] * c1[0] + b1[1] * c1[1] + b1[2] * c1[2];
  const p2v = c1[0] * c2[0] + c1[1] * c2[1] + c1[2] * c2[2];
  return Math.atan2(p1v, p2v);
}

/**
 * Node features (CalphaBasedModel): 17 scalars and 4 vectors per residue.
 * Follows libcg.get_geometry + geom_to_feature line by line, including two upstream quirks the
 * checkpoints were trained with: the last residue's "next" bond lengths are 0 (other chain ends get 1),
 * and the dihedral quadruplets come from `torch.stack([r[:-3], r[1:-2], r[2:-1], r[3:]]).reshape(-1, 4, 3)`,
 * i.e. consecutive rows of the stacked (4, N-3, 3) tensor rather than (r_i, r_i+1, r_i+2, r_i+3).
 */
export function nodeFeatures(pos, continuity) {
  const n = pos.length / 3;
  const notDefined = Uint8Array.from(continuity.prev, (c) => (c ? 0 : 1));
  const f0 = new Float64Array(n * N_NODE_SCALAR);
  const f1 = new Float64Array(n * N_NODE_VECTOR * 3);
  const P = (i) => [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];

  // n_neigh: in-degree in radius_graph(r, 1.0)
  const nNeigh = new Float64Array(n);
  const { dst } = radiusGraph(pos, GRAPH_RADIUS);
  for (const j of dst) nNeigh[j] += 1;

  const bondLen = {};
  const bondVec = {};
  for (const shift of [1, 2]) {
    const m = n + shift;
    const dr = new Float64Array(m * 3);
    const bl = new Float64Array(m);
    for (let i = 0; i < n - shift; i++) {
      const k = shift + i;
      dr[3 * k] = pos[3 * i] - pos[3 * (i + shift)];
      dr[3 * k + 1] = pos[3 * i + 1] - pos[3 * (i + shift) + 1];
      dr[3 * k + 2] = pos[3 * i + 2] - pos[3 * (i + shift) + 2];
      bl[k] = norm3(dr[3 * k], dr[3 * k + 1], dr[3 * k + 2]);
    }
    for (let k = 0; k < m; k++) {
      const s = Math.max(bl[k], EPS);
      dr[3 * k] /= s;
      dr[3 * k + 1] /= s;
      dr[3 * k + 2] /= s;
      bl[k] = Math.min(bl[k], 1.0);
    }
    for (let s = 0; s < shift; s++) {
      for (let i = 0; i < n; i++) {
        if (!notDefined[i]) continue;
        const k = s + i;
        dr[3 * k] = 0;
        dr[3 * k + 1] = 0;
        dr[3 * k + 2] = 0;
        bl[k] = 1.0;
      }
    }
    bondLen[shift] = bl;
    bondVec[shift] = dr;
  }

  for (let i = 0; i < n; i++) {
    const o = i * N_NODE_SCALAR;
    f0[o] = nNeigh[i];
    f0[o + 1] = continuity.prev[i] ? 1 : 0;
    f0[o + 2] = continuity.next[i] ? 1 : 0;
    f0[o + 3] = bondLen[1][i];
    f0[o + 4] = bondLen[1][i + 1];
    f0[o + 5] = bondLen[2][i];
    f0[o + 6] = bondLen[2][i + 2];
    // bond angle: v1 = bond_vector[1][0][i] = dr1[i], v2 = -dr1[i + 1]; "sine" is 1 - cos^2 upstream
    const v1 = [bondVec[1][3 * i], bondVec[1][3 * i + 1], bondVec[1][3 * i + 2]];
    const v2 = [-bondVec[1][3 * i + 3], -bondVec[1][3 * i + 4], -bondVec[1][3 * i + 5]];
    const cosine = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    const masked = notDefined[i] || i === n - 1 || (i < n - 1 && notDefined[i + 1]);
    f0[o + 7] = masked ? 0 : cosine;
    f0[o + 8] = masked ? 0 : 1.0 - cosine * cosine;
    // vectors: bv1 prev, bv1 next, bv2 prev, bv2 next
    const vo = i * N_NODE_VECTOR * 3;
    for (let c = 0; c < 3; c++) {
      f1[vo + c] = bondVec[1][3 * i + c];
      f1[vo + 3 + c] = -bondVec[1][3 * (i + 1) + c];
      f1[vo + 6 + c] = bondVec[2][3 * i + c];
      f1[vo + 9 + c] = -bondVec[2][3 * (i + 2) + c];
    }
  }

  // dihedral angles over the reshaped stack (see doc comment)
  const nd = Math.max(0, n - 3);
  if (nd > 0) {
    const stackRow = (q) => {
      const a = Math.floor(q / nd);
      const i = q % nd;
      return P(i + a);
    };
    const cosArr = new Float64Array(nd);
    const sinArr = new Float64Array(nd);
    for (let p = 0; p < nd; p++) {
      const t = torsionAngle(stackRow(4 * p), stackRow(4 * p + 1), stackRow(4 * p + 2), stackRow(4 * p + 3));
      cosArr[p] = Math.cos(t);
      sinArr[p] = Math.sin(t);
    }
    for (let i = 0; i < 3; i++) {
      for (let p = 0; p < nd - (i + 1); p++) {
        if (notDefined[i + 1 + p]) {
          cosArr[p] = 0;
          sinArr[p] = 0;
        }
      }
    }
    for (let slot = 0; slot < 4; slot++) {
      for (let p = 0; p < nd; p++) {
        const r = slot + p;
        f0[r * N_NODE_SCALAR + 9 + 2 * slot] = cosArr[p];
        f0[r * N_NODE_SCALAR + 10 + 2 * slot] = sinArr[p];
      }
    }
  }
  return { f0, f1 };
}

/**
 * Edge features (libdata.PredictionData): one-hot [bonded, ssbond, space] for every radius-graph edge,
 * plus |r_dst - r_src| appended by SE3Transformer.get_populated_edge_features, and rel_pos = r_dst - r_src.
 */
export function edgeFeatures(pos, graph, continuity, ssbonds) {
  const e = graph.src.length;
  const n = pos.length / 3;
  const feat = new Float64Array(e * N_EDGE_SCALAR);
  const relPos = new Float64Array(e * 3);
  const bonded = new Set();
  for (let i = 1; i < n; i++) {
    if (continuity.prev[i]) {
      bonded.add((i - 1) * n + i);
      bonded.add(i * n + i - 1);
    }
  }
  const ss = new Set();
  for (const [i, j] of ssbonds) {
    ss.add(i * n + j);
    ss.add(j * n + i);
  }
  for (let k = 0; k < e; k++) {
    const s = graph.src[k];
    const d = graph.dst[k];
    const key = s * n + d;
    const isBonded = bonded.has(key);
    const isSs = ss.has(key);
    feat[k * 4] = isBonded ? 1 : 0;
    feat[k * 4 + 1] = isSs ? 1 : 0;
    feat[k * 4 + 2] = isBonded || isSs ? 0 : 1;
    const dx = pos[3 * d] - pos[3 * s];
    const dy = pos[3 * d + 1] - pos[3 * s + 1];
    const dz = pos[3 * d + 2] - pos[3 * s + 2];
    relPos[3 * k] = dx;
    relPos[3 * k + 1] = dy;
    relPos[3 * k + 2] = dz;
    feat[k * 4 + 3] = norm3(dx, dy, dz);
  }
  return { feat, relPos };
}

/**
 * e3nn 0.5.1 real spherical harmonics, degrees 0..4 (normalize=True, normalization='integral'),
 * of a 3-vector; returns 25 values ordered [l=0 | l=1 (3) | l=2 (5) | l=3 (7) | l=4 (9)].
 */
export function sphericalHarmonics(vx, vy, vz, out = new Float64Array(25)) {
  const r = Math.max(norm3(vx, vy, vz), 1e-12);
  const x = vx / r;
  const y = vy / r;
  const z = vz / r;
  const s3 = Math.sqrt(3);
  const s15 = Math.sqrt(15);
  const s5 = Math.sqrt(5);
  const s7 = Math.sqrt(7);
  const s42 = Math.sqrt(42);
  const s168 = Math.sqrt(168);
  const s2 = Math.sqrt(2);
  const s6 = Math.sqrt(6);
  const s14 = Math.sqrt(14);
  const s21 = Math.sqrt(21);
  const s210 = Math.sqrt(210);
  const s105 = Math.sqrt(105);
  const s70 = Math.sqrt(70);
  const sh10 = s3 * x;
  const sh11 = s3 * y;
  const sh12 = s3 * z;
  const sh20 = s15 * x * z;
  const sh21 = s15 * x * y;
  const y2 = y * y;
  const x2z2 = x * x + z * z;
  const sh22 = s5 * (y2 - 0.5 * x2z2);
  const sh23 = s15 * y * z;
  const sh24 = 0.5 * s15 * (z * z - x * x);
  const sh30 = (1 / 6) * s42 * (sh20 * z + sh24 * x);
  const sh31 = s7 * sh20 * y;
  const sh32 = (1 / 8) * s168 * (4.0 * y2 - x2z2) * x;
  const sh33 = 0.5 * s7 * y * (2.0 * y2 - 3.0 * x2z2);
  const sh34 = (1 / 8) * s168 * z * (4.0 * y2 - x2z2);
  const sh35 = s7 * sh24 * y;
  const sh36 = (1 / 6) * s42 * (sh24 * z - sh20 * x);
  const sh40 = (3 / 4) * s2 * (sh30 * z + sh36 * x);
  const sh41 = (3 / 4) * sh30 * y + (3 / 8) * s6 * sh31 * z + (3 / 8) * s6 * sh35 * x;
  const sh42 =
    (-3 / 56) * s14 * sh30 * z +
    (3 / 14) * s21 * sh31 * y +
    (3 / 56) * s210 * sh32 * z +
    (3 / 56) * s210 * sh34 * x +
    (3 / 56) * s14 * sh36 * x;
  const sh43 =
    (-3 / 56) * s42 * sh31 * z + (3 / 28) * s105 * sh32 * y + (3 / 28) * s70 * sh33 * x + (3 / 56) * s42 * sh35 * x;
  const sh44 = (-3 / 28) * s42 * sh32 * x + (3 / 7) * s7 * sh33 * y - (3 / 28) * s42 * sh34 * z;
  const sh45 =
    (-3 / 56) * s42 * sh31 * x + (3 / 28) * s70 * sh33 * z + (3 / 28) * s105 * sh34 * y - (3 / 56) * s42 * sh35 * z;
  const sh46 =
    (-3 / 56) * s14 * sh30 * x -
    (3 / 56) * s210 * sh32 * x +
    (3 / 56) * s210 * sh34 * z +
    (3 / 14) * s21 * sh35 * y -
    (3 / 56) * s14 * sh36 * z;
  const sh47 = (-3 / 8) * s6 * sh31 * x + (3 / 8) * s6 * sh35 * z + (3 / 4) * sh36 * y;
  const sh48 = (3 / 4) * s2 * (-sh30 * x + sh36 * z);
  const vals = [
    1, sh10, sh11, sh12, sh20, sh21, sh22, sh23, sh24, sh30, sh31, sh32, sh33, sh34, sh35, sh36,
    sh40, sh41, sh42, sh43, sh44, sh45, sh46, sh47, sh48,
  ];
  const c = 1 / Math.sqrt(4 * Math.PI);
  for (let i = 0; i < 25; i++) out[i] = vals[i] * c;
  return out;
}

/** The e3nn Y^0 constant, 1/sqrt(4π): the whole (0,0) basis block of a fused ConvSE3. */
export const Y00 = 1 / Math.sqrt(4 * Math.PI);

/** Size of one edge's (d_in, d_out) basis block: (2d_in+1) × (2·min+1) × (2d_out+1). */
export function basisBlockSize(dIn, dOut) {
  return (2 * dIn + 1) * (2 * Math.min(dIn, dOut) + 1) * (2 * dOut + 1);
}

/**
 * TFN basis blocks for every (d_in, d_out) pair except (0, 0) (identity in pairwise convolutions):
 * basis[e][l][f][k] = Σ_m Y_{J_f}(r̂_e)[m] · Q_{J_f}[k][l][m], J_f = |d_in − d_out| + f
 * (se3_transformer.basis.get_basis_script). Packed per pair: `{ pairs: [{dIn,dOut,offset,size}], data }`
 * with each pair stored contiguously as E × size.
 */
export function computeBasis(relPos, maxDegree = MAX_DEGREE) {
  const e = relPos.length / 3;
  const pairs = [];
  let offset = 0;
  for (let dIn = 0; dIn <= maxDegree; dIn++) {
    for (let dOut = 0; dOut <= maxDegree; dOut++) {
      if (dIn === 0 && dOut === 0) continue;
      const size = basisBlockSize(dIn, dOut);
      pairs.push({ dIn, dOut, offset, size });
      offset += size * e;
    }
  }
  const data = new Float64Array(offset);
  const sh = new Float64Array(25);
  const qs = pairs.map(({ dIn, dOut }) => {
    const list = [];
    for (let J = Math.abs(dIn - dOut); J <= dIn + dOut; J++) list.push({ J, q: WIGNER_3J[`${J},${dIn},${dOut}`] });
    return list;
  });
  for (let k = 0; k < e; k++) {
    sphericalHarmonics(relPos[3 * k], relPos[3 * k + 1], relPos[3 * k + 2], sh);
    pairs.forEach(({ dIn, dOut, offset: po, size }, pi) => {
      const L = 2 * dIn + 1;
      const K = 2 * dOut + 1;
      const F = qs[pi].length;
      const base = po + k * size;
      for (let f = 0; f < F; f++) {
        const { J, q } = qs[pi][f];
        const M = 2 * J + 1;
        const shOff = J * J;
        for (let l = 0; l < L; l++) {
          for (let kk = 0; kk < K; kk++) {
            let acc = 0;
            for (let m = 0; m < M; m++) acc += sh[shOff + m] * q[(kk * L + l) * M + m];
            data[base + (l * F + f) * K + kk] = acc;
          }
        }
      }
    });
  }
  return { pairs, data };
}

/** Incoming-edge CSR (edges grouped by destination, original order within a node). */
export function incomingCsr(graph, n) {
  const e = graph.dst.length;
  const offsets = new Int32Array(n + 1);
  for (let k = 0; k < e; k++) offsets[graph.dst[k] + 1]++;
  for (let i = 0; i < n; i++) offsets[i + 1] += offsets[i];
  const fill = offsets.slice(0, n);
  const edges = new Int32Array(e);
  for (let k = 0; k < e; k++) edges[fill[graph.dst[k]]++] = k;
  return { offsets, edges };
}
