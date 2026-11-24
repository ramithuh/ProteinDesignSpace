// Pure-TS structural alignment & protein geometry utilities.
// Implements Kabsch superposition (exact proper rotation via 3x3 symmetric
// Jacobi eigendecomposition), sequence-order TM-score, PDB rigid transform,
// and fast Cα geometric quality telemetry (Rg, bond breaks, clashes, SS%).

export interface KabschResult {
  rmsd: number;
  tmScore: number;
  /** Row-major 3x3 rotation matrix R such that p_aligned = R * (p - srcMean) + dstMean. */
  rotation: Float64Array;
  /** Translation vector t such that p_aligned = R * p + t. */
  translation: Float64Array;
  /** Column-major 4x4 homogeneous transform matrix (Mol* Mat4 convention). */
  transform4x4: Float64Array;
  /** Aligned mobile coordinates [n * 3]. */
  aligned: Float32Array;
}

export interface GeometryStats {
  length: number;
  rg: number;
  meanCaDist: number;
  chainBreaks: number;
  clashes: number;
  helixPct: number;
  sheetPct: number;
  coilPct: number;
}

/** Multiply two 3x3 row-major matrices: C = A * B. */
function mat3Mul(A: Float64Array, B: Float64Array): Float64Array {
  const C = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      C[i * 3 + j] =
        A[i * 3 + 0] * B[0 * 3 + j] +
        A[i * 3 + 1] * B[1 * 3 + j] +
        A[i * 3 + 2] * B[2 * 3 + j];
    }
  }
  return C;
}

function mat3Transpose(A: Float64Array): Float64Array {
  return new Float64Array([
    A[0], A[3], A[6],
    A[1], A[4], A[7],
    A[2], A[5], A[8],
  ]);
}

function mat3Det(M: Float64Array): number {
  return (
    M[0] * (M[4] * M[8] - M[5] * M[7]) -
    M[1] * (M[3] * M[8] - M[5] * M[6]) +
    M[2] * (M[3] * M[7] - M[4] * M[6])
  );
}

/**
 * Exact 3x3 symmetric eigendecomposition via Jacobi rotations.
 * Returns eigenvalues sorted descending and corresponding orthonormal eigenvectors as columns of V.
 */
function jacobiSym3(A: Float64Array): { values: Float64Array; V: Float64Array } {
  const S = new Float64Array(A);
  const V = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

  for (let iter = 0; iter < 50; iter++) {
    // Pick largest off-diagonal element
    let p = 0;
    let q = 1;
    let maxVal = Math.abs(S[1]);
    if (Math.abs(S[2]) > maxVal) {
      p = 0; q = 2; maxVal = Math.abs(S[2]);
    }
    if (Math.abs(S[5]) > maxVal) {
      p = 1; q = 2; maxVal = Math.abs(S[5]);
    }
    if (maxVal < 1e-13) break;

    const app = S[p * 3 + p];
    const aqq = S[q * 3 + q];
    const apq = S[p * 3 + q];
    const tau = (aqq - app) / (2 * apq);
    const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    // Update S = J^T S J
    S[p * 3 + p] = app - t * apq;
    S[q * 3 + q] = aqq + t * apq;
    S[p * 3 + q] = 0;
    S[q * 3 + p] = 0;
    for (let r = 0; r < 3; r++) {
      if (r === p || r === q) continue;
      const arp = S[r * 3 + p];
      const arq = S[r * 3 + q];
      S[r * 3 + p] = S[p * 3 + r] = c * arp - s * arq;
      S[r * 3 + q] = S[q * 3 + r] = s * arp + c * arq;
    }
    // Update V = V J
    for (let r = 0; r < 3; r++) {
      const vrp = V[r * 3 + p];
      const vrq = V[r * 3 + q];
      V[r * 3 + p] = c * vrp - s * vrq;
      V[r * 3 + q] = s * vrp + c * vrq;
    }
  }

  // Sort descending by eigenvalue
  const order = [0, 1, 2].sort((i, j) => S[j * 3 + j] - S[i * 3 + i]);
  const values = new Float64Array([S[order[0] * 3 + order[0]], S[order[1] * 3 + order[1]], S[order[2] * 3 + order[2]]]);
  const Vsorted = new Float64Array(9);
  for (let col = 0; col < 3; col++) {
    const srcCol = order[col];
    for (let row = 0; row < 3; row++) {
      Vsorted[row * 3 + col] = V[row * 3 + srcCol];
    }
  }
  // Ensure right-handed basis (det V = +1)
  if (mat3Det(Vsorted) < 0) {
    Vsorted[2] = -Vsorted[2];
    Vsorted[5] = -Vsorted[5];
    Vsorted[8] = -Vsorted[8];
  }
  return { values, V: Vsorted };
}

/** Weighted Kabsch: finds proper rotation R (det=+1) and translation t minimizing sum w_i ||R * mobile_i + t - target_i||^2. */
function weightedKabsch(
  mobile: Float32Array,
  target: Float32Array,
  weights?: Float64Array,
): { R: Float64Array; t: Float64Array } {
  const n = Math.floor(Math.min(mobile.length, target.length) / 3);
  let wSum = 0;
  let mx = 0, my = 0, mz = 0;
  let tx = 0, ty = 0, tz = 0;
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1;
    wSum += w;
    mx += w * mobile[i * 3 + 0];
    my += w * mobile[i * 3 + 1];
    mz += w * mobile[i * 3 + 2];
    tx += w * target[i * 3 + 0];
    ty += w * target[i * 3 + 1];
    tz += w * target[i * 3 + 2];
  }
  if (wSum <= 0) {
    return {
      R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      t: new Float64Array([0, 0, 0]),
    };
  }
  mx /= wSum; my /= wSum; mz /= wSum;
  tx /= wSum; ty /= wSum; tz /= wSum;

  // Cross-covariance H = sum w_i (mobile_i - m) (target_i - t)^T  (3x3)
  // We want R = V * U^T where H = U * Sigma * V^T, i.e., maximize Tr(R * H).
  const H = new Float64Array(9);
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1;
    const px = mobile[i * 3 + 0] - mx;
    const py = mobile[i * 3 + 1] - my;
    const pz = mobile[i * 3 + 2] - mz;
    const qx = target[i * 3 + 0] - tx;
    const qy = target[i * 3 + 1] - ty;
    const qz = target[i * 3 + 2] - tz;
    H[0] += w * px * qx; H[1] += w * px * qy; H[2] += w * px * qz;
    H[3] += w * py * qx; H[4] += w * py * qy; H[5] += w * py * qz;
    H[6] += w * pz * qx; H[7] += w * pz * qy; H[8] += w * pz * qz;
  }

  // Compute V from eigendecomposition of H^T H
  const Ht = mat3Transpose(H);
  const HtH = mat3Mul(Ht, H);
  const { values, V } = jacobiSym3(HtH);

  // Compute U = H * V * Sigma^-1 (handling near-zero singular values via cross product)
  const HV = mat3Mul(H, V);
  const U = new Float64Array(9);
  const s0 = Math.sqrt(Math.max(0, values[0]));
  const s1 = Math.sqrt(Math.max(0, values[1]));

  if (s0 > 1e-10) {
    U[0] = HV[0] / s0; U[3] = HV[3] / s0; U[6] = HV[6] / s0;
    const l0 = Math.hypot(U[0], U[3], U[6]) || 1;
    U[0] /= l0; U[3] /= l0; U[6] /= l0;
  } else {
    U[0] = 1; U[3] = 0; U[6] = 0;
  }
  if (s1 > 1e-10) {
    let u1x = HV[1] / s1;
    let u1y = HV[4] / s1;
    let u1z = HV[7] / s1;
    // Gram-Schmidt orthogonalize against u0 for numerical stability
    const dot = u1x * U[0] + u1y * U[3] + u1z * U[6];
    u1x -= dot * U[0];
    u1y -= dot * U[3];
    u1z -= dot * U[6];
    const l1 = Math.hypot(u1x, u1y, u1z);
    if (l1 > 1e-10) {
      U[1] = u1x / l1; U[4] = u1y / l1; U[7] = u1z / l1;
    } else {
      const ux = U[0], uy = U[3], uz = U[6];
      let bx = 0, by = 1, bz = 0;
      if (Math.abs(uy) > 0.9) { bx = 1; by = 0; }
      const cx = uy * bz - uz * by;
      const cy = uz * bx - ux * bz;
      const cz = ux * by - uy * bx;
      const cl = Math.hypot(cx, cy, cz) || 1;
      U[1] = cx / cl; U[4] = cy / cl; U[7] = cz / cl;
    }
  } else {
    // Orthogonal vector to u0
    const ux = U[0], uy = U[3], uz = U[6];
    let bx = 0, by = 1, bz = 0;
    if (Math.abs(uy) > 0.9) { bx = 1; by = 0; }
    const cx = uy * bz - uz * by;
    const cy = uz * bx - ux * bz;
    const cz = ux * by - uy * bx;
    const cl = Math.hypot(cx, cy, cz) || 1;
    U[1] = cx / cl; U[4] = cy / cl; U[7] = cz / cl;
  }

  // Third column = +(u0 x u1) unconditionally so det(U) = +1 and det(R) = det(V * U^T) = +1 (proper rotation)
  const u0x = U[0], u0y = U[3], u0z = U[6];
  const u1x = U[1], u1y = U[4], u1z = U[7];
  U[2] = u0y * u1z - u0z * u1y;
  U[5] = u0z * u1x - u0x * u1z;
  U[8] = u0x * u1y - u0y * u1x;

  const R = mat3Mul(V, mat3Transpose(U));
  const t = new Float64Array([
    tx - (R[0] * mx + R[1] * my + R[2] * mz),
    ty - (R[3] * mx + R[4] * my + R[5] * mz),
    tz - (R[6] * mx + R[7] * my + R[8] * mz),
  ]);
  return { R, t };
}

/**
 * Kabsch-align `mobile` Cα coords [n*3] onto `target` Cα coords [n*3].
 * Also computes sequence-order TM-score with iterative weighting refinement.
 */
export function kabschAlign(mobile: Float32Array, target: Float32Array): KabschResult {
  const n = Math.floor(Math.min(mobile.length, target.length) / 3);
  if (n < 3) {
    return {
      rmsd: Number.NaN,
      tmScore: Number.NaN,
      rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      translation: new Float64Array([0, 0, 0]),
      transform4x4: new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      aligned: new Float32Array(mobile),
    };
  }

  const { R, t } = weightedKabsch(mobile, target);
  const aligned = new Float32Array(n * 3);
  let sqSum = 0;
  const d0 = n > 21 ? 1.24 * Math.cbrt(n - 15) - 1.8 : 0.5;
  const d0Sq = Math.max(0.25, d0 * d0);
  let tmSum = 0;

  for (let i = 0; i < n; i++) {
    const x = mobile[i * 3 + 0];
    const y = mobile[i * 3 + 1];
    const z = mobile[i * 3 + 2];
    const ax = R[0] * x + R[1] * y + R[2] * z + t[0];
    const ay = R[3] * x + R[4] * y + R[5] * z + t[1];
    const az = R[6] * x + R[7] * y + R[8] * z + t[2];
    aligned[i * 3 + 0] = ax;
    aligned[i * 3 + 1] = ay;
    aligned[i * 3 + 2] = az;
    const dx = ax - target[i * 3 + 0];
    const dy = ay - target[i * 3 + 1];
    const dz = az - target[i * 3 + 2];
    const d2 = dx * dx + dy * dy + dz * dz;
    sqSum += d2;
    tmSum += 1 / (1 + d2 / d0Sq);
  }
  const rmsd = Math.sqrt(sqSum / n);
  let bestTm = tmSum / n;

  // 3 iterations of TM-weighted refinement for an accurate scTM score
  let curR = R;
  let curT = t;
  const w = new Float64Array(n);
  for (let iter = 0; iter < 3; iter++) {
    for (let i = 0; i < n; i++) {
      const x = mobile[i * 3 + 0];
      const y = mobile[i * 3 + 1];
      const z = mobile[i * 3 + 2];
      const dx = curR[0] * x + curR[1] * y + curR[2] * z + curT[0] - target[i * 3 + 0];
      const dy = curR[3] * x + curR[4] * y + curR[5] * z + curT[1] - target[i * 3 + 1];
      const dz = curR[6] * x + curR[7] * y + curR[8] * z + curT[2] - target[i * 3 + 2];
      w[i] = 1 / (1 + (dx * dx + dy * dy + dz * dz) / d0Sq);
    }
    const next = weightedKabsch(mobile, target, w);
    curR = next.R;
    curT = next.t;
    let s = 0;
    for (let i = 0; i < n; i++) {
      const x = mobile[i * 3 + 0];
      const y = mobile[i * 3 + 1];
      const z = mobile[i * 3 + 2];
      const dx = curR[0] * x + curR[1] * y + curR[2] * z + curT[0] - target[i * 3 + 0];
      const dy = curR[3] * x + curR[4] * y + curR[5] * z + curT[1] - target[i * 3 + 1];
      const dz = curR[6] * x + curR[7] * y + curR[8] * z + curT[2] - target[i * 3 + 2];
      s += 1 / (1 + (dx * dx + dy * dy + dz * dz) / d0Sq);
    }
    if (s / n > bestTm) bestTm = s / n;
  }

  // Column-major 4x4 homogeneous transform (Mol* Mat4 layout)
  const transform4x4 = new Float64Array([
    R[0], R[3], R[6], 0,
    R[1], R[4], R[7], 0,
    R[2], R[5], R[8], 0,
    t[0], t[1], t[2], 1,
  ]);

  return { rmsd, tmScore: bestTm, rotation: R, translation: t, transform4x4, aligned };
}

export interface SubstructureKabschResult extends KabschResult {
  alignedResidues: number;
  isSubstructure: true;
}

export interface SubstructureAlignOptions {
  shortChainIds?: readonly string[] | null;
  longChainIds?: readonly string[] | null;
  /** Known 0-based indices in `longCoords` matching `shortCoords` (possibly permuted for unindexed motifs). */
  knownIndices?: readonly number[] | null;
  /** Native template Cα [x, y, z] coordinates corresponding 1-to-1 with `knownIndices` (resolves arbitrary unindexed permutations). */
  knownNativeCoords?: readonly (readonly number[])[] | null;
  /** Optional 1-letter amino acid sequences for `shortCoords` and `longCoords` to break geometric symmetries. */
  shortSeq?: string | null;
  longSeq?: string | null;
}

interface MotifIsland {
  start: number;
  len: number;
}

/**
 * Align a shorter structure (`shortCoords`, length m) onto the best-matching segment(s) of a
 * longer structure (`longCoords`, length n >= m) via rigid-body-consistent multi-island /
 * distance-matrix Kabsch alignment (or direct alignment when `options.knownIndices` is provided).
 * Supports single-island motifs (5TPN), multi-island/symmetric/permuted motifs (1PRW, 1BCF, 5YUI),
 * and sparse catalytic motifs (4XOJ, 1CTT, 1OS7) without collapsing discontinuous islands.
 */
export function kabschAlignSubstructure(
  shortCoords: Float32Array,
  longCoords: Float32Array,
  shortResNums?: readonly number[] | null,
  options?: SubstructureAlignOptions | null,
): SubstructureKabschResult {
  const m = Math.floor(shortCoords.length / 3);
  const n = Math.floor(longCoords.length / 3);
  if (m < 3 || n < m) {
    return { ...kabschAlign(shortCoords, longCoords), alignedResidues: Math.min(m, n), isSubstructure: true };
  }

  // 1. Direct alignment when known motif indices in longCoords are explicitly provided.
  // Note: Unindexed motif scaffolding (e.g. RFD4 `unindexedMotif: true`) places motif residues at
  // arbitrary scaffold indices whose ascending `knownIndices` order is a PERMUTATION of `shortCoords`.
  // Resolve the exact bijection via `knownNativeCoords` (and/or sequence-guided permutation search for
  // small sparse motifs) before running Kabsch so motifs & co-folded ligands superimpose accurately.
  const knownIndices = options?.knownIndices;
  if (knownIndices && knownIndices.length > 0) {
    const validKnown: number[] = [];
    const validNative: Array<readonly number[]> = [];
    const rawNative = options?.knownNativeCoords;
    const hasNative = Array.isArray(rawNative) && rawNative.length === knownIndices.length;
    for (let k = 0; k < knownIndices.length; k++) {
      const li = knownIndices[k];
      if (Number.isInteger(li) && li >= 0 && li < n) {
        validKnown.push(li);
        if (hasNative && Array.isArray(rawNative[k]) && rawNative[k].length === 3) {
          validNative.push(rawNative[k]);
        }
      }
    }
    const validNativeAligned = hasNative && validNative.length === validKnown.length;

    const evalKabschRmsd = (mob: Float32Array, tgt: Float32Array, count: number): number => {
      const { R, t } = weightedKabsch(mob, tgt);
      let sq = 0;
      for (let i = 0; i < count; i++) {
        const x = mob[i * 3 + 0];
        const y = mob[i * 3 + 1];
        const z = mob[i * 3 + 2];
        const dx = R[0] * x + R[1] * y + R[2] * z + t[0] - tgt[i * 3 + 0];
        const dy = R[3] * x + R[4] * y + R[5] * z + t[1] - tgt[i * 3 + 1];
        const dz = R[6] * x + R[7] * y + R[8] * z + t[2] - tgt[i * 3 + 2];
        sq += dx * dx + dy * dy + dz * dz;
      }
      return Math.sqrt(sq / count);
    };

    if (validKnown.length >= 3) {
      let orderedKnown: number[] | null = null;

      // 1a. Exact native crystal coordinate matching (handles arbitrary unindexed permutations of ANY size!)
      if (validNativeAligned && validKnown.length === m) {
        const usedK = new Uint8Array(m);
        const mapped = new Array<number>(m).fill(-1);
        let maxMatchedDist = 0;
        let allMatched = true;
        for (let i = 0; i < m; i++) {
          const sx = shortCoords[i * 3 + 0];
          const sy = shortCoords[i * 3 + 1];
          const sz = shortCoords[i * 3 + 2];
          let bestK = -1;
          let bestD2 = Infinity;
          for (let k = 0; k < m; k++) {
            if (usedK[k]) continue;
            const dx = sx - validNative[k][0];
            const dy = sy - validNative[k][1];
            const dz = sz - validNative[k][2];
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < bestD2) {
              bestD2 = d2;
              bestK = k;
            }
          }
          if (bestK < 0) {
            allMatched = false;
            break;
          }
          usedK[bestK] = 1;
          mapped[i] = validKnown[bestK];
          maxMatchedDist = Math.max(maxMatchedDist, Math.sqrt(bestD2));
        }
        // Same coordinate frame (e.g. direct crystal coords `crd`): exact < 0.5 Å bijection
        if (allMatched && maxMatchedDist < 0.5) {
          orderedKnown = mapped;
        }
      }

      // 1b. For small sparse motifs (m <= 7, e.g. catalytic triads/tetrads or centered `intCrd`),
      // find the globally optimal permutation of `validKnown` matching `shortCoords` (preferring
      // `validNative` rigid-body match when present, and amino-acid sequence identity when available).
      if (!orderedKnown && validKnown.length === m && m <= 7) {
        const sSeq = options?.shortSeq?.trim().toUpperCase() ?? '';
        const lSeq = options?.longSeq?.trim().toUpperCase() ?? '';
        const hasSeq = sSeq.length === m && lSeq.length === n;

        const targetRefCoords = new Float32Array(m * 3);
        const sourceCoordBuf = validNativeAligned
          ? (() => {
              const buf = new Float32Array(m * 3);
              for (let k = 0; k < m; k++) {
                buf[k * 3 + 0] = validNative[k][0];
                buf[k * 3 + 1] = validNative[k][1];
                buf[k * 3 + 2] = validNative[k][2];
              }
              return buf;
            })()
          : null;

        const perm = Array.from({ length: m }, (_, i) => i);
        const used = new Uint8Array(m);
        let bestPerm: number[] | null = null;
        let bestSeqMatches = -1;
        let bestRmsd = Infinity;

        const dfs = (depth: number, seqMatches: number) => {
          if (depth === m) {
            // If we already found a permutation with strictly more sequence matches (and good geometry),
            // prefer higher sequence identity first (avoiding mirror/symmetry confusion on 4-point motifs).
            if (hasSeq && seqMatches < bestSeqMatches) return;
            for (let i = 0; i < m; i++) {
              const k = perm[i];
              if (sourceCoordBuf) {
                targetRefCoords[i * 3 + 0] = sourceCoordBuf[k * 3 + 0];
                targetRefCoords[i * 3 + 1] = sourceCoordBuf[k * 3 + 1];
                targetRefCoords[i * 3 + 2] = sourceCoordBuf[k * 3 + 2];
              } else {
                const li = validKnown[k] * 3;
                targetRefCoords[i * 3 + 0] = longCoords[li + 0];
                targetRefCoords[i * 3 + 1] = longCoords[li + 1];
                targetRefCoords[i * 3 + 2] = longCoords[li + 2];
              }
            }
            const rmsd = evalKabschRmsd(shortCoords, targetRefCoords, m);
            const betterSeq = bestPerm === null || (hasSeq && seqMatches > bestSeqMatches);
            if (betterSeq || ((!hasSeq || seqMatches === bestSeqMatches) && rmsd < bestRmsd - 1e-6)) {
              bestSeqMatches = seqMatches;
              bestRmsd = rmsd;
              bestPerm = perm.slice();
            }
            return;
          }
          // Prune branches that cannot beat bestSeqMatches when sequence matching is active
          if (hasSeq && seqMatches + (m - depth) < bestSeqMatches) return;
          for (let k = 0; k < m; k++) {
            if (used[k]) continue;
            used[k] = 1;
            perm[depth] = k;
            const aaMatch =
              hasSeq && sSeq[depth] !== 'X' && lSeq[validKnown[k]] !== 'X' && sSeq[depth] === lSeq[validKnown[k]]
                ? 1
                : 0;
            dfs(depth + 1, seqMatches + aaMatch);
            used[k] = 0;
          }
        };

        dfs(0, 0);
        if (bestPerm) {
          orderedKnown = (bestPerm as number[]).map((k) => validKnown[k]);
        }
      }

      // 1c. For larger multi-island unindexed motifs (m > 7, e.g. 2-island 40-aa 1PRW EF-hand),
      // partition `shortCoords` and `validKnown` into contiguous motif islands and search over
      // island-level bijections (K <= 6, K! <= 720) using Kabsch RMSD (preferring `validNative`
      // when present, with sequence identity tie-breaking for symmetric motifs).
      if (!orderedKnown && validKnown.length === m && m > 7) {
        const sChainIds =
          options?.shortChainIds && options.shortChainIds.length === m ? options.shortChainIds : null;
        const lChainIds =
          options?.longChainIds && options.longChainIds.length === n ? options.longChainIds : null;
        const sSeq = options?.shortSeq?.trim().toUpperCase() ?? '';
        const lSeq = options?.longSeq?.trim().toUpperCase() ?? '';
        const hasSeq = sSeq.length === m && lSeq.length === n;

        const shortIslands: MotifIsland[] = [];
        let sStart = 0;
        for (let i = 1; i < m; i++) {
          const chainGap = sChainIds ? sChainIds[i] !== sChainIds[i - 1] : false;
          const resGap =
            shortResNums && shortResNums.length === m ? shortResNums[i] !== shortResNums[i - 1] + 1 : false;
          const dx = shortCoords[i * 3] - shortCoords[(i - 1) * 3];
          const dy = shortCoords[i * 3 + 1] - shortCoords[(i - 1) * 3 + 1];
          const dz = shortCoords[i * 3 + 2] - shortCoords[(i - 1) * 3 + 2];
          const distBreak = dx * dx + dy * dy + dz * dz > 6.5 * 6.5;
          if (chainGap || resGap || distBreak) {
            shortIslands.push({ start: sStart, len: i - sStart });
            sStart = i;
          }
        }
        shortIslands.push({ start: sStart, len: m - sStart });

        const knownIslands: MotifIsland[] = [];
        let kStart = 0;
        for (let k = 1; k < m; k++) {
          const idxGap = validKnown[k] !== validKnown[k - 1] + 1;
          const chainGap = lChainIds ? lChainIds[validKnown[k]] !== lChainIds[validKnown[k - 1]] : false;
          let distBreak = false;
          if (validNativeAligned) {
            const dx = validNative[k][0] - validNative[k - 1][0];
            const dy = validNative[k][1] - validNative[k - 1][1];
            const dz = validNative[k][2] - validNative[k - 1][2];
            distBreak = dx * dx + dy * dy + dz * dz > 6.5 * 6.5;
          } else {
            const a = validKnown[k] * 3;
            const b = validKnown[k - 1] * 3;
            const dx = longCoords[a] - longCoords[b];
            const dy = longCoords[a + 1] - longCoords[b + 1];
            const dz = longCoords[a + 2] - longCoords[b + 2];
            distBreak = dx * dx + dy * dy + dz * dz > 6.5 * 6.5;
          }
          if (idxGap || chainGap || distBreak) {
            knownIslands.push({ start: kStart, len: k - kStart });
            kStart = k;
          }
        }
        knownIslands.push({ start: kStart, len: m - kStart });

        const K = shortIslands.length;
        if (K >= 2 && K <= 6) {
          const targetRefCoords = new Float32Array(m * 3);
          const candKnownBuf = new Array<number>(m);
          let bestCandKnown: number[] | null = null;
          let bestRmsd = Infinity;
          let bestSeqMatches = -1;

          const evaluateCandidateMapping = (kIdxForShort: readonly number[]) => {
            let seqMatches = 0;
            for (let i = 0; i < m; i++) {
              const k = kIdxForShort[i];
              const li = validKnown[k];
              candKnownBuf[i] = li;
              if (validNativeAligned) {
                targetRefCoords[i * 3 + 0] = validNative[k][0];
                targetRefCoords[i * 3 + 1] = validNative[k][1];
                targetRefCoords[i * 3 + 2] = validNative[k][2];
              } else {
                targetRefCoords[i * 3 + 0] = longCoords[li * 3 + 0];
                targetRefCoords[i * 3 + 1] = longCoords[li * 3 + 1];
                targetRefCoords[i * 3 + 2] = longCoords[li * 3 + 2];
              }
              if (hasSeq && sSeq[i] !== 'X' && lSeq[li] !== 'X' && sSeq[i] === lSeq[li]) {
                seqMatches++;
              }
            }
            const rmsd = evalKabschRmsd(shortCoords, targetRefCoords, m);
            if (!Number.isFinite(rmsd)) return;
            const clearlyBetterGeom = rmsd < bestRmsd - 0.25;
            const tiedGeomBetterSeq =
              hasSeq && Math.abs(rmsd - bestRmsd) <= 0.25 && seqMatches > bestSeqMatches;
            const exactBetterGeom =
              (!hasSeq || seqMatches === bestSeqMatches) && rmsd < bestRmsd - 1e-6;
            if (bestCandKnown === null || clearlyBetterGeom || tiedGeomBetterSeq || exactBetterGeom) {
              bestRmsd = rmsd;
              bestSeqMatches = seqMatches;
              bestCandKnown = candKnownBuf.slice();
            }
          };

          const kIdxMap = new Array<number>(m);
          const perm = new Array<number>(K);
          const used = new Uint8Array(K);
          let testedExactIslands = false;

          if (knownIslands.length === K) {
            const dfsExact = (depth: number) => {
              if (depth === K) {
                testedExactIslands = true;
                for (let si = 0; si < K; si++) {
                  const sIsl = shortIslands[si];
                  const kIsl = knownIslands[perm[si]];
                  for (let j = 0; j < sIsl.len; j++) {
                    kIdxMap[sIsl.start + j] = kIsl.start + j;
                  }
                }
                evaluateCandidateMapping(kIdxMap);
                return;
              }
              const needLen = shortIslands[depth].len;
              for (let ki = 0; ki < K; ki++) {
                if (used[ki] || knownIslands[ki].len !== needLen) continue;
                used[ki] = 1;
                perm[depth] = ki;
                dfsExact(depth + 1);
                used[ki] = 0;
              }
            };
            dfsExact(0);
          }

          // Fallback when islands were placed adjacent (0 spacer) so knownIslands merged:
          // partition validKnown sequentially by every permutation of shortIslands lengths.
          if (!testedExactIslands) {
            used.fill(0);
            const dfsOrder = (depth: number) => {
              if (depth === K) {
                let cursor = 0;
                for (let p = 0; p < K; p++) {
                  const sIsl = shortIslands[perm[p]];
                  for (let j = 0; j < sIsl.len; j++) {
                    kIdxMap[sIsl.start + j] = cursor++;
                  }
                }
                evaluateCandidateMapping(kIdxMap);
                return;
              }
              for (let si = 0; si < K; si++) {
                if (used[si]) continue;
                used[si] = 1;
                perm[depth] = si;
                dfsOrder(depth + 1);
                used[si] = 0;
              }
            };
            dfsOrder(0);
          }

          if (bestCandKnown) {
            orderedKnown = bestCandKnown;
          }
        }
      }

      const finalKnown = orderedKnown ?? validKnown;
      const pairCount = Math.min(m, finalKnown.length);
      if (pairCount >= 3) {
        const subMob = new Float32Array(pairCount * 3);
        const subTgt = new Float32Array(pairCount * 3);
        for (let i = 0; i < pairCount; i++) {
          const si = i * 3;
          const li = finalKnown[i] * 3;
          subMob[i * 3 + 0] = shortCoords[si + 0];
          subMob[i * 3 + 1] = shortCoords[si + 1];
          subMob[i * 3 + 2] = shortCoords[si + 2];
          subTgt[i * 3 + 0] = longCoords[li + 0];
          subTgt[i * 3 + 1] = longCoords[li + 1];
          subTgt[i * 3 + 2] = longCoords[li + 2];
        }
        const joint = kabschAlign(subMob, subTgt);
        return { ...joint, alignedResidues: pairCount, isSubstructure: true };
      }
    }
  }

  if (m === n) {
    return { ...kabschAlign(shortCoords, longCoords), alignedResidues: m, isSubstructure: true };
  }

  const shortChainIds =
    options?.shortChainIds && options.shortChainIds.length === m ? options.shortChainIds : null;
  const longChainIds =
    options?.longChainIds && options.longChainIds.length === n ? options.longChainIds : null;

  // 2. Partition shortCoords into contiguous islands (splitting on chain change, resNum gap, or Cα > 6.5 Å)
  const islands: MotifIsland[] = [];
  let islStart = 0;
  for (let i = 1; i < m; i++) {
    const chainGap = shortChainIds ? shortChainIds[i] !== shortChainIds[i - 1] : false;
    const resGap =
      shortResNums && shortResNums.length === m ? shortResNums[i] !== shortResNums[i - 1] + 1 : false;
    const dx = shortCoords[i * 3] - shortCoords[(i - 1) * 3];
    const dy = shortCoords[i * 3 + 1] - shortCoords[(i - 1) * 3 + 1];
    const dz = shortCoords[i * 3 + 2] - shortCoords[(i - 1) * 3 + 2];
    const distBreak = dx * dx + dy * dy + dz * dz > 6.5 * 6.5;
    if (chainGap || resGap || distBreak) {
      islands.push({ start: islStart, len: i - islStart });
      islStart = i;
    }
  }
  islands.push({ start: islStart, len: m - islStart });

  const sliceCoords = (coords: Float32Array, start: number, len: number): Float32Array =>
    coords.subarray(start * 3, (start + len) * 3);

  // Check whether a contiguous window [off .. off + len - 1] in longCoords stays on one chain
  const isSameChainLongWindow = (off: number, len: number): boolean => {
    if (!longChainIds || len <= 1) return true;
    const ch = longChainIds[off];
    for (let k = 1; k < len; k++) {
      if (longChainIds[off + k] !== ch) return false;
    }
    return true;
  };

  const evaluateWindowRmsd = (isl: MotifIsland, off: number): { rmsd: number; R: Float64Array; t: Float64Array } => {
    const islMob = sliceCoords(shortCoords, isl.start, isl.len);
    const winTgt = sliceCoords(longCoords, off, isl.len);
    const { R, t } = weightedKabsch(islMob, winTgt);
    let sq = 0;
    for (let i = 0; i < isl.len; i++) {
      const x = islMob[i * 3], y = islMob[i * 3 + 1], z = islMob[i * 3 + 2];
      const ax = R[0] * x + R[1] * y + R[2] * z + t[0] - winTgt[i * 3];
      const ay = R[3] * x + R[4] * y + R[5] * z + t[1] - winTgt[i * 3 + 1];
      const az = R[6] * x + R[7] * y + R[8] * z + t[2] - winTgt[i * 3 + 2];
      sq += ax * ax + ay * ay + az * az;
    }
    return { rmsd: Math.sqrt(sq / isl.len), R, t };
  };

  // 3. Single contiguous island: exact sliding-window Kabsch
  if (islands.length === 1) {
    const isl = islands[0];
    let bestOffset = -1;
    let bestRmsd = Infinity;
    for (let pass = 0; pass < 2; pass++) {
      const enforceChain = pass === 0 && longChainIds !== null;
      for (let off = 0; off <= n - isl.len; off++) {
        if (enforceChain && !isSameChainLongWindow(off, isl.len)) continue;
        const { rmsd } = evaluateWindowRmsd(isl, off);
        if (rmsd < bestRmsd) {
          bestRmsd = rmsd;
          bestOffset = off;
        }
      }
      if (bestOffset >= 0) break;
    }
    const winTgt = sliceCoords(longCoords, Math.max(0, bestOffset), isl.len);
    const joint = kabschAlign(shortCoords, winTgt);
    return { ...joint, alignedResidues: isl.len, isSubstructure: true };
  }

  // 4. Multi-island or sparse motif: NEVER collapse discontinuous islands into one contiguous window.
  // Rigid-body consensus assignment: given a candidate transform (R, t), project all shortCoords
  // into longCoords space, assign every island (longest first; single-residue islands by min-distance
  // greedy assignment) to non-overlapping windows, and perform 1 iterative joint Kabsch refinement.
  const predBuf = new Float32Array(m * 3);
  const usedBuf = new Uint8Array(n);
  const islandsByLenDesc = islands
    .map((isl, idx) => ({ isl, idx }))
    .sort((a, b) => b.isl.len - a.isl.len || a.idx - b.idx);
  const multiLenIslands = islandsByLenDesc.filter((x) => x.isl.len >= 2);
  const singleLenIslands = islandsByLenDesc.filter((x) => x.isl.len === 1);

  const assignWithRigidTransform = (
    R0: Float64Array,
    t0: Float64Array,
  ): { rmsd: number; shortIdx: Int32Array; longIdx: Int32Array; count: number } | null => {
    let curR = R0;
    let curT = t0;
    let bestShortIdx = new Int32Array(m);
    let bestLongIdx = new Int32Array(m);
    let bestCount = 0;
    let bestRmsd = Infinity;

    for (let iter = 0; iter < 2; iter++) {
      // Transform all shortCoords by (curR, curT)
      for (let i = 0; i < m; i++) {
        const x = shortCoords[i * 3 + 0];
        const y = shortCoords[i * 3 + 1];
        const z = shortCoords[i * 3 + 2];
        predBuf[i * 3 + 0] = curR[0] * x + curR[1] * y + curR[2] * z + curT[0];
        predBuf[i * 3 + 1] = curR[3] * x + curR[4] * y + curR[5] * z + curT[1];
        predBuf[i * 3 + 2] = curR[6] * x + curR[7] * y + curR[8] * z + curT[2];
      }

      usedBuf.fill(0);
      const assignedOff = new Int32Array(islands.length).fill(-1);

      // Assign multi-residue islands first (longest first)
      for (const { isl, idx } of multiLenIslands) {
        let winBest = -1;
        let winSse = Infinity;
        for (let pass = 0; pass < 2; pass++) {
          const enforceChain = pass === 0 && longChainIds !== null;
          for (let off = 0; off <= n - isl.len; off++) {
            if (enforceChain && !isSameChainLongWindow(off, isl.len)) continue;
            let overlap = false;
            for (let k = 0; k < isl.len; k++) {
              if (usedBuf[off + k]) {
                overlap = true;
                break;
              }
            }
            if (overlap) continue;
            let sse = 0;
            for (let k = 0; k < isl.len; k++) {
              const pi = (isl.start + k) * 3;
              const li = (off + k) * 3;
              const dx = predBuf[pi + 0] - longCoords[li + 0];
              const dy = predBuf[pi + 1] - longCoords[li + 1];
              const dz = predBuf[pi + 2] - longCoords[li + 2];
              sse += dx * dx + dy * dy + dz * dz;
              if (sse >= winSse) break;
            }
            if (sse < winSse) {
              winSse = sse;
              winBest = off;
            }
          }
          if (winBest >= 0) break;
        }
        if (winBest >= 0) {
          assignedOff[idx] = winBest;
          for (let k = 0; k < isl.len; k++) usedBuf[winBest + k] = 1;
        }
      }

      // Assign single-residue islands by greedy minimum-distance-first so closer residues aren't stolen
      if (singleLenIslands.length > 0) {
        const remaining = singleLenIslands.slice();
        while (remaining.length > 0) {
          let bestRemPos = -1;
          let bestLong = -1;
          let bestD2 = Infinity;
          for (let r = 0; r < remaining.length; r++) {
            const si = remaining[r].isl.start * 3;
            const px = predBuf[si + 0];
            const py = predBuf[si + 1];
            const pz = predBuf[si + 2];
            for (let j = 0; j < n; j++) {
              if (usedBuf[j]) continue;
              const dx = px - longCoords[j * 3 + 0];
              const dy = py - longCoords[j * 3 + 1];
              const dz = pz - longCoords[j * 3 + 2];
              const d2 = dx * dx + dy * dy + dz * dz;
              if (d2 < bestD2) {
                bestD2 = d2;
                bestRemPos = r;
                bestLong = j;
              }
            }
          }
          if (bestRemPos < 0 || bestLong < 0) break;
          const chosen = remaining.splice(bestRemPos, 1)[0];
          assignedOff[chosen.idx] = bestLong;
          usedBuf[bestLong] = 1;
        }
      }

      // Build matched pairs in original island order
      const sIdx = new Int32Array(m);
      const lIdx = new Int32Array(m);
      let count = 0;
      for (let idx = 0; idx < islands.length; idx++) {
        const off = assignedOff[idx];
        if (off < 0) continue;
        const isl = islands[idx];
        for (let k = 0; k < isl.len; k++) {
          sIdx[count] = isl.start + k;
          lIdx[count] = off + k;
          count++;
        }
      }
      if (count < 3) return null;

      const subMob = new Float32Array(count * 3);
      const subTgt = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        const si = sIdx[i] * 3;
        const li = lIdx[i] * 3;
        subMob[i * 3 + 0] = shortCoords[si + 0];
        subMob[i * 3 + 1] = shortCoords[si + 1];
        subMob[i * 3 + 2] = shortCoords[si + 2];
        subTgt[i * 3 + 0] = longCoords[li + 0];
        subTgt[i * 3 + 1] = longCoords[li + 1];
        subTgt[i * 3 + 2] = longCoords[li + 2];
      }
      const fit = weightedKabsch(subMob, subTgt);
      let sq = 0;
      for (let i = 0; i < count; i++) {
        const x = subMob[i * 3 + 0];
        const y = subMob[i * 3 + 1];
        const z = subMob[i * 3 + 2];
        const dx = fit.R[0] * x + fit.R[1] * y + fit.R[2] * z + fit.t[0] - subTgt[i * 3 + 0];
        const dy = fit.R[3] * x + fit.R[4] * y + fit.R[5] * z + fit.t[1] - subTgt[i * 3 + 1];
        const dz = fit.R[6] * x + fit.R[7] * y + fit.R[8] * z + fit.t[2] - subTgt[i * 3 + 2];
        sq += dx * dx + dy * dy + dz * dz;
      }
      const rmsd = Math.sqrt(sq / count);
      if (count > bestCount || (count === bestCount && rmsd < bestRmsd)) {
        bestRmsd = rmsd;
        bestShortIdx = sIdx;
        bestLongIdx = lIdx;
        bestCount = count;
      }
      curR = fit.R;
      curT = fit.t;
    }

    return bestCount >= 3
      ? { rmsd: bestRmsd, shortIdx: bestShortIdx, longIdx: bestLongIdx, count: bestCount }
      : null;
  };

  let bestMatch: { rmsd: number; shortIdx: Int32Array; longIdx: Int32Array; count: number } | null = null;
  const considerMatch = (cand: { rmsd: number; shortIdx: Int32Array; longIdx: Int32Array; count: number } | null) => {
    if (!cand) return;
    if (
      !bestMatch ||
      cand.count > bestMatch.count ||
      (cand.count === bestMatch.count && cand.rmsd < bestMatch.rmsd)
    ) {
      bestMatch = cand;
    }
  };

  const maxIslandLen = islandsByLenDesc[0]?.isl.len ?? 0;

  // Regime A: Seed from up to the 2 longest islands with len >= 3 (handles 1PRW, 1BCF, 5YUI, permuted motifs, mixed 1,1,3)
  const seedIslands = islandsByLenDesc.filter((x) => x.isl.len >= 3).slice(0, 2);
  for (const { isl } of seedIslands) {
    const candidates: Array<{ rmsd: number; R: Float64Array; t: Float64Array }> = [];
    for (let pass = 0; pass < 2; pass++) {
      const enforceChain = pass === 0 && longChainIds !== null;
      for (let off = 0; off <= n - isl.len; off++) {
        if (enforceChain && !isSameChainLongWindow(off, isl.len)) continue;
        candidates.push(evaluateWindowRmsd(isl, off));
      }
      if (candidates.length > 0) break;
    }
    candidates.sort((a, b) => a.rmsd - b.rmsd);
    const topK = Math.min(24, candidates.length);
    for (let k = 0; k < topK; k++) {
      considerMatch(assignWithRigidTransform(candidates[k].R, candidates[k].t));
    }
  }

  // Regime B: Distance-matrix anchor-triple rigid-body consensus (handles sparse catalytic motifs
  // like 4XOJ, 1CTT, 1OS7, Trypsin triad, and refines short-island motifs where maxIslandLen <= 4)
  if (maxIslandLen < 3 || (maxIslandLen <= 4 && m <= 32) || !bestMatch) {
    // Pick maximally non-collinear anchor triple (a, b, c) in shortCoords
    let anchorA = 0;
    let anchorB = 1;
    let maxDab = -1;
    for (let i = 0; i < m; i++) {
      for (let j = i + 1; j < m; j++) {
        const d = dist(shortCoords, i, j);
        if (d > maxDab) {
          maxDab = d;
          anchorA = i;
          anchorB = j;
        }
      }
    }
    const abx = shortCoords[anchorB * 3 + 0] - shortCoords[anchorA * 3 + 0];
    const aby = shortCoords[anchorB * 3 + 1] - shortCoords[anchorA * 3 + 1];
    const abz = shortCoords[anchorB * 3 + 2] - shortCoords[anchorA * 3 + 2];
    let anchorC = -1;
    let bestArea2 = -1;
    for (let i = 0; i < m; i++) {
      if (i === anchorA || i === anchorB) continue;
      const acx = shortCoords[i * 3 + 0] - shortCoords[anchorA * 3 + 0];
      const acy = shortCoords[i * 3 + 1] - shortCoords[anchorA * 3 + 1];
      const acz = shortCoords[i * 3 + 2] - shortCoords[anchorA * 3 + 2];
      const cx = aby * acz - abz * acy;
      const cy = abz * acx - abx * acz;
      const cz = abx * acy - aby * acx;
      const area2 = cx * cx + cy * cy + cz * cz;
      if (area2 > bestArea2) {
        bestArea2 = area2;
        anchorC = i;
      }
    }
    if (anchorC < 0) {
      anchorC = anchorA !== 0 && anchorB !== 0 ? 0 : anchorA !== 1 && anchorB !== 1 ? 1 : 2;
    }

    const dab = dist(shortCoords, anchorA, anchorB);
    const dac = dist(shortCoords, anchorA, anchorC);
    const dbc = dist(shortCoords, anchorB, anchorC);

    // Precompute pairwise distance matrix for longCoords (bounded for very large complexes)
    const dLong = n <= 1600 ? new Float32Array(n * n) : null;
    if (dLong) {
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const d = dist(longCoords, i, j);
          dLong[i * n + j] = d;
          dLong[j * n + i] = d;
        }
      }
    }
    const getLongDist = (i: number, j: number): number =>
      dLong ? dLong[i * n + j] : dist(longCoords, i, j);

    const anchorMob = new Float32Array([
      shortCoords[anchorA * 3 + 0], shortCoords[anchorA * 3 + 1], shortCoords[anchorA * 3 + 2],
      shortCoords[anchorB * 3 + 0], shortCoords[anchorB * 3 + 1], shortCoords[anchorB * 3 + 2],
      shortCoords[anchorC * 3 + 0], shortCoords[anchorC * 3 + 1], shortCoords[anchorC * 3 + 2],
    ]);
    const anchorTgt = new Float32Array(9);

    const tolerances = [1.5, 3.0, 6.0];
    for (const tol of tolerances) {
      const topTriples: Array<{ u: number; v: number; w: number; err: number }> = [];
      let checks = 0;
      const MAX_CHECKS = 150000;

      for (let u = 0; u < n && checks < MAX_CHECKS; u++) {
        const vList: number[] = [];
        const wList: number[] = [];
        for (let j = 0; j < n; j++) {
          if (j === u) continue;
          const d = getLongDist(u, j);
          if (Math.abs(d - dab) <= tol) vList.push(j);
          if (Math.abs(d - dac) <= tol) wList.push(j);
        }
        for (let vi = 0; vi < vList.length && checks < MAX_CHECKS; vi++) {
          const v = vList[vi];
          const ev = Math.abs(getLongDist(u, v) - dab);
          for (let wi = 0; wi < wList.length; wi++) {
            const w = wList[wi];
            if (w === v) continue;
            checks++;
            const ew2 = Math.abs(getLongDist(v, w) - dbc);
            if (ew2 <= tol) {
              const err = ev + Math.abs(getLongDist(u, w) - dac) + ew2;
              topTriples.push({ u, v, w, err });
            }
          }
        }
      }

      if (topTriples.length > 0) {
        topTriples.sort((x, y) => x.err - y.err);
        const evalLimit = Math.min(48, topTriples.length);
        for (let k = 0; k < evalLimit; k++) {
          const { u, v, w } = topTriples[k];
          anchorTgt[0] = longCoords[u * 3 + 0]; anchorTgt[1] = longCoords[u * 3 + 1]; anchorTgt[2] = longCoords[u * 3 + 2];
          anchorTgt[3] = longCoords[v * 3 + 0]; anchorTgt[4] = longCoords[v * 3 + 1]; anchorTgt[5] = longCoords[v * 3 + 2];
          anchorTgt[6] = longCoords[w * 3 + 0]; anchorTgt[7] = longCoords[w * 3 + 1]; anchorTgt[8] = longCoords[w * 3 + 2];
          const { R, t } = weightedKabsch(anchorMob, anchorTgt);
          considerMatch(assignWithRigidTransform(R, t));
        }
        if (bestMatch && (bestMatch as { rmsd: number }).rmsd < tol) break;
      }
    }
  }

  // Final fallback if completely unrelated geometry produced 0 triangle matches
  if (!bestMatch) {
    const idR = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const zeroT = new Float64Array([0, 0, 0]);
    considerMatch(assignWithRigidTransform(idR, zeroT));
  }

  if (!bestMatch) {
    return { ...kabschAlign(shortCoords, longCoords.subarray(0, m * 3)), alignedResidues: m, isSubstructure: true };
  }

  const finalMatch = bestMatch as { rmsd: number; shortIdx: Int32Array; longIdx: Int32Array; count: number };
  const pairCount = finalMatch.count;
  const subMob = new Float32Array(pairCount * 3);
  const subTgt = new Float32Array(pairCount * 3);
  for (let i = 0; i < pairCount; i++) {
    const si = finalMatch.shortIdx[i] * 3;
    const li = finalMatch.longIdx[i] * 3;
    subMob[i * 3 + 0] = shortCoords[si + 0];
    subMob[i * 3 + 1] = shortCoords[si + 1];
    subMob[i * 3 + 2] = shortCoords[si + 2];
    subTgt[i * 3 + 0] = longCoords[li + 0];
    subTgt[i * 3 + 1] = longCoords[li + 1];
    subTgt[i * 3 + 2] = longCoords[li + 2];
  }

  const joint = kabschAlign(subMob, subTgt);
  return { ...joint, alignedResidues: pairCount, isSubstructure: true };
}

function padLeft(s: string, width: number): string {
  return s.length >= width ? s : ' '.repeat(width - s.length) + s;
}

/**
 * Format a coordinate strictly within PDB's 8-character fixed column width.
 * Uses %8.3f for normal range [-999.999, 9999.999], gracefully reduces precision
 * for moderate overflows so columns 55+ (occupancy/B-factor/element) never shift,
 * and clamps extreme values as a last resort.
 */
function formatPdbCoord8(v: number): string {
  if (!Number.isFinite(v)) return '   0.000';
  const s3 = v.toFixed(3);
  if (s3.length <= 8) return padLeft(s3, 8);
  const s2 = v.toFixed(2);
  if (s2.length <= 8) return padLeft(s2, 8);
  const s1 = v.toFixed(1);
  if (s1.length <= 8) return padLeft(s1, 8);
  const s0 = Math.round(v).toString();
  if (s0.length <= 8) return padLeft(s0, 8);
  const clamped = Math.max(-9999999, Math.min(99999999, Math.round(v)));
  return padLeft(String(clamped), 8);
}

/** Apply rigid rotation R (row-major 3x3) + translation t to every ATOM/HETATM line of a PDB. */
export function applyTransformToPdb(
  pdbText: string,
  R: Float64Array,
  t: Float64Array,
): string {
  const lines = pdbText.split('\n');
  const transformed: Array<{ lineIdx: number; nx: number; ny: number; nz: number }> = [];
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  let extremeOverflow = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if ((line.startsWith('ATOM  ') || line.startsWith('HETATM')) && line.length >= 54) {
      const x = parseFloat(line.slice(30, 38));
      const y = parseFloat(line.slice(38, 46));
      const z = parseFloat(line.slice(46, 54));
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        const nx = R[0] * x + R[1] * y + R[2] * z + t[0];
        const ny = R[3] * x + R[4] * y + R[5] * z + t[1];
        const nz = R[6] * x + R[7] * y + R[8] * z + t[2];
        transformed.push({ lineIdx: i, nx, ny, nz });
        sumX += nx;
        sumY += ny;
        sumZ += nz;
        if (
          nx < -9999999 || nx > 99999999 ||
          ny < -9999999 || ny > 99999999 ||
          nz < -9999999 || nz > 99999999
        ) {
          extremeOverflow = true;
        }
      }
    }
  }

  // Re-center only if coordinates exceed even integer 8-char PDB bounds (preserving relative geometry)
  const offX = extremeOverflow && transformed.length > 0 ? sumX / transformed.length : 0;
  const offY = extremeOverflow && transformed.length > 0 ? sumY / transformed.length : 0;
  const offZ = extremeOverflow && transformed.length > 0 ? sumZ / transformed.length : 0;

  const out = lines.slice();
  for (const item of transformed) {
    const line = lines[item.lineIdx];
    out[item.lineIdx] =
      line.slice(0, 30) +
      formatPdbCoord8(item.nx - offX) +
      formatPdbCoord8(item.ny - offY) +
      formatPdbCoord8(item.nz - offZ) +
      line.slice(54);
  }
  return out.join('\n');
}

/** Fast Cα-only geometric quality & secondary structure estimator (P-SEA-inspired). */
export function computeGeometryStats(
  caCoords: Float32Array,
  chainIds?: readonly string[],
  resNums?: readonly number[],
  insCodes?: readonly string[],
): GeometryStats {
  const L = Math.floor(caCoords.length / 3);
  if (L === 0) {
    return { length: 0, rg: 0, meanCaDist: 0, chainBreaks: 0, clashes: 0, helixPct: 0, sheetPct: 0, coilPct: 100 };
  }
  const hasChains = Boolean(chainIds && chainIds.length === L);
  const hasResNums = Boolean(resNums && resNums.length === L);
  const hasInsCodes = Boolean(insCodes && insCodes.length === L);
  const sameChain = (i: number, j: number): boolean => !hasChains || chainIds![i] === chainIds![j];
  const isBonded = (i: number): boolean => {
    if (!sameChain(i, i + 1)) return false;
    if (!hasResNums) return true;
    const dr = resNums![i + 1] - resNums![i];
    if (dr === 1) return true;
    // Insertion-code neighbors (e.g. 100 -> 100A -> 100B): treat as bonded when
    // insCodes differ or Cα-Cα distance is within peptide bond range (<= 4.5 Å).
    if (dr === 0) {
      if (hasInsCodes && insCodes![i + 1] !== insCodes![i]) return true;
      const d = dist(caCoords, i, i + 1);
      if (Number.isFinite(d) && d <= 4.5) return true;
    }
    return false;
  };

  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < L; i++) {
    cx += caCoords[i * 3 + 0];
    cy += caCoords[i * 3 + 1];
    cz += caCoords[i * 3 + 2];
  }
  cx /= L; cy /= L; cz /= L;

  let rgSq = 0;
  for (let i = 0; i < L; i++) {
    const dx = caCoords[i * 3 + 0] - cx;
    const dy = caCoords[i * 3 + 1] - cy;
    const dz = caCoords[i * 3 + 2] - cz;
    rgSq += dx * dx + dy * dy + dz * dz;
  }
  const rg = Math.sqrt(rgSq / L);

  let distSum = 0;
  let bondedPairs = 0;
  let chainBreaks = 0;
  for (let i = 0; i + 1 < L; i++) {
    if (!isBonded(i)) continue;
    const d = Math.hypot(
      caCoords[(i + 1) * 3 + 0] - caCoords[i * 3 + 0],
      caCoords[(i + 1) * 3 + 1] - caCoords[i * 3 + 1],
      caCoords[(i + 1) * 3 + 2] - caCoords[i * 3 + 2],
    );
    distSum += d;
    bondedPairs++;
    if (d > 4.5 || d < 2.8) chainBreaks++;
  }
  const meanCaDist = bondedPairs > 0 ? distSum / bondedPairs : 0;

  // Sweep in x so pairs separated by at least the clash radius need no distance test.
  // Nonfinite coordinates cannot contribute a finite distance and must not disrupt sorting.
  const byX: number[] = [];
  for (let i = 0; i < L; i++) {
    if (
      Number.isFinite(caCoords[i * 3]) &&
      Number.isFinite(caCoords[i * 3 + 1]) &&
      Number.isFinite(caCoords[i * 3 + 2])
    ) {
      byX.push(i);
    }
  }
  byX.sort((a, b) => caCoords[a * 3] - caCoords[b * 3]);
  let clashes = 0;
  for (let a = 0; a < byX.length; a++) {
    const i = byX[a];
    for (let b = a + 1; b < byX.length; b++) {
      const j = byX[b];
      const dx = caCoords[j * 3] - caCoords[i * 3];
      if (dx >= 3) break;
      const dy = caCoords[j * 3 + 1] - caCoords[i * 3 + 1];
      const dz = caCoords[j * 3 + 2] - caCoords[i * 3 + 2];
      if (Math.abs(dy) >= 3 || Math.abs(dz) >= 3) continue;
      if (Math.abs(i - j) === 1 && isBonded(Math.min(i, j))) continue;
      if (Math.hypot(dx, dy, dz) < 3) clashes++;
    }
  }

  // Contiguous bonded run length forward from each residue (prevents SS windows from bridging
  // chain boundaries or motif/missing-density residue numbering gaps).
  const contigRun = new Int32Array(L);
  for (let i = L - 2; i >= 0; i--) {
    contigRun[i] = isBonded(i) ? contigRun[i + 1] + 1 : 0;
  }

  // Fast Cα distance-based SS estimate: i..i+3 ~5.0-6.0Å & i..i+4 ~5.5-7.0Å => helix;
  // extended i..i+2 ~6.0-7.4Å & i..i+3 ~8.8-11.2Å => strand.
  let helix = 0;
  let sheet = 0;
  for (let i = 0; i < L; i++) {
    const d2 = contigRun[i] >= 2 ? dist(caCoords, i, i + 2) : 0;
    const d3 = contigRun[i] >= 3 ? dist(caCoords, i, i + 3) : 0;
    const d4 = contigRun[i] >= 4 ? dist(caCoords, i, i + 4) : 0;
    if (d3 >= 4.6 && d3 <= 6.2 && d4 >= 5.2 && d4 <= 7.2) {
      helix++;
    } else if (d2 >= 5.9 && d2 <= 7.5 && d3 >= 8.5 && d3 <= 11.4) {
      sheet++;
    }
  }
  const helixPct = (100 * helix) / L;
  const sheetPct = (100 * sheet) / L;
  const coilPct = Math.max(0, 100 - helixPct - sheetPct);

  return { length: L, rg, meanCaDist, chainBreaks, clashes, helixPct, sheetPct, coilPct };
}

function dist(coords: Float32Array, i: number, j: number): number {
  return Math.hypot(
    coords[j * 3 + 0] - coords[i * 3 + 0],
    coords[j * 3 + 1] - coords[i * 3 + 1],
    coords[j * 3 + 2] - coords[i * 3 + 2],
  );
}
