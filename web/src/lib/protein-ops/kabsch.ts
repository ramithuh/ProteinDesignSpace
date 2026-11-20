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
  const n = Math.min(mobile.length, target.length) / 3;
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
  const n = Math.min(mobile.length, target.length) / 3;
  if (n < 3) {
    return {
      rmsd: 0,
      tmScore: 1,
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

function padLeft(s: string, width: number): string {
  return s.length >= width ? s : ' '.repeat(width - s.length) + s;
}

/** Apply rigid rotation R (row-major 3x3) + translation t to every ATOM/HETATM line of a PDB. */
export function applyTransformToPdb(
  pdbText: string,
  R: Float64Array,
  t: Float64Array,
): string {
  const lines = pdbText.split('\n');
  const out = new Array<string>(lines.length);
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
        out[i] =
          line.slice(0, 30) +
          padLeft(nx.toFixed(3), 8) +
          padLeft(ny.toFixed(3), 8) +
          padLeft(nz.toFixed(3), 8) +
          line.slice(54);
        continue;
      }
    }
    out[i] = line;
  }
  return out.join('\n');
}

/** Fast Cα-only geometric quality & secondary structure estimator (P-SEA-inspired). */
export function computeGeometryStats(caCoords: Float32Array, chainIds?: string[]): GeometryStats {
  const L = Math.floor(caCoords.length / 3);
  if (L === 0) {
    return { length: 0, rg: 0, meanCaDist: 0, chainBreaks: 0, clashes: 0, helixPct: 0, sheetPct: 0, coilPct: 100 };
  }
  const hasChains = Boolean(chainIds && chainIds.length === L);
  const sameChain = (i: number, j: number): boolean => !hasChains || chainIds![i] === chainIds![j];

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
    if (!sameChain(i, i + 1)) continue;
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

  let clashes = 0;
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      if (j === i + 1 && sameChain(i, j)) continue;
      const d = Math.hypot(
        caCoords[j * 3 + 0] - caCoords[i * 3 + 0],
        caCoords[j * 3 + 1] - caCoords[i * 3 + 1],
        caCoords[j * 3 + 2] - caCoords[i * 3 + 2],
      );
      if (d < 3.0) clashes++;
    }
  }

  // Fast Cα distance-based SS estimate: i..i+3 ~5.0-6.0Å & i..i+4 ~5.5-7.0Å => helix;
  // extended i..i+2 ~6.0-7.4Å & i..i+3 ~8.8-11.2Å => strand.
  let helix = 0;
  let sheet = 0;
  for (let i = 0; i < L; i++) {
    const d2 = i + 2 < L && sameChain(i, i + 2) ? dist(caCoords, i, i + 2) : 0;
    const d3 = i + 3 < L && sameChain(i, i + 3) ? dist(caCoords, i, i + 3) : 0;
    const d4 = i + 4 < L && sameChain(i, i + 4) ? dist(caCoords, i, i + 4) : 0;
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
