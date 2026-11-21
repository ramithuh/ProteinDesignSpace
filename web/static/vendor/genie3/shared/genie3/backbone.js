/**
 * Ideal protein backbone (`N, CA, C, O`) reconstruction from diffused `Cα` coordinates.
 *
 * Genie 3 diffuses `Cα` coordinates (`[L, 3]`, in Ångströms). Downstream visualization
 * (Mol* cartoon/ribbons), inverse folding (ProteinMPNN full-backbone), and folding
 * validation (LocalFold) benefit from ideal planar trans-peptide `N, CA, C, O` atoms
 * anchored at the exact diffused `Cα` positions.
 *
 * Geometry uses Engh & Huber ideal trans-peptide parameters:
 *   CA-C = 1.525 Å, C-N = 1.329 Å, N-CA = 1.458 Å, C=O = 1.231 Å
 *   CA-C-N = 116.2°, C-N-CA = 121.7°, CA-C-O = 120.8°, N-CA-C = 111.2°, ω = 180°
 */

// Precomputed 2D coordinates in the trans-peptide plane spanned by
// e_x = normalize(CA_{i+1} - CA_i) and e_y = peptide plane normal (pointing toward C=O):
//   alpha = 20.502° (angle of CA_i -> C_i above e_x)
//   CA_i -> C_i = (1.4285, +0.5341)  (|CA-C| = 1.525 Å)
//   C_i -> O_i  = (0.2201, +1.2112)  (|C=O| = 1.231 Å, angle CA-C-O = 120.8°)
//   CA_{i+1} -> N_{i+1} = (-1.4083, -0.3774)  (|N-CA| = 1.458 Å, |C-N| = 1.329 Å at d_CA = 3.804 Å)
const C_EX = 1.4285;
const C_EY = 0.5341;
const O_EX = 0.2201;
const O_EY = 1.2112;
const N_EX_FROM_NEXT_CA = -1.4083;
const N_EY_FROM_NEXT_CA = -0.3774;

const N_CA_LEN = 1.458;
const CA_C_LEN = 1.525;
const C_O_LEN = 1.231;

function normalize3(x, y, z) {
  const len = Math.hypot(x, y, z);
  if (len < 1e-8) return [1, 0, 0];
  return [x / len, y / len, z / len];
}

function cross3(ax, ay, az, bx, by, bz) {
  return [
    ay * bz - az * by,
    az * bx - ax * bz,
    ax * by - ay * bx,
  ];
}

function fallbackOrthogonal(ux, uy, uz) {
  const ax = Math.abs(ux) < 0.9 ? 1 : 0;
  const ay = Math.abs(ux) < 0.9 ? 0 : 1;
  const [ox, oy, oz] = cross3(ux, uy, uz, ax, ay, 0);
  return normalize3(ox, oy, oz);
}

/**
 * Center `Cα` coordinates (`[L * 3]`) by subtracting their (masked) mean.
 * Mirrors the centering step in `save_np_features_to_pdb`.
 *
 * @param {Float32Array} caCoords Flat `[L * 3]`
 * @param {ArrayLike<number>} [mask] Optional `[L]` mask
 * @returns {Float32Array} Centered copy `[L * 3]`
 */
export function centerCaCoords(caCoords, mask = null) {
  const L = Math.floor(caCoords.length / 3);
  let sx = 0, sy = 0, sz = 0, count = 0;
  for (let i = 0; i < L; i += 1) {
    if (mask && mask[i] < 0.5) continue;
    sx += caCoords[i * 3 + 0];
    sy += caCoords[i * 3 + 1];
    sz += caCoords[i * 3 + 2];
    count += 1;
  }
  const out = new Float32Array(L * 3);
  if (count === 0) return out;
  const mx = sx / count;
  const my = sy / count;
  const mz = sz / count;
  for (let i = 0; i < L; i += 1) {
    const m = mask ? (mask[i] > 0.5 ? 1 : 0) : 1;
    out[i * 3 + 0] = (caCoords[i * 3 + 0] - mx) * m;
    out[i * 3 + 1] = (caCoords[i * 3 + 1] - my) * m;
    out[i * 3 + 2] = (caCoords[i * 3 + 2] - mz) * m;
  }
  return out;
}

/**
 * Reconstruct 4-atom backbone coordinates (`N, CA, C, O` = `[L * 4 * 3]`) from `Cα` coordinates (`[L * 3]`).
 *
 * Guarantees:
 * - Exact preservation of `CA_i` coordinates (or centered if `center: true`).
 * - Ideal trans-peptide bond lengths (`CA-C = 1.525 Å`, `C=O = 1.231 Å`, `N-CA = 1.458 Å`,
 *   and `C-N ≈ 1.329 Å` when `d(CA_i, CA_{i+1}) ≈ 3.80 Å`).
 * - Right-handed α-helix-consistent peptide plane orientation (carbonyl `O` aligned with local binormal).
 * - Multi-chain safe via optional `chainIds` / `asymId`.
 *
 * @param {Float32Array} caCoords Flat `[L * 3]`
 * @param {{
 *   chainIds?: ArrayLike<string | number> | null,
 *   center?: boolean,
 * }} [opts]
 * @returns {Float32Array} Flat `[L * 4 * 3]` in `[N, CA, C, O]` order per residue
 */
export function reconstructBackboneFromCa(caCoords, opts = {}) {
  const L = Math.floor(caCoords.length / 3);
  const ca = opts.center ? centerCaCoords(caCoords) : caCoords;
  const chainIds = opts.chainIds ?? null;
  const out = new Float32Array(L * 4 * 3);
  if (L === 0) return out;

  const sameChain = (a, b) => {
    if (a < 0 || b < 0 || a >= L || b >= L) return false;
    if (!chainIds) return true;
    return chainIds[a] === chainIds[b];
  };

  // Copy CA into slot 1 for every residue
  for (let i = 0; i < L; i += 1) {
    const src = i * 3;
    const dst = (i * 4 + 1) * 3;
    out[dst + 0] = ca[src + 0];
    out[dst + 1] = ca[src + 1];
    out[dst + 2] = ca[src + 2];
  }

  // Step 1: Compute local Cα-plane binormal p_i = normalize((CA_i - CA_{i-1}) x (CA_{i+1} - CA_i))
  const binormals = new Float32Array(L * 3);
  const hasBinormal = new Uint8Array(L);
  for (let i = 1; i < L - 1; i += 1) {
    if (!sameChain(i - 1, i) || !sameChain(i, i + 1)) continue;
    const v1x = ca[i * 3 + 0] - ca[(i - 1) * 3 + 0];
    const v1y = ca[i * 3 + 1] - ca[(i - 1) * 3 + 1];
    const v1z = ca[i * 3 + 2] - ca[(i - 1) * 3 + 2];
    const v2x = ca[(i + 1) * 3 + 0] - ca[i * 3 + 0];
    const v2y = ca[(i + 1) * 3 + 1] - ca[i * 3 + 1];
    const v2z = ca[(i + 1) * 3 + 2] - ca[i * 3 + 2];
    const [bx, by, bz] = cross3(v1x, v1y, v1z, v2x, v2y, v2z);
    const len = Math.hypot(bx, by, bz);
    if (len > 1e-6) {
      binormals[i * 3 + 0] = bx / len;
      binormals[i * 3 + 1] = by / len;
      binormals[i * 3 + 2] = bz / len;
      hasBinormal[i] = 1;
    }
  }

  // Step 2: For each peptide bond (i -> i+1) in the same chain, place C_i, O_i, and N_{i+1}
  const hasN = new Uint8Array(L);
  const hasC = new Uint8Array(L);

  for (let i = 0; i < L - 1; i += 1) {
    if (!sameChain(i, i + 1)) continue;
    const ax = ca[i * 3 + 0], ay = ca[i * 3 + 1], az = ca[i * 3 + 2];
    const bx = ca[(i + 1) * 3 + 0], by = ca[(i + 1) * 3 + 1], bz = ca[(i + 1) * 3 + 2];
    const [exX, exY, exZ] = normalize3(bx - ax, by - ay, bz - az);

    // Choose peptide plane normal direction from local binormals
    let px = 0, py = 0, pz = 0;
    if (hasBinormal[i] && hasBinormal[i + 1]) {
      // Align signs if opposite (e.g. β-strand alternation) or average smoothly
      const b0x = binormals[i * 3], b0y = binormals[i * 3 + 1], b0z = binormals[i * 3 + 2];
      const b1x = binormals[(i + 1) * 3], b1y = binormals[(i + 1) * 3 + 1], b1z = binormals[(i + 1) * 3 + 2];
      const dot = b0x * b1x + b0y * b1y + b0z * b1z;
      if (dot >= -0.2) {
        px = b0x + b1x; py = b0y + b1y; pz = b0z + b1z;
      } else {
        // In alternating extended β-strands, use b0 directly so adjacent carbonyls alternate cleanly
        px = b0x; py = b0y; pz = b0z;
      }
    } else if (hasBinormal[i]) {
      px = binormals[i * 3]; py = binormals[i * 3 + 1]; pz = binormals[i * 3 + 2];
    } else if (hasBinormal[i + 1]) {
      px = binormals[(i + 1) * 3]; py = binormals[(i + 1) * 3 + 1]; pz = binormals[(i + 1) * 3 + 2];
    }

    // Orthogonalize p against e_x -> e_y
    const proj = px * exX + py * exY + pz * exZ;
    let eyX = px - proj * exX;
    let eyY = py - proj * exY;
    let eyZ = pz - proj * exZ;
    const eyLen = Math.hypot(eyX, eyY, eyZ);
    if (eyLen < 1e-6) {
      [eyX, eyY, eyZ] = fallbackOrthogonal(exX, exY, exZ);
    } else {
      eyX /= eyLen; eyY /= eyLen; eyZ /= eyLen;
    }

    // Place C_i (slot 2 of residue i)
    const cOff = (i * 4 + 2) * 3;
    const cx = ax + C_EX * exX + C_EY * eyX;
    const cy = ay + C_EX * exY + C_EY * eyY;
    const cz = az + C_EX * exZ + C_EY * eyZ;
    out[cOff + 0] = cx; out[cOff + 1] = cy; out[cOff + 2] = cz;

    // Place O_i (slot 3 of residue i)
    const oOff = (i * 4 + 3) * 3;
    out[oOff + 0] = cx + O_EX * exX + O_EY * eyX;
    out[oOff + 1] = cy + O_EX * exY + O_EY * eyY;
    out[oOff + 2] = cz + O_EX * exZ + O_EY * eyZ;
    hasC[i] = 1;

    // Place N_{i+1} (slot 0 of residue i+1)
    const nOff = ((i + 1) * 4 + 0) * 3;
    out[nOff + 0] = bx + N_EX_FROM_NEXT_CA * exX + N_EY_FROM_NEXT_CA * eyX;
    out[nOff + 1] = by + N_EX_FROM_NEXT_CA * exY + N_EY_FROM_NEXT_CA * eyY;
    out[nOff + 2] = bz + N_EX_FROM_NEXT_CA * exZ + N_EY_FROM_NEXT_CA * eyZ;
    hasN[i + 1] = 1;
  }

  // Step 3: Fill missing N-terminal N and C-terminal C, O with ideal tetrahedral angles (111.2°)
  const cosTheta = Math.cos((111.2 * Math.PI) / 180); // ~ -0.3616
  const sinTheta = Math.sin((111.2 * Math.PI) / 180); // ~  0.9323
  const cosCO = Math.cos((120.8 * Math.PI) / 180);
  const sinCO = Math.sin((120.8 * Math.PI) / 180);

  for (let i = 0; i < L; i += 1) {
    const caX = ca[i * 3 + 0], caY = ca[i * 3 + 1], caZ = ca[i * 3 + 2];

    if (!hasN[i]) {
      let ux, uy, uz;
      if (hasC[i]) {
        const cOff = (i * 4 + 2) * 3;
        [ux, uy, uz] = normalize3(out[cOff] - caX, out[cOff + 1] - caY, out[cOff + 2] - caZ);
      } else {
        [ux, uy, uz] = [1, 0, 0];
      }
      // Choose an orthogonal direction using O_i - C_i if available
      let vx = 0, vy = 0, vz = 0;
      if (hasC[i]) {
        const cOff = (i * 4 + 2) * 3;
        const oOff = (i * 4 + 3) * 3;
        vx = out[oOff] - out[cOff];
        vy = out[oOff + 1] - out[cOff + 1];
        vz = out[oOff + 2] - out[cOff + 2];
        const d = vx * ux + vy * uy + vz * uz;
        vx -= d * ux; vy -= d * uy; vz -= d * uz;
      }
      const vLen = Math.hypot(vx, vy, vz);
      if (vLen < 1e-6) {
        [vx, vy, vz] = fallbackOrthogonal(ux, uy, uz);
      } else {
        vx /= vLen; vy /= vLen; vz /= vLen;
      }
      const nOff = (i * 4 + 0) * 3;
      out[nOff + 0] = caX + N_CA_LEN * (cosTheta * ux - sinTheta * vx);
      out[nOff + 1] = caY + N_CA_LEN * (cosTheta * uy - sinTheta * vy);
      out[nOff + 2] = caZ + N_CA_LEN * (cosTheta * uz - sinTheta * vz);
      hasN[i] = 1;
    }

    if (!hasC[i]) {
      const nOff = (i * 4 + 0) * 3;
      const [ux, uy, uz] = normalize3(out[nOff] - caX, out[nOff + 1] - caY, out[nOff + 2] - caZ);
      let vx = 0, vy = 0, vz = 0;
      if (i > 0 && sameChain(i - 1, i)) {
        const prevCOff = ((i - 1) * 4 + 2) * 3;
        vx = out[prevCOff] - caX;
        vy = out[prevCOff + 1] - caY;
        vz = out[prevCOff + 2] - caZ;
        const d = vx * ux + vy * uy + vz * uz;
        vx -= d * ux; vy -= d * uy; vz -= d * uz;
      }
      const vLen = Math.hypot(vx, vy, vz);
      if (vLen < 1e-6) {
        [vx, vy, vz] = fallbackOrthogonal(ux, uy, uz);
      } else {
        vx /= vLen; vy /= vLen; vz /= vLen;
      }
      const cOff = (i * 4 + 2) * 3;
      const cx = caX + CA_C_LEN * (cosTheta * ux - sinTheta * vx);
      const cy = caY + CA_C_LEN * (cosTheta * uy - sinTheta * vy);
      const cz = caZ + CA_C_LEN * (cosTheta * uz - sinTheta * vz);
      out[cOff + 0] = cx; out[cOff + 1] = cy; out[cOff + 2] = cz;

      // Place terminal O in (N, CA, C) plane at 120.8° from C -> CA
      const [v1x, v1y, v1z] = normalize3(caX - cx, caY - cy, caZ - cz);
      let u2x = out[nOff] - caX;
      let u2y = out[nOff + 1] - caY;
      let u2z = out[nOff + 2] - caZ;
      const dot = u2x * v1x + u2y * v1y + u2z * v1z;
      u2x -= dot * v1x; u2y -= dot * v1y; u2z -= dot * v1z;
      const u2Len = Math.hypot(u2x, u2y, u2z);
      if (u2Len < 1e-6) {
        [u2x, u2y, u2z] = fallbackOrthogonal(v1x, v1y, v1z);
      } else {
        u2x /= u2Len; u2y /= u2Len; u2z /= u2Len;
      }
      const oOff = (i * 4 + 3) * 3;
      out[oOff + 0] = cx + C_O_LEN * (cosCO * v1x + sinCO * u2x);
      out[oOff + 1] = cy + C_O_LEN * (cosCO * v1y + sinCO * u2y);
      out[oOff + 2] = cz + C_O_LEN * (cosCO * v1z + sinCO * u2z);
      hasC[i] = 1;
    }
  }

  return out;
}

/**
 * Extract `Cα` coordinates (`[L * 3]`) from 4-atom backbone coordinates (`[L * 4 * 3]`).
 */
export function extractCaFromBackbone(backbone4, L) {
  const ca = new Float32Array(L * 3);
  for (let i = 0; i < L; i += 1) {
    const src = (i * 4 + 1) * 3;
    const dst = i * 3;
    ca[dst + 0] = backbone4[src + 0];
    ca[dst + 1] = backbone4[src + 1];
    ca[dst + 2] = backbone4[src + 2];
  }
  return ca;
}

/**
 * Compute mean backbone bond lengths from `[L * 4 * 3]` coordinates for validation.
 */
export function computeBackboneGeometryStats(backbone4, L) {
  const dist = (aOff, bOff) =>
    Math.hypot(
      backbone4[aOff + 0] - backbone4[bOff + 0],
      backbone4[aOff + 1] - backbone4[bOff + 1],
      backbone4[aOff + 2] - backbone4[bOff + 2],
    );

  let nCa = 0, caC = 0, cO = 0, cN = 0, caCa = 0;
  for (let i = 0; i < L; i += 1) {
    const nOff = (i * 4 + 0) * 3;
    const caOff = (i * 4 + 1) * 3;
    const cOff = (i * 4 + 2) * 3;
    const oOff = (i * 4 + 3) * 3;
    nCa += dist(nOff, caOff);
    caC += dist(caOff, cOff);
    cO += dist(cOff, oOff);
    if (i < L - 1) {
      const nextNOff = ((i + 1) * 4 + 0) * 3;
      const nextCaOff = ((i + 1) * 4 + 1) * 3;
      cN += dist(cOff, nextNOff);
      caCa += dist(caOff, nextCaOff);
    }
  }
  return {
    meanNCa: L > 0 ? nCa / L : 0,
    meanCaC: L > 0 ? caC / L : 0,
    meanCO: L > 0 ? cO / L : 0,
    meanCN: L > 1 ? cN / (L - 1) : 0,
    meanCaCa: L > 1 ? caCa / (L - 1) : 0,
  };
}
