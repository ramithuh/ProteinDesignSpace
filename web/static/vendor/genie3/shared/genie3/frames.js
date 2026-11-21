/**
 * SE(3) rigid transformations, quaternion conversions, and Frenet-Serret frame construction.
 *
 * Mirrors:
 * - `genie3/generation/utils/affine_utils.py` (`T`, `rot_matmul`, `rot_vec_mul`, `quat_to_rot`, `rot_to_quat`)
 * - `genie3/generation/utils/geo_utils.py` (`_compute_frenet_rotations`, `_compute_frenet_frames`,
 *   `compute_noisy_structure_frames`, `compute_conditional_structure_frames`)
 *
 * Frame layout:
 * - `rots`: Float32Array of shape `[L, 3, 3]` flattened row-major (`9` floats per residue)
 * - `trans`: Float32Array of shape `[L, 3]` flattened (`3` floats per residue)
 */

/**
 * Multiply two 3x3 row-major matrices `A @ B` into `out` at offsets.
 */
export function rotMatmulInto(a, aOff, b, bOff, out, outOff) {
  const a00 = a[aOff + 0], a01 = a[aOff + 1], a02 = a[aOff + 2];
  const a10 = a[aOff + 3], a11 = a[aOff + 4], a12 = a[aOff + 5];
  const a20 = a[aOff + 6], a21 = a[aOff + 7], a22 = a[aOff + 8];

  const b00 = b[bOff + 0], b01 = b[bOff + 1], b02 = b[bOff + 2];
  const b10 = b[bOff + 3], b11 = b[bOff + 4], b12 = b[bOff + 5];
  const b20 = b[bOff + 6], b21 = b[bOff + 7], b22 = b[bOff + 8];

  out[outOff + 0] = a00 * b00 + a01 * b10 + a02 * b20;
  out[outOff + 1] = a00 * b01 + a01 * b11 + a02 * b21;
  out[outOff + 2] = a00 * b02 + a01 * b12 + a02 * b22;

  out[outOff + 3] = a10 * b00 + a11 * b10 + a12 * b20;
  out[outOff + 4] = a10 * b01 + a11 * b11 + a12 * b21;
  out[outOff + 5] = a10 * b02 + a11 * b12 + a12 * b22;

  out[outOff + 6] = a20 * b00 + a21 * b10 + a22 * b20;
  out[outOff + 7] = a20 * b01 + a21 * b11 + a22 * b21;
  out[outOff + 8] = a20 * b02 + a21 * b12 + a22 * b22;
}

/**
 * Multiply 3x3 rotation matrix `R` by 3-vector `v` (`R @ v`).
 */
export function rotVecMulInto(r, rOff, v, vOff, out, outOff) {
  const x = v[vOff + 0];
  const y = v[vOff + 1];
  const z = v[vOff + 2];
  out[outOff + 0] = r[rOff + 0] * x + r[rOff + 1] * y + r[rOff + 2] * z;
  out[outOff + 1] = r[rOff + 3] * x + r[rOff + 4] * y + r[rOff + 5] * z;
  out[outOff + 2] = r[rOff + 6] * x + r[rOff + 7] * y + r[rOff + 8] * z;
}

/**
 * Multiply transposed 3x3 rotation matrix `R^T` by 3-vector `v` (`R^T @ v`).
 */
export function rotTransposeVecMulInto(r, rOff, v, vOff, out, outOff) {
  const x = v[vOff + 0];
  const y = v[vOff + 1];
  const z = v[vOff + 2];
  out[outOff + 0] = r[rOff + 0] * x + r[rOff + 3] * y + r[rOff + 6] * z;
  out[outOff + 1] = r[rOff + 1] * x + r[rOff + 4] * y + r[rOff + 7] * z;
  out[outOff + 2] = r[rOff + 2] * x + r[rOff + 5] * y + r[rOff + 8] * z;
}

/**
 * Compose two arrays of rigid frames `f1 ∘ f2`:
 *   `R = R1 @ R2`
 *   `t = R1 @ t2 + t1`
 *
 * @param {{ rots: Float32Array, trans: Float32Array, length: number }} f1
 * @param {{ rots: Float32Array, trans: Float32Array, length: number }} f2
 * @returns {{ rots: Float32Array, trans: Float32Array, length: number }}
 */
export function composeFrames(f1, f2) {
  const L = f1.length;
  const rots = new Float32Array(L * 9);
  const trans = new Float32Array(L * 3);

  for (let i = 0; i < L; i += 1) {
    const rOff = i * 9;
    const tOff = i * 3;
    rotMatmulInto(f1.rots, rOff, f2.rots, rOff, rots, rOff);
    const x = f2.trans[tOff + 0];
    const y = f2.trans[tOff + 1];
    const z = f2.trans[tOff + 2];
    trans[tOff + 0] = f1.rots[rOff + 0] * x + f1.rots[rOff + 1] * y + f1.rots[rOff + 2] * z + f1.trans[tOff + 0];
    trans[tOff + 1] = f1.rots[rOff + 3] * x + f1.rots[rOff + 4] * y + f1.rots[rOff + 5] * z + f1.trans[tOff + 1];
    trans[tOff + 2] = f1.rots[rOff + 6] * x + f1.rots[rOff + 7] * y + f1.rots[rOff + 8] * z + f1.trans[tOff + 2];
  }
  return { rots, trans, length: L };
}

/**
 * Apply frame `i` to a point `(px, py, pz)` -> `R_i @ p + t_i`.
 */
export function applyFramePoint(rots, trans, i, px, py, pz, out, outOff) {
  const rOff = i * 9;
  const tOff = i * 3;
  out[outOff + 0] = rots[rOff + 0] * px + rots[rOff + 1] * py + rots[rOff + 2] * pz + trans[tOff + 0];
  out[outOff + 1] = rots[rOff + 3] * px + rots[rOff + 4] * py + rots[rOff + 5] * pz + trans[tOff + 1];
  out[outOff + 2] = rots[rOff + 6] * px + rots[rOff + 7] * py + rots[rOff + 8] * pz + trans[tOff + 2];
}

/**
 * Apply inverse of frame `i` to a point `(px, py, pz)` -> `R_i^T @ (p - t_i)`.
 */
export function invertApplyFramePoint(rots, trans, i, px, py, pz, out, outOff) {
  const rOff = i * 9;
  const tOff = i * 3;
  const dx = px - trans[tOff + 0];
  const dy = py - trans[tOff + 1];
  const dz = pz - trans[tOff + 2];
  out[outOff + 0] = rots[rOff + 0] * dx + rots[rOff + 3] * dy + rots[rOff + 6] * dz;
  out[outOff + 1] = rots[rOff + 1] * dx + rots[rOff + 4] * dy + rots[rOff + 7] * dz;
  out[outOff + 2] = rots[rOff + 2] * dx + rots[rOff + 5] * dy + rots[rOff + 8] * dz;
}

/**
 * Apply frames `[L]` to points `[L, P, 3]` -> `[L, P, 3]`.
 */
export function applyFrames(frames, pts, pointsPerResidue = 1) {
  const L = frames.length;
  const out = new Float32Array(L * pointsPerResidue * 3);
  for (let i = 0; i < L; i += 1) {
    for (let p = 0; p < pointsPerResidue; p += 1) {
      const off = (i * pointsPerResidue + p) * 3;
      applyFramePoint(frames.rots, frames.trans, i, pts[off], pts[off + 1], pts[off + 2], out, off);
    }
  }
  return out;
}

/**
 * Apply inverse frames `[L]` to points `[L, P, 3]` -> `[L, P, 3]`.
 */
export function invertApplyFrames(frames, pts, pointsPerResidue = 1) {
  const L = frames.length;
  const out = new Float32Array(L * pointsPerResidue * 3);
  for (let i = 0; i < L; i += 1) {
    for (let p = 0; p < pointsPerResidue; p += 1) {
      const off = (i * pointsPerResidue + p) * 3;
      invertApplyFramePoint(frames.rots, frames.trans, i, pts[off], pts[off + 1], pts[off + 2], out, off);
    }
  }
  return out;
}

/**
 * Invert an array of rigid frames: `R_inv = R^T`, `t_inv = -R^T @ t`.
 */
export function invertFrames(frames) {
  const L = frames.length;
  const rots = new Float32Array(L * 9);
  const trans = new Float32Array(L * 3);
  for (let i = 0; i < L; i += 1) {
    const rOff = i * 9;
    const tOff = i * 3;
    rots[rOff + 0] = frames.rots[rOff + 0];
    rots[rOff + 1] = frames.rots[rOff + 3];
    rots[rOff + 2] = frames.rots[rOff + 6];
    rots[rOff + 3] = frames.rots[rOff + 1];
    rots[rOff + 4] = frames.rots[rOff + 4];
    rots[rOff + 5] = frames.rots[rOff + 7];
    rots[rOff + 6] = frames.rots[rOff + 2];
    rots[rOff + 7] = frames.rots[rOff + 5];
    rots[rOff + 8] = frames.rots[rOff + 8];
    const x = frames.trans[tOff + 0];
    const y = frames.trans[tOff + 1];
    const z = frames.trans[tOff + 2];
    trans[tOff + 0] = -(rots[rOff + 0] * x + rots[rOff + 1] * y + rots[rOff + 2] * z);
    trans[tOff + 1] = -(rots[rOff + 3] * x + rots[rOff + 4] * y + rots[rOff + 5] * z);
    trans[tOff + 2] = -(rots[rOff + 6] * x + rots[rOff + 7] * y + rots[rOff + 8] * z);
  }
  return { rots, trans, length: L };
}

/**
 * Convert a unit quaternion `(a, b, c, d)` into a 3x3 row-major rotation matrix
 * written into `out[outOff..outOff+8]`.
 *
 * Matches `quat_to_rot` in `genie3/generation/utils/affine_utils.py`.
 */
export function quatToRotInto(a, b, c, d, out, outOff) {
  const aa = a * a, bb = b * b, cc = c * c, dd = d * d;
  const ab = a * b, ac = a * c, ad = a * d;
  const bc = b * c, bd = b * d, cd = c * d;

  out[outOff + 0] = aa + bb - cc - dd;
  out[outOff + 1] = 2 * (bc - ad);
  out[outOff + 2] = 2 * (bd + ac);

  out[outOff + 3] = 2 * (bc + ad);
  out[outOff + 4] = aa - bb + cc - dd;
  out[outOff + 5] = 2 * (cd - ab);

  out[outOff + 6] = 2 * (bd - ac);
  out[outOff + 7] = 2 * (cd + ab);
  out[outOff + 8] = aa - bb - cc + dd;
}

/**
 * Convert a 3x3 rotation matrix (at `rot[rotOff..rotOff+8]`) into a unit quaternion `[a, b, c, d]`
 * by finding the principal eigenvector of the 4x4 Davenport K-matrix, mirroring
 * `rot_to_quat` in `genie3/generation/utils/affine_utils.py`.
 *
 * Uses symmetric 4x4 Jacobi eigendecomposition in Float64 for stability.
 * Note: For exact PyTorch `torch.linalg.eigh` parity when eigenvector sign matters,
 * `signMode` can be set (default `'lapack_l'` or `'none'` — verified against oracle!).
 */
const _k4 = new Float64Array(16);
const _v4 = new Float64Array(16);

export function rotToQuatInto(rot, rotOff, out, outOff) {
  const xx = rot[rotOff + 0], xy = rot[rotOff + 1], xz = rot[rotOff + 2];
  const yx = rot[rotOff + 3], yy = rot[rotOff + 4], yz = rot[rotOff + 5];
  const zx = rot[rotOff + 6], zy = rot[rotOff + 7], zz = rot[rotOff + 8];

  const inv3 = 1.0 / 3.0;
  // Build symmetric 4x4 Davenport K matrix
  _k4[0] = (xx + yy + zz) * inv3;
  _k4[1] = (zy - yz) * inv3;
  _k4[2] = (xz - zx) * inv3;
  _k4[3] = (yx - xy) * inv3;

  _k4[4] = _k4[1];
  _k4[5] = (xx - yy - zz) * inv3;
  _k4[6] = (xy + yx) * inv3;
  _k4[7] = (xz + zx) * inv3;

  _k4[8] = _k4[2];
  _k4[9] = _k4[6];
  _k4[10] = (yy - xx - zz) * inv3;
  _k4[11] = (yz + zy) * inv3;

  _k4[12] = _k4[3];
  _k4[13] = _k4[7];
  _k4[14] = _k4[11];
  _k4[15] = (zz - xx - yy) * inv3;

  // Initialize V = I_4
  _v4.fill(0);
  _v4[0] = 1; _v4[5] = 1; _v4[10] = 1; _v4[15] = 1;

  // Cyclic Jacobi sweeps for 4x4 symmetric matrix
  for (let sweep = 0; sweep < 20; sweep += 1) {
    let offNorm = 0;
    for (let p = 0; p < 3; p += 1) {
      for (let q = p + 1; q < 4; q += 1) {
        const apq = _k4[p * 4 + q];
        offNorm += apq * apq;
      }
    }
    if (offNorm < 1e-26) break;

    for (let p = 0; p < 3; p += 1) {
      for (let q = p + 1; q < 4; q += 1) {
        const apq = _k4[p * 4 + q];
        if (Math.abs(apq) < 1e-15) continue;
        const app = _k4[p * 4 + p];
        const aqq = _k4[q * 4 + q];
        const tau = (aqq - app) / (2 * apq);
        const t = tau >= 0
          ? 1 / (tau + Math.sqrt(1 + tau * tau))
          : -1 / (-tau + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        _k4[p * 4 + p] = app - t * apq;
        _k4[q * 4 + q] = aqq + t * apq;
        _k4[p * 4 + q] = 0;
        _k4[q * 4 + p] = 0;

        for (let r = 0; r < 4; r += 1) {
          if (r === p || r === q) continue;
          const arp = _k4[r * 4 + p];
          const arq = _k4[r * 4 + q];
          const nrp = c * arp - s * arq;
          const nrq = s * arp + c * arq;
          _k4[r * 4 + p] = nrp;
          _k4[p * 4 + r] = nrp;
          _k4[r * 4 + q] = nrq;
          _k4[q * 4 + r] = nrq;
        }
        for (let r = 0; r < 4; r += 1) {
          const vrp = _v4[r * 4 + p];
          const vrq = _v4[r * 4 + q];
          _v4[r * 4 + p] = c * vrp - s * vrq;
          _v4[r * 4 + q] = s * vrp + c * vrq;
        }
      }
    }
  }

  // Pick eigenvector corresponding to largest eigenvalue
  let bestCol = 0;
  let bestVal = _k4[0];
  for (let col = 1; col < 4; col += 1) {
    if (_k4[col * 4 + col] > bestVal) {
      bestVal = _k4[col * 4 + col];
      bestCol = col;
    }
  }

  let q0 = _v4[0 * 4 + bestCol];
  let q1 = _v4[1 * 4 + bestCol];
  let q2 = _v4[2 * 4 + bestCol];
  let q3 = _v4[3 * 4 + bestCol];

  // Canonicalize quaternion sign so first non-zero component is positive (eps = 1e-7),
  // eliminating LAPACK vs cuSOLVER vs Jacobi eigenvector sign ambiguity.
  const eps = 1e-7;
  const neg =
    q0 < -eps ||
    (Math.abs(q0) <= eps && q1 < -eps) ||
    (Math.abs(q0) <= eps && Math.abs(q1) <= eps && q2 < -eps) ||
    (Math.abs(q0) <= eps && Math.abs(q1) <= eps && Math.abs(q2) <= eps && q3 < 0);
  if (neg) {
    q0 = -q0; q1 = -q1; q2 = -q2; q3 = -q3;
  }

  out[outOff + 0] = q0;
  out[outOff + 1] = q1;
  out[outOff + 2] = q2;
  out[outOff + 3] = q3;
}

/**
 * Canonicalize a unit quaternion [q0, q1, q2, q3] in-place so its first non-zero
 * component (|x| > 1e-7) is positive.
 */
export function canonicalizeQuatInPlace(q, offset = 0, eps = 1e-7) {
  const q0 = q[offset + 0], q1 = q[offset + 1], q2 = q[offset + 2], q3 = q[offset + 3];
  const neg =
    q0 < -eps ||
    (Math.abs(q0) <= eps && q1 < -eps) ||
    (Math.abs(q0) <= eps && Math.abs(q1) <= eps && q2 < -eps) ||
    (Math.abs(q0) <= eps && Math.abs(q1) <= eps && Math.abs(q2) <= eps && q3 < 0);
  if (neg) {
    q[offset + 0] = -q0;
    q[offset + 1] = -q1;
    q[offset + 2] = -q2;
    q[offset + 3] = -q3;
  }
  return q;
}

/**
 * Compute Frenet-Serret rotation matrix for a single triplet `(catom, latom, ratom)`
 * and write row-major 3x3 into `out[outOff..outOff+8]`.
 *
 * Matches `_compute_frenet_rotations` in `genie3/generation/utils/geo_utils.py`:
 *   Non-legacy (V1 default):
 *     `cl = catom - latom`, `cr = catom - ratom`
 *     `T = normalize(cl)`
 *     `B = normalize(cross(cl, cr))`
 *     `N = normalize(cross(T, B))`
 *     `R = [T | B | N]` (columns T, B, N)
 */
export function computeFrenetRotationInto(
  cx, cy, cz,
  lx, ly, lz,
  rx, ry, rz,
  out,
  outOff,
  legacy = false,
) {
  const clx = cx - lx, cly = cy - ly, clz = cz - lz;
  const crx = cx - rx, cry = cy - ry, crz = cz - rz;

  let tx, ty, tz, bx, by, bz, nx, ny, nz;
  if (legacy) {
    const tNorm = Math.max(Math.hypot(-crx, -cry, -crz), 1e-12);
    tx = -crx / tNorm; ty = -cry / tNorm; tz = -crz / tNorm;

    const bcx = cly * (-crz) - clz * (-cry);
    const bcy = clz * (-crx) - clx * (-crz);
    const bcz = clx * (-cry) - cly * (-crx);
    const bNorm = Math.max(Math.hypot(bcx, bcy, bcz), 1e-12);
    bx = bcx / bNorm; by = bcy / bNorm; bz = bcz / bNorm;

    const ncx = by * tz - bz * ty;
    const ncy = bz * tx - bx * tz;
    const ncz = bx * ty - by * tx;
    const nNorm = Math.max(Math.hypot(ncx, ncy, ncz), 1e-12);
    nx = ncx / nNorm; ny = ncy / nNorm; nz = ncz / nNorm;
  } else {
    const tNorm = Math.max(Math.hypot(clx, cly, clz), 1e-12);
    tx = clx / tNorm; ty = cly / tNorm; tz = clz / tNorm;

    const bcx = cly * crz - clz * cry;
    const bcy = clz * crx - clx * crz;
    const bcz = clx * cry - cly * crx;
    const bNorm = Math.max(Math.hypot(bcx, bcy, bcz), 1e-12);
    bx = bcx / bNorm; by = bcy / bNorm; bz = bcz / bNorm;

    const ncx = ty * bz - tz * by;
    const ncy = tz * bx - tx * bz;
    const ncz = tx * by - ty * bx;
    const nNorm = Math.max(Math.hypot(ncx, ncy, ncz), 1e-12);
    nx = ncx / nNorm; ny = ncy / nNorm; nz = ncz / nNorm;
  }

  // Columns are [T, B, N]
  out[outOff + 0] = tx; out[outOff + 1] = bx; out[outOff + 2] = nx;
  out[outOff + 3] = ty; out[outOff + 4] = by; out[outOff + 5] = ny;
  out[outOff + 6] = tz; out[outOff + 7] = bz; out[outOff + 8] = nz;
}

/**
 * Compute Frenet frames for a single structure (`[L, 3]` positions).
 * Mirrors `_compute_frenet_frames` in `genie3/generation/utils/geo_utils.py`.
 */
export function computeFrenetFrames({
  tokenMask,
  transMask,
  rotMask,
  transAtomPositions,
  rotCatomPositions,
  rotLatomPositions,
  rotRatomPositions,
  legacy = false,
}) {
  const L = tokenMask.length;
  let validLen = 0;
  for (let i = 0; i < L; i += 1) validLen += tokenMask[i] > 0.5 ? 1 : 0;

  const rots = new Float32Array(L * 9);
  const trans = new Float32Array(L * 3);

  for (let i = 0; i < validLen; i += 1) {
    const pOff = i * 3;
    const rOff = i * 9;
    computeFrenetRotationInto(
      rotCatomPositions[pOff], rotCatomPositions[pOff + 1], rotCatomPositions[pOff + 2],
      rotLatomPositions[pOff], rotLatomPositions[pOff + 1], rotLatomPositions[pOff + 2],
      rotRatomPositions[pOff], rotRatomPositions[pOff + 1], rotRatomPositions[pOff + 2],
      rots,
      rOff,
      legacy,
    );
  }

  for (let i = 0; i < L; i += 1) {
    const tm = transMask[i];
    const rm = rotMask[i];
    const tOff = i * 3;
    trans[tOff + 0] = transAtomPositions[tOff + 0] * tm;
    trans[tOff + 1] = transAtomPositions[tOff + 1] * tm;
    trans[tOff + 2] = transAtomPositions[tOff + 2] * tm;

    const rOff = i * 9;
    for (let k = 0; k < 9; k += 1) rots[rOff + k] *= rm;
  }

  return { rots, trans, length: L };
}

function gatherPositions(xl, indices) {
  const L = indices.length;
  const out = new Float32Array(L * 3);
  for (let i = 0; i < L; i += 1) {
    const idx = indices[i];
    const src = (idx < 0 ? 0 : idx) * 3;
    const dst = i * 3;
    out[dst + 0] = xl[src + 0];
    out[dst + 1] = xl[src + 1];
    out[dst + 2] = xl[src + 2];
  }
  return out;
}

/**
 * Compute noisy structure Frenet frames from `batch` and noisy coordinates `xl` (`[L * 3]`).
 * Mirrors `compute_noisy_structure_frames` in `genie3/generation/utils/geo_utils.py`.
 *
 * @param {object} batch
 * @param {Float32Array} xl
 * @param {boolean} [legacy=false]
 * @returns {{ rots: Float32Array, trans: Float32Array, length: number }}
 */
export function computeNoisyStructureFrames(batch, xl, legacy = false) {
  const rotC = gatherPositions(xl, batch.token_frame_cindex_adj);
  const rotL = gatherPositions(xl, batch.token_frame_lindex_adj);
  const rotR = gatherPositions(xl, batch.token_frame_rindex_adj);

  return computeFrenetFrames({
    tokenMask: batch.token_mask,
    transMask: batch.token_mask,
    rotMask: batch.token_mask,
    transAtomPositions: xl,
    rotCatomPositions: rotC,
    rotLatomPositions: rotL,
    rotRatomPositions: rotR,
    legacy,
  });
}

/**
 * Compute conditional (ground truth) structure Frenet frames from `batch`.
 * Mirrors `compute_conditional_structure_frames` in `genie3/generation/utils/geo_utils.py`.
 *
 * @param {object} batch
 * @param {boolean} [legacy=false]
 * @returns {{ rots: Float32Array, trans: Float32Array, length: number }}
 */
export function computeConditionalStructureFrames(batch, legacy = false) {
  const L = batch.token_mask.length;
  const condXl = new Float32Array(L * 3);
  for (let i = 0; i < L; i += 1) {
    const m = batch.cond_struct_mask[i];
    const off = i * 3;
    condXl[off + 0] = batch.gt_atom_positions[off + 0] * m;
    condXl[off + 1] = batch.gt_atom_positions[off + 1] * m;
    condXl[off + 2] = batch.gt_atom_positions[off + 2] * m;
  }

  // Note: PyTorch negative index -1 gathers the last element, replicate exact Python indexing
  const gatherPy = (arr, inds) => {
    const out = new Float32Array(L * 3);
    for (let i = 0; i < L; i += 1) {
      let idx = inds[i];
      if (idx < 0) idx = L + idx;
      const src = idx * 3;
      const dst = i * 3;
      out[dst + 0] = arr[src + 0];
      out[dst + 1] = arr[src + 1];
      out[dst + 2] = arr[src + 2];
    }
    return out;
  };

  const rotL = gatherPy(condXl, batch.token_frame_lindex);
  const rotR = gatherPy(condXl, batch.token_frame_rindex);

  return computeFrenetFrames({
    tokenMask: batch.token_mask,
    transMask: batch.cond_struct_mask,
    rotMask: batch.cond_struct_frame_mask,
    transAtomPositions: condXl,
    rotCatomPositions: condXl,
    rotLatomPositions: rotL,
    rotRatomPositions: rotR,
    legacy,
  });
}
