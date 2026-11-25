/**
 * Genie 3 V1Denoiser on WebGPU (`Genie3DenoiserGpu`).
 *
 * Mirrors `.upstream/genie3/src/genie3/generation/model/implementation/v1.py` op-for-op:
 *   1. Frenet-Serret frames `fi` (`shared/genie3/frames.js`)
 *   2. `V1SingleFeatureNet` -> `si_init [N, c_s]`
 *   3. `V1PairFeatureNet` (outer sum `zi + zj` + static `relpos` + static `cond_template`
 *      + dynamic noisy `template` fused with GPU 4x4 Jacobi `rot_to_quat`) -> `zij_init [N, N, c_z]`
 *   4. `LatentTransformer` (prepend `n_global_token` zero tokens -> 5 blocks of
 *      ReducedIPA + LayerNorm + StructureTransition + Single-to-Pair + TriangleMulOut
 *      + TriangleMulIn + PairTransition -> drop global tokens) -> `si [N, c_s], zij [N, N, c_z]`
 *   5. `SequenceNet` (3-layer ReLU MLP) -> `ai [N, 20]`
 *   6. `StructureNet` (8 layers of Full AF2 IPA + LayerNorm + StructureTransition + BackboneUpdate)
 *      -> `xl_out [N, 3]`
 *
 * All pipelines, bind groups, uniform buffers, and activation buffers are recorded ONCE per length `N`;
 * subsequent `forward()` calls allocate ZERO GPU buffers.
 */

import {
  NUM_ATOM_SYMBOLS,
  NUM_ATOM_TYPES,
  NUM_RESTYPES,
  resolveGenie3Config,
} from '../../shared/genie3/constants.js';
import { sinusoidalEncoding } from '../../shared/genie3/features.js';
import {
  computeConditionalStructureFrames,
  computeNoisyStructureFrames,
  rotToQuatInto,
} from '../../shared/genie3/frames.js';
import { normalizeWeights } from '../../shared/genie3/safetensors.js';
import {
  TRI_GATE_SCATTER_SOURCE,
  recordBackboneUpdate,
  recordFullIpa,
  recordReducedIpa,
} from './kernels_ipa.js';
import { Genie3Ops } from './ops.js';

/** Default scratch byte budget for chunked pair projections & PairTransition (32 MiB). */
export const DEFAULT_PAIR_CHUNK_BYTES = 32 * 1024 * 1024;

/**
 * Fused V1PairFeatureNet dynamic assembly + global token padding:
 * Computes `pExt[(G+i)*M + (G+j), c]` for `0 <= i, j < N` and zeroes global border rows/cols.
 *
 * For each `(i, j)`:
 * - Computes distance `d = sqrt(1e-10 + ||t_i - t_j||^2)` and nearest bin `b in 0..nBins-1`.
 * - Computes relative rotation `R_rel = R_j^T @ R_i` and unit quaternion `q = rot_to_quat(R_rel)`
 *   via 4x4 symmetric Jacobi eigendecomposition of the Davenport K-matrix, with optional sign
 *   canonicalization (first non-zero component `|q[k]| > 1e-7` positive) or external override quats.
 * - Evaluates sparse `linear_template` in O(1) per channel (`W[c, b] + dot(W[c, nBins..nBins+7], tail8)`).
 * - Sums `zi[i, c] + zj[j, c] + zStatic[i, j, c] + templateProj * structPairMask`, masked by `tokenPairMask`.
 */
export const PAIR_INIT_EXT_SOURCE = `
struct P {
  N: u32, G: u32, M: u32, cZ: u32,
  nBins: u32, tmplIn: u32, useOverrideQuat: u32, canonicalQuat: u32,
  distMin: f32, binSize: f32, eps: f32, pad0: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> zizj: array<f32>;
@group(0) @binding(2) var<storage, read> zStatic: array<f32>;
@group(0) @binding(3) var<storage, read> rots: array<f32>;
@group(0) @binding(4) var<storage, read> trans: array<f32>;
@group(0) @binding(5) var<storage, read> pairMasks: array<f32>;
@group(0) @binding(6) var<storage, read> wTmpl: array<f32>;
@group(0) @binding(7) var<storage, read> overrideQuat: array<f32>;
@group(0) @binding(8) var<storage, read_write> pExt: array<f32>;

fn rotToQuat(
  xx: f32, xy: f32, xz: f32,
  yx: f32, yy: f32, yz: f32,
  zx: f32, zy: f32, zz: f32,
  canonical: bool,
) -> vec4<f32> {
  let inv3 = 1.0 / 3.0;
  var k: array<f32, 16>;
  k[0] = (xx + yy + zz) * inv3;
  k[1] = (zy - yz) * inv3;
  k[2] = (xz - zx) * inv3;
  k[3] = (yx - xy) * inv3;
  k[4] = k[1];
  k[5] = (xx - yy - zz) * inv3;
  k[6] = (xy + yx) * inv3;
  k[7] = (xz + zx) * inv3;
  k[8] = k[2];
  k[9] = k[6];
  k[10] = (yy - xx - zz) * inv3;
  k[11] = (yz + zy) * inv3;
  k[12] = k[3];
  k[13] = k[7];
  k[14] = k[11];
  k[15] = (zz - xx - yy) * inv3;

  var v: array<f32, 16>;
  for (var idx = 0u; idx < 16u; idx += 1u) { v[idx] = 0.0; }
  v[0] = 1.0; v[5] = 1.0; v[10] = 1.0; v[15] = 1.0;

  for (var sweep = 0u; sweep < 12u; sweep += 1u) {
    var offNorm = 0.0;
    for (var pp = 0u; pp < 3u; pp += 1u) {
      for (var qq = pp + 1u; qq < 4u; qq += 1u) {
        let apq = k[pp * 4u + qq];
        offNorm += apq * apq;
      }
    }
    if (offNorm < 1e-14) { break; }
    for (var pp = 0u; pp < 3u; pp += 1u) {
      for (var qq = pp + 1u; qq < 4u; qq += 1u) {
        let apq = k[pp * 4u + qq];
        if (abs(apq) < 1e-9) { continue; }
        let app = k[pp * 4u + pp];
        let aqq = k[qq * 4u + qq];
        let tau = (aqq - app) / (2.0 * apq);
        let t = select(
          -1.0 / (-tau + sqrt(1.0 + tau * tau)),
          1.0 / (tau + sqrt(1.0 + tau * tau)),
          tau >= 0.0
        );
        let cs = inverseSqrt(1.0 + t * t);
        let sn = t * cs;
        k[pp * 4u + pp] = app - t * apq;
        k[qq * 4u + qq] = aqq + t * apq;
        k[pp * 4u + qq] = 0.0;
        k[qq * 4u + pp] = 0.0;
        for (var rr = 0u; rr < 4u; rr += 1u) {
          if (rr != pp && rr != qq) {
            let arp = k[rr * 4u + pp];
            let arq = k[rr * 4u + qq];
            let nrp = cs * arp - sn * arq;
            let nrq = sn * arp + cs * arq;
            k[rr * 4u + pp] = nrp;
            k[pp * 4u + rr] = nrp;
            k[rr * 4u + qq] = nrq;
            k[qq * 4u + rr] = nrq;
          }
        }
        for (var rr = 0u; rr < 4u; rr += 1u) {
          let vrp = v[rr * 4u + pp];
          let vrq = v[rr * 4u + qq];
          v[rr * 4u + pp] = cs * vrp - sn * vrq;
          v[rr * 4u + qq] = sn * vrp + cs * vrq;
        }
      }
    }
  }

  var bestCol = 0u;
  var bestVal = k[0];
  for (var col = 1u; col < 4u; col += 1u) {
    let ev = k[col * 4u + col];
    if (ev > bestVal) {
      bestVal = ev;
      bestCol = col;
    }
  }
  var q = vec4<f32>(v[bestCol], v[4u + bestCol], v[8u + bestCol], v[12u + bestCol]);
  if (canonical) {
    var signFlip = 1.0;
    if (abs(q.x) > 1e-7) {
      if (q.x < 0.0) { signFlip = -1.0; }
    } else if (abs(q.y) > 1e-7) {
      if (q.y < 0.0) { signFlip = -1.0; }
    } else if (abs(q.z) > 1e-7) {
      if (q.z < 0.0) { signFlip = -1.0; }
    } else if (abs(q.w) > 1e-7) {
      if (q.w < 0.0) { signFlip = -1.0; }
    }
    q *= signFlip;
  }
  return q;
}

var<workgroup> sharedBin: u32;
var<workgroup> sharedQ: vec4<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let pairIdx = wid.x + wid.y * 65535u;
  let totalPairs = p.M * p.M;
  if (pairIdx >= totalPairs) { return; }
  let mj = pairIdx % p.M;
  let mi = pairIdx / p.M;
  let outBase = pairIdx * p.cZ;

  if (mi < p.G || mj < p.G) {
    for (var c = li; c < p.cZ; c += 64u) {
      pExt[outBase + c] = 0.0;
    }
    return;
  }

  let i = mi - p.G;
  let j = mj - p.G;
  let ij = i * p.N + j;
  let mBase = ij * 5u;
  let tokPairMask = pairMasks[mBase + 0u];
  let structPairMask = pairMasks[mBase + 1u];
  let framePairMask = pairMasks[mBase + 2u];
  let condPairMask = pairMasks[mBase + 3u];
  let condFramePairMask = pairMasks[mBase + 4u];

  // Compute distance bin and relative quaternion ONCE per pair on lane 0 (avoids 64x redundant Jacobi rotToQuat).
  if (li == 0u) {
    let ti = i * 3u;
    let tj = j * 3u;
    let dx = trans[ti + 0u] - trans[tj + 0u];
    let dy = trans[ti + 1u] - trans[tj + 1u];
    let dz = trans[ti + 2u] - trans[tj + 2u];
    let dist = sqrt(p.eps + dx * dx + dy * dy + dz * dz);
    let scaled = (dist - p.distMin) / p.binSize;
    let maxBin = f32(p.nBins - 1u);
    sharedBin = u32(clamp(floor(scaled + 0.5), 0.0, maxBin));

    var q0 = vec4<f32>(0.0);
    if (framePairMask > 0.0) {
      if (p.useOverrideQuat != 0u) {
        let qb = ij * 4u;
        q0 = vec4<f32>(overrideQuat[qb], overrideQuat[qb + 1u], overrideQuat[qb + 2u], overrideQuat[qb + 3u]);
      } else {
        // R_rel = R_j^T @ R_i (exact match to pair/v1.py:253)
        let ri = i * 9u;
        let rj = j * 9u;
        let i00 = rots[ri + 0u]; let i01 = rots[ri + 1u]; let i02 = rots[ri + 2u];
        let i10 = rots[ri + 3u]; let i11 = rots[ri + 4u]; let i12 = rots[ri + 5u];
        let i20 = rots[ri + 6u]; let i21 = rots[ri + 7u]; let i22 = rots[ri + 8u];
        let j00 = rots[rj + 0u]; let j01 = rots[rj + 1u]; let j02 = rots[rj + 2u];
        let j10 = rots[rj + 3u]; let j11 = rots[rj + 4u]; let j12 = rots[rj + 5u];
        let j20 = rots[rj + 6u]; let j21 = rots[rj + 7u]; let j22 = rots[rj + 8u];

        let r00 = j00 * i00 + j10 * i10 + j20 * i20;
        let r01 = j00 * i01 + j10 * i11 + j20 * i21;
        let r02 = j00 * i02 + j10 * i12 + j20 * i22;
        let r10 = j01 * i00 + j11 * i10 + j21 * i20;
        let r11 = j01 * i01 + j11 * i11 + j21 * i21;
        let r12 = j01 * i02 + j11 * i12 + j21 * i22;
        let r20 = j02 * i00 + j12 * i10 + j22 * i20;
        let r21 = j02 * i01 + j12 * i11 + j22 * i21;
        let r22 = j02 * i02 + j12 * i12 + j22 * i22;
        q0 = rotToQuat(r00, r01, r02, r10, r11, r12, r20, r21, r22, p.canonicalQuat != 0u);
      }
    }
    sharedQ = q0 * framePairMask;
  }
  workgroupBarrier();
  let bin = sharedBin;
  let q = sharedQ;

  let ziBase = i * p.cZ;
  let zjBase = (p.N + j) * p.cZ;
  let zsBase = ij * p.cZ;
  for (var c = li; c < p.cZ; c += 64u) {
    let wBase = c * p.tmplIn;
    let tmpl = wTmpl[wBase + bin]
      + wTmpl[wBase + p.nBins + 0u] * q.x
      + wTmpl[wBase + p.nBins + 1u] * q.y
      + wTmpl[wBase + p.nBins + 2u] * q.z
      + wTmpl[wBase + p.nBins + 3u] * q.w
      + wTmpl[wBase + p.nBins + 4u] * structPairMask
      + wTmpl[wBase + p.nBins + 5u] * framePairMask
      + wTmpl[wBase + p.nBins + 6u] * condPairMask
      + wTmpl[wBase + p.nBins + 7u] * condFramePairMask;
    let val = (zizj[ziBase + c] + zizj[zjBase + c] + zStatic[zsBase + c] + tmpl * structPairMask) * tokPairMask;
    pExt[outBase + c] = val;
  }
}
`;

/**
 * List every weight tensor key required by `V1Denoiser` for a given resolved config.
 * Useful for verifying 100% coverage against exported safetensors manifests.
 */
export function expectedWeightKeys(rawCfg = null) {
  const cfg = resolveGenie3Config(rawCfg);
  const pt = cfg.pair_transform_net;
  const st = cfg.structure_net;
  const keys = [
    'single_feature_net.linear.weight',
    'pair_feature_net.linear_zi.weight',
    'pair_feature_net.linear_zj.weight',
    'pair_feature_net.relpos_embedder.linear_relpos.weight',
    'pair_feature_net.linear_template.weight',
    'pair_feature_net.linear_cond_template.weight',
  ];

  for (let b = 0; b < pt.n_pair_transform_layer; b += 1) {
    const p = `pair_transform_net.net.${b}`;
    for (const tri of ['tri_mul_out', 'tri_mul_in']) {
      for (const lin of ['linear_a_p', 'linear_a_g', 'linear_b_p', 'linear_b_g', 'linear_g', 'linear_z']) {
        keys.push(`${p}.${tri}.${lin}.weight`, `${p}.${tri}.${lin}.bias`);
      }
      keys.push(`${p}.${tri}.layer_norm_in.weight`, `${p}.${tri}.layer_norm_in.bias`);
      keys.push(`${p}.${tri}.layer_norm_out.weight`, `${p}.${tri}.layer_norm_out.bias`);
    }
    keys.push(
      `${p}.pair_transition.layer_norm.weight`, `${p}.pair_transition.layer_norm.bias`,
      `${p}.pair_transition.linear_1.weight`, `${p}.pair_transition.linear_1.bias`,
      `${p}.pair_transition.linear_2.weight`, `${p}.pair_transition.linear_2.bias`,
      `${p}.ipa.linear_q.weight`, `${p}.ipa.linear_q.bias`,
      `${p}.ipa.linear_kv.weight`, `${p}.ipa.linear_kv.bias`,
      `${p}.ipa.linear_b.weight`, `${p}.ipa.linear_b.bias`,
      `${p}.ipa.linear_out.weight`, `${p}.ipa.linear_out.bias`,
      `${p}.ipa_layer_norm.weight`, `${p}.ipa_layer_norm.bias`,
    );
    for (let l = 0; l < pt.n_structure_transition_layer; l += 1) {
      for (const lin of ['linear_1', 'linear_2', 'linear_3']) {
        keys.push(`${p}.ipa_transition.layers.${l}.${lin}.weight`, `${p}.ipa_transition.layers.${l}.${lin}.bias`);
      }
    }
    keys.push(
      `${p}.ipa_transition.layer_norm.weight`, `${p}.ipa_transition.layer_norm.bias`,
      `${p}.linear_pi.weight`, `${p}.linear_pj.weight`, `${p}.linear_p.weight`,
    );
  }

  keys.push('sequence_net.net.0.weight', 'sequence_net.net.2.weight', 'sequence_net.net.4.weight');

  for (let l = 0; l < st.n_structure_layer; l += 1) {
    const p = `structure_net.net.${l}`;
    keys.push(
      `${p}.ipa.head_weights`,
      `${p}.ipa.linear_q.weight`, `${p}.ipa.linear_q.bias`,
      `${p}.ipa.linear_kv.weight`, `${p}.ipa.linear_kv.bias`,
      `${p}.ipa.linear_q_points.weight`, `${p}.ipa.linear_q_points.bias`,
      `${p}.ipa.linear_kv_points.weight`, `${p}.ipa.linear_kv_points.bias`,
      `${p}.ipa.linear_b.weight`, `${p}.ipa.linear_b.bias`,
      `${p}.ipa.linear_out.weight`, `${p}.ipa.linear_out.bias`,
      `${p}.ipa_layer_norm.weight`, `${p}.ipa_layer_norm.bias`,
    );
    for (let k = 0; k < st.n_structure_transition_layer; k += 1) {
      for (const lin of ['linear_1', 'linear_2', 'linear_3']) {
        keys.push(`${p}.transition.layers.${k}.${lin}.weight`, `${p}.transition.layers.${k}.${lin}.bias`);
      }
    }
    keys.push(
      `${p}.transition.layer_norm.weight`, `${p}.transition.layer_norm.bias`,
      `${p}.bb_update.linear.weight`, `${p}.bb_update.linear.bias`,
    );
  }

  return keys;
}

/**
 * Estimate activation + weight GPU buffer budget (in bytes) for sequence length `N`.
 */
export function estimateVramBudget(N, {
  cfg: rawCfg = null,
  half = true,
  pairChunkBytes = DEFAULT_PAIR_CHUNK_BYTES,
} = {}) {
  const cfg = resolveGenie3Config(rawCfg);
  const cS = cfg.c_s;
  const cZ = cfg.c_z;
  const G = cfg.pair_transform_net.n_global_token;
  const M = N + G;
  const cH = cfg.pair_transform_net.c_hidden_ipa;
  const H = cfg.pair_transform_net.n_head_ipa;
  const cMul = cfg.pair_transform_net.c_hidden_mul;
  const pairExp = cfg.pair_transform_net.pair_transition_n;
  const nQkPts = cfg.structure_net.n_qk_point;
  const nVPts = cfg.structure_net.n_v_point;

  const rowFloats = Math.max(4 * cMul, pairExp * cZ);
  const R = Math.max(1, Math.min(M * M, Math.floor(pairChunkBytes / (rowFloats * 4))));

  const nn = N * N;
  const mm = M * M;
  const concatMax = Math.max(H * (cH + cZ), H * (cH + 4 * nVPts + cZ));

  const activationFloats =
    N * cfg.single_feature_net.c_s_input + // singleFeat
    N * cS * 3 +                           // sInit, s, sTmp
    M * cS * 3 +                           // sExt, sExtTmp, sExtTmp2
    M * cZ * 4 +                           // pi, pj, piP, pjP
    N * cZ * 2 +                           // zi, zj
    nn * cZ * 2 +                          // zStatic, z
    mm * cZ +                              // pExt
    nn * 5 +                               // pairMasks
    M + mm +                               // maskExt, pairMaskExt
    N +                                    // frameMask
    N * 9 + N * 3 +                        // rots, trans
    cMul * mm * 3 +                        // triAT, triBT, triX
    mm * cZ +                              // triG
    R * cZ + R * cMul + R * rowFloats +    // zLnChunk, xLnChunk, pairScratch
    M * H * cH + M * 2 * H * cH + mm * H + // ipaQ, ipaKv, ipaB
    H * mm +                               // ipaAttn
    M * concatMax +                        // ipaConcat
    N * 3 * H * (nQkPts + nVPts) +         // ipaPtsLocal
    N * H * nQkPts * 3 +                   // ipaQPts
    N * H * (nQkPts + nVPts) * 3 +         // ipaKvPts
    N * 6 +                                // bbParams
    N * cfg.sequence_net.c_hidden * 2 +    // seqH1, seqH2
    N * NUM_RESTYPES;                      // aiOut

  // Exact V1Denoiser parameter count: 23,269,484 params
  const weightBytes = half ? Math.round(44.6 * 1024 * 1024) : Math.round(88.8 * 1024 * 1024);
  const activationBytes = activationFloats * 4;
  return {
    N,
    M,
    pairChunkRows: R,
    pairChunks: Math.ceil(mm / R),
    weightBytes,
    activationBytes,
    totalBytes: weightBytes + activationBytes,
  };
}

function concatArrays(...arrs) {
  let total = 0;
  for (const a of arrs) total += a.length;
  const out = new Float32Array(total);
  let off = 0;
  for (const a of arrs) {
    out.set(a, off);
    off += a.length;
  }
  return out;
}

function canonicalizeQuatInPlace(arr, off) {
  for (let k = 0; k < 4; k += 1) {
    const v = arr[off + k];
    if (Math.abs(v) > 1e-7) {
      if (v < 0) {
        arr[off + 0] = -arr[off + 0];
        arr[off + 1] = -arr[off + 1];
        arr[off + 2] = -arr[off + 2];
        arr[off + 3] = -arr[off + 3];
      }
      return;
    }
  }
}

/** Build static RelposEmbedder one-hot features `[N * N, relposDims]` on CPU (once per batch). */
function buildRelposFeatures(batch, maxRelIdx, maxRelChain) {
  const N = batch.token_mask.length;
  const posBins = 2 * maxRelIdx + 2;
  const chainBins = 2 * maxRelChain + 2;
  const dim = posBins + posBins + 1 + chainBins;
  const out = new Float32Array(N * N * dim);

  const asym = batch.asym_id;
  const sym = batch.sym_id;
  const entity = batch.entity_id;
  const tokIdx = batch.token_index;
  const resIdx = batch.residue_index;

  for (let i = 0; i < N; i += 1) {
    const ai = asym[i], si = sym[i], ei = entity[i], ti = tokIdx[i], ri = resIdx[i];
    for (let j = 0; j < N; j += 1) {
      const base = (i * N + j) * dim;
      const sameChain = ai === asym[j];
      const sameRes = ri === resIdx[j];

      // 1. rel_pos (residue_index, condition = same_chain)
      let pBin;
      if (sameChain) {
        pBin = Math.max(0, Math.min(2 * maxRelIdx, (ri - resIdx[j]) + maxRelIdx));
      } else {
        pBin = 2 * maxRelIdx + 1;
      }
      out[base + pBin] = 1.0;

      // 2. rel_token (token_index, condition = same_chain && same_res)
      let tBin;
      if (sameChain && sameRes) {
        tBin = Math.max(0, Math.min(2 * maxRelIdx, (ti - tokIdx[j]) + maxRelIdx));
      } else {
        tBin = 2 * maxRelIdx + 1;
      }
      out[base + posBins + tBin] = 1.0;

      // 3. same_entity
      out[base + 2 * posBins] = ei === entity[j] ? 1.0 : 0.0;

      // 4. rel_chain (sym_id, condition = !same_chain)
      let cBin;
      if (!sameChain) {
        cBin = Math.max(0, Math.min(2 * maxRelChain, (si - sym[j]) + maxRelChain));
      } else {
        cBin = 2 * maxRelChain + 1;
      }
      out[base + 2 * posBins + 1 + cBin] = 1.0;
    }
  }
  return { data: out, dim };
}

/** Build static pair masks `[N * N, 5]` = [tokPair, structPair, framePair, condPair, condFramePair]. */
function buildPairMasks(batch) {
  const N = batch.token_mask.length;
  const out = new Float32Array(N * N * 5);
  let maxGroup = 0;
  for (let i = 0; i < N; i += 1) {
    if (batch.cond_group[i] > maxGroup) maxGroup = batch.cond_group[i];
  }
  const hasCond = maxGroup > 0;
  for (let i = 0; i < N; i += 1) {
    const tmI = batch.token_mask[i];
    const smI = batch.token_struct_mask[i];
    const fmI = batch.token_struct_frame_mask[i];
    const cgI = batch.cond_group[i];
    const csI = batch.cond_struct_mask[i];
    const cfI = batch.cond_struct_frame_mask[i];
    for (let j = 0; j < N; j += 1) {
      const base = (i * N + j) * 5;
      out[base + 0] = tmI * batch.token_mask[j];
      out[base + 1] = smI * batch.token_struct_mask[j];
      out[base + 2] = fmI * batch.token_struct_frame_mask[j];
      let condPair = 0.0;
      if (hasCond && cgI > 0 && cgI === batch.cond_group[j]) {
        condPair = csI * batch.cond_struct_mask[j];
      }
      out[base + 3] = condPair;
      out[base + 4] = cfI * batch.cond_struct_frame_mask[j] * condPair;
    }
  }
  return { pairMasks: out, hasCond };
}

/** Build conditional template features `[N * N, condBins + 6]` (only called when `hasCond`). */
function buildCondTemplateFeatures(batch, pairMasks, pfCfg, canonicalQuat = true) {
  const N = batch.token_mask.length;
  const nBins = pfCfg.cond_template_dist_no_bins;
  const binMin = pfCfg.cond_template_dist_bin_min;
  const binMax = pfCfg.cond_template_dist_bin_max;
  const binSize = (binMax - binMin) / (nBins - 1);
  const dim = nBins + 6;
  const out = new Float32Array(N * N * dim);
  const condFi = computeConditionalStructureFrames(batch);
  const rRel = new Float32Array(9);
  const qTmp = new Float32Array(4);

  for (let i = 0; i < N; i += 1) {
    const ti = i * 3;
    const ri = i * 9;
    for (let j = 0; j < N; j += 1) {
      const mBase = (i * N + j) * 5;
      const condPair = pairMasks[mBase + 3];
      const condFramePair = pairMasks[mBase + 4];
      const base = (i * N + j) * dim;
      const tj = j * 3;
      const dx = condFi.trans[ti] - condFi.trans[tj];
      const dy = condFi.trans[ti + 1] - condFi.trans[tj + 1];
      const dz = condFi.trans[ti + 2] - condFi.trans[tj + 2];
      const dist = Math.sqrt(1e-10 + dx * dx + dy * dy + dz * dz);
      const bin = Math.max(0, Math.min(nBins - 1, Math.floor((dist - binMin) / binSize + 0.5)));
      out[base + bin] = 1.0;

      if (condFramePair > 0) {
        const rj = j * 9;
        // R_j^T @ R_i
        for (let a = 0; a < 3; a += 1) {
          for (let b = 0; b < 3; b += 1) {
            rRel[a * 3 + b] =
              condFi.rots[rj + 0 * 3 + a] * condFi.rots[ri + 0 * 3 + b] +
              condFi.rots[rj + 1 * 3 + a] * condFi.rots[ri + 1 * 3 + b] +
              condFi.rots[rj + 2 * 3 + a] * condFi.rots[ri + 2 * 3 + b];
          }
        }
        rotToQuatInto(rRel, 0, qTmp, 0);
        if (canonicalQuat) canonicalizeQuatInPlace(qTmp, 0);
        out[base + nBins + 0] = qTmp[0] * condFramePair;
        out[base + nBins + 1] = qTmp[1] * condFramePair;
        out[base + nBins + 2] = qTmp[2] * condFramePair;
        out[base + nBins + 3] = qTmp[3] * condFramePair;
      }
      out[base + nBins + 4] = condPair;
      out[base + nBins + 5] = condFramePair;
    }
  }
  return { data: out, dim };
}

/** Assemble V1SingleFeatureNet input matrix `[N, c_s_input]` on CPU (fast: ~0.05 ms). */
export function buildSingleInputFeatures(batch, tNorm, sfCfg, out = null) {
  const N = batch.token_mask.length;
  const cIn = sfCfg.c_s_input;
  const res = out && out.length === N * cIn ? out : new Float32Array(N * cIn);

  const tokEmb = sinusoidalEncoding(batch.token_index, sfCfg.max_n_res, sfCfg.c_pos_emb);
  const resEmb = sinusoidalEncoding(batch.residue_index, sfCfg.max_n_res, sfCfg.c_pos_emb);
  const chEmb = sinusoidalEncoding(batch.asym_id, sfCfg.max_n_chain, sfCfg.c_chain_emb);
  // Match PyTorch `(t * n_timestep).to(int)` (truncation of FP32 product)
  const tInt = Math.trunc(Math.fround(Math.fround(tNorm) * sfCfg.n_timestep));
  const tSingle = sinusoidalEncoding([tInt], sfCfg.n_timestep, sfCfg.c_timestep_emb);

  const pDim = sfCfg.c_pos_emb;
  const cDim = sfCfg.c_chain_emb;
  const tDim = sfCfg.c_timestep_emb;

  for (let i = 0; i < N; i += 1) {
    let off = i * cIn;
    res.set(tokEmb.subarray(i * pDim, (i + 1) * pDim), off); off += pDim;
    res.set(resEmb.subarray(i * pDim, (i + 1) * pDim), off); off += pDim;
    res.set(chEmb.subarray(i * cDim, (i + 1) * cDim), off); off += cDim;
    res.set(tSingle, off); off += tDim;

    const cSeq = batch.cond_seq_mask[i];
    const rBase = i * NUM_RESTYPES;
    for (let k = 0; k < NUM_RESTYPES; k += 1) {
      res[off + k] = batch.gt_restype[rBase + k] * cSeq;
    }
    off += NUM_RESTYPES;

    const atBase = i * NUM_ATOM_TYPES;
    for (let k = 0; k < NUM_ATOM_TYPES; k += 1) res[off + k] = batch.atom_type[atBase + k];
    off += NUM_ATOM_TYPES;

    const asBase = i * NUM_ATOM_SYMBOLS;
    for (let k = 0; k < NUM_ATOM_SYMBOLS; k += 1) res[off + k] = batch.atom_symbol[asBase + k];
    off += NUM_ATOM_SYMBOLS;

    res[off++] = batch.is_atomized[i];
    res[off++] = batch.token_mask[i];
    res[off++] = batch.token_struct_mask[i];
    res[off++] = batch.token_struct_frame_mask[i];
    res[off++] = batch.cond_group[i] > 0 ? 1.0 : 0.0;
    res[off++] = cSeq;
    res[off++] = batch.cond_struct_mask[i];
    res[off++] = batch.cond_struct_frame_mask[i];
    res[off++] = batch.cond_interface_mask[i];
  }
  return res;
}

export class Genie3DenoiserGpu {
  /**
   * @param {GPUDevice} device
   * @param {object} weights Normalized or raw `{ tensors, config }` / `{ [key]: { data, shape } }`
   * @param {object} [cfg] Optional config override
   * @param {{
   *   half?: boolean,
   *   pairChunkBytes?: number,
   *   canonicalQuat?: boolean,
   *   deferUpload?: boolean,
   * }} [opts]
   */
  constructor(device, weights, cfg = null, opts = {}) {
    const norm = normalizeWeights(weights);
    this.cfg = resolveGenie3Config(cfg ?? norm.config);
    this.ops = new Genie3Ops(device, opts);
    this.device = device;
    this.half = opts.half ?? false;
    this.pairChunkBytes = opts.pairChunkBytes ?? DEFAULT_PAIR_CHUNK_BYTES;
    this.canonicalQuat = opts.canonicalQuat ?? true;
    this.N = 0;
    this.M = 0;
    this.w = null;
    this.buf = null;
    this.cachedBatchRef = null;
    this.hasOverrideQuatProgram = false;
    this._rawTensors = norm.tensors;
    if (!opts.deferUpload) {
      this.uploadWeights(norm.tensors);
      this._rawTensors = null;
    }
  }

  /**
   * Yielding weight upload generator so browsers stay responsive during initialization.
   */
  *uploadSteps(tensors = this._rawTensors) {
    const ops = this.ops;
    const cfg = this.cfg;
    const pt = cfg.pair_transform_net;
    const st = cfg.structure_net;
    const cS = cfg.c_s;
    const cZ = cfg.c_z;
    const half = this.half;

    const get = (k) => {
      const entry = tensors[k];
      if (!entry) throw new Error(`Genie3DenoiserGpu: missing weight "${k}"`);
      return entry.data ?? entry;
    };
    const mat = (k, kDim = null) => ops.weight(k, get(k), half, kDim);
    const vec = (k) => ops.weight(k, get(k), false);

    const w = {
      singleLinear: mat('single_feature_net.linear.weight', cfg.single_feature_net.c_s_input),
      pairZi: mat('pair_feature_net.linear_zi.weight', cS),
      pairZj: mat('pair_feature_net.linear_zj.weight', cS),
      // Relpos (139), template (73), cond_template (71) have odd input dim -> stored in FP32 (~145 KB total)
      pairRelpos: vec('pair_feature_net.relpos_embedder.linear_relpos.weight'),
      pairTemplate: vec('pair_feature_net.linear_template.weight'),
      pairCondTemplate: vec('pair_feature_net.linear_cond_template.weight'),
      blocks: [],
      seq: {
        w0: mat('sequence_net.net.0.weight', cS),
        w1: mat('sequence_net.net.2.weight', cfg.sequence_net.c_hidden),
        w2: mat('sequence_net.net.4.weight', cfg.sequence_net.c_hidden),
      },
      structLayers: [],
    };
    yield;

    const uploadTri = (prefix) => {
      // Stack [a_p; a_g; b_p; b_g] into a single [4 * c_hidden_mul, c_z] weight + bias for fused projection
      const abW = concatArrays(
        get(`${prefix}.linear_a_p.weight`),
        get(`${prefix}.linear_a_g.weight`),
        get(`${prefix}.linear_b_p.weight`),
        get(`${prefix}.linear_b_g.weight`),
      );
      const abB = concatArrays(
        get(`${prefix}.linear_a_p.bias`),
        get(`${prefix}.linear_a_g.bias`),
        get(`${prefix}.linear_b_p.bias`),
        get(`${prefix}.linear_b_g.bias`),
      );
      return {
        lnInW: vec(`${prefix}.layer_norm_in.weight`),
        lnInB: vec(`${prefix}.layer_norm_in.bias`),
        abW: ops.weight(`${prefix}.ab.weight`, abW, half, cZ),
        abB: ops.weight(`${prefix}.ab.bias`, abB, false),
        gW: mat(`${prefix}.linear_g.weight`, cZ),
        gB: vec(`${prefix}.linear_g.bias`),
        lnOutW: vec(`${prefix}.layer_norm_out.weight`),
        lnOutB: vec(`${prefix}.layer_norm_out.bias`),
        zW: mat(`${prefix}.linear_z.weight`, pt.c_hidden_mul),
        zB: vec(`${prefix}.linear_z.bias`),
      };
    };

    const uploadTransition = (prefix, nLayers) => ({
      layers: Array.from({ length: nLayers }, (_, l) => ({
        w1: mat(`${prefix}.layers.${l}.linear_1.weight`, cS),
        b1: vec(`${prefix}.layers.${l}.linear_1.bias`),
        w2: mat(`${prefix}.layers.${l}.linear_2.weight`, cS),
        b2: vec(`${prefix}.layers.${l}.linear_2.bias`),
        w3: mat(`${prefix}.layers.${l}.linear_3.weight`, cS),
        b3: vec(`${prefix}.layers.${l}.linear_3.bias`),
      })),
      lnW: vec(`${prefix}.layer_norm.weight`),
      lnB: vec(`${prefix}.layer_norm.bias`),
    });

    for (let b = 0; b < pt.n_pair_transform_layer; b += 1) {
      const p = `pair_transform_net.net.${b}`;
      w.blocks.push({
        ipa: {
          qW: mat(`${p}.ipa.linear_q.weight`, cS),
          qB: vec(`${p}.ipa.linear_q.bias`),
          kvW: mat(`${p}.ipa.linear_kv.weight`, cS),
          kvB: vec(`${p}.ipa.linear_kv.bias`),
          bW: mat(`${p}.ipa.linear_b.weight`, cZ),
          bB: vec(`${p}.ipa.linear_b.bias`),
          outW: mat(`${p}.ipa.linear_out.weight`, pt.n_head_ipa * (pt.c_hidden_ipa + cZ)),
          outB: vec(`${p}.ipa.linear_out.bias`),
        },
        ipaLnW: vec(`${p}.ipa_layer_norm.weight`),
        ipaLnB: vec(`${p}.ipa_layer_norm.bias`),
        ipaTransition: uploadTransition(`${p}.ipa_transition`, pt.n_structure_transition_layer),
        linPi: mat(`${p}.linear_pi.weight`, cS),
        linPj: mat(`${p}.linear_pj.weight`, cS),
        linP: mat(`${p}.linear_p.weight`, cZ),
        triOut: uploadTri(`${p}.tri_mul_out`),
        triIn: uploadTri(`${p}.tri_mul_in`),
        pairTrans: {
          lnW: vec(`${p}.pair_transition.layer_norm.weight`),
          lnB: vec(`${p}.pair_transition.layer_norm.bias`),
          w1: mat(`${p}.pair_transition.linear_1.weight`, cZ),
          b1: vec(`${p}.pair_transition.linear_1.bias`),
          w2: mat(`${p}.pair_transition.linear_2.weight`, pt.pair_transition_n * cZ),
          b2: vec(`${p}.pair_transition.linear_2.bias`),
        },
      });
      yield;
    }

    for (let l = 0; l < st.n_structure_layer; l += 1) {
      const p = `structure_net.net.${l}`;
      const concatDim = st.n_head_ipa * (st.c_hidden_ipa + 4 * st.n_v_point + cZ);
      w.structLayers.push({
        ipa: {
          headWeights: vec(`${p}.ipa.head_weights`),
          qW: mat(`${p}.ipa.linear_q.weight`, cS),
          qB: vec(`${p}.ipa.linear_q.bias`),
          kvW: mat(`${p}.ipa.linear_kv.weight`, cS),
          kvB: vec(`${p}.ipa.linear_kv.bias`),
          qPtsW: mat(`${p}.ipa.linear_q_points.weight`, cS),
          qPtsB: vec(`${p}.ipa.linear_q_points.bias`),
          kvPtsW: mat(`${p}.ipa.linear_kv_points.weight`, cS),
          kvPtsB: vec(`${p}.ipa.linear_kv_points.bias`),
          bW: mat(`${p}.ipa.linear_b.weight`, cZ),
          bB: vec(`${p}.ipa.linear_b.bias`),
          outW: mat(`${p}.ipa.linear_out.weight`, concatDim),
          outB: vec(`${p}.ipa.linear_out.bias`),
        },
        ipaLnW: vec(`${p}.ipa_layer_norm.weight`),
        ipaLnB: vec(`${p}.ipa_layer_norm.bias`),
        transition: uploadTransition(`${p}.transition`, st.n_structure_transition_layer),
        bbUpdate: {
          w: mat(`${p}.bb_update.linear.weight`, cS),
          b: vec(`${p}.bb_update.linear.bias`),
        },
      });
      yield;
    }

    this.w = w;
    this._rawTensors = null;
  }

  uploadWeights(tensors = this._rawTensors) {
    for (const _ of this.uploadSteps(tensors)) {
      // synchronous consume
    }
    return this.w;
  }

  async _recordStructureTransition(sBuf, tmp1, tmp2, trW, rows, cS) {
    const ops = this.ops;
    const steps = [];
    for (const layer of trW.layers) {
      steps.push(
        await ops.linear({ x: sBuf, w: layer.w1, bias: layer.b1, y: tmp1, rows, inC: cS, outC: cS, act: 'relu', label: 'st.l1' }),
        await ops.linear({ x: tmp1, w: layer.w2, bias: layer.b2, y: tmp2, rows, inC: cS, outC: cS, act: 'relu', label: 'st.l2' }),
        await ops.linear({ x: tmp2, w: layer.w3, bias: layer.b3, y: sBuf, rows, inC: cS, outC: cS, accumulate: true, label: 'st.l3' }),
      );
    }
    steps.push(await ops.layerNorm({ x: sBuf, y: sBuf, rows, width: cS, w: trW.lnW, b: trW.lnB, label: 'st.ln' }));
    return steps;
  }

  async _recordTriangle(triW, incoming) {
    const ops = this.ops;
    const B = this.buf;
    const M = this.M;
    const mm = M * M;
    const cZ = this.cfg.c_z;
    const cMul = this.cfg.pair_transform_net.c_hidden_mul;
    const steps = [];

    for (const [r0, rows] of this.pairChunks) {
      steps.push(
        await ops.layerNorm({
          x: B.pExt, y: B.zLnChunk, rows, width: cZ,
          w: triW.lnInW, b: triW.lnInB, xOff: r0 * cZ, yOff: 0, label: 'tri.lnIn',
        }),
        await ops.linear({
          x: B.zLnChunk, w: triW.abW, bias: triW.abB, y: B.pairScratch,
          rows, inC: cZ, outC: 4 * cMul, label: 'tri.ab',
        }),
        await ops.step(
          'tri-gate-scatter',
          TRI_GATE_SCATTER_SOURCE,
          Math.ceil((rows * cMul) / 256),
          [B.pairScratch, B.pairMaskExt, B.triAT, B.triBT],
          new Uint32Array([rows, cMul, mm, r0]),
          'tri.scatter',
        ),
        await ops.linear({
          x: B.zLnChunk, w: triW.gW, bias: triW.gB, y: B.triG,
          rows, inC: cZ, outC: cZ, act: 'sigmoid', yOff: r0 * cZ, label: 'tri.g',
        }),
      );
    }

    // Batched contraction over cMul channels:
    // Outgoing: x[c, i, j] = sum_k aT[c, i, k] * bT[c, j, k]
    // Incoming: x[c, i, j] = sum_k aT[c, k, i] * bT[c, k, j] (transpose via stride swap!)
    const opA = incoming ? { s0: mm, m: 1, k: M } : { s0: mm, m: M, k: 1 };
    const opB = incoming ? { s0: mm, n: 1, k: M } : { s0: mm, n: M, k: 1 };
    steps.push(
      await ops.gemm({
        m: M, n: M, k: M,
        a: B.triAT, b: B.triBT, c: B.triX,
        batches: [cMul, 1],
        A: opA, B: opB, C: { s0: mm, m: M, n: 1 },
        label: incoming ? 'tri.in.matmul' : 'tri.out.matmul',
      }),
    );

    for (const [r0, rows] of this.pairChunks) {
      steps.push(
        await ops.transposeLayerNorm({
          x: B.triX, y: B.xLnChunk, rows, width: cMul, nn: mm, r0,
          w: triW.lnOutW, b: triW.lnOutB, label: 'tri.lnOut',
        }),
        await ops.linear({
          x: B.xLnChunk, w: triW.zW, bias: triW.zB, y: B.pExt,
          rows, inC: cMul, outC: cZ, yOff: r0 * cZ,
          accumulate: true,
          gate: { sc: B.triG, off: r0 * cZ, stride: cZ, sigmoid: false },
          label: 'tri.outProj',
        }),
      );
    }

    return steps;
  }

  /**
   * Allocate activation buffers and record all compute programs for sequence length `N`.
   * Subsequent calls with the same `N` (and `useOverrideQuat` flag) are no-ops.
   */
  async record(N, useOverrideQuat = false) {
    if (this.N === N && this.hasOverrideQuatProgram === useOverrideQuat) return;
    try {
      await this._record(N, useOverrideQuat);
    } catch (err) {
      this.N = 0;
      throw err;
    }
  }

  async _record(N, useOverrideQuat) {
    const ops = this.ops;
    ops.releaseActivations();
    this.N = N;
    this.hasOverrideQuatProgram = useOverrideQuat;
    this.cachedBatchRef = null;

    const cfg = this.cfg;
    const sf = cfg.single_feature_net;
    const pf = cfg.pair_feature_net;
    const pt = cfg.pair_transform_net;
    const sq = cfg.sequence_net;
    const st = cfg.structure_net;

    const cS = cfg.c_s;
    const cZ = cfg.c_z;
    const G = pt.n_global_token;
    const M = N + G;
    this.M = M;
    const nn = N * N;
    const mm = M * M;
    const cMul = pt.c_hidden_mul;
    const pairExp = pt.pair_transition_n;
    const H = Math.max(pt.n_head_ipa, st.n_head_ipa);
    const cH = Math.max(pt.c_hidden_ipa, st.c_hidden_ipa);
    const nQkPts = st.n_qk_point;
    const nVPts = st.n_v_point;

    const rowFloats = Math.max(4 * cMul, pairExp * cZ);
    const R = Math.max(1, Math.min(mm, Math.floor(this.pairChunkBytes / (rowFloats * 4))));
    const pairChunks = [];
    for (let r0 = 0; r0 < mm; r0 += R) pairChunks.push([r0, Math.min(R, mm - r0)]);
    this.pairChunks = pairChunks;

    const relposDim = 2 * (2 * pf.max_relative_index + 2) + 1 + (2 * pf.max_relative_chain + 2);
    const condTmplDim = pf.cond_template_dist_no_bins + 6;
    const concatMax = Math.max(
      pt.n_head_ipa * (pt.c_hidden_ipa + cZ),
      st.n_head_ipa * (st.c_hidden_ipa + 4 * nVPts + cZ),
    );

    const b = (label, floats) => ops.buffer(label, floats);
    const B = (this.buf = {
      singleFeat: b('singleFeat', N * sf.c_s_input),
      tokenMask: b('tokenMask', N),
      frameMask: b('frameMask', N),
      maskExt: b('maskExt', M),
      pairMaskExt: b('pairMaskExt', mm),
      pairMasks: b('pairMasks', nn * 5),
      rots: b('rots', N * 9),
      trans: b('trans', N * 3),
      overrideQuat: b('overrideQuat', useOverrideQuat ? nn * 4 : 4),
      relposFeat: b('relposFeat', nn * relposDim),
      condTmplFeat: b('condTmplFeat', nn * condTmplDim),

      sInit: b('sInit', N * cS),
      s: b('s', N * cS),
      sTmp1: b('sTmp1', N * cS),
      sTmp2: b('sTmp2', N * cS),

      sExt: b('sExt', M * cS),
      sExtTmp1: b('sExtTmp1', M * cS),
      sExtTmp2: b('sExtTmp2', M * cS),
      pi: b('pi', M * cZ),
      pj: b('pj', M * cZ),
      piP: b('piP', M * cZ),
      pjP: b('pjP', M * cZ),

      zizj: b('zizj', 2 * N * cZ),
      zStatic: b('zStatic', nn * cZ),
      z: b('z', nn * cZ),
      pExt: b('pExt', mm * cZ),

      triAT: b('triAT', cMul * mm),
      triBT: b('triBT', cMul * mm),
      triX: b('triX', cMul * mm),
      triG: b('triG', mm * cZ),
      zLnChunk: b('zLnChunk', R * cZ),
      xLnChunk: b('xLnChunk', R * cMul),
      pairScratch: b('pairScratch', R * rowFloats),

      ipaQ: b('ipaQ', M * H * cH),
      ipaKv: b('ipaKv', M * 2 * H * cH),
      ipaB: b('ipaB', mm * H),
      ipaAttn: b('ipaAttn', H * mm),
      ipaConcat: b('ipaConcat', M * concatMax),
      ipaPtsLocal: b('ipaPtsLocal', N * 3 * st.n_head_ipa * (nQkPts + nVPts)),
      ipaQPts: b('ipaQPts', N * st.n_head_ipa * nQkPts * 3),
      ipaKvPts: b('ipaKvPts', N * st.n_head_ipa * (nQkPts + nVPts) * 3),
      bbParams: b('bbParams', N * 6),

      seqH1: b('seqH1', N * sq.c_hidden),
      seqH2: b('seqH2', N * sq.c_hidden),
      aiOut: b('aiOut', N * NUM_RESTYPES),
    });

    this.singleFeatHost = new Float32Array(N * sf.c_s_input);

    // --- Static pair embedding program (run once per batch) ---
    this.relposProgram = ops.program([
      await ops.linear({
        x: B.relposFeat, w: this.w.pairRelpos, y: B.zStatic,
        rows: nn, inC: relposDim, outC: cZ, label: 'pair.relpos',
      }),
    ]);

    this.condTemplateProgram = ops.program([
      await ops.linear({
        x: B.condTmplFeat, w: this.w.pairCondTemplate, y: B.z,
        rows: nn, inC: condTmplDim, outC: cZ, label: 'pair.condTmpl',
      }),
      await ops.map(
        'add-cond-tmpl',
        nn * cZ,
        [{ buffer: B.zStatic, write: true }, { buffer: B.z }, { buffer: B.pairMasks }],
        `let ij = i / U(0u);
  let m = b2[ij * 5u + 3u];
  b0[i] = b0[i] + b1[i] * m;`,
        [cZ],
      ),
    ]);

    // --- Embed program: single_feature_net + pair_feature_net + global token pad ---
    const uPairInit = new Uint32Array(12);
    const fPairInit = new Float32Array(uPairInit.buffer);
    const tmplBins = pf.template_dist_no_bins;
    const binSize = (pf.template_dist_bin_max - pf.template_dist_bin_min) / (tmplBins - 1);
    uPairInit.set([
      N, G, M, cZ,
      tmplBins, tmplBins + 8, useOverrideQuat ? 1 : 0, this.canonicalQuat ? 1 : 0,
    ]);
    fPairInit[8] = pf.template_dist_bin_min;
    fPairInit[9] = binSize;
    fPairInit[10] = 1e-10;

    this.embedProgram = ops.program([
      await ops.linear({
        x: B.singleFeat, w: this.w.singleLinear, y: B.sInit,
        rows: N, inC: sf.c_s_input, outC: cS, label: 'single.linear',
      }),
      await ops.map(
        'mask-single-pad-ext',
        M * cS,
        [{ buffer: B.sInit, write: true }, { buffer: B.sExt, write: true }, { buffer: B.tokenMask }],
        `let cS = U(0u);
  let G = U(1u);
  let tok = i / cS;
  let ch = i % cS;
  if (tok < G) {
    b1[i] = 0.0;
  } else {
    let realIdx = (tok - G) * cS + ch;
    let v = b0[realIdx] * b2[tok - G];
    b0[realIdx] = v;
    b1[i] = v;
  }`,
        [cS, G],
      ),
      await ops.linear({ x: B.sInit, w: this.w.pairZi, y: B.zizj, rows: N, inC: cS, outC: cZ, yOff: 0, label: 'pair.zi' }),
      await ops.linear({ x: B.sInit, w: this.w.pairZj, y: B.zizj, rows: N, inC: cS, outC: cZ, yOff: N * cZ, label: 'pair.zj' }),
      await ops.step(
        'pair-init-ext',
        PAIR_INIT_EXT_SOURCE,
        mm,
        [B.zizj, B.zStatic, B.rots, B.trans, B.pairMasks, this.w.pairTemplate, B.overrideQuat, B.pExt],
        uPairInit,
        'pair.initExt',
      ),
    ]);

    // --- LatentTransformer blocks (one program per block for watchdog safety & debug inspection) ---
    this.blockPrograms = [];
    for (let bIdx = 0; bIdx < pt.n_pair_transform_layer; bIdx += 1) {
      const bw = this.w.blocks[bIdx];
      const steps = [];

      // 1. ReducedIPA + residual + LayerNorm
      steps.push(
        ...(await recordReducedIpa(ops, {
          s: B.sExt, p: B.pExt, mask: B.maskExt, w: bw.ipa, B,
          M, cS, cZ, cH: pt.c_hidden_ipa, H: pt.n_head_ipa, inf: 1e5,
        })),
        await ops.layerNorm({
          x: B.sExt, y: B.sExt, rows: M, width: cS,
          w: bw.ipaLnW, b: bw.ipaLnB, label: `lt${bIdx}.ipaLn`,
        }),
      );

      // 2. StructureTransition + token_mask
      steps.push(
        ...(await this._recordStructureTransition(B.sExt, B.sExtTmp1, B.sExtTmp2, bw.ipaTransition, M, cS)),
        await ops.map(
          'mask-s-ext',
          M * cS,
          [{ buffer: B.sExt, write: true }, { buffer: B.maskExt }],
          `b0[i] = b0[i] * b1[i / U(0u)];`,
          [cS],
        ),
      );

      // 3. Single -> Pair: p += linear_p(linear_pi(s)[i] + linear_pj(s)[j]) == linear_p(pi)[i] + linear_p(pj)[j]
      steps.push(
        await ops.linear({ x: B.sExt, w: bw.linPi, y: B.pi, rows: M, inC: cS, outC: cZ, label: `lt${bIdx}.pi` }),
        await ops.linear({ x: B.sExt, w: bw.linPj, y: B.pj, rows: M, inC: cS, outC: cZ, label: `lt${bIdx}.pj` }),
        await ops.linear({ x: B.pi, w: bw.linP, y: B.piP, rows: M, inC: cZ, outC: cZ, label: `lt${bIdx}.piP` }),
        await ops.linear({ x: B.pj, w: bw.linP, y: B.pjP, rows: M, inC: cZ, outC: cZ, label: `lt${bIdx}.pjP` }),
        await ops.map(
          'outer-add-pair',
          mm * cZ,
          [{ buffer: B.pExt, write: true }, { buffer: B.piP }, { buffer: B.pjP }],
          `let cZ = U(0u);
  let M = U(1u);
  let c = i % cZ;
  let pair = i / cZ;
  let ri = pair / M;
  let rj = pair % M;
  b0[i] = b0[i] + b1[ri * cZ + c] + b2[rj * cZ + c];`,
          [cZ, M],
        ),
      );

      // 4. TriangleMultiplicationOutgoing & TriangleMultiplicationIncoming
      steps.push(...(await this._recordTriangle(bw.triOut, false)));
      steps.push(...(await this._recordTriangle(bw.triIn, true)));

      // 5. PairTransition (chunked) + pairMaskExt
      for (const [r0, rows] of pairChunks) {
        steps.push(
          await ops.layerNorm({
            x: B.pExt, y: B.zLnChunk, rows, width: cZ,
            w: bw.pairTrans.lnW, b: bw.pairTrans.lnB, xOff: r0 * cZ, yOff: 0, label: `lt${bIdx}.ptLn`,
          }),
          await ops.linear({
            x: B.zLnChunk, w: bw.pairTrans.w1, bias: bw.pairTrans.b1, y: B.pairScratch,
            rows, inC: cZ, outC: pairExp * cZ, act: 'relu', label: `lt${bIdx}.ptL1`,
          }),
          await ops.linear({
            x: B.pairScratch, w: bw.pairTrans.w2, bias: bw.pairTrans.b2, y: B.pExt,
            rows, inC: pairExp * cZ, outC: cZ, yOff: r0 * cZ, accumulate: true, label: `lt${bIdx}.ptL2`,
          }),
        );
      }
      steps.push(
        await ops.map(
          'mask-p-ext',
          mm * cZ,
          [{ buffer: B.pExt, write: true }, { buffer: B.pairMaskExt }],
          `b0[i] = b0[i] * b1[i / U(0u)];`,
          [cZ],
        ),
      );

      this.blockPrograms.push(ops.program(steps));
    }

    // --- Unpad global tokens -> B.s [N, cS], B.z [N*N, cZ] ---
    this.unpadProgram = ops.program([
      await ops.map(
        'unpad-single',
        N * cS,
        [{ buffer: B.s, write: true }, { buffer: B.sExt }],
        `b0[i] = b1[U(0u) + i];`,
        [G * cS],
      ),
      await ops.map(
        'unpad-pair',
        nn * cZ,
        [{ buffer: B.z, write: true }, { buffer: B.pExt }],
        `let cZ = U(0u);
  let N = U(1u);
  let G = U(2u);
  let M = U(3u);
  let c = i % cZ;
  let pair = i / cZ;
  let ri = pair / N;
  let rj = pair % N;
  b0[i] = b1[((ri + G) * M + (rj + G)) * cZ + c];`,
        [cZ, N, G, M],
      ),
    ]);

    // --- SequenceNet: Linear(cS -> cH) -> ReLU -> Linear(cH -> cH) -> ReLU -> Linear(cH -> 20) ---
    this.sequenceProgram = ops.program([
      await ops.linear({ x: B.s, w: this.w.seq.w0, y: B.seqH1, rows: N, inC: cS, outC: sq.c_hidden, act: 'relu', label: 'seq.0' }),
      await ops.linear({ x: B.seqH1, w: this.w.seq.w1, y: B.seqH2, rows: N, inC: sq.c_hidden, outC: sq.c_hidden, act: 'relu', label: 'seq.2' }),
      await ops.linear({ x: B.seqH2, w: this.w.seq.w2, y: B.aiOut, rows: N, inC: sq.c_hidden, outC: NUM_RESTYPES, label: 'seq.4' }),
    ]);

    // --- StructureNet: n_structure_block x n_structure_layer ---
    this.structLayerPrograms = [];
    for (let lIdx = 0; lIdx < st.n_structure_layer; lIdx += 1) {
      const lw = this.w.structLayers[lIdx];
      const steps = [
        ...(await recordFullIpa(ops, {
          s: B.s, z: B.z, rots: B.rots, trans: B.trans, mask: B.frameMask,
          w: lw.ipa, B, N, cS, cZ, cH: st.c_hidden_ipa, H: st.n_head_ipa,
          nQkPts: st.n_qk_point, nVPts: st.n_v_point, inf: 1e5, eps: 1e-8,
        })),
        await ops.layerNorm({
          x: B.s, y: B.s, rows: N, width: cS,
          w: lw.ipaLnW, b: lw.ipaLnB, label: `st${lIdx}.ipaLn`,
        }),
        ...(await this._recordStructureTransition(B.s, B.sTmp1, B.sTmp2, lw.transition, N, cS)),
        ...(await recordBackboneUpdate(ops, {
          s: B.s, rots: B.rots, trans: B.trans, w: lw.bbUpdate, B, N, cS,
        })),
      ];
      this.structLayerPrograms.push(ops.program(steps));
    }
  }

  /** Upload batch-static masks and compute static `zStatic` (relpos + optional cond_template). */
  _syncStaticBatch(batch) {
    const N = batch.token_mask.length;
    const G = this.cfg.pair_transform_net.n_global_token;
    const M = N + G;
    const ops = this.ops;
    const B = this.buf;

    const maskExt = new Float32Array(M);
    for (let i = 0; i < G; i += 1) maskExt[i] = 1.0;
    maskExt.set(batch.token_mask, G);

    const pairMaskExt = new Float32Array(M * M);
    for (let i = 0; i < M; i += 1) {
      const mi = maskExt[i];
      for (let j = 0; j < M; j += 1) pairMaskExt[i * M + j] = mi * maskExt[j];
    }

    const { pairMasks, hasCond } = buildPairMasks(batch);
    const relpos = buildRelposFeatures(
      batch,
      this.cfg.pair_feature_net.max_relative_index,
      this.cfg.pair_feature_net.max_relative_chain,
    );

    ops.write(B.tokenMask, batch.token_mask);
    ops.write(B.frameMask, batch.token_struct_frame_mask);
    ops.write(B.maskExt, maskExt);
    ops.write(B.pairMaskExt, pairMaskExt);
    ops.write(B.pairMasks, pairMasks);
    ops.write(B.relposFeat, relpos.data);

    if (hasCond) {
      const condTmpl = buildCondTemplateFeatures(batch, pairMasks, this.cfg.pair_feature_net, this.canonicalQuat);
      ops.write(B.condTmplFeat, condTmpl.data);
      ops.run([this.relposProgram, this.condTemplateProgram]);
    } else {
      ops.run([this.relposProgram]);
    }

    this.cachedBatchRef = batch;
  }

  /**
   * Run one `V1Denoiser.forward(batch, xl, t)` pass on WebGPU.
   *
   * @param {object} batch Feature batch from `createFeaturesFromLength(L)` or `createFeaturesFromSpec(...)`
   * @param {Float32Array} xl Noisy CA coordinates `[N * 3]`
   * @param {number} t Normalized diffusion timestep (`s / n_timestep`)
   * @param {{
   *   returnIntermediates?: boolean,
   *   overrideRelQuats?: Float32Array | null,
   *   forceSyncBatch?: boolean,
   * }} [opts]
   * @returns {Promise<{ ai: Float32Array, xl: Float32Array, intermediates?: object }>}
   */
  async forward(batch, xl, t, opts = {}) {
    const N = batch.token_mask.length;
    const useOverrideQuat = Boolean(opts.overrideRelQuats);
    await this.record(N, useOverrideQuat);

    if (opts.forceSyncBatch || this.cachedBatchRef !== batch) {
      this._syncStaticBatch(batch);
    }

    const ops = this.ops;
    const B = this.buf;
    const fi = computeNoisyStructureFrames(batch, xl);
    const singleFeat = buildSingleInputFeatures(batch, t, this.cfg.single_feature_net, this.singleFeatHost);

    ops.write(B.singleFeat, singleFeat);
    ops.write(B.rots, fi.rots);
    ops.write(B.trans, fi.trans);
    if (useOverrideQuat) {
      ops.write(B.overrideQuat, opts.overrideRelQuats);
    }

    if (!opts.returnIntermediates) {
      // Fast path: submit all blocks + sequence + structure, single readback of ai & xl!
      const progs = [this.embedProgram, ...this.blockPrograms, this.unpadProgram, this.sequenceProgram];
      for (let blk = 0; blk < this.cfg.structure_net.n_structure_block; blk += 1) {
        progs.push(...this.structLayerPrograms);
      }
      const res = await ops.runAndReadMany(progs, [
        { key: 'ai', buffer: B.aiOut, floats: N * NUM_RESTYPES },
        { key: 'xl', buffer: B.trans, floats: N * 3 },
      ]);
      return res;
    }

    // Debug/parity path: capture intermediate tensors matching `fixtures/oracle_l16.*`
    const cS = this.cfg.c_s;
    const cZ = this.cfg.c_z;
    const M = this.M;
    const G = this.cfg.pair_transform_net.n_global_token;

    const embRes = await ops.runAndReadMany([this.embedProgram], [
      { key: 'si_init', buffer: B.sInit, floats: N * cS },
      { key: 'pExt0', buffer: B.pExt, floats: M * M * cZ },
    ]);
    const zij_init = new Float32Array(N * N * cZ);
    for (let i = 0; i < N; i += 1) {
      for (let j = 0; j < N; j += 1) {
        const src = ((i + G) * M + (j + G)) * cZ;
        zij_init.set(embRes.pExt0.subarray(src, src + cZ), (i * N + j) * cZ);
      }
    }

    const blk0Res = await ops.runAndReadMany([this.blockPrograms[0]], [
      { key: 'lt_block0_si_with_global', buffer: B.sExt, floats: M * cS },
      { key: 'lt_block0_zij_with_global', buffer: B.pExt, floats: M * M * cZ },
    ]);
    const lt_block0_si = blk0Res.lt_block0_si_with_global.slice(G * cS);
    const lt_block0_zij = new Float32Array(N * N * cZ);
    for (let i = 0; i < N; i += 1) {
      for (let j = 0; j < N; j += 1) {
        const src = ((i + G) * M + (j + G)) * cZ;
        lt_block0_zij.set(blk0Res.lt_block0_zij_with_global.subarray(src, src + cZ), (i * N + j) * cZ);
      }
    }

    const ltOutRes = await ops.runAndReadMany(
      [...this.blockPrograms.slice(1), this.unpadProgram, this.sequenceProgram],
      [
        { key: 'lt_out_si', buffer: B.s, floats: N * cS },
        { key: 'lt_out_zij', buffer: B.z, floats: N * N * cZ },
        { key: 'ai', buffer: B.aiOut, floats: N * NUM_RESTYPES },
      ],
    );

    const st0Res = await ops.runAndReadMany([this.structLayerPrograms[0]], [
      { key: 'struct_layer0_si', buffer: B.s, floats: N * cS },
      { key: 'struct_layer0_fi_rots', buffer: B.rots, floats: N * 9 },
      { key: 'struct_layer0_fi_trans', buffer: B.trans, floats: N * 3 },
    ]);

    const remainingStruct = [...this.structLayerPrograms.slice(1)];
    for (let blk = 1; blk < this.cfg.structure_net.n_structure_block; blk += 1) {
      remainingStruct.push(...this.structLayerPrograms);
    }
    const finalRes = await ops.runAndReadMany(remainingStruct, [
      { key: 'struct_final_si', buffer: B.s, floats: N * cS },
      { key: 'struct_final_fi_rots', buffer: B.rots, floats: N * 9 },
      { key: 'xl', buffer: B.trans, floats: N * 3 },
    ]);

    return {
      ai: ltOutRes.ai,
      xl: finalRes.xl,
      intermediates: {
        fi_rots: fi.rots,
        fi_trans: fi.trans,
        si_init: embRes.si_init,
        zij_init,
        lt_block0_si,
        lt_block0_zij,
        lt_block0_si_with_global: blk0Res.lt_block0_si_with_global,
        lt_block0_zij_with_global: blk0Res.lt_block0_zij_with_global,
        lt_out_si: ltOutRes.lt_out_si,
        lt_out_zij: ltOutRes.lt_out_zij,
        struct_layer0_si: st0Res.struct_layer0_si,
        struct_layer0_fi_rots: st0Res.struct_layer0_fi_rots,
        struct_layer0_fi_trans: st0Res.struct_layer0_fi_trans,
        struct_final_si: finalRes.struct_final_si,
        struct_final_fi_rots: finalRes.struct_final_fi_rots,
      },
    };
  }

  /** Memory accounting snapshot in bytes. */
  memoryUsage() {
    return {
      weightBytes: this.ops.weightBytes,
      activationBytes: this.ops.bytes,
      totalBytes: this.ops.weightBytes + this.ops.bytes,
    };
  }

  destroy() {
    this.ops.destroy();
    this.N = 0;
    this.M = 0;
    this.w = null;
    this.buf = null;
    this.cachedBatchRef = null;
  }
}
