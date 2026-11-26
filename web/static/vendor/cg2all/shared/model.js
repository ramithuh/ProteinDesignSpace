// cg2all CalphaBasedModel network (libmodel.Model + huhlim/SE3Transformer) assembled from checkpoint tensors,
// with a float64 CPU forward pass used for parity tests and as the no-WebGPU fallback.
//
// Layout: a fiber feature set is `{ [degree]: { C, data } }`, data row-major [N][C][2d+1].
// Convolutions are described generically as units (one per output degree) of blocks (one per input degree):
//   out[e][co][k] = Σ_blocks Σ_ci Σ_f W_e[(co·Cin + ci)·F + fOff + f] · tmp[ci][f][k],
//   tmp[ci][f][k] = Σ_l x_dIn[src_e][ci][l] · basis_(dIn,dOut)[e][l][f][k]   ((0,0): tmp = x),
//   W_e = radial MLP(edge features) — one MLP per (dIn, dOut) pair for the pairwise ConvSE3s inside the
//   attention blocks (low_memory → ConvSE3FuseLevel.NONE), one per dOut for the final ConvSE3, which
//   SE3Transformer builds with the default fuse level and therefore runs PARTIAL "conv_out" fusion.

import { Y00 } from './features.js';

export const NUM_HEADS = 8;
export const HIDDEN_DIM = 32;
export const EMBED_DIM = 40;
export const NUM_RESIDUE_TYPES = 22;
const NORM_CLAMP = 2 ** -24;
const LN_EPS = 1e-5;

const degreesOf = (fiber) => Object.keys(fiber).map(Number).sort((a, b) => a - b);
const dimOf = (d) => 2 * d + 1;

function take(tensors, name, shape) {
  const t = tensors.get(name);
  if (!t) throw new Error(`cg2all weights: missing tensor ${name}`);
  if (shape && (t.shape.length !== shape.length || t.shape.some((s, i) => shape[i] !== -1 && s !== shape[i]))) {
    throw new Error(`cg2all weights: ${name} has shape [${t.shape}] but [${shape}] was expected`);
  }
  return t;
}

function has(tensors, name) {
  return tensors.has(name);
}

/** NormSE3: per-degree LayerNorm, or one GroupNorm (one group per degree) when all channel counts match. */
function normSpec(tensors, prefix, fiber) {
  const degrees = degreesOf(fiber);
  const out = { kind: 'norm', degrees: {} };
  if (has(tensors, `${prefix}.group_norm.weight`)) {
    const g = take(tensors, `${prefix}.group_norm.weight`).data;
    const b = take(tensors, `${prefix}.group_norm.bias`).data;
    let o = 0;
    for (const d of degrees) {
      const C = fiber[d];
      out.degrees[d] = { C, gamma: g.subarray(o, o + C), beta: b.subarray(o, o + C) };
      o += C;
    }
  } else {
    for (const d of degrees) {
      const C = fiber[d];
      out.degrees[d] = {
        C,
        gamma: take(tensors, `${prefix}.layer_norms.${d}.weight`, [C]).data,
        beta: take(tensors, `${prefix}.layer_norms.${d}.bias`, [C]).data,
      };
    }
  }
  return out;
}

/** LinearSE3: out_d = W_d @ x_d per output degree (no bias). */
function linearSpec(tensors, prefix, fiberIn, fiberOut) {
  const out = { kind: 'linear', degrees: {} };
  for (const d of degreesOf(fiberOut)) {
    const t = take(tensors, `${prefix}.weights.${d}`, [fiberOut[d], fiberIn[d] ?? 0]);
    out.degrees[d] = { cOut: fiberOut[d], cIn: t.shape[1], w: t.data };
  }
  return out;
}

function sequentialSpec(tensors, prefix, fibers) {
  // fibers[i] = [fiberIn, fiberOut] of the i-th (NormSE3, LinearSE3) pair
  const seq = [];
  fibers.forEach(([fin, fout], i) => {
    seq.push(normSpec(tensors, `${prefix}.${2 * i}`, fin));
    seq.push(linearSpec(tensors, `${prefix}.${2 * i + 1}`, fin, fout));
  });
  return seq;
}

function radialMlp(tensors, prefix, rows) {
  return {
    w0: take(tensors, `${prefix}.0.weight`, [HIDDEN_DIM, 4]).data,
    b0: take(tensors, `${prefix}.0.bias`, [HIDDEN_DIM]).data,
    g0: take(tensors, `${prefix}.1.weight`, [HIDDEN_DIM]).data,
    be0: take(tensors, `${prefix}.1.bias`, [HIDDEN_DIM]).data,
    w1: take(tensors, `${prefix}.3.weight`, [HIDDEN_DIM, HIDDEN_DIM]).data,
    b1: take(tensors, `${prefix}.3.bias`, [HIDDEN_DIM]).data,
    g1: take(tensors, `${prefix}.4.weight`, [HIDDEN_DIM]).data,
    be1: take(tensors, `${prefix}.4.bias`, [HIDDEN_DIM]).data,
    w2: take(tensors, `${prefix}.6.weight`, [rows, HIDDEN_DIM]).data,
    rows,
  };
}

function pairwiseConv(tensors, prefix, fiberIn, fiberOut) {
  const mlps = [];
  const units = [];
  for (const dOut of degreesOf(fiberOut)) {
    const blocks = [];
    for (const dIn of degreesOf(fiberIn)) {
      const F = 2 * Math.min(dIn, dOut) + 1;
      mlps.push(radialMlp(tensors, `${prefix}.conv.${dIn},${dOut}.radial_func.net`, F * fiberIn[dIn] * fiberOut[dOut]));
      blocks.push({ dIn, cIn: fiberIn[dIn], mlp: mlps.length - 1, F, fOff: 0, fCount: F });
    }
    units.push({ dOut, cOut: fiberOut[dOut], blocks });
  }
  return { mlps, units, fiberIn, fiberOut, pool: false, self: null };
}

function fusedOutConv(tensors, prefix, fiberIn, fiberOut) {
  const mlps = [];
  const units = [];
  const self = {};
  const degIn = degreesOf(fiberIn);
  const cIn = fiberIn[degIn[0]];
  for (const dOut of degreesOf(fiberOut)) {
    const F = degIn.reduce((s, d) => s + 2 * Math.min(d, dOut) + 1, 0);
    const mlp = radialMlp(tensors, `${prefix}.conv_out.${dOut}.radial_func.net`, F * cIn * fiberOut[dOut]);
    if (dOut === 0) {
      // basis["out0_fused"] carries the constant (0,0) block Y^0 = 1/sqrt(4π); fold it into the radial
      // weights so every engine can treat the (0,0) pair as identity (as in the pairwise convolutions).
      const w2 = Float32Array.from(mlp.w2);
      for (let co = 0; co < fiberOut[0]; co++) {
        for (let ci = 0; ci < cIn; ci++) {
          const row = (co * cIn + ci) * F;
          for (let j = 0; j < HIDDEN_DIM; j++) w2[row * HIDDEN_DIM + j] *= Y00;
        }
      }
      mlp.w2 = w2;
    }
    mlps.push(mlp);
    const blocks = [];
    let fOff = 0;
    for (const dIn of degIn) {
      const fCount = 2 * Math.min(dIn, dOut) + 1;
      blocks.push({ dIn, cIn, mlp: mlps.length - 1, F, fOff, fCount });
      fOff += fCount;
    }
    units.push({ dOut, cOut: fiberOut[dOut], blocks });
    if (fiberIn[dOut]) {
      self[dOut] = take(tensors, `${prefix}.to_kernel_self.${dOut}`, [fiberOut[dOut], fiberIn[dOut]]).data;
    }
  }
  return { mlps, units, fiberIn, fiberOut, pool: true, self };
}

/**
 * Assemble the network from a `Map<name, {shape, data: Float32Array}>` of state_dict tensors
 * (keys without the Lightning "model." prefix).
 */
export function buildNetwork(tensors) {
  const pass = { 0: 64, 1: 32 };
  const hidden = { 0: 32, 1: 32, 2: 32 };
  const init = { 0: 17 + EMBED_DIM, 1: 4 };
  const out = { 0: 20, 1: 3 };
  const embedding = take(tensors, 'embedding_module.layer.weight', [NUM_RESIDUE_TYPES, EMBED_DIM]).data;

  const initSeq = sequentialSpec(tensors, 'initialization_module.linear_module', [
    [init, pass], [pass, pass], [pass, pass], [pass, pass], [pass, pass], [pass, pass],
  ]);
  const layers = [];
  let fiberIn = pass;
  for (let i = 0; i < 6; i++) {
    const p = `interaction_module.graph_module.graph_modules.${2 * i}`;
    const value = {};
    const key = {};
    const kv = {};
    for (const d of degreesOf(hidden)) {
      value[d] = hidden[d] / 2; // channels_div = 2
      if (fiberIn[d]) key[d] = value[d];
      kv[d] = value[d] + (key[d] ?? 0);
    }
    const catFiber = {};
    for (const d of degreesOf(value)) catFiber[d] = value[d] + (fiberIn[d] ?? 0);
    layers.push({
      fiberIn,
      value,
      key,
      kvConv: pairwiseConv(tensors, `${p}.to_key_value`, fiberIn, kv),
      query: linearSpec(tensors, `${p}.to_query`, fiberIn, key),
      project: linearSpec(tensors, `${p}.project`, catFiber, hidden),
      norm: normSpec(tensors, `interaction_module.graph_module.graph_modules.${2 * i + 1}`, hidden),
      keyFeatures: degreesOf(key).reduce((s, d) => s + key[d] * dimOf(d), 0),
    });
    fiberIn = hidden;
  }
  const final = fusedOutConv(tensors, 'interaction_module.graph_module.graph_modules.12', hidden, pass);
  const structSeq = sequentialSpec(tensors, 'structure_module.linear_module', [
    [pass, pass], [pass, pass], [pass, pass], [pass, pass], [pass, pass], [pass, out],
  ]);
  return { embedding, init: initSeq, layers, final, struct: structSeq, fibers: { pass, hidden, init, out } };
}

// ───────────────────────── CPU forward (float64) ─────────────────────────

const elu = (x) => (x > 0 ? x : Math.expm1(x));

function layerNormInPlace(v, gamma, beta) {
  const C = v.length;
  let mean = 0;
  for (let c = 0; c < C; c++) mean += v[c];
  mean /= C;
  let varSum = 0;
  for (let c = 0; c < C; c++) varSum += (v[c] - mean) ** 2;
  const inv = 1 / Math.sqrt(varSum / C + LN_EPS);
  for (let c = 0; c < C; c++) v[c] = (v[c] - mean) * inv * gamma[c] + beta[c];
}

export function applyNorm(spec, x, n) {
  const out = {};
  for (const [dStr, p] of Object.entries(spec.degrees)) {
    const d = Number(dStr);
    const M = dimOf(d);
    const src = x[d].data;
    const dst = new Float64Array(src.length);
    const norms = new Float64Array(p.C);
    const scaled = new Float64Array(p.C);
    for (let i = 0; i < n; i++) {
      const base = i * p.C * M;
      for (let c = 0; c < p.C; c++) {
        let s = 0;
        for (let m = 0; m < M; m++) s += src[base + c * M + m] ** 2;
        norms[c] = Math.max(Math.sqrt(s), NORM_CLAMP);
        scaled[c] = norms[c];
      }
      layerNormInPlace(scaled, p.gamma, p.beta);
      for (let c = 0; c < p.C; c++) {
        const f = elu(scaled[c]) / norms[c];
        for (let m = 0; m < M; m++) dst[base + c * M + m] = src[base + c * M + m] * f;
      }
    }
    out[d] = { C: p.C, data: dst };
  }
  return out;
}

/** LinearSE3 on the channel-concatenation of `x` and optional `extra` (aggregate_residual "cat"). */
export function applyLinear(spec, x, n, extra) {
  const out = {};
  for (const [dStr, p] of Object.entries(spec.degrees)) {
    const d = Number(dStr);
    const M = dimOf(d);
    const a = x[d];
    const b = extra?.[d];
    const ca = a ? a.C : 0;
    const cb = b ? b.C : 0;
    if (ca + cb !== p.cIn) throw new Error(`cg2all: linear degree ${d} expects ${p.cIn} channels, got ${ca + cb}`);
    const dst = new Float64Array(n * p.cOut * M);
    for (let i = 0; i < n; i++) {
      for (let co = 0; co < p.cOut; co++) {
        const wRow = co * p.cIn;
        for (let m = 0; m < M; m++) {
          let s = 0;
          for (let ci = 0; ci < ca; ci++) s += p.w[wRow + ci] * a.data[(i * ca + ci) * M + m];
          for (let ci = 0; ci < cb; ci++) s += p.w[wRow + ca + ci] * b.data[(i * cb + ci) * M + m];
          dst[(i * p.cOut + co) * M + m] = s;
        }
      }
    }
    out[d] = { C: p.cOut, data: dst };
  }
  return out;
}

function runSequence(seq, x, n) {
  let h = x;
  for (const op of seq) h = op.kind === 'norm' ? applyNorm(op, h, n) : applyLinear(op, h, n);
  return h;
}

/** Radial-profile hidden activations (Linear-LN-ELU-Linear-LN-ELU) for every edge: E × 32. */
export function radialHidden(mlp, edgeFeat, e) {
  const H = HIDDEN_DIM;
  const out = new Float64Array(e * H);
  const h0 = new Float64Array(H);
  const h1 = new Float64Array(H);
  for (let k = 0; k < e; k++) {
    for (let j = 0; j < H; j++) {
      let s = mlp.b0[j];
      for (let i = 0; i < 4; i++) s += mlp.w0[j * 4 + i] * edgeFeat[k * 4 + i];
      h0[j] = s;
    }
    layerNormInPlace(h0, mlp.g0, mlp.be0);
    for (let j = 0; j < H; j++) h0[j] = elu(h0[j]);
    for (let j = 0; j < H; j++) {
      let s = mlp.b1[j];
      for (let i = 0; i < H; i++) s += mlp.w1[j * H + i] * h0[i];
      h1[j] = s;
    }
    layerNormInPlace(h1, mlp.g1, mlp.be1);
    for (let j = 0; j < H; j++) out[k * H + j] = elu(h1[j]);
  }
  return out;
}

/** Edge-level ConvSE3 output (pool=False): `{ [dOut]: { C, data: E × C × (2dOut+1) } }`. */
export function convEdges(conv, x, g) {
  const { e, src, edgeFeat, basis } = g;
  const hidden = conv.mlps.map((m) => radialHidden(m, edgeFeat, e));
  const pairIndex = new Map(basis.pairs.map((p) => [`${p.dIn},${p.dOut}`, p]));
  const out = {};
  for (const unit of conv.units) {
    const K = dimOf(unit.dOut);
    const dst = new Float64Array(e * unit.cOut * K);
    for (const b of unit.blocks) {
      const L = dimOf(b.dIn);
      const xin = x[b.dIn].data;
      const pair = b.dIn === 0 && unit.dOut === 0 ? null : pairIndex.get(`${b.dIn},${unit.dOut}`);
      const w2 = conv.mlps[b.mlp].w2;
      const H = hidden[b.mlp];
      const tmp = new Float64Array(b.cIn * b.fCount * K);
      for (let k = 0; k < e; k++) {
        const xs = src[k] * b.cIn * L;
        if (pair) {
          const bb = pair.offset + k * pair.size;
          for (let ci = 0; ci < b.cIn; ci++) {
            for (let f = 0; f < b.fCount; f++) {
              for (let kk = 0; kk < K; kk++) {
                let s = 0;
                for (let l = 0; l < L; l++) s += xin[xs + ci * L + l] * basis.data[bb + (l * b.fCount + f) * K + kk];
                tmp[(ci * b.fCount + f) * K + kk] = s;
              }
            }
          }
        } else {
          for (let ci = 0; ci < b.cIn; ci++) tmp[ci] = xin[xs + ci];
        }
        const hk = k * HIDDEN_DIM;
        for (let co = 0; co < unit.cOut; co++) {
          const ob = (k * unit.cOut + co) * K;
          for (let ci = 0; ci < b.cIn; ci++) {
            const rowBase = (co * b.cIn + ci) * b.F + b.fOff;
            for (let f = 0; f < b.fCount; f++) {
              const r = (rowBase + f) * HIDDEN_DIM;
              let w = 0;
              for (let j = 0; j < HIDDEN_DIM; j++) w += H[hk + j] * w2[r + j];
              const tb = (ci * b.fCount + f) * K;
              for (let kk = 0; kk < K; kk++) dst[ob + kk] += w * tmp[tb + kk];
            }
          }
        }
      }
    }
    out[unit.dOut] = { C: unit.cOut, data: dst };
  }
  return out;
}

/** Final ConvSE3 (pool=True, self_interaction=True): copy_e_sum over incoming edges of (conv + K_self·x_dst). */
export function convPool(conv, x, g) {
  const { n, e, dst: dstIdx } = g;
  const edges = convEdges(conv, x, g);
  const indeg = new Float64Array(n);
  for (let k = 0; k < e; k++) indeg[dstIdx[k]] += 1;
  const out = {};
  for (const unit of conv.units) {
    const K = dimOf(unit.dOut);
    const C = unit.cOut;
    const data = new Float64Array(n * C * K);
    const ed = edges[unit.dOut].data;
    for (let k = 0; k < e; k++) {
      const t = dstIdx[k];
      for (let i = 0; i < C * K; i++) data[t * C * K + i] += ed[k * C * K + i];
    }
    const ks = conv.self[unit.dOut];
    if (ks) {
      const cin = x[unit.dOut].C;
      const xd = x[unit.dOut].data;
      for (let i = 0; i < n; i++) {
        if (!indeg[i]) continue;
        for (let co = 0; co < C; co++) {
          for (let kk = 0; kk < K; kk++) {
            let s = 0;
            for (let ci = 0; ci < cin; ci++) s += ks[co * cin + ci] * xd[(i * cin + ci) * K + kk];
            data[(i * C + co) * K + kk] += indeg[i] * s;
          }
        }
      }
    }
    out[unit.dOut] = { C, data };
  }
  return out;
}

/** AttentionBlockSE3 (key/value TFN conv, linear query, dst-softmax attention, residual cat + projection). */
export function attentionBlock(layer, x, g) {
  const { n, e, dst: dstIdx, csr } = g;
  const kv = convEdges(layer.kvConv, x, g);
  const q = applyLinear(layer.query, x, n);
  const keyDeg = degreesOf(layer.key);
  const valDeg = degreesOf(layer.value);
  const scale = 1 / Math.sqrt(layer.keyFeatures);
  const logits = new Float64Array(e * NUM_HEADS);
  for (let k = 0; k < e; k++) {
    const t = dstIdx[k];
    for (let h = 0; h < NUM_HEADS; h++) {
      let s = 0;
      for (const d of keyDeg) {
        const M = dimOf(d);
        const ck = layer.key[d];
        const cv = layer.value[d];
        const ckv = kv[d].C;
        const per = ck / NUM_HEADS;
        for (let c = h * per; c < (h + 1) * per; c++) {
          for (let m = 0; m < M; m++) {
            s += kv[d].data[(k * ckv + cv + c) * M + m] * q[d].data[(t * ck + c) * M + m];
          }
        }
      }
      logits[k * NUM_HEADS + h] = s * scale;
    }
  }
  const z = {};
  for (const d of valDeg) z[d] = { C: layer.value[d], data: new Float64Array(n * layer.value[d] * dimOf(d)) };
  for (let i = 0; i < n; i++) {
    const a = csr.offsets[i];
    const b = csr.offsets[i + 1];
    if (a === b) continue;
    for (let h = 0; h < NUM_HEADS; h++) {
      let mx = -Infinity;
      for (let p = a; p < b; p++) mx = Math.max(mx, logits[csr.edges[p] * NUM_HEADS + h]);
      let sum = 0;
      for (let p = a; p < b; p++) sum += Math.exp(logits[csr.edges[p] * NUM_HEADS + h] - mx);
      for (let p = a; p < b; p++) {
        const k = csr.edges[p];
        const wk = Math.exp(logits[k * NUM_HEADS + h] - mx) / sum;
        for (const d of valDeg) {
          const M = dimOf(d);
          const cv = layer.value[d];
          const ckv = kv[d].C;
          const per = cv / NUM_HEADS;
          for (let c = h * per; c < (h + 1) * per; c++) {
            for (let m = 0; m < M; m++) z[d].data[(i * cv + c) * M + m] += wk * kv[d].data[(k * ckv + c) * M + m];
          }
        }
      }
    }
  }
  return applyLinear(layer.project, z, n, x);
}

/** Node input features: degree 0 = [17 geometric scalars | 40-d residue embedding], degree 1 = 4 vectors. */
export function inputFiber(net, inp) {
  const { n, f0, f1, residueType } = inp;
  const c0 = 17 + EMBED_DIM;
  const d0 = new Float64Array(n * c0);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 17; c++) d0[i * c0 + c] = f0[i * 17 + c];
    const t = residueType[i];
    for (let c = 0; c < EMBED_DIM; c++) d0[i * c0 + 17 + c] = net.embedding[t * EMBED_DIM + c];
  }
  return { 0: { C: c0, data: d0 }, 1: { C: 4, data: Float64Array.from(f1) } };
}

/**
 * Full forward pass. `inp`: prepared graph (see index.js `prepareGraph`).
 * Returns the structure-module output: `s` (N × 20 scalars) and `v` (N × 3 × 3 vectors), plus stage taps.
 */
export function forwardCpu(net, inp, { taps = false } = {}) {
  const g = inp;
  const x = runSequence(net.init, inputFiber(net, inp), g.n);
  let h = x;
  const layerTaps = [];
  for (const layer of net.layers) {
    h = attentionBlock(layer, h, g);
    if (taps) layerTaps.push(h);
    h = applyNorm(layer.norm, h, g.n);
    if (taps) layerTaps.push(h);
  }
  h = convPool(net.final, h, g);
  if (taps) layerTaps.push(h);
  for (const d of Object.keys(x)) {
    const a = h[d].data;
    const b = x[d].data;
    for (let i = 0; i < a.length; i++) a[i] += b[i];
  }
  const out = runSequence(net.struct, h, g.n);
  return { s: out[0].data, v: out[1].data, ...(taps ? { taps: { init: x, layers: layerTaps } } : {}) };
}
