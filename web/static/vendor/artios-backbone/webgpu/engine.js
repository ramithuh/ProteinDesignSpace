// WebGPU engine for the ArtiosBackbone EDM diffusion module
// (artios/model/modules/diffusion_backbone.py :: DiffusionModule).
//
// One denoiseStep() = DiffusionModule.forward with a cached conditioning:
//   atom encoder   r_noisy * c_in -> linear_no_bias_r_embed (+ self-cond merge)
//                  -> average_atom_to_token
//   conditioning   s = s_base + linear_t(LN(fourier(log(t / sigma) / 4)))
//                  s += transition_s1(s); s += transition_s2(s)
//                  a = (a + linear_no_bias_s(LN(s))) * token_mask
//   transformer    num_blocks AdaLN-Zero DiT blocks, attention with pair bias
//   atom decoder   LN(a * mask) -> broadcast_token_to_atom + q_skip
//                  -> linear_no_bias_a + linear_no_bias_f(ref features)
//                  -> readout transition -> readout LN -> readout_linear
//   EDM output     r_pred = r_noisy * c_skip + F * c_out
//
// Timestep-independent work (pair bias of every block from z, atom
// reference-feature embedding, CSR atom->token tables) is computed once per
// conditioning and reused across steps. Per-shape plans (scratch buffers,
// uniforms, bind groups) are built once and replayed; each step only writes
// x_noisy / t_hat scalars and submits the recorded dispatch list in paced
// chunks.

import {
  attentionKernel,
  broadcastAddKernel,
  edmCombineKernel,
  edmPrecondKernel,
  ELEMWISE_WG,
  fourierKernel,
  gatherAddKernel,
  layerNormKernel,
  matmulKernel,
  MATMUL_TILE,
  scatterMeanKernel,
} from './shaders.js?v=3';
import { linearFormat, packLinearF16, quantizeLinearInt8, roundUp } from './packing.js?v=3';

const BU = globalThis.GPUBufferUsage ?? {
  MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80,
};
const MAP_READ = globalThis.GPUMapMode?.READ ?? 0x1;
const STAGE_COMPUTE = globalThis.GPUShaderStage?.COMPUTE ?? 0x4;
const STORAGE_USAGE = BU.STORAGE | BU.COPY_DST | BU.COPY_SRC;

export const SIGMA_DATA = 16.0;
export const N_REF_FEATURES = 1 + 128 + 4 * 64;

// ---------------------------------------------------------------------------
// Buffer ownership
// ---------------------------------------------------------------------------

class BufferScope {
  constructor(rt, label) {
    this.rt = rt;
    this.label = label;
    this.buffers = new Set();
    this.bytes = 0;
    rt.scopes.add(this);
  }

  raw(bytes, usage, label, init) {
    const size = roundUp(Math.max(bytes, 4), 4);
    const b = this.rt.device.createBuffer({
      size,
      usage,
      label: `${this.label}:${label}`,
      mappedAtCreation: !!init,
    });
    if (init) {
      const view = init instanceof Uint32Array ? new Uint32Array(b.getMappedRange()) : new Float32Array(b.getMappedRange());
      view.set(init);
      b.unmap();
    }
    this.buffers.add(b);
    this.bytes += size;
    return b;
  }

  empty(nFloats, label) {
    return this.raw(nFloats * 4, STORAGE_USAGE, label);
  }

  f32(data, label) {
    const arr = data instanceof Float32Array ? data : Float32Array.from(data);
    return this.raw(arr.byteLength, STORAGE_USAGE, label, arr);
  }

  u32(data, label) {
    const arr = data instanceof Uint32Array ? data : Uint32Array.from(data);
    return this.raw(arr.byteLength, STORAGE_USAGE, label, arr);
  }

  uniform(fields, values) {
    const n = roundUp(fields.length, 4);
    const u = new Uint32Array(n);
    const f = new Float32Array(u.buffer);
    fields.forEach(([name, type], i) => {
      const v = values[name] ?? 0;
      if (type === 'f32') {
        f[i] = v;
      } else {
        if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) {
          throw new Error(`uniform field ${name} must be a u32, got ${v}`);
        }
        u[i] = v;
      }
    });
    return this.raw(u.byteLength, BU.UNIFORM | BU.COPY_DST, 'params', u);
  }

  staging(bytes, label) {
    return this.raw(bytes, BU.MAP_READ | BU.COPY_DST, label);
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
    this.buffers.clear();
    this.bytes = 0;
    this.rt.scopes.delete(this);
  }
}

// ---------------------------------------------------------------------------
// Runtime: pipeline cache, op records, paced submission
// ---------------------------------------------------------------------------

export class GpuRuntime {
  constructor(device, opts = {}) {
    if (!device) throw new Error('GpuRuntime: a GPUDevice is required');
    this.device = device;
    this.pipelines = new Map();
    this.scopes = new Set();
    this.submitBudgetFlops = opts.submitBudgetFlops ?? 16e9;
    this.maxDispatchesPerSubmit = opts.maxDispatchesPerSubmit ?? 64;
    this.maxInFlight = Math.max(1, opts.maxInFlight ?? 2);
    this.counters = { submits: 0, dispatches: 0 };
  }

  scope(label) {
    return new BufferScope(this, label);
  }

  liveBuffers() {
    let n = 0;
    for (const s of this.scopes) n += s.buffers.size;
    return n;
  }

  liveBytes() {
    let n = 0;
    for (const s of this.scopes) n += s.bytes;
    return n;
  }

  pipeline(k) {
    let p = this.pipelines.get(k.key);
    if (p) return p;
    const dev = this.device;
    const layout = dev.createBindGroupLayout({
      label: k.key,
      entries: k.bindings.map((b, i) => ({ binding: i, visibility: STAGE_COMPUTE, buffer: { type: b.type } })),
    });
    const module = dev.createShaderModule({ label: k.key, code: k.code });
    const pipe = dev.createComputePipeline({
      label: k.key,
      layout: dev.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'main' },
    });
    p = { pipe, layout, module, kernel: k };
    this.pipelines.set(k.key, p);
    return p;
  }

  /** WGSL compiler messages of every pipeline built so far. */
  async shaderMessages() {
    const out = [];
    for (const [key, p] of this.pipelines) {
      const info = await p.module.getCompilationInfo();
      for (const m of info.messages) out.push(`${key}: ${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
    }
    return out;
  }

  /** Record one dispatch: uniform params are written once into `scope`. */
  op(scope, k, params, buffers, workgroups, cost, label = k.key) {
    const p = this.pipeline(k);
    const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (workgroups.some((w) => w > limit)) throw new Error(`${label}: dispatch ${workgroups} exceeds ${limit}`);
    const ub = scope.uniform(k.fields, params);
    const entries = k.bindings.map((b, i) => {
      const buf = b.type === 'uniform' ? ub : buffers[b.name];
      if (!buf) throw new Error(`${label}: missing binding ${b.name}`);
      return { binding: i, resource: { buffer: buf } };
    });
    const bindGroup = this.device.createBindGroup({ label, layout: p.layout, entries });
    return { pipe: p.pipe, bindGroup, workgroups, cost: cost || 0, label };
  }

  /**
   * Encode and submit `ops`, splitting into command buffers of at most
   * submitBudgetFlops / maxDispatchesPerSubmit and keeping at most
   * maxInFlight submissions queued, so no single submit can run long enough
   * to trip a GPU watchdog (TDR) and the CPU never races far ahead.
   */
  async run(ops, { readbacks = [] } = {}) {
    const dev = this.device;
    const queue = dev.queue;
    const inflight = [];
    let enc = null;
    let pass = null;
    let cost = 0;
    let count = 0;
    const flush = async (last) => {
      if (!enc) enc = dev.createCommandEncoder();
      if (pass) pass.end();
      if (last) for (const rb of readbacks) enc.copyBufferToBuffer(rb.src, 0, rb.dst, 0, rb.bytes);
      queue.submit([enc.finish()]);
      this.counters.submits++;
      enc = null;
      pass = null;
      cost = 0;
      count = 0;
      inflight.push(queue.onSubmittedWorkDone());
      if (inflight.length >= this.maxInFlight) await inflight.shift();
    };
    for (const op of ops) {
      if (!enc) {
        enc = dev.createCommandEncoder();
        pass = enc.beginComputePass();
      }
      pass.setPipeline(op.pipe);
      pass.setBindGroup(0, op.bindGroup);
      pass.dispatchWorkgroups(op.workgroups[0], op.workgroups[1] ?? 1, op.workgroups[2] ?? 1);
      this.counters.dispatches++;
      cost += op.cost;
      count++;
      if (cost >= this.submitBudgetFlops || count >= this.maxDispatchesPerSubmit) await flush(false);
    }
    if (enc || readbacks.length) await flush(true);
    await Promise.all(inflight);
  }

  async read(buffer, nFloats, offsetFloats = 0) {
    const scope = this.scope('read');
    try {
      const bytes = nFloats * 4;
      const staging = scope.staging(bytes, 'staging');
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(buffer, offsetFloats * 4, staging, 0, roundUp(bytes, 4));
      this.device.queue.submit([enc.finish()]);
      await staging.mapAsync(MAP_READ);
      const out = new Float32Array(staging.getMappedRange().slice(0, bytes));
      staging.unmap();
      return out;
    } finally {
      scope.destroy();
    }
  }

  destroy() {
    for (const s of [...this.scopes]) s.destroy();
    this.pipelines.clear();
  }
}

// ---------------------------------------------------------------------------
// Op builders (one dispatch each)
// ---------------------------------------------------------------------------

function grid1d(total) {
  const wgs = Math.max(1, Math.ceil(total / ELEMWISE_WG));
  const gridX = Math.min(wgs, 65535);
  return { gridX, workgroups: [gridX, Math.ceil(wgs / gridX), 1] };
}

/**
 * Prepared linear layer (PyTorch [out, in] weight). `swiglu` interleaves the
 * (a, b) halves of a SwiGLU projection so the matmul epilogue can emit
 * silu(a) * b directly.
 */
export function prepareLinear(rt, scope, { W, N, K, bias = null, precision = 'fp32', fmt, swiglu = false, label = 'linear' }) {
  if (W.length !== N * K) throw new Error(`${label}: expected ${N}x${K} weight, got ${W.length} values`);
  fmt ??= linearFormat(precision, K);
  let Wt = W;
  let bt = bias;
  if (swiglu) {
    if (N % 2) throw new Error(`${label}: SwiGLU projection needs an even row count`);
    const H = N / 2;
    Wt = new Float32Array(N * K);
    for (let p = 0; p < H; p++) {
      Wt.set(W.subarray(p * K, (p + 1) * K), 2 * p * K);
      Wt.set(W.subarray((H + p) * K, (H + p + 1) * K), (2 * p + 1) * K);
    }
    if (bias) {
      bt = new Float32Array(N);
      for (let p = 0; p < H; p++) {
        bt[2 * p] = bias[p];
        bt[2 * p + 1] = bias[H + p];
      }
    }
  }
  let w;
  let ldw;
  let scales = null;
  if (fmt === 'f32') {
    w = scope.f32(Wt, label);
    ldw = K;
  } else if (fmt === 'f16') {
    const packed = packLinearF16(Wt, N, K);
    w = scope.u32(packed.data, label);
    ldw = packed.ldw;
  } else if (fmt === 'i8') {
    const q = quantizeLinearInt8(Wt, N, K);
    w = scope.u32(q.data, label);
    ldw = q.ldw;
    scales = q.scales;
  } else {
    throw new Error(`${label}: unknown format ${fmt}`);
  }
  let aux = null;
  let scaleOff = 0;
  let biasOff = 0;
  if (scales || bt) {
    const auxData = new Float32Array((scales ? N : 0) + (bt ? N : 0));
    if (scales) auxData.set(scales, 0);
    if (bt) {
      biasOff = scales ? N : 0;
      auxData.set(bt, biasOff);
    }
    aux = scope.f32(auxData, `${label}:aux`);
  }
  return { fmt, N, K, ldw, w, aux, hasBias: !!bt, scaleOff, biasOff, swiglu, label };
}

/**
 * Y = X . W^T over weight rows [wrow, wrow + n).
 * gate: { buf, ld, off }; residual: 'inplace' | { buf, ld, off, mod };
 * rowScale: GPUBuffer [M].
 */
export function matmulOp(rt, scope, {
  lin, x, M, ldx, xoff = 0, out, ldo, ooff = 0, wrow = 0, n,
  act = 'none', sigFrom = 0, sigTo, gate = null, residual = null, rowScale = null, transposeOut = false,
}) {
  const N = n ?? lin.N - wrow;
  const fusedAct = lin.swiglu ? 'swiglu' : act;
  const outCols = fusedAct === 'swiglu' ? N / 2 : N;
  const resMode = residual === 'inplace' ? 'inplace' : residual ? 'buffer' : 'none';
  const k = matmulKernel({
    wfmt: lin.fmt, bias: lin.hasBias, act: fusedAct, gate: !!gate, residual: resMode, rowScale: !!rowScale, transposeOut,
  });
  const params = {
    M, N, K: lin.K, ldx: ldx ?? lin.K, xoff, ldw: lin.ldw, wrow, ldo: ldo ?? outCols, ooff,
    ldg: gate?.ld ?? 0, goff: gate?.off ?? 0,
    ldr: residual?.ld ?? 0, roff: residual?.off ?? 0, resMod: residual?.mod ?? M,
    sigFrom, sigTo: sigTo ?? N, scaleOff: lin.scaleOff, biasOff: lin.biasOff,
  };
  const buffers = { X: x, W: lin.w, AUX: lin.aux, OUT: out, G: gate?.buf, RES: residual?.buf, RS: rowScale };
  const wg = [Math.ceil(N / MATMUL_TILE), Math.ceil(M / MATMUL_TILE), 1];
  return rt.op(scope, k, params, buffers, wg, 2 * M * N * lin.K, `matmul:${lin.label}`);
}

function rowGrid(M) {
  const gridX = Math.min(M, 65535);
  return { gridX, workgroups: [gridX, Math.ceil(M / gridX), 1] };
}

export function layerNormOp(rt, scope, { x, out, M, D, ldx = D, ldo = D, ooff = 0, gamma = null, beta = null, rowScale = null, eps = 1e-5, label = 'layernorm' }) {
  const k = layerNormKernel({ mode: 'ln', affine: !!gamma, bias: !!beta, rowScale: !!rowScale });
  const { gridX, workgroups } = rowGrid(M);
  const params = { M, D, ldx, ldo, ooff, gridX, eps };
  return rt.op(scope, k, params, { X: x, GAMMA: gamma, BETA: beta, RS: rowScale, OUT: out }, workgroups, 8 * M * D, label);
}

/** out = shift + scale * LN(x)   (AdaLN-Zero modulation / AdaptiveLayerNorm) */
export function adaLNOp(rt, scope, { x, out, M, D, ldx = D, ldo = D, ooff = 0, shift, scale, eps = 1e-5, label = 'adaln' }) {
  const k = layerNormKernel({ mode: 'adaln' });
  const { gridX, workgroups } = rowGrid(M);
  const params = {
    M, D, ldx, ldo, ooff, gridX, eps,
    lds: shift.ld ?? D, soff: shift.off ?? 0, ldc: scale.ld ?? D, coff: scale.off ?? 0,
  };
  return rt.op(scope, k, params, { X: x, SHIFT: shift.buf, SCALE: scale.buf, OUT: out }, workgroups, 10 * M * D, label);
}

function attnTiles(rt, D, bq, maxBkt) {
  const tpr = D % 4 === 0 ? 4 : D % 2 === 0 ? 2 : 1;
  const limit = rt.device.limits.maxComputeWorkgroupStorageSize;
  for (const bkt of [64, 32, 16, 8, 4]) {
    if (bkt > maxBkt) continue;
    if (4 * (2 * bkt * D + bq * bkt) <= limit) return { tpr, bkt };
  }
  throw new Error(`attention: head dim ${D} does not fit in ${limit} bytes of workgroup memory`);
}

/** Token attention with pair bias [H, N, N] and token mask [N]. */
export function attentionOp(rt, scope, {
  qkv, ld, offQ, offK, offV, offG = 0, out, ldo, nTok, nSample, numHeads, headDim, bias, tokenMask, gating = true, label = 'attention',
}) {
  const bq = 16;
  const { tpr, bkt } = attnTiles(rt, headDim, bq, 32);
  const k = attentionKernel({ headDim, bq, bkt, tpr, gating });
  const params = { nRows: nTok, nKeys: nTok, ld, offQ, offK, offV, offG, ldo, nq: bq, pad: 0, scale: 1 / Math.sqrt(headDim) };
  const wg = [Math.ceil(nTok / bq), numHeads, nSample];
  return rt.op(scope, k, params, { QKV: qkv, BIAS: bias, MASK: tokenMask, OUT: out }, wg, 4 * nSample * nTok * nTok * numHeads * headDim, label);
}

/**
 * Local atom attention: query blocks of numQueries atoms attend to the
 * numKeys-wide zero-padded key window centred on the block. bias is
 * [H, n_subsets, numQueries, numKeys], mask is [n_subsets, numQueries, numKeys].
 */
export function localAttentionOp(rt, scope, {
  qkv, ld, offQ, offK, offV, offG = 0, out, ldo, nAtom, nSample, numHeads, headDim, numQueries, numKeys, bias, mask, gating = true, label = 'local_attention',
}) {
  if (nAtom % numQueries) throw new Error(`local attention: n_atom ${nAtom} not a multiple of ${numQueries}`);
  if (numKeys <= numQueries || (numKeys - numQueries) % 2) throw new Error('local attention: bad window');
  const { tpr, bkt } = attnTiles(rt, headDim, numQueries, 64);
  const k = attentionKernel({ headDim, bq: numQueries, bkt, tpr, local: true, gating });
  const params = {
    nRows: nAtom, nKeys: numKeys, ld, offQ, offK, offV, offG, ldo, nq: numQueries,
    pad: (numKeys - numQueries) / 2, scale: 1 / Math.sqrt(headDim),
  };
  const wg = [nAtom / numQueries, numHeads, nSample];
  return rt.op(scope, k, params, { QKV: qkv, BIAS: bias, MASK: mask, OUT: out }, wg, 4 * nSample * nAtom * numKeys * numHeads * headDim, label);
}

export function scatterMeanOp(rt, scope, { x, ldx, out, ldo, ooff = 0, C, nTok, nAtom, nSample, meta, denom }) {
  const params = { nTok, nAtom, C, ldx: ldx ?? C, ldo: ldo ?? C, ooff };
  const wg = [Math.ceil(C / 64), nTok, nSample];
  return rt.op(scope, scatterMeanKernel(), params, { X: x, META: meta, DENOM: denom, OUT: out }, wg, nSample * nAtom * C, 'scatter_mean');
}

export function gatherAddOp(rt, scope, { tok, idx, q, ldq, out, C, nTok, nAtom, nSample }) {
  const total = nSample * nAtom * C;
  const { gridX, workgroups } = grid1d(total);
  const params = { total, nTok, nAtom, C, ldq: ldq ?? C, gridX };
  return rt.op(scope, gatherAddKernel(), params, { TOK: tok, IDX: idx, Q: q, OUT: out }, workgroups, total, 'gather_add');
}

export function broadcastAddOp(rt, scope, { base, vec, out, nSample, nRows, C }) {
  const total = nSample * nRows * C;
  const { gridX, workgroups } = grid1d(total);
  return rt.op(scope, broadcastAddKernel(), { total, nRows, C, gridX }, { BASE: base, VEC: vec, OUT: out }, workgroups, total, 'broadcast_add');
}

export function fourierOp(rt, scope, { edm, w, b, out, nSample, C }) {
  const total = nSample * C;
  const { gridX, workgroups } = grid1d(total);
  return rt.op(scope, fourierKernel(), { total, C, gridX }, { EDM: edm, FW: w, FB: b, OUT: out }, workgroups, total, 'fourier');
}

export function edmPrecondOp(rt, scope, { x, edm, out, prev = null, prevOut = null, nSample, perSample, sigmaData = SIGMA_DATA }) {
  const total = nSample * perSample;
  const { gridX, workgroups } = grid1d(total);
  const params = { total, perSample, gridX, invSigma: 1 / sigmaData };
  return rt.op(scope, edmPrecondKernel({ prev: !!prev }), params, { XN: x, EDM: edm, RU: out, PREV: prev, PU: prevOut }, workgroups, total, 'edm_precond');
}

export function edmCombineOp(rt, scope, { x, f, edm, out, nSample, perSample }) {
  const total = nSample * perSample;
  const { gridX, workgroups } = grid1d(total);
  return rt.op(scope, edmCombineKernel(), { total, perSample, gridX }, { XN: x, F: f, EDM: edm, OUT: out }, workgroups, total, 'edm_combine');
}

/** Per-sample EDM scalars [c_in, c_skip, c_out, t'] (float32, sigma_data = 16). */
export function edmScalars(tHat, nSample, sigmaData = SIGMA_DATA) {
  const out = new Float32Array(nSample * 4);
  for (let s = 0; s < nSample; s++) {
    const t = Math.fround(typeof tHat === 'number' ? tHat : tHat[s]);
    if (!(t > 0)) throw new Error(`t_hat must be > 0, got ${t}`);
    const d = Math.sqrt(sigmaData * sigmaData + t * t);
    out[s * 4] = 1 / d;
    out[s * 4 + 1] = (sigmaData * sigmaData) / (d * d);
    out[s * 4 + 2] = (sigmaData * t) / d;
    out[s * 4 + 3] = Math.log(t / sigmaData) / 4;
  }
  return out;
}

export const kernelOps = {
  prepareLinear,
  matmul: matmulOp,
  layerNorm: layerNormOp,
  adaLN: adaLNOp,
  attention: attentionOp,
  localAttention: localAttentionOp,
  scatterMean: scatterMeanOp,
  gatherAdd: gatherAddOp,
  broadcastAdd: broadcastAddOp,
  fourier: fourierOp,
  edmPrecond: edmPrecondOp,
  edmCombine: edmCombineOp,
};

// ---------------------------------------------------------------------------
// Weights / config
// ---------------------------------------------------------------------------

function toF32(v) {
  if (v instanceof Float32Array) return v;
  if (ArrayBuffer.isView(v) || Array.isArray(v)) return Float32Array.from(v);
  throw new Error('weight tensors must be typed arrays, arrays or { data, shape }');
}

/**
 * Weight lookup by PyTorch state_dict name relative to the module, e.g.
 * 'diffusion_transformer.blocks.0.attention.linear_q.weight'. Keys may carry
 * any prefix ('diffusion_module.', 'model.diffusion_module.', ...). Accepts a
 * Map, a plain object, or a backbone/shared loadWeightBundle() result.
 */
export function weightGetter(weights, prefix = '') {
  let map;
  if (weights instanceof Map) map = weights;
  else if (typeof weights?.get === 'function' && Array.isArray(weights.names)) map = new Map(weights.names.map((n) => [n, weights.get(n)]));
  else map = new Map(Object.entries(weights ?? {}));
  const keys = [...map.keys()];
  const cache = new Map();
  return (name, expectedLen, { optional = false } = {}) => {
    const full = prefix + name;
    if (cache.has(full)) {
      const hit = cache.get(full);
      if (expectedLen != null && hit.length !== expectedLen) {
        throw new Error(`weight ${full}: expected ${expectedLen} values, got ${hit.length}`);
      }
      return hit;
    }
    let v = map.get(full);
    if (v === undefined) {
      const hits = keys.filter((k) => k.endsWith(`.${full}`));
      if (hits.length > 1) throw new Error(`weight ${full} is ambiguous: ${hits.join(', ')}`);
      if (hits.length === 1) v = map.get(hits[0]);
    }
    if (v === undefined) {
      if (optional) return null;
      throw new Error(`missing weight ${full}`);
    }
    const data = toF32(v.data ?? v);
    if (expectedLen != null && data.length !== expectedLen) {
      throw new Error(`weight ${full}: expected ${expectedLen} values, got ${data.length}`);
    }
    cache.set(full, data);
    return data;
  };
}

export function normalizeConfig(config = {}) {
  const c = config.diffusion_module ?? config.diffusionModule ?? config;
  const tr = c.transformer ?? {};
  const cond = c.conditioning ?? {};
  const out = {
    cToken: c.cToken ?? c.c_token ?? 768,
    cAtom: c.cAtom ?? c.c_atom ?? 256,
    cS: c.cS ?? c.c_s ?? 384,
    cZ: c.cZ ?? c.c_z ?? 128,
    cTimestep: c.cTimestep ?? c.cTimestepEmbedding ?? c.c_timestep_embedding ?? cond.c_timestep_embedding ?? cond.cTimestep ?? 256,
    numBlocks: c.numBlocks ?? c.num_blocks ?? tr.num_blocks ?? tr.numBlocks ?? 16,
    numHeads: c.numHeads ?? c.num_heads ?? tr.num_heads ?? tr.numHeads ?? 16,
    sigmaData: c.sigmaData ?? c.sigma_data ?? SIGMA_DATA,
  };
  if (out.cToken % out.numHeads) throw new Error('c_token must be divisible by num_heads');
  return out;
}

const isNonZero = (a) => a && a.some((v) => v !== 0);

function prepareLayerNorm(scope, get, name, D) {
  const w = get(`${name}.weight`, D);
  const b = get(`${name}.bias`, D, { optional: true });
  return { gamma: scope.f32(w, `${name}.weight`), beta: isNonZero(b) ? scope.f32(b, `${name}.bias`) : null };
}

function prepareTransition(scope, rt, get, name, c, precision) {
  return {
    ln: prepareLayerNorm(scope, get, `${name}.layernorm`, c),
    swiglu: prepareLinear(rt, scope, { W: get(`${name}.swiglu.linear.weight`, 4 * c * c), N: 4 * c, K: c, precision, swiglu: true, label: `${name}.swiglu` }),
    out: prepareLinear(rt, scope, { W: get(`${name}.linear_no_bias.weight`, 2 * c * c), N: c, K: 2 * c, precision, label: `${name}.linear_no_bias` }),
  };
}

/**
 * Fuse the AdaLN-Zero projections of one block into a single [6C, c_s]
 * linear. Output columns: [scale_attn | gate_attn | scale_mlp | gate_mlp |
 * shift_attn | shift_mlp]; sigmoid applies to the first 4C columns.
 */
function fusedModulation(get, prefix, b, nb, C, cIn) {
  const scaleW = get(`${prefix}scale_proj.0.weight`, 2 * nb * C * cIn);
  const scaleB = get(`${prefix}scale_proj.0.bias`, 2 * nb * C);
  const shiftW = get(`${prefix}shift_proj.weight`, 2 * nb * C * cIn);
  const gateW = get(`${prefix}gate_proj.0.weight`, 2 * nb * C * cIn);
  const gateB = get(`${prefix}gate_proj.0.bias`, 2 * nb * C);
  const W = new Float32Array(6 * C * cIn);
  const bias = new Float32Array(6 * C);
  const rowsAttn = b * C;
  const rowsMlp = nb * C + b * C;
  const parts = [
    [scaleW, scaleB, rowsAttn], [gateW, gateB, rowsAttn], [scaleW, scaleB, rowsMlp], [gateW, gateB, rowsMlp],
    [shiftW, null, rowsAttn], [shiftW, null, rowsMlp],
  ];
  parts.forEach(([w, bb, r0], p) => {
    W.set(w.subarray(r0 * cIn, (r0 + C) * cIn), p * C * cIn);
    if (bb) bias.set(bb.subarray(r0, r0 + C), p * C);
  });
  return { W, bias };
}

function prepareDitBlock(rt, scope, get, prefix, C, precision) {
  const att = `${prefix}attention.`;
  const Wq = get(`${att}linear_q.weight`, C * C);
  const bq = get(`${att}linear_q.bias`, C);
  const Wk = get(`${att}linear_k.weight`, C * C);
  const Wv = get(`${att}linear_v.weight`, C * C);
  const Wg = get(`${att}gate_linear.0.weight`, C * C);
  const bg = get(`${att}gate_linear.0.bias`, C);
  const W = new Float32Array(4 * C * C);
  [Wq, Wk, Wv, Wg].forEach((w, i) => W.set(w, i * C * C));
  const bias = new Float32Array(4 * C);
  bias.set(bq, 0);
  bias.set(bg, 3 * C);
  return {
    qkvg: prepareLinear(rt, scope, { W, N: 4 * C, K: C, bias, precision, label: `${att}qkvg` }),
    out: prepareLinear(rt, scope, { W: get(`${att}linear_output.weight`, C * C), N: C, K: C, precision, label: `${att}linear_output` }),
    mlp1: prepareLinear(rt, scope, { W: get(`${prefix}mlp.0.linear.weight`, 4 * C * C), N: 4 * C, K: C, precision, swiglu: true, label: `${prefix}mlp.0` }),
    mlp2: prepareLinear(rt, scope, { W: get(`${prefix}mlp.1.weight`, 2 * C * C), N: C, K: 2 * C, precision, label: `${prefix}mlp.1` }),
  };
}

/**
 * Record one AdaLN-Zero transformer block (DiTBlock / LocalAttnTransformerBlock):
 *   a += gate_attn * Attn(shift_attn + scale_attn * LN(a))
 *   a += gate_mlp  * MLP(shift_mlp + scale_mlp * LN(a))
 */
function recordAdaLNBlock(rt, scope, ops, { blk, mod, cond, condLd, M, C, a, x, qkvg, attnO, hid, attention }) {
  ops.push(matmulOp(rt, scope, { lin: mod, x: cond, ldx: condLd, M, out: x.mod, act: 'sigmoid', sigFrom: 0, sigTo: 4 * C }));
  const m = x.mod;
  const ld = 6 * C;
  ops.push(adaLNOp(rt, scope, { x: a, out: x.buf, M, D: C, scale: { buf: m, ld, off: 0 }, shift: { buf: m, ld, off: 4 * C } }));
  ops.push(matmulOp(rt, scope, { lin: blk.qkvg, x: x.buf, M, out: qkvg, act: 'sigmoid', sigFrom: 3 * C, sigTo: 4 * C }));
  ops.push(attention());
  ops.push(matmulOp(rt, scope, { lin: blk.out, x: attnO, M, out: a, gate: { buf: m, ld, off: C }, residual: 'inplace' }));
  ops.push(adaLNOp(rt, scope, { x: a, out: x.buf, M, D: C, scale: { buf: m, ld, off: 2 * C }, shift: { buf: m, ld, off: 5 * C } }));
  ops.push(matmulOp(rt, scope, { lin: blk.mlp1, x: x.buf, M, out: hid }));
  ops.push(matmulOp(rt, scope, { lin: blk.mlp2, x: hid, M, out: a, gate: { buf: m, ld, off: 3 * C }, residual: 'inplace' }));
}

// ---------------------------------------------------------------------------
// Conditioning features
// ---------------------------------------------------------------------------

function pick(f, ...names) {
  for (const n of names) if (f[n] !== undefined && f[n] !== null) return f[n];
  return undefined;
}

function refFeatureMatrix(f, A) {
  const full = pick(f, 'refFeatures', 'ref_features', 'atomFeatures');
  if (full) {
    if (full.length !== A * N_REF_FEATURES) throw new Error(`refFeatures must be [n_atom, ${N_REF_FEATURES}]`);
    return toF32(full);
  }
  const charge = pick(f, 'refCharge', 'ref_charge');
  const element = pick(f, 'refElement', 'ref_element');
  const names = pick(f, 'refAtomNameChars', 'ref_atom_name_chars');
  if (!charge || !element || !names) {
    throw new Error('condFeatures needs refFeatures or refCharge + refElement + refAtomNameChars');
  }
  if (charge.length !== A || element.length !== A * 128 || names.length !== A * 256) {
    throw new Error('ref feature shapes must be [n_atom], [n_atom, 128], [n_atom, 256]');
  }
  const out = new Float32Array(A * N_REF_FEATURES);
  for (let a = 0; a < A; a++) {
    const o = a * N_REF_FEATURES;
    out[o] = charge[a];
    for (let k = 0; k < 128; k++) out[o + 1 + k] = element[a * 128 + k];
    for (let k = 0; k < 256; k++) out[o + 129 + k] = names[a * 256 + k];
  }
  return out;
}

/** CSR tables for average_atom_to_token over masked atoms. */
export function atomTokenTables(atomToTokenIdx, atomMask, nTok) {
  const A = atomToTokenIdx.length;
  const counts = new Uint32Array(nTok);
  const valid = (a) => (atomMask ? atomMask[a] : 1) && atomToTokenIdx[a] >= 0 && atomToTokenIdx[a] < nTok;
  for (let a = 0; a < A; a++) if (valid(a)) counts[atomToTokenIdx[a]]++;
  const meta = new Uint32Array(nTok + 1 + A);
  for (let t = 0; t < nTok; t++) meta[t + 1] = meta[t] + counts[t];
  const fill = meta.slice(0, nTok);
  for (let a = 0; a < A; a++) if (valid(a)) meta[nTok + 1 + fill[atomToTokenIdx[a]]++] = a;
  // Reference `artios.model.utils.average_atom_to_token` calls scatter_sum on
  // `atom_mask` (torch.bool), which saturates to 1 for any occupied token.
  // The released weights were trained with this divisor ((occupied ? 1 : 0) + 1e-4).
  const denom = new Float32Array(nTok);
  for (let t = 0; t < nTok; t++) denom[t] = (counts[t] > 0 ? 1 : 0) + 1e-4;
  const idx = new Uint32Array(A);
  for (let a = 0; a < A; a++) idx[a] = Math.min(Math.max(atomToTokenIdx[a] | 0, 0), nTok - 1);
  return { meta, denom, idx };
}

// ---------------------------------------------------------------------------
// ArtiosBackbone GPU engine
// ---------------------------------------------------------------------------

let nextId = 1;

class ArtiosBackboneGpu {
  constructor({ device, weights, config, precision = 'fp32', pacing = {}, maxCachedConditions = 2, condChunkBytes = 64 << 20, ownsDevice = false }) {
    this.cfg = normalizeConfig(config);
    this.precision = precision;
    this.device = device;
    this.ownsDevice = ownsDevice;
    this.rt = new GpuRuntime(device, pacing);
    this.maxCachedConditions = maxCachedConditions;
    this.condChunkBytes = condChunkBytes;
    this.conds = [];
    this.condByFeatures = new WeakMap();
    this.plans = new Map();
    this.scratch = new Map();
    this.lastScratch = null;
    this.chain = Promise.resolve();
    this.destroyed = false;
    this.lastStepMs = null;
    this._loadWeights(weightGetter(weights));
  }

  _loadWeights(get) {
    const { cToken: C, cAtom: cA, cS, cZ, cTimestep: cT, numBlocks: nb, numHeads: nh } = this.cfg;
    const rt = this.rt;
    const scope = rt.scope('weights');
    const p = this.precision;
    const lin = (name, N, K, extra = {}) => prepareLinear(rt, scope, { W: get(name, N * K), N, K, precision: p, label: name, ...extra });
    const dc = 'diffusion_conditioning.';
    const dt = 'diffusion_transformer.';
    const W = {};
    W.rEmbed = lin('linear_no_bias_r_embed.weight', C, 3);
    const prevW = get('linear_no_bias_prev_pred_embed.weight', C * 3, { optional: true });
    const mergeW = get('self_cond_merge.weight', C * 2 * C, { optional: true });
    if (prevW && mergeW) {
      W.prevEmbed = prepareLinear(rt, scope, { W: prevW, N: C, K: 3, precision: p, label: 'prev_pred_embed' });
      W.selfCondMerge = prepareLinear(rt, scope, { W: mergeW, N: C, K: 2 * C, precision: p, label: 'self_cond_merge' });
    }
    W.fourierW = scope.f32(get(`${dc}fourier_embedding.w`, cT), 'fourier.w');
    W.fourierB = scope.f32(get(`${dc}fourier_embedding.b`, cT), 'fourier.b');
    W.lnT = prepareLayerNorm(scope, get, `${dc}layernorm_t`, cT);
    W.linT = lin(`${dc}linear_t.weight`, cS, cT);
    W.trS1 = prepareTransition(scope, rt, get, `${dc}transition_s1`, cS, p);
    W.trS2 = prepareTransition(scope, rt, get, `${dc}transition_s2`, cS, p);
    W.lnS = prepareLayerNorm(scope, get, 'layernorm_s', cS);
    W.linS = lin('linear_no_bias_s.weight', C, cS);
    W.condLn = prepareLayerNorm(scope, get, `${dt}condition_layernorm`, cS);
    W.zLn = prepareLayerNorm(scope, get, `${dt}z_layernorm`, cZ);
    W.zProj = lin(`${dt}z_proj.weight`, nh * nb, cZ);
    W.blocks = [];
    for (let b = 0; b < nb; b++) {
      const blk = prepareDitBlock(rt, scope, get, `${dt}blocks.${b}.`, C, p);
      const m = fusedModulation(get, dt, b, nb, C, cS);
      blk.mod = prepareLinear(rt, scope, { W: m.W, N: 6 * C, K: cS, bias: m.bias, precision: p, label: `${dt}blocks.${b}.adaln` });
      W.blocks.push(blk);
    }
    W.lnA = prepareLayerNorm(scope, get, 'layernorm_a', C);
    W.linA = lin('linear_no_bias_a.weight', cA, C);
    W.linF = lin('linear_no_bias_f.weight', cA, N_REF_FEATURES);
    W.readoutTr = prepareTransition(scope, rt, get, 'readout_transition', cA, p);
    W.readoutLn = prepareLayerNorm(scope, get, 'readout_layernorm', cA);
    W.readout = lin('readout_linear.weight', 3, cA);
    this.W = W;
    this.weightScope = scope;
  }

  _assertAlive() {
    if (this.destroyed) throw new Error('ArtiosBackboneGpu: engine has been destroyed');
  }

  _serial(fn) {
    const run = this.chain.then(() => {
      this._assertAlive();
      return fn();
    });
    this.chain = run.catch(() => {});
    return run;
  }

  prepareConditioning(features) {
    return this._serial(() => this._prepareConditioning(features));
  }

  async _prepareConditioning(features) {
    const { cS, cZ, cAtom: cA, numBlocks: nb, numHeads: nh } = this.cfg;
    const rt = this.rt;
    if (!features) throw new Error('condFeatures are required');
    // A backbone/cpu conditioningCache() result: { sBase, z, features }.
    const f = features.features ? { ...features.features, ...features } : features;
    const sBaseRaw = pick(f, 'sBase', 's_base');
    const zRaw = pick(f, 'z');
    const idxRaw = pick(f, 'atomToTokenIdx', 'atom_to_token_idx');
    if (!sBaseRaw || !zRaw || !idxRaw) throw new Error('condFeatures needs sBase, z and atomToTokenIdx');
    const sBase = toF32(sBaseRaw);
    const z = toF32(zRaw);
    const N = sBase.length / cS;
    if (!Number.isInteger(N) || N < 1) throw new Error(`sBase must be [n_token, ${cS}]`);
    if (z.length !== N * N * cZ) throw new Error(`z must be [${N}, ${N}, ${cZ}]`);
    const A = idxRaw.length;
    const tokenMask = Float32Array.from(pick(f, 'tokenMask', 'token_mask') ?? new Float32Array(N).fill(1), Number);
    const atomMask = pick(f, 'atomMask', 'atom_mask') ?? new Float32Array(A).fill(1);
    if (tokenMask.length !== N || atomMask.length !== A) throw new Error('tokenMask / atomMask length mismatch');
    const refFeat = refFeatureMatrix(f, A);

    const id = nextId++;
    const scope = rt.scope(`cond${id}`);
    const tables = atomTokenTables(idxRaw, atomMask, N);
    const cond = {
      id, N, A, scope, tokenMaskHost: tokenMask, released: false,
      sBase: scope.f32(sBase, 'sBase'),
      tokenMask: scope.f32(tokenMask, 'tokenMask'),
      meta: scope.u32(tables.meta, 'atomTokenMeta'),
      denom: scope.f32(tables.denom, 'atomTokenDenom'),
      idx: scope.u32(tables.idx, 'atomToTokenIdx'),
      atomFeat: scope.empty(A * cA, 'atomFeat'),
      pairBias: Array.from({ length: nb }, (_, b) => scope.empty(nh * N * N, `pairBias${b}`)),
    };

    const tmp = rt.scope(`cond${id}-tmp`);
    try {
      const W = this.W;
      const refBuf = tmp.f32(refFeat, 'refFeat');
      await rt.run([matmulOp(rt, tmp, { lin: W.linF, x: refBuf, M: A, out: cond.atomFeat })]);
      // Pair bias for every block: z * pair_mask -> z_layernorm -> z_proj,
      // stored transposed as [heads, N, N] per block. Chunked by query token
      // rows so huge N never needs one giant binding.
      const maxBytes = Math.min(this.condChunkBytes, this.device.limits.maxStorageBufferBindingSize);
      const tokPerChunk = Math.max(1, Math.min(N, Math.floor(maxBytes / (N * cZ * 4))));
      const zc = tmp.empty(tokPerChunk * N * cZ, 'zChunk');
      const zl = tmp.empty(tokPerChunk * N * cZ, 'zLn');
      const pm = tmp.empty(tokPerChunk * N, 'pairMask');
      for (let i0 = 0; i0 < N; i0 += tokPerChunk) {
        const i1 = Math.min(N, i0 + tokPerChunk);
        const rows = (i1 - i0) * N;
        const pmHost = new Float32Array(rows);
        for (let i = i0; i < i1; i++) for (let j = 0; j < N; j++) pmHost[(i - i0) * N + j] = tokenMask[i] * tokenMask[j];
        this.device.queue.writeBuffer(zc, 0, z, i0 * N * cZ, rows * cZ);
        this.device.queue.writeBuffer(pm, 0, pmHost);
        const ops = [layerNormOp(rt, tmp, { x: zc, out: zl, M: rows, D: cZ, gamma: W.zLn.gamma, beta: W.zLn.beta, rowScale: pm })];
        for (let b = 0; b < nb; b++) {
          ops.push(matmulOp(rt, tmp, {
            lin: W.zProj, x: zl, M: rows, wrow: b * nh, n: nh, out: cond.pairBias[b], transposeOut: true, ldo: N * N, ooff: i0 * N,
          }));
        }
        await rt.run(ops);
      }
      await this.device.queue.onSubmittedWorkDone();
    } finally {
      tmp.destroy();
    }

    this.conds.push(cond);
    while (this.conds.length > this.maxCachedConditions) this._releaseCond(this.conds[0]);
    return cond;
  }

  releaseConditioning(cond) {
    return this._serial(() => this._releaseCond(cond));
  }

  _releaseCond(cond) {
    if (!cond || cond.released) return;
    for (const [key, plan] of this.plans) {
      if (plan.condId === cond.id) {
        plan.scope.destroy();
        this.plans.delete(key);
      }
    }
    cond.scope.destroy();
    cond.released = true;
    this.conds = this.conds.filter((c) => c !== cond);
  }

  async _condFor(features) {
    if (features && features.pairBias && features.scope) {
      if (features.released) throw new Error('conditioning handle has been released');
      return features;
    }
    let cond = this.condByFeatures.get(features);
    if (!cond || cond.released) {
      cond = await this._prepareConditioning(features);
      this.condByFeatures.set(features, cond);
    }
    this.conds = [...this.conds.filter((c) => c !== cond), cond];
    return cond;
  }

  _scratchFor(S, A, N) {
    const key = `${S}:${A}:${N}`;
    let sc = this.scratch.get(key);
    if (!sc) {
      const scope = this.rt.scope(`scratch${key}`);
      const bufs = new Map();
      sc = {
        key, scope, bufs,
        get: (name, nFloats) => {
          let b = bufs.get(name);
          if (!b) {
            b = scope.empty(nFloats, name);
            b.nFloats = nFloats;
            bufs.set(name, b);
          }
          return b;
        },
        staging: scope.staging(S * A * 3 * 4, 'staging'),
      };
      this.scratch.set(key, sc);
      while (this.scratch.size > 2) {
        const [oldKey, old] = this.scratch.entries().next().value;
        for (const [pk, plan] of this.plans) {
          if (plan.scratchKey === oldKey) {
            plan.scope.destroy();
            this.plans.delete(pk);
          }
        }
        old.scope.destroy();
        this.scratch.delete(oldKey);
      }
    } else {
      this.scratch.delete(key);
      this.scratch.set(key, sc);
    }
    return sc;
  }

  _planFor(cond, S, withPrev) {
    const key = `${cond.id}:${S}:${withPrev ? 1 : 0}`;
    const sc = this._scratchFor(S, cond.A, cond.N);
    let plan = this.plans.get(key);
    if (!plan) {
      plan = this._buildPlan(cond, S, withPrev, sc);
      this.plans.set(key, plan);
    }
    return plan;
  }

  _buildPlan(cond, S, withPrev, sc) {
    const { cToken: C, cAtom: cA, cS, cTimestep: cT, numBlocks: nb, numHeads: nh, sigmaData } = this.cfg;
    const { N, A } = cond;
    const rt = this.rt;
    const W = this.W;
    const scope = rt.scope(`plan${cond.id}:${S}`);
    const SA = S * A;
    const SN = S * N;
    const D = C / nh;
    const b = sc.get;
    const io = {
      xIn: b('xIn', SA * 3),
      edm: b('edm', S * 4),
      out: b('out', SA * 3),
      prevIn: withPrev ? b('prevIn', SA * 3) : null,
      staging: sc.staging,
    };
    const rUnit = b('rUnit', SA * 3);
    const qSkip = b('qSkip', SA * C);
    const aTok = b('aTok', SN * C);
    const s = b('s', SN * cS);
    const sLn = b('sLn', SN * cS);
    const sHid = b('sHid', SN * 2 * cS);
    const tmRows = new Float32Array(SN);
    for (let r = 0; r < SN; r++) tmRows[r] = cond.tokenMaskHost[r % N];
    const tm = scope.f32(tmRows, 'tokenMaskRows');
    const ops = [];

    // Atom encoder
    if (withPrev) {
      const prevUnit = b('prevUnit', SA * 3);
      const qCat = b('qCat', SA * 2 * C);
      ops.push(edmPrecondOp(rt, scope, { x: io.xIn, edm: io.edm, out: rUnit, prev: io.prevIn, prevOut: prevUnit, nSample: S, perSample: A * 3, sigmaData }));
      ops.push(matmulOp(rt, scope, { lin: W.rEmbed, x: rUnit, M: SA, out: qCat, ldo: 2 * C, ooff: 0 }));
      ops.push(matmulOp(rt, scope, { lin: W.prevEmbed, x: prevUnit, M: SA, out: qCat, ldo: 2 * C, ooff: C }));
      ops.push(matmulOp(rt, scope, { lin: W.selfCondMerge, x: qCat, M: SA, out: qSkip }));
    } else {
      ops.push(edmPrecondOp(rt, scope, { x: io.xIn, edm: io.edm, out: rUnit, nSample: S, perSample: A * 3, sigmaData }));
      ops.push(matmulOp(rt, scope, { lin: W.rEmbed, x: rUnit, M: SA, out: qSkip }));
    }
    ops.push(scatterMeanOp(rt, scope, { x: qSkip, out: aTok, C, nTok: N, nAtom: A, nSample: S, meta: cond.meta, denom: cond.denom }));

    // Timestep conditioning of the single stream
    const four = b('fourier', S * cT);
    const tLn = b('tLn', S * cT);
    const tVec = b('tVec', S * cS);
    ops.push(fourierOp(rt, scope, { edm: io.edm, w: W.fourierW, b: W.fourierB, out: four, nSample: S, C: cT }));
    ops.push(layerNormOp(rt, scope, { x: four, out: tLn, M: S, D: cT, ...W.lnT }));
    ops.push(matmulOp(rt, scope, { lin: W.linT, x: tLn, M: S, out: tVec }));
    ops.push(broadcastAddOp(rt, scope, { base: cond.sBase, vec: tVec, out: s, nSample: S, nRows: N, C: cS }));
    for (const tr of [W.trS1, W.trS2]) {
      ops.push(layerNormOp(rt, scope, { x: s, out: sLn, M: SN, D: cS, ...tr.ln }));
      ops.push(matmulOp(rt, scope, { lin: tr.swiglu, x: sLn, M: SN, out: sHid }));
      ops.push(matmulOp(rt, scope, { lin: tr.out, x: sHid, M: SN, out: s, residual: 'inplace' }));
    }
    ops.push(layerNormOp(rt, scope, { x: s, out: sLn, M: SN, D: cS, ...W.lnS }));
    ops.push(matmulOp(rt, scope, { lin: W.linS, x: sLn, M: SN, out: aTok, residual: 'inplace', rowScale: tm }));

    // Diffusion transformer
    ops.push(layerNormOp(rt, scope, { x: s, out: sLn, M: SN, D: cS, ...W.condLn }));
    const mod = b('mod', SN * 6 * C);
    const xMod = b('xMod', SN * C);
    const qkvg = b('qkvg', SN * 4 * C);
    const attnO = b('attnO', SN * C);
    const hid = b('hid', SN * 2 * C);
    W.blocks.forEach((blk, i) => {
      recordAdaLNBlock(rt, scope, ops, {
        blk, mod: blk.mod, cond: sLn, condLd: cS, M: SN, C, a: aTok, x: { buf: xMod, mod }, qkvg, attnO, hid,
        attention: () => attentionOp(rt, scope, {
          qkv: qkvg, ld: 4 * C, offQ: 0, offK: C, offV: 2 * C, offG: 3 * C, out: attnO, ldo: C,
          nTok: N, nSample: S, numHeads: nh, headDim: D, bias: cond.pairBias[i], tokenMask: cond.tokenMask,
        }),
      });
    });

    // Atom decoder + readout
    const aLn = b('aLn', SN * C);
    const xAtom = b('xAtom', SA * C);
    const qAtom = b('qAtom', SA * cA);
    const qLn = b('qLn', SA * cA);
    const qHid = b('qHid', SA * 2 * cA);
    const F = b('F', SA * 3);
    ops.push(layerNormOp(rt, scope, { x: aTok, out: aLn, M: SN, D: C, ...W.lnA, rowScale: tm }));
    ops.push(gatherAddOp(rt, scope, { tok: aLn, idx: cond.idx, q: qSkip, out: xAtom, C, nTok: N, nAtom: A, nSample: S }));
    ops.push(matmulOp(rt, scope, { lin: W.linA, x: xAtom, M: SA, out: qAtom, residual: { buf: cond.atomFeat, ld: cA, off: 0, mod: A } }));
    ops.push(layerNormOp(rt, scope, { x: qAtom, out: qLn, M: SA, D: cA, ...W.readoutTr.ln }));
    ops.push(matmulOp(rt, scope, { lin: W.readoutTr.swiglu, x: qLn, M: SA, out: qHid }));
    ops.push(matmulOp(rt, scope, { lin: W.readoutTr.out, x: qHid, M: SA, out: qAtom, residual: 'inplace' }));
    ops.push(layerNormOp(rt, scope, { x: qAtom, out: qLn, M: SA, D: cA, ...W.readoutLn }));
    ops.push(matmulOp(rt, scope, { lin: W.readout, x: qLn, M: SA, out: F }));
    ops.push(edmCombineOp(rt, scope, { x: io.xIn, f: F, edm: io.edm, out: io.out, nSample: S, perSample: A * 3 }));

    return { scope, ops, io, condId: cond.id, scratchKey: sc.key, S, A, N, withPrev };
  }

  denoiseStep(xNoisy, tHat, condFeatures, options = {}) {
    return this._serial(() => this._denoiseStep(xNoisy, tHat, condFeatures, options));
  }

  async _denoiseStep(xNoisy, tHat, condFeatures, { prevPred = null } = {}) {
    const t0 = performance.now();
    const cond = await this._condFor(condFeatures);
    const A = cond.A;
    const S = xNoisy.length / (A * 3);
    if (!Number.isInteger(S) || S < 1) throw new Error(`xNoisy must be [n_sample, ${A}, 3]`);
    if (typeof tHat !== 'number' && tHat.length !== S) throw new Error(`tHat must be a number or have ${S} entries`);
    if (prevPred && prevPred.length !== xNoisy.length) throw new Error('prevPred must match xNoisy shape');
    if (prevPred && !this.W.prevEmbed) throw new Error('self-conditioning weights not loaded');
    const plan = this._planFor(cond, S, !!prevPred);
    const q = this.device.queue;
    q.writeBuffer(plan.io.xIn, 0, toF32(xNoisy));
    if (prevPred) q.writeBuffer(plan.io.prevIn, 0, toF32(prevPred));
    q.writeBuffer(plan.io.edm, 0, edmScalars(tHat, S, this.cfg.sigmaData));
    const bytes = S * A * 3 * 4;
    await this.rt.run(plan.ops, { readbacks: [{ src: plan.io.out, dst: plan.io.staging, bytes }] });
    await plan.io.staging.mapAsync(MAP_READ);
    const result = new Float32Array(plan.io.staging.getMappedRange().slice(0, bytes));
    plan.io.staging.unmap();
    this.lastScratch = this.scratch.get(plan.scratchKey);
    this.lastStepMs = performance.now() - t0;
    return result;
  }

  /** Read a named scratch buffer of the most recent step (debugging / tests). */
  debugRead(name) {
    return this._serial(async () => {
      const b = this.lastScratch?.bufs.get(name);
      if (!b) throw new Error(`no scratch buffer "${name}" from the last step`);
      return this.rt.read(b, b.nFloats);
    });
  }

  stats() {
    return {
      liveBuffers: this.rt.liveBuffers(),
      liveBytes: this.rt.liveBytes(),
      weightBytes: this.weightScope?.bytes ?? 0,
      pipelines: this.rt.pipelines.size,
      plans: this.plans.size,
      conditions: this.conds.length,
      submits: this.rt.counters.submits,
      dispatches: this.rt.counters.dispatches,
      dispatchesPerStep: [...this.plans.values()].at(-1)?.ops.length ?? 0,
      lastStepMs: this.lastStepMs,
    };
  }

  shaderMessages() {
    return this.rt.shaderMessages();
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.rt.destroy();
    this.plans.clear();
    this.scratch.clear();
    this.conds = [];
    this.W = null;
    this.weightScope = null;
    if (this.ownsDevice) this.device.destroy();
  }
}

async function requestBackboneDevice() {
  if (!globalThis.navigator?.gpu) throw new Error('WebGPU is not available');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const want = ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize', 'maxStorageBuffersPerShaderStage'];
  const requiredLimits = {};
  for (const k of want) requiredLimits[k] = adapter.limits[k];
  return adapter.requestDevice({ requiredLimits });
}

/**
 * Create the GPU backbone.
 *
 * @param {object}   opts
 * @param {GPUDevice} [opts.device]   device to use (one is requested if omitted
 *                                    and then owned / destroyed by the engine)
 * @param {Map|object} opts.weights   PyTorch state_dict tensors of
 *                                    DiffusionModule (Float32Array or {data}),
 *                                    or a backbone/shared loadWeightBundle()
 * @param {object}   [opts.config]    DiffusionModuleConfig (snake or camel case)
 * @param {'fp32'|'fp16'|'int8'} [opts.precision='fp32']  linear weight storage
 * @param {object}   [opts.pacing]    { submitBudgetFlops, maxDispatchesPerSubmit, maxInFlight }
 * @param {number}   [opts.maxCachedConditions=2]  conditionings kept on the GPU (LRU)
 * @param {number}   [opts.condChunkBytes=64 MiB]  z rows processed per pair-bias chunk
 *
 * denoiseStep(xNoisy [S*A*3], tHat (number | [S]), condFeatures, { prevPred })
 * returns the EDM denoised coordinates r_pred [S*A*3]. condFeatures holds the
 * cached DiffusionConditioning outputs and atom layout:
 *   sBase [N*c_s], z [N*N*c_z] (post transition_z), atomToTokenIdx [A],
 *   tokenMask [N], atomMask [A], refCharge [A], refElement [A*128],
 *   refAtomNameChars [A*256] (or refFeatures [A*385]); snake_case keys and a
 *   backbone/cpu conditioningCache() result ({ sBase, z, features }) work too.
 * The same features object (or a prepareConditioning() handle) reuses all
 * timestep-independent GPU work across steps.
 */
export async function createArtiosBackboneGpu({ device, weights, config = {}, precision = 'fp32', pacing, maxCachedConditions, condChunkBytes } = {}) {
  let ownsDevice = false;
  if (!device) {
    device = await requestBackboneDevice();
    ownsDevice = true;
  }
  const engine = new ArtiosBackboneGpu({ device, weights, config, precision, pacing, maxCachedConditions, condChunkBytes, ownsDevice });
  return {
    config: engine.cfg,
    precision,
    device,
    denoiseStep: (xNoisy, tHat, condFeatures, options) => engine.denoiseStep(xNoisy, tHat, condFeatures, options),
    prepareConditioning: (features) => engine.prepareConditioning(features),
    releaseConditioning: (handle) => engine.releaseConditioning(handle),
    debugRead: (name) => engine.debugRead(name),
    shaderMessages: () => engine.shaderMessages(),
    stats: () => engine.stats(),
    destroy: () => engine.destroy(),
    get destroyed() {
      return engine.destroyed;
    },
  };
}

// ---------------------------------------------------------------------------
// LocalAttnTransformer (transformer.py) - windowed atom attention tower.
// Not part of DiffusionModule's forward; provided for atom-level encoders.
// ---------------------------------------------------------------------------

export class LocalAttnTransformerGpu {
  constructor(rt, { weights, prefix = '', config, precision = 'fp32' }) {
    this.rt = rt;
    this.cfg = {
      numQueries: 32,
      numKeys: 128,
      ...config,
    };
    const { cAtom: C, cCond, cAtomPair, numBlocks: nb, numHeads: nh } = this.cfg;
    if (C % nh) throw new Error('c_atom must be divisible by num_heads');
    const get = weightGetter(weights, prefix);
    const scope = rt.scope('local-weights');
    this.scope = scope;
    this.condLn = prepareLayerNorm(scope, get, 'condition_layernorm', cCond);
    this.pLn = prepareLayerNorm(scope, get, 'p_layernorm', cAtomPair);
    this.pProj = prepareLinear(rt, scope, { W: get('p_proj.weight', nh * nb * cAtomPair), N: nh * nb, K: cAtomPair, precision, label: 'p_proj' });
    this.blocks = [];
    for (let b = 0; b < nb; b++) {
      const blk = prepareDitBlock(rt, scope, get, `blocks.${b}.`, C, precision);
      const m = fusedModulation(get, '', b, nb, C, cCond);
      blk.mod = prepareLinear(rt, scope, { W: m.W, N: 6 * C, K: cCond, bias: m.bias, precision, label: `blocks.${b}.adaln` });
      this.blocks.push(blk);
    }
  }

  /**
   * q [S*A*c_atom], c [S*A*c_cond]; p [n_sub*NQ*NK*c_atompair] and
   * mask [n_sub*NQ*NK] are shared across the S samples.
   */
  async run({ q, c, p, mask, nSample = 1 }) {
    const { cAtom: C, cCond, cAtomPair, numBlocks: nb, numHeads: nh, numQueries: NQ, numKeys: NK } = this.cfg;
    const rt = this.rt;
    const S = nSample;
    const A = q.length / (S * C);
    const nSub = A / NQ;
    const M = S * A;
    const nPair = nSub * NQ * NK;
    const scope = rt.scope('local-run');
    try {
      const a = scope.f32(q, 'q');
      const cIn = scope.f32(c, 'c');
      const pIn = scope.f32(p, 'p');
      const maskBuf = scope.f32(mask, 'mask');
      const cLn = scope.empty(M * cCond, 'cLn');
      const pLn = scope.empty(nPair * cAtomPair, 'pLn');
      const bias = Array.from({ length: nb }, (_, b) => scope.empty(nh * nPair, `bias${b}`));
      const x = { buf: scope.empty(M * C, 'x'), mod: scope.empty(M * 6 * C, 'mod') };
      const qkvg = scope.empty(M * 4 * C, 'qkvg');
      const attnO = scope.empty(M * C, 'attnO');
      const hid = scope.empty(M * 2 * C, 'hid');
      const ops = [
        layerNormOp(rt, scope, { x: cIn, out: cLn, M, D: cCond, ...this.condLn }),
        layerNormOp(rt, scope, { x: pIn, out: pLn, M: nPair, D: cAtomPair, ...this.pLn }),
      ];
      for (let b = 0; b < nb; b++) {
        ops.push(matmulOp(rt, scope, { lin: this.pProj, x: pLn, M: nPair, wrow: b * nh, n: nh, out: bias[b], transposeOut: true, ldo: nPair }));
      }
      this.blocks.forEach((blk, i) => {
        recordAdaLNBlock(rt, scope, ops, {
          blk, mod: blk.mod, cond: cLn, condLd: cCond, M, C, a, x, qkvg, attnO, hid,
          attention: () => localAttentionOp(rt, scope, {
            qkv: qkvg, ld: 4 * C, offQ: 0, offK: C, offV: 2 * C, offG: 3 * C, out: attnO, ldo: C,
            nAtom: A, nSample: S, numHeads: nh, headDim: C / nh, numQueries: NQ, numKeys: NK, bias: bias[i], mask: maskBuf,
          }),
        });
      });
      await rt.run(ops);
      return await rt.read(a, M * C);
    } finally {
      scope.destroy();
    }
  }

  destroy() {
    this.scope.destroy();
  }
}
