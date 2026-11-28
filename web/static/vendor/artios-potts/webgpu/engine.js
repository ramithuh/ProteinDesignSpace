// WebGPU engine for the ArtiosPotts Potts-parameter head
// (artios/model/artios_potts.py :: ArtiosPotts.compute_potts_params).
//
// One computePottsParams() = the sequence-blind MPNN forward plus the
// factorised GraphPotts head:
//   featurizer   k-NN graph over centre atoms -> edge masks
//                node: ligand_condition_embed(atom feats) averaged atom->token
//                      + entity_type_embed + non_protein_restype_embed
//                      -> node_norm -> * token_mask
//                edge: GaussianRBF(5x5 backbone distances) (+) relpos one-hots
//                      -> embed_edge -> edge_norm -> * mask[j]
//   encoder      num_encoder_layers x (SwiGLU node message -> sum/scale ->
//                norm1; SwiGLU dense -> norm2 * node_mask; SwiGLU edge update
//                -> norm3 * edge_mask)
//   decoder      num_decoder_layers x (SwiGLU node message over
//                [h_i | e_ij | seq_j | h_j] -> sum/scale -> norm1;
//                SwiGLU dense -> norm2 * node_mask)
//   potts head   h = zero_sum_gauge(scale * token_mask * W_h(h_dec))
//                J = zero_sum_gauge((s L)(s R)), s = scale * edge_mask,
//                then J = 0.5 * edge_mask * has_reverse * (J_ij + J_ji^T)
//
// The decoder input sequence is fully masked (ArtiosPotts is sequence-blind),
// so embed_restype collapses to token_mask[n] * W[:, mask_token] and is
// uploaded as a plain [n_token, node_dim] buffer.
//
// Per-shape plans (scratch buffers, uniforms, bind groups, the whole dispatch
// list) are built once and replayed; a repeat call on the same shape only
// writes the input buffers and resubmits, so warm calls skip all validation
// and pipeline work.

import {
  ELEMWISE_WG,
  LINEAR_TILE,
  MAX_GRID,
  aggregateShader,
  edgeInputShader,
  edgeMasksShader,
  gatherConcatShader,
  knnShader,
  layerNormShader,
  linearShader,
  nodeInputShader,
  pottsCouplingShader,
  pottsFieldsShader,
  pottsSymmetrizeShader,
} from './shaders.js';
import {
  normalizeConfig,
  normalizePrecision,
  packLinearWeight,
  resolveWeights,
} from './weights.js';

const BU = globalThis.GPUBufferUsage ?? {
  MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80,
};
const MAP_READ = globalThis.GPUMapMode?.READ ?? 0x1;
const STAGE_COMPUTE = globalThis.GPUShaderStage?.COMPUTE ?? 0x4;
const STORAGE_USAGE = BU.STORAGE | BU.COPY_DST | BU.COPY_SRC;
const BIND_TYPE = { uniform: 'uniform', read: 'read-only-storage', rw: 'storage' };

/** Fixed upstream widths: non_protein_restype_embed(32), ref_element(128), 5 backbone atoms. */
export const NUM_NON_PROTEIN_RESTYPES = 32;
export const NUM_BACKBONE_ATOMS = 5;
export const DEFAULT_PACING = Object.freeze({
  submitBudgetFlops: 16e9,
  maxDispatchesPerSubmit: 64,
  maxInFlight: 2,
  chunkBudgetBytes: 64 << 20,
});

const roundUp = (v, m) => Math.ceil(v / m) * m;

// ---------------------------------------------------------------------------
// Buffer ownership: every GPU buffer belongs to exactly one scope, and
// destroy() walks the scopes, so liveBuffers() is 0 after a clean teardown.
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
      size, usage, label: `${this.label}:${label}`, mappedAtCreation: !!init,
    });
    if (init) {
      const range = b.getMappedRange();
      const View = init instanceof Uint32Array ? Uint32Array : init instanceof Int32Array ? Int32Array : Float32Array;
      new View(range).set(init);
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
    const a = data instanceof Float32Array ? data : Float32Array.from(data);
    return this.raw(a.byteLength, STORAGE_USAGE, label, a);
  }

  u32(data, label) {
    const a = data instanceof Uint32Array ? data : Uint32Array.from(data);
    return this.raw(a.byteLength, STORAGE_USAGE, label, a);
  }

  i32(data, label) {
    const a = data instanceof Int32Array ? data : Int32Array.from(data);
    return this.raw(a.byteLength, STORAGE_USAGE, label, a);
  }

  /** Params uniform for a kernel spec: `params` is [[name, 'u32'|'f32'], ...]. */
  uniform(params, values, label = 'params') {
    const u = new Uint32Array(params.length);
    const f = new Float32Array(u.buffer);
    params.forEach(([name, type], i) => {
      const v = values[name] ?? 0;
      if (type === 'f32') {
        f[i] = v;
      } else {
        if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) throw Error(`uniform '${name}' must be a u32, got ${v}`);
        u[i] = v;
      }
    });
    return this.raw(u.byteLength, BU.UNIFORM | BU.COPY_DST, label, u);
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

/** Scratch recycler. With `reuse` off every request is a fresh buffer, which
 *  keeps every intermediate alive for tracing. */
class Pool {
  constructor(scope, reuse) {
    this.scope = scope;
    this.reuse = reuse;
    this.free = new Map();
  }

  get(nFloats, label) {
    if (this.reuse) {
      const list = this.free.get(nFloats);
      if (list?.length) return list.pop();
    }
    return this.scope.empty(nFloats, label);
  }

  put(buffer, nFloats) {
    if (!this.reuse || !buffer) return;
    let list = this.free.get(nFloats);
    if (!list) this.free.set(nFloats, (list = []));
    list.push(buffer);
  }
}

// ---------------------------------------------------------------------------
// Runtime: kernel/pipeline cache, op records, paced submission
// ---------------------------------------------------------------------------

export class GpuRuntime {
  constructor(device, opts = {}) {
    if (!device) throw Error('GpuRuntime: a GPUDevice is required');
    this.device = device;
    this.kernels = new Map();
    this.pipelines = new Map();
    this.scopes = new Set();
    this.submitBudgetFlops = opts.submitBudgetFlops ?? DEFAULT_PACING.submitBudgetFlops;
    this.maxDispatchesPerSubmit = opts.maxDispatchesPerSubmit ?? DEFAULT_PACING.maxDispatchesPerSubmit;
    this.maxInFlight = Math.max(1, opts.maxInFlight ?? DEFAULT_PACING.maxInFlight);
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

  /** Memoised kernel spec; `key` must capture every code-generating option. */
  kernel(key, factory) {
    let k = this.kernels.get(key);
    if (!k) this.kernels.set(key, (k = factory()));
    return k;
  }

  pipeline(key, kernel) {
    let p = this.pipelines.get(key);
    if (p) return p;
    const dev = this.device;
    const layout = dev.createBindGroupLayout({
      label: key,
      entries: kernel.bindings.map((b, i) => ({
        binding: i, visibility: STAGE_COMPUTE, buffer: { type: BIND_TYPE[b.kind] },
      })),
    });
    const module = dev.createShaderModule({ label: key, code: kernel.code });
    const pipe = dev.createComputePipeline({
      label: key,
      layout: dev.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'main' },
    });
    p = { pipe, layout, module };
    this.pipelines.set(key, p);
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

  /** Record one dispatch. The params uniform is baked into `scope` once. */
  op(scope, key, kernel, params, buffers, workgroups, cost = 0, label = key) {
    const p = this.pipeline(key, kernel);
    const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (workgroups.some((w) => w > limit)) throw Error(`${label}: dispatch ${workgroups} exceeds ${limit}`);
    const ub = scope.uniform(kernel.params, params, `${label}:params`);
    const entries = kernel.bindings.map((b, i) => {
      const buf = b.kind === 'uniform' ? ub : buffers[b.name];
      if (!buf) throw Error(`${label}: missing binding '${b.name}'`);
      return { binding: i, resource: { buffer: buf } };
    });
    const bindGroup = this.device.createBindGroup({ label, layout: p.layout, entries });
    return { pipe: p.pipe, bindGroup, workgroups, cost, label };
  }

  /**
   * Encode and submit `ops` in order, splitting into command buffers of at
   * most submitBudgetFlops / maxDispatchesPerSubmit and keeping at most
   * maxInFlight submissions queued, so no single submit can run long enough
   * to trip a GPU watchdog and the CPU never races far ahead.
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

  destroy() {
    for (const s of [...this.scopes]) s.destroy();
    this.scopes.clear();
    this.pipelines.clear();
    this.kernels.clear();
  }
}

// ---------------------------------------------------------------------------
// Dispatch geometry. Every kernel folds its workgroup id onto a 2D grid as
// wid.x + wid.y * MAX_GRID, which only reconstructs the right id when the x
// extent is exactly MAX_GRID (or the whole launch fits in one row).
// ---------------------------------------------------------------------------

function foldGrid(nWorkgroups) {
  const n = Math.max(1, nWorkgroups);
  const gx = Math.min(n, MAX_GRID);
  return [gx, Math.ceil(n / gx), 1];
}

const threadGrid = (threads, wg) => foldGrid(Math.ceil(Math.max(1, threads) / wg));

// ---------------------------------------------------------------------------
// Op builders (one dispatch each)
// ---------------------------------------------------------------------------

const segKey = (segments) => segments.map((s) => `${s.dim}${s.index[0]}`).join('.');

/**
 * y[m, n] = act(sum_k X[m, k] * W[n, k]) with X a concatenation of gathered
 * segments. `N` is the kernel's output width: the SwiGLU half-width H for a
 * swiglu projection (weight rows [0, H) are `a`, [H, 2H) are `b`), else the
 * full row count.
 */
function linearOp(rt, scope, {
  lin, segments, x, idx = null, out, M, N, inBase = 0, outBase = 0, Kn = 1,
  act = 'none', swiglu = false, label = lin.label,
}) {
  const key = `linear|${lin.precision}|b${+lin.hasBias}|${act}|g${+swiglu}|${segKey(segments)}`;
  const kernel = rt.kernel(key, () => linearShader({
    precision: lin.precision, segments, bias: lin.hasBias, act, swiglu, label: key,
  }));
  const buffers = { w: lin.w, aux: lin.aux, y: out, idx };
  segments.forEach((_, i) => { buffers[`x${i}`] = x[i]; });
  const params = { M, N, inBase, outBase, Kn, ldw: lin.ldw, biasOff: lin.biasOff };
  const tiles = Math.ceil(M / LINEAR_TILE.M) * Math.ceil(N / LINEAR_TILE.N);
  return rt.op(scope, key, kernel, params, buffers, foldGrid(tiles), 2 * M * N * lin.cols * (swiglu ? 2 : 1), `linear:${label}`);
}

/** Materialise [rows, sum(dims)] from gathered segments (trace only). */
function gatherConcatOp(rt, scope, { segments, x, idx = null, out, rows, inBase = 0, outBase = 0, Kn = 1, label = 'gather_concat' }) {
  const key = `gather|${segKey(segments)}`;
  const kernel = rt.kernel(key, () => gatherConcatShader({ segments, label: key }));
  const width = segments.reduce((s, g) => s + g.dim, 0);
  const buffers = { y: out, idx };
  segments.forEach((_, i) => { buffers[`x${i}`] = x[i]; });
  return rt.op(scope, key, kernel, { rows, inBase, outBase, Kn }, buffers,
    threadGrid(rows * width, ELEMWISE_WG), rows * width, label);
}

/** In place: x = (LN(x + res) * gamma + beta) * mask[row] */
function layerNormOp(rt, scope, { x, gb, res = null, mask = null, rows, D, eps, label = 'layernorm' }) {
  const key = `ln|${D}|r${+!!res}|m${+!!mask}`;
  const kernel = rt.kernel(key, () => layerNormShader({ D, residual: !!res, rowMask: !!mask, label: key }));
  return rt.op(scope, key, kernel, { rows, eps }, { x, gb, res, mask }, foldGrid(rows), 8 * rows * D, label);
}

/** out[i, d] = invScale * sum_k w[i, k] * msg[i, k, d] */
function aggregateOp(rt, scope, { msg, w, out, nodes, Kn, D, invScale, mode = 'sum', label = 'aggregate' }) {
  const key = `agg|${D}|${mode}`;
  const kernel = rt.kernel(key, () => aggregateShader({ D, mode }));
  return rt.op(scope, key, kernel, { nodes, Kn, invScale }, { msg, w, y: out },
    threadGrid(nodes * D, ELEMWISE_WG), nodes * Kn * D, label);
}

function knnOp(rt, scope, { xyz, mask, idx, N, Keff, KMAX }) {
  const key = `knn|${KMAX}`;
  const kernel = rt.kernel(key, () => knnShader({ KMAX }));
  return rt.op(scope, key, kernel, { N, Keff }, { xyz, mask, idx }, threadGrid(N, 64), 10 * N * N, 'knn');
}

function edgeMasksOp(rt, scope, { mask, idx, maskJ, edgeMask, NK, Kn }) {
  const key = 'edge_masks';
  const kernel = rt.kernel(key, () => edgeMasksShader());
  return rt.op(scope, key, kernel, { NK, Kn }, { mask, idx, maskJ, edgeMask }, threadGrid(NK, ELEMWISE_WG), NK, key);
}

function edgeInputOp(rt, scope, { bb, tok, idx, aux, out, rows, inBase, Kn, gamma, opts }) {
  const key = `edge_input|${opts.numRbf}|${opts.posDim}|${opts.rMax}|${opts.sMax}`;
  const kernel = rt.kernel(key, () => edgeInputShader(opts));
  const width = 25 * opts.numRbf + opts.posDim;
  return rt.op(scope, key, kernel, { rows, inBase, Kn, gamma }, { bb, tok, idx, aux, y: out },
    threadGrid(rows * width, ELEMWISE_WG), 20 * rows * width, 'edge_input');
}

function nodeInputOp(rt, scope, { tokf, atoms, elem, csr, w, out, N, nAtom, opts }) {
  const key = `node_input|${opts.D}|${opts.nElem}|${opts.nRes}`;
  const kernel = rt.kernel(key, () => nodeInputShader(opts));
  return rt.op(scope, key, kernel, { N }, { tokf, atoms, elem, csr, w, y: out },
    threadGrid(N * opts.D, ELEMWISE_WG), 2 * nAtom * opts.D * opts.nElem, 'node_input');
}

function pottsFieldsOp(rt, scope, { hpre, mask, out, N, R, scale, gauge }) {
  const key = `potts_fields|${R}|${+gauge}`;
  const kernel = rt.kernel(key, () => pottsFieldsShader({ R, gauge }));
  return rt.op(scope, key, kernel, { N, scale }, { hpre, mask, h: out }, threadGrid(N, 64), 4 * N * R, 'potts_fields');
}

function pottsCouplingOp(rt, scope, { lr, edgeMask, out, rows, inBase, R, F, scale, gauge }) {
  const key = `potts_coupling|${R}|${F}|${+gauge}`;
  const kernel = rt.kernel(key, () => pottsCouplingShader({ R, F, gauge }));
  return rt.op(scope, key, kernel, { rows, inBase, scale }, { lr, edgeMask, J: out },
    foldGrid(rows), 2 * rows * R * R * F, 'potts_coupling');
}

function pottsSymmetrizeOp(rt, scope, { Jin, idx, edgeMask, out, NK, Kn, R }) {
  const key = `potts_symmetrize|${R}`;
  const kernel = rt.kernel(key, () => pottsSymmetrizeShader({ R }));
  return rt.op(scope, key, kernel, { NK, Kn }, { Jin, idx, edgeMask, J: out },
    foldGrid(NK), 3 * NK * R * R, 'potts_symmetrize');
}

// ---------------------------------------------------------------------------
// Weight plumbing
// ---------------------------------------------------------------------------

/**
 * Every tensor the tiled linear kernel consumes as its `w` binding, in the
 * order the engine packs them. `swiglu` rows are [a (rows/2) | b (rows/2)].
 */
export function pottsLinearWeights(config) {
  const c = normalizeConfig(config);
  const D = c.nodeDim, E = c.edgeDim, R = c.numResidueTypes, F = c.numFactors;
  const encE = 2 * D + E;
  const decE = 2 * D + E;
  const out = [
    { name: 'mpnn.featurizer.embed_edge.weight', rows: E, cols: 25 * c.numRbfKernels + c.positionalEncodingDim },
  ];
  for (let l = 0; l < c.numEncoderLayers; l++) {
    const p = `mpnn.encoder.${l}`;
    out.push({ name: `${p}.node_message_mlp.W_in.weight`, rows: 2 * D, cols: encE, swiglu: true });
    out.push({ name: `${p}.node_message_mlp.W_out.weight`, rows: D, cols: D });
    out.push({ name: `${p}.dense.W_in.weight`, rows: 4 * D, cols: D, swiglu: true });
    out.push({ name: `${p}.dense.W_out.weight`, rows: D, cols: 2 * D });
    out.push({ name: `${p}.edge_update_mlp.W_in.weight`, rows: 2 * E, cols: encE, swiglu: true });
    out.push({ name: `${p}.edge_update_mlp.W_out.weight`, rows: E, cols: E });
  }
  for (let l = 0; l < c.numDecoderLayers; l++) {
    const p = `mpnn.decoder.${l}`;
    out.push({ name: `${p}.node_message_mlp.W_in.weight`, rows: 2 * D, cols: D + decE, swiglu: true });
    out.push({ name: `${p}.node_message_mlp.W_out.weight`, rows: D, cols: D });
    out.push({ name: `${p}.dense.W_in.weight`, rows: 4 * D, cols: D, swiglu: true });
    out.push({ name: `${p}.dense.W_out.weight`, rows: D, cols: 2 * D });
  }
  out.push({ name: 'potts.W_h.weight', rows: R, cols: D, bias: 'potts.W_h.bias' });
  out.push({ name: 'potts.W_J_left.weight', rows: R * F, cols: decE, bias: 'potts.W_J_left.bias' });
  out.push({ name: 'potts.W_J_right.weight', rows: F * R, cols: decE, bias: 'potts.W_J_right.bias' });
  out.push({ name: 'mpnn.linear_out.weight', rows: R, cols: D, optional: true });
  return out;
}

function prepareLinear(scope, { W, rows, cols, bias = null, precision, label }) {
  const packed = packLinearWeight(W, rows, cols, precision);
  const w = precision === 'f32' ? scope.f32(packed.data, label) : scope.u32(packed.data, label);
  let aux = null;
  let biasOff = 0;
  if (packed.scale || bias) {
    const data = new Float32Array((packed.scale ? rows : 0) + (bias ? rows : 0));
    if (packed.scale) data.set(packed.scale, 0);
    if (bias) {
      biasOff = packed.scale ? rows : 0;
      data.set(bias, biasOff);
    }
    aux = scope.f32(data, `${label}:aux`);
  }
  return { precision, rows, cols, ldw: packed.ldw, w, aux, hasBias: !!bias, biasOff, label };
}

/** GaussianRBF(num_kernels, min_distance, max_distance) centers / gamma. */
export function rbfParameters(config) {
  const c = normalizeConfig(config);
  const n = c.numRbfKernels;
  const centers = new Float32Array(n);
  const span = c.rbfMaxDistance - c.rbfMinDistance;
  for (let i = 0; i < n; i++) centers[i] = n > 1 ? c.rbfMinDistance + (span * i) / (n - 1) : c.rbfMinDistance;
  const spacing = span / (n - 1);
  return { centers, gamma: 1 / (spacing * spacing) };
}

// ---------------------------------------------------------------------------
// Feature preparation
// ---------------------------------------------------------------------------

function pick(f, ...names) {
  for (const n of names) if (f[n] !== undefined && f[n] !== null) return f[n];
  return undefined;
}

/** Flatten typed arrays, nested arrays, booleans and bigints into a number array. */
function flatten(v, name) {
  if (ArrayBuffer.isView(v)) return v;
  if (Array.isArray(v)) {
    if (v.length && (Array.isArray(v[0]) || ArrayBuffer.isView(v[0]))) {
      const parts = v.map((p) => flatten(p, name));
      const out = new Float64Array(parts.reduce((s, p) => s + p.length, 0));
      let o = 0;
      for (const p of parts) for (let i = 0; i < p.length; i++) out[o++] = Number(p[i]);
      return out;
    }
    return Float64Array.from(v, Number);
  }
  if (v && typeof v === 'object' && v.data !== undefined) return flatten(v.data, name);
  throw Error(`ArtiosPotts WebGPU: cannot interpret feature '${name}'`);
}

function needFeature(f, names, name) {
  const v = pick(f, ...names);
  if (v === undefined) throw Error(`ArtiosPotts WebGPU: features.${names[0]} is required`);
  return flatten(v, name ?? names[0]);
}

function sized(v, n, name) {
  if (v.length !== n) throw Error(`ArtiosPotts WebGPU: feature '${name}' has ${v.length} values, expected ${n}`);
  return v;
}

const asF32 = (v, n, name) => Float32Array.from(sized(v, n, name), Number);
const asI32 = (v, n, name) => Int32Array.from(sized(v, n, name), Number);

/**
 * Pack the ArtiosPotts FullContext slice the kernels read. Accepts camelCase
 * or upstream snake_case, Float32Array / nested array / {data} tensors, and
 * booleans for the masks.
 */
export function prepareFeatures(features, config) {
  if (!features || typeof features !== 'object') throw Error('ArtiosPotts WebGPU: `features` object is required');
  const c = normalizeConfig(config);
  const R = c.numResidueTypes;
  const BB = NUM_BACKBONE_ATOMS;
  if (R !== NUM_NON_PROTEIN_RESTYPES) {
    throw Error(`ArtiosPotts WebGPU: numResidueTypes must be ${NUM_NON_PROTEIN_RESTYPES} (got ${R})`);
  }

  const tokenMask = needFeature(features, ['tokenMask', 'token_mask']);
  const N = tokenMask.length;
  if (N < 1) throw Error('ArtiosPotts WebGPU: n_token must be >= 1');
  const centre = needFeature(features, ['centreCoordinate', 'centre_coordinate', 'centerCoordinate', 'center_coordinate']);
  const bbCoord = needFeature(features, ['backboneCoordinate', 'backbone_coordinate']);
  const bbMask = needFeature(features, ['backboneMask', 'backbone_mask']);
  const atomToToken = needFeature(features, ['atomToTokenIdx', 'atom_to_token_idx']);
  const A = atomToToken.length;
  const atomMask = needFeature(features, ['atomMask', 'atom_mask']);
  const isAtomic = needFeature(features, ['isLigandAtom', 'is_ligand_atom', 'isAtomic', 'is_atomic']);
  const refCharge = needFeature(features, ['refCharge', 'ref_charge']);
  const refMask = needFeature(features, ['refMask', 'ref_mask']);

  const zeros = (n) => new Float64Array(n);
  const optional = (names, n) => {
    const v = pick(features, ...names);
    return v === undefined ? zeros(n) : flatten(v, names[0]);
  };

  // --- per-token ---------------------------------------------------------
  const mask = asF32(tokenMask, N, 'tokenMask');
  for (let i = 0; i < N; i++) mask[i] = mask[i] ? 1 : 0;
  const xyz = asF32(centre, N * 3, 'centreCoordinate');
  const bbc = sized(bbCoord, N * BB * 3, 'backboneCoordinate');
  const bbm = sized(bbMask, N * BB, 'backboneMask');
  const bb = new Float32Array(N * BB * 4);
  for (let i = 0; i < N * BB; i++) {
    bb[i * 4] = bbc[i * 3];
    bb[i * 4 + 1] = bbc[i * 3 + 1];
    bb[i * 4 + 2] = bbc[i * 3 + 2];
    bb[i * 4 + 3] = bbm[i] ? 1 : 0;
  }
  const asym = sized(needFeature(features, ['asymId', 'asym_id']), N, 'asymId');
  const resIdx = sized(needFeature(features, ['residueIndex', 'residue_index']), N, 'residueIndex');
  const entity = sized(needFeature(features, ['entityId', 'entity_id']), N, 'entityId');
  const sym = sized(needFeature(features, ['symId', 'sym_id']), N, 'symId');
  const tokIdx = sized(needFeature(features, ['tokenIndex', 'token_index']), N, 'tokenIndex');
  const tok = new Int32Array(N * 5);
  for (let i = 0; i < N; i++) {
    tok[i * 5] = asym[i];
    tok[i * 5 + 1] = resIdx[i];
    tok[i * 5 + 2] = entity[i];
    tok[i * 5 + 3] = sym[i];
    tok[i * 5 + 4] = tokIdx[i];
  }

  let restype = pick(features, 'restype', 'restypeOneHot', 'restype_one_hot');
  if (restype === undefined) {
    const ri = pick(features, 'restypeIndex', 'restype_index');
    if (ri === undefined) throw Error('ArtiosPotts WebGPU: features.restype (or restypeIndex) is required');
    const idx = sized(flatten(ri, 'restypeIndex'), N, 'restypeIndex');
    restype = new Float64Array(N * R);
    for (let i = 0; i < N; i++) {
      const k = Number(idx[i]);
      if (!Number.isInteger(k) || k < 0 || k >= R) throw Error(`ArtiosPotts WebGPU: restypeIndex[${i}] = ${k} out of range`);
      restype[i * R + k] = 1;
    }
  } else {
    restype = sized(flatten(restype, 'restype'), N * R, 'restype');
  }

  // entity_type_embed order is (protein, ligand, dna, rna); the first two also
  // gate non_protein_restype_embed, so they are required rather than defaulted.
  const isProtein = sized(needFeature(features, ['isProteinToken', 'is_protein_token']), N, 'isProteinToken');
  const isLigand = sized(needFeature(features, ['isLigandToken', 'is_ligand_token']), N, 'isLigandToken');
  const isDna = sized(optional(['isDnaToken', 'is_dna_token'], N), N, 'isDnaToken');
  const isRna = sized(optional(['isRnaToken', 'is_rna_token'], N), N, 'isRnaToken');
  const TF = 4 + R;
  const tokf = new Float32Array(N * TF);
  for (let i = 0; i < N; i++) {
    tokf[i * TF] = isProtein[i] ? 1 : 0;
    tokf[i * TF + 1] = isLigand[i] ? 1 : 0;
    tokf[i * TF + 2] = isDna[i] ? 1 : 0;
    tokf[i * TF + 3] = isRna[i] ? 1 : 0;
    for (let r = 0; r < R; r++) tokf[i * TF + 4 + r] = Number(restype[i * R + r]);
  }

  // --- per-atom ----------------------------------------------------------
  const isMotifRelated = optional(['isMotifRelatedAtom', 'is_motif_related_atom'], A);
  sized(atomMask, A, 'atomMask');
  sized(isAtomic, A, 'isLigandAtom');
  sized(isMotifRelated, A, 'isMotifRelatedAtom');
  sized(refCharge, A, 'refCharge');
  sized(refMask, A, 'refMask');
  const atoms = new Float32Array(A * 4);
  for (let a = 0; a < A; a++) {
    atoms[a * 4] = Number(refCharge[a]);
    atoms[a * 4 + 1] = refMask[a] ? 1 : 0;
    atoms[a * 4 + 2] = atomMask[a] ? 1 : 0;
    atoms[a * 4 + 3] = (isAtomic[a] || isMotifRelated[a]) ? 1 : 0;
  }
  const nElem = c.numElements;
  let elemOneHot = pick(features, 'refElement', 'ref_element');
  const elem = new Float32Array(A * nElem);
  if (elemOneHot === undefined) {
    const ei = pick(features, 'refElementIndex', 'ref_element_index');
    if (ei === undefined) throw Error('ArtiosPotts WebGPU: features.refElement (or refElementIndex) is required');
    const idx = sized(flatten(ei, 'refElementIndex'), A, 'refElementIndex');
    for (let a = 0; a < A; a++) {
      const k = Number(idx[a]);
      if (!Number.isInteger(k) || k < 0 || k >= nElem) throw Error(`ArtiosPotts WebGPU: refElementIndex[${a}] = ${k} out of range`);
      elem[a * nElem + k] = 1;
    }
  } else {
    const flatElem = sized(flatten(elemOneHot, 'refElement'), A * nElem, 'refElement');
    for (let i = 0; i < A * nElem; i++) elem[i] = Number(flatElem[i]);
  }

  // CSR atom -> token table: [N + 1 starts | atom ids grouped by token].
  const starts = new Int32Array(N + 1);
  for (let a = 0; a < A; a++) {
    const t = Number(atomToToken[a]);
    if (!Number.isInteger(t) || t < 0 || t >= N) throw Error(`ArtiosPotts WebGPU: atomToTokenIdx[${a}] = ${t} out of range [0, ${N})`);
    starts[t + 1]++;
  }
  for (let i = 0; i < N; i++) starts[i + 1] += starts[i];
  const csr = new Int32Array(N + 1 + A);
  csr.set(starts, 0);
  const cursor = Int32Array.from(starts.subarray(0, N));
  for (let a = 0; a < A; a++) csr[N + 1 + cursor[Number(atomToToken[a])]++] = a;

  return { N, A, K: Math.min(c.numNeighbours, N), mask, xyz, bb, tok, tokf, atoms, elem, csr };
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

class ArtiosPottsGpu {
  constructor({ device, tensors, config, precision, pacing = {}, maxCachedPlans = 2, ownsDevice = false }) {
    this.cfg = normalizeConfig(config);
    this.precision = normalizePrecision(precision);
    this.device = device;
    this.ownsDevice = ownsDevice;
    this.rt = new GpuRuntime(device, pacing);
    this.chunkBudgetBytes = pacing.chunkBudgetBytes ?? DEFAULT_PACING.chunkBudgetBytes;
    this.maxCachedPlans = Math.max(1, maxCachedPlans);
    this.plans = new Map();
    this.chain = Promise.resolve();
    this.destroyed = false;
    this.lastRun = null;
    this._loadWeights(tensors);
  }

  _loadWeights(tensors) {
    const c = this.cfg;
    const D = c.nodeDim, E = c.edgeDim, R = c.numResidueTypes, F = c.numFactors;
    const scope = this.rt.scope('weights');
    this.weightScope = scope;
    const p = this.precision;
    const get = (name) => {
      const t = tensors.get(name);
      if (!t) throw Error(`ArtiosPotts WebGPU: missing tensor '${name}'`);
      return t;
    };
    const lin = (name, rows, cols, biasName = null) => prepareLinear(scope, {
      W: get(name), rows, cols, bias: biasName ? get(biasName) : null, precision: p, label: name,
    });
    const ln = (prefix, dim) => {
      const g = get(`${prefix}.weight`);
      const b = get(`${prefix}.bias`);
      const out = new Float32Array(2 * dim);
      out.set(g, 0);
      out.set(b, dim);
      return scope.f32(out, prefix);
    };

    const W = {};
    const fz = 'mpnn.featurizer';

    // node_input: [D, (2 + numElements) | 4 | 32] per output channel.
    const ligW = get(`${fz}.ligand_condition_embed.weight`);
    const entW = get(`${fz}.entity_type_embed.weight`);
    const resW = get(`${fz}.non_protein_restype_embed.weight`);
    const ligIn = 2 + c.numElements;
    const wRow = ligIn + 4 + NUM_NON_PROTEIN_RESTYPES;
    const nodeW = new Float32Array(D * wRow);
    for (let d = 0; d < D; d++) {
      nodeW.set(ligW.subarray(d * ligIn, (d + 1) * ligIn), d * wRow);
      nodeW.set(entW.subarray(d * 4, (d + 1) * 4), d * wRow + ligIn);
      nodeW.set(resW.subarray(d * NUM_NON_PROTEIN_RESTYPES, (d + 1) * NUM_NON_PROTEIN_RESTYPES), d * wRow + ligIn + 4);
    }
    W.nodeIn = scope.f32(nodeW, 'node_input.w');
    W.nodeNorm = ln(`${fz}.node_norm`, D);

    // edge_input aux: [rbf centers | embed_relpos_features.weight]
    const rbf = rbfParameters(c);
    const centers = tensors.get(`${fz}.rbf.centers`) ?? rbf.centers;
    const gammaT = tensors.get(`${fz}.rbf.gamma`);
    this.rbfGamma = gammaT ? gammaT[0] : rbf.gamma;
    const relW = get(`${fz}.positional_encoder.relpos.embed_relpos_features.weight`);
    const edgeAux = new Float32Array(centers.length + relW.length);
    edgeAux.set(centers, 0);
    edgeAux.set(relW, centers.length);
    W.edgeAux = scope.f32(edgeAux, 'edge_input.aux');
    W.embedEdge = lin(`${fz}.embed_edge.weight`, E, 25 * c.numRbfKernels + c.positionalEncodingDim);
    W.edgeNorm = ln(`${fz}.edge_norm`, E);

    const encE = 2 * D + E;
    W.enc = [];
    for (let l = 0; l < c.numEncoderLayers; l++) {
      const q = `mpnn.encoder.${l}`;
      W.enc.push({
        nmIn: lin(`${q}.node_message_mlp.W_in.weight`, 2 * D, encE),
        nmOut: lin(`${q}.node_message_mlp.W_out.weight`, D, D),
        norm1: ln(`${q}.norm1`, D),
        dIn: lin(`${q}.dense.W_in.weight`, 4 * D, D),
        dOut: lin(`${q}.dense.W_out.weight`, D, 2 * D),
        norm2: ln(`${q}.norm2`, D),
        euIn: lin(`${q}.edge_update_mlp.W_in.weight`, 2 * E, encE),
        euOut: lin(`${q}.edge_update_mlp.W_out.weight`, E, E),
        norm3: ln(`${q}.norm3`, E),
      });
    }
    const decE = 2 * D + E;
    W.dec = [];
    for (let l = 0; l < c.numDecoderLayers; l++) {
      const q = `mpnn.decoder.${l}`;
      W.dec.push({
        nmIn: lin(`${q}.node_message_mlp.W_in.weight`, 2 * D, D + decE),
        nmOut: lin(`${q}.node_message_mlp.W_out.weight`, D, D),
        norm1: ln(`${q}.norm1`, D),
        dIn: lin(`${q}.dense.W_in.weight`, 4 * D, D),
        dOut: lin(`${q}.dense.W_out.weight`, D, 2 * D),
        norm2: ln(`${q}.norm2`, D),
      });
    }

    // Sequence-blind decoder input: embed_restype one-hot at the mask token.
    const embedRestype = get('mpnn.embed_restype.weight');
    const maskCol = new Float32Array(D);
    for (let d = 0; d < D; d++) maskCol[d] = embedRestype[d * (R + 1) + R];
    this.maskTokenEmbed = maskCol;

    W.hLin = lin('potts.W_h.weight', R, D, 'potts.W_h.bias');
    // W_J_left and W_J_right concatenated so one matmul lays out
    // [left (R*F) | right (F*R)] exactly as potts_coupling reads it.
    const left = get('potts.W_J_left.weight');
    const right = get('potts.W_J_right.weight');
    const jW = new Float32Array(left.length + right.length);
    jW.set(left, 0);
    jW.set(right, left.length);
    const jB = new Float32Array(2 * R * F);
    jB.set(get('potts.W_J_left.bias'), 0);
    jB.set(get('potts.W_J_right.bias'), R * F);
    W.jLin = prepareLinear(scope, { W: jW, rows: 2 * R * F, cols: decE, bias: jB, precision: p, label: 'potts.W_J' });
    W.pottsScale = Math.exp(get('potts.log_scale')[0]);
    const outW = tensors.get('mpnn.linear_out.weight');
    W.linearOut = outW ? prepareLinear(scope, { W: outW, rows: R, cols: D, precision: p, label: 'mpnn.linear_out.weight' }) : null;
    this.W = W;
  }

  _assertAlive() {
    if (this.destroyed) throw Error('ArtiosPottsGpu: engine has been destroyed');
  }

  _serial(fn) {
    const run = this.chain.then(() => {
      this._assertAlive();
      return fn();
    });
    this.chain = run.then(() => {}, () => {});
    return run;
  }

  _chunks(width, NK) {
    const limit = Math.min(this.chunkBudgetBytes, this.device.limits.maxStorageBufferBindingSize);
    const rows = Math.max(1, Math.min(NK, Math.floor(limit / (width * 4))));
    const out = [];
    for (let s = 0; s < NK; s += rows) out.push([s, Math.min(rows, NK - s)]);
    return out;
  }

  /** Build (or fetch) the recorded dispatch list for one input shape. */
  _plan(N, A, K, trace) {
    const key = `${N}:${A}:${K}:${trace ? 1 : 0}`;
    const hit = this.plans.get(key);
    if (hit) {
      this.plans.delete(key);
      this.plans.set(key, hit);
      return hit;
    }
    const plan = this._buildPlan(N, A, K, trace);
    this.plans.set(key, plan);
    while (this.plans.size > this.maxCachedPlans) {
      const [oldKey, old] = this.plans.entries().next().value;
      this.plans.delete(oldKey);
      old.scope.destroy();
    }
    return plan;
  }

  _buildPlan(N, A, K, trace) {
    const c = this.cfg;
    const rt = this.rt;
    const D = c.nodeDim, E = c.edgeDim, R = c.numResidueTypes, F = c.numFactors;
    const decE = 2 * D + E;
    const NK = N * K;
    const RR = R * R;
    const eps = c.layerNormEps;
    const invScale = 1 / c.numNeighbours;
    const W = this.W;
    const maxBind = this.device.limits.maxStorageBufferBindingSize;
    if (NK * RR * 4 > maxBind) {
      throw Error(`ArtiosPotts WebGPU: J is ${NK * RR * 4} bytes, over this device's ${maxBind} byte storage binding limit `
        + `(n_token=${N}, neighbours=${K}, R=${R})`);
    }

    const scope = rt.scope(`plan:${N}x${K}`);
    const pool = new Pool(scope, !trace);
    const ops = [];
    const nodeSegs = [{ dim: D, index: 'self' }, { dim: E, index: 'row' }, { dim: D, index: 'nbr' }];
    const decSegs = [...nodeSegs, { dim: D, index: 'nbr' }];
    const jSegs = [{ dim: E, index: 'row' }, { dim: D, index: 'nbr' }, { dim: D, index: 'nbr' }];

    // --- inputs ----------------------------------------------------------
    const inputs = {
      xyz: scope.empty(N * 3, 'in.xyz'),
      mask: scope.empty(N, 'in.mask'),
      bb: scope.empty(N * NUM_BACKBONE_ATOMS * 4, 'in.bb'),
      tok: scope.raw(N * 5 * 4, STORAGE_USAGE, 'in.tok'),
      tokf: scope.empty(N * (4 + R), 'in.tokf'),
      atoms: scope.empty(A * 4, 'in.atoms'),
      elem: scope.empty(A * c.numElements, 'in.elem'),
      csr: scope.raw((N + 1 + A) * 4, STORAGE_USAGE, 'in.csr'),
      seqEmbed: scope.empty(N * D, 'in.seq_embed'),
    };
    const idx = scope.raw(NK * 4, STORAGE_USAGE, 'graph.idx');
    const maskJ = scope.empty(NK, 'graph.mask_j');
    const edgeMask = scope.empty(NK, 'graph.edge_mask');

    // --- graph + masks ---------------------------------------------------
    ops.push(knnOp(rt, scope, { xyz: inputs.xyz, mask: inputs.mask, idx, N, Keff: K, KMAX: c.numNeighbours }));
    ops.push(edgeMasksOp(rt, scope, { mask: inputs.mask, idx, maskJ, edgeMask, NK, Kn: K }));

    // --- node features ---------------------------------------------------
    let nodeIn = pool.get(N * D, 'node');
    ops.push(nodeInputOp(rt, scope, {
      tokf: inputs.tokf, atoms: inputs.atoms, elem: inputs.elem, csr: inputs.csr, w: W.nodeIn,
      out: nodeIn, N, nAtom: A, opts: { D, nElem: c.numElements, nRes: NUM_NON_PROTEIN_RESTYPES },
    }));
    ops.push(layerNormOp(rt, scope, { x: nodeIn, gb: W.nodeNorm, mask: inputs.mask, rows: N, D, eps, label: 'node_norm' }));
    const featNode = nodeIn;

    // --- edge features ---------------------------------------------------
    const edgeWidth = 25 * c.numRbfKernels + c.positionalEncodingDim;
    let edgeCur = pool.get(NK * E, 'edge');
    const edgeOpts = { numRbf: c.numRbfKernels, posDim: c.positionalEncodingDim, rMax: c.relposRMax, sMax: c.relposSMax };
    for (const [start, rows] of this._chunks(edgeWidth, NK)) {
      const buf = pool.get(rows * edgeWidth, 'edge_in');
      ops.push(edgeInputOp(rt, scope, {
        bb: inputs.bb, tok: inputs.tok, idx, aux: W.edgeAux, out: buf,
        rows, inBase: start, Kn: K, gamma: this.rbfGamma, opts: edgeOpts,
      }));
      ops.push(linearOp(rt, scope, {
        lin: W.embedEdge, segments: [{ dim: edgeWidth, index: 'row' }], x: [buf],
        out: edgeCur, M: rows, N: E, inBase: 0, outBase: start, Kn: K, label: 'embed_edge',
      }));
      pool.put(buf, rows * edgeWidth);
    }
    ops.push(layerNormOp(rt, scope, { x: edgeCur, gb: W.edgeNorm, mask: maskJ, rows: NK, D: E, eps, label: 'edge_norm' }));
    const featEdge = edgeCur;

    // --- encoder ---------------------------------------------------------
    for (let l = 0; l < c.numEncoderLayers; l++) {
      const w = W.enc[l];
      const gate = pool.get(NK * D, 'enc.msg_gate');
      ops.push(linearOp(rt, scope, {
        lin: w.nmIn, segments: nodeSegs, x: [nodeIn, edgeCur, nodeIn], idx, out: gate,
        M: NK, N: D, Kn: K, swiglu: true,
      }));
      const msg = pool.get(NK * D, 'enc.msg');
      ops.push(linearOp(rt, scope, { lin: w.nmOut, segments: [{ dim: D, index: 'row' }], x: [gate], out: msg, M: NK, N: D, Kn: K }));
      pool.put(gate, NK * D);
      const nodeMid = pool.get(N * D, 'node');
      ops.push(aggregateOp(rt, scope, { msg, w: edgeMask, out: nodeMid, nodes: N, Kn: K, D, invScale, label: 'enc.aggregate' }));
      pool.put(msg, NK * D);
      ops.push(layerNormOp(rt, scope, { x: nodeMid, gb: w.norm1, res: nodeIn, rows: N, D, eps, label: `enc${l}.norm1` }));
      const dGate = pool.get(N * 2 * D, 'enc.dense_gate');
      ops.push(linearOp(rt, scope, { lin: w.dIn, segments: [{ dim: D, index: 'row' }], x: [nodeMid], out: dGate, M: N, N: 2 * D, Kn: K, swiglu: true }));
      const nodeOut = pool.get(N * D, 'node');
      ops.push(linearOp(rt, scope, { lin: w.dOut, segments: [{ dim: 2 * D, index: 'row' }], x: [dGate], out: nodeOut, M: N, N: D, Kn: K }));
      pool.put(dGate, N * 2 * D);
      ops.push(layerNormOp(rt, scope, { x: nodeOut, gb: w.norm2, res: nodeMid, mask: inputs.mask, rows: N, D, eps, label: `enc${l}.norm2` }));
      pool.put(nodeMid, N * D);
      pool.put(nodeIn, N * D);

      const eGate = pool.get(NK * E, 'enc.edge_gate');
      ops.push(linearOp(rt, scope, {
        lin: w.euIn, segments: nodeSegs, x: [nodeOut, edgeCur, nodeOut], idx, out: eGate,
        M: NK, N: E, Kn: K, swiglu: true,
      }));
      const edgeOut = pool.get(NK * E, 'edge');
      ops.push(linearOp(rt, scope, { lin: w.euOut, segments: [{ dim: E, index: 'row' }], x: [eGate], out: edgeOut, M: NK, N: E, Kn: K }));
      pool.put(eGate, NK * E);
      ops.push(layerNormOp(rt, scope, { x: edgeOut, gb: w.norm3, res: edgeCur, mask: edgeMask, rows: NK, D: E, eps, label: `enc${l}.norm3` }));
      pool.put(edgeCur, NK * E);
      nodeIn = nodeOut;
      edgeCur = edgeOut;
    }
    const encNode = nodeIn;
    const encEdge = edgeCur;

    // --- decoder ---------------------------------------------------------
    let pottsNode = nodeIn;
    for (let l = 0; l < c.numDecoderLayers; l++) {
      const w = W.dec[l];
      const last = l === c.numDecoderLayers - 1;
      const gate = pool.get(NK * D, 'dec.msg_gate');
      ops.push(linearOp(rt, scope, {
        lin: w.nmIn, segments: decSegs, x: [nodeIn, encEdge, inputs.seqEmbed, nodeIn], idx, out: gate,
        M: NK, N: D, Kn: K, swiglu: true,
      }));
      const msg = pool.get(NK * D, 'dec.msg');
      ops.push(linearOp(rt, scope, { lin: w.nmOut, segments: [{ dim: D, index: 'row' }], x: [gate], out: msg, M: NK, N: D, Kn: K }));
      pool.put(gate, NK * D);
      const nodeMid = pool.get(N * D, 'node');
      ops.push(aggregateOp(rt, scope, { msg, w: edgeMask, out: nodeMid, nodes: N, Kn: K, D, invScale, label: 'dec.aggregate' }));
      pool.put(msg, NK * D);
      ops.push(layerNormOp(rt, scope, { x: nodeMid, gb: w.norm1, res: nodeIn, rows: N, D, eps, label: `dec${l}.norm1` }));
      const dGate = pool.get(N * 2 * D, 'dec.dense_gate');
      ops.push(linearOp(rt, scope, { lin: w.dIn, segments: [{ dim: D, index: 'row' }], x: [nodeMid], out: dGate, M: N, N: 2 * D, Kn: K, swiglu: true }));
      const nodeOut = pool.get(N * D, 'node');
      ops.push(linearOp(rt, scope, { lin: w.dOut, segments: [{ dim: 2 * D, index: 'row' }], x: [dGate], out: nodeOut, M: N, N: D, Kn: K }));
      pool.put(dGate, N * 2 * D);
      ops.push(layerNormOp(rt, scope, { x: nodeOut, gb: w.norm2, res: nodeMid, mask: inputs.mask, rows: N, D, eps, label: `dec${l}.norm2` }));
      pool.put(nodeMid, N * D);
      // The Potts head couples on the decoder edge features of the *last*
      // layer, which are built from its input node features (MPNN.decode
      // rebuilds decoder_edge_feats before each layer and keeps the last one).
      if (last) pottsNode = nodeIn;
      else pool.put(nodeIn, N * D);
      nodeIn = nodeOut;
    }
    const decNode = nodeIn;

    // --- Potts head ------------------------------------------------------
    const hpre = pool.get(N * R, 'potts.hpre');
    ops.push(linearOp(rt, scope, { lin: W.hLin, segments: [{ dim: D, index: 'row' }], x: [decNode], out: hpre, M: N, N: R, Kn: K }));
    const hBuf = scope.empty(N * R, 'potts.h');
    ops.push(pottsFieldsOp(rt, scope, {
      hpre, mask: inputs.mask, out: hBuf, N, R, scale: W.pottsScale, gauge: c.zeroSumGauge,
    }));

    const JA = scope.empty(NK * RR, 'potts.J');
    const lrWidth = 2 * R * F;
    for (const [start, rows] of this._chunks(lrWidth, NK)) {
      const lr = pool.get(rows * lrWidth, 'potts.lr');
      ops.push(linearOp(rt, scope, {
        lin: W.jLin, segments: jSegs, x: [encEdge, inputs.seqEmbed, pottsNode], idx, out: lr,
        M: rows, N: lrWidth, inBase: start, outBase: 0, Kn: K, label: 'potts.W_J',
      }));
      ops.push(pottsCouplingOp(rt, scope, {
        lr, edgeMask, out: JA, rows, inBase: start, R, F, scale: W.pottsScale, gauge: c.zeroSumGauge,
      }));
      pool.put(lr, rows * lrWidth);
    }
    let JBuf = JA;
    if (c.symmetricJ) {
      JBuf = scope.empty(NK * RR, 'potts.J_sym');
      ops.push(pottsSymmetrizeOp(rt, scope, { Jin: JA, idx, edgeMask, out: JBuf, NK, Kn: K, R }));
    }

    // --- readbacks -------------------------------------------------------
    const reads = [
      { name: 'h', src: hBuf, floats: N * R, kind: 'f32' },
      { name: 'J', src: JBuf, floats: NK * RR, kind: 'f32' },
      { name: 'edgeIndex', src: idx, floats: NK, kind: 'i32' },
      { name: 'edgeMask', src: edgeMask, floats: NK, kind: 'f32' },
    ];
    if (trace) {
      const decEdge = scope.empty(NK * decE, 'trace.dec_edge');
      for (const [start, rows] of this._chunks(decE, NK)) {
        ops.push(gatherConcatOp(rt, scope, {
          segments: jSegs, x: [encEdge, inputs.seqEmbed, pottsNode], idx, out: decEdge,
          rows, inBase: start, outBase: start, Kn: K, label: 'trace.dec_edge',
        }));
      }
      reads.push(
        { name: 'featNode', src: featNode, floats: N * D, kind: 'f32' },
        { name: 'featEdge', src: featEdge, floats: NK * E, kind: 'f32' },
        { name: 'encNode', src: encNode, floats: N * D, kind: 'f32' },
        { name: 'encEdge', src: encEdge, floats: NK * E, kind: 'f32' },
        { name: 'decNode', src: decNode, floats: N * D, kind: 'f32' },
        { name: 'decEdge', src: decEdge, floats: NK * decE, kind: 'f32' },
      );
      if (W.linearOut) {
        const logits = scope.empty(N * R, 'trace.logits');
        ops.push(linearOp(rt, scope, { lin: W.linearOut, segments: [{ dim: D, index: 'row' }], x: [decNode], out: logits, M: N, N: R, Kn: K }));
        reads.push({ name: 'logits', src: logits, floats: N * R, kind: 'f32' });
      }
    }
    for (const r of reads) {
      r.bytes = roundUp(r.floats * 4, 4);
      r.dst = scope.staging(r.bytes, `read.${r.name}`);
    }

    return { scope, ops, inputs, reads, N, A, K };
  }

  computePottsParams(features, options) {
    return this._serial(() => this._computePottsParams(features, options));
  }

  async _computePottsParams(features, { trace = false } = {}) {
    const t0 = performance.now();
    const f = prepareFeatures(features, this.cfg);
    const { N, A, K } = f;
    const D = this.cfg.nodeDim;
    const seqEmbed = new Float32Array(N * D);
    for (let n = 0; n < N; n++) {
      if (!f.mask[n]) continue;
      seqEmbed.set(this.maskTokenEmbed, n * D);
    }
    const plan = this._plan(N, A, K, trace);
    const t1 = performance.now();

    const q = this.device.queue;
    const put = (buf, data) => q.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
    put(plan.inputs.xyz, f.xyz);
    put(plan.inputs.mask, f.mask);
    put(plan.inputs.bb, f.bb);
    put(plan.inputs.tok, f.tok);
    put(plan.inputs.tokf, f.tokf);
    put(plan.inputs.atoms, f.atoms);
    put(plan.inputs.elem, f.elem);
    put(plan.inputs.csr, f.csr);
    put(plan.inputs.seqEmbed, seqEmbed);
    const t2 = performance.now();

    await this.rt.run(plan.ops, { readbacks: plan.reads });
    const t3 = performance.now();

    const out = {};
    for (const r of plan.reads) {
      await r.dst.mapAsync(MAP_READ);
      const copy = r.dst.getMappedRange(0, r.bytes).slice(0, r.floats * 4);
      r.dst.unmap();
      out[r.name] = r.kind === 'i32' ? new Int32Array(copy) : new Float32Array(copy);
    }
    const t4 = performance.now();

    const R = this.cfg.numResidueTypes;
    const result = {
      h: out.h,
      J: out.J,
      edgeIndex: out.edgeIndex,
      edgeMask: out.edgeMask,
      isMasked: Uint8Array.from(f.mask),
      shape: { nToken: N, nAtom: A, nNeighbours: K, numResidueTypes: R, numFactors: this.cfg.numFactors },
      timings: {
        prepareMs: t1 - t0, uploadMs: t2 - t1, computeMs: t3 - t2, readMs: t4 - t3, totalMs: t4 - t0,
      },
    };
    if (trace) {
      result.trace = {};
      for (const key of ['featNode', 'featEdge', 'encNode', 'encEdge', 'decNode', 'decEdge', 'logits']) {
        if (out[key]) result.trace[key] = out[key];
      }
      result.trace.seqEmbed = seqEmbed;
    }
    this.lastRun = result.timings;
    return result;
  }

  stats() {
    return {
      liveBuffers: this.rt.liveBuffers(),
      liveBytes: this.rt.liveBytes(),
      weightBytes: this.weightScope?.bytes ?? 0,
      pipelines: this.rt.pipelines.size,
      kernels: this.rt.kernels.size,
      plans: this.plans.size,
      dispatchesPerRun: [...this.plans.values()].at(-1)?.ops.length ?? 0,
      submits: this.rt.counters.submits,
      dispatches: this.rt.counters.dispatches,
      lastRun: this.lastRun,
    };
  }

  shaderMessages() {
    return this.rt.shaderMessages();
  }

  /** Drop the cached per-shape plans, keeping the resident weights. */
  releasePlans() {
    for (const p of this.plans.values()) p.scope.destroy();
    this.plans.clear();
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.plans.clear();
    this.rt.destroy();
    this.W = null;
    this.weightScope = null;
    this.maskTokenEmbed = null;
    if (this.ownsDevice) this.device.destroy();
  }
}

async function requestPottsDevice() {
  if (!globalThis.navigator?.gpu) throw Error('ArtiosPotts WebGPU: WebGPU is not available');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw Error('ArtiosPotts WebGPU: no WebGPU adapter');
  const want = ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize', 'maxStorageBuffersPerShaderStage'];
  const requiredLimits = {};
  for (const k of want) if (adapter.limits[k] !== undefined) requiredLimits[k] = adapter.limits[k];
  return adapter.requestDevice({ requiredLimits });
}

/**
 * Create the GPU ArtiosPotts head.
 *
 * @param {object} opts
 * @param {GPUDevice} [opts.device]  device to use; one is requested (and then
 *                                   owned / destroyed by the engine) if omitted
 * @param {Map|object|string|ArrayBuffer} opts.weights  ArtiosPotts state_dict
 *                                   tensors, safetensors bytes, or a URL to a
 *                                   .safetensors file / sharded .json manifest
 * @param {object} [opts.config]     ArtiosPotts config (camelCase, upstream
 *                                   snake_case, or `{ artios_mpnn: {...} }`)
 * @param {'f32'|'f16'|'int8'} [opts.precision='f32']  linear weight storage
 * @param {object} [opts.pacing]     { submitBudgetFlops, maxDispatchesPerSubmit,
 *                                     maxInFlight, chunkBudgetBytes }
 * @param {number} [opts.maxCachedPlans=2]
 *
 * `computePottsParams(features, { trace })` runs the sequence-blind MPNN and
 * the Potts head once and resolves with { h, J, edgeIndex, edgeMask, isMasked,
 * shape, timings } (plus `trace` intermediates when asked). `features` is the
 * FullContext slice the featurizer needs: tokenMask, centreCoordinate,
 * backboneCoordinate, backboneMask, asymId, residueIndex, entityId, symId,
 * tokenIndex, restype (or restypeIndex), isProteinToken / isLigandToken /
 * isDnaToken / isRnaToken, atomToTokenIdx, atomMask, isLigandAtom, refCharge,
 * refMask and refElement (or refElementIndex).
 */
export async function createArtiosPottsGpu({
  device, weights, config = {}, precision = 'f32', pacing, maxCachedPlans, fetch: fetchFn,
} = {}) {
  const cfg = normalizeConfig(config);
  const prec = normalizePrecision(precision);
  const tensors = await resolveWeights(weights, cfg, fetchFn);
  let ownsDevice = false;
  if (!device) {
    device = await requestPottsDevice();
    ownsDevice = true;
  }
  const engine = new ArtiosPottsGpu({ device, tensors, config: cfg, precision: prec, pacing, maxCachedPlans, ownsDevice });
  return {
    config: engine.cfg,
    precision: prec,
    device,
    computePottsParams: (features, options) => engine.computePottsParams(features, options),
    releasePlans: () => engine.releasePlans(),
    shaderMessages: () => engine.shaderMessages(),
    stats: () => engine.stats(),
    destroy: () => engine.destroy(),
    get destroyed() {
      return engine.destroyed;
    },
  };
}
