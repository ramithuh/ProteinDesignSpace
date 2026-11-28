// Weight plumbing for the WebGPU ArtiosPotts engine: config normalisation, the exact
// upstream state_dict names/shapes, safetensors parsing, and FP16 / INT8 weight packing.

export const DEFAULT_CONFIG = Object.freeze({
  numResidueTypes: 32,
  nodeDim: 128,
  edgeDim: 128,
  numEncoderLayers: 3,
  numDecoderLayers: 3,
  numNeighbours: 32,
  positionalEncodingDim: 16,
  numRbfKernels: 16,
  rbfMinDistance: 2.0,
  rbfMaxDistance: 22.0,
  relposRMax: 32,
  relposSMax: 2,
  numElements: 128,
  layerNormEps: 1e-5,
  parameterisation: 'factor',
  numFactors: 32,
  symmetricJ: true,
  zeroSumGauge: true,
});

const ALIASES = {
  num_residue_types: 'numResidueTypes',
  node_dim: 'nodeDim',
  edge_dim: 'edgeDim',
  num_encoder_layers: 'numEncoderLayers',
  num_decoder_layers: 'numDecoderLayers',
  num_neighbours: 'numNeighbours',
  num_neighbors: 'numNeighbours',
  positional_encoding_dim: 'positionalEncodingDim',
  num_rbf_kernels: 'numRbfKernels',
  num_factors: 'numFactors',
  symmetric_J: 'symmetricJ',
  zero_sum_gauge: 'zeroSumGauge',
  layer_norm_eps: 'layerNormEps',
};

/** Accepts camelCase, upstream snake_case, or the nested `{ artios_mpnn: { model, potts } }` form. */
export function normalizeConfig(config = {}) {
  const flat = {};
  const absorb = (obj) => {
    if (!obj || typeof obj !== 'object') return;
    for (const [k, v] of Object.entries(obj)) {
      if (v && typeof v === 'object' && !Array.isArray(v)) continue;
      flat[ALIASES[k] ?? k] = v;
    }
  };
  const root = config.artios_mpnn ?? config.artiosMpnn ?? config;
  absorb(root);
  absorb(root.model);
  absorb(root.potts);
  const cfg = { ...DEFAULT_CONFIG };
  for (const k of Object.keys(DEFAULT_CONFIG)) if (flat[k] !== undefined) cfg[k] = flat[k];
  if (cfg.parameterisation !== 'factor') {
    throw Error(`ArtiosPotts WebGPU: only the 'factor' Potts parameterisation is implemented (got '${cfg.parameterisation}')`);
  }
  if (cfg.numNeighbours > 64) throw Error('ArtiosPotts WebGPU: numNeighbours > 64 is not supported');
  return Object.freeze(cfg);
}

export function normalizePrecision(p = 'f32') {
  const map = { f32: 'f32', fp32: 'f32', float32: 'f32', f16: 'f16', fp16: 'f16', float16: 'f16', half: 'f16', int8: 'int8', i8: 'int8', q8: 'int8' };
  const out = map[String(p).toLowerCase()];
  if (!out) throw Error(`ArtiosPotts WebGPU: unknown precision '${p}' (use 'f32', 'f16' or 'int8')`);
  return out;
}

/**
 * Exact upstream ArtiosPotts state_dict names and shapes ([out, in] for Linear weights).
 * `optional` tensors default as noted (LayerNorm biases are zero buffers upstream).
 */
export function weightSpec(config) {
  const c = normalizeConfig(config);
  const D = c.nodeDim, E = c.edgeDim, R = c.numResidueTypes, F = c.numFactors;
  const relIn = 4 * c.relposRMax + 2 * c.relposSMax + 7;
  const decE = 2 * D + E;
  const specs = [];
  const add = (name, shape, extra = {}) => specs.push({ name, shape, ...extra });
  const ln = (prefix, dim) => {
    add(`${prefix}.weight`, [dim]);
    add(`${prefix}.bias`, [dim], { optional: 'zeros' });
  };
  const fz = 'mpnn.featurizer';
  add(`${fz}.entity_type_embed.weight`, [D, 4]);
  add(`${fz}.non_protein_restype_embed.weight`, [D, 32]);
  add(`${fz}.ligand_condition_embed.weight`, [D, 2 + c.numElements]);
  add(`${fz}.positional_encoder.relpos.embed_relpos_features.weight`, [c.positionalEncodingDim, relIn]);
  add(`${fz}.rbf.centers`, [c.numRbfKernels], { optional: 'computed' });
  add(`${fz}.rbf.gamma`, [], { optional: 'computed' });
  add(`${fz}.embed_edge.weight`, [E, 25 * c.numRbfKernels + c.positionalEncodingDim]);
  ln(`${fz}.edge_norm`, E);
  ln(`${fz}.node_norm`, D);
  for (let l = 0; l < c.numEncoderLayers; l++) {
    const p = `mpnn.encoder.${l}`;
    add(`${p}.node_message_mlp.W_in.weight`, [2 * D, 2 * D + E]);
    add(`${p}.node_message_mlp.W_out.weight`, [D, D]);
    ln(`${p}.norm1`, D);
    add(`${p}.dense.W_in.weight`, [4 * D, D]);
    add(`${p}.dense.W_out.weight`, [D, 2 * D]);
    ln(`${p}.norm2`, D);
    add(`${p}.edge_update_mlp.W_in.weight`, [2 * E, 2 * D + E]);
    add(`${p}.edge_update_mlp.W_out.weight`, [E, E]);
    ln(`${p}.norm3`, E);
  }
  for (let l = 0; l < c.numDecoderLayers; l++) {
    const p = `mpnn.decoder.${l}`;
    add(`${p}.node_message_mlp.W_in.weight`, [2 * D, D + decE]);
    add(`${p}.node_message_mlp.W_out.weight`, [D, D]);
    ln(`${p}.norm1`, D);
    add(`${p}.dense.W_in.weight`, [4 * D, D]);
    add(`${p}.dense.W_out.weight`, [D, 2 * D]);
    ln(`${p}.norm2`, D);
  }
  add('mpnn.embed_restype.weight', [D, R + 1]);
  add('mpnn.linear_out.weight', [R, D], { optional: 'unused' });
  add('potts.log_scale', [1]);
  add('potts.W_h.weight', [R, D]);
  add('potts.W_h.bias', [R]);
  add('potts.W_J_left.weight', [R * F, decE]);
  add('potts.W_J_left.bias', [R * F]);
  add('potts.W_J_right.weight', [F * R, decE]);
  add('potts.W_J_right.bias', [F * R]);
  return specs;
}

// ---------------------------------------------------------------------------
// FP16 helpers
// ---------------------------------------------------------------------------
const cvF32 = new Float32Array(1);
const cvU32 = new Uint32Array(cvF32.buffer);

/** IEEE binary32 -> binary16 bits, round-to-nearest-even. */
export function f32ToF16Bits(value) {
  cvF32[0] = value;
  const x = cvU32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const mid = 1 << (shift - 1);
    if (rem > mid || (rem === mid && (half & 1))) half++;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++;
  return sign | half;
}

export function f16BitsToF32(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

function bf16BitsToF32(h) {
  cvU32[0] = h << 16;
  return cvF32[0];
}

// ---------------------------------------------------------------------------
// Weight packing for the linear kernel. W is [rows, cols] row-major FP32.
// ---------------------------------------------------------------------------
/**
 * @returns {{ precision, rows, cols, ldw, data: Float32Array|Uint32Array, scale?: Float32Array,
 *             dequantized: Float32Array }}  `dequantized` is the exact FP32 matrix the GPU multiplies by.
 */
export function packLinearWeight(w, rows, cols, precision) {
  if (w.length !== rows * cols) throw Error(`packLinearWeight: expected ${rows * cols} values, got ${w.length}`);
  if (precision === 'f32') return { precision, rows, cols, ldw: cols, data: Float32Array.from(w), dequantized: Float32Array.from(w) };
  if (precision === 'f16') {
    const ldw = Math.ceil(cols / 2);
    const data = new Uint32Array(rows * ldw);
    const deq = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c += 2) {
        const lo = f32ToF16Bits(w[r * cols + c]);
        const hi = c + 1 < cols ? f32ToF16Bits(w[r * cols + c + 1]) : 0;
        data[r * ldw + c / 2] = (lo | (hi << 16)) >>> 0;
        deq[r * cols + c] = f16BitsToF32(lo);
        if (c + 1 < cols) deq[r * cols + c + 1] = f16BitsToF32(hi);
      }
    }
    return { precision, rows, cols, ldw, data, dequantized: deq };
  }
  if (precision === 'int8') {
    const ldw = Math.ceil(cols / 4);
    const data = new Uint32Array(rows * ldw);
    const scale = new Float32Array(rows);
    const deq = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      let amax = 0;
      for (let c = 0; c < cols; c++) amax = Math.max(amax, Math.abs(w[r * cols + c]));
      const s = Math.fround(amax > 0 ? amax / 127 : 1);
      scale[r] = s;
      for (let c = 0; c < cols; c++) {
        const q = Math.max(-127, Math.min(127, Math.round(w[r * cols + c] / s)));
        data[r * ldw + (c >> 2)] |= (q & 0xff) << (8 * (c & 3));
        deq[r * cols + c] = Math.fround(q * s);
      }
    }
    for (let i = 0; i < data.length; i++) data[i] >>>= 0;
    return { precision, rows, cols, ldw, data, scale, dequantized: deq };
  }
  throw Error(`packLinearWeight: unknown precision ${precision}`);
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------
function toF32(dtype, bytes) {
  const n = (k) => bytes.byteLength / k;
  const view = (Ctor, k) => new Ctor(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + n(k) * k));
  switch (dtype) {
    case 'F32': return view(Float32Array, 4);
    case 'F64': return Float32Array.from(view(Float64Array, 8));
    case 'F16': { const h = view(Uint16Array, 2); const o = new Float32Array(h.length); for (let i = 0; i < h.length; i++) o[i] = f16BitsToF32(h[i]); return o; }
    case 'BF16': { const h = view(Uint16Array, 2); const o = new Float32Array(h.length); for (let i = 0; i < h.length; i++) o[i] = bf16BitsToF32(h[i]); return o; }
    case 'I64': return Float32Array.from(view(BigInt64Array, 8), (v) => Number(v));
    case 'I32': return Float32Array.from(view(Int32Array, 4));
    case 'I8': return Float32Array.from(view(Int8Array, 1));
    case 'U8': case 'BOOL': return Float32Array.from(view(Uint8Array, 1));
    default: throw Error(`safetensors: unsupported dtype ${dtype}`);
  }
}

/** Parse a .safetensors file into Map<name, { shape, dtype, data: Float32Array }>. */
export function parseSafetensors(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLen)));
  const base = 8 + headerLen;
  const out = new Map();
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [s, e] = info.data_offsets;
    out.set(name, { shape: info.shape, dtype: info.dtype, data: toF32(info.dtype, bytes.subarray(base + s, base + e)) });
  }
  // Dequantize symmetric per-row I8 tensors paired with `<name>.scale`.
  for (const [name, entry] of out) {
    if (entry.dtype !== 'I8') continue;
    const scaleEntry = out.get(`${name}.scale`);
    if (!scaleEntry) continue;
    const rows = entry.shape[0] ?? 1;
    const cols = rows > 0 ? entry.data.length / rows : 0;
    const sc = scaleEntry.data;
    if (sc.length === rows) {
      for (let r = 0; r < rows; r++) {
        const sr = sc[r];
        const off = r * cols;
        for (let c = 0; c < cols; c++) entry.data[off + c] *= sr;
      }
      entry.dtype = 'F32';
    }
  }
  return out;
}

async function fetchBytes(url, fetchFn) {
  const res = await fetchFn(url);
  if (!res.ok) throw Error(`ArtiosPotts WebGPU: failed to fetch ${url} (${res.status})`);
  return new Uint8Array(await res.arrayBuffer());
}

/**
 * Load a safetensors file or a sharded manifest (HF `*.safetensors.index.json` with
 * `weight_map`, `{ shards | files: [...] }`, or `export_potts.py`'s `manifest.precisions`)
 * into a Map.
 */
export async function loadSafetensorsManifest(url, fetchFn = globalThis.fetch, precision = 'fp32') {
  const href = String(url);
  if (!href.endsWith('.json')) return parseSafetensors(await fetchBytes(href, fetchFn));
  const manifest = await (await fetchFn(href)).json();
  let files;
  if (manifest.weight_map) {
    files = [...new Set(Object.values(manifest.weight_map))];
  } else if (manifest.precisions) {
    const spec = manifest.precisions[precision] ?? manifest.precisions.fp32 ?? Object.values(manifest.precisions)[0];
    files = spec?.files ?? [];
  } else {
    files = manifest.shards ?? manifest.files ?? [];
  }
  const baseUrl = href.slice(0, href.lastIndexOf('/') + 1);
  const out = new Map();
  for (const file of files) {
    const name = typeof file === 'string' ? file : file.file ?? file.path ?? file.url;
    const shard = parseSafetensors(await fetchBytes(/^[a-z]+:|^\//i.test(name) ? name : baseUrl + name, fetchFn));
    for (const [k, v] of shard) out.set(k, v);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tensor resolution
// ---------------------------------------------------------------------------
function tensorData(t, name) {
  if (t == null) return null;
  if (t instanceof Float32Array) return t;
  if (ArrayBuffer.isView(t) || Array.isArray(t)) return Float32Array.from(t);
  if (typeof t === 'number') return Float32Array.of(t);
  if (t.data !== undefined) {
    const dtype = (t.dtype ?? '').toUpperCase();
    if (t.data instanceof Uint16Array && (dtype === 'F16' || dtype === 'FLOAT16')) return Float32Array.from(t.data, f16BitsToF32);
    if (t.data instanceof Uint16Array && dtype === 'BF16') return Float32Array.from(t.data, bf16BitsToF32);
    return tensorData(t.data, name);
  }
  throw Error(`ArtiosPotts WebGPU: cannot interpret tensor '${name}'`);
}

/**
 * Normalises any supported weight source into a lookup `(name) => Promise<Float32Array|null>`.
 * Sources: Map / plain object of tensors; a loader with get/getTensor/load (sync or async,
 * optional keys()/names); safetensors bytes (ArrayBuffer/Uint8Array or an array of shards);
 * or a URL string to a .safetensors file / sharded .json manifest.
 * Names resolve exactly, then by unique suffix (so `model.` / `module.` prefixes work).
 */
export async function createWeightSource(weights, fetchFn) {
  if (weights == null) throw Error('ArtiosPotts WebGPU: `weights` is required');
  if (typeof weights === 'string' || weights instanceof URL) weights = await loadSafetensorsManifest(weights, fetchFn);
  else if (weights instanceof ArrayBuffer || weights instanceof Uint8Array) weights = parseSafetensors(weights);
  else if (Array.isArray(weights)) {
    const merged = new Map();
    for (const shard of weights) for (const [k, v] of parseSafetensors(shard)) merged.set(k, v);
    weights = merged;
  }
  let getter, keys;
  if (weights instanceof Map) {
    getter = (k) => weights.get(k);
    keys = [...weights.keys()];
  } else if (typeof weights.getTensor === 'function' || typeof weights.get === 'function' || typeof weights.load === 'function') {
    const fn = (weights.getTensor ?? weights.get ?? weights.load).bind(weights);
    getter = fn;
    const k = typeof weights.keys === 'function' ? weights.keys() : weights.names ?? weights.tensorNames;
    keys = k ? [...k] : null;
  } else {
    getter = (k) => weights[k];
    keys = Object.keys(weights);
  }
  const resolveName = (name) => {
    if (!keys || keys.includes(name)) return name;
    const hits = keys.filter((k) => k.endsWith('.' + name));
    if (hits.length > 1) throw Error(`ArtiosPotts WebGPU: tensor '${name}' is ambiguous (${hits.slice(0, 3).join(', ')})`);
    return hits[0] ?? null;
  };
  return async (name) => {
    const key = resolveName(name);
    if (key == null) return null;
    return tensorData(await getter(key), key);
  };
}

/** Resolve + validate every tensor in weightSpec(config). Returns Map<name, Float32Array>. */
export async function resolveWeights(weights, config, fetchFn) {
  const lookup = await createWeightSource(weights, fetchFn);
  const out = new Map();
  for (const s of weightSpec(config)) {
    let data = await lookup(s.name);
    if (data == null) {
      if (s.optional === 'zeros') data = new Float32Array(s.shape.reduce((a, b) => a * b, 1));
      else if (s.optional) continue;
      else throw Error(`ArtiosPotts WebGPU: missing tensor '${s.name}'`);
    }
    const want = s.shape.reduce((a, b) => a * b, 1);
    if (data.length !== want) throw Error(`ArtiosPotts WebGPU: tensor '${s.name}' has ${data.length} values, expected [${s.shape}]`);
    out.set(s.name, data);
  }
  return out;
}
