/**
 * Zero-dependency safetensors + sharded manifest loader for Genie 3.
 *
 * Supports:
 * - Single-file `.safetensors` (F32, F16, BF16, I32, I16, I8, U8, I64, BOOL)
 * - Sharded `.safetensors` bundles with `manifest.json` (FP32, FP16, INT8 per-tensor/per-channel/blockwise)
 * - LocalFold/RFD4-style raw `.bin` sharded bundles with `manifest.json`
 * - Both browser (`fetch` + optional Cache API) and Node (`readFile` callback or `loadWeightsFromDir`)
 */

const TYPED_ARRAY_CTORS = {
  F32: Float32Array,
  F64: Float64Array,
  I32: Int32Array,
  I16: Int16Array,
  I8: Int8Array,
  U8: Uint8Array,
  BOOL: Uint8Array,
};

const _u32Scratch = new Uint32Array(1);
const _f32Scratch = new Float32Array(_u32Scratch.buffer);

/**
 * Decode an IEEE-754 binary16 (uint16) value into a JS number (FP32 exact).
 * @param {number} h
 * @returns {number}
 */
export function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * (2 ** -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (2 ** (e - 15)) * (1 + f / 1024);
}

/**
 * Decode a bfloat16 (uint16) value into a JS number.
 * @param {number} h
 * @returns {number}
 */
export function bfloat16ToFloat(h) {
  _u32Scratch[0] = h << 16;
  return _f32Scratch[0];
}

function viewAligned(Ctor, buffer, byteOffset, length) {
  const width = Ctor.BYTES_PER_ELEMENT;
  if (byteOffset % width === 0) {
    return new Ctor(buffer, byteOffset, length);
  }
  return new Ctor(buffer.slice(byteOffset, byteOffset + length * width));
}

/**
 * Parse a `.safetensors` binary buffer.
 *
 * By default, F16/BF16 tensors are widened to Float32Array (pass `{ keepRawF16: true }` to keep Uint16Array).
 *
 * @param {ArrayBuffer | ArrayBufferView} input
 * @param {{ keepRawF16?: boolean }} [opts]
 * @returns {{ tensors: Record<string, { shape: number[], dtype: string, data: ArrayBufferView }>, metadata: Record<string, any> }}
 */
export function parseSafetensors(input, opts = {}) {
  const keepRawF16 = opts.keepRawF16 === true;
  let buffer;
  let byteOffset = 0;
  if (ArrayBuffer.isView(input)) {
    buffer = input.buffer;
    byteOffset = input.byteOffset;
  } else {
    buffer = input;
  }

  const dv = new DataView(buffer, byteOffset);
  const headerLen = Number(dv.getBigUint64(0, true));
  const headerBytes = new Uint8Array(buffer, byteOffset + 8, headerLen);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  const base = byteOffset + 8 + headerLen;

  const tensors = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [s, e] = info.data_offsets;
    const at = base + s;
    const bytes = e - s;
    const dtype = info.dtype;
    let data;
    let outDtype = dtype;

    if (dtype === 'I64') {
      const src = new DataView(buffer, at, bytes);
      const count = bytes / 8;
      data = new Int32Array(count);
      for (let i = 0; i < count; i += 1) {
        data[i] = Number(src.getBigInt64(i * 8, true));
      }
      outDtype = 'I32';
    } else if (dtype === 'F16') {
      const count = bytes / 2;
      if (keepRawF16) {
        data = viewAligned(Uint16Array, buffer, at, count);
      } else if (typeof Float16Array === 'function') {
        const f16 = viewAligned(Float16Array, buffer, at, count);
        data = Float32Array.from(f16);
        outDtype = 'F32';
      } else {
        const u16 = viewAligned(Uint16Array, buffer, at, count);
        data = new Float32Array(count);
        for (let i = 0; i < count; i += 1) data[i] = halfToFloat(u16[i]);
        outDtype = 'F32';
      }
    } else if (dtype === 'BF16') {
      const count = bytes / 2;
      const u16 = viewAligned(Uint16Array, buffer, at, count);
      data = new Float32Array(count);
      for (let i = 0; i < count; i += 1) data[i] = bfloat16ToFloat(u16[i]);
      outDtype = 'F32';
    } else {
      const Ctor = TYPED_ARRAY_CTORS[dtype];
      if (!Ctor) {
        throw new Error(`Unsupported safetensors dtype "${dtype}" for tensor "${name}"`);
      }
      data = viewAligned(Ctor, buffer, at, bytes / Ctor.BYTES_PER_ELEMENT);
    }

    tensors[name] = {
      shape: info.shape,
      dtype: outDtype,
      originalDtype: dtype,
      data,
    };
  }

  return {
    tensors,
    metadata: header.__metadata__ ?? {},
  };
}

/**
 * Symmetric INT8 dequantization supporting:
 * - Scalar scale (per-tensor): `typeof scale === 'number'` or `scale.length === 1`
 * - Per-row / per-channel scale (axis 0): `scale.length === shape[0]`
 * - Block-wise scale (e.g. blockSize = 64): `scale.length === Math.ceil(n / blockSize)`
 *
 * @param {Int8Array} codes
 * @param {number | Float32Array | Float64Array | number[]} scale
 * @param {number[]} shape
 * @param {{ blockSize?: number, axis?: number }} [opts]
 * @returns {Float32Array}
 */
export function dequantizeInt8(codes, scale, shape, opts = {}) {
  const n = codes.length;
  const out = new Float32Array(n);

  if (typeof scale === 'number') {
    for (let i = 0; i < n; i += 1) out[i] = codes[i] * scale;
    return out;
  }

  const scaleLen = scale.length;
  if (scaleLen === 1) {
    const s = scale[0];
    for (let i = 0; i < n; i += 1) out[i] = codes[i] * s;
    return out;
  }

  // Block-wise dequantization if blockSize is explicitly specified
  if (opts.blockSize && opts.blockSize > 0) {
    const block = opts.blockSize;
    const blocks = Math.ceil(n / block);
    for (let b = 0; b < blocks; b += 1) {
      const s = scale[b];
      const start = b * block;
      const end = Math.min(start + block, n);
      for (let i = start; i < end; i += 1) out[i] = codes[i] * s;
    }
    return out;
  }

  // Per-row / per-output-channel (axis 0) dequantization
  const rows = shape[0] || 1;
  if (scaleLen === rows) {
    const rowStride = n / rows;
    for (let r = 0; r < rows; r += 1) {
      const s = scale[r];
      const base = r * rowStride;
      for (let c = 0; c < rowStride; c += 1) {
        out[base + c] = codes[base + c] * s;
      }
    }
    return out;
  }

  // Fallback: infer block size from scale length
  const inferredBlock = Math.ceil(n / scaleLen);
  for (let b = 0; b < scaleLen; b += 1) {
    const s = scale[b];
    const start = b * inferredBlock;
    const end = Math.min(start + inferredBlock, n);
    for (let i = start; i < end; i += 1) out[i] = codes[i] * s;
  }
  return out;
}

/**
 * Strip `model.` / `_orig_mod.` prefixes and normalize legacy weight names
 * (`linear_motif_template` -> `linear_cond_template`), matching `runner.py:_load_checkpoint`.
 *
 * @param {string} key
 * @returns {string}
 */
export function normalizeWeightKey(key) {
  let k = key.replace(/_orig_mod\./g, '');
  if (k.startsWith('model.')) k = k.slice(6);
  k = k.replace('.linear_motif_template.', '.linear_cond_template.');
  return k;
}

/**
 * Normalize a weight map (either `{ tensors, config, manifest }` or plain `{ [key]: { data, shape } }`)
 * into a canonical `{ [normalizedKey]: { data: Float32Array, shape: number[], dtype: string } }` map.
 * Also automatically resolves paired `<name>.__scale__` or `<name>.scale` INT8 tensors if present.
 *
 * @param {object} weightsInput
 * @returns {{ tensors: Record<string, { data: Float32Array, shape: number[], dtype: string }>, config: object | null, manifest: object | null }}
 */
export function normalizeWeights(weightsInput) {
  const isWrapper =
    weightsInput &&
    typeof weightsInput === 'object' &&
    weightsInput.tensors &&
    typeof weightsInput.tensors === 'object' &&
    !ArrayBuffer.isView(weightsInput.tensors.data);

  const rawTensors = isWrapper ? weightsInput.tensors : weightsInput;
  const config = isWrapper ? (weightsInput.config ?? weightsInput.manifest?.config ?? null) : null;
  const manifest = isWrapper ? (weightsInput.manifest ?? null) : null;

  const tensors = {};
  const scaleSuffixes = ['.__scale', '.__scale__', '.scale', '_scale'];

  // Index potential scale tensors first
  const scaleKeys = new Set();
  for (const key of Object.keys(rawTensors)) {
    for (const suf of scaleSuffixes) {
      if (key.endsWith(suf)) {
        const base = key.slice(0, -suf.length);
        if (rawTensors[base] && rawTensors[base].dtype === 'I8') {
          scaleKeys.add(key);
        }
      }
    }
  }

  for (const [rawKey, entry] of Object.entries(rawTensors)) {
    if (scaleKeys.has(rawKey)) continue;
    const key = normalizeWeightKey(rawKey);
    const metaRec = manifest?.tensors?.[rawKey] ?? manifest?.tensors?.[key] ?? null;

    if (entry.dtype === 'I8' || entry.data instanceof Int8Array) {
      // Find scale from manifest record or companion scale tensor
      let scale = metaRec?.scale ?? entry.scale ?? null;
      if (scale === null || scale === undefined) {
        const scaleKey =
          metaRec?.quant?.scale_tensor ??
          metaRec?.scale_key ??
          metaRec?.scaleKey;
        if (scaleKey && rawTensors[scaleKey]) {
          scale = rawTensors[scaleKey].data;
        } else {
          for (const suf of scaleSuffixes) {
            if (rawTensors[rawKey + suf]) {
              scale = rawTensors[rawKey + suf].data;
              break;
            }
          }
        }
      }
      if (scale === null || scale === undefined) {
        throw new Error(`INT8 tensor "${rawKey}" is missing scale metadata or scale tensor`);
      }
      const blockSize = metaRec?.block ?? metaRec?.block_size ?? entry.blockSize;
      const dequant = dequantizeInt8(entry.data, scale, entry.shape, { blockSize });
      tensors[key] = {
        shape: entry.shape,
        dtype: 'F32',
        data: dequant,
      };
    } else if (entry.data instanceof Float32Array) {
      tensors[key] = entry;
    } else {
      // Ensure Float32Array for numeric views
      tensors[key] = {
        shape: entry.shape,
        dtype: 'F32',
        data: Float32Array.from(entry.data),
      };
    }
  }

  return { tensors, config, manifest };
}

/**
 * Decode a raw `.bin` sharded tensor record (LocalFold / RFD4 format).
 */
function readRawBinTensor(record, buffer) {
  const elements = record.shape.reduce((acc, v) => acc * v, 1);
  const dtype = (record.dtype || 'float32').toLowerCase();
  const byteOffset = record.byteOffset ?? record.offset ?? 0;

  if (dtype === 'float32' || dtype === 'f32') {
    return viewAligned(Float32Array, buffer, byteOffset, elements);
  }
  if (dtype === 'float16' || dtype === 'f16') {
    if (typeof Float16Array === 'function') {
      return Float32Array.from(viewAligned(Float16Array, buffer, byteOffset, elements));
    }
    const u16 = viewAligned(Uint16Array, buffer, byteOffset, elements);
    const out = new Float32Array(elements);
    for (let i = 0; i < elements; i += 1) out[i] = halfToFloat(u16[i]);
    return out;
  }
  if (dtype === 'int8' || dtype === 'i8') {
    const codes = new Int8Array(buffer, byteOffset, elements);
    if (record.scaleOffset !== undefined) {
      const block = record.block ?? record.block_size ?? 64;
      const blocks = Math.ceil(elements / block);
      const scaleDtype = (record.scaleDtype || 'float16').toLowerCase();
      let scales;
      if (scaleDtype === 'float32' || scaleDtype === 'f32') {
        scales = viewAligned(Float32Array, buffer, record.scaleOffset, blocks);
      } else if (typeof Float16Array === 'function') {
        scales = Float32Array.from(viewAligned(Float16Array, buffer, record.scaleOffset, blocks));
      } else {
        const u16 = viewAligned(Uint16Array, buffer, record.scaleOffset, blocks);
        scales = new Float32Array(blocks);
        for (let b = 0; b < blocks; b += 1) scales[b] = halfToFloat(u16[b]);
      }
      return dequantizeInt8(codes, scales, record.shape, { blockSize: block });
    }
    if (record.scale !== undefined) {
      return dequantizeInt8(codes, record.scale, record.shape, {
        blockSize: record.block ?? record.block_size,
      });
    }
    throw new Error(`Raw int8 record missing scale/scaleOffset`);
  }
  throw new Error(`Unsupported raw bin tensor dtype: ${record.dtype}`);
}

/**
 * Load a Genie 3 sharded weight bundle given a `manifest` JSON and an async `loadShard(filename)`
 * callback returning an `ArrayBuffer` (or `Uint8Array`).
 *
 * Works identically in browsers (via `fetch`) and Node (via `fs.readFile`), supporting:
 * - Sharded `.safetensors` files (`manifest.shards` or `manifest.weight_map` or `manifest.tensors[k].shard/file`)
 * - Sharded raw `.bin` files (`manifest.tensors[k].file` + `byteOffset`)
 *
 * @param {object} manifest
 * @param {(filename: string) => Promise<ArrayBuffer | Uint8Array>} loadShard
 * @param {{ onProgress?: (p: { loadedShards: number, totalShards: number, loadedBytes: number, totalBytes: number, fraction: number }) => void, concurrency?: number }} [opts]
 * @returns {Promise<{ tensors: Record<string, { data: Float32Array, shape: number[], dtype: string }>, config: object | null, manifest: object }>}
 */
export async function loadWeightsFromManifestData(manifest, loadShard, opts = {}) {
  const { onProgress = null, concurrency = 4 } = opts;

  // Determine list of shard files
  const shardSet = new Set();
  const shardBytesMap = new Map();

  if (Array.isArray(manifest.shards)) {
    for (const entry of manifest.shards) {
      const file = typeof entry === 'string' ? entry : (entry.file ?? entry.filename ?? entry.path);
      if (file) {
        shardSet.add(file);
        if (typeof entry === 'object' && (entry.bytes || entry.size)) {
          shardBytesMap.set(file, entry.bytes ?? entry.size);
        }
      }
    }
  }
  if (manifest.weight_map && typeof manifest.weight_map === 'object') {
    for (const file of Object.values(manifest.weight_map)) {
      if (typeof file === 'string') shardSet.add(file);
    }
  }
  if (manifest.tensors && typeof manifest.tensors === 'object') {
    for (const rec of Object.values(manifest.tensors)) {
      const file = rec?.file ?? rec?.shard;
      if (typeof file === 'string') shardSet.add(file);
    }
  }

  const shardFiles = [...shardSet];
  if (shardFiles.length === 0) {
    throw new Error('Manifest lists no weight shard files');
  }

  const totalShards = shardFiles.length;
  const totalBytes = shardFiles.reduce((acc, f) => acc + (shardBytesMap.get(f) ?? 0), 0);
  let loadedShards = 0;
  let loadedBytes = 0;

  // Group raw .bin tensor records by shard if needed
  const rawByShard = new Map();
  if (manifest.tensors) {
    for (const [name, rec] of Object.entries(manifest.tensors)) {
      const file = rec?.file ?? rec?.shard;
      if (file && !file.endsWith('.safetensors')) {
        if (!rawByShard.has(file)) rawByShard.set(file, []);
        rawByShard.get(file).push([name, rec]);
      }
    }
  }

  const rawTensors = {};
  let cursor = 0;

  async function worker() {
    while (cursor < shardFiles.length) {
      const idx = cursor++;
      const file = shardFiles[idx];
      const rawBuf = await loadShard(file);
      const buf = ArrayBuffer.isView(rawBuf)
        ? rawBuf.buffer.slice(rawBuf.byteOffset, rawBuf.byteOffset + rawBuf.byteLength)
        : rawBuf;

      if (file.endsWith('.safetensors')) {
        const parsed = parseSafetensors(buf);
        for (const [k, v] of Object.entries(parsed.tensors)) {
          rawTensors[k] = v;
        }
      } else {
        for (const [name, rec] of rawByShard.get(file) ?? []) {
          rawTensors[name] = {
            shape: rec.shape,
            dtype: 'F32',
            data: readRawBinTensor(rec, buf),
          };
        }
      }

      loadedShards += 1;
      loadedBytes += shardBytesMap.get(file) ?? buf.byteLength;
      if (onProgress) {
        onProgress({
          loadedShards,
          totalShards,
          loadedBytes,
          totalBytes: totalBytes || loadedBytes,
          fraction: totalBytes > 0 ? loadedBytes / totalBytes : loadedShards / totalShards,
        });
      }
    }
  }

  const nWorkers = Math.min(concurrency, shardFiles.length);
  await Promise.all(Array.from({ length: nWorkers }, () => worker()));

  return normalizeWeights({
    tensors: rawTensors,
    config: manifest.config ?? manifest.model_config ?? null,
    manifest,
  });
}

/**
 * Browser/HTTP loader for a sharded Genie 3 weight directory containing `manifest.json` + shards.
 *
 * @param {string} baseUrl
 * @param {{ onProgress?: Function, signal?: AbortSignal, cache?: boolean, concurrency?: number }} [opts]
 */
export async function fetchGenie3Weights(baseUrl, opts = {}) {
  const cleanBase = baseUrl.replace(/\/+$/, '');
  const { signal = null, cache: useCache = false } = opts;

  const manifestRes = await fetch(`${cleanBase}/manifest.json`, { signal });
  if (!manifestRes.ok) {
    throw new Error(`Failed to fetch manifest from ${cleanBase}/manifest.json: HTTP ${manifestRes.status}`);
  }
  const manifest = await manifestRes.json();

  let cacheStore = null;
  if (useCache && typeof caches !== 'undefined') {
    const cacheName = `genie3-weights-${manifest.format ?? manifest.dtype ?? 'v1'}-${manifest.version ?? '1'}`;
    cacheStore = await caches.open(cacheName).catch(() => null);
  }

  const loadShard = async (filename) => {
    const url = `${cleanBase}/${filename}`;
    if (cacheStore) {
      const hit = await cacheStore.match(url).catch(() => null);
      if (hit && hit.ok) return hit.arrayBuffer();
    }
    const res = await fetch(url, { signal });
    if (!res.ok) {
      throw new Error(`Failed to fetch shard ${url}: HTTP ${res.status}`);
    }
    const buf = await res.arrayBuffer();
    if (cacheStore) {
      await cacheStore.put(
        url,
        new Response(buf.slice(0), {
          headers: { 'content-type': 'application/octet-stream' },
        }),
      ).catch(() => null);
    }
    return buf;
  };

  return loadWeightsFromManifestData(manifest, loadShard, opts);
}

/**
 * Node-only convenience loader for a local directory containing `manifest.json` + shard files,
 * or a single `.safetensors` file path. Uses dynamic `import('node:fs/promises')` so this module
 * remains 100% browser-compatible.
 *
 * @param {string} dirOrFilePath
 * @param {{ onProgress?: Function }} [opts]
 */
export async function loadWeightsFromPath(dirOrFilePath, opts = {}) {
  const fs = await import('node:fs/promises');
  const path = await import('node:path');

  if (dirOrFilePath.endsWith('.safetensors')) {
    const buf = await fs.readFile(dirOrFilePath);
    const parsed = parseSafetensors(buf);
    return normalizeWeights({ tensors: parsed.tensors, config: null, manifest: null });
  }

  const manifestPath = path.join(dirOrFilePath, 'manifest.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  return loadWeightsFromManifestData(
    manifest,
    async (filename) => fs.readFile(path.join(dirOrFilePath, filename)),
    opts,
  );
}
