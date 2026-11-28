/**
 * Zero-dependency safetensors reader with FP16 and symmetric per-row INT8
 * dequantization, plus the `manifest.json` bundle loader the ArtiosPotts ports
 * use to fetch weights.
 *
 * Bundles are produced by `tools/potts/export_potts.py`; see that script for the
 * manifest schema.
 */

/** Bytes per element for the safetensors dtypes this reader understands. */
const DTYPE_BYTES = { F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1 };

function asUint8(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError('expected an ArrayBuffer, Uint8Array or other typed array');
}

/** Decode one IEEE-754 binary16 value held in the low 16 bits of `bits`. */
export function float16ToFloat32(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -24 * fraction; // subnormal (and zero)
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** Decode a packed binary16 array. */
export function float16ArrayToFloat32(bits) {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i += 1) out[i] = float16ToFloat32(bits[i]);
  return out;
}

/** Decode a packed bfloat16 array (upper 16 bits of a float32). */
export function bfloat16ArrayToFloat32(bits) {
  const out = new Float32Array(bits.length);
  const view = new DataView(out.buffer);
  for (let i = 0; i < bits.length; i += 1) view.setUint32(i * 4, bits[i] << 16, true);
  return out;
}

/**
 * Parse a safetensors file.
 *
 * @param {ArrayBuffer|Uint8Array} input raw file bytes
 * @returns {{metadata: object, names: string[], header: object,
 *   entry(name: string): object, bytes(name: string): Uint8Array,
 *   float32(name: string): Float32Array, int8(name: string): Int8Array}}
 */
export function parseSafetensors(input) {
  const u8 = asUint8(input);
  if (u8.byteLength < 8) throw new Error('safetensors file is shorter than its length prefix');
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  if (!Number.isSafeInteger(headerLength) || 8 + headerLength > u8.byteLength) {
    throw new Error(`safetensors header length ${headerLength} does not fit the file`);
  }
  const headerText = new TextDecoder('utf-8').decode(u8.subarray(8, 8 + headerLength));
  const header = JSON.parse(headerText);
  const metadata = header.__metadata__ ?? {};
  const dataStart = 8 + headerLength;
  const names = Object.keys(header).filter((k) => k !== '__metadata__').sort();

  const entry = (name) => {
    const found = header[name];
    if (!found) throw new Error(`tensor ${name} is not in this safetensors file`);
    return found;
  };

  const bytes = (name) => {
    const { dtype, shape, data_offsets: [from, to] } = entry(name);
    const width = DTYPE_BYTES[dtype];
    if (width === undefined) throw new Error(`unsupported safetensors dtype ${dtype}`);
    const numel = shape.reduce((a, b) => a * b, 1);
    if (to - from !== numel * width) {
      throw new Error(
        `tensor ${name} declares shape ${JSON.stringify(shape)} (${numel} x ${width}B) ` +
        `but spans ${to - from} bytes`,
      );
    }
    // Copy rather than view the source bytes, so typed-array alignment never
    // depends on where in a larger buffer this file happens to sit.
    return u8.slice(dataStart + from, dataStart + to);
  };

  const float32 = (name) => {
    const { dtype } = entry(name);
    const raw = bytes(name);
    switch (dtype) {
      case 'F32':
        return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
      case 'F16':
        return float16ArrayToFloat32(
          new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2),
        );
      case 'BF16':
        return bfloat16ArrayToFloat32(
          new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2),
        );
      case 'F64': {
        const src = new Float64Array(raw.buffer, raw.byteOffset, raw.byteLength / 8);
        return Float32Array.from(src);
      }
      case 'I8':
        return Float32Array.from(new Int8Array(raw.buffer, raw.byteOffset, raw.byteLength));
      case 'I32':
        return Float32Array.from(new Int32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4));
      case 'U8':
      case 'BOOL':
        return Float32Array.from(raw);
      default:
        throw new Error(`cannot convert safetensors dtype ${dtype} to float32`);
    }
  };

  const int8 = (name) => {
    const { dtype } = entry(name);
    if (dtype !== 'I8') throw new Error(`tensor ${name} is ${dtype}, not I8`);
    const raw = bytes(name);
    return new Int8Array(raw.buffer, raw.byteOffset, raw.byteLength);
  };

  return { header, metadata, names, entry, bytes, float32, int8, has: (n) => n in header };
}

/**
 * Dequantize a symmetric per-row int8 matrix: `w[i, j] = q[i, j] * scale[i]`.
 *
 * @param {Int8Array} q quantized values, row major
 * @param {Float32Array} scale one scale per row
 * @param {number[]} shape `[rows, cols]`
 */
export function dequantizeInt8PerRow(q, scale, shape) {
  const [rows, cols] = shape;
  if (q.length !== rows * cols) {
    throw new Error(`int8 payload has ${q.length} values, expected ${rows * cols}`);
  }
  if (scale.length !== rows) {
    throw new Error(`int8 scale has ${scale.length} values, expected ${rows} rows`);
  }
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    const s = scale[r];
    const base = r * cols;
    for (let c = 0; c < cols; c += 1) out[base + c] = q[base + c] * s;
  }
  return out;
}

const SCALE_SUFFIX_DEFAULT = '.scale';

function joinUrl(baseUrl, path) {
  if (!baseUrl) return path;
  return baseUrl.endsWith('/') ? `${baseUrl}${path}` : `${baseUrl}/${path}`;
}

/** Default `read`: `fetch`, for browsers and any runtime with a global fetch. */
export function fetchReader(fetchImpl = globalThis.fetch) {
  if (typeof fetchImpl !== 'function') {
    throw new Error('no global fetch available; pass an explicit `read` function');
  }
  return async (url) => {
    const response = await fetchImpl(url);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText} for ${url}`);
    return new Uint8Array(await response.arrayBuffer());
  };
}

function decodeJson(bytes, url) {
  try {
    return JSON.parse(new TextDecoder('utf-8').decode(asUint8(bytes)));
  } catch (cause) {
    throw new Error(`could not parse JSON from ${url}: ${cause.message}`, { cause });
  }
}

/**
 * Load a weight bundle written by `tools/potts/export_potts.py`.
 *
 * Tensors are dequantized to float32 on first access and cached, so an int8 or
 * fp16 bundle costs less to download but the same to compute with.
 *
 * @param {object} options
 * @param {string} options.baseUrl directory holding `manifest.json`
 * @param {'fp32'|'fp16'|'int8'} [options.precision]
 * @param {(url: string) => Promise<Uint8Array|ArrayBuffer>} [options.read]
 * @param {boolean} [options.verify] check the manifest SHA-256 of every file
 *   (needs a `crypto.subtle` implementation)
 */
export async function loadWeightBundle({
  baseUrl,
  precision = 'fp32',
  read = fetchReader(),
  verify = false,
} = {}) {
  if (typeof baseUrl !== 'string' || baseUrl.length === 0) {
    throw new Error('loadWeightBundle needs a baseUrl');
  }
  const manifestUrl = joinUrl(baseUrl, 'manifest.json');
  const manifest = decodeJson(await read(manifestUrl), manifestUrl);
  const available = Object.keys(manifest.precisions ?? {});
  const spec = manifest.precisions?.[precision];
  if (!spec) {
    throw new Error(
      `bundle at ${baseUrl} has no "${precision}" precision; it has ${available.join(', ')}`,
    );
  }

  const files = [];
  for (const file of spec.files) {
    const url = joinUrl(baseUrl, file.path);
    const raw = asUint8(await read(url));
    if (raw.byteLength !== file.bytes) {
      throw new Error(`${url} is ${raw.byteLength} bytes, manifest says ${file.bytes}`);
    }
    if (verify) await verifySha256(raw, file.sha256, url);
    files.push(parseSafetensors(raw));
  }

  const quantization = spec.quantization ?? { scheme: 'none' };
  const scaleSuffix = quantization.scale_suffix ?? SCALE_SUFFIX_DEFAULT;
  const quantized = new Set(quantization.quantized ?? []);
  const cache = new Map();

  const locate = (name) => files.find((f) => f.has(name));

  const get = (name) => {
    const cached = cache.get(name);
    if (cached) return cached;
    const file = locate(name);
    if (!file) {
      throw new Error(`tensor ${name} is not in the ${precision} bundle at ${baseUrl}`);
    }
    const { shape, dtype } = file.entry(name);
    let data;
    if (dtype === 'I8') {
      if (!quantized.has(name)) {
        throw new Error(`tensor ${name} is int8 but the manifest does not list it as quantized`);
      }
      const scaleName = `${name}${scaleSuffix}`;
      const scaleFile = locate(scaleName);
      if (!scaleFile) throw new Error(`int8 tensor ${name} has no ${scaleName}`);
      data = dequantizeInt8PerRow(file.int8(name), scaleFile.float32(scaleName), shape);
    } else {
      data = file.float32(name);
    }
    const tensor = { name, data, shape, numel: data.length, sourceDtype: dtype };
    cache.set(name, tensor);
    return tensor;
  };

  const names = [...new Set(files.flatMap((f) => f.names))]
    .filter((n) => !n.endsWith(scaleSuffix))
    .sort();

  return {
    manifest,
    config: manifest.config,
    precision,
    baseUrl,
    quantization,
    names,
    has: (name) => Boolean(locate(name)),
    get,
    /** A 2-D tensor as `{data, rows, cols}`. */
    matrix(name) {
      const t = get(name);
      if (t.shape.length !== 2) {
        throw new Error(`tensor ${name} has shape ${JSON.stringify(t.shape)}, expected 2-D`);
      }
      return { data: t.data, rows: t.shape[0], cols: t.shape[1] };
    },
    /** A 1-D tensor as a `Float32Array`. */
    vector(name) {
      const t = get(name);
      if (t.shape.length !== 1) {
        throw new Error(`tensor ${name} has shape ${JSON.stringify(t.shape)}, expected 1-D`);
      }
      return t.data;
    },
  };
}

async function verifySha256(bytes, expected, url) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('verify: true needs crypto.subtle, which is unavailable');
  const digest = await subtle.digest('SHA-256', bytes);
  const hex = [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  if (hex !== expected) {
    throw new Error(`SHA-256 mismatch for ${url}: expected ${expected}, got ${hex}`);
  }
}
