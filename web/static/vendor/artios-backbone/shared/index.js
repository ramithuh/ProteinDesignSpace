/**
 * Shared, zero-dependency runtime for the ArtiosBackbone browser port.
 *
 * Three independent pieces, all pure ES modules with no npm dependencies:
 *
 *  1. **Safetensors shard loader** — parses the bundles written by
 *     `tools/backbone/export_backbone.py`, dequantises fp16 / int8 payloads to
 *     `Float32Array`, stitches row-split tensors back together and verifies
 *     SHA-256 digests.
 *
 *  2. **Browser-native featurizer** — parses mmCIF structures and RCSB Chemical
 *     Component Dictionary `.cif` files directly (no RDKit, no biotite) and
 *     builds the AlphaFold3 reference-conformer features the model consumes:
 *     `ref_pos`, `ref_mask`, `ref_charge`, `ref_element`,
 *     `ref_atom_name_chars`, `ref_space_uid`. Plus the motif, motif-depth and
 *     RASA conditioning features.
 *
 *  3. **EDM noise schedule and sampler** — the Karras/AF3 inference schedule
 *     and the predictor-corrector integration step from
 *     `artios/model/generator.py`.
 *
 * Reference: `artios/model/generator.py`,
 * `artios/model/modules/diffusion_backbone.py`,
 * `artios/data/features/featurizer.py`, `artios/data/chem/ccd.py`.
 */

// ===========================================================================
// 1. constants
// ===========================================================================

/** EDM coordinate standard deviation used by AF3/Protenix preconditioning. */
export const SIGMA_DATA = 16.0;

/** Karras/AF3 inference schedule parameters (`EDMInferenceNoiseScheduler`). */
export const EDM_SCHEDULE = { sMax: 160.0, sMin: 4e-4, rho: 7.0 };

/** `EDMSampler` integration constants. */
export const EDM_SAMPLER = {
  gamma0: 0.8,
  gammaMin: 1.0,
  noiseScaleLambda: 1.003,
  stepScaleEta: 1.5,
};

/** `EDMTrainingNoiseSampler` lognormal parameters. */
export const EDM_TRAINING_NOISE = { pMean: -1.2, pStd: 1.5 };

/**
 * Uppercase IUPAC element symbols for atomic numbers 1..118, matching
 * `rdkit.Chem.GetPeriodicTable().GetElementSymbol(i).upper()` as used by
 * `artios.common.constants.get_all_elems`.
 */
export const ELEMENT_SYMBOLS = Object.freeze([
  'H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F', 'NE',
  'NA', 'MG', 'AL', 'SI', 'P', 'S', 'CL', 'AR', 'K', 'CA',
  'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
  'GA', 'GE', 'AS', 'SE', 'BR', 'KR', 'RB', 'SR', 'Y', 'ZR',
  'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN',
  'SB', 'TE', 'I', 'XE', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND',
  'PM', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB',
  'LU', 'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG',
  'TL', 'PB', 'BI', 'PO', 'AT', 'RN', 'FR', 'RA', 'AC', 'TH',
  'PA', 'U', 'NP', 'PU', 'AM', 'CM', 'BK', 'CF', 'ES', 'FM',
  'MD', 'NO', 'LR', 'RF', 'DB', 'SG', 'BH', 'HS', 'MT', 'DS',
  'RG', 'CN', 'NH', 'FL', 'MC', 'LV', 'TS', 'OG',
]);

/** The 128-slot `ref_element` vocabulary: 118 elements + 10 UNK padding slots. */
export const ELEMENT_VOCAB = Object.freeze([
  ...ELEMENT_SYMBOLS,
  ...Array.from({ length: 10 }, (_, i) => `UNK_ELEM_${119 + i}`),
]);

const ELEMENT_INDEX = new Map(ELEMENT_VOCAB.map((sym, i) => [sym, i]));

/** `ref_element` one-hot width. */
export const N_ELEMENT = 128;
/** `ref_atom_name_chars` is 4 characters x 64 printable-ASCII slots. */
export const ATOM_NAME_LENGTH = 4;
export const ATOM_NAME_ALPHABET = 64;

/**
 * `STD_RESIDUES_AND_GAP` from `artios.common.constants`: the 32-way `restype`
 * vocabulary (20 amino acids + UNK, 4 RNA + N, 4 DNA + DN, gap).
 */
export const RESIDUE_VOCAB = Object.freeze({
  ALA: 0, ARG: 1, ASN: 2, ASP: 3, CYS: 4, GLN: 5, GLU: 6, GLY: 7,
  HIS: 8, ILE: 9, LEU: 10, LYS: 11, MET: 12, PHE: 13, PRO: 14, SER: 15,
  THR: 16, TRP: 17, TYR: 18, VAL: 19, UNK: 20,
  A: 21, G: 22, C: 23, U: 24,
  DA: 25, DG: 26, DC: 27, DT: 28,
  N: 29, DN: 30,
  '-': 31,
});
export const N_RESTYPE = 32;

/** `DiffusionConditioningConfig` binning parameters. */
export const CONDITIONING_BINS = Object.freeze({
  motifDist: { min: 1.0, max: 15.0, binSize: 0.25, nBins: 56 },
  motifDepth: { max: 2.0, binSize: 0.05, nFineBins: 40, nBins: 42 },
  rasa: { max: 1.0, binSize: 0.05, nFineBins: 20, nBins: 22 },
});

/** `RelativePositionEncoding` defaults: r_max=32, s_max=2 -> 139 features. */
export const RELPOS = Object.freeze({ rMax: 32, sMax: 2, nFeatures: 139 });

const HYDROGEN_LIKE = new Set(['H', 'D', 'T']);

// ===========================================================================
// 2. safetensors shard loader
// ===========================================================================

const SAFETENSORS_DTYPES = Object.freeze({
  F32: { bytes: 4 },
  F16: { bytes: 2 },
  I8: { bytes: 1 },
  U8: { bytes: 1 },
  I32: { bytes: 4 },
  I64: { bytes: 8 },
});

/** Decode one IEEE-754 binary16 bit pattern to a JS number. */
export function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -24 * fraction;
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 25) * (1024 + fraction);
}

/**
 * Parse a safetensors file.
 *
 * Layout: 8-byte little-endian header length, that many bytes of JSON
 * describing `{dtype, shape, data_offsets}` per tensor, then the payload.
 *
 * @param {ArrayBuffer|Uint8Array} buffer
 * @returns {{header: object, raw: (name: string) => {dtype: string, shape: number[], bytes: Uint8Array}, names: string[]}}
 */
export function parseSafetensors(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.byteLength < 8) throw new Error('safetensors: file shorter than its 8-byte length prefix');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  if (!Number.isSafeInteger(headerLength) || 8 + headerLength > bytes.byteLength) {
    throw new Error(`safetensors: header length ${headerLength} exceeds file size`);
  }
  const headerText = new TextDecoder().decode(bytes.subarray(8, 8 + headerLength));
  let header;
  try {
    header = JSON.parse(headerText);
  } catch (cause) {
    throw new Error(`safetensors: header is not valid JSON (${cause.message})`);
  }
  const dataStart = 8 + headerLength;
  const names = Object.keys(header).filter((k) => k !== '__metadata__');

  const raw = (name) => {
    const entry = header[name];
    if (!entry) throw new Error(`safetensors: no tensor named ${JSON.stringify(name)}`);
    const [begin, end] = entry.data_offsets;
    if (dataStart + end > bytes.byteLength) {
      throw new Error(`safetensors: ${name} data range [${begin}, ${end}) is past end of file`);
    }
    const spec = SAFETENSORS_DTYPES[entry.dtype];
    if (!spec) throw new Error(`safetensors: ${name} has unsupported dtype ${entry.dtype}`);
    const expected = entry.shape.reduce((a, b) => a * b, 1) * spec.bytes;
    if (end - begin !== expected) {
      throw new Error(
        `safetensors: ${name} is ${end - begin} bytes, shape ${JSON.stringify(entry.shape)} ` +
          `of ${entry.dtype} needs ${expected}`,
      );
    }
    return {
      dtype: entry.dtype,
      shape: entry.shape.slice(),
      bytes: bytes.subarray(dataStart + begin, dataStart + end),
    };
  };

  return { header, names, raw };
}

/**
 * Lookup table over all 65536 half-precision bit patterns.
 *
 * A 256 KiB table, built on first use. Decoding the fp16 bundle means ~137M
 * conversions, and a table lookup is several times faster than recomputing the
 * sign/exponent/mantissa arithmetic per value.
 */
let HALF_TABLE = null;
function halfTable() {
  if (HALF_TABLE) return HALF_TABLE;
  HALF_TABLE = new Float32Array(65536);
  for (let bits = 0; bits < 65536; bits += 1) HALF_TABLE[bits] = halfToFloat(bits);
  return HALF_TABLE;
}

/**
 * safetensors payloads are little-endian; typed-array views are native-endian.
 * The aliasing fast paths below are only valid where those agree.
 */
const LITTLE_ENDIAN = (() => {
  const probe = new Uint16Array([1]);
  return new Uint8Array(probe.buffer)[0] === 1;
})();

/** Convert a raw safetensors payload to `Float32Array` (copying). */
export function decodeToFloat32(dtype, bytes) {
  const n = bytes.byteLength / SAFETENSORS_DTYPES[dtype].bytes;
  switch (dtype) {
    case 'F32': {
      // Payloads are only guaranteed to be 1-byte aligned inside a shard, so
      // aliasing a Float32Array is a fast path, not the general case.
      if (LITTLE_ENDIAN && bytes.byteOffset % 4 === 0) {
        return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
      }
      const out = new Float32Array(n);
      const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let i = 0; i < n; i += 1) out[i] = dv.getFloat32(i * 4, true);
      return out;
    }
    case 'F16': {
      const table = halfTable();
      const out = new Float32Array(n);
      if (LITTLE_ENDIAN && bytes.byteOffset % 2 === 0) {
        const src = new Uint16Array(bytes.buffer, bytes.byteOffset, n);
        for (let i = 0; i < n; i += 1) out[i] = table[src[i]];
        return out;
      }
      const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let i = 0; i < n; i += 1) out[i] = table[dv.getUint16(i * 2, true)];
      return out;
    }
    case 'I8': {
      const src = new Int8Array(bytes.buffer, bytes.byteOffset, n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i += 1) out[i] = src[i];
      return out;
    }
    default:
      throw new Error(`decodeToFloat32: unsupported dtype ${dtype}`);
  }
}

async function sha256Hex(bytes) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const copy = bytes.slice();
  const digest = await subtle.digest('SHA-256', copy);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Load a sharded weight bundle produced by `tools/backbone/export_backbone.py`.
 *
 * Each shard is fetched at most once, on first use, and every tensor in the
 * manifest is then resolved and dequantised to `Float32Array` before this
 * returns — the CPU and WebGPU denoisers are synchronous, so they need the full
 * set in hand. That means peak memory of roughly the fp32 parameter count
 * (~521 MiB for the released checkpoint) regardless of the bundle's precision.
 *
 * @param {object} options
 * @param {object} options.manifest  parsed `manifest.json`
 * @param {(shardName: string) => Promise<ArrayBuffer|Uint8Array>} options.fetchShard
 * @param {boolean} [options.verify=true] check each shard's SHA-256 against the manifest
 * @returns {Promise<{get: (name: string) => {shape: number[], data: Float32Array},
 *                    has: (name: string) => boolean, names: string[],
 *                    config: object, precision: string, manifest: object,
 *                    loadAll: () => Promise<Record<string, {shape: number[], data: Float32Array}>>}>}
 */
export async function loadWeightBundle({ manifest, fetchShard, verify = true }) {
  if (!manifest || typeof manifest !== 'object') throw new Error('loadWeightBundle: manifest is required');
  if (typeof fetchShard !== 'function') throw new Error('loadWeightBundle: fetchShard is required');
  if (manifest.format && !String(manifest.format).startsWith('artios-backbone-weights/')) {
    throw new Error(`loadWeightBundle: unexpected manifest format ${manifest.format}`);
  }

  const shardDigests = new Map((manifest.shards ?? []).map((s) => [s.name, s.sha256]));
  const parsed = new Map();
  const pending = new Map();

  async function shard(name) {
    if (parsed.has(name)) return parsed.get(name);
    if (pending.has(name)) return pending.get(name);
    const task = (async () => {
      const buffer = await fetchShard(name);
      const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
      if (verify && shardDigests.has(name)) {
        const actual = await sha256Hex(bytes);
        const expected = shardDigests.get(name);
        if (actual && actual !== expected) {
          throw new Error(`loadWeightBundle: ${name} digest ${actual} != manifest ${expected}`);
        }
      }
      const file = parseSafetensors(bytes);
      parsed.set(name, file);
      pending.delete(name);
      return file;
    })();
    pending.set(name, task);
    return task;
  }

  const tensorMeta = manifest.tensors ?? {};
  const cache = new Map();

  /** Read one tensor, stitching row-split parts and dequantising as needed. */
  async function read(name) {
    if (cache.has(name)) return cache.get(name);
    const meta = tensorMeta[name];
    if (!meta) throw new Error(`loadWeightBundle: ${name} is not in the manifest`);

    const shape = meta.shape.slice();
    const total = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(total);
    const rowStride = shape.length > 1 ? total / shape[0] : 1;

    for (const part of meta.parts) {
      const file = await shard(part.shard);
      const { dtype, bytes, shape: partShape } = file.raw(part.key);
      if (dtype !== meta.dtype) {
        throw new Error(`loadWeightBundle: ${part.key} is ${dtype}, manifest says ${meta.dtype}`);
      }
      const values = decodeToFloat32(dtype, bytes);
      const offset = (part.row ?? 0) * rowStride;
      if (offset + values.length > total) {
        throw new Error(
          `loadWeightBundle: ${name} part ${part.key} (${partShape.join('x')}) overruns ` +
            `shape ${shape.join('x')}`,
        );
      }
      data.set(values, offset);
    }

    if (meta.quant) {
      if (meta.quant.mode !== 'symmetric_per_row' || meta.quant.axis !== 0) {
        throw new Error(`loadWeightBundle: ${name} uses unsupported quantisation ${meta.quant.mode}`);
      }
      const scaleFile = await shard(meta.quant.scale_shard);
      const scaleRaw = scaleFile.raw(meta.quant.scale_key);
      const scales = decodeToFloat32(scaleRaw.dtype, scaleRaw.bytes);
      if (scales.length !== shape[0]) {
        throw new Error(
          `loadWeightBundle: ${name} has ${shape[0]} rows but ${scales.length} scales`,
        );
      }
      for (let row = 0; row < shape[0]; row += 1) {
        const scale = scales[row];
        const base = row * rowStride;
        for (let i = 0; i < rowStride; i += 1) data[base + i] *= scale;
      }
    }

    const tensor = { shape, data };
    cache.set(name, tensor);
    return tensor;
  }

  // Resolve every tensor up front: the CPU/WebGPU denoisers are synchronous,
  // so they need the full set in hand. `read` stays available for partial use.
  const names = Object.keys(tensorMeta);
  const resolved = Object.create(null);
  for (const name of names) resolved[name] = await read(name);

  return {
    names,
    manifest,
    precision: manifest.precision ?? 'fp32',
    config: manifest.config ?? {},
    has: (name) => name in resolved,
    get: (name) => {
      const tensor = resolved[name];
      if (!tensor) throw new Error(`weights: missing tensor ${name}`);
      return tensor;
    },
    loadAll: async () => resolved,
  };
}

// ===========================================================================
// 3. deterministic RNG
// ===========================================================================

/**
 * Small splitmix64-seeded xorshift128+ PRNG with a Box-Muller normal.
 *
 * Used so sampling runs are reproducible in the browser. It intentionally does
 * *not* reproduce PyTorch's or NumPy's streams — nothing in the model depends
 * on matching a specific random draw, only on the draws being well distributed.
 */
export function createRng(seed = 1) {
  let s0 = 0x9e3779b9 ^ (seed >>> 0);
  let s1 = 0x243f6a88 ^ Math.imul(seed >>> 0, 0x85ebca6b);
  let s2 = 0xb7e15162;
  let s3 = 0x172b54c1 ^ Math.imul(seed >>> 0, 0xc2b2ae35);
  const next = () => {
    const t = s1 << 9;
    let r = Math.imul(s1, 5);
    r = ((r << 7) | (r >>> 25)) >>> 0;
    r = Math.imul(r, 9) >>> 0;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = ((s3 << 11) | (s3 >>> 21)) >>> 0;
    return r >>> 0;
  };
  for (let i = 0; i < 16; i += 1) next();

  let spare = null;
  const uniform = () => next() / 4294967296;
  const gaussian = () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = uniform();
    while (v === 0) v = uniform();
    const radius = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
  return { uniform, gaussian, next };
}

/** Uniformly random rotation matrix (Shoemake's quaternion method), row-major 3x3. */
export function randomRotation(rng) {
  const u1 = rng.uniform();
  const u2 = rng.uniform();
  const u3 = rng.uniform();
  const a = Math.sqrt(1 - u1);
  const b = Math.sqrt(u1);
  const q = [
    a * Math.sin(2 * Math.PI * u2),
    a * Math.cos(2 * Math.PI * u2),
    b * Math.sin(2 * Math.PI * u3),
    b * Math.cos(2 * Math.PI * u3),
  ];
  const [x, y, z, w] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

// ===========================================================================
// 4. geometry
// ===========================================================================

/** Masked centroid of `[n, 3]` coordinates (`artios.utils.geometry.calc_centroid`). */
export function calcCentroid(coords, mask = null, eps = 1e-12) {
  const n = coords.length / 3;
  let cx = 0;
  let cy = 0;
  let cz = 0;
  let count = 0;
  for (let i = 0; i < n; i += 1) {
    const m = mask ? (mask[i] ? 1 : 0) : 1;
    if (!m) continue;
    cx += coords[i * 3];
    cy += coords[i * 3 + 1];
    cz += coords[i * 3 + 2];
    count += 1;
  }
  const denom = count + eps;
  return [cx / denom, cy / denom, cz / denom];
}

/**
 * `centre_random_augmentation` (AF3 Algorithm 19) for a single sample.
 *
 * Centres on the masked centroid, applies a rotation, adds `sTrans * t`, then
 * zeroes out masked-off rows. Supply `rotation`/`translation` to make it
 * deterministic (the oracle fixtures do exactly this); otherwise they are drawn
 * from `rng`.
 *
 * @returns {{coords: Float32Array, rotation: number[][], translation: number[], centroid: number[]}}
 */
export function centreRandomAugmentation(
  coords,
  { mask = null, sTrans = 1.0, centreOnly = false, rotation = null, translation = null, rng = null, eps = 1e-12 } = {},
) {
  const n = coords.length / 3;
  const centroid = calcCentroid(coords, mask, eps);
  const out = new Float32Array(n * 3);

  if (centreOnly) {
    for (let i = 0; i < n; i += 1) {
      const m = mask ? (mask[i] ? 1 : 0) : 1;
      for (let c = 0; c < 3; c += 1) out[i * 3 + c] = m * (coords[i * 3 + c] - centroid[c]);
    }
    return { coords: out, rotation: null, translation: [0, 0, 0], centroid };
  }

  const source = rng ?? createRng(1);
  const R = rotation ?? randomRotation(source);
  const t = translation ?? [source.gaussian(), source.gaussian(), source.gaussian()];

  for (let i = 0; i < n; i += 1) {
    const m = mask ? (mask[i] ? 1 : 0) : 1;
    const x = coords[i * 3] - centroid[0];
    const y = coords[i * 3 + 1] - centroid[1];
    const z = coords[i * 3 + 2] - centroid[2];
    for (let r = 0; r < 3; r += 1) {
      out[i * 3 + r] = m * (R[r][0] * x + R[r][1] * y + R[r][2] * z + sTrans * t[r]);
    }
  }
  return { coords: out, rotation: R, translation: t, centroid };
}

/**
 * `random_transform` from `artios.utils.geometry`, used per reference conformer.
 * Centres (optionally) then optionally rotates/translates; the translation is
 * uniform in `[-maxTranslation, maxTranslation]` and is applied *before* the
 * rotation, matching `np.dot(points + translation, R.T)`.
 */
export function randomTransform(
  coords,
  { maxTranslation = 1.0, applyAugmentation = false, centralize = true, mask = null, rng = null, eps = 1e-12 } = {},
) {
  const n = coords.length / 3;
  const out = Float32Array.from(coords);
  if (centralize) {
    const centroid = calcCentroid(out, mask, eps);
    for (let i = 0; i < n; i += 1) {
      for (let c = 0; c < 3; c += 1) out[i * 3 + c] -= centroid[c];
    }
  }
  if (applyAugmentation) {
    const source = rng ?? createRng(1);
    const t = [0, 1, 2].map(() => (source.uniform() * 2 - 1) * maxTranslation);
    const R = randomRotation(source);
    for (let i = 0; i < n; i += 1) {
      const x = out[i * 3] + t[0];
      const y = out[i * 3 + 1] + t[1];
      const z = out[i * 3 + 2] + t[2];
      for (let r = 0; r < 3; r += 1) {
        out[i * 3 + r] = R[r][0] * x + R[r][1] * y + R[r][2] * z;
      }
    }
  }
  if (mask) {
    for (let i = 0; i < n; i += 1) {
      if (!mask[i]) {
        out[i * 3] = 0;
        out[i * 3 + 1] = 0;
        out[i * 3 + 2] = 0;
      }
    }
  }
  return out;
}

/**
 * Largest-eigenvalue eigenvector of a symmetric 4x4 matrix, by cyclic Jacobi.
 *
 * Only used by `kabschAlign` below, on the 4x4 Davenport matrix. Jacobi is
 * chosen over an SVD because it is a few lines, needs no pivoting and is
 * numerically well behaved at this size.
 */
function topEigenvector4(matrix) {
  const a = matrix.map((row) => row.slice());
  let v = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  for (let sweep = 0; sweep < 32; sweep += 1) {
    let off = 0;
    for (let p = 0; p < 4; p += 1) {
      for (let q = p + 1; q < 4; q += 1) off += a[p][q] * a[p][q];
    }
    if (off < 1e-24) break;
    for (let p = 0; p < 4; p += 1) {
      for (let q = p + 1; q < 4; q += 1) {
        if (Math.abs(a[p][q]) < 1e-30) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k += 1) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k += 1) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k += 1) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i += 1) if (a[i][i] > a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

/**
 * Weighted Kabsch alignment of `pred` onto `target` (`align_pred_to_true`).
 *
 * Both are `[n, 3]`. Returns the rigidly transformed `pred` plus the rotation
 * and translation used. Reflections are never allowed: the quaternion
 * formulation only produces proper rotations, which is what the reference's
 * `allowing_reflection=False` path does via its determinant correction.
 *
 * Used by the CPU denoiser's `createPredictor` to align the previous
 * prediction onto the current noisy coordinates before self-conditioning.
 */
export function kabschAlign(pred, target, { mask = null, weight = null } = {}) {
  const n = pred.length / 3;
  const w = new Float64Array(n);
  let wSum = 0;
  for (let i = 0; i < n; i += 1) {
    const m = mask ? (mask[i] ? 1 : 0) : 1;
    w[i] = m * (weight ? weight[i] : 1);
    wSum += w[i];
  }
  if (wSum === 0) {
    return { coords: Float32Array.from(pred), rotation: null, translation: [0, 0, 0] };
  }

  const cp = [0, 0, 0];
  const ct = [0, 0, 0];
  for (let i = 0; i < n; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      cp[c] += w[i] * pred[i * 3 + c];
      ct[c] += w[i] * target[i * 3 + c];
    }
  }
  for (let c = 0; c < 3; c += 1) {
    cp[c] /= wSum;
    ct[c] /= wSum;
  }

  // Cross-covariance H = sum_i w_i * (pred_i - cp) (target_i - ct)^T
  const h = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < n; i += 1) {
    if (w[i] === 0) continue;
    const px = pred[i * 3] - cp[0];
    const py = pred[i * 3 + 1] - cp[1];
    const pz = pred[i * 3 + 2] - cp[2];
    const tx = target[i * 3] - ct[0];
    const ty = target[i * 3 + 1] - ct[1];
    const tz = target[i * 3 + 2] - ct[2];
    h[0][0] += w[i] * px * tx; h[0][1] += w[i] * px * ty; h[0][2] += w[i] * px * tz;
    h[1][0] += w[i] * py * tx; h[1][1] += w[i] * py * ty; h[1][2] += w[i] * py * tz;
    h[2][0] += w[i] * pz * tx; h[2][1] += w[i] * pz * ty; h[2][2] += w[i] * pz * tz;
  }

  // Davenport's symmetric 4x4 (Horn 1987 eq. 38-39, with S_ab = sum_i p_a t_b):
  // its largest-eigenvalue eigenvector is the quaternion rotating pred onto
  // target.
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = h;
  const k = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, yy - xx - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, zz - xx - yy],
  ];
  const [qw, qx, qy, qz] = topEigenvector4(k);
  const norm = Math.hypot(qw, qx, qy, qz) || 1;
  const [w0, x0, y0, z0] = [qw / norm, qx / norm, qy / norm, qz / norm];
  const R = [
    [1 - 2 * (y0 * y0 + z0 * z0), 2 * (x0 * y0 - z0 * w0), 2 * (x0 * z0 + y0 * w0)],
    [2 * (x0 * y0 + z0 * w0), 1 - 2 * (x0 * x0 + z0 * z0), 2 * (y0 * z0 - x0 * w0)],
    [2 * (x0 * z0 - y0 * w0), 2 * (y0 * z0 + x0 * w0), 1 - 2 * (x0 * x0 + y0 * y0)],
  ];

  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    const px = pred[i * 3] - cp[0];
    const py = pred[i * 3 + 1] - cp[1];
    const pz = pred[i * 3 + 2] - cp[2];
    for (let r = 0; r < 3; r += 1) {
      out[i * 3 + r] = R[r][0] * px + R[r][1] * py + R[r][2] * pz + ct[r];
    }
  }
  return { coords: out, rotation: R, translation: ct };
}

// ===========================================================================
// 5. EDM noise schedule and sampling
// ===========================================================================

/**
 * Karras/AF3 inference noise schedule (`EDMInferenceNoiseScheduler`).
 *
 * `sigma[i] = SIGMA_DATA * (sMax^(1/rho) + i/n * (sMin^(1/rho) - sMax^(1/rho)))^rho`
 * with the final entry forced to exactly 0.
 *
 * @param {number} nSteps
 * @returns {Float64Array} `nSteps + 1` noise levels, descending
 */
export function edmNoiseSchedule(nSteps, { sigmaData = SIGMA_DATA, ...overrides } = {}) {
  if (!Number.isInteger(nSteps) || nSteps < 1) {
    throw new Error(`edmNoiseSchedule: nSteps must be a positive integer, got ${nSteps}`);
  }
  const { sMax, sMin, rho } = { ...EDM_SCHEDULE, ...overrides };
  const hi = sMax ** (1 / rho);
  const lo = sMin ** (1 / rho);
  const stepSize = 1 / nSteps;
  const out = new Float64Array(nSteps + 1);
  for (let i = 0; i <= nSteps; i += 1) {
    out[i] = sigmaData * (hi + i * stepSize * (lo - hi)) ** rho;
  }
  out[nSteps] = 0;
  return out;
}

/** EDM preconditioning coefficients for a noise level (`DiffusionModule.c_*`). */
export function edmPreconditioning(noiseLevel, sigmaData = SIGMA_DATA) {
  const s2 = sigmaData * sigmaData;
  const n2 = noiseLevel * noiseLevel;
  return {
    cIn: 1 / Math.sqrt(s2 + n2),
    cOut: (sigmaData * noiseLevel) / Math.sqrt(s2 + n2),
    cSkip: s2 / (s2 + n2),
  };
}

/** `EDMTrainingNoiseSampler`: `exp(z * pStd + pMean) * SIGMA_DATA`. */
export function edmTrainingNoiseLevel(standardNormal, sigmaData = SIGMA_DATA) {
  const { pMean, pStd } = EDM_TRAINING_NOISE;
  return Math.exp(standardNormal * pStd + pMean) * sigmaData;
}

/**
 * One `EDMSampler` integration step (AF3 predictor-corrector).
 *
 * Centres and re-augments the current coordinates, injects the churn noise for
 * `gamma`, calls the denoiser at the inflated level `tHat`, then takes the
 * scaled Euler step toward `cTau`. With `gamma = 0` this is the deterministic
 * Euler/Heun predictor; with `gamma > 0` the added `randn` term makes it the
 * Euler-Maruyama form of the same SDE.
 *
 * @param {object} options
 * @param {Float32Array} options.x current coordinates `[n, 3]`
 * @param {number} options.cTauLast noise level at the start of the step
 * @param {number} options.cTau noise level at the end of the step
 * @param {(xNoisy: Float32Array, tHat: number) => Float32Array} options.predict denoiser
 * @param {Uint8Array|boolean[]} [options.mask] per-atom mask used for centring
 * @param {object} [options.rng] deterministic RNG (`createRng`)
 * @param {object} [options.augmentation] fixed `{rotation, translation}` override
 * @returns {{x: Float32Array, xNoisy: Float32Array, xDenoised: Float32Array, tHat: number, gamma: number, deltaNoiseLevel: number}}
 */
export function edmSamplerStep({
  x,
  cTauLast,
  cTau,
  predict,
  mask = null,
  rng = null,
  augmentation = null,
  sampler = EDM_SAMPLER,
}) {
  const { gamma0, gammaMin, noiseScaleLambda, stepScaleEta } = { ...EDM_SAMPLER, ...sampler };
  const source = rng ?? createRng(1);
  const n = x.length / 3;

  const centred = centreRandomAugmentation(x, {
    mask,
    sTrans: 1.0,
    rng: source,
    rotation: augmentation?.rotation ?? null,
    translation: augmentation?.translation ?? null,
  }).coords;

  const gamma = cTau > gammaMin ? gamma0 : 0.0;
  const tHat = cTauLast * (1.0 + gamma);
  const deltaNoiseLevel = Math.sqrt(Math.max(0, tHat * tHat - cTauLast * cTauLast));

  const xNoisy = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i += 1) {
    xNoisy[i] = centred[i] + noiseScaleLambda * deltaNoiseLevel * source.gaussian();
  }

  const xDenoised = predict(xNoisy, tHat);
  if (!xDenoised || xDenoised.length !== n * 3) {
    throw new Error(`edmSamplerStep: predict must return ${n * 3} values, got ${xDenoised?.length}`);
  }

  const dt = cTau - tHat;
  const next = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i += 1) {
    const delta = (xNoisy[i] - xDenoised[i]) / tHat;
    next[i] = xNoisy[i] + stepScaleEta * dt * delta;
  }
  return { x: next, xNoisy, xDenoised, tHat, gamma, deltaNoiseLevel };
}

/**
 * Full `EDMSampler.sample` loop: start from `schedule[0] * randn` and integrate.
 *
 * @returns {{coords: Float32Array, trajectory: Float32Array[]|null, schedule: Float64Array}}
 */
export function edmSample({
  nAtom,
  nSteps,
  predict,
  mask = null,
  rng = null,
  seed = 1,
  xInit = null,
  trackTrajectory = false,
}) {
  const source = rng ?? createRng(seed);
  const schedule = edmNoiseSchedule(nSteps);
  let x = xInit ? Float32Array.from(xInit) : new Float32Array(nAtom * 3);
  if (!xInit) {
    for (let i = 0; i < nAtom * 3; i += 1) x[i] = schedule[0] * source.gaussian();
  }
  const trajectory = trackTrajectory ? [Float32Array.from(x)] : null;

  for (let i = 0; i < nSteps; i += 1) {
    const step = edmSamplerStep({
      x,
      cTauLast: schedule[i],
      cTau: schedule[i + 1],
      predict,
      mask,
      rng: source,
    });
    x = step.x;
    if (trajectory) trajectory.push(Float32Array.from(x));
  }
  return { coords: x, trajectory, schedule };
}

// ===========================================================================
// 6. CIF parsing
// ===========================================================================

const WHITESPACE = new Set([' ', '\t', '\r', '\n']);

/**
 * Tokenise CIF 1.1 text.
 *
 * Handles bare tokens, `'`/`"` quoted strings, `;`-delimited multi-line text
 * fields and `#` comments. Returns `{value, quoted}` so that an unquoted `.` or
 * `?` (CIF's "inapplicable"/"unknown") can be told apart from the literal
 * strings `'.'` and `'?'`.
 */
export function* tokenizeCif(text) {
  const n = text.length;
  let i = 0;
  let atLineStart = true;
  while (i < n) {
    const ch = text[i];
    if (ch === '\n') {
      atLineStart = true;
      i += 1;
      continue;
    }
    if (WHITESPACE.has(ch)) {
      i += 1;
      continue;
    }
    if (ch === '#') {
      while (i < n && text[i] !== '\n') i += 1;
      continue;
    }
    if (ch === ';' && atLineStart) {
      // Multi-line text field: runs until a ';' at the start of a line.
      let end = i + 1;
      const lines = [];
      for (;;) {
        const nl = text.indexOf('\n', end);
        const line = nl === -1 ? text.slice(end) : text.slice(end, nl);
        if (lines.length > 0 && line.startsWith(';')) {
          end += 1; // consume the closing ';'
          break;
        }
        lines.push(line);
        if (nl === -1) {
          end = n;
          break;
        }
        end = nl + 1;
      }
      // The remainder of the opening ';' line is conventionally empty.
      if (lines.length > 1 && lines[0] === '') lines.shift();
      i = end;
      atLineStart = false;
      yield { value: lines.join('\n'), quoted: true };
      continue;
    }
    if (ch === "'" || ch === '"') {
      // A quote only closes when followed by whitespace or end of input.
      let j = i + 1;
      let value = '';
      for (;;) {
        const k = text.indexOf(ch, j);
        if (k === -1) {
          value += text.slice(j);
          j = n;
          break;
        }
        const after = text[k + 1];
        if (after === undefined || WHITESPACE.has(after)) {
          value += text.slice(j, k);
          j = k + 1;
          break;
        }
        value += text.slice(j, k + 1);
        j = k + 1;
      }
      i = j;
      atLineStart = false;
      yield { value, quoted: true };
      continue;
    }
    let j = i;
    while (j < n && !WHITESPACE.has(text[j])) j += 1;
    const value = text.slice(i, j);
    i = j;
    atLineStart = false;
    yield { value, quoted: false };
  }
}

function splitTag(tag) {
  const dot = tag.indexOf('.');
  if (dot === -1) return [tag.slice(1), '']; // e.g. "_entry" with no item
  return [tag.slice(1, dot), tag.slice(dot + 1)];
}

const nullish = (token) => !token.quoted && (token.value === '.' || token.value === '?');

/**
 * Parse CIF text into data blocks.
 *
 * Each block is `{name, categories}` where a category is
 * `{tags: string[], rows: (string|null)[][]}`. Key/value categories become a
 * single-row table, so both forms read the same way.
 *
 * @param {string} text
 * @returns {Array<{name: string, categories: Record<string, {tags: string[], rows: (string|null)[][]}>}>}
 */
export function parseCifBlocks(text) {
  const blocks = [];
  let block = null;

  const ensureBlock = () => {
    if (!block) {
      block = { name: '', categories: Object.create(null) };
      blocks.push(block);
    }
    return block;
  };
  const category = (name) => {
    const b = ensureBlock();
    if (!b.categories[name]) b.categories[name] = { tags: [], rows: [] };
    return b.categories[name];
  };

  const tokens = [...tokenizeCif(text)];
  let i = 0;
  while (i < tokens.length) {
    const token = tokens[i];
    const lower = token.quoted ? '' : token.value.toLowerCase();

    if (lower.startsWith('data_')) {
      block = { name: token.value.slice(5), categories: Object.create(null) };
      blocks.push(block);
      i += 1;
      continue;
    }
    if (lower === 'loop_') {
      i += 1;
      const tags = [];
      while (i < tokens.length && !tokens[i].quoted && tokens[i].value.startsWith('_')) {
        tags.push(tokens[i].value);
        i += 1;
      }
      if (tags.length === 0) continue;
      const [catName] = splitTag(tags[0]);
      const cat = category(catName);
      const items = tags.map((t) => splitTag(t)[1]);
      if (cat.tags.length === 0) cat.tags = items;
      while (i < tokens.length) {
        const head = tokens[i];
        const headLower = head.quoted ? '' : head.value.toLowerCase();
        if (
          !head.quoted &&
          (head.value.startsWith('_') ||
            headLower === 'loop_' ||
            headLower.startsWith('data_') ||
            headLower === 'stop_' ||
            headLower.startsWith('save_'))
        ) {
          break;
        }
        const row = [];
        for (let c = 0; c < items.length; c += 1) {
          if (i >= tokens.length) break;
          row.push(nullish(tokens[i]) ? null : tokens[i].value);
          i += 1;
        }
        if (row.length === items.length) cat.rows.push(row);
      }
      continue;
    }
    if (!token.quoted && token.value.startsWith('_')) {
      const [catName, item] = splitTag(token.value);
      const value = i + 1 < tokens.length ? tokens[i + 1] : null;
      i += 2;
      const cat = category(catName);
      if (cat.rows.length === 0) cat.rows.push([]);
      let col = cat.tags.indexOf(item);
      if (col === -1) {
        cat.tags.push(item);
        col = cat.tags.length - 1;
      }
      cat.rows[0][col] = value && !nullish(value) ? value.value : null;
      continue;
    }
    i += 1; // stop_, save_, or stray token
  }
  return blocks;
}

/** Parse CIF text and return the first data block. */
export function parseCif(text) {
  const blocks = parseCifBlocks(text);
  if (blocks.length === 0) throw new Error('parseCif: no data block found');
  return blocks[0];
}

/** Rows of a category as plain objects keyed by item name (`null` for `.`/`?`). */
export function categoryRows(block, categoryName) {
  const cat = block.categories?.[categoryName];
  if (!cat) return [];
  return cat.rows.map((row) => {
    const out = Object.create(null);
    for (let c = 0; c < cat.tags.length; c += 1) out[cat.tags[c]] = row[c] ?? null;
    return out;
  });
}

const num = (value, fallback = null) => {
  if (value === null || value === undefined || value === '') return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

// ===========================================================================
// 7. Chemical Component Dictionary -> reference conformers
// ===========================================================================

/**
 * Parse an RCSB CCD component file (`https://files.rcsb.org/ligands/download/ATP.cif`,
 * or one data block of the full `components.cif`) into a reference conformer.
 *
 * AlphaFold3 (SI 2.8) generates the reference conformer with RDKit ETKDGv3 and
 * falls back to the CCD *ideal* coordinates, then to the representative
 * coordinates, then to zeros. RDKit cannot run in a browser, so this starts at
 * the ideal coordinates — the first fallback — and records which source each
 * atom used in `coordSource`. Hydrogens and (by default) leaving atoms are
 * dropped, matching `get_component_atom_array(keep_leaving_atoms=False,
 * keep_hydrogens=False)`.
 *
 * @param {string|object} source CIF text or an already-parsed block
 * @param {object} [options]
 * @param {boolean} [options.keepHydrogens=false]
 * @param {boolean} [options.keepLeavingAtoms=false]
 * @returns {{id: string, type: string, molType: string, atoms: Array<object>,
 *            atomIndex: Map<string, number>, bonds: Array<object>, coordSource: string}}
 */
export function parseChemComp(source, { keepHydrogens = false, keepLeavingAtoms = false } = {}) {
  const block = typeof source === 'string' ? parseCif(source) : source;
  const comp = categoryRows(block, 'chem_comp')[0] ?? {};
  const id = (comp.id ?? block.name ?? '').toUpperCase();
  const type = (comp.type ?? '').toUpperCase();

  let idealCount = 0;
  let modelCount = 0;
  const atoms = [];
  for (const row of categoryRows(block, 'chem_comp_atom')) {
    const element = (row.type_symbol ?? '').toUpperCase();
    const leaving = (row.pdbx_leaving_atom_flag ?? 'N').toUpperCase() === 'Y';
    if (!keepHydrogens && HYDROGEN_LIKE.has(element)) continue;
    if (!keepLeavingAtoms && leaving) continue;

    const ideal = [
      num(row.pdbx_model_Cartn_x_ideal),
      num(row.pdbx_model_Cartn_y_ideal),
      num(row.pdbx_model_Cartn_z_ideal),
    ];
    const model = [num(row.model_Cartn_x), num(row.model_Cartn_y), num(row.model_Cartn_z)];
    const hasIdeal = ideal.every((v) => v !== null);
    const hasModel = model.every((v) => v !== null);
    if (hasIdeal) idealCount += 1;
    else if (hasModel) modelCount += 1;

    atoms.push({
      atomId: row.atom_id ?? '',
      altAtomId: row.alt_atom_id ?? null,
      element,
      charge: num(row.charge, 0),
      leaving,
      aromatic: (row.pdbx_aromatic_flag ?? 'N').toUpperCase() === 'Y',
      coord: hasIdeal ? ideal : hasModel ? model : [0, 0, 0],
      coordSource: hasIdeal ? 'ideal' : hasModel ? 'model' : 'zeros',
      hasCoord: hasIdeal || hasModel,
    });
  }

  const bonds = categoryRows(block, 'chem_comp_bond').map((row) => ({
    atom1: row.atom_id_1 ?? '',
    atom2: row.atom_id_2 ?? '',
    order: (row.value_order ?? 'SING').toUpperCase(),
    aromatic: (row.pdbx_aromatic_flag ?? 'N').toUpperCase() === 'Y',
  }));

  const atomIndex = new Map();
  atoms.forEach((atom, i) => {
    atomIndex.set(atom.atomId, i);
    if (atom.altAtomId && !atomIndex.has(atom.altAtomId)) atomIndex.set(atom.altAtomId, i);
  });

  let molType = 'ligand';
  if (type.includes('PEPTIDE') && type !== 'PEPTIDE-LIKE') molType = 'protein';
  else if (type.includes('DNA')) molType = 'dna';
  else if (type.includes('RNA')) molType = 'rna';

  return {
    id,
    type,
    molType,
    atoms,
    atomIndex,
    bonds,
    coordSource: idealCount >= modelCount ? 'ideal' : 'model',
  };
}

/**
 * Parse the `atom_site` loop of an mmCIF structure (a motif CIF, say).
 *
 * Uses the `label_*` hierarchy with `auth_*` as a fallback, drops alternate
 * locations other than the first, and skips hydrogens by default.
 *
 * @returns {Array<{atomName: string, resName: string, chainId: string, resId: number,
 *                  element: string, charge: number, coord: number[], hetero: boolean,
 *                  model: number, insCode: string|null}>}
 */
export function parseMmcifAtoms(text, { keepHydrogens = false, model = null, altLoc = null } = {}) {
  const block = typeof text === 'string' ? parseCif(text) : text;
  const out = [];
  let firstModel = null;
  for (const row of categoryRows(block, 'atom_site')) {
    const element = (row.type_symbol ?? '').toUpperCase();
    if (!keepHydrogens && HYDROGEN_LIKE.has(element)) continue;

    const modelNum = num(row.pdbx_PDB_model_num, 1);
    if (firstModel === null) firstModel = modelNum;
    const wanted = model ?? firstModel;
    if (modelNum !== wanted) continue;

    const alt = row.label_alt_id ?? null;
    if (alt !== null && altLoc !== null && alt !== altLoc) continue;

    const resIdRaw = row.label_seq_id ?? row.auth_seq_id;
    const coord = [num(row.Cartn_x, 0), num(row.Cartn_y, 0), num(row.Cartn_z, 0)];
    out.push({
      atomName: (row.label_atom_id ?? row.auth_atom_id ?? '').replace(/^["']|["']$/g, ''),
      resName: (row.label_comp_id ?? row.auth_comp_id ?? '').toUpperCase(),
      chainId: row.label_asym_id ?? row.auth_asym_id ?? 'A',
      resId: num(resIdRaw, num(row.auth_seq_id, 0)),
      insCode: row.pdbx_PDB_ins_code ?? null,
      element,
      charge: num(row.pdbx_formal_charge, 0),
      coord,
      occupancy: num(row.occupancy, 1),
      hetero: (row.group_PDB ?? 'ATOM').toUpperCase() === 'HETATM',
      model: modelNum,
      altLoc: alt,
    });
  }

  if (altLoc === null) {
    // Keep only the first altloc seen for each (chain, resId, atomName).
    const seen = new Set();
    return out.filter((atom) => {
      const key = `${atom.chainId}|${atom.resId}|${atom.insCode ?? ''}|${atom.atomName}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }
  return out;
}

/** One-hot encode element symbols into `[n, 128]` (`ref_element`). */
export function encodeElementOneHot(symbols) {
  const n = symbols.length;
  const out = new Float32Array(n * N_ELEMENT);
  for (let i = 0; i < n; i += 1) {
    const sym = String(symbols[i] ?? '').toUpperCase();
    const index = ELEMENT_INDEX.get(sym);
    if (index === undefined) {
      throw new Error(`encodeElementOneHot: unknown element ${JSON.stringify(symbols[i])}`);
    }
    out[i * N_ELEMENT + index] = 1;
  }
  return out;
}

/**
 * Encode atom names as `[n, 4, 64]` one-hots (`ref_atom_name_chars`).
 * Each name is right-padded to 4 characters and each character mapped to
 * `clip(ord(c) - 32, 0, 63)`.
 */
export function encodeAtomNameChars(names) {
  const n = names.length;
  const stride = ATOM_NAME_LENGTH * ATOM_NAME_ALPHABET;
  const out = new Float32Array(n * stride);
  for (let i = 0; i < n; i += 1) {
    const name = String(names[i] ?? '');
    for (let j = 0; j < ATOM_NAME_LENGTH; j += 1) {
      const code = j < name.length ? name.charCodeAt(j) : 32; // pad with space
      const slot = Math.min(Math.max(code - 32, 0), ATOM_NAME_ALPHABET - 1);
      out[i * stride + j * ATOM_NAME_ALPHABET + slot] = 1;
    }
  }
  return out;
}

/** One-hot encode residue names into `[n, 32]` (`restype` / `motif_restype`). */
export function encodeRestypeOneHot(resNames) {
  const n = resNames.length;
  const out = new Float32Array(n * N_RESTYPE);
  for (let i = 0; i < n; i += 1) {
    const name = String(resNames[i] ?? 'UNK').toUpperCase();
    const index = RESIDUE_VOCAB[name] ?? RESIDUE_VOCAB.UNK;
    out[i * N_RESTYPE + index] = 1;
  }
  return out;
}

/**
 * `ref_space_uid`: the lexicographic rank of each atom's `(chain, resId)` pair
 * among the unique pairs present, matching
 * `np.unique(np.stack([asym_id_int, res_id]), axis=0, return_inverse=True)`.
 *
 * @param {Array<{chainId: string, resId: number, insCode?: string|null}>} atoms
 * @param {Map<string, number>} [chainOrder] chain id -> asym_id_int
 * @returns {Int32Array}
 */
export function computeRefSpaceUid(atoms, chainOrder = null) {
  const chains = chainOrder ?? new Map();
  if (!chainOrder) {
    for (const atom of atoms) {
      if (!chains.has(atom.chainId)) chains.set(atom.chainId, chains.size);
    }
  }
  const keys = atoms.map((atom) => [chains.get(atom.chainId) ?? 0, atom.resId, atom.insCode ?? '']);
  const unique = [...new Set(keys.map((k) => JSON.stringify(k)))]
    .map((s) => JSON.parse(s))
    .sort((a, b) => a[0] - b[0] || a[1] - b[1] || String(a[2]).localeCompare(String(b[2])));
  const rank = new Map(unique.map((k, i) => [JSON.stringify(k), i]));
  const out = new Int32Array(atoms.length);
  for (let i = 0; i < atoms.length; i += 1) out[i] = rank.get(JSON.stringify(keys[i]));
  return out;
}

/**
 * Build the AlphaFold3 reference-conformer feature block for a parsed structure.
 *
 * Every atom's `ref_pos` comes from its residue's CCD component (ideal
 * coordinates), re-centred per reference conformer instance and optionally
 * randomly rotated/translated the way `ContextGenerator` does
 * (`ref_pos_augment`). Atoms whose component or atom name is unknown get a zero
 * position and `ref_mask = 0`.
 *
 * @param {object} options
 * @param {Array<object>} options.atoms from `parseMmcifAtoms`
 * @param {Map<string, object>|Record<string, object>} options.components CCD code -> `parseChemComp` result
 * @param {boolean} [options.augment=false] apply the random per-conformer transform
 * @param {object} [options.rng] deterministic RNG when `augment` is set
 * @returns {{nAtom: number, ref_pos: Float32Array, ref_mask: Int32Array,
 *            ref_charge: Float32Array, ref_element: Float32Array,
 *            ref_atom_name_chars: Float32Array, ref_space_uid: Int32Array,
 *            unresolvedAtoms: Array<object>, missingComponents: string[]}}
 */
export function buildReferenceFeatures({ atoms, components, augment = false, rng = null, ligandAtomRename = false }) {
  const lookup =
    components instanceof Map ? components : new Map(Object.entries(components ?? {}));
  const nAtom = atoms.length;
  const refPos = new Float32Array(nAtom * 3);
  const refMask = new Int32Array(nAtom);
  const refCharge = new Float32Array(nAtom);
  const refSpaceUid = computeRefSpaceUid(atoms);
  const unresolvedAtoms = [];
  const missingComponents = new Set();

  for (let i = 0; i < nAtom; i += 1) {
    const atom = atoms[i];
    const comp = lookup.get(atom.resName);
    if (!comp) {
      missingComponents.add(atom.resName);
      unresolvedAtoms.push({ index: i, reason: 'missing-component', ...atom });
      refCharge[i] = atom.charge ?? 0;
      continue;
    }
    const slot = comp.atomIndex.get(atom.atomName);
    if (slot === undefined) {
      unresolvedAtoms.push({ index: i, reason: 'missing-atom', ...atom });
      refCharge[i] = atom.charge ?? 0;
      continue;
    }
    const entry = comp.atoms[slot];
    refPos[i * 3] = entry.coord[0];
    refPos[i * 3 + 1] = entry.coord[1];
    refPos[i * 3 + 2] = entry.coord[2];
    refMask[i] = entry.hasCoord ? 1 : 0;
    refCharge[i] = entry.charge ?? 0;
  }

  // Centre (and optionally augment) each reference conformer independently,
  // exactly as ContextGenerator.get_reference_conformer_context does.
  const byUid = new Map();
  for (let i = 0; i < nAtom; i += 1) {
    const uid = refSpaceUid[i];
    if (!byUid.has(uid)) byUid.set(uid, []);
    byUid.get(uid).push(i);
  }
  for (const indices of byUid.values()) {
    const sub = new Float32Array(indices.length * 3);
    const subMask = new Uint8Array(indices.length);
    for (let k = 0; k < indices.length; k += 1) {
      subMask[k] = refMask[indices[k]];
      for (let c = 0; c < 3; c += 1) sub[k * 3 + c] = refPos[indices[k] * 3 + c];
    }
    const transformed = randomTransform(sub, {
      mask: subMask,
      applyAugmentation: augment,
      centralize: true,
      rng,
    });
    for (let k = 0; k < indices.length; k += 1) {
      for (let c = 0; c < 3; c += 1) refPos[indices[k] * 3 + c] = transformed[k * 3 + c];
    }
  }

  const atomNames = ligandAtomRename ? renameLigandAtoms(atoms, lookup) : atoms.map((a) => a.atomName);

  return {
    nAtom,
    ref_pos: refPos,
    ref_mask: refMask,
    ref_charge: refCharge,
    ref_element: encodeElementOneHot(atoms.map((a) => a.element)),
    ref_atom_name_chars: encodeAtomNameChars(atomNames),
    ref_space_uid: refSpaceUid,
    unresolvedAtoms,
    missingComponents: [...missingComponents],
  };
}

/** Exclusive end index of the residue starting at `start` (chain + resId + insCode + resName). */
function residueEnd(atoms, start) {
  const a = atoms[start];
  let j = start;
  while (
    j < atoms.length &&
    atoms[j].chainId === a.chainId &&
    atoms[j].resId === a.resId &&
    (atoms[j].insCode ?? '') === (a.insCode ?? '') &&
    atoms[j].resName === a.resName
  ) {
    j += 1;
  }
  return j;
}

/**
 * `ContextGenerator.get_renamed_atom_names`: rewrite ligand atom names to
 * `<ELEMENT><n>` so the ligand identity does not leak through atom names.
 */
export function renameLigandAtoms(atoms, components) {
  const lookup = components instanceof Map ? components : new Map(Object.entries(components ?? {}));
  const names = atoms.map((a) => a.atomName);
  let i = 0;
  while (i < atoms.length) {
    const j = residueEnd(atoms, i);
    const comp = lookup.get(atoms[i].resName);
    if ((comp?.molType ?? 'ligand') === 'ligand') {
      const counts = new Map();
      for (let k = i; k < j; k += 1) {
        const element = atoms[k].element.toUpperCase();
        const next = (counts.get(element) ?? 0) + 1;
        counts.set(element, next);
        names[k] = `${element}${next}`;
      }
    }
    i = j;
  }
  return names;
}

/**
 * Group atoms into tokens the AlphaFold3 way: one token per polymer residue,
 * one token per atom for ligands and non-standard residues.
 *
 * @returns {{nToken: number, atom_to_token_idx: Int32Array, token_index: Int32Array,
 *            residue_index: Int32Array, asym_id: Int32Array, entity_id: Int32Array,
 *            sym_id: Int32Array, centre_atom_index: Int32Array, token_res_name: string[],
 *            is_ligand_token: Uint8Array, is_protein_token: Uint8Array}}
 */
export function buildTokenization({ atoms, components, centreAtomNames = { protein: 'CA', dna: "C1'", rna: "C1'" } }) {
  const lookup = components instanceof Map ? components : new Map(Object.entries(components ?? {}));
  const atomToToken = new Int32Array(atoms.length);
  const tokens = [];
  const chains = new Map();

  let i = 0;
  while (i < atoms.length) {
    const j = residueEnd(atoms, i);
    const resName = atoms[i].resName;
    const comp = lookup.get(resName);
    const molType = comp?.molType ?? (resName in RESIDUE_VOCAB ? 'protein' : 'ligand');
    const isPolymer = molType !== 'ligand' && resName in RESIDUE_VOCAB;

    if (!chains.has(atoms[i].chainId)) chains.set(atoms[i].chainId, chains.size);
    const asymId = chains.get(atoms[i].chainId);

    if (isPolymer) {
      const wanted = centreAtomNames[molType];
      let centre = i;
      for (let k = i; k < j; k += 1) {
        if (atoms[k].atomName === wanted) {
          centre = k;
          break;
        }
      }
      const tokenId = tokens.length;
      tokens.push({ resName, resId: atoms[i].resId, asymId, molType, centre, chainId: atoms[i].chainId });
      for (let k = i; k < j; k += 1) atomToToken[k] = tokenId;
    } else {
      for (let k = i; k < j; k += 1) {
        const tokenId = tokens.length;
        tokens.push({ resName, resId: atoms[i].resId, asymId, molType, centre: k, chainId: atoms[i].chainId });
        atomToToken[k] = tokenId;
      }
    }
    i = j;
  }

  const nToken = tokens.length;
  const chainEntity = new Map();
  for (const token of tokens) {
    if (!chainEntity.has(token.chainId)) chainEntity.set(token.chainId, chainEntity.size);
  }

  return {
    nToken,
    atom_to_token_idx: atomToToken,
    token_index: Int32Array.from({ length: nToken }, (_, k) => k),
    residue_index: Int32Array.from(tokens, (t) => t.resId),
    asym_id: Int32Array.from(tokens, (t) => t.asymId),
    entity_id: Int32Array.from(tokens, (t) => chainEntity.get(t.chainId)),
    sym_id: new Int32Array(nToken),
    centre_atom_index: Int32Array.from(tokens, (t) => t.centre),
    token_res_name: tokens.map((t) => t.resName),
    is_protein_token: Uint8Array.from(tokens, (t) => (t.molType === 'protein' ? 1 : 0)),
    is_ligand_token: Uint8Array.from(tokens, (t) => (t.molType === 'ligand' ? 1 : 0)),
  };
}

// ===========================================================================
// 8. motif / depth / RASA conditioning features
// ===========================================================================

/**
 * Build the motif feature block.
 *
 * `is_motif_related` expands the motif selection to every atom sharing a
 * `ref_space_uid` with a motif atom (i.e. whole residues/conformers), matching
 * `MotifMaker`. `motif_restype` is the one-hot restype of motif-related,
 * resolved tokens; `motif_coordinate` is the token centre coordinate of motif
 * tokens and zero elsewhere.
 *
 * @param {object} options
 * @param {Int32Array|number[]} options.atomToTokenIdx
 * @param {Int32Array|number[]} options.refSpaceUid
 * @param {Uint8Array|boolean[]} options.isMotifAtom
 * @param {number} options.nToken
 * @param {Int32Array|number[]} options.centreAtomIndex
 * @param {string[]} options.tokenResName
 * @param {Float32Array|number[]} options.coordinate flat `[nAtom, 3]`
 * @param {Uint8Array|boolean[]} [options.isResolved]
 */
export function buildMotifFeatures({
  atomToTokenIdx,
  refSpaceUid,
  isMotifAtom,
  nToken,
  centreAtomIndex,
  tokenResName,
  coordinate,
  isResolved = null,
}) {
  const nAtom = atomToTokenIdx.length;
  const isMotif = Uint8Array.from({ length: nAtom }, (_, i) => (isMotifAtom[i] ? 1 : 0));

  const motifUids = new Set();
  for (let i = 0; i < nAtom; i += 1) if (isMotif[i]) motifUids.add(refSpaceUid[i]);
  const isMotifRelated = Uint8Array.from({ length: nAtom }, (_, i) =>
    motifUids.has(refSpaceUid[i]) ? 1 : 0,
  );

  const isMotifToken = new Uint8Array(nToken);
  const isMotifRelatedToken = new Uint8Array(nToken);
  for (let t = 0; t < nToken; t += 1) {
    const centre = centreAtomIndex[t];
    isMotifToken[t] = isMotif[centre] ? 1 : 0;
    isMotifRelatedToken[t] = isMotifRelated[centre] ? 1 : 0;
  }

  const restype = encodeRestypeOneHot(tokenResName);
  const motifRestype = new Float32Array(nToken * N_RESTYPE);
  const motifCoordinate = new Float32Array(nToken * 3);
  for (let t = 0; t < nToken; t += 1) {
    const centre = centreAtomIndex[t];
    const resolved = isResolved ? (isResolved[centre] ? 1 : 0) : 1;
    if (isMotifRelatedToken[t] && resolved) {
      for (let c = 0; c < N_RESTYPE; c += 1) {
        motifRestype[t * N_RESTYPE + c] = restype[t * N_RESTYPE + c];
      }
    }
    if (isMotifToken[t]) {
      for (let c = 0; c < 3; c += 1) motifCoordinate[t * 3 + c] = coordinate[centre * 3 + c];
    }
  }

  return {
    is_motif: isMotif,
    is_motif_related: isMotifRelated,
    is_motif_token: isMotifToken,
    is_motif_related_token: isMotifRelatedToken,
    motif_restype: motifRestype,
    motif_coordinate: motifCoordinate,
  };
}

/**
 * `compute_motif_depth_norm`: `||CoM(motif) - CoM(non-motif)|| / Rg(complex)`,
 * or -1 (the null / unconditioned value) when either side is empty.
 */
export function computeMotifDepthNorm({ coordinate, isMotif, isResolved = null, eps = 1e-6 }) {
  const n = coordinate.length / 3;
  const motif = [];
  const nonMotif = [];
  const all = [];
  for (let i = 0; i < n; i += 1) {
    if (isResolved && !isResolved[i]) continue;
    all.push(i);
    (isMotif[i] ? motif : nonMotif).push(i);
  }
  if (motif.length === 0 || nonMotif.length === 0) return -1.0;

  const mean = (indices) => {
    const out = [0, 0, 0];
    for (const i of indices) for (let c = 0; c < 3; c += 1) out[c] += coordinate[i * 3 + c];
    return out.map((v) => v / indices.length);
  };
  const cMotif = mean(motif);
  const cNon = mean(nonMotif);
  const cAll = mean(all);
  let sq = 0;
  for (const i of all) {
    for (let c = 0; c < 3; c += 1) {
      const d = coordinate[i * 3 + c] - cAll[c];
      sq += d * d;
    }
  }
  const rg = Math.sqrt(sq / all.length);
  const dist = Math.hypot(cMotif[0] - cNon[0], cMotif[1] - cNon[1], cMotif[2] - cNon[2]);
  return dist / (rg + eps);
}

/** Per-token `motif_depth_norm`, broadcast from a single scalar. */
export function buildMotifDepthFeature(nToken, value = -1.0) {
  return new Float32Array(nToken).fill(value);
}

/**
 * Per-token ligand `rasa`. Non-ligand tokens are always -1 (null); ligand
 * tokens take their per-token value, falling back to `fallback` for NaN.
 */
export function buildRasaFeature({ nToken, isLigandToken, values = null, fallback = -1.0 }) {
  const out = new Float32Array(nToken).fill(-1.0);
  for (let t = 0; t < nToken; t += 1) {
    if (!isLigandToken[t]) continue;
    const value = values ? values[t] : fallback;
    out[t] = Number.isFinite(value) ? value : fallback;
  }
  return out;
}

/**
 * Discretise a conditioning scalar: fine bins over `[0, max)`, one overflow bin
 * at `>= max`, and a dedicated null bin for negatives
 * (`get_motif_depth_condition` / `get_rasa_condition`).
 */
export function conditioningBinIndex(value, binSize, nFineBins) {
  if (value < 0) return nFineBins + 1; // null bin
  const index = Math.floor(value / binSize);
  return Math.min(Math.max(index, 0), nFineBins); // nFineBins == overflow bin
}

/** `motif_depth_norm` -> bin index in `[0, 41]`. */
export const motifDepthBinIndex = (value) =>
  conditioningBinIndex(value, CONDITIONING_BINS.motifDepth.binSize, CONDITIONING_BINS.motifDepth.nFineBins);

/** `rasa` -> bin index in `[0, 21]`. */
export const rasaBinIndex = (value) =>
  conditioningBinIndex(value, CONDITIONING_BINS.rasa.binSize, CONDITIONING_BINS.rasa.nFineBins);

/** Lower/upper edges of the motif pair-distance bins (`motif_lower/upper_bins`). */
export function motifDistanceBins({ min, max, binSize } = CONDITIONING_BINS.motifDist) {
  const n = Math.round((max - min) / binSize);
  const lower = new Float64Array(n);
  const upper = new Float64Array(n);
  for (let i = 0; i < n; i += 1) lower[i] = min + i * binSize;
  for (let i = 0; i < n - 1; i += 1) upper[i] = lower[i + 1];
  upper[n - 1] = 1e6; // open-ended last bin
  return { lower, upper, nBins: n };
}

/**
 * `artios.model.utils.one_hot`: `(x > lower) & (x < upper)`.
 *
 * Note the strict inequalities — a value exactly on a bin edge lands in no bin
 * at all, and values at or below `min` are all-zero rather than clamped.
 */
export function oneHotBins(value, lower, upper) {
  const out = new Float32Array(lower.length);
  for (let i = 0; i < lower.length; i += 1) {
    out[i] = value > lower[i] && value < upper[i] ? 1 : 0;
  }
  return out;
}
