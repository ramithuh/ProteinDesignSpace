/**
 * Proton-PottsMPNN — shared, zero-dependency core.
 *
 * Everything in this module is pure ES2022 with no imports: the v6 protonation
 * vocabulary, a safetensors reader (FP32 / FP16 / per-row INT8), a PDB backbone
 * parser, the N/CA/C/O + virtual-CB kNN/RBF featurizer, the Potts energy
 * (`E_global`), the pH-selective gap (`E_P - E_D`) and the pH-selective
 * block-descent optimizer.
 *
 * The CPU backend (`cpu/index.js`) and the WebGPU backend (`webgpu/index.js`)
 * both build on this file; only the encoder / Potts-head matmuls differ between
 * them.
 *
 * Layout conventions, chosen to match the PyTorch checkpoint so no tensor ever
 * has to be transposed:
 *   - a Linear weight is row-major `[out, in]`, so one output channel is one
 *     contiguous row and `y[n] = bias[n] + dot(x, W[n])`.
 *   - `X`        `[L, 4, 3]`   backbone N, CA, C, O
 *   - `edgeIdx`  `[L, K]`      kNN graph, `edgeIdx[i * K] === i` (the self edge)
 *   - `Eraw`     `[L, K, 416]` 16 positional + 400 RBF edge features
 *   - `hE`       `[L, K, H]`   encoder edge states
 *   - `J`        `[L, K, V, V]` Potts couplings; `J[i, 0]` is diagonal and its
 *     diagonal is the single-body field `h[i]`
 */

/* ========================================================================== *
 * 1. The v6 protonation vocabulary
 * ========================================================================== */

/** The 20 standard amino acids, in the checkpoint's (alphabetical) order. */
export const STANDARD_AA = Object.freeze([
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
]);

export const UNKNOWN_TOKEN = 'UNK';

/**
 * v6 explicit protonation states. v6 predicts the CHARGE state, not the
 * HID/HIE tautomer, so neutral His is the single token `HIS-S`; the v3/v4
 * tautomer tokens (`HID`, `HIE`) and the imidazolate `HIS-D` do not exist here.
 */
export const V6_PROTONATION_TOKENS = Object.freeze([
  'HIS-P', 'HIS-S', 'HIS-A',
  'ASP-P', 'ASP-D', 'ASP-A',
  'GLU-P', 'GLU-D', 'GLU-A',
]);

/** The checkpoint's 30-token vocabulary: 21 standard (incl. UNK) + 9 states. */
export const V6_TOKENS = Object.freeze([
  ...STANDARD_AA, UNKNOWN_TOKEN, ...V6_PROTONATION_TOKENS,
]);

export const TOKEN_TO_IDX = Object.freeze(
  Object.fromEntries(V6_TOKENS.map((t, i) => [t, i])),
);

/** Suffix -> physical protonation state, direction-neutral (see ph/vocab_meta). */
const SUFFIX_STATE = { P: 'protonated', S: 'deprotonated', D: 'deprotonated', A: 'ambiguous' };

/** Side-chain formal charge per titratable parent and state. */
const TITRATABLE_CHARGE = {
  HIS: { protonated: 1, deprotonated: 0 },
  ASP: { protonated: 0, deprotonated: -1 },
  GLU: { protonated: 0, deprotonated: -1 },
};

/** Per-token metadata: `{ name, index, parent, parentIndex, state, charge }`. */
export const V6_TOKEN_META = Object.freeze(V6_TOKENS.map((name, index) => {
  const dash = name.indexOf('-');
  if (dash < 0) {
    return Object.freeze({
      name, index, parent: name, parentIndex: index,
      state: 'unspecified', charge: null,
    });
  }
  const parent = name.slice(0, dash);
  const suffix = name.slice(dash + 1);
  const state = SUFFIX_STATE[suffix];
  if (!state || !TITRATABLE_CHARGE[parent]) throw new Error(`bad v6 token ${name}`);
  return Object.freeze({
    name, index, parent, parentIndex: TOKEN_TO_IDX[parent], state,
    charge: state === 'ambiguous' ? null : TITRATABLE_CHARGE[parent][state],
  });
}));

/** `canonicalMap[t]` = index of token `t`'s bare parent (HIS-P -> HIS). */
export const CANONICAL_MAP = Object.freeze(V6_TOKEN_META.map((m) => m.parentIndex));

/**
 * Default state for a titratable residue that the caller has not pinned, at
 * physiological pH (~7.4): the acids are deprotonated and His is neutral.
 * Every other residue keeps its own three-letter code.
 */
export const PHYSIOLOGICAL_STATE = Object.freeze({
  ASP: 'ASP-D', GLU: 'GLU-D', HIS: 'HIS-S',
});

/** Protonated microstate -> its deprotonated contrast partners, for `E_P - E_D`. */
export const V6_DEP_MAP = Object.freeze({
  'HIS-P': Object.freeze(['HIS-S']),
  'ASP-P': Object.freeze(['ASP-D']),
  'GLU-P': Object.freeze(['GLU-D']),
});

/** Microstates forbidden as *design choices* (centre pinning is separate). */
export const DEFAULT_FORBIDDEN_TOKENS = Object.freeze(['HIS-A', 'ASP-A', 'GLU-A']);

export const THREE_TO_ONE = Object.freeze({
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G',
  HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S',
  THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V', UNK: 'X',
});

/** Parents that have both a protonated and a deprotonated token in v6. */
export const TITRATABLE_PARENTS = Object.freeze(['ASP', 'GLU', 'HIS']);

/** The token name for `parent` in `state` ('protonated' | 'deprotonated'). */
export function resolveToken(parent, state) {
  const hits = V6_TOKEN_META.filter((m) => m.parent === parent && m.state === state);
  if (hits.length !== 1) {
    throw new Error(`v6 has ${hits.length} ${state} tokens for ${parent}`);
  }
  return hits[0].name;
}

/** `[V]` boolean mask of tokens the optimizer may propose. */
export function validTokenMask(forbidden = DEFAULT_FORBIDDEN_TOKENS) {
  const mask = new Uint8Array(V6_TOKENS.length).fill(1);
  mask[TOKEN_TO_IDX[UNKNOWN_TOKEN]] = 0;
  for (const tok of forbidden) {
    const idx = TOKEN_TO_IDX[tok];
    if (idx !== undefined) mask[idx] = 0;
  }
  return mask;
}

/** `[V]` boolean mask of tokens whose canonical parent is in `parents`. */
export function parentTokenMask(parents) {
  const want = new Set(parents);
  return Uint8Array.from(V6_TOKEN_META, (m) => (want.has(m.parent) ? 1 : 0));
}

/** One-letter sequence of the canonical parents of `tokens`. */
export function canonicalSequence(tokens) {
  let out = '';
  for (const t of tokens) out += THREE_TO_ONE[V6_TOKENS[CANONICAL_MAP[t]]] ?? 'X';
  return out;
}

/** Three-letter token names (with protonation state) of `tokens`. */
export function tokenNames(tokens) {
  return Array.from(tokens, (t) => V6_TOKENS[t]);
}

/* ========================================================================== *
 * 2. Numeric helpers
 * ========================================================================== */

const INV_SQRT_PI = 0.5641895835477563;

/**
 * `erf(x)` via W. J. Cody's three-interval rational approximation (the same
 * algorithm as SPECFUN/`calerf`), accurate to ~1e-16.
 *
 * PyTorch's `nn.GELU` is the exact erf form, not the tanh approximation, so an
 * ad-hoc 1e-7 erf would put a floor under the FP32 parity we are checking.
 */
export function erf(x) {
  const y = Math.abs(x);
  if (y <= 0.46875) {
    const A = [3.16112374387056560e0, 1.13864154151050156e2, 3.77485237685302021e2,
      3.20937758913846947e3, 1.85777706184603153e-1];
    const B = [2.36012909523441209e1, 2.44024637934444173e2, 1.28261652607737228e3,
      2.84423683343917062e3];
    const z = y * y;
    let num = A[4] * z;
    let den = z;
    for (let i = 0; i < 3; i += 1) {
      num = (num + A[i]) * z;
      den = (den + B[i]) * z;
    }
    return x * (num + A[3]) / (den + B[3]);
  }
  let res;
  if (y <= 4.0) {
    const C = [5.64188496988670089e-1, 8.88314979438837594e0, 6.61191906371416295e1,
      2.98635138197400131e2, 8.81952221241769090e2, 1.71204761263407058e3,
      2.05107837782607147e3, 1.23033935479799725e3, 2.15311535474403846e-8];
    const D = [1.57449261107098347e1, 1.17693950891312499e2, 5.37181101862009858e2,
      1.62138957456669019e3, 3.29079923573345963e3, 4.36261909014324716e3,
      3.43936767414372164e3, 1.23033935480374942e3];
    let num = C[8] * y;
    let den = y;
    for (let i = 0; i < 7; i += 1) {
      num = (num + C[i]) * y;
      den = (den + D[i]) * y;
    }
    res = (num + C[7]) / (den + D[7]);
  } else {
    const P = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1,
      1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2];
    const Q = [2.56852019228982242e0, 1.87295284992346047e0, 5.27905102951428412e-1,
      6.05183413124413191e-2, 2.33520497626869185e-3];
    const z = 1 / (y * y);
    let num = P[5] * z;
    let den = z;
    for (let i = 0; i < 4; i += 1) {
      num = (num + P[i]) * z;
      den = (den + Q[i]) * z;
    }
    res = z * (num + P[4]) / (den + Q[4]);
    res = (INV_SQRT_PI - res) / y;
  }
  // Cody's split of exp(-y^2) into a table-exact part and a small remainder,
  // which keeps the exponential from losing bits for large y.
  const zt = Math.trunc(y * 16) / 16;
  const del = (y - zt) * (y + zt);
  const erfc = Math.exp(-zt * zt) * Math.exp(-del) * res;
  return x > 0 ? 1 - erfc : erfc - 1;
}

/** Exact (erf-based) GELU, matching `torch.nn.GELU()`. */
export function gelu(x) {
  return 0.5 * x * (1 + erf(x * Math.SQRT1_2));
}

/** In-place exact GELU over a Float32Array. */
export function geluInPlace(a, n = a.length) {
  for (let i = 0; i < n; i += 1) a[i] = gelu(a[i]);
  return a;
}

/**
 * `out[m, n] = bias[n] + dot(A[m, :], W[n, :])` for row-major `A [M, Kin]` and
 * `W [N, Kin]`, i.e. a PyTorch `Linear` applied to `M` stacked rows.
 *
 * Four `m` rows are driven through each weight row at once so `W` is streamed
 * from cache a quarter as often; the accumulators are plain JS doubles, which
 * is a superset of the FP32 accumulation the reference performs.
 */
export function linearForward(A, M, Kin, W, N, bias, out) {
  const dst = out ?? new Float32Array(M * N);
  let m = 0;
  for (; m + 4 <= M; m += 4) {
    const a0 = m * Kin;
    const a1 = a0 + Kin;
    const a2 = a1 + Kin;
    const a3 = a2 + Kin;
    const o0 = m * N;
    for (let n = 0; n < N; n += 1) {
      const w = n * Kin;
      let s0 = 0; let s1 = 0; let s2 = 0; let s3 = 0;
      for (let k = 0; k < Kin; k += 1) {
        const wv = W[w + k];
        s0 += A[a0 + k] * wv;
        s1 += A[a1 + k] * wv;
        s2 += A[a2 + k] * wv;
        s3 += A[a3 + k] * wv;
      }
      const b = bias ? bias[n] : 0;
      dst[o0 + n] = s0 + b;
      dst[o0 + N + n] = s1 + b;
      dst[o0 + 2 * N + n] = s2 + b;
      dst[o0 + 3 * N + n] = s3 + b;
    }
  }
  for (; m < M; m += 1) {
    const a0 = m * Kin;
    const o0 = m * N;
    for (let n = 0; n < N; n += 1) {
      const w = n * Kin;
      let s = 0;
      for (let k = 0; k < Kin; k += 1) s += A[a0 + k] * W[w + k];
      dst[o0 + n] = s + (bias ? bias[n] : 0);
    }
  }
  return dst;
}

/**
 * Row-wise LayerNorm over the last dimension, in place.
 * Biased variance + `eps = 1e-5`, matching `torch.nn.LayerNorm` defaults.
 */
export function layerNormInPlace(a, rows, dim, weight, bias, eps = 1e-5) {
  for (let r = 0; r < rows; r += 1) {
    const off = r * dim;
    let mean = 0;
    for (let i = 0; i < dim; i += 1) mean += a[off + i];
    mean /= dim;
    let varSum = 0;
    for (let i = 0; i < dim; i += 1) {
      const d = a[off + i] - mean;
      varSum += d * d;
    }
    const inv = 1 / Math.sqrt(varSum / dim + eps);
    for (let i = 0; i < dim; i += 1) {
      a[off + i] = (a[off + i] - mean) * inv * weight[i] + bias[i];
    }
  }
  return a;
}

/** `log_softmax` over the last dimension, in place. */
export function logSoftmaxInPlace(a, rows, dim) {
  for (let r = 0; r < rows; r += 1) {
    const off = r * dim;
    let max = -Infinity;
    for (let i = 0; i < dim; i += 1) if (a[off + i] > max) max = a[off + i];
    let sum = 0;
    for (let i = 0; i < dim; i += 1) sum += Math.exp(a[off + i] - max);
    const logSum = max + Math.log(sum);
    for (let i = 0; i < dim; i += 1) a[off + i] -= logSum;
  }
  return a;
}

/** `softmax` of `-(x - min(x)) / T` over a flat array; returns a new array. */
function boltzmann(values, temperature) {
  let min = Infinity;
  for (const v of values) if (v < min) min = v;
  const p = new Float64Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    p[i] = Math.exp(-(values[i] - min) / temperature);
    sum += p[i];
  }
  for (let i = 0; i < p.length; i += 1) p[i] /= sum;
  return p;
}

/** Deterministic, seedable PRNG (mulberry32) for reproducible sampling. */
export function makeRng(seed = 0) {
  let s = (seed >>> 0) || 0x9e3779b9;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleCategorical(probs, rng) {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i += 1) {
    acc += probs[i];
    if (u < acc) return i;
  }
  return probs.length - 1;
}

/** Relative RMS error `||a - b|| / ||b||`, the parity metric used by the tests. */
export function relRMS(a, b) {
  if (a.length !== b.length) throw new Error(`length ${a.length} vs ${b.length}`);
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  if (den === 0) return Math.sqrt(num);
  return Math.sqrt(num / den);
}

/* ========================================================================== *
 * 3. safetensors
 * ========================================================================== */

/** Decode one IEEE-754 binary16 held in the low 16 bits of `h`. */
export function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 5.960464477539063e-8; // 2^-24, subnormal
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * (2 ** (exp - 15));
}

/**
 * Parse a safetensors buffer into `{ header, metadata, tensor(name) }`.
 * `tensor(name)` returns `{ dtype, shape, view }` with `view` a typed array
 * over the original bytes — no copy until a dtype needs converting.
 */
export function parseSafetensors(buffer) {
  const bytes = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
  const base = bytes.byteOffset;
  const dv = new DataView(bytes.buffer, base, 8);
  const headerLen = Number(dv.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(
    bytes.subarray(8, 8 + headerLen),
  );
  const header = JSON.parse(headerJson);
  const metadata = header.__metadata__ ?? {};
  const dataStart = base + 8 + headerLen;
  const entries = new Map();
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    entries.set(name, info);
  }
  const tensor = (name) => {
    const info = entries.get(name);
    if (!info) throw new Error(`safetensors: no tensor "${name}"`);
    const [start, end] = info.data_offsets;
    const off = dataStart + start;
    const n = end - start;
    let view;
    switch (info.dtype) {
      case 'F32': view = new Float32Array(bytes.buffer, off, n / 4); break;
      case 'F16': view = new Uint16Array(bytes.buffer, off, n / 2); break;
      case 'I8': view = new Int8Array(bytes.buffer, off, n); break;
      case 'U8': view = new Uint8Array(bytes.buffer, off, n); break;
      default: throw new Error(`safetensors: unsupported dtype ${info.dtype}`);
    }
    return { dtype: info.dtype, shape: info.shape, view };
  };
  return { header, metadata, names: [...entries.keys()], has: (n) => entries.has(n), tensor };
}

/**
 * Materialise every tensor of a bundle as FP32, whatever it was stored as.
 *
 * FP16 is widened on load; a per-row INT8 tensor `w` is rebuilt from `w` and
 * `w.scale` as `q * scale[row]`. The result is the single representation the CPU
 * kernels consume, so a precision switch never changes a code path — only the
 * values. (The WebGPU backend keeps the packed form instead; see
 * `webgpu/engine.js`.)
 */
export function materializeWeights(bundle) {
  const out = new Map();
  for (const name of bundle.names) {
    if (name.endsWith('.scale')) continue;
    const t = bundle.tensor(name);
    if (t.dtype === 'F32') {
      out.set(name, { data: t.view, shape: t.shape, storedAs: 'F32' });
    } else if (t.dtype === 'F16') {
      const data = new Float32Array(t.view.length);
      for (let i = 0; i < data.length; i += 1) data[i] = halfToFloat(t.view[i]);
      out.set(name, { data, shape: t.shape, storedAs: 'F16' });
    } else if (t.dtype === 'I8') {
      const scaleName = `${name}.scale`;
      if (!bundle.has(scaleName)) throw new Error(`int8 tensor ${name} has no .scale`);
      const scale = bundle.tensor(scaleName).view;
      const [rows, cols] = t.shape;
      if (t.shape.length !== 2) throw new Error(`int8 ${name} must be 2-D`);
      if (scale.length !== rows) throw new Error(`int8 ${name}: scale length mismatch`);
      const data = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r += 1) {
        const s = scale[r];
        const off = r * cols;
        for (let c = 0; c < cols; c += 1) data[off + c] = t.view[off + c] * s;
      }
      out.set(name, { data, shape: t.shape, storedAs: 'I8' });
    } else {
      throw new Error(`unsupported stored dtype ${t.dtype} for ${name}`);
    }
  }
  return out;
}

/* ========================================================================== *
 * 4. PDB backbone parsing
 * ========================================================================== */

const BACKBONE_ATOMS = Object.freeze(['N', 'CA', 'C', 'O']);

/**
 * Parse `ATOM` records into backbone-complete standard-amino-acid residues.
 *
 * Only N/CA/C/O are read (the model is backbone-only; CB is virtual), the first
 * altloc of an atom wins, and a residue that is missing any backbone atom is
 * dropped because its RBF features cannot be built.
 */
export function parseBackbonePDB(text) {
  const byKey = new Map();
  const order = [];
  const standard = new Set(STANDARD_AA);
  for (const line of text.split('\n')) {
    if (!line.startsWith('ATOM')) continue;
    const name = line.slice(12, 16).trim();
    if (!BACKBONE_ATOMS.includes(name)) continue;
    const resName = line.slice(17, 20).trim();
    if (!standard.has(resName)) continue;
    const chain = line[21];
    const resSeq = Number.parseInt(line.slice(22, 26), 10);
    const icode = line[26];
    const key = `${chain}|${resSeq}|${icode}`;
    let res = byKey.get(key);
    if (res === undefined) {
      res = { chain, resSeq, icode, resName, atoms: new Map() };
      byKey.set(key, res);
      order.push(key);
    }
    if (res.atoms.has(name)) continue; // first altloc wins
    res.atoms.set(name, [
      Number.parseFloat(line.slice(30, 38)),
      Number.parseFloat(line.slice(38, 46)),
      Number.parseFloat(line.slice(46, 54)),
    ]);
  }
  const out = [];
  for (const key of order) {
    const res = byKey.get(key);
    if (BACKBONE_ATOMS.every((a) => res.atoms.has(a))) out.push(res);
  }
  return out;
}

/**
 * Turn parsed residues into the model's input tensors.
 *
 * `tokens` default to the physiological state of each residue
 * (`PHYSIOLOGICAL_STATE`); `overrides` pins individual positions, which is how
 * a caller states a protonation state without the HBPLUS/propka labeller. Keys
 * are either a 0-based position or a `"<chain><resSeq>"` label such as `"A42"`.
 *
 * `R_idx` is the 0-based index of a residue *within its chain* and
 * `chainLabels` numbers the chains by first appearance, matching the reference
 * pipeline's `FeaturizeNonAtomizedTokens`.
 */
export function buildStructure(residues, options = {}) {
  const { overrides = {}, residueMask = null, designedResidueMask = null } = options;
  const L = residues.length;
  const X = new Float32Array(L * 4 * 3);
  const tokens = new Int32Array(L);
  const rIdx = new Int32Array(L);
  const chainLabels = new Int32Array(L);
  const chainIds = [];
  const counters = new Map();
  const labels = [];

  const byLabel = new Map();
  for (let i = 0; i < L; i += 1) byLabel.set(`${residues[i].chain}${residues[i].resSeq}`, i);
  const resolvedOverrides = new Map();
  for (const [key, tok] of Object.entries(overrides)) {
    const pos = /^-?\d+$/.test(key) ? Number(key) : byLabel.get(key);
    if (pos === undefined) throw new Error(`override target "${key}" is not in the structure`);
    if (!(tok in TOKEN_TO_IDX)) throw new Error(`override token "${tok}" is not a v6 token`);
    resolvedOverrides.set(pos, tok);
  }

  for (let i = 0; i < L; i += 1) {
    const res = residues[i];
    for (let a = 0; a < 4; a += 1) {
      const c = res.atoms.get(BACKBONE_ATOMS[a]);
      X[(i * 4 + a) * 3] = c[0];
      X[(i * 4 + a) * 3 + 1] = c[1];
      X[(i * 4 + a) * 3 + 2] = c[2];
    }
    const name = resolvedOverrides.get(i) ?? PHYSIOLOGICAL_STATE[res.resName] ?? res.resName;
    const idx = TOKEN_TO_IDX[name];
    if (idx === undefined) throw new Error(`residue ${res.resName} has no v6 token`);
    tokens[i] = idx;
    labels.push(name);
    if (!counters.has(res.chain)) {
      counters.set(res.chain, 0);
      chainIds.push(res.chain);
    }
    chainLabels[i] = chainIds.indexOf(res.chain);
    rIdx[i] = counters.get(res.chain);
    counters.set(res.chain, rIdx[i] + 1);
  }

  const mask = residueMask ? Uint8Array.from(residueMask, (v) => (v ? 1 : 0))
    : new Uint8Array(L).fill(1);
  const designed = designedResidueMask
    ? Uint8Array.from(designedResidueMask, (v) => (v ? 1 : 0))
    : mask.slice();
  for (let i = 0; i < L; i += 1) {
    if (designed[i] && !mask[i]) throw new Error(`designed residue ${i} is not valid`);
  }
  return {
    L,
    X,
    tokens,
    tokenNames: labels,
    rIdx,
    chainLabels,
    chainIds,
    residueMask: mask,
    designedResidueMask: designed,
    resName: residues.map((r) => r.resName),
    resSeq: Int32Array.from(residues, (r) => r.resSeq),
    chain: residues.map((r) => r.chain),
  };
}

/** Parse a PDB and build its input tensors in one step. */
export function structureFromPDB(text, options) {
  return buildStructure(parseBackbonePDB(text), options);
}

/* ========================================================================== *
 * 5. Featurizer: virtual CB, kNN graph, RBF, relative positional encoding
 * ========================================================================== */

export const FEATURIZER_CONSTANTS = Object.freeze({
  numRbf: 16,
  minRbfMean: 2.0,
  maxRbfMean: 22.0,
  numNeighbors: 48,
  maxRelativeFeature: 32,
  numPositionalFeatures: 66, // 2 * 32 + 1 (offsets) + 1 (different chain)
  numPositionalEmbeddings: 16,
  distanceEps: 1e-6,
  virtualAtom: Object.freeze({
    weightNormal: 0.58273431,
    weightBond1: -0.56802827,
    weightBond2: -0.54067466,
  }),
});

/**
 * The virtual CB: `w_n * (b1 x b2) + w_1 * b1 + w_2 * b2 + CA`, where
 * `b1 = N - CA` and `b2 = C - CA` — the reference's single
 * `DATA_TO_CALCULATE_VIRTUAL_ATOMS` entry.
 */
export function computeVirtualCB(X, L, out) {
  const cb = out ?? new Float32Array(L * 3);
  const { weightNormal: wn, weightBond1: w1, weightBond2: w2 } =
    FEATURIZER_CONSTANTS.virtualAtom;
  for (let i = 0; i < L; i += 1) {
    const n = (i * 4 + 0) * 3;
    const ca = (i * 4 + 1) * 3;
    const c = (i * 4 + 2) * 3;
    const b1x = X[n] - X[ca];
    const b1y = X[n + 1] - X[ca + 1];
    const b1z = X[n + 2] - X[ca + 2];
    const b2x = X[c] - X[ca];
    const b2y = X[c + 1] - X[ca + 1];
    const b2z = X[c + 2] - X[ca + 2];
    const nx = b1y * b2z - b1z * b2y;
    const ny = b1z * b2x - b1x * b2z;
    const nz = b1x * b2y - b1y * b2x;
    cb[i * 3] = wn * nx + w1 * b1x + w2 * b2x + X[ca];
    cb[i * 3 + 1] = wn * ny + w1 * b1y + w2 * b2y + X[ca + 1];
    cb[i * 3 + 2] = wn * nz + w1 * b1z + w2 * b2z + X[ca + 2];
  }
  return cb;
}

/** `[L, 5, 3]` = N, CA, C, O, virtual CB — the five atoms the RBFs run over. */
export function buildAtomStack(X, L) {
  const atoms = new Float32Array(L * 5 * 3);
  for (let i = 0; i < L; i += 1) {
    for (let a = 0; a < 4; a += 1) {
      for (let d = 0; d < 3; d += 1) atoms[(i * 5 + a) * 3 + d] = X[(i * 4 + a) * 3 + d];
    }
  }
  const cb = computeVirtualCB(X, L);
  for (let i = 0; i < L; i += 1) {
    for (let d = 0; d < 3; d += 1) atoms[(i * 5 + 4) * 3 + d] = cb[i * 3 + d];
  }
  return atoms;
}

/**
 * The kNN graph over CA atoms, reproducing
 * `compute_representative_atom_pairwise_distances`.
 *
 * Distances use `sqrt(d2 + eps)`, so the self distance is `1e-3` rather than 0
 * and slot 0 of every row is the residue itself. Pairs touching an invalid
 * residue are pushed out to that row's maximum distance instead of being
 * dropped, which keeps `K` fixed.
 */
export function buildKnnGraph(atoms, L, residueMask, numNeighbors) {
  const K = Math.min(numNeighbors, L);
  const eps = FEATURIZER_CONSTANTS.distanceEps;
  const edgeIdx = new Int32Array(L * K);
  const d = new Float64Array(L);
  const idx = new Int32Array(L);
  for (let i = 0; i < L; i += 1) {
    const cai = (i * 5 + 1) * 3;
    let rowMax = 0;
    for (let j = 0; j < L; j += 1) {
      const caj = (j * 5 + 1) * 3;
      const dx = atoms[caj] - atoms[cai];
      const dy = atoms[caj + 1] - atoms[cai + 1];
      const dz = atoms[caj + 2] - atoms[cai + 2];
      const valid = residueMask[i] && residueMask[j];
      const dist = valid ? Math.fround(Math.sqrt(dx * dx + dy * dy + dz * dz + eps)) : 0;
      d[j] = dist;
      if (dist > rowMax) rowMax = dist;
      idx[j] = j;
    }
    for (let j = 0; j < L; j += 1) {
      if (!(residueMask[i] && residueMask[j])) d[j] += rowMax;
    }
    const order = Array.prototype.slice.call(idx);
    order.sort((a, b) => (d[a] - d[b]) || (a - b));
    for (let k = 0; k < K; k += 1) edgeIdx[i * K + k] = order[k];
  }
  return { edgeIdx, K };
}

/**
 * The raw `[L, K, 416]` edge features: 16 relative-positional embeddings
 * followed by 400 atom-pair RBFs (5 focal atoms x 5 neighbour atoms x 16 RBFs,
 * focal-major).
 */
export function buildEdgeFeatures(structure, graph, weights) {
  const { L, rIdx, chainLabels, residueMask } = structure;
  const { edgeIdx, K } = graph;
  const atoms = graph.atoms;
  const {
    numRbf, minRbfMean, maxRbfMean, maxRelativeFeature, numPositionalFeatures,
    numPositionalEmbeddings, distanceEps,
  } = FEATURIZER_CONSTANTS;
  const nPosEmb = numPositionalEmbeddings;
  const rbfDim = 25 * numRbf;
  const dim = nPosEmb + rbfDim;
  const out = new Float32Array(L * K * dim);

  const posW = weights.get(
    'graph_featurization_module.positional_embedding.embed_positional_features.weight',
  ).data;
  const posB = weights.get(
    'graph_featurization_module.positional_embedding.embed_positional_features.bias',
  ).data;

  const mus = new Float64Array(numRbf);
  for (let r = 0; r < numRbf; r += 1) {
    mus[r] = minRbfMean + ((maxRbfMean - minRbfMean) * r) / (numRbf - 1);
  }
  const sigma = (maxRbfMean - minRbfMean) / numRbf;
  const invSigma = 1 / sigma;

  // Only N/CA/C/O-complete residues reach here, so all five atoms of a valid
  // residue are valid; the atom mask therefore reduces to the residue mask,
  // exactly as the reference's `torch.all(X_m == 1)` fast path assumes.
  for (let i = 0; i < L; i += 1) {
    for (let k = 0; k < K; k += 1) {
      const j = edgeIdx[i * K + k];
      const base = (i * K + k) * dim;

      // --- relative positional encoding (one-hot 66 -> Linear 16) ---------
      const sameChain = chainLabels[i] === chainLabels[j];
      let slot;
      if (sameChain) {
        const shifted = rIdx[i] - rIdx[j] + maxRelativeFeature;
        slot = Math.min(Math.max(shifted, 0), 2 * maxRelativeFeature);
      } else {
        slot = 2 * maxRelativeFeature + 1;
      }
      for (let e = 0; e < nPosEmb; e += 1) {
        out[base + e] = posW[e * numPositionalFeatures + slot] + posB[e];
      }

      // --- atom-pair RBFs -------------------------------------------------
      const keep = residueMask[i] && residueMask[j];
      if (!keep) continue;
      for (let a = 0; a < 5; a += 1) {
        const pa = (i * 5 + a) * 3;
        const ax = atoms[pa];
        const ay = atoms[pa + 1];
        const az = atoms[pa + 2];
        for (let b = 0; b < 5; b += 1) {
          const pb = (j * 5 + b) * 3;
          const dx = atoms[pb] - ax;
          const dy = atoms[pb + 1] - ay;
          const dz = atoms[pb + 2] - az;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz + distanceEps);
          const off = base + nPosEmb + (a * 5 + b) * numRbf;
          for (let r = 0; r < numRbf; r += 1) {
            const z = (dist - mus[r]) * invSigma;
            out[off + r] = Math.exp(-(z * z));
          }
        }
      }
    }
  }
  return { Eraw: out, dim };
}

/** `maskE[i, k] = residueMask[i] && residueMask[edgeIdx[i, k]]`. */
export function buildEdgeMask(structure, graph) {
  const { L, residueMask } = structure;
  const { edgeIdx, K } = graph;
  const maskE = new Uint8Array(L * K);
  for (let i = 0; i < L; i += 1) {
    for (let k = 0; k < K; k += 1) {
      maskE[i * K + k] = residueMask[i] && residueMask[edgeIdx[i * K + k]] ? 1 : 0;
    }
  }
  return maskE;
}

/**
 * Everything the network needs that does not depend on the weights beyond the
 * positional-embedding Linear: the atom stack, the kNN graph, `Eraw`, `maskE`.
 */
export function featurize(structure, weights, numNeighbors = FEATURIZER_CONSTANTS.numNeighbors) {
  const atoms = buildAtomStack(structure.X, structure.L);
  const { edgeIdx, K } = buildKnnGraph(atoms, structure.L, structure.residueMask, numNeighbors);
  const graph = { atoms, edgeIdx, K };
  const { Eraw, dim } = buildEdgeFeatures(structure, graph, weights);
  const maskE = buildEdgeMask(structure, graph);
  return { ...graph, Eraw, edgeFeatureDim: dim, maskE };
}

/* ========================================================================== *
 * 6. Potts tables: reciprocal merge, field, energies
 * ========================================================================== */

/**
 * Reverse-edge bookkeeping for the kNN graph.
 *
 * `reverseK[i, k]` is the slot in row `j = edgeIdx[i, k]` that points back at
 * `i` (0 when there is none) and `hasReverse[i, k]` flags whether it exists.
 * The graph is asymmetric — roughly one edge in six is not reciprocated — which
 * is why both the coupling merge and the per-position conditional energy need
 * this explicitly.
 */
export function buildReverseIndex(edgeIdx, L, K) {
  const reverseK = new Int32Array(L * K);
  const hasReverse = new Uint8Array(L * K);
  for (let i = 0; i < L; i += 1) {
    for (let k = 0; k < K; k += 1) {
      const j = edgeIdx[i * K + k];
      let found = 0;
      let slot = 0;
      for (let r = 0; r < K; r += 1) {
        if (edgeIdx[j * K + r] === i) { found = 1; slot = r; break; }
      }
      reverseK[i * K + k] = slot;
      hasReverse[i * K + k] = found;
    }
  }
  return { reverseK, hasReverse };
}

/**
 * Finish the Potts head: mask by the edge mask, force the self edge (slot 0) to
 * be diagonal, then symmetrise reciprocal edges by
 * `0.5 * (J[i, k] + J[j, rev]^T)`.
 *
 * Operates in place on `J [L, K, V, V]` (the raw `etab_out` output).
 */
export function finalizePottsTables(J, L, K, V, maskE, edgeIdx, reverse) {
  const VV = V * V;
  // The reference multiplies the whole table by the edge mask before anything
  // else, so an edge touching an invalid residue contributes nothing.
  for (let i = 0; i < L; i += 1) {
    for (let k = 0; k < K; k += 1) {
      if (maskE[i * K + k]) continue;
      J.fill(0, (i * K + k) * VV, (i * K + k) * VV + VV);
    }
  }
  // Self edge carries the single-body field on its diagonal; the off-diagonal
  // is zeroed so `diag(J[i, 0])` cleanly extracts h_i.
  for (let i = 0; i < L; i += 1) {
    const off = i * K * VV;
    for (let a = 0; a < V; a += 1) {
      for (let b = 0; b < V; b += 1) {
        if (a !== b) J[off + a * V + b] = 0;
      }
    }
  }
  const { reverseK, hasReverse } = reverse;
  const merged = new Float32Array(J.length);
  merged.set(J);
  for (let i = 0; i < L; i += 1) {
    for (let k = 0; k < K; k += 1) {
      const e = i * K + k;
      const j = edgeIdx[e];
      const rev = reverseK[e];
      const valid = hasReverse[e] && maskE[e] && maskE[j * K + rev];
      if (!valid) continue;
      const src = e * VV;
      const other = (j * K + rev) * VV;
      for (let a = 0; a < V; a += 1) {
        for (let b = 0; b < V; b += 1) {
          merged[src + a * V + b] = 0.5 * (J[src + a * V + b] + J[other + b * V + a]);
        }
      }
    }
  }
  J.set(merged);
  return J;
}

/** The single-body field `h [L, V]` = the diagonal of each self edge. */
export function extractField(J, L, K, V) {
  const h = new Float32Array(L * V);
  const VV = V * V;
  for (let i = 0; i < L; i += 1) {
    const off = i * K * VV;
    for (let a = 0; a < V; a += 1) h[i * V + a] = J[off + a * V + a];
  }
  return h;
}

/**
 * The Potts Hamiltonian `E_global = H(s) = sum_i sum_k J[i, k, s_i, s_{j(i,k)}]`.
 *
 * Slot 0 contributes the single-body field; slots >= 1 contribute the pair
 * couplings. Reciprocal pairs are counted from both ends — the reference's
 * convention, kept so that every energy here is directly comparable to it.
 */
export function pottsEnergy(J, edgeIdx, L, K, V, seq) {
  const VV = V * V;
  let total = 0;
  for (let i = 0; i < L; i += 1) {
    const si = seq[i];
    for (let k = 0; k < K; k += 1) {
      const e = i * K + k;
      const sj = seq[edgeIdx[e]];
      total += J[e * VV + si * V + sj];
    }
  }
  return total;
}

/**
 * Incoming (graph-transpose) adjacency: for every position the list of
 * `(source, slot)` pairs with `edgeIdx[source, slot] === position`, `slot >= 1`
 * and `source !== position`.
 *
 * `pottsEnergy` sums every *directed* edge, so a position's identity also
 * enters through the edges that merely list it as a neighbour; a conditional
 * energy that ignored them would not match the Hamiltonian.
 */
export function buildIncomingAdjacency(edgeIdx, L, K) {
  const start = new Int32Array(L + 1);
  const counts = new Int32Array(L);
  for (let i = 0; i < L; i += 1) {
    for (let k = 1; k < K; k += 1) {
      const j = edgeIdx[i * K + k];
      if (j !== i) counts[j] += 1;
    }
  }
  for (let i = 0; i < L; i += 1) start[i + 1] = start[i] + counts[i];
  const cursor = Int32Array.from(start.subarray(0, L));
  const srcArr = new Int32Array(start[L]);
  const slotArr = new Int32Array(start[L]);
  for (let i = 0; i < L; i += 1) {
    for (let k = 1; k < K; k += 1) {
      const j = edgeIdx[i * K + k];
      if (j === i) continue;
      const p = cursor[j];
      srcArr[p] = i;
      slotArr[p] = k;
      cursor[j] = p + 1;
    }
  }
  return { start, src: srcArr, slot: slotArr };
}

/**
 * Per-position conditional energies `[L, V]`:
 * `E[p, a]` is every term of `H` that depends on `s_p`, with `s_p = a` —
 * the single-site field, the outgoing couplings `p -> j` and the incoming
 * couplings `m -> p`. Hence
 * `H(s with s_p := a) - H(s) === E[p, a] - E[p, s_p]`.
 */
export function pottsCandidateEnergies(tables, seq, out) {
  const { J, edgeIdx, L, K, V, incoming } = tables;
  const VV = V * V;
  const dst = out ?? new Float32Array(L * V);
  for (let p = 0; p < L; p += 1) {
    const off = p * V;
    const self = p * K * VV;
    for (let a = 0; a < V; a += 1) dst[off + a] = J[self + a * V + a];
    for (let k = 1; k < K; k += 1) {
      const e = p * K + k;
      const sj = seq[edgeIdx[e]];
      const base = e * VV + sj;
      for (let a = 0; a < V; a += 1) dst[off + a] += J[base + a * V];
    }
    for (let t = incoming.start[p]; t < incoming.start[p + 1]; t += 1) {
      const m = incoming.src[t];
      const base = (m * K + incoming.slot[t]) * VV + seq[m] * V;
      for (let a = 0; a < V; a += 1) dst[off + a] += J[base + a];
    }
  }
  return dst;
}

/** Conditional energies `[V]` at a single position — `pottsCandidateEnergies(...)[i]`. */
export function pottsCandidateEnergiesAt(tables, seq, i, out) {
  const { J, edgeIdx, K, V, incoming } = tables;
  const VV = V * V;
  const dst = out ?? new Float64Array(V);
  const self = i * K * VV;
  for (let a = 0; a < V; a += 1) dst[a] = J[self + a * V + a];
  for (let k = 1; k < K; k += 1) {
    const e = i * K + k;
    const base = e * VV + seq[edgeIdx[e]];
    for (let a = 0; a < V; a += 1) dst[a] += J[base + a * V];
  }
  for (let t = incoming.start[i]; t < incoming.start[i + 1]; t += 1) {
    const m = incoming.src[t];
    const base = (m * K + incoming.slot[t]) * VV + seq[m] * V;
    for (let a = 0; a < V; a += 1) dst[a] += J[base + a];
  }
  return dst;
}

/** Exact `dH` of substituting each position with each token, `[L, V]`. */
export function pottsMutationDelta(tables, seq) {
  const cand = pottsCandidateEnergies(tables, seq);
  const { L, V } = tables;
  const out = new Float32Array(L * V);
  for (let p = 0; p < L; p += 1) {
    const cur = cand[p * V + seq[p]];
    for (let a = 0; a < V; a += 1) out[p * V + a] = cand[p * V + a] - cur;
  }
  return out;
}

/**
 * Bundle the Potts tables with the index structures every consumer needs.
 * `J` is taken as-is (already finalized).
 */
export function makePottsTables(J, edgeIdx, L, K, V) {
  return { J, edgeIdx, L, K, V, incoming: buildIncomingAdjacency(edgeIdx, L, K) };
}

/* ========================================================================== *
 * 7. pH-selective gap:  E_P - E_D
 * ========================================================================== */

/**
 * Resolve a `{ position | resSeq/chain, protonationType }` centre spec against
 * a structure into `{ position, protonationType, protIdx, depIdxs }`.
 */
export function resolveCentres(structure, centres, depMap = V6_DEP_MAP) {
  const byLabel = new Map();
  for (let i = 0; i < structure.L; i += 1) {
    byLabel.set(`${structure.chain[i]}${structure.resSeq[i]}`, i);
  }
  return centres.map((c) => {
    let position = c.position;
    if (position === undefined) {
      const key = c.label ?? `${c.chain}${c.resSeq}`;
      position = byLabel.get(key);
      if (position === undefined) throw new Error(`centre "${key}" is not in the structure`);
    }
    const ptype = c.protonationType;
    const protIdx = TOKEN_TO_IDX[ptype];
    if (protIdx === undefined) {
      throw new Error(`centre protonation token "${ptype}" is not in the v6 vocabulary`);
    }
    const deps = depMap[ptype];
    if (!deps || deps.length === 0) {
      throw new Error(`no deprotonated contrast tokens for centre "${ptype}"`);
    }
    const depIdxs = deps.map((d) => {
      const idx = TOKEN_TO_IDX[d];
      if (idx === undefined) throw new Error(`contrast token "${d}" is not a v6 token`);
      return idx;
    });
    return { position, protonationType: ptype, protIdx, depIdxs, resSeq: structure.resSeq?.[position] };
  });
}

/**
 * One centre's selective gap `e_P - mean_d e_D`: the centre's own conditional
 * energy held protonated, minus the mean over its deprotonated contrasts.
 * Lower is better — the protonated microstate is favoured.
 */
export function selectiveGapOf(tables, seq, centre, scratch) {
  const s = scratch?.seq ?? Int32Array.from(seq);
  if (s !== seq) s.set(seq);
  const row = scratch?.row ?? new Float64Array(tables.V);
  s[centre.position] = centre.protIdx;
  const eP = pottsCandidateEnergiesAt(tables, s, centre.position, row)[centre.protIdx];
  let eD = 0;
  for (const d of centre.depIdxs) {
    s[centre.position] = d;
    eD += pottsCandidateEnergiesAt(tables, s, centre.position, row)[d];
  }
  s[centre.position] = seq[centre.position];
  return eP - eD / centre.depIdxs.length;
}

/**
 * `sum over centres (e_P - mean_d e_D)` with every centre held protonated, so
 * each gap is measured in the presence of the others.
 */
export function selectiveGapSum(tables, seq, centres, scratch) {
  if (centres.length === 0) return 0;
  const s = scratch?.seq ?? Int32Array.from(seq);
  if (s !== seq) s.set(seq);
  for (const c of centres) s[c.position] = c.protIdx;
  let total = 0;
  const inner = { seq: scratch?.seq2 ?? new Int32Array(s.length), row: scratch?.row };
  for (const c of centres) total += selectiveGapOf(tables, s, c, inner);
  return total;
}

/** Per-centre gaps plus their sum, for reporting. */
export function selectiveReport(tables, seq, centres) {
  const s = Int32Array.from(seq);
  for (const c of centres) s[c.position] = c.protIdx;
  const per = centres.map((c) => ({
    position: c.position,
    resSeq: c.resSeq,
    protonationType: c.protonationType,
    gap: selectiveGapOf(tables, s, c),
  }));
  return { centres: per, gapSum: per.reduce((a, b) => a + b.gap, 0) };
}

/* ========================================================================== *
 * 8. pH-selective block-descent optimizer
 * ========================================================================== */

/**
 * `J_block` decomposition of the stability term: a unary table plus the
 * within-block pairwise matrices, so every one of the `V ** B` joint
 * assignments of a block can be scored by a sum.
 *
 * `unary[bi, a]` is block position `bi`'s self field plus all of its couplings
 * to the *fixed* context; `edges` holds `(bi, bj, M)` with `M[a, b]` the
 * directed table for the within-block edge, always emitted with `bi < bj`.
 * The orientation guarantee matters: the pair tables are asymmetric, and the
 * broadcast a consumer uses to add `M` into the `[V] ** B` tensor cannot tell
 * the two orders apart, so an edge with `bi > bj` would silently be added
 * transposed.
 */
export function blockStabilityPotentials(tables, seq, block, selfScale = 1, pairScale = 1) {
  const { J, edgeIdx, K, V, incoming } = tables;
  const VV = V * V;
  const B = block.length;
  const posOf = new Map(block.map((p, bi) => [p, bi]));
  const unary = new Float64Array(B * V);
  const edges = [];
  for (let bi = 0; bi < B; bi += 1) {
    const p = block[bi];
    const self = p * K * VV;
    for (let a = 0; a < V; a += 1) unary[bi * V + a] = selfScale * J[self + a * V + a];
    for (let k = 1; k < K; k += 1) {
      const e = p * K + k;
      const j = edgeIdx[e];
      if (j === p) continue;
      const bj = posOf.get(j);
      if (bj === undefined) {
        const base = e * VV + seq[j];
        for (let a = 0; a < V; a += 1) unary[bi * V + a] += pairScale * J[base + a * V];
      } else if (bi < bj) {
        const M = new Float64Array(VV);
        for (let a = 0; a < VV; a += 1) M[a] = pairScale * J[e * VV + a];
        edges.push({ bi, bj, M });
      } else {
        const M = new Float64Array(VV); // transposed so the low axis is still bi
        for (let a = 0; a < V; a += 1) {
          for (let b = 0; b < V; b += 1) M[b * V + a] = pairScale * J[e * VV + a * V + b];
        }
        edges.push({ bi: bj, bj: bi, M });
      }
    }
    for (let t = incoming.start[p]; t < incoming.start[p + 1]; t += 1) {
      const m = incoming.src[t];
      if (posOf.has(m)) continue; // within-block incoming = the other end's outgoing
      const base = (m * K + incoming.slot[t]) * VV + seq[m] * V;
      for (let a = 0; a < V; a += 1) unary[bi * V + a] += pairScale * J[base + a];
    }
  }
  return { unary, edges, B };
}

/** `p` plus its `blockSize - 1` closest coupled partners drawn from `designable`. */
export function blockPartners(tables, p, designableSet, blockSize) {
  const block = [p];
  if (blockSize <= 1) return block;
  const { edgeIdx, K } = tables;
  for (let k = 1; k < K; k += 1) { // E_idx order is closest-first
    const j = edgeIdx[p * K + k];
    if (j !== p && designableSet.has(j) && !block.includes(j)) {
      block.push(j);
      if (block.length === blockSize) break;
    }
  }
  return block;
}

/** Std of the finite, allowed entries of a `[N, V]` delta table, floored at 1e-6. */
function zscaleOf(deltas, rows, V, validMask) {
  let n = 0;
  let mean = 0;
  let m2 = 0;
  for (let r = 0; r < rows; r += 1) {
    for (let a = 0; a < V; a += 1) {
      if (!validMask[a]) continue;
      const v = deltas[r * V + a];
      if (!Number.isFinite(v)) continue;
      n += 1;
      const d = v - mean;
      mean += d / n;
      m2 += d * (v - mean);
    }
  }
  if (n === 0) return 1.0;
  return Math.max(Math.sqrt(m2 / n), 1e-6);
}

/**
 * Single-mutation delta tables over the design positions, used to freeze the
 * objective's z-scales: `dH[n, a]` / `dSel[n, a]` are the change in stability /
 * selectivity when `designable[n]` is set to token `a`, every centre pinned
 * protonated and the rest of the sequence fixed.
 */
function blockMutationTables(tables, seq, designable, centres, validMask) {
  const { V } = tables;
  const N = designable.length;
  const dH = new Float64Array(N * V).fill(Infinity);
  const dSel = new Float64Array(N * V);
  const s = Int32Array.from(seq);
  for (const c of centres) s[c.position] = c.protIdx;
  const sel0 = selectiveGapSum(tables, s, centres);
  const row = new Float64Array(V);
  for (let n = 0; n < N; n += 1) {
    const i = designable[n];
    const cur = s[i];
    pottsCandidateEnergiesAt(tables, s, i, row);
    const base = row[cur];
    for (let a = 0; a < V; a += 1) {
      if (!validMask[a]) continue;
      dH[n * V + a] = row[a] - base;
    }
    for (let a = 0; a < V; a += 1) {
      if (!validMask[a] || a === cur) continue;
      s[i] = a;
      dSel[n * V + a] = selectiveGapSum(tables, s, centres) - sel0;
    }
    s[i] = cur;
  }
  return { dH, dSel, sel0, pinnedSeq: s };
}

/** Variance of `sum_bi vecs[bi]` over the allowed `V ** B` assignments. */
function unaryBlockVariance(vecs, V, validMask) {
  const B = vecs.length;
  const valid = [];
  for (let a = 0; a < V; a += 1) if (validMask[a]) valid.push(a);
  if (valid.length === 0) return null;
  let n = 0;
  let mean = 0;
  let m2 = 0;
  const idx = new Int32Array(B);
  const total = valid.length ** B;
  for (let t = 0; t < total; t += 1) {
    let rem = t;
    let sum = 0;
    for (let bi = B - 1; bi >= 0; bi -= 1) {
      idx[bi] = rem % valid.length;
      rem = (rem - idx[bi]) / valid.length;
      sum += vecs[bi][valid[idx[bi]]];
    }
    n += 1;
    const d = sum - mean;
    mean += d / n;
    m2 += d * (sum - mean);
  }
  return n > 1 ? m2 / n : null;
}

/**
 * Freeze the z-scales from the block joint spread (`zscaleMode: 'block'`) rather
 * than from single mutations: for every design position's block, build the
 * stability tensor over all `V ** B` assignments through the same decomposition
 * the sweep uses and pool its within-block variance. Selectivity is exactly
 * unary, so its block tensor is the broadcast sum of the per-position deltas.
 */
function blockZScales(tables, seq, designable, centres, validMask, blockSize, dSel) {
  const { V } = tables;
  const designableSet = new Set(designable);
  const rowOf = new Map(designable.map((p, n) => [p, n]));
  const stabVars = [];
  const selVars = [];
  const valid = [];
  for (let a = 0; a < V; a += 1) if (validMask[a]) valid.push(a);

  for (const p of designable) {
    const block = blockPartners(tables, p, designableSet, blockSize);
    const B = block.length;
    const { unary, edges } = blockStabilityPotentials(tables, seq, block);
    // Enumerate the allowed joint assignments; `B` is 1..4 in practice.
    let n = 0;
    let mean = 0;
    let m2 = 0;
    const idx = new Int32Array(B);
    const total = valid.length ** B;
    for (let t = 0; t < total; t += 1) {
      let rem = t;
      for (let bi = B - 1; bi >= 0; bi -= 1) {
        idx[bi] = rem % valid.length;
        rem = (rem - idx[bi]) / valid.length;
      }
      let sum = 0;
      for (let bi = 0; bi < B; bi += 1) sum += unary[bi * V + valid[idx[bi]]];
      for (const { bi, bj, M } of edges) sum += M[valid[idx[bi]] * V + valid[idx[bj]]];
      n += 1;
      const d = sum - mean;
      mean += d / n;
      m2 += d * (sum - mean);
    }
    if (n > 1) stabVars.push(m2 / n);

    const rows = block.map((b) => dSel.subarray(rowOf.get(b) * V, rowOf.get(b) * V + V));
    let anyNonZero = false;
    for (const r of rows) {
      let s = 0;
      for (let a = 0; a < V; a += 1) s += Math.abs(r[a]);
      if (s > 0) { anyNonZero = true; break; }
    }
    if (anyNonZero) {
      const v = unaryBlockVariance(rows, V, validMask);
      if (v !== null) selVars.push(v);
    }
  }
  const pool = (vs) => (vs.length ? Math.max(Math.sqrt(vs.reduce((a, b) => a + b, 0) / vs.length), 1e-6) : 1.0);
  return { sdH: pool(stabVars), sdSel: pool(selVars) };
}

/**
 * pH-selective block coordinate descent on the z-scaled combined objective
 *
 *     O(s) = (1 - lambda) * z(E_global) + lambda * z(sum_centres (E_P - E_D))
 *
 * with the centres pinned protonated as fixed context. Each sweep visits every
 * designable position, forms a block from it plus its closest coupled
 * partners, scores all `V ** blockSize` joint assignments exactly, and commits
 * the argmin (`temperature === 0`) or a Boltzmann sample. It stops when a full
 * sweep makes no change, or after `maxRounds`.
 *
 * z-scaling is what makes `lambda` a true relative weight: the two terms live
 * on very different scales, so each is divided by the spread of its own
 * single-mutation (or block-joint) deltas, frozen once before the first sweep.
 *
 * Only stability has within-block pairwise terms — selectivity is exactly unary
 * — so `blockSize` changes the answer solely through stability coupling.
 */
export function blockDescent(tables, options) {
  const {
    seq,
    centres = [],
    designable,
    lambda = 1.0,
    blockSize = 2,
    zscaleMode = 'block',
    temperature = 0,
    maxRounds = 10,
    validMask = validTokenMask(),
    sweepOrder = 'position',
    selfWeight = 1.0,
    adjacentRepeatWeight = 0,
    repetitiveWindowWeight = 0,
    repetitiveWindowRadius = 2,
    repetitiveWindowParents = [],
    resSeq = null,
    rng = makeRng(0),
    onStep = null,
  } = options;
  const { V } = tables;
  const S = Int32Array.from(seq);
  for (const c of centres) S[c.position] = c.protIdx;
  let order = [...designable].map(Number).sort((a, b) => a - b);
  if (order.length === 0) return { tokens: S, rounds: 0, converged: true, nChanges: 0 };
  if (sweepOrder === 'knn') order = knnRank(tables, centres, order);
  else if (sweepOrder === 'energy') order = centreEnergyRank(tables, S, centres, order);

  // --- freeze the z-scales ------------------------------------------------
  const { dH, dSel } = blockMutationTables(tables, S, order, centres, validMask);
  let sdH;
  let sdSel;
  if (zscaleMode === 'block') {
    ({ sdH, sdSel } = blockZScales(tables, S, order, centres, validMask, Math.max(1, blockSize), dSel));
  } else {
    sdH = zscaleOf(dH, order.length, V, validMask);
    const selRows = [];
    for (let n = 0; n < order.length; n += 1) {
      let s = 0;
      for (let a = 0; a < V; a += 1) s += Math.abs(dSel[n * V + a]);
      if (s > 0) selRows.push(n);
    }
    if (selRows.length) {
      const packed = new Float64Array(selRows.length * V);
      selRows.forEach((n, r) => packed.set(dSel.subarray(n * V, n * V + V), r * V));
      sdSel = zscaleOf(packed, selRows.length, V, validMask);
    } else {
      sdSel = zscaleOf(dSel, order.length, V, validMask);
    }
  }
  const wH = (1 - lambda) / sdH;
  const wSel = lambda / sdSel;

  // --- optional sequence-local density penalties --------------------------
  const wantRepeat = adjacentRepeatWeight !== 0;
  const repParents = [...repetitiveWindowParents];
  const wantWindow = repetitiveWindowWeight !== 0 && repParents.length > 0;
  const repMask = wantWindow ? parentTokenMask(repParents) : null;
  let sameCanon = null;
  if (wantRepeat) {
    sameCanon = new Float64Array(V * V);
    for (let a = 0; a < V; a += 1) {
      for (let b = 0; b < V; b += 1) {
        sameCanon[a * V + b] = CANONICAL_MAP[a] === CANONICAL_MAP[b] ? 1 : 0;
      }
    }
  }
  const resIds = resSeq ?? Int32Array.from({ length: tables.L }, (_, i) => i);
  const posOfResId = new Map();
  for (let i = 0; i < tables.L; i += 1) posOfResId.set(resIds[i], i);

  const designableSet = new Set(order);
  const selScratch = { seq: new Int32Array(tables.L), seq2: new Int32Array(tables.L), row: new Float64Array(V) };

  /** Selective (and density) unary contribution for one block position. */
  const combinedUnary = (p, blockSet) => {
    const cur = S[p];
    const sel0 = selectiveGapSum(tables, S, centres, selScratch);
    const u = new Float64Array(V);
    if (wSel !== 0) {
      for (let a = 0; a < V; a += 1) {
        if (!validMask[a] || a === cur) continue;
        S[p] = a;
        u[a] = wSel * (selectiveGapSum(tables, S, centres, selScratch) - sel0);
      }
      S[p] = cur;
    }
    if (wantWindow) {
      const ri = resIds[p];
      let nCtx = 0;
      for (let d = -repetitiveWindowRadius; d <= repetitiveWindowRadius; d += 1) {
        if (d === 0) continue;
        const q = posOfResId.get(ri + d);
        if (q !== undefined && !blockSet.has(q) && repMask[S[q]]) nCtx += 1;
      }
      if (nCtx) {
        for (let a = 0; a < V; a += 1) u[a] += repetitiveWindowWeight * nCtx * repMask[a];
      }
    }
    return u;
  };

  let changedAny = true;
  let rounds = 0;
  let nChanges = 0;
  const selfScale = selfWeight;
  while (changedAny && rounds < maxRounds) {
    changedAny = false;
    rounds += 1;
    for (const p of order) {
      const block = blockPartners(tables, p, designableSet, Math.max(1, blockSize));
      const B = block.length;
      const blockSet = new Set(block);
      const { unary, edges } = blockStabilityPotentials(tables, S, block, selfScale, 1);
      const cu = new Float64Array(B * V);
      for (let bi = 0; bi < B; bi += 1) {
        const add = combinedUnary(block[bi], blockSet);
        for (let a = 0; a < V; a += 1) cu[bi * V + a] = wH * unary[bi * V + a] + add[a];
      }
      // Score all V**B joint assignments.
      const total = V ** B;
      const Jt = new Float64Array(total);
      const strides = new Int32Array(B);
      strides[B - 1] = 1;
      for (let bi = B - 2; bi >= 0; bi -= 1) strides[bi] = strides[bi + 1] * V;
      const idx = new Int32Array(B);
      for (let t = 0; t < total; t += 1) {
        let rem = t;
        for (let bi = 0; bi < B; bi += 1) {
          idx[bi] = Math.floor(rem / strides[bi]);
          rem -= idx[bi] * strides[bi];
        }
        let sum = 0;
        let invalid = false;
        for (let bi = 0; bi < B; bi += 1) {
          if (!validMask[idx[bi]]) { invalid = true; break; }
          sum += cu[bi * V + idx[bi]];
        }
        if (invalid) { Jt[t] = Infinity; continue; }
        for (const { bi, bj, M } of edges) sum += wH * M[idx[bi] * V + idx[bj]];
        if (wantRepeat) {
          for (let bi = 0; bi < B; bi += 1) {
            for (let bj = bi + 1; bj < B; bj += 1) {
              if (Math.abs(resIds[block[bi]] - resIds[block[bj]]) === 1) {
                sum += adjacentRepeatWeight * sameCanon[idx[bi] * V + idx[bj]];
              }
            }
            for (const d of [-1, 1]) {
              const q = posOfResId.get(resIds[block[bi]] + d);
              if (q !== undefined && !blockSet.has(q)) {
                sum += adjacentRepeatWeight * sameCanon[idx[bi] * V + S[q]];
              }
            }
          }
        }
        if (wantWindow) {
          for (let bi = 0; bi < B; bi += 1) {
            for (let bj = bi + 1; bj < B; bj += 1) {
              if (Math.abs(resIds[block[bi]] - resIds[block[bj]]) <= repetitiveWindowRadius) {
                sum += repetitiveWindowWeight * repMask[idx[bi]] * repMask[idx[bj]];
              }
            }
          }
        }
        Jt[t] = sum;
      }
      let choice;
      if (temperature <= 0) {
        let best = Infinity;
        choice = 0;
        for (let t = 0; t < total; t += 1) {
          if (Jt[t] < best) { best = Jt[t]; choice = t; }
        }
      } else {
        choice = sampleCategorical(boltzmann(Jt, temperature), rng);
      }
      let moved = false;
      let rem = choice;
      for (let bi = 0; bi < B; bi += 1) {
        const a = Math.floor(rem / strides[bi]);
        rem -= a * strides[bi];
        if (a !== S[block[bi]]) { S[block[bi]] = a; moved = true; }
      }
      if (moved) {
        changedAny = true;
        nChanges += 1;
        if (onStep) {
          onStep({
            round: rounds,
            position: p,
            block,
            pottsEnergy: pottsEnergy(tables.J, tables.edgeIdx, tables.L, tables.K, V, S),
            selectiveEnergy: selectiveGapSum(tables, S, centres),
            tokens: Int32Array.from(S),
          });
        }
      }
    }
  }
  return {
    tokens: S,
    rounds,
    converged: !changedAny,
    nChanges,
    zscales: { sdH, sdSel },
    weights: { wH, wSel },
    pottsEnergy: pottsEnergy(tables.J, tables.edgeIdx, tables.L, tables.K, V, S),
    selectiveEnergy: selectiveGapSum(tables, S, centres),
  };
}

/** Designable positions ordered by their best rank in any centre's kNN list. */
export function knnRank(tables, centres, designable) {
  const { edgeIdx, K } = tables;
  const best = new Map(designable.map((j) => [j, 1e6]));
  for (const c of centres) {
    for (let r = 1; r < K; r += 1) {
      const j = edgeIdx[c.position * K + r];
      if (best.has(j)) best.set(j, Math.min(best.get(j), r));
    }
  }
  return [...best.keys()].sort((a, b) => (best.get(a) - best.get(b)) || (a - b));
}

/**
 * Designable positions ordered by `|contribution to the centres' selective gap|`,
 * descending. A designable `j` enters a centre `c`'s conditional energy through
 * exactly two additive pair terms (`c -> j` and `j -> c`), so the contribution
 * is exact and needs no re-scoring of the whole sequence.
 */
export function centreEnergyRank(tables, seq, centres, designable) {
  const { J, edgeIdx, K, V, incoming } = tables;
  const VV = V * V;
  const score = new Map(designable.map((j) => [j, 0]));
  for (const c of centres) {
    const p = c.position;
    const slotOf = new Map();
    for (let k = 1; k < K; k += 1) slotOf.set(edgeIdx[p * K + k], k);
    const incOf = new Map();
    for (let t = incoming.start[p]; t < incoming.start[p + 1]; t += 1) {
      incOf.set(incoming.src[t], incoming.slot[t]);
    }
    for (const j of score.keys()) {
      const sj = seq[j];
      let eP = 0;
      let eD = 0;
      if (slotOf.has(j)) {
        const base = (p * K + slotOf.get(j)) * VV;
        eP += J[base + c.protIdx * V + sj];
        let acc = 0;
        for (const d of c.depIdxs) acc += J[base + d * V + sj];
        eD += acc / c.depIdxs.length;
      }
      if (incOf.has(j)) {
        const base = (j * K + incOf.get(j)) * VV + sj * V;
        eP += J[base + c.protIdx];
        let acc = 0;
        for (const d of c.depIdxs) acc += J[base + d];
        eD += acc / c.depIdxs.length;
      }
      score.set(j, score.get(j) + Math.abs(eP - eD));
    }
  }
  return [...score.keys()].sort((a, b) => (score.get(b) - score.get(a)) || (a - b));
}

/**
 * Greedy single-position coordinate descent on the same objective
 * (`blockSize = 1` block descent, kept as a named entry point because it is the
 * cheap default for long chains).
 */
export function greedyCoordinateDescent(tables, options) {
  return blockDescent(tables, { ...options, blockSize: 1 });
}

/**
 * The designable neighbourhood of a set of centres: their coupled kNN in both
 * graph directions, capped per centre and in total.
 */
export function centreNeighbourhood(tables, centres, { perCentre = 8, max = 0 } = {}) {
  const { edgeIdx, K, L } = tables;
  const picked = new Set();
  for (const c of centres) {
    const p = c.position;
    let n = 0;
    for (let k = 1; k < K && n < perCentre; k += 1) { picked.add(edgeIdx[p * K + k]); n += 1; }
    n = 0;
    for (let m = 0; m < L && n < perCentre; m += 1) {
      for (let k = 1; k < K; k += 1) {
        if (edgeIdx[m * K + k] === p) { picked.add(m); n += 1; break; }
      }
    }
  }
  for (const c of centres) picked.delete(c.position);
  let out = [...picked].sort((a, b) => a - b);
  if (max > 0 && out.length > max) out = knnRank(tables, centres, out).slice(0, max).sort((a, b) => a - b);
  return out;
}

/* ========================================================================== *
 * 9. Encoder / decoder reference implementations (shared by both backends)
 * ========================================================================== */

/** Gather `[L, H]` node states at the kNN neighbours -> `[L, K, H]`. */
function gatherNodes(nodes, edgeIdx, L, K, H, out) {
  const dst = out ?? new Float32Array(L * K * H);
  for (let i = 0; i < L; i += 1) {
    for (let k = 0; k < K; k += 1) {
      const j = edgeIdx[i * K + k];
      dst.set(nodes.subarray(j * H, j * H + H), (i * K + k) * H);
    }
  }
  return dst;
}

/** Build `[h_V_i, h_E_ik, h_V_j]` `[L, K, 3H]`, the EncLayer message input. */
function buildEncoderMessageInput(hV, hE, edgeIdx, L, K, H, out) {
  const dst = out ?? new Float32Array(L * K * 3 * H);
  for (let i = 0; i < L; i += 1) {
    const vi = hV.subarray(i * H, i * H + H);
    for (let k = 0; k < K; k += 1) {
      const e = i * K + k;
      const base = e * 3 * H;
      dst.set(vi, base);
      dst.set(hE.subarray(e * H, e * H + H), base + H);
      const j = edgeIdx[e];
      dst.set(hV.subarray(j * H, j * H + H), base + 2 * H);
    }
  }
  return dst;
}

/** A `PositionWiseFeedForward`: `W_out(gelu(W_in(x)))`. */
function feedForward(x, rows, H, w, prefix, ffnDim, matmul) {
  const mid = matmul(x, rows, H, w.get(`${prefix}.W_in.weight`).data, ffnDim,
    w.get(`${prefix}.W_in.bias`).data);
  geluInPlace(mid);
  return matmul(mid, rows, ffnDim, w.get(`${prefix}.W_out.weight`).data, H,
    w.get(`${prefix}.W_out.bias`).data);
}

/**
 * One `EncLayer`: a node update from the scaled sum of neighbour messages, a
 * position-wise feed-forward, then an edge update — each with a residual and a
 * LayerNorm, exactly as in `message_passing.EncLayer`.
 */
function encoderLayer(hV, hE, ctx, prefix) {
  const { edgeIdx, L, K, H, maskE, weights: w, ffnDim, matmul, messageScale } = ctx;
  const rowsE = L * K;
  const g = (n) => w.get(`${prefix}.${n}`).data;

  let hEV = buildEncoderMessageInput(hV, hE, edgeIdx, L, K, H);
  let msg = matmul(hEV, rowsE, 3 * H, g('W1.weight'), H, g('W1.bias'));
  geluInPlace(msg);
  msg = matmul(msg, rowsE, H, g('W2.weight'), H, g('W2.bias'));
  geluInPlace(msg);
  msg = matmul(msg, rowsE, H, g('W3.weight'), H, g('W3.bias'));

  const dh = new Float32Array(L * H);
  for (let i = 0; i < L; i += 1) {
    for (let k = 0; k < K; k += 1) {
      const e = i * K + k;
      if (!maskE[e]) continue;
      for (let h = 0; h < H; h += 1) dh[i * H + h] += msg[e * H + h];
    }
    for (let h = 0; h < H; h += 1) dh[i * H + h] /= messageScale;
  }
  const hV1 = new Float32Array(L * H);
  for (let i = 0; i < L * H; i += 1) hV1[i] = hV[i] + dh[i];
  layerNormInPlace(hV1, L, H, g('norm1.weight'), g('norm1.bias'));

  const ff = feedForward(hV1, L, H, w, `${prefix}.dense`, ffnDim, matmul);
  for (let i = 0; i < L * H; i += 1) hV1[i] += ff[i];
  layerNormInPlace(hV1, L, H, g('norm2.weight'), g('norm2.bias'));
  for (let i = 0; i < L; i += 1) {
    if (ctx.residueMask[i]) continue;
    hV1.fill(0, i * H, i * H + H);
  }

  hEV = buildEncoderMessageInput(hV1, hE, edgeIdx, L, K, H);
  let emsg = matmul(hEV, rowsE, 3 * H, g('W11.weight'), H, g('W11.bias'));
  geluInPlace(emsg);
  emsg = matmul(emsg, rowsE, H, g('W12.weight'), H, g('W12.bias'));
  geluInPlace(emsg);
  emsg = matmul(emsg, rowsE, H, g('W13.weight'), H, g('W13.bias'));
  const hE1 = new Float32Array(rowsE * H);
  for (let i = 0; i < rowsE * H; i += 1) hE1[i] = hE[i] + emsg[i];
  layerNormInPlace(hE1, rowsE, H, g('norm3.weight'), g('norm3.bias'));

  return { hV: hV1, hE: hE1 };
}

/**
 * Embed the raw edge features and run the message-passing encoder.
 *
 * `matmul` is injected so the WebGPU backend can reuse the surrounding
 * bookkeeping while replacing the dense kernels.
 */
export function runEncoder(structure, features, weights, arch, matmul = linearForward) {
  const { L, residueMask } = structure;
  const { edgeIdx, K, Eraw, maskE } = features;
  const H = arch.hiddenDim;
  const rowsE = L * K;

  const E = matmul(
    Eraw, rowsE, arch.numEdgeInputFeatures,
    weights.get('graph_featurization_module.edge_embedding.weight').data, arch.numEdgeFeatures,
    null,
  );
  layerNormInPlace(
    E, rowsE, arch.numEdgeFeatures,
    weights.get('graph_featurization_module.edge_norm.weight').data,
    weights.get('graph_featurization_module.edge_norm.bias').data,
  );
  let hE = matmul(E, rowsE, arch.numEdgeFeatures, weights.get('W_e.weight').data, H,
    weights.get('W_e.bias').data);
  let hV = new Float32Array(L * H);

  const ctx = {
    edgeIdx, L, K, H, maskE, residueMask, weights, matmul,
    ffnDim: arch.ffnDim, messageScale: arch.messageScale,
  };
  for (let l = 0; l < arch.numEncoderLayers; l += 1) {
    ({ hV, hE } = encoderLayer(hV, hE, ctx, `encoder_layers.${l}`));
  }
  return { E, hV, hE };
}

/**
 * The Potts head: `etab_out` over the encoder edge states, then the mask /
 * diagonal / reciprocal-merge finishing pass. Returns the finalized
 * `J [L, K, V, V]` and the field `h [L, V]`.
 */
export function runPottsHead(structure, features, encoder, weights, arch, matmul = linearForward) {
  const { L } = structure;
  const { edgeIdx, K, maskE } = features;
  const V = arch.pottsVocabSize;
  const H = arch.hiddenDim;
  let head = encoder.hE;
  if (arch.etabSource === 'node_edge_node') {
    head = buildEncoderMessageInput(encoder.hV, encoder.hE, edgeIdx, L, K, H);
  }
  const inDim = arch.etabSource === 'node_edge_node' ? 3 * H : H;
  const J = matmul(head, L * K, inDim, weights.get('etab_out.weight').data, V * V,
    weights.get('etab_out.bias').data);
  const reverse = buildReverseIndex(edgeIdx, L, K);
  finalizePottsTables(J, L, K, V, maskE, edgeIdx, reverse);
  return { J, h: extractField(J, L, K, V), reverse };
}

/**
 * Causal / anti-causal decoder masks `[L, K]`.
 *
 * The all-by-all pattern is permuted by the decoding order, so only
 * `auto_regressive` actually depends on `decodingOrder`; the other three
 * patterns are permutation-invariant and therefore deterministic.
 */
export function buildCausalMasks(structure, features, causalityPattern, decodingOrder = null) {
  const { L, residueMask } = structure;
  const { edgeIdx, K } = features;
  const causal = new Float32Array(L * K);
  const anti = new Float32Array(L * K);
  let rank = null;
  if (causalityPattern === 'auto_regressive') {
    const order = decodingOrder ?? Int32Array.from({ length: L }, (_, i) => i);
    rank = new Int32Array(L);
    for (let step = 0; step < L; step += 1) rank[order[step]] = step;
  }
  for (let i = 0; i < L; i += 1) {
    for (let k = 0; k < K; k += 1) {
      const e = i * K + k;
      const j = edgeIdx[e];
      let m;
      switch (causalityPattern) {
        case 'unconditional': m = 0; break;
        case 'conditional': m = 1; break;
        case 'conditional_minus_self': m = i === j ? 0 : 1; break;
        case 'auto_regressive': m = rank[j] < rank[i] ? 1 : 0; break;
        default: throw new Error(`unknown causality pattern ${causalityPattern}`);
      }
      const rm = residueMask[i] ? 1 : 0;
      causal[e] = m * rm;
      anti[e] = (1 - m) * rm;
    }
  }
  return { causal, anti };
}

/**
 * Teacher-forced decoder: logits and log-probabilities for every position at
 * once. `h_ESV = causal * [h_E, h_S_j, h_Vdec_j] + anti * [h_E, 0, h_V_j]`, so a
 * position only ever sees sequence information from neighbours the causality
 * pattern lets it see.
 */
export function runDecoderTeacherForcing(structure, features, encoder, weights, arch, options = {}) {
  const {
    causalityPattern = 'conditional_minus_self',
    initializeWithGroundTruth = true,
    decodingOrder = null,
    matmul = linearForward,
    seq = structure.tokens,
  } = options;
  const { L, residueMask } = structure;
  const { edgeIdx, K, maskE } = features;
  const H = arch.hiddenDim;
  const Vv = arch.vocabSize;
  const rowsE = L * K;
  const { causal, anti } = buildCausalMasks(structure, features, causalityPattern, decodingOrder);

  const Ws = weights.get('W_s.weight').data; // [V, H] embedding
  const hS = new Float32Array(L * H);
  if (initializeWithGroundTruth) {
    for (let i = 0; i < L; i += 1) hS.set(Ws.subarray(seq[i] * H, seq[i] * H + H), i * H);
  }

  const hVj = gatherNodes(encoder.hV, edgeIdx, L, K, H);
  const hSj = gatherNodes(hS, edgeIdx, L, K, H);

  let hVdec = Float32Array.from(encoder.hV);
  const in4 = new Float32Array(rowsE * 4 * H);
  for (let l = 0; l < arch.numDecoderLayers; l += 1) {
    const hVdecJ = gatherNodes(hVdec, edgeIdx, L, K, H);
    for (let i = 0; i < L; i += 1) {
      const vi = hVdec.subarray(i * H, i * H + H);
      for (let k = 0; k < K; k += 1) {
        const e = i * K + k;
        const base = e * 4 * H;
        const c = causal[e];
        const a = anti[e];
        in4.set(vi, base);
        in4.set(encoder.hE.subarray(e * H, e * H + H), base + H);
        for (let h = 0; h < H; h += 1) {
          in4[base + 2 * H + h] = c * hSj[e * H + h];
          in4[base + 3 * H + h] = c * hVdecJ[e * H + h] + a * hVj[e * H + h];
        }
      }
    }
    const g = (n) => weights.get(`decoder_layers.${l}.${n}`).data;
    let msg = matmul(in4, rowsE, 4 * H, g('W1.weight'), H, g('W1.bias'));
    geluInPlace(msg);
    msg = matmul(msg, rowsE, H, g('W2.weight'), H, g('W2.bias'));
    geluInPlace(msg);
    msg = matmul(msg, rowsE, H, g('W3.weight'), H, g('W3.bias'));
    const dh = new Float32Array(L * H);
    for (let i = 0; i < L; i += 1) {
      for (let k = 0; k < K; k += 1) {
        const e = i * K + k;
        if (!maskE[e]) continue;
        for (let h = 0; h < H; h += 1) dh[i * H + h] += msg[e * H + h];
      }
      for (let h = 0; h < H; h += 1) dh[i * H + h] /= arch.messageScale;
    }
    const next = new Float32Array(L * H);
    for (let i = 0; i < L * H; i += 1) next[i] = hVdec[i] + dh[i];
    layerNormInPlace(next, L, H, g('norm1.weight'), g('norm1.bias'));
    const ff = feedForward(next, L, H, weights, `decoder_layers.${l}.dense`, arch.ffnDim, matmul);
    for (let i = 0; i < L * H; i += 1) next[i] += ff[i];
    layerNormInPlace(next, L, H, g('norm2.weight'), g('norm2.bias'));
    for (let i = 0; i < L; i += 1) {
      if (!residueMask[i]) next.fill(0, i * H, i * H + H);
    }
    hVdec = next;
  }
  const logits = matmul(hVdec, L, H, weights.get('W_out.weight').data, Vv,
    weights.get('W_out.bias').data);
  const logProbs = Float32Array.from(logits);
  logSoftmaxInPlace(logProbs, L, Vv);
  return { logits, logProbs, hVdec };
}

/**
 * Autoregressive decoding: sample one position at a time in `decodingOrder`,
 * feeding each committed token's embedding back in. Designed positions are
 * sampled with `UNK` excluded and renormalised; fixed positions keep their
 * input token. `rng` makes a run reproducible.
 *
 * Implemented on top of the teacher-forced pass with an `auto_regressive`
 * causal mask recomputed per step, which is `O(L)` passes — correct and
 * straightforward, and only used when a caller explicitly asks to sample.
 */
export function runDecoderAutoregressive(structure, features, encoder, weights, arch, options = {}) {
  const {
    decodingOrder = null, rng = makeRng(0), temperature = 1.0,
    matmul = linearForward, designedResidueMask = structure.designedResidueMask,
  } = options;
  const { L, residueMask } = structure;
  const Vv = arch.vocabSize;
  const unk = TOKEN_TO_IDX[UNKNOWN_TOKEN];

  // The reference sorts |noise| scaled by the decode-last mask, so invalid and
  // fixed positions are decoded first; mirror that with the injected rng.
  let order = decodingOrder;
  if (!order) {
    const key = new Float64Array(L);
    for (let i = 0; i < L; i += 1) {
      const decodeLast = residueMask[i] && designedResidueMask[i] ? 1 : 0;
      key[i] = (decodeLast + 1e-4) * Math.abs(gaussian(rng));
    }
    order = Int32Array.from({ length: L }, (_, i) => i).sort((a, b) => key[a] - key[b]);
  }

  const seq = Int32Array.from(structure.tokens);
  const sampled = new Int32Array(L);
  for (let i = 0; i < L; i += 1) sampled[i] = residueMask[i] && designedResidueMask[i] ? unk : seq[i];
  const logProbs = new Float32Array(L * Vv);
  const argmax = new Int32Array(L);

  for (let step = 0; step < L; step += 1) {
    const i = order[step];
    const out = runDecoderTeacherForcing(structure, features, encoder, weights, arch, {
      causalityPattern: 'auto_regressive',
      initializeWithGroundTruth: true,
      decodingOrder: order,
      matmul,
      seq: sampled,
    });
    const row = out.logits.subarray(i * Vv, i * Vv + Vv);
    const scaled = Float64Array.from(row, (v) => v / temperature);
    let max = -Infinity;
    for (const v of scaled) if (v > max) max = v;
    const probs = new Float64Array(Vv);
    let sum = 0;
    for (let a = 0; a < Vv; a += 1) { probs[a] = Math.exp(scaled[a] - max); sum += probs[a]; }
    for (let a = 0; a < Vv; a += 1) probs[a] /= sum;
    logProbs.set(out.logProbs.subarray(i * Vv, i * Vv + Vv), i * Vv);
    probs[unk] = 0;
    let s2 = 0;
    for (const p of probs) s2 += p;
    for (let a = 0; a < Vv; a += 1) probs[a] /= s2;
    let best = 0;
    for (let a = 1; a < Vv; a += 1) if (probs[a] > probs[best]) best = a;
    const pick = sampleCategorical(probs, rng);
    const designed = residueMask[i] && designedResidueMask[i];
    sampled[i] = designed ? pick : seq[i];
    argmax[i] = designed ? best : seq[i];
  }
  return { sampled, argmax, logProbs, decodingOrder: order };
}

/** Box-Muller standard normal from a uniform rng. */
function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/* ========================================================================== *
 * 10. Architecture description
 * ========================================================================== */

/** Normalise a `manifest.json` `arch` block into the camelCase form used here. */
export function archFromManifest(manifest) {
  const a = manifest.arch;
  return {
    hiddenDim: a.hidden_dim,
    vocabSize: a.vocab_size,
    pottsVocabSize: a.potts_vocab_size,
    numEncoderLayers: a.num_encoder_layers,
    numDecoderLayers: a.num_decoder_layers,
    numNeighbors: a.num_neighbors,
    numEdgeInputFeatures: a.num_edge_input_features,
    numEdgeFeatures: a.hidden_dim,
    ffnDim: a.ffn_dim,
    messageScale: a.message_scale,
    etabSource: a.etab_source,
    fieldSource: a.field_source,
    hasDecoder: a.has_decoder,
  };
}
