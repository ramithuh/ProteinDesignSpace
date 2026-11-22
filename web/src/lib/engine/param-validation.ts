// Shared parameter validation, numeric sanitization, and scientific format checks
// for NodeInspector (live feedback) and validateGraph (pre-run checks).

import type { WorkflowNode } from './types';
import { isValidPdbId, normalizePdbId } from '$lib/protein-ops/rcsb';
import { parseStructureText } from '$lib/protein-ops/pdb';
import {
  ADAPTER_REGISTRY,
  LOCALFOLD_MODEL_OPTIONS,
  MPNN_CHECKPOINTS_BY_FAMILY,
  RFD4_BENCHMARK_OPTIONS,
} from './registry';
import { validateParamsWithSchema } from './param-schema';
import { esmfold2ParamIssue } from './localfold-esmfold2';

const VALID_PDB_INPUT_MODES = new Set(['demo-bundle', 'pdb-id', 'custom']);
const SHA256_HEX_RE = /^[0-9a-f]{64}$/i;

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** Synchronous, browser- & Node-safe SHA-256 hex digest of UTF-8 text (FIPS 180-4). */
export function computeUploadDigest(text: string): string {
  const bytes = new TextEncoder().encode(text);
  const bitLenHi = Math.floor((bytes.length * 8) / 0x100000000);
  const bitLenLo = (bytes.length * 8) >>> 0;
  const paddedLen = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLen);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLen - 8, bitLenHi, false);
  view.setUint32(paddedLen - 4, bitLenLo, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i++) {
      w[i] = view.getUint32(offset + i * 4, false);
    }
    for (let i = 16; i < 64; i++) {
      const s0 = ((w[i - 15] >>> 7) | (w[i - 15] << 25)) ^ ((w[i - 15] >>> 18) | (w[i - 15] << 14)) ^ (w[i - 15] >>> 3);
      const s1 = ((w[i - 2] >>> 17) | (w[i - 2] << 15)) ^ ((w[i - 2] >>> 19) | (w[i - 2] << 13)) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3, h4, h5, h6, h7]
    .map((n) => n.toString(16).padStart(8, '0'))
    .join('');
}

export function isValidSha256Hex(raw: string): boolean {
  return SHA256_HEX_RE.test(raw.trim());
}

const VALID_RFD4_MODES = new Set([
  'unconditional',
  'binder',
  'motif',
  'conditioned',
  'preset',
  'partial',
  'inverse-folding',
]);
const VALID_RFD4_PRESET_IDS = new Set([
  ...RFD4_BENCHMARK_OPTIONS.map((b) => b.id),
  'yaml_5o45_binder',
]);
const VALID_RFD4_RECYCLE_SCHEDULES = new Set(['none', 'low', 'medium', 'high']);
const VALID_GENIE3_TASK_MODES = new Set(['unconditional', 'binder', 'motif']);
const VALID_MSA_MODES = new Set(['single', 'mmseqs2', 'target-only', 'a3m']);
const VALID_TOPK_MODES = new Set(['per-backbone', 'global']);
const VALID_TOPK_METRICS = new Set(['score', 'globalScore']);
const VALID_TOPK_DIRECTIONS = new Set(['min', 'max']);
const VALID_MPNN_CATALOG: Record<string, ReadonlySet<string>> = Object.fromEntries(
  Object.entries(MPNN_CHECKPOINTS_BY_FAMILY).map(([fam, opts]) => [
    fam,
    new Set(opts.map((o) => o.value)),
  ]),
);

export interface NumericFieldSpec {
  min: number;
  max: number;
  step?: number;
  integer?: boolean;
  defaultValue: number;
  unit?: string;
  warnBelow?: {
    threshold: number;
    message: string;
  };
}

export interface ValidationFeedback {
  status: 'ok' | 'warn' | 'error';
  message?: string;
  value?: number;
}

export const NUMERIC_SPECS: Record<string, Record<string, NumericFieldSpec>> = {
  'rfd4-proteina': {
    length: { min: 30, max: 256, step: 1, integer: true, defaultValue: 70, unit: 'aa' },
    numDesigns: { min: 1, max: 16, step: 1, integer: true, defaultValue: 1 },
    partialFraction: { min: 0.05, max: 0.95, step: 0.05, integer: false, defaultValue: 0.5 },
    nsteps: {
      min: 10,
      max: 120,
      step: 1,
      integer: true,
      defaultValue: 60,
      warnBelow: {
        threshold: 58,
        message: 'Fewer than 58 steps may reduce structure quality in RFD4-Proteina; set Recycle to None if you need faster sampling.',
      },
    },
    seed: { min: 0, max: 2_147_483_647, step: 1, integer: true, defaultValue: 42 },
  },
  genie3: {
    length: { min: 30, max: 256, step: 1, integer: true, defaultValue: 80, unit: 'aa' },
    numDesigns: { min: 1, max: 16, step: 1, integer: true, defaultValue: 1 },
    nSampleStep: { min: 10, max: 250, step: 10, integer: true, defaultValue: 100 },
    directionScale: { min: 0.0, max: 1.5, step: 0.1, integer: false, defaultValue: 0.8 },
    noiseScale: { min: 0, max: 1.5, step: 0.1, integer: false, defaultValue: 1.0 },
    seed: { min: 0, max: 2_147_483_647, step: 1, integer: true, defaultValue: 42 },
  },
  'web-protein-mpnn': {
    numSequences: { min: 1, max: 8, step: 1, integer: true, defaultValue: 2 },
    temperature: { min: 0.01, max: 1.5, step: 0.05, integer: false, defaultValue: 0.1 },
    backboneNoise: { min: 0, max: 1, step: 0.05, integer: false, defaultValue: 0, unit: 'Å' },
    seed: { min: 0, max: 2_147_483_647, step: 1, integer: true, defaultValue: 42 },
  },
  'localfold-af2': {
    recycles: { min: 0, max: 20, step: 1, integer: true, defaultValue: 1 },
    tolerance: { min: 0, max: 5, step: 0.1, integer: false, defaultValue: 0, unit: 'Å' },
    diffusionSteps: { min: 10, max: 200, step: 5, integer: true, defaultValue: 25 },
    maxSequences: { min: 1, max: 8, step: 1, integer: true, defaultValue: 2 },
    maxTotalFolds: { min: 1, max: 64, step: 1, integer: true, defaultValue: 16 },
    seed: { min: 0, max: 2_147_483_647, step: 1, integer: true, defaultValue: 0 },
  },
  'top-k-filter': {
    topK: { min: 1, max: 16, step: 1, integer: true, defaultValue: 2 },
    maxTotal: { min: 1, max: 64, step: 1, integer: true, defaultValue: 8 },
  },
  'designability-scorer': {
    scRmsdThreshold: { min: 0.5, max: 10, step: 0.1, integer: false, defaultValue: 2.0, unit: 'Å' },
    plddtThreshold: { min: 10, max: 99, step: 5, integer: true, defaultValue: 70 },
    motifRmsdThreshold: { min: 0.1, max: 10, step: 0.1, integer: false, defaultValue: 1.0, unit: 'Å' },
    ipaeThreshold: { min: 1, max: 31.75, step: 0.5, integer: false, defaultValue: 10.0, unit: 'Å' },
    ipsaeThreshold: { min: 0, max: 1, step: 0.05, integer: false, defaultValue: 0 },
    targetAlignedBinderRmsdThreshold: { min: 0.5, max: 20, step: 0.5, integer: false, defaultValue: 4.0, unit: 'Å' },
  },
};

/**
 * Strip non-numeric characters from a live text draft so letters like `5r` or `1e5`
 * can never appear in numeric inputs across Safari, Firefox, or Chrome.
 */
export function sanitizeNumericDraft(raw: string, integer: boolean): string {
  if (integer) {
    return raw.replace(/[^0-9]/g, '');
  }
  let seenDot = false;
  let out = '';
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (ch >= '0' && ch <= '9') {
      out += ch;
    } else if (ch === '.' && !seenDot) {
      seenDot = true;
      out += ch;
    }
  }
  return out;
}

/** Format a number cleanly without trailing floating-point noise (e.g. 0.15000000000000002 -> 0.15). */
export function formatNumericValue(n: number, spec: NumericFieldSpec): string {
  if (!Number.isFinite(n)) return String(spec.defaultValue);
  if (spec.integer) return String(Math.round(n));
  const stepStr = String(spec.step ?? 0.01);
  const dec = stepStr.includes('.') ? stepStr.split('.')[1].length : 2;
  const fixed = Number(n.toFixed(Math.max(dec, 2)));
  return String(fixed);
}

/** Validate a draft string while the user is typing (does not mutate/clamp mid-keystroke). */
export function validateNumericDraft(raw: string, spec: NumericFieldSpec): ValidationFeedback {
  const trimmed = raw.trim();
  const unitSuffix = spec.unit ? ` ${spec.unit}` : '';
  const rangeText = `${spec.min}–${spec.max}${unitSuffix}`;
  if (trimmed === '' || trimmed === '.') {
    return { status: 'error', message: `Enter ${rangeText}` };
  }
  const n = Number(trimmed);
  if (!Number.isFinite(n)) {
    return { status: 'error', message: `Enter a valid number (${rangeText})` };
  }
  if (n < spec.min) {
    return { status: 'error', value: n, message: `Min ${spec.min}${unitSuffix} (${rangeText})` };
  }
  if (n > spec.max) {
    return { status: 'error', value: n, message: `Max ${spec.max}${unitSuffix} (${rangeText})` };
  }
  if (spec.warnBelow && n < spec.warnBelow.threshold) {
    return { status: 'warn', value: n, message: spec.warnBelow.message };
  }
  return { status: 'ok', value: n };
}

/** Parse, round (if integer), and clamp a numeric draft on commit/blur. */
export function commitNumericDraft(raw: string, spec: NumericFieldSpec, fallback: number): number {
  const trimmed = raw.trim();
  if (trimmed === '' || trimmed === '.') {
    return clampNumeric(fallback, spec);
  }
  const n = Number(trimmed);
  if (!Number.isFinite(n)) {
    return clampNumeric(fallback, spec);
  }
  return clampNumeric(n, spec);
}

export function clampNumeric(value: number, spec: NumericFieldSpec): number {
  let n = Number.isFinite(value) ? value : spec.defaultValue;
  if (spec.integer) {
    n = Math.round(n);
  } else {
    n = Math.round(n * 10000) / 10000;
  }
  if (n < spec.min) n = spec.min;
  if (n > spec.max) n = spec.max;
  return n;
}

export interface SequenceValidationResult {
  status: 'ok' | 'warn' | 'error';
  numSequences: number;
  totalResidues: number;
  chainsInFirst: number;
  illegalChars: string[];
  emptyHeaders?: string[];
  message: string;
}

const VALID_AA_SET = new Set('ACDEFGHIKLMNPQRSTVWY'.split(''));

/** Validate plain or multi-FASTA amino acid sequence input (allowing `:` for multi-chain). */
export function validateSequenceText(raw: string): SequenceValidationResult {
  const text = raw.trim();
  if (!text) {
    return {
      status: 'error',
      numSequences: 0,
      totalResidues: 0,
      chainsInFirst: 0,
      illegalChars: [],
      emptyHeaders: [],
      message: 'Enter at least one amino acid sequence.',
    };
  }

  const lines = text.split(/\r?\n/);
  const entries: string[] = [];
  const emptyHeaders: string[] = [];
  let cur = '';
  let activeHeader: string | null = null;
  for (const line of lines) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith('>')) {
      if (activeHeader !== null && !cur) {
        emptyHeaders.push(activeHeader);
      } else if (cur) {
        entries.push(cur);
      }
      activeHeader = t.slice(1).trim().split(/\s+/)[0] || '(unnamed)';
      cur = '';
    } else {
      cur += t.replace(/\s+/g, '').toUpperCase();
    }
  }
  if (activeHeader !== null && !cur) {
    emptyHeaders.push(activeHeader);
  } else if (cur) {
    entries.push(cur);
  }

  if (entries.length === 0) {
    return {
      status: 'error',
      numSequences: 0,
      totalResidues: 0,
      chainsInFirst: 0,
      illegalChars: [],
      emptyHeaders,
      message: 'FASTA header found, but no sequence residues.',
    };
  }

  const illegal = new Set<string>();
  let totalResidues = 0;
  let minLen = Infinity;
  let maxLen = 0;
  for (const seq of entries) {
    let aaCount = 0;
    for (const ch of seq) {
      if (ch === ':' || ch === '/') continue;
      if (!VALID_AA_SET.has(ch)) {
        illegal.add(ch);
      } else {
        aaCount++;
      }
    }
    totalResidues += aaCount;
    if (aaCount < minLen) minLen = aaCount;
    if (aaCount > maxLen) maxLen = aaCount;
  }

  const chainsInFirst = entries[0].split(/[:/]/).filter(Boolean).length;
  if (illegal.size > 0) {
    const shown = [...illegal].slice(0, 6).join(', ');
    const list = illegal.size > 6 ? `${shown} (+${illegal.size - 6} more)` : shown;
    return {
      status: 'error',
      numSequences: entries.length,
      totalResidues,
      chainsInFirst,
      illegalChars: [...illegal],
      emptyHeaders,
      message: `Invalid residue character${illegal.size > 1 ? 's' : ''}: ${list} (use standard 20 AAs, or ':' between chains).`,
    };
  }

  if (totalResidues === 0 || maxLen === 0) {
    return {
      status: 'error',
      numSequences: entries.length,
      totalResidues,
      chainsInFirst,
      illegalChars: [],
      emptyHeaders,
      message: 'Sequence contains no amino acid residues (minimum 5 aa required).',
    };
  }

  if (maxLen < 5) {
    return {
      status: 'error',
      numSequences: entries.length,
      totalResidues,
      chainsInFirst,
      illegalChars: [],
      emptyHeaders,
      message: `Sequence too short (${maxLen} aa) — minimum 5 amino acids required.`,
    };
  }

  if (minLen < 5) {
    return {
      status: 'warn',
      numSequences: entries.length,
      totalResidues,
      chainsInFirst,
      illegalChars: [],
      emptyHeaders,
      message: `Contains an entry shorter than 5 aa (${minLen} aa) which will be skipped.`,
    };
  }

  if (emptyHeaders.length > 0) {
    const preview = emptyHeaders
      .slice(0, 3)
      .map((h) => `>${h}`)
      .join(', ');
    const more = emptyHeaders.length > 3 ? ` (+${emptyHeaders.length - 3} more)` : '';
    return {
      status: 'warn',
      numSequences: entries.length,
      totalResidues,
      chainsInFirst,
      illegalChars: [],
      emptyHeaders,
      message: `Empty FASTA record (${preview}${more}) has no sequence residues and will be skipped.`,
    };
  }

  const chainNote = chainsInFirst > 1 ? ` · ${chainsInFirst} chains` : '';
  const summary =
    entries.length === 1
      ? `${totalResidues} aa${chainNote}`
      : `${entries.length} sequences · ${totalResidues} total aa${chainNote}`;

  return {
    status: 'ok',
    numSequences: entries.length,
    totalResidues,
    chainsInFirst,
    illegalChars: [],
    emptyHeaders: [],
    message: summary,
  };
}

/**
 * Validate a residue selector token like `A54`, `A16-35`, `12`, `10-14`,
 * insertion codes (`A100A`, `A100A-100C`), negative resnums (`A-2`, `A-5--2`, `A-2-5`),
 * or colon-delimited multi-char chain selectors (`AA:1-5`, `AA:100A`), while rejecting reversed ranges (`A999-1`).
 */
export function isValidResidueToken(token: string): boolean {
  const t = token.trim();
  if (!t) return false;
  const colonM = /^([A-Za-z0-9_]+):(-?\d+)([A-Z])?(?:-(-?\d+)([A-Z])?)?$/.exec(t);
  const bareM = !colonM
    ? /^([A-Za-z_])?(-?\d+)([A-Z])?(?:-([A-Za-z_])?(-?\d+)([A-Z])?)?$/.exec(t)
    : null;
  if (!colonM && !bareM) return false;

  let start: number;
  let startIns: string | undefined;
  let end: number;
  let endIns: string | undefined;
  if (colonM) {
    start = Number(colonM[2]);
    startIns = colonM[3];
    if (colonM[4] !== undefined) {
      end = Number(colonM[4]);
      endIns = colonM[5];
    } else {
      end = start;
      endIns = startIns;
    }
  } else {
    const startChain = bareM![1];
    const endChain = bareM![4];
    if (startChain && endChain && startChain.toUpperCase() !== endChain.toUpperCase()) return false;
    start = Number(bareM![2]);
    startIns = bareM![3];
    if (bareM![5] !== undefined) {
      end = Number(bareM![5]);
      endIns = bareM![6];
    } else {
      end = start;
      endIns = startIns;
    }
  }
  if (!Number.isFinite(start) || !Number.isFinite(end)) return false;
  if (end < start) return false;
  if (end === start && (startIns ?? '') > (endIns ?? '')) return false;
  return true;
}

/**
 * Validate tiedPositions syntax (e.g. "A10=B10; A11=B11" or "A10=B10, A11=B11").
 * Returns null if valid or empty, or a human-readable error message.
 */
export function validateTiedPositions(raw: string): string | null {
  const text = (raw ?? '').trim();
  if (!text) return null;
  const rawGroups = text.split(/[;,]+/);
  let groupCount = 0;
  const seenResidues = new Set<string>();
  for (const rawGroup of rawGroups) {
    const groupText = rawGroup.trim();
    if (!groupText) continue;
    groupCount++;
    if (groupText.includes('=') && groupText.split('=').some((part) => !part.trim())) {
      return `Invalid tiedPositions group "${groupText}" (empty side around "="; expected e.g. "A10=B10").`;
    }
    const tokens = groupText.split(/[=\s]+/).filter(Boolean);
    if (tokens.length < 2) {
      return `Invalid tiedPositions group "${groupText}" (expected at least 2 residues, e.g. "A10=B10").`;
    }
    for (const tok of tokens) {
      const m = /^([A-Za-z_]):?(-?\d+)([A-Za-z])?$/.exec(tok);
      if (!m) {
        return `Invalid tiedPositions token "${tok}" in "${groupText}" (expected chain ID + residue number, e.g. "A10=B10" or "A100A=B100A").`;
      }
      const key = `${m[1].toUpperCase()}:${Number(m[2])}${m[3] ?? ''}`;
      if (seenResidues.has(key)) {
        return `Duplicate or overlapping tiedPositions residue "${tok}" in "${groupText}" (each residue may appear in at most one tie group).`;
      }
      seenResidues.add(key);
    }
  }
  if (groupCount === 0) {
    return `Invalid tiedPositions "${text}" (expected at least 2 residues, e.g. "A10=B10").`;
  }
  return null;
}

/** Validate RFD4 raw JSON override text (blank is valid). */
export function validateJsonOverride(raw: string): ValidationFeedback {
  const trimmed = raw.trim();
  if (!trimmed) return { status: 'ok' };
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return { status: 'error', message: 'Override must be a JSON object (e.g. {"contig": "70-70"}).' };
    }
    return { status: 'ok', message: 'Valid JSON object' };
  } catch (err) {
    return {
      status: 'error',
      message: `Invalid JSON: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

/** Validate RFD4 customSpecYaml (JSON object or YAML mapping; blank is valid). */
export function validateCustomSpecYaml(raw: string): ValidationFeedback {
  const trimmed = raw.trim();
  if (!trimmed) return { status: 'ok' };
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    return validateJsonOverride(trimmed);
  }
  let braceDepth = 0;
  let bracketDepth = 0;
  let hasMappingKey = false;
  for (const rawLine of trimmed.split(/\r?\n/)) {
    let inSingle = false;
    let inDouble = false;
    let clean = '';
    for (let i = 0; i < rawLine.length; i++) {
      const ch = rawLine[i];
      if (ch === "'" && !inDouble) {
        inSingle = !inSingle;
      } else if (ch === '"' && !inSingle && rawLine[i - 1] !== '\\') {
        inDouble = !inDouble;
      } else if (!inSingle && !inDouble) {
        if (ch === '#') break;
        if (ch === '{') braceDepth++;
        else if (ch === '}') braceDepth--;
        else if (ch === '[') bracketDepth++;
        else if (ch === ']') bracketDepth--;
        if (braceDepth < 0 || bracketDepth < 0) {
          return { status: 'error', message: 'Invalid customSpecYaml: unmatched closing bracket or brace.' };
        }
      }
      clean += ch;
    }
    if (inSingle || inDouble) {
      return { status: 'error', message: 'Invalid customSpecYaml: unclosed quote string.' };
    }
    const stripped = clean.trim();
    if (!stripped) continue;
    if (/^[A-Za-z_][A-Za-z0-9_-]*\s*:/.test(stripped)) {
      hasMappingKey = true;
    }
  }
  if (braceDepth !== 0 || bracketDepth !== 0) {
    return { status: 'error', message: 'Invalid customSpecYaml: unclosed "{" or "[".' };
  }
  if (!hasMappingKey) {
    return {
      status: 'error',
      message: 'Invalid customSpecYaml: expected a YAML mapping (e.g. "contig: 70-70") or JSON object.',
    };
  }
  return { status: 'ok', message: 'Valid YAML spec' };
}

/** Validate a 4-character RCSB PDB ID (e.g. 1PRW, 5O45) or extended 12-char ID. */
export function validatePdbId(raw: string): ValidationFeedback {
  const norm = normalizePdbId(raw);
  if (!norm) {
    return { status: 'error', message: 'Enter a 4-character RCSB PDB ID (e.g. 1PRW, 5O45).' };
  }
  if (!isValidPdbId(norm)) {
    return {
      status: 'error',
      message: `"${norm}" is not a valid PDB ID (expected 4 characters starting with a digit, e.g. 1PRW).`,
    };
  }
  return { status: 'ok', message: `RCSB PDB: ${norm}` };
}

/** Structure text check for custom PDB / mmCIF paste (enforces >=3 Cα residues unless non-protein context). */
export function validateStructureText(raw: string): ValidationFeedback {
  const trimmed = raw.trim();
  if (!trimmed) {
    return { status: 'error', message: 'Paste a PDB / mmCIF structure or upload a file.' };
  }
  const isCif = /_atom_site\./.test(trimmed);
  const hasAtom = /^(ATOM|HETATM)\s+/m.test(trimmed);
  if (!isCif && !hasAtom) {
    return {
      status: 'error',
      message: 'No PDB ATOM lines or mmCIF _atom_site records found.',
    };
  }
  try {
    const parsed = parseStructureText(trimmed);
    const hasNonProtein = parsed.diagnostics.hasLigands || parsed.diagnostics.hasNucleicAcids;
    if (parsed.nResidues < 3 && !hasNonProtein) {
      return {
        status: 'error',
        message: `Structure contains only ${parsed.nResidues} Cα residue${parsed.nResidues === 1 ? '' : 's'} (at least 3 Cα residues required).`,
      };
    }
    return {
      status: 'ok',
      message:
        parsed.nResidues >= 3
          ? `${isCif ? 'mmCIF' : 'PDB'} detected (${parsed.nResidues} aa)`
          : `${isCif ? 'mmCIF' : 'PDB'} non-protein context detected`,
    };
  } catch (err) {
    return {
      status: 'error',
      message: `Failed to parse structure: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

/**
 * Validate node-level parameter values before running the workflow.
 * Returns human-readable error strings for hard blocking issues.
 */
export function validateNodeParams(node: WorkflowNode): string[] {
  const errs: string[] = [];
  const label = node.label || node.adapterId;
  const p = node.params ?? {};
  const foldModel = node.adapterId === 'localfold-af2'
    ? LOCALFOLD_MODEL_OPTIONS.find((model) => model.id === (p.model ?? 'monomer'))
    : undefined;

  const numSpecs = Object.hasOwn(NUMERIC_SPECS, node.adapterId)
    ? NUMERIC_SPECS[node.adapterId]
    : undefined;
  if (numSpecs) {
    for (const [key, spec] of Object.entries(numSpecs)) {
      if (p[key] === undefined) continue;
      if (foldModel?.family === 'esmfold2' && ['recycles', 'tolerance', 'diffusionSteps'].includes(key)) continue;
      // Skip mode-irrelevant params
      if (node.adapterId === 'rfd4-proteina') {
        const mode = String(p.mode ?? 'unconditional');
        const usesFallbackLength =
          (mode === 'motif' || mode === 'conditioned') &&
          !String(p.contig ?? '').trim() &&
          !String(p.customSpecYaml ?? '').trim();
        if (key === 'length' && mode !== 'unconditional' && mode !== 'binder' && !usesFallbackLength) continue;
        if (key === 'partialFraction' && mode !== 'partial') continue;
      }
      const val = Number(p[key]);
      if (!Number.isFinite(val) || val < spec.min || val > spec.max) {
        const unit = spec.unit ? ` ${spec.unit}` : '';
        errs.push(`"${label}": ${key} must be between ${spec.min} and ${spec.max}${unit}.`);
      } else if (spec.integer && !Number.isInteger(val)) {
        errs.push(`"${label}": ${key} must be an integer.`);
      }
    }
  }

  // Enum & syntax parameter validation (prevents invalid recipes from crashing mid-run)
  if (node.adapterId === 'rfd4-proteina') {
    const mode = p.mode !== undefined ? String(p.mode) : 'unconditional';
    if (!VALID_RFD4_MODES.has(mode)) {
      errs.push(`"${label}": Unknown RFD4 mode "${mode}".`);
    }
    if (mode === 'preset' && p.presetId !== undefined) {
      const presetId = String(p.presetId);
      if (!VALID_RFD4_PRESET_IDS.has(presetId)) {
        errs.push(`"${label}": Unknown RFD4 preset "${presetId}".`);
      }
    }
    if (p.recycleSchedule !== undefined && !VALID_RFD4_RECYCLE_SCHEDULES.has(String(p.recycleSchedule))) {
      errs.push(`"${label}": Invalid recycleSchedule "${String(p.recycleSchedule)}" (expected none, low, medium, or high).`);
    }
    const customYaml = String(p.customSpecYaml ?? '').trim();
    if (customYaml) {
      const yamlCheck = validateCustomSpecYaml(customYaml);
      if (yamlCheck.status === 'error') {
        errs.push(`"${label}": ${yamlCheck.message}`);
      }
    }
  }

  if (node.adapterId === 'genie3') {
    const taskMode = p.taskMode !== undefined ? String(p.taskMode) : 'unconditional';
    if (p.taskMode !== undefined && !VALID_GENIE3_TASK_MODES.has(taskMode)) {
      errs.push(`"${label}": Unknown Genie 3 taskMode "${taskMode}" (expected unconditional, binder, or motif).`);
    }
    if (taskMode === 'motif' && !String(p.motifSpan ?? '').trim() && !String(p.contig ?? '').trim()) {
      errs.push(`"${label}": Genie 3 motif mode requires a motifSpan (e.g. A16-35) or contig specifying motif residues.`);
    }
  }

  if (node.adapterId === 'web-protein-mpnn') {
    const family = p.family !== undefined ? String(p.family) : 'solublempnn';
    const allowedModels = Object.hasOwn(VALID_MPNN_CATALOG, family)
      ? VALID_MPNN_CATALOG[family]
      : undefined;
    if (!allowedModels) {
      errs.push(`"${label}": Unknown ProteinMPNN family "${family}" (expected solublempnn, proteinmpnn, or ca).`);
    } else if (p.modelName !== undefined) {
      const modelName = String(p.modelName);
      if (!allowedModels.has(modelName)) {
        errs.push(
          `"${label}": Unknown ${family} modelName "${modelName}" (expected ${[...allowedModels].join(', ')}).`,
        );
      }
    }
    const fixedSpec = String(p.fixedPositions ?? '').trim();
    if (fixedSpec) {
      for (const tok of fixedSpec.split(/[,\s]+/).filter(Boolean)) {
        if (!/^[A-Za-z_]$/.test(tok) && !isValidResidueToken(tok)) {
          errs.push(
            `"${label}": Invalid fixedPositions token "${tok}" (expected e.g. "A", "A10", "A10-15", "A-2", or "A100A").`,
          );
        }
      }
    }
    if (typeof p.tiedPositions === 'string' && p.tiedPositions.trim()) {
      const tiedErr = validateTiedPositions(p.tiedPositions);
      if (tiedErr) {
        errs.push(`"${label}": ${tiedErr}`);
      }
    }
    if (typeof p.omitAminoAcids === 'string' && p.omitAminoAcids.trim()) {
      const compact = p.omitAminoAcids.trim().toUpperCase().replace(/[\s,;]+/g, '');
      const badChars = new Set<string>();
      const omittedCanonical = new Set<string>();
      for (const ch of compact) {
        if (VALID_AA_SET.has(ch)) omittedCanonical.add(ch);
        else if (ch !== 'X') badChars.add(ch);
      }
      if (badChars.size > 0) {
        errs.push(
          `"${label}": omitAminoAcids contains invalid amino acid character(s): ${[...badChars].join(', ')} (use standard 1-letter AA codes e.g. C, M).`,
        );
      } else if (omittedCanonical.size >= VALID_AA_SET.size) {
        errs.push(
          `"${label}": omitAminoAcids excludes all 20 standard amino acids — leave at least 1 amino acid allowed for design.`,
        );
      }
    }
  }

  if (node.adapterId === 'ligandmpnn' || node.adapterId === 'lasermpnn') {
    if (typeof p.tiedPositions === 'string' && p.tiedPositions.trim()) {
      const tiedErr = validateTiedPositions(p.tiedPositions);
      if (tiedErr) {
        errs.push(`"${label}": ${tiedErr}`);
      }
    }
    if (typeof p.omitAminoAcids === 'string' && p.omitAminoAcids.trim()) {
      const compact = p.omitAminoAcids.trim().toUpperCase().replace(/[\s,;]+/g, '');
      const badChars = new Set<string>();
      const omittedCanonical = new Set<string>();
      for (const ch of compact) {
        if (VALID_AA_SET.has(ch)) omittedCanonical.add(ch);
        else if (ch !== 'X') badChars.add(ch);
      }
      if (badChars.size > 0) {
        errs.push(
          `"${label}": omitAminoAcids contains invalid amino acid character(s): ${[...badChars].join(', ')} (use standard 1-letter AA codes e.g. C, M).`,
        );
      } else if (omittedCanonical.size >= VALID_AA_SET.size) {
        errs.push(
          `"${label}": omitAminoAcids excludes all 20 standard amino acids — leave at least 1 amino acid allowed for design.`,
        );
      }
    }
  }

  if (node.adapterId === 'localfold-af2') {
    const modelId = p.model !== undefined ? String(p.model) : 'monomer';
    if (p.model !== undefined && !foldModel) {
      errs.push(`"${label}": Unknown LocalFold model "${modelId}".`);
    }
    if (foldModel?.family === 'esmfold2') {
      const issue = esmfold2ParamIssue(p);
      if (issue) errs.push(`"${label}": ${issue}`);
    }
    const maxRecycles = modelId === 'mol1' || modelId.startsWith('multimer') ? 20 : 10;
    if (p.recycles !== undefined && foldModel?.family !== 'esmfold2') {
      const rec = Number(p.recycles);
      if (Number.isFinite(rec) && Number.isInteger(rec) && rec > maxRecycles && rec <= 20) {
        errs.push(
          `"${label}": recycles must be <= ${maxRecycles} for "${modelId}" (up to 20 allowed for mol1 and AF2-Multimer).`,
        );
      }
    }
    if (p.msaMode !== undefined && foldModel?.family !== 'esmfold2') {
      const msaMode = String(p.msaMode);
      if (!VALID_MSA_MODES.has(msaMode)) {
        errs.push(
          `"${label}": Invalid msaMode "${msaMode}" (expected single, mmseqs2, target-only, or a3m).`,
        );
      } else if (msaMode === 'a3m' && !String(p.msaA3mText ?? '').trim()) {
        errs.push(
          `"${label}": Custom .a3m MSA mode requires non-empty msaA3mText alignment data.`,
        );
      }
    }
  }

  if (p.unresolvedImportIssues !== undefined && p.unresolvedImportIssues !== null) {
    const formatIssue = (raw: unknown): string => {
      if (typeof raw === 'string') return raw.trim();
      if (raw === undefined || raw === null) return '';
      if (typeof raw === 'object') {
        try {
          return JSON.stringify(raw);
        } catch {
          return String(raw).trim();
        }
      }
      return String(raw).trim();
    };
    if (Array.isArray(p.unresolvedImportIssues)) {
      for (const rawIssue of p.unresolvedImportIssues) {
        const issue = formatIssue(rawIssue);
        if (issue) {
          errs.push(`"${label}": Unresolved imported setting — ${issue}`);
        }
      }
    } else {
      const issue = formatIssue(p.unresolvedImportIssues);
      if (issue) {
        errs.push(`"${label}": Unresolved imported setting — ${issue}`);
      }
    }
  }

  if (node.adapterId === 'pdb-input') {
    const mode = p.mode !== undefined ? String(p.mode) : 'demo-bundle';
    if (!VALID_PDB_INPUT_MODES.has(mode)) {
      errs.push(`"${label}": Unknown PDB Input mode "${mode}" (expected demo-bundle, pdb-id, or custom).`);
    } else if (mode === 'custom') {
      const rawPdb = String(p.pdbText ?? '');
      const missingUpload = Boolean(p.missingUpload);
      const expectedShaRaw = String(p.expectedPdbSha256 ?? '').trim();
      const expectedSha = expectedShaRaw.toLowerCase();
      const expectedFilename = String(p.expectedFilename ?? p.filename ?? '').trim();

      if (expectedSha && !isValidSha256Hex(expectedSha)) {
        errs.push(
          `"${label}": Invalid expectedPdbSha256 digest "${expectedShaRaw}" (expected a 64-character SHA-256 hex digest).`,
        );
      } else if (!rawPdb.trim()) {
        if (missingUpload || expectedSha) {
          const fileNote = expectedFilename ? ` "${expectedFilename}"` : '';
          const digestNote = expectedSha ? ` (expected SHA-256 ${expectedSha.slice(0, 12)}…)` : '';
          errs.push(
            `"${label}": Custom upload${fileNote} was omitted from this shared recipe${digestNote}. Re-upload the matching PDB/mmCIF file, or switch Source mode / clear expectedPdbSha256.`,
          );
        } else {
          const check = validateStructureText(rawPdb);
          if (check.status === 'error') {
            errs.push(`"${label}": ${check.message}`);
          }
        }
      } else {
        const check = validateStructureText(rawPdb);
        if (check.status === 'error') {
          errs.push(`"${label}": ${check.message}`);
        } else if (expectedSha) {
          const actualSha = computeUploadDigest(rawPdb);
          if (actualSha !== expectedSha) {
            const fileNote = expectedFilename ? ` "${expectedFilename}"` : '';
            errs.push(
              `"${label}": Uploaded structure digest (${actualSha.slice(0, 12)}…) does not match expected custom upload${fileNote} (SHA-256 ${expectedSha.slice(0, 12)}…). Re-upload the matching file, or explicitly clear expectedPdbSha256 to replace the input.`,
            );
          }
        }
      }
    } else if (mode === 'pdb-id') {
      const check = validatePdbId(String(p.pdbId ?? ''));
      if (check.status === 'error') {
        errs.push(`"${label}": ${check.message}`);
      }
    }
  }

  if (node.adapterId === 'sequence-input') {
    const check = validateSequenceText(String(p.sequence ?? ''));
    if (check.status === 'error') {
      errs.push(`"${label}": ${check.message}`);
    }
  }

  if (node.adapterId === 'top-k-filter') {
    if (p.mode !== undefined && !VALID_TOPK_MODES.has(String(p.mode))) {
      errs.push(
        `"${label}": Invalid mode "${String(p.mode)}" (expected per-backbone or global).`,
      );
    }
    if (p.metric !== undefined && !VALID_TOPK_METRICS.has(String(p.metric))) {
      errs.push(
        `"${label}": Invalid metric "${String(p.metric)}" (expected score or globalScore).`,
      );
    }
    if (p.direction !== undefined && !VALID_TOPK_DIRECTIONS.has(String(p.direction))) {
      errs.push(
        `"${label}": Invalid direction "${String(p.direction)}" (expected min or max).`,
      );
    }
  }

  // Schema-driven fallback for any plug-and-play adapter not in the built-in hand-tuned set
  const BUILTIN_HAND_VALIDATED = new Set([
    'pdb-input',
    'sequence-input',
    'rfd4-proteina',
    'genie3',
    'web-protein-mpnn',
    'localfold-af2',
    'top-k-filter',
    'designability-scorer',
  ]);
  if (!BUILTIN_HAND_VALIDATED.has(node.adapterId)) {
    const manifest = Object.hasOwn(ADAPTER_REGISTRY, node.adapterId)
      ? ADAPTER_REGISTRY[node.adapterId]
      : undefined;
    if (manifest && (manifest.paramSchema?.length || manifest.modes?.length)) {
      errs.push(...validateParamsWithSchema(manifest, p, { label }));
    }
  }

  return errs;
}
