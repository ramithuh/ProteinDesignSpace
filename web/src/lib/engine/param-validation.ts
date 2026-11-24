// Shared parameter validation, numeric sanitization, and scientific format checks
// for NodeInspector (live feedback) and validateGraph (pre-run checks).

import type { WorkflowNode } from './types';
import { isValidPdbId, normalizePdbId } from '$lib/protein-ops/rcsb';
import { isValidUniProtId, normalizeUniProtId } from '$lib/protein-ops/alphafold-db';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { ADAPTER_REGISTRY } from './registry';
import {
  fieldToNumericSpec,
  isValidResidueSelectorToken,
  validateParamsWithSchema,
} from './param-schema';

const SHA256_HEX_RE = /^[0-9a-f]{64}$/i;

function safeStr(v: unknown): string {
  return typeof v === 'string'
    ? v
    : typeof v === 'number' || typeof v === 'boolean'
      ? String(v)
      : '';
}

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
    fixValue?: number;
  };
}

export interface ValidationFeedback {
  status: 'ok' | 'warn' | 'error';
  message?: string;
  value?: number;
}

/**
 * Derived directly from `ADAPTER_REGISTRY[id].paramSchema` via `fieldToNumericSpec`
 * (zero duplicated bounds, defaults, or `warnBelow` guardrails).
 */
export const NUMERIC_SPECS: Record<string, Record<string, NumericFieldSpec>> = Object.fromEntries(
  Object.entries(ADAPTER_REGISTRY)
    .map(([adapterId, manifest]) => {
      const specs: Record<string, NumericFieldSpec> = {};
      for (const field of manifest.paramSchema ?? []) {
        if (field.kind === 'number' || field.kind === 'integer') {
          specs[field.key] = fieldToNumericSpec(field);
        }
      }
      return [adapterId, specs] as const;
    })
    .filter(([, specs]) => Object.keys(specs).length > 0),
);

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

/**
 * Normalize EvoDiff template / fixedSequence inputs by converting explicit
 * `#MASK#`, `[MASK]`, or `<MASK>` tokens into `#` before stripping whitespace.
 */
export function sanitizeEvoDiffFixedSequence(raw: unknown): string {
  return String(raw ?? '')
    .replace(/#MASK#|\[MASK\]|<MASK>/gi, '#')
    .replace(/\s+/g, '')
    .toUpperCase();
}

/**
 * Validate plain or multi-FASTA amino acid sequence input (allowing `:` for multi-chain). */
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
 * Delegates directly to the canonical `isValidResidueSelectorToken` in `param-schema.ts`.
 */
export function isValidResidueToken(token: string): boolean {
  return isValidResidueSelectorToken(token, { allowBareChain: false });
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

/** Validate a UniProt accession (e.g. P69905, P04637, Q9NZQ7) for AlphaFold DB fetch. */
export function validateUniProtId(raw: string): ValidationFeedback {
  const norm = normalizeUniProtId(raw);
  if (!norm) {
    return { status: 'error', message: 'Enter a UniProt accession (e.g. P69905, P04637, Q9NZQ7).' };
  }
  if (!isValidUniProtId(norm)) {
    return {
      status: 'error',
      message: `"${norm}" is not a valid UniProt accession (expected e.g. P69905 or Q9NZQ7).`,
    };
  }
  return { status: 'ok', message: `AlphaFold DB: ${norm}` };
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
 * Always runs declarative `validateParamsWithSchema` first, followed by
 * `manifest.crossValidate` and domain-specific format/integrity checks.
 * Returns human-readable error strings for hard blocking issues.
 */
export function validateNodeParams(
  node: WorkflowNode,
  opts?: { connectedInputs?: Iterable<string> },
): string[] {
  const errs: string[] = [];
  const label = safeStr(node.label) || safeStr(node.adapterId);
  const p =
    node.params && typeof node.params === 'object' && !Array.isArray(node.params)
      ? node.params
      : {};
  const manifest = Object.hasOwn(ADAPTER_REGISTRY, node.adapterId)
    ? ADAPTER_REGISTRY[node.adapterId]
    : undefined;
  const connectedSet =
    opts?.connectedInputs !== undefined ? new Set(opts.connectedInputs) : undefined;

  // 1. Canonical declarative schema validation (ALWAYS runs for every registered adapter)
  if (manifest && (manifest.paramSchema?.length || manifest.modes?.length)) {
    errs.push(
      ...validateParamsWithSchema(manifest, p, {
        label,
        ...(connectedSet ? { connectedInputs: connectedSet } : {}),
      }),
    );
  }

  // 2. Manifest-declared cross-field & cross-input validation
  if (manifest?.crossValidate) {
    const safeParams: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(p)) {
      if (
        v === null ||
        typeof v !== 'object' ||
        k === 'unresolvedImportIssues' ||
        k === 'residueMapping'
      ) {
        safeParams[k] = v;
      }
    }
    try {
      const crossErrs = manifest.crossValidate(
        safeParams,
        connectedSet ? { connectedInputs: connectedSet } : undefined,
      );
      for (const msg of crossErrs) {
        errs.push(msg.startsWith(`"${label}":`) ? msg : `"${label}": ${msg}`);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Cross-validation failed.';
      errs.push(msg.startsWith(`"${label}":`) ? msg : `"${label}": ${msg}`);
    }
  }

  // 3. Domain-specific syntax, structure, sequence, and upload-integrity checks
  if (node.adapterId === 'rfd4-proteina') {
    if (
      p.customSpecYaml !== undefined &&
      p.customSpecYaml !== null &&
      typeof p.customSpecYaml !== 'string'
    ) {
      errs.push(`"${label}": customSpecYaml must be a string.`);
    } else {
      const customYaml = safeStr(p.customSpecYaml).trim();
      if (customYaml) {
        const yamlCheck = validateCustomSpecYaml(customYaml);
        if (yamlCheck.status === 'error') {
          errs.push(`"${label}": ${yamlCheck.message}`);
        }
      }
    }
  }

  if (node.adapterId === 'web-protein-mpnn' || node.adapterId === 'ligandmpnn') {
    if (typeof p.tiedPositions === 'string' && p.tiedPositions.trim()) {
      const tiedErr = validateTiedPositions(p.tiedPositions);
      if (tiedErr) {
        errs.push(`"${label}": ${tiedErr}`);
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
          return '';
        }
      }
      return safeStr(raw).trim();
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
    if (p.mode !== undefined && p.mode !== null && typeof p.mode !== 'string') {
      errs.push(`"${label}": mode must be a string.`);
    }
    const mode = typeof p.mode === 'string' ? p.mode : 'demo-bundle';
    if (mode === 'custom') {
      if (p.pdbText !== undefined && p.pdbText !== null && typeof p.pdbText !== 'string') {
        errs.push(`"${label}": pdbText must be a string.`);
      }
      if (
        p.expectedPdbSha256 !== undefined &&
        p.expectedPdbSha256 !== null &&
        typeof p.expectedPdbSha256 !== 'string'
      ) {
        errs.push(`"${label}": expectedPdbSha256 must be a string.`);
      }
      if (
        p.expectedFilename !== undefined &&
        p.expectedFilename !== null &&
        typeof p.expectedFilename !== 'string'
      ) {
        errs.push(`"${label}": expectedFilename must be a string.`);
      }
      if (p.filename !== undefined && p.filename !== null && typeof p.filename !== 'string') {
        errs.push(`"${label}": filename must be a string.`);
      }
      const rawPdb = safeStr(p.pdbText);
      const missingUpload = Boolean(p.missingUpload);
      const expectedShaRaw = safeStr(p.expectedPdbSha256).trim();
      const expectedSha = expectedShaRaw.toLowerCase();
      const expectedFilename = safeStr(p.expectedFilename ?? p.filename).trim();

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
      if (p.pdbId !== undefined && p.pdbId !== null && typeof p.pdbId !== 'string') {
        errs.push(`"${label}": pdbId must be a string.`);
      } else {
        const check = validatePdbId(safeStr(p.pdbId));
        if (check.status === 'error') {
          errs.push(`"${label}": ${check.message}`);
        }
      }
    } else if (mode === 'alphafold-db') {
      if (p.uniprotId !== undefined && p.uniprotId !== null && typeof p.uniprotId !== 'string') {
        errs.push(`"${label}": uniprotId must be a string.`);
      } else {
        const check = validateUniProtId(safeStr(p.uniprotId));
        if (check.status === 'error') {
          errs.push(`"${label}": ${check.message}`);
        }
      }
    }
  }

  if (node.adapterId === 'localfold-af2' && !connectedSet?.has('template')) {
    const rawTpl = String(p.templatePdbText ?? '');
    const missingTplUpload = Boolean(p.missingTemplateUpload);
    const expectedShaRaw = String(p.expectedTemplateSha256 ?? '').trim();
    const expectedSha = expectedShaRaw.toLowerCase();
    const expectedFilename = String(p.expectedTemplateFilename ?? p.templateFilename ?? '').trim();

    if (expectedSha && !isValidSha256Hex(expectedSha)) {
      errs.push(
        `"${label}": Invalid expectedTemplateSha256 digest "${expectedShaRaw}" (expected a 64-character SHA-256 hex digest).`,
      );
    } else if (!rawTpl.trim()) {
      if (expectedSha && !missingTplUpload) {
        const fileNote = expectedFilename ? ` "${expectedFilename}"` : '';
        errs.push(
          `"${label}": Expected inline template${fileNote} (SHA-256 ${expectedSha.slice(0, 12)}…) is missing. Re-upload the matching PDB/mmCIF file, or clear expectedTemplateSha256.`,
        );
      }
    } else if (expectedSha) {
      const actualSha = computeUploadDigest(rawTpl);
      if (actualSha !== expectedSha) {
        const fileNote = expectedFilename ? ` "${expectedFilename}"` : '';
        errs.push(
          `"${label}": Uploaded template digest (${actualSha.slice(0, 12)}…) does not match expected inline template${fileNote} (SHA-256 ${expectedSha.slice(0, 12)}…). Re-upload the matching file, or explicitly clear expectedTemplateSha256 to replace the template.`,
        );
      }
    }
  }

  if (node.adapterId === 'sequence-input') {
    if (p.sequence !== undefined && p.sequence !== null && typeof p.sequence !== 'string') {
      errs.push(`"${label}": sequence must be a string.`);
    } else {
      const check = validateSequenceText(safeStr(p.sequence));
      if (check.status === 'error') {
        errs.push(`"${label}": ${check.message}`);
      }
    }
  }

  return errs;
}
