// Shared parameter validation, numeric sanitization, and scientific format checks
// for NodeInspector (live feedback) and validateGraph (pre-run checks).

import type { WorkflowNode } from './types';

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
    nSampleStep: { min: 10, max: 250, step: 10, integer: true, defaultValue: 100 },
    directionScale: { min: 0.1, max: 1.5, step: 0.1, integer: false, defaultValue: 0.8 },
    noiseScale: { min: 0, max: 1.5, step: 0.1, integer: false, defaultValue: 1.0 },
    seed: { min: 0, max: 2_147_483_647, step: 1, integer: true, defaultValue: 42 },
  },
  'web-protein-mpnn': {
    numSequences: { min: 1, max: 8, step: 1, integer: true, defaultValue: 2 },
    temperature: { min: 0.01, max: 1.5, step: 0.05, integer: false, defaultValue: 0.1 },
    seed: { min: 0, max: 2_147_483_647, step: 1, integer: true, defaultValue: 42 },
  },
  'localfold-af2': {
    recycles: { min: 0, max: 4, step: 1, integer: true, defaultValue: 1 },
    diffusionSteps: { min: 10, max: 200, step: 5, integer: true, defaultValue: 25 },
    maxSequences: { min: 1, max: 8, step: 1, integer: true, defaultValue: 2 },
    seed: { min: 0, max: 2_147_483_647, step: 1, integer: true, defaultValue: 0 },
  },
  'designability-scorer': {
    scRmsdThreshold: { min: 0.5, max: 10, step: 0.1, integer: false, defaultValue: 2.0, unit: 'Å' },
    plddtThreshold: { min: 10, max: 99, step: 5, integer: true, defaultValue: 70 },
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
      message: 'Enter at least one amino acid sequence.',
    };
  }

  const lines = text.split(/\r?\n/);
  const entries: string[] = [];
  let cur = '';
  for (const line of lines) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith('>')) {
      if (cur) entries.push(cur);
      cur = '';
    } else {
      cur += t.replace(/\s+/g, '').toUpperCase();
    }
  }
  if (cur) entries.push(cur);

  if (entries.length === 0) {
    return {
      status: 'error',
      numSequences: 0,
      totalResidues: 0,
      chainsInFirst: 0,
      illegalChars: [],
      message: 'FASTA header found, but no sequence residues.',
    };
  }

  const illegal = new Set<string>();
  let totalResidues = 0;
  let minLen = Infinity;
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
  }

  const chainsInFirst = entries[0].split(/[:/]/).filter(Boolean).length;
  if (illegal.size > 0) {
    const list = [...illegal].slice(0, 6).join(', ');
    return {
      status: 'error',
      numSequences: entries.length,
      totalResidues,
      chainsInFirst,
      illegalChars: [...illegal],
      message: `Invalid residue character${illegal.size > 1 ? 's' : ''}: ${list} (use standard 20 AAs, or ':' between chains).`,
    };
  }

  if (minLen < 5) {
    return {
      status: 'warn',
      numSequences: entries.length,
      totalResidues,
      chainsInFirst,
      illegalChars: [],
      message: `Very short sequence (${minLen} aa) — folding models usually need ≥ 10 aa.`,
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
    message: summary,
  };
}

/** Validate a residue selector token like `A54`, `A16-35`, `12`, `10-14`. */
export function isValidResidueToken(token: string): boolean {
  const t = token.trim();
  if (!t) return false;
  return /^[A-Za-z]?\d+(?:-\d+)?$/.test(t);
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

/** Lightweight structure text check for custom PDB / mmCIF paste. */
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
  return { status: 'ok', message: isCif ? 'mmCIF detected' : 'PDB ATOM records detected' };
}

/**
 * Validate node-level parameter values before running the workflow.
 * Returns human-readable error strings for hard blocking issues.
 */
export function validateNodeParams(node: WorkflowNode): string[] {
  const errs: string[] = [];
  const label = node.label || node.adapterId;
  const p = node.params ?? {};

  const numSpecs = NUMERIC_SPECS[node.adapterId];
  if (numSpecs) {
    for (const [key, spec] of Object.entries(numSpecs)) {
      if (p[key] === undefined) continue;
      // Skip mode-irrelevant params
      if (node.adapterId === 'rfd4-proteina') {
        const mode = String(p.mode ?? 'unconditional');
        if (key === 'length' && mode !== 'unconditional') continue;
        if (key === 'partialFraction' && mode !== 'partial') continue;
      }
      const val = Number(p[key]);
      if (!Number.isFinite(val) || val < spec.min || val > spec.max) {
        const unit = spec.unit ? ` ${spec.unit}` : '';
        errs.push(`"${label}": ${key} must be between ${spec.min} and ${spec.max}${unit}.`);
      }
    }
  }

  if (node.adapterId === 'pdb-input' && p.mode === 'custom') {
    const check = validateStructureText(String(p.pdbText ?? ''));
    if (check.status === 'error') {
      errs.push(`"${label}": ${check.message}`);
    }
  }

  if (node.adapterId === 'sequence-input') {
    const check = validateSequenceText(String(p.sequence ?? ''));
    if (check.status === 'error') {
      errs.push(`"${label}": ${check.message}`);
    }
  }

  if (node.adapterId === 'rfd4-conditioner' && String(p.taskPreset ?? 'custom') === 'custom') {
    const rawJson = String(p.rawJsonOverride ?? '').trim();
    if (rawJson) {
      const check = validateJsonOverride(rawJson);
      if (check.status === 'error') {
        errs.push(`"${label}": ${check.message}`);
      }
    } else if (!String(p.contig ?? '').trim()) {
      errs.push(`"${label}": Contig cannot be empty in custom mode.`);
    }
  }

  return errs;
}
