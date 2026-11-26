// Declarative adapter parameter schema helpers:
// Single source of truth for active mode resolution, mode/tier field filtering,
// schema-driven validation, and default parameter synthesis.

import type {
  AdapterManifest,
  AdapterModeSpec,
  AdapterParamField,
} from './types';
import { validateStepCount } from './step-policy';

export interface SchemaNumericSpec {
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

export interface SchemaValidationOptions {
  /** Human-readable node label used in error messages (defaults to manifest.shortName). */
  label?: string;
  /**
   * Connected input port names on the node (e.g. `['structure']`).
   * When provided, active mode `requiredInputs` are checked against this set.
   */
  connectedInputs?: Iterable<string>;
}

const VALID_AA_CHAR_RE = /^[ACDEFGHIKLMNPQRSTVWYX]$/i;

/**
 * Canonical single-source-of-truth residue selector token validator used across
 * `param-schema.ts`, `param-validation.ts`, `SchemaParamFields.svelte`, and `NodeInspector.svelte`.
 *
 * Accepts:
 * - Residue numbers / ranges (`A54`, `A16-35`, `12`, `10-14`)
 * - Insertion codes (`A100A`, `A100A-100C`)
 * - Negative residue numbers (`A-2`, `A-5--2`, `A-2-5`)
 * - Colon-delimited multi-char chain selectors (`AA:1-5`, `AA:100A`)
 * - Bare 1–4 char chain tokens (`A`, `AA`, `HL`) when `opts.allowBareChain` is true
 * - Chain-qualified ligand/CCD tokens (`A:SAM`, `A:ZN`, `A:FE2`) when `opts.allowLigandRef` is true
 *
 * Rejects reversed ranges (`A999-1`, `A10B-10A`), cross-chain ranges (`A1-B5`), and malformed tokens.
 */
export function isValidResidueSelectorToken(
  tok: string,
  opts?: { allowBareChain?: boolean; allowLigandRef?: boolean },
): boolean {
  const t = tok.trim();
  if (!t) return false;
  if (opts?.allowBareChain && /^[A-Za-z_][A-Za-z0-9_]{0,3}$/.test(t) && !/^\d+$/.test(t)) {
    return true;
  }
  if (
    opts?.allowLigandRef &&
    /^[A-Za-z0-9_]{1,4}:(?=[A-Za-z0-9_]*[A-Za-z])[A-Za-z0-9_]{1,5}$/.test(t)
  ) {
    return true;
  }
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
 * Validate `designability-scorer` `residueMapping` syntax before runtime.
 * Mirrors `parseReferenceResidueMapping` in `scheduler.ts`.
 * Returns `null` when empty or valid, or a human-readable error message.
 */
export function validateResidueMappingSyntax(raw: unknown): string | null {
  if (raw === undefined || raw === null || raw === '') return null;
  const checkInt = (v: unknown, role: string): string | null => {
    if (typeof v !== 'number' || !Number.isInteger(v) || v < 0) {
      const shown =
        typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean' ? String(v) : typeof v;
      return `Invalid residueMapping: ${role} must be a strict finite integer (>= 0; got ${shown}).`;
    }
    return null;
  };
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    for (const tok of trimmed.split(/[,\s;]+/).filter(Boolean)) {
      if (!/^(\d+)\s*(?::|->|=)\s*(\d+)$/.test(tok)) {
        return `Invalid residueMapping entry "${tok}". Expected 0-based "foldIdx:refIdx" pairs (e.g. "0:0, 1:1, 2:2").`;
      }
    }
    return null;
  }
  if (Array.isArray(raw)) {
    for (const entry of raw) {
      if (!Array.isArray(entry) || entry.length < 2) {
        return 'Invalid residueMapping array: every entry must be a [foldIdx, refIdx] pair.';
      }
      const e0 = checkInt(entry[0], 'fold index');
      if (e0) return e0;
      const e1 = checkInt(entry[1], 'reference index');
      if (e1) return e1;
    }
    return null;
  }
  if (typeof raw === 'object') {
    const obj = raw as {
      pairs?: unknown;
      foldIndices?: unknown;
      referenceIndices?: unknown;
    };
    if (Array.isArray(obj.pairs)) {
      for (const entry of obj.pairs) {
        if (!Array.isArray(entry) || entry.length < 2) {
          return 'Invalid residueMapping.pairs: every entry must be a [foldIdx, refIdx] pair.';
        }
        const e0 = checkInt(entry[0], 'fold index');
        if (e0) return e0;
        const e1 = checkInt(entry[1], 'reference index');
        if (e1) return e1;
      }
      return null;
    }
    if (Array.isArray(obj.foldIndices) && Array.isArray(obj.referenceIndices)) {
      if (obj.foldIndices.length !== obj.referenceIndices.length) {
        return `Invalid residueMapping: foldIndices (${obj.foldIndices.length}) and referenceIndices (${obj.referenceIndices.length}) must have equal length.`;
      }
      for (let i = 0; i < obj.foldIndices.length; i++) {
        const e0 = checkInt(obj.foldIndices[i], 'fold index');
        if (e0) return e0;
        const e1 = checkInt(obj.referenceIndices[i], 'reference index');
        if (e1) return e1;
      }
      return null;
    }
    return 'Invalid residueMapping object: provide either "pairs" or both "foldIndices" and "referenceIndices".';
  }
  return 'Invalid residueMapping parameter type.';
}

function resolveDefaultModeId(manifest: AdapterManifest): string | null {
  if (!manifest.modes || manifest.modes.length === 0) return null;
  if (manifest.modeParamKey && manifest.paramSchema) {
    const modeField = manifest.paramSchema.find((f) => f.key === manifest.modeParamKey);
    if (modeField && typeof modeField.defaultValue === 'string' && modeField.defaultValue.trim()) {
      return modeField.defaultValue;
    }
  }
  return manifest.modes[0].id;
}

function isPrimitiveParam(v: unknown): v is string | number | boolean {
  return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean';
}

function safeParamStr(v: unknown): string {
  return isPrimitiveParam(v) ? String(v) : '';
}

/**
 * Resolve the active `AdapterModeSpec` for a manifest given current node `params`.
 * - Returns `null` if the adapter declares no `modes`.
 * - Falls back to the schema's default mode (or `modes[0]`) when the mode param is unset/empty.
 * - Returns `null` when an explicit, unrecognized mode value is passed.
 */
export function getActiveModeSpec(
  manifest: AdapterManifest | undefined | null,
  params?: Record<string, unknown>,
): AdapterModeSpec | null {
  if (!manifest?.modes || manifest.modes.length === 0) return null;
  const key = manifest.modeParamKey;
  const raw = key ? params?.[key] : undefined;
  const rawStr = safeParamStr(raw).trim();
  const modeId = rawStr !== '' ? rawStr : resolveDefaultModeId(manifest);
  if (!modeId) return null;
  return manifest.modes.find((m) => m.id === modeId) ?? null;
}

/**
 * Return the `AdapterParamField[]` visible for the current active mode and optional tier.
 * Fields without a `modes` restriction are visible across all modes.
 */
export function getVisibleParamFields(
  manifest: AdapterManifest | undefined | null,
  params?: Record<string, unknown>,
  tier?: 'primary' | 'advanced',
): AdapterParamField[] {
  if (!manifest?.paramSchema || manifest.paramSchema.length === 0) return [];

  let activeModeId: string | null = null;
  if (manifest.modes && manifest.modes.length > 0) {
    const activeSpec = getActiveModeSpec(manifest, params);
    if (activeSpec) {
      activeModeId = activeSpec.id;
    } else if (manifest.modeParamKey) {
      const raw = params?.[manifest.modeParamKey];
      const rawStr = safeParamStr(raw).trim();
      if (rawStr !== '') {
        activeModeId = rawStr;
      }
    }
  }

  return manifest.paramSchema.filter((field) => {
    if (tier !== undefined && field.tier !== tier) return false;
    if (field.modes && field.modes.length > 0 && activeModeId !== null) {
      return field.modes.includes(activeModeId);
    }
    return true;
  });
}

/** Convert a numeric `AdapterParamField` into a `NumericInput`-compatible spec object. */
export function fieldToNumericSpec(
  field: AdapterParamField,
  activeMode?: string | null,
): SchemaNumericSpec {
  const isInt = field.kind === 'integer';
  const defaultNum =
    typeof field.defaultValue === 'number' && Number.isFinite(field.defaultValue)
      ? field.defaultValue
      : typeof field.min === 'number' && Number.isFinite(field.min)
        ? field.min
        : 0;
  let warnBelow: SchemaNumericSpec['warnBelow'];
  if (field.warnBelow) {
    const modeMatches =
      !field.warnBelow.modes ||
      field.warnBelow.modes.length === 0 ||
      activeMode === undefined ||
      (activeMode !== null && field.warnBelow.modes.includes(activeMode));
    if (modeMatches) {
      warnBelow = {
        threshold: field.warnBelow.threshold,
        message: field.warnBelow.message,
        ...(field.warnBelow.fixValue !== undefined ? { fixValue: field.warnBelow.fixValue } : {}),
      };
    }
  }
  return {
    min: typeof field.min === 'number' ? field.min : 0,
    max: typeof field.max === 'number' ? field.max : 1_000_000,
    step: typeof field.step === 'number' ? field.step : isInt ? 1 : 0.1,
    integer: isInt,
    defaultValue: defaultNum,
    ...(field.unit ? { unit: field.unit } : {}),
    ...(warnBelow ? { warnBelow } : {}),
  };
}

/**
 * Synthesize a fresh default parameter object directly from `manifest.paramSchema`.
 * Useful for future plug-and-play adapters without hand-coded switch branches.
 */
export function getDefaultParamsFromSchema(
  manifest: AdapterManifest | undefined | null,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (!manifest?.paramSchema) return out;
  for (const field of manifest.paramSchema) {
    const val = field.defaultValue;
    if (val !== null && typeof val === 'object') {
      out[field.key] = structuredClone(val);
    } else {
      out[field.key] = val;
    }
  }
  return out;
}

/**
 * Validate `params` declaratively against `manifest.modes` and `manifest.paramSchema`:
 * - Checks active mode validity against `manifest.modes`
 * - Checks active mode `requiredInputs` when `opts.connectedInputs` is supplied
 * - Checks numeric bounds and integer constraints for visible fields
 * - Checks `select` options, `boolean` types, `aa-set` amino acid letters, and `residue-list` tokens
 */
export function validateParamsWithSchema(
  manifest: AdapterManifest | undefined | null,
  params?: Record<string, unknown>,
  opts: SchemaValidationOptions = {},
): string[] {
  if (!manifest) return [];
  const errs: string[] = [];
  const label = opts.label || manifest.shortName || manifest.id;
  const p = params ?? {};

  // 1. Mode validation & required mode inputs
  let modeInvalid = false;
  if (manifest.modes && manifest.modes.length > 0 && manifest.modeParamKey) {
    const modeKey = manifest.modeParamKey;
    const rawMode = p[modeKey];
    if (rawMode !== undefined && rawMode !== null && !isPrimitiveParam(rawMode)) {
      const allowed = manifest.modes.map((m) => m.id).join(', ');
      errs.push(`"${label}": ${modeKey} must be a string (expected ${allowed}).`);
      modeInvalid = true;
    } else {
      const activeMode = getActiveModeSpec(manifest, p);
      if (rawMode !== undefined && rawMode !== null && safeParamStr(rawMode).trim() !== '' && !activeMode) {
        const allowed = manifest.modes.map((m) => m.id).join(', ');
        errs.push(`"${label}": Unknown ${modeKey} "${safeParamStr(rawMode)}" (expected ${allowed}).`);
        modeInvalid = true;
      } else if (activeMode && opts.connectedInputs !== undefined) {
        const connected = new Set(opts.connectedInputs);
        for (const reqPort of activeMode.requiredInputs ?? []) {
          if (!connected.has(reqPort)) {
            errs.push(
              `"${label}" (${activeMode.label}) requires a connected "${reqPort}" input.`,
            );
          }
        }
      }
    }
  }

  // 2. Field-level validation over mode-applicable fields
  const fields = getVisibleParamFields(manifest, p);
  const numericErrorKeys = new Set<string>();
  for (const field of fields) {
    const raw = p[field.key];
    if (raw === undefined) continue;
    // Avoid double-reporting an invalid mode key already flagged above
    if (modeInvalid && field.key === manifest.modeParamKey) continue;

    if (field.kind === 'number' || field.kind === 'integer') {
      const val = isPrimitiveParam(raw) ? Number(raw) : NaN;
      const unit = field.unit ? ` ${field.unit}` : '';
      const hasMin = typeof field.min === 'number';
      const hasMax = typeof field.max === 'number';
      if (!Number.isFinite(val)) {
        errs.push(`"${label}": ${field.key} must be a valid number.`);
        numericErrorKeys.add(field.key);
      } else if ((hasMin && val < field.min!) || (hasMax && val > field.max!)) {
        if (hasMin && hasMax) {
          errs.push(`"${label}": ${field.key} must be between ${field.min} and ${field.max}${unit}.`);
        } else if (hasMin) {
          errs.push(`"${label}": ${field.key} must be >= ${field.min}${unit}.`);
        } else {
          errs.push(`"${label}": ${field.key} must be <= ${field.max}${unit}.`);
        }
        numericErrorKeys.add(field.key);
      } else if (field.kind === 'integer' && !Number.isInteger(val)) {
        errs.push(`"${label}": ${field.key} must be an integer.`);
        numericErrorKeys.add(field.key);
      }
      continue;
    }

    if (field.kind === 'select') {
      if (raw !== null && !isPrimitiveParam(raw)) {
        errs.push(`"${label}": ${field.key} must be a string.`);
      } else if (field.options && field.options.length > 0) {
        const str = safeParamStr(raw);
        if (!field.options.some((opt) => opt.value === str)) {
          const allowed = field.options.map((opt) => opt.value).join(', ');
          errs.push(`"${label}": Invalid ${field.key} "${str}" (expected ${allowed}).`);
        }
      }
      continue;
    }

    if (field.kind === 'boolean') {
      if (typeof raw !== 'boolean') {
        errs.push(`"${label}": ${field.key} must be a boolean.`);
      }
      continue;
    }

    if (field.kind === 'aa-set') {
      if (raw !== null && !isPrimitiveParam(raw)) {
        errs.push(`"${label}": ${field.key} must be a string.`);
        continue;
      }
      const str = safeParamStr(raw).trim();
      if (str) {
        const badChars = new Set<string>();
        const canonicalSeen = new Set<string>();
        for (const ch of str.replace(/[\s,;]+/g, '').toUpperCase()) {
          if (!VALID_AA_CHAR_RE.test(ch)) badChars.add(ch);
          else if (ch !== 'X') canonicalSeen.add(ch);
        }
        if (badChars.size > 0) {
          errs.push(
            `"${label}": ${field.key} contains invalid amino acid character(s): ${[...badChars].join(', ')} (use standard 1-letter AA codes e.g. C, M).`,
          );
        } else if (canonicalSeen.size >= 20) {
          errs.push(
            `"${label}": ${field.key} excludes all 20 standard amino acids — leave at least 1 amino acid allowed for design.`,
          );
        }
      }
      continue;
    }

    if (field.kind === 'residue-list') {
      if (raw !== null && !isPrimitiveParam(raw)) {
        errs.push(`"${label}": ${field.key} must be a string.`);
        continue;
      }
      const str = safeParamStr(raw).trim();
      if (str) {
        const allowBareChain = Boolean(field.allowBareChain);
        const allowLigandRef = field.studioRole === 'crop';
        const exampleHint = allowBareChain
          ? '"A", "A10", "A10-15", "A-2", or "A100A"'
          : '"A10", "A10-15", "A-2", or "A100A"';
        for (const tok of str.split(/[,\s]+/).filter(Boolean)) {
          if (!isValidResidueSelectorToken(tok, { allowBareChain, allowLigandRef })) {
            errs.push(
              `"${label}": Invalid ${field.key} token "${tok}" (expected e.g. ${exampleHint}).`,
            );
          }
        }
      }
    }
  }

  // 3. Model-Card-backed mode-aware sampling step envelope check
  const stepCheck = validateStepCount(manifest.id, p);
  if (stepCheck && !stepCheck.ok && stepCheck.message && !numericErrorKeys.has(stepCheck.paramKey)) {
    errs.push(`"${label}": ${stepCheck.message}`);
  }

  return errs;
}
