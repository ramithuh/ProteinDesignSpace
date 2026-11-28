// Layer 3 Plugin helpers — proton-pottsmpnn
// Pure, synchronous input compilation and artifact assembly shared by `proton-pottsmpnn.worker.ts`
// and the unit tests: PDB/mmCIF -> backbone residue list (in the exact order the vendored
// `parseBackbonePDB` emits positions), protonation-override and pH-centre resolution, fixed-position
// and designable-set resolution, the v6 forbidden-token alphabet, SequenceSet / ResidueTrack
// construction, and structure-collection fan-out.
//
// Nothing here touches WebGPU or the vendored runtime, so every branch is testable in Node. The
// constant tables below mirror `static/vendor/proton-pottsmpnn/shared/index.js`; the unit test
// asserts they stay identical to the vendored module.

import { fnv1a } from '$lib/kernel';
import { parseFixedPositionsAndChains } from '$lib/engine/mpnn-spec';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  ResidueTrackArtifact,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

export type ProtonPottsMode = 'design' | 'score';
export type ProtonPottsPrecision = 'fp16' | 'int8' | 'fp32';
export type ProtonPottsDesignScope = 'centre-shell' | 'all-positions';
export type ProtonPottsSweepOrder = 'position' | 'knn' | 'energy';
export type ProtonPottsZScaleMode = 'block' | 'single-mutation';
export type ProtonPottsTitratablePolicy = 'upstream' | 'explicit-states';
export type ProtonPottsCpuFallbackReason = 'unavailable' | 'device-lost';

// ---------------------------------------------------------------------------
// v6 vocabulary (mirrors `shared/index.js`)
// ---------------------------------------------------------------------------

/** `STANDARD_AA` — the 20 parents in the checkpoint's (alphabetical) order. */
export const STANDARD_AA: readonly string[] = [
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
];

/** `V6_PROTONATION_TOKENS` — 9 explicit charge-state microstates. */
export const V6_PROTONATION_TOKENS: readonly string[] = [
  'HIS-P', 'HIS-S', 'HIS-A',
  'ASP-P', 'ASP-D', 'ASP-A',
  'GLU-P', 'GLU-D', 'GLU-A',
];

/** `V6_TOKENS` — the checkpoint's 30-token vocabulary (21 standard incl. UNK + 9 states). */
export const V6_TOKENS: readonly string[] = [...STANDARD_AA, 'UNK', ...V6_PROTONATION_TOKENS];

/** `PHYSIOLOGICAL_STATE` — default state of an unpinned titratable residue at pH ≈ 7.4. */
export const PHYSIOLOGICAL_STATE: Readonly<Record<string, string>> = {
  ASP: 'ASP-D',
  GLU: 'GLU-D',
  HIS: 'HIS-S',
};

/** `V6_DEP_MAP` — protonated microstate -> its deprotonated contrast partners. */
export const V6_DEP_MAP: Readonly<Record<string, readonly string[]>> = {
  'HIS-P': ['HIS-S'],
  'ASP-P': ['ASP-D'],
  'GLU-P': ['GLU-D'],
};

/** `DEFAULT_FORBIDDEN_TOKENS` — microstates never offered as a design choice. */
export const DEFAULT_FORBIDDEN_TOKENS: readonly string[] = ['HIS-A', 'ASP-A', 'GLU-A'];

/** `TITRATABLE_PARENTS` — parents with both a protonated and a deprotonated v6 token. */
export const TITRATABLE_PARENTS: readonly string[] = ['ASP', 'GLU', 'HIS'];

/** `THREE_TO_ONE` restricted to the 20 parents plus UNK. */
export const THREE_TO_ONE: Readonly<Record<string, string>> = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G',
  HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S',
  THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V', UNK: 'X',
};

const ONE_TO_THREE: Readonly<Record<string, string>> = Object.fromEntries(
  Object.entries(THREE_TO_ONE)
    .filter(([three]) => three !== 'UNK')
    .map(([three, one]) => [one, three]),
);

/** The backbone atoms the featurizer needs (CB is virtual, derived from N/CA/C). */
export const BACKBONE_ATOMS: readonly string[] = ['N', 'CA', 'C', 'O'];

/** Canonical parent of a v6 token (`HIS-P` -> `HIS`, `ALA` -> `ALA`). */
export function tokenParent(token: string): string {
  const dash = token.indexOf('-');
  return dash < 0 ? token : token.slice(0, dash);
}

// ---------------------------------------------------------------------------
// Option normalization
// ---------------------------------------------------------------------------

export interface ProtonPottsOptions {
  mode: ProtonPottsMode;
  protonationCentres: string;
  protonationOverrides: string;
  lambda: number;
  numSequences: number;
  temperature: number;
  fixedPositions: string;
  designScope: ProtonPottsDesignScope;
  shellPerCentre: number;
  maxDesignable: number;
  blockSize: number;
  maxRounds: number;
  sweepOrder: ProtonPottsSweepOrder;
  zscaleMode: ProtonPottsZScaleMode;
  omitAAs: string;
  titratableTokens: ProtonPottsTitratablePolicy;
  precision: ProtonPottsPrecision;
  seed: number;
}

const clampNumber = (raw: unknown, fallback: number, min: number, max: number): number => {
  const val = Number(raw ?? fallback);
  if (!Number.isFinite(val)) return fallback;
  return Math.min(max, Math.max(min, val));
};

const clampInt = (raw: unknown, fallback: number, min: number, max: number): number =>
  Math.round(clampNumber(raw, fallback, min, max));

const pickEnum = <T extends string>(raw: unknown, allowed: readonly T[], fallback: T): T => {
  const val = String(raw ?? '').trim().toLowerCase();
  return (allowed as readonly string[]).includes(val) ? (val as T) : fallback;
};

/**
 * Normalize raw node params into the exact option set the worker runs with.
 * Out-of-range numbers are clamped (the declarative schema already rejects them in the UI);
 * unknown enum values fall back to the schema default rather than failing a whole run.
 */
export function protonPottsOptionsFromParams(params: Record<string, unknown>): ProtonPottsOptions {
  return {
    mode: pickEnum(params.mode, ['design', 'score'] as const, 'design'),
    protonationCentres: String(params.protonationCentres ?? '').trim(),
    protonationOverrides: String(params.protonationOverrides ?? '').trim(),
    lambda: clampNumber(params.lambda, 0.5, 0, 1),
    numSequences: clampInt(params.numSequences, 4, 1, 32),
    temperature: clampNumber(params.temperature, 0, 0, 2),
    fixedPositions: String(params.fixedPositions ?? '').trim(),
    designScope: pickEnum(
      params.designScope,
      ['centre-shell', 'all-positions'] as const,
      'centre-shell',
    ),
    shellPerCentre: clampInt(params.shellPerCentre, 8, 1, 32),
    maxDesignable: clampInt(params.maxDesignable, 12, 0, 256),
    blockSize: clampInt(params.blockSize, 2, 1, 3),
    maxRounds: clampInt(params.maxRounds, 10, 1, 50),
    sweepOrder: pickEnum(params.sweepOrder, ['position', 'knn', 'energy'] as const, 'position'),
    zscaleMode: pickEnum(params.zscaleMode, ['block', 'single-mutation'] as const, 'block'),
    omitAAs: sanitizeOmitAAs(params.omitAAs),
    titratableTokens: pickEnum(
      params.titratableTokens,
      ['upstream', 'explicit-states'] as const,
      'upstream',
    ),
    precision: pickEnum(params.precision, ['fp16', 'int8', 'fp32'] as const, 'fp16'),
    seed: clampInt(params.seed, 42, 0, 2_147_483_647),
  };
}

/** Canonical, de-duplicated 1-letter omit set (`X` and separators dropped). */
export function sanitizeOmitAAs(raw: unknown): string {
  const clean = String(raw ?? '')
    .toUpperCase()
    .replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
  return [...new Set(clean)].join('');
}

export interface ProtonPottsAlphabet {
  /** Token names passed to the runtime's `validTokenMask(forbidden)`, in V6_TOKENS order. */
  forbiddenTokens: string[];
  /** Tokens the descent may still propose (UNK is always masked by the runtime). */
  allowedTokens: string[];
  /** 1-letter parents still reachable. */
  designableLetters: string;
}

/**
 * Translate the omit set and the titratable-token policy into the runtime's `forbiddenTokens`.
 *
 * Upstream `validTokenMask()` masks UNK plus the three ambiguous states by default; a caller-passed
 * list *replaces* that default, so `DEFAULT_FORBIDDEN_TOKENS` is always re-included here.
 */
export function resolveProtonPottsAlphabet(
  omitAAs: string,
  titratableTokens: ProtonPottsTitratablePolicy = 'upstream',
): ProtonPottsAlphabet {
  const omitLetters = new Set(sanitizeOmitAAs(omitAAs));
  const omitParents = new Set<string>();
  for (const letter of omitLetters) {
    const parent = ONE_TO_THREE[letter];
    if (parent) omitParents.add(parent);
  }
  const forbidden = new Set<string>(DEFAULT_FORBIDDEN_TOKENS);
  for (const token of V6_TOKENS) {
    if (omitParents.has(tokenParent(token))) forbidden.add(token);
  }
  if (titratableTokens === 'explicit-states') {
    for (const parent of TITRATABLE_PARENTS) forbidden.add(parent);
  }
  const allowedTokens = V6_TOKENS.filter((t) => t !== 'UNK' && !forbidden.has(t));
  if (allowedTokens.length === 0) {
    throw new Error(
      'Proton-PottsMPNN: omitAAs excludes every amino acid, leaving the descent no token to propose.',
    );
  }
  const letters = new Set<string>();
  for (const token of allowedTokens) letters.add(THREE_TO_ONE[tokenParent(token)] ?? 'X');
  return {
    forbiddenTokens: V6_TOKENS.filter((t) => forbidden.has(t)),
    allowedTokens,
    designableLetters: [...STANDARD_AA]
      .map((p) => THREE_TO_ONE[p])
      .filter((l) => letters.has(l))
      .join(''),
  };
}

// ---------------------------------------------------------------------------
// PDB -> backbone residue list
// ---------------------------------------------------------------------------

/** One residue the model sees, in the exact order `parseBackbonePDB` + `buildStructure` emit. */
export interface ProtonPottsResidue {
  /** Selector / display chain: the raw PDB chain character, with blank normalized to `_`. */
  chain: string;
  number: number;
  insertionCode: string;
  resName: string;
  /** Default input token before any override (`PHYSIOLOGICAL_STATE` for titratable parents). */
  defaultToken: string;
}

export interface ProtonPottsInputDiagnostics {
  residueCount: number;
  chainIds: string[];
  /** Titratable residue counts by parent, i.e. how much the missing labeller is guessing. */
  titratable: Record<string, number>;
  /** Residues dropped because a backbone atom was missing (the featurizer needs all four). */
  skippedIncompleteBackbone: number;
  /** Non-standard residues dropped (`MSE`, `HETATM` groups, nucleic acids, water). */
  skippedNonStandard: string[];
  /** Residues carrying a non-finite backbone coordinate (hard error — the Potts tables would be NaN). */
  nonFiniteCoords: number;
  warnings: string[];
}

export interface ProtonPottsInput {
  residues: ProtonPottsResidue[];
  diagnostics: ProtonPottsInputDiagnostics;
}

const num = (text: string): number => Number.parseFloat(text);

/**
 * Minimal fixed-column PDB reader producing the model's backbone residue list.
 *
 * Deliberately a line-for-line mirror of `vendor/proton-pottsmpnn/shared/index.js ::
 * parseBackbonePDB` so position `i` here is position `i` in the tensors the runtime builds:
 * `ATOM` records only, N/CA/C/O only, standard parents only, first altLoc of an atom wins,
 * insertion codes keep residues distinct, and a residue missing any backbone atom is dropped.
 * The worker re-checks `structure.L` and `structure.resSeq` against this list before running.
 *
 * The text is treated purely as data: nothing in it is executed, imported, or evaluated.
 */
export function buildProtonPottsInput(pdbText: string): ProtonPottsInput {
  const standard = new Set(STANDARD_AA);
  interface Raw {
    chain: string;
    number: number;
    insertionCode: string;
    resName: string;
    atoms: Map<string, [number, number, number]>;
  }
  const byKey = new Map<string, Raw>();
  const order: Raw[] = [];
  const nonStandard = new Set<string>();

  for (const line of String(pdbText).split('\n')) {
    if (!line.startsWith('ATOM')) continue;
    const atomName = line.slice(12, 16).trim();
    if (!BACKBONE_ATOMS.includes(atomName)) continue;
    const resName = line.slice(17, 20).trim();
    if (!standard.has(resName)) {
      if (resName) nonStandard.add(resName);
      continue;
    }
    const rawChain = line[21] ?? ' ';
    const number = Number.parseInt(line.slice(22, 26), 10);
    const insertionCode = (line[26] ?? ' ').trim();
    // The vendor keys residues on the raw chain character + resSeq + icode; mirror that exactly.
    const key = `${rawChain}|${number}|${line[26] ?? ' '}`;
    let residue = byKey.get(key);
    if (!residue) {
      residue = {
        chain: rawChain.trim() || '_',
        number,
        insertionCode,
        resName,
        atoms: new Map(),
      };
      byKey.set(key, residue);
      order.push(residue);
    }
    if (residue.atoms.has(atomName)) continue; // first altLoc wins, as upstream
    residue.atoms.set(atomName, [
      num(line.slice(30, 38)),
      num(line.slice(38, 46)),
      num(line.slice(46, 54)),
    ]);
  }

  const residues: ProtonPottsResidue[] = [];
  const chainIds: string[] = [];
  const titratable: Record<string, number> = {};
  let skippedIncompleteBackbone = 0;
  let nonFiniteCoords = 0;

  for (const raw of order) {
    if (!BACKBONE_ATOMS.every((a) => raw.atoms.has(a))) {
      skippedIncompleteBackbone++;
      continue;
    }
    if (
      BACKBONE_ATOMS.some((a) => raw.atoms.get(a)!.some((v) => !Number.isFinite(v)))
    ) {
      nonFiniteCoords++;
    }
    if (!chainIds.includes(raw.chain)) chainIds.push(raw.chain);
    if (TITRATABLE_PARENTS.includes(raw.resName)) {
      titratable[raw.resName] = (titratable[raw.resName] ?? 0) + 1;
    }
    residues.push({
      chain: raw.chain,
      number: raw.number,
      insertionCode: raw.insertionCode,
      resName: raw.resName,
      defaultToken: PHYSIOLOGICAL_STATE[raw.resName] ?? raw.resName,
    });
  }

  if (residues.length === 0) {
    throw new Error(
      'Proton-PottsMPNN: the input structure has no standard amino acid with a complete N/CA/C/O backbone.',
    );
  }
  if (nonFiniteCoords > 0) {
    throw new Error(
      `Proton-PottsMPNN: ${nonFiniteCoords} residue(s) carry a non-finite backbone coordinate; the Potts tables would be NaN.`,
    );
  }

  const warnings: string[] = [];
  if (skippedIncompleteBackbone > 0) {
    warnings.push(
      `${skippedIncompleteBackbone} residue(s) without a complete N/CA/C/O backbone were dropped (the featurizer has no atom stack for them).`,
    );
  }
  if (nonStandard.size > 0) {
    warnings.push(
      `Non-standard residues ${[...nonStandard].sort().join(', ')} were dropped: Proton-PottsMPNN is backbone-only over the 20 standard parents and has no ligand or nucleic-acid pathway.`,
    );
  }
  const titratableTotal = Object.values(titratable).reduce((a, b) => a + b, 0);
  if (titratableTotal > 0) {
    warnings.push(
      `${titratableTotal} titratable residue(s) default to their physiological state (ASP-D / GLU-D / HIS-S); no HBPLUS/propka labeller is bundled, so pin states explicitly with Protonation overrides.`,
    );
  }

  return {
    residues,
    diagnostics: {
      residueCount: residues.length,
      chainIds,
      titratable,
      skippedIncompleteBackbone,
      skippedNonStandard: [...nonStandard].sort(),
      nonFiniteCoords,
      warnings,
    },
  };
}

/** PDB text for one upstream structure, converting mmCIF when that is all we have. */
export function inputPdbText(parent: StructureArtifact): string {
  if (parent.pdbText?.trim()) return parent.pdbText;
  if (parent.cifText?.trim()) return parseStructureText(parent.cifText).pdbText;
  throw new Error(`Proton-PottsMPNN: upstream structure ${parent.id} has no coordinate text.`);
}

/** Insertion-code-safe residue labels (`A1`, `H100A`, …) matching `residues`. */
export function residueIdsOf(residues: readonly ProtonPottsResidue[]): string[] {
  return residues.map((r) => `${r.chain}${r.number}${r.insertionCode}`);
}

/** Contiguous chain layout (ids + lengths) in residue order, for downstream folders. */
export function chainLayoutOf(residues: readonly ProtonPottsResidue[]): {
  chainIds: string[];
  chainLengths: number[];
} {
  const chainIds: string[] = [];
  const chainLengths: number[] = [];
  for (const r of residues) {
    if (chainIds.length === 0 || chainIds[chainIds.length - 1] !== r.chain) {
      chainIds.push(r.chain);
      chainLengths.push(1);
    } else {
      chainLengths[chainLengths.length - 1]++;
    }
  }
  return { chainIds, chainLengths };
}

// ---------------------------------------------------------------------------
// `<residue>=<token>` pinning specs
// ---------------------------------------------------------------------------

export interface ProtonationSpecEntry {
  selector: string;
  token: string;
}

/**
 * Split a `"A42=HIS-P, A58=ASP-P"` style spec into raw entry strings.
 *
 * Whitespace around a separator is folded away first, so `", "`, `";"` and plain spaces all
 * separate entries. `manifest.crossValidate` runs the identical normalization, so the UI never
 * accepts a spec this parser rejects.
 */
export function splitProtonationEntries(raw: unknown): string[] {
  const normalized = String(raw ?? '')
    .trim()
    .replace(/\s*([=:])\s*/g, '$1');
  if (!normalized) return [];
  return normalized.split(/[,;\s]+/).filter(Boolean);
}

/** `"A42=HIS-P"` -> `{ selector: 'A42', token: 'HIS-P' }`, splitting on the LAST `=`/`:`. */
export function parseProtonationEntry(entry: string): { selector: string; token: string } | null {
  // `(.+)` is greedy, so a selector that itself contains a colon (`AA:42`) stays intact.
  const m = /^(.+)[=:]([A-Za-z][A-Za-z-]*)$/.exec(entry);
  if (!m) return null;
  return { selector: m[1], token: m[2].toUpperCase() };
}

/**
 * Parse `"A42=HIS-P, A58=ASP-P"` into `{ selector, token }` entries.
 * Tokens are upper-cased and checked against `allowedTokens`.
 */
export function parseProtonationSpec(
  raw: unknown,
  allowedTokens: readonly string[],
  label = 'protonation spec',
): ProtonationSpecEntry[] {
  const allowed = new Set(allowedTokens);
  const out: ProtonationSpecEntry[] = [];
  for (const entry of splitProtonationEntries(raw)) {
    const parsed = parseProtonationEntry(entry);
    if (!parsed) {
      throw new Error(
        `Proton-PottsMPNN: invalid ${label} entry "${entry}" (expected "<residue>=<token>", e.g. "A42=HIS-P").`,
      );
    }
    if (!allowed.has(parsed.token)) {
      throw new Error(
        `Proton-PottsMPNN: invalid ${label} token "${parsed.token}" (expected one of ${[...allowed].join(', ')}).`,
      );
    }
    out.push(parsed);
  }
  return out;
}

/**
 * Resolve one single-residue selector (`A42`, `A100A`, `AA:42`, `42`) to a 0-based position in
 * `residues`. Chain matching prefers an exact hit and falls back to a unique case-insensitive one.
 */
export function resolveResiduePosition(
  residues: readonly ProtonPottsResidue[],
  selector: string,
): number {
  const tok = selector.trim();
  const m = /^(?:([A-Za-z0-9_]+):|([A-Za-z_]))?(-?\d+)([A-Za-z])?$/.exec(tok);
  if (!m) {
    throw new Error(
      `Proton-PottsMPNN: invalid residue selector "${selector}" (expected e.g. "A42", "A100A", "AA:42" or "42").`,
    );
  }
  const requested = m[1] ?? m[2] ?? residues[0]?.chain ?? 'A';
  const number = Number(m[3]);
  const insertionCode = (m[4] ?? '').toUpperCase();
  const present = new Set(residues.map((r) => r.chain));
  let chain = requested;
  if (!present.has(chain)) {
    const matches = [...present].filter((c) => c.toUpperCase() === requested.toUpperCase());
    if (matches.length === 1) chain = matches[0];
  }
  const idx = residues.findIndex(
    (r) => r.chain === chain && r.number === number && r.insertionCode === insertionCode,
  );
  if (idx < 0) {
    throw new Error(
      `Proton-PottsMPNN: residue "${selector}" is not in the structure (chains ${[...present].join(', ')}).`,
    );
  }
  return idx;
}

export interface ResolvedProtonationOverrides {
  /** `buildStructure({ overrides })` keyed by 0-based position — insertion-code safe. */
  overrides: Record<number, string>;
  entries: Array<{ position: number; residueId: string; resName: string; token: string }>;
  warnings: string[];
}

/**
 * Resolve the `protonationOverrides` control into positional `buildStructure` overrides.
 *
 * Positional keys are used rather than the vendor's `"<chain><resSeq>"` labels because those
 * labels collapse insertion codes onto one residue; positions never do.
 */
export function resolveProtonationOverrides(
  residues: readonly ProtonPottsResidue[],
  spec: string,
): ResolvedProtonationOverrides {
  const entries = parseProtonationSpec(spec, V6_TOKENS, 'protonationOverrides');
  const overrides: Record<number, string> = {};
  const resolved: ResolvedProtonationOverrides['entries'] = [];
  const warnings: string[] = [];
  const ids = residueIdsOf(residues);
  for (const entry of entries) {
    const position = resolveResiduePosition(residues, entry.selector);
    const residue = residues[position];
    if (overrides[position] !== undefined && overrides[position] !== entry.token) {
      warnings.push(
        `Protonation override for ${ids[position]} was given twice (${overrides[position]} then ${entry.token}); the last one wins.`,
      );
    }
    overrides[position] = entry.token;
    const parent = tokenParent(entry.token);
    if (parent !== residue.resName) {
      warnings.push(
        `Override ${ids[position]}=${entry.token} replaces a ${residue.resName} input token: the encoder is sequence-blind, so this scores the backbone as if that residue were ${parent}.`,
      );
    }
    resolved.push({
      position,
      residueId: ids[position],
      resName: residue.resName,
      token: entry.token,
    });
  }
  return { overrides, entries: resolved, warnings };
}

export interface ResolvedCentre {
  position: number;
  residueId: string;
  resName: string;
  protonationType: string;
  contrastTokens: string[];
}

export interface ResolvedCentres {
  centres: ResolvedCentre[];
  /** `resolveCentres(structure, specs)` input: positional specs, insertion-code safe. */
  specs: Array<{ position: number; protonationType: string }>;
  warnings: string[];
}

/** Resolve the `protonationCentres` control into positional centre specs for `resolveCentres`. */
export function resolveProtonPottsCentres(
  residues: readonly ProtonPottsResidue[],
  spec: string,
): ResolvedCentres {
  const entries = parseProtonationSpec(spec, Object.keys(V6_DEP_MAP), 'protonationCentres');
  const centres: ResolvedCentre[] = [];
  const warnings: string[] = [];
  const seen = new Set<number>();
  const ids = residueIdsOf(residues);
  for (const entry of entries) {
    const position = resolveResiduePosition(residues, entry.selector);
    if (seen.has(position)) {
      warnings.push(`pH centre ${ids[position]} was listed twice; the duplicate was dropped.`);
      continue;
    }
    seen.add(position);
    const residue = residues[position];
    const parent = tokenParent(entry.token);
    if (parent !== residue.resName) {
      warnings.push(
        `pH centre ${ids[position]}=${entry.token} sits on a ${residue.resName}: it is pinned ${entry.token} anyway, so the gap is measured for a ${parent} introduced at that position.`,
      );
    }
    centres.push({
      position,
      residueId: ids[position],
      resName: residue.resName,
      protonationType: entry.token,
      contrastTokens: [...(V6_DEP_MAP[entry.token] ?? [])],
    });
  }
  return {
    centres,
    specs: centres.map((c) => ({ position: c.position, protonationType: c.protonationType })),
    warnings,
  };
}

// ---------------------------------------------------------------------------
// Fixed positions & the designable set
// ---------------------------------------------------------------------------

export interface ProtonPottsConstraints {
  /** Indices into `residues` held at their input token. */
  fixedPositions: Set<number>;
  /** Chain IDs locked in full (propagated downstream on the SequenceSet). */
  lockedChains: string[];
}

/**
 * Resolve the user `fixedPositions` selector plus any locks inherited from the upstream generator
 * (target chains, motif residues) into residue indices, reusing the shared selector compiler so
 * the syntax is identical to every other designer in the app.
 */
export function resolveProtonPottsConstraints(
  residues: readonly ProtonPottsResidue[],
  spec: string,
  parent?: Pick<StructureArtifact, 'lockedChains' | 'fixedResidueIndices'>,
): ProtonPottsConstraints {
  const ordered = residues.map((r) => ({
    chain: r.chain,
    number: r.number,
    insertionCode: r.insertionCode,
  }));
  const presentChains = new Set(residues.map((r) => r.chain));
  const upperToChain = new Map<string, string>();
  for (const ch of presentChains) {
    if (!upperToChain.has(ch.toUpperCase())) upperToChain.set(ch.toUpperCase(), ch);
  }
  const defaultChain = residues[0]?.chain ?? 'A';

  const { fixedPositions: refs, lockedChains: rawLocked } = parseFixedPositionsAndChains(
    spec,
    defaultChain,
    ordered,
  );

  const fixedPositions = new Set<number>();
  const lockedChains = new Set<string>();
  const resolveChain = (requested: string): string =>
    presentChains.has(requested) ? requested : upperToChain.get(requested.toUpperCase()) ?? requested;

  const lockChain = (requested: string): void => {
    const ch = resolveChain(requested);
    if (!presentChains.has(ch)) return;
    lockedChains.add(ch);
    residues.forEach((r, idx) => {
      if (r.chain === ch) fixedPositions.add(idx);
    });
  };

  for (const ch of rawLocked) lockChain(ch);
  for (const ch of parent?.lockedChains ?? []) lockChain(String(ch).trim());

  for (const ref of refs) {
    if (typeof ref === 'number') {
      if (ref >= 0 && ref < residues.length) fixedPositions.add(ref);
      continue;
    }
    const ch = resolveChain(ref.chain);
    const ins = ref.insertionCode ?? '';
    residues.forEach((r, idx) => {
      if (r.chain === ch && r.number === ref.number && r.insertionCode === ins) {
        fixedPositions.add(idx);
      }
    });
  }

  // Motif residues propagated from upstream generators are 1-based residue ordinals.
  for (const oneBased of parent?.fixedResidueIndices ?? []) {
    const idx = Number(oneBased) - 1;
    if (Number.isInteger(idx) && idx >= 0 && idx < residues.length) fixedPositions.add(idx);
  }

  return { fixedPositions, lockedChains: [...lockedChains] };
}

export interface DesignableSetOptions {
  /** Scope-derived candidate positions: the vendored centre shell, or every residue. */
  candidates: Iterable<number>;
  fixed: ReadonlySet<number>;
  /** Centre positions, always excluded — the descent pins them as fixed context. */
  centres: Iterable<number>;
  /** `maxDesignable`; 0 removes the cap. */
  max: number;
  /**
   * Optional ranking hook used when the cap bites. The worker passes the vendored
   * `knnRank(tables, centres, positions)` so the cap keeps the positions most coupled to a centre;
   * without it the cap keeps the lowest sequence positions.
   */
  rank?: (positions: readonly number[]) => Iterable<number>;
}

/**
 * The designable set handed to `blockDescent({ designable })`: scope candidates minus fixed
 * positions minus centres, capped last so a cap is never spent on an excluded position.
 */
export function resolveDesignableSet(opts: DesignableSetOptions): number[] {
  const excluded = new Set<number>(opts.centres);
  for (const p of opts.fixed) excluded.add(p);
  const kept = [...new Set([...opts.candidates])]
    .filter((p) => Number.isInteger(p) && !excluded.has(p))
    .sort((a, b) => a - b);
  if (opts.max <= 0 || kept.length <= opts.max) return kept;
  const keptSet = new Set(kept);
  const ranked = opts.rank ? [...opts.rank(kept)].filter((p) => keptSet.has(p)) : kept;
  return ranked.slice(0, opts.max).sort((a, b) => a - b);
}

// ---------------------------------------------------------------------------
// Scores, samples & artifact assembly
// ---------------------------------------------------------------------------

export interface ProtonPottsCentreReport {
  position: number;
  residueId: string;
  resName: string;
  protonationType: string;
  contrastTokens: string[];
  /** `e_P`: the centre's conditional energy held protonated. */
  energyProtonated: number;
  /** `mean_d e_D`: the mean over its deprotonated contrasts. */
  energyDeprotonated: number;
  /** `e_P − mean_d e_D` — lower favours the protonated microstate. */
  gap: number;
}

export interface ProtonPottsScore {
  /** `E_global = H(s)`, the reference Hamiltonian (double-counting convention included). */
  globalEnergy: number;
  /** `E_global / L`, so designs of different length stay comparable. */
  energyPerResidue: number;
  /** `Σ_centres (E_P − E_D)`, or null when no centre is declared. */
  selectiveGapSum: number | null;
  centres: ProtonPottsCentreReport[];
  /** Per-residue conditional (site) energy `E[p, s_p]`, in residue order. */
  perResidueEnergy: Float32Array;
}

/** The vocabulary indices `resolveCentres()` attaches to each centre. */
export interface VendorCentreIndices {
  position: number;
  protonationType: string;
  protIdx: number;
  depIdxs: readonly number[];
}

/** A copy of `seq` with every centre pinned to its protonated microstate, as the descent does. */
export function pinCentres(
  seq: ArrayLike<number>,
  centres: readonly VendorCentreIndices[],
): Int32Array {
  const pinned = Int32Array.from(seq as ArrayLike<number>);
  for (const c of centres) pinned[c.position] = c.protIdx;
  return pinned;
}

/**
 * Split each centre's selective gap into its two halves for display.
 *
 * `rowAt(position)` must return the conditional energies `[V]` at that position with every centre
 * pinned protonated — i.e. `pottsCandidateEnergiesAt(tables, pinCentres(seq, centres), position)`,
 * the exact row `shared :: selectiveGapOf` reads — so `gap` here equals the runtime's own
 * `selectiveReport` gap bit for bit.
 */
export function buildCentreReports(
  meta: readonly ResolvedCentre[],
  centres: readonly VendorCentreIndices[],
  rowAt: (position: number) => ArrayLike<number>,
): ProtonPottsCentreReport[] {
  return centres.map((c, i) => {
    const row = rowAt(c.position);
    const energyProtonated = row[c.protIdx];
    let sum = 0;
    for (const d of c.depIdxs) sum += row[d];
    const energyDeprotonated = c.depIdxs.length > 0 ? sum / c.depIdxs.length : Number.NaN;
    return {
      position: c.position,
      residueId: meta[i]?.residueId ?? String(c.position),
      resName: meta[i]?.resName ?? '',
      protonationType: c.protonationType,
      contrastTokens: [...(meta[i]?.contrastTokens ?? [])],
      energyProtonated,
      energyDeprotonated,
      gap: energyProtonated - energyDeprotonated,
    };
  });
}

/** Per-residue conditional (site) energy `E[p, s_p]` pulled out of an `[L, V]` candidate table. */
export function siteEnergies(
  candidateEnergies: ArrayLike<number>,
  seq: ArrayLike<number>,
  L: number,
  V: number,
): Float32Array {
  const values = new Float32Array(L);
  for (let p = 0; p < L; p++) values[p] = candidateEnergies[p * V + seq[p]];
  return values;
}

export interface ProtonPottsSample {
  sequence: string;
  tokenNames: string[];
  score: ProtonPottsScore;
  sampleIndex: number;
  seed: number;
  /** Sweeps run / early-stop flag / committed block moves — null in score mode. */
  rounds: number | null;
  converged: boolean | null;
  nChanges: number | null;
  /** The frozen-z objective the descent minimized, or null in score mode. */
  objective: number | null;
  /** Positions whose token differs from the (override-applied) input token. */
  mutations: number;
  chainIds: string[];
  chainLengths: number[];
}

/**
 * The value block descent actually minimizes, evaluated on a finished design:
 * `(1−λ)/sdH · E_global + λ/sdSel · Σ(E_P − E_D)`.
 *
 * `wH`/`wSel` are the run's own frozen weights (`blockDescent(...).weights`), so this is comparable
 * across the candidates of one job — they share tables, start sequence and designable set, hence
 * identical z-scales — and is what the candidate ordering uses.
 */
export function protonPottsObjective(
  energies: { pottsEnergy: number; selectiveEnergy: number },
  weights: { wH: number; wSel: number },
): number {
  return weights.wH * energies.pottsEnergy + weights.wSel * energies.selectiveEnergy;
}

const SCORE_PRECISION = 1e6;
const roundScore = (v: number): number =>
  Number.isFinite(v) ? Math.round(v * SCORE_PRECISION) / SCORE_PRECISION : 0;

/**
 * Build one SequenceCandidate.
 *
 * `score` is the per-residue Potts energy (lower = more stable, and lower sorts better everywhere
 * in the app); `globalScore` carries the selective gap sum `ΔE` (lower = more pH-selective) and is
 * null when the node declares no pH centre. The emitted *order* is the descent objective — see
 * `orderProtonPottsCandidates`.
 */
export function buildProtonPottsCandidate(
  sample: ProtonPottsSample,
  parent: StructureArtifact,
  options: ProtonPottsOptions,
  idPrefix = 'proton-pottsmpnn',
  extra: Partial<SequenceCandidate> = {},
): SequenceCandidate {
  return {
    id: `${idPrefix}-${sample.sampleIndex + 1}`,
    sourceStructureId: parent.id,
    sampleIndex: sample.sampleIndex,
    sequence: sample.sequence,
    score: roundScore(sample.score.energyPerResidue),
    globalScore:
      sample.score.selectiveGapSum === null ? null : roundScore(sample.score.selectiveGapSum),
    sourceModelId: 'proton-pottsmpnn',
    temperature: options.temperature,
    seed: sample.seed,
    chainLengths: [...sample.chainLengths],
    chainIds: [...sample.chainIds],
    ...extra,
  };
}

/** Candidate order: the frozen-z descent objective, ties broken by sample index. */
export function orderProtonPottsCandidates<T extends { objective: number | null; sampleIndex: number }>(
  samples: readonly T[],
): T[] {
  return [...samples].sort((a, b) => {
    const av = a.objective === null || !Number.isFinite(a.objective) ? Number.POSITIVE_INFINITY : a.objective;
    const bv = b.objective === null || !Number.isFinite(b.objective) ? Number.POSITIVE_INFINITY : b.objective;
    return av !== bv ? av - bv : a.sampleIndex - b.sampleIndex;
  });
}

export function protonPottsRunHash(
  parent: StructureArtifact,
  options: ProtonPottsOptions,
  fixedPositions: ReadonlySet<number>,
  designable: readonly number[],
): string {
  return fnv1a(
    JSON.stringify({
      structureId: parent.id,
      ...options,
      fixed: [...fixedPositions].sort((a, b) => a - b),
      designable: [...designable],
    }),
  );
}

export function buildProtonPottsSequenceSet(
  parent: StructureArtifact,
  candidates: SequenceCandidate[],
  options: ProtonPottsOptions,
  lockedChains: readonly string[],
  runHash: string,
  extra: Partial<SequenceSet> = {},
): SequenceSet {
  return {
    kind: 'SequenceSet',
    id: `proton-pottsmpnn-seq-${parent.id}-${options.seed}-${runHash}`,
    parentStructureId: parent.id,
    sourceStructureId: parent.id,
    sourceModelId: 'proton-pottsmpnn',
    ...(lockedChains.length > 0 ? { lockedChains: [...lockedChains] } : {}),
    ...(parent.chainRoles ? { chainRoles: { ...parent.chainRoles } } : {}),
    candidates,
    ...extra,
  };
}

/**
 * Per-residue Potts site-energy track for the best (design) or native (score) sequence:
 * every term of `H` that touches that position, with the position at its own token.
 */
export function buildProtonPottsTrack(
  parent: StructureArtifact,
  residues: readonly ProtonPottsResidue[],
  sample: ProtonPottsSample,
  options: ProtonPottsOptions,
  runHash: string,
): ResidueTrackArtifact {
  const values = sample.score.perResidueEnergy;
  let sum = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let counted = 0;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
    counted++;
  }
  const mean = counted > 0 ? sum / counted : Number.NaN;
  const label =
    options.mode === 'score'
      ? `Proton-PottsMPNN site energy (native · mean ${mean.toFixed(3)})`
      : `Proton-PottsMPNN site energy (best design · mean ${mean.toFixed(3)})`;
  return {
    kind: 'ResidueTrack',
    id: `proton-pottsmpnn-track-${options.mode}-${options.precision}-${runHash}`,
    label,
    sourceModelId: 'proton-pottsmpnn',
    parentStructureId: parent.id,
    quantity: 'potts-site-energy',
    unit: 'a.u.',
    residues: sample.sequence,
    residueIds: residueIdsOf(residues),
    values,
    summary: {
      mean,
      max: counted > 0 ? max : Number.NaN,
      min: counted > 0 ? min : Number.NaN,
    },
  };
}

/** One-line human summary of a centre report, reused by worker progress lines and tests. */
export function formatCentreReport(centre: ProtonPottsCentreReport): string {
  return `${centre.residueId} ${centre.protonationType} ΔE ${centre.gap.toFixed(3)}`;
}

// ---------------------------------------------------------------------------
// Structure-collection fan-out
// ---------------------------------------------------------------------------

export function structureItems(structure: StructureArtifact): StructureArtifact[] {
  const items = structure.collection?.items;
  return items && items.length > 1 ? items : [structure];
}

export interface ProtonPottsJob {
  parent: StructureArtifact;
  designKey?: string;
  designIndex: number;
}

/** One job per upstream backbone, keeping the collection's design keys and ordinals. */
export function planProtonPottsJobs(structure: StructureArtifact): ProtonPottsJob[] {
  const items = structureItems(structure);
  if (items.length === 1) return [{ parent: items[0], designIndex: 0 }];
  return items.map((parent, idx) => {
    const designKey = parent.designKey ?? structure.collection?.keys?.[idx] ?? parent.id;
    return {
      parent,
      designKey,
      designIndex: typeof parent.designIndex === 'number' ? parent.designIndex : idx,
    };
  });
}
