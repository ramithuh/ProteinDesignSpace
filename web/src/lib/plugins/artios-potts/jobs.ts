// Layer 3 Plugin helpers — artios-potts
// Pure, synchronous input compilation and artifact assembly shared by `artios-potts.worker.ts`
// and the unit tests: PDB/mmCIF -> ArtiosPotts featurizer spec (protein tokens + context atom
// tokens for ligands / cofactors / metals / nucleic acids), fixed-position and ligand-proximity
// masks, SequenceSet / ResidueTrack construction, and structure-collection fan-out.
//
// Nothing here touches WebGPU or the vendored runtime, so every branch is testable in Node.

import { fnv1a } from '$lib/kernel';
import { parseFixedPositionsAndChains } from '$lib/engine/mpnn-spec';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  ResidueTrackArtifact,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

export type ArtiosPottsMode = 'design' | 'score';
export type ArtiosPottsSampler = 'gibbs' | 'dlmc';
export type ArtiosPottsPrecision = 'fp16' | 'int8' | 'fp32';
export type ArtiosPottsCpuFallbackReason = 'unavailable' | 'device-lost';

/** `artios.common.residue_names.PROTEIN_TYPES` — mirrors `vendor/artios-potts/shared/constants.js`. */
export const PROTEIN_RESIDUE_NAMES: readonly string[] = [
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
];

/** `PROTEIN_BACKBONE` (slot 4 is the virtual CB the featurizer derives from N/CA/C). */
export const PROTEIN_BACKBONE_ATOMS: readonly string[] = ['N', 'CA', 'C', 'O'];

/** Nucleic-acid residue names recognised in ATOM records and routed into the context-atom pathway. */
export const NUCLEIC_RESIDUE_NAMES: readonly string[] = [
  'A', 'G', 'C', 'U', 'DA', 'DG', 'DC', 'DT', 'N', 'DN',
  // Common legacy / variant spellings.
  'ADE', 'GUA', 'CYT', 'URA', 'THY', 'RA', 'RG', 'RC', 'RU',
];

const WATER_RESIDUE_NAMES: readonly string[] = ['HOH', 'DOD', 'WAT', 'H2O', 'TIP', 'TIP3', 'SOL'];

export interface ArtiosPottsOptions {
  mode: ArtiosPottsMode;
  numSequences: number;
  sampler: ArtiosPottsSampler;
  temperature: number;
  temperatureInit: number;
  numSteps: number;
  cfgScale: number;
  ligandCutoff: number;
  omitAAs: string;
  complexityWeight: number;
  fixedPositions: string;
  seed: number;
  precision: ArtiosPottsPrecision;
}

const clampNumber = (raw: unknown, fallback: number, min: number, max: number): number => {
  const val = Number(raw ?? fallback);
  if (!Number.isFinite(val)) return fallback;
  return Math.min(max, Math.max(min, val));
};

const clampInt = (raw: unknown, fallback: number, min: number, max: number): number =>
  Math.round(clampNumber(raw, fallback, min, max));

/**
 * Normalize raw node params into the exact option set the worker runs with.
 * Out-of-range numbers are clamped (the declarative schema already rejects them in the UI);
 * unknown enum values fall back to the upstream default rather than failing a whole run.
 */
export function artiosPottsOptionsFromParams(params: Record<string, unknown>): ArtiosPottsOptions {
  const rawMode = String(params.mode ?? 'design').trim().toLowerCase();
  const mode: ArtiosPottsMode = rawMode === 'score' ? 'score' : 'design';
  const rawSampler = String(params.sampler ?? 'gibbs').trim().toLowerCase();
  const sampler: ArtiosPottsSampler = rawSampler === 'dlmc' ? 'dlmc' : 'gibbs';
  const rawPrecision = String(params.precision ?? 'fp16').trim().toLowerCase();
  const precision: ArtiosPottsPrecision =
    rawPrecision === 'int8' || rawPrecision === 'fp32' ? rawPrecision : 'fp16';
  return {
    mode,
    numSequences: clampInt(params.numSequences, 8, 1, 64),
    sampler,
    temperature: clampNumber(params.temperature, 0.1, 0.01, 2.0),
    temperatureInit: clampNumber(params.temperatureInit, 1.0, 0.01, 5.0),
    numSteps: clampInt(params.numSteps, 150, 10, 500),
    cfgScale: clampNumber(params.cfgScale, 0, 0, 5.0),
    ligandCutoff: clampNumber(params.ligandCutoff, 0, 0, 20),
    // An absent key takes the upstream Cys/Met exclusion; an explicit empty string omits nothing.
    omitAAs: params.omitAAs === undefined || params.omitAAs === null
      ? 'CM'
      : sanitizeOmitAAs(params.omitAAs),
    complexityWeight: clampNumber(params.complexityWeight, 0.6, 0, 5.0),
    fixedPositions: String(params.fixedPositions ?? '').trim(),
    seed: clampInt(params.seed, 42, 0, 2_147_483_647),
    precision,
  };
}

/** Canonical, de-duplicated 1-letter omit set (`X` and separators dropped). */
export function sanitizeOmitAAs(raw: unknown): string {
  const clean = String(raw ?? '')
    .toUpperCase()
    .replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
  return [...new Set(clean)].join('');
}

// ---------------------------------------------------------------------------
// PDB -> featurizer spec
// ---------------------------------------------------------------------------

export interface PottsSpecAtom {
  name: string;
  element: string;
  charge: number;
  coord: [number, number, number];
}

export interface PottsSpecProteinResidue {
  resName: string;
  resSeq: number;
  N?: [number, number, number];
  CA?: [number, number, number];
  C?: [number, number, number];
  O?: [number, number, number];
}

export interface PottsSpecContextGroup {
  resName: string;
  resSeq: number;
  atoms: PottsSpecAtom[];
}

export type PottsSpecChain =
  | { id: string; kind: 'protein'; residues: PottsSpecProteinResidue[] }
  | { id: string; kind: 'ligand'; residues: PottsSpecContextGroup[] };

export interface PottsSpec {
  name: string;
  chains: PottsSpecChain[];
}

/** One designable protein residue, in the exact order `featurize()` emits protein tokens. */
export interface ArtiosPottsResidue {
  chain: string;
  number: number;
  insertionCode: string;
  resName: string;
}

export interface ArtiosPottsInputDiagnostics {
  proteinResidueCount: number;
  contextAtomCount: number;
  contextGroupNames: string[];
  nucleicAcidChains: string[];
  /** Residues skipped because they carry no Cα (the featurizer has no centre atom for them). */
  skippedNoCa: number;
  warnings: string[];
}

export interface ArtiosPottsInput {
  spec: PottsSpec;
  proteinResidues: ArtiosPottsResidue[];
  diagnostics: ArtiosPottsInputDiagnostics;
}

const num = (text: string): number => Number.parseFloat(text);

/**
 * Minimal fixed-column PDB reader producing an ArtiosPotts featurizer spec.
 *
 * Mirrors `vendor/artios-potts/shared/featurize.js :: parsePdbSpec` for proteins and HETATM
 * groups, and additionally (a) keeps insertion codes as distinct residues instead of collapsing
 * them onto one `resSeq` key, and (b) routes nucleic-acid ATOM records into the same context-atom
 * pathway as ligands, so DNA/RNA chains condition the design instead of being dropped.
 *
 * The text is treated purely as data: nothing in it is executed, imported, or evaluated.
 */
export function buildArtiosPottsInput(pdbText: string, name = 'structure'): ArtiosPottsInput {
  const proteinNames = new Set(PROTEIN_RESIDUE_NAMES);
  const nucleicNames = new Set(NUCLEIC_RESIDUE_NAMES);
  const waters = new Set(WATER_RESIDUE_NAMES);

  interface RawProtein extends PottsSpecProteinResidue {
    chain: string;
    insCode: string;
  }
  const protein = new Map<string, RawProtein>();
  const proteinOrder: RawProtein[] = [];
  const context = new Map<string, PottsSpecContextGroup & { chain: string; isNucleic: boolean }>();
  const contextOrder: Array<PottsSpecContextGroup & { chain: string; isNucleic: boolean }> = [];

  for (const line of String(pdbText).split('\n')) {
    const record = line.slice(0, 6).trim();
    if (record !== 'ATOM' && record !== 'HETATM') continue;
    // Keep the unlabelled conformer and altLoc A only, as the reference parser does.
    const altLoc = line[16];
    if (altLoc !== undefined && altLoc !== ' ' && altLoc !== 'A') continue;
    const atomName = line.slice(12, 16).trim();
    const resName = line.slice(17, 20).trim().toUpperCase();
    const chainId = line[21]?.trim() || 'A';
    const resSeq = Number.parseInt(line.slice(22, 26), 10);
    const insCode = (line[26] ?? ' ').trim();
    const x = num(line.slice(30, 38));
    const y = num(line.slice(38, 46));
    const z = num(line.slice(46, 54));
    if (!Number.isFinite(resSeq) || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      continue;
    }
    const element = (line.slice(76, 78).trim() || atomName[0] || '').toUpperCase();
    if (!element || element === 'H' || element === 'D') continue;

    const isProteinResidue = record === 'ATOM' && proteinNames.has(resName);
    const isNucleicResidue = nucleicNames.has(resName) && !proteinNames.has(resName);

    if (isProteinResidue) {
      const key = `${chainId}|${resSeq}|${insCode}`;
      let residue = protein.get(key);
      if (!residue) {
        residue = { chain: chainId, insCode, resName, resSeq };
        protein.set(key, residue);
        proteinOrder.push(residue);
      }
      if (PROTEIN_BACKBONE_ATOMS.includes(atomName) && !residue[atomName as 'N']) {
        residue[atomName as 'N'] = [x, y, z];
      }
      continue;
    }

    if (waters.has(resName)) continue;
    // Everything else — HETATM cofactors, metals, modified residues, plus DNA/RNA ATOM records —
    // becomes a per-atom context token.
    const key = `${chainId}|${resName}|${resSeq}|${insCode}`;
    let group = context.get(key);
    if (!group) {
      group = { chain: chainId, resName, resSeq, atoms: [], isNucleic: isNucleicResidue };
      context.set(key, group);
      contextOrder.push(group);
    }
    group.atoms.push({ name: atomName, element, charge: 0, coord: [x, y, z] });
  }

  const warnings: string[] = [];
  let skippedNoCa = 0;
  const byChain = new Map<string, PottsSpecProteinResidue[]>();
  const residueOfChain = new Map<string, RawProtein[]>();
  for (const residue of proteinOrder) {
    if (!residue.CA) {
      skippedNoCa++;
      continue;
    }
    if (!byChain.has(residue.chain)) {
      byChain.set(residue.chain, []);
      residueOfChain.set(residue.chain, []);
    }
    byChain.get(residue.chain)!.push({
      resName: residue.resName,
      resSeq: residue.resSeq,
      ...(residue.N ? { N: residue.N } : {}),
      ...(residue.CA ? { CA: residue.CA } : {}),
      ...(residue.C ? { C: residue.C } : {}),
      ...(residue.O ? { O: residue.O } : {}),
    });
    residueOfChain.get(residue.chain)!.push(residue);
  }
  if (skippedNoCa > 0) {
    warnings.push(`${skippedNoCa} residue(s) without a Cα were dropped (no centre atom).`);
  }

  const chains: PottsSpecChain[] = [];
  const proteinResidues: ArtiosPottsResidue[] = [];
  for (const [id, residues] of byChain) {
    chains.push({ id, kind: 'protein', residues });
    for (const raw of residueOfChain.get(id)!) {
      proteinResidues.push({
        chain: raw.chain,
        number: raw.resSeq,
        insertionCode: raw.insCode,
        resName: raw.resName,
      });
    }
  }

  const contextGroupNames = new Set<string>();
  const nucleicAcidChains = new Set<string>();
  let contextAtomCount = 0;
  for (const group of contextOrder) {
    chains.push({
      id: group.chain,
      kind: 'ligand',
      residues: [{ resName: group.resName, resSeq: group.resSeq, atoms: group.atoms }],
    });
    contextGroupNames.add(group.resName);
    contextAtomCount += group.atoms.length;
    if (group.isNucleic) nucleicAcidChains.add(group.chain);
  }
  if (nucleicAcidChains.size > 0) {
    warnings.push(
      `DNA/RNA chain(s) ${[...nucleicAcidChains].join(', ')} enter as generic context atoms (ArtiosPotts has no dedicated nucleic entity embedding in this port).`,
    );
  }
  if (proteinResidues.length === 0) {
    throw new Error('ArtiosPotts: the input structure has no protein residue with a Cα atom.');
  }

  return {
    spec: { name, chains },
    proteinResidues,
    diagnostics: {
      proteinResidueCount: proteinResidues.length,
      contextAtomCount,
      contextGroupNames: [...contextGroupNames],
      nucleicAcidChains: [...nucleicAcidChains],
      skippedNoCa,
      warnings,
    },
  };
}

/** PDB text for one upstream structure, converting mmCIF when that is all we have. */
export function inputPdbText(parent: StructureArtifact): string {
  if (parent.pdbText?.trim()) return parent.pdbText;
  if (parent.cifText?.trim()) return parseStructureText(parent.cifText).pdbText;
  throw new Error(`ArtiosPotts: upstream structure ${parent.id} has no coordinate text.`);
}

// ---------------------------------------------------------------------------
// Masks
// ---------------------------------------------------------------------------

export interface ArtiosPottsConstraints {
  /** Indices into `proteinResidues` held at their input identity. */
  fixedProteinIndices: Set<number>;
  /** Chain IDs locked in full (propagated downstream on the SequenceSet). */
  lockedChains: string[];
}

/**
 * Resolve the user `fixedPositions` selector plus any locks inherited from the upstream
 * generator (target chains, motif residues) into protein-residue indices.
 *
 * Mirrors `ligandmpnn.worker.ts :: buildLigandConstraints`, reusing the shared
 * `parseFixedPositionsAndChains` compiler so the selector syntax is identical everywhere.
 */
export function resolveArtiosPottsConstraints(
  proteinResidues: readonly ArtiosPottsResidue[],
  spec: string,
  parent?: Pick<StructureArtifact, 'lockedChains' | 'fixedResidueIndices'>,
): ArtiosPottsConstraints {
  const ordered = proteinResidues.map((r) => ({
    chain: r.chain,
    number: r.number,
    insertionCode: r.insertionCode,
  }));
  const presentChains = new Set(proteinResidues.map((r) => r.chain));
  const upperToChain = new Map<string, string>();
  for (const ch of presentChains) {
    if (!upperToChain.has(ch.toUpperCase())) upperToChain.set(ch.toUpperCase(), ch);
  }
  const defaultChain = proteinResidues[0]?.chain ?? 'A';

  const { fixedPositions: refs, lockedChains: rawLocked } = parseFixedPositionsAndChains(
    spec,
    defaultChain,
    ordered,
  );

  const fixedProteinIndices = new Set<number>();
  const lockedChains = new Set<string>();
  const resolveChain = (requested: string): string =>
    presentChains.has(requested) ? requested : upperToChain.get(requested.toUpperCase()) ?? requested;

  const lockChain = (requested: string): void => {
    const ch = resolveChain(requested);
    if (!presentChains.has(ch)) return;
    lockedChains.add(ch);
    proteinResidues.forEach((r, idx) => {
      if (r.chain === ch) fixedProteinIndices.add(idx);
    });
  };

  for (const ch of rawLocked) lockChain(ch);
  for (const ch of parent?.lockedChains ?? []) lockChain(String(ch).trim());

  for (const ref of refs) {
    if (typeof ref === 'number') {
      if (ref >= 0 && ref < proteinResidues.length) fixedProteinIndices.add(ref);
      continue;
    }
    const ch = resolveChain(ref.chain);
    const ins = ref.insertionCode ?? '';
    proteinResidues.forEach((r, idx) => {
      if (r.chain === ch && r.number === ref.number && (r.insertionCode || '') === ins) {
        fixedProteinIndices.add(idx);
      }
    });
  }

  // Motif residues propagated from upstream generators are 1-based residue ordinals.
  for (const oneBased of parent?.fixedResidueIndices ?? []) {
    const idx = Number(oneBased) - 1;
    if (Number.isInteger(idx) && idx >= 0 && idx < proteinResidues.length) {
      fixedProteinIndices.add(idx);
    }
  }

  return { fixedProteinIndices, lockedChains: [...lockedChains] };
}

/** The minimum featurizer-context shape the mask helpers read. */
export interface PottsContextLike {
  nToken: number;
  tokenMask: ArrayLike<number>;
  isProteinToken: ArrayLike<number>;
}

/**
 * Token-level fixed mask: user/inherited locks plus, when `proximityMask` is supplied,
 * every protein residue outside the ligand cutoff. `sampleFromParams` subtracts this from
 * its design mask, so fixed positions keep their input identity for the whole run.
 */
export function buildFixedTokenMask(
  context: PottsContextLike,
  fixedProteinIndices: ReadonlySet<number>,
  proximityMask: ArrayLike<number> | null = null,
): Uint8Array {
  const fixed = new Uint8Array(context.nToken);
  let proteinOrdinal = 0;
  for (let i = 0; i < context.nToken; i++) {
    if (!context.isProteinToken[i]) continue;
    const ordinal = proteinOrdinal++;
    if (fixedProteinIndices.has(ordinal)) fixed[i] = 1;
    else if (proximityMask && !proximityMask[i]) fixed[i] = 1;
  }
  return fixed;
}

/** Number of protein tokens left designable after masking (0 means the run would be a no-op). */
export function countDesignablePositions(
  context: PottsContextLike,
  fixedMask: ArrayLike<number>,
): number {
  let n = 0;
  for (let i = 0; i < context.nToken; i++) {
    if (context.tokenMask[i] && context.isProteinToken[i] && !fixedMask[i]) n++;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Sequence & artifact assembly
// ---------------------------------------------------------------------------

export interface FlattenedSequence {
  sequence: string;
  chainIds: string[];
  chainLengths: number[];
}

/**
 * Flatten the vendored `sequencesByChain()` map into one token-ordered string plus the
 * chain layout downstream folders need, using the spec's chain order.
 */
export function flattenChainSequences(
  proteinResidues: readonly ArtiosPottsResidue[],
  byChain: Readonly<Record<string, string>>,
): FlattenedSequence {
  const chainIds: string[] = [];
  const chainLengths: number[] = [];
  for (const r of proteinResidues) {
    if (chainIds.length === 0 || chainIds[chainIds.length - 1] !== r.chain) {
      chainIds.push(r.chain);
      chainLengths.push(1);
    } else {
      chainLengths[chainLengths.length - 1]++;
    }
  }
  let sequence = '';
  for (const [i, ch] of chainIds.entries()) {
    const text = byChain[ch] ?? '';
    if (text.length !== chainLengths[i]) {
      throw new Error(
        `ArtiosPotts: chain ${ch} has ${chainLengths[i]} residues but the sampler returned ${text.length}.`,
      );
    }
    sequence += text;
  }
  return { sequence, chainIds, chainLengths };
}

/** Insertion-code-safe residue labels (`A1`, `H100A`, …) matching `proteinResidues`. */
export function residueIdsOf(proteinResidues: readonly ArtiosPottsResidue[]): string[] {
  return proteinResidues.map((r) => `${r.chain}${r.number}${r.insertionCode}`);
}

export interface ArtiosPottsScore {
  /** Potts energy summed over protein tokens (lower is better). */
  proteinEnergy: number;
  /** Total Potts energy over every token. */
  totalEnergy: number;
  /** Mean per-residue pseudo-log-likelihood over the 20 standard amino acids (higher is better). */
  pseudoLogLikelihood: number;
  /** Per-protein-residue pseudo-log-likelihood, in `proteinResidues` order. */
  perResidueLogLikelihood: Float32Array;
  /** Per-protein-residue Potts energy, in `proteinResidues` order. */
  perResidueEnergy: Float32Array;
}

export interface ArtiosPottsSample {
  flat: FlattenedSequence;
  score: ArtiosPottsScore;
  sampleIndex: number;
  seed: number;
  /** Number of accepted DLMC sweeps, or null for Gibbs (which cannot reject). */
  acceptedSweeps: number | null;
}

/**
 * Per-residue values restricted to protein tokens and reordered into `proteinResidues` order.
 * Non-finite entries (positions holding a non-standard restype) become NaN.
 */
export function proteinTokenValues(
  context: PottsContextLike,
  values: ArrayLike<number>,
): Float32Array {
  const out: number[] = [];
  for (let i = 0; i < context.nToken; i++) {
    if (context.isProteinToken[i]) out.push(values[i]);
  }
  return Float32Array.from(out);
}

const DESIGN_SCORE_PRECISION = 1e6;
const roundScore = (v: number): number =>
  Number.isFinite(v) ? Math.round(v * DESIGN_SCORE_PRECISION) / DESIGN_SCORE_PRECISION : 0;

/**
 * Build one SequenceCandidate. `score` is the negative mean pseudo-log-likelihood so that, as
 * everywhere else in the app, lower sorts better; `globalScore` carries the Potts energy per
 * designed residue.
 */
export function buildArtiosPottsCandidate(
  sample: ArtiosPottsSample,
  parent: StructureArtifact,
  options: ArtiosPottsOptions,
  idPrefix = 'artios-potts',
  extra: Partial<SequenceCandidate> = {},
): SequenceCandidate {
  const residues = Math.max(1, sample.flat.sequence.length);
  return {
    id: `${idPrefix}-${sample.sampleIndex + 1}`,
    sourceStructureId: parent.id,
    sampleIndex: sample.sampleIndex,
    sequence: sample.flat.sequence,
    score: Number.isFinite(sample.score.pseudoLogLikelihood)
      ? roundScore(-sample.score.pseudoLogLikelihood)
      : null,
    globalScore: Number.isFinite(sample.score.proteinEnergy)
      ? roundScore(sample.score.proteinEnergy / residues)
      : null,
    sourceModelId: 'artios-potts',
    temperature: options.temperature,
    seed: sample.seed,
    chainLengths: [...sample.flat.chainLengths],
    chainIds: [...sample.flat.chainIds],
    ...extra,
  };
}

export function artiosPottsRunHash(
  parent: StructureArtifact,
  options: ArtiosPottsOptions,
  fixedProteinIndices: ReadonlySet<number>,
): string {
  return fnv1a(
    JSON.stringify({
      structureId: parent.id,
      ...options,
      fixed: [...fixedProteinIndices].sort((a, b) => a - b),
    }),
  );
}

export function buildArtiosPottsSequenceSet(
  parent: StructureArtifact,
  candidates: SequenceCandidate[],
  options: ArtiosPottsOptions,
  lockedChains: readonly string[],
  runHash: string,
  extra: Partial<SequenceSet> = {},
): SequenceSet {
  return {
    kind: 'SequenceSet',
    id: `artios-potts-seq-${parent.id}-${options.seed}-${runHash}`,
    parentStructureId: parent.id,
    sourceStructureId: parent.id,
    sourceModelId: 'artios-potts',
    ...(lockedChains.length > 0 ? { lockedChains: [...lockedChains] } : {}),
    ...(parent.chainRoles ? { chainRoles: { ...parent.chainRoles } } : {}),
    candidates,
    ...extra,
  };
}

/**
 * Per-residue pseudo-log-likelihood track for the best (design) or native (score) sequence.
 * Rendered as a 1D profile and paintable onto the 3D view.
 */
export function buildArtiosPottsTrack(
  parent: StructureArtifact,
  proteinResidues: readonly ArtiosPottsResidue[],
  sample: ArtiosPottsSample,
  options: ArtiosPottsOptions,
  runHash: string,
): ResidueTrackArtifact {
  const values = sample.score.perResidueLogLikelihood;
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
      ? `ArtiosPotts PLL (native · mean ${mean.toFixed(3)})`
      : `ArtiosPotts PLL (best design · mean ${mean.toFixed(3)})`;
  return {
    kind: 'ResidueTrack',
    id: `artios-potts-track-${options.mode}-${options.precision}-${runHash}`,
    label,
    sourceModelId: 'artios-potts',
    parentStructureId: parent.id,
    quantity: 'potts-pll',
    unit: 'log p',
    residues: sample.flat.sequence,
    residueIds: residueIdsOf(proteinResidues),
    values,
    summary: {
      mean,
      max: counted > 0 ? max : Number.NaN,
      min: counted > 0 ? min : Number.NaN,
    },
  };
}

// ---------------------------------------------------------------------------
// Structure-collection fan-out
// ---------------------------------------------------------------------------

export function structureItems(structure: StructureArtifact): StructureArtifact[] {
  const items = structure.collection?.items;
  return items && items.length > 1 ? items : [structure];
}

export interface ArtiosPottsJob {
  parent: StructureArtifact;
  designKey?: string;
  designIndex: number;
}

/** One job per upstream backbone, keeping the collection's design keys and ordinals. */
export function planArtiosPottsJobs(structure: StructureArtifact): ArtiosPottsJob[] {
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

// ---------------------------------------------------------------------------
// Alphabet restriction
// ---------------------------------------------------------------------------

/** The 20 standard amino acids in `PROTEIN_TYPES_ONE_LETTER` order. */
export const STANDARD_AA_ONE_LETTER = 'ARNDCQEGHILKMFPSTWYV';

export interface ArtiosPottsAlphabet {
  /** Upstream `exclude_cys` flag (handled by `build_alphabet_mask`). */
  excludeCys: boolean;
  /** Upstream `exclude_met` flag (handled by `build_alphabet_mask`). */
  excludeMet: boolean;
  /**
   * Any other omitted 1-letter code, expressed through the upstream `energy_bias` channel at
   * `ALPHABET_BIAS` — the same constant and the same `applyFieldBias` path the alphabet mask uses.
   */
  energyBiasCodes: string[];
  /** Letters still available to the sampler. */
  designable: string;
}

/**
 * Split an omit set into the two upstream boolean flags plus a residual energy-bias list, so a
 * general "omit amino acids" control maps onto ArtiosPotts' own masking machinery exactly.
 */
export function resolveArtiosPottsAlphabet(omitAAs: string): ArtiosPottsAlphabet {
  const omit = new Set(sanitizeOmitAAs(omitAAs));
  const designable = [...STANDARD_AA_ONE_LETTER].filter((c) => !omit.has(c)).join('');
  if (designable.length === 0) {
    throw new Error('ArtiosPotts: omitAAs excludes all 20 amino acids, leaving nothing to design.');
  }
  return {
    excludeCys: omit.has('C'),
    excludeMet: omit.has('M'),
    energyBiasCodes: [...omit].filter((c) => c !== 'C' && c !== 'M').sort(),
    designable,
  };
}
