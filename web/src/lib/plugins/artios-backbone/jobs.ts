// Layer 3 Plugin helpers — artios-backbone
// Pure, worker-independent pieces of the ArtiosBackbone run: parameter normalization, PDB atom
// parsing, motif/ligand token-layout planning, AlphaFold3-style feature assembly on top of the
// vendored featurizer (`static/vendor/artios-backbone/shared/index.js`), PDB serialization of the
// sampled coordinates, and StructureArtifact assembly. Shared by the worker and the unit tests.

import { fnv1a } from '$lib/kernel';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  AtomGranularity,
  ChainMolecularKind,
  StructureArtifact,
  StructureMolecularContext,
} from '$lib/engine/types';
import {
  buildStandardComponents,
  SCAFFOLD_RESIDUE,
  type CcdComponent,
} from './ccd-standard';

export type ArtiosBackbonePrecision = 'int8' | 'fp16' | 'fp32';
export type ArtiosBackboneBackend = 'webgpu' | 'cpu';
export type ArtiosBackboneCpuFallbackReason = 'unavailable' | 'device-lost';

export interface ArtiosBackboneOptions {
  length: number;
  numDesigns: number;
  numSteps: number;
  motifContig: string;
  includeLigand: boolean;
  /** Target ligand RASA in [0, 1], or -1 for the unconditioned (null) bin. */
  ligandRasa: number;
  /** Target motif burial depth in [0, 2], or -1 for the unconditioned (null) bin. */
  motifDepth: number;
  stepScaleEta: number;
  noiseScaleLambda: number;
  gamma0: number;
  selfConditioning: boolean;
  seed: number;
  precision: ArtiosBackbonePrecision;
}

const WATER_RESIDUES = new Set(['HOH', 'DOD', 'WAT', 'H2O', 'TIP', 'SOL']);
const HYDROGEN_ELEMENTS = new Set(['H', 'D', 'T']);
const BACKBONE4 = new Set(['N', 'CA', 'C', 'O']);

function numOr(raw: unknown, fallback: number): number {
  if (raw === undefined || raw === null || typeof raw === 'boolean') return fallback;
  if (typeof raw === 'string' && raw.trim() === '') return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function clampInt(raw: unknown, fallback: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(numOr(raw, fallback))));
}

function clampNum(raw: unknown, fallback: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, numOr(raw, fallback)));
}

export function normalizeArtiosBackbonePrecision(raw: unknown): ArtiosBackbonePrecision {
  if (raw === undefined || raw === null || String(raw).trim() === '') return 'int8';
  const val = String(raw).trim().toLowerCase();
  if (val === 'int8' || val === 'fp16' || val === 'fp32') return val;
  throw new Error(
    `Unsupported ArtiosBackbone precision "${String(raw)}" (supported: int8, fp16, fp32).`,
  );
}

/** `-1` means "unconditioned" (the null bin); anything inside the range is a target value. */
function normalizeNullableTarget(raw: unknown, max: number, label: string): number {
  const value = numOr(raw, -1);
  if (value < 0) {
    if (value !== -1) {
      throw new Error(
        `ArtiosBackbone ${label} must be -1 (unconditioned) or within [0, ${max}], got ${value}.`,
      );
    }
    return -1;
  }
  return Math.min(max, value);
}

export function artiosBackboneOptionsFromParams(
  params: Record<string, unknown>,
): ArtiosBackboneOptions {
  return {
    length: clampInt(params.length, 100, 30, 512),
    numDesigns: clampInt(params.numDesigns, 1, 1, 16),
    numSteps: clampInt(params.numSteps, 200, 20, 400),
    motifContig: String(params.motifContig ?? '').trim(),
    includeLigand: params.includeLigand !== false,
    ligandRasa: normalizeNullableTarget(params.ligandRasa, 1, 'ligandRasa'),
    motifDepth: normalizeNullableTarget(params.motifDepth, 2, 'motifDepth'),
    stepScaleEta: clampNum(params.stepScaleEta, 1.5, 0.5, 3),
    noiseScaleLambda: clampNum(params.noiseScaleLambda, 1.003, 0.5, 1.5),
    gamma0: clampNum(params.gamma0, 0.8, 0, 2),
    selfConditioning: params.selfConditioning !== false,
    seed: clampInt(params.seed, 42, 0, 2_147_483_647),
    precision: normalizeArtiosBackbonePrecision(params.precision),
  };
}

// ===========================================================================
// 1. PDB atom records -> featurizer atom records
// ===========================================================================

/** One atom in the layout fed to the vendored featurizer (mirrors `parseMmcifAtoms` rows). */
export interface ArtiosAtom {
  atomName: string;
  resName: string;
  chainId: string;
  resId: number;
  insCode: string;
  element: string;
  charge: number;
  coord: [number, number, number];
  hetero: boolean;
}

function inferElement(rawName: string, trimmed: string): string {
  const alpha = trimmed.toUpperCase().replace(/[^A-Z]/g, '');
  if (!alpha) return 'C';
  // Standard PDB column alignment: single-letter elements are right-shifted by one column,
  // so a name starting in column 13 (no leading space) carries a two-letter element (ZN, FE, CL).
  if (rawName[0] !== ' ' && alpha.length >= 2) return alpha.slice(0, 2);
  return alpha.slice(0, 1);
}

function parseFormalCharge(raw: string): number {
  const text = raw.trim().toUpperCase();
  if (!text) return 0;
  const m = /^(\d)([+-])$|^([+-])(\d)$/.exec(text);
  if (m) {
    const magnitude = Number(m[1] ?? m[4]);
    const sign = (m[2] ?? m[3]) === '-' ? -1 : 1;
    return Number.isFinite(magnitude) ? sign * magnitude : 0;
  }
  const n = Number(text);
  return Number.isFinite(n) ? n : 0;
}

export interface ParsePdbAtomsOptions {
  keepHydrogens?: boolean;
  keepWaters?: boolean;
}

/**
 * Parse ATOM/HETATM records of a PDB file into featurizer atom rows.
 *
 * Mirrors the vendored `parseMmcifAtoms` semantics: hydrogens are dropped, only the first
 * alternate location of each (chain, resId, insCode, atomName) is kept, and only the first
 * MODEL is read.
 */
export function parsePdbAtomRecords(
  pdbText: string,
  { keepHydrogens = false, keepWaters = false }: ParsePdbAtomsOptions = {},
): ArtiosAtom[] {
  const out: ArtiosAtom[] = [];
  const seen = new Set<string>();
  let modelDepth = 0;
  for (const rawLine of pdbText.split('\n')) {
    const line = rawLine.replace(/\r$/, '');
    if (line.startsWith('MODEL')) {
      modelDepth += 1;
      continue;
    }
    if (line.startsWith('ENDMDL')) {
      if (modelDepth >= 1) break; // keep the first model only
      continue;
    }
    const isAtom = line.startsWith('ATOM  ');
    const isHet = line.startsWith('HETATM');
    if (!isAtom && !isHet) continue;
    if (line.length < 54) continue;

    const rawName = line.slice(12, 16);
    const atomName = rawName.trim();
    if (!atomName) continue;
    const resName = line.slice(17, 20).trim().toUpperCase();
    if (!keepWaters && WATER_RESIDUES.has(resName)) continue;

    const elementCol = line.slice(76, 78).trim().toUpperCase();
    const element = elementCol || inferElement(rawName, atomName);
    if (!keepHydrogens && HYDROGEN_ELEMENTS.has(element)) continue;

    const x = Number(line.slice(30, 38));
    const y = Number(line.slice(38, 46));
    const z = Number(line.slice(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new Error(
        `ArtiosBackbone: non-finite coordinates on atom ${atomName} of ${resName} in the input structure.`,
      );
    }

    const chainId = line.slice(21, 22).trim() || 'A';
    const resId = Math.round(Number(line.slice(22, 26)) || 0);
    const insCode = line.slice(26, 27).trim();
    const key = `${chainId}|${resId}|${insCode}|${atomName}`;
    if (seen.has(key)) continue; // first altLoc wins
    seen.add(key);

    out.push({
      atomName,
      resName,
      chainId,
      resId,
      insCode,
      element,
      charge: parseFormalCharge(line.slice(78, 80)),
      coord: [x, y, z],
      hetero: isHet,
    });
  }
  return out;
}

// ===========================================================================
// 2. motif contig parsing & token-layout planning
// ===========================================================================

export interface MotifSpanToken {
  /** `null` when the token carried no chain prefix (matches any chain). */
  chain: string | null;
  start: number;
  end: number;
}

/** Parse `"A10-25, A40-55"` / `"10-25"` / `"B77"` into ordered residue spans. */
export function parseMotifContig(contig: string): MotifSpanToken[] {
  const tokens = String(contig ?? '')
    .split(/[,;\s]+/)
    .filter(Boolean);
  if (tokens.length === 0) return [];
  return tokens.map((tok) => {
    const m = /^(?:([A-Za-z0-9]):?)?(-?\d+)(?:-(-?\d+))?$/.exec(tok);
    if (!m) {
      throw new Error(
        `Invalid ArtiosBackbone motifContig token "${tok}" (expected residue spans like A10-25 or A10-25, A40-55).`,
      );
    }
    const start = Number(m[2]);
    const end = m[3] !== undefined ? Number(m[3]) : start;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
      throw new Error(`Invalid ArtiosBackbone motifContig span "${tok}" (end precedes start).`);
    }
    return { chain: m[1] ? m[1].toUpperCase() : null, start, end };
  });
}

interface InputResidue {
  chainId: string;
  resId: number;
  insCode: string;
  resName: string;
  atoms: ArtiosAtom[];
}

/** Group parsed atoms into residues, preserving file order. */
export function groupResidues(atoms: readonly ArtiosAtom[]): InputResidue[] {
  const residues: InputResidue[] = [];
  let current: InputResidue | null = null;
  for (const atom of atoms) {
    if (
      !current ||
      current.chainId !== atom.chainId ||
      current.resId !== atom.resId ||
      current.insCode !== atom.insCode ||
      current.resName !== atom.resName
    ) {
      current = {
        chainId: atom.chainId,
        resId: atom.resId,
        insCode: atom.insCode,
        resName: atom.resName,
        atoms: [],
      };
      residues.push(current);
    }
    current.atoms.push(atom);
  }
  return residues;
}

/** Deterministic token/atom layout handed to the featurizer and to the PDB writer. */
export interface ArtiosTokenPlan {
  /** Model-order atoms: the designed chain first, then each ligand chain. */
  atoms: ArtiosAtom[];
  /** Per-atom motif flag (motif residue atoms and, when kept, every ligand atom). */
  isMotifAtom: Uint8Array;
  /** Per-atom `ATOM`/`HETATM` record type for the output PDB. */
  isHetAtom: Uint8Array;
  /** Residues in the designed chain (always `options.length`). */
  designedLength: number;
  /** 0-based residue indices of placed motif residues inside the designed chain. */
  motifIndices: number[];
  /** 1-based flat residue indices of placed motif residues (ProteinMPNN fixed positions). */
  fixedResidueIndices: number[];
  /** Native (uncentred) Cα coordinates of the placed motif residues, aligned with `motifIndices`. */
  nativeMotifCoords: number[][];
  /** CCD codes of the kept ligand components, in layout order. */
  ligandResNames: string[];
  /** Output chain IDs of the kept ligand components. */
  ligandChainIds: string[];
  /** CCD codes present in the layout that the bundled standard-residue table does not cover. */
  missingComponentCodes: string[];
  warnings: string[];
}

function scaffoldResidueAtoms(
  index: number,
  chainId: string,
  component: CcdComponent,
): ArtiosAtom[] {
  return component.atoms.map((atom) => ({
    atomName: atom.atomId,
    resName: component.id,
    chainId,
    resId: index + 1,
    insCode: '',
    element: atom.element,
    charge: 0,
    // Placeholder: the sampler replaces every coordinate; `ref_pos` comes from the CCD conformer.
    coord: [0, 0, 0] as [number, number, number],
    hetero: false,
  }));
}

const CHAIN_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

/** Build the unconditional layout: `length` generated scaffold residues on chain A. */
export function buildUnconditionalPlan(
  length: number,
  components: Map<string, CcdComponent> = buildStandardComponents(),
): ArtiosTokenPlan {
  const scaffold = components.get(SCAFFOLD_RESIDUE);
  if (!scaffold) {
    throw new Error(`ArtiosBackbone: missing bundled CCD component for ${SCAFFOLD_RESIDUE}.`);
  }
  const atoms: ArtiosAtom[] = [];
  for (let i = 0; i < length; i++) atoms.push(...scaffoldResidueAtoms(i, 'A', scaffold));
  return {
    atoms,
    isMotifAtom: new Uint8Array(atoms.length),
    isHetAtom: new Uint8Array(atoms.length),
    designedLength: length,
    motifIndices: [],
    fixedResidueIndices: [],
    nativeMotifCoords: [],
    ligandResNames: [],
    ligandChainIds: [],
    missingComponentCodes: [],
    warnings: [],
  };
}

export interface MotifPlanInput {
  pdbText: string;
  options: Pick<ArtiosBackboneOptions, 'length' | 'motifContig' | 'includeLigand'>;
  components?: Map<string, CcdComponent>;
}

/**
 * Build the motif + ligand layout.
 *
 * Motif residues keep their identity, every heavy atom and (when it fits) their native numbering
 * inside the designed chain; the remaining positions are generated scaffold residues. Kept ligand
 * components are appended as one token per atom and are conditioned on their input coordinates.
 */
export function buildMotifPlan({
  pdbText,
  options,
  components = buildStandardComponents(),
}: MotifPlanInput): ArtiosTokenPlan {
  const { length } = options;
  const scaffold = components.get(SCAFFOLD_RESIDUE);
  if (!scaffold) {
    throw new Error(`ArtiosBackbone: missing bundled CCD component for ${SCAFFOLD_RESIDUE}.`);
  }
  const warnings: string[] = [];
  const inputAtoms = parsePdbAtomRecords(pdbText);
  const residues = groupResidues(inputAtoms);
  const spans = parseMotifContig(options.motifContig);

  // --- select motif residues -------------------------------------------------
  const selected: InputResidue[][] = [];
  for (const span of spans) {
    const matched = residues.filter(
      (res) =>
        !res.atoms[0].hetero &&
        (span.chain === null || res.chainId.toUpperCase() === span.chain) &&
        res.resId >= span.start &&
        res.resId <= span.end,
    );
    if (matched.length === 0) {
      throw new Error(
        `ArtiosBackbone motif span ${span.chain ?? ''}${span.start}-${span.end} matched 0 polymer residues in the connected structure.`,
      );
    }
    selected.push(matched);
  }

  const motifResidueCount = selected.reduce((sum, grp) => sum + grp.length, 0);
  if (motifResidueCount > length) {
    throw new Error(
      `ArtiosBackbone motif needs ${motifResidueCount} residues but length is ${length}; raise length or shorten motifContig.`,
    );
  }

  // --- place spans, preferring native numbering ------------------------------
  const placement = new Map<number, InputResidue>();
  let cursor = 0;
  for (let s = 0; s < selected.length; s++) {
    const group = selected[s];
    const remainingAfter = selected.slice(s + 1).reduce((sum, grp) => sum + grp.length, 0);
    const latestStart = length - remainingAfter - group.length;
    const nativeStart = Math.min(...group.map((res) => res.resId)) - 1;
    const start = Math.max(cursor, Math.min(nativeStart, latestStart));
    if (start < 0 || start > latestStart) {
      throw new Error(
        `ArtiosBackbone cannot place ${motifResidueCount} motif residues inside a ${length}-residue design.`,
      );
    }
    if (start !== nativeStart) {
      warnings.push(
        `Motif span ${group[0].chainId}${group[0].resId}-${group[group.length - 1].resId} was shifted to design position ${start + 1} to fit a ${length}-residue chain.`,
      );
    }
    group.forEach((res, k) => placement.set(start + k, res));
    cursor = start + group.length;
  }

  // --- designed chain --------------------------------------------------------
  const atoms: ArtiosAtom[] = [];
  const motifFlags: number[] = [];
  const hetFlags: number[] = [];
  const motifIndices: number[] = [];
  const nativeMotifCoords: number[][] = [];
  const missing = new Set<string>();

  for (let i = 0; i < length; i++) {
    const motifRes = placement.get(i);
    if (!motifRes) {
      for (const atom of scaffoldResidueAtoms(i, 'A', scaffold)) {
        atoms.push(atom);
        motifFlags.push(0);
        hetFlags.push(0);
      }
      continue;
    }
    if (!components.has(motifRes.resName)) missing.add(motifRes.resName);
    for (const atom of motifRes.atoms) {
      atoms.push({ ...atom, chainId: 'A', resId: i + 1, insCode: '', hetero: false });
      motifFlags.push(1);
      hetFlags.push(0);
    }
    const ca = motifRes.atoms.find((a) => a.atomName === 'CA') ?? motifRes.atoms[0];
    motifIndices.push(i);
    nativeMotifCoords.push([...ca.coord]);
  }

  // --- ligand components ------------------------------------------------------
  const ligandResNames: string[] = [];
  const ligandChainIds: string[] = [];
  if (options.includeLigand) {
    const used = new Set(['A']);
    const chainMap = new Map<string, string>();
    for (const res of residues) {
      if (!res.atoms[0].hetero) continue;
      let outChain = chainMap.get(res.chainId);
      if (!outChain) {
        outChain =
          (!used.has(res.chainId.toUpperCase()) && res.chainId.length === 1
            ? res.chainId.toUpperCase()
            : undefined) ?? CHAIN_ALPHABET.split('').find((c) => !used.has(c));
        if (!outChain) {
          warnings.push(`Dropped ligand ${res.resName}: no free output chain ID remained.`);
          continue;
        }
        used.add(outChain);
        chainMap.set(res.chainId, outChain);
      }
      if (!components.has(res.resName)) missing.add(res.resName);
      ligandResNames.push(res.resName);
      ligandChainIds.push(outChain);
      for (const atom of res.atoms) {
        atoms.push({ ...atom, chainId: outChain, hetero: true });
        // Ligand coordinates are given, so every ligand atom conditions the generation.
        motifFlags.push(1);
        hetFlags.push(1);
      }
    }
  }

  return {
    atoms,
    isMotifAtom: Uint8Array.from(motifFlags),
    isHetAtom: Uint8Array.from(hetFlags),
    designedLength: length,
    motifIndices,
    fixedResidueIndices: motifIndices.map((i) => i + 1),
    nativeMotifCoords,
    ligandResNames,
    ligandChainIds,
    missingComponentCodes: [...missing].sort(),
    warnings,
  };
}

// ===========================================================================
// 3. feature assembly on top of the vendored featurizer
// ===========================================================================

type FlatArray = Float32Array | Int32Array | Uint8Array;

/** The subset of `static/vendor/artios-backbone/shared/index.js` the feature builder needs. */
export interface ArtiosSharedFeaturizer {
  ELEMENT_SYMBOLS: readonly string[];
  buildReferenceFeatures(opts: {
    atoms: readonly ArtiosAtom[];
    components: Map<string, CcdComponent>;
    augment?: boolean;
    rng?: unknown;
    ligandAtomRename?: boolean;
  }): {
    nAtom: number;
    ref_pos: Float32Array;
    ref_mask: Int32Array;
    ref_charge: Float32Array;
    ref_element: Float32Array;
    ref_atom_name_chars: Float32Array;
    ref_space_uid: Int32Array;
    missingComponents: string[];
    unresolvedAtoms: Array<{ index: number; reason: string }>;
  };
  buildTokenization(opts: {
    atoms: readonly ArtiosAtom[];
    components: Map<string, CcdComponent>;
  }): {
    nToken: number;
    atom_to_token_idx: Int32Array;
    token_index: Int32Array;
    residue_index: Int32Array;
    asym_id: Int32Array;
    entity_id: Int32Array;
    sym_id: Int32Array;
    centre_atom_index: Int32Array;
    token_res_name: string[];
    is_protein_token: Uint8Array;
    is_ligand_token: Uint8Array;
  };
  buildMotifFeatures(opts: {
    atomToTokenIdx: Int32Array;
    refSpaceUid: Int32Array;
    isMotifAtom: Uint8Array;
    nToken: number;
    centreAtomIndex: Int32Array;
    tokenResName: string[];
    coordinate: Float32Array;
  }): {
    is_motif: Uint8Array;
    is_motif_related: Uint8Array;
    is_motif_token: Uint8Array;
    is_motif_related_token: Uint8Array;
    motif_restype: Float32Array;
    motif_coordinate: Float32Array;
  };
  buildMotifDepthFeature(nToken: number, value: number): Float32Array;
  buildRasaFeature(opts: {
    nToken: number;
    isLigandToken: Uint8Array;
    values?: ArrayLike<number> | null;
    fallback?: number;
  }): Float32Array;
}

/** Full ArtiosBackbone feature block (`prepareFeatures` contract) plus layout metadata. */
export interface ArtiosFeatures extends Record<string, FlatArray | number | string[] | undefined> {
  nAtom: number;
  nToken: number;
  atom_to_token_idx: Int32Array;
  atom_mask: Uint8Array;
  token_mask: Uint8Array;
  token_index: Int32Array;
  residue_index: Int32Array;
  asym_id: Int32Array;
  entity_id: Int32Array;
  sym_id: Int32Array;
  ref_pos: Float32Array;
  ref_mask: Int32Array;
  ref_element: Float32Array;
  ref_charge: Float32Array;
  ref_atom_name_chars: Float32Array;
  ref_space_uid: Int32Array;
  is_protein: Uint8Array;
  is_ligand: Uint8Array;
  is_dna: Uint8Array;
  is_rna: Uint8Array;
  is_protein_token: Uint8Array;
  is_ligand_token: Uint8Array;
  is_dna_token: Uint8Array;
  is_rna_token: Uint8Array;
  is_motif: Uint8Array;
  is_motif_related: Uint8Array;
  is_motif_token: Uint8Array;
  is_motif_related_token: Uint8Array;
  motif_restype: Float32Array;
  motif_coordinate: Float32Array;
  motif_depth_norm: Float32Array;
  rasa: Float32Array;
}

export interface ArtiosFeatureResult {
  features: ArtiosFeatures;
  /** Atoms without a reference conformer (zero `ref_pos`, `ref_mask = 0`). */
  unresolvedAtomCount: number;
  missingComponents: string[];
  warnings: string[];
}

/**
 * Assemble the full feature block for one token layout.
 *
 * Every atom-level and token-level array required by `prepareFeatures` is produced here from the
 * vendored reference-conformer, tokenization and motif featurizers, so the CPU and WebGPU
 * denoisers consume byte-identical inputs.
 */
export function buildArtiosFeatures(
  shared: ArtiosSharedFeaturizer,
  plan: ArtiosTokenPlan,
  options: Pick<ArtiosBackboneOptions, 'ligandRasa' | 'motifDepth'>,
  components: Map<string, CcdComponent>,
): ArtiosFeatureResult {
  const { atoms } = plan;
  const nAtom = atoms.length;
  if (nAtom === 0) throw new Error('ArtiosBackbone: empty token layout (no atoms to generate).');

  const knownElements = new Set(shared.ELEMENT_SYMBOLS);
  for (const atom of atoms) {
    if (!knownElements.has(atom.element)) {
      throw new Error(
        `ArtiosBackbone: unknown element "${atom.element}" on atom ${atom.atomName} of ${atom.resName} ${atom.chainId}${atom.resId}.`,
      );
    }
  }

  const hasLigand = plan.ligandResNames.length > 0;
  const ref = shared.buildReferenceFeatures({
    atoms,
    components,
    augment: false,
    // `ContextGenerator.get_renamed_atom_names`: ligand identity must not leak through atom names.
    ligandAtomRename: hasLigand,
  });
  const tok = shared.buildTokenization({ atoms, components });

  const coordinate = new Float32Array(nAtom * 3);
  for (let i = 0; i < nAtom; i++) {
    coordinate[i * 3] = atoms[i].coord[0];
    coordinate[i * 3 + 1] = atoms[i].coord[1];
    coordinate[i * 3 + 2] = atoms[i].coord[2];
  }

  const motif = shared.buildMotifFeatures({
    atomToTokenIdx: tok.atom_to_token_idx,
    refSpaceUid: ref.ref_space_uid,
    isMotifAtom: plan.isMotifAtom,
    nToken: tok.nToken,
    centreAtomIndex: tok.centre_atom_index,
    tokenResName: tok.token_res_name,
    coordinate,
  });

  // Per-atom molecule-class flags from each atom's component (ligand when unknown).
  const isProtein = new Uint8Array(nAtom);
  const isLigand = new Uint8Array(nAtom);
  const isDna = new Uint8Array(nAtom);
  const isRna = new Uint8Array(nAtom);
  for (let i = 0; i < nAtom; i++) {
    const molType = components.get(atoms[i].resName)?.molType ?? 'ligand';
    if (molType === 'protein') isProtein[i] = 1;
    else if (molType === 'dna') isDna[i] = 1;
    else if (molType === 'rna') isRna[i] = 1;
    else isLigand[i] = 1;
  }
  const isDnaToken = new Uint8Array(tok.nToken);
  const isRnaToken = new Uint8Array(tok.nToken);
  for (let t = 0; t < tok.nToken; t++) {
    const centre = tok.centre_atom_index[t];
    isDnaToken[t] = isDna[centre];
    isRnaToken[t] = isRna[centre];
  }

  const features: ArtiosFeatures = {
    nAtom,
    nToken: tok.nToken,
    atom_to_token_idx: tok.atom_to_token_idx,
    atom_mask: new Uint8Array(nAtom).fill(1),
    token_mask: new Uint8Array(tok.nToken).fill(1),
    token_index: tok.token_index,
    residue_index: tok.residue_index,
    asym_id: tok.asym_id,
    entity_id: tok.entity_id,
    sym_id: tok.sym_id,
    ref_pos: ref.ref_pos,
    ref_mask: ref.ref_mask,
    ref_element: ref.ref_element,
    ref_charge: ref.ref_charge,
    ref_atom_name_chars: ref.ref_atom_name_chars,
    ref_space_uid: ref.ref_space_uid,
    is_protein: isProtein,
    is_ligand: isLigand,
    is_dna: isDna,
    is_rna: isRna,
    is_protein_token: tok.is_protein_token,
    is_ligand_token: tok.is_ligand_token,
    is_dna_token: isDnaToken,
    is_rna_token: isRnaToken,
    is_motif: motif.is_motif,
    is_motif_related: motif.is_motif_related,
    is_motif_token: motif.is_motif_token,
    is_motif_related_token: motif.is_motif_related_token,
    motif_restype: motif.motif_restype,
    motif_coordinate: motif.motif_coordinate,
    motif_depth_norm: shared.buildMotifDepthFeature(tok.nToken, options.motifDepth),
    rasa: shared.buildRasaFeature({
      nToken: tok.nToken,
      isLigandToken: tok.is_ligand_token,
      fallback: options.ligandRasa,
    }),
  };

  const warnings: string[] = [];
  if (ref.unresolvedAtoms.length > 0) {
    const codes = ref.missingComponents.length ? ` (${ref.missingComponents.join(', ')})` : '';
    warnings.push(
      `${ref.unresolvedAtoms.length} atom(s) had no CCD reference conformer${codes}; their ref_pos is zero and ref_mask = 0.`,
    );
  }
  return {
    features,
    unresolvedAtomCount: ref.unresolvedAtoms.length,
    missingComponents: ref.missingComponents,
    warnings,
  };
}

// ===========================================================================
// 4. EDM reverse diffusion
// ===========================================================================

export interface ArtiosRng {
  gaussian(): number;
  uniform(): number;
}

/** The sampling helpers of `static/vendor/artios-backbone/shared/index.js`. */
export interface ArtiosSharedSampler {
  SIGMA_DATA: number;
  createRng(seed: number): ArtiosRng;
  edmNoiseSchedule(nSteps: number, overrides?: Record<string, number>): Float64Array;
  centreRandomAugmentation(
    coords: Float32Array,
    opts: { mask?: ArrayLike<number> | null; sTrans?: number; rng?: ArtiosRng | null },
  ): { coords: Float32Array };
  kabschAlign(
    pred: Float32Array,
    target: Float32Array,
    opts?: { mask?: ArrayLike<number> | null },
  ): { coords: Float32Array };
}

export type ArtiosSharedModule = ArtiosSharedFeaturizer & ArtiosSharedSampler;

/** `gammaMin` of the upstream `EDMSampler`: churn only applies while sigma > 1. */
export const ARTIOS_GAMMA_MIN = 1.0;

/** Per-sample `centre_random_augmentation` over a batched `[S * A * 3]` coordinate block. */
export function centreAugmentBatch(
  shared: ArtiosSharedSampler,
  x: Float32Array,
  nSample: number,
  mask: ArrayLike<number>,
  rng: ArtiosRng,
): Float32Array {
  const perSample = x.length / nSample;
  const out = new Float32Array(x.length);
  for (let s = 0; s < nSample; s++) {
    const { coords } = shared.centreRandomAugmentation(
      x.subarray(s * perSample, (s + 1) * perSample),
      { mask, sTrans: 1.0, rng },
    );
    out.set(coords, s * perSample);
  }
  return out;
}

/** Per-sample Kabsch alignment of the previous prediction onto the current noisy coordinates. */
export function alignPrevPredBatch(
  shared: ArtiosSharedSampler,
  prev: Float32Array,
  xNoisy: Float32Array,
  nSample: number,
  mask: ArrayLike<number>,
): Float32Array {
  const perSample = prev.length / nSample;
  const out = new Float32Array(prev.length);
  for (let s = 0; s < nSample; s++) {
    const { coords } = shared.kabschAlign(
      prev.subarray(s * perSample, (s + 1) * perSample),
      xNoisy.subarray(s * perSample, (s + 1) * perSample),
      { mask },
    );
    out.set(coords, s * perSample);
  }
  return out;
}

/** One denoiser call: `(xNoisy, tHat, alignedPrevPred) -> xDenoised`, batched over `nSample`. */
export type ArtiosDenoiser = (
  xNoisy: Float32Array,
  tHat: number,
  prevPred: Float32Array | null,
) => Promise<Float32Array>;

export interface ArtiosSamplerArgs {
  denoise: ArtiosDenoiser;
  nAtom: number;
  nSample: number;
  atomMask: ArrayLike<number>;
  options: Pick<
    ArtiosBackboneOptions,
    'numSteps' | 'gamma0' | 'noiseScaleLambda' | 'stepScaleEta' | 'selfConditioning'
  >;
  seed: number;
  onStep?: (step: number) => void;
  signal?: AbortSignal;
}

/**
 * Integrate the EDM Karras schedule for `nSample` designs at once.
 *
 * Mirrors `EDMSampler.sample` (`generator.py`): centre + random augmentation, churn noise at
 * `t_hat = sigma * (1 + gamma)`, the denoiser call, then the `eta`-scaled Euler update. The
 * previous denoised prediction is Kabsch-aligned onto the current noisy coordinates before it is
 * fed back (`align_pred_to_true`), which is exactly the parity-verified self-conditioning path.
 */
export async function runArtiosEdmSampler(
  shared: ArtiosSharedSampler,
  { denoise, nAtom, nSample, atomMask, options, seed, onStep, signal }: ArtiosSamplerArgs,
): Promise<Float32Array> {
  const rng = shared.createRng(seed);
  const schedule = shared.edmNoiseSchedule(options.numSteps);
  const total = nSample * nAtom * 3;
  let x = new Float32Array(total);
  for (let i = 0; i < total; i++) x[i] = schedule[0] * rng.gaussian();
  let prevPred: Float32Array | null = null;

  for (let step = 0; step < options.numSteps; step++) {
    signal?.throwIfAborted();
    const cTauLast = schedule[step];
    const cTau = schedule[step + 1];
    const gamma = cTau > ARTIOS_GAMMA_MIN ? options.gamma0 : 0;
    const tHat = cTauLast * (1 + gamma);
    const noiseStd =
      options.noiseScaleLambda * Math.sqrt(Math.max(0, tHat * tHat - cTauLast * cTauLast));

    const centred = centreAugmentBatch(shared, x, nSample, atomMask, rng);
    const xNoisy = new Float32Array(total);
    for (let i = 0; i < total; i++) xNoisy[i] = centred[i] + noiseStd * rng.gaussian();

    const prev =
      options.selfConditioning && prevPred
        ? alignPrevPredBatch(shared, prevPred, xNoisy, nSample, atomMask)
        : null;
    const xDenoised = await denoise(xNoisy, tHat, prev);
    if (xDenoised.length !== total) {
      throw new Error(
        `ArtiosBackbone denoiser returned ${xDenoised.length} values, expected ${total}.`,
      );
    }
    const k = (options.stepScaleEta * (cTau - tHat)) / tHat;
    const next = new Float32Array(total);
    for (let i = 0; i < total; i++) next[i] = xNoisy[i] + k * (xNoisy[i] - xDenoised[i]);
    x = next;
    if (options.selfConditioning) prevPred = xDenoised;
    onStep?.(step + 1);
  }
  return x;
}

/** Designs batched per GPU submission; the pair-bias cache grows with tokens², so taper with N. */
export function artiosGpuBatchLimit(nToken: number): number {
  if (nToken <= 128) return 8;
  if (nToken <= 256) return 4;
  return 2;
}

// ===========================================================================
// 5. sampled coordinates -> PDB
// ===========================================================================

function formatAtomName(name: string): string {
  if (name.length >= 4) return name.slice(0, 4);
  return ` ${name.padEnd(3, ' ')}`;
}

/** Serialize one sampled coordinate set (`[nAtom * 3]`, Å) as a clean PDB model. */
export function coordsToArtiosPdb(
  coords: Float32Array,
  plan: ArtiosTokenPlan,
  { remarks = [] }: { remarks?: readonly string[] } = {},
): string {
  const { atoms, isHetAtom } = plan;
  if (coords.length !== atoms.length * 3) {
    throw new Error(
      `ArtiosBackbone: expected ${atoms.length * 3} coordinates for ${atoms.length} atoms, got ${coords.length}.`,
    );
  }
  const lines: string[] = [];
  for (const remark of remarks) lines.push(`REMARK   1 ${remark}`.slice(0, 80));
  let serial = 1;
  let prevChain = '';
  let prevHet = false;
  for (let i = 0; i < atoms.length; i++) {
    const atom = atoms[i];
    const x = coords[i * 3];
    const y = coords[i * 3 + 1];
    const z = coords[i * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new Error(
        `ArtiosBackbone generated a non-finite coordinate for atom ${atom.atomName} of ${atom.resName} ${atom.chainId}${atom.resId}.`,
      );
    }
    const het = isHetAtom[i] === 1;
    const chain = (atom.chainId || 'A').slice(0, 1);
    if (prevChain && (chain !== prevChain || (het && !prevHet))) lines.push('TER');
    prevChain = chain;
    prevHet = het;
    const record = het ? 'HETATM' : 'ATOM  ';
    const resSeq = String(Math.max(-999, Math.min(9999, atom.resId))).padStart(4, ' ');
    const ins = (atom.insCode || ' ').slice(0, 1);
    lines.push(
      `${record}${String(serial % 100000).padStart(5, ' ')} ${formatAtomName(atom.atomName)} ` +
        `${atom.resName.slice(0, 3).padStart(3, ' ')} ${chain}${resSeq}${ins}   ` +
        `${x.toFixed(3).padStart(8, ' ')}${y.toFixed(3).padStart(8, ' ')}${z.toFixed(3).padStart(8, ' ')}` +
        `  1.00  0.00          ${atom.element.slice(0, 2).padStart(2, ' ')}`,
    );
    serial++;
  }
  lines.push('TER');
  lines.push('END');
  return `${lines.join('\n')}\n`;
}

function granularity(pdbText: string): AtomGranularity {
  for (const line of pdbText.split('\n')) {
    if (!line.startsWith('ATOM')) continue;
    if (!BACKBONE4.has(line.slice(12, 16).trim())) return 'all-atom';
  }
  return 'backbone4';
}

// ===========================================================================
// 6. StructureArtifact assembly
// ===========================================================================

export interface ArtiosDesignResult {
  pdbText: string;
  designIndex: number;
  seed: number;
  /** Steps actually integrated (equals `numSteps` unless the run was truncated). */
  stepsRun: number;
  warnings: string[];
}

export interface ArtiosArtifactContext {
  options: ArtiosBackboneOptions;
  plan: ArtiosTokenPlan;
  backend: ArtiosBackboneBackend;
  fallbackReason?: ArtiosBackboneCpuFallbackReason;
  parent?: StructureArtifact;
  runHash?: string;
}

function buildMolecularContext(plan: ArtiosTokenPlan): StructureMolecularContext | undefined {
  if (plan.ligandResNames.length === 0) return undefined;
  const chainTypes: Record<string, ChainMolecularKind> = { A: 'protein' };
  for (const chain of plan.ligandChainIds) chainTypes[chain] = 'ligand';
  return { ligands: [...new Set(plan.ligandResNames)], chainTypes };
}

export function buildArtiosBackboneArtifact(
  result: ArtiosDesignResult,
  ctx: ArtiosArtifactContext,
): StructureArtifact {
  const { options, plan, backend, fallbackReason, parent } = ctx;
  const parsed = parseStructureText(result.pdbText);
  if (parsed.nResidues !== plan.designedLength) {
    throw new Error(
      `ArtiosBackbone emitted ${parsed.nResidues} polymer residue(s) instead of the requested ${plan.designedLength}.`,
    );
  }
  for (let i = 0; i < parsed.caCoords.length; i++) {
    if (!Number.isFinite(parsed.caCoords[i])) {
      throw new Error(
        `ArtiosBackbone generated non-finite Cα coordinates for design #${result.designIndex + 1}.`,
      );
    }
  }
  const geometry = computeGeometryStats(
    parsed.caCoords,
    parsed.chainIds,
    parsed.resNums,
    parsed.insCodes,
  );
  const runHash =
    ctx.runHash ??
    fnv1a(
      JSON.stringify({
        ...options,
        numDesigns: undefined,
        seed: result.seed,
        parentId: parent?.id ?? null,
        nAtom: plan.atoms.length,
      }),
    );
  const hasMotif = plan.motifIndices.length > 0;
  const tag = hasMotif ? 'motif' : 'de novo';
  const warnings = [...plan.warnings, ...result.warnings];
  if (backend === 'cpu') {
    warnings.push(
      fallbackReason === 'device-lost'
        ? 'ArtiosBackbone fell back to the CPU JS denoiser after a WebGPU device loss.'
        : 'ArtiosBackbone ran on the CPU JS denoiser (WebGPU unavailable); this is far slower than the WebGPU path.',
    );
  }
  const molecularContext = buildMolecularContext(plan);
  const chainRoles: Record<string, 'target' | 'binder' | 'scaffold'> = { A: 'scaffold' };

  return {
    kind: 'StructureArtifact',
    id: `ab-${plan.designedLength}aa-s${result.seed}-${runHash}`,
    label:
      options.numDesigns > 1
        ? `ArtiosBackbone #${result.designIndex + 1} (${plan.designedLength} aa, ${tag})`
        : `ArtiosBackbone (${plan.designedLength} aa, ${tag}, ${result.stepsRun} steps)`,
    sourceModelId: `artios-backbone/${options.precision}`,
    ...(parent?.id ? { parentStructureId: parent.id } : {}),
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: result.pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    chainIdMap: parsed.chainIdMap,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    atomGranularity: granularity(result.pdbText),
    bfactorSemantics: 'none',
    chainRoles,
    ...(molecularContext ? { molecularContext } : {}),
    ...(hasMotif
      ? {
          motifIndices: [...plan.motifIndices],
          fixedResidueIndices: [...plan.fixedResidueIndices],
          nativeMotifCoords: plan.nativeMotifCoords.map((c) => [...c]),
        }
      : {}),
    designKey: `design-${result.designIndex + 1}`,
    designIndex: result.designIndex,
    seed: result.seed,
    provenance: {
      quantization: options.precision,
      diffusionStepsScheduled: options.numSteps,
      diffusionStepsRun: result.stepsRun,
      ...(backend === 'cpu' ? { degradedFallback: 'cpu-js' as const } : {}),
      ...(warnings.length ? { warnings } : {}),
    },
    geometry,
  };
}

/** Wrap multiple designs into one multi-design StructureArtifact collection. */
export function wrapArtiosBackboneOutputs(items: StructureArtifact[]): StructureArtifact {
  if (items.length === 0) throw new Error('ArtiosBackbone produced no designs.');
  const primary = items[0];
  if (items.length === 1) return primary;
  const batchHash = fnv1a(`${items.map((it) => it.id).join('|')}:${items.length}`);
  const representativeSelection = {
    selectedKey: primary.designKey ?? primary.id,
    criterion: 'lowest-creation-ordinal',
    totalMembers: items.length,
    selectedIndex: 0,
    scoreValue: null,
  };
  return {
    ...primary,
    id: `ab-batch-${items.length}-${batchHash}`,
    label: `ArtiosBackbone (${items.length} designs · L=${primary.length})`,
    representativeSelection,
    collection: {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: `ab-col-${batchHash}`,
      familyKey: `artios-backbone-${batchHash}`,
      collectionKind: 'independent-candidates',
      requestedCount: items.length,
      complete: true,
      items,
      keys: items.map((it) => it.designKey ?? it.id),
      representativeSelection,
    },
  };
}

/** RCSB CCD URL for a component code (`fetchCcdComponent`'s source). */
export function rcsbCcdUrl(ccdRaw: string): string {
  const ccd = String(ccdRaw).trim().toUpperCase();
  if (!/^[A-Z0-9]{1,5}$/.test(ccd)) {
    throw new Error(
      `Invalid CCD component code "${ccdRaw}" (expected 1–5 alphanumeric characters, e.g. ATP, HEM, ZN).`,
    );
  }
  return `https://files.rcsb.org/ligands/download/${encodeURIComponent(ccd)}.cif`;
}

/**
 * Last-resort component built from the input structure's own coordinates.
 *
 * Used when a ligand's CCD entry is unavailable (offline, or an unknown code): `ref_pos` then
 * carries the input conformer re-centred by `buildReferenceFeatures` instead of the CCD ideal
 * conformer, which is a documented fidelity caveat rather than a hard failure.
 */
export function componentFromInputLigand(resName: string, atoms: readonly ArtiosAtom[]): CcdComponent {
  const seen = new Map<string, number>();
  const compAtoms = atoms
    .filter((atom) => {
      if (seen.has(atom.atomName)) return false;
      seen.set(atom.atomName, seen.size);
      return true;
    })
    .map((atom) => ({
      atomId: atom.atomName,
      altAtomId: null,
      element: atom.element,
      charge: atom.charge,
      leaving: false,
      aromatic: false,
      coord: [...atom.coord] as [number, number, number],
      coordSource: 'model' as const,
      hasCoord: true,
    }));
  return {
    id: resName,
    type: 'NON-POLYMER',
    molType: 'ligand',
    atoms: compAtoms,
    atomIndex: new Map(compAtoms.map((atom, i) => [atom.atomId, i])),
    bonds: [],
    coordSource: 'model',
  };
}
