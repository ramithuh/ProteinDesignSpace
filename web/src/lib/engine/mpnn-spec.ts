// Pure synchronous ProteinMPNN input compiler & constraint helpers.
// Single source of truth shared by `mpnn.worker.ts` and `input-preview.ts`
// so the Compiled Input Preview matches exact worker execution masks 1:1.

import {
  parsePDB,
  type ModelFamily,
  type ResidueRef,
  type Structure as MpnnStructure,
} from 'proteinmpnn-web';
import type { StructureArtifact } from './types';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Apply deterministic seeded Gaussian noise (Å) to finite backbone coordinates,
 * matching upstream ProteinMPNN `--backbone_noise` semantics while preserving NaN gaps.
 */
export function applyBackboneNoise(
  coordinates: Float32Array,
  sigma: number,
  seed: number,
): Float32Array {
  if (!(sigma > 0) || !Number.isFinite(sigma)) return coordinates;
  const out = new Float32Array(coordinates.length);
  const rand = mulberry32((seed ^ 0x9e3779b9) >>> 0);
  let hasSpare = false;
  let spare = 0;
  const nextGaussian = (): number => {
    if (hasSpare) {
      hasSpare = false;
      return spare;
    }
    let u = 0;
    let v = 0;
    while (u <= 1e-12) u = rand();
    v = rand();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    spare = mag * Math.sin(2.0 * Math.PI * v);
    hasSpare = true;
    return mag * Math.cos(2.0 * Math.PI * v);
  };
  for (let i = 0; i < coordinates.length; i++) {
    const val = coordinates[i];
    out[i] = Number.isFinite(val) ? val + sigma * nextGaussian() : val;
  }
  return out;
}

export function computeChainLengths(chainIds: readonly string[]): number[] {
  const lengths: number[] = [];
  for (let i = 0; i < chainIds.length; i++) {
    if (i === 0 || chainIds[i] !== chainIds[i - 1]) {
      lengths.push(1);
    } else {
      lengths[lengths.length - 1]++;
    }
  }
  return lengths;
}

/**
 * Parse human-friendly fixed-position strings into `{ fixedPositions, lockedChains }`.
 * Supports: whole-chain locking ("A", "B"), single residues ("A10", "A100A", "100A"),
 * ranges ("A10-15", "A10-A15", reversed "A15-10"), negative resids ("A-2", "A-2-5", "A-5--2"),
 * lowercase chain normalization ("a10" -> "A"), and default chain resolution ("_" for blank-chain PDBs).
 */
export function parseFixedPositionsAndChains(
  spec: string,
  defaultChain = 'A',
): { fixedPositions: ResidueRef[]; lockedChains: string[] } {
  const fixedPositions: ResidueRef[] = [];
  const lockedChains: string[] = [];
  const seenChains = new Set<string>();
  const trimmed = spec.trim();
  if (!trimmed) return { fixedPositions, lockedChains };
  for (const raw of trimmed.split(/[,\s]+/)) {
    if (!raw) continue;
    // Whole-chain token e.g. "A", "B", "_"
    const chainOnly = /^([A-Za-z_])$/.exec(raw);
    if (chainOnly) {
      const ch = chainOnly[1].toUpperCase();
      if (!seenChains.has(ch)) {
        seenChains.add(ch);
        lockedChains.push(ch);
      }
      continue;
    }

    const rangeMatch = /^([A-Za-z_])?:?(-?\d+)-([A-Za-z_])?:?(-?\d+)$/.exec(raw);
    if (rangeMatch) {
      const startChain = rangeMatch[1] ? rangeMatch[1].toUpperCase() : null;
      const endChain = rangeMatch[3] ? rangeMatch[3].toUpperCase() : null;
      if (startChain && endChain && startChain !== endChain) {
        throw new Error(
          `Invalid fixed position "${raw}": range must stay within a single chain (${startChain} vs ${endChain})`,
        );
      }
      const chain = startChain ?? endChain ?? defaultChain;
      const a = Number(rangeMatch[2]);
      const b = Number(rangeMatch[4]);
      const start = Math.min(a, b);
      const end = Math.max(a, b);
      if (end - start > 100000) {
        throw new Error(`Invalid fixed position "${raw}": residue range too large`);
      }
      for (let pos = start; pos <= end; pos++) {
        fixedPositions.push({ chain, number: pos });
      }
      continue;
    }

    const singleMatch = /^([A-Za-z_])?:?(-?\d+)([A-Za-z])?$/.exec(raw);
    if (singleMatch) {
      const chain = singleMatch[1] ? singleMatch[1].toUpperCase() : defaultChain;
      const number = Number(singleMatch[2]);
      const insertionCode = singleMatch[3];
      fixedPositions.push(insertionCode ? { chain, number, insertionCode } : { chain, number });
      continue;
    }

    throw new Error(
      `Invalid fixed position token "${raw}". Expected e.g. "A", "A10", "A10-15", "A10-A15", "A-2-5", or "A100A".`,
    );
  }
  return { fixedPositions, lockedChains };
}

/**
 * Parse tied-position groups like "A10=B10, A11=B11" (or semicolon-separated groups "A10 B10; A11 B11")
 * into `ResidueRef[][]` for ProteinMPNN homomer / symmetry tying.
 */
export function parseTiedPositions(spec: unknown, defaultChain = 'A'): ResidueRef[][] {
  if (Array.isArray(spec)) {
    return spec as ResidueRef[][];
  }
  const text = String(spec ?? '').trim();
  if (!text) return [];
  const groups: ResidueRef[][] = [];
  const rawGroups = text.includes(';') ? text.split(';') : text.split(',');
  for (const rawGroup of rawGroups) {
    const tokens = rawGroup
      .trim()
      .split(/[=\s]+/)
      .filter(Boolean);
    if (tokens.length < 2) continue;
    const refs: ResidueRef[] = [];
    for (const tok of tokens) {
      const m = /^([A-Za-z_])?:?(-?\d+)([A-Za-z])?$/.exec(tok);
      if (!m) {
        throw new Error(`Invalid tied position token "${tok}" in "${rawGroup.trim()}".`);
      }
      const chain = m[1] ? m[1].toUpperCase() : defaultChain;
      const number = Number(m[2]);
      const insertionCode = m[3];
      refs.push(insertionCode ? { chain, number, insertionCode } : { chain, number });
    }
    if (refs.length >= 2) groups.push(refs);
  }
  return groups;
}

/**
 * Compute gap-preserving per-chain positional encoding `residueIndices` from PDB residue numbers
 * so missing-density loops retain their true sequence separation instead of collapsing to adjacent indices.
 */
export function buildGapAwareResidueIndices(s: MpnnStructure): Int32Array {
  const out = new Int32Array(s.length);
  const firstNumByChain = new Map<string, number>();
  const lastIdxByChain = new Map<string, number>();
  for (let i = 0; i < s.length; i++) {
    const ch = s.chainIds[i];
    const rn = s.residueNumbers[i];
    if (!firstNumByChain.has(ch)) {
      firstNumByChain.set(ch, rn);
      out[i] = 0;
      lastIdxByChain.set(ch, 0);
    } else {
      const base = firstNumByChain.get(ch)!;
      const prev = lastIdxByChain.get(ch)!;
      const candidate = rn - base;
      const next = candidate > prev ? candidate : prev + 1;
      out[i] = next;
      lastIdxByChain.set(ch, next);
    }
  }
  return out;
}

export interface MpnnResidueSpec {
  /** 0-based flat index in the ProteinMPNN structure arrays. */
  flatIndex: number;
  chain: string;
  resNum: number;
  insCode: string;
  aa1: string;
  /** Exact post-precedence designability mask (`true` = sampled by MPNN, `false` = held fixed or missing backbone). */
  willDesign: boolean;
  /** True when required backbone atoms are finite (`structure.mask[i] > 0.5`). */
  validBackbone: boolean;
  /** True when lock status was inherited from upstream (`lockedChains` or `fixedResidueIndices`) without explicit user override. */
  inheritedLock: boolean;
  /** 1-based symmetry tie group index when residue belongs to a `tiedPositions` group. */
  tieGroupId?: number;
  /** Number of missing sequence positions immediately preceding this residue within the same chain. */
  gapBefore: number;
  /** Gap-aware positional encoding index fed to ProteinMPNN encoder. */
  residueIndex: number;
}

export interface CompiledMpnnInputSpec {
  family: ModelFamily;
  modelName: string;
  atomMode: 'backbone' | 'ca';
  numSequences: number;
  temperature: number;
  backboneNoise: number;
  seed: number;
  ignoreInheritedLocks: boolean;
  allChains: string[];
  chainLengths: number[];
  lockedChains: string[];
  inheritedLockedChains: string[];
  userLockedChains: string[];
  designChains: string[] | undefined;
  fixedPositions: ResidueRef[];
  tiedPositions: ResidueRef[][];
  omitAminoAcids: string | undefined;
  gapAwareIndices: Int32Array;
  structure: MpnnStructure;
  residues: MpnnResidueSpec[];
  designedCount: number;
  fixedCount: number;
  missingBackboneCount: number;
  warnings: string[];
  errors: string[];
}

function tryResolveResidueIndex(s: MpnnStructure, ref: ResidueRef): number | null {
  if (typeof ref === 'number') {
    return Number.isInteger(ref) && ref >= 0 && ref < s.length ? ref : null;
  }
  const ins = ref.insertionCode ?? '';
  for (let i = 0; i < s.length; i++) {
    if (
      s.chainIds[i] === ref.chain &&
      s.residueNumbers[i] === ref.number &&
      s.insertionCodes[i] === ins
    ) {
      return i;
    }
  }
  return null;
}

function formatRefLabel(ref: ResidueRef): string {
  if (typeof ref === 'number') return `#${ref + 1}`;
  return `${ref.chain}${ref.number}${ref.insertionCode ?? ''}`;
}

/**
 * Pure synchronous compiler for ProteinMPNN inputs.
 * Computes the exact post-precedence per-residue designability mask, inherited locks,
 * symmetry tie groups, gap-aware indices, omitted amino acids, and validation diagnostics
 * identical to `mpnn.worker.ts` + `proteinmpnn-web/src/constraints.ts`.
 */
export function compileMpnnInputSpec(
  structure: Pick<StructureArtifact, 'pdbText' | 'lockedChains' | 'fixedResidueIndices' | 'chainRoles'>,
  params: Record<string, unknown> = {},
  parsedOverride?: MpnnStructure,
): CompiledMpnnInputSpec {
  const family = (params.family ?? 'solublempnn') as ModelFamily;
  const modelName = String(params.modelName ?? 'v_48_020');
  const atomMode: 'backbone' | 'ca' = parsedOverride?.atomMode ?? (family === 'ca' ? 'ca' : 'backbone');
  const numSequences = Math.max(1, Math.min(8, Math.round(Number(params.numSequences ?? 2))));
  const temperature = Math.max(0.01, Math.min(1.5, Number(params.temperature ?? 0.1)));
  const rawNoise = Number(params.backboneNoise ?? 0);
  const backboneNoise = Number.isFinite(rawNoise) ? Math.max(0, Math.min(1, rawNoise)) : 0;
  const seed = Number(params.seed ?? 42);
  const fixedPositionsSpec = String(params.fixedPositions ?? '').trim();
  const omitAminoAcidsRaw = String(params.omitAminoAcids ?? '')
    .trim()
    .toUpperCase()
    .replace(/[^ACDEFGHIKLMNPQRSTVWYX]/g, '');
  const omitAminoAcids = omitAminoAcidsRaw
    ? Array.from(new Set((omitAminoAcidsRaw + 'X').split(''))).join('')
    : undefined;
  const ignoreInheritedLocks = Boolean(params.unlockInherited || params.ignoreInheritedLocks);

  const explicitDesignChains: string[] | undefined = Array.isArray(params.designChains)
    ? (params.designChains as string[]).map((c) => String(c).trim().toUpperCase()).filter(Boolean)
    : typeof params.designChains === 'string' && params.designChains.trim()
      ? (params.designChains as string)
          .split(/[,\s]+/)
          .map((c: string) => c.trim().toUpperCase())
          .filter(Boolean)
      : undefined;

  const mpnnStructure =
    parsedOverride ??
    parsePDB(structure.pdbText, {
      fillGaps: false,
      atomMode,
    });
  const gapAwareIndices = buildGapAwareResidueIndices(mpnnStructure);
  const defaultChain = mpnnStructure.chainIds[0] ?? 'A';
  const allChains = [...new Set(mpnnStructure.chainIds)];
  const presentChainSet = new Set(allChains);
  const chainLengths = computeChainLengths(mpnnStructure.chainIds);

  const warnings: string[] = [];
  const errors: string[] = [];

  const { fixedPositions: userFixed, lockedChains: userLockedChainsRaw } =
    parseFixedPositionsAndChains(fixedPositionsSpec, defaultChain);
  const tiedPositions = parseTiedPositions(params.tiedPositions, defaultChain);

  const lockedChainSet = new Set<string>();
  const userLockedChainSet = new Set<string>();
  const inheritedLockedChainSet = new Set<string>();

  for (const ch of userLockedChainsRaw) {
    if (presentChainSet.has(ch)) {
      lockedChainSet.add(ch);
      userLockedChainSet.add(ch);
    } else {
      warnings.push(`Fixed chain "${ch}" is not present in structure (available: ${allChains.join(', ')}).`);
    }
  }
  if (!ignoreInheritedLocks && !explicitDesignChains && Array.isArray(structure.lockedChains)) {
    for (const ch of structure.lockedChains) {
      const upper = String(ch).trim().toUpperCase();
      if (presentChainSet.has(upper)) {
        lockedChainSet.add(upper);
        inheritedLockedChainSet.add(upper);
      }
    }
  }

  let designChains: string[] | undefined;
  if (explicitDesignChains) {
    const unknownDesignChains = explicitDesignChains.filter((ch) => !presentChainSet.has(ch));
    if (unknownDesignChains.length > 0) {
      warnings.push(
        `designChains includes chain(s) not present in structure: ${unknownDesignChains.join(', ')} (available: ${allChains.join(', ')}).`,
      );
    }
    designChains = explicitDesignChains.filter((ch) => presentChainSet.has(ch));
  } else if (lockedChainSet.size > 0) {
    designChains = allChains.filter((ch) => !lockedChainSet.has(ch));
  } else {
    designChains = undefined;
  }

  // Combine explicit fixedPositions with inherited 1-based fixedResidueIndices (e.g. motif residues)
  const fixedPositions: ResidueRef[] = [...userFixed];
  const userFixedIndices = new Set<number>();
  const inheritedFixedIndices = new Set<number>();

  for (const ref of userFixed) {
    const idx = tryResolveResidueIndex(mpnnStructure, ref);
    if (idx === null) {
      errors.push(`Fixed position ${formatRefLabel(ref)} was not found in the input structure.`);
    } else {
      userFixedIndices.add(idx);
    }
  }

  if (!ignoreInheritedLocks && Array.isArray(structure.fixedResidueIndices)) {
    for (const oneBasedIdx of structure.fixedResidueIndices) {
      const zeroBased = Number(oneBasedIdx) - 1;
      if (Number.isInteger(zeroBased) && zeroBased >= 0 && zeroBased < mpnnStructure.length) {
        fixedPositions.push(zeroBased);
        inheritedFixedIndices.add(zeroBased);
      }
    }
  }

  // Compute initial per-residue design mask before tied-group pinning
  const initialWillDesign = new Array<boolean>(mpnnStructure.length);
  const initialInheritedLock = new Array<boolean>(mpnnStructure.length);
  for (let i = 0; i < mpnnStructure.length; i++) {
    const chain = mpnnStructure.chainIds[i];
    const validBackbone = mpnnStructure.mask[i] > 0.5;
    const designChainAllowed = designChains ? designChains.includes(chain) : true;
    const isUserFixed = userFixedIndices.has(i);
    const isInheritedFixedRes = inheritedFixedIndices.has(i);
    const isUserLockedChain = !explicitDesignChains && userLockedChainSet.has(chain);
    const isInheritedLockedChain =
      !explicitDesignChains && !userLockedChainSet.has(chain) && inheritedLockedChainSet.has(chain);
    const isExplicitlyExcludedByDesignChains = Boolean(explicitDesignChains) && !designChainAllowed;

    const willDesign = validBackbone && designChainAllowed && !isUserFixed && !isInheritedFixedRes;
    const inheritedLock =
      !willDesign &&
      validBackbone &&
      !isUserFixed &&
      !isUserLockedChain &&
      !isExplicitlyExcludedByDesignChains &&
      (isInheritedLockedChain || isInheritedFixedRes);

    initialWillDesign[i] = willDesign;
    initialInheritedLock[i] = inheritedLock;
  }

  // Resolve tied-position groups & detect overlaps / conflicting fixed native AAs (exact constraints.ts parity)
  const tieGroupByIndex = new Map<number, number>();
  const tiedPinnedByFixed = new Set<number>();
  const tiedPinnedInherited = new Set<number>();

  for (let gIdx = 0; gIdx < tiedPositions.length; gIdx++) {
    const groupRefs = tiedPositions[gIdx];
    const resolvedIndices: number[] = [];
    for (const ref of groupRefs) {
      const idx = tryResolveResidueIndex(mpnnStructure, ref);
      if (idx === null) {
        errors.push(`Tied position ${formatRefLabel(ref)} was not found in the input structure.`);
      } else {
        resolvedIndices.push(idx);
      }
    }
    if (resolvedIndices.length < 2) continue;

    let hasOverlap = false;
    const localSeen = new Set<number>();
    for (const idx of resolvedIndices) {
      if (tieGroupByIndex.has(idx) || localSeen.has(idx)) {
        hasOverlap = true;
      }
      localSeen.add(idx);
    }
    if (hasOverlap) {
      errors.push(`Tied position group #${gIdx + 1} overlaps or contains duplicate residues.`);
      continue;
    }

    for (const idx of resolvedIndices) {
      tieGroupByIndex.set(idx, gIdx + 1);
    }

    const fixedMembers = resolvedIndices.filter((idx) => !initialWillDesign[idx]);
    if (fixedMembers.length > 0) {
      const nativeAa = mpnnStructure.sequence[fixedMembers[0]];
      if (fixedMembers.some((idx) => mpnnStructure.sequence[idx] !== nativeAa)) {
        errors.push(
          `Tied position group #${gIdx + 1} contains fixed residues with conflicting native amino acids.`,
        );
      }
      if (resolvedIndices.some((idx) => !(mpnnStructure.mask[idx] > 0.5))) {
        errors.push(`Tied position group #${gIdx + 1} contains residues with missing backbone coordinates.`);
      }
      const allFixedInherited = fixedMembers.every((idx) => initialInheritedLock[idx]);
      for (const idx of resolvedIndices) {
        if (initialWillDesign[idx]) {
          tiedPinnedByFixed.add(idx);
          if (allFixedInherited) tiedPinnedInherited.add(idx);
        }
      }
    }
  }

  // Assemble final per-residue specs
  const residues: MpnnResidueSpec[] = [];
  let designedCount = 0;
  let fixedCount = 0;
  let missingBackboneCount = 0;

  for (let i = 0; i < mpnnStructure.length; i++) {
    const chain = mpnnStructure.chainIds[i];
    const resNum = mpnnStructure.residueNumbers[i];
    const insCode = mpnnStructure.insertionCodes[i] ?? '';
    const aa1 = mpnnStructure.sequence[i] ?? 'X';
    const validBackbone = mpnnStructure.mask[i] > 0.5;
    const pinnedByTie = tiedPinnedByFixed.has(i);
    const willDesign = initialWillDesign[i] && !pinnedByTie;
    const inheritedLock = initialInheritedLock[i] || tiedPinnedInherited.has(i);
    const tieGroupId = tieGroupByIndex.get(i);

    let gapBefore = 0;
    if (i > 0 && mpnnStructure.chainIds[i - 1] === chain) {
      const delta = resNum - mpnnStructure.residueNumbers[i - 1] - 1;
      if (delta > 0) gapBefore = delta;
    }

    if (!validBackbone) missingBackboneCount++;
    else if (willDesign) designedCount++;
    else fixedCount++;

    residues.push({
      flatIndex: i,
      chain,
      resNum,
      insCode,
      aa1,
      willDesign,
      validBackbone,
      inheritedLock,
      ...(tieGroupId !== undefined ? { tieGroupId } : {}),
      gapBefore,
      residueIndex: gapAwareIndices[i],
    });
  }

  return {
    family,
    modelName,
    atomMode,
    numSequences,
    temperature,
    backboneNoise,
    seed,
    ignoreInheritedLocks,
    allChains,
    chainLengths,
    lockedChains: [...lockedChainSet],
    inheritedLockedChains: [...inheritedLockedChainSet],
    userLockedChains: [...userLockedChainSet],
    designChains,
    fixedPositions,
    tiedPositions,
    omitAminoAcids,
    gapAwareIndices,
    structure: mpnnStructure,
    residues,
    designedCount,
    fixedCount,
    missingBackboneCount,
    warnings,
    errors,
  };
}
