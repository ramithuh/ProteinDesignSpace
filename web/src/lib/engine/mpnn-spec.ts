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

export interface OrderedStructureResidue {
  chain: string;
  number: number;
  insertionCode?: string;
}

/**
 * Parse human-friendly fixed-position strings into `{ fixedPositions, lockedChains }`.
 * Supports: whole-chain locking ("A", "B"), single residues ("A10", "A100A", "100A"),
 * ranges ("A10-15", "A10-A15", "H100A-100C", reversed "A15-10"), negative resids ("A-2", "A-2-5", "A-5--2"),
 * lowercase chain normalization ("a10" -> "A"), and default chain resolution ("_" for blank-chain PDBs).
 *
 * When `orderedResidues` is provided, ranges expand against actual structure residues in order
 * (capturing Kabat/IMGT insertion codes such as `H82A`, `H82B` inside `H39-94`).
 * When `orderedResidues` is omitted, preserves exact integer expansion for backward compatibility.
 */
export function parseFixedPositionsAndChains(
  spec: string,
  defaultChain = 'A',
  orderedResidues?: readonly OrderedStructureResidue[],
): { fixedPositions: ResidueRef[]; lockedChains: string[] } {
  const fixedPositions: ResidueRef[] = [];
  const lockedChains: string[] = [];
  const seenChains = new Set<string>();
  const trimmed = spec.trim();
  if (!trimmed) return { fixedPositions, lockedChains };

  const resolveStructureChain = (requested: string): string => {
    if (!orderedResidues || orderedResidues.length === 0) return requested;
    if (orderedResidues.some((r) => r.chain === requested)) return requested;
    const upper = requested.toUpperCase();
    const hit = orderedResidues.find((r) => r.chain.toUpperCase() === upper);
    return hit ? hit.chain : requested;
  };

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

    const rangeMatch =
      /^(?:([A-Za-z0-9_]+):|([A-Za-z_]))?(-?\d+)([A-Za-z])?-(?:([A-Za-z0-9_]+):|([A-Za-z_]))?(-?\d+)([A-Za-z])?$/.exec(raw);
    if (rangeMatch) {
      const rawStartChain = rangeMatch[1] ?? rangeMatch[2];
      const rawEndChain = rangeMatch[5] ?? rangeMatch[6];
      const startChain = rawStartChain ? rawStartChain.toUpperCase() : null;
      const endChain = rawEndChain ? rawEndChain.toUpperCase() : null;
      if (startChain && endChain && startChain !== endChain) {
        throw new Error(
          `Invalid fixed position "${raw}": range must stay within a single chain (${startChain} vs ${endChain})`,
        );
      }
      const chain = resolveStructureChain(startChain ?? endChain ?? defaultChain);
      const a = Number(rangeMatch[3]);
      const aIns = (rangeMatch[4] ?? '').toUpperCase();
      const b = Number(rangeMatch[7]);
      const bIns = (rangeMatch[8] ?? '').toUpperCase();
      const forward = a < b || (a === b && aIns <= bIns);
      const start = forward ? a : b;
      const startIns = forward ? aIns : bIns;
      const end = forward ? b : a;
      const endIns = forward ? bIns : aIns;
      if (end - start > 100000) {
        throw new Error(`Invalid fixed position "${raw}": residue range too large`);
      }

      if (orderedResidues && orderedResidues.length > 0) {
        const hits = orderedResidues.filter((r) => {
          if (r.chain !== chain && r.chain.toUpperCase() !== chain.toUpperCase()) return false;
          const rIns = (r.insertionCode ?? '').toUpperCase();
          if (start === end && !startIns && !endIns) {
            return r.number === start && rIns === '';
          }
          const afterStart =
            r.number > start || (r.number === start && (!startIns || rIns >= startIns));
          const beforeEnd =
            r.number < end || (r.number === end && (!endIns || rIns <= endIns));
          return afterStart && beforeEnd;
        });
        if (hits.length > 0) {
          for (const h of hits) {
            fixedPositions.push(
              h.insertionCode
                ? { chain: h.chain, number: h.number, insertionCode: h.insertionCode }
                : { chain: h.chain, number: h.number },
            );
          }
          continue;
        }
        // Fallback when range matches 0 residues so downstream resolution reports missing residues
        if (startIns || endIns) {
          fixedPositions.push(
            startIns ? { chain, number: start, insertionCode: startIns } : { chain, number: start },
          );
          if (start !== end || startIns !== endIns) {
            fixedPositions.push(
              endIns ? { chain, number: end, insertionCode: endIns } : { chain, number: end },
            );
          }
          continue;
        }
      }

      if (startIns || endIns) {
        if (start === end) {
          const loChar = startIns ? startIns.charCodeAt(0) : 64;
          const hiChar = endIns ? endIns.charCodeAt(0) : 90;
          if (!startIns) fixedPositions.push({ chain, number: start });
          for (let code = Math.max(65, loChar); code <= Math.min(90, hiChar); code++) {
            fixedPositions.push({ chain, number: start, insertionCode: String.fromCharCode(code) });
          }
        } else {
          fixedPositions.push(
            startIns ? { chain, number: start, insertionCode: startIns } : { chain, number: start },
          );
          for (let pos = start + 1; pos < end; pos++) {
            fixedPositions.push({ chain, number: pos });
          }
          fixedPositions.push(
            endIns ? { chain, number: end, insertionCode: endIns } : { chain, number: end },
          );
        }
        continue;
      }

      for (let pos = start; pos <= end; pos++) {
        fixedPositions.push({ chain, number: pos });
      }
      continue;
    }

    const singleMatch = /^(?:([A-Za-z0-9_]+):|([A-Za-z_]))?(-?\d+)([A-Za-z])?$/.exec(raw);
    if (singleMatch) {
      const rawChain = singleMatch[1] ?? singleMatch[2];
      const chain = resolveStructureChain(rawChain ? rawChain.toUpperCase() : defaultChain);
      const number = Number(singleMatch[3]);
      const insertionCode = singleMatch[4]?.toUpperCase();
      fixedPositions.push(insertionCode ? { chain, number, insertionCode } : { chain, number });
      continue;
    }

    const matchedMultiChain =
      orderedResidues &&
      /^[A-Za-z0-9_]+$/.test(raw) &&
      orderedResidues.find((r) => r.chain.toUpperCase() === raw.toUpperCase());
    if (matchedMultiChain) {
      const ch = matchedMultiChain.chain;
      if (!seenChains.has(ch)) {
        seenChains.add(ch);
        lockedChains.push(ch);
      }
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
export function parseTiedPositions(spec: unknown, _defaultChain = 'A'): ResidueRef[][] {
  if (Array.isArray(spec)) {
    return spec as ResidueRef[][];
  }
  const text = String(spec ?? '').trim();
  if (!text) return [];
  const groups: ResidueRef[][] = [];
  const rawGroups = text.split(/[;,]+/);
  const seenResidues = new Set<string>();
  for (const rawGroup of rawGroups) {
    const groupText = rawGroup.trim();
    if (!groupText) continue;
    if (groupText.includes('=') && groupText.split('=').some((part) => !part.trim())) {
      throw new Error(`Invalid tied position group "${groupText}" (empty side around "="; expected e.g. "A10=B10").`);
    }
    const tokens = groupText.split(/[=\s]+/).filter(Boolean);
    if (tokens.length < 2) {
      throw new Error(`Invalid tied position group "${groupText}" (expected at least 2 residues, e.g. "A10=B10").`);
    }
    const refs: ResidueRef[] = [];
    for (const tok of tokens) {
      const m = /^([A-Za-z_]):?(-?\d+)([A-Za-z])?$/.exec(tok);
      if (!m) {
        throw new Error(
          `Invalid tied position token "${tok}" in "${groupText}" (expected chain ID + residue number, e.g. "A10=B10" or "A100A=B100A").`,
        );
      }
      const chain = m[1].toUpperCase();
      const number = Number(m[2]);
      const insertionCode = m[3];
      const key = `${chain}:${number}${insertionCode ?? ''}`;
      if (seenResidues.has(key)) {
        throw new Error(
          `Duplicate or overlapping tied position residue "${tok}" in "${groupText}" (each residue may appear in at most one tie group).`,
        );
      }
      seenResidues.add(key);
      refs.push(insertionCode ? { chain, number, insertionCode } : { chain, number });
    }
    groups.push(refs);
  }
  if (groups.length === 0) {
    throw new Error(`Invalid tied position group "${text}" (expected at least 2 residues, e.g. "A10=B10").`);
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
  structure: Pick<
    StructureArtifact,
    'pdbText' | 'lockedChains' | 'fixedResidueIndices' | 'tiedPositions' | 'chainRoles'
  >,
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

  const orderedResidues: OrderedStructureResidue[] = [];
  for (let i = 0; i < mpnnStructure.length; i++) {
    orderedResidues.push({
      chain: mpnnStructure.chainIds[i],
      number: mpnnStructure.residueNumbers[i],
      insertionCode: mpnnStructure.insertionCodes[i] ?? '',
    });
  }

  const { fixedPositions: userFixed, lockedChains: userLockedChainsRaw } =
    parseFixedPositionsAndChains(fixedPositionsSpec, defaultChain, orderedResidues);
  const rawTiedSpec = Array.isArray(params.tiedPositions)
    ? params.tiedPositions
    : typeof params.tiedPositions === 'string' && params.tiedPositions.trim()
      ? params.tiedPositions
      : !ignoreInheritedLocks &&
          typeof structure.tiedPositions === 'string' &&
          structure.tiedPositions.trim()
        ? structure.tiedPositions
        : '';
  const tiedPositions = parseTiedPositions(rawTiedSpec, defaultChain);

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

// ---------------------------------------------------------------------------
// Shared AntiFold / ESM-IF1 Antibody Numbering & VH/VL Chain Detection
// ---------------------------------------------------------------------------

export type AntiFoldNumberingParam = 'auto' | 'imgt' | 'kabat';
export type ResolvedAntiFoldNumbering = 'imgt' | 'kabat';

export const ANTIBODY_CDR_BOUNDS: Record<
  ResolvedAntiFoldNumbering,
  {
    H: readonly [readonly [number, number], readonly [number, number], readonly [number, number]];
    L: readonly [readonly [number, number], readonly [number, number], readonly [number, number]];
    maxH: number;
    maxL: number;
  }
> = {
  imgt: {
    H: [
      [27, 38],
      [56, 65],
      [105, 117],
    ],
    L: [
      [27, 38],
      [56, 65],
      [105, 117],
    ],
    maxH: 128,
    maxL: 128,
  },
  kabat: {
    H: [
      [31, 35],
      [50, 65],
      [95, 102],
    ],
    L: [
      [24, 34],
      [50, 56],
      [89, 97],
    ],
    maxH: 113,
    maxL: 107,
  },
};

export interface NumberedChainInfo {
  id: string;
  residues: ReadonlyArray<{ number: number; insertion: string; aa1: string }>;
}

export function normalizeAntiFoldNumberingParam(raw: unknown): AntiFoldNumberingParam {
  const v = String(raw ?? 'auto')
    .trim()
    .toLowerCase();
  if (v === 'imgt' || v === 'kabat') return v;
  return 'auto';
}

function scoreChainAntibodyHallmarks(chain: NumberedChainInfo) {
  const byNum = new Map<number, string>();
  let hasKabatHIns = false; // 82A/B/C or 52A or 100A
  let hasKabatLIns = false; // 27A or 95A
  let hasImgtIns = false; // 111A / 112A
  for (const r of chain.residues) {
    const ins = r.insertion.toUpperCase();
    if (!ins && !byNum.has(r.number)) byNum.set(r.number, r.aa1.toUpperCase());
    if (ins) {
      if ((r.number === 82 || r.number === 52 || r.number === 100) && ins >= 'A' && ins <= 'Z') {
        hasKabatHIns = true;
      }
      if ((r.number === 27 || r.number === 95) && ins >= 'A' && ins <= 'Z') {
        hasKabatLIns = true;
      }
      if ((r.number === 111 || r.number === 112) && ins >= 'A' && ins <= 'Z') {
        hasImgtIns = true;
      }
    }
  }
  const kabatVH =
    (byNum.get(22) === 'C' ? 1 : 0) +
    (byNum.get(36) === 'W' ? 1 : 0) +
    (byNum.get(92) === 'C' ? 1 : 0) +
    (byNum.get(103) === 'W' ? 1 : 0) +
    (hasKabatHIns ? 2 : 0);
  const kabatVL =
    (byNum.get(23) === 'C' ? 1 : 0) +
    (byNum.get(35) === 'W' ? 1 : 0) +
    (byNum.get(88) === 'C' ? 1 : 0) +
    (byNum.get(98) === 'F' || byNum.get(98) === 'W' ? 1 : 0) +
    (hasKabatLIns ? 2 : 0);
  const imgt =
    (byNum.get(23) === 'C' ? 1 : 0) +
    (byNum.get(41) === 'W' ? 1 : 0) +
    (byNum.get(104) === 'C' ? 1 : 0) +
    (byNum.get(118) === 'W' || byNum.get(118) === 'F' ? 1 : 0) +
    (hasImgtIns ? 2 : 0);
  return { kabatVH, kabatVL, imgt };
}

export interface IgSequenceHallmarkResult {
  role: 'H' | 'L' | 'unknown';
  score: number;
  c1Idx: number;
  wIdx: number;
  c2Idx: number;
  jIdx: number;
  hasCanonicalNumbering: boolean;
}

/**
 * Numbering-agnostic Ig variable-domain motif detector (FR1 Cys -> FR2 Trp -> FR3 Cys -> FR4 WGxG/FGxG).
 * Reliably identifies VH/VHH vs VL even on sequentially numbered RCSB entries (e.g. 1N8Z, 1MEL)
 * and rejects non-Ig antigens that happen to carry chain ID "L" (e.g. Lysozyme in 1MEL).
 */
export function scoreIgSequenceHallmarks(chain: NumberedChainInfo): IgSequenceHallmarkResult {
  const n = Math.min(chain.residues.length, 155);
  const empty: IgSequenceHallmarkResult = {
    role: 'unknown',
    score: 0,
    c1Idx: -1,
    wIdx: -1,
    c2Idx: -1,
    jIdx: -1,
    hasCanonicalNumbering: false,
  };
  if (n < 85) return empty;
  const seq = chain.residues
    .slice(0, n)
    .map((r) => (r.aa1 || 'X').toUpperCase())
    .join('');

  let best: IgSequenceHallmarkResult = empty;
  for (let c1 = 14; c1 <= Math.min(34, n - 70); c1++) {
    if (seq[c1] !== 'C') continue;
    for (let w = c1 + 10; w <= Math.min(c1 + 20, n - 55); w++) {
      if (seq[w] !== 'W') continue;
      const fr2Trip = seq.slice(w, w + 3);
      const isHFr2 = /^W[VFILY][RQK]$/.test(fr2Trip);
      const isLFr2 = /^W[YFL][QLH]$/.test(fr2Trip);
      for (let c2 = w + 44; c2 <= Math.min(w + 72, n - 8); c2++) {
        if (seq[c2] !== 'C') continue;
        const fr3Pre = c2 >= 2 ? seq.slice(c2 - 2, c2 + 1) : '';
        const hasYyc = /^[YFHT][YFVIALM]C$/.test(fr3Pre);
        for (let j = c2 + 5; j <= Math.min(c2 + 30, n - 4); j++) {
          const j4 = seq.slice(j, j + 4);
          const isH = j4[0] === 'W' && j4[1] === 'G' && j4[3] === 'G';
          const isL = j4[0] === 'F' && j4[1] === 'G' && j4[3] === 'G';
          if (!isH && !isL) continue;
          let score = 4; // C1 + W + C2 + WGxG/FGxG with canonical spacing
          if (hasYyc) score += 1;
          if ((isH && isHFr2) || (isL && isLFr2)) score += 1;
          const rC1 = chain.residues[c1]?.number;
          const rW = chain.residues[w]?.number;
          const rC2 = chain.residues[c2]?.number;
          const rJ = chain.residues[j]?.number;
          const kabatHits = isH
            ? (rC1 === 22 ? 1 : 0) + (rW === 36 ? 1 : 0) + (rC2 === 92 ? 1 : 0) + (rJ === 103 ? 1 : 0)
            : (rC1 === 23 ? 1 : 0) + (rW === 35 ? 1 : 0) + (rC2 === 88 ? 1 : 0) + (rJ === 98 ? 1 : 0);
          const imgtHits =
            (rC1 === 23 ? 1 : 0) + (rW === 41 ? 1 : 0) + (rC2 === 104 ? 1 : 0) + (rJ === 118 ? 1 : 0);
          const hasCanonicalNumbering = kabatHits >= 3 || imgtHits >= 3;
          if (score > best.score) {
            best = {
              role: isH ? 'H' : 'L',
              score,
              c1Idx: c1,
              wIdx: w,
              c2Idx: c2,
              jIdx: j,
              hasCanonicalNumbering,
            };
          }
        }
      }
    }
  }
  return best;
}

export function detectAntibodyHeavyLightChains(
  chains: readonly NumberedChainInfo[],
  chainTypes: Readonly<Record<string, string>>,
  lockedChainSet: ReadonlySet<string>,
  explicitTarget?: string,
): { heavyChain?: string; lightChain?: string } {
  const byId = new Map(chains.map((c) => [c.id, c]));
  const upperToId = new Map<string, string>();
  for (const c of chains) if (!upperToId.has(c.id.toUpperCase())) upperToId.set(c.id.toUpperCase(), c.id);
  const resolve = (k: string) => (byId.has(k) ? k : upperToId.get(k.toUpperCase()));

  const numScores = new Map(chains.map((c) => [c.id, scoreChainAntibodyHallmarks(c)]));
  const seqScores = new Map(chains.map((c) => [c.id, scoreIgSequenceHallmarks(c)]));
  const anyStrongIg = chains.some((c) => {
    const ss = seqScores.get(c.id)!;
    const ns = numScores.get(c.id)!;
    return ss.score >= 4 || Math.max(ns.kabatVH, ns.kabatVL, ns.imgt) >= 3;
  });

  // 1. Explicit molecularContext chainTypes take highest precedence
  let heavyChain = chains.find(
    (c) => chainTypes[c.id] === 'antibody-H' || chainTypes[c.id] === 'nanobody',
  )?.id;
  let lightChain = chains.find(
    (c) => chainTypes[c.id] === 'antibody-L' && c.id !== (explicitTarget ?? heavyChain),
  )?.id;

  // 2. Ig sequence + numbering hallmarks across all chains (handles 1N8Z, 1MEL, and locked-H 6W41)
  if (!heavyChain) {
    let bestHScore = 0;
    for (const c of chains) {
      const ns = numScores.get(c.id)!;
      const ss = seqScores.get(c.id)!;
      const isSeqH = ss.role === 'H' && ss.score >= 4;
      const isNumH = Math.max(ns.kabatVH, ns.imgt) >= 3 && ns.kabatVH >= ns.kabatVL && ss.role !== 'L';
      if (!isSeqH && !isNumH) continue;
      const combined = (isSeqH ? ss.score * 10 : 0) + Math.max(ns.kabatVH, ns.imgt) + (c.id.toUpperCase() === 'H' ? 0.5 : 0);
      if (combined > bestHScore) {
        bestHScore = combined;
        heavyChain = c.id;
      }
    }
  }
  if (!lightChain) {
    let bestLScore = 0;
    for (const c of chains) {
      if (c.id === (explicitTarget ?? heavyChain)) continue;
      const ns = numScores.get(c.id)!;
      const ss = seqScores.get(c.id)!;
      const isSeqL = ss.role === 'L' && ss.score >= 4;
      const isNumL = ns.kabatVL >= 3 && ns.kabatVL > ns.kabatVH && ss.role !== 'H';
      if (!isSeqL && !isNumL) continue;
      const combined = (isSeqL ? ss.score * 10 : 0) + ns.kabatVL + (c.id.toUpperCase() === 'L' ? 0.5 : 0);
      if (combined > bestLScore) {
        bestLScore = combined;
        lightChain = c.id;
      }
    }
  }

  // 3. Fallback to conventional H / L chain IDs only when safe (never mistake an antigen chain "L"
  //    like Lysozyme in 1MEL for a light chain when real Ig hallmarks exist elsewhere in the complex).
  if (!heavyChain) {
    const h = resolve('H');
    if (h) {
      const ss = seqScores.get(h)!;
      const ns = numScores.get(h)!;
      if (!anyStrongIg || ss.score >= 3 || Math.max(ns.kabatVH, ns.imgt) >= 2) {
        heavyChain = h;
      }
    }
  }
  if (!lightChain) {
    const l = resolve('L');
    if (l && l !== (explicitTarget ?? heavyChain)) {
      const ss = seqScores.get(l)!;
      const ns = numScores.get(l)!;
      if (!anyStrongIg || ss.score >= 3 || Math.max(ns.kabatVL, ns.imgt) >= 2) {
        lightChain = l;
      }
    }
  }

  return { heavyChain, lightChain };
}

/**
 * When an RCSB antibody uses sequential 1..N numbering instead of IMGT/Kabat (e.g. 1N8Z),
 * translate symbolic CDR/FW region names (CDR1-3, CDRH1-3, CDRL1-3, FW1-4, all) into explicit
 * `CHAIN:start-end` ranges anchored on the conserved Ig Cys/Trp/WGxG/FGxG motif positions.
 */
export function remapSequentialAntibodyRegions(
  regions: readonly string[],
  chains: readonly NumberedChainInfo[],
  heavyChain: string | undefined,
  lightChain: string | undefined,
  numberingParam: AntiFoldNumberingParam,
): { regions: string[]; remappedChains: string[] } {
  if (numberingParam !== 'auto' || regions.length === 0) {
    return { regions: [...regions], remappedChains: [] };
  }
  const byId = new Map(chains.map((c) => [c.id, c]));
  type AnchorBounds = {
    chain: NumberedChainInfo;
    cdrs: Array<[number, number]>;
    maxIdx: number;
  };
  const seqAnchored = new Map<'H' | 'L', AnchorBounds>();
  for (const [role, chId] of [
    ['H', heavyChain],
    ['L', lightChain],
  ] as const) {
    if (!chId) continue;
    const ch = byId.get(chId);
    if (!ch) continue;
    const h = scoreIgSequenceHallmarks(ch);
    if (h.score >= 4 && !h.hasCanonicalNumbering) {
      const n = ch.residues.length;
      const cdr1: [number, number] =
        role === 'H'
          ? [Math.min(h.c1Idx + 4, h.wIdx - 1), h.wIdx - 1]
          : [Math.min(h.c1Idx + 1, h.wIdx - 1), h.wIdx - 1];
      const cdr2Start = role === 'H' ? h.wIdx + 14 : h.wIdx + 15;
      const cdr2End = Math.max(cdr2Start + 5, role === 'H' ? h.c2Idx - 31 : h.c2Idx - 32);
      const cdr2: [number, number] = [cdr2Start, Math.min(cdr2End, h.c2Idx - 1)];
      const cdr3Start = role === 'H' ? h.c2Idx + 3 : h.c2Idx + 1;
      const cdr3: [number, number] = [Math.min(cdr3Start, h.jIdx - 1), h.jIdx - 1];
      const maxIdx = Math.min(n - 1, h.jIdx + (role === 'H' ? 10 : 9));
      seqAnchored.set(role, { chain: ch, cdrs: [cdr1, cdr2, cdr3], maxIdx });
    }
  }
  if (seqAnchored.size === 0) {
    return { regions: [...regions], remappedChains: [] };
  }

  const fmtSpan = (ch: NumberedChainInfo, aIdx: number, bIdx: number): string | null => {
    const lo = Math.max(0, Math.min(aIdx, bIdx));
    const hi = Math.min(ch.residues.length - 1, Math.max(aIdx, bIdx));
    const ra = ch.residues[lo];
    const rb = ch.residues[hi];
    if (!ra || !rb) return null;
    const aTok = `${ra.number}${ra.insertion || ''}`;
    const bTok = `${rb.number}${rb.insertion || ''}`;
    return aTok === bTok ? aTok : `${aTok}-${bTok}`;
  };

  const out: string[] = [];
  const usedChains = new Set<string>();
  for (const reg of regions) {
    const m = /^(all|CDR|FW)([HL]?)([1-4]?)$/.exec(reg);
    if (!m) {
      out.push(reg);
      continue;
    }
    const [, type, roleSpec, numStr] = m;
    const targetRoles: Array<'H' | 'L'> =
      roleSpec === 'H' ? ['H'] : roleSpec === 'L' ? ['L'] : ['H', 'L'];
    let emittedAny = false;
    for (const r of targetRoles) {
      const anchor = seqAnchored.get(r);
      const chId = r === 'H' ? heavyChain : lightChain;
      if (!chId) continue;
      if (!anchor) {
        // Keep canonical IMGT/Kabat token scoped to the non-sequential chain
        out.push(`${type}${r}${numStr}`);
        emittedAny = true;
        continue;
      }
      usedChains.add(chId);
      const spans: string[] = [];
      if (type === 'all') {
        const sp = fmtSpan(anchor.chain, 0, anchor.maxIdx);
        if (sp) spans.push(sp);
      } else if (type === 'CDR') {
        const list = numStr ? [anchor.cdrs[Number(numStr) - 1]] : anchor.cdrs;
        for (const pair of list) {
          if (!pair) continue;
          const sp = fmtSpan(anchor.chain, pair[0], pair[1]);
          if (sp) spans.push(sp);
        }
      } else {
        const fws: Array<[number, number]> = [
          [0, anchor.cdrs[0][0] - 1],
          [anchor.cdrs[0][1] + 1, anchor.cdrs[1][0] - 1],
          [anchor.cdrs[1][1] + 1, anchor.cdrs[2][0] - 1],
          [anchor.cdrs[2][1] + 1, anchor.maxIdx],
        ];
        const list = numStr ? [fws[Number(numStr) - 1]] : fws;
        for (const pair of list) {
          if (!pair || pair[1] < pair[0]) continue;
          const sp = fmtSpan(anchor.chain, pair[0], pair[1]);
          if (sp) spans.push(sp);
        }
      }
      if (spans.length > 0) {
        out.push(`${chId}:${spans.join(',')}`);
        emittedAny = true;
      }
    }
    if (!emittedAny) out.push(reg);
  }
  return { regions: out, remappedChains: [...usedChains] };
}

export function resolveAntibodyNumbering(
  param: AntiFoldNumberingParam,
  chains: readonly NumberedChainInfo[],
  heavyChain?: string,
  lightChain?: string,
): ResolvedAntiFoldNumbering {
  if (param === 'imgt' || param === 'kabat') return param;
  const targets = chains.filter((c) => c.id === heavyChain || c.id === lightChain);
  let kabatTotal = 0;
  let imgtTotal = 0;
  for (const c of targets.length > 0 ? targets : chains) {
    const s = scoreChainAntibodyHallmarks(c);
    kabatTotal += c.id === lightChain ? s.kabatVL : Math.max(s.kabatVH, s.kabatVL);
    imgtTotal += s.imgt;
  }
  return kabatTotal >= 2 && kabatTotal > imgtTotal ? 'kabat' : 'imgt';
}

export interface AntiFoldDesignPlanInput {
  chainInfos: readonly NumberedChainInfo[];
  orderedResidues: readonly OrderedStructureResidue[];
  chainTypes?: Readonly<Record<string, string>>;
  inheritedLockedChains?: readonly string[];
  inheritedFixedResidueIndices?: readonly number[];
  variant: 'antifold' | 'esmif1';
  samplingMode: 'independent' | 'autoregressive';
  targetChainSpec: string;
  numberingParam: AntiFoldNumberingParam;
  regionsSpec: string;
  fixedPositionsSpec: string;
  unlockInherited: boolean;
  /** When true (worker), throws if `targetChainSpec` is absent from structure; when false (preview), records a warning. */
  strictTargetChain?: boolean;
}

export interface AntiFoldResidueDesignState {
  flatIndex: number;
  chain: string;
  number: number;
  insertion: string;
  willDesign: boolean;
  lockedByTargetChain: boolean;
  inRegion: boolean;
  isUserFixed: boolean;
  isInheritedFixed: boolean;
}

export interface CompiledAntiFoldDesignPlan {
  availableChains: string[];
  explicitTarget?: string;
  /** Passed to vendor `prepareInput` only when explicitly specified by user (prevents AR truncation on H+L). */
  effectiveTargetChain?: string;
  resolvedTargetChain: string;
  heavyChain?: string;
  lightChain?: string;
  detectedHeavy?: string;
  detectedLight?: string;
  resolvedNumbering: ResolvedAntiFoldNumbering;
  rawRegions: string[];
  effectiveRegions: string[];
  remappedChains: string[];
  combineOpts: Record<string, unknown>;
  lockedChainSet: Set<string>;
  userLockedChainSet: Set<string>;
  inheritedLockedChainSet: Set<string>;
  userFixedKeys: Set<string>;
  inheritedFixedIndices: Set<number>;
  residueStates: AntiFoldResidueDesignState[];
  designedCount: number;
  fixedCount: number;
  warnings: string[];
}

function makeAntiFoldResidueKey(chain: string, num: number, ins: string): string {
  return `${chain}:${num}:${ins}`;
}

/**
 * Pure single-source-of-truth AntiFold / ESM-IF1 design plan compiler (`#77`).
 * Encapsulates Ig sequence hallmarks, `firstNonLight` heavy/light resolution, numbering resolution,
 * `remapSequentialAntibodyRegions`, locked chains, fixed residues, and exact `regionMask` evaluation
 * so `antifold.worker.ts` and `input-preview.ts` match 1:1 on both canonical and sequentially numbered
 * RCSB entries (e.g. `1N8Z`, `1MEL`, `6W41`).
 */
export function compileAntiFoldDesignPlan(
  input: AntiFoldDesignPlanInput,
): CompiledAntiFoldDesignPlan {
  const {
    chainInfos,
    orderedResidues,
    chainTypes = {},
    inheritedLockedChains,
    inheritedFixedResidueIndices,
    variant,
    samplingMode,
    targetChainSpec,
    numberingParam,
    regionsSpec,
    fixedPositionsSpec,
    unlockInherited,
    strictTargetChain = false,
  } = input;

  const warnings: string[] = [];
  const availableChains = chainInfos.map((c) => c.id);
  const chainSet = new Set(availableChains);
  const upperToChain = new Map<string, string>();
  for (const ch of availableChains) {
    if (!upperToChain.has(ch.toUpperCase())) upperToChain.set(ch.toUpperCase(), ch);
  }
  const resolveChain = (raw: string): string | undefined => {
    if (!raw) return undefined;
    if (chainSet.has(raw)) return raw;
    return upperToChain.get(raw.toUpperCase());
  };

  const originalOrderKeys = orderedResidues.map((r) =>
    makeAntiFoldResidueKey(r.chain, r.number, r.insertionCode ?? ''),
  );

  // Pass 1: collect whole-chain locks before target/VH/VL resolution
  const defaultChain = availableChains[0] ?? 'A';
  const { lockedChains: preUserLockedChains } = parseFixedPositionsAndChains(
    fixedPositionsSpec,
    defaultChain,
    orderedResidues,
  );
  const lockedChainSet = new Set<string>();
  const userLockedChainSet = new Set<string>();
  const inheritedLockedChainSet = new Set<string>();

  for (const ch of preUserLockedChains) {
    const resolved = resolveChain(ch);
    if (resolved) {
      lockedChainSet.add(resolved);
      userLockedChainSet.add(resolved);
    }
  }
  if (!unlockInherited && Array.isArray(inheritedLockedChains)) {
    for (const rawCh of inheritedLockedChains) {
      const resolved = resolveChain(String(rawCh).trim());
      if (resolved) {
        lockedChainSet.add(resolved);
        inheritedLockedChainSet.add(resolved);
      }
    }
  }

  const trimmedTarget = targetChainSpec.trim();
  const explicitTarget = trimmedTarget ? resolveChain(trimmedTarget) : undefined;
  if (trimmedTarget && !explicitTarget) {
    const msg = `Target chain "${trimmedTarget}" not found in structure (available: ${availableChains.join(', ')}).`;
    if (strictTargetChain) {
      throw new Error(msg);
    }
    warnings.push(msg);
  }

  const { heavyChain: detectedHeavy, lightChain: detectedLight } = detectAntibodyHeavyLightChains(
    chainInfos,
    chainTypes,
    lockedChainSet,
    explicitTarget,
  );

  const unlockedHeavy =
    detectedHeavy && !lockedChainSet.has(detectedHeavy) ? detectedHeavy : undefined;
  const unlockedLight =
    detectedLight && !lockedChainSet.has(detectedLight) ? detectedLight : undefined;
  const firstUnlocked = availableChains.find((c) => !lockedChainSet.has(c)) ?? defaultChain;

  // Only set `effectiveTargetChain` when explicitly specified by user so H+L antibodies
  // design both H & L CDRs and AR mode does not truncate at H.
  const effectiveTargetChain = explicitTarget;
  const resolvedTargetChain =
    explicitTarget ??
    (variant === 'antifold' ? unlockedHeavy ?? unlockedLight ?? firstUnlocked : undefined) ??
    defaultChain;

  // Pass 2: parse chainless fixedPositions against resolvedTargetChain
  const { fixedPositions: userRefs, lockedChains: userLockedChains } = parseFixedPositionsAndChains(
    fixedPositionsSpec,
    resolvedTargetChain,
    orderedResidues,
  );
  for (const ch of userLockedChains) {
    const resolved = resolveChain(ch);
    if (resolved) {
      lockedChainSet.add(resolved);
      userLockedChainSet.add(resolved);
    }
  }

  const rawRegions =
    variant === 'antifold'
      ? regionsSpec
          .split(/[,\s]+/)
          .map((t) => t.trim())
          .filter(Boolean)
      : [];

  const firstNonLight =
    availableChains.find((c) => c !== detectedLight && !lockedChainSet.has(c)) ??
    availableChains.find((c) => c !== detectedLight);
  const heavyChain =
    variant === 'antifold'
      ? detectedHeavy ??
        explicitTarget ??
        (!detectedLight ? firstUnlocked : firstNonLight ?? firstUnlocked)
      : undefined;
  const lightChain =
    variant === 'antifold'
      ? detectedLight && detectedLight !== heavyChain
        ? detectedLight
        : explicitTarget && explicitTarget !== heavyChain
          ? explicitTarget
          : undefined
      : undefined;

  const anchorHeavy = detectedHeavy ?? (heavyChain !== detectedLight ? heavyChain : undefined);
  const anchorLight = detectedLight ?? lightChain;

  const resolvedNumbering =
    variant === 'antifold'
      ? resolveAntibodyNumbering(numberingParam, chainInfos, anchorHeavy, anchorLight)
      : 'imgt';

  const remapped =
    variant === 'antifold' && rawRegions.length > 0
      ? remapSequentialAntibodyRegions(
          rawRegions,
          chainInfos,
          anchorHeavy,
          anchorLight,
          numberingParam,
        )
      : { regions: rawRegions, remappedChains: [] };
  const effectiveRegions = remapped.regions;

  const combineOpts: Record<string, unknown> = {
    ...(effectiveTargetChain ? { targetChain: effectiveTargetChain } : {}),
    ...(heavyChain ? { heavyChain } : {}),
    ...(lightChain ? { lightChain } : {}),
    ...(variant === 'antifold' ? { numbering: resolvedNumbering } : {}),
    ...(effectiveRegions.length > 0 ? { regions: effectiveRegions } : {}),
  };

  // Collect user & inherited fixed residue keys
  const userFixedKeys = new Set<string>();
  for (const ref of userRefs) {
    if (typeof ref === 'number') {
      const k = originalOrderKeys[ref];
      if (k) userFixedKeys.add(k);
    } else {
      const ch = resolveChain(ref.chain) ?? ref.chain;
      userFixedKeys.add(makeAntiFoldResidueKey(ch, ref.number, ref.insertionCode ?? ''));
    }
  }
  const inheritedFixedIndices = new Set<number>();
  if (!unlockInherited && Array.isArray(inheritedFixedResidueIndices)) {
    for (const oneBased of inheritedFixedResidueIndices) {
      const idx = Number(oneBased) - 1;
      if (Number.isInteger(idx) && idx >= 0 && idx < originalOrderKeys.length) {
        inheritedFixedIndices.add(idx);
      }
    }
  }

  // Evaluate regionMask using exact `antifold-esmif1/shared/structure.js` semantics on `effectiveRegions`
  const regionSelectors: Array<(r: { chain: string; number: number; insertion: string }) => boolean> = [];
  if (effectiveRegions.length > 0 && heavyChain) {
    const schemeBounds = ANTIBODY_CDR_BOUNDS[resolvedNumbering];
    for (const region of effectiveRegions) {
      const explicitSpan =
        /^([^:]+):(-?\d+[A-Za-z]?(?:--?\d+[A-Za-z]?)?(?:,-?\d+[A-Za-z]?(?:--?\d+[A-Za-z]?)?)*)$/.exec(
          region,
        );
      if (explicitSpan) {
        const chainId =
          explicitSpan[1] === 'H'
            ? heavyChain
            : explicitSpan[1] === 'L'
              ? lightChain
              : resolveChain(explicitSpan[1]) ?? explicitSpan[1];
        const ranges: Array<{ a: number; b: number; startIns: string; endIns: string }> = [];
        let valid = true;
        for (const part of explicitSpan[2].split(',')) {
          const rm = /^(-?\d+)([A-Za-z]?)(?:-(-?\d+)([A-Za-z]?))?$/.exec(part);
          if (!rm) {
            valid = false;
            break;
          }
          const a = Number(rm[1]);
          const startIns = (rm[2] ?? '').toUpperCase();
          const isRange = rm[3] !== undefined;
          const b = isRange ? Number(rm[3]) : a;
          const endIns = isRange ? (rm[4] ?? '').toUpperCase() : startIns;
          if (b < a || (b === a && startIns > endIns)) {
            valid = false;
            break;
          }
          ranges.push({ a, b, startIns, endIns });
        }
        if (valid && chainId) {
          regionSelectors.push(
            (r) =>
              r.chain === chainId &&
              ranges.some((x) => {
                const ins = (r.insertion ?? '').toUpperCase();
                const afterStart =
                  r.number > x.a || (r.number === x.a && (!x.startIns || ins >= x.startIns));
                const beforeEnd =
                  r.number < x.b || (r.number === x.b && (!x.endIns || ins <= x.endIns));
                return afterStart && beforeEnd;
              }),
          );
        } else {
          warnings.push(`Invalid AntiFold region selector "${region}".`);
        }
        continue;
      }
      const m = /^(all|CDR|FW)([HL]?)([1-4]?)$/.exec(region);
      if (!m || (m[1] === 'CDR' && m[3] === '4')) {
        warnings.push(
          `Invalid AntiFold region "${region}" (expected CDR1-3, CDRH1-3, CDRL1-3, FW1-4, all, or chain:start-end).`,
        );
        continue;
      }
      const [, type, role, num] = m;
      regionSelectors.push((r) => {
        const roleHere = r.chain === heavyChain ? 'H' : r.chain === lightChain ? 'L' : null;
        if (!roleHere || (role && role !== roleHere)) return false;
        const maxPos = roleHere === 'H' ? schemeBounds.maxH : schemeBounds.maxL;
        if (r.number < 1 || r.number > maxPos) return false;
        if (type === 'all') return true;
        const b = schemeBounds[roleHere];
        const regionsHere: ReadonlyArray<readonly [number, number]> =
          type === 'CDR'
            ? b
            : [
                [1, b[0][0] - 1],
                [b[0][1] + 1, b[1][0] - 1],
                [b[1][1] + 1, b[2][0] - 1],
                [b[2][1] + 1, maxPos],
              ];
        return regionsHere.some(
          ([lo, hi], idx) => (!num || idx === Number(num) - 1) && r.number >= lo && r.number <= hi,
        );
      });
    }
  }
  const useRegionMask = effectiveRegions.length > 0;

  const residueStates: AntiFoldResidueDesignState[] = [];
  let designedCount = 0;
  let fixedCount = 0;
  for (let i = 0; i < orderedResidues.length; i++) {
    const r = orderedResidues[i];
    const ins = r.insertionCode ?? '';
    const k = originalOrderKeys[i];
    const lockedByTargetChain =
      (Boolean(explicitTarget) && r.chain !== explicitTarget) ||
      (samplingMode === 'autoregressive' &&
        effectiveTargetChain !== undefined &&
        r.chain !== effectiveTargetChain);
    const inRegion =
      !useRegionMask ||
      regionSelectors.some((sel) => sel({ chain: r.chain, number: r.number, insertion: ins }));
    const isUserFixed = userLockedChainSet.has(r.chain) || userFixedKeys.has(k);
    const isInheritedFixed =
      !isUserFixed && (inheritedLockedChainSet.has(r.chain) || inheritedFixedIndices.has(i));
    const willDesign = !lockedByTargetChain && inRegion && !isUserFixed && !isInheritedFixed;
    if (willDesign) designedCount++;
    else fixedCount++;
    residueStates.push({
      flatIndex: i,
      chain: r.chain,
      number: r.number,
      insertion: ins,
      willDesign,
      lockedByTargetChain,
      inRegion,
      isUserFixed,
      isInheritedFixed,
    });
  }

  return {
    availableChains,
    ...(explicitTarget ? { explicitTarget } : {}),
    ...(effectiveTargetChain ? { effectiveTargetChain } : {}),
    resolvedTargetChain,
    ...(heavyChain ? { heavyChain } : {}),
    ...(lightChain ? { lightChain } : {}),
    ...(detectedHeavy ? { detectedHeavy } : {}),
    ...(detectedLight ? { detectedLight } : {}),
    resolvedNumbering,
    rawRegions,
    effectiveRegions,
    remappedChains: remapped.remappedChains,
    combineOpts,
    lockedChainSet,
    userLockedChainSet,
    inheritedLockedChainSet,
    userFixedKeys,
    inheritedFixedIndices,
    residueStates,
    designedCount,
    fixedCount,
    warnings,
  };
}
