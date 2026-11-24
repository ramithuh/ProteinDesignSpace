// Shared multi-island motif placement, geometric inter-island linker sizing,
// Genie 2 multi-island feature assembly, BoltzGen template motif extraction,
// and RFdiffusion empty-contig synthesis.
// Pure functions with zero browser/WebGPU dependencies for fast unit testing.

import { parseStructureText, type ParsedStructure } from './pdb';
import type { StructureArtifact } from '$lib/engine/types';

export interface MotifResidueSpec {
  /** Author chain ID (e.g. 'A', 'B'). */
  chain: string;
  /** Author residue number (e.g. 93, 198). */
  resNum: number;
  /** Optional insertion code (e.g. '', 'A'). */
  insCode?: string;
  /** Optional 3D Cα coordinate `[x, y, z]` in Ångströms for geometric linker lower bounds. */
  caCoord?: readonly [number, number, number] | null;
}

export interface MotifPlacementOptions {
  /** Desired total scaffold length. */
  targetLength: number;
  /**
   * Optional preferred N-terminal flank offset (e.g. BoltzGen `motifOffset`).
   * When omitted, motif span is centered between N- and C-terminal flanks.
   */
  preferredNFlank?: number;
  /** Minimum N-terminal flank length (default: 4; clamped to `preferredNFlank` if smaller). */
  minNFlank?: number;
  /** Minimum C-terminal flank length (default: 4). */
  minCFlank?: number;
  /** Minimum inter-island linker length when 3D distance is small or unavailable (default: 4). */
  minInterIslandGap?: number;
  /** Default desired gap for chain breaks or non-monotonic island jumps (default: 10). */
  defaultBreakGap?: number;
  /**
   * Whether `effectiveLength` may expand past `targetLength` if even minimal flanks + geometric
   * linkers exceed `targetLength` (default: `true`; set `false` for fixed-length samplers like Genie 2).
   */
  allowExpandLength?: boolean;
  /** Optional 0-based indices into `residues` that force a new island break before that residue. */
  explicitIslandBreaks?: ReadonlySet<number>;
}

export interface MotifIslandPlacement {
  /** 0-based inclusive start index into input `residues`. */
  inputStart: number;
  /** 0-based inclusive end index into input `residues`. */
  inputEnd: number;
  /** 0-based inclusive start index in the output scaffold. */
  scaffoldStart: number;
  /** 0-based inclusive end index in the output scaffold. */
  scaffoldEnd: number;
  /** Number of residues in this contiguous motif island. */
  length: number;
  chain: string;
  startResNum: number;
  endResNum: number;
  startInsCode: string;
  endInsCode: string;
}

export interface MotifPlacementResult {
  /** Final scaffold length (equals `targetLength` whenever physically feasible). */
  effectiveLength: number;
  /** 0-based scaffold positions for every input motif residue (strictly increasing). */
  targetIndices: number[];
  /** 0-based offsets relative to `targetIndices[0]`. */
  relativeOffsets: number[];
  /** Resolved N-terminal flank length (`targetIndices[0]`). */
  nFlank: number;
  /** Resolved C-terminal flank length (`effectiveLength - 1 - targetIndices[last]`). */
  cFlank: number;
  /** Resolved inter-island linker lengths between consecutive islands. */
  interIslandGaps: number[];
  /** Geometric lower-bound gap for each inter-island break. */
  minInterIslandGaps: number[];
  /** Native/desired gap for each inter-island break before budget scaling. */
  desiredInterIslandGaps: number[];
  /** Resolved contiguous motif islands. */
  islands: MotifIslandPlacement[];
}

function isFiniteCoord3(c: readonly [number, number, number] | null | undefined): c is readonly [number, number, number] {
  return (
    Boolean(c) &&
    c!.length >= 3 &&
    Number.isFinite(c![0]) &&
    Number.isFinite(c![1]) &&
    Number.isFinite(c![2])
  );
}

/**
 * Compute the strict 3D geometric minimum linker length (in residues) needed to span two motif
 * island endpoints without covalent fusion or impossible Cα–Cα stretch.
 * Formula: `Math.max(baseMinGap, Math.ceil(caDistAngstrom / 3.0) + 2)` when finite Cα coordinates
 * are available, else `baseMinGap` (default 4).
 */
export function computeGeometricMinGap(
  prevCa: readonly [number, number, number] | null | undefined,
  curCa: readonly [number, number, number] | null | undefined,
  baseMinGap = 4,
): number {
  const floorGap = Math.max(1, Math.round(baseMinGap));
  if (!isFiniteCoord3(prevCa) || !isFiniteCoord3(curCa)) {
    return floorGap;
  }
  const dist = Math.hypot(curCa[0] - prevCa[0], curCa[1] - prevCa[1], curCa[2] - prevCa[2]);
  if (!Number.isFinite(dist)) return floorGap;
  return Math.max(floorGap, Math.ceil(dist / 3.0) + 2);
}

/**
 * Distribute `extraBudget` non-negative integer residues proportionally to `weights`, capping each
 * slot at `caps[i]` (using largest-remainder Hamilton apportionment).
 */
function allocateProportionalExtra(
  extraBudget: number,
  caps: readonly number[],
): number[] {
  const n = caps.length;
  const alloc = new Array<number>(n).fill(0);
  let remaining = Math.max(0, Math.floor(extraBudget));
  if (n === 0 || remaining === 0) return alloc;

  const totalCap = caps.reduce((s, c) => s + Math.max(0, c), 0);
  if (totalCap <= 0) return alloc;
  if (remaining >= totalCap) {
    return caps.map((c) => Math.max(0, c));
  }

  const remainders: Array<{ idx: number; frac: number }> = [];
  for (let i = 0; i < n; i++) {
    const cap = Math.max(0, caps[i]);
    const exact = (remaining * cap) / totalCap;
    const base = Math.min(cap, Math.floor(exact));
    alloc[i] = base;
    remainders.push({ idx: i, frac: exact - base });
  }

  let used = alloc.reduce((s, v) => s + v, 0);
  let leftover = remaining - used;

  // Assign leftover residues to largest fractional parts (deterministic tie-break by index)
  remainders.sort((a, b) => (b.frac !== a.frac ? b.frac - a.frac : a.idx - b.idx));
  for (const r of remainders) {
    if (leftover <= 0) break;
    const cap = Math.max(0, caps[r.idx]);
    if (alloc[r.idx] < cap) {
      alloc[r.idx] += 1;
      leftover -= 1;
    }
  }

  // Fallback sweep if any slot hit its cap before leftover reached 0
  if (leftover > 0) {
    for (let i = 0; i < n && leftover > 0; i++) {
      const cap = Math.max(0, caps[i]);
      const room = cap - alloc[i];
      if (room > 0) {
        const add = Math.min(room, leftover);
        alloc[i] += add;
        leftover -= add;
      }
    }
  }

  return alloc;
}

/**
 * Plan physically feasible scaffold positions for single- or multi-island motifs.
 *
 * Guarantees:
 * 1. Across ANY residue discontinuity (`delta > 1`), non-monotonic jump (`delta < 0`), chain break,
 *    or explicit break, inter-island `gap` is NEVER collapsed to 0 (eliminates the `delta <= 65`
 *    covalent-fusion bug on `5YUI` and `4XOJ`).
 * 2. Every inter-island gap satisfies `gap >= Math.max(minInterIslandGap, Math.ceil(caDist / 3.0) + 2)`
 *    when 3D Cα coordinates are available.
 * 3. When native inter-island spacing fits inside `targetLength` (with flanks), preserves exact
 *    native spacing and honors `targetLength` cleanly (e.g. `1BCF` stays 125 instead of expanding to 127).
 * 4. When total native span exceeds `targetLength`, scales inter-island gaps proportionally to fit
 *    `targetLength` while strictly respecting every 3D geometric lower bound.
 */
export function planMultiIslandMotifPlacement(
  residues: readonly MotifResidueSpec[],
  options: MotifPlacementOptions,
): MotifPlacementResult {
  const n = residues.length;
  const targetLength = Math.max(1, Math.round(options.targetLength));
  if (n === 0) {
    return {
      effectiveLength: targetLength,
      targetIndices: [],
      relativeOffsets: [],
      nFlank: 0,
      cFlank: 0,
      interIslandGaps: [],
      minInterIslandGaps: [],
      desiredInterIslandGaps: [],
      islands: [],
    };
  }

  const baseMinGap = Math.max(1, Math.round(options.minInterIslandGap ?? 4));
  const defaultBreakGap = Math.max(baseMinGap, Math.round(options.defaultBreakGap ?? 10));
  const allowExpandLength = options.allowExpandLength !== false;
  const preferredNFlank =
    options.preferredNFlank !== undefined
      ? Math.max(0, Math.round(options.preferredNFlank))
      : undefined;
  const rawMinN = Math.max(0, Math.round(options.minNFlank ?? 4));
  const reqMinN = preferredNFlank !== undefined ? Math.min(preferredNFlank, rawMinN) : rawMinN;
  const reqMinC = Math.max(0, Math.round(options.minCFlank ?? 4));

  // Step 1: Partition input residues into contiguous motif islands and compute per-break bounds
  interface RawIsland {
    inputStart: number;
    inputEnd: number;
  }
  const rawIslands: RawIsland[] = [];
  const minGaps: number[] = [];
  const desiredGaps: number[] = [];

  let islandStart = 0;
  for (let i = 1; i < n; i++) {
    const prev = residues[i - 1];
    const cur = residues[i];
    const sameChain = prev.chain === cur.chain;
    const delta = cur.resNum - prev.resNum;
    const prevIns = (prev.insCode ?? '').trim();
    const curIns = (cur.insCode ?? '').trim();

    const isExplicit = Boolean(options.explicitIslandBreaks?.has(i));
    const isInsertionContinuation = sameChain && delta === 0 && prevIns !== curIns;
    const isContiguous = !isExplicit && sameChain && (delta === 1 || isInsertionContinuation);

    if (!isContiguous) {
      rawIslands.push({ inputStart: islandStart, inputEnd: i - 1 });
      islandStart = i;

      const minGap = computeGeometricMinGap(prev.caCoord, cur.caCoord, baseMinGap);
      const desiredGap =
        sameChain && delta > 1
          ? Math.max(minGap, delta - 1)
          : Math.max(minGap, defaultBreakGap);

      minGaps.push(minGap);
      desiredGaps.push(desiredGap);
    }
  }
  rawIslands.push({ inputStart: islandStart, inputEnd: n - 1 });

  const nBreaks = minGaps.length;
  const sumMinGaps = minGaps.reduce((s, g) => s + g, 0);
  const sumDesiredGaps = desiredGaps.reduce((s, g) => s + g, 0);
  const totalScaffoldBudget = targetLength - n;

  let resolvedGaps: number[];
  let nFlank: number;
  let cFlank: number;
  let effectiveLength: number;

  if (totalScaffoldBudget >= sumDesiredGaps + reqMinN + reqMinC) {
    // Case A: Exact native inter-island gaps fit cleanly inside targetLength with >= min flanks.
    resolvedGaps = [...desiredGaps];
    effectiveLength = targetLength;
    const flankBudget = totalScaffoldBudget - sumDesiredGaps;
    if (preferredNFlank !== undefined) {
      nFlank = Math.max(reqMinN, Math.min(preferredNFlank, flankBudget - reqMinC));
      cFlank = flankBudget - nFlank;
    } else {
      nFlank = Math.floor(flankBudget / 2);
      cFlank = flankBudget - nFlank;
    }
  } else if (totalScaffoldBudget >= sumMinGaps + reqMinN + reqMinC) {
    // Case B: Native span exceeds targetLength, but targetLength can accommodate all motif residues,
    // minimal flanks, and strict geometric lower-bound linkers. Scale inter-island gaps proportionally.
    effectiveLength = targetLength;
    const maxFlanks = totalScaffoldBudget - sumMinGaps;
    const fairFlank = Math.round(totalScaffoldBudget / (nBreaks + 2));

    let wantN =
      preferredNFlank !== undefined
        ? Math.max(reqMinN, Math.min(preferredNFlank, maxFlanks - reqMinC))
        : Math.max(reqMinN, Math.min(12, fairFlank));
    let wantC = Math.max(reqMinC, Math.min(12, fairFlank));

    if (wantN + wantC > maxFlanks) {
      const availExtra = maxFlanks - reqMinN - reqMinC;
      const nWantExtra = Math.max(0, wantN - reqMinN);
      const cWantExtra = Math.max(0, wantC - reqMinC);
      const sumWantExtra = nWantExtra + cWantExtra;
      const nExtra =
        sumWantExtra > 0
          ? Math.floor((availExtra * nWantExtra) / sumWantExtra)
          : Math.floor(availExtra / 2);
      wantN = reqMinN + nExtra;
      wantC = reqMinC + (availExtra - nExtra);
    }

    nFlank = wantN;
    cFlank = wantC;

    const gapBudget = totalScaffoldBudget - nFlank - cFlank;
    const extraGapBudget = Math.max(0, gapBudget - sumMinGaps);
    const desiredExtras = desiredGaps.map((d, i) => Math.max(0, d - minGaps[i]));
    const extraAlloc = allocateProportionalExtra(extraGapBudget, desiredExtras);
    resolvedGaps = minGaps.map((m, i) => m + extraAlloc[i]);

    // If desiredExtras were capped below gapBudget (rare), absorb any leftover into flanks
    const actualGapSum = resolvedGaps.reduce((s, g) => s + g, 0);
    const leftover = gapBudget - actualGapSum;
    if (leftover > 0) {
      const addN = Math.floor(leftover / 2);
      nFlank += addN;
      cFlank += leftover - addN;
    }
  } else if (!allowExpandLength && totalScaffoldBudget >= sumMinGaps) {
    // Case C1 (fixed-length sampler like Genie 2): Fit strictly inside targetLength by trimming
    // N/C flanks below minNFlank/minCFlank while STILL upholding strict geometric minGaps.
    resolvedGaps = [...minGaps];
    effectiveLength = targetLength;
    const remFlank = totalScaffoldBudget - sumMinGaps;
    if (preferredNFlank !== undefined) {
      nFlank = Math.min(preferredNFlank, remFlank);
      cFlank = remFlank - nFlank;
    } else {
      nFlank = Math.floor(remFlank / 2);
      cFlank = remFlank - nFlank;
    }
  } else {
    // Case C2: Even minimal geometric linkers (+ minimal flanks when expandable) exceed targetLength.
    resolvedGaps = [...minGaps];
    nFlank = allowExpandLength ? reqMinN : 0;
    cFlank = allowExpandLength ? reqMinC : 0;
    effectiveLength = n + sumMinGaps + nFlank + cFlank;
  }

  // Step 2: Materialize targetIndices, relativeOffsets, and island descriptors
  const targetIndices: number[] = [];
  const relativeOffsets: number[] = [];
  const islands: MotifIslandPlacement[] = [];

  let cursor = nFlank;
  for (let islIdx = 0; islIdx < rawIslands.length; islIdx++) {
    const raw = rawIslands[islIdx];
    const scaffoldStart = cursor;
    for (let k = raw.inputStart; k <= raw.inputEnd; k++) {
      targetIndices.push(cursor);
      relativeOffsets.push(cursor - nFlank);
      cursor += 1;
    }
    const scaffoldEnd = cursor - 1;
    const firstRes = residues[raw.inputStart];
    const lastRes = residues[raw.inputEnd];
    islands.push({
      inputStart: raw.inputStart,
      inputEnd: raw.inputEnd,
      scaffoldStart,
      scaffoldEnd,
      length: raw.inputEnd - raw.inputStart + 1,
      chain: firstRes.chain,
      startResNum: firstRes.resNum,
      endResNum: lastRes.resNum,
      startInsCode: (firstRes.insCode ?? '').trim(),
      endInsCode: (lastRes.insCode ?? '').trim(),
    });
    if (islIdx < resolvedGaps.length) {
      cursor += resolvedGaps[islIdx];
    }
  }

  return {
    effectiveLength,
    targetIndices,
    relativeOffsets,
    nFlank,
    cFlank,
    interIslandGaps: resolvedGaps,
    minInterIslandGaps: minGaps,
    desiredInterIslandGaps: desiredGaps,
    islands,
  };
}

export interface ProtpardelleParsedTemplateLike {
  aatype: ArrayLike<number>;
  residues?: ReadonlyArray<{ chain?: string; resid?: number; icode?: string }>;
  coords?: ArrayLike<number>;
  atomMask?: ArrayLike<number>;
}

export interface ProtpardelleMotifPlanOptions {
  targetLength: number;
  sourceIndices?: readonly number[];
  minNFlank?: number;
  minCFlank?: number;
}

export interface ProtpardelleMotifPlan {
  effectiveLength: number;
  sourceIndices: number[];
  targetIndices: number[];
  motifSpecs: MotifResidueSpec[];
  placement: MotifPlacementResult;
}

/**
 * Validate, deduplicate, and sort ProtPardelle template `sourceIndices` (defaulting to all
 * template residues when `motifSelection` is empty), extract 3D Cα coordinates from
 * `parsedTemplate.coords` (`CA` at `srcIdx * 111 + 3..5`), and compute physically bounded
 * multi-island scaffold positions via `planMultiIslandMotifPlacement`.
 */
export function planProtpardelleMotifPlacement(
  parsedTemplate: ProtpardelleParsedTemplateLike,
  options: ProtpardelleMotifPlanOptions,
): ProtpardelleMotifPlan {
  const tplLen = parsedTemplate.aatype.length;
  if (tplLen <= 0) {
    throw new Error('ProtPardelle motif template contains 0 residues.');
  }

  let sourceIndices: number[];
  if (options.sourceIndices !== undefined) {
    for (const idx of options.sourceIndices) {
      if (!Number.isInteger(idx) || idx < 0 || idx >= tplLen) {
        throw new Error(
          `Invalid ProtPardelle motif source index ${String(idx)} (expected integer in [0, ${tplLen})).`,
        );
      }
    }
    sourceIndices = [...new Set(options.sourceIndices)].sort((a, b) => a - b);
  } else {
    sourceIndices = Array.from({ length: tplLen }, (_, i) => i);
  }

  if (sourceIndices.length === 0) {
    throw new Error('ProtPardelle motif placement requires at least 1 selected motif residue.');
  }

  const tplRes = parsedTemplate.residues;
  const tplCoords = parsedTemplate.coords;
  const tplAtomMask = parsedTemplate.atomMask;

  const motifSpecs: MotifResidueSpec[] = sourceIndices.map((srcIdx) => {
    const r = tplRes?.[srcIdx];
    let caCoord: [number, number, number] | null = null;
    if (
      tplCoords &&
      srcIdx * 111 + 5 < tplCoords.length &&
      (!tplAtomMask || tplAtomMask[srcIdx * 37 + 1] > 0.5)
    ) {
      const x = tplCoords[srcIdx * 111 + 3];
      const y = tplCoords[srcIdx * 111 + 4];
      const z = tplCoords[srcIdx * 111 + 5];
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        caCoord = [x, y, z];
      }
    }
    return {
      chain: r?.chain ?? 'A',
      resNum: typeof r?.resid === 'number' ? r.resid : srcIdx + 1,
      insCode: r?.icode ?? '',
      caCoord,
    };
  });

  const placement = planMultiIslandMotifPlacement(motifSpecs, {
    targetLength: options.targetLength,
    minNFlank: options.minNFlank ?? 4,
    minCFlank: options.minCFlank ?? 4,
    allowExpandLength: true,
  });

  return {
    effectiveLength: placement.effectiveLength,
    sourceIndices,
    targetIndices: placement.targetIndices,
    motifSpecs,
    placement,
  };
}

// ---------------------------------------------------------------------------
// BoltzGen motif placement & native Cα extraction
// ---------------------------------------------------------------------------

export interface BoltzGenMotifPlan {
  effectiveLength: number;
  motifIndices: number[];
  fixedResidueIndices: number[];
  nativeMotifCoords: number[][];
  placement: MotifPlacementResult;
}

/**
 * Extract BoltzGen motif residues and native Cα coordinates directly from the actual `targetPDB`
 * text passed to BoltzGen (rather than indexing blindly into a potentially uncropped
 * `upstreamStruct.caCoords`), and compute physically bounded multi-island scaffold indices.
 */
export function buildBoltzGenMotifPlan(
  upstreamStruct: Pick<StructureArtifact, 'pdbText' | 'caCoords' | 'sourceResidues'>,
  options: { targetLength: number; motifOffset: number },
): BoltzGenMotifPlan {
  const parsedTpl = parseStructureText(upstreamStruct.pdbText ?? '');
  if (parsedTpl.nResidues === 0) {
    throw new Error(
      'BoltzGen-1 in "motif" mode requires a template structure with at least 1 protein residue.',
    );
  }

  const motifSpecs: MotifResidueSpec[] = [];
  const nativeMotifCoords: number[][] = [];
  for (let i = 0; i < parsedTpl.nResidues; i++) {
    const sr = parsedTpl.sourceResidues[i];
    const x = parsedTpl.caCoords[i * 3];
    const y = parsedTpl.caCoords[i * 3 + 1];
    const z = parsedTpl.caCoords[i * 3 + 2];
    const hasFiniteCa = Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z);
    if (!hasFiniteCa) {
      throw new Error(
        `BoltzGen-1 motif template has non-finite Cα coordinates at residue ${sr?.authChainId ?? 'A'}${sr?.resNum ?? i + 1}.`,
      );
    }
    const ca: [number, number, number] = [x, y, z];
    motifSpecs.push({
      chain: sr?.authChainId ?? parsedTpl.chainIds[i] ?? 'A',
      resNum: sr?.resNum ?? parsedTpl.resNums[i] ?? i + 1,
      insCode: sr?.insCode ?? parsedTpl.insCodes[i] ?? '',
      caCoord: ca,
    });
    nativeMotifCoords.push(ca);
  }

  const motifOffset = Math.max(0, Math.round(options.motifOffset));
  const placement = planMultiIslandMotifPlacement(motifSpecs, {
    targetLength: options.targetLength,
    preferredNFlank: motifOffset,
    minNFlank: Math.min(4, motifOffset),
    minCFlank: 4,
    allowExpandLength: true,
  });

  return {
    effectiveLength: placement.effectiveLength,
    motifIndices: placement.targetIndices,
    fixedResidueIndices: placement.targetIndices.map((idx) => idx + 1),
    nativeMotifCoords,
    placement,
  };
}

// ---------------------------------------------------------------------------
// Genie 2 multi-island motif parser & feature builder
// ---------------------------------------------------------------------------

export interface BuiltGenie2Motif {
  motifs: Array<{ start: number; sequence: string; coordinates: Float32Array; group: number }>;
  motifIndices: number[];
  fixedResidueIndices: number[];
  nativeMotifCoords: number[][];
  placement: MotifPlacementResult;
}

/**
 * Parse single- or multi-island `motifRange` specifications for Genie 2
 * (e.g. `"A163-181"`, `"A16-35, A52-71"`, `"A93-97; A118-120; A198-200"`, `"A55, A99, A190-192"`)
 * and emit non-overlapping, physically-spaced Genie 2 motif segments sharing a single rigid-body
 * `group: 1` and shared 3D centroid so inter-island catalytic geometry is preserved cleanly.
 */
export function buildGenie2OptionalMotifs(
  template: Pick<StructureArtifact, 'pdbText'> | undefined,
  motifRange: string,
  targetLength: number,
): BuiltGenie2Motif | undefined {
  const trimmed = motifRange.trim();
  if (!trimmed) return undefined;
  if (!template?.pdbText) {
    throw new Error('Genie 2 motifRange requires a connected template structure.');
  }
  const parsed = parseStructureText(template.pdbText);
  if (parsed.nResidues === 0) {
    throw new Error('Genie 2 motif template contains no valid residues.');
  }

  const tokens = trimmed
    .split(/[,;\s]+/)
    .map((t) => t.trim())
    .filter(Boolean);
  if (tokens.length === 0) return undefined;

  const uniqueChains = [...new Set(parsed.chainIds)];
  const picked: number[] = [];
  const seenTemplateIdx = new Set<number>();

  for (const tok of tokens) {
    const match = /^(?:([A-Za-z]):?)?(-?\d+)(?:-(?:([A-Za-z]):?)?(-?\d+))?$/.exec(tok);
    if (!match) {
      throw new Error(
        `Invalid Genie 2 motifRange token "${tok}" in "${trimmed}" (expected e.g. A10-25, A55, or A16-35, A52-71).`,
      );
    }
    const rawChainStart = match[1];
    const rawChainEnd = match[3];
    if (
      rawChainStart &&
      rawChainEnd &&
      rawChainStart.toUpperCase() !== rawChainEnd.toUpperCase()
    ) {
      throw new Error(
        `Invalid Genie 2 motifRange token "${tok}": range crosses chains (${rawChainStart} vs ${rawChainEnd}).`,
      );
    }
    const rawChainFilter = rawChainStart ?? rawChainEnd;
    if (!rawChainFilter && uniqueChains.length > 1) {
      throw new Error(
        `Ambiguous Genie 2 motifRange "${tok}" on multi-chain template (chains ${uniqueChains.join(', ')}). Prefix with a chain ID such as "${uniqueChains[0]}${tok}".`,
      );
    }
    let resolvedChainFilter: string | undefined;
    if (rawChainFilter) {
      if (uniqueChains.includes(rawChainFilter)) {
        resolvedChainFilter = rawChainFilter;
      } else {
        const upper = rawChainFilter.toUpperCase();
        const ciMatches = uniqueChains.filter((c) => c.toUpperCase() === upper);
        resolvedChainFilter = ciMatches.length === 1 ? ciMatches[0] : rawChainFilter;
      }
    }
    const startRes = Number(match[2]);
    const endRes = match[4] !== undefined ? Number(match[4]) : startRes;
    const lo = Math.min(startRes, endRes);
    const hi = Math.max(startRes, endRes);

    let tokenMatched = 0;
    for (let i = 0; i < parsed.nResidues; i++) {
      if (resolvedChainFilter && parsed.chainIds[i] !== resolvedChainFilter) continue;
      const r = parsed.resNums[i];
      if (r >= lo && r <= hi) {
        tokenMatched += 1;
        if (!seenTemplateIdx.has(i)) {
          seenTemplateIdx.add(i);
          picked.push(i);
        }
      }
    }
    if (tokenMatched === 0) {
      throw new Error(
        `Genie 2 motifRange token "${tok}" matched 0 residues in template.`,
      );
    }
  }

  // Enforce Genie 2's 3-residue minimum across ALL parsed islands combined (fixes 4XOJ A55 = 1 res)
  if (picked.length < 3) {
    throw new Error(
      `Genie 2 motifRange "${trimmed}" matched only ${picked.length} residue(s) across all islands (minimum 3 required).`,
    );
  }
  if (picked.length >= targetLength) {
    throw new Error(
      `Genie 2 motif length (${picked.length} aa) must be smaller than target length (${targetLength} aa).`,
    );
  }

  // Validate coordinates & compute ONE shared centroid across all islands so relative 3D geometry is exact
  let cx = 0;
  let cy = 0;
  let cz = 0;
  const nativeMotifCoords: number[][] = [];
  const motifSpecs: MotifResidueSpec[] = [];

  for (let k = 0; k < picked.length; k++) {
    const idx = picked[k];
    const x = parsed.caCoords[idx * 3];
    const y = parsed.caCoords[idx * 3 + 1];
    const z = parsed.caCoords[idx * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new Error(
        `Genie 2 motif template has non-finite Cα coordinates at residue index ${idx}.`,
      );
    }
    cx += x;
    cy += y;
    cz += z;
    nativeMotifCoords.push([x, y, z]);
    motifSpecs.push({
      chain: parsed.sourceResidues[idx]?.authChainId ?? parsed.chainIds[idx] ?? 'A',
      resNum: parsed.resNums[idx],
      insCode: parsed.insCodes[idx] ?? '',
      caCoord: [x, y, z],
    });
  }
  cx /= picked.length;
  cy /= picked.length;
  cz /= picked.length;

  const placement = planMultiIslandMotifPlacement(motifSpecs, {
    targetLength,
    minNFlank: 4,
    minCFlank: 4,
    allowExpandLength: false,
  });

  if (placement.effectiveLength > targetLength) {
    throw new Error(
      `Genie 2 motif (${picked.length} residues across ${placement.islands.length} island(s), minimum geometric span ${placement.effectiveLength} aa) exceeds target length (${targetLength} aa).`,
    );
  }

  const motifs: Array<{ start: number; sequence: string; coordinates: Float32Array; group: number }> =
    [];
  for (const island of placement.islands) {
    const coords = new Float32Array(island.length * 3);
    let seq = '';
    for (let j = 0; j < island.length; j++) {
      const k = island.inputStart + j;
      const idx = picked[k];
      coords[j * 3] = parsed.caCoords[idx * 3] - cx;
      coords[j * 3 + 1] = parsed.caCoords[idx * 3 + 1] - cy;
      coords[j * 3 + 2] = parsed.caCoords[idx * 3 + 2] - cz;
      const rawAa = (parsed.sequence[idx] || 'A').toUpperCase();
      seq += /^[ACDEFGHIKLMNPQRSTVWY]$/.test(rawAa) ? rawAa : 'A';
    }
    motifs.push({
      start: island.scaffoldStart,
      sequence: seq,
      coordinates: coords,
      group: 1,
    });
  }

  const motifIndices = [...placement.targetIndices];
  const fixedResidueIndices = motifIndices.map((idx) => idx + 1);

  return {
    motifs,
    motifIndices,
    fixedResidueIndices,
    nativeMotifCoords,
    placement,
  };
}

// ---------------------------------------------------------------------------
// RFdiffusion motif template guard & empty-contig multi-island synthesis
// ---------------------------------------------------------------------------

/**
 * Validate an RFdiffusion motif template structure (failing fast on 0-protein-residue PDBs
 * such as ligand-only or DNA-only files) and resolve or synthesize a valid `/`-delimited
 * RFdiffusion contig string.
 *
 * When `rawContig` is empty on a cropped multi-island template (e.g. `5YUI` with residues
 * `A93-97, A118-120, A198-200`), detects all contiguous residue runs and synthesizes a
 * valid multi-island contig (`flank/A93-97/gap/A118-120/gap/A198-200/flank`) with physically
 * feasible inter-island linkers.
 */
export function resolveRfdiffusionMotifContig(
  upstreamStruct: Pick<StructureArtifact, 'pdbText' | 'label'>,
  rawContig: string,
  targetLength: number,
): string {
  const parsed: ParsedStructure = parseStructureText(upstreamStruct.pdbText ?? '');
  if (parsed.nResidues === 0) {
    const labelHint = upstreamStruct.label ? ` ("${upstreamStruct.label}")` : '';
    throw new Error(
      `RFdiffusion in "motif" mode requires a template structure with at least 1 protein residue; connected PDB${labelHint} contains 0 protein ATOM residues (ligand/DNA-only structures are not supported by RFdiffusion Base).`,
    );
  }

  const trimmed = rawContig.trim();
  if (trimmed) {
    return trimmed
      .split(/[\s,]+/)
      .map((t) => t.trim())
      .filter(Boolean)
      .join('/');
  }

  // Empty contig: inspect contiguous residue runs in the template
  const allSpecs: MotifResidueSpec[] = [];
  for (let i = 0; i < parsed.nResidues; i++) {
    const x = parsed.caCoords[i * 3];
    const y = parsed.caCoords[i * 3 + 1];
    const z = parsed.caCoords[i * 3 + 2];
    const caCoord: [number, number, number] | null =
      Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) ? [x, y, z] : null;
    allSpecs.push({
      // Use projected 1-char PDB chain ID matching PDB column 22 parsed by RFdiffusion
      chain: parsed.chainIds[i] ?? 'A',
      resNum: parsed.resNums[i],
      insCode: parsed.insCodes[i] ?? '',
      caCoord,
    });
  }

  // Probe island count first: if the template is a single large uncropped chain (> targetLength - 8
  // and > 20 aa), take the first 20 residues as a safe default motif instead of expanding the scaffold.
  const initialProbe = planMultiIslandMotifPlacement(allSpecs, {
    targetLength,
    minNFlank: 4,
    minCFlank: 4,
    allowExpandLength: true,
  });

  const specs =
    initialProbe.islands.length === 1 &&
    allSpecs.length > 20 &&
    allSpecs.length > Math.max(20, targetLength - 8)
      ? allSpecs.slice(0, 20)
      : allSpecs;

  const placement =
    specs === allSpecs
      ? initialProbe
      : planMultiIslandMotifPlacement(specs, {
          targetLength,
          minNFlank: 4,
          minCFlank: 4,
          allowExpandLength: true,
        });

  const tokens: string[] = [];
  if (placement.nFlank > 0) {
    tokens.push(`${placement.nFlank}-${placement.nFlank}`);
  }
  for (let idx = 0; idx < placement.islands.length; idx++) {
    const isl = placement.islands[idx];
    const startTok = `${isl.chain}${isl.startResNum}${isl.startInsCode}`;
    const endTok = `${isl.endResNum}${isl.endInsCode}`;
    tokens.push(`${startTok}-${endTok}`);
    if (idx < placement.interIslandGaps.length) {
      const gap = placement.interIslandGaps[idx];
      if (gap > 0) tokens.push(`${gap}-${gap}`);
    }
  }
  if (placement.cFlank > 0) {
    tokens.push(`${placement.cFlank}-${placement.cFlank}`);
  }

  return tokens.join('/');
}
