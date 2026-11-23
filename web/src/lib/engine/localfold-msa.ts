// Pure-TypeScript MSA resolution, .a3m validation, target-only chain masking,
// in-run MMseqs2 search caching, and LocalFold fidelity helpers.
// Testable in Node/Vitest without WebGPU or browser globals.

import { selectFoldRepresentative } from './bio-capabilities';
import type {
  CollectionKind,
  FoldedCandidate,
  MsaMode,
  RepresentativeSelection,
} from './types';

export interface MmseqsProgressEvent {
  phase?: string;
  status?: string;
  search?: string | number;
  searches?: number;
  chain?: number;
  ticket?: string;
  elapsedMilliseconds?: number;
}

export interface MmseqsSingleResult {
  a3m: string;
  ticket?: string;
  depth: number;
  elapsedMilliseconds?: number;
  templateHits?: Map<number, unknown[]>;
}

export interface MmseqsComplexResult {
  a3m: string;
  blocks?: {
    paired: string | null;
    unpaired: string;
    unpairedProfile: string;
  };
  chainA3ms: string[];
  pairedA3ms?: Map<string, string>;
  depth: number;
  pairedDepth?: number;
  tickets?: string[];
  templateHits?: Map<number, unknown[]>;
  elapsedMilliseconds?: number;
}

export interface MmseqsRequestOptions {
  model?: string;
  useEnvironmental?: boolean;
  onProgress?: (ev: MmseqsProgressEvent) => void;
  fetchImplementation?: typeof fetch;
  signal?: AbortSignal;
  wait?: (ms: number, signal?: AbortSignal) => Promise<void>;
  decompress?: (archive: ArrayBuffer | Uint8Array) => Promise<Uint8Array> | Uint8Array;
}

export interface ParsedA3mLike {
  query: string;
  descriptions: string[];
  rawSequences: string[];
  sequences: string[];
  depth: number;
  length: number;
}

/** Injectable subset of vendored `/vendor/localfold/shared/input/{mmseqs2-api,a3m}.js`. */
export interface MmseqsApi {
  generateMmseqs2Msa: (
    sequence: string,
    options?: MmseqsRequestOptions,
  ) => Promise<MmseqsSingleResult>;
  generateMmseqs2ComplexMsa: (
    sequences: readonly string[],
    options?: MmseqsRequestOptions,
  ) => Promise<MmseqsComplexResult>;
  expandSearchedChains: (input: {
    chains: string[];
    off: boolean[];
    parts: string[];
    pairedA3ms?: Map<string, string>;
    templateHits?: Map<number, unknown[]>;
  }) => {
    chainA3ms: string[];
    pairedA3ms?: Map<string, string>;
    templateHits: Map<number, unknown[]>;
  };
  mergeSearchedChains: (input: {
    sequences: string[];
    chainA3ms: string[];
    pairedA3ms?: Map<string, string>;
    model: string;
  }) => {
    a3m: string;
    blocks: {
      paired: string | null;
      unpaired: string;
      unpairedProfile: string;
    };
  };
  singleSequenceA3m: (sequences: string[], model: string) => string;
  parseA3m?: (text: string) => ParsedA3mLike;
  graphFamily?: (model: string) => string;
}

export type Af3AlignmentInput =
  | null
  | string
  | {
      paired: string | null;
      unpaired: string | null;
      unpairedProfile?: string | null;
    };

export type MsaSearchCacheEntry =
  | {
      kind: 'single';
      a3m: string;
      depth: number;
      templateHits?: Map<number, unknown[]>;
    }
  | {
      kind: 'complex';
      chainA3ms: string[];
      pairedA3ms?: Map<string, string>;
      depth: number;
      pairedDepth: number;
      templateHits?: Map<number, unknown[]>;
    };

export type MsaSearchCache = Map<string, MsaSearchCacheEntry>;

export interface MsaProgressUpdate {
  phase: 'msa';
  detail: string;
  raw?: MmseqsProgressEvent;
}

export interface ValidatedA3mResult {
  /** Single or merged A3M text (clean, parseA3m-compatible). */
  a3m: string;
  /** Per-chain A3M blocks when the input supplied separate blocks per chain. */
  perChainA3ms?: string[];
  /** Concatenated ungapped query sequence. */
  query: string;
  /** Total row count (including query). */
  depth: number;
  /** Number of distinct aligned sequences. */
  distinctDepth: number;
  /** Aligned column count. */
  length: number;
  /** True when the input was split into per-chain A3M blocks. */
  wasPerChainBlocks: boolean;
}

export interface ResolveMsaOptions {
  msaMode: MsaMode;
  modelKey: string;
  chainSeqs: readonly string[];
  chainIds: readonly string[];
  lockedChains?: readonly string[];
  chainRoles?: Readonly<Record<string, 'target' | 'binder' | 'scaffold'>>;
  msaA3mText?: string;
  mmseqsApi: MmseqsApi;
  cache?: MsaSearchCache;
  onProgress?: (ev: MsaProgressUpdate) => void;
  signal?: AbortSignal;
  requestOptions?: Omit<MmseqsRequestOptions, 'model' | 'onProgress' | 'signal'>;
}

export interface ResolvedMsaResult {
  msaMode: MsaMode;
  /** Model-formatted A3M text (ready for AF2 `predictA3m`). */
  a3m: string;
  /** Alignment payload ready for `af3BatchFromA3m(colonSeq, af3Alignment, opts)`. */
  af3Alignment: Af3AlignmentInput;
  /** Total MSA row depth (1 for single-sequence). */
  msaDepth: number;
  /** Number of distinct aligned sequences. */
  distinctDepth: number;
  /** Paired row count when species pairing ran (0 otherwise). */
  pairedDepth: number;
  /** Per-chain boolean mask (`true` = skipped / query-only). */
  offMask: boolean[];
  /** True when fulfilled from the in-run cache without querying MMseqs2. */
  cacheHit: boolean;
}

const AMBIGUOUS_AA: Readonly<Record<string, string>> = {
  B: 'D',
  Z: 'E',
  U: 'C',
  J: 'X',
  O: 'X',
};

const VALID_ALIGNED_RE = /^[ACDEFGHIKLMNPQRSTVWYX-]+$/;

interface ParsedBlock {
  query: string;
  descriptions: string[];
  rawSequences: string[];
  sequences: string[];
  depth: number;
  distinctDepth: number;
  length: number;
  normalizedA3m: string;
}

function normalizeResidueCode(ch: string): string {
  return AMBIGUOUS_AA[ch] ?? ch;
}

function seqPreview(seq: string, maxLen = 16): string {
  return seq.length <= maxLen ? seq : `${seq.slice(0, maxLen)}…`;
}

function extractAlignedSequence(raw: string, description: string): { aligned: string; normalizedRaw: string } {
  if (raw.length === 0) {
    throw new Error(`A3M sequence "${description}" is empty`);
  }
  let aligned = '';
  let normalizedRaw = '';
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (ch >= 'a' && ch <= 'z') {
      normalizedRaw += ch;
      continue;
    }
    const mapped = normalizeResidueCode(ch);
    if (!VALID_ALIGNED_RE.test(mapped)) {
      throw new Error(`A3M sequence "${description}" contains invalid residue ${JSON.stringify(ch)}`);
    }
    aligned += mapped;
    normalizedRaw += mapped;
  }
  return { aligned, normalizedRaw };
}

function parseSingleA3mBlock(blockText: string): ParsedBlock {
  const descriptions: string[] = [];
  const rawInputs: string[] = [];
  let current = -1;

  for (const rawLine of blockText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line === '' || line.startsWith('#')) continue;
    if (line.startsWith('>')) {
      const desc = line.slice(1).trim();
      if (!desc) {
        throw new Error('A3M contains an empty FASTA header (">")');
      }
      descriptions.push(desc);
      rawInputs.push('');
      current += 1;
      continue;
    }
    if (current < 0) {
      throw new Error('A3M sequence data appears before the first FASTA header (">")');
    }
    if (/\s/.test(line)) {
      throw new Error(`A3M sequence "${descriptions[current]}" contains unexpected whitespace`);
    }
    rawInputs[current] += line;
  }

  if (rawInputs.length === 0) {
    throw new Error('A3M contains no sequences');
  }

  const sequences: string[] = [];
  const rawSequences: string[] = [];
  for (let i = 0; i < rawInputs.length; i++) {
    const { aligned, normalizedRaw } = extractAlignedSequence(rawInputs[i], descriptions[i]);
    sequences.push(aligned);
    rawSequences.push(normalizedRaw);
  }

  const length = sequences[0].length;
  if (length === 0 || sequences[0].includes('-')) {
    throw new Error('The first A3M sequence must be a non-empty, ungapped query sequence');
  }

  for (let i = 1; i < sequences.length; i++) {
    if (sequences[i].length !== length) {
      throw new Error(
        `A3M row "${descriptions[i]}" has aligned length ${sequences[i].length}; expected ${length}`,
      );
    }
  }

  const lines: string[] = [];
  for (let i = 0; i < sequences.length; i++) {
    lines.push(`>${descriptions[i]}`, rawSequences[i]);
  }

  return {
    query: sequences[0],
    descriptions,
    rawSequences,
    sequences,
    depth: sequences.length,
    distinctDepth: new Set(sequences).size,
    length,
    normalizedA3m: `${lines.join('\n')}\n`,
  };
}

/**
 * Split text into `#`-delimited per-chain A3M blocks (ignoring leading ColabFold `#length` headers).
 */
function splitByHashBlocks(text: string): string[] {
  const blocks: string[] = [];
  let currentLines: string[] = [];
  let sawSequence = false;

  for (const rawLine of text.split(/\r?\n/)) {
    const trimmed = rawLine.trim();
    if (trimmed.startsWith('#')) {
      if (sawSequence && currentLines.length > 0) {
        blocks.push(currentLines.join('\n'));
        currentLines = [];
        sawSequence = false;
      }
      continue;
    }
    if (trimmed.startsWith('>')) {
      sawSequence = true;
    }
    if (trimmed.length > 0) {
      currentLines.push(trimmed);
    }
  }
  if (sawSequence && currentLines.length > 0) {
    blocks.push(currentLines.join('\n'));
  }
  return blocks;
}

/**
 * Split concatenated FASTA records into per-chain blocks when the user pasted
 * chain A's A3M directly followed by chain B's A3M (`>` boundary without `#`).
 */
function splitByFastaChainBoundaries(text: string, expectedChains: readonly string[]): string[] | null {
  if (expectedChains.length <= 1) return null;

  const records: Array<{ desc: string; raw: string }> = [];
  let curDesc: string | null = null;
  let curRaw = '';

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line === '' || line.startsWith('#')) continue;
    if (line.startsWith('>')) {
      if (curDesc !== null) {
        records.push({ desc: curDesc, raw: curRaw });
      }
      curDesc = line.slice(1).trim();
      if (!curDesc) return null;
      curRaw = '';
    } else {
      if (curDesc === null || /\s/.test(line)) return null;
      curRaw += line;
    }
  }
  if (curDesc !== null) {
    records.push({ desc: curDesc, raw: curRaw });
  }
  if (records.length < expectedChains.length) return null;

  const grouped: Array<Array<{ desc: string; raw: string }>> = [];
  let chainIdx = -1;

  for (const rec of records) {
    let aligned: string;
    try {
      aligned = extractAlignedSequence(rec.raw, rec.desc).aligned;
    } catch {
      return null;
    }

    if (chainIdx === -1) {
      if (aligned !== expectedChains[0]) return null;
      chainIdx = 0;
      grouped.push([rec]);
      continue;
    }

    const nextIdx = chainIdx + 1;
    const canStartNext =
      nextIdx < expectedChains.length &&
      aligned === expectedChains[nextIdx] &&
      (aligned.length !== expectedChains[chainIdx].length ||
        expectedChains[nextIdx] !== expectedChains[chainIdx] ||
        grouped[chainIdx].length >= 1);

    // Only transition on identical-length chains when the header looks like a new query
    // or when remaining records exactly require starting the next chain, OR when lengths/seqs differ.
    const mustTransition =
      nextIdx < expectedChains.length &&
      aligned === expectedChains[nextIdx] &&
      (aligned.length !== expectedChains[chainIdx].length ||
        aligned !== expectedChains[chainIdx]);

    if (mustTransition || (canStartNext && grouped.length < expectedChains.length && /^query|101|^chain/i.test(rec.desc))) {
      chainIdx = nextIdx;
      grouped.push([rec]);
    } else if (aligned.length === expectedChains[chainIdx].length) {
      grouped[chainIdx].push(rec);
    } else {
      return null;
    }
  }

  if (grouped.length !== expectedChains.length) return null;
  return grouped.map((recs) => `${recs.map((r) => `>${r.desc}\n${r.raw}`).join('\n')}\n`);
}

function normalizeExpectedSeq(seq: string): string {
  return seq
    .toUpperCase()
    .replace(/\s+/g, '')
    .split('')
    .map(normalizeResidueCode)
    .join('');
}

/**
 * Parse and validate user-supplied `.a3m` text.
 * Supports:
 * - Single-chain A3M
 * - Concatenated multi-chain A3M (with or without leading `#length` metadata line)
 * - Per-chain A3M blocks separated by NUL (`\0`), `#` dividers, or `>` chain boundaries
 */
export function validateAndParseA3mText(
  rawText: string,
  expectedChainSeqs?: readonly string[],
): ValidatedA3mResult {
  if (typeof rawText !== 'string' || rawText.trim().length === 0) {
    throw new Error('Custom A3M alignment is empty. Paste or upload a valid .a3m file.');
  }

  const normalizedChains = (expectedChainSeqs ?? [])
    .map(normalizeExpectedSeq)
    .filter((s) => s.length > 0);
  const expectedConcat = normalizedChains.join('');

  const tryPerChainBlocks = (rawBlocks: string[]): ValidatedA3mResult | null => {
    if (normalizedChains.length <= 1 || rawBlocks.length !== normalizedChains.length) {
      return null;
    }
    const parsedBlocks = rawBlocks.map((b) => parseSingleA3mBlock(b));
    for (let i = 0; i < normalizedChains.length; i++) {
      if (parsedBlocks[i].query !== normalizedChains[i]) {
        return null;
      }
    }
    const totalDepth = Math.max(...parsedBlocks.map((b) => b.depth));
    const distinctDepth = Math.max(...parsedBlocks.map((b) => b.distinctDepth));
    return {
      a3m: parsedBlocks.map((b) => b.normalizedA3m).join('\n'),
      perChainA3ms: parsedBlocks.map((b) => b.normalizedA3m),
      query: parsedBlocks.map((b) => b.query).join(''),
      depth: totalDepth,
      distinctDepth,
      length: expectedConcat.length,
      wasPerChainBlocks: true,
    };
  };

  // 1. NUL-separated per-chain blocks (e.g. ColabFold uniref.a3m / pair.a3m)
  if (rawText.includes('\0')) {
    const nulBlocks = rawText
      .split(/\0+/)
      .map((s) => s.trim())
      .filter(Boolean);
    const matched = tryPerChainBlocks(nulBlocks);
    if (matched) return matched;
    if (normalizedChains.length === 1 && nulBlocks.length >= 1) {
      const single = parseSingleA3mBlock(nulBlocks[0]);
      if (single.query === expectedConcat) {
        return {
          a3m: single.normalizedA3m,
          query: single.query,
          depth: single.depth,
          distinctDepth: single.distinctDepth,
          length: single.length,
          wasPerChainBlocks: false,
        };
      }
    }
  }

  // 2. `#`-separated per-chain blocks for multi-chain targets
  if (normalizedChains.length > 1) {
    const hashBlocks = splitByHashBlocks(rawText);
    if (hashBlocks.length === normalizedChains.length) {
      try {
        const matched = tryPerChainBlocks(hashBlocks);
        if (matched) return matched;
      } catch {
        // Fall through to single-block / FASTA-boundary parsing
      }
    }
  }

  // 3. Single-block A3M (monomer or pre-concatenated complex)
  let singleError: Error | null = null;
  let singleBlock: ParsedBlock | null = null;
  try {
    singleBlock = parseSingleA3mBlock(rawText);
  } catch (err) {
    singleError = err instanceof Error ? err : new Error(String(err));
  }

  if (singleBlock) {
    if (expectedConcat.length === 0 || singleBlock.query === expectedConcat) {
      return {
        a3m: singleBlock.normalizedA3m,
        query: singleBlock.query,
        depth: singleBlock.depth,
        distinctDepth: singleBlock.distinctDepth,
        length: singleBlock.length,
        wasPerChainBlocks: false,
      };
    }
  }

  // 4. Concatenated per-chain A3M blocks separated only by `>` FASTA headers
  if (normalizedChains.length > 1) {
    const fastaBlocks = splitByFastaChainBoundaries(rawText, normalizedChains);
    if (fastaBlocks) {
      const matched = tryPerChainBlocks(fastaBlocks);
      if (matched) return matched;
    }
  }

  if (singleBlock && expectedConcat.length > 0) {
    throw new Error(
      `Custom A3M query sequence mismatch: expected ${expectedConcat.length} aa (${seqPreview(expectedConcat)})` +
        ` across ${normalizedChains.length} chain${normalizedChains.length > 1 ? 's' : ''}, ` +
        `but A3M query is ${singleBlock.query.length} aa (${seqPreview(singleBlock.query)}).`,
    );
  }

  throw singleError ?? new Error('Invalid custom A3M alignment');
}

/**
 * Compute the per-chain `off` mask for `target-only` MSA mode.
 *
 * Rules:
 * - Chains explicitly marked `'target'` in `chainRoles` OR listed in `lockedChains` get `off = false` (searched via MMseqs2).
 * - De novo `'binder'` / `'scaffold'` chains get `off = true` (query-only, skipped from MMseqs2).
 * - If at least one chain has a target/locked signal, any remaining unmarked chains are treated as de novo (`off = true`).
 * - Security fail-closed rule: if NO chain has a positively identified `'target'` role or lock (e.g. an unannotated complex or mismatched chain IDs), skip ALL chains (`off = true` for every chain) so private de novo sequences are NEVER silently transmitted to ColabFold.
 *
 * @param chainIds Physical protein chain IDs in fold order (e.g. `['A', 'B']`).
 */
export function computeTargetOnlyOffMask(
  chainIds: readonly string[],
  lockedChains?: readonly string[],
  chainRoles?: Readonly<Record<string, 'target' | 'binder' | 'scaffold'>>,
): boolean[] {
  if (chainIds.length === 0) return [];

  const lockedSet = new Set(
    (lockedChains ?? []).map((c) => String(c).trim()).filter((c) => c.length > 0),
  );
  const roles = chainRoles ?? {};

  const isTarget = chainIds.map((id) => roles[id] === 'target' || lockedSet.has(id));
  const hasTargetSignal = isTarget.some(Boolean);

  if (!hasTargetSignal) {
    // Fail closed: without an explicit target/locked chain annotation, keep all chains local.
    return chainIds.map(() => true);
  }

  // Search positively identified target/locked chains only; skip binders, scaffolds, and unmarked chains.
  return chainIds.map((_, idx) => !isTarget[idx]);
}

/**
 * Parse a candidate protein sequence while preserving inline chain separators (`:` or `/`)
 * BEFORE stripping whitespace/delimiters, and validate against supplied `chainLengths`.
 */
export function parseCandidateProteinChains(
  rawSequence: string,
  declaredChainLengths?: readonly number[],
): { cleanSeq: string; chainLengths: number[]; chainSeqs: string[] } {
  const upper = rawSequence.toUpperCase().trim();
  const hasInlineSep = /[:/]/.test(upper);
  const inlineParts = hasInlineSep
    ? upper
        .split(/[:/]+/)
        .map((seg) => seg.replace(/[\s,]+/g, ''))
        .filter(Boolean)
    : [];

  const stripped = hasInlineSep
    ? inlineParts.join('')
    : upper.replace(/[\s/:,]+/g, '');
  const cleanSeq = stripped.replace(/[^ACDEFGHIKLMNPQRSTVWYX]/g, 'X');

  const validDeclared =
    Array.isArray(declaredChainLengths) &&
    declaredChainLengths.length > 0 &&
    declaredChainLengths.every((n) => Number.isInteger(n) && n > 0) &&
    declaredChainLengths.reduce((a, b) => a + b, 0) === cleanSeq.length;

  let chainLengths: number[];
  if (hasInlineSep && inlineParts.length > 1) {
    const inlineLengths = inlineParts.map((s) => s.length);
    if (
      validDeclared &&
      declaredChainLengths!.length === inlineLengths.length &&
      declaredChainLengths!.every((n, i) => n === inlineLengths[i])
    ) {
      chainLengths = [...declaredChainLengths!];
    } else if (validDeclared) {
      // Inline separators present AND declared chainLengths conflict -> trust inline separators
      chainLengths = inlineLengths;
    } else {
      chainLengths = inlineLengths;
    }
  } else if (validDeclared) {
    chainLengths = [...declaredChainLengths!];
  } else {
    chainLengths = [cleanSeq.length];
  }

  const chainSeqs: string[] = [];
  let offset = 0;
  for (const len of chainLengths) {
    chainSeqs.push(cleanSeq.slice(offset, offset + len));
    offset += len;
  }
  return { cleanSeq, chainLengths, chainSeqs };
}

/**
 * Resolve ordered physical chain IDs for target-only MSA role lookup.
 * When a candidate carries explicit `candChainIds` matching `chainCount`, uses them verbatim.
 * Otherwise synthesizes `A, B, ...` ONLY if all declared `lockedChains` / `chainRoles` keys
 * are accounted for within that synthetic alphabet; if any role/lock references a chain ID
 * outside the synthetic list (e.g. physical chains `B, C` when synthetic is `A, B`), returns
 * `ambiguous: true` so target-only MSA fails closed and never leaks a binder sequence.
 */
export function resolveMsaChainIdsSafely(opts: {
  chainCount: number;
  candidateChainIds?: readonly string[];
  lockedChains?: readonly string[];
  chainRoles?: Readonly<Record<string, 'target' | 'binder' | 'scaffold'>>;
}): { chainIds: string[]; ambiguous: boolean } {
  const { chainCount, candidateChainIds, lockedChains, chainRoles } = opts;
  if (Array.isArray(candidateChainIds) && candidateChainIds.length === chainCount) {
    return { chainIds: [...candidateChainIds], ambiguous: false };
  }
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const synthetic = Array.from({ length: chainCount }, (_, idx) =>
    idx < alphabet.length ? alphabet[idx] : `C${idx + 1}`,
  );
  const synthSet = new Set(synthetic);
  const annotatedKeys = new Set<string>([
    ...(lockedChains ?? []),
    ...Object.keys(chainRoles ?? {}),
  ]);
  for (const k of annotatedKeys) {
    if (!synthSet.has(k)) {
      return { chainIds: [], ambiguous: true };
    }
  }
  return { chainIds: synthetic, ambiguous: false };
}

export function createMsaSearchCache(): MsaSearchCache {
  return new Map<string, MsaSearchCacheEntry>();
}

function isMonomerGraphModel(modelKey: string, graphFamilyFn?: (model: string) => string): boolean {
  if (graphFamilyFn) {
    return graphFamilyFn(modelKey) === 'monomer';
  }
  return modelKey === 'monomer' || /^monomer-[2-5]$/.test(modelKey);
}

function countRowsAndDistinct(
  a3m: string,
  parseA3mFn?: (text: string) => ParsedA3mLike,
): { depth: number; distinctDepth: number } {
  if (parseA3mFn) {
    const parsed = parseA3mFn(a3m);
    return {
      depth: parsed.depth,
      distinctDepth: new Set(parsed.sequences).size,
    };
  }
  const parsed = parseSingleA3mBlock(a3m);
  return {
    depth: parsed.depth,
    distinctDepth: parsed.distinctDepth,
  };
}

function formatMmseqsProgress(ev: MmseqsProgressEvent): string {
  const phase = ev.phase ?? ev.status ?? 'searching';
  const searchTag =
    ev.search === 'paired'
      ? ' (paired)'
      : typeof ev.search === 'number' && typeof ev.searches === 'number' && ev.searches > 1
        ? ` (chain ${ev.search + 1}/${ev.searches})`
        : '';
  return `Fetching MMseqs2 MSA${searchTag}: ${phase}...`;
}

/**
 * Resolve the MSA alignment for one candidate sequence across all 4 MSA modes
 * (`single`, `mmseqs2`, `target-only`, `a3m`), reusing an in-run search cache so
 * folding 4–8 MPNN binder candidates against the same target queries `api.colabfold.com` ONCE.
 */
export async function resolveMsaForCandidate(opts: ResolveMsaOptions): Promise<ResolvedMsaResult> {
  const {
    msaMode,
    modelKey,
    chainSeqs,
    chainIds,
    lockedChains,
    chainRoles,
    msaA3mText,
    mmseqsApi,
    cache = createMsaSearchCache(),
    onProgress,
    signal,
    requestOptions,
  } = opts;

  const seqs = Array.from(chainSeqs);
  if (seqs.length === 0) {
    throw new Error('resolveMsaForCandidate requires at least one protein chain sequence');
  }

  // 1. Single-sequence mode (100% local, zero external calls)
  if (msaMode === 'single') {
    const a3m =
      seqs.length > 1
        ? mmseqsApi.singleSequenceA3m(seqs, modelKey)
        : `>query\n${seqs[0]}\n`;
    return {
      msaMode: 'single',
      a3m,
      af3Alignment: null,
      msaDepth: 1,
      distinctDepth: 1,
      pairedDepth: 0,
      offMask: seqs.map(() => true),
      cacheHit: false,
    };
  }

  // 2. Custom user-supplied .a3m alignment (100% local)
  if (msaMode === 'a3m') {
    const validated = validateAndParseA3mText(msaA3mText ?? '', seqs);
    if (validated.wasPerChainBlocks && validated.perChainA3ms) {
      const merged = mmseqsApi.mergeSearchedChains({
        sequences: seqs,
        chainA3ms: validated.perChainA3ms,
        model: modelKey,
      });
      const counts = countRowsAndDistinct(merged.a3m, mmseqsApi.parseA3m);
      onProgress?.({
        phase: 'msa',
        detail: `MSA depth: ${counts.depth} seqs (.a3m)`,
      });
      return {
        msaMode: 'a3m',
        a3m: merged.a3m,
        af3Alignment: merged.blocks,
        msaDepth: counts.depth,
        distinctDepth: counts.distinctDepth,
        pairedDepth: 0,
        offMask: seqs.map(() => false),
        cacheHit: false,
      };
    }

    const counts = countRowsAndDistinct(validated.a3m, mmseqsApi.parseA3m);
    onProgress?.({
      phase: 'msa',
      detail: `MSA depth: ${counts.depth} seqs (.a3m)`,
    });
    return {
      msaMode: 'a3m',
      a3m: validated.a3m,
      af3Alignment: validated.a3m,
      msaDepth: counts.depth,
      distinctDepth: counts.distinctDepth,
      pairedDepth: 0,
      offMask: seqs.map(() => false),
      cacheHit: false,
    };
  }

  // 3. MMseqs2 (`mmseqs2` or `target-only`)
  const offMask =
    msaMode === 'target-only'
      ? computeTargetOnlyOffMask(chainIds, lockedChains, chainRoles)
      : seqs.map(() => false);
  const keptSeqs = seqs.filter((_, idx) => !offMask[idx]);

  if (keptSeqs.length === 0) {
    const a3m =
      seqs.length > 1
        ? mmseqsApi.singleSequenceA3m(seqs, modelKey)
        : `>query\n${seqs[0]}\n`;
    onProgress?.({
      phase: 'msa',
      detail: 'MSA depth: 1 seqs (all chains de novo; skipped MMseqs2)',
    });
    return {
      msaMode,
      a3m,
      af3Alignment: null,
      msaDepth: 1,
      distinctDepth: 1,
      pairedDepth: 0,
      offMask,
      cacheHit: false,
    };
  }

  // 3a. Exactly one chain is searched via MMseqs2 (monomer OR target-only with 1 target chain)
  if (keptSeqs.length === 1) {
    const targetSeq = keptSeqs[0];
    const cacheKey = `single:${targetSeq}`;
    let entry = cache.get(cacheKey);
    let cacheHit = false;

    if (entry && entry.kind === 'single') {
      cacheHit = true;
    } else {
      onProgress?.({ phase: 'msa', detail: 'Fetching MMseqs2 MSA...' });
      const searched = await mmseqsApi.generateMmseqs2Msa(targetSeq, {
        ...requestOptions,
        model: modelKey,
        signal,
        onProgress: (ev) => onProgress?.({ phase: 'msa', detail: formatMmseqsProgress(ev), raw: ev }),
      });
      entry = {
        kind: 'single',
        a3m: searched.a3m,
        depth: searched.depth,
        templateHits: searched.templateHits,
      };
      cache.set(cacheKey, entry);
    }

    if (seqs.length === 1) {
      const counts = countRowsAndDistinct(entry.a3m, mmseqsApi.parseA3m);
      onProgress?.({
        phase: 'msa',
        detail: `MSA depth: ${counts.depth} seqs${cacheHit ? ' (cached)' : ''}`,
      });
      return {
        msaMode,
        a3m: entry.a3m,
        af3Alignment: entry.a3m,
        msaDepth: counts.depth,
        distinctDepth: counts.distinctDepth,
        pairedDepth: 0,
        offMask,
        cacheHit,
      };
    }

    // Multi-chain fold where only 1 chain was searched (e.g. target-only binder complex)
    const expanded = mmseqsApi.expandSearchedChains({
      chains: seqs,
      off: offMask,
      parts: [entry.a3m],
      pairedA3ms: undefined,
      templateHits: entry.templateHits,
    });
    const merged = mmseqsApi.mergeSearchedChains({
      sequences: seqs,
      chainA3ms: expanded.chainA3ms,
      pairedA3ms: expanded.pairedA3ms,
      model: modelKey,
    });
    const counts = countRowsAndDistinct(merged.a3m, mmseqsApi.parseA3m);
    onProgress?.({
      phase: 'msa',
      detail: `MSA depth: ${counts.depth} seqs${cacheHit ? ' (cached)' : ''}`,
    });
    return {
      msaMode,
      a3m: merged.a3m,
      af3Alignment: merged.blocks,
      msaDepth: counts.depth,
      distinctDepth: counts.distinctDepth,
      pairedDepth: 0,
      offMask,
      cacheHit,
    };
  }

  // 3b. Multiple chains searched via MMseqs2 (with species pairing when model is not AF2 monomer)
  const needsPairing =
    !isMonomerGraphModel(modelKey, mmseqsApi.graphFamily) && new Set(keptSeqs).size > 1;
  const baseKey = `complex:${JSON.stringify(keptSeqs)}`;
  const cacheKey = `${baseKey}|pair:${needsPairing ? 1 : 0}`;
  let entry =
    cache.get(cacheKey) ?? (!needsPairing ? cache.get(`${baseKey}|pair:1`) : undefined);
  let cacheHit = false;

  if (entry && entry.kind === 'complex') {
    cacheHit = true;
  } else {
    onProgress?.({ phase: 'msa', detail: 'Fetching MMseqs2 MSA...' });
    const searched = await mmseqsApi.generateMmseqs2ComplexMsa(keptSeqs, {
      ...requestOptions,
      model: modelKey,
      signal,
      onProgress: (ev) => onProgress?.({ phase: 'msa', detail: formatMmseqsProgress(ev), raw: ev }),
    });
    entry = {
      kind: 'complex',
      chainA3ms: searched.chainA3ms,
      pairedA3ms: searched.pairedA3ms,
      depth: searched.depth,
      pairedDepth: searched.pairedDepth ?? 0,
      templateHits: searched.templateHits,
    };
    cache.set(cacheKey, entry);
  }

  const expanded = offMask.some(Boolean)
    ? mmseqsApi.expandSearchedChains({
        chains: seqs,
        off: offMask,
        parts: entry.chainA3ms,
        pairedA3ms: entry.pairedA3ms,
        templateHits: entry.templateHits,
      })
    : { chainA3ms: entry.chainA3ms, pairedA3ms: entry.pairedA3ms };

  const merged = mmseqsApi.mergeSearchedChains({
    sequences: seqs,
    chainA3ms: expanded.chainA3ms,
    pairedA3ms: expanded.pairedA3ms,
    model: modelKey,
  });
  const counts = countRowsAndDistinct(merged.a3m, mmseqsApi.parseA3m);
  onProgress?.({
    phase: 'msa',
    detail: `MSA depth: ${counts.depth} seqs${cacheHit ? ' (cached)' : ''}`,
  });
  return {
    msaMode,
    a3m: merged.a3m,
    af3Alignment: merged.blocks,
    msaDepth: counts.depth,
    distinctDepth: counts.distinctDepth,
    pairedDepth: needsPairing ? entry.pairedDepth : 0,
    offMask,
    cacheHit,
  };
}

/** Max allowed recycle count per LocalFold model (`20` for `mol1` & `multimer*`, `10` otherwise). */
export function maxRecyclesForLocalFoldModel(modelKey: string): number {
  if (modelKey === 'mol1' || modelKey === 'multimer' || /^multimer-[2-5]$/.test(modelKey)) {
    return 20;
  }
  return 10;
}

/** Clamp recycles to the upstream-faithful ceiling for `modelKey`. */
export function clampLocalFoldRecycles(
  modelKey: string,
  rawRecycles: unknown,
  fallback = 1,
): number {
  const max = maxRecyclesForLocalFoldModel(modelKey);
  const parsed = Number(rawRecycles ?? fallback);
  const safe = Number.isFinite(parsed) ? Math.round(parsed) : fallback;
  return Math.max(0, Math.min(max, safe));
}

/**
 * Resolve AF2 recycle early-stopping tolerance (Å):
 * - AF2 Monomer defaults to `0` (run all requested recycles, matching upstream AF2 monomer).
 * - AF2-Multimer defaults to `0.5` Å unless explicitly overridden.
 */
export function resolveAf2Tolerance(modelKey: string, rawTolerance: unknown): number {
  const isMultimer = modelKey === 'multimer' || /^multimer-[2-5]$/.test(modelKey);
  const defaultTol = isMultimer ? 0.5 : 0;
  if (rawTolerance === undefined || rawTolerance === null || rawTolerance === '') {
    return defaultTol;
  }
  const parsed = Number(rawTolerance);
  if (!Number.isFinite(parsed)) return defaultTol;
  return Math.max(0, Math.min(5, parsed));
}

/**
 * Compute AF2 / AF2-Multimer MSA cluster & extra-MSA budgets.
 * Single-sequence folds stay minimal (`chainCount + 1` / `1`, `maskFraction: 0`),
 * while real MSAs (`msaDepth > 1`) unlock AF2's full 508-cluster / 1024-extra capacity.
 */
export function computeAf2MsaBudgets(
  msaDepth: number,
  chainCount: number,
  distinctDepth: number = msaDepth,
): { maxMsaSequences: number; maxExtraSequences: number; maskFraction: number } {
  if (msaDepth > 1) {
    return {
      maxMsaSequences: Math.min(508, Math.max(chainCount + 1, msaDepth)),
      maxExtraSequences: Math.min(1024, msaDepth),
      maskFraction: distinctDepth > 1 ? 0.15 : 0,
    };
  }
  return {
    maxMsaSequences: Math.max(1, chainCount + 1),
    maxExtraSequences: 1,
    maskFraction: 0,
  };
}

function resolveCandidateMultimerScore(cand: FoldedCandidate): number | null {
  if (cand.iptmIsMonomerApprox) return null;
  if (typeof cand.multimerScore === 'number' && Number.isFinite(cand.multimerScore)) {
    return cand.multimerScore;
  }
  if (
    typeof cand.iptm === 'number' &&
    Number.isFinite(cand.iptm) &&
    typeof cand.ptm === 'number' &&
    Number.isFinite(cand.ptm)
  ) {
    return 0.8 * cand.iptm + 0.2 * cand.ptm;
  }
  return null;
}

/**
 * Select the best representative fold for `bestStructure`.
 * Multi-chain complexes with real `multimerScore` (`0.8 * ipTM + 0.2 * pTM`) are ranked by
 * `multimerScore`; monomers and folds without interface scores delegate to `selectFoldRepresentative`.
 */
export function selectLocalFoldRepresentative(
  folds: readonly FoldedCandidate[],
  opts: { collectionKind?: CollectionKind } = {},
): { fold: FoldedCandidate; selection: RepresentativeSelection } | null {
  if (folds.length === 0) return null;

  const isMultiChain = folds.some((f) => (f.chainLengths?.length ?? 1) > 1);
  if (isMultiChain) {
    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < folds.length; i++) {
      const score = resolveCandidateMultimerScore(folds[i]);
      if (score !== null && score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) {
      const best = folds[bestIdx];
      return {
        fold: best,
        selection: {
          selectedKey: best.key ?? best.sequenceId,
          criterion: 'highest-multimer-score',
          totalMembers: folds.length,
          selectedIndex: bestIdx,
          scoreValue: bestScore,
        },
      };
    }
  }

  return selectFoldRepresentative(folds, opts);
}
