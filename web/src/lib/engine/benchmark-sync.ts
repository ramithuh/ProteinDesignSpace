// Canonical pure benchmark seeding & symmetric reconciliation module.
// Shared across live UI graph updates (`page-state-helpers.ts`, `+page.svelte`),
// AI workflow planning (`workflow-planner.ts`), and URL share typing (`share.ts`).

import {
  BENCHMARK_TARGETS,
  adaptBenchmarkContigForLength,
  getBenchmarkById,
  type BenchmarkCategory,
  type BenchmarkTargetMeta,
} from './benchmarks';
import type { WorkflowGraph, WorkflowNode } from './types';

/**
 * UI provenance / tracking keys persisted on generator nodes so auto-seeding
 * stays symmetric across benchmark switches, disconnects, undo/redo, and `#p=` share links.
 */
export const UI_TRACKING_PARAM_HINTS: Record<string, unknown> = {
  autoSeededBenchmarkId: '',
  autoSeededFromBenchmark: '',
  autoSeededLength: 0,
  userEditedTask: false,
  userEditedLength: false,
};

export const UI_TRACKING_PARAM_KEYS: ReadonlySet<string> = new Set(
  Object.keys(UI_TRACKING_PARAM_HINTS),
);

/** All template-conditioned generator adapters participating in symmetric benchmark sync (`WF-06`). */
export const BENCHMARK_SYNC_ADAPTER_IDS: ReadonlySet<string> = new Set([
  'rfd4-proteina',
  'genie3',
  'boltzgen',
  'protpardelle',
  'rfdiffusion',
  'genie2',
]);

/** Return whether a curated benchmark can be auto-seeded onto `adapterId`. */
export function isBenchmarkSupportedForAdapter(
  adapterId: string,
  bench: BenchmarkTargetMeta,
): boolean {
  switch (adapterId) {
    case 'rfd4-proteina':
      return true;
    case 'genie3':
      return Boolean(bench.supportsGenie3);
    case 'boltzgen':
      return bench.category === 'binder' || bench.category === 'motif';
    case 'protpardelle':
      return bench.category === 'motif';
    case 'rfdiffusion':
      return (
        bench.category === 'motif' &&
        Boolean(bench.supportsGenie3) &&
        !bench.conditionParams.unindexedMotif
      );
    case 'genie2':
      return bench.category === 'motif' && Boolean(bench.supportsGenie3);
    default:
      return false;
  }
}

/** Convert comma-separated motif contigs (`10-10,A16-35,10-10`) to RFdiffusion `/`-delimited syntax. */
export function formatRfdiffusionContig(contig: string | undefined): string {
  const raw = String(contig ?? '').trim();
  if (!raw) return '';
  return raw
    .split(/[\s,]+/)
    .map((t) => t.trim())
    .filter((t) => Boolean(t) && t !== '/0' && t !== '0')
    .join('/');
}

/** Extract the first single contiguous chain-residue span for Genie 2's single-motif `motifRange`. */
export function extractFirstContiguousMotifRange(spec: string | undefined): string {
  const raw = String(spec ?? '').trim();
  if (!raw) return '';
  const tokens = raw.split(/[,;\s]+/).filter(Boolean);
  for (const tok of tokens) {
    if (/^(?:([A-Za-z]):?)?(-?\d+)(?:-(-?\d+))?$/.test(tok)) {
      return tok;
    }
  }
  return '';
}

/** Build cleared task parameters for `"↺ Reset to benchmark"` in the inspector (`WF-04`, `WF-06`). */
export function buildResetToBenchmarkParams(
  adapterId: string,
  prevParams: Record<string, unknown>,
): Record<string, unknown> {
  const next: Record<string, unknown> = { ...prevParams };
  delete next.userEditedTask;
  delete next.userEditedLength;
  delete next.autoSeededBenchmarkId;
  delete next.autoSeededFromBenchmark;
  delete next.autoSeededLength;
  const defLen = getAdapterDefaultLength(adapterId);
  if (adapterId === 'rfd4-proteina') {
    next.mode = 'unconditional';
    next.length = defLen;
    next.contig = '';
    next.lockChain = '';
    next.hotspots = '';
    next.motifSpan = '';
    next.motifAtoms = '';
    next.unindexedMotif = false;
    next.hbondMode = 'none';
    next.hbondSelection = '';
    next.hbondDonorSelection = '';
    next.hbondAcceptorSelection = '';
    next.cathCode = '';
    next.customSpecYaml = '';
  } else if (adapterId === 'genie3') {
    next.taskMode = 'unconditional';
    next.length = defLen;
    next.contig = '';
    next.lockChain = '';
    next.hotspots = '';
    next.motifSpan = '';
  } else if (adapterId === 'boltzgen') {
    next.taskMode = 'unconditional';
    next.length = defLen;
    next.hotspots = '';
  } else if (adapterId === 'protpardelle') {
    next.taskMode = 'unconditional';
    next.length = defLen;
    next.motifSelection = '';
  } else if (adapterId === 'rfdiffusion') {
    next.taskMode = 'unconditional';
    next.length = defLen;
    next.contig = '';
  } else if (adapterId === 'genie2') {
    next.length = defLen;
    next.motifRange = '';
  }
  return next;
}

export interface BenchmarkReconcileResult {
  graph: WorkflowGraph;
  changedNodeIds: string[];
  patches: Record<string, Record<string, unknown>>;
}

export interface ParsedResidueInterval {
  chain: string;
  start: number;
  end: number;
}

/**
 * Parse the designed/target residue length from a benchmark or generator contig string.
 * Supports binder `/0` suffixes (`A17-132,/0,70-70` -> 70), bare ranges (`180-180` -> 180),
 * and motif scaffolding contigs (`28-28,B19-27,28-28` -> 65).
 */
export function parseLengthFromContig(contig: string | undefined): number | null {
  const raw = String(contig ?? '').trim();
  if (!raw) return null;
  if (raw.includes('/0')) {
    const afterBreak = raw.split('/0').slice(1).join(',');
    const tokens = afterBreak.split(/[,\s]+/).filter(Boolean);
    for (const tok of tokens) {
      const m = tok.match(/^(\d+)(?:-(\d+))?[PRDN]?$/i);
      if (m) {
        const lo = Number(m[1]);
        const hi = m[2] ? Number(m[2]) : lo;
        return Math.round((lo + hi) / 2);
      }
    }
  }
  const bare = raw.match(/^(\d+)(?:-(\d+))?[PRDN]?$/i);
  if (bare) {
    const lo = Number(bare[1]);
    const hi = bare[2] ? Number(bare[2]) : lo;
    return Math.round((lo + hi) / 2);
  }
  const tokens = raw.split(/[,\s]+/).filter(Boolean);
  if (tokens.length > 0) {
    let total = 0;
    let matchedAny = false;
    for (const tok of tokens) {
      const lenMatch = tok.match(/^(\d+)(?:-(\d+))?[PRDN]?$/i);
      if (lenMatch) {
        const lo = Number(lenMatch[1]);
        const hi = lenMatch[2] ? Number(lenMatch[2]) : lo;
        total += Math.round((lo + hi) / 2);
        matchedAny = true;
        continue;
      }
      const spanMatch = tok.match(/^[A-Za-z_]+(-?\d+)(?:-(-?\d+))?$/);
      if (spanMatch) {
        const lo = Number(spanMatch[1]);
        const hi = spanMatch[2] !== undefined ? Number(spanMatch[2]) : lo;
        if (hi >= lo) {
          total += hi - lo + 1;
          matchedAny = true;
          continue;
        }
      }
      return null;
    }
    if (matchedAny && total > 0) return total;
  }
  return null;
}

/**
 * Parse chain/residue intervals from a residue selector, cropSegments, or contig string
 * (ignoring bare length tokens like `20-20`, chain breaks `/0`, and ligand selectors `A:ZN`).
 */
export function parseResidueIntervals(spec: string | undefined): ParsedResidueInterval[] {
  const raw = String(spec ?? '').trim();
  if (!raw) return [];
  const out: ParsedResidueInterval[] = [];
  const tokens = raw.split(/[,;\s/]+/).filter(Boolean);
  for (const tok of tokens) {
    if (tok === '0' || /^\d+(?:-\d+)?[PRDN]?$/i.test(tok)) continue;
    if (tok.includes(':')) continue;
    const span = /^([A-Za-z_]+)(-?\d+)[A-Za-z]?(?:-(-?\d+)[A-Za-z]?)?$/.exec(tok);
    if (span) {
      const chain = span[1].toUpperCase();
      const a = Number(span[2]);
      const b = span[3] !== undefined ? Number(span[3]) : a;
      if (Number.isFinite(a) && Number.isFinite(b)) {
        out.push({ chain, start: Math.min(a, b), end: Math.max(a, b) });
      }
      continue;
    }
    if (/^[A-Za-z_]+$/.test(tok)) {
      out.push({ chain: tok.toUpperCase(), start: -Infinity, end: Infinity });
    }
  }
  return out;
}

function mergeResidueIntervalsByChain(
  intervals: readonly ParsedResidueInterval[],
): Map<string, Array<{ start: number; end: number }>> {
  const byChain = new Map<string, Array<{ start: number; end: number }>>();
  for (const r of intervals) {
    const ch = r.chain.toUpperCase();
    let list = byChain.get(ch);
    if (!list) {
      list = [];
      byChain.set(ch, list);
    }
    list.push({ start: Math.min(r.start, r.end), end: Math.max(r.start, r.end) });
  }
  for (const [ch, list] of byChain.entries()) {
    list.sort((a, b) => a.start - b.start || a.end - b.end);
    const merged: Array<{ start: number; end: number }> = [];
    for (const seg of list) {
      const prev = merged[merged.length - 1];
      if (!prev) {
        merged.push({ ...seg });
      } else if (!Number.isFinite(prev.end) || seg.start <= prev.end + 1) {
        prev.end = Math.max(prev.end, seg.end);
      } else {
        merged.push({ ...seg });
      }
    }
    byChain.set(ch, merged);
  }
  return byChain;
}

/**
 * Parse residue intervals from `motifAtoms` entries (`A102:ND1,CG; A128:SG,CB` -> `[A102, A128]`),
 * ignoring global shorthands (`TIP`, `BKBN`, `ALL`).
 */
export function parseMotifAtomsResidueIntervals(motifAtoms: string | undefined): ParsedResidueInterval[] {
  const raw = String(motifAtoms ?? '').trim();
  if (!raw) return [];
  const upper = raw.toUpperCase();
  if (upper === 'ALL' || upper === '*' || upper === 'BKBN' || upper === 'BACKBONE' || upper === 'TIP') {
    return [];
  }
  const out: ParsedResidueInterval[] = [];
  for (const part of raw.split(/[;\n]+/).map((s) => s.trim()).filter(Boolean)) {
    const colonIdx = part.lastIndexOf(':');
    if (colonIdx <= 0 || colonIdx >= part.length - 1) continue;
    const lhs = part.slice(0, colonIdx).trim();
    out.push(...parseResidueIntervals(lhs));
  }
  return out;
}

/**
 * Return `true` when `spec` specifies at least one residue interval and EITHER:
 * - NONE of its intervals overlap any interval in `refIntervals` (100% disjoint across chains/spans), OR
 * - ANY residue/range token on a polymer chain covered by `refIntervals` falls outside `refIntervals`
 *   (e.g. `A102, A104, A128, A132` against `A102, A104, A129, A132` where `A128` is missing!).
 */
export function isResidueSpecDisjointFromIntervals(
  spec: string | undefined,
  refIntervals: readonly ParsedResidueInterval[],
): boolean {
  const specIntervals = parseResidueIntervals(spec);
  if (specIntervals.length === 0 || refIntervals.length === 0) return false;
  const mergedRefByChain = mergeResidueIntervalsByChain(refIntervals);
  let overlappedAny = false;
  for (const s of specIntervals) {
    const refOnChain = mergedRefByChain.get(s.chain);
    if (!refOnChain) {
      continue;
    }
    const covered = refOnChain.some((r) => r.start <= s.start && s.end <= r.end);
    if (!covered) {
      return true;
    }
    overlappedAny = true;
  }
  return !overlappedAny;
}

/**
 * Return `true` when a residue selector (`motifSpan`, `hotspots`, `motifSelection`, or
 * fixed spans inside `contig`) is completely disjoint from a curated benchmark's canonical
 * `cropSegments` / `conditionParams` intervals (e.g. LLM hallucinating `A1-10` on `1PRW`
 * whose curated motif & crop is `A16-35, A52-71`).
 */
export function isResidueSpecDisjointFromBenchmark(
  spec: string | undefined,
  bench: BenchmarkTargetMeta,
): boolean {
  const refIntervals = [
    ...parseResidueIntervals(bench.cropSegments),
    ...parseResidueIntervals(bench.conditionParams.motifSpan),
    ...parseResidueIntervals(bench.conditionParams.contig),
  ];
  return isResidueSpecDisjointFromIntervals(spec, refIntervals);
}

/** Return all curated benchmarks matching a 4-character RCSB PDB ID (case-insensitive). */
export function getBenchmarksForPdbId(pdbId: string): BenchmarkTargetMeta[] {
  const clean = String(pdbId ?? '').trim().toUpperCase();
  if (!clean) return [];
  return BENCHMARK_TARGETS.filter((b) => b.pdbId.toUpperCase() === clean);
}

/**
 * Detect a curated benchmark match for a typed or quick-chip PDB ID (e.g. `5O45` -> `pdl1_5o45`,
 * `1PRW` -> `rfd_1prw_efhand`) so the inspector can offer a 1-click "Apply benchmark" chip (`WF-03`).
 */
export function detectQuickPdbBenchmarkMatch(
  pdbId: string,
  preferredCategory?: BenchmarkCategory,
): BenchmarkTargetMeta | null {
  const matches = getBenchmarksForPdbId(pdbId);
  if (matches.length === 0) return null;
  if (preferredCategory) {
    const exact = matches.find((b) => b.category === preferredCategory);
    if (exact) return exact;
  }
  return matches[0];
}

/**
 * Build updated `pdb-input` params when applying a curated benchmark, preserving already-fetched
 * `pdbText` / `fetchedPdbId` whenever `fetchedPdbId === bench.pdbId` to avoid redundant RCSB fetches (`WF-03`).
 */
export function buildBenchmarkPdbParams(
  prevParams: Record<string, unknown>,
  bench: BenchmarkTargetMeta,
): Record<string, unknown> {
  const prevFetched = String(prevParams.fetchedPdbId ?? '').trim().toUpperCase();
  const prevPdbText = String(prevParams.pdbText ?? '');
  const sameFetchedPdb =
    prevFetched === bench.pdbId.toUpperCase() && prevPdbText.trim().length > 0;
  return {
    ...prevParams,
    mode: 'pdb-id',
    pdbId: bench.pdbId,
    benchmarkId: bench.id,
    cropSegments: bench.cropSegments ?? '',
    fetchedPdbId: sameFetchedPdb ? prevParams.fetchedPdbId : '',
    pdbText: sameFetchedPdb ? prevPdbText : '',
    filename:
      sameFetchedPdb && String(prevParams.filename ?? '').trim()
        ? prevParams.filename
        : `${bench.pdbId}.cif`,
  };
}

function getSeededBenchmarkId(cp: Record<string, unknown>): string {
  return String(cp.autoSeededBenchmarkId ?? cp.autoSeededFromBenchmark ?? '').trim();
}

function getAdapterDefaultLength(adapterId: string): number {
  if (adapterId === 'boltzgen') return 48;
  if (adapterId === 'rfd4-proteina') return 70;
  return 80;
}

function resolvePrevAutoLength(
  adapterId: string,
  cp: Record<string, unknown>,
  seededId: string,
): number | null {
  const prevBench = seededId ? getBenchmarkById(seededId) : undefined;
  const prevBenchValidForAdapter =
    Boolean(prevBench) && isBenchmarkSupportedForAdapter(adapterId, prevBench!);
  const prevBenchLen = prevBenchValidForAdapter
    ? (parseLengthFromContig(prevBench!.conditionParams.contig) ?? prevBench!.approxTokens)
    : null;
  return typeof cp.autoSeededLength === 'number' &&
    Number.isFinite(cp.autoSeededLength) &&
    cp.autoSeededLength > 0
    ? cp.autoSeededLength
    : prevBenchLen;
}

/**
 * Compute benchmark auto-seed AND symmetric teardown patches for template-conditioned generator
 * nodes (`rfd4-proteina`, `genie3`, `boltzgen`, `protpardelle`, `rfdiffusion`, `genie2`)
 * across a `WorkflowGraph`:
 * - Forward seeding: seeds unconfigured or previously auto-seeded nodes wired to a benchmark `pdb-input`,
 *   adapting the benchmark contig whenever a custom/planned `length` is already present.
 * - Symmetric teardown (`WF-02`): when a previously auto-seeded node (`autoSeededBenchmarkId` set
 *   and `!userEditedTask`) loses its upstream structure wire OR its upstream `pdb-input` exits
 *   benchmark mode (`!benchmarkId`), cleanly reverts the auto-seeded benchmark fields and tracking keys.
 */
export function computeBenchmarkAutoSeedPatches(
  graph: WorkflowGraph,
  opts?: { teardown?: boolean },
): Record<string, Record<string, unknown>> | null {
  const allowTeardown = opts?.teardown !== false;
  let patches: Record<string, Record<string, unknown>> | null = null;

  for (const node of graph.nodes) {
    if (!BENCHMARK_SYNC_ADAPTER_IDS.has(node.adapterId)) {
      continue;
    }
    const cp = node.params ?? {};
    if (Boolean(cp.userEditedTask)) continue;

    const seededId = getSeededBenchmarkId(cp);
    const structEdge = graph.edges.find(
      (e) => e.toNode === node.id && (e.toPort === 'structure' || e.toPort === 'template'),
    );
    const srcNode = structEdge ? graph.nodes.find((n) => n.id === structEdge.fromNode) : undefined;
    const benchId =
      srcNode?.adapterId === 'pdb-input' ? String(srcNode.params.benchmarkId ?? '').trim() : '';
    const bench = benchId ? getBenchmarkById(benchId) : undefined;

    const adapterDefaultLen = getAdapterDefaultLength(node.adapterId);
    const prevAutoLen = resolvePrevAutoLength(node.adapterId, cp, seededId);
    const curLen = Number(cp.length);
    const hasCustomLength =
      Boolean(cp.userEditedLength) ||
      (Number.isFinite(curLen) &&
        curLen > 0 &&
        (seededId
          ? curLen !== prevAutoLen && curLen !== adapterDefaultLen
          : curLen !== adapterDefaultLen));

    // Symmetric teardown when upstream structure wire is removed or upstream exits benchmark mode
    if (!bench) {
      if (!seededId || !allowTeardown) continue;

      if (node.adapterId === 'rfd4-proteina') {
        const curMode = String(cp.mode ?? 'unconditional').trim() || 'unconditional';
        if (curMode === 'partial' || curMode === 'inverse-folding') continue;

        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          mode: 'unconditional',
          contig: '',
          lockChain: '',
          hotspots: '',
          motifSpan: '',
          motifAtoms: '',
          unindexedMotif: false,
          hbondMode: 'none',
          hbondSelection: '',
          hbondDonorSelection: '',
          hbondAcceptorSelection: '',
          cathCode: '',
          customSpecYaml: '',
          ...(shouldResetLen ? { length: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededBenchmarkId;
        delete nextPatch.autoSeededFromBenchmark;
        delete nextPatch.autoSeededLength;
        if (!shouldResetLen && Number.isFinite(curLen) && curLen > 0) {
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (node.adapterId === 'genie3') {
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'unconditional',
          contig: '',
          lockChain: '',
          hotspots: '',
          motifSpan: '',
          ...(shouldResetLen ? { length: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededBenchmarkId;
        delete nextPatch.autoSeededFromBenchmark;
        delete nextPatch.autoSeededLength;
        if (!shouldResetLen && Number.isFinite(curLen) && curLen > 0) {
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (node.adapterId === 'boltzgen') {
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'unconditional',
          hotspots: '',
          ...(shouldResetLen ? { length: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededBenchmarkId;
        delete nextPatch.autoSeededFromBenchmark;
        delete nextPatch.autoSeededLength;
        if (!shouldResetLen && Number.isFinite(curLen) && curLen > 0) {
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (node.adapterId === 'protpardelle') {
        const curMode = String(cp.taskMode ?? 'unconditional').trim() || 'unconditional';
        if (curMode === 'partial') continue;
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'unconditional',
          motifSelection: '',
          ...(shouldResetLen ? { length: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededBenchmarkId;
        delete nextPatch.autoSeededFromBenchmark;
        delete nextPatch.autoSeededLength;
        if (!shouldResetLen && Number.isFinite(curLen) && curLen > 0) {
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (node.adapterId === 'rfdiffusion') {
        const curMode = String(cp.taskMode ?? 'unconditional').trim() || 'unconditional';
        if (curMode === 'symmetry') continue;
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'unconditional',
          contig: '',
          ...(shouldResetLen ? { length: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededBenchmarkId;
        delete nextPatch.autoSeededFromBenchmark;
        delete nextPatch.autoSeededLength;
        if (!shouldResetLen && Number.isFinite(curLen) && curLen > 0) {
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (node.adapterId === 'genie2') {
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          motifRange: '',
          ...(shouldResetLen ? { length: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededBenchmarkId;
        delete nextPatch.autoSeededFromBenchmark;
        delete nextPatch.autoSeededLength;
        if (!shouldResetLen && Number.isFinite(curLen) && curLen > 0) {
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      }
      continue;
    }

    if (seededId === bench.id) {
      // When the user edits the `length` slider on an auto-seeded benchmark without manually editing
      // `contig` (`!userEditedTask`), keep `contig` synchronized via `adaptBenchmarkContigForLength`.
      if (Number.isFinite(curLen) && curLen > 0) {
        const targetLen = Math.round(curLen);
        const bp = bench.conditionParams;
        const parsedBenchLen = parseLengthFromContig(bp.contig);
        const rawCurContig = String(cp.contig ?? '').trim();
        if (
          node.adapterId === 'rfd4-proteina' &&
          rawCurContig &&
          bp.taskPreset === 'custom' &&
          !['partial', 'inverse-folding', 'preset'].includes(String(cp.mode ?? 'unconditional'))
        ) {
          const expectedContig =
            parsedBenchLen !== null && targetLen === parsedBenchLen
              ? bp.contig
              : adaptBenchmarkContigForLength(bp.contig, targetLen, bench.category);
          if (expectedContig !== rawCurContig) {
            const nextPatch: Record<string, unknown> = {
              ...cp,
              contig: expectedContig,
            };
            if (parsedBenchLen !== null && targetLen !== parsedBenchLen) {
              delete nextPatch.autoSeededLength;
            }
            patches ??= {};
            patches[node.id] = nextPatch;
          }
        } else if (node.adapterId === 'genie3' && rawCurContig && bench.supportsGenie3) {
          const expectedContig =
            parsedBenchLen !== null && targetLen === parsedBenchLen
              ? bp.contig
              : adaptBenchmarkContigForLength(bp.contig, targetLen, bench.category);
          if (expectedContig !== rawCurContig) {
            const nextPatch: Record<string, unknown> = {
              ...cp,
              contig: expectedContig,
            };
            if (parsedBenchLen !== null && targetLen !== parsedBenchLen) {
              delete nextPatch.autoSeededLength;
            }
            patches ??= {};
            patches[node.id] = nextPatch;
          }
        } else if (
          node.adapterId === 'rfdiffusion' &&
          rawCurContig &&
          isBenchmarkSupportedForAdapter('rfdiffusion', bench) &&
          String(cp.taskMode ?? 'unconditional') !== 'symmetry'
        ) {
          const adapted =
            parsedBenchLen !== null && targetLen === parsedBenchLen
              ? bp.contig
              : adaptBenchmarkContigForLength(bp.contig, targetLen, bench.category);
          const expectedContig = formatRfdiffusionContig(adapted);
          if (expectedContig !== rawCurContig) {
            const nextPatch: Record<string, unknown> = {
              ...cp,
              contig: expectedContig,
            };
            if (parsedBenchLen !== null && targetLen !== parsedBenchLen) {
              delete nextPatch.autoSeededLength;
            }
            patches ??= {};
            patches[node.id] = nextPatch;
          }
        }
      }
      continue;
    }

    if (node.adapterId === 'rfd4-proteina') {
      const curMode = String(cp.mode ?? 'unconditional').trim() || 'unconditional';
      // Never overwrite partial diffusion or inverse-folding (even if autoSeededBenchmarkId is set)
      if (curMode === 'partial' || curMode === 'inverse-folding') continue;
      // Treat any non-unconditional mode on an unseeded node as explicit (partial / inverse-folding / preset / motif / binder)
      if (!seededId && curMode !== 'unconditional') continue;
      const hasNoTaskParams =
        !String(cp.contig ?? '').trim() &&
        !String(cp.hotspots ?? '').trim() &&
        !String(cp.motifSpan ?? '').trim() &&
        !String(cp.motifAtoms ?? '').trim() &&
        !String(cp.lockChain ?? '').trim() &&
        !String(cp.hbondDonorSelection ?? '').trim() &&
        !String(cp.hbondAcceptorSelection ?? '').trim() &&
        !String(cp.cathCode ?? '').trim() &&
        !String(cp.customSpecYaml ?? '').trim();
      if (!hasNoTaskParams && !seededId) continue;

      const bp = bench.conditionParams;
      const parsedLen = parseLengthFromContig(bp.contig);
      const nextLength = hasCustomLength ? Math.round(curLen) : parsedLen;
      const seededContig =
        hasCustomLength && nextLength !== null
          ? adaptBenchmarkContigForLength(bp.contig, nextLength, bench.category)
          : bp.contig;
      const nextMode =
        bp.taskPreset !== 'custom'
          ? 'preset'
          : bench.category === 'binder'
            ? 'binder'
            : 'motif';
      const nextPatch: Record<string, unknown> = {
        ...cp,
        mode: nextMode,
        ...(bp.taskPreset !== 'custom' ? { presetId: bp.taskPreset } : {}),
        ...(nextLength !== null ? { length: nextLength } : {}),
        contig: seededContig,
        lockChain: bp.lockChain ?? '',
        hotspots: bp.hotspots ?? '',
        motifSpan: bp.motifSpan ?? '',
        motifAtoms: bp.motifAtoms ?? '',
        unindexedMotif: Boolean(bp.unindexedMotif),
        hbondMode: bp.hbondMode ?? 'none',
        hbondSelection: bp.hbondSelection ?? '',
        hbondDonorSelection: bp.hbondDonorSelection ?? '',
        hbondAcceptorSelection: bp.hbondAcceptorSelection ?? '',
        cathCode: '',
        customSpecYaml: '',
        autoSeededBenchmarkId: bench.id,
      };
      delete nextPatch.autoSeededFromBenchmark;
      if (!hasCustomLength && nextLength !== null) {
        nextPatch.autoSeededLength = nextLength;
      } else {
        delete nextPatch.autoSeededLength;
        if (hasCustomLength) {
          nextPatch.userEditedLength = true;
        }
      }
      patches ??= {};
      patches[node.id] = nextPatch;
    } else if (node.adapterId === 'genie3') {
      if (bench.supportsGenie3) {
        const curTaskMode = String(cp.taskMode ?? 'unconditional').trim() || 'unconditional';
        if (!seededId && curTaskMode !== 'unconditional') continue;
        const hasNoTaskParams =
          !String(cp.contig ?? '').trim() &&
          !String(cp.hotspots ?? '').trim() &&
          !String(cp.motifSpan ?? '').trim() &&
          !String(cp.lockChain ?? '').trim();
        if (!hasNoTaskParams && !seededId) continue;

        const bp = bench.conditionParams;
        const parsedLen = parseLengthFromContig(bp.contig);
        const nextLength = hasCustomLength ? Math.round(curLen) : parsedLen;
        const seededContig =
          hasCustomLength && nextLength !== null
            ? adaptBenchmarkContigForLength(bp.contig, nextLength, bench.category)
            : bp.contig;
        const nextTaskMode = bench.category === 'binder' ? 'binder' : 'motif';
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: nextTaskMode,
          ...(nextLength !== null ? { length: nextLength } : {}),
          contig: seededContig,
          lockChain: bp.lockChain ?? '',
          hotspots: bp.hotspots ?? '',
          motifSpan: bp.motifSpan ?? '',
          autoSeededBenchmarkId: bench.id,
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!hasCustomLength && nextLength !== null) {
          nextPatch.autoSeededLength = nextLength;
        } else {
          delete nextPatch.autoSeededLength;
          if (hasCustomLength) {
            nextPatch.userEditedLength = true;
          }
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (seededId) {
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'unconditional',
          contig: '',
          lockChain: '',
          hotspots: '',
          motifSpan: '',
          autoSeededBenchmarkId: bench.id,
          ...(shouldResetLen ? { length: adapterDefaultLen, autoSeededLength: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!shouldResetLen) {
          delete nextPatch.autoSeededLength;
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      }
    } else if (node.adapterId === 'boltzgen') {
      if (isBenchmarkSupportedForAdapter('boltzgen', bench)) {
        const curTaskMode = String(cp.taskMode ?? 'unconditional').trim() || 'unconditional';
        if (!seededId && curTaskMode !== 'unconditional') continue;
        if (!seededId && String(cp.hotspots ?? '').trim()) continue;
        const bp = bench.conditionParams;
        const parsedLen = parseLengthFromContig(bp.contig);
        const clampedParsedLen =
          parsedLen !== null ? Math.max(16, Math.min(160, parsedLen)) : null;
        const nextLength = hasCustomLength ? Math.round(curLen) : clampedParsedLen;
        const nextTaskMode = bench.category === 'binder' ? 'binder' : 'motif';
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: nextTaskMode,
          ...(nextLength !== null ? { length: nextLength } : {}),
          hotspots: nextTaskMode === 'binder' ? (bp.hotspots ?? '') : '',
          autoSeededBenchmarkId: bench.id,
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!hasCustomLength && nextLength !== null) {
          nextPatch.autoSeededLength = nextLength;
        } else {
          delete nextPatch.autoSeededLength;
          if (hasCustomLength) nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (seededId) {
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'unconditional',
          hotspots: '',
          autoSeededBenchmarkId: bench.id,
          ...(shouldResetLen ? { length: adapterDefaultLen, autoSeededLength: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!shouldResetLen) {
          delete nextPatch.autoSeededLength;
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      }
    } else if (node.adapterId === 'protpardelle') {
      const curTaskMode = String(cp.taskMode ?? 'unconditional').trim() || 'unconditional';
      if (curTaskMode === 'partial') continue;
      if (isBenchmarkSupportedForAdapter('protpardelle', bench)) {
        if (!seededId && curTaskMode !== 'unconditional') continue;
        if (!seededId && String(cp.motifSelection ?? '').trim()) continue;
        const bp = bench.conditionParams;
        const parsedLen = parseLengthFromContig(bp.contig);
        const clampedParsedLen =
          parsedLen !== null ? Math.max(20, Math.min(256, parsedLen)) : null;
        const nextLength = hasCustomLength ? Math.round(curLen) : clampedParsedLen;
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'motif',
          ...(nextLength !== null ? { length: nextLength } : {}),
          motifSelection: bp.motifSpan ?? '',
          autoSeededBenchmarkId: bench.id,
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!hasCustomLength && nextLength !== null) {
          nextPatch.autoSeededLength = nextLength;
        } else {
          delete nextPatch.autoSeededLength;
          if (hasCustomLength) nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (seededId) {
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'unconditional',
          motifSelection: '',
          autoSeededBenchmarkId: bench.id,
          ...(shouldResetLen ? { length: adapterDefaultLen, autoSeededLength: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!shouldResetLen) {
          delete nextPatch.autoSeededLength;
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      }
    } else if (node.adapterId === 'rfdiffusion') {
      const curTaskMode = String(cp.taskMode ?? 'unconditional').trim() || 'unconditional';
      if (curTaskMode === 'symmetry') continue;
      if (isBenchmarkSupportedForAdapter('rfdiffusion', bench)) {
        if (!seededId && curTaskMode !== 'unconditional') continue;
        if (!seededId && String(cp.contig ?? '').trim()) continue;
        const bp = bench.conditionParams;
        const parsedLen = parseLengthFromContig(bp.contig);
        const nextLength = hasCustomLength ? Math.round(curLen) : parsedLen;
        const adapted =
          hasCustomLength && nextLength !== null
            ? adaptBenchmarkContigForLength(bp.contig, nextLength, bench.category)
            : bp.contig;
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'motif',
          ...(nextLength !== null ? { length: nextLength } : {}),
          contig: formatRfdiffusionContig(adapted),
          autoSeededBenchmarkId: bench.id,
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!hasCustomLength && nextLength !== null) {
          nextPatch.autoSeededLength = nextLength;
        } else {
          delete nextPatch.autoSeededLength;
          if (hasCustomLength) nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (seededId) {
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          taskMode: 'unconditional',
          contig: '',
          autoSeededBenchmarkId: bench.id,
          ...(shouldResetLen ? { length: adapterDefaultLen, autoSeededLength: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!shouldResetLen) {
          delete nextPatch.autoSeededLength;
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      }
    } else if (node.adapterId === 'genie2') {
      if (isBenchmarkSupportedForAdapter('genie2', bench)) {
        if (!seededId && String(cp.motifRange ?? '').trim()) continue;
        const bp = bench.conditionParams;
        const parsedLen = parseLengthFromContig(bp.contig);
        const nextLength = hasCustomLength ? Math.round(curLen) : parsedLen;
        const fullMotifRange = String(bp.motifSpan ?? '').trim();
        const nextPatch: Record<string, unknown> = {
          ...cp,
          ...(nextLength !== null ? { length: nextLength } : {}),
          motifRange: fullMotifRange,
          autoSeededBenchmarkId: bench.id,
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!hasCustomLength && nextLength !== null) {
          nextPatch.autoSeededLength = nextLength;
        } else {
          delete nextPatch.autoSeededLength;
          if (hasCustomLength) nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      } else if (seededId) {
        const shouldResetLen =
          !cp.userEditedLength &&
          (!Number.isFinite(curLen) ||
            curLen <= 0 ||
            curLen === prevAutoLen ||
            curLen === adapterDefaultLen);
        const nextPatch: Record<string, unknown> = {
          ...cp,
          motifRange: '',
          autoSeededBenchmarkId: bench.id,
          ...(shouldResetLen ? { length: adapterDefaultLen, autoSeededLength: adapterDefaultLen } : {}),
        };
        delete nextPatch.autoSeededFromBenchmark;
        if (!shouldResetLen) {
          delete nextPatch.autoSeededLength;
          nextPatch.userEditedLength = true;
        }
        patches ??= {};
        patches[node.id] = nextPatch;
      }
    }
  }

  return patches;
}

/**
 * Pure synchronous graph-level benchmark reconciler.
 * Returns the updated `WorkflowGraph` along with `changedNodeIds` and raw `patches`.
 */
export function reconcileBenchmarkSeeding(
  graph: WorkflowGraph,
  opts?: { teardown?: boolean },
): BenchmarkReconcileResult {
  const patches = computeBenchmarkAutoSeedPatches(graph, opts) ?? {};
  const changedNodeIds = Object.keys(patches);
  if (changedNodeIds.length === 0) {
    return { graph, changedNodeIds: [], patches: {} };
  }
  const nextGraph: WorkflowGraph = {
    ...graph,
    nodes: graph.nodes.map((n) => (patches[n.id] ? { ...n, params: patches[n.id] } : n)),
  };
  return { graph: nextGraph, changedNodeIds, patches };
}

/**
 * Switch a `pdb-input` node to another curated benchmark (preserving `pdbText` when
 * `fetchedPdbId === bench.pdbId`) or exit benchmark mode (`benchmarkId = null | ''`),
 * then reconcile all downstream generator nodes symmetrically.
 */
export function switchBenchmark(
  graph: WorkflowGraph,
  pdbNodeId: string,
  benchmarkId: string | null,
): BenchmarkReconcileResult {
  const bench = benchmarkId ? getBenchmarkById(benchmarkId) : undefined;
  const updatedGraph: WorkflowGraph = {
    ...graph,
    nodes: graph.nodes.map((n) => {
      if (n.id !== pdbNodeId || n.adapterId !== 'pdb-input') return n;
      if (bench) {
        return {
          ...n,
          params: buildBenchmarkPdbParams(n.params ?? {}, bench),
        };
      }
      const nextParams = { ...(n.params ?? {}) };
      nextParams.benchmarkId = '';
      return { ...n, params: nextParams };
    }),
  };
  return reconcileBenchmarkSeeding(updatedGraph);
}

export interface PlannerBenchmarkSeedOptions {
  /** Explicit user-requested length (e.g. from `ConditionedDesignWorkflowPlan.length`), or null if unspecified. */
  explicitLength?: number | null;
  /** When true, overwrite even overlapping non-empty motifSpan/hotspots/contig (e.g. prompt had no residue selectors). */
  overwriteUnpromptedResidues?: boolean;
}

/**
 * Canonical planner benchmark hydrator shared by all workflow-planner compile & repair paths (`WF-05`):
 * - Hydrates `mode`/`taskMode`, `contig`, `lockChain`, `hotspots`, `motifSpan`, `unindexedMotif`,
 *   `hbondMode`, `hbondSelection`, `presetId`, `autoSeededBenchmarkId`, and `autoSeededLength`.
 * - Overwrites hallucinated `motifSpan`/`hotspots`/`contig` when disjoint from the benchmark's
 *   curated `cropSegments`/`conditionParams` (or when `overwriteUnpromptedResidues` is set).
 * - Leaves absent optional fields (`motifSpan` on binder benchmarks, `hotspots` on motif benchmarks)
 *   unset rather than clobbering `undefined` with empty strings.
 */
export function applyPlannerBenchmarkSeedingToNode(
  gen: WorkflowNode,
  bench: BenchmarkTargetMeta,
  opts?: PlannerBenchmarkSeedOptions,
): { repairedHallucination: boolean } {
  const bp = bench.conditionParams;
  const params = gen.params;
  let repairedHallucination = false;

  if (gen.adapterId === 'rfd4-proteina') {
    const curMode = String(params.mode ?? 'unconditional').trim() || 'unconditional';
    if (curMode === 'partial' || curMode === 'inverse-folding') {
      params.autoSeededBenchmarkId = bench.id;
      return { repairedHallucination: false };
    }
    if (!params.mode || params.mode === 'unconditional') {
      params.mode =
        bp.taskPreset !== 'custom'
          ? 'preset'
          : bench.category === 'binder'
            ? 'binder'
            : 'motif';
    }
    if (bp.taskPreset !== 'custom' && !params.presetId) {
      params.presetId = bp.taskPreset;
    }

    const curHotspots = String(params.hotspots ?? '').trim();
    if (bench.category === 'binder' && bp.hotspots) {
      const disjoint = curHotspots && isResidueSpecDisjointFromBenchmark(curHotspots, bench);
      if (!curHotspots || disjoint || (opts?.overwriteUnpromptedResidues && curHotspots !== bp.hotspots)) {
        if (curHotspots && curHotspots !== bp.hotspots) repairedHallucination = true;
        params.hotspots = bp.hotspots;
      }
    }

    const curMotif = String(params.motifSpan ?? '').trim();
    if (bp.motifSpan) {
      const disjoint = curMotif && isResidueSpecDisjointFromBenchmark(curMotif, bench);
      if (!curMotif || disjoint || (opts?.overwriteUnpromptedResidues && curMotif !== bp.motifSpan)) {
        if (curMotif && curMotif !== bp.motifSpan) repairedHallucination = true;
        params.motifSpan = bp.motifSpan;
      }
    } else if (curMotif && (isResidueSpecDisjointFromBenchmark(curMotif, bench) || opts?.overwriteUnpromptedResidues)) {
      repairedHallucination = true;
      delete params.motifSpan;
    }

    if (bp.lockChain && !String(params.lockChain ?? '').trim()) {
      params.lockChain = bp.lockChain;
    }
    const curMotifAtoms = String(params.motifAtoms ?? '').trim();
    const motifAtomsIntervals = parseMotifAtomsResidueIntervals(curMotifAtoms);
    const motifAtomsDisjoint =
      motifAtomsIntervals.length > 0 &&
      isResidueSpecDisjointFromBenchmark(
        motifAtomsIntervals.map((r) => `${r.chain}${r.start}-${r.end}`).join(', '),
        bench,
      );
    if (bp.motifAtoms && (!curMotifAtoms || motifAtomsDisjoint || (opts?.overwriteUnpromptedResidues && curMotifAtoms !== bp.motifAtoms))) {
      if (curMotifAtoms && curMotifAtoms !== bp.motifAtoms) repairedHallucination = true;
      params.motifAtoms = bp.motifAtoms;
    }
    if (bp.unindexedMotif && params.unindexedMotif !== true) {
      params.unindexedMotif = true;
    }
    if (bp.hbondMode && (!params.hbondMode || params.hbondMode === 'none')) {
      params.hbondMode = bp.hbondMode;
    }
    if (bp.hbondSelection && !String(params.hbondSelection ?? '').trim()) {
      params.hbondSelection = bp.hbondSelection;
    }
    if (bp.hbondDonorSelection && !String(params.hbondDonorSelection ?? '').trim()) {
      params.hbondDonorSelection = bp.hbondDonorSelection;
    }
    if (bp.hbondAcceptorSelection && !String(params.hbondAcceptorSelection ?? '').trim()) {
      params.hbondAcceptorSelection = bp.hbondAcceptorSelection;
    }

    const parsedBenchLen = parseLengthFromContig(bp.contig);
    const curLen = typeof params.length === 'number' && Number.isFinite(params.length) ? params.length : null;
    const explicitLen =
      opts?.explicitLength !== undefined
        ? opts.explicitLength
        : curLen !== null && curLen !== getAdapterDefaultLength('rfd4-proteina')
          ? curLen
          : null;
    const targetLen = explicitLen ?? parsedBenchLen;

    const curContig = String(params.contig ?? '').trim();
    const contigDisjoint = curContig && isResidueSpecDisjointFromBenchmark(curContig, bench);
    if (bp.contig && (!curContig || contigDisjoint || opts?.overwriteUnpromptedResidues)) {
      if (contigDisjoint) repairedHallucination = true;
      params.contig =
        explicitLen !== null
          ? adaptBenchmarkContigForLength(bp.contig, explicitLen, bench.category)
          : bp.contig;
    }
    if (targetLen !== null) {
      params.length = targetLen;
    }
    params.autoSeededBenchmarkId = bench.id;
    if (explicitLen === null && parsedBenchLen !== null) {
      params.autoSeededLength = parsedBenchLen;
    } else if (explicitLen !== null) {
      delete params.autoSeededLength;
    }
  } else if (gen.adapterId === 'genie3') {
    if (!bench.supportsGenie3) {
      return { repairedHallucination: false };
    }
    if (!params.taskMode || params.taskMode === 'unconditional') {
      params.taskMode = bench.category === 'binder' ? 'binder' : 'motif';
    }

    const curHotspots = String(params.hotspots ?? '').trim();
    if (bench.category === 'binder' && bp.hotspots) {
      const disjoint = curHotspots && isResidueSpecDisjointFromBenchmark(curHotspots, bench);
      if (!curHotspots || disjoint || (opts?.overwriteUnpromptedResidues && curHotspots !== bp.hotspots)) {
        if (curHotspots && curHotspots !== bp.hotspots) repairedHallucination = true;
        params.hotspots = bp.hotspots;
      }
    }

    const curMotif = String(params.motifSpan ?? '').trim();
    if (bp.motifSpan) {
      const disjoint = curMotif && isResidueSpecDisjointFromBenchmark(curMotif, bench);
      if (!curMotif || disjoint || (opts?.overwriteUnpromptedResidues && curMotif !== bp.motifSpan)) {
        if (curMotif && curMotif !== bp.motifSpan) repairedHallucination = true;
        params.motifSpan = bp.motifSpan;
      }
    } else if (curMotif && (isResidueSpecDisjointFromBenchmark(curMotif, bench) || opts?.overwriteUnpromptedResidues)) {
      repairedHallucination = true;
      delete params.motifSpan;
    }

    if (bp.lockChain && !String(params.lockChain ?? '').trim()) {
      params.lockChain = bp.lockChain;
    }

    const parsedBenchLen = parseLengthFromContig(bp.contig);
    const curLen = typeof params.length === 'number' && Number.isFinite(params.length) ? params.length : null;
    const explicitLen =
      opts?.explicitLength !== undefined
        ? opts.explicitLength
        : curLen !== null && curLen !== getAdapterDefaultLength('genie3')
          ? curLen
          : null;
    const targetLen = explicitLen ?? parsedBenchLen;

    const curContig = String(params.contig ?? '').trim();
    const contigDisjoint = curContig && isResidueSpecDisjointFromBenchmark(curContig, bench);
    if (bp.contig && (!curContig || contigDisjoint || opts?.overwriteUnpromptedResidues)) {
      if (contigDisjoint) repairedHallucination = true;
      params.contig =
        explicitLen !== null
          ? adaptBenchmarkContigForLength(bp.contig, explicitLen, bench.category)
          : bp.contig;
    }
    if (targetLen !== null) {
      params.length = targetLen;
    }
    params.autoSeededBenchmarkId = bench.id;
    if (explicitLen === null && parsedBenchLen !== null) {
      params.autoSeededLength = parsedBenchLen;
    } else if (explicitLen !== null) {
      delete params.autoSeededLength;
    }
  } else if (gen.adapterId === 'protpardelle') {
    const curMode = String(params.taskMode ?? 'unconditional').trim() || 'unconditional';
    if (curMode === 'partial') return { repairedHallucination: false };
    if (!isBenchmarkSupportedForAdapter('protpardelle', bench)) {
      return { repairedHallucination: false };
    }
    if (!params.taskMode || params.taskMode === 'unconditional') {
      params.taskMode = 'motif';
    }
    const curSel = String(params.motifSelection ?? '').trim();
    if (bp.motifSpan) {
      const disjoint = curSel && isResidueSpecDisjointFromBenchmark(curSel, bench);
      if (!curSel || disjoint || (opts?.overwriteUnpromptedResidues && curSel !== bp.motifSpan)) {
        if (curSel && curSel !== bp.motifSpan) repairedHallucination = true;
        params.motifSelection = bp.motifSpan;
      }
    }
    const parsedBenchLen = parseLengthFromContig(bp.contig);
    const curLen = typeof params.length === 'number' && Number.isFinite(params.length) ? params.length : null;
    const explicitLen =
      opts?.explicitLength !== undefined
        ? opts.explicitLength
        : curLen !== null && curLen !== getAdapterDefaultLength('protpardelle')
          ? curLen
          : null;
    const targetLen = explicitLen ?? parsedBenchLen;
    if (targetLen !== null) params.length = Math.max(20, Math.min(256, targetLen));
    params.autoSeededBenchmarkId = bench.id;
    if (explicitLen === null && parsedBenchLen !== null) {
      params.autoSeededLength = Math.max(20, Math.min(256, parsedBenchLen));
    } else if (explicitLen !== null) {
      delete params.autoSeededLength;
    }
  } else if (gen.adapterId === 'boltzgen') {
    if (!isBenchmarkSupportedForAdapter('boltzgen', bench)) {
      return { repairedHallucination: false };
    }
    if (!params.taskMode || params.taskMode === 'unconditional') {
      params.taskMode = bench.category === 'binder' ? 'binder' : 'motif';
    }
    const curHotspots = String(params.hotspots ?? '').trim();
    if (bench.category === 'binder' && bp.hotspots) {
      const disjoint = curHotspots && isResidueSpecDisjointFromBenchmark(curHotspots, bench);
      if (!curHotspots || disjoint || (opts?.overwriteUnpromptedResidues && curHotspots !== bp.hotspots)) {
        if (curHotspots && curHotspots !== bp.hotspots) repairedHallucination = true;
        params.hotspots = bp.hotspots;
      }
    }
    const parsedBenchLen = parseLengthFromContig(bp.contig);
    const curLen = typeof params.length === 'number' && Number.isFinite(params.length) ? params.length : null;
    const explicitLen =
      opts?.explicitLength !== undefined
        ? opts.explicitLength
        : curLen !== null && curLen !== getAdapterDefaultLength('boltzgen')
          ? curLen
          : null;
    const targetLen = explicitLen ?? parsedBenchLen;
    if (targetLen !== null) params.length = Math.max(16, Math.min(160, targetLen));
    params.autoSeededBenchmarkId = bench.id;
    if (explicitLen === null && parsedBenchLen !== null) {
      params.autoSeededLength = Math.max(16, Math.min(160, parsedBenchLen));
    } else if (explicitLen !== null) {
      delete params.autoSeededLength;
    }
  } else if (gen.adapterId === 'rfdiffusion') {
    const curMode = String(params.taskMode ?? 'unconditional').trim() || 'unconditional';
    if (curMode === 'symmetry' || !isBenchmarkSupportedForAdapter('rfdiffusion', bench)) {
      return { repairedHallucination: false };
    }
    if (!params.taskMode || params.taskMode === 'unconditional') {
      params.taskMode = 'motif';
    }
    const parsedBenchLen = parseLengthFromContig(bp.contig);
    const curLen = typeof params.length === 'number' && Number.isFinite(params.length) ? params.length : null;
    const explicitLen =
      opts?.explicitLength !== undefined
        ? opts.explicitLength
        : curLen !== null && curLen !== getAdapterDefaultLength('rfdiffusion')
          ? curLen
          : null;
    const targetLen = explicitLen ?? parsedBenchLen;
    const curContig = String(params.contig ?? '').trim();
    const contigDisjoint = curContig && isResidueSpecDisjointFromBenchmark(curContig, bench);
    if (bp.contig && (!curContig || contigDisjoint || opts?.overwriteUnpromptedResidues)) {
      if (contigDisjoint) repairedHallucination = true;
      const adapted =
        explicitLen !== null
          ? adaptBenchmarkContigForLength(bp.contig, explicitLen, bench.category)
          : bp.contig;
      params.contig = formatRfdiffusionContig(adapted);
    }
    if (targetLen !== null) params.length = targetLen;
    params.autoSeededBenchmarkId = bench.id;
    if (explicitLen === null && parsedBenchLen !== null) {
      params.autoSeededLength = parsedBenchLen;
    } else if (explicitLen !== null) {
      delete params.autoSeededLength;
    }
  } else if (gen.adapterId === 'genie2') {
    if (!isBenchmarkSupportedForAdapter('genie2', bench)) {
      return { repairedHallucination: false };
    }
    const fullMotifRange = String(bp.motifSpan ?? '').trim();
    const curRange = String(params.motifRange ?? '').trim();
    if (fullMotifRange) {
      const disjoint = curRange && isResidueSpecDisjointFromBenchmark(curRange, bench);
      if (!curRange || disjoint || (opts?.overwriteUnpromptedResidues && curRange !== fullMotifRange)) {
        if (curRange && curRange !== fullMotifRange) repairedHallucination = true;
        params.motifRange = fullMotifRange;
      }
    }
    const parsedBenchLen = parseLengthFromContig(bp.contig);
    const curLen = typeof params.length === 'number' && Number.isFinite(params.length) ? params.length : null;
    const explicitLen =
      opts?.explicitLength !== undefined
        ? opts.explicitLength
        : curLen !== null && curLen !== getAdapterDefaultLength('genie2')
          ? curLen
          : null;
    const targetLen = explicitLen ?? parsedBenchLen;
    if (targetLen !== null) params.length = targetLen;
    params.autoSeededBenchmarkId = bench.id;
    if (explicitLen === null && parsedBenchLen !== null) {
      params.autoSeededLength = parsedBenchLen;
    } else if (explicitLen !== null) {
      delete params.autoSeededLength;
    }
  }

  return { repairedHallucination };
}
