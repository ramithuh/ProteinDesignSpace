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
      const m = tok.match(/^(\d+)(?:-(\d+))?$/);
      if (m) {
        const lo = Number(m[1]);
        const hi = m[2] ? Number(m[2]) : lo;
        return Math.round((lo + hi) / 2);
      }
    }
  }
  const bare = raw.match(/^(\d+)(?:-(\d+))?$/);
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
      const lenMatch = tok.match(/^(\d+)(?:-(\d+))?$/);
      if (lenMatch) {
        const lo = Number(lenMatch[1]);
        const hi = lenMatch[2] ? Number(lenMatch[2]) : lo;
        total += Math.round((lo + hi) / 2);
        matchedAny = true;
        continue;
      }
      const spanMatch = tok.match(/^[A-Za-z](\d+)-(\d+)$/);
      if (spanMatch) {
        const lo = Number(spanMatch[1]);
        const hi = Number(spanMatch[2]);
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
    const span = /^([A-Za-z_])(-?\d+)(?:-(-?\d+))?$/.exec(tok);
    if (span) {
      const chain = span[1].toUpperCase();
      const a = Number(span[2]);
      const b = span[3] !== undefined ? Number(span[3]) : a;
      if (Number.isFinite(a) && Number.isFinite(b)) {
        out.push({ chain, start: Math.min(a, b), end: Math.max(a, b) });
      }
      continue;
    }
    if (/^[A-Za-z_]$/.test(tok)) {
      out.push({ chain: tok.toUpperCase(), start: -Infinity, end: Infinity });
    }
  }
  return out;
}

/**
 * Return `true` when `spec` specifies at least one residue interval and NONE of its
 * residue intervals overlap any interval in `refIntervals`.
 */
export function isResidueSpecDisjointFromIntervals(
  spec: string | undefined,
  refIntervals: readonly ParsedResidueInterval[],
): boolean {
  const specIntervals = parseResidueIntervals(spec);
  if (specIntervals.length === 0 || refIntervals.length === 0) return false;
  for (const s of specIntervals) {
    for (const r of refIntervals) {
      if (s.chain === r.chain && s.start <= r.end && r.start <= s.end) {
        return false;
      }
    }
  }
  return true;
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
  return adapterId === 'genie3' ? 80 : 70;
}

function resolvePrevAutoLength(
  adapterId: string,
  cp: Record<string, unknown>,
  seededId: string,
): number | null {
  const prevBench = seededId ? getBenchmarkById(seededId) : undefined;
  const prevBenchValidForAdapter =
    Boolean(prevBench) && (adapterId !== 'genie3' || prevBench!.supportsGenie3);
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
 * Compute benchmark auto-seed AND symmetric teardown patches for `rfd4-proteina` / `genie3`
 * nodes across a `WorkflowGraph`:
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
    if (node.adapterId !== 'genie3' && node.adapterId !== 'rfd4-proteina') {
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
          unindexedMotif: false,
          hbondMode: 'none',
          hbondSelection: '',
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
      }
      continue;
    }

    if (seededId === bench.id) continue;

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
        !String(cp.lockChain ?? '').trim() &&
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
        unindexedMotif: Boolean(bp.unindexedMotif),
        hbondMode: bp.hbondMode ?? 'none',
        hbondSelection: bp.hbondSelection ?? '',
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
    if (bp.unindexedMotif && params.unindexedMotif !== true) {
      params.unindexedMotif = true;
    }
    if (bp.hbondMode && (!params.hbondMode || params.hbondMode === 'none')) {
      params.hbondMode = bp.hbondMode;
    }
    if (bp.hbondSelection && !String(params.hbondSelection ?? '').trim()) {
      params.hbondSelection = bp.hbondSelection;
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
  }

  return { repairedHallucination };
}
