import { adaptBenchmarkContigForLength, getBenchmarkById } from '$lib/engine/benchmarks';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { validateGraph } from '$lib/engine/graph-utils';
import type {
  BioCompatibilityIssue,
  CanonicalArtifact,
  WorkflowGraph,
} from '$lib/engine/types';
import type { PlannedWorkflowResult } from '$lib/engine/workflow-planner';

export interface PreviewChecksResult {
  displayedValidationErrors: string[];
  bioIssues: BioCompatibilityIssue[];
  hideBioFixes: boolean;
  hideRunError: boolean;
}

/**
 * Evaluate Workflow Check validation errors and Bio Compatibility issues against either
 * the active Draft preview graph (R5) or the committed graph, preventing stale banners
 * from leaking across unrelated workflow previews.
 */
export function computePreviewChecks(
  committedGraph: WorkflowGraph,
  activeDraftPreview: PlannedWorkflowResult | null,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
  inputPreviewErrorsByNode: Record<string, string>,
): PreviewChecksResult {
  if (!activeDraftPreview) {
    const inputErrs = Object.values(inputPreviewErrorsByNode);
    return {
      displayedValidationErrors: [...validateGraph(committedGraph), ...inputErrs],
      bioIssues: validateBioCapabilities(committedGraph, outputsByNode),
      hideBioFixes: false,
      hideRunError: false,
    };
  }

  const previewGraph = activeDraftPreview.graph;
  if (!activeDraftPreview.isIncrementalEdit) {
    return {
      displayedValidationErrors: validateGraph(previewGraph),
      bioIssues: validateBioCapabilities(previewGraph, {}),
      hideBioFixes: true,
      hideRunError: true,
    };
  }

  const committedAdapterById = new Map(committedGraph.nodes.map((n) => [n.id, n.adapterId]));
  const filteredOutputs: Record<string, Record<string, CanonicalArtifact>> = {};
  const filteredInputErrs: string[] = [];

  for (const node of previewGraph.nodes) {
    const prevAdapterId = committedAdapterById.get(node.id);
    if (prevAdapterId !== node.adapterId) continue;
    const nodeOut = outputsByNode[node.id];
    if (nodeOut) {
      filteredOutputs[node.id] = nodeOut;
    }
    const err = inputPreviewErrorsByNode[node.id];
    if (err) {
      filteredInputErrs.push(err);
    }
  }

  return {
    displayedValidationErrors: [...validateGraph(previewGraph), ...filteredInputErrs],
    bioIssues: validateBioCapabilities(previewGraph, filteredOutputs),
    hideBioFixes: true,
    hideRunError: false,
  };
}

/**
 * Parse the designed/target residue length from a benchmark or generator contig string.
 * Supports binder `/0` suffixes (`A17-132,/0,70-70` -> 70), bare ranges (`150-150` -> 150),
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
 * Compute benchmark auto-seed patches for unconfigured or previously auto-seeded
 * `rfd4-proteina` / `genie3` blocks wired to a benchmark `pdb-input`, adapting
 * the benchmark contig whenever a custom/planned `length` is already present.
 */
export function computeBenchmarkAutoSeedPatches(
  graph: WorkflowGraph,
): Record<string, Record<string, unknown>> | null {
  let patches: Record<string, Record<string, unknown>> | null = null;
  for (const node of graph.nodes) {
    if (node.adapterId !== 'genie3' && node.adapterId !== 'rfd4-proteina') {
      continue;
    }
    const structEdge = graph.edges.find(
      (e) => e.toNode === node.id && (e.toPort === 'structure' || e.toPort === 'template'),
    );
    if (!structEdge) continue;
    const srcNode = graph.nodes.find((n) => n.id === structEdge.fromNode);
    if (!srcNode || srcNode.adapterId !== 'pdb-input') continue;
    const benchId = String(srcNode.params.benchmarkId ?? '').trim();
    if (!benchId) continue;
    const bench = getBenchmarkById(benchId);
    if (!bench) continue;

    const cp = node.params ?? {};
    if (Boolean(cp.userEditedTask)) continue;
    const seededId = String(cp.autoSeededBenchmarkId ?? '').trim();
    if (seededId === bench.id) continue;

    const adapterDefaultLen = node.adapterId === 'genie3' ? 80 : 70;
    const prevBench = seededId ? getBenchmarkById(seededId) : undefined;
    const prevBenchValidForAdapter =
      Boolean(prevBench) && (node.adapterId !== 'genie3' || prevBench!.supportsGenie3);
    const prevBenchLen = prevBenchValidForAdapter
      ? (parseLengthFromContig(prevBench!.conditionParams.contig) ?? prevBench!.approxTokens)
      : null;
    const prevAutoLen =
      typeof cp.autoSeededLength === 'number' &&
      Number.isFinite(cp.autoSeededLength) &&
      cp.autoSeededLength > 0
        ? cp.autoSeededLength
        : prevBenchLen;
    const curLen = Number(cp.length);
    const hasCustomLength =
      Boolean(cp.userEditedLength) ||
      (Number.isFinite(curLen) &&
        curLen > 0 &&
        (seededId
          ? curLen !== prevAutoLen && curLen !== adapterDefaultLen
          : curLen !== adapterDefaultLen));

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

export function isSharedWorkflowHash(rawHash: string): boolean {
  const trimmed = rawHash.trim();
  return trimmed.startsWith('#g=') || trimmed.startsWith('#p=');
}

export function shouldSuppressOnboardingForHash(
  loadedFromHash: boolean,
  attemptedSharedHash: boolean,
): boolean {
  return loadedFromHash || attemptedSharedHash;
}

export interface ElementLikeForUndoGuard {
  tagName?: string;
  type?: string;
  disabled?: boolean;
  readOnly?: boolean;
  isContentEditable?: boolean;
  classList?: { contains(cls: string): boolean };
  className?: string;
}

const NON_TEXT_INPUT_TYPES = new Set([
  'checkbox',
  'radio',
  'range',
  'file',
  'button',
  'submit',
  'reset',
  'color',
]);

/**
 * Return `true` only when native browser text undo (`Cmd/Ctrl+Z`) should stay active
 * (textareas, contenteditable regions, and standard text inputs—NOT `<select>`,
 * checkboxes, sliders, file inputs, or `.num-input` NumericInput controls).
 */
export function shouldAllowNativeTextUndo(el: unknown): boolean {
  if (!el || typeof el !== 'object') return false;
  if (typeof HTMLElement !== 'undefined' && !(el instanceof HTMLElement)) {
    return false;
  }
  const elem = el as ElementLikeForUndoGuard;
  const tag = String(elem.tagName ?? '').toUpperCase();
  if (!tag) return false;
  if (elem.disabled || elem.readOnly) return false;
  if (elem.isContentEditable) return true;
  if (tag === 'TEXTAREA') return true;
  if (tag === 'SELECT') return false;
  if (tag !== 'INPUT') return false;

  const inputType = String(elem.type ?? 'text').toLowerCase();
  if (NON_TEXT_INPUT_TYPES.has(inputType)) return false;
  const hasNumClass =
    Boolean(elem.classList?.contains('num-input')) ||
    (typeof elem.className === 'string' && /\bnum-input\b/.test(elem.className));
  if (hasNumClass) return false;
  return true;
}

export function shouldSyncNumericDraft(focused: boolean, isSelfEcho: boolean): boolean {
  return !focused || !isSelfEcho;
}

export function computeChangedParamKeys(
  prevParams: Record<string, unknown> | undefined,
  nextParams: Record<string, unknown>,
): string[] {
  const prev = prevParams ?? {};
  const keys = new Set([...Object.keys(prev), ...Object.keys(nextParams)]);
  const changed: string[] = [];
  for (const k of keys) {
    if (k === 'userEditedTask' || k === 'userEditedLength') continue;
    if (!Object.is(prev[k], nextParams[k])) {
      changed.push(k);
    }
  }
  changed.sort();
  return changed;
}

export interface ParamUndoMarker {
  nodeId: string;
  paramKey: string;
  at: number;
}

export interface LabelUndoMarker {
  nodeId: string;
  at: number;
}

export const GRAPH_PARAM_UNDO_COALESCE_MS = 800;
export const GRAPH_LABEL_PARAM_ATOMIC_MS = 25;

export function classifyParamUndoAction(
  nodeId: string,
  paramSignature: string,
  now: number,
  lastParamUndo: ParamUndoMarker | null,
  lastLabelUndo: LabelUndoMarker | null,
): 'merge-with-label' | 'coalesce' | 'push' {
  if (
    lastLabelUndo &&
    lastLabelUndo.nodeId === nodeId &&
    now - lastLabelUndo.at < GRAPH_LABEL_PARAM_ATOMIC_MS
  ) {
    return 'merge-with-label';
  }
  if (
    lastParamUndo &&
    lastParamUndo.nodeId === nodeId &&
    lastParamUndo.paramKey === paramSignature &&
    now - lastParamUndo.at < GRAPH_PARAM_UNDO_COALESCE_MS
  ) {
    return 'coalesce';
  }
  return 'push';
}
