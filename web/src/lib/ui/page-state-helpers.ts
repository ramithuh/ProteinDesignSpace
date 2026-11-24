import { validateBioCapabilities, type BioFoldLimitOptions } from '$lib/engine/bio-capabilities';
import { validateGraph } from '$lib/engine/graph-utils';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  BioCompatibilityIssue,
  CanonicalArtifact,
  MutationMatrixArtifact,
  ResidueTrackArtifact,
  StructureArtifact,
  WorkflowGraph,
  WorkflowNode,
} from '$lib/engine/types';
import type { PlannedWorkflowResult } from '$lib/engine/workflow-planner';

export {
  UI_TRACKING_PARAM_HINTS,
  UI_TRACKING_PARAM_KEYS,
  buildBenchmarkPdbParams,
  computeBenchmarkAutoSeedPatches,
  detectQuickPdbBenchmarkMatch,
  getBenchmarksForPdbId,
  parseLengthFromContig,
  reconcileBenchmarkSeeding,
  switchBenchmark,
  type BenchmarkReconcileResult,
} from '$lib/engine/benchmark-sync';

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
  foldLimitOpts?: BioFoldLimitOptions,
): PreviewChecksResult {
  if (!activeDraftPreview) {
    const inputErrs = Object.values(inputPreviewErrorsByNode);
    return {
      displayedValidationErrors: [...validateGraph(committedGraph), ...inputErrs],
      bioIssues: validateBioCapabilities(committedGraph, outputsByNode, foldLimitOpts),
      hideBioFixes: false,
      hideRunError: false,
    };
  }

  const previewGraph = activeDraftPreview.graph;
  if (!activeDraftPreview.isIncrementalEdit) {
    return {
      displayedValidationErrors: validateGraph(previewGraph),
      bioIssues: validateBioCapabilities(previewGraph, {}, foldLimitOpts),
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
    bioIssues: validateBioCapabilities(previewGraph, filteredOutputs, foldLimitOpts),
    hideBioFixes: true,
    hideRunError: false,
  };
}

export function isSharedWorkflowHash(rawHash: string): boolean {
  const trimmed = rawHash.trim();
  return (
    trimmed.startsWith('#g=') ||
    trimmed.startsWith('#p=') ||
    trimmed.startsWith('?g=') ||
    trimmed.startsWith('?p=')
  );
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

const COMPARE_DISPLAY_ONLY_PARAM_KEYS = new Set(['layout', 'colorMode', 'alignKabsch']);

/**
 * Return `true` when editing `structure-compare` only changes display-only parameters
 * (`layout`, `colorMode`, `alignKabsch`), which must not invalidate cached outputs.
 */
export function isCompareDisplayOnlyParamChange(
  prevNode: Pick<WorkflowNode, 'adapterId' | 'params'> | undefined,
  nextParams: Record<string, unknown>,
): boolean {
  if (!prevNode || prevNode.adapterId !== 'structure-compare') return false;
  const changedKeys = computeChangedParamKeys(prevNode.params, nextParams);
  return (
    changedKeys.length > 0 &&
    changedKeys.every((k) => COMPARE_DISPLAY_ONLY_PARAM_KEYS.has(k))
  );
}

/**
 * Map a clicked per-residue position/ID from a `MutationMatrix` or `ResidueTrack`
 * to a Mol*-resolvable residue focus token (handling non-1-based PDBs, multi-chain complexes,
 * insertion codes, and multi-char / lowercase mmCIF chain IDs).
 */
export function resolveResidueFocusToken(
  positionIndex: number,
  residueId: string,
  artifact: MutationMatrixArtifact | ResidueTrackArtifact | null,
  structure: StructureArtifact | null,
): string | null {
  let sourceResidues = structure?.sourceResidues ?? [];
  if (sourceResidues.length === 0 && structure && (structure.cifText || structure.pdbText)) {
    try {
      sourceResidues = parseStructureText(structure.cifText || structure.pdbText).sourceResidues ?? [];
    } catch {
      sourceResidues = [];
    }
  }

  const useAuthChain = Boolean(structure?.cifText);
  const formatSrToken = (sr: (typeof sourceResidues)[number]): string => {
    const chain = (useAuthChain ? sr.authChainId : sr.projectedChainId) || sr.authChainId || 'A';
    return chain.length > 1 || /[a-z]/.test(chain)
      ? `${chain}:${sr.resNum}${sr.insCode}`
      : `${chain}${sr.resNum}${sr.insCode}`;
  };

  const artifactLength =
    artifact?.kind === 'MutationMatrix'
      ? artifact.residueIds && artifact.residueIds.length > 0
        ? artifact.residueIds.length
        : (artifact.residues?.length ?? 0)
      : artifact?.kind === 'ResidueTrack'
        ? artifact.residueIds && artifact.residueIds.length > 0
          ? artifact.residueIds.length
          : (artifact.values?.length ?? artifact.residues?.length ?? 0)
        : 0;

  if (
    artifact?.sourceModelId === 'esmc' &&
    sourceResidues.length > 0 &&
    sourceResidues.length === artifactLength &&
    positionIndex >= 0 &&
    positionIndex < sourceResidues.length
  ) {
    return formatSrToken(sourceResidues[positionIndex]);
  }

  const raw = (residueId ?? '').trim();
  if (raw && sourceResidues.length > 0) {
    for (const sr of sourceResidues) {
      const authCompact = `${sr.authChainId}${sr.resNum}${sr.insCode}`;
      const authColon = `${sr.authChainId}:${sr.resNum}${sr.insCode}`;
      const projCompact = `${sr.projectedChainId}${sr.resNum}${sr.insCode}`;
      const projColon = `${sr.projectedChainId}:${sr.resNum}${sr.insCode}`;
      if (
        raw === authCompact ||
        raw === authColon ||
        raw === projCompact ||
        raw === projColon
      ) {
        return formatSrToken(sr);
      }
    }
    const rawLower = raw.toLowerCase();
    for (const sr of sourceResidues) {
      const authCompact = `${sr.authChainId}${sr.resNum}${sr.insCode}`.toLowerCase();
      const authColon = `${sr.authChainId}:${sr.resNum}${sr.insCode}`.toLowerCase();
      const projCompact = `${sr.projectedChainId}${sr.resNum}${sr.insCode}`.toLowerCase();
      const projColon = `${sr.projectedChainId}:${sr.resNum}${sr.insCode}`.toLowerCase();
      if (
        rawLower === authCompact ||
        rawLower === authColon ||
        rawLower === projCompact ||
        rawLower === projColon
      ) {
        return formatSrToken(sr);
      }
    }
  }

  if (
    sourceResidues.length > 0 &&
    sourceResidues.length === artifactLength &&
    positionIndex >= 0 &&
    positionIndex < sourceResidues.length
  ) {
    return formatSrToken(sourceResidues[positionIndex]);
  }

  if (raw) return raw;
  return positionIndex >= 0 ? `A${positionIndex + 1}` : null;
}
