// Pure TypeScript Workbench Controller Helpers
// Keeps deterministic graph-sync, preview signature, and viewport label helpers
// testable and decoupled from Svelte component state in +page.svelte.

import {
  BENCHMARK_SYNC_ADAPTER_IDS,
  extractFirstContiguousMotifRange,
  formatRfdiffusionContig,
  parseLengthFromContig,
} from '$lib/engine/benchmark-sync';
import { computeUploadDigest } from '$lib/engine/param-validation';
import { isValidPdbId, normalizePdbId } from '$lib/protein-ops/rcsb';
import type { WorkflowGraph, WorkflowNode } from '$lib/engine/types';

export function buildNodePreviewSignature(
  node: Pick<WorkflowNode, 'adapterId' | 'params'>,
): string | null {
  const p = node.params ?? {};
  if (node.adapterId === 'pdb-input') {
    const mode = String(p.mode ?? 'demo-bundle');
    const crop = String(p.cropSegments ?? '').trim();
    if (mode === 'pdb-id') {
      const normId = normalizePdbId(String(p.pdbId ?? ''));
      if (!isValidPdbId(normId)) return null;
      const cachedFor = normalizePdbId(String(p.fetchedPdbId ?? ''));
      const txt = String(p.pdbText ?? '');
      const hasCached = txt.trim().length > 0 && cachedFor === normId;
      return `pdb-id|${normId}|${hasCached ? computeUploadDigest(txt).slice(0, 16) : 0}|${String(p.benchmarkId ?? '')}|${crop}`;
    }
    if (mode === 'custom') {
      const txt = String(p.pdbText ?? '').trim();
      if (txt.length === 0) return null;
      return `custom|${txt.length}|${computeUploadDigest(txt).slice(0, 16)}|${String(p.filename ?? '')}|${String(p.benchmarkId ?? '')}|${crop}`;
    }
    return `demo-bundle|${String(p.benchmarkId ?? '')}|${crop}`;
  }
  if (node.adapterId === 'sequence-input') {
    const raw = String(p.sequence ?? '').trim();
    if (raw.length < 5) return null;
    return `seq|${raw.length}|${computeUploadDigest(raw).slice(0, 16)}|${String(p.label ?? '')}`;
  }
  return null;
}

export function getSiblingGeneratorIds(graph: WorkflowGraph, nodeId: string): string[] {
  const structEdge = graph.edges.find(
    (e) => e.toNode === nodeId && (e.toPort === 'structure' || e.toPort === 'template'),
  );
  if (!structEdge) return [];
  return graph.nodes
    .filter(
      (n) =>
        n.id !== nodeId &&
        BENCHMARK_SYNC_ADAPTER_IDS.has(n.adapterId) &&
        graph.edges.some(
          (e) =>
            e.fromNode === structEdge.fromNode &&
            e.toNode === n.id &&
            (e.toPort === 'structure' || e.toPort === 'template'),
        ),
    )
    .map((n) => n.id);
}

export interface SiblingTaskSyncResult {
  nextGraph: WorkflowGraph;
  siblingIds: Set<string>;
  unsupportedSiblingLabels: string[];
  srcMode: string;
  syncLen: number;
}

export function syncTaskToSiblingGenerators(
  graph: WorkflowGraph,
  sourceNodeId: string,
): SiblingTaskSyncResult | null {
  const src = graph.nodes.find((n) => n.id === sourceNodeId);
  if (!src) return null;
  const siblingIds = new Set(getSiblingGeneratorIds(graph, sourceNodeId));
  if (siblingIds.size === 0) return null;

  const sp = src.params ?? {};
  const srcMode =
    src.adapterId === 'rfd4-proteina'
      ? String(sp.mode ?? 'unconditional')
      : src.adapterId === 'genie2'
        ? String(sp.motifRange ?? '').trim()
          ? 'motif'
          : 'unconditional'
        : String(sp.taskMode ?? 'unconditional');
  const parsedLen = parseLengthFromContig(String(sp.contig ?? ''));
  const defaultSrcLen =
    src.adapterId === 'boltzgen' ? 48 : src.adapterId === 'rfd4-proteina' ? 70 : 80;
  const rawLen =
    typeof sp.length === 'number' || (typeof sp.length === 'string' && sp.length.trim() !== '')
      ? Number(sp.length)
      : NaN;
  const syncLen =
    Number.isFinite(rawLen) && rawLen > 0
      ? rawLen
      : parsedLen !== null && Number.isFinite(parsedLen) && parsedLen > 0
        ? parsedLen
        : defaultSrcLen;
  const srcMotif = String(sp.motifSpan ?? sp.motifSelection ?? sp.motifRange ?? '').trim();
  const srcHotspots = String(sp.hotspots ?? '').trim();
  const srcContig = String(sp.contig ?? '').trim();
  const srcLockChain = String(sp.lockChain ?? '').trim();
  const unsupportedSiblingLabels: string[] = [];

  const nextGraph: WorkflowGraph = {
    ...graph,
    nodes: graph.nodes.map((n) => {
      if (!siblingIds.has(n.id)) return n;
      const shared = {
        length: syncLen,
        contig: srcContig,
        lockChain: srcLockChain,
        hotspots: srcHotspots,
        motifSpan: srcMotif,
        userEditedTask: true,
      };
      if (n.adapterId === 'genie3') {
        const clampedLen = Math.max(16, Math.min(512, Math.round(syncLen)));
        const isSupported =
          srcMode === 'binder' || srcMode === 'motif' || srcMode === 'unconditional';
        if (!isSupported) {
          unsupportedSiblingLabels.push(n.label ?? 'Genie 3');
          return {
            ...n,
            params: {
              ...n.params,
              taskMode: 'unconditional',
              length: clampedLen,
              contig: '',
              lockChain: '',
              hotspots: '',
              motifSpan: '',
              userEditedTask: true,
            },
          };
        }
        if (srcMode === 'unconditional') {
          return {
            ...n,
            params: {
              ...n.params,
              taskMode: 'unconditional',
              length: clampedLen,
              contig: '',
              lockChain: '',
              hotspots: '',
              motifSpan: '',
              userEditedTask: true,
            },
          };
        }
        return { ...n, params: { ...n.params, ...shared, length: clampedLen, taskMode: srcMode } };
      }
      if (n.adapterId === 'boltzgen') {
        const clampedLen = Math.max(16, Math.min(160, Math.round(syncLen)));
        const isSupported =
          srcMode === 'binder' || srcMode === 'motif' || srcMode === 'unconditional';
        if (!isSupported) {
          unsupportedSiblingLabels.push(n.label ?? 'BoltzGen-1');
          return {
            ...n,
            params: {
              ...n.params,
              taskMode: 'unconditional',
              length: clampedLen,
              hotspots: '',
              userEditedTask: true,
            },
          };
        }
        return {
          ...n,
          params: {
            ...n.params,
            taskMode: srcMode,
            length: clampedLen,
            hotspots: srcMode === 'binder' ? srcHotspots : '',
            userEditedTask: true,
          },
        };
      }
      if (n.adapterId === 'protpardelle') {
        const clampedLen = Math.max(20, Math.min(256, Math.round(syncLen)));
        const isSupported = srcMode === 'motif' || srcMode === 'unconditional';
        if (!isSupported) {
          unsupportedSiblingLabels.push(n.label ?? 'ProtPardelle-1c');
          return {
            ...n,
            params: {
              ...n.params,
              taskMode: 'unconditional',
              length: clampedLen,
              motifSelection: '',
              userEditedTask: true,
            },
          };
        }
        return {
          ...n,
          params: {
            ...n.params,
            taskMode: srcMode,
            length: clampedLen,
            motifSelection: srcMode === 'motif' ? srcMotif : '',
            userEditedTask: true,
          },
        };
      }
      if (n.adapterId === 'rfdiffusion') {
        const clampedLen = Math.max(16, Math.min(256, Math.round(syncLen)));
        const isSupported = srcMode === 'motif' || srcMode === 'unconditional';
        if (!isSupported) {
          unsupportedSiblingLabels.push(n.label ?? 'RFdiffusion');
          return {
            ...n,
            params: {
              ...n.params,
              taskMode: 'unconditional',
              length: clampedLen,
              contig: '',
              userEditedTask: true,
            },
          };
        }
        return {
          ...n,
          params: {
            ...n.params,
            taskMode: srcMode,
            length: clampedLen,
            contig: srcMode === 'motif' ? formatRfdiffusionContig(srcContig) : '',
            userEditedTask: true,
          },
        };
      }
      if (n.adapterId === 'genie2') {
        const clampedLen = Math.max(20, Math.min(256, Math.round(syncLen)));
        const isSupported = srcMode === 'motif' || srcMode === 'unconditional';
        if (!isSupported) {
          unsupportedSiblingLabels.push(n.label ?? 'Genie 2');
          return {
            ...n,
            params: { ...n.params, length: clampedLen, motifRange: '', userEditedTask: true },
          };
        }
        return {
          ...n,
          params: {
            ...n.params,
            length: clampedLen,
            motifRange: srcMode === 'motif' ? extractFirstContiguousMotifRange(srcMotif) : '',
            userEditedTask: true,
          },
        };
      }
      return {
        ...n,
        params: {
          ...n.params,
          ...shared,
          mode: srcMode,
          ...(src.adapterId === 'rfd4-proteina'
            ? {
                presetId: String(sp.presetId ?? n.params.presetId ?? 'custom_cath_monomer'),
                partialFraction: Number(sp.partialFraction ?? n.params.partialFraction ?? 0.5),
                unindexedMotif: Boolean(sp.unindexedMotif),
                hbondMode: String(sp.hbondMode ?? 'none'),
                hbondSelection: String(sp.hbondSelection ?? ''),
                cathCode: String(sp.cathCode ?? ''),
              }
            : {}),
        },
      };
    }),
  };

  return {
    nextGraph,
    siblingIds,
    unsupportedSiblingLabels,
    srcMode,
    syncLen,
  };
}

export function formatGeomStripLabel(
  viewportRole: string | undefined,
  sourceModelId: string | undefined,
  hasPeekMetrics: boolean,
): string {
  if (viewportRole === 'receptor') return 'Receptor:';
  if (viewportRole === 'input') return 'Input:';
  if (viewportRole === 'predicted') return 'Predicted:';
  return sourceModelId === 'pdb-input' && !hasPeekMetrics ? 'Input:' : 'Designed:';
}
