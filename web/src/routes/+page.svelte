<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { SvelteFlowProvider } from '@xyflow/svelte';
  import {
    ExecutionScheduler,
    RunCancellationError,
    attachProgrammaticApi,
    materializePdbInputNode,
    materializeSequenceInputNode,
    type ActiveRunState,
  } from '$lib/engine/scheduler';
  import { clearRcsbCache, isValidPdbId, normalizePdbId } from '$lib/protein-ops/rcsb';
  import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from '$lib/engine/registry';
  import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
  import { autoLayoutGraph, defaultParamsForAdapter, insertBlockSmart, validateGraph } from '$lib/engine/graph-utils';
  import {
    extractSubgraphUpTo,
    findContextStructure,
    getNodePeekSlots,
    makeEdgeId,
    pickDefaultPeekSlot,
  } from '$lib/engine/peek';
  import {
    decodeHashToGraph,
    encodeGraphToHash,
    parseRecipeToGraph,
    serializeWorkflowJson,
  } from '$lib/engine/share';
  import {
    clearAllModelCaches,
    createIndexedDbCheckpointStore,
    getCacheStats,
  } from '$lib/engine/opfs-cache';
  import {
    COMPARE_PALETTE,
    buildComparisonState,
    buildMultiDesignViewerModels,
    type ComparisonPairingMode,
  } from '$lib/engine/compare';
  import { estimateRunRequirements, type RunPreflightSummary } from '$lib/engine/run-preflight';
  import type { PlannedWorkflowResult } from '$lib/engine/workflow-planner';
  import type {
    AtomGranularity,
    BfactorSemantics,
    BioCompatibilityIssue,
    CanonicalArtifact,
    DesignabilityMetrics,
    DesignabilityRow,
    FoldedStructure,
    NodeProgressEvent,
    SequenceSet,
    StructureArtifact,
    WorkflowEdge,
    WorkflowGraph,
  } from '$lib/engine/types';
  import MolSuperposeViewer from '$lib/viewer/MolSuperposeViewer.svelte';
  import ArtifactPeekPanel from '$lib/ui/ArtifactPeekPanel.svelte';
  import ComparePeekPanel, { type CompareBranchMeta } from '$lib/ui/ComparePeekPanel.svelte';
  import CreditsModal from '$lib/ui/CreditsModal.svelte';
  import ExamplesGalleryModal from '$lib/ui/ExamplesGalleryModal.svelte';
  import PaperWalkthroughOverlay from '$lib/ui/PaperWalkthroughOverlay.svelte';
  import FirstRunModal from '$lib/ui/FirstRunModal.svelte';
  import { type PaperInsightExample } from '$lib/engine/paper-gallery';
  import ModelCardPopover from '$lib/ui/ModelCardPopover.svelte';
  import OnboardingModal from '$lib/ui/OnboardingModal.svelte';
  import NodeInspector from '$lib/ui/NodeInspector.svelte';
  import NodeStudioPopover from '$lib/ui/NodeStudioPopover.svelte';
  import type { StudioBrushOption } from '$lib/ui/AnnotatedFastaView.svelte';
  import { hasInvalidNumericInputs } from '$lib/ui/NumericInput.svelte';
  import WorkflowCanvas from '$lib/ui/WorkflowCanvas.svelte';
  import RecipeDrawer from '$lib/ui/RecipeDrawer.svelte';
  import {
    classifyParamUndoAction,
    computeChangedParamKeys,
    computePreviewChecks,
    isCompareDisplayOnlyParamChange,
    isSharedWorkflowHash,
    parseLengthFromContig,
    reconcileBenchmarkSeeding,
    resolveResidueFocusToken,
    shouldAllowNativeTextUndo,
    shouldSuppressOnboardingForHash,
    type LabelUndoMarker,
    type ParamUndoMarker,
  } from '$lib/ui/page-state-helpers';
  import {
    BENCHMARK_SYNC_ADAPTER_IDS,
    extractFirstContiguousMotifRange,
    formatRfdiffusionContig,
  } from '$lib/engine/benchmark-sync';
  import { computeUpstreamLockSummary } from '$lib/ui/node-display-helpers';
  import { compileNodeInputPreview } from '$lib/engine/input-preview';
  import {
    formatWebGpuAdapterSummary,
    requestResilientWebGpuAdapter,
    type WebGpuAdapterInfoSnapshot,
  } from '$lib/engine/webgpu-device';
  import {
    checkRemoteServerHealth,
    DEFAULT_HOST_MAX_FOLD_RESIDUES,
    defaultComputeBackendConfig,
    loadComputeBackendConfig,
    releaseRemoteServerGpu,
    resolveNodeFoldLimitOptions,
    runNodeOnRemoteServer,
    sanitizeMaxFoldResidues,
    saveComputeBackendConfig,
    shouldRouteNodeToRemote,
    type ComputeBackendConfig,
    type ComputeRoutingMode,
    type RemoteServerHealth,
  } from '$lib/engine/compute-backend';
  import type { BioFoldLimitOptions } from '$lib/engine/bio-capabilities';

  const scheduler = new ExecutionScheduler({
    checkpointStore: createIndexedDbCheckpointStore(),
  });

  let rightPanelTab = $state<'params' | 'peek'>('params');
  let expandedPane = $state<'none' | 'viewer' | 'inspector' | 'studio'>('none');
  let studioTab = $state<'input' | 'controls' | 'output'>('input');
  let studioHoveredToken = $state<string | null>(null);
  let studioFocusToken = $state<string | null>(null);
  let studioActiveBrush = $state<StudioBrushOption | null>(null);
  let viewerFocusToken = $state<string | null>(null);
  let selectedDesignKey = $state<string>('all');
  let multiDesignViewMode = $state<'grid' | 'overlay' | 'single'>('grid');
  let multiDesignHiddenIds = $state<string[]>([]);
  let splitMode = $state<'horizontal' | 'vertical'>('horizontal');
  let splitPct = $state<number>(65);
  let verticalSplitPct = $state<number>(48);
  let draggingSplit = $state(false);
  let splitWrapEl: HTMLDivElement | null = $state(null);
  let splitMoveRafId: number | null = null;
  let pendingSplitClientX: number | null = null;
  let pendingSplitClientY: number | null = null;
  let theme = $state<'dark' | 'light'>('dark');
  let webgpuSupported = $state<boolean | null>(null);
  let webgpuDetail = $state<string>('checking…');
  let webgpuIsFallback = $state<boolean>(false);
  let webgpuAdapterInfo = $state<WebGpuAdapterInfoSnapshot | null>(null);
  let webMcpReady = $state(false);
  let activeGpuWorker = $state<string | null>(null);
  let activeGpuWorkerLabel = $derived(
    activeGpuWorker ? (ADAPTER_REGISTRY[activeGpuWorker]?.shortName ?? activeGpuWorker) : null,
  );
  let cacheUsedMB = $state<number>(0);
  let computeBackend = $state<ComputeBackendConfig>(defaultComputeBackendConfig());
  let remoteHealth = $state<RemoteServerHealth | null>(null);
  let remoteChecking = $state(false);

  function buildFoldLimitOpts(
    cfg: ComputeBackendConfig = computeBackend,
    health: RemoteServerHealth | null = remoteHealth,
  ): BioFoldLimitOptions {
    return {
      resolveNodeFoldLimits: (node) =>
        resolveNodeFoldLimitOptions(node, cfg, health, {
          localWebGpuAvailable: webgpuSupported !== false,
          localWebGpuFallback: webgpuIsFallback,
        }),
    };
  }

  function syncSchedulerRemoteExecutor(cfg: ComputeBackendConfig) {
    scheduler.setFoldLimitResolver((node) =>
      resolveNodeFoldLimitOptions(node, cfg, remoteHealth, {
        localWebGpuAvailable: webgpuSupported !== false,
        localWebGpuFallback: webgpuIsFallback,
      }),
    );
    if (cfg.mode === 'local') {
      scheduler.setRemoteGpuExecutor(null);
      return;
    }
    scheduler.setRemoteGpuExecutor((node, inputs, onProgress, signal) => {
      const routeRemote = shouldRouteNodeToRemote(node, cfg, {
        localWebGpuAvailable: webgpuSupported !== false,
        localWebGpuFallback: webgpuIsFallback,
      });
      if (!routeRemote) return null;
      return runNodeOnRemoteServer({
        serverUrl: cfg.serverUrl,
        apiToken: cfg.apiToken,
        node,
        inputs,
        onProgress,
        signal,
      });
    });
  }

  async function probeRemoteBackend(cfg: ComputeBackendConfig = computeBackend) {
    remoteChecking = true;
    try {
      remoteHealth = await checkRemoteServerHealth(cfg.serverUrl, cfg.apiToken);
      syncSchedulerRemoteExecutor(computeBackend);
    } finally {
      remoteChecking = false;
    }
  }

  function updateComputeBackend(patch: Partial<ComputeBackendConfig>) {
    computeBackend = { ...computeBackend, ...patch };
    saveComputeBackendConfig(computeBackend);
    syncSchedulerRemoteExecutor(computeBackend);
    if (computeBackend.mode !== 'local') {
      void probeRemoteBackend(computeBackend);
    }
  }

  async function freeRemoteVram() {
    if (isRunning || activeRunState.leaseHeld) return;
    try {
      await releaseRemoteServerGpu(computeBackend.serverUrl, computeBackend.apiToken);
      await probeRemoteBackend(computeBackend);
    } catch {
      /* ignore */
    }
  }

  // Header popovers & first-run confirmation dialog
  let sysMenuOpen = $state(false);
  let sysPopAlignLeft = $state(false);
  let moreMenuOpen = $state(false);
  let shareMenuOpen = $state(false);
  let sharePopoverState = $state<{
    url: string;
    mode: string;
    strippedPdb: boolean;
    copied: boolean;
    error: string | null;
  } | null>(null);
  let sysWrapEl: HTMLDivElement | null = $state(null);
  let moreWrapEl: HTMLDivElement | null = $state(null);
  let shareWrapEl: HTMLDivElement | null = $state(null);
  let brandBtnEl: HTMLButtonElement | null = $state(null);
  let cacheClearBtnEl: HTMLButtonElement | null = $state(null);
  let brandClearConfirmPending = $state(false);
  let brandClearConfirmTimer: ReturnType<typeof setTimeout> | null = null;
  let cacheClearConfirmPending = $state(false);
  let cacheClearConfirmTimer: ReturnType<typeof setTimeout> | null = null;
  let appliedDraftNotice = $state<{ graphId: string; isAiDraft: boolean } | null>(null);
  let importFileInput: HTMLInputElement | null = $state(null);
  let firstRunAck = $state(false);
  let showOnboarding = $state(false);
  let showExamplesGallery = $state(false);
  let activePaperInsight = $state<PaperInsightExample | null>(null);
  let paperWalkthroughIdx = $state(0);
  let showPaperWalkthrough = $state(false);
  let hasActivePaperOutputs = $derived.by(() => {
    if (!activePaperInsight) return false;
    return activePaperInsight.walkthrough.some((step) => {
      if (!step.nodeId) return false;
      const outs = outputsByNode[step.nodeId];
      return Boolean(outs && Object.keys(outs).length > 0);
    });
  });
  let activePaperMutationHighlight = $derived.by(() => {
    if (!activePaperInsight || !showPaperWalkthrough || !hasActivePaperOutputs) return null;
    const step = activePaperInsight.walkthrough[paperWalkthroughIdx];
    if (!step?.mutationHighlight || !step.nodeId || step.nodeId !== selectedNodeId) return null;
    return {
      key: `${activePaperInsight.id}:${paperWalkthroughIdx}:${step.nodeId}`,
      ...step.mutationHighlight,
    };
  });
  let onboardingInteractive = $state(false);
  let canvasFitViewTrigger = $state(0);
  let showSeqParentBackbone = $state(false);
  let pendingRun = $state<{
    bypassCache: boolean;
    upToNodeId?: string;
    preflight: RunPreflightSummary;
  } | null>(null);

  function mergeAdapterDefaults(g: WorkflowGraph): WorkflowGraph {
    return {
      ...g,
      nodes: g.nodes.map((n) => {
        const defaults = defaultParamsForAdapter(n.adapterId);
        const merged: Record<string, unknown> = { ...defaults };
        for (const [k, v] of Object.entries(n.params ?? {})) {
          if (v !== undefined) merged[k] = v;
        }
        return { ...n, params: merged };
      }),
    };
  }

  // Active workflow state
  let selectedPresetId = $state(WORKFLOW_PRESETS[0].id);
  let graph = $state<WorkflowGraph>(
    autoLayoutGraph(mergeAdapterDefaults(structuredClone(WORKFLOW_PRESETS[0].graph))),
  );
  let activeDraftPreview = $state<PlannedWorkflowResult | null>(null);
  interface GraphHistoryEntry {
    graph: WorkflowGraph;
    selectedPresetId: string;
  }
  const MAX_GRAPH_HISTORY = 40;
  let graphUndoStack: GraphHistoryEntry[] = [];
  let graphRedoStack: GraphHistoryEntry[] = [];
  let lastParamUndo: ParamUndoMarker | null = null;
  let lastLabelUndo: LabelUndoMarker | null = null;
  let selectedNodeIds = $state<string[]>([WORKFLOW_PRESETS[0].graph.nodes[0].id]);
  let selectedNodeId = $derived(selectedNodeIds[selectedNodeIds.length - 1] ?? '');
  let activeCompareNode = $derived.by(() => {
    if (selectedNodeIds.length !== 1) return null;
    const n = graph.nodes.find((node) => node.id === selectedNodeIds[0]);
    return n?.adapterId === 'structure-compare' ? n : null;
  });
  let activeCompareRefNodeId = $derived.by((): string | null => {
    if (!activeCompareNode) return null;
    return (
      graph.edges.find(
        (e) => e.toNode === activeCompareNode.id && e.toPort === 'reference',
      )?.fromNode ?? null
    );
  });
  let activeCompareInputIds = $derived.by((): string[] => {
    if (!activeCompareNode) return [];
    const ordered: string[] = [];
    const seen = new Set<string>();
    const refEdge = graph.edges.find(
      (e) => e.toNode === activeCompareNode.id && e.toPort === 'reference',
    );
    if (refEdge && !seen.has(refEdge.fromNode)) {
      seen.add(refEdge.fromNode);
      ordered.push(refEdge.fromNode);
    }
    for (const e of graph.edges) {
      if (e.toNode === activeCompareNode.id && e.toPort === 'structures' && !seen.has(e.fromNode)) {
        seen.add(e.fromNode);
        ordered.push(e.fromNode);
      }
    }
    return ordered.slice(0, COMPARE_PALETTE.length);
  });
  let effectiveCompareNodeIds = $derived(
    selectedNodeIds.length >= 2
      ? selectedNodeIds.slice(0, COMPARE_PALETTE.length)
      : activeCompareNode
        ? activeCompareInputIds
        : [],
  );
  let isCompareMode = $derived(
    selectedNodeIds.length >= 2 || (Boolean(activeCompareNode) && activeCompareInputIds.length >= 1),
  );
  let compareLayout = $state<'overlay' | 'side-by-side'>('overlay');
  let compareColorMode = $state<'by-model' | 'plddt'>('by-model');
  let compareAlignKabsch = $state(true);
  let comparePairingMode = $state<ComparisonPairingMode | undefined>(undefined);
  let compareSelectedCandidateKey = $state<string | null>(null);
  let hiddenCompareModelIds = $state<string[]>([]);
  let branchCandidateIdx = $state<Record<string, number>>({});
  let localRunning = $state(false);
  let activeRunState = $state<ActiveRunState>(scheduler.getActiveRunState());
  let isRunning = $derived(
    localRunning || activeRunState.leaseHeld || activeRunState.status === 'running',
  );
  let hasInvalidNumeric = $derived(hasInvalidNumericInputs());

  function hasAnyInvalidNumericInput(): boolean {
    return (
      hasInvalidNumericInputs() ||
      (typeof document !== 'undefined' &&
        Boolean(document.querySelector('.num-input[aria-invalid="true"]')))
    );
  }

  function setSingleSelection(id: string) {
    selectedNodeIds = id ? [id] : [];
    hiddenCompareModelIds = [];
    comparePairingMode = undefined;
    compareSelectedCandidateKey = null;
    branchCandidateIdx = {};
    viewerFocusToken = null;
  }

  function handleSelectNodes(ids: string[]) {
    const valid = ids.filter((id) => graph.nodes.some((n) => n.id === id)).slice(0, COMPARE_PALETTE.length);
    selectedNodeIds = valid;
    hiddenCompareModelIds = [];
    comparePairingMode = undefined;
    compareSelectedCandidateKey = null;
    branchCandidateIdx = {};
    selectedPeekSlotId = null;
    selectedEdgeId = null;
    viewerFocusToken = null;
    if (selectedNodeIds.length >= 2) {
      rightPanelTab = 'peek';
    }
  }

  function toggleCompareModelVisibility(modelId: string) {
    if (hiddenCompareModelIds.includes(modelId)) {
      hiddenCompareModelIds = hiddenCompareModelIds.filter((id) => id !== modelId);
    } else {
      hiddenCompareModelIds = [...hiddenCompareModelIds, modelId];
    }
  }

  function soloCompareModel(modelId: string) {
    const allIds = comparisonState?.models.map((m) => m.id) ?? [];
    const refWasHidden = hiddenCompareModelIds.includes('ref-designed');
    if (modelId === 'ref-designed') {
      const nonRef = allIds.filter((id) => id !== 'ref-designed');
      const alreadySolo =
        nonRef.length > 0 &&
        nonRef.every((id) => hiddenCompareModelIds.includes(id)) &&
        !refWasHidden;
      hiddenCompareModelIds = alreadySolo ? [] : nonRef;
      return;
    }
    const others = allIds.filter((id) => id !== modelId && id !== 'ref-designed');
    const alreadySolo =
      others.length > 0 &&
      others.every((id) => hiddenCompareModelIds.includes(id)) &&
      !hiddenCompareModelIds.includes(modelId);
    if (alreadySolo) {
      hiddenCompareModelIds = refWasHidden ? ['ref-designed'] : [];
    } else {
      hiddenCompareModelIds = refWasHidden ? ['ref-designed', ...others] : others;
    }
  }
  let runTargetIds = $state<string[]>([]);
  let showRecipe = $state(false);
  let showCredits = $state(false);
  let shareToast = $state<string | null>(null);
  let shareToastTone = $state<'ok' | 'error'>('ok');
  let shareToastTimer: ReturnType<typeof setTimeout> | null = null;
  let hashError = $state<string | null>(null);
  let hashRequestSeq = 0;
  let runError = $state<string | null>(null);
  let totalElapsedMs = $state<number | null>(null);

  let nodeStates = $state.raw<Record<string, NodeProgressEvent>>({});
  let outputsByNode = $state.raw<Record<string, Record<string, CanonicalArtifact>>>({});

  // Progress throttling / rAF coalescing (~100ms, flushing status transitions immediately)
  let pendingProgressEvents: Record<string, NodeProgressEvent> = {};
  let progressFlushTimer: ReturnType<typeof setTimeout> | null = null;
  let progressRafId: number | null = null;
  let lastProgressFlushMs = 0;

  function clearPendingProgress() {
    pendingProgressEvents = {};
    if (progressFlushTimer !== null) {
      clearTimeout(progressFlushTimer);
      progressFlushTimer = null;
    }
    if (progressRafId !== null && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(progressRafId);
      progressRafId = null;
    }
  }

  function flushPendingProgress() {
    if (progressFlushTimer !== null) {
      clearTimeout(progressFlushTimer);
      progressFlushTimer = null;
    }
    if (progressRafId !== null && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(progressRafId);
      progressRafId = null;
    }
    const keys = Object.keys(pendingProgressEvents);
    if (keys.length === 0) return;
    const batch = pendingProgressEvents;
    pendingProgressEvents = {};
    lastProgressFlushMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    nodeStates = { ...nodeStates, ...batch };
    activeGpuWorker = scheduler.getActiveGpuAdapter();
  }

  const unsubscribeRunState = scheduler.subscribeRunState((nextState) => {
    activeRunState = nextState;
    activeGpuWorker = scheduler.getActiveGpuAdapter();
  });

  function shallowParamsEqual(
    a: Record<string, unknown> | undefined,
    b: Record<string, unknown> | undefined,
  ): boolean {
    if (a === b) return true;
    if (!a || !b) return false;
    const keysA = Object.keys(a);
    const keysB = Object.keys(b);
    if (keysA.length !== keysB.length) return false;
    for (let i = 0; i < keysA.length; i++) {
      const k = keysA[i];
      if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
      const va = a[k];
      const vb = b[k];
      if (!Object.is(va, vb)) {
        if (Array.isArray(va) && Array.isArray(vb) && va.length === vb.length) {
          for (let j = 0; j < va.length; j++) {
            if (!Object.is(va[j], vb[j])) return false;
          }
        } else {
          return false;
        }
      }
    }
    return true;
  }

  // Viewer & intermediate peek state
  let selectedPeekSlotId = $state<string | null>(null);
  let selectedEdgeId = $state<string | null>(null);
  let selectedRowIdx = $state(0);
  let selectedFoldIdx = $state(0);
  let showDesigned = $state(true);
  let showRefold = $state(true);
  let showTrajectory = $state(false);
  let illustrative = $state(false);
  let viewerSpin = $state(true);
  let trajFrame = $state(0);
  let molPickMode = $state<'auto' | 'off' | 'hotspots' | 'motif' | 'mpnn'>('auto');
  let lastPickedResidue = $state<string | null>(null);

  function showToastMessage(msg: string, ms = 3200, tone: 'ok' | 'error' = 'ok') {
    shareToast = msg;
    shareToastTone = tone;
    if (shareToastTimer) clearTimeout(shareToastTimer);
    shareToastTimer = setTimeout(() => {
      shareToast = null;
    }, ms);
  }

  function applyTheme(next: 'dark' | 'light') {
    theme = next;
    if (typeof document !== 'undefined') {
      document.documentElement.dataset.theme = next;
    }
    try {
      localStorage.setItem('denovo-theme', next);
    } catch {
      /* ignore */
    }
  }

  function toggleTheme() {
    applyTheme(theme === 'dark' ? 'light' : 'dark');
  }

  function toggleSplitMode() {
    splitMode = splitMode === 'horizontal' ? 'vertical' : 'horizontal';
    try {
      localStorage.setItem('denovo-split-mode', splitMode);
    } catch {
      /* ignore */
    }
  }

  function startSplitDrag(ev: PointerEvent) {
    if (splitMode !== 'horizontal' && splitMode !== 'vertical') return;
    ev.preventDefault();
    draggingSplit = true;
  }

  function persistSplitSizes() {
    try {
      localStorage.setItem('denovo-split-h-pct', String(splitPct));
      localStorage.setItem('denovo-split-v-pct', String(verticalSplitPct));
    } catch {
      /* ignore */
    }
  }

  function handleWindowPointerMove(ev: PointerEvent) {
    if (!draggingSplit || !splitWrapEl) return;
    pendingSplitClientX = ev.clientX;
    pendingSplitClientY = ev.clientY;
    if (splitMoveRafId !== null) return;
    splitMoveRafId = requestAnimationFrame(() => {
      splitMoveRafId = null;
      if (!draggingSplit || !splitWrapEl) return;
      const rect = splitWrapEl.getBoundingClientRect();
      if (splitMode === 'horizontal') {
        if (pendingSplitClientX === null || rect.width <= 0) return;
        const rawPct = ((pendingSplitClientX - rect.left) / rect.width) * 100;
        splitPct = Math.min(78, Math.max(32, Math.round(rawPct * 10) / 10));
      } else {
        if (pendingSplitClientY === null || rect.height <= 0) return;
        const rawPct = ((pendingSplitClientY - rect.top) / rect.height) * 100;
        verticalSplitPct = Math.min(76, Math.max(24, Math.round(rawPct * 10) / 10));
      }
    });
  }

  function handleWindowPointerUp() {
    if (splitMoveRafId !== null) {
      cancelAnimationFrame(splitMoveRafId);
      splitMoveRafId = null;
    }
    pendingSplitClientX = null;
    pendingSplitClientY = null;
    if (draggingSplit) {
      draggingSplit = false;
      persistSplitSizes();
    }
  }

  function clearStaleUrlHash() {
    if (typeof window !== 'undefined' && window.location.hash) {
      window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`);
    }
  }

  function cancelBrandClearConfirm() {
    brandClearConfirmPending = false;
    if (brandClearConfirmTimer) {
      clearTimeout(brandClearConfirmTimer);
      brandClearConfirmTimer = null;
    }
  }

  function cancelCacheClearConfirm() {
    cacheClearConfirmPending = false;
    if (cacheClearConfirmTimer) {
      clearTimeout(cacheClearConfirmTimer);
      cacheClearConfirmTimer = null;
    }
  }

  function handleBrandLogoClick() {
    if (isRunning) return;
    if (graph.nodes.length === 0 && graph.edges.length === 0) {
      cancelBrandClearConfirm();
      resetToHomeCanvas();
      return;
    }
    if (!brandClearConfirmPending) {
      brandClearConfirmPending = true;
      if (brandClearConfirmTimer) clearTimeout(brandClearConfirmTimer);
      brandClearConfirmTimer = setTimeout(() => {
        brandClearConfirmPending = false;
        brandClearConfirmTimer = null;
      }, 10000);
      return;
    }
    cancelBrandClearConfirm();
    resetToHomeCanvas();
  }

  function handleClearCacheClick() {
    if (isRunning || activeRunState.leaseHeld) return;
    if (!cacheClearConfirmPending) {
      cacheClearConfirmPending = true;
      if (cacheClearConfirmTimer) clearTimeout(cacheClearConfirmTimer);
      cacheClearConfirmTimer = setTimeout(() => {
        cacheClearConfirmPending = false;
        cacheClearConfirmTimer = null;
      }, 10000);
      return;
    }
    cancelCacheClearConfirm();
    void handleClearCache();
  }

  function resetToHomeCanvas() {
    if (isRunning) return;
    cancelBrandClearConfirm();
    cancelCacheClearConfirm();
    sysMenuOpen = false;
    moreMenuOpen = false;
    shareMenuOpen = false;
    sharePopoverState = null;
    showRecipe = false;
    expandedPane = 'none';
    studioHoveredToken = null;
    studioFocusToken = null;
    studioActiveBrush = null;
    appliedDraftNotice = null;
    hashError = null;
    activePaperInsight = null;
    showPaperWalkthrough = false;
    clearStaleUrlHash();
    if (graph.nodes.length === 0 && graph.edges.length === 0) {
      selectedPresetId = 'custom-workflow';
      setSingleSelection('');
      return;
    }
    handleGraphChange({
      id: 'custom-workflow',
      name: 'Custom Workflow',
      description: '',
      nodes: [],
      edges: [],
    });
    selectedPresetId = 'custom-workflow';
    setSingleSelection('');
    rightPanelTab = 'params';
    showToastMessage('Cleared canvas (⌘Z to undo)', 2200);
  }

  function handleWindowPointerDown(ev: PointerEvent) {
    const target = ev.target as Node | null;
    if (!target) return;
    if (brandClearConfirmPending && !brandBtnEl?.contains(target)) {
      cancelBrandClearConfirm();
    }
    if (cacheClearConfirmPending && !cacheClearBtnEl?.contains(target)) {
      cancelCacheClearConfirm();
    }
    if (sysMenuOpen && !sysWrapEl?.contains(target)) {
      sysMenuOpen = false;
      cancelCacheClearConfirm();
    }
    if (moreMenuOpen && !moreWrapEl?.contains(target)) {
      moreMenuOpen = false;
    }
    if (shareMenuOpen && !shareWrapEl?.contains(target)) {
      shareMenuOpen = false;
    }
  }

  function isEditableElement(el: EventTarget | null): boolean {
    if (!(el instanceof HTMLElement)) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
  }

  function isInteractiveOrEditableElement(el: EventTarget | null): boolean {
    if (!(el instanceof Element)) return false;
    if (el instanceof HTMLElement && isEditableElement(el)) return true;
    return Boolean(
      el.closest(
        'button, a[href], summary, [role="button"], input, textarea, select, [contenteditable]:not([contenteditable="false"])',
      ),
    );
  }

  function handleWindowKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      if (ev.defaultPrevented) return;
      // Top-most ModelCardPopover manages its own Escape; never collapse Studio/overlays underneath it
      if (typeof document !== 'undefined' && document.querySelector('.mc-backdrop')) {
        return;
      }
      if (pendingRun) {
        ev.preventDefault();
        cancelPendingRun();
        return;
      }
      if (showExamplesGallery) {
        ev.preventDefault();
        showExamplesGallery = false;
        return;
      }
      if (showOnboarding) {
        if (
          onboardingInteractive &&
          typeof document !== 'undefined' &&
          document.querySelector('.catalog-shell')
        ) {
          return;
        }
        ev.preventDefault();
        dismissOnboarding();
        return;
      }
      if (activePaperInsight && showPaperWalkthrough && !showCredits) {
        ev.preventDefault();
        showPaperWalkthrough = false;
        return;
      }
      if (showCredits) {
        ev.preventDefault();
        showCredits = false;
        return;
      }
      if (expandedPane !== 'none') {
        ev.preventDefault();
        expandedPane = 'none';
        studioHoveredToken = null;
        studioFocusToken = null;
        studioActiveBrush = null;
        return;
      }
      if (sysMenuOpen || moreMenuOpen || shareMenuOpen) {
        ev.preventDefault();
        sysMenuOpen = false;
        moreMenuOpen = false;
        shareMenuOpen = false;
        cancelCacheClearConfirm();
        return;
      }
      if (brandClearConfirmPending) {
        ev.preventDefault();
        cancelBrandClearConfirm();
        return;
      }
      if (showRecipe) {
        ev.preventDefault();
        showRecipe = false;
        return;
      }
      return;
    }
    if (
      pendingRun ||
      showCredits ||
      showExamplesGallery ||
      (showOnboarding && !onboardingInteractive)
    ) {
      return;
    }

    // Cmd/Ctrl + Z undo, Cmd/Ctrl + Shift + Z or Ctrl + Y redo (when not typing in a text field or previewing a draft)
    if (
      (ev.metaKey || ev.ctrlKey) &&
      !ev.altKey &&
      !activeDraftPreview &&
      !shouldAllowNativeTextUndo(ev.target) &&
      !shouldAllowNativeTextUndo(document.activeElement)
    ) {
      const k = ev.key.toLowerCase();
      if (k === 'z' && !ev.shiftKey) {
        if (!isRunning && graphUndoStack.length > 0) {
          ev.preventDefault();
          undoGraphEdit();
        }
        return;
      }
      if ((k === 'z' && ev.shiftKey) || (k === 'y' && !ev.metaKey)) {
        if (!isRunning && graphRedoStack.length > 0) {
          ev.preventDefault();
          redoGraphEdit();
        }
        return;
      }
    }

    // Cmd/Ctrl + Enter runs workflow from anywhere (never hijacks Cmd+R browser reload)
    if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey) && !ev.altKey) {
      if (hasAnyInvalidNumericInput()) {
        ev.preventDefault();
        runError = 'Fix invalid numeric parameter(s) before running.';
        return;
      }
      if (hasBioErrors) {
        ev.preventDefault();
        runError = bioErrorIssues.map((i) => i.message).join('\n');
        return;
      }
      if (!isRunning && validationErrors.length === 0) {
        ev.preventDefault();
        void runWorkflow(false);
      }
      return;
    }

    // Space toggles Node Studio when a block is selected and focus is not on an interactive/editable control
    if (
      ev.key === ' ' &&
      !ev.metaKey &&
      !ev.ctrlKey &&
      !ev.altKey &&
      !isInteractiveOrEditableElement(ev.target) &&
      !isInteractiveOrEditableElement(document.activeElement)
    ) {
      if (selectedNode) {
        ev.preventDefault();
        if (expandedPane === 'studio') {
          expandedPane = 'none';
          studioHoveredToken = null;
          studioFocusToken = null;
          studioActiveBrush = null;
        } else {
          openStudioForNode(selectedNode.id);
        }
        return;
      }
    }

    // [ and ] step upstream/downstream across workflow blocks while Node Studio is open
    if (
      expandedPane === 'studio' &&
      (ev.key === '[' || ev.key === ']') &&
      !ev.metaKey &&
      !ev.ctrlKey &&
      !ev.altKey &&
      !isEditableElement(ev.target) &&
      !isEditableElement(document.activeElement)
    ) {
      const idx = graph.nodes.findIndex((n) => n.id === selectedNodeId);
      if (idx >= 0) {
        const nextIdx = ev.key === '[' ? idx - 1 : idx + 1;
        if (nextIdx >= 0 && nextIdx < graph.nodes.length) {
          ev.preventDefault();
          handleSelectNode(graph.nodes[nextIdx].id);
        }
      }
      return;
    }

    // Bare 'R' runs workflow when not typing in an input/textarea/select
    if (
      (ev.key === 'r' || ev.key === 'R') &&
      !ev.metaKey &&
      !ev.ctrlKey &&
      !ev.altKey &&
      !isEditableElement(ev.target) &&
      !isEditableElement(document.activeElement)
    ) {
      if (hasAnyInvalidNumericInput()) {
        ev.preventDefault();
        runError = 'Fix invalid numeric parameter(s) before running.';
        return;
      }
      if (hasBioErrors) {
        ev.preventDefault();
        runError = bioErrorIssues.map((i) => i.message).join('\n');
        return;
      }
      if (!isRunning && validationErrors.length === 0) {
        ev.preventDefault();
        void runWorkflow(false);
      }
    }
  }

  async function applySharedHash(
    rawHash: string,
    isInitialMount = false,
  ): Promise<{ loadedFromHash: boolean; attemptedSharedHash: boolean }> {
    const reqId = ++hashRequestSeq;
    const trimmed = rawHash.trim();
    const isSharedHash = isSharedWorkflowHash(trimmed);
    if (trimmed.length <= 2 && !isSharedHash) {
      return { loadedFromHash: false, attemptedSharedHash: false };
    }
    if (!isInitialMount && isRunning) {
      const msg = 'Cannot load shared workflow URL while a run is active — cancel the run first.';
      hashError = msg;
      showToastMessage(msg, 4200, 'error');
      return { loadedFromHash: false, attemptedSharedHash: isSharedHash };
    }
    try {
      const decoded = await decodeHashToGraph(rawHash);
      if (reqId !== hashRequestSeq) {
        return { loadedFromHash: false, attemptedSharedHash: isSharedHash };
      }
      if (decoded) {
        hashError = null;
        showOnboarding = false;
        loadFullWorkflowGraph(decoded, { clearOutputs: true, preserveHash: true });
        canvasFitViewTrigger += 1;
        showToastMessage('Loaded shared workflow from URL', 3000, 'ok');
        return { loadedFromHash: true, attemptedSharedHash: isSharedHash };
      }
      if (isSharedHash) {
        const msg = isInitialMount
          ? 'Could not decode shared workflow link — loaded default workflow.'
          : 'Could not decode shared workflow link — kept current workflow.';
        hashError = msg;
        showToastMessage(msg, 4500, 'error');
        return { loadedFromHash: false, attemptedSharedHash: true };
      }
    } catch (err) {
      if (reqId !== hashRequestSeq) {
        return { loadedFromHash: false, attemptedSharedHash: isSharedHash };
      }
      if (isSharedHash) {
        const msg = `Invalid shared workflow URL: ${err instanceof Error ? err.message : 'decode failed'}`;
        hashError = msg;
        showToastMessage(msg, 4500, 'error');
        return { loadedFromHash: false, attemptedSharedHash: true };
      }
    }
    return { loadedFromHash: false, attemptedSharedHash: isSharedHash };
  }

  onMount(async () => {
    try {
      const saved = localStorage.getItem('denovo-theme');
      if (saved === 'light' || saved === 'dark') {
        applyTheme(saved);
      } else if (window.matchMedia?.('(prefers-color-scheme: light)').matches) {
        applyTheme('light');
      } else {
        applyTheme('dark');
      }
      const savedSplit = localStorage.getItem('denovo-split-mode');
      if (savedSplit === 'horizontal' || savedSplit === 'vertical') {
        splitMode = savedSplit;
      }
      const savedHPct = Number(localStorage.getItem('denovo-split-h-pct'));
      if (Number.isFinite(savedHPct) && savedHPct >= 32 && savedHPct <= 78) {
        splitPct = savedHPct;
      }
      const savedVPct = Number(localStorage.getItem('denovo-split-v-pct'));
      if (Number.isFinite(savedVPct) && savedVPct >= 24 && savedVPct <= 76) {
        verticalSplitPct = savedVPct;
      }
      firstRunAck = localStorage.getItem('denovo-first-run-ack') === '1';
    } catch {
      applyTheme('dark');
    }

    if (!window.isSecureContext) {
      webgpuSupported = false;
      webgpuIsFallback = false;
      webgpuAdapterInfo = null;
      webgpuDetail = 'needs HTTPS/localhost';
    } else if (!('gpu' in navigator) || !navigator.gpu) {
      webgpuSupported = false;
      webgpuIsFallback = false;
      webgpuAdapterInfo = null;
      webgpuDetail = 'unavailable in browser';
    } else {
      try {
        const res = await requestResilientWebGpuAdapter({
          retries: 3,
          retryDelayMs: 250,
          allowSoftwareFallback: true,
        });
        webgpuSupported = true;
        webgpuIsFallback = res.isSoftwareFallback;
        webgpuAdapterInfo = res.info;
        webgpuDetail = formatWebGpuAdapterSummary(res.info, res.hasF16);
      } catch {
        webgpuSupported = false;
        webgpuIsFallback = false;
        webgpuAdapterInfo = null;
        webgpuDetail = 'no GPU adapter';
      }
    }

    // Load saved self-hosted compute backend config & auto-detect same-origin selfhost-server.mjs
    const loadedBackend = loadComputeBackendConfig();
    computeBackend = loadedBackend;
    syncSchedulerRemoteExecutor(loadedBackend);
    if (loadedBackend.mode !== 'local') {
      void probeRemoteBackend(loadedBackend);
    } else if (typeof window !== 'undefined' && window.location.protocol.startsWith('http')) {
      void checkRemoteServerHealth('', '', fetch, 1500).then((health) => {
        if (!health.ok) return;
        remoteHealth = health;
        const hasSavedPref = Boolean(localStorage.getItem('denovo.computeBackend.v1'));
        if (!hasSavedPref) {
          const autoMode: ComputeRoutingMode =
            webgpuSupported === false || webgpuIsFallback ? 'remote' : 'hybrid';
          computeBackend = { ...computeBackend, mode: autoMode, serverUrl: '' };
          syncSchedulerRemoteExecutor(computeBackend);
        }
      });
    }
    const { webMcpRegistered } = attachProgrammaticApi(scheduler, handleProgress);
    webMcpReady = webMcpRegistered;

    // Load shared workflow from URL hash (#p=... or #g=...) if present
    const { loadedFromHash, attemptedSharedHash } = await applySharedHash(window.location.hash, true);

    // Returning visitors without a shared link land on a clean blank canvas;
    // first-time visitors see the starter workflow + quick tour.
    try {
      const isReturning = localStorage.getItem('denovo-visited') === '1';
      localStorage.setItem('denovo-visited', '1');
      if (isReturning && !loadedFromHash && !attemptedSharedHash) {
        loadFullWorkflowGraph(
          {
            id: 'custom-workflow',
            name: 'Custom Workflow',
            description: '',
            nodes: [],
            edges: [],
          },
          { clearOutputs: true },
        );
      } else if (
        !shouldSuppressOnboardingForHash(loadedFromHash, attemptedSharedHash) &&
        localStorage.getItem('denovo-onboarding-seen') !== '1'
      ) {
        showOnboarding = true;
      }
    } catch {
      /* ignore */
    }

    await refreshCacheStats();
  });

  onDestroy(() => {
    unsubscribeRunState();
    clearPendingProgress();
    if (splitMoveRafId !== null && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(splitMoveRafId);
    }
    if (shareToastTimer) clearTimeout(shareToastTimer);
    if (brandClearConfirmTimer) clearTimeout(brandClearConfirmTimer);
    if (cacheClearConfirmTimer) clearTimeout(cacheClearConfirmTimer);
    if (!activeRunState.leaseHeld) {
      void scheduler.releaseActiveGpu();
    }
  });

  async function refreshCacheStats() {
    const st = await getCacheStats();
    cacheUsedMB = st.usedBytes / 1024 / 1024;
  }

  function handleProgress(ev: NodeProgressEvent) {
    const node = graph.nodes.find((n) => n.id === ev.nodeId);
    const isRemoteEv =
      ev.phase === 'remote' ||
      Boolean(ev.detail?.startsWith('[Host GPU]')) ||
      (Boolean(node) &&
        shouldRouteNodeToRemote(node!, computeBackend, {
          localWebGpuAvailable: webgpuSupported !== false,
          localWebGpuFallback: webgpuIsFallback,
        }));
    if (!isRemoteEv && webgpuSupported === false && (ev.status === 'running' || ev.status === 'done')) {
      if (node && ADAPTER_REGISTRY[node.adapterId]?.runtime !== 'cpu-ts') {
        webgpuSupported = true;
        webgpuDetail = 'ready (worker verified)';
      }
    }
    const prev = pendingProgressEvents[ev.nodeId] ?? nodeStates[ev.nodeId];
    const statusChanged = !prev || prev.status !== ev.status;
    pendingProgressEvents[ev.nodeId] = ev;
    if (statusChanged) {
      flushPendingProgress();
      return;
    }
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const elapsed = now - lastProgressFlushMs;
    if (elapsed >= 100) {
      if (progressRafId === null) {
        if (typeof requestAnimationFrame === 'function') {
          progressRafId = requestAnimationFrame(() => {
            progressRafId = null;
            flushPendingProgress();
          });
        } else {
          flushPendingProgress();
        }
      }
    } else if (progressFlushTimer === null) {
      progressFlushTimer = setTimeout(() => {
        progressFlushTimer = null;
        if (typeof requestAnimationFrame === 'function') {
          progressRafId = requestAnimationFrame(() => {
            progressRafId = null;
            flushPendingProgress();
          });
        } else {
          flushPendingProgress();
        }
      }, Math.max(0, 100 - elapsed));
    }
  }

  function selectPreset(presetId: string) {
    if (isRunning) return;
    const preset = WORKFLOW_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    clearStaleUrlHash();
    cancelBrandClearConfirm();
    sharePopoverState = null;
    appliedDraftNotice = null;
    showExamplesGallery = false;
    activePaperInsight = null;
    showPaperWalkthrough = false;
    graphUndoStack = [];
    graphRedoStack = [];
    lastParamUndo = null;
    lastLabelUndo = null;
    hashError = null;
    selectedPresetId = presetId;
    const merged = mergeAdapterDefaults(structuredClone(preset.graph));
    const { graph: g } = reconcileBenchmarkSeeding(merged, { teardown: false });
    graph = g.nodes.some((n) => !n.position) ? autoLayoutGraph(g) : g;
    setSingleSelection(graph.nodes[0]?.id ?? '');
    branchCandidateIdx = {};
    selectedPeekSlotId = null;
    selectedEdgeId = null;
    rightPanelTab = 'params';
    clearPendingProgress();
    nodeStates = {};
    outputsByNode = {};
    previewedInputSigs.clear();
    inFlightInputSigs.clear();
    failedInputSigs.clear();
    runError = null;
    totalElapsedMs = null;
    selectedRowIdx = 0;
    selectedFoldIdx = 0;
    showTrajectory = false;
    lastPickedResidue = null;
  }

  function applyPaperWalkthroughStep(example: PaperInsightExample, stepIdx: number) {
    if (!example.walkthrough.length) return;
    const clamped = Math.max(0, Math.min(example.walkthrough.length - 1, stepIdx));
    paperWalkthroughIdx = clamped;
    const step = example.walkthrough[clamped];
    if (!step.nodeId || !graph.nodes.some((n) => n.id === step.nodeId)) return;
    const targetTab = step.tab ?? 'peek';
    const switchingNode = selectedNodeId !== step.nodeId;
    setSingleSelection(step.nodeId);
    selectedEdgeId = null;
    rightPanelTab = targetTab;
    selectedPeekSlotId =
      targetTab === 'peek' && step.outputPort ? `out:${step.outputPort}` : null;
    if (step.focusResidue && targetTab === 'peek') {
      const targetTok = step.focusResidue;
      viewerFocusToken = null;
      setTimeout(() => {
        if (activePaperInsight?.id === example.id && paperWalkthroughIdx === clamped) {
          viewerFocusToken = targetTok;
        }
      }, switchingNode ? 180 : 30);
    }
  }

  async function selectPaperInsight(example: PaperInsightExample) {
    if (isRunning) return;
    const decoded = await decodeHashToGraph(example.hash);
    if (!decoded) {
      showToastMessage('Could not load example workflow.', 3200, 'error');
      return;
    }
    clearStaleUrlHash();
    appliedDraftNotice = null;
    showExamplesGallery = false;
    activePaperInsight = example;
    paperWalkthroughIdx = 0;
    showPaperWalkthrough = true;
    selectedPresetId = example.id;
    loadFullWorkflowGraph(
      { ...decoded, id: example.id, name: example.title },
      { clearOutputs: true, keepPaperInsight: true },
    );
    canvasFitViewTrigger += 1;
    showToastMessage(`Loaded example: ${example.title}`, 2600, 'ok');
  }

  function dismissOnboarding() {
    showOnboarding = false;
    onboardingInteractive = false;
    try {
      localStorage.setItem('denovo-onboarding-seen', '1');
    } catch {
      /* ignore */
    }
  }

  function prepareTourStarter() {
    if (isRunning) return;
    clearStaleUrlHash();
    appliedDraftNotice = null;
    sharePopoverState = null;
    graphUndoStack = [];
    graphRedoStack = [];
    lastParamUndo = null;
    lastLabelUndo = null;
    hashError = null;
    selectedPresetId = 'tour-starter';
    graph = autoLayoutGraph({
      id: 'tour-starter',
      name: 'De Novo Design with Genie 3',
      nodes: [
        {
          id: 'genie3_1',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: { ...defaultParamsForAdapter('genie3'), length: 70, nSampleStep: 50 },
          position: { x: 60, y: 120 },
        },
      ],
      edges: [],
    });
    setSingleSelection('genie3_1');
    clearPendingProgress();
    nodeStates = {};
    outputsByNode = {};
    previewedInputSigs.clear();
    inFlightInputSigs.clear();
    failedInputSigs.clear();
    runError = null;
    rightPanelTab = 'params';
    canvasFitViewTrigger += 1;
  }

  function autoStepTourBlock(adapterId: 'web-protein-mpnn' | 'localfold-af2' | 'designability-scorer') {
    if (isRunning) return;
    const opts =
      adapterId === 'web-protein-mpnn'
        ? { autoCommit: true, params: { family: 'ca', modelName: 'v_48_020' }, label: 'Cα-ProteinMPNN' }
        : adapterId === 'localfold-af2'
          ? { autoCommit: true, params: { model: 'monomer' }, label: 'AF2 Model 1' }
          : { autoCommit: true, label: 'scRMSD / scTM' };
    const anchorId =
      adapterId === 'designability-scorer'
        ? (graph.nodes.find((n) => n.adapterId === 'localfold-af2')?.id ?? selectedNodeId)
        : adapterId === 'localfold-af2'
          ? (graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn')?.id ?? selectedNodeId)
          : (graph.nodes.find((n) => n.adapterId === 'genie3')?.id ??
            graph.nodes.find((n) => n.adapterId === 'pdb-input')?.id ??
            selectedNodeId);

    const prevGraph = $state.snapshot(graph) as WorkflowGraph;
    pushGraphUndo(prevGraph);
    const { graph: nextGraph, nodeId } = insertBlockSmart(
      prevGraph,
      adapterId,
      anchorId || undefined,
      undefined,
      opts,
    );
    const { graph: reconciled } = reconcileBenchmarkSeeding(nextGraph);
    const laid = autoLayoutGraph(reconciled);
    graph = laid;
    scheduler.invalidateEditedGraph(prevGraph, laid);
    setSingleSelection(nodeId);
    runError = null;
    canvasFitViewTrigger += 1;
  }

  function handleSelectNode(nodeId: string) {
    setSingleSelection(nodeId);
    selectedPeekSlotId = null;
    selectedEdgeId = null;
    showSeqParentBackbone = false;
    const clicked = graph.nodes.find((n) => n.id === nodeId);
    const hasOutputs = Boolean(
      nodeId && outputsByNode[nodeId] && Object.keys(outputsByNode[nodeId]).length > 0,
    );
    if (clicked?.adapterId === 'designability-scorer' && hasOutputs) {
      rightPanelTab = 'peek';
    } else {
      rightPanelTab = 'params';
    }
  }

  function handleBlockAdded(nodeId: string) {
    setSingleSelection(nodeId);
    selectedPeekSlotId = null;
    selectedEdgeId = null;
    showSeqParentBackbone = false;
    rightPanelTab = 'params';
  }

  function handleSelectEdge(edge: WorkflowEdge) {
    setSingleSelection(edge.toNode);
    selectedPeekSlotId = `in:${edge.toPort}`;
    selectedEdgeId = makeEdgeId(edge.fromNode, edge.fromPort, edge.toNode, edge.toPort);
    showSeqParentBackbone = false;
    rightPanelTab = 'peek';
  }

  function handlePeekPort(nodeId: string, direction: 'in' | 'out', portName: string) {
    setSingleSelection(nodeId);
    selectedPeekSlotId = `${direction}:${portName}`;
    showSeqParentBackbone = false;
    if (direction === 'in') {
      const edge = graph.edges.find((e) => e.toNode === nodeId && e.toPort === portName);
      selectedEdgeId = edge ? makeEdgeId(edge.fromNode, edge.fromPort, edge.toNode, edge.toPort) : null;
    } else {
      selectedEdgeId = null;
    }
    rightPanelTab = 'peek';
  }

  function handleSelectPeekSlot(slotId: string) {
    selectedPeekSlotId = slotId;
    showSeqParentBackbone = false;
    viewerFocusToken = null;
    const slot = peekSlots.find((s) => s.id === slotId);
    selectedEdgeId = slot?.direction === 'in' ? (slot.edgeId ?? null) : null;
  }

  function handleSelectPerResidue(positionIndex: number, residueId: string) {
    const tok = resolveResidueFocusToken(
      positionIndex,
      residueId,
      peekPerResidue,
      perResidueDisplayStructure,
    );
    if (!tok) return;
    if (studioOpen) {
      studioFocusToken = null;
      queueMicrotask(() => {
        studioFocusToken = tok;
      });
    } else {
      viewerFocusToken = null;
      queueMicrotask(() => {
        viewerFocusToken = tok;
      });
    }
  }

  function getDownstreamNodeIds(g: WorkflowGraph, seedIds: Iterable<string>): Set<string> {
    const visited = new Set<string>(seedIds);
    const queue = [...visited];
    while (queue.length > 0) {
      const cur = queue.shift()!;
      for (const e of g.edges) {
        if (e.fromNode === cur && !visited.has(e.toNode)) {
          visited.add(e.toNode);
          queue.push(e.toNode);
        }
      }
    }
    return visited;
  }

  function invalidateNodes(
    idsToClear: Set<string>,
    opts?: { keepSeedIds?: Set<string>; keepStateIds?: Set<string> },
  ) {
    if (idsToClear.size === 0) return;
    let outChanged = false;
    const nextOutputs: Record<string, Record<string, CanonicalArtifact>> = {};
    for (const [id, outs] of Object.entries(outputsByNode)) {
      if (idsToClear.has(id) && !opts?.keepSeedIds?.has(id)) {
        outChanged = true;
      } else {
        nextOutputs[id] = outs;
      }
    }
    if (outChanged) {
      outputsByNode = nextOutputs;
      totalElapsedMs = null;
    }

    let stChanged = false;
    const nextStates: Record<string, NodeProgressEvent> = {};
    for (const [id, st] of Object.entries(nodeStates)) {
      if (idsToClear.has(id) && !opts?.keepSeedIds?.has(id) && !opts?.keepStateIds?.has(id)) {
        stChanged = true;
      } else {
        nextStates[id] = st;
      }
    }
    if (stChanged) nodeStates = nextStates;
  }

  function pushGraphUndo(
    snapshot: WorkflowGraph,
    presetId = selectedPresetId,
    opts?: { preserveCoalescing?: boolean },
  ) {
    graphUndoStack.push({ graph: structuredClone(snapshot), selectedPresetId: presetId });
    if (graphUndoStack.length > MAX_GRAPH_HISTORY) {
      graphUndoStack.shift();
    }
    graphRedoStack = [];
    if (!opts?.preserveCoalescing) {
      lastParamUndo = null;
      lastLabelUndo = null;
    }
  }

  function undoGraphEdit() {
    if (isRunning || graphUndoStack.length === 0) return;
    const entry = graphUndoStack.pop()!;
    graphRedoStack.push({
      graph: structuredClone($state.snapshot(graph) as WorkflowGraph),
      selectedPresetId,
    });
    if (graphRedoStack.length > MAX_GRAPH_HISTORY) {
      graphRedoStack.shift();
    }
    lastParamUndo = null;
    lastLabelUndo = null;
    handleGraphChange(entry.graph, { recordHistory: false });
    selectedPresetId = entry.selectedPresetId;
    canvasFitViewTrigger += 1;
    showToastMessage('Undid graph edit', 1800);
  }

  function redoGraphEdit() {
    if (isRunning || graphRedoStack.length === 0) return;
    const entry = graphRedoStack.pop()!;
    graphUndoStack.push({
      graph: structuredClone($state.snapshot(graph) as WorkflowGraph),
      selectedPresetId,
    });
    if (graphUndoStack.length > MAX_GRAPH_HISTORY) {
      graphUndoStack.shift();
    }
    lastParamUndo = null;
    lastLabelUndo = null;
    handleGraphChange(entry.graph, { recordHistory: false });
    selectedPresetId = entry.selectedPresetId;
    canvasFitViewTrigger += 1;
    showToastMessage('Redid graph edit', 1800);
  }

  function handleApplyPlannedGraph(next: WorkflowGraph, summary: string, isAiDraft = false) {
    if (isRunning) return;
    const prevPresetId = selectedPresetId;
    hashError = null;
    appliedDraftNotice = { graphId: next.id, isAiDraft: Boolean(isAiDraft) };
    handleGraphChange(next, { prevPresetId });
    selectedPresetId = next.id;
    const firstId = next.nodes[0]?.id ?? '';
    setSingleSelection(firstId);
    rightPanelTab = 'params';
    canvasFitViewTrigger += 1;
    showToastMessage(`Drafted workflow: ${summary} (⌘Z to undo)`, 3200);
  }

  function handleGraphChange(
    next: WorkflowGraph,
    opts?: { recordHistory?: boolean; prevPresetId?: string },
  ) {
    const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
    if (opts?.recordHistory !== false) {
      pushGraphUndo(prevSnapshot, opts?.prevPresetId ?? selectedPresetId);
    }
    clearStaleUrlHash();
    sharePopoverState = null;

    // Synchronously reconcile benchmark auto-seeding (for unconfigured hasNoTaskParams or previously
    // auto-seeded Genie 3 & RFD4-Proteina blocks) and symmetric teardown on forward graph edits
    // so adding/wiring a generator seeds immediately before Parameters opens and 1 ⌘Z undoes both.
    const reconciledNext =
      opts?.recordHistory !== false ? reconcileBenchmarkSeeding(next).graph : next;

    const nextById = new Map(reconciledNext.nodes.map((n) => [n.id, n]));
    const prevById = new Map(prevSnapshot.nodes.map((n) => [n.id, n]));

    const dirtySeeds = new Set<string>();
    for (const [id, cur] of nextById.entries()) {
      const prev = prevById.get(id);
      if (!prev || cur.adapterId !== prev.adapterId) {
        dirtySeeds.add(id);
      } else if (!shallowParamsEqual(cur.params, prev.params)) {
        const displayOnlyCompare =
          cur.adapterId === 'structure-compare' &&
          prev.adapterId === 'structure-compare' &&
          isCompareDisplayOnlyParamChange(prev, cur.params);
        if (!displayOnlyCompare) {
          dirtySeeds.add(id);
        }
      }
    }

    const prevEdgeKeys = new Set(
      prevSnapshot.edges.map((e) => makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort)),
    );
    const nextEdgeKeys = new Set(
      reconciledNext.edges.map((e) => makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort)),
    );
    for (const e of reconciledNext.edges) {
      if (!prevEdgeKeys.has(makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort))) {
        dirtySeeds.add(e.toNode);
      }
    }
    for (const e of prevSnapshot.edges) {
      if (!nextEdgeKeys.has(makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort)) && nextById.has(e.toNode)) {
        dirtySeeds.add(e.toNode);
      }
    }

    const staleIds = getDownstreamNodeIds(reconciledNext, dirtySeeds);

    const nextOutputs: Record<string, Record<string, CanonicalArtifact>> = {};
    for (const [id, outs] of Object.entries(outputsByNode)) {
      const cur = nextById.get(id);
      const prev = prevById.get(id);
      if (cur && prev && cur.adapterId === prev.adapterId && !staleIds.has(id)) {
        nextOutputs[id] = outs;
      }
    }
    outputsByNode = nextOutputs;

    const nextStates: Record<string, NodeProgressEvent> = {};
    for (const [id, st] of Object.entries(nodeStates)) {
      const cur = nextById.get(id);
      const prev = prevById.get(id);
      if (cur && prev && cur.adapterId === prev.adapterId && !staleIds.has(id)) {
        nextStates[id] = st;
      }
    }
    nodeStates = nextStates;
    if (staleIds.size > 0) {
      totalElapsedMs = null;
    }

    // Reset stale workflow id/name/preset whenever the canvas becomes empty (e.g. Clear or deleting last node)
    const normalizedNext: WorkflowGraph =
      reconciledNext.nodes.length === 0 && prevSnapshot.nodes.length > 0
        ? {
            ...reconciledNext,
            id: 'custom-workflow',
            name: 'Custom Workflow',
            description: '',
          }
        : reconciledNext;
    if (reconciledNext.nodes.length === 0 && prevSnapshot.nodes.length > 0) {
      selectedPresetId = 'custom-workflow';
      appliedDraftNotice = null;
      cancelBrandClearConfirm();
    } else if (appliedDraftNotice && normalizedNext.id !== appliedDraftNotice.graphId) {
      appliedDraftNotice = null;
    }

    if (selectedEdgeId && !nextEdgeKeys.has(selectedEdgeId)) {
      selectedEdgeId = null;
      selectedPeekSlotId = null;
    }

    graph = normalizedNext;
    if (
      staleIds.size > 0 ||
      prevSnapshot.nodes.length !== normalizedNext.nodes.length ||
      prevSnapshot.edges.length !== normalizedNext.edges.length
    ) {
      scheduler.invalidateEditedGraph(prevSnapshot, normalizedNext);
    }

    // Prune deleted node IDs from selection & hidden compare models (resetting if all remaining would be hidden)
    let nextSelected = selectedNodeIds;
    if (selectedNodeIds.length > 0) {
      const kept = selectedNodeIds.filter((id) => nextById.has(id));
      if (kept.length !== selectedNodeIds.length) {
        nextSelected = kept.length > 0 ? kept : reconciledNext.nodes[0]?.id ? [reconciledNext.nodes[0].id] : [];
        selectedNodeIds = nextSelected;
      }
    }
    if (hiddenCompareModelIds.length > 0) {
      if (nextSelected.length < 2) {
        hiddenCompareModelIds = [];
      } else {
        const selectedSet = new Set(nextSelected);
        const pruned = hiddenCompareModelIds.filter((modelId) => {
          if (modelId === 'ref-designed') return true;
          if (modelId.startsWith('branch-')) {
            const nid = modelId.slice('branch-'.length);
            return nextById.has(nid) && selectedSet.has(nid);
          }
          return nextById.has(modelId) && selectedSet.has(modelId);
        });
        const allBranchesHidden = nextSelected.every((nid) => pruned.includes(`branch-${nid}`));
        hiddenCompareModelIds = allBranchesHidden ? [] : pruned;
      }
    }
    runError = null;
  }

  function updateNodeParams(nodeId: string, params: Record<string, unknown>) {
    const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
    const prevNode = prevSnapshot.nodes.find((n) => n.id === nodeId);
    const paramsChanged = !prevNode || !shallowParamsEqual(prevNode.params, params);
    const displayOnlyCompare = isCompareDisplayOnlyParamChange(prevNode, params);
    if (paramsChanged) {
      clearStaleUrlHash();
      sharePopoverState = null;
      const changedKeys = computeChangedParamKeys(prevNode?.params, params);
      if (changedKeys.length > 0) {
        const now = Date.now();
        const paramSignature = changedKeys.join(',');
        const action = classifyParamUndoAction(
          nodeId,
          paramSignature,
          now,
          lastParamUndo,
          lastLabelUndo,
        );
        if (action === 'coalesce' && lastParamUndo) {
          lastParamUndo.at = now;
          graphRedoStack = [];
        } else if (action === 'push') {
          pushGraphUndo(prevSnapshot, selectedPresetId, { preserveCoalescing: true });
          lastParamUndo = { nodeId, paramKey: paramSignature, at: now };
        }
      }
    }
    const rawNextGraph: WorkflowGraph = {
      ...prevSnapshot,
      nodes: prevSnapshot.nodes.map((n) => (n.id === nodeId ? { ...n, params } : n)),
    };
    const { graph: nextGraph, changedNodeIds } = reconcileBenchmarkSeeding(rawNextGraph);
    graph = nextGraph;
    const invalidatedSeeds = new Set<string>();
    if (paramsChanged && !displayOnlyCompare) {
      invalidatedSeeds.add(nodeId);
    }
    for (const cid of changedNodeIds) {
      invalidatedSeeds.add(cid);
    }
    if (invalidatedSeeds.size > 0) {
      scheduler.invalidateEditedGraph(prevSnapshot, nextGraph);
      invalidateNodes(getDownstreamNodeIds(nextGraph, invalidatedSeeds));
    }
  }

  function updateNodeLabel(nodeId: string, label: string) {
    const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
    const prevNode = prevSnapshot.nodes.find((n) => n.id === nodeId);
    if (prevNode?.label === label) return;
    clearStaleUrlHash();
    sharePopoverState = null;
    const now = Date.now();
    if (lastLabelUndo && lastLabelUndo.nodeId === nodeId && now - lastLabelUndo.at < 800) {
      lastLabelUndo.at = now;
      graphRedoStack = [];
    } else {
      pushGraphUndo(prevSnapshot, selectedPresetId, { preserveCoalescing: true });
      lastLabelUndo = { nodeId, at: now };
    }
    graph = {
      ...prevSnapshot,
      nodes: prevSnapshot.nodes.map((n) => (n.id === nodeId ? { ...n, label } : n)),
    };
  }

  let graphValidationErrors = $derived(validateGraph(graph));
  let inputPreviewErrorsByNode = $derived.by((): Record<string, string> => {
    const errs: Record<string, string> = {};
    for (const node of graph.nodes) {
      if (node.adapterId !== 'pdb-input' && node.adapterId !== 'sequence-input') continue;
      const st = nodeStates[node.id];
      if (st?.status === 'error' && st.error) {
        const label = node.label || node.adapterId;
        const coveredByGraph = graphValidationErrors.some(
          (msg) => msg.startsWith(`"${label}":`) && msg.includes(st.error!),
        );
        if (!coveredByGraph) {
          errs[node.id] = `${label}: ${st.error}`;
        }
      }
    }
    return errs;
  });
  let inputPreviewErrors = $derived(Object.values(inputPreviewErrorsByNode));
  let validationErrors = $derived([...graphValidationErrors, ...inputPreviewErrors]);
  let bioFoldLimitOpts = $derived.by(() =>
    buildFoldLimitOpts(computeBackend, remoteHealth),
  );
  let previewChecks = $derived(
    computePreviewChecks(
      graph,
      activeDraftPreview,
      outputsByNode,
      inputPreviewErrorsByNode,
      bioFoldLimitOpts,
    ),
  );
  let displayedValidationErrors = $derived(previewChecks.displayedValidationErrors);
  let bioIssues = $derived(previewChecks.bioIssues);
  let bioErrorIssues = $derived(bioIssues.filter((i) => i.severity === 'error'));
  let hasBioErrors = $derived(bioErrorIssues.length > 0);

  // Instant CPU/fetch preview for starting inputs (PDB Input & Sequence Input) so users can
  // inspect structures in 3D or compare PDBs immediately without needing to click Run.
  const previewedInputSigs = new Map<string, string>();
  const inFlightInputSigs = new Map<string, string>();
  const failedInputSigs = new Map<string, string>();

  function computeInputPreviewSig(node: { adapterId: string; params: Record<string, unknown> }): string | null {
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
        return `pdb-id|${normId}|${hasCached ? txt.length : 0}|${String(p.benchmarkId ?? '')}|${crop}`;
      }
      if (mode === 'custom') {
        const txt = String(p.pdbText ?? '').trim();
        if (txt.length === 0) return null;
        return `custom|${txt.length}|${txt.slice(0, 64)}|${String(p.filename ?? '')}|${String(p.benchmarkId ?? '')}|${crop}`;
      }
      return `demo-bundle|${String(p.benchmarkId ?? '')}|${crop}`;
    }
    if (node.adapterId === 'sequence-input') {
      const raw = String(p.sequence ?? '').trim();
      if (raw.length < 5) return null;
      return `seq|${raw.length}|${raw.slice(0, 64)}|${String(p.label ?? '')}`;
    }
    return null;
  }

  function getSiblingGeneratorIds(nodeId: string): string[] {
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

  function handleSyncTaskToSiblings(sourceNodeId: string) {
    const src = graph.nodes.find((n) => n.id === sourceNodeId);
    if (!src) return;
    const siblingIds = new Set(getSiblingGeneratorIds(sourceNodeId));
    if (siblingIds.size === 0) return;

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
    const syncLen = Number(sp.length ?? parsedLen ?? defaultSrcLen);
    const srcMotif = String(sp.motifSpan ?? sp.motifSelection ?? sp.motifRange ?? '').trim();
    const srcHotspots = String(sp.hotspots ?? '').trim();
    const srcContig = String(sp.contig ?? '').trim();
    const srcLockChain = String(sp.lockChain ?? '').trim();
    const unsupportedSiblingLabels: string[] = [];

    const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
    pushGraphUndo(prevSnapshot);
    clearStaleUrlHash();
    sharePopoverState = null;
    const nextGraph: WorkflowGraph = {
      ...prevSnapshot,
      nodes: prevSnapshot.nodes.map((n) => {
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
          const isSupported = srcMode === 'binder' || srcMode === 'motif' || srcMode === 'unconditional';
          if (!isSupported) {
            unsupportedSiblingLabels.push(n.label ?? 'Genie 3');
            return {
              ...n,
              params: {
                ...n.params,
                taskMode: 'unconditional',
                length: syncLen,
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
                length: syncLen,
                contig: '',
                lockChain: '',
                hotspots: '',
                motifSpan: '',
                userEditedTask: true,
              },
            };
          }
          return { ...n, params: { ...n.params, ...shared, taskMode: srcMode } };
        }
        if (n.adapterId === 'boltzgen') {
          const clampedLen = Math.max(16, Math.min(160, Math.round(syncLen)));
          const isSupported = srcMode === 'binder' || srcMode === 'motif' || srcMode === 'unconditional';
          if (!isSupported) {
            unsupportedSiblingLabels.push(n.label ?? 'BoltzGen-1');
            return {
              ...n,
              params: { ...n.params, taskMode: 'unconditional', length: clampedLen, hotspots: '', userEditedTask: true },
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
              params: { ...n.params, taskMode: 'unconditional', length: clampedLen, motifSelection: '', userEditedTask: true },
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
              params: { ...n.params, taskMode: 'unconditional', length: clampedLen, contig: '', userEditedTask: true },
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
    graph = nextGraph;
    scheduler.invalidateEditedGraph(prevSnapshot, nextGraph);
    invalidateNodes(getDownstreamNodeIds(nextGraph, siblingIds));
    if (unsupportedSiblingLabels.length > 0) {
      showToastMessage(
        `${unsupportedSiblingLabels.join(', ')} does not support "${srcMode}" mode — reset to unconditional (${syncLen} aa) and cleared incompatible constraints.`,
        4500,
      );
    } else {
      showToastMessage(`Synced task settings to ${siblingIds.size} sibling generator${siblingIds.size > 1 ? 's' : ''}.`, 2600);
    }
  }

  $effect(() => {
    if (isRunning) return;
    const nodes = graph.nodes;
    const activeIds = new Set(nodes.map((n) => n.id));
    for (const id of [...previewedInputSigs.keys()]) {
      if (!activeIds.has(id)) {
        previewedInputSigs.delete(id);
        inFlightInputSigs.delete(id);
        failedInputSigs.delete(id);
      }
    }
    for (const id of [...failedInputSigs.keys()]) {
      if (!activeIds.has(id)) {
        failedInputSigs.delete(id);
      }
    }

    for (const node of nodes) {
      if (node.adapterId !== 'pdb-input' && node.adapterId !== 'sequence-input') continue;
      const sig = computeInputPreviewSig(node);
      if (!sig) {
        const isCustomEmptyPdb =
          node.adapterId === 'pdb-input' &&
          String(node.params.mode ?? 'demo-bundle') === 'custom' &&
          !String(node.params.pdbText ?? '').trim();
        const hadTracked =
          previewedInputSigs.has(node.id) ||
          inFlightInputSigs.has(node.id) ||
          failedInputSigs.has(node.id) ||
          Boolean(outputsByNode[node.id]) ||
          (!isCustomEmptyPdb && Boolean(nodeStates[node.id]));
        previewedInputSigs.delete(node.id);
        inFlightInputSigs.delete(node.id);
        failedInputSigs.delete(node.id);
        if (hadTracked) {
          invalidateNodes(
            getDownstreamNodeIds(graph, [node.id]),
            isCustomEmptyPdb ? { keepStateIds: new Set([node.id]) } : undefined,
          );
        }
        if (isCustomEmptyPdb) {
          const errMsg = Boolean(node.params.missingUpload)
            ? 'Missing uploaded PDB/mmCIF file.'
            : 'Paste a PDB / mmCIF structure or upload a file.';
          const curSt = nodeStates[node.id];
          if (curSt?.status !== 'error' || curSt.error !== errMsg) {
            nodeStates = {
              ...nodeStates,
              [node.id]: { nodeId: node.id, status: 'error', error: errMsg },
            };
          }
        }
        continue;
      }
      const hasOut = Boolean(outputsByNode[node.id] && Object.keys(outputsByNode[node.id]).length > 0);
      if (inFlightInputSigs.get(node.id) === sig) continue;
      if (failedInputSigs.get(node.id) === sig) continue;
      if (previewedInputSigs.get(node.id) === sig && hasOut) continue;

      const prevSig = previewedInputSigs.get(node.id);
      previewedInputSigs.set(node.id, sig);
      failedInputSigs.delete(node.id);
      const nodeSnap = $state.snapshot(node);

      if (node.adapterId === 'sequence-input') {
        try {
          const seqSet = materializeSequenceInputNode(nodeSnap);
          if (prevSig && prevSig !== sig) {
            invalidateNodes(getDownstreamNodeIds(graph, [node.id]), { keepSeedIds: new Set([node.id]) });
          }
          outputsByNode = { ...outputsByNode, [node.id]: { sequences: seqSet } };
          nodeStates = {
            ...nodeStates,
            [node.id]: { nodeId: node.id, status: 'done', detail: `${seqSet.candidates.length} sequence(s)` },
          };
        } catch {
          failedInputSigs.set(node.id, sig);
          invalidateNodes(getDownstreamNodeIds(graph, [node.id]));
        }
        continue;
      }

      // pdb-input (sync parse for demo/custom/cached, async RCSB fetch for pdb-id)
      inFlightInputSigs.set(node.id, sig);
      const mode = String(nodeSnap.params.mode ?? 'demo-bundle');
      const needsFetch =
        mode === 'pdb-id' &&
        !(
          String(nodeSnap.params.pdbText ?? '').trim().length > 0 &&
          normalizePdbId(String(nodeSnap.params.fetchedPdbId ?? '')) ===
            normalizePdbId(String(nodeSnap.params.pdbId ?? ''))
        );
      if (needsFetch) {
        const normId = normalizePdbId(String(nodeSnap.params.pdbId ?? ''));
        nodeStates = {
          ...nodeStates,
          [node.id]: {
            nodeId: node.id,
            status: 'loading',
            phase: 'fetch',
            progress: 0.25,
            detail: `Fetching ${normId} from RCSB PDB...`,
          },
        };
      }

      void materializePdbInputNode(nodeSnap)
        .then((structure) => {
          if (inFlightInputSigs.get(node.id) === sig) inFlightInputSigs.delete(node.id);
          if (previewedInputSigs.get(node.id) !== sig) return;
          if (!graph.nodes.some((n) => n.id === node.id)) return;
          failedInputSigs.delete(node.id);
          if (prevSig && prevSig !== sig) {
            invalidateNodes(getDownstreamNodeIds(graph, [node.id]), { keepSeedIds: new Set([node.id]) });
          }
          outputsByNode = { ...outputsByNode, [node.id]: { structure } };
          nodeStates = {
            ...nodeStates,
            [node.id]: {
              nodeId: node.id,
              status: 'done',
              detail: `L=${structure.length} aa`,
            },
          };
        })
        .catch((err) => {
          if (inFlightInputSigs.get(node.id) === sig) inFlightInputSigs.delete(node.id);
          if (previewedInputSigs.get(node.id) !== sig) return;
          if (!graph.nodes.some((n) => n.id === node.id)) return;
          failedInputSigs.set(node.id, sig);
          invalidateNodes(getDownstreamNodeIds(graph, [node.id]), { keepStateIds: new Set([node.id]) });
          nodeStates = {
            ...nodeStates,
            [node.id]: {
              nodeId: node.id,
              status: 'error',
              error: err instanceof Error ? err.message : String(err),
            },
          };
        });
    }
  });

  function applyBioCompatibilityFix(fix: NonNullable<BioCompatibilityIssue['fix']>) {
    if (isRunning) return;
    const { nodeId, paramsPatch, nodeLabel } = fix;
    const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
    pushGraphUndo(prevSnapshot);
    clearStaleUrlHash();
    sharePopoverState = null;
    const rawNextGraph: WorkflowGraph = {
      ...prevSnapshot,
      nodes: prevSnapshot.nodes.map((n) =>
        n.id === nodeId
          ? {
              ...n,
              params: { ...n.params, ...paramsPatch },
              ...(nodeLabel ? { label: nodeLabel } : {}),
            }
          : n,
      ),
    };
    const { graph: nextGraph, changedNodeIds } = reconcileBenchmarkSeeding(rawNextGraph);
    graph = nextGraph;
    scheduler.invalidateEditedGraph(prevSnapshot, nextGraph);
    invalidateNodes(getDownstreamNodeIds(nextGraph, new Set([nodeId, ...changedNodeIds])));
    runError = null;
  }

  function applyBioFix(issue: BioCompatibilityIssue) {
    if (!issue.fix) return;
    applyBioCompatibilityFix(issue.fix);
  }

  function openStudioForNode(nodeId: string) {
    setSingleSelection(nodeId);
    selectedPeekSlotId = null;
    selectedEdgeId = null;
    studioTab = 'input';
    studioHoveredToken = null;
    studioFocusToken = null;
    studioActiveBrush = null;
    expandedPane = 'studio';
  }

  function confirmPendingRun(rememberChoice: boolean) {
    const req = pendingRun;
    pendingRun = null;
    if (rememberChoice) {
      firstRunAck = true;
      try {
        localStorage.setItem('denovo-first-run-ack', '1');
      } catch {
        /* ignore */
      }
    }
    if (req) {
      void runWorkflow(req.bypassCache, req.upToNodeId, true);
    }
  }

  function cancelPendingRun() {
    pendingRun = null;
  }

  function handleCancelRun() {
    if (scheduler.cancelActiveRun('Run cancelled from UI')) {
      showToastMessage('Cancelling active run…', 2400);
    }
  }

  async function runWorkflow(bypassCache = false, upToNodeId?: string, confirmed = false) {
    if (isRunning) return;
    if (hasAnyInvalidNumericInput()) {
      runError = 'Fix invalid numeric parameter(s) before running.';
      return;
    }
    moreMenuOpen = false;
    const plainGraph = $state.snapshot(graph) as WorkflowGraph;
    const targetGraph = upToNodeId ? extractSubgraphUpTo(plainGraph, upToNodeId) : plainGraph;
    const errs = validateGraph(targetGraph);
    if (errs.length > 0) {
      runError = errs.join('\n');
      return;
    }
    const targetBioErrors = validateBioCapabilities(targetGraph, outputsByNode).filter(
      (i) => i.severity === 'error',
    );
    if (targetBioErrors.length > 0) {
      runError = targetBioErrors.map((i) => i.message).join('\n');
      return;
    }
    if (!confirmed) {
      const preflight = estimateRunRequirements(targetGraph);
      const hasRemoteMsa = (preflight.remoteMsaTransfers?.length ?? 0) > 0;
      if (hasRemoteMsa || (!firstRunAck && (preflight.usesWebGpu || preflight.totalDownloadMB > 0))) {
        pendingRun = { bypassCache, upToNodeId, preflight };
        return;
      }
    }
    localRunning = true;
    appliedDraftNotice = null;
    runTargetIds = targetGraph.nodes.map((n) => n.id);
    runError = null;
    hashError = null;
    clearPendingProgress();
    if (!upToNodeId) {
      totalElapsedMs = null;
      nodeStates = {};
      outputsByNode = {};
    } else {
      const targetIds = new Set(targetGraph.nodes.map((n) => n.id));
      const downstreamOfTargets = getDownstreamNodeIds(plainGraph, targetIds);
      invalidateNodes(downstreamOfTargets);
    }

    try {
      const res = await scheduler.run(targetGraph, handleProgress, {
        source: 'canvas',
        bypassCache,
        onNodeOutput: (nodeId, outs) => {
          outputsByNode = { ...outputsByNode, [nodeId]: outs };
        },
      });
      flushPendingProgress();
      let needsOutputMerge = false;
      for (const [nid, outs] of Object.entries(res.outputsByNode)) {
        if (outputsByNode[nid] !== outs) {
          needsOutputMerge = true;
          break;
        }
      }
      if (needsOutputMerge) {
        outputsByNode = { ...outputsByNode, ...res.outputsByNode };
      }
      for (const n of targetGraph.nodes) {
        if (n.adapterId === 'pdb-input' || n.adapterId === 'sequence-input') {
          const sig = computeInputPreviewSig(n);
          if (sig && res.outputsByNode[n.id]) {
            previewedInputSigs.set(n.id, sig);
            failedInputSigs.delete(n.id);
          }
        }
      }
      if (!upToNodeId) {
        totalElapsedMs = res.totalElapsedMs;
      }
      selectedRowIdx = 0;
      selectedFoldIdx = 0;
      showTrajectory = false;
      activeGpuWorker = scheduler.getActiveGpuAdapter();

      if (upToNodeId) {
        setSingleSelection(upToNodeId);
        selectedPeekSlotId = null;
        selectedEdgeId = null;
      } else {
        // Auto-focus the terminal output block (3D Compare -> multi-scorer -> multi-fold -> scorer -> fold -> generator -> last executed node)
        const lastCompare = [...graph.nodes]
          .reverse()
          .find((n) => n.adapterId === 'structure-compare' && outputsByNode[n.id]);
        const scorersWithData = graph.nodes.filter(
          (n) => n.adapterId === 'designability-scorer' && outputsByNode[n.id],
        );
        const foldsWithData = graph.nodes.filter(
          (n) => n.adapterId === 'localfold-af2' && outputsByNode[n.id],
        );
        const lastScorer = [...graph.nodes].reverse().find((n) => n.adapterId === 'designability-scorer');
        const lastFold = [...graph.nodes].reverse().find((n) => n.adapterId === 'localfold-af2');
        const last3DGenerator = [...graph.nodes]
          .reverse()
          .find(
            (n) =>
              (n.adapterId === 'genie3' || n.adapterId === 'rfd4-proteina') &&
              outputsByNode[n.id]?.structure,
          );
        const lastWithOutput = [...graph.nodes]
          .reverse()
          .find((n) => outputsByNode[n.id] && Object.keys(outputsByNode[n.id]).length > 0);
        if (lastCompare) {
          setSingleSelection(lastCompare.id);
          selectedPeekSlotId = null;
          selectedEdgeId = null;
        } else if (scorersWithData.length >= 2) {
          // Forked benchmark (e.g. AF2 vs RoseTTAFold3): auto-open multi-branch 3D comparison!
          handleSelectNodes(scorersWithData.slice(0, COMPARE_PALETTE.length).map((n) => n.id));
        } else if (foldsWithData.length >= 2 && scorersWithData.length === 0) {
          // Parallel pure-folding benchmark (e.g. AF3 vs AF2 vs ESMFold2): auto-open 3D Compare!
          handleSelectNodes(foldsWithData.slice(0, COMPARE_PALETTE.length).map((n) => n.id));
        } else if (lastScorer) {
          setSingleSelection(lastScorer.id);
          selectedPeekSlotId = null;
          selectedEdgeId = null;
        } else if (lastFold) {
          setSingleSelection(lastFold.id);
          selectedPeekSlotId = null;
          selectedEdgeId = null;
        } else if (last3DGenerator) {
          setSingleSelection(last3DGenerator.id);
          selectedPeekSlotId = null;
          selectedEdgeId = null;
        } else if (lastWithOutput) {
          setSingleSelection(lastWithOutput.id);
          selectedPeekSlotId = null;
          selectedEdgeId = null;
        }
      }
      rightPanelTab = 'peek';
      if (activePaperInsight && !upToNodeId) {
        const firstResultIdx = activePaperInsight.walkthrough.findIndex(
          (st) => st.phase === 'result',
        );
        showPaperWalkthrough = true;
        applyPaperWalkthroughStep(activePaperInsight, firstResultIdx >= 0 ? firstResultIdx : 0);
      }

      void refreshCacheStats();
    } catch (err) {
      flushPendingProgress();
      if (err instanceof RunCancellationError) {
        if (Object.keys(err.outputsByNode).length > 0) {
          outputsByNode = { ...outputsByNode, ...err.outputsByNode };
        }
        const cleanedStates: Record<string, NodeProgressEvent> = {};
        for (const [nid, st] of Object.entries(nodeStates)) {
          if (st.status === 'running' || st.status === 'loading') {
            cleanedStates[nid] = {
              ...st,
              status: 'idle',
              progress: undefined,
              detail: 'Cancelled',
            };
          } else {
            cleanedStates[nid] = st;
          }
        }
        nodeStates = cleanedStates;
        runError = null;
        showToastMessage('Run cancelled — completed steps preserved.', 3200);
      } else {
        runError = err instanceof Error ? err.message : String(err);
      }
    } finally {
      flushPendingProgress();
      localRunning = false;
      runTargetIds = [];
      activeGpuWorker = scheduler.getActiveGpuAdapter();
    }
  }

  async function freeVram() {
    if (isRunning || activeRunState.leaseHeld) return;
    const releasedLabel = activeGpuWorkerLabel ?? 'GPU worker';
    try {
      await scheduler.releaseActiveGpu();
      activeGpuWorker = null;
      showToastMessage(`Released GPU memory (${releasedLabel})`, 2600, 'ok');
    } catch (err) {
      showToastMessage(
        `Failed to release GPU memory: ${err instanceof Error ? err.message : String(err)}`,
        4200,
        'error',
      );
    }
  }

  async function handleClearCache() {
    if (isRunning || activeRunState.leaseHeld) return;
    cancelCacheClearConfirm();
    try {
      await scheduler.releaseActiveGpu();
      scheduler.clearMemo();
      clearRcsbCache();
      await clearAllModelCaches({ preserveCheckpoints: true });
      await refreshCacheStats();
      activeGpuWorker = null;
      showToastMessage('Cleared cached model weights and structures', 2800, 'ok');
    } catch (err) {
      showToastMessage(
        `Failed to clear cached weights: ${err instanceof Error ? err.message : String(err)}`,
        4200,
        'error',
      );
    }
  }

  async function openSharePopover() {
    sysMenuOpen = false;
    moreMenuOpen = false;
    if (shareMenuOpen) {
      shareMenuOpen = false;
      return;
    }
    shareMenuOpen = true;
    sharePopoverState = null;
    try {
      const plain = $state.snapshot(graph) as WorkflowGraph;
      const { hash, mode, strippedPdb } = await encodeGraphToHash(plain);
      const url = `${window.location.origin}${window.location.pathname}${hash}`;
      window.history.replaceState(null, '', hash);
      sharePopoverState = { url, mode, strippedPdb, copied: false, error: null };
    } catch (err) {
      sharePopoverState = {
        url: '',
        mode: '',
        strippedPdb: false,
        copied: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  async function copySharePopoverUrl() {
    if (!sharePopoverState?.url) return;
    const { url, mode, strippedPdb } = sharePopoverState;
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error('Clipboard API unavailable');
      }
      await navigator.clipboard.writeText(url);
      sharePopoverState = { ...sharePopoverState, copied: true };
      showToastMessage(
        strippedPdb
          ? `Copied ${url.length}-char link (${mode}; large PDB omitted)`
          : `Copied ${url.length}-char link (${mode})`,
      );
    } catch {
      sharePopoverState = {
        ...sharePopoverState,
        copied: false,
        error: 'Clipboard access was blocked — copy the link directly from the field below.',
      };
      showToastMessage(
        strippedPdb
          ? `Clipboard blocked — copy ${mode} link from URL bar (large PDB omitted)`
          : `Clipboard blocked — copy ${mode} link from URL bar`,
        4200,
        'error',
      );
    }
  }

  async function copyQuickShareLink() {
    await openSharePopover();
  }

  function buildBugReportUrl(errorText?: string | null): string {
    const blocks = graph.nodes.map((n) => `${n.label} (${n.adapterId})`).join(' -> ');
    const title = errorText
      ? `Bug: ${errorText.split('\n')[0].slice(0, 72)}`
      : `Bug report: ${graph.name || 'Workflow'}`;
    const lines = [
      '### Summary',
      errorText ? `Hit an error while running the workflow:` : 'Describe what happened and what you expected:',
      '',
      ...(errorText ? ['```', errorText.slice(0, 900), '```', ''] : []),
      '### Environment',
      `- **Workflow blocks**: ${blocks || 'None'}`,
      `- **WebGPU**: ${webgpuDetail || 'unknown'}`,
      `- **Browser**: ${typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown'}`,
      `- **Page**: ${typeof window !== 'undefined' ? window.location.origin : 'https://denovo.run'}`,
    ];
    const params = new URLSearchParams({
      title,
      body: lines.join('\n').slice(0, 1800),
    });
    return `https://github.com/ramithuh/ProteinDesignSpace/issues/new?${params.toString()}`;
  }

  function loadFullWorkflowGraph(
    next: WorkflowGraph,
    opts?: { clearOutputs?: boolean; keepPaperInsight?: boolean; preserveHash?: boolean },
  ) {
    if (!opts?.preserveHash) {
      clearStaleUrlHash();
    }
    cancelBrandClearConfirm();
    sharePopoverState = null;
    appliedDraftNotice = null;
    if (!opts?.keepPaperInsight) {
      activePaperInsight = null;
      showPaperWalkthrough = false;
    }
    if (expandedPane === 'studio') {
      expandedPane = 'none';
      studioHoveredToken = null;
      studioFocusToken = null;
      studioActiveBrush = null;
    }
    const withDefaults = mergeAdapterDefaults(next);
    const { graph: reconciled } = reconcileBenchmarkSeeding(withDefaults, { teardown: false });
    const laid = reconciled.nodes.some((n) => !n.position)
      ? autoLayoutGraph(reconciled)
      : reconciled;
    graphUndoStack = [];
    graphRedoStack = [];
    lastParamUndo = null;
    lastLabelUndo = null;
    hashError = null;
    selectedPresetId = laid.id;
    graph = laid;
    setSingleSelection(laid.nodes[0]?.id ?? '');
    branchCandidateIdx = {};
    selectedPeekSlotId = null;
    selectedEdgeId = null;
    showSeqParentBackbone = false;
    rightPanelTab = 'params';
    if (opts?.clearOutputs !== false) {
      clearPendingProgress();
      nodeStates = {};
      outputsByNode = {};
      previewedInputSigs.clear();
      inFlightInputSigs.clear();
      failedInputSigs.clear();
      runError = null;
      totalElapsedMs = null;
      selectedRowIdx = 0;
      selectedFoldIdx = 0;
      showTrajectory = false;
      lastPickedResidue = null;
    }
  }

  function exportWorkflowJson() {
    moreMenuOpen = false;
    const plain = $state.snapshot(graph) as WorkflowGraph;
    const blob = new Blob([serializeWorkflowJson(plain)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${graph.id.replace(/[/:\\*?<>|]+/g, '_')}.denovo.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function importGraphJson(ev: Event) {
    const input = ev.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    moreMenuOpen = false;
    if (!file || isRunning) {
      input.value = '';
      return;
    }
    try {
      const text = await file.text();
      const normalized = parseRecipeToGraph(text);
      loadFullWorkflowGraph(normalized, { clearOutputs: true });
      showToastMessage(`Imported ${file.name}`);
    } catch (err) {
      runError = err instanceof Error ? err.message : 'Failed to parse workflow JSON / YAML';
    } finally {
      input.value = '';
    }
  }

  function downloadText(text: string, filename: string) {
    const safeName = filename.replace(/[/:\\*?<>|]+/g, '_');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = safeName;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function toggleTokenInList(prevRaw: unknown, tag: string): { next: string; added: boolean } {
    const prev = String(prevRaw ?? '').trim();
    const parts = prev ? prev.split(/[,\s]+/).filter(Boolean) : [];
    const idx = parts.findIndex((p) => p.toUpperCase() === tag.toUpperCase());
    if (idx >= 0) {
      parts.splice(idx, 1);
      return { next: parts.join(', '), added: false };
    }
    parts.push(tag);
    return { next: parts.join(', '), added: true };
  }

  function toggleTokenInTiedList(prevRaw: unknown, tag: string): { next: string; added: boolean } {
    const prev = String(prevRaw ?? '').trim();
    const groups = prev
      ? prev
          .split(';')
          .map((g) => g.trim())
          .filter(Boolean)
          .map((g) => g.split(/[=,\s]+/).map((t) => t.trim()).filter(Boolean))
      : [];
    const upper = tag.toUpperCase();
    let existed = false;
    const pruned = groups
      .map((g) => {
        const filtered = g.filter((t) => {
          if (t.toUpperCase() === upper) {
            existed = true;
            return false;
          }
          return true;
        });
        return filtered;
      })
      .filter((g) => g.length > 0);
    if (existed) {
      return { next: pruned.map((g) => g.join('=')).join('; '), added: false };
    }
    if (pruned.length === 0) {
      pruned.push([tag]);
    } else {
      pruned[pruned.length - 1].push(tag);
    }
    return { next: pruned.map((g) => g.join('=')).join('; '), added: true };
  }

  /**
   * Isolated 3D residue click router for Node Studio's `'input'` tab (`UX-09`):
   * strictly routes clicks to the active Studio brush on `selectedNode` (or inspect-only
   * when no Studio brush is active), never mutating unrelated nodes via global `molPickMode`.
   */
  function handleStudioInputResidueClick(info: {
    chain: string;
    seqId: number;
    insCode?: string;
    residueToken?: string;
    compId: string;
    atomName?: string;
    isNonPolymer?: boolean;
    entityType?: 'protein' | 'dna' | 'rna' | 'ligand';
  }) {
    if (!selectedNode) return;
    const tag =
      (info.residueToken && info.residueToken.trim()) ||
      `${info.chain}${info.seqId}${info.insCode ?? ''}`;
    const nodeName =
      selectedNode.label || ADAPTER_REGISTRY[selectedNode.adapterId]?.shortName || selectedNode.adapterId;
    const brush = studioActiveBrush;

    if (info.isNonPolymer || info.entityType === 'ligand') {
      const ligToken = `${info.chain}:${info.compId}`;
      const atomSelector = `${info.chain}/${info.compId}/*/${info.atomName || '*'}`;
      if (brush?.role === 'crop') {
        const { next, added } = toggleTokenInList(selectedNode.params[brush.fieldKey], ligToken);
        updateNodeParams(selectedNode.id, { ...selectedNode.params, [brush.fieldKey]: next });
        lastPickedResidue = `${ligToken} ${added ? 'added to' : 'removed from'} ${brush.label} (${nodeName})`;
        return;
      }
      if (
        selectedNode.adapterId === 'rfd4-proteina' &&
        String(selectedNode.params.hbondMode ?? 'none') !== 'none'
      ) {
        updateNodeParams(selectedNode.id, {
          ...selectedNode.params,
          hbondSelection: atomSelector,
          userEditedTask: true,
        });
        lastPickedResidue = `${atomSelector} → H-Bond target (${nodeName})`;
        showToastMessage(`Set RFD4 H-bond target to ${atomSelector}`, 2800);
        return;
      }
      lastPickedResidue = `${ligToken}${info.atomName ? `.${info.atomName}` : ''} (ligand)`;
      return;
    }

    if (info.entityType === 'dna' || info.entityType === 'rna') {
      lastPickedResidue = `${tag} (${info.compId} · ${info.entityType.toUpperCase()})`;
      return;
    }

    const baseLabel = `${tag} (${info.compId})`;
    if (!brush) {
      lastPickedResidue = baseLabel;
      return;
    }

    const { next, added } =
      brush.role === 'tied'
        ? toggleTokenInTiedList(selectedNode.params[brush.fieldKey], tag)
        : toggleTokenInList(selectedNode.params[brush.fieldKey], tag);
    const nextParams: Record<string, unknown> = {
      ...selectedNode.params,
      [brush.fieldKey]: next,
    };
    if (
      (selectedNode.adapterId === 'rfd4-proteina' || selectedNode.adapterId === 'genie3') &&
      ['contig', 'lockChain', 'hotspots', 'motifSpan'].includes(brush.fieldKey)
    ) {
      nextParams.userEditedTask = true;
    }
    updateNodeParams(selectedNode.id, nextParams);
    lastPickedResidue = `${baseLabel} ${added ? '→' : '✕'} ${brush.label} (${nodeName})`;
  }

  function appendToken(prevRaw: unknown, tag: string): string | null {
    const prev = String(prevRaw ?? '').trim();
    const parts = prev ? prev.split(/[,\s]+/).filter(Boolean) : [];
    if (parts.includes(tag)) return null;
    parts.push(tag);
    return parts.join(', ');
  }

  function isConditionedGenerator(n: { adapterId: string; params: Record<string, unknown> }): boolean {
    if (n.adapterId === 'rfd4-proteina') {
      const m = String(n.params?.mode ?? 'unconditional');
      return m === 'binder' || m === 'motif';
    }
    if (n.adapterId === 'genie3') {
      const tm = String(n.params?.taskMode ?? 'unconditional');
      return tm === 'binder' || tm === 'motif';
    }
    return false;
  }

  // Resolve the explicit target block & field for 3D residue picking (so the toolbar
  // displays exactly which block will be edited and auto mode never mutates unselected branches).
  let pickTargetInfo = $derived.by(
    (): {
      node: WorkflowGraph['nodes'][number];
      nodeName: string;
      fieldKey: 'hotspots' | 'motifSpan' | 'fixedPositions';
      fieldLabel: string;
      switchMode?: 'binder' | 'motif';
    } | null => {
      if (isCompareMode || molPickMode === 'off') return null;

      const labelOf = (n: WorkflowGraph['nodes'][number]) =>
        n.label || ADAPTER_REGISTRY[n.adapterId]?.shortName || n.adapterId;

      const isFixedPosAdapter = (aid: string) =>
        aid === 'web-protein-mpnn' ||
        aid === 'ligandmpnn' ||
        aid === 'lasermpnn' ||
        aid === 'antifold-esmif1';

      if (molPickMode === 'mpnn') {
        const allMpnn = graph.nodes.filter((n) => isFixedPosAdapter(n.adapterId));
        const target =
          selectedNode && isFixedPosAdapter(selectedNode.adapterId)
            ? selectedNode
            : allMpnn.length === 1
              ? allMpnn[0]
              : null;
        if (!target) return null;
        return {
          node: target,
          nodeName: labelOf(target),
          fieldKey: 'fixedPositions',
          fieldLabel: 'Fix Positions',
        };
      }

      if (molPickMode === 'hotspots' || molPickMode === 'motif') {
        const isCondOrGen = (n: WorkflowGraph['nodes'][number]) =>
          n.adapterId === 'rfd4-proteina' || n.adapterId === 'genie3';
        const candidates = graph.nodes.filter(isCondOrGen);
        const conditionedOnly = candidates.filter(isConditionedGenerator);
        const target =
          selectedNode && isCondOrGen(selectedNode)
            ? selectedNode
            : candidates.length === 1
              ? candidates[0]
              : conditionedOnly.length === 1
                ? conditionedOnly[0]
                : null;
        if (!target) return null;
        return {
          node: target,
          nodeName: labelOf(target),
          fieldKey: molPickMode === 'motif' ? 'motifSpan' : 'hotspots',
          fieldLabel: molPickMode === 'motif' ? 'Motif' : 'Hotspots',
          switchMode: molPickMode === 'motif' ? 'motif' : 'binder',
        };
      }

      // Auto mode: ONLY edit the selected block (if MPNN or already-conditioned generator),
      // OR a single unambiguous conditioned generator when viewing an input structure.
      // Never silently flip an unconditional generator to binder, and never mutate an unselected branch.
      if (selectedNode && isFixedPosAdapter(selectedNode.adapterId)) {
        return {
          node: selectedNode,
          nodeName: labelOf(selectedNode),
          fieldKey: 'fixedPositions',
          fieldLabel: 'Fix Positions',
        };
      }
      if (selectedNode?.adapterId === 'rfd4-proteina' || selectedNode?.adapterId === 'genie3') {
        if (!isConditionedGenerator(selectedNode)) return null;
        const m = String(
          selectedNode.adapterId === 'genie3'
            ? (selectedNode.params.taskMode ?? 'unconditional')
            : (selectedNode.params.mode ?? 'unconditional'),
        );
        const key = m === 'motif' ? 'motifSpan' : 'hotspots';
        return {
          node: selectedNode,
          nodeName: labelOf(selectedNode),
          fieldKey: key,
          fieldLabel: key === 'motifSpan' ? 'Motif' : 'Hotspots',
        };
      }

      const conditionedNodes = graph.nodes.filter(isConditionedGenerator);
      if (conditionedNodes.length === 1) {
        const target = conditionedNodes[0];
        const m = String(
          target.adapterId === 'genie3'
            ? (target.params.taskMode ?? 'unconditional')
            : (target.params.mode ?? 'unconditional'),
        );
        const key = m === 'motif' ? 'motifSpan' : 'hotspots';
        return {
          node: target,
          nodeName: labelOf(target),
          fieldKey: key,
          fieldLabel: key === 'motifSpan' ? 'Motif' : 'Hotspots',
        };
      }

      return null;
    },
  );

  function handleResidueClick(info: {
    chain: string;
    seqId: number;
    insCode?: string;
    residueToken?: string;
    compId: string;
    atomName?: string;
    isNonPolymer?: boolean;
    entityType?: 'protein' | 'dna' | 'rna' | 'ligand';
  }) {
    if (isCompareMode) return;
    const tag =
      (info.residueToken && info.residueToken.trim()) ||
      `${info.chain}${info.seqId}${info.insCode ?? ''}`;

    if (info.isNonPolymer || info.entityType === 'ligand') {
      const atomSelector = `${info.chain}/${info.compId}/*/${info.atomName || '*'}`;
      if (
        molPickMode !== 'off' &&
        selectedNode?.adapterId === 'rfd4-proteina' &&
        String(selectedNode.params.hbondMode ?? 'none') !== 'none'
      ) {
        updateNodeParams(selectedNode.id, {
          ...selectedNode.params,
          hbondSelection: atomSelector,
          userEditedTask: true,
        });
        lastPickedResidue = `${atomSelector} → H-Bond target (${selectedNode.label || 'RFD4-Proteina'})`;
        showToastMessage(`Set RFD4 H-bond target to ${atomSelector}`, 2800);
        return;
      }
      lastPickedResidue = `${info.chain}:${info.compId}${info.atomName ? `.${info.atomName}` : ''} (ligand)`;
      showToastMessage(
        `Clicked ligand ${info.compId} (${atomSelector}) — enable RFD4 H-bond conditioning to pick ligand atoms directly, or click protein residues for hotspots/motifs/locks.`,
        3400,
      );
      return;
    }

    if (info.entityType === 'dna' || info.entityType === 'rna') {
      lastPickedResidue = `${tag} (${info.compId} · ${info.entityType.toUpperCase()})`;
      showToastMessage(
        `Clicked ${info.entityType.toUpperCase()} nucleotide ${tag} (${info.compId}) — hotspots, motifs, and MPNN fixed positions target protein residues.`,
        3200,
      );
      return;
    }

    const baseLabel = `${tag} (${info.compId})`;

    if (molPickMode === 'off') {
      lastPickedResidue = baseLabel;
      return;
    }

    const target = pickTargetInfo;
    if (!target) {
      lastPickedResidue = baseLabel;
      if (molPickMode !== 'auto') {
        showToastMessage('Multiple target blocks exist — click a block on the canvas first to pick residues into it.', 3400);
      } else if (
        selectedNode &&
        (selectedNode.adapterId === 'rfd4-proteina' || selectedNode.adapterId === 'genie3')
      ) {
        showToastMessage(
          'Auto pick only edits Binder/Motif tasks — switch Design Task or choose Pick: Hotspots / Motif.',
          3400,
        );
      }
      return;
    }

    const updated = appendToken(target.node.params[target.fieldKey], tag);
    if (updated !== null) {
      const nextParams: Record<string, unknown> = {
        ...target.node.params,
        [target.fieldKey]: updated,
      };
      if (target.node.adapterId === 'genie3') {
        if (target.switchMode) nextParams.taskMode = target.switchMode;
        nextParams.userEditedTask = true;
      } else if (target.node.adapterId === 'rfd4-proteina') {
        if (target.switchMode) {
          nextParams.mode = target.switchMode;
          const prevContig = String(target.node.params.contig ?? '').trim();
          if (target.switchMode === 'binder' && /^\d+(?:-\d+)?$/.test(prevContig)) {
            nextParams.contig = '';
          }
        }
        nextParams.userEditedTask = true;
      }
      updateNodeParams(target.node.id, nextParams);
    }
    lastPickedResidue = `${baseLabel} → ${target.fieldLabel} (${target.nodeName})`;
  }

  // Preset & pick-target helpers
  let activePreset = $derived(WORKFLOW_PRESETS.find((p) => p.id === selectedPresetId) ?? null);
  let hasConditionerNode = $derived(
    graph.nodes.some((n) => n.adapterId === 'rfd4-proteina' || n.adapterId === 'genie3'),
  );
  let hasMpnnNode = $derived(
    graph.nodes.some(
      (n) =>
        n.adapterId === 'web-protein-mpnn' ||
        n.adapterId === 'ligandmpnn' ||
        n.adapterId === 'lasermpnn' ||
        n.adapterId === 'antifold-esmif1',
    ),
  );

  $effect(() => {
    if ((molPickMode === 'hotspots' || molPickMode === 'motif') && !hasConditionerNode) {
      molPickMode = 'auto';
    } else if (molPickMode === 'mpnn' && !hasMpnnNode) {
      molPickMode = 'auto';
    }
  });

  // Node- and port-scoped intermediate peek state (do not fall back to graph.nodes[0] when deselected)
  let selectedNode = $derived(
    selectedNodeId ? (graph.nodes.find((n) => n.id === selectedNodeId) ?? null) : null,
  );
  let selectedUpstreamBenchmarkId = $derived.by((): string | null => {
    if (!selectedNode) return null;
    const edge = graph.edges.find(
      (e) => e.toNode === selectedNode.id && (e.toPort === 'structure' || e.toPort === 'template'),
    );
    if (!edge) return null;
    const src = graph.nodes.find((n) => n.id === edge.fromNode);
    if (src?.adapterId !== 'pdb-input') return null;
    const bid = String(src.params.benchmarkId ?? '').trim();
    return bid || null;
  });

  // Lazy single-source-of-truth compiled input preview (computed only while Node Studio is open)
  let studioOpen = $derived(expandedPane === 'studio' && Boolean(selectedNode));
  let studioPreview = $derived.by(() => {
    if (!studioOpen || !selectedNode) return null;
    return compileNodeInputPreview(graph, selectedNode.id, outputsByNode, bioFoldLimitOpts);
  });
  let studioInputViewerActive = $derived(
    studioOpen && studioTab === 'input' && Boolean(studioPreview?.previewStructurePdb),
  );

  let peekSlots = $derived.by(() =>
    selectedNode ? getNodePeekSlots(graph, selectedNode.id, outputsByNode) : [],
  );

  let hasPeekData = $derived(peekSlots.some((s) => s.artifact !== null));

  let resolvedPeekSlotId = $derived.by(() => {
    if (selectedPeekSlotId && peekSlots.some((s) => s.id === selectedPeekSlotId)) {
      return selectedPeekSlotId;
    }
    return pickDefaultPeekSlot(peekSlots);
  });

  let activePeekSlot = $derived.by(
    () => peekSlots.find((s) => s.id === resolvedPeekSlotId) ?? null,
  );
  let activeArtifact = $derived<CanonicalArtifact | null>(activePeekSlot?.artifact ?? null);
  let contextStructure = $derived.by(() =>
    selectedNode ? findContextStructure(graph, selectedNode.id, outputsByNode) : null,
  );

  // Strictly upstream StructureArtifact for live motif/hotspot resolution and backbone counting
  // (never falls back to selectedNode's own output structure, e.g. AF2's representative fold).
  let selectedUpstreamStructure = $derived.by((): StructureArtifact | null => {
    if (!selectedNode) return null;
    const edge = graph.edges.find(
      (e) => e.toNode === selectedNode.id && (e.toPort === 'structure' || e.toPort === 'template'),
    );
    if (edge) {
      const art = outputsByNode[edge.fromNode]?.[edge.fromPort];
      if (art?.kind === 'StructureArtifact') return art;
    }
    const incoming = graph.edges.filter((e) => e.toNode === selectedNode.id);
    for (const inc of incoming) {
      const upstream = findContextStructure(graph, inc.fromNode, outputsByNode);
      if (upstream) return upstream;
    }
    return null;
  });

  // Directly wired upstream SequenceSet for accurate multi-backbone / sequence fan-out telemetry
  let selectedUpstreamSequences = $derived.by((): SequenceSet | null => {
    if (!selectedNode) return null;
    const edge = graph.edges.find(
      (e) => e.toNode === selectedNode.id && e.toPort === 'sequences',
    );
    if (!edge) return null;
    const art = outputsByNode[edge.fromNode]?.[edge.fromPort];
    return art?.kind === 'SequenceSet' ? art : null;
  });

  // Typed views of the currently peeked modality
  let peekMetrics = $derived<DesignabilityMetrics | null>(
    activeArtifact?.kind === 'DesignabilityMetrics' ? activeArtifact : null,
  );
  let peekFoldSet = $derived<FoldedStructure | null>(
    activeArtifact?.kind === 'FoldedStructure' ? activeArtifact : null,
  );
  let peekStructure = $derived<StructureArtifact | null>(
    activeArtifact?.kind === 'StructureArtifact' ? activeArtifact : null,
  );
  let peekStructureItems = $derived.by((): StructureArtifact[] => {
    if (!peekStructure) return [];
    const items = peekStructure.collection?.items;
    return Array.isArray(items) && items.length > 1 ? items : [];
  });
  let hasMultiDesignStructures = $derived(peekStructureItems.length > 1);

  function designKeyForItem(item: StructureArtifact, idx: number): string {
    return item.designKey ?? item.id ?? `design-${idx + 1}`;
  }

  $effect(() => {
    void peekStructure?.id;
    selectedDesignKey = 'all';
    multiDesignHiddenIds = [];
  });

  let activePeekStructureIdx = $derived.by((): number => {
    if (!hasMultiDesignStructures || selectedDesignKey === 'all') return 0;
    const found = peekStructureItems.findIndex(
      (item, idx) => designKeyForItem(item, idx) === selectedDesignKey,
    );
    return found >= 0 ? found : 0;
  });

  let activePeekStructure = $derived<StructureArtifact | null>(
    hasMultiDesignStructures && selectedDesignKey !== 'all'
      ? (peekStructureItems[activePeekStructureIdx] ?? peekStructure)
      : peekStructure,
  );

  let isMultiDesignAll3d = $derived(
    !isCompareMode &&
      !studioInputViewerActive &&
      hasMultiDesignStructures &&
      selectedDesignKey === 'all' &&
      multiDesignViewMode !== 'single',
  );

  let multiDesignModels = $derived.by(() => {
    if (!isMultiDesignAll3d) return null;
    return buildMultiDesignViewerModels(
      peekStructureItems,
      multiDesignViewMode === 'overlay' ? 'overlay' : 'grid',
      new Set(multiDesignHiddenIds),
    );
  });

  function toggleMultiDesignModelVisibility(modelId: string) {
    if (multiDesignHiddenIds.includes(modelId)) {
      multiDesignHiddenIds = multiDesignHiddenIds.filter((id) => id !== modelId);
    } else {
      multiDesignHiddenIds = [...multiDesignHiddenIds, modelId];
    }
  }

  function soloMultiDesignModel(modelId: string) {
    if (!multiDesignModels) return;
    const allIds = multiDesignModels.map((m) => m.id);
    const others = allIds.filter((id) => id !== modelId);
    const alreadySolo =
      !multiDesignHiddenIds.includes(modelId) &&
      others.every((id) => multiDesignHiddenIds.includes(id));
    multiDesignHiddenIds = alreadySolo ? [] : others;
  }

  function stepMultiDesign(delta: number) {
    if (!hasMultiDesignStructures) return;
    const n = peekStructureItems.length;
    const cur = selectedDesignKey === 'all' ? (delta > 0 ? 0 : n - 1) : (activePeekStructureIdx + delta + n) % n;
    selectedDesignKey = designKeyForItem(peekStructureItems[cur], cur);
    multiDesignViewMode = 'single';
  }
  let peekSequences = $derived<SequenceSet | null>(
    activeArtifact?.kind === 'SequenceSet' ? activeArtifact : null,
  );
  let peekPerResidue = $derived(
    activeArtifact?.kind === 'MutationMatrix' || activeArtifact?.kind === 'ResidueTrack'
      ? activeArtifact
      : null,
  );
  // When peeking a per-residue heatmap/track (e.g. ThermoMPNN mutations or Dyna-1 track),
  // display the node's sibling B-factor-painted StructureArtifact in 3D if present.
  let perResidueDisplayStructure = $derived.by((): StructureArtifact | null => {
    if (!peekPerResidue) return null;
    if (selectedNode) {
      const outs = outputsByNode[selectedNode.id];
      if (outs) {
        const sibling = Object.values(outs).find(
          (a): a is StructureArtifact => a.kind === 'StructureArtifact',
        );
        if (sibling) return sibling;
      }
    }
    return contextStructure;
  });

  // Latest scorer result anywhere in the graph (for "+ Save to Campaign" and "Scorer summary →")
  let latestScorerHit = $derived.by((): { nodeId: string; metrics: DesignabilityMetrics } | null => {
    if (peekMetrics && selectedNode) {
      return { nodeId: selectedNode.id, metrics: peekMetrics };
    }
    for (const n of [...graph.nodes].reverse()) {
      const outs = outputsByNode[n.id];
      if (!outs) continue;
      for (const art of Object.values(outs)) {
        if (art.kind === 'DesignabilityMetrics') {
          return { nodeId: n.id, metrics: art };
        }
      }
    }
    return null;
  });

  let activeRow = $derived<DesignabilityRow | null>(
    peekMetrics?.rows[selectedRowIdx] ?? peekMetrics?.best ?? null,
  );
  let activeFold = $derived(
    peekFoldSet?.folds[selectedFoldIdx] ?? peekFoldSet?.folds[0] ?? null,
  );

  let isPeekingSequenceSlot = $derived(Boolean(peekSequences && rightPanelTab === 'peek'));

  // 3D Mol* viewport bindings derived strictly from the active peek modality
  let viewerDesignedPdb = $derived.by((): string | null => {
    if (peekMetrics) return activeRow?.designedPdb || contextStructure?.cifText || contextStructure?.pdbText || null;
    if (activePeekStructure) return activePeekStructure.cifText || activePeekStructure.pdbText || null;
    if (perResidueDisplayStructure) {
      return perResidueDisplayStructure.cifText || perResidueDisplayStructure.pdbText || null;
    }
    if (peekSequences) {
      // Clear 3D viewport when peeking a sequence-only wire/port unless user explicitly
      // previews the parent backbone (keeps Parameters-tab 3D residue picking intact).
      if (rightPanelTab === 'peek' && !showSeqParentBackbone) return null;
      return contextStructure?.cifText || contextStructure?.pdbText || null;
    }
    // Standalone folds are not Kabsch-aligned to the designed backbone, so keep designedPdb null
    return null;
  });

  let viewerRefoldPdb = $derived.by((): string | null => {
    if (peekMetrics) return activeRow?.alignedFoldPdb ?? null;
    if (peekFoldSet) return activeFold?.pdbText ?? null;
    return null;
  });

  let viewerStructureMeta = $derived<StructureArtifact | null>(
    activePeekStructure ??
      perResidueDisplayStructure ??
      (peekMetrics || (peekSequences && (!isPeekingSequenceSlot || showSeqParentBackbone))
        ? contextStructure
        : null),
  );

  let viewerAtomGranularity = $derived.by((): AtomGranularity | undefined => {
    if (isCompareMode) return undefined;
    if (peekStructure) return peekStructure.atomGranularity;
    if (perResidueDisplayStructure) return perResidueDisplayStructure.atomGranularity;
    if (peekFoldSet) return 'all-atom';
    if (peekMetrics) {
      return showDesigned && !showRefold ? contextStructure?.atomGranularity : undefined;
    }
    if (peekSequences && (!isPeekingSequenceSlot || showSeqParentBackbone)) {
      return contextStructure?.atomGranularity;
    }
    return undefined;
  });

  let viewerBfactorSemantics = $derived.by((): BfactorSemantics | undefined => {
    if (isCompareMode) return undefined;
    if (peekMetrics) {
      if (showRefold) {
        return activeRow?.confidenceAvailable === false ? 'none' : 'plddt-0-100';
      }
      return (
        contextStructure?.bfactorSemantics ??
        (contextStructure?.sourceModelId === 'pdb-input' ? 'crystal-b' : undefined)
      );
    }
    if (peekFoldSet) {
      return activeFold?.confidenceAvailable === false ? 'none' : 'plddt-0-100';
    }
    if (peekStructure) {
      return (
        peekStructure.bfactorSemantics ??
        (peekStructure.sourceModelId === 'pdb-input' ? 'crystal-b' : undefined)
      );
    }
    if (perResidueDisplayStructure) {
      return (
        perResidueDisplayStructure.bfactorSemantics ??
        (perResidueDisplayStructure.sourceModelId === 'pdb-input' ? 'crystal-b' : undefined)
      );
    }
    if (peekSequences && (!isPeekingSequenceSlot || showSeqParentBackbone)) {
      return (
        contextStructure?.bfactorSemantics ??
        (contextStructure?.sourceModelId === 'pdb-input' ? 'crystal-b' : undefined)
      );
    }
    return undefined;
  });

  let viewerEmptyMessage = $derived.by((): string | undefined => {
    if (isPeekingSequenceSlot && peekSequences && !showSeqParentBackbone) {
      const n = peekSequences.candidates.length;
      const len = peekSequences.candidates[0]?.sequence.length ?? 0;
      const seqNote = n > 0 ? `${n} sequence${n === 1 ? '' : 's'}${len > 0 ? ` · ${len} aa` : ''}` : 'sequences';
      return `Sequence-only connection (${seqNote}) — no 3D coordinates flow on this wire.`;
    }
    return undefined;
  });

  let viewerLabel = $derived.by((): string => {
    const dirTag = activePeekSlot
      ? `${activePeekSlot.direction === 'in' ? 'IN' : 'OUT'} · ${activePeekSlot.portName}`
      : '';
    if (peekMetrics && activeRow) {
      const rmsdTxt = Number.isFinite(activeRow.scRmsd) ? `${activeRow.scRmsd.toFixed(2)} Å` : '—';
      const plddtTxt =
        activeRow.confidenceAvailable !== false && Number.isFinite(activeRow.meanPlddt)
          ? activeRow.meanPlddt.toFixed(1)
          : 'N/A';
      return `${selectedNode?.label ?? 'Scorer'} · ${activeRow.sequenceId} · scRMSD ${rmsdTxt} · pLDDT ${plddtTxt}`;
    }
    if (peekFoldSet && activeFold) {
      const plddtTxt =
        activeFold.confidenceAvailable !== false && Number.isFinite(activeFold.meanPlddt)
          ? activeFold.meanPlddt.toFixed(1)
          : 'N/A';
      return `${dirTag} · ${activeFold.sequenceId} · pLDDT ${plddtTxt}`;
    }
    if (activePeekStructure) {
      const fromTag = activePeekSlot?.peerNodeLabel ? ` ← ${activePeekSlot.peerNodeLabel}` : '';
      if (isMultiDesignAll3d) {
        const modeTxt = multiDesignViewMode === 'overlay' ? 'Overlay' : 'Grid';
        return `${dirTag}${fromTag} · All ${peekStructureItems.length} designs (${modeTxt})`;
      }
      if (hasMultiDesignStructures && selectedDesignKey !== 'all') {
        return `${dirTag}${fromTag} · Design ${activePeekStructureIdx + 1}/${peekStructureItems.length} · ${activePeekStructure.label}`;
      }
      return `${dirTag}${fromTag} · ${activePeekStructure.label}`;
    }
    if (peekPerResidue && perResidueDisplayStructure) {
      return `${dirTag} · ${perResidueDisplayStructure.label}`;
    }
    if (peekSequences) {
      if (isPeekingSequenceSlot && !showSeqParentBackbone) {
        return `${dirTag} · sequences only`;
      }
      if (contextStructure) {
        return `Parent backbone (${contextStructure.label}) · peeking ${dirTag}`;
      }
    }
    return '';
  });

  // Active run step counter for Run button ("Running 2/4…")
  let runStepLabel = $derived.by((): string => {
    if (!isRunning) return '▶ Run';
    const total = runTargetIds.length || graph.nodes.length;
    if (total <= 0) return 'Running…';
    let doneCount = 0;
    let activeCount = 0;
    for (const id of runTargetIds) {
      const st = nodeStates[id]?.status;
      if (st === 'done' || st === 'cached') doneCount++;
      else if (st === 'running' || st === 'loading') activeCount++;
    }
    const current = Math.min(total, doneCount + (activeCount > 0 ? 1 : 0));
    return current > 0 ? `Running ${current}/${total}…` : `Running 1/${total}…`;
  });

  // Active scorer thresholds (for dynamic table highlights)
  let activeScorerNode = $derived(
    selectedNode?.adapterId === 'designability-scorer'
      ? selectedNode
      : [...graph.nodes].reverse().find((n) => n.adapterId === 'designability-scorer') ?? null,
  );
  let scRmsdThreshold = $derived(Number(activeScorerNode?.params.scRmsdThreshold ?? 2.0));
  let plddtThreshold = $derived(Number(activeScorerNode?.params.plddtThreshold ?? 70));
  let ipaeThreshold = $derived(Number(activeScorerNode?.params.ipaeThreshold ?? 10.0));
  let motifRmsdThreshold = $derived(Number(activeScorerNode?.params.motifRmsdThreshold ?? 1.0));
  let ipsaeThreshold = $derived(Number(activeScorerNode?.params.ipsaeThreshold ?? 0));

  let thresholdsByNodeId = $derived.by(
    (): Record<string, { scRmsd: number; plddt: number; ipae: number; motifRmsd: number; ipsae: number }> => {
      const map: Record<string, { scRmsd: number; plddt: number; ipae: number; motifRmsd: number; ipsae: number }> = {};
      for (const n of graph.nodes) {
        if (n.adapterId === 'designability-scorer') {
          map[n.id] = {
            scRmsd: Number(n.params.scRmsdThreshold ?? 2.0),
            plddt: Number(n.params.plddtThreshold ?? 70),
            ipae: Number(n.params.ipaeThreshold ?? 10.0),
            motifRmsd: Number(n.params.motifRmsdThreshold ?? 1.0),
            ipsae: Number(n.params.ipsaeThreshold ?? 0),
          };
        }
      }
      return map;
    },
  );

  // Per-branch lineage, MSA, ipSAE, failed-gate & provenance metadata for ComparePeekPanel
  let compareBranchMetaByNodeId = $derived.by((): Record<string, CompareBranchMeta> => {
    if (!isCompareMode) return {};
    const selectedIdxByNode = new Map<string, number>();
    if (comparisonState) {
      for (const br of comparisonState.branches) {
        selectedIdxByNode.set(br.nodeId, br.selectedIndex);
      }
    }
    const map: Record<string, CompareBranchMeta> = {};
    for (const id of effectiveCompareNodeIds) {
      const outs = outputsByNode[id];
      if (!outs) continue;
      const values = Object.values(outs);
      const selIdx = selectedIdxByNode.get(id) ?? branchCandidateIdx[id] ?? 0;
      const metrics = values.find((a): a is DesignabilityMetrics => a.kind === 'DesignabilityMetrics');
      if (metrics) {
        const row = metrics.rows[selIdx] ?? metrics.best ?? metrics.rows[0] ?? null;
        const ctxStruct = findContextStructure(graph, id, outputsByNode);
        map[id] = {
          referenceRelation: row?.referenceRelation ?? metrics.referenceRelation,
          referenceLabel: row?.referenceLabel ?? metrics.referenceLabel,
          lineageWarning: metrics.lineageWarning,
          lineageMismatch: row?.lineageMismatch ?? metrics.lineageMismatch,
          collectionKind: metrics.collectionKind,
          ipsae: row?.ipsae ?? null,
          ipsaeMin: row?.ipsaeMin ?? null,
          failedGates: row?.failedGates,
          msaMode: row?.msaMode,
          msaDepth: row?.msaDepth,
          atomGranularity: ctxStruct?.atomGranularity,
          provenance: ctxStruct?.provenance,
          chainLengths: ctxStruct?.chainLengths ?? ctxStruct?.diagnostics?.chainLengths,
        };
        continue;
      }
      const folds = values.find((a): a is FoldedStructure => a.kind === 'FoldedStructure');
      if (folds) {
        const fold = folds.folds[selIdx] ?? folds.folds[0] ?? null;
        map[id] = {
          representativeSelection: folds.representativeSelection,
          collectionKind: folds.collectionKind,
          ipsae: fold?.ipsae ?? null,
          ipsaeMin: fold?.ipsaeMin ?? null,
          msaMode: fold?.msaMode ?? fold?.provenance?.msaMode,
          msaDepth: fold?.msaDepth ?? fold?.provenance?.msaDepth,
          provenance: fold?.provenance,
          chainLengths: fold?.chainLengths,
          foldModelId: fold?.foldModelId,
        };
        continue;
      }
      const struct = values.find((a): a is StructureArtifact => a.kind === 'StructureArtifact');
      if (struct) {
        map[id] = {
          representativeSelection: struct.representativeSelection,
          atomGranularity: struct.atomGranularity,
          provenance: struct.provenance,
          msaMode: struct.provenance?.msaMode,
          msaDepth: struct.provenance?.msaDepth,
          chainLengths: struct.chainLengths ?? struct.diagnostics?.chainLengths,
        };
        continue;
      }
      const seqSet = values.find((a): a is SequenceSet => a.kind === 'SequenceSet');
      if (seqSet) {
        const cand = seqSet.candidates[selIdx] ?? seqSet.candidates[0] ?? null;
        const ctxStruct = findContextStructure(graph, id, outputsByNode);
        map[id] = {
          collectionKind: seqSet.collectionKind,
          chainLengths: cand?.chainLengths ?? ctxStruct?.chainLengths ?? ctxStruct?.diagnostics?.chainLengths,
        };
      }
    }
    return map;
  });

  // Multi-select & 3D Compare block derived state
  let effectiveCompareLayout = $derived<'overlay' | 'side-by-side'>(
    activeCompareNode
      ? String(activeCompareNode.params.layout ?? 'overlay') === 'side-by-side'
        ? 'side-by-side'
        : 'overlay'
      : compareLayout,
  );
  let effectiveCompareColorMode = $derived<'by-model' | 'plddt'>(
    activeCompareNode
      ? String(activeCompareNode.params.colorMode ?? 'by-model') === 'plddt'
        ? 'plddt'
        : 'by-model'
      : compareColorMode,
  );
  let effectiveCompareAlignKabsch = $derived<boolean>(
    activeCompareNode ? activeCompareNode.params.alignKabsch !== false : compareAlignKabsch,
  );

  function setActiveCompareOption(patch: {
    layout?: 'overlay' | 'side-by-side';
    colorMode?: 'by-model' | 'plddt';
    alignKabsch?: boolean;
  }) {
    if (activeCompareNode) {
      updateNodeParams(activeCompareNode.id, {
        ...$state.snapshot(activeCompareNode.params),
        ...patch,
      });
    } else {
      if (patch.layout !== undefined) compareLayout = patch.layout;
      if (patch.colorMode !== undefined) compareColorMode = patch.colorMode;
      if (patch.alignKabsch !== undefined) compareAlignKabsch = patch.alignKabsch;
    }
  }

  let compareColors = $derived.by((): Record<string, { badge: string; css: string }> => {
    if (!isCompareMode) return {};
    const map: Record<string, { badge: string; css: string }> = {};
    effectiveCompareNodeIds.forEach((id, idx) => {
      const slot = COMPARE_PALETTE[idx];
      map[id] = { badge: slot.badge, css: slot.css };
    });
    return map;
  });

  let selectedUpstreamLockSummary = $derived(
    selectedNode ? computeUpstreamLockSummary(graph, selectedNode.id, outputsByNode) : null,
  );

  let comparisonState = $derived.by(() => {
    if (!isCompareMode || effectiveCompareNodeIds.length === 0) return null;
    return buildComparisonState(graph, effectiveCompareNodeIds, outputsByNode, {
      layout: effectiveCompareLayout,
      colorMode: effectiveCompareColorMode,
      alignKabsch: effectiveCompareAlignKabsch,
      branchCandidateIdx,
      hiddenModelIds: new Set(hiddenCompareModelIds),
      pairingMode: comparePairingMode,
      selectedCandidateKey: compareSelectedCandidateKey,
      referenceNodeId: activeCompareRefNodeId ?? undefined,
    });
  });

  // Discoverable 1-click compare CTA when a graph has >= 2 scorers (or >= 2 folds) with outputs
  let comparableBranchIds = $derived.by((): string[] => {
    const scorers = graph.nodes
      .filter((n) => n.adapterId === 'designability-scorer' && outputsByNode[n.id])
      .map((n) => n.id);
    if (scorers.length >= 2) return scorers.slice(0, COMPARE_PALETTE.length);
    const folds = graph.nodes
      .filter((n) => n.adapterId === 'localfold-af2' && outputsByNode[n.id])
      .map((n) => n.id);
    if (folds.length >= 2) return folds.slice(0, COMPARE_PALETTE.length);
    return [];
  });

  let isCanonicalViewerLayout = $derived(
    Boolean(
      (isCompareMode && comparisonState?.effectiveLayout === 'side-by-side') ||
        (isMultiDesignAll3d && multiDesignViewMode === 'grid'),
    ),
  );

  let compareEmptyMessage = $derived.by((): string | undefined => {
    if (!isCompareMode || !comparisonState || comparisonState.models.length > 0) return undefined;
    const allHaveOutputs = comparisonState.branches.every((b) => b.hasOutputs);
    const anySeq = comparisonState.branches.some((b) => b.kind === 'sequence');
    if (allHaveOutputs && anySeq) {
      return 'Sequence-only comparison — no 3D coordinates emitted by these blocks (see scores, recovery, AA composition & sequence diff in the Compare table).';
    }
    if (anySeq) {
      return 'Sequence-only comparison — run any un-run blocks to compare scores & sequences.';
    }
    return undefined;
  });

  let compareViewerLabel = $derived.by((): string => {
    if (!comparisonState) return viewerLabel;
    if (
      comparisonState.models.length === 0 &&
      comparisonState.branches.some((b) => b.kind === 'sequence')
    ) {
      return `Compare ${comparisonState.branches.length} blocks · Sequence comparison`;
    }
    const alignTag = comparisonState.kabschAligned
      ? comparisonState.pairwise?.isSubstructure
        ? `✓ Motif Kabsch (${comparisonState.pairwise.alignedResidues} aa)`
        : '✓ Kabsch'
      : comparisonState.effectiveLayout === 'side-by-side'
        ? 'Side-by-side'
        : 'Overlay (raw)';
    if (comparisonState.pairwise) {
      return `Compare ${comparisonState.branches.length} blocks · ${alignTag} · ${comparisonState.pairwise.badgeA}↔${comparisonState.pairwise.badgeB} RMSD ${comparisonState.pairwise.rmsd.toFixed(2)} Å`;
    }
    return `Compare ${comparisonState.branches.length} blocks · ${alignTag}`;
  });

  // Reset row index when switching between different scorer branches
  $effect(() => {
    void peekMetrics?.id;
    selectedRowIdx = 0;
  });

  // Reset fold index when switching between different fold sets
  $effect(() => {
    void peekFoldSet?.id;
    selectedFoldIdx = 0;
  });

  // Reset trajectory frame index when structure changes so modelIndex never goes out of bounds
  $effect(() => {
    void viewerStructureMeta?.id;
    trajFrame = 0;
    if (!viewerStructureMeta?.trajectoryPdb) showTrajectory = false;
  });
</script>

<svelte:window
  onpointerdown={handleWindowPointerDown}
  onpointermove={handleWindowPointerMove}
  onpointerup={handleWindowPointerUp}
  onkeydown={handleWindowKeydown}
  onhashchange={() => void applySharedHash(window.location.hash, false)}
/>

<!-- Single Compact Top Header Bar (~48px) -->
<header class="app-header">
  <div class="hdr-left">
    <h1 class="brand-logo">
      <button
        bind:this={brandBtnEl}
        type="button"
        class="brand-logo-btn"
        class:confirm-danger={brandClearConfirmPending}
        disabled={isRunning}
        onclick={handleBrandLogoClick}
        title={brandClearConfirmPending
          ? 'Click again to clear all blocks and start a clean canvas'
          : graph.nodes.length > 0
            ? 'Start a clean empty canvas (click twice to confirm)'
            : 'Start a clean empty canvas'}
        aria-label={brandClearConfirmPending
          ? 'Confirm clearing canvas'
          : 'denovo.run — start a clean empty canvas'}
      >
        <span class="brand-wordmark"><span class="accent">denovo</span>.run</span>{#if brandClearConfirmPending}<span class="brand-confirm-pill">Clear? Confirm</span>{/if}
      </button>
    </h1>

    <button
      type="button"
      class="gallery-trigger"
      class:disabled={isRunning}
      data-tour="preset-select"
      disabled={isRunning}
      onclick={() => {
        sysMenuOpen = false;
        moreMenuOpen = false;
        shareMenuOpen = false;
        showExamplesGallery = true;
      }}
      title="Browse Examples & Starter Pipelines"
    >
      <span class="gallery-trigger-lbl">Examples</span>
      <span class="gallery-trigger-val">
        {activePaperInsight
          ? activePaperInsight.title
          : activePreset
            ? activePreset.name
            : graph.name || 'Browse examples'}
      </span>
      <span class="gallery-trigger-chev" aria-hidden="true">▾</span>
    </button>

    {#if activePaperInsight && !showPaperWalkthrough && !isRunning}
      <button
        type="button"
        class="walkthrough-reopen-btn"
        onclick={() => {
          showPaperWalkthrough = true;
          if (hasActivePaperOutputs && activePaperInsight) {
            applyPaperWalkthroughStep(activePaperInsight, paperWalkthroughIdx);
          }
        }}
        title="Reopen guided walkthrough for this literature example"
      >
        Walkthrough
      </button>
    {/if}
  </div>

  <div class="hdr-right">
    <!-- System Status Pill & Popover -->
    <div class="pop-wrap" bind:this={sysWrapEl}>
      <button
        type="button"
        class="hdr-pill"
        class:ok={((computeBackend.mode !== 'local' && remoteHealth?.ok) || (webgpuSupported && !webgpuIsFallback)) && !activeGpuWorker}
        class:warn={computeBackend.mode === 'local' && webgpuSupported && webgpuIsFallback && !activeGpuWorker}
        class:bad={(computeBackend.mode === 'local' && webgpuSupported === false) || (computeBackend.mode !== 'local' && remoteHealth?.ok === false)}
        class:gpu={Boolean(activeGpuWorker)}
        aria-expanded={sysMenuOpen}
        onclick={() => {
          moreMenuOpen = false;
          shareMenuOpen = false;
          const nextOpen = !sysMenuOpen;
          if (nextOpen && sysWrapEl) {
            const rect = sysWrapEl.getBoundingClientRect();
            sysPopAlignLeft = rect.right < 300;
          }
          sysMenuOpen = nextOpen;
          if (nextOpen && computeBackend.mode !== 'local') {
            void probeRemoteBackend(computeBackend);
          }
        }}
        title="WebGPU & Self-Hosted GPU backend, VRAM & weight cache"
      >
        <span class="status-dot"></span>
        {activeGpuWorkerLabel
          ? `GPU: ${activeGpuWorkerLabel}`
          : computeBackend.mode === 'remote'
            ? remoteHealth?.ok
              ? 'Host GPU'
              : 'Host GPU (!)'
            : computeBackend.mode === 'hybrid'
              ? remoteHealth?.ok
                ? 'Hybrid GPU'
                : 'Hybrid (!)'
              : webgpuIsFallback
                ? 'WebGPU (SW)'
                : 'WebGPU'}
      </button>

      {#if sysMenuOpen}
        <div
          class="hdr-popover sys-pop"
          class:align-left={sysPopAlignLeft}
          role="dialog"
          aria-label="System & WebGPU status"
        >
          <div class="pop-row">
            <span class="pop-k">Browser GPU</span>
            <span
              class="pop-v"
              class:ok-t={webgpuSupported && !webgpuIsFallback}
              class:warn-t={webgpuSupported && webgpuIsFallback}
              class:bad-t={webgpuSupported === false}
            >
              {webgpuDetail}
            </span>
          </div>
          {#if webgpuAdapterInfo && (webgpuAdapterInfo.vendor || webgpuAdapterInfo.architecture)}
            <div class="pop-row">
              <span class="pop-k">Adapter</span>
              <span class="pop-v">
                {[webgpuAdapterInfo.vendor, webgpuAdapterInfo.architecture].filter(Boolean).join(' · ')}
              </span>
            </div>
          {/if}
          {#if webgpuIsFallback && computeBackend.mode === 'local'}
            <div class="pop-warn-note">
              Software WebGPU fallback detected. Switch Compute to <b>Host GPU</b> or launch Linux Chrome
              with <code>--enable-features=Vulkan</code>.
            </div>
          {/if}

          <div class="pop-divider"></div>
          <div class="pop-compute-sec">
            <div class="pop-row">
              <span class="pop-k">Compute</span>
              <div class="compute-seg" role="group" aria-label="Compute backend mode">
                {#each [['local', 'Browser'], ['hybrid', 'Hybrid'], ['remote', 'Host GPU']] as [mVal, mLbl] (mVal)}
                  <button
                    type="button"
                    class="compute-seg-btn"
                    class:active={computeBackend.mode === mVal}
                    disabled={isRunning || activeRunState.leaseHeld}
                    onclick={() => updateComputeBackend({ mode: mVal as ComputeRoutingMode })}
                  >
                    {mLbl}
                  </button>
                {/each}
              </div>
            </div>

            {#if computeBackend.mode !== 'local'}
              <label class="pop-field">
                <span>Host Server URL (blank = same origin)</span>
                <input
                  type="text"
                  value={computeBackend.serverUrl}
                  placeholder="http://localhost:8787"
                  disabled={isRunning}
                  onchange={(e) => updateComputeBackend({ serverUrl: (e.currentTarget as HTMLInputElement).value })}
                />
              </label>
              <label class="pop-field">
                <span>API Token (optional)</span>
                <input
                  type="password"
                  value={computeBackend.apiToken}
                  placeholder="Bearer token if set"
                  disabled={isRunning}
                  onchange={(e) => updateComputeBackend({ apiToken: (e.currentTarget as HTMLInputElement).value })}
                />
              </label>
              {#if computeBackend.mode === 'hybrid'}
                <div class="pop-row">
                  <span class="pop-k">Offload &gt; {computeBackend.hybridThresholdMB} MB</span>
                  <input
                    class="pop-thresh"
                    type="number"
                    min="0"
                    max="1000"
                    step="10"
                    value={computeBackend.hybridThresholdMB}
                    disabled={isRunning}
                    onchange={(e) =>
                      updateComputeBackend({
                        hybridThresholdMB: Math.max(0, Math.min(1000, Number((e.currentTarget as HTMLInputElement).value) || 40)),
                      })}
                  />
                </div>
              {/if}
              <div class="pop-row">
                <span
                  class="pop-v"
                  class:ok-t={remoteHealth?.ok}
                  class:bad-t={remoteHealth?.ok === false}
                  style="text-align: left; overflow: hidden; text-overflow: ellipsis;"
                  title={remoteHealth?.error || remoteHealth?.gpu?.description || ''}
                >
                  {#if remoteChecking}
                    Checking host GPU…
                  {:else if remoteHealth?.ok}
                    {remoteHealth.gpu?.description || remoteHealth.gpu?.vendor || 'Host GPU ready'}
                    {(remoteHealth.queueDepth ?? 0) > 0 ? ` · q=${remoteHealth.queueDepth}` : ''}
                  {:else if remoteHealth?.error}
                    Offline: {remoteHealth.error}
                  {:else}
                    Not tested yet
                  {/if}
                </span>
                <button
                  type="button"
                  class="pop-action"
                  disabled={remoteChecking}
                  onclick={() => void probeRemoteBackend(computeBackend)}
                >
                  Test
                </button>
              </div>
              {#if remoteHealth?.ok && remoteHealth.activeAdapter}
                <div class="pop-row">
                  <span class="pop-k">Host VRAM</span>
                  <button
                    type="button"
                    class="pop-action"
                    disabled={isRunning || activeRunState.leaseHeld}
                    onclick={freeRemoteVram}
                  >
                    Free ({remoteHealth.activeAdapter})
                  </button>
                </div>
              {/if}
            {/if}
            <div class="pop-row">
              <span
                class="pop-k"
                title="Maximum total residues allowed in LocalFold (AF3, Boltz, Protenix, AF2, ESMFold2). Auto uses 512/768 in Browser mode and the host GPU ceiling on self-hosted servers."
              >
                Max fold len
              </span>
              <select
                class="pop-select"
                aria-label="Maximum folding residue ceiling"
                value={computeBackend.maxFoldResidues === null ? 'auto' : String(computeBackend.maxFoldResidues)}
                disabled={isRunning || activeRunState.leaseHeld}
                onchange={(e) =>
                  updateComputeBackend({
                    maxFoldResidues: sanitizeMaxFoldResidues((e.currentTarget as HTMLSelectElement).value),
                  })}
              >
                <option value="auto">
                  {computeBackend.mode === 'local'
                    ? 'Auto (512 / 768)'
                    : `Auto (Host ${remoteHealth?.maxFoldResidues ?? DEFAULT_HOST_MAX_FOLD_RESIDUES})`}
                </option>
                <option value="512">512 aa</option>
                <option value="768">768 aa</option>
                <option value="1024">1024 aa</option>
                <option value="1536">1536 aa</option>
                <option value="2048">2048 aa</option>
                <option value="3072">3072 aa</option>
                <option value="4096">4096 aa</option>
              </select>
            </div>
          </div>
          <div class="pop-divider"></div>
          {#if webMcpReady}
            <div class="pop-row">
              <span class="pop-k">Agent API</span>
              <span class="badge-ok">WebMCP active</span>
            </div>
          {/if}
          {#if activeGpuWorkerLabel}
            <div class="pop-row">
              <span class="pop-k">Active worker</span>
              <button
                type="button"
                class="pop-action"
                disabled={isRunning || activeRunState.leaseHeld}
                onclick={freeVram}
              >
                Free VRAM ({activeGpuWorkerLabel})
              </button>
            </div>
          {/if}
          <div class="pop-row">
            <span class="pop-k">Cache: {cacheUsedMB.toFixed(0)} MB</span>
            <button
              bind:this={cacheClearBtnEl}
              type="button"
              class="pop-action danger"
              class:confirm-danger={cacheClearConfirmPending}
              disabled={isRunning || activeRunState.leaseHeld}
              onclick={handleClearCacheClick}
              title={cacheClearConfirmPending
                ? 'Click again to clear cached model weights from browser storage'
                : 'Clear cached model weights from browser storage (click twice to confirm)'}
            >
              {cacheClearConfirmPending ? 'Clear? Confirm' : 'Clear'}
            </button>
          </div>
        </div>
      {/if}
    </div>

    <!-- Share Protocol Popover -->
    <div class="pop-wrap" bind:this={shareWrapEl}>
      <button
        type="button"
        class="hdr-share-btn"
        class:active={shareMenuOpen}
        aria-expanded={shareMenuOpen}
        aria-label="Share workflow link"
        title="Share workflow pipeline as a zero-install link"
        onclick={() => void openSharePopover()}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8" />
          <polyline points="16 6 12 2 8 6" />
          <line x1="12" y1="2" x2="12" y2="15" />
        </svg>
        <span class="hdr-share-label">Share</span>
      </button>

      {#if shareMenuOpen}
        <div class="hdr-popover share-pop" role="dialog" aria-label="Share workflow link">
          <div class="share-pop-head">
            <span class="share-pop-title">Share workflow pipeline</span>
            {#if sharePopoverState?.url}
              <span class="share-pop-meta">{sharePopoverState.url.length} chars</span>
            {/if}
          </div>
          <p class="share-pop-note">
            <strong>Only the workflow pipeline is shared</strong> (blocks, wiring &amp; parameters).
            Computed outputs and 3D structures are <strong>not</strong> cached or uploaded — recipients run it locally in their browser.
          </p>
          {#if sharePopoverState?.strippedPdb}
            <p class="share-pop-warn">
              Uploaded local PDB coordinates were omitted to keep the URL compact; recipients will need to re-attach that file.
            </p>
          {/if}
          {#if sharePopoverState?.error}
            <p class="share-pop-warn">{sharePopoverState.error}</p>
          {:else if sharePopoverState?.url}
            <div class="share-pop-url-row">
              <input
                type="text"
                class="share-pop-input"
                readonly
                value={sharePopoverState.url}
                aria-label="Shareable workflow URL"
                onclick={(e) => e.currentTarget.select()}
              />
              <button
                type="button"
                class="share-pop-copy"
                class:copied={sharePopoverState.copied}
                onclick={() => void copySharePopoverUrl()}
              >
                {sharePopoverState.copied ? '✓ Copied' : 'Copy'}
              </button>
            </div>
          {:else}
            <p class="share-pop-note">Generating compact workflow link…</p>
          {/if}
          <div class="share-pop-foot">
            <button
              type="button"
              class="share-pop-link-btn"
              onclick={() => {
                shareMenuOpen = false;
                showRecipe = true;
              }}
            >
              Open &lt;/&gt; YAML / JSON Recipe
            </button>
          </div>
        </div>
      {/if}
    </div>

    <!-- GitHub Repository Link -->
    <a
      class="hdr-gh-btn"
      href="https://github.com/ramithuh/ProteinDesignSpace/blob/main/ADDING_A_METHOD.md"
      target="_blank"
      rel="noopener noreferrer"
      title="Add a new method to denovo.run (guide & coding-agent prompt)"
      aria-label="Add a new method to denovo.run (guide on GitHub)"
    >
      <span class="hdr-gh-label">Add a method?</span>
      <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
        <path
          fill-rule="evenodd"
          d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
        />
      </svg>
    </a>

    <!-- Layout Split Toggle (Horizontal Side-by-Side vs Vertical Stacked) -->
    <button
      type="button"
      class="hdr-icon-btn"
      onclick={toggleSplitMode}
      title={splitMode === 'horizontal'
        ? 'Switch to top/bottom stacked layout'
        : 'Switch to left/right horizontal split'}
      aria-label="Toggle horizontal or vertical split layout"
    >
      {#if splitMode === 'horizontal'}
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <line x1="3" y1="12" x2="21" y2="12" />
        </svg>
      {:else}
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <line x1="12" y1="3" x2="12" y2="21" />
        </svg>
      {/if}
    </button>

    <!-- Theme Toggle -->
    <button
      type="button"
      class="hdr-icon-btn"
      onclick={toggleTheme}
      title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
      aria-label="Toggle color theme"
    >
      {#if theme === 'dark'}
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
        </svg>
      {:else}
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      {/if}
    </button>

    <!-- Overflow Menu Popover -->
    <div class="pop-wrap" bind:this={moreWrapEl}>
      <button
        type="button"
        class="hdr-icon-btn"
        class:active={moreMenuOpen}
        aria-expanded={moreMenuOpen}
        aria-label="Workflow actions menu"
        title="Recipe, Import/Export, Report a bug & Run options"
        onclick={() => {
          sysMenuOpen = false;
          shareMenuOpen = false;
          moreMenuOpen = !moreMenuOpen;
        }}
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <circle cx="5" cy="12" r="1.8" />
          <circle cx="12" cy="12" r="1.8" />
          <circle cx="19" cy="12" r="1.8" />
        </svg>
      </button>

      {#if moreMenuOpen}
        <div class="hdr-popover menu-pop" role="menu">
          <button
            type="button"
            class="menu-item"
            role="menuitem"
            onclick={() => {
              showRecipe = !showRecipe;
              moreMenuOpen = false;
            }}
          >
            <span>{showRecipe ? 'Hide Recipe' : '</> Recipe'}</span>
          </button>
          <button type="button" class="menu-item" role="menuitem" onclick={copyQuickShareLink}>
            <span>Share Link</span>
          </button>
          <button type="button" class="menu-item" role="menuitem" onclick={exportWorkflowJson}>
            <span>Export JSON</span>
          </button>
          <button
            type="button"
            class="menu-item"
            role="menuitem"
            disabled={isRunning}
            onclick={() => importFileInput?.click()}
          >
            <span>Import JSON / YAML</span>
          </button>
          <button
            type="button"
            class="menu-item"
            role="menuitem"
            onclick={() => {
              moreMenuOpen = false;
              showOnboarding = true;
            }}
          >
            <span>Quick Tour</span>
          </button>
          <button
            type="button"
            class="menu-item"
            role="menuitem"
            onclick={() => {
              moreMenuOpen = false;
              showCredits = true;
            }}
          >
            <span>Credits &amp; Licenses</span>
          </button>
          <a
            class="menu-item menu-link"
            role="menuitem"
            href={buildBugReportUrl(runError)}
            target="_blank"
            rel="noopener noreferrer"
            onclick={() => {
              moreMenuOpen = false;
            }}
          >
            <span>Report a bug</span>
          </a>
          <div class="menu-sep"></div>
          <button
            type="button"
            class="menu-item"
            role="menuitem"
            disabled={isRunning || validationErrors.length > 0 || hasInvalidNumeric || hasBioErrors}
            onclick={() => runWorkflow(true)}
          >
            <span>Re-run Fresh</span>
          </button>
        </div>
      {/if}
      <input
        bind:this={importFileInput}
        type="file"
        accept=".json,.yaml,.yml"
        hidden
        tabindex="-1"
        aria-hidden="true"
        disabled={isRunning}
        onchange={importGraphJson}
      />
    </div>

    <!-- Cancel CTA while any canvas or API run is active -->
    {#if isRunning}
      <button
        type="button"
        class="btn-cancel"
        onclick={handleCancelRun}
        title="Cancel active run and preserve completed steps"
      >
        ■ Cancel
      </button>
    {/if}

    <!-- Primary Run CTA -->
    <button
      type="button"
      class="btn-primary"
      class:is-running={isRunning}
      data-tour="run-btn"
      disabled={isRunning || validationErrors.length > 0 || hasInvalidNumeric || hasBioErrors}
      onclick={() => runWorkflow(false)}
      title={hasInvalidNumeric
        ? 'Fix invalid numeric parameter(s) before running'
        : hasBioErrors
          ? `Biological compatibility error: ${bioErrorIssues[0].message}`
          : 'Run workflow (R or ⌘↵ / Ctrl+Enter)'}
    >
      {#if isRunning}
        <span class="run-pulse"></span>
        {runStepLabel}
      {:else}
        ▶ Run
        <kbd class="run-kbd">R</kbd>
      {/if}
    </button>
  </div>
</header>

<div
  class="workbench-split"
    class:horizontal={splitMode === 'horizontal'}
    class:vertical={splitMode === 'vertical'}
    class:dragging={draggingSplit}
    bind:this={splitWrapEl}
    style={splitMode === 'horizontal'
      ? `--left-pane: ${splitPct}%;`
      : `--top-pane: ${verticalSplitPct}%;`}
  >
    <!-- Graph Canvas Pane (Left in Horizontal Split, Top in Vertical Split) -->
    <section class="flow-panel" data-tour="workflow-canvas">
      <SvelteFlowProvider>
        <WorkflowCanvas
          {graph}
          {selectedNodeId}
          {selectedNodeIds}
          {compareColors}
          {selectedEdgeId}
          activePeekPort={
            rightPanelTab === 'peek' && !isCompareMode && selectedNode && resolvedPeekSlotId
              ? { nodeId: selectedNode.id, slotId: resolvedPeekSlotId }
              : null
          }
          {nodeStates}
          {outputsByNode}
          {theme}
          disabled={isRunning}
          {isRunning}
          fillHeight={true}
          autoCommitProposed={showOnboarding && onboardingInteractive}
          fitViewTrigger={canvasFitViewTrigger}
          onGraphChange={handleGraphChange}
          onApplyPlannedGraph={handleApplyPlannedGraph}
          onDraftPreviewChange={(draft) => (activeDraftPreview = draft)}
          onToast={(msg, tone) => showToastMessage(msg, 3200, tone ?? 'error')}
          onSelectNode={handleSelectNode}
          onSelectNodes={handleSelectNodes}
          onBlockAdded={handleBlockAdded}
          onSelectEdge={handleSelectEdge}
          onPeekPort={handlePeekPort}
          onNodeDoubleClick={openStudioForNode}
          onOpenStudio={openStudioForNode}
          onOpenTour={() => (showOnboarding = true)}
        />
      </SvelteFlowProvider>

      {#if shareToast || (activeDraftPreview && !activeDraftPreview.isIncrementalEdit) || (appliedDraftNotice && !activeDraftPreview && !isRunning && graph.nodes.length > 0)}
        <div class="canvas-top-spine">
          {#if shareToast}
            <div
              class="share-toast"
              class:is-error={shareToastTone === 'error'}
              role={shareToastTone === 'error' ? 'alert' : 'status'}
            >
              {shareToast}
            </div>
          {/if}
          {#if activeDraftPreview && !activeDraftPreview.isIncrementalEdit}
            <div class="draft-preview-banner" role="status">
              <span class="dpb-spark" aria-hidden="true">✦</span>
              <span>
                Preview: <strong>New graph</strong> · {activeDraftPreview.graph.nodes.length} blocks
                {#if graph.nodes.length > 0}
                  — current graph kept until applied
                {/if}
              </span>
            </div>
          {:else if appliedDraftNotice && !activeDraftPreview && !isRunning && graph.nodes.length > 0}
            <div class="ai-draft-warning" class:is-ai={appliedDraftNotice.isAiDraft} role="status">
              <span class="adw-badge" class:is-ai={appliedDraftNotice.isAiDraft} aria-hidden="true">
                {appliedDraftNotice.isAiDraft ? '⚠ AI draft' : '✦ Draft'}
              </span>
              <span class="adw-copy">
                Double-check inputs, residue ranges &amp; parameters before running.
              </span>
              <button
                type="button"
                class="adw-dismiss"
                onclick={() => (appliedDraftNotice = null)}
                aria-label="Dismiss draft warning"
                title="Dismiss"
              >
                ×
              </button>
            </div>
          {/if}
        </div>
      {/if}

      {#if showRecipe}
        <div class="drawer-scroll">
          <RecipeDrawer
            {graph}
            disabled={isRunning}
            onApplyGraph={(next) => loadFullWorkflowGraph(next, { clearOutputs: true })}
          />
        </div>
      {/if}

      {#if displayedValidationErrors.length > 0}
        <div class="warn-banner">
          <b>Workflow check:</b> {displayedValidationErrors.join(' · ')}
        </div>
      {/if}

      {#if bioIssues.length > 0}
        <div class="bio-banner-list" role="status" aria-live="polite">
          {#each bioIssues as issue (`${issue.code}:${issue.nodeId}`)}
            <div class="bio-banner" class:is-err={issue.severity === 'error'}>
              <span class="bio-msg">
                <b>Biological compatibility:</b> {issue.message}
              </span>
              {#if issue.fix && !previewChecks.hideBioFixes}
                <button
                  type="button"
                  class="bio-fix-btn"
                  disabled={isRunning}
                  onclick={() => applyBioFix(issue)}
                >
                  {issue.fix.label}
                </button>
              {/if}
            </div>
          {/each}
        </div>
      {/if}

      {#if runError && !previewChecks.hideRunError}
        <div class="error-banner error-banner-row" role="alert">
          <div class="error-banner-text">
            <b>Pipeline error:</b> {runError}
          </div>
          <a
            class="error-report-btn"
            href={buildBugReportUrl(runError)}
            target="_blank"
            rel="noopener noreferrer"
            title="Open a pre-filled GitHub issue with this error and workflow summary"
          >
            Report bug
          </a>
        </div>
      {/if}

      {#if hashError}
        <div class="error-banner dismissible" role="alert">
          <span><b>Shared link error:</b> {hashError}</span>
          <button
            type="button"
            class="banner-dismiss"
            onclick={() => (hashError = null)}
            aria-label="Dismiss shared link error"
          >
            ×
          </button>
        </div>
      {/if}
    </section>

    {#if splitMode === 'horizontal'}
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <div
        class="split-handle"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize graph and output panes"
        title="Drag to resize · Double-click to reset"
        onpointerdown={startSplitDrag}
        ondblclick={() => {
          splitPct = 65;
          persistSplitSizes();
        }}
      >
        <div class="split-bar"></div>
      </div>
    {:else}
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <div
        class="split-handle is-vertical"
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize graph and output panes vertically"
        title="Drag to resize · Double-click to reset"
        onpointerdown={startSplitDrag}
        ondblclick={() => {
          verticalSplitPct = 48;
          persistSplitSizes();
        }}
      >
        <div class="split-bar"></div>
      </div>
    {/if}

    <!-- Output & Inspector Pane (Right in Horizontal Split, Bottom in Vertical Split) -->
    {#if expandedPane !== 'none'}
      <div
        class="expand-backdrop"
        role="presentation"
        onclick={() => (expandedPane = 'none')}
      ></div>
    {/if}
    <section
      class="results-grid"
      class:stacked-pane={splitMode === 'horizontal'}
      class:is-studio-open={studioOpen}
    >
      <div class="viewer-col" class:is-expanded={expandedPane === 'viewer'} data-tour="viewer-panel">
        <div class="viewer-bar">
          {#if isCompareMode && comparisonState}
            <div class="seg-group" role="group" aria-label="3D comparison layout">
              <button
                type="button"
                class="seg-btn"
                class:active={comparisonState.effectiveLayout === 'overlay'}
                disabled={comparisonState.forcedSideBySide}
                title={comparisonState.forcedSideBySide
                  ? 'Structures have different lengths — using side-by-side'
                  : 'Superpose models in one frame'}
                onclick={() => setActiveCompareOption({ layout: 'overlay' })}
              >
                Overlay
              </button>
              <button
                type="button"
                class="seg-btn"
                class:active={comparisonState.effectiveLayout === 'side-by-side'}
                onclick={() => setActiveCompareOption({ layout: 'side-by-side' })}
              >
                Side-by-Side
              </button>
            </div>
            <div class="seg-group" role="group" aria-label="3D ribbon color mode">
              <button
                type="button"
                class="seg-btn"
                class:active={effectiveCompareColorMode === 'by-model'}
                onclick={() => setActiveCompareOption({ colorMode: 'by-model' })}
              >
                By Model
              </button>
              <button
                type="button"
                class="seg-btn"
                class:active={effectiveCompareColorMode === 'plddt'}
                onclick={() => setActiveCompareOption({ colorMode: 'plddt' })}
              >
                pLDDT
              </button>
            </div>
            {@const isSideBySide = comparisonState.effectiveLayout === 'side-by-side'}
            {@const sharedPreAligned = Boolean(comparisonState.sharedReferenceLabel) && !isSideBySide}
            {@const kabschDisabled = sharedPreAligned || isSideBySide}
            <label
              class:disabled-opt={kabschDisabled}
              title={sharedPreAligned
                ? 'Branches share a designed backbone and are pre-aligned to it in Overlay mode'
                : isSideBySide
                  ? 'Kabsch alignment applies in Overlay layout'
                  : 'Superpose models with Kabsch alignment (supports motif/substructure alignment for different lengths)'}
            >
              <input
                type="checkbox"
                disabled={kabschDisabled}
                checked={effectiveCompareAlignKabsch}
                onchange={(e) => {
                  setActiveCompareOption({ alignKabsch: e.currentTarget.checked });
                }}
              />
              {sharedPreAligned ? 'Kabsch align (shared)' : 'Kabsch align'}
            </label>
            <label><input type="checkbox" bind:checked={illustrative} /> Illustrative</label>
            <label
              class:disabled-opt={isCanonicalViewerLayout}
              title={isCanonicalViewerLayout
                ? 'Spin is paused in Side-by-Side / Grid layout to preserve A (left) → B (right) orientation'
                : 'Slowly rotate the 3D view (pauses while dragging)'}
            >
              <input type="checkbox" disabled={isCanonicalViewerLayout} bind:checked={viewerSpin} /> Spin
            </label>
            <button
              type="button"
              class="cmp-exit-mini"
              onclick={() => setSingleSelection(selectedNodeId)}
              title="Exit comparison mode (Esc)"
            >
              × Exit
            </button>
          {:else}
            {#if hasMultiDesignStructures && !studioInputViewerActive}
              <div class="seg-group" role="group" aria-label="Multi-design 3D view mode">
                <button
                  type="button"
                  class="seg-btn"
                  class:active={selectedDesignKey === 'all' && multiDesignViewMode === 'grid'}
                  onclick={() => {
                    selectedDesignKey = 'all';
                    multiDesignViewMode = 'grid';
                  }}
                  title="Show all generated designs arranged side-by-side in a 3D gallery"
                >
                  Grid ({peekStructureItems.length})
                </button>
                <button
                  type="button"
                  class="seg-btn"
                  class:active={selectedDesignKey === 'all' && multiDesignViewMode === 'overlay'}
                  onclick={() => {
                    selectedDesignKey = 'all';
                    multiDesignViewMode = 'overlay';
                  }}
                  title="Superpose all generated designs in one frame"
                >
                  Overlay ({peekStructureItems.length})
                </button>
                <button
                  type="button"
                  class="seg-btn"
                  class:active={selectedDesignKey !== 'all' || multiDesignViewMode === 'single'}
                  onclick={() => {
                    if (selectedDesignKey === 'all') {
                      selectedDesignKey = designKeyForItem(peekStructureItems[0], 0);
                    }
                    multiDesignViewMode = 'single';
                  }}
                  title="Inspect a single design"
                >
                  Single
                </button>
              </div>
              <div class="seg-group" role="group" aria-label="Step between generated designs">
                <button
                  type="button"
                  class="seg-btn"
                  onclick={() => stepMultiDesign(-1)}
                  title="Previous design"
                  aria-label="Previous design"
                >
                  ◀
                </button>
                <span class="seg-btn" style="cursor: default; pointer-events: none;">
                  {selectedDesignKey === 'all'
                    ? `All ${peekStructureItems.length}`
                    : `D${activePeekStructureIdx + 1}/${peekStructureItems.length}`}
                </span>
                <button
                  type="button"
                  class="seg-btn"
                  onclick={() => stepMultiDesign(1)}
                  title="Next design"
                  aria-label="Next design"
                >
                  ▶
                </button>
              </div>
            {/if}
            <label><input type="checkbox" name="viewer-show-designed" bind:checked={showDesigned} /> Designed</label>
            <label><input type="checkbox" name="viewer-show-refold" bind:checked={showRefold} /> Refold (pLDDT)</label>
            <label><input type="checkbox" name="viewer-illustrative" bind:checked={illustrative} /> Illustrative</label>
            <label
              class:disabled-opt={isCanonicalViewerLayout}
              title={isCanonicalViewerLayout
                ? 'Spin is paused in Side-by-Side / Grid layout to preserve A (left) → B (right) orientation'
                : 'Slowly rotate the 3D view (pauses while dragging)'}
            >
              <input type="checkbox" name="viewer-spin" disabled={isCanonicalViewerLayout} bind:checked={viewerSpin} /> Spin
            </label>
            {#if viewerStructureMeta?.trajectoryPdb}
              <label><input type="checkbox" name="viewer-show-trajectory" bind:checked={showTrajectory} /> Trajectory</label>
              {#if showTrajectory}
                <input
                  type="range"
                  name="viewer-trajectory-frame"
                  min="0"
                  max={Math.max(0, (viewerStructureMeta.trajectoryFrameCount ?? 1) - 1)}
                  bind:value={trajFrame}
                  class="slider"
                />
                <code class="mono">{trajFrame + 1}/{viewerStructureMeta.trajectoryFrameCount}</code>
              {/if}
            {/if}

            {#if studioInputViewerActive}
              <span
                class="studio-brush-tag"
                title={studioActiveBrush
                  ? `Clicking 3D residues toggles them in ${studioActiveBrush.label} on ${selectedNode?.label ?? selectedNode?.adapterId ?? 'this block'}`
                  : 'Inspect-only in Studio Input view (this block has no editable residue brush)'}
              >
                {studioActiveBrush
                  ? `Studio brush: ${studioActiveBrush.label}`
                  : 'Studio view (inspect)'}
              </span>
            {:else if hasConditionerNode || hasMpnnNode}
              <label class="pick-mode" title="Choose what clicking a 3D protein residue modifies">
                Pick:
                <select class="pick-select" name="viewer-pick-mode" bind:value={molPickMode}>
                  <option value="auto">Auto</option>
                  <option value="off">Off</option>
                  {#if hasConditionerNode}
                    <option value="hotspots">Hotspots</option>
                    <option value="motif">Motif</option>
                  {/if}
                  {#if hasMpnnNode}
                    <option value="mpnn">Fix Positions (MPNN)</option>
                  {/if}
                </select>
                {#if pickTargetInfo}
                  <span class="pick-target-tag" title={`3D clicks append to ${pickTargetInfo.fieldLabel} on ${pickTargetInfo.nodeName}`}>
                    → {pickTargetInfo.nodeName}
                  </span>
                {:else if molPickMode !== 'off'}
                  <span class="pick-target-tag muted" title="Click a conditioned block on the canvas to target 3D picking">
                    → inspect
                  </span>
                {/if}
              </label>
            {/if}

            {#if lastPickedResidue}
              <span class="picked">Clicked: <code>{lastPickedResidue}</code></span>
            {/if}
            {#if isPeekingSequenceSlot && showSeqParentBackbone}
              <button
                type="button"
                class="cmp-exit-mini"
                onclick={() => (showSeqParentBackbone = false)}
                title="Hide parent backbone preview"
              >
                Hide backbone
              </button>
            {/if}
          {/if}
          {#if !studioOpen}
            <button
              type="button"
              class="pane-expand-btn"
              class:active={expandedPane === 'viewer'}
              onclick={() => (expandedPane = expandedPane === 'viewer' ? 'none' : 'viewer')}
              title={expandedPane === 'viewer' ? 'Collapse 3D viewer (Esc)' : 'Expand 3D viewer to full view'}
              aria-label={expandedPane === 'viewer' ? 'Collapse 3D viewer' : 'Expand 3D viewer'}
            >
              {expandedPane === 'viewer' ? '⤡ Collapse' : '⤢ Expand'}
            </button>
          {/if}
        </div>

        <div class="viewer-stage">
          <MolSuperposeViewer
            designedPdb={isCompareMode || isMultiDesignAll3d
              ? null
              : studioInputViewerActive
                ? (studioPreview?.previewStructurePdb ?? viewerDesignedPdb)
                : viewerDesignedPdb}
            refoldPdb={isCompareMode || isMultiDesignAll3d || studioInputViewerActive ? null : viewerRefoldPdb}
            models={isCompareMode && comparisonState
              ? comparisonState.models
              : isMultiDesignAll3d
                ? multiDesignModels
                : null}
            trajectoryPdb={isCompareMode || isMultiDesignAll3d || studioInputViewerActive
              ? null
              : (viewerStructureMeta?.trajectoryPdb ?? null)}
            frameIndex={showTrajectory ? trajFrame : undefined}
            {showDesigned}
            showRefold={studioInputViewerActive || isMultiDesignAll3d ? false : showRefold}
            {showTrajectory}
            {illustrative}
            spin={viewerSpin && !isRunning && !isCanonicalViewerLayout}
            canonicalCamera={isCanonicalViewerLayout}
            expanded={expandedPane === 'viewer' || studioOpen}
            defaultStyleMenuOpen={expandedPane === 'viewer' && !studioOpen}
            {theme}
            height={expandedPane === 'viewer' || studioOpen ? '100%' : splitMode === 'horizontal' ? 350 : 460}
            label={isCompareMode
              ? compareViewerLabel
              : studioInputViewerActive
                ? (studioPreview?.previewStructureMeta?.label ?? viewerLabel)
                : viewerLabel}
            atomGranularity={!isCompareMode && studioInputViewerActive
              ? (studioPreview?.previewStructureMeta?.atomGranularity ?? viewerAtomGranularity)
              : viewerAtomGranularity}
            bfactorSemantics={!isCompareMode && studioInputViewerActive
              ? (studioPreview?.previewStructureMeta?.bfactorSemantics ?? 'none')
              : viewerBfactorSemantics}
            highlightLayers={!isCompareMode && studioInputViewerActive ? studioPreview?.highlightLayers : undefined}
            hoveredToken={studioOpen ? studioHoveredToken : null}
            focusToken={studioOpen ? studioFocusToken : viewerFocusToken}
            emptyMessage={isCompareMode ? compareEmptyMessage : isMultiDesignAll3d ? undefined : viewerEmptyMessage}
            emptyActionLabel={!isCompareMode && !isMultiDesignAll3d && isPeekingSequenceSlot && !showSeqParentBackbone && contextStructure
              ? 'Preview parent backbone'
              : undefined}
            onEmptyAction={() => (showSeqParentBackbone = true)}
            onResidueClick={isCompareMode || isMultiDesignAll3d
              ? undefined
              : studioInputViewerActive
                ? handleStudioInputResidueClick
                : handleResidueClick}
            onToggleModelVisibility={isMultiDesignAll3d
              ? toggleMultiDesignModelVisibility
              : toggleCompareModelVisibility}
            onSoloModel={isMultiDesignAll3d ? soloMultiDesignModel : soloCompareModel}
            onNotify={(msg, tone) => showToastMessage(msg, tone === 'error' ? 4000 : 2400, tone ?? 'ok')}
          />
        </div>

        {#if !isCompareMode && viewerStructureMeta}
          <div class="geom-strip">
            <span class="geom-lbl">
              {viewerStructureMeta.sourceModelId === 'pdb-input' && !peekMetrics ? 'Input:' : 'Designed:'}
            </span>
            <span>L = <b>{viewerStructureMeta.length} aa</b></span>
            <span>R<sub>g</sub> = <b>{viewerStructureMeta.geometry.rg.toFixed(1)} Å</b></span>
            <span>Cα breaks: <b>{viewerStructureMeta.geometry.chainBreaks}</b></span>
            <span>Clashes: <b>{viewerStructureMeta.geometry.clashes}</b></span>
            {#if totalElapsedMs !== null}
              <span>Total: <b>{(totalElapsedMs / 1000).toFixed(1)}s</b></span>
            {/if}
          </div>
        {/if}
      </div>

      <div class="right-panel" class:is-expanded={expandedPane === 'inspector'} data-tour="inspector-panel">
        {#if studioOpen && selectedNode}
          <NodeStudioPopover
            node={selectedNode}
            allNodes={graph.nodes}
            preview={studioPreview}
            activeTab={studioTab}
            hoveredToken={studioHoveredToken}
            {isRunning}
            onSelectTab={(t) => (studioTab = t)}
            onSelectNode={(id) => {
              setSingleSelection(id);
              selectedPeekSlotId = null;
              selectedEdgeId = null;
              studioHoveredToken = null;
              studioFocusToken = null;
              studioActiveBrush = null;
            }}
            onClose={() => {
              expandedPane = 'none';
              studioHoveredToken = null;
              studioFocusToken = null;
              studioActiveBrush = null;
            }}
            onRunUpstreamAndThis={() => runWorkflow(false, selectedNode.id)}
            onUpdateNodeParams={updateNodeParams}
            onApplyFix={applyBioCompatibilityFix}
            onHoverToken={(tok) => (studioHoveredToken = tok)}
            onFocusToken={(tok) => {
              studioFocusToken = null;
              queueMicrotask(() => {
                studioFocusToken = tok;
              });
            }}
            onActiveBrushChange={(brush) => (studioActiveBrush = brush)}
            onNotify={(msg, tone) => showToastMessage(msg, tone === 'error' ? 3600 : 2200, tone ?? 'ok')}
          >
            {#snippet controlsSlot()}
              <NodeInspector
                node={selectedNode}
                embedded
                disabled={isRunning}
                nodeOutput={outputsByNode[selectedNode.id]}
                nodeState={nodeStates[selectedNode.id]}
                upstreamStructure={selectedUpstreamStructure}
                upstreamSequences={selectedUpstreamSequences}
                upstreamLockSummary={selectedUpstreamLockSummary}
                upstreamBenchmarkId={selectedUpstreamBenchmarkId}
                hasStructureInput={graph.edges.some(
                  (e) => e.toNode === selectedNode.id && (e.toPort === 'structure' || e.toPort === 'template'),
                )}
                siblingGeneratorIds={
                  BENCHMARK_SYNC_ADAPTER_IDS.has(selectedNode.adapterId)
                    ? getSiblingGeneratorIds(selectedNode.id)
                    : []
                }
                onUpdate={updateNodeParams}
                onUpdateLabel={updateNodeLabel}
                onSyncTaskToSiblings={handleSyncTaskToSiblings}
                onRunUpTo={(id) => runWorkflow(false, id)}
                onNotify={(msg, tone) => showToastMessage(msg, tone === 'error' ? 3600 : 2200, tone ?? 'ok')}
              />
            {/snippet}
            {#snippet outputSlot()}
              <ArtifactPeekPanel
                node={selectedNode}
                embedded
                slots={peekSlots}
                activeSlot={activePeekSlot}
                {contextStructure}
                {selectedRowIdx}
                {selectedFoldIdx}
                {selectedDesignKey}
                {scRmsdThreshold}
                {plddtThreshold}
                {ipaeThreshold}
                {motifRmsdThreshold}
                mutationHighlight={activePaperMutationHighlight}
                {ipsaeThreshold}
                {isRunning}
                hasScorerResult={Boolean(latestScorerHit && latestScorerHit.nodeId !== selectedNode?.id)}
                onSelectSlot={handleSelectPeekSlot}
                onSelectRow={(idx) => (selectedRowIdx = idx)}
                onSelectFold={(idx) => (selectedFoldIdx = idx)}
                onSelectDesignKey={(key) => {
                  selectedDesignKey = key;
                  if (key === 'all' && multiDesignViewMode === 'single') {
                    multiDesignViewMode = 'grid';
                  }
                }}
                onRunUpToNode={(id) => runWorkflow(false, id)}
                onJumpToScorer={() => {
                  if (latestScorerHit) {
                    setSingleSelection(latestScorerHit.nodeId);
                    selectedPeekSlotId = null;
                    selectedEdgeId = null;
                    studioTab = 'output';
                  }
                }}
                onDownloadText={downloadText}
                onSelectResidue={handleSelectPerResidue}
              />
            {/snippet}
          </NodeStudioPopover>
        {:else}
        <div class="right-tabs">
          <div class="r-tab-group" role="tablist" aria-label="Block inspector tabs">
            <button
              type="button"
              role="tab"
              aria-selected={rightPanelTab === 'params'}
              class="r-tab"
              class:active={rightPanelTab === 'params'}
              onclick={() => (rightPanelTab = 'params')}
            >
              Parameters
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={rightPanelTab === 'peek'}
              class="r-tab"
              class:active={rightPanelTab === 'peek'}
              onclick={() => (rightPanelTab = 'peek')}
            >
              {isCompareMode ? `Compare (${effectiveCompareNodeIds.length})` : 'Outputs'}
              {#if hasPeekData || isCompareMode}
                <span class="live-dot" title="Outputs or comparison ready"></span>
              {/if}
            </button>
          </div>
          <div class="r-tab-actions">
            {#if !isCompareMode && comparableBranchIds.length >= 2}
              <button
                type="button"
                class="cmp-launch-btn"
                title="Compare branches in 3D (or Ctrl/⌘-click / Shift-click blocks on the canvas)"
                onclick={() => handleSelectNodes(comparableBranchIds)}
              >
                Compare branches ({comparableBranchIds.length})
              </button>
            {:else if selectedNode && (!isCompareMode || rightPanelTab === 'params')}
              <span class="tab-node-ctx" title={selectedNode.label ?? selectedNode.adapterId}>
                {isCompareMode ? 'Editing: ' : ''}{selectedNode.label ?? ADAPTER_REGISTRY[selectedNode.adapterId]?.shortName ?? selectedNode.adapterId}
              </span>
            {/if}
            {#if selectedNode && !isCompareMode}
              <ModelCardPopover adapterId={selectedNode.adapterId} params={selectedNode.params} compact />
              <button
                type="button"
                class="pane-expand-btn studio-launch-btn"
                onclick={() => openStudioForNode(selectedNode.id)}
                title="Open Node Studio — inspect compiled inputs, annotated FASTA, and 3D overlays (Space)"
                aria-label="Open Node Studio"
              >
                ⌖ Studio
              </button>
            {/if}
            <button
              type="button"
              class="pane-expand-btn tab-expand-btn"
              class:active={expandedPane === 'inspector'}
              onclick={() => (expandedPane = expandedPane === 'inspector' ? 'none' : 'inspector')}
              title={expandedPane === 'inspector' ? 'Collapse panel (Esc)' : 'Expand panel to full view'}
              aria-label={expandedPane === 'inspector' ? 'Collapse inspector panel' : 'Expand inspector panel'}
            >
              {expandedPane === 'inspector' ? '⤡' : '⤢'}
            </button>
          </div>
        </div>

        <div class="right-body">
          {#if rightPanelTab === 'params'}
            {#if selectedNode}
              <NodeInspector
                node={selectedNode}
                embedded
                disabled={isRunning}
                nodeOutput={outputsByNode[selectedNode.id]}
                nodeState={nodeStates[selectedNode.id]}
                upstreamStructure={selectedUpstreamStructure}
                upstreamSequences={selectedUpstreamSequences}
                upstreamLockSummary={selectedUpstreamLockSummary}
                upstreamBenchmarkId={selectedUpstreamBenchmarkId}
                hasStructureInput={graph.edges.some(
                  (e) => e.toNode === selectedNode.id && (e.toPort === 'structure' || e.toPort === 'template'),
                )}
                siblingGeneratorIds={
                  BENCHMARK_SYNC_ADAPTER_IDS.has(selectedNode.adapterId)
                    ? getSiblingGeneratorIds(selectedNode.id)
                    : []
                }
                onUpdate={updateNodeParams}
                onUpdateLabel={updateNodeLabel}
                onSyncTaskToSiblings={handleSyncTaskToSiblings}
                onRunUpTo={(id) => runWorkflow(false, id)}
                onOpenStudio={() => openStudioForNode(selectedNode.id)}
                onNotify={(msg, tone) => showToastMessage(msg, tone === 'error' ? 3600 : 2200, tone ?? 'ok')}
              />
            {:else}
              <div class="empty-inspector">
                Select a block on the canvas or click <b>+ Add Block</b> to configure parameters.
              </div>
            {/if}
          {:else if isCompareMode && comparisonState}
            <ComparePeekPanel
              comparison={comparisonState}
              {scRmsdThreshold}
              {plddtThreshold}
              {ipaeThreshold}
              {motifRmsdThreshold}
              {ipsaeThreshold}
              {thresholdsByNodeId}
              branchMetaByNodeId={compareBranchMetaByNodeId}
              {isRunning}
              onSelectBranchCandidate={(nodeId, idx) => {
                branchCandidateIdx = { ...branchCandidateIdx, [nodeId]: idx };
              }}
              onChangePairingMode={(mode) => {
                comparePairingMode = mode;
                branchCandidateIdx = {};
              }}
              onSelectCandidateKey={(key) => {
                compareSelectedCandidateKey = key;
              }}
              onFocusSingle={(nodeId) => handleSelectNode(nodeId)}
              onRunUpToNode={(id) => runWorkflow(false, id)}
              onDownloadText={downloadText}
              onNotify={(msg, tone) => showToastMessage(msg, tone === 'error' ? 3600 : 2200, tone ?? 'ok')}
              onExitCompare={() => setSingleSelection(selectedNodeId)}
            />
          {:else}
            <ArtifactPeekPanel
              node={selectedNode}
              embedded
              slots={peekSlots}
              activeSlot={activePeekSlot}
              {contextStructure}
              {selectedRowIdx}
              {selectedFoldIdx}
              {selectedDesignKey}
              {scRmsdThreshold}
              {plddtThreshold}
              {ipaeThreshold}
              {motifRmsdThreshold}
              mutationHighlight={activePaperMutationHighlight}
              {ipsaeThreshold}
              {isRunning}
              hasScorerResult={Boolean(latestScorerHit && latestScorerHit.nodeId !== selectedNode?.id)}
              onSelectSlot={handleSelectPeekSlot}
              onSelectRow={(idx) => (selectedRowIdx = idx)}
              onSelectFold={(idx) => (selectedFoldIdx = idx)}
              onSelectDesignKey={(key) => {
                selectedDesignKey = key;
                if (key === 'all' && multiDesignViewMode === 'single') {
                  multiDesignViewMode = 'grid';
                }
              }}
              onRunUpToNode={(id) => runWorkflow(false, id)}
              onJumpToScorer={() => {
                if (latestScorerHit) {
                  setSingleSelection(latestScorerHit.nodeId);
                  selectedPeekSlotId = null;
                  selectedEdgeId = null;
                  rightPanelTab = 'peek';
                }
              }}
              onDownloadText={downloadText}
              onSelectResidue={handleSelectPerResidue}
            />
          {/if}
        </div>
        {/if}
      </div>
    </section>
  </div>

{#if pendingRun}
  <FirstRunModal
    preflight={pendingRun.preflight}
    webgpuSupported={webgpuSupported !== false}
    {webgpuDetail}
    isPartialRun={Boolean(pendingRun.upToNodeId)}
    onConfirm={confirmPendingRun}
    onCancel={cancelPendingRun}
  />
{/if}

  <ExamplesGalleryModal
    open={showExamplesGallery}
    activeId={selectedPresetId}
    {isRunning}
    onClose={() => (showExamplesGallery = false)}
    onSelectPaper={selectPaperInsight}
    onSelectPreset={selectPreset}
  />

{#if showOnboarding}
  <OnboardingModal
    {graph}
    {isRunning}
    onDismiss={dismissOnboarding}
    onPrepareTourStarter={prepareTourStarter}
    onAutoStepBlock={autoStepTourBlock}
    onInteractiveChange={(active) => (onboardingInteractive = active)}
  />
{/if}

{#if activePaperInsight && showPaperWalkthrough && !showOnboarding && !showExamplesGallery && !pendingRun && !showCredits}
  <PaperWalkthroughOverlay
    example={activePaperInsight}
    stepIdx={paperWalkthroughIdx}
    hasOutputs={hasActivePaperOutputs}
    {isRunning}
    canRun={!isRunning && validationErrors.length === 0 && !hasInvalidNumeric}
    onSelectStep={(idx) => applyPaperWalkthroughStep(activePaperInsight!, idx)}
    onRun={() => void runWorkflow(false)}
    onDismiss={() => (showPaperWalkthrough = false)}
  />
{/if}

{#if showCredits}
  <CreditsModal onClose={() => (showCredits = false)} />
{/if}

<style>
  .app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    min-height: 46px;
    padding: 4px 0 10px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .hdr-left {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    min-width: 0;
    max-width: 100%;
  }
  .hdr-right {
    display: flex;
    align-items: center;
    gap: 7px;
    flex-wrap: wrap;
    max-width: 100%;
  }
  .brand-logo {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.4px;
    color: var(--text-primary);
    white-space: nowrap;
    display: inline-flex;
    align-items: center;
  }
  .brand-logo-btn {
    margin: 0;
    padding: 4px 6px;
    margin-left: -6px;
    border: 1px solid transparent;
    border-radius: 6px;
    background: transparent;
    color: inherit;
    font: inherit;
    line-height: 1;
    letter-spacing: inherit;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: opacity 0.12s ease, background 0.12s ease, border-color 0.12s ease;
  }
  .brand-wordmark {
    display: inline-flex;
    align-items: baseline;
    line-height: 1;
  }
  .brand-logo-btn:hover:not(:disabled) {
    background: var(--bg-surface-hover);
  }
  .brand-logo-btn.confirm-danger {
    border-color: var(--err-border);
    background: var(--err-bg);
    padding: 4px 8px;
  }
  .brand-confirm-pill {
    display: inline-flex;
    align-items: center;
    line-height: 1;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0;
    color: var(--err-strong);
  }
  .brand-logo-btn:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .brand-logo-btn:disabled {
    cursor: default;
  }
  .accent {
    color: var(--accent-bright);
  }

  .gallery-trigger {
    display: inline-flex;
    align-items: center;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    overflow: hidden;
    min-width: 0;
    max-width: min(390px, 100%);
    padding: 0;
    cursor: pointer;
    color: var(--text-body);
    font-family: inherit;
    transition: border-color 120ms ease, box-shadow 120ms ease;
  }
  .gallery-trigger:hover:not(:disabled) {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.14);
  }
  .gallery-trigger:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  .gallery-trigger-lbl {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--text-dim);
    background: var(--bg-surface-2);
    border-right: 1px solid var(--border);
    padding: 6px 9px;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .gallery-trigger-val {
    padding: 5px 8px 5px 10px;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .gallery-trigger-chev {
    padding-right: 9px;
    font-size: 11px;
    color: var(--text-dim);
    flex-shrink: 0;
  }

  .walkthrough-reopen-btn {
    background: transparent;
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .walkthrough-reopen-btn:hover {
    border-color: var(--accent);
    background: var(--bg-pill);
    color: var(--accent-bright);
  }

  .pop-wrap {
    position: relative;
  }
  .hdr-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-surface);
    border: 1px solid var(--border);
    padding: 5px 10px;
    border-radius: 999px;
    cursor: pointer;
    white-space: nowrap;
  }
  .hdr-pill:hover {
    border-color: var(--border-strong);
    color: var(--text-primary);
  }
  .hdr-pill.ok {
    color: var(--ok-text);
    border-color: var(--ok-border);
    background: var(--ok-bg);
  }
  .hdr-pill.warn {
    color: var(--warn-text);
    border-color: var(--warn-border);
    background: var(--warn-bg);
  }
  .hdr-pill.bad {
    color: var(--err-strong);
    border-color: var(--err-border);
    background: var(--err-bg);
  }
  .hdr-pill.gpu {
    color: var(--accent-soft);
    border-color: var(--accent);
    background: var(--bg-pill);
  }
  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
  }

  .hdr-icon-btn {
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    width: 30px;
    height: 30px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    cursor: pointer;
    padding: 0;
    text-decoration: none;
  }
  .hdr-icon-btn:hover,
  .hdr-icon-btn.active {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
    border-color: var(--accent);
  }

  .hdr-share-btn {
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    height: 30px;
    padding: 0 9px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  .hdr-share-btn:hover,
  .hdr-share-btn.active {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
    border-color: var(--accent);
  }

  .hdr-gh-btn {
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    height: 30px;
    padding: 0 9px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    font-weight: 600;
    text-decoration: none;
    white-space: nowrap;
  }
  .hdr-gh-btn:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
    border-color: var(--accent);
  }
  @media (max-width: 980px) {
    .hdr-gh-label {
      display: none;
    }
    .hdr-gh-btn {
      padding: 0;
      width: 30px;
      justify-content: center;
    }
  }

  .share-pop {
    width: 310px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .share-pop-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .share-pop-title {
    font-size: 12.5px;
    font-weight: 700;
    color: var(--text-primary);
  }
  .share-pop-meta {
    font-size: 10.5px;
    color: var(--text-muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .share-pop-note {
    margin: 0;
    font-size: 11.5px;
    line-height: 1.45;
    color: var(--text-secondary);
    background: var(--bg-muted, rgba(148, 163, 184, 0.08));
    border: 1px solid var(--border-subtle, var(--border-strong));
    border-radius: 6px;
    padding: 7px 9px;
  }
  .share-pop-warn {
    margin: 0;
    font-size: 11px;
    line-height: 1.4;
    color: #d97706;
  }
  .share-pop-url-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .share-pop-input {
    flex: 1;
    min-width: 0;
    height: 28px;
    padding: 0 8px;
    font-size: 11px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--text-primary);
    background: var(--bg-canvas, var(--bg-surface));
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    outline: none;
  }
  .share-pop-input:focus {
    border-color: var(--accent);
  }
  .share-pop-copy {
    height: 28px;
    padding: 0 10px;
    border-radius: 6px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: #ffffff;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  .share-pop-copy.copied {
    background: #10b981;
    border-color: #10b981;
  }
  .share-pop-foot {
    display: flex;
    justify-content: flex-end;
    padding-top: 2px;
  }
  .share-pop-link-btn {
    background: none;
    border: none;
    padding: 0;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    cursor: pointer;
  }
  .share-pop-link-btn:hover {
    color: var(--accent);
    text-decoration: underline;
  }
  .menu-link {
    text-decoration: none;
    box-sizing: border-box;
  }
  .error-banner-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
  }
  .error-banner-text {
    flex: 1;
    min-width: 0;
  }
  .error-report-btn {
    flex-shrink: 0;
    padding: 4px 10px;
    border-radius: 5px;
    border: 1px solid currentColor;
    color: inherit;
    font-size: 11.5px;
    font-weight: 600;
    text-decoration: none;
    white-space: nowrap;
    opacity: 0.9;
  }
  .error-report-btn:hover {
    opacity: 1;
    background: rgba(239, 68, 68, 0.14);
  }

  .hdr-popover {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: 60;
    max-width: calc(100vw - 24px);
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35), var(--node-shadow);
  }
  .sys-pop {
    width: min(300px, calc(100vw - 24px));
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 11.5px;
  }
  .pop-divider {
    height: 1px;
    background: var(--border-subtle);
    margin: 1px 0;
  }
  .pop-compute-sec {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }
  .compute-seg {
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: var(--bg-surface-2);
  }
  .compute-seg-btn {
    border: none;
    background: transparent;
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 600;
    padding: 3px 7px;
    cursor: pointer;
    font-family: inherit;
  }
  .compute-seg-btn:hover:not(:disabled) {
    color: var(--text-primary);
  }
  .compute-seg-btn.active {
    background: var(--accent);
    color: #fff;
  }
  .pop-field {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 10px;
    color: var(--text-muted);
  }
  .pop-field input,
  .pop-thresh,
  .pop-select {
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 6px;
    font-size: 11px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .pop-select {
    font-size: 10.5px;
    padding: 3px 6px;
    max-width: 138px;
    cursor: pointer;
  }
  .pop-thresh {
    width: 66px;
    text-align: right;
  }
  .sys-pop.align-left {
    left: 0;
    right: auto;
  }
  @media (max-width: 900px) {
    .sys-pop {
      left: 0;
      right: auto;
      width: min(260px, calc(100vw - 24px));
    }
  }
  @media (max-width: 640px) {
    .hdr-left {
      width: 100%;
    }
    .gallery-trigger {
      display: flex;
      flex: 1 1 100%;
      width: 100%;
      min-width: 0;
      max-width: 100%;
    }
    .gallery-trigger-lbl {
      padding: 6px 7px;
      font-size: 9.5px;
    }
    .gallery-trigger-val {
      flex: 1;
    }
  }
  .pop-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }
  .pop-k {
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
  .pop-v {
    color: var(--text-primary);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    text-align: right;
  }
  .pop-v.ok-t {
    color: var(--ok-text);
  }
  .pop-v.warn-t {
    color: var(--warn-text);
  }
  .pop-v.bad-t {
    color: var(--err-strong);
  }
  .pop-warn-note {
    font-size: 10.5px;
    line-height: 1.4;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 6px;
    padding: 6px 8px;
  }
  .pop-warn-note code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
  }
  .badge-ok {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--ok-text);
    background: var(--ok-bg);
    border: 1px solid var(--ok-border);
    padding: 1px 7px;
    border-radius: 999px;
  }
  .pop-action {
    background: var(--bg-surface-2);
    color: var(--text-body);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 3px 8px;
    font-size: 11px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    cursor: pointer;
  }
  .pop-action:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }
  .pop-action.danger:hover:not(:disabled),
  .pop-action.confirm-danger {
    border-color: var(--err-border);
    color: var(--err-strong);
    background: var(--err-bg);
    font-weight: 600;
  }
  .pop-action:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .menu-pop {
    min-width: 178px;
    padding: 5px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .menu-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    box-sizing: border-box;
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 12px;
    color: var(--text-body);
    cursor: pointer;
    font-family: inherit;
  }
  .menu-item:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }
  .menu-item:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .menu-sep {
    height: 1px;
    background: var(--border-subtle);
    margin: 3px 4px;
  }

  .btn-cancel {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--err-bg);
    color: var(--err-strong);
    border: 1px solid var(--err-border);
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  .btn-cancel:hover {
    background: var(--err-strong);
    color: #fff;
  }
  .btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #2563eb;
    color: #fff;
    border: 1px solid #3b82f6;
    padding: 6px 13px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  .run-kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.85);
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 3px;
    padding: 0 4px;
    line-height: 1.4;
  }
  .btn-primary:hover:not(:disabled) {
    background: #1d4ed8;
  }
  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn-primary.is-running:disabled {
    opacity: 0.9;
    background: #b45309;
    border-color: #f59e0b;
  }
  .run-pulse {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #fde68a;
    animation: run-dot-pulse 1.2s ease-in-out infinite;
    flex-shrink: 0;
  }
  @keyframes run-dot-pulse {
    0%,
    100% {
      opacity: 0.45;
      transform: scale(0.85);
    }
    50% {
      opacity: 1;
      transform: scale(1.15);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .run-pulse {
      animation: none;
    }
  }

  .workbench-split {
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 100%;
    min-width: 0;
    max-width: 100%;
  }
  .workbench-split.horizontal {
    flex-direction: row;
    align-items: stretch;
    gap: 0;
    height: calc(100dvh - 76px);
    min-height: 620px;
  }
  .workbench-split.vertical {
    flex-direction: column;
    align-items: stretch;
    gap: 0;
    height: calc(100dvh - 76px);
    min-height: 640px;
  }
  .workbench-split.dragging {
    user-select: none;
    cursor: col-resize;
  }
  .workbench-split.vertical.dragging {
    cursor: row-resize;
  }
  .workbench-split.horizontal > .flow-panel {
    width: var(--left-pane, 65%);
    flex-shrink: 0;
    min-width: 0;
    margin-bottom: 0;
    height: 100%;
  }
  .workbench-split.vertical > .flow-panel {
    height: var(--top-pane, 48%);
    flex-shrink: 0;
    min-height: 220px;
    margin-bottom: 0;
  }
  .workbench-split.vertical > .results-grid {
    flex: 1;
    min-height: 220px;
    overflow-y: auto;
    align-content: start;
    padding-top: 2px;
  }
  .drawer-scroll {
    max-height: 240px;
    overflow-y: auto;
  }
  .split-handle {
    width: 12px;
    flex-shrink: 0;
    cursor: col-resize;
    display: flex;
    align-items: center;
    justify-content: center;
    touch-action: none;
  }
  .split-handle.is-vertical {
    width: 100%;
    height: 12px;
    cursor: row-resize;
  }
  .split-bar {
    width: 3px;
    height: 44px;
    border-radius: 999px;
    background: var(--border-strong);
    transition: background 0.15s ease, height 0.15s ease, width 0.15s ease;
  }
  .split-handle.is-vertical .split-bar {
    width: 56px;
    height: 3px;
  }
  .split-handle:hover .split-bar,
  .workbench-split.dragging .split-bar {
    background: var(--accent);
    height: 72px;
  }
  .split-handle.is-vertical:hover .split-bar,
  .workbench-split.vertical.dragging .split-handle.is-vertical .split-bar {
    background: var(--accent);
    width: 88px;
    height: 3px;
  }

  .flow-panel {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 100%;
    min-width: 0;
    max-width: 100%;
    margin-bottom: 0;
  }
  .canvas-top-spine {
    position: absolute;
    top: 46px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 40;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    max-width: min(640px, calc(100% - 24px));
    pointer-events: none;
  }
  .share-toast {
    pointer-events: auto;
    max-width: 100%;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--ok-text);
    background: var(--ok-bg);
    border: 1px solid var(--ok-border);
    padding: 6px 10px;
    border-radius: 6px;
  }
  .share-toast.is-error {
    color: var(--err-text);
    background: var(--err-bg);
    border-color: var(--err-border);
  }
  .draft-preview-banner {
    pointer-events: auto;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--bg-surface) 94%, transparent);
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
    border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--border-strong));
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
    font-size: 11.5px;
    color: var(--text-secondary);
    white-space: nowrap;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .draft-preview-banner strong {
    color: var(--text-primary);
    font-weight: 600;
  }
  .dpb-spark {
    color: var(--accent-bright);
    flex-shrink: 0;
  }
  .ai-draft-warning {
    pointer-events: auto;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px 5px 6px;
    border-radius: 10px;
    border: 1px solid rgba(251, 146, 60, 0.42);
    background: color-mix(in srgb, #f97316 12%, color-mix(in srgb, var(--bg-surface) 95%, transparent));
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 14px rgba(234, 88, 12, 0.14);
    font-size: 11.5px;
    line-height: 1.35;
    color: var(--text-body);
    max-width: 100%;
  }
  .ai-draft-warning.is-ai {
    border-color: rgba(249, 115, 22, 0.62);
    background: color-mix(in srgb, #ea580c 16%, color-mix(in srgb, var(--bg-surface) 95%, transparent));
  }
  .adw-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 6px;
    background: rgba(249, 115, 22, 0.18);
    border: 1px solid rgba(251, 146, 60, 0.36);
    color: #fdba74;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .adw-badge.is-ai {
    background: rgba(234, 88, 12, 0.24);
    border-color: rgba(251, 146, 60, 0.5);
    color: #fed7aa;
  }
  .adw-copy {
    color: var(--text-body);
  }
  .adw-dismiss {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 6px;
    background: none;
    border: none;
    color: inherit;
    opacity: 0.68;
    cursor: pointer;
    font-size: 14px;
    line-height: 1;
    padding: 0;
    flex-shrink: 0;
    transition: opacity 0.12s ease, background 0.12s ease;
  }
  .adw-dismiss:hover {
    opacity: 1;
    background: rgba(249, 115, 22, 0.16);
  }
  :global([data-theme='light']) .ai-draft-warning {
    border-color: #fdba74;
    background: rgba(255, 247, 237, 0.94);
    box-shadow: 0 4px 14px rgba(234, 88, 12, 0.1);
  }
  :global([data-theme='light']) .ai-draft-warning.is-ai {
    border-color: #fb923c;
    background: rgba(255, 237, 213, 0.96);
  }
  :global([data-theme='light']) .adw-badge {
    background: #ffedd5;
    border-color: #fdba74;
    color: #c2410c;
  }
  :global([data-theme='light']) .adw-badge.is-ai {
    background: #fed7aa;
    border-color: #fb923c;
    color: #9a3412;
  }
  :global([data-theme='light']) .adw-copy,
  :global([data-theme='light']) .adw-dismiss {
    color: #7c2d12;
  }
  .studio-brush-tag {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    color: var(--accent-soft);
    background: var(--bg-pill);
    border: 1px solid color-mix(in srgb, var(--accent) 42%, var(--border-strong));
    padding: 2px 8px;
    border-radius: 999px;
    white-space: nowrap;
  }
  .warn-banner {
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .bio-banner-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .bio-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 11.5px;
    line-height: 1.4;
  }
  .bio-banner.is-err {
    background: var(--err-bg);
    border-color: var(--err-border);
    color: var(--err-text);
  }
  .bio-msg {
    flex: 1 1 240px;
  }
  .bio-fix-btn {
    flex-shrink: 0;
    background: var(--bg-surface);
    color: var(--text-primary);
    border: 1px solid var(--warn-border);
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    white-space: nowrap;
  }
  .bio-fix-btn:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--accent-bright);
  }
  .bio-fix-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .error-banner {
    background: var(--err-bg);
    border: 1px solid var(--err-border);
    color: var(--err-text);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    white-space: pre-wrap;
  }
  .error-banner.dismissible {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }
  .banner-dismiss {
    flex-shrink: 0;
    background: transparent;
    border: 1px solid var(--err-border);
    color: var(--err-text);
    border-radius: 4px;
    width: 22px;
    height: 22px;
    line-height: 1;
    font-size: 14px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .banner-dismiss:hover {
    background: rgba(255, 255, 255, 0.08);
  }

  .results-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
    gap: 14px;
    align-items: start;
    width: 100%;
    min-width: 0;
    max-width: 100%;
  }
  .results-grid > * {
    min-width: 0;
    max-width: 100%;
  }
  .results-grid.stacked-pane {
    flex: 1;
    min-width: 0;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 10px;
    overflow-y: auto;
    padding-right: 2px;
  }
  .results-grid.stacked-pane > * {
    width: 100%;
    min-width: 0;
  }
  .results-grid.stacked-pane .right-panel {
    flex: 1;
    min-height: 240px;
  }
  .results-grid.stacked-pane .right-body {
    max-height: none;
    flex: 1;
    overflow-y: auto;
  }
  @media (max-width: 1024px) {
    .workbench-split.horizontal,
    .workbench-split.vertical {
      flex-direction: column;
      height: auto;
      min-height: 0;
      gap: 12px;
    }
    .workbench-split.horizontal > .flow-panel,
    .workbench-split.vertical > .flow-panel {
      width: 100%;
      height: 360px;
      min-height: 300px;
    }
    .workbench-split.vertical > .results-grid {
      overflow: visible;
      min-height: 0;
    }
    .split-handle {
      display: none;
    }
    .results-grid.stacked-pane {
      height: auto;
      overflow: visible;
    }
    .results-grid.stacked-pane .right-body {
      max-height: 520px;
    }
  }
  @media (max-width: 960px) {
    .results-grid {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .viewer-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--text-muted);
  }
  .viewer-bar label {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
  }
  .viewer-stage {
    min-height: 0;
  }
  .expand-backdrop {
    position: fixed;
    inset: 0;
    z-index: 155;
    background: rgba(10, 14, 24, 0.58);
    -webkit-backdrop-filter: blur(4px);
    backdrop-filter: blur(4px);
  }
  .viewer-col.is-expanded,
  .right-panel.is-expanded {
    position: fixed;
    inset: 16px;
    z-index: 160;
    display: flex;
    flex-direction: column;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.52);
    width: auto !important;
    max-width: none !important;
    height: auto !important;
  }
  .viewer-col.is-expanded .viewer-stage {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .viewer-col.is-expanded .viewer-stage > :global(.mol-card) {
    flex: 1;
    height: 100% !important;
  }
  .right-panel.is-expanded {
    padding: 0;
    overflow: hidden;
  }
  .right-panel.is-expanded .right-body {
    flex: 1;
    max-height: none !important;
    overflow-y: auto;
  }
  .results-grid.is-studio-open {
    position: fixed;
    inset: 16px;
    width: auto !important;
    height: auto !important;
    z-index: 160;
    display: grid !important;
    grid-template-columns: minmax(340px, 46%) minmax(420px, 1fr);
    grid-template-rows: minmax(0, 1fr);
    align-items: stretch;
    gap: 12px;
    padding: 12px;
    box-sizing: border-box;
    background: var(--bg-canvas);
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    box-shadow: 0 28px 72px rgba(0, 0, 0, 0.58);
    overflow: hidden !important;
  }
  .results-grid.is-studio-open > .viewer-col {
    display: flex;
    flex-direction: column;
    width: auto;
    height: 100%;
    min-width: 0;
    min-height: 0;
    box-sizing: border-box;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px;
    overflow: hidden;
  }
  .results-grid.is-studio-open > .viewer-col .viewer-stage {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .results-grid.is-studio-open > .viewer-col .viewer-stage > :global(.mol-card) {
    flex: 1;
    height: 100% !important;
  }
  .results-grid.is-studio-open > .right-panel {
    display: flex;
    flex-direction: column;
    width: auto;
    height: 100%;
    min-width: 0;
    min-height: 0;
    box-sizing: border-box;
    overflow: hidden;
    border-radius: 10px;
  }
  @media (max-width: 1024px) {
    .results-grid.is-studio-open {
      inset: 0;
      border-radius: 0;
      padding: 8px;
      gap: 8px;
      grid-template-columns: 1fr;
      grid-template-rows: 220px minmax(0, 1fr);
    }
  }
  .studio-launch-btn {
    margin-left: 0;
    color: var(--accent);
    border-color: color-mix(in srgb, var(--accent) 45%, var(--border-strong));
    flex-shrink: 0;
  }
  .pane-expand-btn {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 5px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  .pane-expand-btn:hover,
  .pane-expand-btn.active {
    border-color: var(--accent);
    color: var(--text-primary);
    background: var(--bg-pill);
  }
  .tab-expand-btn {
    margin-left: 0;
    padding: 2px 6px;
    flex-shrink: 0;
  }
  .pick-mode {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: var(--text-secondary);
    font-size: 11.5px;
  }
  .pick-select {
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    cursor: pointer;
  }
  .slider {
    flex: 1;
    min-width: 100px;
    accent-color: var(--accent);
  }
  .picked {
    margin-left: auto;
    font-size: 11px;
    color: var(--accent-soft);
  }
  .seg-group {
    display: inline-flex;
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    overflow: hidden;
    background: var(--bg-surface);
  }
  .seg-btn {
    border: none;
    background: transparent;
    color: var(--text-muted);
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    font-family: inherit;
  }
  .seg-btn:not(:last-child) {
    border-right: 1px solid var(--border-subtle);
  }
  .seg-btn.active {
    background: var(--accent);
    color: #fff;
    font-weight: 600;
  }
  .seg-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .cmp-exit-mini {
    margin-left: auto;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-secondary);
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 11px;
    cursor: pointer;
  }
  .cmp-exit-mini:hover {
    color: var(--text-primary);
    border-color: var(--accent);
  }
  .cmp-launch-btn {
    border: 1px solid var(--accent);
    background: var(--bg-pill);
    color: var(--accent-soft);
    border-radius: 999px;
    padding: 2px 9px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .cmp-launch-btn:hover {
    background: var(--accent);
    color: #fff;
  }
  .geom-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 16px;
    margin-top: 8px;
    padding: 7px 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .viewer-bar label.disabled-opt {
    opacity: 0.7;
    cursor: default;
  }
  .pick-target-tag {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--accent-soft);
    max-width: 130px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .pick-target-tag.muted {
    color: var(--text-dim);
  }
  .geom-lbl {
    font-weight: 700;
    color: var(--text-secondary);
  }
  .geom-strip b {
    color: var(--text-primary);
  }

  /* Unified Tabbed Right Panel */
  .right-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    min-width: 0;
    overflow: hidden;
  }
  .right-tabs {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 4px 8px;
    min-width: 0;
    padding: 6px 10px 0;
    border-bottom: 1px solid var(--border);
    background: var(--bg-surface-2);
  }
  .r-tab-group {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    flex-shrink: 0;
  }
  .r-tab-actions {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 4px 6px;
    min-width: 0;
    max-width: 100%;
    padding-bottom: 4px;
  }
  .r-tab {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-muted);
    padding: 7px 12px 8px;
    font-size: 12.5px;
    font-weight: 500;
    cursor: pointer;
    font-family: inherit;
    white-space: nowrap;
    margin-bottom: -1px;
  }
  .r-tab:hover {
    color: var(--text-primary);
  }
  .r-tab.active {
    color: var(--text-primary);
    font-weight: 600;
    border-bottom-color: var(--accent);
  }
  .live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ok-text);
    box-shadow: 0 0 0 2px var(--ok-bg);
  }
  .tab-node-ctx {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    line-height: 1;
    color: var(--text-dim);
    min-width: 0;
    flex: 0 1 auto;
    max-width: 160px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .r-tab-actions > :global(.mc-pill) {
    flex-shrink: 0;
  }
  .r-tab-actions .pane-expand-btn {
    margin-left: 0;
    flex-shrink: 0;
  }
  .right-body {
    max-height: 520px;
    min-width: 0;
    width: 100%;
    overflow-y: auto;
  }
  .empty-inspector {
    padding: 28px 16px;
    text-align: center;
    font-size: 12px;
    color: var(--text-dim);
  }
  code,
  .mono {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
</style>
