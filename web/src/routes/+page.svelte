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
  import FirstRunModal from '$lib/ui/FirstRunModal.svelte';
  import ModelCardPopover from '$lib/ui/ModelCardPopover.svelte';
  import OnboardingModal from '$lib/ui/OnboardingModal.svelte';
  import NodeInspector from '$lib/ui/NodeInspector.svelte';
  import NodeStudioPopover from '$lib/ui/NodeStudioPopover.svelte';
  import { hasInvalidNumericInputs } from '$lib/ui/NumericInput.svelte';
  import WorkflowCanvas from '$lib/ui/WorkflowCanvas.svelte';
  import RecipeDrawer from '$lib/ui/RecipeDrawer.svelte';
  import {
    classifyParamUndoAction,
    computeBenchmarkAutoSeedPatches,
    computeChangedParamKeys,
    computePreviewChecks,
    isSharedWorkflowHash,
    parseLengthFromContig,
    shouldAllowNativeTextUndo,
    shouldSuppressOnboardingForHash,
    type LabelUndoMarker,
    type ParamUndoMarker,
  } from '$lib/ui/page-state-helpers';
  import { compileNodeInputPreview } from '$lib/engine/input-preview';
  import {
    formatWebGpuAdapterSummary,
    requestResilientWebGpuAdapter,
    type WebGpuAdapterInfoSnapshot,
  } from '$lib/engine/webgpu-device';

  const scheduler = new ExecutionScheduler({
    checkpointStore: createIndexedDbCheckpointStore(),
  });

  let rightPanelTab = $state<'params' | 'peek'>('params');
  let expandedPane = $state<'none' | 'viewer' | 'inspector' | 'studio'>('none');
  let studioTab = $state<'input' | 'controls' | 'output'>('input');
  let studioHoveredToken = $state<string | null>(null);
  let studioFocusToken = $state<string | null>(null);
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

  // Header popovers & first-run confirmation dialog
  let sysMenuOpen = $state(false);
  let sysPopAlignLeft = $state(false);
  let moreMenuOpen = $state(false);
  let sysWrapEl: HTMLDivElement | null = $state(null);
  let moreWrapEl: HTMLDivElement | null = $state(null);
  let importFileInput: HTMLInputElement | null = $state(null);
  let firstRunAck = $state(false);
  let showOnboarding = $state(false);
  let onboardingInteractive = $state(false);
  let canvasFitViewTrigger = $state(0);
  let showSeqParentBackbone = $state(false);
  let pendingRun = $state<{
    bypassCache: boolean;
    upToNodeId?: string;
    preflight: RunPreflightSummary;
  } | null>(null);

  // Active workflow state
  let selectedPresetId = $state(WORKFLOW_PRESETS[0].id);
  let graph = $state<WorkflowGraph>(autoLayoutGraph(structuredClone(WORKFLOW_PRESETS[0].graph)));
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
  let isCompareMode = $derived(selectedNodeIds.length >= 2);
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

  function resetToHomeCanvas() {
    if (isRunning) return;
    sysMenuOpen = false;
    moreMenuOpen = false;
    showRecipe = false;
    expandedPane = 'none';
    hashError = null;
    if (typeof window !== 'undefined' && window.location.hash) {
      window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`);
    }
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
    if (sysMenuOpen && !sysWrapEl?.contains(target)) {
      sysMenuOpen = false;
    }
    if (moreMenuOpen && !moreWrapEl?.contains(target)) {
      moreMenuOpen = false;
    }
  }

  function isEditableElement(el: EventTarget | null): boolean {
    if (!(el instanceof HTMLElement)) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
  }

  function handleWindowKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      if (expandedPane !== 'none') {
        expandedPane = 'none';
        return;
      }
      if (showCredits) {
        showCredits = false;
        return;
      }
      sysMenuOpen = false;
      moreMenuOpen = false;
      return;
    }
    if (pendingRun || showCredits || (showOnboarding && !onboardingInteractive)) return;

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
      if (!isRunning && validationErrors.length === 0) {
        ev.preventDefault();
        void runWorkflow(false);
      }
      return;
    }

    // Space toggles Node Studio when a block is selected and not typing in an input
    if (
      ev.key === ' ' &&
      !ev.metaKey &&
      !ev.ctrlKey &&
      !ev.altKey &&
      !isEditableElement(ev.target) &&
      !isEditableElement(document.activeElement)
    ) {
      if (selectedNode) {
        ev.preventDefault();
        if (expandedPane === 'studio') {
          expandedPane = 'none';
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
        loadFullWorkflowGraph(decoded, { clearOutputs: true });
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
    if (!activeRunState.leaseHeld) {
      void scheduler.releaseActiveGpu();
    }
  });

  async function refreshCacheStats() {
    const st = await getCacheStats();
    cacheUsedMB = st.usedBytes / 1024 / 1024;
  }

  function handleProgress(ev: NodeProgressEvent) {
    if (webgpuSupported === false && (ev.status === 'running' || ev.status === 'done')) {
      const node = graph.nodes.find((n) => n.id === ev.nodeId);
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
    graphUndoStack = [];
    graphRedoStack = [];
    lastParamUndo = null;
    lastLabelUndo = null;
    hashError = null;
    selectedPresetId = presetId;
    const g = structuredClone(preset.graph);
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
    const laid = autoLayoutGraph(nextGraph);
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
    const slot = peekSlots.find((s) => s.id === slotId);
    selectedEdgeId = slot?.direction === 'in' ? (slot.edgeId ?? null) : null;
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

  function handleApplyPlannedGraph(next: WorkflowGraph, summary: string) {
    if (isRunning) return;
    const prevPresetId = selectedPresetId;
    hashError = null;
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
    const nextById = new Map(next.nodes.map((n) => [n.id, n]));
    const prevById = new Map(prevSnapshot.nodes.map((n) => [n.id, n]));

    const dirtySeeds = new Set<string>();
    for (const [id, cur] of nextById.entries()) {
      const prev = prevById.get(id);
      if (!prev || cur.adapterId !== prev.adapterId) {
        dirtySeeds.add(id);
      } else if (!shallowParamsEqual(cur.params, prev.params)) {
        dirtySeeds.add(id);
      }
    }

    const prevEdgeKeys = new Set(
      prevSnapshot.edges.map((e) => makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort)),
    );
    const nextEdgeKeys = new Set(
      next.edges.map((e) => makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort)),
    );
    for (const e of next.edges) {
      if (!prevEdgeKeys.has(makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort))) {
        dirtySeeds.add(e.toNode);
      }
    }
    for (const e of prevSnapshot.edges) {
      if (!nextEdgeKeys.has(makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort)) && nextById.has(e.toNode)) {
        dirtySeeds.add(e.toNode);
      }
    }

    const staleIds = getDownstreamNodeIds(next, dirtySeeds);

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
      next.nodes.length === 0 && prevSnapshot.nodes.length > 0
        ? {
            ...next,
            id: 'custom-workflow',
            name: 'Custom Workflow',
            description: '',
          }
        : next;
    if (next.nodes.length === 0 && prevSnapshot.nodes.length > 0) {
      selectedPresetId = 'custom-workflow';
    }

    graph = normalizedNext;
    scheduler.invalidateEditedGraph(prevSnapshot, normalizedNext);

    // Prune deleted node IDs from selection & hidden compare models (resetting if all remaining would be hidden)
    let nextSelected = selectedNodeIds;
    if (selectedNodeIds.length > 0) {
      const kept = selectedNodeIds.filter((id) => nextById.has(id));
      if (kept.length !== selectedNodeIds.length) {
        nextSelected = kept.length > 0 ? kept : next.nodes[0]?.id ? [next.nodes[0].id] : [];
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
    if (paramsChanged) {
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
    const nextGraph: WorkflowGraph = {
      ...prevSnapshot,
      nodes: prevSnapshot.nodes.map((n) => (n.id === nodeId ? { ...n, params } : n)),
    };
    graph = nextGraph;
    if (paramsChanged) {
      scheduler.invalidateEditedGraph(prevSnapshot, nextGraph);
      invalidateNodes(getDownstreamNodeIds(nextGraph, [nodeId]));
    }
  }

  function updateNodeLabel(nodeId: string, label: string) {
    const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
    const prevNode = prevSnapshot.nodes.find((n) => n.id === nodeId);
    if (prevNode?.label === label) return;
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
  let previewChecks = $derived(
    computePreviewChecks(graph, activeDraftPreview, outputsByNode, inputPreviewErrorsByNode),
  );
  let displayedValidationErrors = $derived(previewChecks.displayedValidationErrors);
  let bioIssues = $derived(previewChecks.bioIssues);

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

  // Auto-seed unconfigured (hasNoTaskParams) or previously auto-seeded Genie 3 & RFD4-Proteina blocks
  // when wired to a curated benchmark PDB Input on their unified `structure` port, adapting the benchmark
  // contig to preserve custom/planned lengths (e.g. 65 aa on 5O45, 80 aa on 1YCR) without pushing undo.
  $effect(() => {
    if (isRunning) return;
    const patches = computeBenchmarkAutoSeedPatches(graph);
    if (patches) {
      const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
      const nextGraph: WorkflowGraph = {
        ...prevSnapshot,
        nodes: prevSnapshot.nodes.map((n) => (patches[n.id] ? { ...n, params: patches[n.id] } : n)),
      };
      graph = nextGraph;
      scheduler.invalidateEditedGraph(prevSnapshot, nextGraph);
      invalidateNodes(getDownstreamNodeIds(nextGraph, Object.keys(patches)));
    }
  });

  function getSiblingGeneratorIds(nodeId: string): string[] {
    const structEdge = graph.edges.find(
      (e) => e.toNode === nodeId && (e.toPort === 'structure' || e.toPort === 'template'),
    );
    if (!structEdge) return [];
    return graph.nodes
      .filter(
        (n) =>
          n.id !== nodeId &&
          (n.adapterId === 'genie3' || n.adapterId === 'rfd4-proteina') &&
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
      src.adapterId === 'genie3'
        ? String(sp.taskMode ?? 'unconditional')
        : String(sp.mode ?? 'unconditional');
    const parsedLen = parseLengthFromContig(String(sp.contig ?? ''));
    const syncLen = Number(sp.length ?? parsedLen ?? (src.adapterId === 'genie3' ? 80 : 70));
    const unsupportedGenie3Labels: string[] = [];

    const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
    pushGraphUndo(prevSnapshot);
    const nextGraph: WorkflowGraph = {
      ...prevSnapshot,
      nodes: prevSnapshot.nodes.map((n) => {
        if (!siblingIds.has(n.id)) return n;
        const shared = {
          length: syncLen,
          contig: String(sp.contig ?? ''),
          lockChain: String(sp.lockChain ?? ''),
          hotspots: String(sp.hotspots ?? ''),
          motifSpan: String(sp.motifSpan ?? ''),
          userEditedTask: true,
        };
        if (n.adapterId === 'genie3') {
          const isSupported = srcMode === 'binder' || srcMode === 'motif' || srcMode === 'unconditional';
          if (!isSupported) {
            unsupportedGenie3Labels.push(n.label ?? 'Genie 3');
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
    if (unsupportedGenie3Labels.length > 0) {
      showToastMessage(
        `${unsupportedGenie3Labels.join(', ')} does not support RFD4 "${srcMode}" mode — reset to unconditional (${syncLen} aa) and cleared incompatible constraints.`,
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
    const nextGraph: WorkflowGraph = {
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
    graph = nextGraph;
    scheduler.invalidateEditedGraph(prevSnapshot, nextGraph);
    invalidateNodes(getDownstreamNodeIds(nextGraph, [nodeId]));
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
    if (!confirmed) {
      const preflight = estimateRunRequirements(targetGraph);
      const hasRemoteMsa = (preflight.remoteMsaTransfers?.length ?? 0) > 0;
      if (hasRemoteMsa || (!firstRunAck && (preflight.usesWebGpu || preflight.totalDownloadMB > 0))) {
        pendingRun = { bypassCache, upToNodeId, preflight };
        return;
      }
    }
    localRunning = true;
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
        // Auto-focus the terminal output block (scorer -> fold -> generator -> last executed node)
        // so its 3D structure appears in the viewport immediately without needing an scRMSD block.
        const scorersWithData = graph.nodes.filter(
          (n) => n.adapterId === 'designability-scorer' && outputsByNode[n.id],
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
        if (scorersWithData.length >= 2) {
          // Forked benchmark (e.g. AF2 vs RoseTTAFold3): auto-open multi-branch 3D comparison!
          handleSelectNodes(scorersWithData.slice(0, COMPARE_PALETTE.length).map((n) => n.id));
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
    await scheduler.releaseActiveGpu();
    activeGpuWorker = null;
  }

  async function handleClearCache() {
    if (isRunning || activeRunState.leaseHeld) return;
    await scheduler.releaseActiveGpu();
    scheduler.clearMemo();
    clearRcsbCache();
    await clearAllModelCaches({ preserveCheckpoints: true });
    await refreshCacheStats();
    activeGpuWorker = null;
  }

  async function copyQuickShareLink() {
    moreMenuOpen = false;
    try {
      const plain = $state.snapshot(graph) as WorkflowGraph;
      const { hash, mode, strippedPdb } = await encodeGraphToHash(plain);
      const url = `${window.location.origin}${window.location.pathname}${hash}`;
      window.history.replaceState(null, '', hash);
      try {
        if (!navigator.clipboard?.writeText) {
          throw new Error('Clipboard API unavailable');
        }
        await navigator.clipboard.writeText(url);
        showToastMessage(
          strippedPdb
            ? `Copied ${url.length}-char link (${mode}; large PDB omitted)`
            : `Copied ${url.length}-char link (${mode})`,
        );
      } catch {
        showToastMessage(
          strippedPdb
            ? `Share link updated in address bar (${mode}; large PDB omitted — copy from URL bar)`
            : `Share link updated in address bar (${mode} — copy from URL bar)`,
          4200,
        );
      }
    } catch (err) {
      showToastMessage(
        `Could not create share link: ${err instanceof Error ? err.message : String(err)}`,
        4200,
      );
    }
  }

  function loadFullWorkflowGraph(next: WorkflowGraph, opts?: { clearOutputs?: boolean }) {
    const laid = next.nodes.some((n) => !n.position) ? autoLayoutGraph(next) : next;
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

  // Lazy single-source-of-truth compiled input preview (computed only while Node Studio is open)
  let studioOpen = $derived(expandedPane === 'studio' && Boolean(selectedNode));
  let studioPreview = $derived.by(() => {
    if (!studioOpen || !selectedNode) return null;
    return compileNodeInputPreview(graph, selectedNode.id, outputsByNode);
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
    if (peekMetrics) return activeRow?.designedPdb ?? contextStructure?.cifText ?? contextStructure?.pdbText ?? null;
    if (activePeekStructure) return activePeekStructure.cifText ?? activePeekStructure.pdbText;
    if (perResidueDisplayStructure) {
      return perResidueDisplayStructure.cifText ?? perResidueDisplayStructure.pdbText;
    }
    if (peekSequences) {
      // Clear 3D viewport when peeking a sequence-only wire/port unless user explicitly
      // previews the parent backbone (keeps Parameters-tab 3D residue picking intact).
      if (rightPanelTab === 'peek' && !showSeqParentBackbone) return null;
      return contextStructure?.cifText ?? contextStructure?.pdbText ?? null;
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
    return selectedNode
      ? `${selectedNode.label ?? selectedNode.adapterId} · denovo.run viewport`
      : 'denovo.run viewport';
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
    for (const id of selectedNodeIds) {
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

  // Multi-select 3D comparison derived state
  let compareColors = $derived.by((): Record<string, { badge: string; css: string }> => {
    if (!isCompareMode) return {};
    const map: Record<string, { badge: string; css: string }> = {};
    selectedNodeIds.slice(0, COMPARE_PALETTE.length).forEach((id, idx) => {
      const slot = COMPARE_PALETTE[idx];
      map[id] = { badge: slot.badge, css: slot.css };
    });
    return map;
  });

  let comparisonState = $derived.by(() => {
    if (!isCompareMode) return null;
    return buildComparisonState(graph, selectedNodeIds, outputsByNode, {
      layout: compareLayout,
      colorMode: compareColorMode,
      alignKabsch: compareAlignKabsch,
      branchCandidateIdx,
      hiddenModelIds: new Set(hiddenCompareModelIds),
      pairingMode: comparePairingMode,
      selectedCandidateKey: compareSelectedCandidateKey,
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
        type="button"
        class="brand-logo-btn"
        disabled={isRunning}
        onclick={resetToHomeCanvas}
        title="Start a clean empty canvas"
      >
        <span class="accent">denovo</span>.run
      </button>
    </h1>

    <label class="example-picker" class:disabled={isRunning} data-tour="preset-select">
      <span class="example-lbl">Example Workflow</span>
      <select
        class="preset-select"
        name="workflow-preset"
        aria-label="Select an example workflow"
        disabled={isRunning}
        value={selectedPresetId}
        title={activePreset?.summary ?? graph.name}
        onchange={(e) => selectPreset(e.currentTarget.value)}
      >
        {#if !activePreset}
          <option value={selectedPresetId}>Custom: {graph.name}</option>
        {/if}
        <optgroup label="Select an example workflow">
          {#each WORKFLOW_PRESETS as p (p.id)}
            <option value={p.id} title={p.summary}>{p.name} — {p.badge}</option>
          {/each}
        </optgroup>
      </select>
    </label>
  </div>

  <div class="hdr-right">
    <!-- System Status Pill & Popover -->
    <div class="pop-wrap" bind:this={sysWrapEl}>
      <button
        type="button"
        class="hdr-pill"
        class:ok={webgpuSupported && !webgpuIsFallback && !activeGpuWorker}
        class:warn={webgpuSupported && webgpuIsFallback && !activeGpuWorker}
        class:bad={webgpuSupported === false}
        class:gpu={Boolean(activeGpuWorker)}
        aria-expanded={sysMenuOpen}
        onclick={() => {
          moreMenuOpen = false;
          const nextOpen = !sysMenuOpen;
          if (nextOpen && sysWrapEl) {
            const rect = sysWrapEl.getBoundingClientRect();
            sysPopAlignLeft = rect.right < 276;
          }
          sysMenuOpen = nextOpen;
        }}
        title="WebGPU status, VRAM & weight cache"
      >
        <span class="status-dot"></span>
        {activeGpuWorkerLabel
          ? `GPU: ${activeGpuWorkerLabel}`
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
            <span class="pop-k">WebGPU</span>
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
          {#if webgpuIsFallback}
            <div class="pop-warn-note">
              Software WebGPU fallback detected. Heavy GPU models will be slower (ThermoMPNN,
              LigandMPNN &amp; ProteinMPNN fall back to CPU automatically). On Linux Chrome with
              NVIDIA/AMD, try launching with <code>--enable-features=Vulkan</code>.
            </div>
          {/if}
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
              type="button"
              class="pop-action danger"
              disabled={isRunning || activeRunState.leaseHeld}
              onclick={handleClearCache}
              title="Clear cached model weights from browser storage"
            >
              Clear
            </button>
          </div>
        </div>
      {/if}
    </div>

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
      {splitMode === 'horizontal' ? '⬓' : '◫'}
    </button>

    <!-- Theme Toggle -->
    <button
      type="button"
      class="hdr-icon-btn"
      onclick={toggleTheme}
      title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
      aria-label="Toggle color theme"
    >
      {theme === 'dark' ? '☀' : '☾'}
    </button>

    <!-- Overflow ⋯ Menu Popover -->
    <div class="pop-wrap" bind:this={moreWrapEl}>
      <button
        type="button"
        class="hdr-icon-btn"
        class:active={moreMenuOpen}
        aria-expanded={moreMenuOpen}
        aria-label="Workflow actions menu"
        title="Recipe, Share, Import/Export & Run options"
        onclick={() => {
          sysMenuOpen = false;
          moreMenuOpen = !moreMenuOpen;
        }}
      >
        ⋯
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
            <span>🔗 Share Link</span>
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
          <div class="menu-sep"></div>
          <button
            type="button"
            class="menu-item"
            role="menuitem"
            disabled={isRunning || validationErrors.length > 0 || hasInvalidNumeric}
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
      disabled={isRunning || validationErrors.length > 0 || hasInvalidNumeric}
      onclick={() => runWorkflow(false)}
      title={hasInvalidNumeric
        ? 'Fix invalid numeric parameter(s) before running'
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
    <section class="flow-panel">
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

      {#if shareToast}
        <div
          class="share-toast"
          class:is-error={shareToastTone === 'error'}
          role={shareToastTone === 'error' ? 'alert' : 'status'}
        >
          {shareToast}
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
        <div class="error-banner">
          <b>Pipeline error:</b> {runError}
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
      <div class="viewer-col" class:is-expanded={expandedPane === 'viewer'}>
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
                onclick={() => (compareLayout = 'overlay')}
              >
                Overlay
              </button>
              <button
                type="button"
                class="seg-btn"
                class:active={comparisonState.effectiveLayout === 'side-by-side'}
                onclick={() => (compareLayout = 'side-by-side')}
              >
                Side-by-Side
              </button>
            </div>
            <div class="seg-group" role="group" aria-label="3D ribbon color mode">
              <button
                type="button"
                class="seg-btn"
                class:active={compareColorMode === 'by-model'}
                onclick={() => (compareColorMode = 'by-model')}
              >
                By Model
              </button>
              <button
                type="button"
                class="seg-btn"
                class:active={compareColorMode === 'plddt'}
                onclick={() => (compareColorMode = 'plddt')}
              >
                pLDDT
              </button>
            </div>
            {@const sharedPreAligned = Boolean(comparisonState.sharedReferenceLabel) && comparisonState.effectiveLayout === 'overlay'}
            <label
              class:disabled-opt={sharedPreAligned}
              title={sharedPreAligned
                ? 'Branches share a designed backbone and are pre-aligned to it in Overlay mode'
                : 'Superpose models with Kabsch alignment (supports motif/substructure alignment for different lengths)'}
            >
              <input
                type="checkbox"
                disabled={sharedPreAligned}
                checked={sharedPreAligned || (compareAlignKabsch && comparisonState.effectiveLayout === 'overlay')}
                onchange={(e) => {
                  const checked = e.currentTarget.checked;
                  compareAlignKabsch = checked;
                  if (checked && compareLayout === 'side-by-side') {
                    compareLayout = 'overlay';
                  }
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

            {#if hasConditionerNode || hasMpnnNode}
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
            {theme}
            height={expandedPane === 'viewer' || studioOpen ? '100%' : splitMode === 'horizontal' ? 350 : 460}
            label={isCompareMode
              ? compareViewerLabel
              : studioInputViewerActive
                ? (studioPreview?.previewStructureMeta?.label ?? viewerLabel)
                : viewerLabel}
            atomGranularity={studioInputViewerActive
              ? (studioPreview?.previewStructureMeta?.atomGranularity ?? viewerAtomGranularity)
              : viewerAtomGranularity}
            bfactorSemantics={studioInputViewerActive
              ? (studioPreview?.previewStructureMeta?.bfactorSemantics ?? 'none')
              : viewerBfactorSemantics}
            highlightLayers={studioInputViewerActive ? studioPreview?.highlightLayers : undefined}
            hoveredToken={studioOpen ? studioHoveredToken : null}
            focusToken={studioOpen ? studioFocusToken : null}
            emptyMessage={isCompareMode ? compareEmptyMessage : isMultiDesignAll3d ? undefined : viewerEmptyMessage}
            emptyActionLabel={!isCompareMode && !isMultiDesignAll3d && isPeekingSequenceSlot && !showSeqParentBackbone && contextStructure
              ? 'Preview parent backbone'
              : undefined}
            onEmptyAction={() => (showSeqParentBackbone = true)}
            onResidueClick={isCompareMode || isMultiDesignAll3d ? undefined : handleResidueClick}
            onToggleModelVisibility={isMultiDesignAll3d
              ? toggleMultiDesignModelVisibility
              : toggleCompareModelVisibility}
            onSoloModel={isMultiDesignAll3d ? soloMultiDesignModel : soloCompareModel}
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
            }}
            onClose={() => {
              expandedPane = 'none';
              studioHoveredToken = null;
              studioFocusToken = null;
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
                hasStructureInput={graph.edges.some(
                  (e) => e.toNode === selectedNode.id && (e.toPort === 'structure' || e.toPort === 'template'),
                )}
                siblingGeneratorIds={
                  selectedNode.adapterId === 'genie3' || selectedNode.adapterId === 'rfd4-proteina'
                    ? getSiblingGeneratorIds(selectedNode.id)
                    : []
                }
                onUpdate={updateNodeParams}
                onUpdateLabel={updateNodeLabel}
                onSyncTaskToSiblings={handleSyncTaskToSiblings}
                onRunUpTo={(id) => runWorkflow(false, id)}
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
              {isCompareMode ? `Compare (${selectedNodeIds.length})` : 'Outputs'}
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
                hasStructureInput={graph.edges.some(
                  (e) => e.toNode === selectedNode.id && (e.toPort === 'structure' || e.toPort === 'template'),
                )}
                siblingGeneratorIds={
                  selectedNode.adapterId === 'genie3' || selectedNode.adapterId === 'rfd4-proteina'
                    ? getSiblingGeneratorIds(selectedNode.id)
                    : []
                }
                onUpdate={updateNodeParams}
                onUpdateLabel={updateNodeLabel}
                onSyncTaskToSiblings={handleSyncTaskToSiblings}
                onRunUpTo={(id) => runWorkflow(false, id)}
                onOpenStudio={() => openStudioForNode(selectedNode.id)}
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
    letter-spacing: -0.4px;
    color: var(--text-primary);
    white-space: nowrap;
  }
  .brand-logo-btn {
    margin: 0;
    padding: 2px 4px;
    margin-left: -4px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: inherit;
    font: inherit;
    letter-spacing: inherit;
    cursor: pointer;
    transition: opacity 0.12s ease, background 0.12s ease;
  }
  .brand-logo-btn:hover:not(:disabled) {
    background: var(--bg-surface-hover);
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

  .example-picker {
    display: inline-flex;
    align-items: center;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    overflow: hidden;
    min-width: 0;
    max-width: 100%;
  }
  .example-picker:hover:not(.disabled),
  .example-picker:focus-within:not(.disabled) {
    border-color: var(--accent);
  }
  .example-picker.disabled {
    opacity: 0.6;
  }
  .example-lbl {
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
    user-select: none;
  }
  .preset-select {
    background: var(--bg-surface);
    color: var(--text-body);
    border: none;
    outline: none;
    padding: 5px 10px;
    font-size: 12px;
    font-family: inherit;
    min-width: 0;
    flex: 1;
    width: 100%;
    max-width: min(340px, 100%);
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: pointer;
  }
  .preset-select:disabled {
    cursor: not-allowed;
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
  }
  .hdr-icon-btn:hover,
  .hdr-icon-btn.active {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
    border-color: var(--accent);
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
    width: min(260px, calc(100vw - 24px));
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 11.5px;
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
    .example-picker {
      display: flex;
      flex: 1 1 100%;
      width: 100%;
      min-width: 0;
      max-width: 100%;
    }
    .example-lbl {
      padding: 6px 7px;
      font-size: 9.5px;
    }
    .preset-select {
      min-width: 0;
      flex: 1;
      width: 100%;
      max-width: 100%;
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
  .pop-action.danger:hover:not(:disabled) {
    border-color: var(--err-border);
    color: var(--err-strong);
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
  .share-toast {
    position: absolute;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 40;
    max-width: calc(100% - 24px);
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
