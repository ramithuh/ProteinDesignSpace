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
  import { getBenchmarkById } from '$lib/engine/benchmarks';
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
    type ComparisonPairingMode,
  } from '$lib/engine/compare';
  import { estimateRunRequirements, type RunPreflightSummary } from '$lib/engine/run-preflight';
  import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
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
  import { compileNodeInputPreview } from '$lib/engine/input-preview';

  const scheduler = new ExecutionScheduler({
    checkpointStore: createIndexedDbCheckpointStore(),
  });

  let rightPanelTab = $state<'params' | 'peek'>('params');
  let expandedPane = $state<'none' | 'viewer' | 'inspector' | 'studio'>('none');
  let studioTab = $state<'input' | 'controls' | 'output'>('input');
  let studioHoveredToken = $state<string | null>(null);
  let studioFocusToken = $state<string | null>(null);
  let splitMode = $state<'horizontal' | 'vertical'>('horizontal');
  let splitPct = $state<number>(65);
  let draggingSplit = $state(false);
  let splitWrapEl: HTMLDivElement | null = $state(null);
  let splitMoveRafId: number | null = null;
  let pendingSplitClientX: number | null = null;
  let theme = $state<'dark' | 'light'>('dark');
  let webgpuSupported = $state<boolean | null>(null);
  let webgpuDetail = $state<string>('checking…');
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
  let firstRunAck = $state(false);
  let showOnboarding = $state(false);
  let onboardingInteractive = $state(false);
  let showSeqParentBackbone = $state(false);
  let pendingRun = $state<{
    bypassCache: boolean;
    upToNodeId?: string;
    preflight: RunPreflightSummary;
  } | null>(null);

  // Active workflow state
  let selectedPresetId = $state(WORKFLOW_PRESETS[0].id);
  let graph = $state<WorkflowGraph>(autoLayoutGraph(structuredClone(WORKFLOW_PRESETS[0].graph)));
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
  }

  function handleSelectNodes(ids: string[]) {
    const valid = ids.filter((id) => graph.nodes.some((n) => n.id === id)).slice(0, COMPARE_PALETTE.length);
    selectedNodeIds = valid;
    hiddenCompareModelIds = [];
    comparePairingMode = undefined;
    compareSelectedCandidateKey = null;
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
  let shareToastTimer: ReturnType<typeof setTimeout> | null = null;
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

  function showToastMessage(msg: string, ms = 3200) {
    shareToast = msg;
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
    if (splitMode !== 'horizontal') return;
    ev.preventDefault();
    draggingSplit = true;
  }

  function handleWindowPointerMove(ev: PointerEvent) {
    if (!draggingSplit || !splitWrapEl) return;
    pendingSplitClientX = ev.clientX;
    if (splitMoveRafId !== null) return;
    splitMoveRafId = requestAnimationFrame(() => {
      splitMoveRafId = null;
      if (!draggingSplit || !splitWrapEl || pendingSplitClientX === null) return;
      const rect = splitWrapEl.getBoundingClientRect();
      if (rect.width <= 0) return;
      const rawPct = ((pendingSplitClientX - rect.left) / rect.width) * 100;
      splitPct = Math.min(78, Math.max(32, Math.round(rawPct * 10) / 10));
    });
  }

  function handleWindowPointerUp() {
    if (splitMoveRafId !== null) {
      cancelAnimationFrame(splitMoveRafId);
      splitMoveRafId = null;
    }
    pendingSplitClientX = null;
    if (draggingSplit) draggingSplit = false;
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
      firstRunAck = localStorage.getItem('denovo-first-run-ack') === '1';
    } catch {
      applyTheme('dark');
    }

    if (!window.isSecureContext) {
      webgpuSupported = false;
      webgpuDetail = 'needs HTTPS/localhost';
    } else if (!('gpu' in navigator) || !navigator.gpu) {
      webgpuSupported = false;
      webgpuDetail = 'unavailable in browser';
    } else {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          webgpuSupported = false;
          webgpuDetail = 'no GPU adapter';
        } else {
          webgpuSupported = true;
          const info = (adapter as unknown as { info?: { vendor?: string; architecture?: string } }).info;
          const f16 = adapter.features.has('shader-f16') ? 'f16' : 'f32';
          const tag = info?.vendor || info?.architecture || 'ready';
          webgpuDetail = `${tag} (${f16})`;
        }
      } catch {
        webgpuSupported = false;
        webgpuDetail = 'adapter error';
      }
    }
    const { webMcpRegistered } = attachProgrammaticApi(scheduler, handleProgress);
    webMcpReady = webMcpRegistered;

    // Load shared workflow from URL hash (#p=... or #g=...) if present
    let loadedFromHash = false;
    if (window.location.hash.length > 2) {
      const rawHash = window.location.hash;
      const isSharedHash = rawHash.startsWith('#g=') || rawHash.startsWith('#p=');
      try {
        const decoded = await decodeHashToGraph(rawHash);
        if (decoded) {
          graph = decoded;
          selectedPresetId = decoded.id;
          setSingleSelection(decoded.nodes[0]?.id ?? '');
          showToastMessage('Loaded shared workflow from URL', 3000);
          loadedFromHash = true;
        } else if (isSharedHash) {
          showToastMessage('Could not decode shared workflow link — loaded default workflow', 4200);
        }
      } catch (err) {
        if (isSharedHash) {
          showToastMessage(
            `Invalid shared workflow URL: ${err instanceof Error ? err.message : 'decode failed'}`,
            4500,
          );
        }
      }
    }

    // Show welcome onboarding once on first direct visit (never interrupt shared links)
    try {
      if (!loadedFromHash && localStorage.getItem('denovo-onboarding-seen') !== '1') {
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

  function handleGraphChange(next: WorkflowGraph) {
    const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
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

    graph = next;
    scheduler.invalidateEditedGraph(prevSnapshot, next);

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
    graph = {
      ...graph,
      nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, label } : n)),
    };
  }

  let validationErrors = $derived(validateGraph(graph));
  let bioIssues = $derived(validateBioCapabilities(graph, outputsByNode));

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

  function parseLengthFromContig(contig: string | undefined): number | null {
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
    return null;
  }

  // Auto-seed unconfigured (or previously auto-seeded) Genie 3 & RFD4-Proteina blocks when wired
  // to a curated benchmark PDB Input on their unified `structure` port — fixes forked workflows
  // like 1PRW -> Genie 3 + RFD4 so both immediately pick up the exact motif/binder/enzyme task,
  // and resets stale auto-seeded Genie 3 constraints when switching to an RFD4-only benchmark (5O45 -> 1CTT).
  $effect(() => {
    if (isRunning) return;
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

      if (node.adapterId === 'rfd4-proteina') {
        const isUntouched =
          String(cp.mode ?? 'unconditional') === 'unconditional' &&
          !String(cp.hotspots ?? '').trim() &&
          !String(cp.motifSpan ?? '').trim() &&
          !String(cp.lockChain ?? '').trim() &&
          !String(cp.customSpecYaml ?? '').trim();
        if (!isUntouched && !seededId) continue;

        const bp = bench.conditionParams;
        const parsedLen = parseLengthFromContig(bp.contig);
        const nextMode =
          bp.taskPreset !== 'custom'
            ? 'preset'
            : bench.category === 'binder'
              ? 'binder'
              : 'motif';
        patches ??= {};
        patches[node.id] = {
          ...cp,
          mode: nextMode,
          ...(bp.taskPreset !== 'custom' ? { presetId: bp.taskPreset } : {}),
          ...(parsedLen !== null ? { length: parsedLen } : {}),
          contig: bp.contig,
          lockChain: bp.lockChain ?? '',
          hotspots: bp.hotspots ?? '',
          motifSpan: bp.motifSpan ?? '',
          unindexedMotif: Boolean(bp.unindexedMotif),
          hbondMode: bp.hbondMode ?? 'none',
          hbondSelection: bp.hbondSelection ?? '',
          autoSeededBenchmarkId: bench.id,
        };
      } else if (node.adapterId === 'genie3') {
        if (bench.supportsGenie3) {
          const isUntouched =
            String(cp.taskMode ?? 'unconditional') === 'unconditional' &&
            !String(cp.hotspots ?? '').trim() &&
            !String(cp.motifSpan ?? '').trim() &&
            !String(cp.lockChain ?? '').trim();
          if (!isUntouched && !seededId) continue;

          const bp = bench.conditionParams;
          const parsedLen = parseLengthFromContig(bp.contig);
          const nextTaskMode = bench.category === 'binder' ? 'binder' : 'motif';
          patches ??= {};
          patches[node.id] = {
            ...cp,
            taskMode: nextTaskMode,
            ...(parsedLen !== null ? { length: parsedLen } : {}),
            contig: bp.contig,
            lockChain: bp.lockChain ?? '',
            hotspots: bp.hotspots ?? '',
            motifSpan: bp.motifSpan ?? '',
            autoSeededBenchmarkId: bench.id,
          };
        } else if (seededId) {
          // Previously auto-seeded for a Genie 3-compatible benchmark, now switched to an
          // RFD4-only benchmark (e.g. 5O45 -> 1CTT): reset stale constraints cleanly.
          patches ??= {};
          patches[node.id] = {
            ...cp,
            taskMode: 'unconditional',
            contig: '',
            lockChain: '',
            hotspots: '',
            motifSpan: '',
            autoSeededBenchmarkId: bench.id,
          };
        }
      }
    }
    if (patches) {
      const map = patches;
      const prevSnapshot = $state.snapshot(graph) as WorkflowGraph;
      const nextGraph: WorkflowGraph = {
        ...prevSnapshot,
        nodes: prevSnapshot.nodes.map((n) => (map[n.id] ? { ...n, params: map[n.id] } : n)),
      };
      graph = nextGraph;
      scheduler.invalidateEditedGraph(prevSnapshot, nextGraph);
      invalidateNodes(getDownstreamNodeIds(nextGraph, Object.keys(map)));
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
        const hadTracked =
          previewedInputSigs.has(node.id) ||
          inFlightInputSigs.has(node.id) ||
          failedInputSigs.has(node.id) ||
          Boolean(outputsByNode[node.id]) ||
          Boolean(nodeStates[node.id]);
        previewedInputSigs.delete(node.id);
        inFlightInputSigs.delete(node.id);
        failedInputSigs.delete(node.id);
        if (hadTracked) {
          invalidateNodes(getDownstreamNodeIds(graph, [node.id]));
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
    if (!confirmed && !firstRunAck) {
      const preflight = estimateRunRequirements(targetGraph);
      if (preflight.usesWebGpu || preflight.totalDownloadMB > 0) {
        pendingRun = { bypassCache, upToNodeId, preflight };
        return;
      }
    }
    localRunning = true;
    runTargetIds = targetGraph.nodes.map((n) => n.id);
    runError = null;
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
    await clearAllModelCaches();
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

      if (molPickMode === 'mpnn') {
        const allMpnn = graph.nodes.filter((n) => n.adapterId === 'web-protein-mpnn');
        const target =
          selectedNode?.adapterId === 'web-protein-mpnn'
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
      if (selectedNode?.adapterId === 'web-protein-mpnn') {
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
  let hasMpnnNode = $derived(graph.nodes.some((n) => n.adapterId === 'web-protein-mpnn'));

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

  // Directly wired upstream template StructureArtifact for live motif/hotspot source-residue resolution
  let selectedUpstreamStructure = $derived.by((): StructureArtifact | null => {
    if (!selectedNode) return null;
    const edge = graph.edges.find(
      (e) => e.toNode === selectedNode.id && (e.toPort === 'structure' || e.toPort === 'template'),
    );
    if (edge) {
      const art = outputsByNode[edge.fromNode]?.[edge.fromPort];
      if (art?.kind === 'StructureArtifact') return art;
    }
    return contextStructure;
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
  let peekSequences = $derived<SequenceSet | null>(
    activeArtifact?.kind === 'SequenceSet' ? activeArtifact : null,
  );

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
    if (peekStructure) return peekStructure.cifText ?? peekStructure.pdbText;
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
    peekStructure ??
      (peekMetrics || (peekSequences && (!isPeekingSequenceSlot || showSeqParentBackbone))
        ? contextStructure
        : null),
  );

  let viewerAtomGranularity = $derived.by((): AtomGranularity | undefined => {
    if (isCompareMode) return undefined;
    if (peekStructure) return peekStructure.atomGranularity;
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
    if (peekStructure) {
      const fromTag = activePeekSlot?.peerNodeLabel ? ` ← ${activePeekSlot.peerNodeLabel}` : '';
      return `${dirTag}${fromTag} · ${peekStructure.label}`;
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

  let compareViewerLabel = $derived.by((): string => {
    if (!comparisonState) return viewerLabel;
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
/>

<!-- Single Compact Top Header Bar (~48px) -->
<header class="app-header">
  <div class="hdr-left">
    <h1 class="brand-logo"><span class="accent">denovo</span>.run</h1>

    <label class="example-picker" class:disabled={isRunning} data-tour="preset-select">
      <span class="example-lbl">Example Workflow</span>
      <select
        class="preset-select"
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
        class:ok={webgpuSupported && !activeGpuWorker}
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
        {activeGpuWorkerLabel ? `GPU: ${activeGpuWorkerLabel}` : 'WebGPU'}
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
            <span class="pop-v" class:ok-t={webgpuSupported} class:bad-t={webgpuSupported === false}>
              {webgpuDetail}
            </span>
          </div>
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
          <label class="menu-item" class:disabled={isRunning}>
            <span>Import JSON / YAML</span>
            <input
              type="file"
              accept=".json,.yaml,.yml"
              hidden
              disabled={isRunning}
              onchange={importGraphJson}
            />
          </label>
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
    class:dragging={draggingSplit}
    bind:this={splitWrapEl}
    style={splitMode === 'horizontal' ? `--left-pane: ${splitPct}%;` : undefined}
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
          fillHeight={splitMode === 'horizontal'}
          autoCommitProposed={showOnboarding && onboardingInteractive}
          onGraphChange={handleGraphChange}
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
        <div class="share-toast">{shareToast}</div>
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

      {#if validationErrors.length > 0}
        <div class="warn-banner">
          <b>Workflow check:</b> {validationErrors.join(' · ')}
        </div>
      {/if}

      {#if bioIssues.length > 0}
        <div class="bio-banner-list" role="status" aria-live="polite">
          {#each bioIssues as issue (`${issue.code}:${issue.nodeId}`)}
            <div class="bio-banner" class:is-err={issue.severity === 'error'}>
              <span class="bio-msg">
                <b>Biological compatibility:</b> {issue.message}
              </span>
              {#if issue.fix}
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

      {#if runError}
        <div class="error-banner">
          <b>Pipeline error:</b> {runError}
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
        ondblclick={() => (splitPct = 65)}
      >
        <div class="split-bar"></div>
      </div>
    {/if}

    <!-- Output & Inspector Pane (Right in Horizontal Split, Bottom in Vertical Split) -->
    <section
      class="results-grid"
      class:stacked-pane={splitMode === 'horizontal'}
      class:is-studio-open={studioOpen}
    >
      {#if expandedPane !== 'none'}
        <div
          class="expand-backdrop"
          role="presentation"
          onclick={() => (expandedPane = 'none')}
        ></div>
      {/if}
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
            <label title="Slowly rotate the 3D view (pauses while dragging)"><input type="checkbox" bind:checked={viewerSpin} /> Spin</label>
            <button
              type="button"
              class="cmp-exit-mini"
              onclick={() => setSingleSelection(selectedNodeId)}
              title="Exit comparison mode (Esc)"
            >
              × Exit
            </button>
          {:else}
            <label><input type="checkbox" bind:checked={showDesigned} /> Designed</label>
            <label><input type="checkbox" bind:checked={showRefold} /> Refold (pLDDT)</label>
            <label><input type="checkbox" bind:checked={illustrative} /> Illustrative</label>
            <label title="Slowly rotate the 3D view (pauses while dragging)"><input type="checkbox" bind:checked={viewerSpin} /> Spin</label>
            {#if viewerStructureMeta?.trajectoryPdb}
              <label><input type="checkbox" bind:checked={showTrajectory} /> Trajectory</label>
              {#if showTrajectory}
                <input
                  type="range"
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
                <select class="pick-select" bind:value={molPickMode}>
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
        </div>

        <div class="viewer-stage">
          <MolSuperposeViewer
            designedPdb={isCompareMode
              ? null
              : studioInputViewerActive
                ? (studioPreview?.previewStructurePdb ?? viewerDesignedPdb)
                : viewerDesignedPdb}
            refoldPdb={isCompareMode || studioInputViewerActive ? null : viewerRefoldPdb}
            models={isCompareMode && comparisonState ? comparisonState.models : null}
            trajectoryPdb={isCompareMode || studioInputViewerActive
              ? null
              : (viewerStructureMeta?.trajectoryPdb ?? null)}
            frameIndex={showTrajectory ? trajFrame : undefined}
            {showDesigned}
            showRefold={studioInputViewerActive ? false : showRefold}
            {showTrajectory}
            {illustrative}
            spin={viewerSpin && !isRunning}
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
            bfactorSemantics={studioInputViewerActive ? 'none' : viewerBfactorSemantics}
            highlightLayers={studioInputViewerActive ? studioPreview?.highlightLayers : undefined}
            hoveredToken={studioOpen ? studioHoveredToken : null}
            focusToken={studioOpen ? studioFocusToken : null}
            emptyMessage={isCompareMode ? undefined : viewerEmptyMessage}
            emptyActionLabel={!isCompareMode && isPeekingSequenceSlot && !showSeqParentBackbone && contextStructure
              ? 'Preview parent backbone'
              : undefined}
            onEmptyAction={() => (showSeqParentBackbone = true)}
            onResidueClick={isCompareMode ? undefined : handleResidueClick}
            onToggleModelVisibility={toggleCompareModelVisibility}
            onSoloModel={soloCompareModel}
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
        <div class="right-tabs" role="tablist" aria-label="Block inspector tabs">
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
  .pop-v.bad-t {
    color: var(--err-strong);
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
  .menu-item:hover:not(:disabled):not(.disabled) {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }
  .menu-item:disabled,
  .menu-item.disabled {
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
  }
  .workbench-split.horizontal {
    flex-direction: row;
    align-items: stretch;
    gap: 0;
    height: calc(100dvh - 76px);
    min-height: 620px;
  }
  .workbench-split.dragging {
    user-select: none;
    cursor: col-resize;
  }
  .workbench-split.horizontal > .flow-panel {
    width: var(--left-pane, 65%);
    flex-shrink: 0;
    min-width: 0;
    margin-bottom: 0;
    height: 100%;
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
  .split-bar {
    width: 3px;
    height: 44px;
    border-radius: 999px;
    background: var(--border-strong);
    transition: background 0.15s ease, height 0.15s ease;
  }
  .split-handle:hover .split-bar,
  .workbench-split.dragging .split-bar {
    background: var(--accent);
    height: 72px;
  }

  .flow-panel {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 0;
  }
  .share-toast {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--ok-text);
    background: var(--ok-bg);
    border: 1px solid var(--ok-border);
    padding: 6px 10px;
    border-radius: 6px;
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

  .results-grid {
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    gap: 14px;
    align-items: start;
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
    .workbench-split.horizontal {
      flex-direction: column;
      height: auto;
      min-height: 0;
      gap: 12px;
    }
    .workbench-split.horizontal > .flow-panel {
      width: 100%;
      height: auto;
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
      grid-template-columns: 1fr;
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
    z-index: 160;
    display: grid !important;
    grid-template-columns: minmax(340px, 46%) minmax(420px, 1fr);
    grid-template-rows: minmax(0, 1fr);
    gap: 12px;
    padding: 12px;
    background: var(--bg-canvas);
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    box-shadow: 0 28px 72px rgba(0, 0, 0, 0.58);
    overflow: hidden;
  }
  .results-grid.is-studio-open > .expand-backdrop {
    z-index: -1;
  }
  .results-grid.is-studio-open > .viewer-col {
    display: flex;
    flex-direction: column;
    min-height: 0;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 10px;
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
    min-height: 0;
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
    margin-left: 6px;
    margin-bottom: 4px;
    color: var(--accent);
    border-color: color-mix(in srgb, var(--accent) 45%, var(--border-strong));
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
    margin-left: 6px;
    margin-bottom: 4px;
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
    margin-left: auto;
    margin-bottom: 4px;
    border: 1px solid var(--accent);
    background: var(--bg-pill);
    color: var(--accent-soft);
    border-radius: 999px;
    padding: 2px 9px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    cursor: pointer;
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
    overflow: hidden;
  }
  .right-tabs {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 10px 0;
    border-bottom: 1px solid var(--border);
    background: var(--bg-surface-2);
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
    margin-left: auto;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-dim);
    padding-bottom: 4px;
    max-width: 180px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .right-body {
    max-height: 520px;
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
