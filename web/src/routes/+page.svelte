<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { SvelteFlowProvider } from '@xyflow/svelte';
  import {
    ExecutionScheduler,
    attachProgrammaticApi,
    materializePdbInputNode,
    materializeSequenceInputNode,
  } from '$lib/engine/scheduler';
  import { isValidPdbId, normalizePdbId } from '$lib/protein-ops/rcsb';
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
  import { clearAllModelCaches, getCacheStats } from '$lib/engine/opfs-cache';
  import { COMPARE_PALETTE, buildComparisonState } from '$lib/engine/compare';
  import { estimateRunRequirements, type RunPreflightSummary } from '$lib/engine/run-preflight';
  import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
  import type {
    BioCompatibilityIssue,
    CanonicalArtifact,
    ConditionSpecArtifact,
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
  import ComparePeekPanel from '$lib/ui/ComparePeekPanel.svelte';
  import FirstRunModal from '$lib/ui/FirstRunModal.svelte';
  import OnboardingModal from '$lib/ui/OnboardingModal.svelte';
  import NodeInspector from '$lib/ui/NodeInspector.svelte';
  import WorkflowCanvas from '$lib/ui/WorkflowCanvas.svelte';
  import RecipeDrawer from '$lib/ui/RecipeDrawer.svelte';

  const scheduler = new ExecutionScheduler();

  let rightPanelTab = $state<'params' | 'peek'>('params');
  let splitMode = $state<'horizontal' | 'vertical'>('horizontal');
  let splitPct = $state<number>(65);
  let draggingSplit = $state(false);
  let splitWrapEl: HTMLDivElement | null = $state(null);
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
  let hiddenCompareModelIds = $state<string[]>([]);
  let branchCandidateIdx = $state<Record<string, number>>({});
  let isRunning = $state(false);

  function setSingleSelection(id: string) {
    selectedNodeIds = id ? [id] : [];
    hiddenCompareModelIds = [];
  }

  function handleSelectNodes(ids: string[]) {
    const valid = ids.filter((id) => graph.nodes.some((n) => n.id === id)).slice(0, COMPARE_PALETTE.length);
    selectedNodeIds = valid.length > 0 ? valid : graph.nodes[0]?.id ? [graph.nodes[0].id] : [];
    hiddenCompareModelIds = [];
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
    const others = allIds.filter((id) => id !== modelId && id !== 'ref-designed');
    const alreadySolo =
      others.length > 0 &&
      others.every((id) => hiddenCompareModelIds.includes(id)) &&
      !hiddenCompareModelIds.includes(modelId);
    if (alreadySolo) {
      hiddenCompareModelIds = [];
    } else {
      hiddenCompareModelIds = others;
    }
  }
  let runTargetIds = $state<string[]>([]);
  let showRecipe = $state(false);
  let shareToast = $state<string | null>(null);
  let shareToastTimer: ReturnType<typeof setTimeout> | null = null;
  let runError = $state<string | null>(null);
  let totalElapsedMs = $state<number | null>(null);

  let nodeStates = $state<Record<string, NodeProgressEvent>>({});
  let outputsByNode = $state.raw<Record<string, Record<string, CanonicalArtifact>>>({});

  // Viewer & intermediate peek state
  let selectedPeekSlotId = $state<string | null>(null);
  let selectedEdgeId = $state<string | null>(null);
  let selectedRowIdx = $state(0);
  let selectedFoldIdx = $state(0);
  let showDesigned = $state(true);
  let showRefold = $state(true);
  let showTrajectory = $state(false);
  let illustrative = $state(false);
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
    const rect = splitWrapEl.getBoundingClientRect();
    if (rect.width <= 0) return;
    const rawPct = ((ev.clientX - rect.left) / rect.width) * 100;
    splitPct = Math.min(78, Math.max(32, Math.round(rawPct * 10) / 10));
  }

  function handleWindowPointerUp() {
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
    if (pendingRun || (showOnboarding && !onboardingInteractive)) return;
    if (ev.key === 'Escape') {
      sysMenuOpen = false;
      moreMenuOpen = false;
      return;
    }

    // Cmd/Ctrl + Enter runs workflow from anywhere (never hijacks Cmd+R browser reload)
    if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey) && !ev.altKey) {
      if (!isRunning && validationErrors.length === 0) {
        ev.preventDefault();
        void runWorkflow(false);
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
      try {
        const decoded = await decodeHashToGraph(window.location.hash);
        if (decoded) {
          graph = decoded;
          selectedPresetId = decoded.id;
          setSingleSelection(decoded.nodes[0]?.id ?? '');
          showToastMessage('Loaded shared workflow from URL', 3000);
          loadedFromHash = true;
        }
      } catch {
        /* ignore malformed hash */
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
    if (shareToastTimer) clearTimeout(shareToastTimer);
    void scheduler.releaseActiveGpu();
  });

  async function refreshCacheStats() {
    const st = await getCacheStats();
    cacheUsedMB = st.usedBytes / 1024 / 1024;
  }

  function handleProgress(ev: NodeProgressEvent) {
    nodeStates = { ...nodeStates, [ev.nodeId]: ev };
    activeGpuWorker = scheduler.getActiveGpuAdapter();
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
    nodeStates = {};
    outputsByNode = {};
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
    nodeStates = {};
    outputsByNode = {};
    runError = null;
    rightPanelTab = 'params';
  }

  function autoStepTourBlock(adapterId: 'web-protein-mpnn' | 'localfold-af2' | 'designability-scorer') {
    if (isRunning) return;
    const opts =
      adapterId === 'web-protein-mpnn'
        ? { autoCommit: true, params: { family: 'solublempnn', modelName: 'v_48_020' }, label: 'SolubleMPNN' }
        : adapterId === 'localfold-af2'
          ? { autoCommit: true, params: { model: 'monomer' }, label: 'LocalFold (AF2 Model 1)' }
          : { autoCommit: true, label: 'scRMSD / scTM' };
    const anchorId =
      adapterId === 'designability-scorer'
        ? (graph.nodes.find((n) => n.adapterId === 'localfold-af2')?.id ?? selectedNodeId)
        : adapterId === 'localfold-af2'
          ? (graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn')?.id ?? selectedNodeId)
          : (graph.nodes.find((n) => n.adapterId === 'genie3')?.id ??
            graph.nodes.find((n) => n.adapterId === 'pdb-input')?.id ??
            selectedNodeId);

    const { graph: nextGraph, nodeId } = insertBlockSmart(
      $state.snapshot(graph) as WorkflowGraph,
      adapterId,
      anchorId || undefined,
      undefined,
      opts,
    );
    graph = autoLayoutGraph(nextGraph);
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

  function handleGraphChange(next: WorkflowGraph) {
    const nextById = new Map(next.nodes.map((n) => [n.id, n]));
    const prevById = new Map(graph.nodes.map((n) => [n.id, n]));

    const nextOutputs: Record<string, Record<string, CanonicalArtifact>> = {};
    for (const [id, outs] of Object.entries(outputsByNode)) {
      const cur = nextById.get(id);
      const prev = prevById.get(id);
      if (cur && prev && cur.adapterId === prev.adapterId) {
        nextOutputs[id] = outs;
      }
    }
    outputsByNode = nextOutputs;

    const nextStates: Record<string, NodeProgressEvent> = {};
    for (const [id, st] of Object.entries(nodeStates)) {
      const cur = nextById.get(id);
      const prev = prevById.get(id);
      if (cur && prev && cur.adapterId === prev.adapterId) {
        nextStates[id] = st;
      }
    }
    nodeStates = nextStates;

    graph = next;
    const kept = selectedNodeIds.filter((id) => nextById.has(id));
    if (kept.length !== selectedNodeIds.length) {
      selectedNodeIds = kept.length > 0 ? kept : next.nodes[0]?.id ? [next.nodes[0].id] : [];
    }
    runError = null;
  }

  function updateNodeParams(nodeId: string, params: Record<string, unknown>) {
    graph = {
      ...graph,
      nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, params } : n)),
    };
  }

  function updateNodeLabel(nodeId: string, label: string) {
    graph = {
      ...graph,
      nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, label } : n)),
    };
  }

  let validationErrors = $derived(validateGraph(graph));
  let bioIssues = $derived(validateBioCapabilities(graph));

  // Instant CPU/fetch preview for starting inputs (PDB Input & Sequence Input) so users can
  // inspect structures in 3D or compare PDBs immediately without needing to click Run.
  const previewedInputSigs = new Map<string, string>();
  const inFlightInputSigs = new Map<string, string>();

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

  // Auto-populate untouched (or previously auto-synced) Conditioning blocks when wired to a
  // curated benchmark PDB Input so contig, motif spans, lockChain, and hotspots stay in sync.
  $effect(() => {
    if (isRunning) return;
    let patches: Record<string, Record<string, unknown>> | null = null;
    for (const condNode of graph.nodes) {
      if (condNode.adapterId !== 'rfd4-conditioner') continue;
      const tplEdge = graph.edges.find((e) => e.toNode === condNode.id && e.toPort === 'template');
      if (!tplEdge) continue;
      const srcNode = graph.nodes.find((n) => n.id === tplEdge.fromNode);
      if (!srcNode || srcNode.adapterId !== 'pdb-input') continue;
      const benchId = String(srcNode.params.benchmarkId ?? '').trim();
      if (!benchId) continue;
      const bench = getBenchmarkById(benchId);
      if (!bench) continue;

      const cp = condNode.params ?? {};
      const syncedId = String(cp.autoSyncedBenchmarkId ?? '').trim();
      if (syncedId === bench.id) continue;

      const isUntouchedDefault =
        String(cp.taskPreset ?? 'custom') === 'custom' &&
        (String(cp.contig ?? '70-70').trim() === '' || String(cp.contig ?? '70-70').trim() === '70-70') &&
        !String(cp.lockChain ?? '').trim() &&
        !String(cp.hotspots ?? '').trim() &&
        !String(cp.motifSpan ?? '').trim() &&
        !String(cp.cathCode ?? '').trim() &&
        String(cp.hbondMode ?? 'none') === 'none' &&
        !String(cp.rawJsonOverride ?? '').trim();

      if (!isUntouchedDefault && !syncedId) continue;

      patches ??= {};
      patches[condNode.id] = {
        ...cp,
        taskPreset: bench.conditionParams.taskPreset,
        contig: bench.conditionParams.contig,
        lockChain: bench.conditionParams.lockChain ?? '',
        hotspots: bench.conditionParams.hotspots ?? '',
        motifSpan: bench.conditionParams.motifSpan ?? '',
        unindexedMotif: Boolean(bench.conditionParams.unindexedMotif),
        hbondMode: bench.conditionParams.hbondMode ?? 'none',
        hbondSelection: bench.conditionParams.hbondSelection ?? '',
        autoSyncedBenchmarkId: bench.id,
      };
    }
    if (patches) {
      const map = patches;
      graph = {
        ...graph,
        nodes: graph.nodes.map((n) => (map[n.id] ? { ...n, params: map[n.id] } : n)),
      };
    }
  });

  $effect(() => {
    if (isRunning) return;
    const nodes = graph.nodes;
    const activeIds = new Set(nodes.map((n) => n.id));
    for (const id of [...previewedInputSigs.keys()]) {
      if (!activeIds.has(id)) {
        previewedInputSigs.delete(id);
        inFlightInputSigs.delete(id);
      }
    }

    for (const node of nodes) {
      if (node.adapterId !== 'pdb-input' && node.adapterId !== 'sequence-input') continue;
      const sig = computeInputPreviewSig(node);
      if (!sig) continue;
      const hasOut = Boolean(outputsByNode[node.id] && Object.keys(outputsByNode[node.id]).length > 0);
      if (inFlightInputSigs.get(node.id) === sig) continue;
      if (previewedInputSigs.get(node.id) === sig && hasOut) continue;

      previewedInputSigs.set(node.id, sig);
      const nodeSnap = $state.snapshot(node);

      if (node.adapterId === 'sequence-input') {
        try {
          const seqSet = materializeSequenceInputNode(nodeSnap);
          outputsByNode = { ...outputsByNode, [node.id]: { sequences: seqSet } };
          nodeStates = {
            ...nodeStates,
            [node.id]: { nodeId: node.id, status: 'done', detail: `${seqSet.candidates.length} sequence(s)` },
          };
        } catch {
          // Wait quietly until user finishes typing a valid sequence
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

  function applyBioFix(issue: BioCompatibilityIssue) {
    if (!issue.fix || isRunning) return;
    const { nodeId, paramsPatch, nodeLabel } = issue.fix;
    graph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === nodeId
          ? {
              ...n,
              params: { ...n.params, ...paramsPatch },
              ...(nodeLabel ? { label: nodeLabel } : {}),
            }
          : n,
      ),
    };
    runError = null;
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

  async function runWorkflow(bypassCache = false, upToNodeId?: string, confirmed = false) {
    if (isRunning) return;
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
    isRunning = true;
    runTargetIds = targetGraph.nodes.map((n) => n.id);
    runError = null;
    if (!upToNodeId) {
      totalElapsedMs = null;
      nodeStates = {};
      outputsByNode = {};
    } else {
      const targetIds = new Set(targetGraph.nodes.map((n) => n.id));
      const nextStates: Record<string, NodeProgressEvent> = {};
      for (const [id, st] of Object.entries(nodeStates)) {
        if (!targetIds.has(id)) nextStates[id] = st;
      }
      nodeStates = nextStates;
    }

    try {
      const res = await scheduler.run(targetGraph, handleProgress, {
        bypassCache,
        onNodeOutput: (nodeId, outs) => {
          outputsByNode = { ...outputsByNode, [nodeId]: outs };
        },
      });
      outputsByNode = { ...outputsByNode, ...res.outputsByNode };
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
        // Auto-focus a scorer or fold node if present so results appear immediately
        const scorersWithData = graph.nodes.filter(
          (n) => n.adapterId === 'designability-scorer' && outputsByNode[n.id],
        );
        const lastScorer = [...graph.nodes].reverse().find((n) => n.adapterId === 'designability-scorer');
        const lastFold = [...graph.nodes].reverse().find((n) => n.adapterId === 'localfold-af2');
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
        }
      }
      rightPanelTab = 'peek';

      await refreshCacheStats();
    } catch (err) {
      runError = err instanceof Error ? err.message : String(err);
    } finally {
      isRunning = false;
      runTargetIds = [];
      activeGpuWorker = scheduler.getActiveGpuAdapter();
    }
  }

  async function freeVram() {
    if (isRunning) return;
    await scheduler.releaseActiveGpu();
    activeGpuWorker = null;
  }

  async function handleClearCache() {
    if (isRunning) return;
    await scheduler.releaseActiveGpu();
    scheduler.clearMemo();
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
      await navigator.clipboard.writeText(url);
      showToastMessage(
        strippedPdb
          ? `Copied ${url.length}-char link (${mode}; large PDB omitted)`
          : `Copied ${url.length}-char link (${mode})`,
      );
    } catch (err) {
      runError = err instanceof Error ? err.message : String(err);
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
      nodeStates = {};
      outputsByNode = {};
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
    a.download = `${graph.id}.denovo.json`;
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
      runError = err instanceof Error ? err.message : 'Failed to parse workflow JSON';
    } finally {
      input.value = '';
    }
  }

  function downloadText(text: string, filename: string) {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
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

  function handleResidueClick(info: { chain: string; seqId: number; compId: string }) {
    const tag = `${info.chain}${info.seqId}`;
    const baseLabel = `${tag} (${info.compId})`;

    if (molPickMode === 'off') {
      lastPickedResidue = baseLabel;
      return;
    }

    const condTarget =
      (selectedNode?.adapterId === 'rfd4-conditioner' ? selectedNode : null) ??
      graph.nodes.find((n) => n.adapterId === 'rfd4-conditioner') ??
      null;
    const mpnnTarget =
      (selectedNode?.adapterId === 'web-protein-mpnn' ? selectedNode : null) ??
      graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn') ??
      null;

    if (molPickMode === 'hotspots' || molPickMode === 'motif') {
      if (!condTarget) {
        lastPickedResidue = baseLabel;
        return;
      }
      const key = molPickMode === 'motif' ? 'motifSpan' : 'hotspots';
      const updated = appendToken(condTarget.params[key], tag);
      if (updated !== null) {
        updateNodeParams(condTarget.id, {
          ...condTarget.params,
          taskPreset: 'custom',
          [key]: updated,
        });
      }
      lastPickedResidue = `${baseLabel} → ${molPickMode === 'motif' ? 'Motif' : 'Hotspots'}`;
      return;
    }

    if (molPickMode === 'mpnn') {
      if (!mpnnTarget) {
        lastPickedResidue = baseLabel;
        return;
      }
      const updated = appendToken(mpnnTarget.params.fixedPositions, tag);
      if (updated !== null) {
        updateNodeParams(mpnnTarget.id, { ...mpnnTarget.params, fixedPositions: updated });
      }
      lastPickedResidue = `${baseLabel} → Fix Positions (MPNN)`;
      return;
    }

    // Auto mode: target the selected conditioner or MPNN block explicitly (never silently mutate an unrelated block)
    if (selectedNode?.adapterId === 'rfd4-conditioner') {
      const brush = String(selectedNode.params.paintBrush ?? 'hotspots');
      const key = brush === 'motifSpan' ? 'motifSpan' : 'hotspots';
      const updated = appendToken(selectedNode.params[key], tag);
      if (updated !== null) {
        updateNodeParams(selectedNode.id, {
          ...selectedNode.params,
          taskPreset: 'custom',
          [key]: updated,
        });
      }
      lastPickedResidue = `${baseLabel} → ${key === 'motifSpan' ? 'Motif' : 'Hotspots'}`;
      return;
    }

    if (selectedNode?.adapterId === 'web-protein-mpnn') {
      const updated = appendToken(selectedNode.params.fixedPositions, tag);
      if (updated !== null) {
        updateNodeParams(selectedNode.id, { ...selectedNode.params, fixedPositions: updated });
      }
      lastPickedResidue = `${baseLabel} → Fix Positions (MPNN)`;
      return;
    }

    lastPickedResidue = baseLabel;
  }

  // Preset & pick-target helpers
  let activePreset = $derived(WORKFLOW_PRESETS.find((p) => p.id === selectedPresetId) ?? null);
  let hasConditionerNode = $derived(graph.nodes.some((n) => n.adapterId === 'rfd4-conditioner'));
  let hasMpnnNode = $derived(graph.nodes.some((n) => n.adapterId === 'web-protein-mpnn'));

  $effect(() => {
    if ((molPickMode === 'hotspots' || molPickMode === 'motif') && !hasConditionerNode) {
      molPickMode = 'auto';
    } else if (molPickMode === 'mpnn' && !hasMpnnNode) {
      molPickMode = 'auto';
    }
  });

  // Node- and port-scoped intermediate peek state
  let selectedNode = $derived(graph.nodes.find((n) => n.id === selectedNodeId) ?? graph.nodes[0] ?? null);

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
  let peekCondition = $derived<ConditionSpecArtifact | null>(
    activeArtifact?.kind === 'ConditionSpec' ? activeArtifact : null,
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

  let metricsArtifact = $derived<DesignabilityMetrics | null>(latestScorerHit?.metrics ?? null);

  let activeRow = $derived<DesignabilityRow | null>(
    peekMetrics?.rows[selectedRowIdx] ?? peekMetrics?.best ?? null,
  );
  let activeFold = $derived(
    peekFoldSet?.folds[selectedFoldIdx] ?? peekFoldSet?.folds[0] ?? null,
  );

  let isPeekingSequenceSlot = $derived(Boolean(peekSequences && rightPanelTab === 'peek'));

  // 3D Mol* viewport bindings derived strictly from the active peek modality
  let viewerDesignedPdb = $derived.by((): string | null => {
    if (peekMetrics) return activeRow?.designedPdb ?? contextStructure?.pdbText ?? null;
    if (peekStructure) return peekStructure.pdbText;
    if (peekCondition) return peekCondition.inputCif ?? contextStructure?.pdbText ?? null;
    if (peekSequences) {
      // Clear 3D viewport when peeking a sequence-only wire/port unless user explicitly
      // previews the parent backbone (keeps Parameters-tab 3D residue picking intact).
      if (rightPanelTab === 'peek' && !showSeqParentBackbone) return null;
      return contextStructure?.pdbText ?? null;
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
      (peekMetrics || peekCondition || (peekSequences && (!isPeekingSequenceSlot || showSeqParentBackbone))
        ? contextStructure
        : null),
  );

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
      return `${selectedNode?.label ?? 'Scorer'} · ${activeRow.sequenceId} · scRMSD ${activeRow.scRmsd.toFixed(2)} Å · pLDDT ${activeRow.meanPlddt.toFixed(1)}`;
    }
    if (peekFoldSet && activeFold) {
      return `${dirTag} · ${activeFold.sequenceId} · pLDDT ${activeFold.meanPlddt.toFixed(1)}`;
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
    if (peekCondition) {
      return viewerDesignedPdb
        ? `${dirTag} template · ${peekCondition.label}`
        : `${dirTag} · ${peekCondition.label}`;
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
      branchCandidateIdx,
      hiddenModelIds: new Set(hiddenCompareModelIds),
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
    if (comparisonState.pairwise) {
      return `Compare ${comparisonState.branches.length} blocks · ${comparisonState.pairwise.badgeA}↔${comparisonState.pairwise.badgeB} RMSD ${comparisonState.pairwise.rmsd.toFixed(2)} Å`;
    }
    return `Compare ${comparisonState.branches.length} blocks · ${comparisonState.effectiveLayout === 'side-by-side' ? 'Side-by-side' : 'Overlay'}`;
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
          sysMenuOpen = !sysMenuOpen;
        }}
        title="WebGPU status, VRAM & weight cache"
      >
        <span class="status-dot"></span>
        {activeGpuWorkerLabel ? `GPU: ${activeGpuWorkerLabel}` : 'WebGPU'}
      </button>

      {#if sysMenuOpen}
        <div class="hdr-popover sys-pop" role="dialog" aria-label="System & WebGPU status">
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
                disabled={isRunning}
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
              disabled={isRunning}
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
            <span>Import JSON</span>
            <input
              type="file"
              accept=".json"
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
          <div class="menu-sep"></div>
          <button
            type="button"
            class="menu-item"
            role="menuitem"
            disabled={isRunning || validationErrors.length > 0}
            onclick={() => runWorkflow(true)}
          >
            <span>Re-run Fresh</span>
          </button>
        </div>
      {/if}
    </div>

    <!-- Primary Run CTA -->
    <button
      type="button"
      class="btn-primary"
      class:is-running={isRunning}
      data-tour="run-btn"
      disabled={isRunning || validationErrors.length > 0}
      onclick={() => runWorkflow(false)}
      title="Run workflow (R or ⌘↵ / Ctrl+Enter)"
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
          onOpenTour={() => (showOnboarding = true)}
        />
      </SvelteFlowProvider>

      {#if shareToast}
        <div class="share-toast">{shareToast}</div>
      {/if}

      {#if showRecipe}
        <div class="drawer-scroll">
          <RecipeDrawer {graph} disabled={isRunning} onApplyGraph={handleGraphChange} />
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
    <section class="results-grid" class:stacked-pane={splitMode === 'horizontal'}>
      <div class="viewer-col">
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
            <label><input type="checkbox" bind:checked={illustrative} /> Illustrative</label>
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
              <label class="pick-mode" title="Choose what clicking a 3D residue modifies">
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
        </div>

        <MolSuperposeViewer
          designedPdb={isCompareMode ? null : viewerDesignedPdb}
          refoldPdb={isCompareMode ? null : viewerRefoldPdb}
          models={isCompareMode && comparisonState ? comparisonState.models : null}
          trajectoryPdb={isCompareMode ? null : (viewerStructureMeta?.trajectoryPdb ?? null)}
          frameIndex={showTrajectory ? trajFrame : undefined}
          {showDesigned}
          {showRefold}
          {showTrajectory}
          {illustrative}
          {theme}
          height={splitMode === 'horizontal' ? 350 : 460}
          label={isCompareMode ? compareViewerLabel : viewerLabel}
          emptyMessage={isCompareMode ? undefined : viewerEmptyMessage}
          emptyActionLabel={!isCompareMode && isPeekingSequenceSlot && !showSeqParentBackbone && contextStructure
            ? 'Preview parent backbone'
            : undefined}
          onEmptyAction={() => (showSeqParentBackbone = true)}
          onResidueClick={handleResidueClick}
          onToggleModelVisibility={toggleCompareModelVisibility}
          onSoloModel={soloCompareModel}
        />

        {#if !isCompareMode && viewerStructureMeta}
          <div class="geom-strip">
            <span>L = <b>{viewerStructureMeta.length} aa</b></span>
            <span>R<sub>g</sub> = <b>{viewerStructureMeta.geometry.rg.toFixed(1)} Å</b></span>
            <span>α <b>{viewerStructureMeta.geometry.helixPct.toFixed(0)}%</b> · β <b>{viewerStructureMeta.geometry.sheetPct.toFixed(0)}%</b></span>
            <span>Cα breaks: <b>{viewerStructureMeta.geometry.chainBreaks}</b></span>
            <span>Clashes: <b>{viewerStructureMeta.geometry.clashes}</b></span>
            {#if totalElapsedMs !== null}
              <span>Total: <b>{(totalElapsedMs / 1000).toFixed(1)}s</b></span>
            {/if}
          </div>
        {/if}
      </div>

      <div class="right-panel" data-tour="inspector-panel">
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
              title="Compare branches in 3D (or ⌘-click / Shift-click blocks on the canvas)"
              onclick={() => handleSelectNodes(comparableBranchIds)}
            >
              Compare branches ({comparableBranchIds.length})
            </button>
          {:else if selectedNode && !isCompareMode}
            <span class="tab-node-ctx" title={selectedNode.label ?? selectedNode.adapterId}>
              {selectedNode.label ?? ADAPTER_REGISTRY[selectedNode.adapterId]?.shortName ?? selectedNode.adapterId}
            </span>
          {/if}
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
                onUpdate={updateNodeParams}
                onUpdateLabel={updateNodeLabel}
                onRunUpTo={(id) => runWorkflow(false, id)}
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
              {isRunning}
              onSelectBranchCandidate={(nodeId, idx) => {
                branchCandidateIdx = { ...branchCandidateIdx, [nodeId]: idx };
              }}
              onFocusSingle={(nodeId) => handleSelectNode(nodeId)}
              onRunUpToNode={(id) => runWorkflow(false, id)}
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
  }
  .hdr-right {
    display: flex;
    align-items: center;
    gap: 7px;
    flex-wrap: wrap;
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
    max-width: 340px;
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
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35), var(--node-shadow);
  }
  .sys-pop {
    width: 260px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 11.5px;
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
