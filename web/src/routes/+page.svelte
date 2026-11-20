<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { SvelteFlowProvider } from '@xyflow/svelte';
  import { ExecutionScheduler, attachProgrammaticApi } from '$lib/engine/scheduler';
  import { ADAPTER_REGISTRY, SAMPLE_CAMPAIGN, WORKFLOW_PRESETS } from '$lib/engine/registry';
  import { autoLayoutGraph, validateGraph } from '$lib/engine/graph-utils';
  import {
    extractSubgraphUpTo,
    findContextStructure,
    getNodePeekSlots,
    makeEdgeId,
    pickDefaultPeekSlot,
  } from '$lib/engine/peek';
  import { decodeHashToGraph, encodeGraphToHash } from '$lib/engine/share';
  import { clearAllModelCaches, getCacheStats } from '$lib/engine/opfs-cache';
  import type {
    CampaignState,
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
  import NodeInspector from '$lib/ui/NodeInspector.svelte';
  import WorkflowCanvas from '$lib/ui/WorkflowCanvas.svelte';
  import RecipeDrawer from '$lib/ui/RecipeDrawer.svelte';
  import CampaignViewer from '$lib/ui/CampaignViewer.svelte';

  const scheduler = new ExecutionScheduler();

  let activeTab = $state<'workbench' | 'campaign'>('workbench');
  let theme = $state<'dark' | 'light'>('dark');
  let webgpuSupported = $state<boolean | null>(null);
  let webgpuDetail = $state<string>('checking…');
  let webMcpReady = $state(false);
  let activeGpuWorker = $state<string | null>(null);
  let cacheUsedMB = $state<number>(0);

  // Active workflow state
  let selectedPresetId = $state(WORKFLOW_PRESETS[0].id);
  let graph = $state<WorkflowGraph>(autoLayoutGraph(structuredClone(WORKFLOW_PRESETS[0].graph)));
  let selectedNodeId = $state<string>(WORKFLOW_PRESETS[0].graph.nodes[0].id);
  let isRunning = $state(false);
  let showRecipe = $state(false);
  let shareToast = $state<string | null>(null);
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
  let lastPickedResidue = $state<string | null>(null);

  // Campaign Inspector state (BindClaw-style multi-round log)
  let campaign = $state<CampaignState>(structuredClone(SAMPLE_CAMPAIGN));

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
    if (window.location.hash.length > 2) {
      try {
        const decoded = await decodeHashToGraph(window.location.hash);
        if (decoded) {
          graph = decoded;
          selectedPresetId = decoded.id;
          selectedNodeId = decoded.nodes[0]?.id ?? '';
          shareToast = 'Loaded shared workflow from URL';
          setTimeout(() => (shareToast = null), 3000);
        }
      } catch {
        /* ignore malformed hash */
      }
    }

    await refreshCacheStats();
  });

  onDestroy(() => {
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
    selectedNodeId = graph.nodes[0]?.id ?? '';
    selectedPeekSlotId = null;
    selectedEdgeId = null;
    nodeStates = {};
    outputsByNode = {};
    runError = null;
    totalElapsedMs = null;
    selectedRowIdx = 0;
    selectedFoldIdx = 0;
    showTrajectory = false;
  }

  function handleSelectNode(nodeId: string) {
    selectedNodeId = nodeId;
    selectedPeekSlotId = null;
    selectedEdgeId = null;
  }

  function handleSelectEdge(edge: WorkflowEdge) {
    selectedNodeId = edge.toNode;
    selectedPeekSlotId = `in:${edge.toPort}`;
    selectedEdgeId = makeEdgeId(edge.fromNode, edge.fromPort, edge.toNode, edge.toPort);
  }

  function handlePeekPort(nodeId: string, direction: 'in' | 'out', portName: string) {
    selectedNodeId = nodeId;
    selectedPeekSlotId = `${direction}:${portName}`;
    if (direction === 'in') {
      const edge = graph.edges.find((e) => e.toNode === nodeId && e.toPort === portName);
      selectedEdgeId = edge ? makeEdgeId(edge.fromNode, edge.fromPort, edge.toNode, edge.toPort) : null;
    } else {
      selectedEdgeId = null;
    }
  }

  function handleSelectPeekSlot(slotId: string) {
    selectedPeekSlotId = slotId;
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
    if (!nextById.has(selectedNodeId)) {
      selectedNodeId = next.nodes[0]?.id ?? '';
    }
    runError = null;
  }

  function updateNodeParams(nodeId: string, params: Record<string, unknown>) {
    graph = {
      ...graph,
      nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, params } : n)),
    };
  }

  let validationErrors = $derived(validateGraph(graph));

  async function runWorkflow(bypassCache = false, upToNodeId?: string) {
    if (isRunning) return;
    const plainGraph = $state.snapshot(graph) as WorkflowGraph;
    const targetGraph = upToNodeId ? extractSubgraphUpTo(plainGraph, upToNodeId) : plainGraph;
    const errs = validateGraph(targetGraph);
    if (errs.length > 0) {
      runError = errs.join('\n');
      return;
    }
    isRunning = true;
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
        selectedNodeId = upToNodeId;
        selectedPeekSlotId = null;
        selectedEdgeId = null;
      } else {
        // Auto-focus a scorer or fold node if present so results appear immediately
        const lastScorer = [...graph.nodes].reverse().find((n) => n.adapterId === 'designability-scorer');
        const lastFold = [...graph.nodes].reverse().find((n) => n.adapterId === 'localfold-af2');
        if (lastScorer) handleSelectNode(lastScorer.id);
        else if (lastFold) handleSelectNode(lastFold.id);
      }

      await refreshCacheStats();
    } catch (err) {
      runError = err instanceof Error ? err.message : String(err);
    } finally {
      isRunning = false;
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
    try {
      const plain = $state.snapshot(graph) as WorkflowGraph;
      const { hash, mode, strippedPdb } = await encodeGraphToHash(plain);
      const url = `${window.location.origin}${window.location.pathname}${hash}`;
      window.history.replaceState(null, '', hash);
      await navigator.clipboard.writeText(url);
      shareToast = strippedPdb
        ? `Copied ${url.length}-char link (${mode}; large PDB omitted)`
        : `Copied ${url.length}-char link (${mode})`;
      setTimeout(() => (shareToast = null), 3200);
    } catch (err) {
      runError = err instanceof Error ? err.message : String(err);
    }
  }

  function exportWorkflowJson() {
    const plain = $state.snapshot(graph);
    const blob = new Blob([JSON.stringify(plain, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${graph.id}.denovo.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
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

  function handleResidueClick(info: { chain: string; seqId: number; compId: string }) {
    const tag = `${info.chain}${info.seqId}`;
    lastPickedResidue = `${tag} (${info.compId})`;

    // If an RFD4 Condition block is selected, paint into its active brush field (hotspots or motifSpan)
    if (selectedNode?.adapterId === 'rfd4-conditioner') {
      const brush = String(selectedNode.params.paintBrush ?? 'hotspots');
      const key = brush === 'motifSpan' ? 'motifSpan' : 'hotspots';
      const prev = String(selectedNode.params[key] ?? '').trim();
      const parts = prev ? prev.split(/[,\s]+/).filter(Boolean) : [];
      if (!parts.includes(tag)) {
        parts.push(tag);
        updateNodeParams(selectedNode.id, {
          ...selectedNode.params,
          taskPreset: 'custom',
          [key]: parts.join(', '),
        });
      }
      return;
    }

    const mpnnNode =
      (selectedNode?.adapterId === 'web-protein-mpnn' ? selectedNode : null) ??
      graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn');
    if (!mpnnNode) return;
    const prev = String(mpnnNode.params.fixedPositions ?? '').trim();
    const parts = prev ? prev.split(/[,\s]+/).filter(Boolean) : [];
    if (!parts.includes(tag)) {
      parts.push(tag);
      updateNodeParams(mpnnNode.id, { ...mpnnNode.params, fixedPositions: parts.join(', ') });
    }
  }

  function recordRunToCampaign() {
    if (!metricsArtifact) return;
    const nextRoundNum = (campaign.rounds.at(-1)?.round ?? 0) + 1;
    const genNode =
      graph.nodes.find((n) => ADAPTER_REGISTRY[n.adapterId]?.stage === 'generate') ??
      graph.nodes.find((n) => ADAPTER_REGISTRY[n.adapterId]?.stage === 'source') ??
      graph.nodes[0];
    const condNode = graph.nodes.find((n) => n.adapterId === 'rfd4-conditioner');
    const foldNode = graph.nodes.find((n) => ADAPTER_REGISTRY[n.adapterId]?.stage === 'fold');
    const cleanParams: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(genNode?.params ?? {})) {
      if (k === 'pdbText' || k === 'rawJsonOverride' || k === 'customSpecYaml') continue;
      cleanParams[k] = v;
    }
    const condSummary = condNode
      ? `cond(${String(condNode.params.taskPreset !== 'custom' ? condNode.params.taskPreset : condNode.params.contig ?? '')}) · `
      : '';
    const best = metricsArtifact.best;
    const newRound = {
      round: nextRoundNum,
      timestamp: new Date().toISOString(),
      label: `Round ${nextRoundNum} — ${selectedNode?.label ?? graph.name}`,
      generator: ADAPTER_REGISTRY[genNode?.adapterId ?? '']?.displayName ?? genNode?.adapterId ?? 'unknown',
      folder: ADAPTER_REGISTRY[foldNode?.adapterId ?? '']?.displayName ?? 'LocalFold AF2 (WebGPU INT5)',
      nDesigned: metricsArtifact.rows.length,
      nPassed: metricsArtifact.rows.filter((r) => r.designable).length,
      hitRate: metricsArtifact.passRate,
      bestScRmsd: best ? best.scRmsd : null,
      bestPlddt: best ? best.meanPlddt : null,
      conditioningSummary: `${condSummary}${JSON.stringify(cleanParams)}`.slice(0, 120),
      agentNotes: `Recorded live in browser from denovo.run (${((totalElapsedMs ?? 0) / 1000).toFixed(1)}s).`,
      designedPdb: best?.designedPdb,
      alignedFoldPdb: best?.alignedFoldPdb,
      sequence: best?.sequence,
      pae: best?.pae ? Array.from(best.pae) : undefined,
    };
    campaign = { ...campaign, rounds: [...campaign.rounds, newRound] };
    activeTab = 'campaign';
  }

  // Node- and port-scoped intermediate peek state
  let selectedNode = $derived(graph.nodes.find((n) => n.id === selectedNodeId) ?? graph.nodes[0] ?? null);

  let peekSlots = $derived.by(() =>
    selectedNode ? getNodePeekSlots(graph, selectedNode.id, outputsByNode) : [],
  );

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

  // 3D Mol* viewport bindings derived strictly from the active peek modality
  let viewerDesignedPdb = $derived.by((): string | null => {
    if (peekMetrics) return activeRow?.designedPdb ?? contextStructure?.pdbText ?? null;
    if (peekStructure) return peekStructure.pdbText;
    if (peekCondition) return peekCondition.inputCif ?? contextStructure?.pdbText ?? null;
    if (peekSequences) return contextStructure?.pdbText ?? null;
    // Standalone folds are not Kabsch-aligned to the designed backbone, so keep designedPdb null
    return null;
  });

  let viewerRefoldPdb = $derived.by((): string | null => {
    if (peekMetrics) return activeRow?.alignedFoldPdb ?? null;
    if (peekFoldSet) return activeFold?.pdbText ?? null;
    return null;
  });

  let viewerStructureMeta = $derived<StructureArtifact | null>(
    peekStructure ?? (peekMetrics || peekSequences || peekCondition ? contextStructure : null),
  );

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
    if (peekSequences && contextStructure) {
      return `Context backbone (${contextStructure.label}) · peeking ${dirTag}`;
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

  // Active scorer thresholds (for dynamic table highlights)
  let activeScorerNode = $derived(
    selectedNode?.adapterId === 'designability-scorer'
      ? selectedNode
      : [...graph.nodes].reverse().find((n) => n.adapterId === 'designability-scorer') ?? null,
  );
  let scRmsdThreshold = $derived(Number(activeScorerNode?.params.scRmsdThreshold ?? 2.0));
  let plddtThreshold = $derived(Number(activeScorerNode?.params.plddtThreshold ?? 70));

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

<header class="topbar">
  <div class="brand">
    <h1><span class="accent">denovo</span>.run</h1>
    <span class="tagline">model-agnostic WebGPU protein design canvas</span>
  </div>
  <div class="sys-pills">
    <span class="chip" class:ok={webgpuSupported} class:bad={webgpuSupported === false}>
      WebGPU: {webgpuDetail}
    </span>
    {#if activeGpuWorker}
      <button class="chip gpu" disabled={isRunning} onclick={freeVram} title="Dispose & terminate active WebGPU worker">
        GPU: {activeGpuWorker} · free VRAM ×
      </button>
    {/if}
    {#if webMcpReady}
      <span class="chip ok">WebMCP active</span>
    {/if}
    <button class="chip" disabled={isRunning} onclick={handleClearCache} title="Clear OPFS & CacheStorage weight caches">
      Cache: {cacheUsedMB.toFixed(0)} MB · clear
    </button>
    <button class="chip theme-btn" onclick={toggleTheme} title="Switch light / dark theme">
      {theme === 'dark' ? '☀ Light' : '☾ Dark'}
    </button>
  </div>
</header>

<nav class="tabs">
  <button class:active={activeTab === 'workbench'} onclick={() => (activeTab = 'workbench')}>
    Node Canvas &amp; Workbench
  </button>
  <button class:active={activeTab === 'campaign'} onclick={() => (activeTab = 'campaign')}>
    Campaign Inspector (BindClaw-style · {campaign.rounds.length} rounds)
  </button>
</nav>

{#if activeTab === 'workbench'}
  <!-- Compact Preset Strip -->
  <section class="presets">
    {#each WORKFLOW_PRESETS as p (p.id)}
      <button
        class="preset-card"
        class:selected={p.id === selectedPresetId}
        disabled={isRunning}
        onclick={() => selectPreset(p.id)}
      >
        <div class="p-top">
          <span class="p-name">{p.name}</span>
          <span class="p-badge">{p.badge}</span>
        </div>
      </button>
    {/each}
  </section>

  <!-- Interactive Node-Graph Canvas + Run Bar -->
  <section class="flow-panel">
    <div class="flow-head">
      <div>
        <span class="section-title">WORKFLOW CANVAS</span>
        <span class="flow-sub">drag blocks · connect colored ports · click any block, port, or wire to peek intermediate data</span>
      </div>
      <div class="flow-actions">
        {#if shareToast}
          <span class="share-toast">{shareToast}</span>
        {/if}
        <button class="btn-secondary" onclick={() => (showRecipe = !showRecipe)}>
          {showRecipe ? 'Hide Recipe' : '</> Recipe'}
        </button>
        <button class="btn-secondary" onclick={copyQuickShareLink}>🔗 Share Link</button>
        <button class="btn-secondary" onclick={exportWorkflowJson}>Export JSON</button>
        {#if metricsArtifact}
          <button class="btn-secondary" onclick={recordRunToCampaign}>+ Save to Campaign</button>
        {/if}
        <button class="btn-secondary" disabled={isRunning} onclick={() => runWorkflow(true)}>
          Re-run Fresh
        </button>
        <button
          class="btn-primary"
          disabled={isRunning || validationErrors.length > 0}
          onclick={() => runWorkflow(false)}
        >
          {isRunning ? 'Running on WebGPU…' : '▶ Run Canvas'}
        </button>
      </div>
    </div>

    <SvelteFlowProvider>
      <WorkflowCanvas
        {graph}
        {selectedNodeId}
        {selectedEdgeId}
        {nodeStates}
        {outputsByNode}
        {theme}
        disabled={isRunning}
        onGraphChange={handleGraphChange}
        onSelectNode={handleSelectNode}
        onSelectEdge={handleSelectEdge}
        onPeekPort={handlePeekPort}
      />
    </SvelteFlowProvider>

    {#if showRecipe}
      <RecipeDrawer {graph} disabled={isRunning} onApplyGraph={handleGraphChange} />
    {/if}

    {#if validationErrors.length > 0}
      <div class="warn-banner">
        <b>Wire check:</b> {validationErrors.join(' · ')}
      </div>
    {/if}

    {#if runError}
      <div class="error-banner">
        <b>Pipeline error:</b> {runError}
      </div>
    {/if}

    {#if selectedNode}
      <NodeInspector
        node={selectedNode}
        disabled={isRunning}
        onUpdate={updateNodeParams}
        onRunUpTo={(id) => runWorkflow(false, id)}
      />
    {/if}
  </section>

  <!-- Live Results & Intermediate Peek: Mol* Viewport + Modality Inspector -->
  <section class="results-grid">
    <div class="viewer-col">
      <div class="viewer-bar">
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
        {#if lastPickedResidue}
          <span class="picked">Clicked: <code>{lastPickedResidue}</code></span>
        {/if}
      </div>

      <MolSuperposeViewer
        designedPdb={viewerDesignedPdb}
        refoldPdb={viewerRefoldPdb}
        trajectoryPdb={viewerStructureMeta?.trajectoryPdb ?? null}
        frameIndex={showTrajectory ? trajFrame : undefined}
        {showDesigned}
        {showRefold}
        {showTrajectory}
        {illustrative}
        {theme}
        height={440}
        label={viewerLabel}
        onResidueClick={handleResidueClick}
      />

      {#if viewerStructureMeta}
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

    <div class="metrics-col">
      <ArtifactPeekPanel
        node={selectedNode}
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
          if (latestScorerHit) handleSelectNode(latestScorerHit.nodeId);
        }}
        onDownloadText={downloadText}
      />
    </div>
  </section>
{:else}
  <CampaignViewer {campaign} {theme} onLoadCampaign={(c) => (campaign = c)} />
{/if}

<style>
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 14px;
  }
  .brand {
    display: flex;
    align-items: baseline;
    gap: 12px;
    flex-wrap: wrap;
  }
  h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--text-primary);
  }
  .accent {
    color: var(--accent-bright);
  }
  .tagline {
    font-size: 12px;
    color: var(--text-dim);
  }
  .sys-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }
  .chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-surface);
    border: 1px solid var(--border);
    padding: 4px 9px;
    border-radius: 999px;
  }
  button.chip {
    cursor: pointer;
  }
  button.chip:hover:not(:disabled) {
    border-color: var(--border-strong);
    color: var(--text-primary);
  }
  button.chip:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .chip.theme-btn {
    color: var(--text-body);
    font-weight: 500;
  }
  .chip.ok {
    color: var(--ok-text);
    border-color: var(--ok-border);
    background: var(--ok-bg);
  }
  .chip.bad {
    color: var(--err-strong);
    border-color: var(--err-border);
    background: var(--err-bg);
  }
  .chip.gpu {
    color: var(--accent-soft);
    border-color: var(--accent);
    background: var(--bg-pill);
  }

  .tabs {
    display: flex;
    gap: 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
  }
  .tabs button {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-muted);
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
  }
  .tabs button.active {
    color: var(--text-primary);
    border-bottom-color: var(--accent);
  }

  .presets {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 4px;
    margin-bottom: 12px;
  }
  .preset-card {
    text-align: left;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 8px 11px;
    cursor: pointer;
    color: var(--text-body);
    flex-shrink: 0;
  }
  .preset-card:hover:not(:disabled) {
    border-color: var(--border-strong);
  }
  .preset-card.selected {
    border-color: var(--accent);
    background: var(--bg-surface-selected);
  }
  .preset-card:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  .p-top {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .p-name {
    font-size: 12px;
    font-weight: 600;
  }
  .p-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--accent-soft);
    background: var(--bg-pill);
    padding: 2px 6px;
    border-radius: 4px;
    white-space: nowrap;
  }

  .flow-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 18px;
  }
  .flow-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
  }
  .section-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: var(--text-muted);
  }
  .flow-sub {
    margin-left: 10px;
    font-size: 11px;
    color: var(--text-dim);
  }
  .flow-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }
  .share-toast {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--ok-text);
    background: var(--ok-bg);
    border: 1px solid var(--ok-border);
    padding: 4px 8px;
    border-radius: 5px;
  }
  .btn-primary {
    background: #2563eb;
    color: #fff;
    border: 1px solid #3b82f6;
    padding: 7px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
  }
  .btn-primary:hover:not(:disabled) {
    background: #1d4ed8;
  }
  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn-secondary {
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
  }
  .btn-secondary:hover:not(:disabled) {
    background: var(--bg-surface-hover);
  }

  .warn-banner {
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .error-banner {
    background: var(--err-bg);
    border: 1px solid var(--err-border);
    color: var(--err-text);
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    white-space: pre-wrap;
  }

  .results-grid {
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    gap: 16px;
    align-items: start;
  }
  @media (max-width: 960px) {
    .results-grid {
      grid-template-columns: 1fr;
    }
  }
  .viewer-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 8px;
    font-size: 12px;
    color: var(--text-muted);
  }
  .viewer-bar label {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
  }
  .slider {
    flex: 1;
    min-width: 120px;
    accent-color: var(--accent);
  }
  .picked {
    margin-left: auto;
    font-size: 11px;
    color: var(--accent-soft);
  }
  .geom-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 18px;
    margin-top: 8px;
    padding: 8px 12px;
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
  code,
  .mono {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
</style>
