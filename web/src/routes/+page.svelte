<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { ExecutionScheduler, attachProgrammaticApi } from '$lib/engine/scheduler';
  import { ADAPTER_REGISTRY, SAMPLE_CAMPAIGN, WORKFLOW_PRESETS } from '$lib/engine/registry';
  import { autoLayoutGraph, validateGraph } from '$lib/engine/graph-utils';
  import { clearAllModelCaches, getCacheStats } from '$lib/engine/opfs-cache';
  import type {
    CampaignState,
    CanonicalArtifact,
    DesignabilityMetrics,
    DesignabilityRow,
    FoldedStructure,
    NodeProgressEvent,
    SequenceSet,
    StructureArtifact,
    WorkflowGraph,
  } from '$lib/engine/types';
  import MolSuperposeViewer from '$lib/viewer/MolSuperposeViewer.svelte';
  import PaePlot from '$lib/viewer/PaePlot.svelte';
  import NodeInspector from '$lib/ui/NodeInspector.svelte';
  import WorkflowCanvas from '$lib/ui/WorkflowCanvas.svelte';
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
  let runError = $state<string | null>(null);
  let totalElapsedMs = $state<number | null>(null);

  let nodeStates = $state<Record<string, NodeProgressEvent>>({});
  let outputsByNode = $state<Record<string, Record<string, CanonicalArtifact>>>({});

  // Viewer toggles
  let selectedRowIdx = $state(0);
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
    nodeStates = {};
    outputsByNode = {};
    runError = null;
    totalElapsedMs = null;
    selectedRowIdx = 0;
    showTrajectory = false;
  }

  function handleGraphChange(next: WorkflowGraph) {
    graph = next;
    runError = null;
  }

  function updateNodeParams(nodeId: string, params: Record<string, unknown>) {
    graph = {
      ...graph,
      nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, params } : n)),
    };
  }

  let validationErrors = $derived(validateGraph(graph));

  async function runWorkflow(bypassCache = false) {
    if (isRunning) return;
    const errs = validateGraph(graph);
    if (errs.length > 0) {
      runError = errs.join('\n');
      return;
    }
    isRunning = true;
    runError = null;
    totalElapsedMs = null;
    nodeStates = {};
    try {
      const plainGraph = $state.snapshot(graph) as WorkflowGraph;
      const res = await scheduler.run(plainGraph, handleProgress, { bypassCache });
      outputsByNode = res.outputsByNode;
      totalElapsedMs = res.totalElapsedMs;
      selectedRowIdx = 0;
      showTrajectory = false;
      activeGpuWorker = scheduler.getActiveGpuAdapter();

      // Auto-focus a scorer or fold node if present so results appear immediately
      const lastScorer = [...graph.nodes].reverse().find((n) => n.adapterId === 'designability-scorer');
      const lastFold = [...graph.nodes].reverse().find((n) => n.adapterId === 'localfold-af2');
      if (lastScorer) selectedNodeId = lastScorer.id;
      else if (lastFold) selectedNodeId = lastFold.id;

      await refreshCacheStats();
    } catch (err) {
      runError = err instanceof Error ? err.message : String(err);
    } finally {
      isRunning = false;
      activeGpuWorker = scheduler.getActiveGpuAdapter();
    }
  }

  async function freeVram() {
    await scheduler.releaseActiveGpu();
    activeGpuWorker = null;
  }

  async function handleClearCache() {
    await scheduler.releaseActiveGpu();
    scheduler.clearMemo();
    await clearAllModelCaches();
    await refreshCacheStats();
    activeGpuWorker = null;
  }

  function exportWorkflowJson() {
    const plain = $state.snapshot(graph);
    const blob = new Blob([JSON.stringify(plain, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${graph.id}.denovo.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadText(text: string, filename: string) {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleResidueClick(info: { chain: string; seqId: number; compId: string }) {
    const tag = `${info.chain}${info.seqId}`;
    lastPickedResidue = `${tag} (${info.compId})`;
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
    const genNode = graph.nodes[0];
    const best = metricsArtifact.best;
    const newRound = {
      round: nextRoundNum,
      timestamp: new Date().toISOString(),
      label: `Round ${nextRoundNum} — ${selectedNode?.label ?? graph.name}`,
      generator: ADAPTER_REGISTRY[genNode?.adapterId ?? '']?.displayName ?? genNode?.adapterId ?? 'unknown',
      folder: 'LocalFold AF2 (WebGPU INT5)',
      nDesigned: metricsArtifact.rows.length,
      nPassed: metricsArtifact.rows.filter((r) => r.designable).length,
      hitRate: metricsArtifact.passRate,
      bestScRmsd: best ? best.scRmsd : null,
      bestPlddt: best ? best.meanPlddt : null,
      conditioningSummary: JSON.stringify(genNode?.params ?? {}).slice(0, 120),
      agentNotes: `Recorded live in browser from denovo.run (${((totalElapsedMs ?? 0) / 1000).toFixed(1)}s).`,
      designedPdb: best?.designedPdb,
      alignedFoldPdb: best?.alignedFoldPdb,
      sequence: best?.sequence,
      pae: best?.pae ? Array.from(best.pae) : undefined,
    };
    campaign = { ...campaign, rounds: [...campaign.rounds, newRound] };
    activeTab = 'campaign';
  }

  // Node-scoped artifact lookup (prefers the currently selected canvas node so branched DAGs can be compared!)
  let selectedNode = $derived(graph.nodes.find((n) => n.id === selectedNodeId) ?? graph.nodes[0] ?? null);

  function findArtifact<T extends CanonicalArtifact>(kind: T['kind']): T | null {
    const selOuts = selectedNodeId ? outputsByNode[selectedNodeId] : undefined;
    if (selOuts) {
      for (const art of Object.values(selOuts)) {
        if (art.kind === kind) return art as T;
      }
    }
    // Fallback to most recent node emitting this kind
    for (const n of [...graph.nodes].reverse()) {
      const outs = outputsByNode[n.id];
      if (!outs) continue;
      for (const art of Object.values(outs)) {
        if (art.kind === kind) return art as T;
      }
    }
    return null;
  }

  let structureArtifact = $derived.by(() => findArtifact<StructureArtifact>('StructureArtifact'));
  let sequenceArtifact = $derived.by(() => findArtifact<SequenceSet>('SequenceSet'));
  let foldArtifact = $derived.by(() => findArtifact<FoldedStructure>('FoldedStructure'));
  let metricsArtifact = $derived.by(() => findArtifact<DesignabilityMetrics>('DesignabilityMetrics'));

  let activeRow = $derived<DesignabilityRow | null>(
    metricsArtifact?.rows[selectedRowIdx] ?? metricsArtifact?.best ?? null,
  );

  // If the user selected a standalone LocalFold node (e.g. Sequence Input -> LocalFold without a Scorer)
  let standaloneFold = $derived(
    !activeRow && foldArtifact?.folds[0] ? foldArtifact.folds[0] : null,
  );

  // Reset row index when switching between different scorer branches
  $effect(() => {
    void metricsArtifact?.id;
    selectedRowIdx = 0;
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
      <button class="chip gpu" onclick={freeVram} title="Dispose & terminate active WebGPU worker">
        GPU: {activeGpuWorker} · free VRAM ×
      </button>
    {/if}
    {#if webMcpReady}
      <span class="chip ok">WebMCP active</span>
    {/if}
    <button class="chip" onclick={handleClearCache} title="Clear OPFS & CacheStorage weight caches">
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
        <span class="flow-sub">drag blocks · connect matching colored ports · Backspace deletes · click any block to inspect its parameters &amp; outputs</span>
      </div>
      <div class="flow-actions">
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

    <WorkflowCanvas
      {graph}
      {selectedNodeId}
      {nodeStates}
      {outputsByNode}
      {theme}
      disabled={isRunning}
      onGraphChange={handleGraphChange}
      onSelectNode={(id) => (selectedNodeId = id)}
    />

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
      <NodeInspector node={selectedNode} disabled={isRunning} onUpdate={updateNodeParams} />
    {/if}
  </section>

  <!-- Live Results: Mol* Superposition + Designability Table + PAE -->
  <section class="results-grid">
    <div class="viewer-col">
      <div class="viewer-bar">
        <label><input type="checkbox" bind:checked={showDesigned} /> Designed</label>
        <label><input type="checkbox" bind:checked={showRefold} /> Refold (pLDDT)</label>
        <label><input type="checkbox" bind:checked={illustrative} /> Illustrative</label>
        {#if structureArtifact?.trajectoryPdb}
          <label><input type="checkbox" bind:checked={showTrajectory} /> Trajectory</label>
          {#if showTrajectory}
            <input
              type="range"
              min="0"
              max={Math.max(0, (structureArtifact.trajectoryFrameCount ?? 1) - 1)}
              bind:value={trajFrame}
              class="slider"
            />
            <code class="mono">{trajFrame + 1}/{structureArtifact.trajectoryFrameCount}</code>
          {/if}
        {/if}
        {#if lastPickedResidue}
          <span class="picked">Clicked: <code>{lastPickedResidue}</code></span>
        {/if}
      </div>

      <MolSuperposeViewer
        designedPdb={activeRow?.designedPdb ?? structureArtifact?.pdbText ?? null}
        refoldPdb={activeRow?.alignedFoldPdb ?? standaloneFold?.pdbText ?? null}
        trajectoryPdb={structureArtifact?.trajectoryPdb ?? null}
        frameIndex={showTrajectory ? trajFrame : undefined}
        {showDesigned}
        {showRefold}
        {showTrajectory}
        {illustrative}
        {theme}
        height={440}
        label={activeRow
          ? `${selectedNode?.label ?? 'Scorer'} · ${activeRow.sequenceId} · scRMSD ${activeRow.scRmsd.toFixed(2)} Å · pLDDT ${activeRow.meanPlddt.toFixed(1)}`
          : standaloneFold
            ? `${standaloneFold.sequenceId} · pLDDT ${standaloneFold.meanPlddt.toFixed(1)}`
            : structureArtifact?.label ?? 'denovo.run viewport'}
        onResidueClick={handleResidueClick}
      />

      {#if structureArtifact}
        <div class="geom-strip">
          <span>L = <b>{structureArtifact.length} aa</b></span>
          <span>R<sub>g</sub> = <b>{structureArtifact.geometry.rg.toFixed(1)} Å</b></span>
          <span>α <b>{structureArtifact.geometry.helixPct.toFixed(0)}%</b> · β <b>{structureArtifact.geometry.sheetPct.toFixed(0)}%</b></span>
          <span>Cα breaks: <b>{structureArtifact.geometry.chainBreaks}</b></span>
          <span>Clashes: <b>{structureArtifact.geometry.clashes}</b></span>
          {#if totalElapsedMs !== null}
            <span>Total: <b>{(totalElapsedMs / 1000).toFixed(1)}s</b></span>
          {/if}
        </div>
      {/if}
    </div>

    <div class="metrics-col">
      <div class="panel">
        <div class="panel-head">
          <span class="section-title">
            {selectedNode ? `OUTPUTS · ${selectedNode.label ?? selectedNode.adapterId}` : 'PIPELINE OUTPUTS'}
          </span>
          {#if metricsArtifact}
            <span class="pass-pill" class:good={metricsArtifact.passRate > 0}>
              {(metricsArtifact.passRate * 100).toFixed(0)}% designable
            </span>
          {/if}
        </div>

        {#if metricsArtifact && metricsArtifact.rows.length > 0}
          <table class="scores">
            <thead>
              <tr>
                <th>Seq</th>
                <th>scRMSD</th>
                <th>scTM</th>
                <th>pLDDT</th>
                <th>pTM</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {#each metricsArtifact.rows as row, i (row.sequenceId)}
                <tr
                  class:selected={i === selectedRowIdx}
                  onclick={() => (selectedRowIdx = i)}
                >
                  <td><code>{row.sequenceId}</code></td>
                  <td class:good-val={row.scRmsd < 2.0}>{row.scRmsd.toFixed(2)} Å</td>
                  <td>{row.scTm.toFixed(3)}</td>
                  <td class:good-val={row.meanPlddt >= 70}>{row.meanPlddt.toFixed(1)}</td>
                  <td>{row.ptm !== null ? row.ptm.toFixed(2) : '—'}</td>
                  <td>
                    <span class="verdict" class:pass={row.designable}>
                      {row.designable ? 'PASS' : 'FAIL'}
                    </span>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>

          {#if activeRow}
            <div class="seq-box">
              <div class="seq-head">
                <span>Selected Sequence ({activeRow.sequence.length} aa)</span>
                <div class="dl-links">
                  <button
                    class="link-btn"
                    onclick={() => downloadText(activeRow.designedPdb, `${activeRow.structureId}_designed.pdb`)}
                  >
                    ↓ Designed PDB
                  </button>
                  <button
                    class="link-btn"
                    onclick={() => downloadText(activeRow.alignedFoldPdb, `${activeRow.sequenceId}_refold_aligned.pdb`)}
                  >
                    ↓ Refold PDB
                  </button>
                </div>
              </div>
              <div class="seq-text">{activeRow.sequence}</div>
            </div>

            <div class="pae-row">
              <PaePlot pae={activeRow.pae} size={200} />
              <div class="pae-notes">
                <div><b>Fold time:</b> {(activeRow.foldElapsedMs / 1000).toFixed(1)} s</div>
                <div><b>Refold R<sub>g</sub>:</b> {activeRow.geometry.rg.toFixed(1)} Å</div>
                <div><b>Refold α/β:</b> {activeRow.geometry.helixPct.toFixed(0)}% / {activeRow.geometry.sheetPct.toFixed(0)}%</div>
                <p class="tip">
                  Tip: In branched graphs (like <b>RFD4 Fork</b>), click each Scorer block on the canvas to compare its superposition &amp; PAE!
                </p>
              </div>
            </div>
          {/if}
        {:else if standaloneFold}
          <div class="seq-box">
            <div class="seq-head">
              <span>Folded {standaloneFold.sequenceId} · pLDDT {standaloneFold.meanPlddt.toFixed(1)}</span>
              <button
                class="link-btn"
                onclick={() => downloadText(standaloneFold.pdbText, `${standaloneFold.sequenceId}_af2.pdb`)}
              >
                ↓ Folded PDB
              </button>
            </div>
            <div class="seq-text">{standaloneFold.sequence}</div>
          </div>
          <PaePlot pae={standaloneFold.pae} size={200} />
        {:else if sequenceArtifact}
          <div class="seq-box">
            <div class="seq-head">
              <span>{sequenceArtifact.candidates.length} Sequence(s) · {sequenceArtifact.sourceModelId}</span>
            </div>
            {#each sequenceArtifact.candidates as c (c.id)}
              <div class="seq-text" style="margin-bottom: 6px">
                <b>{c.id}</b> {c.score !== null ? `(score ${c.score.toFixed(2)})` : ''}: {c.sequence}
              </div>
            {/each}
          </div>
        {:else}
          <div class="placeholder-box">
            Wire any blocks on the canvas (or pick a preset above) and click <b>▶ Run Canvas</b>.
            <br /><br />
            Try <b>RFD4 Fork: Native vs SolubleMPNN</b> to see branched DAG execution comparing two folding arms on one RFD4 backbone!
          </div>
        {/if}
      </div>
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
  button.chip:hover {
    border-color: var(--border-strong);
    color: var(--text-primary);
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

  .panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .panel-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .pass-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--warn-bg);
    color: var(--warn-text);
  }
  .pass-pill.good {
    background: var(--ok-bg);
    color: var(--ok-text);
  }

  table.scores {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  table.scores th {
    text-align: left;
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-dim);
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
  }
  table.scores td {
    padding: 7px 8px;
    border-bottom: 1px solid var(--border-subtle);
  }
  table.scores tr {
    cursor: pointer;
  }
  table.scores tr:hover {
    background: var(--bg-surface-hover);
  }
  table.scores tr.selected {
    background: var(--bg-surface-selected);
  }
  .good-val {
    color: var(--ok-text);
    font-weight: 600;
  }
  .verdict {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--err-bg);
    color: var(--err-text);
  }
  .verdict.pass {
    background: var(--ok-bg);
    color: var(--ok-text);
  }

  .seq-box {
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }
  .seq-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }
  .dl-links {
    display: flex;
    gap: 10px;
  }
  .link-btn {
    background: none;
    border: none;
    color: var(--accent-bright);
    font-size: 11px;
    cursor: pointer;
    padding: 0;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .link-btn:hover {
    text-decoration: underline;
  }
  .seq-text {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-secondary);
    word-break: break-all;
    line-height: 1.5;
  }

  .pae-row {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    flex-wrap: wrap;
  }
  .pae-notes {
    flex: 1;
    min-width: 160px;
    font-size: 12px;
    color: var(--text-secondary);
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .tip {
    margin: 8px 0 0;
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1.45;
  }
  .placeholder-box {
    padding: 20px 10px;
    text-align: center;
    font-size: 12px;
    color: var(--text-dim);
    line-height: 1.6;
  }
  code,
  .mono {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
</style>
