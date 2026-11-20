<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { ExecutionScheduler, attachProgrammaticApi } from '$lib/engine/scheduler';
  import { ADAPTER_REGISTRY, SAMPLE_CAMPAIGN, WORKFLOW_PRESETS } from '$lib/engine/registry';
  import { clearAllModelCaches, getCacheStats } from '$lib/engine/opfs-cache';
  import type {
    CampaignState,
    CanonicalArtifact,
    DesignabilityMetrics,
    DesignabilityRow,
    NodeProgressEvent,
    SequenceSet,
    StructureArtifact,
    WorkflowGraph,
  } from '$lib/engine/types';
  import MolSuperposeViewer from '$lib/viewer/MolSuperposeViewer.svelte';
  import PaePlot from '$lib/viewer/PaePlot.svelte';
  import NodeInspector from '$lib/ui/NodeInspector.svelte';
  import CampaignViewer from '$lib/ui/CampaignViewer.svelte';

  const scheduler = new ExecutionScheduler();

  let activeTab = $state<'workbench' | 'campaign'>('workbench');
  let webgpuSupported = $state<boolean | null>(null);
  let webMcpReady = $state(false);
  let activeGpuWorker = $state<string | null>(null);
  let cacheUsedMB = $state<number>(0);

  // Active workflow state
  let selectedPresetId = $state(WORKFLOW_PRESETS[0].id);
  let graph = $state<WorkflowGraph>(structuredClone(WORKFLOW_PRESETS[0].graph));
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

  onMount(async () => {
    webgpuSupported = 'gpu' in navigator;
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
    graph = structuredClone(preset.graph);
    selectedNodeId = graph.nodes[0]?.id ?? '';
    nodeStates = {};
    outputsByNode = {};
    runError = null;
    totalElapsedMs = null;
    selectedRowIdx = 0;
    showTrajectory = false;
  }

  function updateNodeParams(nodeId: string, params: Record<string, unknown>) {
    graph = {
      ...graph,
      nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, params } : n)),
    };
  }

  async function runWorkflow(bypassCache = false) {
    if (isRunning) return;
    isRunning = true;
    runError = null;
    totalElapsedMs = null;
    nodeStates = {};
    try {
      const res = await scheduler.run(graph, handleProgress, { bypassCache });
      outputsByNode = res.outputsByNode;
      totalElapsedMs = res.totalElapsedMs;
      selectedRowIdx = 0;
      showTrajectory = false;
      activeGpuWorker = scheduler.getActiveGpuAdapter();
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
    const blob = new Blob([JSON.stringify(graph, null, 2)], { type: 'application/json' });
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
    // If an MPNN node exists in the current workflow, append clicked residue to fixedPositions
    const mpnnNode = graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn');
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
      label: `Round ${nextRoundNum} — ${graph.name}`,
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

  // Derived outputs from the current run
  let selectedNode = $derived(graph.nodes.find((n) => n.id === selectedNodeId) ?? graph.nodes[0]);

  let structureArtifact = $derived.by((): StructureArtifact | null => {
    for (const outs of Object.values(outputsByNode)) {
      for (const art of Object.values(outs)) {
        if (art.kind === 'StructureArtifact') return art;
      }
    }
    return null;
  });

  let sequenceArtifact = $derived.by((): SequenceSet | null => {
    for (const outs of Object.values(outputsByNode)) {
      for (const art of Object.values(outs)) {
        if (art.kind === 'SequenceSet') return art;
      }
    }
    return null;
  });

  let metricsArtifact = $derived.by((): DesignabilityMetrics | null => {
    for (const outs of Object.values(outputsByNode)) {
      for (const art of Object.values(outs)) {
        if (art.kind === 'DesignabilityMetrics') return art;
      }
    }
    return null;
  });

  let activeRow = $derived<DesignabilityRow | null>(
    metricsArtifact?.rows[selectedRowIdx] ?? metricsArtifact?.best ?? null,
  );

  let activePreset = $derived(WORKFLOW_PRESETS.find((p) => p.id === selectedPresetId));
</script>

<header class="topbar">
  <div class="brand">
    <h1><span class="accent">denovo</span>.run</h1>
    <span class="tagline">browser-native WebGPU protein design &amp; refolding orchestrator</span>
  </div>
  <div class="sys-pills">
    <span class="chip" class:ok={webgpuSupported} class:bad={webgpuSupported === false}>
      WebGPU: {webgpuSupported === null ? '…' : webgpuSupported ? 'ready' : 'unavailable'}
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
  </div>
</header>

<nav class="tabs">
  <button class:active={activeTab === 'workbench'} onclick={() => (activeTab = 'workbench')}>
    Pipeline Workbench
  </button>
  <button class:active={activeTab === 'campaign'} onclick={() => (activeTab = 'campaign')}>
    Campaign Inspector (BindClaw-style · {campaign.rounds.length} rounds)
  </button>
</nav>

{#if activeTab === 'workbench'}
  <!-- 1-Click Presets -->
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
        <div class="p-desc">{p.summary}</div>
      </button>
    {/each}
  </section>

  <!-- Visual Flowchart Stage Strip -->
  <section class="flow-panel">
    <div class="flow-head">
      <div>
        <span class="section-title">WORKFLOW GRAPH</span>
        <span class="flow-sub">click any node to edit parameters · upstream results memoize automatically</span>
      </div>
      <div class="flow-actions">
        <button class="btn-secondary" onclick={exportWorkflowJson}>Export JSON</button>
        {#if metricsArtifact}
          <button class="btn-secondary" onclick={recordRunToCampaign}>+ Save to Campaign</button>
        {/if}
        <button class="btn-secondary" disabled={isRunning} onclick={() => runWorkflow(true)}>
          Re-run Fresh
        </button>
        <button class="btn-primary" disabled={isRunning} onclick={() => runWorkflow(false)}>
          {isRunning ? 'Running on WebGPU…' : '▶ Run Pipeline'}
        </button>
      </div>
    </div>

    <div class="stage-strip">
      {#each graph.nodes as node, idx (node.id)}
        {@const st = nodeStates[node.id]}
        {@const manifest = ADAPTER_REGISTRY[node.adapterId]}
        {#if idx > 0}
          <div class="arrow">→</div>
        {/if}
        <button
          class="node-box"
          class:selected={node.id === selectedNodeId}
          class:running={st?.status === 'running' || st?.status === 'loading'}
          class:done={st?.status === 'done' || st?.status === 'cached'}
          class:err={st?.status === 'error'}
          onclick={() => (selectedNodeId = node.id)}
        >
          <div class="n-top">
            <span class="n-stage">{manifest?.stage ?? 'node'}</span>
            <span class="n-status">{st?.status ?? 'idle'}</span>
          </div>
          <div class="n-title">{node.label ?? manifest?.shortName ?? node.adapterId}</div>
          <div class="n-detail">{st?.detail ?? (manifest?.approxWeightsMB ? `~${manifest.approxWeightsMB} MB · ${manifest.runtime}` : manifest?.runtime)}</div>
          {#if st && (st.status === 'loading' || st.status === 'running') && typeof st.progress === 'number'}
            <div class="n-bar"><div style="width: {Math.round(st.progress * 100)}%"></div></div>
          {/if}
        </button>
      {/each}
    </div>

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
        refoldPdb={activeRow?.alignedFoldPdb ?? null}
        trajectoryPdb={structureArtifact?.trajectoryPdb ?? null}
        frameIndex={showTrajectory ? trajFrame : undefined}
        {showDesigned}
        {showRefold}
        {showTrajectory}
        {illustrative}
        height={440}
        label={activeRow
          ? `${activeRow.sequenceId} · scRMSD ${activeRow.scRmsd.toFixed(2)} Å · pLDDT ${activeRow.meanPlddt.toFixed(1)}`
          : structureArtifact?.label ?? activePreset?.name}
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
          <span class="section-title">DESIGNABILITY &amp; REFOLDING RESULTS</span>
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
                  Tip: Click any residue in the 3D viewer to pin it in ProteinMPNN's fixed positions, then click <b>Run Pipeline</b> — upstream generation stays memoized.
                </p>
              </div>
            </div>
          {/if}
        {:else if sequenceArtifact}
          <div class="placeholder-box">
            Designed {sequenceArtifact.candidates.length} sequence(s). Refolding in progress…
          </div>
        {:else}
          <div class="placeholder-box">
            Pick a preset above and click <b>▶ Run Pipeline</b> to execute the full client-side WebGPU loop.
            <br /><br />
            Start with <b>PDB → ProteinMPNN → LocalFold AF2</b> (~107 MB total, cached after first run) for the fastest end-to-end refolding test.
          </div>
        {/if}
      </div>
    </div>
  </section>
{:else}
  <CampaignViewer {campaign} onLoadCampaign={(c) => (campaign = c)} />
{/if}

<style>
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
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
    color: #f8fafc;
  }
  .accent {
    color: #60a5fa;
  }
  .tagline {
    font-size: 12px;
    color: #64748b;
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
    color: #94a3b8;
    background: #13151b;
    border: 1px solid #262932;
    padding: 4px 9px;
    border-radius: 999px;
  }
  button.chip {
    cursor: pointer;
  }
  button.chip:hover {
    border-color: #475569;
    color: #e2e8f0;
  }
  .chip.ok {
    color: #4ade80;
    border-color: #166534;
  }
  .chip.bad {
    color: #f87171;
    border-color: #7f1d1d;
  }
  .chip.gpu {
    color: #93c5fd;
    border-color: #1d4ed8;
  }

  .tabs {
    display: flex;
    gap: 6px;
    border-bottom: 1px solid #1e222b;
    margin-bottom: 16px;
  }
  .tabs button {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: #94a3b8;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
  }
  .tabs button.active {
    color: #f8fafc;
    border-bottom-color: #3b82f6;
  }

  .presets {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 10px;
    margin-bottom: 16px;
  }
  .preset-card {
    text-align: left;
    background: #13151b;
    border: 1px solid #262932;
    border-radius: 8px;
    padding: 12px 14px;
    cursor: pointer;
    color: #e2e8f0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .preset-card:hover:not(:disabled) {
    border-color: #475569;
  }
  .preset-card.selected {
    border-color: #3b82f6;
    background: #151c28;
  }
  .preset-card:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  .p-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .p-name {
    font-size: 13px;
    font-weight: 600;
  }
  .p-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: #93c5fd;
    background: #1e293b;
    padding: 2px 6px;
    border-radius: 4px;
    white-space: nowrap;
  }
  .p-desc {
    font-size: 11px;
    color: #94a3b8;
    line-height: 1.4;
  }

  .flow-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
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
    color: #94a3b8;
  }
  .flow-sub {
    margin-left: 10px;
    font-size: 11px;
    color: #64748b;
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
    opacity: 0.6;
    cursor: wait;
  }
  .btn-secondary {
    background: #13151b;
    color: #cbd5e1;
    border: 1px solid #334155;
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
  }
  .btn-secondary:hover:not(:disabled) {
    background: #1e293b;
  }

  .stage-strip {
    display: flex;
    align-items: stretch;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 4px;
  }
  .arrow {
    display: flex;
    align-items: center;
    color: #475569;
    font-size: 18px;
    font-weight: 700;
  }
  .node-box {
    min-width: 190px;
    flex: 1;
    text-align: left;
    background: #13151b;
    border: 1px solid #262932;
    border-radius: 8px;
    padding: 10px 12px;
    color: #e2e8f0;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .node-box.selected {
    border-color: #60a5fa;
    box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.35);
  }
  .node-box.running {
    border-color: #f59e0b;
  }
  .node-box.done {
    border-color: #22c55e;
  }
  .node-box.err {
    border-color: #ef4444;
  }
  .n-top {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    text-transform: uppercase;
  }
  .n-stage {
    color: #64748b;
  }
  .n-status {
    color: #93c5fd;
  }
  .n-title {
    font-size: 13px;
    font-weight: 600;
  }
  .n-detail {
    font-size: 11px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .n-bar {
    height: 3px;
    background: #1e222b;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 4px;
  }
  .n-bar > div {
    height: 100%;
    background: #3b82f6;
    transition: width 0.12s;
  }

  .error-banner {
    background: #2a1215;
    border: 1px solid #991b1b;
    color: #fecaca;
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
    color: #94a3b8;
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
    accent-color: #3b82f6;
  }
  .picked {
    margin-left: auto;
    font-size: 11px;
    color: #93c5fd;
  }
  .geom-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 18px;
    margin-top: 8px;
    padding: 8px 12px;
    background: #13151b;
    border: 1px solid #262932;
    border-radius: 6px;
    font-size: 11px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .geom-strip b {
    color: #e2e8f0;
  }

  .panel {
    background: #13151b;
    border: 1px solid #262932;
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
    background: #2a1c12;
    color: #fbbf24;
  }
  .pass-pill.good {
    background: #0f291a;
    color: #4ade80;
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
    color: #64748b;
    padding: 6px 8px;
    border-bottom: 1px solid #262932;
  }
  table.scores td {
    padding: 7px 8px;
    border-bottom: 1px solid #1e222b;
  }
  table.scores tr {
    cursor: pointer;
  }
  table.scores tr:hover {
    background: #191d26;
  }
  table.scores tr.selected {
    background: #172033;
  }
  .good-val {
    color: #4ade80;
    font-weight: 600;
  }
  .verdict {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    background: #3f181c;
    color: #fca5a5;
  }
  .verdict.pass {
    background: #123320;
    color: #4ade80;
  }

  .seq-box {
    background: #0d0e11;
    border: 1px solid #262932;
    border-radius: 6px;
    padding: 10px;
  }
  .seq-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: #94a3b8;
    margin-bottom: 6px;
  }
  .dl-links {
    display: flex;
    gap: 10px;
  }
  .link-btn {
    background: none;
    border: none;
    color: #60a5fa;
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
    color: #cbd5e1;
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
    color: #cbd5e1;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .tip {
    margin: 8px 0 0;
    font-size: 11px;
    color: #64748b;
    line-height: 1.45;
  }
  .placeholder-box {
    padding: 20px 10px;
    text-align: center;
    font-size: 12px;
    color: #64748b;
    line-height: 1.6;
  }
  code,
  .mono {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
</style>
