<script lang="ts">
  import { Handle, Position } from '@xyflow/svelte';
  import {
    ADAPTER_REGISTRY,
    LOCALFOLD_MODEL_OPTIONS,
    RFD4_BENCHMARK_OPTIONS,
  } from '$lib/engine/registry';
  import { PORT_COLORS, STAGE_COLORS, STAGE_LABELS } from '$lib/engine/graph-utils';
  import type { CanonicalArtifact, NodeProgressEvent } from '$lib/engine/types';

  export interface ProteinNodeData {
    adapterId: string;
    label: string;
    params?: Record<string, unknown>;
    state?: NodeProgressEvent;
    outputs?: Record<string, CanonicalArtifact>;
    isPipelineRunning?: boolean;
    inputPortsConnected?: Record<string, boolean>;
    inputPortsReady?: Record<string, boolean>;
    outputPortsReady?: Record<string, boolean>;
    compareColor?: string;
    compareBadge?: string;
    autoModeHint?: string;
    onDelete?: (nodeId: string) => void;
    onPeekPort?: (nodeId: string, direction: 'in' | 'out', portName: string) => void;
    [key: string]: unknown;
  }

  interface Props {
    id: string;
    data: ProteinNodeData;
    selected?: boolean;
  }

  let { id, data, selected = false }: Props = $props();

  let manifest = $derived(ADAPTER_REGISTRY[data.adapterId]);
  let inputs = $derived(Object.entries(manifest?.inputs ?? {}));
  let outputs = $derived(Object.entries(manifest?.outputs ?? {}));
  let stageColor = $derived(STAGE_COLORS[manifest?.stage ?? 'source'] ?? '#64748b');
  let stageLabel = $derived(STAGE_LABELS[manifest?.stage ?? ''] ?? manifest?.stage ?? 'node');
  let runtimeLabel = $derived(manifest?.runtime === 'cpu-ts' ? 'CPU' : 'GPU');
  let activeWeightsMB = $derived.by((): number => {
    if (data.adapterId === 'localfold-af2') {
      const modelId = String(data.params?.model ?? 'monomer');
      const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
      return meta.approxMB;
    }
    return manifest?.approxWeightsMB ?? 0;
  });
  let st = $derived(data.state);
  let isPipelineRunning = $derived(Boolean(data.isPipelineRunning));
  let isNodeRunning = $derived(st?.status === 'running' || st?.status === 'loading');
  let isNodeDone = $derived(st?.status === 'done' || st?.status === 'cached');
  let isNodeQueued = $derived(isPipelineRunning && (!st || st.status === 'idle'));

  let paramSummary = $derived.by((): string => {
    const p = data.params ?? {};
    switch (data.adapterId) {
      case 'pdb-input': {
        if (p.mode === 'pdb-id') {
          const id = String(p.pdbId ?? '').trim().toUpperCase();
          return id ? `RCSB ${id}` : 'Fetch by PDB ID';
        }
        if (p.mode === 'custom') {
          const fname = String(p.filename ?? '').trim();
          return fname || 'custom PDB';
        }
        return '53 aa 3-helix demo';
      }
      case 'sequence-input': {
        const raw = String(p.sequence ?? '');
        const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
        const headers = lines.filter((l) => l.startsWith('>')).length;
        const aaLen = lines
          .filter((l) => !l.startsWith('>'))
          .join('')
          .replace(/[^A-Za-z]/g, '').length;
        if (headers > 1) return `${headers} seqs · ${aaLen} aa`;
        return `${aaLen || 0} aa FASTA`;
      }
      case 'rfd4-conditioner': {
        const tp = String(p.taskPreset ?? 'custom');
        if (tp !== 'custom') {
          const opt = RFD4_BENCHMARK_OPTIONS.find((o) => o.id === tp);
          return opt ? opt.label : tp;
        }
        const bits: string[] = [];
        const contig = String(p.contig ?? '').trim();
        if (contig) bits.push(contig);
        const hotCount = String(p.hotspots ?? '')
          .split(/[,\s]+/)
          .filter(Boolean).length;
        if (hotCount > 0) bits.push(`${hotCount} hotspot${hotCount === 1 ? '' : 's'}`);
        const motif = String(p.motifSpan ?? '').trim();
        if (motif) bits.push('motif');
        const cath = String(p.cathCode ?? '').trim();
        if (cath) bits.push(`CATH ${cath}`);
        return bits.join(' · ') || 'custom constraints';
      }
      case 'rfd4-proteina': {
        const mode = String(p.mode ?? 'unconditional');
        const steps = Number(p.nsteps ?? 60);
        if (mode === 'conditioned') return `Conditioned · ${steps} steps`;
        if (mode === 'preset') {
          const opt = RFD4_BENCHMARK_OPTIONS.find((o) => o.id === String(p.presetId ?? ''));
          return `${opt?.label ?? 'Preset task'} · ${steps} steps`;
        }
        if (mode === 'partial') return `Partial diffusion (${Number(p.partialFraction ?? 0.5)}) · ${steps} steps`;
        if (mode === 'inverse-folding') return `Inverse folding · ${steps} steps`;
        return `${Number(p.length ?? 70)} aa · ${steps} steps`;
      }
      case 'genie3': {
        const len = Number(p.length ?? 80);
        const steps = Number(p.nSampleStep ?? 100);
        if (data.autoModeHint) return `${data.autoModeHint} · ${len} aa · ${steps} steps`;
        return `${len} aa · ${steps} steps`;
      }
      case 'web-protein-mpnn': {
        const fam =
          p.family === 'proteinmpnn'
            ? 'ProteinMPNN'
            : p.family === 'ca'
              ? 'Cα-ProteinMPNN'
              : 'SolubleMPNN';
        const ckpt = String(p.modelName ?? 'v_48_020');
        const n = Number(p.numSequences ?? 2);
        const t = Number(p.temperature ?? 0.1);
        return `${fam} ${ckpt} · ${n} seq${n === 1 ? '' : 's'} · T=${t}`;
      }
      case 'localfold-af2': {
        const modelId = String(p.model ?? 'monomer');
        const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
        const rec = Number(p.recycles ?? 1);
        if (meta.family === 'af3') {
          const steps = Number(p.diffusionSteps ?? 25);
          return `${meta.shortLabel} · single-seq · ${steps} steps`;
        }
        return `${meta.shortLabel} · single-seq · ${rec} rec`;
      }
      case 'designability-scorer': {
        return `scRMSD < ${Number(p.scRmsdThreshold ?? 2.0).toFixed(1)} Å · pLDDT ≥ ${Number(p.plddtThreshold ?? 70)}`;
      }
      default:
        return '';
    }
  });

  let outputBadge = $derived.by((): string | null => {
    const outs = data.outputs;
    if (!outs) return null;
    for (const art of Object.values(outs)) {
      if (art.kind === 'DesignabilityMetrics' && art.best) {
        return `scRMSD ${art.best.scRmsd.toFixed(2)} Å · pLDDT ${art.best.meanPlddt.toFixed(1)}`;
      }
      if (art.kind === 'FoldedStructure' && art.folds.length > 0) {
        const bestP = Math.max(...art.folds.map((f) => f.meanPlddt));
        return `${art.folds.length} fold${art.folds.length > 1 ? 's' : ''} · pLDDT ${bestP.toFixed(1)}`;
      }
      if (art.kind === 'SequenceSet') {
        return `${art.candidates.length} seq${art.candidates.length > 1 ? 's' : ''} (${art.candidates[0]?.sequence.length ?? 0} aa)`;
      }
      if (art.kind === 'StructureArtifact') {
        return `L=${art.length} aa · Rg ${art.geometry.rg.toFixed(1)} Å`;
      }
      if (art.kind === 'ConditionSpec') {
        return art.summary;
      }
    }
    return null;
  });
</script>

<div
  class="p-node"
  class:selected
  class:comparing={Boolean(data.compareColor)}
  class:pipeline-running={isPipelineRunning}
  class:queued={isNodeQueued}
  class:running={isNodeRunning}
  class:loading={st?.status === 'loading'}
  class:done={isNodeDone}
  class:err={st?.status === 'error'}
  style="--stage-color: {stageColor}{data.compareColor ? `; --compare-color: ${data.compareColor}` : ''}"
>
  <div class="stripe-clip" aria-hidden="true"><div class="stripe"></div></div>

  {#if data.compareBadge && data.compareColor}
    <span
      class="compare-badge"
      style="background: {data.compareColor}"
      title={`Comparing in 3D as model ${data.compareBadge}`}
    >
      {data.compareBadge}
    </span>
  {/if}

  {#if data.onDelete}
    <button
      type="button"
      class="del-btn"
      title="Remove block (Backspace)"
      aria-label="Remove block"
      onclick={(e) => {
        e.stopPropagation();
        data.onDelete?.(id);
      }}
    >
      <svg width="9" height="9" viewBox="0 0 10 10" fill="none" aria-hidden="true">
        <path d="M2 2L8 8M8 2L2 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
      </svg>
    </button>
  {/if}

  <div class="header">
    <span class="stage">{stageLabel}</span>
    <span class="rt">{runtimeLabel}</span>
  </div>

  <div class="title">{data.label || manifest?.shortName || data.adapterId}</div>

  {#if paramSummary}
    <div class="param-chip" title={paramSummary}>{paramSummary}</div>
  {/if}

  <!-- Typed Input & Output Port Rows -->
  <div class="ports">
    <div class="col in-col">
      {#each inputs as [portName, spec] (portName)}
        {@const color = PORT_COLORS[spec.type]?.bg ?? '#94a3b8'}
        {@const portLabel = PORT_COLORS[spec.type]?.label ?? spec.type}
        {@const connected = Boolean(data.inputPortsConnected?.[portName])}
        {@const ready = Boolean(data.inputPortsReady?.[portName])}
        <div class="port-row left">
          <Handle
            type="target"
            position={Position.Left}
            id={portName}
            style={connected
              ? 'background: transparent; border: none; width: 8px; height: 12px; left: -1px;'
              : `background: ${color}; border: 1.5px solid var(--bg-surface); width: 8px; height: 8px; left: -4px;`}
          />
          <button
            type="button"
            class="port-label port-btn"
            class:ready
            title={`Click to inspect input "${portName}" (${portLabel})${spec.required === false ? ' [optional]' : ''}`}
            onclick={(e) => {
              e.stopPropagation();
              data.onPeekPort?.(id, 'in', portName);
            }}
          >
            {portName}{#if spec.required === false}<span class="opt">?</span>{/if}{#if ready}<span class="ready-dot" style="background: {color}"></span>{/if}
          </button>
        </div>
      {/each}
    </div>

    <div class="col out-col">
      {#each outputs as [portName, spec] (portName)}
        {@const color = PORT_COLORS[spec.type]?.bg ?? '#94a3b8'}
        {@const portLabel = PORT_COLORS[spec.type]?.label ?? spec.type}
        {@const ready = Boolean(data.outputPortsReady?.[portName])}
        <div class="port-row right">
          <button
            type="button"
            class="port-label port-btn"
            class:ready
            title={`Click to inspect output "${portName}" (${portLabel})`}
            onclick={(e) => {
              e.stopPropagation();
              data.onPeekPort?.(id, 'out', portName);
            }}
          >
            {#if ready}<span class="ready-dot" style="background: {color}"></span>{/if}{portName}
          </button>
          <Handle
            type="source"
            position={Position.Right}
            id={portName}
            style="background: {color}; border: 1.5px solid var(--bg-surface); width: 8px; height: 8px; right: -4px;"
          />
        </div>
      {/each}
    </div>
  </div>

  <!-- Status / Progress / Output Telemetry -->
  <div class="footer">
    {#if st && (st.status === 'loading' || st.status === 'running')}
      <div class="status-line active">{st.detail ?? st.status}</div>
      {#if typeof st.progress === 'number'}
        <div class="pbar"><div style="width: {Math.round(st.progress * 100)}%"></div></div>
      {/if}
    {:else if st?.status === 'error'}
      <div class="status-line err-text" title={st.error}>{st.error ?? 'Error'}</div>
    {:else if outputBadge}
      <div class="status-line ok-text">
        {#if st?.status === 'cached'}<span class="cache-tag">CACHED</span>{/if}
        {outputBadge}
      </div>
    {:else if isNodeQueued}
      <div class="status-line queued-text">queued…</div>
    {:else}
      <div class="status-line muted">
        {activeWeightsMB > 0 ? `~${activeWeightsMB} MB` : 'ready'}
      </div>
    {/if}
  </div>
</div>

<style>
  .p-node {
    position: relative;
    width: 224px;
    background: var(--bg-surface);
    --node-border-w: 1px;
    --node-border-style: solid;
    --node-border-color: var(--border);
    border-radius: 8px;
    color: var(--text-body);
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    box-shadow: var(--node-shadow);
    overflow: visible;
    transition:
      opacity 180ms ease,
      filter 180ms ease,
      box-shadow 180ms ease;
  }
  /* Crisp card border rendered in front of internal fills & top stage stripe */
  .p-node::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    border: var(--node-border-w) var(--node-border-style) var(--node-border-color);
    pointer-events: none;
    z-index: 2;
    transition: border-color 180ms ease;
  }
  .p-node :global(.svelte-flow__handle) {
    z-index: 4;
  }
  .p-node.running {
    opacity: 1;
    --node-border-w: 1.5px;
    --node-border-color: #f59e0b;
    box-shadow:
      0 0 0 3px rgba(245, 158, 11, 0.22),
      0 8px 22px rgba(245, 158, 11, 0.16);
    animation: runner-breathe 2s ease-in-out infinite;
  }
  .p-node.done {
    --node-border-color: rgba(34, 197, 94, 0.7);
  }
  .p-node.pipeline-running.done {
    opacity: 0.86;
  }
  .p-node.pipeline-running.done:hover {
    opacity: 1;
  }
  .p-node.queued {
    opacity: 0.48;
    filter: saturate(0.35);
    --node-border-style: dashed;
    --node-border-color: var(--border-strong);
  }
  .p-node.queued:hover {
    opacity: 0.8;
    filter: saturate(0.75);
  }
  .p-node.comparing {
    --node-border-color: var(--compare-color);
    box-shadow:
      0 0 0 2.5px color-mix(in srgb, var(--compare-color) 42%, transparent),
      var(--node-shadow);
  }
  .compare-badge {
    position: absolute;
    top: -8px;
    left: -8px;
    z-index: 5;
    width: 18px;
    height: 18px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 800;
    color: #090d16;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
    pointer-events: none;
  }
  .p-node.err {
    opacity: 1;
    filter: none;
    --node-border-w: 1.5px;
    --node-border-color: #ef4444;
    box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.22), var(--node-shadow);
  }
  /* Selected / peeking focus ring stays visible alongside state border & overrides dimming */
  .p-node.selected {
    opacity: 1;
    filter: none;
    --node-border-color: var(--accent);
    box-shadow: 0 0 0 2.5px rgba(59, 130, 246, 0.35), var(--node-shadow);
  }
  .p-node.selected.running {
    --node-border-w: 1.5px;
    --node-border-color: #f59e0b;
    box-shadow:
      0 0 0 2.5px var(--accent),
      0 0 0 5px rgba(245, 158, 11, 0.24),
      var(--node-shadow);
  }
  .p-node.selected.done {
    --node-border-color: #22c55e;
    box-shadow: 0 0 0 2.5px rgba(59, 130, 246, 0.4), var(--node-shadow);
  }
  .p-node.selected.err {
    --node-border-w: 1.5px;
    --node-border-color: #ef4444;
    box-shadow:
      0 0 0 2.5px var(--accent),
      0 0 0 5px rgba(239, 68, 68, 0.22),
      var(--node-shadow);
  }
  .stripe-clip {
    position: relative;
    height: 8px;
    border-radius: 8px 8px 0 0;
    overflow: hidden;
    margin-bottom: -4px;
    pointer-events: none;
  }
  .stripe {
    position: relative;
    height: 4px;
    background: var(--stage-color);
    overflow: hidden;
  }
  .p-node.loading .stripe::after,
  .p-node.running .stripe::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
      90deg,
      transparent 0%,
      rgba(255, 255, 255, 0.65) 50%,
      transparent 100%
    );
    animation: stripe-sheen 1.4s linear infinite;
  }
  .status-line.queued-text {
    color: var(--text-dim);
    font-style: italic;
  }
  @keyframes runner-breathe {
    0%,
    100% {
      box-shadow:
        0 0 0 3px rgba(245, 158, 11, 0.2),
        0 6px 18px rgba(245, 158, 11, 0.14);
    }
    50% {
      box-shadow:
        0 0 0 4.5px rgba(245, 158, 11, 0.34),
        0 10px 26px rgba(245, 158, 11, 0.24);
    }
  }
  @keyframes stripe-sheen {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(100%);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .p-node.running {
      animation: none;
    }
    .p-node.loading .stripe::after,
    .p-node.running .stripe::after {
      animation: none;
      display: none;
    }
  }
  .header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px 2px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .stage {
    color: var(--stage-color);
    font-weight: 700;
  }
  .rt {
    margin-left: auto;
    color: var(--text-dim);
    opacity: 0.72;
    font-size: 8.5px;
  }
  .del-btn {
    position: absolute;
    top: -7px;
    right: -7px;
    z-index: 4;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    padding: 0;
    border-radius: 999px;
    background: var(--bg-surface);
    color: var(--text-muted);
    border: 1px solid var(--border-strong);
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.14);
    cursor: pointer;
    opacity: 0;
    transform: scale(0.88);
    pointer-events: none;
    transition:
      opacity 120ms ease,
      transform 120ms ease,
      background-color 120ms ease,
      border-color 120ms ease,
      color 120ms ease;
  }
  .p-node:hover .del-btn,
  .p-node.selected .del-btn,
  .del-btn:focus-visible {
    opacity: 1;
    transform: scale(1);
    pointer-events: auto;
  }
  .del-btn:hover {
    background: #ef4444;
    border-color: #ef4444;
    color: #fff;
  }
  .title {
    padding: 0 10px 3px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .param-chip {
    margin: 0 10px 6px;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ports {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    padding: 4px 0 6px;
    border-top: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
    min-height: 26px;
  }
  .col {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .port-row {
    position: relative;
    display: flex;
    align-items: center;
    height: 20px;
  }
  .port-row.left {
    padding-left: 11px;
  }
  .port-row.right {
    padding-right: 11px;
    justify-content: flex-end;
  }
  .p-node :global(.svelte-flow__handle)::after {
    content: '';
    position: absolute;
    inset: -6px;
    border-radius: 50%;
  }
  .port-label {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-secondary);
  }
  .port-btn {
    background: none;
    border: none;
    padding: 1px 4px;
    border-radius: 3px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .port-btn:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
    text-decoration: underline;
  }
  .port-btn.ready {
    color: var(--text-primary);
  }
  .ready-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
  }
  .opt {
    color: var(--text-dim);
    margin-left: 1px;
  }
  .footer {
    padding: 6px 10px 7px;
  }
  .status-line {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .status-line.muted {
    color: var(--text-dim);
  }
  .status-line.active {
    color: #d97706;
  }
  .status-line.ok-text {
    color: var(--ok-text);
  }
  .status-line.err-text {
    color: var(--err-strong);
  }
  .cache-tag {
    background: var(--bg-pill);
    color: var(--accent-soft);
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9px;
    margin-right: 4px;
  }
  .pbar {
    height: 3px;
    background: var(--border-subtle);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 4px;
  }
  .pbar > div {
    height: 100%;
    background: #f59e0b;
    transition: width 0.1s;
  }
</style>
