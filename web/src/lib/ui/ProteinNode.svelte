<script lang="ts">
  import { Handle, Position } from '@xyflow/svelte';
  import { ADAPTER_REGISTRY, RFD4_BENCHMARK_OPTIONS } from '$lib/engine/registry';
  import { PORT_COLORS, STAGE_COLORS } from '$lib/engine/graph-utils';
  import type { CanonicalArtifact, NodeProgressEvent } from '$lib/engine/types';

  export interface ProteinNodeData {
    adapterId: string;
    label: string;
    params?: Record<string, unknown>;
    state?: NodeProgressEvent;
    outputs?: Record<string, CanonicalArtifact>;
    inputPortsReady?: Record<string, boolean>;
    outputPortsReady?: Record<string, boolean>;
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
  let st = $derived(data.state);

  let paramSummary = $derived.by((): string => {
    const p = data.params ?? {};
    switch (data.adapterId) {
      case 'pdb-input': {
        if (p.mode === 'custom') {
          const fname = String(p.filename ?? '').trim();
          return fname || 'custom PDB';
        }
        return '53 aa demo';
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
          return opt ? `${opt.category} · ${tp}` : tp;
        }
        const bits: string[] = [];
        const contig = String(p.contig ?? '').trim();
        if (contig) bits.push(contig);
        const hotCount = String(p.hotspots ?? '')
          .split(/[,\s]+/)
          .filter(Boolean).length;
        if (hotCount > 0) bits.push(`${hotCount} hot`);
        const motif = String(p.motifSpan ?? '').trim();
        if (motif) bits.push('motif');
        const cath = String(p.cathCode ?? '').trim();
        if (cath) bits.push(`CATH ${cath}`);
        return bits.join(' · ') || 'custom spec';
      }
      case 'rfd4-proteina': {
        const mode = String(p.mode ?? 'unconditional');
        const steps = Number(p.nsteps ?? 60);
        if (mode === 'conditioned') return `conditioned · ${steps} steps`;
        if (mode === 'preset') return `${String(p.presetId ?? 'preset')} · ${steps} steps`;
        if (mode === 'partial') return `partial ${Number(p.partialFraction ?? 0.5)} · ${steps} steps`;
        if (mode === 'inverse-folding') return `inv-fold · ${steps} steps`;
        return `${Number(p.length ?? 70)} aa · ${steps} steps`;
      }
      case 'web-protein-mpnn': {
        const fam =
          p.family === 'proteinmpnn'
            ? 'ProteinMPNN'
            : p.family === 'ca'
              ? 'Cα-MPNN'
              : 'SolubleMPNN';
        const n = Number(p.numSequences ?? 2);
        const t = Number(p.temperature ?? 0.1);
        return `${fam} · ${n} seq${n === 1 ? '' : 's'} · T=${t}`;
      }
      case 'localfold-af2': {
        return `${Number(p.recycles ?? 1)} rec · max ${Number(p.maxSequences ?? 2)}`;
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
  class:running={st?.status === 'running' || st?.status === 'loading'}
  class:done={st?.status === 'done' || st?.status === 'cached'}
  class:err={st?.status === 'error'}
  style="--stage-color: {stageColor}"
>
  <div class="stripe"></div>

  <div class="header">
    <span class="stage">{manifest?.stage ?? 'node'}</span>
    <span class="rt">{manifest?.runtime ?? 'cpu'}</span>
    {#if data.onDelete}
      <button
        type="button"
        class="del-btn"
        title="Remove block"
        aria-label="Remove block"
        onclick={(e) => {
          e.stopPropagation();
          data.onDelete?.(id);
        }}
      >
        ×
      </button>
    {/if}
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
        {@const ready = Boolean(data.inputPortsReady?.[portName])}
        <div class="port-row left">
          <Handle
            type="target"
            position={Position.Left}
            id={portName}
            style="background: {color}; border: 2px solid var(--bg-surface); width: 13px; height: 13px; left: -7px;"
          />
          <button
            type="button"
            class="port-label port-btn"
            class:ready
            title={`Click to peek input "${portName}" (${spec.type})${spec.required === false ? ' [optional]' : ''}`}
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
        {@const ready = Boolean(data.outputPortsReady?.[portName])}
        <div class="port-row right">
          <button
            type="button"
            class="port-label port-btn"
            class:ready
            title={`Click to peek output "${portName}" (${spec.type})`}
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
            style="background: {color}; border: 2px solid var(--bg-surface); width: 13px; height: 13px; right: -7px;"
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
    {:else}
      <div class="status-line muted">
        {manifest?.approxWeightsMB ? `~${manifest.approxWeightsMB} MB` : 'ready'}
      </div>
    {/if}
  </div>
</div>

<style>
  .p-node {
    position: relative;
    width: 224px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-body);
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    box-shadow: var(--node-shadow);
    overflow: visible;
  }
  .p-node.selected {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3), var(--node-shadow);
  }
  .p-node.running {
    border-color: #f59e0b;
  }
  .p-node.done {
    border-color: #22c55e;
  }
  .p-node.err {
    border-color: #ef4444;
  }
  .stripe {
    height: 3px;
    background: var(--stage-color);
    border-radius: 7px 7px 0 0;
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
    background: var(--err-bg);
    color: var(--err-text);
    border: 1px solid var(--err-border);
    border-radius: 3px;
    width: 16px;
    height: 16px;
    line-height: 13px;
    padding: 0;
    font-size: 12px;
    cursor: pointer;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.12s ease;
  }
  .p-node:hover .del-btn,
  .p-node.selected .del-btn,
  .del-btn:focus-visible {
    opacity: 1;
    pointer-events: auto;
  }
  .del-btn:hover {
    background: #dc2626;
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
    padding-left: 15px;
  }
  .port-row.right {
    padding-right: 15px;
    justify-content: flex-end;
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
