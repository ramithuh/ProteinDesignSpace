<script lang="ts">
  import { Handle, Position } from '@xyflow/svelte';
  import { ADAPTER_REGISTRY } from '$lib/engine/registry';
  import { PORT_COLORS, STAGE_COLORS } from '$lib/engine/graph-utils';
  import type { CanonicalArtifact, NodeProgressEvent } from '$lib/engine/types';

  export interface ProteinNodeData {
    adapterId: string;
    label: string;
    state?: NodeProgressEvent;
    outputs?: Record<string, CanonicalArtifact>;
    onDelete?: (nodeId: string) => void;
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
    {#if selected && data.onDelete}
      <button
        class="del-btn"
        title="Remove block"
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

  <!-- Typed Input & Output Port Rows -->
  <div class="ports">
    <div class="col in-col">
      {#each inputs as [portName, spec] (portName)}
        {@const color = PORT_COLORS[spec.type]?.bg ?? '#94a3b8'}
        <div class="port-row left">
          <Handle
            type="target"
            position={Position.Left}
            id={portName}
            style="background: {color}; border: 2px solid var(--bg-surface); width: 11px; height: 11px; left: -6px;"
          />
          <span class="port-label" title={`${portName}: ${spec.type}${spec.required === false ? ' (optional)' : ''}`}>
            {portName}{#if spec.required === false}<span class="opt">?</span>{/if}
          </span>
        </div>
      {/each}
    </div>

    <div class="col out-col">
      {#each outputs as [portName, spec] (portName)}
        {@const color = PORT_COLORS[spec.type]?.bg ?? '#94a3b8'}
        <div class="port-row right">
          <span class="port-label" title={`${portName}: ${spec.type}`}>{portName}</span>
          <Handle
            type="source"
            position={Position.Right}
            id={portName}
            style="background: {color}; border: 2px solid var(--bg-surface); width: 11px; height: 11px; right: -6px;"
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
  }
  .del-btn:hover {
    background: #dc2626;
    color: #fff;
  }
  .title {
    padding: 0 10px 6px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
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
    height: 18px;
  }
  .port-row.left {
    padding-left: 12px;
  }
  .port-row.right {
    padding-right: 12px;
    justify-content: flex-end;
  }
  .port-label {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-secondary);
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
