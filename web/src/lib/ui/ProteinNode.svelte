<script lang="ts">
  import { Handle, Position } from '@xyflow/svelte';
  import {
    ADAPTER_REGISTRY,
    LOCALFOLD_MODEL_OPTIONS,
    RFD4_BENCHMARK_OPTIONS,
  } from '$lib/engine/registry';
  import { PORT_COLORS, STAGE_COLORS, STAGE_LABELS } from '$lib/engine/graph-utils';
  import { selectPrimaryOutputPort } from '$lib/engine/peek';
  import type { NodeFlowInfo } from '$lib/engine/batch-flow';
  import type { CanonicalArtifact, NodeProgressEvent } from '$lib/engine/types';
  import {
    formatMutationMatrixBadge,
    formatNodeParamSummary,
    type UpstreamLockSummary,
  } from './node-display-helpers';
  import ModelCardPopover from './ModelCardPopover.svelte';

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
    activePeekSlotId?: string;
    autoModeHint?: string;
    upstreamLockSummary?: UpstreamLockSummary | null;
    flowInfo?: NodeFlowInfo | null;
    isDraft?: boolean;
    onDelete?: (nodeId: string) => void;
    onPeekPort?: (nodeId: string, direction: 'in' | 'out', portName: string) => void;
    onOpenStudio?: (nodeId: string) => void;
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
    if (data.adapterId === 'rfd4-proteina') {
      const mode = String(data.params?.mode ?? 'unconditional');
      const needsEnc =
        mode === 'partial' ||
        mode === 'inverse-folding' ||
        (mode === 'preset' &&
          Boolean(
            RFD4_BENCHMARK_OPTIONS.find((o) => o.id === String(data.params?.presetId ?? ''))
              ?.needsEncoder,
          ));
      return (manifest?.approxWeightsMB ?? 330) + (needsEnc ? 138 : 0);
    }
    const precKey = String(data.params?.precision ?? '').trim().toLowerCase();
    if (precKey && manifest?.precisionWeightsMB && precKey in manifest.precisionWeightsMB) {
      return manifest.precisionWeightsMB[precKey];
    }
    return manifest?.approxWeightsMB ?? 0;
  });
  let st = $derived(data.state);
  let isPipelineRunning = $derived(Boolean(data.isPipelineRunning));
  let isNodeRunning = $derived(st?.status === 'running' || st?.status === 'loading');
  let isNodeDone = $derived(st?.status === 'done' || st?.status === 'cached');
  let isNodeQueued = $derived(isPipelineRunning && (!st || st.status === 'idle'));

  let paramSummary = $derived(
    formatNodeParamSummary(data.adapterId, data.params, {
      autoModeHint: data.autoModeHint,
    }),
  );

  let primaryOutput = $derived(selectPrimaryOutputPort(data.outputs));
  let primaryOutputInfo = $derived(
    primaryOutput ? { portName: primaryOutput.portName, is3D: primaryOutput.is3D } : null,
  );

  let outputBadge = $derived.by((): string | null => {
    const outs = data.outputs;
    if (!outs) return null;
    const ordered = primaryOutput
      ? [primaryOutput.artifact, ...Object.values(outs).filter((a) => a !== primaryOutput.artifact)]
      : Object.values(outs);
    for (const art of ordered) {
      if (art.kind === 'DesignabilityMetrics' && art.best) {
        return `scRMSD ${art.best.scRmsd.toFixed(2)} Å · pLDDT ${art.best.meanPlddt.toFixed(1)}`;
      }
      if (art.kind === 'MutationMatrix') {
        return formatMutationMatrixBadge(art);
      }
      if (art.kind === 'ResidueTrack') {
        const nPos = art.residueIds?.length ?? art.residues.length;
        const mean = art.summary?.mean;
        return Number.isFinite(mean)
          ? `${nPos} pos · μ=${(mean as number).toFixed(2)}`
          : `${nPos} pos · ${art.quantity}`;
      }
      if (art.kind === 'FoldedStructure' && art.folds.length > 0) {
        const bestP = Math.max(...art.folds.map((f) => f.meanPlddt));
        const skipped = art.excludedCandidates?.length ?? 0;
        const countTag =
          skipped > 0
            ? `${art.folds.length}/${art.folds.length + skipped} folds`
            : `${art.folds.length} fold${art.folds.length > 1 ? 's' : ''}`;
        return `${countTag} · pLDDT ${bestP.toFixed(1)}`;
      }
      if (art.kind === 'SequenceSet') {
        const skipped = art.excludedCandidates?.length ?? 0;
        const countTag =
          skipped > 0
            ? `${art.candidates.length}/${art.candidates.length + skipped} seqs`
            : `${art.candidates.length} seq${art.candidates.length > 1 ? 's' : ''}`;
        return `${countTag} (${art.candidates[0]?.sequence.length ?? 0} aa)`;
      }
      if (art.kind === 'SequenceEmbedding') {
        return `${art.sequence.length} aa · ${art.dim}d`;
      }
      if (art.kind === 'StructureArtifact') {
        if (art.dockingMetrics?.rows?.[0]) {
          const r = art.dockingMetrics.rows[0];
          const posePrefix =
            art.dockingMetrics.taskMode === 'dock' && art.dockingMetrics.rows.length > 1
              ? `${art.dockingMetrics.rows.length} poses · `
              : '';
          return `${posePrefix}CNN ${r.cnnScore.toFixed(2)} · pK ${r.cnnAffinity.toFixed(2)} · Vina ${r.vinaScore.toFixed(1)}`;
        }
        const ligNames = art.diagnostics?.ligandNames ?? [];
        const naChains = art.diagnostics?.nucleicAcidChains ?? [];
        if (art.length === 0) {
          const parts: string[] = [];
          if (ligNames.length > 0) parts.push(`Ligand: ${ligNames.join(', ')}`);
          if (naChains.length > 0) parts.push(`NA (${naChains.join(', ')})`);
          return parts.length > 0 ? parts.join(' + ') : art.label;
        }
        const extras: string[] = [];
        if (ligNames.length > 0) extras.push(ligNames.join(','));
        if (naChains.length > 0) extras.push(`NA:${naChains.join(',')}`);
        const extraTag = extras.length > 0 ? ` + ${extras.join(' + ')}` : '';
        return `L=${art.length} aa${extraTag} · Rg ${art.geometry.rg.toFixed(1)} Å`;
      }
      if (art.kind === 'ConditionSpec') {
        return art.summary;
      }
    }
    return null;
  });
  let displayTitle = $derived.by((): string => {
    if (data.label) return data.label;
    if (data.adapterId === 'localfold-af2') {
      const modelId = String(data.params?.model ?? 'monomer');
      const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
      return meta.shortLabel;
    }
    return manifest?.shortName || data.adapterId;
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
  class:draft={Boolean(data.isDraft)}
  style="--stage-color: {stageColor}{data.compareColor ? `; --compare-color: ${data.compareColor}` : ''}"
>
  <div class="stripe-clip" aria-hidden="true"><div class="stripe"></div></div>

  {#if data.isDraft}
    <span class="draft-badge" title="Draft preview — press Enter or Apply to add to canvas">
      ✦ NEW
    </span>
  {/if}

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
    {#if data.onOpenStudio}
      <button
        type="button"
        class="studio-btn"
        title="Inspect in Studio (Space)"
        aria-label="Inspect in Studio"
        onclick={(e) => {
          e.stopPropagation();
          data.onOpenStudio?.(id);
        }}
      >
        <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <rect x="1.8" y="2.4" width="12.4" height="11.2" rx="2" stroke="currentColor" stroke-width="1.4" />
          <path d="M6.2 2.6V13.4" stroke="currentColor" stroke-width="1.3" />
          <path d="M8.6 6H11.8M8.6 8.5H11.8M8.6 11H10.6" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" />
        </svg>
      </button>
    {/if}
  </div>

  <div class="title">{displayTitle}</div>

  {#if paramSummary}
    <div class="param-chip" title={paramSummary}>{paramSummary}</div>
  {/if}
  {#if data.upstreamLockSummary}
    <div
      class="lock-chip"
      class:is-warn={data.upstreamLockSummary.status === 'warning'}
      class:is-unlocked={data.upstreamLockSummary.status === 'unlocked'}
      title={data.upstreamLockSummary.detailText}
    >
      {data.upstreamLockSummary.nodeBadge}
    </div>
  {/if}
  {#if data.flowInfo && data.flowInfo.droppedCount > 0 && data.flowInfo.badgeText}
    <div
      class="lock-chip"
      class:is-warn={data.flowInfo.severity === 'warning'}
      title={data.flowInfo.detailText ?? data.flowInfo.badgeText}
      data-flow-drop={data.flowInfo.dropKind}
    >
      {data.flowInfo.badgeText}
    </div>
  {/if}

  <!-- Typed Input & Output Port Rows -->
  <div class="ports">
    <div class="col in-col">
      {#each inputs as [portName, spec] (portName)}
        {@const color = PORT_COLORS[spec.type]?.bg ?? '#94a3b8'}
        {@const portLabel = PORT_COLORS[spec.type]?.label ?? spec.type}
        {@const connected = Boolean(data.inputPortsConnected?.[portName])}
        {@const ready = Boolean(data.inputPortsReady?.[portName])}
        {@const peeking = data.activePeekSlotId === `in:${portName}`}
        <div class="port-row left">
          <Handle
            type="target"
            position={Position.Left}
            id={portName}
            style={connected
              ? 'background: transparent; border: none; width: 8px; height: 12px; left: -1px;'
              : `background: ${color}; border: 1.5px solid ${peeking ? 'var(--text-primary)' : 'var(--bg-surface)'}; width: ${peeking ? 9 : 8}px; height: ${peeking ? 9 : 8}px; left: -4px;`}
          />
          <button
            type="button"
            class="port-label port-btn"
            class:ready
            class:peeking
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
        {@const peeking = data.activePeekSlotId === `out:${portName}`}
        <div class="port-row right">
          <button
            type="button"
            class="port-label port-btn"
            class:ready
            class:peeking
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
            style={`background: ${color}; border: 1.5px solid ${peeking ? 'var(--text-primary)' : 'var(--bg-surface)'}; width: ${peeking ? 9 : 8}px; height: ${peeking ? 9 : 8}px; right: -4px;`}
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
      <div class="footer-done-row">
        <ModelCardPopover adapterId={data.adapterId} params={data.params} iconOnly />
        <button
          type="button"
          class="status-line ok-text probe-footer-btn"
          title={primaryOutputInfo?.is3D
            ? 'Click to probe & view this 3D structure in the viewport'
            : 'Click to inspect this block’s output'}
          onclick={(e) => {
            e.stopPropagation();
            if (primaryOutputInfo) {
              data.onPeekPort?.(id, 'out', primaryOutputInfo.portName);
            }
          }}
        >
          <span class="probe-badge-text">
            {#if st?.status === 'cached'}<span class="cache-tag">CACHED</span>{/if}
            {outputBadge}
          </span>
          <span class="probe-pill" aria-hidden="true">
            <svg width="11" height="11" viewBox="0 0 16 16" fill="none">
              <path d="M1.5 8C3 4.8 5.3 3.2 8 3.2C10.7 3.2 13 4.8 14.5 8C13 11.2 10.7 12.8 8 12.8C5.3 12.8 3 11.2 1.5 8Z" stroke="currentColor" stroke-width="1.5"/>
              <circle cx="8" cy="8" r="2.2" fill="currentColor"/>
            </svg>
            {primaryOutputInfo?.is3D ? '3D' : 'Peek'}
          </span>
        </button>
      </div>
    {:else if isNodeQueued}
      <div class="footer-idle-row">
        <span class="status-line queued-text">queued…</span>
        <ModelCardPopover adapterId={data.adapterId} params={data.params} iconOnly />
      </div>
    {:else if st?.status === 'idle' && st?.detail === 'Cancelled'}
      <div class="footer-idle-row">
        <span class="status-line muted">Cancelled</span>
        <ModelCardPopover adapterId={data.adapterId} params={data.params} iconOnly />
      </div>
    {:else}
      <div class="footer-idle-row">
        <span class="status-line muted">
          {activeWeightsMB > 0 ? `~${activeWeightsMB} MB` : 'ready'}
        </span>
        <ModelCardPopover adapterId={data.adapterId} params={data.params} iconOnly />
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
      0 0 0 3px rgba(245, 158, 11, 0.2),
      0 6px 18px rgba(245, 158, 11, 0.14);
  }
  .p-node.running::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
    z-index: 0;
    box-shadow:
      0 0 0 4.5px rgba(245, 158, 11, 0.34),
      0 10px 26px rgba(245, 158, 11, 0.24);
    will-change: opacity;
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
    --node-border-w: 2.5px;
    --node-border-color: var(--compare-color);
    box-shadow:
      0 0 0 4px color-mix(in srgb, var(--compare-color) 36%, transparent),
      0 10px 22px color-mix(in srgb, var(--compare-color) 20%, transparent),
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
  .p-node.draft {
    opacity: 0.92;
    --node-border-w: 1.5px;
    --node-border-style: dashed;
    --node-border-color: var(--accent);
    box-shadow:
      0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent),
      var(--node-shadow);
  }
  .draft-badge {
    position: absolute;
    top: -9px;
    right: 10px;
    z-index: 5;
    padding: 1px 6px;
    border-radius: 999px;
    background: var(--accent);
    color: #fff;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 8.5px;
    font-weight: 700;
    letter-spacing: 0.04em;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.22);
    pointer-events: none;
  }
  /* Selected / peeking focus ring matches the node's stage color (top stripe) */
  .p-node.selected {
    opacity: 1;
    filter: none;
    --node-border-w: 2.5px;
    --node-border-color: var(--stage-color);
    box-shadow:
      0 0 0 4px color-mix(in srgb, var(--stage-color) 28%, transparent),
      0 10px 24px color-mix(in srgb, var(--stage-color) 20%, transparent),
      var(--node-shadow);
  }
  .p-node.selected.running {
    --node-border-w: 2.5px;
    --node-border-color: #f59e0b;
    box-shadow:
      0 0 0 3px var(--stage-color),
      0 0 0 6px rgba(245, 158, 11, 0.28),
      var(--node-shadow);
  }
  .p-node.selected.done {
    --node-border-w: 2.5px;
    --node-border-color: var(--stage-color);
    box-shadow:
      0 0 0 4px color-mix(in srgb, var(--stage-color) 28%, transparent),
      0 10px 24px color-mix(in srgb, var(--stage-color) 20%, transparent),
      var(--node-shadow);
  }
  .p-node.selected.comparing {
    --node-border-w: 2.5px;
    --node-border-color: var(--compare-color);
    box-shadow:
      0 0 0 4px color-mix(in srgb, var(--compare-color) 40%, transparent),
      0 10px 24px color-mix(in srgb, var(--compare-color) 24%, transparent),
      var(--node-shadow);
  }
  .p-node.selected.err {
    --node-border-w: 2.5px;
    --node-border-color: #ef4444;
    box-shadow:
      0 0 0 3px var(--accent),
      0 0 0 6px rgba(239, 68, 68, 0.26),
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
      opacity: 0.35;
    }
    50% {
      opacity: 1;
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
    .p-node.running::before {
      animation: none;
      opacity: 0.65;
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
  .studio-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    padding: 0;
    border-radius: 4px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-dim);
    cursor: pointer;
    flex-shrink: 0;
    transition:
      background-color 120ms ease,
      border-color 120ms ease,
      color 120ms ease;
  }
  .p-node:hover .studio-btn,
  .p-node.selected .studio-btn {
    color: var(--text-secondary);
  }
  .studio-btn:hover,
  .studio-btn:focus-visible {
    background: var(--bg-pill);
    border-color: var(--accent);
    color: var(--accent-bright);
    outline: none;
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
  .lock-chip {
    margin: -2px 10px 6px;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--bg-pill);
    border: 1px solid var(--border-strong);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    color: var(--accent-soft);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .lock-chip.is-warn {
    background: var(--warn-bg);
    border-color: var(--warn-border);
    color: var(--warn-text);
  }
  .lock-chip.is-unlocked {
    background: var(--bg-surface-2);
    border-color: var(--border-subtle);
    color: var(--text-dim);
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
  .port-btn.peeking {
    color: var(--text-primary);
    font-weight: 700;
    background: var(--bg-surface-2);
    border: 1.5px solid var(--text-primary);
    border-radius: 999px;
    padding: 0 6px;
    line-height: 16px;
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
  .footer-idle-row {
    display: flex;
    align-items: center;
    gap: 4px;
    min-width: 0;
  }
  .footer-done-row {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
  }
  .footer-done-row .probe-footer-btn {
    flex: 1;
    min-width: 0;
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
  .probe-footer-btn {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    width: 100%;
    padding: 1px 0;
    margin: 0;
    background: transparent;
    border: none;
    cursor: pointer;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    line-height: 1.3;
    text-align: left;
  }
  .probe-badge-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .probe-pill {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    flex-shrink: 0;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 9px;
    font-weight: 600;
    color: var(--accent-bright);
    background: var(--bg-pill);
    border: 1px solid var(--border);
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
  }
  .probe-footer-btn:hover .probe-pill {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
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
