<script lang="ts">
  import { untrack } from 'svelte';
  import {
    BaseEdge,
    EdgeLabel,
    getSmoothStepPath,
    useNodes,
    type EdgeProps,
  } from '@xyflow/svelte';
  import { computeSmartEdgePath, type ObstacleRect } from './edge-routing';

  let {
    id,
    source,
    target,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    markerStart,
    markerEnd,
    selected,
    style,
    interactionWidth,
    label,
    labelStyle,
    data,
  }: EdgeProps = $props();

  const nodes = useNodes();

  let isProposed = $derived(Boolean(data?.proposed));
  let proposedCount = $derived(typeof data?.proposedCount === 'number' ? data.proposedCount : 1);
  let targetPortLabel = $derived(typeof data?.toPort === 'string' ? data.toPort : '');
  let laneStagger = $derived(typeof data?.laneStagger === 'number' ? data.laneStagger : 0);
  let onAccept = $derived(
    typeof data?.onAccept === 'function' ? (data.onAccept as () => void) : undefined,
  );
  let onDismiss = $derived(
    typeof data?.onDismiss === 'function' ? (data.onDismiss as () => void) : undefined,
  );
  let onDelete = $derived(
    typeof data?.onDelete === 'function' ? (data.onDelete as () => void) : undefined,
  );

  // Memoize obstacle rects strictly by node geometry so progress/data ticks never re-route edges
  let geomKey = $derived(
    nodes.current
      .map(
        (n) =>
          `${n.id}:${Math.round(n.position.x)},${Math.round(n.position.y)},${Math.round(n.measured?.width ?? n.width ?? 230)},${Math.round(n.measured?.height ?? n.height ?? 160)}`,
      )
      .join('|'),
  );

  let obstacleRects = $derived.by((): ObstacleRect[] => {
    void geomKey;
    return untrack(() =>
      nodes.current.map((n) => ({
        id: n.id,
        x: n.position.x,
        y: n.position.y,
        width: n.measured?.width ?? n.width ?? 230,
        height: n.measured?.height ?? n.height ?? 160,
      })),
    );
  });

  let routed = $derived.by(() => {
    const smart = computeSmartEdgePath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourceId: source,
      targetId: target,
      nodes: obstacleRects,
      stub: 24,
      padding: 18,
      borderRadius: 12,
      laneStagger,
    });

    if (smart) {
      return { path: smart.path, labelX: smart.labelX, labelY: smart.labelY };
    }

    const [path, labelX, labelY] = getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      borderRadius: 14,
      offset: 26,
    });
    return { path, labelX, labelY };
  });
</script>

<BaseEdge
  {id}
  path={routed.path}
  labelX={routed.labelX}
  labelY={routed.labelY}
  label={isProposed ? undefined : label}
  {labelStyle}
  {markerStart}
  {markerEnd}
  interactionWidth={interactionWidth ?? 22}
  class={isProposed ? 'ghost-wire-path' : undefined}
  {style}
/>

{#if selected && !isProposed && onDelete}
  <EdgeLabel x={routed.labelX} y={routed.labelY} transparent selectEdgeOnClick={false}>
    <button
      type="button"
      class="edge-delete-btn"
      title="Delete wire (Backspace / Delete)"
      aria-label="Delete wire"
      onpointerdown={(e) => e.stopPropagation()}
      onmousedown={(e) => e.stopPropagation()}
      onclick={(e) => {
        e.stopPropagation();
        onDelete?.();
      }}
    >
      ×
    </button>
  </EdgeLabel>
{/if}

{#if isProposed && onAccept && onDismiss}
  <EdgeLabel x={routed.labelX} y={routed.labelY} transparent selectEdgeOnClick={false}>
    <div class="ghost-pill" role="group" aria-label="Proposed wire action">
      <button
        type="button"
        class="ghost-accept"
        title="Accept proposed connection (Tab or Enter)"
        onclick={(e) => {
          e.stopPropagation();
          onAccept?.();
        }}
      >
        <span class="chk">✓</span>
        {#if proposedCount === 1}
          Connect <kbd>Tab</kbd>
        {:else}
          +{targetPortLabel || 'wire'}
        {/if}
      </button>
      <button
        type="button"
        class="ghost-dismiss"
        title="Dismiss proposed connection (Esc)"
        aria-label="Dismiss proposed connection"
        onclick={(e) => {
          e.stopPropagation();
          onDismiss?.();
        }}
      >
        ×
      </button>
    </div>
  </EdgeLabel>
{/if}

<style>
  :global(.svelte-flow__edge-path.ghost-wire-path) {
    stroke-dasharray: 6 5;
    animation: ghost-dash 0.75s linear infinite;
  }
  :global(.svelte-flow__edge:hover .svelte-flow__edge-path.ghost-wire-path) {
    opacity: 1 !important;
    stroke-width: 2.8px !important;
  }
  @keyframes ghost-dash {
    to {
      stroke-dashoffset: -11;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    :global(.svelte-flow__edge-path.ghost-wire-path) {
      animation: none;
    }
  }

  :global(.svelte-flow__edgelabel-renderer:has(.ghost-pill)),
  :global(.svelte-flow__edgelabel-renderer:has(.edge-delete-btn)) {
    z-index: 1005 !important;
  }

  .edge-delete-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    padding: 0;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-size: 14px;
    font-weight: 600;
    line-height: 1;
    cursor: pointer;
    pointer-events: auto;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.22);
    transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease, transform 0.12s ease;
  }
  .edge-delete-btn:hover {
    background: var(--err-strong, #ef4444);
    border-color: var(--err-strong, #ef4444);
    color: #fff;
    transform: scale(1.1);
  }

  .ghost-pill {
    display: inline-flex;
    align-items: center;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.22);
    overflow: hidden;
    pointer-events: auto;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
  }
  .ghost-accept {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border: none;
    background: transparent;
    color: var(--text-primary);
    font-size: 10.5px;
    font-weight: 600;
    padding: 3px 7px 3px 8px;
    cursor: pointer;
    white-space: nowrap;
  }
  .ghost-accept:hover {
    background: var(--accent);
    color: #fff;
  }
  .ghost-accept .chk {
    color: #10b981;
    font-weight: 700;
  }
  .ghost-accept:hover .chk {
    color: #fff;
  }
  .ghost-accept kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    padding: 0 4px;
    border-radius: 3px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
    color: var(--text-muted);
  }
  .ghost-accept:hover kbd {
    border-color: rgba(255, 255, 255, 0.35);
    background: rgba(255, 255, 255, 0.16);
    color: #fff;
  }
  .ghost-dismiss {
    border: none;
    border-left: 1px solid var(--border-subtle);
    background: transparent;
    color: var(--text-muted);
    font-size: 12px;
    line-height: 1;
    padding: 3px 7px;
    cursor: pointer;
  }
  .ghost-dismiss:hover {
    background: var(--err-bg);
    color: var(--err-strong);
  }
</style>
