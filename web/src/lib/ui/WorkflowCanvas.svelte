<script lang="ts">
  import { tick } from 'svelte';
  import {
    Background,
    Controls,
    SvelteFlow,
    useSvelteFlow,
    type Connection,
    type Edge,
    type IsValidConnection,
    type Node,
    type NodeTypes,
  } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';
  import { ADAPTER_REGISTRY } from '$lib/engine/registry';
  import { makeEdgeId } from '$lib/engine/peek';
  import {
    PORT_COLORS,
    autoLayoutGraph,
    insertBlockSmart,
    validateConnection,
  } from '$lib/engine/graph-utils';
  import type {
    CanonicalArtifact,
    NodeProgressEvent,
    WorkflowEdge,
    WorkflowGraph,
  } from '$lib/engine/types';
  import ProteinNode, { type ProteinNodeData } from './ProteinNode.svelte';
  import BlockCatalogPopover from './BlockCatalogPopover.svelte';

  interface Props {
    graph: WorkflowGraph;
    selectedNodeId: string;
    selectedEdgeId?: string | null;
    nodeStates: Record<string, NodeProgressEvent>;
    outputsByNode: Record<string, Record<string, CanonicalArtifact>>;
    theme?: 'dark' | 'light';
    disabled?: boolean;
    fillHeight?: boolean;
    onGraphChange: (next: WorkflowGraph) => void;
    onSelectNode: (nodeId: string) => void;
    onBlockAdded?: (nodeId: string) => void;
    onSelectEdge?: (edge: WorkflowEdge) => void;
    onPeekPort?: (nodeId: string, direction: 'in' | 'out', portName: string) => void;
  }

  let {
    graph,
    selectedNodeId,
    selectedEdgeId = null,
    nodeStates,
    outputsByNode,
    theme = 'dark',
    disabled = false,
    fillHeight = false,
    onGraphChange,
    onSelectNode,
    onBlockAdded = undefined,
    onSelectEdge = undefined,
    onPeekPort = undefined,
  }: Props = $props();

  const nodeTypes: NodeTypes = {
    protein: ProteinNode as unknown as NodeTypes[string],
  };

  let validationToast = $state<string | null>(null);
  let toastTimer: ReturnType<typeof setTimeout> | null = null;
  let canvasEl: HTMLDivElement | null = $state(null);
  let addBtnEl: HTMLButtonElement | null = $state(null);
  let wireBtnEl: HTMLButtonElement | null = $state(null);
  let wirePopoverEl: HTMLDivElement | null = $state(null);
  let catalogOpen = $state(false);
  let wireLegendOpen = $state(false);

  let flowApi: ReturnType<typeof useSvelteFlow> | null = null;
  try {
    flowApi = useSvelteFlow();
  } catch {
    flowApi = null;
  }

  const FIT_VIEW_OPTS = { padding: 0.2, maxZoom: 1, duration: 220 } as const;

  let fitTimer: ReturnType<typeof setTimeout> | null = null;
  let fitRaf1: number | null = null;
  let fitRaf2: number | null = null;

  function cancelPendingFit() {
    if (fitTimer) {
      clearTimeout(fitTimer);
      fitTimer = null;
    }
    if (fitRaf1 !== null) {
      cancelAnimationFrame(fitRaf1);
      fitRaf1 = null;
    }
    if (fitRaf2 !== null) {
      cancelAnimationFrame(fitRaf2);
      fitRaf2 = null;
    }
  }

  function scheduleFitView(duration = 220) {
    cancelPendingFit();
    void tick().then(() => {
      fitRaf1 = requestAnimationFrame(() => {
        fitRaf1 = null;
        fitRaf2 = requestAnimationFrame(() => {
          fitRaf2 = null;
          void flowApi?.fitView({ ...FIT_VIEW_OPTS, duration });
          // Backup pass once SvelteFlow's ResizeObserver finishes measuring new node boxes
          fitTimer = setTimeout(() => {
            fitTimer = null;
            void flowApi?.fitView({ ...FIT_VIEW_OPTS, duration });
          }, 80);
        });
      });
    });
  }

  // Cleanup timers/rAFs on destroy
  $effect(() => {
    return () => {
      if (toastTimer) clearTimeout(toastTimer);
      cancelPendingFit();
    };
  });

  // Fit the entire workflow whenever switching presets/graphs or toggling split layout
  $effect(() => {
    void graph.id;
    void fillHeight;
    scheduleFitView(220);
  });

  // Close wire legend on click-outside
  $effect(() => {
    if (!wireLegendOpen) return;
    function handlePointerDown(ev: PointerEvent) {
      const target = ev.target;
      if (!(target instanceof globalThis.Node)) return;
      if (wirePopoverEl?.contains(target) || wireBtnEl?.contains(target)) return;
      wireLegendOpen = false;
    }
    window.addEventListener('pointerdown', handlePointerDown);
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
    };
  });

  function isEditableElement(el: EventTarget | null): boolean {
    if (!(el instanceof HTMLElement)) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
  }

  function handleWindowKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      if (wireLegendOpen) {
        wireLegendOpen = false;
      }
      if (catalogOpen) {
        catalogOpen = false;
      }
      return;
    }
    if (disabled || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (isEditableElement(ev.target) || isEditableElement(document.activeElement)) return;

    if (ev.key === '/' || (ev.shiftKey && (ev.key === 'A' || ev.key === 'a'))) {
      ev.preventDefault();
      wireLegendOpen = false;
      catalogOpen = true;
    }
  }

  function flashToast(msg: string) {
    validationToast = msg;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      validationToast = null;
    }, 3000);
  }

  function edgeColor(fromNodeId: string, fromPort: string): string {
    const n = graph.nodes.find((x) => x.id === fromNodeId);
    const portType = n ? ADAPTER_REGISTRY[n.adapterId]?.outputs[fromPort]?.type : undefined;
    return portType ? PORT_COLORS[portType]?.bg ?? '#64748b' : '#64748b';
  }

  // Convert canonical WorkflowGraph -> SvelteFlow Node[] / Edge[]
  let flowNodes = $state.raw<Node<ProteinNodeData>[]>([]);
  let flowEdges = $state.raw<Edge[]>([]);

  $effect(() => {
    const laid = graph.nodes.some((n) => !n.position) ? autoLayoutGraph(graph) : graph;
    flowNodes = laid.nodes.map((n) => {
      const inputPortsReady: Record<string, boolean> = {};
      for (const e of laid.edges) {
        if (e.toNode === n.id && outputsByNode[e.fromNode]?.[e.fromPort]) {
          inputPortsReady[e.toPort] = true;
        }
      }
      const outputPortsReady: Record<string, boolean> = {};
      for (const port of Object.keys(outputsByNode[n.id] ?? {})) {
        outputPortsReady[port] = true;
      }
      return {
        id: n.id,
        type: 'protein',
        position: n.position ?? { x: 40, y: 80 },
        selected: n.id === selectedNodeId,
        data: {
          adapterId: n.adapterId,
          label: n.label ?? ADAPTER_REGISTRY[n.adapterId]?.shortName ?? n.adapterId,
          params: n.params,
          state: nodeStates[n.id],
          outputs: outputsByNode[n.id],
          inputPortsReady,
          outputPortsReady,
          onDelete: disabled ? undefined : removeNode,
          onPeekPort,
        },
      };
    });

    flowEdges = laid.edges.map((e) => {
      const eid = makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort);
      const col = edgeColor(e.fromNode, e.fromPort);
      const srcState = nodeStates[e.fromNode]?.status;
      const dstState = nodeStates[e.toNode]?.status;
      const active =
        srcState === 'running' ||
        srcState === 'loading' ||
        dstState === 'running' ||
        dstState === 'loading';
      const isSel = eid === selectedEdgeId;
      return {
        id: eid,
        type: 'smoothstep',
        source: e.fromNode,
        sourceHandle: e.fromPort,
        target: e.toNode,
        targetHandle: e.toPort,
        selected: isSel,
        animated: active,
        style: isSel
          ? `stroke: ${col}; stroke-width: 3.5px; filter: drop-shadow(0 0 4px ${col}); cursor: pointer;`
          : `stroke: ${col}; stroke-width: 2px; cursor: pointer;`,
        pathOptions: { borderRadius: 14, offset: 26 },
      };
    });
  });

  const isValidConnection: IsValidConnection = (conn) => {
    if (disabled) return false;
    const src = conn.source;
    const dst = conn.target;
    const srcHandle = conn.sourceHandle ?? '';
    const dstHandle = conn.targetHandle ?? '';
    if (!src || !dst || !srcHandle || !dstHandle) return false;
    return validateConnection(graph, src, srcHandle, dst, dstHandle).valid;
  };

  function handleConnect(conn: Connection) {
    if (disabled) return;
    const src = conn.source;
    const dst = conn.target;
    const fromPort = conn.sourceHandle ?? '';
    const toPort = conn.targetHandle ?? '';
    const check = validateConnection(graph, src, fromPort, dst, toPort);
    if (!check.valid) {
      flashToast(check.reason ?? 'Invalid connection');
      return;
    }
    // Replace any existing wire occupying the same target input port (ComfyUI behavior)
    const filtered = graph.edges.filter((e) => !(e.toNode === dst && e.toPort === toPort));
    const newEdge: WorkflowEdge = { fromNode: src, fromPort, toNode: dst, toPort };
    onGraphChange({ ...graph, edges: [...filtered, newEdge] });
  }

  function handleNodeDragStop({ targetNode, nodes: draggedNodes }: { targetNode: Node | null; nodes: Node[] }) {
    if (disabled) return;
    const moved = new Map<string, { x: number; y: number }>();
    for (const dn of draggedNodes ?? []) {
      moved.set(dn.id, { x: Math.round(dn.position.x), y: Math.round(dn.position.y) });
    }
    if (targetNode) {
      moved.set(targetNode.id, { x: Math.round(targetNode.position.x), y: Math.round(targetNode.position.y) });
    }
    if (moved.size === 0) return;
    onGraphChange({
      ...graph,
      nodes: graph.nodes.map((n) => {
        const pos = moved.get(n.id);
        return pos ? { ...n, position: pos } : n;
      }),
    });
  }

  function handleDelete({ nodes: delNodes, edges: delEdges }: { nodes: Node[]; edges: Edge[] }) {
    if (disabled) return;
    const rmNodeIds = new Set(delNodes.map((n) => n.id));
    const rmEdgeIds = new Set(delEdges.map((e) => e.id));
    const nextNodes = graph.nodes.filter((n) => !rmNodeIds.has(n.id));
    const nextEdges = graph.edges.filter((e) => {
      const eid = `${e.fromNode}:${e.fromPort}->${e.toNode}:${e.toPort}`;
      return !rmEdgeIds.has(eid) && !rmNodeIds.has(e.fromNode) && !rmNodeIds.has(e.toNode);
    });
    onGraphChange({ ...graph, nodes: nextNodes, edges: nextEdges });
    if (rmNodeIds.has(selectedNodeId)) {
      onSelectNode(nextNodes[0]?.id ?? '');
    }
  }

  function removeNode(nodeId: string) {
    if (disabled) return;
    const nextNodes = graph.nodes.filter((n) => n.id !== nodeId);
    const nextEdges = graph.edges.filter((e) => e.fromNode !== nodeId && e.toNode !== nodeId);
    onGraphChange({ ...graph, nodes: nextNodes, edges: nextEdges });
    if (selectedNodeId === nodeId) onSelectNode(nextNodes[0]?.id ?? '');
  }

  function addBlock(adapterId: string, dropPos?: { x: number; y: number }) {
    if (disabled) return;
    if (!ADAPTER_REGISTRY[adapterId]) return;
    const { graph: nextGraph, nodeId } = insertBlockSmart(
      graph,
      adapterId,
      selectedNodeId || undefined,
      dropPos,
    );
    catalogOpen = false;
    onGraphChange(nextGraph);
    if (onBlockAdded) {
      onBlockAdded(nodeId);
    } else {
      onSelectNode(nodeId);
    }
    scheduleFitView(220);
  }

  function handleDragStart(ev: DragEvent, adapterId: string) {
    ev.dataTransfer?.setData('application/denovo-adapter', adapterId);
    if (ev.dataTransfer) ev.dataTransfer.effectAllowed = 'copy';
  }

  function handleDrop(ev: DragEvent) {
    ev.preventDefault();
    if (disabled) return;
    const adapterId = ev.dataTransfer?.getData('application/denovo-adapter');
    if (!adapterId || !canvasEl) return;
    let dropPos: { x: number; y: number };
    if (flowApi) {
      const flowPos = flowApi.screenToFlowPosition({ x: ev.clientX, y: ev.clientY });
      dropPos = { x: Math.round(flowPos.x - 110), y: Math.round(flowPos.y - 40) };
    } else {
      const rect = canvasEl.getBoundingClientRect();
      dropPos = {
        x: Math.max(20, Math.round(ev.clientX - rect.left - 110)),
        y: Math.max(20, Math.round(ev.clientY - rect.top - 40)),
      };
    }
    addBlock(adapterId, dropPos);
  }

  function handlePaneDblClick(ev: MouseEvent) {
    if (disabled) return;
    const target = ev.target as HTMLElement | null;
    if (
      target?.closest(
        '.svelte-flow__node, .svelte-flow__edge, .canvas-overlay, .svelte-flow__controls, .catalog-popover, .wire-popover',
      )
    ) {
      return;
    }
    wireLegendOpen = false;
    catalogOpen = true;
  }

  async function handleAutoLayout() {
    if (disabled) return;
    onGraphChange(autoLayoutGraph(graph));
    scheduleFitView(250);
  }

  function handleClear() {
    if (disabled) return;
    catalogOpen = false;
    wireLegendOpen = false;
    onGraphChange({ ...graph, nodes: [], edges: [] });
    onSelectNode('');
  }

  const WIRE_LEGEND_ITEMS = [
    PORT_COLORS.StructureArtifact,
    PORT_COLORS.ConditionSpec,
    PORT_COLORS.SequenceSet,
    PORT_COLORS.FoldedStructure,
    PORT_COLORS.DesignabilityMetrics,
  ];
</script>

<svelte:window onkeydown={handleWindowKeydown} />

<div
  class="flow-wrap"
  class:fill-height={fillHeight}
  bind:this={canvasEl}
  role="application"
  aria-label="Protein design workflow canvas"
  ondragover={(e) => e.preventDefault()}
  ondrop={handleDrop}
  ondblclick={handlePaneDblClick}
>
  <!-- Floating Top-Left: + Add Block trigger & searchable catalog -->
  <div class="canvas-left-tools canvas-overlay">
    <button
      bind:this={addBtnEl}
      type="button"
      class="add-block-btn"
      class:open={catalogOpen}
      {disabled}
      onclick={() => {
        wireLegendOpen = false;
        catalogOpen = !catalogOpen;
      }}
      title="Add a workflow block (shortcut: / or Shift+A, or double-click canvas)"
    >
      <span class="plus">+</span> Add Block
      <kbd class="kbd-hint">/</kbd>
    </button>
  </div>

  {#if catalogOpen}
    <BlockCatalogPopover
      {disabled}
      triggerEl={addBtnEl}
      onInsert={(aid) => addBlock(aid)}
      onDragStart={handleDragStart}
      onClose={() => (catalogOpen = false)}
    />
  {/if}

  <!-- Floating Top-Right: Compact canvas controls -->
  <div class="canvas-tools canvas-overlay">
    <button type="button" class="c-btn" {disabled} onclick={handleAutoLayout} title="Re-arrange DAG cleanly and fit view">
      Auto Layout
    </button>
    <button
      bind:this={wireBtnEl}
      type="button"
      class="c-btn icon-q"
      class:active={wireLegendOpen}
      onclick={() => {
        catalogOpen = false;
        wireLegendOpen = !wireLegendOpen;
      }}
      title="Typed wire colors"
      aria-label="Typed wire colors"
    >
      ?
    </button>
    <button type="button" class="c-btn danger" {disabled} onclick={handleClear} title="Remove all blocks from canvas">
      Clear
    </button>
  </div>

  {#if wireLegendOpen}
    <div class="wire-popover canvas-overlay" bind:this={wirePopoverEl} role="dialog" aria-label="Typed wire colors">
      <div class="wire-pop-title">Typed Wire Colors</div>
      {#each WIRE_LEGEND_ITEMS as item (item.label)}
        <div class="wire-row">
          <span class="wire-dot" style="background: {item.bg}"></span>
          <span>{item.label}</span>
        </div>
      {/each}
      <div class="wire-pop-hint">Only matching port types connect. Click any wire or port label to peek data.</div>
    </div>
  {/if}

  {#if validationToast}
    <div class="toast">{validationToast}</div>
  {/if}

  <div class="flow-inner">
    <SvelteFlow
      bind:nodes={flowNodes}
      bind:edges={flowEdges}
      {nodeTypes}
      forceColorMode={theme}
      nodesDraggable={!disabled}
      nodesConnectable={!disabled}
      zoomOnDoubleClick={false}
      deleteKey={disabled ? null : ['Backspace', 'Delete']}
      fitView
      fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
      {isValidConnection}
      onconnect={handleConnect}
      onnodedragstop={handleNodeDragStop}
      ondelete={handleDelete}
      onpaneclick={() => {
        catalogOpen = false;
        wireLegendOpen = false;
      }}
      onnodeclick={({ node }) => {
        catalogOpen = false;
        wireLegendOpen = false;
        onSelectNode(node.id);
      }}
      onedgeclick={({ edge }) => {
        catalogOpen = false;
        wireLegendOpen = false;
        const found = graph.edges.find(
          (e) => makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort) === edge.id,
        );
        if (found) onSelectEdge?.(found);
      }}
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={20} size={1} />
      <Controls showLock={false} />
    </SvelteFlow>
  </div>
</div>

<style>
  .flow-wrap {
    position: relative;
    width: 100%;
    height: 340px;
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: visible;
  }
  .flow-wrap.fill-height {
    height: 100%;
    min-height: 440px;
    flex: 1;
  }
  .flow-inner {
    width: 100%;
    height: 100%;
    border-radius: 7px;
    overflow: hidden;
  }
  @media (max-width: 1024px) {
    .flow-wrap.fill-height {
      height: 340px;
      min-height: 300px;
    }
  }
  @media (max-width: 780px) {
    .flow-wrap,
    .flow-wrap.fill-height {
      height: 290px;
      min-height: 260px;
    }
  }
  .canvas-left-tools {
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 25;
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .add-block-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-surface);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    box-shadow: var(--node-shadow);
  }
  .add-block-btn:hover:not(:disabled),
  .add-block-btn.open {
    border-color: var(--accent);
    background: var(--bg-surface-selected);
  }
  .add-block-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .plus {
    color: var(--accent-bright);
    font-size: 14px;
    line-height: 1;
    font-weight: 700;
  }
  .kbd-hint {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    color: var(--text-dim);
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0 4px;
    line-height: 1.4;
  }
  .canvas-tools {
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 25;
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .c-btn {
    background: var(--overlay-bg);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 4px 9px;
    font-size: 11px;
    cursor: pointer;
    font-family: inherit;
    backdrop-filter: blur(6px);
  }
  .c-btn:hover:not(:disabled),
  .c-btn.active {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
    border-color: var(--accent);
  }
  .c-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .c-btn.icon-q {
    width: 24px;
    padding: 4px 0;
    text-align: center;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-weight: 700;
  }
  .c-btn.danger:hover:not(:disabled) {
    border-color: var(--err-border);
    color: var(--err-strong);
  }
  .wire-popover {
    position: absolute;
    top: 40px;
    right: 10px;
    z-index: 35;
    width: 210px;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 10px 12px;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.3), var(--node-shadow);
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .wire-pop-title {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 2px;
  }
  .wire-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-body);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .wire-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .wire-pop-hint {
    margin-top: 4px;
    padding-top: 6px;
    border-top: 1px solid var(--border-subtle);
    font-size: 10px;
    color: var(--text-dim);
    line-height: 1.4;
  }
  .toast {
    position: absolute;
    bottom: 12px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 30;
    background: var(--err-bg);
    border: 1px solid var(--err-border);
    color: var(--err-text);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
</style>
