<script lang="ts">
  import { tick } from 'svelte';
  import {
    Background,
    Controls,
    MarkerType,
    SvelteFlow,
    useSvelteFlow,
    type Connection,
    type Edge,
    type EdgeTypes,
    type IsValidConnection,
    type Node,
    type NodeTypes,
  } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';
  import { ADAPTER_REGISTRY } from '$lib/engine/registry';
  import type { CatalogInsertRequest } from '$lib/engine/catalog';
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
  import SmartEdge from './SmartEdge.svelte';
  import BlockCatalogPopover from './BlockCatalogPopover.svelte';

  interface Props {
    graph: WorkflowGraph;
    selectedNodeId: string;
    selectedNodeIds?: string[];
    compareColors?: Record<string, { badge: string; css: string }>;
    selectedEdgeId?: string | null;
    nodeStates: Record<string, NodeProgressEvent>;
    outputsByNode: Record<string, Record<string, CanonicalArtifact>>;
    theme?: 'dark' | 'light';
    disabled?: boolean;
    isRunning?: boolean;
    fillHeight?: boolean;
    onGraphChange: (next: WorkflowGraph) => void;
    onSelectNode: (nodeId: string) => void;
    onSelectNodes?: (nodeIds: string[]) => void;
    onBlockAdded?: (nodeId: string) => void;
    onSelectEdge?: (edge: WorkflowEdge) => void;
    onPeekPort?: (nodeId: string, direction: 'in' | 'out', portName: string) => void;
  }

  let {
    graph,
    selectedNodeId,
    selectedNodeIds = [],
    compareColors = {},
    selectedEdgeId = null,
    nodeStates,
    outputsByNode,
    theme = 'dark',
    disabled = false,
    isRunning = false,
    fillHeight = false,
    onGraphChange,
    onSelectNode,
    onSelectNodes = undefined,
    onBlockAdded = undefined,
    onSelectEdge = undefined,
    onPeekPort = undefined,
  }: Props = $props();

  const nodeTypes: NodeTypes = {
    protein: ProteinNode as unknown as NodeTypes[string],
  };

  const edgeTypes: EdgeTypes = {
    smart: SmartEdge as unknown as EdgeTypes[string],
  };

  let validationToast = $state<string | null>(null);
  let toastTimer: ReturnType<typeof setTimeout> | null = null;
  let canvasEl: HTMLDivElement | null = $state(null);
  let addBtnEl: HTMLButtonElement | null = $state(null);
  let wireBtnEl: HTMLButtonElement | null = $state(null);
  let wirePopoverEl: HTMLDivElement | null = $state(null);
  let catalogOpen = $state(false);
  let wireLegendOpen = $state(false);
  // Ephemeral ghost auto-connection proposals (never committed until user accepts via Tab/Enter/click)
  let proposedEdges = $state<WorkflowEdge[]>([]);
  let proposedForNodeId = $state<string | null>(null);
  let blockAcceptUntil = 0;

  function clearProposals() {
    if (proposedEdges.length > 0 || proposedForNodeId !== null) {
      proposedEdges = [];
      proposedForNodeId = null;
    }
  }

  function commitProposedEdge(edge: WorkflowEdge) {
    if (disabled) return;
    const filtered = graph.edges.filter(
      (e) => !(e.toNode === edge.toNode && e.toPort === edge.toPort),
    );
    const check = validateConnection(
      { ...graph, edges: filtered },
      edge.fromNode,
      edge.fromPort,
      edge.toNode,
      edge.toPort,
    );
    if (!check.valid) {
      dismissProposedEdge(edge);
      return;
    }
    onGraphChange({ ...graph, edges: [...filtered, edge] });
    proposedEdges = proposedEdges.filter(
      (p) =>
        !(
          p.fromNode === edge.fromNode &&
          p.fromPort === edge.fromPort &&
          p.toNode === edge.toNode &&
          p.toPort === edge.toPort
        ),
    );
    if (proposedEdges.length === 0) proposedForNodeId = null;
  }

  function dismissProposedEdge(edge: WorkflowEdge) {
    proposedEdges = proposedEdges.filter(
      (p) =>
        !(
          p.fromNode === edge.fromNode &&
          p.fromPort === edge.fromPort &&
          p.toNode === edge.toNode &&
          p.toPort === edge.toPort
        ),
    );
    if (proposedEdges.length === 0) proposedForNodeId = null;
  }

  function commitAllProposedEdges() {
    if (disabled || proposedEdges.length === 0) return;
    let nextEdges = [...graph.edges];
    for (const edge of proposedEdges) {
      const filtered = nextEdges.filter(
        (e) => !(e.toNode === edge.toNode && e.toPort === edge.toPort),
      );
      const check = validateConnection(
        { ...graph, edges: filtered },
        edge.fromNode,
        edge.fromPort,
        edge.toNode,
        edge.toPort,
      );
      if (check.valid) {
        nextEdges = [...filtered, edge];
      }
    }
    clearProposals();
    onGraphChange({ ...graph, edges: nextEdges });
  }

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

  // Clear stale ghost proposals when switching workflows or starting a run
  $effect(() => {
    void graph.id;
    if (isRunning) clearProposals();
  });
  $effect(() => {
    void graph.id;
    clearProposals();
  });

  function handleWindowKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      if (proposedEdges.length > 0) {
        clearProposals();
        return;
      }
      if (wireLegendOpen) {
        wireLegendOpen = false;
      }
      if (catalogOpen) {
        catalogOpen = false;
      }
      if (selectedNodeIds.length > 1 && onSelectNodes) {
        onSelectNodes(selectedNodeId ? [selectedNodeId] : []);
      }
      return;
    }
    if (disabled || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (isEditableElement(ev.target) || isEditableElement(document.activeElement)) return;

    if (
      proposedEdges.length > 0 &&
      !catalogOpen &&
      (ev.key === 'Tab' || ev.key === 'Enter') &&
      performance.now() >= blockAcceptUntil
    ) {
      ev.preventDefault();
      commitAllProposedEdges();
      return;
    }

    if (ev.key === '/' || (ev.shiftKey && (ev.key === 'A' || ev.key === 'a'))) {
      ev.preventDefault();
      clearProposals();
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
    const selSet = new Set(selectedNodeIds.length > 0 ? selectedNodeIds : selectedNodeId ? [selectedNodeId] : []);
    flowNodes = laid.nodes.map((n) => {
      const inputPortsConnected: Record<string, boolean> = {};
      const inputPortsReady: Record<string, boolean> = {};
      for (const e of laid.edges) {
        if (e.toNode === n.id) {
          inputPortsConnected[e.toPort] = true;
          if (outputsByNode[e.fromNode]?.[e.fromPort]) {
            inputPortsReady[e.toPort] = true;
          }
        }
      }
      const outputPortsReady: Record<string, boolean> = {};
      for (const port of Object.keys(outputsByNode[n.id] ?? {})) {
        outputPortsReady[port] = true;
      }
      const cmp = compareColors[n.id];
      return {
        id: n.id,
        type: 'protein',
        position: n.position ?? { x: 40, y: 80 },
        selected: selSet.has(n.id),
        data: {
          adapterId: n.adapterId,
          label: n.label ?? ADAPTER_REGISTRY[n.adapterId]?.shortName ?? n.adapterId,
          params: n.params,
          state: nodeStates[n.id],
          outputs: outputsByNode[n.id],
          isPipelineRunning: isRunning,
          inputPortsConnected,
          inputPortsReady,
          outputPortsReady,
          compareColor: cmp?.css,
          compareBadge: cmp?.badge,
          onDelete: disabled ? undefined : removeNode,
          onPeekPort,
        },
      };
    });

    // Assign small lane staggers if multiple edges fan out from the same source node
    // so parallel skip wires don't overlap on the exact same horizontal track.
    const outCounts = new Map<string, number>();

    flowEdges = laid.edges.map((e) => {
      const eid = makeEdgeId(e.fromNode, e.fromPort, e.toNode, e.toPort);
      const col = edgeColor(e.fromNode, e.fromPort);
      const srcState = nodeStates[e.fromNode]?.status;
      const dstState = nodeStates[e.toNode]?.status;
      const srcDone = srcState === 'done' || srcState === 'cached';
      const dstRunning = dstState === 'running' || dstState === 'loading';
      const dstDone = dstState === 'done' || dstState === 'cached';
      const isSel = eid === selectedEdgeId;

      // Only animate incoming edges actively feeding the currently running block
      const activeFeed = isRunning && srcDone && dstRunning;
      const completedEdge = isRunning && srcDone && dstDone;
      const unreachedEdge = isRunning && !activeFeed && !completedEdge;

      let style: string;
      if (isSel) {
        style = `stroke: ${col}; stroke-width: 3.5px; opacity: 1; filter: drop-shadow(0 0 4px ${col}); cursor: pointer;`;
      } else if (activeFeed) {
        style = `stroke: ${col}; stroke-width: 2.5px; opacity: 1; filter: drop-shadow(0 0 4px ${col}); cursor: pointer;`;
      } else if (completedEdge) {
        style = `stroke: ${col}; stroke-width: 2px; opacity: 0.78; cursor: pointer;`;
      } else if (unreachedEdge) {
        const dimOpacity = theme === 'light' ? 0.34 : 0.25;
        style = `stroke: ${col}; stroke-width: 1.5px; stroke-dasharray: 4 4; opacity: ${dimOpacity}; cursor: pointer;`;
      } else {
        style = `stroke: ${col}; stroke-width: 2px; cursor: pointer;`;
      }

      const fanIdx = outCounts.get(e.fromNode) ?? 0;
      outCounts.set(e.fromNode, fanIdx + 1);
      // Only stagger when >1 edge leaves the same node and routes as a detour
      const laneStagger = fanIdx > 0 ? (fanIdx - 1) * 12 : 0;

      return {
        id: eid,
        type: 'smart',
        source: e.fromNode,
        sourceHandle: e.fromPort,
        target: e.toNode,
        targetHandle: e.toPort,
        selected: isSel,
        animated: activeFeed,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 16,
          height: 16,
          color: col,
        },
        style,
        data: { laneStagger },
      };
    });

    // Append ephemeral ghost proposals (never stored in graph.edges until accepted)
    if (proposedEdges.length > 0 && !isRunning) {
      const nodeIdSet = new Set(laid.nodes.map((n) => n.id));
      const validGhosts = proposedEdges.filter(
        (pe) => nodeIdSet.has(pe.fromNode) && nodeIdSet.has(pe.toNode),
      );
      const ghostFlowEdges: Edge[] = validGhosts.map((pe) => {
        const col = edgeColor(pe.fromNode, pe.fromPort);
        const fanIdx = outCounts.get(pe.fromNode) ?? 0;
        outCounts.set(pe.fromNode, fanIdx + 1);
        const laneStagger = fanIdx > 0 ? (fanIdx - 1) * 12 : 0;
        return {
          id: `ghost:${makeEdgeId(pe.fromNode, pe.fromPort, pe.toNode, pe.toPort)}`,
          type: 'smart',
          source: pe.fromNode,
          sourceHandle: pe.fromPort,
          target: pe.toNode,
          targetHandle: pe.toPort,
          selected: false,
          selectable: false,
          animated: false,
          markerEnd: {
            type: MarkerType.ArrowClosed,
            width: 15,
            height: 15,
            color: col,
          },
          style: `stroke: ${col}; stroke-width: 2px; opacity: 0.68; cursor: pointer;`,
          data: {
            laneStagger,
            proposed: true,
            proposedCount: validGhosts.length,
            toPort: pe.toPort,
            onAccept: () => commitProposedEdge(pe),
            onDismiss: () => dismissProposedEdge(pe),
          },
        };
      });
      flowEdges = [...flowEdges, ...ghostFlowEdges];
    }
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
    // Clear any ghost proposal targeting the same input port
    proposedEdges = proposedEdges.filter((pe) => !(pe.toNode === dst && pe.toPort === toPort));
    if (proposedEdges.length === 0) proposedForNodeId = null;
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
    clearProposals();
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
    clearProposals();
    const nextNodes = graph.nodes.filter((n) => n.id !== nodeId);
    const nextEdges = graph.edges.filter((e) => e.fromNode !== nodeId && e.toNode !== nodeId);
    onGraphChange({ ...graph, nodes: nextNodes, edges: nextEdges });
    if (selectedNodeId === nodeId) onSelectNode(nextNodes[0]?.id ?? '');
  }

  function addBlock(req: string | CatalogInsertRequest, dropPos?: { x: number; y: number }) {
    if (disabled) return;
    const normalized: CatalogInsertRequest =
      typeof req === 'string' ? { adapterId: req } : req;
    if (!ADAPTER_REGISTRY[normalized.adapterId]) return;
    const { graph: nextGraph, nodeId, proposedEdges: nextProposals } = insertBlockSmart(
      graph,
      normalized.adapterId,
      selectedNodeId || undefined,
      dropPos,
      {
        autoCommit: false,
        params: normalized.params,
        label: normalized.label,
      },
    );
    catalogOpen = false;
    // Guard against Enter keystroke from BlockCatalogPopover immediately auto-committing
    blockAcceptUntil = performance.now() + 180;
    proposedEdges = nextProposals;
    proposedForNodeId = nextProposals.length > 0 ? nodeId : null;
    onGraphChange(nextGraph);
    if (onBlockAdded) {
      onBlockAdded(nodeId);
    } else {
      onSelectNode(nodeId);
    }
    scheduleFitView(220);
  }

  function handleDragStart(ev: DragEvent, req: CatalogInsertRequest) {
    ev.dataTransfer?.setData('application/denovo-adapter', req.adapterId);
    ev.dataTransfer?.setData('application/denovo-config', JSON.stringify(req));
    if (ev.dataTransfer) ev.dataTransfer.effectAllowed = 'copy';
  }

  function handleDrop(ev: DragEvent) {
    ev.preventDefault();
    if (disabled) return;
    const rawConfig = ev.dataTransfer?.getData('application/denovo-config');
    const adapterId = ev.dataTransfer?.getData('application/denovo-adapter');
    let req: CatalogInsertRequest | null = null;
    if (rawConfig) {
      try {
        const parsed = JSON.parse(rawConfig) as CatalogInsertRequest;
        if (parsed && typeof parsed.adapterId === 'string') {
          req = parsed;
        }
      } catch {
        /* fallback to adapterId */
      }
    }
    if (!req && adapterId) {
      req = { adapterId };
    }
    if (!req || !canvasEl) return;
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
    addBlock(req, dropPos);
  }

  function handlePaneDblClick(ev: MouseEvent) {
    if (disabled) return;
    const target = ev.target as HTMLElement | null;
    if (
      target?.closest(
        '.svelte-flow__node, .svelte-flow__edge, .canvas-overlay, .svelte-flow__controls, .catalog-shell, .catalog-popover, .catalog-flyout, .wire-popover',
      )
    ) {
      return;
    }
    wireLegendOpen = false;
    catalogOpen = true;
  }

  async function handleAutoLayout() {
    if (disabled) return;
    clearProposals();
    onGraphChange(autoLayoutGraph(graph));
    scheduleFitView(250);
  }

  function handleClear() {
    if (disabled) return;
    clearProposals();
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
  class:is-running={isRunning}
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
      onInsert={(req) => addBlock(req)}
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
      <div class="wire-pop-hint">Only matching data types connect. Click any wire or port label to inspect outputs.</div>
    </div>
  {/if}

  {#if validationToast}
    <div class="toast">{validationToast}</div>
  {/if}

  {#if proposedEdges.length > 1 && !isRunning}
    <div class="ghost-bar canvas-overlay" role="region" aria-label="Proposed connections">
      <button type="button" class="gbar-accept" onclick={commitAllProposedEdges}>
        ✓ Connect {proposedEdges.length} wires <kbd>Tab</kbd>
      </button>
      <button type="button" class="gbar-dismiss" onclick={clearProposals}>
        Dismiss <kbd>Esc</kbd>
      </button>
    </div>
  {/if}

  <div class="flow-inner">
    <SvelteFlow
      bind:nodes={flowNodes}
      bind:edges={flowEdges}
      {nodeTypes}
      {edgeTypes}
      forceColorMode={theme}
      nodesDraggable={!disabled}
      nodesConnectable={!disabled}
      zoomOnDoubleClick={false}
      multiSelectionKey={['Meta', 'Control', 'Shift']}
      deleteKey={disabled ? null : ['Backspace', 'Delete']}
      fitView
      fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
      {isValidConnection}
      onconnect={handleConnect}
      onnodedragstop={handleNodeDragStop}
      ondelete={handleDelete}
      onpaneclick={() => {
        clearProposals();
        catalogOpen = false;
        wireLegendOpen = false;
      }}
      onnodeclick={({ node, event }) => {
        catalogOpen = false;
        wireLegendOpen = false;
        if (proposedForNodeId && node.id !== proposedForNodeId) {
          clearProposals();
        }
        const multiKey = Boolean(event?.metaKey || event?.ctrlKey || event?.shiftKey);
        if (multiKey && onSelectNodes) {
          const cur =
            selectedNodeIds.length > 0
              ? selectedNodeIds
              : selectedNodeId
                ? [selectedNodeId]
                : [];
          if (cur.includes(node.id)) {
            if (cur.length > 1) {
              onSelectNodes(cur.filter((id) => id !== node.id));
            }
          } else if (cur.length >= 4) {
            flashToast('Compare mode supports up to 4 blocks at once');
          } else {
            onSelectNodes([...cur, node.id]);
          }
          return;
        }
        onSelectNode(node.id);
      }}
      onedgeclick={({ edge }) => {
        catalogOpen = false;
        wireLegendOpen = false;
        if (edge.id.startsWith('ghost:')) {
          const matched = proposedEdges.find(
            (pe) => `ghost:${makeEdgeId(pe.fromNode, pe.fromPort, pe.toNode, pe.toPort)}` === edge.id,
          );
          if (matched) commitProposedEdge(matched);
          return;
        }
        clearProposals();
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
  .flow-wrap.is-running :global(.svelte-flow__node) {
    cursor: pointer;
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
  .ghost-bar {
    position: absolute;
    bottom: 14px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 24;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 8px;
    border-radius: 999px;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.26);
  }
  .gbar-accept,
  .gbar-dismiss {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border: none;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
  }
  .gbar-accept {
    background: #2563eb;
    color: #fff;
  }
  .gbar-accept:hover {
    background: #1d4ed8;
  }
  .gbar-dismiss {
    background: transparent;
    color: var(--text-muted);
  }
  .gbar-dismiss:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }
  .ghost-bar kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    padding: 0 4px;
    border-radius: 3px;
    border: 1px solid rgba(255, 255, 255, 0.28);
    background: rgba(255, 255, 255, 0.14);
  }
  .gbar-dismiss kbd {
    border-color: var(--border-strong);
    background: var(--bg-surface-2);
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
