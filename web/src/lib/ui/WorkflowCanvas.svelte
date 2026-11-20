<script lang="ts">
  import {
    Background,
    Controls,
    MiniMap,
    SvelteFlow,
    type Connection,
    type Edge,
    type IsValidConnection,
    type Node,
    type NodeTypes,
  } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';
  import { ADAPTER_REGISTRY } from '$lib/engine/registry';
  import {
    PORT_COLORS,
    STAGE_COLORS,
    autoLayoutGraph,
    defaultParamsForAdapter,
    validateConnection,
  } from '$lib/engine/graph-utils';
  import type {
    CanonicalArtifact,
    NodeProgressEvent,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
  } from '$lib/engine/types';
  import ProteinNode, { type ProteinNodeData } from './ProteinNode.svelte';

  interface Props {
    graph: WorkflowGraph;
    selectedNodeId: string;
    nodeStates: Record<string, NodeProgressEvent>;
    outputsByNode: Record<string, Record<string, CanonicalArtifact>>;
    theme?: 'dark' | 'light';
    disabled?: boolean;
    onGraphChange: (next: WorkflowGraph) => void;
    onSelectNode: (nodeId: string) => void;
  }

  let {
    graph,
    selectedNodeId,
    nodeStates,
    outputsByNode,
    theme = 'dark',
    disabled = false,
    onGraphChange,
    onSelectNode,
  }: Props = $props();

  const nodeTypes: NodeTypes = {
    protein: ProteinNode as unknown as NodeTypes[string],
  };

  let validationToast = $state<string | null>(null);
  let toastTimer: ReturnType<typeof setTimeout> | null = null;
  let canvasEl: HTMLDivElement | null = $state(null);

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
    flowNodes = laid.nodes.map((n) => ({
      id: n.id,
      type: 'protein',
      position: n.position ?? { x: 40, y: 80 },
      selected: n.id === selectedNodeId,
      data: {
        adapterId: n.adapterId,
        label: n.label ?? ADAPTER_REGISTRY[n.adapterId]?.shortName ?? n.adapterId,
        state: nodeStates[n.id],
        outputs: outputsByNode[n.id],
        onDelete: disabled ? undefined : removeNode,
      },
    }));

    flowEdges = laid.edges.map((e) => {
      const col = edgeColor(e.fromNode, e.fromPort);
      const srcState = nodeStates[e.fromNode]?.status;
      const dstState = nodeStates[e.toNode]?.status;
      const active =
        srcState === 'running' ||
        srcState === 'loading' ||
        dstState === 'running' ||
        dstState === 'loading';
      return {
        id: `${e.fromNode}:${e.fromPort}->${e.toNode}:${e.toPort}`,
        type: 'smoothstep',
        source: e.fromNode,
        sourceHandle: e.fromPort,
        target: e.toNode,
        targetHandle: e.toPort,
        animated: active,
        style: `stroke: ${col}; stroke-width: 2px;`,
      };
    });
  });

  const isValidConnection: IsValidConnection = (conn) => {
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

  function handleNodeDragStop({ targetNode }: { targetNode: Node | null }) {
    if (!targetNode) return;
    onGraphChange({
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === targetNode.id
          ? { ...n, position: { x: Math.round(targetNode.position.x), y: Math.round(targetNode.position.y) } }
          : n,
      ),
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
    const manifest = ADAPTER_REGISTRY[adapterId];
    if (!manifest) return;
    const count = graph.nodes.filter((n) => n.adapterId === adapterId).length;
    const id = `${adapterId.replace(/[^a-z0-9]/gi, '_')}_${Date.now().toString(36).slice(-4)}`;
    const maxX = graph.nodes.reduce((m, n) => Math.max(m, n.position?.x ?? 0), 0);
    const pos = dropPos ?? {
      x: graph.nodes.length === 0 ? 50 : Math.min(maxX + 60, 700),
      y: 60 + (graph.nodes.length % 4) * 90,
    };
    const newNode: WorkflowNode = {
      id,
      adapterId,
      label: count > 0 ? `${manifest.shortName} #${count + 1}` : manifest.shortName,
      params: defaultParamsForAdapter(adapterId),
      position: pos,
    };
    onGraphChange({ ...graph, nodes: [...graph.nodes, newNode] });
    onSelectNode(id);
  }

  function handleDragStart(ev: DragEvent, adapterId: string) {
    ev.dataTransfer?.setData('application/denovo-adapter', adapterId);
    if (ev.dataTransfer) ev.dataTransfer.effectAllowed = 'copy';
  }

  function handleDrop(ev: DragEvent) {
    ev.preventDefault();
    const adapterId = ev.dataTransfer?.getData('application/denovo-adapter');
    if (!adapterId || !canvasEl) return;
    const rect = canvasEl.getBoundingClientRect();
    addBlock(adapterId, {
      x: Math.max(20, Math.round(ev.clientX - rect.left - 110)),
      y: Math.max(20, Math.round(ev.clientY - rect.top - 40)),
    });
  }

  function handleAutoLayout() {
    onGraphChange(autoLayoutGraph(graph));
  }

  function handleClear() {
    if (disabled) return;
    onGraphChange({ ...graph, nodes: [], edges: [] });
    onSelectNode('');
  }

  async function importGraphJson(ev: Event) {
    const file = (ev.currentTarget as HTMLInputElement).files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text()) as WorkflowGraph;
      if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
        flashToast('Invalid workflow JSON (expected nodes[] and edges[])');
        return;
      }
      const laid = parsed.nodes.some((n) => !n.position) ? autoLayoutGraph(parsed) : parsed;
      onGraphChange(laid);
      onSelectNode(laid.nodes[0]?.id ?? '');
    } catch {
      flashToast('Failed to parse workflow JSON');
    }
  }

  const PALETTE_GROUPS = [
    { title: 'Inputs', ids: ['pdb-input', 'sequence-input'] },
    { title: 'Design / Generate', ids: ['rfd4-proteina', 'legacy-teacher'] },
    { title: 'Inverse Fold', ids: ['web-protein-mpnn'] },
    { title: 'Structure Fold', ids: ['localfold-af2'] },
    { title: 'Evaluate', ids: ['designability-scorer'] },
  ];
</script>

<div class="canvas-shell">
  <!-- Left Block Palette -->
  <aside class="palette">
    <div class="pal-head">BLOCKS <span class="pal-hint">drag or click</span></div>
    {#each PALETTE_GROUPS as grp (grp.title)}
      <div class="pal-group">
        <div class="pal-title">{grp.title}</div>
        {#each grp.ids as aid (aid)}
          {@const m = ADAPTER_REGISTRY[aid]}
          {#if m}
            <button
              class="pal-item"
              draggable={!disabled}
              {disabled}
              ondragstart={(e) => handleDragStart(e, aid)}
              onclick={() => addBlock(aid)}
              style="--sc: {STAGE_COLORS[m.stage] ?? '#64748b'}"
              title={m.description}
            >
              <span class="dot"></span>
              <span class="name">{m.shortName}</span>
              <span class="wt">{m.approxWeightsMB ? `${m.approxWeightsMB}M` : 'CPU'}</span>
            </button>
          {/if}
        {/each}
      </div>
    {/each}

    <div class="legend-box">
      <div class="pal-title">Typed Wires</div>
      <div class="leg-item"><span class="l-dot" style="background: {PORT_COLORS.StructureArtifact.bg}"></span> Structure</div>
      <div class="leg-item"><span class="l-dot" style="background: {PORT_COLORS.SequenceSet.bg}"></span> Sequences</div>
      <div class="leg-item"><span class="l-dot" style="background: {PORT_COLORS.FoldedStructure.bg}"></span> Folds</div>
      <div class="leg-item"><span class="l-dot" style="background: {PORT_COLORS.DesignabilityMetrics.bg}"></span> Metrics</div>
    </div>
  </aside>

  <!-- Interactive SvelteFlow DAG Canvas -->
  <div
    class="flow-wrap"
    bind:this={canvasEl}
    role="application"
    aria-label="Protein design workflow canvas"
    ondragover={(e) => e.preventDefault()}
    ondrop={handleDrop}
  >
    <div class="canvas-tools">
      <button class="c-btn" onclick={handleAutoLayout}>Auto Layout</button>
      <label class="c-btn">
        Import JSON
        <input type="file" accept=".json" hidden {disabled} onchange={importGraphJson} />
      </label>
      <button class="c-btn danger" {disabled} onclick={handleClear}>Clear</button>
    </div>

    {#if validationToast}
      <div class="toast">{validationToast}</div>
    {/if}

    <SvelteFlow
      bind:nodes={flowNodes}
      bind:edges={flowEdges}
      {nodeTypes}
      forceColorMode={theme}
      fitView
      fitViewOptions={{ padding: 0.18 }}
      {isValidConnection}
      onconnect={handleConnect}
      onnodedragstop={handleNodeDragStop}
      ondelete={handleDelete}
      onnodeclick={({ node }) => onSelectNode(node.id)}
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={20} size={1} />
      <Controls showLock={false} />
      <MiniMap
        nodeColor={(n) => {
          const aid = (n.data as ProteinNodeData)?.adapterId;
          const st = ADAPTER_REGISTRY[aid]?.stage ?? 'source';
          return STAGE_COLORS[st] ?? '#475569';
        }}
      />
    </SvelteFlow>
  </div>
</div>

<style>
  .canvas-shell {
    display: grid;
    grid-template-columns: 196px 1fr;
    gap: 10px;
    height: 410px;
  }
  @media (max-width: 780px) {
    .canvas-shell {
      grid-template-columns: 1fr;
      height: auto;
    }
    .flow-wrap {
      height: 360px;
    }
  }
  .palette {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .pal-head {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    color: var(--text-secondary);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .pal-hint {
    font-weight: 400;
    color: var(--text-dim);
  }
  .pal-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .pal-title {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    margin-bottom: 2px;
  }
  .pal-item {
    display: flex;
    align-items: center;
    gap: 7px;
    width: 100%;
    text-align: left;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 6px 8px;
    color: var(--text-body);
    font-size: 11px;
    cursor: grab;
  }
  .pal-item:hover:not(:disabled) {
    border-color: var(--sc);
    background: var(--bg-surface-hover);
  }
  .pal-item:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--sc);
    flex-shrink: 0;
  }
  .name {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .wt {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    color: var(--text-dim);
  }
  .legend-box {
    margin-top: auto;
    padding-top: 8px;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .leg-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .l-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .flow-wrap {
    position: relative;
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .canvas-tools {
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 20;
    display: flex;
    gap: 6px;
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
  }
  .c-btn:hover:not(:disabled) {
    background: var(--bg-surface-hover);
  }
  .c-btn.danger:hover:not(:disabled) {
    border-color: var(--err-border);
    color: var(--err-strong);
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
