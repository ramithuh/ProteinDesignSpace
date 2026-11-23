<script lang="ts">
  import { onDestroy } from 'svelte';
  import { defaultPluginRegistry } from '$lib/plugins';
  import {
    encodeGraphToHash,
    graphToYamlRecipe,
    parseRecipeToGraph,
    serializeWorkflowJson,
  } from '$lib/engine/share';
  import type { WorkflowGraph } from '$lib/engine/types';

  interface Props {
    graph: WorkflowGraph;
    disabled?: boolean;
    onApplyGraph: (next: WorkflowGraph) => void;
  }

  let { graph, disabled = false, onApplyGraph }: Props = $props();

  let format = $state<'yaml' | 'json'>('yaml');
  let editorText = $state('');
  let dirty = $state(false);
  let canvasChangedWhileDirty = $state(false);
  let parseErr = $state<string | null>(null);
  let pendingSwitchFormat = $state<'yaml' | 'json' | null>(null);
  let shareStatus = $state<string | null>(null);
  let lastUrlLen = $state<number | null>(null);
  let textareaEl: HTMLTextAreaElement | null = $state(null);
  let statusTimer: ReturnType<typeof setTimeout> | null = null;
  let lastGraphId = '';
  let lastSyncedCanvasText = '';

  function serializeCanvas(g: WorkflowGraph, fmt: 'yaml' | 'json'): string {
    return fmt === 'yaml' ? graphToYamlRecipe(g) : serializeWorkflowJson(g);
  }

  function clearStatus() {
    shareStatus = null;
    if (statusTimer) {
      clearTimeout(statusTimer);
      statusTimer = null;
    }
  }

  function flashStatus(msg: string, ms = 3000) {
    shareStatus = msg;
    if (statusTimer) clearTimeout(statusTimer);
    statusTimer = setTimeout(() => {
      shareStatus = null;
      statusTimer = null;
    }, ms);
  }

  onDestroy(() => {
    clearStatus();
  });

  $effect(() => {
    const gid = graph.id;
    const plain = $state.snapshot(graph) as WorkflowGraph;
    const latestCanvasText = serializeCanvas(plain, format);
    if (gid !== lastGraphId) {
      lastGraphId = gid;
      dirty = false;
      canvasChangedWhileDirty = false;
      parseErr = null;
      pendingSwitchFormat = null;
      lastSyncedCanvasText = latestCanvasText;
      editorText = latestCanvasText;
      return;
    }
    if (dirty) {
      if (latestCanvasText !== lastSyncedCanvasText) {
        canvasChangedWhileDirty = true;
      }
      return;
    }
    canvasChangedWhileDirty = false;
    lastSyncedCanvasText = latestCanvasText;
    editorText = latestCanvasText;
  });

  function reloadFromCanvas() {
    const plain = $state.snapshot(graph) as WorkflowGraph;
    const latestCanvasText = serializeCanvas(plain, format);
    dirty = false;
    canvasChangedWhileDirty = false;
    parseErr = null;
    pendingSwitchFormat = null;
    lastSyncedCanvasText = latestCanvasText;
    editorText = latestCanvasText;
    flashStatus('Reloaded recipe from canvas', 2000);
  }

  function switchFormat(next: 'yaml' | 'json') {
    if (next === format) return;
    clearStatus();
    pendingSwitchFormat = null;
    if (!dirty) {
      format = next;
      parseErr = null;
      return;
    }
    // Preserve valid unapplied edits across YAML <-> JSON without losing work
    try {
      const parsed = parseRecipeToGraph(editorText);
      format = next;
      editorText = next === 'yaml' ? graphToYamlRecipe(parsed) : serializeWorkflowJson(parsed);
      parseErr = null;
    } catch (err) {
      // Unapplied edits have a syntax error — ask before discarding instead of silently wiping
      parseErr = err instanceof Error ? err.message : String(err);
      pendingSwitchFormat = next;
    }
  }

  function confirmDiscardAndSwitch() {
    if (!pendingSwitchFormat) return;
    const next = pendingSwitchFormat;
    clearStatus();
    pendingSwitchFormat = null;
    dirty = false;
    parseErr = null;
    format = next;
  }

  function applyEdits() {
    parseErr = null;
    clearStatus();
    try {
      const next = parseRecipeToGraph(editorText);
      const prevById = new Map(graph.nodes.map((n) => [n.id, n]));
      const mergedNodes = next.nodes.map((n) => {
        if (format === 'yaml') {
          const plugin = defaultPluginRegistry.getPlugin(n.adapterId);
          const prev = prevById.get(n.id);
          if (plugin?.share.restoreOmittedInlineAsset && prev && prev.adapterId === n.adapterId) {
            const restored = plugin.share.restoreOmittedInlineAsset(
              prev.params,
              n.params,
              editorText,
              n.id,
            );
            if (restored) {
              return { ...n, params: restored };
            }
          }
        }
        return n;
      });
      dirty = false;
      onApplyGraph({ ...next, nodes: mergedNodes });
      flashStatus('Applied recipe to canvas', 2000);
    } catch (err) {
      clearStatus();
      parseErr = err instanceof Error ? err.message : String(err);
    }
  }

  async function copyShareUrl() {
    parseErr = null;
    clearStatus();
    try {
      const plain = $state.snapshot(graph) as WorkflowGraph;
      const { hash, length, mode, strippedPdb } = await encodeGraphToHash(plain);
      const url = `${window.location.origin}${window.location.pathname}${hash}`;
      window.history.replaceState(null, '', hash);
      lastUrlLen = url.length;
      try {
        if (!navigator.clipboard?.writeText) {
          throw new Error('Clipboard API unavailable');
        }
        await navigator.clipboard.writeText(url);
        flashStatus(
          strippedPdb
            ? `Copied ${url.length}-char link (${mode}; large inline PDB omitted)`
            : `Copied ${url.length}-char link (${mode}, hash ${length} chars)`,
          3500,
        );
      } catch {
        flashStatus(
          strippedPdb
            ? `Share link updated in address bar (${mode}; large PDB omitted — copy from URL bar)`
            : `Share link updated in address bar (${mode} — copy from URL bar)`,
          4200,
        );
      }
    } catch (err) {
      clearStatus();
      parseErr = err instanceof Error ? err.message : String(err);
    }
  }

  async function copyText() {
    parseErr = null;
    clearStatus();
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error('Clipboard API unavailable');
      }
      await navigator.clipboard.writeText(editorText);
      flashStatus(`Copied ${format.toUpperCase()} recipe`, 2000);
    } catch {
      textareaEl?.focus();
      textareaEl?.select();
      flashStatus(
        `Clipboard unavailable — ${format.toUpperCase()} selected, copy with Ctrl/⌘+C`,
        3800,
      );
    }
  }
</script>

<div class="drawer">
  <div class="d-head">
    <div class="d-left">
      <span class="d-title">WORKFLOW LANGUAGE</span>
      <div class="fmt-tabs">
        <button class:active={format === 'yaml'} onclick={() => switchFormat('yaml')}>YAML Recipe</button>
        <button class:active={format === 'json'} onclick={() => switchFormat('json')}>Full JSON</button>
      </div>
      {#if shareStatus}
        <span class="status-ok">{shareStatus}</span>
      {:else if lastUrlLen}
        <span class="status-dim">last link: {lastUrlLen} chars</span>
      {/if}
    </div>
    <div class="d-actions">
      <button class="d-btn" onclick={copyText}>Copy {format.toUpperCase()}</button>
      <button class="d-btn primary" onclick={copyShareUrl}>🔗 Copy Short Link</button>
      {#if dirty}
        <button class="d-btn" onclick={reloadFromCanvas} title="Discard text edits and reload current canvas">
          ↺ Reload
        </button>
        <button class="d-btn apply" {disabled} onclick={applyEdits}>Apply to Canvas</button>
      {/if}
    </div>
  </div>

  {#if dirty && canvasChangedWhileDirty}
    <div class="discard-warn" role="status">
      <span>Canvas changed since you started editing this recipe.</span>
      <div class="discard-actions">
        <button type="button" class="d-btn" onclick={reloadFromCanvas}>
          ↻ Reload from canvas
        </button>
      </div>
    </div>
  {/if}

  <textarea
    bind:this={textareaEl}
    rows="9"
    spellcheck="false"
    value={editorText}
    oninput={(e) => {
      editorText = e.currentTarget.value;
      const matchesCanvas = editorText === lastSyncedCanvasText && !canvasChangedWhileDirty;
      dirty = !matchesCanvas;
      parseErr = null;
      pendingSwitchFormat = null;
      clearStatus();
    }}
  ></textarea>

  {#if pendingSwitchFormat}
    <div class="discard-warn" role="alert">
      <span>
        Unapplied edits have a syntax error and cannot be converted to
        <b>{pendingSwitchFormat === 'yaml' ? 'YAML Recipe' : 'Full JSON'}</b>.
      </span>
      <div class="discard-actions">
        <button type="button" class="d-btn danger" onclick={confirmDiscardAndSwitch}>
          Discard edits &amp; switch
        </button>
        <button type="button" class="d-btn" onclick={() => (pendingSwitchFormat = null)}>
          Stay here
        </button>
      </div>
    </div>
  {/if}

  {#if parseErr}
    <div class="d-err" role="alert">{parseErr}</div>
  {/if}
</div>

<style>
  .drawer {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .d-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }
  .d-left {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .d-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.7px;
    color: var(--text-muted);
  }
  .fmt-tabs {
    display: inline-flex;
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    overflow: hidden;
  }
  .fmt-tabs button {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 11px;
    padding: 3px 9px;
    cursor: pointer;
  }
  .fmt-tabs button.active {
    background: var(--bg-pill);
    color: var(--accent-soft);
    font-weight: 600;
  }
  .status-ok {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--ok-text);
  }
  .status-dim {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
  }
  .d-actions {
    display: flex;
    gap: 6px;
  }
  .d-btn {
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 11px;
    cursor: pointer;
    font-family: inherit;
  }
  .d-btn:hover:not(:disabled) {
    background: var(--bg-surface-hover);
  }
  .d-btn.primary {
    background: var(--bg-pill);
    color: var(--accent-soft);
    border-color: var(--accent);
    font-weight: 600;
  }
  .d-btn.apply {
    background: #2563eb;
    color: #fff;
    border-color: #3b82f6;
    font-weight: 600;
  }
  .d-btn.danger {
    background: var(--err-bg);
    color: var(--err-text);
    border-color: var(--err-border);
    font-weight: 600;
  }
  .discard-warn {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 6px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
    font-size: 11px;
  }
  .discard-actions {
    display: flex;
    gap: 6px;
    flex-shrink: 0;
  }
  textarea {
    width: 100%;
    box-sizing: border-box;
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    padding: 10px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11.5px;
    line-height: 1.5;
    resize: vertical;
  }
  .d-err {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--err-strong);
  }
</style>
