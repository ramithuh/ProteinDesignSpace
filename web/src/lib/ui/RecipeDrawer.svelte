<script lang="ts">
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
  let parseErr = $state<string | null>(null);
  let shareStatus = $state<string | null>(null);
  let lastUrlLen = $state<number | null>(null);

  $effect(() => {
    if (dirty) return;
    const plain = $state.snapshot(graph) as WorkflowGraph;
    editorText = format === 'yaml' ? graphToYamlRecipe(plain) : serializeWorkflowJson(plain);
  });

  function switchFormat(next: 'yaml' | 'json') {
    format = next;
    dirty = false;
    parseErr = null;
  }

  function applyEdits() {
    parseErr = null;
    try {
      const next = parseRecipeToGraph(editorText);
      // When applying a YAML recipe that omitted an oversized inline pdbText (>1500 chars),
      // restore the canvas node's existing custom PDB so editing another param never wipes it.
      const prevById = new Map(graph.nodes.map((n) => [n.id, n]));
      const mergedNodes = next.nodes.map((n) => {
        if (format === 'yaml' && n.adapterId === 'pdb-input' && !String(n.params.pdbText ?? '').trim()) {
          const prev = prevById.get(n.id);
          if (
            prev?.adapterId === 'pdb-input' &&
            prev.params.mode === 'custom' &&
            typeof prev.params.pdbText === 'string' &&
            prev.params.pdbText.length > 1500
          ) {
            return {
              ...n,
              params: {
                ...n.params,
                mode: 'custom',
                filename: prev.params.filename,
                pdbText: prev.params.pdbText,
              },
            };
          }
        }
        return n;
      });
      dirty = false;
      onApplyGraph({ ...next, nodes: mergedNodes });
      shareStatus = 'Applied recipe to canvas';
      setTimeout(() => (shareStatus = null), 2000);
    } catch (err) {
      parseErr = err instanceof Error ? err.message : String(err);
    }
  }

  async function copyShareUrl() {
    try {
      const plain = $state.snapshot(graph) as WorkflowGraph;
      const { hash, length, mode, strippedPdb } = await encodeGraphToHash(plain);
      const url = `${window.location.origin}${window.location.pathname}${hash}`;
      window.history.replaceState(null, '', hash);
      await navigator.clipboard.writeText(url);
      lastUrlLen = url.length;
      shareStatus = strippedPdb
        ? `Copied ${url.length}-char link (${mode}; large inline PDB omitted)`
        : `Copied ${url.length}-char link (${mode}, hash ${length} chars)`;
      setTimeout(() => (shareStatus = null), 3500);
    } catch (err) {
      parseErr = err instanceof Error ? err.message : String(err);
    }
  }

  async function copyText() {
    try {
      await navigator.clipboard.writeText(editorText);
      shareStatus = `Copied ${format.toUpperCase()} recipe`;
      setTimeout(() => (shareStatus = null), 2000);
    } catch {
      /* ignore */
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
        <button class="d-btn apply" {disabled} onclick={applyEdits}>Apply to Canvas</button>
      {/if}
    </div>
  </div>

  <textarea
    rows="9"
    spellcheck="false"
    value={editorText}
    oninput={(e) => {
      editorText = e.currentTarget.value;
      dirty = true;
      parseErr = null;
    }}
  ></textarea>

  {#if parseErr}
    <div class="d-err">{parseErr}</div>
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
