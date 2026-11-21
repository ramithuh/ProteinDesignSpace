<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import {
    requestWorkflowPlan,
    type PlannedWorkflowResult,
  } from '$lib/engine/workflow-planner';
  import type { WorkflowGraph } from '$lib/engine/types';

  interface Props {
    disabled?: boolean;
    triggerEl?: HTMLElement | null;
    onApply: (graph: WorkflowGraph, summary: string) => void;
    onClose: () => void;
  }

  let { disabled = false, triggerEl = null, onApply, onClose }: Props = $props();

  const EXAMPLES = [
    'Evaluate Genie 3 and RFD4 on the di-iron 4-helix bundle motif using AF2, RF3, and Protenix',
    'Try RFD4 and Genie 3 for 100 length protein design and check refolding',
    'Crop 3PTB to chain A keeping BEN and CA and design sequences with LigandMPNN',
    'Design CDR3 on heavy chain H of 1FDL using AntiFold',
  ];

  let promptText = $state('');
  let loading = $state(false);
  let errorMsg = $state<string | null>(null);
  let result = $state<PlannedWorkflowResult | null>(null);
  let shellEl: HTMLDivElement | null = $state(null);
  let textareaEl: HTMLTextAreaElement | null = $state(null);
  let activeController: AbortController | null = null;

  function abortInFlight() {
    if (activeController) {
      activeController.abort();
      activeController = null;
    }
  }

  async function submitPrompt(overrideText?: string) {
    const q = (overrideText ?? promptText).trim();
    if (!q || loading || disabled) return;
    if (overrideText !== undefined) promptText = overrideText;

    abortInFlight();
    const ctrl = new AbortController();
    activeController = ctrl;
    loading = true;
    errorMsg = null;
    result = null;

    try {
      const planned = await requestWorkflowPlan(q, ctrl.signal);
      if (ctrl.signal.aborted) return;
      result = planned;
    } catch (err) {
      if (ctrl.signal.aborted) return;
      errorMsg = err instanceof Error ? err.message : String(err);
    } finally {
      if (activeController === ctrl) activeController = null;
      loading = false;
    }
  }

  function handleApply() {
    if (!result || disabled) return;
    onApply(result.graph, result.plan.summary);
    onClose();
  }

  function handleKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      ev.preventDefault();
      ev.stopPropagation();
      onClose();
      return;
    }
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      void submitPrompt();
    }
  }

  function handleOutsidePointerDown(ev: PointerEvent) {
    const target = ev.target as Node | null;
    if (!target) return;
    if (shellEl?.contains(target) || triggerEl?.contains(target)) return;
    onClose();
  }

  onMount(() => {
    window.addEventListener('pointerdown', handleOutsidePointerDown, true);
    void tick().then(() => textareaEl?.focus());
    return () => {
      window.removeEventListener('pointerdown', handleOutsidePointerDown, true);
    };
  });

  onDestroy(() => {
    abortInFlight();
  });

  let shortModelName = $derived.by(() => {
    const raw = result?.model || '@cf/google/gemma-4-26b-a4b-it';
    return raw.split('/').pop() || raw;
  });
</script>

<div
  bind:this={shellEl}
  class="planner-popover canvas-overlay"
  role="dialog"
  aria-label="Describe workflow in high level"
>
  <div class="pp-header">
    <div class="pp-title-row">
      <span class="pp-sparkle" aria-hidden="true">✦</span>
      <span class="pp-title">Describe your workflow</span>
      <span class="pp-badge">Cloudflare Workers AI</span>
    </div>
    <button
      type="button"
      class="pp-close"
      onclick={onClose}
      aria-label="Close workflow planner"
      title="Close (Esc)"
    >
      ×
    </button>
  </div>

  <div class="pp-notice" role="note">
    <strong>External model notice:</strong> This helper sends your short text description to
    <strong>Cloudflare Workers AI</strong> (<code>gemma-4-26b-a4b-it</code>, free tier — no API key needed)
    to draft the block diagram. All protein generation, sequence design, and folding still run
    100% locally in your browser on WebGPU.
  </div>

  <div class="pp-input-wrap">
    <label class="sr-only" for="planner-prompt-input">Describe your protein design workflow</label>
    <textarea
      bind:this={textareaEl}
      id="planner-prompt-input"
      name="plannerPrompt"
      class="pp-textarea"
      rows="3"
      maxlength={320}
      bind:value={promptText}
      onkeydown={handleKeydown}
      disabled={loading || disabled}
      placeholder="e.g. Evaluate Genie 3 and RFD4 on the di-iron 4-helix bundle motif using AF2, RF3, and Protenix..."
    ></textarea>
    <div class="pp-input-foot">
      <span class="pp-char-count">{promptText.length}/320</span>
      <button
        type="button"
        class="pp-submit-btn"
        disabled={loading || disabled || !promptText.trim()}
        onclick={() => void submitPrompt()}
      >
        {#if loading}
          <span class="pp-spinner" aria-hidden="true"></span>
          Drafting graph...
        {:else}
          Draft Workflow <kbd>↵</kbd>
        {/if}
      </button>
    </div>
  </div>

  {#if !result && !loading}
    <div class="pp-examples">
      <div class="pp-ex-label">Try an example</div>
      <div class="pp-ex-list">
        {#each EXAMPLES as ex (ex)}
          <button
            type="button"
            class="pp-ex-chip"
            disabled={loading || disabled}
            onclick={() => void submitPrompt(ex)}
          >
            {ex}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  {#if errorMsg}
    <div class="pp-error" role="alert">
      {errorMsg}
    </div>
  {/if}

  {#if result}
    <div class="pp-preview">
      <div class="pp-prev-head">
        <div class="pp-prev-summary">{result.plan.summary}</div>
        <span class="pp-prev-meta">
          {result.graph.nodes.length} blocks · {result.graph.edges.length} wires · {shortModelName}{#if result.latencyMs > 0} · {(result.latencyMs / 1000).toFixed(1)}s{/if}
        </span>
      </div>

      <div class="pp-node-pills">
        {#each result.graph.nodes as n (n.id)}
          <span class="pp-node-pill">{n.label || n.adapterId}</span>
        {/each}
      </div>

      {#if result.notes.length > 0}
        <div class="pp-notes">
          {#each result.notes as note (note)}
            <div class="pp-note-item">{note}</div>
          {/each}
        </div>
      {/if}

      <div class="pp-prev-actions">
        <button type="button" class="pp-apply-btn" {disabled} onclick={handleApply}>
          Apply to Canvas
        </button>
        <button
          type="button"
          class="pp-retry-btn"
          disabled={loading}
          onclick={() => {
            result = null;
            void tick().then(() => textareaEl?.focus());
          }}
        >
          Edit prompt
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .planner-popover {
    position: absolute;
    left: 50%;
    bottom: 62px;
    transform: translateX(-50%);
    z-index: 42;
    width: min(500px, calc(100% - 20px));
    max-height: min(460px, calc(100% - 82px));
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 12px 14px;
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-surface) 97%, transparent);
    -webkit-backdrop-filter: blur(14px);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border-strong);
    box-shadow: var(--shadow-lg, 0 18px 44px rgba(15, 23, 42, 0.24));
    color: var(--text-primary);
  }

  .pp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .pp-title-row {
    display: flex;
    align-items: center;
    gap: 7px;
    flex-wrap: wrap;
  }
  .pp-sparkle {
    color: var(--accent-bright);
    font-size: 13px;
    line-height: 1;
  }
  .pp-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary);
  }
  .pp-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 999px;
    background: var(--accent-soft);
    border: 1px solid var(--accent-border);
    color: var(--accent-bright);
  }
  .pp-close {
    background: none;
    border: none;
    color: var(--text-dim);
    font-size: 17px;
    line-height: 1;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 6px;
  }
  .pp-close:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }

  .pp-notice {
    font-size: 11px;
    line-height: 1.45;
    color: var(--text-secondary);
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 8px 10px;
  }
  .pp-notice strong {
    color: var(--text-primary);
    font-weight: 600;
  }
  .pp-notice code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    padding: 1px 4px;
    border-radius: 4px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
  }

  .pp-input-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .pp-textarea {
    width: 100%;
    box-sizing: border-box;
    resize: vertical;
    min-height: 66px;
    max-height: 130px;
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-family: inherit;
    font-size: 12.5px;
    line-height: 1.45;
  }
  .pp-textarea:focus {
    outline: none;
    border-color: var(--accent-bright);
    box-shadow: 0 0 0 2px var(--accent-soft);
  }
  .pp-input-foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .pp-char-count {
    font-size: 10.5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-dim);
  }
  .pp-submit-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 8px;
    border: 1px solid var(--accent-border);
    background: var(--accent-bright);
    color: #fff;
    font-family: inherit;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .pp-submit-btn:hover:not(:disabled) {
    filter: brightness(1.06);
  }
  .pp-submit-btn:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
  .pp-submit-btn kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    padding: 0 4px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.2);
  }
  .pp-spinner {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    border: 2px solid rgba(255, 255, 255, 0.35);
    border-top-color: #fff;
    animation: pp-spin 0.7s linear infinite;
  }
  @keyframes pp-spin {
    to {
      transform: rotate(360deg);
    }
  }

  .pp-examples {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .pp-ex-label {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
  }
  .pp-ex-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .pp-ex-chip {
    text-align: left;
    padding: 6px 9px;
    border-radius: 7px;
    border: 1px solid var(--border-subtle);
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    font-family: inherit;
    font-size: 11.5px;
    line-height: 1.35;
    cursor: pointer;
    transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
  }
  .pp-ex-chip:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    border-color: var(--accent-border);
    color: var(--text-primary);
  }

  .pp-error {
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid rgba(239, 68, 68, 0.35);
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    font-size: 11.5px;
    line-height: 1.4;
  }

  .pp-preview {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px;
    border-radius: 9px;
    border: 1px solid var(--accent-border);
    background: color-mix(in srgb, var(--accent-soft) 55%, var(--bg-surface));
  }
  .pp-prev-head {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .pp-prev-summary {
    font-size: 12.5px;
    font-weight: 700;
    color: var(--text-primary);
  }
  .pp-prev-meta {
    font-size: 10.5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-muted);
  }
  .pp-node-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .pp-node-pill {
    font-size: 10.5px;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 6px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    color: var(--text-secondary);
  }
  .pp-notes {
    font-size: 11px;
    color: var(--text-secondary);
    border-left: 2px solid var(--accent-bright);
    padding-left: 8px;
  }
  .pp-prev-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 2px;
  }
  .pp-apply-btn {
    padding: 6px 13px;
    border-radius: 8px;
    border: 1px solid var(--accent-border);
    background: var(--accent-bright);
    color: #fff;
    font-family: inherit;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .pp-apply-btn:hover:not(:disabled) {
    filter: brightness(1.06);
  }
  .pp-retry-btn {
    padding: 6px 10px;
    border-radius: 8px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-secondary);
    font-family: inherit;
    font-size: 11.5px;
    font-weight: 500;
    cursor: pointer;
  }
  .pp-retry-btn:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }
</style>
