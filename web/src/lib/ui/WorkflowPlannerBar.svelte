<script lang="ts">
  import { onMount, tick } from 'svelte';
  import {
    requestWorkflowPlan,
    type PlannedWorkflowResult,
  } from '$lib/engine/workflow-planner';
  import type { WorkflowGraph } from '$lib/engine/types';

  interface ExampleChip {
    label: string;
    prompt: string;
    requiresExisting?: boolean;
  }

  interface Props {
    graph: WorkflowGraph;
    disabled?: boolean;
    modKeyLabel?: string;
    draftPreview: PlannedWorkflowResult | null;
    onDraftChange: (draft: PlannedWorkflowResult | null) => void;
    onPromptStateChange?: (hasText: boolean, loading: boolean) => void;
    onApply: (graph: WorkflowGraph, summary: string) => void;
    onClose: () => void;
  }

  let {
    graph,
    disabled = false,
    modKeyLabel = '⌘',
    draftPreview,
    onDraftChange,
    onPromptStateChange = undefined,
    onApply,
    onClose,
  }: Props = $props();

  const ALL_EXAMPLES: ExampleChip[] = [
    {
      label: '+ ThermoMPNN & RF3',
      prompt: 'Add ThermoMPNN and a RoseTTAFold 3 refolding branch to my current graph',
      requiresExisting: true,
    },
    {
      label: 'Di-iron motif: Genie 3 vs RFD4',
      prompt:
        'Evaluate Genie 3 and RFD4 on the di-iron 4-helix bundle motif using AF2, RF3, and Protenix',
    },
    {
      label: 'LigandMPNN vs LASErMPNN (3PTB)',
      prompt: 'Compare LigandMPNN vs LASErMPNN on 3PTB keeping BEN and CA',
    },
    {
      label: '100aa de novo: RFD4 + Genie 3',
      prompt: 'Try RFD4 and Genie 3 for 100 length protein design and check refolding',
    },
  ];

  let examples = $derived.by(() => {
    const hasNodes = graph.nodes.length > 0;
    return ALL_EXAMPLES.filter((ex) => !ex.requiresExisting || hasNodes).slice(0, 3);
  });

  let prompt = $state('');
  let loading = $state(false);
  let errorMsg = $state<string | null>(null);
  let infoOpen = $state(false);
  let notesExpanded = $state(false);
  let isMultiline = $state(false);
  let textareaEl: HTMLTextAreaElement | null = $state(null);
  let infoWrapEl: HTMLDivElement | null = $state(null);
  let abortCtrl: AbortController | null = null;

  let trimmedPrompt = $derived(prompt.trim());
  let showExamples = $derived(
    trimmedPrompt.length === 0 && !loading && !draftPreview && !errorMsg && !infoOpen,
  );
  let showCharCount = $derived(prompt.length > 220 && !draftPreview);

  $effect(() => {
    onPromptStateChange?.(trimmedPrompt.length > 0, loading);
  });

  function resizeTextarea() {
    const el = textareaEl;
    if (!el) return;
    if (prompt.length === 0) {
      el.style.height = '24px';
      isMultiline = false;
      return;
    }
    el.style.height = '24px';
    const nextH = Math.min(el.scrollHeight, 84);
    el.style.height = `${Math.max(24, nextH)}px`;
    isMultiline = nextH > 32;
  }

  export function focusInput() {
    void tick().then(() => {
      resizeTextarea();
      textareaEl?.focus();
    });
  }

  export function abortPending() {
    if (abortCtrl) {
      abortCtrl.abort();
      abortCtrl = null;
    }
    loading = false;
  }

  export function handleEscape(): boolean {
    if (infoOpen) {
      infoOpen = false;
      return true;
    }
    if (loading) {
      abortPending();
      focusInput();
      return true;
    }
    if (draftPreview) {
      notesExpanded = false;
      onDraftChange(null);
      focusInput();
      return true;
    }
    onClose();
    return true;
  }

  export function handleConfirmShortcut(): boolean {
    if (!draftPreview || disabled) return false;
    applyDraft();
    return true;
  }

  onMount(() => {
    focusInput();
    function onDocPointerDown(ev: PointerEvent) {
      if (!infoOpen) return;
      const target = ev.target;
      if (!(target instanceof Node)) return;
      if (infoWrapEl?.contains(target)) return;
      infoOpen = false;
    }
    window.addEventListener('pointerdown', onDocPointerDown, true);
    return () => {
      abortCtrl?.abort();
      window.removeEventListener('pointerdown', onDocPointerDown, true);
    };
  });

  async function submitPrompt(overrideText?: string) {
    const text = (overrideText ?? prompt).trim();
    if (!text || loading || disabled) return;
    if (overrideText !== undefined) {
      prompt = overrideText;
      await tick();
      resizeTextarea();
    }

    abortCtrl?.abort();
    abortCtrl = new AbortController();
    loading = true;
    errorMsg = null;
    infoOpen = false;
    notesExpanded = false;
    onDraftChange(null);

    try {
      const planned = await requestWorkflowPlan(text, {
        currentGraph: graph.nodes.length > 0 ? graph : undefined,
        signal: abortCtrl.signal,
      });
      onDraftChange(planned);
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') return;
      errorMsg = err instanceof Error ? err.message : 'Failed to plan workflow.';
      focusInput();
    } finally {
      loading = false;
      abortCtrl = null;
    }
  }

  function applyDraft() {
    if (!draftPreview || disabled) return;
    const { graph: nextGraph, plan } = draftPreview;
    onDraftChange(null);
    onApply(nextGraph, plan.summary);
    onClose();
  }

  function refineDraft() {
    notesExpanded = false;
    onDraftChange(null);
    focusInput();
  }

  function discardAndClose() {
    abortPending();
    onDraftChange(null);
    onClose();
  }

  function onTextareaKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      void submitPrompt();
    } else if (ev.key === 'Escape') {
      ev.preventDefault();
      ev.stopPropagation();
      handleEscape();
    }
  }
</script>

<div class="planner-bar-root" class:is-multiline={isMultiline} class:is-review={Boolean(draftPreview)}>
  <!-- Floating strip ABOVE the dock bar: error OR starter chips OR compiler notes -->
  {#if errorMsg}
    <div class="above-strip err-pill" role="alert">
      <span class="err-msg">{errorMsg}</span>
      <button
        type="button"
        class="strip-x"
        onclick={() => {
          errorMsg = null;
          focusInput();
        }}
        aria-label="Dismiss error"
      >
        ×
      </button>
    </div>
  {:else if showExamples}
    <div class="above-strip chip-row" role="list" aria-label="Example workflow prompts">
      {#each examples as ex (ex.label)}
        <button
          type="button"
          class="ex-pill"
          title={ex.prompt}
          onclick={() => void submitPrompt(ex.prompt)}
          {disabled}
        >
          <span class="ex-spark" aria-hidden="true">✦</span>
          <span>{ex.label}</span>
        </button>
      {/each}
    </div>
  {:else if draftPreview && draftPreview.notes.length > 0}
    <div class="above-strip note-pill">
      <button
        type="button"
        class="note-toggle"
        onclick={() => (notesExpanded = !notesExpanded)}
        title={draftPreview.notes.join(' · ')}
      >
        <span class="note-spark" aria-hidden="true">✦</span>
        <span class="note-text" class:expanded={notesExpanded}>
          {notesExpanded ? draftPreview.notes.join(' · ') : draftPreview.notes[0]}
        </span>
        {#if draftPreview.notes.length > 1}
          <span class="note-more">+{draftPreview.notes.length - 1}</span>
        {/if}
      </button>
    </div>
  {/if}

  <!-- Main inline dock surface: Typing Bar OR Review Bar -->
  {#if draftPreview}
    <div class="review-row" role="region" aria-label="Workflow preview controls">
      <div class="review-info">
        <span class="mode-pill" class:edit={draftPreview.isIncrementalEdit}>
          {draftPreview.isIncrementalEdit
            ? `+${draftPreview.addedNodeCount} block${draftPreview.addedNodeCount === 1 ? '' : 's'}`
            : `New · ${draftPreview.graph.nodes.length} blocks`}
        </span>
        <span class="review-summary" title={draftPreview.plan.summary}>
          {draftPreview.plan.summary}
        </span>
        {#if draftPreview.latencyMs > 0}
          <span class="review-ms">{draftPreview.latencyMs}ms</span>
        {/if}
      </div>
      <div class="review-actions">
        <button
          type="button"
          class="btn-apply"
          onclick={applyDraft}
          {disabled}
          title="Apply drafted workflow to canvas (Enter or Tab)"
        >
          ✓ Apply <kbd>↵</kbd>
        </button>
        <button
          type="button"
          class="btn-ghost"
          onclick={refineDraft}
          {disabled}
          title="Edit prompt and re-draft (Esc)"
        >
          Refine
        </button>
        <button
          type="button"
          class="btn-icon-x"
          onclick={discardAndClose}
          title="Discard preview and close"
          aria-label="Discard preview and close"
        >
          ×
        </button>
      </div>
    </div>
  {:else}
    <div class="input-row">
      <span class="bar-sparkle" class:spinning={loading} aria-hidden="true">✦</span>
      <textarea
        bind:this={textareaEl}
        bind:value={prompt}
        class="bar-textarea"
        class:is-empty={prompt.length === 0}
        rows="1"
        maxlength="320"
        placeholder={graph.nodes.length > 0
          ? 'Describe a workflow or edit this graph…'
          : 'Describe your protein design workflow…'}
        disabled={disabled || loading}
        oninput={resizeTextarea}
        onkeydown={onTextareaKeydown}
        aria-label="Describe your protein design workflow"
      ></textarea>

      {#if showCharCount}
        <span class="char-hint">{prompt.length}/320</span>
      {/if}

      {#if loading}
        <span class="loading-pill">Drafting…</span>
        <button
          type="button"
          class="btn-ghost compact"
          onclick={() => {
            abortPending();
            focusInput();
          }}
          title="Cancel drafting (Esc)"
        >
          Cancel
        </button>
      {:else}
        <button
          type="button"
          class="btn-send"
          disabled={disabled || trimmedPrompt.length === 0}
          onclick={() => void submitPrompt()}
          title="Draft workflow on canvas (Enter)"
          aria-label="Draft workflow"
        >
          <span class="send-label">Draft</span>
          <kbd>↵</kbd>
        </button>
        <button
          type="button"
          class="btn-icon-x"
          onclick={discardAndClose}
          title="Close (Esc)"
          aria-label="Close workflow prompt"
        >
          ×
        </button>
      {/if}
    </div>
  {/if}

  <!-- Quiet micro-disclosure sitting in the existing 28px bottom canvas margin -->
  <div class="micro-footer" bind:this={infoWrapEl}>
    {#if draftPreview}
      <span class="micro-text">
        Previewing on canvas · <kbd>↵</kbd> apply · <kbd>Esc</kbd> refine · <kbd>{modKeyLabel}Z</kbd> undoes after apply
      </span>
    {:else}
      <span class="micro-text desktop-disclosure">
        Prompt → Cloudflare Workers AI (gemma-4) · Design models run locally
      </span>
      <span class="micro-text mobile-disclosure">
        Cloudflare AI (gemma-4) · Design models run locally
      </span>
      <button
        type="button"
        class="info-btn"
        class:open={infoOpen}
        onclick={() => (infoOpen = !infoOpen)}
        aria-label="About external AI workflow drafting"
        aria-expanded={infoOpen}
        title="How workflow drafting works"
      >
        i
      </button>
      {#if infoOpen}
        <div class="info-tooltip" role="tooltip">
          <strong>External text-to-graph helper</strong>
          <p>
            Your short text prompt (and current block names, if editing) is sent to
            <strong>Cloudflare Workers AI</strong> (<code>gemma-4-26b-a4b-it</code>, free tier — no
            API key required) to draft the DAG.
          </p>
          <p>
            All protein generation, sequence design, folding, and model weights still run
            <strong>100% locally in your browser on WebGPU</strong>.
          </p>
        </div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .planner-bar-root {
    position: relative;
    display: flex;
    align-items: center;
    width: 100%;
    min-height: 30px;
  }

  /* ---- Floating strip ABOVE the dock ---- */
  .above-strip {
    position: absolute;
    left: 50%;
    bottom: calc(100% + 22px);
    transform: translateX(-50%);
    max-width: min(760px, calc(100vw - 24px));
    z-index: 3;
    animation: strip-float-in 240ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  @keyframes strip-float-in {
    from {
      opacity: 0;
      transform: translateX(-50%) translateY(6px);
    }
    to {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
  }

  .chip-row {
    display: flex;
    align-items: center;
    gap: 8px;
    overflow-x: auto;
    scrollbar-width: none;
    padding: 4px 4px;
    margin: -4px 0;
    white-space: nowrap;
  }
  .chip-row::-webkit-scrollbar {
    display: none;
  }

  .ex-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 13px;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-secondary);
    font-family: inherit;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    flex-shrink: 0;
    transition:
      background 0.14s ease,
      border-color 0.14s ease,
      color 0.14s ease,
      transform 0.14s ease;
  }
  .ex-pill:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    border-color: color-mix(in srgb, var(--accent) 55%, var(--border-strong));
    color: var(--text-primary);
    transform: translateY(-1px);
  }
  .ex-spark {
    color: var(--accent-bright);
    font-size: 10px;
    line-height: 1;
  }

  .err-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid var(--err-border);
    background: var(--err-bg);
    color: var(--err-text);
    font-size: 11px;
    line-height: 1.3;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
    white-space: normal;
  }
  .err-msg {
    max-width: 480px;
  }
  .strip-x {
    background: none;
    border: none;
    color: inherit;
    opacity: 0.75;
    cursor: pointer;
    font-size: 14px;
    line-height: 1;
    padding: 0 2px;
  }
  .strip-x:hover {
    opacity: 1;
  }

  .note-pill {
    display: flex;
    justify-content: center;
  }
  .note-toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--border-strong));
    background: color-mix(in srgb, var(--bg-surface) 94%, transparent);
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
    color: var(--text-secondary);
    font-family: inherit;
    font-size: 11px;
    cursor: pointer;
    max-width: min(560px, calc(100vw - 32px));
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.14);
  }
  .note-spark {
    color: var(--accent-bright);
    flex-shrink: 0;
  }
  .note-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .note-text.expanded {
    white-space: normal;
    text-align: left;
  }
  .note-more {
    font-size: 10px;
    font-weight: 600;
    color: var(--accent-bright);
    flex-shrink: 0;
  }

  /* ---- Typing Input Row ---- */
  .input-row {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 1px 4px 1px 10px;
    overflow: hidden;
    animation: row-morph-in 200ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  @keyframes row-morph-in {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
  .planner-bar-root.is-multiline .input-row {
    align-items: flex-end;
    padding-top: 4px;
    padding-bottom: 4px;
  }

  .bar-sparkle {
    color: var(--accent-bright);
    font-size: 15px;
    line-height: 1;
    flex-shrink: 0;
    align-self: center;
  }
  .bar-sparkle.spinning {
    animation: bar-pulse 1.1s ease-in-out infinite;
  }
  @keyframes bar-pulse {
    0%,
    100% {
      opacity: 0.45;
      transform: scale(0.9);
    }
    50% {
      opacity: 1;
      transform: scale(1.15);
    }
  }

  .bar-textarea {
    flex: 1;
    min-width: 0;
    resize: none;
    border: none;
    outline: none;
    background: transparent;
    color: var(--text-primary);
    font-family: inherit;
    font-size: 14px;
    line-height: 24px;
    height: 24px;
    padding: 0;
    margin: 0;
    max-height: 84px;
    overflow-y: auto;
  }
  .bar-textarea.is-empty {
    white-space: nowrap;
    overflow: hidden;
  }
  .bar-textarea::placeholder {
    color: var(--text-dim);
  }
  .bar-textarea:disabled {
    opacity: 0.65;
  }

  .char-hint {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
    flex-shrink: 0;
  }

  .loading-pill {
    font-size: 11px;
    font-weight: 600;
    color: var(--accent-bright);
    white-space: nowrap;
    flex-shrink: 0;
  }

  .btn-send {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 7px 14px;
    border-radius: 999px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: #fff;
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    flex-shrink: 0;
    transition: filter 0.14s ease, opacity 0.14s ease;
  }
  .btn-send:hover:not(:disabled) {
    filter: brightness(1.08);
  }
  .btn-send:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .btn-send kbd,
  .btn-apply kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    padding: 0 4px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.22);
    line-height: 1.4;
  }

  .btn-icon-x {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 999px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-dim);
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
    flex-shrink: 0;
    transition: background 0.12s ease, color 0.12s ease;
  }
  .btn-icon-x:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }

  /* ---- Review Bar Row (Live Ghost Preview Active) ---- */
  .review-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    width: 100%;
    padding: 2px 4px 2px 8px;
    animation: row-morph-in 220ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  .review-info {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    flex: 1;
  }
  .mode-pill {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 999px;
    background: var(--bg-pill);
    border: 1px solid color-mix(in srgb, var(--accent) 40%, var(--border));
    color: var(--accent-bright);
    white-space: nowrap;
    flex-shrink: 0;
  }
  .mode-pill.edit {
    background: var(--ok-bg);
    border-color: var(--ok-border);
    color: var(--ok-text);
  }
  .review-summary {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .review-ms {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-dim);
    flex-shrink: 0;
  }

  .review-actions {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-shrink: 0;
  }
  .btn-apply {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 7px 14px;
    border-radius: 999px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: #fff;
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: filter 0.14s ease;
  }
  .btn-apply:hover:not(:disabled) {
    filter: brightness(1.08);
  }
  .btn-ghost {
    padding: 7px 12px;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: transparent;
    color: var(--text-secondary);
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.12s ease, color 0.12s ease;
  }
  .btn-ghost.compact {
    padding: 4px 9px;
    font-size: 11px;
  }
  .btn-ghost:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }

  /* ---- Quiet Micro-Disclosure BELOW the dock ---- */
  .micro-footer {
    position: absolute;
    left: 50%;
    top: calc(100% + 22px);
    transform: translateX(-50%);
    z-index: 12;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 1px 6px;
    border-radius: 999px;
    background: transparent;
    white-space: nowrap;
    max-width: calc(100vw - 20px);
    pointer-events: auto;
    animation: footer-fade-in 240ms ease 60ms both;
  }
  @keyframes footer-fade-in {
    from {
      opacity: 0;
      transform: translateX(-50%) translateY(-3px);
    }
    to {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
  }
  .micro-text {
    font-size: 11.5px;
    color: var(--text-dim);
    line-height: 1.3;
  }
  .micro-text kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    padding: 0 3px;
    border-radius: 3px;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    color: var(--text-secondary);
  }
  .mobile-disclosure {
    display: none;
  }

  .info-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    font-weight: 700;
    cursor: pointer;
    padding: 0;
    flex-shrink: 0;
  }
  .info-btn:hover,
  .info-btn.open {
    border-color: var(--accent);
    color: var(--accent-bright);
  }

  .info-tooltip {
    position: absolute;
    bottom: calc(100% + 8px);
    right: 0;
    width: min(290px, calc(100vw - 28px));
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
    white-space: normal;
    z-index: 10;
  }
  .info-tooltip strong {
    color: var(--text-primary);
    font-weight: 600;
  }
  .info-tooltip p {
    margin: 5px 0 0;
  }
  .info-tooltip code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    padding: 0 3px;
    border-radius: 3px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
  }

  @media (max-width: 640px) {
    .desktop-disclosure,
    .review-ms,
    .send-label {
      display: none;
    }
    .mobile-disclosure {
      display: inline;
    }
    .btn-send {
      padding: 5px 9px;
    }
  }
  @media (max-width: 500px) {
    .review-row {
      flex-direction: column;
      align-items: stretch;
      gap: 6px;
      padding: 4px 6px;
    }
    .review-actions {
      justify-content: flex-end;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .bar-sparkle.spinning,
    .above-strip,
    .input-row,
    .review-row,
    .micro-footer {
      animation: none;
    }
    .ex-pill {
      transition: none;
    }
  }
</style>
