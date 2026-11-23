<script lang="ts">
  import { onMount, tick } from 'svelte';
  import {
    compileLocalStarterPlan,
    LOCAL_EXAMPLE_PLANS,
    PLANNER_MIN_PROMPT_LENGTH,
    PlannerHttpError,
    requestWorkflowPlan,
    UnsupportedWorkflowError,
    type LocalStarterExample,
    type PlannedWorkflowResult,
  } from '$lib/engine/workflow-planner';
  import type { WorkflowGraph } from '$lib/engine/types';

  interface Props {
    graph: WorkflowGraph;
    disabled?: boolean;
    modKeyLabel?: string;
    draftPreview: PlannedWorkflowResult | null;
    onDraftChange: (draft: PlannedWorkflowResult | null) => void;
    onPromptStateChange?: (hasText: boolean, loading: boolean) => void;
    onApply: (graph: WorkflowGraph, summary: string, isAiDraft?: boolean) => void;
    onOpenCatalog?: () => void;
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
    onOpenCatalog = undefined,
    onClose,
  }: Props = $props();

  let examples = $derived.by(() => {
    const hasNodes = graph.nodes.length > 0;
    return LOCAL_EXAMPLE_PLANS.filter((ex) => !ex.requiresExisting || hasNodes).slice(0, 3);
  });

  interface UnsupportedNotice {
    reason: string;
    unsupported: string[];
    suggestions: string[];
  }

  let prompt = $state('');
  let loading = $state(false);
  let errorMsg = $state<string | null>(null);
  let errorShowCatalog = $state(false);
  let cooldownSec = $state(0);
  let unsupportedNotice = $state<UnsupportedNotice | null>(null);
  let infoOpen = $state(false);
  let notesExpanded = $state(false);
  let isMultiline = $state(false);
  let textareaEl: HTMLTextAreaElement | null = $state(null);
  let infoWrapEl: HTMLDivElement | null = $state(null);
  let abortCtrl: AbortController | null = null;
  let cooldownTimer: ReturnType<typeof setInterval> | null = null;

  let trimmedPrompt = $derived(prompt.trim());
  let canSubmit = $derived(
    !disabled &&
      !loading &&
      cooldownSec === 0 &&
      trimmedPrompt.length >= PLANNER_MIN_PROMPT_LENGTH,
  );
  let showExamples = $derived(
    trimmedPrompt.length === 0 &&
      !loading &&
      !draftPreview &&
      !errorMsg &&
      !unsupportedNotice &&
      !infoOpen,
  );
  let showCharCount = $derived(prompt.length > 220 && !draftPreview);
  let isAiDraft = $derived(Boolean(draftPreview && draftPreview.model !== 'local-preset'));

  $effect(() => {
    onPromptStateChange?.(trimmedPrompt.length > 0, loading);
  });

  function clearCooldown() {
    if (cooldownTimer) {
      clearInterval(cooldownTimer);
      cooldownTimer = null;
    }
    cooldownSec = 0;
  }

  function startCooldown(seconds: number) {
    clearCooldown();
    const clamped = Math.max(1, Math.min(120, Math.ceil(seconds)));
    cooldownSec = clamped;
    cooldownTimer = setInterval(() => {
      if (cooldownSec <= 1) {
        clearCooldown();
      } else {
        cooldownSec -= 1;
      }
    }, 1000);
  }

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

  function handleTextareaInput() {
    resizeTextarea();
    if (errorMsg) {
      errorMsg = null;
      errorShowCatalog = false;
    }
    if (unsupportedNotice) {
      unsupportedNotice = null;
    }
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
    if (unsupportedNotice || errorMsg) {
      unsupportedNotice = null;
      errorMsg = null;
      focusInput();
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
      clearCooldown();
      window.removeEventListener('pointerdown', onDocPointerDown, true);
    };
  });

  async function applyLocalExample(example: LocalStarterExample) {
    if (loading || disabled) return;
    prompt = example.prompt;
    await tick();
    resizeTextarea();
    errorMsg = null;
    errorShowCatalog = false;
    unsupportedNotice = null;
    infoOpen = false;
    notesExpanded = false;

    try {
      const snapshotGraph =
        graph.nodes.length > 0 ? ($state.snapshot(graph) as WorkflowGraph) : undefined;
      const planned = compileLocalStarterPlan(example.plan, snapshotGraph);
      onDraftChange(planned);
    } catch (err) {
      errorMsg = err instanceof Error ? err.message : 'Failed to compile starter workflow.';
      errorShowCatalog = Boolean(onOpenCatalog);
      focusInput();
    }
  }

  async function submitPrompt(overrideText?: string) {
    const text = (overrideText ?? prompt).trim();
    if (!text || loading || disabled || cooldownSec > 0) return;
    if (text.length < PLANNER_MIN_PROMPT_LENGTH) {
      errorMsg = `Please enter at least ${PLANNER_MIN_PROMPT_LENGTH} characters to describe a workflow.`;
      errorShowCatalog = false;
      return;
    }
    if (overrideText !== undefined) {
      prompt = overrideText;
      await tick();
      resizeTextarea();
    }

    abortCtrl?.abort();
    abortCtrl = new AbortController();
    loading = true;
    errorMsg = null;
    errorShowCatalog = false;
    unsupportedNotice = null;
    infoOpen = false;
    notesExpanded = false;
    onDraftChange(null);

    try {
      const snapshotGraph =
        graph.nodes.length > 0 ? ($state.snapshot(graph) as WorkflowGraph) : undefined;
      const planned = await requestWorkflowPlan(text, {
        currentGraph: snapshotGraph,
        signal: abortCtrl.signal,
      });
      onDraftChange(planned);
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') return;
      if (err instanceof UnsupportedWorkflowError) {
        unsupportedNotice = {
          reason: err.message,
          unsupported: err.unsupported,
          suggestions: err.suggestions,
        };
      } else if (err instanceof PlannerHttpError) {
        errorMsg = err.message;
        errorShowCatalog =
          Boolean(onOpenCatalog) &&
          (err.status === 429 ||
            err.status === 403 ||
            err.status === 503 ||
            err.status === 504 ||
            err.status === 0 ||
            err.code === 'quota_exhausted');
        if (err.status === 429 && err.code !== 'quota_exhausted' && err.retryAfterSec) {
          startCooldown(err.retryAfterSec);
        }
      } else {
        errorMsg = err instanceof Error ? err.message : 'Failed to plan workflow.';
        errorShowCatalog = Boolean(onOpenCatalog);
      }
      focusInput();
    } finally {
      loading = false;
      abortCtrl = null;
    }
  }

  function applyDraft() {
    if (!draftPreview || disabled) return;
    const nextGraph = $state.snapshot(draftPreview.graph) as WorkflowGraph;
    const summary = draftPreview.plan.summary;
    const aiDraft = draftPreview.model !== 'local-preset';
    onDraftChange(null);
    onApply(nextGraph, summary, aiDraft);
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
  <!-- Floating strip ABOVE the dock bar: error OR unsupported notice OR starter chips OR partial-match/compiler notes -->
  {#if errorMsg}
    <div class="above-strip err-pill" role="alert">
      <span class="err-msg">{errorMsg}</span>
      {#if errorShowCatalog && onOpenCatalog}
        <button
          type="button"
          class="err-fallback-btn"
          onclick={() => {
            errorMsg = null;
            errorShowCatalog = false;
            onOpenCatalog();
          }}
        >
          + Add Block (/)
        </button>
      {/if}
      <button
        type="button"
        class="strip-x"
        onclick={() => {
          errorMsg = null;
          errorShowCatalog = false;
          focusInput();
        }}
        aria-label="Dismiss error"
      >
        ×
      </button>
    </div>
  {:else if unsupportedNotice}
    <div class="above-strip warn-card" role="status">
      <div class="warn-head">
        <span class="warn-icon" aria-hidden="true">⚠</span>
        <span class="warn-title">
          {#if unsupportedNotice.unsupported.length > 0}
            Not in denovo.run: <strong>{unsupportedNotice.unsupported.join(', ')}</strong>
          {:else}
            Not supported in denovo.run yet
          {/if}
        </span>
        <button
          type="button"
          class="strip-x"
          onclick={() => {
            unsupportedNotice = null;
            focusInput();
          }}
          aria-label="Dismiss notice"
        >
          ×
        </button>
      </div>
      <div class="warn-reason">{unsupportedNotice.reason}</div>
      {#if unsupportedNotice.suggestions.length > 0}
        <div class="warn-suggestions">
          <span class="warn-try">Available instead:</span>
          {unsupportedNotice.suggestions.join(' · ')}
        </div>
      {/if}
    </div>
  {:else if showExamples}
    <div class="above-strip chip-row" role="list" aria-label="Example workflow prompts">
      {#each examples as ex (ex.id)}
        <button
          type="button"
          class="ex-pill"
          title={`${ex.prompt} (instant local preview)`}
          onclick={() => void applyLocalExample(ex)}
          {disabled}
        >
          <span class="ex-spark" aria-hidden="true">✦</span>
          <span>{ex.label}</span>
        </button>
      {/each}
    </div>
  {:else if draftPreview}
    <div class="above-strip note-stack">
      <div class="draft-notice" class:is-ai={isAiDraft} role="status">
        {#if isAiDraft}
          <span class="draft-badge is-ai" aria-hidden="true">⚠ AI draft</span>
          <span class="draft-copy">
            Review parameters, PDB/chain IDs &amp; wiring before running.
          </span>
        {:else}
          <span class="draft-badge" aria-hidden="true">✦ Starter</span>
          <span class="draft-copy">
            Review parameters, PDB/chain IDs &amp; wiring before running.
          </span>
        {/if}
      </div>
      {#if draftPreview.unsupported.length > 0}
        <div
          class="warn-pill"
          role="status"
          title={`Unavailable models omitted: ${draftPreview.unsupported.join(', ')}`}
        >
          <span class="warn-icon" aria-hidden="true">⚠</span>
          <span>
            Not in denovo.run: <strong>{draftPreview.unsupported.join(', ')}</strong> — drafted with
            available models
          </span>
        </div>
      {/if}
      {#if draftPreview.notes.length > 0}
        <div class="note-pill">
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
        {#if draftPreview.unsupported.length > 0}
          <span
            class="partial-pill"
            title={`Omitted unavailable models: ${draftPreview.unsupported.join(', ')}`}
          >
            Partial
          </span>
        {/if}
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
          title="Apply drafted workflow to canvas (Enter)"
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
        oninput={handleTextareaInput}
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
          disabled={!canSubmit}
          onclick={() => void submitPrompt()}
          title={cooldownSec > 0
            ? `Rate limit cooldown (${cooldownSec}s remaining)`
            : trimmedPrompt.length > 0 && trimmedPrompt.length < PLANNER_MIN_PROMPT_LENGTH
              ? `Enter at least ${PLANNER_MIN_PROMPT_LENGTH} characters`
              : 'Draft workflow on canvas (Enter)'}
          aria-label="Draft workflow"
        >
          {#if cooldownSec > 0}
            <span class="send-label">Wait {cooldownSec}s</span>
          {:else}
            <span class="send-label">Draft</span>
            <kbd>↵</kbd>
          {/if}
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
          <p>
            LLM drafts can miss residue ranges, chain IDs, or model settings — always double-check
            each block before running.
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
  .err-fallback-btn {
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, var(--err-border) 80%, transparent);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-family: inherit;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .err-fallback-btn:hover {
    border-color: var(--accent);
    color: var(--accent-bright);
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

  .note-stack {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  .draft-notice {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 11px 5px 6px;
    border-radius: 10px;
    border: 1px solid rgba(251, 146, 60, 0.42);
    background: color-mix(in srgb, #f97316 12%, color-mix(in srgb, var(--bg-surface) 95%, transparent));
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
    color: var(--text-body);
    font-size: 11.5px;
    line-height: 1.35;
    box-shadow: 0 4px 14px rgba(234, 88, 12, 0.14);
    max-width: min(560px, calc(100vw - 28px));
    text-align: left;
  }
  .draft-notice.is-ai {
    border-color: rgba(249, 115, 22, 0.62);
    background: color-mix(in srgb, #ea580c 16%, color-mix(in srgb, var(--bg-surface) 95%, transparent));
  }
  .draft-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 6px;
    background: rgba(249, 115, 22, 0.18);
    border: 1px solid rgba(251, 146, 60, 0.36);
    color: #fdba74;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .draft-badge.is-ai {
    background: rgba(234, 88, 12, 0.24);
    border-color: rgba(251, 146, 60, 0.5);
    color: #fed7aa;
  }
  .draft-copy {
    color: var(--text-body);
  }
  :global([data-theme='light']) .draft-notice {
    border-color: #fdba74;
    background: rgba(255, 247, 237, 0.94);
    box-shadow: 0 4px 14px rgba(234, 88, 12, 0.1);
  }
  :global([data-theme='light']) .draft-notice.is-ai {
    border-color: #fb923c;
    background: rgba(255, 237, 213, 0.96);
  }
  :global([data-theme='light']) .draft-badge {
    background: #ffedd5;
    border-color: #fdba74;
    color: #c2410c;
  }
  :global([data-theme='light']) .draft-badge.is-ai {
    background: #fed7aa;
    border-color: #fb923c;
    color: #9a3412;
  }
  :global([data-theme='light']) .draft-copy {
    color: #7c2d12;
  }
  .warn-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 12px;
    border-radius: 10px;
    border: 1px solid var(--warn-border);
    background: var(--warn-bg);
    color: var(--warn-text);
    font-size: 11.5px;
    line-height: 1.35;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    white-space: normal;
    text-align: left;
    max-width: min(560px, calc(100vw - 28px));
  }
  .warn-card {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 9px 13px;
    border-radius: 12px;
    border: 1px solid var(--warn-border);
    background: var(--warn-bg);
    color: var(--warn-text);
    font-size: 11.5px;
    line-height: 1.4;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    width: min(540px, calc(100vw - 28px));
    white-space: normal;
  }
  .warn-head {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .warn-title {
    flex: 1;
    font-weight: 600;
  }
  .warn-icon {
    flex-shrink: 0;
    font-size: 12px;
  }
  .warn-reason {
    opacity: 0.92;
  }
  .warn-suggestions {
    font-size: 11px;
    opacity: 0.88;
    border-top: 1px solid color-mix(in srgb, var(--warn-border) 45%, transparent);
    padding-top: 4px;
    margin-top: 1px;
  }
  .warn-try {
    font-weight: 600;
  }
  .partial-pill {
    font-size: 10.5px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 999px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
    white-space: nowrap;
    flex-shrink: 0;
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
