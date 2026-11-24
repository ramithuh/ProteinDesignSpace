<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { MAX_FEEDBACK_MESSAGE_CHARS, type FeedbackCategory } from '$lib/og/feedback-api';

  interface Props {
    category: FeedbackCategory;
    message: string;
    contact: string;
    includeContext: boolean;
    includeWorkflowUrl: boolean;
    includeErrorSnippet: boolean;
    blocksSummary: string;
    presetOrGraphId: string;
    webgpuSummary: string;
    computeMode: 'local' | 'remote';
    theme: 'dark' | 'light';
    errorText: string | null;
    modKeyLabel?: string;
    getShareableWorkflowUrl?: () => Promise<string | null>;
    buildGitHubIssueUrl: (opts: {
      category: FeedbackCategory;
      message: string;
      errorSnippet?: string | null;
    }) => string;
    onCategoryChange: (c: FeedbackCategory) => void;
    onMessageChange: (m: string) => void;
    onContactChange: (c: string) => void;
    onIncludeContextChange: (v: boolean) => void;
    onIncludeWorkflowUrlChange: (v: boolean) => void;
    onIncludeErrorSnippetChange: (v: boolean) => void;
    onSubmitted?: () => void;
    onClose: () => void;
  }

  let {
    category,
    message,
    contact,
    includeContext,
    includeWorkflowUrl,
    includeErrorSnippet,
    blocksSummary,
    presetOrGraphId,
    webgpuSummary,
    computeMode,
    theme,
    errorText,
    modKeyLabel = undefined,
    getShareableWorkflowUrl = undefined,
    buildGitHubIssueUrl,
    onCategoryChange,
    onMessageChange,
    onContactChange,
    onIncludeContextChange,
    onIncludeWorkflowUrlChange,
    onIncludeErrorSnippetChange,
    onSubmitted = undefined,
    onClose,
  }: Props = $props();

  let popEl = $state<HTMLDivElement | null>(null);
  let textareaEl = $state<HTMLTextAreaElement | null>(null);
  let honeypot = $state('');
  let submitting = $state(false);
  let submitted = $state(false);
  let submitError = $state<string | null>(null);
  let fallbackMode = $state(false);
  let copiedFallback = $state(false);
  let nudgeX = $state(0);
  let detectedModKey = $state('Ctrl+');

  const effectiveModKey = $derived(modKeyLabel ?? detectedModKey);

  const CATEGORIES: Array<{ id: FeedbackCategory; label: string; hint: string }> = [
    {
      id: 'idea',
      label: 'Idea',
      hint: 'Missing a model, benchmark, or workflow feature? Tell us what would help…',
    },
    {
      id: 'bug',
      label: 'Bug',
      hint: 'What broke or felt confusing? Steps or PDB ID help if relevant…',
    },
    {
      id: 'general',
      label: 'Note',
      hint: 'Thoughts, questions, or how you are using denovo.run…',
    },
  ];

  const activePlaceholder = $derived(
    CATEGORIES.find((c) => c.id === category)?.hint ?? CATEGORIES[0].hint,
  );

  const trimmedMessage = $derived(message.trim());
  const canSubmit = $derived(!submitting && trimmedMessage.length >= 3);

  const activeErrorSnippet = $derived(
    errorText && includeErrorSnippet ? errorText.slice(0, 600) : null,
  );

  const githubFallbackUrl = $derived(
    buildGitHubIssueUrl({
      category,
      message: trimmedMessage,
      errorSnippet: activeErrorSnippet,
    }),
  );

  const contextPreviewLabel = $derived.by(() => {
    const parts: string[] = [];
    if (blocksSummary) parts.push(blocksSummary);
    else parts.push('empty canvas');
    if (webgpuSummary) parts.push(webgpuSummary);
    return parts.join(' · ');
  });

  function updateViewportClamping() {
    if (!popEl || typeof window === 'undefined') return;
    nudgeX = 0;
    const rect = popEl.getBoundingClientRect();
    const margin = 12;
    if (rect.left < margin) {
      nudgeX = Math.round(margin - rect.left);
    } else if (rect.right > window.innerWidth - margin) {
      nudgeX = Math.round(window.innerWidth - margin - rect.right);
    }
  }

  onMount(() => {
    const plat = navigator.platform || navigator.userAgent || '';
    detectedModKey = /Mac|iPhone|iPad|iPod/i.test(plat) ? '⌘' : 'Ctrl+';
    void tick().then(() => {
      updateViewportClamping();
      textareaEl?.focus();
    });
  });

  function handleFormKeydown(ev: KeyboardEvent) {
    if ((ev.metaKey || ev.ctrlKey) && ev.key === 'Enter') {
      ev.preventDefault();
      if (canSubmit) {
        void submitFeedback();
      }
    }
  }

  async function submitFeedback(ev?: SubmitEvent) {
    ev?.preventDefault();
    if (!canSubmit) return;

    submitting = true;
    submitError = null;
    fallbackMode = false;

    try {
      let workflowUrl: string | undefined;
      if (includeWorkflowUrl && getShareableWorkflowUrl) {
        const built = await getShareableWorkflowUrl();
        if (built) workflowUrl = built;
      }

      const contextPayload =
        includeContext || activeErrorSnippet
          ? {
              ...(includeContext
                ? {
                    blocks: blocksSummary || 'Empty canvas',
                    presetOrGraphId,
                    webgpu: webgpuSummary,
                    computeMode,
                    theme,
                    viewport:
                      typeof window !== 'undefined'
                        ? `${window.innerWidth}x${window.innerHeight}`
                        : undefined,
                    userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : undefined,
                  }
                : {}),
              ...(activeErrorSnippet ? { errorSnippet: activeErrorSnippet } : {}),
            }
          : undefined;

      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          category,
          message: trimmedMessage,
          ...(contact.trim() ? { contact: contact.trim() } : {}),
          ...(workflowUrl ? { workflowUrl } : {}),
          ...(contextPayload ? { context: contextPayload } : {}),
          ...(honeypot ? { website: honeypot } : {}),
        }),
      });

      let data: { ok?: boolean; code?: string; error?: string } | null = null;
      try {
        data = (await res.json()) as { ok?: boolean; code?: string; error?: string };
      } catch {
        data = null;
      }

      if (res.ok && data?.ok) {
        submitted = true;
        onSubmitted?.();
        return;
      }

      if (res.status === 404 || res.status === 503 || data?.code === 'KV_NOT_BOUND') {
        fallbackMode = true;
        submitError =
          'Direct feedback inbox is not active on this host — send via GitHub below (your note is pre-filled) or copy it.';
        return;
      }

      submitError = data?.error || `Could not send feedback (HTTP ${res.status}).`;
    } catch {
      fallbackMode = true;
      submitError = 'Network offline — you can copy your note or open a pre-filled GitHub issue below.';
    } finally {
      submitting = false;
    }
  }

  async function copyDraftText() {
    try {
      await navigator.clipboard?.writeText(trimmedMessage);
      copiedFallback = true;
      setTimeout(() => {
        copiedFallback = false;
      }, 2000);
    } catch {
      /* ignore clipboard denial */
    }
  }

  function sendAnother() {
    submitted = false;
    submitError = null;
    fallbackMode = false;
    void tick().then(() => textareaEl?.focus());
  }
</script>

<div
  bind:this={popEl}
  class="hdr-popover feedback-pop"
  style:transform={nudgeX !== 0 ? `translateX(${nudgeX}px)` : undefined}
  role="dialog"
  aria-label="Send feedback"
>
  {#if submitted}
    <div class="fb-done">
      <div class="fb-done-badge" aria-hidden="true">✓</div>
      <div class="fb-done-title">Thanks — feedback saved</div>
      <p class="fb-done-sub">
        Every note goes straight to the <strong>denovo.run</strong> inbox and helps prioritize new WebGPU models and fixes.
      </p>
      <div class="fb-done-actions">
        <button type="button" class="fb-btn-ghost" onclick={sendAnother}>
          Send another
        </button>
        <button type="button" class="fb-btn-primary" onclick={onClose}>
          Done
        </button>
      </div>
    </div>
  {:else}
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <form class="fb-form" onsubmit={submitFeedback} onkeydown={handleFormKeydown}>
      <div class="fb-head">
        <span class="fb-title">Send feedback</span>
        <div class="fb-cats" role="radiogroup" aria-label="Feedback type">
          {#each CATEGORIES as item (item.id)}
            <button
              type="button"
              role="radio"
              aria-checked={category === item.id}
              class="fb-cat-pill"
              class:active={category === item.id}
              onclick={() => onCategoryChange(item.id)}
            >
              {item.label}
            </button>
          {/each}
        </div>
      </div>

      <div class="fb-textarea-wrap">
        <textarea
          bind:this={textareaEl}
          class="fb-textarea"
          rows="4"
          maxlength={MAX_FEEDBACK_MESSAGE_CHARS}
          placeholder={activePlaceholder}
          aria-label="Feedback message"
          value={message}
          oninput={(e) => {
            submitError = null;
            onMessageChange(e.currentTarget.value);
          }}
        ></textarea>
        {#if message.length >= 1500}
          <span class="fb-char-count">{message.length}/{MAX_FEEDBACK_MESSAGE_CHARS}</span>
        {/if}
      </div>

      <input
        type="text"
        class="fb-contact"
        maxlength="120"
        placeholder="Email, GitHub, or @handle (optional — if you want a reply)"
        aria-label="Optional email or handle for replies"
        value={contact}
        oninput={(e) => onContactChange(e.currentTarget.value)}
      />

      <!-- Honeypot field (hidden from assistive tech & keyboard) -->
      <input
        type="text"
        name="website"
        class="fb-hp"
        tabindex="-1"
        autocomplete="off"
        aria-hidden="true"
        bind:value={honeypot}
      />

      <div class="fb-opts">
        {#if errorText}
          <label class="fb-check fb-check-err">
            <input
              type="checkbox"
              checked={includeErrorSnippet}
              onchange={(e) => onIncludeErrorSnippetChange(e.currentTarget.checked)}
            />
            <span class="fb-check-body">
              <span class="fb-check-lbl">Attach pipeline error message</span>
              <span class="fb-check-meta fb-err-preview">{errorText.split('\n')[0]}</span>
            </span>
          </label>
        {/if}

        <label class="fb-check">
          <input
            type="checkbox"
            checked={includeContext}
            onchange={(e) => onIncludeContextChange(e.currentTarget.checked)}
          />
          <span class="fb-check-body">
            <span class="fb-check-lbl">Include block names &amp; WebGPU summary</span>
            <span class="fb-check-meta">No sequences or PDB coordinates · {contextPreviewLabel}</span>
          </span>
        </label>

        {#if getShareableWorkflowUrl}
          <label class="fb-check">
            <input
              type="checkbox"
              checked={includeWorkflowUrl}
              onchange={(e) => onIncludeWorkflowUrlChange(e.currentTarget.checked)}
            />
            <span class="fb-check-body">
              <span class="fb-check-lbl">Attach shareable workflow link (<code>?g=…</code>)</span>
            </span>
          </label>
        {/if}
      </div>

      {#if submitError}
        <div class="fb-alert" class:is-fallback={fallbackMode} role="alert">
          <span>{submitError}</span>
          {#if fallbackMode}
            <div class="fb-fallback-row">
              <a
                class="fb-fallback-gh"
                href={githubFallbackUrl}
                target="_blank"
                rel="noopener noreferrer"
                onclick={onClose}
              >
                Open pre-filled GitHub Issue ↗
              </a>
              <button type="button" class="fb-btn-ghost fb-btn-xs" onclick={copyDraftText}>
                {copiedFallback ? '✓ Copied' : 'Copy text'}
              </button>
            </div>
          {/if}
        </div>
      {/if}

      <div class="fb-foot">
        <a
          class="fb-gh-link"
          href={githubFallbackUrl}
          target="_blank"
          rel="noopener noreferrer"
          title="Prefer a public GitHub issue thread? Opens pre-filled with your note."
          onclick={onClose}
        >
          GitHub Issue ↗
        </a>

        <div class="fb-foot-right">
          <button
            type="submit"
            class="fb-btn-primary"
            disabled={!canSubmit}
            title="Send feedback ({effectiveModKey}↵)"
          >
            {#if submitting}
              Sending…
            {:else}
              Send <kbd class="fb-kbd">{effectiveModKey}↵</kbd>
            {/if}
          </button>
        </div>
      </div>
    </form>
  {/if}
</div>

<style>
  .feedback-pop {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: 65;
    width: min(348px, calc(100vw - 24px));
    max-width: calc(100vw - 24px);
    box-sizing: border-box;
    padding: 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 9px;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.38), var(--node-shadow);
    color: var(--text-body);
  }

  .fb-form {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin: 0;
  }

  .fb-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .fb-title {
    font-size: 12.5px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.1px;
  }

  .fb-cats {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    padding: 2px;
    border-radius: 7px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
  }

  .fb-cat-pill {
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
  }

  .fb-cat-pill:hover:not(.active) {
    color: var(--text-primary);
    background: var(--bg-surface-hover);
  }

  .fb-cat-pill.active {
    background: var(--bg-pill);
    color: var(--accent-bright);
    border-color: var(--accent);
    font-weight: 600;
  }

  .fb-textarea-wrap {
    position: relative;
    display: flex;
  }

  .fb-textarea {
    width: 100%;
    min-height: 84px;
    max-height: 200px;
    resize: vertical;
    box-sizing: border-box;
    padding: 8px 9px;
    font-family: inherit;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-primary);
    background: var(--bg-input);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    outline: none;
    transition: border-color 0.12s ease, box-shadow 0.12s ease;
  }

  .fb-textarea::placeholder,
  .fb-contact::placeholder {
    color: var(--text-dim);
  }

  .fb-textarea:focus,
  .fb-contact:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.14);
  }

  .fb-char-count {
    position: absolute;
    right: 7px;
    bottom: 5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    color: var(--text-dim);
    background: var(--bg-surface);
    padding: 1px 4px;
    border-radius: 4px;
    pointer-events: none;
  }

  .fb-contact {
    width: 100%;
    height: 29px;
    box-sizing: border-box;
    padding: 0 9px;
    font-family: inherit;
    font-size: 11.5px;
    color: var(--text-primary);
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 6px;
    outline: none;
    transition: border-color 0.12s ease, box-shadow 0.12s ease;
  }

  .fb-hp {
    position: absolute;
    left: -9999px;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
  }

  .fb-opts {
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 6px 8px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
  }

  .fb-check {
    display: flex;
    align-items: flex-start;
    gap: 7px;
    cursor: pointer;
    user-select: none;
  }

  .fb-check input[type='checkbox'] {
    margin: 2px 0 0;
    accent-color: var(--accent);
    cursor: pointer;
    flex-shrink: 0;
  }

  .fb-check-body {
    display: flex;
    flex-direction: column;
    gap: 1px;
    min-width: 0;
  }

  .fb-check-lbl {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    line-height: 1.3;
  }

  .fb-check-lbl code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--accent-bright);
  }

  .fb-check-meta {
    font-size: 10px;
    color: var(--text-dim);
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .fb-check-err .fb-check-lbl {
    color: var(--err-strong);
  }

  .fb-err-preview {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-muted);
  }

  .fb-alert {
    font-size: 11px;
    line-height: 1.4;
    padding: 7px 8px;
    border-radius: 6px;
    color: var(--err-text);
    background: var(--err-bg);
    border: 1px solid var(--err-border);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .fb-alert.is-fallback {
    color: var(--warn-text);
    background: var(--warn-bg);
    border-color: var(--warn-border);
  }

  .fb-fallback-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }

  .fb-fallback-gh {
    font-size: 11px;
    font-weight: 600;
    color: var(--accent-bright);
    text-decoration: none;
  }

  .fb-fallback-gh:hover {
    text-decoration: underline;
  }

  .fb-foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding-top: 2px;
  }

  .fb-gh-link {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    text-decoration: none;
    white-space: nowrap;
  }

  .fb-gh-link:hover {
    color: var(--accent-bright);
    text-decoration: underline;
  }

  .fb-foot-right {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .fb-btn-primary {
    height: 29px;
    padding: 0 11px;
    border-radius: 6px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: #ffffff;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
    transition: filter 0.12s ease, opacity 0.12s ease;
  }

  .fb-btn-primary:hover:not(:disabled) {
    filter: brightness(1.08);
  }

  .fb-btn-primary:disabled {
    opacity: 0.48;
    cursor: not-allowed;
  }

  .fb-kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 500;
    opacity: 0.86;
    background: rgba(255, 255, 255, 0.16);
    padding: 1px 4px;
    border-radius: 3px;
  }

  .fb-btn-ghost {
    height: 28px;
    padding: 0 10px;
    border-radius: 6px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
    color: var(--text-body);
    font-size: 11.5px;
    font-weight: 500;
    cursor: pointer;
  }

  .fb-btn-ghost:hover {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }

  .fb-btn-xs {
    height: 24px;
    padding: 0 8px;
    font-size: 10.5px;
  }

  .fb-done {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 7px;
    padding: 4px 2px 2px;
  }

  .fb-done-badge {
    width: 24px;
    height: 24px;
    border-radius: 999px;
    background: var(--ok-bg);
    border: 1px solid var(--ok-border);
    color: var(--ok-text);
    font-size: 13px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .fb-done-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .fb-done-sub {
    margin: 0;
    font-size: 11.5px;
    line-height: 1.45;
    color: var(--text-muted);
  }

  .fb-done-actions {
    width: 100%;
    display: flex;
    justify-content: flex-end;
    gap: 6px;
    padding-top: 4px;
  }
</style>
