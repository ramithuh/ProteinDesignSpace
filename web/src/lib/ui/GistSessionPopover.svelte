<script lang="ts">
  import { onMount, tick } from 'svelte';
  import {
    GIST_SESSION_FILENAME,
    buildGistShareUrl,
    extractGistQueryParam,
    parseGistReference,
    type SerializedGistSession,
  } from '$lib/engine/gist-session';

  interface Props {
    snapshot: SerializedGistSession | null;
    snapshotError?: string | null;
    isRunning: boolean;
    onLoadGist: (refInput: string) => Promise<void>;
    onNotify?: (msg: string, tone?: 'ok' | 'error') => void;
    onClose: () => void;
  }

  let {
    snapshot,
    snapshotError = null,
    isRunning,
    onLoadGist,
    onNotify = undefined,
    onClose,
  }: Props = $props();

  let popEl = $state<HTMLDivElement | null>(null);
  let nudgeX = $state(0);
  let activeTab = $state<'save' | 'load'>('save');

  // Save view state
  let copiedJson = $state(false);
  let copyJsonError = $state<string | null>(null);
  let gistUrlInput = $state('');
  let copiedShareLink = $state(false);
  let copyLinkError = $state<string | null>(null);

  // Load view state
  let loadInput = $state('');
  let loadBusy = $state(false);
  let loadError = $state<string | null>(null);

  const parsedShareRef = $derived(parseGistReference(gistUrlInput));
  const readyShareUrl = $derived.by(() => {
    if (!parsedShareRef) return '';
    const origin =
      typeof window !== 'undefined' ? window.location.origin : 'https://denovo.run';
    const pathname = typeof window !== 'undefined' ? window.location.pathname : '/';
    return buildGistShareUrl(origin, parsedShareRef, pathname);
  });

  const resolvedLoadRefInput = $derived(extractGistQueryParam(loadInput));

  function formatBytesLabel(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }

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
    void tick().then(updateViewportClamping);
  });

  async function copySessionJson() {
    if (!snapshot) return;
    copyJsonError = null;
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error('Clipboard API unavailable');
      }
      await navigator.clipboard.writeText(snapshot.json);
      copiedJson = true;
      onNotify?.(`Copied ${GIST_SESSION_FILENAME} (${formatBytesLabel(snapshot.bytes)})`, 'ok');
      setTimeout(() => {
        copiedJson = false;
      }, 2200);
    } catch {
      copyJsonError = 'Clipboard blocked — use Download .json instead.';
      onNotify?.('Clipboard blocked — download .json instead', 'error');
    }
  }

  function downloadSessionJson() {
    if (!snapshot) return;
    const blob = new Blob([snapshot.json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = GIST_SESSION_FILENAME;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    onNotify?.(`Downloaded ${GIST_SESSION_FILENAME}`, 'ok');
  }

  async function copyGeneratedShareLink() {
    if (!readyShareUrl) return;
    copyLinkError = null;
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error('Clipboard API unavailable');
      }
      await navigator.clipboard.writeText(readyShareUrl);
      copiedShareLink = true;
      onNotify?.('Copied shareable ?gist= link', 'ok');
      setTimeout(() => {
        copiedShareLink = false;
      }, 2200);
    } catch {
      copyLinkError = 'Clipboard blocked — select and copy the URL below.';
    }
  }

  async function handleLoadSubmit(ev?: SubmitEvent) {
    ev?.preventDefault();
    if (loadBusy || isRunning) return;
    if (!resolvedLoadRefInput) {
      loadError = 'Paste a valid gist.github.com URL, ?gist= link, or Gist ID.';
      return;
    }
    loadBusy = true;
    loadError = null;
    try {
      await onLoadGist(resolvedLoadRefInput);
      onClose();
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loadBusy = false;
    }
  }
</script>

<div
  bind:this={popEl}
  class="hdr-popover gist-pop"
  style:transform={nudgeX !== 0 ? `translateX(${nudgeX}px)` : undefined}
  role="dialog"
  aria-label="Save or load results via GitHub Gist"
>
  <div class="gp-head">
    <div class="gp-tabs" role="tablist" aria-label="Gist session mode">
      <button
        type="button"
        role="tab"
        aria-selected={activeTab === 'save'}
        class="gp-tab"
        class:active={activeTab === 'save'}
        onclick={() => {
          activeTab = 'save';
          void tick().then(updateViewportClamping);
        }}
      >
        Save Results
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={activeTab === 'load'}
        class="gp-tab"
        class:active={activeTab === 'load'}
        onclick={() => {
          activeTab = 'load';
          void tick().then(updateViewportClamping);
        }}
      >
        Load from Gist
      </button>
    </div>
    <button
      type="button"
      class="gp-close"
      aria-label="Close save/load popover"
      onclick={onClose}
    >
      ×
    </button>
  </div>

  {#if activeTab === 'save'}
    {#if snapshotError}
      <p class="gp-warn">{snapshotError}</p>
    {:else if snapshot}
      <div class="gp-stats">
        {snapshot.nodeCount} node{snapshot.nodeCount === 1 ? '' : 's'} ·
        {snapshot.resultCount} cached result{snapshot.resultCount === 1 ? '' : 's'} ·
        {formatBytesLabel(snapshot.bytes)}
      </div>

      {#if snapshot.warnings.length > 0}
        <p class="gp-warn">{snapshot.warnings.join(' ')}</p>
      {/if}

      <p class="gp-note">
        GitHub Secret Gists are unlisted links — anyone with the link can view them.
      </p>

      <div class="gp-step">
        <div class="gp-step-lbl">
          <span class="gp-step-num">1</span>
          <span>Grab session snapshot</span>
        </div>
        <div class="gp-btn-row">
          <button
            type="button"
            class="gp-btn-primary"
            class:copied={copiedJson}
            onclick={() => void copySessionJson()}
          >
            {copiedJson ? '✓ Copied!' : 'Copy Session'}
          </button>
          <button
            type="button"
            class="gp-btn-ghost"
            onclick={downloadSessionJson}
          >
            Download .json
          </button>
        </div>
        {#if copyJsonError}
          <p class="gp-warn">{copyJsonError}</p>
        {/if}
      </div>

      <div class="gp-step">
        <div class="gp-step-lbl">
          <span class="gp-step-num">2</span>
          <span>Create a secret Gist</span>
        </div>
        <a
          class="gp-gh-link"
          href="https://gist.github.com/"
          target="_blank"
          rel="noopener noreferrer"
        >
          Open gist.github.com ↗
        </a>
        <p class="gp-hint">
          Name the file <code>{GIST_SESSION_FILENAME}</code>, paste, and click <strong>Create secret gist</strong>.
        </p>
      </div>

      <div class="gp-step">
        <label class="gp-step-lbl" for="gist-share-url-input">
          <span class="gp-step-num">3</span>
          <span>Paste your Gist URL to get a denovo.run link</span>
        </label>
        <input
          id="gist-share-url-input"
          type="text"
          class="gp-input"
          placeholder="https://gist.github.com/user/..."
          value={gistUrlInput}
          oninput={(e) => {
            gistUrlInput = e.currentTarget.value;
            copiedShareLink = false;
            copyLinkError = null;
          }}
        />
        {#if readyShareUrl}
          <div class="gp-url-row">
            <input
              type="text"
              class="gp-input gp-input-ro"
              readonly
              aria-label="Ready to share denovo.run Gist URL"
              value={readyShareUrl}
              onclick={(e) => e.currentTarget.select()}
            />
            <button
              type="button"
              class="gp-btn-primary"
              class:copied={copiedShareLink}
              onclick={() => void copyGeneratedShareLink()}
            >
              {copiedShareLink ? '✓ Copied!' : 'Copy Link'}
            </button>
          </div>
          {#if copyLinkError}
            <p class="gp-warn">{copyLinkError}</p>
          {/if}
        {:else if gistUrlInput.trim().length > 0}
          <p class="gp-warn">Enter a valid GitHub Gist URL or Gist ID (20+ hex chars).</p>
        {/if}
      </div>
    {:else}
      <p class="gp-hint">Preparing session snapshot…</p>
    {/if}
  {:else}
    <form class="gp-load-form" onsubmit={handleLoadSubmit}>
      <p class="gp-note">
        Hydrates workflow blocks and cached 3D/score results directly into memory without running GPU models.
      </p>
      <label class="gp-load-lbl" for="gist-load-input">
        GitHub Gist URL, <code>?gist=</code> link, or Gist ID
      </label>
      <input
        id="gist-load-input"
        type="text"
        class="gp-input"
        placeholder="https://gist.github.com/owner/id or owner/id"
        disabled={loadBusy}
        value={loadInput}
        oninput={(e) => {
          loadInput = e.currentTarget.value;
          loadError = null;
        }}
      />
      {#if isRunning}
        <p class="gp-warn">Cancel the active workflow run before loading a Gist session.</p>
      {/if}
      {#if loadError}
        <p class="gp-err" role="alert">{loadError}</p>
      {/if}
      <div class="gp-load-actions">
        <button
          type="submit"
          class="gp-btn-primary gp-btn-full"
          disabled={loadBusy || isRunning || !resolvedLoadRefInput}
        >
          {loadBusy ? 'Loading Gist…' : 'Load Results'}
        </button>
      </div>
    </form>
  {/if}
</div>

<style>
  .gist-pop {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: 65;
    width: min(356px, calc(100vw - 24px));
    max-width: calc(100vw - 24px);
    box-sizing: border-box;
    padding: 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 9px;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.38), var(--node-shadow);
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 9px;
  }

  .gp-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .gp-tabs {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    padding: 2px;
    border-radius: 7px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
  }

  .gp-tab {
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    padding: 3px 9px;
    border-radius: 5px;
    cursor: pointer;
    font-family: inherit;
    white-space: nowrap;
  }

  .gp-tab:hover:not(.active) {
    color: var(--text-primary);
    background: var(--bg-surface-hover);
  }

  .gp-tab.active {
    background: var(--bg-pill);
    color: var(--accent-bright);
    border-color: var(--accent);
    font-weight: 600;
  }

  .gp-close {
    width: 24px;
    height: 24px;
    border-radius: 5px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-muted);
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .gp-close:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
    border-color: var(--border);
  }

  .gp-stats {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 8px;
  }

  .gp-note {
    margin: 0;
    font-size: 11px;
    line-height: 1.42;
    color: var(--text-secondary);
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 6px 8px;
  }

  .gp-warn {
    margin: 0;
    font-size: 11px;
    line-height: 1.4;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 6px;
    padding: 5px 8px;
  }

  .gp-err {
    margin: 0;
    font-size: 11px;
    line-height: 1.4;
    color: var(--err-text);
    background: var(--err-bg);
    border: 1px solid var(--err-border);
    border-radius: 6px;
    padding: 6px 8px;
  }

  .gp-step {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .gp-step-lbl,
  .gp-load-lbl {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .gp-step-num {
    width: 17px;
    height: 17px;
    border-radius: 999px;
    background: var(--bg-pill);
    border: 1px solid var(--accent);
    color: var(--accent-bright);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .gp-btn-row,
  .gp-url-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .gp-btn-primary {
    height: 28px;
    padding: 0 10px;
    border-radius: 6px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: #ffffff;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    font-family: inherit;
  }

  .gp-btn-primary:hover:not(:disabled) {
    filter: brightness(1.08);
  }

  .gp-btn-primary:disabled {
    opacity: 0.48;
    cursor: not-allowed;
  }

  .gp-btn-primary.copied {
    background: #10b981;
    border-color: #10b981;
  }

  .gp-btn-full {
    width: 100%;
    height: 30px;
  }

  .gp-btn-ghost {
    height: 28px;
    padding: 0 10px;
    border-radius: 6px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
    color: var(--text-body);
    font-size: 11.5px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    font-family: inherit;
  }

  .gp-btn-ghost:hover {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }

  .gp-gh-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 28px;
    padding: 0 10px;
    border-radius: 6px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
    color: var(--accent-bright);
    font-size: 11.5px;
    font-weight: 600;
    text-decoration: none;
    align-self: flex-start;
  }

  .gp-gh-link:hover {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }

  .gp-hint {
    margin: 0;
    font-size: 11px;
    line-height: 1.4;
    color: var(--text-muted);
  }

  .gp-hint code,
  .gp-load-lbl code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--accent-bright);
  }

  .gp-input {
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
    height: 28px;
    padding: 0 8px;
    font-size: 11px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-primary);
    background: var(--bg-input);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    outline: none;
  }

  .gp-input:focus {
    border-color: var(--accent);
  }

  .gp-input-ro {
    flex: 1;
  }

  .gp-load-form {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin: 0;
  }

  .gp-load-actions {
    display: flex;
    padding-top: 2px;
  }
</style>
