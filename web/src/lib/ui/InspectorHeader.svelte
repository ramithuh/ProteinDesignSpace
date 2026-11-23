<script lang="ts">
  interface Props {
    nodeId: string;
    stageColor: string;
    titleDraft: string;
    placeholder: string;
    displayTitle: string;
    activeModelBadge: string;
    disabled?: boolean;
    host?: 'sidebar' | 'studio';
    onTitleDraftChange?: (next: string) => void;
    onUpdateLabel?: (nodeId: string, label: string) => void;
    onFlushTextCommit?: () => void;
    onOpenStudio?: () => void;
    onRunUpTo?: (nodeId: string) => void;
  }

  let {
    nodeId,
    stageColor,
    titleDraft,
    placeholder,
    displayTitle,
    activeModelBadge,
    disabled = false,
    host = 'sidebar',
    onTitleDraftChange = undefined,
    onUpdateLabel = undefined,
    onFlushTextCommit = undefined,
    onOpenStudio = undefined,
    onRunUpTo = undefined,
  }: Props = $props();

  let showStudioButton = $derived(host === 'sidebar' && Boolean(onOpenStudio));
  let showRunHereButton = $derived(host === 'sidebar' && Boolean(onRunUpTo));
</script>

<div class="head" class:is-studio={host === 'studio'}>
  <div class="title-wrap">
    <span class="stage-dot" style="background: {stageColor}"></span>
    {#if onUpdateLabel}
      <input
        type="text"
        class="title-input"
        aria-label="Block title"
        title="Click to rename this block"
        {disabled}
        {placeholder}
        value={titleDraft}
        oninput={(e) => {
          const val = e.currentTarget.value;
          onTitleDraftChange?.(val);
          onUpdateLabel(nodeId, val);
        }}
        onblur={() => {
          const trimmed = titleDraft.trim();
          if (trimmed !== titleDraft) {
            onTitleDraftChange?.(trimmed);
            onUpdateLabel(nodeId, trimmed);
          }
        }}
        onkeydown={(e) => {
          if (e.key === 'Enter') {
            e.currentTarget.blur();
          }
        }}
      />
    {:else}
      <h3>{displayTitle}</h3>
    {/if}
    <span class="sub-kind" title="Underlying model / engine">{activeModelBadge}</span>
  </div>
  {#if showStudioButton || showRunHereButton}
    <div class="head-actions">
      {#if showStudioButton && onOpenStudio}
        <button
          type="button"
          class="studio-open-btn"
          onclick={() => {
            onFlushTextCommit?.();
            onOpenStudio();
          }}
          title="Inspect what the model sees & paint residue constraints in Studio (Space)"
        >
          <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <rect x="1.8" y="2.4" width="12.4" height="11.2" rx="2" stroke="currentColor" stroke-width="1.4" />
            <path d="M6.2 2.6V13.4" stroke="currentColor" stroke-width="1.3" />
            <path d="M8.6 6H11.8M8.6 8.5H11.8M8.6 11H10.6" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" />
          </svg>
          Inspect Studio
        </button>
      {/if}
      {#if showRunHereButton && onRunUpTo}
        <button
          type="button"
          class="run-here-btn"
          {disabled}
          onclick={() => {
            onFlushTextCommit?.();
            onRunUpTo(nodeId);
          }}
          title="Execute only upstream blocks up to this block (memoized for full runs)"
        >
          ▶ Run to here
        </button>
      {/if}
    </div>
  {/if}
</div>

<style>
  .head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    min-height: 32px;
  }
  .head.is-studio {
    min-height: 28px;
  }
  .title-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  .stage-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  h3 {
    margin: 0;
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .title-input {
    margin: 0;
    padding: 2px 6px;
    font-family: inherit;
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    min-width: 90px;
    max-width: 240px;
    transition: border-color 120ms ease, background 120ms ease;
  }
  .title-input:hover:not(:disabled) {
    border-color: var(--border-strong);
    background: var(--bg-surface-2);
  }
  .title-input:focus {
    outline: none;
    border-color: var(--accent);
    background: var(--bg-input);
  }
  .sub-kind {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-muted);
  }
  .head-actions {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }
  .studio-open-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 4px 9px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    flex-shrink: 0;
    white-space: nowrap;
  }
  .studio-open-btn:hover {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .run-here-btn {
    background: var(--bg-surface-2);
    color: var(--accent-bright);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    flex-shrink: 0;
  }
  .run-here-btn:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }
  .run-here-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
