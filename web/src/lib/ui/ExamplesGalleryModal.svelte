<script lang="ts">
  import { PAPER_INSIGHT_EXAMPLES, type PaperInsightExample } from '$lib/engine/paper-gallery';
  import { WORKFLOW_PRESETS } from '$lib/engine/registry';

  interface Props {
    open: boolean;
    activeId: string;
    isRunning: boolean;
    onClose: () => void;
    onSelectPaper: (example: PaperInsightExample) => void;
    onSelectPreset: (presetId: string) => void;
  }

  let {
    open,
    activeId,
    isRunning,
    onClose,
    onSelectPaper,
    onSelectPreset,
  }: Props = $props();

  function handleKeydown(ev: KeyboardEvent) {
    if (open && ev.key === 'Escape') {
      ev.preventDefault();
      onClose();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="gal-backdrop"
    role="presentation"
    onclick={(ev) => {
      if (ev.target === ev.currentTarget) onClose();
    }}
  >
    <div class="gal-modal" role="dialog" aria-modal="true" aria-labelledby="gal-title">
      <div class="gal-header">
        <h2 id="gal-title">Examples</h2>
        <button type="button" class="gal-close" onclick={onClose} aria-label="Close examples">
          ✕
        </button>
      </div>

      <div class="gal-body">
        <section class="gal-section">
          <div class="gal-sec-label">From the literature</div>
          <div class="paper-grid">
            {#each PAPER_INSIGHT_EXAMPLES as ex (ex.id)}
              <button
                type="button"
                class="paper-card"
                class:selected={activeId === ex.id}
                disabled={isRunning}
                onclick={() => onSelectPaper(ex)}
              >
                <div class="paper-cite">{ex.citation}</div>
                <div class="paper-title">{ex.title}</div>
                <p class="paper-finding">{ex.finding}</p>
                <div class="paper-meta">
                  <span class="meta-lbl">Tools</span>
                  <span>{ex.tools.join(' · ')}</span>
                </div>
              </button>
            {/each}
          </div>
        </section>

        <section class="gal-section">
          <div class="gal-sec-label">Other pipelines</div>
          <div class="preset-list">
            {#each WORKFLOW_PRESETS as p (p.id)}
              <button
                type="button"
                class="preset-row"
                class:selected={activeId === p.id}
                disabled={isRunning}
                onclick={() => onSelectPreset(p.id)}
              >
                <div class="preset-main">
                  <span class="preset-name">{p.name}</span>
                  <span class="preset-sum">{p.summary}</span>
                </div>
                <span class="preset-badge">{p.badge}</span>
              </button>
            {/each}
          </div>
        </section>
      </div>
    </div>
  </div>
{/if}

<style>
  .gal-backdrop {
    position: fixed;
    inset: 0;
    z-index: 230;
    background: rgba(10, 14, 24, 0.52);
    -webkit-backdrop-filter: blur(4px);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    box-sizing: border-box;
  }

  .gal-modal {
    width: 100%;
    max-width: min(860px, calc(100vw - 32px));
    max-height: min(84vh, 760px);
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    color: var(--text-body);
  }

  .gal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 15px 20px;
    border-bottom: 1px solid var(--border);
  }

  .gal-header h2 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text-primary);
  }

  .gal-close {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-muted);
    font-size: 13px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .gal-close:hover {
    background: var(--bg-surface-2);
    color: var(--text-primary);
  }

  .gal-body {
    overflow-y: auto;
    padding: 18px 20px 20px;
    display: flex;
    flex-direction: column;
    gap: 22px;
  }

  .gal-sec-label {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-dim);
    margin-bottom: 10px;
  }

  .paper-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
  }

  @media (max-width: 760px) {
    .paper-grid {
      grid-template-columns: 1fr;
    }
  }

  .paper-card {
    text-align: left;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 7px;
    cursor: pointer;
    color: inherit;
    font-family: inherit;
    transition: border-color 120ms ease, background 120ms ease;
  }
  .paper-card:hover:not(:disabled) {
    border-color: var(--accent);
    background: var(--bg-surface);
  }
  .paper-card.selected {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
  }
  .paper-card:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .paper-cite {
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1.35;
  }

  .paper-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.3;
    letter-spacing: -0.01em;
  }

  .paper-finding {
    margin: 0;
    font-size: 12.5px;
    line-height: 1.48;
    color: var(--text-muted);
  }

  .paper-meta {
    margin-top: auto;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-muted);
  }

  .meta-lbl {
    font-family: inherit;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
  }

  .preset-list {
    display: flex;
    flex-direction: column;
    border: 1px solid var(--border);
    border-radius: 9px;
    overflow: hidden;
  }

  .preset-row {
    text-align: left;
    background: var(--bg-surface);
    border: none;
    border-bottom: 1px solid var(--border);
    padding: 10px 13px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    cursor: pointer;
    color: inherit;
    font-family: inherit;
    transition: background 100ms ease;
  }
  .preset-row:last-child {
    border-bottom: none;
  }
  .preset-row:hover:not(:disabled) {
    background: var(--bg-surface-2);
  }
  .preset-row.selected {
    background: var(--bg-surface-2);
    box-shadow: inset 2px 0 0 var(--accent);
  }
  .preset-row:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .preset-main {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .preset-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .preset-sum {
    font-size: 11.5px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .preset-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-dim);
    white-space: nowrap;
    flex-shrink: 0;
  }
</style>
