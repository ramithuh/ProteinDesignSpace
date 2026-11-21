<script lang="ts">
  import {
    MODEL_CARDS,
    formatModelCardPillLabel,
    getActivePrecision,
    getModelCardForAdapter,
    type ModelCardSpec,
  } from '$lib/engine/model-cards';

  interface Props {
    adapterId: string;
    params?: Record<string, unknown>;
    cardId?: string;
    compact?: boolean;
  }

  let { adapterId, params = undefined, cardId = undefined, compact = false }: Props = $props();

  let open = $state(false);
  let closeBtn = $state<HTMLButtonElement | null>(null);

  const card = $derived.by((): ModelCardSpec | null => {
    if (cardId) {
      return getModelCardForAdapter(cardId, params) ?? MODEL_CARDS[cardId] ?? null;
    }
    return getModelCardForAdapter(adapterId, params);
  });

  const activePrecision = $derived(card ? getActivePrecision(card) : null);
  const pillLabel = $derived(card ? formatModelCardPillLabel(card, compact) : '');
  const dialogTitleId = $derived(card ? `model-card-title-${card.id}` : 'model-card-title');

  function shortSha(sha: string | undefined): string {
    if (!sha) return '';
    return sha.length > 12 ? sha.slice(0, 12) : sha;
  }

  function formatSizeMB(mb: number): string {
    if (mb >= 1000) return `${(mb / 1000).toFixed(2)} GB`;
    if (mb < 20 && !Number.isInteger(mb)) return `${mb.toFixed(1)} MB`;
    return `${Math.round(mb)} MB`;
  }

  function repoLabel(url: string): string {
    return url.replace(/^https?:\/\/(www\.)?/, '').replace(/\/$/, '');
  }

  function openModal(ev: MouseEvent) {
    ev.stopPropagation();
    open = true;
    queueMicrotask(() => closeBtn?.focus());
  }

  function closeModal() {
    open = false;
  }

  function handleKeydown(ev: KeyboardEvent) {
    if (open && ev.key === 'Escape') {
      ev.preventDefault();
      ev.stopPropagation();
      closeModal();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if card && activePrecision}
  <button
    type="button"
    class="mc-pill"
    class:mc-pill-compact={compact}
    class:mc-pill-hosted={card.status === 'weights-hosted'}
    aria-haspopup="dialog"
    aria-expanded={open}
    title={`${card.displayName} · ${activePrecision.label} (${formatSizeMB(activePrecision.approxSizeMB)}) — Inspect precision, measured PyTorch FP32 parity & provenance`}
    onclick={openModal}
  >
    <span class="mc-pill-dot" aria-hidden="true"></span>
    <span class="mc-pill-text">{pillLabel}</span>
    <svg
      class="mc-pill-icon"
      viewBox="0 0 16 16"
      width="11"
      height="11"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.35" />
      <path d="M8 7.2v3.7" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" />
      <circle cx="8" cy="4.8" r="0.8" fill="currentColor" />
    </svg>
  </button>

  {#if open}
    <div
      class="mc-backdrop"
      role="presentation"
      onclick={(ev) => {
        ev.stopPropagation();
        if (ev.target === ev.currentTarget) closeModal();
      }}
    >
      <div
        class="mc-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={dialogTitleId}
      >
        <header class="mc-hdr">
          <div class="mc-hdr-top">
            <div class="mc-badges">
              <span class="mc-status-badge" class:is-hosted={card.status === 'weights-hosted'}>
                {card.status === 'live' ? 'LIVE IN BROWSER' : 'WEIGHTS HOSTED · WEBGPU PORT'}
              </span>
              <span class="mc-prec-badge">{activePrecision.label}</span>
              <span class="mc-size-badge">~{formatSizeMB(activePrecision.approxSizeMB)}</span>
            </div>
            <button
              type="button"
              class="mc-close"
              aria-label="Close model card"
              bind:this={closeBtn}
              onclick={closeModal}
            >
              ×
            </button>
          </div>
          <h2 id={dialogTitleId} class="mc-title">{card.displayName} — Model &amp; Parity Card</h2>
          <p class="mc-quant-summary">{activePrecision.quantizationDetail}</p>
          {#if activePrecision.latencyHint}
            <p class="mc-latency">Measured latency: <span>{activePrecision.latencyHint}</span></p>
          {/if}
        </header>

        <div class="mc-scroll">
          <section class="mc-sec">
            <div class="mc-sec-title">Numerical Parity vs PyTorch FP32</div>
            <div class="mc-table-wrap">
              <table class="mc-table">
                <thead>
                  <tr>
                    <th>Precision</th>
                    <th>Size</th>
                    <th>Measured Parity</th>
                    <th>Scope &amp; Hardware</th>
                  </tr>
                </thead>
                <tbody>
                  {#each card.precisions as tier (tier.id)}
                    {@const isActive = tier.id === card.activePrecisionId}
                    <tr class:is-active={isActive}>
                      <td class="mc-col-prec">
                        <div class="mc-tier-name">
                          <span>{tier.label}</span>
                          {#if isActive}
                            <span class="mc-active-tag">ACTIVE</span>
                          {/if}
                        </div>
                      </td>
                      <td class="mc-col-size">{formatSizeMB(tier.approxSizeMB)}</td>
                      <td class="mc-col-parity">
                        {#if tier.parityVsPytorchFp32}
                          <div class="mc-metric-label">{tier.parityVsPytorchFp32.metric}</div>
                          <div class="mc-metric-val">{tier.parityVsPytorchFp32.value}</div>
                          {#if tier.parityVsPytorchFp32.maxAbsDiff}
                            <div class="mc-maxabs">
                              maxAbsDiff: <code>{tier.parityVsPytorchFp32.maxAbsDiff}</code>
                            </div>
                          {/if}
                        {:else}
                          <div class="mc-metric-val mc-muted">No parity benchmark recorded</div>
                        {/if}
                        {#if tier.caveat}
                          <div class="mc-tier-caveat">{tier.caveat}</div>
                        {/if}
                      </td>
                      <td class="mc-col-scope">
                        {#if tier.parityVsPytorchFp32}
                          <div>{tier.parityVsPytorchFp32.scope}</div>
                          {#if tier.parityVsPytorchFp32.hardware}
                            <div class="mc-hw">{tier.parityVsPytorchFp32.hardware}</div>
                          {/if}
                        {/if}
                        {#if tier.latencyHint}
                          <div class="mc-tier-lat">{tier.latencyHint}</div>
                        {/if}
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          </section>

          {#if card.limitations.length > 0}
            <details class="mc-notes">
              <summary class="mc-notes-summary">
                Port &amp; scope notes ({card.limitations.length})
              </summary>
              <ul class="mc-limits">
                {#each card.limitations as lim (lim)}
                  <li>{lim}</li>
                {/each}
              </ul>
            </details>
          {/if}

          <footer class="mc-prov">
            <div class="mc-sec-title">Provenance &amp; Licensing</div>
            <div class="mc-prov-grid">
              <div class="mc-prov-row">
                <span class="mc-prov-key">Upstream</span>
                <span class="mc-prov-val">
                  <a
                    href={card.upstreamRepo}
                    target="_blank"
                    rel="noopener noreferrer"
                    class="mc-link"
                  >
                    {repoLabel(card.upstreamRepo)} ↗
                  </a>
                  {#if card.upstreamCommit}
                    <code class="mc-sha" title={card.upstreamCommit}>@{shortSha(card.upstreamCommit)}</code>
                  {/if}
                </span>
              </div>

              {#if card.weightsRepo}
                <div class="mc-prov-row">
                  <span class="mc-prov-key">Weights</span>
                  <span class="mc-prov-val">
                    <a
                      href={card.weightsRepo}
                      target="_blank"
                      rel="noopener noreferrer"
                      class="mc-link"
                    >
                      {repoLabel(card.weightsRepo)} ↗
                    </a>
                    {#if card.weightsRevision}
                      <code class="mc-sha" title={card.weightsRevision}>rev {shortSha(card.weightsRevision)}</code>
                    {/if}
                  </span>
                </div>
              {/if}

              <div class="mc-prov-row">
                <span class="mc-prov-key">License</span>
                <span class="mc-prov-val">
                  <span class="mc-lic-badge">{card.license}</span>
                </span>
              </div>

              <div class="mc-prov-row">
                <span class="mc-prov-key">Citation</span>
                <span class="mc-prov-val mc-citation">{card.citation}</span>
              </div>
            </div>
          </footer>
        </div>
      </div>
    </div>
  {/if}
{/if}

<style>
  .mc-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 7px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 600;
    line-height: 1.35;
    letter-spacing: 0.01em;
    cursor: pointer;
    white-space: nowrap;
    max-width: 100%;
    transition: border-color 0.14s ease, background 0.14s ease, color 0.14s ease;
  }

  .mc-pill:hover {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
    color: var(--text-primary);
  }

  .mc-pill:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }

  .mc-pill-compact {
    height: 20px;
    padding: 0 7px;
    font-size: 9.5px;
    line-height: 1;
    gap: 4px;
    box-sizing: border-box;
  }

  .mc-pill-dot {
    width: 5px;
    height: 5px;
    border-radius: 999px;
    background: var(--ok-text);
    flex-shrink: 0;
  }

  .mc-pill-hosted .mc-pill-dot {
    background: var(--accent-bright);
  }

  .mc-pill-text {
    line-height: 1;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .mc-pill-icon {
    display: block;
    opacity: 0.72;
    flex-shrink: 0;
  }

  .mc-notes {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-surface-2);
    padding: 7px 10px;
  }

  .mc-notes-summary {
    cursor: pointer;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--text-muted);
    user-select: none;
  }

  .mc-notes[open] .mc-notes-summary {
    margin-bottom: 6px;
  }

  .mc-backdrop {
    position: fixed;
    inset: 0;
    z-index: 220;
    background: rgba(10, 14, 24, 0.62);
    -webkit-backdrop-filter: blur(4px);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    box-sizing: border-box;
  }

  .mc-modal {
    width: 100%;
    max-width: min(720px, calc(100vw - 24px));
    max-height: min(86vh, 760px);
    box-sizing: border-box;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.38);
    padding: 18px 20px 16px;
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 12px;
    text-align: left;
  }

  .mc-hdr {
    display: flex;
    flex-direction: column;
    gap: 6px;
    flex-shrink: 0;
  }

  .mc-hdr-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .mc-badges {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .mc-status-badge,
  .mc-prec-badge,
  .mc-size-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 2px 8px;
    border-radius: 999px;
  }

  .mc-status-badge {
    color: var(--ok-text);
    background: var(--ok-bg);
    border: 1px solid var(--ok-border);
  }

  .mc-status-badge.is-hosted {
    color: var(--accent-bright);
    background: var(--bg-pill);
    border: 1px solid var(--border-strong);
  }

  .mc-prec-badge {
    color: var(--accent-bright);
    background: var(--bg-pill);
  }

  .mc-size-badge {
    color: var(--text-muted);
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
  }

  .mc-close {
    width: 26px;
    height: 26px;
    border-radius: 6px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-muted);
    font-size: 18px;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .mc-close:hover {
    background: var(--bg-surface-hover);
    border-color: var(--border);
    color: var(--text-primary);
  }

  .mc-title {
    margin: 0;
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .mc-quant-summary {
    margin: 0;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-secondary);
  }

  .mc-latency {
    margin: 0;
    font-size: 11px;
    color: var(--text-muted);
  }

  .mc-latency span {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-secondary);
  }

  .mc-scroll {
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding-right: 4px;
  }

  .mc-sec {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .mc-sec-title {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.45px;
    color: var(--text-dim);
  }

  .mc-table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-surface-2);
  }

  .mc-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11.5px;
  }

  .mc-table th {
    text-align: left;
    padding: 7px 9px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.35px;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }

  .mc-table td {
    padding: 8px 9px;
    vertical-align: top;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    line-height: 1.4;
  }

  .mc-table tbody tr:last-child td {
    border-bottom: none;
  }

  .mc-table tbody tr.is-active {
    background: var(--bg-surface-selected);
  }

  .mc-col-prec {
    white-space: nowrap;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-weight: 700;
    color: var(--text-primary);
  }

  .mc-tier-name {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }

  .mc-active-tag {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 1px 5px;
    border-radius: 4px;
    color: var(--accent-bright);
    background: var(--bg-pill);
  }

  .mc-col-size {
    white-space: nowrap;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-muted);
  }

  .mc-metric-label {
    font-size: 10.5px;
    color: var(--text-muted);
  }

  .mc-metric-val {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-primary);
    margin-top: 2px;
  }

  .mc-maxabs {
    font-size: 10.5px;
    color: var(--text-muted);
    margin-top: 3px;
  }

  .mc-maxabs code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-secondary);
  }

  .mc-tier-caveat {
    margin-top: 4px;
    font-size: 10.8px;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 5px;
    padding: 3px 6px;
  }

  .mc-col-scope {
    font-size: 11px;
    color: var(--text-muted);
  }

  .mc-hw {
    margin-top: 3px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
  }

  .mc-tier-lat {
    margin-top: 3px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .mc-muted {
    color: var(--text-dim);
  }

  .mc-limits {
    margin: 0;
    padding-left: 16px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 11.8px;
    line-height: 1.45;
    color: var(--text-secondary);
  }

  .mc-prov {
    display: flex;
    flex-direction: column;
    gap: 7px;
    border-top: 1px solid var(--border-subtle);
    padding-top: 10px;
  }

  .mc-prov-grid {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 11.5px;
  }

  .mc-prov-row {
    display: grid;
    grid-template-columns: 74px 1fr;
    gap: 8px;
    align-items: baseline;
  }

  .mc-prov-key {
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  .mc-prov-val {
    display: flex;
    align-items: baseline;
    gap: 6px;
    flex-wrap: wrap;
    color: var(--text-secondary);
    min-width: 0;
  }

  .mc-link {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--accent-bright);
    text-decoration: none;
    word-break: break-all;
  }

  .mc-link:hover {
    text-decoration: underline;
  }

  .mc-sha {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0 5px;
  }

  .mc-lic-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-secondary);
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 1px 6px;
  }

  .mc-citation {
    font-size: 11.2px;
    line-height: 1.4;
    color: var(--text-muted);
  }
</style>
