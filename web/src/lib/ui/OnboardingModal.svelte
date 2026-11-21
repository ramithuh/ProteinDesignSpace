<script lang="ts">
  import { onMount } from 'svelte';

  interface Props {
    onDismiss: () => void;
    onSelectPreset: (presetId: string) => void;
  }

  let { onDismiss, onSelectPreset }: Props = $props();

  let startBtn = $state<HTMLButtonElement | null>(null);

  onMount(() => {
    startBtn?.focus();
  });

  function handleKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      ev.preventDefault();
      onDismiss();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div
  class="onb-backdrop"
  role="presentation"
  onclick={(ev) => {
    if (ev.target === ev.currentTarget) onDismiss();
  }}
>
  <div
    class="onb-card"
    role="dialog"
    aria-modal="true"
    aria-labelledby="onb-title"
  >
    <div class="onb-hero">
      <svg class="onb-mark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="7" fill="#0f172a"/>
        <path d="M8 22 C11 10, 15 10, 16 16 C17 22, 21 22, 24 10" fill="none" stroke="#60a5fa" stroke-width="2.8" stroke-linecap="round"/>
        <circle cx="8" cy="22" r="2.6" fill="#38bdf8"/>
        <circle cx="16" cy="16" r="2.6" fill="#a78bfa"/>
        <circle cx="24" cy="10" r="2.6" fill="#34d399"/>
      </svg>
      <div>
        <h2 id="onb-title">Welcome to <span class="brand-accent">denovo</span>.run</h2>
        <p class="onb-sub">
          Browser-native protein design, sequence design, and 3D structure prediction — running 100% locally on your GPU via WebGPU.
        </p>
      </div>
    </div>

    <div class="onb-steps">
      <div class="step-item">
        <span class="step-num">1</span>
        <div>
          <strong>Build or pick a workflow</strong>
          <p>
            Choose an <b>Example Workflow</b> above or press <kbd>/</kbd> (<b>+ Add Block</b>) to browse
            <b>Structure Generation</b> (RFD4-Proteina, Genie 3), <b>Sequence Design</b> (SolubleMPNN / ProteinMPNN),
            and <b>Structure Prediction</b> (all 13 LocalFold models including AlphaFold2, RoseTTAFold3, Boltz-2 &amp; AlphaFold 3).
          </p>
        </div>
      </div>

      <div class="step-item">
        <span class="step-num">2</span>
        <div>
          <strong>Inspect intermediates &amp; compare in 3D</strong>
          <p>
            Click <b>▶ Run</b> (<kbd>R</kbd>), then click any block or colored wire to <b>Peek</b> at structures,
            sequences, or predicted folds — or <kbd>⌘</kbd>/<kbd>Ctrl</kbd>-click multiple blocks to superpose them in 3D.
          </p>
        </div>
      </div>

      <div class="step-item">
        <span class="step-num">3</span>
        <div>
          <strong>Private &amp; cached on your device</strong>
          <p>
            No servers, no queues, and no data upload. Model weights download once into local browser storage and are reused automatically on future runs.
          </p>
        </div>
      </div>
    </div>

    <div class="quick-starts">
      <span class="qs-lbl">Quick-start presets</span>
      <div class="qs-row">
        <button
          type="button"
          class="qs-btn"
          onclick={() => onSelectPreset('pdb-refold-qc')}
        >
          <span class="qs-title">Fastest Test</span>
          <span class="qs-meta">PDB → SolubleMPNN → AF2 · ~107 MB</span>
        </button>
        <button
          type="button"
          class="qs-btn"
          onclick={() => onSelectPreset('genie3-mpnn-af2')}
        >
          <span class="qs-title">De Novo with Genie 3</span>
          <span class="qs-meta">Genie 3 → SolubleMPNN → AF2 · ~152 MB</span>
        </button>
      </div>
    </div>

    <div class="onb-footer">
      <span class="reopen-hint">Reopen anytime from <b>⋯ → Quick Tour</b></span>
      <button
        type="button"
        class="btn-start"
        bind:this={startBtn}
        onclick={onDismiss}
      >
        Start exploring →
      </button>
    </div>
  </div>
</div>

<style>
  .onb-backdrop {
    position: fixed;
    inset: 0;
    z-index: 210;
    background: rgba(10, 14, 24, 0.64);
    backdrop-filter: blur(5px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
  }

  .onb-card {
    width: 100%;
    max-width: 540px;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.38);
    padding: 22px 24px 20px;
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .onb-hero {
    display: flex;
    align-items: flex-start;
    gap: 14px;
  }

  .onb-mark {
    width: 44px;
    height: 44px;
    flex-shrink: 0;
    filter: drop-shadow(0 6px 14px rgba(37, 99, 235, 0.24));
  }

  h2 {
    margin: 0;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-primary);
  }

  .brand-accent {
    color: var(--accent-bright);
  }

  .onb-sub {
    margin: 4px 0 0;
    font-size: 12.5px;
    line-height: 1.45;
    color: var(--text-muted);
  }

  .onb-steps {
    display: flex;
    flex-direction: column;
    gap: 10px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }

  .step-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }

  .step-num {
    width: 20px;
    height: 20px;
    border-radius: 999px;
    background: var(--bg-pill);
    color: var(--accent-bright);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
  }

  .step-item strong {
    display: block;
    font-size: 12.5px;
    color: var(--text-primary);
    margin-bottom: 2px;
  }

  .step-item p {
    margin: 0;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-secondary);
  }

  .step-item b {
    color: var(--text-primary);
    font-weight: 600;
  }

  kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    padding: 1px 5px;
    border-radius: 4px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-primary);
  }

  .quick-starts {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .qs-lbl {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--text-dim);
  }

  .qs-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .qs-btn {
    text-align: left;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 2px;
    transition: border-color 120ms ease, background 120ms ease;
  }

  .qs-btn:hover {
    border-color: var(--accent);
    background: var(--bg-surface-hover);
  }

  .qs-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .qs-meta {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-muted);
  }

  .onb-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding-top: 2px;
  }

  .reopen-hint {
    font-size: 11px;
    color: var(--text-dim);
  }

  .reopen-hint b {
    color: var(--text-muted);
    font-weight: 600;
  }

  .btn-start {
    background: var(--accent);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }

  .btn-start:hover {
    filter: brightness(1.08);
  }

  .btn-start:focus-visible {
    outline: 2px solid var(--accent-soft);
    outline-offset: 2px;
  }
</style>
