<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import type { WorkflowGraph } from '$lib/engine/types';

  interface Props {
    graph: WorkflowGraph;
    isRunning: boolean;
    onDismiss: () => void;
    onSelectPreset: (presetId: string) => void;
    onPrepareTourStarter: () => void;
    onAutoStepBlock: (adapterId: 'web-protein-mpnn' | 'localfold-af2' | 'designability-scorer') => void;
    onInteractiveChange?: (active: boolean) => void;
  }

  let {
    graph,
    isRunning,
    onDismiss,
    onSelectPreset,
    onPrepareTourStarter,
    onAutoStepBlock,
    onInteractiveChange = undefined,
  }: Props = $props();

  type TourStepId =
    | 'welcome'
    | 'add-mpnn'
    | 'add-fold'
    | 'add-scorer'
    | 'inspect-params'
    | 'run-workflow';

  interface SpotlightRect {
    top: number;
    left: number;
    width: number;
    height: number;
  }

  const INTERACTIVE_STEPS: TourStepId[] = [
    'add-mpnn',
    'add-fold',
    'add-scorer',
    'inspect-params',
    'run-workflow',
  ];

  let step = $state<TourStepId>('welcome');
  let cardEl = $state<HTMLDivElement | null>(null);
  let targetRect = $state<SpotlightRect | null>(null);
  let catalogIsOpen = $state(false);
  let viewportW = $state(1280);
  let viewportH = $state(800);

  let interactiveIdx = $derived(INTERACTIVE_STEPS.indexOf(step));
  let isInteractive = $derived(interactiveIdx >= 0);

  $effect(() => {
    onInteractiveChange?.(isInteractive);
  });

  function targetSelectorFor(s: TourStepId): string | null {
    switch (s) {
      case 'add-mpnn':
      case 'add-fold':
      case 'add-scorer':
        return '[data-tour="add-block-btn"]';
      case 'inspect-params':
        return '[data-tour="inspector-panel"]';
      case 'run-workflow':
        return '[data-tour="run-btn"]';
      default:
        return null;
    }
  }

  function measureTarget() {
    if (typeof window === 'undefined') return;
    viewportW = window.innerWidth;
    viewportH = window.innerHeight;
    const sel = targetSelectorFor(step);
    if (!sel) {
      targetRect = null;
      return;
    }
    const el = document.querySelector<HTMLElement>(sel);
    if (!el) {
      targetRect = null;
      return;
    }
    const r = el.getBoundingClientRect();
    const pad = step === 'inspect-params' ? 4 : 6;
    targetRect = {
      top: Math.max(4, r.top - pad),
      left: Math.max(4, r.left - pad),
      width: Math.min(viewportW - 8, r.width + pad * 2),
      height: Math.min(viewportH - 8, r.height + pad * 2),
    };
  }

  function startInteractiveTour() {
    onPrepareTourStarter();
    step = 'add-mpnn';
    void tick().then(() => {
      requestAnimationFrame(measureTarget);
    });
  }

  // Live auto-advance when user clicks real blocks in + Add Block or starts Run!
  $effect(() => {
    const hasMpnn = graph.nodes.some((n) => n.adapterId === 'web-protein-mpnn');
    const hasFold = graph.nodes.some((n) => n.adapterId === 'localfold-af2');
    const hasScorer = graph.nodes.some((n) => n.adapterId === 'designability-scorer');

    if (step === 'add-mpnn' && hasMpnn) {
      step = hasFold ? (hasScorer ? 'inspect-params' : 'add-scorer') : 'add-fold';
    } else if (step === 'add-fold' && hasFold) {
      step = hasScorer ? 'inspect-params' : 'add-scorer';
    } else if (step === 'add-scorer' && hasScorer) {
      step = 'inspect-params';
    } else if (step === 'run-workflow' && isRunning) {
      onDismiss();
    }
  });

  // Re-measure spotlight whenever step or layout changes
  $effect(() => {
    void step;
    void graph.nodes.length;
    if (typeof window === 'undefined') return;
    const raf = requestAnimationFrame(measureTarget);
    return () => cancelAnimationFrame(raf);
  });

  // Compute non-overlapping coachmark placement for each spotlight step
  let coachStyle = $derived.by((): string => {
    const cardW = Math.min(350, Math.max(280, viewportW - 24));
    if (!targetRect) {
      return `bottom: 24px; right: 24px; width: ${cardW}px;`;
    }

    if (step === 'add-mpnn' || step === 'add-fold' || step === 'add-scorer') {
      // Keep bottom-left/middle clear of the + Add Block popover & right flyout:
      // place cleanly near bottom-right of viewport so user can browse the full catalog freely!
      return `bottom: 22px; right: 22px; width: ${cardW}px;`;
    }

    if (step === 'inspect-params') {
      // Place to the left of the right inspector panel (or bottom-left on narrow viewports)
      if (targetRect.left > cardW + 32) {
        const left = Math.max(16, targetRect.left - cardW - 18);
        const top = Math.min(Math.max(70, targetRect.top + 12), Math.max(70, viewportH - 280));
        return `top: ${top}px; left: ${left}px; width: ${cardW}px;`;
      }
      return `bottom: 20px; left: 16px; width: ${cardW}px;`;
    }

    if (step === 'run-workflow') {
      const top = Math.min(viewportH - 250, targetRect.top + targetRect.height + 14);
      const right = Math.max(16, viewportW - (targetRect.left + targetRect.width));
      return `top: ${top}px; right: ${right}px; width: ${cardW}px;`;
    }

    return `bottom: 22px; right: 22px; width: ${cardW}px;`;
  });

  let domObserver: MutationObserver | null = null;

  onMount(() => {
    cardEl?.focus({ preventScroll: true });
    measureTarget();
    window.addEventListener('resize', measureTarget);
    window.addEventListener('scroll', measureTarget, true);
    if (typeof MutationObserver !== 'undefined') {
      const syncCatalog = () => {
        catalogIsOpen = Boolean(document.querySelector('.catalog-shell'));
      };
      syncCatalog();
      domObserver = new MutationObserver(syncCatalog);
      domObserver.observe(document.body, { childList: true, subtree: true });
    }
  });

  onDestroy(() => {
    onInteractiveChange?.(false);
    domObserver?.disconnect();
    domObserver = null;
    if (typeof window !== 'undefined') {
      window.removeEventListener('resize', measureTarget);
      window.removeEventListener('scroll', measureTarget, true);
    }
  });

  function handleKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      // If the block catalog popover is open during interactive steps, let the popover close first
      if (isInteractive && document.querySelector('.catalog-shell')) {
        return;
      }
      ev.preventDefault();
      onDismiss();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if step === 'welcome'}
  <div
    class="onb-backdrop"
    role="presentation"
    onclick={(ev) => {
      if (ev.target === ev.currentTarget) onDismiss();
    }}
  >
    <div
      class="onb-card"
      bind:this={cardEl}
      tabindex="-1"
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
            Browser-native protein design, inverse folding, and 3D structure prediction — running 100% locally on your GPU via WebGPU.
          </p>
        </div>
      </div>

      <div class="onb-highlights">
        <div class="hl-item">
          <span class="hl-dot" style="background: #3b82f6"></span>
          <span><b>Design &amp; Conditioning:</b> RFD4-Proteina, Genie 3, RCSB PDB IDs &amp; 16 curated benchmark targets.</span>
        </div>
        <div class="hl-item">
          <span class="hl-dot" style="background: #a855f7"></span>
          <span><b>Sequence Design:</b> SolubleMPNN, ProteinMPNN &amp; Cα-only with residue-level locking.</span>
        </div>
        <div class="hl-item">
          <span class="hl-dot" style="background: #10b981"></span>
          <span><b>Structure Prediction:</b> 18 LocalFold models (AF2 Monomer &amp; Multimer v3, RF3, Boltz-2, AF3 &amp; more) cached locally in your browser.</span>
        </div>
      </div>

      <div class="quick-starts">
        <span class="qs-lbl">Or jump straight into a preset</span>
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
        <button type="button" class="btn-skip" onclick={onDismiss}>
          Skip &amp; explore
        </button>
        <button
          type="button"
          class="btn-start"
          onclick={startInteractiveTour}
        >
          Build your first workflow (30s) →
        </button>
      </div>
      <div class="reopen-foot">Reopen anytime from <b>⋯ → Quick Tour</b></div>
    </div>
  </div>
{:else}
  <!-- Interactive Spotlight Ring (non-blocking so user clicks real UI elements & popovers) -->
  {#if targetRect}
    <div
      class="spotlight-ring"
      class:soft={catalogIsOpen}
      style="top: {targetRect.top}px; left: {targetRect.left}px; width: {targetRect.width}px; height: {targetRect.height}px;"
      aria-hidden="true"
    ></div>
  {/if}

  <!-- Floating Coachmark Card -->
  <div
    class="coach-card"
    style={coachStyle}
    role="dialog"
    aria-label="Interactive walkthrough"
  >
    <div class="coach-top">
      <div class="coach-dots" aria-label={`Step ${interactiveIdx + 1} of ${INTERACTIVE_STEPS.length}`}>
        {#each INTERACTIVE_STEPS as s, i (s)}
          <span class="c-dot" class:done={i < interactiveIdx} class:active={i === interactiveIdx}></span>
        {/each}
        <span class="c-count">Step {interactiveIdx + 1} / {INTERACTIVE_STEPS.length}</span>
      </div>
      <button type="button" class="coach-x" onclick={onDismiss} title="Exit tour (Esc)" aria-label="Exit tour">
        ×
      </button>
    </div>

    {#if step === 'add-mpnn'}
      <div class="coach-title">1. Add Sequence Design</div>
      <p class="coach-body">
        We placed a starter <b>PDB Input</b> on the canvas. Click the highlighted <b>+ Add Block</b> button
        (or press <kbd>/</kbd>), hover <b>Sequence Design</b>, and pick <b>SolubleMPNN</b> — it will connect automatically.
      </p>
      <div class="coach-actions">
        <button type="button" class="coach-link" onclick={onDismiss}>Skip tour</button>
        <button type="button" class="coach-btn" onclick={() => onAutoStepBlock('web-protein-mpnn')}>
          Do it for me →
        </button>
      </div>

    {:else if step === 'add-fold'}
      <div class="coach-ok">✓ SolubleMPNN added &amp; wired!</div>
      <div class="coach-title">2. Add Structure Prediction</div>
      <p class="coach-body">
        Next, verify whether your designed sequences fold back to the target shape. Click <b>+ Add Block</b>,
        hover <b>Structure Prediction</b>, and choose <b>AlphaFold2 · Model 1</b> (or any of the 18 LocalFold models).
      </p>
      <div class="coach-actions">
        <button type="button" class="coach-link" onclick={onDismiss}>Skip tour</button>
        <button type="button" class="coach-btn" onclick={() => onAutoStepBlock('localfold-af2')}>
          Do it for me →
        </button>
      </div>

    {:else if step === 'add-scorer'}
      <div class="coach-ok">✓ LocalFold added &amp; wired!</div>
      <div class="coach-title">3. Add Designability Scoring</div>
      <p class="coach-body">
        Complete the loop! Click <b>+ Add Block</b> and choose <b>Evaluation → scRMSD / scTM</b> to superimpose
        the predicted fold onto the input backbone and score pass rate.
      </p>
      <div class="coach-actions">
        <button type="button" class="coach-link" onclick={onDismiss}>Skip tour</button>
        <button type="button" class="coach-btn" onclick={() => onAutoStepBlock('designability-scorer')}>
          Do it for me →
        </button>
      </div>

    {:else if step === 'inspect-params'}
      <div class="coach-ok">✓ Full 4-block pipeline built!</div>
      <div class="coach-title">4. Configure &amp; Peek Intermediates</div>
      <p class="coach-body">
        Edit any block's parameters here — try switching <b>PDB Input</b> to <b>Fetch from RCSB PDB ID</b> or picking
        a curated benchmark target. After running, switch to <b>Outputs</b> or click any colored wire to inspect intermediates,
        or <kbd>⌘</kbd>/<kbd>Ctrl</kbd>-click blocks to compare in 3D.
      </p>
      <div class="coach-actions">
        <button type="button" class="coach-link" onclick={onDismiss}>Skip tour</button>
        <button type="button" class="coach-btn primary" onclick={() => (step = 'run-workflow')}>
          Next →
        </button>
      </div>

    {:else if step === 'run-workflow'}
      <div class="coach-title">5. Run Locally on WebGPU</div>
      <p class="coach-body">
        Everything is wired and validated! Click <b>▶ Run</b> (or press <kbd>R</kbd>) to execute locally on your GPU —
        model weights download once and cache in your browser. Explore more templates anytime in <b>Example Workflow</b>.
      </p>
      <div class="coach-actions">
        <button type="button" class="coach-link" onclick={onDismiss}>Finish tour ✓</button>
      </div>
    {/if}
  </div>
{/if}

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
    padding: 22px 24px 16px;
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 15px;
    outline: none;
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

  .onb-highlights {
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }

  .hl-item {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-secondary);
  }

  .hl-item b {
    color: var(--text-primary);
    font-weight: 600;
  }

  .hl-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 5px;
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

  .btn-skip {
    background: transparent;
    border: 1px solid var(--border-strong);
    color: var(--text-secondary);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12.5px;
    font-weight: 500;
    cursor: pointer;
  }

  .btn-skip:hover {
    color: var(--text-primary);
    border-color: var(--accent);
    background: var(--bg-surface-hover);
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

  .reopen-foot {
    font-size: 11px;
    color: var(--text-dim);
    text-align: center;
    margin-top: -6px;
  }

  .reopen-foot b {
    color: var(--text-muted);
    font-weight: 600;
  }

  /* ── Interactive Spotlight Overlay ── */
  .spotlight-ring {
    position: fixed;
    z-index: 195;
    border-radius: 10px;
    border: 2px solid var(--accent-bright, #60a5fa);
    box-shadow:
      0 0 0 4px rgba(59, 130, 246, 0.28),
      0 0 0 9999px rgba(10, 14, 24, 0.44);
    pointer-events: none;
    transition:
      top 260ms cubic-bezier(0.22, 1, 0.36, 1),
      left 260ms cubic-bezier(0.22, 1, 0.36, 1),
      width 260ms cubic-bezier(0.22, 1, 0.36, 1),
      height 260ms cubic-bezier(0.22, 1, 0.36, 1),
      box-shadow 180ms ease;
    animation: tour-pulse 2s ease-in-out infinite;
  }

  .spotlight-ring.soft {
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.25);
  }

  @keyframes tour-pulse {
    0%,
    100% {
      border-color: #60a5fa;
    }
    50% {
      border-color: #93c5fd;
    }
  }

  .coach-card {
    position: fixed;
    z-index: 220;
    background: var(--bg-surface);
    border: 1.5px solid var(--accent);
    border-radius: 12px;
    box-shadow:
      0 18px 48px rgba(0, 0, 0, 0.36),
      0 0 0 1px rgba(59, 130, 246, 0.2);
    padding: 14px 16px 13px;
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 8px;
    box-sizing: border-box;
  }

  .coach-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .coach-dots {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .c-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: var(--border-strong);
    transition: background 180ms ease, width 180ms ease;
  }

  .c-dot.done {
    background: #10b981;
  }

  .c-dot.active {
    width: 16px;
    background: var(--accent);
  }

  .c-count {
    margin-left: 4px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
  }

  .coach-x {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 17px;
    line-height: 1;
    cursor: pointer;
    padding: 0 4px;
    border-radius: 4px;
  }

  .coach-x:hover {
    color: var(--text-primary);
    background: var(--bg-surface-hover);
  }

  .coach-ok {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    color: var(--ok-text, #10b981);
    background: var(--ok-bg);
    border: 1px solid var(--ok-border);
    padding: 3px 8px;
    border-radius: 5px;
    align-self: flex-start;
  }

  .coach-title {
    font-size: 13.5px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .coach-body {
    margin: 0;
    font-size: 12px;
    line-height: 1.48;
    color: var(--text-secondary);
  }

  .coach-body b {
    color: var(--text-primary);
    font-weight: 600;
  }

  kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    padding: 1px 5px;
    border-radius: 4px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
    color: var(--text-primary);
  }

  .coach-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-top: 2px;
  }

  .coach-link {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 11.5px;
    cursor: pointer;
    padding: 4px 2px;
  }

  .coach-link:hover {
    color: var(--text-primary);
    text-decoration: underline;
  }

  .coach-btn {
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--accent);
    border-radius: 7px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }

  .coach-btn:hover {
    background: var(--accent);
    color: #fff;
  }

  .coach-btn.primary {
    background: var(--accent);
    color: #fff;
  }

  .coach-btn.primary:hover {
    filter: brightness(1.08);
  }

  @media (prefers-reduced-motion: reduce) {
    .spotlight-ring {
      animation: none;
      transition: none;
    }
  }
</style>
