<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import type { WorkflowGraph } from '$lib/engine/types';
  import { modalFocusTrap } from './modal-focus-trap';

  interface Props {
    graph: WorkflowGraph;
    isRunning: boolean;
    onDismiss: () => void;
    onPrepareTourStarter: () => void;
    onAutoStepBlock: (adapterId: 'web-protein-mpnn' | 'localfold-af2' | 'designability-scorer') => void;
    onInteractiveChange?: (active: boolean) => void;
  }

  let {
    graph,
    isRunning,
    onDismiss,
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

  type SubStage = 'button' | 'category' | 'item';
  type CaretSide = 'left' | 'right' | 'top' | 'bottom' | 'none';

  interface RectBox {
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

  const ADD_STEP_TARGETS: Record<
    'add-mpnn' | 'add-fold' | 'add-scorer',
    { categoryId: string; categoryLabel: string; subKey: string; subLabel: string }
  > = {
    'add-mpnn': {
      categoryId: 'sequence-design',
      categoryLabel: 'Sequence Design',
      subKey: 'ca',
      subLabel: 'Cα-ProteinMPNN',
    },
    'add-fold': {
      categoryId: 'prediction',
      categoryLabel: 'Structure Prediction',
      subKey: 'monomer',
      subLabel: 'AlphaFold2 · Model 1',
    },
    'add-scorer': {
      categoryId: 'evaluation',
      categoryLabel: 'Evaluation',
      subKey: 'designability-scorer',
      subLabel: 'scRMSD / scTM',
    },
  };

  let step = $state<TourStepId>('welcome');
  let subStage = $state<SubStage>('button');
  let cardEl = $state<HTMLDivElement | null>(null);
  let targetRect = $state<RectBox | null>(null);
  let interactiveHole = $state<RectBox | null>(null);
  let viewportW = $state(1280);
  let viewportH = $state(800);
  let firstRunModalOpen = $state(false);
  let lastScrolledStep: TourStepId | null = null;

  let interactiveIdx = $derived(INTERACTIVE_STEPS.indexOf(step));
  let isInteractive = $derived(interactiveIdx >= 0);

  $effect(() => {
    onInteractiveChange?.(isInteractive);
  });

  function unionRects(rects: RectBox[]): RectBox | null {
    if (rects.length === 0) return null;
    let minL = Infinity;
    let minT = Infinity;
    let maxR = -Infinity;
    let maxB = -Infinity;
    for (const r of rects) {
      if (r.left < minL) minL = r.left;
      if (r.top < minT) minT = r.top;
      if (r.left + r.width > maxR) maxR = r.left + r.width;
      if (r.top + r.height > maxB) maxB = r.top + r.height;
    }
    return {
      left: Math.max(0, minL),
      top: Math.max(0, minT),
      width: Math.min(viewportW, maxR) - Math.max(0, minL),
      height: Math.min(viewportH, maxB) - Math.max(0, minT),
    };
  }

  function toPaddedBox(el: Element, pad: number): RectBox {
    const r = el.getBoundingClientRect();
    const top = Math.max(4, r.top - pad);
    const left = Math.max(4, r.left - pad);
    const right = Math.min(viewportW - 4, r.right + pad);
    const bottom = Math.min(viewportH - 4, r.bottom + pad);
    return {
      top,
      left,
      width: Math.max(12, right - left),
      height: Math.max(12, bottom - top),
    };
  }

  function measureTarget() {
    if (typeof window === 'undefined') return;
    viewportW = window.innerWidth;
    viewportH = window.innerHeight;

    const hasFirstRun = Boolean(document.querySelector('.fr-backdrop'));
    firstRunModalOpen = hasFirstRun;
    if (hasFirstRun) {
      targetRect = null;
      interactiveHole = null;
      if (step === 'run-workflow') {
        onDismiss();
      }
      return;
    }

    if (step === 'welcome') {
      targetRect = null;
      interactiveHole = null;
      subStage = 'button';
      return;
    }

    if (step === 'add-mpnn' || step === 'add-fold' || step === 'add-scorer') {
      const cfg = ADD_STEP_TARGETS[step];
      const addBtn = document.querySelector<HTMLElement>('[data-tour="add-block-btn"]');
      const catalogShell = document.querySelector<HTMLElement>('.catalog-shell');
      const subEl = catalogShell?.querySelector<HTMLElement>(`[data-sub-key="${cfg.subKey}"]`) ?? null;
      const catEl = catalogShell?.querySelector<HTMLElement>(`[data-category-id="${cfg.categoryId}"]`) ?? null;

      if (subEl) {
        subStage = 'item';
        const box = toPaddedBox(subEl, 4);
        targetRect = box;
        // Keep the entire open catalog + flyout interactive so hovering across items never gets blocked
        const holeEls: RectBox[] = [box];
        if (addBtn) holeEls.push(toPaddedBox(addBtn, 4));
        const mainPane = catalogShell?.querySelector<HTMLElement>('.catalog-main');
        const flyoutPane = catalogShell?.querySelector<HTMLElement>('.catalog-flyout');
        if (mainPane) holeEls.push(toPaddedBox(mainPane, 4));
        if (flyoutPane) holeEls.push(toPaddedBox(flyoutPane, 4));
        interactiveHole = unionRects(holeEls);
        return;
      }

      if (catEl) {
        subStage = 'category';
        const box = toPaddedBox(catEl, 4);
        targetRect = box;
        const holeEls: RectBox[] = [box];
        if (addBtn) holeEls.push(toPaddedBox(addBtn, 4));
        const mainPane = catalogShell?.querySelector<HTMLElement>('.catalog-main');
        const flyoutPane = catalogShell?.querySelector<HTMLElement>('.catalog-flyout');
        if (mainPane) holeEls.push(toPaddedBox(mainPane, 4));
        if (flyoutPane) holeEls.push(toPaddedBox(flyoutPane, 4));
        interactiveHole = unionRects(holeEls);
        return;
      }

      subStage = 'button';
      if (addBtn) {
        const box = toPaddedBox(addBtn, 6);
        targetRect = box;
        interactiveHole = box;
      } else {
        targetRect = null;
        interactiveHole = null;
      }
      return;
    }

    subStage = 'button';
    const sel = step === 'inspect-params' ? '[data-tour="inspector-panel"]' : '[data-tour="run-btn"]';
    const el = document.querySelector<HTMLElement>(sel);
    if (!el) {
      targetRect = null;
      interactiveHole = null;
      return;
    }
    const box = toPaddedBox(el, step === 'inspect-params' ? 4 : 6);
    targetRect = box;
    interactiveHole = box;
  }

  function startInteractiveTour() {
    onPrepareTourStarter();
    step = 'add-mpnn';
    void tick().then(() => {
      requestAnimationFrame(measureTarget);
    });
  }

  // Live auto-advance when user clicks real blocks in + Add Block or clicks Run!
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

  // Re-measure spotlight whenever step or graph changes; scroll target into view on Step 4
  $effect(() => {
    const curStep = step;
    void graph.nodes.length;
    if (typeof window === 'undefined') return;
    let settleTimer: ReturnType<typeof setTimeout> | null = null;
    if (curStep === 'inspect-params' && lastScrolledStep !== 'inspect-params') {
      lastScrolledStep = curStep;
      const el = document.querySelector<HTMLElement>('[data-tour="inspector-panel"]');
      if (el) {
        el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        settleTimer = setTimeout(measureTarget, 320);
      }
    } else {
      lastScrolledStep = curStep;
    }
    const raf = requestAnimationFrame(measureTarget);
    return () => {
      cancelAnimationFrame(raf);
      if (settleTimer) clearTimeout(settleTimer);
    };
  });

  // Position the coachmark popover RIGHT NEXT TO the spotlighted element (where the cursor should go!)
  let coachLayout = $derived.by((): { style: string; caret: CaretSide; caretOffset: number } => {
    const cardW = Math.min(336, Math.max(260, viewportW - 24));
    const estH = 215;
    const gap = 14;
    if (!targetRect) {
      return { style: `bottom: 24px; right: 24px; width: ${cardW}px;`, caret: 'none', caretOffset: 24 };
    }

    const r = targetRect;
    const targetCenterY = r.top + r.height / 2;
    const targetCenterX = r.left + r.width / 2;

    // Step 5 (▶ Run at top-right): anchor directly below ▶ Run with top caret pointing up at it
    if (step === 'run-workflow') {
      const top = Math.min(viewportH - estH - 12, r.top + r.height + gap);
      const left = Math.max(12, Math.min(viewportW - cardW - 12, r.left + r.width - cardW));
      const caretOffset = Math.max(18, Math.min(cardW - 26, targetCenterX - left - 6));
      return {
        style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
        caret: 'top',
        caretOffset: Math.round(caretOffset),
      };
    }

    // Step 4 (Inspector panel on right): anchor directly to the left of the panel with right caret
    if (step === 'inspect-params') {
      if (r.left >= cardW + gap + 12) {
        const left = r.left - cardW - gap;
        const top = Math.max(64, Math.min(viewportH - estH - 16, r.top + 22));
        const caretOffset = Math.max(20, Math.min(estH - 28, r.top + 48 - top));
        return {
          style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
          caret: 'right',
          caretOffset: Math.round(caretOffset),
        };
      }
      return {
        style: `bottom: 16px; left: 16px; width: ${cardW}px;`,
        caret: 'none',
        caretOffset: 24,
      };
    }

    // Steps 1–3 (+ Add Block -> Category -> Flyout Item):
    // Anchor directly to the RIGHT of the highlighted button/category/flyout item with a left caret!
    const anchorRightEdge =
      subStage === 'category' && interactiveHole
        ? Math.max(r.left + r.width, interactiveHole.left + interactiveHole.width)
        : r.left + r.width;

    if (anchorRightEdge + gap + cardW <= viewportW - 10) {
      const left = anchorRightEdge + gap;
      const top = Math.max(58, Math.min(viewportH - estH - 12, targetCenterY - 46));
      const caretOffset = Math.max(16, Math.min(estH - 28, targetCenterY - top - 6));
      const isFlushToTarget = anchorRightEdge - (r.left + r.width) < 12;
      return {
        style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
        caret: isFlushToTarget ? 'left' : 'none',
        caretOffset: Math.round(caretOffset),
      };
    }

    // Fallback if right side is tight: anchor directly to the LEFT of target (with right caret)
    if (r.left - gap - cardW >= 12) {
      const left = r.left - gap - cardW;
      const top = Math.max(58, Math.min(viewportH - estH - 12, targetCenterY - 46));
      const caretOffset = Math.max(16, Math.min(estH - 28, targetCenterY - top - 6));
      return {
        style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
        caret: 'right',
        caretOffset: Math.round(caretOffset),
      };
    }

    // Fallback if viewport is narrow: anchor directly BELOW the target (with top caret)
    if (r.top + r.height + gap + estH <= viewportH - 10) {
      const top = r.top + r.height + gap;
      const left = Math.max(12, Math.min(viewportW - cardW - 12, r.left));
      const caretOffset = Math.max(18, Math.min(cardW - 26, targetCenterX - left - 6));
      return {
        style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
        caret: 'top',
        caretOffset: Math.round(caretOffset),
      };
    }

    // Final fallback: above target
    const top = Math.max(12, r.top - estH - gap);
    const left = Math.max(12, Math.min(viewportW - cardW - 12, r.left));
    return {
      style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
      caret: 'none',
      caretOffset: 24,
    };
  });

  // 4-panel transparent click-shield around `interactiveHole` so stray clicks outside
  // the guided flow don't accidentally dismiss popovers or break the sequence
  let shieldRects = $derived.by(() => {
    const h = interactiveHole;
    if (!h) return null;
    const right = h.left + h.width;
    const bottom = h.top + h.height;
    return {
      top: { top: 0, left: 0, width: viewportW, height: Math.max(0, h.top) },
      bottom: { top: bottom, left: 0, width: viewportW, height: Math.max(0, viewportH - bottom) },
      left: { top: h.top, left: 0, width: Math.max(0, h.left), height: h.height },
      right: { top: h.top, left: right, width: Math.max(0, viewportW - right), height: h.height },
    };
  });

  let domObserver: MutationObserver | null = null;
  let measureRaf: number | null = null;

  function scheduleMeasure() {
    if (measureRaf !== null) return;
    measureRaf = requestAnimationFrame(() => {
      measureRaf = null;
      measureTarget();
    });
  }

  onMount(() => {
    cardEl?.focus({ preventScroll: true });
    measureTarget();
    window.addEventListener('resize', scheduleMeasure);
    window.addEventListener('scroll', scheduleMeasure, true);
    if (typeof MutationObserver !== 'undefined') {
      domObserver = new MutationObserver(scheduleMeasure);
      domObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'style'],
      });
    }
  });

  onDestroy(() => {
    onInteractiveChange?.(false);
    if (measureRaf !== null) cancelAnimationFrame(measureRaf);
    domObserver?.disconnect();
    domObserver = null;
    if (typeof window !== 'undefined') {
      window.removeEventListener('resize', scheduleMeasure);
      window.removeEventListener('scroll', scheduleMeasure, true);
    }
  });

  function handleKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      if (firstRunModalOpen || document.querySelector('.fr-backdrop')) {
        return;
      }
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
      use:modalFocusTrap
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
            Protein design and structure prediction, running locally on your GPU.
          </p>
        </div>
      </div>

      <div class="onb-video-wrap">
        <video
          class="onb-video"
          src="/demo-welcome.mp4"
          poster="/demo-welcome-poster.webp"
          autoplay
          muted
          loop
          playsinline
          controls
          preload="metadata"
        ></video>
      </div>

      <div class="gpu-notice" role="note">
        <svg class="warn-ico" viewBox="0 0 20 20" fill="none" aria-hidden="true">
          <path d="M10 2.6L18.2 16.8C18.55 17.4 18.12 18.15 17.42 18.15H2.58C1.88 18.15 1.45 17.4 1.8 16.8L10 2.6Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
          <path d="M10 7.6V11.8" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
          <circle cx="10" cy="14.7" r="1" fill="currentColor"/>
        </svg>
        <span>
          <b>GPU heads-up:</b> Initial model setup and structure prediction may take a moment and use GPU memory; weights stay cached for future runs.
        </span>
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
          Build your first workflow →
        </button>
      </div>
      <div class="reopen-foot">Reopen anytime from <b>Quick Tour</b></div>
    </div>
  </div>
{:else if !firstRunModalOpen}
  <!-- Click-constraining transparent scrim outside the active interactive hole -->
  {#if shieldRects}
    <div
      class="tour-shield"
      style="top: {shieldRects.top.top}px; left: {shieldRects.top.left}px; width: {shieldRects.top.width}px; height: {shieldRects.top.height}px;"
      aria-hidden="true"
    ></div>
    <div
      class="tour-shield"
      style="top: {shieldRects.bottom.top}px; left: {shieldRects.bottom.left}px; width: {shieldRects.bottom.width}px; height: {shieldRects.bottom.height}px;"
      aria-hidden="true"
    ></div>
    <div
      class="tour-shield"
      style="top: {shieldRects.left.top}px; left: {shieldRects.left.left}px; width: {shieldRects.left.width}px; height: {shieldRects.left.height}px;"
      aria-hidden="true"
    ></div>
    <div
      class="tour-shield"
      style="top: {shieldRects.right.top}px; left: {shieldRects.right.left}px; width: {shieldRects.right.width}px; height: {shieldRects.right.height}px;"
      aria-hidden="true"
    ></div>
  {/if}

  <!-- Tight glowing spotlight cutout directly on the exact button / category / model item to click -->
  {#if targetRect}
    <div
      class="spotlight-ring"
      style="top: {targetRect.top}px; left: {targetRect.left}px; width: {targetRect.width}px; height: {targetRect.height}px;"
      aria-hidden="true"
    ></div>
  {/if}

  <!-- Anchored Coachmark Popover right next to the highlighted element -->
  <div
    class="coach-card"
    style={coachLayout.style}
    role="dialog"
    aria-label="Interactive walkthrough"
  >
    {#if coachLayout.caret === 'left'}
      <span class="caret caret-left" style="top: {coachLayout.caretOffset}px" aria-hidden="true"></span>
    {:else if coachLayout.caret === 'right'}
      <span class="caret caret-right" style="top: {coachLayout.caretOffset}px" aria-hidden="true"></span>
    {:else if coachLayout.caret === 'top'}
      <span class="caret caret-top" style="left: {coachLayout.caretOffset}px" aria-hidden="true"></span>
    {/if}

    <div class="coach-top">
      <div class="coach-dots" aria-label={`Step ${interactiveIdx + 1} of ${INTERACTIVE_STEPS.length}`}>
        {#each INTERACTIVE_STEPS as s, i (s)}
          <span class="c-dot" class:done={i < interactiveIdx} class:active={i === interactiveIdx}></span>
        {/each}
        <span class="c-count">{interactiveIdx + 1}/{INTERACTIVE_STEPS.length}</span>
      </div>
      <button type="button" class="coach-x" onclick={onDismiss} title="Exit tour (Esc)" aria-label="Exit tour">
        ×
      </button>
    </div>

    {#if step === 'add-mpnn'}
      {#if subStage === 'item'}
        <div class="coach-title">Click <b>ProteinMPNN (Cα-only)</b></div>
        <p class="coach-body">
          Click <b>Cα-ProteinMPNN</b> to design sequences directly from Genie 3's Cα backbone trace — it connects automatically.
        </p>
      {:else if subStage === 'category'}
        <div class="coach-title">Hover <b>Sequence Design ›</b></div>
        <p class="coach-body">
          Open the highlighted <b>Sequence Design</b> category to select <b>Cα-ProteinMPNN</b> for Genie 3's Cα trace.
        </p>
      {:else}
        <div class="coach-title">1. Click <b>+ Add Block</b></div>
        <p class="coach-body">
          We placed <b>Genie 3</b> on the canvas to generate a de novo Cα backbone trace. Click <b>+ Add Block</b> (or press <kbd>/</kbd>) to add <b>Cα-ProteinMPNN</b> next.
        </p>
      {/if}
      <div class="coach-actions">
        <button type="button" class="coach-link" onclick={onDismiss}>Skip tour</button>
        <button type="button" class="coach-btn" onclick={() => onAutoStepBlock('web-protein-mpnn')}>
          Do it for me →
        </button>
      </div>

    {:else if step === 'add-fold'}
      <div class="coach-ok">✓ Cα-ProteinMPNN connected</div>
      {#if subStage === 'item'}
        <div class="coach-title">Click <b>AlphaFold2 · Model 1</b></div>
        <p class="coach-body">
          Click to refold Cα-ProteinMPNN's designed sequences and check whether they adopt the Genie 3 backbone.
        </p>
      {:else if subStage === 'category'}
        <div class="coach-title">Hover <b>Structure Prediction ›</b></div>
        <p class="coach-body">
          Hover the highlighted category to pick from LocalFold models (AF2, AF2-Multimer, RF3, Boltz-2, AF3, ESMFold2 &amp; more).
        </p>
      {:else}
        <div class="coach-title">2. Add Structure Prediction</div>
        <p class="coach-body">
          Click <b>+ Add Block</b> again to add a structure predictor and verify your designed sequences fold properly.
        </p>
      {/if}
      <div class="coach-actions">
        <button type="button" class="coach-link" onclick={onDismiss}>Skip tour</button>
        <button type="button" class="coach-btn" onclick={() => onAutoStepBlock('localfold-af2')}>
          Do it for me →
        </button>
      </div>

    {:else if step === 'add-scorer'}
      <div class="coach-ok">✓ LocalFold connected</div>
      {#if subStage === 'item'}
        <div class="coach-title">Click <b>scRMSD / scTM</b></div>
        <p class="coach-body">
          Click to drop the designability scorer — it wires both your Genie 3 backbone and predicted folds automatically.
        </p>
      {:else if subStage === 'category'}
        <div class="coach-title">Hover <b>Evaluation ›</b></div>
        <p class="coach-body">
          Hover <b>Evaluation</b> to add Kabsch alignment &amp; designability scoring.
        </p>
      {:else}
        <div class="coach-title">3. Add Designability Scoring</div>
        <p class="coach-body">
          One last block: click <b>+ Add Block</b> to close the loop with <b>scRMSD / scTM</b> evaluation.
        </p>
      {/if}
      <div class="coach-actions">
        <button type="button" class="coach-link" onclick={onDismiss}>Skip tour</button>
        <button type="button" class="coach-btn" onclick={() => onAutoStepBlock('designability-scorer')}>
          Do it for me →
        </button>
      </div>

    {:else if step === 'inspect-params'}
      <div class="coach-ok">✓ De novo pipeline ready</div>
      <div class="coach-title">4. Configure &amp; Peek Intermediates</div>
      <p class="coach-body">
        Adjust protein length, diffusion steps, or model checkpoints here. After running, open <b>Outputs</b> or click any colored wire to inspect intermediates, or <kbd>⌘</kbd>/<kbd>Ctrl</kbd>-click blocks to compare in 3D.
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
        Click <b>▶ Run</b> above (or press <kbd>R</kbd>) to execute locally on your GPU — weights download once and stay cached in your browser.
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
    -webkit-backdrop-filter: blur(5px);
    backdrop-filter: blur(5px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    box-sizing: border-box;
  }

  .onb-card {
    width: 100%;
    max-width: min(660px, calc(100vw - 24px));
    max-height: calc(100vh - 24px);
    overflow-y: auto;
    box-sizing: border-box;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.38);
    padding: 22px 24px 16px;
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 13px;
    outline: none;
  }

  .onb-video-wrap {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border-strong);
    background: #0b1020;
  }

  .onb-video {
    display: block;
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: cover;
    background: #0b1020;
  }

  .onb-hero {
    display: flex;
    align-items: flex-start;
    gap: 15px;
  }

  .onb-mark {
    width: 48px;
    height: 48px;
    flex-shrink: 0;
    filter: drop-shadow(0 6px 14px rgba(37, 99, 235, 0.24));
  }

  h2 {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-primary);
  }

  .brand-accent {
    color: var(--accent-bright);
  }

  .onb-sub {
    margin: 4px 0 0;
    font-size: 14.5px;
    line-height: 1.45;
    color: var(--text-muted);
  }

  .gpu-notice {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 9px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
    font-size: 13px;
    line-height: 1.45;
  }

  .gpu-notice b {
    font-weight: 600;
  }

  .warn-ico {
    width: 18px;
    height: 18px;
    flex-shrink: 0;
    margin-top: 1px;
  }

  .onb-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px 12px;
    padding-top: 2px;
  }

  @media (max-width: 480px) {
    .onb-backdrop {
      padding: 12px;
    }
    .onb-card {
      padding: 18px 16px 14px;
      gap: 12px;
    }
    .onb-footer {
      justify-content: stretch;
    }
    .btn-skip,
    .btn-start {
      flex: 1 1 auto;
      text-align: center;
      white-space: normal;
    }
  }

  .btn-skip {
    background: transparent;
    border: 1px solid var(--border-strong);
    color: var(--text-secondary);
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 14px;
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
    padding: 10px 20px;
    font-size: 14.5px;
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
    font-size: 10.5px;
    color: var(--text-dim);
    opacity: 0.75;
    text-align: center;
    margin-top: -4px;
  }

  .reopen-foot b {
    color: var(--text-muted);
    font-weight: 600;
  }

  /* ── Interactive Spotlight Overlay + Gentle Click Constraint ── */
  .tour-shield {
    position: fixed;
    z-index: 190;
    background: transparent;
    pointer-events: auto;
    cursor: default;
  }

  .spotlight-ring {
    position: fixed;
    z-index: 195;
    border-radius: 9px;
    border: 2px solid var(--accent-bright, #60a5fa);
    box-shadow:
      0 0 0 4px rgba(59, 130, 246, 0.32),
      0 0 0 9999px rgba(10, 14, 24, 0.56);
    pointer-events: none;
    transition:
      top 220ms cubic-bezier(0.22, 1, 0.36, 1),
      left 220ms cubic-bezier(0.22, 1, 0.36, 1),
      width 220ms cubic-bezier(0.22, 1, 0.36, 1),
      height 220ms cubic-bezier(0.22, 1, 0.36, 1);
    animation: tour-pulse 1.8s ease-in-out infinite;
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
      0 18px 48px rgba(0, 0, 0, 0.38),
      0 0 0 1px rgba(59, 130, 246, 0.22);
    padding: 13px 15px 12px;
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 7px;
    box-sizing: border-box;
    transition:
      top 220ms cubic-bezier(0.22, 1, 0.36, 1),
      left 220ms cubic-bezier(0.22, 1, 0.36, 1);
  }

  .caret {
    position: absolute;
    width: 10px;
    height: 10px;
    background: var(--bg-surface);
    border-left: 1.5px solid var(--accent);
    border-top: 1.5px solid var(--accent);
    pointer-events: none;
  }

  .caret-left {
    left: -6px;
    transform: rotate(-45deg);
  }

  .caret-right {
    right: -6px;
    transform: rotate(135deg);
  }

  .caret-top {
    top: -6px;
    transform: rotate(45deg);
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
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: var(--border-strong);
    transition: background 180ms ease, width 180ms ease;
  }

  .c-dot.done {
    background: #10b981;
  }

  .c-dot.active {
    width: 14px;
    background: var(--accent);
  }

  .c-count {
    margin-left: 4px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-dim);
  }

  .coach-x {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 18px;
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
    font-size: 11px;
    font-weight: 600;
    color: var(--ok-text, #10b981);
    background: var(--ok-bg);
    border: 1px solid var(--ok-border);
    padding: 2px 7px;
    border-radius: 5px;
    align-self: flex-start;
  }

  .coach-title {
    font-size: 14.5px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .coach-title b {
    color: var(--accent-bright);
    font-weight: 700;
  }

  .coach-body {
    margin: 0;
    font-size: 13px;
    line-height: 1.48;
    color: var(--text-secondary);
  }

  .coach-body b {
    color: var(--text-primary);
    font-weight: 600;
  }

  kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11.5px;
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
    font-size: 12.5px;
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
    font-size: 12.5px;
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
    .spotlight-ring,
    .coach-card {
      animation: none;
      transition: none;
    }
  }
</style>
