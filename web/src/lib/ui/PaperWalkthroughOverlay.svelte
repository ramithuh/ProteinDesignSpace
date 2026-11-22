<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { PaperInsightExample } from '$lib/engine/paper-gallery';

  interface Props {
    example: PaperInsightExample;
    stepIdx: number;
    hasOutputs: boolean;
    isRunning: boolean;
    canRun: boolean;
    onSelectStep: (idx: number) => void;
    onRun: () => void;
    onDismiss: () => void;
  }

  let {
    example,
    stepIdx,
    hasOutputs,
    isRunning,
    canRun,
    onSelectStep,
    onRun,
    onDismiss,
  }: Props = $props();

  interface RectBox {
    top: number;
    left: number;
    width: number;
    height: number;
  }

  type CaretSide = 'left' | 'right' | 'top' | 'bottom' | 'none';

  let targetRect = $state<RectBox | null>(null);
  let viewportW = $state(typeof window !== 'undefined' ? window.innerWidth : 1280);
  let viewportH = $state(typeof window !== 'undefined' ? window.innerHeight : 800);
  let coachEl = $state<HTMLDivElement | null>(null);
  let coachH = $state(200);

  let steps = $derived(example.walkthrough);
  let curStep = $derived(steps[stepIdx] ?? steps[0]);
  let isLastStep = $derived(stepIdx >= steps.length - 1);

  let targetSelector = $derived.by(() => {
    const tgt = curStep?.spotlightTarget ?? 'canvas';
    if (tgt === 'canvas') return '[data-tour="workflow-canvas"]';
    if (tgt === 'run') return '[data-tour="run-btn"]';
    if (tgt === 'viewer') return '[data-tour="viewer-panel"]';
    return '[data-tour="inspector-panel"]';
  });

  function isStepAccessible(idx: number): boolean {
    const st = steps[idx];
    if (!st) return false;
    return st.phase !== 'result' || hasOutputs;
  }

  function toPaddedBox(el: HTMLElement, pad: number): RectBox {
    const r = el.getBoundingClientRect();
    const top = Math.max(2, r.top - pad);
    const left = Math.max(2, r.left - pad);
    const width = Math.min(viewportW - left - 2, r.width + pad * 2);
    const height = Math.min(viewportH - top - 2, r.height + pad * 2);
    return { top, left, width: Math.max(0, width), height: Math.max(0, height) };
  }

  function measureTarget() {
    if (typeof window === 'undefined' || isRunning) {
      targetRect = null;
      return;
    }
    viewportW = window.innerWidth;
    viewportH = window.innerHeight;
    if (coachEl) {
      const h = coachEl.offsetHeight;
      if (h > 60) coachH = h;
    }
    const el = document.querySelector<HTMLElement>(targetSelector);
    if (!el) {
      targetRect = null;
      return;
    }
    const pad = curStep?.spotlightTarget === 'run' ? 6 : 4;
    targetRect = toPaddedBox(el, pad);
  }

  let resizeObserver: ResizeObserver | null = null;
  let measureRaf: number | null = null;

  function scheduleMeasure() {
    if (measureRaf !== null) return;
    measureRaf = requestAnimationFrame(() => {
      measureRaf = null;
      measureTarget();
    });
  }

  $effect(() => {
    void example.id;
    void stepIdx;
    void hasOutputs;
    void isRunning;
    const sel = targetSelector;
    if (typeof window === 'undefined') return;

    resizeObserver?.disconnect();
    resizeObserver = null;

    if (!isRunning && typeof ResizeObserver !== 'undefined') {
      const el = document.querySelector<HTMLElement>(sel);
      if (el) {
        resizeObserver = new ResizeObserver(scheduleMeasure);
        resizeObserver.observe(el);
      }
    }

    const raf = requestAnimationFrame(measureTarget);
    const settle = setTimeout(measureTarget, 120);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(settle);
    };
  });

  let coachLayout = $derived.by((): { style: string; caret: CaretSide; caretOffset: number } => {
    const cardW = Math.min(360, Math.max(272, viewportW - 24));
    const estH = Math.max(155, coachH);
    const gap = 14;
    const tgt = curStep?.spotlightTarget ?? 'canvas';

    if (!targetRect) {
      return {
        style: `bottom: 20px; left: 20px; width: ${cardW}px;`,
        caret: 'none',
        caretOffset: 24,
      };
    }

    const r = targetRect;
    const targetCenterX = r.left + r.width / 2;

    // Step targeting ▶ Run at top-right: anchor directly below ▶ Run with top caret
    if (tgt === 'run') {
      const top = Math.min(viewportH - estH - 12, r.top + r.height + gap);
      const left = Math.max(12, Math.min(viewportW - cardW - 12, r.left + r.width - cardW));
      const caretOffset = Math.max(18, Math.min(cardW - 26, targetCenterX - left - 6));
      return {
        style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
        caret: 'top',
        caretOffset: Math.round(caretOffset),
      };
    }

    // Step targeting Workflow Canvas on left: anchor directly to the RIGHT with left caret
    if (tgt === 'canvas') {
      const rightEdge = r.left + r.width;
      if (rightEdge + gap + cardW <= viewportW - 12) {
        const left = rightEdge + gap;
        const anchorY = r.top + Math.min(180, Math.max(88, r.height * 0.28));
        const top = Math.max(64, Math.min(viewportH - estH - 16, anchorY - 46));
        const caretOffset = Math.max(20, Math.min(estH - 28, anchorY - top - 6));
        return {
          style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
          caret: 'left',
          caretOffset: Math.round(caretOffset),
        };
      }
      // Vertical split / narrow fallback: directly below canvas
      if (r.top + r.height + gap + estH <= viewportH - 10) {
        const top = r.top + r.height + gap;
        const left = Math.max(12, Math.min(viewportW - cardW - 12, targetCenterX - cardW / 2));
        const caretOffset = Math.max(18, Math.min(cardW - 26, targetCenterX - left - 6));
        return {
          style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
          caret: 'top',
          caretOffset: Math.round(caretOffset),
        };
      }
      return {
        style: `bottom: 16px; right: 16px; width: ${cardW}px;`,
        caret: 'none',
        caretOffset: 24,
      };
    }

    // Steps targeting 3D Viewer or Inspector panel on right: float to the LEFT with right caret
    if (r.left >= cardW + gap + 12) {
      const left = r.left - cardW - gap;
      const anchorY = r.top + Math.min(86, Math.max(36, r.height * 0.24));
      const top = Math.max(60, Math.min(viewportH - estH - 16, anchorY - 40));
      const caretOffset = Math.max(20, Math.min(estH - 28, anchorY - top - 6));
      return {
        style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
        caret: 'right',
        caretOffset: Math.round(caretOffset),
      };
    }

    // Fallback (vertical split / narrow viewport): place above or below target
    if (r.top >= estH + gap + 12) {
      const top = r.top - estH - gap;
      const left = Math.max(12, Math.min(viewportW - cardW - 12, targetCenterX - cardW / 2));
      const caretOffset = Math.max(18, Math.min(cardW - 26, targetCenterX - left - 6));
      return {
        style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
        caret: 'bottom',
        caretOffset: Math.round(caretOffset),
      };
    }

    if (r.top + r.height + gap + estH <= viewportH - 10) {
      const top = r.top + r.height + gap;
      const left = Math.max(12, Math.min(viewportW - cardW - 12, targetCenterX - cardW / 2));
      const caretOffset = Math.max(18, Math.min(cardW - 26, targetCenterX - left - 6));
      return {
        style: `top: ${Math.round(top)}px; left: ${Math.round(left)}px; width: ${cardW}px;`,
        caret: 'top',
        caretOffset: Math.round(caretOffset),
      };
    }

    return {
      style: `bottom: 16px; left: 16px; width: ${cardW}px;`,
      caret: 'none',
      caretOffset: 24,
    };
  });

  onMount(() => {
    measureTarget();
    window.addEventListener('resize', scheduleMeasure);
    window.addEventListener('scroll', scheduleMeasure, true);
  });

  onDestroy(() => {
    if (measureRaf !== null) cancelAnimationFrame(measureRaf);
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (typeof window !== 'undefined') {
      window.removeEventListener('resize', scheduleMeasure);
      window.removeEventListener('scroll', scheduleMeasure, true);
    }
  });

  function isEditableElement(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    const tag = target.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
  }

  function handleKeydown(ev: KeyboardEvent) {
    if (isRunning) return;
    if (ev.key === 'Escape') {
      if (document.querySelector('.fr-backdrop')) return;
      ev.preventDefault();
      onDismiss();
      return;
    }
    if (isEditableElement(ev.target)) return;
    if (ev.key === 'ArrowRight' && !isLastStep && isStepAccessible(stepIdx + 1)) {
      ev.preventDefault();
      onSelectStep(stepIdx + 1);
    } else if (ev.key === 'ArrowLeft' && stepIdx > 0 && isStepAccessible(stepIdx - 1)) {
      ev.preventDefault();
      onSelectStep(stepIdx - 1);
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if !isRunning && curStep}
  <!-- Non-blocking glowing spotlight cutout (users can freely orbit 3D Mol*, scroll tables, or edit sequences) -->
  {#if targetRect}
    <div
      class="spotlight-ring"
      style="top: {targetRect.top}px; left: {targetRect.left}px; width: {targetRect.width}px; height: {targetRect.height}px;"
      aria-hidden="true"
    ></div>
  {/if}

  <!-- Anchored Coachmark Popover -->
  <div
    bind:this={coachEl}
    class="coach-card"
    style={coachLayout.style}
    role="dialog"
    aria-label="Literature example walkthrough"
  >
    {#if coachLayout.caret === 'left'}
      <span class="caret caret-left" style="top: {coachLayout.caretOffset}px" aria-hidden="true"></span>
    {:else if coachLayout.caret === 'right'}
      <span class="caret caret-right" style="top: {coachLayout.caretOffset}px" aria-hidden="true"></span>
    {:else if coachLayout.caret === 'top'}
      <span class="caret caret-top" style="left: {coachLayout.caretOffset}px" aria-hidden="true"></span>
    {:else if coachLayout.caret === 'bottom'}
      <span class="caret caret-bottom" style="left: {coachLayout.caretOffset}px" aria-hidden="true"></span>
    {/if}

    <div class="coach-top">
      <div class="coach-dots" role="tablist" aria-label={`Step ${stepIdx + 1} of ${steps.length}`}>
        {#each steps as st, i (st.title)}
          {@const locked = !isStepAccessible(i)}
          <button
            type="button"
            role="tab"
            class="c-dot"
            class:done={i < stepIdx}
            class:active={i === stepIdx}
            class:locked={locked}
            disabled={locked}
            aria-selected={i === stepIdx}
            aria-label={`Step ${i + 1}: ${st.title}`}
            title={locked ? 'Run the workflow first to inspect results' : `${i + 1}. ${st.title}`}
            onclick={() => onSelectStep(i)}
          ></button>
        {/each}
        <span class="c-count">{stepIdx + 1}/{steps.length}</span>
        <span class="c-cite" title={example.citation}>· {example.citation}</span>
      </div>
      <button
        type="button"
        class="coach-x"
        onclick={onDismiss}
        title="Hide walkthrough (Esc)"
        aria-label="Hide walkthrough"
      >
        ×
      </button>
    </div>

    <div class="coach-title">{curStep.title}</div>
    <p class="coach-body">{curStep.body}</p>

    <div class="coach-actions">
      {#if stepIdx === 0}
        <button type="button" class="coach-link" onclick={onDismiss}>Dismiss</button>
      {:else}
        <button
          type="button"
          class="coach-btn-subtle"
          onclick={() => onSelectStep(stepIdx - 1)}
        >
          ← Back
        </button>
      {/if}

      {#if curStep.phase === 'run' && !hasOutputs}
        <button type="button" class="coach-btn primary" disabled={!canRun} onclick={onRun}>
          Run →
        </button>
      {:else if isLastStep}
        <button type="button" class="coach-btn primary" onclick={onDismiss}>
          Done
        </button>
      {:else}
        <button
          type="button"
          class="coach-btn primary"
          onclick={() => onSelectStep(stepIdx + 1)}
        >
          Next →
        </button>
      {/if}
    </div>
  </div>
{/if}

<style>
  .spotlight-ring {
    position: fixed;
    z-index: 185;
    border-radius: 9px;
    border: 2px solid var(--accent-bright, #60a5fa);
    box-shadow:
      0 0 0 4px rgba(59, 130, 246, 0.28),
      0 0 0 9999px rgba(10, 14, 24, 0.34);
    pointer-events: none;
    transition:
      top 220ms cubic-bezier(0.22, 1, 0.36, 1),
      left 220ms cubic-bezier(0.22, 1, 0.36, 1),
      width 220ms cubic-bezier(0.22, 1, 0.36, 1),
      height 220ms cubic-bezier(0.22, 1, 0.36, 1);
    animation: pw-pulse 1.8s ease-in-out infinite;
  }

  @keyframes pw-pulse {
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
    z-index: 210;
    background: var(--bg-surface);
    border: 1.5px solid var(--accent);
    border-radius: 12px;
    box-shadow:
      0 18px 48px rgba(0, 0, 0, 0.32),
      0 0 0 1px rgba(59, 130, 246, 0.2);
    padding: 13px 15px 12px;
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 8px;
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

  .caret-bottom {
    bottom: -6px;
    transform: rotate(-135deg);
  }

  .coach-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    min-width: 0;
  }

  .coach-dots {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
    overflow: hidden;
  }

  .c-dot {
    width: 7px;
    height: 7px;
    padding: 0;
    border: none;
    border-radius: 999px;
    background: var(--border-strong);
    cursor: pointer;
    flex-shrink: 0;
    transition: background 180ms ease, width 180ms ease;
  }

  .c-dot:hover:not(:disabled) {
    background: var(--text-muted);
  }

  .c-dot.done {
    background: #10b981;
  }

  .c-dot.active {
    width: 16px;
    background: var(--accent);
  }

  .c-dot.locked {
    opacity: 0.38;
    cursor: not-allowed;
  }

  .c-count {
    margin-left: 3px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-dim);
    flex-shrink: 0;
  }

  .c-cite {
    font-size: 11px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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
    flex-shrink: 0;
  }

  .coach-x:hover {
    color: var(--text-primary);
    background: var(--bg-surface-hover);
  }

  .coach-title {
    font-size: 14px;
    font-weight: 600;
    line-height: 1.35;
    color: var(--text-primary);
  }

  .coach-body {
    margin: 0;
    font-size: 12.8px;
    line-height: 1.5;
    color: var(--text-secondary);
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

  .coach-btn-subtle {
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
  }

  .coach-btn-subtle:hover:not(:disabled) {
    color: var(--text-primary);
    border-color: var(--accent);
    background: var(--bg-surface-hover);
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

  .coach-btn.primary {
    background: var(--accent);
    color: #fff;
  }

  .coach-btn.primary:hover:not(:disabled) {
    filter: brightness(1.08);
  }

  .coach-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  @media (prefers-reduced-motion: reduce) {
    .spotlight-ring,
    .coach-card {
      animation: none;
      transition: none;
    }
  }
</style>
