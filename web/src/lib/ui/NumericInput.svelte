<script module lang="ts">
  let invalidNumericCount = $state(0);

  export function hasInvalidNumericInputs(): boolean {
    return invalidNumericCount > 0;
  }
</script>

<script lang="ts">
  import { untrack } from 'svelte';
  import {
    clampNumeric,
    commitNumericDraft,
    formatNumericValue,
    sanitizeNumericDraft,
    validateNumericDraft,
    type NumericFieldSpec,
  } from '$lib/engine/param-validation';

  interface Props {
    value: number;
    spec: NumericFieldSpec;
    disabled?: boolean;
    ariaLabel?: string;
    name?: string;
    id?: string;
    onChange: (next: number) => void;
  }

  let {
    value,
    spec,
    disabled = false,
    ariaLabel = undefined,
    name = undefined,
    id = undefined,
    onChange,
  }: Props = $props();

  let resolvedName = $derived(
    name ??
      (ariaLabel
        ? `num-${ariaLabel.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')}`
        : 'numeric-param'),
  );

  let draft = $state(untrack(() => formatNumericValue(value, spec)));
  let focused = $state(false);

  // Sync draft from external value changes (preset switches, node switches, stepper clicks)
  $effect(() => {
    const formatted = formatNumericValue(value, spec);
    if (!focused) {
      draft = formatted;
    }
  });

  let feedback = $derived(validateNumericDraft(draft, spec));
  let isInvalid = $derived(feedback.status === 'error');

  $effect(() => {
    if (!isInvalid) return;
    invalidNumericCount += 1;
    return () => {
      invalidNumericCount = Math.max(0, invalidNumericCount - 1);
    };
  });

  function handleInput(ev: Event) {
    const el = ev.currentTarget as HTMLInputElement;
    const cleaned = sanitizeNumericDraft(el.value, Boolean(spec.integer));
    if (el.value !== cleaned) {
      el.value = cleaned;
    }
    draft = cleaned;

    const check = validateNumericDraft(cleaned, spec);
    if ((check.status === 'ok' || check.status === 'warn') && check.value !== undefined) {
      onChange(spec.integer ? Math.round(check.value) : check.value);
    }
  }

  function commit() {
    const committed = commitNumericDraft(draft, spec, value);
    draft = formatNumericValue(committed, spec);
    if (committed !== value) {
      onChange(committed);
    }
  }

  function handleBlur() {
    focused = false;
    commit();
  }

  function stepBy(delta: number) {
    if (disabled) return;
    const base = Number.isFinite(Number(draft)) && draft.trim() !== '' ? Number(draft) : value;
    const step = spec.step ?? 1;
    const next = clampNumeric(base + delta * step, spec);
    draft = formatNumericValue(next, spec);
    onChange(next);
  }

  function handleKeydown(ev: KeyboardEvent) {
    if (ev.key === 'ArrowUp') {
      ev.preventDefault();
      stepBy(1);
    } else if (ev.key === 'ArrowDown') {
      ev.preventDefault();
      stepBy(-1);
    } else if (ev.key === 'Enter' && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
      commit();
      (ev.currentTarget as HTMLInputElement).blur();
    } else if (ev.key === 'Escape') {
      draft = formatNumericValue(value, spec);
      (ev.currentTarget as HTMLInputElement).blur();
    }
  }
</script>

<div class="num-wrap">
  <div
    class="num-box"
    class:is-error={feedback.status === 'error'}
    class:is-warn={feedback.status === 'warn'}
    class:is-disabled={disabled}
  >
    <input
      type="text"
      class="num-input"
      {id}
      name={resolvedName}
      inputmode={spec.integer ? 'numeric' : 'decimal'}
      autocomplete="off"
      spellcheck="false"
      aria-label={ariaLabel}
      aria-invalid={isInvalid}
      {disabled}
      value={draft}
      onfocus={() => (focused = true)}
      oninput={handleInput}
      onblur={handleBlur}
      onkeydown={handleKeydown}
    />
    <div class="steppers" aria-hidden="true">
      <button
        type="button"
        class="step-btn"
        tabindex="-1"
        {disabled}
        onclick={() => stepBy(1)}
        title={`Increase by ${spec.step ?? 1}`}
      >▲</button>
      <button
        type="button"
        class="step-btn"
        tabindex="-1"
        {disabled}
        onclick={() => stepBy(-1)}
        title={`Decrease by ${spec.step ?? 1}`}
      >▼</button>
    </div>
  </div>
  {#if feedback.message}
    <span
      class="num-hint"
      class:err={feedback.status === 'error'}
      class:warn={feedback.status === 'warn'}
    >
      {feedback.message}
    </span>
  {/if}
</div>

<style>
  .num-wrap {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .num-box {
    display: inline-flex;
    align-items: stretch;
    background: var(--bg-input);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    overflow: hidden;
    width: 88px;
    transition: border-color 120ms ease, box-shadow 120ms ease;
  }
  .num-box:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 18%, transparent);
  }
  .num-box.is-error {
    border-color: var(--err-strong);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--err-strong) 18%, transparent);
  }
  .num-box.is-warn {
    border-color: var(--warn-border);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--warn-border) 22%, transparent);
  }
  .num-box.is-disabled {
    opacity: 0.6;
  }
  input {
    width: 100%;
    min-width: 0;
    border: none;
    background: transparent;
    color: var(--text-primary);
    padding: 6px 4px 6px 8px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
    outline: none;
  }
  .steppers {
    display: flex;
    flex-direction: column;
    border-left: 1px solid var(--border-subtle);
    width: 18px;
    flex-shrink: 0;
  }
  .step-btn {
    flex: 1;
    border: none;
    background: var(--bg-surface-2);
    color: var(--text-muted);
    font-size: 7px;
    line-height: 1;
    padding: 0;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .step-btn:first-child {
    border-bottom: 1px solid var(--border-subtle);
  }
  .step-btn:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }
  .step-btn:disabled {
    cursor: not-allowed;
  }
  .num-hint {
    font-size: 10px;
    line-height: 1.3;
    max-width: 190px;
  }
  .num-hint.err {
    color: var(--err-strong);
  }
  .num-hint.warn {
    color: var(--warn-text);
  }
</style>
