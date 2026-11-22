<script lang="ts">
  import NumericInput from './NumericInput.svelte';
  import { fieldToNumericSpec } from '$lib/engine/param-schema';
  import {
    isValidResidueToken,
    validateTiedPositions,
    type NumericFieldSpec,
  } from '$lib/engine/param-validation';
  import type { AdapterParamField, AdapterParamOption } from '$lib/engine/types';

  interface Props {
    fields: AdapterParamField[];
    params: Record<string, unknown>;
    disabled?: boolean;
    /** Optional per-field select option overrides (e.g. family-dependent MPNN checkpoints). */
    optionOverrides?: Record<string, AdapterParamOption[]>;
    /** Optional per-field NumericFieldSpec overrides (e.g. dynamic max or warnBelow). */
    numericSpecOverrides?: Record<string, Partial<NumericFieldSpec>>;
    /** Optional field keys to force into disabled/muted state. */
    disabledKeys?: readonly string[];
    onChange: (key: string, value: unknown) => void;
  }

  let {
    fields,
    params,
    disabled = false,
    optionOverrides = undefined,
    numericSpecOverrides = undefined,
    disabledKeys = [],
    onChange,
  }: Props = $props();

  const VALID_AA_SET = new Set('ACDEFGHIKLMNPQRSTVWYX'.split(''));

  function isFieldDisabled(key: string): boolean {
    return disabled || disabledKeys.includes(key);
  }

  function resolveNumericSpec(field: AdapterParamField): NumericFieldSpec {
    const base = fieldToNumericSpec(field);
    const override = numericSpecOverrides?.[field.key];
    return override ? { ...base, ...override } : base;
  }

  function resolveSelectOptions(field: AdapterParamField): AdapterParamOption[] {
    return optionOverrides?.[field.key] ?? field.options ?? [];
  }

  function resolveSelectValue(field: AdapterParamField, opts: AdapterParamOption[]): string {
    const raw = String(params[field.key] ?? field.defaultValue ?? '');
    if (opts.length === 0 || opts.some((o) => o.value === raw)) {
      return raw;
    }
    return opts[0].value;
  }

  function parseResidueTokens(raw: unknown): string[] {
    return String(raw ?? '')
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function removeResidueTokenAt(key: string, idx: number) {
    const tokens = parseResidueTokens(params[key]);
    tokens.splice(idx, 1);
    onChange(key, tokens.join(', '));
  }

  function parseAaTokens(raw: unknown): Array<{ char: string; valid: boolean }> {
    const compact = String(raw ?? '')
      .toUpperCase()
      .replace(/[\s,;]+/g, '');
    const seen = new Set<string>();
    const out: Array<{ char: string; valid: boolean }> = [];
    for (const ch of compact) {
      if (seen.has(ch)) continue;
      seen.add(ch);
      out.push({ char: ch, valid: VALID_AA_SET.has(ch) });
    }
    return out;
  }

  function removeAaChar(key: string, target: string) {
    const next = String(params[key] ?? '')
      .toUpperCase()
      .split('')
      .filter((ch) => ch !== target)
      .join('');
    onChange(key, next);
  }

  function labelWithUnit(field: AdapterParamField): string {
    if (field.unit && !field.label.includes(field.unit)) {
      return `${field.label} (${field.unit})`;
    }
    return field.label;
  }
</script>

{#each fields as field (field.key)}
  {@const fieldDisabled = isFieldDisabled(field.key)}
  {#if field.kind === 'number' || field.kind === 'integer'}
    {@const numSpec = resolveNumericSpec(field)}
    <label class="schema-field">
      <span class="label-row">
        <span>{labelWithUnit(field)}</span>
        {#if field.upstreamDefaultHint}
          <span class="default-badge" title={field.upstreamDefaultHint}>{field.upstreamDefaultHint}</span>
        {/if}
      </span>
      <NumericInput
        disabled={fieldDisabled}
        id={`param-${field.key}`}
        name={field.key}
        ariaLabel={field.label}
        spec={numSpec}
        value={Number(params[field.key] ?? field.defaultValue ?? numSpec.defaultValue)}
        onChange={(v) => onChange(field.key, v)}
      />
      {#if field.description}
        <span class="field-hint">{field.description}</span>
      {/if}
    </label>
  {:else if field.kind === 'boolean'}
    <div class="bool-wrap full">
      <label class="inline-check" class:is-muted={fieldDisabled}>
        <input
          type="checkbox"
          id={`param-${field.key}`}
          name={field.key}
          disabled={fieldDisabled}
          checked={Boolean(params[field.key] ?? field.defaultValue)}
          onchange={(e) => onChange(field.key, e.currentTarget.checked)}
        />
        <span>{field.label}</span>
        {#if field.upstreamDefaultHint}
          <span class="default-badge" title={field.upstreamDefaultHint}>{field.upstreamDefaultHint}</span>
        {/if}
      </label>
      {#if field.description}
        <span class="field-hint">{field.description}</span>
      {/if}
    </div>
  {:else if field.kind === 'select'}
    {@const opts = resolveSelectOptions(field)}
    <label class="schema-field wide">
      <span class="label-row">
        <span>{field.label}</span>
        {#if field.upstreamDefaultHint}
          <span class="default-badge" title={field.upstreamDefaultHint}>{field.upstreamDefaultHint}</span>
        {/if}
      </span>
      <select
        id={`param-${field.key}`}
        name={field.key}
        disabled={fieldDisabled}
        value={resolveSelectValue(field, opts)}
        onchange={(e) => onChange(field.key, e.currentTarget.value)}
      >
        {#each opts as opt (opt.value)}
          <option value={opt.value}>{opt.label}{opt.hint ? ` — ${opt.hint}` : ''}</option>
        {/each}
      </select>
      {#if field.description}
        <span class="field-hint">{field.description}</span>
      {/if}
    </label>
  {:else if field.kind === 'textarea'}
    <label class="schema-field full">
      <span class="label-row">
        <span>{field.label}</span>
        {#if field.upstreamDefaultHint}
          <span class="default-badge" title={field.upstreamDefaultHint}>{field.upstreamDefaultHint}</span>
        {/if}
      </span>
      <textarea
        id={`param-${field.key}`}
        name={field.key}
        rows="3"
        spellcheck="false"
        disabled={fieldDisabled}
        placeholder={field.placeholder ?? ''}
        value={String(params[field.key] ?? field.defaultValue ?? '')}
        oninput={(e) => onChange(field.key, e.currentTarget.value)}
      ></textarea>
      {#if field.description}
        <span class="field-hint">{field.description}</span>
      {/if}
    </label>
  {:else if field.kind === 'residue-list'}
    {@const tokens = parseResidueTokens(params[field.key])}
    <div class="chip-field full">
      <label class="schema-field">
        <span class="label-row">
          <span>{field.label}</span>
          {#if field.upstreamDefaultHint}
            <span class="default-badge" title={field.upstreamDefaultHint}>{field.upstreamDefaultHint}</span>
          {/if}
        </span>
        <input
          type="text"
          id={`param-${field.key}`}
          name={field.key}
          disabled={fieldDisabled}
          placeholder={field.placeholder ?? 'e.g. A10, A12-15'}
          value={String(params[field.key] ?? field.defaultValue ?? '')}
          oninput={(e) => onChange(field.key, e.currentTarget.value)}
        />
      </label>
      {#if tokens.length > 0}
        <div class="res-chips">
          {#each tokens as tok, i (`res_${field.key}_${tok}_${i}`)}
            {@const badTok = !/^[A-Za-z_]$/.test(tok) && !isValidResidueToken(tok)}
            <span
              class="res-chip"
              class:invalid-chip={badTok}
              title={badTok ? 'Expected format like A, A10, or A12-15' : undefined}
            >
              {tok}
              <button
                type="button"
                class="chip-x"
                disabled={fieldDisabled}
                aria-label={`Remove ${tok}`}
                onclick={() => removeResidueTokenAt(field.key, i)}
              >×</button>
            </span>
          {/each}
        </div>
      {/if}
      {#if field.description}
        <span class="field-hint">{field.description}</span>
      {/if}
    </div>
  {:else if field.kind === 'aa-set'}
    {@const aaTokens = parseAaTokens(params[field.key])}
    {@const hasInvalidAa = aaTokens.some((t) => !t.valid)}
    <div class="chip-field full">
      <label class="schema-field">
        <span class="label-row">
          <span>{field.label}</span>
          {#if field.upstreamDefaultHint}
            <span class="default-badge" title={field.upstreamDefaultHint}>{field.upstreamDefaultHint}</span>
          {/if}
        </span>
        <input
          type="text"
          id={`param-${field.key}`}
          name={field.key}
          class:input-error={hasInvalidAa}
          disabled={fieldDisabled}
          placeholder={field.placeholder ?? 'e.g. C'}
          value={String(params[field.key] ?? field.defaultValue ?? '')}
          oninput={(e) => onChange(field.key, e.currentTarget.value.toUpperCase())}
        />
      </label>
      {#if aaTokens.length > 0}
        <div class="res-chips">
          {#each aaTokens as tok (`aa_${field.key}_${tok.char}`)}
            <span
              class="res-chip"
              class:invalid-chip={!tok.valid}
              title={tok.valid ? `Omit ${tok.char}` : `"${tok.char}" is not a standard amino acid`}
            >
              {tok.char}
              <button
                type="button"
                class="chip-x"
                disabled={fieldDisabled}
                aria-label={`Remove ${tok.char}`}
                onclick={() => removeAaChar(field.key, tok.char)}
              >×</button>
            </span>
          {/each}
        </div>
      {/if}
      {#if hasInvalidAa}
        <span class="field-hint err">Use standard 1-letter amino acid codes (A–Y).</span>
      {:else if field.description}
        <span class="field-hint">{field.description}</span>
      {/if}
    </div>
  {:else}
    {@const rawVal = String(params[field.key] ?? field.defaultValue ?? '')}
    {@const fieldErr = field.key === 'tiedPositions' ? validateTiedPositions(rawVal) : null}
    <label class="schema-field wide">
      <span class="label-row">
        <span>{field.label}</span>
        {#if field.upstreamDefaultHint}
          <span class="default-badge" title={field.upstreamDefaultHint}>{field.upstreamDefaultHint}</span>
        {/if}
      </span>
      <input
        type="text"
        id={`param-${field.key}`}
        name={field.key}
        class:input-error={Boolean(fieldErr)}
        aria-invalid={fieldErr ? 'true' : undefined}
        disabled={fieldDisabled}
        placeholder={field.placeholder ?? ''}
        value={rawVal}
        oninput={(e) => onChange(field.key, e.currentTarget.value)}
      />
      {#if fieldErr}
        <span class="field-hint err" role="alert">{fieldErr}</span>
      {:else if field.description}
        <span class="field-hint">{field.description}</span>
      {/if}
    </label>
  {/if}
{/each}

<style>
  .schema-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }
  .schema-field.wide {
    min-width: 180px;
    flex: 1;
  }
  .full {
    width: 100%;
  }
  .label-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .default-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 500;
    background: var(--bg-pill);
    color: var(--text-dim);
    border: 1px solid var(--border);
    line-height: 1.35;
  }
  .bool-wrap {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  label.inline-check {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-muted);
    cursor: pointer;
    flex-wrap: wrap;
  }
  label.inline-check.is-muted {
    opacity: 0.72;
    cursor: not-allowed;
  }
  .chip-field {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .res-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .res-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--accent-soft);
    background: var(--bg-pill);
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    padding: 2px 6px 2px 8px;
  }
  .res-chip.invalid-chip {
    border-color: var(--err-strong);
    color: var(--err-strong);
    background: color-mix(in srgb, var(--err-strong) 10%, transparent);
  }
  .chip-x {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 12px;
    line-height: 1;
    padding: 0 2px;
    border-radius: 50%;
  }
  .chip-x:hover:not(:disabled) {
    color: var(--err-strong);
  }
  .chip-x:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  input[type='text'],
  select,
  textarea {
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 6px 8px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
  }
  textarea {
    width: 100%;
    box-sizing: border-box;
    resize: vertical;
  }
  .input-error {
    border-color: var(--err-strong) !important;
  }
  .field-hint {
    font-size: 10.5px;
    line-height: 1.35;
    color: var(--text-dim);
  }
  .field-hint.err {
    color: var(--err-strong);
  }
</style>
