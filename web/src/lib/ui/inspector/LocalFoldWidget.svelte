<script lang="ts">
  import { ESMFOLD2_STEPS, LOCALFOLD_MODEL_OPTIONS } from '$lib/engine/registry';
  import { sanitizeParamsOnModeSwitch } from '$lib/engine/graph-utils';
  import { NUMERIC_SPECS, type NumericFieldSpec } from '$lib/engine/param-validation';
  import NumericInput from '../NumericInput.svelte';
  import type { MsaMode, SequenceSet, WorkflowNode } from '$lib/engine/types';

  interface Props {
    node: WorkflowNode;
    disabled?: boolean;
    selectedLocalFoldMeta: (typeof LOCALFOLD_MODEL_OPTIONS)[number] | null;
    msaA3mDraft: string;
    lfMultiDesignInfo: { backboneCount: number; sequenceCount: number | null } | null;
    upstreamSequences: SequenceSet | null;
    maxTotalFoldsSpec: NumericFieldSpec;
    onUpdate: (nodeId: string, params: Record<string, unknown>) => void;
    onUpdateLabel?: (nodeId: string, label: string) => void;
    onSetParam: (key: string, value: unknown) => void;
    onScheduleTextParam: (key: 'msaA3mText', value: string) => void;
    onFlushPendingTextCommit: () => void;
    onA3mFileUpload: (ev: Event) => void;
  }

  let {
    node,
    disabled = false,
    selectedLocalFoldMeta,
    msaA3mDraft,
    lfMultiDesignInfo,
    upstreamSequences,
    maxTotalFoldsSpec,
    onUpdate,
    onUpdateLabel = undefined,
    onSetParam,
    onScheduleTextParam,
    onFlushPendingTextCommit,
    onA3mFileUpload,
  }: Props = $props();

  function localFoldMaxRecycles(modelId: string, family: string): number {
    return modelId === 'mol1' || family === 'af2-multimer' ? 20 : 10;
  }

  function localFoldDefaultRecyclesHint(modelId: string, family: string): string {
    if (family === 'af2-multimer') return 'Upstream default: 20 (max 20)';
    if (modelId === 'mol1') return 'Upstream default: 10 (max 20)';
    if (family === 'af3') return 'Upstream default: 10 (max 10)';
    return 'Upstream default: 3 (max 10)';
  }

  function formatLocalFoldMsaBadge(modeRaw: unknown): string {
    const mode = String(modeRaw ?? 'single') as MsaMode;
    if (mode === 'mmseqs2') return 'MMseqs2 MSA';
    if (mode === 'target-only') return 'Target-only MSA';
    if (mode === 'a3m') return 'Custom .a3m MSA';
    return 'Single-sequence (no MSA)';
  }

  function parseA3mSequenceSummary(raw: string): { seqCount: number; queryLength: number } {
    const trimmed = raw.trim();
    if (!trimmed) return { seqCount: 0, queryLength: 0 };
    const lines = trimmed.split(/\r?\n/);
    let seqCount = 0;
    let firstSeq = '';
    let curSeq = '';
    let seenHeader = false;
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#')) continue;
      if (line.startsWith('>')) {
        if (seenHeader && curSeq.length > 0) {
          seqCount++;
          if (!firstSeq) firstSeq = curSeq;
        }
        seenHeader = true;
        curSeq = '';
      } else {
        const clean = line.replace(/\s+/g, '');
        if (seenHeader) {
          curSeq += clean;
        } else {
          seqCount++;
          if (!firstSeq) firstSeq = clean;
        }
      }
    }
    if (seenHeader && curSeq.length > 0) {
      seqCount++;
      if (!firstSeq) firstSeq = curSeq;
    }
    const queryLength = firstSeq.replace(/[a-z.-]/g, '').length;
    return { seqCount, queryLength };
  }

  function isAutoLocalFoldLabel(label: string | undefined): { auto: boolean; suffix: string } {
    const raw = (label ?? '').trim();
    if (!raw || raw === 'LocalFold') return { auto: true, suffix: '' };
    const parenMatch = raw.match(/^(.*?)\s*(\((?:MPNN|Native|AF2|RF3)\))$/);
    const base = parenMatch ? parenMatch[1].trim() : raw;
    const suffix = parenMatch ? ` ${parenMatch[2]}` : '';
    if (
      base === 'LocalFold' ||
      /^LocalFold\s*\([^)]+\)$/i.test(raw) ||
      LOCALFOLD_MODEL_OPTIONS.some((m) => m.shortLabel === base || m.label === base)
    ) {
      return { auto: true, suffix: /^LocalFold\s*\(/i.test(raw) && !parenMatch ? '' : suffix };
    }
    return { auto: false, suffix: '' };
  }

  let lfModelId = $derived(String(node.params.model ?? 'monomer'));
  let lfMeta = $derived(selectedLocalFoldMeta ?? LOCALFOLD_MODEL_OPTIONS[0]);
  let lfMsaMode = $derived(String(node.params.msaMode ?? 'single') as MsaMode);
  let lfMaxRecycles = $derived(localFoldMaxRecycles(lfModelId, lfMeta.family));
  let lfRecyclesSpec = $derived({
    ...NUMERIC_SPECS['localfold-af2'].recycles,
    max: lfMaxRecycles,
  });
  let lfDefaultTol = $derived(lfMeta.family === 'af2-multimer' ? 0.5 : 0);
  let lfTolSpec = $derived({
    min: 0,
    max: 5,
    step: 0.1,
    integer: false,
    defaultValue: lfDefaultTol,
    unit: 'Å',
  });
  let lfA3mSummary = $derived(lfMsaMode === 'a3m' ? parseA3mSequenceSummary(msaA3mDraft) : null);
  let lfHasHoloContext = $derived(
    Boolean(String(node.params.ligandCcdCodes ?? '').trim()) ||
      Boolean(String(node.params.dnaSequences ?? '').trim()) ||
      Boolean(String(node.params.rnaSequences ?? '').trim()),
  );
</script>

<label class="wide">
  Model
  <select
    {disabled}
    value={lfModelId}
    onchange={(e) => {
      const nextModel = e.currentTarget.value;
      const nextMeta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === nextModel);
      const autoInfo = isAutoLocalFoldLabel(node.label);
      if (autoInfo.auto && nextMeta) {
        onUpdateLabel?.(node.id, `${nextMeta.shortLabel}${autoInfo.suffix}`);
      }
      const sanitized = sanitizeParamsOnModeSwitch('localfold-af2', node.params, nextModel);
      onUpdate(node.id, sanitized);
    }}
  >
    <optgroup label="AlphaFold2-Class (Monomer)">
      {#each LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af2') as opt (opt.id)}
        <option value={opt.id}>{opt.label}</option>
      {/each}
    </optgroup>
    <optgroup label="AlphaFold2-Multimer v3 (Complexes)">
      {#each LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af2-multimer') as opt (opt.id)}
        <option value={opt.id}>{opt.label}</option>
      {/each}
    </optgroup>
    <optgroup label="AF3-Class (All-Atom)">
      {#each LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3') as opt (opt.id)}
        <option value={opt.id}>{opt.label}</option>
      {/each}
    </optgroup>
    <optgroup label="ESMFold2 Experimental Fast">
      {#each LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'esmfold2') as opt (opt.id)}
        <option value={opt.id}>{opt.label}</option>
      {/each}
    </optgroup>
  </select>
  <span class="field-hint">
    {formatLocalFoldMsaBadge(lfMeta.family === 'esmfold2' ? 'single' : lfMsaMode)} · Weights: ~{lfMeta.approxMB} MB · License: {lfMeta.license}
  </span>
</label>

{#if lfMeta.family === 'esmfold2'}
  <label>
    Diffusion schedule
    <select
      {disabled}
      value={Number(node.params.esmfold2Steps ?? ESMFOLD2_STEPS[0])}
      onchange={(e) => onSetParam('esmfold2Steps', Number(e.currentTarget.value))}
    >
      {#each ESMFOLD2_STEPS as steps (steps)}
        <option value={steps}>{steps} scheduled steps</option>
      {/each}
    </select>
    <span class="field-hint">
      Single-sequence ESM-C + all-atom diffusion. The default 15-step schedule executes 11 diffusion updates.
    </span>
  </label>
  {#if lfMsaMode !== 'single'}
    <div class="full">
      <span class="field-hint warn">This checkpoint runs in single-sequence mode without an external MSA.</span>
      <button
        type="button"
        {disabled}
        onclick={() =>
          onUpdate(node.id, {
            ...node.params,
            msaMode: 'single',
          })}
      >
        Switch to single-sequence
      </button>
    </div>
  {/if}
{:else}
  <label class="full">
    MSA Mode
    <select
      {disabled}
      value={lfMsaMode}
      onchange={(e) => {
        onFlushPendingTextCommit();
        onSetParam('msaMode', e.currentTarget.value);
      }}
    >
      <option value="single">Single-sequence (100% local)</option>
      <option value="mmseqs2">MMseqs2 MSA (ColabFold API)</option>
      <option value="target-only">Target-only MSA (binders — skips de novo chain)</option>
      <option value="a3m">Custom .a3m alignment (local)</option>
    </select>
    {#if lfMsaMode === 'mmseqs2' || lfMsaMode === 'target-only'}
      <span class="field-hint">
        Queries api.colabfold.com for evolutionary alignments; cached across candidates
      </span>
    {/if}
  </label>

  {#if lfMsaMode === 'a3m'}
    <div class="a3m-box full">
      <div class="a3m-head">
        <label class="mu-upload-btn a3m-upload-btn" class:is-disabled={disabled}>
          <span>↑ Upload .a3m file</span>
          <input
            type="file"
            accept=".a3m,.fas,.fasta,.txt"
            hidden
            {disabled}
            onchange={onA3mFileUpload}
          />
        </label>
        {#if lfA3mSummary && lfA3mSummary.seqCount > 0}
          <span class="field-hint ok">
            ✓ {lfA3mSummary.seqCount} sequence{lfA3mSummary.seqCount === 1 ? '' : 's'} parsed{lfA3mSummary.queryLength > 0 ? ` · ${lfA3mSummary.queryLength} aa query` : ''}
          </span>
        {:else}
          <span class="field-hint warn">Paste or upload a .a3m alignment</span>
        {/if}
      </div>
      <textarea
        rows="4"
        spellcheck="false"
        class:input-error={!lfA3mSummary || lfA3mSummary.seqCount === 0}
        {disabled}
        placeholder=">query&#10;MKTAYIAKQRQISFVK...&#10;>hit1&#10;MKTAYIAKQRQISFVK..."
        value={msaA3mDraft}
        oninput={(e) => onScheduleTextParam('msaA3mText', e.currentTarget.value)}
        onblur={onFlushPendingTextCommit}
      ></textarea>
    </div>
  {/if}

  <label>
    Recycles
    <NumericInput
      {disabled}
      ariaLabel="Recycles"
      spec={lfRecyclesSpec}
      value={Number(node.params.recycles ?? 1)}
      onChange={(v) => onSetParam('recycles', v)}
    />
    <span class="field-hint">{localFoldDefaultRecyclesHint(lfModelId, lfMeta.family)}</span>
  </label>
{/if}

{#if lfMeta.family === 'af3'}
  <label>
    Diffusion steps
    <NumericInput
      {disabled}
      ariaLabel="Diffusion steps"
      spec={NUMERIC_SPECS['localfold-af2'].diffusionSteps}
      value={Number(node.params.diffusionSteps ?? 25)}
      onChange={(v) => onSetParam('diffusionSteps', v)}
    />
  </label>
{/if}

<label>
  Max seqs to fold
  <NumericInput
    {disabled}
    ariaLabel="Max sequences to fold"
    spec={NUMERIC_SPECS['localfold-af2'].maxSequences}
    value={Number(node.params.maxSequences ?? 2)}
    onChange={(v) => onSetParam('maxSequences', v)}
  />
</label>

{#if lfMultiDesignInfo}
  {@const lfPerBackboneCap = Math.max(1, Math.min(8, Math.round(Number(node.params.maxSequences ?? 2)) || 2))}
  {@const lfCampaignCap = Math.max(1, Math.min(64, Math.round(Number(node.params.maxTotalFolds ?? 16)) || 16))}
  {@const lfPlanned = Math.min(
    lfCampaignCap,
    lfMultiDesignInfo.backboneCount * lfPerBackboneCap,
    lfMultiDesignInfo.sequenceCount ?? lfCampaignCap,
  )}
  <div class="fanout-chip full" role="status">
    <span class="fanout-dot" aria-hidden="true"></span>
    <span>
      <b>{lfMultiDesignInfo.backboneCount} backbones</b>
      {#if lfMultiDesignInfo.sequenceCount !== null}
        → {lfMultiDesignInfo.sequenceCount} sequences
      {/if}
      → up to <b>{lfPlanned} refold{lfPlanned === 1 ? '' : 's'}</b>
      (≤{lfPerBackboneCap}/backbone, cap {lfCampaignCap})
    </span>
  </div>
{:else if upstreamSequences && upstreamSequences.candidates.length > Math.max(1, Math.min(8, Math.round(Number(node.params.maxSequences ?? 2)) || 2))}
  {@const lfCapNow = Math.max(1, Math.min(8, Math.round(Number(node.params.maxSequences ?? 2)) || 2))}
  {@const lfUpCount = upstreamSequences.candidates.length}
  {@const lfTargetBump = Math.min(8, lfUpCount)}
  <div class="full lf-warn-banner" role="status">
    <span>Folding <b>{lfCapNow} of {lfUpCount}</b> upstream sequences</span>
    <button
      type="button"
      class="mini-btn"
      {disabled}
      onclick={() => onSetParam('maxSequences', lfTargetBump)}
    >
      Fold all {lfTargetBump}
    </button>
  </div>
{/if}

{#if lfMeta.nonCommercial}
  <div class="full lf-warn-note">
    AlphaFold 3 weights are subject to DeepMind's Non-Commercial Academic Terms. For permissive open weights, use <b>RoseTTAFold3</b> (BSD-3), <b>Boltz-2</b> (MIT), or <b>OpenBind-0</b> (Apache-2.0).
  </div>
{/if}

{#if lfMeta.family === 'af3' || lfMeta.family === 'esmfold2'}
  <details class="adv-details full" open={lfHasHoloContext}>
    <summary>
      Holo co-folding (ligands / DNA / RNA)
      {#if lfHasHoloContext}
        <span class="adv-badge">Active</span>
      {/if}
    </summary>
    <div class="adv-fields">
      <label class="full">
        Ligand CCD codes
        <input
          type="text"
          {disabled}
          placeholder="e.g. SAM, ZN, ATP"
          value={String(node.params.ligandCcdCodes ?? '')}
          oninput={(e) => onSetParam('ligandCcdCodes', e.currentTarget.value)}
        />
        <span class="field-hint">Comma-separated PDB Chemical Component Dictionary (CCD) IDs.</span>
      </label>
      <label class="full">
        DNA sequences
        <input
          type="text"
          {disabled}
          placeholder="e.g. ACGTACGT (separate multiple chains with : or ,)"
          value={String(node.params.dnaSequences ?? '')}
          oninput={(e) => onSetParam('dnaSequences', e.currentTarget.value)}
        />
      </label>
      <label class="full">
        RNA sequences
        <input
          type="text"
          {disabled}
          placeholder="e.g. AUGCAUGC (separate multiple chains with : or ,)"
          value={String(node.params.rnaSequences ?? '')}
          oninput={(e) => onSetParam('rnaSequences', e.currentTarget.value)}
        />
      </label>
    </div>
  </details>
{/if}

<details class="adv-details full">
  <summary>
    Advanced parameters
    {#if Number(node.params.maxTotalFolds ?? 16) !== 16}
      <span class="adv-badge">Cap {Number(node.params.maxTotalFolds ?? 16)}</span>
    {/if}
  </summary>
  <div class="adv-fields">
    <label>
      Max total folds
      <NumericInput
        {disabled}
        ariaLabel="Max total folds across designs"
        spec={maxTotalFoldsSpec}
        value={Number(node.params.maxTotalFolds ?? 16)}
        onChange={(v) => onSetParam('maxTotalFolds', v)}
      />
      <span class="field-hint">
        Campaign cap across multi-backbone batches (round-robin up to {Number(node.params.maxSequences ?? 2)}/backbone).
      </span>
    </label>
    <label>
      Seed
      <NumericInput
        {disabled}
        ariaLabel="Seed"
        spec={NUMERIC_SPECS['localfold-af2'].seed}
        value={Number(node.params.seed ?? 0)}
        onChange={(v) => onSetParam('seed', v)}
      />
    </label>
    {#if lfMeta.family === 'af2' || lfMeta.family === 'af2-multimer'}
      <label>
        Early-stop tolerance (Å)
        <NumericInput
          {disabled}
          ariaLabel="Early-stop tolerance (Å)"
          spec={lfTolSpec}
          value={Number(node.params.tolerance ?? lfDefaultTol)}
          onChange={(v) => onSetParam('tolerance', v)}
        />
        <span class="field-hint">
          0 = run all recycles on AF2 monomer (0.5 Å default on AF2-Multimer).
        </span>
      </label>
    {/if}
  </div>
</details>

<style>
  label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }
  label.wide {
    min-width: 180px;
    flex: 1;
  }
  .full {
    width: 100%;
  }
  select,
  input[type='text'],
  textarea {
    background: var(--bg-input);
    color: var(--text-body);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 5px 8px;
    font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  textarea {
    resize: vertical;
    width: 100%;
    box-sizing: border-box;
  }
  .input-error {
    border-color: var(--err-strong) !important;
  }
  .field-hint {
    font-size: 10.5px;
    color: var(--text-muted);
    line-height: 1.35;
  }
  .field-hint.ok {
    color: var(--ok-text);
  }
  .field-hint.warn {
    color: var(--warn-text);
  }
  .a3m-box {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .a3m-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }
  .mu-upload-btn {
    display: inline-flex;
    flex-direction: row;
    align-items: center;
    gap: 5px;
    padding: 4px 9px;
    border-radius: 5px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-strong);
    color: var(--accent-soft);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .mu-upload-btn.is-disabled {
    opacity: 0.5;
    pointer-events: none;
  }
  .fanout-chip {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 9px;
  }
  .fanout-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
  }
  .lf-warn-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 10.5px;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    padding: 6px 8px;
    border-radius: 5px;
  }
  .lf-warn-note {
    font-size: 10.5px;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    padding: 6px 8px;
    border-radius: 5px;
  }
  .mini-btn {
    background: var(--bg-surface);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 10.5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    cursor: pointer;
  }
  .adv-details {
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 6px 9px;
    background: var(--bg-surface-2);
  }
  .adv-details summary {
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .adv-badge {
    font-size: 9.5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    padding: 1px 6px;
    border-radius: 999px;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
  }
  .adv-fields {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 12px;
    margin-top: 8px;
  }
</style>
