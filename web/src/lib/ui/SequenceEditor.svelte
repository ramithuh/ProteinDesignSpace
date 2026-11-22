<script lang="ts">
  import {
    AA_BY_CODE,
    STANDARD_AAS,
    STANDARD_AA_SET,
    cursorToResiduePosition,
    diffEntryResidues,
    mutateResidueInRaw,
    parseMutationCommand,
    parseSequenceEntries,
    type CursorResidueInfo,
    type ParsedResidueCell,
  } from '$lib/engine/sequence-editor';
  import type { SequenceValidationResult } from '$lib/engine/param-validation';

  interface Props {
    nodeId: string;
    value: string;
    disabled?: boolean;
    validation: SequenceValidationResult;
    onChange: (nextValue: string) => void;
    onBlur?: () => void;
  }

  let {
    nodeId,
    value,
    disabled = false,
    validation,
    onChange,
    onBlur = undefined,
  }: Props = $props();

  const MAX_INTERACTIVE_RESIDUES = 2000;
  const BLOCK_SIZE = 10;

  let mode = $state<'residues' | 'raw'>('residues');
  let activeEntryIdx = $state(0);
  let selectedPos = $state<number | null>(null);
  let hoveredPos = $state<number | null>(null);
  let cmdInput = $state('');
  let cmdError = $state<string | null>(null);
  let rawCursor = $state<CursorResidueInfo | null>(null);
  let copied = $state(false);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;

  // Baseline sequence per nodeId so we can highlight point mutations (e.g. Y45L) and offer 1-click Revert
  let baselineNodeId = $state('');
  let baselineValue = $state('');

  $effect(() => {
    if (nodeId !== baselineNodeId) {
      baselineNodeId = nodeId;
      baselineValue = value;
      activeEntryIdx = 0;
      selectedPos = null;
      hoveredPos = null;
      cmdInput = '';
      cmdError = null;
      rawCursor = null;
    } else if (!baselineValue && value) {
      baselineValue = value;
    }
  });

  let entries = $derived(parseSequenceEntries(value));
  let totalResidues = $derived(entries.reduce((sum, e) => sum + e.residues.length, 0));
  let canUseResidueMode = $derived(
    entries.length > 0 &&
      totalResidues > 0 &&
      totalResidues <= MAX_INTERACTIVE_RESIDUES &&
      validation.status !== 'error',
  );
  let effectiveMode = $derived<'residues' | 'raw'>(
    mode === 'residues' && canUseResidueMode ? 'residues' : 'raw',
  );

  let clampedEntryIdx = $derived(
    entries.length > 0 ? Math.max(0, Math.min(entries.length - 1, activeEntryIdx)) : 0,
  );
  let curEntry = $derived(entries[clampedEntryIdx] ?? null);

  let diffs = $derived(diffEntryResidues(baselineValue, value, clampedEntryIdx));
  let mutatedPosSet = $derived(new Set(diffs.map((d) => d.pos)));

  // Chunk residues into 10-mer blocks for clean numbered reading
  interface ResidueBlock {
    startPos: number;
    endPos: number;
    cells: ParsedResidueCell[];
  }

  let blocks = $derived.by((): ResidueBlock[] => {
    if (!curEntry) return [];
    const out: ResidueBlock[] = [];
    for (let i = 0; i < curEntry.residues.length; i += BLOCK_SIZE) {
      const slice = curEntry.residues.slice(i, i + BLOCK_SIZE);
      if (slice.length > 0) {
        out.push({
          startPos: slice[0].pos,
          endPos: slice[slice.length - 1].pos,
          cells: slice,
        });
      }
    }
    return out;
  });

  let selectedCell = $derived(
    curEntry && selectedPos !== null ? (curEntry.residues[selectedPos - 1] ?? null) : null,
  );
  let readoutCell = $derived.by((): ParsedResidueCell | null => {
    if (!curEntry) return null;
    if (hoveredPos !== null) return curEntry.residues[hoveredPos - 1] ?? null;
    if (selectedPos !== null) return curEntry.residues[selectedPos - 1] ?? null;
    return null;
  });

  function applyMutation(pos1: number, nextAa: string) {
    if (disabled) return;
    const updated = mutateResidueInRaw(value, clampedEntryIdx, pos1, nextAa);
    if (updated === null || updated === value) return;
    onChange(updated);
    cmdError = null;
  }

  function submitCommand(ev: Event) {
    ev.preventDefault();
    if (!curEntry) return;
    const parsed = parseMutationCommand(cmdInput);
    if (parsed.kind === 'error') {
      cmdError = parsed.message;
      return;
    }
    if (parsed.pos > curEntry.residues.length) {
      cmdError = `Position ${parsed.pos} exceeds length (${curEntry.residues.length} aa).`;
      return;
    }
    const existing = curEntry.residues[parsed.pos - 1];
    if (parsed.kind === 'jump') {
      selectedPos = parsed.pos;
      cmdError = null;
      cmdInput = '';
      return;
    }
    if (parsed.expectedWt && existing && existing.aa !== parsed.expectedWt) {
      cmdError = `Pos ${parsed.pos} is ${existing.aa}, not ${parsed.expectedWt}.`;
      selectedPos = parsed.pos;
      return;
    }
    selectedPos = parsed.pos;
    applyMutation(parsed.pos, parsed.targetAa);
    cmdInput = '';
    cmdError = null;
  }

  function handleCellKeydown(ev: KeyboardEvent, pos1: number) {
    if (!curEntry) return;
    if (ev.key === 'ArrowRight') {
      ev.preventDefault();
      ev.stopPropagation();
      selectedPos = Math.min(curEntry.residues.length, pos1 + 1);
      return;
    }
    if (ev.key === 'ArrowLeft') {
      ev.preventDefault();
      ev.stopPropagation();
      selectedPos = Math.max(1, pos1 - 1);
      return;
    }
    if (ev.key === 'Escape') {
      ev.preventDefault();
      ev.stopPropagation();
      selectedPos = null;
      return;
    }
    if (
      !disabled &&
      !ev.metaKey &&
      !ev.ctrlKey &&
      !ev.altKey &&
      ev.key.length === 1 &&
      STANDARD_AA_SET.has(ev.key.toUpperCase())
    ) {
      ev.preventDefault();
      ev.stopPropagation();
      applyMutation(pos1, ev.key.toUpperCase());
    }
  }

  function updateRawCursor(el: HTMLTextAreaElement) {
    const pos = el.selectionStart ?? 0;
    rawCursor = cursorToResiduePosition(el.value, pos);
  }

  function copySequence() {
    if (typeof navigator === 'undefined' || !navigator.clipboard) return;
    void navigator.clipboard.writeText(value).then(() => {
      copied = true;
      if (copyTimer) clearTimeout(copyTimer);
      copyTimer = setTimeout(() => {
        copied = false;
      }, 1500);
    });
  }

  function revertChanges() {
    if (disabled || !baselineValue) return;
    onChange(baselineValue);
    cmdError = null;
  }
</script>

<div class="seq-editor" class:is-disabled={disabled}>
  <div class="seq-hdr">
    <span class="seq-lbl">Amino acid sequence</span>
    <div class="seq-hdr-actions">
      {#if entries.length > 1 && effectiveMode === 'residues'}
        <select
          class="seq-entry-sel"
          aria-label="Select FASTA entry"
          value={clampedEntryIdx}
          onchange={(e) => {
            activeEntryIdx = Number(e.currentTarget.value) || 0;
            selectedPos = null;
            hoveredPos = null;
          }}
        >
          {#each entries as ent, i (i)}
            <option value={i}>
              {ent.header || `Sequence ${i + 1}`} ({ent.residues.length} aa)
            </option>
          {/each}
        </select>
      {/if}
      <div class="seq-mode-seg" role="tablist" aria-label="Sequence editor view">
        <button
          type="button"
          role="tab"
          class="seg-btn"
          class:active={effectiveMode === 'residues'}
          aria-selected={effectiveMode === 'residues'}
          disabled={!canUseResidueMode}
          title={canUseResidueMode
            ? 'Numbered interactive residue editor'
            : 'Available for valid sequences up to 2,000 aa'}
          onclick={() => (mode = 'residues')}
        >
          Residues
        </button>
        <button
          type="button"
          role="tab"
          class="seg-btn"
          class:active={effectiveMode === 'raw'}
          aria-selected={effectiveMode === 'raw'}
          onclick={() => (mode = 'raw')}
        >
          Raw FASTA
        </button>
      </div>
      <button
        type="button"
        class="seq-copy-btn"
        onclick={copySequence}
        title="Copy sequence text"
      >
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  </div>

  {#if effectiveMode === 'residues' && curEntry}
    <div class="seq-toolbar">
      <form class="seq-cmd-form" onsubmit={submitCommand}>
        <input
          type="text"
          class="seq-cmd-input"
          class:has-err={Boolean(cmdError)}
          {disabled}
          spellcheck="false"
          autocomplete="off"
          placeholder="Mutate or jump (e.g. Y45L)"
          aria-label="Jump to residue or apply point mutation"
          bind:value={cmdInput}
          oninput={() => (cmdError = null)}
        />
        {#if cmdInput.trim().length > 0}
          <button type="submit" class="seq-cmd-go" {disabled}>Apply</button>
        {/if}
      </form>
      <div class="seq-readout" aria-live="polite">
        {#if readoutCell}
          {@const meta = AA_BY_CODE[readoutCell.aa]}
          <span class="ro-pos">Pos {readoutCell.pos}</span>
          <span class="ro-aa">{readoutCell.aa}</span>
          {#if meta}
            <span class="ro-three">{meta.three}</span>
          {/if}
        {:else}
          <span class="ro-hint">Click any residue to mutate</span>
        {/if}
      </div>
    </div>

    {#if cmdError}
      <div class="seq-cmd-err" role="alert">{cmdError}</div>
    {/if}

    <div
      class="seq-grid-box"
      class:input-warn={validation.status === 'warn'}
      role="group"
      aria-label="Interactive numbered amino acid sequence"
    >
      {#if curEntry.header}
        <div class="seq-entry-hdr">&gt;{curEntry.header}</div>
      {/if}
      <div class="seq-blocks">
        {#each blocks as blk (blk.startPos)}
          <div class="seq-blk">
            <div class="seq-ruler" aria-hidden="true">
              <span class="ruler-start">{blk.startPos}</span>
              {#if blk.endPos > blk.startPos}
                <span class="ruler-end">{blk.endPos}</span>
              {/if}
            </div>
            <div class="seq-cells">
              {#each blk.cells as cell (cell.pos)}
                {@const meta = AA_BY_CODE[cell.aa]}
                {@const isSelected = selectedPos === cell.pos}
                {@const isMutated = mutatedPosSet.has(cell.pos)}
                {#if cell.startsNewChain}
                  <span class="chain-sep" title="Chain break">/</span>
                {/if}
                <button
                  type="button"
                  class="aa-cell grp-{meta?.group ?? 'special'}"
                  class:selected={isSelected}
                  class:mutated={isMutated}
                  {disabled}
                  title={`Pos ${cell.pos}: ${cell.aa}${meta ? ` (${meta.name})` : ''}${isMutated ? ' · edited' : ''}`}
                  aria-label={`Position ${cell.pos}, ${meta ? meta.name : cell.aa}`}
                  aria-pressed={isSelected}
                  onmouseenter={() => (hoveredPos = cell.pos)}
                  onmouseleave={() => {
                    if (hoveredPos === cell.pos) hoveredPos = null;
                  }}
                  onclick={() => {
                    selectedPos = selectedPos === cell.pos ? null : cell.pos;
                    cmdError = null;
                  }}
                  onkeydown={(ev) => handleCellKeydown(ev, cell.pos)}
                >
                  {cell.aa}
                </button>
              {/each}
            </div>
          </div>
        {/each}
      </div>
    </div>
  {:else}
    <textarea
      class="seq-raw-textarea"
      rows="5"
      spellcheck="false"
      autocomplete="off"
      autocorrect="off"
      autocapitalize="off"
      class:input-error={validation.status === 'error'}
      class:input-warn={validation.status === 'warn'}
      {disabled}
      placeholder=">seq1&#10;MKTAYIAKQRQISFVK..."
      {value}
      oninput={(e) => {
        onChange(e.currentTarget.value);
        updateRawCursor(e.currentTarget);
      }}
      onclick={(e) => updateRawCursor(e.currentTarget)}
      onkeyup={(e) => updateRawCursor(e.currentTarget)}
      onselect={(e) => updateRawCursor(e.currentTarget)}
      onblur={() => onBlur?.()}
    ></textarea>
  {/if}

  <div class="seq-foot">
    <span
      class="field-hint"
      class:err={validation.status === 'error'}
      class:warn={validation.status === 'warn'}
      class:ok={validation.status === 'ok'}
    >
      {validation.message}
    </span>

    <div class="seq-foot-right">
      {#if effectiveMode === 'raw' && rawCursor}
        {@const cMeta = AA_BY_CODE[rawCursor.aa]}
        <span class="raw-cursor-chip">
          Pos {rawCursor.pos}/{rawCursor.total} · {rawCursor.aa}{cMeta ? ` (${cMeta.three})` : ''}
        </span>
      {/if}
      {#if diffs.length > 0}
        <span class="mut-diff-list" title="Edits since loading this block">
          {#each diffs.slice(0, 4) as d (d.label)}
            <button
              type="button"
              class="mut-diff-pill"
              onclick={() => {
                if (canUseResidueMode) {
                  mode = 'residues';
                  selectedPos = d.pos;
                }
              }}
            >
              {d.label}
            </button>
          {/each}
          {#if diffs.length > 4}
            <span class="mut-diff-more">+{diffs.length - 4}</span>
          {/if}
          <button type="button" class="mut-revert-btn" {disabled} onclick={revertChanges}>
            Revert
          </button>
        </span>
      {/if}
    </div>
  </div>

  {#if effectiveMode === 'residues' && selectedCell}
    {@const selMeta = AA_BY_CODE[selectedCell.aa]}
    <div class="mut-picker" role="region" aria-label={`Mutate position ${selectedCell.pos}`}>
      <div class="mut-picker-top">
        <span class="mut-picker-title">
          Pos <b>{selectedCell.pos}</b> · {selectedCell.aa}{selMeta ? ` (${selMeta.name})` : ''}
        </span>
        <span class="mut-picker-sub">Pick replacement or type key</span>
        <button
          type="button"
          class="mut-close"
          onclick={() => (selectedPos = null)}
          aria-label="Close mutation picker"
        >
          ×
        </button>
      </div>
      <div class="aa-palette">
        {#each STANDARD_AAS as aa (aa.code)}
          <button
            type="button"
            class="pal-btn grp-{aa.group}"
            class:current={aa.code === selectedCell.aa}
            {disabled}
            title={`${aa.code} · ${aa.name} (${aa.group})`}
            onclick={() => applyMutation(selectedCell.pos, aa.code)}
          >
            <span class="pal-code">{aa.code}</span>
            <span class="pal-three">{aa.three}</span>
          </button>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .seq-editor {
    display: flex;
    flex-direction: column;
    gap: 6px;
    width: 100%;
  }

  .seq-hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }

  .seq-lbl {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
  }

  .seq-hdr-actions {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .seq-entry-sel {
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: var(--bg-surface-2);
    color: var(--text-primary);
    max-width: 140px;
  }

  .seq-mode-seg {
    display: inline-flex;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1px;
  }

  .seg-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 10.5px;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 4px;
    cursor: pointer;
  }

  .seg-btn.active {
    background: var(--bg-surface);
    color: var(--text-primary);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
  }

  .seg-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .seq-copy-btn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text-muted);
    font-size: 10.5px;
    padding: 2px 7px;
    cursor: pointer;
  }

  .seq-copy-btn:hover {
    color: var(--text-primary);
    border-color: var(--border-strong);
  }

  .seq-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }

  .seq-cmd-form {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    flex: 1 1 150px;
    min-width: 0;
  }

  .seq-cmd-input {
    width: 100%;
    max-width: 205px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 8px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-primary);
  }

  .seq-cmd-input:focus {
    outline: none;
    border-color: var(--accent);
  }

  .seq-cmd-input.has-err {
    border-color: var(--err-strong, #ef4444);
  }

  .seq-cmd-go {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }

  .seq-cmd-err {
    font-size: 11px;
    color: var(--err-strong, #dc2626);
  }

  .seq-readout {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-muted);
    white-space: nowrap;
  }

  .ro-pos {
    color: var(--text-secondary);
    font-weight: 600;
  }

  .ro-aa {
    color: var(--accent-bright);
    font-weight: 700;
  }

  .ro-three {
    color: var(--text-dim);
  }

  .ro-hint {
    font-family: inherit;
    font-size: 10.5px;
    color: var(--text-dim);
  }

  .seq-grid-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 8px 10px;
    max-height: 210px;
    overflow-y: auto;
  }

  .seq-grid-box.input-warn {
    border-color: var(--warn-border, #f59e0b);
  }

  .seq-entry-hdr {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-dim);
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .seq-blocks {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 10px;
    align-items: flex-start;
  }

  .seq-blk {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .seq-ruler {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    line-height: 1;
    color: var(--text-dim);
    padding: 0 2px;
    user-select: none;
  }

  .seq-cells {
    display: inline-flex;
    align-items: center;
    gap: 1px;
  }

  .chain-sep {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-dim);
    padding: 0 3px;
    user-select: none;
  }

  .aa-cell {
    width: 17px;
    height: 21px;
    padding: 0;
    border: 1px solid transparent;
    border-bottom-width: 2px;
    border-radius: 3px;
    background: transparent;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: background 90ms ease, border-color 90ms ease;
  }

  .aa-cell:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    border-color: var(--border-strong);
  }

  .aa-cell:focus-visible {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25);
  }

  .aa-cell.selected {
    background: var(--bg-pill);
    border-color: var(--accent);
    color: var(--accent-bright);
  }

  .aa-cell.mutated {
    background: rgba(16, 185, 129, 0.14);
    border-color: rgba(16, 185, 129, 0.55);
    color: var(--ok-text, #059669);
  }

  .aa-cell.mutated.selected {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
  }

  /* Subtle biochemical group indicator on bottom edge */
  .aa-cell.grp-hydrophobic { border-bottom-color: rgba(100, 116, 139, 0.28); }
  .aa-cell.grp-aromatic    { border-bottom-color: rgba(168, 85, 247, 0.36); }
  .aa-cell.grp-polar       { border-bottom-color: rgba(20, 184, 166, 0.36); }
  .aa-cell.grp-positive    { border-bottom-color: rgba(59, 130, 246, 0.42); }
  .aa-cell.grp-negative    { border-bottom-color: rgba(239, 68, 68, 0.42); }
  .aa-cell.grp-special     { border-bottom-color: rgba(245, 158, 11, 0.42); }

  .mut-picker {
    background: var(--bg-surface-2);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 7px 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .mut-picker-top {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .mut-picker-title {
    font-size: 11.5px;
    color: var(--text-primary);
  }

  .mut-picker-title b {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--accent-bright);
  }

  .mut-picker-sub {
    margin-left: auto;
    font-size: 10.5px;
    color: var(--text-dim);
  }

  .mut-close {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 15px;
    line-height: 1;
    padding: 0 3px;
    cursor: pointer;
    border-radius: 4px;
  }

  .mut-close:hover {
    color: var(--text-primary);
    background: var(--bg-surface-hover);
  }

  .aa-palette {
    display: grid;
    grid-template-columns: repeat(10, minmax(0, 1fr));
    gap: 3px;
  }

  .pal-btn {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-bottom-width: 2px;
    border-radius: 5px;
    padding: 3px 1px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1px;
    cursor: pointer;
    color: var(--text-primary);
  }

  .pal-btn:hover:not(:disabled):not(.current) {
    border-color: var(--accent);
    background: var(--bg-surface-hover);
  }

  .pal-btn.current {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }

  .pal-btn.current .pal-three {
    color: rgba(255, 255, 255, 0.82);
  }

  .pal-btn.grp-hydrophobic:not(.current) { border-bottom-color: rgba(100, 116, 139, 0.45); }
  .pal-btn.grp-aromatic:not(.current)    { border-bottom-color: rgba(168, 85, 247, 0.55); }
  .pal-btn.grp-polar:not(.current)       { border-bottom-color: rgba(20, 184, 166, 0.55); }
  .pal-btn.grp-positive:not(.current)    { border-bottom-color: rgba(59, 130, 246, 0.6); }
  .pal-btn.grp-negative:not(.current)    { border-bottom-color: rgba(239, 68, 68, 0.6); }
  .pal-btn.grp-special:not(.current)     { border-bottom-color: rgba(245, 158, 11, 0.6); }

  .pal-code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11.5px;
    font-weight: 700;
    line-height: 1.05;
  }

  .pal-three {
    font-size: 8.5px;
    color: var(--text-dim);
    line-height: 1;
  }

  .seq-raw-textarea {
    width: 100%;
    box-sizing: border-box;
    background: var(--bg-surface);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 8px 10px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
    line-height: 1.5;
    resize: vertical;
  }

  .seq-raw-textarea:focus {
    outline: none;
    border-color: var(--accent);
  }

  .seq-raw-textarea.input-error {
    border-color: var(--err-strong, #ef4444);
  }

  .seq-raw-textarea.input-warn {
    border-color: var(--warn-border, #f59e0b);
  }

  .seq-foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
    min-height: 18px;
  }

  .field-hint {
    font-size: 11px;
    color: var(--text-dim);
  }

  .field-hint.ok {
    color: var(--ok-text, #16a34a);
  }

  .field-hint.warn {
    color: var(--warn-text, #d97706);
  }

  .field-hint.err {
    color: var(--err-strong, #dc2626);
  }

  .seq-foot-right {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .raw-cursor-chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-muted);
  }

  .mut-diff-list {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
  }

  .mut-diff-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    color: var(--ok-text, #059669);
    background: var(--ok-bg, rgba(16, 185, 129, 0.1));
    border: 1px solid var(--ok-border, rgba(16, 185, 129, 0.35));
    border-radius: 5px;
    padding: 1px 6px;
    cursor: pointer;
  }

  .mut-diff-more {
    font-size: 10.5px;
    color: var(--text-dim);
  }

  .mut-revert-btn {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    padding: 1px 4px;
    text-decoration: underline;
  }

  .mut-revert-btn:hover:not(:disabled) {
    color: var(--text-primary);
  }
</style>
