<script lang="ts">
  import { PORT_COLORS } from '$lib/engine/graph-utils';
  import { analyzeSequence, previewTextLines, type PeekSlot } from '$lib/engine/peek';
  import type {
    ConditionSpecArtifact,
    DesignabilityMetrics,
    FoldedStructure,
    SequenceSet,
    StructureArtifact,
    WorkflowNode,
  } from '$lib/engine/types';
  import PaePlot from '$lib/viewer/PaePlot.svelte';

  interface Props {
    node: WorkflowNode | null;
    slots: PeekSlot[];
    activeSlot: PeekSlot | null;
    contextStructure?: StructureArtifact | null;
    selectedRowIdx: number;
    selectedFoldIdx: number;
    scRmsdThreshold: number;
    plddtThreshold: number;
    isRunning: boolean;
    embedded?: boolean;
    hasScorerResult?: boolean;
    onSelectSlot: (slotId: string) => void;
    onSelectRow: (idx: number) => void;
    onSelectFold: (idx: number) => void;
    onRunUpToNode?: (nodeId: string) => void;
    onJumpToScorer?: () => void;
    onDownloadText: (text: string, filename: string) => void;
  }

  let {
    node,
    slots,
    activeSlot,
    contextStructure = null,
    selectedRowIdx,
    selectedFoldIdx,
    scRmsdThreshold,
    plddtThreshold,
    isRunning,
    embedded = false,
    hasScorerResult = false,
    onSelectSlot,
    onSelectRow,
    onSelectFold,
    onRunUpToNode = undefined,
    onJumpToScorer = undefined,
    onDownloadText,
  }: Props = $props();

  let copiedKey = $state<string | null>(null);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;

  async function copyText(text: string, key: string) {
    try {
      await navigator.clipboard.writeText(text);
      copiedKey = key;
      if (copyTimer) clearTimeout(copyTimer);
      copyTimer = setTimeout(() => {
        copiedKey = null;
      }, 1600);
    } catch {
      /* clipboard unavailable */
    }
  }

  let inSlots = $derived(slots.filter((s) => s.direction === 'in'));
  let outSlots = $derived(slots.filter((s) => s.direction === 'out'));

  let artifact = $derived(activeSlot?.artifact ?? null);
  let metricsArt = $derived(artifact?.kind === 'DesignabilityMetrics' ? (artifact as DesignabilityMetrics) : null);
  let foldArt = $derived(artifact?.kind === 'FoldedStructure' ? (artifact as FoldedStructure) : null);
  let seqArt = $derived(artifact?.kind === 'SequenceSet' ? (artifact as SequenceSet) : null);
  let structArt = $derived(artifact?.kind === 'StructureArtifact' ? (artifact as StructureArtifact) : null);
  let condArt = $derived(artifact?.kind === 'ConditionSpec' ? (artifact as ConditionSpecArtifact) : null);

  let activeRow = $derived(metricsArt?.rows[selectedRowIdx] ?? metricsArt?.best ?? null);
  let activeFold = $derived(foldArt?.folds[selectedFoldIdx] ?? foldArt?.folds[0] ?? null);

  function formatFasta(seqSet: SequenceSet): string {
    return seqSet.candidates.map((c) => `>${c.id}\n${c.sequence}`).join('\n') + '\n';
  }

  function plddtSummary(arr: Float32Array) {
    if (!arr || arr.length === 0) return null;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let ge70 = 0;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      if (v >= 70) ge70++;
    }
    return {
      min,
      max,
      mean: sum / arr.length,
      ge70Pct: (ge70 / arr.length) * 100,
    };
  }
</script>

<div class="panel" class:embedded>
  <div class="panel-head">
    <div class="head-left">
      <span class="section-title">
        {node ? `PEEK · ${node.label ?? node.adapterId}` : 'PIPELINE PEEK'}
      </span>
      {#if activeSlot}
        <span class="dir-badge" class:is-in={activeSlot.direction === 'in'}>
          {activeSlot.direction === 'in' ? 'INPUT' : 'OUTPUT'} · {activeSlot.portName}
        </span>
      {/if}
    </div>
    <div class="head-right">
      {#if metricsArt}
        <span class="pass-pill" class:good={metricsArt.passRate > 0}>
          {(metricsArt.passRate * 100).toFixed(0)}% designable
        </span>
      {:else if hasScorerResult && onJumpToScorer}
        <button class="jump-btn" onclick={onJumpToScorer}>Scorer summary →</button>
      {/if}
    </div>
  </div>

  <!-- Port Peek Selector (Inputs vs Outputs of the selected step) -->
  {#if slots.length > 0}
    <div class="slot-bar">
      {#if inSlots.length > 0}
        <div class="slot-group">
          <span class="grp-lbl">IN</span>
          {#each inSlots as slot (slot.id)}
            {@const col = PORT_COLORS[slot.portType]?.bg ?? '#64748b'}
            <button
              class="slot-pill"
              class:active={activeSlot?.id === slot.id}
              class:has-data={slot.artifact !== null}
              class:unwired={!slot.wired}
              onclick={() => onSelectSlot(slot.id)}
              title={slot.wired
                ? `Peek input "${slot.portName}" (${slot.portType}) from ${slot.peerNodeLabel ?? slot.peerNodeId}`
                : `Optional input "${slot.portName}" (${slot.portType}) is not connected`}
            >
              <span class="pdot" style="background: {col}"></span>
              <span>{slot.portName}</span>
              {#if slot.peerNodeLabel}
                <span class="peer">← {slot.peerNodeLabel}</span>
              {:else if !slot.wired}
                <span class="peer">unwired</span>
              {/if}
            </button>
          {/each}
        </div>
      {/if}

      {#if outSlots.length > 0}
        <div class="slot-group">
          <span class="grp-lbl">OUT</span>
          {#each outSlots as slot (slot.id)}
            {@const col = PORT_COLORS[slot.portType]?.bg ?? '#64748b'}
            <button
              class="slot-pill"
              class:active={activeSlot?.id === slot.id}
              class:has-data={slot.artifact !== null}
              onclick={() => onSelectSlot(slot.id)}
              title={`Peek output "${slot.portName}" (${slot.portType})`}
            >
              <span class="pdot" style="background: {col}"></span>
              <span>{slot.portName}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <!-- Modality-Specific Content -->
  {#if metricsArt && metricsArt.rows.length > 0}
    <table class="scores">
      <thead>
        <tr>
          <th>Seq</th>
          <th>scRMSD</th>
          <th>scTM</th>
          <th>pLDDT</th>
          <th>pTM</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {#each metricsArt.rows as row, i (`${row.sequenceId}_${i}`)}
          <tr class:selected={i === selectedRowIdx} onclick={() => onSelectRow(i)}>
            <td><code>{row.sequenceId}</code></td>
            <td class:good-val={row.scRmsd < scRmsdThreshold}>{row.scRmsd.toFixed(2)} Å</td>
            <td>{row.scTm.toFixed(3)}</td>
            <td class:good-val={row.meanPlddt >= plddtThreshold}>{row.meanPlddt.toFixed(1)}</td>
            <td>{row.ptm !== null ? row.ptm.toFixed(2) : '—'}</td>
            <td>
              <span class="verdict" class:pass={row.designable}>
                {row.designable ? 'PASS' : 'FAIL'}
              </span>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>

    {#if activeRow}
      {@const seqStats = analyzeSequence(activeRow.sequence)}
      <div class="seq-box">
        <div class="seq-head">
          <span>Selected Sequence ({activeRow.sequence.length} aa · {seqStats.uniqueAaCount} unique AAs)</span>
          <div class="dl-links">
            <button class="link-btn" onclick={() => copyText(activeRow.sequence, 'metric-seq')}>
              {copiedKey === 'metric-seq' ? '✓ Copied' : 'Copy seq'}
            </button>
            <button
              class="link-btn"
              onclick={() => onDownloadText(activeRow.designedPdb, `${activeRow.structureId}_designed.pdb`)}
            >
              ↓ Designed PDB
            </button>
            <button
              class="link-btn"
              onclick={() => onDownloadText(activeRow.alignedFoldPdb, `${activeRow.sequenceId}_refold_aligned.pdb`)}
            >
              ↓ Refold PDB
            </button>
          </div>
        </div>
        {#if seqStats.warnings.length > 0}
          <div class="warn-chips">
            {#each seqStats.warnings as w (w)}
              <span class="warn-chip">⚠ {w}</span>
            {/each}
          </div>
        {/if}
        <div class="seq-text">{activeRow.sequence}</div>
      </div>

      <div class="pae-row">
        <PaePlot pae={activeRow.pae} size={200} />
        <div class="pae-notes">
          <div><b>Fold time:</b> {(activeRow.foldElapsedMs / 1000).toFixed(1)} s</div>
          <div><b>Refold R<sub>g</sub>:</b> {activeRow.geometry.rg.toFixed(1)} Å</div>
          <div><b>Refold α/β:</b> {activeRow.geometry.helixPct.toFixed(0)}% / {activeRow.geometry.sheetPct.toFixed(0)}%</div>
          <p class="tip">
            Tip: Click any block, port label, or colored wire on the canvas to peek at intermediate structures, sequences, or conditioning specs!
          </p>
        </div>
      </div>
    {/if}
  {:else if foldArt && foldArt.folds.length > 0}
    <table class="scores">
      <thead>
        <tr>
          <th>Seq</th>
          <th>Len</th>
          <th>pLDDT</th>
          <th>pTM</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        {#each foldArt.folds as f, i (`${f.sequenceId}_${i}`)}
          <tr class:selected={i === selectedFoldIdx} onclick={() => onSelectFold(i)}>
            <td><code>{f.sequenceId}</code></td>
            <td>{f.sequence.length} aa</td>
            <td class:good-val={f.meanPlddt >= plddtThreshold}>{f.meanPlddt.toFixed(1)}</td>
            <td>{f.ptm !== null ? f.ptm.toFixed(2) : '—'}</td>
            <td>{(f.elapsedMs / 1000).toFixed(1)}s</td>
          </tr>
        {/each}
      </tbody>
    </table>

    {#if activeFold}
      {@const pStats = plddtSummary(activeFold.plddt)}
      <div class="seq-box">
        <div class="seq-head">
          <span>{activeFold.sequenceId} · {activeFold.foldModelId}</span>
          <div class="dl-links">
            <button class="link-btn" onclick={() => copyText(activeFold.sequence, 'fold-seq')}>
              {copiedKey === 'fold-seq' ? '✓ Copied' : 'Copy seq'}
            </button>
            <button
              class="link-btn"
              onclick={() => onDownloadText(activeFold.pdbText, `${activeFold.sequenceId}_af2.pdb`)}
            >
              ↓ Folded PDB
            </button>
          </div>
        </div>
        <div class="seq-text">{activeFold.sequence}</div>
      </div>

      <div class="pae-row">
        <PaePlot pae={activeFold.pae} size={200} />
        <div class="pae-notes">
          <div><b>Mean pLDDT:</b> {activeFold.meanPlddt.toFixed(1)}</div>
          {#if pStats}
            <div><b>pLDDT range:</b> {pStats.min.toFixed(1)} – {pStats.max.toFixed(1)}</div>
            <div><b>Confident (≥70):</b> {pStats.ge70Pct.toFixed(0)}%</div>
          {/if}
          {#if activeFold.ptm !== null}
            <div><b>pTM:</b> {activeFold.ptm.toFixed(3)}</div>
          {/if}
          <div><b>Fold time:</b> {(activeFold.elapsedMs / 1000).toFixed(1)} s</div>
        </div>
      </div>
    {/if}
  {:else if seqArt}
    <div class="meta-bar">
      <span><b>{seqArt.candidates.length}</b> sequence{seqArt.candidates.length === 1 ? '' : 's'} · <code>{seqArt.sourceModelId}</code></span>
      <div class="dl-links">
        <button class="link-btn" onclick={() => copyText(formatFasta(seqArt), 'all-fasta')}>
          {copiedKey === 'all-fasta' ? '✓ Copied FASTA' : 'Copy FASTA'}
        </button>
        <button
          class="link-btn"
          onclick={() => onDownloadText(formatFasta(seqArt), `${seqArt.id}.fasta`)}
        >
          ↓ FASTA
        </button>
      </div>
    </div>

    <div class="seq-list">
      {#each seqArt.candidates as c, i (`${c.id}_${i}`)}
        {@const st = analyzeSequence(c.sequence, contextStructure?.sequence)}
        <div class="seq-box">
          <div class="seq-head">
            <span class="cand-title">
              <code>{c.id}</code> · {st.length} aa
              {#if c.chainLengths && c.chainLengths.length > 1}
                · chains [{c.chainLengths.join('+')}]
              {/if}
              {#if c.score !== null}
                · score {c.score.toFixed(2)}
              {/if}
              {#if st.recoveryPct !== null}
                · recovery {st.recoveryPct.toFixed(0)}%
              {/if}
            </span>
            <button class="link-btn" onclick={() => copyText(c.sequence, `cand-${i}`)}>
              {copiedKey === `cand-${i}` ? '✓ Copied' : 'Copy'}
            </button>
          </div>
          <div class="stat-chips">
            <span class="stat-chip">{st.uniqueAaCount} unique AAs</span>
            <span class="stat-chip">hydro {st.hydrophobicPct.toFixed(0)}%</span>
            <span class="stat-chip">polar {st.polarPct.toFixed(0)}%</span>
            <span class="stat-chip">charged {st.chargedPct.toFixed(0)}%</span>
            {#if st.topAa}
              <span class="stat-chip">top: {st.topAa.aa} ({st.topAa.pct.toFixed(0)}%)</span>
            {/if}
          </div>
          {#if st.warnings.length > 0}
            <div class="warn-chips">
              {#each st.warnings as w (w)}
                <span class="warn-chip">⚠ {w}</span>
              {/each}
            </div>
          {/if}
          <div class="seq-text">{c.sequence}</div>
        </div>
      {/each}
    </div>
  {:else if structArt}
    {@const st = analyzeSequence(structArt.sequence)}
    {@const rawPrev = previewTextLines(structArt.pdbText, 36)}
    <div class="meta-bar">
      <span><b>{structArt.label}</b> · <code>{structArt.sourceModelId}</code></span>
      <div class="dl-links">
        <button class="link-btn" onclick={() => onDownloadText(structArt.pdbText, `${structArt.id}.pdb`)}>
          ↓ PDB
        </button>
        {#if structArt.cifText}
          <button class="link-btn" onclick={() => onDownloadText(structArt.cifText!, `${structArt.id}.cif`)}>
            ↓ mmCIF
          </button>
        {/if}
        {#if structArt.trajectoryPdb}
          <button
            class="link-btn"
            onclick={() => onDownloadText(structArt.trajectoryPdb!, `${structArt.id}_traj.pdb`)}
          >
            ↓ Trajectory ({structArt.trajectoryFrameCount ?? '?'}f)
          </button>
        {/if}
      </div>
    </div>

    <div class="stat-chips">
      <span class="stat-chip">L = {structArt.length} aa</span>
      <span class="stat-chip">R<sub>g</sub> = {structArt.geometry.rg.toFixed(1)} Å</span>
      <span class="stat-chip">α {structArt.geometry.helixPct.toFixed(0)}% · β {structArt.geometry.sheetPct.toFixed(0)}%</span>
      <span class="stat-chip">breaks: {structArt.geometry.chainBreaks}</span>
      <span class="stat-chip">clashes: {structArt.geometry.clashes}</span>
      {#if structArt.trajectoryFrameCount}
        <span class="stat-chip">{structArt.trajectoryFrameCount} traj frames</span>
      {/if}
    </div>

    <div class="seq-box">
      <div class="seq-head">
        <span>Backbone residue sequence ({structArt.sequence.length} aa · {st.uniqueAaCount} unique AAs)</span>
        <button class="link-btn" onclick={() => copyText(structArt.sequence, 'struct-seq')}>
          {copiedKey === 'struct-seq' ? '✓ Copied' : 'Copy seq'}
        </button>
      </div>
      {#if st.warnings.length > 0}
        <div class="warn-chips">
          {#each st.warnings as w (w)}
            <span class="warn-chip">⚠ {w}</span>
          {/each}
        </div>
      {/if}
      <div class="seq-text">{structArt.sequence}</div>
    </div>

    <details class="raw-details">
      <summary>Raw PDB preview ({rawPrev.totalLines} lines)</summary>
      <pre class="raw-pre">{rawPrev.preview}{rawPrev.truncated ? `\n… (${rawPrev.totalLines - 36} more lines)` : ''}</pre>
    </details>
  {:else if condArt}
    {@const condKeys = Object.keys(condArt.conditions ?? {})}
    {@const condJson = JSON.stringify(
      {
        contig: condArt.contig,
        conditions: condArt.conditions,
        ...(condArt.cifUrl ? { cifUrl: condArt.cifUrl } : {}),
        ...(condArt.partialFraction != null ? { partialFraction: condArt.partialFraction } : {}),
        ...(condArt.inverseFolding ? { inverseFolding: true } : {}),
        ...(condArt.inputCif ? { inlineTemplateChars: condArt.inputCif.length } : {}),
      },
      null,
      2,
    )}
    <div class="meta-bar">
      <span><b>{condArt.label}</b></span>
      <button class="link-btn" onclick={() => copyText(condJson, 'cond-json')}>
        {copiedKey === 'cond-json' ? '✓ Copied JSON' : 'Copy spec JSON'}
      </button>
    </div>

    <div class="stat-chips">
      {#if condArt.contig}
        <span class="stat-chip">contig: {condArt.contig}</span>
      {/if}
      {#if condArt.cifUrl}
        <span class="stat-chip">source: {condArt.cifUrl}</span>
      {/if}
      {#if condArt.inputCif}
        <span class="stat-chip">template: inline ({(condArt.inputCif.length / 1024).toFixed(1)} KB)</span>
      {/if}
      {#each condKeys as k (k)}
        <span class="stat-chip cond-key">{k}</span>
      {/each}
    </div>

    <pre class="raw-pre">{condJson}</pre>
  {:else}
    <div class="placeholder-box">
      {#if activeSlot && !activeSlot.wired}
        Optional input <code>{activeSlot.portName}</code> (<code>{activeSlot.portType}</code>) is not connected.
        <br />
        Drag a matching colored wire into this port on the canvas to pass data in.
      {:else if activeSlot}
        No data on <b>{activeSlot.direction === 'in' ? 'IN' : 'OUT'} · {activeSlot.portName}</b> yet.
        <br />
        {#if node && onRunUpToNode}
          <button
            class="run-to-btn"
            disabled={isRunning}
            onclick={() => onRunUpToNode(node.id)}
          >
            {isRunning ? 'Running…' : `▶ Run up to ${node.label ?? node.adapterId}`}
          </button>
        {/if}
      {:else}
        Wire any blocks on the canvas (or pick a preset above) and click <b>▶ Run</b>.
      {/if}
      <div class="hint-foot">
        Click any block, port label, or colored wire to peek at intermediate structures, sequences, folds, or conditioning specs.
      </div>
    </div>
  {/if}
</div>

<style>
  .panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .panel.embedded {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 12px 14px 14px;
  }
  .panel-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .head-left {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .head-right {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .section-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: var(--text-muted);
  }
  .dir-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 4px;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
  }
  .dir-badge.is-in {
    background: var(--bg-surface-2);
    color: var(--text-secondary);
  }
  .jump-btn {
    background: var(--bg-surface-2);
    color: var(--accent-bright);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 3px 8px;
    font-size: 11px;
    cursor: pointer;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .jump-btn:hover {
    background: var(--bg-surface-hover);
  }
  .pass-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--warn-bg);
    color: var(--warn-text);
  }
  .pass-pill.good {
    background: var(--ok-bg);
    color: var(--ok-text);
  }

  .slot-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 12px;
    padding: 8px 10px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    align-items: center;
  }
  .slot-group {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
  }
  .grp-lbl {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    font-weight: 700;
    color: var(--text-dim);
    letter-spacing: 0.6px;
    margin-right: 2px;
  }
  .slot-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    cursor: pointer;
  }
  .slot-pill:hover {
    border-color: var(--border-strong);
    color: var(--text-primary);
  }
  .slot-pill.has-data {
    color: var(--text-body);
  }
  .slot-pill.unwired {
    opacity: 0.55;
    border-style: dashed;
  }
  .slot-pill.active {
    border-color: var(--accent);
    background: var(--bg-surface-selected);
    color: var(--text-primary);
    font-weight: 600;
  }
  .pdot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .peer {
    color: var(--text-dim);
    font-size: 10px;
    font-weight: 400;
  }

  .meta-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--text-secondary);
  }
  .stat-chips,
  .warn-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 6px;
  }
  .stat-chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    padding: 2px 7px;
    border-radius: 4px;
  }
  .stat-chip.cond-key {
    color: var(--accent-soft);
    background: var(--bg-pill);
    border-color: var(--border-strong);
  }
  .warn-chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    padding: 2px 7px;
    border-radius: 4px;
  }

  table.scores {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  table.scores th {
    text-align: left;
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-dim);
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
  }
  table.scores td {
    padding: 7px 8px;
    border-bottom: 1px solid var(--border-subtle);
  }
  table.scores tr {
    cursor: pointer;
  }
  table.scores tr:hover {
    background: var(--bg-surface-hover);
  }
  table.scores tr.selected {
    background: var(--bg-surface-selected);
  }
  .good-val {
    color: var(--ok-text);
    font-weight: 600;
  }
  .verdict {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--err-bg);
    color: var(--err-text);
  }
  .verdict.pass {
    background: var(--ok-bg);
    color: var(--ok-text);
  }

  .seq-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 360px;
    overflow-y: auto;
  }
  .seq-box {
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }
  .seq-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }
  .cand-title {
    color: var(--text-secondary);
  }
  .dl-links {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .link-btn {
    background: none;
    border: none;
    color: var(--accent-bright);
    font-size: 11px;
    cursor: pointer;
    padding: 0;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .link-btn:hover {
    text-decoration: underline;
  }
  .seq-text {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-secondary);
    word-break: break-all;
    line-height: 1.5;
  }

  .pae-row {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    flex-wrap: wrap;
  }
  .pae-notes {
    flex: 1;
    min-width: 160px;
    font-size: 12px;
    color: var(--text-secondary);
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .tip {
    margin: 8px 0 0;
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1.45;
  }

  .raw-details summary {
    cursor: pointer;
    font-size: 11px;
    color: var(--accent-soft);
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .raw-pre {
    margin: 0;
    padding: 10px;
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-secondary);
    overflow: auto;
    max-height: 240px;
    line-height: 1.45;
    white-space: pre;
  }

  .placeholder-box {
    padding: 20px 12px;
    text-align: center;
    font-size: 12px;
    color: var(--text-dim);
    line-height: 1.6;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
  }
  .run-to-btn {
    background: #2563eb;
    color: #fff;
    border: 1px solid #3b82f6;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .run-to-btn:hover:not(:disabled) {
    background: #1d4ed8;
  }
  .run-to-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .hint-foot {
    font-size: 11px;
    color: var(--text-dim);
    max-width: 360px;
  }
  code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
</style>
