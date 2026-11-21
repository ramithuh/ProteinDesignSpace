<script lang="ts">
  import { ADAPTER_REGISTRY } from '$lib/engine/registry';
  import { PORT_COLORS } from '$lib/engine/graph-utils';
  import {
    analyzeSequence,
    foldModelSlug,
    formatFoldModelName,
    previewTextLines,
    type PeekSlot,
  } from '$lib/engine/peek';
  import type {
    ArtifactProvenance,
    CollectionKind,
    ConditionSpecArtifact,
    DesignabilityMetrics,
    FoldedStructure,
    MsaMode,
    ReferenceRelation,
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
    ipaeThreshold?: number;
    motifRmsdThreshold?: number;
    ipsaeThreshold?: number;
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
    ipaeThreshold = 10.0,
    motifRmsdThreshold = 1.0,
    ipsaeThreshold = 0,
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
  let copyFailedKey = $state<string | null>(null);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;
  let rawPreviewOpen = $state(false);

  async function copyText(text: string, key: string) {
    if (copyTimer) clearTimeout(copyTimer);
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error('Clipboard API unavailable');
      }
      await navigator.clipboard.writeText(text);
      copyFailedKey = null;
      copiedKey = key;
      copyTimer = setTimeout(() => {
        copiedKey = null;
      }, 1600);
    } catch {
      copiedKey = null;
      copyFailedKey = key;
      copyTimer = setTimeout(() => {
        copyFailedKey = null;
      }, 2000);
    }
  }

  function copyBtnLabel(key: string, idleLabel: string, copiedLabel = '✓ Copied'): string {
    if (copiedKey === key) return copiedLabel;
    if (copyFailedKey === key) return 'Copy failed';
    return idleLabel;
  }

  function isFiniteNum(v: unknown): v is number {
    return typeof v === 'number' && Number.isFinite(v);
  }

  function fmtNum(v: number | null | undefined, digits: number, suffix = ''): string {
    if (!isFiniteNum(v)) return '—';
    return `${v.toFixed(digits)}${suffix}`;
  }

  function safeFilename(name: string): string {
    return name.replace(/[/:\\*?<>|]+/g, '_');
  }

  function formatCollectionKind(kind?: CollectionKind): string | null {
    if (kind === 'independent-candidates') return 'Independent candidates';
    if (kind === 'multi-sample') return 'Multi-sample';
    if (kind === 'ordered-frames') return 'Ordered frames';
    return null;
  }

  function formatReferenceRelationBadge(
    relation?: ReferenceRelation,
    lineageMismatch?: boolean,
  ): { text: string; tone: 'ok' | 'info' | 'warn' | 'err' } | null {
    if (relation === 'self-consistency') {
      return { text: 'Self-consistency', tone: 'ok' };
    }
    if (relation === 'external-mapped') {
      return { text: 'External ref (mapped)', tone: 'info' };
    }
    if (relation === 'external-unlinked') {
      return { text: 'External ref', tone: 'warn' };
    }
    if (relation === 'external-mismatch' || lineageMismatch) {
      return { text: 'External ref · Lineage mismatch', tone: 'err' };
    }
    return null;
  }

  function isSelfGeneratorId(id?: string): boolean {
    if (!id) return false;
    return id.startsWith('rfd4-proteina') || id.startsWith('genie3');
  }

  function formatMsaBadge(msaMode?: MsaMode, msaDepth?: number): string {
    const depth = isFiniteNum(msaDepth) && msaDepth > 0 ? Math.round(msaDepth) : null;
    if (msaMode === 'mmseqs2') {
      return depth !== null ? `MMseqs2 · ${depth} seqs` : 'MMseqs2';
    }
    if (msaMode === 'target-only') {
      return depth !== null ? `Target MSA · ${depth} seqs` : 'Target MSA';
    }
    if (msaMode === 'a3m') {
      return depth !== null ? `.a3m · ${depth} seqs` : '.a3m';
    }
    return 'single-seq';
  }

  function formatQuantChip(provenance?: ArtifactProvenance, foldModelId?: string): string | null {
    if (provenance?.quantization) {
      return provenance.quantization.toUpperCase();
    }
    if (foldModelId && foldModelId.toLowerCase().includes('-int5')) {
      return 'INT5';
    }
    return null;
  }

  function computeIpaeFromPae(
    pae: Float32Array | undefined,
    chainLengths: readonly number[] | undefined,
  ): number | null {
    if (!pae || !chainLengths || chainLengths.length < 2) return null;
    const L = chainLengths.reduce((acc, n) => acc + n, 0);
    if (L <= 0 || pae.length < L * L) return null;
    const dim = Math.round(Math.sqrt(pae.length));
    if (dim < L || dim * dim !== pae.length) return null;
    const chainOf = new Int32Array(L);
    let off = 0;
    for (let c = 0; c < chainLengths.length; c++) {
      for (let k = 0; k < chainLengths[c]; k++) chainOf[off++] = c;
    }
    let sum = 0;
    let count = 0;
    for (let i = 0; i < L; i++) {
      const ci = chainOf[i];
      for (let j = 0; j < L; j++) {
        if (ci === chainOf[j]) continue;
        const v = pae[i * dim + j];
        if (Number.isFinite(v)) {
          sum += v;
          count++;
        }
      }
    }
    return count > 0 ? sum / count : null;
  }

  function inferPdbChainLengths(pdbText?: string): number[] | undefined {
    if (!pdbText) return undefined;
    const counts: number[] = [];
    const seen = new Set<string>();
    let lastChain = '';
    for (const line of pdbText.split('\n')) {
      if (!line.startsWith('ATOM  ') || line.length < 27) continue;
      if (line.slice(12, 16).trim() !== 'CA') continue;
      const chain = line[21] || 'A';
      const resKey = `${chain}:${line.slice(22, 27)}`;
      if (seen.has(resKey)) continue;
      seen.add(resKey);
      if (counts.length === 0 || chain !== lastChain) {
        counts.push(1);
        lastChain = chain;
      } else {
        counts[counts.length - 1] += 1;
      }
    }
    return counts.length > 1 ? counts : undefined;
  }

  let effectiveIpsaePass = $derived(ipsaeThreshold > 0 ? ipsaeThreshold : 0.6);

  let inSlots = $derived(slots.filter((s) => s.direction === 'in'));
  let outSlots = $derived(slots.filter((s) => s.direction === 'out'));

  let artifact = $derived(activeSlot?.artifact ?? null);
  let metricsArt = $derived(artifact?.kind === 'DesignabilityMetrics' ? (artifact as DesignabilityMetrics) : null);
  let foldArt = $derived(artifact?.kind === 'FoldedStructure' ? (artifact as FoldedStructure) : null);
  let seqArt = $derived(artifact?.kind === 'SequenceSet' ? (artifact as SequenceSet) : null);
  let structArt = $derived(artifact?.kind === 'StructureArtifact' ? (artifact as StructureArtifact) : null);
  let condArt = $derived(artifact?.kind === 'ConditionSpec' ? (artifact as ConditionSpecArtifact) : null);

  $effect(() => {
    void structArt?.id;
    rawPreviewOpen = false;
  });

  let activeRow = $derived(metricsArt?.rows[selectedRowIdx] ?? metricsArt?.best ?? null);
  let activeFold = $derived(foldArt?.folds[selectedFoldIdx] ?? foldArt?.folds[0] ?? null);
  let activeRowSeqStats = $derived(activeRow ? analyzeSequence(activeRow.sequence) : null);
  let structSeqStats = $derived(structArt ? analyzeSequence(structArt.sequence) : null);
  let skipSelfRecovery = $derived(
    Boolean(
      seqArt &&
        ((activeSlot?.direction === 'out' && isSelfGeneratorId(node?.adapterId)) ||
          (contextStructure?.id === seqArt.parentStructureId && isSelfGeneratorId(seqArt.sourceModelId))),
    ),
  );
  let seqCandidateStats = $derived.by(() => {
    if (!seqArt) return [];
    const refSeq = skipSelfRecovery ? undefined : contextStructure?.sequence;
    return seqArt.candidates.map((c) => analyzeSequence(c.sequence, refSeq));
  });
  let rawStructText = $derived(structArt ? (structArt.cifText ?? structArt.pdbText) : '');
  let rawPrev = $derived(
    rawPreviewOpen && structArt ? previewTextLines(rawStructText, 36) : null,
  );
  let metricsRefBadge = $derived(
    metricsArt
      ? formatReferenceRelationBadge(
          activeRow?.referenceRelation ?? metricsArt.referenceRelation,
          activeRow?.lineageMismatch ?? metricsArt.lineageMismatch,
        )
      : null,
  );
  let metricsRefLabel = $derived(
    activeRow?.referenceLabel ?? metricsArt?.referenceLabel ?? null,
  );
  let hasBinderMetrics = $derived(
    Boolean(
      metricsArt?.rows.some(
        (r) =>
          isFiniteNum(r.ipae) ||
          isFiniteNum(r.binderScRmsd) ||
          isFiniteNum(r.targetAlignedBinderRmsd),
      ),
    ),
  );
  let hasIptmMetrics = $derived(
    Boolean(metricsArt?.rows.some((r) => isFiniteNum(r.iptm))),
  );
  let hasIpsaeMetrics = $derived(
    Boolean(metricsArt?.rows.some((r) => isFiniteNum(r.ipsae))),
  );
  let hasMotifMetrics = $derived(
    Boolean(
      metricsArt?.rows.some(
        (r) => isFiniteNum(r.refoldMotifRmsd) || isFiniteNum(r.genMotifRmsd),
      ),
    ),
  );
  let activeRowChainLengths = $derived.by((): number[] | undefined => {
    if (!activeRow) return undefined;
    const fromPdb = inferPdbChainLengths(activeRow.alignedFoldPdb);
    if (fromPdb && fromPdb.length > 1) return fromPdb;
    const ctxLens = contextStructure?.chainLengths ?? contextStructure?.diagnostics?.chainLengths;
    return ctxLens && ctxLens.length > 1 ? ctxLens : undefined;
  });
  let hasFoldMultiChainMetrics = $derived(
    Boolean(
      foldArt?.folds.some(
        (f) =>
          isFiniteNum(f.iptm) ||
          isFiniteNum(f.ipsae) ||
          (f.chainLengths && f.chainLengths.length > 1),
      ),
    ),
  );

  function formatAdapterName(sourceModelId: string): string {
    return ADAPTER_REGISTRY[sourceModelId]?.shortName ?? sourceModelId;
  }

  const CONDITION_KEY_LABELS: Record<string, string> = {
    C_HOT: 'Binding Hotspots',
    C_CRDGRP: 'Fixed Motif',
    C_IDX: 'Unindexed Motif',
    C_HB: 'H-Bond Constraint',
    C_CATH: 'CATH Fold',
    C_CRD: 'Locked Backbone',
    C_SEQ: 'Locked Sequence',
  };

  function formatConditionKey(key: string): string {
    return CONDITION_KEY_LABELS[key] ?? key;
  }

  function formatCandidateSequence(sequence: string, chainLengths?: number[]): string {
    if (!chainLengths || chainLengths.length <= 1) return sequence;
    const sum = chainLengths.reduce((acc, n) => acc + n, 0);
    if (sum !== sequence.length) return sequence;
    const parts: string[] = [];
    let offset = 0;
    for (const len of chainLengths) {
      parts.push(sequence.slice(offset, offset + len));
      offset += len;
    }
    return parts.join(':');
  }

  function formatFasta(seqSet: SequenceSet): string {
    return (
      seqSet.candidates
        .map((c) => `>${c.id}\n${formatCandidateSequence(c.sequence, c.chainLengths)}`)
        .join('\n') + '\n'
    );
  }

  function plddtSummary(arr: Float32Array) {
    if (!arr || arr.length === 0) return null;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let ge70 = 0;
    let count = 0;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (!Number.isFinite(v)) continue;
      count++;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      if (v >= 70) ge70++;
    }
    if (count === 0) return null;
    return {
      min,
      max,
      mean: sum / count,
      ge70Pct: (ge70 / count) * 100,
    };
  }
</script>

<div class="panel" class:embedded>
  <div class="panel-head">
    <div class="head-left">
      <span class="section-title">
        {node ? `OUTPUTS · ${node.label ?? node.adapterId}` : 'PIPELINE OUTPUTS'}
      </span>
      {#if activeSlot}
        <span class="dir-badge" class:is-in={activeSlot.direction === 'in'}>
          {activeSlot.direction === 'in' ? 'INPUT' : 'OUTPUT'} · {activeSlot.portName}
        </span>
      {/if}
    </div>
    <div class="head-right">
      {#if metricsArt}
        <span class="pass-pill" class:good={isFiniteNum(metricsArt.passRate) && metricsArt.passRate > 0}>
          {isFiniteNum(metricsArt.passRate) ? `${(metricsArt.passRate * 100).toFixed(0)}% designable` : '—'}
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
            {@const typeLabel = PORT_COLORS[slot.portType]?.label ?? slot.portType}
            <button
              class="slot-pill"
              class:active={activeSlot?.id === slot.id}
              class:has-data={slot.artifact !== null}
              class:unwired={!slot.wired}
              onclick={() => onSelectSlot(slot.id)}
              title={slot.wired
                ? `Inspect input "${slot.portName}" (${typeLabel}) from ${slot.peerNodeLabel ?? slot.peerNodeId}`
                : `${slot.required ? 'Required' : 'Optional'} input "${slot.portName}" (${typeLabel}) is not connected`}
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
            {@const typeLabel = PORT_COLORS[slot.portType]?.label ?? slot.portType}
            <button
              class="slot-pill"
              class:active={activeSlot?.id === slot.id}
              class:has-data={slot.artifact !== null}
              onclick={() => onSelectSlot(slot.id)}
              title={`Inspect output "${slot.portName}" (${typeLabel})`}
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
    {#if metricsRefBadge || metricsArt.collectionKind}
      {@const colLabel = formatCollectionKind(metricsArt.collectionKind)}
      <div class="lineage-meta-bar">
        {#if metricsRefBadge}
          <span
            class="ref-pill"
            class:tone-ok={metricsRefBadge.tone === 'ok'}
            class:tone-info={metricsRefBadge.tone === 'info'}
            class:tone-warn={metricsRefBadge.tone === 'warn'}
            class:tone-err={metricsRefBadge.tone === 'err'}
            title={metricsRefLabel ?? metricsRefBadge.text}
          >
            {metricsRefBadge.text}
          </span>
        {/if}
        {#if metricsRefLabel}
          <span class="ref-desc">{metricsRefLabel}</span>
        {/if}
        {#if colLabel}
          <span class="stat-chip">{colLabel}</span>
        {/if}
      </div>
    {/if}
    {#if metricsArt.lineageWarning}
      <div class="lineage-warn-banner" role="status">
        ⚠ {metricsArt.lineageWarning}
      </div>
    {/if}

    <table class="scores">
      <thead>
        <tr>
          <th>Seq</th>
          <th>{metricsRefBadge && metricsRefBadge.tone !== 'ok' ? 'Ref RMSD' : 'scRMSD'}</th>
          <th>{metricsRefBadge && metricsRefBadge.tone !== 'ok' ? 'Ref TM' : 'scTM'}</th>
          <th>pLDDT</th>
          <th>pTM</th>
          {#if hasIptmMetrics || hasBinderMetrics || hasIpsaeMetrics}
            <th title="Interface predicted TM-score (multi-chain)">ipTM</th>
          {/if}
          {#if hasBinderMetrics}
            <th title={`Mean interface PAE (Å, pass < ${ipaeThreshold})`}>iPAE</th>
          {/if}
          {#if hasIpsaeMetrics || hasBinderMetrics}
            <th title={`Dunbrack ipSAE interface score (pass ≥ ${effectiveIpsaePass.toFixed(2)})`}>ipSAE</th>
          {/if}
          {#if hasBinderMetrics}
            <th title={`Target-aligned or independent binder Cα RMSD (Å, pass < ${scRmsdThreshold})`}>Binder RMSD</th>
          {/if}
          {#if hasMotifMetrics}
            <th title={`Refolded motif Cα RMSD vs native template (Å, pass < ${motifRmsdThreshold})`}>Motif RMSD</th>
          {/if}
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {#each metricsArt.rows as row, i (`${row.sequenceId}_${i}`)}
          {@const bRmsd = isFiniteNum(row.targetAlignedBinderRmsd) ? row.targetAlignedBinderRmsd : row.binderScRmsd}
          {@const rowConfAvail = row.confidenceAvailable !== false && isFiniteNum(row.meanPlddt)}
          {@const rowFailedGates = row.failedGates ?? []}
          <tr class:selected={i === selectedRowIdx} onclick={() => onSelectRow(i)}>
            <td><code>{row.sequenceId}</code></td>
            <td class:good-val={isFiniteNum(row.scRmsd) && row.scRmsd < scRmsdThreshold}>{fmtNum(row.scRmsd, 2, ' Å')}</td>
            <td>{fmtNum(row.scTm, 3)}</td>
            <td class:good-val={rowConfAvail && row.meanPlddt >= plddtThreshold}>
              {rowConfAvail ? fmtNum(row.meanPlddt, 1) : 'N/A'}
            </td>
            <td>{fmtNum(row.ptm, 2)}</td>
            {#if hasIptmMetrics || hasBinderMetrics || hasIpsaeMetrics}
              <td>{fmtNum(row.iptm, 2)}</td>
            {/if}
            {#if hasBinderMetrics}
              <td class:good-val={isFiniteNum(row.ipae) && row.ipae < ipaeThreshold}>
                {fmtNum(row.ipae, 1, ' Å')}
              </td>
            {/if}
            {#if hasIpsaeMetrics || hasBinderMetrics}
              <td class:good-val={isFiniteNum(row.ipsae) && row.ipsae >= effectiveIpsaePass}>
                {fmtNum(row.ipsae, 2)}
              </td>
            {/if}
            {#if hasBinderMetrics}
              <td class:good-val={isFiniteNum(bRmsd) && bRmsd < scRmsdThreshold}>
                {fmtNum(bRmsd, 2, ' Å')}
              </td>
            {/if}
            {#if hasMotifMetrics}
              <td class:good-val={isFiniteNum(row.refoldMotifRmsd) && row.refoldMotifRmsd < motifRmsdThreshold}>
                {fmtNum(row.refoldMotifRmsd, 2, ' Å')}
              </td>
            {/if}
            <td>
              <div class="verdict-cell">
                <span
                  class="verdict"
                  class:pass={row.designable}
                  title={row.designable
                    ? 'All designability gates passed'
                    : rowFailedGates.length > 0
                      ? `Failed gates: ${rowFailedGates.join(' · ')}`
                      : 'Failed designability gate'}
                >
                  {row.designable ? 'PASS' : 'FAIL'}
                </span>
                {#if !row.designable && rowFailedGates.length > 0}
                  <span class="gate-summary" title={rowFailedGates.join(' · ')}>
                    {rowFailedGates[0]}{rowFailedGates.length > 1 ? ` +${rowFailedGates.length - 1}` : ''}
                  </span>
                {/if}
              </div>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>

    {#if activeRow && activeRowSeqStats}
      {@const designedIsCif = activeRow.designedPdb.trimStart().startsWith('data_')}
      {@const activeFailedGates = activeRow.failedGates ?? []}
      {@const activeMsaBadge = formatMsaBadge(activeRow.msaMode, activeRow.msaDepth)}
      {#if !activeRow.designable && activeFailedGates.length > 0}
        <div class="failed-gates-box" role="status">
          <span class="failed-gates-lbl">Failed gates:</span>
          {#each activeFailedGates as gate (gate)}
            <span class="fail-gate-chip">{gate}</span>
          {/each}
        </div>
      {/if}
      <div class="stat-chips">
        <span class="stat-chip cond-key" title="Multiple sequence alignment mode used during refolding">
          MSA: {activeMsaBadge}
        </span>
        {#if contextStructure?.atomGranularity === 'ca-only'}
          <span class="warn-chip" title="Designed reference structure contains Cα coordinates only">
            Cα-only ref
          </span>
        {/if}
        {#if contextStructure?.provenance?.quantization}
          <span class="stat-chip">{contextStructure.provenance.quantization.toUpperCase()}</span>
        {/if}
        {#if contextStructure?.provenance?.degradedFallback}
          <span class="warn-chip">⚠ {contextStructure.provenance.degradedFallback}</span>
        {/if}
      </div>
      <div class="seq-box">
        <div class="seq-head">
          <span>Selected Sequence ({activeRow.sequence.length} aa · {activeRowSeqStats.uniqueAaCount} unique AAs)</span>
          <div class="dl-links">
            <button class="link-btn" onclick={() => copyText(activeRow.sequence, 'metric-seq')}>
              {copyBtnLabel('metric-seq', 'Copy seq')}
            </button>
            {#if activeRow.designedPdb.trim().length > 0}
              <button
                class="link-btn"
                onclick={() =>
                  onDownloadText(
                    activeRow.designedPdb,
                    safeFilename(`${activeRow.structureId}_designed.${designedIsCif ? 'cif' : 'pdb'}`),
                  )}
              >
                {designedIsCif ? '↓ Designed mmCIF' : '↓ Designed PDB'}
              </button>
            {/if}
            <button
              class="link-btn"
              onclick={() =>
                onDownloadText(
                  activeRow.alignedFoldPdb,
                  safeFilename(`${activeRow.sequenceId}_refold_aligned.pdb`),
                )}
            >
              ↓ Refold PDB
            </button>
          </div>
        </div>
        {#if activeRowSeqStats.warnings.length > 0}
          <div class="warn-chips">
            {#each activeRowSeqStats.warnings as w (w)}
              <span class="warn-chip">⚠ {w}</span>
            {/each}
          </div>
        {/if}
        <div class="seq-text">{activeRow.sequence}</div>
      </div>

      <div class="pae-row">
        <PaePlot pae={activeRow.pae} chainLengths={activeRowChainLengths} size={200} />
        <div class="pae-notes">
          <div><b>Fold time:</b> {isFiniteNum(activeRow.foldElapsedMs) ? `${(activeRow.foldElapsedMs / 1000).toFixed(1)} s` : '—'}</div>
          <div><b>MSA:</b> {activeMsaBadge}</div>
          {#if isFiniteNum(activeRow.iptm)}
            <div><b>ipTM:</b> {activeRow.iptm.toFixed(3)}</div>
          {/if}
          {#if isFiniteNum(activeRow.ipae)}
            <div><b>iPAE:</b> {activeRow.ipae.toFixed(2)} Å</div>
          {/if}
          {#if isFiniteNum(activeRow.ipsae)}
            <div>
              <b>ipSAE:</b>
              <span class:good-val={activeRow.ipsae >= effectiveIpsaePass}>
                {activeRow.ipsae.toFixed(2)}
              </span>
              {#if isFiniteNum(activeRow.ipsaeMin)}
                <span class="sub-note">(min {activeRow.ipsaeMin.toFixed(2)})</span>
              {/if}
            </div>
          {/if}
          {#if isFiniteNum(activeRow.binderScRmsd)}
            <div><b>Binder scRMSD:</b> {activeRow.binderScRmsd.toFixed(2)} Å</div>
          {/if}
          {#if isFiniteNum(activeRow.targetAlignedBinderRmsd)}
            <div><b>Target-aligned Binder RMSD:</b> {activeRow.targetAlignedBinderRmsd.toFixed(2)} Å</div>
          {/if}
          {#if isFiniteNum(activeRow.binderPlddt) || isFiniteNum(activeRow.targetPlddt)}
            <div>
              <b>Chain pLDDT:</b>
              {isFiniteNum(activeRow.binderPlddt) ? `Binder ${activeRow.binderPlddt.toFixed(1)}` : ''}
              {isFiniteNum(activeRow.binderPlddt) && isFiniteNum(activeRow.targetPlddt) ? ' · ' : ''}
              {isFiniteNum(activeRow.targetPlddt) ? `Target ${activeRow.targetPlddt.toFixed(1)}` : ''}
            </div>
          {/if}
          {#if isFiniteNum(activeRow.genMotifRmsd)}
            <div><b>Gen Motif RMSD:</b> {activeRow.genMotifRmsd.toFixed(2)} Å</div>
          {/if}
          {#if isFiniteNum(activeRow.refoldMotifRmsd)}
            <div><b>Refold Motif RMSD:</b> {activeRow.refoldMotifRmsd.toFixed(2)} Å</div>
          {/if}
          <div><b>Refold R<sub>g</sub>:</b> {fmtNum(activeRow.geometry?.rg, 1, ' Å')}</div>
          <div><b>Refold α/β:</b> {fmtNum(activeRow.geometry?.helixPct, 0, '%')} / {fmtNum(activeRow.geometry?.sheetPct, 0, '%')}</div>
          <p class="tip">
            Tip: Click any block, port label, or colored wire on the canvas to peek at intermediate structures, sequences, or folds!
          </p>
        </div>
      </div>
    {/if}
  {:else if foldArt && foldArt.folds.length > 0}
    {@const foldColLabel = formatCollectionKind(foldArt.collectionKind)}
    {#if foldColLabel || foldArt.representativeSelection || (foldArt.excludedCandidates && foldArt.excludedCandidates.length > 0)}
      <div class="stat-chips">
        {#if foldColLabel}
          <span class="stat-chip">{foldColLabel}</span>
        {/if}
        {#if foldArt.representativeSelection}
          <span
            class="stat-chip cond-key"
            title={`Selected ${foldArt.representativeSelection.selectedKey} from ${foldArt.representativeSelection.totalMembers} member(s) by ${foldArt.representativeSelection.criterion}`}
          >
            Rep: {foldArt.representativeSelection.selectedKey} · {foldArt.representativeSelection.criterion} ({foldArt.representativeSelection.totalMembers})
          </span>
        {/if}
        {#if foldArt.excludedCandidates && foldArt.excludedCandidates.length > 0}
          <span class="warn-chip">
            {foldArt.excludedCandidates.length} excluded candidate{foldArt.excludedCandidates.length === 1 ? '' : 's'}
          </span>
        {/if}
      </div>
    {/if}

    <table class="scores">
      <thead>
        <tr>
          <th>Seq</th>
          <th>Len</th>
          <th>pLDDT</th>
          <th>pTM</th>
          {#if hasFoldMultiChainMetrics}
            <th title="Interface predicted TM-score (multi-chain)">ipTM</th>
            <th title={`Mean interface PAE (Å, pass < ${ipaeThreshold})`}>iPAE</th>
            <th title={`Dunbrack ipSAE interface score (pass ≥ ${effectiveIpsaePass.toFixed(2)})`}>ipSAE</th>
          {/if}
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        {#each foldArt.folds as f, i (`${f.sequenceId}_${i}`)}
          {@const confAvail = f.confidenceAvailable !== false && isFiniteNum(f.meanPlddt)}
          {@const fIpae = hasFoldMultiChainMetrics ? computeIpaeFromPae(f.pae, f.chainLengths) : null}
          <tr class:selected={i === selectedFoldIdx} onclick={() => onSelectFold(i)}>
            <td><code>{f.sequenceId}</code></td>
            <td>{f.sequence.length} aa</td>
            <td class:good-val={confAvail && f.meanPlddt >= plddtThreshold}>
              {confAvail ? fmtNum(f.meanPlddt, 1) : 'N/A'}
            </td>
            <td>{fmtNum(f.ptm, 2)}</td>
            {#if hasFoldMultiChainMetrics}
              <td>{fmtNum(f.iptm, 2)}</td>
              <td class:good-val={isFiniteNum(fIpae) && fIpae < ipaeThreshold}>
                {fmtNum(fIpae, 1, ' Å')}
              </td>
              <td class:good-val={isFiniteNum(f.ipsae) && f.ipsae >= effectiveIpsaePass}>
                {fmtNum(f.ipsae, 2)}
              </td>
            {/if}
            <td>{isFiniteNum(f.elapsedMs) ? `${(f.elapsedMs / 1000).toFixed(1)}s` : '—'}</td>
          </tr>
        {/each}
      </tbody>
    </table>

    {#if activeFold}
      {@const activeFoldConfAvail = activeFold.confidenceAvailable !== false && isFiniteNum(activeFold.meanPlddt)}
      {@const pStats = activeFoldConfAvail ? plddtSummary(activeFold.plddt) : null}
      {@const activeFoldMsa = formatMsaBadge(
        activeFold.msaMode ?? activeFold.provenance?.msaMode,
        activeFold.msaDepth ?? activeFold.provenance?.msaDepth,
      )}
      {@const activeFoldQuant = formatQuantChip(activeFold.provenance, activeFold.foldModelId)}
      {@const activeFoldIpae = computeIpaeFromPae(activeFold.pae, activeFold.chainLengths)}
      <div class="stat-chips">
        <span class="stat-chip cond-key" title="Multiple sequence alignment mode & depth">
          MSA: {activeFoldMsa}
        </span>
        {#if activeFoldQuant}
          <span class="stat-chip" title="Weight quantization used for inference">{activeFoldQuant}</span>
        {/if}
        {#if activeFold.provenance?.recyclesRun !== undefined}
          <span class="stat-chip">{activeFold.provenance.recyclesRun} recycles</span>
        {/if}
        {#if activeFold.provenance?.degradedFallback}
          <span class="warn-chip">⚠ {activeFold.provenance.degradedFallback}</span>
        {/if}
        {#if activeFold.provenance?.warnings?.length}
          {#each activeFold.provenance.warnings as pw (pw)}
            <span class="warn-chip">⚠ {pw}</span>
          {/each}
        {/if}
      </div>
      <div class="seq-box">
        <div class="seq-head">
          <span>{activeFold.sequenceId} · {formatFoldModelName(activeFold.foldModelId)}</span>
          <div class="dl-links">
            <button class="link-btn" onclick={() => copyText(activeFold.sequence, 'fold-seq')}>
              {copyBtnLabel('fold-seq', 'Copy seq')}
            </button>
            <button
              class="link-btn"
              onclick={() =>
                onDownloadText(
                  activeFold.pdbText,
                  safeFilename(`${activeFold.sequenceId}_${foldModelSlug(activeFold.foldModelId)}.pdb`),
                )}
            >
              ↓ Folded PDB
            </button>
          </div>
        </div>
        <div class="seq-text">{activeFold.sequence}</div>
      </div>

      <div class="pae-row">
        <PaePlot pae={activeFold.pae} chainLengths={activeFold.chainLengths} size={200} />
        <div class="pae-notes">
          <div><b>Mean pLDDT:</b> {activeFoldConfAvail ? fmtNum(activeFold.meanPlddt, 1) : 'N/A'}</div>
          {#if pStats}
            <div><b>pLDDT range:</b> {pStats.min.toFixed(1)} – {pStats.max.toFixed(1)}</div>
            <div><b>Confident (≥70):</b> {pStats.ge70Pct.toFixed(0)}%</div>
          {/if}
          {#if isFiniteNum(activeFold.ptm)}
            <div><b>pTM:</b> {activeFold.ptm.toFixed(3)}</div>
          {/if}
          {#if isFiniteNum(activeFold.iptm)}
            <div>
              <b>ipTM:</b> {activeFold.iptm.toFixed(3)}{activeFold.iptmIsMonomerApprox ? ' (monomer approx)' : ''}
            </div>
          {/if}
          {#if isFiniteNum(activeFoldIpae)}
            <div>
              <b>iPAE:</b>
              <span class:good-val={activeFoldIpae < ipaeThreshold}>{activeFoldIpae.toFixed(2)} Å</span>
            </div>
          {/if}
          {#if isFiniteNum(activeFold.ipsae)}
            <div>
              <b>ipSAE:</b>
              <span class:good-val={activeFold.ipsae >= effectiveIpsaePass}>
                {activeFold.ipsae.toFixed(2)}
              </span>
              {#if isFiniteNum(activeFold.ipsaeMin)}
                <span class="sub-note">(min {activeFold.ipsaeMin.toFixed(2)})</span>
              {/if}
            </div>
          {/if}
          <div><b>MSA:</b> {activeFoldMsa}</div>
          <div><b>Fold time:</b> {isFiniteNum(activeFold.elapsedMs) ? `${(activeFold.elapsedMs / 1000).toFixed(1)} s` : '—'}</div>
        </div>
      </div>
    {/if}
  {:else if seqArt}
    {@const seqColLabel = formatCollectionKind(seqArt.collectionKind)}
    <div class="meta-bar">
      <span>
        <b>{seqArt.candidates.length}</b> sequence{seqArt.candidates.length === 1 ? '' : 's'} · <span>{formatAdapterName(seqArt.sourceModelId)}</span>
        {#if seqColLabel}
          <span class="stat-chip inline-chip">{seqColLabel}</span>
        {/if}
      </span>
      <div class="dl-links">
        <button class="link-btn" onclick={() => copyText(formatFasta(seqArt), 'all-fasta')}>
          {copyBtnLabel('all-fasta', 'Copy FASTA', '✓ Copied FASTA')}
        </button>
        <button
          class="link-btn"
          onclick={() => onDownloadText(formatFasta(seqArt), safeFilename(`${seqArt.id}.fasta`))}
        >
          ↓ FASTA
        </button>
      </div>
    </div>

    <div class="seq-list">
      {#each seqArt.candidates as c, i (`${c.id}_${i}`)}
        {@const st = seqCandidateStats[i]}
        {@const displaySeq = formatCandidateSequence(c.sequence, c.chainLengths)}
        <div class="seq-box">
          <div class="seq-head">
            <span class="cand-title">
              <code>{c.id}</code> · {st.length} aa
              {#if c.chainLengths && c.chainLengths.length > 1}
                · chains [{c.chainLengths.join('+')}]
              {/if}
              {#if isFiniteNum(c.score)}
                · score {c.score.toFixed(2)}
              {/if}
              {#if isFiniteNum(st.recoveryPct)}
                · recovery {st.recoveryPct.toFixed(0)}%
              {/if}
            </span>
            <button class="link-btn" onclick={() => copyText(displaySeq, `cand-${i}`)}>
              {copyBtnLabel(`cand-${i}`, 'Copy')}
            </button>
          </div>
          <div class="stat-chips">
            <span class="stat-chip">{st.uniqueAaCount} unique AAs</span>
            <span class="stat-chip">hydro {fmtNum(st.hydrophobicPct, 0, '%')}</span>
            <span class="stat-chip">polar {fmtNum(st.polarPct, 0, '%')}</span>
            <span class="stat-chip">charged {fmtNum(st.chargedPct, 0, '%')}</span>
            {#if st.topAa}
              <span class="stat-chip">top: {st.topAa.aa} ({fmtNum(st.topAa.pct, 0, '%')})</span>
            {/if}
          </div>
          {#if st.warnings.length > 0}
            <div class="warn-chips">
              {#each st.warnings as w (w)}
                <span class="warn-chip">⚠ {w}</span>
              {/each}
            </div>
          {/if}
          <div class="seq-text">{displaySeq}</div>
        </div>
      {/each}
    </div>
  {:else if structArt && structSeqStats}
    {@const hasValidPdb = Boolean(
      structArt.pdbText &&
        structArt.pdbText.trim().length > 0 &&
        !structArt.pdbText.trimStart().startsWith('data_'),
    )}
    {@const isCifPreview = Boolean(structArt.cifText) || rawStructText.trimStart().startsWith('data_')}
    <div class="meta-bar">
      <span><b>{structArt.label}</b> · <span>{formatAdapterName(structArt.sourceModelId)}</span></span>
      <div class="dl-links">
        {#if hasValidPdb}
          <button class="link-btn" onclick={() => onDownloadText(structArt.pdbText, safeFilename(`${structArt.id}.pdb`))}>
            ↓ PDB
          </button>
        {/if}
        {#if structArt.cifText || (!hasValidPdb && structArt.pdbText.trimStart().startsWith('data_'))}
          <button
            class="link-btn"
            onclick={() => onDownloadText(structArt.cifText ?? structArt.pdbText, safeFilename(`${structArt.id}.cif`))}
          >
            ↓ mmCIF
          </button>
        {/if}
        {#if structArt.trajectoryPdb}
          <button
            class="link-btn"
            onclick={() => onDownloadText(structArt.trajectoryPdb!, safeFilename(`${structArt.id}_traj.pdb`))}
          >
            ↓ Trajectory ({structArt.trajectoryFrameCount ?? '?'}f)
          </button>
        {/if}
      </div>
    </div>

    <div class="stat-chips">
      <span class="stat-chip">L = {structArt.length} aa</span>
      {#if structArt.diagnostics?.chainCount && structArt.diagnostics.chainCount > 1}
        <span class="stat-chip">{structArt.diagnostics.chainCount} chains</span>
      {/if}
      <span class="stat-chip">R<sub>g</sub> = {fmtNum(structArt.geometry?.rg, 1, ' Å')}</span>
      <span class="stat-chip">α {fmtNum(structArt.geometry?.helixPct, 0, '%')} · β {fmtNum(structArt.geometry?.sheetPct, 0, '%')}</span>
      <span class="stat-chip">breaks: {structArt.geometry?.chainBreaks ?? '—'}</span>
      <span class="stat-chip">clashes: {structArt.geometry?.clashes ?? '—'}</span>
      {#if structArt.representativeSelection}
        <span
          class="stat-chip cond-key"
          title={`Selected ${structArt.representativeSelection.selectedKey} from ${structArt.representativeSelection.totalMembers} member(s) by ${structArt.representativeSelection.criterion}`}
        >
          Rep: {structArt.representativeSelection.selectedKey} · {structArt.representativeSelection.criterion} ({structArt.representativeSelection.totalMembers})
        </span>
      {/if}
      {#if structArt.diagnostics?.ligandNames?.length}
        <span class="stat-chip cond-key">Ligands: {structArt.diagnostics.ligandNames.join(', ')}</span>
      {/if}
      {#if structArt.diagnostics?.nucleicAcidChains?.length}
        <span class="stat-chip cond-key">Nucleic acid: {structArt.diagnostics.nucleicAcidChains.join(', ')}</span>
      {/if}
      {#if structArt.trajectoryFrameCount}
        <span class="stat-chip">{structArt.trajectoryFrameCount} traj frames</span>
      {/if}
      {#if structArt.atomGranularity === 'ca-only'}
        <span class="warn-chip" title="Only Cα backbone coordinates were emitted by this model">
          Cα-only
        </span>
      {/if}
      {#if structArt.provenance?.quantization}
        <span class="stat-chip" title="Weight quantization">{structArt.provenance.quantization.toUpperCase()}</span>
      {/if}
      {#if structArt.provenance?.msaMode}
        <span class="stat-chip cond-key">
          MSA: {formatMsaBadge(structArt.provenance.msaMode, structArt.provenance.msaDepth)}
        </span>
      {/if}
      {#if structArt.provenance?.degradedFallback}
        <span class="warn-chip">⚠ {structArt.provenance.degradedFallback}</span>
      {/if}
      {#if structArt.provenance?.warnings?.length}
        {#each structArt.provenance.warnings as pw (pw)}
          <span class="warn-chip">⚠ {pw}</span>
        {/each}
      {/if}
    </div>

    <div class="seq-box">
      <div class="seq-head">
        <span>Backbone residue sequence ({structArt.sequence.length} aa · {structSeqStats.uniqueAaCount} unique AAs)</span>
        <button class="link-btn" onclick={() => copyText(structArt.sequence, 'struct-seq')}>
          {copyBtnLabel('struct-seq', 'Copy seq')}
        </button>
      </div>
      {#if structSeqStats.warnings.length > 0}
        <div class="warn-chips">
          {#each structSeqStats.warnings as w (w)}
            <span class="warn-chip">⚠ {w}</span>
          {/each}
        </div>
      {/if}
      <div class="seq-text">{structArt.sequence}</div>
    </div>

    <details class="raw-details" bind:open={rawPreviewOpen}>
      <summary>
        Raw {isCifPreview ? 'mmCIF' : 'PDB'} preview
        {#if rawPrev}
          ({rawPrev.totalLines} lines)
        {:else}
          ({(rawStructText.length / 1024).toFixed(1)} KB)
        {/if}
      </summary>
      {#if rawPrev}
        <pre class="raw-pre">{rawPrev.preview}{rawPrev.truncated ? `\n… (${rawPrev.totalLines - 36} more lines)` : ''}</pre>
      {/if}
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
        {copyBtnLabel('cond-json', 'Copy spec JSON', '✓ Copied JSON')}
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
        <span class="stat-chip cond-key">{formatConditionKey(k)}</span>
      {/each}
    </div>

    <details class="raw-details">
      <summary>Raw constraint spec JSON</summary>
      <pre class="raw-pre">{condJson}</pre>
    </details>
  {:else}
    <div class="placeholder-box">
      {#if activeSlot && !activeSlot.wired}
        <p class="placeholder-msg">
          {activeSlot.required ? 'Required' : 'Optional'} input <code>{activeSlot.portName}</code> ({PORT_COLORS[activeSlot.portType]?.label ?? activeSlot.portType}) is not connected.
          Drag a matching colored wire into this port on the canvas to pass data in.
        </p>
      {:else if activeSlot}
        <p class="placeholder-msg">
          No data on <b>{activeSlot.direction === 'in' ? 'IN' : 'OUT'} · {activeSlot.portName}</b> yet.
        </p>
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
        <p class="placeholder-msg">
          Wire any blocks on the canvas (or pick a preset above) and click <b>▶ Run</b>.
        </p>
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
  .inline-chip {
    margin-left: 6px;
  }
  .lineage-meta-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    font-size: 11px;
  }
  .ref-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
    color: var(--text-secondary);
  }
  .ref-pill.tone-ok {
    background: var(--ok-bg);
    color: var(--ok-text);
    border-color: var(--ok-border);
  }
  .ref-pill.tone-info {
    background: var(--bg-pill);
    color: var(--accent-soft);
    border-color: var(--border-strong);
  }
  .ref-pill.tone-warn {
    background: var(--warn-bg);
    color: var(--warn-text);
    border-color: var(--warn-border);
  }
  .ref-pill.tone-err {
    background: var(--err-bg);
    color: var(--err-strong);
    border-color: var(--err-border);
  }
  .ref-desc {
    font-size: 11px;
    color: var(--text-muted);
    overflow-wrap: anywhere;
  }
  .lineage-warn-banner {
    font-size: 11px;
    line-height: 1.4;
    padding: 6px 9px;
    border-radius: 6px;
    background: var(--warn-bg);
    color: var(--warn-text);
    border: 1px solid var(--warn-border);
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
  .verdict-cell {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
  }
  .gate-summary {
    font-size: 9.5px;
    color: var(--err-strong);
    max-width: 150px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .failed-gates-box {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    padding: 6px 8px;
    border-radius: 6px;
    background: var(--err-bg);
    border: 1px solid var(--err-border);
    font-size: 10.5px;
  }
  .failed-gates-lbl {
    font-weight: 700;
    color: var(--err-strong);
  }
  .fail-gate-chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--bg-surface) 85%, transparent);
    color: var(--err-strong);
    border: 1px solid var(--err-border);
  }
  .sub-note {
    font-size: 10.5px;
    color: var(--text-dim);
    margin-left: 4px;
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
  .placeholder-msg {
    margin: 0;
    color: var(--text-secondary);
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
