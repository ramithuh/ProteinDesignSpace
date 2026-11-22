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
  import {
    computeCampaignSummary,
    normalizeToCollection,
    type CampaignSummary,
    type CollectionMember,
    type DesignCollection,
    type FoldEvaluationRecord,
    type SequenceCollection,
  } from '$lib/engine/collections';
  import {
    exportCollectionCsv,
    exportCollectionFasta,
    type CollectionCsvRowInput,
  } from '$lib/engine/export-collection';
  import type {
    ArtifactProvenance,
    CollectionKind,
    ConditionSpecArtifact,
    DesignabilityMetrics,
    DesignabilityRow,
    FoldedCandidate,
    FoldedStructure,
    MsaMode,
    MutationMatrixArtifact,
    ReferenceRelation,
    ResidueTrackArtifact,
    SequenceCandidate,
    SequenceEmbeddingArtifact,
    SequenceSet,
    StructureArtifact,
    WorkflowNode,
  } from '$lib/engine/types';
  import MutationHeatmap from '$lib/viewer/MutationHeatmap.svelte';
  import PaePlot from '$lib/viewer/PaePlot.svelte';
  import ResidueTrackPlot from '$lib/viewer/ResidueTrackPlot.svelte';

  interface Props {
    node: WorkflowNode | null;
    slots: PeekSlot[];
    activeSlot: PeekSlot | null;
    contextStructure?: StructureArtifact | null;
    selectedRowIdx: number;
    selectedFoldIdx: number;
    selectedDesignKey?: string;
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
    onSelectDesignKey?: (key: string) => void;
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
    selectedDesignKey = undefined,
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
    onSelectDesignKey = undefined,
    onRunUpToNode = undefined,
    onJumpToScorer = undefined,
    onDownloadText,
  }: Props = $props();

  let copiedKey = $state<string | null>(null);
  let copyFailedKey = $state<string | null>(null);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;
  let rawPreviewOpen = $state(false);
  let localDesignFilter = $state<string>('all');
  let selectedDesignFilter = $derived(selectedDesignKey ?? localDesignFilter);

  function setDesignFilter(nextKey: string) {
    localDesignFilter = nextKey;
    onSelectDesignKey?.(nextKey);
  }

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
      if (provenance.languageModelQuantization) {
        return `${provenance.quantization.toUpperCase()} fold · ${provenance.languageModelQuantization.toUpperCase()} ESM-C`;
      }
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
  let mutArt = $derived(artifact?.kind === 'MutationMatrix' ? (artifact as MutationMatrixArtifact) : null);
  let trackArt = $derived(artifact?.kind === 'ResidueTrack' ? (artifact as ResidueTrackArtifact) : null);
  let embArt = $derived(artifact?.kind === 'SequenceEmbedding' ? (artifact as SequenceEmbeddingArtifact) : null);
  let embStats = $derived(embArt ? embeddingSummary(embArt.pooled) : null);

  $effect(() => {
    void artifact?.id;
    localDesignFilter = 'all';
    rawPreviewOpen = false;
  });

  interface DesignGroupPill {
    key: string;
    index: number;
    label: string;
    count: number;
    passedCount?: number;
  }

  function fmtPct(v: number | null | undefined): string {
    if (!isFiniteNum(v)) return '—';
    return `${Math.round(v * 100)}%`;
  }

  function getDesignKeyFromRow(row: DesignabilityRow): string {
    const ext = row as DesignabilityRow & { designKey?: string };
    return (ext.designKey ?? row.sourceStructureId ?? row.structureId ?? '').trim();
  }

  function getDesignKeyFromFold(fold: FoldedCandidate): string {
    const ext = fold as FoldedCandidate & { designKey?: string };
    return (ext.designKey ?? fold.sourceStructureId ?? '').trim();
  }

  function getDesignKeyFromSeq(cand: SequenceCandidate): string {
    const ext = cand as SequenceCandidate & { designKey?: string };
    return (ext.designKey ?? cand.sourceStructureId ?? '').trim();
  }

  function getDesignKeyFromStruct(struct: StructureArtifact, idx: number): string {
    const ext = struct as StructureArtifact & { designKey?: string };
    return (ext.designKey ?? struct.id ?? `design-${idx + 1}`).trim();
  }

  function extractCollectionStructures(struct: StructureArtifact | null): StructureArtifact[] {
    if (!struct) return [];
    const ext = struct as StructureArtifact & {
      collection?: { items?: unknown[]; members?: unknown[] };
      structureCollection?: { items?: unknown[]; members?: unknown[] };
    };
    const rawList =
      ext.collection?.items ??
      ext.collection?.members ??
      ext.structureCollection?.items ??
      ext.structureCollection?.members;
    if (!Array.isArray(rawList) || rawList.length === 0) return [];
    const out: StructureArtifact[] = [];
    for (const entry of rawList) {
      if (!entry || typeof entry !== 'object') continue;
      const obj = entry as Record<string, unknown>;
      if (obj.kind === 'StructureArtifact') {
        out.push(entry as StructureArtifact);
      } else if (obj.payload && typeof obj.payload === 'object' && (obj.payload as { kind?: unknown }).kind === 'StructureArtifact') {
        out.push(obj.payload as StructureArtifact);
      }
    }
    return out;
  }

  function inferCanonicalDesignIndex(designKey: string, explicitIndex?: number): number | null {
    if (typeof explicitIndex === 'number' && Number.isInteger(explicitIndex) && explicitIndex >= 0) {
      return explicitIndex;
    }
    const trimmed = designKey.trim();
    // 1-based suffixes: `design-3`, `d3`, `..._d3`, `...:d3`
    const oneBased = trimmed.match(/(?:^|[:_/-])d(?:esign)?[-_]?(\d+)(?:$|[/-])/i);
    if (oneBased) {
      const n = Number(oneBased[1]);
      if (Number.isInteger(n) && n >= 1) return n - 1;
    }
    // 0-based canonical member key suffix: `genie3:abc123:2`
    const zeroBased = trimmed.match(/:(\d+)$/);
    if (zeroBased) {
      const n = Number(zeroBased[1]);
      if (Number.isInteger(n) && n >= 0) return n;
    }
    return null;
  }

  function buildDesignPills(
    entries: readonly { designKey: string; designIndex?: number; passed?: boolean }[],
  ): DesignGroupPill[] {
    const order: string[] = [];
    const counts = new Map<
      string,
      { count: number; passed: number; hasPassTrack: boolean; firstSeen: number; canonicalIdx: number | null }
    >();
    for (const entry of entries) {
      const k = entry.designKey.trim();
      if (!k) continue;
      let rec = counts.get(k);
      const inferred = inferCanonicalDesignIndex(k, entry.designIndex);
      if (!rec) {
        rec = {
          count: 0,
          passed: 0,
          hasPassTrack: entry.passed !== undefined,
          firstSeen: order.length,
          canonicalIdx: inferred,
        };
        counts.set(k, rec);
        order.push(k);
      } else if (rec.canonicalIdx === null && inferred !== null) {
        rec.canonicalIdx = inferred;
      }
      rec.count++;
      if (entry.passed === true) rec.passed++;
    }
    const sortedKeys = [...order].sort((a, b) => {
      const ra = counts.get(a)!;
      const rb = counts.get(b)!;
      if (ra.canonicalIdx !== null && rb.canonicalIdx !== null && ra.canonicalIdx !== rb.canonicalIdx) {
        return ra.canonicalIdx - rb.canonicalIdx;
      }
      if (ra.canonicalIdx !== null && rb.canonicalIdx === null) return -1;
      if (ra.canonicalIdx === null && rb.canonicalIdx !== null) return 1;
      return ra.firstSeen - rb.firstSeen;
    });
    return sortedKeys.map((key, idx) => {
      const rec = counts.get(key)!;
      const displayIdx = rec.canonicalIdx ?? idx;
      return {
        key,
        index: displayIdx,
        label: `Design ${displayIdx + 1}`,
        count: rec.count,
        ...(rec.hasPassTrack ? { passedCount: rec.passed } : {}),
      };
    });
  }

  let metricsDesignPills = $derived.by((): DesignGroupPill[] => {
    if (!metricsArt || metricsArt.rows.length === 0) return [];
    return buildDesignPills(
      metricsArt.rows.map((r) => ({
        designKey: getDesignKeyFromRow(r),
        designIndex: r.designIndex,
        passed: r.designable,
      })),
    );
  });

  let metricsMultiDesign = $derived(metricsDesignPills.length > 1);

  let metricsCampaignSummary = $derived.by((): CampaignSummary | null => {
    if (!metricsArt || !metricsMultiDesign) return null;
    const designKeys = metricsDesignPills.map((p) => p.key);
    const records: FoldEvaluationRecord[] = metricsArt.rows.map((row, idx) => {
      const ext = row as DesignabilityRow & { designKey?: string; sequenceKey?: string; foldKey?: string };
      const designKey = getDesignKeyFromRow(row) || `design-${idx + 1}`;
      const sequenceKey = (ext.sequenceKey ?? row.candidateKey ?? row.sequenceId ?? `seq-${idx + 1}`).trim();
      const foldKey = (ext.foldKey ?? `${sequenceKey}:fold:${idx}`).trim();
      const confAvail = row.confidenceAvailable !== false && isFiniteNum(row.meanPlddt);
      return {
        foldKey,
        designKey,
        sequenceKey,
        status: 'completed',
        designable: row.designable,
        confidenceAvailable: confAvail,
        evaluable: !row.lineageMismatch,
      };
    });
    return computeCampaignSummary(records, {
      designKeys,
      requestedDesigns: designKeys.length,
    });
  });

  let filteredMetricsRows = $derived.by(() => {
    if (!metricsArt) return [] as Array<{ row: DesignabilityRow; origIdx: number; designLabel: string }>;
    const labelByKey = new Map(metricsDesignPills.map((p) => [p.key, p.label]));
    return metricsArt.rows
      .map((row, origIdx) => ({
        row,
        origIdx,
        designKey: getDesignKeyFromRow(row),
        designLabel: labelByKey.get(getDesignKeyFromRow(row)) ?? '',
      }))
      .filter((entry) => !metricsMultiDesign || selectedDesignFilter === 'all' || entry.designKey === selectedDesignFilter);
  });

  let foldDesignPills = $derived.by((): DesignGroupPill[] => {
    if (!foldArt || foldArt.folds.length === 0) return [];
    return buildDesignPills(
      foldArt.folds.map((f) => ({
        designKey: getDesignKeyFromFold(f),
        designIndex: f.designIndex,
      })),
    );
  });

  let foldMultiDesign = $derived(foldDesignPills.length > 1);

  let foldCampaignSummary = $derived.by((): CampaignSummary | null => {
    if (!foldArt || !foldMultiDesign) return null;
    const designKeys = foldDesignPills.map((p) => p.key);
    const records: FoldEvaluationRecord[] = foldArt.folds.map((f, idx) => {
      const ext = f as FoldedCandidate & { designKey?: string; sequenceKey?: string; foldKey?: string };
      const designKey = getDesignKeyFromFold(f) || `design-${idx + 1}`;
      const sequenceKey = (ext.sequenceKey ?? f.key ?? f.sequenceId ?? `seq-${idx + 1}`).trim();
      const foldKey = (ext.foldKey ?? `${sequenceKey}:fold:${idx}`).trim();
      const confAvail = f.confidenceAvailable !== false && isFiniteNum(f.meanPlddt);
      return {
        foldKey,
        designKey,
        sequenceKey,
        status: 'completed',
        designable: null,
        confidenceAvailable: confAvail,
      };
    });
    for (const excl of foldArt.excludedCandidates ?? []) {
      const extExcl = excl as typeof excl & { designKey?: string; sourceStructureId?: string };
      const dk = (extExcl.designKey ?? extExcl.sourceStructureId ?? designKeys[0] ?? 'design-1').trim();
      records.push({
        foldKey: `excl:${excl.key || excl.id}`,
        designKey: dk,
        sequenceKey: excl.key || excl.id,
        status: 'excluded',
        reason: excl.reason,
      });
    }
    return computeCampaignSummary(records, {
      designKeys,
      requestedDesigns: designKeys.length,
    });
  });

  let filteredFolds = $derived.by(() => {
    if (!foldArt) return [] as Array<{ fold: FoldedCandidate; origIdx: number; designLabel: string }>;
    const labelByKey = new Map(foldDesignPills.map((p) => [p.key, p.label]));
    return foldArt.folds
      .map((fold, origIdx) => ({
        fold,
        origIdx,
        designKey: getDesignKeyFromFold(fold),
        designLabel: labelByKey.get(getDesignKeyFromFold(fold)) ?? '',
      }))
      .filter((entry) => !foldMultiDesign || selectedDesignFilter === 'all' || entry.designKey === selectedDesignFilter);
  });

  let seqDesignPills = $derived.by((): DesignGroupPill[] => {
    if (!seqArt || seqArt.candidates.length === 0) return [];
    return buildDesignPills(
      seqArt.candidates.map((c) => ({
        designKey: getDesignKeyFromSeq(c),
        designIndex: c.designIndex,
      })),
    );
  });

  let seqMultiDesign = $derived(seqDesignPills.length > 1);

  let seqCampaignSummary = $derived.by((): CampaignSummary | null => {
    if (!seqArt || !seqMultiDesign) return null;
    const designKeys = seqDesignPills.map((p) => p.key);
    const records: FoldEvaluationRecord[] = seqArt.candidates.map((c, idx) => {
      const ext = c as SequenceCandidate & { designKey?: string; sequenceKey?: string };
      const designKey = getDesignKeyFromSeq(c) || `design-${idx + 1}`;
      const sequenceKey = (ext.sequenceKey ?? c.key ?? c.id ?? `seq-${idx + 1}`).trim();
      return {
        foldKey: sequenceKey,
        designKey,
        sequenceKey,
        status: 'completed',
        designable: null,
      };
    });
    return computeCampaignSummary(records, {
      designKeys,
      requestedDesigns: designKeys.length,
    });
  });

  let filteredSeqEntries = $derived.by(() => {
    if (!seqArt) return [] as Array<{ cand: SequenceCandidate; origIdx: number; designLabel: string }>;
    const labelByKey = new Map(seqDesignPills.map((p) => [p.key, p.label]));
    return seqArt.candidates
      .map((cand, origIdx) => ({
        cand,
        origIdx,
        designKey: getDesignKeyFromSeq(cand),
        designLabel: labelByKey.get(getDesignKeyFromSeq(cand)) ?? '',
      }))
      .filter((entry) => !seqMultiDesign || selectedDesignFilter === 'all' || entry.designKey === selectedDesignFilter);
  });

  let structCollectionItems = $derived(extractCollectionStructures(structArt));
  let structMultiDesign = $derived(structCollectionItems.length > 1);
  let structDesignPills = $derived.by((): DesignGroupPill[] => {
    if (!structMultiDesign) return [];
    return structCollectionItems.map((item, idx) => ({
      key: getDesignKeyFromStruct(item, idx),
      index: idx,
      label: `Design ${idx + 1}`,
      count: 1,
    }));
  });
  let structCampaignSummary = $derived.by((): CampaignSummary | null => {
    if (!structMultiDesign) return null;
    const designKeys = structDesignPills.map((p) => p.key);
    const records: FoldEvaluationRecord[] = structCollectionItems.map((item, idx) => {
      const dk = getDesignKeyFromStruct(item, idx);
      return {
        foldKey: dk,
        designKey: dk,
        sequenceKey: dk,
        status: 'completed',
        designable: null,
      };
    });
    return computeCampaignSummary(records, {
      designKeys,
      requestedDesigns: designKeys.length,
    });
  });
  let displayedStruct = $derived.by((): StructureArtifact | null => {
    if (!structArt) return null;
    if (!structMultiDesign || selectedDesignFilter === 'all') return structArt;
    const idx = structCollectionItems.findIndex(
      (item, i) => getDesignKeyFromStruct(item, i) === selectedDesignFilter,
    );
    return idx >= 0 ? structCollectionItems[idx] : structArt;
  });

  let activeRow = $derived.by(() => {
    if (!metricsArt) return null;
    const exact = filteredMetricsRows.find((r) => r.origIdx === selectedRowIdx);
    if (exact) return exact.row;
    if (filteredMetricsRows.length > 0) return filteredMetricsRows[0].row;
    return metricsArt.rows[selectedRowIdx] ?? metricsArt.best ?? null;
  });
  let activeFold = $derived.by(() => {
    if (!foldArt) return null;
    const exact = filteredFolds.find((f) => f.origIdx === selectedFoldIdx);
    if (exact) return exact.fold;
    if (filteredFolds.length > 0) return filteredFolds[0].fold;
    return foldArt.folds[selectedFoldIdx] ?? foldArt.folds[0] ?? null;
  });
  let activeRowSeqStats = $derived(activeRow ? analyzeSequence(activeRow.sequence) : null);
  let structSeqStats = $derived(displayedStruct ? analyzeSequence(displayedStruct.sequence) : null);
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
  let rawStructText = $derived(displayedStruct ? (displayedStruct.cifText ?? displayedStruct.pdbText) : '');
  let rawPrev = $derived(
    rawPreviewOpen && displayedStruct ? previewTextLines(rawStructText, 36) : null,
  );

  function downloadMetricsCampaignFasta(m: DesignabilityMetrics) {
    const members: CollectionMember<SequenceCandidate>[] = m.rows.map((row, idx) => {
      const ext = row as DesignabilityRow & { designKey?: string; sequenceKey?: string; foldKey?: string };
      const designKey = getDesignKeyFromRow(row) || undefined;
      const seqKey = (ext.sequenceKey ?? row.candidateKey ?? row.sequenceId ?? `seq-${idx + 1}`).trim();
      const foldKey = ext.foldKey?.trim();
      return {
        key: seqKey,
        ordinal: idx,
        memberKind: 'SequenceCandidate',
        lineage: {
          ...(designKey ? { designKey } : {}),
          sequenceKey: seqKey,
          ...(foldKey ? { foldKey } : {}),
          parentKeys: designKey ? [designKey] : [],
        },
        payload: {
          id: row.sequenceId,
          key: seqKey,
          sequence: row.sequence,
          score: null,
          globalScore: null,
          sourceModelId: 'scorer',
          ...(row.sourceStructureId ? { sourceStructureId: row.sourceStructureId } : {}),
        },
        summary: {
          label: row.sequenceId,
          residueCount: row.sequence.length,
          confidence: row.confidenceAvailable === false ? null : row.meanPlddt,
          confidenceAvailable: row.confidenceAvailable !== false,
          designable: row.designable,
          scRmsd: row.scRmsd,
        },
      };
    });
    const col: SequenceCollection = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: m.id,
      familyKey: m.id,
      revision: 1,
      memberKind: 'SequenceCandidate',
      collectionKind: m.collectionKind ?? 'independent-candidates',
      label: m.id,
      sourceModelId: 'scorer',
      members,
      outcomes: [],
      requestedCount: members.length,
      complete: true,
    };
    onDownloadText(exportCollectionFasta(col), safeFilename(`${m.id}_campaign.fasta`));
  }

  function downloadMetricsCampaignCsv(m: DesignabilityMetrics) {
    const col = normalizeToCollection(m);
    const csv = exportCollectionCsv(col, {
      snapshotId: m.id,
      thresholds: {
        scRmsd: scRmsdThreshold,
        plddt: plddtThreshold,
        ...(ipsaeThreshold > 0 ? { ipsae: ipsaeThreshold } : {}),
      },
      ...(metricsCampaignSummary ? { aggregationPolicy: metricsCampaignSummary.aggregationPolicy } : {}),
    });
    onDownloadText(csv, safeFilename(`${m.id}_campaign.csv`));
  }

  function downloadFoldCampaignFasta(f: FoldedStructure) {
    const col = normalizeToCollection(f);
    onDownloadText(exportCollectionFasta(col), safeFilename(`${f.id}_campaign.fasta`));
  }

  function downloadFoldCampaignCsv(f: FoldedStructure) {
    const col = normalizeToCollection(f);
    const csv = exportCollectionCsv(col, {
      snapshotId: f.id,
      thresholds: {
        scRmsd: scRmsdThreshold,
        plddt: plddtThreshold,
        ...(ipsaeThreshold > 0 ? { ipsae: ipsaeThreshold } : {}),
      },
    });
    onDownloadText(csv, safeFilename(`${f.id}_campaign.csv`));
  }

  function downloadSeqCampaignFasta(s: SequenceSet) {
    const col = normalizeToCollection(s);
    onDownloadText(exportCollectionFasta(col), safeFilename(`${s.id}_campaign.fasta`));
  }

  function downloadSeqCampaignCsv(s: SequenceSet) {
    const rows: CollectionCsvRowInput[] = s.candidates.map((c, idx) => {
      const ext = c as SequenceCandidate & { designKey?: string; sequenceKey?: string };
      return {
        designKey: getDesignKeyFromSeq(c) || undefined,
        sequenceKey: (ext.sequenceKey ?? c.key ?? c.id ?? `seq-${idx + 1}`).trim(),
        referenceStructureId: c.sourceStructureId ?? s.sourceStructureId ?? s.parentStructureId,
        status: 'completed',
        modelId: c.sourceModelId || s.sourceModelId,
        seed: c.seed,
        length: c.sequence.length,
        mpnnScore: c.score,
      };
    });
    onDownloadText(exportCollectionCsv(rows, { snapshotId: s.id }), safeFilename(`${s.id}_campaign.csv`));
  }

  function downloadStructCampaignFasta(items: StructureArtifact[], baseId: string) {
    const members: CollectionMember<StructureArtifact>[] = items.map((item, idx) => {
      const dk = getDesignKeyFromStruct(item, idx);
      return {
        key: dk,
        ordinal: idx,
        memberKind: 'StructureArtifact',
        lineage: {
          designKey: dk,
          designIndex: idx,
          parentKeys: [],
        },
        payload: item,
        summary: {
          label: item.label || item.id,
          residueCount: item.length,
        },
      };
    });
    const col: DesignCollection<StructureArtifact> = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: baseId,
      familyKey: baseId,
      revision: 1,
      memberKind: 'StructureArtifact',
      collectionKind: 'independent-candidates',
      label: baseId,
      sourceModelId: items[0]?.sourceModelId ?? 'generator',
      members,
      outcomes: [],
      requestedCount: members.length,
      complete: true,
    };
    onDownloadText(exportCollectionFasta(col), safeFilename(`${baseId}_campaign.fasta`));
  }

  function downloadStructCampaignCsv(items: StructureArtifact[], baseId: string) {
    const rows: CollectionCsvRowInput[] = items.map((item, idx) => ({
      designKey: getDesignKeyFromStruct(item, idx),
      referenceStructureId: item.id,
      status: 'completed',
      modelId: item.sourceModelId,
      length: item.length,
    }));
    onDownloadText(exportCollectionCsv(rows, { snapshotId: baseId }), safeFilename(`${baseId}_campaign.csv`));
  }
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

  function embeddingSummary(arr: Float32Array) {
    if (!arr || arr.length === 0) return null;
    let sumSq = 0;
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    let count = 0;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (!Number.isFinite(v)) continue;
      count++;
      sumSq += v * v;
      sum += v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (count === 0) return null;
    const mean = sum / count;
    const l2Norm = Math.sqrt(sumSq);
    const preview: number[] = [];
    for (let i = 0; i < Math.min(8, arr.length); i++) {
      preview.push(arr[i]);
    }
    return { count, l2Norm, mean, min, max, preview };
  }

  function formatEmbeddingJson(emb: SequenceEmbeddingArtifact): string {
    const stats = embeddingSummary(emb.pooled);
    const perResNorms =
      emb.perResidue && emb.dim > 0
        ? (() => {
            const L = Math.floor(emb.perResidue.length / emb.dim);
            const norms: number[] = new Array(L);
            for (let i = 0; i < L; i++) {
              let sq = 0;
              const off = i * emb.dim;
              for (let d = 0; d < emb.dim; d++) {
                const v = emb.perResidue[off + d];
                if (Number.isFinite(v)) sq += v * v;
              }
              norms[i] = Number(Math.sqrt(sq).toFixed(6));
            }
            return norms;
          })()
        : undefined;
    return (
      JSON.stringify(
        {
          id: emb.id,
          ...(emb.label ? { label: emb.label } : {}),
          sourceModelId: emb.sourceModelId,
          ...(emb.parentStructureId ? { parentStructureId: emb.parentStructureId } : {}),
          ...(emb.sequenceId ? { sequenceId: emb.sequenceId } : {}),
          sequence: emb.sequence,
          length: emb.sequence.length,
          dim: emb.dim,
          ...(isFiniteNum(emb.pseudoPerplexity) ? { pseudoPerplexity: emb.pseudoPerplexity } : {}),
          ...(stats
            ? {
                pooledNorm: Number(stats.l2Norm.toFixed(6)),
                pooledMean: Number(stats.mean.toFixed(6)),
                pooledRange: [Number(stats.min.toFixed(6)), Number(stats.max.toFixed(6))],
              }
            : {}),
          pooled: Array.from(emb.pooled, (v) => (Number.isFinite(v) ? Number(v.toFixed(6)) : null)),
          ...(perResNorms
            ? {
                perResidueShape: [perResNorms.length, emb.dim],
                perResidueL2Norms: perResNorms,
              }
            : {}),
          ...(emb.provenance ? { provenance: emb.provenance } : {}),
        },
        null,
        2,
      ) + '\n'
    );
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

    {#if metricsMultiDesign && metricsCampaignSummary}
      <div class="campaign-bar">
        <div class="campaign-top">
          <div class="campaign-stats">
            <span class="campaign-badge">{metricsDesignPills.length} backbones</span>
            <span class="campaign-metric" title="Designs with at least one passing sequence / evaluated designs">
              Design hit rate: <b>{fmtPct(metricsCampaignSummary.designHitRate)}</b>
              <span class="campaign-sub">({metricsCampaignSummary.passingDesigns}/{metricsCampaignSummary.evaluatedDesigns})</span>
            </span>
            <span class="campaign-metric" title="Passing folds / evaluable folds">
              Fold pass rate: <b>{fmtPct(metricsCampaignSummary.foldPassRate)}</b>
              <span class="campaign-sub">({metricsCampaignSummary.passedFolds}/{metricsCampaignSummary.evaluableFolds})</span>
            </span>
            <span class="campaign-metric" title="Completed & evaluable folds vs planned folds">
              Coverage: <b>{fmtPct(metricsCampaignSummary.coverage.evaluableFraction)}</b>
              <span class="campaign-sub">({metricsCampaignSummary.coverage.evaluableFolds}/{metricsCampaignSummary.coverage.plannedFolds})</span>
            </span>
          </div>
          <div class="campaign-exports">
            <button
              type="button"
              class="link-btn"
              onclick={() => downloadMetricsCampaignFasta(metricsArt)}
              title="Export campaign sequences as FASTA with stable design & sequence lineage headers"
            >
              ↓ Campaign FASTA
            </button>
            <button
              type="button"
              class="link-btn"
              onclick={() => downloadMetricsCampaignCsv(metricsArt)}
              title="Export campaign scores, lineage keys, and thresholds as CSV"
            >
              ↓ Campaign CSV
            </button>
          </div>
        </div>
        <div class="design-pills" role="group" aria-label="Filter by design backbone">
          <button
            type="button"
            class="design-pill"
            class:active={selectedDesignFilter === 'all'}
            aria-pressed={selectedDesignFilter === 'all'}
            onclick={() => setDesignFilter('all')}
          >
            All designs ({metricsDesignPills.length})
          </button>
          {#each metricsDesignPills as pill (pill.key)}
            <button
              type="button"
              class="design-pill"
              class:active={selectedDesignFilter === pill.key}
              aria-pressed={selectedDesignFilter === pill.key}
              title={`Filter to ${pill.label} (${pill.key})`}
              onclick={() => setDesignFilter(pill.key)}
            >
              {pill.label}
              <span class="pill-count">{pill.passedCount ?? 0}/{pill.count}</span>
            </button>
          {/each}
        </div>
      </div>
    {/if}

    <table class="scores">
      <thead>
        <tr>
          {#if metricsMultiDesign}
            <th>Design</th>
          {/if}
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
        {#each filteredMetricsRows as entry (`${entry.row.sequenceId}_${entry.origIdx}`)}
          {@const row = entry.row}
          {@const i = entry.origIdx}
          {@const bRmsd = isFiniteNum(row.targetAlignedBinderRmsd) ? row.targetAlignedBinderRmsd : row.binderScRmsd}
          {@const rowConfAvail = row.confidenceAvailable !== false && isFiniteNum(row.meanPlddt)}
          {@const rowFailedGates = row.failedGates ?? []}
          <tr class:selected={i === selectedRowIdx} onclick={() => onSelectRow(i)}>
            {#if metricsMultiDesign}
              <td><span class="design-tag">{entry.designLabel || '—'}</span></td>
            {/if}
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

    {#if foldMultiDesign && foldCampaignSummary}
      <div class="campaign-bar">
        <div class="campaign-top">
          <div class="campaign-stats">
            <span class="campaign-badge">{foldDesignPills.length} backbones</span>
            <span class="campaign-metric" title="Completed folds vs planned folds">
              Coverage: <b>{fmtPct(foldCampaignSummary.coverage.completedFraction)}</b>
              <span class="campaign-sub">({foldCampaignSummary.completedFolds}/{foldCampaignSummary.plannedFolds})</span>
            </span>
            <span class="campaign-metric" title="Connect Designability Scorer to compute hit & pass rates">
              Design hit rate: <b>{fmtPct(foldCampaignSummary.designHitRate)}</b>
            </span>
            <span class="campaign-metric" title="Connect Designability Scorer to compute fold pass rate">
              Fold pass rate: <b>{fmtPct(foldCampaignSummary.foldPassRate)}</b>
            </span>
          </div>
          <div class="campaign-exports">
            <button
              type="button"
              class="link-btn"
              onclick={() => downloadFoldCampaignFasta(foldArt)}
              title="Export folded campaign sequences as FASTA with lineage keys"
            >
              ↓ Campaign FASTA
            </button>
            <button
              type="button"
              class="link-btn"
              onclick={() => downloadFoldCampaignCsv(foldArt)}
              title="Export campaign fold metrics & exclusions as CSV"
            >
              ↓ Campaign CSV
            </button>
          </div>
        </div>
        <div class="design-pills" role="group" aria-label="Filter folds by design backbone">
          <button
            type="button"
            class="design-pill"
            class:active={selectedDesignFilter === 'all'}
            aria-pressed={selectedDesignFilter === 'all'}
            onclick={() => setDesignFilter('all')}
          >
            All designs ({foldDesignPills.length})
          </button>
          {#each foldDesignPills as pill (pill.key)}
            <button
              type="button"
              class="design-pill"
              class:active={selectedDesignFilter === pill.key}
              aria-pressed={selectedDesignFilter === pill.key}
              title={`Filter folds to ${pill.label} (${pill.key})`}
              onclick={() => setDesignFilter(pill.key)}
            >
              {pill.label}
              <span class="pill-count">{pill.count}</span>
            </button>
          {/each}
        </div>
      </div>
    {/if}

    <table class="scores">
      <thead>
        <tr>
          {#if foldMultiDesign}
            <th>Design</th>
          {/if}
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
        {#each filteredFolds as entry (`${entry.fold.sequenceId}_${entry.origIdx}`)}
          {@const f = entry.fold}
          {@const i = entry.origIdx}
          {@const confAvail = f.confidenceAvailable !== false && isFiniteNum(f.meanPlddt)}
          {@const fIpae = hasFoldMultiChainMetrics ? computeIpaeFromPae(f.pae, f.chainLengths) : null}
          <tr class:selected={i === selectedFoldIdx} onclick={() => onSelectFold(i)}>
            {#if foldMultiDesign}
              <td><span class="design-tag">{entry.designLabel || '—'}</span></td>
            {/if}
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
        {#if activeFold.provenance?.diffusionStepsRun !== undefined}
          <span class="stat-chip" title={`${activeFold.provenance.diffusionStepsScheduled} steps in the diffusion schedule`}>{activeFold.provenance.diffusionStepsRun} diffusion updates</span>
        {/if}
        {#if activeFold.provenance?.confidenceHead}
          <span class="stat-chip">{activeFold.provenance.confidenceHead} confidence</span>
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

    {#if seqMultiDesign && seqCampaignSummary}
      <div class="campaign-bar">
        <div class="campaign-top">
          <div class="campaign-stats">
            <span class="campaign-badge">{seqDesignPills.length} backbones</span>
            <span class="campaign-metric">
              Sequences: <b>{seqArt.candidates.length}</b>
            </span>
            <span class="campaign-metric">
              Coverage: <b>{fmtPct(seqCampaignSummary.coverage.completedFraction)}</b>
              <span class="campaign-sub">({seqCampaignSummary.completedFolds}/{seqCampaignSummary.plannedFolds})</span>
            </span>
          </div>
          <div class="campaign-exports">
            <button
              type="button"
              class="link-btn"
              onclick={() => downloadSeqCampaignFasta(seqArt)}
              title="Export all campaign sequences as FASTA with design & sequence lineage keys"
            >
              ↓ Campaign FASTA
            </button>
            <button
              type="button"
              class="link-btn"
              onclick={() => downloadSeqCampaignCsv(seqArt)}
              title="Export campaign sequences & scores as CSV"
            >
              ↓ Campaign CSV
            </button>
          </div>
        </div>
        <div class="design-pills" role="group" aria-label="Filter sequences by design backbone">
          <button
            type="button"
            class="design-pill"
            class:active={selectedDesignFilter === 'all'}
            aria-pressed={selectedDesignFilter === 'all'}
            onclick={() => setDesignFilter('all')}
          >
            All designs ({seqDesignPills.length})
          </button>
          {#each seqDesignPills as pill (pill.key)}
            <button
              type="button"
              class="design-pill"
              class:active={selectedDesignFilter === pill.key}
              aria-pressed={selectedDesignFilter === pill.key}
              title={`Filter sequences to ${pill.label} (${pill.key})`}
              onclick={() => setDesignFilter(pill.key)}
            >
              {pill.label}
              <span class="pill-count">{pill.count}</span>
            </button>
          {/each}
        </div>
      </div>
    {/if}

    <div class="seq-list">
      {#each filteredSeqEntries as entry (`${entry.cand.id}_${entry.origIdx}`)}
        {@const c = entry.cand}
        {@const i = entry.origIdx}
        {@const st = seqCandidateStats[i]}
        {@const displaySeq = formatCandidateSequence(c.sequence, c.chainLengths)}
        <div class="seq-box">
          <div class="seq-head">
            <span class="cand-title">
              {#if seqMultiDesign && entry.designLabel}
                <span class="design-tag">{entry.designLabel}</span>
              {/if}
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
  {:else if displayedStruct && structSeqStats}
    {@const activeStruct = displayedStruct}
    {@const hasValidPdb = Boolean(
      activeStruct.pdbText &&
        activeStruct.pdbText.trim().length > 0 &&
        !activeStruct.pdbText.trimStart().startsWith('data_'),
    )}
    {@const isCifPreview = Boolean(activeStruct.cifText) || rawStructText.trimStart().startsWith('data_')}
    {#if structMultiDesign && structCampaignSummary}
      <div class="campaign-bar">
        <div class="campaign-top">
          <div class="campaign-stats">
            <span class="campaign-badge">{structCollectionItems.length} backbones</span>
            <span class="campaign-metric">
              Coverage: <b>{fmtPct(structCampaignSummary.coverage.completedFraction)}</b>
              <span class="campaign-sub">({structCampaignSummary.completedFolds}/{structCampaignSummary.plannedFolds})</span>
            </span>
            <span class="campaign-metric">
              Design hit rate: <b>{fmtPct(structCampaignSummary.designHitRate)}</b>
            </span>
            <span class="campaign-metric">
              Fold pass rate: <b>{fmtPct(structCampaignSummary.foldPassRate)}</b>
            </span>
          </div>
          <div class="campaign-exports">
            <button
              type="button"
              class="link-btn"
              onclick={() => downloadStructCampaignFasta(structCollectionItems, structArt?.id ?? activeStruct.id)}
              title="Export all backbone sequences as FASTA with stable design keys"
            >
              ↓ Campaign FASTA
            </button>
            <button
              type="button"
              class="link-btn"
              onclick={() => downloadStructCampaignCsv(structCollectionItems, structArt?.id ?? activeStruct.id)}
              title="Export backbone collection summary as CSV"
            >
              ↓ Campaign CSV
            </button>
          </div>
        </div>
        <div class="design-pills" role="group" aria-label="Select generated backbone design">
          <button
            type="button"
            class="design-pill"
            class:active={selectedDesignFilter === 'all'}
            aria-pressed={selectedDesignFilter === 'all'}
            onclick={() => setDesignFilter('all')}
          >
            All designs ({structDesignPills.length})
          </button>
          {#each structDesignPills as pill (pill.key)}
            <button
              type="button"
              class="design-pill"
              class:active={selectedDesignFilter === pill.key}
              aria-pressed={selectedDesignFilter === pill.key}
              title={`Inspect ${pill.label} (${pill.key})`}
              onclick={() => setDesignFilter(pill.key)}
            >
              {pill.label}
            </button>
          {/each}
        </div>
      </div>
      {#if selectedDesignFilter === 'all'}
        <table class="scores">
          <thead>
            <tr>
              <th>Design</th>
              <th>ID</th>
              <th>Len</th>
              <th>R<sub>g</sub></th>
              <th>Breaks / Clashes</th>
            </tr>
          </thead>
          <tbody>
            {#each structCollectionItems as item, idx (`${item.id}_${idx}`)}
              {@const dk = getDesignKeyFromStruct(item, idx)}
              <tr onclick={() => setDesignFilter(dk)}>
                <td><span class="design-tag">Design {idx + 1}</span></td>
                <td><code>{item.id}</code></td>
                <td>{item.length} aa</td>
                <td>{fmtNum(item.geometry?.rg, 1, ' Å')}</td>
                <td>{item.geometry?.chainBreaks ?? '—'} / {item.geometry?.clashes ?? '—'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    {/if}
    <div class="meta-bar">
      <span><b>{activeStruct.label}</b> · <span>{formatAdapterName(activeStruct.sourceModelId)}</span></span>
      <div class="dl-links">
        {#if hasValidPdb}
          <button class="link-btn" onclick={() => onDownloadText(activeStruct.pdbText, safeFilename(`${activeStruct.id}.pdb`))}>
            ↓ PDB
          </button>
        {/if}
        {#if activeStruct.cifText || (!hasValidPdb && activeStruct.pdbText.trimStart().startsWith('data_'))}
          <button
            class="link-btn"
            onclick={() => onDownloadText(activeStruct.cifText ?? activeStruct.pdbText, safeFilename(`${activeStruct.id}.cif`))}
          >
            ↓ mmCIF
          </button>
        {/if}
        {#if activeStruct.trajectoryPdb}
          <button
            class="link-btn"
            onclick={() => onDownloadText(activeStruct.trajectoryPdb!, safeFilename(`${activeStruct.id}_traj.pdb`))}
          >
            ↓ Trajectory ({activeStruct.trajectoryFrameCount ?? '?'}f)
          </button>
        {/if}
      </div>
    </div>

    <div class="stat-chips">
      <span class="stat-chip">L = {activeStruct.length} aa</span>
      {#if activeStruct.diagnostics?.chainCount && activeStruct.diagnostics.chainCount > 1}
        <span class="stat-chip">{activeStruct.diagnostics.chainCount} chains</span>
      {/if}
      <span class="stat-chip">R<sub>g</sub> = {fmtNum(activeStruct.geometry?.rg, 1, ' Å')}</span>
      <span class="stat-chip">breaks: {activeStruct.geometry?.chainBreaks ?? '—'}</span>
      <span class="stat-chip">clashes: {activeStruct.geometry?.clashes ?? '—'}</span>
      {#if activeStruct.representativeSelection}
        <span
          class="stat-chip cond-key"
          title={`Selected ${activeStruct.representativeSelection.selectedKey} from ${activeStruct.representativeSelection.totalMembers} member(s) by ${activeStruct.representativeSelection.criterion}`}
        >
          Rep: {activeStruct.representativeSelection.selectedKey} · {activeStruct.representativeSelection.criterion} ({activeStruct.representativeSelection.totalMembers})
        </span>
      {/if}
      {#if activeStruct.diagnostics?.ligandNames?.length}
        <span class="stat-chip cond-key">Ligands: {activeStruct.diagnostics.ligandNames.join(', ')}</span>
      {/if}
      {#if activeStruct.diagnostics?.nucleicAcidChains?.length}
        <span class="stat-chip cond-key">Nucleic acid: {activeStruct.diagnostics.nucleicAcidChains.join(', ')}</span>
      {/if}
      {#if activeStruct.trajectoryFrameCount}
        <span class="stat-chip">{activeStruct.trajectoryFrameCount} traj frames</span>
      {/if}
      {#if activeStruct.atomGranularity === 'ca-only'}
        <span class="warn-chip" title="Only Cα backbone coordinates were emitted by this model">
          Cα-only
        </span>
      {/if}
      {#if activeStruct.provenance?.quantization}
        <span class="stat-chip" title="Weight quantization">{activeStruct.provenance.quantization.toUpperCase()}</span>
      {/if}
      {#if activeStruct.provenance?.msaMode}
        <span class="stat-chip cond-key">
          MSA: {formatMsaBadge(activeStruct.provenance.msaMode, activeStruct.provenance.msaDepth)}
        </span>
      {/if}
      {#if activeStruct.provenance?.degradedFallback}
        <span class="warn-chip">⚠ {activeStruct.provenance.degradedFallback}</span>
      {/if}
      {#if activeStruct.provenance?.warnings?.length}
        {#each activeStruct.provenance.warnings as pw (pw)}
          <span class="warn-chip">⚠ {pw}</span>
        {/each}
      {/if}
    </div>

    <div class="seq-box">
      <div class="seq-head">
        <span>Backbone residue sequence ({activeStruct.sequence.length} aa · {structSeqStats.uniqueAaCount} unique AAs)</span>
        <button class="link-btn" onclick={() => copyText(activeStruct.sequence, 'struct-seq')}>
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
      <div class="seq-text">{activeStruct.sequence}</div>
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
  {:else if mutArt}
    <div class="meta-bar">
      <span>
        <b>{mutArt.label ?? 'Mutation Matrix'}</b> · <span>{formatAdapterName(mutArt.sourceModelId)}</span>
      </span>
      <span class="stat-chip">{mutArt.metric}</span>
    </div>
    {#if mutArt.provenance?.quantization || mutArt.provenance?.degradedFallback || mutArt.provenance?.warnings?.length}
      <div class="stat-chips">
        {#if mutArt.provenance?.quantization}
          <span class="stat-chip">{mutArt.provenance.quantization.toUpperCase()}</span>
        {/if}
        {#if mutArt.provenance?.degradedFallback}
          <span class="warn-chip">⚠ {mutArt.provenance.degradedFallback}</span>
        {/if}
        {#if mutArt.provenance?.warnings?.length}
          {#each mutArt.provenance.warnings as pw (pw)}
            <span class="warn-chip">⚠ {pw}</span>
          {/each}
        {/if}
      </div>
    {/if}
    <MutationHeatmap matrix={mutArt} {onDownloadText} />
  {:else if trackArt}
    <div class="meta-bar">
      <span>
        <b>{trackArt.label ?? trackArt.quantity}</b> · <span>{formatAdapterName(trackArt.sourceModelId)}</span>
      </span>
      {#if trackArt.unit}
        <span class="stat-chip">{trackArt.unit}</span>
      {/if}
    </div>
    <ResidueTrackPlot track={trackArt} {onDownloadText} />
  {:else if embArt}
    <div class="meta-bar">
      <span>
        <b>{embArt.label ?? 'Sequence Embedding'}</b> · <span>{formatAdapterName(embArt.sourceModelId)}</span>
      </span>
      <div class="dl-links">
        <button class="link-btn" onclick={() => copyText(embArt.sequence, 'emb-seq')}>
          {copyBtnLabel('emb-seq', 'Copy seq')}
        </button>
        <button
          class="link-btn"
          onclick={() =>
            onDownloadText(
              formatEmbeddingJson(embArt),
              safeFilename(`${embArt.id}.embedding.json`),
            )}
        >
          ↓ JSON
        </button>
      </div>
    </div>

    <div class="stat-chips">
      <span class="stat-chip cond-key">dim = {embArt.dim}</span>
      <span class="stat-chip">L = {embArt.sequence.length} aa</span>
      {#if isFiniteNum(embArt.pseudoPerplexity)}
        <span class="stat-chip cond-key" title="Sequence pseudo-perplexity (lower = more natural)">
          pseudo-PPL = {embArt.pseudoPerplexity.toFixed(2)}
        </span>
      {/if}
      {#if embStats}
        <span class="stat-chip" title="L2 norm of mean-pooled embedding vector">
          ‖pooled‖₂ = {embStats.l2Norm.toFixed(3)}
        </span>
        <span class="stat-chip">mean = {embStats.mean.toFixed(4)}</span>
        <span class="stat-chip">range [{embStats.min.toFixed(3)}, {embStats.max.toFixed(3)}]</span>
      {/if}
      {#if embArt.perResidue}
        <span class="stat-chip">per-residue [{embArt.sequence.length}×{embArt.dim}]</span>
      {/if}
      {#if embArt.provenance?.quantization}
        <span class="stat-chip">{embArt.provenance.quantization.toUpperCase()}</span>
      {/if}
      {#if embArt.provenance?.degradedFallback}
        <span class="warn-chip">⚠ {embArt.provenance.degradedFallback}</span>
      {/if}
      {#if embArt.provenance?.warnings?.length}
        {#each embArt.provenance.warnings as pw (pw)}
          <span class="warn-chip">⚠ {pw}</span>
        {/each}
      {/if}
    </div>

    {#if embStats && embStats.preview.length > 0}
      <div class="seq-box">
        <div class="seq-head">
          <span>Pooled embedding vector preview (first {embStats.preview.length} of {embArt.dim} dims)</span>
          {#if embArt.sequenceId}
            <code>{embArt.sequenceId}</code>
          {/if}
        </div>
        <div class="seq-text">
          [{embStats.preview.map((v) => (Number.isFinite(v) ? v.toFixed(4) : 'NaN')).join(', ')}{embArt.dim > embStats.preview.length ? ', …' : ''}]
        </div>
      </div>
    {/if}

    <div class="seq-box">
      <div class="seq-head">
        <span>Embedded sequence ({embArt.sequence.length} aa)</span>
        <button class="link-btn" onclick={() => copyText(embArt.sequence, 'emb-seq-box')}>
          {copyBtnLabel('emb-seq-box', 'Copy seq')}
        </button>
      </div>
      <div class="seq-text">{embArt.sequence}</div>
    </div>
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
  .campaign-bar {
    display: flex;
    flex-direction: column;
    gap: 7px;
    padding: 8px 10px;
    border-radius: 6px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
  }
  .campaign-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .campaign-stats {
    display: flex;
    align-items: center;
    gap: 8px 10px;
    flex-wrap: wrap;
    font-size: 11px;
    color: var(--text-secondary);
  }
  .campaign-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 999px;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
  }
  .campaign-metric {
    font-size: 11px;
    color: var(--text-muted);
  }
  .campaign-metric b {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-primary);
  }
  .campaign-sub {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
  }
  .campaign-exports {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .design-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    align-items: center;
  }
  .design-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    cursor: pointer;
  }
  .design-pill:hover {
    border-color: var(--border-strong);
    color: var(--text-primary);
  }
  .design-pill.active {
    border-color: var(--accent);
    background: var(--bg-surface-selected);
    color: var(--text-primary);
    font-weight: 600;
  }
  .pill-count {
    font-size: 9.5px;
    color: var(--text-dim);
  }
  .design-tag {
    display: inline-block;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 4px;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
    margin-right: 4px;
    white-space: nowrap;
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
