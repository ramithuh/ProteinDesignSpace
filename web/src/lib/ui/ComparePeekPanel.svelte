<script lang="ts">
  import PaePlot from '$lib/viewer/PaePlot.svelte';
  import {
    exportJoinedCollectionCsv,
    type ComparisonPairingMode,
    type ComparisonState,
  } from '$lib/engine/compare';
  import type {
    ArtifactProvenance,
    AtomGranularity,
    CollectionKind,
    MsaMode,
    ReferenceRelation,
    RepresentativeSelection,
  } from '$lib/engine/types';

  export interface CompareBranchMeta {
    referenceRelation?: ReferenceRelation;
    referenceLabel?: string;
    lineageWarning?: string;
    lineageMismatch?: boolean;
    representativeSelection?: RepresentativeSelection;
    collectionKind?: CollectionKind;
    ipsae?: number | null;
    ipsaeMin?: number | null;
    failedGates?: string[];
    msaMode?: MsaMode;
    msaDepth?: number;
    provenance?: ArtifactProvenance;
    atomGranularity?: AtomGranularity;
    chainLengths?: number[];
    foldModelId?: string;
  }

  interface Props {
    comparison: ComparisonState;
    scRmsdThreshold?: number;
    plddtThreshold?: number;
    ipaeThreshold?: number;
    motifRmsdThreshold?: number;
    ipsaeThreshold?: number;
    thresholdsByNodeId?: Record<
      string,
      { scRmsd: number; plddt: number; ipae?: number; motifRmsd?: number; ipsae?: number }
    >;
    branchMetaByNodeId?: Record<string, CompareBranchMeta>;
    isRunning?: boolean;
    onSelectBranchCandidate: (nodeId: string, idx: number) => void;
    onChangePairingMode?: (mode: ComparisonPairingMode) => void;
    onSelectCandidateKey?: (key: string) => void;
    onFocusSingle: (nodeId: string) => void;
    onRunUpToNode?: (nodeId: string) => void;
    onDownloadText?: (text: string, filename: string) => void;
    onExitCompare: () => void;
  }

  let {
    comparison,
    scRmsdThreshold = 2.0,
    plddtThreshold = 70,
    ipaeThreshold = 10.0,
    motifRmsdThreshold = 1.0,
    ipsaeThreshold = 0,
    thresholdsByNodeId = undefined,
    branchMetaByNodeId = undefined,
    isRunning = false,
    onSelectBranchCandidate,
    onChangePairingMode = undefined,
    onSelectCandidateKey = undefined,
    onFocusSingle,
    onRunUpToNode = undefined,
    onDownloadText = undefined,
    onExitCompare,
  }: Props = $props();

  let branches = $derived(comparison.branches);

  function isFiniteNum(v: unknown): v is number {
    return typeof v === 'number' && Number.isFinite(v);
  }

  function fmtNum(v: number | null | undefined, digits: number, suffix = ''): string {
    if (!isFiniteNum(v)) return '—';
    return `${v.toFixed(digits)}${suffix}`;
  }

  function branchScRmsdThreshold(nodeId: string): number {
    return thresholdsByNodeId?.[nodeId]?.scRmsd ?? scRmsdThreshold;
  }

  function branchPlddtThreshold(nodeId: string): number {
    return thresholdsByNodeId?.[nodeId]?.plddt ?? plddtThreshold;
  }

  function branchIpaeThreshold(nodeId: string): number {
    return thresholdsByNodeId?.[nodeId]?.ipae ?? ipaeThreshold;
  }

  function branchMotifRmsdThreshold(nodeId: string): number {
    return thresholdsByNodeId?.[nodeId]?.motifRmsd ?? motifRmsdThreshold;
  }

  function branchIpsaeThreshold(nodeId: string): number {
    const t = thresholdsByNodeId?.[nodeId]?.ipsae ?? ipsaeThreshold;
    return t > 0 ? t : 0.6;
  }

  function getBranchIpsae(b: ComparisonState['branches'][number]): number | null {
    const direct = (b as { ipsae?: number | null }).ipsae;
    if (isFiniteNum(direct)) return direct;
    const metaVal = branchMetaByNodeId?.[b.nodeId]?.ipsae;
    return isFiniteNum(metaVal) ? metaVal : null;
  }

  function formatMsaBadge(msaMode?: MsaMode, msaDepth?: number): string {
    const depth = isFiniteNum(msaDepth) && msaDepth > 0 ? Math.round(msaDepth) : null;
    if (msaMode === 'mmseqs2') return depth !== null ? `MMseqs2 · ${depth} seqs` : 'MMseqs2';
    if (msaMode === 'target-only') return depth !== null ? `Target MSA · ${depth} seqs` : 'Target MSA';
    if (msaMode === 'a3m') return depth !== null ? `.a3m · ${depth} seqs` : '.a3m';
    return 'single-seq';
  }

  function formatQuantChip(meta?: CompareBranchMeta): string | null {
    if (meta?.provenance?.quantization) return meta.provenance.quantization.toUpperCase();
    if (meta?.foldModelId?.toLowerCase().includes('-int5')) return 'INT5';
    return null;
  }

  function formatCollectionKind(kind?: CollectionKind): string | null {
    if (kind === 'independent-candidates') return 'Candidates';
    if (kind === 'multi-sample') return 'Multi-sample';
    if (kind === 'ordered-frames') return 'Ordered frames';
    return null;
  }

  function formatRefBadge(meta?: CompareBranchMeta): { text: string; tone: 'ok' | 'info' | 'warn' | 'err' } | null {
    if (!meta) return null;
    if (meta.referenceRelation === 'self-consistency') return { text: 'Self-consistency', tone: 'ok' };
    if (meta.referenceRelation === 'external-mapped') return { text: 'External ref (mapped)', tone: 'info' };
    if (meta.referenceRelation === 'external-unlinked') return { text: 'External ref', tone: 'warn' };
    if (meta.referenceRelation === 'external-mismatch' || meta.lineageMismatch) {
      return { text: 'External ref · Mismatch', tone: 'err' };
    }
    return null;
  }

  function handleExportJoinedCsv() {
    if (!onDownloadText || comparison.pairedCandidates.length === 0) return;
    const completeRowCount = comparison.pairedCandidates.filter(
      (r) => r.missingBranchIds.length === 0,
    ).length;
    const csv = exportJoinedCollectionCsv({
      keys: comparison.candidateKeys,
      branchIds: branches.map((b) => b.nodeId),
      rows: comparison.pairedCandidates,
      completeRowCount,
      partialRowCount: comparison.pairedCandidates.length - completeRowCount,
    });
    onDownloadText(csv, 'denovo_compare_joined.csv');
  }

  // Best-value highlights per metric row (strictly finite-guarded to avoid NaN/Infinity)
  let bestRmsd = $derived.by((): number | null => {
    const vals = branches.map((b) => b.scRmsd).filter(isFiniteNum);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestTm = $derived.by((): number | null => {
    const vals = branches.map((b) => b.scTm).filter(isFiniteNum);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestPlddt = $derived.by((): number | null => {
    const vals = branches
      .filter((b) => b.confidenceAvailable !== false)
      .map((b) => b.meanPlddt)
      .filter(isFiniteNum);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestPtm = $derived.by((): number | null => {
    const vals = branches.map((b) => b.ptm).filter(isFiniteNum);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestIptm = $derived.by((): number | null => {
    const vals = branches.map((b) => b.iptm).filter(isFiniteNum);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestIpae = $derived.by((): number | null => {
    const vals = branches.map((b) => b.ipae).filter(isFiniteNum);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestIpsae = $derived.by((): number | null => {
    const vals = branches.map((b) => getBranchIpsae(b)).filter(isFiniteNum);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestBinderScRmsd = $derived.by((): number | null => {
    const vals = branches.map((b) => b.binderScRmsd).filter(isFiniteNum);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestTargetAlignedBinderRmsd = $derived.by((): number | null => {
    const vals = branches.map((b) => b.targetAlignedBinderRmsd).filter(isFiniteNum);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestBinderPlddt = $derived.by((): number | null => {
    const vals = branches.map((b) => b.binderPlddt).filter(isFiniteNum);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestRefoldMotifRmsd = $derived.by((): number | null => {
    const vals = branches.map((b) => b.refoldMotifRmsd).filter(isFiniteNum);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestGenMotifRmsd = $derived.by((): number | null => {
    const vals = branches.map((b) => b.genMotifRmsd).filter(isFiniteNum);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });

  let hasAnyRmsd = $derived(branches.some((b) => isFiniteNum(b.scRmsd)));
  let hasAnyPlddt = $derived(
    branches.some((b) => isFiniteNum(b.meanPlddt) || b.confidenceAvailable === false),
  );
  let hasAnyPtm = $derived(branches.some((b) => isFiniteNum(b.ptm)));
  let hasAnyIptm = $derived(branches.some((b) => isFiniteNum(b.iptm)));
  let hasAnyIpae = $derived(branches.some((b) => isFiniteNum(b.ipae)));
  let hasAnyIpsae = $derived(branches.some((b) => isFiniteNum(getBranchIpsae(b))));
  let hasAnyMsa = $derived(
    branches.some((b) => {
      const meta = branchMetaByNodeId?.[b.nodeId];
      return b.kind === 'fold' || b.kind === 'metrics' || Boolean(meta?.msaMode || meta?.provenance?.msaMode);
    }),
  );
  let hasAnyProvenance = $derived(
    branches.some((b) => {
      const meta = branchMetaByNodeId?.[b.nodeId];
      return Boolean(
        formatQuantChip(meta) ||
          meta?.atomGranularity === 'ca-only' ||
          meta?.provenance?.degradedFallback,
      );
    }),
  );
  let hasAnyBinderScRmsd = $derived(branches.some((b) => isFiniteNum(b.binderScRmsd)));
  let hasAnyTargetAlignedBinderRmsd = $derived(
    branches.some((b) => isFiniteNum(b.targetAlignedBinderRmsd)),
  );
  let hasAnyBinderPlddt = $derived(branches.some((b) => isFiniteNum(b.binderPlddt)));
  let hasAnyRefoldMotifRmsd = $derived(branches.some((b) => isFiniteNum(b.refoldMotifRmsd)));
  let hasAnyGenMotifRmsd = $derived(branches.some((b) => isFiniteNum(b.genMotifRmsd)));
  let hasAnyTime = $derived(branches.some((b) => isFiniteNum(b.foldElapsedMs)));
  let hasAnyRefMeta = $derived(
    branches.some((b) => Boolean(formatRefBadge(branchMetaByNodeId?.[b.nodeId]))),
  );
  let hasAnyCollectionMeta = $derived(
    branches.some((b) => {
      const meta = branchMetaByNodeId?.[b.nodeId];
      return Boolean(meta?.representativeSelection || b.collectionKind || meta?.collectionKind);
    }),
  );
  let lineageWarnings = $derived(
    branches
      .map((b) => {
        const w = branchMetaByNodeId?.[b.nodeId]?.lineageWarning;
        return w ? { badge: b.badge, label: b.label, warning: w } : null;
      })
      .filter((x): x is { badge: string; label: string; warning: string } => x !== null),
  );
  let paeBranches = $derived(branches.filter((b) => b.pae !== null));
</script>

<div class="cmp-panel">
  <div class="cmp-head">
    <div class="cmp-title-wrap">
      <span class="cmp-title">COMPARE · {branches.length} blocks</span>
      {#if comparison.forcedSideBySide}
        <span class="cmp-note">Different lengths · Side-by-side</span>
      {:else if comparison.isPreAlignedSharedBackbone}
        <span class="cmp-note-info" title="All branches are scorer refolds already Kabsch-aligned to the same designed backbone">
          Shared backbone frame
        </span>
      {/if}
    </div>
    <div class="cmp-head-actions">
      {#if comparison.pairedCandidates.length > 0 && onDownloadText}
        <button
          type="button"
          class="csv-btn"
          onclick={handleExportJoinedCsv}
          title="Export keyed multi-branch candidate comparison table as CSV (with explicit MISSING & N/A markers)"
        >
          ↓ Export Joined CSV
        </button>
      {/if}
      <button type="button" class="exit-btn" onclick={onExitCompare} title="Return to single-block view (Esc)">
        × Exit Compare
      </button>
    </div>
  </div>

  {#if onChangePairingMode || comparison.candidateKeys.length > 0}
    <div class="pairing-bar">
      {#if onChangePairingMode}
        <div class="pair-seg" role="group" aria-label="Candidate pairing mode" title={comparison.pairingLabel}>
          <button
            type="button"
            class="pair-seg-btn"
            class:active={comparison.pairingMode === 'by-candidate'}
            onclick={() => onChangePairingMode('by-candidate')}
          >
            By candidate
          </button>
          <button
            type="button"
            class="pair-seg-btn"
            class:active={comparison.pairingMode === 'branch-best'}
            onclick={() => onChangePairingMode('branch-best')}
          >
            Branch best
          </button>
        </div>
      {:else}
        <span class="cmp-note-info">{comparison.pairingLabel}</span>
      {/if}

      {#if comparison.pairingMode === 'by-candidate' && comparison.candidateKeys.length > 0 && onSelectCandidateKey}
        <label class="cand-key-pick">
          <span>Candidate:</span>
          <select
            class="cand-sel"
            value={comparison.selectedCandidateKey ?? ''}
            onchange={(e) => onSelectCandidateKey(e.currentTarget.value)}
          >
            {#each comparison.pairedCandidates as row (row.key)}
              <option value={row.key}>
                {row.key} ({row.presentCount}/{branches.length}{row.missingBranchIds.length > 0 ? ' · partial' : ''})
              </option>
            {/each}
          </select>
        </label>
      {:else if comparison.pairingMode === 'branch-best'}
        <span class="pairing-hint">Each branch shows its own top-ranked candidate independently.</span>
      {/if}
    </div>
  {/if}

  {#if lineageWarnings.length > 0}
    <div class="lineage-warn-list" role="status">
      {#each lineageWarnings as lw (lw.badge)}
        <div class="lineage-warn-item">
          <b>{lw.badge} ({lw.label}):</b> {lw.warning}
        </div>
      {/each}
    </div>
  {/if}

  {#if comparison.pairwise && isFiniteNum(comparison.pairwise.rmsd)}
    <div class="agreement-banner">
      <span>
        {#if comparison.pairwise.isSubstructure}
          {comparison.pairwise.alignmentLabel ?? (comparison.pairwise.hasSharedTemplate ? 'Motif Kabsch' : 'Substructure Kabsch')} Cα RMSD (<b>{comparison.pairwise.badgeA} ↔ {comparison.pairwise.badgeB}</b> · {comparison.pairwise.alignedResidues} aa):
          <b>{fmtNum(comparison.pairwise.rmsd, 2, ' Å')}</b>
        {:else}
          Inter-model Kabsch Cα RMSD (<b>{comparison.pairwise.badgeA} ↔ {comparison.pairwise.badgeB}</b>):
          <b>{fmtNum(comparison.pairwise.rmsd, 2, ' Å')}</b> · TM <b>{fmtNum(comparison.pairwise.tmScore, 3)}</b>
        {/if}
      </span>
      {#if comparison.allPairwise && comparison.allPairwise.length > 1}
        <div class="extra-pairs">
          {#each comparison.allPairwise.slice(1) as pair (`${pair.badgeA}-${pair.badgeB}`)}
            {#if isFiniteNum(pair.rmsd)}
              <span class="pair-chip">
                <b>{pair.badgeA}↔{pair.badgeB}</b>: {fmtNum(pair.rmsd, 2, ' Å')}{#if !pair.isSubstructure && isFiniteNum(pair.tmScore)} · TM {fmtNum(pair.tmScore, 2)}{/if}
              </span>
            {/if}
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <div class="table-scroll">
    <table class="cmp-table">
      <thead>
        <tr>
          <th class="metric-col">Metric</th>
          {#each branches as b (b.nodeId)}
            <th>
              <button
                type="button"
                class="col-hdr-btn"
                title="Click to focus only this block"
                onclick={() => onFocusSingle(b.nodeId)}
              >
                <span class="badge-dot" style="background: {b.colorCss}">{b.badge}</span>
                <span class="col-name">{b.label}</span>
              </button>
            </th>
          {/each}
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="m-k">Candidate</td>
          {#each branches as b (b.nodeId)}
            <td>
              {#if !b.hasOutputs}
                {#if onRunUpToNode}
                  <button
                    type="button"
                    class="mini-run"
                    disabled={isRunning}
                    onclick={() => onRunUpToNode(b.nodeId)}
                  >
                    ▶ Run
                  </button>
                {:else}
                  <span class="dim">Not run</span>
                {/if}
              {:else if b.missingSelectedCandidate}
                <span class="missing-badge" title={`Candidate ${b.candidateKey ?? ''} is not present in this branch`}>
                  MISSING
                </span>
              {:else if !b.has3D}
                <span class="dim">No 3D</span>
              {:else if comparison.pairingMode !== 'by-candidate' && b.candidateCount > 1}
                <select
                  class="cand-sel"
                  value={b.selectedIndex}
                  onchange={(e) => onSelectBranchCandidate(b.nodeId, Number(e.currentTarget.value))}
                >
                  {#each Array.from({ length: b.candidateCount }, (_, idx) => idx) as idx (idx)}
                    <option value={idx}>#{idx + 1}</option>
                  {/each}
                </select>
              {:else}
                <code class="mono-sm">{b.candidateKey ?? b.sequenceId ?? (b.length ? `${b.length} aa` : '—')}</code>
              {/if}
            </td>
          {/each}
        </tr>

        {#if hasAnyRefMeta}
          <tr>
            <td class="m-k">Reference</td>
            {#each branches as b (b.nodeId)}
              {@const meta = branchMetaByNodeId?.[b.nodeId]}
              {@const refBadge = formatRefBadge(meta)}
              <td>
                {#if refBadge}
                  <span
                    class="ref-badge"
                    class:tone-ok={refBadge.tone === 'ok'}
                    class:tone-info={refBadge.tone === 'info'}
                    class:tone-warn={refBadge.tone === 'warn'}
                    class:tone-err={refBadge.tone === 'err'}
                    title={meta?.referenceLabel ?? refBadge.text}
                  >
                    {refBadge.text}
                  </span>
                {:else}
                  <span class="dim">—</span>
                {/if}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyCollectionMeta}
          <tr>
            <td class="m-k">Collection</td>
            {#each branches as b (b.nodeId)}
              {@const meta = branchMetaByNodeId?.[b.nodeId]}
              {@const colKind = formatCollectionKind(b.collectionKind ?? meta?.collectionKind)}
              {@const rep = meta?.representativeSelection}
              <td>
                {#if rep}
                  <span
                    class="mono-sm"
                    title={`Representative: ${rep.selectedKey} (${rep.criterion} of ${rep.totalMembers})`}
                  >
                    {colKind ? `${colKind} · ` : ''}{rep.criterion}
                  </span>
                {:else if colKind}
                  <span class="mono-sm">{colKind}</span>
                {:else}
                  <span class="dim">—</span>
                {/if}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyRmsd}
          <tr>
            <td class="m-k">scRMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestRmsd !== null && isFiniteNum(b.scRmsd) && b.scRmsd === bestRmsd}
                class:pass-val={isFiniteNum(b.scRmsd) && b.scRmsd < branchScRmsdThreshold(b.nodeId)}
              >
                {fmtNum(b.scRmsd, 2, ' Å')}
              </td>
            {/each}
          </tr>
          <tr>
            <td class="m-k">scTM</td>
            {#each branches as b (b.nodeId)}
              <td class:win={bestTm !== null && isFiniteNum(b.scTm) && b.scTm === bestTm}>
                {fmtNum(b.scTm, 3)}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyBinderScRmsd}
          <tr>
            <td class="m-k">Binder scRMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestBinderScRmsd !== null && isFiniteNum(b.binderScRmsd) && b.binderScRmsd === bestBinderScRmsd}
                class:pass-val={isFiniteNum(b.binderScRmsd) && b.binderScRmsd < branchScRmsdThreshold(b.nodeId)}
              >
                {fmtNum(b.binderScRmsd, 2, ' Å')}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyTargetAlignedBinderRmsd}
          <tr>
            <td class="m-k">Target-aligned RMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestTargetAlignedBinderRmsd !== null && isFiniteNum(b.targetAlignedBinderRmsd) && b.targetAlignedBinderRmsd === bestTargetAlignedBinderRmsd}
              >
                {fmtNum(b.targetAlignedBinderRmsd, 2, ' Å')}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyRefoldMotifRmsd}
          <tr>
            <td class="m-k">Motif RMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestRefoldMotifRmsd !== null && isFiniteNum(b.refoldMotifRmsd) && b.refoldMotifRmsd === bestRefoldMotifRmsd}
                class:pass-val={isFiniteNum(b.refoldMotifRmsd) && b.refoldMotifRmsd < branchMotifRmsdThreshold(b.nodeId)}
              >
                {fmtNum(b.refoldMotifRmsd, 2, ' Å')}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyGenMotifRmsd}
          <tr>
            <td class="m-k">Gen Motif RMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestGenMotifRmsd !== null && isFiniteNum(b.genMotifRmsd) && b.genMotifRmsd === bestGenMotifRmsd}
                class:pass-val={isFiniteNum(b.genMotifRmsd) && b.genMotifRmsd < branchMotifRmsdThreshold(b.nodeId)}
              >
                {fmtNum(b.genMotifRmsd, 2, ' Å')}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyPlddt}
          <tr>
            <td class="m-k">pLDDT</td>
            {#each branches as b (b.nodeId)}
              {@const confAvail = b.confidenceAvailable !== false && isFiniteNum(b.meanPlddt)}
              <td
                class:win={bestPlddt !== null && confAvail && b.meanPlddt === bestPlddt}
                class:pass-val={confAvail && b.meanPlddt! >= branchPlddtThreshold(b.nodeId)}
              >
                {#if b.confidenceAvailable === false}
                  <span class="dim">N/A</span>
                {:else}
                  {fmtNum(b.meanPlddt, 1)}
                {/if}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyBinderPlddt}
          <tr>
            <td class="m-k">Binder pLDDT</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestBinderPlddt !== null && isFiniteNum(b.binderPlddt) && b.binderPlddt === bestBinderPlddt}
                class:pass-val={isFiniteNum(b.binderPlddt) && b.binderPlddt >= branchPlddtThreshold(b.nodeId)}
              >
                {fmtNum(b.binderPlddt, 1)}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyPtm}
          <tr>
            <td class="m-k">pTM</td>
            {#each branches as b (b.nodeId)}
              <td class:win={bestPtm !== null && isFiniteNum(b.ptm) && b.ptm === bestPtm}>
                {fmtNum(b.ptm, 3)}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyIptm}
          <tr>
            <td class="m-k">ipTM</td>
            {#each branches as b (b.nodeId)}
              <td class:win={bestIptm !== null && isFiniteNum(b.iptm) && b.iptm === bestIptm}>
                {fmtNum(b.iptm, 3)}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyIpae}
          <tr>
            <td class="m-k">iPAE</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestIpae !== null && isFiniteNum(b.ipae) && b.ipae === bestIpae}
                class:pass-val={isFiniteNum(b.ipae) && b.ipae < branchIpaeThreshold(b.nodeId)}
              >
                {fmtNum(b.ipae, 1, ' Å')}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyIpsae}
          <tr>
            <td class="m-k">ipSAE</td>
            {#each branches as b (b.nodeId)}
              {@const ipsaeVal = getBranchIpsae(b)}
              <td
                class:win={bestIpsae !== null && isFiniteNum(ipsaeVal) && ipsaeVal === bestIpsae}
                class:pass-val={isFiniteNum(ipsaeVal) && ipsaeVal >= branchIpsaeThreshold(b.nodeId)}
              >
                {fmtNum(ipsaeVal, 2)}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyRmsd}
          <tr>
            <td class="m-k">Pass</td>
            {#each branches as b (b.nodeId)}
              {@const gates = branchMetaByNodeId?.[b.nodeId]?.failedGates ?? []}
              <td class="pass-cell">
                {#if b.designable !== null}
                  <div class="verdict-wrap">
                    <span
                      class="verdict"
                      class:pass={b.designable}
                      title={b.designable
                        ? 'All gates passed'
                        : gates.length > 0
                          ? `Failed gates: ${gates.join(' · ')}`
                          : 'Failed designability gate'}
                    >
                      {b.designable ? 'PASS' : 'FAIL'}
                    </span>
                    {#if !b.designable && gates.length > 0}
                      <div class="gate-chips">
                        {#each gates as g (g)}
                          <span class="gate-chip" title={g}>{g}</span>
                        {/each}
                      </div>
                    {/if}
                  </div>
                {:else}
                  —
                {/if}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyMsa}
          <tr>
            <td class="m-k">MSA</td>
            {#each branches as b (b.nodeId)}
              {@const meta = branchMetaByNodeId?.[b.nodeId]}
              {@const msaMode = meta?.msaMode ?? meta?.provenance?.msaMode}
              {@const msaDepth = meta?.msaDepth ?? meta?.provenance?.msaDepth}
              <td>
                {#if b.kind === 'fold' || b.kind === 'metrics' || msaMode}
                  <span class="msa-chip">{formatMsaBadge(msaMode, msaDepth)}</span>
                {:else}
                  <span class="dim">—</span>
                {/if}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyProvenance}
          <tr>
            <td class="m-k">Provenance</td>
            {#each branches as b (b.nodeId)}
              {@const meta = branchMetaByNodeId?.[b.nodeId]}
              {@const qChip = formatQuantChip(meta)}
              <td class="prov-cell">
                {#if qChip || meta?.atomGranularity === 'ca-only' || meta?.provenance?.degradedFallback}
                  <div class="prov-chips">
                    {#if qChip}
                      <span class="prov-chip">{qChip}</span>
                    {/if}
                    {#if meta?.atomGranularity === 'ca-only'}
                      <span class="prov-chip warn" title="Cα-only backbone coordinates">Cα-only</span>
                    {/if}
                    {#if meta?.provenance?.degradedFallback}
                      <span class="prov-chip warn" title={meta.provenance.degradedFallback}>
                        ⚠ {meta.provenance.degradedFallback}
                      </span>
                    {/if}
                  </div>
                {:else}
                  <span class="dim">—</span>
                {/if}
              </td>
            {/each}
          </tr>
        {/if}

        <tr>
          <td class="m-k">Length / R<sub>g</sub></td>
          {#each branches as b (b.nodeId)}
            <td>
              {#if b.length !== null}
                {b.length} aa{#if b.geometry && isFiniteNum(b.geometry.rg)} · {b.geometry.rg.toFixed(1)} Å{/if}
              {:else}
                —
              {/if}
            </td>
          {/each}
        </tr>

        {#if hasAnyTime}
          <tr>
            <td class="m-k">Fold time</td>
            {#each branches as b (b.nodeId)}
              <td>{isFiniteNum(b.foldElapsedMs) ? `${(b.foldElapsedMs / 1000).toFixed(1)} s` : '—'}</td>
            {/each}
          </tr>
        {/if}
      </tbody>
    </table>
  </div>

  {#if paeBranches.length > 0}
    <div class="pae-grid">
      {#each paeBranches as b (b.nodeId)}
        <div class="pae-item">
          <div class="pae-lbl">
            <span class="badge-dot sm" style="background: {b.colorCss}">{b.badge}</span>
            <span>{b.label}</span>
          </div>
          <PaePlot
            pae={b.pae!}
            chainLengths={b.chainLengths ?? branchMetaByNodeId?.[b.nodeId]?.chainLengths}
            size={136}
          />
        </div>
      {/each}
    </div>
  {/if}

  <p class="cmp-tip">
    Hold <kbd>⌘</kbd>/<kbd>Ctrl</kbd> or <kbd>Shift</kbd> while clicking blocks on the canvas to add/remove up to 4 blocks.
  </p>
</div>

<style>
  .cmp-panel {
    padding: 12px 14px 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 0;
    max-width: 100%;
    box-sizing: border-box;
  }
  .cmp-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }
  .cmp-title-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .cmp-title {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.3px;
  }
  .cmp-note {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    padding: 1px 6px;
    border-radius: 999px;
  }
  .cmp-note-info {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-secondary);
    background: var(--bg-surface-2);
    border: 1px solid var(--border-strong);
    padding: 1px 6px;
    border-radius: 999px;
  }
  .extra-pairs {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 5px;
    padding-top: 5px;
    border-top: 1px solid var(--border-subtle);
  }
  .pair-chip {
    font-size: 10px;
    color: var(--text-secondary);
  }
  .exit-btn {
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    border-radius: 5px;
    padding: 3px 8px;
    font-size: 11px;
    cursor: pointer;
  }
  .exit-btn:hover {
    color: var(--text-primary);
    border-color: var(--accent);
  }
  .cmp-head-actions {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .csv-btn {
    border: 1px solid var(--border-strong);
    background: var(--bg-pill);
    color: var(--accent-soft);
    border-radius: 5px;
    padding: 3px 8px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    cursor: pointer;
  }
  .csv-btn:hover {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .pairing-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 11px;
  }
  .pair-seg {
    display: inline-flex;
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    overflow: hidden;
    background: var(--bg-surface-2);
  }
  .pair-seg-btn {
    border: none;
    background: transparent;
    color: var(--text-muted);
    padding: 3px 8px;
    font-size: 10.5px;
    font-weight: 500;
    cursor: pointer;
    font-family: inherit;
  }
  .pair-seg-btn:not(:last-child) {
    border-right: 1px solid var(--border-subtle);
  }
  .pair-seg-btn.active {
    background: var(--accent);
    color: #fff;
    font-weight: 600;
  }
  .cand-key-pick {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: var(--text-muted);
    font-size: 10.5px;
  }
  .pairing-hint {
    font-size: 10.5px;
    color: var(--text-dim);
  }
  .lineage-warn-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .lineage-warn-item {
    font-size: 10.5px;
    line-height: 1.4;
    padding: 5px 8px;
    border-radius: 5px;
    background: var(--warn-bg);
    color: var(--warn-text);
    border: 1px solid var(--warn-border);
  }
  .missing-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--warn-bg);
    color: var(--warn-text);
    border: 1px solid var(--warn-border);
  }
  .ref-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
    color: var(--text-secondary);
  }
  .ref-badge.tone-ok {
    background: var(--ok-bg);
    color: var(--ok-text);
    border-color: var(--ok-border);
  }
  .ref-badge.tone-info {
    background: var(--bg-pill);
    color: var(--accent-soft);
  }
  .ref-badge.tone-warn {
    background: var(--warn-bg);
    color: var(--warn-text);
    border-color: var(--warn-border);
  }
  .ref-badge.tone-err {
    background: var(--err-bg);
    color: var(--err-strong);
    border-color: var(--err-border);
  }
  .agreement-banner {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--accent-soft);
    background: var(--bg-pill);
    border: 1px solid var(--border-strong);
    padding: 6px 9px;
    border-radius: 6px;
  }
  .table-scroll {
    overflow-x: auto;
  }
  .cmp-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
  .cmp-table th,
  .cmp-table td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border-subtle);
    text-align: left;
    white-space: nowrap;
  }
  .cmp-table th {
    color: var(--text-muted);
    font-weight: 600;
    font-size: 10.5px;
  }
  .m-k {
    color: var(--text-muted);
    font-weight: 500;
  }
  .col-hdr-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: none;
    border: none;
    padding: 0;
    color: var(--text-primary);
    font-family: inherit;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }
  .col-hdr-btn:hover .col-name {
    text-decoration: underline;
  }
  .col-name {
    max-width: 110px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .badge-dot {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 999px;
    color: #090d16;
    font-size: 9.5px;
    font-weight: 800;
    flex-shrink: 0;
  }
  .badge-dot.sm {
    width: 14px;
    height: 14px;
    font-size: 8.5px;
  }
  .win {
    color: var(--ok-text);
    font-weight: 700;
  }
  .pass-val {
    color: var(--ok-text);
  }
  .dim {
    color: var(--text-dim);
  }
  .mono-sm {
    font-size: 10px;
    color: var(--text-secondary);
  }
  .cand-sel {
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 1px 4px;
    font-family: inherit;
    font-size: 10.5px;
  }
  .mini-run {
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
  }
  .mini-run:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .verdict {
    font-size: 9.5px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 999px;
    background: var(--err-bg);
    color: var(--err-strong);
    border: 1px solid var(--err-border);
  }
  .verdict.pass {
    background: var(--ok-bg);
    color: var(--ok-text);
    border-color: var(--ok-border);
  }
  .pass-cell,
  .prov-cell {
    white-space: normal;
  }
  .verdict-wrap {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
  .gate-chips,
  .prov-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
  }
  .gate-chip {
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 4px;
    background: var(--err-bg);
    color: var(--err-strong);
    border: 1px solid var(--err-border);
    white-space: nowrap;
  }
  .msa-chip {
    font-size: 9.5px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
  }
  .prov-chip {
    font-size: 9.5px;
    padding: 1px 5px;
    border-radius: 4px;
    background: var(--bg-surface-2);
    color: var(--text-muted);
    border: 1px solid var(--border-subtle);
    white-space: nowrap;
  }
  .prov-chip.warn {
    background: var(--warn-bg);
    color: var(--warn-text);
    border-color: var(--warn-border);
  }
  .pae-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding-top: 4px;
  }
  .pae-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .pae-lbl {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-secondary);
  }
  .cmp-tip {
    margin: 2px 0 0;
    font-size: 10.5px;
    color: var(--text-dim);
  }
  .cmp-tip kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    padding: 0 4px;
    border-radius: 3px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
  }
</style>
