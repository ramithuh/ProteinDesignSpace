<script lang="ts">
  import PaePlot from '$lib/viewer/PaePlot.svelte';
  import type { ComparisonState } from '$lib/engine/compare';

  interface Props {
    comparison: ComparisonState;
    scRmsdThreshold?: number;
    plddtThreshold?: number;
    thresholdsByNodeId?: Record<string, { scRmsd: number; plddt: number }>;
    isRunning?: boolean;
    onSelectBranchCandidate: (nodeId: string, idx: number) => void;
    onFocusSingle: (nodeId: string) => void;
    onRunUpToNode?: (nodeId: string) => void;
    onExitCompare: () => void;
  }

  let {
    comparison,
    scRmsdThreshold = 2.0,
    plddtThreshold = 70,
    thresholdsByNodeId = undefined,
    isRunning = false,
    onSelectBranchCandidate,
    onFocusSingle,
    onRunUpToNode = undefined,
    onExitCompare,
  }: Props = $props();

  let branches = $derived(comparison.branches);

  function branchScRmsdThreshold(nodeId: string): number {
    return thresholdsByNodeId?.[nodeId]?.scRmsd ?? scRmsdThreshold;
  }

  function branchPlddtThreshold(nodeId: string): number {
    return thresholdsByNodeId?.[nodeId]?.plddt ?? plddtThreshold;
  }

  // Best-value highlights per metric row
  let bestRmsd = $derived.by((): number | null => {
    const vals = branches.map((b) => b.scRmsd).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestTm = $derived.by((): number | null => {
    const vals = branches.map((b) => b.scTm).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestPlddt = $derived.by((): number | null => {
    const vals = branches.map((b) => b.meanPlddt).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestPtm = $derived.by((): number | null => {
    const vals = branches.map((b) => b.ptm).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestIptm = $derived.by((): number | null => {
    const vals = branches.map((b) => b.iptm).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestIpae = $derived.by((): number | null => {
    const vals = branches.map((b) => b.ipae).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestBinderScRmsd = $derived.by((): number | null => {
    const vals = branches.map((b) => b.binderScRmsd).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestTargetAlignedBinderRmsd = $derived.by((): number | null => {
    const vals = branches
      .map((b) => b.targetAlignedBinderRmsd)
      .filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestBinderPlddt = $derived.by((): number | null => {
    const vals = branches.map((b) => b.binderPlddt).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.max(...vals) : null;
  });
  let bestRefoldMotifRmsd = $derived.by((): number | null => {
    const vals = branches.map((b) => b.refoldMotifRmsd).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });
  let bestGenMotifRmsd = $derived.by((): number | null => {
    const vals = branches.map((b) => b.genMotifRmsd).filter((v): v is number => v !== null);
    return vals.length >= 2 ? Math.min(...vals) : null;
  });

  let hasAnyRmsd = $derived(branches.some((b) => b.scRmsd !== null));
  let hasAnyPlddt = $derived(branches.some((b) => b.meanPlddt !== null));
  let hasAnyPtm = $derived(branches.some((b) => b.ptm !== null));
  let hasAnyIptm = $derived(branches.some((b) => b.iptm !== null));
  let hasAnyIpae = $derived(branches.some((b) => b.ipae !== null));
  let hasAnyBinderScRmsd = $derived(branches.some((b) => b.binderScRmsd !== null));
  let hasAnyTargetAlignedBinderRmsd = $derived(
    branches.some((b) => b.targetAlignedBinderRmsd !== null),
  );
  let hasAnyBinderPlddt = $derived(branches.some((b) => b.binderPlddt !== null));
  let hasAnyRefoldMotifRmsd = $derived(branches.some((b) => b.refoldMotifRmsd !== null));
  let hasAnyGenMotifRmsd = $derived(branches.some((b) => b.genMotifRmsd !== null));
  let hasAnyTime = $derived(branches.some((b) => b.foldElapsedMs !== null));
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
    <button type="button" class="exit-btn" onclick={onExitCompare} title="Return to single-block view (Esc)">
      × Exit Compare
    </button>
  </div>

  {#if comparison.pairwise}
    <div class="agreement-banner">
      <span>
        {#if comparison.pairwise.isSubstructure}
          {comparison.pairwise.alignmentLabel ?? (comparison.pairwise.hasSharedTemplate ? 'Motif Kabsch' : 'Substructure Kabsch')} Cα RMSD (<b>{comparison.pairwise.badgeA} ↔ {comparison.pairwise.badgeB}</b> · {comparison.pairwise.alignedResidues} aa):
          <b>{comparison.pairwise.rmsd.toFixed(2)} Å</b>
        {:else}
          Inter-model Kabsch Cα RMSD (<b>{comparison.pairwise.badgeA} ↔ {comparison.pairwise.badgeB}</b>):
          <b>{comparison.pairwise.rmsd.toFixed(2)} Å</b> · TM <b>{comparison.pairwise.tmScore.toFixed(3)}</b>
        {/if}
      </span>
      {#if comparison.allPairwise && comparison.allPairwise.length > 1}
        <div class="extra-pairs">
          {#each comparison.allPairwise.slice(1) as pair (`${pair.badgeA}-${pair.badgeB}`)}
            <span class="pair-chip">
              <b>{pair.badgeA}↔{pair.badgeB}</b>: {pair.rmsd.toFixed(2)} Å{#if !pair.isSubstructure} · TM {pair.tmScore.toFixed(2)}{/if}
            </span>
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
              {:else if !b.has3D}
                <span class="dim">No 3D</span>
              {:else if b.candidateCount > 1}
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
                <code class="mono-sm">{b.sequenceId ?? (b.length ? `${b.length} aa` : '—')}</code>
              {/if}
            </td>
          {/each}
        </tr>

        {#if hasAnyRmsd}
          <tr>
            <td class="m-k">scRMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestRmsd !== null && b.scRmsd === bestRmsd}
                class:pass-val={b.scRmsd !== null && b.scRmsd < branchScRmsdThreshold(b.nodeId)}
              >
                {b.scRmsd !== null ? `${b.scRmsd.toFixed(2)} Å` : '—'}
              </td>
            {/each}
          </tr>
          <tr>
            <td class="m-k">scTM</td>
            {#each branches as b (b.nodeId)}
              <td class:win={bestTm !== null && b.scTm === bestTm}>
                {b.scTm !== null ? b.scTm.toFixed(3) : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyBinderScRmsd}
          <tr>
            <td class="m-k">Binder scRMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestBinderScRmsd !== null && b.binderScRmsd === bestBinderScRmsd}
                class:pass-val={b.binderScRmsd !== null && b.binderScRmsd < branchScRmsdThreshold(b.nodeId)}
              >
                {b.binderScRmsd !== null ? `${b.binderScRmsd.toFixed(2)} Å` : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyTargetAlignedBinderRmsd}
          <tr>
            <td class="m-k">Target-aligned RMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestTargetAlignedBinderRmsd !== null && b.targetAlignedBinderRmsd === bestTargetAlignedBinderRmsd}
              >
                {b.targetAlignedBinderRmsd !== null ? `${b.targetAlignedBinderRmsd.toFixed(2)} Å` : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyRefoldMotifRmsd}
          <tr>
            <td class="m-k">Motif RMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestRefoldMotifRmsd !== null && b.refoldMotifRmsd === bestRefoldMotifRmsd}
                class:pass-val={b.refoldMotifRmsd !== null && b.refoldMotifRmsd < 1.0}
              >
                {b.refoldMotifRmsd !== null ? `${b.refoldMotifRmsd.toFixed(2)} Å` : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyGenMotifRmsd}
          <tr>
            <td class="m-k">Gen Motif RMSD</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestGenMotifRmsd !== null && b.genMotifRmsd === bestGenMotifRmsd}
                class:pass-val={b.genMotifRmsd !== null && b.genMotifRmsd < 1.0}
              >
                {b.genMotifRmsd !== null ? `${b.genMotifRmsd.toFixed(2)} Å` : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyPlddt}
          <tr>
            <td class="m-k">pLDDT</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestPlddt !== null && b.meanPlddt === bestPlddt}
                class:pass-val={b.meanPlddt !== null && b.meanPlddt >= branchPlddtThreshold(b.nodeId)}
              >
                {b.meanPlddt !== null ? b.meanPlddt.toFixed(1) : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyBinderPlddt}
          <tr>
            <td class="m-k">Binder pLDDT</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestBinderPlddt !== null && b.binderPlddt === bestBinderPlddt}
                class:pass-val={b.binderPlddt !== null && b.binderPlddt >= branchPlddtThreshold(b.nodeId)}
              >
                {b.binderPlddt !== null ? b.binderPlddt.toFixed(1) : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyPtm}
          <tr>
            <td class="m-k">pTM</td>
            {#each branches as b (b.nodeId)}
              <td class:win={bestPtm !== null && b.ptm === bestPtm}>
                {b.ptm !== null ? b.ptm.toFixed(3) : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyIptm}
          <tr>
            <td class="m-k">ipTM</td>
            {#each branches as b (b.nodeId)}
              <td class:win={bestIptm !== null && b.iptm === bestIptm}>
                {b.iptm !== null ? b.iptm.toFixed(3) : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyIpae}
          <tr>
            <td class="m-k">iPAE</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestIpae !== null && b.ipae === bestIpae}
                class:pass-val={b.ipae !== null && b.ipae < 10.0}
              >
                {b.ipae !== null ? `${b.ipae.toFixed(1)} Å` : '—'}
              </td>
            {/each}
          </tr>
        {/if}

        {#if hasAnyRmsd}
          <tr>
            <td class="m-k">Pass</td>
            {#each branches as b (b.nodeId)}
              <td>
                {#if b.designable !== null}
                  <span class="verdict" class:pass={b.designable}>
                    {b.designable ? 'PASS' : 'FAIL'}
                  </span>
                {:else}
                  —
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
                {b.length} aa{#if b.geometry} · {b.geometry.rg.toFixed(1)} Å{/if}
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
              <td>{b.foldElapsedMs !== null ? `${(b.foldElapsedMs / 1000).toFixed(1)} s` : '—'}</td>
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
          <PaePlot pae={b.pae!} size={136} />
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
