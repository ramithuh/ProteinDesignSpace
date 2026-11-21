<script lang="ts">
  import PaePlot from '$lib/viewer/PaePlot.svelte';
  import type { ComparisonState } from '$lib/engine/compare';

  interface Props {
    comparison: ComparisonState;
    scRmsdThreshold?: number;
    plddtThreshold?: number;
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
    isRunning = false,
    onSelectBranchCandidate,
    onFocusSingle,
    onRunUpToNode = undefined,
    onExitCompare,
  }: Props = $props();

  let branches = $derived(comparison.branches);

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

  let hasAnyRmsd = $derived(branches.some((b) => b.scRmsd !== null));
  let hasAnyPlddt = $derived(branches.some((b) => b.meanPlddt !== null));
  let hasAnyPtm = $derived(branches.some((b) => b.ptm !== null));
  let hasAnyTime = $derived(branches.some((b) => b.foldElapsedMs !== null));
  let paeBranches = $derived(branches.filter((b) => b.pae !== null));
</script>

<div class="cmp-panel">
  <div class="cmp-head">
    <div class="cmp-title-wrap">
      <span class="cmp-title">COMPARE · {branches.length} blocks</span>
      {#if comparison.forcedSideBySide}
        <span class="cmp-note">Different lengths · Side-by-side</span>
      {/if}
    </div>
    <button type="button" class="exit-btn" onclick={onExitCompare} title="Return to single-block view (Esc)">
      × Exit Compare
    </button>
  </div>

  {#if comparison.pairwise}
    <div class="agreement-banner">
      <span>
        Inter-model Cα RMSD (<b>{comparison.pairwise.badgeA} ↔ {comparison.pairwise.badgeB}</b>):
        <b>{comparison.pairwise.rmsd.toFixed(2)} Å</b> · TM <b>{comparison.pairwise.tmScore.toFixed(3)}</b>
      </span>
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
                class:pass-val={b.scRmsd !== null && b.scRmsd < scRmsdThreshold}
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

        {#if hasAnyPlddt}
          <tr>
            <td class="m-k">pLDDT</td>
            {#each branches as b (b.nodeId)}
              <td
                class:win={bestPlddt !== null && b.meanPlddt === bestPlddt}
                class:pass-val={b.meanPlddt !== null && b.meanPlddt >= plddtThreshold}
              >
                {b.meanPlddt !== null ? b.meanPlddt.toFixed(1) : '—'}
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
