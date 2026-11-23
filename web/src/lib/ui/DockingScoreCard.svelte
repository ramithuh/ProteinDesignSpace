<script lang="ts">
  import type { StructureArtifact } from '$lib/engine/types';

  type DockingMetrics = NonNullable<StructureArtifact['dockingMetrics']>;

  interface Props {
    structure: StructureArtifact;
    dockingMetrics: DockingMetrics;
    onDownloadText: (text: string, filename: string) => void;
    safeFilename: (name: string) => string;
  }

  let { structure, dockingMetrics, onDownloadText, safeFilename }: Props = $props();

  let exportOpen = $state(false);
  let exportWrapEl = $state<HTMLDivElement | null>(null);
  let exportTriggerEl = $state<HTMLButtonElement | null>(null);

  $effect(() => {
    if (!exportOpen || typeof document === 'undefined') return;
    const onPointerDown = (ev: PointerEvent) => {
      if (exportWrapEl && ev.target instanceof Node && !exportWrapEl.contains(ev.target)) {
        exportOpen = false;
      }
    };
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        ev.stopPropagation();
        exportOpen = false;
        exportTriggerEl?.focus();
      }
    };
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown, true);
      document.removeEventListener('keydown', onKeyDown, true);
    };
  });

  function isFiniteNum(v: unknown): v is number {
    return typeof v === 'number' && Number.isFinite(v);
  }

  function csvEscape(raw: unknown): string {
    const str = String(raw ?? '');
    if (/[",\r\n]/.test(str)) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  }

  let primaryDock = $derived(dockingMetrics.rows[0] ?? null);
  let modelEntries = $derived(primaryDock ? Object.entries(primaryDock.models ?? {}) : []);

  let modeLabel = $derived(
    dockingMetrics.taskMode === 'dock'
      ? `Flexible Dock (${dockingMetrics.rows.length} pose${dockingMetrics.rows.length === 1 ? '' : 's'})`
      : dockingMetrics.taskMode === 'pocket-scan'
        ? `Pocket Scan (${dockingMetrics.pocketResiduesScanned ?? 0} res ≤ ${dockingMetrics.pocketRadius ?? 5} Å)`
        : dockingMetrics.taskMode === 'minimize'
          ? 'Minimized Pose'
          : '3-Model Ensemble',
  );

  function downloadDockingCsv() {
    const dm = dockingMetrics;
    const hdr = [
      'rank',
      'pose_id',
      'ligand',
      'mode',
      'precision',
      'heavy_atoms',
      'rotors',
      'cnn_score',
      'cnn_affinity_pk',
      'cnn_vs',
      'vina_kcal_mol',
      'rmsd_to_best_angstrom',
      'initial_cnn_score',
      'initial_vina_kcal_mol',
      'pose_rmsd_angstrom',
    ];
    const csvLines = [
      hdr.join(','),
      ...dm.rows.map((r, idx) =>
        [
          r.rank ?? idx + 1,
          csvEscape(r.poseId),
          csvEscape(r.ligandCode),
          csvEscape(dm.taskMode),
          csvEscape(dm.precision),
          r.heavyAtoms,
          r.rotors,
          isFiniteNum(r.cnnScore) ? r.cnnScore.toFixed(4) : '',
          isFiniteNum(r.cnnAffinity) ? r.cnnAffinity.toFixed(4) : '',
          isFiniteNum(r.cnnVS) ? r.cnnVS.toFixed(4) : '',
          isFiniteNum(r.vinaScore) ? r.vinaScore.toFixed(4) : '',
          isFiniteNum(r.rmsdToBest) ? r.rmsdToBest.toFixed(3) : '',
          isFiniteNum(r.initialCnnScore) ? r.initialCnnScore.toFixed(4) : '',
          isFiniteNum(r.initialVinaScore) ? r.initialVinaScore.toFixed(4) : '',
          isFiniteNum(r.poseRmsd) ? r.poseRmsd.toFixed(3) : '',
        ].join(','),
      ),
    ];
    onDownloadText(csvLines.join('\n') + '\n', safeFilename(`${structure.id}_gnina.csv`));
    exportOpen = false;
  }

  function downloadDockingJson() {
    const payload = JSON.stringify(dockingMetrics, null, 2) + '\n';
    onDownloadText(payload, safeFilename(`${structure.id}_gnina.json`));
    exportOpen = false;
  }
</script>

{#if primaryDock}
  <div class="gnina-card" role="region" aria-label="GNINA 1.3 docking and affinity scores">
    <div class="gnina-head">
      <span class="gnina-title">
        GNINA 1.3 · {primaryDock.ligandCode}
        <span class="gnina-sub">
          ({primaryDock.heavyAtoms} heavy atoms · {primaryDock.rotors} rotatable bond{primaryDock.rotors === 1 ? '' : 's'})
        </span>
      </span>
      <div class="gnina-head-actions">
        <span class="gnina-mode-pill">{modeLabel}</span>
        <div class="export-menu" bind:this={exportWrapEl}>
          <button
            bind:this={exportTriggerEl}
            type="button"
            class="export-trigger"
            aria-expanded={exportOpen}
            aria-controls={`gnina-export-${structure.id}`}
            onclick={() => (exportOpen = !exportOpen)}
          >
            Export ▾
          </button>
          {#if exportOpen}
            <div id={`gnina-export-${structure.id}`} class="export-popover">
              <button type="button" class="export-item" onclick={downloadDockingCsv}>
                ↓ Docking CSV
              </button>
              <button type="button" class="export-item" onclick={downloadDockingJson}>
                ↓ Scores JSON
              </button>
            </div>
          {/if}
        </div>
      </div>
    </div>

    {#if dockingMetrics.taskMode === 'dock' && (dockingMetrics.dockStats || dockingMetrics.box)}
      <div class="gnina-dock-meta">
        {#if dockingMetrics.box}
          <span>Box {dockingMetrics.box.size.map((v) => v.toFixed(1)).join('×')} Å</span>
        {/if}
        {#if dockingMetrics.dockStats}
          <span>
            Grid {(dockingMetrics.dockStats.gridMs / 1000).toFixed(2)}s · Search {(dockingMetrics.dockStats.searchMs / 1000).toFixed(2)}s · CNN {(dockingMetrics.dockStats.rescoreMs / 1000).toFixed(2)}s
          </span>
          <span>{dockingMetrics.dockStats.evaluatedCandidates} candidates</span>
        {/if}
        {#if structure.trajectoryPdb && (structure.trajectoryFrameCount ?? 0) > 1}
          <span>Use 3D frame slider to scrub poses #1–#{structure.trajectoryFrameCount}</span>
        {/if}
      </div>
    {/if}

    <div class="gnina-kpi-grid">
      <div class="gnina-kpi" title="Ensemble pose probability (0–1; higher = more native-like)">
        <span class="gnina-kpi-lbl">CNNscore</span>
        <span class="gnina-kpi-val" class:good-val={primaryDock.cnnScore >= 0.5}>
          {primaryDock.cnnScore.toFixed(3)}
        </span>
        {#if isFiniteNum(primaryDock.initialCnnScore)}
          <span class="gnina-kpi-delta">from {primaryDock.initialCnnScore.toFixed(3)}</span>
        {/if}
      </div>
      <div class="gnina-kpi" title="Predicted binding affinity in pK units (−log10 Kd; higher = tighter)">
        <span class="gnina-kpi-lbl">CNNaffinity</span>
        <span class="gnina-kpi-val">{primaryDock.cnnAffinity.toFixed(2)}&nbsp;<small>pK</small></span>
        {#if isFiniteNum(primaryDock.initialCnnAffinity)}
          <span class="gnina-kpi-delta">from {primaryDock.initialCnnAffinity.toFixed(2)}</span>
        {/if}
      </div>
      <div class="gnina-kpi" title="Virtual screening score = CNNscore × CNNaffinity">
        <span class="gnina-kpi-lbl">CNN_VS</span>
        <span class="gnina-kpi-val" class:good-val={primaryDock.cnnVS >= 3.0}>
          {primaryDock.cnnVS.toFixed(2)}
        </span>
        {#if isFiniteNum(primaryDock.initialCnnVS)}
          <span class="gnina-kpi-delta">from {primaryDock.initialCnnVS.toFixed(2)}</span>
        {/if}
      </div>
      <div class="gnina-kpi" title="Empirical AutoDock Vina interaction energy (kcal/mol; more negative = stronger)">
        <span class="gnina-kpi-lbl">Vina</span>
        <span class="gnina-kpi-val" class:good-val={primaryDock.vinaScore < -5}>
          {primaryDock.vinaScore.toFixed(2)}&nbsp;<small>kcal/mol</small>
        </span>
        {#if isFiniteNum(primaryDock.initialVinaScore)}
          <span class="gnina-kpi-delta">
            from {primaryDock.initialVinaScore.toFixed(2)}{isFiniteNum(primaryDock.poseRmsd) ? ` · Δ${primaryDock.poseRmsd.toFixed(2)} Å` : ''}
          </span>
        {/if}
      </div>
    </div>

    {#if dockingMetrics.rows.length > 1}
      <div class="gnina-table-wrap">
        <table class="gnina-scores">
          <thead>
            <tr>
              {#if dockingMetrics.taskMode === 'dock'}
                <th scope="col">Rank</th>
              {/if}
              <th scope="col">Pose</th>
              <th scope="col">CNNscore</th>
              <th scope="col">CNNaffinity (pK)</th>
              <th scope="col">CNN_VS</th>
              <th scope="col">Vina (kcal/mol)</th>
              {#if dockingMetrics.taskMode === 'dock'}
                <th scope="col">RMSD (Å)</th>
              {/if}
            </tr>
          </thead>
          <tbody>
            {#each dockingMetrics.rows as row, idx (`${row.poseId}_${idx}`)}
              <tr>
                {#if dockingMetrics.taskMode === 'dock'}
                  <td>#{row.rank ?? idx + 1}</td>
                {/if}
                <td><code>{row.poseId}</code></td>
                <td class:good-val={row.cnnScore >= 0.5}>{row.cnnScore.toFixed(3)}</td>
                <td>{row.cnnAffinity.toFixed(2)}</td>
                <td>{row.cnnVS.toFixed(2)}</td>
                <td>{row.vinaScore.toFixed(2)}</td>
                {#if dockingMetrics.taskMode === 'dock'}
                  <td>{isFiniteNum(row.rmsdToBest) ? row.rmsdToBest.toFixed(2) : '—'}</td>
                {/if}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    {#if modelEntries.length > 0 || primaryDock.vinaTerms}
      <details class="gnina-breakdown">
        <summary>Ensemble models ({modelEntries.length}) &amp; Vina energy terms</summary>
        <div class="gnina-breakdown-grid">
          {#if modelEntries.length > 0}
            <table class="gnina-scores gnina-mini-tbl">
              <thead>
                <tr>
                  <th scope="col">Ensemble Model</th>
                  <th scope="col">CNNscore</th>
                  <th scope="col">CNNaffinity (pK)</th>
                </tr>
              </thead>
              <tbody>
                {#each modelEntries as [mName, mScore] (mName)}
                  <tr>
                    <td><code>{mName}</code></td>
                    <td>{mScore.cnnScore.toFixed(3)}</td>
                    <td>{mScore.cnnAffinity.toFixed(2)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}
          {#if primaryDock.vinaTerms}
            <div class="gnina-vina-terms">
              <span><b>gauss1:</b> {primaryDock.vinaTerms.gauss1.toFixed(2)}</span>
              <span><b>gauss2:</b> {primaryDock.vinaTerms.gauss2.toFixed(2)}</span>
              <span><b>repulsion:</b> {primaryDock.vinaTerms.repulsion.toFixed(2)}</span>
              <span><b>hydrophobic:</b> {primaryDock.vinaTerms.hydrophobic.toFixed(2)}</span>
              <span><b>h-bond:</b> {primaryDock.vinaTerms.hydrogen.toFixed(2)}</span>
              <span><b>rotors:</b> {primaryDock.vinaTerms.rotors}</span>
            </div>
          {/if}
        </div>
      </details>
    {/if}
  </div>
{/if}

<style>
  .gnina-card {
    container-type: inline-size;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    border-radius: 8px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .gnina-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }
  .gnina-head-actions {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .gnina-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .gnina-sub {
    font-weight: 400;
    font-size: 11px;
    color: var(--text-muted);
  }
  .gnina-mode-pill {
    font-size: 10.5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    padding: 2px 7px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    background: var(--bg-pill);
  }
  .export-menu {
    position: relative;
  }
  .export-trigger {
    background: var(--bg-surface-2);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 10.5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    cursor: pointer;
  }
  .export-trigger:hover {
    background: var(--bg-surface-hover);
  }
  .export-popover {
    position: absolute;
    right: 0;
    top: calc(100% + 4px);
    z-index: 20;
    min-width: 132px;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    box-shadow: var(--node-shadow);
    padding: 4px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .export-item {
    background: transparent;
    border: none;
    color: var(--text-body);
    text-align: left;
    padding: 5px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    cursor: pointer;
    white-space: nowrap;
  }
  .export-item:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }
  .gnina-dock-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 12px;
    font-size: 10.5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-muted);
  }
  .gnina-kpi-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
  }
  @container (min-width: 520px) {
    .gnina-kpi-grid {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
  }
  .gnina-kpi {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 7px 9px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-surface-2);
    min-width: 0;
  }
  .gnina-kpi-lbl {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
  }
  .gnina-kpi-val {
    font-size: 14px;
    font-weight: 700;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-primary);
    white-space: nowrap;
  }
  .gnina-kpi-val.good-val {
    color: var(--ok-text);
  }
  .gnina-kpi-val small {
    font-size: 10px;
    font-weight: 500;
    color: var(--text-muted);
  }
  .gnina-kpi-delta {
    font-size: 10px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-muted);
  }
  .gnina-table-wrap {
    overflow-x: auto;
  }
  .gnina-scores {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .gnina-scores th,
  .gnina-scores td {
    text-align: left;
    padding: 5px 6px;
    border-bottom: 1px solid var(--border-subtle);
    white-space: nowrap;
  }
  .gnina-scores th {
    color: var(--text-muted);
    font-weight: 600;
    font-size: 10px;
  }
  .gnina-scores td.good-val {
    color: var(--ok-text);
    font-weight: 600;
  }
  .gnina-breakdown summary {
    cursor: pointer;
    font-size: 11px;
    color: var(--accent-soft);
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .gnina-breakdown-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 6px;
  }
  .gnina-vina-terms {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    font-size: 11px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--text-secondary);
  }
  code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
</style>
