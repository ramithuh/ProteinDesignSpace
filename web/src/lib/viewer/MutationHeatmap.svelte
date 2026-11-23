<script lang="ts">
  import type { MutationMatrixArtifact } from '$lib/engine/types';
  import {
    formatMutationLabel,
    formatMutationMatrixCsv,
    hasMultipleResidueChains,
    rankMutations,
  } from '$lib/protein-ops/bfactor-paint';

  // Precomputed 256-entry diverging color LUT:
  // idx 0 (unfavorable) = deep warm red -> idx 128 (neutral 0) = light slate -> idx 255 (favorable) = deep teal/blue
  const DIVERGING_LUT: string[] = (() => {
    const stops: Array<[number, [number, number, number]]> = [
      [0.0, [153, 27, 27]], // #991b1b deep red (strongly unfavorable)
      [0.2, [220, 38, 38]], // #dc2626 red
      [0.36, [245, 158, 11]], // #f59e0b warm amber
      [0.5, [241, 245, 249]], // #f1f5f9 neutral light slate
      [0.64, [125, 211, 252]], // #7dd3fc sky
      [0.82, [2, 132, 199]], // #0284c7 blue
      [1.0, [15, 92, 115]], // #0f5c73 deep teal/blue (strongly favorable)
    ];
    const lut: string[] = new Array(256);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      let s0 = stops[0];
      let s1 = stops[stops.length - 1];
      for (let k = 0; k < stops.length - 1; k++) {
        if (t >= stops[k][0] && t <= stops[k + 1][0]) {
          s0 = stops[k];
          s1 = stops[k + 1];
          break;
        }
      }
      const span = s1[0] - s0[0] || 1;
      const u = (t - s0[0]) / span;
      const r = Math.round(s0[1][0] + (s1[1][0] - s0[1][0]) * u);
      const g = Math.round(s0[1][1] + (s1[1][1] - s0[1][1]) * u);
      const b = Math.round(s0[1][2] + (s1[1][2] - s0[1][2]) * u);
      lut[i] = `rgb(${r},${g},${b})`;
    }
    return lut;
  })();

  export interface HeatmapHighlightSpec {
    key: string;
    residues: string[];
    sort?: 'best' | 'worst';
    query?: string;
    limit?: number;
  }

  interface Props {
    matrix: MutationMatrixArtifact;
    onDownloadText?: (text: string, filename: string) => void;
    onSelectResidue?: (positionIndex: number, residueId: string) => void;
    topLimit?: number;
    highlightSpec?: HeatmapHighlightSpec | null;
  }

  let {
    matrix,
    onDownloadText = undefined,
    onSelectResidue = undefined,
    topLimit = 10,
    highlightSpec = null,
  }: Props = $props();

  let canvas: HTMLCanvasElement | null = $state(null);
  let scrollEl: HTMLDivElement | null = $state(null);
  let cachedRect: DOMRect | null = null;

  let sortMode = $state<'best' | 'worst'>('best');
  let filterQuery = $state('');
  let rowLimit = $state<number>(10);
  let lastAppliedSpecKey = $state<string | null>(null);
  let lastMatrixId = $state<string>('');

  $effect(() => {
    if (matrix.id !== lastMatrixId) {
      lastMatrixId = matrix.id;
      if (!highlightSpec) {
        sortMode = 'best';
        filterQuery = '';
        rowLimit = topLimit;
      }
      lastAppliedSpecKey = null;
    }
  });

  $effect(() => {
    const spec = highlightSpec;
    if (!spec) {
      lastAppliedSpecKey = null;
      return;
    }
    if (spec.key === lastAppliedSpecKey) return;
    lastAppliedSpecKey = spec.key;
    sortMode = spec.sort ?? 'best';
    filterQuery = spec.query ?? '';
    rowLimit = spec.limit ?? topLimit;
  });

  let hoverCell = $state<{
    col: number;
    row: number;
    label: string;
    isWt: boolean;
    score: number | null;
    residueId: string;
    clientX: number;
    clientY: number;
  } | null>(null);

  let copiedRank = $state<number | null>(null);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;

  let residues = $derived((matrix.residues ?? '').toUpperCase());
  let L = $derived(residues.length);
  let alphabet = $derived((matrix.alphabet || 'ACDEFGHIKLMNPQRSTVWY').toUpperCase());
  let K = $derived(alphabet.length);
  let residueIds = $derived(matrix.residueIds ?? []);
  let multiChain = $derived(hasMultipleResidueChains(residueIds));
  let lowerBetter = $derived(matrix.direction === 'lower-is-better');
  let unitSuffix = $derived(matrix.unit ? ` ${matrix.unit}` : '');

  // Responsive cell sizing so short proteins look spacious and long proteins scroll cleanly
  let cellW = $derived(L <= 45 ? 14 : L <= 80 ? 11 : L <= 150 ? 9 : 7);
  let cellH = 13;
  let axisTopH = 18;
  let canvasCssW = $derived(Math.max(1, L * cellW));
  let canvasCssH = $derived(axisTopH + Math.max(1, K * cellH));

  // Symmetric color scale bound centered at 0 (98th percentile abs so outliers don't wash out signal)
  let colorBound = $derived.by((): number => {
    if (!matrix.values || matrix.values.length === 0 || L === 0 || K === 0) return 2.0;
    const absVals: number[] = [];
    for (let i = 0; i < L; i++) {
      const wt = residues[i];
      const off = i * K;
      for (let j = 0; j < K; j++) {
        const idx = off + j;
        if (idx >= matrix.values.length) break;
        if (alphabet[j] === wt) continue;
        const v = matrix.values[idx];
        if (Number.isFinite(v)) absVals.push(Math.abs(v));
      }
    }
    if (absVals.length === 0) return 2.0;
    absVals.sort((a, b) => a - b);
    const p98 = absVals[Math.min(absVals.length - 1, Math.floor(absVals.length * 0.98))];
    return Math.max(1.0, p98);
  });

  let rankedResult = $derived(
    rankMutations(matrix, {
      sort: sortMode,
      query: filterQuery,
      limit: rowLimit,
    }),
  );
  let topMutations = $derived(rankedResult.items);

  let highlightedCols = $derived.by((): Array<{ col: number; label: string }> => {
    const raw = highlightSpec?.residues;
    if (!raw || raw.length === 0 || L === 0) return [];
    const want = new Set(raw.map((r) => r.trim().toUpperCase()).filter(Boolean));
    const hits: Array<{ col: number; label: string }> = [];
    for (let i = 0; i < L; i++) {
      const resId = (residueIds[i] ?? `A${i + 1}`).toUpperCase();
      const numMatch = /(\d+[A-Z]?)$/.exec(resId);
      const numPart = numMatch ? numMatch[1] : String(i + 1);
      const wt = residues[i] ?? '';
      const wtNum = `${wt}${numPart}`;
      if (want.has(resId) || want.has(wtNum) || want.has(numPart)) {
        hits.push({ col: i, label: wtNum || resId });
      }
    }
    return hits;
  });
  let highlightedColSet = $derived(new Set(highlightedCols.map((h) => h.col)));

  // Auto-scroll heatmap horizontally so the first highlighted walkthrough residue is visible
  $effect(() => {
    const specKey = highlightSpec?.key;
    const cols = highlightedCols;
    const sc = scrollEl;
    if (!specKey || !sc || cols.length === 0) return;
    const firstX = cols[0].col * cellW;
    const targetScroll = Math.max(0, firstX - Math.max(40, sc.clientWidth * 0.25));
    sc.scrollTo({ left: targetScroll, behavior: 'smooth' });
  });

  let summaryStats = $derived.by(() => {
    let finite = 0;
    let favorableCount = 0;
    let bestScore: number | null = null;
    for (let i = 0; i < L; i++) {
      const wt = residues[i];
      const off = i * K;
      for (let j = 0; j < K; j++) {
        const idx = off + j;
        if (idx >= matrix.values.length) break;
        if (alphabet[j] === wt) continue;
        const v = matrix.values[idx];
        if (!Number.isFinite(v)) continue;
        finite++;
        if (lowerBetter ? v < 0 : v > 0) favorableCount++;
        if (
          bestScore === null ||
          (lowerBetter ? v < bestScore : v > bestScore)
        ) {
          bestScore = v;
        }
      }
    }
    return { finite, favorableCount, bestScore };
  });

  function refreshRect() {
    cachedRect = canvas ? canvas.getBoundingClientRect() : null;
  }

  function invalidateRect() {
    cachedRect = null;
  }

  $effect(() => {
    void canvas;
    void scrollEl;
    void canvasCssW;
    void canvasCssH;
    cachedRect = null;
    if (typeof window === 'undefined') return;
    const sc = scrollEl;
    window.addEventListener('resize', invalidateRect);
    window.addEventListener('scroll', invalidateRect, true);
    sc?.addEventListener('scroll', invalidateRect, { passive: true });
    return () => {
      window.removeEventListener('resize', invalidateRect);
      window.removeEventListener('scroll', invalidateRect, true);
      sc?.removeEventListener('scroll', invalidateRect);
    };
  });

  $effect(() => {
    if (!canvas || L <= 0 || K <= 0) return;
    const dpr = typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1;
    canvas.width = Math.round(canvasCssW * dpr);
    canvas.height = Math.round(canvasCssH * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, canvasCssW, canvasCssH);

    // Top X-axis tick labels
    const tickStep = L <= 40 ? 5 : L <= 120 ? 10 : L <= 300 ? 20 : 50;
    ctx.font = '600 9px "JetBrains Mono", ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    const isLight =
      typeof document !== 'undefined' &&
      document.documentElement.getAttribute('data-theme') === 'light';
    ctx.fillStyle = isLight ? '#475569' : '#94a3b8';

    for (let i = 0; i < L; i++) {
      if (i === 0 || (i + 1) % tickStep === 0 || i === L - 1) {
        const x = i * cellW;
        // Avoid overlapping the last label with a nearby tick
        if (i === L - 1 && L > 4 && (L - 1) % tickStep < Math.ceil(tickStep * 0.4)) {
          continue;
        }
        const tickLbl = residueIds[i] ?? String(i + 1);
        ctx.fillText(tickLbl, x + 1, axisTopH * 0.52);
      }
    }

    const favSign = lowerBetter ? -1 : 1;
    const bound = colorBound;
    const missingColor = isLight ? '#e2e8f0' : '#1e293b';

    for (let i = 0; i < L; i++) {
      const x = i * cellW;
      const wt = residues[i];
      const rowOff = i * K;
      for (let j = 0; j < K; j++) {
        const y = axisTopH + j * cellH;
        const idx = rowOff + j;
        const val = idx < matrix.values.length ? matrix.values[idx] : Number.NaN;
        if (!Number.isFinite(val)) {
          ctx.fillStyle = missingColor;
        } else {
          const norm = Math.max(-1, Math.min(1, (val * favSign) / bound));
          const lutIdx = Math.round(((norm + 1) * 0.5) * 255);
          ctx.fillStyle = DIVERGING_LUT[lutIdx];
        }
        ctx.fillRect(x, y, cellW, cellH);

        // Subtle grid line for wide cells
        if (cellW >= 9) {
          ctx.strokeStyle = 'rgba(15, 23, 42, 0.08)';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x + 0.25, y + 0.25, cellW - 0.5, cellH - 0.5);
        }

        // Wild-type residue dot marker
        if (alphabet[j] === wt) {
          const cx = x + cellW * 0.5;
          const cy = y + cellH * 0.5;
          const r = Math.max(2, Math.min(3.1, cellW * 0.26));
          ctx.beginPath();
          ctx.arc(cx, cy, r + 0.8, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
          ctx.fill();
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.fillStyle = '#0f172a';
          ctx.fill();
        }
      }
    }

    ctx.restore();
  });

  function onMove(ev: MouseEvent) {
    if (!canvas || L <= 0 || K <= 0) return;
    const rect = cachedRect ?? (cachedRect = canvas.getBoundingClientRect());
    if (rect.width <= 0 || rect.height <= 0) return;
    const relX = ev.clientX - rect.left;
    const relY = ev.clientY - rect.top;
    if (relY < axisTopH || relY >= canvasCssH || relX < 0 || relX >= canvasCssW) {
      hoverCell = null;
      return;
    }
    const col = Math.min(L - 1, Math.max(0, Math.floor(relX / cellW)));
    const row = Math.min(K - 1, Math.max(0, Math.floor((relY - axisTopH) / cellH)));
    const wt = residues[col] ?? 'X';
    const mut = alphabet[row] ?? 'X';
    const resId = residueIds[col] ?? String(col + 1);
    const idx = col * K + row;
    const raw = idx < matrix.values.length ? matrix.values[idx] : Number.NaN;
    const score = Number.isFinite(raw) ? raw : null;
    hoverCell = {
      col,
      row,
      label: formatMutationLabel(wt, resId, mut, multiChain, col + 1),
      isWt: mut === wt,
      score,
      residueId: resId,
      clientX: ev.clientX,
      clientY: ev.clientY,
    };
  }

  function onClick() {
    if (hoverCell && onSelectResidue) {
      onSelectResidue(hoverCell.col, hoverCell.residueId);
    }
  }

  function formatSignedScore(score: number): string {
    return `${score >= 0 ? '+' : ''}${score.toFixed(2)}${unitSuffix}`;
  }

  function formatHoverText(h: NonNullable<typeof hoverCell>): string {
    const valStr = h.score !== null ? formatSignedScore(h.score) : 'N/A';
    return h.isWt ? `${h.label} (WT) · ${valStr}` : `${h.label} · ${valStr}`;
  }

  function isFavorableScore(score: number): boolean {
    return lowerBetter ? score < 0 : score > 0;
  }

  function triggerDownloadCsv() {
    const csv = formatMutationMatrixCsv(matrix);
    const filename = `${(matrix.id || 'mutation_matrix').replace(/[/:\\*?<>|]+/g, '_')}.csv`;
    if (onDownloadText) {
      onDownloadText(csv, filename);
      return;
    }
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function copyMutantSeq(seq: string, rank: number) {
    if (copyTimer) clearTimeout(copyTimer);
    try {
      await navigator.clipboard?.writeText(seq);
      copiedRank = rank;
      copyTimer = setTimeout(() => {
        copiedRank = null;
      }, 1500);
    } catch {
      copiedRank = null;
    }
  }
</script>

<div class="mh-wrap">
  <div class="mh-head">
    <div class="mh-title-row">
      <span class="mh-title">
        {matrix.label ?? (matrix.metric === 'ddg-kcal-mol' ? 'Mutation ΔΔG Landscape' : 'Mutation Fitness Landscape')}
      </span>
      <span class="mh-chip">{L}×{K}</span>
      <span class="mh-chip">
        {lowerBetter ? 'Lower is better' : 'Higher is better'}
      </span>
      {#if summaryStats.bestScore !== null}
        <span class="mh-chip good">
          Best: {formatSignedScore(summaryStats.bestScore)}
        </span>
      {/if}
      {#if summaryStats.finite > 0}
        <span class="mh-chip">
          {summaryStats.favorableCount}/{summaryStats.finite} favorable
        </span>
      {/if}
    </div>
    <div class="mh-actions">
      <span class="mh-hover-readout" aria-live="polite">
        {#if hoverCell}
          {formatHoverText(hoverCell)}
        {:else}
          Hover cell for score · ● = WT
        {/if}
      </span>
      <button type="button" class="mh-btn" onclick={triggerDownloadCsv}>↓ CSV</button>
    </div>
  </div>

  {#if L > 0 && K > 0 && matrix.values.length > 0}
    <div class="mh-scroll" bind:this={scrollEl}>
      <div class="mh-grid">
        <div class="mh-yaxis" style="padding-top: {axisTopH}px" aria-hidden="true">
          {#each alphabet.split('') as aa, rowIdx (`y_${aa}_${rowIdx}`)}
            <div
              class="mh-ylabel"
              class:active={hoverCell?.row === rowIdx}
              style="height: {cellH}px"
            >
              {aa}
            </div>
          {/each}
        </div>
        <div class="mh-canvas-box" style="width: {canvasCssW}px; height: {canvasCssH}px">
          <canvas
            bind:this={canvas}
            class="mh-canvas"
            style="width: {canvasCssW}px; height: {canvasCssH}px"
            onmouseenter={refreshRect}
            onmousemove={onMove}
            onclick={onClick}
            onmouseleave={() => {
              hoverCell = null;
              cachedRect = null;
            }}
          ></canvas>
          {#each highlightedCols as pin (pin.col)}
            <div
              class="mh-col-pin"
              style="left: {pin.col * cellW}px; top: {axisTopH}px; width: {cellW}px; height: {K * cellH}px"
              aria-hidden="true"
            >
              <span class="mh-col-pin-lbl">{pin.label}</span>
            </div>
          {/each}
          {#if hoverCell}
            <div
              class="mh-highlight"
              style="left: {hoverCell.col * cellW}px; top: {axisTopH + hoverCell.row * cellH}px; width: {cellW}px; height: {cellH}px"
              aria-hidden="true"
            ></div>
          {/if}
        </div>
      </div>
    </div>

    <div class="mh-legend" aria-label="Diverging score color scale centered at zero">
      <div class="mh-leg-scale">
        <span class="mh-leg-end fav">
          Favorable ({lowerBetter ? `−${colorBound.toFixed(1)}` : `+${colorBound.toFixed(1)}`}{unitSuffix})
        </span>
        <span class="mh-leg-bar-wrap" aria-hidden="true">
          <span class="mh-leg-bar"></span>
          <span class="mh-leg-zero">0</span>
        </span>
        <span class="mh-leg-end unfav">
          Unfavorable ({lowerBetter ? `+${colorBound.toFixed(1)}` : `−${colorBound.toFixed(1)}`}{unitSuffix})
        </span>
      </div>
      <span class="mh-wt-key"><span class="mh-wt-dot"></span>WT</span>
    </div>
  {:else}
    <div class="mh-empty">No L×20 mutation matrix values available</div>
  {/if}

  {#if rankedResult.totalAll > 0}
    <div class="mh-top-box">
      <div class="mh-top-head">
        <div class="mh-mode-seg" role="tablist" aria-label="Sort mutants">
          <button
            type="button"
            role="tab"
            class="mh-seg-btn"
            class:active={sortMode === 'best'}
            aria-selected={sortMode === 'best'}
            onclick={() => (sortMode = 'best')}
          >
            {lowerBetter ? 'Stabilizing' : 'Beneficial'}
          </button>
          <button
            type="button"
            role="tab"
            class="mh-seg-btn"
            class:active={sortMode === 'worst'}
            aria-selected={sortMode === 'worst'}
            onclick={() => (sortMode = 'worst')}
          >
            {lowerBetter ? 'Destabilizing' : 'Deleterious'}
          </button>
        </div>

        <div class="mh-filter-wrap">
          <input
            type="text"
            class="mh-filter-input"
            spellcheck="false"
            autocomplete="off"
            placeholder="Filter (e.g. G12, Q61, P19S)"
            aria-label="Filter mutations by residue or mutant name"
            bind:value={filterQuery}
          />
          {#if filterQuery.trim().length > 0}
            <button
              type="button"
              class="mh-filter-clear"
              aria-label="Clear mutation filter"
              onclick={() => (filterQuery = '')}
            >
              ×
            </button>
          {/if}
        </div>

        <div class="mh-limit-seg" role="group" aria-label="Rows shown">
          {#each [10, 20, 50] as lim (lim)}
            <button
              type="button"
              class="mh-lim-btn"
              class:active={rowLimit === lim}
              onclick={() => (rowLimit = lim)}
            >
              {lim}
            </button>
          {/each}
        </div>
      </div>

      <div class="mh-top-meta">
        <span>
          Showing {topMutations.length} of {rankedResult.totalMatched}{filterQuery.trim() ? ` matching "${filterQuery.trim()}"` : ''} ({rankedResult.totalAll} total)
        </span>
        <span class="mh-top-sub">Click row to focus 3D</span>
      </div>

      {#if topMutations.length > 0}
        <div class="mh-table-scroll">
          <table class="mh-table">
            <thead>
              <tr>
                <th title={sortMode === 'best' ? 'Rank from most favorable' : 'Rank from most unfavorable'}>#</th>
                <th>Mutant</th>
                <th>Residue</th>
                <th>{matrix.metric === 'ddg-kcal-mol' ? 'ΔΔG' : 'Score'}</th>
                <th>Sequence</th>
              </tr>
            </thead>
            <tbody>
              {#each topMutations as m (`${sortMode}_${m.rank}_${m.label}`)}
                {@const isPinned = highlightedColSet.has(m.positionIndex)}
                {@const isGood = isFavorableScore(m.score)}
                {@const isBad = lowerBetter ? m.score > 0 : m.score < 0}
                <tr
                  class:is-pinned={isPinned}
                  class:is-clickable={Boolean(onSelectResidue) && m.positionIndex >= 0}
                  onclick={() => {
                    if (onSelectResidue && m.positionIndex >= 0) {
                      onSelectResidue(m.positionIndex, m.residueId);
                    }
                  }}
                >
                  <td class="num">{m.rank}</td>
                  <td><code>{m.label}</code></td>
                  <td class="muted">{m.residueId} ({m.wildType}→{m.mutant})</td>
                  <td class="num" class:good-val={isGood} class:bad-val={!isGood && isBad}>
                    {formatSignedScore(m.score)}
                  </td>
                  <td>
                    <button
                      type="button"
                      class="mh-copy-btn"
                      onclick={(ev) => {
                        ev.stopPropagation();
                        copyMutantSeq(m.mutantSequence, m.rank);
                      }}
                      title={`Copy full ${m.mutantSequence.length}-aa sequence with ${m.label}`}
                    >
                      {copiedRank === m.rank ? '✓ Copied' : 'Copy seq'}
                    </button>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {:else}
        <div class="mh-no-match">
          No mutations match "{filterQuery.trim()}". Try a residue like G12, Q61, or A44.
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .mh-wrap {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .mh-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .mh-title-row {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
  }
  .mh-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .mh-chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    padding: 1px 6px;
    border-radius: 4px;
  }
  .mh-chip.good {
    color: var(--ok-text);
    background: var(--ok-bg);
    border-color: var(--ok-border);
  }
  .mh-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .mh-hover-readout {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }
  .mh-btn {
    background: var(--bg-surface-2);
    color: var(--accent-bright);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    cursor: pointer;
  }
  .mh-btn:hover {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }
  .mh-scroll {
    overflow-x: auto;
    overflow-y: hidden;
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 6px 6px 0;
  }
  .mh-grid {
    display: inline-flex;
    align-items: flex-start;
  }
  .mh-yaxis {
    position: sticky;
    left: 0;
    z-index: 3;
    display: flex;
    flex-direction: column;
    background: var(--bg-canvas);
    border-right: 1px solid var(--border-subtle);
    padding: 0 4px 0 6px;
    user-select: none;
  }
  .mh-ylabel {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 600;
    color: var(--text-dim);
  }
  .mh-ylabel.active {
    color: var(--accent-bright);
    font-weight: 700;
  }
  .mh-canvas-box {
    position: relative;
    flex-shrink: 0;
  }
  .mh-canvas {
    display: block;
    cursor: crosshair;
  }
  .mh-highlight {
    position: absolute;
    pointer-events: none;
    box-sizing: border-box;
    border: 1.5px solid #0f172a;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.9);
  }
  .mh-col-pin {
    position: absolute;
    pointer-events: none;
    box-sizing: border-box;
    border-left: 1.5px solid var(--accent);
    border-right: 1.5px solid var(--accent);
    box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.28);
    z-index: 2;
  }
  .mh-col-pin-lbl {
    position: absolute;
    top: -17px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--accent);
    color: #fff;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    font-weight: 700;
    line-height: 1;
    padding: 2px 4px;
    border-radius: 3px;
    white-space: nowrap;
  }
  .mh-mode-seg,
  .mh-limit-seg {
    display: inline-flex;
    align-items: center;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1px;
    flex-shrink: 0;
  }
  .mh-seg-btn,
  .mh-lim-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 10.5px;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 4px;
    cursor: pointer;
  }
  .mh-lim-btn {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    padding: 2px 5px;
  }
  .mh-seg-btn.active,
  .mh-lim-btn.active {
    background: var(--accent);
    color: #fff;
  }
  .mh-filter-wrap {
    position: relative;
    display: inline-flex;
    align-items: center;
    flex: 1 1 130px;
    min-width: 110px;
    max-width: 220px;
  }
  .mh-filter-input {
    width: 100%;
    background: var(--bg-surface);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 3px 20px 3px 7px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
  }
  .mh-filter-input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .mh-filter-clear {
    position: absolute;
    right: 3px;
    background: none;
    border: none;
    color: var(--text-dim);
    font-size: 13px;
    line-height: 1;
    cursor: pointer;
    padding: 0 3px;
  }
  .mh-filter-clear:hover {
    color: var(--text-primary);
  }
  .mh-top-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 3px 10px 4px;
    font-size: 10px;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border-subtle);
  }
  .mh-table tbody tr.is-clickable {
    cursor: pointer;
  }
  .mh-table tbody tr.is-clickable:hover {
    background: var(--bg-surface-hover);
  }
  .mh-table tbody tr.is-pinned {
    background: color-mix(in srgb, var(--accent) 9%, transparent);
  }
  .mh-table tbody tr.is-pinned td:first-child {
    box-shadow: inset 3px 0 0 var(--accent);
  }
  .mh-table td.bad-val {
    color: var(--err-strong, #dc2626);
    font-weight: 600;
  }
  .mh-no-match {
    padding: 12px 10px;
    font-size: 11px;
    color: var(--text-dim);
    text-align: center;
  }
  .mh-legend {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
  }
  .mh-leg-scale {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex: 1 1 auto;
    min-width: 0;
  }
  .mh-leg-bar-wrap {
    position: relative;
    display: inline-flex;
    align-items: center;
    flex: 1 1 72px;
    min-width: 56px;
    max-width: 180px;
  }
  .mh-leg-bar {
    width: 100%;
    height: 7px;
    border-radius: 999px;
    border: 1px solid var(--border-subtle);
    background: linear-gradient(
      90deg,
      #0f5c73 0%,
      #0284c7 22%,
      #f1f5f9 50%,
      #f59e0b 75%,
      #991b1b 100%
    );
  }
  .mh-leg-zero {
    position: absolute;
    left: 50%;
    top: -1px;
    transform: translateX(-50%);
    font-size: 8px;
    line-height: 9px;
    padding: 0 2px;
    border-radius: 2px;
    background: var(--bg-surface);
    color: var(--text-muted);
    font-weight: 700;
  }
  .mh-leg-end {
    white-space: nowrap;
  }
  .mh-leg-end.fav {
    color: var(--accent-soft);
  }
  .mh-leg-end.unfav {
    color: var(--err-strong);
  }
  .mh-wt-key {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
    flex-shrink: 0;
    color: var(--text-muted);
  }
  .mh-wt-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #0f172a;
    border: 1.5px solid #ffffff;
    box-shadow: 0 0 0 1px var(--border-strong);
  }
  .mh-empty {
    padding: 20px;
    text-align: center;
    font-size: 11.5px;
    color: var(--text-dim);
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .mh-top-box {
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .mh-top-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
  }
  .mh-top-sub {
    font-size: 10px;
    font-weight: 400;
    color: var(--text-dim);
  }
  .mh-table-scroll {
    overflow-x: auto;
  }
  .mh-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
  .mh-table th {
    text-align: left;
    font-size: 9.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    padding: 4px 6px;
    border-bottom: 1px solid var(--border);
  }
  .mh-table td {
    padding: 4px 6px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-body);
  }
  .mh-table tr:last-child td {
    border-bottom: none;
  }
  .mh-table td.num {
    font-variant-numeric: tabular-nums;
  }
  .mh-table td.muted {
    color: var(--text-muted);
  }
  .mh-table td.good-val {
    color: var(--ok-text);
    font-weight: 600;
  }
  .mh-copy-btn {
    background: none;
    border: none;
    color: var(--accent-bright);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    cursor: pointer;
    padding: 0;
  }
  .mh-copy-btn:hover {
    text-decoration: underline;
  }
</style>
