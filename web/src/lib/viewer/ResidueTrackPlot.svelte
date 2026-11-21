<script lang="ts">
  import type { ResidueTrackArtifact } from '$lib/engine/types';
  import { AA1_TO_AA3 } from '$lib/protein-ops/pdb';
  import { formatResidueTrackCsv } from '$lib/protein-ops/bfactor-paint';

  interface Props {
    track: ResidueTrackArtifact;
    onDownloadText?: (text: string, filename: string) => void;
    onSelectResidue?: (positionIndex: number, residueId: string) => void;
    height?: number;
  }

  let {
    track,
    onDownloadText = undefined,
    onSelectResidue = undefined,
    height = 176,
  }: Props = $props();

  const VB_W = 640;
  const PAD_L = 44;
  const PAD_R = 14;
  const PAD_T = 14;
  const PAD_B = 26;

  let svgEl: SVGSVGElement | null = $state(null);
  let cachedRect: DOMRect | null = null;
  let hoverIdx = $state<number | null>(null);

  const clipUid = `rt-clip-${Math.random().toString(36).slice(2, 9)}`;

  let vbH = $derived(Math.max(120, height));
  let plotW = $derived(VB_W - PAD_L - PAD_R);
  let plotH = $derived(vbH - PAD_T - PAD_B);

  let L = $derived(track.values?.length ?? 0);
  let residues = $derived((track.residues ?? '').toUpperCase());
  let residueIds = $derived(track.residueIds ?? []);
  let hasThreshold = $derived(
    track.threshold !== undefined && Number.isFinite(track.threshold),
  );
  let thresholdVal = $derived(hasThreshold ? (track.threshold as number) : null);

  let stats = $derived.by(() => {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let count = 0;
    let above = 0;
    for (let i = 0; i < L; i++) {
      const v = track.values[i];
      if (!Number.isFinite(v)) continue;
      count++;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      if (thresholdVal !== null && v >= thresholdVal) above++;
    }
    const mean =
      track.summary && Number.isFinite(track.summary.mean)
        ? track.summary.mean
        : count > 0
          ? sum / count
          : null;
    const maxStat =
      track.summary && Number.isFinite(track.summary.max)
        ? track.summary.max
        : count > 0
          ? max
          : null;
    const minStat =
      track.summary?.min !== undefined && Number.isFinite(track.summary.min)
        ? track.summary.min
        : count > 0
          ? min
          : null;
    const fracAbove =
      thresholdVal !== null
        ? track.summary?.fractionAboveThreshold !== undefined &&
          Number.isFinite(track.summary.fractionAboveThreshold)
          ? track.summary.fractionAboveThreshold
          : count > 0
            ? above / count
            : null
        : null;
    const pseudoPerplexity =
      track.summary?.pseudoPerplexity !== undefined &&
      Number.isFinite(track.summary.pseudoPerplexity)
        ? track.summary.pseudoPerplexity
        : null;

    return {
      count,
      above,
      mean,
      max: maxStat,
      min: minStat,
      fracAbove,
      pseudoPerplexity,
      dataMin: count > 0 ? min : 0,
      dataMax: count > 0 ? max : 1,
    };
  });

  let yDomain = $derived.by((): [number, number] => {
    let lo =
      track.range && Number.isFinite(track.range[0]) ? track.range[0] : stats.dataMin;
    let hi =
      track.range && Number.isFinite(track.range[1]) ? track.range[1] : stats.dataMax;

    if (stats.count > 0) {
      if (stats.dataMin < lo) lo = stats.dataMin;
      if (stats.dataMax > hi) hi = stats.dataMax;
    }
    if (thresholdVal !== null) {
      if (thresholdVal < lo) lo = thresholdVal;
      if (thresholdVal > hi) hi = thresholdVal;
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
    if (Math.abs(hi - lo) < 1e-6) {
      return [lo - 0.5, hi + 0.5];
    }
    // Add slight top/bottom headroom when range wasn't explicitly fixed
    if (!track.range) {
      const pad = (hi - lo) * 0.08;
      lo -= pad;
      hi += pad;
    }
    return [lo, hi];
  });

  function xAt(i: number): number {
    if (L <= 1) return PAD_L + plotW * 0.5;
    return PAD_L + (i / (L - 1)) * plotW;
  }

  function yAt(v: number): number {
    const [lo, hi] = yDomain;
    const t = (v - lo) / (hi - lo || 1);
    const clamped = Math.max(0, Math.min(1, t));
    return PAD_T + (1 - clamped) * plotH;
  }

  let baselineY = $derived.by(() => {
    const [lo, hi] = yDomain;
    if (lo <= 0 && hi >= 0) return yAt(0);
    return PAD_T + plotH;
  });

  let thresholdY = $derived(thresholdVal !== null ? yAt(thresholdVal) : null);

  let geometry = $derived.by(() => {
    const lineSegs: string[] = [];
    const areaSegs: string[] = [];
    let curPts: Array<[number, number]> = [];

    const flush = () => {
      if (curPts.length === 0) return;
      const linePath = curPts
        .map(([x, y], idx) => `${idx === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
        .join(' ');
      lineSegs.push(linePath);
      if (curPts.length >= 2) {
        const firstX = curPts[0][0].toFixed(1);
        const lastX = curPts[curPts.length - 1][0].toFixed(1);
        const base = baselineY.toFixed(1);
        areaSegs.push(`${linePath} L${lastX},${base} L${firstX},${base} Z`);
      }
      curPts = [];
    };

    for (let i = 0; i < L; i++) {
      const v = track.values[i];
      if (!Number.isFinite(v)) {
        flush();
        continue;
      }
      curPts.push([xAt(i), yAt(v)]);
    }
    flush();

    return {
      linePath: lineSegs.join(' '),
      areaPath: areaSegs.join(' '),
    };
  });

  let yTicks = $derived.by(() => {
    const [lo, hi] = yDomain;
    const n = 4;
    const out: Array<{ val: number; y: number; label: string }> = [];
    const span = hi - lo;
    const digits = span <= 2 ? 2 : span <= 20 ? 1 : 0;
    for (let k = 0; k <= n; k++) {
      const val = lo + (span * k) / n;
      out.push({ val, y: yAt(val), label: val.toFixed(digits) });
    }
    return out;
  });

  let xTicks = $derived.by(() => {
    if (L <= 0) return [];
    const targetTicks = Math.min(L, 6);
    const out: Array<{ idx: number; x: number; label: string }> = [];
    if (targetTicks === 1) {
      return [{ idx: 0, x: xAt(0), label: residueIds[0] ?? '1' }];
    }
    for (let k = 0; k < targetTicks; k++) {
      const idx = Math.round(((L - 1) * k) / (targetTicks - 1));
      out.push({ idx, x: xAt(idx), label: residueIds[idx] ?? String(idx + 1) });
    }
    return out;
  });

  let hoverDetail = $derived.by(() => {
    if (hoverIdx === null || hoverIdx < 0 || hoverIdx >= L) return null;
    const i = hoverIdx;
    const val = track.values[i];
    const resId = residueIds[i] ?? String(i + 1);
    const aa1 = residues[i] ?? 'X';
    const aa3 = AA1_TO_AA3[aa1] ?? 'UNK';
    const finite = Number.isFinite(val);
    const valStr = finite
      ? `${val.toFixed(Math.abs(yDomain[1] - yDomain[0]) <= 2 ? 2 : 2)}${track.unit ? ` ${track.unit}` : ''}`
      : 'N/A';
    const text = `${resId} · ${aa3} · ${valStr}`;
    const x = xAt(i);
    const y = finite ? yAt(val) : PAD_T + plotH * 0.5;
    const isAbove = finite && thresholdVal !== null && val >= thresholdVal;
    return { i, resId, aa1, aa3, val, finite, text, x, y, isAbove };
  });

  function refreshRect() {
    cachedRect = svgEl ? svgEl.getBoundingClientRect() : null;
  }

  function invalidateRect() {
    cachedRect = null;
  }

  $effect(() => {
    void svgEl;
    void vbH;
    cachedRect = null;
    if (typeof window === 'undefined') return;
    window.addEventListener('resize', invalidateRect);
    window.addEventListener('scroll', invalidateRect, true);
    return () => {
      window.removeEventListener('resize', invalidateRect);
      window.removeEventListener('scroll', invalidateRect, true);
    };
  });

  function onMove(ev: MouseEvent) {
    if (!svgEl || L <= 0) return;
    const rect = cachedRect ?? (cachedRect = svgEl.getBoundingClientRect());
    if (rect.width <= 0) return;
    const svgX = ((ev.clientX - rect.left) / rect.width) * VB_W;
    if (svgX < PAD_L - 6 || svgX > VB_W - PAD_R + 6) {
      hoverIdx = null;
      return;
    }
    const frac = Math.max(0, Math.min(1, (svgX - PAD_L) / plotW));
    hoverIdx = L === 1 ? 0 : Math.round(frac * (L - 1));
  }

  function onClick() {
    if (hoverDetail && onSelectResidue) {
      onSelectResidue(hoverDetail.i, hoverDetail.resId);
    }
  }

  function fmtStat(v: number | null, digits = 2): string {
    if (v === null || !Number.isFinite(v)) return '—';
    return `${v.toFixed(digits)}${track.unit ? ` ${track.unit}` : ''}`;
  }

  function triggerDownloadCsv() {
    const csv = formatResidueTrackCsv(track);
    const filename = `${(track.id || 'residue_track').replace(/[/:\\*?<>|]+/g, '_')}.csv`;
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
</script>

<div class="rt-wrap">
  <div class="rt-head">
    <div class="rt-title-row">
      <span class="rt-title">{track.quantity}</span>
      {#if track.unit}
        <span class="rt-chip">{track.unit}</span>
      {/if}
      <span class="rt-chip">L = {L}</span>
    </div>
    <div class="rt-actions">
      <span class="rt-hover-readout" aria-live="polite">
        {#if hoverDetail}
          {hoverDetail.text}
        {:else}
          Hover plot for residue value
        {/if}
      </span>
      <button type="button" class="rt-btn" onclick={triggerDownloadCsv}>↓ CSV</button>
    </div>
  </div>

  <div class="rt-stats">
    <span class="rt-chip">mean {fmtStat(stats.mean)}</span>
    <span class="rt-chip">max {fmtStat(stats.max)}</span>
    {#if stats.fracAbove !== null && thresholdVal !== null}
      <span class="rt-chip hi" title={`${stats.above} of ${stats.count} residues ≥ ${thresholdVal}`}>
        ≥{thresholdVal}: {(stats.fracAbove * 100).toFixed(1)}% ({stats.above}/{stats.count})
      </span>
    {/if}
    {#if stats.pseudoPerplexity !== null}
      <span class="rt-chip accent">pseudo-PPL {stats.pseudoPerplexity.toFixed(2)}</span>
    {/if}
  </div>

  {#if L > 0}
    <div class="rt-chart-box">
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <svg
        bind:this={svgEl}
        class="rt-svg"
        viewBox={`0 0 ${VB_W} ${vbH}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${track.quantity} per-residue track across ${L} residues`}
        onmouseenter={refreshRect}
        onmousemove={onMove}
        onclick={onClick}
        onmouseleave={() => {
          hoverIdx = null;
          cachedRect = null;
        }}
      >
        <defs>
          <linearGradient id={`${clipUid}-base`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.32" />
            <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.03" />
          </linearGradient>
          <linearGradient id={`${clipUid}-hi`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#f97316" stop-opacity="0.52" />
            <stop offset="100%" stop-color="#ef4444" stop-opacity="0.16" />
          </linearGradient>
          {#if thresholdY !== null}
            <clipPath id={`${clipUid}-above`}>
              <rect
                x={PAD_L}
                y={PAD_T}
                width={plotW}
                height={Math.max(0, thresholdY - PAD_T)}
              />
            </clipPath>
          {/if}
        </defs>

        <!-- Horizontal gridlines & Y ticks -->
        {#each yTicks as tk, idx (`yt_${idx}`)}
          <line
            x1={PAD_L}
            y1={tk.y}
            x2={VB_W - PAD_R}
            y2={tk.y}
            stroke="var(--border-subtle)"
            stroke-width="1"
          />
          <text
            x={PAD_L - 6}
            y={tk.y + 3}
            text-anchor="end"
            class="rt-axis-txt"
          >
            {tk.label}
          </text>
        {/each}

        <!-- Base area + line -->
        {#if geometry.areaPath}
          <path d={geometry.areaPath} fill={`url(#${clipUid}-base)`} />
        {/if}
        {#if geometry.linePath}
          <path
            d={geometry.linePath}
            fill="none"
            stroke="#3b82f6"
            stroke-width="1.6"
            stroke-linejoin="round"
            stroke-linecap="round"
          />
        {/if}

        <!-- Shaded high-signal segments above threshold -->
        {#if thresholdY !== null && geometry.areaPath}
          <g clip-path={`url(#${clipUid}-above)`}>
            <path d={geometry.areaPath} fill={`url(#${clipUid}-hi)`} />
            <path
              d={geometry.linePath}
              fill="none"
              stroke="#f97316"
              stroke-width="2"
              stroke-linejoin="round"
              stroke-linecap="round"
            />
          </g>
          <line
            x1={PAD_L}
            y1={thresholdY}
            x2={VB_W - PAD_R}
            y2={thresholdY}
            stroke="#f59e0b"
            stroke-width="1.25"
            stroke-dasharray="5 3"
          />
          <text
            x={VB_W - PAD_R - 2}
            y={Math.max(PAD_T + 9, thresholdY - 4)}
            text-anchor="end"
            class="rt-thr-txt"
          >
            thr {thresholdVal}
          </text>
        {/if}

        <!-- Single-residue dot fallback -->
        {#if L === 1 && Number.isFinite(track.values[0])}
          <circle
            cx={xAt(0)}
            cy={yAt(track.values[0])}
            r="3.5"
            fill="#3b82f6"
          />
        {/if}

        <!-- X-axis baseline & residue labels -->
        <line
          x1={PAD_L}
          y1={PAD_T + plotH}
          x2={VB_W - PAD_R}
          y2={PAD_T + plotH}
          stroke="var(--border)"
          stroke-width="1"
        />
        {#each xTicks as xt, idx (`xt_${idx}`)}
          <line
            x1={xt.x}
            y1={PAD_T + plotH}
            x2={xt.x}
            y2={PAD_T + plotH + 4}
            stroke="var(--border-strong)"
            stroke-width="1"
          />
          <text
            x={xt.x}
            y={vbH - 6}
            text-anchor={idx === 0 ? 'start' : idx === xTicks.length - 1 ? 'end' : 'middle'}
            class="rt-axis-txt"
          >
            {xt.label}
          </text>
        {/each}

        <!-- Interactive hover crosshair & marker -->
        {#if hoverDetail}
          <line
            x1={hoverDetail.x}
            y1={PAD_T}
            x2={hoverDetail.x}
            y2={PAD_T + plotH}
            stroke="var(--text-muted)"
            stroke-width="1"
            stroke-dasharray="3 2"
          />
          {#if hoverDetail.finite}
            <circle
              cx={hoverDetail.x}
              cy={hoverDetail.y}
              r="4"
              fill={hoverDetail.isAbove ? '#f97316' : '#3b82f6'}
              stroke="var(--bg-surface)"
              stroke-width="1.5"
            />
          {/if}
        {/if}
      </svg>

      {#if hoverDetail}
        {@const pctX = Math.max(8, Math.min(92, (hoverDetail.x / VB_W) * 100))}
        <div
          class="rt-tooltip"
          class:above={hoverDetail.isAbove}
          style="left: {pctX}%"
          aria-hidden="true"
        >
          {hoverDetail.text}
        </div>
      {/if}
    </div>
  {:else}
    <div class="rt-empty">No per-residue track values available</div>
  {/if}
</div>

<style>
  .rt-wrap {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }
  .rt-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .rt-title-row {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
  }
  .rt-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .rt-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .rt-hover-readout {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }
  .rt-btn {
    background: var(--bg-surface-2);
    color: var(--accent-bright);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    cursor: pointer;
  }
  .rt-btn:hover {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }
  .rt-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .rt-chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    padding: 2px 7px;
    border-radius: 4px;
  }
  .rt-chip.hi {
    color: var(--warn-text);
    background: var(--warn-bg);
    border-color: var(--warn-border);
  }
  .rt-chip.accent {
    color: var(--accent-soft);
    background: var(--bg-pill);
    border-color: var(--border-strong);
  }
  .rt-chart-box {
    position: relative;
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  .rt-svg {
    display: block;
    width: 100%;
    height: auto;
    cursor: crosshair;
  }
  .rt-axis-txt {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    fill: var(--text-dim);
  }
  .rt-thr-txt {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    font-weight: 600;
    fill: #f59e0b;
  }
  .rt-tooltip {
    position: absolute;
    top: 6px;
    transform: translateX(-50%);
    pointer-events: none;
    white-space: nowrap;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-primary);
    background: color-mix(in srgb, var(--bg-surface) 94%, transparent);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 2px 7px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
  }
  .rt-tooltip.above {
    border-color: #f97316;
  }
  .rt-empty {
    padding: 20px;
    text-align: center;
    font-size: 11.5px;
    color: var(--text-dim);
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 6px;
  }
</style>
