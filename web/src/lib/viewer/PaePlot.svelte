<script lang="ts">
  // Precomputed 256-entry RGBA color LUT (0 Å = deep green #004d25 -> 31.75 Å = near-white #f2f9fa)
  const MAX_PAE = 31.75;
  const PAE_LUT = (() => {
    const lut = new Uint32Array(256);
    const bytes = new Uint8ClampedArray(lut.buffer);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      bytes[i * 4 + 0] = Math.round(0 + t * 242);
      bytes[i * 4 + 1] = Math.round(77 + t * 172);
      bytes[i * 4 + 2] = Math.round(37 + t * 213);
      bytes[i * 4 + 3] = 255;
    }
    return lut;
  })();

  interface Props {
    pae: Float32Array | number[] | undefined;
    chainLengths?: number[];
    size?: number;
  }

  let { pae, chainLengths = undefined, size = 210 }: Props = $props();
  let canvas: HTMLCanvasElement | null = $state(null);
  let hoverInfo = $state<string | null>(null);
  let cachedRect: DOMRect | null = null;

  let L = $derived(pae ? Math.floor(Math.sqrt(pae.length)) : 0);
  let compact = $derived(size < 180);

  let chainBoundaryPositions = $derived.by((): number[] => {
    if (!chainLengths || chainLengths.length <= 1 || L <= 0) return [];
    const valid = chainLengths.filter((n) => Number.isFinite(n) && n > 0);
    if (valid.length <= 1) return [];
    const out: number[] = [];
    let cum = 0;
    for (let i = 0; i < valid.length - 1; i++) {
      cum += valid[i];
      if (cum > 0 && cum < L) {
        out.push((cum / L) * size);
      }
    }
    return out;
  });

  function refreshRect() {
    cachedRect = canvas ? canvas.getBoundingClientRect() : null;
  }

  function invalidateRect() {
    cachedRect = null;
  }

  $effect(() => {
    void canvas;
    void size;
    cachedRect = null;
    if (typeof window === 'undefined') return;
    window.addEventListener('resize', invalidateRect);
    window.addEventListener('scroll', invalidateRect, true);
    return () => {
      window.removeEventListener('resize', invalidateRect);
      window.removeEventListener('scroll', invalidateRect, true);
    };
  });

  $effect(() => {
    if (!canvas || !pae || L <= 0) return;
    canvas.width = L;
    canvas.height = L;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(L, L);
    const pixels = new Uint32Array(img.data.buffer);
    const total = L * L;
    const scale = 255 / MAX_PAE;
    for (let i = 0; i < total; i++) {
      const val = pae[i];
      const idx = Number.isFinite(val)
        ? val <= 0
          ? 0
          : val >= MAX_PAE
            ? 255
            : (val * scale + 0.5) | 0
        : 255;
      pixels[i] = PAE_LUT[idx];
    }
    ctx.putImageData(img, 0, 0);
  });

  function onMove(ev: MouseEvent) {
    if (!canvas || !pae || L <= 0) return;
    const rect = cachedRect ?? (cachedRect = canvas.getBoundingClientRect());
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = Math.min(L - 1, Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * L)));
    const y = Math.min(L - 1, Math.max(0, Math.floor(((ev.clientY - rect.top) / rect.height) * L)));
    const val = pae[y * L + x];
    if (!Number.isFinite(val)) {
      hoverInfo = null;
      return;
    }
    hoverInfo = compact
      ? `(${y + 1},${x + 1}) ${val.toFixed(1)} Å`
      : `(${y + 1}, ${x + 1}) = ${val.toFixed(1)} Å`;
  }
</script>

<div class="pae-box" style="width: {size}px">
  <div class="pae-head" class:compact>
    <span class="pae-title">{compact && hoverInfo ? 'PAE' : `PAE (${L}×${L})`}</span>
    <span class="hover">{hoverInfo ?? (compact ? '0–31.75 Å' : '0 Å — 31.75 Å')}</span>
  </div>
  {#if pae && L > 0}
    <div class="pae-canvas-wrap" style="width: {size}px; height: {size}px">
      <canvas
        bind:this={canvas}
        class="pae-canvas"
        style="width: {size}px; height: {size}px"
        onmouseenter={refreshRect}
        onmousemove={onMove}
        onmouseleave={() => {
          hoverInfo = null;
          cachedRect = null;
        }}
      ></canvas>
      {#if chainBoundaryPositions.length > 0}
        <svg
          class="pae-dividers"
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          aria-hidden="true"
        >
          {#each chainBoundaryPositions as pos, idx (`b_${idx}`)}
            <line
              x1={pos}
              y1="0"
              x2={pos}
              y2={size}
              stroke="rgba(15, 23, 42, 0.55)"
              stroke-width="2"
              stroke-dasharray="4 3"
            />
            <line
              x1={pos}
              y1="0"
              x2={pos}
              y2={size}
              stroke="rgba(255, 255, 255, 0.9)"
              stroke-width="1"
              stroke-dasharray="4 3"
            />
            <line
              x1="0"
              y1={pos}
              x2={size}
              y2={pos}
              stroke="rgba(15, 23, 42, 0.55)"
              stroke-width="2"
              stroke-dasharray="4 3"
            />
            <line
              x1="0"
              y1={pos}
              x2={size}
              y2={pos}
              stroke="rgba(255, 255, 255, 0.9)"
              stroke-width="1"
              stroke-dasharray="4 3"
            />
          {/each}
        </svg>
      {/if}
    </div>
    <div class="pae-legend" class:compact aria-label="PAE scale 0 Å to 31.75 Å">
      <span>0 Å</span>
      <span class="pae-grad" aria-hidden="true"></span>
      <span>31.75 Å</span>
    </div>
  {:else}
    <div class="pae-empty" style="width: {size}px; height: {size}px">No PAE data</div>
  {/if}
</div>

<style>
  .pae-box {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .pae-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 4px;
    height: 16px;
    line-height: 16px;
    white-space: nowrap;
    overflow: hidden;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .pae-head.compact {
    font-size: 10px;
  }
  .pae-title {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .hover {
    flex-shrink: 0;
    color: var(--text-secondary);
  }
  .pae-canvas-wrap {
    position: relative;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-sizing: border-box;
  }
  .pae-canvas {
    display: block;
    image-rendering: pixelated;
    cursor: crosshair;
  }
  .pae-dividers {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }
  .pae-legend {
    display: flex;
    align-items: center;
    gap: 5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    color: var(--text-dim);
    white-space: nowrap;
  }
  .pae-legend.compact {
    font-size: 8.5px;
    gap: 4px;
  }
  .pae-grad {
    flex: 1;
    height: 5px;
    border-radius: 999px;
    border: 1px solid var(--border-subtle);
    background: linear-gradient(90deg, #004d25 0%, #f2f9fa 100%);
  }
  .pae-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-dim);
    font-size: 11px;
  }
</style>
