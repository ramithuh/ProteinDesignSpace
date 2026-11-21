<script lang="ts">
  // Precomputed 256-entry RGBA color LUT (0 Å = deep green #004d25 -> 30 Å = near-white #f2f9fa)
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
    size?: number;
  }

  let { pae, size = 210 }: Props = $props();
  let canvas: HTMLCanvasElement | null = $state(null);
  let hoverInfo = $state<string | null>(null);
  let cachedRect: DOMRect | null = null;

  let L = $derived(pae ? Math.floor(Math.sqrt(pae.length)) : 0);
  let compact = $derived(size < 180);

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
    const scale = 255 / 30;
    for (let i = 0; i < total; i++) {
      const val = pae[i];
      const idx = Number.isFinite(val)
        ? val <= 0
          ? 0
          : val >= 30
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
    <span class="hover">{hoverInfo ?? '0–30 Å'}</span>
  </div>
  {#if pae && L > 0}
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
  .pae-canvas {
    border: 1px solid var(--border);
    border-radius: 4px;
    image-rendering: pixelated;
    cursor: crosshair;
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
