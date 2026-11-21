<script lang="ts">
  interface Props {
    pae: Float32Array | number[] | undefined;
    size?: number;
  }

  let { pae, size = 210 }: Props = $props();
  let canvas: HTMLCanvasElement | null = $state(null);
  let hoverInfo = $state<string | null>(null);

  let L = $derived(pae ? Math.floor(Math.sqrt(pae.length)) : 0);
  let compact = $derived(size < 180);

  $effect(() => {
    if (!canvas || !pae || L <= 0) return;
    canvas.width = L;
    canvas.height = L;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(L, L);
    const data = img.data;
    for (let i = 0; i < L * L; i++) {
      // Standard AlphaFold PAE color ramp: 0 Å = deep green (#004d25), 30 Å = near-white (#f8fafc)
      const val = pae[i];
      const t = Number.isFinite(val) ? Math.max(0, Math.min(1, val / 30)) : 1;
      const r = Math.round(0 + t * 242);
      const g = Math.round(77 + t * 172);
      const b = Math.round(37 + t * 213);
      data[i * 4 + 0] = r;
      data[i * 4 + 1] = g;
      data[i * 4 + 2] = b;
      data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  });

  function onMove(ev: MouseEvent) {
    if (!canvas || !pae || L <= 0) return;
    const rect = canvas.getBoundingClientRect();
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
      onmousemove={onMove}
      onmouseleave={() => (hoverInfo = null)}
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
