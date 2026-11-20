<script lang="ts">
  import { onDestroy, untrack } from 'svelte';

  interface Props {
    designedPdb: string | null;
    refoldPdb?: string | null;
    trajectoryPdb?: string | null;
    /** 0-based user frame index (0..T-1). Mapped to Mol* modelIndex = frame + 1. */
    frameIndex?: number;
    showDesigned?: boolean;
    showRefold?: boolean;
    showTrajectory?: boolean;
    illustrative?: boolean;
    theme?: 'dark' | 'light';
    height?: number;
    label?: string;
    onResidueClick?: (info: { chain: string; seqId: number; compId: string }) => void;
  }

  let {
    designedPdb,
    refoldPdb = null,
    trajectoryPdb = null,
    frameIndex = undefined,
    showDesigned = true,
    showRefold = true,
    showTrajectory = false,
    illustrative = false,
    theme = 'dark',
    height = 420,
    label = undefined,
    onResidueClick = undefined,
  }: Props = $props();

  let containerEl: HTMLDivElement | null = $state(null);
  let canvasEl: HTMLCanvasElement | null = $state(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let plugin: any = $state(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let pluginPromise: Promise<any> | null = null;
  let destroyed = false;
  let trajModelRef = $state<string | null>(null);
  let loadSeq = 0;
  let initError = $state<string | null>(null);

  let scrubPending = false;
  let queuedFrame: number | null = null;

  async function ensurePlugin() {
    if (destroyed) return null;
    if (plugin) return plugin;
    if (pluginPromise) return pluginPromise;
    if (!containerEl || !canvasEl) return null;

    const cEl = containerEl;
    const cvEl = canvasEl;

    pluginPromise = (async () => {
      const [{ PluginContext }, { DefaultPluginSpec }] = await Promise.all([
        import('molstar/lib/mol-plugin/context'),
        import('molstar/lib/mol-plugin/spec'),
      ]);
      if (destroyed) return null;

      const spec = DefaultPluginSpec();
      const ctx = new PluginContext(spec);
      await ctx.init();
      if (destroyed) {
        try {
          ctx.dispose();
        } catch {
          /* ignore */
        }
        return null;
      }

      const ok = await ctx.initViewerAsync(cvEl, cEl);
      if (destroyed) {
        try {
          ctx.dispose();
        } catch {
          /* ignore */
        }
        return null;
      }
      if (!ok) {
        try {
          ctx.dispose();
        } catch {
          /* ignore */
        }
        throw new Error('Failed to initialize Mol* WebGL2 canvas');
      }

      ctx.canvas3d?.setProps({
        renderer: { backgroundColor: (theme === 'light' ? 0xf8fafc : 0x0d0e11) as unknown as never },
        cameraFog: { name: 'off', params: {} },
        cameraClipping: { far: false, radius: 0, minNear: 0.1 },
      });

      // Click picking -> emit residue info for MPNN fixed-position / conditioning selection
      const { StructureElement, StructureProperties } = await import('molstar/lib/mol-model/structure');
      if (destroyed) {
        try {
          ctx.dispose();
        } catch {
          /* ignore */
        }
        return null;
      }
      ctx.behaviors.interaction.click.subscribe((ev) => {
        if (!onResidueClick) return;
        const loci = ev.current.loci;
        if (!StructureElement.Loci.is(loci) || loci.elements.length === 0) return;
        const loc = StructureElement.Loci.getFirstLocation(loci);
        if (!loc) return;
        const chain = StructureProperties.chain.auth_asym_id(loc) || 'A';
        const seqId = StructureProperties.residue.auth_seq_id(loc);
        const compId = StructureProperties.atom.label_comp_id(loc) || 'UNK';
        onResidueClick({ chain, seqId, compId });
      });

      plugin = ctx;
      return ctx;
    })();

    try {
      return await pluginPromise;
    } finally {
      pluginPromise = null;
    }
  }

  async function renderScene(
    dPdb: string | null,
    rPdb: string | null,
    tPdb: string | null,
    useTraj: boolean,
    sDesigned: boolean,
    sRefold: boolean,
    fIdx: number | undefined,
  ) {
    const mySeq = ++loadSeq;
    initError = null;
    if (!dPdb && !rPdb) {
      if (plugin) await plugin.clear();
      trajModelRef = null;
      return;
    }

    try {
      const ctx = await ensurePlugin();
      if (destroyed || !ctx || mySeq !== loadSeq) return;

      await ctx.clear();
      trajModelRef = null;
      if (destroyed || mySeq !== loadSeq) return;

      const { Color } = await import('molstar/lib/mol-util/color');
      if (destroyed || mySeq !== loadSeq) return;
      const hasOverlay = Boolean(sDesigned && dPdb && sRefold && rPdb && !useTraj);

      // 1. Designed structure (or multi-model trajectory)
      if (sDesigned && (useTraj && tPdb ? tPdb : dPdb)) {
        const text = useTraj && tPdb ? tPdb : dPdb!;
        const fmt = /^\s*(data_|loop_|_atom_site\.)/m.test(text) ? 'mmcif' : 'pdb';
        const data = await ctx.builders.data.rawData({
          data: text,
          label: useTraj ? 'Trajectory' : 'Designed Backbone',
        });
        const traj = await ctx.builders.structure.parseTrajectory(data, fmt);
        // Fixed off-by-one: MODEL 1 = index 0 (topology), frames 0..T-1 = indices 1..T
        const targetIdx = useTraj && fIdx !== undefined ? fIdx + 1 : 0;
        const model = await ctx.builders.structure.createModel(traj, { modelIndex: targetIdx });
        if (destroyed || mySeq !== loadSeq) return;
        if (useTraj) trajModelRef = model.ref;
        const struct = await ctx.builders.structure.createStructure(model);
        await ctx.builders.structure.representation.addRepresentation(struct, {
          type: 'cartoon',
          color: 'uniform',
          colorParams: { value: Color(hasOverlay ? 0x94a3b8 : 0x60a5fa) },
          typeParams: { alpha: hasOverlay ? 0.45 : 1.0 },
        });
      }

      if (destroyed || mySeq !== loadSeq) return;

      // 2. Kabsch-aligned refolded structure colored by pLDDT (B-factor)
      if (sRefold && rPdb && !useTraj) {
        const data = await ctx.builders.data.rawData({
          data: rPdb,
          label: 'LocalFold Refold (pLDDT)',
        });
        const traj = await ctx.builders.structure.parseTrajectory(data, 'pdb');
        const model = await ctx.builders.structure.createModel(traj, { modelIndex: 0 });
        const struct = await ctx.builders.structure.createStructure(model);
        await ctx.builders.structure.representation.addRepresentation(struct, {
          type: 'cartoon',
          color: 'plddt-confidence',
        });
      }

      if (destroyed || mySeq !== loadSeq) return;
      applyIllustrative(illustrative);
      ctx.canvas3d?.requestCameraReset();
    } catch (err) {
      if (!destroyed && mySeq === loadSeq) {
        initError = err instanceof Error ? err.message : String(err);
      }
    }
  }

  async function scrubFrame(userFrameIdx: number) {
    if (destroyed || !plugin || !trajModelRef) return;
    if (scrubPending) {
      queuedFrame = userFrameIdx;
      return;
    }
    scrubPending = true;
    try {
      const ctx = plugin;
      const ref = trajModelRef;
      if (destroyed || !ctx || !ref) return;
      if (ctx.state?.data?.cells && !ctx.state.data.cells.has(ref)) return;
      await ctx.state.data
        .build()
        .to(ref)
        .update({ modelIndex: userFrameIdx + 1 })
        .commit();
    } catch {
      /* ignore stale trajModelRef while scene reloads */
    } finally {
      scrubPending = false;
      if (!destroyed && queuedFrame !== null) {
        const next = queuedFrame;
        queuedFrame = null;
        void scrubFrame(next);
      } else {
        queuedFrame = null;
      }
    }
  }

  function applyIllustrative(on: boolean) {
    if (!plugin?.canvas3d) return;
    plugin.canvas3d.setProps({
      postprocessing: {
        outline: on
          ? {
              name: 'on',
              params: { scale: 1, threshold: 0.33, color: 0x000000, includeTransparent: true },
            }
          : { name: 'off', params: {} },
        illustrative: on ? { name: 'on', params: {} } : { name: 'off', params: {} },
      },
    });
  }

  // Re-render full scene when structures or visibility toggles change
  $effect(() => {
    const d = designedPdb;
    const r = refoldPdb;
    const t = trajectoryPdb;
    const useT = showTrajectory && Boolean(t);
    const sd = showDesigned;
    const sr = showRefold;
    void renderScene(d, r, t, useT, sd, sr, untrack(() => frameIndex));
  });

  // Fast incremental scrub when only frameIndex changes
  $effect(() => {
    if (showTrajectory && frameIndex !== undefined && trajModelRef) {
      void scrubFrame(frameIndex);
    }
  });

  $effect(() => {
    const on = illustrative;
    if (plugin) applyIllustrative(on);
  });

  $effect(() => {
    const bg = theme === 'light' ? 0xf8fafc : 0x0d0e11;
    if (plugin?.canvas3d) {
      plugin.canvas3d.setProps({
        renderer: { backgroundColor: bg as unknown as never },
      });
    }
  });

  onDestroy(() => {
    destroyed = true;
    loadSeq++;
    try {
      plugin?.dispose();
    } catch {
      /* ignore */
    }
    plugin = null;
  });
</script>

<div class="mol-card" style="height: {height}px">
  {#if label}
    <div class="badge">{label}</div>
  {/if}
  {#if designedPdb && refoldPdb && showDesigned && showRefold && !showTrajectory}
    <div class="legend">
      <span class="swatch designed"></span> Designed
      <span class="swatch plddt"></span> Refold (pLDDT)
    </div>
  {/if}
  <div bind:this={containerEl} class="canvas-wrap">
    <canvas bind:this={canvasEl} class="mol-canvas"></canvas>
  </div>
  {#if !designedPdb && !refoldPdb}
    <div class="empty">Run a pipeline to visualize designed &amp; refolded 3D structures</div>
  {/if}
  {#if initError}
    <div class="err-overlay">Mol* error: {initError}</div>
  {/if}
</div>

<style>
  .mol-card {
    position: relative;
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .canvas-wrap,
  .mol-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  .badge {
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 10;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--overlay-bg);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 4px;
    pointer-events: none;
  }
  .legend {
    position: absolute;
    bottom: 10px;
    left: 10px;
    z-index: 10;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--overlay-bg);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 4px;
    pointer-events: none;
  }
  .swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    display: inline-block;
  }
  .swatch.designed {
    background: rgba(148, 163, 184, 0.65);
    border: 1px solid #94a3b8;
  }
  .swatch.plddt {
    background: linear-gradient(90deg, #ff7d45, #ffdb13, #65cbf3, #0053d6);
  }
  .empty {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-dim);
    font-size: 13px;
    pointer-events: none;
  }
  .err-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    background: var(--err-bg);
    color: var(--err-text);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
  }
</style>
