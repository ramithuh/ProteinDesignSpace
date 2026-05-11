<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { loadMolstar } from './molstar-loader';

  interface Props {
    pdbText: string | null;
    /** 1-based; defaults to last model (= the final folded structure). */
    modelIndex?: number;
    height?: number;
    label?: string;
    illustrative?: boolean;
  }

  let {
    pdbText,
    modelIndex = undefined,
    height = 380,
    label = undefined,
    illustrative = false,
  }: Props = $props();

  let containerEl: HTMLDivElement | null = $state(null);
  let viewerId = `mol-${Math.random().toString(36).slice(2, 10)}`;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let viewer = $state<any>(null);
  let frameCount = $state(1);
  let userFrameCount = $derived(Math.max(1, frameCount - 1));
  let loadedPdb = $state<string | null>(null);

  let pending = false;
  let pendingFrame: number | null = null;

  onMount(() => {
    // initOrUpdate is triggered by the pdbText effect below
  });

  $effect(() => {
    if (!pdbText || pdbText === loadedPdb) return;
    void loadPdb(pdbText);
  });

  $effect(() => {
    if (!viewer) return;
    const target = modelIndex ?? userFrameCount; // default: final frame
    void setFrame(target);
  });

  $effect(() => {
    if (!viewer) return;
    applyIllustrative(illustrative);
  });

  async function loadPdb(text: string) {
    const molstar = await loadMolstar();
    if (!containerEl) return;
    if (!viewer) {
      viewer = await molstar.Viewer.create(containerEl, {
        layoutIsExpanded: false,
        layoutShowControls: false,
        layoutShowRemoteState: false,
        layoutShowSequence: false,
        layoutShowLog: false,
        viewportShowExpand: false,
        viewportShowSelectionMode: false,
        viewportShowAnimation: false,
      });
    } else {
      await viewer.plugin.clear();
    }

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    try {
      await viewer.loadStructureFromUrl(url, 'pdb', false);
    } finally {
      URL.revokeObjectURL(url);
    }

    await new Promise((r) => setTimeout(r, 200));

    viewer.plugin.canvas3d.setProps({
      cameraFog: { name: 'off', params: {} },
      cameraClipping: { far: false, radius: 0, minNear: 0.1 },
      trackball: { animate: { name: 'rock', params: { speed: 0.1, angle: 10 } } },
    });
    applyIllustrative(illustrative);

    let fc = 1;
    for (const tr of viewer.plugin.managers.structure.hierarchy.current.trajectories) {
      const c = tr.cell?.obj?.data?.frameCount;
      if (c && c > fc) fc = c;
    }
    frameCount = fc;
    loadedPdb = text;

    // Snap to the requested frame (default = final user-visible frame)
    const target = modelIndex ?? Math.max(1, fc - 1);
    await setFrame(target);
  }

  async function setFrame(idx: number) {
    if (pending) {
      pendingFrame = idx;
      return;
    }
    if (!viewer) return;
    pending = true;
    try {
      const plugin = viewer.plugin;
      for (const tr of plugin.managers.structure.hierarchy.current.trajectories) {
        if (!tr.models.length) continue;
        const cell = tr.models[0].cell;
        const fc = tr.cell?.obj?.data?.frameCount;
        if (!fc || fc <= 1) continue;
        await plugin.state.data
          .build()
          .to(cell.transform.ref)
          .update({ modelIndex: idx })
          .commit();
      }
    } finally {
      pending = false;
      if (pendingFrame !== null) {
        const n = pendingFrame;
        pendingFrame = null;
        void setFrame(n);
      }
    }
  }

  function applyIllustrative(on: boolean) {
    if (!viewer) return;
    viewer.plugin.canvas3d.setProps({
      postprocessing: {
        outline: on
          ? {
              name: 'on',
              params: {
                scale: 1,
                threshold: 0.33,
                color: 0x000000,
                includeTransparent: true,
              },
            }
          : { name: 'off', params: {} },
        illustrative: on ? { name: 'on', params: {} } : { name: 'off', params: {} },
      },
    });
  }

  onDestroy(() => {
    try {
      viewer?.dispose?.();
    } catch (_) {
      /* ignore */
    }
    viewer = null;
  });

  // Re-export for parent layout debugging if needed.
  export function getUserFrameCount() {
    return userFrameCount;
  }
</script>

<div class="card" style="height: {height}px">
  {#if label}
    <div class="label">{label}</div>
  {/if}
  <div id={viewerId} bind:this={containerEl} class="viewer"></div>
  {#if !pdbText}
    <div class="placeholder">no structure yet</div>
  {/if}
</div>

<style>
  .card {
    position: relative;
    background: #1a1a1c;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid #2a2a2c;
  }
  .label {
    position: absolute;
    top: 6px;
    left: 8px;
    font-size: 11px;
    color: #aaa;
    background: rgba(15, 15, 16, 0.85);
    padding: 2px 8px;
    border-radius: 3px;
    z-index: 10;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .viewer {
    width: 100%;
    height: 100%;
    background: #0f0f10;
  }
  .placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #444;
    font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    pointer-events: none;
  }
</style>
