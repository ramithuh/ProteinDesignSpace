<script lang="ts">
  import { onMount } from 'svelte';
  import InferenceWorker from '$lib/worker/inference.worker?worker';
  import type { WorkerRequest, WorkerResponse } from '$lib/worker/protocol';
  import type { ScheduleKind } from '$lib/protein-ops/schedules';
  import { coordsToMultiModelPdb } from '$lib/protein-ops/pdb';
  import MolTrajectoryViewer from '$lib/viewer/MolTrajectoryViewer.svelte';

  type Status = 'idle' | 'loading' | 'loaded' | 'sampling' | 'done' | 'error';

  let worker: Worker | null = $state(null);
  let status: Status = $state('idle');
  let loadProgress = $state({ loaded: 0, total: 0 });
  let modelInfo = $state<{
    provider: string;
    sizeBytes: number;
    initLatencyMs: number;
  } | null>(null);
  let errorMsg = $state<string | null>(null);
  let webgpuSupported = $state<boolean | null>(null);

  // Sampling controls
  let length = $state(100);
  let nSamples = $state(4);
  let nfe = $state(100);
  let schedule: ScheduleKind = $state('log');
  let schedulP = $state(2.0);
  let seed = $state(42);

  // Sampling progress / results
  let sampleProgress = $state({ step: 0, total: 0 });
  let stepLatency = $state(0);
  let totalLatencyMs = $state(0);
  let pdbsBySample = $state<string[]>([]);
  let sampleLength = $state(0); // residue count of the active result
  let trajectoryFrameCount = $state(1); // user-facing frame count
  let sliderValue = $state(0); // 0..trajectoryFrameCount-1

  // Viewer controls
  let illustrative = $state(false);
  let showTrajectory = $state(true);

  let nextReqId = 0;

  onMount(() => {
    webgpuSupported = 'gpu' in navigator;
    const w = new InferenceWorker();
    w.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const m = ev.data;
      switch (m.type) {
        case 'progress':
          loadProgress = { loaded: m.loaded, total: m.total };
          break;
        case 'loaded':
          status = 'loaded';
          modelInfo = {
            provider: m.provider,
            sizeBytes: m.modelSizeBytes,
            initLatencyMs: m.initLatencyMs,
          };
          break;
        case 'sampleStep':
          sampleProgress = { step: m.step, total: sampleProgress.total };
          stepLatency = m.latencyMs;
          break;
        case 'sampleDone':
          totalLatencyMs = m.totalLatencyMs;
          // Build per-sample multi-model PDBs from the streamed trajectories.
          const pdbs = m.trajectories.map((traj) => coordsToMultiModelPdb(traj, m.length));
          pdbsBySample = pdbs;
          sampleLength = m.length;
          // Trajectory model layout: model 1 = final (topology), models 2..T+1 = frames 0..T-1
          trajectoryFrameCount = m.trajectories[0]?.length ?? 1;
          sliderValue = trajectoryFrameCount - 1; // start at final
          status = 'done';
          break;
        case 'error':
          status = 'error';
          errorMsg = m.message;
          break;
      }
    };
    worker = w;
    return () => w.terminate();
  });

  function send(req: WorkerRequest) {
    worker?.postMessage(req);
  }

  function loadModel() {
    status = 'loading';
    errorMsg = null;
    loadProgress = { loaded: 0, total: 0 };
    send({ id: nextReqId++, type: 'load', modelUrl: '/models/legacy-teacher/v1/model.onnx' });
  }

  function generate() {
    if (status !== 'loaded' && status !== 'done') return;
    status = 'sampling';
    errorMsg = null;
    sampleProgress = { step: 0, total: nfe };
    pdbsBySample = [];
    totalLatencyMs = 0;
    send({
      id: nextReqId++,
      type: 'sample',
      length,
      nSamples,
      nfe,
      schedule,
      schedulP,
      seed,
    });
  }

  function fmtBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  let loadPct = $derived(
    loadProgress.total ? (100 * loadProgress.loaded) / loadProgress.total : 0,
  );
  let samplePct = $derived(
    sampleProgress.total ? (100 * sampleProgress.step) / sampleProgress.total : 0,
  );
  // Slider value -> modelIndex (model 1 is topology / final, models 2..N+1 are frames 0..N-1)
  let modelIndex = $derived(sliderValue + 2);

  // Convenient handles to the multi-PDB array — 4 slots padded for the grid.
  let gridSlots = $derived<Array<string | null>>([
    pdbsBySample[0] ?? null,
    pdbsBySample[1] ?? null,
    pdbsBySample[2] ?? null,
    pdbsBySample[3] ?? null,
  ]);
</script>

<header>
  <h1>ProteinDesignSpace</h1>
  <p class="subtitle">
    browser playground for protein generative models —
    <span class="muted">V0 single-model (legacy-teacher)</span>
  </p>
</header>

<section class="status">
  <div class="status-item">
    <span class="label">WebGPU</span>
    <code class:ok={webgpuSupported} class:bad={webgpuSupported === false}>
      {#if webgpuSupported === null}…{:else if webgpuSupported}available{:else}n/a{/if}
    </code>
  </div>
  <div class="status-item">
    <span class="label">model</span>
    {#if status === 'idle'}
      <button onclick={loadModel}>load model (855 MB, cached after)</button>
    {:else if status === 'loading'}
      <code>loading {fmtBytes(loadProgress.loaded)} / {fmtBytes(loadProgress.total)} ({loadPct.toFixed(
          0,
        )}%)</code>
    {:else if modelInfo}
      <code class="ok">{modelInfo.provider}</code>
      <span class="dim"
        >·  {fmtBytes(modelInfo.sizeBytes)}  ·  init {modelInfo.initLatencyMs.toFixed(0)} ms</span
      >
    {/if}
  </div>
</section>

{#if status === 'loading' && loadProgress.total > 0}
  <div class="bar"><div style="width: {loadPct}%"></div></div>
{/if}

{#if status === 'loaded' || status === 'sampling' || status === 'done' || status === 'error'}
  <section>
    <h2>generation</h2>
    <div class="controls">
      <label> length <input type="number" min="30" max="256" bind:value={length} /></label>
      <label> samples <input type="number" min="1" max="8" bind:value={nSamples} /></label>
      <label> NFE <input type="number" min="4" max="500" bind:value={nfe} /></label>
      <label>
        schedule
        <select bind:value={schedule}>
          <option value="log">log</option>
          <option value="uniform">uniform</option>
          <option value="power">power</option>
        </select>
      </label>
      <label>
        p <input type="number" min="0.1" max="5" step="0.1" bind:value={schedulP} />
      </label>
      <label> seed <input type="number" min="0" bind:value={seed} /></label>
      <button disabled={status === 'sampling'} onclick={generate}>
        {status === 'sampling' ? 'sampling…' : 'generate'}
      </button>
    </div>

    {#if status === 'sampling'}
      <div class="bar"><div style="width: {samplePct}%"></div></div>
      <div class="muted small">
        step {sampleProgress.step} / {sampleProgress.total}
        · last step {stepLatency.toFixed(0)} ms
      </div>
    {:else if status === 'done'}
      <div class="muted small">
        {sampleProgress.total} steps · {totalLatencyMs.toFixed(0)} ms total ·
        {(totalLatencyMs / sampleProgress.total).toFixed(1)} ms/step ·
        L={sampleLength} · n={pdbsBySample.length}
      </div>
    {/if}
  </section>

  <section>
    <div class="viewer-toolbar">
      <label class="inline">
        <input type="checkbox" bind:checked={illustrative} />
        illustrative
      </label>
      <label class="inline">
        <input type="checkbox" bind:checked={showTrajectory} />
        scrub trajectory
      </label>
      {#if showTrajectory && pdbsBySample.length > 0}
        <input
          type="range"
          min="0"
          max={Math.max(0, trajectoryFrameCount - 1)}
          step="1"
          bind:value={sliderValue}
          class="trajectory-slider"
        />
        <code class="frame-label"
          >frame {sliderValue + 1} / {trajectoryFrameCount}</code
        >
      {/if}
    </div>

    <div class="grid">
      {#each gridSlots as pdb, i (i)}
        <MolTrajectoryViewer
          pdbText={pdb}
          modelIndex={showTrajectory ? modelIndex : undefined}
          height={360}
          label={`sample ${i}`}
          {illustrative}
        />
      {/each}
    </div>
  </section>
{/if}

{#if errorMsg}
  <section class="err">
    <h2>error</h2>
    <pre>{errorMsg}</pre>
  </section>
{/if}

<style>
  header {
    margin-bottom: 20px;
  }
  h1 {
    font-size: 22px;
    margin: 0 0 2px;
    font-weight: 500;
  }
  .subtitle {
    color: #888;
    font-size: 12px;
    margin: 0;
  }
  h2 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #aaa;
    border-bottom: 1px solid #2a2a2c;
    padding-bottom: 6px;
    margin: 20px 0 12px;
    font-weight: 500;
  }
  section {
    margin-bottom: 20px;
  }

  /* Status row */
  .status {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 24px;
    align-items: center;
    padding: 10px 12px;
    background: #1a1a1c;
    border-radius: 4px;
    border: 1px solid #2a2a2c;
    font-size: 12px;
  }
  .status-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
  }
  .label {
    color: #777;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .dim {
    color: #555;
  }

  /* Controls */
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px 16px;
    margin-bottom: 8px;
  }
  .controls label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #aaa;
  }
  .controls input[type='number'],
  .controls select {
    background: #1a1a1c;
    color: #e6e6e6;
    border: 1px solid #333;
    padding: 4px 6px;
    font-family: inherit;
    font-size: 12px;
  }
  .controls input[type='number'] {
    width: 64px;
  }

  /* Buttons */
  button {
    background: #2a2a2c;
    color: #e6e6e6;
    border: 1px solid #444;
    padding: 6px 12px;
    font-size: 12px;
    font-family: inherit;
    cursor: pointer;
    border-radius: 4px;
  }
  button:hover:not(:disabled) {
    background: #353537;
  }
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Viewer toolbar */
  .viewer-toolbar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 4px;
    flex-wrap: wrap;
  }
  .viewer-toolbar .inline {
    font-size: 12px;
    color: #aaa;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .trajectory-slider {
    flex: 1;
    min-width: 200px;
    accent-color: #4f8af6;
  }
  .frame-label {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: #aaa;
    min-width: 110px;
    text-align: right;
  }

  /* Grid */
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  @media (max-width: 720px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }

  /* Misc */
  .bar {
    width: 100%;
    height: 4px;
    background: #1a1a1c;
    margin: 8px 0;
    border-radius: 2px;
    overflow: hidden;
  }
  .bar > div {
    height: 100%;
    background: #4f8af6;
    transition: width 0.1s;
  }
  .small {
    font-size: 11px;
  }
  .muted {
    color: #777;
    font-size: 12px;
  }
  code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
    color: #c0d8ff;
  }
  code.ok {
    color: #a0e0a0;
  }
  code.bad {
    color: #e0a0a0;
  }
  .err {
    color: #ff8888;
  }
  .err pre {
    background: #2a1a1a;
    padding: 10px;
    border-radius: 3px;
    white-space: pre-wrap;
    font-size: 12px;
  }
</style>
