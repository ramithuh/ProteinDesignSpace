<script lang="ts">
  import { onMount } from 'svelte';
  import InferenceWorker from '$lib/worker/inference.worker?worker';
  import type { WorkerRequest, WorkerResponse } from '$lib/worker/protocol';

  type Status = 'idle' | 'loading' | 'loaded' | 'running' | 'error';

  let worker: Worker | null = $state(null);
  let status: Status = $state('idle');
  let progress = $state({ loaded: 0, total: 0 });
  let modelInfo = $state<{
    provider: string;
    inputs: string[];
    outputs: string[];
    sizeBytes: number;
    initLatencyMs: number;
  } | null>(null);
  let lastResult = $state<{
    shape: readonly number[];
    latencyMs: number;
    sample: number[];
  } | null>(null);
  let errorMsg = $state<string | null>(null);

  let length = $state(100);
  let seed = $state(42);
  let timestep = $state(0.5);

  let nextReqId = 0;
  let webgpuSupported = $state<boolean | null>(null);

  onMount(() => {
    // Detect WebGPU on the main thread for display purposes (worker does its own check).
    webgpuSupported = 'gpu' in navigator;

    const w = new InferenceWorker();
    w.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const m = ev.data;
      switch (m.type) {
        case 'progress':
          progress = { loaded: m.loaded, total: m.total };
          break;
        case 'loaded':
          status = 'loaded';
          modelInfo = {
            provider: m.provider,
            inputs: m.inputNames,
            outputs: m.outputNames,
            sizeBytes: m.modelSizeBytes,
            initLatencyMs: m.initLatencyMs,
          };
          break;
        case 'forwardDone':
          status = 'loaded';
          lastResult = { shape: m.outputShape, latencyMs: m.latencyMs, sample: m.sample };
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
    progress = { loaded: 0, total: 0 };
    send({ id: nextReqId++, type: 'load', modelUrl: '/models/legacy-teacher/v1/model.onnx' });
  }

  function runForward() {
    status = 'running';
    errorMsg = null;
    send({ id: nextReqId++, type: 'forward', length, seed, t: timestep });
  }

  function fmtBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  let progressPct = $derived(progress.total ? (100 * progress.loaded) / progress.total : 0);
</script>

<h1>ProteinDesignSpace</h1>
<p class="subtitle">
  browser playground for protein generative models —
  <span class="muted">V0 toolchain validator</span>
</p>

<section>
  <h2>environment</h2>
  <div class="info">
    <div>
      <span class="label">WebGPU:</span>
      <code class:ok={webgpuSupported} class:bad={webgpuSupported === false}>
        {#if webgpuSupported === null}detecting…{:else if webgpuSupported}available{:else}not available — will fall back to WASM (slow){/if}
      </code>
    </div>
  </div>
</section>

<section>
  <h2>1. load model</h2>
  <p class="muted">
    First-visit download is ~855 MB (legacy-teacher fp32). Subsequent loads are HTTP-cached.
  </p>
  <button disabled={status === 'loading' || status === 'loaded'} onclick={loadModel}>
    {#if status === 'loading'}
      loading… {fmtBytes(progress.loaded)} / {fmtBytes(progress.total)} ({progressPct.toFixed(0)}%)
    {:else if status === 'loaded'}
      loaded ✓
    {:else}
      load model
    {/if}
  </button>
  {#if status === 'loading' && progress.total > 0}
    <div class="bar"><div style="width: {progressPct}%"></div></div>
  {/if}
  {#if modelInfo}
    <div class="info">
      <div><span class="label">EP:</span> <code class="ok">{modelInfo.provider}</code></div>
      <div><span class="label">size:</span> <code>{fmtBytes(modelInfo.sizeBytes)}</code></div>
      <div>
        <span class="label">session init:</span>
        <code>{modelInfo.initLatencyMs.toFixed(0)} ms</code>
      </div>
      <div><span class="label">inputs:</span> <code>{modelInfo.inputs.join(', ')}</code></div>
      <div><span class="label">outputs:</span> <code>{modelInfo.outputs.join(', ')}</code></div>
    </div>
  {/if}
</section>

{#if status === 'loaded' || status === 'running' || lastResult}
  <section>
    <h2>2. test forward pass</h2>
    <p class="muted">
      Single forward at the chosen <code>t</code>. Random Gaussian input, zero-centered. No sampling
      loop yet.
    </p>
    <div class="row">
      <label>
        length <input type="number" min="30" max="256" step="1" bind:value={length} />
      </label>
      <label>
        seed <input type="number" min="0" bind:value={seed} />
      </label>
      <label>
        t <input type="number" min="0" max="1" step="0.05" bind:value={timestep} />
      </label>
      <button disabled={status === 'running'} onclick={runForward}>
        {status === 'running' ? 'running…' : 'run forward'}
      </button>
    </div>
    {#if lastResult}
      <div class="info">
        <div>
          <span class="label">output shape:</span>
          <code>[{lastResult.shape.join(', ')}]</code>
        </div>
        <div>
          <span class="label">latency:</span>
          <code>{lastResult.latencyMs.toFixed(1)} ms</code>
        </div>
        <div><span class="label">first 3 atoms:</span></div>
        <pre>{lastResult.sample
            .map((v, i) => (i % 3 === 0 && i > 0 ? '\n' : '') + v.toFixed(3).padStart(8))
            .join('  ')}</pre>
      </div>
    {/if}
  </section>
{/if}

{#if errorMsg}
  <section class="err">
    <h2>error</h2>
    <pre>{errorMsg}</pre>
  </section>
{/if}

<style>
  h1 {
    font-size: 24px;
    margin: 0 0 4px;
    font-weight: 500;
  }
  .subtitle {
    color: #888;
    font-size: 13px;
    margin: 0 0 32px;
  }
  h2 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #aaa;
    border-bottom: 1px solid #2a2a2c;
    padding-bottom: 6px;
    margin: 28px 0 12px;
    font-weight: 500;
  }
  section {
    margin-bottom: 28px;
  }
  button {
    background: #2a2a2c;
    color: #e6e6e6;
    border: 1px solid #444;
    padding: 8px 14px;
    font-size: 13px;
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
  .row {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  label {
    font-size: 12px;
    color: #aaa;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  input[type='number'] {
    background: #1a1a1c;
    color: #e6e6e6;
    border: 1px solid #333;
    padding: 4px 6px;
    width: 80px;
    font-family: inherit;
    font-size: 12px;
  }
  .info {
    margin-top: 12px;
    font-size: 12px;
    line-height: 1.8;
  }
  .label {
    color: #888;
    display: inline-block;
    min-width: 110px;
  }
  code,
  pre {
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
    font-size: 12px;
    color: #c0d8ff;
  }
  code.ok {
    color: #a0e0a0;
  }
  code.bad {
    color: #e0a0a0;
  }
  pre {
    background: #1a1a1c;
    padding: 8px 10px;
    border-radius: 3px;
    margin: 4px 0 0;
    overflow-x: auto;
    white-space: pre;
  }
  .bar {
    width: 100%;
    height: 4px;
    background: #1a1a1c;
    margin-top: 8px;
    border-radius: 2px;
    overflow: hidden;
  }
  .bar > div {
    height: 100%;
    background: #4f8af6;
    transition: width 0.1s;
  }
  .err {
    color: #ff8888;
  }
  .err pre {
    background: #2a1a1a;
    white-space: pre-wrap;
  }
  .muted {
    color: #777;
    font-size: 12px;
    margin: 0 0 12px;
  }
</style>
