<script lang="ts">
  import { onMount } from 'svelte';
  import { formatApproxSize, type RunPreflightSummary } from '$lib/engine/run-preflight';
  import { modalFocusTrap } from './modal-focus-trap';

  interface Props {
    preflight: RunPreflightSummary;
    webgpuSupported: boolean;
    webgpuDetail?: string;
    isPartialRun?: boolean;
    /** Non-null when one or more GPU nodes in this run will upload inputs to a self-hosted Host GPU server (H8). */
    remoteHostOrigin?: string | null;
    onConfirm: (rememberChoice: boolean) => void;
    onCancel: () => void;
  }

  let {
    preflight,
    webgpuSupported,
    webgpuDetail = '',
    isPartialRun = false,
    remoteHostOrigin = null,
    onConfirm,
    onCancel,
  }: Props = $props();

  let rememberChoice = $state(true);
  let confirmBtn = $state<HTMLButtonElement | null>(null);
  let remoteMsaList = $derived(preflight.remoteMsaTransfers ?? []);
  let hasRemoteMsa = $derived(remoteMsaList.length > 0);
  let hasRemoteHost = $derived(Boolean(remoteHostOrigin));
  let hasAnyExternalTransfer = $derived(hasRemoteMsa || hasRemoteHost);

  onMount(() => {
    confirmBtn?.focus();
  });

  function handleKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      ev.preventDefault();
      onCancel();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div
  class="modal-backdrop fr-backdrop"
  role="presentation"
  onclick={(ev) => {
    if (ev.target === ev.currentTarget) onCancel();
  }}
>
  <div
    class="modal-card"
    role="dialog"
    aria-modal="true"
    aria-labelledby="first-run-title"
    use:modalFocusTrap
  >
    <div class="modal-hdr">
      <div class="hdr-badge">
        {hasRemoteHost
          ? hasRemoteMsa
            ? 'HOST GPU + EXTERNAL MSA RUN'
            : 'SELF-HOSTED GPU RUN'
          : hasRemoteMsa
            ? 'EXTERNAL MSA + LOCAL GPU RUN'
            : 'LOCAL BROWSER RUN'}
      </div>
      <h2 id="first-run-title">
        {hasRemoteHost
          ? 'Confirm Host GPU upload & run'
          : hasRemoteMsa
            ? 'Confirm external MSA & run'
            : 'Before your first run'}
      </h2>
      <p class="hdr-sub">
        {#if hasRemoteHost}
          Routed GPU nodes will send their parameters and input artifacts to your configured self-hosted server (<code>{remoteHostOrigin}</code>) for host WebGPU execution.
        {:else if hasRemoteMsa}
          This workflow has remote ColabFold MMseqs2 MSA enabled. Selected chain sequences will be sent to api.colabfold.com before local GPU folding.
        {:else if isPartialRun}
          This partial run will download the required model weights and run protein design &amp; folding locally on your device's GPU.
        {:else}
          This workflow downloads model weights once into your browser and runs protein design &amp; folding locally on your device's GPU.
        {/if}
      </p>
    </div>

    {#if hasRemoteHost}
      <div class="gpu-warn" role="alert">
        <strong>Remote Host GPU transfer ({remoteHostOrigin}):</strong>
        <span>
          Inputs for GPU blocks routed to Host GPU will be uploaded to <code>{remoteHostOrigin}</code>. Switch Compute to <code>Browser</code> in the top-right GPU menu to keep 100% of data local.
        </span>
      </div>
    {/if}

    {#if hasRemoteMsa}
      <div class="gpu-warn" role="alert">
        <strong>External sequence transfer (api.colabfold.com):</strong>
        <span>
          {#each remoteMsaList as item, idx (item.nodeId)}
            {idx > 0 ? '; ' : ''}{item.nodeLabel} ({item.mode === 'target-only'
              ? 'target/locked chains only — unannotated chains stay local'
              : 'all chains sent to ColabFold MMseqs2'})
          {/each}. Switch MSA mode to <code>Single-sequence</code> in the block inspector to keep 100% of sequences offline.
        </span>
      </div>
    {/if}

    {#if !webgpuSupported}
      <div class="gpu-warn" role="alert">
        <strong>WebGPU not detected{webgpuDetail ? ` (${webgpuDetail})` : ''}.</strong>
        <span>
          This pipeline requires a WebGPU-capable browser (Chrome, Edge, Arc, or Firefox 145+ on macOS) over HTTPS.
        </span>
      </div>
    {/if}

    <div class="stat-grid">
      <div class="stat-box">
        <span class="stat-lbl">Est. one-time download</span>
        <strong class="stat-val">{formatApproxSize(preflight.totalDownloadMB)}</strong>
      </div>
      <div class="stat-box">
        <span class="stat-lbl">Peak GPU memory</span>
        <strong class="stat-val">
          {formatApproxSize(preflight.peakVramMB)}
        </strong>
      </div>
    </div>

    {#if preflight.models.length > 0}
      <div class="model-list">
        <div class="list-hdr">Models in this run</div>
        {#each preflight.models as m (m.key)}
          <div class="model-row">
            <span class="model-name">
              {m.label}
              {#if m.nonCommercial}
                <span class="nc-tag">Non-Commercial</span>
              {/if}
            </span>
            <span class="model-size">{formatApproxSize(m.approxMB)}</span>
          </div>
        {/each}
      </div>
    {/if}

    <ul class="notes">
      {#if !hasAnyExternalTransfer}
        <li>
          <strong>100% local &amp; private:</strong> Nothing is uploaded — sequences and structures stay in your browser.
        </li>
      {/if}
      <li>
        <strong>Cached locally in your browser:</strong> Weights download once and are reused automatically on future runs.
      </li>
      <li>
        <strong>GPU heads-up:</strong> Initial model setup and structure prediction may take a moment and use GPU memory; weights stay cached for future runs.
      </li>
    </ul>

    {#if !hasAnyExternalTransfer}
      <label class="remember-row">
        <input type="checkbox" bind:checked={rememberChoice} />
        <span>Don't show this again on this browser</span>
      </label>
    {/if}

    <div class="modal-actions">
      <button type="button" class="btn-cancel" onclick={onCancel}>
        Cancel
      </button>
      <button
        type="button"
        class="btn-confirm"
        bind:this={confirmBtn}
        onclick={() => onConfirm(hasAnyExternalTransfer ? false : rememberChoice)}
      >
        {hasRemoteHost
          ? `▶ Send to Host GPU & Run`
          : hasRemoteMsa
            ? `▶ Send MSA & Run (${formatApproxSize(preflight.totalDownloadMB)})`
            : `▶ Download & Run (${formatApproxSize(preflight.totalDownloadMB)})`}
      </button>
    </div>
  </div>
</div>

<style>
  .modal-backdrop {
    position: fixed;
    inset: 0;
    z-index: 230;
    background: rgba(10, 14, 24, 0.62);
    -webkit-backdrop-filter: blur(4px);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    box-sizing: border-box;
  }

  .modal-card {
    width: 100%;
    max-width: min(470px, calc(100vw - 24px));
    max-height: calc(100vh - 24px);
    overflow-y: auto;
    box-sizing: border-box;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
    padding: 20px 22px 18px;
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .modal-hdr {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .hdr-badge {
    align-self: flex-start;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--accent-bright);
    background: var(--bg-pill);
    padding: 2px 8px;
    border-radius: 999px;
  }

  h2 {
    margin: 2px 0 0;
    font-size: 17px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .hdr-sub {
    margin: 0;
    font-size: 12.5px;
    line-height: 1.45;
    color: var(--text-muted);
  }

  .gpu-warn {
    display: flex;
    flex-direction: column;
    gap: 2px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
    border-radius: 8px;
    padding: 9px 11px;
    font-size: 12px;
    line-height: 1.4;
  }

  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .stat-box {
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 11px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .stat-lbl {
    font-size: 10.5px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.35px;
    font-weight: 600;
  }

  .stat-val {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 13.5px;
    color: var(--text-primary);
  }

  .model-list {
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 11px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .list-hdr {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.35px;
    color: var(--text-dim);
  }

  .model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    font-size: 12px;
  }

  .model-name {
    color: var(--text-primary);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }

  .nc-tag {
    font-size: 9.5px;
    font-weight: 600;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    padding: 1px 5px;
    border-radius: 4px;
    white-space: nowrap;
  }

  .model-size {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11.5px;
    color: var(--text-muted);
    white-space: nowrap;
  }

  .notes {
    margin: 0;
    padding-left: 17px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-secondary);
  }

  .notes strong {
    color: var(--text-primary);
  }

  .remember-row {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
  }

  .remember-row input {
    accent-color: var(--accent);
    cursor: pointer;
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding-top: 2px;
  }

  .btn-cancel {
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 8px 14px;
    font-size: 12.5px;
    font-weight: 500;
    cursor: pointer;
  }

  .btn-cancel:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }

  .btn-confirm {
    background: var(--accent);
    color: #ffffff;
    border: none;
    border-radius: 7px;
    padding: 8px 16px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
  }

  .btn-confirm:hover {
    filter: brightness(1.08);
  }

  .btn-confirm:focus-visible {
    outline: 2px solid var(--accent-soft);
    outline-offset: 2px;
  }
</style>
