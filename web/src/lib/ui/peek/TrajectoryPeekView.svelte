<script lang="ts">
  import {
    buildTrajectoryMultiModelPdb,
    inferTrajectoryResidueCount,
    previewTextLines,
  } from '$lib/engine/peek';
  import type { CollectionKind, StructureArtifact, TrajectoryFramesArtifact } from '$lib/engine/types';

  interface Props {
    artifact: TrajectoryFramesArtifact;
    contextStructure?: StructureArtifact | null;
    formatAdapterName: (id: string) => string;
    formatCollectionKind: (kind?: CollectionKind) => string | null;
    safeFilename: (name: string) => string;
    onDownloadText: (text: string, filename: string) => void;
  }

  let {
    artifact,
    contextStructure = null,
    formatAdapterName,
    formatCollectionKind,
    safeFilename,
    onDownloadText,
  }: Props = $props();

  let rawPreviewOpen = $state(false);

  function isFiniteNum(v: unknown): v is number {
    return typeof v === 'number' && Number.isFinite(v);
  }

  let frames = $derived(artifact.frames ?? []);
  let effectiveFrameCount = $derived(frames.length > 0 ? frames.length : artifact.frameCount);
  let trajColLabel = $derived(formatCollectionKind(artifact.collectionKind ?? artifact.ordering));
  let firstStep = $derived(frames[0]?.timeStep);
  let lastStep = $derived(frames[frames.length - 1]?.timeStep);
  let trajResCount = $derived(inferTrajectoryResidueCount(artifact, contextStructure));
  let trajPdbText = $derived(buildTrajectoryMultiModelPdb(artifact, contextStructure));
  let trajRawPrev = $derived(
    rawPreviewOpen && trajPdbText ? previewTextLines(trajPdbText, 36) : null,
  );
  let previewFrames = $derived(
    frames.length <= 10
      ? frames
      : [...frames.slice(0, 4), frames[Math.floor(frames.length / 2)], ...frames.slice(-4)],
  );
</script>

<div class="meta-bar">
  <span>
    <b>{artifact.label ?? 'Diffusion Trajectory'}</b>
    {#if artifact.sourceModelId}
      · <span>{formatAdapterName(artifact.sourceModelId)}</span>
    {/if}
  </span>
  <div class="dl-links">
    {#if trajPdbText}
      <button
        type="button"
        class="link-btn"
        onclick={() => onDownloadText(trajPdbText, safeFilename(`${artifact.id}_trajectory.pdb`))}
      >
        ↓ Trajectory PDB
      </button>
    {/if}
    {#if artifact.cifText}
      <button
        type="button"
        class="link-btn"
        onclick={() =>
          onDownloadText(artifact.cifText!, safeFilename(`${artifact.id}_trajectory.cif`))}
      >
        ↓ Trajectory mmCIF
      </button>
    {/if}
  </div>
</div>

<div class="stat-chips">
  <span class="stat-chip cond-key">
    {effectiveFrameCount} {effectiveFrameCount === 1 ? 'frame' : 'frames'}
  </span>
  {#if trajResCount !== null}
    <span class="stat-chip">L = {trajResCount} aa</span>
  {/if}
  {#if trajColLabel}
    <span class="stat-chip">{trajColLabel}</span>
  {/if}
  {#if isFiniteNum(firstStep) && isFiniteNum(lastStep)}
    <span class="stat-chip">t: {firstStep} → {lastStep}</span>
  {/if}
  {#if !artifact.pdbText && frames.some((f) => f.caCoords && !f.pdbText)}
    <span class="stat-chip">Cα trace</span>
  {/if}
</div>

{#if previewFrames.length > 0}
  <table class="scores">
    <thead>
      <tr>
        <th scope="col">Frame</th>
        <th scope="col">Key</th>
        <th scope="col">Step (t)</th>
        <th scope="col">Residues</th>
        {#if frames.some((f) => isFiniteNum(f.confidence))}
          <th scope="col">Conf</th>
        {/if}
      </tr>
    </thead>
    <tbody>
      {#each previewFrames as fr (`${fr.key}_${fr.frameIndex}`)}
        {@const frameRes = fr.caCoords ? Math.floor(fr.caCoords.length / 3) : trajResCount}
        <tr>
          <td>#{fr.frameIndex + 1}</td>
          <td><code>{fr.key}</code></td>
          <td>{isFiniteNum(fr.timeStep) ? fr.timeStep : '—'}</td>
          <td>{frameRes !== null ? `${frameRes} aa` : '—'}</td>
          {#if frames.some((f) => isFiniteNum(f.confidence))}
            <td>{isFiniteNum(fr.confidence) ? fr.confidence.toFixed(1) : '—'}</td>
          {/if}
        </tr>
      {/each}
    </tbody>
  </table>
{/if}

{#if trajPdbText}
  <details class="raw-details" bind:open={rawPreviewOpen}>
    <summary>
      Raw Trajectory PDB preview
      {#if trajRawPrev}
        ({trajRawPrev.totalLines} lines)
      {:else}
        ({(trajPdbText.length / 1024).toFixed(1)} KB)
      {/if}
    </summary>
    {#if trajRawPrev}
      <pre class="raw-pre">{trajRawPrev.preview}{trajRawPrev.truncated ? `\n… (${trajRawPrev.totalLines - 36} more lines)` : ''}</pre>
    {/if}
  </details>
{/if}

<style>
  .meta-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--text-secondary);
  }
  .dl-links {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .link-btn {
    background: transparent;
    border: none;
    color: var(--accent-soft);
    font-size: 11px;
    cursor: pointer;
    padding: 0;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .link-btn:hover {
    text-decoration: underline;
  }
  .stat-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 6px;
  }
  .stat-chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    padding: 2px 7px;
    border-radius: 4px;
  }
  .stat-chip.cond-key {
    color: var(--accent-soft);
    background: var(--bg-pill);
    border-color: var(--border-strong);
  }
  .scores {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .scores th,
  .scores td {
    text-align: left;
    padding: 5px 6px;
    border-bottom: 1px solid var(--border-subtle);
  }
  .scores th {
    color: var(--text-muted);
    font-weight: 600;
    font-size: 10px;
  }
  .raw-details summary {
    cursor: pointer;
    font-size: 11px;
    color: var(--accent-soft);
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .raw-pre {
    margin: 0;
    padding: 10px;
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-secondary);
    overflow: auto;
    max-height: 240px;
    line-height: 1.45;
    white-space: pre;
  }
  code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
</style>
