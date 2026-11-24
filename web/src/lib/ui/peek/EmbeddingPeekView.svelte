<script lang="ts">
  import type { SequenceEmbeddingArtifact } from '$lib/engine/types';

  interface Props {
    artifact: SequenceEmbeddingArtifact;
    formatAdapterName: (id: string) => string;
    safeFilename: (name: string) => string;
    copyText: (text: string, key: string) => void;
    copyBtnLabel: (key: string, idleLabel: string, copiedLabel?: string) => string;
    onDownloadText: (text: string, filename: string) => void;
  }

  let {
    artifact,
    formatAdapterName,
    safeFilename,
    copyText,
    copyBtnLabel,
    onDownloadText,
  }: Props = $props();

  function isFiniteNum(v: unknown): v is number {
    return typeof v === 'number' && Number.isFinite(v);
  }

  function embeddingSummary(vec: Float32Array | undefined) {
    if (!vec || vec.length === 0) return null;
    let sumSq = 0;
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    let count = 0;
    for (let i = 0; i < vec.length; i++) {
      const v = vec[i];
      if (!Number.isFinite(v)) continue;
      sumSq += v * v;
      sum += v;
      if (v < min) min = v;
      if (v > max) max = v;
      count++;
    }
    if (count === 0) return null;
    return {
      count,
      l2Norm: Math.sqrt(sumSq),
      mean: sum / count,
      min,
      max,
      preview: Array.from(vec.slice(0, 12)),
    };
  }

  function formatEmbeddingJson(emb: SequenceEmbeddingArtifact): string {
    const stats = embeddingSummary(emb.pooled);
    const perResNorms =
      emb.perResidue && emb.dim > 0
        ? (() => {
            const L = Math.floor(emb.perResidue.length / emb.dim);
            const norms: number[] = new Array(L);
            for (let i = 0; i < L; i++) {
              let sq = 0;
              const off = i * emb.dim;
              for (let d = 0; d < emb.dim; d++) {
                const v = emb.perResidue[off + d];
                if (Number.isFinite(v)) sq += v * v;
              }
              norms[i] = Number(Math.sqrt(sq).toFixed(6));
            }
            return norms;
          })()
        : undefined;
    return (
      JSON.stringify(
        {
          id: emb.id,
          ...(emb.label ? { label: emb.label } : {}),
          sourceModelId: emb.sourceModelId,
          ...(emb.parentStructureId ? { parentStructureId: emb.parentStructureId } : {}),
          ...(emb.sequenceId ? { sequenceId: emb.sequenceId } : {}),
          sequence: emb.sequence,
          length: emb.sequence.length,
          dim: emb.dim,
          ...(isFiniteNum(emb.pseudoPerplexity) ? { pseudoPerplexity: emb.pseudoPerplexity } : {}),
          ...(stats
            ? {
                pooledNorm: Number(stats.l2Norm.toFixed(6)),
                pooledMean: Number(stats.mean.toFixed(6)),
                pooledRange: [Number(stats.min.toFixed(6)), Number(stats.max.toFixed(6))],
              }
            : {}),
          pooled: Array.from(emb.pooled, (v) => (Number.isFinite(v) ? Number(v.toFixed(6)) : null)),
          ...(perResNorms
            ? {
                perResidueShape: [perResNorms.length, emb.dim],
                perResidueL2Norms: perResNorms,
              }
            : {}),
          ...(emb.provenance ? { provenance: emb.provenance } : {}),
        },
        null,
        2,
      ) + '\n'
    );
  }

  let embStats = $derived(embeddingSummary(artifact.pooled));
</script>

<div class="meta-bar">
  <span>
    <b>{artifact.label ?? 'Sequence Embedding'}</b> · <span>{formatAdapterName(artifact.sourceModelId)}</span>
  </span>
  <div class="dl-links">
    <button type="button" class="link-btn" onclick={() => copyText(artifact.sequence, 'emb-seq')}>
      {copyBtnLabel('emb-seq', 'Copy seq')}
    </button>
    <button
      type="button"
      class="link-btn"
      onclick={() =>
        onDownloadText(formatEmbeddingJson(artifact), safeFilename(`${artifact.id}.embedding.json`))}
    >
      ↓ JSON
    </button>
  </div>
</div>

<div class="stat-chips">
  <span class="stat-chip cond-key">dim = {artifact.dim}</span>
  <span class="stat-chip">L = {artifact.sequence.length} aa</span>
  {#if isFiniteNum(artifact.pseudoPerplexity)}
    <span class="stat-chip cond-key" title="Sequence pseudo-perplexity (lower = more natural)">
      pseudo-PPL = {artifact.pseudoPerplexity.toFixed(2)}
    </span>
  {/if}
  {#if embStats}
    <span class="stat-chip" title="L2 norm of mean-pooled embedding vector">
      ‖pooled‖₂ = {embStats.l2Norm.toFixed(3)}
    </span>
    <span class="stat-chip">mean = {embStats.mean.toFixed(4)}</span>
    <span class="stat-chip">range [{embStats.min.toFixed(3)}, {embStats.max.toFixed(3)}]</span>
  {/if}
  {#if artifact.perResidue}
    <span class="stat-chip">per-residue [{artifact.sequence.length}×{artifact.dim}]</span>
  {/if}
  {#if artifact.provenance?.quantization}
    <span class="stat-chip">{artifact.provenance.quantization.toUpperCase()}</span>
  {/if}
  {#if artifact.provenance?.degradedFallback}
    <span class="warn-chip">⚠ {artifact.provenance.degradedFallback}</span>
  {/if}
  {#if artifact.provenance?.warnings?.length}
    {#each artifact.provenance.warnings as pw (pw)}
      <span class="warn-chip">⚠ {pw}</span>
    {/each}
  {/if}
</div>

{#if embStats && embStats.preview.length > 0}
  <div class="seq-box">
    <div class="seq-head">
      <span>Pooled embedding vector preview (first {embStats.preview.length} of {artifact.dim} dims)</span>
      {#if artifact.sequenceId}
        <code>{artifact.sequenceId}</code>
      {/if}
    </div>
    <div class="seq-text">
      [{embStats.preview.map((v) => (Number.isFinite(v) ? v.toFixed(4) : 'NaN')).join(', ')}{artifact.dim > embStats.preview.length ? ', …' : ''}]
    </div>
  </div>
{/if}

<div class="seq-box">
  <div class="seq-head">
    <span>Embedded sequence ({artifact.sequence.length} aa)</span>
    <button type="button" class="link-btn" onclick={() => copyText(artifact.sequence, 'emb-seq-box')}>
      {copyBtnLabel('emb-seq-box', 'Copy seq')}
    </button>
  </div>
  <div class="seq-text">{artifact.sequence}</div>
</div>

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
  .warn-chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    padding: 2px 7px;
    border-radius: 4px;
  }
  .seq-box {
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 8px 10px;
  }
  .seq-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }
  .seq-text {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-secondary);
    word-break: break-all;
    line-height: 1.5;
  }
  code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
</style>
