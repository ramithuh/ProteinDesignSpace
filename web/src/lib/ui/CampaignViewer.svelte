<script lang="ts">
  import MolSuperposeViewer from '$lib/viewer/MolSuperposeViewer.svelte';
  import PaePlot from '$lib/viewer/PaePlot.svelte';
  import type { CampaignState } from '$lib/engine/types';

  interface Props {
    campaign: CampaignState;
    theme?: 'dark' | 'light';
    onLoadCampaign: (c: CampaignState) => void;
  }

  let { campaign, theme = 'dark', onLoadCampaign }: Props = $props();
  let selectedIdx = $state(0);

  let activeRound = $derived(campaign.rounds[selectedIdx] ?? campaign.rounds[0] ?? null);

  async function importJson(ev: Event) {
    const input = ev.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as Partial<CampaignState>;
      if (parsed.schema !== 'denovo.campaign/v1' || !Array.isArray(parsed.rounds)) {
        alert('Expected a denovo.campaign/v1 JSON file');
        return;
      }
      const normalized: CampaignState = {
        schema: 'denovo.campaign/v1',
        campaignId: parsed.campaignId || 'imported-campaign',
        title: parsed.title || 'Imported Campaign',
        goal: parsed.goal || '',
        assay: {
          scRmsdThreshold: parsed.assay?.scRmsdThreshold ?? 2.0,
          plddtThreshold: parsed.assay?.plddtThreshold ?? 70,
          targetName: parsed.assay?.targetName,
          offTargetName: parsed.assay?.offTargetName,
        },
        rounds: parsed.rounds,
      };
      onLoadCampaign(normalized);
      selectedIdx = Math.max(0, normalized.rounds.length - 1);
    } catch (err) {
      alert(`Failed to import campaign JSON: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      input.value = '';
    }
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify(campaign, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${campaign.campaignId}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
</script>

<div class="campaign">
  <div class="camp-header">
    <div>
      <h2>{campaign.title}</h2>
      <p class="goal">{campaign.goal}</p>
      <div class="assay">
        Assay: <code>scRMSD &lt; {campaign.assay.scRmsdThreshold} Å</code> ·
        <code>pLDDT &ge; {campaign.assay.plddtThreshold}</code>
        {#if campaign.assay.targetName}
          · Target: <span>{campaign.assay.targetName}</span>
        {/if}
      </div>
    </div>
    <div class="actions">
      <label class="btn">
        Import Campaign JSON
        <input type="file" accept=".json" hidden onchange={importJson} />
      </label>
      <button class="btn" onclick={exportJson}>Export JSON</button>
    </div>
  </div>

  <div class="rounds">
    {#each campaign.rounds as r, i (`${r.round}_${i}`)}
      <button
        class="round-card"
        class:selected={i === selectedIdx}
        onclick={() => (selectedIdx = i)}
      >
        <div class="r-top">
          <span class="r-num">Round {r.round}</span>
          <span class="r-hit" class:good={r.hitRate >= 0.5}>
            {(r.hitRate * 100).toFixed(0)}% pass ({r.nPassed}/{r.nDesigned})
          </span>
        </div>
        <div class="r-label">{r.label}</div>
        <div class="r-metrics">
          <span>scRMSD: <b>{r.bestScRmsd !== null ? `${r.bestScRmsd.toFixed(2)} Å` : '—'}</b></span>
          <span>pLDDT: <b>{r.bestPlddt !== null ? r.bestPlddt.toFixed(1) : '—'}</b></span>
        </div>
        <div class="bar"><div style="width: {Math.round(r.hitRate * 100)}%"></div></div>
      </button>
    {/each}
  </div>

  {#if activeRound}
    <div class="detail-grid">
      <div class="detail-panel">
        <h3>Round {activeRound.round} — {activeRound.label}</h3>
        <dl>
          <dt>Generator</dt>
          <dd>{activeRound.generator}</dd>
          <dt>Refolder</dt>
          <dd>{activeRound.folder}</dd>
          {#if activeRound.conditioningSummary}
            <dt>Conditioning</dt>
            <dd><code>{activeRound.conditioningSummary}</code></dd>
          {/if}
          {#if activeRound.agentNotes}
            <dt>Agent / Recipe Rationale</dt>
            <dd class="notes">{activeRound.agentNotes}</dd>
          {/if}
          {#if activeRound.sequence}
            <dt>Best Sequence</dt>
            <dd class="seq">{activeRound.sequence}</dd>
          {/if}
        </dl>
        {#if activeRound.pae}
          <PaePlot pae={activeRound.pae} size={180} />
        {/if}
      </div>

      <div class="viewer-panel">
        <MolSuperposeViewer
          designedPdb={activeRound.designedPdb ?? null}
          refoldPdb={activeRound.alignedFoldPdb ?? null}
          {theme}
          height={360}
          label={`Round ${activeRound.round} best candidate`}
        />
      </div>
    </div>
  {/if}
</div>

<style>
  .campaign {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .camp-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  h2 {
    margin: 0 0 6px;
    font-size: 16px;
    color: var(--text-primary);
  }
  .goal {
    margin: 0 0 8px;
    font-size: 13px;
    color: var(--text-muted);
  }
  .assay {
    font-size: 12px;
    color: var(--text-secondary);
  }
  .actions {
    display: flex;
    gap: 8px;
    flex-shrink: 0;
  }
  .btn {
    background: var(--bg-surface-2);
    color: var(--text-body);
    border: 1px solid var(--border-strong);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    font-family: inherit;
  }
  .btn:hover {
    background: var(--bg-surface-hover);
  }
  .rounds {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 10px;
  }
  .round-card {
    text-align: left;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    color: var(--text-body);
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .round-card.selected {
    border-color: var(--accent);
    background: var(--bg-surface-selected);
  }
  .r-top {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .r-num {
    color: var(--accent-soft);
    font-weight: 600;
  }
  .r-hit {
    color: #d97706;
  }
  .r-hit.good {
    color: var(--ok-text);
  }
  .r-label {
    font-size: 12px;
    font-weight: 500;
  }
  .r-metrics {
    display: flex;
    gap: 12px;
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
  .bar {
    height: 4px;
    background: var(--border-subtle);
    border-radius: 2px;
    overflow: hidden;
  }
  .bar > div {
    height: 100%;
    background: var(--accent);
  }
  .detail-grid {
    display: grid;
    grid-template-columns: 1fr 1.2fr;
    gap: 14px;
  }
  @media (max-width: 860px) {
    .detail-grid {
      grid-template-columns: 1fr;
    }
  }
  .detail-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  h3 {
    margin: 0 0 10px;
    font-size: 14px;
    color: var(--text-primary);
  }
  dl {
    margin: 0 0 12px;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 6px 12px;
    font-size: 12px;
  }
  dt {
    color: var(--text-dim);
    font-weight: 500;
  }
  dd {
    margin: 0;
    color: var(--text-body);
  }
  .notes {
    color: var(--text-secondary);
    line-height: 1.45;
  }
  .seq {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    word-break: break-all;
    color: var(--accent-soft);
  }
  code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--accent-soft);
  }
</style>
