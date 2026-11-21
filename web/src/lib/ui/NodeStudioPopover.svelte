<script lang="ts">
  import { onDestroy, type Snippet } from 'svelte';
  import { ADAPTER_REGISTRY } from '$lib/engine/registry';
  import { getVisibleParamFields } from '$lib/engine/param-schema';
  import { STAGE_COLORS } from '$lib/engine/graph-utils';
  import type { NodeInputPreview, StudioBrushRole } from '$lib/engine/studio-types';
  import type { BioCompatibilityFix, WorkflowNode } from '$lib/engine/types';
  import AnnotatedFastaView, { type StudioBrushOption } from './AnnotatedFastaView.svelte';

  interface Props {
    node: WorkflowNode;
    allNodes: WorkflowNode[];
    preview: NodeInputPreview | null;
    activeTab: 'input' | 'controls' | 'output';
    hoveredToken?: string | null;
    isRunning?: boolean;
    onSelectTab: (tab: 'input' | 'controls' | 'output') => void;
    onSelectNode: (nodeId: string) => void;
    onClose: () => void;
    onRunUpstreamAndThis: () => void;
    onUpdateNodeParams: (nodeId: string, nextParams: Record<string, unknown>) => void;
    onApplyFix?: (fix: BioCompatibilityFix) => void;
    onHoverToken?: (token: string | null) => void;
    onFocusToken?: (token: string) => void;
    controlsSlot?: Snippet;
    outputSlot?: Snippet;
  }

  let {
    node,
    allNodes,
    preview,
    activeTab,
    hoveredToken = null,
    isRunning = false,
    onSelectTab,
    onSelectNode,
    onClose,
    onRunUpstreamAndThis,
    onUpdateNodeParams,
    onApplyFix = undefined,
    onHoverToken = undefined,
    onFocusToken = undefined,
    controlsSlot = undefined,
    outputSlot = undefined,
  }: Props = $props();

  let manifest = $derived(ADAPTER_REGISTRY[node.adapterId]);
  let currentIndex = $derived(allNodes.findIndex((n) => n.id === node.id));
  let prevNode = $derived(currentIndex > 0 ? allNodes[currentIndex - 1] : null);
  let nextNode = $derived(
    currentIndex >= 0 && currentIndex < allNodes.length - 1 ? allNodes[currentIndex + 1] : null,
  );

  // Declaratively derive interactive paint brushes from ADAPTER_REGISTRY[node.adapterId].paramSchema
  let availableBrushes = $derived.by((): StudioBrushOption[] => {
    if (!manifest) return [];
    const visible = getVisibleParamFields(manifest, node.params);
    return visible
      .filter((f): f is typeof f & { studioRole: StudioBrushRole } => Boolean(f.studioRole))
      .map((f) => ({
        fieldKey: f.key,
        role: f.studioRole,
        label: f.label,
      }));
  });

  let selectedBrushKey = $state<string | null>(null);
  let activeBrush = $derived.by((): StudioBrushOption | null => {
    if (availableBrushes.length === 0) return null;
    if (selectedBrushKey) {
      const found = availableBrushes.find((b) => b.fieldKey === selectedBrushKey);
      if (found) return found;
    }
    return availableBrushes[0];
  });

  // Reset brush choice when switching nodes
  $effect(() => {
    void node.id;
    selectedBrushKey = null;
  });

  let showOmitStrip = $derived(
    Boolean(manifest?.paramSchema?.some((f) => f.key === 'omitAminoAcids')),
  );

  let currentBrushValue = $derived(
    activeBrush ? String(node.params?.[activeBrush.fieldKey] ?? '') : '',
  );

  let omitAminoAcidsValue = $derived(String(node.params?.omitAminoAcids ?? ''));

  let issueCount = $derived(preview?.issues.length ?? 0);
  let errorIssueCount = $derived(
    preview?.issues.filter((i) => i.severity === 'error').length ?? 0,
  );

  // Resolve missing upstream node IDs to friendly display names
  let missingUpstreamLabels = $derived.by((): string[] => {
    if (!preview?.missingUpstreamNodeIds?.length) return [];
    return preview.missingUpstreamNodeIds.map((uid) => {
      const found = allNodes.find((n) => n.id === uid);
      return found?.label || ADAPTER_REGISTRY[found?.adapterId ?? '']?.shortName || uid;
    });
  });

  // Format compiledSummary entries into compact key-value badges
  interface SpecBadge {
    key: string;
    label: string;
    value: string;
    tone: 'neutral' | 'warn' | 'info' | 'ok';
  }

  let specBadges = $derived.by((): SpecBadge[] => {
    const out: SpecBadge[] = [];
    if (!preview) return out;

    if (preview.previewStructureMeta?.atomGranularity === 'ca-only') {
      out.push({
        key: '__ca_only',
        label: 'Granularity',
        value: 'Cα-only backbone',
        tone: 'warn',
      });
    }

    const rolePairs = preview.chains
      .filter((c) => c.roleTag)
      .map((c) => `${c.chainId}${c.labelChainId && c.labelChainId !== c.chainId ? `→${c.labelChainId}` : ''}:${c.roleTag}`);
    if (rolePairs.length > 0) {
      out.push({
        key: '__chain_roles',
        label: 'Chain roles',
        value: rolePairs.join(' · '),
        tone: 'info',
      });
    }

    if (preview.entities.filteredBuffers.length > 0) {
      out.push({
        key: '__buffers',
        label: 'Buffers stripped',
        value: preview.entities.filteredBuffers.join(', '),
        tone: 'neutral',
      });
    }

    const summary = preview.payload.compiledSummary ?? {};
    for (const [k, rawVal] of Object.entries(summary)) {
      if (rawVal === '' || rawVal === null || rawVal === undefined) continue;
      if (Array.isArray(rawVal) && rawVal.length === 0) continue;
      const valStr = Array.isArray(rawVal)
        ? rawVal.join(', ')
        : typeof rawVal === 'boolean'
          ? rawVal
            ? 'Yes'
            : 'No'
          : String(rawVal);
      const lowerKey = k.toLowerCase();
      const lowerVal = valStr.toLowerCase();
      let tone: SpecBadge['tone'] = 'neutral';
      if (
        lowerKey.includes('warn') ||
        lowerKey.includes('apo') ||
        lowerVal.includes('cα-only') ||
        lowerVal.includes('ca-only') ||
        lowerVal.includes('apo')
      ) {
        tone = 'warn';
      } else if (lowerKey.includes('chain') || lowerKey.includes('msa') || lowerKey.includes('model')) {
        tone = 'info';
      }
      const prettyLabel = k
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .replace(/[_-]+/g, ' ')
        .replace(/^\w/, (c) => c.toUpperCase());
      out.push({ key: k, label: prettyLabel, value: valStr, tone });
    }
    return out;
  });

  function handleUpdateField(fieldKey: string, nextValue: string) {
    const next: Record<string, unknown> = { ...node.params, [fieldKey]: nextValue };
    if (
      (node.adapterId === 'rfd4-proteina' || node.adapterId === 'genie3') &&
      ['contig', 'lockChain', 'hotspots', 'motifSpan'].includes(fieldKey)
    ) {
      next.userEditedTask = true;
    }
    onUpdateNodeParams(node.id, next);
  }

  function handleUpdateOmitAa(nextOmit: string) {
    onUpdateNodeParams(node.id, { ...node.params, omitAminoAcids: nextOmit });
  }

  let copiedRaw = $state(false);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;
  async function copyRawSpec() {
    if (!preview?.payload.rawSpecText) return;
    if (copyTimer) clearTimeout(copyTimer);
    try {
      await navigator.clipboard.writeText(preview.payload.rawSpecText);
      copiedRaw = true;
      copyTimer = setTimeout(() => {
        copiedRaw = false;
      }, 1500);
    } catch {
      copiedRaw = false;
    }
  }

  function nodeDisplayName(n: WorkflowNode): string {
    return n.label || ADAPTER_REGISTRY[n.adapterId]?.shortName || n.adapterId;
  }

  function nodeStageColor(n: WorkflowNode): string {
    const stage = ADAPTER_REGISTRY[n.adapterId]?.stage ?? 'source';
    return STAGE_COLORS[stage] ?? '#64748b';
  }

  onDestroy(() => {
    if (copyTimer) clearTimeout(copyTimer);
  });
</script>

<section class="studio-shell" aria-label="Node Studio Inspector">
  <!-- Studio Header: Breadcrumb Switcher + Budgets + Run/Close Actions -->
  <header class="studio-head">
    <div class="head-top">
      <div class="nav-strip" role="navigation" aria-label="Workflow node switcher">
        <button
          type="button"
          class="nav-arrow"
          disabled={!prevNode}
          onclick={() => prevNode && onSelectNode(prevNode.id)}
          title={prevNode ? `Previous: ${nodeDisplayName(prevNode)}` : 'First block'}
          aria-label="Previous block"
        >
          ◀
        </button>
        <div class="node-chips">
          {#each allNodes as n (n.id)}
            <button
              type="button"
              class="node-chip"
              class:active={n.id === node.id}
              onclick={() => onSelectNode(n.id)}
              title={`Inspect ${nodeDisplayName(n)}`}
            >
              <span class="dot" style="background: {nodeStageColor(n)}"></span>
              <span>{nodeDisplayName(n)}</span>
            </button>
          {/each}
        </div>
        <button
          type="button"
          class="nav-arrow"
          disabled={!nextNode}
          onclick={() => nextNode && onSelectNode(nextNode.id)}
          title={nextNode ? `Next: ${nodeDisplayName(nextNode)}` : 'Last block'}
          aria-label="Next block"
        >
          ▶
        </button>
      </div>

      <div class="head-actions">
        <button
          type="button"
          class="run-btn"
          disabled={isRunning}
          onclick={onRunUpstreamAndThis}
          title="Run upstream blocks and execute this block"
        >
          {isRunning ? 'Running…' : '▶ Run to Here'}
        </button>
        <button
          type="button"
          class="close-btn"
          onclick={onClose}
          title="Close Node Studio (Esc)"
          aria-label="Close Node Studio"
        >
          ✕ Close <kbd>Esc</kbd>
        </button>
      </div>
    </div>

    <!-- Budget Telemetry Strip -->
    {#if preview?.budget}
      {@const b = preview.budget}
      <div class="budget-row">
        <span class="b-pill" title="Total residues entering this block">
          <b>{b.totalResidues}</b> aa
          {#if b.designedResidues > 0 || b.fixedResidues > 0}
            <span class="b-sub">({b.designedResidues} design · {b.fixedResidues} fixed)</span>
          {/if}
        </span>
        <span class="b-pill" title="Estimated model token budget">
          <b>~{b.estimatedTokens}</b> tok
        </span>
        {#if b.nodeVramMB > 0 || b.subgraphPeakVramMB > 0}
          <span
            class="b-pill"
            title={`This block: ~${b.nodeVramMB} MB VRAM · Subgraph peak: ~${b.subgraphPeakVramMB} MB VRAM`}
          >
            VRAM <b>~{b.nodeVramMB || b.subgraphPeakVramMB} MB</b>
          </span>
        {/if}
        {#if b.nodeWeightsMB > 0 || b.subgraphDownloadMB > 0}
          <span
            class="b-pill"
            title={`This block weights: ~${b.nodeWeightsMB} MB · Subgraph total: ~${b.subgraphDownloadMB} MB`}
          >
            Weights <b>~{b.nodeWeightsMB} MB</b>
            {#if b.subgraphDownloadMB > b.nodeWeightsMB}
              <span class="b-sub">(path ~{b.subgraphDownloadMB} MB)</span>
            {/if}
          </span>
        {/if}
      </div>
    {/if}

    <!-- Tab Bar -->
    <div class="tab-bar" role="tablist" aria-label="Node Studio sections">
      <button
        type="button"
        role="tab"
        aria-selected={activeTab === 'input'}
        class:active={activeTab === 'input'}
        onclick={() => onSelectTab('input')}
      >
        What Model Sees
        {#if issueCount > 0}
          <span class="tab-badge" class:is-err={errorIssueCount > 0}>{issueCount}</span>
        {/if}
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={activeTab === 'controls'}
        class:active={activeTab === 'controls'}
        onclick={() => onSelectTab('controls')}
      >
        Controls
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={activeTab === 'output'}
        class:active={activeTab === 'output'}
        onclick={() => onSelectTab('output')}
      >
        Output / Intermediates
      </button>
    </div>
  </header>

  <!-- Studio Body -->
  <div class="studio-body">
    {#if activeTab === 'input'}
      <div class="input-tab">
        {#if !preview}
          <div class="pending-banner">
            <div class="pb-text">
              <strong>Input preview unavailable</strong>
              <span>Run upstream blocks to materialize inputs for {nodeDisplayName(node)}.</span>
            </div>
            <button
              type="button"
              class="cta-btn"
              disabled={isRunning}
              onclick={onRunUpstreamAndThis}
            >
              ▶ Run Upstream to Materialize
            </button>
          </div>
        {:else}
          {#if preview.availability === 'pending-upstream'}
            <div class="pending-banner">
              <div class="pb-text">
                <strong>Waiting on upstream output</strong>
                <span>
                  {#if missingUpstreamLabels.length > 0}
                    Run {missingUpstreamLabels.join(', ')} to materialize live coordinates &amp; sequences for this block.
                  {:else}
                    Upstream blocks have not run yet; showing static parameter preview.
                  {/if}
                </span>
              </div>
              <button
                type="button"
                class="cta-btn"
                disabled={isRunning}
                onclick={onRunUpstreamAndThis}
              >
                ▶ Run Upstream to Materialize
              </button>
            </div>
          {/if}

          <!-- Compiled Spec Headline & Compact Key-Value Badges -->
          <div class="spec-card">
            <div class="spec-head">
              <div class="spec-title">{preview.payload.headline}</div>
              <span class="fmt-tag">{preview.payload.format}</span>
            </div>
            {#if specBadges.length > 0}
              <div class="spec-badges">
                {#each specBadges as sb (sb.key)}
                  <span class="kv-badge tone-{sb.tone}">
                    <span class="kv-k">{sb.label}:</span>
                    <span class="kv-v">{sb.value}</span>
                  </span>
                {/each}
              </div>
            {/if}
          </div>

          <!-- Merged Issues with 1-Click Fix Buttons -->
          {#if preview.issues.length > 0}
            <div class="issues-list" role="region" aria-label="Compatibility and compiler checks">
              {#each preview.issues as issue, i (`${issue.source}-${i}-${issue.message}`)}
                <div class="issue-row sev-{issue.severity}">
                  <div class="issue-main">
                    <span class="issue-src">{issue.source}</span>
                    <span class="issue-msg">{issue.message}</span>
                  </div>
                  {#if issue.fix && onApplyFix}
                    <button
                      type="button"
                      class="fix-btn"
                      onclick={() => issue.fix && onApplyFix(issue.fix)}
                    >
                      {issue.fix.label}
                    </button>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}

          <!-- Interactive Annotated FASTA & Constraint Paint Strip -->
          <AnnotatedFastaView
            chains={preview.chains}
            fastaEntries={preview.fastaEntries}
            entities={preview.entities}
            {activeBrush}
            {availableBrushes}
            currentParamValue={currentBrushValue}
            omitAminoAcids={omitAminoAcidsValue}
            {showOmitStrip}
            {hoveredToken}
            atomGranularity={preview.previewStructureMeta?.atomGranularity}
            onSelectBrush={(key) => (selectedBrushKey = key)}
            onUpdateField={handleUpdateField}
            onUpdateOmitAa={handleUpdateOmitAa}
            {onHoverToken}
            {onFocusToken}
          />

          <!-- Collapsible Raw Compiled Spec Drawer -->
          {#if preview.payload.rawSpecText}
            <details class="raw-drawer">
              <summary>
                <span>Raw Compiled Spec ({preview.payload.format})</span>
                <button
                  type="button"
                  class="raw-copy-btn"
                  onclick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    void copyRawSpec();
                  }}
                >
                  {copiedRaw ? '✓ Copied' : 'Copy'}
                </button>
              </summary>
              <pre class="raw-pre">{preview.payload.rawSpecText}</pre>
            </details>
          {/if}
        {/if}
      </div>
    {:else if activeTab === 'controls'}
      <div class="slot-pane">
        {#if controlsSlot}
          {@render controlsSlot()}
        {:else}
          <div class="slot-empty">No controls available for {nodeDisplayName(node)}.</div>
        {/if}
      </div>
    {:else}
      <div class="slot-pane">
        {#if outputSlot}
          {@render outputSlot()}
        {:else}
          <div class="slot-empty">Run this block to inspect outputs and intermediate artifacts.</div>
        {/if}
      </div>
    {/if}
  </div>
</section>

<style>
  .studio-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    color: var(--text-body);
  }
  .studio-head {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 12px 0;
    background: var(--bg-surface-2);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .head-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }
  .nav-strip {
    display: flex;
    align-items: center;
    gap: 4px;
    min-width: 0;
    flex: 1;
  }
  .nav-arrow {
    width: 24px;
    height: 24px;
    padding: 0;
    border-radius: 5px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-secondary);
    font-size: 10px;
    cursor: pointer;
    flex-shrink: 0;
  }
  .nav-arrow:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--text-primary);
  }
  .nav-arrow:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .node-chips {
    display: flex;
    align-items: center;
    gap: 4px;
    overflow-x: auto;
    min-width: 0;
    padding: 1px 0;
    scrollbar-width: thin;
  }
  .node-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    font-family: inherit;
  }
  .node-chip:hover {
    border-color: var(--border-strong);
    color: var(--text-primary);
  }
  .node-chip.active {
    border-color: var(--accent);
    background: var(--bg-surface-selected);
    color: var(--text-primary);
    font-weight: 600;
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .head-actions {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }
  .run-btn {
    background: var(--accent);
    color: #fff;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  .run-btn:hover:not(:disabled) {
    filter: brightness(1.08);
  }
  .run-btn:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
  .close-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    white-space: nowrap;
  }
  .close-btn:hover {
    border-color: var(--accent);
    color: var(--text-primary);
  }
  .close-btn kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    padding: 0 4px;
    border-radius: 3px;
    border: 1px solid var(--border);
    background: var(--bg-surface-2);
    color: var(--text-dim);
  }
  .budget-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    align-items: center;
  }
  .b-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    padding: 2px 8px;
  }
  .b-pill b {
    color: var(--text-primary);
  }
  .b-sub {
    color: var(--text-dim);
    margin-left: 3px;
  }
  .tab-bar {
    display: flex;
    align-items: center;
    gap: 2px;
    margin-top: 2px;
  }
  .tab-bar button {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 600;
    padding: 7px 12px;
    cursor: pointer;
    font-family: inherit;
    white-space: nowrap;
  }
  .tab-bar button:hover {
    color: var(--text-primary);
  }
  .tab-bar button.active {
    color: var(--accent-bright);
    border-bottom-color: var(--accent);
  }
  .tab-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    padding: 0 5px;
    border-radius: 999px;
    background: var(--warn-bg);
    color: var(--warn-text);
    border: 1px solid var(--warn-border);
  }
  .tab-badge.is-err {
    background: var(--err-bg);
    color: var(--err-strong);
    border-color: var(--err-border);
  }
  .studio-body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 12px;
  }
  .input-tab {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .pending-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
    padding: 10px 12px;
    border-radius: 8px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
  }
  .pb-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 11.5px;
    line-height: 1.4;
  }
  .cta-btn {
    background: var(--accent);
    color: #fff;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 5px 11px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .cta-btn:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
  .spec-card {
    display: flex;
    flex-direction: column;
    gap: 7px;
    padding: 9px 11px;
    border-radius: 8px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
  }
  .spec-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }
  .spec-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .fmt-tag {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
  }
  .spec-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .kv-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    color: var(--text-secondary);
  }
  .kv-badge.tone-warn {
    background: var(--warn-bg);
    border-color: var(--warn-border);
    color: var(--warn-text);
  }
  .kv-badge.tone-info {
    background: var(--bg-pill);
    border-color: var(--border-strong);
    color: var(--accent-soft);
  }
  .kv-k {
    opacity: 0.8;
  }
  .kv-v {
    font-weight: 600;
  }
  .issues-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .issue-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
    padding: 7px 10px;
    border-radius: 7px;
    font-size: 11.5px;
    line-height: 1.4;
  }
  .issue-row.sev-error {
    background: var(--err-bg);
    border: 1px solid var(--err-border);
    color: var(--err-text);
  }
  .issue-row.sev-warning {
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
  }
  .issue-main {
    display: flex;
    align-items: baseline;
    gap: 6px;
    min-width: 0;
    flex: 1;
  }
  .issue-src {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 1px 5px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.16);
    flex-shrink: 0;
  }
  .fix-btn {
    background: var(--bg-surface);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 10.5px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    flex-shrink: 0;
  }
  .fix-btn:hover {
    border-color: var(--accent);
    color: var(--accent-bright);
  }
  .raw-drawer {
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 7px;
    overflow: hidden;
  }
  .raw-drawer summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 7px 10px;
    cursor: pointer;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    user-select: none;
  }
  .raw-drawer summary:hover {
    color: var(--text-primary);
    background: var(--bg-surface-hover);
  }
  .raw-copy-btn {
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 2px 7px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
  }
  .raw-copy-btn:hover {
    border-color: var(--accent);
    color: var(--text-primary);
  }
  .raw-pre {
    margin: 0;
    padding: 10px;
    border-top: 1px solid var(--border-subtle);
    background: var(--bg-canvas);
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    line-height: 1.45;
    max-height: 260px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .slot-pane {
    min-width: 0;
  }
  .slot-empty {
    padding: 18px 12px;
    text-align: center;
    font-size: 12px;
    color: var(--text-dim);
  }
</style>
