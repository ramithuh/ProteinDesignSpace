<script lang="ts">
  import { ADAPTER_REGISTRY } from '$lib/engine/registry';
  import { PORT_COLORS, STAGE_COLORS } from '$lib/engine/graph-utils';

  interface Props {
    disabled?: boolean;
    triggerEl?: HTMLElement | null;
    onInsert: (adapterId: string) => void;
    onDragStart?: (ev: DragEvent, adapterId: string) => void;
    onClose: () => void;
  }

  let {
    disabled = false,
    triggerEl = null,
    onInsert,
    onDragStart = undefined,
    onClose,
  }: Props = $props();

  const STAGE_GROUPS = [
    { title: 'Inputs', ids: ['pdb-input', 'sequence-input'] },
    { title: 'Conditioning', ids: ['rfd4-conditioner'] },
    { title: 'Generate', ids: ['rfd4-proteina', 'genie3'] },
    { title: 'Inverse Fold', ids: ['web-protein-mpnn'] },
    { title: 'Fold', ids: ['localfold-af2'] },
    { title: 'Evaluate', ids: ['designability-scorer'] },
  ];

  const CONCISE_ROLES: Record<string, string> = {
    'pdb-input': 'Built-in 3-helix demo or upload/paste PDB & mmCIF',
    'sequence-input': 'Paste amino acid sequences (plain or FASTA)',
    'rfd4-conditioner': 'Contigs, hotspots, motifs, H-bonds & CATH folds',
    'rfd4-proteina': 'All-atom sequence + backbone diffusion (WebGPU INT8)',
    genie3: 'Fast SE(3)-equivariant protein diffusion (~45 MB WebGPU FP16)',
    'web-protein-mpnn': 'Inverse-fold sequences with Soluble / Protein / Cα MPNN',
    'localfold-af2': 'Fold with AF2, RoseTTAFold3, Boltz-2, OpenBind-0, AF3 (WebGPU)',
    'designability-scorer': 'Kabsch superposition, scRMSD, scTM & pLDDT pass rate',
  };

  let query = $state('');
  let activeIdx = $state(0);
  let popoverEl: HTMLDivElement | null = $state(null);
  let searchInputEl: HTMLInputElement | null = $state(null);

  interface CatalogItem {
    id: string;
    group: string;
    shortName: string;
    displayName: string;
    stage: string;
    role: string;
    weightLabel: string;
    searchBlob: string;
  }

  const ALL_ITEMS: CatalogItem[] = STAGE_GROUPS.flatMap((grp) =>
    grp.ids
      .map((id): CatalogItem | null => {
        const m = ADAPTER_REGISTRY[id];
        if (!m) return null;
        const portTokens = [
          ...Object.values(m.inputs),
          ...Object.values(m.outputs),
        ].flatMap((spec) => {
          const pc = PORT_COLORS[spec.type];
          return pc ? [spec.type, pc.label, pc.short] : [spec.type];
        });
        return {
          id,
          group: grp.title,
          shortName: m.shortName,
          displayName: m.displayName,
          stage: m.stage,
          role: CONCISE_ROLES[id] ?? m.description,
          weightLabel: m.approxWeightsMB > 0 ? `~${m.approxWeightsMB} MB` : 'CPU',
          searchBlob: [
            m.id,
            m.displayName,
            m.shortName,
            m.stage,
            grp.title,
            m.description,
            CONCISE_ROLES[id] ?? '',
            ...portTokens,
          ]
            .join(' ')
            .toLowerCase(),
        };
      })
      .filter((x): x is CatalogItem => x !== null),
  );

  let filteredItems = $derived.by(() => {
    const q = query.trim().toLowerCase();
    if (!q) return ALL_ITEMS;
    const terms = q.split(/\s+/).filter(Boolean);
    return ALL_ITEMS.filter((item) => terms.every((t) => item.searchBlob.includes(t)));
  });

  let groupedItems = $derived.by(() => {
    const groups: { title: string; items: { item: CatalogItem; flatIdx: number }[] }[] = [];
    const byTitle = new Map<string, { title: string; items: { item: CatalogItem; flatIdx: number }[] }>();
    filteredItems.forEach((item, flatIdx) => {
      let g = byTitle.get(item.group);
      if (!g) {
        g = { title: item.group, items: [] };
        byTitle.set(item.group, g);
        groups.push(g);
      }
      g.items.push({ item, flatIdx });
    });
    return groups;
  });

  // Reset active index to the top match whenever the search query changes
  $effect(() => {
    void query;
    activeIdx = 0;
  });

  // Clamp active index if filtered list shrinks
  $effect(() => {
    if (filteredItems.length === 0 || activeIdx >= filteredItems.length) {
      activeIdx = 0;
    }
  });

  // Autofocus search input on mount
  $effect(() => {
    searchInputEl?.focus();
  });

  // Close on click-outside
  $effect(() => {
    function handlePointerDown(ev: PointerEvent) {
      const target = ev.target as Node | null;
      if (!target) return;
      if (popoverEl?.contains(target)) return;
      if (triggerEl?.contains(target)) return;
      onClose();
    }
    window.addEventListener('pointerdown', handlePointerDown);
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
    };
  });

  function scrollActiveIntoView(idx: number) {
    requestAnimationFrame(() => {
      const el = popoverEl?.querySelector<HTMLElement>(`[data-cat-idx="${idx}"]`);
      el?.scrollIntoView({ block: 'nearest' });
    });
  }

  function handleKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      ev.preventDefault();
      ev.stopPropagation();
      onClose();
      return;
    }
    if (ev.key === 'ArrowDown') {
      ev.preventDefault();
      if (filteredItems.length === 0) return;
      activeIdx = (activeIdx + 1) % filteredItems.length;
      scrollActiveIntoView(activeIdx);
      return;
    }
    if (ev.key === 'ArrowUp') {
      ev.preventDefault();
      if (filteredItems.length === 0) return;
      activeIdx = (activeIdx - 1 + filteredItems.length) % filteredItems.length;
      scrollActiveIntoView(activeIdx);
      return;
    }
    if (ev.key === 'Enter') {
      ev.preventDefault();
      const target = filteredItems[activeIdx];
      if (target && !disabled) {
        onInsert(target.id);
      }
    }
  }
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div
  class="catalog-popover"
  bind:this={popoverEl}
  role="dialog"
  aria-label="Add workflow block"
  tabindex="-1"
  onkeydown={handleKeydown}
>
  <div class="search-row">
    <span class="search-icon" aria-hidden="true">⌕</span>
    <input
      bind:this={searchInputEl}
      type="text"
      class="search-input"
      placeholder="Search blocks, stages, or port types…"
      bind:value={query}
      aria-label="Filter blocks"
    />
    <kbd class="esc-kbd">Esc</kbd>
  </div>

  <div class="catalog-list" role="listbox">
    {#if groupedItems.length === 0}
      <div class="empty">No blocks match “{query}”.</div>
    {:else}
      {#each groupedItems as grp (grp.title)}
        <div class="group-hdr">{grp.title}</div>
        {#each grp.items as { item, flatIdx } (item.id)}
          {@const stageColor = STAGE_COLORS[item.stage] ?? '#64748b'}
          <button
            type="button"
            class="cat-item"
            class:active={flatIdx === activeIdx}
            data-cat-idx={flatIdx}
            role="option"
            aria-selected={flatIdx === activeIdx}
            draggable={!disabled}
            {disabled}
            onmouseenter={() => (activeIdx = flatIdx)}
            ondragstart={(e) => onDragStart?.(e, item.id)}
            onclick={() => onInsert(item.id)}
          >
            <span class="stage-dot" style="background: {stageColor}"></span>
            <span class="item-body">
              <span class="item-name">{item.shortName}</span>
              <span class="item-role">{item.role}</span>
            </span>
            <span class="wt-pill">{item.weightLabel}</span>
          </button>
        {/each}
      {/each}
    {/if}
  </div>

  <div class="catalog-foot">
    <span><kbd>↑↓</kbd> navigate · <kbd>↵</kbd> insert &amp; auto-wire · drag to place</span>
  </div>
</div>

<style>
  .catalog-popover {
    position: absolute;
    top: 44px;
    left: 10px;
    z-index: 45;
    width: 340px;
    max-width: calc(100% - 20px);
    max-height: min(380px, 70vh);
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.35), var(--node-shadow);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .search-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-surface-2);
  }
  .search-icon {
    color: var(--text-dim);
    font-size: 14px;
    line-height: 1;
    user-select: none;
  }
  .search-input {
    flex: 1;
    min-width: 0;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text-primary);
    font-size: 12px;
    font-family: inherit;
  }
  .search-input::placeholder {
    color: var(--text-dim);
  }
  .esc-kbd,
  .catalog-foot kbd {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    color: var(--text-dim);
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1px 4px;
  }
  .catalog-list {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 6px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .group-hdr {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-dim);
    padding: 6px 8px 3px;
  }
  .group-hdr:first-child {
    padding-top: 2px;
  }
  .cat-item {
    display: flex;
    align-items: center;
    gap: 9px;
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 8px;
    color: var(--text-body);
    cursor: pointer;
    font-family: inherit;
  }
  .cat-item:hover:not(:disabled),
  .cat-item.active:not(:disabled) {
    background: var(--bg-surface-selected);
    border-color: var(--border-strong);
  }
  .cat-item:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .stage-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .item-body {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }
  .item-name {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.25;
  }
  .item-role {
    font-size: 10.5px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .wt-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    color: var(--text-muted);
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 1px 6px;
    flex-shrink: 0;
    white-space: nowrap;
  }
  .empty {
    padding: 18px 12px;
    text-align: center;
    font-size: 12px;
    color: var(--text-dim);
  }
  .catalog-foot {
    padding: 5px 10px;
    border-top: 1px solid var(--border-subtle);
    background: var(--bg-surface-2);
    font-size: 10px;
    color: var(--text-dim);
  }
</style>
