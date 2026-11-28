<script module lang="ts">
  // Ensure only one nested benchmark flyout is open at a time (no stacked shadows).
  let closeActiveFlyout: (() => void) | null = null;
</script>

<script lang="ts">
  import { tick } from 'svelte';
  import { AME_BENCHMARK_TARGETS, type BenchmarkTargetMeta } from '$lib/engine/benchmarks';
  import type { CatalogSubItem } from '$lib/engine/catalog';

  interface Props {
    selectedId?: string;
    disabled?: boolean;
    triggerLabel?: string;
    compact?: boolean;
    catalogItems?: CatalogSubItem[];
    onSelect?: (bench: BenchmarkTargetMeta) => void;
    onDragStart?: (ev: DragEvent, bench: BenchmarkTargetMeta) => void;
    onSelectCatalogItem?: (item: CatalogSubItem) => void;
    onDragCatalogItem?: (ev: DragEvent, item: CatalogSubItem) => void;
    onFlyoutMouseEnter?: () => void;
    onFlyoutMouseLeave?: () => void;
  }

  let {
    selectedId = '',
    disabled = false,
    triggerLabel = 'AME 41 Benchmarks',
    compact = false,
    catalogItems = undefined,
    onSelect = undefined,
    onDragStart = undefined,
    onSelectCatalogItem = undefined,
    onDragCatalogItem = undefined,
    onFlyoutMouseEnter = undefined,
    onFlyoutMouseLeave = undefined,
  }: Props = $props();

  let open = $state(false);
  let pinnedByClick = $state(false);
  let filterQuery = $state('');
  let wrapEl: HTMLDivElement | null = $state(null);
  let flyoutEl: HTMLDivElement | null = $state(null);
  let triggerEl: HTMLButtonElement | null = $state(null);
  let searchEl: HTMLInputElement | null = $state(null);

  /** Portal fixed flyout to document.body so ancestor backdrop-filter/transform cannot offset it. */
  function portal(node: HTMLElement) {
    if (typeof document !== 'undefined') document.body.appendChild(node);
    return {
      destroy() {
        node.remove();
      },
    };
  }

  let panelStyle = $state('');
  let closeTimer: ReturnType<typeof setTimeout> | null = null;

  let filteredTasks = $derived.by(() => {
    const q = filterQuery.trim().toLowerCase();
    if (!q) return AME_BENCHMARK_TARGETS;
    const tokens = q.split(/\s+/).filter(Boolean);
    return AME_BENCHMARK_TARGETS.filter((b) => {
      const blob = [
        b.mcsaId ?? '',
        b.pdbId,
        b.name,
        b.shortLabel,
        b.biology,
        ...(b.ligands ?? []),
        `${b.motifResidueCount ?? ''} res`,
        `${b.targetLength ?? 180} aa`,
      ]
        .join(' ')
        .toLowerCase();
      return tokens.every((tok) => blob.includes(tok));
    });
  });

  let filteredCatalogItems = $derived.by(() => {
    if (!catalogItems) return [];
    const q = filterQuery.trim().toLowerCase();
    if (!q) return catalogItems;
    const tokens = q.split(/\s+/).filter(Boolean);
    return catalogItems.filter((item) => tokens.every((tok) => item.searchBlob.includes(tok)));
  });

  let totalCount = $derived(catalogItems ? catalogItems.length : AME_BENCHMARK_TARGETS.length);

  let activeAmeBench = $derived(
    !catalogItems && selectedId
      ? AME_BENCHMARK_TARGETS.find((b) => b.id === selectedId) ?? null
      : null,
  );

  function supportsHover(): boolean {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return true;
    return window.matchMedia('(hover: hover)').matches;
  }

  function clearCloseTimer() {
    if (closeTimer !== null) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }
  }

  function scheduleClose() {
    if (pinnedByClick) return;
    clearCloseTimer();
    closeTimer = setTimeout(() => {
      closeTimer = null;
      open = false;
    }, 180);
  }

  function computePanelPosition() {
    if (typeof window === 'undefined' || !triggerEl) return;
    const rect = triggerEl.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const margin = 10;
    const gap = 6;
    const preferredW = Math.min(390, vw - margin * 2);
    const maxH = Math.min(460, Math.max(240, vh - margin * 2));

    const spaceRight = vw - rect.right - margin;
    const spaceLeft = rect.left - margin;

    let left: number;
    let top: number;

    if (spaceRight >= 300) {
      const w = Math.min(preferredW, spaceRight - gap);
      left = rect.right + gap;
      top = Math.max(margin, Math.min(rect.top - 12, vh - maxH - margin));
      panelStyle = `left:${Math.round(left)}px;top:${Math.round(top)}px;width:${Math.round(w)}px;max-height:${Math.round(maxH)}px;`;
      return;
    }

    if (spaceLeft >= 300) {
      const w = Math.min(preferredW, spaceLeft - gap);
      left = rect.left - gap - w;
      top = Math.max(margin, Math.min(rect.top - 12, vh - maxH - margin));
      panelStyle = `left:${Math.round(left)}px;top:${Math.round(top)}px;width:${Math.round(w)}px;max-height:${Math.round(maxH)}px;`;
      return;
    }

    // Fallback: position below/above the trigger
    const w = preferredW;
    left = Math.max(margin, Math.min(rect.left, vw - w - margin));
    const spaceBelow = vh - rect.bottom - margin;
    if (spaceBelow >= 260) {
      top = rect.bottom + gap;
    } else {
      top = Math.max(margin, rect.top - gap - maxH);
    }
    panelStyle = `left:${Math.round(left)}px;top:${Math.round(top)}px;width:${Math.round(w)}px;max-height:${Math.round(maxH)}px;`;
  }

  function openMenu(pin: boolean) {
    if (disabled) return;
    clearCloseTimer();
    if (closeActiveFlyout && closeActiveFlyout !== closeMenu) {
      closeActiveFlyout();
    }
    closeActiveFlyout = closeMenu;
    open = true;
    if (pin) pinnedByClick = true;
    void tick().then(() => {
      computePanelPosition();
      if (pin) searchEl?.focus();
    });
  }

  function closeMenu() {
    clearCloseTimer();
    if (closeActiveFlyout === closeMenu) closeActiveFlyout = null;
    open = false;
    pinnedByClick = false;
  }

  function handleTriggerClick(ev: MouseEvent) {
    ev.stopPropagation();
    if (disabled) return;
    if (open && pinnedByClick) {
      closeMenu();
    } else {
      openMenu(true);
    }
  }

  function handleSelect(bench: BenchmarkTargetMeta) {
    if (disabled) return;
    onSelect?.(bench);
    closeMenu();
  }

  function handleSelectItem(item: CatalogSubItem) {
    if (disabled) return;
    onSelectCatalogItem?.(item);
    closeMenu();
  }

  function handleKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape' && open) {
      ev.preventDefault();
      ev.stopPropagation();
      closeMenu();
      triggerEl?.focus();
    }
  }

  $effect(() => {
    if (!open) return;
    function onPointerDown(ev: PointerEvent) {
      const target = ev.target as Node | null;
      if (!target) return;
      if (wrapEl?.contains(target) || flyoutEl?.contains(target)) return;
      closeMenu();
    }
    function onResizeOrScroll() {
      computePanelPosition();
    }
    window.addEventListener('pointerdown', onPointerDown, true);
    window.addEventListener('resize', onResizeOrScroll);
    window.addEventListener('scroll', onResizeOrScroll, true);
    return () => {
      window.removeEventListener('pointerdown', onPointerDown, true);
      window.removeEventListener('resize', onResizeOrScroll);
      window.removeEventListener('scroll', onResizeOrScroll, true);
      clearCloseTimer();
    };
  });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="ame-submenu-wrap"
  class:compact
  bind:this={wrapEl}
  onmouseenter={() => {
    if (supportsHover()) openMenu(false);
  }}
  onmouseleave={() => {
    if (supportsHover()) scheduleClose();
  }}
  onkeydown={handleKeydown}
>
  <button
    bind:this={triggerEl}
    type="button"
    class="ame-trigger"
    class:is-open={open}
    class:has-active={Boolean(activeAmeBench)}
    {disabled}
    aria-haspopup="listbox"
    aria-expanded={open}
    onclick={handleTriggerClick}
    title={catalogItems
      ? `Hover or click to browse ${catalogItems.length} ${triggerLabel} benchmarks`
      : 'Hover or click to browse & filter all 41 AME M-CSA enzyme benchmarks'}
  >
    <span class="ame-trigger-main">
      <span class="ame-trigger-title">{triggerLabel}</span>
      {#if activeAmeBench}
        <span class="ame-trigger-active">
          {activeAmeBench.mcsaId} · {activeAmeBench.pdbId}
        </span>
      {:else}
        <span class="ame-trigger-count">{totalCount}</span>
      {/if}
    </span>
    <span class="ame-trigger-chev" aria-hidden="true">▸</span>
  </button>

  {#if open}
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      use:portal
      bind:this={flyoutEl}
      data-ame-flyout="true"
      class="ame-flyout"
      style={panelStyle}
      role="dialog"
      tabindex="-1"
      aria-label={triggerLabel}
      onmouseenter={() => {
        clearCloseTimer();
        onFlyoutMouseEnter?.();
      }}
      onmouseleave={() => {
        if (supportsHover()) {
          scheduleClose();
          onFlyoutMouseLeave?.();
        }
      }}
    >
      <div class="ame-flyout-hdr">
        <div class="ame-flyout-title-row">
          <span class="ame-flyout-title">{triggerLabel}</span>
          <span class="ame-flyout-sub">
            {#if catalogItems}
              {filteredCatalogItems.length} / {catalogItems.length} target{catalogItems.length === 1 ? '' : 's'}
            {:else}
              {filteredTasks.length} / {AME_BENCHMARK_TARGETS.length} M-CSA enzymes
            {/if}
          </span>
        </div>
        {#if !catalogItems || catalogItems.length > 6}
          <div class="ame-search-wrap">
            <span class="ame-search-ico" aria-hidden="true">⌕</span>
            <input
              bind:this={searchEl}
              type="text"
              class="ame-search-input"
              placeholder={catalogItems
                ? `Filter ${triggerLabel}…`
                : 'Filter by M-CSA, PDB, enzyme, or ligand…'}
              bind:value={filterQuery}
              aria-label={`Filter ${triggerLabel}`}
              onclick={(e) => e.stopPropagation()}
            />
            {#if filterQuery}
              <button
                type="button"
                class="ame-search-clear"
                aria-label="Clear filter"
                onclick={(e) => {
                  e.stopPropagation();
                  filterQuery = '';
                  searchEl?.focus();
                }}
              >
                ×
              </button>
            {/if}
          </div>
        {/if}
      </div>

      <div class="ame-list" role="listbox" aria-label={triggerLabel}>
        {#if catalogItems}
          {#if filteredCatalogItems.length === 0}
            <div class="ame-empty">No targets match “{filterQuery}”.</div>
          {:else}
            {#each filteredCatalogItems as item (item.key)}
              <button
                type="button"
                class="ame-row"
                role="option"
                aria-selected="false"
                draggable={!disabled && Boolean(onDragCatalogItem)}
                onclick={(e) => {
                  e.stopPropagation();
                  handleSelectItem(item);
                }}
                ondragstart={(e) => onDragCatalogItem?.(e, item)}
              >
                <div class="ame-row-top">
                  <span class="ame-enzyme" title={item.label}>{item.label}</span>
                  {#if item.badge}
                    <span class="ame-lig-pill">{item.badge}</span>
                  {/if}
                </div>
                {#if item.subtitle}
                  <div class="ame-cat-desc">{item.subtitle}</div>
                {/if}
                {#if item.meta}
                  <div class="ame-row-meta">
                    <span class="ame-stat">{item.meta}</span>
                  </div>
                {/if}
              </button>
            {/each}
          {/if}
        {:else if filteredTasks.length === 0}
          <div class="ame-empty">No AME tasks match “{filterQuery}”.</div>
        {:else}
          {#each filteredTasks as bench (bench.id)}
            {@const isSelected = bench.id === selectedId}
            {@const shortEnzyme = bench.name.replace(/^AME:\s*/, '')}
            <button
              type="button"
              class="ame-row"
              class:selected={isSelected}
              role="option"
              aria-selected={isSelected}
              draggable={!disabled && Boolean(onDragStart)}
              onclick={(e) => {
                e.stopPropagation();
                handleSelect(bench);
              }}
              ondragstart={(e) => onDragStart?.(e, bench)}
            >
              <div class="ame-row-top">
                <span class="ame-mcsa">{bench.mcsaId ?? 'AME'}</span>
                <span class="ame-pdb">{bench.pdbId}</span>
                <span class="ame-enzyme" title={shortEnzyme}>{shortEnzyme}</span>
              </div>
              <div class="ame-row-meta">
                <span class="ame-ligands">
                  {#each bench.ligands ?? [] as lig (lig)}
                    <span class="ame-lig-pill">{lig}</span>
                  {/each}
                </span>
                <span class="ame-stats">
                  <span>{bench.targetLength ?? 180} aa</span>
                  <span>·</span>
                  <span>{bench.motifResidueCount ?? 0} res</span>
                </span>
              </div>
            </button>
          {/each}
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .ame-submenu-wrap {
    position: relative;
    width: 100%;
  }
  .ame-trigger {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    background: var(--bg-surface-2);
    color: var(--text-primary);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    text-align: left;
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
  }
  .ame-submenu-wrap.compact .ame-trigger {
    margin-top: 4px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
  }
  .ame-trigger:hover:not(:disabled),
  .ame-trigger.is-open {
    background: var(--bg-surface-hover, rgba(99, 102, 241, 0.1));
    border-color: var(--accent, #6366f1);
    color: var(--text-primary);
  }
  .ame-trigger.has-active {
    border-color: var(--accent, #6366f1);
  }
  .ame-trigger:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .ame-trigger-main {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    flex: 1;
  }
  .ame-trigger-title {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ame-trigger-count,
  .ame-trigger-active {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.14);
    color: var(--accent, #818cf8);
    text-transform: none;
    letter-spacing: 0;
    white-space: nowrap;
  }
  .ame-trigger-chev {
    color: var(--text-dim);
    font-size: 11px;
    flex-shrink: 0;
    transition: transform 120ms ease;
  }
  .ame-trigger.is-open .ame-trigger-chev {
    color: var(--accent, #818cf8);
  }

  .ame-flyout {
    position: fixed;
    z-index: 120;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.14), 0 2px 6px rgba(15, 23, 42, 0.06);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .ame-flyout-hdr {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-surface-2);
    display: flex;
    flex-direction: column;
    gap: 6px;
    flex-shrink: 0;
  }
  .ame-flyout-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .ame-flyout-title {
    font-size: 11.5px;
    font-weight: 700;
    color: var(--text-primary);
  }
  .ame-flyout-sub {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    color: var(--text-dim);
  }
  .ame-search-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-surface);
  }
  .ame-search-wrap:focus-within {
    border-color: var(--accent, #6366f1);
  }
  .ame-search-ico {
    color: var(--text-dim);
    font-size: 12px;
    line-height: 1;
    user-select: none;
  }
  .ame-search-input {
    flex: 1;
    min-width: 0;
    border: none;
    outline: none;
    background: transparent;
    color: var(--text-primary);
    font-size: 11px;
    font-family: inherit;
  }
  .ame-search-input::placeholder {
    color: var(--text-dim);
  }
  .ame-search-clear {
    border: none;
    background: transparent;
    color: var(--text-dim);
    cursor: pointer;
    font-size: 13px;
    line-height: 1;
    padding: 0 2px;
  }
  .ame-search-clear:hover {
    color: var(--text-primary);
  }

  .ame-list {
    overflow-y: auto;
    padding: 5px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
    min-height: 0;
  }
  .ame-empty {
    padding: 18px 12px;
    text-align: center;
    font-size: 11px;
    color: var(--text-dim);
  }
  .ame-row {
    width: 100%;
    border: 1px solid transparent;
    border-radius: 6px;
    background: transparent;
    padding: 6px 8px;
    text-align: left;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: background 100ms ease, border-color 100ms ease;
  }
  .ame-row:hover {
    background: var(--bg-surface-2);
    border-color: var(--border);
  }
  .ame-row.selected {
    background: rgba(99, 102, 241, 0.14);
    border-color: var(--accent, #6366f1);
  }
  .ame-row-top {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }
  .ame-mcsa {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 700;
    padding: 1px 4px;
    border-radius: 4px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    flex-shrink: 0;
  }
  .ame-pdb {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 700;
    color: var(--accent, #818cf8);
    flex-shrink: 0;
  }
  .ame-enzyme {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
    flex: 1;
  }
  .ame-row-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .ame-cat-desc {
    font-size: 10.5px;
    line-height: 1.35;
    color: var(--text-muted, #64748b);
    white-space: normal;
  }
  .ame-ligands {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
    min-width: 0;
  }
  .ame-lig-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    font-weight: 600;
    padding: 1px 4px;
    border-radius: 3px;
    background: rgba(16, 185, 129, 0.14);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
  }
  .ame-stats {
    display: flex;
    align-items: center;
    gap: 4px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    color: var(--text-dim);
    flex-shrink: 0;
  }
</style>
