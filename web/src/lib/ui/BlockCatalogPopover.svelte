<script lang="ts">
  import { tick } from 'svelte';
  import {
    filterCatalogCategories,
    type CatalogInsertRequest,
    type CatalogSubItem,
    type FilteredCategoryEntry,
  } from '$lib/engine/catalog';
  import { STAGE_COLORS } from '$lib/engine/graph-utils';
  import type { BenchmarkTargetMeta } from '$lib/engine/benchmarks';
  import AmeBenchmarkSubmenu from './AmeBenchmarkSubmenu.svelte';

  interface Props {
    disabled?: boolean;
    triggerEl?: HTMLElement | null;
    onInsert: (req: CatalogInsertRequest) => void;
    onDragStart?: (ev: DragEvent, req: CatalogInsertRequest) => void;
    onClose: () => void;
  }

  let {
    disabled = false,
    triggerEl = null,
    onInsert,
    onDragStart = undefined,
    onClose,
  }: Props = $props();

  let query = $state('');
  let activeIdx = $state(0);
  let activeSubIdx = $state(0);
  let focusPane = $state<'main' | 'flyout'>('main');
  let flyoutFromKeyboard = $state(false);
  let flyoutCategoryId = $state<string | null>(null);
  let expandedGroups = $state<Record<string, boolean>>({});

  let shellEl: HTMLDivElement | null = $state(null);
  let mainEl: HTMLDivElement | null = $state(null);
  let flyoutEl: HTMLDivElement | null = $state(null);
  let searchInputEl: HTMLInputElement | null = $state(null);

  let flyoutLeft = $state(316);
  let flyoutTop = $state(0);
  let flyoutWidth = $state(340);
  let flyoutMaxHeight = $state(440);
  let flyoutSide = $state<'right' | 'left' | 'overlay'>('right');

  let closeGraceTimer: ReturnType<typeof setTimeout> | null = null;

  let filteredCategories = $derived(filterCatalogCategories(query));
  let activeEntry = $derived<FilteredCategoryEntry | null>(filteredCategories[activeIdx] ?? null);

  let flyoutEntry = $derived.by<FilteredCategoryEntry | null>(() => {
    if (!flyoutCategoryId) return null;
    const found = filteredCategories.find((e) => e.category.id === flyoutCategoryId) ?? null;
    if (!found || found.matchingSubGroups.length === 0) return null;
    return found;
  });

  function isSubGroupOpen(
    catId: string,
    subGrp: { title: string; collapsible?: boolean; defaultCollapsed?: boolean; nestedFlyout?: 'ame41' | 'benchmarks' },
  ): boolean {
    if (subGrp.nestedFlyout) return query.trim().length > 0;
    if (!subGrp.collapsible) return true;
    if (query.trim().length > 0) return true;
    const key = `${catId}::${subGrp.title}`;
    if (key in expandedGroups) return expandedGroups[key];
    return !subGrp.defaultCollapsed;
  }

  function buildAmeInsertRequest(bench: BenchmarkTargetMeta): CatalogInsertRequest {
    return {
      adapterId: 'pdb-input',
      params: {
        mode: 'pdb-id',
        pdbId: bench.pdbId,
        filename: `${bench.pdbId}.cif`,
        pdbText: '',
        benchmarkId: bench.id,
        ...(bench.cropSegments ? { cropSegments: bench.cropSegments } : {}),
      },
      label: `PDB: ${bench.name} (${bench.pdbId})`,
    };
  }

  function toggleSubGroup(catId: string, title: string, currentlyOpen: boolean) {
    const key = `${catId}::${title}`;
    expandedGroups = { ...expandedGroups, [key]: !currentlyOpen };
  }

  let visibleFlyoutSubs = $derived.by((): CatalogSubItem[] => {
    if (!flyoutEntry) return [];
    const out: CatalogSubItem[] = [];
    for (const grp of flyoutEntry.matchingSubGroups) {
      if (isSubGroupOpen(flyoutEntry.category.id, grp)) {
        out.push(...grp.items);
      }
    }
    return out;
  });

  let activeDescendantId = $derived.by((): string | undefined => {
    if (focusPane === 'flyout' && flyoutEntry) {
      const sub = visibleFlyoutSubs[activeSubIdx];
      return sub ? `cat-sub-${flyoutEntry.category.id}-${sub.key}` : undefined;
    }
    return activeEntry ? `cat-item-${activeEntry.category.id}` : undefined;
  });

  function supportsHover(): boolean {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return true;
    return window.matchMedia('(hover: hover)').matches;
  }

  function clearCloseTimer() {
    if (closeGraceTimer !== null) {
      clearTimeout(closeGraceTimer);
      closeGraceTimer = null;
    }
  }

  function scheduleFlyoutClose() {
    if (!supportsHover()) return;
    clearCloseTimer();
    closeGraceTimer = setTimeout(() => {
      closeGraceTimer = null;
      if (focusPane !== 'flyout' && !query.trim()) {
        flyoutCategoryId = null;
      }
    }, 150);
  }

  function handleShellMouseLeave() {
    if (!supportsHover()) return;
    if (!flyoutFromKeyboard) {
      focusPane = 'main';
    }
    scheduleFlyoutClose();
  }

  let placement = $state<'above' | 'below'>('above');

  function updatePlacement() {
    if (typeof window === 'undefined' || !triggerEl) return;
    const rect = triggerEl.getBoundingClientRect();
    const spaceAbove = rect.top - 12;
    const spaceBelow = window.innerHeight - rect.bottom - 12;
    placement = spaceAbove >= 310 || spaceAbove >= spaceBelow ? 'above' : 'below';
  }

  function updateFlyoutGeometry() {
    if (typeof window === 'undefined' || !shellEl || !mainEl || !flyoutEntry) return;
    const shellRect = shellEl.getBoundingClientRect();
    const mainRect = mainEl.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const margin = 12;
    const gap = 6;

    const spaceRight = vw - mainRect.right - margin;
    const spaceLeft = mainRect.left - margin;
    const preferredW = 340;

    let side: 'right' | 'left' | 'overlay' = 'right';
    let width = preferredW;
    let left = mainRect.width + gap;

    if (spaceRight >= 250) {
      side = 'right';
      width = Math.min(preferredW, spaceRight);
      left = mainRect.width + gap;
    } else if (spaceLeft >= 250) {
      side = 'left';
      width = Math.min(preferredW, spaceLeft);
      left = -(width + gap);
    } else {
      side = 'overlay';
      width = Math.min(preferredW, Math.max(220, vw - margin * 2));
      const desiredGlobalLeft = Math.max(margin, vw - margin - width);
      left = desiredGlobalLeft - shellRect.left;
    }

    const availH = Math.min(460, Math.max(200, vh - margin * 2));
    const maxH = availH;

    // Align vertically near the hovered/active row, allowing upward expansion when placed above the dock
    const rowEl = mainEl.querySelector<HTMLElement>(`[data-category-id="${flyoutEntry.category.id}"]`);
    let desiredTop = 0;
    if (rowEl) {
      const rowRect = rowEl.getBoundingClientRect();
      desiredTop = rowRect.top - shellRect.top - 8;
    }
    const minTop = -(shellRect.top - margin);
    const estFlyoutH = Math.min(maxH, Math.max(200, flyoutEntry.matchingSubItems.length * 56 + 68));
    const maxTop = vh - shellRect.top - estFlyoutH - margin;
    const clampedTop = Math.max(minTop, Math.min(desiredTop, Math.max(minTop, maxTop)));

    flyoutSide = side;
    flyoutWidth = Math.round(width);
    flyoutLeft = Math.round(left);
    flyoutTop = Math.round(clampedTop);
    flyoutMaxHeight = Math.round(maxH);
  }

  // Reset navigation & auto-open flyout when searching
  $effect(() => {
    const q = query.trim();
    activeIdx = 0;
    activeSubIdx = 0;
    focusPane = 'main';
    flyoutFromKeyboard = false;
    clearCloseTimer();
    if (q) {
      const top = filteredCategories[0];
      if (top && top.matchingSubItems.length > 0) {
        flyoutCategoryId = top.category.id;
      } else {
        flyoutCategoryId = null;
      }
    } else {
      flyoutCategoryId = null;
    }
  });

  // Clamp indices if filtered lists shrink
  $effect(() => {
    if (filteredCategories.length === 0 || activeIdx >= filteredCategories.length) {
      activeIdx = 0;
    }
  });

  $effect(() => {
    const subs = visibleFlyoutSubs;
    if (subs.length === 0 || activeSubIdx >= subs.length) {
      activeSubIdx = 0;
    }
  });

  // Recompute flyout geometry whenever open target or viewport changes
  $effect(() => {
    if (!flyoutEntry) return;
    void query;
    void tick().then(() => updateFlyoutGeometry());
  });

  // Compute initial auto-flip placement and recompute on resize
  $effect(() => {
    void triggerEl;
    updatePlacement();
    function onResize() {
      updatePlacement();
      if (flyoutEntry) updateFlyoutGeometry();
    }
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      clearCloseTimer();
    };
  });

  // Autofocus search input on mount (skip on touch/coarse pointers so soft keyboard stays closed)
  $effect(() => {
    if (supportsHover()) {
      searchInputEl?.focus();
    }
  });

  // Close on click-outside
  $effect(() => {
    function handlePointerDown(ev: PointerEvent) {
      const target = ev.target as Node | null;
      if (!target) return;
      if (shellEl?.contains(target)) return;
      if (triggerEl?.contains(target)) return;
      if (target instanceof Element && target.closest('[data-ame-flyout]')) return;
      onClose();
    }
    window.addEventListener('pointerdown', handlePointerDown);
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
    };
  });

  function scrollActiveIntoView(idx: number) {
    requestAnimationFrame(() => {
      const el = mainEl?.querySelector<HTMLElement>(`[data-cat-idx="${idx}"]`);
      el?.scrollIntoView({ block: 'nearest' });
    });
  }

  function scrollFlyoutItemIntoView(idx: number) {
    requestAnimationFrame(() => {
      const el = flyoutEl?.querySelector<HTMLElement>(`[data-sub-idx="${idx}"]`);
      el?.scrollIntoView({ block: 'nearest' });
    });
  }

  function handleCategoryHover(entry: FilteredCategoryEntry, flatIdx: number) {
    if (!supportsHover()) return;
    activeIdx = flatIdx;
    focusPane = 'main';
    flyoutFromKeyboard = false;
    clearCloseTimer();
    if (flyoutCategoryId !== entry.category.id) {
      activeSubIdx = 0;
    }
    flyoutCategoryId = entry.category.id;
  }

  function handleCategoryClick(entry: FilteredCategoryEntry, flatIdx: number) {
    if (disabled) return;
    activeIdx = flatIdx;
    // Direct-insert only when there is a single unambiguous option; otherwise open & pin the flyout
    if (entry.matchingSubItems.length === 1) {
      onInsert(entry.matchingSubItems[0].insert);
      return;
    }
    // Otherwise open & focus its flyout so touch/click users can pick cleanly
    clearCloseTimer();
    flyoutCategoryId = entry.category.id;
    focusPane = 'flyout';
    flyoutFromKeyboard = true;
    activeSubIdx = 0;
    scrollFlyoutItemIntoView(0);
  }

  function handleInsertSubItem(sub: CatalogSubItem) {
    if (disabled) return;
    onInsert(sub.insert);
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
      if (focusPane === 'flyout' && flyoutEntry && visibleFlyoutSubs.length > 0) {
        activeSubIdx = (activeSubIdx + 1) % visibleFlyoutSubs.length;
        scrollFlyoutItemIntoView(activeSubIdx);
      } else if (filteredCategories.length > 0) {
        activeIdx = (activeIdx + 1) % filteredCategories.length;
        scrollActiveIntoView(activeIdx);
        const nextEntry = filteredCategories[activeIdx];
        if (flyoutCategoryId !== null && nextEntry) {
          flyoutCategoryId = nextEntry.category.id;
          activeSubIdx = 0;
        }
      }
      return;
    }

    if (ev.key === 'ArrowUp') {
      ev.preventDefault();
      if (focusPane === 'flyout' && flyoutEntry && visibleFlyoutSubs.length > 0) {
        const len = visibleFlyoutSubs.length;
        activeSubIdx = (activeSubIdx - 1 + len) % len;
        scrollFlyoutItemIntoView(activeSubIdx);
      } else if (filteredCategories.length > 0) {
        activeIdx = (activeIdx - 1 + filteredCategories.length) % filteredCategories.length;
        scrollActiveIntoView(activeIdx);
        const prevEntry = filteredCategories[activeIdx];
        if (flyoutCategoryId !== null && prevEntry) {
          flyoutCategoryId = prevEntry.category.id;
          activeSubIdx = 0;
        }
      }
      return;
    }

    if (ev.key === 'ArrowRight') {
      if (focusPane === 'main' && activeEntry && activeEntry.matchingSubItems.length > 0) {
        const inp = searchInputEl;
        const atEnd = !inp || inp.selectionStart === inp.value.length;
        if (atEnd) {
          ev.preventDefault();
          clearCloseTimer();
          flyoutCategoryId = activeEntry.category.id;
          focusPane = 'flyout';
          flyoutFromKeyboard = true;
          activeSubIdx = 0;
          scrollFlyoutItemIntoView(0);
          searchInputEl?.focus();
        }
      }
      return;
    }

    if (ev.key === 'ArrowLeft') {
      if (focusPane === 'flyout') {
        ev.preventDefault();
        focusPane = 'main';
        flyoutFromKeyboard = false;
        searchInputEl?.focus();
      }
      return;
    }

    if (ev.key === 'Enter') {
      ev.preventDefault();
      ev.stopPropagation();
      if (disabled) return;
      if (focusPane === 'flyout' && flyoutEntry) {
        const sub = visibleFlyoutSubs[activeSubIdx];
        if (sub) {
          onInsert(sub.insert);
          return;
        }
      }
      if (activeEntry) {
        handleCategoryClick(activeEntry, activeIdx);
      }
    }
  }
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div
  class="catalog-shell"
  class:place-above={placement === 'above'}
  class:place-below={placement === 'below'}
  bind:this={shellEl}
  role="dialog"
  aria-label="Add workflow block"
  tabindex="-1"
  onkeydown={handleKeydown}
  onmouseleave={handleShellMouseLeave}
  onmouseenter={clearCloseTimer}
>
  <div class="catalog-main" bind:this={mainEl}>
    <div class="search-row">
      <span class="search-icon" aria-hidden="true">⌕</span>
      <input
        bind:this={searchInputEl}
        type="text"
        class="search-input"
        placeholder="Search models, blocks, or tasks…"
        bind:value={query}
        aria-label="Filter blocks"
        aria-activedescendant={activeDescendantId}
      />
      <kbd class="esc-kbd">Esc</kbd>
    </div>

    <div class="catalog-list" role="listbox" aria-label="Workflow categories">
      {#if filteredCategories.length === 0}
        <div class="empty">No blocks match “{query}”.</div>
      {:else}
        {#each filteredCategories as entry, flatIdx (entry.category.id)}
          {@const cat = entry.category}
          {@const stageColor = STAGE_COLORS[cat.stage] ?? '#64748b'}
          {@const isFlyoutOpen = flyoutEntry?.category.id === cat.id}
          {@const singleSub = entry.matchingSubItems.length === 1 ? entry.matchingSubItems[0] : null}
          <button
            id={`cat-item-${cat.id}`}
            type="button"
            class="cat-item"
            class:active={flatIdx === activeIdx && focusPane === 'main'}
            class:flyout-open={isFlyoutOpen}
            data-cat-idx={flatIdx}
            data-category-id={cat.id}
            role="option"
            aria-selected={flatIdx === activeIdx}
            tabindex="-1"
            draggable={!disabled && Boolean(singleSub)}
            {disabled}
            onmouseenter={() => handleCategoryHover(entry, flatIdx)}
            ondragstart={(e) => {
              if (!singleSub) return;
              searchInputEl?.focus();
              onDragStart?.(e, singleSub.insert);
            }}
            onclick={() => {
              if (supportsHover()) searchInputEl?.focus();
              handleCategoryClick(entry, flatIdx);
            }}
          >
            <span class="stage-dot" style="background: {stageColor}"></span>
            <span class="item-body">
              <span class="item-name">{cat.title}</span>
              <span class="item-role">{cat.subtitle}</span>
            </span>
            <span class="sub-trigger" title="Hover or press → to choose">
              {entry.hasNarrowedSubMatch
                ? `${entry.matchingSubItems.length} match${entry.matchingSubItems.length === 1 ? '' : 'es'}`
                : cat.countLabel}
              <span class="chev" aria-hidden="true">›</span>
            </span>
          </button>
        {/each}
      {/if}
    </div>

    <div class="catalog-foot">
      <span><kbd>↑↓</kbd> categories · <kbd>→</kbd> models · <kbd>↵</kbd> select · drag to place</span>
    </div>
  </div>

  {#if flyoutEntry}
    {@const stageColor = STAGE_COLORS[flyoutEntry.category.stage] ?? '#64748b'}
    {@const flatSubs = flyoutEntry.matchingSubItems}
    <div
      class="catalog-flyout"
      class:side-left={flyoutSide === 'left'}
      class:side-overlay={flyoutSide === 'overlay'}
      bind:this={flyoutEl}
      style="left: {flyoutLeft}px; top: {flyoutTop}px; width: {flyoutWidth}px; max-height: {flyoutMaxHeight}px;"
      role="listbox"
      tabindex="-1"
      aria-label={`${flyoutEntry.category.title} options`}
      onmouseenter={clearCloseTimer}
    >
      <div class="flyout-hdr">
        {#if flyoutSide === 'overlay'}
          <button
            type="button"
            class="flyout-back"
            onclick={() => {
              flyoutCategoryId = null;
              focusPane = 'main';
              flyoutFromKeyboard = false;
            }}
            aria-label="Back to categories"
          >
            ← Back
          </button>
        {/if}
        <span class="stage-dot" style="background: {stageColor}"></span>
        <span class="flyout-title">{flyoutEntry.category.title}</span>
        <span class="flyout-sub">{flatSubs.length} option{flatSubs.length === 1 ? '' : 's'}</span>
      </div>

      <div class="flyout-list">
        {#each flyoutEntry.matchingSubGroups as subGrp (subGrp.title)}
          {@const grpOpen = isSubGroupOpen(flyoutEntry.category.id, subGrp)}
          {#if subGrp.nestedFlyout === 'ame41' && !query.trim()}
            <AmeBenchmarkSubmenu
              compact={true}
              {disabled}
              triggerLabel="AME 41 Benchmarks"
              onSelect={(bench) => {
                if (disabled) return;
                onInsert(buildAmeInsertRequest(bench));
              }}
              onDragStart={onDragStart
                ? (ev, bench) => {
                    searchInputEl?.focus();
                    onDragStart(ev, buildAmeInsertRequest(bench));
                  }
                : undefined}
              onFlyoutMouseEnter={clearCloseTimer}
              onFlyoutMouseLeave={scheduleFlyoutClose}
            />
          {:else if subGrp.nestedFlyout === 'benchmarks' && !query.trim()}
            <AmeBenchmarkSubmenu
              compact={true}
              {disabled}
              triggerLabel={subGrp.title}
              catalogItems={subGrp.items}
              onSelectCatalogItem={(item) => {
                if (disabled) return;
                onInsert(item.insert);
              }}
              onDragCatalogItem={onDragStart
                ? (ev, item) => {
                    searchInputEl?.focus();
                    onDragStart(ev, item.insert);
                  }
                : undefined}
              onFlyoutMouseEnter={clearCloseTimer}
              onFlyoutMouseLeave={scheduleFlyoutClose}
            />
          {:else if subGrp.collapsible}
            <button
              type="button"
              class="group-toggle"
              class:open={grpOpen}
              aria-expanded={grpOpen}
              onclick={() => {
                searchInputEl?.focus();
                toggleSubGroup(flyoutEntry.category.id, subGrp.title, grpOpen);
              }}
            >
              <span class="grp-chev" aria-hidden="true">▸</span>
              <span class="grp-label">{subGrp.title}</span>
              <span class="grp-count">{subGrp.items.length}</span>
            </button>
          {:else}
            <div class="group-hdr">{subGrp.title}</div>
          {/if}
          {#if grpOpen}
            {#each subGrp.items as sub (sub.key)}
              {@const subIdx = visibleFlyoutSubs.indexOf(sub)}
              {@const isSubActive =
                (focusPane === 'flyout' && subIdx === activeSubIdx) ||
                (focusPane === 'main' && Boolean(query.trim()) && flyoutEntry.matchingSubItems.length === 1 && subIdx === 0)}
              <button
                id={`cat-sub-${flyoutEntry.category.id}-${sub.key}`}
                type="button"
                class="flyout-item"
                class:active={isSubActive}
                data-sub-idx={subIdx}
                data-sub-key={sub.key}
                role="option"
                aria-selected={isSubActive}
                tabindex="-1"
                draggable={!disabled}
                {disabled}
                onmouseenter={() => {
                  if (!supportsHover()) return;
                  clearCloseTimer();
                  flyoutFromKeyboard = false;
                  focusPane = 'flyout';
                  activeSubIdx = subIdx;
                }}
                ondragstart={(e) => {
                  searchInputEl?.focus();
                  onDragStart?.(e, sub.insert);
                }}
                onclick={() => {
                  if (supportsHover()) searchInputEl?.focus();
                  handleInsertSubItem(sub);
                }}
              >
                <div class="sub-main">
                  <div class="sub-top">
                    <span class="sub-label">{sub.label}</span>
                    {#if sub.badge}
                      <span class="sub-badge" class:warn={sub.nonCommercial}>{sub.badge}</span>
                    {/if}
                  </div>
                  {#if sub.subtitle}
                    <div class="sub-desc">{sub.subtitle}</div>
                  {/if}
                  {#if sub.meta}
                    <div class="sub-meta">{sub.meta}</div>
                  {/if}
                </div>
              </button>
            {/each}
          {/if}
        {/each}
      </div>

      {#if flyoutEntry.category.footnote}
        <div class="flyout-foot">{flyoutEntry.category.footnote}</div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .catalog-shell {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    z-index: 45;
    overflow: visible;
    outline: none;
  }
  .catalog-shell.place-above {
    bottom: 94px;
    top: auto;
  }
  .catalog-shell.place-below {
    top: calc(100% + 4px);
    bottom: auto;
  }
  .catalog-main {
    width: 310px;
    max-width: calc(100vw - 24px);
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
  .group-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    margin-top: 4px;
    padding: 6px 8px;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    cursor: pointer;
    text-align: left;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
  }
  .group-toggle:hover {
    background: var(--bg-surface-selected);
    border-color: var(--border-strong);
    color: var(--text-primary);
  }
  .group-toggle.open {
    margin-bottom: 2px;
    color: var(--text-primary);
    border-color: var(--border-strong);
  }
  .grp-chev {
    display: inline-block;
    font-size: 11px;
    line-height: 1;
    color: var(--text-dim);
    transition: transform 140ms ease;
  }
  .group-toggle.open .grp-chev {
    transform: rotate(90deg);
    color: var(--accent-bright);
  }
  .grp-label {
    flex: 1;
    min-width: 0;
  }
  .grp-count {
    font-size: 9.5px;
    font-weight: 600;
    color: var(--text-dim);
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0 6px;
    line-height: 16px;
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
    padding: 7px 8px;
    color: var(--text-body);
    cursor: pointer;
    font-family: inherit;
  }
  .cat-item:hover:not(:disabled),
  .cat-item.active:not(:disabled),
  .cat-item.flyout-open:not(:disabled) {
    background: var(--bg-surface-selected);
    border-color: var(--border-strong);
  }
  .cat-item:disabled,
  .flyout-item:disabled {
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
    font-size: 12.5px;
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
  .sub-trigger {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 1px 6px 1px 7px;
    flex-shrink: 0;
    white-space: nowrap;
  }
  .cat-item.flyout-open .sub-trigger {
    border-color: var(--accent);
    color: var(--accent-soft);
  }
  .chev {
    font-size: 12px;
    line-height: 1;
    font-weight: 700;
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

  /* Right-hand hover flyout with invisible hover bridge */
  .catalog-flyout {
    position: absolute;
    z-index: 46;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.42), var(--node-shadow);
    display: flex;
    flex-direction: column;
    overflow: visible;
  }
  .catalog-flyout::before {
    content: '';
    position: absolute;
    top: -12px;
    bottom: -12px;
    left: -14px;
    width: 14px;
  }
  .catalog-flyout.side-left::before {
    left: auto;
    right: -14px;
  }
  .catalog-flyout.side-overlay::before {
    display: none;
  }
  .flyout-hdr {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-surface-2);
    border-radius: 9px 9px 0 0;
  }
  .flyout-back {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 5px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-size: 11px;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    flex-shrink: 0;
    white-space: nowrap;
  }
  .flyout-back:hover {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }
  .flyout-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--text-primary);
  }
  .flyout-sub {
    margin-left: auto;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
  }
  .flyout-list {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 6px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .flyout-foot {
    padding: 6px 10px;
    border-top: 1px solid var(--border-subtle);
    background: var(--bg-surface-2);
    font-size: 10px;
    color: var(--text-dim);
    border-radius: 0 0 9px 9px;
  }
  .flyout-item {
    display: flex;
    align-items: flex-start;
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
  .flyout-item:hover:not(:disabled),
  .flyout-item.active:not(:disabled) {
    background: var(--bg-surface-selected);
    border-color: var(--border-strong);
  }
  .sub-main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .sub-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
  }
  .sub-label {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.25;
  }
  .sub-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px;
    color: var(--text-muted);
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 5px;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .sub-badge.warn {
    color: var(--warn-text);
    background: var(--warn-bg);
    border-color: var(--warn-border);
  }
  .sub-desc {
    font-size: 10.5px;
    color: var(--text-muted);
    line-height: 1.3;
  }
  .sub-meta {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    color: var(--text-dim);
  }
</style>
