<script lang="ts">
  import { onDestroy } from 'svelte';
  import {
    expandResidueTokenList,
    formatMultiChainFasta,
    formatOrderReadyAa,
    formatResidueToken,
    toggleResidueTokens,
  } from '$lib/engine/residue-tokens';
  import type {
    FastaPreviewEntry,
    NodeInputPreview,
    PreviewChainTrack,
    PreviewResidue,
    ResidueRole,
    StudioBrushRole,
  } from '$lib/engine/studio-types';
  import type { AtomGranularity } from '$lib/engine/types';

  export interface StudioBrushOption {
    fieldKey: string;
    role: StudioBrushRole;
    label: string;
  }

  interface Props {
    chains: PreviewChainTrack[];
    fastaEntries?: FastaPreviewEntry[];
    entities?: NodeInputPreview['entities'];
    activeBrush?: StudioBrushOption | null;
    availableBrushes?: StudioBrushOption[];
    currentParamValue?: string;
    omitAminoAcids?: string;
    showOmitStrip?: boolean;
    hoveredToken?: string | null;
    atomGranularity?: AtomGranularity;
    onSelectBrush?: (fieldKey: string) => void;
    onUpdateField?: (fieldKey: string, nextValue: string) => void;
    onUpdateOmitAa?: (nextOmit: string) => void;
    onHoverToken?: (token: string | null) => void;
    onFocusToken?: (token: string) => void;
  }

  let {
    chains,
    fastaEntries = [],
    entities = undefined,
    activeBrush = null,
    availableBrushes = [],
    currentParamValue = '',
    omitAminoAcids = '',
    showOmitStrip = false,
    hoveredToken = null,
    atomGranularity = undefined,
    onSelectBrush = undefined,
    onUpdateField = undefined,
    onUpdateOmitAa = undefined,
    onHoverToken = undefined,
    onFocusToken = undefined,
  }: Props = $props();

  type ColorMode = 'role' | 'plddt' | 'physicochemical' | 'diff';
  type ViewMode = 'annotated' | 'alignment';

  const CHUNK_SIZE = 50;
  const WINDOW_THRESHOLD_RESIDUES = 400;
  const ESTIMATED_ROW_HEIGHT = 64;
  const WINDOW_OVERSCAN = 4;
  const CANONICAL_AAS = 'ACDEFGHIKLMNPQRSTVWY'.split('');

  const BRUSH_META: Record<StudioBrushRole, { icon: string; color: string; hint: string }> = {
    fixed: { icon: '🔒', color: '#3b82f6', hint: 'Lock residues from redesign' },
    hotspot: { icon: '🔥', color: '#f97316', hint: 'Mark target binding hotspots' },
    motif: { icon: '🧩', color: '#a855f7', hint: 'Mark fixed motif scaffold spans' },
    tied: { icon: '🔗', color: '#14b8a6', hint: 'Link symmetry-tied positions' },
    crop: { icon: '✂️', color: '#eab308', hint: 'Select kept crop segments' },
  };

  let colorMode = $state<ColorMode>('role');
  let viewMode = $state<ViewMode>('annotated');
  let copiedKey = $state<string | null>(null);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;

  // Local hover token + floating tooltip position (resolve residue live from residueByToken
  // so locking/unlocking a residue updates its tooltip roles immediately without pointer exit)
  let localHoveredToken = $state<string | null>(null);
  let tooltipPos = $state<{ x: number; y: number } | null>(null);

  // Drag-paint state
  let isDragging = $state(false);
  let dragChainIdx = $state(-1);
  let dragStartIdx = $state(-1);
  let dragEndIdx = $state(-1);
  let dragMode = $state<'add' | 'remove'>('add');

  // Simple row windowing state (activated when total residues > 400)
  let scrollEl: HTMLDivElement | null = $state(null);
  let scrollTop = $state(0);
  let viewportHeight = $state(360);

  let hasMultiFasta = $derived(fastaEntries.length > 1);
  $effect(() => {
    if (!hasMultiFasta && viewMode === 'alignment') {
      viewMode = 'annotated';
    }
  });

  // Total residues across all chains
  let totalResidues = $derived(
    chains.reduce((acc, ch) => acc + (ch.residues.length || ch.sequence.length), 0),
  );
  let useWindowing = $derived(totalResidues > WINDOW_THRESHOLD_RESIDUES);

  // Normalized per-chain residue arrays (fallback synthesis if residues[] is empty while sequence is non-empty)
  interface NormalizedChain {
    chain: PreviewChainTrack;
    residues: PreviewResidue[];
    rows: PreviewResidue[][];
    refSequence: string;
  }

  let normalizedChains = $derived.by((): NormalizedChain[] => {
    // Reference sequence split by chain for "Diff vs Ref" mode when fastaEntries[0] is available
    const refEntrySeq = fastaEntries[0]?.sequence ?? '';
    const refParts = refEntrySeq ? refEntrySeq.split(/[:/]/) : [];

    return chains.map((chain, cIdx) => {
      const residues: PreviewResidue[] =
        chain.residues.length > 0
          ? chain.residues
          : chain.sequence.split('').map((aa, i) => ({
              flatIndex: i,
              authChain: chain.chainId,
              labelChain: chain.labelChainId,
              authSeqId: i + 1,
              insCode: '',
              labelSeqId: i + 1,
              token: formatResidueToken(chain.chainId, i + 1),
              aa1: aa,
              resName: aa,
              roles: ['designable'] as ResidueRole[],
            }));

      const rows: PreviewResidue[][] = [];
      for (let i = 0; i < residues.length; i += CHUNK_SIZE) {
        rows.push(residues.slice(i, i + CHUNK_SIZE));
      }

      const refSequence = refParts[cIdx] ?? (chains.length === 1 ? refEntrySeq : '') ?? '';
      return { chain, residues, rows, refSequence };
    });
  });

  // Flat lookup map for quick hover readout from externally hoveredToken (e.g. 3D viewer hover)
  let residueByToken = $derived.by(() => {
    const map = new Map<string, PreviewResidue>();
    for (const nc of normalizedChains) {
      for (const r of nc.residues) {
        map.set(r.token, r);
      }
    }
    return map;
  });

  let localHoveredRes = $derived<PreviewResidue | null>(
    localHoveredToken ? (residueByToken.get(localHoveredToken) ?? null) : null,
  );

  let activeTooltipRes = $derived.by((): PreviewResidue | null => {
    const tok = localHoveredToken ?? hoveredToken;
    return tok ? (residueByToken.get(tok) ?? null) : null;
  });

  // Flat virtualized row items when windowing is enabled
  type FlatRowItem =
    | { kind: 'chain-header'; key: string; chainIdx: number }
    | { kind: 'seq-row'; key: string; chainIdx: number; rowIdx: number; startOffset: number; residues: PreviewResidue[] };

  let flatRows = $derived.by((): FlatRowItem[] => {
    const out: FlatRowItem[] = [];
    for (let cIdx = 0; cIdx < normalizedChains.length; cIdx++) {
      const nc = normalizedChains[cIdx];
      out.push({ kind: 'chain-header', key: `hdr:${nc.chain.chainId}:${cIdx}`, chainIdx: cIdx });
      for (let rIdx = 0; rIdx < nc.rows.length; rIdx++) {
        out.push({
          kind: 'seq-row',
          key: `row:${nc.chain.chainId}:${cIdx}:${rIdx}`,
          chainIdx: cIdx,
          rowIdx: rIdx,
          startOffset: rIdx * CHUNK_SIZE,
          residues: nc.rows[rIdx],
        });
      }
    }
    return out;
  });

  let visibleWindow = $derived.by(() => {
    if (!useWindowing) {
      return { start: 0, end: flatRows.length, topPad: 0, bottomPad: 0 };
    }
    const start = Math.max(0, Math.floor(scrollTop / ESTIMATED_ROW_HEIGHT) - WINDOW_OVERSCAN);
    const visibleCount = Math.ceil(viewportHeight / ESTIMATED_ROW_HEIGHT) + WINDOW_OVERSCAN * 2;
    const end = Math.min(flatRows.length, start + visibleCount);
    const topPad = start * ESTIMATED_ROW_HEIGHT;
    const bottomPad = Math.max(0, (flatRows.length - end) * ESTIMATED_ROW_HEIGHT);
    return { start, end, topPad, bottomPad };
  });

  // Current active brush selected token set (for live preview and initial drag-mode detection)
  let currentBrushTokenSet = $derived.by((): Set<string> => {
    const set = new Set<string>();
    if (!activeBrush) return set;
    const raw = String(currentParamValue ?? '').trim();
    if (!raw) return set;
    if (activeBrush.role === 'tied') {
      for (const grp of raw.split(';')) {
        for (const tok of grp.split(/[=,\s]+/).filter(Boolean)) {
          const parsed = expandResidueTokenList(tok, chains[0]?.chainId ?? 'A');
          for (const t of parsed.tokens) set.add(t.raw);
        }
      }
      return set;
    }
    const parsed = expandResidueTokenList(raw, chains[0]?.chainId ?? 'A');
    for (const t of parsed.tokens) set.add(t.raw);
    for (const ch of parsed.wholeChains) {
      const nc = normalizedChains.find((x) => x.chain.chainId.toUpperCase() === ch);
      if (nc) {
        for (const r of nc.residues) set.add(r.token);
      }
    }
    return set;
  });

  let dragPreviewTokens = $derived.by((): Set<string> => {
    const set = new Set<string>();
    if (!isDragging || dragChainIdx < 0 || dragStartIdx < 0 || dragEndIdx < 0) return set;
    const nc = normalizedChains[dragChainIdx];
    if (!nc) return set;
    const lo = Math.min(dragStartIdx, dragEndIdx);
    const hi = Math.max(dragStartIdx, dragEndIdx);
    for (let i = lo; i <= hi; i++) {
      const r = nc.residues[i];
      if (r) set.add(r.token);
    }
    return set;
  });

  // Omit AA set
  let omitAaSet = $derived.by((): Set<string> => {
    const set = new Set<string>();
    const raw = String(omitAminoAcids ?? '').toUpperCase().replace(/[^A-Z]/g, '');
    for (const ch of raw) set.add(ch);
    return set;
  });

  function toggleOmitAa(aa: string) {
    if (!onUpdateOmitAa) return;
    const next = new Set(omitAaSet);
    if (next.has(aa)) next.delete(aa);
    else next.add(aa);
    const ordered = CANONICAL_AAS.filter((c) => next.has(c)).join('');
    onUpdateOmitAa(ordered);
  }

  function clearOmitAa() {
    onUpdateOmitAa?.('');
  }

  // Binder / non-target chains for "Copy Binder-Only AA"
  let binderChains = $derived.by((): PreviewChainTrack[] => {
    const proteinChains = chains.filter((c) => c.entityKind === 'protein' && c.sequence.trim().length > 0);
    const explicitBinders = proteinChains.filter(
      (c) => c.roleTag === 'binder' || c.roleTag === 'designed-span',
    );
    if (explicitBinders.length > 0) return explicitBinders;
    const hasTarget = proteinChains.some((c) => c.roleTag === 'target');
    if (hasTarget) {
      return proteinChains.filter((c) => c.roleTag !== 'target');
    }
    return [];
  });

  let hasEntities = $derived(
    Boolean(
      entities &&
        (entities.includedLigands.length > 0 ||
          entities.filteredBuffers.length > 0 ||
          entities.nucleicChains.length > 0),
    ),
  );

  // Copy helpers
  async function copyText(text: string, key: string) {
    if (copyTimer) clearTimeout(copyTimer);
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(text);
      copiedKey = key;
      copyTimer = setTimeout(() => {
        copiedKey = null;
      }, 1500);
    } catch {
      copiedKey = null;
    }
  }

  function buildFullFasta(): string {
    if (fastaEntries.length > 0) {
      return fastaEntries
        .map((e) => {
          const hdr = e.header.startsWith('>') ? e.header : `>${e.header}`;
          const wrapped = e.sequence.replace(/(.{1,60})/g, '$1\n').trimEnd();
          return `${hdr}\n${wrapped}`;
        })
        .join('\n');
    }
    return formatMultiChainFasta(
      'denovo_design',
      chains.map((c) => ({ chainId: c.chainId, roleTag: c.roleTag, sequence: c.sequence })),
    );
  }

  function buildOrderReadyAa(): string {
    const targetChains =
      binderChains.length > 0
        ? binderChains
        : chains.filter((c) => c.entityKind === 'protein');
    return targetChains
      .map((c) => formatOrderReadyAa(c.sequence))
      .filter(Boolean)
      .join('\n');
  }

  function buildBinderOnlyAa(): string {
    return binderChains
      .map((c) => formatOrderReadyAa(c.sequence))
      .filter(Boolean)
      .join('\n');
  }

  function downloadFasta() {
    const text = buildFullFasta();
    if (!text.trim()) return;
    const blob = new Blob([`${text}\n`], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'denovo_sequences.fasta';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // Tied positions toggle helper (supports `A10=B10; A11=B11`)
  function toggleTiedPositions(
    existingSpec: string,
    selected: PreviewResidue[],
    mode: 'add' | 'remove',
  ): string {
    const groups = existingSpec
      .split(';')
      .map((g) => g.trim())
      .filter(Boolean)
      .map((g) => g.split(/[=,\s]+/).map((t) => t.trim()).filter(Boolean));
    const selTokens = selected.map((r) => formatResidueToken(r.authChain, r.authSeqId, r.insCode));
    if (mode === 'remove') {
      const rmSet = new Set(selTokens);
      const nextGroups = groups
        .map((g) => g.filter((t) => !rmSet.has(t)))
        .filter((g) => g.length > 0);
      return nextGroups.map((g) => g.join('=')).join('; ');
    }
    // Add into last group if last group has < 2 chains or append as new group
    const nextGroups = groups.map((g) => [...g]);
    if (nextGroups.length === 0) {
      nextGroups.push([...selTokens]);
    } else {
      const last = nextGroups[nextGroups.length - 1];
      const existingSet = new Set(last);
      for (const tok of selTokens) {
        if (!existingSet.has(tok)) {
          last.push(tok);
          existingSet.add(tok);
        }
      }
    }
    return nextGroups.map((g) => g.join('=')).join('; ');
  }

  function commitDragSelection() {
    if (!isDragging || !activeBrush || !onUpdateField) {
      isDragging = false;
      return;
    }
    const nc = normalizedChains[dragChainIdx];
    if (nc && dragStartIdx >= 0 && dragEndIdx >= 0) {
      const lo = Math.min(dragStartIdx, dragEndIdx);
      const hi = Math.max(dragStartIdx, dragEndIdx);
      const selected = nc.residues.slice(lo, hi + 1);
      if (selected.length > 0) {
        const nextVal =
          activeBrush.role === 'tied'
            ? toggleTiedPositions(currentParamValue ?? '', selected, dragMode)
            : toggleResidueTokens(
                currentParamValue ?? '',
                selected,
                dragMode,
                nc.chain.chainId,
              );
        onUpdateField(activeBrush.fieldKey, nextVal);
      }
    }
    isDragging = false;
    dragChainIdx = -1;
    dragStartIdx = -1;
    dragEndIdx = -1;
  }

  function findCellMeta(target: EventTarget | null): {
    chainIdx: number;
    resIdx: number;
    token: string;
    res: PreviewResidue;
  } | null {
    if (!(target instanceof Element)) return null;
    const cell = target.closest('[data-token]') as HTMLElement | null;
    if (!cell) return null;
    const chainIdx = Number(cell.dataset.chainIdx);
    const resIdx = Number(cell.dataset.resIdx);
    const token = cell.dataset.token ?? '';
    const res = normalizedChains[chainIdx]?.residues[resIdx];
    if (!res) return null;
    return { chainIdx, resIdx, token, res };
  }

  function handleTrackPointerDown(ev: PointerEvent) {
    if (ev.button !== 0) return;
    const hit = findCellMeta(ev.target);
    if (!hit) return;

    // Alt+click or no active brush -> focus 3D camera on residue
    if (ev.altKey || !activeBrush || !onUpdateField) {
      onFocusToken?.(hit.token);
      return;
    }

    ev.preventDefault();
    isDragging = true;
    dragChainIdx = hit.chainIdx;
    dragStartIdx = hit.resIdx;
    dragEndIdx = hit.resIdx;
    dragMode = currentBrushTokenSet.has(hit.token) ? 'remove' : 'add';
  }

  function handleTrackPointerMove(ev: PointerEvent) {
    const hit = findCellMeta(ev.target);
    if (hit) {
      if (localHoveredToken !== hit.res.token) {
        localHoveredToken = hit.res.token;
        onHoverToken?.(hit.token);
      }
      tooltipPos = { x: ev.clientX, y: ev.clientY };
      if (isDragging && hit.chainIdx === dragChainIdx) {
        if (hit.resIdx !== dragEndIdx) {
          dragEndIdx = hit.resIdx;
        }
      }
    } else if (localHoveredToken) {
      localHoveredToken = null;
      tooltipPos = null;
      onHoverToken?.(null);
    }
  }

  function handleTrackPointerLeave() {
    if (localHoveredToken) {
      localHoveredToken = null;
      tooltipPos = null;
      onHoverToken?.(null);
    }
  }

  function handleTrackDblClick(ev: MouseEvent) {
    const hit = findCellMeta(ev.target);
    if (hit) {
      onFocusToken?.(hit.token);
    }
  }

  function handleWindowPointerUp() {
    if (isDragging) {
      commitDragSelection();
    }
  }

  function handleScrollContainer() {
    if (!scrollEl) return;
    scrollTop = scrollEl.scrollTop;
    viewportHeight = scrollEl.clientHeight || 360;
  }

  // Color resolvers
  function getPlddtColor(bfactor: number | undefined): { bg: string; fg: string } | null {
    if (bfactor === undefined || !Number.isFinite(bfactor)) return null;
    const score = bfactor <= 1.05 && bfactor >= 0 ? bfactor * 100 : bfactor;
    if (score >= 90) return { bg: 'rgba(0, 83, 214, 0.28)', fg: '#60a5fa' };
    if (score >= 70) return { bg: 'rgba(101, 203, 243, 0.24)', fg: '#38bdf8' };
    if (score >= 50) return { bg: 'rgba(255, 219, 19, 0.24)', fg: '#eab308' };
    return { bg: 'rgba(255, 125, 69, 0.26)', fg: '#fb923c' };
  }

  function getPhysicochemColor(aa: string): { bg: string; fg: string } {
    const u = aa.toUpperCase();
    if ('KRH'.includes(u)) return { bg: 'rgba(59, 130, 246, 0.22)', fg: '#60a5fa' }; // basic (+)
    if ('DE'.includes(u)) return { bg: 'rgba(239, 68, 68, 0.22)', fg: '#f87171' }; // acidic (-)
    if ('STNQ'.includes(u)) return { bg: 'rgba(16, 185, 129, 0.20)', fg: '#34d399' }; // polar
    if ('AVILMFWY'.includes(u)) return { bg: 'rgba(245, 158, 11, 0.18)', fg: '#fbbf24' }; // hydrophobic
    if (u === 'C') return { bg: 'rgba(234, 179, 8, 0.30)', fg: '#facc15' }; // cysteine
    if (u === 'P') return { bg: 'rgba(168, 85, 247, 0.22)', fg: '#c084fc' }; // proline
    return { bg: 'transparent', fg: 'var(--text-secondary)' }; // G / X / -
  }

  function getRoleClass(res: PreviewResidue): string {
    const r = res.roles;
    if (r.includes('cropped-out')) return 'role-cropped';
    if (r.includes('missing-backbone')) return 'role-missing';
    if (r.includes('hotspot')) return 'role-hotspot';
    if (r.includes('motif') || r.includes('unindexed-motif')) return 'role-motif';
    if (r.includes('tied')) return 'role-tied';
    if (r.includes('fixed-sequence') || r.includes('fixed-backbone')) {
      return res.inheritedLock ? 'role-fixed-inherited' : 'role-fixed';
    }
    if (r.includes('ligand')) return 'role-ligand';
    if (r.includes('nucleic')) return 'role-nucleic';
    return 'role-designable';
  }

  function getCellInlineStyle(res: PreviewResidue, resIdx: number, refSeq: string): string {
    if (colorMode === 'plddt') {
      const c = getPlddtColor(res.bfactor);
      return c ? `background:${c.bg};color:${c.fg};` : '';
    }
    if (colorMode === 'physicochemical') {
      const c = getPhysicochemColor(res.aa1);
      return `background:${c.bg};color:${c.fg};`;
    }
    if (colorMode === 'diff') {
      const refAa = refSeq[resIdx];
      if (refAa && refAa !== res.aa1) {
        return 'background:rgba(249,115,22,0.28);color:#fb923c;font-weight:700;';
      }
      return 'color:var(--text-dim);';
    }
    return '';
  }

  function getRoleGlyph(res: PreviewResidue): { glyph: string; kind: string; title: string } | null {
    const r = res.roles;
    if (r.includes('cropped-out')) {
      return { glyph: '✂', kind: 'cropped', title: 'Cropped out' };
    }
    if (r.includes('hotspot')) {
      return { glyph: '🔥', kind: 'hotspot', title: 'Binding hotspot' };
    }
    if (r.includes('motif') || r.includes('unindexed-motif')) {
      return {
        glyph: '🧩',
        kind: 'motif',
        title: r.includes('unindexed-motif') ? 'Unindexed motif' : 'Fixed motif',
      };
    }
    if (r.includes('tied')) {
      return {
        glyph: res.tieGroupId ? `T${res.tieGroupId}` : '🔗',
        kind: 'tied',
        title: `Symmetry tie group${res.tieGroupId ? ` T${res.tieGroupId}` : ''}`,
      };
    }
    if (r.includes('fixed-sequence') || r.includes('fixed-backbone')) {
      return {
        glyph: '🔒',
        kind: res.inheritedLock ? 'lock-inherited' : 'lock-user',
        title: res.inheritedLock ? 'Inherited lock from upstream' : 'User-locked fixed residue',
      };
    }
    if (r.includes('missing-backbone')) {
      return { glyph: '!', kind: 'missing', title: 'Missing backbone atoms' };
    }
    return null;
  }

  function computeIdentityPct(refSeq: string, seq: string): number {
    const len = Math.min(refSeq.length, seq.length);
    if (len === 0) return 0;
    let same = 0;
    for (let i = 0; i < len; i++) {
      if (refSeq[i] === seq[i]) same++;
    }
    return Math.round((same / len) * 100);
  }

  onDestroy(() => {
    if (copyTimer) clearTimeout(copyTimer);
  });
</script>

<svelte:window onpointerup={handleWindowPointerUp} />

<div class="afv-root">
  <!-- Top Controls Bar: View tabs, Brushes, Color Mode, Granularity -->
  <div class="afv-toolbar">
    {#if hasMultiFasta}
      <div class="seg-group" role="tablist" aria-label="Sequence view mode">
        <button
          type="button"
          role="tab"
          aria-selected={viewMode === 'annotated'}
          class:active={viewMode === 'annotated'}
          onclick={() => (viewMode = 'annotated')}
        >
          Annotated
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={viewMode === 'alignment'}
          class:active={viewMode === 'alignment'}
          onclick={() => (viewMode = 'alignment')}
        >
          Candidates ({fastaEntries.length})
        </button>
      </div>
    {/if}

    {#if viewMode === 'annotated'}
      <div class="seg-group" role="group" aria-label="Coloring mode">
        <button
          type="button"
          class:active={colorMode === 'role'}
          onclick={() => (colorMode = 'role')}
          title="Color by design role (locked, hotspot, motif, tied, cropped)"
        >
          Role
        </button>
        <button
          type="button"
          class:active={colorMode === 'plddt'}
          onclick={() => (colorMode = 'plddt')}
          title="Color by pLDDT / B-factor"
        >
          pLDDT
        </button>
        <button
          type="button"
          class:active={colorMode === 'physicochemical'}
          onclick={() => (colorMode = 'physicochemical')}
          title="Color by amino acid physicochemical properties"
        >
          Physicochem
        </button>
        <button
          type="button"
          class:active={colorMode === 'diff'}
          onclick={() => (colorMode = 'diff')}
          title="Highlight residue differences vs reference sequence"
        >
          Diff vs Ref
        </button>
      </div>
    {/if}

    {#if atomGranularity === 'ca-only'}
      <span
        class="gran-pill"
        title="Only Cα coordinates are available in this backbone preview."
      >
        Cα-only
      </span>
    {/if}

    <!-- Copy / Export bar -->
    <div class="export-bar">
      <button
        type="button"
        class="exp-btn"
        onclick={() => void copyText(buildFullFasta(), 'fasta')}
        title="Copy multi-chain FASTA to clipboard"
      >
        {copiedKey === 'fasta' ? '✓ Copied' : 'Copy FASTA'}
      </button>
      {#if binderChains.length > 0}
        <button
          type="button"
          class="exp-btn accent"
          onclick={() => void copyText(buildBinderOnlyAa(), 'binder')}
          title="Copy clean binder / designed chain sequence only"
        >
          {copiedKey === 'binder' ? '✓ Copied' : 'Copy Binder-Only AA'}
        </button>
      {/if}
      <button
        type="button"
        class="exp-btn"
        onclick={() => void copyText(buildOrderReadyAa(), 'order')}
        title="Copy clean 20-AA gene-synthesis order-ready sequence"
      >
        {copiedKey === 'order' ? '✓ Copied' : 'Copy Order-Ready AA'}
      </button>
      <button
        type="button"
        class="exp-btn"
        onclick={downloadFasta}
        title="Download FASTA file"
      >
        ↓ .fasta
      </button>
    </div>
  </div>

  <!-- Interactive Residue Selection Bar (shown only when a sequence is loaded) -->
  {#if viewMode === 'annotated' && chains.length > 0 && availableBrushes.length > 0}
    <div class="brush-bar" role="toolbar" aria-label="Residue selection controls">
      <span class="brush-lbl">Click/drag to set:</span>
      {#each availableBrushes as b (b.fieldKey)}
        {@const meta = BRUSH_META[b.role]}
        {@const isActive = activeBrush?.fieldKey === b.fieldKey}
        <button
          type="button"
          class="brush-chip"
          class:active={isActive}
          style="--brush-color: {meta.color}"
          title={meta.hint}
          onclick={() => onSelectBrush?.(b.fieldKey)}
        >
          <span class="b-ico" aria-hidden="true">{meta.icon}</span>
          <span>{b.label}</span>
        </button>
      {/each}
      <span class="brush-hint">
        Click or drag residues to toggle · Double-click / Alt-click focuses 3D
      </span>
    </div>
  {/if}

  <!-- Optional inline 20-AA Omit Strip -->
  {#if viewMode === 'annotated' && showOmitStrip}
    <div class="omit-strip" role="group" aria-label="Omit amino acids from design">
      <span class="omit-lbl">Omit AA:</span>
      <button
        type="button"
        class="omit-quick"
        class:active={omitAaSet.has('C')}
        onclick={() => toggleOmitAa('C')}
        title="Exclude Cysteine (C) to prevent unwanted disulfide formation"
      >
        No Cys
      </button>
      <div class="omit-grid">
        {#each CANONICAL_AAS as aa (aa)}
          <button
            type="button"
            class="aa-btn"
            class:omitted={omitAaSet.has(aa)}
            onclick={() => toggleOmitAa(aa)}
            title={omitAaSet.has(aa) ? `Allow ${aa}` : `Omit ${aa} from sampling`}
          >
            {aa}
          </button>
        {/each}
      </div>
      {#if omitAaSet.size > 0}
        <button type="button" class="omit-clear" onclick={clearOmitAa} title="Allow all 20 amino acids">
          Clear
        </button>
      {/if}
    </div>
  {/if}

  <!-- Non-protein Entity Pills (Retained ligands vs Stripped buffers + Nucleic) -->
  {#if hasEntities && entities}
    <div class="entity-bar">
      {#if entities.includedLigands.length > 0}
        <div class="ent-group retained">
          <span class="ent-lbl">Retained ligands:</span>
          {#each entities.includedLigands as lig (lig)}
            <span class="ent-pill retained">{lig}</span>
          {/each}
        </div>
      {/if}
      {#if entities.nucleicChains.length > 0}
        <div class="ent-group nucleic">
          <span class="ent-lbl">Nucleic:</span>
          {#each entities.nucleicChains as na (na)}
            <span class="ent-pill nucleic">{na}</span>
          {/each}
        </div>
      {/if}
      {#if entities.filteredBuffers.length > 0}
        <div class="ent-group stripped">
          <span class="ent-lbl">Stripped buffers:</span>
          {#each entities.filteredBuffers as buf (buf)}
            <span class="ent-pill stripped" title="Filtered out as crystallization solvent/buffer">{buf}</span>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <!-- MAIN CONTENT: Annotated Sequence View OR Multi-Candidate Alignment Tab -->
  {#if viewMode === 'alignment' && hasMultiFasta}
    {@const refEntry = fastaEntries[0]}
    <div class="align-wrap">
      <div class="align-legend">
        <span>Reference: <b>{refEntry.header.replace(/^>/, '')}</b> ({refEntry.sequence.length} aa)</span>
        <span class="align-hint">Dots (<code>.</code>) match candidate #1 · highlighted letters show mutations</span>
      </div>
      <div class="align-scroll">
        {#each fastaEntries as entry, eIdx (entry.id)}
          {@const idPct = eIdx === 0 ? 100 : computeIdentityPct(refEntry.sequence, entry.sequence)}
          {@const rawHdr = entry.header.replace(/^>/, '')}
          {@const cleanName = entry.scoreLabel && rawHdr.includes(' | ') ? rawHdr.split(' | ')[0] : rawHdr}
          <div class="align-row" class:is-selected={entry.selected}>
            <div class="align-meta">
              <div class="align-hdr" title={entry.header}>
                <span class="align-idx">#{eIdx + 1}</span>
                <span class="align-name">{cleanName}</span>
                {#if entry.scoreLabel}
                  <span class="align-score">{entry.scoreLabel}</span>
                {/if}
                <span class="align-id">{idPct}% id</span>
              </div>
              <button
                type="button"
                class="mini-copy"
                onclick={() => void copyText(formatOrderReadyAa(entry.sequence), `cand-${entry.id}`)}
              >
                {copiedKey === `cand-${entry.id}` ? '✓' : 'Copy'}
              </button>
            </div>
            <div class="align-seq" aria-label={`Sequence ${entry.header}`}>
              {#if eIdx === 0}
                <span class="align-ref-text">{entry.sequence}</span>
              {:else}
                {#each entry.sequence.split('') as ch, pos (pos)}
                  {#if ch === refEntry.sequence[pos]}<span class="al-dot">.</span>{:else}<span class="al-mut" title={`Pos ${pos + 1}: ${refEntry.sequence[pos] ?? '-'} → ${ch}`}>{ch}</span>{/if}
                {/each}
              {/if}
            </div>
          </div>
        {/each}
      </div>
    </div>
  {:else if normalizedChains.length === 0}
    <div class="empty-seq">No sequence or chain tracks available for this block yet.</div>
  {:else}
    <!-- Delegated sequence track container (windowed when > 400 residues) -->
    <div
      bind:this={scrollEl}
      class="seq-scroll"
      class:is-windowed={useWindowing}
      class:can-paint={Boolean(activeBrush && onUpdateField)}
      class:is-dragging={isDragging}
      role="region"
      aria-label="Annotated protein sequence viewer"
      onscroll={useWindowing ? handleScrollContainer : undefined}
      onpointerdown={handleTrackPointerDown}
      onpointermove={handleTrackPointerMove}
      onpointerleave={handleTrackPointerLeave}
      ondblclick={handleTrackDblClick}
    >
      {#if visibleWindow.topPad > 0}
        <div class="seq-spacer" style="height: {visibleWindow.topPad}px; flex-shrink: 0;" aria-hidden="true"></div>
      {/if}

      {#each flatRows.slice(visibleWindow.start, visibleWindow.end) as item (item.key)}
        {#if item.kind === 'chain-header'}
          {@const nc = normalizedChains[item.chainIdx]}
          {@const ch = nc.chain}
          <div class="chain-head">
            <div class="chain-id-row">
              <span class="chain-title">&gt;Chain {ch.chainId}</span>
              {#if ch.labelChainId && ch.labelChainId !== ch.chainId}
                <span class="remap-badge" title={`Remapped to model chain ${ch.labelChainId}`}>
                  → {ch.labelChainId}
                </span>
              {/if}
              <span class="kind-pill kind-{ch.entityKind}">{ch.entityKind}</span>
              {#if ch.roleTag}
                <span class="role-tag role-tag-{ch.roleTag}">{ch.roleTag}</span>
              {/if}
              {#if ch.msaStatus}
                <span class="msa-pill">
                  MSA: {ch.msaStatus}{#if typeof ch.msaDepth === 'number'} ({ch.msaDepth}){/if}
                </span>
              {/if}
              <span class="len-pill">{nc.residues.length} aa</span>
            </div>
          </div>
        {:else}
          {@const nc = normalizedChains[item.chainIdx]}
          <div class="seq-row">
            {#each item.residues as res, localIdx (res.token + ':' + localIdx)}
              {@const absIdx = item.startOffset + localIdx}
              {@const showTick = localIdx % 10 === 0}
              {@const hasGap = (res.gapBefore ?? 0) > 0}
              {@const glyph = getRoleGlyph(res)}
              {@const isHovered = hoveredToken === res.token || localHoveredRes?.token === res.token}
              {@const isDragHit = dragPreviewTokens.has(res.token)}
              {#if hasGap}
                <span
                  class="gap-div"
                  title={`Chain break: ${res.gapBefore} missing residue(s) before ${res.token}`}
                  aria-label={`Gap of ${res.gapBefore} missing residues`}
                >
                  ╱╲
                </span>
              {/if}
              <span
                class="res-cell {colorMode === 'role' ? getRoleClass(res) : ''}"
                class:block-start={localIdx > 0 && localIdx % 10 === 0}
                class:is-hovered={isHovered}
                class:is-drag-add={isDragHit && dragMode === 'add'}
                class:is-drag-rm={isDragHit && dragMode === 'remove'}
                data-chain-idx={item.chainIdx}
                data-res-idx={absIdx}
                data-token={res.token}
              >
                <span class="r-tick">
                  {#if showTick || res.insCode}
                    <span class="tick-num">{res.authSeqId}{#if res.insCode}<u class="ins">{res.insCode}</u>{/if}</span>
                  {/if}
                </span>
                <span
                  class="r-lane"
                  class:lane-inherited={glyph?.kind === 'lock-inherited'}
                  class:lane-user={glyph?.kind === 'lock-user'}
                  class:lane-hotspot={glyph?.kind === 'hotspot'}
                  class:lane-motif={glyph?.kind === 'motif'}
                  class:lane-tied={glyph?.kind === 'tied'}
                  class:lane-cropped={glyph?.kind === 'cropped'}
                  class:lane-missing={glyph?.kind === 'missing'}
                  title={glyph?.title}
                >
                  {glyph ? glyph.glyph : ''}
                </span>
                <span
                  class="r-aa"
                  style={getCellInlineStyle(res, absIdx, nc.refSequence)}
                >
                  {res.aa1}
                </span>
              </span>
            {/each}
          </div>
        {/if}
      {/each}

      {#if visibleWindow.bottomPad > 0}
        <div class="seq-spacer" style="height: {visibleWindow.bottomPad}px; flex-shrink: 0;" aria-hidden="true"></div>
      {/if}
    </div>
  {/if}

  <!-- Persistent compact 3-coordinate status readout bar (always legible & screen-reader friendly) -->
  <div class="coord-footer" aria-live="polite">
    {#if activeTooltipRes}
      <span><b>Auth:</b> {activeTooltipRes.authChain}{activeTooltipRes.authSeqId}{activeTooltipRes.insCode}</span>
      <span><b>Label:</b> {activeTooltipRes.labelChain ?? activeTooltipRes.authChain}{activeTooltipRes.labelSeqId ?? (activeTooltipRes.flatIndex >= 0 ? activeTooltipRes.flatIndex + 1 : '—')}</span>
      <span><b>Model idx:</b> {activeTooltipRes.flatIndex >= 0 ? activeTooltipRes.flatIndex : 'cropped'}</span>
      <span><b>Res:</b> {activeTooltipRes.resName} ({activeTooltipRes.aa1})</span>
      <span><b>Roles:</b> {activeTooltipRes.roles.join(', ') || 'designable'}</span>
      {#if activeTooltipRes.backboneAtoms}
        <span><b>Backbone:</b> {activeTooltipRes.backboneAtoms}</span>
      {/if}
      {#if activeTooltipRes.bfactor !== undefined}
        <span><b>B/pLDDT:</b> {activeTooltipRes.bfactor.toFixed(1)}</span>
      {/if}
    {:else}
      <span class="coord-muted">
        Hover any residue to inspect Auth / Label / Model coordinates · {totalResidues} total residue{totalResidues === 1 ? '' : 's'}
      </span>
    {/if}
  </div>

  <!-- Floating hover tooltip near pointer -->
  {#if localHoveredRes && tooltipPos && !isDragging}
    <div
      class="float-tip"
      style="left: {Math.min(tooltipPos.x + 14, (typeof window !== 'undefined' ? window.innerWidth : 1200) - 245)}px; top: {Math.max(8, tooltipPos.y - 96)}px;"
      role="tooltip"
    >
      <div class="ft-head">
        <strong>{localHoveredRes.resName} ({localHoveredRes.aa1})</strong>
        <span class="ft-tok">{localHoveredRes.token}</span>
      </div>
      <div class="ft-grid">
        <span>Auth</span><span>{localHoveredRes.authChain}{localHoveredRes.authSeqId}{localHoveredRes.insCode}</span>
        <span>Label</span><span>{localHoveredRes.labelChain ?? localHoveredRes.authChain}{localHoveredRes.labelSeqId ?? (localHoveredRes.flatIndex >= 0 ? localHoveredRes.flatIndex + 1 : '—')}</span>
        <span>Model idx</span><span>{localHoveredRes.flatIndex >= 0 ? localHoveredRes.flatIndex : 'cropped'}</span>
        <span>Roles</span><span>{localHoveredRes.roles.join(', ') || 'designable'}</span>
        {#if localHoveredRes.backboneAtoms}
          <span>Backbone</span><span>{localHoveredRes.backboneAtoms}</span>
        {/if}
        {#if localHoveredRes.bfactor !== undefined}
          <span>B / pLDDT</span><span>{localHoveredRes.bfactor.toFixed(1)}</span>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .afv-root {
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 11px;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    min-width: 0;
  }
  .afv-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }
  .seg-group {
    display: inline-flex;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    overflow: hidden;
    background: var(--bg-surface-2);
  }
  .seg-group button {
    border: none;
    background: transparent;
    color: var(--text-muted);
    font-size: 10.5px;
    font-weight: 600;
    padding: 3px 8px;
    cursor: pointer;
    font-family: inherit;
    white-space: nowrap;
  }
  .seg-group button:hover {
    color: var(--text-primary);
    background: var(--bg-surface-hover);
  }
  .seg-group button.active {
    background: var(--accent);
    color: #fff;
  }
  .gran-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 999px;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
  }
  .export-bar {
    margin-left: auto;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .exp-btn {
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 3px 7px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  .exp-btn:hover {
    border-color: var(--accent);
    color: var(--text-primary);
  }
  .exp-btn.accent {
    background: var(--bg-pill);
    color: var(--accent-soft);
    border-color: var(--accent);
  }
  .brush-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    padding: 6px 8px;
    border-radius: 6px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
  }
  .brush-lbl {
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-muted);
    margin-right: 2px;
  }
  .brush-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-secondary);
    font-size: 10.5px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
  }
  .brush-chip:hover {
    border-color: var(--brush-color);
    color: var(--text-primary);
  }
  .brush-chip.active {
    border-color: var(--brush-color);
    background: color-mix(in srgb, var(--brush-color) 18%, var(--bg-surface));
    color: var(--text-primary);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--brush-color) 45%, transparent);
  }
  .b-ico {
    font-size: 10.5px;
    line-height: 1;
  }
  .brush-hint {
    margin-left: auto;
    font-size: 10px;
    color: var(--text-dim);
  }
  .omit-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    padding: 5px 8px;
    border-radius: 6px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
  }
  .omit-lbl {
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-muted);
  }
  .omit-quick {
    padding: 2px 7px;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface);
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
  }
  .omit-quick.active {
    background: var(--warn-bg);
    border-color: var(--warn-border);
    color: var(--warn-text);
  }
  .omit-grid {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 2px;
  }
  .aa-btn {
    width: 21px;
    height: 21px;
    padding: 0;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .aa-btn:hover {
    border-color: var(--accent);
    color: var(--text-primary);
  }
  .aa-btn.omitted {
    background: var(--err-bg);
    border-color: var(--err-border);
    color: var(--err-strong);
    text-decoration: line-through;
  }
  .omit-clear {
    background: none;
    border: none;
    color: var(--text-dim);
    font-size: 10px;
    cursor: pointer;
    text-decoration: underline;
    padding: 2px 4px;
  }
  .omit-clear:hover {
    color: var(--text-primary);
  }
  .entity-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    font-size: 10.5px;
  }
  .ent-group {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
  }
  .ent-lbl {
    color: var(--text-dim);
    font-weight: 600;
  }
  .ent-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 999px;
    font-weight: 600;
  }
  .ent-pill.retained {
    background: var(--ok-bg);
    color: var(--ok-text);
    border: 1px solid var(--ok-border);
  }
  .ent-pill.nucleic {
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
  }
  .ent-pill.stripped {
    background: var(--bg-surface-2);
    color: var(--text-dim);
    border: 1px dashed var(--border-strong);
    text-decoration: line-through;
  }
  .seq-scroll {
    display: flex;
    flex-direction: column;
    gap: 4px;
    overflow-x: auto;
    padding: 4px 2px;
    user-select: none;
  }
  .seq-scroll.is-windowed {
    max-height: 360px;
    overflow-y: auto;
  }
  .seq-scroll.can-paint {
    cursor: crosshair;
  }
  .seq-spacer {
    flex-shrink: 0;
  }
  .chain-head {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    gap: 5px;
    padding: 6px 8px;
    margin-top: 4px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
  }
  .chain-id-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
  }
  .chain-title {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11.5px;
    font-weight: 700;
    color: var(--text-primary);
  }
  .remap-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 4px;
    background: var(--warn-bg);
    color: var(--warn-text);
    border: 1px solid var(--warn-border);
  }
  .kind-pill,
  .role-tag,
  .msa-pill,
  .len-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    color: var(--text-muted);
  }
  .kind-protein {
    color: var(--accent-soft);
    border-color: var(--border-strong);
  }
  .kind-dna,
  .kind-rna {
    color: #c084fc;
    border-color: rgba(168, 85, 247, 0.45);
  }
  .kind-ligand {
    color: var(--ok-text);
    border-color: var(--ok-border);
  }
  .role-tag-target {
    background: rgba(59, 130, 246, 0.14);
    color: #60a5fa;
    border-color: rgba(59, 130, 246, 0.4);
  }
  .role-tag-binder,
  .role-tag-designed-span {
    background: rgba(16, 185, 129, 0.14);
    color: #34d399;
    border-color: rgba(16, 185, 129, 0.4);
  }
  .role-tag-scaffold {
    background: rgba(168, 85, 247, 0.14);
    color: #c084fc;
    border-color: rgba(168, 85, 247, 0.4);
  }
  .len-pill {
    margin-left: auto;
    color: var(--text-secondary);
  }
  .seq-row {
    display: flex;
    flex-wrap: nowrap;
    flex-shrink: 0;
    align-items: flex-end;
    padding: 3px 4px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    min-width: max-content;
  }
  .gap-div {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0 3px;
    margin: 0 2px;
    height: 22px;
    font-size: 10px;
    font-weight: 700;
    color: var(--warn-text);
    background: var(--warn-bg);
    border: 1px dashed var(--warn-border);
    border-radius: 3px;
    align-self: flex-end;
  }
  .res-cell {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    width: 15px;
    flex-shrink: 0;
    border-radius: 3px;
    position: relative;
  }
  .res-cell.block-start {
    margin-left: 7px;
  }
  .r-tick {
    height: 13px;
    font-size: 8.5px;
    line-height: 13px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: visible;
    align-self: flex-start;
    pointer-events: none;
  }
  .tick-num {
    display: inline-block;
  }
  .ins {
    text-decoration: underline;
    color: var(--accent-bright);
    font-weight: 700;
  }
  .r-lane {
    height: 13px;
    width: 100%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 8.5px;
    line-height: 1;
    border-radius: 2px 2px 0 0;
  }
  .r-lane.lane-inherited {
    background: repeating-linear-gradient(
      -45deg,
      rgba(59, 130, 246, 0.25),
      rgba(59, 130, 246, 0.25) 2px,
      transparent 2px,
      transparent 4px
    );
  }
  .r-lane.lane-user {
    background: rgba(59, 130, 246, 0.28);
  }
  .r-lane.lane-hotspot {
    background: rgba(249, 115, 22, 0.28);
  }
  .r-lane.lane-motif {
    background: rgba(168, 85, 247, 0.28);
  }
  .r-lane.lane-tied {
    background: rgba(20, 184, 166, 0.28);
    color: #2dd4bf;
    font-weight: 700;
    font-size: 7.5px;
  }
  .r-lane.lane-cropped {
    background: rgba(100, 116, 139, 0.24);
  }
  .r-lane.lane-missing {
    background: rgba(239, 68, 68, 0.3);
    color: #f87171;
    font-weight: 800;
  }
  .r-aa {
    width: 15px;
    height: 20px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 600;
    border-radius: 0 0 3px 3px;
    color: var(--text-primary);
  }
  /* Role color styles */
  .res-cell.role-fixed .r-aa {
    background: rgba(59, 130, 246, 0.22);
    color: #93c5fd;
  }
  .res-cell.role-fixed-inherited .r-aa {
    background: rgba(59, 130, 246, 0.14);
    color: var(--text-secondary);
  }
  .res-cell.role-hotspot .r-aa {
    background: rgba(249, 115, 22, 0.28);
    color: #fdba74;
    font-weight: 700;
  }
  .res-cell.role-motif .r-aa {
    background: rgba(168, 85, 247, 0.26);
    color: #d8b4fe;
    font-weight: 700;
  }
  .res-cell.role-tied .r-aa {
    background: rgba(20, 184, 166, 0.24);
    color: #5eead4;
  }
  .res-cell.role-cropped .r-aa {
    background: rgba(100, 116, 139, 0.16);
    color: var(--text-dim);
    text-decoration: line-through;
    opacity: 0.55;
  }
  .res-cell.role-missing .r-aa {
    background: rgba(239, 68, 68, 0.2);
    color: #fca5a5;
  }
  .res-cell.role-ligand .r-aa {
    background: rgba(34, 197, 94, 0.2);
    color: #86efac;
  }
  .res-cell.role-nucleic .r-aa {
    background: rgba(139, 92, 246, 0.2);
    color: #c4b5fd;
  }
  :global(:root[data-theme='light']) .res-cell.role-fixed .r-aa {
    color: #1d4ed8;
  }
  :global(:root[data-theme='light']) .res-cell.role-hotspot .r-aa {
    color: #c2410c;
  }
  :global(:root[data-theme='light']) .res-cell.role-motif .r-aa {
    color: #7e22ce;
  }
  :global(:root[data-theme='light']) .res-cell.role-tied .r-aa {
    color: #0f766e;
  }
  .res-cell.is-hovered {
    outline: 1.5px solid var(--accent-bright);
    z-index: 2;
    background: color-mix(in srgb, var(--accent) 16%, transparent);
  }
  .res-cell.is-drag-add {
    outline: 1.5px solid #22c55e;
    background: rgba(34, 197, 94, 0.25);
  }
  .res-cell.is-drag-rm {
    outline: 1.5px solid #ef4444;
    background: rgba(239, 68, 68, 0.22);
  }
  .coord-footer {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 12px;
    padding: 5px 8px;
    border-radius: 5px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-secondary);
  }
  .coord-footer b {
    color: var(--text-dim);
    font-weight: 600;
  }
  .coord-muted {
    color: var(--text-dim);
  }
  .float-tip {
    position: fixed;
    z-index: 80;
    pointer-events: none;
    min-width: 195px;
    max-width: 240px;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 7px 9px;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.36);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-body);
  }
  .ft-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
    padding-bottom: 4px;
    margin-bottom: 4px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 10.5px;
  }
  .ft-tok {
    color: var(--accent-bright);
    font-weight: 700;
  }
  .ft-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 2px 8px;
  }
  .ft-grid span:nth-child(odd) {
    color: var(--text-dim);
  }
  .align-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .align-legend {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 6px;
    font-size: 10.5px;
    color: var(--text-muted);
  }
  .align-hint {
    color: var(--text-dim);
  }
  .align-scroll {
    display: flex;
    flex-direction: column;
    gap: 5px;
    max-height: 340px;
    overflow: auto;
  }
  .align-row {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 6px 8px;
    border-radius: 6px;
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
  }
  .align-row.is-selected {
    border-color: var(--accent);
    background: var(--bg-surface-selected);
  }
  .align-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .align-hdr {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
  }
  .align-idx {
    font-weight: 700;
    color: var(--accent-bright);
  }
  .align-name {
    color: var(--text-primary);
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .align-score {
    color: var(--ok-text);
    font-size: 10px;
  }
  .align-id {
    color: var(--text-dim);
    font-size: 10px;
  }
  .mini-copy {
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    color: var(--text-secondary);
    border-radius: 4px;
    padding: 1px 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    cursor: pointer;
    flex-shrink: 0;
  }
  .mini-copy:hover {
    border-color: var(--accent);
    color: var(--text-primary);
  }
  .align-seq {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11.5px;
    letter-spacing: 0.6px;
    white-space: pre;
    overflow-x: auto;
    line-height: 1.45;
  }
  .align-ref-text {
    color: var(--text-primary);
    font-weight: 600;
  }
  .al-dot {
    color: var(--text-dim);
  }
  .al-mut {
    color: #fb923c;
    background: rgba(249, 115, 22, 0.2);
    border-radius: 2px;
    font-weight: 700;
  }
  .empty-seq {
    padding: 14px 10px;
    text-align: center;
    font-size: 11.5px;
    color: var(--text-dim);
    background: var(--bg-surface-2);
    border: 1px dashed var(--border);
    border-radius: 6px;
  }
</style>
