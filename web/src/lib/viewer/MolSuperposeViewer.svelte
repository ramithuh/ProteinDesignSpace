<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import type { Camera } from 'molstar/lib/mol-canvas3d/camera';
  import type { Scene } from 'molstar/lib/mol-gl/scene';
  import { Vec3 } from 'molstar/lib/mol-math/linear-algebra/3d/vec3';
  import type { ViewerModelSpec } from '$lib/engine/compare';
  import { isTrackSemantics, type TrackSemantics } from '$lib/engine/compare-identity';
  import { expandResidueTokenList } from '$lib/engine/residue-tokens';
  import type { ResidueHighlightLayer } from '$lib/engine/studio-types';
  import type { AtomGranularity, BfactorSemantics } from '$lib/engine/types';
  import {
    poolUncertaintyColorArgs,
    resolveUncertaintyColorArgFromValues,
    type UncertaintyColorArg,
  } from './bfactor-domain';
  import { pruneFalseNucleicMonomers } from './molstar-sanitize';
  import { resolveViewerSpinState } from './viewer-spin';

  interface Props {
    designedPdb: string | null;
    refoldPdb?: string | null;
    trajectoryPdb?: string | null;
    /** Optional multi-model comparison specs (takes precedence over designedPdb/refoldPdb when non-null). */
    models?: ViewerModelSpec[] | null;
    /** 0-based user frame index (0..T-1). Mapped to Mol* modelIndex = frame + 1. */
    frameIndex?: number;
    showDesigned?: boolean;
    showRefold?: boolean;
    showTrajectory?: boolean;
    illustrative?: boolean;
    spin?: boolean;
    /** Rotations per second when `spin` is enabled (Mol* default is 0.1; 0.02 ≈ 1 turn / 50s). */
    spinSpeed?: number;
    /** Lock camera reset to canonical front view (+Z -> -Z, up +Y) and pause spin (side-by-side / grid layouts). */
    canonicalCamera?: boolean;
    /** Temporarily pause spin animation (e.g. while a GPU pipeline run is active) without mutating `spin` or enabling SSAO occlusion. */
    pauseSpin?: boolean;
    /** When true (full-view expanded mode), shows the floating Representation & View menu. */
    expanded?: boolean;
    /** Override whether the Representation menu opens automatically upon entering `expanded` mode (default: true on viewports >= 640px). */
    defaultStyleMenuOpen?: boolean;
    theme?: 'dark' | 'light';
    height?: number | string;
    label?: string;
    atomGranularity?: AtomGranularity;
    bfactorSemantics?: BfactorSemantics;
    /** Zero-reload GPU overpaint layers (fixed, hotspot, motif, tied, cropped). */
    highlightLayers?: ResidueHighlightLayer[];
    /** Residue token currently hovered in sequence view (e.g. 'A54', 'H100A'). */
    hoveredToken?: string | null;
    /** Residue token to smoothly focus the camera onto (e.g. 'A54', 'H100A'). */
    focusToken?: string | null;
    /** Multi-model mode: restrict `focusToken` to this model spec ID (falls back to any visible model). */
    focusModelId?: string | null;
    emptyMessage?: string;
    emptyActionLabel?: string;
    onEmptyAction?: () => void;
    onResidueClick?: (info: {
      chain: string;
      seqId: number;
      insCode?: string;
      residueToken?: string;
      compId: string;
      atomName?: string;
      isNonPolymer?: boolean;
      entityType?: 'protein' | 'dna' | 'rna' | 'ligand';
    }) => void;
    onToggleModelVisibility?: (modelId: string) => void;
    onSoloModel?: (modelId: string) => void;
    /** Switch which painted track a merged cartoon shows (instant: only visibility changes). */
    onSelectTrack?: (groupId: string, nodeId: string) => void;
    /** Split merged identical structures back into separate models (or re-merge). */
    onToggleMergeIdentical?: () => void;
    onNotify?: (message: string, tone?: 'ok' | 'error') => void;
  }

  let {
    designedPdb,
    refoldPdb = null,
    trajectoryPdb = null,
    models = null,
    frameIndex = undefined,
    showDesigned = true,
    showRefold = true,
    showTrajectory = false,
    illustrative = false,
    spin = true,
    spinSpeed = 0.02,
    canonicalCamera = false,
    pauseSpin = false,
    expanded = false,
    defaultStyleMenuOpen = undefined,
    theme = 'dark',
    height = 420,
    label = undefined,
    atomGranularity = undefined,
    bfactorSemantics = undefined,
    highlightLayers = undefined,
    hoveredToken = null,
    focusToken = null,
    focusModelId = null,
    emptyMessage = undefined,
    emptyActionLabel = undefined,
    onEmptyAction = undefined,
    onResidueClick = undefined,
    onToggleModelVisibility = undefined,
    onSoloModel = undefined,
    onSelectTrack = undefined,
    onToggleMergeIdentical = undefined,
    onNotify = undefined,
  }: Props = $props();

  let containerEl: HTMLDivElement | null = $state(null);
  let canvasEl: HTMLCanvasElement | null = $state(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let plugin: any = $state(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let pluginPromise: Promise<any> | null = null;
  let destroyed = false;

  const COMP_TO_AA1: Record<string, string> = {
    ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C',
    GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
    LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P',
    SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
    MSE: 'M', SEC: 'U', PYL: 'O',
  };

  interface HoveredResidueInfo {
    compId: string;
    aa1: string | null;
    chain: string;
    seqId: number;
    insCode: string;
    atomName: string;
    isNonPolymer: boolean;
    bFactor: number | null;
    /** Mol* `Model.id` under the cursor (resolves per-model B-factor units in compare mode). */
    modelId: string | null;
  }

  let hoveredResidue = $state<HoveredResidueInfo | null>(null);
  // Mol* Model.id -> B-factor semantics of the rendered model ('none' suppresses the readout)
  let modelSemanticsById = new Map<string, BfactorSemantics | undefined>();
  // Structure state ref -> ViewerModelSpec.id (multi-model focus / hover targeting)
  let structSpecIds = new Map<string, string>();

  let hoveredResidueLabel = $derived.by((): string => {
    const h = hoveredResidue;
    if (!h) return '';
    const parts = [h.compId];
    if (!h.isNonPolymer && Number.isFinite(h.seqId)) {
      parts.push(`${h.seqId}${h.insCode}`);
    }
    if (h.aa1) {
      parts.push(`(${h.aa1})`);
    }
    return parts.join(' ');
  });

  let hoveredMetricText = $derived.by((): string | null => {
    const h = hoveredResidue;
    if (!h || h.isNonPolymer || h.bFactor === null || !Number.isFinite(h.bFactor)) return null;
    const sem = ((h.modelId ? modelSemanticsById.get(h.modelId) : undefined) ?? bfactorSemantics) as
      | string
      | undefined;
    if (sem === 'dynamics-prob') return `${h.bFactor.toFixed(2)} p(exch)`;
    if (sem === 'ddg-stability') {
      return `${h.bFactor >= 0 ? '+' : ''}${h.bFactor.toFixed(2)} kcal/mol`;
    }
    if (sem === 'llr-fitness') {
      return `${h.bFactor >= 0 ? '+' : ''}${h.bFactor.toFixed(2)} LLR`;
    }
    if (sem === 'gnina-contrib') {
      return `${h.bFactor >= 0 ? '+' : ''}${h.bFactor.toFixed(2)} ΔCNN_VS`;
    }
    if (sem === 'plddt' || sem === 'plddt-0-100') return `pLDDT ${h.bFactor.toFixed(1)}`;
    if (sem === 'plddt-0-1') {
      const scaled = h.bFactor <= 1.01 ? h.bFactor * 100 : h.bFactor;
      return `pLDDT ${scaled.toFixed(1)}`;
    }
    return null;
  });
  let trajModelRef = $state<string | null>(null);
  let modelRefs = new Map<string, string[]>();
  let loadSeq = 0;
  let initError = $state<string | null>(null);
  let webglFailed = false;
  let webglFailReason = '';

  const NUCLEIC_COMP_IDS = new Set([
    'DA',
    'DT',
    'DG',
    'DC',
    'DU',
    'DI',
    'A',
    'U',
    'G',
    'C',
    'I',
    'T',
  ]);

  let scrubPending = false;
  let queuedFrame: number | null = null;

  // Interactive Representation & View controls (shown in expanded full-view mode)
  type PolymerRepType =
    | 'cartoon'
    | 'ball-and-stick'
    | 'molecular-surface'
    | 'gaussian-surface'
    | 'putty'
    | 'spacefill'
    | 'backbone';
  type ColorSchemeOption =
    | 'auto'
    | 'chain-id'
    | 'plddt-confidence'
    | 'uncertainty'
    | 'secondary-structure'
    | 'hydrophobicity'
    | 'sequence-id'
    | 'element-symbol';

  let styleMenuWrapEl: HTMLDivElement | null = $state(null);
  let styleMenuOpen = $state(false);
  let lastStyleMenuTrigger: string | null = null;
  let pngStatus = $state<'idle' | 'saved' | 'failed'>('idle');
  let pngStatusTimer: ReturnType<typeof setTimeout> | null = null;

  $effect(() => {
    const exp = expanded;
    const defOpen = defaultStyleMenuOpen;
    const triggerKey = `${exp ? 1 : 0}:${defOpen === undefined ? 'auto' : defOpen ? 1 : 0}`;
    if (lastStyleMenuTrigger === triggerKey) return;
    lastStyleMenuTrigger = triggerKey;
    const shouldAutoOpen = exp && (defOpen ?? true);
    styleMenuOpen =
      shouldAutoOpen && (typeof window !== 'undefined' ? window.innerWidth >= 640 : true);
  });

  // Close Representation / Style menu on Escape before outer overlays or Node Studio collapse
  $effect(() => {
    if (!styleMenuOpen || typeof window === 'undefined') return;
    function handleStyleMenuEscape(ev: KeyboardEvent) {
      if (ev.key !== 'Escape' || ev.defaultPrevented) return;
      if (
        typeof document !== 'undefined' &&
        document.querySelector('.mc-backdrop, .gal-backdrop, .pw-coach')
      ) {
        return;
      }
      ev.preventDefault();
      ev.stopPropagation();
      styleMenuOpen = false;
    }
    window.addEventListener('keydown', handleStyleMenuEscape, true);
    return () => {
      window.removeEventListener('keydown', handleStyleMenuEscape, true);
    };
  });

  // In compact split view, close the floating style popover on click-outside
  $effect(() => {
    if (expanded || !styleMenuOpen || typeof window === 'undefined') return;
    function handleOutsidePointerDown(ev: PointerEvent) {
      const target = ev.target;
      if (!(target instanceof globalThis.Node)) return;
      if (styleMenuWrapEl?.contains(target)) return;
      styleMenuOpen = false;
    }
    window.addEventListener('pointerdown', handleOutsidePointerDown, true);
    return () => {
      window.removeEventListener('pointerdown', handleOutsidePointerDown, true);
    };
  });
  let repType = $state<PolymerRepType>('cartoon');
  let colorScheme = $state<ColorSchemeOption>('auto');
  let showSidechains = $state(false);
  let showSurfaceOverlay = $state(false);
  let showLigands = $state(true);
  let cameraMode = $state<'perspective' | 'orthographic'>('perspective');
  let localSpinSpeed = $state<number | null>(null);
  let effectiveSpinSpeed = $derived(localSpinSpeed ?? spinSpeed);
  let isSemanticBfactor = $derived(isTrackSemantics(bfactorSemantics));
  let plddtAllowed = $derived(
    bfactorSemantics !== 'none' &&
      bfactorSemantics !== 'crystal-b' &&
      !isSemanticBfactor,
  );
  let lastStructureKey = '';
  let lastRenderedKey = '';
  let sceneRenderEpoch = $state(0);
  let overpaintSeq = 0;
  let lastAppliedOverpaintKey = '';
  // Numeric color domain per track semantics from the last render (pooled across all painted models)
  let trackDomainsBySemantics = $state<Record<string, [number, number]>>({});

  /**
   * Collect polymer Cα B-factors (falling back to all polymer, then all atoms) so color domains
   * are not skewed by untouched crystal waters (`HOH`), ligands, or peptide caps (`ACE`).
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function collectPolymerBfactors(struct: any): number[] {
    try {
      const units = struct?.data?.units;
      const caValues: number[] = [];
      const polyValues: number[] = [];
      const allValues: number[] = [];
      if (Array.isArray(units)) {
        for (const u of units) {
          const bCol = u?.model?.atomicConformation?.B_iso_or_equiv;
          const atomIdCol = u?.model?.atomicHierarchy?.atoms?.label_atom_id;
          const polyElems = u?.polymerElements;
          const allElems = u?.elements;
          if (!bCol || typeof bCol.value !== 'function') continue;

          // Prefer polymer Cα atoms (excludes waters, ligands, and terminal caps like ACE/NME)
          if (polyElems && polyElems.length > 0) {
            const np = polyElems.length;
            for (let i = 0; i < np; i++) {
              const el = polyElems[i];
              const v = bCol.value(el);
              if (typeof v !== 'number' || !Number.isFinite(v)) continue;
              polyValues.push(v);
              if (atomIdCol && typeof atomIdCol.value === 'function' && atomIdCol.value(el) === 'CA') {
                caValues.push(v);
              }
            }
          }
          if (allElems && allElems.length > 0 && caValues.length === 0) {
            const na = allElems.length;
            for (let i = 0; i < na; i++) {
              const v = bCol.value(allElems[i]);
              if (typeof v === 'number' && Number.isFinite(v)) {
                allValues.push(v);
              }
            }
          }
        }
      }
      return caValues.length > 0 ? caValues : polyValues.length > 0 ? polyValues : allValues;
    } catch {
      return [];
    }
  }

  /**
   * Build a Mol* `'uncertainty'` (B-factor continuous ramp) color argument, adapting the
   * numeric domain to the observed polymer Cα B-factors so both raw (`0..1` dynamics prob,
   * signed `ΔΔG` kcal/mol) and `0..100`-scaled structures display full blue↔white↔red contrast.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function resolveUncertaintyColorArg(struct: any, semantics?: BfactorSemantics): UncertaintyColorArg {
    return resolveUncertaintyColorArgFromValues(collectPolymerBfactors(struct), semantics);
  }

  function hasMultiplePdbChains(pdbText: string): boolean {
    let firstChain = '';
    let start = 0;
    const len = pdbText.length;
    while (start < len) {
      let end = pdbText.indexOf('\n', start);
      if (end === -1) end = len;
      if (end - start >= 22 && pdbText.startsWith('ATOM  ', start)) {
        const ch = pdbText[start + 21];
        if (!firstChain) {
          firstChain = ch;
        } else if (ch !== firstChain) {
          return true;
        }
      }
      start = end + 1;
    }
    return false;
  }

  function resetStyleControls() {
    repType = 'cartoon';
    colorScheme = 'auto';
    showSidechains = false;
    showSurfaceOverlay = false;
    showLigands = true;
    cameraMode = 'perspective';
    localSpinSpeed = null;
  }

  let renderChain: Promise<void> = Promise.resolve();
  let forceRenderTick = $state(0);

  function enqueueSceneRender(task: () => Promise<void>) {
    renderChain = renderChain.then(task, task);
    return renderChain;
  }

  function hashStructureSample(text: string): string {
    let h = 2166136261;
    const n = text.length;
    const step = 1;
    for (let i = 0; i < n; i += step) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0).toString(36);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function resetViewerCamera(ctx: any, canonical: boolean) {
    if (!ctx?.canvas3d) return;
    if (canonical) {
      ctx.canvas3d.requestCameraReset({
        durationMs: 0,
        snapshot: (scene: Scene, camera: Camera) => {
          const { center, radius } = scene.boundingSphereVisible;
          return camera.getInvariantFocus(
            center,
            radius,
            Vec3.create(0, 1, 0),
            Vec3.create(0, 0, -1),
          );
        },
      });
    } else {
      ctx.canvas3d.requestCameraReset();
    }
  }

  function handleResetCamera() {
    const liveStructs = plugin?.managers?.structure?.hierarchy?.current?.structures?.length ?? 0;
    const expectsStructure = Boolean(
      (models && models.length > 0) ||
        (showDesigned && (designedPdb || (showTrajectory && trajectoryPdb))) ||
        (showRefold && refoldPdb),
    );
    if (expectsStructure && liveStructs === 0) {
      lastRenderedKey = '';
      lastStructureKey = '';
      forceRenderTick += 1;
      return;
    }
    resetViewerCamera(plugin, canonicalCamera);
  }

  function flashPngStatus(status: 'saved' | 'failed') {
    pngStatus = status;
    if (pngStatusTimer) clearTimeout(pngStatusTimer);
    pngStatusTimer = setTimeout(() => {
      pngStatus = 'idle';
      pngStatusTimer = null;
    }, status === 'failed' ? 2400 : 1600);
  }

  /** Capture the current Mol* 3D WebGL viewport as a PNG data URL (used by PNG save & social card). */
  export async function capturePngDataUrl(): Promise<string | null> {
    if (!canvasEl || !plugin || initError || webglFailed) return null;
    try {
      let url: string | null = null;
      if (plugin.helpers?.viewportScreenshot?.getImageDataUri) {
        try {
          url = await plugin.helpers.viewportScreenshot.getImageDataUri();
        } catch {
          url = null;
        }
      }
      if (!url) {
        plugin.canvas3d?.requestDraw?.();
        plugin.canvas3d?.commit?.(true);
        plugin.canvas3d?.tick?.(performance.now(), { isSynchronous: true, manualDraw: true });
        url = canvasEl.toDataURL('image/png');
      }
      return url && url.startsWith('data:image/') ? url : null;
    } catch {
      return null;
    }
  }

  async function handleSavePng() {
    if (!canvasEl || !plugin || initError || webglFailed) {
      flashPngStatus('failed');
      onNotify?.('Could not export PNG — 3D WebGL viewer is unavailable.', 'error');
      return;
    }
    try {
      const url = await capturePngDataUrl();
      if (!url) {
        flashPngStatus('failed');
        onNotify?.('Could not capture 3D PNG screenshot from WebGL canvas.', 'error');
        return;
      }
      const filename = `${(label || 'structure').replace(/[^a-z0-9._-]+/gi, '_').slice(0, 48)}.png`;
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      flashPngStatus('saved');
      onNotify?.(`Saved 3D screenshot (${filename})`, 'ok');
    } catch (err) {
      flashPngStatus('failed');
      onNotify?.(
        `Could not export 3D PNG screenshot: ${err instanceof Error ? err.message : 'canvas export failed'}`,
        'error',
      );
    }
  }

  async function ensurePlugin() {
    if (destroyed || webglFailed) return null;
    if (plugin) return plugin;
    if (pluginPromise) return pluginPromise;
    if (!containerEl || !canvasEl) return null;

    if (typeof WebGL2RenderingContext === 'undefined') {
      webglFailed = true;
      webglFailReason = 'WebGL2 is unavailable in this browser';
      initError = webglFailReason;
      return null;
    }

    const cEl = containerEl;
    const cvEl = canvasEl;

    pluginPromise = (async () => {
      try {
        const [
          { PluginContext },
          { DefaultPluginSpec, PluginSpec },
          { PluginConfig },
          { MAQualityAssessment },
        ] = await Promise.all([
          import('molstar/lib/mol-plugin/context'),
          import('molstar/lib/mol-plugin/spec'),
          import('molstar/lib/mol-plugin/config'),
          import('molstar/lib/extensions/model-archive/quality-assessment/behavior'),
        ]);
        if (destroyed) return null;

        const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
        // Target crisp 2x effective DPR on 2x/3x Retina phones & laptops (avoiding Mol*'s mobile 'auto' double-division by DPR)
        const pixelScale = Math.min(1, 2 / dpr);

        const spec = DefaultPluginSpec();
        spec.behaviors.push(PluginSpec.Behavior(MAQualityAssessment));
        spec.config = [
          ...(spec.config ?? []),
          [PluginConfig.General.DisablePreserveDrawingBuffer, true],
          [PluginConfig.General.PixelScale, pixelScale],
        ];
        const ctx = new PluginContext(spec);
        await ctx.init();
        if (destroyed) {
          try {
            ctx.dispose();
          } catch {
            /* ignore */
          }
          return null;
        }

        const ok = await ctx.initViewerAsync(cvEl, cEl);
        if (destroyed) {
          try {
            ctx.dispose();
          } catch {
            /* ignore */
          }
          return null;
        }
        if (!ok) {
          try {
            ctx.dispose();
          } catch {
            /* ignore */
          }
          webglFailed = true;
          webglFailReason = 'Failed to initialize Mol* WebGL2 canvas';
          initError = webglFailReason;
          return null;
        }

        ctx.canvas3dContext?.setProps?.({ resolutionMode: 'native', pixelScale });
        ctx.handleResize();
        ctx.canvas3d?.setProps({
          renderer: { backgroundColor: (theme === 'light' ? 0xf8fafc : 0x0d0e11) as unknown as never },
          cameraFog: { name: 'off', params: {} },
          cameraClipping: { far: false, radius: 0, minNear: 0.1 },
          postprocessing: buildPostprocessingProps(illustrative, spinState.disableOcclusion),
          trackball: {
            animate: spinState.effectiveSpin
              ? { name: 'spin', params: { speed: effectiveSpinSpeed, axis: [0, -1, 0] as unknown as never } }
              : { name: 'off', params: {} },
          },
        });

        // Click picking -> emit structured residue/entity info (protein, DNA/RNA, and non-polymer ligands/ions)
        const { StructureElement, StructureProperties } = await import('molstar/lib/mol-model/structure');
        if (destroyed) {
          try {
            ctx.dispose();
          } catch {
            /* ignore */
          }
          return null;
        }
        const DNA_COMP_IDS = new Set(['DA', 'DT', 'DG', 'DC', 'DI', 'T']);
        const RNA_COMP_IDS = new Set(['A', 'U', 'G', 'C', 'I']);
        const WATER_COMP_IDS = new Set(['HOH', 'WAT', 'DOD', 'H2O']);

        ctx.behaviors.interaction.hover.subscribe((ev) => {
          const loci = ev.current.loci;
          if (!StructureElement.Loci.is(loci) || loci.elements.length === 0) {
            hoveredResidue = null;
            return;
          }
          const loc = StructureElement.Loci.getFirstLocation(loci);
          if (!loc) {
            hoveredResidue = null;
            return;
          }
          const compId = String(StructureProperties.atom.label_comp_id(loc) || 'UNK').toUpperCase();
          if (WATER_COMP_IDS.has(compId)) {
            hoveredResidue = null;
            return;
          }
          const chain = String(StructureProperties.chain.auth_asym_id(loc) || 'A');
          const seqId = Number(StructureProperties.residue.auth_seq_id(loc));
          let insCode = '';
          try {
            const rawIns = String(StructureProperties.residue.pdbx_PDB_ins_code(loc) ?? '').trim();
            if (rawIns && rawIns !== '?' && rawIns !== '.') insCode = rawIns;
          } catch {
            insCode = '';
          }
          let atomName = '';
          try {
            atomName = String(StructureProperties.atom.label_atom_id(loc) ?? '').trim();
          } catch {
            atomName = '';
          }
          let rawEntityType = 'polymer';
          try {
            rawEntityType = String(StructureProperties.entity.type(loc) ?? 'polymer').toLowerCase();
          } catch {
            rawEntityType = 'polymer';
          }
          let bFactor: number | null = null;
          try {
            const bf = Number(StructureProperties.atom.B_iso_or_equiv(loc));
            if (Number.isFinite(bf)) bFactor = bf;
          } catch {
            bFactor = null;
          }
          const hoveredModelId = loc.unit?.model?.id;
          hoveredResidue = {
            compId,
            aa1: COMP_TO_AA1[compId] ?? null,
            chain,
            seqId,
            insCode,
            atomName,
            isNonPolymer: rawEntityType !== 'polymer',
            bFactor,
            modelId: typeof hoveredModelId === 'string' ? hoveredModelId : null,
          };
        });

        ctx.behaviors.interaction.click.subscribe((ev) => {
          if (!onResidueClick) return;
          const loci = ev.current.loci;
          if (!StructureElement.Loci.is(loci) || loci.elements.length === 0) return;
          const loc = StructureElement.Loci.getFirstLocation(loci);
          if (!loc) return;

          const compId = StructureProperties.atom.label_comp_id(loc) || 'UNK';
          const compUpper = compId.toUpperCase();
          if (WATER_COMP_IDS.has(compUpper)) return;

          let rawEntityType = 'polymer';
          let entitySubtype = '';
          try {
            rawEntityType = String(StructureProperties.entity.type(loc) ?? '').toLowerCase();
          } catch {
            rawEntityType = 'polymer';
          }
          try {
            entitySubtype = String(StructureProperties.entity.subtype(loc) ?? '').toLowerCase();
          } catch {
            entitySubtype = '';
          }

          let clickedEntityType: 'protein' | 'dna' | 'rna' | 'ligand' = 'protein';
          if (rawEntityType !== 'polymer') {
            clickedEntityType = 'ligand';
          } else if (
            entitySubtype.includes('deoxyribonucleotide') ||
            DNA_COMP_IDS.has(compUpper)
          ) {
            clickedEntityType = 'dna';
          } else if (
            entitySubtype.includes('ribonucleotide') ||
            entitySubtype.includes('nucleic') ||
            RNA_COMP_IDS.has(compUpper) ||
            NUCLEIC_COMP_IDS.has(compUpper)
          ) {
            clickedEntityType = 'rna';
          } else if (
            entitySubtype &&
            !entitySubtype.includes('polypeptide') &&
            !entitySubtype.includes('peptide') &&
            entitySubtype !== 'other'
          ) {
            clickedEntityType = 'ligand';
          }

          const chain = StructureProperties.chain.auth_asym_id(loc) || 'A';
          const seqId = StructureProperties.residue.auth_seq_id(loc);
          let insCode = '';
          try {
            const rawIns = String(StructureProperties.residue.pdbx_PDB_ins_code(loc) ?? '').trim();
            if (rawIns && rawIns !== '?' && rawIns !== '.') {
              insCode = rawIns;
            }
          } catch {
            insCode = '';
          }
          let atomName = '';
          try {
            atomName = String(
              StructureProperties.atom.label_atom_id(loc) ||
                StructureProperties.atom.auth_atom_id(loc) ||
                '',
            ).trim();
          } catch {
            atomName = '';
          }

          onResidueClick({
            chain,
            seqId,
            ...(insCode ? { insCode } : {}),
            residueToken: `${chain}${seqId}${insCode}`,
            compId,
            ...(atomName ? { atomName } : {}),
            isNonPolymer: clickedEntityType === 'ligand',
            entityType: clickedEntityType,
          });
        });

        plugin = ctx;
        return ctx;
      } catch (err) {
        webglFailed = true;
        webglFailReason = err instanceof Error ? err.message : String(err);
        if (!destroyed) {
          initError = webglFailReason;
        }
        return null;
      }
    })();

    try {
      return await pluginPromise;
    } finally {
      pluginPromise = null;
    }
  }

  /**
   * Build `typeParams` for primary polymer representations, omitting Mol*'s dashed
   * `polymer-gap` cylinders so disjoint motifs and missing loops render cleanly without
   * phantom dashed sticks connecting non-contiguous Cα atoms.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function buildPrimaryTypeParams(
    rep: PolymerRepType,
    struct: any,
    alpha?: number,
  ): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    if (alpha !== undefined) params.alpha = alpha;
    if (rep === 'cartoon') {
      const units = struct?.data?.units;
      let hasNucleotides = false;
      if (Array.isArray(units)) {
        for (const u of units) {
          if (u?.nucleotideElements?.length > 0) {
            hasNucleotides = true;
            break;
          }
        }
      }
      params.visuals = hasNucleotides ? ['polymer-trace', 'nucleotide-ring'] : ['polymer-trace'];
    } else if (rep === 'putty') {
      params.visuals = ['polymer-tube'];
    } else if (rep === 'backbone') {
      params.visuals = ['polymer-backbone-cylinder', 'polymer-backbone-sphere'];
    }
    return params;
  }

  /**
   * Add ligand (ball-and-stick), ion (spacefill), and sparse theozyme (<= 15 polymer residues,
   * ball-and-stick) representations to a Mol* structure and return their state refs for visibility toggling.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async function addAccessoryRepresentations(
    ctx: any,
    struct: any,
    opts: {
      includeLigands: boolean;
      includeSidechains: boolean;
      includeSurfaceOverlay: boolean;
      primaryRep: PolymerRepType;
      colorArg: Record<string, unknown>;
      alpha?: number;
    },
  ): Promise<string[]> {
    const refs: string[] = [];
    const sData = struct.data;
    const polymerResidues = sData?.polymerResidueCount ?? 0;
    const isSparseTheozyme = polymerResidues > 0 && polymerResidues <= 15;
    if (opts.includeLigands) {
      let renderedAnyLigand = false;
      try {
        const ligComp = await ctx.builders.structure.tryCreateComponentStatic(struct, 'ligand');
        if (ligComp) {
          const ligRepr = await ctx.builders.structure.representation.addRepresentation(ligComp, {
            type: 'ball-and-stick',
            color: 'element-symbol',
            typeParams: { alpha: opts.alpha ?? 1.0 },
          });
          if (ligRepr?.ref) refs.push(ligRepr.ref);
          else if (ligComp.ref) refs.push(ligComp.ref);
          renderedAnyLigand = true;
        }
      } catch {
        /* ignore if no ligand component */
      }
      // Also render non-standard / branched / amino-acid-like non-polymer HETATM cofactors
      // (e.g. SAM from RF3/Boltz-2/AF3) that Mol*'s static 'ligand' preset can omit.
      try {
        const nonStdComp = await ctx.builders.structure.tryCreateComponentStatic(struct, 'non-standard');
        if (nonStdComp) {
          const nonStdRepr = await ctx.builders.structure.representation.addRepresentation(nonStdComp, {
            type: 'ball-and-stick',
            color: 'element-symbol',
            typeParams: { alpha: opts.alpha ?? 1.0 },
          });
          if (nonStdRepr?.ref) refs.push(nonStdRepr.ref);
          else if (nonStdComp.ref) refs.push(nonStdComp.ref);
          renderedAnyLigand = true;
        }
      } catch {
        /* ignore if no non-standard component */
      }
      if (!renderedAnyLigand) {
        try {
          const { MolScriptBuilder: MS } = await import('molstar/lib/mol-script/language/builder');
          const nonPolyExpr = MS.struct.generator.atomGroups({
            'entity-test': MS.core.rel.eq([MS.ammp('entityType'), 'non-polymer']),
          });
          const nonPolyComp = await ctx.builders.structure.tryCreateComponentFromExpression(
            struct,
            nonPolyExpr,
            'non-polymer-ligands',
            { label: 'Non-Polymer Ligands' },
          );
          if (nonPolyComp) {
            const npRepr = await ctx.builders.structure.representation.addRepresentation(nonPolyComp, {
              type: 'ball-and-stick',
              color: 'element-symbol',
              typeParams: { alpha: opts.alpha ?? 1.0 },
            });
            if (npRepr?.ref) refs.push(npRepr.ref);
            else if (nonPolyComp.ref) refs.push(nonPolyComp.ref);
          }
        } catch {
          /* ignore if MolScript non-polymer query unavailable */
        }
      }
      try {
        const ionComp = await ctx.builders.structure.tryCreateComponentStatic(struct, 'ion');
        if (ionComp) {
          const ionRepr = await ctx.builders.structure.representation.addRepresentation(ionComp, {
            type: 'spacefill',
            color: 'element-symbol',
            typeParams: { alpha: opts.alpha ?? 1.0 },
          });
          if (ionRepr?.ref) refs.push(ionRepr.ref);
          else if (ionComp.ref) refs.push(ionComp.ref);
        }
      } catch {
        /* ignore if no ion component */
      }
      if (isSparseTheozyme && opts.primaryRep !== 'ball-and-stick') {
        try {
          const sparseRepr = await ctx.builders.structure.representation.addRepresentation(struct, {
            type: 'ball-and-stick',
            color: 'element-symbol',
            typeParams: { alpha: opts.alpha ?? 1.0 },
          });
          if (sparseRepr?.ref) refs.push(sparseRepr.ref);
        } catch {
          /* ignore */
        }
      }
    }
    if (opts.includeSidechains && opts.primaryRep !== 'ball-and-stick' && !isSparseTheozyme) {
      try {
        const scRepr = await ctx.builders.structure.representation.addRepresentation(struct, {
          type: 'ball-and-stick',
          color: 'element-symbol',
          typeParams: { sizeFactor: 0.16, alpha: opts.alpha ?? 1.0 },
        });
        if (scRepr?.ref) refs.push(scRepr.ref);
      } catch {
        /* ignore */
      }
    }
    if (
      opts.includeSurfaceOverlay &&
      opts.primaryRep !== 'molecular-surface' &&
      opts.primaryRep !== 'gaussian-surface'
    ) {
      try {
        const surfRepr = await ctx.builders.structure.representation.addRepresentation(struct, {
          type: 'molecular-surface',
          ...opts.colorArg,
          typeParams: { alpha: 0.24 },
        });
        if (surfRepr?.ref) refs.push(surfRepr.ref);
      } catch {
        /* ignore */
      }
    }
    return refs;
  }

  /**
   * Strip Mol*'s geometric distance-heuristic `cyclicPolymerMap` entries (`atomic-ranges.js:56-63`,
   * which closes any chain with `d(CA_1, CA_L) < 10 Å` on Cα-only / O-less models or
   * `d(N_1, C_L) < 3 Å` on all-atom models) unless head-to-tail cyclization is explicitly annotated
   * via `_entity_poly.type = 'cyclic-pseudo-peptide'`, `_struct_conn` (`covale`, non-disulfide) /
   * PDB `CONECT` on terminal `N`↔`C`, or a PDB `LINK` record.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function pruneFalseCyclicPolymers(trajSel: any, modelSel: any, rawText: string, StructConn?: any) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const models = new Set<any>();
    const frames = trajSel?.data?.frames ?? trajSel?.cell?.obj?.data?.frames;
    if (Array.isArray(frames)) {
      for (const f of frames) if (f) models.add(f);
    }
    const activeModel = modelSel?.data ?? modelSel?.cell?.obj?.data;
    if (activeModel) models.add(activeModel);
    if (models.size === 0) return;

    const isCif = /^\s*(data_|loop_|_atom_site\.)/m.test(rawText);
    const hasPdbBackboneLink =
      !isCif && /^LINK\s+.{6}\s*(?:N|C)\s+.{24}\s*(?:N|C)\s+/m.test(rawText);
    if (hasPdbBackboneLink) return;

    for (const m of models) {
      const cyclicMap: Map<number, number> | undefined = m?.atomicRanges?.cyclicPolymerMap;
      if (!cyclicMap || cyclicMap.size === 0) continue;

      const hierarchy = m.atomicHierarchy;
      const entities = m.entities;
      const structConn = StructConn?.Provider?.get?.(m);
      const atomNames = hierarchy?.atoms?.label_atom_id;

      for (const [riA, riB] of [...cyclicMap.entries()]) {
        if (riA > riB) continue;
        let explicitlyCyclic = false;

        // 1. mmCIF _entity_poly.type === 'cyclic-pseudo-peptide'
        try {
          const firstAtom = hierarchy.residueAtomSegments.offsets[riA];
          const cI = hierarchy.chainAtomSegments.index[firstAtom];
          const entityId = hierarchy.chains.label_entity_id.value(cI);
          const eI = entities.getEntityIndex(entityId);
          if (entities.subtype?.value?.(eI) === 'cyclic-pseudo-peptide') {
            explicitlyCyclic = true;
          }
        } catch {
          /* ignore */
        }

        // 2. Explicit _struct_conn (covale) or PDB CONECT bonding terminal N <-> C
        if (!explicitlyCyclic && structConn?.entries?.length) {
          for (const e of structConn.entries) {
            // BondType.Flag.Covalent = 1, BondType.Flag.Disulfide = 8 (exclude Cys-Cys disulfides)
            if ((e.flags & 1) === 0 || (e.flags & 8) !== 0) continue;
            const r1 = e.partnerA?.residueIndex;
            const r2 = e.partnerB?.residueIndex;
            if ((r1 === riA && r2 === riB) || (r1 === riB && r2 === riA)) {
              const a1 = String(atomNames?.value?.(e.partnerA.atomIndex) ?? '').trim();
              const a2 = String(atomNames?.value?.(e.partnerB.atomIndex) ?? '').trim();
              if ((a1 === 'N' && a2 === 'C') || (a1 === 'C' && a2 === 'N')) {
                explicitlyCyclic = true;
                break;
              }
            }
          }
        }

        if (!explicitlyCyclic) {
          cyclicMap.delete(riA);
          cyclicMap.delete(riB);
        }
      }
    }
  }

  async function renderScene(
    dPdb: string | null,
    rPdb: string | null,
    tPdb: string | null,
    useTraj: boolean,
    sDesigned: boolean,
    sRefold: boolean,
    fIdx: number | undefined,
    styleOpts: {
      rep: PolymerRepType;
      color: ColorSchemeOption;
      sidechains: boolean;
      surfaceOverlay: boolean;
      ligands: boolean;
      plddtAllowed: boolean;
      bfactorSemantics?: BfactorSemantics;
      preserveCamera: boolean;
      structureKey: string;
      fullKey: string;
    },
  ) {
    if (webglFailed) {
      initError = webglFailReason || 'WebGL2 is unavailable in this browser';
      return;
    }
    const mySeq = ++loadSeq;
    initError = null;
    await enqueueSceneRender(async () => {
      if (destroyed || mySeq !== loadSeq) return;
      if (!dPdb && !rPdb) {
        if (plugin) await plugin.clear();
        trajModelRef = null;
        lastStructureKey = '';
        lastRenderedKey = styleOpts.fullKey;
        return;
      }

      try {
        const ctx = await ensurePlugin();
        if (destroyed || !ctx || mySeq !== loadSeq) return;
        const savedCam = styleOpts.preserveCamera ? ctx.canvas3d?.camera?.getSnapshot() : null;

        await ctx.clear();
        trajModelRef = null;
        if (destroyed || mySeq !== loadSeq) return;

        const [{ Color }, { StructConn }] = await Promise.all([
          import('molstar/lib/mol-util/color'),
          import('molstar/lib/mol-model-formats/structure/property/bonds/struct_conn'),
        ]);
        if (destroyed || mySeq !== loadSeq) return;
        const hasOverlay = Boolean(sDesigned && dPdb && sRefold && rPdb && !useTraj);
        const useSemanticUncertainty = isTrackSemantics(styleOpts.bfactorSemantics);
        const nextSemantics = new Map<string, BfactorSemantics | undefined>();
        const nextDomains: Record<string, [number, number]> = {};
        const recordDomain = (arg: Record<string, unknown>) => {
          const domain = (arg as UncertaintyColorArg).colorParams?.domain;
          if (isTrackSemantics(styleOpts.bfactorSemantics) && arg.color === 'uncertainty' && domain) {
            nextDomains[styleOpts.bfactorSemantics] = domain;
          }
        };

        // 1. Designed structure (or multi-model trajectory)
        if (sDesigned && (useTraj && tPdb ? tPdb : dPdb)) {
          const text = useTraj && tPdb ? tPdb : dPdb!;
          const fmt = /^\s*(data_|loop_|_atom_site\.)/m.test(text) ? 'mmcif' : 'pdb';
          const data = await ctx.builders.data.rawData({
            data: text,
            label: useTraj ? 'Trajectory' : 'Designed Backbone',
          });
          const traj = await ctx.builders.structure.parseTrajectory(data, fmt);
          // Fixed off-by-one: MODEL 1 = index 0 (topology), frames 0..T-1 = indices 1..T
          const targetIdx = useTraj && fIdx !== undefined ? fIdx + 1 : 0;
          const model = await ctx.builders.structure.createModel(traj, { modelIndex: targetIdx });
          if (destroyed || mySeq !== loadSeq) return;
          pruneFalseCyclicPolymers(traj, model, text, StructConn);
          pruneFalseNucleicMonomers(traj, model);
          if (useTraj) trajModelRef = model.ref;
          const struct = await ctx.builders.structure.createStructure(model, {
            name: 'model',
            params: {},
          });
          // Color multi-chain complexes (e.g. binder + target) by chain so binder and target
          // stand apart clearly in 3D without needing an overlay/scRMSD block.
          let isMultiChain = false;
          if (!hasOverlay) {
            if (fmt === 'pdb') {
              isMultiChain = hasMultiplePdbChains(text);
            } else {
              // mmCIF: count distinct polymer units in the asymmetric/cropped model
              const sData = struct.data;
              if (sData) {
                let polymerUnits = 0;
                for (const u of sData.units) {
                  if (u.polymerElements.length > 0) {
                    polymerUnits++;
                    if (polymerUnits >= 2) {
                      isMultiChain = true;
                      break;
                    }
                  }
                }
              }
            }
          }
          const effectiveDesignedColor =
            styleOpts.color === 'plddt-confidence' && !styleOpts.plddtAllowed
              ? 'auto'
              : styleOpts.color;
          const designedColorArg =
            effectiveDesignedColor === 'uncertainty'
              ? resolveUncertaintyColorArg(struct, styleOpts.bfactorSemantics)
              : effectiveDesignedColor !== 'auto'
                ? { color: effectiveDesignedColor }
                : hasOverlay
                  ? { color: 'uniform', colorParams: { value: Color(0x94a3b8) } }
                  : useSemanticUncertainty
                    ? resolveUncertaintyColorArg(struct, styleOpts.bfactorSemantics)
                    : isMultiChain
                      ? { color: 'chain-id' }
                      : { color: 'uniform', colorParams: { value: Color(0x60a5fa) } };
          const designedAlpha = hasOverlay ? 0.45 : 1.0;
          recordDomain(designedColorArg);
          const designedModelId = model.data?.id;
          if (typeof designedModelId === 'string') {
            // Overlay ghost B-factors are not the refold's pLDDT: suppress the hover metric
            nextSemantics.set(designedModelId, hasOverlay ? 'none' : styleOpts.bfactorSemantics);
          }
          await ctx.builders.structure.representation.addRepresentation(struct, {
            type: styleOpts.rep,
            ...designedColorArg,
            typeParams: buildPrimaryTypeParams(styleOpts.rep, struct, designedAlpha),
          });
          await addAccessoryRepresentations(ctx, struct, {
            includeLigands: styleOpts.ligands,
            includeSidechains: styleOpts.sidechains && !hasOverlay,
            includeSurfaceOverlay: styleOpts.surfaceOverlay && !hasOverlay,
            primaryRep: styleOpts.rep,
            colorArg: designedColorArg,
            alpha: designedAlpha,
          });
        }

        if (destroyed || mySeq !== loadSeq) return;

        // 2. Kabsch-aligned refolded structure colored by pLDDT or semantic B-factor
        if (sRefold && rPdb && !useTraj) {
          const rFmt = /^\s*(data_|loop_|_atom_site\.)/m.test(rPdb) ? 'mmcif' : 'pdb';
          const data = await ctx.builders.data.rawData({
            data: rPdb,
            label: 'LocalFold Refold (pLDDT)',
          });
          const traj = await ctx.builders.structure.parseTrajectory(data, rFmt);
          const model = await ctx.builders.structure.createModel(traj, { modelIndex: 0 });
          pruneFalseCyclicPolymers(traj, model, rPdb, StructConn);
          pruneFalseNucleicMonomers(traj, model);
          const struct = await ctx.builders.structure.createStructure(model, {
            name: 'model',
            params: {},
          });
          const refoldColorArg =
            styleOpts.color === 'uncertainty'
              ? resolveUncertaintyColorArg(struct, styleOpts.bfactorSemantics)
              : styleOpts.color !== 'auto'
                ? styleOpts.color === 'plddt-confidence' && !styleOpts.plddtAllowed
                  ? { color: 'secondary-structure' }
                  : { color: styleOpts.color }
                : useSemanticUncertainty
                  ? resolveUncertaintyColorArg(struct, styleOpts.bfactorSemantics)
                  : styleOpts.plddtAllowed
                    ? { color: 'plddt-confidence' }
                    : { color: 'uniform', colorParams: { value: Color(0x60a5fa) } };
          recordDomain(refoldColorArg);
          const refoldModelId = model.data?.id;
          if (typeof refoldModelId === 'string') {
            nextSemantics.set(refoldModelId, styleOpts.bfactorSemantics);
          }
          await ctx.builders.structure.representation.addRepresentation(struct, {
            type: styleOpts.rep,
            ...refoldColorArg,
            typeParams: buildPrimaryTypeParams(styleOpts.rep, struct),
          });
          await addAccessoryRepresentations(ctx, struct, {
            includeLigands: styleOpts.ligands,
            includeSidechains: styleOpts.sidechains,
            includeSurfaceOverlay: styleOpts.surfaceOverlay,
            primaryRep: styleOpts.rep,
            colorArg: refoldColorArg,
          });
        }

        if (destroyed || mySeq !== loadSeq) return;
        const requestedRender = Boolean(
          (sDesigned && (useTraj && tPdb ? tPdb : dPdb)) || (sRefold && rPdb && !useTraj),
        );
        const liveStructs = ctx.managers?.structure?.hierarchy?.current?.structures?.length ?? 0;
        modelSemanticsById = nextSemantics;
        structSpecIds = new Map();
        trackDomainsBySemantics = nextDomains;
        applyPostprocessing(illustrative, spinState.disableOcclusion);
        if (savedCam && liveStructs > 0) {
          ctx.canvas3d?.camera?.setState(savedCam, 0);
        } else {
          ctx.canvas3d?.requestCameraReset();
        }
        if (!requestedRender || liveStructs > 0) {
          lastStructureKey = styleOpts.structureKey;
          lastRenderedKey = styleOpts.fullKey;
        } else {
          lastStructureKey = '';
          lastRenderedKey = '';
        }
        lastAppliedOverpaintKey = '';
        sceneRenderEpoch += 1;
      } catch (err) {
        if (!destroyed && mySeq === loadSeq) {
          lastRenderedKey = '';
          initError = err instanceof Error ? err.message : String(err);
        }
      }
    });
  }

  async function scrubFrame(userFrameIdx: number) {
    if (destroyed || !plugin || !trajModelRef) return;
    if (scrubPending) {
      queuedFrame = userFrameIdx;
      return;
    }
    scrubPending = true;
    try {
      const ctx = plugin;
      const ref = trajModelRef;
      if (destroyed || !ctx || !ref) return;
      if (ctx.state?.data?.cells && !ctx.state.data.cells.has(ref)) return;
      await ctx.state.data
        .build()
        .to(ref)
        .update({ modelIndex: userFrameIdx + 1 })
        .commit();
    } catch {
      /* ignore stale trajModelRef while scene reloads */
    } finally {
      scrubPending = false;
      if (!destroyed && queuedFrame !== null) {
        const next = queuedFrame;
        queuedFrame = null;
        void scrubFrame(next);
      } else {
        queuedFrame = null;
      }
    }
  }

  function buildPostprocessingProps(illustrativeOn: boolean, disableOcclusion: boolean) {
    return {
      bloom: { name: 'off' as const, params: {} },
      shadow: { name: 'off' as const, params: {} },
      occlusion: disableOcclusion
        ? { name: 'off' as const, params: {} }
        : {
            name: 'on' as const,
            params: {
              samples: 16,
              multiScale: { name: 'off' as const, params: {} },
              radius: 5,
              bias: 0.8,
              blurKernelSize: 11,
              blurDepthBias: 0.5,
              resolutionScale: 0.5,
              color: 0x000000 as unknown as never,
              transparentThreshold: 0.4,
            },
          },
      antialiasing: {
        name: 'fxaa' as const,
        params: {
          edgeThresholdMin: 0.0312,
          edgeThresholdMax: 0.063,
          iterations: 12,
          subpixelQuality: 0.3,
        },
      },
      outline: illustrativeOn
        ? {
            name: 'on' as const,
            params: {
              scale: 1,
              threshold: 0.33,
              color: 0x000000 as unknown as never,
              includeTransparent: true,
            },
          }
        : { name: 'off' as const, params: {} },
      illustrative: illustrativeOn
        ? { name: 'on' as const, params: {} }
        : { name: 'off' as const, params: {} },
    };
  }

  function applyPostprocessing(illustrativeOn: boolean, disableOcclusion: boolean) {
    if (!plugin?.canvas3d) return;
    plugin.canvas3d.setProps({
      postprocessing: buildPostprocessingProps(illustrativeOn, disableOcclusion),
    });
  }

  let hasAnyStructure = $derived(
    Boolean((models && models.length > 0) || designedPdb || refoldPdb),
  );
  let spinState = $derived(
    resolveViewerSpinState({
      spin,
      hasAnyStructure,
      canonicalCamera,
      pauseSpin,
    }),
  );
  let effectiveSpin = $derived(spinState.effectiveSpin);
  // Distinct track semantics whose color ramp is actually on screen. Compare models carry their own
  // semantics: only visible painted ones count (Track mode under Auto, or any track model under
  // B-factor / Uncertainty); single-structure / multi-design views fall back to the prop.
  let legendSemantics = $derived.by((): TrackSemantics[] => {
    if (!hasAnyStructure || (colorScheme !== 'auto' && colorScheme !== 'uncertainty')) return [];
    if (models && models.some((m) => m.bfactorSemantics !== undefined)) {
      const out: TrackSemantics[] = [];
      for (const m of models) {
        if (!m.visible || !isTrackSemantics(m.bfactorSemantics)) continue;
        const painted = colorScheme === 'uncertainty' || Boolean(m.colorByTrack);
        if (painted && !out.includes(m.bfactorSemantics)) out.push(m.bfactorSemantics);
      }
      return out;
    }
    return isTrackSemantics(bfactorSemantics) ? [bfactorSemantics] : [];
  });
  let showSemanticBfactorLegend = $derived(legendSemantics.length > 0);
  // Alternative tracks of a merged cartoon stay in the scene hidden but get no chip of their own
  let legendModels = $derived(
    (models ?? []).filter((m) => m.mergedFrom?.role !== 'secondary'),
  );

  const BFACTOR_LEGEND_LABELS: Record<
    TrackSemantics,
    { lo: string; hi: string; aria: string; reversed: boolean }
  > = {
    'dynamics-prob': {
      lo: 'Rigid',
      hi: 'Flexible',
      aria: 'Dynamics probability B-factor scale: Rigid 0.0 to Flexible 1.0',
      reversed: false,
    },
    'ddg-stability': {
      lo: 'Stabilizing',
      hi: 'Destabilizing (ΔΔG)',
      aria: 'Stability B-factor scale: Stabilizing to Destabilizing (ΔΔG)',
      reversed: false,
    },
    'llr-fitness': {
      lo: 'Deleterious',
      hi: 'Beneficial (LLR)',
      aria: 'LLR fitness scale: Deleterious to Beneficial',
      reversed: true,
    },
    'gnina-contrib': {
      lo: 'Unfavorable',
      hi: 'Favorable (ΔCNN_VS)',
      aria: 'GNINA pocket contribution scale: Unfavorable to Favorable (ΔCNN_VS)',
      reversed: true,
    },
  };

  function formatLegendEnd(v: number, signed: boolean): string {
    const abs = Math.abs(v);
    const txt = abs >= 10 ? abs.toFixed(0) : abs.toFixed(1);
    if (v < 0 && Number(txt) !== 0) return `−${txt}`;
    return signed && v > 0 ? `+${txt}` : txt;
  }

  /** Numeric [left, right] legend endpoints (reversed ramps are drawn reversed, so min is always left). */
  function legendEnds(sem: TrackSemantics, domain: [number, number] | undefined): [string, string] | null {
    const d = domain ?? (sem === 'dynamics-prob' ? ([0, 1] as [number, number]) : undefined);
    if (!d) return null;
    const signed = sem !== 'dynamics-prob';
    return [
      formatLegendEnd(Math.min(d[0], d[1]), signed),
      formatLegendEnd(Math.max(d[0], d[1]), signed),
    ];
  }

  function trackSwatchBackground(sem: BfactorSemantics | undefined): string {
    return isTrackSemantics(sem) && BFACTOR_LEGEND_LABELS[sem].reversed
      ? 'linear-gradient(90deg, #dc2626, #f8fafc, #2563eb)'
      : 'linear-gradient(90deg, #2563eb, #f8fafc, #dc2626)';
  }

  function applySpin(
    on: boolean,
    speed: number,
    illustrativeOn = illustrative,
    disableOcclusion = on,
  ) {
    if (!plugin?.canvas3d) return;
    plugin.canvas3d.setProps({
      postprocessing: buildPostprocessingProps(illustrativeOn, disableOcclusion),
      trackball: {
        animate: on
          ? { name: 'spin', params: { speed, axis: [0, -1, 0] as unknown as never } }
          : { name: 'off', params: {} },
      },
    });
  }

  async function renderMultiModels(
    specs: ViewerModelSpec[],
    styleOpts: {
      rep: PolymerRepType;
      color: ColorSchemeOption;
      sidechains: boolean;
      surfaceOverlay: boolean;
      ligands: boolean;
      plddtAllowed: boolean;
      bfactorSemantics?: BfactorSemantics;
      canonicalCamera: boolean;
      preserveCamera: boolean;
      structureKey: string;
      fullKey: string;
    },
  ) {
    if (webglFailed) {
      initError = webglFailReason || 'WebGL2 is unavailable in this browser';
      return;
    }
    const mySeq = ++loadSeq;
    initError = null;
    await enqueueSceneRender(async () => {
      if (destroyed || mySeq !== loadSeq) return;
      modelRefs = new Map();
      structSpecIds = new Map();
      trajModelRef = null;
      if (specs.length === 0) {
        if (plugin) await plugin.clear();
        lastStructureKey = '';
        lastRenderedKey = styleOpts.fullKey;
        return;
      }

      try {
        const ctx = await ensurePlugin();
        if (destroyed || !ctx || mySeq !== loadSeq) return;
        const savedCam = styleOpts.preserveCamera ? ctx.canvas3d?.camera?.getSnapshot() : null;

        await ctx.clear();
        if (destroyed || mySeq !== loadSeq) return;

        const [{ Color }, { setSubtreeVisibility }, { StructConn }] = await Promise.all([
          import('molstar/lib/mol-util/color'),
          import('molstar/lib/mol-plugin/behavior/static/state'),
          import('molstar/lib/mol-model-formats/structure/property/bonds/struct_conn'),
        ]);
        if (destroyed || mySeq !== loadSeq) return;

        const useSemanticUncertainty = isTrackSemantics(styleOpts.bfactorSemantics);

        // Pass 1: create every structure and pool polymer Cα B-factors per track semantics across
        // ALL painted models (including hidden alternative tracks of a merged cartoon), so one
        // shared scale means the same color is the same value in every structure and switching
        // tracks needs no reload.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const built: Array<{ m: ViewerModelSpec; struct: any; sem: BfactorSemantics | undefined }> = [];
        const pooledEntries: Array<{ semantics: TrackSemantics; values: number[] }> = [];
        const nextSemantics = new Map<string, BfactorSemantics | undefined>();
        const nextStructSpecIds = new Map<string, string>();
        for (const m of specs) {
          const fmt = /^\s*(data_|loop_|_atom_site\.)/m.test(m.pdbText) ? 'mmcif' : 'pdb';
          const data = await ctx.builders.data.rawData({
            data: m.pdbText,
            label: m.label,
          });
          const traj = await ctx.builders.structure.parseTrajectory(data, fmt);
          const model = await ctx.builders.structure.createModel(traj, { modelIndex: 0 });
          if (destroyed || mySeq !== loadSeq) return;
          pruneFalseCyclicPolymers(traj, model, m.pdbText, StructConn);
          pruneFalseNucleicMonomers(traj, model);
          const struct = await ctx.builders.structure.createStructure(model, {
            name: 'model',
            params: {},
          });
          if (destroyed || mySeq !== loadSeq) return;
          const modelUuid = model.data?.id;
          if (typeof modelUuid === 'string') {
            nextSemantics.set(
              modelUuid,
              m.bfactorSemantics ?? (m.colorByPlddt ? 'plddt-0-100' : undefined),
            );
          }
          if (struct?.ref) nextStructSpecIds.set(struct.ref, m.id);
          if (
            isTrackSemantics(m.bfactorSemantics) &&
            (m.colorByTrack || styleOpts.color === 'uncertainty')
          ) {
            pooledEntries.push({ semantics: m.bfactorSemantics, values: collectPolymerBfactors(struct) });
          }
          built.push({ m, struct, sem: m.bfactorSemantics ?? styleOpts.bfactorSemantics });
        }

        const pooledArgs: Map<string, UncertaintyColorArg> = poolUncertaintyColorArgs(pooledEntries);
        const nextDomains: Record<string, [number, number]> = {};
        for (const [sem, arg] of pooledArgs) {
          if (arg.colorParams?.domain) nextDomains[sem] = arg.colorParams.domain;
        }

        // Pass 2: per-representation themes, so painted track models and muted companions share one scene
        const nextRefs = new Map<string, string[]>();
        for (const { m, struct, sem } of built) {
          const trackColorArg = (): UncertaintyColorArg =>
            (isTrackSemantics(sem) ? pooledArgs.get(sem) : undefined) ??
            resolveUncertaintyColorArg(struct, sem);
          // pLDDT coloring would misread Dyna-1 / ΔΔG / LLR / GNINA B-factors as confidence
          const effectiveColor =
            styleOpts.color === 'plddt-confidence' &&
            (!styleOpts.plddtAllowed || isTrackSemantics(m.bfactorSemantics))
              ? 'auto'
              : styleOpts.color;
          const modelColorArg =
            effectiveColor === 'uncertainty'
              ? trackColorArg()
              : effectiveColor !== 'auto'
                ? { color: effectiveColor }
                : m.colorByTrack && isTrackSemantics(sem)
                  ? trackColorArg()
                  : useSemanticUncertainty && (m.colorByPlddt || specs.length === 1)
                    ? resolveUncertaintyColorArg(struct, styleOpts.bfactorSemantics)
                    : m.colorByPlddt && styleOpts.plddtAllowed
                      ? { color: 'plddt-confidence' }
                      : { color: 'uniform', colorParams: { value: Color(m.colorHex) } };
          const repr = await ctx.builders.structure.representation.addRepresentation(struct, {
            type: styleOpts.rep,
            ...modelColorArg,
            typeParams: buildPrimaryTypeParams(styleOpts.rep, struct, m.alpha),
          });
          if (destroyed || mySeq !== loadSeq) return;
          const extraRefs = await addAccessoryRepresentations(ctx, struct, {
            includeLigands: styleOpts.ligands,
            includeSidechains: styleOpts.sidechains,
            includeSurfaceOverlay: styleOpts.surfaceOverlay,
            primaryRep: styleOpts.rep,
            colorArg: modelColorArg,
            alpha: m.alpha,
          });
          if (destroyed || mySeq !== loadSeq) return;
          const refs = [repr?.ref ?? struct.ref, ...extraRefs];
          nextRefs.set(m.id, refs);
          if (!m.visible) {
            for (const ref of refs) {
              if (ctx.state?.data?.cells?.has(ref)) {
                setSubtreeVisibility(ctx.state.data, ref, true);
              }
            }
          }
        }

        if (destroyed || mySeq !== loadSeq) return;
        const liveStructs = ctx.managers?.structure?.hierarchy?.current?.structures?.length ?? 0;
        modelRefs = nextRefs;
        modelSemanticsById = nextSemantics;
        structSpecIds = nextStructSpecIds;
        trackDomainsBySemantics = nextDomains;
        applyPostprocessing(illustrative, spinState.disableOcclusion);
        if (savedCam && liveStructs > 0) {
          ctx.canvas3d?.camera?.setState(savedCam, 0);
        } else {
          resetViewerCamera(ctx, styleOpts.canonicalCamera);
        }
        if (liveStructs > 0) {
          lastStructureKey = styleOpts.structureKey;
          lastRenderedKey = styleOpts.fullKey;
        } else {
          lastStructureKey = '';
          lastRenderedKey = '';
        }
        lastAppliedOverpaintKey = '';
        sceneRenderEpoch += 1;
      } catch (err) {
        if (!destroyed && mySeq === loadSeq) {
          lastRenderedKey = '';
          initError = err instanceof Error ? err.message : String(err);
        }
      }
    });
  }

  async function syncModelVisibility(specs: ViewerModelSpec[]) {
    if (destroyed || !plugin || modelRefs.size === 0) return;
    try {
      const { setSubtreeVisibility } = await import('molstar/lib/mol-plugin/behavior/static/state');
      if (destroyed || !plugin) return;
      for (const m of specs) {
        const refs = modelRefs.get(m.id);
        if (!refs) continue;
        for (const ref of refs) {
          if (plugin.state?.data?.cells?.has(ref)) {
            // Mol* setSubtreeVisibility(state, ref, isHidden): true hides, false shows
            setSubtreeVisibility(plugin.state.data, ref, !m.visible);
          }
        }
      }
    } catch {
      /* ignore transient state updates */
    }
  }

  // ---------------------------------------------------------------------------
  // Zero-reload GPU Overpaint, Hover Highlight, and Camera Focus (Node Studio)
  // ---------------------------------------------------------------------------

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function collectOverpaintComponents(ctx: any): any[] {
    const out: any[] = [];
    const structs = ctx?.managers?.structure?.hierarchy?.current?.structures ?? [];
    for (const s of structs) {
      for (const c of s.components ?? []) {
        if (c?.representations?.length > 0) out.push(c);
      }
      const directReprs = (s.genericRepresentations ?? []).filter(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (r: any) => Boolean(r?.cell?.obj?.data?.sourceData),
      );
      if (directReprs.length > 0) {
        out.push({ representations: directReprs });
      }
    }
    return out;
  }

  function resolveLayerResidues(
    layer: ResidueHighlightLayer,
  ): Array<{ chain: string; seqId: number; insCode: string }> {
    if (layer.residues && layer.residues.length > 0) {
      return layer.residues.map((r) => ({
        chain: (r.chain || 'A').trim(),
        seqId: r.seqId,
        insCode: (r.insCode ?? '').trim(),
      }));
    }
    if (layer.tokens && layer.tokens.length > 0) {
      const parsed = expandResidueTokenList(layer.tokens.join(','));
      return parsed.tokens.map((t) => ({
        chain: t.chain,
        seqId: t.seqId,
        insCode: t.insCode,
      }));
    }
    return [];
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function buildResiduesExpression(
    MS: any,
    residues: ReadonlyArray<{ chain: string; seqId: number; insCode: string }>,
  ) {
    if (residues.length === 0) return null;
    // Group by (chain, insCode) so large residue sets share one fast `core.set.has` query per chain
    const grouped = new Map<string, { chain: string; insCode: string; seqIds: number[] }>();
    for (const r of residues) {
      const key = `${r.chain}::${r.insCode}`;
      let g = grouped.get(key);
      if (!g) {
        g = { chain: r.chain, insCode: r.insCode, seqIds: [] };
        grouped.set(key, g);
      }
      g.seqIds.push(r.seqId);
    }

    const queries = [...grouped.values()].map((g) => {
      const seqTest =
        g.seqIds.length === 1
          ? MS.core.rel.eq([MS.struct.atomProperty.macromolecular.auth_seq_id(), g.seqIds[0]])
          : MS.core.set.has([
              MS.set(...g.seqIds),
              MS.struct.atomProperty.macromolecular.auth_seq_id(),
            ]);
      const insTest = MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.pdbx_PDB_ins_code(),
        g.insCode,
      ]);
      return MS.struct.generator.atomGroups({
        'chain-test': MS.core.rel.eq([
          MS.struct.atomProperty.macromolecular.auth_asym_id(),
          g.chain,
        ]),
        'residue-test': MS.core.logic.and([seqTest, insTest]),
      });
    });

    return queries.length === 1 ? queries[0] : MS.struct.combinator.merge(queries);
  }

  /**
   * Resolve a residue token to Mol* loci, skipping hidden compare models and (when `modelId`
   * names a rendered model) targeting that model only, so click-to-focus lands on the right
   * structure in Side-by-Side and merged multi-track overlays.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async function resolveTokenLoci(
    ctx: any,
    token: string,
    opts: { modelId?: string | null } = {},
  ): Promise<any | null> {
    const trimmed = token.trim();
    let target: { chain: string; seqId: number; insCode: string } | null = null;
    const colon = /^([A-Za-z0-9_]+):(-?\d+)([A-Za-z]?)$/.exec(trimmed);
    if (colon) {
      target = {
        chain: colon[1],
        seqId: Number.parseInt(colon[2], 10),
        insCode: colon[3] ? colon[3].toUpperCase() : '',
      };
    } else {
      const parsed = expandResidueTokenList(trimmed);
      const first = parsed.tokens[0];
      if (first) {
        target = { chain: first.chain, seqId: first.seqId, insCode: first.insCode };
      } else {
        const multi = /^([A-Za-z0-9_]*[A-Za-z_])(-?\d+)([A-Za-z]?)$/.exec(trimmed);
        if (multi) {
          target = {
            chain: multi[1],
            seqId: Number.parseInt(multi[2], 10),
            insCode: multi[3] ? multi[3].toUpperCase() : '',
          };
        }
      }
    }
    if (!target || !Number.isFinite(target.seqId)) return null;
    const structs = ctx?.managers?.structure?.hierarchy?.current?.structures ?? [];
    if (structs.length === 0) return null;

    const [{ MolScriptBuilder: MS }, { Script }, { StructureSelection }, { Loci }] =
      await Promise.all([
        import('molstar/lib/mol-script/language/builder'),
        import('molstar/lib/mol-script/script'),
        import('molstar/lib/mol-model/structure'),
        import('molstar/lib/mol-model/loci'),
      ]);
    const expr = buildResiduesExpression(MS, [target]);
    if (!expr) return null;

    const specs = untrack(() => models) ?? [];
    const hiddenSpecIds = new Set(specs.filter((m) => !m.visible).map((m) => m.id));
    const renderedSpecIds = new Set(structSpecIds.values());
    const targetSpecId = opts.modelId && renderedSpecIds.has(opts.modelId) ? opts.modelId : null;

    for (const sRef of structs) {
      const structData = sRef?.cell?.obj?.data;
      if (!structData) continue;
      const specId = structSpecIds.get(sRef?.cell?.transform?.ref);
      if (specId && hiddenSpecIds.has(specId)) continue;
      if (targetSpecId && specId !== targetSpecId) continue;
      const sel = Script.getStructureSelection(expr, structData);
      const loci = StructureSelection.toLociWithSourceUnits(sel);
      if (loci && !Loci.isEmpty(loci)) {
        return loci;
      }
    }
    return null;
  }

  async function applyHighlightOverpaint(layers: ResidueHighlightLayer[], sigKey: string) {
    if (destroyed || !plugin) return;
    const mySeq = ++overpaintSeq;
    try {
      const ctx = plugin;
      const components = collectOverpaintComponents(ctx);
      if (components.length === 0) return;

      const [
        { setStructureOverpaint, clearStructureOverpaint },
        { setStructureTransparency, clearStructureTransparency },
        { MolScriptBuilder: MS },
        { Script },
        { StructureSelection },
        { Color },
        { EmptyLoci },
      ] = await Promise.all([
        import('molstar/lib/mol-plugin-state/helpers/structure-overpaint'),
        import('molstar/lib/mol-plugin-state/helpers/structure-transparency'),
        import('molstar/lib/mol-script/language/builder'),
        import('molstar/lib/mol-script/script'),
        import('molstar/lib/mol-model/structure'),
        import('molstar/lib/mol-util/color'),
        import('molstar/lib/mol-model/loci'),
      ]);
      if (destroyed || mySeq !== overpaintSeq || !plugin) return;

      // Clear previous overpaint & transparency without clearing structure or touching camera
      await clearStructureOverpaint(ctx, components);
      await clearStructureTransparency(ctx, components);
      if (destroyed || mySeq !== overpaintSeq || !plugin) return;

      for (const layer of layers) {
        const residues = resolveLayerResidues(layer);
        if (residues.length === 0) continue;
        const expr = buildResiduesExpression(MS, residues);
        if (!expr) continue;

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const lociGetter = async (structure: any) => {
          try {
            const sel = Script.getStructureSelection(expr, structure);
            return StructureSelection.toLociWithSourceUnits(sel);
          } catch {
            return EmptyLoci;
          }
        };

        await setStructureOverpaint(ctx, components, Color(layer.colorHex), lociGetter);
        if (destroyed || mySeq !== overpaintSeq || !plugin) return;

        if (typeof layer.alpha === 'number' && layer.alpha >= 0 && layer.alpha < 0.99) {
          const transparency = Math.max(0, Math.min(1, 1 - layer.alpha));
          await setStructureTransparency(ctx, components, transparency, lociGetter);
          if (destroyed || mySeq !== overpaintSeq || !plugin) return;
        }
      }

      lastAppliedOverpaintKey = sigKey;
    } catch {
      /* degrade safely if Mol* internal structure state is mid-transition */
    }
  }

  let highlightSig = $derived.by((): string => {
    if (!highlightLayers || highlightLayers.length === 0) return '';
    return highlightLayers
      .map((l) => {
        const resPart =
          l.residues && l.residues.length > 0
            ? l.residues.map((r) => `${r.chain}:${r.seqId}:${r.insCode ?? ''}`).join(',')
            : (l.tokens ?? []).join(',');
        return `${l.id}|${l.colorHex}|${l.alpha ?? ''}|${resPart}`;
      })
      .join(';;');
  });

  // Zero-reload GPU overpaint effect: triggers ONLY on serialized highlightSig or scene load completion
  $effect(() => {
    const epoch = sceneRenderEpoch;
    const sig = highlightSig;
    const ctx = plugin;
    if (!ctx || epoch === 0) return;
    const key = `${epoch}##${sig}`;
    if (key === lastAppliedOverpaintKey) return;
    const layersSnapshot = untrack(() => highlightLayers ?? []);
    if (layersSnapshot.length === 0 && !lastAppliedOverpaintKey) {
      lastAppliedOverpaintKey = key;
      return;
    }
    void applyHighlightOverpaint(layersSnapshot, key);
  });

  // Lightweight 3D hover highlight synced with hoveredToken
  $effect(() => {
    const tok = hoveredToken;
    const ctx = plugin;
    const epoch = sceneRenderEpoch;
    if (!ctx || epoch === 0) return;
    let cancelled = false;
    (async () => {
      try {
        if (!tok) {
          ctx.managers?.interactivity?.lociHighlights?.clearHighlights?.();
          return;
        }
        const loci = await resolveTokenLoci(ctx, tok);
        if (cancelled || destroyed) return;
        if (loci) {
          ctx.managers?.interactivity?.lociHighlights?.highlightOnly?.({ loci });
        } else {
          ctx.managers?.interactivity?.lociHighlights?.clearHighlights?.();
        }
      } catch {
        /* ignore hover highlight errors */
      }
    })();
    return () => {
      cancelled = true;
    };
  });

  // Smooth camera focus when focusToken changes
  $effect(() => {
    const tok = focusToken;
    const targetModelId = focusModelId;
    const ctx = plugin;
    const epoch = sceneRenderEpoch;
    if (!ctx || epoch === 0 || !tok) return;
    let cancelled = false;
    (async () => {
      try {
        const loci = await resolveTokenLoci(ctx, tok, { modelId: targetModelId });
        if (cancelled || destroyed || !loci) return;
        ctx.managers?.camera?.focusLoci?.(loci, { durationMs: 250 });
      } catch {
        /* ignore camera focus errors */
      }
    })();
    return () => {
      cancelled = true;
    };
  });

  // Structural signature for multi-model mode (excludes `visible` so eye/solo toggles never reload or
  // reset camera, and excludes color/alpha/semantics so By Model | pLDDT | Track keeps the camera)
  let multiRenderKey = $derived.by((): string | null => {
    if (models === null) return null;
    return models
      .map((m) => `${m.id}|${m.pdbText.length}|${hashStructureSample(m.pdbText)}`)
      .join('||');
  });

  // Per-model appearance signature (part of the style key: re-renders with the camera preserved)
  let multiStyleKey = $derived.by((): string => {
    if (models === null) return '';
    return models
      .map(
        (m) =>
          `${m.id}:${m.colorHex}:${m.alpha}:${m.colorByPlddt ? 1 : 0}:${m.colorByTrack ? 1 : 0}:${m.bfactorSemantics ?? ''}`,
      )
      .join(',');
  });

  // Re-render full scene ONLY when (structureKey + styleKey) actually changes
  $effect(() => {
    void forceRenderTick;
    const canonical = canonicalCamera;
    const styleOpts = {
      rep: repType,
      color: colorScheme,
      sidechains: showSidechains,
      surfaceOverlay: showSurfaceOverlay,
      ligands: showLigands,
      plddtAllowed,
      bfactorSemantics,
    };
    const styleKey = `${styleOpts.rep}|${styleOpts.color}|${styleOpts.sidechains ? 1 : 0}|${styleOpts.surfaceOverlay ? 1 : 0}|${styleOpts.ligands ? 1 : 0}|${styleOpts.plddtAllowed ? 1 : 0}|${styleOpts.bfactorSemantics ?? ''}`;
    if (multiRenderKey !== null) {
      const mKey = `multi:${canonical ? 1 : 0}:${multiRenderKey}`;
      const fullKey = `${mKey}##${styleKey}##${multiStyleKey}`;
      if (fullKey === lastRenderedKey && !initError) return;
      const sameStructure = lastStructureKey === mKey;
      const snapshot = untrack(() => models ?? []);
      void renderMultiModels(snapshot, {
        ...styleOpts,
        canonicalCamera: canonical,
        preserveCamera: sameStructure,
        structureKey: mKey,
        fullKey,
      });
      return;
    }
    const d = designedPdb;
    const r = refoldPdb;
    const t = trajectoryPdb;
    const useT = showTrajectory && Boolean(t);
    const sd = showDesigned;
    const sr = showRefold;
    const sKey = `single:${sd ? d?.length ?? 0 : -1}:${d?.slice(0, 48) ?? ''}:${d?.slice(-48) ?? ''}|${sr ? r?.length ?? 0 : -1}:${r?.slice(0, 48) ?? ''}:${r?.slice(-48) ?? ''}|${useT ? `${t?.length ?? 0}:${t?.slice(-48) ?? ''}` : 0}`;
    const fullKey = `${sKey}##${styleKey}`;
    if (fullKey === lastRenderedKey && !initError) return;
    const sameStructure = lastStructureKey === sKey;
    void renderScene(d, r, t, useT, sd, sr, untrack(() => frameIndex), {
      ...styleOpts,
      preserveCamera: sameStructure,
      structureKey: sKey,
      fullKey,
    });
  });

  // Camera-preserving incremental visibility sync when only `visible` flags change in compare mode
  $effect(() => {
    if (!models) return;
    const visSig = models.map((m) => `${m.id}:${m.visible ? 1 : 0}`).join(',');
    void visSig;
    void syncModelVisibility(models);
  });

  // Fast incremental scrub when only frameIndex changes
  $effect(() => {
    if (showTrajectory && frameIndex !== undefined && trajModelRef) {
      void scrubFrame(frameIndex);
    }
  });

  $effect(() => {
    const illus = illustrative;
    const { effectiveSpin: on, disableOcclusion } = spinState;
    const speed = effectiveSpinSpeed;
    if (plugin) applySpin(on, speed, illus, disableOcclusion);
  });

  $effect(() => {
    const mode = cameraMode;
    if (plugin?.canvas3d) {
      plugin.canvas3d.setProps({
        camera: { mode },
      });
    }
  });

  $effect(() => {
    const bg = theme === 'light' ? 0xf8fafc : 0x0d0e11;
    if (plugin?.canvas3d) {
      plugin.canvas3d.setProps({
        renderer: { backgroundColor: bg as unknown as never },
      });
    }
  });

  $effect(() => {
    const el = containerEl;
    const ctx = plugin;
    if (!el || !ctx?.canvas3d || typeof ResizeObserver === 'undefined') return;
    let resizeRaf: number | null = null;
    const ro = new ResizeObserver(() => {
      if (resizeRaf !== null) return;
      resizeRaf = requestAnimationFrame(() => {
        resizeRaf = null;
        try {
          ctx.canvas3d?.handleResize();
        } catch {
          /* ignore resize during teardown */
        }
      });
    });
    ro.observe(el);
    return () => {
      if (resizeRaf !== null) {
        cancelAnimationFrame(resizeRaf);
        resizeRaf = null;
      }
      ro.disconnect();
    };
  });

  $effect(() => {
    const el = containerEl;
    const loop = plugin?.animationLoop;
    if (!el || !loop) return;
    let inViewport = true;
    let observing = true;
    function syncAnimation() {
      if (!observing) return;
      if (document.hidden || !inViewport) {
        if (loop.isAnimating) loop.stop({ noDraw: true });
      } else if (!loop.isAnimating) {
        loop.start();
      }
    }
    const observer = typeof IntersectionObserver === 'undefined' ? null : new IntersectionObserver((entries) => {
      if (entries.length === 0) return;
      inViewport = entries[entries.length - 1].isIntersecting;
      syncAnimation();
    });
    observer?.observe(el);
    document.addEventListener('visibilitychange', syncAnimation);
    syncAnimation();
    return () => {
      observing = false;
      observer?.disconnect();
      document.removeEventListener('visibilitychange', syncAnimation);
    };
  });

  onDestroy(() => {
    destroyed = true;
    loadSeq++;
    if (pngStatusTimer) clearTimeout(pngStatusTimer);
    try {
      plugin?.dispose();
    } catch {
      /* ignore */
    }
    plugin = null;
  });
</script>

{#snippet bfactorLegend(sem: TrackSemantics, domain: [number, number] | undefined, inHud: boolean)}
  {@const spec = BFACTOR_LEGEND_LABELS[sem]}
  {@const ends = legendEnds(sem, domain)}
  <div class:legend={!inHud} class:bfactor-legend={!inHud} class:bfactor-pill={inHud} aria-label={spec.aria}>
    <span>{spec.lo}</span>
    {#if ends}<span class="bfactor-end">{ends[0]}</span>{/if}
    <span class="bfactor-grad" class:is-reversed={spec.reversed} aria-hidden="true"></span>
    {#if ends}<span class="bfactor-end">{ends[1]}</span>{/if}
    <span>{spec.hi}</span>
  </div>
{/snippet}

<div class="mol-card" style="height: {typeof height === 'number' ? `${height}px` : height}">
  {#if label || (atomGranularity === 'ca-only' && hasAnyStructure) || (showSemanticBfactorLegend && models && models.length > 0)}
    <div class="hud-top-left" class:has-style-menu={hasAnyStructure}>
      {#if label}
        <div class="badge" title={label}>{label}</div>
      {/if}
      {#if atomGranularity === 'ca-only' && hasAnyStructure}
        <div
          class="granularity-pill"
          title="Only Cα backbone coordinates are present; Mol* renders an inferred smooth ribbon."
        >
          Cα-only (inferred ribbon)
        </div>
      {/if}
      {#if showSemanticBfactorLegend && models && models.length > 0}
        {#each legendSemantics as sem (sem)}
          {@render bfactorLegend(sem, trackDomainsBySemantics[sem], true)}
        {/each}
      {/if}
    </div>
  {/if}
  {#if hasAnyStructure}
    <div class="style-menu-wrap" bind:this={styleMenuWrapEl}>
      <button
        type="button"
        class="style-toggle-btn"
        class:active={styleMenuOpen}
        onclick={() => (styleMenuOpen = !styleMenuOpen)}
        title="Toggle structure representation, view & PNG export controls"
        aria-expanded={styleMenuOpen}
      >
        ⚙ {expanded ? 'Representation' : 'Style'} {styleMenuOpen ? '▾' : '▸'}
      </button>
      {#if styleMenuOpen}
        <div class="style-panel" role="region" aria-label="Structure representation controls">
          <label class="sp-field">
            <span>Polymer</span>
            <select name="mol-polymer-rep" bind:value={repType}>
              <option value="cartoon">Cartoon</option>
              <option value="ball-and-stick">Ball &amp; Stick</option>
              <option value="molecular-surface">Molecular Surface</option>
              <option value="gaussian-surface">Gaussian Surface</option>
              <option value="putty">Putty (B-factor / pLDDT)</option>
              <option value="backbone">Backbone</option>
              <option value="spacefill">Spacefill (VDW)</option>
            </select>
          </label>
          <label class="sp-field">
            <span>Color</span>
            <select name="mol-color-scheme" bind:value={colorScheme}>
              <option value="auto">Auto (Default)</option>
              <option value="chain-id">By Chain</option>
              <option value="plddt-confidence" disabled={!plddtAllowed}>
                {plddtAllowed ? 'pLDDT Confidence' : 'pLDDT Confidence (unavailable)'}
              </option>
              <option value="uncertainty">B-factor / Uncertainty</option>
              <option value="secondary-structure">Secondary Structure</option>
              <option value="hydrophobicity">Hydrophobicity</option>
              <option value="sequence-id">Rainbow (N → C)</option>
              <option value="element-symbol">Element</option>
            </select>
          </label>
          <div class="sp-checks">
            <label><input type="checkbox" name="mol-show-sidechains" bind:checked={showSidechains} /> All-atom sticks</label>
            <label><input type="checkbox" name="mol-show-surface" bind:checked={showSurfaceOverlay} /> Translucent surface</label>
            <label><input type="checkbox" name="mol-show-ligands" bind:checked={showLigands} /> Ligands &amp; ions</label>
          </div>
          <div class="sp-field">
            <span>Projection</span>
            <div class="sp-seg">
              <button
                type="button"
                class:active={cameraMode === 'perspective'}
                onclick={() => (cameraMode = 'perspective')}
              >
                Perspective
              </button>
              <button
                type="button"
                class:active={cameraMode === 'orthographic'}
                onclick={() => (cameraMode = 'orthographic')}
              >
                Ortho
              </button>
            </div>
          </div>
          {#if spinState.spinConfigured}
            <label class="sp-field">
              <span>
                Spin speed ({effectiveSpinSpeed.toFixed(3)} rot/s){pauseSpin && !effectiveSpin ? ' · paused' : ''}
              </span>
              <input
                type="range"
                min="0.005"
                max="0.15"
                step="0.005"
                value={effectiveSpinSpeed}
                oninput={(e) => (localSpinSpeed = Number(e.currentTarget.value))}
              />
            </label>
          {/if}
          <div class="sp-actions">
            <button type="button" onclick={handleResetCamera} title="Reset camera framing">
              ⟲ Reset view
            </button>
            <button
              type="button"
              class:is-failed={pngStatus === 'failed'}
              disabled={!plugin || Boolean(initError)}
              onclick={() => void handleSavePng()}
              title={!plugin || initError ? '3D viewer unavailable' : 'Download PNG screenshot'}
            >
              {pngStatus === 'saved' ? '✓ Saved' : pngStatus === 'failed' ? 'PNG failed' : '📷 PNG'}
            </button>
            <button type="button" onclick={resetStyleControls} title="Restore default style">
              ↺ Default
            </button>
          </div>
        </div>
      {/if}
    </div>
  {/if}
  {#if models && models.length > 0}
    <div class="compare-legend" role="group" aria-label="Compared 3D structures">
      {#each legendModels as m (m.id)}
        {@const merge = m.mergedFrom}
        <div class="cmp-chip" class:is-hidden={!m.visible} class:is-merged={Boolean(merge)}>
          <button
            type="button"
            class="cmp-chip-main"
            title={`${m.visible ? 'Hide' : 'Show'} ${merge ? merge.caption : m.label}`}
            onclick={() => onToggleModelVisibility?.(m.id)}
          >
            <span
              class="cmp-swatch"
              class:ref={m.isReference}
              class:is-track={Boolean(m.colorByTrack)}
              style="background: {m.colorByTrack ? trackSwatchBackground(m.bfactorSemantics) : m.colorCss}; opacity: {m.visible ? 1 : 0.35}"
            ></span>
            {#if m.badge}
              <span class="cmp-badge">{m.badge}</span>
            {/if}
            <span class="cmp-lbl">{m.label}</span>
            {#if m.metricSummary}
              <span class="cmp-stat">{m.metricSummary}</span>
            {/if}
          </button>
          {#if merge && merge.tracks.length > 1 && onSelectTrack}
            <span class="cmp-track-switch" role="group" aria-label="Painted track on merged cartoon">
              {#each merge.tracks as t (t.nodeId)}
                <button
                  type="button"
                  class:active={t.nodeId === merge.activeTrackNodeId}
                  aria-pressed={t.nodeId === merge.activeTrackNodeId}
                  title={`Show ${t.label} on this backbone`}
                  onclick={() => onSelectTrack?.(merge.groupId, t.nodeId)}
                >
                  {t.badge}
                </button>
              {/each}
            </span>
          {/if}
          {#if merge}
            <span
              class="cmp-merge-tag"
              title={`${merge.caption} — identical structures merged (${merge.evidence === 'lineage' ? 'linked by structure lineage' : 'matched by geometry'})`}
            >
              ⧉ merged · {merge.rmsd.toFixed(2)} Å · {merge.evidence}
            </span>
            {#if onToggleMergeIdentical}
              <button
                type="button"
                class="split-btn"
                title="Show the merged identical structures as separate models"
                onclick={() => onToggleMergeIdentical?.()}
              >
                Split
              </button>
            {/if}
          {/if}
          {#if onSoloModel && !m.isReference && legendModels.length > 1}
            <button
              type="button"
              class="solo-btn"
              title={`Solo ${m.label} (click again to show all)`}
              onclick={() => onSoloModel?.(m.id)}
            >
              S
            </button>
          {/if}
        </div>
      {/each}
    </div>
  {:else if showSemanticBfactorLegend}
    {@render bfactorLegend(legendSemantics[0], trackDomainsBySemantics[legendSemantics[0]], false)}
  {:else if designedPdb && refoldPdb && showDesigned && showRefold && !showTrajectory}
    <div class="legend">
      <span class="swatch designed"></span> Designed
      <span class="swatch plddt"></span> Refold (pLDDT)
    </div>
  {/if}
  {#if hoveredResidue && hasAnyStructure}
    <div
      class="hover-readout"
      class:has-compare-legend={Boolean(models && models.length > 0)}
      role="status"
      aria-live="polite"
    >
      <span class="hr-res">{hoveredResidueLabel}</span>
      <span class="hr-sep" aria-hidden="true">·</span>
      <span class="hr-chain">Chain {hoveredResidue.chain}</span>
      {#if hoveredResidue.isNonPolymer && hoveredResidue.atomName}
        <span class="hr-sep" aria-hidden="true">·</span>
        <span class="hr-meta">{hoveredResidue.atomName}</span>
      {:else if hoveredMetricText}
        <span class="hr-sep" aria-hidden="true">·</span>
        <span class="hr-metric">{hoveredMetricText}</span>
      {/if}
    </div>
  {/if}
  <div
    bind:this={containerEl}
    class="canvas-wrap"
    role="presentation"
    onmouseleave={() => (hoveredResidue = null)}
  >
    <canvas bind:this={canvasEl} class="mol-canvas"></canvas>
  </div>
  {#if (!models || models.length === 0) && !designedPdb && !refoldPdb}
    <div class="empty">
      <span>
        {emptyMessage ??
          (models
            ? 'Run the selected blocks to compare their 3D structures'
            : 'Run a pipeline to visualize designed & refolded 3D structures')}
      </span>
      {#if emptyActionLabel && onEmptyAction}
        <button type="button" class="empty-action-btn" onclick={onEmptyAction}>
          {emptyActionLabel}
        </button>
      {/if}
    </div>
  {/if}
  {#if initError}
    <div class="err-overlay">3D viewer error: {initError}</div>
  {/if}
</div>

<style>
  .mol-card {
    position: relative;
    background: var(--bg-canvas);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .canvas-wrap,
  .mol-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  .hud-top-left {
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 10;
    max-width: calc(100% - 20px);
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 5px;
    pointer-events: none;
  }
  .hud-top-left.has-style-menu {
    max-width: calc(100% - 175px);
  }
  .badge {
    max-width: 100%;
    box-sizing: border-box;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-secondary);
    background: color-mix(in srgb, var(--bg-surface) 92%, transparent);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 4px;
    pointer-events: none;
  }
  .granularity-pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 600;
    color: var(--warn-text);
    background: color-mix(in srgb, var(--warn-bg) 92%, transparent);
    border: 1px solid var(--warn-border);
    padding: 2px 7px;
    border-radius: 999px;
    pointer-events: auto;
    white-space: nowrap;
  }
  .legend {
    position: absolute;
    bottom: 10px;
    left: 10px;
    z-index: 10;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--overlay-bg);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 4px;
    pointer-events: none;
  }
  .hover-readout {
    position: absolute;
    bottom: 10px;
    right: 10px;
    z-index: 11;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-primary);
    background: color-mix(in srgb, var(--bg-surface) 94%, transparent);
    border: 1px solid var(--border-strong);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    padding: 3px 9px;
    border-radius: 5px;
    pointer-events: none;
    white-space: nowrap;
  }
  .hover-readout.has-compare-legend {
    bottom: 42px;
  }
  .hr-res {
    font-weight: 600;
    color: var(--text-primary);
  }
  .hr-sep {
    color: var(--text-dim);
  }
  .hr-chain,
  .hr-meta {
    color: var(--text-muted);
  }
  .hr-metric {
    color: var(--accent-bright);
    font-weight: 600;
  }
  .compare-legend {
    position: absolute;
    bottom: 10px;
    left: 10px;
    right: 10px;
    z-index: 10;
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    pointer-events: none;
  }
  .cmp-chip {
    pointer-events: auto;
    display: inline-flex;
    align-items: center;
    background: color-mix(in srgb, var(--bg-surface) 92%, transparent);
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    overflow: hidden;
    transition: opacity 120ms ease;
  }
  .cmp-chip.is-hidden {
    opacity: 0.5;
  }
  .cmp-chip-main {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: transparent;
    border: none;
    color: var(--text-primary);
    padding: 3px 7px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    cursor: pointer;
  }
  .cmp-chip-main:hover {
    background: var(--bg-surface-hover);
  }
  .cmp-swatch {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .cmp-swatch.ref {
    border: 1px dashed rgba(255, 255, 255, 0.5);
  }
  .cmp-swatch.is-track {
    width: 14px;
    border-radius: 999px;
  }
  .cmp-track-switch {
    display: inline-flex;
    border-left: 1px solid var(--border-subtle);
  }
  .cmp-track-switch button,
  .split-btn {
    border: none;
    background: transparent;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 700;
    padding: 3px 5px;
    cursor: pointer;
  }
  .cmp-track-switch button.active {
    background: var(--accent);
    color: #fff;
  }
  .cmp-track-switch button:hover:not(.active),
  .split-btn:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }
  .cmp-merge-tag {
    border-left: 1px solid var(--border-subtle);
    color: var(--text-dim);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    padding: 3px 6px;
    white-space: nowrap;
  }
  .split-btn {
    border-left: 1px solid var(--border-subtle);
    font-weight: 600;
  }
  .cmp-badge {
    font-weight: 700;
    color: var(--text-primary);
  }
  .cmp-lbl {
    max-width: 115px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cmp-stat {
    color: var(--text-dim);
  }
  .solo-btn {
    border: none;
    border-left: 1px solid var(--border-subtle);
    background: transparent;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 700;
    padding: 3px 6px;
    cursor: pointer;
  }
  .solo-btn:hover {
    background: var(--accent);
    color: #fff;
  }
  .swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    display: inline-block;
  }
  .swatch.designed {
    background: rgba(148, 163, 184, 0.65);
    border: 1px solid #94a3b8;
  }
  .swatch.plddt {
    background: linear-gradient(90deg, #ff7d45, #ffdb13, #65cbf3, #0053d6);
  }
  .bfactor-legend,
  .bfactor-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--text-secondary);
    background: color-mix(in srgb, var(--bg-surface) 92%, transparent);
    border: 1px solid var(--border-strong);
    padding: 3px 9px;
    border-radius: 999px;
    white-space: nowrap;
  }
  .bfactor-grad {
    width: 48px;
    height: 6px;
    border-radius: 999px;
    border: 1px solid var(--border-subtle);
    background: linear-gradient(90deg, #2563eb 0%, #f8fafc 50%, #dc2626 100%);
    flex-shrink: 0;
  }
  .bfactor-grad.is-reversed {
    background: linear-gradient(90deg, #dc2626 0%, #f8fafc 50%, #2563eb 100%);
  }
  .bfactor-end {
    color: var(--text-dim);
    font-variant-numeric: tabular-nums;
  }
  .empty {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 18px;
    text-align: center;
    color: var(--text-dim);
    font-size: 13px;
    pointer-events: none;
  }
  .empty-action-btn {
    pointer-events: auto;
    background: var(--bg-surface);
    color: var(--text-body);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    padding: 5px 11px;
    font-size: 11.5px;
    font-weight: 500;
    cursor: pointer;
  }
  .empty-action-btn:hover {
    border-color: var(--accent);
    color: var(--text-primary);
    background: var(--bg-surface-hover);
  }
  .err-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    background: var(--err-bg);
    color: var(--err-text);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
  }
  .style-menu-wrap {
    position: absolute;
    top: 10px;
    right: 10px;
    max-height: calc(100% - 20px);
    z-index: 15;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
    pointer-events: none;
  }
  .style-toggle-btn {
    pointer-events: auto;
    flex-shrink: 0;
    background: color-mix(in srgb, var(--bg-surface) 92%, transparent);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }
  .style-toggle-btn:hover,
  .style-toggle-btn.active {
    border-color: var(--accent);
    color: var(--accent-bright);
  }
  .style-panel {
    pointer-events: auto;
    width: 252px;
    max-width: calc(100% - 20px);
    max-height: min(420px, calc(100vh - 120px));
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    box-sizing: border-box;
    background: color-mix(in srgb, var(--bg-surface) 95%, transparent);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 10px 11px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    display: flex;
    flex-direction: column;
    gap: 9px;
    font-size: 11.5px;
    color: var(--text-body);
  }
  .sp-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }
  .sp-field > span {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.35px;
    color: var(--text-dim);
    font-weight: 600;
  }
  .sp-field select {
    width: 100%;
    max-width: 100%;
    min-width: 0;
    box-sizing: border-box;
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 4px 6px;
    font-size: 11.5px;
    font-family: inherit;
    cursor: pointer;
  }
  .sp-field input[type='range'] {
    width: 100%;
    accent-color: var(--accent);
    cursor: pointer;
  }
  .sp-checks {
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 2px 0;
  }
  .sp-checks label {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    font-size: 11.5px;
    color: var(--text-secondary);
  }
  .sp-seg {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    overflow: hidden;
  }
  .sp-seg button {
    border: none;
    background: var(--bg-surface);
    color: var(--text-muted);
    font-size: 10.5px;
    font-weight: 600;
    padding: 4px 6px;
    cursor: pointer;
  }
  .sp-seg button.active {
    background: var(--accent);
    color: #fff;
  }
  .sp-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    padding-top: 2px;
    border-top: 1px solid var(--border-subtle);
  }
  .sp-actions button {
    flex: 1 1 auto;
    min-width: 0;
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    cursor: pointer;
    white-space: nowrap;
  }
  .sp-actions button:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--text-primary);
  }
  .sp-actions button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .sp-actions button.is-failed {
    border-color: var(--err-border);
    color: var(--err-strong);
  }
  @media (max-width: 640px) {
    .hud-top-left.has-style-menu {
      max-width: calc(100% - 155px);
    }
    .badge {
      font-size: 10px;
    }
    .style-toggle-btn {
      font-size: 10px;
      padding: 3px 8px;
    }
    .style-panel {
      width: min(230px, calc(100vw - 32px));
    }
  }
</style>
