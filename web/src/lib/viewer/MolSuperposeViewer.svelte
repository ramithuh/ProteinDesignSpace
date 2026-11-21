<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import type { ViewerModelSpec } from '$lib/engine/compare';

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
    /** When true (full-view expanded mode), shows the floating Representation & View menu. */
    expanded?: boolean;
    theme?: 'dark' | 'light';
    height?: number | string;
    label?: string;
    emptyMessage?: string;
    emptyActionLabel?: string;
    onEmptyAction?: () => void;
    onResidueClick?: (info: {
      chain: string;
      seqId: number;
      insCode?: string;
      residueToken?: string;
      compId: string;
      isNonPolymer?: boolean;
    }) => void;
    onToggleModelVisibility?: (modelId: string) => void;
    onSoloModel?: (modelId: string) => void;
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
    expanded = false,
    theme = 'dark',
    height = 420,
    label = undefined,
    emptyMessage = undefined,
    emptyActionLabel = undefined,
    onEmptyAction = undefined,
    onResidueClick = undefined,
    onToggleModelVisibility = undefined,
    onSoloModel = undefined,
  }: Props = $props();

  let containerEl: HTMLDivElement | null = $state(null);
  let canvasEl: HTMLCanvasElement | null = $state(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let plugin: any = $state(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let pluginPromise: Promise<any> | null = null;
  let destroyed = false;
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
    | 'secondary-structure'
    | 'hydrophobicity'
    | 'sequence-id'
    | 'element-symbol';

  let styleMenuOpen = $state(typeof window !== 'undefined' ? window.innerWidth >= 640 : true);
  let repType = $state<PolymerRepType>('cartoon');
  let colorScheme = $state<ColorSchemeOption>('auto');
  let showSidechains = $state(false);
  let showSurfaceOverlay = $state(false);
  let showLigands = $state(true);
  let cameraMode = $state<'perspective' | 'orthographic'>('perspective');
  let localSpinSpeed = $state<number | null>(null);
  let effectiveSpinSpeed = $derived(localSpinSpeed ?? spinSpeed);
  let lastStructureKey = '';
  let lastRenderedKey = '';

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

  function handleResetCamera() {
    plugin?.canvas3d?.requestCameraReset();
  }

  async function handleSavePng() {
    if (!canvasEl || !plugin || initError || webglFailed) return;
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
      if (!url) return;
      const a = document.createElement('a');
      a.href = url;
      a.download = `${(label || 'structure').replace(/[^a-z0-9._-]+/gi, '_').slice(0, 48)}.png`;
      a.click();
    } catch {
      /* ignore if canvas cannot export */
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
        const pixelScale = Math.min(1, 1.5 / dpr);

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

        ctx.canvas3dContext?.setProps?.({ pixelScale });
        ctx.canvas3d?.setProps({
          renderer: { backgroundColor: (theme === 'light' ? 0xf8fafc : 0x0d0e11) as unknown as never },
          cameraFog: { name: 'off', params: {} },
          cameraClipping: { far: false, radius: 0, minNear: 0.1 },
          postprocessing: buildPostprocessingProps(illustrative, spin && hasAnyStructure),
          trackball: {
            animate: spin && hasAnyStructure
              ? { name: 'spin', params: { speed: effectiveSpinSpeed, axis: [0, -1, 0] as unknown as never } }
              : { name: 'off', params: {} },
          },
        });

        // Click picking -> emit protein residue info (including insertion code, ignoring DNA/RNA & ligands)
        const { StructureElement, StructureProperties } = await import('molstar/lib/mol-model/structure');
        if (destroyed) {
          try {
            ctx.dispose();
          } catch {
            /* ignore */
          }
          return null;
        }
        ctx.behaviors.interaction.click.subscribe((ev) => {
          if (!onResidueClick) return;
          const loci = ev.current.loci;
          if (!StructureElement.Loci.is(loci) || loci.elements.length === 0) return;
          const loc = StructureElement.Loci.getFirstLocation(loci);
          if (!loc) return;

          // Ignore non-polymers (ligands, ions, waters, carbohydrates) and DNA/RNA nucleic polymers
          let entityType = 'polymer';
          let entitySubtype = '';
          try {
            entityType = String(StructureProperties.entity.type(loc) ?? '');
          } catch {
            entityType = 'polymer';
          }
          if (entityType !== 'polymer') return;

          try {
            entitySubtype = String(StructureProperties.entity.subtype(loc) ?? '').toLowerCase();
          } catch {
            entitySubtype = '';
          }
          if (
            entitySubtype.includes('ribonucleotide') ||
            entitySubtype.includes('nucleic') ||
            (entitySubtype &&
              !entitySubtype.includes('polypeptide') &&
              !entitySubtype.includes('peptide') &&
              entitySubtype !== 'other')
          ) {
            return;
          }

          const compId = StructureProperties.atom.label_comp_id(loc) || 'UNK';
          if (NUCLEIC_COMP_IDS.has(compId.toUpperCase())) return;

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
          onResidueClick({
            chain,
            seqId,
            ...(insCode ? { insCode } : {}),
            residueToken: `${chain}${seqId}${insCode}`,
            compId,
            isNonPolymer: false,
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
      try {
        const ligComp = await ctx.builders.structure.tryCreateComponentStatic(struct, 'ligand');
        if (ligComp) {
          const ligRepr = await ctx.builders.structure.representation.addRepresentation(ligComp, {
            type: 'ball-and-stick',
            color: 'element-symbol',
          });
          if (ligRepr?.ref) refs.push(ligRepr.ref);
          else if (ligComp.ref) refs.push(ligComp.ref);
        }
      } catch {
        /* ignore if no ligand component */
      }
      try {
        const ionComp = await ctx.builders.structure.tryCreateComponentStatic(struct, 'ion');
        if (ionComp) {
          const ionRepr = await ctx.builders.structure.representation.addRepresentation(ionComp, {
            type: 'spacefill',
            color: 'element-symbol',
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

      const { Color } = await import('molstar/lib/mol-util/color');
      if (destroyed || mySeq !== loadSeq) return;
      const hasOverlay = Boolean(sDesigned && dPdb && sRefold && rPdb && !useTraj);

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
        const designedColorArg =
          styleOpts.color !== 'auto'
            ? { color: styleOpts.color }
            : hasOverlay
              ? { color: 'uniform', colorParams: { value: Color(0x94a3b8) } }
              : isMultiChain
                ? { color: 'chain-id' }
                : { color: 'uniform', colorParams: { value: Color(0x60a5fa) } };
        const designedAlpha = hasOverlay ? 0.45 : 1.0;
        await ctx.builders.structure.representation.addRepresentation(struct, {
          type: styleOpts.rep,
          ...designedColorArg,
          typeParams: { alpha: designedAlpha },
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

      // 2. Kabsch-aligned refolded structure colored by pLDDT (B-factor)
      if (sRefold && rPdb && !useTraj) {
        const rFmt = /^\s*(data_|loop_|_atom_site\.)/m.test(rPdb) ? 'mmcif' : 'pdb';
        const data = await ctx.builders.data.rawData({
          data: rPdb,
          label: 'LocalFold Refold (pLDDT)',
        });
        const traj = await ctx.builders.structure.parseTrajectory(data, rFmt);
        const model = await ctx.builders.structure.createModel(traj, { modelIndex: 0 });
        const struct = await ctx.builders.structure.createStructure(model, {
          name: 'model',
          params: {},
        });
        const refoldColorArg =
          styleOpts.color !== 'auto' ? { color: styleOpts.color } : { color: 'plddt-confidence' };
        await ctx.builders.structure.representation.addRepresentation(struct, {
          type: styleOpts.rep,
          ...refoldColorArg,
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
      applyPostprocessing(illustrative, spin && hasAnyStructure);
      if (savedCam) {
        ctx.canvas3d?.camera?.setState(savedCam, 0);
      } else {
        ctx.canvas3d?.requestCameraReset();
      }
      lastStructureKey = styleOpts.structureKey;
      lastRenderedKey = styleOpts.fullKey;
    } catch (err) {
      if (!destroyed && mySeq === loadSeq) {
        lastRenderedKey = '';
        initError = err instanceof Error ? err.message : String(err);
      }
    }
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

  function buildPostprocessingProps(illustrativeOn: boolean, spinning: boolean) {
    return {
      bloom: { name: 'off' as const, params: {} },
      shadow: { name: 'off' as const, params: {} },
      occlusion: spinning
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
      antialiasing: spinning
        ? { name: 'off' as const, params: {} }
        : {
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

  function applyPostprocessing(illustrativeOn: boolean, spinning: boolean) {
    if (!plugin?.canvas3d) return;
    plugin.canvas3d.setProps({
      postprocessing: buildPostprocessingProps(illustrativeOn, spinning),
    });
  }

  let hasAnyStructure = $derived(
    Boolean((models && models.length > 0) || designedPdb || refoldPdb),
  );

  function applySpin(on: boolean, speed: number) {
    if (!plugin?.canvas3d) return;
    plugin.canvas3d.setProps({
      postprocessing: buildPostprocessingProps(illustrative, on),
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
    modelRefs = new Map();
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

      const [{ Color }, { setSubtreeVisibility }] = await Promise.all([
        import('molstar/lib/mol-util/color'),
        import('molstar/lib/mol-plugin/behavior/static/state'),
      ]);
      if (destroyed || mySeq !== loadSeq) return;

      const nextRefs = new Map<string, string[]>();
      for (const m of specs) {
        const fmt = /^\s*(data_|loop_|_atom_site\.)/m.test(m.pdbText) ? 'mmcif' : 'pdb';
        const data = await ctx.builders.data.rawData({
          data: m.pdbText,
          label: m.label,
        });
        const traj = await ctx.builders.structure.parseTrajectory(data, fmt);
        const model = await ctx.builders.structure.createModel(traj, { modelIndex: 0 });
        if (destroyed || mySeq !== loadSeq) return;
        const struct = await ctx.builders.structure.createStructure(model, {
          name: 'model',
          params: {},
        });
        const modelColorArg =
          styleOpts.color !== 'auto'
            ? { color: styleOpts.color }
            : m.colorByPlddt
              ? { color: 'plddt-confidence' }
              : { color: 'uniform', colorParams: { value: Color(m.colorHex) } };
        const repr = await ctx.builders.structure.representation.addRepresentation(struct, {
          type: styleOpts.rep,
          ...modelColorArg,
          typeParams: { alpha: m.alpha },
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
      modelRefs = nextRefs;
      applyPostprocessing(illustrative, spin && hasAnyStructure);
      if (savedCam) {
        ctx.canvas3d?.camera?.setState(savedCam, 0);
      } else {
        ctx.canvas3d?.requestCameraReset();
      }
      lastStructureKey = styleOpts.structureKey;
      lastRenderedKey = styleOpts.fullKey;
    } catch (err) {
      if (!destroyed && mySeq === loadSeq) {
        lastRenderedKey = '';
        initError = err instanceof Error ? err.message : String(err);
      }
    }
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

  // Structural signature for multi-model mode (excludes `visible` so eye/solo toggles never reload or reset camera)
  let multiRenderKey = $derived.by((): string | null => {
    if (models === null) return null;
    return models
      .map(
        (m) =>
          `${m.id}|${m.colorHex}|${m.alpha}|${m.colorByPlddt ? 1 : 0}|${m.pdbText.length}|${m.pdbText.slice(0, 64)}|${m.pdbText.slice(-64)}`,
      )
      .join('||');
  });

  // Re-render full scene ONLY when (structureKey + styleKey) actually changes
  $effect(() => {
    const styleOpts = {
      rep: repType,
      color: colorScheme,
      sidechains: showSidechains,
      surfaceOverlay: showSurfaceOverlay,
      ligands: showLigands,
    };
    const styleKey = `${styleOpts.rep}|${styleOpts.color}|${styleOpts.sidechains ? 1 : 0}|${styleOpts.surfaceOverlay ? 1 : 0}|${styleOpts.ligands ? 1 : 0}`;
    if (multiRenderKey !== null) {
      const mKey = `multi:${multiRenderKey}`;
      const fullKey = `${mKey}##${styleKey}`;
      if (fullKey === lastRenderedKey && !initError) return;
      const sameStructure = lastStructureKey === mKey;
      const snapshot = untrack(() => models ?? []);
      void renderMultiModels(snapshot, {
        ...styleOpts,
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
    const on = illustrative;
    const spinning = spin && hasAnyStructure;
    if (plugin) applyPostprocessing(on, spinning);
  });

  $effect(() => {
    const on = spin && hasAnyStructure;
    const speed = effectiveSpinSpeed;
    if (plugin) applySpin(on, speed);
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
    try {
      plugin?.dispose();
    } catch {
      /* ignore */
    }
    plugin = null;
  });
</script>

<div class="mol-card" style="height: {typeof height === 'number' ? `${height}px` : height}">
  {#if label}
    <div class="badge" class:has-style-menu={expanded && hasAnyStructure} title={label}>{label}</div>
  {/if}
  {#if expanded && hasAnyStructure}
    <div class="style-menu-wrap">
      <button
        type="button"
        class="style-toggle-btn"
        class:active={styleMenuOpen}
        onclick={() => (styleMenuOpen = !styleMenuOpen)}
        title="Toggle structure representation & view controls"
      >
        ⚙ Representation {styleMenuOpen ? '▾' : '▸'}
      </button>
      {#if styleMenuOpen}
        <div class="style-panel" role="region" aria-label="Structure representation controls">
          <label class="sp-field">
            <span>Polymer</span>
            <select bind:value={repType}>
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
            <select bind:value={colorScheme}>
              <option value="auto">Auto (Default)</option>
              <option value="chain-id">By Chain</option>
              <option value="plddt-confidence">pLDDT Confidence</option>
              <option value="secondary-structure">Secondary Structure</option>
              <option value="hydrophobicity">Hydrophobicity</option>
              <option value="sequence-id">Rainbow (N → C)</option>
              <option value="element-symbol">Element</option>
            </select>
          </label>
          <div class="sp-checks">
            <label><input type="checkbox" bind:checked={showSidechains} /> All-atom sticks</label>
            <label><input type="checkbox" bind:checked={showSurfaceOverlay} /> Translucent surface</label>
            <label><input type="checkbox" bind:checked={showLigands} /> Ligands &amp; ions</label>
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
          {#if spin}
            <label class="sp-field">
              <span>Spin speed ({effectiveSpinSpeed.toFixed(3)} rot/s)</span>
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
              disabled={!plugin || Boolean(initError)}
              onclick={handleSavePng}
              title={!plugin || initError ? '3D viewer unavailable' : 'Download PNG screenshot'}
            >
              📷 PNG
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
      {#each models as m (m.id)}
        <div class="cmp-chip" class:is-hidden={!m.visible}>
          <button
            type="button"
            class="cmp-chip-main"
            title={m.visible ? `Hide ${m.label}` : `Show ${m.label}`}
            onclick={() => onToggleModelVisibility?.(m.id)}
          >
            <span
              class="cmp-swatch"
              class:ref={m.isReference}
              style="background: {m.colorCss}; opacity: {m.visible ? 1 : 0.35}"
            ></span>
            {#if m.badge}
              <span class="cmp-badge">{m.badge}</span>
            {/if}
            <span class="cmp-lbl">{m.label}</span>
            {#if m.metricSummary}
              <span class="cmp-stat">{m.metricSummary}</span>
            {/if}
          </button>
          {#if onSoloModel && !m.isReference && models.length > 1}
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
  {:else if designedPdb && refoldPdb && showDesigned && showRefold && !showTrajectory}
    <div class="legend">
      <span class="swatch designed"></span> Designed
      <span class="swatch plddt"></span> Refold (pLDDT)
    </div>
  {/if}
  <div bind:this={containerEl} class="canvas-wrap">
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
  .badge {
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 10;
    max-width: calc(100% - 20px);
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
  .badge.has-style-menu {
    max-width: calc(100% - 175px);
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
    width: 230px;
    max-height: min(420px, calc(100vh - 120px));
    min-height: 0;
    overflow-y: auto;
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
    display: grid;
    grid-template-columns: 1.2fr 0.85fr 0.95fr;
    gap: 5px;
    padding-top: 2px;
    border-top: 1px solid var(--border-subtle);
  }
  .sp-actions button {
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 5px;
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
  @media (max-width: 640px) {
    .badge.has-style-menu {
      max-width: calc(100% - 155px);
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
