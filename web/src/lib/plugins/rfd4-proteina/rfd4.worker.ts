/// <reference lib="webworker" />
// Layer 3 Plugin Worker — rfd4-proteina (sokrypton/rfd4-proteina-webgpu)
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress, and disposal.
// Supports: Unconditional, Partial Diffusion (lazy +138MB encoder), RFD4 Inverse Folding,
// Upstream Benchmark Presets (PD-L1 binder, 1PRW motif, SAM H-bond pocket, DNA, CATH),
// and custom ConditionSpec / YAML / JSON conditioning.

import { AA3_TO_AA1, parsePdbBackbone, type SourceResidueId } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { fetchCachedBytes } from '$lib/engine/opfs-cache';
import {
  assertRfd4ConditionSelectorsMatch,
  extractRfd4DesignMetadata,
  extractRfd4TemplateContext,
  pdbTextToSplitCifText,
  resolveRfd4CompiledInput,
  type Rfd4TemplateContext,
} from '$lib/engine/conditioning';
import { makeSequenceKey } from '$lib/engine/collections';
import { deriveStableItemSeed, planGeneratorDesigns } from '$lib/engine/work-plan';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import { releaseRfd4GeneratorActivations } from './lifecycle';
import type {
  ConditionSpecArtifact,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const RFD4_ORIGIN = 'https://web.rfd4-proteina.org';
const RFD4_SHARDED_URL = `${RFD4_ORIGIN}/weights/int8`;

function computeChainLengths(chainIds: readonly string[]): number[] {
  const lengths: number[] = [];
  for (let i = 0; i < chainIds.length; i++) {
    if (i === 0 || chainIds[i] !== chainIds[i - 1]) {
      lengths.push(1);
    } else {
      lengths[lengths.length - 1]++;
    }
  }
  return lengths;
}

function collectOwnedTransferables(views: Array<ArrayBufferView | undefined>): Transferable[] {
  const seen = new Set<ArrayBuffer>();
  const out: Transferable[] = [];
  for (const v of views) {
    if (!v) continue;
    const buf = v.buffer;
    if (!(buf instanceof ArrayBuffer)) continue;
    if (v.byteOffset !== 0 || v.byteLength !== buf.byteLength) continue;
    if (seen.has(buf)) continue;
    seen.add(buf);
    out.push(buf);
  }
  return out;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4Mod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4CifMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4CcdMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4PrepareMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4SpecMod: any = null;
let gpuDevice: GPUDevice | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let shardedBundle: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let generator: any = null;

async function ensureLoaded(ctx: WorkerRunContext, needEncoder: boolean) {
  if (!rfd4Mod) {
    ctx.progress('init', 0.02, 'Loading RFD4-Proteina WebGPU modules...');
    const base = `${self.location.origin}/vendor/rfd4-proteina`;
    const [idxMod, cifMod, ccdMod, prepareMod, specMod] = await Promise.all([
      import(/* @vite-ignore */ `${base}/webgpu/index.js`),
      import(/* @vite-ignore */ `${base}/shared/rfd4p/pipeline/cif.js`),
      import(/* @vite-ignore */ `${base}/shared/rfd4p/pipeline/ccd.js`),
      import(/* @vite-ignore */ `${base}/shared/rfd4p/pipeline/prepare.js`),
      import(/* @vite-ignore */ `${base}/shared/rfd4p/pipeline/spec.js`),
    ]);
    rfd4Mod = idxMod;
    rfd4CifMod = cifMod;
    rfd4CcdMod = ccdMod;
    rfd4PrepareMod = prepareMod;
    rfd4SpecMod = specMod;
  }

  if (!gpuDevice) {
    // Retry + fallback to default requestAdapter() (without powerPreference) because on Linux
    // Chrome/Vulkan (especially right after a prior GPU worker disposes), the first
    // `requestAdapter({ powerPreference: 'high-performance' })` inside a new worker can return null.
    let dev: GPUDevice | null = null;
    let lastErr: unknown = null;

    for (let attempt = 0; attempt < 3 && !dev; attempt++) {
      if (attempt > 0) {
        await new Promise((r) => setTimeout(r, attempt * 250));
      }
      try {
        const res = await rfd4Mod.requestWebGpuDevice();
        if (res?.device) {
          dev = res.device;
          break;
        }
      } catch (err) {
        lastErr = err;
        // Retry withholding experimental subgroup features if the driver rejected them
        try {
          const res = await rfd4Mod.requestWebGpuDevice({
            exclude: ['chromium-experimental-subgroup-matrix', 'subgroup-size-control'],
          });
          if (res?.device) {
            dev = res.device;
            break;
          }
        } catch (err2) {
          lastErr = err2;
        }
      }

      // Direct adapter fallback (mirrors Genie 3: tries default requestAdapter() with no powerPreference)
      if (!dev && typeof navigator !== 'undefined' && navigator.gpu) {
        try {
          const adapter =
            (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
            (await navigator.gpu.requestAdapter());
          if (adapter) {
            const base = `${self.location.origin}/vendor/rfd4-proteina`;
            const devMod = await import(
              /* @vite-ignore */ `${base}/vendor/localfold-1edc50a7/webgpu/runtime/device.js`
            );
            try {
              dev = await devMod.requestAlphaFoldDevice(adapter);
            } catch {
              dev = await devMod.requestAlphaFoldDevice(adapter, {
                withholdFeatures: [
                  'chromium-experimental-subgroup-matrix',
                  'subgroup-size-control',
                ],
              });
            }
            if (dev && !(dev as unknown as { adapterInfo?: unknown }).adapterInfo && adapter.info) {
              (dev as unknown as { adapterInfo?: unknown }).adapterInfo = adapter.info;
            }
          }
        } catch (err) {
          lastErr = err;
        }
      }
    }

    if (!dev) {
      const detail = lastErr instanceof Error ? ` (${lastErr.message})` : '';
      throw new Error(`No compatible WebGPU device found for RFD4-Proteina${detail}`);
    }
    const activeDev = dev;
    gpuDevice = activeDev;
    void activeDev.lost.then(() => {
      if (gpuDevice === activeDev) {
        gpuDevice = null;
        generator = null;
      }
    });
  }

  if (!shardedBundle) {
    ctx.progress('weights', 0.05, 'Fetching RFD4-Proteina weights (~330 MB, cached after first run)...');
    shardedBundle = await rfd4Mod.fetchShardedWeights(RFD4_SHARDED_URL, {
      includeEncoder: needEncoder,
      onProgress: (prog: { loaded?: number; total?: number; loadedBytes?: number; totalBytes?: number; message?: string }) => {
        const loaded = prog.loadedBytes ?? prog.loaded ?? 0;
        const total = prog.totalBytes ?? prog.total ?? 0;
        const frac = total > 0 ? loaded / total : 0.2;
        const mb = (loaded / 1024 / 1024).toFixed(1);
        const totMb = total > 0 ? (total / 1024 / 1024).toFixed(0) : needEncoder ? '469' : '330';
        ctx.progress(
          'weights',
          Math.min(0.95, frac),
          prog.message || `RFD4-Proteina weights: ${mb} / ${totMb} MB`,
        );
      },
    });
  } else if (needEncoder && !shardedBundle.encoderLoaded && typeof shardedBundle.loadEncoder === 'function') {
    ctx.progress(
      'weights',
      0.1,
      'Loading RFD4 Autoencoder Encoder (+138 MB for partial diffusion / inverse folding)...',
    );
    await shardedBundle.loadEncoder((prog: { loadedBytes?: number; totalBytes?: number }) => {
      const loaded = prog.loadedBytes ?? 0;
      const total = prog.totalBytes ?? 138 * 1024 * 1024;
      ctx.progress(
        'weights',
        Math.min(0.95, loaded / Math.max(1, total)),
        `RFD4 Encoder: ${(loaded / 1024 / 1024).toFixed(1)} / ${(total / 1024 / 1024).toFixed(0)} MB`,
      );
    });
  }

  if (!generator) {
    ctx.progress('init', 0.96, 'Uploading RFD4-Proteina weights to GPU...');
    // Match upstream web.rfd4-proteina.org: always use packed FP16 weights ({ half: true })
    // for half the VRAM, 2-3x faster kernels, and exact parity with upstream WebGPU CI gates.
    generator = await rfd4Mod.Rfd4pGeneratorGpu.create(
      gpuDevice,
      shardedBundle.tensors ?? shardedBundle,
      shardedBundle.config,
      { half: true },
    );
  }
}

const SAFE_EXAMPLE_CIF_RE = /^[a-zA-Z0-9/_-]+\.cif(?:\.zst)?$/;

async function fetchExampleCifBytes(ctx: WorkerRunContext, relUrl: string): Promise<Uint8Array> {
  const cleanRel = relUrl.replace(/^\.\//, '').replace(/^\/+/, '');
  if (
    /^https?:\/\//i.test(relUrl) ||
    cleanRel.includes('..') ||
    !SAFE_EXAMPLE_CIF_RE.test(cleanRel)
  ) {
    throw new Error(`Rejected unsafe RFD4 example CIF path: "${relUrl}"`);
  }
  const fullUrl = `${RFD4_ORIGIN}/${cleanRel}`;
  ctx.progress('featurize', 0.04, `Fetching context CIF (${cleanRel.split('/').pop()})...`);
  return await fetchCachedBytes(`rfd4-ex-${cleanRel}`, fullUrl);
}

/**
 * Prepare raw PDB or mmCIF template input for RFD4 `buildBatch`:
 * 1. Convert PDB inputs via `pdbTextToSplitCifText` so HETATM ligands/ions get distinct
 *    `label_asym_id` / `label_entity_id` chains and proper `NON_POLYMER` classification.
 * 2. Fetch missing CCD chemical component definitions (with graceful offline fallback).
 * 3. Parse into `AtomArray`, convert `MSE -> MET`, rebuild bonds if needed, and assign
 *    CCD formal charges to monoatomic ions (`ZN`, `FE2`, `MG`, `CA`, etc.).
 * 4. Build an `auth_* -> label_*` remapping context (`Rfd4TemplateContext`).
 */
async function prepareTemplateForRfd4(
  ctx: WorkerRunContext,
  rawInput: string | Uint8Array | undefined,
): Promise<{
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  atomArray: any;
  templateContext: Rfd4TemplateContext;
  cifText: string;
} | null> {
  if (!rawInput) return null;
  const decoded =
    typeof rawInput === 'string' ? rawInput : String(rfd4Mod.decodeTextOrZstd(rawInput));
  const isCif = /^\s*(data_|loop_|_atom_site\.)/m.test(decoded);
  const cifText = isCif ? decoded : pdbTextToSplitCifText(decoded);

  const templateContext = extractRfd4TemplateContext(cifText);
  const cifBlock = rfd4CifMod.parseCifBlock(cifText);

  let ccdBlocks: Map<string, unknown> | undefined;
  try {
    const compCol =
      cifBlock?.atom_site?.label_comp_id ?? cifBlock?.atom_site?.auth_comp_id ?? [];
    const resNames = Array.isArray(compCol) ? compCol : [compCol];
    const missing = rfd4CcdMod.componentsNeedingCcd(resNames, cifBlock) as string[];
    if (missing.length > 0) {
      ctx.progress('featurize', 0.05, `Fetching CCD ligand definitions (${missing.join(', ')})...`);
      const fetched = await rfd4CcdMod.fetchCcdComponents(missing);
      if (fetched?.blocks?.size > 0) {
        ccdBlocks = fetched.blocks;
      }
    }
  } catch (ccdErr) {
    console.warn('[rfd4] CCD fetch skipped (offline or unavailable):', ccdErr);
  }

  const atomArray = rfd4Mod.parseCifToAtomArray(cifText, {
    fixArginines: true,
    ...(ccdBlocks ? { ccdBlocks } : {}),
  });

  const hadMse = Array.isArray(atomArray.res_name) && atomArray.res_name.includes('MSE');
  rfd4PrepareMod.convertMseToMet(atomArray);
  if (hadMse && typeof rfd4CifMod.buildBondsForAtomArray === 'function') {
    rfd4CifMod.buildBondsForAtomArray(atomArray, cifBlock);
  }
  rfd4PrepareMod.assignIonCharges(atomArray);

  return { atomArray, templateContext, cifText };
}

/**
 * Format RFD4's output runLabel (e.g. `RFD4 · 20-20,A5-12,20-20 · motif:A5-12`) using
 * author PDB residue numbering (`sourceResidues` / `templateContext.byLabelRes` / input `motifSpans`)
 * instead of internal 1-based `label_seq_id` indices when the input PDB uses negative or
 * non-1-based residue numbers.
 */
export function formatRfd4AuthorRunLabel(
  runLabel: string,
  opts?: {
    motifSpans?: readonly string[];
    templateContext?: Rfd4TemplateContext | null;
    sourceResidues?: readonly SourceResidueId[] | null;
  },
): string {
  const parts = runLabel.split(' · ');
  if (parts.length <= 1) return runLabel;

  const byLabel = new Map<string, { authChain: string; authSeqId: number; insCode: string }>();
  if (opts?.templateContext) {
    for (const [k, entry] of opts.templateContext.byLabelRes.entries()) {
      byLabel.set(k, {
        authChain: entry.authChain,
        authSeqId: entry.authSeqId,
        insCode: entry.insCode ?? '',
      });
    }
  } else if (opts?.sourceResidues && opts.sourceResidues.length > 0) {
    const perChainCount = new Map<string, number>();
    for (const sr of opts.sourceResidues) {
      const ch = sr.labelChainId ?? sr.authChainId;
      const seqId = sr.labelSeqId ?? (perChainCount.get(ch) ?? 0) + 1;
      perChainCount.set(ch, seqId);
      byLabel.set(`${ch}:${seqId}`, {
        authChain: sr.authChainId,
        authSeqId: sr.resNum,
        insCode: sr.insCode ?? '',
      });
    }
  }

  const remapLabelTokenToAuthor = (rawTok: string): string => {
    const tok = rawTok.trim();
    const m = /^([A-Za-z][A-Za-z0-9]*)(\d+)(?:-(\d+))?$/.exec(tok);
    if (!m) return rawTok;
    const ch = m[1];
    const lo = parseInt(m[2], 10);
    const hi = m[3] !== undefined ? parseInt(m[3], 10) : lo;
    const startEntry = byLabel.get(`${ch}:${lo}`);
    const endEntry = byLabel.get(`${ch}:${hi}`);
    if (!startEntry || !endEntry) return rawTok;
    const startStr = `${startEntry.authChain}${startEntry.authSeqId}${startEntry.insCode}`;
    if (lo === hi) return startStr;
    return `${startStr}-${endEntry.authSeqId}${endEntry.insCode}`;
  };

  const cleanAuthorMotifs = (opts?.motifSpans ?? []).map((s) => s.trim()).filter(Boolean);
  const outParts = [parts[0]];
  for (let i = 1; i < parts.length; i++) {
    const seg = parts[i];
    const motifMatch = /^(unidx-motif|motif):(.*)$/.exec(seg);
    if (motifMatch) {
      const prefix = motifMatch[1];
      if (byLabel.size > 0) {
        const remapped = motifMatch[2]
          .split(',')
          .map((t) => remapLabelTokenToAuthor(t))
          .join(', ');
        outParts.push(`${prefix}:${remapped}`);
      } else if (cleanAuthorMotifs.length > 0) {
        outParts.push(`${prefix}:${cleanAuthorMotifs.join(', ')}`);
      } else {
        outParts.push(seg);
      }
      continue;
    }
    const hotMatch = /^hot:(.*)$/.exec(seg);
    if (hotMatch && byLabel.size > 0) {
      const remapped = hotMatch[1]
        .split(',')
        .map((t) => remapLabelTokenToAuthor(t))
        .join(', ');
      outParts.push(`hot:${remapped}`);
      continue;
    }
    if (byLabel.size > 0 && !seg.includes(':')) {
      const remapped = seg
        .split(',')
        .map((t) => remapLabelTokenToAuthor(t))
        .join(',');
      outParts.push(remapped);
      continue;
    }
    outParts.push(seg);
  }
  return outParts.join(' · ');
}

createWorkerHarness({
  async run(ctx) {
    const p = ctx.params ?? {};
    const template = (ctx.inputs?.structure ?? ctx.inputs?.template) as StructureArtifact | undefined;
    const condArtifact = ctx.inputs?.condition as ConditionSpecArtifact | undefined;
    const modeParam = String(p.mode ?? 'unconditional');
    const requestedEarlyStop = Boolean(p.earlyStop);
    const effectiveEarlyStop = modeParam === 'inverse-folding' ? false : requestedEarlyStop;
    const rawNumDesigns = Number(p.numDesigns ?? 1);
    const numDesigns = Number.isFinite(rawNumDesigns)
      ? Math.max(1, Math.min(16, Math.round(rawNumDesigns)))
      : 1;
    // Keep explicit p.<key> reads intact for param-schema bidirectional coverage guards
    const rawParams: Record<string, unknown> = {
      mode: modeParam,
      length: p.length,
      numDesigns,
      nsteps: p.nsteps,
      recycleSchedule: p.recycleSchedule,
      earlyStop: effectiveEarlyStop,
      seed: p.seed,
      presetId: p.presetId,
      partialFraction: p.partialFraction,
      contig: p.contig,
      lockChain: p.lockChain,
      hotspots: p.hotspots,
      motifSpan: p.motifSpan,
      motifAtoms: p.motifAtoms,
      unindexedMotif: p.unindexedMotif,
      hbondMode: p.hbondMode,
      hbondSelection: p.hbondSelection,
      hbondDonorSelection: p.hbondDonorSelection,
      hbondAcceptorSelection: p.hbondAcceptorSelection,
      cathCode: p.cathCode,
      customSpecYaml: p.customSpecYaml,
    };
    const inlineContig = String(rawParams.contig ?? '').trim();
    const inlineLock = String(rawParams.lockChain ?? '').trim();
    const inlineHotspots = String(rawParams.hotspots ?? '').trim();
    const inlineMotif = String(rawParams.motifSpan ?? '').trim();
    const inlineMotifAtoms = String(rawParams.motifAtoms ?? '').trim();
    const inlineUnidx = Boolean(rawParams.unindexedMotif);
    const inlineHbondMode = String(rawParams.hbondMode ?? 'none');
    const inlineHbondSel = String(rawParams.hbondSelection ?? '').trim();
    const inlineHbondDonorSel = String(rawParams.hbondDonorSelection ?? '').trim();
    const inlineHbondAcceptorSel = String(rawParams.hbondAcceptorSelection ?? '').trim();
    const inlineCath = String(rawParams.cathCode ?? '').trim();

    // Fail fast BEFORE downloading 330–468 MB weights if a required template structure is missing
    const hasTemplateStructure = Boolean(
      template?.cifText || template?.pdbText || condArtifact?.inputCif || condArtifact?.cifUrl,
    );
    if (
      (modeParam === 'motif' ||
        modeParam === 'binder' ||
        modeParam === 'partial' ||
        modeParam === 'inverse-folding') &&
      !hasTemplateStructure
    ) {
      throw new Error(
        `RFD4 "${modeParam}" mode requires a wired structure input (e.g. from PDB Input).`,
      );
    }

    // Resolve mode, contig, and conditions via single-source-of-truth resolver
    const preResolved = resolveRfd4CompiledInput(rawParams, template, condArtifact);
    const {
      mode,
      length,
      nsteps,
      recycleSchedule,
      seed,
      customSpecYaml,
      partialFraction,
      activePresetId,
      needEncoder,
    } = preResolved;
    const earlyStop = mode === 'inverse-folding' ? false : requestedEarlyStop;

    // Validate partial trajectory step count (round-half-to-even matching vendor partial.js)
    // BEFORE downloading 330-468 MB weights!
    if (preResolved.effectivePartialFraction != null) {
      const prod = preResolved.effectivePartialFraction * nsteps;
      const fl = Math.floor(prod);
      const diff = prod - fl;
      const roundedSteps = Math.abs(diff - 0.5) < 1e-12 ? (fl % 2 === 0 ? fl : fl + 1) : Math.round(prod);
      if (roundedSteps < 1) {
        throw new Error(
          `RFD4 partial diffusion fraction (${preResolved.effectivePartialFraction}) with nsteps=${nsteps} rounds to 0 denoising steps. Increase partialFraction or nsteps.`,
        );
      }
    }

    await ensureLoaded(ctx, needEncoder);

    // Prepare static inputs / template / preset CIF ONCE before the resident-weight design loop
    const baseGenOpts: Record<string, unknown> = {
      nsteps,
      recycleSchedule,
      earlyStop,
      inputFrame: true,
    };

    let runLabel = preResolved.runLabel;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let buildBatchForSeed: (itemSeed: number) => { batch: Record<string, any>; atoms?: any; n?: number };

    if (preResolved.kind === 'preset') {
      const presetId = activePresetId!;
      const presetsList = Array.isArray(rfd4Mod.RFD4P_PRESETS) ? rfd4Mod.RFD4P_PRESETS : [];
      // Validate directly against RFD4P_PRESETS instead of getRfd4pPresetById (which falls back to RFD4P_PRESETS[0])
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const preset = presetsList.find((item: any) => item.id === presetId);
      if (!preset) {
        throw new Error(`Unknown RFD4 preset "${presetId}"`);
      }
      runLabel = `RFD4 · ${preset.id}`;

      const presetNeedsEncoder =
        preset.partialTrajectoryFractionSteps != null || Boolean(preset.inverseFolding);
      if (presetNeedsEncoder) {
        await ensureLoaded(ctx, true);
      }

      if (preset.partialTrajectoryFractionSteps != null) {
        baseGenOpts.partialTrajectoryFractionSteps = preset.partialTrajectoryFractionSteps;
      }
      if (preset.inverseFolding) {
        baseGenOpts.inverseFolding = true;
        baseGenOpts.earlyStop = false;
      }
      if (preResolved.effectivePartialFraction != null) {
        await ensureLoaded(ctx, true);
        baseGenOpts.partialTrajectoryFractionSteps = preResolved.effectivePartialFraction;
      }
      if (preResolved.inverseFolding) {
        await ensureLoaded(ctx, true);
        baseGenOpts.inverseFolding = true;
        baseGenOpts.earlyStop = false;
      }

      const presetCifBytes = preset.cifUrl ? await fetchExampleCifBytes(ctx, preset.cifUrl) : null;
      buildBatchForSeed = (itemSeed: number) => {
        if (presetCifBytes) {
          if (preset.specMode) {
            return rfd4Mod.buildBatch(
              { contig: preset.contig, conditions: preset.conditions ?? {} },
              { inputCif: presetCifBytes, seed: itemSeed },
            );
          }
          // Annotated upstream .cif.zst already carries condition columns
          return rfd4Mod.buildBatch(presetCifBytes, { seed: itemSeed });
        }
        return rfd4Mod.buildBatch(
          { contig: preset.contig, conditions: preset.conditions ?? {} },
          { seed: itemSeed },
        );
      };
    } else {
      let inputCif: string | Uint8Array | undefined =
        mode === 'partial' || mode === 'inverse-folding'
          ? (template?.cifText || template?.pdbText)
          : (condArtifact?.inputCif || template?.cifText || template?.pdbText);
      if (!inputCif && condArtifact?.cifUrl && !activePresetId) {
        inputCif = await fetchExampleCifBytes(ctx, condArtifact.cifUrl);
      }
      const prepared =
        preResolved.usesTemplate || Boolean(inputCif && (customSpecYaml || mode !== 'unconditional'))
          ? await prepareTemplateForRfd4(ctx, inputCif)
          : null;
      const resolved = prepared
        ? resolveRfd4CompiledInput(rawParams, template, condArtifact, prepared.templateContext)
        : preResolved;
      runLabel = formatRfd4AuthorRunLabel(resolved.runLabel, {
        motifSpans: resolved.intent?.motifSpans,
        templateContext: prepared?.templateContext ?? resolved.templateContext,
        sourceResidues: template?.sourceResidues,
      });

      if (resolved.effectivePartialFraction != null) {
        baseGenOpts.partialTrajectoryFractionSteps = resolved.effectivePartialFraction;
      }
      if (resolved.inverseFolding) {
        baseGenOpts.inverseFolding = true;
        baseGenOpts.earlyStop = false;
      }

      if (resolved.kind === 'template-custom-yaml') {
        const parsedYaml = rfd4Mod.parseConditionSpecYaml(resolved.customSpecYaml);
        const yamlConds = (parsedYaml?.conditions ?? {}) as Record<string, unknown>;
        if (prepared && Object.keys(yamlConds).length > 0) {
          assertRfd4ConditionSelectorsMatch(
            prepared.atomArray,
            parsedYaml?.contig ?? null,
            yamlConds,
            rfd4SpecMod,
          );
        }
        buildBatchForSeed = (itemSeed: number) =>
          rfd4Mod.buildBatch(
            parsedYaml?.contig || parsedYaml?.input_as_annotated_cif
              ? parsedYaml
              : { input: 'template', conditions: yamlConds },
            { inputAtomArray: prepared!.atomArray, seed: itemSeed },
          );
      } else if (resolved.kind === 'template-spec') {
        if (prepared && Object.keys(resolved.conditions).length > 0) {
          assertRfd4ConditionSelectorsMatch(
            prepared.atomArray,
            null,
            resolved.conditions,
            rfd4SpecMod,
          );
        }
        buildBatchForSeed = (itemSeed: number) =>
          rfd4Mod.buildBatch(
            { input: 'template', conditions: resolved.conditions },
            { inputAtomArray: prepared!.atomArray, seed: itemSeed },
          );
      } else if (resolved.kind === 'custom-yaml') {
        buildBatchForSeed = (itemSeed: number) =>
          rfd4Mod.buildBatch(
            resolved.customSpecYaml,
            prepared
              ? { inputAtomArray: prepared.atomArray, seed: itemSeed }
              : { inputCif: template?.cifText || template?.pdbText, seed: itemSeed },
          );
      } else {
        if (prepared && Object.keys(resolved.conditions).length > 0) {
          assertRfd4ConditionSelectorsMatch(
            prepared.atomArray,
            resolved.contig,
            resolved.conditions,
            rfd4SpecMod,
          );
        }
        buildBatchForSeed = (itemSeed: number) =>
          rfd4Mod.buildBatch(
            { contig: resolved.contig, conditions: resolved.conditions },
            prepared
              ? { inputAtomArray: prepared.atomArray, seed: itemSeed }
              : inputCif && condArtifact
                ? { inputCif, seed: itemSeed }
                : { seed: itemSeed },
          );
      }
    }

    const releaseGeneratorActivations = () => {
      releaseRfd4GeneratorActivations(generator);
    };

    const tokens: string[] = rfd4Mod.UNIFIED_ATOM37_TOKENS ?? [];
    const rtToSeq = (rtArr: ArrayLike<number>): string => {
      let s = '';
      for (let i = 0; i < rtArr.length; i++) {
        const rt = rtArr[i];
        if (rt <= 0 || rt >= tokens.length) continue;
        const aa1 = AA3_TO_AA1[tokens[rt]];
        if (aa1) s += aa1;
      }
      return s;
    };

    const runOneDesign = async (
      designIndex: number,
      totalDesigns: number,
      itemSeed: number,
      designKey?: string,
    ): Promise<{
      struct: StructureArtifact;
      candidate: SequenceCandidate;
      designMeta: ReturnType<typeof extractRfd4DesignMetadata>;
    }> => {
      const built = buildBatchForSeed(itemSeed);
      const genOpts: Record<string, unknown> = {
        ...baseGenOpts,
        seed: itemSeed,
        ...(built.atoms ? { inputAtoms: built.atoms } : {}),
        onStep: (info: { step: number; nsteps: number }) => {
          const stepFrac = (info.step + 1) / Math.max(1, info.nsteps);
          const progress =
            totalDesigns > 1 ? (designIndex + stepFrac) / totalDesigns : stepFrac;
          const detail =
            totalDesigns > 1
              ? `Design ${designIndex + 1}/${totalDesigns} · RFD4 step ${info.step + 1}/${info.nsteps}`
              : `RFD4 step ${info.step + 1}/${info.nsteps}`;
          ctx.progress('sampling', progress, detail);
        },
      };

      // Capture any silent WebGPU validation errors (which otherwise leave output buffers all-zero!)
      const scopeDevice = gpuDevice;
      const pushedScope = typeof scopeDevice?.pushErrorScope === 'function';
      if (pushedScope) {
        scopeDevice!.pushErrorScope('validation');
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let res: any;
      let gpuValErr: GPUError | null = null;
      try {
        res = await generator.generate(built.batch, genOpts);
      } finally {
        releaseGeneratorActivations();
        if (pushedScope && typeof scopeDevice?.popErrorScope === 'function') {
          gpuValErr = await scopeDevice.popErrorScope().catch(() => null);
        }
      }
      if (gpuValErr) {
        console.error('[rfd4] WebGPU validation error during generate():', gpuValErr.message);
        throw new Error(`RFD4-Proteina WebGPU validation error: ${gpuValErr.message}`);
      }

      // Inspect trunk sequence logits & decoded residueType to catch all-zero / poly-ALA collapse
      const logitsArr = res.logits as Float32Array | null | undefined;
      let logitsNonZero = 0;
      let logitsNan = 0;
      let logitsMin = Infinity;
      let logitsMax = -Infinity;
      if (logitsArr) {
        for (let i = 0; i < logitsArr.length; i++) {
          const v = logitsArr[i];
          if (Number.isNaN(v)) logitsNan++;
          else {
            if (v !== 0) logitsNonZero++;
            if (v < logitsMin) logitsMin = v;
            if (v > logitsMax) logitsMax = v;
          }
        }
      }

      let nativeSeq = rtToSeq(res.residueType ?? []);
      const uniqueNativeAAs = new Set(nativeSeq.replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '').split('')).size;
      let degradedFallback: string | undefined;

      // Fallback recovery: if trunk logits were zero/NaN or collapsed to a single residue (e.g. poly-ALA)
      // while local_latents are valid, decode sequence directly using the autoencoder decoder's own
      // logit_linear head (by passing unconditioned C_SEQ from the batch).
      if (nativeSeq.length >= 10 && (logitsNonZero === 0 || logitsNan > 0 || uniqueNativeAAs <= 1)) {
        degradedFallback = 'Degenerate trunk logits; attempted autoencoder decoder logit_linear recovery';
        console.warn('[rfd4] Degenerate trunk sequence readout detected; retrying via autoencoder decoder logit_linear head', {
          logitsNonZero,
          logitsNan,
          logitsMin,
          logitsMax,
          uniqueNativeAAs,
          seqSample: nativeSeq.slice(0, 24),
        });
        try {
          const rawCondSeq = built.batch.condition_sequence_1_residue?.data ?? built.batch.condition_sequence_1_residue;
          const maskedSeq = rawCondSeq ? Int32Array.from(rawCondSeq) : new Int32Array(res.n);
          const decRecovered = await generator.decoder.decode(built.batch, {
            n: res.n,
            caNm: res.x.bb_ca,
            z: res.x.local_latents,
            seq: maskedSeq,
          });
          const formatted = rfd4Mod.formatSampleLocalLatents(built.batch, {
            n: res.n,
            caNm: res.x.bb_ca,
            decoded: decRecovered,
          });
          const recSeq = rtToSeq(formatted.residueType ?? []);
          const recUnique = new Set(recSeq.replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '').split('')).size;
          if (recUnique > uniqueNativeAAs) {
            const structure = rfd4Mod.processSampleToAtoms({
              coors: formatted.coors,
              atomMask: formatted.atomMask,
              residueType: formatted.residueType,
              paddingMask: built.batch.padding_mask?.data ?? built.batch.padding_mask,
              insertionLogits: res.insertionLogits,
              inputAtoms: built.atoms ?? built.batch.__atoms ?? null,
              batch: built.batch,
              inputFrame: true,
            });
            res = {
              ...res,
              residueType: formatted.residueType,
              structure,
              cif: rfd4Mod.toCifString(structure),
              pdb: rfd4Mod.toPdbString(structure),
            };
            nativeSeq = recSeq;
            degradedFallback = 'Recovered sequence via autoencoder decoder logit_linear head (trunk logits degraded)';
            console.info('[rfd4] Recovered sequence via decoder head:', recSeq.slice(0, 30), `(${recUnique} unique AAs)`);
          }
        } catch (recErr) {
          console.warn('[rfd4] Decoder-head sequence recovery failed:', recErr);
        } finally {
          releaseGeneratorActivations();
        }
      }

      const pdbText: string = res.pdb;
      const cifText: string = res.cif;
      const parsed = parsePdbBackbone(pdbText);
      const finalSeq =
        parsed.sequence.length === parsed.nResidues && !parsed.sequence.includes('X')
          ? parsed.sequence
          : nativeSeq.length === parsed.nResidues
            ? nativeSeq
            : parsed.sequence;

      const uniqueFinal = new Set(finalSeq.replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '').split('')).size;
      console.info(
        `[rfd4] L=${parsed.nResidues} steps=${res.stepsRun}/${res.nsteps} half=true logits=[${logitsMin.toFixed(2)},${logitsMax.toFixed(2)}] nz=${logitsNonZero} uniqueAA=${uniqueFinal} seq=${finalSeq.slice(0, 25)}...`,
      );
      if (parsed.nResidues >= 15 && uniqueFinal <= 1) {
        const stepHint =
          nsteps < 58
            ? ` Steps=${nsteps} is below the recommended minimum (58) — set Steps to 60 and Recycle to none for reliable convergence.`
            : ' Try reloading the page or switching browser/GPU backend.';
        throw new Error(
          `RFD4-Proteina produced a degenerate single-residue sequence (${finalSeq.slice(0, 12)}…, logits nz=${logitsNonZero}, nan=${logitsNan}).${stepHint}`,
        );
      }

      const geometry = computeGeometryStats(
        parsed.caCoords,
        parsed.chainIds,
        parsed.resNums,
        parsed.insCodes,
      );
      const chainLengths = computeChainLengths(parsed.chainIds);
      const candChainLengths =
        chainLengths.length > 0 && chainLengths.reduce((a, b) => a + b, 0) === finalSeq.length
          ? chainLengths
          : undefined;

      const runHash = fnv1a(
        JSON.stringify({
          mode,
          length: mode === 'unconditional' && !customSpecYaml ? length : null,
          nsteps,
          recycleSchedule,
          earlyStop: Boolean(genOpts.earlyStop),
          seed: itemSeed,
          activePresetId,
          condId: condArtifact?.id ?? null,
          inlineContig,
          inlineLock,
          inlineHotspots,
          inlineMotif,
          inlineMotifAtoms,
          inlineUnidx,
          inlineHbondMode,
          inlineHbondSel,
          inlineHbondDonorSel,
          inlineHbondAcceptorSel,
          inlineCath,
          customSpecYaml,
          partialFraction: mode === 'partial' ? partialFraction : null,
          templateId: template?.id ?? null,
        }),
      );

      const designMeta = extractRfd4DesignMetadata(parsed, res.structure?.atoms, built.batch);
      const earlyStopped =
        typeof res.stepsRun === 'number' &&
        typeof res.nsteps === 'number' &&
        res.stepsRun < res.nsteps;

      const structId =
        totalDesigns > 1
          ? `rfd4-${mode}-d${designIndex + 1}-L${parsed.nResidues}-s${itemSeed}-${runHash}`
          : `rfd4-${mode}-L${parsed.nResidues}-s${itemSeed}-${runHash}`;
      const label =
        totalDesigns > 1
          ? `${runLabel} #${designIndex + 1}/${totalDesigns} (L=${parsed.nResidues}, steps=${res.stepsRun})`
          : `${runLabel} (L=${parsed.nResidues}, steps=${res.stepsRun})`;

      const struct: StructureArtifact = {
        kind: 'StructureArtifact',
        id: structId,
        label,
        sourceModelId: 'rfd4-proteina',
        length: parsed.nResidues,
        sequence: finalSeq,
        caCoords: parsed.caCoords,
        backbone4Coords: parsed.backbone4Coords,
        pdbText,
        cifText,
        chainIds: parsed.chainIds,
        chainLengths: parsed.diagnostics.chainLengths,
        insCodes: parsed.insCodes,
        sourceResidues: parsed.sourceResidues,
        chainIdMap: parsed.chainIdMap,
        atomPresence: parsed.atomPresence,
        observedBackboneMask: parsed.observedBackboneMask,
        diagnostics: parsed.diagnostics,
        bfactorSemantics:
          mode === 'partial' || mode === 'inverse-folding'
            ? (template?.bfactorSemantics ?? 'none')
            : 'none',
        atomGranularity: 'all-atom',
        provenance: {
          quantization: 'int8',
          earlyStopped,
          ...(degradedFallback ? { degradedFallback } : {}),
        },
        ...(designKey ? { designKey, designIndex, seed: itemSeed } : {}),
        ...designMeta,
        geometry,
      };

      const candBaseId = mode === 'inverse-folding' ? 'rfd4-invfold' : 'rfd4-native';
      const candidate: SequenceCandidate =
        totalDesigns > 1
          ? {
              id: `${candBaseId}-d${designIndex + 1}`,
              ...(designKey ? { designKey, designIndex } : {}),
              sourceStructureId: struct.id,
              sampleIndex: designIndex,
              sequence: finalSeq,
              score: null,
              globalScore: null,
              sourceModelId: 'rfd4-proteina',
              seed: itemSeed,
              ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
            }
          : {
              id: `${candBaseId}-1`,
              sequence: finalSeq,
              score: null,
              globalScore: null,
              sourceModelId: 'rfd4-proteina',
              seed: itemSeed,
              ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
            };

      return { struct, candidate, designMeta };
    };

    if (numDesigns === 1) {
      // Legacy singleton path: exact seed, artifact ID, and singleton StructureArtifact semantics
      const { struct: structArtifact, candidate, designMeta } = await runOneDesign(0, 1, seed);
      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `rfd4-seq-${structArtifact.id}`,
        parentStructureId: structArtifact.id,
        sourceStructureId: structArtifact.id,
        sourceModelId: `rfd4-proteina (${mode})`,
        ...(designMeta.lockedChains ? { lockedChains: designMeta.lockedChains } : {}),
        ...(designMeta.chainRoles ? { chainRoles: designMeta.chainRoles } : {}),
        candidates: [candidate],
      };

      const transferables = collectOwnedTransferables([
        structArtifact.caCoords,
        structArtifact.backbone4Coords,
        structArtifact.observedBackboneMask,
      ]);
      return {
        outputs: { structure: structArtifact, sequences: seqSet },
        transferables,
      };
    }

    // Multi-design resident-weight sequential loop (weights loaded ONCE above)
    const plan = planGeneratorDesigns({
      modelId: 'rfd4-proteina',
      numDesigns,
      baseSeed: seed,
      scientificParams: {
        mode,
        length: mode === 'unconditional' && !customSpecYaml ? length : null,
        nsteps,
        recycleSchedule,
        earlyStop: Boolean(baseGenOpts.earlyStop),
        activePresetId,
        inlineContig,
        inlineLock,
        inlineHotspots,
        inlineMotif,
        inlineMotifAtoms,
        inlineUnidx,
        inlineHbondMode,
        inlineHbondSel,
        inlineHbondDonorSel,
        inlineHbondAcceptorSel,
        inlineCath,
        customSpecYaml,
        partialFraction: mode === 'partial' ? partialFraction : null,
      },
      inputDigest: fnv1a(
        JSON.stringify({
          condId: condArtifact?.id ?? null,
          templateId: template?.id ?? null,
        }),
      ),
    });

    const designs: StructureArtifact[] = [];
    const candidates: SequenceCandidate[] = [];
    for (let d = 0; d < plan.items.length; d++) {
      const planned = plan.items[d];
      const itemSeed = deriveStableItemSeed(seed, 'generation', planned.key);
      const { struct, candidate } = await runOneDesign(d, numDesigns, itemSeed, planned.key);
      const seqKey = makeSequenceKey(plan.familyKey, planned.key, 0);
      candidate.key = seqKey;
      candidate.sequenceKey = seqKey;
      designs.push(struct);
      candidates.push(candidate);
    }

    const representative = designs[0];
    const repSelection = {
      selectedKey: representative.designKey ?? representative.id,
      criterion: 'lowest-creation-ordinal',
      totalMembers: designs.length,
      selectedIndex: 0,
      scoreValue: null,
    };
    const structArtifact: StructureArtifact = {
      ...representative,
      id: `rfd4-batch-${designs.length}-${plan.planDigest.slice(0, 16)}`,
      label: `${runLabel} (${numDesigns} designs · L=${representative.length})`,
      representativeSelection: repSelection,
      collection: {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id: `rfd4-col-${plan.planDigest.slice(0, 16)}`,
        familyKey: plan.familyKey,
        collectionKind: 'independent-candidates',
        requestedCount: numDesigns,
        complete: true,
        items: designs,
        keys: designs.map((item) => item.designKey ?? item.id),
        representativeSelection: repSelection,
      },
    };

    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: `rfd4-seq-col-${plan.planDigest.slice(0, 16)}`,
      parentStructureId: representative.id,
      sourceStructureId: representative.id,
      sourceStructureIds: designs.map((item) => item.id),
      sourceModelId: `rfd4-proteina (${mode})`,
      collectionKind: 'independent-candidates',
      ...(representative.lockedChains ? { lockedChains: representative.lockedChains } : {}),
      ...(representative.chainRoles ? { chainRoles: representative.chainRoles } : {}),
      candidates,
    };

    const transferViews: Array<ArrayBufferView | undefined> = [];
    for (const item of designs) {
      transferViews.push(item.caCoords, item.backbone4Coords, item.observedBackboneMask);
    }
    const transferables = collectOwnedTransferables(transferViews);
    return {
      outputs: { structure: structArtifact, sequences: seqSet },
      transferables,
    };
  },

  async dispose() {
    try {
      generator?.destroy?.();
    } catch {
      /* ignore */
    }
    generator = null;
    shardedBundle = null;
    gpuDevice?.destroy();
    gpuDevice = null;
  },
});
