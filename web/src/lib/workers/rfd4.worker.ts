/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping sokrypton/rfd4-proteina-webgpu
// Supports: Unconditional, Partial Diffusion (lazy +138MB encoder), RFD4 Inverse Folding,
// Upstream Benchmark Presets (PD-L1 binder, 1PRW motif, SAM H-bond pocket, DNA, CATH),
// and custom ConditionSpec / YAML / JSON conditioning.

import { AA3_TO_AA1, parsePdbBackbone } from '$lib/protein-ops/pdb';
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
import type { ConditionSpecArtifact, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const RFD4_ORIGIN = 'https://web.rfd4-proteina.org';
const RFD4_SHARDED_URL = `${RFD4_ORIGIN}/weights/int8`;

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

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

async function ensureLoaded(id: number, needEncoder: boolean) {
  if (!rfd4Mod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading RFD4-Proteina WebGPU modules...',
    });
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
    const res = await rfd4Mod.requestWebGpuDevice();
    if (!res?.device) {
      throw new Error('No compatible WebGPU device found for RFD4-Proteina');
    }
    const dev: GPUDevice = res.device;
    gpuDevice = dev;
    void dev.lost.then(() => {
      if (gpuDevice === dev) {
        gpuDevice = null;
        generator = null;
      }
    });
  }

  if (!shardedBundle) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.05,
      detail: 'Fetching RFD4-Proteina weights (~330 MB, cached after first run)...',
    });
    shardedBundle = await rfd4Mod.fetchShardedWeights(RFD4_SHARDED_URL, {
      includeEncoder: needEncoder,
      onProgress: (p: { loaded?: number; total?: number; loadedBytes?: number; totalBytes?: number; message?: string }) => {
        const loaded = p.loadedBytes ?? p.loaded ?? 0;
        const total = p.totalBytes ?? p.total ?? 0;
        const frac = total > 0 ? loaded / total : 0.2;
        const mb = (loaded / 1024 / 1024).toFixed(1);
        const totMb = total > 0 ? (total / 1024 / 1024).toFixed(0) : needEncoder ? '469' : '330';
        self.postMessage({
          id,
          type: 'progress',
          phase: 'weights',
          progress: Math.min(0.95, frac),
          detail: p.message || `RFD4-Proteina weights: ${mb} / ${totMb} MB`,
        });
      },
    });
  } else if (needEncoder && !shardedBundle.encoderLoaded && typeof shardedBundle.loadEncoder === 'function') {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.1,
      detail: 'Loading RFD4 Autoencoder Encoder (+138 MB for partial diffusion / inverse folding)...',
    });
    await shardedBundle.loadEncoder((p: { loadedBytes?: number; totalBytes?: number }) => {
      const loaded = p.loadedBytes ?? 0;
      const total = p.totalBytes ?? 138 * 1024 * 1024;
      self.postMessage({
        id,
        type: 'progress',
        phase: 'weights',
        progress: Math.min(0.95, loaded / Math.max(1, total)),
        detail: `RFD4 Encoder: ${(loaded / 1024 / 1024).toFixed(1)} / ${(total / 1024 / 1024).toFixed(0)} MB`,
      });
    });
  }

  if (!generator) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.96,
      detail: 'Uploading RFD4-Proteina weights to GPU...',
    });
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

async function fetchExampleCifBytes(id: number, relUrl: string): Promise<Uint8Array> {
  const cleanRel = relUrl.replace(/^\.\//, '');
  const fullUrl = relUrl.startsWith('http') ? relUrl : `${RFD4_ORIGIN}/${cleanRel}`;
  self.postMessage({
    id,
    type: 'progress',
    phase: 'featurize',
    progress: 0.04,
    detail: `Fetching context CIF (${cleanRel.split('/').pop()})...`,
  });
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
  id: number,
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
      self.postMessage({
        id,
        type: 'progress',
        phase: 'featurize',
        progress: 0.05,
        detail: `Fetching CCD ligand definitions (${missing.join(', ')})...`,
      });
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

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'dispose') {
      try {
        generator?.destroy?.();
      } catch {
        /* ignore */
      }
      generator = null;
      shardedBundle = null;
      gpuDevice?.destroy();
      gpuDevice = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const p = msg.params ?? {};
      const template = (msg.inputs?.structure ?? msg.inputs?.template) as StructureArtifact | undefined;
      const condArtifact = msg.inputs?.condition as ConditionSpecArtifact | undefined;
      const modeParam = String(p.mode ?? 'unconditional');
      const requestedEarlyStop = Boolean(p.earlyStop);
      const effectiveEarlyStop = modeParam === 'inverse-folding' ? false : requestedEarlyStop;
      // Keep explicit p.<key> reads intact for param-schema bidirectional coverage guards
      const rawParams: Record<string, unknown> = {
        mode: modeParam,
        length: p.length,
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
        unindexedMotif: p.unindexedMotif,
        hbondMode: p.hbondMode,
        hbondSelection: p.hbondSelection,
        cathCode: p.cathCode,
        customSpecYaml: p.customSpecYaml,
      };
      const inlineContig = String(rawParams.contig ?? '').trim();
      const inlineLock = String(rawParams.lockChain ?? '').trim().toUpperCase();
      const inlineHotspots = String(rawParams.hotspots ?? '').trim();
      const inlineMotif = String(rawParams.motifSpan ?? '').trim();
      const inlineUnidx = Boolean(rawParams.unindexedMotif);
      const inlineHbondMode = String(rawParams.hbondMode ?? 'none');
      const inlineHbondSel = String(rawParams.hbondSelection ?? '').trim();
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

      await ensureLoaded(msg.id, needEncoder);

      // Build input batch + generation options
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let built: { batch: Record<string, any>; atoms?: any; n?: number };
      const genOpts: Record<string, unknown> = {
        seed,
        nsteps,
        recycleSchedule,
        earlyStop,
        inputFrame: true,
        onStep: (info: { step: number; nsteps: number }) => {
          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'sampling',
            progress: (info.step + 1) / Math.max(1, info.nsteps),
            detail: `RFD4 step ${info.step + 1}/${info.nsteps}`,
          });
        },
      };

      let runLabel = preResolved.runLabel;

      const buildFromPreset = async (presetId: string) => {
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
          await ensureLoaded(msg.id, true);
        }

        if (preset.partialTrajectoryFractionSteps != null) {
          genOpts.partialTrajectoryFractionSteps = preset.partialTrajectoryFractionSteps;
        }
        if (preset.inverseFolding) {
          genOpts.inverseFolding = true;
          genOpts.earlyStop = false;
        }

        if (preset.cifUrl) {
          const cifBytes = await fetchExampleCifBytes(msg.id, preset.cifUrl);
          if (preset.specMode) {
            return rfd4Mod.buildBatch(
              { contig: preset.contig, conditions: preset.conditions ?? {} },
              { inputCif: cifBytes, seed },
            );
          }
          // Annotated upstream .cif.zst already carries condition columns
          return rfd4Mod.buildBatch(cifBytes, { seed });
        }
        return rfd4Mod.buildBatch(
          { contig: preset.contig, conditions: preset.conditions ?? {} },
          { seed },
        );
      };

      if (preResolved.kind === 'preset') {
        built = await buildFromPreset(activePresetId!);
        if (preResolved.effectivePartialFraction != null) {
          await ensureLoaded(msg.id, true);
          genOpts.partialTrajectoryFractionSteps = preResolved.effectivePartialFraction;
        }
        if (preResolved.inverseFolding) {
          await ensureLoaded(msg.id, true);
          genOpts.inverseFolding = true;
          genOpts.earlyStop = false;
        }
      } else {
        let inputCif: string | Uint8Array | undefined =
          mode === 'partial' || mode === 'inverse-folding'
            ? (template?.cifText ?? template?.pdbText)
            : (condArtifact?.inputCif ?? template?.cifText ?? template?.pdbText);
        if (!inputCif && condArtifact?.cifUrl && !activePresetId) {
          inputCif = await fetchExampleCifBytes(msg.id, condArtifact.cifUrl);
        }
        const prepared =
          preResolved.usesTemplate || Boolean(inputCif && (customSpecYaml || mode !== 'unconditional'))
            ? await prepareTemplateForRfd4(msg.id, inputCif)
            : null;
        const resolved = prepared
          ? resolveRfd4CompiledInput(rawParams, template, condArtifact, prepared.templateContext)
          : preResolved;
        runLabel = resolved.runLabel;

        if (resolved.effectivePartialFraction != null) {
          genOpts.partialTrajectoryFractionSteps = resolved.effectivePartialFraction;
        }
        if (resolved.inverseFolding) {
          genOpts.inverseFolding = true;
          genOpts.earlyStop = false;
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
          built = rfd4Mod.buildBatch(
            parsedYaml?.contig ? parsedYaml : { input: 'template', conditions: yamlConds },
            { inputAtomArray: prepared!.atomArray, seed },
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
          built = rfd4Mod.buildBatch(
            { input: 'template', conditions: resolved.conditions },
            { inputAtomArray: prepared!.atomArray, seed },
          );
        } else if (resolved.kind === 'custom-yaml') {
          built = rfd4Mod.buildBatch(
            resolved.customSpecYaml,
            prepared
              ? { inputAtomArray: prepared.atomArray, seed }
              : { inputCif: template?.cifText ?? template?.pdbText, seed },
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
          built = rfd4Mod.buildBatch(
            { contig: resolved.contig, conditions: resolved.conditions },
            prepared
              ? { inputAtomArray: prepared.atomArray, seed }
              : inputCif && condArtifact
                ? { inputCif, seed }
                : { seed },
          );
        }
      }

      if (built.atoms) {
        genOpts.inputAtoms = built.atoms;
      }

      // Capture any silent WebGPU validation errors (which otherwise leave output buffers all-zero!)
      const scopeDevice = gpuDevice;
      const pushedScope = typeof scopeDevice?.pushErrorScope === 'function';
      if (pushedScope) {
        scopeDevice!.pushErrorScope('validation');
      }
      const releaseGeneratorActivations = () => {
        try {
          generator?.releaseActivations?.();
          generator?.trunk?.engine?.releaseActivations?.();
          generator?.decoder?.engine?.releaseActivations?.();
          generator?.encoder?.engine?.releaseActivations?.();
        } catch {
          /* ignore */
        }
      };

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
          gpuValErr: gpuValErr?.message ?? null,
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
        const errHint = gpuValErr ? ` (WebGPU validation error: ${gpuValErr.message})` : '';
        const stepHint =
          nsteps < 58
            ? ` Steps=${nsteps} is below the recommended minimum (58) — set Steps to 60 and Recycle to none for reliable convergence.`
            : ' Try reloading the page or switching browser/GPU backend.';
        throw new Error(
          `RFD4-Proteina produced a degenerate single-residue sequence (${finalSeq.slice(0, 12)}…, logits nz=${logitsNonZero}, nan=${logitsNan})${errHint}.${stepHint}`,
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
          seed,
          activePresetId,
          condId: condArtifact?.id ?? null,
          inlineContig,
          inlineLock,
          inlineHotspots,
          inlineMotif,
          inlineUnidx,
          inlineHbondMode,
          inlineHbondSel,
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

      const structArtifact: StructureArtifact = {
        kind: 'StructureArtifact',
        id: `rfd4-${mode}-L${parsed.nResidues}-s${seed}-${runHash}`,
        label: `${runLabel} (L=${parsed.nResidues}, steps=${res.stepsRun})`,
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
        bfactorSemantics: 'crystal-b',
        atomGranularity: 'all-atom',
        provenance: {
          quantization: 'int8',
          earlyStopped,
          ...(degradedFallback ? { degradedFallback } : {}),
        },
        ...designMeta,
        geometry,
      };

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `rfd4-seq-${structArtifact.id}`,
        parentStructureId: structArtifact.id,
        sourceStructureId: structArtifact.id,
        sourceModelId: `rfd4-proteina (${mode})`,
        ...(designMeta.lockedChains ? { lockedChains: designMeta.lockedChains } : {}),
        ...(designMeta.chainRoles ? { chainRoles: designMeta.chainRoles } : {}),
        candidates: [
          {
            id: mode === 'inverse-folding' ? 'rfd4-invfold-1' : 'rfd4-native-1',
            sequence: finalSeq,
            score: null,
            globalScore: null,
            sourceModelId: 'rfd4-proteina',
            seed,
            ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
          },
        ],
      };

      const transferables = collectOwnedTransferables([
        structArtifact.caCoords,
        structArtifact.backbone4Coords,
        structArtifact.observedBackboneMask,
      ]);
      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          outputs: { structure: structArtifact, sequences: seqSet },
          elapsedMs: performance.now() - t0,
        },
        transferables,
      );
    }
  } catch (err) {
    self.postMessage({
      id: msg.id,
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
    });
  }
};

export {};
