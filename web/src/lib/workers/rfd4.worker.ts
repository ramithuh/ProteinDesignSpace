/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping sokrypton/rfd4-proteina-webgpu
// Supports: Unconditional, Partial Diffusion (lazy +138MB encoder), RFD4 Inverse Folding,
// Upstream Benchmark Presets (PD-L1 binder, 1PRW motif, SAM H-bond pocket, DNA, CATH),
// and custom ConditionSpec / YAML / JSON conditioning.

import { AA3_TO_AA1, parsePdbBackbone } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { fetchCachedBytes } from '$lib/engine/opfs-cache';
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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4Mod: any = null;
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
    const url = `${self.location.origin}/vendor/rfd4-proteina/webgpu/index.js`;
    rfd4Mod = await import(/* @vite-ignore */ url);
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
      detail: 'Fetching RFD4-Proteina INT8 shards (~330 MB, cached after first run)...',
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
          detail: p.message || `RFD4 INT8 weights: ${mb} / ${totMb} MB`,
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
      detail: 'Uploading RFD4-Proteina FP16 weights to WebGPU...',
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
      const mode = String(p.mode ?? 'unconditional');
      const length = Math.max(30, Math.min(256, Number(p.length ?? 70)));
      const nsteps = Math.max(10, Math.min(120, Number(p.nsteps ?? 60)));
      const recycleSchedule = (p.recycleSchedule ?? 'low') as 'none' | 'low' | 'medium';
      const earlyStop = p.earlyStop !== false;
      const seed = Number(p.seed ?? 42);
      const customSpecYaml = String(p.customSpecYaml ?? '').trim();
      const partialFraction = Math.max(0.05, Math.min(0.95, Number(p.partialFraction ?? 0.5)));
      const template = msg.inputs?.template as StructureArtifact | undefined;
      const condArtifact = msg.inputs?.condition as ConditionSpecArtifact | undefined;

      // Validate mode and resolve active preset ID strictly based on `mode`
      if (
        mode !== 'unconditional' &&
        mode !== 'conditioned' &&
        mode !== 'preset' &&
        mode !== 'partial' &&
        mode !== 'inverse-folding'
      ) {
        throw new Error(`Unknown RFD4 mode "${mode}"`);
      }

      if (mode === 'conditioned' && !condArtifact) {
        throw new Error(
          'RFD4 "conditioned" mode requires a wired ConditionSpec input (e.g. from an RFD4 Condition node)',
        );
      }

      const activePresetId =
        mode === 'preset'
          ? String(p.presetId ?? 'custom_cath_monomer')
          : mode === 'conditioned' && condArtifact?.cifUrl?.startsWith('preset:')
            ? condArtifact.cifUrl.slice(7)
            : null;

      const needEncoder =
        mode === 'partial' ||
        mode === 'inverse-folding' ||
        activePresetId === 'partial_diffusion_1prw' ||
        activePresetId === 'inverse_folding_1prw' ||
        (mode === 'conditioned' &&
          (Boolean(condArtifact?.inverseFolding) ||
            (condArtifact?.partialFraction != null && condArtifact.partialFraction > 0)));

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

      let runLabel = `RFD4 (L=${length})`;

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

      if (mode === 'preset') {
        built = await buildFromPreset(activePresetId!);
      } else if (mode === 'conditioned') {
        const cond = condArtifact!;
        if (activePresetId) {
          built = await buildFromPreset(activePresetId);
          if (cond.partialFraction != null && cond.partialFraction > 0) {
            await ensureLoaded(msg.id, true);
            genOpts.partialTrajectoryFractionSteps = cond.partialFraction;
          }
          if (cond.inverseFolding) {
            await ensureLoaded(msg.id, true);
            genOpts.inverseFolding = true;
          }
        } else {
          runLabel = `RFD4 · ${cond.label}`;
          let inputCif: string | Uint8Array | undefined =
            cond.inputCif ?? template?.cifText ?? template?.pdbText;
          if (!inputCif && cond.cifUrl) {
            inputCif = await fetchExampleCifBytes(msg.id, cond.cifUrl);
          }
          if (cond.partialFraction != null && cond.partialFraction > 0) {
            genOpts.partialTrajectoryFractionSteps = cond.partialFraction;
          }
          if (cond.inverseFolding) genOpts.inverseFolding = true;
          built = rfd4Mod.buildBatch(
            { contig: cond.contig, conditions: cond.conditions ?? {} },
            { inputCif, seed },
          );
        }
      } else if (mode === 'partial' || mode === 'inverse-folding') {
        const inputCif = template?.cifText ?? template?.pdbText;
        if (!inputCif) {
          throw new Error(`RFD4 "${mode}" mode requires a wired template StructureArtifact (e.g. from PDB Input)`);
        }
        if (mode === 'partial') {
          genOpts.partialTrajectoryFractionSteps = partialFraction;
          runLabel = `RFD4 partial (frac=${partialFraction})`;
        } else {
          genOpts.inverseFolding = true;
          runLabel = `RFD4 inverse-fold`;
        }
        // Omit hardcoded contig so non-1-indexed, gapped, or multi-chain templates are preserved verbatim
        built = rfd4Mod.buildBatch(
          { input: 'template', conditions: {} },
          { inputCif, seed },
        );
      } else if (customSpecYaml) {
        runLabel = 'RFD4 custom spec';
        built = rfd4Mod.buildBatch(customSpecYaml, {
          inputCif: template?.cifText ?? template?.pdbText,
          seed,
        });
      } else {
        built = rfd4Mod.buildBatch(
          { contig: `${length}-${length}`, conditions: {} },
          { seed },
        );
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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let res: any;
      let gpuValErr: GPUError | null = null;
      try {
        res = await generator.generate(built.batch, genOpts);
      } finally {
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

      // Fallback recovery: if trunk logits were zero/NaN or collapsed to a single residue (e.g. poly-ALA)
      // while local_latents are valid, decode sequence directly using the autoencoder decoder's own
      // logit_linear head (by passing unconditioned C_SEQ from the batch).
      if (nativeSeq.length >= 10 && (logitsNonZero === 0 || logitsNan > 0 || uniqueNativeAAs <= 1)) {
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
            console.info('[rfd4] Recovered sequence via decoder head:', recSeq.slice(0, 30), `(${recUnique} unique AAs)`);
          }
        } catch (recErr) {
          console.warn('[rfd4] Decoder-head sequence recovery failed:', recErr);
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
            ? ` Steps=${nsteps} is below the stable floor (~58) for RFD4's vf_tsr schedule — set Steps to 60 and use Recycle=none for speed.`
            : ' Try reloading the page or switching browser/GPU backend.';
        throw new Error(
          `RFD4-Proteina produced a degenerate single-residue sequence (${finalSeq.slice(0, 12)}…, logits nz=${logitsNonZero}, nan=${logitsNan})${errHint}.${stepHint}`,
        );
      }

      const geometry = computeGeometryStats(parsed.caCoords);
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
          earlyStop,
          seed,
          activePresetId,
          condId: mode === 'conditioned' ? (condArtifact?.id ?? null) : null,
          customSpecYaml: mode === 'unconditional' ? customSpecYaml : '',
          partialFraction: mode === 'partial' ? partialFraction : null,
          templateId: template?.id ?? null,
        }),
      );

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
        geometry,
      };

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `rfd4-seq-${structArtifact.id}`,
        parentStructureId: structArtifact.id,
        sourceModelId: `rfd4-proteina (${mode})`,
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

      self.postMessage({
        id: msg.id,
        type: 'result',
        outputs: { structure: structArtifact, sequences: seqSet },
        elapsedMs: performance.now() - t0,
      });
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
