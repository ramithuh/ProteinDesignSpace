/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping sokrypton/rfd4-proteina-webgpu
// Supports: Unconditional, Partial Diffusion (lazy +138MB encoder), RFD4 Inverse Folding,
// Upstream Benchmark Presets (PD-L1 binder, 1PRW motif, SAM H-bond pocket, DNA, CATH),
// and custom ConditionSpec / YAML / JSON conditioning.

import { parsePdbBackbone } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { fetchCachedBytes } from '$lib/engine/opfs-cache';
import type { ConditionSpecArtifact, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const RFD4_ORIGIN = 'https://web.rfd4-proteina.org';
const RFD4_SHARDED_URL = `${RFD4_ORIGIN}/weights/int8`;

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
    gpuDevice = res.device;
    void gpuDevice!.lost.then(() => {
      gpuDevice = null;
      generator = null;
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
      detail: 'Uploading RFD4-Proteina weights to WebGPU...',
    });
    generator = await rfd4Mod.Rfd4pGeneratorGpu.create(
      gpuDevice,
      shardedBundle.tensors ?? shardedBundle,
      shardedBundle.config,
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
      const nsteps = Math.max(10, Math.min(120, Number(p.nsteps ?? 30)));
      const recycleSchedule = (p.recycleSchedule ?? 'low') as 'low' | 'medium';
      const seed = Number(p.seed ?? 42);
      const customSpecYaml = String(p.customSpecYaml ?? '').trim();
      const template = msg.inputs?.template as StructureArtifact | undefined;
      const condArtifact = msg.inputs?.condition as ConditionSpecArtifact | undefined;

      // Determine preset ID if either mode === 'preset' or condition block references preset:<id>
      const presetFromCond =
        condArtifact?.cifUrl?.startsWith('preset:') ? condArtifact.cifUrl.slice(7) : null;
      const activePresetId = presetFromCond ?? (mode === 'preset' ? String(p.presetId ?? 'custom_cath_monomer') : null);

      const needEncoder =
        mode === 'partial' ||
        mode === 'inverse-folding' ||
        activePresetId === 'partial_diffusion_1prw' ||
        activePresetId === 'inverse_folding_1prw' ||
        Boolean(condArtifact?.inverseFolding) ||
        (condArtifact?.partialFraction != null && condArtifact.partialFraction > 0);

      await ensureLoaded(msg.id, needEncoder);

      // Build input batch + generation options
      let built: { batch: Record<string, unknown> };
      const genOpts: Record<string, unknown> = {
        seed,
        nsteps,
        recycleSchedule,
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

      if (activePresetId) {
        const preset = rfd4Mod.getRfd4pPresetById(activePresetId);
        if (!preset) throw new Error(`Unknown RFD4 preset "${activePresetId}"`);
        runLabel = `RFD4 · ${preset.id}`;

        if (preset.partialTrajectoryFractionSteps != null) {
          genOpts.partialTrajectoryFractionSteps = preset.partialTrajectoryFractionSteps;
        }
        if (preset.inverseFolding) {
          genOpts.inverseFolding = true;
        }

        if (preset.cifUrl) {
          const cifBytes = await fetchExampleCifBytes(msg.id, preset.cifUrl);
          if (preset.specMode) {
            built = rfd4Mod.buildBatch(
              { contig: preset.contig, conditions: preset.conditions ?? {} },
              { inputCif: cifBytes, seed },
            );
          } else {
            // Annotated upstream .cif.zst already carries condition columns
            built = rfd4Mod.buildBatch(cifBytes, { seed });
          }
        } else {
          built = rfd4Mod.buildBatch(
            { contig: preset.contig, conditions: preset.conditions ?? {} },
            { seed },
          );
        }
      } else if (condArtifact) {
        runLabel = `RFD4 · ${condArtifact.label}`;
        const inputCif = condArtifact.inputCif ?? template?.cifText ?? template?.pdbText;
        if (condArtifact.partialFraction != null && condArtifact.partialFraction > 0) {
          genOpts.partialTrajectoryFractionSteps = condArtifact.partialFraction;
        }
        if (condArtifact.inverseFolding) genOpts.inverseFolding = true;
        built = rfd4Mod.buildBatch(
          { contig: condArtifact.contig, conditions: condArtifact.conditions ?? {} },
          { inputCif, seed },
        );
      } else if (mode === 'partial' || mode === 'inverse-folding') {
        const inputCif = template?.cifText ?? template?.pdbText;
        if (!inputCif) {
          throw new Error(`RFD4 "${mode}" mode requires a wired template StructureArtifact (e.g. from PDB Input)`);
        }
        if (mode === 'partial') {
          const frac = Math.max(0.05, Math.min(0.95, Number(p.partialFraction ?? 0.5)));
          genOpts.partialTrajectoryFractionSteps = frac;
          runLabel = `RFD4 partial (frac=${frac})`;
        } else {
          genOpts.inverseFolding = true;
          runLabel = `RFD4 inverse-fold`;
        }
        const tplLen = template?.length ?? length;
        built = rfd4Mod.buildBatch(
          { contig: `A1-${tplLen}`, conditions: {} },
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

      const res = await generator.generate(built.batch, genOpts);
      const pdbText: string = res.pdb;
      const cifText: string = res.cif;
      const parsed = parsePdbBackbone(pdbText);
      const geometry = computeGeometryStats(parsed.caCoords);

      const structArtifact: StructureArtifact = {
        kind: 'StructureArtifact',
        id: `rfd4-${mode}-L${parsed.nResidues}-s${seed}`,
        label: `${runLabel} (L=${parsed.nResidues}, steps=${res.stepsRun})`,
        sourceModelId: 'rfd4-proteina',
        length: parsed.nResidues,
        sequence: parsed.sequence,
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
            sequence: parsed.sequence,
            score: null,
            globalScore: null,
            sourceModelId: 'rfd4-proteina',
            seed,
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
