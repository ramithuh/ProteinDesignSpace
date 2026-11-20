/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping sokrypton/rfd4-proteina-webgpu
// loaded directly from /vendor/rfd4-proteina/ as plain ES modules.
// Fetches INT8 sharded weights from https://web.rfd4-proteina.org/weights/int8 (CORS *).

import { parsePdbBackbone } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import type { SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const RFD4_SHARDED_URL = 'https://web.rfd4-proteina.org/weights/int8';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4Mod: any = null;
let gpuDevice: GPUDevice | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let shardedBundle: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let generator: any = null;

async function ensureLoaded(id: number) {
  if (generator && gpuDevice) return;

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
      includeEncoder: false,
      onProgress: (p: { loaded?: number; total?: number; loadedBytes?: number; totalBytes?: number; message?: string }) => {
        const loaded = p.loadedBytes ?? p.loaded ?? 0;
        const total = p.totalBytes ?? p.total ?? 0;
        const frac = total > 0 ? loaded / total : 0.2;
        const mb = (loaded / 1024 / 1024).toFixed(1);
        const totMb = total > 0 ? (total / 1024 / 1024).toFixed(0) : '330';
        self.postMessage({
          id,
          type: 'progress',
          phase: 'weights',
          progress: Math.min(0.95, frac),
          detail: p.message || `RFD4 INT8 weights: ${mb} / ${totMb} MB`,
        });
      },
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
      shardedBundle.weights ?? shardedBundle,
      shardedBundle.config,
    );
  }
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
      const length = Math.max(30, Math.min(256, Number(p.length ?? 70)));
      const nsteps = Math.max(10, Math.min(120, Number(p.nsteps ?? 30)));
      const recycleSchedule = (p.recycleSchedule ?? 'low') as 'low' | 'medium';
      const seed = Number(p.seed ?? 42);
      const customSpecYaml = String(p.customSpecYaml ?? '').trim();
      const template = msg.inputs?.template as StructureArtifact | undefined;

      await ensureLoaded(msg.id);

      self.postMessage({
        id: msg.id,
        type: 'progress',
        phase: 'featurize',
        progress: 0.02,
        detail: customSpecYaml
          ? 'Building conditioned RFD4 batch from YAML/JSON spec...'
          : `Building unconditional RFD4 monomer batch (L=${length})...`,
      });

      const specInput = customSpecYaml
        ? customSpecYaml
        : { contig: `${length}-${length}`, conditions: {} };

      const built = rfd4Mod.buildBatch(specInput, {
        inputCif: template?.cifText ?? template?.pdbText,
        seed,
      });

      const res = await generator.generate(built.batch, {
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
      });

      const pdbText: string = res.pdb;
      const cifText: string = res.cif;
      const parsed = parsePdbBackbone(pdbText);
      const geometry = computeGeometryStats(parsed.caCoords);

      const structArtifact: StructureArtifact = {
        kind: 'StructureArtifact',
        id: `rfd4-L${parsed.nResidues}-s${seed}`,
        label: `RFD4-Proteina (L=${parsed.nResidues}, steps=${res.stepsRun})`,
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
        sourceModelId: 'rfd4-proteina (native co-design)',
        candidates: [
          {
            id: 'rfd4-native-1',
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
