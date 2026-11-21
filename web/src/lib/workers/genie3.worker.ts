/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/genie3-webgpu (Apache-2.0 port of aqlaboratory/genie3)
// Generates de novo protein backbones (ideal N/CA/C/O + Cα) and optional 1-step native sequences
// via SE(3)-equivariant DDIM diffusion on WebGPU (~45 MB FP16 weights).

import { parsePdbBackbone } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import {
  buildDefaultGenie3ConditionFromTemplate,
  compileConditionToGenie3Spec,
} from '$lib/engine/conditioning';
import type { ConditionSpecArtifact, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let genie3Mod: any = null;
let gpuDevice: GPUDevice | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let weightBundle: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let generator: any = null;

async function ensureLoaded(id: number) {
  if (!genie3Mod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading Genie 3 WebGPU modules...',
    });
    const url = `${self.location.origin}/vendor/genie3/webgpu/index.js`;
    genie3Mod = await import(/* @vite-ignore */ url);
  }

  if (!gpuDevice) {
    const res = await genie3Mod.requestWebGpuDevice();
    if (!res?.device) {
      throw new Error('No compatible WebGPU device found for Genie 3');
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

  if (!weightBundle) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.05,
      detail: 'Fetching Genie 3 weights (~45 MB, cached after first run)...',
    });
    const envWeightsUrl = (import.meta.env.VITE_GENIE3_WEIGHTS_URL as string | undefined)?.trim();
    const weightsUrl =
      envWeightsUrl || 'https://huggingface.co/ramithuh/genie3-webgpu/resolve/main/fp16';
    weightBundle = await genie3Mod.fetchGenie3Weights(weightsUrl, {
      cache: true,
      onProgress: (p: { loaded?: number; total?: number; shardIndex?: number; totalShards?: number; shard?: string }) => {
        const loaded = p.loaded ?? 0;
        const total = p.total ?? 45 * 1024 * 1024;
        const frac = total > 0 ? loaded / total : 0.2;
        const mb = (loaded / 1024 / 1024).toFixed(1);
        const totMb = (total / 1024 / 1024).toFixed(1);
        self.postMessage({
          id,
          type: 'progress',
          phase: 'weights',
          progress: Math.min(0.92, frac),
          detail: `Genie 3 weights: ${mb} / ${totMb} MB`,
        });
      },
    });
  }

  if (!generator) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.94,
      detail: 'Uploading Genie 3 weights to GPU...',
    });
    generator = await genie3Mod.Genie3GeneratorGpu.create(
      gpuDevice,
      weightBundle,
      weightBundle.config ?? null,
      { half: true },
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
      weightBundle = null;
      gpuDevice?.destroy();
      gpuDevice = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const p = msg.params ?? {};
      const length = Math.max(30, Math.min(256, Math.round(Number(p.length ?? 80))));
      const nSampleStep = Math.max(10, Math.min(250, Math.round(Number(p.nSampleStep ?? 100))));
      const directionScale = Math.max(0.1, Math.min(1.5, Number(p.directionScale ?? 0.8)));
      const noiseScale = Math.max(0, Math.min(1.5, Number(p.noiseScale ?? 1.0)));
      const predictSequence = p.predictSequence !== false;
      const seed = Math.max(0, Math.round(Number(p.seed ?? 42)));
      const condArtifact = msg.inputs?.condition as ConditionSpecArtifact | undefined;
      const template = msg.inputs?.template as StructureArtifact | undefined;

      await ensureLoaded(msg.id);

      let batch: Record<string, unknown> | undefined;
      let compiledSummary = '';
      let expectedChainLengths: number[] | undefined;
      const effectiveCond =
        condArtifact ?? (template ? buildDefaultGenie3ConditionFromTemplate(template, length) : undefined);
      if (effectiveCond) {
        const compiled = compileConditionToGenie3Spec(effectiveCond, template, length);
        batch = genie3Mod.createFeaturesFromSpec({
          length: compiled.length,
          asymId: compiled.asymId,
          symId: compiled.symId,
          entityId: compiled.entityId,
          residueIndex: compiled.residueIndex,
          condGroup: compiled.condGroup,
          condSeqMask: compiled.condSeqMask,
          condStructMask: compiled.condStructMask,
          condInterfaceMask: compiled.condInterfaceMask,
          gtRestype: compiled.gtRestype,
          gtAtomPositions: compiled.gtAtomPositions,
        });
        compiledSummary = compiled.summary;
        expectedChainLengths = compiled.chainLengths;
      }

      const res = await generator.generate({
        ...(batch ? { batch } : { length }),
        nSampleStep,
        directionScale,
        noiseScale,
        eta: 1.0,
        predictSequence,
        sequenceTemperature: 0.1,
        seed,
        yieldBetweenSteps: true,
        onStep: (info: { stepIndex: number; totalSteps: number; timestep: number }) => {
          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'sampling',
            progress: (info.stepIndex + 1) / Math.max(1, info.totalSteps),
            detail: `Genie 3 DDIM step ${info.stepIndex + 1}/${info.totalSteps} (t=${info.timestep})`,
          });
        },
      });

      const pdbText: string = res.pdb;
      const cifText: string = res.cif;
      const parsed = parsePdbBackbone(pdbText);
      const finalSeq: string =
        typeof res.sequence === 'string' && res.sequence.length === parsed.nResidues
          ? res.sequence
          : parsed.sequence;

      const geometry = computeGeometryStats(parsed.caCoords, parsed.chainIds);
      const candChainLengths =
        expectedChainLengths && expectedChainLengths.length > 1 &&
        expectedChainLengths.reduce((a, b) => a + b, 0) === finalSeq.length
          ? expectedChainLengths
          : parsed.diagnostics.chainLengths.length > 1
            ? parsed.diagnostics.chainLengths
            : undefined;
      const runHash = fnv1a(
        JSON.stringify({
          length,
          nSampleStep,
          directionScale,
          noiseScale,
          predictSequence,
          seed,
          condId: condArtifact?.id ?? null,
          templateId: template?.id ?? null,
        }),
      );

      const structArtifact: StructureArtifact = {
        kind: 'StructureArtifact',
        id: `genie3-L${parsed.nResidues}-s${seed}-${runHash}`,
        label: compiledSummary
          ? `Genie 3 (${compiledSummary}, ${res.stepsRun} steps)`
          : `Genie 3 (L=${parsed.nResidues}, ${res.stepsRun} DDIM steps)`,
        sourceModelId: 'genie3',
        length: parsed.nResidues,
        sequence: finalSeq,
        caCoords: parsed.caCoords,
        backbone4Coords: parsed.backbone4Coords,
        pdbText,
        cifText,
        chainIds: parsed.chainIds,
        chainLengths: candChainLengths ?? parsed.diagnostics.chainLengths,
        diagnostics: parsed.diagnostics,
        geometry,
      };

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `genie3-seq-${structArtifact.id}`,
        parentStructureId: structArtifact.id,
        sourceModelId: predictSequence ? 'genie3 (1-step SequenceNet)' : 'genie3 (poly-ALA backbone)',
        candidates: [
          {
            id: 'genie3-native-1',
            sequence: finalSeq,
            score: null,
            globalScore: null,
            sourceModelId: 'genie3',
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
