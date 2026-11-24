/// <reference lib="webworker" />
// Layer 3 Plugin Worker — rfdiffusion (ramithuh/rfdiffusion-webgpu, BSD · RosettaCommons/RFdiffusion Base_ckpt.pt)
// Runs original RFdiffusion SE(3) backbone diffusion (unconditional, contig motif scaffolding, Cn/Dn symmetry).
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress, and disposal.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import {
  collectOwnedTransferables,
  createWorkerHarness,
  fnv1a,
  type WorkerRunContext,
} from '$lib/kernel';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { resolveRfdiffusionMotifContig } from '$lib/protein-ops/motif-placement';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { StructureArtifact, TrajectoryFramesArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/rfdiffusion-webgpu/resolve/main';

type Precision = 'int8' | 'fp16';
type TaskMode = 'unconditional' | 'motif' | 'symmetry';

const VALID_SYMMETRIES = new Set(['c2', 'c3', 'c4', 'c6', 'd2', 'd3']);
const SYMMETRY_COPIES: Record<string, number> = { c2: 2, c3: 3, c4: 4, c6: 6, d2: 4, d3: 6 };

function normalizePrecision(raw: unknown): Precision {
  if (raw == null || String(raw).trim() === '') return 'int8';
  const val = String(raw).trim().toLowerCase();
  if (val === 'int8' || val === 'fp16') return val;
  throw new Error(`Unsupported RFdiffusion precision "${String(raw)}" (supported: int8, fp16).`);
}

function normalizeTaskMode(raw: unknown): TaskMode {
  const val = String(raw ?? 'unconditional').trim().toLowerCase();
  if (val === 'motif' || val === 'symmetry') return val;
  return 'unconditional';
}

function normalizeSymmetry(raw: unknown): string {
  const val = String(raw ?? 'c3').trim().toLowerCase();
  return VALID_SYMMETRIES.has(val) ? val : 'c3';
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfdMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let ownedDevice: GPUDevice | null = null;
let loadedPrecision: Precision | null = null;

function disposeEngine() {
  try {
    engine?.destroy();
  } catch {
    /* ignore */
  }
  engine = null;
  try {
    ownedDevice?.destroy();
  } catch {
    /* ignore */
  }
  ownedDevice = null;
  loadedPrecision = null;
}

async function ensureEngine(ctx: WorkerRunContext, precision: Precision) {
  if (engine && loadedPrecision === precision) return engine;
  disposeEngine();

  if (!rfdMod) {
    ctx.progress('init', 0.02, 'Loading RFdiffusion WebGPU modules...');
    const base = `${self.location.origin}/vendor/rfdiffusion`;
    rfdMod = await import(/* @vite-ignore */ `${base}/webgpu/index.js`);
  }

  const { adapter } = await requestResilientWebGpuAdapter({ allowSoftwareFallback: true });
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  ownedDevice = device;
  void device.lost.then(() => {
    if (ownedDevice === device) {
      engine = null;
      ownedDevice = null;
      loadedPrecision = null;
    }
  });

  const envBaseUrl = (import.meta.env.VITE_RFDIFFUSION_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.06, `Fetching RFdiffusion Base (${precision.toUpperCase()}) weights...`);

    try {
      return await rfdMod.createRFdiffusionEngine({
        baseUrl,
        precision,
        device,
        cache: 'none',
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { loaded?: number; total?: number; shard?: number; shards?: number }) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac =
            total > 0
              ? loaded / total
              : prog.shards && prog.shard !== undefined
                ? prog.shard / prog.shards
                : 0.4;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          ctx.progress(
            'weights',
            Math.min(0.45, 0.06 + frac * 0.39),
            total > 0
              ? `RFdiffusion weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
              : `Loading RFdiffusion weights (${precision.toUpperCase()})...`,
          );
        },
      });
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        /integrity|sha-?256|manifest|mismatch/i.test(errMsg)
      ) {
        await evictCachedUrls(fetchedUrls);
        return await loadOnce(true);
      }
      throw err;
    }
  };

  try {
    engine = await loadOnce(false);
    loadedPrecision = precision;
    return engine;
  } catch (err) {
    disposeEngine();
    throw err;
  }
}

createWorkerHarness({
  async run(ctx) {
    const p = ctx.params ?? {};
    const taskMode = normalizeTaskMode(p.taskMode);
    const precision = normalizePrecision(p.precision);
    const rawLength = Math.max(16, Math.min(256, Math.round(Number(p.length ?? 80))));
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));
    const numSteps = Math.max(5, Math.min(50, Math.round(Number(p.numSteps ?? 50))));
    const contig = String(p.contig ?? '').trim();
    const symmetry = taskMode === 'symmetry' ? normalizeSymmetry(p.symmetry) : undefined;
    const seed = Math.max(0, Math.round(Number(p.seed ?? 42)));

    const upstreamStruct = ctx.inputs?.structure as StructureArtifact | undefined;
    if (taskMode === 'motif' && !upstreamStruct) {
      throw new Error('RFdiffusion in "motif" mode requires a connected upstream structure.');
    }

    let effectiveLength = rawLength;
    if (symmetry) {
      const copies = SYMMETRY_COPIES[symmetry] ?? 3;
      const maxSymLen = Math.floor(256 / copies) * copies;
      effectiveLength = Math.min(maxSymLen, Math.max(copies * 8, Math.round(rawLength / copies) * copies));
    }

    let effectiveContig: string | undefined;
    if (taskMode === 'motif' && upstreamStruct) {
      effectiveContig = resolveRfdiffusionMotifContig(upstreamStruct, contig, effectiveLength);
    }

    const activeEngine = await ensureEngine(ctx, precision);

    const items: StructureArtifact[] = [];
    let primaryTrajFrames: Float32Array[] = [];
    let primaryTrajSteps: number[] = [];

    const runHash = fnv1a(
      JSON.stringify({
        taskMode,
        precision,
        effectiveLength,
        numDesigns,
        numSteps,
        effectiveContig: effectiveContig ?? '',
        symmetry: symmetry ?? '',
        seed,
        upstreamId: upstreamStruct?.id ?? null,
      }),
    );

    for (let d = 0; d < numDesigns; d++) {
      if (ctx.signal.aborted) throw new Error('Workflow run cancelled');
      const sampled = await activeEngine.sample({
        ...(effectiveContig
          ? { contig: effectiveContig, inputPdb: upstreamStruct?.pdbText }
          : { length: effectiveLength }),
        ...(symmetry ? { symmetry } : {}),
        numSteps,
        seed: seed + d,
        signal: ctx.signal,
        onProgress: (prog: {
          step?: number;
          numSteps?: number;
          total?: number;
          block?: number;
          blockTotal?: number;
          t?: number;
        }) => {
          const step = prog.step ?? 0;
          const tot = prog.numSteps ?? (prog.block !== undefined ? numSteps : (prog.total ?? numSteps));
          const blockFrac =
            prog.block !== undefined && (prog.blockTotal ?? 0) > 0
              ? Math.min(1, prog.block / prog.blockTotal!)
              : 1;
          const frac = tot > 0 ? Math.min(1, (Math.max(0, step - 1) + blockFrac) / tot) : 0;
          const overall = 0.48 + ((d + frac) / numDesigns) * 0.5;
          ctx.progress(
            'diffuse',
            Math.min(0.98, overall),
            `RFdiffusion design ${d + 1}/${numDesigns}: step ${step}/${tot}`,
          );
        },
      });

      const pdbText = String(sampled.pdbText ?? '');
      const parsed = parseStructureText(pdbText);
      const geometry = computeGeometryStats(
        parsed.caCoords,
        parsed.chainIds,
        parsed.resNums,
        parsed.insCodes,
      );
      const designKey = `design-${d + 1}`;
      const structId = `rfd-${taskMode}-${parsed.nResidues}aa-d${d + 1}-${runHash}`;

      const fixedResidueIndices: number[] = [];
      const maskArr = sampled.diffusionMask as ArrayLike<number> | undefined;
      if (maskArr) {
        for (let i = 0; i < Math.min(maskArr.length, parsed.nResidues); i++) {
          if (Number(maskArr[i]) === 1) fixedResidueIndices.push(i + 1);
        }
      }
      const motifIndicesRaw = Array.isArray(sampled.motifIndices)
        ? (sampled.motifIndices as number[])
        : undefined;
      const nativeMotifCoordsRaw = Array.isArray(sampled.nativeMotifCoords)
        ? (sampled.nativeMotifCoords as number[][])
        : undefined;
      const hasValidMotif =
        Boolean(motifIndicesRaw?.length) &&
        Boolean(nativeMotifCoordsRaw?.length) &&
        motifIndicesRaw!.length === nativeMotifCoordsRaw!.length;

      const uniqueChains = [...new Set(parsed.chainIds)];
      const chainRoles: Record<string, 'target' | 'binder' | 'scaffold'> = {};
      for (const ch of uniqueChains) chainRoles[ch] = 'scaffold';

      let tiedPositions: string | undefined;
      if (symmetry) {
        const copies = SYMMETRY_COPIES[symmetry] ?? 0;
        if (copies >= 2 && parsed.nResidues > 0 && parsed.nResidues % copies === 0) {
          const unit = parsed.nResidues / copies;
          const groups: string[] = [];
          for (let u = 0; u < unit; u++) {
            const tokens: string[] = [];
            for (let c = 0; c < copies; c++) {
              const idx = c * unit + u;
              tokens.push(`${parsed.chainIds[idx]}${parsed.resNums[idx]}`);
            }
            groups.push(tokens.join('='));
          }
          tiedPositions = groups.join('; ');
        }
      }

      const artifact: StructureArtifact = {
        kind: 'StructureArtifact',
        id: structId,
        label: `RFdiffusion${symmetry ? ` (${symmetry.toUpperCase()})` : ''} (${parsed.nResidues} aa${numDesigns > 1 ? ` · #${d + 1}` : ''})`,
        sourceModelId: 'rfdiffusion',
        parentStructureId: upstreamStruct?.id,
        length: parsed.nResidues,
        sequence: parsed.sequence,
        caCoords: parsed.caCoords,
        backbone4Coords: parsed.backbone4Coords,
        pdbText: parsed.pdbText,
        chainIds: parsed.chainIds,
        chainLengths: parsed.diagnostics.chainLengths,
        insCodes: parsed.insCodes,
        sourceResidues: parsed.sourceResidues,
        observedBackboneMask: parsed.observedBackboneMask,
        diagnostics: parsed.diagnostics,
        geometry,
        atomGranularity: 'backbone4',
        bfactorSemantics: 'none',
        chainRoles,
        ...(fixedResidueIndices.length > 0 ? { fixedResidueIndices } : {}),
        ...(hasValidMotif
          ? { motifIndices: motifIndicesRaw, nativeMotifCoords: nativeMotifCoordsRaw }
          : {}),
        ...(tiedPositions ? { tiedPositions } : {}),
        designKey,
        designIndex: d,
        seed: seed + d,
        provenance: {
          quantization: precision,
          diffusionStepsScheduled: numSteps,
          diffusionStepsRun: numSteps,
        },
      };
      items.push(artifact);

      if (d === 0 && Array.isArray(sampled.trajectory)) {
        const frames: Float32Array[] = [];
        const steps: number[] = [];
        const nRes = parsed.nResidues;
        for (const rec of sampled.trajectory) {
          const raw = (rec.px0 ?? rec.coords) as Float32Array | undefined;
          if (!raw || raw.length < nRes * 9) continue;
          const stride = raw.length >= nRes * 12 ? 12 : 9;
          const ca = new Float32Array(nRes * 3);
          for (let i = 0; i < nRes; i++) {
            ca[i * 3] = raw[i * stride + 3];
            ca[i * 3 + 1] = raw[i * stride + 4];
            ca[i * 3 + 2] = raw[i * stride + 5];
          }
          frames.push(ca);
          steps.push(Number(rec.t ?? frames.length));
        }
        primaryTrajFrames = frames;
        primaryTrajSteps = steps;
      }
    }

    const primary = items[0];
    const batchHash = fnv1a(`${items.map((it) => it.id).join('|')}:${items.length}`);
    const repSelection = {
      selectedKey: primary.designKey ?? primary.id,
      criterion: 'lowest-creation-ordinal',
      totalMembers: items.length,
      selectedIndex: 0,
      scoreValue: null,
    };
    const structureOut: StructureArtifact =
      items.length > 1
        ? {
            ...primary,
            id: `rfd-batch-${items.length}-${batchHash}`,
            label: `RFdiffusion (${items.length} designs · L=${primary.length})`,
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `rfd-col-${batchHash}`,
              familyKey: `rfdiffusion-${batchHash}`,
              collectionKind: 'independent-candidates',
              requestedCount: numDesigns,
              complete: true,
              items,
              keys: items.map((it) => it.designKey ?? it.id),
              representativeSelection: repSelection,
            },
          }
        : primary;

    const trajFrames = (primaryTrajFrames.length > 0 ? primaryTrajFrames : [primary.caCoords.slice()]).map(
      (caCoords, idx) => ({
        key: `frame-${idx}`,
        frameIndex: idx,
        caCoords,
        timeStep: primaryTrajSteps[idx] ?? idx,
      }),
    );

    const trajectoryOut: TrajectoryFramesArtifact = {
      kind: 'TrajectoryFrames',
      id: `rfd-traj-${runHash}`,
      label: `RFdiffusion trajectory (${trajFrames.length} steps)`,
      parentStructureId: primary.id,
      sourceModelId: 'rfdiffusion',
      frameCount: trajFrames.length,
      ordering: 'ordered-frames',
      collectionKind: 'ordered-frames',
      frames: trajFrames,
    };

    const transferables = collectOwnedTransferables([
      ...items.flatMap((it) => [it.caCoords, it.backbone4Coords, it.observedBackboneMask]),
      ...trajFrames.map((f) => f.caCoords),
    ]);

    return {
      outputs: { structure: structureOut, trajectory: trajectoryOut },
      transferables,
    };
  },

  async dispose() {
    disposeEngine();
  },
});

export {};
