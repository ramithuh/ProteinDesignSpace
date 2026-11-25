/// <reference lib="webworker" />
// Layer 3 Plugin Worker — frameflow (ramithuh/frameflow-webgpu · MIT · microsoft/protein-frame-flow)
// Runs FrameFlow SE(3) flow matching backbone generation natively on WebGPU (5-atom N/CA/C/O/CB backbone).
// Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL =
  'https://huggingface.co/ramithuh/frameflow-webgpu/resolve/main/published';

type Precision = 'int8' | 'fp16';
type Schedule = 'exp' | 'linear';

function makeMulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalizePrecision(raw: unknown): Precision {
  if (raw == null || String(raw).trim() === '') return 'fp16';
  const val = String(raw).trim().toLowerCase();
  if (val === 'int8' || val === 'fp16') return val;
  throw new Error(`Unsupported FrameFlow precision "${String(raw)}" (supported: fp16, int8).`);
}

function normalizeSchedule(raw: unknown): Schedule {
  return String(raw ?? 'exp').trim().toLowerCase() === 'linear' ? 'linear' : 'exp';
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let gpuMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let ownedDevice: GPUDevice | null = null;
let loadedPrecision: Precision | null = null;
let activeAbort: AbortController | null = null;
let abortRequested = false;

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

  if (!gpuMod) {
    ctx.progress('init', 0.02, 'Loading FrameFlow WebGPU modules...');
    const base = `${self.location.origin}/vendor/frameflow`;
    gpuMod = await import(/* @vite-ignore */ `${base}/webgpu/index.js`);
  }

  const { adapter } = await requestResilientWebGpuAdapter({ allowSoftwareFallback: true });
  const wantF16 = adapter.features.has('shader-f16');
  const device = await adapter.requestDevice({
    requiredFeatures: wantF16 ? ['shader-f16'] : [],
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

  const envBaseUrl = (import.meta.env.VITE_FRAMEFLOW_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.05, `Fetching FrameFlow (${precision.toUpperCase()}) weights...`);

    try {
      return await gpuMod.createFrameFlowEngine({
        baseUrl,
        precision,
        device,
        useF16: device.features.has('shader-f16'),
        allowSoftwareFallback: true,
        cache: 'none',
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { loaded?: number; total?: number }) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac = total > 0 ? Math.min(1, loaded / total) : 0.4;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          ctx.progress(
            'weights',
            Math.min(0.4, 0.05 + frac * 0.35),
            total > 0
              ? `FrameFlow (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
              : `Loading FrameFlow (${precision.toUpperCase()})...`,
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
    abortRequested = false;
    const p = ctx.params ?? {};
    const length = Math.max(20, Math.min(300, Math.round(Number(p.length ?? 80))));
    const steps = Math.max(10, Math.min(200, Math.round(Number(p.steps ?? 100))));
    const schedule = normalizeSchedule(p.schedule);
    const expRate = Math.max(1, Math.min(30, Number(p.expRate ?? 10)));
    const selfCondition = p.selfCondition !== false;
    const seed = Math.round(Number(p.seed ?? 42));
    const precision = normalizePrecision(p.precision);
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));

    if (schedule === 'exp') {
      const dt = 0.99 / Math.max(1, steps - 1);
      if (dt * expRate > 1.0) {
        throw new Error(
          `FrameFlow exponential rotation schedule is unstable for steps=${steps} and expRate=${expRate} (dt*expRate=${(dt * expRate).toFixed(2)} > 1.0). Increase steps or lower expRate.`,
        );
      }
    }

    if (abortRequested || ctx.signal.aborted) {
      throw new Error('Run aborted');
    }

    const eng = await ensureEngine(ctx, precision);
    if (abortRequested || ctx.signal.aborted) {
      throw new Error('Run aborted');
    }

    const abortCtrl = new AbortController();
    activeAbort = abortCtrl;
    const onCtxAbort = () => abortCtrl.abort();
    if (ctx.signal.aborted || abortRequested) {
      abortCtrl.abort();
    } else {
      ctx.signal.addEventListener('abort', onCtxAbort, { once: true });
    }

    const items: StructureArtifact[] = [];
    try {
      for (let d = 0; d < numDesigns; d++) {
        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }

        const designSeed = seed + d;
        const designKey = `d${d + 1}`;
        const progressBase = 0.4 + (d / numDesigns) * 0.56;
        const progressSpan = 0.56 / numDesigns;

        ctx.progress(
          'sample',
          progressBase,
          numDesigns > 1
            ? `FrameFlow design ${d + 1}/${numDesigns} (${steps} steps, ${length} aa)...`
            : `FrameFlow SE(3) flow matching (${steps} steps, ${length} aa)...`,
        );

        const res = await eng.sample({
          length,
          steps,
          schedule,
          expRate,
          selfCondition,
          random: makeMulberry32(designSeed),
          keepTrajectory: false,
          signal: abortCtrl.signal,
          onProgress: (info: { step?: number; steps?: number; total?: number }) => {
            if (abortRequested || ctx.signal.aborted) return;
            const step = info.step ?? 0;
            const totalSteps = info.total ?? info.steps ?? steps;
            const frac = totalSteps > 0 ? Math.min(1, step / totalSteps) : 0.5;
            ctx.progress(
              'sample',
              progressBase + frac * progressSpan,
              numDesigns > 1
                ? `FrameFlow ${d + 1}/${numDesigns} · step ${step}/${totalSteps}`
                : `FrameFlow · step ${step}/${totalSteps}`,
            );
          },
        });

        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }

        const pdbText = String(res.pdb ?? '');
        const parsed = parseStructureText(pdbText);
        if (parsed.nResidues !== length) {
          throw new Error(
            `FrameFlow returned ${parsed.nResidues} valid residue(s) instead of requested ${length} aa (possible non-finite coordinates).`,
          );
        }
        for (let i = 0; i < parsed.caCoords.length; i++) {
          if (!Number.isFinite(parsed.caCoords[i])) {
            throw new Error(`FrameFlow generated non-finite Cα coordinates for design #${d + 1}.`);
          }
        }
        const geom = computeGeometryStats(parsed.caCoords);
        const runHash = fnv1a(
          JSON.stringify({ length, steps, schedule, expRate, selfCondition, seed: designSeed, precision }),
        );
        const resolvedLen = parsed.nResidues;
        items.push({
          kind: 'StructureArtifact',
          id: `frameflow-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `FrameFlow #${d + 1} (${length} aa)`
              : `FrameFlow (${length} aa, ${steps} steps)`,
          sourceModelId: `frameflow/${precision}`,
          pdbText,
          length: resolvedLen,
          chainIds:
            parsed.chainIds.length === resolvedLen
              ? parsed.chainIds
              : Array<string>(resolvedLen).fill('A'),
          chainLengths: parsed.diagnostics.chainLengths,
          sequence: parsed.sequence,
          caCoords: parsed.caCoords,
          backbone4Coords: parsed.backbone4Coords,
          atomGranularity: 'backbone4',
          bfactorSemantics: 'none',
          atomPresence: parsed.atomPresence,
          diagnostics: parsed.diagnostics,
          geometry: geom,
          designKey,
          designIndex: d,
        });
      }
    } finally {
      ctx.signal.removeEventListener('abort', onCtxAbort);
      if (activeAbort === abortCtrl) activeAbort = null;
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
            id: `frameflow-batch-${items.length}-${batchHash}`,
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `frameflow-col-${batchHash}`,
              familyKey: `frameflow-${batchHash}`,
              collectionKind: 'independent-candidates',
              requestedCount: numDesigns,
              complete: true,
              items,
              keys: items.map((it) => it.designKey ?? it.id),
              representativeSelection: repSelection,
            },
          }
        : primary;

    return { outputs: { structure: structureOut } };
  },

  abort() {
    abortRequested = true;
    activeAbort?.abort();
  },

  async dispose() {
    abortRequested = true;
    activeAbort?.abort();
    activeAbort = null;
    disposeEngine();
  },
});

export {};
