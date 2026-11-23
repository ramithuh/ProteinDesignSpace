/// <reference lib="webworker" />
// Layer 3 Plugin Worker — proteina (ramithuh/proteina-webgpu · NVIDIA Source Code License · Proteina v1.2 200M no-tri)
// Runs Proteina v1.2 200M Cα flow-matching backbone generation natively on WebGPU.
// Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL =
  'https://huggingface.co/ramithuh/proteina-webgpu/resolve/main/v1.2_200m_notri';

type Precision = 'int8' | 'fp16';
type SampleMode = 'sc' | 'vf';

function normalizePrecision(raw: unknown): Precision {
  return String(raw ?? 'int8').trim().toLowerCase() === 'fp16' ? 'fp16' : 'int8';
}

function normalizeMode(raw: unknown): SampleMode {
  const v = String(raw ?? 'sc').trim().toLowerCase();
  return v === 'vf' || v === 'ode' ? 'vf' : 'sc';
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let gpuMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let sharedMod: any = null;
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

  if (!gpuMod || !sharedMod) {
    ctx.progress('init', 0.02, 'Loading Proteina v1.2 WebGPU modules...');
    const base = `${self.location.origin}/vendor/proteina`;
    [gpuMod, sharedMod] = await Promise.all([
      import(/* @vite-ignore */ `${base}/webgpu/index.js`),
      import(/* @vite-ignore */ `${base}/shared/index.js`),
    ]);
  }

  const { adapter } = await requestResilientWebGpuAdapter({ allowSoftwareFallback: true });
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  ownedDevice = device;

  const envBaseUrl = (import.meta.env.VITE_PROTEINA_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.05, `Fetching Proteina v1.2 200M (${precision.toUpperCase()}) weights...`);

    try {
      return await gpuMod.createProteinaEngine({
        baseUrl,
        precision,
        device,
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
            Math.min(0.45, 0.05 + frac * 0.4),
            total > 0
              ? `Proteina 200M (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
              : `Loading Proteina 200M (${precision.toUpperCase()})...`,
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
    const steps = Math.max(10, Math.min(400, Math.round(Number(p.steps ?? 400))));
    const mode = normalizeMode(p.mode);
    const noiseScale = Math.max(0.05, Math.min(1.0, Number(p.noiseScale ?? 0.45)));
    const selfCondition = p.selfCondition !== false;
    const seed = Math.round(Number(p.seed ?? 42));
    const precision = normalizePrecision(p.precision);
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));

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
        const progressBase = 0.45 + (d / numDesigns) * 0.52;
        const progressSpan = 0.52 / numDesigns;

        ctx.progress(
          'sample',
          progressBase,
          numDesigns > 1
            ? `Proteina 200M design ${d + 1}/${numDesigns} (${steps} steps, ${length} aa)...`
            : `Proteina 200M Cα flow matching (${steps} steps, ${length} aa)...`,
        );

        const res = await eng.sample({
          length,
          steps,
          seed: designSeed,
          schedule: 'log',
          schedulePower: 2,
          mode,
          noiseScale,
          selfCondition,
          signal: abortCtrl.signal,
          onProgress: (info: { step?: number; steps?: number }) => {
            if (abortRequested || ctx.signal.aborted) return;
            const step = info.step ?? 0;
            const totalSteps = info.steps ?? steps;
            const frac = totalSteps > 0 ? Math.min(1, step / totalSteps) : 0.5;
            ctx.progress(
              'sample',
              progressBase + frac * progressSpan,
              numDesigns > 1
                ? `Proteina 200M ${d + 1}/${numDesigns} · step ${step}/${totalSteps}`
                : `Proteina 200M · step ${step}/${totalSteps}`,
            );
          },
        });

        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }

        const pdbText = String(sharedMod.toPDB(res.coordinates));
        const parsed = parseStructureText(pdbText);
        const geom = computeGeometryStats(parsed.caCoords);
        const runHash = fnv1a(
          JSON.stringify({ length, steps, mode, noiseScale, selfCondition, seed: designSeed, precision }),
        );
        const resolvedLen = parsed.nResidues || length;
        items.push({
          kind: 'StructureArtifact',
          id: `proteina-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `Proteina #${d + 1} (${length} aa)`
              : `Proteina v1.2 (${length} aa, ${steps} steps)`,
          sourceModelId: `proteina-v1.2-200m/${precision}`,
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
          atomGranularity: 'ca-only',
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
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `proteina-col-${primary.id}`,
              familyKey: 'proteina',
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
