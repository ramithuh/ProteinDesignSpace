/// <reference lib="webworker" />
// Layer 3 Plugin Worker — sid-protein (ramithuh/sid-protein-webgpu · LY-Xie/SiD_Protein · Distilled 16-step Proteina)
// Runs ultra-fast 16-step distilled Cα backbone generation natively on WebGPU.
// Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL =
  'https://huggingface.co/ramithuh/sid-protein-webgpu/resolve/main/distilled_16step';

type Precision = 'int8' | 'fp16';

function normalizePrecision(raw: unknown): Precision {
  if (raw == null || String(raw).trim() === '') return 'int8';
  const val = String(raw).trim().toLowerCase();
  if (val === 'fp16' || val === 'int8') return val;
  throw new Error(`Unsupported SiD-Protein precision "${String(raw)}" (supported: int8, fp16).`);
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
    ctx.progress('init', 0.02, 'Loading SiD-Protein WebGPU modules...');
    const base = `${self.location.origin}/vendor/sid-protein`;
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

  const envBaseUrl = (import.meta.env.VITE_SIDPROTEIN_WEIGHTS_URL as string | undefined)?.trim();
  const rootUrl = (envBaseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
  const baseUrl = `${rootUrl}/${precision}`;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.05, `Fetching SiD-Protein 16-step (${precision.toUpperCase()}) weights...`);

    try {
      return await gpuMod.createSidProteinEngine({
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
            Math.min(0.45, 0.05 + frac * 0.4),
            total > 0
              ? `SiD-Protein (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
              : `Loading SiD-Protein (${precision.toUpperCase()})...`,
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
    const steps = Math.max(1, Math.min(32, Math.round(Number(p.steps ?? 16))));
    const noiseScale = Math.max(0.05, Math.min(1.0, Number(p.noiseScale ?? 0.45)));
    const seed = Math.round(Number(p.seed ?? 5));
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
            ? `SiD-Protein design ${d + 1}/${numDesigns} (${steps} distilled steps, ${length} aa)...`
            : `SiD-Protein distilled Cα sampler (${steps} steps, ${length} aa)...`,
        );

        const res = await eng.generate({
          length,
          steps,
          noiseScale,
          seed: designSeed,
          signal: abortCtrl.signal,
          onStepProgress: ({ step, steps: totalSteps }: { step: number; steps: number }) => {
            if (abortRequested || ctx.signal.aborted) return;
            const frac = Math.min(1, step / Math.max(1, totalSteps));
            ctx.progress(
              'sample',
              progressBase + frac * progressSpan,
              numDesigns > 1
                ? `SiD-Protein ${d + 1}/${numDesigns} · step ${step}/${totalSteps}`
                : `SiD-Protein · step ${step}/${totalSteps}`,
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
            `SiD-Protein returned ${parsed.nResidues} valid residue(s) instead of requested ${length} aa.`,
          );
        }
        const geom = computeGeometryStats(parsed.caCoords);
        const runHash = fnv1a(
          JSON.stringify({
            length,
            steps,
            // 1-step distilled mode ignores noiseScale (onestepGamma=1)
            noiseScale: steps === 1 ? null : noiseScale,
            seed: designSeed,
            precision,
          }),
        );
        const resolvedLen = parsed.nResidues;
        items.push({
          kind: 'StructureArtifact',
          id: `sidprotein-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `SiD-Protein #${d + 1} (${length} aa)`
              : `SiD-Protein (${length} aa, ${steps} steps)`,
          sourceModelId: `sid-protein/${precision}`,
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
            id: `sidprotein-batch-${items.length}-${batchHash}`,
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `sidprotein-col-${batchHash}`,
              familyKey: `sid-protein-${batchHash}`,
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
