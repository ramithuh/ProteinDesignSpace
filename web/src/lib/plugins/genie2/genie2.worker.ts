/// <reference lib="webworker" />
// Layer 3 Plugin Worker — genie2 (ramithuh/genie2-webgpu · Apache-2.0 · aqlaboratory/genie2 epoch-40)
// Runs Genie 2 Cα diffusion (unconditional and optional soft motif scaffolding) natively on WebGPU.
// Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { buildGenie2OptionalMotifs } from '$lib/protein-ops/motif-placement';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/genie2-webgpu/resolve/main/base';

type Precision = 'int8' | 'fp16';

function normalizePrecision(raw: unknown): Precision {
  if (raw == null || String(raw).trim() === '') return 'fp16';
  const val = String(raw).trim().toLowerCase();
  if (val === 'int8' || val === 'fp16') return val;
  throw new Error(`Unsupported Genie 2 precision "${String(raw)}" (supported: fp16, int8).`);
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
    ctx.progress('init', 0.02, 'Loading Genie 2 WebGPU modules...');
    const base = `${self.location.origin}/vendor/genie2`;
    [gpuMod, sharedMod] = await Promise.all([
      import(/* @vite-ignore */ `${base}/webgpu/index.js`),
      import(/* @vite-ignore */ `${base}/shared/index.js`),
    ]);
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

  const envBaseUrl = (import.meta.env.VITE_GENIE2_WEIGHTS_URL as string | undefined)?.trim();
  const rootUrl = (envBaseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
  const baseUrl = `${rootUrl}/${precision}`;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.05, `Fetching Genie 2 (${precision.toUpperCase()}) weights...`);

    try {
      return await gpuMod.createGenie2Engine({
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
              ? `Genie 2 (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
              : `Loading Genie 2 (${precision.toUpperCase()})...`,
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
    const template = ctx.inputs?.template as StructureArtifact | undefined;
    const length = Math.max(20, Math.min(256, Math.round(Number(p.length ?? 80))));
    const steps = Math.max(10, Math.min(1000, Math.round(Number(p.steps ?? 1000))));
    const scale = Math.max(0.1, Math.min(1.0, Number(p.scale ?? 0.6)));
    const seed = Math.round(Number(p.seed ?? 42));
    const precision = normalizePrecision(p.precision);
    const motifRange = String(p.motifRange ?? '').trim();
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));

    const motifMeta = buildGenie2OptionalMotifs(template, motifRange, length);
    const motifs = motifMeta?.motifs;

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
            ? `Genie 2 design ${d + 1}/${numDesigns} (${steps} steps, ${length} aa)...`
            : `Genie 2 Cα diffusion (${steps} steps, ${length} aa)...`,
        );

        let completedSteps = 0;
        const res = await eng.sample({
          length,
          steps,
          scale,
          seed: designSeed,
          ...(motifs ? { motifs } : {}),
          readStepCoordinates: false,
          signal: abortCtrl.signal,
          onStep: () => {
            if (abortRequested || ctx.signal.aborted) return;
            completedSteps++;
            const frac = Math.min(1, completedSteps / steps);
            if (completedSteps === 1 || completedSteps === steps || completedSteps % 5 === 0) {
              ctx.progress(
                'sample',
                progressBase + frac * progressSpan,
                numDesigns > 1
                  ? `Genie 2 ${d + 1}/${numDesigns} · step ${completedSteps}/${steps}`
                  : `Genie 2 · step ${completedSteps}/${steps}`,
              );
            }
          },
        });

        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }

        const pdbText = String(sharedMod.writePDB(res));
        const parsed = parseStructureText(pdbText);
        if (parsed.nResidues !== length) {
          throw new Error(
            `Genie 2 returned ${parsed.nResidues} valid Cα residue(s) instead of requested ${length} aa (possible non-finite coordinates).`,
          );
        }
        for (let i = 0; i < parsed.caCoords.length; i++) {
          if (!Number.isFinite(parsed.caCoords[i])) {
            throw new Error(`Genie 2 generated non-finite Cα coordinates for design #${d + 1}.`);
          }
        }
        const geom = computeGeometryStats(parsed.caCoords);
        const runHash = fnv1a(
          JSON.stringify({
            length,
            steps,
            scale,
            seed: designSeed,
            precision,
            motifRange,
            templateId: motifMeta ? (template?.id ?? null) : null,
          }),
        );
        const resolvedLen = parsed.nResidues;
        items.push({
          kind: 'StructureArtifact',
          id: `genie2-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `Genie 2 #${d + 1} (${length} aa)`
              : `Genie 2 (${length} aa, ${steps} steps)`,
          sourceModelId: `genie2/${precision}`,
          ...(motifMeta && template?.id ? { parentStructureId: template.id } : {}),
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
          ...(motifMeta
            ? {
                fixedResidueIndices: motifMeta.fixedResidueIndices,
                motifIndices: motifMeta.motifIndices,
                nativeMotifCoords: motifMeta.nativeMotifCoords,
              }
            : {}),
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
            id: `genie2-batch-${items.length}-${batchHash}`,
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `genie2-col-${batchHash}`,
              familyKey: `genie2-${batchHash}`,
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
