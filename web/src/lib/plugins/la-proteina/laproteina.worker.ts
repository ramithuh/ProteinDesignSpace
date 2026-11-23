/// <reference lib="webworker" />
// Layer 3 Plugin Worker — la-proteina (ramithuh/la-proteina-webgpu · Apache-2.0 code / NVIDIA Open Model License weights)
// Runs La-Proteina LD1 latent diffusion + AE1 all-atom & sequence decoder natively on WebGPU.
// Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/la-proteina-webgpu/resolve/main';

type Precision = 'int8' | 'fp16';
type SamplerMode = 'sde' | 'ode';

function normalizePrecision(raw: unknown): Precision {
  return String(raw ?? 'int8').trim().toLowerCase() === 'fp16' ? 'fp16' : 'int8';
}

function normalizeMode(raw: unknown): SamplerMode {
  return String(raw ?? 'sde').trim().toLowerCase() === 'ode' ? 'ode' : 'sde';
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let mod: any = null;
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

  if (!mod) {
    ctx.progress('init', 0.02, 'Loading La-Proteina WebGPU modules...');
    const base = `${self.location.origin}/vendor/la-proteina`;
    mod = await import(/* @vite-ignore */ `${base}/webgpu/index.js`);
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

  const envBaseUrl = (import.meta.env.VITE_LAPROTEINA_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.05, `Fetching La-Proteina LD1 + AE1 (${precision.toUpperCase()}) weights...`);

    try {
      return await mod.createLaProteinaEngine({
        baseUrl,
        precision,
        device,
        adapter,
        allowSoftwareFallback: true,
        cache: 'none',
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { loaded?: number; total?: number; shard?: string }) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac = total > 0 ? Math.min(1, loaded / total) : 0.4;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          ctx.progress(
            'weights',
            Math.min(0.45, 0.05 + frac * 0.4),
            total > 0
              ? `La-Proteina (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
              : `Loading La-Proteina LD1 + AE1 (${precision.toUpperCase()})...`,
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
    const seqCandidates: SequenceCandidate[] = [];

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
            ? `La-Proteina design ${d + 1}/${numDesigns}: LD1 sampling (${steps} steps, ${length} aa)...`
            : `La-Proteina: LD1 latent diffusion (${steps} steps, ${length} aa) + AE1 decode...`,
        );

        const res = await eng.sample({
          length,
          steps,
          seed: designSeed,
          mode,
          signal: abortCtrl.signal,
          onProgress: (info: {
            step?: number;
            steps?: number;
            layer?: number;
            total?: number;
            phase?: string;
          }) => {
            if (abortRequested || ctx.signal.aborted) return;
            const isAe1 = info.phase === 'ae1' || info.phase === 'decode';
            const isSampling = info.phase === 'sampling' || (!info.phase && info.step !== undefined);
            // Ignore inner LD1 layer progress ticks so the overall sampling progress bar never jumps backward
            if (!isAe1 && !isSampling) return;
            const step = isAe1 ? (info.layer ?? info.step ?? 0) : (info.step ?? 0);
            const totalSteps = isAe1
              ? (info.total ?? info.steps ?? 12)
              : (info.steps ?? info.total ?? steps);
            const frac = totalSteps > 0 ? Math.min(1, step / totalSteps) : 0.5;
            const scaledFrac = isAe1 ? 0.9 + frac * 0.1 : frac * 0.9;
            ctx.progress(
              'sample',
              progressBase + scaledFrac * progressSpan,
              isAe1
                ? `La-Proteina AE1 decode · layer ${step}/${totalSteps} (${length} aa)`
                : numDesigns > 1
                  ? `La-Proteina ${d + 1}/${numDesigns} · step ${step}/${totalSteps}`
                  : `La-Proteina LD1 · step ${step}/${totalSteps} (${mode.toUpperCase()})`,
            );
          },
        });

        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }

        const pdbText = String(res.pdb ?? '');
        if (!pdbText) throw new Error('La-Proteina did not return PDB coordinates.');
        const parsed = parseStructureText(pdbText);
        const geom = computeGeometryStats(parsed.caCoords);
        const sequence = String(res.sequence ?? parsed.sequence ?? '');

        const runHash = fnv1a(JSON.stringify({ length, steps, mode, seed: designSeed, precision }));
        const resolvedLen = parsed.nResidues || length;
        const artifact: StructureArtifact = {
          kind: 'StructureArtifact',
          id: `laproteina-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `La-Proteina #${d + 1} (${length} aa)`
              : `La-Proteina (${length} aa, ${steps} steps)`,
          sourceModelId: `la-proteina/${precision}`,
          pdbText,
          length: resolvedLen,
          chainIds:
            parsed.chainIds.length === resolvedLen
              ? parsed.chainIds
              : Array<string>(resolvedLen).fill('A'),
          chainLengths: parsed.diagnostics.chainLengths,
          sequence,
          caCoords: parsed.caCoords,
          backbone4Coords: parsed.backbone4Coords,
          atomGranularity: 'all-atom',
          bfactorSemantics: 'none',
          atomPresence: parsed.atomPresence,
          diagnostics: parsed.diagnostics,
          geometry: geom,
          designKey,
          designIndex: d,
        };
        items.push(artifact);

        if (sequence.length > 0) {
          seqCandidates.push({
            id: `laproteina-seq-${d + 1}`,
            key: `${designKey}-native`,
            sequenceKey: `${designKey}-native`,
            designKey,
            designIndex: d,
            sourceStructureId: artifact.id,
            sampleIndex: 0,
            sequence,
            score: null,
            globalScore: null,
            sourceModelId: `la-proteina/${precision}`,
            seed: designSeed,
          });
        }
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
            id: `laproteina-batch-${items.length}-${batchHash}`,
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `laproteina-col-${batchHash}`,
              familyKey: `la-proteina-${batchHash}`,
              collectionKind: 'independent-candidates',
              requestedCount: numDesigns,
              complete: true,
              items,
              keys: items.map((it) => it.designKey ?? it.id),
              representativeSelection: repSelection,
            },
          }
        : primary;

    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: items.length > 1 ? `seqset-laproteina-batch-${items.length}-${batchHash}` : `seqset-${primary.id}`,
      parentStructureId: primary.id,
      sourceStructureId: primary.id,
      sourceStructureIds: items.map((it) => it.id),
      sourceModelId: `la-proteina/${precision}`,
      candidates: seqCandidates,
    };

    return { outputs: { structure: structureOut, sequences: seqSet } };
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
