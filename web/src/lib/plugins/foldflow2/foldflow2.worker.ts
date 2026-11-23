/// <reference lib="webworker" />
// Layer 3 Plugin Worker — foldflow2 (ramithuh/foldflow2-webgpu · CC BY-NC 4.0 · DreamFold/FoldFlow ReFT + ESM-2 650M)
// Runs FoldFlow 2 SE(3) flow matching (unconditional or sequence-conditioned) natively on WebGPU.
// Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL =
  'https://huggingface.co/ramithuh/foldflow2-webgpu/resolve/main/ff2_reft';

type Precision = 'int8' | 'fp16';

// Standard ESM-2 vocabulary (BOS=0, PAD=1, EOS=2, UNK=3, L=4, A=5, G=6, V=7, S=8, E=9, R=10, T=11, I=12, D=13, P=14, K=15, Q=16, N=17, F=18, Y=19, M=20, H=21, W=22, C=23, X=24)
const ESM2_AA_TO_ID: Record<string, number> = {
  L: 4, A: 5, G: 6, V: 7, S: 8, E: 9, R: 10, T: 11, I: 12, D: 13,
  P: 14, K: 15, Q: 16, N: 17, F: 18, Y: 19, M: 20, H: 21, W: 22, C: 23, X: 24,
};

function encodeEsmTokens(seq: string): Int32Array {
  const clean = seq.toUpperCase().replace(/[^A-Z]/g, '');
  const tokens = new Int32Array(clean.length + 2);
  tokens[0] = 0; // BOS
  for (let i = 0; i < clean.length; i++) {
    tokens[i + 1] = ESM2_AA_TO_ID[clean[i]] ?? 24;
  }
  tokens[clean.length + 1] = 2; // EOS
  return tokens;
}

function normalizePrecision(raw: unknown): Precision {
  if (raw == null || String(raw).trim() === '') return 'int8';
  const val = String(raw).trim().toLowerCase();
  if (val === 'fp16' || val === 'int8') return val;
  throw new Error(`Unsupported FoldFlow 2 precision "${String(raw)}" (supported: int8, fp16).`);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let gpuMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pdbMod: any = null;
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

  if (!gpuMod || !pdbMod) {
    ctx.progress('init', 0.02, 'Loading FoldFlow 2 WebGPU modules...');
    const base = `${self.location.origin}/vendor/foldflow2`;
    [gpuMod, pdbMod] = await Promise.all([
      import(/* @vite-ignore */ `${base}/webgpu/index.js`),
      import(/* @vite-ignore */ `${base}/shared/pdb.js`),
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
  void device.lost.then(() => {
    if (ownedDevice === device) {
      engine = null;
      ownedDevice = null;
      loadedPrecision = null;
    }
  });

  const envBaseUrl = (import.meta.env.VITE_FOLDFLOW2_WEIGHTS_URL as string | undefined)?.trim();
  const rootUrl = (envBaseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
  const baseUrl = `${rootUrl}/${precision}`;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.05, `Fetching FoldFlow 2 + ESM-2 650M (${precision.toUpperCase()}) weights...`);

    try {
      return await gpuMod.createFoldFlow2Engine({
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
              ? `FoldFlow 2 (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
              : `Loading FoldFlow 2 + ESM-2 (${precision.toUpperCase()})...`,
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
    const seqInput = ctx.inputs?.sequences as SequenceSet | undefined;
    const rawCondSeq = seqInput?.candidates?.[0]?.sequence?.trim() || '';
    const condSeq = rawCondSeq.toUpperCase().replace(/[^A-Z]/g, '');
    const esmTokens = condSeq.length > 0 ? encodeEsmTokens(condSeq) : undefined;
    const length =
      condSeq.length > 0
        ? condSeq.length
        : Math.max(20, Math.min(256, Math.round(Number(p.length ?? 80))));
    const steps = Math.max(5, Math.min(100, Math.round(Number(p.steps ?? 50))));
    const rotationScaling = Math.max(1, Math.min(20, Number(p.rotationScaling ?? 10)));
    const selfConditioning = p.selfConditioning !== false;
    const seed = Math.round(Number(p.seed ?? 42));
    const precision = normalizePrecision(p.precision);
    const numDesigns = Math.max(1, Math.min(6, Math.round(Number(p.numDesigns ?? 1))));

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
            ? `FoldFlow 2 design ${d + 1}/${numDesigns} (${steps} steps, ${length} aa)...`
            : `FoldFlow 2 SE(3) + ESM-2 (${steps} steps, ${length} aa)...`,
        );

        const res = await eng.sample({
          length,
          steps,
          seed: designSeed,
          selfConditioning,
          rotationScaling,
          ...(esmTokens ? { esmTokens } : {}),
          signal: abortCtrl.signal,
          onProgress: (info: {
            step?: number;
            steps?: number;
            layer?: number;
            total?: number;
            phase?: string;
          }) => {
            if (abortRequested || ctx.signal.aborted) return;
            const isEsm = info.phase === 'esm';
            const step = isEsm ? (info.layer ?? 0) : (info.step ?? 0);
            const totalSteps = isEsm ? (info.total ?? 33) : (info.steps ?? info.total ?? steps);
            const frac = totalSteps > 0 ? Math.min(1, step / totalSteps) : 0.5;
            const scaledFrac = isEsm ? frac * 0.25 : 0.25 + frac * 0.75;
            ctx.progress(
              'sample',
              progressBase + scaledFrac * progressSpan,
              isEsm
                ? `FoldFlow 2 · ESM-2 650M layer ${step}/${totalSteps}`
                : numDesigns > 1
                  ? `FoldFlow 2 ${d + 1}/${numDesigns} · step ${step}/${totalSteps}`
                  : `FoldFlow 2 · step ${step}/${totalSteps}`,
            );
          },
        });

        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }

        const pdbText = String(pdbMod.toPDB(res, condSeq ? { sequence: condSeq } : undefined));
        const parsed = parseStructureText(pdbText);
        const geom = computeGeometryStats(parsed.caCoords);
        const runHash = fnv1a(
          JSON.stringify({
            length,
            steps,
            rotationScaling,
            selfConditioning,
            seed: designSeed,
            precision,
            condSeqHash: condSeq ? fnv1a(condSeq) : null,
          }),
        );
        const resolvedLen = parsed.nResidues || length;
        items.push({
          kind: 'StructureArtifact',
          id: `foldflow2-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `FoldFlow 2 #${d + 1} (${length} aa)`
              : `FoldFlow 2 (${length} aa, ${steps} steps)`,
          sourceModelId: `foldflow2/${precision}`,
          pdbText,
          length: resolvedLen,
          chainIds:
            parsed.chainIds.length === resolvedLen
              ? parsed.chainIds
              : Array<string>(resolvedLen).fill('A'),
          chainLengths: parsed.diagnostics.chainLengths,
          sequence: condSeq || parsed.sequence,
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
            id: `foldflow2-batch-${items.length}-${batchHash}`,
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `foldflow2-col-${batchHash}`,
              familyKey: `foldflow2-${batchHash}`,
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
