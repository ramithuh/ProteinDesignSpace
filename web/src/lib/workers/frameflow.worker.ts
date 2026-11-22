/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/frameflow-webgpu (MIT · microsoft/protein-frame-flow)
// Runs FrameFlow SE(3) flow matching backbone generation natively on WebGPU (5-atom N/CA/C/O/CB backbone).

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL =
  'https://huggingface.co/ramithuh/frameflow-webgpu/resolve/main/published';

type Precision = 'int8' | 'fp16';
type Schedule = 'exp' | 'linear';

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

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
  return String(raw ?? 'fp16').trim().toLowerCase() === 'int8' ? 'int8' : 'fp16';
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

async function ensureEngine(id: number, precision: Precision) {
  if (engine && loadedPrecision === precision) return engine;
  disposeEngine();

  if (!gpuMod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading FrameFlow WebGPU modules...',
    });
    const base = `${self.location.origin}/vendor/frameflow`;
    gpuMod = await import(/* @vite-ignore */ `${base}/webgpu/index.js`);
  }

  const { adapter } = await requestResilientWebGpuAdapter({ allowSoftwareFallback: true });
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  ownedDevice = device;

  const envBaseUrl = (import.meta.env.VITE_FRAMEFLOW_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.05,
      detail: `Fetching FrameFlow (${precision.toUpperCase()}) weights...`,
    });

    try {
      return await gpuMod.createFrameFlowEngine({
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
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.4, 0.05 + frac * 0.35),
            detail:
              total > 0
                ? `FrameFlow (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
                : `Loading FrameFlow (${precision.toUpperCase()})...`,
          });
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

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'abort') {
      activeAbort?.abort();
      return;
    }
    if (msg.type === 'dispose') {
      activeAbort?.abort();
      disposeEngine();
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }
    if (msg.type === 'run') {
      const t0 = performance.now();
    const p = msg.params ?? {};
    const length = Math.max(20, Math.min(300, Math.round(Number(p.length ?? 80))));
    const steps = Math.max(10, Math.min(200, Math.round(Number(p.steps ?? 100))));
    const schedule = normalizeSchedule(p.schedule);
    const expRate = Math.max(1, Math.min(30, Number(p.expRate ?? 10)));
    const selfCondition = p.selfCondition !== false;
    const seed = Math.round(Number(p.seed ?? 42));
    const precision = normalizePrecision(p.precision);
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));

    const eng = await ensureEngine(msg.id, precision);
    const abortCtrl = new AbortController();
    activeAbort = abortCtrl;

    const items: StructureArtifact[] = [];
    try {
      for (let d = 0; d < numDesigns; d++) {
        const designSeed = seed + d;
        const designKey = `d${d + 1}`;
        const progressBase = 0.4 + (d / numDesigns) * 0.56;
        const progressSpan = 0.56 / numDesigns;

        self.postMessage({
          id: msg.id,
          type: 'progress',
          phase: 'sample',
          progress: progressBase,
          detail:
            numDesigns > 1
              ? `FrameFlow design ${d + 1}/${numDesigns} (${steps} steps, ${length} aa)...`
              : `FrameFlow SE(3) flow matching (${steps} steps, ${length} aa)...`,
        });

        const res = await eng.sample({
          length,
          steps,
          schedule,
          expRate,
          selfCondition,
          random: makeMulberry32(designSeed),
          keepTrajectory: false,
          signal: abortCtrl.signal,
          onProgress: (info: { step?: number; steps?: number }) => {
            const step = info.step ?? 0;
            const totalSteps = info.steps ?? steps;
            const frac = totalSteps > 0 ? Math.min(1, step / totalSteps) : 0.5;
            self.postMessage({
              id: msg.id,
              type: 'progress',
              phase: 'sample',
              progress: progressBase + frac * progressSpan,
              detail:
                numDesigns > 1
                  ? `FrameFlow ${d + 1}/${numDesigns} · step ${step}/${totalSteps}`
                  : `FrameFlow · step ${step}/${totalSteps}`,
            });
          },
        });

        const pdbText = String(res.pdb ?? '');
        const parsed = parseStructureText(pdbText);
        const geom = computeGeometryStats(parsed.caCoords);
        const runHash = fnv1a(
          JSON.stringify({ length, steps, schedule, expRate, selfCondition, seed: designSeed, precision }),
        );
        const uniqueChains = [...new Set(parsed.chainIds)];
        items.push({
          kind: 'StructureArtifact',
          id: `frameflow-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `FrameFlow #${d + 1} (${length} aa)`
              : `FrameFlow (${length} aa, ${steps} steps)`,
          sourceModelId: `frameflow/${precision}`,
          pdbText,
          length: parsed.nResidues || length,
          chainIds: uniqueChains.length > 0 ? uniqueChains : ['A'],
          sequence: parsed.sequence,
          caCoords: parsed.caCoords,
          backbone4Coords: parsed.backbone4Coords,
          atomGranularity: 'backbone4',
          atomPresence: parsed.atomPresence,
          diagnostics: parsed.diagnostics,
          geometry: geom,
          designKey,
          designIndex: d,
        });
      }
    } finally {
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
              id: `frameflow-col-${primary.id}`,
              familyKey: 'frameflow',
              collectionKind: 'independent-candidates',
              requestedCount: numDesigns,
              complete: true,
              items,
              keys: items.map((it) => it.designKey ?? it.id),
              representativeSelection: repSelection,
            },
          }
        : primary;

    self.postMessage({
      id: msg.id,
      type: 'result',
      outputs: { structure: structureOut },
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
