/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/sid-protein-webgpu (LY-Xie/SiD_Protein · Distilled 16-step Proteina)
// Runs ultra-fast 16-step distilled Cα backbone generation natively on WebGPU.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL =
  'https://huggingface.co/ramithuh/sid-protein-webgpu/resolve/main/distilled_16step';

type Precision = 'int8' | 'fp16';

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function normalizePrecision(raw: unknown): Precision {
  return String(raw ?? 'int8').trim().toLowerCase() === 'fp16' ? 'fp16' : 'int8';
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
      detail: 'Loading SiD-Protein WebGPU modules...',
    });
    const base = `${self.location.origin}/vendor/sid-protein`;
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

    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.05,
      detail: `Fetching SiD-Protein 16-step (${precision.toUpperCase()}) weights...`,
    });

    try {
      return await gpuMod.createSidProteinEngine({
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
            progress: Math.min(0.45, 0.05 + frac * 0.4),
            detail:
              total > 0
                ? `SiD-Protein (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
                : `Loading SiD-Protein (${precision.toUpperCase()})...`,
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
    const steps = Math.max(1, Math.min(32, Math.round(Number(p.steps ?? 16))));
    const noiseScale = Math.max(0.05, Math.min(1.0, Number(p.noiseScale ?? 0.45)));
    const seed = Math.round(Number(p.seed ?? 5));
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
        const progressBase = 0.45 + (d / numDesigns) * 0.52;
        const progressSpan = 0.52 / numDesigns;

        self.postMessage({
          id: msg.id,
          type: 'progress',
          phase: 'sample',
          progress: progressBase,
          detail:
            numDesigns > 1
              ? `SiD-Protein design ${d + 1}/${numDesigns} (${steps} distilled steps, ${length} aa)...`
              : `SiD-Protein distilled Cα sampler (${steps} steps, ${length} aa)...`,
        });

        let completedSteps = 0;
        const res = await eng.generate({
          length,
          steps,
          noiseScale,
          seed: designSeed,
          signal: abortCtrl.signal,
          onStep: () => {
            completedSteps++;
            const frac = Math.min(1, completedSteps / steps);
            self.postMessage({
              id: msg.id,
              type: 'progress',
              phase: 'sample',
              progress: progressBase + frac * progressSpan,
              detail:
                numDesigns > 1
                  ? `SiD-Protein ${d + 1}/${numDesigns} · step ${completedSteps}/${steps}`
                  : `SiD-Protein · step ${completedSteps}/${steps}`,
            });
          },
        });

        const pdbText = String(res.pdb ?? '');
        const parsed = parseStructureText(pdbText);
        const geom = computeGeometryStats(parsed.caCoords);
        const runHash = fnv1a(
          JSON.stringify({ length, steps, noiseScale, seed: designSeed, precision }),
        );
        const uniqueChains = [...new Set(parsed.chainIds)];
        items.push({
          kind: 'StructureArtifact',
          id: `sidprotein-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `SiD-Protein #${d + 1} (${length} aa)`
              : `SiD-Protein (${length} aa, ${steps} steps)`,
          sourceModelId: `sid-protein/${precision}`,
          pdbText,
          length: parsed.nResidues || length,
          chainIds: uniqueChains.length > 0 ? uniqueChains : ['A'],
          sequence: parsed.sequence,
          caCoords: parsed.caCoords,
          backbone4Coords: parsed.backbone4Coords,
          atomGranularity: 'ca-only',
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
              id: `sidprotein-col-${primary.id}`,
              familyKey: 'sid-protein',
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
