/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/genie2-webgpu (Apache-2.0 · aqlaboratory/genie2 epoch-40)
// Runs Genie 2 Cα diffusion (unconditional and optional soft motif scaffolding) natively on WebGPU.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/genie2-webgpu/resolve/main/base';

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
  return String(raw ?? 'fp16').trim().toLowerCase() === 'int8' ? 'int8' : 'fp16';
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

  if (!gpuMod || !sharedMod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading Genie 2 WebGPU modules...',
    });
    const base = `${self.location.origin}/vendor/genie2`;
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

    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.05,
      detail: `Fetching Genie 2 (${precision.toUpperCase()}) weights...`,
    });

    try {
      return await gpuMod.createGenie2Engine({
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
                ? `Genie 2 (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
                : `Loading Genie 2 (${precision.toUpperCase()})...`,
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

function buildOptionalMotifs(
  template: StructureArtifact | undefined,
  motifRange: string,
  targetLength: number,
) {
  if (!template?.pdbText || !motifRange.trim()) return undefined;
  const parsed = parseStructureText(template.pdbText);
  if (parsed.nResidues === 0) return undefined;
  // Parse simple "A10-25" or "10-25" range against the template residues
  const match = motifRange.trim().match(/^(?:([A-Za-z]):?)?(\d+)(?:-(\d+))?$/);
  if (!match) return undefined;
  const chainFilter = match[1]?.toUpperCase();
  const startRes = Number(match[2]);
  const endRes = match[3] ? Number(match[3]) : startRes;
  const picked: number[] = [];
  for (let i = 0; i < parsed.nResidues; i++) {
    if (chainFilter && parsed.chainIds[i]?.toUpperCase() !== chainFilter) continue;
    const r = parsed.resNums[i];
    if (r >= startRes && r <= endRes) picked.push(i);
  }
  if (picked.length < 3 || picked.length >= targetLength) return undefined;
  const coords = new Float32Array(picked.length * 3);
  let seq = '';
  for (let k = 0; k < picked.length; k++) {
    const idx = picked[k];
    coords[k * 3] = parsed.caCoords[idx * 3];
    coords[k * 3 + 1] = parsed.caCoords[idx * 3 + 1];
    coords[k * 3 + 2] = parsed.caCoords[idx * 3 + 2];
    seq += parsed.sequence[idx] || 'A';
  }
  const startOffset = Math.max(0, Math.floor((targetLength - picked.length) / 2));
  return [{ start: startOffset, sequence: seq, coordinates: coords, group: 1 }];
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
    const template = msg.inputs?.template as StructureArtifact | undefined;
    const length = Math.max(20, Math.min(256, Math.round(Number(p.length ?? 80))));
    const steps = Math.max(10, Math.min(1000, Math.round(Number(p.steps ?? 200))));
    const scale = Math.max(0.1, Math.min(1.5, Number(p.scale ?? 0.6)));
    const seed = Math.round(Number(p.seed ?? 42));
    const precision = normalizePrecision(p.precision);
    const motifRange = String(p.motifRange ?? '').trim();
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));

    const motifs = buildOptionalMotifs(template, motifRange, length);
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
              ? `Genie 2 design ${d + 1}/${numDesigns} (${steps} steps, ${length} aa)...`
              : `Genie 2 Cα diffusion (${steps} steps, ${length} aa)...`,
        });

        let completedSteps = 0;
        const res = await eng.sample({
          length,
          steps,
          scale,
          seed: designSeed,
          ...(motifs ? { motifs } : {}),
          signal: abortCtrl.signal,
          onStep: () => {
            completedSteps++;
            const frac = Math.min(1, completedSteps / steps);
            if (completedSteps === 1 || completedSteps === steps || completedSteps % 5 === 0) {
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'sample',
                progress: progressBase + frac * progressSpan,
                detail:
                  numDesigns > 1
                    ? `Genie 2 ${d + 1}/${numDesigns} · step ${completedSteps}/${steps}`
                    : `Genie 2 · step ${completedSteps}/${steps}`,
              });
            }
          },
        });

        const pdbText = String(sharedMod.writePDB(res));
        const parsed = parseStructureText(pdbText);
        const geom = computeGeometryStats(parsed.caCoords);
        const runHash = fnv1a(
          JSON.stringify({ length, steps, scale, seed: designSeed, precision, motifRange }),
        );
        const uniqueChains = [...new Set(parsed.chainIds)];
        items.push({
          kind: 'StructureArtifact',
          id: `genie2-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `Genie 2 #${d + 1} (${length} aa)`
              : `Genie 2 (${length} aa, ${steps} steps)`,
          sourceModelId: `genie2/${precision}`,
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
              id: `genie2-col-${primary.id}`,
              familyKey: 'genie2',
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
