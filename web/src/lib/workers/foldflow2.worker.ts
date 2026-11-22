/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/foldflow2-webgpu (CC BY-NC 4.0 · DreamFold/FoldFlow ReFT + ESM-2 650M)
// Runs FoldFlow 2 SE(3) flow matching (unconditional or sequence-conditioned) natively on WebGPU.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
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
let pdbMod: any = null;
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

  if (!gpuMod || !pdbMod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading FoldFlow 2 WebGPU modules...',
    });
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

    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.05,
      detail: `Fetching FoldFlow 2 + ESM-2 650M (${precision.toUpperCase()}) weights...`,
    });

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
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.45, 0.05 + frac * 0.4),
            detail:
              total > 0
                ? `FoldFlow 2 (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
                : `Loading FoldFlow 2 + ESM-2 (${precision.toUpperCase()})...`,
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
    const seqInput = msg.inputs?.sequences as SequenceSet | undefined;
    const condSeq = seqInput?.candidates?.[0]?.sequence?.trim() || '';
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
              ? `FoldFlow 2 design ${d + 1}/${numDesigns} (${steps} steps, ${length} aa)...`
              : `FoldFlow 2 SE(3) + ESM-2 (${steps} steps, ${length} aa)...`,
        });

        const res = await eng.sample({
          length,
          steps,
          seed: designSeed,
          selfConditioning,
          rotationScaling,
          ...(esmTokens ? { esmTokens } : {}),
          signal: abortCtrl.signal,
          onProgress: (info: { step?: number; steps?: number; phase?: string }) => {
            const step = info.step ?? 0;
            const totalSteps = info.steps ?? steps;
            const frac = totalSteps > 0 ? Math.min(1, step / totalSteps) : 0.5;
            self.postMessage({
              id: msg.id,
              type: 'progress',
              phase: 'sample',
              progress: progressBase + frac * progressSpan,
              detail:
                info.phase === 'esm'
                  ? `FoldFlow 2 · running frozen ESM-2 650M encoder...`
                  : numDesigns > 1
                    ? `FoldFlow 2 ${d + 1}/${numDesigns} · step ${step}/${totalSteps}`
                    : `FoldFlow 2 · step ${step}/${totalSteps}`,
            });
          },
        });

        const pdbText = String(pdbMod.toPDB(res));
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
        const uniqueChains = [...new Set(parsed.chainIds)];
        items.push({
          kind: 'StructureArtifact',
          id: `foldflow2-${length}aa-s${designSeed}-${runHash}`,
          label:
            numDesigns > 1
              ? `FoldFlow 2 #${d + 1} (${length} aa)`
              : `FoldFlow 2 (${length} aa, ${steps} steps)`,
          sourceModelId: `foldflow2/${precision}`,
          pdbText,
          length: parsed.nResidues || length,
          chainIds: uniqueChains.length > 0 ? uniqueChains : ['A'],
          sequence: condSeq || parsed.sequence,
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
              id: `foldflow2-col-${primary.id}`,
              familyKey: 'foldflow2',
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
