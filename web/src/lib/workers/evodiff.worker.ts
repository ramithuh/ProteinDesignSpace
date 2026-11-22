/// <reference lib="webworker" />
// Dedicated WebGPU (+ CPU fallback) worker wrapping ramithuh/evodiff-webgpu (MIT / CC-BY-4.0 · OA_DM_38M)
// Runs order-agnostic discrete protein sequence diffusion (de novo generation & masked sequence inpainting).

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { makeSequenceKey } from '$lib/engine/collections';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import type { SequenceCandidate, SequenceSet } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/evodiff-webgpu/resolve/main/oa_dm_38m';

type Precision = 'int8' | 'fp16';
type TaskMode = 'unconditional' | 'infill';

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

function normalizeTaskMode(raw: unknown): TaskMode {
  return String(raw ?? 'unconditional').trim().toLowerCase() === 'infill' ? 'infill' : 'unconditional';
}

function parseMaskPositions(raw: unknown, seqLen: number): number[] {
  const text = String(raw ?? '').trim();
  if (!text) return [];
  const positions = new Set<number>();
  for (const tok of text.split(/[,\s]+/).filter(Boolean)) {
    const m = /^(\d+)(?:-(\d+))?$/.exec(tok);
    if (!m) {
      throw new Error(`Invalid mask position "${tok}". Use 1-based numbers or ranges like "1-15, 45-60".`);
    }
    const lo = Number(m[1]);
    const hi = m[2] ? Number(m[2]) : lo;
    if (lo < 1 || hi < lo || hi > seqLen) {
      throw new Error(`Mask position "${tok}" is out of range for a ${seqLen}-aa sequence.`);
    }
    for (let pos = lo; pos <= hi; pos++) {
      positions.add(pos - 1);
    }
  }
  return [...positions].sort((a, b) => a - b);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let evoWebgpuMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let evoCpuMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let ownedDevice: GPUDevice | null = null;
let loadedPrecision: Precision | null = null;
let loadedBackend: 'webgpu' | 'cpu' | null = null;
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
  loadedBackend = null;
}

async function ensureEngine(id: number, precision: Precision) {
  if (engine && loadedPrecision === precision) {
    return { engine, backend: loadedBackend ?? 'webgpu' };
  }
  disposeEngine();

  const base = `${self.location.origin}/vendor/evodiff`;
  const envBaseUrl = (import.meta.env.VITE_EVODIFF_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const loadWithFactory = async (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    factory: (opts: any) => Promise<any>,
    extraOpts: Record<string, unknown>,
    retried: boolean,
  ) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    try {
      return await factory({
        baseUrl,
        precision,
        cache: 'none',
        ...extraOpts,
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { loaded?: number; total?: number }) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac = total > 0 ? loaded / total : 0.4;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.45, 0.06 + frac * 0.39),
            detail: `EvoDiff-38M weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
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
        return await loadWithFactory(factory, extraOpts, true);
      }
      throw err;
    }
  };

  // Try hardware WebGPU first; gracefully fall back to pure-JS CPU if WebGPU is unavailable or software-only.
  try {
    const { adapter, isSoftwareFallback } = await requestResilientWebGpuAdapter({
      allowSoftwareFallback: false,
    });
    if (!isSoftwareFallback) {
      if (!evoWebgpuMod) {
        self.postMessage({
          id,
          type: 'progress',
          phase: 'init',
          progress: 0.02,
          detail: 'Loading EvoDiff-38M WebGPU modules...',
        });
        evoWebgpuMod = await import(/* @vite-ignore */ `${base}/webgpu/index.js`);
      }
      const device = await adapter.requestDevice();
      ownedDevice = device;
      engine = await loadWithFactory(evoWebgpuMod.createEvoDiffEngine, { device }, false);
      loadedPrecision = precision;
      loadedBackend = 'webgpu';
      return { engine, backend: 'webgpu' as const };
    }
  } catch {
    disposeEngine();
  }

  self.postMessage({
    id,
    type: 'progress',
    phase: 'init',
    progress: 0.04,
    detail: 'Hardware WebGPU unavailable — falling back to EvoDiff-38M pure-JS CPU...',
  });
  if (!evoCpuMod) {
    evoCpuMod = await import(/* @vite-ignore */ `${base}/cpu/index.js`);
  }
  engine = await loadWithFactory(evoCpuMod.createEvoDiffCPU, {}, false);
  loadedPrecision = precision;
  loadedBackend = 'cpu';
  return { engine, backend: 'cpu' as const };
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'dispose') {
      activeAbort?.abort();
      activeAbort = null;
      disposeEngine();
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const p = msg.params ?? {};
      const taskMode = normalizeTaskMode(p.taskMode);
      const precision = normalizePrecision(p.precision);
      const length = Math.max(12, Math.min(512, Math.round(Number(p.length ?? 64))));
      const numSequences = Math.max(1, Math.min(8, Math.round(Number(p.numSequences ?? 2))));
      const rawFixedSeq = String(p.fixedSequence ?? '').replace(/\s+/g, '').toUpperCase();
      const rawMaskPositions = String(p.maskPositions ?? '').trim();
      const seed = Math.max(0, Math.round(Number(p.seed ?? 42)));

      const upstreamSeqs = msg.inputs?.sequences as SequenceSet | undefined;
      let templateSeq = rawFixedSeq;
      if (taskMode === 'infill' && !templateSeq && upstreamSeqs?.candidates?.length) {
        templateSeq = upstreamSeqs.candidates[0].sequence.replace(/[:/,\s-]+/g, '').toUpperCase();
      }

      let maskPositions: number[] | undefined;
      if (taskMode === 'infill') {
        if (!templateSeq) {
          throw new Error(
            'EvoDiff in "infill" mode requires either a connected SequenceSet or a Fixed Sequence template (use # for masked positions).',
          );
        }
        const parsedMasks = parseMaskPositions(rawMaskPositions, templateSeq.length);
        if (parsedMasks.length > 0) {
          maskPositions = parsedMasks;
        } else if (!templateSeq.includes('#')) {
          throw new Error(
            'EvoDiff "infill" mode requires "#" mask characters in the template or 1-based Mask Positions (e.g. "1-12, 40-50").',
          );
        }
      }

      const { engine: activeEngine, backend } = await ensureEngine(msg.id, precision);
      const controller = new AbortController();
      activeAbort = controller;

      const candidates: SequenceCandidate[] = [];
      const runHash = fnv1a(
        JSON.stringify({
          taskMode,
          precision,
          length,
          numSequences,
          templateSeq,
          rawMaskPositions,
          seed,
          upstreamId: upstreamSeqs?.id ?? null,
        }),
      );

      try {
        for (let s = 0; s < numSequences; s++) {
          const res = await activeEngine.generate({
            ...(taskMode === 'infill'
              ? { fixedSequence: templateSeq, ...(maskPositions ? { maskPositions } : {}) }
              : { length }),
            seed: seed + s,
            signal: controller.signal,
            onProgress: (prog: { step?: number; total?: number }) => {
              const step = prog.step ?? 0;
              const tot = prog.total ?? (taskMode === 'infill' ? templateSeq.length : length);
              const frac = tot > 0 ? Math.min(1, step / tot) : 0;
              const overall = 0.48 + ((s + frac) / numSequences) * 0.5;
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'diffuse',
                progress: Math.min(0.98, overall),
                detail: `EvoDiff-38M (${backend}) seq ${s + 1}/${numSequences}: residue ${step}/${tot}`,
              });
            },
          });

          // Replace any rare ambiguous 'B' from upstream 21-token vocabulary with 'D' for downstream folders
          const seq = String(res.sequence ?? '').replace(/B/g, 'D');
          const designKey = `evodiff-${s + 1}`;
          const seqKey = makeSequenceKey('evodiff-oa-dm-38m', designKey, s);
          candidates.push({
            id: `evodiff-s${s + 1}-${runHash}`,
            key: seqKey,
            sequenceKey: seqKey,
            sequence: seq,
            score: null,
            globalScore: null,
            sourceModelId: 'evodiff',
            designKey,
            designIndex: s,
            sampleIndex: s,
            seed: seed + s,
          });
        }
      } finally {
        if (activeAbort === controller) activeAbort = null;
      }

      const sequencesOut: SequenceSet = {
        kind: 'SequenceSet',
        id: `evodiff-seqs-${runHash}`,
        parentStructureId: upstreamSeqs?.parentStructureId ?? 'evodiff-de-novo',
        sourceModelId: 'evodiff',
        collectionKind: candidates.length > 1 ? 'multi-sample' : undefined,
        candidates,
      };

      self.postMessage({
        id: msg.id,
        type: 'result',
        outputs: { sequences: sequencesOut },
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
