/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/rfdiffusion-webgpu (BSD · RosettaCommons/RFdiffusion Base_ckpt.pt)
// Runs original RFdiffusion SE(3) backbone diffusion (unconditional, contig motif scaffolding, Cn/Dn symmetry).

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { StructureArtifact, TrajectoryFramesArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/rfdiffusion-webgpu/resolve/main';

type Precision = 'int8' | 'fp16';
type TaskMode = 'unconditional' | 'motif' | 'symmetry';

const VALID_SYMMETRIES = new Set(['c2', 'c3', 'c4', 'c6', 'd2', 'd3']);
const SYMMETRY_COPIES: Record<string, number> = { c2: 2, c3: 3, c4: 4, c6: 6, d2: 4, d3: 6 };

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function collectOwnedTransferables(views: Array<ArrayBufferView | undefined>): Transferable[] {
  const seen = new Set<ArrayBuffer>();
  const out: Transferable[] = [];
  for (const v of views) {
    if (!v) continue;
    const buf = v.buffer;
    if (!(buf instanceof ArrayBuffer)) continue;
    if (v.byteOffset !== 0 || v.byteLength !== buf.byteLength) continue;
    if (seen.has(buf)) continue;
    seen.add(buf);
    out.push(buf);
  }
  return out;
}

function normalizePrecision(raw: unknown): Precision {
  return String(raw ?? 'int8').trim().toLowerCase() === 'fp16' ? 'fp16' : 'int8';
}

function normalizeTaskMode(raw: unknown): TaskMode {
  const val = String(raw ?? 'unconditional').trim().toLowerCase();
  if (val === 'motif' || val === 'symmetry') return val;
  return 'unconditional';
}

function normalizeSymmetry(raw: unknown): string {
  const val = String(raw ?? 'c3').trim().toLowerCase();
  return VALID_SYMMETRIES.has(val) ? val : 'c3';
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfdMod: any = null;
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

  if (!rfdMod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading RFdiffusion WebGPU modules...',
    });
    const base = `${self.location.origin}/vendor/rfdiffusion`;
    rfdMod = await import(/* @vite-ignore */ `${base}/webgpu/index.js`);
  }

  const { adapter } = await requestResilientWebGpuAdapter({ allowSoftwareFallback: true });
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  ownedDevice = device;

  const envBaseUrl = (import.meta.env.VITE_RFDIFFUSION_WEIGHTS_URL as string | undefined)?.trim();
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
      progress: 0.06,
      detail: `Fetching RFdiffusion Base (${precision.toUpperCase()}) weights...`,
    });

    try {
      return await rfdMod.createRFdiffusionEngine({
        baseUrl,
        precision,
        device,
        cache: 'none',
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { loaded?: number; total?: number; shard?: number; shards?: number }) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac =
            total > 0
              ? loaded / total
              : prog.shards && prog.shard !== undefined
                ? prog.shard / prog.shards
                : 0.4;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.45, 0.06 + frac * 0.39),
            detail:
              total > 0
                ? `RFdiffusion weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
                : `Loading RFdiffusion weights (${precision.toUpperCase()})...`,
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
      const rawLength = Math.max(16, Math.min(256, Math.round(Number(p.length ?? 80))));
      const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));
      const numSteps = Math.max(5, Math.min(50, Math.round(Number(p.numSteps ?? 25))));
      const contig = String(p.contig ?? '').trim();
      const symmetry = taskMode === 'symmetry' ? normalizeSymmetry(p.symmetry) : undefined;
      const seed = Math.max(0, Math.round(Number(p.seed ?? 42)));

      const upstreamStruct = msg.inputs?.structure as StructureArtifact | undefined;
      if (taskMode === 'motif' && !upstreamStruct) {
        throw new Error('RFdiffusion in "motif" mode requires a connected upstream structure.');
      }

      let effectiveLength = rawLength;
      if (symmetry) {
        const copies = SYMMETRY_COPIES[symmetry] ?? 3;
        effectiveLength = Math.max(copies * 8, Math.round(rawLength / copies) * copies);
      }

      let effectiveContig: string | undefined;
      if (taskMode === 'motif' && upstreamStruct) {
        if (contig) {
          effectiveContig = contig;
        } else {
          const ch = upstreamStruct.chainIds?.[0] ?? 'A';
          const firstRes = upstreamStruct.sourceResidues?.[0]?.resNum ?? 1;
          const motifLen = Math.min(upstreamStruct.length, 20);
          const endRes = firstRes + motifLen - 1;
          const flank = Math.max(10, Math.floor((effectiveLength - motifLen) / 2));
          effectiveContig = `${flank}-${flank}/${ch}${firstRes}-${endRes}/${flank}-${flank}`;
        }
      }

      const activeEngine = await ensureEngine(msg.id, precision);
      const controller = new AbortController();
      activeAbort = controller;

      const items: StructureArtifact[] = [];
      let primaryTrajFrames: Float32Array[] = [];
      let primaryTrajSteps: number[] = [];

      const runHash = fnv1a(
        JSON.stringify({
          taskMode,
          precision,
          effectiveLength,
          numDesigns,
          numSteps,
          effectiveContig: effectiveContig ?? '',
          symmetry: symmetry ?? '',
          seed,
          upstreamId: upstreamStruct?.id ?? null,
        }),
      );

      try {
        for (let d = 0; d < numDesigns; d++) {
          const sampled = await activeEngine.sample({
            ...(effectiveContig
              ? { contig: effectiveContig, inputPdb: upstreamStruct?.pdbText }
              : { length: effectiveLength }),
            ...(symmetry ? { symmetry } : {}),
            numSteps,
            seed: seed + d,
            signal: controller.signal,
            onProgress: (prog: { step?: number; total?: number; t?: number }) => {
              const step = prog.step ?? 0;
              const tot = prog.total ?? numSteps;
              const frac = tot > 0 ? Math.min(1, step / tot) : 0;
              const overall = 0.48 + ((d + frac) / numDesigns) * 0.5;
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'diffuse',
                progress: Math.min(0.98, overall),
                detail: `RFdiffusion design ${d + 1}/${numDesigns}: step ${step}/${tot}`,
              });
            },
          });

          const pdbText = String(sampled.pdbText ?? '');
          const parsed = parseStructureText(pdbText);
          const geometry = computeGeometryStats(
            parsed.caCoords,
            parsed.chainIds,
            parsed.resNums,
            parsed.insCodes,
          );
          const designKey = `design-${d + 1}`;
          const structId = `rfd-${taskMode}-${parsed.nResidues}aa-d${d + 1}-${runHash}`;

          const artifact: StructureArtifact = {
            kind: 'StructureArtifact',
            id: structId,
            label: `RFdiffusion${symmetry ? ` (${symmetry.toUpperCase()})` : ''} (${parsed.nResidues} aa${numDesigns > 1 ? ` · #${d + 1}` : ''})`,
            sourceModelId: 'rfdiffusion',
            parentStructureId: upstreamStruct?.id,
            length: parsed.nResidues,
            sequence: parsed.sequence,
            caCoords: parsed.caCoords,
            backbone4Coords: parsed.backbone4Coords,
            pdbText: parsed.pdbText,
            chainIds: parsed.chainIds,
            chainLengths: parsed.diagnostics.chainLengths,
            insCodes: parsed.insCodes,
            sourceResidues: parsed.sourceResidues,
            observedBackboneMask: parsed.observedBackboneMask,
            diagnostics: parsed.diagnostics,
            geometry,
            designKey,
            designIndex: d,
            seed: seed + d,
            provenance: {
              quantization: precision,
              diffusionStepsScheduled: numSteps,
              diffusionStepsRun: numSteps,
            },
          };
          items.push(artifact);

          if (d === 0 && Array.isArray(sampled.trajectory)) {
            const frames: Float32Array[] = [];
            const steps: number[] = [];
            const nRes = parsed.nResidues;
            for (const rec of sampled.trajectory) {
              const raw = (rec.px0 ?? rec.coords) as Float32Array | undefined;
              if (!raw || raw.length < nRes * 9) continue;
              const stride = raw.length >= nRes * 12 ? 12 : 9;
              const ca = new Float32Array(nRes * 3);
              for (let i = 0; i < nRes; i++) {
                ca[i * 3] = raw[i * stride + 3];
                ca[i * 3 + 1] = raw[i * stride + 4];
                ca[i * 3 + 2] = raw[i * stride + 5];
              }
              frames.push(ca);
              steps.push(Number(rec.t ?? frames.length));
            }
            primaryTrajFrames = frames;
            primaryTrajSteps = steps;
          }
        }
      } finally {
        if (activeAbort === controller) activeAbort = null;
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
              label: `RFdiffusion (${items.length} designs · L=${primary.length})`,
              representativeSelection: repSelection,
              collection: {
                kind: 'Collection',
                schema: 'denovo.collection/v1',
                id: `rfd-col-${runHash}`,
                familyKey: 'rfdiffusion',
                collectionKind: 'independent-candidates',
                requestedCount: numDesigns,
                complete: true,
                items,
                keys: items.map((it) => it.designKey ?? it.id),
                representativeSelection: repSelection,
              },
            }
          : primary;

      const trajFrames = (primaryTrajFrames.length > 0 ? primaryTrajFrames : [primary.caCoords.slice()]).map(
        (caCoords, idx) => ({
          key: `frame-${idx}`,
          frameIndex: idx,
          caCoords,
          timeStep: primaryTrajSteps[idx] ?? idx,
        }),
      );

      const trajectoryOut: TrajectoryFramesArtifact = {
        kind: 'TrajectoryFrames',
        id: `rfd-traj-${runHash}`,
        label: `RFdiffusion trajectory (${trajFrames.length} steps)`,
        parentStructureId: primary.id,
        sourceModelId: 'rfdiffusion',
        frameCount: trajFrames.length,
        ordering: 'ordered-frames',
        collectionKind: 'ordered-frames',
        frames: trajFrames,
      };

      const transferables = collectOwnedTransferables([
        ...items.flatMap((it) => [it.caCoords, it.backbone4Coords, it.observedBackboneMask]),
        ...trajFrames.map((f) => f.caCoords),
      ]);

      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          outputs: { structure: structureOut, trajectory: trajectoryOut },
          elapsedMs: performance.now() - t0,
        },
        transferables,
      );
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
