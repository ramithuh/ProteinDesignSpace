/// <reference lib="webworker" />
// Layer 3 Plugin Worker — cg2all (huhlim/cg2all CalphaBasedModel, Apache-2.0; WebGPU port ramithuh/cg2all-webgpu)
// Rebuilds all-atom models from Cα traces: SE(3)-Transformer forward pass on WebGPU (vendor/cg2all/webgpu),
// rigid-group assembly + PDB writing on the CPU (vendor/cg2all/shared). Fans out over structure collections
// and SequenceSet candidates like the PULCHRA plugin. Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  createDeviceLostGuard,
  isWebGpuDeviceLostError,
  requestResilientWebGpuDevice,
} from '$lib/engine/webgpu-device';
import { collectOwnedTransferables, createWorkerHarness, type WorkerRunContext } from '$lib/kernel';
import type { SequenceSet, StructureArtifact } from '$lib/engine/types';
import {
  buildCg2allArtifact,
  cg2allOptionsFromParams,
  inputPdbText,
  planCg2allJobs,
  structureItems,
  wrapCg2allOutputs,
  type Cg2allConversion,
  type Cg2allPrecision,
  type Cg2allVariant,
} from './jobs';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/cg2all-webgpu/resolve/main';
const VENDOR_VERSION = 1;

type Backend = 'webgpu' | 'cpu';

interface Cg2allEngine {
  backend: Backend;
  predict(prepared: unknown): Promise<unknown>;
  destroy(): void;
}

interface Cg2allSharedModule {
  loadWeights(
    baseUrl: string,
    opts: {
      read: (name: string) => Promise<Uint8Array>;
      onProgress?: (p: { loaded?: number; total?: number }) => void;
    },
  ): Promise<{ manifest: { fixAtom?: boolean }; tensors: unknown }>;
  convert(
    engine: Cg2allEngine,
    pdbText: string,
    opts: { sequence?: string; fixAtom: boolean; hydrogens: boolean; ligands: boolean; remarks?: string[] },
  ): Promise<Cg2allConversion>;
}

interface Cg2allGpuModule extends Cg2allSharedModule {
  createCg2allEngine(opts: { device: GPUDevice; weights: unknown }): Promise<Cg2allEngine>;
}

interface Cg2allCpuModule extends Cg2allSharedModule {
  createCg2allCpuEngine(opts: { weights: unknown }): Promise<Cg2allEngine>;
}

let gpuMod: Cg2allGpuModule | null = null;
let cpuMod: Cg2allCpuModule | null = null;
let engine: Cg2allEngine | null = null;
let loadedKey: string | null = null;
// Resilient GPUDevice injected into the engine: kept across checkpoint switches, destroyed only on dispose.
let gpu: { device: GPUDevice; hasF16: boolean } | null = null;
const deviceLostGuard = createDeviceLostGuard();

async function ensureGpu(): Promise<{ device: GPUDevice; hasF16: boolean }> {
  if (gpu) return gpu;
  const { device, hasF16 } = await requestResilientWebGpuDevice({ optionalFeatures: ['shader-f16'] });
  const acquired = { device, hasF16 };
  gpu = acquired;
  deviceLostGuard.watch(device, () => {
    if (gpu !== acquired) return;
    gpu = null;
    // Buffers and pipelines died with the device: the next run rebuilds the engine on a fresh one.
    loadedKey = null;
  });
  return acquired;
}

function releaseEngine(): void {
  try {
    engine?.destroy();
  } catch {
    /* ignore */
  }
  engine = null;
  loadedKey = null;
}

async function loadModule(ctx: WorkerRunContext, backend: Backend): Promise<Cg2allSharedModule> {
  if (backend === 'webgpu') {
    if (!gpuMod) {
      ctx.progress('init', 0.02, 'Loading cg2all WebGPU modules...');
      const url = new URL(`/vendor/cg2all/webgpu/index.js?v=${VENDOR_VERSION}`, self.location.origin).href;
      gpuMod = (await import(/* @vite-ignore */ url)) as Cg2allGpuModule;
    }
    return gpuMod;
  }
  if (!cpuMod) {
    ctx.progress('init', 0.02, 'WebGPU unavailable — loading cg2all CPU fallback...');
    const url = new URL(`/vendor/cg2all/cpu/index.js?v=${VENDOR_VERSION}`, self.location.origin).href;
    cpuMod = (await import(/* @vite-ignore */ url)) as Cg2allCpuModule;
  }
  return cpuMod;
}

async function ensureEngine(
  ctx: WorkerRunContext,
  variant: Cg2allVariant,
  precision: Cg2allPrecision,
  backend: Backend,
): Promise<{ engine: Cg2allEngine; mod: Cg2allSharedModule; fixAtom: boolean }> {
  const key = `${variant}/${precision}/${backend}`;
  // `gpu` is cleared when its device is lost, even mid-load, so a cache hit always runs on a live device.
  if (engine && loadedKey === key && (backend === 'cpu' || gpu)) {
    return { engine, mod: (backend === 'webgpu' ? gpuMod : cpuMod)!, fixAtom: variant === 'ca-fix' };
  }
  releaseEngine();
  const mod = await loadModule(ctx, backend);
  const envBaseUrl = (import.meta.env.VITE_CG2ALL_WEIGHTS_URL as string | undefined)?.trim();
  const rootUrl = (envBaseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
  const baseUrl = `${rootUrl}/${variant}/${precision}`;
  const device = backend === 'webgpu' ? (await ensureGpu()).device : null;
  const tag = `${variant === 'ca-fix' ? 'CalphaBasedModel-FIX' : 'CalphaBasedModel'} ${precision.toUpperCase()}${backend === 'cpu' ? ' · CPU fallback' : ''}`;

  const loadOnce = async (retried: boolean): Promise<Cg2allEngine> => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });
    ctx.progress('weights', 0.06, `Fetching cg2all ${tag} weights...`);
    try {
      const weights = await mod.loadWeights(baseUrl, {
        read: async (name: string): Promise<Uint8Array> => {
          const fileUrl = `${baseUrl}/${name}`;
          const resp = await cachedFetch(fileUrl, { signal: ctx.signal });
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${fileUrl}`);
          return new Uint8Array(await resp.arrayBuffer());
        },
        onProgress: (prog) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac = total > 0 ? loaded / total : 0.5;
          ctx.progress(
            'weights',
            Math.min(0.9, 0.08 + frac * 0.8),
            `cg2all weights (${tag}): ${(loaded / 2 ** 20).toFixed(1)} / ${total > 0 ? (total / 2 ** 20).toFixed(1) : '?'} MB`,
          );
        },
      });
      ctx.progress('init', 0.92, backend === 'webgpu' ? 'Compiling cg2all WGSL pipelines...' : 'Preparing cg2all CPU engine...');
      return backend === 'webgpu'
        ? await (mod as Cg2allGpuModule).createCg2allEngine({ device: device!, weights })
        : await (mod as Cg2allCpuModule).createCg2allCpuEngine({ weights });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (!retried && anyFromCache && fetchedUrls.length > 0 && /checksum|truncated|manifest|mismatch/i.test(msg)) {
        await evictCachedUrls(fetchedUrls);
        return await loadOnce(true);
      }
      throw err;
    }
  };

  engine = await loadOnce(false);
  loadedKey = key;
  return { engine, mod, fixAtom: variant === 'ca-fix' };
}

function webGpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);
}

createWorkerHarness({
  async run(ctx) {
    const structure = ctx.inputs?.structure as StructureArtifact | undefined;
    if (!structure || structure.kind !== 'StructureArtifact') {
      throw new Error('cg2all requires an upstream StructureArtifact on its structure input.');
    }
    const seqInput = ctx.inputs?.sequences as SequenceSet | undefined;
    const sequences = seqInput?.kind === 'SequenceSet' ? seqInput : undefined;
    const p = ctx.params ?? {};
    const options = cg2allOptionsFromParams({
      variant: p.variant,
      precision: p.precision,
      outputHydrogens: p.outputHydrogens,
      preserveLigands: p.preserveLigands,
    });
    const jobs = planCg2allJobs(structureItems(structure), sequences);

    let backend: Backend = webGpuAvailable() ? 'webgpu' : 'cpu';
    let gpuRecoveries = 0;
    const acquire = async () => {
      try {
        return await ensureEngine(ctx, options.variant, options.precision, backend);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (backend === 'cpu' || /HTTP \d+|fetch|network|checksum|truncated|manifest|abort/i.test(msg)) throw err;
        // No usable adapter/device: run the pure-JS engine instead.
        backend = 'cpu';
        return await ensureEngine(ctx, options.variant, options.precision, backend);
      }
    };
    let active = await acquire();

    const outputs: StructureArtifact[] = [];
    let atoms = 0;
    for (const [i, job] of jobs.entries()) {
      ctx.signal.throwIfAborted();
      ctx.progress(
        'infer',
        0.93 + (0.06 * i) / jobs.length,
        `Backmapping all-atom model ${i + 1}/${jobs.length}${backend === 'cpu' ? ' (CPU fallback)' : ''}...`,
      );
      const text = inputPdbText(job.parent);
      const convertOpts = {
        ...(job.candidate ? { sequence: job.candidate.sequence } : {}),
        fixAtom: active.fixAtom,
        hydrogens: options.outputHydrogens,
        ligands: options.preserveLigands,
        remarks: [`cg2all ${options.variant === 'ca-fix' ? 'CalphaBasedModel-FIX' : 'CalphaBasedModel'} (${options.precision}, ${backend})`],
      };
      let result: Cg2allConversion;
      for (;;) {
        try {
          result = await active.mod.convert(active.engine, text, convertOpts);
          break;
        } catch (err) {
          if (backend !== 'webgpu' || !isWebGpuDeviceLostError(err)) throw err;
          // Fail fast on device loss: rebuild once on a fresh device, then fall back to the CPU engine.
          releaseEngine();
          if (gpuRecoveries++ >= 1) backend = 'cpu';
          active = await acquire();
        }
      }
      outputs.push(buildCg2allArtifact(job, result, options, backend));
      atoms += result.atomCount;
    }
    ctx.progress('infer', 1, `Rebuilt ${outputs.length} all-atom model(s), ${atoms} atoms`);

    const out = wrapCg2allOutputs(outputs, structure);
    const transferables = collectOwnedTransferables(
      outputs.flatMap((it) => [it.caCoords, it.backbone4Coords, it.observedBackboneMask]),
    );
    return { outputs: { structure: out }, transferables };
  },

  async dispose() {
    releaseEngine();
    deviceLostGuard.destroy(gpu?.device);
    gpu = null;
  },
});

export {};
