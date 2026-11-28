/// <reference lib="webworker" />
// Layer 3 Plugin Worker — artios-backbone (ArtiosBackbone all-atom EDM diffusion, Apache-2.0;
// Ligo-Biosciences/artios, WebGPU port ramithuh/artios-webgpu).
//
// One run = build the token layout and AF3-style features (vendor/artios-backbone/shared),
// stream the sharded safetensors bundle, compute the timestep-independent conditioning cache once
// on the CPU (vendor/artios-backbone/cpu), then integrate the EDM Karras reverse-diffusion
// schedule with Kabsch-aligned self-conditioning on WebGPU (vendor/artios-backbone/webgpu),
// falling back to the CPU denoiser when WebGPU is unavailable or its device is lost.
// Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  createDeviceLostGuard,
  isWebGpuDeviceLostError,
  requestResilientWebGpuDevice,
} from '$lib/engine/webgpu-device';
import { collectOwnedTransferables, createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { StructureArtifact } from '$lib/engine/types';
import { buildStandardComponents, type CcdComponent } from './ccd-standard';
import {
  artiosBackboneOptionsFromParams,
  artiosGpuBatchLimit,
  buildArtiosFeatures,
  buildArtiosBackboneArtifact,
  buildMotifPlan,
  buildUnconditionalPlan,
  componentFromInputLigand,
  coordsToArtiosPdb,
  groupResidues,
  parsePdbAtomRecords,
  rcsbCcdUrl,
  runArtiosEdmSampler,
  stitchUnindexedMotifCoords,
  wrapArtiosBackboneOutputs,
  type ArtiosBackboneBackend,
  type ArtiosBackboneCpuFallbackReason,
  type ArtiosBackbonePrecision,
  type ArtiosDenoiser,
  type ArtiosFeatures,
  type ArtiosSharedModule,
  type ArtiosTokenPlan,
} from './jobs';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/artios-webgpu/resolve/main/backbone';
const VENDOR_VERSION = 3;
/* eslint-disable @typescript-eslint/no-explicit-any */
type AnyModule = any;

interface WeightBundle {
  names: string[];
  manifest: Record<string, unknown>;
  precision: string;
  config: Record<string, unknown>;
  has(name: string): boolean;
  get(name: string): { shape: number[]; data: Float32Array };
}

interface GpuEngine {
  prepareConditioning(features: unknown): Promise<unknown>;
  denoiseStep(
    xNoisy: Float32Array,
    tHat: number | ArrayLike<number>,
    cond: unknown,
    options?: { prevPred?: Float32Array | null },
  ): Promise<Float32Array>;
  destroy(): void;
}

interface CpuModel {
  conditioningCache(features: unknown, options?: { motifAugmentation?: unknown }): unknown;
  denoise(options: {
    features: unknown;
    rNoisy: Float32Array;
    noiseLevel: number;
    cache?: unknown;
    prevPred?: Float32Array | null;
  }): Float32Array;
}

let sharedMod: AnyModule = null;
let cpuMod: AnyModule = null;
let gpuMod: AnyModule = null;

let bundle: WeightBundle | null = null;
let bundlePrecision: ArtiosBackbonePrecision | null = null;
let gpuEngine: GpuEngine | null = null;
let gpuEnginePrecision: ArtiosBackbonePrecision | null = null;
let gpu: { device: GPUDevice } | null = null;
const deviceLostGuard = createDeviceLostGuard();
/** CCD components fetched from RCSB, kept for the worker's lifetime. */
const fetchedComponents = new Map<string, CcdComponent>();

function releaseGpuEngine(): void {
  try {
    gpuEngine?.destroy();
  } catch {
    /* ignore */
  }
  gpuEngine = null;
  gpuEnginePrecision = null;
}

async function ensureGpuDevice(): Promise<GPUDevice> {
  if (gpu) return gpu.device;
  const { device } = await requestResilientWebGpuDevice({ optionalFeatures: ['shader-f16'] });
  const acquired = { device };
  gpu = acquired;
  deviceLostGuard.watch(device, () => {
    if (gpu !== acquired) return;
    gpu = null;
    // Buffers and pipelines died with the device: the next attempt rebuilds on a fresh one.
    releaseGpuEngine();
  });
  return device;
}

function webGpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);
}

async function loadSharedModules(ctx: WorkerRunContext): Promise<void> {
  if (sharedMod && cpuMod) return;
  ctx.progress('init', 0.02, 'Loading ArtiosBackbone modules...');
  const base = `${self.location.origin}/vendor/artios-backbone`;
  [sharedMod, cpuMod] = await Promise.all([
    import(/* @vite-ignore */ `${base}/shared/index.js?v=${VENDOR_VERSION}`),
    import(/* @vite-ignore */ `${base}/cpu/index.js?v=${VENDOR_VERSION}`),
  ]);
}

async function loadGpuModule(ctx: WorkerRunContext): Promise<AnyModule> {
  if (gpuMod) return gpuMod;
  ctx.progress('init', 0.04, 'Loading ArtiosBackbone WebGPU modules...');
  const url = `${self.location.origin}/vendor/artios-backbone/webgpu/index.js?v=${VENDOR_VERSION}`;
  gpuMod = await import(/* @vite-ignore */ url);
  return gpuMod;
}

// ===========================================================================
// weights
// ===========================================================================

function weightsBaseUrl(precision: ArtiosBackbonePrecision): string {
  const envBaseUrl = (import.meta.env.VITE_ARTIOS_BACKBONE_WEIGHTS_URL as string | undefined)?.trim();
  const rootUrl = (envBaseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
  return `${rootUrl}/${precision}`;
}

async function ensureWeights(
  ctx: WorkerRunContext,
  precision: ArtiosBackbonePrecision,
): Promise<WeightBundle> {
  if (bundle && bundlePrecision === precision) return bundle;
  bundle = null;
  bundlePrecision = null;
  releaseGpuEngine();
  const baseUrl = weightsBaseUrl(precision);
  const tag = precision.toUpperCase();

  const loadOnce = async (retried: boolean): Promise<WeightBundle> => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((url, fromCache) => {
      fetchedUrls.push(url);
      if (fromCache) anyFromCache = true;
    });
    const read = async (name: string): Promise<Uint8Array> => {
      const url = `${baseUrl}/${name}`;
      const resp = await cachedFetch(url, { signal: ctx.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
      return new Uint8Array(await resp.arrayBuffer());
    };

    try {
      ctx.progress('weights', 0.06, `Fetching ArtiosBackbone ${tag} manifest...`);
      const manifestBytes = await read('manifest.json');
      const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as {
        shards?: Array<{ name: string; bytes?: number }>;
      };
      const shards = manifest.shards ?? [];
      const totalBytes = shards.reduce((sum, s) => sum + (s.bytes ?? 0), 0);
      let loadedBytes = 0;
      let shardIndex = 0;
      const loaded = await sharedMod.loadWeightBundle({
        manifest,
        fetchShard: async (name: string) => {
          ctx.signal.throwIfAborted();
          const bytes = await read(name);
          loadedBytes += bytes.byteLength;
          shardIndex += 1;
          const frac = totalBytes > 0 ? loadedBytes / totalBytes : shardIndex / Math.max(1, shards.length);
          ctx.progress(
            'weights',
            Math.min(0.42, 0.08 + frac * 0.32),
            `ArtiosBackbone ${tag} weights: ${(loadedBytes / 2 ** 20).toFixed(1)} / ${
              totalBytes > 0 ? (totalBytes / 2 ** 20).toFixed(1) : '?'
            } MiB (shard ${shardIndex}/${shards.length})`,
          );
          return bytes;
        },
        verify: true,
      });
      return loaded as WeightBundle;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        (err instanceof SyntaxError ||
          /digest|checksum|sha-?256|truncated|manifest|mismatch|json|unexpected token/i.test(msg))
      ) {
        await evictCachedUrls(fetchedUrls);
        return await loadOnce(true);
      }
      throw err;
    }
  };

  const loadedBundle = await loadOnce(false);
  bundle = loadedBundle;
  bundlePrecision = precision;
  return loadedBundle;
}

// ===========================================================================
// CCD components
// ===========================================================================

/**
 * Resolve every component code in the layout: bundled standard residues first, then the RCSB CCD
 * for anything else, falling back to the input structure's own ligand conformer when that fetch
 * fails (offline, unknown code, or a non-CIF response).
 */
async function resolveComponents(
  ctx: WorkerRunContext,
  codes: readonly string[],
  ligandAtomsByCode: Map<string, ReturnType<typeof parsePdbAtomRecords>>,
  components: Map<string, CcdComponent>,
  warnings: string[],
): Promise<void> {
  for (const code of codes) {
    if (components.has(code)) continue;
    const cached = fetchedComponents.get(code);
    if (cached) {
      components.set(code, cached);
      continue;
    }
    try {
      ctx.progress('init', 0.05, `Fetching CCD ideal conformer for ${code}...`);
      const resp = await fetch(rcsbCcdUrl(code), { signal: ctx.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const text = await resp.text();
      if (!/^\s*data_/m.test(text)) throw new Error('response was not a CCD CIF');
      const parsed = sharedMod.parseChemComp(text) as CcdComponent;
      if (!parsed.atoms.length) throw new Error('component has no heavy atoms');
      fetchedComponents.set(code, parsed);
      components.set(code, parsed);
    } catch (err) {
      if (ctx.signal.aborted) throw err;
      const atoms = ligandAtomsByCode.get(code);
      if (!atoms || atoms.length === 0) {
        warnings.push(
          `No CCD entry for ${code} (${err instanceof Error ? err.message : String(err)}); its atoms keep a zero reference conformer (ref_mask = 0).`,
        );
        continue;
      }
      const fallback = componentFromInputLigand(code, atoms);
      components.set(code, fallback);
      warnings.push(
        `Could not fetch the RCSB CCD ideal conformer for ${code} (${err instanceof Error ? err.message : String(err)}); using the input structure's own ${code} conformer for ref_pos instead.`,
      );
    }
  }
}

// ===========================================================================
// run
// ===========================================================================

createWorkerHarness({
  async run(ctx) {
    const options = artiosBackboneOptionsFromParams(ctx.params ?? {});
    const upstream = ctx.inputs?.structure as StructureArtifact | undefined;
    const structure = upstream?.kind === 'StructureArtifact' ? upstream : undefined;
    if (options.motifContig && !structure) {
      throw new Error(
        'ArtiosBackbone motifContig requires a StructureArtifact on the structure input (leave motifContig empty for unconditional design).',
      );
    }

    await loadSharedModules(ctx);
    const shared = sharedMod as unknown as ArtiosSharedModule;
    const warnings: string[] = [];

    // --- token layout ------------------------------------------------------
    const components = buildStandardComponents();
    let plan: ArtiosTokenPlan;
    if (structure) {
      const pdbText = structure.pdbText?.trim();
      if (!pdbText) {
        throw new Error(`ArtiosBackbone: upstream structure ${structure.id} has no PDB coordinates.`);
      }
      plan = buildMotifPlan({ pdbText, options, components });
      if (plan.missingComponentCodes.length > 0) {
        const ligandAtomsByCode = new Map<string, ReturnType<typeof parsePdbAtomRecords>>();
        for (const res of groupResidues(parsePdbAtomRecords(pdbText))) {
          if (!ligandAtomsByCode.has(res.resName)) ligandAtomsByCode.set(res.resName, []);
          ligandAtomsByCode.get(res.resName)!.push(...res.atoms);
        }
        await resolveComponents(ctx, plan.missingComponentCodes, ligandAtomsByCode, components, warnings);
      }
    } else {
      plan = buildUnconditionalPlan(options.length, components);
    }

    const motifCount = plan.motifIndices.length + plan.unindexedMotifResidues.length;
    ctx.progress(
      'init',
      0.05,
      motifCount > 0 || plan.ligandResNames.length > 0
        ? `Featurizing ${plan.designedLength} aa design (${motifCount} motif residue(s)${
            plan.ligandResNames.length ? `, ligand ${[...new Set(plan.ligandResNames)].join('/')}` : ''
          })...`
        : `Featurizing ${plan.designedLength} aa unconditional design...`,
    );
    const featureResult = buildArtiosFeatures(shared, plan, options, components);
    // `prepareFeatures` validates every width and stamps nAtom/nToken, so both denoisers and the
    // conditioning cache see one already-validated object.
    const features = cpuMod.prepareFeatures(featureResult.features) as ArtiosFeatures &
      Record<string, unknown>;
    warnings.push(...featureResult.warnings);
    const nAtom = features.nAtom;
    const nToken = features.nToken;
    const atomMask = features.atom_mask;
    if (structure && motifCount === 0 && plan.ligandResNames.length === 0) {
      warnings.push(
        `The connected structure ${structure.id} is not conditioning this run: set motifContig to scaffold residues from it, or enable includeLigand when it carries a HETATM ligand.`,
      );
    }

    // --- weights + conditioning cache --------------------------------------
    const weights = await ensureWeights(ctx, options.precision);
    ctx.progress('init', 0.44, `Computing ArtiosBackbone conditioning cache (${nToken} tokens, ${nAtom} atoms)...`);
    const cpuModel = cpuMod.createArtiosBackboneCPU({
      weights,
      config: weights.config,
    }) as CpuModel;
    // Timestep- and sample-independent, so it is computed once and shared by every design.
    const cache = cpuModel.conditioningCache(features, {
      motifAugmentation: { rng: shared.createRng(options.seed) },
    });

    // --- denoiser (WebGPU, falling back to CPU) -----------------------------
    let backend: ArtiosBackboneBackend = webGpuAvailable() ? 'webgpu' : 'cpu';
    let fallbackReason: ArtiosBackboneCpuFallbackReason | undefined =
      backend === 'cpu' ? 'unavailable' : undefined;
    let gpuRecoveries = 0;
    let gpuCond: unknown = null;

    const acquireGpu = async (): Promise<GpuEngine> => {
      const mod = await loadGpuModule(ctx);
      const device = await ensureGpuDevice();
      if (!gpuEngine || gpuEnginePrecision !== options.precision) {
        releaseGpuEngine();
        ctx.progress('init', 0.5, `Compiling ArtiosBackbone WGSL pipelines (${options.precision.toUpperCase()})...`);
        gpuEngine = (await mod.createArtiosBackboneGpu({
          device,
          weights,
          config: weights.config,
          precision: options.precision,
        })) as GpuEngine;
        gpuEnginePrecision = options.precision;
        gpuCond = null;
      }
      if (!gpuCond) gpuCond = await gpuEngine.prepareConditioning(cache);
      return gpuEngine;
    };

    if (backend === 'webgpu') {
      try {
        await acquireGpu();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (/HTTP \d+|fetch|network|digest|checksum|truncated|manifest|json|abort/i.test(msg)) throw err;
        releaseGpuEngine();
        backend = 'cpu';
        fallbackReason = isWebGpuDeviceLostError(err) ? 'device-lost' : 'unavailable';
        console.warn(`[artios-backbone] WebGPU unavailable (${msg}); falling back to CPU denoiser.`);
        warnings.push(`WebGPU unavailable for ArtiosBackbone (${msg}); running the CPU JS denoiser.`);
      }
    }

    const gpuDenoise: ArtiosDenoiser = async (xNoisy, tHat, prevPred) => {
      for (;;) {
        try {
          const engine = await acquireGpu();
          return await engine.denoiseStep(xNoisy, tHat, gpuCond, prevPred ? { prevPred } : undefined);
        } catch (err) {
          if (!isWebGpuDeviceLostError(err) || gpuRecoveries >= 1) throw err;
          // Rebuild once on a fresh device; a second loss drops to the CPU denoiser.
          gpuRecoveries += 1;
          releaseGpuEngine();
          gpu = null;
        }
      }
    };
    const cpuDenoise: ArtiosDenoiser = async (xNoisy, tHat, prevPred) =>
      cpuModel.denoise({ features, rNoisy: xNoisy, noiseLevel: tHat, cache, prevPred });

    // --- sample -------------------------------------------------------------
    const runHash = fnv1a(
      JSON.stringify({
        ...options,
        numDesigns: undefined,
        upstreamId: structure?.id ?? null,
        nAtom,
        nToken,
      }),
    );
    const items: StructureArtifact[] = [];
    const sampleBase = 0.52;
    const sampleSpan = 0.46;
    let done = 0;

    while (items.length < options.numDesigns) {
      ctx.signal.throwIfAborted();
      const remaining = options.numDesigns - items.length;
      const nSample = backend === 'webgpu' ? Math.min(remaining, artiosGpuBatchLimit(nToken)) : 1;
      const firstIndex = items.length;
      const batchSeed = options.seed + firstIndex;

      let coords: Float32Array;
      try {
        coords = await runArtiosEdmSampler(shared, {
          denoise: backend === 'webgpu' ? gpuDenoise : cpuDenoise,
          nAtom,
          nSample,
          atomMask,
          options,
          seed: batchSeed,
          signal: ctx.signal,
          onStep: (step) => {
            const frac = (done + step / options.numSteps * nSample) / options.numDesigns;
            ctx.progress(
              'infer',
              Math.min(0.98, sampleBase + frac * sampleSpan),
              options.numDesigns > 1
                ? `ArtiosBackbone designs ${firstIndex + 1}–${firstIndex + nSample}/${options.numDesigns} · step ${step}/${options.numSteps}${
                    backend === 'cpu' ? ' (CPU fallback)' : ''
                  }`
                : `ArtiosBackbone · step ${step}/${options.numSteps}${backend === 'cpu' ? ' (CPU fallback)' : ''}`,
            );
          },
        });
      } catch (err) {
        if (backend !== 'webgpu' || ctx.signal.aborted || !isWebGpuDeviceLostError(err)) throw err;
        // Device lost twice: finish the run on the CPU denoiser instead of failing.
        releaseGpuEngine();
        gpu = null;
        backend = 'cpu';
        fallbackReason = 'device-lost';
        warnings.push('ArtiosBackbone lost its WebGPU device twice; the remaining designs ran on the CPU denoiser.');
        continue;
      }

      for (let s = 0; s < nSample; s++) {
        const rawDesignCoords = coords.slice(s * nAtom * 3, (s + 1) * nAtom * 3);
        const { coords: designCoords, plan: outputPlan } =
          plan.unindexedMotifResidues.length > 0
            ? stitchUnindexedMotifCoords(rawDesignCoords, plan)
            : { coords: rawDesignCoords, plan };
        const designIndex = firstIndex + s;
        const seed = options.seed + designIndex;
        const pdbText = coordsToArtiosPdb(designCoords, outputPlan, {
          remarks: [
            `ArtiosBackbone ${options.precision} (${backend === 'cpu' ? 'cpu-js' : 'webgpu'}) ` +
              `L=${outputPlan.designedLength} steps=${options.numSteps} seed=${seed}`,
            outputPlan.motifIndices.length > 0
              ? `motif residues: ${outputPlan.motifIndices.map((i) => i + 1).join(',')}`
              : outputPlan.ligandResNames.length > 0
                ? `ligand-conditioned design: ${[...new Set(outputPlan.ligandResNames)].join(',')}`
                : 'unconditional de novo design',
          ],
        });
        items.push(
          buildArtiosBackboneArtifact(
            { pdbText, designIndex, seed, stepsRun: options.numSteps, warnings },
            { options, plan: outputPlan, backend, fallbackReason, parent: structure, runHash },
          ),
        );
      }
      done += nSample;
    }

    ctx.progress('infer', 1, `Generated ${items.length} ArtiosBackbone design(s) (${plan.designedLength} aa)`);
    const structureOut = wrapArtiosBackboneOutputs(items);
    const transferables = collectOwnedTransferables(
      items.flatMap((it) => [it.caCoords, it.backbone4Coords, it.observedBackboneMask]),
    );
    return { outputs: { structure: structureOut }, transferables };
  },

  async dispose() {
    releaseGpuEngine();
    deviceLostGuard.destroy(gpu?.device);
    gpu = null;
    bundle = null;
    bundlePrecision = null;
  },
});

export {};
