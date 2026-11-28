/// <reference lib="webworker" />
// Layer 3 Plugin Worker — artios-potts (ArtiosPotts, Ligo-Biosciences/artios, Apache-2.0;
// WebGPU port ramithuh/artios-webgpu)
//
// Two stages per structure:
//   1. Potts parameters `(h, J, edgeIndex, edgeMask)` from the sequence-blind MPNN encoder/decoder
//      and the factorized GraphPotts head. On WebGPU this is `createArtiosPottsGpu()`
//      (vendor/artios-potts/webgpu); on the CPU fallback it is `createArtiosPottsCPU()`.
//   2. The fast pure-JS MCMC sampler / scorer from `vendor/artios-potts/shared`, run on whichever
//      backend produced `(h, J)`. With `cfgScale > 0` and context atoms present, a second forward
//      pass over the ligand-free context supplies the unconditional `(h0, J0)` for CFG.
//
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress and disposal.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  createDeviceLostGuard,
  isWebGpuDeviceLostError,
  requestResilientWebGpuDevice,
} from '$lib/engine/webgpu-device';
import { collectOwnedTransferables, createWorkerHarness, type WorkerRunContext } from '$lib/kernel';
import type { ResidueTrackArtifact, SequenceCandidate, StructureArtifact } from '$lib/engine/types';
import { ARTIOS_POTTS_WEIGHTS_BASE_URL } from './manifest';
import {
  artiosPottsOptionsFromParams,
  artiosPottsRunHash,
  buildArtiosPottsCandidate,
  buildArtiosPottsInput,
  buildArtiosPottsSequenceSet,
  buildArtiosPottsTrack,
  buildFixedTokenMask,
  countDesignablePositions,
  flattenChainSequences,
  inputPdbText,
  planArtiosPottsJobs,
  proteinTokenValues,
  resolveArtiosPottsAlphabet,
  resolveArtiosPottsConstraints,
  type ArtiosPottsCpuFallbackReason,
  type ArtiosPottsInput,
  type ArtiosPottsJob,
  type ArtiosPottsOptions,
  type ArtiosPottsPrecision,
  type ArtiosPottsSample,
  type ArtiosPottsScore,
} from './jobs';

declare const self: DedicatedWorkerGlobalScope;

const VENDOR_VERSION = 1;

type Backend = 'webgpu' | 'cpu';

const CPU_FALLBACK_WARNING: Record<ArtiosPottsCpuFallbackReason, string> = {
  unavailable: 'ArtiosPotts ran on the CPU JS fallback (WebGPU unavailable).',
  'device-lost': 'ArtiosPotts fell back to the CPU JS engine after a WebGPU device loss.',
};

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Raw Potts parameters the shared sampler consumes. */
interface PottsParams {
  h: Float32Array;
  J: Float32Array;
  edgeIndex: Int32Array;
  edgeMask: Float32Array | Uint8Array;
  nToken: number;
  k: number;
  numResidueTypes: number;
}

interface PottsContext {
  nToken: number;
  nAtom: number;
  tokenMask: Uint8Array;
  isProteinToken: Uint8Array;
  isLigandToken: Uint8Array;
  restypeIndex: Int32Array;
  [key: string]: unknown;
}

/** The subset of `vendor/artios-potts/shared/index.js` this worker calls. */
interface SharedModule {
  ALPHABET_BIAS: number;
  featurize(spec: unknown): PottsContext;
  withoutLigand(context: PottsContext): PottsContext;
  buildLigandProximityMask(context: PottsContext, cutoff: number): Uint8Array;
  sequencesByChain(context: PottsContext, sequence?: ArrayLike<number>): Record<string, string>;
  scoreSequence(
    params: PottsParams,
    sequence: ArrayLike<number>,
    options?: { isProteinToken?: ArrayLike<number> },
  ): {
    perNodeEnergy: Float32Array;
    totalEnergy: number;
    proteinEnergy: number;
    logLikelihood: Float32Array;
    pseudoLogLikelihood: number;
  };
  sampleFromParams(
    params: PottsParams,
    options: Record<string, unknown>,
  ): { sequence: Int32Array; totalEnergy: number; acceptedSweeps: number | null };
  loadWeightBundle(options: {
    baseUrl: string;
    precision: ArtiosPottsPrecision;
    read: (url: string) => Promise<Uint8Array>;
    verify?: boolean;
  }): Promise<{ config: Record<string, unknown>; names: string[]; get: (name: string) => unknown }>;
}

/** A backend that turns one featurized context into Potts parameters. */
interface ParamsBackend {
  backend: Backend;
  computeParams(context: PottsContext): Promise<PottsParams>;
  destroy(): void;
}

let sharedMod: SharedModule | null = null;
let gpuMod: any = null;
let cpuMod: any = null;
let engine: ParamsBackend | null = null;
let loadedKey: string | null = null;
// Resilient GPUDevice injected into the engine: kept across precision switches, destroyed on dispose.
let gpu: { device: GPUDevice; hasF16: boolean } | null = null;
const deviceLostGuard = createDeviceLostGuard();

function vendorUrl(rel: string): string {
  return new URL(`/vendor/artios-potts/${rel}?v=${VENDOR_VERSION}`, self.location.origin).href;
}

async function ensureShared(ctx: WorkerRunContext): Promise<SharedModule> {
  if (!sharedMod) {
    ctx.progress('init', 0.02, 'Loading ArtiosPotts featurizer & Potts sampler...');
    sharedMod = (await import(/* @vite-ignore */ vendorUrl('shared/index.js'))) as SharedModule;
  }
  return sharedMod;
}

async function ensureGpu(): Promise<{ device: GPUDevice; hasF16: boolean }> {
  if (gpu) return gpu;
  const { device, hasF16 } = await requestResilientWebGpuDevice({ optionalFeatures: ['shader-f16'] });
  const acquired = { device, hasF16 };
  gpu = acquired;
  deviceLostGuard.watch(device, () => {
    if (gpu !== acquired) return;
    gpu = null;
    // Buffers and pipelines died with the device: the next run rebuilds the engine on a fresh one.
    if (engine?.backend === 'webgpu') loadedKey = null;
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

function webGpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);
}

function weightsBaseUrl(params: Record<string, unknown>): string {
  const override = String(params.weightsBaseUrl ?? '').trim();
  const envBaseUrl = (import.meta.env.VITE_ARTIOS_POTTS_WEIGHTS_URL as string | undefined)?.trim();
  return (override || envBaseUrl || ARTIOS_POTTS_WEIGHTS_BASE_URL).replace(/\/+$/, '');
}

/**
 * Stream one bundle file through the shared weight cache, reporting byte progress.
 * Keeps entries in CacheStorage so a corrupt download can be evicted and retried.
 */
function makeBundleReader(
  ctx: WorkerRunContext,
  tag: string,
  onFile: (url: string, fromCache: boolean) => void,
): (url: string) => Promise<Uint8Array> {
  const cachedFetch = makeCachedFetch(onFile);
  return async (url: string): Promise<Uint8Array> => {
    ctx.signal.throwIfAborted();
    const resp = await cachedFetch(url, { signal: ctx.signal });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
    const total = Number(resp.headers.get('content-length') ?? 0);
    // Small files (manifest.json) are not worth a progress line.
    if (!resp.body || total <= 256 * 1024) {
      return new Uint8Array(await resp.arrayBuffer());
    }
    const chunks: Uint8Array[] = [];
    const reader = resp.body.getReader();
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      chunks.push(value);
      loaded += value.byteLength;
      ctx.progress(
        'weights',
        Math.min(0.9, 0.08 + (loaded / total) * 0.8),
        `ArtiosPotts weights (${tag}): ${(loaded / 2 ** 20).toFixed(1)} / ${(total / 2 ** 20).toFixed(1)} MiB`,
      );
    }
    const out = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return out;
  };
}

/**
 * GPU precision ids differ from bundle precision ids (`fp16` -> `f16`).
 *
 * The bundle is downloaded at the matching precision (so FP16 really is a 5.2 MB download) and the
 * engine then packs the dequantized values back to the same width. For FP16 that round trip is
 * exact; for INT8 it re-derives per-row scales, which can add up to one extra rounding step on top
 * of the bundle's own quantization (disclosed in the model card's INT8 caveat).
 */
function gpuPrecision(precision: ArtiosPottsPrecision): 'f32' | 'f16' | 'int8' {
  if (precision === 'fp32') return 'f32';
  if (precision === 'fp16') return 'f16';
  return 'int8';
}

async function buildBackend(
  ctx: WorkerRunContext,
  shared: SharedModule,
  baseUrl: string,
  precision: ArtiosPottsPrecision,
  backend: Backend,
  fallbackReason?: ArtiosPottsCpuFallbackReason,
): Promise<ParamsBackend> {
  const tag = `${precision.toUpperCase()}${backend === 'cpu' ? ' · CPU fallback' : ''}`;

  const loadOnce = async (retried: boolean): Promise<ParamsBackend> => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const read = makeBundleReader(ctx, tag, (url, fromCache) => {
      fetchedUrls.push(url);
      if (fromCache) anyFromCache = true;
    });
    ctx.progress('weights', 0.06, `Fetching ArtiosPotts ${tag} weights...`);
    try {
      if (backend === 'webgpu') {
        if (!gpuMod) {
          ctx.progress('init', 0.04, 'Loading ArtiosPotts WebGPU modules...');
          gpuMod = await import(/* @vite-ignore */ vendorUrl('webgpu/index.js'));
        }
        const bundle = await shared.loadWeightBundle({ baseUrl, precision, read });
        ctx.progress('init', 0.92, 'Compiling ArtiosPotts WGSL pipelines...');
        const { device } = await ensureGpu();
        const gpuEngine = await gpuMod.createArtiosPottsGpu({
          device,
          weights: bundle,
          config: bundle.config,
          precision: gpuPrecision(precision),
        });
        return {
          backend: 'webgpu',
          async computeParams(context) {
            const out = await gpuEngine.computePottsParams(context);
            return {
              h: out.h,
              J: out.J,
              edgeIndex: out.edgeIndex,
              edgeMask: out.edgeMask,
              nToken: out.shape.nToken,
              k: out.shape.nNeighbours,
              numResidueTypes: out.shape.numResidueTypes,
            };
          },
          destroy: () => gpuEngine.destroy(),
        };
      }

      if (!cpuMod) {
        ctx.progress(
          'init',
          0.04,
          fallbackReason === 'device-lost'
            ? 'WebGPU device lost — loading ArtiosPotts CPU fallback...'
            : 'WebGPU unavailable — loading ArtiosPotts CPU fallback...',
        );
        cpuMod = await import(/* @vite-ignore */ vendorUrl('cpu/index.js'));
      }
      const cpuEngine = await cpuMod.createArtiosPottsCPU({ baseUrl, precision, read });
      ctx.progress('init', 0.92, 'Preparing ArtiosPotts CPU engine...');
      return {
        backend: 'cpu',
        async computeParams(context) {
          return cpuEngine.computeParams(context) as PottsParams;
        },
        destroy: () => {
          /* the CPU engine holds only JS arrays */
        },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        (err instanceof SyntaxError ||
          /checksum|truncated|manifest|mismatch|json|syntaxerror|unexpected token|bytes, manifest says/i.test(msg))
      ) {
        await evictCachedUrls(fetchedUrls);
        return await loadOnce(true);
      }
      throw err;
    }
  };

  return await loadOnce(false);
}

async function ensureEngine(
  ctx: WorkerRunContext,
  shared: SharedModule,
  baseUrl: string,
  precision: ArtiosPottsPrecision,
  backend: Backend,
  fallbackReason?: ArtiosPottsCpuFallbackReason,
): Promise<ParamsBackend> {
  const key = `${baseUrl}|${precision}|${backend}`;
  // `gpu` is cleared when its device is lost, even mid-load, so a cache hit always runs live.
  if (engine && loadedKey === key && (backend === 'cpu' || gpu)) return engine;
  releaseEngine();
  engine = await buildBackend(ctx, shared, baseUrl, precision, backend, fallbackReason);
  loadedKey = key;
  return engine;
}

// ---------------------------------------------------------------------------
// Sampling & scoring on a prepared parameter set
// ---------------------------------------------------------------------------

function toScore(
  shared: SharedModule,
  params: PottsParams,
  context: PottsContext,
  sequence: ArrayLike<number>,
): ArtiosPottsScore {
  const scored = shared.scoreSequence(params, sequence, { isProteinToken: context.isProteinToken });
  return {
    proteinEnergy: scored.proteinEnergy,
    totalEnergy: scored.totalEnergy,
    pseudoLogLikelihood: scored.pseudoLogLikelihood,
    perResidueLogLikelihood: proteinTokenValues(context, scored.logLikelihood),
    perResidueEnergy: proteinTokenValues(context, scored.perNodeEnergy),
  };
}

function samplerOptions(
  shared: SharedModule,
  options: ArtiosPottsOptions,
  context: PottsContext,
  fixedMask: Uint8Array,
  uncond: PottsParams | null,
  seed: number,
): Record<string, unknown> {
  const alphabet = resolveArtiosPottsAlphabet(options.omitAAs);
  const energyBias =
    alphabet.energyBiasCodes.length > 0
      ? Object.fromEntries(alphabet.energyBiasCodes.map((c) => [c, shared.ALPHABET_BIAS]))
      : null;
  return {
    context,
    fixedMask,
    proposal: options.sampler,
    nSweeps: options.numSteps,
    temperature: options.temperature,
    temperatureInit: options.temperatureInit,
    // Upstream `RELEASED_SAMPLING_DEFAULTS`: anneal over the first 80% of sweeps, DLMC rejects.
    annealingFraction: 0.8,
    dt: 0.1,
    rejectionStep: options.sampler === 'dlmc',
    excludeCys: alphabet.excludeCys,
    excludeMet: alphabet.excludeMet,
    ...(energyBias ? { energyBias } : {}),
    complexityPenaltyWeight: options.complexityWeight,
    complexityWindowSize: 30,
    complexityEntropyMin: 2.32,
    activeSiteDistanceCutoff: 10.0,
    granularActiveSiteColoring: true,
    ...(uncond ? { uncond, cfgScale: options.cfgScale } : {}),
    seed,
  };
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface JobResult {
  candidates: SequenceCandidate[];
  lockedChains: string[];
  fixedProteinIndices: Set<number>;
  best: ArtiosPottsSample;
  input: ArtiosPottsInput;
  warnings: string[];
}

createWorkerHarness({
  async run(ctx) {
    const structure = ctx.inputs?.structure as StructureArtifact | undefined;
    if (!structure || structure.kind !== 'StructureArtifact') {
      throw new Error('ArtiosPotts requires an upstream StructureArtifact on its structure input.');
    }
    const p = ctx.params ?? {};
    const options = artiosPottsOptionsFromParams(p);
    // Fail fast on an impossible alphabet before any weight download.
    resolveArtiosPottsAlphabet(options.omitAAs);
    const baseUrl = weightsBaseUrl(p);
    const jobs = planArtiosPottsJobs(structure);

    const shared = await ensureShared(ctx);

    let backend: Backend = webGpuAvailable() ? 'webgpu' : 'cpu';
    let cpuFallbackReason: ArtiosPottsCpuFallbackReason | undefined =
      backend === 'cpu' ? 'unavailable' : undefined;
    let gpuRecoveries = 0;

    const acquire = async (): Promise<ParamsBackend> => {
      try {
        return await ensureEngine(ctx, shared, baseUrl, options.precision, backend, cpuFallbackReason);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (
          backend === 'cpu' ||
          err instanceof SyntaxError ||
          /HTTP \d+|fetch|network|checksum|truncated|manifest|json|syntaxerror|unexpected token|abort/i.test(msg)
        ) {
          throw err;
        }
        // No usable adapter/device: run the pure-JS engine instead.
        backend = 'cpu';
        cpuFallbackReason = isWebGpuDeviceLostError(err) || gpuRecoveries > 0 ? 'device-lost' : 'unavailable';
        return await ensureEngine(ctx, shared, baseUrl, options.precision, backend, cpuFallbackReason);
      }
    };
    let active = await acquire();

    /** Potts parameters for one context, retrying once on a fresh device then on the CPU. */
    const computeParams = async (context: PottsContext): Promise<PottsParams> => {
      for (;;) {
        try {
          return await active.computeParams(context);
        } catch (err) {
          if (backend !== 'webgpu' || !isWebGpuDeviceLostError(err)) throw err;
          releaseEngine();
          if (gpuRecoveries++ >= 1) {
            backend = 'cpu';
            cpuFallbackReason = 'device-lost';
          }
          active = await acquire();
        }
      }
    };

    const results: JobResult[] = [];
    const totalChains = jobs.length * (options.mode === 'score' ? 1 : options.numSequences);
    let completedChains = 0;

    for (const job of jobs) {
      ctx.signal.throwIfAborted();
      results.push(
        await runJob(ctx, shared, job, jobs.length, options, computeParams, () => {
          completedChains++;
          return completedChains / Math.max(1, totalChains);
        }),
      );
    }

    const degradedWarning =
      backend === 'cpu' && cpuFallbackReason ? CPU_FALLBACK_WARNING[cpuFallbackReason] : null;

    const outputs = assembleOutputs(structure, jobs, results, options);
    const notes = [
      ...new Set(results.flatMap((r) => r.warnings).concat(degradedWarning ? [degradedWarning] : [])),
    ];
    const headline =
      options.mode === 'score'
        ? `Scored ${results.length} structure(s) with the Potts model`
        : `Designed ${outputs.sequences.candidates.length} sequence(s)`;
    ctx.progress('sample', 1, notes.length > 0 ? `${headline} \u2014 ${notes.join(' ')}` : headline);
    return {
      outputs: { sequences: outputs.sequences, track: outputs.track },
      transferables: collectOwnedTransferables([outputs.track.values]),
    };
  },

  async dispose() {
    releaseEngine();
    deviceLostGuard.destroy(gpu?.device);
    gpu = null;
  },
});

/** Featurize one backbone, compute its Potts parameters, then score or sample. */
async function runJob(
  ctx: WorkerRunContext,
  shared: SharedModule,
  job: ArtiosPottsJob,
  jobCount: number,
  options: ArtiosPottsOptions,
  computeParams: (context: PottsContext) => Promise<PottsParams>,
  tickChain: () => number,
): Promise<JobResult> {
  const { parent } = job;
  const input = buildArtiosPottsInput(inputPdbText(parent), parent.label || parent.id);
  const warnings = [...input.diagnostics.warnings];
  const context = shared.featurize(input.spec);

  const jobTag = jobCount > 1 ? `Backbone ${job.designIndex + 1}/${jobCount} · ` : '';
  ctx.progress('infer', 0.93, `${jobTag}ArtiosPotts Potts parameters (${context.nToken} tokens)...`);
  const params = await computeParams(context);

  const hasContextAtoms = input.diagnostics.contextAtomCount > 0;
  let uncond: PottsParams | null = null;
  if (options.mode === 'design' && options.cfgScale > 0) {
    if (hasContextAtoms) {
      ctx.progress('infer', 0.95, `${jobTag}Ligand-free pass for classifier-free guidance...`);
      uncond = await computeParams(shared.withoutLigand(context));
    } else {
      warnings.push('Ligand guidance (CFG) was requested but the input has no context atoms; it was skipped.');
    }
  }

  const nativeSequence = Int32Array.from(context.restypeIndex);
  if (options.mode === 'score') {
    const score = toScore(shared, params, context, nativeSequence);
    const flat = flattenChainSequences(input.proteinResidues, shared.sequencesByChain(context, nativeSequence));
    const sample: ArtiosPottsSample = { flat, score, sampleIndex: 0, seed: options.seed, acceptedSweeps: null };
    tickChain();
    return {
      candidates: [
        buildArtiosPottsCandidate(sample, parent, options, idPrefix(job, 'native'), extraOf(job)),
      ],
      lockedChains: [],
      fixedProteinIndices: new Set<number>(),
      best: sample,
      input,
      warnings,
    };
  }

  const { fixedProteinIndices, lockedChains } = resolveArtiosPottsConstraints(
    input.proteinResidues,
    options.fixedPositions,
    parent,
  );
  let proximity: Uint8Array | null = null;
  if (options.ligandCutoff > 0) {
    if (hasContextAtoms) {
      proximity = shared.buildLigandProximityMask(context, options.ligandCutoff);
    } else {
      warnings.push(
        `A ${options.ligandCutoff} Å ligand cutoff was requested but the input has no context atoms; every residue stays designable.`,
      );
    }
  }
  const fixedMask = buildFixedTokenMask(context, fixedProteinIndices, proximity);
  const designable = countDesignablePositions(context, fixedMask);
  if (designable === 0) {
    throw new Error(
      'ArtiosPotts has no designable residue left: every protein position is fixed by fixedPositions, an inherited lock, or the ligand cutoff.',
    );
  }

  const candidates: SequenceCandidate[] = [];
  const samples: ArtiosPottsSample[] = [];
  for (let i = 0; i < options.numSequences; i++) {
    ctx.signal.throwIfAborted();
    const seed = options.seed + i;
    const result = shared.sampleFromParams(
      params,
      samplerOptions(shared, options, context, fixedMask, uncond, seed),
    );
    const score = toScore(shared, params, context, result.sequence);
    const flat = flattenChainSequences(
      input.proteinResidues,
      shared.sequencesByChain(context, result.sequence),
    );
    const sample: ArtiosPottsSample = {
      flat,
      score,
      sampleIndex: i,
      seed,
      acceptedSweeps: result.acceptedSweeps,
    };
    samples.push(sample);
    candidates.push(buildArtiosPottsCandidate(sample, parent, options, idPrefix(job, 'd'), extraOf(job)));
    const frac = tickChain();
    ctx.progress(
      'sample',
      frac,
      `${jobTag}ArtiosPotts ${options.sampler.toUpperCase()} seq ${i + 1}/${options.numSequences} (${designable} designable, PLL ${score.pseudoLogLikelihood.toFixed(3)})`,
    );
  }

  const order = candidates
    .map((c, idx) => ({ c, idx }))
    .sort((a, b) => {
      const diff = (a.c.score ?? 99) - (b.c.score ?? 99);
      return diff !== 0 ? diff : a.idx - b.idx;
    });
  const best = samples[order[0].idx];

  return {
    candidates: order.map((o) => o.c),
    lockedChains,
    fixedProteinIndices,
    best,
    input,
    warnings,
  };
}

function idPrefix(job: ArtiosPottsJob, kind: string): string {
  return job.designKey ? `d${job.designIndex + 1}-artios-potts-${kind}` : `artios-potts-${kind}`;
}

function extraOf(job: ArtiosPottsJob): Partial<SequenceCandidate> {
  if (!job.designKey) return {};
  return { designKey: job.designKey, designIndex: job.designIndex };
}

function assembleOutputs(
  structure: StructureArtifact,
  jobs: ArtiosPottsJob[],
  results: JobResult[],
  options: ArtiosPottsOptions,
): { sequences: ReturnType<typeof buildArtiosPottsSequenceSet>; track: ResidueTrackArtifact } {
  const primaryJob = jobs[0];
  const primary = results[0];
  const allCandidates = results.flatMap((r) => r.candidates);
  const lockedChains = new Set<string>();
  for (const r of results) for (const ch of r.lockedChains) lockedChains.add(ch);

  const runHash = artiosPottsRunHash(primaryJob.parent, options, primary.fixedProteinIndices);

  const multi = jobs.length > 1;
  const sequences = buildArtiosPottsSequenceSet(
    multi ? structure : primaryJob.parent,
    allCandidates,
    options,
    [...lockedChains],
    runHash,
    multi
      ? {
          sourceStructureIds: jobs.map((j) => j.parent.id),
          collectionKind: 'independent-candidates',
        }
      : {},
  );

  const track = buildArtiosPottsTrack(
    primaryJob.parent,
    primary.input.proteinResidues,
    primary.best,
    options,
    runHash,
  );
  return { sequences, track };
}

export {};
