/// <reference lib="webworker" />
// Layer 3 Plugin Worker — proton-pottsmpnn (Proton-PottsMPNN; christian-creator/ProtonPottsMPNN,
// MIT, with PottsMPNN from IPD rc-foundry under BSD-3-Clause; WebGPU port
// ramithuh/proton-pottsmpnn-webgpu)
//
// Two stages per structure:
//   1. The Potts tables `(h, J, edgeIdx)` from the backbone-only MPNN encoder and the `etab_out`
//      head over the 30-token v6 protonation vocabulary. On WebGPU this is
//      `createProtonPottsWebGPU()` (vendor/proton-pottsmpnn/webgpu); on the CPU fallback it is
//      `createProtonPottsCPU()`. Both handles expose the same analysis surface.
//   2. Either the pH-selective block-descent optimizer or the closed-form energy readout, both of
//      which are the same `vendor/proton-pottsmpnn/shared` code on either backend (they are
//      index-heavy and sequential rather than dense arithmetic, so they stay on the host).
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
import { PROTON_POTTSMPNN_WEIGHTS_BASE_URL } from './manifest';
import {
  buildCentreReports,
  buildProtonPottsCandidate,
  buildProtonPottsInput,
  buildProtonPottsSequenceSet,
  buildProtonPottsTrack,
  chainLayoutOf,
  formatCentreReport,
  inputPdbText,
  orderProtonPottsCandidates,
  pinCentres,
  planProtonPottsJobs,
  protonPottsObjective,
  protonPottsOptionsFromParams,
  protonPottsRunHash,
  resolveDesignableSet,
  resolveProtonPottsAlphabet,
  resolveProtonPottsCentres,
  resolveProtonPottsConstraints,
  resolveProtonationOverrides,
  siteEnergies,
  type ProtonPottsCentreReport,
  type ProtonPottsCpuFallbackReason,
  type ProtonPottsInput,
  type ProtonPottsJob,
  type ProtonPottsOptions,
  type ProtonPottsPrecision,
  type ProtonPottsSample,
  type ProtonPottsScore,
  type ResolvedCentre,
  type VendorCentreIndices,
} from './jobs';

declare const self: DedicatedWorkerGlobalScope;

const VENDOR_VERSION = 1;

type Backend = 'webgpu' | 'cpu';

const CPU_FALLBACK_WARNING: Record<ProtonPottsCpuFallbackReason, string> = {
  unavailable: 'Proton-PottsMPNN ran on the CPU JS fallback (WebGPU unavailable).',
  'device-lost': 'Proton-PottsMPNN fell back to the CPU JS engine after a WebGPU device loss.',
};

/* eslint-disable @typescript-eslint/no-explicit-any */

/** A `resolveCentres()` result: the centre plus its vocabulary indices. */
type VendorCentre = VendorCentreIndices;

/** The subset of `vendor/proton-pottsmpnn/shared/index.js` this worker calls directly. */
interface SharedModule {
  /** Conditional energies `[V]` at one position — the exact function `selectiveGapOf` uses. */
  pottsCandidateEnergiesAt(
    tables: unknown,
    seq: ArrayLike<number>,
    i: number,
    out?: Float64Array,
  ): Float64Array;
  /** Designable positions ordered by best kNN rank to any centre. */
  knnRank(tables: unknown, centres: readonly VendorCentre[], designable: readonly number[]): number[];
  makeRng(seed: number): () => number;
  canonicalSequence(tokens: ArrayLike<number>): string;
  tokenNames(tokens: ArrayLike<number>): string[];
  V6_TOKENS: readonly string[];
}

/** One structure's Potts tables plus the shapes the analysis helpers need. */
interface PottsRun {
  structure: any;
  tables: unknown;
  L: number;
  K: number;
  V: number;
  energy: number;
}

/** A backend that turns one PDB text into Potts tables and runs the shared analysis on them. */
interface PottsBackend {
  backend: Backend;
  structureFromPDB(text: string, options?: { overrides?: Record<number, string> }): any;
  run(structure: any): Promise<PottsRun>;
  resolveCentres(
    structure: any,
    specs: ReadonlyArray<{ position: number; protonationType: string }>,
  ): VendorCentre[];
  selectiveGap(
    out: PottsRun,
    centres: readonly VendorCentre[],
    seq?: ArrayLike<number>,
  ): { centres: Array<{ position: number; gap: number }>; gapSum: number };
  candidateEnergies(out: PottsRun, seq: ArrayLike<number>): ArrayLike<number>;
  centreNeighbourhood(
    out: PottsRun,
    centres: readonly VendorCentre[],
    o: { perCentre: number; max: number },
  ): number[];
  optimize(out: PottsRun, options: Record<string, unknown>): any;
  destroy(): void;
}

let sharedMod: SharedModule | null = null;
let gpuMod: any = null;
let cpuMod: any = null;
let engine: PottsBackend | null = null;
let loadedKey: string | null = null;
// Resilient GPUDevice injected into the engine: kept across precision switches, destroyed on dispose.
let gpu: { device: GPUDevice; hasF16: boolean } | null = null;
const deviceLostGuard = createDeviceLostGuard();

function vendorUrl(rel: string): string {
  return new URL(`/vendor/proton-pottsmpnn/${rel}?v=${VENDOR_VERSION}`, self.location.origin).href;
}

async function ensureShared(ctx: WorkerRunContext): Promise<SharedModule> {
  if (!sharedMod) {
    ctx.progress('init', 0.02, 'Loading Proton-PottsMPNN featurizer & Potts analysis...');
    sharedMod = (await import(/* @vite-ignore */ vendorUrl('shared/index.js'))) as SharedModule;
  }
  return sharedMod;
}

async function ensureGpu(): Promise<{ device: GPUDevice; hasF16: boolean }> {
  if (gpu) return gpu;
  // `shader-f16` is never required: the FP16 bundle is unpacked to f32 inside the GEMM.
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
  const envBaseUrl = (import.meta.env.VITE_PROTON_POTTSMPNN_WEIGHTS_URL as string | undefined)?.trim();
  return (override || envBaseUrl || PROTON_POTTSMPNN_WEIGHTS_BASE_URL).replace(/\/+$/, '');
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
        `Proton-PottsMPNN weights (${tag}): ${(loaded / 2 ** 20).toFixed(1)} / ${(total / 2 ** 20).toFixed(1)} MiB`,
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

/** Wrap one vendored handle (CPU or WebGPU) in the uniform backend surface. */
function wrapHandle(handle: any, backend: Backend): PottsBackend {
  return {
    backend,
    structureFromPDB: (text, options) => handle.structureFromPDB(text, options),
    async run(structure) {
      const out = await handle.run(structure);
      return out as PottsRun;
    },
    resolveCentres: (structure, specs) => handle.resolveCentres(structure, specs) as VendorCentre[],
    selectiveGap: (out, centres, seq) => handle.selectiveGap(out, centres, seq),
    candidateEnergies: (out, seq) => handle.candidateEnergies(out, seq),
    centreNeighbourhood: (out, centres, o) => handle.centreNeighbourhood(out, centres, o),
    optimize: (out, options) => handle.optimize(out, options),
    destroy: () => handle.destroy(),
  };
}

async function buildBackend(
  ctx: WorkerRunContext,
  baseUrl: string,
  precision: ProtonPottsPrecision,
  backend: Backend,
  fallbackReason?: ProtonPottsCpuFallbackReason,
): Promise<PottsBackend> {
  const tag = `${precision.toUpperCase()}${backend === 'cpu' ? ' · CPU fallback' : ''}`;

  const loadOnce = async (retried: boolean): Promise<PottsBackend> => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const read = makeBundleReader(ctx, tag, (url, fromCache) => {
      fetchedUrls.push(url);
      if (fromCache) anyFromCache = true;
    });
    ctx.progress('weights', 0.06, `Fetching Proton-PottsMPNN ${tag} weights...`);
    try {
      if (backend === 'webgpu') {
        if (!gpuMod) {
          ctx.progress('init', 0.04, 'Loading Proton-PottsMPNN WebGPU modules...');
          gpuMod = await import(/* @vite-ignore */ vendorUrl('webgpu/index.js'));
        }
        const { device } = await ensureGpu();
        ctx.progress('init', 0.5, 'Uploading weights & compiling WGSL pipelines...');
        const handle = await gpuMod.createProtonPottsWebGPU({ baseUrl, precision, device, read });
        return wrapHandle(handle, 'webgpu');
      }

      if (!cpuMod) {
        ctx.progress(
          'init',
          0.04,
          fallbackReason === 'device-lost'
            ? 'WebGPU device lost — loading Proton-PottsMPNN CPU fallback...'
            : 'WebGPU unavailable — loading Proton-PottsMPNN CPU fallback...',
        );
        cpuMod = await import(/* @vite-ignore */ vendorUrl('cpu/index.js'));
      }
      const handle = await cpuMod.createProtonPottsCPU({ baseUrl, precision, read });
      ctx.progress('init', 0.92, 'Preparing Proton-PottsMPNN CPU engine...');
      return wrapHandle(handle, 'cpu');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        (err instanceof SyntaxError ||
          /checksum|truncated|manifest|mismatch|json|syntaxerror|unexpected token|safetensors/i.test(msg))
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
  baseUrl: string,
  precision: ProtonPottsPrecision,
  backend: Backend,
  fallbackReason?: ProtonPottsCpuFallbackReason,
): Promise<PottsBackend> {
  const key = `${baseUrl}|${precision}|${backend}`;
  // `gpu` is cleared when its device is lost, even mid-load, so a cache hit always runs live.
  if (engine && loadedKey === key && (backend === 'cpu' || gpu)) return engine;
  releaseEngine();
  engine = await buildBackend(ctx, baseUrl, precision, backend, fallbackReason);
  loadedKey = key;
  return engine;
}

// ---------------------------------------------------------------------------
// Closed-form readouts on a prepared Potts table
// ---------------------------------------------------------------------------

/**
 * Per-centre `e_P` / `mean_d e_D` / gap with every centre pinned protonated, so each gap is
 * measured in the presence of the others — the exact arithmetic of `shared :: selectiveGapOf`
 * (same `pottsCandidateEnergiesAt` row, same FP64 accumulation), decomposed for reporting.
 */
function centreReports(
  shared: SharedModule,
  out: PottsRun,
  seq: ArrayLike<number>,
  centres: readonly VendorCentre[],
  meta: readonly ResolvedCentre[],
): ProtonPottsCentreReport[] {
  if (centres.length === 0) return [];
  const pinned = pinCentres(seq, centres);
  const row = new Float64Array(out.V);
  return buildCentreReports(meta, centres, (position) =>
    shared.pottsCandidateEnergiesAt(out.tables, pinned, position, row),
  );
}

function scoreOf(
  shared: SharedModule,
  backend: PottsBackend,
  out: PottsRun,
  seq: ArrayLike<number>,
  centres: readonly VendorCentre[],
  meta: readonly ResolvedCentre[],
  globalEnergy: number,
): ProtonPottsScore {
  const reports = centreReports(shared, out, seq, centres, meta);
  // The headline gap sum comes from the runtime's own `selectiveReport`; `centreReports` only
  // splits each of those gaps into its `e_P` and `mean_d e_D` halves for display.
  const gap = centres.length > 0 ? backend.selectiveGap(out, centres, seq) : null;
  return {
    globalEnergy,
    energyPerResidue: globalEnergy / Math.max(1, out.L),
    selectiveGapSum: gap ? gap.gapSum : null,
    centres: reports,
    perResidueEnergy: siteEnergies(backend.candidateEnergies(out, seq), seq, out.L, out.V),
  };
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface JobResult {
  candidates: SequenceCandidate[];
  lockedChains: string[];
  fixedPositions: Set<number>;
  designable: number[];
  best: ProtonPottsSample;
  input: ProtonPottsInput;
  warnings: string[];
}

createWorkerHarness({
  async run(ctx) {
    const structure = ctx.inputs?.structure as StructureArtifact | undefined;
    if (!structure || structure.kind !== 'StructureArtifact') {
      throw new Error(
        'Proton-PottsMPNN requires an upstream StructureArtifact on its structure input.',
      );
    }
    const p = ctx.params ?? {};
    const options = protonPottsOptionsFromParams(p);
    // Fail fast on an impossible alphabet before any weight download.
    const alphabet = resolveProtonPottsAlphabet(options.omitAAs, options.titratableTokens);
    const baseUrl = weightsBaseUrl(p);
    const jobs = planProtonPottsJobs(structure);

    const shared = await ensureShared(ctx);

    let backend: Backend = webGpuAvailable() ? 'webgpu' : 'cpu';
    let cpuFallbackReason: ProtonPottsCpuFallbackReason | undefined =
      backend === 'cpu' ? 'unavailable' : undefined;
    let gpuRecoveries = 0;

    const acquire = async (): Promise<PottsBackend> => {
      try {
        return await ensureEngine(ctx, baseUrl, options.precision, backend, cpuFallbackReason);
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
        cpuFallbackReason =
          isWebGpuDeviceLostError(err) || gpuRecoveries > 0 ? 'device-lost' : 'unavailable';
        return await ensureEngine(ctx, baseUrl, options.precision, backend, cpuFallbackReason);
      }
    };
    let active = await acquire();

    /** Potts tables for one structure, retrying once on a fresh device then on the CPU. */
    const computeTables = async (
      pdbText: string,
      overrides: Record<number, string>,
    ): Promise<{ out: PottsRun; backend: PottsBackend }> => {
      for (;;) {
        try {
          const structureTensors = active.structureFromPDB(pdbText, { overrides });
          return { out: await active.run(structureTensors), backend: active };
        } catch (err) {
          if (active.backend !== 'webgpu' || !isWebGpuDeviceLostError(err)) throw err;
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
    const totalUnits = jobs.length * (options.mode === 'score' ? 1 : Math.max(1, options.numSequences));
    let completedUnits = 0;

    for (const job of jobs) {
      ctx.signal.throwIfAborted();
      results.push(
        await runJob(ctx, shared, job, jobs.length, options, alphabet.forbiddenTokens, computeTables, () => {
          completedUnits++;
          return Math.min(0.99, completedUnits / Math.max(1, totalUnits));
        }),
      );
    }

    const degradedWarning =
      active.backend === 'cpu' && cpuFallbackReason ? CPU_FALLBACK_WARNING[cpuFallbackReason] : null;

    const outputs = assembleOutputs(structure, jobs, results, options);
    const notes = [
      ...new Set(results.flatMap((r) => r.warnings).concat(degradedWarning ? [degradedWarning] : [])),
    ];
    const headline =
      options.mode === 'score'
        ? `Scored ${results.length} structure(s) with the v6 Potts model`
        : `Designed ${outputs.sequences.candidates.length} sequence(s)`;
    ctx.progress('sample', 1, notes.length > 0 ? `${headline} — ${notes.join(' ')}` : headline);
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

/** Featurize one backbone, compute its Potts tables, then score or design. */
async function runJob(
  ctx: WorkerRunContext,
  shared: SharedModule,
  job: ProtonPottsJob,
  jobCount: number,
  options: ProtonPottsOptions,
  forbiddenTokens: readonly string[],
  computeTables: (
    pdbText: string,
    overrides: Record<number, string>,
  ) => Promise<{ out: PottsRun; backend: PottsBackend }>,
  tickUnit: () => number,
): Promise<JobResult> {
  const { parent } = job;
  const pdbText = inputPdbText(parent);
  const input = buildProtonPottsInput(pdbText);
  const residues = input.residues;
  const warnings = [...input.diagnostics.warnings];

  const overrideSpec = resolveProtonationOverrides(residues, options.protonationOverrides);
  warnings.push(...overrideSpec.warnings);
  const centreSpec = resolveProtonPottsCentres(residues, options.protonationCentres);
  warnings.push(...centreSpec.warnings);

  const jobTag = jobCount > 1 ? `Backbone ${job.designIndex + 1}/${jobCount} · ` : '';
  ctx.progress('infer', 0.93, `${jobTag}Proton-PottsMPNN Potts tables (${residues.length} residues)...`);
  const { out, backend } = await computeTables(pdbText, overrideSpec.overrides);

  // The vendored parser and this plugin's own reader must agree position for position, or every
  // residue label, override and centre would be off by the difference.
  if (out.L !== residues.length) {
    throw new Error(
      `Proton-PottsMPNN: the runtime featurized ${out.L} residues but this plugin resolved ${residues.length}; the PDB reader and the vendored parser disagree.`,
    );
  }
  for (let i = 0; i < out.L; i++) {
    if (out.structure.resSeq[i] !== residues[i].number) {
      throw new Error(
        `Proton-PottsMPNN: residue ${i} is ${residues[i].chain}${residues[i].number} here but resSeq ${out.structure.resSeq[i]} in the runtime; the PDB reader and the vendored parser disagree.`,
      );
    }
  }

  const centres = backend.resolveCentres(out.structure, centreSpec.specs);
  const nativeTokens = Int32Array.from(out.structure.tokens as ArrayLike<number>);
  const layout = chainLayoutOf(residues);

  if (options.mode === 'score') {
    const score = scoreOf(
      shared,
      backend,
      out,
      nativeTokens,
      centres,
      centreSpec.centres,
      out.energy,
    );
    const sample: ProtonPottsSample = {
      sequence: shared.canonicalSequence(nativeTokens),
      tokenNames: shared.tokenNames(nativeTokens),
      score,
      sampleIndex: 0,
      seed: options.seed,
      rounds: null,
      converged: null,
      nChanges: null,
      objective: null,
      mutations: 0,
      chainIds: layout.chainIds,
      chainLengths: layout.chainLengths,
    };
    const frac = tickUnit();
    ctx.progress(
      'sample',
      frac,
      `${jobTag}E_global ${score.globalEnergy.toFixed(3)}${
        score.centres.length > 0
          ? ` · ${score.centres.map(formatCentreReport).join(' · ')}`
          : ' · no pH centre declared'
      }`,
    );
    return {
      candidates: [
        buildProtonPottsCandidate(sample, parent, options, idPrefix(job, 'native'), extraOf(job)),
      ],
      lockedChains: [],
      fixedPositions: new Set<number>(),
      designable: [],
      best: sample,
      input,
      warnings,
    };
  }

  // --- design -------------------------------------------------------------
  const { fixedPositions, lockedChains } = resolveProtonPottsConstraints(
    residues,
    options.fixedPositions,
    parent,
  );
  const centrePositions = centres.map((c) => c.position);
  const useShell = options.designScope === 'centre-shell' && centres.length > 0;
  if (options.designScope === 'centre-shell' && centres.length === 0) {
    warnings.push(
      'Design scope is the centre shell but no pH centre is declared, so every residue stays designable.',
    );
  }
  const candidates = useShell
    ? backend.centreNeighbourhood(out, centres, { perCentre: options.shellPerCentre, max: 0 })
    : Array.from({ length: out.L }, (_, i) => i);
  const designable = resolveDesignableSet({
    candidates,
    fixed: fixedPositions,
    centres: centrePositions,
    max: options.maxDesignable,
    // Rank-capping uses the runtime's own kNN ranking, exactly as `centreNeighbourhood` does.
    ...(centres.length > 0
      ? { rank: (positions: readonly number[]) => shared.knnRank(out.tables, centres, positions) }
      : {}),
  });
  if (designable.length === 0) {
    throw new Error(
      'Proton-PottsMPNN has no designable residue left: every position is a pH centre, fixed by fixedPositions, or removed by an inherited lock.',
    );
  }

  let lambda = options.lambda;
  if (centres.length === 0) {
    if (lambda >= 1) {
      throw new Error(
        'Proton-PottsMPNN: λ = 1 is pure pH selectivity but no pH centre is declared, which leaves the descent with no objective. Add a pH centre (e.g. "A42=HIS-P") or lower λ.',
      );
    }
    if (lambda > 0) {
      warnings.push(
        `With no pH centre the selectivity term is identically zero, so λ = ${lambda} reduces to stability-only design (and, above Temp 0, an effective temperature of Temp/(1−λ)).`,
      );
    }
  }

  let plannedSequences = options.numSequences;
  if (options.temperature <= 0 && plannedSequences > 1) {
    warnings.push(
      `Block descent at Temp 0 is an exact argmin and therefore deterministic, so 1 sequence was designed instead of ${plannedSequences}; raise Temp above 0 to sample distinct designs.`,
    );
    plannedSequences = 1;
  }

  const samples: ProtonPottsSample[] = [];
  let lastProgress = 0;
  for (let i = 0; i < plannedSequences; i++) {
    ctx.signal.throwIfAborted();
    const seed = options.seed + i;
    const result = backend.optimize(out, {
      centres,
      designable,
      lambda,
      blockSize: options.blockSize,
      // The runtime switches on `=== 'block'`, so any other string selects the single-mutation scales.
      zscaleMode: options.zscaleMode === 'block' ? 'block' : 'single-mutation',
      sweepOrder: options.sweepOrder,
      temperature: options.temperature,
      maxRounds: options.maxRounds,
      forbiddenTokens: [...forbiddenTokens],
      rng: shared.makeRng(seed),
      // Block descent is a long synchronous loop; this is the only place it yields a cancel check.
      onStep: (step: { round: number; position: number }) => {
        ctx.signal.throwIfAborted();
        const now = Date.now();
        if (now - lastProgress < 250) return;
        lastProgress = now;
        ctx.progress(
          'sample',
          Math.min(0.99, (i + step.round / (options.maxRounds + 1)) / Math.max(1, plannedSequences)),
          `${jobTag}Descent ${i + 1}/${plannedSequences} · sweep ${step.round}/${options.maxRounds} · position ${step.position}`,
        );
      },
    });
    const tokens = Int32Array.from(result.tokens as ArrayLike<number>);
    const score = scoreOf(
      shared,
      backend,
      out,
      tokens,
      centres,
      centreSpec.centres,
      result.pottsEnergy,
    );
    let mutations = 0;
    for (let q = 0; q < tokens.length; q++) if (tokens[q] !== nativeTokens[q]) mutations++;
    samples.push({
      sequence: result.sequence as string,
      tokenNames: result.tokenNames as string[],
      score,
      sampleIndex: i,
      seed,
      rounds: result.rounds ?? null,
      converged: result.converged ?? null,
      nChanges: result.nChanges ?? null,
      objective: protonPottsObjective(
        { pottsEnergy: result.pottsEnergy, selectiveEnergy: result.selectiveEnergy },
        result.weights,
      ),
      mutations,
      chainIds: layout.chainIds,
      chainLengths: layout.chainLengths,
    });
    const frac = tickUnit();
    ctx.progress(
      'sample',
      frac,
      `${jobTag}Design ${i + 1}/${plannedSequences} · ${designable.length} designable · ${mutations} mutation(s) · E ${result.pottsEnergy.toFixed(2)}${
        centres.length > 0 ? ` · ΔE ${result.selectiveEnergy.toFixed(3)}` : ''
      }`,
    );
  }

  const ordered = orderProtonPottsCandidates(samples);
  return {
    candidates: ordered.map((sample) =>
      buildProtonPottsCandidate(sample, parent, options, idPrefix(job, 'd'), extraOf(job)),
    ),
    lockedChains,
    fixedPositions,
    designable,
    best: ordered[0],
    input,
    warnings,
  };
}

function idPrefix(job: ProtonPottsJob, kind: string): string {
  return job.designKey
    ? `d${job.designIndex + 1}-proton-pottsmpnn-${kind}`
    : `proton-pottsmpnn-${kind}`;
}

function extraOf(job: ProtonPottsJob): Partial<SequenceCandidate> {
  if (!job.designKey) return {};
  return { designKey: job.designKey, designIndex: job.designIndex };
}

function assembleOutputs(
  structure: StructureArtifact,
  jobs: ProtonPottsJob[],
  results: JobResult[],
  options: ProtonPottsOptions,
): { sequences: ReturnType<typeof buildProtonPottsSequenceSet>; track: ResidueTrackArtifact } {
  const primaryJob = jobs[0];
  const primary = results[0];
  const allCandidates = results.flatMap((r) => r.candidates);
  const lockedChains = new Set<string>();
  for (const r of results) for (const ch of r.lockedChains) lockedChains.add(ch);

  const runHash = protonPottsRunHash(
    primaryJob.parent,
    options,
    primary.fixedPositions,
    primary.designable,
  );

  const multi = jobs.length > 1;
  const sequences = buildProtonPottsSequenceSet(
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

  const track = buildProtonPottsTrack(
    primaryJob.parent,
    primary.input.residues,
    primary.best,
    options,
    runHash,
  );
  return { sequences, track };
}

export {};
