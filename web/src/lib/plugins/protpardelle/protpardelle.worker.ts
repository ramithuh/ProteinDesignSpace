/// <reference lib="webworker" />
// Layer 3 Plugin Worker — protpardelle (ramithuh/protpardelle-webgpu, MIT code / CC-BY-4.0 weights)
// Runs ProtPardelle-1c all-atom & backbone EDM diffusion + optional MiniMPNN sequence co-design.
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress, and disposal.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { assertSafeStepCount, formatLowStepGeometryWarning } from '$lib/engine/step-policy';
import { makeSequenceKey } from '$lib/engine/collections';
import { buildMotifLineage } from '$lib/engine/conditioning';
import { createDeviceLostGuard, requestResilientWebGpuDevice } from '$lib/engine/webgpu-device';
import {
  collectOwnedTransferables,
  createWorkerHarness,
  fnv1a,
  type WorkerRunContext,
} from '$lib/kernel';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { planProtpardelleMotifPlacement } from '$lib/protein-ops/motif-placement';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/protpardelle-webgpu/resolve/main';

type Precision = 'int8' | 'fp16';
type TaskMode = 'unconditional' | 'motif' | 'partial';
type MotifGuidance = 'replacement' | 'replacement+reconstruction' | 'crop-only';

const VALID_CHECKPOINTS = new Set([
  'cc91_epoch383',
  'cc91_tip_epoch480',
  'cc58_epoch416',
  'cc94_epoch3100',
  'cc95_epoch3490',
  'bb81_epoch450',
]);

const ALLATOM_CHECKPOINTS = new Set([
  'cc91_epoch383',
  'cc91_tip_epoch480',
  'cc94_epoch3100',
]);

const TIP_CHECKPOINTS = new Set(['cc91_tip_epoch480']);

function normalizePrecision(raw: unknown): Precision {
  if (raw == null || String(raw).trim() === '') return 'int8';
  const val = String(raw).trim().toLowerCase();
  if (val === 'fp16' || val === 'int8') return val;
  throw new Error(`Unsupported ProtPardelle precision "${String(raw)}" (supported: int8, fp16).`);
}

function normalizeCheckpoint(raw: unknown): string {
  const val = String(raw ?? 'cc91_epoch383').trim();
  return VALID_CHECKPOINTS.has(val) ? val : 'cc91_epoch383';
}

function normalizeTaskMode(raw: unknown): TaskMode {
  const val = String(raw ?? 'unconditional').trim().toLowerCase();
  if (val === 'motif' || val === 'partial') return val;
  return 'unconditional';
}

function normalizeMotifGuidance(raw: unknown): MotifGuidance {
  const val = String(raw ?? 'replacement').trim().toLowerCase();
  if (val === 'replacement+reconstruction' || val === 'crop-only') return val;
  return 'replacement';
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ppWebgpuMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ppInputsMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ppStructMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ppSamplerMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let loadedKey: string | null = null;
let lastMiniMpnnUrls: string[] = [];
// Resilient GPUDevice injected into the engine: kept across checkpoint/precision switches, destroyed only on dispose.
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
    // The engine's buffers and pipelines died with the device: the next run rebuilds it on a fresh one.
    loadedKey = null;
  });
  return acquired;
}

function makeSeededReplay(seed: number) {
  let state = seed >>> 0;
  const nextRand = () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const replay = new ppSamplerMod.Replay();
  replay.take = (kind: string, size: number) => {
    const out = new Float32Array(size);
    if (kind === 'numpy_rand' || kind === 'categorical_uniform') {
      for (let i = 0; i < size; i++) out[i] = nextRand();
    } else {
      for (let i = 0; i < size; i += 2) {
        const u1 = Math.max(1e-15, nextRand());
        const u2 = nextRand();
        const radius = Math.sqrt(-2 * Math.log(u1));
        const angle = 2 * Math.PI * u2;
        out[i] = radius * Math.cos(angle);
        if (i + 1 < size) out[i + 1] = radius * Math.sin(angle);
      }
    }
    return out;
  };
  return replay;
}

async function ensureModules(ctx: WorkerRunContext) {
  if (ppWebgpuMod && ppInputsMod && ppStructMod && ppSamplerMod) return;
  ctx.progress('init', 0.02, 'Loading ProtPardelle-1c WebGPU modules...');
  const base = `${self.location.origin}/vendor/protpardelle`;
  const [webgpuMod, inputsMod, structMod, samplerMod] = await Promise.all([
    import(/* @vite-ignore */ `${base}/webgpu/index.js`),
    import(/* @vite-ignore */ `${base}/shared/inputs.js`),
    import(/* @vite-ignore */ `${base}/shared/structure.js`),
    import(/* @vite-ignore */ `${base}/shared/sampler.js`),
  ]);
  ppWebgpuMod = webgpuMod;
  ppInputsMod = inputsMod;
  ppStructMod = structMod;
  ppSamplerMod = samplerMod;
}

async function ensureEngine(ctx: WorkerRunContext, checkpoint: string, precision: Precision) {
  const key = `${checkpoint}:${precision}`;
  // `gpu` is cleared when its device is lost, even mid-load, so a cache hit always runs on a live device.
  if (engine && loadedKey === key && gpu) return engine;

  if (engine) {
    try {
      engine.destroy();
    } catch {
      /* ignore */
    }
    engine = null;
    loadedKey = null;
  }

  await ensureModules(ctx);

  const envBaseUrl = (import.meta.env.VITE_PROTPARDELLE_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;
  const { device, hasF16 } = await ensureGpu();

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (u.includes('minimpnn')) lastMiniMpnnUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress(
      'weights',
      0.06,
      `Fetching ProtPardelle-1c (${checkpoint} · ${precision.toUpperCase()}) weights...`,
    );

    try {
      return await ppWebgpuMod.createProtpardelleEngine({
        baseUrl,
        variant: checkpoint,
        precision,
        device,
        useF16: hasF16,
        cache: 'none',
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
          ctx.progress(
            'weights',
            Math.min(0.45, 0.06 + frac * 0.39),
            `ProtPardelle weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
          );
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

  engine = await loadOnce(false);
  loadedKey = key;
  return engine;
}

createWorkerHarness({
  async run(ctx) {
    const p = ctx.params ?? {};
    const taskMode = normalizeTaskMode(p.taskMode);
    const checkpoint = normalizeCheckpoint(p.checkpoint);
    const precision = normalizePrecision(p.precision);
    const length = Math.max(20, Math.min(256, Math.round(Number(p.length ?? 80))));
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));
    const numSteps = assertSafeStepCount('protpardelle', { ...p, taskMode, numSteps: p.numSteps }, 200);
    const sidechainMode = Boolean(p.sidechainMode ?? true) && ALLATOM_CHECKPOINTS.has(checkpoint);
    const motifSelection = String(p.motifSelection ?? '').trim();
    const motifGuidance = normalizeMotifGuidance(p.motifGuidance);
    const rewindSteps = Math.max(5, Math.min(numSteps - 1, Math.round(Number(p.rewindSteps ?? 40))));
    const stepScale = Math.max(0.5, Math.min(2.0, Number(p.stepScale ?? 1.2)));
    const temperature = Math.max(0.05, Math.min(2.0, Number(p.temperature ?? 0.1)));
    const seed = Math.max(0, Math.round(Number(p.seed ?? 42)));

    const upstreamStruct = ctx.inputs?.structure as StructureArtifact | undefined;
    if ((taskMode === 'motif' || taskMode === 'partial') && !upstreamStruct) {
      throw new Error(
        `ProtPardelle-1c in "${taskMode}" mode requires a connected upstream structure.`,
      );
    }
    if (taskMode === 'motif' && motifGuidance === 'crop-only' && checkpoint === 'bb81_epoch450') {
      throw new Error(
        'ProtPardelle backbone-only checkpoint (bb81_epoch450) does not support crop-only motif conditioning; use "replacement" guidance or a crop-conditioned checkpoint (e.g. cc91_epoch383).',
      );
    }

    const activeEngine = await ensureEngine(ctx, checkpoint, precision);
    const chainGap = Number(activeEngine.config?.data?.chain_residx_gap ?? 200);

    // Parse optional upstream template structure for motif scaffolding or partial diffusion
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let parsedTemplate: any = null;
    let sourceIndices: number[] = [];
    let targetIndices: number[] = [];
    let sampleLen = length;

    if (upstreamStruct && (taskMode === 'motif' || taskMode === 'partial')) {
      const rawText = upstreamStruct.cifText ?? upstreamStruct.pdbText;
      const format = upstreamStruct.cifText ? 'cif' : 'pdb';
      parsedTemplate = ppStructMod.parseStructure(rawText, {
        format,
        chainResidueGap: chainGap,
      });
      const tplLen = parsedTemplate.aatype.length;
      if (taskMode === 'partial') {
        sampleLen = tplLen;
      } else {
        const rawSourceIndices = motifSelection
          ? (ppInputsMod.selectPositions(motifSelection, parsedTemplate) as number[])
          : undefined;
        if (motifSelection && (!rawSourceIndices || rawSourceIndices.length === 0)) {
          throw new Error(
            `Motif selection "${motifSelection}" matched 0 residues in "${upstreamStruct.label}".`,
          );
        }
        const plan = planProtpardelleMotifPlacement(parsedTemplate, {
          targetLength: length,
          sourceIndices: rawSourceIndices,
        });
        sourceIndices = plan.sourceIndices;
        targetIndices = plan.targetIndices;
        sampleLen = plan.effectiveLength;
      }
    }

    const layout =
      taskMode === 'partial' && parsedTemplate
        ? {
            seqMask: parsedTemplate.seqMask,
            residueIndex: parsedTemplate.residueIndex,
            chainIndex: parsedTemplate.chainIndex,
          }
        : ppInputsMod.chainLayout([sampleLen], chainGap);

    const items: StructureArtifact[] = [];
    const seqCandidates: SequenceCandidate[] = [];
    const runHash = fnv1a(
      JSON.stringify({
        taskMode,
        checkpoint,
        precision,
        sampleLen,
        numDesigns,
        numSteps,
        sidechainMode,
        motifSelection,
        motifGuidance: taskMode === 'motif' ? motifGuidance : null,
        rewindSteps,
        stepScale,
        temperature,
        seed,
        upstreamId: upstreamStruct?.id ?? null,
      }),
    );

    const effectiveStepsRun = taskMode === 'partial' ? Math.min(numSteps, rewindSteps) : numSteps;
    // Extract native motif Cα directly from parsedTemplate.coords (atom index 1 = CA) so interleaved
    // multi-chain templates never suffer index misalignment between parsedTemplate and upstreamStruct.
    let templateCaCoords: Float32Array | undefined;
    if (parsedTemplate?.coords && parsedTemplate?.aatype?.length) {
      const tplN = parsedTemplate.aatype.length as number;
      templateCaCoords = new Float32Array(tplN * 3);
      for (let i = 0; i < tplN; i++) {
        templateCaCoords[i * 3] = parsedTemplate.coords[i * 111 + 3];
        templateCaCoords[i * 3 + 1] = parsedTemplate.coords[i * 111 + 4];
        templateCaCoords[i * 3 + 2] = parsedTemplate.coords[i * 111 + 5];
      }
    }
    const motifLineage =
      taskMode === 'motif' && targetIndices.length > 0
        ? buildMotifLineage(
            targetIndices,
            sourceIndices,
            templateCaCoords ??
              (upstreamStruct && upstreamStruct.caCoords?.length >= upstreamStruct.length * 3
                ? upstreamStruct.caCoords
                : undefined),
          )
        : undefined;

    for (let d = 0; d < numDesigns; d++) {
      if (ctx.signal.aborted) throw new Error('Workflow run cancelled');
      const sampleInput: Record<string, unknown> = {
        ...layout,
        numSteps,
        stepScale,
        temperature,
        sidechainMode,
        replay: makeSeededReplay(seed + d),
        sChurn: taskMode === 'partial' ? 0 : Math.min(50, numSteps * 0.25),
        ...(sidechainMode ? { jumpSteps: true } : {}),
        ...(TIP_CHECKPOINTS.has(checkpoint) && taskMode === 'motif'
          ? { tipAtomConditioning: true }
          : {}),
        ...(taskMode === 'motif' && parsedTemplate
          ? {
              motif: {
                structure: parsedTemplate,
                sourceIndices,
                targetIndices,
                strategy: TIP_CHECKPOINTS.has(checkpoint)
                  ? 'sidechain-tip'
                  : sidechainMode
                    ? 'backbone-sidechain'
                    : 'backbone',
              },
              conditioning: {
                crop: true,
                replacement: motifGuidance !== 'crop-only',
                reconstruction: motifGuidance === 'replacement+reconstruction',
              },
            }
          : {}),
        ...(taskMode === 'partial' && parsedTemplate
          ? {
              partial: {
                structure: parsedTemplate,
                rewindSteps,
                repack: sidechainMode,
              },
            }
          : {}),
      };

      const runSampleOnce = async (retriedMiniMpnn: boolean) => {
        try {
          return await activeEngine.sample(sampleInput, {
            signal: ctx.signal,
            onProgress: (prog: {
              phase?: string;
              step?: number;
              totalSteps?: number;
              loaded?: number;
              total?: number;
            }) => {
              const isWeights = prog.phase === 'weights';
              if (isWeights) {
                const mb = ((prog.loaded ?? 0) / 1024 / 1024).toFixed(1);
                const totMb = prog.total ? ` / ${(prog.total / 1024 / 1024).toFixed(1)} MB` : ' MB';
                ctx.progress('weights', 0.47, `Loading MiniMPNN (${precision.toUpperCase()}): ${mb}${totMb}`);
                return;
              }
              const step = prog.step ?? prog.loaded ?? 0;
              const tot = prog.totalSteps ?? prog.total ?? effectiveStepsRun;
              const stepFrac = tot > 0 ? Math.min(1, step / tot) : 0;
              const overall = 0.48 + ((d + stepFrac) / numDesigns) * 0.5;
              ctx.progress(
                'diffuse',
                Math.min(0.98, overall),
                `ProtPardelle design ${d + 1}/${numDesigns}: step ${step}/${tot}`,
              );
            },
          });
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          if (
            !retriedMiniMpnn &&
            lastMiniMpnnUrls.length > 0 &&
            /integrity|sha-?256|manifest|mismatch|corrupt/i.test(errMsg)
          ) {
            const toEvict = [...new Set(lastMiniMpnnUrls)];
            lastMiniMpnnUrls = [];
            await evictCachedUrls(toEvict);
            return await runSampleOnce(true);
          }
          throw err;
        }
      };
      const sampled = await runSampleOnce(false);

      const pdbText = ppStructMod.toPDB({
        ...layout,
        coords: sampled.coords,
        aatype: sampled.aatype,
        atomMask: sampled.atomMask,
      }) as string;

      const parsed = parseStructureText(pdbText);
      const geometry = computeGeometryStats(
        parsed.caCoords,
        parsed.chainIds,
        parsed.resNums,
        parsed.insCodes,
      );
      const stepGeomWarn = formatLowStepGeometryWarning('protpardelle', p, geometry);
      if (stepGeomWarn) console.warn(stepGeomWarn);
      const designKey = `design-${d + 1}`;
      const structId = `pp-${checkpoint}-${sampleLen}aa-d${d + 1}-${runHash}`;

      const artifact: StructureArtifact = {
        kind: 'StructureArtifact',
        id: structId,
        label: `ProtPardelle ${checkpoint} (${parsed.nResidues} aa${numDesigns > 1 ? ` · #${d + 1}` : ''})`,
        sourceModelId: 'protpardelle',
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
        atomGranularity: sidechainMode ? 'all-atom' : 'backbone4',
        bfactorSemantics: 'none',
        ...(motifLineage && motifLineage.motifIndices.length > 0
          ? {
              fixedResidueIndices: motifLineage.fixedResidueIndices,
              motifIndices: motifLineage.motifIndices,
            }
          : {}),
        ...(motifLineage?.nativeMotifCoords
          ? { nativeMotifCoords: motifLineage.nativeMotifCoords.map((c) => [...c]) }
          : {}),
        diagnostics: parsed.diagnostics,
        geometry,
        designKey,
        designIndex: d,
        seed: seed + d,
        provenance: {
          quantization: precision,
          diffusionStepsScheduled: numSteps,
          diffusionStepsRun: effectiveStepsRun,
          ...(stepGeomWarn ? { warnings: [stepGeomWarn] } : {}),
        },
      };
      items.push(artifact);

      const seqKey = makeSequenceKey(`protpardelle-${checkpoint}`, designKey, 0);
      seqCandidates.push({
        id: `pp-d${d + 1}-s1`,
        key: seqKey,
        sequenceKey: seqKey,
        sequence: parsed.sequence,
        chainLengths: parsed.diagnostics.chainLengths,
        score: null,
        globalScore: null,
        sourceModelId: sidechainMode ? 'protpardelle-minimpnn' : 'protpardelle',
        sourceStructureId: structId,
        designKey,
        designIndex: d,
        sampleIndex: 0,
        seed: seed + d,
      });
    }

    const primary = items[0];
    const batchHash = fnv1a(`${items.map((it) => it.id).join('|')}:${items.length}`);
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
            id: `pp-batch-${checkpoint}-${items.length}-${batchHash}`,
            label: `ProtPardelle ${checkpoint} (${items.length} designs · L=${primary.length})`,
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `pp-col-${checkpoint}-${batchHash}`,
              familyKey: `protpardelle-${checkpoint}-${batchHash}`,
              collectionKind: 'independent-candidates',
              requestedCount: numDesigns,
              complete: true,
              items,
              keys: items.map((it) => it.designKey ?? it.id),
              representativeSelection: repSelection,
            },
          }
        : primary;

    const sequencesOut: SequenceSet = {
      kind: 'SequenceSet',
      id: `pp-seqs-${checkpoint}-${runHash}`,
      parentStructureId: primary.id,
      sourceStructureId: primary.id,
      sourceStructureIds: items.map((it) => it.id),
      sourceModelId: sidechainMode ? 'protpardelle-minimpnn' : 'protpardelle',
      collectionKind: items.length > 1 ? 'independent-candidates' : undefined,
      candidates: seqCandidates,
    };

    const transferables = collectOwnedTransferables(
      items.flatMap((it) => [it.caCoords, it.backbone4Coords, it.observedBackboneMask]),
    );

    return {
      outputs: { structure: structureOut, sequences: sequencesOut },
      transferables,
    };
  },

  async dispose() {
    try {
      engine?.destroy();
    } catch {
      /* ignore */
    }
    engine = null;
    loadedKey = null;
    deviceLostGuard.destroy(gpu?.device);
    gpu = null;
  },
});

export {};
