/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping sokrypton/localfold
// Supports AlphaFold2 monomer (models 1–5), AlphaFold2-Multimer v3 (models 1–5),
// and all AF3-lineage models (RoseTTAFold3, Boltz-2, OpenBind-0, Protenix-v2,
// OpenDDE, Mol-1, IntelliFold-2, AF3) loaded directly from /vendor/localfold/.

import { atom37ToPdb, chainIdsFromLengths, parsePdbBackbone } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { normalizeLocalFoldChainScores } from '$lib/engine/conditioning';
import { esmfold2Chains, esmfold2ParamIssue } from '$lib/engine/localfold-esmfold2';
import { computeCandidateIpsae, loadEsmfold2 } from './localfold-esmfold2';
import { selectFoldWorkItems } from '$lib/engine/bio-capabilities';
import {
  clampLocalFoldRecycles,
  computeAf2MsaBudgets,
  createMsaSearchCache,
  resolveAf2Tolerance,
  resolveMsaForCandidate,
  selectLocalFoldRepresentative,
  type MmseqsApi,
} from '$lib/engine/localfold-msa';
import { fetchCachedBytes } from '$lib/engine/opfs-cache';
import { sha256Hex, type CollectionMember } from '$lib/engine/collections';
import { planLocalFoldFanout } from '$lib/engine/work-plan';
import type {
  CollectionItemOutcome,
  ExcludedCandidateRecord,
  FoldedCandidate,
  FoldedStructure,
  MsaMode,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const AF2_MODELS = new Set(['monomer', 'monomer-2', 'monomer-3', 'monomer-4', 'monomer-5']);
const AF2_MULTIMER_MODELS = new Set([
  'multimer',
  'multimer-2',
  'multimer-3',
  'multimer-4',
  'multimer-5',
]);
const ESMFOLD2_MODELS = new Set(['ef2-fast-300m', 'ef2-fast-600m']);
const AF3_MODELS = new Set([
  'rosettafold3',
  'boltz2',
  'openbind0',
  'protenix2',
  'opendde',
  'mol1',
  'intellifold2',
  'af3',
]);

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

function sequentialResNumsFromChainLengths(chainLengths: readonly number[]): number[] {
  const out: number[] = [];
  for (const len of chainLengths) {
    for (let k = 1; k <= len; k++) out.push(k);
  }
  return out;
}

/**
 * Safari omits `navigator.deviceMemory`, which would cause LocalFold's default `budgetForDevice()`
 * to assume a 4 GiB mobile phone (~1.33 GiB ceiling) on Apple Silicon / macOS desktops.
 * Provide a sensible desktop macOS fallback while preserving Chromium `deviceMemory` and mobile caps.
 */
function resolveLocalFoldMemoryBudgetBytes(): number | null {
  const nav =
    typeof navigator === 'object' && navigator !== null
      ? (navigator as Navigator & { deviceMemory?: number })
      : null;
  if (
    nav &&
    typeof nav.deviceMemory === 'number' &&
    Number.isFinite(nav.deviceMemory) &&
    nav.deviceMemory > 0
  ) {
    return null;
  }
  const ua = nav?.userAgent ?? '';
  const isMobile = /iPhone|iPad|iPod|Android|Mobile/i.test(ua);
  const isMacDesktop = !isMobile && /Macintosh|Mac OS X/i.test(ua);
  if (isMacDesktop) {
    const cores = typeof nav?.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : 8;
    const estGiB = cores >= 10 ? 16 : 8;
    return Math.round((estGiB * 1024 * 1024 * 1024) / 3);
  }
  return null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let lfCore: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let lfAf3: any = null;
let gpuDevice: GPUDevice | null = null;
let loadedModelKey: string | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadedAf2: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadedAf3: any = null;
let predictEsmfold2: Awaited<ReturnType<typeof loadEsmfold2>> | null = null;
let esmfold2Abort: AbortController | null = null;

async function importVendor(relPath: string) {
  const url = `${self.location.origin}/vendor/localfold/${relPath}`;
  return await import(/* @vite-ignore */ url);
}

async function ensureCoreModules(id: number) {
  if (lfCore) return;
  self.postMessage({
    id,
    type: 'progress',
    phase: 'init',
    progress: 0.02,
    detail: 'Loading LocalFold WebGPU modules...',
  });
  const [
    monomerMod,
    unifiedMod,
    chainsMod,
    a3mMod,
    mmseqsMod,
    fixtureMod,
    storeMod,
    deltaStoreMod,
    manifestsMod,
    deviceMod,
    residentMod,
  ] = await Promise.all([
    importVendor('webgpu/af2/model/monomer.js'),
    importVendor('webgpu/af2/multimer/model.js'),
    importVendor('shared/input/chains.js'),
    importVendor('shared/input/a3m.js'),
    importVendor('shared/input/mmseqs2-api.js'),
    importVendor('shared/bundles/alphafold-fixture.js'),
    importVendor('shared/bundles/http-tensor-store.js'),
    importVendor('shared/bundles/delta-tensor-store.js'),
    importVendor('shared/bundles/manifests/index.js'),
    importVendor('webgpu/runtime/device.js'),
    importVendor('webgpu/runtime/resident.js'),
  ]);
  lfCore = {
    AlphaFoldMonomerGpu: monomerMod.AlphaFoldMonomerGpu,
    AlphaFoldUnifiedGpu: unifiedMod.AlphaFoldUnifiedGpu,
    mergeChainA3ms: chainsMod.mergeChainA3ms,
    mergeUnpairedChainA3ms: chainsMod.mergeUnpairedChainA3ms,
    parseA3m: a3mMod.parseA3m,
    generateMmseqs2Msa: mmseqsMod.generateMmseqs2Msa,
    generateMmseqs2ComplexMsa: mmseqsMod.generateMmseqs2ComplexMsa,
    expandSearchedChains: mmseqsMod.expandSearchedChains,
    mergeSearchedChains: mmseqsMod.mergeSearchedChains,
    singleSequenceA3m: mmseqsMod.singleSequenceA3m,
    AlphaFoldFixture: fixtureMod.AlphaFoldFixture,
    HttpTensorStore: storeMod.HttpTensorStore,
    DeltaTensorStore: deltaStoreMod.DeltaTensorStore,
    MODEL_BUNDLES: manifestsMod.MODEL_BUNDLES,
    graphFamily: manifestsMod.graphFamily,
    bundleBaseUrl: manifestsMod.bundleBaseUrl,
    loadManifest: manifestsMod.loadManifest,
    requestAlphaFoldDevice: deviceMod.requestAlphaFoldDevice,
    releaseAllWeights: residentMod.releaseAllWeights,
    releaseResidentWeights: residentMod.releaseResidentWeights,
  };
}

async function ensureAf3Modules(id: number) {
  if (lfAf3) return;
  self.postMessage({
    id,
    type: 'progress',
    phase: 'init',
    progress: 0.04,
    detail: 'Loading LocalFold AF3 / RF3 / Boltz-2 WebGPU kernels...',
  });
  const [foldMod, diffWeightsMod, weightsMod, featuriseMod, batchMod, dialectMod, residentMod, ccdMod] =
    await Promise.all([
      importVendor('webgpu/af3/fold.js'),
      importVendor('shared/af3/weights/diffusion-weights.js'),
      importVendor('shared/af3/weights/weights.js'),
      importVendor('shared/af3/featurise/featurise.js'),
      importVendor('shared/af3/featurise/batch.js'),
      importVendor('shared/af3/dialect.js'),
      importVendor('webgpu/runtime/resident.js'),
      importVendor('shared/af3/featurise/ccd-component.js'),
    ]);
  lfAf3 = {
    foldBatch: foldMod.foldBatch,
    foldWeights: diffWeightsMod.foldWeights,
    releaseWeights: weightsMod.releaseWeights,
    featuriseProtein: featuriseMod.featuriseProtein,
    af3BatchFromA3m: batchMod.af3BatchFromA3m,
    featuriserDialect: dialectMod.featuriserDialect,
    releaseResidentWeights: residentMod.releaseResidentWeights,
    ccdUrl: ccdMod.ccdUrl,
    parseCcdComponent: ccdMod.parseCcdComponent,
  };
}

function releaseCurrentModel() {
  try {
    loadedAf2?.store?.releaseDecoded?.();
    loadedAf2?.baseStore?.releaseDecoded?.();
  } catch {
    /* ignore */
  }
  try {
    if (loadedAf3?.weights && lfAf3?.releaseWeights) {
      lfAf3.releaseWeights(loadedAf3.weights);
    }
    loadedAf3?.store?.releaseDecoded?.();
  } catch {
    /* ignore */
  }
  try {
    // Release resident & streamed GPU weight buffers for BOTH AF2 and AF3 models
    if (gpuDevice && lfCore?.releaseAllWeights) {
      lfCore.releaseAllWeights(gpuDevice);
    } else if (gpuDevice && lfAf3?.releaseResidentWeights) {
      lfAf3.releaseResidentWeights(gpuDevice);
    }
  } catch {
    /* ignore */
  }
  esmfold2Abort?.abort();
  esmfold2Abort = null;
  predictEsmfold2 = null;
  loadedAf2 = null;
  loadedAf3 = null;
  loadedModelKey = null;
}

async function ensureDevice() {
  if (gpuDevice) return;
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new Error('WebGPU is required for LocalFold but is not available in this browser');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No compatible WebGPU adapter found for LocalFold');
  const dev: GPUDevice = await lfCore.requestAlphaFoldDevice(adapter, {
    memoryBudgetBytes: resolveLocalFoldMemoryBudgetBytes(),
  });
  gpuDevice = dev;
  void dev.lost.then(() => {
    if (gpuDevice === dev) {
      gpuDevice = null;
      releaseCurrentModel();
    }
  });
}

async function ensureModelLoaded(id: number, modelKey: string) {
  await ensureCoreModules(id);

  if (loadedModelKey && loadedModelKey !== modelKey) {
    releaseCurrentModel();
    try {
      gpuDevice?.destroy();
    } catch {
      /* ignore */
    }
    gpuDevice = null;
  }

  await ensureDevice();

  if (loadedModelKey === modelKey && (loadedAf2 || loadedAf3 || predictEsmfold2)) {
    return;
  }

  const makeWeightProgress = (label: string) => (p: { loadedBytes: number; totalBytes: number }) => {
    const frac = p.totalBytes > 0 ? p.loadedBytes / p.totalBytes : 0;
    const mb = (p.loadedBytes / 1024 / 1024).toFixed(1);
    const totMb = (p.totalBytes / 1024 / 1024).toFixed(0);
    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: frac,
      detail: `${label} weights: ${mb} / ${totMb} MB`,
    });
  };

  if (AF2_MODELS.has(modelKey) || AF2_MULTIMER_MODELS.has(modelKey)) {
    const isMultimer = AF2_MULTIMER_MODELS.has(modelKey);
    const labelPrefix = isMultimer ? 'AF2-Multimer v3' : 'AlphaFold2';
    const bundleMeta = lfCore.MODEL_BUNDLES[modelKey];
    let store;
    let baseStore = null;

    if (bundleMeta?.delta?.base) {
      const baseKey = bundleMeta.delta.base;
      const [baseManifest, deltaManifest] = await Promise.all([
        lfCore.loadManifest(baseKey),
        lfCore.loadManifest(modelKey),
      ]);
      baseStore = await lfCore.HttpTensorStore.fromManifest(
        lfCore.bundleBaseUrl(baseKey),
        baseManifest,
        makeWeightProgress(`${labelPrefix} base`),
      );
      baseStore.prefetch?.();
      const deltaStore = await lfCore.HttpTensorStore.fromManifest(
        lfCore.bundleBaseUrl(modelKey),
        deltaManifest,
        makeWeightProgress(`${labelPrefix} (${modelKey})`),
      );
      deltaStore.prefetch?.();
      store = new lfCore.DeltaTensorStore(baseStore, deltaStore);
    } else {
      const manifest = await lfCore.loadManifest(modelKey);
      store = await lfCore.HttpTensorStore.fromManifest(
        lfCore.bundleBaseUrl(modelKey),
        manifest,
        makeWeightProgress(labelPrefix),
      );
      store.prefetch?.();
    }

    const fixture = lfCore.AlphaFoldFixture.fromStore(store);
    const [
      embedding,
      templateTrack,
      extraStack,
      mainStack,
      structure,
      confidence,
      geometry,
      featureTables,
      paeBreaks,
    ] = await Promise.all([
      fixture.embeddingWeights(),
      isMultimer ? fixture.templateEmbeddingWeights() : fixture.templateWeights(),
      fixture.extraStackWeights(),
      fixture.mainStackWeights(),
      fixture.structureWeights(),
      fixture.confidenceWeights(),
      fixture.geometryTables(),
      fixture.queryOnlyFeatureTables(),
      fixture.tensor('confidencePaeBreaks'),
    ]);

    loadedAf2 = {
      store,
      baseStore,
      featureTables,
      paeBreaks,
      isMultimer,
      weights: {
        embedding,
        template: isMultimer ? undefined : templateTrack,
        templateEmbedding: isMultimer ? templateTrack : undefined,
        extraStack,
        mainStack,
        structure,
        lddt: confidence.lddt,
        pae: confidence.pae,
        geometry,
        distogram: undefined,
      },
    };
    loadedModelKey = modelKey;
    return;
  }

  if (ESMFOLD2_MODELS.has(modelKey)) {
    predictEsmfold2 = await loadEsmfold2(modelKey, importVendor, makeWeightProgress(modelKey));
    loadedModelKey = modelKey;
    return;
  }

  if (AF3_MODELS.has(modelKey)) {
    await ensureAf3Modules(id);
    const manifest = await lfCore.loadManifest(modelKey);
    const base = lfCore.bundleBaseUrl(modelKey);
    const store = await lfCore.HttpTensorStore.fromManifest(
      base,
      manifest,
      makeWeightProgress(`LocalFold ${modelKey}`),
    );
    store.prefetch?.();

    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.96,
      detail: `Preparing ${modelKey} weights...`,
    });
    const weights = await lfAf3.foldWeights(store);
    loadedAf3 = { store, weights };
    loadedModelKey = modelKey;
    return;
  }

  throw new Error(`Unsupported LocalFold model "${modelKey}"`);
}

function splitChainSequences(cleanSeq: string, chainLengths: number[]): string[] {
  if (chainLengths.length <= 1) return [cleanSeq];
  const parts: string[] = [];
  let offset = 0;
  for (const len of chainLengths) {
    parts.push(cleanSeq.slice(offset, offset + len));
    offset += len;
  }
  return parts;
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'dispose') {
      releaseCurrentModel();
      gpuDevice?.destroy();
      gpuDevice = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const seqSet = msg.inputs?.sequences as SequenceSet | undefined;
      if (!seqSet?.candidates?.length) {
        throw new Error('LocalFold requires an upstream SequenceSet with at least one sequence');
      }

      const p = msg.params ?? {};
      const modelKey = String(p.model ?? 'monomer');
      const isAf2Mono = AF2_MODELS.has(modelKey);
      const isAf2Multi = AF2_MULTIMER_MODELS.has(modelKey);
      const isAf3 = AF3_MODELS.has(modelKey);
      const isEsmfold2 = ESMFOLD2_MODELS.has(modelKey);
      if (!isAf2Mono && !isAf2Multi && !isAf3 && !isEsmfold2) {
        throw new Error(`Unsupported LocalFold model "${modelKey}"`);
      }
      const recycles = clampLocalFoldRecycles(modelKey, p.recycles, 1);
      const tolerance = resolveAf2Tolerance(modelKey, p.tolerance);
      const diffusionSteps = Math.max(10, Math.min(200, Math.round(Number(p.diffusionSteps ?? 25))));
      const rawMsaMode = String(p.msaMode ?? 'single').trim();
      const msaMode: MsaMode =
        rawMsaMode === 'mmseqs2' || rawMsaMode === 'target-only' || rawMsaMode === 'a3m'
          ? rawMsaMode
          : 'single';
      const msaA3mText = typeof p.msaA3mText === 'string' ? p.msaA3mText : '';
      const lockedChains: string[] | undefined = Array.isArray(seqSet.lockedChains)
        ? seqSet.lockedChains
        : Array.isArray(p.lockedChains)
          ? (p.lockedChains as unknown[]).map((c) => String(c).trim()).filter(Boolean)
          : undefined;
      const chainRoles: Record<string, 'target' | 'binder' | 'scaffold'> | undefined =
        seqSet.chainRoles ??
        (p.chainRoles && typeof p.chainRoles === 'object'
          ? (p.chainRoles as Record<string, 'target' | 'binder' | 'scaffold'>)
          : undefined);
      const work = selectFoldWorkItems(seqSet, p);
      const maxSeqs = work.maxSequences;
      const rawMaxTotalFolds = Number(p.maxTotalFolds ?? 16);
      const maxTotalFolds = Number.isFinite(rawMaxTotalFolds)
        ? Math.max(1, Math.min(64, Math.round(rawMaxTotalFolds)))
        : 16;
      const seed = Number(p.seed ?? 0);
      const ligandCcdCodes: string[] = Array.isArray(p.ligandCcdCodes)
        ? (p.ligandCcdCodes as string[]).map((s) => String(s).trim().toUpperCase()).filter(Boolean)
        : String(p.ligandCcdCodes ?? '')
            .split(/[,\s]+/)
            .map((s) => s.trim().toUpperCase())
            .filter(Boolean);
      const dnaSeqs: string[] = Array.isArray(p.dnaSequences)
        ? (p.dnaSequences as string[]).map((s) => String(s).replace(/[^ACGT]/gi, '').toUpperCase()).filter(Boolean)
        : String(p.dnaSequences ?? '')
            .split(/[:,;\s]+/)
            .map((s) => s.replace(/[^ACGT]/gi, '').toUpperCase())
            .filter(Boolean);
      const rnaSeqs: string[] = Array.isArray(p.rnaSequences)
        ? (p.rnaSequences as string[]).map((s) => String(s).replace(/[^ACGU]/gi, '').toUpperCase()).filter(Boolean)
        : String(p.rnaSequences ?? '')
            .split(/[:,;\s]+/)
            .map((s) => s.replace(/[^ACGU]/gi, '').toUpperCase())
            .filter(Boolean);

      const esmfold2Steps = Number(p.esmfold2Steps ?? 15);
      if (isEsmfold2) {
        const issue = esmfold2ParamIssue(p);
        if (issue) throw new Error(issue);
      }

      interface ResolvedFoldTarget {
        cand: SequenceCandidate;
        seed: number;
        designKey?: string;
        designIndex?: number;
        sequenceKey?: string;
        foldKey?: string;
      }

      // Detect multi-backbone SequenceSet via distinct non-empty designKey / sourceStructureId anchors
      const distinctParentAnchors = new Set<string>();
      for (const cand of seqSet.candidates) {
        const anchor = (cand.designKey ?? cand.sourceStructureId ?? '').trim();
        if (anchor && anchor.toLowerCase() !== 'none') {
          distinctParentAnchors.add(anchor);
        }
      }
      const isMultiBackbone = distinctParentAnchors.size > 1;

      let targets: ResolvedFoldTarget[];
      let excludedCandidates: ExcludedCandidateRecord[];

      if (!isMultiBackbone) {
        // Legacy single-backbone path: exact selectFoldWorkItems slice & seed + i semantics
        targets = work.selected.map((cand, i) => ({
          cand,
          seed: seed + i,
          ...(cand.designKey ? { designKey: cand.designKey } : {}),
          ...(cand.designIndex !== undefined ? { designIndex: cand.designIndex } : {}),
          ...(cand.sequenceKey ? { sequenceKey: cand.sequenceKey } : {}),
        }));
        excludedCandidates = work.excludedCandidates;
      } else {
        // Multi-backbone fan-out via authoritative planLocalFoldFanout (round-robin across backbones, capped at maxTotalFolds)
        const groupOrder = new Map<string, number>();
        const origIndexByKey = new Map<string, number>();
        const origIdByKey = new Map<string, string>();
        const usedKeys = new Set<string>();

        const seqMembers: CollectionMember<SequenceCandidate>[] = seqSet.candidates.map((cand, idx) => {
          const designAnchor = (cand.designKey ?? cand.sourceStructureId ?? '').trim();
          const hasDesign = designAnchor.length > 0 && designAnchor.toLowerCase() !== 'none';
          const designKey = hasDesign ? designAnchor : undefined;
          let designIndex = cand.designIndex;
          if (designKey !== undefined && designIndex === undefined) {
            if (!groupOrder.has(designKey)) groupOrder.set(designKey, groupOrder.size);
            designIndex = groupOrder.get(designKey);
          }
          let seqKey = (cand.sequenceKey ?? cand.key ?? cand.id).trim() || `seq-${idx}`;
          if (usedKeys.has(seqKey)) {
            seqKey = `${seqKey}::${idx}`;
          }
          usedKeys.add(seqKey);
          origIndexByKey.set(seqKey, idx);
          origIdByKey.set(seqKey, cand.id);
          const ordinal =
            typeof cand.sampleIndex === 'number' && Number.isInteger(cand.sampleIndex) && cand.sampleIndex >= 0
              ? cand.sampleIndex
              : idx;
          return {
            key: seqKey,
            ordinal,
            memberKind: 'SequenceCandidate',
            lineage: {
              ...(designKey ? { designKey, designIndex } : {}),
              sequenceKey: seqKey,
              sequenceIndex: ordinal,
              ...(cand.sourceStructureId && cand.sourceStructureId.toLowerCase() !== 'none'
                ? {
                    sourceStructure: {
                      artifactId: cand.sourceStructureId,
                      payloadDigest: '',
                      ...(designKey ? { designKey } : {}),
                    },
                  }
                : {}),
              parentKeys: designKey ? [designKey] : [],
            },
            payload: cand,
            summary: {
              label: cand.id,
              residueCount: cand.sequence.length,
              sequenceDigest: sha256Hex(cand.sequence),
              score: cand.score,
              globalScore: cand.globalScore,
            },
          };
        });

        const fanoutPlan = planLocalFoldFanout(seqMembers, {
          foldModelId: modelKey,
          baseSeed: seed,
          perDesignTopK: maxSeqs,
          samplesPerSequence: 1,
          maxFolds: maxTotalFolds,
          ranking: 'mpnn-score',
        });

        targets = fanoutPlan.selected.map((planned) => ({
          cand: planned.sequenceMember.payload!,
          seed: planned.seed,
          designKey: planned.designKey ?? planned.sequenceMember.payload?.designKey,
          designIndex: planned.designIndex ?? planned.sequenceMember.payload?.designIndex,
          sequenceKey: planned.sequenceKey,
          foldKey: planned.foldKey,
        }));

        excludedCandidates = fanoutPlan.excluded.map((ex, exIdx) => {
          const refKey = ex.lineage?.sequenceKey ?? ex.key;
          return {
            key: ex.key,
            id: origIdByKey.get(refKey) ?? ex.key,
            index: origIndexByKey.get(refKey) ?? targets.length + exIdx,
            reason: ex.reason ?? `Excluded by LocalFold fan-out policy (maxSequences=${maxSeqs}, maxTotalFolds=${maxTotalFolds})`,
          };
        });
      }

      const ef2Chains = isEsmfold2 ? targets.map(({ cand }) => esmfold2Chains(cand)) : [];
      await ensureModelLoaded(msg.id, modelKey);
      const folds: FoldedCandidate[] = [];
      const sourceStructureId = seqSet.sourceStructureId ?? seqSet.parentStructureId;
      const msaCache = createMsaSearchCache();
      const mmseqsApi: MmseqsApi = {
        generateMmseqs2Msa: lfCore.generateMmseqs2Msa,
        generateMmseqs2ComplexMsa: lfCore.generateMmseqs2ComplexMsa,
        expandSearchedChains: lfCore.expandSearchedChains,
        mergeSearchedChains: lfCore.mergeSearchedChains,
        singleSequenceA3m: lfCore.singleSequenceA3m,
        parseA3m: lfCore.parseA3m,
        graphFamily: lfCore.graphFamily,
      };

      if (isEsmfold2) {
        if (!gpuDevice || !predictEsmfold2) throw new Error('ESMFold2 was not initialized');
        const parsedLigands: unknown[] = [];
        if (ligandCcdCodes.length > 0) {
          const ccdMod = await importVendor('shared/af3/featurise/ccd-component.js');
          if (!ccdMod?.ccdUrl || !ccdMod?.parseCcdComponent) {
            throw new Error(
              `Cannot co-fold requested CCD ligand(s) [${ligandCcdCodes.join(', ')}]: LocalFold CCD component parser is unavailable.`,
            );
          }
          const textDec = new TextDecoder();
          for (const code of ligandCcdCodes) {
            try {
              const url = ccdMod.ccdUrl(code);
              const bytes = await fetchCachedBytes(`lf-ccd-${code}`, url);
              parsedLigands.push(ccdMod.parseCcdComponent(textDec.decode(bytes)));
            } catch (err) {
              throw new Error(
                `Failed to fetch/parse requested CCD ligand "${code}" for ESMFold2 holo co-folding (${err instanceof Error ? err.message : String(err)}). Verify the CCD code or remove it to fold apo.`,
              );
            }
          }
        }
        const controller = new AbortController();
        esmfold2Abort = controller;
        try {
          for (const [i, target] of targets.entries()) {
            const { cand } = target;
            const start = performance.now();
            const result = await predictEsmfold2(
              gpuDevice,
              ef2Chains[i],
              esmfold2Steps,
              target.seed,
              controller.signal,
              (progress, stage) => self.postMessage({
                id: msg.id, type: 'progress', phase: 'fold',
                progress: (i + progress) / targets.length, detail: `${modelKey} ${cand.id}: ${stage}`,
              }),
              { dnaSeqs, rnaSeqs, parsedLigands },
            );
            folds.push({
              ...result, sequenceId: cand.id, key: target.sequenceKey ?? cand.key ?? cand.id,
              designKey: target.designKey, designIndex: target.designIndex,
              sequenceKey: target.sequenceKey, foldKey: target.foldKey, seed: target.seed,
              sourceStructureId: cand.sourceStructureId ?? sourceStructureId,
              sampleIndex: cand.sampleIndex, foldModelId: `localfold-${modelKey}-int5`,
              elapsedMs: performance.now() - start,
            });
          }
        } finally {
          if (esmfold2Abort === controller) esmfold2Abort = null;
        }
      } else if (!isAf3) {
        const runner = isAf2Multi
          ? new lfCore.AlphaFoldUnifiedGpu(gpuDevice)
          : new lfCore.AlphaFoldMonomerGpu(gpuDevice);
        for (let i = 0; i < targets.length; i++) {
          const target = targets[i];
          const cand = target.cand;
          const itemSeed = target.seed;
          const stripped = cand.sequence.toUpperCase().replace(/[\s/:,]+/g, '');
          if (stripped.length === 0) {
            throw new Error(`Refusing to fold "${cand.id}": sequence is empty`);
          }
          const validCount = (stripped.match(/[ACDEFGHIKLMNPQRSTVWY]/g) ?? []).length;
          if (validCount / stripped.length < 0.5) {
            throw new Error(
              `Refusing to fold "${cand.id}": ${stripped.length - validCount}/${stripped.length} residues are unknown/non-canonical (>50% unknown). Check the upstream generator output.`,
            );
          }
          // Preserve 'X' natively (LocalFold encodes 'X' as aatype=20); map any other non-canonical char to 'X'
          const cleanSeq = stripped.replace(/[^ACDEFGHIKLMNPQRSTVWYX]/g, 'X');
          const chainLengths =
            Array.isArray(cand.chainLengths) &&
            cand.chainLengths.length > 0 &&
            cand.chainLengths.every((n) => Number.isInteger(n) && n > 0) &&
            cand.chainLengths.reduce((a, b) => a + b, 0) === cleanSeq.length
              ? cand.chainLengths
              : [cleanSeq.length];
          const chainSeqs = splitChainSequences(cleanSeq, chainLengths);
          const chainIds = chainIdsFromLengths(chainLengths);
          const uniqueChains = [...new Set(chainIds)];

          const msa = await resolveMsaForCandidate({
            msaMode,
            modelKey,
            chainSeqs,
            chainIds: uniqueChains,
            lockedChains,
            chainRoles,
            msaA3mText,
            mmseqsApi,
            cache: msaCache,
            onProgress: (ev) => {
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'msa',
                progress: i / targets.length,
                detail: `${cand.id}: ${ev.detail}`,
              });
            },
          });
          const a3m = msa.a3m;
          const msaBudgets = computeAf2MsaBudgets(msa.msaDepth, chainSeqs.length, msa.distinctDepth);
          const foldStart = performance.now();

          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'fold',
            progress: i / targets.length,
            detail: `Folding ${cand.id} with ${isAf2Multi ? 'AF2-Multimer' : 'AF2'} ${modelKey} (${cleanSeq.length} aa${chainLengths.length > 1 ? `, ${chainLengths.length} chains` : ''}, ${recycles} rec${msa.msaDepth > 1 ? `, MSA ${msa.msaDepth}` : ''})...`,
          });

          const predictOpts = isAf2Multi
            ? {
                recycles,
                randomSeed: itemSeed,
                maxMsaSequences: msaBudgets.maxMsaSequences,
                maxExtraSequences: msaBudgets.maxExtraSequences,
                maskFraction: msaBudgets.maskFraction,
                chainLengths,
                chainSequences: chainSeqs,
                chainAware: true,
                outerProductMeanFirst: true,
                positionScale: 20,
                tolerance,
              }
            : {
                recycles,
                randomSeed: itemSeed,
                maxMsaSequences: msaBudgets.maxMsaSequences,
                maxExtraSequences: msaBudgets.maxExtraSequences,
                maskFraction: msaBudgets.maskFraction,
                chainLengths,
                tolerance,
              };

          const prediction = await runner.predictA3m(
            a3m,
            loadedAf2.weights,
            loadedAf2.featureTables,
            predictOpts,
            loadedAf2.paeBreaks,
            undefined,
            (prog: { completed: number; total: number }) => {
              const sub = prog.total > 0 ? Math.min(1, Math.max(0, prog.completed / prog.total)) : 0;
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'fold',
                progress: (i + sub) / targets.length,
                detail: `Folding ${cand.id} (${i + 1}/${targets.length}): ${(sub * 100).toFixed(0)}%`,
              });
            },
          );

          const fin = prediction.final;
          const passesRun =
            Array.isArray(prediction.recycles) && prediction.recycles.length > 0
              ? prediction.recycles.length
              : recycles + 1;
          const recyclesRun = Math.max(0, passesRun - 1);
          const earlyStopped = recyclesRun < recycles;
          const plddt = new Float32Array(fin.confidence.plddt as Float32Array);
          let sumPlddt = 0;
          for (let k = 0; k < plddt.length; k++) sumPlddt += plddt[k];
          const meanPlddt = plddt.length ? sumPlddt / plddt.length : 0;

          const { pdbText, caCoords } = atom37ToPdb(
            fin.structure.atom37 as Float32Array,
            fin.structure.atom37Mask as Float32Array,
            cleanSeq,
            plddt,
            chainIds,
          );

          const paeRaw = fin.confidence.predictedAlignedError as Float32Array | undefined;
          const ptmVal =
            typeof fin.confidence.ptm === 'number'
              ? fin.confidence.ptm
              : typeof fin.confidence.predictedTmScore === 'number'
                ? fin.confidence.predictedTmScore
                : null;
          const iptmVal =
            typeof fin.confidence.iptm === 'number' && Number.isFinite(fin.confidence.iptm)
              ? fin.confidence.iptm
              : null;
          const chainScores = normalizeLocalFoldChainScores(
            {
              ptm: ptmVal,
              iptm: iptmVal,
              multimerScore: fin.confidence.multimerScore,
              chainPtm: fin.confidence.chainPtm,
              chainIptm: fin.confidence.chainIptm,
              chainPairIptm: fin.confidence.chainPairIptm,
            },
            uniqueChains,
            0,
          );
          const iptmIsMonomerApprox = isAf2Mono && chainLengths.length > 1 && iptmVal !== null;
          const ipsaeRes = computeCandidateIpsae(paeRaw, chainLengths, chainIds);

          folds.push({
            sequenceId: cand.id,
            key: target.sequenceKey ?? cand.key ?? cand.id,
            ...(target.designKey ? { designKey: target.designKey } : {}),
            ...(target.designIndex !== undefined ? { designIndex: target.designIndex } : {}),
            ...(target.sequenceKey ? { sequenceKey: target.sequenceKey } : {}),
            ...(target.foldKey ? { foldKey: target.foldKey } : {}),
            seed: itemSeed,
            sequence: cleanSeq,
            foldModelId: `localfold-af2-${modelKey}-int5`,
            sourceStructureId: cand.sourceStructureId ?? sourceStructureId,
            ...(cand.sampleIndex !== undefined ? { sampleIndex: cand.sampleIndex } : {}),
            pdbText,
            caCoords,
            plddt,
            meanPlddt,
            confidence: Number.isFinite(meanPlddt) ? meanPlddt : null,
            confidenceAvailable: Number.isFinite(meanPlddt),
            ptm: ptmVal,
            iptm: iptmVal,
            ...chainScores,
            ...(iptmIsMonomerApprox ? { iptmIsMonomerApprox: true } : {}),
            chainLengths,
            chainIds,
            resNums: sequentialResNumsFromChainLengths(chainLengths),
            pae: paeRaw ? new Float32Array(paeRaw) : undefined,
            ...(ipsaeRes ? { ipsae: ipsaeRes.ipsae, ipsaeMin: ipsaeRes.ipsaeMin } : {}),
            msaMode,
            msaDepth: msa.msaDepth,
            provenance: {
              quantization: 'int5',
              recyclesRun,
              ...(earlyStopped ? { earlyStopped: true } : {}),
              msaMode,
              msaDepth: msa.msaDepth,
            },
            elapsedMs: performance.now() - foldStart,
          });
        }
      } else {
        // AF3-lineage unified driver (RF3, Boltz-2, OpenBind-0, Protenix-v2, OpenDDE, Mol-1, IntelliFold-2, AF3)
        const dialect = loadedAf3.weights.trunk.dialect;
        const featOpts = lfAf3.featuriserDialect(dialect);

        // Optional ligand CCD passthrough for AF3-lineage holo refolding (fail loud — never silently fold apo!)
        const parsedLigands: unknown[] = [];
        if (ligandCcdCodes.length > 0) {
          if (!lfAf3.ccdUrl || !lfAf3.parseCcdComponent) {
            throw new Error(
              `Cannot co-fold requested CCD ligand(s) [${ligandCcdCodes.join(', ')}]: LocalFold CCD component parser is unavailable.`,
            );
          }
          const textDec = new TextDecoder();
          for (const code of ligandCcdCodes) {
            try {
              const url = lfAf3.ccdUrl(code);
              const bytes = await fetchCachedBytes(`lf-ccd-${code}`, url);
              parsedLigands.push(lfAf3.parseCcdComponent(textDec.decode(bytes)));
            } catch (err) {
              throw new Error(
                `Failed to fetch/parse requested CCD ligand "${code}" for holo co-folding (${err instanceof Error ? err.message : String(err)}). Verify the CCD code or remove it to fold apo.`,
              );
            }
          }
        }

        for (let i = 0; i < targets.length; i++) {
          const target = targets[i];
          const cand = target.cand;
          const itemSeed = target.seed;
          const stripped = cand.sequence.toUpperCase().replace(/[\s/:,]+/g, '');
          if (stripped.length === 0) {
            throw new Error(`Refusing to fold "${cand.id}": sequence is empty`);
          }
          const validCount = (stripped.match(/[ACDEFGHIKLMNPQRSTVWY]/g) ?? []).length;
          if (validCount / stripped.length < 0.5) {
            throw new Error(
              `Refusing to fold "${cand.id}": ${stripped.length - validCount}/${stripped.length} residues are unknown/non-canonical (>50% unknown). Check the upstream generator output.`,
            );
          }
          const cleanSeq = stripped.replace(/[^ACDEFGHIKLMNPQRSTVWYX]/g, 'X');
          const chainLengths =
            Array.isArray(cand.chainLengths) &&
            cand.chainLengths.length > 0 &&
            cand.chainLengths.every((n) => Number.isInteger(n) && n > 0) &&
            cand.chainLengths.reduce((a, b) => a + b, 0) === cleanSeq.length
              ? cand.chainLengths
              : [cleanSeq.length];
          const proteinChainSeqs = splitChainSequences(cleanSeq, chainLengths);
          const proteinUniqueChains = [...new Set(chainIdsFromLengths(chainLengths))];

          const msa = await resolveMsaForCandidate({
            msaMode,
            modelKey,
            chainSeqs: proteinChainSeqs,
            chainIds: proteinUniqueChains,
            lockedChains,
            chainRoles,
            msaA3mText,
            mmseqsApi,
            cache: msaCache,
            onProgress: (ev) => {
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'msa',
                progress: i / targets.length,
                detail: `${cand.id}: ${ev.detail}`,
              });
            },
          });

          const allPolymerSeqs = [...proteinChainSeqs, ...dnaSeqs, ...rnaSeqs];
          const chainKinds: Array<'protein' | 'dna' | 'rna'> = [
            ...proteinChainSeqs.map(() => 'protein' as const),
            ...dnaSeqs.map(() => 'dna' as const),
            ...rnaSeqs.map(() => 'rna' as const),
          ];
          const colonSeq = allPolymerSeqs.join(':');
          const hasExtraContext = dnaSeqs.length > 0 || rnaSeqs.length > 0 || parsedLigands.length > 0;
          const foldStart = performance.now();

          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'fold',
            progress: i / targets.length,
            detail: `Folding ${cand.id} with ${modelKey} (${cleanSeq.length} aa${parsedLigands.length ? ` + ${ligandCcdCodes.join(',')}` : ''}${dnaSeqs.length ? ` + ${dnaSeqs.length} DNA` : ''}${rnaSeqs.length ? ` + ${rnaSeqs.length} RNA` : ''}, ${recycles} rec, ${diffusionSteps} steps${msa.msaDepth > 1 ? `, MSA ${msa.msaDepth}` : ''})...`,
          });

          // Use upstream af3BatchFromA3m for BOTH single-sequence (alignment = null) and MSA paths:
          // it enforces dialect boundary inversions (`dedupeSelfMsa === false` -> `duplicateQueryRow: true`
          // for AF3/OpenBind-0 depth-2 parity, `dropTerminalAtoms` -> `terminalAtoms`, `seed` -> `conformerSeed`).
          const { batch } = lfAf3.af3BatchFromA3m(colonSeq, msa.af3Alignment, {
            ...featOpts,
            seed: itemSeed,
            conformerSeed: itemSeed,
            ...(hasExtraContext
              ? {
                  chainKinds,
                  ...(parsedLigands.length > 0 ? { ligands: parsedLigands } : {}),
                }
              : {}),
          });
          const result = await lfAf3.foldBatch(gpuDevice, batch, loadedAf3.weights, {
            mode: 'diffusion',
            steps: diffusionSteps,
            recycles,
            seed: itemSeed,
            onStage: (name: string, detail: Record<string, unknown> = {}) => {
              let sub = 0.1;
              let label = name;
              if (name === 'pairformer-block') {
                const idx = Number(detail.index ?? 0);
                const tot = Math.max(1, Number(detail.total ?? 48));
                const pass = Number(detail.pass ?? 0);
                const passes = Math.max(1, Number(detail.passes ?? 1));
                sub = 0.05 + 0.55 * ((pass + idx / tot) / passes);
                label = `trunk ${pass + 1}/${passes} (${idx + 1}/${tot})`;
              } else if (name === 'trunk-done') {
                sub = 0.62;
                label = 'trunk done';
              } else if (name === 'confidence') {
                sub = 0.94;
                label = 'confidence head';
              }
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'fold',
                progress: (i + sub) / targets.length,
                detail: `${modelKey} ${cand.id}: ${label}`,
              });
            },
            onStep: (stepInfo: { step?: number; total?: number }) => {
              const s = Number(stepInfo.step ?? 0) + 1;
              const tot = Math.max(1, Number(stepInfo.total ?? diffusionSteps));
              const sub = 0.62 + 0.3 * Math.min(1, s / tot);
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'fold',
                progress: (i + sub) / targets.length,
                detail: `${modelKey} ${cand.id}: diffusion ${s}/${tot}`,
              });
            },
          });

          const recyclesRun =
            typeof result.reusable?.recycles === 'number' && Number.isFinite(result.reusable.recycles)
              ? result.reusable.recycles
              : recycles;
          const earlyStopped = recyclesRun < recycles;
          const pdbText = String(result.pdb ?? '');
          const parsed = parsePdbBackbone(pdbText);
          const L = cleanSeq.length;
          const plddt = new Float32Array(L);
          if (Array.isArray(result.perResiduePlddt)) {
            for (let k = 0; k < L; k++) {
              const v = result.perResiduePlddt[k];
              plddt[k] = typeof v === 'number' && Number.isFinite(v) ? v : 0;
            }
          }
          let sumPlddt = 0;
          for (let k = 0; k < plddt.length; k++) sumPlddt += plddt[k];
          // Keep meanPlddt mathematically consistent with the per-residue protein plddt array
          // (upstream AF3/RF3 result.meanPlddt averages over ligand atoms too, which caused mean < min(range)).
          const meanPlddt =
            plddt.length > 0 && sumPlddt > 0
              ? sumPlddt / plddt.length
              : typeof result.meanPlddt === 'number' && Number.isFinite(result.meanPlddt)
                ? result.meanPlddt
                : 0;

          const paeRaw = result.scores?.pae as Float32Array | undefined;
          const ptmVal =
            typeof result.ptm === 'number' && Number.isFinite(result.ptm) ? result.ptm : null;
          const rawIptm = result.iptm ?? result.scores?.iptm;
          const iptmVal = typeof rawIptm === 'number' && Number.isFinite(rawIptm) ? rawIptm : null;
          const chainIds =
            parsed.chainIds.length === cleanSeq.length
              ? parsed.chainIds
              : chainIdsFromLengths(chainLengths);
          const uniqueChains = [...new Set(chainIds)];
          const chainScores = normalizeLocalFoldChainScores(
            {
              ptm: ptmVal,
              iptm: iptmVal,
              chainPtm: result.chainPtm,
              chainIptm: result.chainIptm,
              chainPairIptm: result.chainPairIptm,
            },
            uniqueChains,
            1,
          );

          // For AF3/RF3 protein+ligand/nucleic complexes, PAE is (L_protein + N_nonpoly)^2;
          // append the non-polymer token span so interface PAE & ipSAE compute across protein<->ligand.
          let ipsaeChainLengths = chainLengths;
          let ipsaeChainIds = chainIds;
          if (paeRaw && paeRaw.length > cleanSeq.length * cleanSeq.length) {
            const paeDim = Math.round(Math.sqrt(paeRaw.length));
            if (paeDim * paeDim === paeRaw.length && paeDim > cleanSeq.length) {
              const extraTokens = paeDim - cleanSeq.length;
              ipsaeChainLengths = [...chainLengths, extraTokens];
              ipsaeChainIds = [...chainIds, ...Array<string>(extraTokens).fill('LIG')];
            }
          }
          const ipsaeRes = computeCandidateIpsae(paeRaw, ipsaeChainLengths, ipsaeChainIds);

          folds.push({
            sequenceId: cand.id,
            key: target.sequenceKey ?? cand.key ?? cand.id,
            ...(target.designKey ? { designKey: target.designKey } : {}),
            ...(target.designIndex !== undefined ? { designIndex: target.designIndex } : {}),
            ...(target.sequenceKey ? { sequenceKey: target.sequenceKey } : {}),
            ...(target.foldKey ? { foldKey: target.foldKey } : {}),
            seed: itemSeed,
            sequence: cleanSeq,
            foldModelId: `localfold-${modelKey}-int5`,
            sourceStructureId: cand.sourceStructureId ?? sourceStructureId,
            ...(cand.sampleIndex !== undefined ? { sampleIndex: cand.sampleIndex } : {}),
            pdbText,
            caCoords: parsed.caCoords,
            plddt,
            meanPlddt,
            confidence: Number.isFinite(meanPlddt) ? meanPlddt : null,
            confidenceAvailable: Number.isFinite(meanPlddt),
            ptm: ptmVal,
            iptm: iptmVal,
            ...chainScores,
            chainLengths,
            chainIds,
            resNums:
              parsed.resNums.length === cleanSeq.length
                ? parsed.resNums
                : sequentialResNumsFromChainLengths(chainLengths),
            pae: paeRaw ? new Float32Array(paeRaw) : undefined,
            ...(ipsaeRes ? { ipsae: ipsaeRes.ipsae, ipsaeMin: ipsaeRes.ipsaeMin } : {}),
            msaMode,
            msaDepth: msa.msaDepth,
            provenance: {
              quantization: 'int5',
              recyclesRun,
              ...(earlyStopped ? { earlyStopped: true } : {}),
              msaMode,
              msaDepth: msa.msaDepth,
            },
            elapsedMs: performance.now() - foldStart,
          });
        }
      }

      const runHash = fnv1a(
        JSON.stringify({
          seqSetId: seqSet.id,
          modelKey,
          recycles: isEsmfold2 ? null : recycles,
          ...(isEsmfold2 ? { esmfold2Steps } : {}),
          tolerance: isAf3 || isEsmfold2 ? null : tolerance,
          diffusionSteps: isAf3 ? diffusionSteps : null,
          maxSequences: maxSeqs,
          ...(isMultiBackbone ? { maxTotalFolds } : {}),
          seed,
          msaMode,
          msaA3mHash: msaMode === 'a3m' ? fnv1a(msaA3mText) : null,
          ligandCcdCodes: isAf3 || isEsmfold2 ? ligandCcdCodes : [],
          dnaSeqs: isAf3 || isEsmfold2 ? dnaSeqs : [],
          rnaSeqs: isAf3 || isEsmfold2 ? rnaSeqs : [],
        }),
      );

      const collectionKind = seqSet.collectionKind ?? 'independent-candidates';
      const rep = selectLocalFoldRepresentative(folds, { collectionKind });
      const itemOutcomes: CollectionItemOutcome[] = [
        ...folds.map((f): CollectionItemOutcome => ({
          key: f.foldKey ?? f.key ?? f.sequenceId,
          status: 'completed',
        })),
        ...excludedCandidates.map((ex): CollectionItemOutcome => ({
          key: ex.key,
          status: 'excluded',
          reason: ex.reason,
        })),
      ];

      const out: FoldedStructure = {
        kind: 'FoldedStructure',
        id: `folds-${modelKey}-${seqSet.id}-${isEsmfold2 ? `d${esmfold2Steps}` : `r${recycles}`}-${runHash}`,
        parentStructureId: seqSet.parentStructureId,
        sourceStructureId,
        collectionKind,
        ...(excludedCandidates.length > 0 ? { excludedCandidates } : {}),
        itemOutcomes,
        ...(rep ? { representativeSelection: rep.selection } : {}),
        folds,
      };

      // Also export the selected representative fold as a StructureArtifact so downstream
      // ProteinMPNN / RFD4 template / Scorer blocks can chain directly out of LocalFold,
      // carrying explicit `representativeSelection` provenance.
      let bestStructure: StructureArtifact | undefined;
      if (rep) {
        const bestFold = rep.fold;
        const bestParsed = parsePdbBackbone(bestFold.pdbText);
        const bestChainIds =
          bestFold.chainIds && bestFold.chainIds.length === bestParsed.nResidues
            ? bestFold.chainIds
            : bestParsed.chainIds;
        const confTag =
          rep.selection.scoreValue !== null && rep.selection.scoreValue !== undefined
            ? rep.selection.criterion === 'highest-multimer-score'
              ? `score ${rep.selection.scoreValue.toFixed(2)}`
              : `pLDDT ${rep.selection.scoreValue.toFixed(1)}`
            : 'no confidence';
        bestStructure = {
          kind: 'StructureArtifact',
          id: `foldstruct-${modelKey}-${bestFold.sequenceId}-${runHash}`,
          label: `${bestFold.sequenceId} (${modelKey}, ${confTag})`,
          sourceModelId: 'localfold-af2',
          parentStructureId: sourceStructureId,
          representativeSelection: rep.selection,
          bfactorSemantics: 'plddt-0-100',
          atomGranularity: 'all-atom',
          ...(bestFold.provenance ? { provenance: bestFold.provenance } : {}),
          length: bestParsed.nResidues,
          sequence: bestFold.sequence,
          caCoords: bestParsed.caCoords,
          backbone4Coords: bestParsed.backbone4Coords,
          pdbText: bestFold.pdbText,
          chainIds: bestChainIds,
          chainLengths: bestFold.chainLengths ?? bestParsed.diagnostics.chainLengths,
          insCodes: bestParsed.insCodes,
          sourceResidues: bestParsed.sourceResidues,
          chainIdMap: bestParsed.chainIdMap,
          atomPresence: bestParsed.atomPresence,
          observedBackboneMask: bestParsed.observedBackboneMask,
          diagnostics: bestParsed.diagnostics,
          geometry: computeGeometryStats(
            bestParsed.caCoords,
            bestChainIds,
            bestParsed.resNums,
            bestParsed.insCodes,
          ),
        };
      }

      const transferViews: Array<ArrayBufferView | undefined> = [];
      for (const f of folds) {
        transferViews.push(f.caCoords, f.plddt, f.pae);
      }
      if (bestStructure) {
        transferViews.push(
          bestStructure.caCoords,
          bestStructure.backbone4Coords,
          bestStructure.observedBackboneMask,
        );
      }
      const transferables = collectOwnedTransferables(transferViews);
      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          outputs: bestStructure ? { folds: out, structure: bestStructure } : { folds: out },
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
