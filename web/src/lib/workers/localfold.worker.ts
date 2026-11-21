/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping sokrypton/localfold
// Supports AlphaFold2 monomer (models 1–5), AlphaFold2-Multimer v3 (models 1–5),
// and all AF3-lineage models (RoseTTAFold3, Boltz-2, OpenBind-0, Protenix-v2,
// OpenDDE, Mol-1, IntelliFold-2, AF3) loaded directly from /vendor/localfold/.

import { atom37ToPdb, chainIdsFromLengths, parsePdbBackbone } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { normalizeLocalFoldChainScores } from '$lib/engine/conditioning';
import { selectFoldRepresentative, selectFoldWorkItems } from '$lib/engine/bio-capabilities';
import { fetchCachedBytes } from '$lib/engine/opfs-cache';
import type {
  CollectionItemOutcome,
  FoldedCandidate,
  FoldedStructure,
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
    AlphaFoldFixture: fixtureMod.AlphaFoldFixture,
    HttpTensorStore: storeMod.HttpTensorStore,
    DeltaTensorStore: deltaStoreMod.DeltaTensorStore,
    MODEL_BUNDLES: manifestsMod.MODEL_BUNDLES,
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
  const [foldMod, diffWeightsMod, weightsMod, featuriseMod, dialectMod, residentMod, ccdMod] =
    await Promise.all([
      importVendor('webgpu/af3/fold.js'),
      importVendor('shared/af3/weights/diffusion-weights.js'),
      importVendor('shared/af3/weights/weights.js'),
      importVendor('shared/af3/featurise/featurise.js'),
      importVendor('shared/af3/dialect.js'),
      importVendor('webgpu/runtime/resident.js'),
      importVendor('shared/af3/featurise/ccd-component.js'),
    ]);
  lfAf3 = {
    foldBatch: foldMod.foldBatch,
    foldWeights: diffWeightsMod.foldWeights,
    releaseWeights: weightsMod.releaseWeights,
    featuriseProtein: featuriseMod.featuriseProtein,
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

  if (loadedModelKey === modelKey && (loadedAf2 || loadedAf3)) {
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
      if (!isAf2Mono && !isAf2Multi && !isAf3) {
        throw new Error(`Unsupported LocalFold model "${modelKey}"`);
      }
      const recycles = Math.max(0, Math.min(4, Math.round(Number(p.recycles ?? 1))));
      const tolerance = Math.max(0, Math.min(5, Number(p.tolerance ?? 0.5)));
      const diffusionSteps = Math.max(10, Math.min(200, Math.round(Number(p.diffusionSteps ?? 25))));
      const work = selectFoldWorkItems(seqSet, p);
      const maxSeqs = work.maxSequences;
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

      await ensureModelLoaded(msg.id, modelKey);
      const targets = work.selected;
      const folds: FoldedCandidate[] = [];
      const sourceStructureId = seqSet.sourceStructureId ?? seqSet.parentStructureId;

      if (!isAf3) {
        const runner = isAf2Multi
          ? new lfCore.AlphaFoldUnifiedGpu(gpuDevice)
          : new lfCore.AlphaFoldMonomerGpu(gpuDevice);
        for (let i = 0; i < targets.length; i++) {
          const cand = targets[i];
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
          const a3m =
            chainSeqs.length > 1
              ? isAf2Multi
                ? lfCore.mergeChainA3ms(chainSeqs.map((s) => `>q\n${s}\n`))
                : lfCore.mergeUnpairedChainA3ms(chainSeqs.map((s) => `>q\n${s}\n`))
              : `>query\n${cleanSeq}\n`;
          const foldStart = performance.now();

          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'fold',
            progress: i / targets.length,
            detail: `Folding ${cand.id} with ${isAf2Multi ? 'AF2-Multimer' : 'AF2'} ${modelKey} (${cleanSeq.length} aa${chainLengths.length > 1 ? `, ${chainLengths.length} chains` : ''}, ${recycles} rec)...`,
          });

          const predictOpts = isAf2Multi
            ? {
                recycles,
                randomSeed: seed + i,
                maxMsaSequences: Math.max(1, chainSeqs.length + 1),
                maxExtraSequences: 1,
                maskFraction: 0,
                chainLengths,
                chainSequences: chainSeqs,
                chainAware: true,
                outerProductMeanFirst: true,
                positionScale: 20,
                tolerance,
              }
            : {
                recycles,
                randomSeed: seed + i,
                maxMsaSequences: Math.max(1, chainSeqs.length + 1),
                maxExtraSequences: 1,
                maskFraction: 0,
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
          const plddt = new Float32Array(fin.confidence.plddt as Float32Array);
          let sumPlddt = 0;
          for (let k = 0; k < plddt.length; k++) sumPlddt += plddt[k];
          const meanPlddt = plddt.length ? sumPlddt / plddt.length : 0;

          const chainIds = chainIdsFromLengths(chainLengths);
          const uniqueChains = [...new Set(chainIds)];
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

          folds.push({
            sequenceId: cand.id,
            key: cand.key ?? cand.id,
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
            elapsedMs: performance.now() - foldStart,
          });
        }
      } else {
        // AF3-lineage unified driver (RF3, Boltz-2, OpenBind-0, Protenix-v2, OpenDDE, Mol-1, IntelliFold-2, AF3)
        const dialect = loadedAf3.weights.trunk.dialect;
        const featOpts = lfAf3.featuriserDialect(dialect);

        // Optional ligand CCD passthrough for AF3-lineage holo refolding
        const parsedLigands: unknown[] = [];
        if (ligandCcdCodes.length > 0 && lfAf3.ccdUrl && lfAf3.parseCcdComponent) {
          const textDec = new TextDecoder();
          for (const code of ligandCcdCodes) {
            try {
              const url = lfAf3.ccdUrl(code);
              const bytes = await fetchCachedBytes(`lf-ccd-${code}`, url);
              parsedLigands.push(lfAf3.parseCcdComponent(textDec.decode(bytes)));
            } catch (err) {
              console.warn(`[localfold] Failed to fetch/parse CCD ligand "${code}":`, err);
            }
          }
        }

        for (let i = 0; i < targets.length; i++) {
          const cand = targets[i];
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
            detail: `Folding ${cand.id} with ${modelKey} (${cleanSeq.length} aa${parsedLigands.length ? ` + ${ligandCcdCodes.join(',')}` : ''}${dnaSeqs.length ? ` + ${dnaSeqs.length} DNA` : ''}, ${recycles} rec, ${diffusionSteps} steps)...`,
          });

          const batch = lfAf3.featuriseProtein(
            colonSeq,
            hasExtraContext
              ? {
                  ...featOpts,
                  chainKinds,
                  ...(parsedLigands.length > 0 ? { ligands: parsedLigands } : {}),
                }
              : featOpts,
          );
          const result = await lfAf3.foldBatch(gpuDevice, batch, loadedAf3.weights, {
            mode: 'diffusion',
            steps: diffusionSteps,
            recycles,
            seed: seed + i,
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
          const meanPlddt =
            typeof result.meanPlddt === 'number' && Number.isFinite(result.meanPlddt)
              ? result.meanPlddt
              : plddt.length
                ? sumPlddt / plddt.length
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

          folds.push({
            sequenceId: cand.id,
            key: cand.key ?? cand.id,
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
            elapsedMs: performance.now() - foldStart,
          });
        }
      }

      const runHash = fnv1a(
        JSON.stringify({
          seqSetId: seqSet.id,
          modelKey,
          recycles,
          tolerance: isAf3 ? null : tolerance,
          diffusionSteps: isAf3 ? diffusionSteps : null,
          maxSequences: maxSeqs,
          seed,
          ligandCcdCodes: isAf3 ? ligandCcdCodes : [],
          dnaSeqs: isAf3 ? dnaSeqs : [],
          rnaSeqs: isAf3 ? rnaSeqs : [],
        }),
      );

      const collectionKind = seqSet.collectionKind ?? 'independent-candidates';
      const rep = selectFoldRepresentative(folds, { collectionKind });
      const itemOutcomes: CollectionItemOutcome[] = [
        ...folds.map((f): CollectionItemOutcome => ({
          key: f.key ?? f.sequenceId,
          status: 'completed',
        })),
        ...work.excludedCandidates.map((ex): CollectionItemOutcome => ({
          key: ex.key,
          status: 'excluded',
          reason: ex.reason,
        })),
      ];

      const out: FoldedStructure = {
        kind: 'FoldedStructure',
        id: `folds-${modelKey}-${seqSet.id}-r${recycles}-${runHash}`,
        parentStructureId: seqSet.parentStructureId,
        sourceStructureId,
        collectionKind,
        ...(work.excludedCandidates.length > 0 ? { excludedCandidates: work.excludedCandidates } : {}),
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
            ? `pLDDT ${rep.selection.scoreValue.toFixed(1)}`
            : 'no confidence';
        bestStructure = {
          kind: 'StructureArtifact',
          id: `foldstruct-${modelKey}-${bestFold.sequenceId}-${runHash}`,
          label: `${bestFold.sequenceId} (${modelKey}, ${confTag})`,
          sourceModelId: 'localfold-af2',
          parentStructureId: sourceStructureId,
          representativeSelection: rep.selection,
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
