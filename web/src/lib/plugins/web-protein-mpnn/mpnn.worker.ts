/// <reference lib="webworker" />
// Layer 3 Plugin Worker — web-protein-mpnn (ONNX Runtime WebGPU + WASM fallback)
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress, and disposal.

import {
  ProteinMPNN,
  modelManifestURL,
  parsePDB,
  type ModelFamily,
} from 'proteinmpnn-web';
import {
  applyBackboneNoise,
  buildGapAwareResidueIndices,
  compileMpnnInputSpec,
  computeChainLengths,
} from '$lib/engine/mpnn-spec';
import {
  computeStructurePayloadDigest,
  type CollectionMember,
} from '$lib/engine/collections';
import { deriveStableItemSeed, planMpnnFanout } from '$lib/engine/work-plan';
import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { isWebGpuDeviceLostError } from '$lib/engine/webgpu-device';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

// First-party same-origin ONNX Runtime 1.24.3 WASM/MJS binaries synced from npm-locked node_modules
// (zero third-party CDN code execution; all files < 25 MiB Cloudflare Pages limit).
const WASM_PATHS = new URL('/vendor/ort/', self.location.origin).href;

let model: ProteinMPNN | null = null;
let loadedKey: string | null = null;
let loadedBackendMode: 'auto' | 'wasm' = 'auto';

function supportsWebAssemblyJspi(): boolean {
  return (
    typeof WebAssembly !== 'undefined' &&
    typeof (WebAssembly as unknown as { Suspending?: unknown }).Suspending === 'function'
  );
}

function isOrtWebGpuFallbackError(err: unknown): boolean {
  if (isWebGpuDeviceLostError(err)) return true;
  const msg = err instanceof Error ? err.message : String(err);
  return /no available backend|initWasm|webgpu|jspi|suspending|requestAdapter|adapter/i.test(msg);
}

async function ensureModel(
  ctx: WorkerRunContext,
  family: ModelFamily,
  modelName: string,
  backendOverride: 'auto' | 'wasm' = 'auto',
) {
  const effectiveBackend: 'auto' | 'wasm' =
    backendOverride === 'wasm' ||
    !supportsWebAssemblyJspi() ||
    typeof navigator === 'undefined' ||
    !('gpu' in navigator)
      ? 'wasm'
      : 'auto';
  const key = `${family}:${modelName}:${effectiveBackend}`;
  if (model && loadedKey === key) return;
  if (model) {
    await model.dispose().catch(() => {});
    model = null;
    loadedKey = null;
  }

  const url = modelManifestURL(family, modelName);
  const backendTag = effectiveBackend === 'wasm' ? ' (WASM CPU)' : '';
  ctx.progress('weights', 0.1, `Loading ${family}/${modelName}${backendTag} (~9.5 MB from HF)...`);

  const loadOnce = async (retried: boolean): Promise<ProteinMPNN> => {
    let fetched = 0;
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    try {
      // Suppress non-fatal ONNX Runtime WARNING logs (which ORT Web emits via console.error
      // in DevTools) while preserving genuine ERROR/FATAL diagnostics.
      try {
        const ortMod =
          effectiveBackend === 'wasm'
            ? await import('onnxruntime-web')
            : await import('onnxruntime-web/webgpu');
        if (ortMod?.env) ortMod.env.logLevel = 'error';
      } catch {
        /* ignore if unavailable */
      }
      return await ProteinMPNN.load(url, {
        backend: effectiveBackend,
        wasmPaths: WASM_PATHS,
        numThreads: 1,
        sessionOptions: { logSeverityLevel: 3 },
        fetch: makeCachedFetch((u, fromCache) => {
          fetchedUrls.push(u);
          if (fromCache) anyFromCache = true;
          fetched++;
          ctx.progress(
            'weights',
            Math.min(0.95, fetched / 5),
            fromCache
              ? `Loaded ${family}/${modelName}${backendTag} asset ${fetched}/5 from cache`
              : `Downloaded ${family}/${modelName}${backendTag} asset ${fetched}/5`,
          );
        }),
      });
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        fetchedUrls.length > 0 &&
        /SHA-256|integrity|manifest|embedding size/i.test(errMsg)
      ) {
        await evictCachedUrls(fetchedUrls);
        if (anyFromCache) {
          return await loadOnce(true);
        }
      }
      throw err;
    }
  };

  model = await loadOnce(false);
  loadedKey = key;
  loadedBackendMode = effectiveBackend;
}

createWorkerHarness({
  async run(ctx) {
    const structure = ctx.inputs?.structure as StructureArtifact | undefined;
    if (!structure?.pdbText) throw new Error('ProteinMPNN requires an upstream StructureArtifact');

    const p = ctx.params ?? {};
    // Keep explicit p.<key> reads and aligned bounds intact for param-schema & param-validation guards.
    // Constraint parsing (parseFixedPositionsAndChains, parseTiedPositions) & lock precedence live in mpnn-spec.ts.
    const family = (p.family ?? 'solublempnn') as ModelFamily;
    const modelName = String(p.modelName ?? 'v_48_020');
    const numSequences = Math.max(1, Math.min(8, Math.round(Number(p.numSequences ?? 2))));
    const temperature = Math.max(0.01, Math.min(1.5, Number(p.temperature ?? 0.1)));
    const rawNoise = Number(p.backboneNoise ?? 0);
    const backboneNoise = Number.isFinite(rawNoise) ? Math.max(0, Math.min(1, rawNoise)) : 0;
    const seed = Number(p.seed ?? 42);
    const rawParams: Record<string, unknown> = {
      family,
      modelName,
      numSequences,
      temperature,
      backboneNoise,
      seed,
      fixedPositions: p.fixedPositions,
      omitAminoAcids: p.omitAminoAcids,
      tiedPositions: p.tiedPositions,
      designChains: p.designChains,
      unlockInherited: p.unlockInherited,
      ignoreInheritedLocks: p.ignoreInheritedLocks,
    };

    const executeRun = async (backendOverride: 'auto' | 'wasm'): Promise<SequenceSet> => {
      await ensureModel(ctx, family, modelName, backendOverride);
      if (!model) throw new Error('ProteinMPNN model not initialized');
      const activeModel = model;

      const collectionItems =
        structure.collection?.items && structure.collection.items.length > 1
          ? structure.collection.items
          : null;

      if (!collectionItems) {
        // Legacy single-backbone path: exact existing behavior & IDs
        ctx.progress('encode', 0.05, `Encoding ${structure.length}-aa backbone graph...`);

        const rawParsed = parsePDB(structure.pdbText, {
          fillGaps: false,
          atomMode: activeModel.manifest.atomMode,
        });
        const noisyCoords =
          backboneNoise > 0
            ? applyBackboneNoise(rawParsed.coordinates, backboneNoise, seed)
            : rawParsed.coordinates;
        const gapAwareIndices = buildGapAwareResidueIndices(rawParsed);
        const prepared = await activeModel.prepare({
          ...rawParsed,
          coordinates: noisyCoords,
          residueIndices: gapAwareIndices,
        });

        const compiled = compileMpnnInputSpec(
          {
            pdbText: structure.pdbText,
            sourceResidues: structure.sourceResidues,
            lockedChains: structure.lockedChains,
            fixedResidueIndices: structure.fixedResidueIndices,
            tiedPositions: structure.tiedPositions,
            chainRoles: structure.chainRoles,
          } as Parameters<typeof compileMpnnInputSpec>[0],
          rawParams,
          prepared.structure,
        );
        const {
          designChains,
          fixedPositions,
          omitAminoAcids,
          tiedPositions,
          ignoreInheritedLocks,
          lockedChains: propagatedLockedChains,
        } = compiled;

        const chainLengths = computeChainLengths(prepared.structure.chainIds);
        const uniqueChainIds: string[] = [];
        for (const c of prepared.structure.chainIds) {
          if (uniqueChainIds.length === 0 || uniqueChainIds[uniqueChainIds.length - 1] !== c) {
            uniqueChainIds.push(c);
          }
        }
        const candidates: SequenceCandidate[] = [];

        for (let i = 0; i < numSequences; i++) {
          const res = await prepared.sample({
            temperature,
            seed: seed + i,
            ...(designChains ? { designChains } : {}),
            fixedPositions: fixedPositions.length > 0 ? fixedPositions : undefined,
            ...(omitAminoAcids ? { omitAminoAcids } : {}),
            ...(tiedPositions.length > 0 ? { tiedPositions } : {}),
            onProgress: ({ completed, total }) => {
              const frac = (i + completed / Math.max(1, total)) / numSequences;
              ctx.progress(
                'sample',
                frac,
                `Designing seq ${i + 1}/${numSequences} (${completed}/${total} aa)${backendOverride === 'wasm' ? ' · WASM' : ''}`,
              );
            },
          });
          const candChainLengths =
            chainLengths.length > 0 &&
            chainLengths.reduce((a, b) => a + b, 0) === res.sequence.length
              ? chainLengths
              : undefined;
          const candChainIds =
            candChainLengths && uniqueChainIds.length === candChainLengths.length
              ? uniqueChainIds
              : undefined;
          candidates.push({
            id: `mpnn-${i + 1}`,
            sequence: res.sequence,
            score: res.score,
            globalScore: res.globalScore,
            sourceModelId: `${family}/${modelName}`,
            temperature,
            seed: seed + i,
            ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
            ...(candChainIds ? { chainIds: candChainIds } : {}),
          });
        }

        candidates.sort((a, b) => (a.score ?? 99) - (b.score ?? 99));

        const runHash = fnv1a(
          JSON.stringify({
            structureId: structure.id,
            family,
            modelName,
            numSequences,
            temperature,
            backboneNoise,
            seed,
            designChains: designChains ?? null,
            fixedPositions,
            omitAminoAcids: omitAminoAcids ?? null,
            tiedPositions,
            ignoreInheritedLocks,
          }),
        );

        return {
          kind: 'SequenceSet',
          id: `seqset-${structure.id}-${family}-${seed}-${runHash}`,
          parentStructureId: structure.id,
          sourceStructureId: structure.id,
          sourceModelId: `${family}/${modelName}`,
          ...(propagatedLockedChains.length > 0 ? { lockedChains: propagatedLockedChains } : {}),
          ...(structure.chainRoles ? { chainRoles: structure.chainRoles } : {}),
          candidates,
        };
      }

      // Multi-backbone resident-weight sequential loop: prepare once per backbone, sample K sequentially
      const structureMembers: CollectionMember<StructureArtifact>[] = collectionItems.map((item, idx) => {
        const designKey = item.designKey ?? structure.collection?.keys?.[idx] ?? item.id;
        const designIndex = typeof item.designIndex === 'number' ? item.designIndex : idx;
        const payloadDigest = computeStructurePayloadDigest(item);
        return {
          key: designKey,
          ordinal: designIndex,
          memberKind: 'StructureArtifact',
          lineage: {
            designKey,
            designIndex,
            sourceStructure: {
              artifactId: item.id,
              payloadDigest,
              designKey,
            },
            parentKeys: [],
          },
          payload: item,
          summary: {
            label: item.label || item.id,
            residueCount: item.length,
          },
        };
      });

      const fanoutPlan = planMpnnFanout(structureMembers, {
        modelId: `${family}/${modelName}`,
        sequencesPerBackbone: numSequences,
        baseSeed: seed,
        temperature,
        scientificParams: {
          family,
          modelName,
          backboneNoise,
          fixedPositions: rawParams.fixedPositions,
          omitAminoAcids: rawParams.omitAminoAcids,
          tiedPositions: rawParams.tiedPositions,
          designChains: rawParams.designChains,
          unlockInherited: rawParams.unlockInherited,
        },
      });

      const backboneById = new Map(collectionItems.map((b) => [b.id, b]));
      const allCandidates: SequenceCandidate[] = [];
      const allPropagatedLocked = new Set<string>();
      const totalPlanned = Math.max(1, fanoutPlan.totalPlannedSequences);
      let completedSeqs = 0;

      for (let bIdx = 0; bIdx < fanoutPlan.perBackbonePlans.length; bIdx++) {
        const subplan = fanoutPlan.perBackbonePlans[bIdx];
        const backbone = backboneById.get(subplan.sourceStructure.artifactId) ?? collectionItems[bIdx];
        if (!backbone?.pdbText) {
          throw new Error(`Backbone "${subplan.sourceStructure.artifactId}" is missing PDB text for ProteinMPNN.`);
        }

        ctx.progress(
          'encode',
          completedSeqs / totalPlanned,
          `Backbone ${bIdx + 1}/${fanoutPlan.backboneCount} · encoding ${backbone.length}-aa graph...`,
        );

        const rawParsed = parsePDB(backbone.pdbText, {
          fillGaps: false,
          atomMode: activeModel.manifest.atomMode,
        });
        const noiseSeed = deriveStableItemSeed(seed, 'conditioning', subplan.designKey);
        const noisyCoords =
          backboneNoise > 0
            ? applyBackboneNoise(rawParsed.coordinates, backboneNoise, noiseSeed)
            : rawParsed.coordinates;
        const gapAwareIndices = buildGapAwareResidueIndices(rawParsed);
        const prepared = await activeModel.prepare({
          ...rawParsed,
          coordinates: noisyCoords,
          residueIndices: gapAwareIndices,
        });

        // Compile constraints STRICTLY for THIS backbone (zero cross-backbone lock leakage)
        const compiled = compileMpnnInputSpec(
          {
            pdbText: backbone.pdbText,
            sourceResidues: backbone.sourceResidues,
            lockedChains: [...subplan.lockedChains],
            fixedResidueIndices: [...subplan.fixedResidueIndices],
            ...(subplan.tiedPositions ? { tiedPositions: subplan.tiedPositions } : {}),
            chainRoles: subplan.chainRoles ? { ...subplan.chainRoles } : undefined,
          } as Parameters<typeof compileMpnnInputSpec>[0],
          rawParams,
          prepared.structure,
        );
        const {
          designChains,
          fixedPositions,
          omitAminoAcids,
          tiedPositions,
          lockedChains: compiledLockedChains,
        } = compiled;
        for (const ch of compiledLockedChains) allPropagatedLocked.add(ch);

        const chainLengths = computeChainLengths(prepared.structure.chainIds);
        const backboneCandidates: SequenceCandidate[] = [];

        for (let q = 0; q < subplan.items.length; q++) {
          const item = subplan.items[q];
          const itemSeed = deriveStableItemSeed(seed, 'mpnn', item.key);
          const seqOrdinalOffset = completedSeqs;
          const res = await prepared.sample({
            temperature,
            seed: itemSeed,
            ...(designChains ? { designChains } : {}),
            fixedPositions: fixedPositions.length > 0 ? fixedPositions : undefined,
            ...(omitAminoAcids ? { omitAminoAcids } : {}),
            ...(tiedPositions.length > 0 ? { tiedPositions } : {}),
            onProgress: ({ completed, total }) => {
              const frac = (seqOrdinalOffset + completed / Math.max(1, total)) / totalPlanned;
              ctx.progress(
                'sample',
                frac,
                `Backbone ${bIdx + 1}/${fanoutPlan.backboneCount} · seq ${q + 1}/${subplan.items.length} (${completed}/${total} aa)${backendOverride === 'wasm' ? ' · WASM' : ''}`,
              );
            },
          });
          completedSeqs++;
          const candChainLengths =
            chainLengths.length > 0 &&
            chainLengths.reduce((a, b) => a + b, 0) === res.sequence.length
              ? chainLengths
              : undefined;
          backboneCandidates.push({
            id: `d${subplan.designIndex + 1}-mpnn-${item.sequenceIndex + 1}`,
            key: item.key,
            sequenceKey: item.key,
            designKey: subplan.designKey,
            designIndex: subplan.designIndex,
            sourceStructureId: backbone.id,
            sampleIndex: item.sequenceIndex,
            sequence: res.sequence,
            score: res.score,
            globalScore: res.globalScore,
            sourceModelId: `${family}/${modelName}`,
            temperature,
            seed: itemSeed,
            ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
          });
        }

        // Sort within each backbone by score while keeping key/sequenceKey/seed immutable
        backboneCandidates.sort((a, b) => {
          const diff = (a.score ?? 99) - (b.score ?? 99);
          return diff !== 0 ? diff : (a.sampleIndex ?? 0) - (b.sampleIndex ?? 0);
        });
        allCandidates.push(...backboneCandidates);
      }

      return {
        kind: 'SequenceSet',
        id: `seqset-col-${fanoutPlan.planDigest.slice(0, 16)}`,
        parentStructureId: structure.id,
        sourceStructureId: structure.id,
        sourceStructureIds: collectionItems.map((b) => b.id),
        sourceModelId: `${family}/${modelName}`,
        collectionKind: 'independent-candidates',
        ...(allPropagatedLocked.size > 0
          ? { lockedChains: [...allPropagatedLocked] }
          : {}),
        ...(structure.chainRoles ? { chainRoles: { ...structure.chainRoles } } : {}),
        candidates: allCandidates,
      };
    };

    let seqSet: SequenceSet;
    try {
      seqSet = await executeRun(loadedBackendMode);
    } catch (runErr) {
      if (loadedBackendMode !== 'wasm' && isOrtWebGpuFallbackError(runErr)) {
        ctx.progress(
          'weights',
          0.05,
          'WebGPU session unavailable — retrying ProteinMPNN on WASM CPU fallback...',
        );
        seqSet = await executeRun('wasm');
      } else {
        throw runErr;
      }
    }

    return { outputs: { sequences: seqSet } };
  },

  async dispose() {
    if (model) {
      await model.dispose().catch(() => {});
      model = null;
    }
    loadedKey = null;
    loadedBackendMode = 'auto';
  },
});

export {};
