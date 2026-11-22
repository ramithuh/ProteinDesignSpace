/// <reference lib="webworker" />
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
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

// First-party same-origin ONNX Runtime 1.24.3 WASM/MJS binaries synced from npm-locked node_modules
// (zero third-party CDN code execution; all files < 25 MiB Cloudflare Pages limit).
const WASM_PATHS = new URL('/vendor/ort/', self.location.origin).href;

let model: ProteinMPNN | null = null;
let loadedKey: string | null = null;
let loadedBackendMode: 'auto' | 'wasm' = 'auto';

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

async function ensureModel(
  id: number,
  family: ModelFamily,
  modelName: string,
  backendOverride: 'auto' | 'wasm' = 'auto',
) {
  const key = `${family}:${modelName}:${backendOverride}`;
  if (model && loadedKey === key) return;
  if (model) {
    await model.dispose().catch(() => {});
    model = null;
    loadedKey = null;
  }

  const url = modelManifestURL(family, modelName);
  const backendTag = backendOverride === 'wasm' ? ' (WASM CPU fallback)' : '';
  self.postMessage({
    id,
    type: 'progress',
    phase: 'weights',
    progress: 0.1,
    detail: `Loading ${family}/${modelName}${backendTag} (~9.5 MB from HF)...`,
  });

  const loadOnce = async (retried: boolean): Promise<ProteinMPNN> => {
    let fetched = 0;
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    try {
      // Suppress non-fatal ONNX Runtime WARNING logs (which ORT Web emits via console.error
      // in DevTools) while preserving genuine ERROR/FATAL diagnostics.
      try {
        const ortWeb = await import('onnxruntime-web');
        if (ortWeb?.env) ortWeb.env.logLevel = 'error';
      } catch {
        /* ignore if unavailable */
      }
      try {
        const ortGpu = await import('onnxruntime-web/webgpu');
        if (ortGpu?.env) ortGpu.env.logLevel = 'error';
      } catch {
        /* ignore if unavailable */
      }
      return await ProteinMPNN.load(url, {
        backend: backendOverride,
        wasmPaths: WASM_PATHS,
        numThreads: 1,
        sessionOptions: { logSeverityLevel: 3 },
        fetch: makeCachedFetch((u, fromCache) => {
          fetchedUrls.push(u);
          if (fromCache) anyFromCache = true;
          fetched++;
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.95, fetched / 5),
            detail: fromCache
              ? `Loaded ${family}/${modelName}${backendTag} asset ${fetched}/5 from cache`
              : `Downloaded ${family}/${modelName}${backendTag} asset ${fetched}/5`,
          });
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
  loadedBackendMode = backendOverride;
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'dispose') {
      if (model) {
        await model.dispose().catch(() => {});
        model = null;
      }
      loadedKey = null;
      loadedBackendMode = 'auto';
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const structure = msg.inputs?.structure as StructureArtifact | undefined;
      if (!structure?.pdbText) throw new Error('ProteinMPNN requires an upstream StructureArtifact');

      const p = msg.params ?? {};
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
        await ensureModel(msg.id, family, modelName, backendOverride);
        if (!model) throw new Error('ProteinMPNN model not initialized');
        const activeModel = model;

        const collectionItems =
          structure.collection?.items && structure.collection.items.length > 1
            ? structure.collection.items
            : null;

        if (!collectionItems) {
          // Legacy single-backbone path: exact existing behavior & IDs
          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'encode',
            progress: 0.05,
            detail: `Encoding ${structure.length}-aa backbone graph...`,
          });

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
              lockedChains: structure.lockedChains,
              fixedResidueIndices: structure.fixedResidueIndices,
              chainRoles: structure.chainRoles,
            },
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
                self.postMessage({
                  id: msg.id,
                  type: 'progress',
                  phase: 'sample',
                  progress: frac,
                  detail: `Designing seq ${i + 1}/${numSequences} (${completed}/${total} aa)${backendOverride === 'wasm' ? ' · WASM' : ''}`,
                });
              },
            });
            const candChainLengths =
              chainLengths.length > 0 &&
              chainLengths.reduce((a, b) => a + b, 0) === res.sequence.length
                ? chainLengths
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
        const totalPlanned = Math.max(1, fanoutPlan.totalPlannedSequences);
        let completedSeqs = 0;

        for (let bIdx = 0; bIdx < fanoutPlan.perBackbonePlans.length; bIdx++) {
          const subplan = fanoutPlan.perBackbonePlans[bIdx];
          const backbone = backboneById.get(subplan.sourceStructure.artifactId) ?? collectionItems[bIdx];
          if (!backbone?.pdbText) {
            throw new Error(`Backbone "${subplan.sourceStructure.artifactId}" is missing PDB text for ProteinMPNN.`);
          }

          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'encode',
            progress: completedSeqs / totalPlanned,
            detail: `Backbone ${bIdx + 1}/${fanoutPlan.backboneCount} · encoding ${backbone.length}-aa graph...`,
          });

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
              lockedChains: [...subplan.lockedChains],
              fixedResidueIndices: [...subplan.fixedResidueIndices],
              chainRoles: subplan.chainRoles ? { ...subplan.chainRoles } : undefined,
            },
            rawParams,
            prepared.structure,
          );
          const {
            designChains,
            fixedPositions,
            omitAminoAcids,
            tiedPositions,
          } = compiled;

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
                self.postMessage({
                  id: msg.id,
                  type: 'progress',
                  phase: 'sample',
                  progress: frac,
                  detail: `Backbone ${bIdx + 1}/${fanoutPlan.backboneCount} · seq ${q + 1}/${subplan.items.length} (${completed}/${total} aa)${backendOverride === 'wasm' ? ' · WASM' : ''}`,
                });
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
          ...(structure.lockedChains && structure.lockedChains.length > 0
            ? { lockedChains: [...structure.lockedChains] }
            : {}),
          ...(structure.chainRoles ? { chainRoles: { ...structure.chainRoles } } : {}),
          candidates: allCandidates,
        };
      };

      let seqSet: SequenceSet;
      try {
        seqSet = await executeRun(loadedBackendMode);
      } catch (runErr) {
        if (loadedBackendMode !== 'wasm' && isWebGpuDeviceLostError(runErr)) {
          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'weights',
            progress: 0.05,
            detail: 'WebGPU session lost — retrying ProteinMPNN on WASM CPU fallback...',
          });
          seqSet = await executeRun('wasm');
        } else {
          throw runErr;
        }
      }

      self.postMessage({
        id: msg.id,
        type: 'result',
        outputs: { sequences: seqSet },
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
