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
import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const WASM_PATHS = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';

let model: ProteinMPNN | null = null;
let loadedKey: string | null = null;

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

async function ensureModel(id: number, family: ModelFamily, modelName: string) {
  const key = `${family}:${modelName}`;
  if (model && loadedKey === key) return;
  if (model) {
    await model.dispose().catch(() => {});
    model = null;
  }

  const url = modelManifestURL(family, modelName);
  self.postMessage({
    id,
    type: 'progress',
    phase: 'weights',
    progress: 0.1,
    detail: `Loading ${family}/${modelName} (~9.5 MB from HF)...`,
  });

  const loadOnce = async (retried: boolean): Promise<ProteinMPNN> => {
    let fetched = 0;
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    try {
      return await ProteinMPNN.load(url, {
        backend: 'auto',
        wasmPaths: WASM_PATHS,
        numThreads: 1,
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
              ? `Loaded ${family}/${modelName} asset ${fetched}/5 from cache`
              : `Downloaded ${family}/${modelName} asset ${fetched}/5`,
          });
        }),
      });
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        /SHA-256|integrity|manifest|embedding size/i.test(errMsg)
      ) {
        await evictCachedUrls(fetchedUrls);
        return await loadOnce(true);
      }
      throw err;
    }
  };

  model = await loadOnce(false);
  loadedKey = key;
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

      await ensureModel(msg.id, family, modelName);
      if (!model) throw new Error('ProteinMPNN model not initialized');

      self.postMessage({
        id: msg.id,
        type: 'progress',
        phase: 'encode',
        progress: 0.05,
        detail: `Encoding ${structure.length}-aa backbone graph...`,
      });

      const rawParsed = parsePDB(structure.pdbText, {
        fillGaps: false,
        atomMode: model.manifest.atomMode,
      });
      const noisyCoords =
        backboneNoise > 0
          ? applyBackboneNoise(rawParsed.coordinates, backboneNoise, seed)
          : rawParsed.coordinates;
      const gapAwareIndices = buildGapAwareResidueIndices(rawParsed);
      const prepared = await model.prepare({
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
              detail: `Designing seq ${i + 1}/${numSequences} (${completed}/${total} aa)`,
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

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `seqset-${structure.id}-${family}-${seed}-${runHash}`,
        parentStructureId: structure.id,
        sourceStructureId: structure.id,
        sourceModelId: `${family}/${modelName}`,
        ...(propagatedLockedChains.length > 0 ? { lockedChains: propagatedLockedChains } : {}),
        ...(structure.chainRoles ? { chainRoles: structure.chainRoles } : {}),
        candidates,
      };

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
