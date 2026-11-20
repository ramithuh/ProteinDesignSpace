/// <reference lib="webworker" />
import {
  ProteinMPNN,
  modelManifestURL,
  type ModelFamily,
  type ResidueRef,
} from 'proteinmpnn-web';
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

function computeChainLengths(chainIds: readonly string[]): number[] {
  const lengths: number[] = [];
  for (let i = 0; i < chainIds.length; i++) {
    if (i === 0 || chainIds[i] !== chainIds[i - 1]) {
      lengths.push(1);
    } else {
      lengths[lengths.length - 1]++;
    }
  }
  return lengths;
}

/**
 * Parse human-friendly fixed-position strings into ResidueRef[].
 * Supports: "A10", "A10-15", "A10-A15", reversed "A15-10", negative resids ("A-2", "A-2-5", "A-5--2"),
 * insertion codes ("A100A", "100A"), lowercase chain normalization ("a10" -> "A"),
 * and default chain resolution (including "_" for blank-chain PDBs).
 */
function parseFixedPositions(spec: string, defaultChain = 'A'): ResidueRef[] {
  const out: ResidueRef[] = [];
  const trimmed = spec.trim();
  if (!trimmed) return out;
  for (const raw of trimmed.split(/[,\s]+/)) {
    if (!raw) continue;
    const rangeMatch = /^([A-Za-z_])?:?(-?\d+)-([A-Za-z_])?:?(-?\d+)$/.exec(raw);
    if (rangeMatch) {
      const startChain = rangeMatch[1] ? rangeMatch[1].toUpperCase() : null;
      const endChain = rangeMatch[3] ? rangeMatch[3].toUpperCase() : null;
      if (startChain && endChain && startChain !== endChain) {
        throw new Error(
          `Invalid fixed position "${raw}": range must stay within a single chain (${startChain} vs ${endChain})`,
        );
      }
      const chain = startChain ?? endChain ?? defaultChain;
      const a = Number(rangeMatch[2]);
      const b = Number(rangeMatch[4]);
      const start = Math.min(a, b);
      const end = Math.max(a, b);
      if (end - start > 100000) {
        throw new Error(`Invalid fixed position "${raw}": residue range too large`);
      }
      for (let pos = start; pos <= end; pos++) {
        out.push({ chain, number: pos });
      }
      continue;
    }

    const singleMatch = /^([A-Za-z_])?:?(-?\d+)([A-Za-z])?$/.exec(raw);
    if (singleMatch) {
      const chain = singleMatch[1] ? singleMatch[1].toUpperCase() : defaultChain;
      const number = Number(singleMatch[2]);
      const insertionCode = singleMatch[3];
      out.push(insertionCode ? { chain, number, insertionCode } : { chain, number });
      continue;
    }

    throw new Error(
      `Invalid fixed position token "${raw}". Expected e.g. "A10", "A10-15", "A10-A15", "A-2-5", or "A100A".`,
    );
  }
  return out;
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
      const family = (p.family ?? 'solublempnn') as ModelFamily;
      const modelName = String(p.modelName ?? 'v_48_020');
      const numSequences = Math.max(1, Math.min(16, Number(p.numSequences ?? 2)));
      const temperature = Math.max(0.01, Math.min(2.0, Number(p.temperature ?? 0.1)));
      const seed = Number(p.seed ?? 42);
      const fixedPositionsSpec = String(p.fixedPositions ?? '').trim();

      await ensureModel(msg.id, family, modelName);
      if (!model) throw new Error('ProteinMPNN model not initialized');

      self.postMessage({
        id: msg.id,
        type: 'progress',
        phase: 'encode',
        progress: 0.05,
        detail: `Encoding ${structure.length}-aa backbone graph...`,
      });

      const prepared = await model.preparePDB(structure.pdbText, { fillGaps: false });
      const defaultChain = prepared.structure.chainIds[0] ?? 'A';
      const fixedPositions = parseFixedPositions(fixedPositionsSpec, defaultChain);
      const chainLengths = computeChainLengths(prepared.structure.chainIds);
      const candidates: SequenceCandidate[] = [];

      for (let i = 0; i < numSequences; i++) {
        const res = await prepared.sample({
          temperature,
          seed: seed + i,
          fixedPositions: fixedPositions.length > 0 ? fixedPositions : undefined,
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
          seed,
          fixedPositions,
        }),
      );

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `seqset-${structure.id}-${family}-${seed}-${runHash}`,
        parentStructureId: structure.id,
        sourceModelId: `${family}/${modelName}`,
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
