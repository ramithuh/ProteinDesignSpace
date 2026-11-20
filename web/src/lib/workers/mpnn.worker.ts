/// <reference lib="webworker" />
import {
  ProteinMPNN,
  modelManifestURL,
  type ModelFamily,
  type ResidueRef,
} from 'proteinmpnn-web';
import { makeCachedFetch } from '$lib/engine/opfs-cache';
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const WASM_PATHS = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';

let model: ProteinMPNN | null = null;
let loadedKey: string | null = null;

/** Parse human-friendly fixed-position strings like "A10,A12-15,20" into ResidueRef[]. */
function parseFixedPositions(spec: string, defaultChain = 'A'): ResidueRef[] {
  const out: ResidueRef[] = [];
  const trimmed = spec.trim();
  if (!trimmed) return out;
  for (const raw of trimmed.split(/[,\s]+/)) {
    if (!raw) continue;
    const m = /^([A-Za-z])?:?(\d+)(?:-(\d+))?$/.exec(raw);
    if (!m) continue;
    const chain = m[1] || defaultChain;
    const start = Number(m[2]);
    const end = m[3] ? Number(m[3]) : start;
    for (let pos = start; pos <= end; pos++) {
      out.push({ chain, number: pos });
    }
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
  let fetched = 0;
  self.postMessage({
    id,
    type: 'progress',
    phase: 'weights',
    progress: 0.1,
    detail: `Loading ${family}/${modelName} (~9.5 MB from HF)...`,
  });

  model = await ProteinMPNN.load(url, {
    backend: 'auto',
    wasmPaths: WASM_PATHS,
    numThreads: 1,
    fetch: makeCachedFetch((_u, fromCache) => {
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
      const fixedPositions = parseFixedPositions(String(p.fixedPositions ?? ''));

      await ensureModel(msg.id, family, modelName);
      if (!model) throw new Error('ProteinMPNN model not initialized');

      self.postMessage({
        id: msg.id,
        type: 'progress',
        phase: 'encode',
        progress: 0.05,
        detail: `Encoding ${structure.length}-aa backbone graph...`,
      });

      const prepared = await model.preparePDB(structure.pdbText);
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
        candidates.push({
          id: `mpnn-${i + 1}`,
          sequence: res.sequence,
          score: res.score,
          globalScore: res.globalScore,
          sourceModelId: `${family}/${modelName}`,
          temperature,
          seed: seed + i,
        });
      }

      candidates.sort((a, b) => (a.score ?? 99) - (b.score ?? 99));

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `seqset-${structure.id}-${family}-${seed}`,
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
