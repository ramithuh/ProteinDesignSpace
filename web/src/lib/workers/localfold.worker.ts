/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping sokrypton/localfold (AlphaFold2 monomer INT5)
// loaded directly from /vendor/localfold/ as plain ES modules.

import { atom37ToPdb } from '$lib/protein-ops/pdb';
import type { FoldedCandidate, FoldedStructure, SequenceSet } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let lfModules: any = null;
let gpuDevice: GPUDevice | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadedAf2: any = null;

async function importVendor(relPath: string) {
  const url = `${self.location.origin}/vendor/localfold/${relPath}`;
  return await import(/* @vite-ignore */ url);
}

async function ensureLoaded(id: number) {
  if (loadedAf2 && gpuDevice) return;

  if (!lfModules) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading LocalFold WebGPU modules...',
    });
    const [monomerMod, fixtureMod, storeMod, manifestsMod, deviceMod] = await Promise.all([
      importVendor('webgpu/af2/model/monomer.js'),
      importVendor('shared/bundles/alphafold-fixture.js'),
      importVendor('shared/bundles/http-tensor-store.js'),
      importVendor('shared/bundles/manifests/index.js'),
      importVendor('webgpu/runtime/device.js'),
    ]);
    lfModules = {
      AlphaFoldMonomerGpu: monomerMod.AlphaFoldMonomerGpu,
      AlphaFoldFixture: fixtureMod.AlphaFoldFixture,
      HttpTensorStore: storeMod.HttpTensorStore,
      bundleBaseUrl: manifestsMod.bundleBaseUrl,
      loadManifest: manifestsMod.loadManifest,
      requestAlphaFoldDevice: deviceMod.requestAlphaFoldDevice,
    };
  }

  if (!gpuDevice) {
    if (!('gpu' in navigator) || !navigator.gpu) {
      throw new Error('WebGPU is required for LocalFold AF2 but is not available in this browser');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No compatible WebGPU adapter found for LocalFold');
    gpuDevice = await lfModules.requestAlphaFoldDevice(adapter, { memoryBudgetBytes: null });
    void gpuDevice!.lost.then(() => {
      gpuDevice = null;
    });
  }

  if (!loadedAf2) {
    const manifest = await lfModules.loadManifest('monomer');
    const base = lfModules.bundleBaseUrl('monomer');
    const store = await lfModules.HttpTensorStore.fromManifest(
      base,
      manifest,
      (p: { loadedBytes: number; totalBytes: number }) => {
        const frac = p.totalBytes > 0 ? p.loadedBytes / p.totalBytes : 0;
        const mb = (p.loadedBytes / 1024 / 1024).toFixed(1);
        const totMb = (p.totalBytes / 1024 / 1024).toFixed(0);
        self.postMessage({
          id,
          type: 'progress',
          phase: 'weights',
          progress: frac,
          detail: `LocalFold AF2 INT5 weights: ${mb} / ${totMb} MB`,
        });
      },
    );
    store.prefetch?.();

    const fixture = lfModules.AlphaFoldFixture.fromStore(store);
    const [
      embedding,
      template,
      extraStack,
      mainStack,
      structure,
      confidence,
      geometry,
      featureTables,
      paeBreaks,
    ] = await Promise.all([
      fixture.embeddingWeights(),
      fixture.templateWeights(),
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
      featureTables,
      paeBreaks,
      weights: {
        embedding,
        template,
        templateEmbedding: undefined,
        extraStack,
        mainStack,
        structure,
        lddt: confidence.lddt,
        pae: confidence.pae,
        geometry,
        distogram: undefined,
      },
    };
  }
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'dispose') {
      loadedAf2?.store?.releaseDecoded?.();
      loadedAf2 = null;
      gpuDevice?.destroy();
      gpuDevice = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const seqSet = msg.inputs?.sequences as SequenceSet | undefined;
      if (!seqSet?.candidates?.length) {
        throw new Error('LocalFold AF2 requires an upstream SequenceSet with at least one sequence');
      }

      const p = msg.params ?? {};
      const recycles = Math.max(0, Math.min(6, Number(p.recycles ?? 1)));
      const maxSeqs = Math.max(1, Math.min(8, Number(p.maxSequences ?? 2)));
      const seed = Number(p.seed ?? 0);

      await ensureLoaded(msg.id);
      const runner = new lfModules.AlphaFoldMonomerGpu(gpuDevice);
      const targets = seqSet.candidates.slice(0, maxSeqs);
      const folds: FoldedCandidate[] = [];

      for (let i = 0; i < targets.length; i++) {
        const cand = targets[i];
        const rawUpper = cand.sequence.toUpperCase();
        const validCount = (rawUpper.match(/[ACDEFGHIKLMNPQRSTVWY]/g) ?? []).length;
        if (rawUpper.length > 0 && validCount / rawUpper.length < 0.5) {
          throw new Error(
            `Refusing to fold "${cand.id}": ${rawUpper.length - validCount}/${rawUpper.length} residues are unknown/non-canonical (would collapse to poly-ALA). Check the upstream generator output.`,
          );
        }
        const cleanSeq = rawUpper.replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, 'A');
        const a3m = `>query\n${cleanSeq}\n`;
        const foldStart = performance.now();

        self.postMessage({
          id: msg.id,
          type: 'progress',
          phase: 'fold',
          progress: i / targets.length,
          detail: `Folding ${cand.id} (${cleanSeq.length} aa, ${recycles} recycles)...`,
        });

        const prediction = await runner.predictA3m(
          a3m,
          loadedAf2.weights,
          loadedAf2.featureTables,
          {
            recycles,
            randomSeed: seed + i,
            maxMsaSequences: 1,
            maxExtraSequences: 1,
            chainLengths: [cleanSeq.length],
            tolerance: 0.5,
          },
          loadedAf2.paeBreaks,
          undefined,
          (prog: { completed: number; total: number }) => {
            const sub = prog.total > 0 ? prog.completed / prog.total : 0;
            self.postMessage({
              id: msg.id,
              type: 'progress',
              phase: 'fold',
              progress: (i + sub) / targets.length,
              detail: `Folding ${cand.id} (${i + 1}/${targets.length}): block ${prog.completed}/${prog.total}`,
            });
          },
        );

        const fin = prediction.final;
        const plddtRaw = fin.confidence.plddt as Float32Array;
        // Ensure pLDDT is on 0..100 scale
        const maxP = plddtRaw.reduce((a, b) => (b > a ? b : a), 0);
        const plddt =
          maxP <= 1.5
            ? Float32Array.from(plddtRaw, (v) => v * 100)
            : new Float32Array(plddtRaw);
        let sumPlddt = 0;
        for (let k = 0; k < plddt.length; k++) sumPlddt += plddt[k];
        const meanPlddt = plddt.length ? sumPlddt / plddt.length : 0;

        const { pdbText, caCoords } = atom37ToPdb(
          fin.structure.atom37 as Float32Array,
          fin.structure.atom37Mask as Float32Array,
          cleanSeq,
          plddt,
        );

        const paeRaw = fin.confidence.predictedAlignedError as Float32Array | undefined;
        const ptmVal =
          typeof fin.confidence.ptm === 'number'
            ? fin.confidence.ptm
            : typeof fin.confidence.predictedTmScore === 'number'
              ? fin.confidence.predictedTmScore
              : null;

        folds.push({
          sequenceId: cand.id,
          sequence: cleanSeq,
          foldModelId: 'localfold-af2-monomer-int5',
          pdbText,
          caCoords,
          plddt,
          meanPlddt,
          ptm: ptmVal,
          pae: paeRaw ? new Float32Array(paeRaw) : undefined,
          elapsedMs: performance.now() - foldStart,
        });
      }

      const out: FoldedStructure = {
        kind: 'FoldedStructure',
        id: `folds-${seqSet.id}-r${recycles}`,
        parentStructureId: seqSet.parentStructureId,
        folds,
      };

      self.postMessage({
        id: msg.id,
        type: 'result',
        outputs: { folds: out },
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
