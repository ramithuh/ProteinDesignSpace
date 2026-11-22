import { normalizeLocalFoldChainScores } from '$lib/engine/conditioning';
import { parsePdbBackbone } from '$lib/protein-ops/pdb';

type ImportVendor = (path: string) => Promise<Record<string, any>>;

export async function loadEsmfold2(
  family: string,
  importVendor: ImportVendor,
  onWeights: (p: { loadedBytes: number; totalBytes: number }) => void,
) {
  const [model, fold, featuresApi, pdb] = await Promise.all([
    importVendor('web/esmfold2-model.js'),
    importVendor('webgpu/esmfold2/fold.js'),
    importVendor('shared/esmfold2/featurise.js'),
    importVendor('shared/af3/structure/pdb.js'),
  ]);
  // This upstream loader pairs each folding checkpoint with its own ESM-C tower and shim.
  const loaded = await model.loadEsmfold2Weights(onWeights, { family, languageModel: true });
  return async function predict(
    device: GPUDevice, chains: string[], steps: number, seed: number, signal: AbortSignal,
    onProgress: (progress: number, stage: string) => void,
  ) {
    let stage = 'Language model';
    const sequence = chains.join('');
    const result = await fold.foldEsmfold2(device, {
      sequence: chains.join(':'),
      shape: loaded.shape,
      weights: loaded.weights,
      confidenceWeights: loaded.confidenceWeights,
      tower: model.languageModelRunner(device, undefined, loaded, loaded.shape.pairChannels),
      languageModelMiB: loaded.language.megabytes,
      sampler: `diffusion-${steps}`,
      contacts: false,
      seed, signal,
      onStatus: (status: string) => { stage = status; },
      onProgress: (fraction: number) => onProgress(fraction, stage),
    });
    const { features, coordinates, confidence } = result;
    if (features.tokens !== sequence.length || coordinates.length !== features.atoms * 3 ||
        !confidence || confidence.plddt?.length !== sequence.length ||
        confidence.plddtPerAtom?.length !== features.atoms || confidence.pae?.length !== sequence.length ** 2) {
      throw new Error(`Invalid structure or confidence dimensions from ${family}.`);
    }
    // The learned Synthyra head emits atom and token pLDDT on [0,1].
    const plddt = Float32Array.from(confidence.plddt as Float32Array, (v) => v * 100);
    const denseConfidence = new Float32Array(features.tokens * features.dense);
    for (let i = 0; i < features.atoms; i++) {
      const slot = features.denseSlot[i];
      if (slot >= 0) denseConfidence[slot] = confidence.plddtPerAtom[i] * 100;
    }
    const densePositions = featuresApi.toDensePositions(features, coordinates) as Float32Array;
    if (!densePositions.every(Number.isFinite) || !denseConfidence.every(Number.isFinite) ||
        !plddt.every(Number.isFinite) || !confidence.pae.every(Number.isFinite) ||
        !Number.isFinite(confidence.ptm) || (chains.length > 1 && !Number.isFinite(confidence.iptm))) {
      throw new Error(`Nonfinite structure or confidence from ${family}.`);
    }
    const pdbText = pdb.toPdb(features.batch, densePositions, denseConfidence) as string;
    const parsed = parsePdbBackbone(pdbText);
    if (parsed.sequence !== sequence || parsed.diagnostics.chainCount !== chains.length) {
      throw new Error(`ESMFold2 exported residues or chains inconsistent with the input.`);
    }
    const ptm: number = confidence.ptm;
    const iptm: number | null = chains.length > 1 ? confidence.iptm : null;
    const meanPlddt = plddt.reduce((sum, v) => sum + v, 0) / plddt.length;
    return {
      sequence, pdbText, caCoords: parsed.caCoords, plddt, meanPlddt, confidence: meanPlddt,
      confidenceAvailable: true, ptm, iptm, pae: confidence.pae as Float32Array,
      chainIds: parsed.chainIds, chainLengths: chains.map((chain) => chain.length), resNums: parsed.resNums,
      ...normalizeLocalFoldChainScores({ ...confidence, ptm, iptm }, [...new Set(parsed.chainIds)], 1),
      provenance: {
        quantization: 'int5' as const, languageModelQuantization: 'int3' as const,
        confidenceHead: 'Synthyra', recyclesRun: loaded.shape.loops - 1,
        diffusionStepsScheduled: steps, diffusionStepsRun: result.steps as number,
      },
    };
  };
}
