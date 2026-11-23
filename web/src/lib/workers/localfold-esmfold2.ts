import { normalizeLocalFoldChainScores } from '$lib/engine/conditioning';
import { computeIpsae, sliceProteinPae } from '$lib/protein-ops/ipsae';
import { parsePdbBackbone } from '$lib/protein-ops/pdb';

export function computeCandidateIpsae(
  paeRaw: Float32Array | undefined,
  chainLengths: readonly number[],
  chainIds: readonly string[],
): { ipsae: number; ipsaeMin: number } | null {
  if (!paeRaw || chainLengths.length <= 1) return null;
  const proteinLen = chainIds.length;
  if (proteinLen <= 0 || paeRaw.length < proteinLen * proteinLen) return null;
  const proteinPae = sliceProteinPae(paeRaw, proteinLen);
  if (!proteinPae) return null;
  const res = computeIpsae(proteinPae, chainIds, 10);
  return res ? { ipsae: res.ipsae, ipsaeMin: res.ipsaeMin } : null;
}

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
    holo?: { dnaSeqs?: string[]; rnaSeqs?: string[]; parsedLigands?: unknown[] },
  ) {
    let stage = 'Language model';
    const sequence = chains.join('');
    const dnaSeqs = holo?.dnaSeqs ?? [];
    const rnaSeqs = holo?.rnaSeqs ?? [];
    const parsedLigands = holo?.parsedLigands ?? [];
    const hasExtraContext = dnaSeqs.length > 0 || rnaSeqs.length > 0 || parsedLigands.length > 0;
    const effectiveSteps =
      typeof model.esmfold2StepsFor === 'function'
        ? (model.esmfold2StepsFor(steps, {
            atomised: hasExtraContext,
            residues: sequence.length,
          }) as number)
        : hasExtraContext || sequence.length > 2048
          ? Math.max(steps, 64)
          : steps;
    const allPolymerSeqs = [...chains, ...dnaSeqs, ...rnaSeqs];
    const chainKinds: Array<'protein' | 'dna' | 'rna'> = [
      ...chains.map(() => 'protein' as const),
      ...dnaSeqs.map(() => 'dna' as const),
      ...rnaSeqs.map(() => 'rna' as const),
    ];
    const result = await fold.foldEsmfold2(device, {
      sequence: chains.join(':'),
      ...(hasExtraContext
        ? {
            entities: {
              sequence: allPolymerSeqs.join(':'),
              chainKinds,
              ...(parsedLigands.length > 0 ? { ligands: parsedLigands } : {}),
              conformerSeed: seed,
            },
          }
        : {}),
      shape: loaded.shape,
      weights: loaded.weights,
      confidenceWeights: loaded.confidenceWeights,
      tower: model.languageModelRunner(device, undefined, loaded, loaded.shape.pairChannels),
      languageModelMiB: loaded.language.megabytes,
      sampler: `diffusion-${effectiveSteps}`,
      contacts: false,
      seed, signal,
      onStatus: (status: string) => { stage = status; },
      onProgress: (fraction: number) => onProgress(fraction, stage),
    });
    const { features, coordinates, confidence } = result;
    const totalEntities = chains.length + dnaSeqs.length + rnaSeqs.length + parsedLigands.length;
    if (features.tokens < sequence.length || coordinates.length !== features.atoms * 3 ||
        !confidence || confidence.plddt?.length !== features.tokens ||
        confidence.plddtPerAtom?.length !== features.atoms || confidence.pae?.length !== features.tokens ** 2) {
      throw new Error(`Invalid structure or confidence dimensions from ${family}.`);
    }
    // The learned Synthyra head emits atom and token pLDDT on [0,1]; keep per-residue pLDDT on protein tokens.
    const plddt = Float32Array.from((confidence.plddt as Float32Array).subarray(0, sequence.length), (v) => v * 100);
    const denseConfidence = new Float32Array(features.tokens * features.dense);
    for (let i = 0; i < features.atoms; i++) {
      const slot = features.denseSlot[i];
      if (slot >= 0) denseConfidence[slot] = confidence.plddtPerAtom[i] * 100;
    }
    const densePositions = featuresApi.toDensePositions(features, coordinates) as Float32Array;
    if (!densePositions.every(Number.isFinite) || !denseConfidence.every(Number.isFinite) ||
        !plddt.every(Number.isFinite) || !confidence.pae.every(Number.isFinite) ||
        !Number.isFinite(confidence.ptm) || (totalEntities > 1 && !Number.isFinite(confidence.iptm))) {
      throw new Error(`Nonfinite structure or confidence from ${family}.`);
    }
    const pdbText = pdb.toPdb(features.batch, densePositions, denseConfidence) as string;
    const parsed = parsePdbBackbone(pdbText);
    if (parsed.sequence !== sequence || parsed.diagnostics.chainCount !== chains.length) {
      throw new Error(`ESMFold2 exported residues or chains inconsistent with the input.`);
    }
    const ptm: number = confidence.ptm;
    const iptm: number | null = totalEntities > 1 ? confidence.iptm : null;
    const meanPlddt = plddt.reduce((sum, v) => sum + v, 0) / plddt.length;
    const chainLengths = chains.map((chain) => chain.length);
    const pae = confidence.pae as Float32Array;
    let ipsaeChainLengths = chainLengths;
    let ipsaeChainIds = parsed.chainIds;
    if (pae.length > sequence.length * sequence.length && features.tokens > sequence.length) {
      const extraTokens = features.tokens - sequence.length;
      ipsaeChainLengths = [...chainLengths, extraTokens];
      ipsaeChainIds = [...parsed.chainIds, ...Array<string>(extraTokens).fill('LIG')];
    }
    const ipsaeRes = computeCandidateIpsae(pae, ipsaeChainLengths, ipsaeChainIds);
    // Vendor featurise.js:289 zero-bases asymId (0..totalEntities-1) in entity order: protein, DNA, RNA, ligands.
    const entityLabels: string[] = [...new Set(parsed.chainIds)];
    for (let i = 0; i < dnaSeqs.length; i++) entityLabels.push(`DNA${i + 1}`);
    for (let i = 0; i < rnaSeqs.length; i++) entityLabels.push(`RNA${i + 1}`);
    const ligCounts = new Map<string, number>();
    for (let i = 0; i < parsedLigands.length; i++) {
      const lig = parsedLigands[i] as { ccdCode?: string; name?: string; id?: string } | undefined;
      const base = String(lig?.ccdCode ?? lig?.name ?? lig?.id ?? `LIG${i + 1}`).trim() || `LIG${i + 1}`;
      const seen = (ligCounts.get(base) ?? 0) + 1;
      ligCounts.set(base, seen);
      entityLabels.push(seen > 1 ? `${base}#${seen}` : base);
    }
    return {
      sequence, pdbText, caCoords: parsed.caCoords, plddt, meanPlddt, confidence: meanPlddt,
      confidenceAvailable: true, ptm, iptm, pae,
      chainIds: parsed.chainIds, chainLengths, resNums: parsed.resNums,
      ...(ipsaeRes ? { ipsae: ipsaeRes.ipsae, ipsaeMin: ipsaeRes.ipsaeMin } : {}),
      ...normalizeLocalFoldChainScores({ ...confidence, ptm, iptm }, entityLabels, 0),
      provenance: {
        quantization: 'int5' as const, languageModelQuantization: 'int3' as const,
        confidenceHead: 'Synthyra', recyclesRun: loaded.shape.loops - 1,
        diffusionStepsScheduled: effectiveSteps, diffusionStepsRun: result.steps as number,
      },
    };
  };
}
