import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it, vi } from 'vitest';
import { esmfold2Chains } from '$lib/engine/localfold-esmfold2';
import { computeCandidateIpsae, loadEsmfold2 } from '$lib/workers/localfold-esmfold2';
import { computeIpsae, sliceProteinPae } from '$lib/protein-ops/ipsae';

const vendor = (path: string) =>
  import(pathToFileURL(resolve(__dirname, '../../../static/vendor/localfold', path)).href);

const candidate = {
  id: 'input', sequence: 'ACDEFG', sourceModelId: 'sequence-input', score: null, globalScore: null,
};

describe('ESMFold2 worker contract', () => {
  it('preserves explicit and inline boundaries, and rejects invalid sequence metadata', () => {
    expect(esmfold2Chains({ ...candidate, chainLengths: [3, 3] })).toEqual(['ACD', 'EFG']);
    expect(esmfold2Chains({ ...candidate, sequence: 'ACD/EFG' })).toEqual(['ACD', 'EFG']);
    for (const invalid of [
      { sequence: 'ACDXFG' }, { sequence: 'ACD::EFG' }, { chainLengths: [2, 2] },
      { sequence: 'ACD:EFG', chainLengths: [4, 2] }, { sequence: 'A'.repeat(513) },
      { sequence: Array(63).fill('A').join(':') },
    ]) {
      expect(() => esmfold2Chains({ ...candidate, ...invalid })).toThrow();
    }
  });

  it('rejects unsupported MSA or diffusion schedules before loading weights', async () => {
    const postMessage = vi.fn();
    const scope = { postMessage, onmessage: async (_event: { data: unknown }) => {} };
    vi.stubGlobal('self', scope);
    try {
      await import('$lib/workers/localfold.worker');
      for (const params of [
        { msaMode: 'mmseqs2' }, { esmfold2Steps: 25 },
      ]) {
        postMessage.mockClear();
        await scope.onmessage({ data: { id: 1, type: 'run',
          params: { model: 'ef2-fast-300m', ...params },
          inputs: { sequences: { id: 'input', candidates: [candidate] } },
        } });
        expect(postMessage).toHaveBeenCalledTimes(1);
        expect(postMessage).toHaveBeenCalledWith(expect.objectContaining({
          type: 'error', message: expect.stringContaining('ESMFold2'),
        }));
      }
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('writes upstream ragged atom positions and learned atom/token confidence in the correct units', async () => {
    const featuresApi = await vendor('shared/esmfold2/featurise.js');
    const model = {
      loadEsmfold2Weights: vi.fn(async () => ({
        shape: { loops: 4, pairChannels: 256 }, language: { megabytes: 130 },
      })),
      languageModelRunner: () => () => {},
    };
    let missingConfidence = false;
    const predict = await loadEsmfold2('ef2-fast-300m', async (path) => {
      if (path === 'web/esmfold2-model.js') return model;
      if (path !== 'webgpu/esmfold2/fold.js') return vendor(path);
      return { foldEsmfold2: (_device: unknown, args: { sequence: string; sampler: string }) => {
        expect(args.sampler).toBe('diffusion-15');
        const f = featuresApi.featuriseForEsmfold2(args.sequence);
        const confidence = {
          plddt: new Float32Array(f.tokens).fill(0.75), plddtPerAtom: new Float32Array(f.atoms).fill(0.875),
          pae: new Float32Array(f.tokens ** 2).fill(3.25), ptm: 0.6, iptm: 0.5,
          chainPtm: { '0': 0.6, '1': 0.7 },
          chainPairIptm: { '0|1': 0.55 },
        };
        return { features: f, coordinates: f.refPos, steps: 11, confidence: missingConfidence ? null : confidence };
      } };
    }, () => {});
    expect(model.loadEsmfold2Weights).toHaveBeenCalledWith(expect.any(Function), {
      family: 'ef2-fast-300m', languageModel: true,
    });
    const fold = (chains: string[]) => predict({} as GPUDevice, chains, 15, 42, new AbortController().signal, () => {});
    const result = await fold(['ACD', 'EFG']);
    expect(result).toMatchObject({ sequence: 'ACDEFG', chainLengths: [3, 3], chainIds: ['A','A','A','B','B','B'],
      meanPlddt: 75, ptm: 0.6, iptm: 0.5, chainPtm: { A: 0.6, B: 0.7 },
      provenance: { confidenceHead: 'Synthyra', diffusionStepsRun: 11, diffusionStepsScheduled: 15, recyclesRun: 3 },
    });
    expect(result.caCoords).toHaveLength(18);
    expect(result.pae).toHaveLength(36);
    expect(result.pae[1]).toBe(3.25);
    // QA #7: multimer ESMFold2 populates ipsae & ipsaeMin directly on candidate
    const expectedIpsae = computeIpsae(result.pae, result.chainIds, 10)!;
    expect(result.ipsae).toBeCloseTo(expectedIpsae.ipsae, 6);
    expect(result.ipsaeMin).toBeCloseTo(expectedIpsae.ipsaeMin, 6);
    expect(result.ipsae!).toBeGreaterThan(0);

    const bFactors = result.pdbText.split('\n').filter(line => line.startsWith('ATOM')).map(line => +line.slice(60, 66));
    expect(new Set(bFactors)).toEqual(new Set([87.5]));
    const mono = await fold(['ACDEFG']);
    expect(mono.iptm).toBeNull();
    expect(mono.ipsae).toBeUndefined();
    expect(mono.ipsaeMin).toBeUndefined();

    // QA #7: AF3 co-fold PAE with trailing ligand tokens slices protein submatrix cleanly
    const af3Dim = 8; // 6 protein residues + 2 ligand tokens
    const af3Pae = new Float32Array(af3Dim * af3Dim).fill(2.0);
    const af3Ipsae = computeCandidateIpsae(af3Pae, [3, 3], ['A', 'A', 'A', 'B', 'B', 'B']);
    const expectedSliced = computeIpsae(sliceProteinPae(af3Pae, 6)!, ['A', 'A', 'A', 'B', 'B', 'B'], 10)!;
    expect(af3Ipsae).toEqual({ ipsae: expectedSliced.ipsae, ipsaeMin: expectedSliced.ipsaeMin });
    expect(computeCandidateIpsae(af3Pae, [6], ['A', 'A', 'A', 'A', 'A', 'A'])).toBeNull();

    missingConfidence = true;
    await expect(fold(['ACDEFG'])).rejects.toThrow(/confidence dimensions/);
  });

  it('supports all-atom holo entities (DNA/ligands) through featuriseForEsmfold2 and computes iptm/ipsae', async () => {
    const featuresApi = await vendor('shared/esmfold2/featurise.js');
    const model = {
      loadEsmfold2Weights: vi.fn(async () => ({
        shape: { loops: 4, pairChannels: 256 }, language: { megabytes: 130 },
      })),
      languageModelRunner: () => () => {},
    };
    const holoPredict = await loadEsmfold2('ef2-fast-300m', async (path) => {
      if (path === 'web/esmfold2-model.js') return model;
      if (path !== 'webgpu/esmfold2/fold.js') return vendor(path);
      return {
        foldEsmfold2: (_device: unknown, args: { entities?: Record<string, unknown>; sequence: string }) => {
          expect(args.entities).toBeDefined();
          const f = featuresApi.featuriseForEsmfold2(args.entities ?? args.sequence);
          const confidence = {
            plddt: new Float32Array(f.tokens).fill(0.8),
            plddtPerAtom: new Float32Array(f.atoms).fill(0.82),
            pae: new Float32Array(f.tokens ** 2).fill(2.5),
            ptm: 0.72,
            iptm: 0.68,
            chainPtm: { '0': 0.93, '1': 0.19 },
            chainPairIptm: { '0|1': 0.925 },
          };
          return { features: f, coordinates: f.refPos, steps: 11, confidence };
        },
      };
    }, () => {});
    const holoRes = await holoPredict(
      {} as GPUDevice,
      ['ACDEFG'],
      15,
      42,
      new AbortController().signal,
      () => {},
      { dnaSeqs: ['ACGT'] },
    );
    expect(holoRes.sequence).toBe('ACDEFG');
    expect(holoRes.plddt).toHaveLength(6);
    expect(holoRes.meanPlddt).toBeCloseTo(80, 4);
    expect(holoRes.iptm).toBeCloseTo(0.68, 4);
    expect(holoRes.chainPtm).toEqual({ A: 0.93, DNA1: 0.19 });
    expect(holoRes.chainPairIptm?.[0]?.[1]).toBeCloseTo(0.925, 4);
    expect(holoRes.ipsae).toBeGreaterThan(0);
  });
});
