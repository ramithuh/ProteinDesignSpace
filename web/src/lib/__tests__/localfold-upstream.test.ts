import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { LOCALFOLD_MODEL_OPTIONS } from '$lib/engine/registry';
import { normalizeLocalFoldChainScores } from '$lib/engine/conditioning';

const VENDOR_ROOT = resolve(__dirname, '../../../static/vendor/localfold');

async function importLocalFold(relPath: string) {
  return await import(pathToFileURL(resolve(VENDOR_ROOT, relPath)).href);
}

describe('upstream localfold multi-model integration', () => {
  it('every LOCALFOLD_MODEL_OPTIONS entry maps to a valid upstream MODEL_BUNDLES manifest', async () => {
    const manifestsMod = await importLocalFold('shared/bundles/manifests/index.js');

    for (const opt of LOCALFOLD_MODEL_OPTIONS) {
      const bundle = manifestsMod.MODEL_BUNDLES[opt.id];
      expect(bundle, `Missing MODEL_BUNDLES entry for "${opt.id}"`).toBeDefined();
      expect(manifestsMod.bundleBaseUrl(opt.id)).toMatch(/^https:\/\/huggingface\.co\/sokrypton\/localfold\//);

      const manifest = await manifestsMod.loadManifest(opt.id);
      expect(manifest, `Manifest failed to load for "${opt.id}"`).toBeDefined();
      expect(Object.keys(manifest.tensors ?? {}).length).toBeGreaterThan(0);
    }
  });

  it('every AF3-lineage model resolves a valid dialect and featurises single- & multi-chain proteins', async () => {
    const [dialectMod, featuriseMod, manifestsMod] = await Promise.all([
      importLocalFold('shared/af3/dialect.js'),
      importLocalFold('shared/af3/featurise/featurise.js'),
      importLocalFold('shared/bundles/manifests/index.js'),
    ]);

    const af3Options = LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3');
    expect(af3Options.map((m) => m.id)).toContain('rosettafold3');
    expect(af3Options.map((m) => m.id)).toContain('boltz2');
    expect(af3Options.map((m) => m.id)).toContain('openbind0');

    for (const opt of af3Options) {
      const bundle = manifestsMod.MODEL_BUNDLES[opt.id];
      const dialect = dialectMod.dialectFor(bundle.model);
      expect(dialect, `Missing dialect for ${opt.id} (${bundle.model})`).toBeDefined();

      const featOpts = dialectMod.featuriserDialect(dialect);
      const monoBatch = featuriseMod.featuriseProtein('MKTAYIAKQRQISFVKSHFSRQ', featOpts);
      expect(monoBatch.tokens).toBe(22);

      const complexBatch = featuriseMod.featuriseProtein('MKTAYIAKQR:QISFVKSHFS', featOpts);
      expect(complexBatch.tokens).toBe(20);
      expect(complexBatch.asymId[0]).not.toBe(complexBatch.asymId[19]);
    }
  });

  it('uses mergeChainA3ms for AF2-Multimer homomers, normalizes chain TM scores, and supports AF3 DNA/ligand featurisation', async () => {
    const [chainsMod, dialectMod, featuriseMod, ccdMod] = await Promise.all([
      importLocalFold('shared/input/chains.js'),
      importLocalFold('shared/af3/dialect.js'),
      importLocalFold('shared/af3/featurise/featurise.js'),
      importLocalFold('shared/af3/featurise/ccd-component.js'),
    ]);

    // Finding #18: mergeChainA3ms keeps identical homomer chains dense vs mergeUnpairedChainA3ms block-diagonal
    const homomerA3m = '>q\nMKTAYIAKQR\n>h1\nMKTAYIAKQK\n';
    const multimerMerged = chainsMod.mergeChainA3ms([homomerA3m, homomerA3m]);
    const monomerMerged = chainsMod.mergeUnpairedChainA3ms([homomerA3m, homomerA3m]);
    // Dense homomer has 2 rows (query + paired h1); block-diagonal has 5 rows (query + 2 gap-padded rows per copy)
    expect(multimerMerged.match(/^>/gm)?.length).toBe(2);
    expect(monomerMerged.match(/^>/gm)?.length).toBeGreaterThan(2);
    expect(monomerMerged).toContain('----------');
    expect(multimerMerged).not.toContain('----------');

    // Finding #8: normalizeLocalFoldChainScores preserves chainPtm, chainIptm, chainPairIptm matrix, multimerScore
    const af2Norm = normalizeLocalFoldChainScores(
      {
        ptm: 0.8,
        iptm: 0.7,
        multimerScore: 0.72,
        chainPtm: { '0': 0.85, '1': 0.75 },
        chainIptm: { '0': 0.7, '1': 0.7 },
        chainPairIptm: { '0|1': 0.7 },
      },
      ['A', 'B'],
      0,
    );
    expect(af2Norm.chainPtm).toEqual({ A: 0.85, B: 0.75 });
    expect(af2Norm.chainIptm).toEqual({ A: 0.7, B: 0.7 });
    expect(af2Norm.chainPairIptm).toEqual([
      [0.85, 0.7],
      [0.7, 0.75],
    ]);
    expect(af2Norm.multimerScore).toBeCloseTo(0.72, 5);

    const af3Norm = normalizeLocalFoldChainScores(
      {
        ptm: 0.82,
        iptm: 0.76,
        chainPtm: { '1': 0.88, '2': 0.79 },
        chainIptm: { '1': 0.76, '2': 0.76 },
        chainPairIptm: { '1|2': 0.76 },
      },
      ['A', 'B'],
      1,
    );
    expect(af3Norm.chainPtm).toEqual({ A: 0.88, B: 0.79 });
    expect(af3Norm.chainPairIptm).toEqual([
      [0.88, 0.76],
      [0.76, 0.79],
    ]);
    expect(af3Norm.multimerScore).toBeCloseTo(0.8 * 0.76 + 0.2 * 0.82, 5);

    // Finding #17: AF3 featuriser handles protein + DNA + CCD ligand cleanly
    const znCcd = ccdMod.parseCcdComponent([
      'data_ZN',
      '_chem_comp.id ZN',
      '_chem_comp.type non-polymer',
      '_chem_comp_atom.comp_id ZN',
      '_chem_comp_atom.atom_id ZN',
      '_chem_comp_atom.type_symbol ZN',
      '_chem_comp_atom.charge 2',
      '_chem_comp_atom.pdbx_model_Cartn_x_ideal 0.0',
      '_chem_comp_atom.pdbx_model_Cartn_y_ideal 0.0',
      '_chem_comp_atom.pdbx_model_Cartn_z_ideal 0.0',
      '#',
    ].join('\n'));
    const rf3FeatOpts = dialectMod.featuriserDialect(dialectMod.dialectFor('rosettafold3'));
    const holoDnaBatch = featuriseMod.featuriseProtein('MKTAYIAKQR:ACGTACGT', {
      ...rf3FeatOpts,
      chainKinds: ['protein', 'dna'],
      ligands: [znCcd],
    });
    // 10 protein + 8 DNA + 1 ZN ligand = 19 tokens
    expect(holoDnaBatch.tokens).toBe(19);

    // Verify localfold.worker.ts wires mergeChainA3ms, iptmIsMonomerApprox, recycles clamp 0..4,
    // AF3 ligand/DNA passthrough, shared selectFoldWorkItems (Issue #4), and selectFoldRepresentative (Issue #5)
    const workerSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/localfold.worker.ts'), 'utf8');
    expect(workerSrc).toContain('mergeChainA3ms');
    expect(workerSrc).toContain('iptmIsMonomerApprox');
    expect(workerSrc).toContain('Math.min(4,');
    expect(workerSrc).toContain('ligandCcdCodes');
    expect(workerSrc).toContain('selectFoldWorkItems');
    expect(workerSrc).toContain('selectFoldRepresentative');
    expect(workerSrc).toContain('representativeSelection');
    expect(workerSrc).toContain('excludedCandidates');
  });
});
