import { describe, expect, it } from 'vitest';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { LOCALFOLD_MODEL_OPTIONS } from '$lib/engine/registry';

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
});
