import { describe, expect, it } from 'vitest';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

function collectFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      out.push(...collectFiles(full));
    } else {
      out.push(full);
    }
  }
  return out;
}

function extractWorkerRunParamKeys(workerRelPath: string): string[] {
  const src = readFileSync(resolve(process.cwd(), workerRelPath), 'utf8');
  const runIdx = src.indexOf("msg.type === 'run'");
  expect(runIdx, `Expected msg.type === 'run' handler in ${workerRelPath}`).toBeGreaterThan(-1);
  const runSlice = src.slice(runIdx);
  const keys = new Set<string>();
  for (const m of runSlice.matchAll(/\b(?:p|params|msg\.params)\??\.([A-Za-z_][A-Za-z0-9_]*)\b/g)) {
    keys.add(m[1]);
  }
  return [...keys].sort();
}

const WAVE7A_SPECS = [
  {
    name: 'ThermoMPNN',
    vendorDir: 'static/vendor/thermompnn',
    workerFile: 'src/lib/workers/thermompnn.worker.ts',
    factoryExport: 'createThermoMPNNEngine',
    hfBaseUrl: 'https://huggingface.co/ramithuh/thermompnn-webgpu/resolve/main/default',
    expectedParamKeys: ['bfactorAggregate', 'chain', 'precision'],
  },
  {
    name: 'LigandMPNN',
    vendorDir: 'static/vendor/ligandmpnn',
    workerFile: 'src/lib/workers/ligandmpnn.worker.ts',
    factoryExport: 'createLigandMpnnEngine',
    hfBaseUrl: 'https://huggingface.co/ramithuh/ligandmpnn-webgpu/resolve/main',
    expectedParamKeys: [
      'checkpoint',
      'fixedPositions',
      'numSequences',
      'omitAminoAcids',
      'precision',
      'seed',
      'temperature',
      'unlockInherited',
      'useLigandContext',
    ],
  },
  {
    name: 'LASErMPNN',
    vendorDir: 'static/vendor/lasermpnn',
    workerFile: 'src/lib/workers/lasermpnn.worker.ts',
    factoryExport: 'createLaserMpnnEngine',
    hfBaseUrl: 'https://huggingface.co/ramithuh/lasermpnn-webgpu/resolve/main',
    expectedParamKeys: [
      'fixedPositions',
      'numSequences',
      'omitAminoAcids',
      'precision',
      'seed',
      'temperature',
      'unlockInherited',
    ],
  },
  {
    name: 'AntiFold / ESM-IF1',
    vendorDir: 'static/vendor/antifold-esmif1',
    workerFile: 'src/lib/workers/antifold.worker.ts',
    factoryExport: 'createGVPTransformerEngine',
    hfBaseUrl: 'https://huggingface.co/ramithuh/antifold-esmif1-webgpu/resolve/main',
    expectedParamKeys: [
      'fixedPositions',
      'numSequences',
      'precision',
      'regions',
      'samplingMode',
      'seed',
      'targetChain',
      'temperature',
      'unlockInherited',
      'variant',
    ],
  },
  {
    name: 'ProtPardelle-1c',
    vendorDir: 'static/vendor/protpardelle',
    workerFile: 'src/lib/workers/protpardelle.worker.ts',
    factoryExport: 'createProtpardelleEngine',
    hfBaseUrl: 'https://huggingface.co/ramithuh/protpardelle-webgpu/resolve/main',
    expectedParamKeys: [
      'checkpoint',
      'length',
      'motifSelection',
      'numDesigns',
      'numSteps',
      'precision',
      'rewindSteps',
      'seed',
      'sidechainMode',
      'stepScale',
      'taskMode',
      'temperature',
    ],
  },
  {
    name: 'ESM-C 300M',
    vendorDir: 'static/vendor/esmc',
    workerFile: 'src/lib/workers/esmc.worker.ts',
    factoryExport: 'createEsmcEngine',
    hfBaseUrl: 'https://huggingface.co/ramithuh/esmc-webgpu/resolve/main/esmc-300m',
    expectedParamKeys: ['maskPositions', 'maxSequences', 'mode', 'precision'],
  },
] as const;

describe('Wave-7A WebGPU vendored engines & workers', () => {
  it('bundles all 4 vendored zero-dep WebGPU engines cleanly with no bundled weights', async () => {
    const legacyBrand = ['xy', 'mol'].join('');
    for (const spec of WAVE7A_SPECS) {
      const dir = resolve(process.cwd(), spec.vendorDir);
      expect(existsSync(dir), `${spec.name} vendor dir missing`).toBe(true);
      expect(existsSync(resolve(dir, 'weights')), `${spec.name} must not bundle weights in static/`).toBe(false);

      const entryPath = resolve(dir, 'webgpu/index.js');
      expect(existsSync(entryPath), `${spec.name} webgpu/index.js missing`).toBe(true);

      const mod = await import(pathToFileURL(entryPath).href);
      expect(
        typeof mod[spec.factoryExport],
        `${spec.name} must export ${spec.factoryExport}`,
      ).toBe('function');

      for (const file of collectFiles(dir)) {
        const text = readFileSync(file, 'utf8');
        expect(text.toLowerCase(), `Unexpected legacy brand reference in ${file}`).not.toContain(legacyBrand);
      }
    }
  });

  it('wires all 4 workers to live Hugging Face URLs, exact factory calls, dispose handlers, and exact param keys', () => {
    const legacyBrand = ['xy', 'mol'].join('');
    for (const spec of WAVE7A_SPECS) {
      const workerPath = resolve(process.cwd(), spec.workerFile);
      expect(existsSync(workerPath), `${spec.workerFile} missing`).toBe(true);
      const src = readFileSync(workerPath, 'utf8');

      expect(src).toContain(spec.hfBaseUrl);
      expect(src).toContain(spec.factoryExport);
      expect(src).toContain("msg.type === 'dispose'");
      expect(src).toContain("type: 'disposed'");
      expect(src).toContain("msg.type === 'run'");
      expect(src).toContain("type: 'result'");
      expect(src).toContain("type: 'error'");
      expect(src).toContain('/* @vite-ignore */');
      expect(src).toContain('engine?.destroy()');
      expect(src.toLowerCase()).not.toContain(legacyBrand);

      // Exact bidirectional param key match (aligns with param-schema.test.ts regex)
      expect(extractWorkerRunParamKeys(spec.workerFile)).toEqual([...spec.expectedParamKeys]);
    }
  });

  it('validates worker-specific scientific contracts (B-factor painting, holo context, all-atom packing, deterministic uniforms)', () => {
    const thermoSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/thermompnn.worker.ts'), 'utf8');
    expect(thermoSrc).toContain('paintStructureBfactors');
    expect(thermoSrc).toContain("bfactorSemantics: 'ddg-stability'");
    expect(thermoSrc).toContain("kind: 'MutationMatrix'");
    expect(thermoSrc).toContain("metric: 'ddg-kcal-mol'");
    expect(thermoSrc).toContain("direction: 'lower-is-better'");
    expect(thermoSrc).toContain('makeCachedFetch');
    expect(thermoSrc).toContain('collectOwnedTransferables');

    const ligandSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/ligandmpnn.worker.ts'), 'utf8');
    expect(ligandSrc).toContain('useAtomContext: useLigandContext');
    expect(ligandSrc).toContain('structure.collection');
    expect(ligandSrc).toContain('parseFixedPositionsAndChains');
    expect(ligandSrc).toContain('computeChainLengths');
    expect(ligandSrc).toContain('cache: true');

    const laserSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/lasermpnn.worker.ts'), 'utf8');
    expect(laserSrc).toContain("atomGranularity: 'all-atom'");
    expect(laserSrc).toContain('parseStructureText');
    expect(laserSrc).toContain('computeGeometryStats');
    expect(laserSrc).toContain('hasExplicitHydrogens: true');
    expect(laserSrc).toContain('makeCachedFetch');
    expect(laserSrc).toContain('collectOwnedTransferables');

    const antiSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/antifold.worker.ts'), 'utf8');
    expect(antiSrc).toContain('mulberry32');
    expect(antiSrc).toContain('sampleIndependent');
    expect(antiSrc).toContain('activeEngine.sample(');
    expect(antiSrc).toContain('makeCachedFetch');
    expect(antiSrc).toContain('parseFixedPositionsAndChains');
    expect(antiSrc).toContain('computeChainLengths');
  });

  it('matches vendored pure-JS parser and constraint contracts across all 4 engines on a holo PDB fixture', async () => {
    const miniHoloPdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  ',
      'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  ',
      'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C  ',
      'ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O  ',
      'ATOM      5  N   GLY A   2       3.320   1.510   0.000  1.00 20.00           N  ',
      'ATOM      6  CA  GLY A   2       3.960   2.810   0.000  1.00 20.00           C  ',
      'ATOM      7  C   GLY A   2       5.470   2.700   0.000  1.00 20.00           C  ',
      'ATOM      8  O   GLY A   2       6.050   1.620   0.000  1.00 20.00           O  ',
      'HETATM    9 ZN   ZN  Z 101       3.000   4.500   1.500  1.00 10.00          ZN  ',
      'END',
      '',
    ].join('\n');

    // 1. ThermoMPNN parser & canonical 20-AA alphabet
    const thermoStruct = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/thermompnn/shared/structure.js')).href
    );
    const tParsed = thermoStruct.parseStructure(miniHoloPdb);
    expect(tParsed.length).toBe(2);
    expect(tParsed.sequence).toBe('AG');
    expect(thermoStruct.ALPHABET.slice(0, 20)).toBe('ACDEFGHIKLMNPQRSTVWY');

    // 2. LigandMPNN holo parser + featurize with useAtomContext & fixedPositions token format ("A1")
    const ligParser = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/ligandmpnn/shared/ligandmpnn/parser.js')).href
    );
    const ligFeat = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/ligandmpnn/shared/ligandmpnn/features.js')).href
    );
    const lParsed = ligParser.parseStructure(miniHoloPdb);
    expect(lParsed.L).toBe(2);
    expect(lParsed.context.length).toBe(1); // ZN ion preserved
    const lFeatHolo = ligFeat.featurize(lParsed, {
      fixedPositions: ['A1'],
      omitAAs: 'C',
      useAtomContext: true,
    });
    expect(Array.from(lFeatHolo.chain_mask)).toEqual([0, 1]);
    expect(lFeatHolo.Y_m[0]).toBe(1);
    const lFeatApo = ligFeat.featurize(lParsed, { useAtomContext: false });
    expect(lFeatApo.Y_m[0]).toBe(0);

    // 3. LASErMPNN parser ID format ("A:1", "A:2") + ligand preservation
    const laserStruct = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/lasermpnn/shared/lasermpnn/structure.js')).href
    );
    const rParsed = laserStruct.parseStructure(miniHoloPdb);
    expect(rParsed.residues.map((r: { id: string }) => r.id)).toEqual(['A:1', 'A:2']);
    expect(rParsed.ligands.length).toBe(1);

    // 4. AntiFold / ESM-IF1 parser + independentPlan with heavyChain, regions, uniforms & fixed
    const antiShared = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/antifold-esmif1/shared/index.js')).href
    );
    const antiSampling = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/antifold-esmif1/shared/sampling.js')).href
    );
    const aParsed = antiShared.parseStructure(miniHoloPdb);
    const plan = antiSampling.independentPlan(aParsed, {
      targetChain: 'A',
      heavyChain: 'A',
      regions: ['CDR1', 'CDR2', 'CDR3'],
      temperature: 0.2,
      uniforms: new Float32Array([0.25, 0.75]),
      fixed: { 0: antiShared.prepareInput(aParsed, { targetChain: 'A' }).tokens[0] },
    });
    expect(plan.p.L).toBe(2);
    expect(plan.selected[0]).toBe(0);
  });
});
