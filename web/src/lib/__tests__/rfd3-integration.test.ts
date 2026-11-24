import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { defaultPluginRegistry } from '$lib/plugins';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { filterCatalogCategories, CATALOG_CATEGORIES, CATALOG_ITEMS } from '$lib/engine/catalog';
import { getModelCardForAdapter, MODEL_CARDS } from '$lib/engine/model-cards';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import {
  decodeHashToGraph,
  encodeGraphToHash,
  graphToYamlRecipe,
  parseRecipeToGraph,
} from '$lib/engine/share';
import { estimateRunRequirements } from '$lib/engine/run-preflight';
import { validateBioCapabilities, validateRuntimeNodeInputs } from '$lib/engine/bio-capabilities';
import { validateNodeParams } from '$lib/engine/param-validation';
import { compileNodeInputPreview } from '$lib/engine/input-preview';
import { computeUpstreamLockSummary, formatNodeParamSummary } from '$lib/ui/node-display-helpers';
import type { WorkflowGraph, WorkflowNode } from '$lib/engine/types';

const MINI_HOLO_PDB = [
  'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  ',
  'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  ',
  'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C  ',
  'ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O  ',
  'ATOM      5  CB  ALA A   1       1.986  -0.773  -1.209  1.00 20.00           C  ',
  'ATOM      6  N   GLY A   2       3.320   1.510   0.000  1.00 20.00           N  ',
  'ATOM      7  CA  GLY A   2       3.960   2.810   0.000  1.00 20.00           C  ',
  'ATOM      8  C   GLY A   2       5.470   2.700   0.000  1.00 20.00           C  ',
  'ATOM      9  O   GLY A   2       6.050   1.620   0.000  1.00 20.00           O  ',
  'ATOM     10  N   SER A   3       6.100   3.870   0.000  1.00 20.00           N  ',
  'ATOM     11  CA  SER A   3       7.550   3.960   0.000  1.00 20.00           C  ',
  'ATOM     12  C   SER A   3       8.100   5.380   0.000  1.00 20.00           C  ',
  'ATOM     13  O   SER A   3       7.340   6.350   0.000  1.00 20.00           O  ',
  'ATOM     14  CB  SER A   3       8.080   3.190  -1.210  1.00 20.00           C  ',
  'ATOM     15  OG  SER A   3       9.480   3.280  -1.310  1.00 20.00           O  ',
  'HETATM   16 ZN   ZN  Z 101       5.000   4.500   2.500  1.00 10.00          ZN  ',
  'END',
  '',
].join('\n');

describe('RFdiffusion3 (rfd3) end-to-end Layer 1–3 plugin integration', () => {
  it('registers rfd3Plugin natively in defaultPluginRegistry with expected capabilities & cache policy', () => {
    const plugin = defaultPluginRegistry.getPlugin('rfd3');
    expect(plugin).toBeDefined();
    expect(plugin!.id).toBe('rfd3');
    expect(plugin!.execution.runtime).toBe('wgsl-custom');
    expect(plugin!.execution.supportsCooperativeAbort).toBe(true);
    expect(plugin!.execution.vramLeaseMode).toBe('exclusive');
    expect(plugin!.cachePolicy).toEqual({
      cacheStorageNames: ['rfd3-v1'],
      opfsDirs: ['rfd3-v1'],
    });
    expect(plugin!.share.shortCode).toBe('r3');
    expect(defaultPluginRegistry.resolveShortCode('r3')).toBe('rfd3');

    const manifest = ADAPTER_REGISTRY.rfd3;
    expect(manifest).toBeDefined();
    expect(manifest.stage).toBe('generate');
    expect(manifest.approxWeightsMB).toBe(337);
    expect(manifest.precisionWeightsMB).toEqual({ fp16: 337, int8: 170 });
    expect(manifest.estimatedVramMB).toBe(1400);
    expect(manifest.outputs.structure.type).toBe('StructureArtifact');
    expect(manifest.outputs.sequences.type).toBe('SequenceSet');
  });

  it('ships clean vendored runtime under static/vendor/rfd3 without dev/weight bloat', async () => {
    const vendorDir = resolve(process.cwd(), 'static/vendor/rfd3');
    for (const rel of [
      'webgpu/index.js',
      'shared/input.js',
      'cpu/index.js',
      'fixtures/architecture.json',
      'LICENSE',
      'LICENSE-APACHE-2.0',
      'NOTICE',
      'README_WEBGPU.md',
    ]) {
      expect(existsSync(resolve(vendorDir, rel)), `Missing vendor file ${rel}`).toBe(true);
    }
    for (const forbidden of ['demo', 'tools/serve.mjs', 'LICENSE-PYTORCH', 'weights', '.cache']) {
      expect(existsSync(resolve(vendorDir, forbidden)), `Forbidden vendor file ${forbidden}`).toBe(false);
    }

    const mod = await import(pathToFileURL(resolve(vendorDir, 'webgpu/index.js')).href);
    expect(typeof mod.createRfd3Engine).toBe('function');
    expect(typeof mod.createRFD3Engine).toBe('function');

    const arch = JSON.parse(readFileSync(resolve(vendorDir, 'fixtures/architecture.json'), 'utf8'));
    expect(arch.checkpoint_sha256).toBeTruthy();
    expect(arch.config?.c_z).toBeGreaterThan(0);
  });

  it('publishes complete rfd3-webgpu model card with pinned HF revision and FP16 default', () => {
    const card = getModelCardForAdapter('rfd3', defaultParamsForAdapter('rfd3'));
    expect(card).not.toBeNull();
    expect(card!.id).toBe('rfd3-webgpu');
    expect(MODEL_CARDS['rfd3-webgpu']).toBeDefined();
    expect(card!.weightsRepo).toBe('https://huggingface.co/ramithuh/rfd3-webgpu');
    expect(card!.weightsRevision).toBe('d318c4b6acb601f56b85b1a3e8fb565682cd7360');
    expect(card!.activePrecisionId).toBe('fp16');

    const fp16 = card!.precisions.find((p) => p.id === 'fp16');
    const int8 = card!.precisions.find((p) => p.id === 'int8');
    expect(fp16?.approxSizeMB).toBe(337);
    expect(int8?.approxSizeMB).toBe(170);
    expect(int8?.caveat).toMatch(/experimental/i);
  });

  it('computes accurate preflight download & VRAM estimates for FP16 default, INT8 compact, and batches', () => {
    const fp16Graph: WorkflowGraph = {
      id: 'rfd3-fp16',
      name: 'RFD3 FP16',
      nodes: [
        {
          id: 'g1',
          adapterId: 'rfd3',
          label: 'RFdiffusion3',
          params: { ...defaultParamsForAdapter('rfd3'), precision: 'fp16', numDesigns: 4 },
        },
      ],
      edges: [],
    };
    const fp16Pre = estimateRunRequirements(fp16Graph);
    expect(fp16Pre.totalDownloadMB).toBe(337);
    expect(fp16Pre.peakVramMB).toBe(1400);
    expect(fp16Pre.plannedBackbones).toBe(4);
    expect(fp16Pre.plannedSequences).toBe(4);
    expect(fp16Pre.usesWebGpu).toBe(true);

    const int8Graph: WorkflowGraph = {
      id: 'rfd3-int8',
      name: 'RFD3 INT8',
      nodes: [
        {
          id: 'g1',
          adapterId: 'rfd3',
          label: 'RFdiffusion3',
          params: { ...defaultParamsForAdapter('rfd3'), precision: 'int8', numDesigns: 1 },
        },
      ],
      edges: [],
    };
    const int8Pre = estimateRunRequirements(int8Graph);
    expect(int8Pre.totalDownloadMB).toBe(170);
    expect(int8Pre.peakVramMB).toBe(1100);
  });

  it('is discoverable in Structure Generation catalog by all key scientific queries', () => {
    const item = CATALOG_ITEMS.find((i) => i.id === 'rfd3');
    expect(item).toBeDefined();
    expect(item!.stage).toBe('generate');
    expect(item!.subGroups[0]?.items.length).toBeGreaterThanOrEqual(5);

    const genCat = CATALOG_CATEGORIES.find((c) => c.id === 'generation');
    expect(genCat?.subGroups[0]?.items.some((i) => i.key === 'rfd3')).toBe(true);

    for (const q of [
      'rfd3',
      'rfdiffusion3',
      'all-atom',
      'enzyme',
      'ligand',
      'motif',
      'binder',
      'cyclic',
      'symmetry',
      'unindexed',
      'tip-atom',
      'partial',
      'baker',
    ]) {
      const results = filterCatalogCategories(q);
      const found = results.some((r) => r.matchingSubItems.some((it) => it.key === 'rfd3'));
      expect(found, `Expected catalog search "${q}" to match rfd3`).toBe(true);
    }
  });

  it('round-trips rfd3 workflows cleanly through #g= share hashes and denovo: v1 YAML', async () => {
    const graph: WorkflowGraph = {
      id: 'rfd3-share-test',
      name: 'RFD3 Enzyme Motif Workflow',
      nodes: [
        {
          id: 'src',
          adapterId: 'pdb-input',
          label: 'Template',
          params: { ...defaultParamsForAdapter('pdb-input'), mode: 'demo-bundle' },
        },
        {
          id: 'gen',
          adapterId: 'rfd3',
          label: 'RFdiffusion3 (Motif)',
          params: {
            ...defaultParamsForAdapter('rfd3'),
            mode: 'motif',
            length: 72,
            steps: 40,
            unindex: 'A1,A3',
            fixedAtoms: 'A3:CB,OG',
            ligand: 'ZN',
            precision: 'fp16',
          },
        },
      ],
      edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'gen', toPort: 'structure' }],
    };

    const { hash, mode } = await encodeGraphToHash(graph);
    expect(mode).toBe('compressed-dag');
    expect(hash.startsWith('#g=')).toBe(true);
    const decoded = await decodeHashToGraph(hash);
    expect(decoded).not.toBeNull();
    const genNode = decoded!.nodes.find((n) => n.id === 'gen')!;
    expect(genNode.adapterId).toBe('rfd3');
    expect(genNode.params.mode).toBe('motif');
    expect(genNode.params.unindex).toBe('A1,A3');
    expect(genNode.params.fixedAtoms).toBe('A3:CB,OG');
    expect(genNode.params.ligand).toBe('ZN');

    const yaml = graphToYamlRecipe(graph);
    const fromYaml = parseRecipeToGraph(yaml);
    const yamlGen = fromYaml.nodes.find((n) => n.id === 'gen')!;
    expect(yamlGen.adapterId).toBe('rfd3');
    expect(yamlGen.params.length).toBe(72);
    expect(yamlGen.params.steps).toBe(40);
  });

  it('enforces BioGuard & param validation rules across unconditional, motif, binder, ligand, partial, cyclic, and symmetry modes', () => {
    // 1. Valid defaults pass
    const uncondNode: WorkflowNode = {
      id: 'r1',
      adapterId: 'rfd3',
      params: defaultParamsForAdapter('rfd3'),
    };
    expect(validateNodeParams(uncondNode, { connectedInputs: [] })).toEqual([]);

    // 2. Symmetry cross-validation guards
    expect(
      validateNodeParams(
        {
          id: 'r_sym_cyc',
          adapterId: 'rfd3',
          params: { ...defaultParamsForAdapter('rfd3'), symmetry: 'C3', cyclic: true },
        },
        { connectedInputs: [] },
      ).join(' '),
    ).toMatch(/cannot combine cyclic/i);

    expect(
      validateNodeParams(
        {
          id: 'r_sym_cfg',
          adapterId: 'rfd3',
          params: { ...defaultParamsForAdapter('rfd3'), symmetry: 'C2', cfgScale: 2.0 },
        },
        { connectedInputs: [] },
      ).join(' '),
    ).toMatch(/cfgScale = 1\.0/i);

    expect(
      validateNodeParams(
        {
          id: 'r_sym_g0',
          adapterId: 'rfd3',
          params: { ...defaultParamsForAdapter('rfd3'), symmetry: 'C2', gamma0: 0.4 },
        },
        { connectedInputs: [] },
      ).join(' '),
    ).toMatch(/gamma0 > 0\.5/i);

    // 3. Conditioned modes require connected structure input
    for (const mode of ['motif', 'binder', 'partial'] as const) {
      const n: WorkflowNode = {
        id: `r_${mode}`,
        adapterId: 'rfd3',
        params: { ...defaultParamsForAdapter('rfd3'), mode },
      };
      expect(validateNodeParams(n, { connectedInputs: [] }).join(' ')).toMatch(/structure/);
      expect(validateNodeParams(n, { connectedInputs: ['structure'] })).toEqual([]);
      expect(() => validateRuntimeNodeInputs(n, {})).toThrow(/requires a wired structure/i);
    }

    // 4. BioGuard: unused template warning in unconditional mode
    const unusedTplGraph: WorkflowGraph = {
      id: 'rfd3-unused-tpl',
      name: 'Unused Template',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'gen', adapterId: 'rfd3', params: { ...defaultParamsForAdapter('rfd3'), mode: 'unconditional' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'gen', toPort: 'structure' }],
    };
    const unusedIssues = validateBioCapabilities(unusedTplGraph);
    expect(unusedIssues.some((i) => i.code === 'rfd3-unused-template')).toBe(true);

    // 5. BioGuard: ligand/enzyme context flows through RFD3 motif mode -> downstream ProteinMPNN gets non-protein warning, LigandMPNN gets zero error
    const enzymeGraph: WorkflowGraph = {
      id: 'rfd3-enzyme',
      name: 'RFD3 Enzyme',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: MINI_HOLO_PDB } },
        {
          id: 'gen',
          adapterId: 'rfd3',
          params: { ...defaultParamsForAdapter('rfd3'), mode: 'motif', ligand: 'ZN', unindex: 'A3' },
        },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: defaultParamsForAdapter('web-protein-mpnn') },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'gen', toPort: 'structure' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
      ],
    };
    const enzymeIssues = validateBioCapabilities(enzymeGraph);
    expect(enzymeIssues.some((i) => i.code === 'mpnn-ca-only-backbone')).toBe(false);
    expect(enzymeIssues.some((i) => i.code === 'mpnn-non-protein-stripped')).toBe(true);

    // 6. Upstream lock summary for binder and symmetry modes
    const binderGraph: WorkflowGraph = {
      id: 'rfd3-binder',
      name: 'RFD3 Binder',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        {
          id: 'gen',
          adapterId: 'rfd3',
          params: { ...defaultParamsForAdapter('rfd3'), mode: 'binder', length: 64, hotspots: 'A10,A14' },
        },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: defaultParamsForAdapter('web-protein-mpnn') },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'gen', toPort: 'structure' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
      ],
    };
    const lock = computeUpstreamLockSummary(binderGraph, 'mpnn', {});
    expect(lock?.lockedChains).toEqual(['B']);
    expect(lock?.designingChains).toEqual(['A']);

    // 7. Canvas summary formatting
    expect(formatNodeParamSummary('rfd3', { mode: 'unconditional', length: 80, steps: 50, precision: 'fp16' })).toContain('80 aa');
    expect(formatNodeParamSummary('rfd3', { mode: 'unconditional', symmetry: 'C3', length: 48, steps: 50 })).toContain('C3');
    expect(formatNodeParamSummary('rfd3', { mode: 'unconditional', cyclic: true, length: 16, steps: 50 })).toContain('Cyclic');
    expect(formatNodeParamSummary('rfd3', { mode: 'binder', length: 64, hotspots: 'A10,A12', steps: 50 })).toContain('2 hot');
    expect(formatNodeParamSummary('rfd3', { mode: 'partial', partialT: 5, steps: 50 })).toContain('Partial');
  });

  it('compiles clean input previews for unconditional and conditioned modes', () => {
    const uncondGraph: WorkflowGraph = {
      id: 'preview-uncond',
      name: 'Preview Uncond',
      nodes: [{ id: 'gen', adapterId: 'rfd3', params: defaultParamsForAdapter('rfd3') }],
      edges: [],
    };
    const uncondPreview = compileNodeInputPreview(uncondGraph, 'gen', {});
    expect(uncondPreview.availability).toBe('ready');

    const motifGraph: WorkflowGraph = {
      id: 'preview-motif',
      name: 'Preview Motif',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'gen', adapterId: 'rfd3', params: { ...defaultParamsForAdapter('rfd3'), mode: 'motif' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'gen', toPort: 'structure' }],
    };
    const motifPreview = compileNodeInputPreview(motifGraph, 'gen', {});
    expect(motifPreview.availability).toBe('ready');
  });

  it('executes vendored shared/input.js buildInput across all RFD3 conditioning modes without errors', async () => {
    const inputMod = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/rfd3/shared/input.js')).href
    );

    const uncond = inputMod.buildInput({ mode: 'unconditional', length: 32 });
    expect(uncond.tokens.length).toBe(32);

    const cyclic = inputMod.buildInput({ mode: 'unconditional', length: 12, cyclic: true });
    expect(cyclic.features.cyclic_asym_ids).toBeInstanceOf(Float32Array);

    const sym = inputMod.buildInput({ mode: 'unconditional', length: 16, symmetry: { id: 'C2' } });
    expect(sym.tokens.length).toBe(32);

    const motif = inputMod.buildInput({
      mode: 'motif',
      structure: MINI_HOLO_PDB,
      contig: '10,A1-3,10',
      fixedAtoms: { A3: 'CB,OG' },
      ligand: 'ZN',
    });
    expect(motif.tokens.some((t: { ligand?: boolean }) => t.ligand)).toBe(true);

    const unidx = inputMod.buildInput({
      mode: 'motif',
      structure: MINI_HOLO_PDB,
      length: 24,
      unindex: 'A3',
      fixedAtoms: { A3: 'CB,OG' },
    });
    expect(unidx.tokens.some((t: { unindexed?: boolean }) => t.unindexed)).toBe(true);

    const binder = inputMod.buildInput({
      mode: 'binder',
      structure: MINI_HOLO_PDB,
      length: 20,
      hotspots: 'A1,A3',
    });
    expect(binder.tokens.length).toBe(23);

    const partial = inputMod.buildInput({
      mode: 'conditioned',
      structure: MINI_HOLO_PDB,
      contig: 'A1-3',
      fixedAtoms: false,
      partialT: 5,
    });
    expect(partial.features.partial_t).toBeInstanceOf(Float32Array);
  });

  it('allowlists rfd3 on the self-hosted WebGPU server (/api/run-node)', () => {
    const serverPath = resolve(process.cwd(), '../server/selfhost-server.mjs');
    const req = createRequire(import.meta.url);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const serverMod = req(serverPath) as any;
    expect(serverMod.ALLOWED_GPU_ADAPTER_IDS.has('rfd3')).toBe(true);
  });
});
