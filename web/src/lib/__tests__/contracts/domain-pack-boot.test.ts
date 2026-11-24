import { describe, expect, it } from 'vitest';
import {
  aggregatePluginCachePolicies,
  hydrateLegacyArtifactToEnvelope,
  type DomainPack,
  type ModelPlugin,
} from '$lib/kernel';
import { PROTEIN_DOMAIN_ID, proteinBiopolymerDomainPack } from '$lib/domains/protein-biopolymer';
import { createDefaultPluginRegistry, defaultPluginRegistry } from '$lib/plugins';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';

describe('Phase 2 contract: @domain/protein-biopolymer DomainPack & live 26-plugin registry boot', () => {
  it('boots and freezes defaultPluginRegistry with all 6 protein stages, 9 port types, and all 26 adapters', () => {
    expect(defaultPluginRegistry.isFrozen()).toBe(true);
    expect(defaultPluginRegistry.domains.has(PROTEIN_DOMAIN_ID)).toBe(true);
    expect(defaultPluginRegistry.domains.listStages(PROTEIN_DOMAIN_ID).map((s) => s.id)).toEqual([
      'source',
      'condition',
      'generate',
      'inverse-fold',
      'fold',
      'score',
    ]);

    const expectedPortTypes = [
      'StructureArtifact',
      'TrajectoryFrames',
      'SequenceSet',
      'FoldedStructure',
      'DesignabilityMetrics',
      'MutationMatrix',
      'ResidueTrack',
      'SequenceEmbedding',
      'ConditionSpec',
    ];
    for (const pt of expectedPortTypes) {
      expect(defaultPluginRegistry.portTypes.has(pt), `Missing PortType ${pt}`).toBe(true);
    }

    const plugins = defaultPluginRegistry.listPlugins();
    expect(plugins).toHaveLength(27);
    expect(plugins.map((p) => p.id).sort()).toEqual(Object.keys(ADAPTER_REGISTRY).sort());

    // Short codes are bijective across all 27 plugins
    const { adapterToShort, shortToAdapter } = defaultPluginRegistry.getShortCodeMaps();
    expect(Object.keys(adapterToShort)).toHaveLength(27);
    expect(Object.keys(shortToAdapter)).toHaveLength(27);

    // LocalFold ESMFold2 worker isolation hook works without scheduler hardcoding
    const lf = defaultPluginRegistry.getPlugin('localfold-af2')!;
    expect(lf.execution.getWorkerIsolationKey?.({ model: 'ef2-fast-300m' })).toBe('ef2-fast-300m');
    expect(lf.execution.getWorkerIsolationKey?.({ model: 'af3' })).toBeUndefined();

    // Migrated createWorkerHarness() workers with cooperative abort support declare supportsCooperativeAbort + abortGraceMs >= 1500
    const coopPlugins = plugins
      .filter((p) => p.execution.supportsCooperativeAbort)
      .sort((a, b) => a.id.localeCompare(b.id));
    expect(coopPlugins.map((p) => p.id)).toEqual([
      'antifold-esmif1',
      'boltzgen',
      'dyna1',
      'esmc',
      'evodiff',
      'foldflow2',
      'frameflow',
      'genie2',
      'gnina',
      'la-proteina',
      'lasermpnn',
      'ligandmpnn',
      'proteina',
      'protpardelle',
      'rfd3',
      'rfdiffusion',
      'sid-protein',
    ]);
    for (const p of coopPlugins) {
      expect(p.execution.abortGraceMs ?? 0).toBeGreaterThanOrEqual(1500);
    }
  });

  it('aggregates all 18 CacheStorage names, 4 prefixes, and 17 OPFS weight dirs from static plugin policies', () => {
    const ns = aggregatePluginCachePolicies(defaultPluginRegistry);
    expect(ns.cacheStorageNames.size).toBe(18);
    expect(ns.cacheStoragePrefixes).toEqual(
      expect.arrayContaining(['denovo-run', 'localfold-', 'rfd4p-', 'genie3-weights-']),
    );
    expect(ns.opfsDirs).toHaveLength(17);
  });

  it('formats protein KPI badges & preflight summaries cleanly via DomainPack', () => {
    const gninaEnv = hydrateLegacyArtifactToEnvelope({
      kind: 'StructureArtifact',
      id: 'gnina-1',
      sourceModelId: 'gnina',
      length: 162,
      dockingMetrics: { cnnScore: 0.967 },
    });
    const structDesc = defaultPluginRegistry.portTypes.get('StructureArtifact')!;
    expect(structDesc.formatNodeFooterKpi?.(gninaEnv)).toEqual({
      text: 'CNNscore 0.967',
      tone: 'ok',
    });

    expect(
      proteinBiopolymerDomainPack.formatPreflightSummary?.({
        generate: 4,
        'inverse-fold': 8,
        fold: 4,
      }),
    ).toBe('4 backbones → 8 sequences → 4 refolds');
  });

  it('cleanly co-registers a second scientific DomainPack (@domain/property-prediction) with zero Layer 1 changes', () => {
    const propertyPack: DomainPack = {
      id: 'property-prediction',
      displayName: 'Molecular Property Prediction',
      stages: [{ id: 'property-pred', label: 'Property Prediction', order: 6 }],
      portTypes: [
        {
          typeId: 'MoleculeSet',
          domainId: 'property-prediction',
          label: 'Molecules (SMILES)',
          color: { bg: '#0ea5e9', fg: '#ffffff', border: '#0284c7' },
          peekPriority: 45,
          summarize: (art) => ({ kind: art.kind, id: art.id }),
          viewerId: 'molecule-table-viewer',
        },
        {
          typeId: 'PropertyTable',
          domainId: 'property-prediction',
          label: 'ADMET Properties',
          color: { bg: '#22c55e', fg: '#ffffff', border: '#16a34a' },
          peekPriority: 75,
          summarize: (art) => ({ kind: art.kind, id: art.id }),
          viewerId: 'property-table-viewer',
        },
      ],
      artifactViewers: [
        {
          id: 'molecule-table-viewer',
          domainId: 'property-prediction',
          label: 'Molecule Table',
          componentLoader: async () => ({ default: {} }),
        },
        {
          id: 'property-table-viewer',
          domainId: 'property-prediction',
          label: 'Property Table',
          componentLoader: async () => ({ default: {} }),
        },
      ],
    };

    const admetPlugin: ModelPlugin = {
      id: 'admet-chemprop',
      domainId: 'property-prediction',
      manifest: {
        id: 'admet-chemprop',
        displayName: 'ADMET Chemprop WebGPU',
        shortName: 'ADMET',
        stage: 'property-pred',
        runtime: 'wgsl-custom',
        estimatedVramMB: 96,
        approxWeightsMB: 18,
        licenseNote: 'MIT',
        description: 'Browser-native ADMET & solubility prediction',
        inputs: { molecules: { type: 'MoleculeSet', required: true } },
        outputs: { properties: { type: 'PropertyTable' } },
      },
      execution: {
        runtime: 'wgsl-custom',
        fallback: { kind: 'none', reason: 'WebGPU' },
        supportsCooperativeAbort: true,
        abortGraceMs: 1500,
        createWorker: () => ({}) as Worker,
      },
      cachePolicy: { opfsDirs: ['admet-chemprop-v1'], cacheStorageNames: ['admet-chemprop-v1'] },
      share: { shortCode: 'ac' },
      catalog: { categoryId: 'property-pred', conciseRole: 'Predict ADMET & solubility' },
      display: { formatCanvasSummary: () => 'ADMET (WebGPU)' },
      preflight: () => ({
        estimatedVramMB: 96,
        estimatedDownloadMB: 18,
        workItems: 8,
        outputCardinality: { properties: 500 },
        stageBreakdown: { 'property-pred': 500 },
      }),
    };

    const multiDomainReg = createDefaultPluginRegistry({
      extraDomainPacks: [propertyPack],
      extraPlugins: [admetPlugin],
    });

    expect(multiDomainReg.listPlugins()).toHaveLength(28);
    expect(multiDomainReg.resolveShortCode('ac')).toBe('admet-chemprop');
    expect(multiDomainReg.portTypes.has('PropertyTable')).toBe(true);
    expect(multiDomainReg.portTypes.has('StructureArtifact')).toBe(true);
  });
});
