import { describe, expect, it } from 'vitest';
import {
  buildPlannerCatalogIndex,
  joinKeyedCollections,
  setActivePluginRegistry,
  type ArtifactEnvelope,
  type DomainPack,
  type ModelPlugin,
} from '$lib/kernel';
import { createDefaultPluginRegistry, defaultPluginRegistry } from '$lib/plugins';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import {
  decodeHashToGraph,
  encodeGraphToHash,
  graphToYamlRecipe,
  parseRecipeToGraph,
} from '$lib/engine/share';
import { estimateRunRequirements } from '$lib/engine/run-preflight';
import { getNodePeekSlots, pickDefaultPeekSlot } from '$lib/engine/peek';
import type { CanonicalArtifact, WorkflowGraph } from '$lib/engine/types';

const MOLECULAR_PROPS_DOMAIN_ID = 'molecular-properties';

interface PropertyRow {
  compoundId: string;
  smiles: string;
  logS: number;
  logP: number;
}

const molecularPropertiesDomainPack: DomainPack = {
  id: MOLECULAR_PROPS_DOMAIN_ID,
  displayName: 'Molecular Property Prediction',
  stages: [
    { id: 'mol-source', label: 'Molecule Input', order: 0 },
    { id: 'mol-predict', label: 'Property Prediction', order: 1 },
  ],
  portTypes: [
    {
      typeId: 'SmilesBatch',
      domainId: MOLECULAR_PROPS_DOMAIN_ID,
      label: 'SMILES Batch',
      color: { bg: '#0ea5e9', fg: '#ffffff', border: '#0284c7' },
      peekPriority: 40,
      viewerId: 'mol-smiles-viewer',
      summarize: (art) => {
        const p = art.payload as { items?: unknown[] };
        return { kind: art.kind, count: Array.isArray(p.items) ? p.items.length : 0 };
      },
    },
    {
      typeId: 'PropertyTable',
      domainId: MOLECULAR_PROPS_DOMAIN_ID,
      label: 'ADMET Property Table',
      color: { bg: '#22c55e', fg: '#ffffff', border: '#16a34a' },
      peekPriority: 85,
      viewerId: 'mol-property-table-viewer',
      summarize: (art) => {
        const p = art.payload as { rows?: PropertyRow[] };
        const rows = Array.isArray(p.rows) ? p.rows : [];
        const bestLogS = rows.length > 0 ? Math.max(...rows.map((r) => r.logS)) : null;
        return { kind: art.kind, rowCount: rows.length, bestLogS };
      },
      formatNodeFooterKpi: (art) => {
        const p = art.payload as { rows?: PropertyRow[] };
        const rows = Array.isArray(p.rows) ? p.rows : [];
        if (rows.length === 0) return null;
        const bestLogS = Math.max(...rows.map((r) => r.logS));
        return { text: `logS ${bestLogS.toFixed(2)}`, tone: bestLogS >= -3 ? 'ok' : 'warn' };
      },
    },
    {
      typeId: 'ExplainabilityMap',
      domainId: MOLECULAR_PROPS_DOMAIN_ID,
      label: 'Atom Attribution Map',
      color: { bg: '#f97316', fg: '#ffffff', border: '#ea580c' },
      peekPriority: 60,
      viewerId: 'mol-attribution-viewer',
      summarize: (art) => ({ kind: art.kind }),
    },
  ],
  paramFieldKinds: [],
  formatPreflightSummary: (stageTotals) => {
    const preds = stageTotals['mol-predict'] ?? 0;
    return preds > 0 ? `${preds} ADMET prediction${preds === 1 ? '' : 's'}` : undefined;
  },
  artifactViewers: [
    {
      id: 'mol-property-table-viewer',
      domainId: MOLECULAR_PROPS_DOMAIN_ID,
      label: 'ADMET Property Table',
      supportedPortTypes: ['PropertyTable'],
      componentLoader: () => Promise.resolve({ default: {} }),
    },
  ],
  stageViewports: [],
};

const solubilityGnnPlugin: ModelPlugin = {
  id: 'solubility-gnn-demo',
  domainId: MOLECULAR_PROPS_DOMAIN_ID,
  manifest: {
    id: 'solubility-gnn-demo',
    displayName: 'Solubility GNN (ADMET)',
    shortName: 'Solubility GNN',
    stage: 'mol-predict',
    runtime: 'cpu-ts',
    estimatedVramMB: 0,
    approxWeightsMB: 0,
    licenseNote: 'MIT',
    description: 'Predicts aqueous solubility (logS) and lipophilicity (logP) for small molecules.',
    inputs: {},
    outputs: {
      properties: { type: 'PropertyTable' },
      attributions: { type: 'ExplainabilityMap', required: false },
    },
    modeParamKey: 'taskMode',
    modes: [
      { id: 'fast', label: 'Fast Predict', description: 'Predict logS/logP only.' },
      { id: 'explain', label: 'Predict + Atom Attribution', description: 'Emit per-atom attribution map.' },
    ],
    paramSchema: [
      {
        key: 'taskMode',
        label: 'Mode',
        kind: 'select',
        tier: 'primary',
        defaultValue: 'fast',
        options: [
          { value: 'fast', label: 'Fast Predict' },
          { value: 'explain', label: 'Predict + Attribution' },
        ],
      },
      {
        key: 'smilesCsv',
        label: 'SMILES List',
        kind: 'text',
        tier: 'primary',
        defaultValue: 'aspirin=CC(=O)OC1=CC=CC=C1C(=O)O,caffeine=CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
      },
    ],
  },
  execution: {
    runtime: 'cpu-ts',
    fallback: { kind: 'none', reason: 'Pure TS mock GNN' },
    vramLeaseMode: 'exclusive',
    executeCpu: async (ctx) => {
      const mode = String(ctx.node.params.taskMode ?? 'fast');
      const raw = String(ctx.node.params.smilesCsv ?? '');
      const rows: PropertyRow[] = raw
        .split(',')
        .map((pair) => pair.trim())
        .filter(Boolean)
        .map((pair, idx) => {
          const [name, smi] = pair.includes('=') ? pair.split('=') : [`cmpd_${idx + 1}`, pair];
          return {
            compoundId: name.trim(),
            smiles: (smi ?? '').trim(),
            logS: Number((-1.2 - idx * 0.65).toFixed(2)),
            logP: Number((1.1 + idx * 0.4).toFixed(2)),
          };
        });

      const tableArtifact: ArtifactEnvelope<'PropertyTable'> = {
        kind: 'PropertyTable',
        schemaVersion: '1.0',
        id: `${ctx.node.id}-props`,
        label: 'ADMET Solubility',
        sourceModelId: 'solubility-gnn-demo',
        domainId: MOLECULAR_PROPS_DOMAIN_ID,
        payload: { rows },
      };

      const out: Record<string, ArtifactEnvelope> = { properties: tableArtifact };
      if (mode === 'explain') {
        out.attributions = {
          kind: 'ExplainabilityMap',
          schemaVersion: '1.0',
          id: `${ctx.node.id}-attr`,
          label: 'Atom Attribution',
          sourceModelId: 'solubility-gnn-demo',
          domainId: MOLECULAR_PROPS_DOMAIN_ID,
          payload: { atoms: rows.length * 10 },
        };
      }
      return out;
    },
  },
  cachePolicy: {},
  share: {
    shortCode: 'sg',
  },
  catalog: {
    categoryId: 'mol-predict',
    conciseRole: 'Predict small-molecule aqueous solubility (logS) & logP',
    searchKeywords: ['admet', 'solubility', 'logs', 'logp', 'smiles', 'gnn'],
  },
  display: {
    formatCanvasSummary: (params) => `ADMET (${String(params.taskMode ?? 'fast')})`,
    resolveSlots: (params) => {
      const mode = String(params.taskMode ?? 'fast');
      return [
        {
          portName: 'properties',
          direction: 'out',
          availability: 'active',
          viewportRole: 'table',
        },
        {
          portName: 'attributions',
          direction: 'out',
          availability: mode === 'explain' ? 'active' : 'inactive-in-mode',
          inactiveReason:
            mode === 'explain' ? undefined : 'Only emitted in "Predict + Atom Attribution" mode.',
        },
      ];
    },
  },
  preflight: (ctx) => {
    const count = String(ctx.params.smilesCsv ?? '')
      .split(',')
      .filter((s) => s.trim()).length;
    return {
      estimatedVramMB: 0,
      estimatedDownloadMB: 0,
      workItems: Math.max(1, count),
      outputCardinality: { properties: Math.max(1, count) },
      stageBreakdown: { 'mol-predict': Math.max(1, count) },
    };
  },
  planner: {
    plannerRoles: ['scorer'],
    keywords: ['solubility', 'admet', 'logS', 'smiles'],
    capabilities: ['small-molecule', 'property-prediction'],
  },
};

describe('Phase 6: End-to-End Multi-Domain Extensibility Smoke Certification', () => {
  it('boots a non-protein DomainPack + ModelPlugin through Registry -> Scheduler -> Share -> Preflight -> Peek -> Compare -> Planner with ZERO core edits', async () => {
    const customRegistry = createDefaultPluginRegistry({
      extraDomainPacks: [molecularPropertiesDomainPack],
      extraPlugins: [solubilityGnnPlugin],
      freeze: true,
    });

    setActivePluginRegistry(customRegistry);
    try {
      // 1. Registry boot invariants
      expect(customRegistry.domains.has(MOLECULAR_PROPS_DOMAIN_ID)).toBe(true);
      expect(customRegistry.portTypes.get('PropertyTable')?.domainId).toBe(
        MOLECULAR_PROPS_DOMAIN_ID,
      );
      expect(customRegistry.resolveShortCode('sg')).toBe('solubility-gnn-demo');

      const graph: WorkflowGraph = {
        id: 'admet-smoke',
        name: 'ADMET Solubility Screen',
        nodes: [
          {
            id: 'admet1',
            adapterId: 'solubility-gnn-demo',
            label: 'Solubility GNN',
            params: {
              taskMode: 'fast',
              smilesCsv:
                'aspirin=CC(=O)OC1=CC=CC=C1C(=O)O,caffeine=CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            },
          },
        ],
        edges: [],
      };

      // 2. ExecutionScheduler executes the new non-protein CPU plugin cleanly
      const scheduler = new ExecutionScheduler({ pluginRegistry: customRegistry });
      const runResult = await scheduler.run(graph, () => {});
      const propsArt = runResult.outputsByNode.admet1?.properties as unknown as ArtifactEnvelope<
        'PropertyTable',
        { rows: PropertyRow[] }
      >;
      expect(propsArt).toBeDefined();
      expect(propsArt.kind).toBe('PropertyTable');
      expect(propsArt.payload.rows).toHaveLength(2);
      expect(propsArt.payload.rows[0].compoundId).toBe('aspirin');

      // PortType.summarize + formatNodeFooterKpi work on the produced artifact
      const ptDesc = customRegistry.portTypes.get('PropertyTable')!;
      expect(ptDesc.summarize(propsArt)).toEqual({
        kind: 'PropertyTable',
        rowCount: 2,
        bestLogS: -1.2,
      });
      expect(ptDesc.formatNodeFooterKpi?.(propsArt)).toEqual({
        text: 'logS -1.20',
        tone: 'ok',
      });

      // 3. Share (#g= URL hash AND YAML recipe) round-trip cleanly via PluginRegistry
      const encoded = await encodeGraphToHash(graph);
      expect(encoded.hash.startsWith('#g=')).toBe(true);
      const decoded = await decodeHashToGraph(encoded.hash);
      expect(decoded).not.toBeNull();
      expect(decoded!.nodes).toHaveLength(1);
      expect(decoded!.nodes[0].adapterId).toBe('solubility-gnn-demo');
      expect(decoded!.nodes[0].params.taskMode).toBe('fast');

      const yaml = graphToYamlRecipe(graph);
      const fromYaml = parseRecipeToGraph(yaml);
      expect(fromYaml.nodes[0].adapterId).toBe('solubility-gnn-demo');
      expect(fromYaml.nodes[0].params.smilesCsv).toContain('caffeine');

      // 4. Preflight runs without crashing on non-protein stages
      const preflight = estimateRunRequirements(graph);
      expect(preflight.totalDownloadMB).toBe(0);
      expect(preflight.peakVramMB).toBe(0);
      expect(preflight.usesWebGpu).toBe(false);

      // 5. Peek resolves custom resolveSlots (active viewportRole + inactive-in-mode slot)
      const peekSlots = getNodePeekSlots(
        graph,
        'admet1',
        runResult.outputsByNode as Record<string, Record<string, CanonicalArtifact>>,
      );
      const propSlot = peekSlots.find((s) => s.id === 'out:properties');
      const attrSlot = peekSlots.find((s) => s.id === 'out:attributions');
      expect(propSlot?.availability).toBe('active');
      expect(propSlot?.viewportRole).toBe('table');
      expect(attrSlot?.availability).toBe('inactive-in-mode');
      expect(attrSlot?.inactiveReason).toContain('Atom Attribution');
      expect(pickDefaultPeekSlot(peekSlots)).toBe('out:properties');

      // 6. Layer 1 compare-kernel joins PropertyTable rows across two model branches
      const joined = joinKeyedCollections([
        {
          branchId: 'gnn-v1',
          items: propsArt.payload.rows.map((r, i) => ({
            key: r.compoundId,
            rank: i + 1,
            score: r.logS,
            confidence: null,
            confidenceAvailable: false,
            value: r,
          })),
        },
        {
          branchId: 'gnn-v2',
          items: [
            {
              key: 'caffeine',
              rank: 1,
              score: -1.5,
              confidence: null,
              confidenceAvailable: false,
              value: propsArt.payload.rows[1],
            },
          ],
        },
      ]);
      expect(joined.keys).toEqual(['aspirin', 'caffeine']);
      expect(joined.completeRowCount).toBe(1);
      expect(joined.partialRowCount).toBe(1);
      expect(joined.rows[0].byBranch['gnn-v2'].present).toBe(false);

      // 7. Layer 1 Planner Catalog Index auto-discovers the new domain plugin
      const plannerIndex = buildPlannerCatalogIndex(customRegistry);
      expect(plannerIndex.isSupportedBlockName('solubility-gnn-demo')).toBe(true);
      expect(plannerIndex.hasRole('solubility-gnn-demo', 'scorer')).toBe(true);
      expect(plannerIndex.findByRole('scorer').map((e) => e.id)).toContain('solubility-gnn-demo');
      expect(plannerIndex.findByKeyword('solubility').map((e) => e.id)).toContain(
        'solubility-gnn-demo',
      );
    } finally {
      setActivePluginRegistry(defaultPluginRegistry);
    }
  });
});
