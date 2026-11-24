// Phase 4 Contract Gate: Dynamic PlannerCatalogIndex & Plugin #27 Auto-Discovery.
// Proves that registering a brand-new plugin #27 with `plannerRole: 'generator'` (or a
// multi-role omni-model) automatically makes it discoverable to the planner & Block-Graph
// IR compiler with ZERO edits to `workflow-planner.ts`.

import { describe, expect, it, afterEach } from 'vitest';
import {
  buildPlannerCatalogIndex,
  setActivePluginRegistry,
  type ModelPlugin,
} from '$lib/kernel';
import { createDefaultPluginRegistry, defaultPluginRegistry } from '$lib/plugins';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import {
  compileWorkflowPlanToGraph,
  reconcilePlannedWorkflow,
  type BlockGraphWorkflowPlan,
} from '$lib/engine/workflow-planner';
import { validateGraph } from '$lib/engine/graph-utils';

function makeMockPlugin27(): ModelPlugin {
  return {
    id: 'superdiff-27',
    domainId: PROTEIN_DOMAIN_ID,
    manifest: {
      id: 'superdiff-27',
      displayName: 'SuperDiff 27 (Mock Generator)',
      shortName: 'SuperDiff-27',
      stage: 'generate',
      runtime: 'cpu-ts',
      estimatedVramMB: 0,
      approxWeightsMB: 0,
      licenseNote: 'MIT',
      description: 'Mock 27th plugin generator for Phase 4 auto-discovery contract test',
      inputs: {
        template: { type: 'StructureArtifact', required: false },
      },
      outputs: {
        structure: { type: 'StructureArtifact' },
        sequences: { type: 'SequenceSet' },
      },
      paramSchema: [
        {
          key: 'length',
          label: 'Length',
          kind: 'integer',
          tier: 'primary',
          defaultValue: 80,
          min: 16,
          max: 512,
        },
        {
          key: 'numDesigns',
          label: 'Designs',
          kind: 'integer',
          tier: 'primary',
          defaultValue: 1,
          min: 1,
          max: 8,
        },
      ],
    },
    execution: {
      runtime: 'cpu-ts',
      fallback: { kind: 'none', reason: 'test' },
      executeCpu: async () => ({}),
    },
    cachePolicy: {},
    share: { shortCode: 'sd7' },
    catalog: {
      categoryId: 'generate',
      conciseRole: 'Mock 27th generator plugin',
      searchKeywords: ['superdiff', 'diff27', 'mockgen'],
    },
    display: {
      formatCanvasSummary: () => 'SuperDiff-27',
    },
    preflight: () => ({
      estimatedVramMB: 0,
      estimatedDownloadMB: 0,
      workItems: 1,
    }),
    planner: {
      plannerRole: 'generator',
      keywords: ['superdiff', 'superdiff27', 'diffusion27'],
      capabilities: ['de-novo', 'motif'],
      repairRules: [
        {
          issueCode: 'MOCK_LEN_TOO_HIGH',
          description: 'Clamp length to 256',
          apply: (node) => ({ params: { ...node.params, length: 256 } }),
        },
      ],
    },
  };
}

function makeOmniModelPlugin28(): ModelPlugin {
  return {
    id: 'omni-fold-score-28',
    domainId: PROTEIN_DOMAIN_ID,
    manifest: {
      id: 'omni-fold-score-28',
      displayName: 'OmniFold+Score 28',
      shortName: 'Omni-28',
      stage: 'fold',
      secondaryStages: ['score'],
      runtime: 'cpu-ts',
      estimatedVramMB: 0,
      approxWeightsMB: 0,
      licenseNote: 'MIT',
      description: 'Multi-role omni predictor + scorer',
      inputs: {
        sequences: { type: 'SequenceSet', required: true },
      },
      outputs: {
        folds: { type: 'FoldedStructure' },
      },
      paramSchema: [
        { key: 'recycles', label: 'Recycles', kind: 'integer', tier: 'primary', defaultValue: 3 },
      ],
    },
    execution: {
      runtime: 'cpu-ts',
      fallback: { kind: 'none', reason: 'test' },
      executeCpu: async () => ({}),
    },
    cachePolicy: {},
    share: { shortCode: 'om8' },
    catalog: {
      categoryId: 'fold',
      conciseRole: 'Multi-role predictor and affinity scorer',
      searchKeywords: ['omni28', 'cofold-affinity'],
    },
    display: { formatCanvasSummary: () => 'Omni-28' },
    preflight: () => ({ estimatedVramMB: 0, estimatedDownloadMB: 0, workItems: 1 }),
    planner: {
      plannerRoles: ['predictor', 'scorer'],
      keywords: ['omni28', 'omnifold'],
      capabilities: ['folding', 'affinity-scoring'],
    },
  };
}

describe('Phase 4 contract: PlannerCatalogIndex & Plugin #27 auto-discovery', () => {
  afterEach(() => {
    // Always restore production defaultPluginRegistry after test
    setActivePluginRegistry(defaultPluginRegistry);
  });

  it('auto-discovers plugin #27 (generator) and plugin #28 (multi-role omni) with zero edits to workflow-planner.ts', () => {
    const reg = createDefaultPluginRegistry({
      extraPlugins: [makeMockPlugin27(), makeOmniModelPlugin28()],
      freeze: true,
    });
    setActivePluginRegistry(reg);

    const idx = buildPlannerCatalogIndex(reg);
    expect(idx.plugins).toHaveLength(29);
    expect(idx.getGeneratorIds()).toContain('superdiff-27');
    expect(idx.findByKeyword('superdiff').map((p) => p.id)).toContain('superdiff-27');
    expect(idx.isSupportedBlockName('SuperDiff-27')).toBe(true);
    expect(idx.isSupportedBlockName('diffusion27')).toBe(true);

    // Multi-role omni plugin appears in BOTH predictor and scorer role queries
    expect(idx.findByRole('predictor').map((p) => p.id)).toContain('omni-fold-score-28');
    expect(idx.findByRole('scorer').map((p) => p.id)).toContain('omni-fold-score-28');

    // Plugin-owned repairRules execute cleanly through PlannerCatalogIndex
    const repaired = idx.applyRepairRules(
      { id: 'g27', adapterId: 'superdiff-27', params: { length: 999 } },
      { severity: 'error', code: 'MOCK_LEN_TOO_HIGH', nodeId: 'g27', message: 'too long' },
    );
    expect(repaired?.params?.length).toBe(256);

    // End-to-end Block-Graph IR reconciliation + compilation + validation with plugin #27 -> MPNN -> Omni-28 -> Scorer
    const plan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'gen_27', adapter: 'superdiff-27', label: 'SuperDiff-27', params: { length: 64 } },
        { id: 'mpnn_1', adapter: 'web-protein-mpnn', label: 'SolubleMPNN', params: { family: 'solublempnn' } },
        { id: 'omni_1', adapter: 'omni-fold-score-28', label: 'Omni-28', params: { recycles: 3 } },
        { id: 'score_1', adapter: 'designability-scorer', label: 'Scorer', params: {} },
      ],
      edges: [
        ['gen_27', 'mpnn_1'],
        ['mpnn_1', 'omni_1'],
        ['omni_1', 'score_1'],
      ],
      // Deliberately list SuperDiff-27 in unsupported to confirm reconcile filters it out dynamically!
      unsupported: ['SuperDiff-27', 'NonExistentModelXYZ'],
      summary: 'Design 64aa protein with SuperDiff-27 -> SolubleMPNN -> Omni-28 -> Scorer',
    };

    const reconciled = reconcilePlannedWorkflow(plan, 'Design with SuperDiff-27 and Omni-28') as BlockGraphWorkflowPlan;
    const { graph, unsupported } = compileWorkflowPlanToGraph(reconciled);
    expect(unsupported).toEqual(['NonExistentModelXYZ']);
    expect(validateGraph(graph)).toEqual([]);
    expect(graph.nodes.map((n) => n.adapterId)).toEqual([
      'superdiff-27',
      'web-protein-mpnn',
      'omni-fold-score-28',
      'designability-scorer',
    ]);
    // Scorer auto-wired back to superdiff-27 because isGeneratorAdapter('superdiff-27') === true!
    expect(
      graph.edges.some(
        (e) => e.fromNode === 'gen_27' && e.toNode === 'score_1' && e.toPort === 'structure',
      ),
    ).toBe(true);
  });
});
