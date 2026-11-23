// Layer 3 Native ModelPlugin — structure-compare

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { ArtifactEnvelope, KernelAdapterManifest, ModelPlugin } from '$lib/kernel';
import type { CanonicalArtifact, WorkflowNode } from '$lib/engine/types';
import { runStructureCompareCpu } from './cpu-runner';

export { runStructureCompareCpu };

export const structureComparePlugin: ModelPlugin = {
  id: 'structure-compare',
  domainId: PROTEIN_DOMAIN_ID,
  manifest: ADAPTER_REGISTRY['structure-compare'] as unknown as KernelAdapterManifest,
  execution: {
    runtime: 'cpu-ts',
    fallback: { kind: 'none', reason: 'Pure TypeScript CPU adapter' },
    vramLeaseMode: 'exclusive',
    executeCpu: async (ctx) => {
      const out = await runStructureCompareCpu(
        ctx.node as unknown as WorkflowNode,
        ctx.inputs as unknown as Record<string, CanonicalArtifact>,
        ctx.multiInputs as unknown as Record<string, CanonicalArtifact[]> | undefined,
        (ev) => ctx.onProgress(ev),
      );
      return out as unknown as Record<string, ArtifactEnvelope>;
    },
  },
  cachePolicy: {},
  share: {
    shortCode: 'c',
  },
  catalog: {
    categoryId: 'score',
    conciseRole: 'Multi-model 3D Kabsch / motif overlay & pairwise RMSD',
    searchKeywords: ['compare', 'overlay', 'kabsch', 'rmsd', 'superpose', 'side-by-side'],
  },
  display: {
    formatCanvasSummary: (params) => {
      const layout = String(params.layout ?? 'overlay');
      return layout === 'side-by-side' ? 'Side-by-Side' : '3D Overlay';
    },
    displayOnlyParamKeys: ['layout', 'colorMode', 'alignKabsch'],
    compareEligible: true,
  },
  preflight: () => ({
    estimatedVramMB: 0,
    estimatedDownloadMB: 0,
    workItems: 0,
    outputCardinality: { structure: 1 },
    stageBreakdown: { score: 1 },
  }),
};
