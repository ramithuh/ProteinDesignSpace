// Layer 3 Native ModelPlugin — pulchra (PULCHRA 3.06 all-atom reconstruction)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { ArtifactEnvelope, KernelAdapterManifest, ModelPlugin } from '$lib/kernel';
import type { CanonicalArtifact, WorkflowNode } from '$lib/engine/types';
import { runPulchraCpu } from './cpu-runner';

export { runPulchraCpu };

/** Size of static/vendor/pulchra/pulchra-tables.bin, fetched once per page. */
const PULCHRA_TABLES_MB = 1.7;

export const pulchraPlugin: ModelPlugin = {
  id: 'pulchra',
  domainId: PROTEIN_DOMAIN_ID,
  manifest: ADAPTER_REGISTRY['pulchra'] as unknown as KernelAdapterManifest,
  execution: {
    runtime: 'cpu-ts',
    fallback: { kind: 'none', reason: 'Pure TypeScript CPU adapter' },
    vramLeaseMode: 'exclusive',
    executeCpu: async (ctx) => {
      const out = await runPulchraCpu(
        ctx.node as unknown as WorkflowNode,
        ctx.inputs as unknown as Record<string, CanonicalArtifact>,
        (ev) => ctx.onProgress(ev),
        { signal: ctx.abortSignal },
      );
      return out as unknown as Record<string, ArtifactEnvelope>;
    },
  },
  cachePolicy: {},
  share: {
    shortCode: 'pu',
  },
  catalog: {
    categoryId: 'refinement',
    conciseRole: 'Cα / backbone trace → all-atom model (PULCHRA 3.06 backbone, rotamers & H-bonds)',
    searchKeywords: ['pulchra', 'all-atom', 'reconstruct', 'rebuild', 'side chain', 'sidechain', 'backbone', 'c-alpha', 'ca trace'],
  },
  planner: {
    plannerRole: 'optimizer',
    keywords: ['pulchra', 'all-atom', 'reconstruct', 'side chains', 'ca trace'],
    capabilities: ['ca-to-all-atom', 'backbone-reconstruction', 'sidechain-packing', 'sequence-threading'],
  },
  display: {
    formatCanvasSummary: (params) => {
      const rebuildBb = params.reconstructBackbone !== false;
      const rebuildSc = params.reconstructSidechains !== false;
      const modeLabel = !rebuildBb
        ? rebuildSc
          ? 'Sidechains'
          : 'Cα only'
        : rebuildSc
          ? 'All-atom'
          : 'Backbone';
      const parts = [modeLabel];
      if (params.optimizeCa === true) parts.push('Cα refine');
      if (rebuildBb && params.optimizeHbonds === true) parts.push('H-bonds');
      if (rebuildBb && params.outputHydrogens === true) parts.push('+H');
      return parts.join(' · ');
    },
    displayOnlyParamKeys: [],
    compareEligible: true,
  },
  preflight: (ctx) => {
    const items = ctx.connectedInputs.has('sequences')
      ? (ctx.upstreamProfiles.sequences?.itemCount ?? 1)
      : (ctx.upstreamProfiles.structure?.itemCount ?? 1);
    return {
      estimatedVramMB: 0,
      estimatedDownloadMB: PULCHRA_TABLES_MB,
      workItems: items,
      outputCardinality: { structure: items },
      stageBreakdown: { score: items },
    };
  },
};
