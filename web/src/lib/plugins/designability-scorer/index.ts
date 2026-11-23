// Layer 3 Native ModelPlugin — designability-scorer

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { ArtifactEnvelope, KernelAdapterManifest, ModelPlugin } from '$lib/kernel';
import type { CanonicalArtifact, WorkflowNode } from '$lib/engine/types';
import {
  resolveDesignabilityUpstreamLineage,
  runDesignabilityScorerCpu,
} from './cpu-runner';

export { resolveDesignabilityUpstreamLineage, runDesignabilityScorerCpu };

export const designabilityScorerPlugin: ModelPlugin = {
  id: 'designability-scorer',
  domainId: PROTEIN_DOMAIN_ID,
  manifest: ADAPTER_REGISTRY['designability-scorer'] as unknown as KernelAdapterManifest,
  execution: {
    runtime: 'cpu-ts',
    fallback: { kind: 'none', reason: 'Pure TypeScript CPU adapter' },
    vramLeaseMode: 'exclusive',
    resolveUpstreamLineage: ({ node, upstreamGraph }) =>
      resolveDesignabilityUpstreamLineage(node.id, upstreamGraph),
    executeCpu: async (ctx) => {
      const lineageHint = resolveDesignabilityUpstreamLineage(ctx.node.id, ctx.upstreamGraph);
      const out = await runDesignabilityScorerCpu(
        ctx.node as unknown as WorkflowNode,
        ctx.inputs as unknown as Record<string, CanonicalArtifact>,
        lineageHint,
      );
      return out as unknown as Record<string, ArtifactEnvelope>;
    },
  },
  cachePolicy: {},
  share: {
    shortCode: 's',
  },
  catalog: {
    categoryId: 'score',
    conciseRole: 'Kabsch alignment, scRMSD, scTM & pLDDT pass rate',
    searchKeywords: ['scrmsd', 'sctm', 'plddt', 'designability', 'score', 'kabsch', 'ipsae', 'ipae'],
  },
  display: {
    formatCanvasSummary: (params) => {
      const sc = Number(params.scRmsdThreshold ?? 2.0);
      const pl = Number(params.plddtThreshold ?? 70);
      return `scRMSD < ${sc} Å · pLDDT ≥ ${pl}`;
    },
    compareEligible: true,
    autoFocusPriority: 90,
  },
  preflight: () => ({
    estimatedVramMB: 0,
    estimatedDownloadMB: 0,
    workItems: 0,
    outputCardinality: { metrics: 1 },
    stageBreakdown: { score: 1 },
  }),
};
