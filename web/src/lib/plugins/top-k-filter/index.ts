// Layer 3 Native ModelPlugin — top-k-filter

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { ArtifactEnvelope, KernelAdapterManifest, ModelPlugin } from '$lib/kernel';
import type { CanonicalArtifact, WorkflowNode } from '$lib/engine/types';
import { runTopKFilterCpu } from './cpu-runner';

export { runTopKFilterCpu };

export const topKFilterPlugin: ModelPlugin = {
  id: 'top-k-filter',
  domainId: PROTEIN_DOMAIN_ID,
  manifest: ADAPTER_REGISTRY['top-k-filter'] as unknown as KernelAdapterManifest,
  execution: {
    runtime: 'cpu-ts',
    fallback: { kind: 'none', reason: 'Pure TypeScript CPU adapter' },
    vramLeaseMode: 'exclusive',
    executeCpu: async (ctx) => {
      const out = await runTopKFilterCpu(
        ctx.node as unknown as WorkflowNode,
        ctx.inputs as unknown as Record<string, CanonicalArtifact>,
      );
      return out as unknown as Record<string, ArtifactEnvelope>;
    },
  },
  cachePolicy: {},
  share: {
    shortCode: 'tk',
  },
  catalog: {
    categoryId: 'score',
    conciseRole: 'Rank and filter top-K sequences per backbone or globally before folding',
    searchKeywords: ['top-k', 'filter', 'rank', 'select', 'sequences', 'triage'],
  },
  display: {
    formatCanvasSummary: (params) => {
      const mode = String(params.mode ?? 'per-backbone');
      const k = Number(params.topK ?? 2);
      const cap = Number(params.maxTotal ?? 8);
      return mode === 'global' ? `Top ${k} global (≤${cap})` : `Top ${k}/backbone (≤${cap})`;
    },
  },
  preflight: (ctx) => {
    const upSeqs = ctx.upstreamProfiles.sequences?.itemCount ?? 1;
    const topK = Math.max(1, Math.min(16, Math.round(Number(ctx.params.topK ?? 2) || 2)));
    const maxTotal = Math.max(1, Math.min(64, Math.round(Number(ctx.params.maxTotal ?? 8) || 8)));
    const kept = Math.min(upSeqs, maxTotal, Math.max(topK, 1));
    return {
      estimatedVramMB: 0,
      estimatedDownloadMB: 0,
      workItems: kept,
      outputCardinality: { sequences: kept },
      stageBreakdown: { score: kept },
    };
  },
};
