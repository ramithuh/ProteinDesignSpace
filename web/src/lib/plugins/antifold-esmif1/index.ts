// Layer 3 Native ModelPlugin — antifold-esmif1 (AntiFold antibody CDR & ESM-IF1 GVP-Transformer inverse folding)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY['antifold-esmif1'] as unknown as KernelAdapterManifest;

export const antifoldEsmif1Plugin: ModelPlugin = {
  id: 'antifold-esmif1',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./antifold.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['gvp-transformer-v1'],
    opfsDirs: ['gvp-transformer-v1'],
  },
  share: {
    shortCode: 'af',
  },
  catalog: {
    categoryId: 'inverse-fold',
    conciseRole: 'Antibody CDR (AntiFold) & general protein inverse folding (ESM-IF1)',
    searchKeywords: [
      'antifold',
      'esmif1',
      'esm-if1',
      'antibody',
      'nanobody',
      'vhh',
      'cdr',
      'imgt',
      'kabat',
      'gvp',
      'inverse folding',
    ],
    buildSubGroups: () => [
      {
        id: 'antifold',
        label: 'AntiFold (Antibody CDRs)',
        description: 'Antibody & nanobody CDR inverse folding (IMGT/Kabat regions)',
        paramsPatch: { variant: 'antifold' },
        nodeLabel: 'AntiFold',
        approxMB: 271,
      },
      {
        id: 'esmif1',
        label: 'ESM-IF1',
        description: 'General GVP-Transformer inverse folding (Meta AI · 142M params)',
        paramsPatch: { variant: 'esmif1' },
        nodeLabel: 'ESM-IF1',
        approxMB: 271,
      },
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('antifold-esmif1', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const variant = String(params.variant ?? 'antifold');
      const n = Number(params.numSequences ?? 2);
      const t = Number(params.temperature ?? 0.2);
      if (variant === 'esmif1') {
        return `ESM-IF1 · ${n} seq${n === 1 ? '' : 's'} · T=${t}`;
      }
      const regions = String(params.regions ?? 'CDR1 CDR2 CDR3').trim();
      return `AntiFold${regions ? ` (${regions})` : ''} · ${n} seq${n === 1 ? '' : 's'} · T=${t}`;
    },
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'fp16').toLowerCase();
    const approxMB = manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB;
    const estimatedVramMB = precision === 'int8' ? 520 : manifest.estimatedVramMB;
    const upBackbones = ctx.upstreamProfiles.structure?.itemCount ?? 1;
    const k = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numSequences ?? 2) || 2)));
    const total = upBackbones * k;
    return {
      estimatedVramMB,
      estimatedDownloadMB: approxMB,
      workItems: total,
      outputCardinality: { sequences: total },
      stageBreakdown: { 'inverse-fold': total },
    };
  },
};
