// Layer 3 Native ModelPlugin — sid-protein (SiD-Protein 16-Step Distilled)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY['sid-protein'] as unknown as KernelAdapterManifest;

export const sidProteinPlugin: ModelPlugin = {
  id: 'sid-protein',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./sidprotein.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['sid-protein-v1'],
    opfsDirs: ['sid-protein-v1'],
  },
  share: {
    shortCode: 'sp',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'Distilled 16-step Proteina sampler for rapid Cα backbone generation',
    searchKeywords: [
      'sid-protein',
      'sid',
      'distilled',
      '16-step',
      'fast',
      'proteina',
      'backbone',
      'ca',
      'non-commercial',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('sid-protein', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const len = Number(params.length ?? 80);
      const steps = Number(params.steps ?? 16);
      const prec = String(params.precision ?? 'int8').toUpperCase();
      const n = Number(params.numDesigns ?? 1);
      return `${len} aa · ${steps} steps · ${prec}${n > 1 ? ` · ×${n}` : ''}`;
    },
  },
  preflight: (ctx) => {
    const prec = String(ctx.params.precision ?? 'int8').toLowerCase() === 'fp16' ? 'fp16' : 'int8';
    const n = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numDesigns ?? 1) || 1)));
    return {
      estimatedVramMB: prec === 'fp16' ? 900 : manifest.estimatedVramMB,
      estimatedDownloadMB: manifest.precisionWeightsMB?.[prec] ?? manifest.approxWeightsMB,
      workItems: n,
      outputCardinality: { structure: n },
      stageBreakdown: { generate: n },
      precisionOptions: {
        int8: { weightsMB: 185, vramMB: 720 },
        fp16: { weightsMB: 366, vramMB: 900 },
      },
    };
  },
};
