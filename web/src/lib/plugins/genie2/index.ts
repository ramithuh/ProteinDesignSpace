// Layer 3 Native ModelPlugin — genie2 (Genie 2 Cα Diffusion)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.genie2 as unknown as KernelAdapterManifest;

export const genie2Plugin: ModelPlugin = {
  id: 'genie2',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./genie2.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['genie2-v1'],
    opfsDirs: ['genie2-v1'],
  },
  share: {
    shortCode: 'g2',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'Compact 15.7M Cα diffusion with optional soft motif scaffolding',
    searchKeywords: [
      'genie',
      'genie2',
      'motif',
      'scaffolding',
      'diffusion',
      'de novo',
      'backbone',
      'ca',
      'apache',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('genie2', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const len = Number(params.length ?? 80);
      const steps = Number(params.steps ?? 1000);
      const motif = String(params.motifRange ?? '').trim();
      const prec = String(params.precision ?? 'fp16').toUpperCase();
      const n = Number(params.numDesigns ?? 1);
      const nTag = n > 1 ? ` · ×${n}` : '';
      if (motif) {
        return `Motif (${motif}) · ${len} aa · ${steps} steps${nTag}`;
      }
      return `${len} aa · ${steps} steps · ${prec}${nTag}`;
    },
  },
  preflight: (ctx) => {
    const prec = String(ctx.params.precision ?? 'fp16').toLowerCase() === 'int8' ? 'int8' : 'fp16';
    const n = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numDesigns ?? 1) || 1)));
    return {
      estimatedVramMB: prec === 'int8' ? 320 : manifest.estimatedVramMB,
      estimatedDownloadMB: manifest.precisionWeightsMB?.[prec] ?? manifest.approxWeightsMB,
      workItems: n,
      outputCardinality: { structure: n },
      stageBreakdown: { generate: n },
      precisionOptions: {
        int8: { weightsMB: 16, vramMB: 320 },
        fp16: { weightsMB: 31, vramMB: 360 },
      },
    };
  },
};
