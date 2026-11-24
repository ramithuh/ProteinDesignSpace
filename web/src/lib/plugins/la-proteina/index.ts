// Layer 3 Native ModelPlugin — la-proteina (La-Proteina LD1 + AE1)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY['la-proteina'] as unknown as KernelAdapterManifest;

export const laProteinaPlugin: ModelPlugin = {
  id: 'la-proteina',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./laproteina.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['la-proteina-v1'],
    opfsDirs: ['la-proteina-v1'],
  },
  share: {
    shortCode: 'lp',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'All-atom latent diffusion (LD1 + AE1) co-designing structure and sequence',
    searchKeywords: [
      'la-proteina',
      'laproteina',
      'ld1',
      'ae1',
      'latent',
      'all-atom',
      'co-design',
      'diffusion',
      'nvidia',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('la-proteina', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const len = Number(params.length ?? 80);
      const steps = Number(params.steps ?? 400);
      const mode = String(params.mode ?? 'sde').toUpperCase();
      const n = Number(params.numDesigns ?? 1);
      return `${len} aa · ${steps} steps · ${mode}${n > 1 ? ` · ×${n}` : ''}`;
    },
  },
  preflight: (ctx) => {
    const prec = String(ctx.params.precision ?? 'int8').toLowerCase() === 'fp16' ? 'fp16' : 'int8';
    const n = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numDesigns ?? 1) || 1)));
    return {
      estimatedVramMB: prec === 'fp16' ? 1200 : manifest.estimatedVramMB,
      estimatedDownloadMB: manifest.precisionWeightsMB?.[prec] ?? manifest.approxWeightsMB,
      workItems: n,
      outputCardinality: { structure: n, sequences: n },
      stageBreakdown: { generate: n },
      precisionOptions: {
        int8: { weightsMB: 277, vramMB: 900 },
        fp16: { weightsMB: 548, vramMB: 1200 },
      },
    };
  },
};
