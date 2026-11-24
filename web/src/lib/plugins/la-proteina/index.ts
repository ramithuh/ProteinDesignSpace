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
    conciseRole: 'All-atom latent diffusion (LD1–LD7 + AE1/AE3) for de novo co-design & motif scaffolding',
    searchKeywords: [
      'la-proteina',
      'laproteina',
      'ld1',
      'ld4',
      'ld5',
      'ld6',
      'ld7',
      'ae1',
      'ae3',
      'motif',
      'scaffolding',
      'tip-atom',
      'unindexed',
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
      const variant = String(params.variant ?? 'ld1').trim().toLowerCase();
      const len = Number(params.length ?? 80);
      const steps = Number(params.steps ?? 400);
      const mode = String(params.mode ?? 'sde').toUpperCase();
      const n = Number(params.numDesigns ?? 1);
      if (variant !== 'ld1' && /^ld[4-7]$/.test(variant)) {
        const contig = String(params.contig ?? '').trim();
        const tag = variant.toUpperCase();
        const span = contig ? `(${contig})` : `${len} aa`;
        return `${tag} motif ${span} · ${steps} steps · ${mode}${n > 1 ? ` · ×${n}` : ''}`;
      }
      return `${len} aa · ${steps} steps · ${mode}${n > 1 ? ` · ×${n}` : ''}`;
    },
  },
  preflight: (ctx) => {
    const prec = String(ctx.params.precision ?? 'int8').toLowerCase() === 'fp16' ? 'fp16' : 'int8';
    const variant = String(ctx.params.variant ?? 'ld1').trim().toLowerCase();
    const isMotif = /^ld[4-7]$/.test(variant);
    const n = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numDesigns ?? 1) || 1)));
    const int8MB = isMotif ? 402 : (manifest.precisionWeightsMB?.int8 ?? manifest.approxWeightsMB);
    const fp16MB = isMotif ? 794 : (manifest.precisionWeightsMB?.fp16 ?? 548);
    const int8Vram = isMotif ? 1100 : manifest.estimatedVramMB;
    const fp16Vram = isMotif ? 1500 : 1200;
    return {
      estimatedVramMB: prec === 'fp16' ? fp16Vram : int8Vram,
      estimatedDownloadMB: prec === 'fp16' ? fp16MB : int8MB,
      workItems: n,
      outputCardinality: { structure: n, sequences: n },
      stageBreakdown: { generate: n },
      precisionOptions: {
        int8: { weightsMB: int8MB, vramMB: int8Vram },
        fp16: { weightsMB: fp16MB, vramMB: fp16Vram },
      },
    };
  },
};
