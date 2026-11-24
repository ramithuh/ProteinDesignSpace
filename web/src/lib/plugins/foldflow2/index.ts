// Layer 3 Native ModelPlugin — foldflow2 (FoldFlow 2 ReFT + ESM-2 650M)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.foldflow2 as unknown as KernelAdapterManifest;

export const foldflow2Plugin: ModelPlugin = {
  id: 'foldflow2',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./foldflow2.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['foldflow2-v1'],
    opfsDirs: ['foldflow2-v1'],
  },
  share: {
    shortCode: 'f2',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'ReFT rigid-frame flow matching with frozen ESM-2 650M encoder',
    searchKeywords: [
      'foldflow',
      'foldflow2',
      'reft',
      'esm2',
      'flow matching',
      'sequence conditioned',
      'dreamfold',
      'non-commercial',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('foldflow2', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const len = Number(params.length ?? 80);
      const steps = Number(params.steps ?? 50);
      const prec = String(params.precision ?? 'int8').toUpperCase();
      const n = Number(params.numDesigns ?? 1);
      return `${len} aa · ${steps} steps · ${prec}${n > 1 ? ` · ×${n}` : ''}`;
    },
  },
  preflight: (ctx) => {
    const prec = String(ctx.params.precision ?? 'int8').toLowerCase() === 'fp16' ? 'fp16' : 'int8';
    const n = Math.max(1, Math.min(6, Math.round(Number(ctx.params.numDesigns ?? 1) || 1)));
    return {
      estimatedVramMB: prec === 'fp16' ? 2400 : manifest.estimatedVramMB,
      estimatedDownloadMB: manifest.precisionWeightsMB?.[prec] ?? manifest.approxWeightsMB,
      workItems: n,
      outputCardinality: { structure: n },
      stageBreakdown: { generate: n },
      precisionOptions: {
        int8: { weightsMB: 644, vramMB: 1400 },
        fp16: { weightsMB: 1340, vramMB: 2400 },
      },
    };
  },
};
