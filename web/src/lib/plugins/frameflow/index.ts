// Layer 3 Native ModelPlugin — frameflow (FrameFlow SE(3) Flow Matching)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.frameflow as unknown as KernelAdapterManifest;

export const frameflowPlugin: ModelPlugin = {
  id: 'frameflow',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./frameflow.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['frameflow-v1'],
    opfsDirs: ['frameflow-v1'],
  },
  share: {
    shortCode: 'ff',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'Compact 16.8M rigid-frame flow matching for 5-atom backbones',
    searchKeywords: [
      'frameflow',
      'flow matching',
      'rigid frame',
      'se3',
      'backbone',
      'microsoft',
      'mit',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('frameflow', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const len = Number(params.length ?? 80);
      const steps = Number(params.steps ?? 100);
      const prec = String(params.precision ?? 'fp16').toUpperCase();
      const n = Number(params.numDesigns ?? 1);
      return `${len} aa · ${steps} steps · ${prec}${n > 1 ? ` · ×${n}` : ''}`;
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
        int8: { weightsMB: 17, vramMB: 320 },
        fp16: { weightsMB: 33, vramMB: 380 },
      },
    };
  },
};
