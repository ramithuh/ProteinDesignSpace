// Layer 3 Native ModelPlugin — proteina (Proteina v1.2 200M)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.proteina as unknown as KernelAdapterManifest;

export const proteinaPlugin: ModelPlugin = {
  id: 'proteina',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./proteina.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['proteina-v1'],
    opfsDirs: ['proteina-v1'],
  },
  share: {
    shortCode: 'pr',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'Scalable 200M transformer flow matching for de novo Cα backbones',
    searchKeywords: [
      'proteina',
      '200m',
      'v1.2',
      'flow matching',
      'backbone',
      'ca',
      'nvidia',
      'non-commercial',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('proteina', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const len = Number(params.length ?? 80);
      const steps = Number(params.steps ?? 400);
      const modeTag = String(params.mode ?? 'sc') === 'vf' ? 'ODE' : 'SDE';
      const n = Number(params.numDesigns ?? 1);
      return `${len} aa · ${steps} steps · ${modeTag}${n > 1 ? ` · ×${n}` : ''}`;
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
        int8: { weightsMB: 186, vramMB: 720 },
        fp16: { weightsMB: 366, vramMB: 900 },
      },
    };
  },
};
