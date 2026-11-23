// Layer 3 Native ModelPlugin — genie3 (SE(3)-equivariant Cα Backbone Diffusion)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.genie3 as unknown as KernelAdapterManifest;

function numOr(val: unknown, fallback: number): number {
  if (val === undefined || val === null || typeof val === 'boolean') return fallback;
  if (typeof val === 'string' && val.trim() === '') return fallback;
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

export const genie3Plugin: ModelPlugin = {
  id: 'genie3',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./genie3.worker?worker')).default(),
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStoragePrefixes: ['genie3-weights-'],
  },
  share: {
    shortCode: 'g',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'Fast de novo backbone diffusion (23.3M parameters)',
    searchKeywords: [
      'genie3',
      'genie',
      'diffusion',
      'ddim',
      'backbone',
      'ca',
      'binder',
      'motif',
      'sequencenet',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('genie3', params) as unknown as KernelModelCardSpec,
  },
  display: {
    isSelfGenerator: true,
    formatCanvasSummary: (params) => {
      const len = numOr(params.length, 80);
      const steps = numOr(params.nSampleStep, 100);
      const tm = String(params.taskMode ?? 'unconditional');
      if (tm === 'binder') {
        const hotCount = String(params.hotspots ?? '').split(/[,\s]+/).filter(Boolean).length;
        return `Binder · ${len} aa${hotCount > 0 ? ` · ${hotCount} hot` : ''} · ${steps} steps`;
      }
      if (tm === 'motif') {
        const motif = String(params.motifSpan ?? '').trim();
        return `Motif${motif ? ` (${motif})` : ''} · ${steps} steps`;
      }
      return `${len} aa · ${steps} steps`;
    },
  },
  preflight: (ctx) => {
    const n = Math.max(1, Math.min(16, Math.round(numOr(ctx.params.numDesigns, 1))));
    return {
      estimatedVramMB: manifest.estimatedVramMB,
      estimatedDownloadMB: manifest.approxWeightsMB,
      workItems: n,
      outputCardinality: { structure: n, sequences: n },
      stageBreakdown: { generate: n },
    };
  },
};
