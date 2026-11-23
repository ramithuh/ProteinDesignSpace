// Layer 3 Native ModelPlugin — thermompnn (ThermoMPNN ΔΔG Stability)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.thermompnn as unknown as KernelAdapterManifest;

export const thermompnnPlugin: ModelPlugin = {
  id: 'thermompnn',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: {
      kind: 'cpu-js',
      moduleUrl: '/vendor/thermompnn/cpu/index.js',
      degradedWarning: 'Running ThermoMPNN on CPU JS fallback (slower than WebGPU).',
    },
    createWorker: async () => new (await import('./thermompnn.worker?worker')).default(),
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['thermompnn-v1'],
    opfsDirs: ['thermompnn-v1'],
  },
  share: {
    shortCode: 'tm',
  },
  catalog: {
    categoryId: 'score',
    conciseRole: 'Full L×20 site-saturation ΔΔG stability scan & B-factor painting',
    searchKeywords: [
      'thermompnn',
      'ddg',
      'stability',
      'mutation',
      'ssm',
      'site-saturation',
      'thermostability',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('thermompnn', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const chain = String(params.chain ?? '').trim();
      const agg = String(params.bfactorAggregate ?? 'best-stabilizing');
      const aggLbl =
        agg === 'mean-ddg'
          ? 'mean ΔΔG'
          : agg === 'worst-destabilizing'
            ? 'max ΔΔG'
            : 'min ΔΔG';
      const prec = String(params.precision ?? 'fp16').toUpperCase();
      return `${chain ? `Chain ${chain} · ` : ''}${aggLbl} · ${prec}`;
    },
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'fp16').toLowerCase();
    const approxMB = manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB;
    const upBackbones = ctx.upstreamProfiles.structure?.itemCount ?? 1;
    return {
      estimatedVramMB: manifest.estimatedVramMB,
      estimatedDownloadMB: approxMB,
      workItems: upBackbones,
      outputCardinality: { mutations: 1, structure: upBackbones },
      stageBreakdown: { score: upBackbones },
    };
  },
};
