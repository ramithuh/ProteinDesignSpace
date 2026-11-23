// Layer 3 Native ModelPlugin — lasermpnn (Ligand-aware sequence & all-atom sidechain rotamer co-design)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.lasermpnn as unknown as KernelAdapterManifest;

export const lasermpnnPlugin: ModelPlugin = {
  id: 'lasermpnn',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./lasermpnn.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['lasermpnn-v1'],
    usesSharedOpfsStore: true,
  },
  share: {
    shortCode: 'la',
  },
  catalog: {
    categoryId: 'inverse-fold',
    conciseRole: 'Ligand-aware sequence & all-atom sidechain rotamer co-design',
    searchKeywords: [
      'lasermpnn',
      'laser',
      'sidechain',
      'rotamer',
      'packing',
      'all-atom',
      'ligand',
      'hydrogens',
      'inverse folding',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('lasermpnn', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const n = Number(params.numSequences ?? 2);
      const t = Number(params.temperature ?? 0.1);
      const prec = String(params.precision ?? 'fp16').toUpperCase();
      return `All-atom · ${n} seq${n === 1 ? '' : 's'} · T=${t} · ${prec}`;
    },
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'fp16').toLowerCase();
    const approxMB = manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB;
    const upBackbones = ctx.upstreamProfiles.structure?.itemCount ?? 1;
    const k = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numSequences ?? 2) || 2)));
    const total = upBackbones * k;
    return {
      estimatedVramMB: manifest.estimatedVramMB,
      estimatedDownloadMB: approxMB,
      workItems: total,
      outputCardinality: { sequences: total, structure: total },
      stageBreakdown: { 'inverse-fold': total },
    };
  },
};
