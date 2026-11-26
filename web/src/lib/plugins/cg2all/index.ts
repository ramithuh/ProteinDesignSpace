// Layer 3 Native ModelPlugin — cg2all (Cα trace → all-atom backmapping, huhlim/cg2all CalphaBasedModel)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.cg2all as unknown as KernelAdapterManifest;

export const cg2allPlugin: ModelPlugin = {
  id: 'cg2all',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: {
      kind: 'cpu-js',
      moduleUrl: '/vendor/cg2all/cpu/index.js',
      degradedWarning: 'Running cg2all on the CPU JS fallback (much slower than WebGPU).',
    },
    createWorker: async () => new (await import('./cg2all.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    // Weights stream through makeCachedFetch into the shared denovo-run store.
    usesSharedOpfsStore: true,
  },
  share: {
    shortCode: 'cg',
  },
  catalog: {
    categoryId: 'score',
    conciseRole: 'Neural Cα trace → all-atom model (cg2all equivariant transformer, rigid-group side chains)',
    searchKeywords: [
      'cg2all',
      'all-atom',
      'backmapping',
      'backmap',
      'coarse-grained',
      'reconstruct',
      'rebuild',
      'side chain',
      'sidechain',
      'c-alpha',
      'ca trace',
      'se3',
    ],
  },
  planner: {
    plannerRole: 'optimizer',
    keywords: ['cg2all', 'all-atom', 'backmapping', 'side chains', 'ca trace', 'coarse-grained'],
    capabilities: ['ca-to-all-atom', 'backbone-reconstruction', 'sidechain-packing', 'sequence-threading'],
  },
  modelCard: {
    resolveCard: (params) => getLegacyModelCardForAdapter('cg2all', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const parts = [params.variant === 'ca-fix' ? 'Fixed Cα' : 'Cα refined'];
      parts.push(String(params.precision ?? 'fp16').toUpperCase());
      if (params.outputHydrogens === true) parts.push('+H');
      if (params.preserveLigands === false) parts.push('no ligands');
      return parts.join(' · ');
    },
    displayOnlyParamKeys: [],
    compareEligible: true,
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'fp16').toLowerCase();
    const items = ctx.connectedInputs.has('sequences')
      ? (ctx.upstreamProfiles.sequences?.itemCount ?? 1)
      : (ctx.upstreamProfiles.structure?.itemCount ?? 1);
    return {
      estimatedVramMB: manifest.estimatedVramMB,
      estimatedDownloadMB: manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB,
      workItems: items,
      outputCardinality: { structure: items },
      stageBreakdown: { score: items },
    };
  },
};
