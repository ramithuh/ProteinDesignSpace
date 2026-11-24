// Layer 3 Native ModelPlugin — boltzgen (BoltzGen-1 all-atom unconditional, binder & motif diffusion)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.boltzgen as unknown as KernelAdapterManifest;

export const boltzgenPlugin: ModelPlugin = {
  id: 'boltzgen',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./boltzgen.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['boltzgen-v1'],
    opfsDirs: ['boltzgen-v1'],
  },
  share: {
    shortCode: 'b',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'BoltzGen-1 all-atom unconditional, binder & motif diffusion',
    searchKeywords: [
      'boltzgen',
      'boltzgen1',
      'all-atom',
      'binder',
      'target',
      'ppi',
      'motif',
      'diverse',
      'adherence',
      'diffusion',
    ],
    buildSubGroups: () => [
      {
        id: 'bg-binder',
        label: 'BoltzGen-1 · Binder Design (Diverse)',
        description: 'All-atom de novo binder generation against a connected target PDB',
        paramsPatch: { taskMode: 'binder', variant: 'diverse', precision: 'int8' },
        nodeLabel: 'BoltzGen (Binder)',
        approxMB: 482,
      },
      {
        id: 'bg-unconditional',
        label: 'BoltzGen-1 · Unconditional De Novo',
        description: 'All-atom de novo monomer generation (boltzgen1_diverse)',
        paramsPatch: { taskMode: 'unconditional', variant: 'diverse', precision: 'int8' },
        nodeLabel: 'BoltzGen-1',
        approxMB: 482,
      },
      {
        id: 'bg-motif',
        label: 'BoltzGen-1 · Motif Scaffolding (Adherence)',
        description: 'High-adherence all-atom motif scaffolding inside a de novo scaffold',
        paramsPatch: { taskMode: 'motif', variant: 'adherence', precision: 'int8' },
        nodeLabel: 'BoltzGen (Motif)',
        approxMB: 482,
      },
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('boltzgen', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const tm = String(params.taskMode ?? 'unconditional');
      const variant = String(params.variant ?? 'diverse');
      const len = Number(params.length ?? 48);
      const steps = Number(params.numSteps ?? 200);
      const n = Number(params.numDesigns ?? 1);
      const nTag = n > 1 ? ` · ×${n}` : '';
      if (tm === 'binder') return `Binder (${variant}) · ${len} aa · ${steps} steps${nTag}`;
      if (tm === 'motif') return `Motif (${variant}) · ${len} aa · ${steps} steps${nTag}`;
      return `${variant} · ${len} aa · ${steps} steps${nTag}`;
    },
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'int8').toLowerCase();
    const approxMB = manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB;
    const estimatedVramMB = precision === 'fp16' ? 2200 : manifest.estimatedVramMB;
    const numDesigns = Math.max(1, Math.min(4, Math.round(Number(ctx.params.numDesigns ?? 1) || 1)));
    return {
      estimatedVramMB,
      estimatedDownloadMB: approxMB,
      workItems: numDesigns,
      outputCardinality: { structure: numDesigns, trajectory: 1 },
      stageBreakdown: { generate: numDesigns },
    };
  },
};
