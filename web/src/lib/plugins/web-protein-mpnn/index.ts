// Layer 3 Native ModelPlugin — web-protein-mpnn (SolubleMPNN / ProteinMPNN / Cα-ProteinMPNN)

import { ADAPTER_REGISTRY, MPNN_VARIANT_OPTIONS } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY['web-protein-mpnn'] as unknown as KernelAdapterManifest;

export const webProteinMpnnPlugin: ModelPlugin = {
  id: 'web-protein-mpnn',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'ort-webgpu',
    fallback: { kind: 'ort-wasm', wasmAssetDir: '/vendor/ort/' },
    createWorker: async () => new (await import('./mpnn.worker?worker')).default(),
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    usesSharedOpfsStore: true,
    opfsDirs: ['denovo-weights'],
    cacheStoragePrefixes: ['denovo-run'],
  },
  share: {
    shortCode: 'm',
  },
  catalog: {
    categoryId: 'inverse-fold',
    conciseRole: 'Inverse folding (SolubleMPNN, ProteinMPNN, Cα-only)',
    searchKeywords: ['mpnn', 'proteinmpnn', 'solublempnn', 'ca-mpnn', 'inverse folding', 'sequence'],
    buildSubGroups: () =>
      MPNN_VARIANT_OPTIONS.map((v) => ({
        id: v.id,
        label: v.label,
        description: v.description,
        paramsPatch: { family: v.family, modelName: v.modelName },
        nodeLabel: v.shortLabel,
        approxMB: v.approxMB,
      })),
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('web-protein-mpnn', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const fam = String(params.family ?? 'solublempnn');
      const v = MPNN_VARIANT_OPTIONS.find((m) => m.family === fam);
      const n = Number(params.numSequences ?? 2);
      const t = Number(params.temperature ?? 0.1);
      return `${v?.shortLabel ?? 'ProteinMPNN'} · ${n} seq · T=${t}`;
    },
  },
  preflight: (ctx) => {
    const upBackbones = ctx.upstreamProfiles.structure?.itemCount ?? 1;
    const k = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numSequences ?? 2) || 2)));
    const total = upBackbones * k;
    return {
      estimatedVramMB: manifest.estimatedVramMB,
      estimatedDownloadMB: manifest.approxWeightsMB,
      workItems: total,
      outputCardinality: { sequences: total },
      stageBreakdown: { 'inverse-fold': total },
    };
  },
};
