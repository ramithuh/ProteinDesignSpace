// Layer 3 Native ModelPlugin — ligandmpnn (Ligand-, metal- & nucleic-acid-aware inverse folding)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.ligandmpnn as unknown as KernelAdapterManifest;

export const ligandmpnnPlugin: ModelPlugin = {
  id: 'ligandmpnn',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: {
      kind: 'cpu-js',
      moduleUrl: '/vendor/ligandmpnn/cpu/index.js',
      degradedWarning: 'Running LigandMPNN on CPU JS fallback (slower than WebGPU).',
    },
    createWorker: async () => new (await import('./ligandmpnn.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['ligandmpnn-v1'],
    usesSharedOpfsStore: true,
  },
  share: {
    shortCode: 'lm',
  },
  catalog: {
    categoryId: 'inverse-fold',
    conciseRole: 'Ligand-, metal- & nucleic-acid-aware protein sequence design',
    searchKeywords: [
      'ligandmpnn',
      'ligand',
      'metal',
      'cofactor',
      'nucleic acid',
      'holo',
      'inverse folding',
      'sequence',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('ligandmpnn', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const ckptShort = String(params.checkpoint ?? 'ligandmpnn_v_32_020_25').replace(
        /^ligandmpnn_/,
        '',
      );
      const n = Number(params.numSequences ?? 2);
      const t = Number(params.temperature ?? 0.1);
      const useLig = params.useLigandContext !== false;
      return `${ckptShort} · ${n} seq${n === 1 ? '' : 's'} · T=${t} · ${useLig ? 'holo' : 'apo'}`;
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
      outputCardinality: { sequences: total },
      stageBreakdown: { 'inverse-fold': total },
    };
  },
};
