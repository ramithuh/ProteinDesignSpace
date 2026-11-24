// Layer 3 Native ModelPlugin — esmc (ESM-C 300M Zero-Shot LLR)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.esmc as unknown as KernelAdapterManifest;

export const esmcPlugin: ModelPlugin = {
  id: 'esmc',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./esmc.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['esmc-v1'],
    opfsDirs: ['esmc-v1'],
  },
  share: {
    shortCode: 'ec',
  },
  catalog: {
    categoryId: 'score',
    conciseRole: 'ESM-C 300M zero-shot L×20 LLR fitness heatmap, naturalness & masked fill',
    searchKeywords: [
      'esmc',
      'esm-c',
      'esm',
      'cambrian',
      '300m',
      'llr',
      'zero-shot',
      'fitness',
      'naturalness',
      'pseudo-perplexity',
      'pppl',
      'plm',
    ],
    buildSubGroups: () => [
      {
        id: 'esmc-wildtype',
        label: 'ESM-C 300M · Wild-Type LLR (Fast)',
        description: '1-pass L×20 zero-shot fitness heatmap & WT log-p track (~40 ms)',
        paramsPatch: { mode: 'wildtype', precision: 'int8' },
        nodeLabel: 'ESM-C 300M (WT LLR)',
        approxMB: 335,
      },
      {
        id: 'esmc-masked',
        label: 'ESM-C 300M · Masked-Marginal & PPPL',
        description: 'Full L-pass masked-marginal LLR matrix & pseudo-perplexity naturalness',
        paramsPatch: { mode: 'masked', precision: 'int8' },
        nodeLabel: 'ESM-C 300M (Masked)',
        approxMB: 335,
      },
      {
        id: 'esmc-fill',
        label: 'ESM-C 300M · Masked Sequence Fill',
        description: 'Mask selected residue positions and predict top-1 amino acids',
        paramsPatch: { mode: 'fill', precision: 'int8' },
        nodeLabel: 'ESM-C 300M (Fill)',
        approxMB: 335,
      },
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('esmc', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const mode = String(params.mode ?? 'wildtype');
      const prec = String(params.precision ?? 'int8').toUpperCase();
      const tag = mode === 'masked' ? 'Masked LLR' : mode === 'fill' ? 'Fill' : 'WT LLR';
      return `${tag} · ${prec}`;
    },
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'int8').toLowerCase();
    const approxMB = manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB;
    const estimatedVramMB = precision === 'fp16' ? 1150 : manifest.estimatedVramMB;
    const maxSeqs = Math.max(1, Math.min(8, Math.round(Number(ctx.params.maxSequences ?? 1) || 1)));
    const rawCount =
      ctx.upstreamProfiles.sequences?.itemCount ??
      ctx.upstreamProfiles.structure?.itemCount ??
      1;
    const items = Math.max(1, Math.min(maxSeqs, rawCount));
    return {
      estimatedVramMB,
      estimatedDownloadMB: approxMB,
      workItems: items,
      outputCardinality: {
        mutations: 1,
        track: 1,
        sequences: items,
        ...(ctx.connectedInputs.has('structure') ? { structure: items } : {}),
      },
      stageBreakdown: { score: items },
    };
  },
};
