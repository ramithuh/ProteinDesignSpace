// Layer 3 Native ModelPlugin — dyna1 (Dyna-1 Conformational Dynamics)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.dyna1 as unknown as KernelAdapterManifest;

export const dyna1Plugin: ModelPlugin = {
  id: 'dyna1',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./dyna1.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['dyna1-v1'],
    opfsDirs: ['dyna1-v1'],
  },
  share: {
    shortCode: 'dy',
  },
  catalog: {
    categoryId: 'score',
    conciseRole: 'Millisecond conformational exchange & flexibility prediction (ESM-3 + Dyna-1)',
    searchKeywords: [
      'dyna1',
      'dyna-1',
      'dynamics',
      'flexibility',
      'nmr',
      'exchange',
      'esm3',
      'esm-3',
      'conformational',
      'rigid',
    ],
    buildSubGroups: () => [
      {
        id: 'dyna1-standard',
        label: 'Dyna-1 · Standard (Recommended)',
        description: 'Millisecond conformational exchange probability (0..1 · fast local profile)',
        paramsPatch: { precision: 'int8' },
        nodeLabel: 'Dyna-1 (Dynamics)',
        approxMB: 844,
      },
      {
        id: 'dyna1-high-precision',
        label: 'Dyna-1 · High Precision',
        description: 'Higher-precision exchange probability profile (for high-memory GPUs)',
        paramsPatch: { precision: 'fp16' },
        nodeLabel: 'Dyna-1 (High Precision)',
        approxMB: 1745,
      },
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('dyna1', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const prec = String(params.precision ?? 'int8').toUpperCase();
      const mode = String(params.inputMode ?? 'auto');
      const modeLbl =
        mode === 'seq+struct'
          ? 'Seq+Struct'
          : mode === 'seq-only'
            ? 'Seq-only'
            : mode === 'struct-only'
              ? 'Struct-only'
              : 'Auto (Seq+Struct)';
      return `${modeLbl} · ${prec}`;
    },
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'int8').toLowerCase();
    const inputMode = String(ctx.params.inputMode ?? 'auto');
    const needsStructEnc = ctx.connectedInputs.has('structure') && inputMode !== 'seq-only';
    const baseMB = manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB;
    const approxMB = baseMB + (needsStructEnc ? 88 : 0);
    const estimatedVramMB =
      (precision === 'fp32' ? 3900 : precision === 'fp16' ? 2200 : manifest.estimatedVramMB) +
      (needsStructEnc ? 140 : 0);
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
        track: 1,
        sequences: items,
        ...(ctx.connectedInputs.has('structure') ? { structure: items } : {}),
      },
      stageBreakdown: { score: items },
    };
  },
};
