// Layer 3 Native ModelPlugin — evodiff (EvoDiff-Seq OA-DM 38M)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.evodiff as unknown as KernelAdapterManifest;

export const evodiffPlugin: ModelPlugin = {
  id: 'evodiff',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: {
      kind: 'cpu-js',
      moduleUrl: '/vendor/evodiff/cpu/index.js',
      degradedWarning: 'Running EvoDiff-38M on pure-JS CPU fallback (slower than WebGPU).',
    },
    createWorker: async () => new (await import('./evodiff.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['evodiff-v1'],
    opfsDirs: ['evodiff-v1'],
  },
  share: {
    shortCode: 'e',
  },
  catalog: {
    categoryId: 'inverse-fold',
    conciseRole: 'MSA-free discrete sequence diffusion & masked sequence inpainting (OA-DM 38M)',
    searchKeywords: [
      'evodiff',
      'oadm',
      '38m',
      'sequence',
      'diffusion',
      'infill',
      'inpainting',
      'msa-free',
      'de novo',
    ],
    buildSubGroups: () => [
      {
        id: 'evodiff-unconditional',
        label: 'EvoDiff-Seq · Unconditional De Novo (38M)',
        description: 'MSA-free order-agnostic discrete sequence diffusion (~37 MB)',
        paramsPatch: { taskMode: 'unconditional', precision: 'int8' },
        nodeLabel: 'EvoDiff-38M',
        approxMB: 37,
      },
      {
        id: 'evodiff-infill',
        label: 'EvoDiff-Seq · Masked Sequence Inpainting',
        description: 'Preserve sequence motif residues and diffuse # / masked scaffold spans',
        paramsPatch: { taskMode: 'infill', precision: 'int8' },
        nodeLabel: 'EvoDiff (Infill)',
        approxMB: 37,
      },
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('evodiff', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const tm = String(params.taskMode ?? 'unconditional');
      const len = Number(params.length ?? 64);
      const n = Number(params.numSequences ?? 2);
      const prec = String(params.precision ?? 'int8').toUpperCase();
      if (tm === 'infill') return `Infill · ${n} seq${n === 1 ? '' : 's'} · ${prec}`;
      return `${len} aa · ${n} seq${n === 1 ? '' : 's'} · ${prec}`;
    },
  },
  preflight: (ctx) => {
    const prec = String(ctx.params.precision ?? 'int8').toLowerCase() === 'fp16' ? 'fp16' : 'int8';
    const n = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numSequences ?? 2) || 2)));
    return {
      estimatedVramMB: prec === 'fp16' ? 380 : manifest.estimatedVramMB,
      estimatedDownloadMB: manifest.precisionWeightsMB?.[prec] ?? manifest.approxWeightsMB,
      workItems: n,
      outputCardinality: { sequences: n },
      stageBreakdown: { 'inverse-fold': n },
      precisionOptions: {
        int8: { weightsMB: 37, vramMB: 180 },
        fp16: { weightsMB: 73, vramMB: 380 },
      },
    };
  },
};
