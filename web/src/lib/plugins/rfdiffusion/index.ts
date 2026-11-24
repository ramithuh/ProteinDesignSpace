// Layer 3 Native ModelPlugin — rfdiffusion (Original RFdiffusion Base SE(3) backbone diffusion)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.rfdiffusion as unknown as KernelAdapterManifest;

export const rfdiffusionPlugin: ModelPlugin = {
  id: 'rfdiffusion',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./rfdiffusion.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['rfdiffusion-v1'],
    opfsDirs: ['rfdiffusion-v1'],
  },
  share: {
    shortCode: 'd',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'Original RFdiffusion backbone diffusion, motif contigs & Cn/Dn symmetry',
    searchKeywords: [
      'rfdiffusion',
      'rfd',
      'rfd1',
      'base',
      'backbone',
      'se3',
      'diffusion',
      'motif',
      'contig',
      'symmetry',
      'oligomer',
    ],
    buildSubGroups: () => [
      {
        id: 'rfd-unconditional',
        label: 'RFdiffusion · Unconditional De Novo',
        description: 'Original rigid-frame backbone diffusion (Base_ckpt.pt · ~69 MB)',
        paramsPatch: { taskMode: 'unconditional', precision: 'int8' },
        nodeLabel: 'RFdiffusion',
        approxMB: 69,
      },
      {
        id: 'rfd-motif',
        label: 'RFdiffusion · Motif Scaffolding (Contig)',
        description: 'Contig-driven motif scaffolding from a connected template PDB',
        paramsPatch: { taskMode: 'motif', precision: 'int8' },
        nodeLabel: 'RFdiffusion (Motif)',
        approxMB: 69,
      },
      {
        id: 'rfd-symmetry',
        label: 'RFdiffusion · Symmetric Oligomer (C3)',
        description: 'Cyclic / dihedral symmetric assembly generation (C2–C6, D2–D3)',
        paramsPatch: { taskMode: 'symmetry', symmetry: 'c3', length: 90, precision: 'int8' },
        nodeLabel: 'RFdiffusion (C3)',
        approxMB: 69,
      },
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('rfdiffusion', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const tm = String(params.taskMode ?? 'unconditional');
      const len = Number(params.length ?? 80);
      const steps = Number(params.numSteps ?? 50);
      const prec = String(params.precision ?? 'int8').toUpperCase();
      const n = Number(params.numDesigns ?? 1);
      const nTag = n > 1 ? ` · ×${n}` : '';
      if (tm === 'motif') {
        const contig = String(params.contig ?? '').trim();
        return `Motif${contig ? ` (${contig})` : ''} · ${steps} steps${nTag}`;
      }
      if (tm === 'symmetry') {
        const sym = String(params.symmetry ?? 'c3').toUpperCase();
        return `${sym} · ${len} aa · ${steps} steps${nTag}`;
      }
      return `${len} aa · ${steps} steps · ${prec}${nTag}`;
    },
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'int8').toLowerCase();
    const approxMB = manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB;
    const estimatedVramMB = precision === 'fp16' ? 650 : manifest.estimatedVramMB;
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numDesigns ?? 1) || 1)));
    return {
      estimatedVramMB,
      estimatedDownloadMB: approxMB,
      workItems: numDesigns,
      outputCardinality: { structure: numDesigns, trajectory: 1 },
      stageBreakdown: { generate: numDesigns },
    };
  },
};
