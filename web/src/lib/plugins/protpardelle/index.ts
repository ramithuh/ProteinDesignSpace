// Layer 3 Native ModelPlugin — protpardelle (ProtPardelle-1c all-atom & backbone EDM diffusion)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.protpardelle as unknown as KernelAdapterManifest;

export const protpardellePlugin: ModelPlugin = {
  id: 'protpardelle',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./protpardelle.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['protpardelle-v1'],
    opfsDirs: ['protpardelle-v1'],
  },
  share: {
    shortCode: 'pp',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'All-atom & backbone EDM diffusion, motif scaffolding & MiniMPNN co-design',
    searchKeywords: [
      'protpardelle',
      '1c',
      'cc91',
      'cc91_tip',
      'cc58',
      'cc94',
      'all-atom',
      'motif',
      'partial',
      'diffusion',
      'minimpnn',
      'co-design',
    ],
    buildSubGroups: () => [
      {
        id: 'cc91_epoch383',
        label: 'ProtPardelle-1c · All-Atom (cc91)',
        description: 'All-atom monomer diffusion, motif scaffolding & MiniMPNN co-design',
        paramsPatch: { checkpoint: 'cc91_epoch383', sidechainMode: true },
        nodeLabel: 'ProtPardelle (All-Atom)',
        approxMB: 26,
      },
      {
        id: 'cc91_tip_epoch480',
        label: 'ProtPardelle-1c · Tip-Atom Motif (cc91_tip)',
        description: 'Sidechain-tip atom motif scaffolding & enzyme active-site framing',
        paramsPatch: { checkpoint: 'cc91_tip_epoch480', taskMode: 'motif', sidechainMode: true },
        nodeLabel: 'ProtPardelle (Tip Motif)',
        approxMB: 26,
      },
      {
        id: 'cc58_epoch416',
        label: 'ProtPardelle-1c · Backbone (cc58)',
        description: 'Fast backbone diffusion & crop-conditioned motif scaffolding',
        paramsPatch: { checkpoint: 'cc58_epoch416', sidechainMode: false },
        nodeLabel: 'ProtPardelle (Backbone)',
        approxMB: 26,
      },
      {
        id: 'cc94_epoch3100',
        label: 'ProtPardelle-1c · Multichain All-Atom (cc94)',
        description: 'Multichain all-atom complex generation & conditioning',
        paramsPatch: { checkpoint: 'cc94_epoch3100', sidechainMode: true },
        nodeLabel: 'ProtPardelle (Multichain)',
        approxMB: 26,
      },
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('protpardelle', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const ckpt = String(params.checkpoint ?? 'cc91_epoch383').replace(/_epoch\d+$/, '');
      const mode = String(params.taskMode ?? 'unconditional');
      const steps = Number(params.numSteps ?? 200);
      if (mode === 'motif') return `${ckpt} · Motif · ${steps} steps`;
      if (mode === 'partial') return `${ckpt} · Partial (${Number(params.rewindSteps ?? 40)})`;
      return `${ckpt} · ${Number(params.length ?? 80)} aa · ${steps} steps`;
    },
  },
  preflight: (ctx) => {
    const checkpoint = String(ctx.params.checkpoint ?? 'cc91_epoch383');
    const precision = String(ctx.params.precision ?? 'int8');
    const sidechain = Boolean(ctx.params.sidechainMode ?? true);
    const isBb = checkpoint.startsWith('bb');
    const baseMB = isBb ? (precision === 'fp16' ? 71 : 36) : precision === 'fp16' ? 51 : 26;
    const mpnnMB =
      sidechain && !isBb && checkpoint !== 'cc58_epoch416' && checkpoint !== 'cc95_epoch3490'
        ? precision === 'fp16'
          ? 57
          : 29
        : 0;
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numDesigns ?? 1) || 1)));
    return {
      estimatedVramMB: manifest.estimatedVramMB,
      estimatedDownloadMB: baseMB + mpnnMB,
      workItems: numDesigns,
      outputCardinality: { structure: numDesigns, sequences: numDesigns },
      stageBreakdown: { generate: numDesigns },
    };
  },
};
