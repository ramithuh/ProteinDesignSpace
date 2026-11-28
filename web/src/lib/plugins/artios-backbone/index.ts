// Layer 3 Native ModelPlugin — artios-backbone (ArtiosBackbone all-atom EDM diffusion · Ligo Biosciences)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';
import { ARTIOS_BACKBONE_MANIFEST, ARTIOS_BACKBONE_WEIGHTS_MB } from './manifest';

// `registry.ts` re-exports `ARTIOS_BACKBONE_MANIFEST` under this key, so the lookup and the local
// definition are the same object; the fallback keeps the plugin loadable on its own (unit tests).
const manifest = (ADAPTER_REGISTRY['artios-backbone'] ??
  ARTIOS_BACKBONE_MANIFEST) as unknown as KernelAdapterManifest;

function numOr(val: unknown, fallback: number): number {
  if (val === undefined || val === null || typeof val === 'boolean') return fallback;
  if (typeof val === 'string' && val.trim() === '') return fallback;
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

function normalizePrecisionId(raw: unknown): 'int8' | 'fp16' | 'fp32' {
  const val = String(raw ?? 'int8').trim().toLowerCase();
  return val === 'fp16' || val === 'fp32' ? val : 'int8';
}

/** Designs batched per GPU submission (mirrors `gpuBatchLimit` in the worker). */
function batchLimit(nToken: number): number {
  if (nToken <= 128) return 8;
  if (nToken <= 256) return 4;
  return 2;
}

/**
 * Peak VRAM estimate: weight bundle + the per-conditioning pair caches (`z` and the per-block
 * pair bias, both O(tokens²)) + per-sample activations, which is what the WGSL engine allocates.
 */
export function estimateArtiosBackboneVramMB(
  length: number,
  numDesigns: number,
  precision: 'int8' | 'fp16' | 'fp32',
): number {
  const nToken = Math.max(1, Math.round(length));
  const nAtom = nToken * 4;
  const MB = 2 ** 20;
  const pairMB = (nToken * nToken * 128 * 4) / MB;
  const biasMB = (16 * 16 * nToken * nToken * 4) / MB;
  const perSampleMB = ((nAtom * 256 * 6 + nToken * 768 * 10) * 4) / MB;
  const batch = Math.min(Math.max(1, Math.round(numDesigns)), batchLimit(nToken));
  return Math.ceil(
    ARTIOS_BACKBONE_WEIGHTS_MB[precision] + pairMB + biasMB + batch * perSampleMB + 48,
  );
}

export const artiosBackbonePlugin: ModelPlugin = {
  id: 'artios-backbone',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: {
      kind: 'cpu-js',
      moduleUrl: '/vendor/artios-backbone/cpu/index.js',
      degradedWarning:
        'Running ArtiosBackbone on the CPU JS denoiser (far slower than WebGPU; use few steps and one design).',
    },
    createWorker: async () => new (await import('./artios-backbone.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
    maxBatchItems: 16,
  },
  cachePolicy: {
    // Shards stream through makeCachedFetch into the shared denovo-run store.
    usesSharedOpfsStore: true,
  },
  share: {
    shortCode: 'ab',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole:
      'All-atom EDM diffusion (136.7M) for de novo backbones, motif scaffolding & ligand-conditioned design',
    searchKeywords: [
      'artios',
      'artios-backbone',
      'artiosbackbone',
      'ligo',
      'ligo biosciences',
      'all-atom',
      'edm',
      'diffusion',
      'dit',
      'motif',
      'scaffolding',
      'ligand',
      'enzyme',
      'rasa',
      'burial',
      'de novo',
      'backbone',
      'karras',
      'self-conditioning',
      'apache',
    ],
  },
  planner: {
    plannerRole: 'generator',
    keywords: [
      'artios',
      'artios backbone',
      'all-atom diffusion',
      'motif scaffolding',
      'ligand scaffolding',
      'enzyme design',
      'edm diffusion',
    ],
    capabilities: [
      'unconditional-backbone-generation',
      'motif-scaffolding',
      'ligand-conditioned-generation',
      'all-atom-generation',
      'multi-design-batching',
      'burial-depth-conditioning',
      'ligand-rasa-conditioning',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('artios-backbone', params) as unknown as KernelModelCardSpec,
  },
  display: {
    isSelfGenerator: true,
    compareEligible: true,
    formatCanvasSummary: (params) => {
      const length = Math.round(numOr(params.length, 100));
      const steps = Math.round(numOr(params.numSteps, 200));
      const precision = normalizePrecisionId(params.precision).toUpperCase();
      const n = Math.round(numOr(params.numDesigns, 1));
      const nTag = n > 1 ? ` · ×${n}` : '';
      const motif = String(params.motifContig ?? '').trim();
      const ligTag = params.includeLigand === false ? ' · no ligand' : '';
      if (motif) {
        return `Motif (${motif}) · ${length} aa · ${steps} steps${ligTag}${nTag}`;
      }
      return `${length} aa · ${steps} steps · ${precision}${ligTag}${nTag}`;
    },
  },
  preflight: (ctx) => {
    const precision = normalizePrecisionId(ctx.params.precision);
    const length = Math.max(30, Math.min(512, Math.round(numOr(ctx.params.length, 100))));
    const n = Math.max(1, Math.min(16, Math.round(numOr(ctx.params.numDesigns, 1))));
    const weightsMB = ARTIOS_BACKBONE_WEIGHTS_MB[precision];
    return {
      estimatedVramMB: estimateArtiosBackboneVramMB(length, n, precision),
      estimatedDownloadMB: weightsMB,
      workItems: n,
      outputCardinality: { structure: n },
      stageBreakdown: { generate: n },
      precisionOptions: {
        int8: {
          weightsMB: ARTIOS_BACKBONE_WEIGHTS_MB.int8,
          vramMB: estimateArtiosBackboneVramMB(length, n, 'int8'),
        },
        fp16: {
          weightsMB: ARTIOS_BACKBONE_WEIGHTS_MB.fp16,
          vramMB: estimateArtiosBackboneVramMB(length, n, 'fp16'),
        },
        fp32: {
          weightsMB: ARTIOS_BACKBONE_WEIGHTS_MB.fp32,
          vramMB: estimateArtiosBackboneVramMB(length, n, 'fp32'),
        },
      },
    };
  },
};
