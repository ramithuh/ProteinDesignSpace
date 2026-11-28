// Layer 3 Native ModelPlugin — artios-potts (ArtiosPotts all-atom ligand-aware Potts
// sequence designer & scorer; Ligo-Biosciences/artios, Apache-2.0)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';
import { ARTIOS_POTTS_MANIFEST } from './manifest';

// Prefers the shared registry entry; the local literal keeps the plugin, its worker and its
// tests self-contained. `artios-potts.test.ts` asserts the two are identical once wired.
const manifest = ((ADAPTER_REGISTRY as Record<string, unknown>)['artios-potts'] ??
  ARTIOS_POTTS_MANIFEST) as unknown as KernelAdapterManifest;

export { ARTIOS_POTTS_MANIFEST } from './manifest';

export const artiosPottsPlugin: ModelPlugin = {
  id: 'artios-potts',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: {
      kind: 'cpu-js',
      moduleUrl: '/vendor/artios-potts/cpu/index.js',
      degradedWarning:
        'Running ArtiosPotts on the CPU JS fallback (the MPNN forward pass is ~100x slower than WebGPU).',
    },
    createWorker: async () => new (await import('./artios-potts.worker?worker')).default(),
    supportsCooperativeAbort: true,
    // One MCMC chain runs to completion between abort checks, so allow a slightly longer grace.
    abortGraceMs: 2500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    // Weight bundles stream through makeCachedFetch into the shared denovo-run store.
    usesSharedOpfsStore: true,
  },
  share: {
    shortCode: 'ap',
  },
  catalog: {
    categoryId: 'sequence-design',
    conciseRole:
      'All-atom ligand-aware Potts sequence design & scoring (energy + per-residue pseudo-log-likelihood)',
    searchKeywords: [
      'artios',
      'artiospotts',
      'potts',
      'ligo',
      'ligo biosciences',
      'mrf',
      'markov random field',
      'gibbs',
      'dlmc',
      'mcmc',
      'couplings',
      'ligand',
      'cofactor',
      'metal',
      'active site',
      'pocket',
      'pseudo-log-likelihood',
      'pll',
      'energy',
      'sequence design',
      'inverse folding',
    ],
  },
  planner: {
    plannerRole: 'sequence-designer',
    plannerRoles: ['sequence-designer', 'scorer'],
    keywords: [
      'artios',
      'artiospotts',
      'potts',
      'ligand-aware design',
      'active site',
      'pocket design',
      'pseudo-log-likelihood',
      'sequence energy',
    ],
    capabilities: [
      'ligand-aware-sequence-design',
      'active-site-design',
      'sequence-scoring',
      'pseudo-log-likelihood',
      'potts-energy',
      'fixed-position-design',
      'per-residue-track',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('artios-potts', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => formatArtiosPottsSummary(params),
    compareEligible: true,
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'fp16').toLowerCase();
    const backbones = ctx.upstreamProfiles.structure?.itemCount ?? 1;
    const mode = String(ctx.params.mode ?? 'design');
    const perBackbone =
      mode === 'score' ? 1 : Math.max(1, Math.min(64, Math.round(Number(ctx.params.numSequences ?? 8) || 8)));
    const total = backbones * perBackbone;
    return {
      estimatedVramMB: manifest.estimatedVramMB,
      estimatedDownloadMB: manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB,
      workItems: total,
      outputCardinality: { sequences: total, track: 1 },
      stageBreakdown: { 'inverse-fold': total },
    };
  },
};

/** Canvas / node summary, shared with `node-display-helpers.ts` via the wiring spec. */
export function formatArtiosPottsSummary(params: Record<string, unknown>): string {
  const num = (v: unknown, fallback: number): number => {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  };
  if (String(params.mode ?? 'design') === 'score') {
    const parts = ['Score native · Energy + PLL'];
    const cutoff = num(params.ligandCutoff, 0);
    if (cutoff > 0) parts.push(`${cutoff} Å pocket`);
    parts.push(String(params.precision ?? 'fp16').toUpperCase());
    return parts.join(' · ');
  }
  const n = num(params.numSequences, 8);
  const t = num(params.temperature, 0.1);
  const sampler = String(params.sampler ?? 'gibbs') === 'dlmc' ? 'DLMC' : 'Gibbs';
  const parts = [`${n} seq${n === 1 ? '' : 's'}`, `T=${t}`, `${sampler} ×${num(params.numSteps, 150)}`];
  const cutoff = num(params.ligandCutoff, 0);
  if (cutoff > 0) parts.push(`≤${cutoff} Å of ligand`);
  const cfg = num(params.cfgScale, 0);
  if (cfg > 0) parts.push(`CFG ${cfg}`);
  parts.push(String(params.precision ?? 'fp16').toUpperCase());
  return parts.join(' · ');
}
