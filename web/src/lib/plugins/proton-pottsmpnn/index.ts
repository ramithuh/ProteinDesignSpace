// Layer 3 Native ModelPlugin — proton-pottsmpnn (Proton-PottsMPNN pH-aware Potts sequence
// designer & scorer; christian-creator/ProtonPottsMPNN, MIT + BSD-3-Clause rc-foundry PottsMPNN)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';
import { PROTON_POTTSMPNN_MANIFEST } from './manifest';

// Prefers the shared registry entry; the local literal keeps the plugin, its worker and its
// tests self-contained. `proton-pottsmpnn.test.ts` asserts the two are identical once wired.
const manifest = ((ADAPTER_REGISTRY as Record<string, unknown>)['proton-pottsmpnn'] ??
  PROTON_POTTSMPNN_MANIFEST) as unknown as KernelAdapterManifest;

export { PROTON_POTTSMPNN_MANIFEST } from './manifest';

export const protonPottsMpnnPlugin: ModelPlugin = {
  id: 'proton-pottsmpnn',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: {
      kind: 'cpu-js',
      moduleUrl: '/vendor/proton-pottsmpnn/cpu/index.js',
      degradedWarning:
        'Running Proton-PottsMPNN on the CPU JS fallback (the MPNN forward pass is ~230x slower than WebGPU).',
    },
    createWorker: async () => new (await import('./proton-pottsmpnn.worker?worker')).default(),
    supportsCooperativeAbort: true,
    // Block descent only yields a cancel check when it commits a move, so allow a longer grace.
    abortGraceMs: 3000,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    // Weight bundles stream through makeCachedFetch into the shared denovo-run store.
    usesSharedOpfsStore: true,
  },
  share: {
    shortCode: 'pm',
  },
  catalog: {
    categoryId: 'sequence-design',
    conciseRole:
      'pH-aware Potts sequence design & scoring over explicit protonation states (pH-selective block descent, E_P − E_D)',
    searchKeywords: [
      'proton',
      'protonpotts',
      'protonpottsmpnn',
      'proton-pottsmpnn',
      'pottsmpnn',
      'potts',
      'ph',
      'ph-selective',
      'protonation',
      'titratable',
      'histidine',
      'his',
      'aspartate',
      'glutamate',
      'pka',
      'charge state',
      'microstate',
      'block descent',
      'couplings',
      'energy',
      'sequence design',
      'inverse folding',
      'rc-foundry',
    ],
  },
  planner: {
    plannerRole: 'sequence-designer',
    plannerRoles: ['sequence-designer', 'scorer'],
    keywords: [
      'proton-pottsmpnn',
      'pottsmpnn',
      'ph selective design',
      'ph switch',
      'protonation state',
      'titratable residue',
      'histidine design',
      'potts energy',
    ],
    capabilities: [
      'ph-selective-sequence-design',
      'protonation-state-aware-design',
      'sequence-scoring',
      'potts-energy',
      'fixed-position-design',
      'per-residue-track',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('proton-pottsmpnn', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => formatProtonPottsMpnnSummary(params),
    compareEligible: true,
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'fp16').toLowerCase();
    const backbones = ctx.upstreamProfiles.structure?.itemCount ?? 1;
    const mode = String(ctx.params.mode ?? 'design');
    // Temp 0 block descent is an exact argmin, so extra sequences would be byte-identical copies.
    const temperature = Number(ctx.params.temperature ?? 0);
    const requested = Math.max(1, Math.min(32, Math.round(Number(ctx.params.numSequences ?? 4) || 4)));
    const perBackbone =
      mode === 'score' ? 1 : Number.isFinite(temperature) && temperature > 0 ? requested : 1;
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
export function formatProtonPottsMpnnSummary(params: Record<string, unknown>): string {
  const num = (v: unknown, fallback: number): number => {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  };
  const centres = String(params.protonationCentres ?? '')
    .split(/[,;\s]+/)
    .filter(Boolean).length;
  const precision = String(params.precision ?? 'fp16').toUpperCase();

  if (String(params.mode ?? 'design') === 'score') {
    const parts = ['Score native · Potts energy'];
    parts.push(centres > 0 ? `${centres} pH centre${centres === 1 ? '' : 's'} · ΔE` : 'no pH centre');
    parts.push(precision);
    return parts.join(' · ');
  }

  const temperature = num(params.temperature, 0);
  const requested = num(params.numSequences, 4);
  const n = temperature > 0 ? requested : 1;
  const parts = [`${n} seq${n === 1 ? '' : 's'}`];
  parts.push(centres > 0 ? `λ=${num(params.lambda, 0.5)} · ${centres} pH centre${centres === 1 ? '' : 's'}` : 'stability only');
  parts.push(`B=${num(params.blockSize, 2)}`);
  parts.push(temperature > 0 ? `T=${temperature}` : 'argmin');
  parts.push(precision);
  return parts.join(' · ');
}
