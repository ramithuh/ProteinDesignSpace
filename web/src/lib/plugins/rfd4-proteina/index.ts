// Layer 3 Native ModelPlugin — rfd4-proteina (RFD4-Proteina All-Atom Flow Matching)

import { ADAPTER_REGISTRY, RFD4_BENCHMARK_OPTIONS } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY['rfd4-proteina'] as unknown as KernelAdapterManifest;

function numOr(val: unknown, fallback: number): number {
  if (val === undefined || val === null || typeof val === 'boolean') return fallback;
  if (typeof val === 'string' && val.trim() === '') return fallback;
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

export const rfd4ProteinaPlugin: ModelPlugin = {
  id: 'rfd4-proteina',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./rfd4.worker?worker')).default(),
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStoragePrefixes: ['rfd4p-'],
  },
  share: {
    shortCode: 'r',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole: 'All-atom backbone and sequence co-design',
    searchKeywords: [
      'rfd4',
      'rfdiffusion4',
      'proteina',
      'all-atom',
      'flow matching',
      'co-design',
      'binder',
      'motif',
      'cath',
      'partial diffusion',
      'inverse folding',
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('rfd4-proteina', params) as unknown as KernelModelCardSpec,
  },
  display: {
    isSelfGenerator: true,
    formatCanvasSummary: (params) => {
      const mode = String(params.mode ?? 'unconditional');
      const steps = numOr(params.nsteps, 60);
      if (mode === 'binder') {
        const hotCount = String(params.hotspots ?? '').split(/[,\s]+/).filter(Boolean).length;
        const hotTag = hotCount > 0 ? ` · ${hotCount} hotspots` : '';
        return `Binder${hotTag} · ${steps} steps`;
      }
      if (mode === 'motif') {
        const tag = params.unindexedMotif ? 'Unindexed Motif' : 'Motif';
        const motif = String(params.motifSpan ?? '').trim();
        return `${tag}${motif ? ` (${motif})` : ''} · ${steps} steps`;
      }
      if (mode === 'conditioned') {
        const contig = String(params.contig ?? '').trim();
        return `Conditioned${contig ? ` · ${contig}` : ''} · ${steps} steps`;
      }
      if (mode === 'preset') {
        const presetId = String(params.presetId ?? '');
        const opt = RFD4_BENCHMARK_OPTIONS.find((o) => o.id === presetId);
        const presetLabel =
          opt?.label ?? (presetId === 'yaml_5o45_binder' ? '5O45 PD-L1 Binder (YAML)' : 'Preset task');
        return `${presetLabel} · ${steps} steps`;
      }
      if (mode === 'partial') return `Partial (${numOr(params.partialFraction, 0.5)}) · ${steps} steps`;
      if (mode === 'inverse-folding') return `Inverse folding · ${steps} steps`;
      const cath = String(params.cathCode ?? '').trim();
      return `${cath ? `CATH ${cath} · ` : ''}${numOr(params.length, 70)} aa · ${steps} steps`;
    },
  },
  preflight: (ctx) => {
    const mode = String(ctx.params.mode ?? 'unconditional');
    const presetId = String(ctx.params.presetId ?? '');
    const presetMeta = RFD4_BENCHMARK_OPTIONS.find((o) => o.id === presetId);
    const needsEncoder =
      mode === 'partial' ||
      mode === 'inverse-folding' ||
      (mode === 'preset' && Boolean(presetMeta?.needsEncoder));
    const n = Math.max(1, Math.min(16, Math.round(numOr(ctx.params.numDesigns, 1))));
    return {
      estimatedVramMB: manifest.estimatedVramMB + (needsEncoder ? 250 : 0),
      estimatedDownloadMB: manifest.approxWeightsMB + (needsEncoder ? 138 : 0),
      workItems: n,
      outputCardinality: { structure: n, sequences: n },
      stageBreakdown: { generate: n },
    };
  },
};
