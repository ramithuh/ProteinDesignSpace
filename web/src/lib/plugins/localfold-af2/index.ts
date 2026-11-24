// Layer 3 Native ModelPlugin — localfold-af2 (AlphaFold2, AF2-Multimer, RF3, Boltz-2, AF3, ESMFold2)

import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY['localfold-af2'] as unknown as KernelAdapterManifest;

function numOr(val: unknown, fallback: number): number {
  if (val === undefined || val === null || typeof val === 'boolean') return fallback;
  if (typeof val === 'string' && val.trim() === '') return fallback;
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

export const localfoldAf2Plugin: ModelPlugin = {
  id: 'localfold-af2',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./localfold.worker?worker')).default(),
    vramLeaseMode: 'exclusive',
    getWorkerIsolationKey: (params) =>
      LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === params.model && m.family === 'esmfold2')?.id,
  },
  cachePolicy: {
    cacheStoragePrefixes: ['localfold-'],
  },
  share: {
    shortCode: 'f',
    yamlKeyAliases: { model: 'foldModel' },
  },
  catalog: {
    categoryId: 'fold',
    conciseRole: 'Predict 3D structures (AlphaFold2, RF3, Boltz-2, AlphaFold 3 & more)',
    searchKeywords: [
      'localfold',
      'alphafold',
      'alphafold2',
      'af2',
      'multimer',
      'af3',
      'alphafold3',
      'rosettafold3',
      'rf3',
      'boltz2',
      'openbind0',
      'protenix2',
      'opendde',
      'mol1',
      'intellifold2',
      'esmfold2',
      'fold',
    ],
    buildSubGroups: () =>
      LOCALFOLD_MODEL_OPTIONS.map((opt) => ({
        id: opt.id,
        label: opt.label,
        paramsPatch:
          opt.family === 'esmfold2' ? { model: opt.id, msaMode: 'single' } : { model: opt.id },
        nodeLabel: opt.shortLabel,
        approxMB: opt.approxMB,
        ...(opt.nonCommercial ? { badge: 'Non-Commercial' } : {}),
      })),
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('localfold-af2', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const modelId = String(params.model ?? 'monomer');
      const meta =
        LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
      const rec = numOr(params.recycles, 1);
      const msaMode = String(params.msaMode ?? 'single');
      const msaTag =
        msaMode === 'mmseqs2'
          ? 'MMseqs2'
          : msaMode === 'target-only'
            ? 'target MSA'
            : msaMode === 'a3m'
              ? '.a3m'
              : 'single-seq';
      if (meta.family === 'esmfold2') {
        return `${meta.shortLabel} · single-seq · ${numOr(params.esmfold2Steps, 15)} scheduled steps`;
      }
      if (meta.family === 'af3') {
        const steps = numOr(params.diffusionSteps, 25);
        return `LocalFold · ${msaTag} · ${steps} steps · ${rec} rec`;
      }
      return `LocalFold · ${msaTag} · ${rec} rec`;
    },
  },
  preflight: (ctx) => {
    const modelId = String(ctx.params.model ?? 'monomer');
    const meta =
      LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
    const upSeqs = ctx.upstreamProfiles.sequences?.itemCount ?? 1;
    const isFollowUpstream =
      String(ctx.params.capMode ?? 'follow-upstream') === 'follow-upstream';
    const maxSeqs = isFollowUpstream
      ? 8
      : Math.max(1, Math.min(8, Math.round(numOr(ctx.params.maxSequences, 2))));
    const maxTotalFolds = Math.max(1, Math.min(64, Math.round(numOr(ctx.params.maxTotalFolds, 16))));
    const workItems = Math.max(1, Math.min(upSeqs, maxTotalFolds, Math.max(maxSeqs, 1)));
    const msaMode = String(ctx.params.msaMode ?? 'single');
    const usesColabFold =
      meta.family !== 'esmfold2' && (msaMode === 'mmseqs2' || msaMode === 'target-only');
    return {
      estimatedVramMB: manifest.estimatedVramMB,
      estimatedDownloadMB: meta.approxMB,
      workItems,
      outputCardinality: { folds: workItems, structure: workItems },
      stageBreakdown: { fold: workItems },
      ...(usesColabFold
        ? {
            remoteTransfers: [
              {
                destination: 'api.colabfold.com',
                purpose: 'MMseqs2 evolutionary MSA search',
                mode: msaMode,
              },
            ],
          }
        : {}),
    };
  },
  inspector: {
    customWidgetId: 'localfold-checkpoint-msa-holo',
  },
};
