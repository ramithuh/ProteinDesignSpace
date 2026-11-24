// Layer 3 Native ModelPlugin — localfold-af2 (AlphaFold2, AF2-Multimer, RF3, Boltz-2, AF3, ESMFold2)

import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { computeUploadDigest, isValidSha256Hex } from '$lib/engine/param-validation';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY['localfold-af2'] as unknown as KernelAdapterManifest;
const TEMPLATE_UPLOAD_META_KEYS = new Set([
  'missingTemplateUpload',
  'expectedTemplateSha256',
  'expectedTemplateFilename',
]);
const INLINE_TEMPLATE_SHARE_LIMIT = 1500;

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
    prepareForShare: (params, { stripLargeUploads }) => {
      const next: Record<string, unknown> = {};
      let stripped = false;
      let strippedTemplateText: string | null = null;

      for (const [k, v] of Object.entries(params)) {
        if (TEMPLATE_UPLOAD_META_KEYS.has(k)) continue;
        if (
          k === 'templatePdbText' &&
          typeof v === 'string' &&
          v.length > INLINE_TEMPLATE_SHARE_LIMIT &&
          stripLargeUploads
        ) {
          stripped = true;
          strippedTemplateText = v;
          continue;
        }
        next[k] = v;
      }

      const existingShaRaw = String(params.expectedTemplateSha256 ?? '').trim().toLowerCase();
      const existingExpName = String(params.expectedTemplateFilename ?? '').trim();
      const curFilename = String(params.templateFilename ?? '').trim();
      let uploadDigest: string | undefined;
      let expectedFilename: string | undefined;

      if (stripped && stripLargeUploads && strippedTemplateText !== null) {
        delete next.templatePdbText;
        const computedSha = computeUploadDigest(strippedTemplateText);
        const expectedSha = existingShaRaw ? existingShaRaw : computedSha;
        next.missingTemplateUpload = true;
        next.expectedTemplateSha256 = expectedSha;
        uploadDigest = expectedSha;
        const expName = existingExpName || curFilename;
        if (expName) {
          next.expectedTemplateFilename = expName;
          expectedFilename = expName;
        }
      } else {
        const rawTpl = typeof params.templatePdbText === 'string' ? params.templatePdbText : '';
        const hasMissingFlag = Boolean(params.missingTemplateUpload);
        if (!rawTpl.trim() && (hasMissingFlag || existingShaRaw)) {
          stripped = true;
          delete next.templatePdbText;
          if (hasMissingFlag) next.missingTemplateUpload = true;
          if (existingShaRaw) {
            next.expectedTemplateSha256 = existingShaRaw;
            uploadDigest = existingShaRaw;
          }
          const expName = existingExpName || curFilename;
          if (expName) {
            next.expectedTemplateFilename = expName;
            expectedFilename = expName;
          }
        } else if (rawTpl.trim() && existingShaRaw) {
          const actualSha = computeUploadDigest(rawTpl);
          if (!isValidSha256Hex(existingShaRaw) || actualSha !== existingShaRaw) {
            if (hasMissingFlag) next.missingTemplateUpload = true;
            next.expectedTemplateSha256 = existingShaRaw;
            uploadDigest = existingShaRaw;
            if (existingExpName) {
              next.expectedTemplateFilename = existingExpName;
              expectedFilename = existingExpName;
            }
          }
        }
      }

      return {
        params: next,
        stripped,
        ...(uploadDigest ? { uploadDigest } : {}),
        ...(expectedFilename ? { expectedFilename } : {}),
      };
    },
    restoreOmittedInlineAsset: (prevParams, nextParams) => {
      if (String(nextParams.templatePdbText ?? '').trim()) return null;
      const prevTplText = prevParams.templatePdbText;
      if (typeof prevTplText !== 'string' || prevTplText.length <= INLINE_TEMPLATE_SHARE_LIMIT) {
        return null;
      }
      const hasMissingMarker =
        Boolean(nextParams.missingTemplateUpload) ||
        Boolean(String(nextParams.expectedTemplateSha256 ?? '').trim());
      if (!hasMissingMarker) return null;

      const expectedSha = String(nextParams.expectedTemplateSha256 ?? '')
        .trim()
        .toLowerCase();
      if (expectedSha) {
        if (!isValidSha256Hex(expectedSha)) return null;
        if (computeUploadDigest(prevTplText) !== expectedSha) return null;
      }

      const restored: Record<string, unknown> = {
        ...nextParams,
        templatePdbText: prevTplText,
      };
      const restoredFilename =
        String(nextParams.templateFilename ?? '').trim() ||
        String(nextParams.expectedTemplateFilename ?? '').trim() ||
        String(prevParams.templateFilename ?? '').trim();
      if (restoredFilename) {
        restored.templateFilename = restoredFilename;
      }
      delete restored.missingTemplateUpload;
      delete restored.expectedTemplateSha256;
      delete restored.expectedTemplateFilename;
      return restored;
    },
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
