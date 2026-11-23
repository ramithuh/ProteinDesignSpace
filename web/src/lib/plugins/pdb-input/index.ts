// Layer 3 Native ModelPlugin — pdb-input

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { computeUploadDigest, isValidSha256Hex } from '$lib/engine/param-validation';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { ArtifactEnvelope, KernelAdapterManifest, ModelPlugin } from '$lib/kernel';
import type { WorkflowNode } from '$lib/engine/types';
import { materializePdbInputNode, deriveSequenceSetFromStructure, runPdbInputCpu } from './cpu-runner';

export { materializePdbInputNode, deriveSequenceSetFromStructure, runPdbInputCpu };

const UPLOAD_META_KEYS = new Set(['missingUpload', 'expectedPdbSha256', 'expectedFilename']);

export const pdbInputPlugin: ModelPlugin = {
  id: 'pdb-input',
  domainId: PROTEIN_DOMAIN_ID,
  manifest: ADAPTER_REGISTRY['pdb-input'] as unknown as KernelAdapterManifest,
  execution: {
    runtime: 'cpu-ts',
    fallback: { kind: 'none', reason: 'Pure TypeScript CPU adapter' },
    vramLeaseMode: 'exclusive',
    executeCpu: async (ctx) => {
      const node = ctx.node as unknown as WorkflowNode;
      const out = await runPdbInputCpu(node, (ev) => ctx.onProgress(ev));
      return out as unknown as Record<string, ArtifactEnvelope>;
    },
    materializePreview: async (params) => {
      const syntheticNode: WorkflowNode = {
        id: 'pdb-preview',
        adapterId: 'pdb-input',
        params,
      };
      const out = await runPdbInputCpu(syntheticNode);
      return out as unknown as Record<string, ArtifactEnvelope>;
    },
  },
  cachePolicy: {},
  share: {
    shortCode: 'p',
    stripOnShareKeys: ['fetchedPdbId'],
    prepareForShare: (params, { stripLargeUploads }) => {
      const mode = String(params.mode ?? 'demo-bundle');
      const next: Record<string, unknown> = {};
      let stripped = false;
      let strippedPdbText: string | null = null;

      for (const [k, v] of Object.entries(params)) {
        if (k === 'fetchedPdbId') continue;
        if (mode === 'pdb-id' || mode === 'demo-bundle') {
          if (k === 'pdbText' || UPLOAD_META_KEYS.has(k)) continue;
        } else if (mode === 'custom') {
          if (UPLOAD_META_KEYS.has(k)) continue;
          if (k === 'pdbText' && typeof v === 'string' && v.length > 1500 && stripLargeUploads) {
            stripped = true;
            strippedPdbText = v;
            continue;
          }
        }
        next[k] = v;
      }

      let uploadDigest: string | undefined;
      let expectedFilename: string | undefined;

      if (mode === 'custom') {
        const existingShaRaw = String(params.expectedPdbSha256 ?? '').trim().toLowerCase();
        const existingExpName = String(params.expectedFilename ?? '').trim();
        const curFilename = String(params.filename ?? '').trim();

        if (stripped && stripLargeUploads && strippedPdbText !== null) {
          next.mode = 'custom';
          next.pdbText = '';
          const computedSha = computeUploadDigest(strippedPdbText);
          const expectedSha = existingShaRaw ? existingShaRaw : computedSha;
          next.missingUpload = true;
          next.expectedPdbSha256 = expectedSha;
          uploadDigest = expectedSha;
          const expName = existingExpName || curFilename;
          if (expName) {
            next.expectedFilename = expName;
            expectedFilename = expName;
          }
        } else {
          const rawPdb = typeof params.pdbText === 'string' ? params.pdbText : '';
          const hasMissingFlag = Boolean(params.missingUpload);
          if (!rawPdb.trim() && (hasMissingFlag || existingShaRaw)) {
            stripped = true;
            next.mode = 'custom';
            if (hasMissingFlag) next.missingUpload = true;
            if (existingShaRaw) {
              next.expectedPdbSha256 = existingShaRaw;
              uploadDigest = existingShaRaw;
            }
            const expName = existingExpName || curFilename;
            if (expName) {
              next.expectedFilename = expName;
              expectedFilename = expName;
            }
          } else if (rawPdb.trim() && existingShaRaw) {
            const actualSha = computeUploadDigest(rawPdb);
            if (!isValidSha256Hex(existingShaRaw) || actualSha !== existingShaRaw) {
              if (hasMissingFlag) next.missingUpload = true;
              next.expectedPdbSha256 = existingShaRaw;
              uploadDigest = existingShaRaw;
              if (existingExpName) {
                next.expectedFilename = existingExpName;
                expectedFilename = existingExpName;
              }
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
    sanitizeOnImport: (params) => {
      const next = { ...params };
      delete next.fetchedPdbId;
      const mode = String(next.mode ?? 'demo-bundle');
      if (mode === 'pdb-id' || mode === 'demo-bundle') {
        if ('pdbText' in next) {
          next.pdbText = '';
        }
      }
      return next;
    },
  },
  catalog: {
    categoryId: 'source',
    conciseRole: 'Fetch by PDB ID, upload/paste PDB & mmCIF, or 3-helix demo',
    searchKeywords: ['pdb', 'mmcif', 'rcsb', 'structure', 'upload', 'template', 'input'],
  },
  display: {
    formatCanvasSummary: (params) => {
      const mode = String(params.mode ?? 'demo-bundle');
      if (mode === 'pdb-id' && params.pdbId) return `PDB ${String(params.pdbId).toUpperCase()}`;
      if (mode === 'custom' && params.filename) return String(params.filename);
      return 'PDB Input';
    },
    sourceRole: 'structure-source',
  },
  preflight: () => ({
    estimatedVramMB: 0,
    estimatedDownloadMB: 0,
    workItems: 0,
    outputCardinality: { structure: 1, sequences: 1 },
    stageBreakdown: { source: 1 },
  }),
  inspector: {
    customWidgetId: 'pdb-input-structure-source',
  },
};
