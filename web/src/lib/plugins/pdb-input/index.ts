// Layer 3 Native ModelPlugin — pdb-input

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { computeUploadDigest, isValidSha256Hex } from '$lib/engine/param-validation';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { ArtifactEnvelope, KernelAdapterManifest, ModelPlugin } from '$lib/kernel';
import type { WorkflowNode } from '$lib/engine/types';
import { materializePdbInputNode, deriveSequenceSetFromStructure, runPdbInputCpu } from './cpu-runner';

export { materializePdbInputNode, deriveSequenceSetFromStructure, runPdbInputCpu };

const UPLOAD_META_KEYS = new Set(['missingUpload', 'expectedPdbSha256', 'expectedFilename']);
const INLINE_PDB_SHARE_LIMIT = 1500;

function yamlNodeExplicitlySetsMode(yamlText: string, nodeId: string): boolean {
  const trimmed = yamlText.trim();
  if (!trimmed || trimmed.startsWith('{')) return true;
  let inNodes = false;
  for (const rawLine of trimmed.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    if (line === 'nodes:') {
      inNodes = true;
      continue;
    }
    if (line === 'wires:') {
      inNodes = false;
      continue;
    }
    if (!inNodes) continue;
    const colon = line.indexOf(':');
    if (colon <= 0) continue;
    if (line.slice(0, colon).trim() !== nodeId) continue;
    try {
      const obj = JSON.parse(line.slice(colon + 1).trim()) as Record<string, unknown>;
      return Object.hasOwn(obj, 'mode');
    } catch {
      // Safer default: never silently revert an unparseable node line to a prior upload
      return true;
    }
  }
  return false;
}

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
          if (k === 'pdbText' && typeof v === 'string' && v.length > INLINE_PDB_SHARE_LIMIT && stripLargeUploads) {
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
    restoreOmittedInlineAsset: (prevParams, nextParams, rawRecipeText, nodeId) => {
      if (String(nextParams.pdbText ?? '').trim()) return null;
      const prevHasLargeCustom =
        prevParams.mode === 'custom' &&
        typeof prevParams.pdbText === 'string' &&
        prevParams.pdbText.length > INLINE_PDB_SHARE_LIMIT;
      if (!prevHasLargeCustom) return null;

      const isMissingUploadMarker =
        nextParams.mode === 'custom' && Boolean(nextParams.missingUpload);
      const isImplicitLegacyOmission =
        nextParams.mode === 'demo-bundle' &&
        Boolean(rawRecipeText && nodeId) &&
        !yamlNodeExplicitlySetsMode(rawRecipeText!, nodeId!);

      if (!isMissingUploadMarker && !isImplicitLegacyOmission) return null;

      const expectedSha = String(nextParams.expectedPdbSha256 ?? '')
        .trim()
        .toLowerCase();
      const prevPdbText = prevParams.pdbText as string;
      const digestMatches = !expectedSha || computeUploadDigest(prevPdbText) === expectedSha;
      if (!digestMatches) return null;

      const restored: Record<string, unknown> = {
        ...nextParams,
        mode: 'custom',
        filename: String(nextParams.filename ?? '').trim() || prevParams.filename,
        pdbText: prevPdbText,
      };
      delete restored.missingUpload;
      delete restored.expectedPdbSha256;
      delete restored.expectedFilename;
      return restored;
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
