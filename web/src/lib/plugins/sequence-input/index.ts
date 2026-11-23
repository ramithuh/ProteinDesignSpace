// Layer 3 Native ModelPlugin — sequence-input

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { validateSequenceText } from '$lib/engine/param-validation';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { ArtifactEnvelope, KernelAdapterManifest, ModelPlugin } from '$lib/kernel';
import type { WorkflowNode } from '$lib/engine/types';
import { materializeSequenceInputNode, runSequenceInputCpu } from './cpu-runner';

export { materializeSequenceInputNode, runSequenceInputCpu };

export const sequenceInputPlugin: ModelPlugin = {
  id: 'sequence-input',
  domainId: PROTEIN_DOMAIN_ID,
  manifest: ADAPTER_REGISTRY['sequence-input'] as unknown as KernelAdapterManifest,
  execution: {
    runtime: 'cpu-ts',
    fallback: { kind: 'none', reason: 'Pure TypeScript CPU adapter' },
    vramLeaseMode: 'exclusive',
    executeCpu: (ctx) => {
      const node = ctx.node as unknown as WorkflowNode;
      return Promise.resolve({
        sequences: materializeSequenceInputNode(node),
      } as unknown as Record<string, ArtifactEnvelope>);
    },
    materializePreview: async (params) => {
      const syntheticNode: WorkflowNode = {
        id: 'seq-preview',
        adapterId: 'sequence-input',
        params,
      };
      const out = await runSequenceInputCpu(syntheticNode);
      return out as unknown as Record<string, ArtifactEnvelope>;
    },
  },
  cachePolicy: {},
  share: {
    shortCode: 'q',
    yamlKeyAliases: { label: 'seqLabel' },
  },
  catalog: {
    categoryId: 'source',
    conciseRole: 'Paste amino acid sequences (plain text or FASTA)',
    searchKeywords: ['fasta', 'sequence', 'amino acid', 'input'],
  },
  display: {
    formatCanvasSummary: (params) => {
      const raw = String(params.sequence ?? '').trim();
      if (!raw) return 'Sequence Input';
      const check = validateSequenceText(raw);
      return check.numSequences > 1 ? `${check.numSequences} sequences` : `${check.totalResidues} aa`;
    },
    sourceRole: 'sequence-source',
  },
  preflight: (ctx) => {
    const raw = typeof ctx.params.sequence === 'string' ? ctx.params.sequence.trim() : '';
    const count = raw ? Math.max(1, validateSequenceText(raw).numSequences) : 1;
    return {
      estimatedVramMB: 0,
      estimatedDownloadMB: 0,
      workItems: 0,
      outputCardinality: { sequences: count },
      stageBreakdown: { source: count },
    };
  },
  inspector: {
    customWidgetId: 'sequence-input-editor',
  },
};
