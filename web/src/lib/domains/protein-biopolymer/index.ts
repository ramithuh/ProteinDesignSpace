// Layer 2 Domain Pack — @domain/protein-biopolymer
// Encapsulates shared protein & multi-entity biopolymer PortTypes, stages, param kinds,
// benchmarks, preflight summaries, and onboarding metadata.
// Depends ONLY on Layer 1 Core Kernel interfaces (and existing protein-ops/engine helpers during Phase 2-3 transition).

import { BENCHMARK_TARGETS, getBenchmarkById } from '$lib/engine/benchmarks';
import type {
  ArtifactEnvelope,
  DomainPack,
  KernelBenchmarkTargetMeta,
  PortTypeDescriptor,
} from '$lib/kernel';

export * from '$lib/protein-ops/pdb';
export * from '$lib/protein-ops/kabsch';
export * from '$lib/protein-ops/hetatm-policy';
export * from '$lib/protein-ops/gnina-prep';
export * from '$lib/protein-ops/bfactor-paint';
export * from '$lib/protein-ops/rcsb';
export * from '$lib/protein-ops/ipsae';

export const PROTEIN_DOMAIN_ID = 'protein-biopolymer';

const PROTEIN_STAGES = [
  { id: 'source', label: 'Input', order: 0 },
  { id: 'condition', label: 'Conditioning', order: 1 },
  { id: 'generate', label: 'Generation', order: 2 },
  { id: 'inverse-fold', label: 'Sequence Design', order: 3 },
  { id: 'fold', label: 'Prediction', order: 4 },
  { id: 'score', label: 'Evaluation', order: 5 },
] as const;

function payloadOf(art: ArtifactEnvelope): Record<string, unknown> {
  return art.payload && typeof art.payload === 'object'
    ? (art.payload as Record<string, unknown>)
    : {};
}

const PROTEIN_PORT_TYPES: PortTypeDescriptor[] = [
  {
    typeId: 'DesignabilityMetrics',
    domainId: PROTEIN_DOMAIN_ID,
    label: 'Designability Scores',
    color: { bg: '#f59e0b', fg: '#ffffff', border: '#d97706' },
    peekPriority: 80,
    hasStageViewport: true,
    stageViewportId: 'mol-superpose-3d',
    viewerId: 'protein-metrics-viewer',
    summarize: (art) => {
      const p = payloadOf(art);
      const rows = Array.isArray(p.rows) ? p.rows : [];
      return {
        kind: art.kind,
        rowCount: rows.length,
        bestScRmsd: typeof p.bestScRmsd === 'number' ? p.bestScRmsd : null,
        passCount: typeof p.passCount === 'number' ? p.passCount : 0,
      };
    },
    formatNodeFooterKpi: (art) => {
      const p = payloadOf(art);
      if (typeof p.bestScRmsd === 'number' && Number.isFinite(p.bestScRmsd)) {
        return {
          text: `scRMSD ${p.bestScRmsd.toFixed(2)} Å`,
          tone: p.bestScRmsd <= 2.0 ? 'ok' : 'warn',
        };
      }
      return null;
    },
  },
  {
    typeId: 'MutationMatrix',
    domainId: PROTEIN_DOMAIN_ID,
    label: 'Mutation Matrix (L×20)',
    color: { bg: '#ec4899', fg: '#ffffff', border: '#db2777' },
    peekPriority: 70,
    viewerId: 'protein-mutation-heatmap',
    summarize: (art) => {
      const p = payloadOf(art);
      return {
        kind: art.kind,
        length: typeof p.length === 'number' ? p.length : 0,
        metricName: typeof p.metricName === 'string' ? p.metricName : 'ddG',
      };
    },
  },
  {
    typeId: 'ResidueTrack',
    domainId: PROTEIN_DOMAIN_ID,
    label: 'Residue Profile',
    color: { bg: '#14b8a6', fg: '#ffffff', border: '#0d9488' },
    peekPriority: 65,
    viewerId: 'protein-residue-track',
    summarize: (art) => {
      const p = payloadOf(art);
      return {
        kind: art.kind,
        length: typeof p.length === 'number' ? p.length : 0,
        metricName: typeof p.metricName === 'string' ? p.metricName : 'score',
      };
    },
  },
  {
    typeId: 'FoldedStructure',
    domainId: PROTEIN_DOMAIN_ID,
    label: 'Predicted Structure',
    color: { bg: '#10b981', fg: '#ffffff', border: '#059669' },
    peekPriority: 60,
    hasStageViewport: true,
    stageViewportId: 'mol-superpose-3d',
    viewerId: 'protein-fold-viewer',
    summarize: (art) => {
      const p = payloadOf(art);
      const folds = Array.isArray(p.folds) ? p.folds : [];
      const top = folds[0] as Record<string, unknown> | undefined;
      return {
        kind: art.kind,
        foldCount: folds.length,
        topMeanPlddt: typeof top?.meanPlddt === 'number' ? top.meanPlddt : null,
      };
    },
    formatNodeFooterKpi: (art) => {
      const p = payloadOf(art);
      const folds = Array.isArray(p.folds) ? p.folds : [];
      const top = folds[0] as Record<string, unknown> | undefined;
      if (typeof top?.meanPlddt === 'number' && Number.isFinite(top.meanPlddt)) {
        return {
          text: `pLDDT ${top.meanPlddt.toFixed(1)}`,
          tone: top.meanPlddt >= 80 ? 'ok' : top.meanPlddt >= 65 ? 'neutral' : 'warn',
        };
      }
      return null;
    },
  },
  {
    typeId: 'StructureArtifact',
    domainId: PROTEIN_DOMAIN_ID,
    label: 'Structure (3D)',
    color: { bg: '#3b82f6', fg: '#ffffff', border: '#2563eb' },
    peekPriority: 50,
    hasStageViewport: true,
    stageViewportId: 'mol-superpose-3d',
    viewerId: 'protein-structure-viewer',
    summarize: (art) => {
      const p = payloadOf(art);
      const dm = p.dockingMetrics as Record<string, unknown> | undefined;
      return {
        kind: art.kind,
        length: typeof p.length === 'number' ? p.length : 0,
        ...(dm && typeof dm.cnnScore === 'number' ? { cnnScore: dm.cnnScore } : {}),
      };
    },
    formatNodeFooterKpi: (art) => {
      const p = payloadOf(art);
      const dm = p.dockingMetrics as Record<string, unknown> | undefined;
      if (dm && typeof dm.cnnScore === 'number' && Number.isFinite(dm.cnnScore)) {
        return {
          text: `CNNscore ${dm.cnnScore.toFixed(3)}`,
          tone: dm.cnnScore >= 0.7 ? 'ok' : dm.cnnScore >= 0.4 ? 'neutral' : 'warn',
        };
      }
      if (typeof p.length === 'number' && p.length > 0) {
        return { text: `${p.length} aa`, tone: 'neutral' };
      }
      return null;
    },
  },
  {
    typeId: 'SequenceSet',
    domainId: PROTEIN_DOMAIN_ID,
    label: 'Sequences (AA)',
    color: { bg: '#a855f7', fg: '#ffffff', border: '#9333ea' },
    peekPriority: 40,
    viewerId: 'protein-sequence-viewer',
    summarize: (art) => {
      const p = payloadOf(art);
      const candidates = Array.isArray(p.candidates) ? p.candidates : [];
      return { kind: art.kind, candidateCount: candidates.length };
    },
  },
  {
    typeId: 'SequenceEmbedding',
    domainId: PROTEIN_DOMAIN_ID,
    label: 'Embeddings',
    color: { bg: '#6366f1', fg: '#ffffff', border: '#4f46e5' },
    peekPriority: 30,
    viewerId: 'protein-embedding-viewer',
    summarize: (art) => {
      const p = payloadOf(art);
      return {
        kind: art.kind,
        hiddenDim: typeof p.hiddenDim === 'number' ? p.hiddenDim : 0,
        length: typeof p.length === 'number' ? p.length : 0,
      };
    },
  },
  {
    typeId: 'TrajectoryFrames',
    domainId: PROTEIN_DOMAIN_ID,
    label: 'Trajectory',
    color: { bg: '#06b6d4', fg: '#ffffff', border: '#0891b2' },
    peekPriority: 20,
    hasStageViewport: true,
    stageViewportId: 'mol-superpose-3d',
    viewerId: 'protein-trajectory-viewer',
    summarize: (art) => {
      const p = payloadOf(art);
      const frames = Array.isArray(p.frames) ? p.frames : [];
      return { kind: art.kind, frameCount: frames.length };
    },
  },
  {
    typeId: 'ConditionSpec',
    domainId: PROTEIN_DOMAIN_ID,
    label: 'Design Constraints',
    color: { bg: '#f43f5e', fg: '#ffffff', border: '#e11d48' },
    peekPriority: 10,
    viewerId: 'protein-condition-viewer',
    summarize: (art) => {
      const p = payloadOf(art);
      return { kind: art.kind, mode: typeof p.mode === 'string' ? p.mode : 'motif' };
    },
  },
];

const AA20_RE = /^[ACDEFGHIKLMNPQRSTVWY,\s]*$/i;

export const proteinBiopolymerDomainPack: DomainPack = {
  id: PROTEIN_DOMAIN_ID,
  displayName: 'Protein & Biopolymer Design',
  stages: PROTEIN_STAGES,
  portTypes: PROTEIN_PORT_TYPES,
  paramFieldKinds: [
    {
      kind: 'residue-list',
      validateValue: (value) => {
        if (value === undefined || value === null || value === '') return null;
        if (typeof value !== 'string') return 'Expected residue token list string.';
        return null;
      },
    },
    {
      kind: 'aa-set',
      validateValue: (value) => {
        if (value === undefined || value === null || value === '') return null;
        if (typeof value !== 'string' || !AA20_RE.test(value)) {
          return 'Amino acid set must contain canonical 1-letter residue codes (A–Y).';
        }
        return null;
      },
    },
  ],
  benchmarkProvider: {
    targets: BENCHMARK_TARGETS as unknown as readonly KernelBenchmarkTargetMeta[],
    getTargetById: (id) => getBenchmarkById(id) as unknown as KernelBenchmarkTargetMeta | undefined,
  },
  formatPreflightSummary: (stageTotals) => {
    const parts: string[] = [];
    if ((stageTotals.generate ?? 0) > 0) {
      parts.push(`${stageTotals.generate} backbone${stageTotals.generate === 1 ? '' : 's'}`);
    }
    if ((stageTotals['inverse-fold'] ?? 0) > 0) {
      parts.push(
        `${stageTotals['inverse-fold']} sequence${stageTotals['inverse-fold'] === 1 ? '' : 's'}`,
      );
    }
    if ((stageTotals.fold ?? 0) > 0) {
      parts.push(`${stageTotals.fold} refold${stageTotals.fold === 1 ? '' : 's'}`);
    }
    return parts.length > 0 ? parts.join(' → ') : undefined;
  },
  onboardingTour: [
    {
      id: 'add-mpnn',
      title: 'Design sequences',
      body: 'Add ProteinMPNN to design amino-acid sequences for the backbone.',
      targetCategoryId: 'sequence-design',
      targetSubKey: 'ca',
      autoAddPluginId: 'web-protein-mpnn',
    },
    {
      id: 'add-fold',
      title: 'Predict 3D fold',
      body: 'Add LocalFold (AF2 / Boltz / AF3 / ESMFold2) to verify self-consistency.',
      targetCategoryId: 'prediction',
      targetSubKey: 'monomer',
      autoAddPluginId: 'localfold-af2',
    },
    {
      id: 'add-scorer',
      title: 'Score designability',
      body: 'Add Designability Scorer to compute scRMSD, scTM, and pLDDT.',
      targetCategoryId: 'evaluation',
      targetSubKey: 'designability-scorer',
      autoAddPluginId: 'designability-scorer',
    },
  ],
  artifactViewers: [
    {
      id: 'protein-structure-viewer',
      domainId: PROTEIN_DOMAIN_ID,
      label: 'Structure Inspector',
      supportedPortTypes: ['StructureArtifact'],
      componentLoader: () => import('$lib/ui/ArtifactPeekPanel.svelte'),
    },
    {
      id: 'protein-sequence-viewer',
      domainId: PROTEIN_DOMAIN_ID,
      label: 'Sequence Set Inspector',
      supportedPortTypes: ['SequenceSet'],
      componentLoader: () => import('$lib/ui/ArtifactPeekPanel.svelte'),
    },
    {
      id: 'protein-fold-viewer',
      domainId: PROTEIN_DOMAIN_ID,
      label: 'Predicted Structure Inspector',
      supportedPortTypes: ['FoldedStructure'],
      componentLoader: () => import('$lib/ui/ArtifactPeekPanel.svelte'),
    },
    {
      id: 'protein-metrics-viewer',
      domainId: PROTEIN_DOMAIN_ID,
      label: 'Designability Metrics Inspector',
      supportedPortTypes: ['DesignabilityMetrics'],
      componentLoader: () => import('$lib/ui/ArtifactPeekPanel.svelte'),
    },
    {
      id: 'protein-mutation-heatmap',
      domainId: PROTEIN_DOMAIN_ID,
      label: 'Mutation Heatmap',
      supportedPortTypes: ['MutationMatrix'],
      componentLoader: () => import('$lib/viewer/MutationHeatmap.svelte'),
    },
    {
      id: 'protein-residue-track',
      domainId: PROTEIN_DOMAIN_ID,
      label: 'Residue Track Plot',
      supportedPortTypes: ['ResidueTrack'],
      componentLoader: () => import('$lib/viewer/ResidueTrackPlot.svelte'),
    },
    {
      id: 'protein-embedding-viewer',
      domainId: PROTEIN_DOMAIN_ID,
      label: 'Sequence Embedding Inspector',
      supportedPortTypes: ['SequenceEmbedding'],
      componentLoader: () => import('$lib/ui/ArtifactPeekPanel.svelte'),
    },
    {
      id: 'protein-trajectory-viewer',
      domainId: PROTEIN_DOMAIN_ID,
      label: 'Trajectory Inspector',
      supportedPortTypes: ['TrajectoryFrames'],
      componentLoader: () => import('$lib/ui/ArtifactPeekPanel.svelte'),
    },
    {
      id: 'protein-condition-viewer',
      domainId: PROTEIN_DOMAIN_ID,
      label: 'Condition Spec Inspector',
      supportedPortTypes: ['ConditionSpec'],
      componentLoader: () => import('$lib/ui/ArtifactPeekPanel.svelte'),
    },
  ],
  stageViewports: [
    {
      id: 'mol-superpose-3d',
      domainId: PROTEIN_DOMAIN_ID,
      label: '3D Molecular Viewport',
      supportedFormats: ['pdb', 'mmcif'],
      supportsMultiModelOverlay: true,
      componentLoader: () => import('$lib/viewer/MolSuperposeViewer.svelte'),
    },
  ],
};
