// Core denovo.run orchestration types (aligned with RFC-0001 + CampaignState hooks).

import type { GeometryStats } from '$lib/protein-ops/kabsch';

export type RuntimeKind = 'ort-webgpu' | 'wgsl-custom' | 'cpu-ts';

export type PortType =
  | 'StructureArtifact'
  | 'TrajectoryFrames'
  | 'SequenceSet'
  | 'FoldedStructure'
  | 'DesignabilityMetrics'
  | 'ConditionSpec';

export type BioEntityKind = 'protein' | 'dna' | 'rna' | 'ligand';

export interface StructureParseDiagnostics {
  chainCount: number;
  chainLengths: number[];
  hasLigands: boolean;
  ligandNames: string[];
  hasNucleicAcids: boolean;
  nucleicAcidChains: string[];
}

export interface BioCompatibilityFix {
  label: string;
  nodeId: string;
  paramsPatch: Record<string, unknown>;
  nodeLabel?: string;
}

export interface BioCompatibilityIssue {
  severity: 'error' | 'warning';
  code: string;
  nodeId: string;
  message: string;
  fix?: BioCompatibilityFix;
}

export interface StructureArtifact {
  kind: 'StructureArtifact';
  id: string;
  label: string;
  sourceModelId: string;
  length: number;
  sequence: string;
  /** Canonical [L * 3] Cα coordinates in Å. */
  caCoords: Float32Array;
  /** Canonical [L * 4 * 3] backbone N/CA/C/O coordinates in Å. */
  backbone4Coords: Float32Array;
  pdbText: string;
  cifText?: string;
  chainIds?: string[];
  chainLengths?: number[];
  diagnostics?: StructureParseDiagnostics;
  /** Optional curated benchmark ID when loaded from a benchmark target preset. */
  benchmarkId?: string;
  /** Optional multi-model PDB when trajectory frames are available. */
  trajectoryPdb?: string;
  trajectoryFrameCount?: number;
  geometry: GeometryStats;
}

export interface SequenceCandidate {
  id: string;
  sequence: string;
  score: number | null;
  globalScore: number | null;
  sourceModelId: string;
  temperature?: number;
  seed?: number;
  chainLengths?: number[];
}

export interface SequenceSet {
  kind: 'SequenceSet';
  id: string;
  parentStructureId: string;
  sourceModelId: string;
  candidates: SequenceCandidate[];
}

export interface FoldedCandidate {
  sequenceId: string;
  sequence: string;
  foldModelId: string;
  pdbText: string;
  caCoords: Float32Array;
  plddt: Float32Array;
  meanPlddt: number;
  ptm: number | null;
  iptm?: number | null;
  chainLengths?: number[];
  chainIds?: string[];
  pae?: Float32Array;
  elapsedMs: number;
}

export interface FoldedStructure {
  kind: 'FoldedStructure';
  id: string;
  parentStructureId: string;
  folds: FoldedCandidate[];
}

export interface DesignabilityRow {
  structureId: string;
  sequenceId: string;
  sequence: string;
  scRmsd: number;
  scTm: number;
  meanPlddt: number;
  ptm: number | null;
  iptm?: number | null;
  designable: boolean; // scRMSD < 2.0 Å && meanPlddt >= 70
  alignedFoldPdb: string;
  designedPdb: string;
  pae?: Float32Array;
  plddt: Float32Array;
  geometry: GeometryStats;
  foldElapsedMs: number;
}

export interface DesignabilityMetrics {
  kind: 'DesignabilityMetrics';
  id: string;
  rows: DesignabilityRow[];
  best: DesignabilityRow | null;
  passRate: number;
}

export interface CanonicalConditionIntent {
  contig: string;
  lockChain?: string;
  hotspots?: string[];
  motifSpans?: string[];
  unindexedMotif?: boolean;
  hbondMode?: 'none' | 'ACCEPTOR' | 'DONOR' | 'BOTH';
  hbondSelection?: string;
  cathCode?: string;
  benchmarkId?: string;
  pdbId?: string;
}

export interface ConditionSpecArtifact {
  kind: 'ConditionSpec';
  id: string;
  label: string;
  contig: string;
  conditions: Record<string, unknown>;
  /** Model-agnostic canonical conditioning intent (compiled per generator in RFD4 & Genie 3). */
  intent?: CanonicalConditionIntent;
  /** Optional remote .cif.zst / .cif URL (e.g. RFD4 upstream benchmark context). */
  cifUrl?: string | null;
  /** Optional inline PDB/mmCIF structure text from upstream template. */
  inputCif?: string;
  specMode?: boolean;
  partialFraction?: number | null;
  inverseFolding?: boolean;
  summary: string;
}

export type CanonicalArtifact =
  | StructureArtifact
  | SequenceSet
  | FoldedStructure
  | DesignabilityMetrics
  | ConditionSpecArtifact;

/** Content-addressed lightweight handle suitable for Agents / WebMCP / Campaign logs. */
export interface ArtifactRef {
  id: string;
  kind: CanonicalArtifact['kind'];
  summary: Record<string, unknown>;
}

export interface PortSpec {
  type: PortType;
  required?: boolean;
}

export interface AdapterManifest {
  id: string;
  displayName: string;
  shortName: string;
  stage: 'source' | 'condition' | 'generate' | 'inverse-fold' | 'fold' | 'score';
  runtime: RuntimeKind;
  estimatedVramMB: number;
  approxWeightsMB: number;
  licenseNote: string;
  description: string;
  inputs: Record<string, PortSpec>;
  outputs: Record<string, PortSpec>;
}

export interface WorkflowNode {
  id: string;
  adapterId: string;
  label?: string;
  params: Record<string, unknown>;
  /** Optional canvas coordinates for visual node-graph editor. */
  position?: { x: number; y: number };
}

export interface WorkflowEdge {
  fromNode: string;
  fromPort: string;
  toNode: string;
  toPort: string;
}

export interface WorkflowGraph {
  id: string;
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

export type NodeRunStatus = 'idle' | 'cached' | 'loading' | 'running' | 'done' | 'error';

export interface NodeProgressEvent {
  nodeId: string;
  status: NodeRunStatus;
  phase?: string;
  progress?: number; // 0..1
  detail?: string;
  elapsedMs?: number;
  error?: string;
}

/**
 * BindClaw-compatible multi-round campaign record.
 * Supports both recording local browser rounds AND importing external
 * BindClaw / LangGraph campaign JSON files for visual inspection.
 */
export interface CampaignRound {
  round: number;
  timestamp: string;
  label: string;
  generator: string;
  folder: string;
  nDesigned: number;
  nPassed: number;
  hitRate: number;
  bestScRmsd: number | null;
  bestPlddt: number | null;
  agentNotes?: string;
  conditioningSummary?: string;
  designedPdb?: string;
  alignedFoldPdb?: string;
  sequence?: string;
  pae?: number[];
}

export interface CampaignState {
  schema: 'denovo.campaign/v1';
  campaignId: string;
  title: string;
  goal: string;
  assay: {
    scRmsdThreshold: number;
    plddtThreshold: number;
    targetName?: string;
    offTargetName?: string;
  };
  rounds: CampaignRound[];
}
