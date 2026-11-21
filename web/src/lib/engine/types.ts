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

export type CollectionKind = 'independent-candidates' | 'multi-sample' | 'ordered-frames';

export type ReferenceRelation =
  | 'self-consistency'
  | 'external-mismatch'
  | 'external-unlinked'
  | 'external-mapped';

export interface RepresentativeSelection {
  selectedKey: string;
  criterion: string;
  totalMembers: number;
  selectedIndex?: number;
  scoreValue?: number | null;
}

export interface CollectionItemOutcome {
  key: string;
  status: 'completed' | 'excluded' | 'failed';
  reason?: string;
}

export interface ExcludedCandidateRecord {
  key: string;
  id: string;
  index: number;
  reason: string;
}

export interface ReferenceResidueMapping {
  /** Explicit 0-based [foldIndex, referenceIndex] residue pairs. */
  pairs?: Array<[number, number]>;
  /** Explicit 0-based residue indices into the fold coordinates. */
  foldIndices?: number[];
  /** Explicit 0-based residue indices into the reference coordinates. */
  referenceIndices?: number[];
  /** Optional human-readable description of the external mapping. */
  label?: string;
}

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
  /** Chain IDs whose sequence/backbone must remain locked during inverse folding (e.g. binder targets). */
  lockedChains?: string[];
  /** 1-based flat residue indices in the output structure to keep fixed in ProteinMPNN (e.g. scaffolded motifs). */
  fixedResidueIndices?: number[];
  /** Semantic role per output protein chain ID (e.g. target vs binder vs scaffold). */
  chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>;
  /** 0-based residue indices into `caCoords` corresponding to `nativeMotifCoords`. */
  motifIndices?: number[];
  /** Native template Cα [x, y, z] coordinates (Å) for each residue in `motifIndices`. */
  nativeMotifCoords?: number[][];
  /** Optional parent structure ID when this structure was derived from an upstream structure/collection. */
  parentStructureId?: string;
  /** Optional provenance recording how this structure was selected from a multi-member collection. */
  representativeSelection?: RepresentativeSelection;
  geometry: GeometryStats;
}

export interface TrajectoryFrame {
  key: string;
  frameIndex: number;
  pdbText?: string;
  cifText?: string;
  caCoords?: Float32Array;
  timeStep?: number;
  confidence?: number | null;
}

export interface TrajectoryFramesArtifact {
  kind: 'TrajectoryFrames';
  id: string;
  label?: string;
  parentStructureId: string;
  sourceModelId?: string;
  frameCount: number;
  ordering: 'ordered-frames';
  collectionKind?: 'ordered-frames';
  pdbText?: string;
  cifText?: string;
  frames?: TrajectoryFrame[];
  representativeSelection?: RepresentativeSelection;
}

export interface SequenceCandidate {
  id: string;
  /** Canonical immutable candidate key across workflow stages (defaults to `id`). */
  key?: string;
  sequence: string;
  score: number | null;
  globalScore: number | null;
  sourceModelId: string;
  sourceStructureId?: string;
  sampleIndex?: number;
  temperature?: number;
  seed?: number;
  chainLengths?: number[];
}

export interface SequenceSet {
  kind: 'SequenceSet';
  id: string;
  parentStructureId: string;
  /** Explicit source structure lineage (mirrors `parentStructureId` when derived from a structure). */
  sourceStructureId?: string;
  sourceModelId: string;
  collectionKind?: CollectionKind;
  candidates: SequenceCandidate[];
}

export interface FoldedCandidate {
  sequenceId: string;
  /** Canonical immutable candidate key across workflow stages (defaults to `sequenceId`). */
  key?: string;
  sequence: string;
  foldModelId: string;
  sourceStructureId?: string;
  sampleIndex?: number;
  pdbText: string;
  caCoords: Float32Array;
  plddt: Float32Array;
  meanPlddt: number;
  /** Optional model confidence score; `null` or `undefined` when the predictor has no confidence head. */
  confidence?: number | null;
  /** Explicitly indicates whether a confidence score is available (`false` for confidence-less samplers). */
  confidenceAvailable?: boolean;
  ptm: number | null;
  iptm?: number | null;
  /** Per-chain pTM keyed by chain ID (e.g. { A: 0.85, B: 0.72 }). */
  chainPtm?: Record<string, number>;
  /** Per-chain mean interface pTM keyed by chain ID. */
  chainIptm?: Record<string, number>;
  /** Symmetric C×C chain-pair ipTM matrix (diagonal = chain pTM, off-diagonal = pairwise ipTM). */
  chainPairIptm?: number[][];
  /** Composite multimer confidence (0.8 * ipTM + 0.2 * pTM) for multi-chain predictions. */
  multimerScore?: number;
  /** True when `iptm` was estimated using AF2-Monomer weights rather than AF2-Multimer/AF3. */
  iptmIsMonomerApprox?: boolean;
  chainLengths?: number[];
  chainIds?: string[];
  pae?: Float32Array;
  elapsedMs: number;
}

export interface FoldedStructure {
  kind: 'FoldedStructure';
  id: string;
  parentStructureId: string;
  /** Explicit upstream source structure ID carried from SequenceSet lineage. */
  sourceStructureId?: string;
  collectionKind?: CollectionKind;
  excludedCandidates?: ExcludedCandidateRecord[];
  itemOutcomes?: CollectionItemOutcome[];
  representativeSelection?: RepresentativeSelection;
  folds: FoldedCandidate[];
}

export interface DesignabilityRow {
  structureId: string;
  sequenceId: string;
  /** Canonical immutable candidate key (defaults to `sequenceId`). */
  candidateKey?: string;
  /** Recorded source structure lineage of the folded candidate. */
  sourceStructureId?: string;
  sequence: string;
  scRmsd: number;
  scTm: number;
  meanPlddt: number;
  /** Optional explicit confidence availability (`false` for confidence-less predictors). */
  confidenceAvailable?: boolean;
  ptm: number | null;
  iptm?: number | null;
  /** Mean interface PAE (Å) across cross-chain residue pairs for multi-chain complexes. */
  ipae?: number;
  /** Mean pLDDT on the designed binder chain(s). */
  binderPlddt?: number;
  /** Mean pLDDT on the locked target chain(s). */
  targetPlddt?: number;
  /** Self-consistency Cα RMSD (Å) of the binder chain aligned independently. */
  binderScRmsd?: number;
  /** Cα RMSD (Å) of the binder chain after superimposing the target chain. */
  targetAlignedBinderRmsd?: number;
  /** Generated motif Cα RMSD (Å) against native template motif coordinates. */
  genMotifRmsd?: number;
  /** Refolded motif Cα RMSD (Å) against native template motif coordinates. */
  refoldMotifRmsd?: number;
  /** Reference comparison classification for this row. */
  referenceMode?: 'self-consistency' | 'external-reference' | 'lineage-mismatch';
  /** Exact semantic relationship between the fold's lineage and the wired reference structure. */
  referenceRelation?: ReferenceRelation;
  /** Human-readable claim label (e.g. "Self-consistency against <ref>" vs "Comparison against <ref> · Fold originated from <parent>"). */
  referenceLabel?: string;
  /** True only when the fold descends directly from the wired reference structure (`parentStructureId === ref.id`). */
  selfConsistencyValid?: boolean;
  /** True when fold lineage conflicted with the wired reference without an explicit external-reference config. */
  lineageMismatch?: boolean;
  /** True when an explicit residue mapping was applied to align against an external reference. */
  referenceMappingApplied?: boolean;
  /** Number of aligned residue pairs used in Kabsch superposition. */
  alignedResidueCount?: number;
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
  /** Wired reference StructureArtifact.id evaluated against. */
  referenceStructureId?: string;
  /** Recorded parent/source structure ID from the incoming FoldedStructure. */
  foldParentStructureId?: string;
  referenceMode?: 'self-consistency' | 'external-reference' | 'lineage-mismatch';
  /** Exact semantic relationship between the fold's lineage and the wired reference structure. */
  referenceRelation?: ReferenceRelation;
  /** Human-readable claim label distinguishing self-consistency from external-reference comparisons. */
  referenceLabel?: string;
  /** True only when folds descend directly from the wired reference (`parentStructureId === ref.id`). */
  selfConsistencyValid?: boolean;
  lineageStatus?: 'matched' | 'external-mapped' | 'external-unmapped' | 'mismatch' | 'untracked';
  lineageMismatch?: boolean;
  lineageWarning?: string;
  referenceMapping?: ReferenceResidueMapping;
  collectionKind?: CollectionKind;
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
  | TrajectoryFramesArtifact
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
