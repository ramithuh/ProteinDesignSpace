// Core denovo.run orchestration types (aligned with RFC-0001 + CampaignState hooks).

import type { GeometryStats } from '$lib/protein-ops/kabsch';
import type { BackboneAtomPresence, SourceResidueId } from '$lib/protein-ops/pdb';

export type RuntimeKind = 'ort-webgpu' | 'wgsl-custom' | 'cpu-ts';

export type PortType =
  | 'StructureArtifact'
  | 'TrajectoryFrames'
  | 'SequenceSet'
  | 'FoldedStructure'
  | 'DesignabilityMetrics'
  | 'MutationMatrix'
  | 'ResidueTrack'
  | 'SequenceEmbedding'
  | 'ConditionSpec';

export type BioEntityKind = 'protein' | 'dna' | 'rna' | 'ligand';

export type ChainMolecularKind =
  | 'protein'
  | 'antibody-H'
  | 'antibody-L'
  | 'nanobody'
  | 'dna'
  | 'rna'
  | 'ligand';

export type MsaMode = 'single' | 'mmseqs2' | 'target-only' | 'a3m';

export type BfactorSemantics =
  | 'none'
  | 'plddt-0-100'
  | 'plddt-0-1'
  | 'dynamics-prob'
  | 'ddg-stability'
  | 'llr-fitness'
  | 'gnina-contrib'
  | 'crystal-b';

export type AtomGranularity = 'ca-only' | 'backbone4' | 'all-atom';

export interface ArtifactProvenance {
  quantization?: 'fp32' | 'fp16' | 'int8' | 'int5';
  languageModelQuantization?: 'int3';
  confidenceHead?: string;
  diffusionStepsScheduled?: number;
  diffusionStepsRun?: number;
  recyclesRun?: number;
  earlyStopped?: boolean;
  msaMode?: MsaMode;
  msaDepth?: number;
  degradedFallback?: string;
  warnings?: string[];
}

export interface ResidueTrackArtifact {
  kind: 'ResidueTrack';
  id: string;
  label?: string;
  sourceModelId: string;
  parentStructureId?: string;
  quantity: string;
  unit?: string;
  range?: [number, number];
  threshold?: number;
  residues: string;
  /** Insertion-code-safe residue labels matching `residues` (e.g. `['A1', 'A2', 'H100A']`). */
  residueIds?: string[];
  values: Float32Array;
  rawLogits?: Float32Array;
  summary?: {
    mean: number;
    max: number;
    min?: number;
    fractionAboveThreshold?: number;
    pseudoPerplexity?: number;
  };
}

export interface TargetedMutationScore {
  mutation: string;
  chainId?: string;
  residueId?: string;
  wildType: string;
  mutant: string;
  score: number;
}

export interface MutationMatrixArtifact {
  kind: 'MutationMatrix';
  id: string;
  label?: string;
  sourceModelId: string;
  parentStructureId?: string;
  /** Semantic score type (e.g. ThermoMPNN `ddg-kcal-mol` or ESMC `llr-fitness`). */
  metric: 'ddg-kcal-mol' | 'llr-fitness' | string;
  unit?: string;
  /** Whether lower scores are favorable (`ddg-kcal-mol`) or higher scores are favorable (`llr-fitness`). */
  direction: 'lower-is-better' | 'higher-is-better';
  residues: string;
  /** Insertion-code-safe residue labels matching `residues` (length L). */
  residueIds: string[];
  /** Canonical 20-AA column alphabet matching `values` stride (default `ACDEFGHIKLMNPQRSTVWY`). */
  alphabet: string;
  /** Row-major `[L * alphabet.length]` matrix of per-mutation scores. */
  values: Float32Array;
  /** Optional explicit point-mutant table when running targeted mutation mode. */
  targetedMutations?: TargetedMutationScore[];
  /** Optional per-position aggregate score (length L) for 3D B-factor painting. */
  meanPerPosition?: Float32Array;
  provenance?: ArtifactProvenance;
}

export interface SequenceEmbeddingArtifact {
  kind: 'SequenceEmbedding';
  id: string;
  label?: string;
  sourceModelId: string;
  parentStructureId?: string;
  sequenceId?: string;
  sequence: string;
  dim: number;
  /** Mean-pooled sequence embedding `[dim]`. */
  pooled: Float32Array;
  /** Optional row-major per-residue embeddings `[L * dim]`. */
  perResidue?: Float32Array;
  /** Optional sequence pseudo-perplexity / naturalness score. */
  pseudoPerplexity?: number;
  provenance?: ArtifactProvenance;
}

export interface StructureMolecularContext {
  /** Preserved non-water ligand/cofactor/metal residue names (CCD codes). */
  ligands?: string[];
  /** Preserved nucleic acid chain IDs. */
  nucleicAcidChains?: string[];
  /** Whether explicit hydrogens are present (e.g. LASErMPNN packed outputs). */
  hasExplicitHydrogens?: boolean;
  /** Per-chain molecular classification (protein, antibody-H/L, nanobody, dna, rna, ligand). */
  chainTypes?: Record<string, ChainMolecularKind>;
}

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

export interface ArtifactCollection<T = unknown> {
  kind: 'Collection';
  schema?: 'denovo.collection/v1';
  id: string;
  familyKey?: string;
  collectionKind: CollectionKind;
  requestedCount: number;
  complete?: boolean;
  /** Ordered list of materialized members (e.g., N independent generated backbones). */
  items: T[];
  /** Optional deterministic member keys aligned 1:1 with `items`. */
  keys?: string[];
  representativeSelection?: RepresentativeSelection;
}

export interface SequenceFilterMetadata {
  mode: 'per-backbone' | 'global';
  topK: number;
  maxTotal: number | null;
  metric: 'score' | 'globalScore';
  direction: 'min' | 'max';
  totalInput: number;
  selectedCount: number;
  excludedCount: number;
  description: string;
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
  /** Non-buffer ligands/ions present in the source structure that were dropped by an active crop. */
  omittedLigands?: string[];
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

export interface GninaPoseScoreRow {
  poseId: string;
  rank?: number;
  ligandCode: string;
  ligandSource: 'holo-hetatm' | 'external-sdf' | 'external-mol2' | 'rcsb-ccd';
  heavyAtoms: number;
  rotors: number;
  cnnScore: number;
  cnnAffinity: number;
  cnnVS: number;
  vinaScore: number;
  /** Heavy-atom RMSD (Å) to rank-1 pose (present in `dock` mode). */
  rmsdToBest?: number;
  /** Present in `minimize` mode: initial pose scores before rigid-body optimization. */
  initialCnnScore?: number;
  initialCnnAffinity?: number;
  initialCnnVS?: number;
  initialVinaScore?: number;
  /** Heavy-atom RMSD (Å) moved during rigid-body minimization. */
  poseRmsd?: number;
  /** Per-model ensemble breakdown (`dense_1.3`, `dense_1.3_PT_KD_3`, `crossdock_default2018_KD_4`). */
  models: Record<string, { cnnScore: number; cnnAffinity: number }>;
  /** Optional Vina empirical energy decomposition. */
  vinaTerms?: {
    gauss1: number;
    gauss2: number;
    repulsion: number;
    hydrophobic: number;
    hydrogen: number;
    rotors: number;
  };
}

export interface DockingScoreSummary {
  engine: 'gnina-1.3';
  taskMode: 'score' | 'pocket-scan' | 'minimize' | 'dock';
  precision: 'int8' | 'fp16' | 'fp32';
  ensemble: string[];
  receptorName: string;
  includeMetals: boolean;
  detectedLigands: string[];
  selectedLigand: string;
  pocketResiduesScanned?: number;
  pocketRadius?: number;
  box?: { center: [number, number, number]; size: [number, number, number] };
  dockStats?: {
    gridMs: number;
    searchMs: number;
    rescoreMs: number;
    evaluatedCandidates: number;
    rotors: number;
  };
  rows: GninaPoseScoreRow[];
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
  /** Per-residue insertion codes (`''` when absent, e.g. `'A'`, `'B'`). */
  insCodes?: string[];
  /** Per-residue source identity mapping each residue back to its original chain/resNum/insCode/resName. */
  sourceResidues?: SourceResidueId[];
  /** Bijective map from projected 1-char PDB chain ID -> original source `authChainId`. */
  chainIdMap?: Record<string, string>;
  /** Per-residue observed backbone atom presence flags (`hasN`, `hasCA`, `hasC`, `hasO`, `oReconstructed`). */
  atomPresence?: BackboneAtomPresence[];
  /** Compact per-residue observed backbone atom bitmask (`BACKBONE_MASK_N/CA/C/O`). */
  observedBackboneMask?: Uint8Array;
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
  /** Optional symmetric tied-residue specification (e.g. `"A1=B25=C49; ..."`). */
  tiedPositions?: string;
  /** Semantic role per output protein chain ID (e.g. target vs binder vs scaffold). */
  chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>;
  /** 0-based residue indices into `caCoords` corresponding to `nativeMotifCoords`. */
  motifIndices?: number[];
  /** Native template Cα [x, y, z] coordinates (Å) for each residue in `motifIndices`. */
  nativeMotifCoords?: number[][];
  /** Optional parent structure ID when this structure was derived from an upstream structure/collection. */
  parentStructureId?: string;
  /** Deterministic immutable design key when produced as part of a multi-design collection. */
  designKey?: string;
  /** 0-based creation ordinal within a multi-design collection. */
  designIndex?: number;
  /** Seed used to generate this specific backbone. */
  seed?: number;
  /** Optional multi-backbone collection envelope flowing cleanly over the existing `'structure'` port. */
  collection?: ArtifactCollection<StructureArtifact>;
  /** Optional provenance recording how this structure was selected from a multi-member collection. */
  representativeSelection?: RepresentativeSelection;
  /** Semantic meaning of PDB/mmCIF B-factor values for faithful viewer coloring. */
  bfactorSemantics?: BfactorSemantics;
  /** True atomic resolution of emitted coordinates (avoids pretending Cα-only is all-atom). */
  atomGranularity?: AtomGranularity;
  /** Preserved non-protein / antibody molecular context for holo & specialized models. */
  molecularContext?: StructureMolecularContext;
  /** Optional GNINA 1.3 protein–ligand docking & affinity summary attached to scored/minimized complexes. */
  dockingMetrics?: DockingScoreSummary;
  /** Execution provenance (quantization, early stopping, fallback warnings). */
  provenance?: ArtifactProvenance;
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
  /** Deterministic immutable sequence key (independent of score sorting / Top-K). */
  sequenceKey?: string;
  /** Parent backbone design key when derived from a multi-backbone collection. */
  designKey?: string;
  /** 0-based parent design ordinal when derived from a multi-backbone collection. */
  designIndex?: number;
  sequence: string;
  score: number | null;
  globalScore: number | null;
  sourceModelId: string;
  sourceStructureId?: string;
  sampleIndex?: number;
  temperature?: number;
  seed?: number;
  chainLengths?: number[];
  /** Ordered physical chain IDs matching `chainLengths` (prevents target-only MSA chain-role mismatch). */
  chainIds?: string[];
}

export interface SequenceSet {
  kind: 'SequenceSet';
  id: string;
  parentStructureId: string;
  /** Explicit source structure lineage (mirrors `parentStructureId` when derived from a structure). */
  sourceStructureId?: string;
  /** Distinct parent backbone IDs when candidates span multiple designs. */
  sourceStructureIds?: string[];
  sourceModelId: string;
  collectionKind?: CollectionKind;
  /** Locked target chain IDs propagated from upstream structures (used by target-only MSA). */
  lockedChains?: string[];
  /** Chain roles propagated from upstream structures (used by target-only MSA & binder scoring). */
  chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>;
  /** Explicit excluded candidates recorded by Top-K filtering or pre-folding caps. */
  excludedCandidates?: ExcludedCandidateRecord[];
  /** Item-level outcomes recorded by Top-K filtering. */
  itemOutcomes?: CollectionItemOutcome[];
  /** Metadata describing the Top-K filter policy applied to produce this SequenceSet. */
  filterMetadata?: SequenceFilterMetadata;
  candidates: SequenceCandidate[];
}

export interface FoldedCandidate {
  sequenceId: string;
  /** Canonical immutable candidate key across workflow stages (defaults to `sequenceId`). */
  key?: string;
  /** Deterministic parent design key when folding multi-backbone designs. */
  designKey?: string;
  /** 0-based parent design ordinal. */
  designIndex?: number;
  /** Deterministic parent sequence key. */
  sequenceKey?: string;
  /** Deterministic model-specific fold key (distinct from `sequenceKey`). */
  foldKey?: string;
  /** Stable seed used to fold this candidate. */
  seed?: number;
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
  resNums?: number[];
  pae?: Float32Array;
  /** Dunbrack-lab ipSAE interface score (max directional ipSAE in [0, 1]) for multi-chain complexes. */
  ipsae?: number;
  ipsaeMin?: number;
  /** MSA mode & alignment depth used for this fold. */
  msaMode?: MsaMode;
  msaDepth?: number;
  provenance?: ArtifactProvenance;
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
  /** Deterministic parent design key when evaluated against a multi-backbone collection. */
  designKey?: string;
  /** 0-based parent design ordinal. */
  designIndex?: number;
  /** Deterministic parent sequence key. */
  sequenceKey?: string;
  /** Deterministic fold key. */
  foldKey?: string;
  /** Recorded source structure lineage of the folded candidate. */
  sourceStructureId?: string;
  sequence: string;
  chainLengths?: number[];
  scRmsd: number;
  scTm: number;
  meanPlddt: number;
  /** Optional explicit confidence availability (`false` for confidence-less predictors). */
  confidenceAvailable?: boolean;
  ptm: number | null;
  iptm?: number | null;
  /** Mean interface PAE (Å) across cross-chain residue pairs for multi-chain complexes. */
  ipae?: number;
  /** Dunbrack-lab ipSAE interface score (max directional ipSAE in [0, 1]) for multi-chain complexes. */
  ipsae?: number;
  ipsaeMin?: number;
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
  /** Explicit list of human-readable gate failures when `designable === false`. */
  failedGates?: string[];
  /** MSA mode & depth carried from the evaluated fold. */
  msaMode?: MsaMode;
  msaDepth?: number;
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
  /** True when evaluated against a multi-backbone structure collection. */
  multiDesign?: boolean;
  /** Number of distinct backbones evaluated in multi-design scoring. */
  evaluatedDesignCount?: number;
  /** Fraction of evaluated backbones with at least one passing sequence (`null` when 0 evaluated). */
  designHitRate?: number | null;
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
  | MutationMatrixArtifact
  | ResidueTrackArtifact
  | SequenceEmbeddingArtifact
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
  /** When true, this input port accepts multiple incoming wires (fan-in, e.g. 3D Compare). */
  multiple?: boolean;
}

export interface AdapterParamOption {
  value: string;
  label: string;
  hint?: string;
}

export interface AdapterParamField {
  key: string;
  label: string;
  kind: 'number' | 'integer' | 'boolean' | 'select' | 'text' | 'textarea' | 'residue-list' | 'aa-set';
  /** Controls whether this field renders in the primary area or inside Advanced parameters. */
  tier: 'primary' | 'advanced';
  defaultValue: unknown;
  description?: string;
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: AdapterParamOption[];
  /** Optional list of adapter mode IDs where this field applies (omit = all modes). */
  modes?: string[];
  /** Optional upstream default hint shown next to the field label. */
  upstreamDefaultHint?: string;
  /** Optional interactive Node Studio brush role for residue-list / tied-position fields. */
  studioRole?: 'fixed' | 'hotspot' | 'motif' | 'tied' | 'crop';
  /** Optional non-blocking guardrail warning shown when numeric value drops below `threshold`. */
  warnBelow?: {
    threshold: number;
    message: string;
    modes?: string[];
    fixValue?: number;
  };
  /** For `residue-list` fields: whether bare chain tokens (e.g. "A", "AA") are valid selectors. */
  allowBareChain?: boolean;
}

export interface AdapterModeSpec {
  id: string;
  label: string;
  description: string;
  /** Input port names that are strictly required when this mode is selected. */
  requiredInputs?: string[];
}

export interface AdapterManifest {
  id: string;
  displayName: string;
  shortName: string;
  stage: 'source' | 'condition' | 'generate' | 'inverse-fold' | 'fold' | 'score';
  runtime: RuntimeKind;
  estimatedVramMB: number;
  approxWeightsMB: number;
  /** Optional per-precision weight download sizes in MB (SSOT for preflight & catalog). */
  precisionWeightsMB?: Record<string, number>;
  /** True when model weights carry non-commercial license restrictions. */
  nonCommercial?: boolean;
  licenseNote: string;
  description: string;
  inputs: Record<string, PortSpec>;
  outputs: Record<string, PortSpec>;
  /** Optional parameter key storing the active mode (e.g. `mode`, `taskMode`, `msaMode`). */
  modeParamKey?: string;
  /** First-class modes supported by this adapter (drives preflight + UI). */
  modes?: AdapterModeSpec[];
  /** Declarative parameter schema (single source of truth for UI, validation & coverage tests). */
  paramSchema?: AdapterParamField[];
  /** Optional cross-field or cross-input validation hook run after declarative schema validation. */
  crossValidate?: (
    params: Record<string, unknown>,
    ctx?: { connectedInputs?: ReadonlySet<string> },
  ) => string[];
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
