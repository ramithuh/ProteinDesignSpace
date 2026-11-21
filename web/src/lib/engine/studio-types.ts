import type { AtomGranularity, BioCompatibilityFix } from './types';

export type StudioBrushRole = 'fixed' | 'hotspot' | 'motif' | 'tied' | 'crop';

export type ResidueRole =
  | 'designable'
  | 'fixed-sequence'
  | 'fixed-backbone'
  | 'motif'
  | 'unindexed-motif'
  | 'hotspot'
  | 'tied'
  | 'cropped-out'
  | 'missing-backbone'
  | 'ligand'
  | 'nucleic';

export interface PreviewResidue {
  /** 0-based flat index in the model input sequence/coordinates (or -1 if cropped out). */
  flatIndex: number;
  /** Original PDB/mmCIF auth_asym_id (e.g. 'A', 'H'). */
  authChain: string;
  /** Remapped internal model chain ID if renamed (e.g. Genie 3 'B' or RFD4 label_asym_id). */
  labelChain?: string;
  /** Original PDB auth_seq_id (supports negative numbers). */
  authSeqId: number;
  /** PDB insertion code ('' or 'A'..'Z'). */
  insCode: string;
  /** 1-based remapped model label_seq_id if remapped. */
  labelSeqId?: number;
  /** Canonical writeback token preserving insertion codes, e.g. 'A54', 'H100A', 'A-2'. */
  token: string;
  /** Single-letter amino acid ('A'..'Y', 'X', or '-' for de novo placeholder). */
  aa1: string;
  /** Three-letter residue name ('ALA', 'MSE', 'DA', etc.). */
  resName: string;
  /** Ordered semantic roles at this residue position. */
  roles: ResidueRole[];
  /** Symmetry tie group index (1-based) when role includes 'tied'. */
  tieGroupId?: number;
  /** True when lock/fixed status was inherited from an upstream generator (lockedChains / fixedResidueIndices). */
  inheritedLock?: boolean;
  /** Number of missing sequence positions immediately before this residue (chain break / disordered loop). */
  gapBefore?: number;
  /** Per-residue scalar (pLDDT, crystal B-factor, ΔΔG, etc.) if available. */
  bfactor?: number;
  /** Compact backbone atom availability string when known (e.g. 'N/CA/C/O' or 'CA-only'). */
  backboneAtoms?: string;
}

export interface PreviewChainTrack {
  chainId: string;
  labelChainId?: string;
  entityKind: 'protein' | 'dna' | 'rna' | 'ligand';
  roleTag?: 'target' | 'binder' | 'scaffold' | 'designed-span';
  sequence: string;
  residues: PreviewResidue[];
  msaStatus?: 'single' | 'mmseqs2-search' | 'target-only-masked' | 'a3m-provided';
  msaDepth?: number | null;
}

export interface CompiledModelPayloadView {
  format:
    | 'rfd4-spec'
    | 'genie3-spec'
    | 'mpnn-masks'
    | 'localfold-batch'
    | 'scorer-mapping'
    | 'raw-input';
  headline: string;
  compiledSummary: Record<string, string | number | boolean | string[]>;
  rawSpecText: string;
}

export interface PreviewIssue {
  severity: 'error' | 'warning';
  source: 'schema' | 'bio' | 'runtime' | 'compiler';
  message: string;
  fix?: BioCompatibilityFix;
}

export interface ResidueHighlightLayer {
  id: string;
  role: ResidueRole;
  label: string;
  colorHex: number;
  /** Tokens of the form `${authChain}${authSeqId}${insCode}` (e.g. 'A54', 'H100A'). */
  tokens: string[];
  /** Optional structured residue selectors for exact Mol* matching. */
  residues: Array<{ chain: string; seqId: number; insCode?: string }>;
  /** Render sparse sidechain ball-and-stick accent when residue count is small (<= 40). */
  accentSticks?: boolean;
  /** Optional alpha override (e.g. 0.25 for cropped-out ghosting). */
  alpha?: number;
}

export interface FastaPreviewEntry {
  id: string;
  header: string;
  sequence: string;
  chainLengths?: number[];
  selected?: boolean;
  scoreLabel?: string;
}

export interface NodeInputPreview {
  nodeId: string;
  adapterId: string;
  availability: 'ready' | 'partial' | 'pending-upstream' | 'error';
  missingUpstreamNodeIds: string[];
  chains: PreviewChainTrack[];
  entities: {
    includedLigands: string[];
    filteredBuffers: string[];
    nucleicChains: string[];
  };
  fastaEntries: FastaPreviewEntry[];
  previewStructurePdb: string | null;
  previewStructureMeta?: {
    label: string;
    atomGranularity?: AtomGranularity;
    length: number;
  };
  /** Derived 3D highlight layers for zero-reload Mol* overpaint. */
  highlightLayers: ResidueHighlightLayer[];
  payload: CompiledModelPayloadView;
  budget: {
    totalResidues: number;
    designedResidues: number;
    fixedResidues: number;
    estimatedTokens: number;
    nodeWeightsMB: number;
    nodeVramMB: number;
    subgraphDownloadMB: number;
    subgraphPeakVramMB: number;
  };
  issues: PreviewIssue[];
  previewDigest: string;
}
