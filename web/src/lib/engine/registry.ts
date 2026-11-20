// Static registry of denovo.run node adapters + curated 1-click workflow presets.

import type { AdapterManifest, CampaignState, WorkflowGraph } from './types';

export const ADAPTER_REGISTRY: Record<string, AdapterManifest> = {
  'pdb-input': {
    id: 'pdb-input',
    displayName: 'Load PDB / mmCIF',
    shortName: 'PDB Input',
    stage: 'source',
    runtime: 'cpu-ts',
    estimatedVramMB: 0,
    approxWeightsMB: 0,
    licenseNote: 'Built-in parser',
    description:
      'Use a built-in 53 aa 3-helix bundle demo, paste/upload any PDB or mmCIF (e.g., from web.rfd4-proteina.org or RFdiffusion), and pipe it into inverse folding & refolding.',
    inputs: {},
    outputs: {
      structure: { type: 'StructureArtifact' },
    },
  },
  'sequence-input': {
    id: 'sequence-input',
    displayName: 'Sequence Input (FASTA)',
    shortName: 'Sequence Input',
    stage: 'source',
    runtime: 'cpu-ts',
    estimatedVramMB: 0,
    approxWeightsMB: 0,
    licenseNote: 'Built-in sequence parser',
    description:
      'Paste one or more amino acid sequences (plain or FASTA) to fold directly with LocalFold AF2 without running an upstream generator.',
    inputs: {},
    outputs: {
      sequences: { type: 'SequenceSet' },
    },
  },
  'rfd4-conditioner': {
    id: 'rfd4-conditioner',
    displayName: 'RFD4 Conditioning (Hotspots / Motif / H-Bond)',
    shortName: 'RFD4 Condition',
    stage: 'condition',
    runtime: 'cpu-ts',
    estimatedVramMB: 0,
    approxWeightsMB: 0,
    licenseNote: 'RFD4-Proteina condition_spec DSL compiler',
    description:
      'Compose RFD4 contigs, PPI hotspots (C_HOT), indexed/unindexed motifs (C_CRDGRP/C_IDX), H-bond targets (C_HB), CATH folds, or 1-click official RFD4 benchmark contexts.',
    inputs: {
      template: { type: 'StructureArtifact', required: false },
    },
    outputs: {
      condition: { type: 'ConditionSpec' },
    },
  },
  'rfd4-proteina': {
    id: 'rfd4-proteina',
    displayName: 'RFD4-Proteina (WebGPU INT8)',
    shortName: 'RFD4-Proteina',
    stage: 'generate',
    runtime: 'wgsl-custom',
    estimatedVramMB: 900,
    approxWeightsMB: 330,
    licenseNote: 'MIT code (sokrypton/rfd4-proteina-webgpu); NVIDIA RFD4-Proteina weights',
    description:
      'All-atom sequence-structure co-design flow matching (Baker Lab / NVIDIA) running via custom WGSL INT8 kernels. Supports unconditional monomers, partial diffusion, inverse folding, and full AtomWorks/RFD4 conditioning.',
    inputs: {
      template: { type: 'StructureArtifact', required: false },
      condition: { type: 'ConditionSpec', required: false },
    },
    outputs: {
      structure: { type: 'StructureArtifact' },
      sequences: { type: 'SequenceSet' },
    },
  },
  'web-protein-mpnn': {
    id: 'web-protein-mpnn',
    displayName: 'ProteinMPNN / SolubleMPNN (ONNX)',
    shortName: 'ProteinMPNN',
    stage: 'inverse-fold',
    runtime: 'ort-webgpu',
    estimatedVramMB: 120,
    approxWeightsMB: 10,
    licenseNote: 'MIT (MurrellGroup/webProteinMPNN & dauparas/ProteinMPNN)',
    description:
      'Client-side inverse folding via ONNX Runtime Web (~9.5 MB from HF MurrellLab/webports). Supports ProteinMPNN, SolubleMPNN, and CA-only models with temperature & fixed-position controls.',
    inputs: {
      structure: { type: 'StructureArtifact' },
    },
    outputs: {
      sequences: { type: 'SequenceSet' },
    },
  },
  'localfold-af2': {
    id: 'localfold-af2',
    displayName: 'LocalFold (AF2 / RF3 / Boltz-2 / AF3)',
    shortName: 'LocalFold',
    stage: 'fold',
    runtime: 'wgsl-custom',
    estimatedVramMB: 1400,
    approxWeightsMB: 97,
    licenseNote: 'Beerware engine (sokrypton/localfold); model weights fetched per-checkpoint from Hugging Face (AF2 CC-BY-4.0, RF3 BSD-3, Boltz-2 MIT, OpenBind-0/Protenix-2/OpenDDE/Mol-1/IF2 Apache-2.0, AF3 Non-Commercial)',
    description:
      'Client-side WebGPU structure prediction powered by sokrypton/localfold. Choose AlphaFold2 (models 1–5), RoseTTAFold3 (RF3), Boltz-2, OpenBind-0, Protenix-v2, OpenDDE, Mol-1, IntelliFold-2, or AlphaFold 3.',
    inputs: {
      sequences: { type: 'SequenceSet' },
    },
    outputs: {
      folds: { type: 'FoldedStructure' },
    },
  },
  'designability-scorer': {
    id: 'designability-scorer',
    displayName: 'Designability Scorer (Kabsch)',
    shortName: 'scRMSD / scTM',
    stage: 'score',
    runtime: 'cpu-ts',
    estimatedVramMB: 0,
    approxWeightsMB: 0,
    licenseNote: 'Built-in pure-TS Kabsch & geometry engine',
    description:
      'Aligns every refolded structure onto the designed backbone via exact Kabsch SVD superposition and computes scRMSD (Å), scTM, mean pLDDT, pTM, Rg, clashes, and pass rate.',
    inputs: {
      structure: { type: 'StructureArtifact' },
      folds: { type: 'FoldedStructure' },
    },
    outputs: {
      metrics: { type: 'DesignabilityMetrics' },
    },
  },
};

export interface Rfd4BenchmarkMeta {
  id: string;
  label: string;
  category: string;
  needsEncoder?: boolean;
}

export const RFD4_BENCHMARK_OPTIONS: Rfd4BenchmarkMeta[] = [
  { id: 'custom_cath_monomer', label: 'CATH 3.40.50 Rossmann (54 aa, α/β SS)', category: 'Fold' },
  { id: 'uncond_short', label: 'Unconditional Monomer (100 aa)', category: 'Unconditional' },
  { id: 'motif_idx_1prw', label: 'Indexed Motif — 1PRW EF-Hand (76 aa)', category: 'Motif' },
  { id: 'motif_unidx_1prw', label: 'Unindexed Motif — 1PRW + Hungarian Insertion', category: 'Motif' },
  { id: 'ppi_pdl1', label: 'PD-L1 PPI Binder w/ Hotspots (208 tokens)', category: 'Binder' },
  { id: 'sm_sam', label: 'SAM Small-Molecule Pocket w/ H-Bonds', category: 'Ligand' },
  { id: 'dna_7m5w', label: 'DNA-Binding Protein Complex (B-DNA)', category: 'Nucleic Acid' },
  { id: 'partial_diffusion_1prw', label: 'Partial Diffusion — 1PRW (frac=0.5, +138 MB enc)', category: 'Partial', needsEncoder: true },
  { id: 'inverse_folding_1prw', label: 'RFD4 Inverse Folding — 1PRW (+138 MB enc)', category: 'InvFold', needsEncoder: true },
];

export interface LocalFoldModelMeta {
  id: string;
  label: string;
  shortLabel: string;
  family: 'af2' | 'af3';
  approxMB: number;
  license: string;
  nonCommercial?: boolean;
}

export const LOCALFOLD_MODEL_OPTIONS: LocalFoldModelMeta[] = [
  { id: 'monomer', label: 'AlphaFold2 · model_1_ptm', shortLabel: 'AF2 m1_ptm', family: 'af2', approxMB: 97, license: 'CC-BY-4.0' },
  { id: 'monomer-2', label: 'AlphaFold2 · model_2_ptm (delta)', shortLabel: 'AF2 m2_ptm', family: 'af2', approxMB: 140, license: 'CC-BY-4.0' },
  { id: 'monomer-3', label: 'AlphaFold2 · model_3_ptm (template-free)', shortLabel: 'AF2 m3_ptm', family: 'af2', approxMB: 140, license: 'CC-BY-4.0' },
  { id: 'monomer-4', label: 'AlphaFold2 · model_4_ptm (template-free)', shortLabel: 'AF2 m4_ptm', family: 'af2', approxMB: 140, license: 'CC-BY-4.0' },
  { id: 'monomer-5', label: 'AlphaFold2 · model_5_ptm (template-free)', shortLabel: 'AF2 m5_ptm', family: 'af2', approxMB: 140, license: 'CC-BY-4.0' },
  { id: 'rosettafold3', label: 'RoseTTAFold3 (RF3)', shortLabel: 'RF3', family: 'af3', approxMB: 266, license: 'BSD-3-Clause' },
  { id: 'boltz2', label: 'Boltz-2', shortLabel: 'Boltz-2', family: 'af3', approxMB: 364, license: 'MIT' },
  { id: 'openbind0', label: 'OpenBind-0 (OpenFold3 v0.5.0)', shortLabel: 'OpenBind-0', family: 'af3', approxMB: 265, license: 'Apache-2.0' },
  { id: 'protenix2', label: 'Protenix-v2', shortLabel: 'Protenix-v2', family: 'af3', approxMB: 334, license: 'Apache-2.0' },
  { id: 'opendde', label: 'OpenDDE', shortLabel: 'OpenDDE', family: 'af3', approxMB: 473, license: 'Apache-2.0' },
  { id: 'mol1', label: 'Mol-1', shortLabel: 'Mol-1', family: 'af3', approxMB: 510, license: 'Apache-2.0' },
  { id: 'intellifold2', label: 'IntelliFold-2', shortLabel: 'IntelliFold-2', family: 'af3', approxMB: 612, license: 'Apache-2.0' },
  { id: 'af3', label: 'AlphaFold 3 (Non-Commercial)', shortLabel: 'AF3', family: 'af3', approxMB: 265, license: 'AF3 Non-Commercial', nonCommercial: true },
];

export interface PresetDefinition {
  id: string;
  name: string;
  badge: string;
  summary: string;
  graph: WorkflowGraph;
}

export const WORKFLOW_PRESETS: PresetDefinition[] = [
  {
    id: 'rfd4-refold-native',
    name: 'RFD4-Proteina → LocalFold AF2',
    badge: 'All-Atom Co-Design + Designability',
    summary:
      'Generate a de novo protein (sequence + structure) on WebGPU with RFD4-Proteina INT8 (~330 MB), then immediately refold its native designed sequence with LocalFold AF2 and measure scRMSD.',
    graph: {
      id: 'rfd4-refold-native',
      name: 'RFD4-Proteina → LocalFold AF2',
      nodes: [
        {
          id: 'gen',
          adapterId: 'rfd4-proteina',
          label: 'RFD4-Proteina',
          params: {
            length: 70,
            nsteps: 60,
            recycleSchedule: 'low',
            earlyStop: true,
            seed: 42,
            customSpecYaml: '',
          },
        },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          label: 'LocalFold AF2',
          params: { recycles: 1, maxSequences: 1, seed: 0 },
        },
        {
          id: 'score',
          adapterId: 'designability-scorer',
          label: 'scRMSD / scTM',
          params: { scRmsdThreshold: 2.0, plddtThreshold: 70 },
        },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    },
  },
  {
    id: 'pdb-refold-qc',
    name: 'PDB → ProteinMPNN → LocalFold AF2',
    badge: 'Fastest Test (~107 MB)',
    summary:
      'Start from the built-in 53 aa 3-helix bundle (or drop any PDB/mmCIF), design sequences with SolubleMPNN, refold in WebGPU with LocalFold AF2, and inspect live Kabsch scRMSD + Mol* superposition.',
    graph: {
      id: 'pdb-refold-qc',
      name: 'PDB → ProteinMPNN → LocalFold AF2',
      nodes: [
        {
          id: 'src',
          adapterId: 'pdb-input',
          label: 'Backbone Input',
          params: { mode: 'demo-bundle', pdbText: '', filename: 'demo_3helix_53aa.pdb' },
        },
        {
          id: 'mpnn',
          adapterId: 'web-protein-mpnn',
          label: 'SolubleMPNN',
          params: {
            family: 'solublempnn',
            modelName: 'v_48_020',
            numSequences: 2,
            temperature: 0.1,
            seed: 42,
            fixedPositions: '',
          },
        },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          label: 'LocalFold AF2',
          params: { recycles: 1, maxSequences: 2, seed: 0 },
        },
        {
          id: 'score',
          adapterId: 'designability-scorer',
          label: 'scRMSD / scTM',
          params: { scRmsdThreshold: 2.0, plddtThreshold: 70 },
        },
      ],
      edges: [
        { fromNode: 'src', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    },
  },
  {
    id: 'rfd4-mpnn-refold',
    name: 'RFD4-Proteina → SolubleMPNN → LocalFold',
    badge: 'Full 3-Model Loop',
    summary:
      'Generate backbone with RFD4-Proteina, re-sequence with SolubleMPNN, refold with LocalFold AF2, and compare Kabsch superposition & pLDDT — all 3 heterogeneous WebGPU runtimes in one sequential pipeline.',
    graph: {
      id: 'rfd4-mpnn-refold',
      name: 'RFD4-Proteina → SolubleMPNN → LocalFold',
      nodes: [
        {
          id: 'gen',
          adapterId: 'rfd4-proteina',
          label: 'RFD4-Proteina',
          params: {
            length: 70,
            nsteps: 60,
            recycleSchedule: 'low',
            earlyStop: true,
            seed: 42,
            customSpecYaml: '',
          },
        },
        {
          id: 'mpnn',
          adapterId: 'web-protein-mpnn',
          label: 'SolubleMPNN',
          params: {
            family: 'solublempnn',
            modelName: 'v_48_020',
            numSequences: 2,
            temperature: 0.1,
            seed: 42,
            fixedPositions: '',
          },
        },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          label: 'LocalFold AF2',
          params: { recycles: 1, maxSequences: 2, seed: 0 },
        },
        {
          id: 'score',
          adapterId: 'designability-scorer',
          label: 'scRMSD / scTM',
          params: { scRmsdThreshold: 2.0, plddtThreshold: 70 },
        },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    },
  },
  {
    id: 'rfd4-branch-compare',
    name: 'RFD4 Fork: Native vs SolubleMPNN',
    badge: 'Branched DAG',
    summary:
      'Fork one RFD4-Proteina backbone into two parallel scoring arms: native co-designed sequence vs. SolubleMPNN re-sequencing, both refolded by LocalFold AF2.',
    graph: {
      id: 'rfd4-branch-compare',
      name: 'RFD4 Fork: Native vs SolubleMPNN',
      nodes: [
        {
          id: 'gen',
          adapterId: 'rfd4-proteina',
          label: 'RFD4-Proteina',
          params: { length: 65, nsteps: 60, recycleSchedule: 'low', earlyStop: true, seed: 42, customSpecYaml: '' },
          position: { x: 40, y: 24 },
        },
        {
          id: 'mpnn',
          adapterId: 'web-protein-mpnn',
          label: 'SolubleMPNN',
          params: { family: 'solublempnn', modelName: 'v_48_020', numSequences: 2, temperature: 0.1, seed: 42, fixedPositions: '' },
          position: { x: 340, y: 194 },
        },
        {
          id: 'fold_mpnn',
          adapterId: 'localfold-af2',
          label: 'LocalFold (MPNN)',
          params: { recycles: 1, maxSequences: 2, seed: 0 },
          position: { x: 640, y: 194 },
        },
        {
          id: 'score_mpnn',
          adapterId: 'designability-scorer',
          label: 'Score (MPNN)',
          params: { scRmsdThreshold: 2.0, plddtThreshold: 70 },
          position: { x: 940, y: 24 },
        },
        {
          id: 'fold_native',
          adapterId: 'localfold-af2',
          label: 'LocalFold (Native)',
          params: { recycles: 1, maxSequences: 1, seed: 0 },
          position: { x: 340, y: 368 },
        },
        {
          id: 'score_native',
          adapterId: 'designability-scorer',
          label: 'Score (Native)',
          params: { scRmsdThreshold: 2.0, plddtThreshold: 70 },
          position: { x: 640, y: 344 },
        },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold_native', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score_native', toPort: 'structure' },
        { fromNode: 'fold_native', fromPort: 'folds', toNode: 'score_native', toPort: 'folds' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold_mpnn', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score_mpnn', toPort: 'structure' },
        { fromNode: 'fold_mpnn', fromPort: 'folds', toNode: 'score_mpnn', toPort: 'folds' },
      ],
    },
  },
  {
    id: 'rfd4-conditioned-cath',
    name: 'Conditioned RFD4 (CATH / Motif / Hotspots) → LocalFold',
    badge: 'Conditioning Block',
    summary:
      'Compose RFD4 conditioning (CATH fold + α/β secondary structure, or switch to PD-L1 hotspots / 1PRW motif / SAM H-bond pocket) into RFD4-Proteina and refold with LocalFold AF2.',
    graph: {
      id: 'rfd4-conditioned-cath',
      name: 'Conditioned RFD4 → LocalFold',
      nodes: [
        {
          id: 'cond',
          adapterId: 'rfd4-conditioner',
          label: 'RFD4 Condition',
          params: {
            taskPreset: 'custom_cath_monomer',
            contig: 'h1:16-16P,l1:4-4P,s1:8-8P,l2:4-4P,h2:14-14P,s2:8-8P',
            lockChain: '',
            hotspots: '',
            motifSpan: '',
            unindexedMotif: false,
            hbondMode: 'none',
            hbondSelection: '',
            cathCode: '3.40.50',
            paintBrush: 'hotspots',
            rawJsonOverride: '',
          },
          position: { x: 40, y: 48 },
        },
        {
          id: 'gen',
          adapterId: 'rfd4-proteina',
          label: 'RFD4-Proteina',
          params: {
            mode: 'conditioned',
            length: 70,
            nsteps: 60,
            recycleSchedule: 'low',
            earlyStop: true,
            seed: 42,
            presetId: 'custom_cath_monomer',
            partialFraction: 0.5,
            customSpecYaml: '',
          },
          position: { x: 340, y: 24 },
        },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          label: 'LocalFold AF2',
          params: { recycles: 1, maxSequences: 1, seed: 0 },
          position: { x: 640, y: 194 },
        },
        {
          id: 'score',
          adapterId: 'designability-scorer',
          label: 'scRMSD / scTM',
          params: { scRmsdThreshold: 2.0, plddtThreshold: 70 },
          position: { x: 940, y: 24 },
        },
      ],
      edges: [
        { fromNode: 'cond', fromPort: 'condition', toNode: 'gen', toPort: 'condition' },
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    },
  },
  {
    id: 'rfd4-af2-vs-rf3',
    name: 'RFD4 Fork: AF2 vs RoseTTAFold3',
    badge: 'Cross-Folder Check',
    summary:
      'Generate one de novo protein with RFD4-Proteina, then cross-validate its native sequence with both AlphaFold2 (model_1_ptm) and RoseTTAFold3 (RF3) in parallel branches.',
    graph: {
      id: 'rfd4-af2-vs-rf3',
      name: 'RFD4 Fork: AF2 vs RoseTTAFold3',
      nodes: [
        {
          id: 'gen',
          adapterId: 'rfd4-proteina',
          label: 'RFD4-Proteina',
          params: { length: 65, nsteps: 60, recycleSchedule: 'low', earlyStop: true, seed: 42, customSpecYaml: '' },
          position: { x: 40, y: 120 },
        },
        {
          id: 'fold_af2',
          adapterId: 'localfold-af2',
          label: 'LocalFold (AF2)',
          params: { model: 'monomer', recycles: 1, diffusionSteps: 25, maxSequences: 1, seed: 0 },
          position: { x: 350, y: 24 },
        },
        {
          id: 'score_af2',
          adapterId: 'designability-scorer',
          label: 'Score (AF2)',
          params: { scRmsdThreshold: 2.0, plddtThreshold: 70 },
          position: { x: 660, y: 24 },
        },
        {
          id: 'fold_rf3',
          adapterId: 'localfold-af2',
          label: 'LocalFold (RF3)',
          params: { model: 'rosettafold3', recycles: 1, diffusionSteps: 25, maxSequences: 1, seed: 0 },
          position: { x: 350, y: 230 },
        },
        {
          id: 'score_rf3',
          adapterId: 'designability-scorer',
          label: 'Score (RF3)',
          params: { scRmsdThreshold: 2.0, plddtThreshold: 70 },
          position: { x: 660, y: 230 },
        },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold_af2', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score_af2', toPort: 'structure' },
        { fromNode: 'fold_af2', fromPort: 'folds', toNode: 'score_af2', toPort: 'folds' },
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold_rf3', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score_rf3', toPort: 'structure' },
        { fromNode: 'fold_rf3', fromPort: 'folds', toNode: 'score_rf3', toPort: 'folds' },
      ],
    },
  },
];

/** Sample multi-round BindClaw-style campaign for demonstrating the Campaign Inspector tab. */
export const SAMPLE_CAMPAIGN: CampaignState = {
  schema: 'denovo.campaign/v1',
  campaignId: 'demo-bindclaw-campaign-01',
  title: 'Example Multi-Round Campaign (BindClaw-style)',
  goal: 'Design a compact 60–80 aa de novo α-helical scaffold with scRMSD < 1.5 Å and pLDDT > 85 via iterative generator + MPNN tuning.',
  assay: {
    scRmsdThreshold: 2.0,
    plddtThreshold: 70,
    targetName: 'De novo monomer (single-seq AF2)',
  },
  rounds: [
    {
      round: 1,
      timestamp: '2026-11-20T12:00:00Z',
      label: 'Round 1 — Unconditional baseline (T=0.2)',
      generator: 'RFD4-Proteina (WebGPU INT8, 40 steps)',
      folder: 'LocalFold AF2 (single-seq, 1 recycle)',
      nDesigned: 4,
      nPassed: 1,
      hitRate: 0.25,
      bestScRmsd: 1.92,
      bestPlddt: 78.4,
      conditioningSummary: 'Unconditional L=70, SolubleMPNN T=0.2',
      agentNotes:
        'Loops between helices 2–3 showed elevated flexibility (pLDDT ~62). Increasing diffusion steps to 60 and lowering SolubleMPNN temperature to 0.1 for tighter core packing.',
    },
    {
      round: 2,
      timestamp: '2026-11-20T12:14:00Z',
      label: 'Round 2 — SolubleMPNN T=0.1 core tightening',
      generator: 'RFD4-Proteina (WebGPU INT8, 60 steps)',
      folder: 'LocalFold AF2 (single-seq, 2 recycles)',
      nDesigned: 4,
      nPassed: 3,
      hitRate: 0.75,
      bestScRmsd: 1.18,
      bestPlddt: 86.7,
      conditioningSummary: 'Unconditional L=70, SolubleMPNN T=0.1',
      agentNotes:
        'Hit rate jumped from 25% → 75%. Testing RFD4-Proteina native co-design in Round 3 to compare native co-designed sequences against SolubleMPNN re-sequencing.',
    },
    {
      round: 3,
      timestamp: '2026-11-20T12:28:00Z',
      label: 'Round 3 — RFD4-Proteina all-atom co-design',
      generator: 'RFD4-Proteina (WebGPU INT8, 60 steps)',
      folder: 'LocalFold AF2 (single-seq, 2 recycles)',
      nDesigned: 4,
      nPassed: 4,
      hitRate: 1.0,
      bestScRmsd: 0.84,
      bestPlddt: 91.2,
      conditioningSummary: 'contig: 70-70 (native co-designed sequence)',
      agentNotes:
        'All 4 candidates passed scRMSD < 1.5 Å and pLDDT > 88. Campaign goal met.',
    },
  ],
};
