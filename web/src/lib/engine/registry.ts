// Static registry of denovo.run node adapters + curated 1-click workflow presets.

import type { AdapterManifest, CampaignState, WorkflowGraph } from './types';

export const ADAPTER_REGISTRY: Record<string, AdapterManifest> = {
  'pdb-input': {
    id: 'pdb-input',
    displayName: 'Load Structure (PDB / mmCIF)',
    shortName: 'PDB Input',
    stage: 'source',
    runtime: 'cpu-ts',
    estimatedVramMB: 0,
    approxWeightsMB: 0,
    licenseNote: 'Built-in structure parser',
    description:
      'Start with the built-in 53-residue 3-helix bundle or upload/paste any PDB or mmCIF structure for sequence design and structure validation.',
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
      'Paste one or more amino acid sequences (plain text or FASTA) to fold directly with LocalFold without running an upstream backbone generator.',
    inputs: {},
    outputs: {
      sequences: { type: 'SequenceSet' },
    },
  },
  'rfd4-conditioner': {
    id: 'rfd4-conditioner',
    displayName: 'Design Conditioning (Hotspots / Motif / Fold)',
    shortName: 'Conditioning',
    stage: 'condition',
    runtime: 'cpu-ts',
    estimatedVramMB: 0,
    approxWeightsMB: 0,
    licenseNote: 'Model-agnostic constraint builder (RFD4-Proteina & Genie 3)',
    description:
      'Define target binding hotspots, fixed structural motifs (indexed or unindexed), hydrogen-bond contacts, CATH fold topology, or curated benchmark tasks for RFD4-Proteina and Genie 3.',
    inputs: {
      template: { type: 'StructureArtifact', required: false },
    },
    outputs: {
      condition: { type: 'ConditionSpec' },
    },
  },
  'rfd4-proteina': {
    id: 'rfd4-proteina',
    displayName: 'RFD4-Proteina',
    shortName: 'RFD4-Proteina',
    stage: 'generate',
    runtime: 'wgsl-custom',
    estimatedVramMB: 900,
    approxWeightsMB: 330,
    licenseNote: 'MIT code (sokrypton/rfd4-proteina-webgpu); NVIDIA RFD4-Proteina weights',
    description:
      'All-atom backbone and sequence co-design via flow matching (Baker Lab / NVIDIA). Supports unconditional monomers, conditioned design, partial diffusion, and inverse folding.',
    inputs: {
      template: { type: 'StructureArtifact', required: false },
      condition: { type: 'ConditionSpec', required: false },
    },
    outputs: {
      structure: { type: 'StructureArtifact' },
      sequences: { type: 'SequenceSet' },
    },
  },
  genie3: {
    id: 'genie3',
    displayName: 'Genie 3',
    shortName: 'Genie 3',
    stage: 'generate',
    runtime: 'wgsl-custom',
    estimatedVramMB: 220,
    approxWeightsMB: 45,
    licenseNote: 'Apache-2.0 (aqlaboratory/genie3 · WebGPU port ramithuh/genie3-webgpu)',
    description:
      'Fast de novo protein backbone diffusion (AlQuraishi Lab · 23.3M parameters). Supports unconditional generation, motif scaffolding, and PPI binder design with ideal N/Cα/C/O backbones and optional single-step sequence prediction.',
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
    displayName: 'ProteinMPNN',
    shortName: 'ProteinMPNN',
    stage: 'inverse-fold',
    runtime: 'ort-webgpu',
    estimatedVramMB: 120,
    approxWeightsMB: 10,
    licenseNote: 'MIT (MurrellGroup/webProteinMPNN & dauparas/ProteinMPNN)',
    description:
      'Structure-conditioned inverse folding supporting SolubleMPNN, ProteinMPNN, and Cα-only models with sampling temperature and fixed-residue controls.',
    inputs: {
      structure: { type: 'StructureArtifact' },
    },
    outputs: {
      sequences: { type: 'SequenceSet' },
    },
  },
  'localfold-af2': {
    id: 'localfold-af2',
    displayName: 'LocalFold',
    shortName: 'LocalFold',
    stage: 'fold',
    runtime: 'wgsl-custom',
    estimatedVramMB: 1400,
    approxWeightsMB: 97,
    licenseNote: 'Beerware engine (sokrypton/localfold); model weights fetched per-checkpoint from Hugging Face (AF2 & AF2-Multimer CC-BY-4.0, RF3 BSD-3, Boltz-2 MIT, OpenBind-0/Protenix-2/OpenDDE/Mol-1/IF2 Apache-2.0, AF3 Non-Commercial)',
    description:
      'In-browser protein structure prediction supporting AlphaFold2 Monomer (models 1–5), AlphaFold2-Multimer v3 (models 1–5), RoseTTAFold3, Boltz-2, OpenBind-0, Protenix-v2, OpenDDE, Mol-1, IntelliFold-2, and AlphaFold 3.',
    inputs: {
      sequences: { type: 'SequenceSet' },
    },
    outputs: {
      folds: { type: 'FoldedStructure' },
      structure: { type: 'StructureArtifact' },
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
    licenseNote: 'Built-in Kabsch superposition & geometry analysis',
    description:
      'Superimposes each predicted structure onto the designed backbone via exact Kabsch alignment and computes scRMSD (Å), scTM, mean pLDDT, pTM, radius of gyration (Rg), steric clashes, and pass rate.',
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
  { id: 'custom_cath_monomer', label: 'CATH 3.40.50 Rossmann Fold (54 aa, α/β)', category: 'Fold' },
  { id: 'uncond_short', label: 'Unconditional Monomer (100 aa)', category: 'Unconditional' },
  { id: 'motif_idx_1prw', label: 'Indexed Motif — 1PRW EF-Hand (76 aa)', category: 'Motif' },
  { id: 'motif_unidx_1prw', label: 'Unindexed Motif — 1PRW EF-Hand', category: 'Motif' },
  { id: 'ppi_pdl1', label: 'PD-L1 PPI Binder with Hotspots', category: 'Binder' },
  { id: 'sm_sam', label: 'SAM Small-Molecule Pocket with H-Bonds', category: 'Ligand' },
  { id: 'dna_7m5w', label: 'DNA-Binding Protein Complex (B-DNA)', category: 'Nucleic Acid' },
  { id: 'partial_diffusion_1prw', label: 'Partial Diffusion — 1PRW (50% noise)', category: 'Partial', needsEncoder: true },
  { id: 'inverse_folding_1prw', label: 'RFD4 Inverse Folding — 1PRW', category: 'InvFold', needsEncoder: true },
];

export interface MpnnVariantMeta {
  id: 'solublempnn' | 'proteinmpnn' | 'ca';
  family: 'solublempnn' | 'proteinmpnn' | 'ca';
  modelName: string;
  label: string;
  shortLabel: string;
  description: string;
  approxMB: number;
  license: string;
}

export const MPNN_VARIANT_OPTIONS: MpnnVariantMeta[] = [
  {
    id: 'solublempnn',
    family: 'solublempnn',
    modelName: 'v_48_020',
    label: 'SolubleMPNN',
    shortLabel: 'SolubleMPNN',
    description: 'Soluble-surface optimized sequence design (recommended)',
    approxMB: 10,
    license: 'MIT',
  },
  {
    id: 'proteinmpnn',
    family: 'proteinmpnn',
    modelName: 'v_48_020',
    label: 'ProteinMPNN',
    shortLabel: 'ProteinMPNN',
    description: 'Standard backbone-conditioned sequence design',
    approxMB: 10,
    license: 'MIT',
  },
  {
    id: 'ca',
    family: 'ca',
    modelName: 'v_48_020',
    label: 'ProteinMPNN (Cα-only)',
    shortLabel: 'Cα-ProteinMPNN',
    description: 'Sequence design from Cα backbone coordinates only',
    approxMB: 10,
    license: 'MIT',
  },
];

export interface LocalFoldModelMeta {
  id: string;
  label: string;
  shortLabel: string;
  family: 'af2' | 'af2-multimer' | 'af3';
  approxMB: number;
  license: string;
  nonCommercial?: boolean;
}

export const LOCALFOLD_MODEL_OPTIONS: LocalFoldModelMeta[] = [
  { id: 'monomer', label: 'AlphaFold2 · Model 1 (pTM)', shortLabel: 'AF2 Model 1', family: 'af2', approxMB: 97, license: 'CC-BY-4.0' },
  { id: 'monomer-2', label: 'AlphaFold2 · Model 2 (pTM)', shortLabel: 'AF2 Model 2', family: 'af2', approxMB: 140, license: 'CC-BY-4.0' },
  { id: 'monomer-3', label: 'AlphaFold2 · Model 3 (pTM, template-free)', shortLabel: 'AF2 Model 3', family: 'af2', approxMB: 140, license: 'CC-BY-4.0' },
  { id: 'monomer-4', label: 'AlphaFold2 · Model 4 (pTM, template-free)', shortLabel: 'AF2 Model 4', family: 'af2', approxMB: 140, license: 'CC-BY-4.0' },
  { id: 'monomer-5', label: 'AlphaFold2 · Model 5 (pTM, template-free)', shortLabel: 'AF2 Model 5', family: 'af2', approxMB: 140, license: 'CC-BY-4.0' },
  { id: 'multimer', label: 'AlphaFold2-Multimer v3 · Model 1', shortLabel: 'AF2-Multimer 1', family: 'af2-multimer', approxMB: 74, license: 'CC-BY-4.0' },
  { id: 'multimer-2', label: 'AlphaFold2-Multimer v3 · Model 2', shortLabel: 'AF2-Multimer 2', family: 'af2-multimer', approxMB: 118, license: 'CC-BY-4.0' },
  { id: 'multimer-3', label: 'AlphaFold2-Multimer v3 · Model 3', shortLabel: 'AF2-Multimer 3', family: 'af2-multimer', approxMB: 118, license: 'CC-BY-4.0' },
  { id: 'multimer-4', label: 'AlphaFold2-Multimer v3 · Model 4', shortLabel: 'AF2-Multimer 4', family: 'af2-multimer', approxMB: 118, license: 'CC-BY-4.0' },
  { id: 'multimer-5', label: 'AlphaFold2-Multimer v3 · Model 5', shortLabel: 'AF2-Multimer 5', family: 'af2-multimer', approxMB: 118, license: 'CC-BY-4.0' },
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
    name: 'RFD4-Proteina → LocalFold',
    badge: 'All-Atom Co-Design + Designability',
    summary:
      'Generate a de novo protein (backbone and sequence) with RFD4-Proteina, fold its native designed sequence with LocalFold (AlphaFold2), and score scRMSD.',
    graph: {
      id: 'rfd4-refold-native',
      name: 'RFD4-Proteina → LocalFold',
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
          label: 'LocalFold',
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
    id: 'genie3-mpnn-af2',
    name: 'Genie 3 → ProteinMPNN → LocalFold',
    badge: 'Fast Backbone Diffusion',
    summary:
      'Generate a de novo protein backbone with Genie 3 (23.3M parameters), design sequences with SolubleMPNN, fold with LocalFold, and score scRMSD and scTM.',
    graph: {
      id: 'genie3-mpnn-af2',
      name: 'Genie 3 → ProteinMPNN → LocalFold',
      nodes: [
        {
          id: 'gen',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: {
            length: 80,
            nSampleStep: 100,
            directionScale: 0.8,
            noiseScale: 1.0,
            predictSequence: true,
            seed: 42,
          },
        },
        {
          id: 'mpnn',
          adapterId: 'web-protein-mpnn',
          label: 'SolubleMPNN',
          params: { family: 'solublempnn', modelName: 'v_48_020', numSequences: 2, temperature: 0.1, seed: 42 },
        },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          label: 'LocalFold',
          params: { model: 'monomer', recycles: 1, maxSequences: 2, seed: 0 },
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
    id: 'pdb-refold-qc',
    name: 'PDB → ProteinMPNN → LocalFold',
    badge: 'Quick Start Pipeline',
    summary:
      'Start from the built-in 53-residue 3-helix bundle (or any uploaded PDB/mmCIF), design sequences with SolubleMPNN, fold with LocalFold, and inspect Kabsch superposition and scRMSD.',
    graph: {
      id: 'pdb-refold-qc',
      name: 'PDB → ProteinMPNN → LocalFold',
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
          label: 'LocalFold',
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
      'Generate a backbone with RFD4-Proteina, re-sequence with SolubleMPNN, fold with LocalFold, and evaluate Kabsch superposition and pLDDT.',
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
          label: 'LocalFold',
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
    badge: 'Branched Comparison',
    summary:
      'Fork one RFD4-Proteina backbone into two parallel evaluation branches: native co-designed sequence vs. SolubleMPNN re-sequencing, both folded by LocalFold.',
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
      'Compose RFD4 design constraints (CATH fold topology, binding hotspots, fixed motifs, or H-bond targets) into RFD4-Proteina and fold with LocalFold.',
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
          label: 'LocalFold',
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
    name: 'RFD4 Fork: AlphaFold2 vs RoseTTAFold3',
    badge: 'Cross-Folder Comparison',
    summary:
      'Generate one de novo protein with RFD4-Proteina, then cross-validate its native sequence with both AlphaFold2 (Model 1) and RoseTTAFold3 (RF3) in parallel branches.',
    graph: {
      id: 'rfd4-af2-vs-rf3',
      name: 'RFD4 Fork: AlphaFold2 vs RoseTTAFold3',
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
    targetName: 'De novo monomer (single-sequence folding)',
  },
  rounds: [
    {
      round: 1,
      timestamp: '2026-11-20T12:00:00Z',
      label: 'Round 1 — Unconditional baseline (T=0.2)',
      generator: 'RFD4-Proteina (40 steps)',
      folder: 'LocalFold · AlphaFold2 (single-sequence, 1 recycle)',
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
      generator: 'RFD4-Proteina (60 steps)',
      folder: 'LocalFold · AlphaFold2 (single-sequence, 2 recycles)',
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
      generator: 'RFD4-Proteina (60 steps)',
      folder: 'LocalFold · AlphaFold2 (single-sequence, 2 recycles)',
      nDesigned: 4,
      nPassed: 4,
      hitRate: 1.0,
      bestScRmsd: 0.84,
      bestPlddt: 91.2,
      conditioningSummary: '70-residue monomer (native co-designed sequence)',
      agentNotes:
        'All 4 candidates passed scRMSD < 1.5 Å and pLDDT > 88. Campaign goal met.',
    },
  ],
};
