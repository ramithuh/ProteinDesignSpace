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
      'Use a built-in 56 aa 3-helix bundle demo, paste/upload any PDB or mmCIF (e.g., from web.rfd4-proteina.org or RFdiffusion), and pipe it into inverse folding & refolding.',
    inputs: {},
    outputs: {
      structure: { type: 'StructureArtifact' },
    },
  },
  'legacy-teacher': {
    id: 'legacy-teacher',
    displayName: 'legacy-model Flow Matching (ONNX)',
    shortName: 'legacy-model',
    stage: 'generate',
    runtime: 'ort-webgpu',
    estimatedVramMB: 1400,
    approxWeightsMB: 855,
    licenseNote: 'Internal legacy-model-flow-map checkpoint (ONNX fp32)',
    description:
      'De novo monomer backbone flow-matching model running via ONNX Runtime Web (Euler-FM ODE or SDE-Proteina). Automatically reconstructs carbonyl O for ProteinMPNN.',
    inputs: {},
    outputs: {
      structure: { type: 'StructureArtifact' },
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
      'All-atom sequence-structure co-design flow matching (Baker Lab / NVIDIA) running via custom WGSL INT8 kernels. Supports unconditional monomers or YAML/JSON condition specs.',
    inputs: {
      template: { type: 'StructureArtifact', required: false },
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
    displayName: 'LocalFold AF2 (WebGPU INT5)',
    shortName: 'LocalFold AF2',
    stage: 'fold',
    runtime: 'wgsl-custom',
    estimatedVramMB: 1100,
    approxWeightsMB: 97,
    licenseNote: 'Beerware code (sokrypton/localfold); DeepMind AF2 CC-BY-4.0 weights',
    description:
      'Single-sequence AlphaFold2 monomer refolding on WebGPU (~97 MB INT5 shards from HF sokrypton/localfold). Produces all-atom PDB, pLDDT, pTM, and PAE locally with zero server upload.',
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

export interface PresetDefinition {
  id: string;
  name: string;
  badge: string;
  summary: string;
  graph: WorkflowGraph;
}

export const WORKFLOW_PRESETS: PresetDefinition[] = [
  {
    id: 'pdb-refold-qc',
    name: 'PDB → ProteinMPNN → LocalFold AF2',
    badge: 'Fastest Test (~107 MB)',
    summary:
      'Start from the built-in 56 aa 3-helix bundle (or drop any PDB/mmCIF), design sequences with SolubleMPNN, refold in WebGPU with LocalFold AF2, and inspect live Kabsch scRMSD + Mol* superposition.',
    graph: {
      id: 'pdb-refold-qc',
      name: 'PDB → ProteinMPNN → LocalFold AF2',
      nodes: [
        {
          id: 'src',
          adapterId: 'pdb-input',
          label: 'Backbone Input',
          params: { mode: 'demo-bundle', pdbText: '', filename: 'demo_3helix_56aa.pdb' },
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
    id: 'rfd4-refold-native',
    name: 'RFD4-Proteina → LocalFold AF2',
    badge: 'All-Atom Co-Design',
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
            nsteps: 30,
            recycleSchedule: 'low',
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
            nsteps: 30,
            recycleSchedule: 'low',
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
    id: 'legacy-model-mpnn-refold',
    name: 'legacy-model (ONNX) → ProteinMPNN → LocalFold',
    badge: 'Local legacy-model Checkpoint',
    summary:
      'Run your local legacy-teacher flow-matching model (Euler-FM or SDE-Proteina) with OPFS caching, reconstruct carbonyl O, design sequences with SolubleMPNN, and refold in LocalFold AF2.',
    graph: {
      id: 'legacy-model-mpnn-refold',
      name: 'legacy-model → ProteinMPNN → LocalFold',
      nodes: [
        {
          id: 'gen',
          adapterId: 'legacy-teacher',
          label: 'legacy-teacher',
          params: {
            modelUrl: '/models/legacy-teacher/v1/model.onnx',
            length: 80,
            nfe: 32,
            sampler: 'sde-proteina',
            schedule: 'log',
            scheduleP: 2.0,
            seed: 42,
            sdeGamma: 0.1,
            sdeGtMax: 3.0,
            sdeGtMode: 'one-minus-t-over-t',
            sdeTLimOde: 0.99,
            sdeNoiseSeed: 7,
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
      label: 'Round 1 — Exploratory ODE baseline',
      generator: 'legacy-teacher (Euler-FM, NFE=32)',
      folder: 'LocalFold AF2 (single-seq, 1 recycle)',
      nDesigned: 4,
      nPassed: 1,
      hitRate: 0.25,
      bestScRmsd: 1.92,
      bestPlddt: 78.4,
      conditioningSummary: 'Unconditional L=70, SolubleMPNN T=0.2',
      agentNotes:
        'Loops between helices 2–3 showed elevated flexibility (pLDDT ~62). Switching to SDE-Proteina (γ=0.1) and lowering MPNN temperature to 0.1 for tighter core packing.',
    },
    {
      round: 2,
      timestamp: '2026-11-20T12:14:00Z',
      label: 'Round 2 — SDE-Proteina + T=0.1 core tightening',
      generator: 'legacy-teacher (SDE-Proteina, γ=0.1)',
      folder: 'LocalFold AF2 (single-seq, 2 recycles)',
      nDesigned: 4,
      nPassed: 3,
      hitRate: 0.75,
      bestScRmsd: 1.18,
      bestPlddt: 86.7,
      conditioningSummary: 'Unconditional L=70, SolubleMPNN T=0.1',
      agentNotes:
        'Hit rate jumped from 25% → 75%. Testing RFD4-Proteina co-design in Round 3 to compare native co-designed sequences against SolubleMPNN re-sequencing.',
    },
    {
      round: 3,
      timestamp: '2026-11-20T12:28:00Z',
      label: 'Round 3 — RFD4-Proteina all-atom co-design',
      generator: 'RFD4-Proteina (WebGPU INT8, 30 steps)',
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
