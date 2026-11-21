// Scientific Model Card & Precision/Parity Disclosure registry for denovo.run.
// Every numerical parity value, bundle size, commit SHA, and HF revision below is sourced
// directly from measured repository artifacts and upstream manifests — never estimated or fabricated.

import { LOCALFOLD_MODEL_OPTIONS, MPNN_VARIANT_OPTIONS } from './registry';

export interface ParityMeasurement {
  /** Measured quantity and comparison basis (e.g. "WebGPU ΔΔG relRMS vs PyTorch FP32"). */
  metric: string;
  /** Exact recorded numerical value or honest upstream-port disclosure string. */
  value: string;
  /** Optional recorded maximum absolute difference (maxAbsDiff). */
  maxAbsDiff?: string;
  /** Exact fixture, sequence length, and test scope for the measurement. */
  scope: string;
  /** Optional hardware / browser backend where parity was measured. */
  hardware?: string;
}

export interface PrecisionTierSpec {
  id: string;
  label: string;
  quantizationDetail: string;
  approxSizeMB: number;
  parityVsPytorchFp32: ParityMeasurement | null;
  latencyHint?: string;
  caveat?: string;
}

export interface ModelCardSpec {
  id: string;
  adapterId?: string;
  displayName: string;
  status: 'live' | 'weights-hosted';
  activePrecisionId: string;
  precisions: PrecisionTierSpec[];
  upstreamRepo: string;
  upstreamCommit?: string;
  weightsRepo?: string;
  weightsRevision?: string;
  license: string;
  citation: string;
  limitations: string[];
}

export const HOSTED_PORT_CARD_IDS = [
  'thermompnn-webgpu',
  'ligandmpnn-webgpu',
  'lasermpnn-webgpu',
  'antifold-esmif1-webgpu',
  'esmc-webgpu',
  'dyna1-webgpu',
] as const;

interface LocalFoldCheckpointMeta {
  weightsRevision: string;
  quantizationDetail: string;
  familyNote: string;
}

const LOCALFOLD_CHECKPOINT_META: Record<string, LocalFoldCheckpointMeta> = {
  monomer: {
    weightsRevision: '8a8aa55554b7',
    quantizationDetail: '5-bit (INT5) quantized AlphaFold2 Model 1 (pTM) weights (~97 MB)',
    familyNote:
      'AF2 Monomer Model 1 (CC-BY-4.0): protein-only monomer predictor with template embedder; multi-chain ipTM from monomer weights is an approximation.',
  },
  'monomer-2': {
    weightsRevision: '8a8aa55554b7',
    quantizationDetail: 'INT5 base Model 1 + ~43 MB delta bundle (~140 MB combined; within 0.01 Å of standalone on 5CAJ)',
    familyNote:
      'AF2 Monomer Model 2 (CC-BY-4.0): loaded as a delta on Model 1; protein-only monomer predictor.',
  },
  'monomer-3': {
    weightsRevision: '8a8aa55554b7',
    quantizationDetail: 'INT5 base Model 1 + ~43 MB template-free delta bundle (~140 MB combined)',
    familyNote:
      'AF2 Monomer Model 3 (CC-BY-4.0): template-free checkpoint (67 template embedder tensors absent); protein-only.',
  },
  'monomer-4': {
    weightsRevision: '8a8aa55554b7',
    quantizationDetail: 'INT5 base Model 1 + ~43 MB template-free delta bundle (~140 MB combined)',
    familyNote:
      'AF2 Monomer Model 4 (CC-BY-4.0): template-free checkpoint; LocalFold avoids 2-bit deltas because they cost ~24 pLDDT points in single-sequence mode.',
  },
  'monomer-5': {
    weightsRevision: '8a8aa55554b7',
    quantizationDetail: 'INT5 base Model 1 + ~43 MB template-free delta bundle (~140 MB combined)',
    familyNote:
      'AF2 Monomer Model 5 (CC-BY-4.0): template-free checkpoint; protein-only monomer predictor.',
  },
  multimer: {
    weightsRevision: 'b61b5a3ed564',
    quantizationDetail: '5-bit (INT5) quantized AlphaFold2-Multimer v3 Model 1 weights (~74 MB)',
    familyNote:
      'AF2-Multimer v3 Model 1 (CC-BY-4.0): native protein-protein complex predictor (ipTM + chain-pair pTM); protein chains only (no ligands or nucleic acids).',
  },
  'multimer-2': {
    weightsRevision: 'e960ea140c77',
    quantizationDetail: 'INT5 Multimer Model 1 + ~44 MB delta bundle (~118 MB combined)',
    familyNote:
      'AF2-Multimer v3 Model 2 (CC-BY-4.0): protein complex predictor loaded as a delta on Multimer Model 1; no ligand or nucleic acid support.',
  },
  'multimer-3': {
    weightsRevision: 'e960ea140c77',
    quantizationDetail: 'INT5 Multimer Model 1 + ~44 MB delta bundle (~118 MB combined)',
    familyNote:
      'AF2-Multimer v3 Model 3 (CC-BY-4.0): protein complex predictor loaded as a delta on Multimer Model 1; no ligand or nucleic acid support.',
  },
  'multimer-4': {
    weightsRevision: 'e960ea140c77',
    quantizationDetail: 'INT5 Multimer Model 1 + ~44 MB delta bundle (~118 MB combined)',
    familyNote:
      'AF2-Multimer v3 Model 4 (CC-BY-4.0): protein complex predictor loaded as a delta on Multimer Model 1; no ligand or nucleic acid support.',
  },
  'multimer-5': {
    weightsRevision: 'e960ea140c77',
    quantizationDetail: 'INT5 Multimer Model 1 + ~44 MB delta bundle (~118 MB combined)',
    familyNote:
      'AF2-Multimer v3 Model 5 (CC-BY-4.0): protein complex predictor loaded as a delta on Multimer Model 1; no ligand or nucleic acid support.',
  },
  rosettafold3: {
    weightsRevision: '5e5d29485700',
    quantizationDetail: '446 tensors at INT5 across 16 shards (~266 MB); AF3-class diffusion trunk + confidence head',
    familyNote:
      'RoseTTAFold3 / RF3 (BSD-3-Clause, Baker Lab / RosettaCommons): all-atom AF3-class diffusion supporting proteins, DNA, RNA, and CCD ligands (upstream 6MRR check: 1.68 Å RMSD, TM 0.911, pLDDT 81.5).',
  },
  boltz2: {
    weightsRevision: '651a25ba8a57',
    quantizationDetail: '442 tensors at INT5 across 8 shards (~364 MB); 384-ch trunk, 768-ch diffusion conditioning, 8-block confidence head',
    familyNote:
      'Boltz-2 (MIT): all-atom AF3-class diffusion supporting proteins, DNA, RNA, and CCD ligands with an 8-block confidence head.',
  },
  openbind0: {
    weightsRevision: '71ece3571b80',
    quantizationDetail: 'INT5 quantized OpenBind-0 (OpenFold3 v0.5.0) AF3-class weights (~265 MB)',
    familyNote:
      'OpenBind-0 / OpenFold3 v0.5.0 (Apache-2.0): permissive all-atom AF3-class predictor (protein, DNA, RNA, CCD ligands) without AlphaFold 3 non-commercial restrictions.',
  },
  protenix2: {
    weightsRevision: '068c905dfb0f',
    quantizationDetail: '404 tensors at INT5 across 8 shards (~334 MB); concatenated template pair projection',
    familyNote:
      'Protenix-v2 (Apache-2.0, ByteDance): all-atom AF3-class diffusion supporting proteins, nucleic acids, and CCD ligands.',
  },
  opendde: {
    weightsRevision: '068c905dfb0f',
    quantizationDetail: '481 tensors (655.8M params) at INT5 across 12 shards (~473 MB); residue trunk + backbone/sidechain diffusion tokens',
    familyNote:
      'OpenDDE (Apache-2.0, Aureka Research): independent AF3-class implementation with two token spaces (upstream 6MRR check: 1.68 Å RMSD, TM 0.865).',
  },
  mol1: {
    weightsRevision: '32ffc3c1e4fb',
    quantizationDetail: '441 tensors at INT5 group-32 across 32 shards (~510 MB); trunk single transition2 retained in FP32 to prevent pLDDT collapse',
    familyNote:
      'Mol-1 (Apache-2.0, Molence Bio): 448-channel pair AF3-class model with 8 PairMixer blocks; keeps trunk single transition2 in FP32 for confidence calibration.',
  },
  intellifold2: {
    weightsRevision: '6634c1398a78',
    quantizationDetail: '406 tensors at INT5 across 40 shards (~612 MB); 512-ch trunk pair, 256-ch template stack',
    familyNote:
      'IntelliFold-2 (Apache-2.0, intelligenAI): wide-pair (512-ch) all-atom AF3-class model with highest download size (~612 MB).',
  },
  af3: {
    weightsRevision: '71ece3571b80',
    quantizationDetail: 'INT5 quantized AlphaFold 3 trunk, diffusion head & confidence head (~265 MB)',
    familyNote:
      'AlphaFold 3 (Google DeepMind Non-Commercial / Prohibited Use Policy): all-atom diffusion; strictly non-commercial use only (use OpenBind-0 or Boltz-2 for permissive commercial licensing).',
  },
};

export const MODEL_CARDS: Record<string, ModelCardSpec> = {
  'rfd4-proteina': {
    id: 'rfd4-proteina',
    adapterId: 'rfd4-proteina',
    displayName: 'RFD4-Proteina',
    status: 'live',
    activePrecisionId: 'int8-fp16',
    precisions: [
      {
        id: 'int8-fp16',
        label: 'INT8 · FP16 GPU',
        quantizationDetail:
          'Block-64 INT8 sharded weights (~330 MB trunk + decoder; +138 MB lazy autoencoder encoder for partial diffusion / inverse folding) uploaded as packed FP16 on WebGPU ({ half: true })',
        approxSizeMB: 330,
        parityVsPytorchFp32: {
          metric: 'Upstream WebGPU port parity',
          value: 'Upstream block-64 INT8 / packed-FP16 port — no custom relRMS benchmark recorded in-repo',
          scope:
            'Matches upstream web.rfd4-proteina.org runtime ({ half: true }); trunk + decoder (~330 MB) and optional encoder (+138 MB)',
        },
        caveat:
          'Block-64 INT8 quantization and packed FP16 GPU execution trade storage/VRAM for precision; no scalar relRMS against NVIDIA PyTorch FP32 is published in the vendored bundle.',
      },
    ],
    upstreamRepo: 'https://github.com/sokrypton/rfd4-proteina-webgpu',
    upstreamCommit: '3d1fe0398653334e385f4fbc301efaf988e95b39',
    weightsRepo: 'https://huggingface.co/nvidia/RFD4-Proteina',
    license: 'MIT (WebGPU runtime) · BSD-3-Clause (NVIDIA RFD4-Proteina weights)',
    citation:
      'Baker Lab & NVIDIA (2025/2026) RFD4-Proteina; WebGPU inference runtime by Sergey Ovchinnikov (sokrypton/rfd4-proteina-webgpu).',
    limitations: [
      'Block-64 INT8 weights unpacked to FP16 on GPU ({ half: true }); no custom relRMS benchmark vs PyTorch FP32 is recorded in this repository.',
      'Flow-matching convergence degrades below ~58 steps (default 60); if trunk logits collapse, the worker attempts recovery via the autoencoder decoder logit_linear head.',
      'Partial diffusion and inverse-folding modes lazily download an additional ~138 MB autoencoder encoder (~469 MB total).',
      'Non-protein ligand/ion conditioning fetches RCSB CCD component definitions when online and skips external CCD lookups offline.',
    ],
  },

  genie3: {
    id: 'genie3',
    adapterId: 'genie3',
    displayName: 'Genie 3',
    status: 'live',
    activePrecisionId: 'fp16',
    precisions: [
      {
        id: 'fp16',
        label: 'FP16',
        quantizationDetail:
          '209 F16 2D weight tensors + 295 F32 1D norms/biases (2 shards, 44.57 MB); packed FP16 GPU weights with FP32 accumulation',
        approxSizeMB: 45,
        parityVsPytorchFp32: {
          metric: 'Forward xl / ai relRMS vs PyTorch FP32',
          value: 'xl 2.243e-4 · ai 4.390e-4 (weights 1.968e-4)',
          scope: 'Unconditional L=16 V1Denoiser oracle (seed=42, canonicalized pairwise quaternions)',
        },
        caveat:
          'Pairwise quaternion eigenvector sign in rot_to_quat varies across linear-algebra backends; parity is measured on canonical-sign traces.',
      },
      {
        id: 'fp32',
        label: 'FP32',
        quantizationDetail: '504 F32 tensors (23,269,484 params, 4 shards, 88.82 MB); exact checkpoint weights',
        approxSizeMB: 89,
        parityVsPytorchFp32: {
          metric: 'Weight round-trip & forward relRMS vs PyTorch FP32',
          value: '0.000e+00 (weight round-trip); WebGPU trace gate < 1e-4',
          scope: 'Unconditional L=16 V1Denoiser + 3-step DDIM + 1-step SequenceNet oracle',
        },
      },
      {
        id: 'int8',
        label: 'INT8',
        quantizationDetail:
          '188 symmetric per-row I8 weights + colocated F32 scales, 316 F32 tensors (1 shard, 22.75 MB)',
        approxSizeMB: 23,
        parityVsPytorchFp32: {
          metric: 'Forward xl / ai relRMS vs PyTorch FP32',
          value: 'xl 1.065e-2 · ai 1.894e-2 (weights 9.383e-3)',
          scope: 'Unconditional L=16 V1Denoiser export quantization benchmark (not active in live worker)',
        },
        caveat:
          'Not enabled in the live browser worker; ~1–2% single-step relative error accumulates over 100-step DDIM sampling.',
      },
    ],
    upstreamRepo: 'https://github.com/aqlaboratory/genie3',
    upstreamCommit: '0c53b2e5b49d18f3c6d7722952ed1287a116509a',
    weightsRepo: 'https://huggingface.co/ramithuh/genie3-webgpu',
    license: 'Apache-2.0 (aqlaboratory/genie3 & ramithuh/genie3-webgpu)',
    citation:
      'Lin et al., Genie 3 (AlQuraishi Lab, Columbia University; weights yeqinglin/genie3 step=600000); WebGPU port ramithuh/genie3-webgpu.',
    limitations: [
      'Cα-only protein backbone diffusion model (23.3M params); pair with Cα-ProteinMPNN for sequence design.',
      'Cannot condition on small-molecule ligands, metals, or nucleic acids (use RFD4-Proteina for all-atom conditioning).',
      '1-step SequenceNet head is fast but lower recovery than iterative inverse folding; chain into Cα-ProteinMPNN before refolding.',
      'Oracle parity is verified on L=16 V1Denoiser/DDIM traces; multi-step stochastic trajectories diverge across different RNG implementations.',
    ],
  },

  'web-protein-mpnn': {
    id: 'web-protein-mpnn',
    adapterId: 'web-protein-mpnn',
    displayName: 'ProteinMPNN',
    status: 'live',
    activePrecisionId: 'onnx-fp32',
    precisions: [
      {
        id: 'onnx-fp32',
        label: 'ONNX FP32 · WebGPU/WASM',
        quantizationDetail:
          'Float32 ONNX weights (~9.5 MB per checkpoint) executed via ONNX Runtime Web (backend: auto — WebGPU when available, automatic WASM fallback)',
        approxSizeMB: 10,
        parityVsPytorchFp32: {
          metric: 'Upstream ONNX / WASM maxAbsDiff vs PyTorch FP32',
          value: '≤ 7.28e-5 (Python ORT) · ≤ 1e-4–2e-4 (WASM); no hardware WebGPU relRMS recorded upstream',
          maxAbsDiff: '7.28e-5 (Python ORT), 1e-4–2e-4 (WASM)',
          scope:
            '216 upstream checks across all 9 official checkpoints (L=1–55; ProteinMPNN, SolubleMPNN, Cα-only)',
          hardware: 'CPU / WASM (ORT Web 1.24.3; hardware WebGPU unbenchmarked upstream)',
        },
        caveat:
          'Upstream proteinmpnn-web validates Python ONNX and WASM sessions; no custom hardware WebGPU relRMS benchmark is published in-repo.',
      },
    ],
    upstreamRepo: 'https://github.com/MurrellGroup/webProteinMPNN',
    upstreamCommit: '8907e6671bfbfc92303b5f79c4b5e6ce47cdef57',
    weightsRepo: 'https://huggingface.co/MurrellLab/webports/tree/main/webProteinMPNN',
    license: 'MIT (MurrellGroup/webProteinMPNN & dauparas/ProteinMPNN)',
    citation:
      'Dauparas et al. (2022) Robust deep learning-based protein sequence design using ProteinMPNN, Science 378(6615):49–56; browser port MurrellGroup/webProteinMPNN.',
    limitations: [
      'Backbone-only inverse folding (N, CA, C, O — or Cα-only in the "ca" variant); ligands, metals, waters, nucleic acids, and sidechains are ignored.',
      'Upstream validation covers Python ONNX and WASM CPU sessions (maxAbsDiff ≤ 7.28e-5 / 2e-4) and does not publish a hardware WebGPU relRMS table.',
      'Built-in seeded PRNG is deterministic within the browser library but does not reproduce PyTorch RNG streams; FP32 differences can flip draws near boundaries.',
      'Neighbor graph construction runs in O(N²) on CPU, and incremental WebGPU decoding transfers neighbor caches through CPU memory.',
    ],
  },

  'localfold-af2': {
    id: 'localfold-af2',
    adapterId: 'localfold-af2',
    displayName: 'LocalFold',
    status: 'live',
    activePrecisionId: 'int5',
    precisions: [
      {
        id: 'int5',
        label: 'INT5',
        quantizationDetail:
          '5-bit weight quantization (INT5) across 18 pinned checkpoints (97–612 MB; AF2/Multimer models 2–5 load as ~43–44 MB deltas on model 1; Mol-1 keeps trunk single transition2 in FP32)',
        approxSizeMB: 97,
        parityVsPytorchFp32: {
          metric: 'Upstream LocalFold INT5 port',
          value: 'Upstream INT5 WebGPU port — no custom relRMS benchmark recorded in-repo',
          scope:
            'Upstream sokrypton/localfold structural checks (e.g. AF2 deltas within 0.01 Å of standalone on 5CAJ; RF3 6MRR 1.68 Å / TM 0.911; OpenDDE 6MRR 1.68 Å / TM 0.865)',
        },
        caveat:
          'INT5 weight quantization enables 100–600 MB browser folding but has no custom relRMS parity table against upstream FP32 checkpoints in this repository.',
      },
    ],
    upstreamRepo: 'https://github.com/sokrypton/localfold',
    upstreamCommit: 'f98fc0083d0fe9dc475180a176509d5e16b811f7',
    weightsRepo: 'https://huggingface.co/sokrypton/localfold',
    weightsRevision: '8a8aa55554b7',
    license:
      'Beer-Ware (runtime); per-checkpoint weights (AF2/Multimer CC-BY-4.0, RF3 BSD-3-Clause, Boltz-2 MIT, OpenBind-0/Protenix-v2/OpenDDE/Mol-1/IntelliFold-2 Apache-2.0, AF3 Non-Commercial)',
    citation:
      'Ovchinnikov, S. (sokrypton/localfold); Jumper et al. (2021) AlphaFold2, Nature; Abramson et al. (2024) AlphaFold 3, Nature; Passaro et al. (2025) Boltz-2; Baker Lab RoseTTAFold3; OpenFold3 / OpenBind-0.',
    limitations: [
      'AF2 Monomer (models 1–5, ~97–140 MB INT5, CC-BY-4.0): protein-only monomer folding; models 3–5 are template-free; multi-chain ipTM is an approximation.',
      'AF2-Multimer v3 (models 1–5, ~74–118 MB INT5, CC-BY-4.0): protein complexes only (no ligands, DNA, or RNA); models 2–5 load as ~44 MB deltas on model 1.',
      'RoseTTAFold3 / RF3 (~266 MB INT5, BSD-3-Clause): all-atom AF3-class diffusion (proteins, DNA, RNA, CCD ligands); diffusion outputs vary across seeds.',
      'Boltz-2 (~364 MB INT5, MIT): all-atom AF3-class diffusion with 768-ch diffusion conditioning and an 8-block confidence head.',
      'OpenBind-0 (~265 MB INT5, Apache-2.0): OpenFold3 v0.5.0 permissive all-atom AF3-class model (no AF3 Prohibited Use Policy); Protenix-v2 (334 MB), OpenDDE (473 MB), Mol-1 (510 MB, FP32 transition2), and IntelliFold-2 (612 MB) are also Apache-2.0; AlphaFold 3 (265 MB) is strictly non-commercial.',
      'Default single-sequence mode runs 100% locally; enabling MMseqs2 or Target-Only MSA queries api.colabfold.com with your sequence.',
      'No custom relRMS benchmark against PyTorch/JAX FP32 is recorded in-repo; numerical fidelity relies on upstream sokrypton/localfold INT5 bundles.',
    ],
  },

  'thermompnn-webgpu': {
    id: 'thermompnn-webgpu',
    displayName: 'ThermoMPNN',
    status: 'weights-hosted',
    activePrecisionId: 'fp32',
    precisions: [
      {
        id: 'fp32',
        label: 'FP32',
        quantizationDetail:
          '128 F32 tensors (1,835,740 inference params, 1 shard, 7.36 MB); full FP32 weights, activations & accumulation in WGSL',
        approxSizeMB: 7.4,
        parityVsPytorchFp32: {
          metric: 'WebGPU ΔΔG relRMS vs PyTorch FP32',
          value: '4.285e-7 – 1.426e-6 (all traces ≤ 2.044e-6; 194-aa 2OCJ 7.335e-7)',
          scope: '9 PDB/mmCIF mode fixtures + complete 194-aa 2OCJ chain A (SSM & targeted ΔΔG)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 153.0.8010.12',
        },
        latencyHint: '5.0 ms (12 aa) · 23.3 ms (76 aa) · 50.6 ms (194 aa) warm',
      },
      {
        id: 'fp16',
        label: 'FP16',
        quantizationDetail:
          'F16 2D weight storage (core unpack2x16float or shader-f16), F32 embeddings, norms, biases & accumulation (1 shard, 3.72 MB)',
        approxSizeMB: 3.7,
        parityVsPytorchFp32: {
          metric: 'ΔΔG relRMS vs original PyTorch FP32 (weight quant)',
          value: '3.711e-4 – 5.881e-4 (GPU vs FP16 oracle ≤ 1.298e-6)',
          maxAbsDiff: '0.0045 kcal/mol (194-aa 2OCJ)',
          scope: '9 PDB/mmCIF fixtures + 194-aa 2OCJ (weight relRMS 2.065e-4; r ≥ 0.9999998)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 153.0.8010.12',
        },
        latencyHint: '3.4 ms (12 aa) · 20.6 ms (76 aa) · 49.6 ms (194 aa) warm',
      },
      {
        id: 'int8',
        label: 'INT8',
        quantizationDetail:
          'Symmetric per-output-row I8 weights + colocated F32 scales; F32 embeddings, norms, biases & accumulation (1 shard, 1.93 MB)',
        approxSizeMB: 1.9,
        parityVsPytorchFp32: {
          metric: 'ΔΔG relRMS vs original PyTorch FP32 (weight quant)',
          value: '1.772e-2 – 4.657e-2 (GPU vs INT8 oracle ≤ 9.960e-7)',
          maxAbsDiff: '0.201 kcal/mol (194-aa 2OCJ)',
          scope: '9 PDB/mmCIF fixtures + 194-aa 2OCJ (weight relRMS 1.047e-2)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 153.0.8010.12',
        },
        latencyHint: '4.5 ms (12 aa) · 20.0 ms (76 aa) · 53.4 ms (194 aa) warm',
        caveat:
          'INT8 trades accuracy for ~1.9 MB size (up to 4.66% relative ΔΔG error, 0.201 kcal/mol maxAbsDiff); FP32 is only 7.4 MB and recommended.',
      },
    ],
    upstreamRepo: 'https://github.com/Kuhlman-Lab/ThermoMPNN',
    upstreamCommit: '370f76ec62bd929f7425e311d8df04a0d094990f',
    weightsRepo: 'https://huggingface.co/ramithuh/thermompnn-webgpu',
    weightsRevision: '9adab6c70209',
    license: 'MIT',
    citation:
      'Dieckhaus et al. (2024) Transfer learning to leverage larger datasets for improving prediction of protein stability changes, PNAS 121(6):e2314853121.',
    limitations: [
      'Single point-mutation ΔΔG stability predictor only (kcal/mol; negative = stabilizing); double-mutant epistasis and indels require separate ThermoMPNN-D/I checkpoints.',
      'Backbone-only (N, CA, C, O + virtual Cβ); no ligand, ion, or nucleic-acid conditioning and no sequence-only scoring mode.',
      'Input capped at 2,048 positions including gaps; parity fixtures validate numerical fidelity against PyTorch rather than experimental ΔΔG accuracy.',
    ],
  },

  'ligandmpnn-webgpu': {
    id: 'ligandmpnn-webgpu',
    displayName: 'LigandMPNN',
    status: 'weights-hosted',
    activePrecisionId: 'fp32',
    precisions: [
      {
        id: 'fp32',
        label: 'FP32',
        quantizationDetail:
          '197 F32 tensors (2,618,693 params, 1 shard, 10.49 MB per variant v_32_010_25 / v_32_020_25); tiled 16×16 F32 GEMM in WGSL',
        approxSizeMB: 10.5,
        parityVsPytorchFp32: {
          metric: 'WebGPU logits relRMS vs PyTorch FP32',
          value: '1.392e-6 (1BC8 Zn+DNA) · 1.584e-6 (3HS4 ligand+Zn) · worst AR trace 3.483e-6',
          maxAbsDiff: '1.097e-5 (1BC8) · 9.403e-6 (3HS4)',
          scope: 'Holo 1BC8 (L=32, Zn+DNA) & 3HS4 (L=32, acetazolamide+Zn), teacher-forced + 4 exact AR replays',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Dawn Vulkan)',
        },
        latencyHint: '260.7 ms/seq warm (L=32) · 632.9 ms warm (93-aa 1BC8)',
      },
      {
        id: 'fp16',
        label: 'FP16',
        quantizationDetail:
          'F16 2D weight matrices + F32 1D norms/biases (1 shard, 5.29 MB); F32 activations & accumulation',
        approxSizeMB: 5.3,
        parityVsPytorchFp32: {
          metric: 'Forward logits relRMS vs original PyTorch FP32',
          value: '5.951e-4 (010 GPU) · 4.968e-4 (020); GPU vs FP16 oracle 1.373e-6',
          maxAbsDiff: '4.676e-3 (010 GPU vs FP32) · 1.359e-5 (vs FP16 oracle)',
          scope: 'Holo 1BC8 (L=32) teacher-forced logits (log-probs relRMS 1.804e-4 / 1.523e-4)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Dawn Vulkan)',
        },
        caveat:
          'v_32_010_25 FP16 weight rounding reaches 5.95e-4 logit relRMS vs FP32 in PyTorch itself (requestedLogitsToleranceMet=false); packed-half path currently keeps F32 copies in GPU memory.',
      },
      {
        id: 'int8',
        label: 'INT8',
        quantizationDetail:
          'Symmetric per-row I8 2D weights + colocated F32 scales, F32 1D tensors (1 shard, 2.75 MB); dequantized to F32 on GPU load',
        approxSizeMB: 2.8,
        parityVsPytorchFp32: {
          metric: 'Forward logits relRMS vs original PyTorch FP32',
          value: '2.390e-2 (010) · 2.658e-2 (020) (log-probs 7.880e-3 / 8.251e-3)',
          scope: 'Holo 1BC8 (L=32) teacher-forced reload into PyTorch FP32 (weight relRMS 7.667e-3 / 7.767e-3)',
        },
        caveat:
          '~2.4–2.7% logit relRMS vs FP32; dequantized to FP32 on load (reduces download size only, not GPU VRAM).',
      },
    ],
    upstreamRepo: 'https://github.com/dauparas/LigandMPNN',
    upstreamCommit: '26ec57ac976ade5379920dbd43c7f97a91cf82de',
    weightsRepo: 'https://huggingface.co/ramithuh/ligandmpnn-webgpu',
    weightsRevision: 'cecd9ee340dc',
    license: 'MIT',
    citation:
      'Dauparas et al. (2025) Atomic context-conditioned protein sequence design using LigandMPNN, Nature Methods 22:717–723.',
    limitations: [
      'Sidechain packing model (ligandmpnn_sc) is deferred; fixed sidechain context is supported, but packed all-atom sidechains are not generated.',
      'RNA heavy atoms use the same context pathway as DNA; no independent RNA crystal-structure oracle was included in parity validation.',
      'Strict per-layer oracles cover 32-residue holo pockets (1BC8, 3HS4); full 93-residue 1BC8 is verified as a warm latency and determinism smoke check.',
      'JavaScript PRNG does not replicate PyTorch multinomial streams from the same seed (exact replay verified via saved CDF uniforms).',
    ],
  },

  'lasermpnn-webgpu': {
    id: 'lasermpnn-webgpu',
    displayName: 'LASErMPNN',
    status: 'weights-hosted',
    activePrecisionId: 'fp32',
    precisions: [
      {
        id: 'fp32',
        label: 'FP32',
        quantizationDetail:
          '492 F32 state tensors incl. embedded SPICE ligand encoder (20,599,412 params, 4 shards, 82.54 MB); full WGSL neural ops + double-precision rotamer geometry',
        approxSizeMB: 82.5,
        parityVsPytorchFp32: {
          metric: 'WebGPU AA / χ4 logits & protonated coords relRMS vs PyTorch FP32',
          value: 'AA 8.872e-7 · χ4 1.533e-6 · coords 1.144e-8 (API PDB coords 9.735e-6)',
          maxAbsDiff: 'AA 2.480e-5 · χ4 6.027e-4 · coords 3.815e-6 Å',
          scope: 'Full protonated holo example (115 residues, 31 ligand atoms; exact sequence & χ replay)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Dawn Vulkan)',
        },
        latencyHint: '4.85–5.12 s full PDB→design (115 aa + 31 ligand atoms)',
      },
      {
        id: 'fp16',
        label: 'FP16 (experimental)',
        quantizationDetail:
          'F16 learned 2D matrices; F32 1D norms/biases, geometry buffers & higher-rank attention params (2 shards, 41.46 MB)',
        approxSizeMB: 41.5,
        parityVsPytorchFp32: {
          metric: 'AA / χ3 logits & coords relRMS vs original PyTorch FP32',
          value: 'AA 6.046e-4 · χ3 1.570e-3 · coords 2.393e-7 (GPU vs FP16 oracle AA 8.952e-7)',
          maxAbsDiff: 'AA 1.047e-2 · χ3 1.313e-1 · χ4 7.709e-1 · coords 8.736e-5 Å',
          scope: '115-aa / 31-ligand-atom holo fixture (fixed FP32 trajectory; weight relRMS 2.060e-4)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Dawn Vulkan)',
        },
        latencyHint: '4.91 s (fp16-core) · 5.58 s (fp16-shader) encoder+decoder',
        caveat:
          'Experimental: FP16 weight rounding exceeds 5e-4 vs original FP32 on AA (6.046e-4) and χ3/χ4 (1.570e-3 / 1.430e-3) logits (fp16OriginalFp32LimitMet=false).',
      },
      {
        id: 'int8',
        label: 'INT8',
        quantizationDetail:
          'Symmetric per-row I8 learned 2D matrices + colocated F32 scales; F32 norms, biases, geometry & higher-rank params (1 shard, 21.12 MB)',
        approxSizeMB: 21.1,
        parityVsPytorchFp32: {
          metric: 'AA / χ4 logits & coords relRMS vs original PyTorch FP32',
          value: 'AA 3.457e-2 · χ4 9.351e-2 · coords 3.078e-5 (weights 1.012e-2)',
          maxAbsDiff: 'AA 6.408e-1 · χ4 53.17 · coords 1.187e-2 Å',
          scope: '115-aa / 31-ligand-atom holo fixture (export quantization benchmark; dequantized to F32 on load)',
        },
        caveat:
          'High χ-head logit drift (up to 9.35% relRMS on χ4); FP32 is strongly recommended for sidechain rotamer design.',
      },
    ],
    upstreamRepo: 'https://github.com/polizzilab/LASErMPNN',
    upstreamCommit: 'e70f2c6d765416f7e29d51bfd6d4e08496438878',
    weightsRepo: 'https://huggingface.co/ramithuh/lasermpnn-webgpu',
    weightsRevision: 'a9ae906fc718',
    license: 'MIT',
    citation:
      'Polizzi Lab (polizzilab/LASErMPNN), checkpoint laser_weights_0p1A_nothing_heldout.pt with embedded SPICE ligand encoder.',
    limitations: [
      'Requires pre-protonated ligand coordinates matching upstream input expectations; no automatic protonation or bond-order inference is performed.',
      'Sequence and χ scores are untempered natural-log probabilities (over discrete 5° χ bins, not continuous offsets) and are not physical binding-energy predictions.',
      'FP16 weight rounding exceeds 5e-4 vs original FP32 on AA (6.05e-4) and χ3/χ4 (1.57e-3 / 1.43e-3) logits; use FP32 for exact parity.',
      'Parity and latency were measured on one complete 115-residue / 31-ligand-atom holo benchmark rather than a multi-length scaling suite.',
    ],
  },

  'antifold-esmif1-webgpu': {
    id: 'antifold-esmif1-webgpu',
    displayName: 'AntiFold & ESM-IF1',
    status: 'weights-hosted',
    activePrecisionId: 'fp16',
    precisions: [
      {
        id: 'fp16',
        label: 'FP16',
        quantizationDetail:
          '141.7M-param GVP-Transformer (489 tensors); F16 2D weight storage (core unpack2x16float or shader-f16) + F32 embeddings, norms, biases & accumulation (14 shards, 283.84 MB per variant)',
        approxSizeMB: 284,
        parityVsPytorchFp32: {
          metric: 'Worst logit relRMS vs original PyTorch FP32 (all 14 modes)',
          value: '2.482e-4 (AntiFold) · 4.331e-4 (ESM-IF1); GPU vs FP16 oracle ≤ 9.68e-6 / 9.45e-5',
          scope: '14 modes incl. 236-aa antibody H/L, 105-aa VHH nanobody, complexes, single/double mutants (weight relRMS 2.064e-4)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Vulkan WebGPU)',
        },
        latencyHint: '395.3 ms (105-aa VHH) · 763.3 ms (236-aa H/L) · 161.7 ms (20-aa) warm',
      },
      {
        id: 'fp32',
        label: 'FP32',
        quantizationDetail:
          'All 489 tensors in F32 (141,662,150 params, 31 shards, 566.70 MB per variant); bit-identical reload vs PyTorch FP32',
        approxSizeMB: 567,
        parityVsPytorchFp32: {
          metric: 'WebGPU worst trace relRMS vs PyTorch FP32',
          value: '5.09e-6 – 9.68e-6 (AntiFold) · 9.03e-6 – 9.45e-5 (ESM-IF1)',
          scope: 'All 14 antibody, nanobody, monomer, complex & mutation modes (exact AR + CDR replay)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Vulkan WebGPU)',
        },
        latencyHint: '473.1 ms (105-aa VHH) · 656.8 ms (236-aa H/L) · 156.0 ms (20-aa) warm',
      },
      {
        id: 'int8',
        label: 'INT8',
        quantizationDetail:
          'Symmetric per-output-row I8 2D matrices + colocated F32 scales (6 shards, 143.01 MB AntiFold / 143.21 MB ESM-IF1; ESM-IF1 keeps one 256×256 GVP weight in F32 for norm-mask stability)',
        approxSizeMB: 143,
        parityVsPytorchFp32: {
          metric: 'Worst logit relRMS vs original PyTorch FP32 (all 14 modes)',
          value: '1.947e-2 (AntiFold) · 5.073e-2 (ESM-IF1); GPU vs INT8 oracle ≤ 9.45e-5',
          scope: 'All 14 modes (weight relRMS 1.258e-2 AntiFold / 1.259e-2 ESM-IF1)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Vulkan WebGPU)',
        },
        latencyHint: '326.3 ms (105-aa VHH) · 731.6 ms (236-aa H/L) · 146.9 ms (20-aa) warm',
        caveat:
          'INT8 can noticeably shift amino-acid preferences on missing-coordinate inputs (up to 5.07% worst logit relRMS on ESM-IF1).',
      },
    ],
    upstreamRepo: 'https://github.com/oxpig/AntiFold',
    upstreamCommit: '9918344067f34a6b026f2634fdba212fc3e6e986',
    weightsRepo: 'https://huggingface.co/ramithuh/antifold-esmif1-webgpu',
    weightsRevision: 'a85f8071d594',
    license: 'BSD-3-Clause (AntiFold) · MIT (Meta ESM-IF1 & GVP)',
    citation:
      'Høie et al. (2024) AntiFold: Improved antibody structure-based design using inverse folding, Bioinformatics; Hsu et al. (2022) Learning inverse folding from millions of predicted structures (ESM-IF1), ICML.',
    limitations: [
      'Backbone-only (N, CA, C); ligands, waters, ions, nucleic acids, hydrogens, and sidechain atoms do not enter the model.',
      'IMGT or Kabat numbering must be supplied by the caller (--skip_anarcii_numbering contract); ANARCII auto-numbering is not bundled.',
      'Autoregressive decoding recomputes causal decoder prefixes without a KV cache to guarantee exact numerical parity.',
      'ESM-IF1 INT8 retains one 256×256 GVP projection in FP32 (+196 KB) to avoid crossing the hard 2e-4 vector-norm mask threshold.',
    ],
  },

  'esmc-webgpu': {
    id: 'esmc-webgpu',
    displayName: 'ESM C 300M',
    status: 'weights-hosted',
    activePrecisionId: 'int8',
    precisions: [
      {
        id: 'int8',
        label: 'INT8',
        quantizationDetail:
          'Symmetric per-output-row I8 transformer weights + colocated F32 scales; F32 embeddings, norms, biases & accumulation (15 shards, 334.95 MB)',
        approxSizeMB: 335,
        parityVsPytorchFp32: {
          metric: 'MLM logits / embeddings / masked LLR relRMS vs original FP32',
          value: 'logits 3.688e-3–8.403e-3 · emb 1.358e-2–2.012e-2 · masked LLR 8.581e-3–2.808e-2 (GPU vs INT8 oracle ≤ 7.132e-6)',
          maxAbsDiff: 'logits 0.347 (Trp-cage) / 1.539 (ubiquitin) · masked LLR 0.104 / 0.252',
          scope: 'Trp-cage (20 aa) & ubiquitin (76 aa), all 30 blocks + 96 single-mask sweeps (weight relRMS 8.558e-3)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Dawn Vulkan)',
        },
        latencyHint: '23.0 ms (20 aa) · 42.8 ms (76 aa) warm; 1.31 s cold',
        caveat:
          'All 4 masked-fill top-1 predictions match FP32, but INT8 adds ~0.8–2.8% masked LLR quantization error (use FP16 for <0.08% LLR error).',
      },
      {
        id: 'fp16',
        label: 'FP16',
        quantizationDetail:
          'F16 2D weight storage (core unpack2x16float or shader-f16); F32 embeddings, norms, biases & accumulation (30 shards, 666.51 MB)',
        approxSizeMB: 667,
        parityVsPytorchFp32: {
          metric: 'MLM logits / embeddings / masked LLR relRMS vs original FP32',
          value: 'logits 1.416e-4–2.162e-4 · emb 2.988e-4–4.781e-4 · masked LLR 2.194e-4–7.613e-4 (GPU vs FP16 oracle ≤ 6.807e-6)',
          maxAbsDiff: 'logits 0.0117 (Trp-cage) / 0.0443 (ubiquitin) · masked LLR 0.0034 / 0.0080',
          scope: 'Trp-cage (20 aa) & ubiquitin (76 aa), all 30 blocks + 96 single-mask sweeps (weight relRMS 1.906e-4)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Dawn Vulkan)',
        },
        latencyHint: '24.2 ms (20 aa) · 43.4 ms (76 aa) warm; 2.53 s cold',
      },
      {
        id: 'fp32',
        label: 'FP32',
        quantizationDetail:
          'All 308 tensors in F32 (332,997,184 params, 61 shards, 1,332.03 MB); bit-identical checkpoint reload',
        approxSizeMB: 1332,
        parityVsPytorchFp32: {
          metric: 'WebGPU max trace relRMS vs PyTorch FP32',
          value: '2.840e-6 (emb) · 2.937e-6 (MLM logits) · 7.834e-6 (masked LLR)',
          scope: 'Trp-cage (20 aa) & ubiquitin (76 aa), all 30 blocks, attention maps & 96 single-mask sweeps',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 1243 (Dawn Vulkan)',
        },
        latencyHint: '26.6 ms (20 aa) · 47.2 ms (76 aa) warm; 4.91 s cold',
      },
    ],
    upstreamRepo: 'https://github.com/Biohub/esm',
    upstreamCommit: '26b0bc2b771e3e419ea74f445a5f35cc094a1509',
    weightsRepo: 'https://huggingface.co/ramithuh/esmc-webgpu',
    weightsRevision: '773a9473acb5',
    license: 'MIT (Biohub ESM C 300M code & weights)',
    citation:
      'ESM Team (2024) ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning (Biohub / EvolutionaryScale, biohub/esmc-300m-2024-12).',
    limitations: [
      'Sequence-only protein language model; 3D structure coordinates, ligands, and sidechain packing never condition ESM C.',
      'Input admission policy caps sequences at 1,022 internal residues (+ BOS/EOS); masked-marginal LLR and naturalness sweeps execute L forward passes.',
      'Multi-mutant LLR scoring sums single-site log-likelihood ratios additively and does not model epistasis.',
      'Full FP32 bundle is 1.33 GB (61 shards); INT8 (335 MB) is practical for interactive browser use with ~0.8–2.8% LLR quantization error.',
    ],
  },

  'dyna1-webgpu': {
    id: 'dyna1-webgpu',
    displayName: 'Dyna-1',
    status: 'weights-hosted',
    activePrecisionId: 'int8',
    precisions: [
      {
        id: 'int8',
        label: 'INT8',
        quantizationDetail:
          'Symmetric per-output-row I8 matrices + colocated F32 scales; F32 norms, biases & accumulation (43 shards, 843.63 MB; 23 ESM-3 blocks + 12 Dyna-1 classifier blocks)',
        approxSizeMB: 844,
        parityVsPytorchFp32: {
          metric: 'WebGPU probability / logit relRMS vs PyTorch FP32',
          value: 'prob 3.444e-3 (Trp-cage) · 5.799e-3 (ubiquitin); logit 5.465e-3 / 5.031e-3 (r ≥ 0.9995)',
          maxAbsDiff: 'prob 4.843e-3 (Trp-cage) / 8.755e-3 (ubiquitin) · logit 0.0203 / 0.0376',
          scope: 'Trp-cage (20 aa) & ubiquitin (76 aa), 23 ESM-3 blocks + 12 classifier blocks (weight relRMS 9.863e-3)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 153.0.8010.12 (Dawn Vulkan)',
        },
        latencyHint: '40.5 ms (20 aa) · 85.5 ms (76 aa) warm',
      },
      {
        id: 'fp16',
        label: 'FP16 mixed',
        quantizationDetail:
          'F16 matrices except Dyna-1 classifier blocks 0–1 retained in F32 (+62.9 MB accuracy exception); F32 norms, biases & accumulation (97 shards, 1,744.62 MB)',
        approxSizeMB: 1745,
        parityVsPytorchFp32: {
          metric: 'WebGPU probability / logit relRMS vs PyTorch FP32',
          value: 'prob 7.195e-5 (Trp-cage) · 1.511e-4 (ubiquitin); logit 1.227e-4 / 1.263e-4 (r ≥ 0.99999999)',
          maxAbsDiff: 'prob 7.653e-5 (Trp-cage) / 1.087e-4 (ubiquitin) · logit 3.209e-4 / 4.368e-4',
          scope: 'Trp-cage (20 aa) & ubiquitin (76 aa), core & shader-f16 paths (weight relRMS 3.172e-5)',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 153.0.8010.12 (Dawn Vulkan)',
        },
        latencyHint: '42.2–43.4 ms (20 aa) · 87.6–89.6 ms (76 aa) warm',
      },
      {
        id: 'fp32',
        label: 'FP32',
        quantizationDetail:
          '840,417,793 params in F32 (210 shards, 3,361.74 MB); bit-identical to full 48-block upstream ESM-3 + Dyna-1 forward',
        approxSizeMB: 3362,
        parityVsPytorchFp32: {
          metric: 'WebGPU probability / logit relRMS vs PyTorch FP32',
          value: 'prob 5.285e-7 (Trp-cage) · 6.453e-7 (ubiquitin); logit 8.054e-7 / 5.329e-7 (r = 1.0000)',
          scope: 'Trp-cage (20 aa) & ubiquitin (76 aa), all 35 retained transformer blocks verified < 1e-4',
          hardware: 'NVIDIA GB10 (ARM64), Chromium 153.0.8010.12 (Dawn Vulkan)',
        },
        latencyHint: '59.1 ms (20 aa) · 116.6 ms (76 aa) warm',
        caveat: '3.36 GB weight download / GPU buffer footprint; intended for high-VRAM desktop GPUs and reference verification.',
      },
    ],
    upstreamRepo: 'https://github.com/WaymentSteeleLab/Dyna-1',
    upstreamCommit: '814a453fd85305536b919901e41062b91d5c9d86',
    weightsRepo: 'https://huggingface.co/ramithuh/dyna1-webgpu',
    weightsRevision: 'eec62e7debef',
    license: 'MIT (Dyna-1 classifier & port) · ESM-3 small open v1 backbone terms',
    citation:
      'Wayment-Steele et al. (2026) Learning millisecond protein dynamics from what is missing in NMR spectra, Nature, doi:10.1038/s41586-026-10989-4.',
    limitations: [
      'Sequence-only inference (1–1,024 canonical amino acids); structure-conditioned ESM-3 tracks are fixed to missing-coordinate embeddings.',
      'Outputs prior-corrected conformational exchange probabilities and classifier logits, not calibrated physical exchange rates (s⁻¹).',
      'FP16 mixed keeps classifier blocks 0–1 in FP32 (+62.9 MB) because pure FP16 exceeded the 5e-4 logit/probability relRMS gate.',
      'Parity and latency were measured on Trp-cage (20 aa) and ubiquitin (76 aa), not across every accepted sequence length; INT8 GPU weight buffers require ~844 MB.',
    ],
  },
};

export function getActivePrecision(card: ModelCardSpec): PrecisionTierSpec {
  return card.precisions.find((p) => p.id === card.activePrecisionId) ?? card.precisions[0];
}

/**
 * Format a concise, zero-clutter trigger pill label for an active model card.
 * Extracts the leading numerical parity figure when verified, or shows an honest runtime badge.
 */
export function formatModelCardPillLabel(card: ModelCardSpec, compact = false): string {
  const active = getActivePrecision(card);
  if (compact) return active.label;

  const parity = active.parityVsPytorchFp32;
  if (!parity) return active.label;

  const sciMatch = parity.value.match(/\b\d+\.\d+e[+-]\d+\b/i);
  if (sciMatch) {
    const isMaxAbs = /maxAbsDiff/i.test(parity.metric);
    return `${active.label} · ${isMaxAbs ? 'maxAbs' : 'relRMS'} ${sciMatch[0]}`;
  }
  return `${active.label} · upstream port`;
}

/**
 * Resolve a ModelCardSpec for a workflow adapter (or direct card ID) with optional
 * parameter-aware specialization (LocalFold checkpoint, ProteinMPNN family, RFD4 encoder mode,
 * or selected precision tier). Returns null for non-ML built-in CPU utilities.
 */
export function getModelCardForAdapter(
  adapterId: string,
  params?: Record<string, unknown>,
): ModelCardSpec | null {
  const base =
    MODEL_CARDS[adapterId] ??
    Object.values(MODEL_CARDS).find((c) => c.adapterId === adapterId) ??
    null;
  if (!base) return null;

  let resolved: ModelCardSpec = {
    ...base,
    precisions: base.precisions.map((p) => ({
      ...p,
      parityVsPytorchFp32: p.parityVsPytorchFp32 ? { ...p.parityVsPytorchFp32 } : null,
    })),
    limitations: [...base.limitations],
  };

  if (adapterId === 'localfold-af2' && typeof params?.model === 'string') {
    const opt = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === params.model);
    const meta = LOCALFOLD_CHECKPOINT_META[params.model];
    if (opt) {
      resolved.displayName = `LocalFold · ${opt.shortLabel}`;
      resolved.license = `Beer-Ware (runtime) · ${opt.license} (${opt.shortLabel} weights)`;
      if (resolved.precisions[0]) {
        resolved.precisions[0].approxSizeMB = opt.approxMB;
        if (meta) {
          resolved.precisions[0].quantizationDetail = meta.quantizationDetail;
        }
      }
      if (meta) {
        resolved.weightsRevision = meta.weightsRevision;
        resolved.limitations = [meta.familyNote, ...resolved.limitations];
      }
    }
  } else if (adapterId === 'web-protein-mpnn' && typeof params?.family === 'string') {
    const variant = MPNN_VARIANT_OPTIONS.find((v) => v.family === params.family);
    if (variant) {
      resolved.displayName = variant.label;
      if (variant.family === 'ca' && resolved.precisions[0]) {
        resolved.precisions[0].quantizationDetail =
          'Float32 ONNX Cα-only weights (~9.5 MB) executed via ONNX Runtime Web (backend: auto — WebGPU with WASM fallback)';
      }
    }
  } else if (
    adapterId === 'rfd4-proteina' &&
    (params?.mode === 'partial' || params?.mode === 'inverse-folding')
  ) {
    if (resolved.precisions[0]) {
      resolved.precisions[0].approxSizeMB = 469;
      resolved.precisions[0].quantizationDetail =
        'Block-64 INT8 sharded weights (~330 MB trunk + decoder + ~138 MB autoencoder encoder = ~469 MB) uploaded as packed FP16 on WebGPU ({ half: true })';
    }
  }

  if (
    typeof params?.precision === 'string' &&
    resolved.precisions.some((p) => p.id === params.precision)
  ) {
    resolved.activePrecisionId = params.precision;
  }

  return resolved;
}
