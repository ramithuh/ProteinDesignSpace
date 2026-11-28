// Layer 3 Plugin manifest — artios-potts (ArtiosPotts all-atom ligand-aware Potts designer & scorer)
//
// Declared here (inside the plugin folder) so the manifest, the worker and the unit tests all read
// one object. `engine/registry.ts` carries a byte-identical literal under the `'artios-potts'` key;
// `artios-potts.test.ts` asserts the two stay in sync once the registry entry is wired up.

import type { AdapterManifest } from '$lib/engine/types';

export const ARTIOS_POTTS_WEIGHTS_BASE_URL =
  'https://huggingface.co/ramithuh/artios-webgpu/resolve/main/potts';

/** Weight-bundle download size per precision, measured on the exported bundle (decimal MB). */
export const ARTIOS_POTTS_WEIGHTS_MB = { int8: 2.7, fp16: 5.2, fp32: 10.4 } as const;

export const ARTIOS_POTTS_MANIFEST: AdapterManifest = {
  id: 'artios-potts',
  displayName: 'ArtiosPotts',
  shortName: 'ArtiosPotts',
  // `inverse-fold` is the "Sequence Design" stage in this domain (STAGE_LABELS in graph-utils.ts);
  // there is no separate `design` stage in the protein-biopolymer DomainPack.
  stage: 'inverse-fold',
  runtime: 'wgsl-custom',
  estimatedVramMB: 180,
  approxWeightsMB: ARTIOS_POTTS_WEIGHTS_MB.fp16,
  precisionWeightsMB: { ...ARTIOS_POTTS_WEIGHTS_MB },
  licenseNote: 'Apache-2.0 (Ligo-Biosciences/artios · WebGPU port ramithuh/artios-webgpu)',
  description:
    'All-atom ligand-aware Potts sequence design and scoring. A sequence-blind MPNN encoder/decoder emits a factorized Potts model over a 32-nearest-neighbour token graph (per-residue fields h and pairwise couplings J), which is then either sampled by chromatic Gibbs or DLMC MCMC with temperature annealing, low-complexity penalties and optional classifier-free ligand guidance, or evaluated in closed form to score the input sequence by Potts energy and per-residue pseudo-log-likelihood. Small molecules, cofactors, metals and nucleic-acid heavy atoms enter as context atom tokens.',
  inputs: {
    structure: { type: 'StructureArtifact' },
  },
  outputs: {
    sequences: { type: 'SequenceSet' },
    track: { type: 'ResidueTrack', required: false },
  },
  modeParamKey: 'mode',
  modes: [
    {
      id: 'design',
      label: 'Design Sequences',
      description:
        'Sample new sequences from the Potts model with chromatic Gibbs or DLMC MCMC (annealed temperature, low-complexity penalty, optional ligand CFG).',
    },
    {
      id: 'score',
      label: 'Score Input Sequence',
      description:
        'Score the native sequence already present in the structure: Potts energy, per-residue pseudo-log-likelihood and per-node energy. No MCMC is run.',
    },
  ],
  paramSchema: [
    {
      key: 'mode',
      label: 'Task',
      kind: 'select',
      tier: 'primary',
      defaultValue: 'design',
      options: [
        { value: 'design', label: 'Design sequences (Potts MCMC)' },
        { value: 'score', label: 'Score input sequence (energy + PLL)' },
      ],
      description:
        'Design samples sequences from the Potts model; Score evaluates the structure’s own sequence in closed form.',
    },
    {
      key: 'numSequences',
      label: 'Sequences',
      kind: 'integer',
      tier: 'primary',
      defaultValue: 8,
      min: 1,
      max: 64,
      step: 1,
      modes: ['design'],
      description: 'Number of independent MCMC chains to run (one designed sequence each).',
    },
    {
      key: 'temperature',
      label: 'Temp',
      kind: 'number',
      tier: 'primary',
      defaultValue: 0.1,
      min: 0.01,
      max: 2.0,
      step: 0.05,
      upstreamDefaultHint: 'Upstream default: 0.1',
      modes: ['design'],
      description: 'Final sampling temperature after annealing (lower = greedier, higher recovery).',
    },
    {
      key: 'numSteps',
      label: 'MCMC sweeps',
      kind: 'integer',
      tier: 'primary',
      defaultValue: 150,
      min: 10,
      max: 500,
      step: 10,
      upstreamDefaultHint: 'Upstream default: 150',
      modes: ['design'],
      description:
        'Gibbs / DLMC sweeps per sequence. The first 80% anneal from the initial temperature down to Temp.',
    },
    {
      key: 'ligandCutoff',
      label: 'Design within',
      kind: 'number',
      tier: 'primary',
      defaultValue: 0,
      min: 0,
      max: 20,
      step: 0.5,
      unit: 'Å',
      description:
        'Restrict design to residues with a backbone atom within this distance of a context (ligand / cofactor / metal / nucleic-acid) atom; 0 designs every residue. Ignored when the input has no context atoms.',
    },
    {
      key: 'fixedPositions',
      label: 'Fixed positions',
      kind: 'residue-list',
      tier: 'primary',
      studioRole: 'fixed',
      allowBareChain: true,
      defaultValue: '',
      placeholder: 'e.g. A5, A10-14',
      modes: ['design'],
      description: 'Residues or whole chains held at their input identity during Potts sampling.',
    },
    {
      key: 'sampler',
      label: 'Sampler',
      kind: 'select',
      tier: 'advanced',
      defaultValue: 'gibbs',
      options: [
        { value: 'gibbs', label: 'Chromatic Gibbs (default)' },
        { value: 'dlmc', label: 'DLMC + Metropolis–Hastings' },
      ],
      modes: ['design'],
      description:
        'Chromatic Gibbs updates graph-colour groups in place; DLMC proposes all positions at once and accepts via Metropolis–Hastings.',
    },
    {
      key: 'temperatureInit',
      label: 'Initial temp',
      kind: 'number',
      tier: 'advanced',
      defaultValue: 1.0,
      min: 0.01,
      max: 5.0,
      step: 0.05,
      upstreamDefaultHint: 'Upstream default: 1.0',
      modes: ['design'],
      description:
        'Annealing start temperature. The schedule interpolates linearly to Temp over the first 80% of sweeps.',
    },
    {
      key: 'cfgScale',
      label: 'Ligand guidance (CFG)',
      kind: 'number',
      tier: 'advanced',
      defaultValue: 0,
      min: 0,
      max: 5.0,
      step: 0.1,
      modes: ['design'],
      description:
        'Classifier-free guidance weight blending the ligand-conditioned and ligand-free Potts models (p = p_uncond + w·(p_cond − p_uncond)). 0 disables guidance and skips the second forward pass; 1 is the plain conditional model; >1 amplifies ligand conditioning. Requires context atoms in the input.',
    },
    {
      key: 'complexityWeight',
      label: 'Complexity penalty',
      kind: 'number',
      tier: 'advanced',
      defaultValue: 0.6,
      min: 0,
      max: 5.0,
      step: 0.1,
      upstreamDefaultHint: 'Upstream default: 0.6',
      modes: ['design'],
      description:
        'Weight of the low-complexity-proteome (LCP) windowed-entropy penalty that suppresses homopolymeric and repetitive stretches. 0 disables it.',
    },
    {
      key: 'omitAAs',
      label: 'Omit amino acids',
      kind: 'aa-set',
      tier: 'advanced',
      defaultValue: 'CM',
      placeholder: 'e.g. CM',
      upstreamDefaultHint: 'Upstream default: C, M excluded',
      modes: ['design'],
      description:
        '1-letter amino acid codes excluded from the design alphabet. Upstream ArtiosPotts excludes cysteine and methionine by default.',
    },
    {
      key: 'precision',
      label: 'Weight Precision',
      kind: 'select',
      tier: 'advanced',
      defaultValue: 'fp16',
      options: [
        { value: 'fp16', label: 'FP16 (default · ~5.2 MB)' },
        { value: 'int8', label: 'INT8 (compact · ~2.7 MB)' },
        { value: 'fp32', label: 'FP32 (reference · ~10.4 MB)' },
      ],
      description:
        'Storage precision of the tiled-matmul weights. Activations and accumulation stay FP32 on every tier.',
    },
    {
      key: 'seed',
      label: 'Seed',
      kind: 'integer',
      tier: 'advanced',
      defaultValue: 42,
      min: 0,
      max: 2_147_483_647,
      step: 1,
      modes: ['design'],
      description: 'Base PRNG seed; chain i uses seed + i.',
    },
  ],
};
