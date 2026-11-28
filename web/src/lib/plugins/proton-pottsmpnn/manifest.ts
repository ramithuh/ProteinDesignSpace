// Layer 3 Plugin manifest — proton-pottsmpnn (Proton-PottsMPNN pH-aware Potts designer & scorer)
//
// Declared here (inside the plugin folder) so the manifest, the worker and the unit tests all read
// one object. `engine/registry.ts` re-exports this exact object under the `'proton-pottsmpnn'` key;
// `proton-pottsmpnn.test.ts` asserts the two stay in sync once the registry entry is wired up.
//
// Every parameter below maps 1:1 onto a real option of the vendored runtime
// (`static/vendor/proton-pottsmpnn/shared/index.js`): `blockDescent({ centres, designable, lambda,
// blockSize, zscaleMode, sweepOrder, temperature, maxRounds, forbiddenTokens, rng })`,
// `centreNeighbourhood({ perCentre, max })`, `structureFromPDB(text, { overrides })` and
// `resolveCentres(structure, centres)`. Nothing is invented: the upstream optimizer has no
// `alpha`/`beta`/`pH` knob — selectivity vs stability is the single z-scaled `lambda`.

import type { AdapterManifest } from '$lib/engine/types';

export const PROTON_POTTSMPNN_WEIGHTS_BASE_URL =
  'https://huggingface.co/ramithuh/proton-pottsmpnn-webgpu/resolve/main';

/** Weight-bundle download size per precision, measured on the exported bundles (decimal MB). */
export const PROTON_POTTSMPNN_WEIGHTS_MB = { int8: 1.87, fp16: 3.6, fp32: 7.1 } as const;

/** The 9 explicit v6 protonation microstates, mirroring `shared/index.js :: V6_PROTONATION_TOKENS`. */
export const PROTON_POTTSMPNN_STATE_TOKENS = [
  'HIS-P', 'HIS-S', 'HIS-A',
  'ASP-P', 'ASP-D', 'ASP-A',
  'GLU-P', 'GLU-D', 'GLU-A',
] as const;

/** Protonated microstates that have a deprotonated contrast partner (`V6_DEP_MAP` keys). */
export const PROTON_POTTSMPNN_CENTRE_TOKENS = ['HIS-P', 'ASP-P', 'GLU-P'] as const;

export const PROTON_POTTSMPNN_MANIFEST: AdapterManifest = {
  id: 'proton-pottsmpnn',
  displayName: 'Proton-PottsMPNN',
  shortName: 'Proton-PottsMPNN',
  // `inverse-fold` is the "Sequence Design" stage in this domain (STAGE_LABELS in graph-utils.ts);
  // there is no separate `design` stage in the protein-biopolymer DomainPack.
  stage: 'inverse-fold',
  runtime: 'wgsl-custom',
  estimatedVramMB: 200,
  approxWeightsMB: PROTON_POTTSMPNN_WEIGHTS_MB.fp16,
  precisionWeightsMB: { ...PROTON_POTTSMPNN_WEIGHTS_MB },
  licenseNote:
    'MIT (christian-creator/ProtonPottsMPNN · WebGPU port ramithuh/proton-pottsmpnn-webgpu) · BSD-3-Clause (PottsMPNN from IPD rc-foundry)',
  description:
    'pH-aware Potts sequence design and scoring in an explicit protonation-state vocabulary. A backbone-only MPNN encoder (3 EncLayers over a 48-nearest-neighbour Cα graph with 400 atom-pair RBFs) emits a factorized Potts model over 30 tokens — the 20 amino acids, UNK and the 9 microstates HIS-P/S/A, ASP-P/D/A, GLU-P/D/A — giving per-residue fields h and pairwise couplings J. Sequences are then designed by exact block coordinate descent on the z-scaled objective (1−λ)·z(E_global) + λ·z(Σ (E_P − E_D)), which trades plain stability against the pH-selective gap of user-pinned protonation centres, or the input sequence is scored in closed form (E_global, per-centre E_P/E_D/ΔE and per-residue site energies). Titratable residues default to their physiological state and can be pinned explicitly; no HBPLUS/propka labeller is bundled.',
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
        'pH-selective (or plain stability) design by exact block coordinate descent on the Potts model, with protonation centres pinned as fixed context.',
    },
    {
      id: 'score',
      label: 'Score Input Sequence',
      description:
        'Evaluate the structure’s own sequence in closed form: E_global, per-centre E_P / E_D / ΔE = E_P − E_D and per-residue site energies. No optimization is run.',
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
        { value: 'design', label: 'Design sequences (pH block descent)' },
        { value: 'score', label: 'Score input sequence (Potts energies)' },
      ],
      description:
        'Design runs the pH-selective block-descent optimizer; Score evaluates the input sequence’s Potts energies and selective gap in closed form.',
    },
    {
      key: 'protonationCentres',
      label: 'pH centres',
      kind: 'text',
      tier: 'primary',
      defaultValue: '',
      placeholder: 'e.g. A42=HIS-P, A58=GLU-P',
      description:
        'Residues whose protonated microstate the design should favour, written as `<residue>=<token>` with a token from HIS-P, ASP-P or GLU-P. Each centre is pinned protonated and contrasted against its deprotonated partner (HIS-S / ASP-D / GLU-D) to form the selective gap E_P − E_D. Centres are never redesigned. Leave empty for plain stability design (λ is then inert) or to score E_global only.',
    },
    {
      key: 'protonationOverrides',
      label: 'Protonation overrides',
      kind: 'text',
      tier: 'primary',
      defaultValue: '',
      placeholder: 'e.g. A42=HIS-P, A58=ASP-P',
      description:
        'Input protonation states for titratable residues, written as `<residue>=<token>` over the 30-token v6 vocabulary. Unpinned ASP / GLU / HIS default to their physiological state at pH ≈ 7.4 (ASP-D, GLU-D, HIS-S). This control replaces the upstream HBPLUS + propka/FLAML labeller, which is not bundled.',
    },
    {
      key: 'lambda',
      label: 'pH selectivity (λ)',
      kind: 'number',
      tier: 'primary',
      defaultValue: 0.5,
      min: 0,
      max: 1,
      step: 0.05,
      modes: ['design'],
      upstreamDefaultHint: 'Upstream blockDescent default: 1.0 (pure selectivity)',
      description:
        'Relative weight of the two z-scaled terms in (1−λ)·z(E_global) + λ·z(Σ (E_P − E_D)): 0 is pure stability, 1 is pure selectivity. Each term is divided by the spread of its own frozen deltas, which is what makes λ a true relative weight. Requires at least one pH centre; λ = 1 with no centre has no objective at all and is rejected.',
    },
    {
      key: 'numSequences',
      label: 'Sequences',
      kind: 'integer',
      tier: 'primary',
      defaultValue: 4,
      min: 1,
      max: 32,
      step: 1,
      modes: ['design'],
      description:
        'Independent descents, run with seed + i. Block descent at Temp 0 is an exact argmin and therefore deterministic, so distinct sequences need Temp > 0; at Temp 0 only one descent is run.',
    },
    {
      key: 'temperature',
      label: 'Temp',
      kind: 'number',
      tier: 'primary',
      defaultValue: 0,
      min: 0,
      max: 2,
      step: 0.05,
      modes: ['design'],
      upstreamDefaultHint: 'Upstream default: 0 (exact block argmin)',
      description:
        'Block sampling temperature. 0 commits the exact argmin over all V^blockSize joint assignments; above 0 the block is drawn from the Boltzmann distribution of those assignments.',
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
      description:
        'Residues or whole chains held at their input token for the whole descent (removed from the designable set). pH centres are always fixed on top of this.',
    },
    {
      key: 'designScope',
      label: 'Design scope',
      kind: 'select',
      tier: 'advanced',
      defaultValue: 'centre-shell',
      options: [
        { value: 'centre-shell', label: 'Coupled shell around the pH centres (default)' },
        { value: 'all-positions', label: 'Every residue in the structure' },
      ],
      modes: ['design'],
      description:
        'Which positions the descent may change. The shell is each centre’s coupled kNN in both graph directions (`centreNeighbourhood`); with no centre declared it falls back to every residue.',
    },
    {
      key: 'shellPerCentre',
      label: 'Shell per centre',
      kind: 'integer',
      tier: 'advanced',
      defaultValue: 8,
      min: 1,
      max: 32,
      step: 1,
      modes: ['design'],
      description:
        'Neighbours taken per centre in each graph direction when Design scope is the coupled shell (`centreNeighbourhood.perCentre`).',
    },
    {
      key: 'maxDesignable',
      label: 'Max designable',
      kind: 'integer',
      tier: 'advanced',
      defaultValue: 12,
      min: 0,
      max: 256,
      step: 1,
      modes: ['design'],
      description:
        'Cap on the designable set, kept by best kNN rank to the centres (`centreNeighbourhood.max`). 0 removes the cap. Applies to both design scopes.',
    },
    {
      key: 'blockSize',
      label: 'Block size',
      kind: 'integer',
      tier: 'advanced',
      defaultValue: 2,
      min: 1,
      max: 3,
      step: 1,
      modes: ['design'],
      upstreamDefaultHint: 'Upstream default: 2',
      description:
        'Positions optimized jointly per step: each visited position plus its closest designable partners. All 30^blockSize assignments are scored exactly, so cost grows as 30^blockSize. Only stability has within-block pairwise terms (selectivity is exactly unary), so this changes the answer solely through stability coupling. 1 is plain greedy coordinate descent.',
    },
    {
      key: 'maxRounds',
      label: 'Max sweeps',
      kind: 'integer',
      tier: 'advanced',
      defaultValue: 10,
      min: 1,
      max: 50,
      step: 1,
      modes: ['design'],
      upstreamDefaultHint: 'Upstream default: 10',
      description:
        'Upper bound on full sweeps over the designable set. The descent stops early as soon as a whole sweep makes no change.',
    },
    {
      key: 'sweepOrder',
      label: 'Sweep order',
      kind: 'select',
      tier: 'advanced',
      defaultValue: 'position',
      options: [
        { value: 'position', label: 'Sequence position (default)' },
        { value: 'knn', label: 'Best kNN rank to a centre' },
        { value: 'energy', label: 'Largest centre-energy coupling' },
      ],
      modes: ['design'],
      description:
        'Order in which the designable positions are visited inside a sweep. Block descent is not order-invariant, so this changes the trajectory (and usually the optimum reached).',
    },
    {
      key: 'zscaleMode',
      label: 'z-scale source',
      kind: 'select',
      tier: 'advanced',
      defaultValue: 'block',
      options: [
        { value: 'block', label: 'Block joint spread (default)' },
        { value: 'single-mutation', label: 'Single-mutation spread' },
      ],
      modes: ['design'],
      description:
        'How the two frozen z-scales are measured before the first sweep: from the within-block joint spread over all 30^blockSize assignments, or from the single-mutation delta tables.',
    },
    {
      key: 'omitAAs',
      label: 'Omit amino acids',
      kind: 'aa-set',
      tier: 'advanced',
      defaultValue: '',
      placeholder: 'e.g. CM',
      modes: ['design'],
      upstreamDefaultHint: 'Upstream default: nothing omitted',
      description:
        '1-letter codes excluded from the design alphabet. Every v6 token of that parent is forbidden, so omitting H also removes HIS-P and HIS-S. UNK and the three ambiguous states (HIS-A, ASP-A, GLU-A) are always forbidden, as upstream.',
    },
    {
      key: 'titratableTokens',
      label: 'Titratable tokens',
      kind: 'select',
      tier: 'advanced',
      defaultValue: 'upstream',
      options: [
        { value: 'upstream', label: 'Upstream: states + bare ASP/GLU/HIS (default)' },
        { value: 'explicit-states', label: 'Explicit protonation states only' },
      ],
      modes: ['design'],
      description:
        'Upstream `validTokenMask` lets the optimizer also pick the state-unspecified parent tokens ASP, GLU and HIS. Explicit-states-only forbids those three, so every titratable residue the descent writes carries a protonation state.',
    },
    {
      key: 'precision',
      label: 'Weight Precision',
      kind: 'select',
      tier: 'advanced',
      defaultValue: 'fp16',
      options: [
        { value: 'fp16', label: 'FP16 (default · ~3.6 MB)' },
        { value: 'int8', label: 'INT8 (compact · ~1.87 MB)' },
        { value: 'fp32', label: 'FP32 (reference · ~7.1 MB)' },
      ],
      description:
        'Storage precision of the tiled-matmul weights. Activations and accumulation stay FP32 on every tier, and `shader-f16` is never required.',
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
      description:
        'Base PRNG seed for Boltzmann block sampling; descent i uses seed + i. Inert at Temp 0, where the descent is deterministic.',
    },
  ],
  crossValidate: (p) => {
    const errs: string[] = [];
    const mode = String(p.mode ?? 'design');
    const tokenSet = new Set<string>([
      'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
      'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
      'UNK', ...PROTON_POTTSMPNN_STATE_TOKENS,
    ]);
    const centreSet = new Set<string>(PROTON_POTTSMPNN_CENTRE_TOKENS);

    /**
     * Shared `<residue>=<token>` syntax check for both pinning controls.
     * Kept byte-identical in behaviour to `jobs.ts :: splitProtonationEntries` +
     * `parseProtonationEntry`, so the UI never accepts a spec the worker then rejects.
     */
    const checkSpec = (raw: unknown, key: string, allowed: ReadonlySet<string>): number => {
      const normalized = String(raw ?? '').trim().replace(/\s*([=:])\s*/g, '$1');
      if (!normalized) return 0;
      let count = 0;
      for (const entry of normalized.split(/[,;\s]+/).filter(Boolean)) {
        const m = /^(.+)[=:]([A-Za-z][A-Za-z-]*)$/.exec(entry);
        if (!m) {
          errs.push(
            `Invalid ${key} entry "${entry}" (expected "<residue>=<token>", e.g. "A42=HIS-P").`,
          );
          continue;
        }
        const token = m[2].toUpperCase();
        if (!allowed.has(token)) {
          errs.push(
            `Invalid ${key} token "${m[2]}" (expected one of ${[...allowed].join(', ')}).`,
          );
          continue;
        }
        count++;
      }
      return count;
    };

    const centres = checkSpec(p.protonationCentres, 'protonationCentres', centreSet);
    checkSpec(p.protonationOverrides, 'protonationOverrides', tokenSet);

    if (mode === 'design') {
      const lambda = Number(p.lambda ?? 0.5);
      if (Number.isFinite(lambda) && lambda >= 1 && centres === 0) {
        errs.push(
          'λ = 1 is pure pH selectivity but no pH centre is declared, which leaves the descent with no objective. Add a pH centre (e.g. "A42=HIS-P") or lower λ below 1.',
        );
      }
    }
    return errs;
  },
};
