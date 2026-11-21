/**
 * Protein structure constants and default V1 model / diffusion configuration for Genie 3.
 *
 * Mirrors:
 * - `genie3/generation/np/protein_constants.py`
 * - `genie3/generation/config/common.py`
 * - `genie3/generation/config/model/v1.py`
 * - `genie3/generation/config/diffusion/ddpm.py`
 * - `genie3/generation/config/diffusion/sampler/ddim.py`
 */

export const ATOM_TYPES = Object.freeze([
  'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1',
  'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE',
  'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1',
  'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT',
]);

export const ATOM_SYMBOLS = Object.freeze(['N', 'C', 'O', 'S']);

/** Canonical 20 amino acid 3-letter codes in Genie 3 / OpenFold alphabetical order. */
export const PROTEIN_RESTYPES = Object.freeze([
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
]);

/** Canonical 20 amino acid 1-letter codes matching PROTEIN_RESTYPES index order. */
export const PROTEIN_RESTYPES_1 = 'ARNDCQEGHILKMFPSTWYV';

export const RESTYPE_1_TO_3 = Object.freeze({
  A: 'ALA', R: 'ARG', N: 'ASN', D: 'ASP', C: 'CYS',
  Q: 'GLN', E: 'GLU', G: 'GLY', H: 'HIS', I: 'ILE',
  L: 'LEU', K: 'LYS', M: 'MET', F: 'PHE', P: 'PRO',
  S: 'SER', T: 'THR', W: 'TRP', Y: 'TYR', V: 'VAL',
  X: 'UNK',
});

export const RESTYPE_3_TO_1 = Object.freeze({
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C',
  GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P',
  SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
  UNK: 'X',
});

export const ATOM_TYPE_INDEX = Object.freeze(
  Object.fromEntries(ATOM_TYPES.map((name, i) => [name, i])),
);

export const ATOM_SYMBOL_INDEX = Object.freeze(
  Object.fromEntries(ATOM_SYMBOLS.map((sym, i) => [sym, i])),
);

export const RESTYPE_INDEX = Object.freeze(
  Object.fromEntries(PROTEIN_RESTYPES.map((name, i) => [name, i])),
);

export const NUM_ATOM_TYPES = ATOM_TYPES.length; // 37
export const NUM_ATOM_SYMBOLS = ATOM_SYMBOLS.length; // 4
export const NUM_RESTYPES = PROTEIN_RESTYPES.length; // 20
export const CA_ATOM_TYPE_INDEX = ATOM_TYPE_INDEX.CA; // 1
export const C_ATOM_SYMBOL_INDEX = ATOM_SYMBOL_INDEX.C; // 1

/**
 * Default V1 Genie 3 model & diffusion configuration.
 * All parameters can be overridden via config JSON / manifest metadata.
 */
export const DEFAULT_V1_CONFIG = Object.freeze({
  c_s: 384,
  c_z: 128,
  n_timestep: 1000,
  max_n_res: 256,
  max_n_chain: 4,
  eps: 1e-10,
  inf: 1e10,

  single_feature_net: Object.freeze({
    c_s_input: 1158,
    c_s: 384,
    n_timestep: 1000,
    c_pos_emb: 256,
    c_chain_emb: 64,
    c_timestep_emb: 512,
    max_n_res: 256,
    max_n_chain: 4,
  }),

  pair_feature_net: Object.freeze({
    c_s: 384,
    c_z: 128,
    max_relative_index: 32,
    max_relative_chain: 2,
    template_dist_bin_min: 0.0,
    template_dist_bin_max: 32.0,
    template_dist_no_bins: 65,
    cond_template_dist_bin_min: 0.0,
    cond_template_dist_bin_max: 32.0,
    cond_template_dist_no_bins: 65,
  }),

  pair_transform_net: Object.freeze({
    c_s: 384,
    c_p: 128,
    n_pair_transform_layer: 5,
    c_hidden_mul: 128,
    pair_transition_n: 4,
    c_hidden_ipa: 16,
    n_head_ipa: 12,
    n_structure_transition_layer: 1,
    n_global_token: 10,
  }),

  sequence_net: Object.freeze({
    c_s: 384,
    c_hidden: 256,
  }),

  structure_net: Object.freeze({
    c_s: 384,
    c_z: 128,
    n_structure_layer: 8,
    n_structure_block: 1,
    c_hidden_ipa: 16,
    n_head_ipa: 12,
    n_qk_point: 4,
    n_v_point: 8,
    n_structure_transition_layer: 1,
  }),

  diffusion: Object.freeze({
    n_timestep: 1000,
    schedule: 'cosine',
  }),

  sampler: Object.freeze({
    n_sample_step: 100,
    direction_scale: 0.8,
    noise_scale: 1.0,
    eta: 1.0,
    predict_sequence: false,
    sequence_temperature: 0.1,
  }),
});

/**
 * Deep-merge a partial config override (e.g. from manifest.json or upstream config.yaml)
 * onto DEFAULT_V1_CONFIG.
 *
 * Also handles nested upstream shapes like `{ model: { model: { ... } }, diffusion: { ... } }`
 * or `{ single_feature_net, pair_feature_net, ... }`.
 *
 * @param {object} [rawCfg]
 * @returns {object}
 */
export function resolveGenie3Config(rawCfg = null) {
  if (!rawCfg) {
    return {
      ...DEFAULT_V1_CONFIG,
      single_feature_net: { ...DEFAULT_V1_CONFIG.single_feature_net },
      pair_feature_net: { ...DEFAULT_V1_CONFIG.pair_feature_net },
      pair_transform_net: { ...DEFAULT_V1_CONFIG.pair_transform_net },
      sequence_net: { ...DEFAULT_V1_CONFIG.sequence_net },
      structure_net: { ...DEFAULT_V1_CONFIG.structure_net },
      diffusion: { ...DEFAULT_V1_CONFIG.diffusion },
      sampler: { ...DEFAULT_V1_CONFIG.sampler },
    };
  }

  // Unwrap upstream YAML shapes if present
  const modelRoot =
    rawCfg.model?.model ??
    (rawCfg.model?.single_feature_net ? rawCfg.model : rawCfg);
  const diffRoot = rawCfg.diffusion?.diffusion ?? rawCfg.diffusion ?? {};
  const samplerRoot =
    rawCfg.sampler?.sampler ??
    rawCfg.sampler ??
    rawCfg.default_sampler ??
    {};

  const c_s = modelRoot.c_s ?? modelRoot.single_feature_net?.c_s ?? DEFAULT_V1_CONFIG.c_s;
  const c_z = modelRoot.c_z ?? modelRoot.pair_feature_net?.c_z ?? DEFAULT_V1_CONFIG.c_z;
  const n_timestep =
    diffRoot.n_timestep ??
    modelRoot.n_timestep ??
    modelRoot.single_feature_net?.n_timestep ??
    DEFAULT_V1_CONFIG.n_timestep;
  const max_n_res =
    modelRoot.max_n_res ??
    modelRoot.single_feature_net?.max_n_res ??
    DEFAULT_V1_CONFIG.max_n_res;
  const max_n_chain =
    modelRoot.max_n_chain ??
    modelRoot.single_feature_net?.max_n_chain ??
    DEFAULT_V1_CONFIG.max_n_chain;

  return {
    c_s,
    c_z,
    n_timestep,
    max_n_res,
    max_n_chain,
    eps: rawCfg.eps ?? DEFAULT_V1_CONFIG.eps,
    inf: rawCfg.inf ?? DEFAULT_V1_CONFIG.inf,
    single_feature_net: {
      ...DEFAULT_V1_CONFIG.single_feature_net,
      c_s,
      n_timestep,
      max_n_res,
      max_n_chain,
      ...(modelRoot.single_feature_net ?? {}),
    },
    pair_feature_net: {
      ...DEFAULT_V1_CONFIG.pair_feature_net,
      c_s,
      c_z,
      ...(modelRoot.pair_feature_net ?? {}),
    },
    pair_transform_net: {
      ...DEFAULT_V1_CONFIG.pair_transform_net,
      c_s,
      c_p: c_z,
      ...(modelRoot.pair_transform_net ?? {}),
    },
    sequence_net: {
      ...DEFAULT_V1_CONFIG.sequence_net,
      c_s,
      ...(modelRoot.sequence_net ?? {}),
    },
    structure_net: {
      ...DEFAULT_V1_CONFIG.structure_net,
      c_s,
      c_z,
      ...(modelRoot.structure_net ?? {}),
    },
    diffusion: {
      ...DEFAULT_V1_CONFIG.diffusion,
      n_timestep,
      ...diffRoot,
    },
    sampler: {
      ...DEFAULT_V1_CONFIG.sampler,
      ...samplerRoot,
    },
  };
}
