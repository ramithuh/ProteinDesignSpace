/**
 * Vocabularies, element table and residue groupings shared by the CPU and WebGPU
 * ArtiosPotts ports.
 *
 * Every table here mirrors a constant of the reference PyTorch implementation;
 * the comment on each one names its source so drift is easy to spot.
 */

/** `artios.common.residue_names.PROTEIN_TYPES` — the 20 standard amino acids. */
export const PROTEIN_THREE = Object.freeze([
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
]);

/** `artios.common.residue_names.PROTEIN_TYPES_ONE_LETTER`, same order. */
export const ONE_LETTER = 'ARNDCQEGHILKMFPSTWYV';

/**
 * `artios.common.constants.STD_RESIDUES_AND_GAP` — the 32 restype channels:
 * 20 amino acids, UNK, RNA A/G/C/U, DNA DA/DG/DC/DT, RNA-unk, DNA-unk, gap.
 */
export const RESTYPES = Object.freeze([
  ...PROTEIN_THREE, 'UNK',
  'A', 'G', 'C', 'U', 'DA', 'DG', 'DC', 'DT', 'N', 'DN', '-',
]);

export const NUM_RESIDUE_TYPES = RESTYPES.length; // 32
export const NUM_STANDARD_AA = PROTEIN_THREE.length; // 20
export const UNK_INDEX = RESTYPES.indexOf('UNK'); // 20
export const CYS_INDEX = RESTYPES.indexOf('CYS'); // 4
export const MET_INDEX = RESTYPES.indexOf('MET'); // 12

export const RESTYPE_INDEX = Object.freeze(
  Object.fromEntries(RESTYPES.map((name, i) => [name, i])),
);

/** One-letter code -> restype channel, with `X` -> UNK. */
export const ONE_LETTER_TO_INDEX = Object.freeze({
  ...Object.fromEntries([...ONE_LETTER].map((c, i) => [c, i])),
  X: UNK_INDEX,
});

/** restype channel -> one-letter code (`X` for anything without one). */
export const INDEX_TO_ONE_LETTER = Object.freeze(
  RESTYPES.map((name) => {
    const i = PROTEIN_THREE.indexOf(name);
    return i >= 0 ? ONE_LETTER[i] : 'X';
  }),
);

/** `artios.common.constants.BACKBONE_DEFS["protein"]`; slot 4 is the virtual CB. */
export const PROTEIN_BACKBONE = Object.freeze(['N', 'CA', 'C', 'O']);
export const NUM_BACKBONE_ATOMS = 5;
export const BACKBONE_PAIRS = NUM_BACKBONE_ATOMS * NUM_BACKBONE_ATOMS; // 25

/**
 * `artios.common.constants.get_all_elems()` lists RDKit symbols for atomic
 * numbers 1..118 followed by ten `UNK_ELEM` slots, so an element's one-hot
 * channel is `Z - 1`.
 */
export const NUM_ELEMENTS = 128;

export const ELEMENT_Z = Object.freeze({
  H: 1, HE: 2, LI: 3, BE: 4, B: 5, C: 6, N: 7, O: 8, F: 9, NE: 10,
  NA: 11, MG: 12, AL: 13, SI: 14, P: 15, S: 16, CL: 17, AR: 18, K: 19, CA: 20,
  SC: 21, TI: 22, V: 23, CR: 24, MN: 25, FE: 26, CO: 27, NI: 28, CU: 29, ZN: 30,
  GA: 31, GE: 32, AS: 33, SE: 34, BR: 35, KR: 36, RB: 37, SR: 38, Y: 39, ZR: 40,
  NB: 41, MO: 42, TC: 43, RU: 44, RH: 45, PD: 46, AG: 47, CD: 48, IN: 49, SN: 50,
  SB: 51, TE: 52, I: 53, XE: 54, CS: 55, BA: 56, LA: 57, CE: 58, PR: 59, ND: 60,
  PM: 61, SM: 62, EU: 63, GD: 64, TB: 65, DY: 66, HO: 67, ER: 68, TM: 69, YB: 70,
  LU: 71, HF: 72, TA: 73, W: 74, RE: 75, OS: 76, IR: 77, PT: 78, AU: 79, HG: 80,
  TL: 81, PB: 82, BI: 83, PO: 84, AT: 85, RN: 86, FR: 87, RA: 88, AC: 89, TH: 90,
  PA: 91, U: 92, NP: 93, PU: 94, AM: 95, CM: 96, BK: 97, CF: 98, ES: 99, FM: 100,
  MD: 101, NO: 102, LR: 103, RF: 104, DB: 105, SG: 106, BH: 107, HS: 108, MT: 109,
  DS: 110, RG: 111, CN: 112, NH: 113, FL: 114, MC: 115, LV: 116, TS: 117, OG: 118,
});

const indicesOf = (...names) => Object.freeze(names.map((n) => RESTYPE_INDEX[n]));

/** `artios.common.constants.AROMATIC_INDICES` — PHE/TRP/TYR (no HIS). */
export const AROMATIC_INDICES = indicesOf('PHE', 'TRP', 'TYR');
/** `artios.common.constants.CHARGED_INDICES`. */
export const CHARGED_INDICES = indicesOf('ASP', 'GLU', 'ARG', 'LYS', 'HIS');
/** `artios.model.modules.penalty._POS_CHARGE_INDICES` / `_NEG_CHARGE_INDICES`. */
export const POS_CHARGE_INDICES = indicesOf('LYS', 'ARG');
export const NEG_CHARGE_INDICES = indicesOf('ASP', 'GLU');

/**
 * `artios.model.modules.potts.RESIDUE_GROUPS` — note AROMATIC here *does*
 * include HIS, unlike `AROMATIC_INDICES` above.
 */
export const RESIDUE_GROUPS = Object.freeze({
  hydrophobic: indicesOf('ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'),
  aromatic: indicesOf('PHE', 'TYR', 'TRP', 'HIS'),
  anionic: indicesOf('ASP', 'GLU'),
  cationic: indicesOf('ARG', 'LYS', 'HIS'),
  polar: indicesOf('SER', 'THR', 'ASN', 'GLN'),
});

/** `artios.model.artios_potts.ALPHABET_BIAS`. */
export const ALPHABET_BIAS = 1e4;

/**
 * `artios.common.residue_stats.AA_FREQUENCIES`, in `PROTEIN_TYPES` order.
 * Used by the composition and natural-prior complexity penalties.
 */
export const AA_FREQUENCIES = Object.freeze({
  swissprot: {
    mean: [0.0760, 0.0523, 0.0410, 0.0537, 0.0173, 0.0426, 0.0691, 0.0650, 0.0236, 0.0500, 0.0930, 0.0570, 0.0214, 0.0364, 0.0553, 0.0828, 0.0578, 0.0120, 0.0289, 0.0647],
    std: [0.0302, 0.0189, 0.0180, 0.0147, 0.0121, 0.0175, 0.0233, 0.0217, 0.0097, 0.0191, 0.0231, 0.0258, 0.0087, 0.0143, 0.0216, 0.0286, 0.0170, 0.0084, 0.0125, 0.0170],
  },
  ecoli: {
    mean: [0.0979, 0.0563, 0.0382, 0.0540, 0.0104, 0.0437, 0.0623, 0.0768, 0.0226, 0.0588, 0.1032, 0.0446, 0.0283, 0.0376, 0.0453, 0.0536, 0.0529, 0.0137, 0.0279, 0.0718],
    std: [0.0234, 0.0178, 0.0126, 0.0161, 0.0079, 0.0169, 0.0205, 0.0184, 0.0108, 0.0178, 0.0254, 0.0173, 0.0104, 0.0153, 0.0141, 0.0140, 0.0127, 0.0096, 0.0120, 0.0166],
  },
  bsubtilis: {
    mean: [0.0787, 0.0419, 0.0401, 0.0551, 0.0073, 0.0388, 0.0762, 0.0710, 0.0226, 0.0710, 0.0927, 0.0706, 0.0263, 0.0408, 0.0381, 0.0618, 0.0553, 0.0094, 0.0342, 0.0681],
    std: [0.0224, 0.0148, 0.0142, 0.0157, 0.0065, 0.0134, 0.0233, 0.0194, 0.0104, 0.0200, 0.0222, 0.0214, 0.0099, 0.0170, 0.0129, 0.0184, 0.0139, 0.0078, 0.0135, 0.0177],
  },
  yeast: {
    mean: [0.0597, 0.0439, 0.0601, 0.0606, 0.0111, 0.0410, 0.0672, 0.0529, 0.0212, 0.0647, 0.0927, 0.0731, 0.0207, 0.0418, 0.0445, 0.0857, 0.0600, 0.0095, 0.0325, 0.0572],
    std: [0.0230, 0.0163, 0.0207, 0.0170, 0.0075, 0.0190, 0.0214, 0.0228, 0.0091, 0.0157, 0.0204, 0.0213, 0.0082, 0.0138, 0.0140, 0.0300, 0.0192, 0.0065, 0.0122, 0.0164],
  },
  human: {
    mean: [0.0719, 0.0574, 0.0361, 0.0490, 0.0222, 0.0482, 0.0711, 0.0663, 0.0260, 0.0424, 0.0984, 0.0553, 0.0209, 0.0363, 0.0652, 0.0831, 0.0519, 0.0123, 0.0267, 0.0592],
    std: [0.0239, 0.0188, 0.0137, 0.0154, 0.0152, 0.0169, 0.0250, 0.0220, 0.0111, 0.0180, 0.0255, 0.0231, 0.0089, 0.0159, 0.0297, 0.0238, 0.0135, 0.0080, 0.0121, 0.0169],
  },
  mouse: {
    mean: [0.0689, 0.0552, 0.0358, 0.0499, 0.0215, 0.0473, 0.0731, 0.0645, 0.0253, 0.0433, 0.0971, 0.0579, 0.0208, 0.0360, 0.0634, 0.0850, 0.0546, 0.0116, 0.0265, 0.0623],
    std: [0.0237, 0.0188, 0.0129, 0.0150, 0.0137, 0.0162, 0.0259, 0.0234, 0.0110, 0.0167, 0.0253, 0.0245, 0.0087, 0.0151, 0.0275, 0.0236, 0.0136, 0.0075, 0.0113, 0.0164],
  },
  arabidopsis: {
    mean: [0.0661, 0.0532, 0.0437, 0.0565, 0.0164, 0.0375, 0.0707, 0.0668, 0.0218, 0.0519, 0.0963, 0.0632, 0.0231, 0.0396, 0.0474, 0.0900, 0.0509, 0.0116, 0.0268, 0.0664],
    std: [0.0198, 0.0167, 0.0139, 0.0149, 0.0092, 0.0153, 0.0224, 0.0210, 0.0099, 0.0141, 0.0223, 0.0191, 0.0088, 0.0131, 0.0171, 0.0233, 0.0130, 0.0074, 0.0114, 0.0156],
  },
  fly: {
    mean: [0.0758, 0.0525, 0.0493, 0.0535, 0.0182, 0.0566, 0.0638, 0.0637, 0.0274, 0.0477, 0.0859, 0.0552, 0.0235, 0.0328, 0.0558, 0.0860, 0.0575, 0.0091, 0.0284, 0.0571],
    std: [0.0253, 0.0170, 0.0154, 0.0155, 0.0128, 0.0271, 0.0226, 0.0274, 0.0139, 0.0162, 0.0223, 0.0222, 0.0097, 0.0140, 0.0223, 0.0277, 0.0151, 0.0067, 0.0119, 0.0162],
  },
  pdb: {
    mean: [0.0756, 0.0519, 0.0449, 0.0598, 0.0147, 0.0387, 0.0699, 0.0703, 0.0238, 0.0566, 0.0907, 0.0639, 0.0157, 0.0402, 0.0448, 0.0620, 0.0548, 0.0140, 0.0360, 0.0719],
    std: [0.0362, 0.0274, 0.0241, 0.0207, 0.0196, 0.0209, 0.0283, 0.0360, 0.0153, 0.0253, 0.0316, 0.0316, 0.0142, 0.0181, 0.0211, 0.0252, 0.0248, 0.0116, 0.0186, 0.0251],
  },
});
