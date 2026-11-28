// Layer 3 Plugin helpers — artios-backbone bundled Chemical Component Dictionary conformers.
//
// Heavy-atom ideal conformers (`pdbx_model_Cartn_{x,y,z}_ideal`) of the 20 standard amino
// acids plus UNK, extracted verbatim from the wwPDB Chemical Component Dictionary
// (https://files.rcsb.org/ligands/download/<CCD>.cif, retrieved 2026-11-27; wwPDB data is
// distributed under CC0 1.0). Hydrogens and leaving atoms (e.g. ALA `OXT`) are dropped, which
// reproduces the vendored `parseChemComp(cif)` defaults in
// `static/vendor/artios-backbone/shared/index.js` exactly — verified per-atom in the unit tests.
//
// Bundling these keeps `ref_pos` faithful for protein residues without a network round trip;
// ligands and non-standard components are fetched from the RCSB CCD at run time
// (`fetchCcdComponent` in `jobs.ts`) and parsed with the vendored CIF parser.

/** `[atomId, element, formalCharge, idealX, idealY, idealZ]`. */
export type CcdAtomRow = readonly [string, string, number, number, number, number];

export interface CcdComponentRows {
  /** `_chem_comp.type`, used by the vendored featurizer to derive `molType`. */
  readonly type: string;
  readonly atoms: readonly CcdAtomRow[];
}

/** One atom of a parsed CCD component, matching the vendored `parseChemComp` atom shape. */
export interface CcdAtom {
  atomId: string;
  altAtomId: string | null;
  element: string;
  charge: number;
  leaving: boolean;
  aromatic: boolean;
  coord: [number, number, number];
  coordSource: 'ideal' | 'model' | 'zeros';
  hasCoord: boolean;
}

/** A parsed CCD component, matching the vendored `parseChemComp` return shape. */
export interface CcdComponent {
  id: string;
  type: string;
  molType: 'protein' | 'dna' | 'rna' | 'ligand';
  atoms: CcdAtom[];
  atomIndex: Map<string, number>;
  bonds: Array<{ atom1: string; atom2: string; order: string; aromatic: boolean }>;
  coordSource: 'ideal' | 'model';
}

export const STANDARD_CCD_ROWS: Readonly<Record<string, CcdComponentRows>> = {
  ALA: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, -0.966, 0.493, 1.5], ['CA', 'C', 0, 0.257, 0.418, 0.692], ['C', 'C', 0, -0.094, 0.017, -0.716], ['O', 'O', 0, -1.056, -0.682, -0.923], ['CB', 'C', 0, 1.204, -0.62, 1.296]],
  },
  ARG: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, -0.469, 1.11, -0.993], ['CA', 'C', 0, 0.004, 2.294, -1.708], ['C', 'C', 0, -0.907, 2.521, -2.901], ['O', 'O', 0, -1.827, 1.789, -3.242], ['CB', 'C', 0, 1.475, 2.15, -2.127], ['CG', 'C', 0, 1.745, 1.017, -3.13], ['CD', 'C', 0, 3.21, 0.954, -3.557], ['NE', 'N', 0, 4.071, 0.726, -2.421], ['CZ', 'C', 0, 5.469, 0.624, -2.528], ['NH1', 'N', 0, 6.259, 0.404, -1.405], ['NH2', 'N', 1, 6.078, 0.744, -3.773]],
  },
  ASN: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, -0.293, 1.686, 0.094], ['CA', 'C', 0, -0.448, 0.292, -0.34], ['C', 'C', 0, -1.846, -0.179, -0.031], ['O', 'O', 0, -2.51, 0.402, 0.794], ['CB', 'C', 0, 0.562, -0.588, 0.401], ['CG', 'C', 0, 1.96, -0.197, -0.002], ['OD1', 'O', 0, 2.132, 0.697, -0.804], ['ND2', 'N', 0, 3.019, -0.841, 0.527]],
  },
  ASP: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, -0.317, 1.688, 0.066], ['CA', 'C', 0, -0.47, 0.286, -0.344], ['C', 'C', 0, -1.868, -0.18, -0.029], ['O', 'O', 0, -2.534, 0.415, 0.786], ['CB', 'C', 0, 0.539, -0.58, 0.413], ['CG', 'C', 0, 1.938, -0.195, 0.004], ['OD1', 'O', 0, 2.109, 0.681, -0.81], ['OD2', 'O', 0, 2.992, -0.826, 0.543]],
  },
  CYS: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.585, 0.483, -0.081], ['CA', 'C', 0, 0.141, 0.45, 0.186], ['C', 'C', 0, -0.095, 0.006, 1.606], ['O', 'O', 0, 0.685, -0.742, 2.143], ['CB', 'C', 0, -0.533, -0.53, -0.774], ['SG', 'S', 0, -0.247, 0.004, -2.484]],
  },
  GLN: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.858, -0.148, 1.125], ['CA', 'C', 0, 0.517, 0.451, 1.112], ['C', 'C', 0, -0.236, 0.022, 2.344], ['O', 'O', 0, -0.005, -1.049, 2.851], ['CB', 'C', 0, -0.236, -0.013, -0.135], ['CG', 'C', 0, 0.529, 0.421, -1.385], ['CD', 'C', 0, -0.213, -0.036, -2.614], ['OE1', 'O', 0, -1.252, -0.65, -2.5], ['NE2', 'N', 0, 0.277, 0.236, -3.839]],
  },
  GLU: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.199, 1.867, -0.117], ['CA', 'C', 0, 1.138, 0.515, 0.453], ['C', 'C', 0, 2.364, -0.26, 0.041], ['O', 'O', 0, 3.01, 0.096, -0.916], ['CB', 'C', 0, -0.113, -0.2, -0.062], ['CG', 'C', 0, -1.36, 0.517, 0.461], ['CD', 'C', 0, -2.593, -0.187, -0.046], ['OE1', 'O', 0, -2.485, -1.161, -0.753], ['OE2', 'O', 0, -3.811, 0.269, 0.287]],
  },
  GLY: {
    type: 'PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.931, 0.09, -0.034], ['CA', 'C', 0, 0.761, -0.799, -0.008], ['C', 'C', 0, -0.498, 0.029, -0.005], ['O', 'O', 0, -0.429, 1.235, -0.023]],
  },
  HIS: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, -0.04, -1.21, 0.053], ['CA', 'C', 0, 1.172, -1.709, 0.652], ['C', 'C', 0, 1.083, -3.207, 0.905], ['O', 'O', 0, 0.04, -3.77, 1.222], ['CB', 'C', 0, 1.484, -0.975, 1.962], ['CG', 'C', 0, 2.94, -1.06, 2.353], ['ND1', 'N', 1, 3.38, -2.075, 3.129], ['CD2', 'C', 0, 3.96, -0.251, 2.046], ['CE1', 'C', 0, 4.693, -1.908, 3.317], ['NE2', 'N', 0, 5.058, -0.801, 2.662]],
  },
  ILE: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, -1.944, 0.335, -0.343], ['CA', 'C', 0, -0.487, 0.519, -0.369], ['C', 'C', 0, 0.066, -0.032, -1.657], ['O', 'O', 0, -0.484, -0.958, -2.203], ['CB', 'C', 0, 0.14, -0.219, 0.814], ['CG1', 'C', 0, -0.421, 0.341, 2.122], ['CG2', 'C', 0, 1.658, -0.027, 0.788], ['CD1', 'C', 0, 0.206, -0.397, 3.305]],
  },
  LEU: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, -1.661, 0.627, -0.406], ['CA', 'C', 0, -0.205, 0.441, -0.467], ['C', 'C', 0, 0.18, -0.055, -1.836], ['O', 'O', 0, -0.591, -0.731, -2.474], ['CB', 'C', 0, 0.221, -0.583, 0.585], ['CG', 'C', 0, -0.17, -0.079, 1.976], ['CD1', 'C', 0, 0.256, -1.104, 3.029], ['CD2', 'C', 0, 0.526, 1.254, 2.25]],
  },
  LYS: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.422, 1.796, 0.198], ['CA', 'C', 0, 1.394, 0.355, 0.484], ['C', 'C', 0, 2.657, -0.284, -0.032], ['O', 'O', 0, 3.316, 0.275, -0.876], ['CB', 'C', 0, 0.184, -0.278, -0.206], ['CG', 'C', 0, -1.102, 0.282, 0.407], ['CD', 'C', 0, -2.313, -0.351, -0.283], ['CE', 'C', 0, -3.598, 0.208, 0.329], ['NZ', 'N', 1, -4.761, -0.4, -0.332]],
  },
  MET: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, -1.816, 0.142, -1.166], ['CA', 'C', 0, -0.392, 0.499, -1.214], ['C', 'C', 0, 0.206, 0.002, -2.504], ['O', 'O', 0, -0.236, -0.989, -3.033], ['CB', 'C', 0, 0.334, -0.145, -0.032], ['CG', 'C', 0, -0.273, 0.359, 1.277], ['SD', 'S', 0, 0.589, -0.405, 2.678], ['CE', 'C', 0, -0.314, 0.353, 4.056]],
  },
  PHE: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.317, 0.962, 1.014], ['CA', 'C', 0, -0.02, 0.426, 1.3], ['C', 'C', 0, -0.109, 0.047, 2.756], ['O', 'O', 0, 0.879, -0.317, 3.346], ['CB', 'C', 0, -0.27, -0.809, 0.434], ['CG', 'C', 0, -0.181, -0.43, -1.02], ['CD1', 'C', 0, 1.031, -0.498, -1.68], ['CD2', 'C', 0, -1.314, -0.018, -1.698], ['CE1', 'C', 0, 1.112, -0.15, -3.015], ['CE2', 'C', 0, -1.231, 0.333, -3.032], ['CZ', 'C', 0, -0.018, 0.265, -3.691]],
  },
  PRO: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, -0.816, 1.108, 0.254], ['CA', 'C', 0, 0.001, -0.107, 0.509], ['C', 'C', 0, 1.408, 0.091, 0.005], ['O', 'O', 0, 1.65, 0.98, -0.777], ['CB', 'C', 0, -0.703, -1.227, -0.286], ['CG', 'C', 0, -2.163, -0.753, -0.439], ['CD', 'C', 0, -2.218, 0.614, 0.276]],
  },
  SER: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.525, 0.493, -0.608], ['CA', 'C', 0, 0.1, 0.469, -0.252], ['C', 'C', 0, -0.053, 0.004, 1.173], ['O', 'O', 0, 0.751, -0.76, 1.649], ['CB', 'C', 0, -0.642, -0.489, -1.184], ['OG', 'O', 0, -0.496, -0.049, -2.535]],
  },
  THR: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.543, -0.702, 0.43], ['CA', 'C', 0, 0.122, -0.706, 0.056], ['C', 'C', 0, -0.038, -0.09, -1.309], ['O', 'O', 0, 0.732, 0.761, -1.683], ['CB', 'C', 0, -0.675, 0.104, 1.079], ['OG1', 'O', 0, -0.193, 1.448, 1.103], ['CG2', 'C', 0, -0.511, -0.521, 2.466]],
  },
  TRP: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.278, 1.121, 2.059], ['CA', 'C', 0, -0.008, 0.417, 1.97], ['C', 'C', 0, -0.49, 0.076, 3.357], ['O', 'O', 0, 0.308, -0.13, 4.24], ['CB', 'C', 0, 0.168, -0.868, 1.161], ['CG', 'C', 0, 0.65, -0.526, -0.225], ['CD1', 'C', 0, 1.928, -0.418, -0.622], ['CD2', 'C', 0, -0.186, -0.256, -1.396], ['NE1', 'N', 0, 1.978, -0.095, -1.951], ['CE2', 'C', 0, 0.701, 0.014, -2.454], ['CE3', 'C', 0, -1.564, -0.21, -1.615], ['CZ2', 'C', 0, 0.19, 0.314, -3.712], ['CZ3', 'C', 0, -2.044, 0.086, -2.859], ['CH2', 'C', 0, -1.173, 0.348, -3.907]],
  },
  TYR: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.32, 0.952, 1.428], ['CA', 'C', 0, -0.018, 0.429, 1.734], ['C', 'C', 0, -0.103, 0.094, 3.201], ['O', 'O', 0, 0.886, -0.254, 3.799], ['CB', 'C', 0, -0.274, -0.831, 0.907], ['CG', 'C', 0, -0.189, -0.496, -0.559], ['CD1', 'C', 0, 1.022, -0.589, -1.219], ['CD2', 'C', 0, -1.324, -0.102, -1.244], ['CE1', 'C', 0, 1.103, -0.282, -2.563], ['CE2', 'C', 0, -1.247, 0.21, -2.587], ['CZ', 'C', 0, -0.032, 0.118, -3.252], ['OH', 'O', 0, 0.044, 0.42, -4.574]],
  },
  VAL: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.564, -0.642, 0.454], ['CA', 'C', 0, 0.145, -0.698, 0.079], ['C', 'C', 0, -0.037, -0.093, -1.288], ['O', 'O', 0, 0.703, 0.784, -1.664], ['CB', 'C', 0, -0.682, 0.086, 1.098], ['CG1', 'C', 0, -0.497, -0.528, 2.487], ['CG2', 'C', 0, -0.218, 1.543, 1.119]],
  },
  UNK: {
    type: 'L-PEPTIDE LINKING',
    atoms: [['N', 'N', 0, 1.827, -0.498, -1.278], ['CA', 'C', 0, 2.145, -1.162, -2.517], ['C', 'C', 0, 1.576, -0.429, -3.719], ['O', 'O', 0, 0.85, 0.554, -3.635], ['CB', 'C', 0, 1.719, -2.634, -2.491], ['CG', 'C', 0, 0.211, -2.855, -2.469]],
  },
};

/** `parseChemComp`'s molType derivation (`_chem_comp.type` -> polymer class). */
export function molTypeFromCompType(type: string): CcdComponent['molType'] {
  const t = type.toUpperCase();
  if (t.includes('PEPTIDE') && t !== 'PEPTIDE-LIKE') return 'protein';
  if (t.includes('DNA')) return 'dna';
  if (t.includes('RNA')) return 'rna';
  return 'ligand';
}

/** Expand a compact row table into the vendored `parseChemComp` component shape. */
export function componentFromRows(id: string, rows: CcdComponentRows): CcdComponent {
  const atoms: CcdAtom[] = rows.atoms.map(([atomId, element, charge, x, y, z]) => ({
    atomId,
    altAtomId: null,
    element,
    charge,
    leaving: false,
    aromatic: false,
    coord: [x, y, z],
    coordSource: 'ideal',
    hasCoord: true,
  }));
  const atomIndex = new Map<string, number>();
  atoms.forEach((atom, i) => {
    if (!atomIndex.has(atom.atomId)) atomIndex.set(atom.atomId, i);
  });
  return {
    id,
    type: rows.type,
    molType: molTypeFromCompType(rows.type),
    atoms,
    atomIndex,
    bonds: [],
    coordSource: 'ideal',
  };
}

/** Fresh `Map` of the bundled standard-residue components (CCD code -> component). */
export function buildStandardComponents(): Map<string, CcdComponent> {
  const out = new Map<string, CcdComponent>();
  for (const [id, rows] of Object.entries(STANDARD_CCD_ROWS)) {
    out.set(id, componentFromRows(id, rows));
  }
  return out;
}

/** CCD code used for generated (sequence-unknown) scaffold residues: backbone N/CA/C/O only. */
export const SCAFFOLD_RESIDUE = 'GLY';
