// Pure-TypeScript structural template parsing, sequence-to-template alignment,
// model capability taxonomy, and AF2 / AF3 featurization slot builders for LocalFold.
//
// Self-contained and Node-importable: ZERO runtime imports from `static/vendor/localfold/`,
// `node:*`, Svelte, DOM, or WebGPU globals.

import { NUCLEIC_ACID_RESNAMES } from '../protein-ops/hetatm-policy';
import type { StructureArtifact } from './types';

// ---------------------------------------------------------------------------
// Constants matching vendored LocalFold (template-input.js, template-features.js,
// reference-conformers.js, modified-parents.js, dialect.js)
// ---------------------------------------------------------------------------

/** AF2 & AF3 gap residue index (restypes_with_x_and_gap[21] = '-'). */
export const GAP_AATYPE = 21;

/** Unknown amino acid index ('X' / 'UNK'). */
export const UNK_AATYPE = 20;

/** Canonical AF2 / AF3 restype order (indices 0..19; 20 = X/UNK, 21 = GAP). */
export const AF2_RESTYPE_ORDER = 'ARNDCQEGHILKMFPSTWYV';

const AATYPE_INDEX_BY_CHAR: Readonly<Record<string, number>> = Object.fromEntries(
  [...AF2_RESTYPE_ORDER].map((ch, i) => [ch, i]),
);

/**
 * PDB atom name of each slot in AF2's 37-atom representation, matching
 * `ATOM37` in vendored `shared/af3/featurise/template-features.js`.
 */
export const AF2_ATOM37_NAMES: readonly string[] = [
  'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1',
  'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2',
  'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT',
];

const AF2_ATOM37_SLOT_BY_NAME: ReadonlyMap<string, number> = new Map(
  AF2_ATOM37_NAMES.map((name, idx) => [name, idx]),
);

/**
 * Dense-24 internal conformer atom order per 1-letter residue code, matching
 * `REFERENCE_CONFORMERS[code].internal` in vendored `reference-conformers.js`.
 */
export const AF3_DENSE_INTERNAL_ATOMS: Readonly<Record<string, readonly string[]>> = {
  A: ['N', 'CA', 'C', 'O', 'CB'],
  C: ['N', 'CA', 'C', 'O', 'CB', 'SG'],
  D: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
  E: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
  F: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
  G: ['N', 'CA', 'C', 'O'],
  H: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
  I: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
  K: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
  L: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
  M: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
  N: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
  P: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
  Q: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
  R: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
  S: ['N', 'CA', 'C', 'O', 'CB', 'OG'],
  T: ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
  V: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
  W: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
  Y: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
  X: ['N', 'CA', 'C', 'O'],
};

const NUM_AF3_DENSE = 24;

const STANDARD_ONE_LETTER: Readonly<Record<string, string>> = {
  ALA: 'A',
  ARG: 'R',
  ASN: 'N',
  ASP: 'D',
  CYS: 'C',
  GLN: 'Q',
  GLU: 'E',
  GLY: 'G',
  HIS: 'H',
  ILE: 'I',
  LEU: 'L',
  LYS: 'K',
  MET: 'M',
  PHE: 'F',
  PRO: 'P',
  SER: 'S',
  THR: 'T',
  TRP: 'W',
  TYR: 'Y',
  VAL: 'V',
};

const STANDARD_AMINO_3: ReadonlySet<string> = new Set(Object.keys(STANDARD_ONE_LETTER));

const HETERO_ONE_LETTER: Readonly<Record<string, string>> = {
  MSE: 'M',
};

const PEPTIDE_CAP_RESNAMES: ReadonlySet<string> = new Set(['ACE', 'NME', 'NH2', 'FOR']);
const SOLVENT_WATER_RESNAMES: ReadonlySet<string> = new Set([
  'HOH', 'WAT', 'DOD', 'H2O', 'TIP', 'TIP3',
]);

/**
 * CCD modified peptide components mapped to their standard parent 3-letter amino acid,
 * matching vendored `shared/af3/featurise/modified-parents.js` (generated from wwPDB CCD).
 */
export const MODIFIED_PARENTS: Readonly<Record<string, string>> = Object.freeze({
  '143': 'CYS', '200': 'PHE', '432': 'SER', '823': 'ASN', '999': 'ASP',
  '02K': 'ALA', '02L': 'ASN', '02O': 'ALA', '02Y': 'ALA', '037': 'PRO',
  '03Y': 'CYS', '04U': 'PRO', '04V': 'PRO', '05N': 'PRO', '07O': 'CYS',
  '0A0': 'ASP', '0A1': 'TYR', '0A2': 'LYS', '0A8': 'CYS', '0A9': 'PHE',
  '0AA': 'VAL', '0AB': 'VAL', '0AC': 'GLY', '0AF': 'TRP', '0AG': 'LEU',
  '0AH': 'SER', '0AK': 'ASP', '0AR': 'ARG', '0BN': 'PHE', '0CS': 'ALA',
  '0E5': 'THR', '0EA': 'TYR', '0FL': 'ALA', '0LF': 'PRO', '0NC': 'ALA',
  '0PR': 'TYR', '0QL': 'CYS', '0TD': 'ASP', '0UO': 'TRP', '0WZ': 'TYR',
  '0X9': 'ARG', '0Y8': 'PRO', '11Q': 'PRO', '11W': 'GLU', '12L': 'PRO',
  '12X': 'PRO', '12Y': 'PRO', '1AC': 'ALA', '1L1': 'ALA', '1OP': 'TYR',
  '1PA': 'PHE', '1PI': 'ALA', '1TQ': 'TRP', '1TY': 'TYR', '1X6': 'SER',
  '23F': 'PHE', '23P': 'ALA', '26B': 'THR', '28X': 'THR', '2AG': 'ALA',
  '2CO': 'CYS', '2FM': 'MET', '2GX': 'PHE', '2HF': 'HIS', '2JG': 'SER',
  '2KK': 'LYS', '2KP': 'LYS', '2LT': 'TYR', '2LU': 'LEU', '2ML': 'LEU',
  '2MR': 'ARG', '2MT': 'PRO', '2OR': 'ARG', '2P0': 'PRO', '2QZ': 'THR',
  '2R3': 'TYR', '2RA': 'ALA', '2RX': 'SER', '2SO': 'HIS', '2TY': 'TYR',
  '2VA': 'VAL', '2XA': 'CYS', '2ZC': 'SER', '30V': 'CYS', '31Q': 'CYS',
  '32S': 'TRP', '32T': 'TRP', '33S': 'PHE', '33W': 'ALA', '34E': 'VAL',
  '3AH': 'HIS', '3BY': 'PRO', '3CF': 'PHE', '3CT': 'TYR', '3GA': 'ALA',
  '3GL': 'GLU', '3MD': 'ASP', '3MY': 'TYR', '3NF': 'TYR', '3O3': 'GLU',
  '3PX': 'PRO', '3QN': 'LYS', '3TY': 'TYR', '3WS': 'ALA', '3WX': 'PRO',
  '3X9': 'CYS', '3XH': 'GLY', '3YM': 'TYR', '3ZH': 'HIS', '41H': 'PHE',
  '41Q': 'ASN', '4AF': 'PHE', '4AK': 'LYS', '4AR': 'ARG', '4AW': 'TRP',
  '4BF': 'TYR', '4CF': 'PHE', '4CY': 'MET', '4D4': 'ARG', '4DP': 'TRP',
  '4FB': 'PRO', '4FW': 'TRP', '4GJ': 'CYS', '4HH': 'SER', '4HJ': 'SER',
  '4HL': 'TYR', '4HT': 'TRP', '4II': 'PHE', '4IN': 'TRP', '4J4': 'CYS',
  '4J5': 'ARG', '4KY': 'PRO', '4L0': 'PRO', '4LZ': 'TYR', '4MM': 'MET',
  '4N7': 'PRO', '4N8': 'PRO', '4N9': 'PRO', '4OG': 'TRP', '4OU': 'PHE',
  '4OV': 'SER', '4OZ': 'SER', '4PH': 'PHE', '4PQ': 'TRP', '4SJ': 'PHE',
  '4U7': 'ALA', '4WQ': 'ALA', '51T': 'TYR', '54C': 'TRP', '55I': 'PHE',
  '56A': 'HIS', '5AB': 'ALA', '5CR': 'PHE', '5CS': 'CYS', '5CT': 'LYS',
  '5CW': 'TRP', '5FQ': 'ALA', '5GG': 'LYS', '5GM': 'ILE', '5HP': 'GLU',
  '5MW': 'LYS', '5OH': 'ALA', '5OW': 'LYS', '5PG': 'GLY', '5R5': 'SER',
  '5T3': 'LYS', '5VV': 'ASN', '5XU': 'ALA', '60F': 'CYS', '66D': 'ILE',
  '6BR': 'THR', '6CL': 'LYS', '6CV': 'ALA', '6CW': 'TRP', '6G4': 'LYS',
  '6GL': 'ALA', '6HN': 'LYS', '6M6': 'CYS', '6V1': 'CYS', '6WK': 'CYS',
  '6Y9': 'PRO', '73C': 'SER', '73N': 'ARG', '73O': 'TYR', '73P': 'LYS',
  '74P': 'LYS', '7ID': 'ASP', '7JA': 'ILE', '7N8': 'PHE', '7O5': 'ALA',
  '7OZ': 'ALA', '7QK': 'LYS', '7T2': 'PHE', '7TK': 'ASP', '7VN': 'ALA',
  '7VU': 'PHE', '7W2': 'PHE', '7XC': 'PHE', '7YO': 'PRO', '85F': 'CYS',
  '85L': 'CYS', '86N': 'GLU', '8JB': 'CYS', '8LJ': 'PRO', '8RE': 'LYS',
  '8SP': 'SER', '8WY': 'LEU', '96Z': 'ALA', '9DN': 'ASN', '9DS': 'GLY',
  '9DT': 'PHE', '9E7': 'LYS', '9KP': 'LYS', '9NE': 'GLU', '9NF': 'PHE',
  '9NR': 'ARG', '9NV': 'VAL', '9TR': 'LYS', '9TU': 'LYS', '9TX': 'LYS',
  '9U0': 'LYS', '9WV': 'ALA',
  A1A3M: 'PRO', A1A5S: 'TRP', A1A5T: 'LEU', A1A7R: 'HIS', A1A7V: 'CYS',
  A1A8T: 'VAL', A1ADO: 'LEU', A1AL8: 'TRP', A1ALE: 'ALA', A1AMB: 'ALA',
  A1AMC: 'ALA', A1AS0: 'HIS', A1ASZ: 'HIS', A1AT6: 'ALA', A1AT7: 'ALA',
  A1AVU: 'VAL', A1AWS: 'TRP', A1AZ2: 'HIS', A1AZQ: 'GLY', A1B9V: 'TRP',
  A1BC0: 'CYS', A1BC5: 'SER', A1BDE: 'LYS', A1BEB: 'LYS', A1BX4: 'LYS',
  A1BZO: 'SER', A1C0O: 'PHE', A1C3V: 'PHE', A1CAB: 'PRO', A1CCQ: 'TYR',
  A1CCR: 'PHE', A1CCS: 'ARG', A1CH8: 'ARG', A1CHT: 'THR', A1CLH: 'TRP',
  A1CM8: 'ALA', A1CM9: 'ALA', A1CRT: 'CYS', A1CTT: 'ALA', A1D5B: 'LYS',
  A1D5E: 'PRO', A1D5P: 'TYR', A1D64: 'CYS', A1D7R: 'TRP', A1D9O: 'PRO',
  A1ECI: 'CYS', A1ECK: 'PHE', A1ECL: 'PHE', A1ECM: 'VAL', A1EFT: 'ARG',
  A1EG1: 'LYS', A1EIC: 'SER', A1EIU: 'LYS', A1EJ1: 'LYS', A1EJ4: 'LYS',
  A1EJR: 'LYS', A1EJS: 'LYS', A1EJT: 'LYS', A1EJZ: 'LYS', A1EMP: 'LYS',
  A1ER5: 'ASP', A1EUR: 'ALA', A1EYP: 'LYS', A1EZB: 'TRP', A1H2H: 'PHE',
  A1H2I: 'TRP', A1H45: 'TRP', A1H5N: 'LYS', A1H5W: 'ALA', A1H96: 'LYS',
  A1I9G: 'HIS', A1ICB: 'PRO', A1ICC: 'PRO', A1IHG: 'PHE', A1IID: 'ALA',
  A1IKP: 'SER', A1IPL: 'PHE', A1IQX: 'HIS', A1JCN: 'HIS', A1JNL: 'ALA',
  A1JS9: 'PRO', A1L2U: 'PRO', A1L4B: 'GLN', A1L4D: 'PHE', A1L7E: 'TRP',
  A1LTQ: 'ARG', A1LUD: 'MET', A1LUT: 'ASN', A1LUV: 'LYS', A1LWV: 'LYS',
  A1MCG: 'PHE', A1MF6: 'LEU', A30: 'TYR', A3U: 'PHE', A5N: 'ASN',
  A66: 'LYS', A8E: 'VAL', A9D: 'SER', AA3: 'ALA', AA4: 'ALA',
  AAR: 'ARG', ABA: 'ALA', ACL: 'ARG', AEA: 'CYS', AEI: 'THR',
  AFA: 'ASN', AGM: 'ARG', AGQ: 'TYR', AGT: 'CYS', AHB: 'ASN',
  AHO: 'ALA', AHP: 'ALA', AIB: 'ALA', AKL: 'ASP', AKZ: 'ASP',
  ALC: 'ALA', ALM: 'ALA', ALN: 'ALA', ALO: 'THR', ALS: 'ALA',
  ALT: 'ALA', ALV: 'ALA', ALY: 'LYS', AME: 'MET', AN6: 'LEU',
  AN8: 'ALA', API: 'LYS', APK: 'LYS', AR2: 'ARG', AR4: 'GLU',
  AR7: 'ARG', ARM: 'ARG', ARO: 'ARG', AS7: 'ASN', ASA: 'ASP',
  ASB: 'ASP', ASI: 'ASP', ASK: 'ASP', ASL: 'ASP', ASQ: 'ASP',
  AYA: 'ALA', AZH: 'ALA', AZK: 'LYS', AZS: 'SER', AZY: 'TYR',
  B1F: 'PHE', B27: 'THR', B2A: 'ALA', B2F: 'PHE', B2I: 'ILE',
  B2V: 'VAL', B3A: 'ALA', B3D: 'ASP', B3E: 'GLU', B3K: 'LYS',
  B3U: 'HIS', B3X: 'ASN', B3Y: 'TYR', BB6: 'CYS', BB7: 'CYS',
  BB8: 'PHE', BB9: 'CYS', BBC: 'CYS', BCS: 'CYS', BCX: 'CYS',
  BFD: 'ASP', BG1: 'SER', BG4: 'PRO', BG5: 'PRO', BH2: 'ASP',
  BHD: 'ASP', BIF: 'PHE', BIU: 'ILE', BL2: 'LEU', BLE: 'LEU',
  BLY: 'LYS', BMT: 'THR', BNN: 'PHE', BNO: 'LEU', BOR: 'ARG',
  BP5: 'ALA', BPE: 'CYS', BSE: 'SER', BTA: 'LEU', BTC: 'CYS',
  BTK: 'LYS', BTR: 'TRP', BUC: 'CYS', BUG: 'VAL', BWB: 'SER',
  BWV: 'ARG', BXT: 'SER', BYR: 'TYR', C1T: 'CYS', C1X: 'LYS',
  C22: 'ALA', C3Y: 'CYS', C4R: 'CYS', C5C: 'CYS', C6C: 'CYS',
  CAF: 'CYS', CAS: 'CYS', CAY: 'CYS', CCS: 'CYS', CE7: 'ASN',
  CEA: 'CYS', CG6: 'CYS', CGA: 'GLU', CGL: 'CYS', CGU: 'GLU',
  CGV: 'CYS', CHP: 'GLY', CIR: 'ARG', CLE: 'LEU', CLG: 'LYS',
  CLH: 'LYS', CME: 'CYS', CMH: 'CYS', CML: 'CYS', CMT: 'CYS',
  CR5: 'GLY', CS0: 'CYS', CS1: 'CYS', CS3: 'CYS', CS4: 'CYS',
  CSA: 'CYS', CSB: 'CYS', CSD: 'CYS', CSE: 'CYS', CSJ: 'CYS',
  CSO: 'CYS', CSP: 'CYS', CSR: 'CYS', CSS: 'CYS', CSU: 'CYS',
  CSW: 'CYS', CSX: 'CYS', CSZ: 'CYS', CTE: 'TRP', CTH: 'THR',
  CWD: 'ALA', CWR: 'SER', CXM: 'MET', CY0: 'CYS', CY1: 'CYS',
  CY3: 'CYS', CY4: 'CYS', CYA: 'CYS', CYD: 'CYS', CYF: 'CYS',
  CYG: 'CYS', CYJ: 'LYS', CYM: 'CYS', CYQ: 'CYS', CYR: 'CYS',
  CYW: 'CYS', CZ2: 'CYS', CZS: 'ALA', CZZ: 'CYS', D2T: 'ASP',
  D8R: 'LYS', DA2: 'ARG', DAB: 'ALA', DAH: 'TYR', DBS: 'SER',
  DBU: 'THR', DBY: 'TYR', DBZ: 'ALA', DC2: 'CYS', DDE: 'HIS',
  DDZ: 'ALA', DHA: 'SER', DHN: 'VAL', DI7: 'TYR', DIR: 'ARG',
  DISEP: 'SER', DJD: 'PHE', DLS: 'LYS', DM0: 'LYS', DMH: 'ASN',
  DMK: 'ASP', DNL: 'LYS', DNP: 'ALA', DNS: 'LYS', DNW: 'ALA',
  DOH: 'ASP', DON: 'LEU', DPL: 'PRO', DPP: 'ALA', DPQ: 'TYR',
  DV9: 'GLU', DYA: 'ASP', DYJ: 'PRO', DYS: 'CYS', E0Y: 'PRO',
  E9C: 'TYR', E9M: 'TRP', E9V: 'HIS', ECC: 'GLN', ECX: 'CYS',
  EFC: 'CYS', EHP: 'PHE', EI4: 'ARG', EJA: 'CYS', EJJ: 'GLY',
  ELY: 'LYS', EME: 'GLU', ESB: 'TYR', ESC: 'MET', EU0: 'VAL',
  EUP: 'THR', EW6: 'SER', EXA: 'LYS', EXL: 'TRP', EXY: 'LEU',
  F2F: 'PHE', F2Y: 'TYR', F7Q: 'TYR', F7W: 'TRP', FAK: 'LYS',
  FB5: 'ALA', FB6: 'ALA', FC0: 'PHE', FCL: 'PHE', FDL: 'LYS',
  FF9: 'LYS', FFM: 'CYS', FGL: 'GLY', FGP: 'SER', FH7: 'LYS',
  FHL: 'LYS', FHO: 'LYS', FIO: 'ARG', FL6: 'ASP', FLA: 'ALA',
  FLE: 'LEU', FLT: 'TYR', FME: 'MET', FOE: 'CYS', FP9: 'PRO',
  FPK: 'PRO', FQA: 'LYS', FT6: 'TRP', FTR: 'TRP', FTY: 'TYR',
  FVA: 'VAL', FX9: 'PHE', FY2: 'TYR', FY3: 'TYR', FZN: 'LYS',
  G01: 'GLU', G1X: 'TYR', G3M: 'ARG', G5G: 'LEU', G8M: 'GLU',
  G8X: 'PRO', GAM: 'GLU', GAU: 'GLU', GEE: 'GLY', GFT: 'SER',
  GHC: 'GLU', GHG: 'GLN', GHW: 'GLU', GL3: 'GLY', GLH: 'GLN',
  GLJ: 'GLU', GLK: 'GLU', GLQ: 'GLU', GLZ: 'GLY', GMA: 'GLU',
  GME: 'GLU', GNC: 'GLN', GND: 'LYS', GPL: 'LYS', GQI: 'CYS',
  GSC: 'GLY', GSU: 'GLU', GT9: 'CYS', GVL: 'SER', H14: 'PHE',
  H5M: 'PRO', H7V: 'ALA', HAC: 'ALA', HAR: 'ARG', HBN: 'HIS',
  HCM: 'CYS', HGY: 'GLY', HHI: 'HIS', HIA: 'HIS', HIC: 'HIS',
  HIP: 'HIS', HIQ: 'HIS', HIX: 'ALA', HL2: 'LEU', HL5: 'LEU',
  HLU: 'LEU', HLY: 'LYS', HMR: 'ARG', HNC: 'CYS', HOO: 'HIS',
  HOX: 'PHE', HP9: 'PHE', HPC: 'PHE', HPE: 'PHE', HPH: 'PHE',
  HPQ: 'PHE', HQA: 'ALA', HR7: 'ARG', HRG: 'ARG', HRP: 'TRP',
  HS8: 'HIS', HS9: 'HIS', HSE: 'SER', HSK: 'HIS', HSL: 'SER',
  HSO: 'HIS', HSV: 'HIS', HT7: 'TRP', HTI: 'CYS', HTR: 'TRP',
  HV5: 'ALA', HVA: 'VAL', HY3: 'PRO', HYP: 'PRO', HZP: 'PRO',
  I2F: 'LYS', I2M: 'ILE', I3L: 'CYS', I4G: 'GLY', I4O: 'HIS',
  I58: 'LYS', I7F: 'SER', IAE: 'PHE', IAM: 'ALA', IAR: 'ARG',
  IB9: 'TYR', IC0: 'GLY', ICY: 'CYS', IEL: 'LYS', IGL: 'GLY',
  IIL: 'ILE', ILG: 'GLU', ILM: 'ILE', ILX: 'ILE', IML: 'ILE',
  IOR: 'ARG', IPG: 'GLY', IT1: 'LYS', IYR: 'TYR', IZO: 'MET',
  J2F: 'TYR', J3D: 'CYS', J8W: 'SER', J9Y: 'ARG', JGO: 'LYS',
  JJJ: 'CYS', JJK: 'CYS', JJL: 'CYS', JKH: 'PRO', JLP: 'LYS',
  K1R: 'CYS', K5H: 'CYS', K5L: 'SER', K7K: 'SER', KBE: 'LYS',
  KCR: 'LYS', KCX: 'LYS', KEO: 'LYS', KFP: 'LYS', KGC: 'LYS',
  KHB: 'LYS', KKD: 'ASP', KLF: 'PRO', KNB: 'ALA', KOR: 'MET',
  KPF: 'LYS', KPI: 'LYS', KPY: 'LYS', KR3: 'LYS', KST: 'LYS',
  KYN: 'TRP', KYQ: 'LYS', L3O: 'LEU', L5P: 'LYS', LA2: 'LYS',
  LAA: 'ASP', LAL: 'ALA', LAY: 'LEU', LBY: 'LYS', LBZ: 'LYS',
  LCK: 'LYS', LCX: 'LYS', LDH: 'LYS', LE1: 'VAL', LED: 'LEU',
  LEF: 'LEU', LEH: 'LEU', LEM: 'LEU', LEN: 'LEU', LET: 'LYS',
  LEX: 'LEU', LGY: 'LYS', LHV: 'VAL', LLO: 'LYS', LLP: 'LYS',
  LLY: 'LYS', LLZ: 'LYS', LME: 'GLU', LMF: 'LYS', LMQ: 'GLN',
  LNE: 'LEU', LNM: 'LEU', LOH: 'LYS', LP6: 'LYS', LPD: 'PRO',
  LPG: 'GLY', LPS: 'SER', LRK: 'LYS', LSO: 'LYS', LTR: 'TRP',
  LTU: 'TRP', LVG: 'GLY', LVN: 'VAL', LWI: 'PHE', LWY: 'PRO',
  LYF: 'LYS', LYK: 'LYS', LYM: 'LYS', LYN: 'LYS', LYO: 'LYS',
  LYR: 'LYS', LYU: 'LYS', LYX: 'LYS', LYZ: 'LYS', M0H: 'CYS',
  M2L: 'LYS', M2S: 'MET', M30: 'GLY', M3L: 'LYS', M3R: 'LYS',
  MA: 'ALA', MAA: 'ALA', MAI: 'ARG', MBQ: 'TYR', MC1: 'SER',
  MCL: 'LYS', MCS: 'CYS', MD3: 'CYS', MD5: 'CYS', MD6: 'GLY',
  MDF: 'TYR', ME0: 'MET', MEA: 'PHE', MEG: 'GLU', MEN: 'ASN',
  MEQ: 'GLN', MEU: 'GLY', MGG: 'ARG', MGN: 'GLN', MGY: 'GLY',
  MH1: 'HIS', MH6: 'SER', MHL: 'LEU', MHO: 'MET', MHS: 'HIS',
  MHU: 'PHE', MIR: 'SER', MIS: 'SER', MK8: 'LEU', MKF: 'SER',
  ML3: 'LYS', MLE: 'LEU', MLL: 'LEU', MLY: 'LYS', MLZ: 'LYS',
  MME: 'MET', MMO: 'ARG', MNL: 'LEU', MNV: 'VAL', MP8: 'PRO',
  MPQ: 'GLY', MS6: 'MET', MSA: 'GLY', MSE: 'MET', MSL: 'MET',
  MSO: 'MET', MT2: 'MET', MTY: 'TYR', MV6: 'MET', MVA: 'VAL',
  MYK: 'LYS', MYN: 'ARG', N0A: 'PHE', N10: 'SER', N65: 'LYS',
  N7P: 'PRO', N80: 'PRO', N9P: 'ALA', NA8: 'ALA', NAL: 'ALA',
  NAM: 'ALA', NBQ: 'TYR', NC1: 'SER', NCB: 'ALA', NEM: 'HIS',
  NEP: 'HIS', NFA: 'PHE', NIY: 'TYR', NLB: 'LEU', NLE: 'LEU',
  NLN: 'LEU', NLO: 'LEU', NLP: 'LEU', NLQ: 'GLN', NLW: 'LEU',
  NLY: 'GLY', NMC: 'GLY', NMM: 'ARG', NNH: 'ARG', NOT: 'LYS',
  NPH: 'CYS', NPI: 'ALA', NTR: 'TYR', NTY: 'TYR', NVA: 'VAL',
  NWD: 'ALA', NYB: 'CYS', NYS: 'CYS', NZC: 'THR', NZH: 'HIS',
  O2E: 'SER', O6H: 'TRP', O7A: 'THR', O7D: 'TRP', O7G: 'VAL',
  OAR: 'ARG', OAS: 'SER', OBS: 'LYS', OCS: 'CYS', OCY: 'CYS',
  OGC: 'HIS', OHI: 'HIS', OHS: 'ASP', OLD: 'HIS', OLT: 'THR',
  OLZ: 'SER', OMH: 'SER', OMT: 'MET', OMX: 'TYR', OMY: 'TYR',
  ONH: 'ALA', ORN: 'ALA', ORQ: 'ARG', OSE: 'SER', OTH: 'THR',
  OXX: 'ASP', OYL: 'HIS', OZ3: 'GLY', OZW: 'PHE', P1L: 'CYS',
  P2Q: 'TYR', P2Y: 'PRO', P3Q: 'TYR', P9S: 'CYS', PAQ: 'TYR',
  PAS: 'ASP', PAT: 'TRP', PBB: 'CYS', PBF: 'PHE', PCA: 'GLN',
  PCC: 'PRO', PCS: 'PHE', PE1: 'LYS', PEC: 'CYS', PF5: 'PHE',
  PFF: 'PHE', PG1: 'SER', PGY: 'GLY', PH6: 'PRO', PHA: 'PHE',
  PHD: 'ASP', PHI: 'PHE', PHL: 'PHE', PHM: 'PHE', PJ3: 'HIS',
  PKR: 'PRO', PLJ: 'PRO', PM3: 'PHE', POK: 'ARG', POM: 'PRO',
  PPN: 'PHE', PR3: 'CYS', PR4: 'PRO', PR7: 'PRO', PR9: 'PRO',
  PRJ: 'PRO', PRK: 'LYS', PRS: 'PRO', PRV: 'GLY', PSA: 'PHE',
  PSH: 'HIS', PTH: 'TYR', PTM: 'TYR', PTR: 'TYR', PVH: 'HIS',
  PXU: 'PRO', PYA: 'ALA', PYH: 'LYS', PYX: 'CYS', Q2E: 'TRP',
  Q3P: 'LYS', Q75: 'HIS', Q8X: 'CYS', QCI: 'GLN', QCS: 'CYS',
  QIL: 'ILE', QM8: 'ALA', QMB: 'ALA', QMM: 'GLN', QNQ: 'CYS',
  QNT: 'CYS', QNW: 'CYS', QNY: 'THR', QO2: 'CYS', QO5: 'CYS',
  QO8: 'CYS', QPA: 'CYS', QPH: 'PHE', QQ8: 'GLN', QVA: 'CYS',
  QX7: 'ALA', R0K: 'GLU', R1A: 'CYS', R4K: 'TRP', RE0: 'TRP',
  RE3: 'TRP', RGL: 'ARG', RGP: 'GLU', RPI: 'ARG', RT0: 'PRO',
  RVJ: 'ALA', RVX: 'SER', RX9: 'ILE', RXL: 'VAL', RZ4: 'SER',
  S0R: 'LEU', S12: 'SER', S1H: 'SER', S2C: 'CYS', S2P: 'ALA',
  SAC: 'SER', SAR: 'GLY', SBL: 'SER', SCH: 'CYS', SCS: 'CYS',
  SCY: 'CYS', SD4: 'ASN', SDP: 'SER', SEB: 'SER', SEE: 'SER',
  SEG: 'ALA', SEL: 'SER', SEM: 'SER', SEN: 'SER', SEP: 'SER',
  SET: 'SER', SGB: 'SER', SHC: 'CYS', SHP: 'GLY', SHR: 'LYS',
  SIB: 'CYS', SKH: 'LYS', SLL: 'LYS', SLZ: 'LYS', SMC: 'CYS',
  SME: 'MET', SMF: 'PHE', SNC: 'CYS', SNK: 'HIS', SNM: 'SER',
  SNN: 'ASN', SOY: 'SER', SRZ: 'SER', STY: 'TYR', SUN: 'SER',
  SVA: 'SER', SVV: 'SER', SVW: 'SER', SVX: 'SER', SVY: 'SER',
  SVZ: 'SER', SWW: 'SER', SXE: 'SER', T0I: 'TYR', T11: 'PHE',
  T3R: 'LEU', T8L: 'THR', T9E: 'THR', TAV: 'ASP', TBG: 'VAL',
  TBM: 'THR', TCQ: 'TYR', TCR: 'TRP', TEF: 'PHE', TFQ: 'PHE',
  TGH: 'TRP', TH5: 'THR', TH6: 'THR', THC: 'THR', THZ: 'ARG',
  TIH: 'ALA', TIS: 'SER', TLY: 'LYS', TMB: 'THR', TMD: 'THR',
  TNB: 'CYS', TNQ: 'TRP', TNR: 'SER', TNY: 'THR', TOQ: 'TRP',
  TOX: 'TRP', TPJ: 'PRO', TPK: 'PRO', TPL: 'TRP', TPO: 'THR',
  TPQ: 'TYR', TQI: 'TRP', TQQ: 'TRP', TQZ: 'CYS', TRF: 'TRP',
  TRG: 'LYS', TRN: 'TRP', TRO: 'TRP', TRQ: 'TRP', TRW: 'TRP',
  TRX: 'TRP', TRY: 'TRP', TS9: 'ILE', TSQ: 'PHE', TSY: 'CYS',
  TTQ: 'TRP', TTS: 'TYR', TXY: 'TYR', TY1: 'TYR', TY2: 'TYR',
  TY3: 'TYR', TY5: 'TYR', TY8: 'TYR', TY9: 'TYR', TYB: 'TYR',
  TYC: 'TYR', TYE: 'TYR', TYI: 'TYR', TYJ: 'TYR', TYN: 'TYR',
  TYO: 'TYR', TYQ: 'TYR', TYS: 'TYR', TYT: 'TYR', TYW: 'TYR',
  TYY: 'TYR', U2X: 'TYR', U3X: 'PHE', UF0: 'SER', UGY: 'GLY',
  UJR: 'CYS', UM1: 'ALA', UM2: 'ALA', UMA: 'ALA', UX8: 'TRP',
  UXQ: 'PHE', V1C: 'PHE', V3C: 'PRO', V3E: 'PHE', V44: 'CYS',
  V5N: 'HIS', V61: 'PHE', V6W: 'TRP', V7T: 'LYS', VAD: 'VAL',
  VAF: 'VAL', VAH: 'VAL', VAI: 'VAL', VB1: 'LYS', VH0: 'PRO',
  VHF: 'GLU', VI3: 'CYS', VPV: 'LYS', VR0: 'ARG', VVK: 'ALA',
  WFP: 'PHE', WLU: 'LEU', WPA: 'PHE', WRP: 'TRP', WVL: 'VAL',
  WWB: 'TRP', WYK: 'ARG', X2W: 'GLU', X5P: 'ALA', X5U: 'PHE',
  X5V: 'THR', X60: 'VAL', XA6: 'PHE', XCN: 'CYS', XOK: 'LYS',
  XPR: 'PRO', XRE: 'PRO', XRW: 'LYS', XSN: 'ASN', XW1: 'ALA',
  XX1: 'LYS', XYC: 'ALA', Y1V: 'LEU', Y57: 'LYS', YC3: 'PHE',
  YCM: 'CYS', YFQ: 'ALA', YHA: 'LYS', YNM: 'TYR', YOF: 'TYR',
  YPR: 'PRO', YPZ: 'TYR', YRV: 'CYS', YTF: 'GLN', YTH: 'THR',
  YTL: 'TYR', Z01: 'ALA', Z3E: 'THR', Z70: 'HIS', ZAI: 'LYS',
  ZBZ: 'CYS', ZCL: 'PHE', ZDJ: 'TYR', ZI0: 'PRO', ZIQ: 'TRP',
  ZJU: 'ASP', ZKO: 'GLN', ZLF: 'CYS', ZPO: 'PRO', ZQN: 'VAL',
  ZSX: 'GLY', ZT6: 'TYR', ZTK: 'ALA', ZU0: 'THR', ZV4: 'PHE',
  ZYJ: 'PRO', ZYK: 'PRO', ZZD: 'CYS', ZZJ: 'ALA',
});

// ---------------------------------------------------------------------------
// Synchronous FIPS 180-4 SHA-256 helper (matches computeUploadDigest bit-for-bit,
// kept local to avoid a circular dependency registry -> localfold-template -> param-validation)
// ---------------------------------------------------------------------------

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function computeTemplateDigest(text: string): string {
  const bytes = new TextEncoder().encode(text);
  const bitLenHi = Math.floor((bytes.length * 8) / 0x100000000);
  const bitLenLo = (bytes.length * 8) >>> 0;
  const paddedLen = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLen);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLen - 8, bitLenHi, false);
  view.setUint32(paddedLen - 4, bitLenLo, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i += 1) {
      w[i] = view.getUint32(offset + i * 4, false);
    }
    for (let i = 16; i < 64; i += 1) {
      const s0 =
        ((w[i - 15] >>> 7) | (w[i - 15] << 25)) ^
        ((w[i - 15] >>> 18) | (w[i - 15] << 14)) ^
        (w[i - 15] >>> 3);
      const s1 =
        ((w[i - 2] >>> 17) | (w[i - 2] << 15)) ^
        ((w[i - 2] >>> 19) | (w[i - 2] << 13)) ^
        (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i += 1) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3, h4, h5, h6, h7]
    .map((n) => n.toString(16).padStart(8, '0'))
    .join('');
}

// ---------------------------------------------------------------------------
// 1. Model Capability & Dialect Policy API
// ---------------------------------------------------------------------------

export type LocalFoldTemplateFamily = 'af2-monomer' | 'af2-multimer' | 'af3' | 'none';
export type LocalFoldAf3ModifiedPolicy = 'unk_n' | 'unk_restype' | 'drop_hit' | null;

export interface LocalFoldTemplateDialectPolicy {
  modified: LocalFoldAf3ModifiedPolicy;
  altlocFirst: boolean;
  gapWhereNoAtoms: boolean;
}

const AF2_TEMPLATE_MONOMER_MODELS: ReadonlySet<string> = new Set(['monomer', 'monomer-2']);
const AF2_TEMPLATE_MULTIMER_MODELS: ReadonlySet<string> = new Set([
  'multimer',
  'multimer-2',
  'multimer-3',
  'multimer-4',
  'multimer-5',
]);
const AF3_TEMPLATE_MODELS: ReadonlySet<string> = new Set([
  'rosettafold3',
  'boltz2',
  'openbind0',
  'protenix2',
  'opendde',
  'mol1',
  'intellifold2',
  'af3',
]);

export function localFoldTemplateFamily(modelId: string): LocalFoldTemplateFamily {
  const id = String(modelId ?? '').trim();
  if (AF2_TEMPLATE_MONOMER_MODELS.has(id)) return 'af2-monomer';
  if (AF2_TEMPLATE_MULTIMER_MODELS.has(id)) return 'af2-multimer';
  if (AF3_TEMPLATE_MODELS.has(id)) return 'af3';
  return 'none';
}

export function isLocalFoldTemplateCapable(modelId: string): boolean {
  return localFoldTemplateFamily(modelId) !== 'none';
}

export function getLocalFoldTemplateDialectPolicy(modelId: string): LocalFoldTemplateDialectPolicy {
  const id = String(modelId ?? '').trim().toLowerCase();
  if (id === 'openbind0') {
    return { modified: 'unk_restype', altlocFirst: false, gapWhereNoAtoms: true };
  }
  if (id === 'opendde' || id === 'protenix2' || id === 'mol1') {
    return { modified: 'unk_n', altlocFirst: false, gapWhereNoAtoms: false };
  }
  if (id === 'boltz2') {
    return { modified: 'unk_n', altlocFirst: true, gapWhereNoAtoms: false };
  }
  if (id === 'rosettafold3') {
    return { modified: null, altlocFirst: true, gapWhereNoAtoms: false };
  }
  return { modified: null, altlocFirst: false, gapWhereNoAtoms: false };
}

export function suggestTemplateCapableModel(
  _currentModelId: string,
  opts?: { isMultimer?: boolean; hasHolo?: boolean },
): { modelId: string; label: string } {
  if (opts?.hasHolo) {
    return { modelId: 'boltz2', label: 'Boltz-2' };
  }
  if (opts?.isMultimer) {
    return { modelId: 'multimer', label: 'AF2-Multimer 1' };
  }
  return { modelId: 'monomer', label: 'AF2 Model 1' };
}

// ---------------------------------------------------------------------------
// 2. Typed Errors & Template Source Resolution
// ---------------------------------------------------------------------------

export type LocalFoldTemplateErrorCode =
  | 'EMPTY_TEMPLATE'
  | 'INVALID_FORMAT'
  | 'NO_POLYMER_RESIDUES'
  | 'TEMPLATE_FREE_MODEL'
  | 'ZERO_SEQUENCE_OVERLAP'
  | 'LOW_COVERAGE_OR_IDENTITY'
  | 'CHAIN_MISMATCH'
  | 'MISSING_UPLOAD';

export class LocalFoldTemplateError extends Error {
  readonly code: LocalFoldTemplateErrorCode;
  readonly details?: Record<string, unknown>;

  constructor(
    code: LocalFoldTemplateErrorCode,
    message: string,
    details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = 'LocalFoldTemplateError';
    this.code = code;
    this.details = details;
  }
}

export interface ResolvedLocalFoldTemplateSource {
  sourceKind: 'port' | 'inline';
  text: string;
  format: 'pdb' | 'mmcif';
  filename?: string;
  digest: string;
  spanChains: boolean;
  bothProvided: boolean;
  warnings: string[];
}

const TEMPLATE_SHA256_HEX_RE = /^[0-9a-f]{64}$/i;

export function detectTemplateFormat(text: string): 'pdb' | 'mmcif' {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new LocalFoldTemplateError('EMPTY_TEMPLATE', 'Template structure text is empty.');
  }
  const firstNonCommentLine = trimmed
    .split(/\r?\n/)
    .map((l) => l.trim())
    .find((l) => l.length > 0 && !l.startsWith('#'));
  const hasCifDataHeader = Boolean(firstNonCommentLine && /^data_\S*/i.test(firstNonCommentLine));
  const hasAtomSiteTagLine = /^\s*(?:loop_\s+)?_atom_site\.[A-Za-z0-9_]+(?:\s|$)/im.test(text);
  if (hasCifDataHeader || hasAtomSiteTagLine) {
    return 'mmcif';
  }
  if (/^(?:ATOM  |HETATM|MODEL |HEADER|CRYST1|SEQRES|TITLE )/m.test(text)) {
    return 'pdb';
  }
  throw new LocalFoldTemplateError(
    'INVALID_FORMAT',
    'Unrecognized structural template format (expected PDB ATOM/HETATM records or mmCIF _atom_site loop).',
  );
}

/**
 * Return true when inline template text or shared-upload expectation metadata
 * (`missingTemplateUpload` / `expectedTemplateSha256`) is present on node params.
 */
export function hasInlineTemplateOrExpectation(
  params: Record<string, unknown> | undefined,
): boolean {
  if (!params) return false;
  const rawTpl = typeof params.templatePdbText === 'string' ? params.templatePdbText : '';
  return (
    rawTpl.trim().length > 0 ||
    Boolean(params.missingTemplateUpload) ||
    Boolean(String(params.expectedTemplateSha256 ?? '').trim())
  );
}

/**
 * Enforce shared inline-template upload integrity (`missingTemplateUpload` and
 * `expectedTemplateSha256`) whenever no live `template` port artifact overrides inline params.
 */
export function assertInlineTemplateIntegrity(
  params: Record<string, unknown> | undefined,
  hasConnectedTemplatePort: boolean,
): void {
  if (hasConnectedTemplatePort || !params) return;

  const rawTpl = typeof params.templatePdbText === 'string' ? params.templatePdbText : '';
  const missingTplUpload = Boolean(params.missingTemplateUpload);
  const expectedShaRaw = String(params.expectedTemplateSha256 ?? '').trim();
  const expectedSha = expectedShaRaw.toLowerCase();
  const expectedFilename = String(
    params.expectedTemplateFilename ?? params.templateFilename ?? '',
  ).trim();
  const fileNote = expectedFilename ? ` "${expectedFilename}"` : '';

  if (expectedSha && !TEMPLATE_SHA256_HEX_RE.test(expectedSha)) {
    throw new LocalFoldTemplateError(
      'MISSING_UPLOAD',
      `Invalid expectedTemplateSha256 digest "${expectedShaRaw}" (expected a 64-character SHA-256 hex digest).`,
    );
  }

  if (!rawTpl.trim()) {
    if (missingTplUpload) {
      const digestNote = expectedSha ? ` (expected SHA-256 ${expectedSha.slice(0, 12)}…)` : '';
      throw new LocalFoldTemplateError(
        'MISSING_UPLOAD',
        `Shared inline template${fileNote}${digestNote} was omitted; re-upload the matching PDB/mmCIF structure or clear the expectation before running.`,
      );
    }
    if (expectedSha) {
      throw new LocalFoldTemplateError(
        'MISSING_UPLOAD',
        `Expected inline template${fileNote} (SHA-256 ${expectedSha.slice(0, 12)}…) is missing; re-upload the matching PDB/mmCIF file or clear expectedTemplateSha256.`,
      );
    }
    return;
  }

  if (expectedSha) {
    const actualSha = computeTemplateDigest(rawTpl);
    if (actualSha !== expectedSha) {
      throw new LocalFoldTemplateError(
        'MISSING_UPLOAD',
        `Uploaded template digest (${actualSha.slice(0, 12)}…) does not match expected inline template${fileNote} (SHA-256 ${expectedSha.slice(0, 12)}…). Re-upload the matching file or explicitly clear expectedTemplateSha256.`,
      );
    }
  }
}

export function resolveLocalFoldTemplateSource(opts: {
  templateInput?: StructureArtifact | null;
  params?: Record<string, unknown>;
}): ResolvedLocalFoldTemplateSource | null {
  const hasPortObj = opts.templateInput !== undefined && opts.templateInput !== null;
  const portText = hasPortObj
    ? (opts.templateInput!.cifText?.trim()
        ? opts.templateInput!.cifText!
        : (opts.templateInput!.pdbText ?? ''))
    : '';
  const hasPortText = portText.trim().length > 0;
  if (hasPortObj && !hasPortText) {
    throw new LocalFoldTemplateError(
      'EMPTY_TEMPLATE',
      'Connected template StructureArtifact contains no PDB or mmCIF coordinate text.',
    );
  }

  const inlineText = typeof opts.params?.templatePdbText === 'string' ? opts.params.templatePdbText : '';
  const hasInlineText = inlineText.trim().length > 0;

  if (!hasPortText && !hasInlineText) {
    return null;
  }

  const sourceKind: 'port' | 'inline' = hasPortText ? 'port' : 'inline';
  const text = hasPortText ? portText : inlineText;
  const bothProvided = hasPortText && hasInlineText;
  const warnings: string[] = [];
  if (bothProvided) {
    warnings.push(
      'Both a wired "template" port and inline template text are provided; using the wired template port.',
    );
  }

  const format = detectTemplateFormat(text);
  const inlineFilename =
    typeof opts.params?.templateFilename === 'string' && opts.params.templateFilename.trim()
      ? opts.params.templateFilename.trim()
      : undefined;
  const filename =
    sourceKind === 'port'
      ? (opts.templateInput?.label?.trim() || opts.templateInput?.id || undefined)
      : inlineFilename;

  return {
    sourceKind,
    text,
    format,
    ...(filename ? { filename } : {}),
    digest: computeTemplateDigest(text),
    spanChains: opts.params?.templateSpanChains !== false,
    bothProvided,
    warnings,
  };
}

// ---------------------------------------------------------------------------
// 3. All-Atom PDB & mmCIF Template Parser
// ---------------------------------------------------------------------------

export interface TemplateResidue {
  number: string;
  resName: string;
  code: string;
  atoms: Map<string, [number, number, number]>;
  confidence: number;
  isStandard: boolean;
}

export interface ParsedTemplateChain {
  chainId: string;
  sequence: string;
  residues: TemplateResidue[];
}

export interface ParsedTemplateStructure {
  format: 'pdb' | 'mmcif';
  chains: ParsedTemplateChain[];
  totalResidues: number;
  hasAltLocs: boolean;
  modelNumber: number;
}

interface RawTemplateAtom {
  chainId: string;
  resKey: string;
  resName: string;
  atomName: string;
  altLoc: string;
  coords: [number, number, number];
  occupancy: number;
  bFactor: number;
}

function isStandardAmino3(resName: string): boolean {
  return STANDARD_AMINO_3.has(resName);
}

function keepsTemplateResidue(resName: string): boolean {
  if (PEPTIDE_CAP_RESNAMES.has(resName) || SOLVENT_WATER_RESNAMES.has(resName)) {
    return false;
  }
  if (NUCLEIC_ACID_RESNAMES.has(resName)) {
    return false;
  }
  return (
    STANDARD_AMINO_3.has(resName) ||
    resName === 'UNK' ||
    HETERO_ONE_LETTER[resName] !== undefined ||
    MODIFIED_PARENTS[resName] !== undefined
  );
}

function templateResidueCode(resName: string): string {
  return (
    STANDARD_ONE_LETTER[resName] ??
    HETERO_ONE_LETTER[resName] ??
    STANDARD_ONE_LETTER[MODIFIED_PARENTS[resName]] ??
    'X'
  );
}

function isHydrogenAtom(elementRaw: string, atomNameRaw: string): boolean {
  const el = elementRaw.trim().toUpperCase();
  if (el && el !== '.' && el !== '?') {
    return el === 'H' || el === 'D';
  }
  const name = atomNameRaw.trim().toUpperCase();
  if (!name) return false;
  return /^(?:\d*H[A-Z0-9']*|\d*D[A-Z0-9']*)$/.test(name);
}

function chosenAltLocs(
  atoms: ReadonlyArray<{ key: string; altLoc: string; occupancy: number }>,
  first = false,
): Map<string, string> {
  if (first) {
    const chosen = new Map<string, string>();
    for (const { key, altLoc } of atoms) {
      if (altLoc !== '' && !chosen.has(key)) chosen.set(key, altLoc);
    }
    return chosen;
  }
  const totals = new Map<string, Map<string, [number, number]>>();
  for (const { key, altLoc, occupancy } of atoms) {
    if (altLoc === '') continue;
    let per = totals.get(key);
    if (!per) {
      per = new Map();
      totals.set(key, per);
    }
    const prev = per.get(altLoc) ?? [0, 0];
    per.set(altLoc, [prev[0] + occupancy, prev[1] + 1]);
  }
  const chosen = new Map<string, string>();
  for (const [key, per] of totals) {
    let best = '';
    let bestOccupancy = -Infinity;
    const sorted = [...per.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    for (const [altLoc, [sum, count]] of sorted) {
      const mean = sum / count;
      if (mean > bestOccupancy) {
        best = altLoc;
        bestOccupancy = mean;
      }
    }
    if (best !== '') chosen.set(key, best);
  }
  return chosen;
}

function finishTemplateResidue(
  residue: TemplateResidue,
  opts?: { arginines?: boolean },
): TemplateResidue {
  if (residue.resName === 'MSE' && residue.atoms.has('SE') && !residue.atoms.has('SD')) {
    residue.atoms.set('SD', residue.atoms.get('SE')!);
    residue.atoms.delete('SE');
  }
  if (opts?.arginines !== false && residue.resName === 'ARG') {
    const cd = residue.atoms.get('CD');
    const nh1 = residue.atoms.get('NH1');
    const nh2 = residue.atoms.get('NH2');
    if (cd && nh1 && nh2) {
      const d2 = (a: [number, number, number], b: [number, number, number]): number =>
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
      if (d2(nh1, cd) > d2(nh2, cd)) {
        residue.atoms.set('NH1', nh2);
        residue.atoms.set('NH2', nh1);
      }
    }
  }
  return residue;
}

function parseRawPdbAtoms(text: string): { atoms: RawTemplateAtom[]; modelNumber: number } {
  const atoms: RawTemplateAtom[] = [];
  let modelNumber = 1;
  let seenModel = false;
  let seenAnyCoordInModel = false;

  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (line.startsWith('MODEL ')) {
      if (seenModel && seenAnyCoordInModel) break;
      const num = parseInt(line.slice(10, 16).trim(), 10);
      if (Number.isFinite(num) && num > 0) modelNumber = num;
      seenModel = true;
      continue;
    }
    if (line.startsWith('ENDMDL')) {
      if (seenAnyCoordInModel || seenModel) break;
      continue;
    }
    if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) continue;

    seenAnyCoordInModel = true;
    const atomName = line.slice(12, 16).trim();
    const altLoc = (line[16] ?? ' ').trim();
    const resName = line.slice(17, 20).trim().toUpperCase();
    const chainId = (line[21] ?? ' ').trim() || 'A';
    const resKey = line.slice(22, 27).trim() || '1';
    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    if (!atomName || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    const element = line.slice(76, 78).trim();
    if (isHydrogenAtom(element, atomName)) continue;
    if (!keepsTemplateResidue(resName)) continue;

    const occRaw = parseFloat(line.slice(54, 60));
    const occupancy = Number.isFinite(occRaw) ? occRaw : 1.0;
    const bRaw = parseFloat(line.slice(60, 66));
    const bFactor = Number.isFinite(bRaw) ? bRaw : 0.0;

    atoms.push({
      chainId,
      resKey,
      resName,
      atomName,
      altLoc,
      coords: [x, y, z],
      occupancy,
      bFactor,
    });
  }
  return { atoms, modelNumber };
}

function cleanCifToken(raw: string | undefined): string {
  if (!raw || raw === '.' || raw === '?') return '';
  return raw;
}

function tokenizeCifLineWithQuotes(line: string): Array<{ value: string; quoted: boolean }> {
  const toks: Array<{ value: string; quoted: boolean }> = [];
  const n = line.length;
  let i = 0;
  while (i < n) {
    while (i < n && (line[i] === ' ' || line[i] === '\t' || line[i] === '\r')) i += 1;
    if (i >= n) break;
    const ch = line[i];
    if (ch === '#') break;
    if (ch === "'" || ch === '"') {
      const quote = ch;
      i += 1;
      const start = i;
      while (i < n) {
        if (
          line[i] === quote &&
          (i + 1 === n || line[i + 1] === ' ' || line[i + 1] === '\t' || line[i + 1] === '\r')
        ) {
          break;
        }
        i += 1;
      }
      toks.push({ value: line.slice(start, i), quoted: true });
      if (i < n && line[i] === quote) i += 1;
    } else {
      const start = i;
      while (i < n && line[i] !== ' ' && line[i] !== '\t' && line[i] !== '\r') i += 1;
      toks.push({ value: line.slice(start, i), quoted: false });
    }
  }
  return toks;
}

function parseRawMmcifAtoms(text: string): { atoms: RawTemplateAtom[]; modelNumber: number } {
  const atoms: RawTemplateAtom[] = [];
  const lines = text.split(/\r?\n/);

  let inAtomLoop = false;
  let readingCols = false;
  const cols: string[] = [];
  let pendingRow: string[] = [];
  let inTextBlock = false;
  let textBlockLines: string[] = [];
  let firstModelNum: string | null = null;
  let modelNumber = 1;

  const colIdx = new Map<string, number>();

  const processRow = (row: string[]) => {
    const get = (name: string): string => {
      const idx = colIdx.get(name.toLowerCase());
      return idx !== undefined ? cleanCifToken(row[idx]) : '';
    };

    const modelStr = get('pdbx_PDB_model_num');
    if (modelStr) {
      if (firstModelNum === null) {
        firstModelNum = modelStr;
        const parsedNum = parseInt(modelStr, 10);
        if (Number.isFinite(parsedNum) && parsedNum > 0) modelNumber = parsedNum;
      } else if (modelStr !== firstModelNum) {
        return;
      }
    }

    const resName = (get('label_comp_id') || get('auth_comp_id')).trim().toUpperCase();
    const atomName = (get('label_atom_id') || get('auth_atom_id')).trim();
    if (!resName || !atomName) return;

    const element = get('type_symbol');
    if (isHydrogenAtom(element, atomName)) return;
    if (!keepsTemplateResidue(resName)) return;

    const x = parseFloat(get('Cartn_x'));
    const y = parseFloat(get('Cartn_y'));
    const z = parseFloat(get('Cartn_z'));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;

    const chainId = (get('auth_asym_id') || get('label_asym_id')).trim() || 'A';
    const seqStr = (get('auth_seq_id') || get('label_seq_id')).trim() || '1';
    const insCode = get('pdbx_PDB_ins_code').trim();
    const resKey = insCode ? `${seqStr}${insCode}` : seqStr;
    const altLoc = get('label_alt_id').trim();

    const occRaw = parseFloat(get('occupancy'));
    const occupancy = Number.isFinite(occRaw) ? occRaw : 1.0;
    const bRaw = parseFloat(get('B_iso_or_equiv'));
    const bFactor = Number.isFinite(bRaw) ? bRaw : 0.0;

    atoms.push({
      chainId,
      resKey,
      resName,
      atomName,
      altLoc,
      coords: [x, y, z],
      occupancy,
      bFactor,
    });
  };

  let stopParsing = false;
  for (let li = 0; li < lines.length && !stopParsing; li += 1) {
    const rawLine = lines[li];
    const trimmed = rawLine.trim();

    if (inTextBlock) {
      if (rawLine.startsWith(';')) {
        inTextBlock = false;
        readingCols = false;
        const blockValue = textBlockLines.join('\n');
        textBlockLines = [];
        if (inAtomLoop) {
          pendingRow.push(blockValue);
          while (pendingRow.length >= cols.length && cols.length > 0) {
            const row = pendingRow.slice(0, cols.length);
            pendingRow = pendingRow.slice(cols.length);
            processRow(row);
          }
        }
      } else {
        textBlockLines.push(rawLine);
      }
      continue;
    }
    if (!trimmed || trimmed.startsWith('#')) continue;
    if (rawLine.startsWith(';')) {
      inTextBlock = true;
      textBlockLines = [rawLine.slice(1)];
      continue;
    }

    const toks = tokenizeCifLineWithQuotes(rawLine);
    if (toks.length === 0) continue;

    for (let ti = 0; ti < toks.length; ti += 1) {
      const tok = toks[ti];
      const lower = tok.quoted ? '' : tok.value.toLowerCase();

      if (!tok.quoted && lower === 'loop_') {
        if (inAtomLoop && cols.length > 0 && atoms.length > 0) {
          stopParsing = true;
          break;
        }
        inAtomLoop = false;
        readingCols = true;
        cols.length = 0;
        colIdx.clear();
        pendingRow = [];
        continue;
      }

      if (
        !tok.quoted &&
        (lower === 'stop_' || lower.startsWith('save_') || lower.startsWith('data_'))
      ) {
        if (inAtomLoop) {
          stopParsing = true;
          break;
        }
        readingCols = false;
        pendingRow = [];
        break;
      }

      if (readingCols && !tok.quoted && lower.startsWith('_')) {
        if (lower.startsWith('_atom_site.')) {
          inAtomLoop = true;
          const fieldName = lower.slice('_atom_site.'.length);
          colIdx.set(fieldName, cols.length);
          cols.push(fieldName);
          continue;
        }
        readingCols = false;
        inAtomLoop = false;
        cols.length = 0;
        colIdx.clear();
      }

      if (inAtomLoop) {
        if (!tok.quoted && lower.startsWith('_')) {
          stopParsing = true;
          break;
        }
        readingCols = false;
        pendingRow.push(tok.value);
        while (pendingRow.length >= cols.length && cols.length > 0) {
          const row = pendingRow.slice(0, cols.length);
          pendingRow = pendingRow.slice(cols.length);
          processRow(row);
        }
      }
    }
  }

  return { atoms, modelNumber };
}

export function parseTemplateStructure(
  text: string,
  opts?: { altlocFirst?: boolean; arginines?: boolean },
): ParsedTemplateStructure {
  const format = detectTemplateFormat(text);
  const { atoms, modelNumber } =
    format === 'mmcif' ? parseRawMmcifAtoms(text) : parseRawPdbAtoms(text);

  if (atoms.length === 0) {
    throw new LocalFoldTemplateError(
      'NO_POLYMER_RESIDUES',
      'Template structure contains no valid polymer amino acid residues.',
    );
  }

  const byChain = new Map<string, RawTemplateAtom[]>();
  let hasAltLocs = false;
  for (const atom of atoms) {
    if (atom.altLoc !== '') hasAltLocs = true;
    let list = byChain.get(atom.chainId);
    if (!list) {
      list = [];
      byChain.set(atom.chainId, list);
    }
    list.push(atom);
  }

  const chains: ParsedTemplateChain[] = [];
  let totalResidues = 0;

  for (const [chainId, chainAtoms] of byChain) {
    const keepAlt = chosenAltLocs(
      chainAtoms.map((a) => ({
        key: a.resKey,
        altLoc: a.altLoc,
        occupancy: a.occupancy,
      })),
      opts?.altlocFirst === true,
    );

    const byResKey = new Map<string, { residue: TemplateResidue; atomCount: number }>();
    const ordered: Array<{ residue: TemplateResidue; atomCount: number }> = [];

    for (const a of chainAtoms) {
      if (a.altLoc !== '' && keepAlt.get(a.resKey) !== a.altLoc) continue;
      let entry = byResKey.get(a.resKey);
      if (!entry) {
        const residue: TemplateResidue = {
          number: a.resKey,
          resName: a.resName,
          code: templateResidueCode(a.resName),
          atoms: new Map(),
          confidence: 0,
          isStandard: isStandardAmino3(a.resName),
        };
        entry = { residue, atomCount: 0 };
        byResKey.set(a.resKey, entry);
        ordered.push(entry);
      }
      if (!entry.residue.atoms.has(a.atomName)) {
        entry.residue.atoms.set(a.atomName, a.coords);
        entry.atomCount += 1;
        entry.residue.confidence += (a.bFactor - entry.residue.confidence) / entry.atomCount;
      }
    }

    const residues: TemplateResidue[] = [];
    for (const { residue } of ordered) {
      if (
        !residue.atoms.has('N') &&
        !residue.atoms.has('CA') &&
        !residue.atoms.has('C') &&
        !residue.atoms.has('CB')
      ) {
        continue;
      }
      finishTemplateResidue(residue, { arginines: opts?.arginines !== false });
      residues.push(residue);
    }

    if (residues.length > 0) {
      chains.push({
        chainId,
        sequence: residues.map((r) => r.code).join(''),
        residues,
      });
      totalResidues += residues.length;
    }
  }

  if (totalResidues === 0 || chains.length === 0) {
    throw new LocalFoldTemplateError(
      'NO_POLYMER_RESIDUES',
      'Template structure contains no polymer amino acid residues with backbone atoms.',
    );
  }

  return {
    format,
    chains,
    totalResidues,
    hasAltLocs,
    modelNumber,
  };
}

// ---------------------------------------------------------------------------
// 4. Query-to-Template Chain Matching & Semi-Global Alignment
// ---------------------------------------------------------------------------

export interface QueryChainSpec {
  chainId?: string;
  sequence: string;
}

export interface ChainTemplateAlignment {
  queryChainIndex: number;
  queryChainId: string;
  templateChainId: string | null;
  templateChainIndex: number | null;
  /** query residue index (0-based within chain) -> template residue index (0-based within chain) */
  residueMap: Map<number, number>;
  queryLength: number;
  templateLength: number;
  alignedCount: number;
  identicalCount: number;
  identity: number;
  coverage: number;
}

export interface TemplateAlignmentResult {
  parsed: ParsedTemplateStructure;
  chainAlignments: ChainTemplateAlignment[];
  matchedQueryResidues: number;
  totalQueryResidues: number;
  matchedChains: number;
  totalQueryChains: number;
  overallIdentity: number;
  overallCoverage: number;
  warnings: string[];
}

interface AlignmentSliceStats {
  alignedCount: number;
  identicalCount: number;
  gapOpens: number;
  gapCols: number;
  blosumSum: number;
  subSum: number;
  rawFirstPair: [number, number];
  rawLastPair: [number, number];
  prefixId: Int32Array;
  prefixBlosum: Int32Array;
  prefixGapOpens: Int32Array;
  prefixGapCols: Int32Array;
}

interface PairwiseDpResult extends AlignmentSliceStats {
  score: number;
  coreScore: number;
  residueMap: Map<number, number>;
  rawIdenticalCount: number;
  identity: number;
  coverage: number;
  rawAlignedCount: number;
  preFilterStats: AlignmentSliceStats | null;
}

const DP_NEG_INF = -1_000_000_000;
const DP_MATCH = 400;
const DP_MISMATCH = -100;
const DP_X_MATCH = 0;
const DP_GAP_OPEN = -1600;
const DP_GAP_EXTEND = -200;
const DP_END_GAP = 55;

// Standard NCBI BLOSUM62 matrix over AF2_RESTYPE_ORDER ('ARNDCQEGHILKMFPSTWYV')
const BLOSUM62_FLAT: readonly number[] = [
  // A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
   4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, // A
  -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, // R
  -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, // N
  -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, // D
   0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, // C
  -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, // Q
  -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, // E
   0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, // G
  -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, // H
  -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, // I
  -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, // L
  -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, // K
  -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, // M
  -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, // F
  -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, // P
   1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, // S
   0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, // T
  -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, // W
  -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, // Y
   0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, // V
];

function blosum62Score(a: string, b: string): number {
  const ia = AATYPE_INDEX_BY_CHAR[a];
  const ib = AATYPE_INDEX_BY_CHAR[b];
  if (ia === undefined || ib === undefined) return 0;
  return BLOSUM62_FLAT[ia * 20 + ib];
}

const LANCZOS_COEFFS = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

function lgamma(z: number): number {
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  }
  const x = z - 1;
  let a = LANCZOS_COEFFS[0];
  const t = x + 7.5;
  for (let i = 1; i < LANCZOS_COEFFS.length; i += 1) {
    a += LANCZOS_COEFFS[i] / (x + i);
  }
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function logBinomUpperTail(n: number, k: number, p: number): number {
  if (k <= 0) return 0;
  if (k > n) return -Infinity;
  if (k <= n * p) return 0;
  const logCk =
    lgamma(n + 1) -
    lgamma(k + 1) -
    lgamma(n - k + 1) +
    k * Math.log(p) +
    (n - k) * Math.log(1 - p);
  let sum = 1;
  let term = 1;
  for (let j = k; j < n; j += 1) {
    term *= ((n - j) * p) / ((j + 1) * (1 - p));
    sum += term;
    if (term < 1e-12 * sum) break;
  }
  return Math.min(0, logCk + Math.log(sum));
}

function binomUpperTail(n: number, k: number, p: number): number {
  const logTotal = logBinomUpperTail(n, k, p);
  if (!Number.isFinite(logTotal)) return 0;
  return logTotal < -745 ? 0 : Math.min(1, Math.exp(logTotal));
}

function trimTerminalTags(
  pairs: ReadonlyArray<readonly [number, number]>,
  qSeq: string,
  tSeq: string,
): Array<[number, number]> {
  const n = pairs.length;
  if (n < 16) {
    return pairs.map(([qi, ti]) => [qi, ti]);
  }

  const isMatch = (idx: number): number => {
    const [qi, ti] = pairs[idx];
    return qSeq[qi] !== 'X' && qSeq[qi] === tSeq[ti] ? 1 : 0;
  };

  const prefixMatchSum = new Int32Array(n + 1);
  for (let i = 0; i < n; i += 1) {
    prefixMatchSum[i + 1] = prefixMatchSum[i] + isMatch(i);
  }

  // N-terminal pass
  let start = 0;
  {
    let s = 0;
    let idCount = 0;
    let minS = 0;
    let minK = 0;
    let minId = 0;
    for (let i = 0; i < n; i += 1) {
      if (i > 0) {
        const [pqi, pti] = pairs[i - 1];
        const [qi, ti] = pairs[i];
        const internalGapCols = Math.max(0, qi - pqi - 1) + Math.max(0, ti - pti - 1);
        s -= 200 * internalGapCols;
      }
      const m = isMatch(i);
      idCount += m;
      s += m ? 450 : -100;
      if (s < minS) {
        minS = s;
        minK = i + 1;
        minId = idCount;
      }
    }

    let bestK = minK;
    if (
      minK >= 6 &&
      minK + 2 < n &&
      minS <= -600 &&
      isMatch(minK) + isMatch(minK + 1) + isMatch(minK + 2) < 2
    ) {
      const limit = Math.min(n - 15, minK + 5);
      let extraId = 0;
      for (let j = minK; j < limit; j += 1) {
        const m = isMatch(j);
        if (
          m === 0 &&
          (minId + extraId) / (j + 1) <= 0.125 &&
          isMatch(j + 1) + isMatch(j + 2) + isMatch(j + 3) >= 2
        ) {
          bestK = j + 1;
        }
        extraId += m;
      }
    }

    if (bestK >= 8 && n - bestK >= 15) {
      const trimId = prefixMatchSum[bestK];
      const remId = prefixMatchSum[n] - trimId;
      if (trimId / bestK <= 0.125 && remId / (n - bestK) >= 0.20) {
        start = bestK;
      }
    }
  }

  // C-terminal pass
  let end = n;
  if (n - start >= 16) {
    let s = 0;
    let idCount = 0;
    let minS = 0;
    let minK = 0;
    let minId = 0;
    for (let k = 1; k <= n - start; k += 1) {
      const idx = n - k;
      if (k > 1) {
        const [qi, ti] = pairs[idx];
        const [nqi, nti] = pairs[idx + 1];
        const internalGapCols = Math.max(0, nqi - qi - 1) + Math.max(0, nti - ti - 1);
        s -= 200 * internalGapCols;
      }
      const m = isMatch(idx);
      idCount += m;
      s += m ? 450 : -100;
      if (s < minS) {
        minS = s;
        minK = k;
        minId = idCount;
      }
    }

    let bestK = minK;
    if (
      minK >= 6 &&
      n - 3 - minK >= start &&
      minS <= -600 &&
      isMatch(n - 1 - minK) + isMatch(n - 2 - minK) + isMatch(n - 3 - minK) < 2
    ) {
      const limit = Math.min(n - start - 15, minK + 5);
      let extraId = 0;
      for (let k = minK; k < limit; k += 1) {
        const idx = n - 1 - k;
        const m = isMatch(idx);
        if (
          m === 0 &&
          (minId + extraId) / (k + 1) <= 0.125 &&
          isMatch(idx - 1) + isMatch(idx - 2) + isMatch(idx - 3) >= 2
        ) {
          bestK = k + 1;
        }
        extraId += m;
      }
    }

    if (bestK >= 8 && n - start - bestK >= 15) {
      const trimId = prefixMatchSum[n] - prefixMatchSum[n - bestK];
      const remId = prefixMatchSum[n - bestK] - prefixMatchSum[start];
      if (trimId / bestK <= 0.125 && remId / (n - start - bestK) >= 0.20) {
        end = n - bestK;
      }
    }
  }

  return pairs.slice(start, end).map(([qi, ti]) => [qi, ti]);
}

// Traceback codes
const TB_STOP = 0; // start fresh at (i, j)
const TB_M_FROM_M = 1;
const TB_M_FROM_IQ = 2;
const TB_M_FROM_IT = 3;
const TB_IQ_FROM_M = 4;
const TB_IQ_FROM_IQ = 5;
const TB_IQ_FROM_IT = 6;
const TB_IT_FROM_M = 7;
const TB_IT_FROM_IT = 8;
const TB_IT_FROM_IQ = 9;

function alignSingleChainPair(
  qSeqRaw: string,
  tChain: ParsedTemplateChain,
  minConfidence = 0,
): PairwiseDpResult {
  const qSeq = qSeqRaw.trim().toUpperCase();
  const tSeq = tChain.sequence.trim().toUpperCase();
  const qLen = qSeq.length;
  const tLen = tSeq.length;

  if (qLen === 0 || tLen === 0) {
    return {
      score: DP_NEG_INF,
      coreScore: DP_NEG_INF,
      residueMap: new Map(),
      alignedCount: 0,
      identicalCount: 0,
      rawIdenticalCount: 0,
      identity: 0,
      coverage: 0,
      gapOpens: 0,
      gapCols: 0,
      blosumSum: 0,
      subSum: 0,
      rawAlignedCount: 0,
      rawFirstPair: [0, 0],
      rawLastPair: [0, 0],
      prefixId: new Int32Array(1),
      prefixBlosum: new Int32Array(1),
      prefixGapOpens: new Int32Array(1),
      prefixGapCols: new Int32Array(1),
      preFilterStats: null,
    };
  }

  let pairs: Array<[number, number]> = [];
  let score = 0;

  if (qSeq === tSeq) {
    for (let i = 0; i < qLen; i += 1) {
      pairs.push([i, i]);
    }
    score = qLen * DP_MATCH;
  } else {
    const stride = tLen + 1;
    const size = (qLen + 1) * stride;
    const M = new Int32Array(size).fill(DP_NEG_INF);
    const Iq = new Int32Array(size).fill(DP_NEG_INF);
    const It = new Int32Array(size).fill(DP_NEG_INF);
    const tbM = new Uint8Array(size);
    const tbIq = new Uint8Array(size);
    const tbIt = new Uint8Array(size);

    let bestScore = DP_NEG_INF;
    let bestI = -1;
    let bestJ = -1;

    for (let i = 1; i <= qLen; i += 1) {
      const qc = qSeq[i - 1];
      const rowOffset = i * stride;
      const prevRowOffset = (i - 1) * stride;

      for (let j = 1; j <= tLen; j += 1) {
        const tc = tSeq[j - 1];
        const idx = rowOffset + j;
        const diag = prevRowOffset + (j - 1);
        const up = prevRowOffset + j;
        const left = rowOffset + (j - 1);

        const sub = qc === 'X' || tc === 'X' ? DP_X_MATCH : qc === tc ? DP_MATCH : DP_MISMATCH;

        // Semi-global free start: allow free terminal overhang starts ONLY on boundary cells (i === 1 || j === 1),
        // never in the interior (i > 1 && j > 1), so low-identity N-terminal core segments are not dropped.
        const canFreeStart = i === 1 || j === 1;
        let mVal = canFreeStart ? sub - ((i - 1) + (j - 1)) * DP_END_GAP : DP_NEG_INF;
        let mTb = canFreeStart ? TB_STOP : 0;

        if (M[diag] !== DP_NEG_INF && M[diag] + sub > mVal) {
          mVal = M[diag] + sub;
          mTb = TB_M_FROM_M;
        }
        if (Iq[diag] !== DP_NEG_INF && Iq[diag] + sub > mVal) {
          mVal = Iq[diag] + sub;
          mTb = TB_M_FROM_IQ;
        }
        if (It[diag] !== DP_NEG_INF && It[diag] + sub > mVal) {
          mVal = It[diag] + sub;
          mTb = TB_M_FROM_IT;
        }
        M[idx] = mVal;
        tbM[idx] = mTb;

        // Gap in template (consuming q[i-1])
        let iqVal = DP_NEG_INF;
        let iqTb = 0;
        if (M[up] !== DP_NEG_INF) {
          iqVal = M[up] + DP_GAP_OPEN + DP_GAP_EXTEND;
          iqTb = TB_IQ_FROM_M;
        }
        if (Iq[up] !== DP_NEG_INF && Iq[up] + DP_GAP_EXTEND > iqVal) {
          iqVal = Iq[up] + DP_GAP_EXTEND;
          iqTb = TB_IQ_FROM_IQ;
        }
        if (It[up] !== DP_NEG_INF && It[up] + DP_GAP_OPEN + DP_GAP_EXTEND > iqVal) {
          iqVal = It[up] + DP_GAP_OPEN + DP_GAP_EXTEND;
          iqTb = TB_IQ_FROM_IT;
        }
        Iq[idx] = iqVal;
        tbIq[idx] = iqTb;

        // Gap in query (consuming t[j-1])
        let itVal = DP_NEG_INF;
        let itTb = 0;
        if (M[left] !== DP_NEG_INF) {
          itVal = M[left] + DP_GAP_OPEN + DP_GAP_EXTEND;
          itTb = TB_IT_FROM_M;
        }
        if (It[left] !== DP_NEG_INF && It[left] + DP_GAP_EXTEND > itVal) {
          itVal = It[left] + DP_GAP_EXTEND;
          itTb = TB_IT_FROM_IT;
        }
        if (Iq[left] !== DP_NEG_INF && Iq[left] + DP_GAP_OPEN + DP_GAP_EXTEND > itVal) {
          itVal = Iq[left] + DP_GAP_OPEN + DP_GAP_EXTEND;
          itTb = TB_IT_FROM_IQ;
        }
        It[idx] = itVal;
        tbIt[idx] = itTb;

        // Semi-global free end: terminate only on boundary cells (i === qLen || j === tLen)
        if (i === qLen || j === tLen) {
          const endScore = mVal - ((qLen - i) + (tLen - j)) * DP_END_GAP;
          if (endScore > bestScore) {
            bestScore = endScore;
            bestI = i;
            bestJ = j;
          }
        }
      }
    }

    if (bestI <= 0 || bestJ <= 0) {
      return {
        score: DP_NEG_INF,
        coreScore: DP_NEG_INF,
        residueMap: new Map(),
        alignedCount: 0,
        identicalCount: 0,
        rawIdenticalCount: 0,
        identity: 0,
        coverage: 0,
        gapOpens: 0,
        gapCols: 0,
        blosumSum: 0,
        subSum: 0,
        rawAlignedCount: 0,
        rawFirstPair: [0, 0],
        rawLastPair: [0, 0],
        prefixId: new Int32Array(1),
        prefixBlosum: new Int32Array(1),
        prefixGapOpens: new Int32Array(1),
        prefixGapCols: new Int32Array(1),
        preFilterStats: null,
      };
    }

    let ci = bestI;
    let cj = bestJ;
    let state: 'M' | 'IQ' | 'IT' = 'M';

    while (ci > 0 && cj > 0) {
      const idx = ci * stride + cj;
      if (state === 'M') {
        const tb = tbM[idx];
        pairs.push([ci - 1, cj - 1]);
        ci -= 1;
        cj -= 1;
        if (tb === TB_STOP) break;
        if (tb === TB_M_FROM_IQ) state = 'IQ';
        else if (tb === TB_M_FROM_IT) state = 'IT';
        else state = 'M';
      } else if (state === 'IQ') {
        const tb = tbIq[idx];
        ci -= 1;
        if (tb === TB_IQ_FROM_M) state = 'M';
        else if (tb === TB_IQ_FROM_IT) state = 'IT';
        else state = 'IQ';
      } else {
        const tb = tbIt[idx];
        cj -= 1;
        if (tb === TB_IT_FROM_M) state = 'M';
        else if (tb === TB_IT_FROM_IQ) state = 'IQ';
        else state = 'IT';
      }
    }

    pairs.reverse();
    pairs = trimTerminalTags(pairs, qSeq, tSeq);
    score = bestScore;
  }

  const rawPairs = pairs;
  const rawAlignedCount = rawPairs.length;
  let rawIdenticalCount = 0;
  for (let r = 0; r < rawAlignedCount; r += 1) {
    const [qi, ti] = rawPairs[r];
    if (qSeq[qi] !== 'X' && qSeq[qi] === tSeq[ti]) {
      rawIdenticalCount += 1;
    }
  }

  if (rawAlignedCount === 0) {
    return {
      score: DP_NEG_INF,
      coreScore: DP_NEG_INF,
      residueMap: new Map(),
      alignedCount: 0,
      identicalCount: 0,
      rawIdenticalCount: 0,
      identity: 0,
      coverage: 0,
      gapOpens: 0,
      gapCols: 0,
      blosumSum: 0,
      subSum: 0,
      rawAlignedCount: 0,
      rawFirstPair: [0, 0],
      rawLastPair: [0, 0],
      prefixId: new Int32Array(1),
      prefixBlosum: new Int32Array(1),
      prefixGapOpens: new Int32Array(1),
      prefixGapCols: new Int32Array(1),
      preFilterStats: null,
    };
  }

  // Compute cumulative sequence-alignment DP gaps on pre-confidence rawPairs
  const rawCumGapOpens = new Int32Array(rawAlignedCount);
  const rawCumGapCols = new Int32Array(rawAlignedCount);
  for (let r = 1; r < rawAlignedCount; r += 1) {
    const [pqi, pti] = rawPairs[r - 1];
    const [qi, ti] = rawPairs[r];
    const dq = qi - pqi - 1;
    const dt = ti - pti - 1;
    let stepGo = 0;
    let stepGc = 0;
    if (dq > 0) {
      stepGo += 1;
      stepGc += dq;
    }
    if (dt > 0) {
      stepGo += 1;
      stepGc += dt;
    }
    rawCumGapOpens[r] = rawCumGapOpens[r - 1] + stepGo;
    rawCumGapCols[r] = rawCumGapCols[r - 1] + stepGc;
  }

  const survivingRawIdx: number[] = [];
  if (minConfidence > 0) {
    for (let r = 0; r < rawAlignedCount; r += 1) {
      const ti = rawPairs[r][1];
      const res = tChain.residues[ti];
      if (res && res.confidence >= minConfidence) {
        survivingRawIdx.push(r);
      }
    }
  } else {
    for (let r = 0; r < rawAlignedCount; r += 1) {
      survivingRawIdx.push(r);
    }
  }

  if (survivingRawIdx.length === 0) {
    return {
      score: DP_NEG_INF,
      coreScore: DP_NEG_INF,
      residueMap: new Map(),
      alignedCount: 0,
      identicalCount: 0,
      rawIdenticalCount,
      identity: 0,
      coverage: 0,
      gapOpens: 0,
      gapCols: 0,
      blosumSum: 0,
      subSum: 0,
      rawAlignedCount,
      rawFirstPair: [0, 0],
      rawLastPair: [0, 0],
      prefixId: new Int32Array(1),
      prefixBlosum: new Int32Array(1),
      prefixGapOpens: new Int32Array(1),
      prefixGapCols: new Int32Array(1),
      preFilterStats: null,
    };
  }

  const computeStats = (
    indices: ReadonlyArray<number>,
    outMap: Map<number, number> | null,
  ): AlignmentSliceStats => {
    const alLen = indices.length;
    const prefixId = new Int32Array(alLen + 1);
    const prefixBlosum = new Int32Array(alLen + 1);
    const prefixGapOpens = new Int32Array(alLen + 1);
    const prefixGapCols = new Int32Array(alLen + 1);
    let gapOpens = 0;
    let gapCols = 0;
    let blosumSum = 0;
    let subSum = 0;
    let identicalCount = 0;

    for (let i = 0; i < alLen; i += 1) {
      const r = indices[i];
      const [qi, ti] = rawPairs[r];
      if (outMap) outMap.set(qi, ti);
      const stepGo = i > 0 ? rawCumGapOpens[r] - rawCumGapOpens[indices[i - 1]] : 0;
      const stepGc = i > 0 ? rawCumGapCols[r] - rawCumGapCols[indices[i - 1]] : 0;
      gapOpens += stepGo;
      gapCols += stepGc;
      const qc = qSeq[qi];
      const tc = tSeq[ti];
      const bScore = blosum62Score(qc, tc);
      const isExact = qc !== 'X' && qc === tc ? 1 : 0;
      blosumSum += bScore;
      identicalCount += isExact;
      subSum += qc === 'X' || tc === 'X' ? DP_X_MATCH : qc === tc ? DP_MATCH : DP_MISMATCH;

      prefixId[i + 1] = prefixId[i] + isExact;
      prefixBlosum[i + 1] = prefixBlosum[i] + bScore;
      prefixGapOpens[i + 1] = prefixGapOpens[i] + stepGo;
      prefixGapCols[i + 1] = prefixGapCols[i] + stepGc;
    }

    const first = rawPairs[indices[0]];
    const last = rawPairs[indices[alLen - 1]];
    return {
      alignedCount: alLen,
      identicalCount,
      gapOpens,
      gapCols,
      blosumSum,
      subSum,
      rawFirstPair: [first[0], first[1]],
      rawLastPair: [last[0], last[1]],
      prefixId,
      prefixBlosum,
      prefixGapOpens,
      prefixGapCols,
    };
  };

  const map = new Map<number, number>();
  const stats = computeStats(survivingRawIdx, map);

  let preFilterStats: AlignmentSliceStats | null = null;
  if (survivingRawIdx.length < rawAlignedCount) {
    const allIdx = Array.from({ length: rawAlignedCount }, (_, idx) => idx);
    preFilterStats = computeStats(allIdx, null);
  }

  const overhangCols =
    stats.rawFirstPair[0] +
    stats.rawFirstPair[1] +
    (qLen - 1 - stats.rawLastPair[0]) +
    (tLen - 1 - stats.rawLastPair[1]);
  const coreScore =
    stats.subSum +
    stats.gapOpens * DP_GAP_OPEN +
    stats.gapCols * DP_GAP_EXTEND -
    Math.min(overhangCols, 40) * 10;

  const alignedCount = map.size;
  const identity = alignedCount > 0 ? stats.identicalCount / alignedCount : 0;
  const coverage = qLen > 0 ? alignedCount / qLen : 0;

  return {
    score,
    coreScore,
    residueMap: map,
    ...stats,
    alignedCount,
    rawIdenticalCount,
    identity,
    coverage,
    rawAlignedCount,
    preFilterStats,
  };
}

function isBodySliceViable(
  qLen: number,
  tLen: number,
  stats: AlignmentSliceStats,
  start: number,
  end: number,
  minIdentity: number,
  minAligned: number,
  minLen: number,
): boolean {
  const restLen = end - start;
  if (restLen <= 0) return false;
  const restId = stats.prefixId[end] - stats.prefixId[start];
  if (restId < Math.min(2, restLen)) return false;
  if (restLen < Math.min(10, minAligned) && restId / restLen < 0.60) return false;

  const restGo =
    start === 0
      ? stats.prefixGapOpens[end]
      : stats.prefixGapOpens[end] - stats.prefixGapOpens[start + 1];
  const restGc =
    start === 0
      ? stats.prefixGapCols[end]
      : stats.prefixGapCols[end] - stats.prefixGapCols[start + 1];
  const restBlosumAdj =
    stats.prefixBlosum[end] - stats.prefixBlosum[start] - 11 * restGo - restGc;
  const restEff = restLen + restGc + 6 * restGo;
  if (restEff <= 0) return false;
  if (restId / restLen < minIdentity && restBlosumAdj < 0.40 * restLen) return false;

  const restLogP = logBinomUpperTail(restLen + restGc, restId, 0.058) + restGo * Math.log(35);
  const restUnalignedMin = Math.max(0, minLen - restLen);
  const bodyTrimmedN = start > 0 || (stats.rawFirstPair[0] > 0 && stats.rawFirstPair[1] > 0);
  const bodyTrimmedC =
    end < stats.alignedCount ||
    (stats.rawLastPair[0] < qLen - 1 && stats.rawLastPair[1] < tLen - 1);
  let restSh = Math.max(1, Math.abs(qLen - tLen) + 1 + 2 * restUnalignedMin);
  if (bodyTrimmedN && bodyTrimmedC) {
    restSh = Math.max(restSh, 0.25 * qLen * tLen);
  }
  const restLogEVal = restLogP + Math.log(restSh);

  if (restLen < minAligned || (restLen < 0.5 * minLen && bodyTrimmedN && bodyTrimmedC)) {
    return restLogEVal <= Math.log(0.002) && restBlosumAdj >= 0.40 * restLen;
  }
  if (restLogP <= Math.log(0.01) && restBlosumAdj >= 0) return true;
  if (restLogP <= Math.log(0.05) && restBlosumAdj >= 0.40 * restLen) return true;
  return false;
}

function hasViableTagStrippedBodies(
  qLen: number,
  tLen: number,
  stats: AlignmentSliceStats,
  minIdentity: number,
  minAligned: number,
  minLen: number,
): boolean {
  const al = stats.alignedCount;
  if (al < 12) return true;

  const maxWin = Math.min(24, al - 4);
  for (let w = 8; w <= maxWin; w += 1) {
    const restLen = al - w;
    // N-terminal window [0..w) + C-terminal body [w..al)
    const nWinId = stats.prefixId[w];
    const nRestId = stats.identicalCount - nWinId;
    if (nWinId / w >= 0.60 && nWinId / w - nRestId / restLen >= 0.30) {
      if (!isBodySliceViable(qLen, tLen, stats, w, al, minIdentity, minAligned, minLen)) {
        return false;
      }
    }

    // C-terminal window [al-w..al) + N-terminal body [0..al-w)
    const split = restLen;
    const cWinId = stats.prefixId[al] - stats.prefixId[split];
    const cRestId = stats.prefixId[split];
    if (cWinId / w >= 0.60 && cWinId / w - cRestId / restLen >= 0.30) {
      if (!isBodySliceViable(qLen, tLen, stats, 0, split, minIdentity, minAligned, minLen)) {
        return false;
      }
    }
  }

  // Dual-terminal tag check: interior body [w1 .. al - w2)
  if (al >= 20) {
    const maxW = Math.min(24, Math.floor((al - 4) / 2));
    for (let w1 = 8; w1 <= maxW; w1 += 1) {
      const nWinId = stats.prefixId[w1];
      if (nWinId / w1 < 0.60) continue;
      for (let w2 = 8; w2 <= Math.min(24, al - w1 - 4); w2 += 1) {
        const end = al - w2;
        const cWinId = stats.prefixId[al] - stats.prefixId[end];
        if (cWinId / w2 < 0.60) continue;
        const midLen = end - w1;
        const midId = stats.prefixId[end] - stats.prefixId[w1];
        if (Math.min(nWinId / w1, cWinId / w2) - midId / midLen >= 0.30) {
          if (!isBodySliceViable(qLen, tLen, stats, w1, end, minIdentity, minAligned, minLen)) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

function isAlignmentViable(
  qLen: number,
  tLen: number,
  dp: PairwiseDpResult,
  minIdentity: number,
  minCoverageOpt: number | undefined,
): boolean {
  const minLen = Math.min(qLen, tLen);
  const al = dp.alignedCount;
  const idn = dp.identicalCount;
  const go = dp.gapOpens;
  const gc = dp.gapCols;

  if (minLen <= 0 || al <= 0) return false;

  // 1. Tiny peptides (minLen <= 4)
  if (minLen <= 4) {
    return al === minLen && idn >= Math.min(2, minLen) && dp.identity >= minIdentity;
  }

  // 2. Coverage floor over Math.min(qLen, tLen)
  const covFrac = minCoverageOpt !== undefined ? Math.min(0.35, minCoverageOpt) : 0.35;
  const minAligned = Math.max(Math.min(4, minLen), Math.ceil(covFrac * minLen));
  if (al < minAligned) return false;

  // 3. Gap-penalized effective identity & minimum matches
  const effAligned = al + gc + 6 * go;
  if (
    effAligned <= 0 ||
    idn / effAligned < minIdentity ||
    dp.identity < minIdentity ||
    idn < Math.min(2, minLen)
  ) {
    return false;
  }

  // 3b. Reject shared N/C-terminal or dual-terminal tags/linkers carrying a non-viable protein body
  // (evaluated on both filtered alignment and pre-confidence rawPairs so B-factor masking of the body
  // cannot leave a shared tag passing on its own).
  if (!hasViableTagStrippedBodies(qLen, tLen, dp, minIdentity, minAligned, minLen)) {
    return false;
  }
  if (
    dp.preFilterStats &&
    !hasViableTagStrippedBodies(qLen, tLen, dp.preFilterStats, minIdentity, minAligned, minLen)
  ) {
    return false;
  }

  // 3c. Reject short (<= 24 aa) tag-length fragments flanked at the same terminus by >= 12 unaligned
  // residues on BOTH query and template (e.g. trimTerminalTags stripping unrelated short bodies),
  // while keeping 25+ aa conserved domains intact.
  const tailStats = dp.preFilterStats ?? dp;
  const qLead = tailStats.rawFirstPair[0];
  const tLead = tailStats.rawFirstPair[1];
  const qTrail = qLen - 1 - tailStats.rawLastPair[0];
  const tTrail = tLen - 1 - tailStats.rawLastPair[1];
  if (
    tailStats.alignedCount <= 24 &&
    ((qLead >= 12 && tLead >= 12) || (qTrail >= 12 && tTrail >= 12))
  ) {
    return false;
  }

  // 4. Statistical significance + BLOSUM62 conservation gate
  const unalignedMin = Math.max(0, minLen - al);
  const trimmedN = dp.rawFirstPair[0] > 0 && dp.rawFirstPair[1] > 0;
  const trimmedC = dp.rawLastPair[0] < qLen - 1 && dp.rawLastPair[1] < tLen - 1;
  const blosumAdj = dp.blosumSum - 11 * go - gc;

  if (minLen <= 15) {
    const p = binomUpperTail(al + gc, idn, 0.055);
    const eVal = p * (1 + 2 * unalignedMin) * Math.pow(35, go);
    return eVal <= 0.05 && dp.identity >= Math.max(minIdentity, 0.20);
  }

  const logP = logBinomUpperTail(al + gc, idn, 0.058);
  let sh = Math.max(1, Math.abs(qLen - tLen) + 1 + 2 * unalignedMin);
  if (trimmedN && trimmedC) {
    sh = Math.max(sh, 0.25 * qLen * tLen);
  }
  const logSh = Math.log(sh);
  const logEVal = logP + logSh + go * Math.log(35);

  if (minLen <= 25) {
    return (
      logEVal <= Math.log(0.003) ||
      (logEVal <= Math.log(0.006) && blosumAdj >= 0.5 * al)
    );
  }

  const hasGapsOrShift = go > 0 || unalignedMin > 0;
  const okE = logEVal <= Math.log(0.002) && (!hasGapsOrShift || blosumAdj >= 0.15 * al);
  const log10Go = go * Math.log(10);
  const okRescue =
    (logP + log10Go <= Math.log(0.01) &&
      blosumAdj >= 0.55 * al &&
      al >= 0.5 * minLen &&
      Math.abs(dp.rawFirstPair[0] - dp.rawFirstPair[1]) <= 2 &&
      !(trimmedN && trimmedC)) ||
    (logP + log10Go <= Math.log(0.01) &&
      trimmedN &&
      trimmedC &&
      blosumAdj >= 0.50 * al &&
      al >= Math.max(15, 0.5 * minLen)) ||
    (logP + logSh + log10Go <= Math.log(0.015) &&
      blosumAdj >= 0.60 * al &&
      al >= 0.85 * minLen);
  return okE || okRescue;
}

export function alignQueryToTemplate(
  queryChainsInput: ReadonlyArray<QueryChainSpec>,
  parsed: ParsedTemplateStructure,
  opts?: { minIdentity?: number; minCoverage?: number; minConfidence?: number },
): TemplateAlignmentResult {
  if (parsed.chains.length === 0 || parsed.totalResidues === 0) {
    throw new LocalFoldTemplateError(
      'NO_POLYMER_RESIDUES',
      'Template structure contains no polymer chains.',
    );
  }

  const normalizedQuery = queryChainsInput
    .map((c, idx) => ({
      origIndex: idx,
      chainId: c.chainId?.trim() || (idx < 26 ? String.fromCharCode(65 + idx) : `C${idx + 1}`),
      sequence: c.sequence.trim().toUpperCase().replace(/[^A-Z]/g, ''),
    }))
    .filter((c) => c.sequence.length > 0);

  if (normalizedQuery.length === 0) {
    throw new LocalFoldTemplateError(
      'ZERO_SEQUENCE_OVERLAP',
      'Query sequence contains no protein amino acid residues to align against the template.',
    );
  }

  const minIdentity = opts?.minIdentity ?? 0.15;
  const minConfidence = opts?.minConfidence ?? 0;

  const Q = normalizedQuery.length;
  const T = parsed.chains.length;

  interface CandidateEdge {
    qi: number;
    ti: number;
    dp: PairwiseDpResult;
    tieScore: number;
    viable: boolean;
  }

  const matrix: CandidateEdge[][] = Array.from({ length: Q }, () => []);
  let bestIdenticalAcrossAll = 0;
  let bestRawIdenticalAcrossAll = 0;
  let bestIdentityAcrossAll = 0;

  for (let qi = 0; qi < Q; qi += 1) {
    const q = normalizedQuery[qi];
    for (let ti = 0; ti < T; ti += 1) {
      const t = parsed.chains[ti];
      const dp = alignSingleChainPair(q.sequence, t, minConfidence);
      if (dp.identicalCount > bestIdenticalAcrossAll) {
        bestIdenticalAcrossAll = dp.identicalCount;
      }
      if (dp.rawIdenticalCount > bestRawIdenticalAcrossAll) {
        bestRawIdenticalAcrossAll = dp.rawIdenticalCount;
      }
      if (dp.identity > bestIdentityAcrossAll) {
        bestIdentityAcrossAll = dp.identity;
      }
      const viable = isAlignmentViable(
        q.sequence.length,
        t.sequence.length,
        dp,
        minIdentity,
        opts?.minCoverage,
      );
      const chainBonus = q.chainId === t.chainId ? 50 : 0;
      matrix[qi][ti] = {
        qi,
        ti,
        dp,
        tieScore: dp.coreScore + chainBonus,
        viable,
      };
    }
  }

  // Optimal 1-to-1 bipartite assignment (query chain -> template chain)
  const assignedTemplate = new Array<number | null>(Q).fill(null);

  if (Math.max(Q, T) <= 8) {
    let bestTotal = -Infinity;
    const current = new Array<number | null>(Q).fill(null);
    const usedT = new Uint8Array(T);

    const search = (qi: number, acc: number) => {
      if (qi === Q) {
        if (acc > bestTotal) {
          bestTotal = acc;
          for (let k = 0; k < Q; k += 1) assignedTemplate[k] = current[k];
        }
        return;
      }
      for (let ti = 0; ti < T; ti += 1) {
        if (usedT[ti]) continue;
        const edge = matrix[qi][ti];
        if (!edge.viable) continue;
        usedT[ti] = 1;
        current[qi] = ti;
        // Map viable edges strictly > 0 so valid low-identity matches (15-20%, dp.score <= 0)
        // still beat leaving the chain unmatched (+0) while preserving score ordering.
        const utility = edge.tieScore > 0 ? edge.tieScore + 1 : 1 / (1 - edge.tieScore);
        search(qi + 1, acc + utility);
        usedT[ti] = 0;
        current[qi] = null;
      }
      // Always allow leaving qi unmatched (e.g. subset template or better global match)
      current[qi] = null;
      search(qi + 1, acc);
    };
    search(0, 0);
  } else {
    const edges: CandidateEdge[] = [];
    for (let qi = 0; qi < Q; qi += 1) {
      for (let ti = 0; ti < T; ti += 1) {
        if (matrix[qi][ti].viable) edges.push(matrix[qi][ti]);
      }
    }
    edges.sort((a, b) => b.tieScore - a.tieScore);
    const usedQ = new Uint8Array(Q);
    const usedT = new Uint8Array(T);
    for (const edge of edges) {
      if (usedQ[edge.qi] || usedT[edge.ti]) continue;
      usedQ[edge.qi] = 1;
      usedT[edge.ti] = 1;
      assignedTemplate[edge.qi] = edge.ti;
    }
  }

  const chainAlignments: ChainTemplateAlignment[] = [];
  const warnings: string[] = [];
  let matchedQueryResidues = 0;
  let totalQueryResidues = 0;
  let matchedChains = 0;
  let totalIdentical = 0;

  for (let qi = 0; qi < Q; qi += 1) {
    const q = normalizedQuery[qi];
    totalQueryResidues += q.sequence.length;
    const ti = assignedTemplate[qi];
    if (ti === null) {
      chainAlignments.push({
        queryChainIndex: q.origIndex,
        queryChainId: q.chainId,
        templateChainId: null,
        templateChainIndex: null,
        residueMap: new Map(),
        queryLength: q.sequence.length,
        templateLength: 0,
        alignedCount: 0,
        identicalCount: 0,
        identity: 0,
        coverage: 0,
      });
      if (Q > 1) {
        warnings.push(
          `Query chain ${q.chainId} (${q.sequence.length} aa) has no matching chain in the template and will remain uncovered.`,
        );
      }
      continue;
    }

    const edge = matrix[qi][ti];
    const t = parsed.chains[ti];
    matchedChains += 1;
    matchedQueryResidues += edge.dp.alignedCount;
    totalIdentical += edge.dp.identicalCount;
    chainAlignments.push({
      queryChainIndex: q.origIndex,
      queryChainId: q.chainId,
      templateChainId: t.chainId,
      templateChainIndex: ti,
      residueMap: edge.dp.residueMap,
      queryLength: q.sequence.length,
      templateLength: t.residues.length,
      alignedCount: edge.dp.alignedCount,
      identicalCount: edge.dp.identicalCount,
      identity: edge.dp.identity,
      coverage: edge.dp.coverage,
    });
  }

  if (matchedQueryResidues === 0) {
    if (bestRawIdenticalAcrossAll === 0) {
      throw new LocalFoldTemplateError(
        'ZERO_SEQUENCE_OVERLAP',
        'Template sequence has zero overlap with the query sequence(s). Verify that the template matches the target protein.',
        { bestIdenticalAcrossAll, bestRawIdenticalAcrossAll, bestIdentityAcrossAll },
      );
    }
    throw new LocalFoldTemplateError(
      'LOW_COVERAGE_OR_IDENTITY',
      `Template alignment failed minimum identity/overlap thresholds (best identity ${(bestIdentityAcrossAll * 100).toFixed(1)}%, ${bestIdenticalAcrossAll} identical residues).`,
      { bestIdenticalAcrossAll, bestRawIdenticalAcrossAll, bestIdentityAcrossAll, minIdentity },
    );
  }

  const overallIdentity = matchedQueryResidues > 0 ? totalIdentical / matchedQueryResidues : 0;
  const overallCoverage = totalQueryResidues > 0 ? matchedQueryResidues / totalQueryResidues : 0;

  if (opts?.minCoverage !== undefined && overallCoverage < opts.minCoverage) {
    throw new LocalFoldTemplateError(
      'LOW_COVERAGE_OR_IDENTITY',
      `Template covers ${(overallCoverage * 100).toFixed(1)}% of query residues, below the required ${(opts.minCoverage * 100).toFixed(1)}% minimum.`,
      { overallCoverage, minCoverage: opts.minCoverage },
    );
  }

  if (overallCoverage < 0.5) {
    warnings.push(
      `Partial template coverage: ${matchedQueryResidues}/${totalQueryResidues} query residues (${(overallCoverage * 100).toFixed(0)}%) matched.`,
    );
  }

  return {
    parsed,
    chainAlignments,
    matchedQueryResidues,
    totalQueryResidues,
    matchedChains,
    totalQueryChains: Q,
    overallIdentity,
    overallCoverage,
    warnings,
  };
}

// ---------------------------------------------------------------------------
// 5. Model-Family Featurization Builders (AF2 atom37 & AF3 dense-24)
// ---------------------------------------------------------------------------

export interface Af2TemplateObject {
  aatype: Int32Array;
  atomPositions: Float32Array;
  atomMask: Float32Array;
  spanChains: boolean;
}

export interface Af2TemplatePayload {
  template: Af2TemplateObject;
  asymId: Float32Array;
  multichainMask2d: Float32Array;
  coveredResidues: number;
  totalResidues: number;
}

function aatypeForOneLetter(code: string): number {
  const idx = AATYPE_INDEX_BY_CHAR[code];
  return idx !== undefined ? idx : UNK_AATYPE;
}

export function buildAf2TemplatePayload(
  queryChains: ReadonlyArray<QueryChainSpec | string>,
  alignment: TemplateAlignmentResult,
  _family: 'af2-monomer' | 'af2-multimer',
  opts?: { spanChains?: boolean; minConfidence?: number },
): Af2TemplatePayload {
  const normalizedLengths = queryChains.map((c) =>
    typeof c === 'string'
      ? c.trim().replace(/[^A-Za-z]/g, '').length
      : c.sequence.trim().replace(/[^A-Za-z]/g, '').length,
  );
  const totalResidues = normalizedLengths.reduce((s, n) => s + n, 0);
  const spanChains = opts?.spanChains ?? true;
  const minConfidence = opts?.minConfidence ?? 0;

  const aatype = new Int32Array(totalResidues).fill(GAP_AATYPE);
  const atomPositions = new Float32Array(totalResidues * 37 * 3);
  const atomMask = new Float32Array(totalResidues * 37);
  const asymId = new Float32Array(totalResidues);
  const coverage = new Float32Array(totalResidues);

  let chainOffset = 0;
  let coveredResidues = 0;

  for (let qi = 0; qi < normalizedLengths.length; qi += 1) {
    const qLen = normalizedLengths[qi];
    for (let k = 0; k < qLen; k += 1) {
      asymId[chainOffset + k] = qi;
    }

    const chainAlign = alignment.chainAlignments.find((a) => a.queryChainIndex === qi);
    if (chainAlign && chainAlign.templateChainIndex !== null) {
      const tChain = alignment.parsed.chains[chainAlign.templateChainIndex];
      if (tChain) {
        for (const [qLocal, tLocal] of chainAlign.residueMap) {
          if (qLocal < 0 || qLocal >= qLen) continue;
          const tRes = tChain.residues[tLocal];
          if (!tRes) continue;
          if (minConfidence > 0 && tRes.confidence < minConfidence) continue;

          const globalIdx = chainOffset + qLocal;
          aatype[globalIdx] = aatypeForOneLetter(tRes.code);
          let hasAnyAtom = false;
          for (let slot = 0; slot < AF2_ATOM37_NAMES.length; slot += 1) {
            const pt = tRes.atoms.get(AF2_ATOM37_NAMES[slot]);
            if (!pt) continue;
            const base = (globalIdx * 37 + slot) * 3;
            atomPositions[base] = pt[0];
            atomPositions[base + 1] = pt[1];
            atomPositions[base + 2] = pt[2];
            atomMask[globalIdx * 37 + slot] = 1;
            hasAnyAtom = true;
          }
          if (hasAnyAtom) {
            coverage[globalIdx] = 1;
            coveredResidues += 1;
          }
        }
      }
    }

    chainOffset += qLen;
  }

  const multichainMask2d = new Float32Array(totalResidues * totalResidues);
  for (let i = 0; i < totalResidues; i += 1) {
    for (let j = 0; j < totalResidues; j += 1) {
      const same = asymId[i] === asymId[j];
      const spanned = spanChains && coverage[i] > 0 && coverage[j] > 0;
      multichainMask2d[i * totalResidues + j] = same || spanned ? 1 : 0;
    }
  }

  return {
    template: {
      aatype,
      atomPositions,
      atomMask,
      spanChains,
    },
    asymId,
    multichainMask2d,
    coveredResidues,
    totalResidues,
  };
}

export interface Af3TemplateSlot {
  aatype: Int32Array;
  atomPositions: Float32Array;
  atomMask: Float32Array;
  covered: number;
  atoms: number;
  spanChains: boolean;
}

export interface Af3BatchLayout {
  tokens: number;
  residueOfToken?: ArrayLike<number>;
  tokenOf?: (globalProteinResidueIndex: number) => number;
}

export function buildAf3TemplateSlots(
  batchLayout: Af3BatchLayout | number,
  alignment: TemplateAlignmentResult,
  modelIdOrDialect: string,
  opts?: { spanChains?: boolean; minConfidence?: number },
): Af3TemplateSlot[] {
  const tokens = typeof batchLayout === 'number' ? batchLayout : batchLayout.tokens;
  const spanChains = opts?.spanChains ?? true;
  const minConfidence = opts?.minConfidence ?? 0;
  const policy = getLocalFoldTemplateDialectPolicy(modelIdOrDialect);

  let tokenForResidue: (globalResIdx: number) => number;
  if (typeof batchLayout !== 'number' && batchLayout.tokenOf) {
    tokenForResidue = batchLayout.tokenOf;
  } else if (typeof batchLayout !== 'number' && batchLayout.residueOfToken) {
    const firstTokenByResidue = new Map<number, number>();
    const rot = batchLayout.residueOfToken;
    for (let t = 0; t < rot.length; t += 1) {
      const r = rot[t];
      if (r >= 0 && !firstTokenByResidue.has(r)) {
        firstTokenByResidue.set(r, t);
      }
    }
    tokenForResidue = (r) => firstTokenByResidue.get(r) ?? -1;
  } else {
    tokenForResidue = (r) => r;
  }

  const aatype = new Int32Array(tokens).fill(GAP_AATYPE);
  const atomPositions = new Float32Array(tokens * NUM_AF3_DENSE * 3);
  const atomMask = new Float32Array(tokens * NUM_AF3_DENSE);
  let covered = 0;
  let atoms = 0;

  let chainOffset = 0;
  for (const chainAlign of alignment.chainAlignments) {
    if (chainAlign.templateChainIndex !== null) {
      const tChain = alignment.parsed.chains[chainAlign.templateChainIndex];
      if (tChain) {
        for (const [qLocal, tLocal] of chainAlign.residueMap) {
          if (qLocal < 0 || qLocal >= chainAlign.queryLength) continue;
          const tRes = tChain.residues[tLocal];
          if (!tRes) continue;
          if (minConfidence > 0 && tRes.confidence < minConfidence) continue;

          const token = tokenForResidue(chainOffset + qLocal);
          if (token < 0 || token >= tokens) continue;

          const rawName = tRes.resName;
          const nonStandard = !tRes.isStandard;

          if (
            nonStandard &&
            policy.modified === 'unk_n' &&
            rawName !== 'MSE' &&
            rawName !== 'UNK'
          ) {
            const nPt = tRes.atoms.get('N');
            aatype[token] = UNK_AATYPE;
            if (nPt) {
              for (let slot = 0; slot < NUM_AF3_DENSE; slot += 1) {
                const base = (token * NUM_AF3_DENSE + slot) * 3;
                atomPositions[base] = nPt[0];
                atomPositions[base + 1] = nPt[1];
                atomPositions[base + 2] = nPt[2];
                atomMask[token * NUM_AF3_DENSE + slot] = 1;
              }
              atoms += 1;
            }
            covered += 1;
            continue;
          }

          if (nonStandard && policy.modified === 'unk_restype') {
            const reqNames = ['N', 'CA', 'C', 'CB'] as const;
            const pts = reqNames.map((n) => tRes.atoms.get(n));
            if (pts.some((p) => p === undefined)) continue;
            aatype[token] = UNK_AATYPE;
            const denseSlots = [0, 1, 2, 4] as const;
            for (let k = 0; k < denseSlots.length; k += 1) {
              const slot = denseSlots[k];
              const pt = pts[k]!;
              const base = (token * NUM_AF3_DENSE + slot) * 3;
              atomPositions[base] = pt[0];
              atomPositions[base + 1] = pt[1];
              atomPositions[base + 2] = pt[2];
              atomMask[token * NUM_AF3_DENSE + slot] = 1;
            }
            atoms += 4;
            covered += 1;
            continue;
          }

          aatype[token] = aatypeForOneLetter(tRes.code);
          const layout = AF3_DENSE_INTERNAL_ATOMS[tRes.code] ?? AF3_DENSE_INTERNAL_ATOMS.X;
          for (let slot = 0; slot < layout.length && slot < NUM_AF3_DENSE; slot += 1) {
            const pt = tRes.atoms.get(layout[slot]);
            if (!pt) continue;
            const base = (token * NUM_AF3_DENSE + slot) * 3;
            atomPositions[base] = pt[0];
            atomPositions[base + 1] = pt[1];
            atomPositions[base + 2] = pt[2];
            atomMask[token * NUM_AF3_DENSE + slot] = 1;
            atoms += 1;
          }
          covered += 1;
        }
      }
    }
    chainOffset += chainAlign.queryLength;
  }

  return [
    {
      aatype,
      atomPositions,
      atomMask,
      covered,
      atoms,
      spanChains,
    },
  ];
}

export { AF2_ATOM37_SLOT_BY_NAME };
