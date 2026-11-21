/**
 * PDB and mmCIF serialization utilities for Genie 3 generated structures.
 *
 * Compatible with Mol*, ProteinMPNN, LocalFold, and upstream Genie 3 `save_np_features_to_pdb`.
 */

import {
  NUM_RESTYPES,
  PROTEIN_RESTYPES,
  PROTEIN_RESTYPES_1,
  RESTYPE_1_TO_3,
} from './constants.js';
import { centerCaCoords, reconstructBackboneFromCa } from './backbone.js';

const BACKBONE4_NAMES = ['N', 'CA', 'C', 'O'];

function padL(s, n) {
  return s.length >= n ? s : ' '.repeat(n - s.length) + s;
}

function padR(s, n) {
  return s.length >= n ? s : s + ' '.repeat(n - s.length);
}

/**
 * Decode a 1-letter amino acid sequence string from Genie 3 `ai` logits or one-hot `[L * 20]`.
 *
 * @param {ArrayLike<number>} logitsOrOneHot Flat `[L * 20]`
 * @param {number} L
 * @param {{ defaultAa?: string }} [opts]
 * @returns {string}
 */
export function sequenceFromRestype(logitsOrOneHot, L, opts = {}) {
  const defaultAa = opts.defaultAa ?? 'A';
  let seq = '';
  for (let i = 0; i < L; i += 1) {
    const base = i * NUM_RESTYPES;
    let bestIdx = -1;
    let bestVal = -Infinity;
    let absSum = 0;
    for (let c = 0; c < NUM_RESTYPES; c += 1) {
      const v = logitsOrOneHot[base + c];
      absSum += Math.abs(v);
      if (v > bestVal) {
        bestVal = v;
        bestIdx = c;
      }
    }
    if (absSum === 0 || bestIdx < 0) {
      seq += defaultAa;
    } else {
      seq += PROTEIN_RESTYPES_1[bestIdx];
    }
  }
  return seq;
}

/**
 * Convert 4-atom backbone coordinates (`[L * 4 * 3]` in `N, CA, C, O` order) into a PDB string.
 *
 * @param {Float32Array} backbone4Coords Flat `[L * 4 * 3]`
 * @param {number} L Number of residues
 * @param {{
 *   sequence?: string | null,
 *   defaultResname?: string,
 *   chainId?: string,
 *   chainIds?: ArrayLike<string | number> | null,
 *   residueIndices?: ArrayLike<number> | null,
 *   bFactors?: ArrayLike<number> | null,
 *   caOnly?: boolean,
 * }} [opts]
 * @returns {string}
 */
export function backboneToPdb(backbone4Coords, L, opts = {}) {
  const defaultResname = opts.defaultResname ?? 'ALA';
  const defaultChain = (opts.chainId ?? 'A')[0] ?? 'A';
  const caOnly = opts.caOnly === true;
  const atomSlots = caOnly ? [1] : [0, 1, 2, 3];

  const lines = [];
  let atomIdx = 1;
  let prevChain = null;
  let prevResName = defaultResname;
  let prevResSeq = 1;

  for (let r = 0; r < L; r += 1) {
    let chain = defaultChain;
    if (opts.chainIds) {
      const raw = opts.chainIds[r];
      chain = typeof raw === 'number' ? String.fromCharCode(65 + raw) : String(raw)[0] ?? defaultChain;
    }

    if (prevChain !== null && chain !== prevChain) {
      lines.push(`TER   ${padL(String(atomIdx), 5)}      ${prevResName} ${prevChain}${padL(String(prevResSeq), 4)}`);
      atomIdx += 1;
    }

    const aa1 = opts.sequence?.[r]?.toUpperCase();
    const resName = aa1 && RESTYPE_1_TO_3[aa1] ? RESTYPE_1_TO_3[aa1] : defaultResname;
    const resSeq = opts.residueIndices ? opts.residueIndices[r] : r + 1;
    const bVal = opts.bFactors ? opts.bFactors[r] : 0.0;

    for (const slot of atomSlots) {
      const off = (r * 4 + slot) * 3;
      const x = backbone4Coords[off + 0];
      const y = backbone4Coords[off + 1];
      const z = backbone4Coords[off + 2];
      const name = BACKBONE4_NAMES[slot];
      const elem = name[0];
      lines.push(
        `ATOM  ${padL(String(atomIdx), 5)}  ${padR(name, 3)} ${resName} ${chain}${padL(String(resSeq), 4)}    ` +
          `${padL(x.toFixed(3), 8)}${padL(y.toFixed(3), 8)}${padL(z.toFixed(3), 8)}` +
          `  1.00${padL(bVal.toFixed(2), 6)}           ${elem}  `,
      );
      atomIdx += 1;
    }

    prevChain = chain;
    prevResName = resName;
    prevResSeq = resSeq;
  }

  lines.push('END');
  return `${lines.join('\n')}\n`;
}

/**
 * Convert `Cα` coordinates (`[L * 3]`) directly into a PDB string.
 * By default reconstructs ideal `N, CA, C, O` backbone atoms and centers on `Cα` centroid.
 * Pass `{ caOnly: true }` for upstream Genie 3 Cα-only PDB format.
 *
 * @param {Float32Array} caCoords Flat `[L * 3]`
 * @param {number} [L]
 * @param {{
 *   sequence?: string | null,
 *   defaultResname?: string,
 *   chainId?: string,
 *   chainIds?: ArrayLike<string | number> | null,
 *   residueIndices?: ArrayLike<number> | null,
 *   bFactors?: ArrayLike<number> | null,
 *   caOnly?: boolean,
 *   center?: boolean,
 * }} [opts]
 * @returns {string}
 */
export function caToPdb(caCoords, L = Math.floor(caCoords.length / 3), opts = {}) {
  const centered = opts.center === false ? caCoords : centerCaCoords(caCoords);
  if (opts.caOnly) {
    const bb4 = new Float32Array(L * 4 * 3);
    for (let i = 0; i < L; i += 1) {
      bb4[(i * 4 + 1) * 3 + 0] = centered[i * 3 + 0];
      bb4[(i * 4 + 1) * 3 + 1] = centered[i * 3 + 1];
      bb4[(i * 4 + 1) * 3 + 2] = centered[i * 3 + 2];
    }
    return backboneToPdb(bb4, L, { ...opts, caOnly: true });
  }
  const bb4 = reconstructBackboneFromCa(centered, {
    chainIds: opts.chainIds,
    center: false,
  });
  return backboneToPdb(bb4, L, { ...opts, caOnly: false });
}

/**
 * Convert 4-atom backbone coordinates (`[L * 4 * 3]`) into a minimal, Mol*-compatible mmCIF string.
 *
 * @param {Float32Array} backbone4Coords Flat `[L * 4 * 3]`
 * @param {number} L
 * @param {{
 *   entryId?: string,
 *   sequence?: string | null,
 *   defaultResname?: string,
 *   chainId?: string,
 *   chainIds?: ArrayLike<string | number> | null,
 *   residueIndices?: ArrayLike<number> | null,
 *   bFactors?: ArrayLike<number> | null,
 *   caOnly?: boolean,
 * }} [opts]
 * @returns {string}
 */
export function backboneToMmcif(backbone4Coords, L, opts = {}) {
  const entryId = opts.entryId ?? 'GENIE3';
  const defaultResname = opts.defaultResname ?? 'ALA';
  const defaultChain = (opts.chainId ?? 'A')[0] ?? 'A';
  const atomSlots = opts.caOnly ? [1] : [0, 1, 2, 3];

  const lines = [
    `data_${entryId}`,
    '#',
    `_entry.id ${entryId}`,
    '#',
    'loop_',
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.type_symbol',
    '_atom_site.label_atom_id',
    '_atom_site.label_alt_id',
    '_atom_site.label_comp_id',
    '_atom_site.label_asym_id',
    '_atom_site.label_entity_id',
    '_atom_site.label_seq_id',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.auth_seq_id',
    '_atom_site.auth_asym_id',
    '_atom_site.pdbx_PDB_model_num',
  ];

  let atomIdx = 1;
  for (let r = 0; r < L; r += 1) {
    let chain = defaultChain;
    if (opts.chainIds) {
      const raw = opts.chainIds[r];
      chain = typeof raw === 'number' ? String.fromCharCode(65 + raw) : String(raw)[0] ?? defaultChain;
    }
    const aa1 = opts.sequence?.[r]?.toUpperCase();
    const resName = aa1 && RESTYPE_1_TO_3[aa1] ? RESTYPE_1_TO_3[aa1] : defaultResname;
    const resSeq = opts.residueIndices ? opts.residueIndices[r] : r + 1;
    const bVal = opts.bFactors ? opts.bFactors[r] : 0.0;

    for (const slot of atomSlots) {
      const off = (r * 4 + slot) * 3;
      const x = backbone4Coords[off + 0];
      const y = backbone4Coords[off + 1];
      const z = backbone4Coords[off + 2];
      const name = BACKBONE4_NAMES[slot];
      const elem = name[0];
      lines.push(
        `ATOM ${atomIdx} ${elem} ${name} . ${resName} ${chain} 1 ${resSeq} ` +
          `${x.toFixed(3)} ${y.toFixed(3)} ${z.toFixed(3)} 1.00 ${bVal.toFixed(2)} ${resSeq} ${chain} 1`,
      );
      atomIdx += 1;
    }
  }

  lines.push('#');
  return `${lines.join('\n')}\n`;
}

export { PROTEIN_RESTYPES };
