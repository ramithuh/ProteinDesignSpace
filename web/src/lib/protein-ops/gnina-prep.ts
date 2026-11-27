// Pure-TS protein–ligand preparation, pocket detection, and pose coordinate re-injection
// helpers for GNINA 1.3 WebGPU docking & affinity scoring.

import {
  classifyHetatmCompId,
  NUCLEIC_ACID_RESNAMES,
  STANDARD_AA_RESNAMES,
  tokenizeCifLine,
} from './hetatm-policy';
import { mmcifToPdb } from './pdb';

export type LigandBondSource =
  | 'chem_comp_bond'
  | 'conect_complete'
  | 'conect_partial_fallback'
  | 'distance_fallback';

export interface ChemCompBond {
  atom1: string;
  atom2: string;
  order: number;
  aromatic: boolean;
}

export interface CifLigandResidueMeta {
  fullCompId: string;
  authChain: string;
  labelChain: string;
  authSeq: string;
  labelSeq: string;
  insCode: string;
  heavyAtomNames: Set<string>;
  labelToAuthAtomName: Map<string, string>;
  authToLabelAtomName: Map<string, string>;
  formalChargesByAtomName: Map<string, number>;
  centroid: [number, number, number];
}

export interface CifLigandOverlay {
  componentBonds: Map<string, ChemCompBond[]>;
  cifResidues: CifLigandResidueMeta[];
}

export interface ParsedPdbAtom {
  lineIndex: number;
  record: 'ATOM' | 'HETATM';
  serial: number;
  name: string;
  altLoc: string;
  resName: string;
  chain: string;
  resSeq: string;
  insCode: string;
  x: number;
  y: number;
  z: number;
  occ: number;
  bFactor: number;
  element: string;
  charge: number;
}

export interface DetectedLigandInstance {
  /** Unique key e.g. `BTN:A:300` or `DHZ:A:301A`. */
  instanceKey: string;
  compId: string;
  chain: string;
  resSeq: string;
  insCode: string;
  heavyAtomCount: number;
  totalAtomCount: number;
  atoms: ParsedPdbAtom[];
  /** Line indices of heavy atoms in the normalized PDB text, ordered identically to `sdf` atoms. */
  heavyLineIndices: number[];
  /** All atom line indices in the normalized PDB text belonging to this ligand instance. */
  allLineIndices: number[];
}

export interface GninaComplexPrep {
  /** Normalized PDB text used for splitting & coordinate re-injection. */
  normalizedPdb: string;
  /** Receptor-only PDB text (protein polymer + modified AAs + optional metal cofactors). */
  receptorPdb: string;
  /** All detected `organic-ligand` HETATM instances sorted by heavy-atom count descending. */
  detectedLigands: DetectedLigandInstance[];
  /** Selected holo ligand instance (null if none detected or matched). */
  selectedLigand: DetectedLigandInstance | null;
  /** V2000 SDF text synthesized for `selectedLigand` (with inferred covalent bonds & orders). */
  selectedLigandSdf: string | null;
  /** Provenance of bond topology used to build `selectedLigandSdf`. */
  selectedLigandBondSource: LigandBondSource | null;
  /** Monoatomic metal ions kept in receptor. */
  keptMetals: string[];
  /** Distinct organic ligand compIds detected. */
  detectedLigandCodes: string[];
}

export interface PocketResidueEntry {
  /** Exact key used by vendor/gnina `typeMolecule`: `[chain, seq, ins, res].filter(Boolean).join('/')`. */
  gninaKey: string;
  /** Canonical denovo.run residue ID: `${chain}${seq}${ins}`. */
  residueId: string;
  chain: string;
  resSeq: string;
  insCode: string;
  resName: string;
  minDist: number;
}

const COVALENT_RADII: Record<string, number> = {
  H: 0.31,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  S: 1.05,
  P: 1.07,
  F: 0.57,
  Cl: 1.02,
  Br: 1.2,
  I: 1.39,
  B: 0.85,
  Se: 1.2,
  Si: 1.11,
  As: 1.19,
  Fe: 1.32,
  Zn: 1.22,
  Mg: 1.41,
  Ca: 1.76,
  Mn: 1.39,
};

const MAX_BONDS: Record<string, number> = {
  H: 1,
  F: 1,
  Cl: 1,
  Br: 1,
  I: 1,
  O: 2,
  Se: 2,
  As: 3,
  C: 4,
  N: 4,
  B: 4,
  Si: 4,
  S: 6,
  P: 6,
};

const METALS_SET = new Set(
  'Li Na K Rb Cs Be Mg Ca Sr Ba Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Mo Ru Rh Pd Ag Cd Sn W Re Os Ir Pt Au Hg Al Ga In Bi'.split(
    ' ',
  ),
);

function normalizeElement(raw: string): string {
  const s = raw.trim().toLowerCase();
  if (!s) return 'C';
  if (s === 'd') return 'H';
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function inferPdbAtomElement(nameCol: string, elemCol: string): string {
  if (elemCol.trim()) return normalizeElement(elemCol);
  if (nameCol[0] === ' ' || /^\d/.test(nameCol)) {
    const stripped = nameCol.replace(/[\d ]/g, '');
    return normalizeElement(stripped[0] ?? 'C');
  }
  const pair = normalizeElement(nameCol.slice(0, 2));
  return pair in COVALENT_RADII || METALS_SET.has(pair) ? pair : normalizeElement(nameCol[0] ?? 'C');
}

function parsePdbCharge(raw: string): number {
  const s = raw.trim().replace(/^(\d)([+-])$/, '$2$1');
  return Number(s) || 0;
}

function isMmcifText(rawText: string): boolean {
  const trimmed = rawText.trimStart();
  return trimmed.startsWith('data_') || rawText.includes('_atom_site.');
}

function normalizeToPdbText(rawText: string): string {
  if (isMmcifText(rawText)) {
    return mmcifToPdb(rawText, { firstModelOnly: true });
  }
  return rawText;
}

function mapCifBondOrder(rawOrder: string, rawArom: string): { order: number; aromatic: boolean } {
  const ord = rawOrder.trim().toUpperCase();
  const aromFlag = rawArom.trim().toUpperCase() === 'Y';
  if (ord === 'AROM' || ord === 'DELO' || ord === 'PI') {
    return { order: 4, aromatic: true };
  }
  if (ord === 'DOUB') return { order: 2, aromatic: aromFlag };
  if (ord === 'TRIP' || ord === 'QUAD') return { order: 3, aromatic: aromFlag };
  return { order: 1, aromatic: aromFlag };
}

/**
 * Quote-aware, `#`-safe token-stream parser for mmCIF `_chem_comp_bond` records.
 * Supports `loop_` tables (arbitrary column order, multi-line wrapped rows, `# #` terminators)
 * and single-row `_chem_comp_bond.<col> <val>` blocks.
 */
export function parseChemCompBonds(cifText: string): Map<string, ChemCompBond[]> {
  const out = new Map<string, ChemCompBond[]>();
  if (!cifText || !cifText.includes('_chem_comp_bond.')) return out;

  const lines = cifText.split(/\r?\n/);
  const pushBond = (compId: string, a1: string, a2: string, rawOrder: string, rawArom: string) => {
    const cid = compId.trim().toUpperCase();
    const atom1 = a1.trim();
    const atom2 = a2.trim();
    if (!cid || !atom1 || !atom2 || atom1 === atom2 || cid === '.' || cid === '?') return;
    const { order, aromatic } = mapCifBondOrder(rawOrder, rawArom);
    let list = out.get(cid);
    if (!list) {
      list = [];
      out.set(cid, list);
    }
    list.push({ atom1, atom2, order, aromatic });
  };

  let inSemicolon = false;
  let inLoop = false;
  let loopCols: string[] = [];
  let loopTokens: string[] = [];
  let singleRow: Record<string, string> | null = null;

  const flushLoop = () => {
    if (!inLoop) return;
    const iComp = loopCols.indexOf('comp_id');
    const iA1 = loopCols.indexOf('atom_id_1');
    const iA2 = loopCols.indexOf('atom_id_2');
    const iOrd = loopCols.indexOf('value_order');
    const iArom = loopCols.indexOf('pdbx_aromatic_flag');
    const nCols = loopCols.length;
    if (nCols > 0 && iComp >= 0 && iA1 >= 0 && iA2 >= 0) {
      for (let off = 0; off + nCols <= loopTokens.length; off += nCols) {
        pushBond(
          loopTokens[off + iComp],
          loopTokens[off + iA1],
          loopTokens[off + iA2],
          iOrd >= 0 ? loopTokens[off + iOrd] : 'SING',
          iArom >= 0 ? loopTokens[off + iArom] : 'N',
        );
      }
    }
    inLoop = false;
    loopCols = [];
    loopTokens = [];
  };

  const flushSingle = () => {
    if (!singleRow) return;
    if (singleRow.comp_id && singleRow.atom_id_1 && singleRow.atom_id_2) {
      pushBond(
        singleRow.comp_id,
        singleRow.atom_id_1,
        singleRow.atom_id_2,
        singleRow.value_order ?? 'SING',
        singleRow.pdbx_aromatic_flag ?? 'N',
      );
    }
    singleRow = null;
  };

  let pendingLoop = false;
  for (const raw of lines) {
    if (inSemicolon) {
      if (raw.startsWith(';')) inSemicolon = false;
      continue;
    }
    if (raw.startsWith(';')) {
      inSemicolon = true;
      continue;
    }
    const tr = raw.trim();
    if (!tr || tr.startsWith('#')) {
      if (inLoop && loopTokens.length > 0) flushLoop();
      if (singleRow) flushSingle();
      pendingLoop = false;
      continue;
    }
    if (tr === 'loop_' || /^loop_\s/i.test(tr)) {
      flushLoop();
      flushSingle();
      pendingLoop = true;
      continue;
    }
    if (/^(?:data_|save_|stop_|global_)/i.test(tr)) {
      flushLoop();
      flushSingle();
      pendingLoop = false;
      continue;
    }
    if (tr.startsWith('_chem_comp_bond.')) {
      const toks = tokenizeCifLine(tr);
      const colName = toks[0].slice('_chem_comp_bond.'.length);
      if (pendingLoop || inLoop) {
        if (!inLoop) {
          inLoop = true;
          loopCols = [];
          loopTokens = [];
        }
        pendingLoop = false;
        loopCols.push(colName);
      } else {
        if (!singleRow) singleRow = {};
        if (toks.length >= 2) {
          singleRow[colName] = toks[1];
        }
      }
      continue;
    }
    if (tr.startsWith('_')) {
      flushLoop();
      flushSingle();
      pendingLoop = false;
      continue;
    }
    if (inLoop) {
      const toks = tokenizeCifLine(tr);
      for (const t of toks) loopTokens.push(t);
    }
  }
  flushLoop();
  flushSingle();
  return out;
}

/**
 * Parse `_chem_comp_bond` and model-1 HETATM metadata (full 4–5 char `comp_id`,
 * `pdbx_formal_charge`, `label_atom_id`↔`auth_atom_id`, centroid) from mmCIF text.
 */
export function parseCifLigandOverlay(cifText: string): CifLigandOverlay {
  const componentBonds = parseChemCompBonds(cifText);
  const cifResidues: CifLigandResidueMeta[] = [];
  if (!cifText || !cifText.includes('_atom_site.')) {
    return { componentBonds, cifResidues };
  }

  const lines = cifText.split(/\r?\n/);
  const cols: string[] = [];
  let inAtomSite = false;
  let seenRow = false;
  let inSemicolon = false;
  let colReady = false;
  let iGroup = -1,
    iModel = -1,
    iAlt = -1,
    iLabelAtom = -1,
    iAuthAtom = -1,
    iAuthAsym = -1,
    iLabelAsym = -1,
    iAuthSeq = -1,
    iLabelSeq = -1,
    iIns = -1,
    iLabelComp = -1,
    iAuthComp = -1,
    iX = -1,
    iY = -1,
    iZ = -1,
    iElem = -1,
    iCharge = -1;

  let firstModelNum: string | null = null;
  interface BucketAcc {
    fullCompId: string;
    authChain: string;
    labelChain: string;
    authSeq: string;
    labelSeq: string;
    insCode: string;
    heavyAtomNames: Set<string>;
    labelToAuthAtomName: Map<string, string>;
    authToLabelAtomName: Map<string, string>;
    formalChargesByAtomName: Map<string, number>;
    seenAtomPrio: Map<string, number>;
    sumX: number;
    sumY: number;
    sumZ: number;
    heavyCount: number;
  }
  const buckets = new Map<string, BucketAcc>();

  for (const raw of lines) {
    if (!inAtomSite) {
      if (inSemicolon) {
        if (raw.startsWith(';')) inSemicolon = false;
        continue;
      }
      if (raw.startsWith(';')) {
        inSemicolon = true;
        continue;
      }
    }
    const tr = raw.trim();
    if (tr.startsWith('_atom_site.')) {
      inAtomSite = true;
      colReady = false;
      cols.push(tr.slice('_atom_site.'.length).split(/\s+/)[0]);
      continue;
    }
    if (
      inAtomSite &&
      (tr === '' ||
        tr.startsWith('#') ||
        tr.startsWith('loop_') ||
        tr.startsWith('_') ||
        /^(?:data_|save_|stop_|global_)/i.test(tr))
    ) {
      if (seenRow) break;
      continue;
    }
    if (!inAtomSite) continue;
    const toks = tokenizeCifLine(tr);
    if (toks.length < cols.length) continue;
    seenRow = true;

    if (!colReady) {
      iGroup = cols.indexOf('group_PDB');
      iModel = cols.indexOf('pdbx_PDB_model_num');
      iAlt = cols.indexOf('label_alt_id');
      iLabelAtom = cols.indexOf('label_atom_id');
      iAuthAtom = cols.indexOf('auth_atom_id');
      iAuthAsym = cols.indexOf('auth_asym_id');
      iLabelAsym = cols.indexOf('label_asym_id');
      iAuthSeq = cols.indexOf('auth_seq_id');
      iLabelSeq = cols.indexOf('label_seq_id');
      iIns = cols.indexOf('pdbx_PDB_ins_code');
      iLabelComp = cols.indexOf('label_comp_id');
      iAuthComp = cols.indexOf('auth_comp_id');
      iX = cols.indexOf('Cartn_x');
      iY = cols.indexOf('Cartn_y');
      iZ = cols.indexOf('Cartn_z');
      iElem = cols.indexOf('type_symbol');
      iCharge = cols.indexOf('pdbx_formal_charge');
      colReady = true;
    }

    const cifVal = (idx: number): string => {
      const v = idx >= 0 ? toks[idx] : '';
      return v === '.' || v === '?' ? '' : v;
    };

    const modelNum = cifVal(iModel) || '1';
    if (firstModelNum === null) firstModelNum = modelNum;
    if (modelNum !== firstModelNum) continue;

    const grp = (cifVal(iGroup) || 'ATOM').toUpperCase();
    if (grp !== 'HETATM') continue;

    const fullCompId = (cifVal(iLabelComp) || cifVal(iAuthComp) || '').toUpperCase();
    if (!fullCompId || STANDARD_AA_RESNAMES.has(fullCompId) || NUCLEIC_ACID_RESNAMES.has(fullCompId)) {
      continue;
    }
    const alt = cifVal(iAlt);
    const prio = !alt ? 0 : alt === 'A' ? 1 : 2;
    if (prio > 1) continue;

    const authChain = cifVal(iAuthAsym) || cifVal(iLabelAsym) || 'A';
    const labelChain = cifVal(iLabelAsym) || authChain;
    const authSeq = cifVal(iAuthSeq) || cifVal(iLabelSeq);
    const labelSeq = cifVal(iLabelSeq) || authSeq;
    const insCode = cifVal(iIns).slice(0, 1);
    const labelAtom = cifVal(iLabelAtom);
    const authAtom = cifVal(iAuthAtom) || labelAtom;
    const primaryAtom = authAtom || labelAtom;
    if (!primaryAtom) continue;

    const x = iX >= 0 ? parseFloat(toks[iX]) : NaN;
    const y = iY >= 0 ? parseFloat(toks[iY]) : NaN;
    const z = iZ >= 0 ? parseFloat(toks[iZ]) : NaN;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    const elem = normalizeElement(cifVal(iElem) || primaryAtom[0] || 'C');
    const chargeRaw = iCharge >= 0 ? parseInt(cifVal(iCharge), 10) : 0;
    const charge = Number.isFinite(chargeRaw) ? chargeRaw : 0;

    const bKey = `${fullCompId}|${authChain}|${labelChain}|${authSeq}|${labelSeq}|${insCode}`;
    let acc = buckets.get(bKey);
    if (!acc) {
      acc = {
        fullCompId,
        authChain,
        labelChain,
        authSeq,
        labelSeq,
        insCode,
        heavyAtomNames: new Set<string>(),
        labelToAuthAtomName: new Map<string, string>(),
        authToLabelAtomName: new Map<string, string>(),
        formalChargesByAtomName: new Map<string, number>(),
        seenAtomPrio: new Map<string, number>(),
        sumX: 0,
        sumY: 0,
        sumZ: 0,
        heavyCount: 0,
      };
      buckets.set(bKey, acc);
    }

    const prevPrio = acc.seenAtomPrio.get(primaryAtom);
    if (prevPrio !== undefined && prio >= prevPrio) continue;
    acc.seenAtomPrio.set(primaryAtom, prio);

    if (labelAtom && authAtom) {
      acc.labelToAuthAtomName.set(labelAtom, authAtom);
      acc.authToLabelAtomName.set(authAtom, labelAtom);
    }
    if (charge !== 0) {
      if (authAtom) acc.formalChargesByAtomName.set(authAtom, charge);
      if (labelAtom) acc.formalChargesByAtomName.set(labelAtom, charge);
    }
    if (elem !== 'H') {
      if (authAtom) acc.heavyAtomNames.add(authAtom);
      if (labelAtom) acc.heavyAtomNames.add(labelAtom);
      if (prevPrio === undefined) {
        acc.sumX += x;
        acc.sumY += y;
        acc.sumZ += z;
        acc.heavyCount++;
      }
    }
  }

  for (const acc of buckets.values()) {
    if (acc.heavyCount === 0) continue;
    cifResidues.push({
      fullCompId: acc.fullCompId,
      authChain: acc.authChain,
      labelChain: acc.labelChain,
      authSeq: acc.authSeq,
      labelSeq: acc.labelSeq,
      insCode: acc.insCode,
      heavyAtomNames: acc.heavyAtomNames,
      labelToAuthAtomName: acc.labelToAuthAtomName,
      authToLabelAtomName: acc.authToLabelAtomName,
      formalChargesByAtomName: acc.formalChargesByAtomName,
      centroid: [acc.sumX / acc.heavyCount, acc.sumY / acc.heavyCount, acc.sumZ / acc.heavyCount],
    });
  }

  return { componentBonds, cifResidues };
}

function parseAllPdbAtomLines(pdbText: string): {
  lines: string[];
  atoms: ParsedPdbAtom[];
  conectPairs: Array<[number, number]>;
} {
  const lines = pdbText.split(/\r?\n/);
  const atoms: ParsedPdbAtom[] = [];
  const conectPairs: Array<[number, number]> = [];
  const seenAtomKey = new Map<string, number>();
  let pastFirstModel = false;

  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (l.startsWith('CONECT')) {
      const serials = l
        .slice(6)
        .match(/.{1,5}/g)
        ?.map((s) => Number(s.trim()))
        .filter((n) => Number.isFinite(n) && n > 0) ?? [];
      for (let k = 1; k < serials.length; k++) {
        conectPairs.push([serials[0], serials[k]]);
      }
      continue;
    }
    if (l.startsWith('ENDMDL') && atoms.length > 0) {
      pastFirstModel = true;
      continue;
    }
    if (pastFirstModel) continue;
    const isAtom = l.startsWith('ATOM  ');
    const isHet = l.startsWith('HETATM');
    if ((!isAtom && !isHet) || l.length < 54) continue;

    const x = parseFloat(l.slice(30, 38));
    const y = parseFloat(l.slice(38, 46));
    const z = parseFloat(l.slice(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    const nameCol = l.slice(12, 16);
    const altLoc = (l[16] ?? ' ').trim();
    const resName = l.slice(17, 20).trim().toUpperCase();
    const chain = (l[21] ?? ' ').trim() || 'A';
    const resSeq = l.slice(22, 26).trim();
    const insCode = (l[26] ?? ' ').trim();
    const occRaw = l.length >= 60 ? parseFloat(l.slice(54, 60)) : 1;
    const occ = Number.isFinite(occRaw) ? occRaw : 1;
    const bRaw = l.length >= 66 ? parseFloat(l.slice(60, 66)) : 0;
    const bFactor = Number.isFinite(bRaw) ? bRaw : 0;
    const elemCol = l.length >= 78 ? l.slice(76, 78) : '';
    const chargeCol = l.length >= 80 ? l.slice(78, 80) : '';

    const parsed: ParsedPdbAtom = {
      lineIndex: i,
      record: isAtom ? 'ATOM' : 'HETATM',
      serial: Number(l.slice(6, 11).trim()) || 0,
      name: nameCol.trim(),
      altLoc,
      resName,
      chain,
      resSeq,
      insCode,
      x,
      y,
      z,
      occ,
      bFactor,
      element: inferPdbAtomElement(nameCol, elemCol),
      charge: parsePdbCharge(chargeCol),
    };

    const dedupKey = `${parsed.chain}/${parsed.resSeq}/${parsed.insCode}/${parsed.resName}/${parsed.name}`;
    const prevIdx = seenAtomKey.get(dedupKey);
    if (prevIdx !== undefined) {
      const prev = atoms[prevIdx];
      if (parsed.occ > prev.occ || (parsed.occ === prev.occ && parsed.altLoc < prev.altLoc)) {
        atoms[prevIdx] = parsed;
      }
    } else {
      seenAtomKey.set(dedupKey, atoms.length);
      atoms.push(parsed);
    }
  }

  return { lines, atoms, conectPairs };
}

function linearQuantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 1.0;
  if (sorted.length === 1) return sorted[0];
  const idx = p * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

function bondAngleDeg(a: ParsedPdbAtom, center: ParsedPdbAtom, b: ParsedPdbAtom): number {
  const ux = a.x - center.x,
    uy = a.y - center.y,
    uz = a.z - center.z;
  const vx = b.x - center.x,
    vy = b.y - center.y,
    vz = b.z - center.z;
  const lu = Math.hypot(ux, uy, uz);
  const lv = Math.hypot(vx, vy, vz);
  if (lu < 1e-6 || lv < 1e-6) return 0;
  const cos = Math.max(-1, Math.min(1, (ux * vx + uy * vy + uz * vz) / (lu * lv)));
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * Exact mirror of `vendor/gnina/shared/structure.js:151-170` (`aromaticRings`) so Tier 1
 * can verify which `pdbx_aromatic_flag='Y'` cycles GNINA already detects under Kekulé orders.
 */
function detectGninaAromaticAtoms(
  ordered: ReadonlyArray<ParsedPdbAtom>,
  bonds: ReadonlyArray<[number, number, number]>,
): Set<number> {
  const n = ordered.length;
  const adj: Array<Array<{ j: number; order: number }>> = Array.from({ length: n }, () => []);
  for (const [u, v, order] of bonds) {
    adj[u].push({ j: v, order });
    adj[v].push({ j: u, order });
  }
  const aromatic = new Set<number>();
  for (let start = 0; start < n; start++) {
    if (!adj[start].some((b) => b.order >= 2)) continue;
    const stack: Array<{ path: number[]; orders: number[] }> = [{ path: [start], orders: [] }];
    while (stack.length > 0) {
      const { path, orders } = stack.pop()!;
      const cur = path[path.length - 1];
      for (const { j, order } of adj[cur]) {
        if (j === start && (path.length === 5 || path.length === 6)) {
          const allOrd = [...orders, order];
          const all4 = allOrd.every((o) => o === 4);
          const doubles = allOrd.filter((o) => o === 2).length;
          const kek6 = path.length === 6 && doubles === 3;
          const kek5 =
            path.length === 5 &&
            doubles === 2 &&
            path.some((k) => ordered[k].element === 'N' || ordered[k].element === 'O' || ordered[k].element === 'S');
          if (all4 || kek6 || kek5) {
            for (const k of path) aromatic.add(k);
          }
        } else if (j > start && path.length < 6 && !path.includes(j)) {
          stack.push({ path: [...path, j], orders: [...orders, order] });
        }
      }
    }
  }
  return aromatic;
}

function findCyclesOfLength5Or6(
  n: number,
  edgePairs: ReadonlyArray<[number, number]>,
): Array<{ nodes: number[]; edgeIndices: number[] }> {
  const adj: Array<Array<{ j: number; ei: number }>> = Array.from({ length: n }, () => []);
  edgePairs.forEach(([u, v], ei) => {
    adj[u].push({ j: v, ei });
    adj[v].push({ j: u, ei });
  });
  const seen = new Set<string>();
  const cycles: Array<{ nodes: number[]; edgeIndices: number[] }> = [];
  for (let start = 0; start < n; start++) {
    const walk = (path: number[], ePath: number[]) => {
      const cur = path[path.length - 1];
      for (const { j, ei } of adj[cur]) {
        if (j === start && (path.length === 5 || path.length === 6)) {
          const sig = [...path].sort((a, b) => a - b).join(',');
          if (seen.has(sig)) continue;
          seen.add(sig);
          cycles.push({ nodes: path, edgeIndices: [...ePath, ei] });
        } else if (j > start && path.length < 6 && !path.includes(j)) {
          walk([...path, j], [...ePath, ei]);
        }
      }
    };
    walk([start], []);
  }
  return cycles;
}

function countConnectedComponents(nodes: ReadonlyArray<number>, edges: ReadonlyArray<[number, number]>): number {
  if (nodes.length === 0) return 0;
  const adj = new Map<number, number[]>();
  for (const u of nodes) adj.set(u, []);
  for (const [u, v] of edges) {
    adj.get(u)?.push(v);
    adj.get(v)?.push(u);
  }
  const visited = new Set<number>();
  let comps = 0;
  for (const start of nodes) {
    if (visited.has(start)) continue;
    comps++;
    const q = [start];
    visited.add(start);
    while (q.length > 0) {
      const cur = q.pop()!;
      for (const nxt of adj.get(cur) ?? []) {
        if (!visited.has(nxt)) {
          visited.add(nxt);
          q.push(nxt);
        }
      }
    }
  }
  return comps;
}

export interface BuildLigandSdfOptions {
  templateBonds?: ReadonlyArray<ChemCompBond>;
  atomNameAliases?: ReadonlyMap<string, string>;
}

/**
 * Synthesize a V2000 SDF block for a set of PDB ligand atoms using a strict 3-tier bond
 * precedence:
 *   1. Inline `_chem_comp_bond` template (validated against geometry & atom names)
 *   2. Explicit PDB `CONECT` records (`conect_complete` or `conect_partial_fallback`)
 *   3. Hardened, scale-normalized geometric bond & bond-order perception (`distance_fallback`)
 */
export function buildLigandV2000Sdf(
  atoms: ParsedPdbAtom[],
  conectPairs: ReadonlyArray<[number, number]>,
  title = 'LIG',
  opts?: BuildLigandSdfOptions,
): { sdf: string; heavyLineIndices: number[]; bondSource: LigandBondSource } {
  const ordered = atoms.slice();

  const serializeSdf = (
    atomList: ParsedPdbAtom[],
    bondList: Array<[number, number, number]>,
    bondSource: LigandBondSource,
  ): { sdf: string; heavyLineIndices: number[]; bondSource: LigandBondSource } => {
    const padL = (s: string, w: number) =>
      s.length >= w ? s.slice(0, w) : ' '.repeat(w - s.length) + s;
    const padR = (s: string, w: number) =>
      s.length >= w ? s.slice(0, w) : s + ' '.repeat(w - s.length);

    const out: string[] = [
      title.slice(0, 60),
      '  denovo.run GNINA 1.3 prep',
      '',
      `${padL(String(atomList.length), 3)}${padL(String(bondList.length), 3)}  0  0  0  0  0  0  0  0999 V2000`,
    ];

    const heavyLineIndices: number[] = [];
    const chargeEntries: Array<[number, number]> = [];
    for (let i = 0; i < atomList.length; i++) {
      const a = atomList[i];
      if (a.element !== 'H') heavyLineIndices.push(a.lineIndex);
      if (a.charge !== 0) chargeEntries.push([i + 1, a.charge]);
      const el = padR(a.element, 3);
      out.push(
        `${padL(a.x.toFixed(4), 10)}${padL(a.y.toFixed(4), 10)}${padL(a.z.toFixed(4), 10)} ${el} 0  0  0  0  0  0  0  0  0  0  0  0`,
      );
    }
    for (const [i, j, order] of bondList) {
      out.push(`${padL(String(i + 1), 3)}${padL(String(j + 1), 3)}${padL(String(order), 3)}  0  0  0  0`);
    }
    if (chargeEntries.length > 0) {
      for (let k = 0; k < chargeEntries.length; k += 8) {
        const chunk = chargeEntries.slice(k, k + 8);
        const body = chunk
          .map(([idx, chg]) => ` ${padL(String(idx), 3)} ${padL(String(chg), 3)}`)
          .join('');
        out.push(`M  CHG${padL(String(chunk.length), 3)}${body}`);
      }
    }
    out.push('M  END', '$$$$', '');
    return { sdf: out.join('\n'), heavyLineIndices, bondSource };
  };

  // --- Tier 1: Inline `_chem_comp_bond` exact topology ---
  const templateBonds = opts?.templateBonds;
  if (templateBonds && templateBonds.length > 0) {
    const nameToIdx = new Map<string, number>();
    let duplicateHeavyName = false;
    const heavyIndices: number[] = [];
    for (let i = 0; i < ordered.length; i++) {
      const nm = ordered[i].name.trim();
      if (ordered[i].element !== 'H') {
        heavyIndices.push(i);
        if (!nm || nameToIdx.has(nm)) {
          duplicateHeavyName = true;
          break;
        }
        nameToIdx.set(nm, i);
        const alias = opts?.atomNameAliases?.get(nm);
        if (alias && !nameToIdx.has(alias)) nameToIdx.set(alias, i);
      } else if (nm && !nameToIdx.has(nm)) {
        nameToIdx.set(nm, i);
        const alias = opts?.atomNameAliases?.get(nm);
        if (alias && !nameToIdx.has(alias)) nameToIdx.set(alias, i);
      }
    }

    if (!duplicateHeavyName) {
      const mapped = new Map<string, { u: number; v: number; order: number; aromatic: boolean }>();
      for (const tb of templateBonds) {
        const u = nameToIdx.get(tb.atom1.trim());
        const v = nameToIdx.get(tb.atom2.trim());
        if (u === undefined || v === undefined || u === v) continue;
        const a = Math.min(u, v);
        const b = Math.max(u, v);
        const key = `${a}/${b}`;
        const prev = mapped.get(key);
        const clampedOrder = Math.max(1, Math.min(4, tb.order));
        if (!prev) {
          mapped.set(key, { u: a, v: b, order: clampedOrder, aromatic: tb.aromatic });
        } else {
          if (clampedOrder > prev.order) prev.order = clampedOrder;
          if (tb.aromatic) prev.aromatic = true;
        }
      }

      const heavyEdges: Array<{ u: number; v: number; order: number; aromatic: boolean }> = [];
      const hEdges: Array<[number, number]> = [];
      const heavyBonded = new Set<number>();
      for (const e of mapped.values()) {
        const uH = ordered[e.u].element === 'H';
        const vH = ordered[e.v].element === 'H';
        if (!uH && !vH) {
          heavyEdges.push(e);
          heavyBonded.add(e.u);
          heavyBonded.add(e.v);
        } else if (uH !== vH) {
          hEdges.push([e.u, e.v]);
        }
      }

      let valid = true;
      if (heavyIndices.length > 1) {
        if (heavyEdges.length === 0) valid = false;
        for (const hi of heavyIndices) {
          if (!heavyBonded.has(hi)) {
            valid = false;
            break;
          }
        }
      }

      // Geometric connectedness parity (non-tautological): mappedComps <= geomComps
      if (valid && heavyIndices.length > 1) {
        const mappedPairs: Array<[number, number]> = heavyEdges.map((e) => [e.u, e.v]);
        const mappedComps = countConnectedComponents(heavyIndices, mappedPairs);
        const geomPairs: Array<[number, number]> = [];
        for (let a = 0; a < heavyIndices.length; a++) {
          const i = heavyIndices[a];
          const ri = COVALENT_RADII[ordered[i].element] ?? 0.76;
          for (let b = a + 1; b < heavyIndices.length; b++) {
            const j = heavyIndices[b];
            const rj = COVALENT_RADII[ordered[j].element] ?? 0.76;
            const d = Math.hypot(
              ordered[i].x - ordered[j].x,
              ordered[i].y - ordered[j].y,
              ordered[i].z - ordered[j].z,
            );
            if (d > 0.4 && d < ri + rj + 0.42) geomPairs.push([i, j]);
          }
        }
        const geomComps = countConnectedComponents(heavyIndices, geomPairs);
        if (mappedComps > geomComps) valid = false;
      }

      // Geometric sanity & scale-invariant swap detection
      let medianRatio = 1.0;
      if (valid && heavyEdges.length > 0) {
        const ratios: number[] = [];
        for (const e of heavyEdges) {
          const au = ordered[e.u];
          const av = ordered[e.v];
          const dist = Math.hypot(au.x - av.x, au.y - av.y, au.z - av.z);
          if (dist <= 0.4) {
            valid = false;
            break;
          }
          const rSum = (COVALENT_RADII[au.element] ?? 0.76) + (COVALENT_RADII[av.element] ?? 0.76);
          const rk = dist / rSum;
          if (rk > 2.5) {
            valid = false;
            break;
          }
          ratios.push(rk);
        }
        if (valid) {
          const sortedR = [...ratios].sort((a, b) => a - b);
          medianRatio = linearQuantile(sortedR, 0.5);
          for (const rk of ratios) {
            const rel = rk / medianRatio;
            if (rel < 0.5 || rel > 1.5) {
              valid = false;
              break;
            }
          }
        }
      }

      if (valid) {
        // Assemble all Tier 1 bonds (heavy-heavy + template H + nearest-heavy fallback for unbonded H)
        const t1Bonds: Array<{ u: number; v: number; order: number; aromatic: boolean }> = [
          ...heavyEdges.map((e) => ({ ...e })),
        ];
        const bondedH = new Set<number>();
        for (const [u, v] of hEdges) {
          t1Bonds.push({ u, v, order: 1, aromatic: false });
          if (ordered[u].element === 'H') bondedH.add(u);
          if (ordered[v].element === 'H') bondedH.add(v);
        }
        const hScale = Math.max(0.55, Math.min(1.0, medianRatio));
        for (let i = 0; i < ordered.length; i++) {
          if (ordered[i].element !== 'H' || bondedH.has(i)) continue;
          let bestJ = -1;
          let bestD = Infinity;
          for (const hi of heavyIndices) {
            const d = Math.hypot(
              ordered[i].x - ordered[hi].x,
              ordered[i].y - ordered[hi].y,
              ordered[i].z - ordered[hi].z,
            );
            const maxH = ((COVALENT_RADII[ordered[hi].element] ?? 0.76) + 0.31 + 0.5) * hScale;
            if (d > 0.2 && d < maxH && d < bestD) {
              bestD = d;
              bestJ = hi;
            }
          }
          if (bestJ >= 0) {
            t1Bonds.push({ u: Math.min(i, bestJ), v: Math.max(i, bestJ), order: 1, aromatic: false });
          }
        }

        // 5-ring SING+SING heteroatom ordering so `structure.js:156,165` detects 5-ring aromaticity
        // without needing order=4 (which would delete polar H on NH donors).
        const t1Atoms = ordered.slice();
        const edgePairs = t1Bonds.map((b): [number, number] => [b.u, b.v]);
        const aromCyclesInit = findCyclesOfLength5Or6(t1Atoms.length, edgePairs).filter((c) =>
          c.edgeIndices.every((ei) => t1Bonds[ei].aromatic),
        );
        const swapAtomOrder = (idxA: number, idxB: number) => {
          if (idxA === idxB) return;
          const tmp = t1Atoms[idxA];
          t1Atoms[idxA] = t1Atoms[idxB];
          t1Atoms[idxB] = tmp;
          for (const b of t1Bonds) {
            if (b.u === idxA) b.u = idxB;
            else if (b.u === idxB) b.u = idxA;
            if (b.v === idxA) b.v = idxB;
            else if (b.v === idxB) b.v = idxA;
            if (b.u > b.v) {
              const s = b.u;
              b.u = b.v;
              b.v = s;
            }
          }
        };
        for (const cyc of aromCyclesInit) {
          if (cyc.nodes.length !== 5) continue;
          const minNode = Math.min(...cyc.nodes);
          const el = t1Atoms[minNode].element;
          if (el !== 'N' && el !== 'O' && el !== 'S') continue;
          const incidentInCyc = cyc.edgeIndices.filter(
            (ei) => t1Bonds[ei].u === minNode || t1Bonds[ei].v === minNode,
          );
          if (incidentInCyc.length === 2 && incidentInCyc.every((ei) => t1Bonds[ei].order === 1)) {
            const doubAtom = cyc.nodes.find(
              (nd) =>
                nd !== minNode &&
                cyc.edgeIndices.some(
                  (ei) =>
                    t1Bonds[ei].order >= 2 && (t1Bonds[ei].u === nd || t1Bonds[ei].v === nd),
                ),
            );
            if (doubAtom !== undefined && doubAtom > minNode) {
              swapAtomOrder(minNode, doubAtom);
            }
          }
        }

        // Fixed-point targeted `order = 4` upgrade on uncovered `pdbx_aromatic_flag='Y'` cycles
        // whose heteroatoms (N/O/S) all have emitted SDF bond-order sum >= 3 (preserving NH donors).
        const finalCycles = findCyclesOfLength5Or6(
          t1Atoms.length,
          t1Bonds.map((b): [number, number] => [b.u, b.v]),
        ).filter((c) => c.edgeIndices.every((ei) => t1Bonds[ei].aromatic));

        let changed = true;
        while (changed) {
          changed = false;
          const currentTuples: Array<[number, number, number]> = t1Bonds.map((b) => [
            b.u,
            b.v,
            b.order,
          ]);
          const coveredAtoms = detectGninaAromaticAtoms(t1Atoms, currentTuples);
          for (const cyc of finalCycles) {
            if (cyc.nodes.every((nd) => coveredAtoms.has(nd))) continue;
            if (cyc.edgeIndices.every((ei) => t1Bonds[ei].order === 4)) continue;
            const heteroSafe = cyc.nodes.every((nd) => {
              const el = t1Atoms[nd].element;
              if (el !== 'N' && el !== 'O' && el !== 'S') return true;
              let sum = 0;
              for (const b of t1Bonds) {
                if (b.u === nd || b.v === nd) sum += b.order === 4 ? 1.5 : b.order;
              }
              return sum >= 3 - 1e-6;
            });
            if (!heteroSafe) continue;
            for (const ei of cyc.edgeIndices) {
              if (t1Bonds[ei].order !== 4) {
                t1Bonds[ei].order = 4;
                changed = true;
              }
            }
          }
        }

        return serializeSdf(
          t1Atoms,
          t1Bonds.map((b) => [b.u, b.v, b.order]),
          'chem_comp_bond',
        );
      }
    }
  }

  // --- Tier 2 & Tier 3: Explicit `CONECT` and Hardened Scale-Normalized Geometry ---
  const serialToIdx = new Map<number, number>();
  ordered.forEach((a, idx) => {
    if (a.serial > 0 && !serialToIdx.has(a.serial)) serialToIdx.set(a.serial, idx);
  });

  const bondMap = new Map<string, [number, number, number]>();
  const lockedPairs = new Set<string>();
  const addBond = (i: number, j: number, order = 1, lock = false) => {
    if (i === j || i < 0 || j < 0 || i >= ordered.length || j >= ordered.length) return;
    const a = Math.min(i, j);
    const b = Math.max(i, j);
    const key = `${a}/${b}`;
    if (lock) lockedPairs.add(key);
    const existing = bondMap.get(key);
    if (!existing) {
      bondMap.set(key, [a, b, order]);
    } else if (order > existing[2]) {
      existing[2] = order;
    }
  };

  // Preserve explicit PDB CONECT bond multiplicity: repeated directed entries from the same
  // source atom (`CONECT 1 2 2` -> order 2, `CONECT 1 2 2 2` -> order 3) encode double/triple
  // bonds, whereas reciprocal `CONECT 2 1` is the opposite direction of the same bond.
  const directedCounts = new Map<string, number>();
  for (const [s1, s2] of conectPairs) {
    const i = serialToIdx.get(s1);
    const j = serialToIdx.get(s2);
    if (i !== undefined && j !== undefined && i !== j) {
      const dirKey = `${i}->${j}`;
      directedCounts.set(dirKey, (directedCounts.get(dirKey) ?? 0) + 1);
    }
  }
  const heavyConectDeg = new Int32Array(ordered.length);
  for (const [dirKey, count] of directedCounts.entries()) {
    const [iStr, jStr] = dirKey.split('->');
    const i = Number(iStr);
    const j = Number(jStr);
    const revCount = directedCounts.get(`${j}->${i}`) ?? 0;
    const explicitOrder = Math.min(3, Math.max(1, count, revCount));
    const key = `${Math.min(i, j)}/${Math.max(i, j)}`;
    const isNew = !bondMap.has(key);
    addBond(i, j, explicitOrder, true);
    if (isNew && ordered[i].element !== 'H' && ordered[j].element !== 'H') {
      heavyConectDeg[i]++;
      heavyConectDeg[j]++;
    }
  }

  const heavyIndices: number[] = [];
  for (let i = 0; i < ordered.length; i++) {
    if (ordered[i].element !== 'H') heavyIndices.push(i);
  }
  const hasAnyConect = bondMap.size > 0;
  const hasCompleteConect =
    hasAnyConect && (heavyIndices.length <= 1 || heavyIndices.every((i) => heavyConectDeg[i] >= 1));

  // Upper-quantile downward scale normalization (zero-regression: s === 1.0 for normal 1.0x ligands)
  let scale = 1.0;
  if (heavyIndices.length >= 6) {
    const minRatios: number[] = [];
    for (let a = 0; a < heavyIndices.length; a++) {
      const i = heavyIndices[a];
      const ri = COVALENT_RADII[ordered[i].element] ?? 0.76;
      let minR = Infinity;
      for (let b = 0; b < heavyIndices.length; b++) {
        if (a === b) continue;
        const j = heavyIndices[b];
        const rj = COVALENT_RADII[ordered[j].element] ?? 0.76;
        const d = Math.hypot(
          ordered[i].x - ordered[j].x,
          ordered[i].y - ordered[j].y,
          ordered[i].z - ordered[j].z,
        );
        if (d > 1e-4) {
          const r = d / (ri + rj);
          if (r < minR) minR = r;
        }
      }
      if (Number.isFinite(minR)) minRatios.push(minR);
    }
    if (minRatios.length >= 6) {
      minRatios.sort((a, b) => a - b);
      const q90 = linearQuantile(minRatios, 0.9);
      if (q90 < 0.88) {
        scale = Math.max(0.55, Math.min(1.0, q90 / 0.99));
      }
    }
  }

  let bondSource: LigandBondSource;
  if (hasCompleteConect) {
    bondSource = 'conect_complete';
    // Skip heavy-heavy distance perception; only attach any unbonded explicit H atoms.
    const hBonded = new Set<number>();
    for (const [u, v] of bondMap.values()) {
      if (ordered[u].element === 'H') hBonded.add(u);
      if (ordered[v].element === 'H') hBonded.add(v);
    }
    for (let i = 0; i < ordered.length; i++) {
      if (ordered[i].element !== 'H' || hBonded.has(i)) continue;
      let bestJ = -1;
      let bestD = Infinity;
      for (const hi of heavyIndices) {
        const d = Math.hypot(
          ordered[i].x - ordered[hi].x,
          ordered[i].y - ordered[hi].y,
          ordered[i].z - ordered[hi].z,
        );
        const maxH = ((COVALENT_RADII[ordered[hi].element] ?? 0.76) + 0.31 + 0.42) * scale;
        if (d > 0.4 * scale && d < maxH && d < bestD) {
          bestD = d;
          bestJ = hi;
        }
      }
      if (bestJ >= 0) addBond(i, bestJ, 1);
    }
  } else {
    bondSource = hasAnyConect ? 'conect_partial_fallback' : 'distance_fallback';
    // Build working heavy-atom adjacency from any locked CONECT bonds
    const workAdj: Array<Array<{ j: number; norm: number }>> = ordered.map(() => []);
    for (const [u, v] of bondMap.values()) {
      if (ordered[u].element === 'H' || ordered[v].element === 'H') continue;
      const ru = COVALENT_RADII[ordered[u].element] ?? 0.76;
      const rv = COVALENT_RADII[ordered[v].element] ?? 0.76;
      const d = Math.hypot(
        ordered[u].x - ordered[v].x,
        ordered[u].y - ordered[v].y,
        ordered[u].z - ordered[v].z,
      );
      const norm = d / (ru + rv);
      workAdj[u].push({ j: v, norm });
      workAdj[v].push({ j: u, norm });
    }

    // Collect & sort heavy-heavy candidates by normalized length ascending
    const candidates: Array<{
      u: number;
      v: number;
      dist: number;
      norm: number;
      maxD: number;
    }> = [];
    for (let a = 0; a < heavyIndices.length; a++) {
      const u = heavyIndices[a];
      const ru = COVALENT_RADII[ordered[u].element] ?? 0.76;
      for (let b = a + 1; b < heavyIndices.length; b++) {
        const v = heavyIndices[b];
        const key = `${Math.min(u, v)}/${Math.max(u, v)}`;
        if (lockedPairs.has(key)) continue;
        const rv = COVALENT_RADII[ordered[v].element] ?? 0.76;
        const d = Math.hypot(
          ordered[u].x - ordered[v].x,
          ordered[u].y - ordered[v].y,
          ordered[u].z - ordered[v].z,
        );
        const maxD = ru + rv + 0.42;
        if (d > 0.4 * scale && d < maxD) {
          candidates.push({ u, v, dist: d, norm: d / (ru + rv), maxD });
        }
      }
    }
    candidates.sort((a, b) => a.norm - b.norm || a.u - b.u || a.v - b.v);

    const shortestPathUpTo5 = (
      src: number,
      dst: number,
    ): { length: number; maxNorm: number; midNode: number } | null => {
      const visited = new Set<number>([src]);
      const queue: Array<{ node: number; depth: number; maxNorm: number; firstHop: number }> = [
        { node: src, depth: 0, maxNorm: 0, firstHop: -1 },
      ];
      let head = 0;
      while (head < queue.length) {
        const cur = queue[head++];
        if (cur.depth >= 5) continue;
        for (const e of workAdj[cur.node]) {
          const nextMax = Math.max(cur.maxNorm, e.norm);
          const nextDepth = cur.depth + 1;
          if (e.j === dst) {
            return {
              length: nextDepth,
              maxNorm: nextMax,
              midNode: nextDepth === 2 ? cur.node : -1,
            };
          }
          if (!visited.has(e.j) && nextDepth < 5) {
            visited.add(e.j);
            queue.push({
              node: e.j,
              depth: nextDepth,
              maxNorm: nextMax,
              firstHop: cur.depth === 0 ? e.j : cur.firstHop,
            });
          }
        }
      }
      return null;
    };

    const violatesSp2OutOfPlane = (center: number, other: number): boolean => {
      if (workAdj[center].length !== 3) return false;
      const [n0, n1, n2] = workAdj[center];
      const sum =
        bondAngleDeg(ordered[n0.j], ordered[center], ordered[n1.j]) +
        bondAngleDeg(ordered[n1.j], ordered[center], ordered[n2.j]) +
        bondAngleDeg(ordered[n2.j], ordered[center], ordered[n0.j]);
      if (sum < 350) return false;
      for (const n of workAdj[center]) {
        if (bondAngleDeg(ordered[n.j], ordered[center], ordered[other]) < 95) return true;
      }
      return false;
    };

    for (const c of candidates) {
      const elU = ordered[c.u].element;
      const elV = ordered[c.v].element;
      if (!METALS_SET.has(elU) && workAdj[c.u].length >= (MAX_BONDS[elU] ?? 4)) continue;
      if (!METALS_SET.has(elV) && workAdj[c.v].length >= (MAX_BONDS[elV] ?? 4)) continue;

      const path = shortestPathUpTo5(c.u, c.v);
      const distEff = c.dist / scale;
      const passesScaledCutoff = distEff < c.maxD;
      const passesAnisotropicRing = path !== null && c.norm <= 1.4 * path.maxNorm;
      if (!passesScaledCutoff && !passesAnisotropicRing) continue;

      if (path !== null) {
        if (path.length === 2 && path.midNode >= 0) {
          if (bondAngleDeg(ordered[c.u], ordered[path.midNode], ordered[c.v]) >= 85) continue;
        }
        if (c.norm > 1.4 * path.maxNorm) continue;
      }

      if (violatesSp2OutOfPlane(c.u, c.v) || violatesSp2OutOfPlane(c.v, c.u)) continue;

      addBond(c.u, c.v, 1);
      workAdj[c.u].push({ j: c.v, norm: c.norm });
      workAdj[c.v].push({ j: c.u, norm: c.norm });
    }

    // Attach explicit H atoms to nearest heavy atom within scaled covalent cutoff
    const hBonded = new Set<number>();
    for (const [u, v] of bondMap.values()) {
      if (ordered[u].element === 'H') hBonded.add(u);
      if (ordered[v].element === 'H') hBonded.add(v);
    }
    for (let i = 0; i < ordered.length; i++) {
      if (ordered[i].element !== 'H' || hBonded.has(i)) continue;
      let bestJ = -1;
      let bestD = Infinity;
      for (const hi of heavyIndices) {
        const d = Math.hypot(
          ordered[i].x - ordered[hi].x,
          ordered[i].y - ordered[hi].y,
          ordered[i].z - ordered[hi].z,
        );
        const maxH = ((COVALENT_RADII[ordered[hi].element] ?? 0.76) + 0.31 + 0.42) * scale;
        if (d > 0.4 * scale && d < maxH && d < bestD) {
          bestD = d;
          bestJ = hi;
        }
      }
      if (bestJ >= 0) addBond(i, bestJ, 1);
    }
  }

  const bonds = [...bondMap.values()];
  const adj: Array<Array<{ j: number; bondIdx: number; dist: number; distEff: number }>> =
    ordered.map(() => []);
  bonds.forEach(([i, j], bondIdx) => {
    const a = ordered[i];
    const b = ordered[j];
    const dist = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
    const distEff = dist / scale;
    adj[i].push({ j, bondIdx, dist, distEff });
    adj[j].push({ j: i, bondIdx, dist, distEff });
  });

  // Geometric bond-order perception for PDB HETATMs (scale-normalized & angle-guarded):
  const heavyNbrs = ordered.map((_, i) => adj[i].filter((e) => ordered[e.j].element !== 'H'));
  const heavyDeg = heavyNbrs.map((list) => list.length);

  const atomValence = (idx: number): number =>
    adj[idx].reduce((sum, e) => sum + (bonds[e.bondIdx][2] === 4 ? 1.5 : bonds[e.bondIdx][2]), 0);

  const maxValence = (idx: number): number => {
    const { element: el, charge } = ordered[idx];
    if (el === 'C') return Math.max(1, 4 - Math.abs(charge));
    if (el === 'N') return Math.max(1, 3 + charge);
    if (el === 'O') return Math.max(0, 2 + charge);
    if (el === 'S') return Math.max(2, 6 - Math.abs(charge));
    if (el === 'P') return Math.max(3, 5 - Math.abs(charge));
    if (el === 'B') return Math.max(1, 3 - charge);
    return 4;
  };

  const canAddOrder = (idx: number, delta: number): boolean =>
    atomValence(idx) + delta <= maxValence(idx) + 1e-6;

  const isTrigonalForCarbonyl = (idx: number): boolean => {
    const nbrs = heavyNbrs[idx];
    if (nbrs.length <= 1) return true;
    if (nbrs.length === 2) {
      return bondAngleDeg(ordered[nbrs[0].j], ordered[idx], ordered[nbrs[1].j]) >= 114;
    }
    if (nbrs.length === 3) {
      const sum =
        bondAngleDeg(ordered[nbrs[0].j], ordered[idx], ordered[nbrs[1].j]) +
        bondAngleDeg(ordered[nbrs[1].j], ordered[idx], ordered[nbrs[2].j]) +
        bondAngleDeg(ordered[nbrs[2].j], ordered[idx], ordered[nbrs[0].j]);
      return sum >= 340;
    }
    return false;
  };

  const isTrigonal = (idx: number): boolean => {
    const nbrs = heavyNbrs[idx];
    if (nbrs.length <= 1) return true;
    if (nbrs.length === 2) {
      return bondAngleDeg(ordered[nbrs[0].j], ordered[idx], ordered[nbrs[1].j]) >= 100;
    }
    if (nbrs.length === 3) {
      const sum =
        bondAngleDeg(ordered[nbrs[0].j], ordered[idx], ordered[nbrs[1].j]) +
        bondAngleDeg(ordered[nbrs[1].j], ordered[idx], ordered[nbrs[2].j]) +
        bondAngleDeg(ordered[nbrs[2].j], ordered[idx], ordered[nbrs[0].j]);
      return sum >= 340;
    }
    return false;
  };

  const isLinear = (idx: number): boolean => {
    const nbrs = heavyNbrs[idx];
    if (nbrs.length <= 1) return true;
    if (nbrs.length === 2) {
      return bondAngleDeg(ordered[nbrs[0].j], ordered[idx], ordered[nbrs[1].j]) >= 155;
    }
    return false;
  };

  const isAmideOrUreidoCarbon = (cIdx: number): boolean =>
    ordered[cIdx].element === 'C' &&
    adj[cIdx].some((e) => {
      const nel = ordered[e.j].element;
      if (nel === 'O') return bonds[e.bondIdx][2] >= 2 || (heavyDeg[e.j] === 1 && e.distEff <= 1.32);
      if (nel === 'S') return bonds[e.bondIdx][2] >= 2 || (heavyDeg[e.j] === 1 && e.distEff <= 1.72);
      return false;
    });

  // 1. Perceive short exocyclic C=O carbonyls / carboxylates & P=O / S=O first
  const oxoCandidates: Array<{
    oIdx: number;
    edge: { j: number; bondIdx: number; dist: number; distEff: number };
  }> = [];
  for (let i = 0; i < ordered.length; i++) {
    if (ordered[i].element !== 'O' || heavyDeg[i] !== 1) continue;
    const edge = adj[i].find((e) => ordered[e.j].element !== 'H');
    if (edge) oxoCandidates.push({ oIdx: i, edge });
  }
  oxoCandidates.sort(
    (a, b) =>
      Math.abs(ordered[a.oIdx].charge) - Math.abs(ordered[b.oIdx].charge) ||
      a.edge.distEff - b.edge.distEff ||
      a.edge.bondIdx - b.edge.bondIdx,
  );

  for (const { oIdx, edge } of oxoCandidates) {
    const parentIdx = edge.j;
    const parent = ordered[parentIdx];
    if (
      parent.element === 'C' &&
      edge.distEff <= 1.29 &&
      heavyDeg[parentIdx] <= 3 &&
      isTrigonalForCarbonyl(parentIdx)
    ) {
      const termO = adj[parentIdx].filter(
        (e) => ordered[e.j].element === 'O' && heavyDeg[e.j] === 1 && e.distEff <= 1.32,
      );
      if (termO.length === 2) {
        if (!termO.some((t) => bonds[t.bondIdx][2] >= 2)) {
          const sorted = [...termO].sort(
            (x, y) =>
              Math.abs(ordered[x.j].charge) - Math.abs(ordered[y.j].charge) ||
              x.distEff - y.distEff ||
              x.bondIdx - y.bondIdx,
          );
          for (const target of sorted) {
            if (
              bonds[target.bondIdx][2] === 1 &&
              canAddOrder(parentIdx, 1) &&
              canAddOrder(target.j, 1)
            ) {
              bonds[target.bondIdx][2] = 2;
              break;
            }
          }
        }
      } else if (bonds[edge.bondIdx][2] === 1 && canAddOrder(parentIdx, 1) && canAddOrder(oIdx, 1)) {
        bonds[edge.bondIdx][2] = 2;
      }
    } else if ((parent.element === 'P' || parent.element === 'S') && edge.distEff <= 1.52) {
      if (bonds[edge.bondIdx][2] === 1 && canAddOrder(parentIdx, 1) && canAddOrder(oIdx, 1)) {
        bonds[edge.bondIdx][2] = 2;
      }
    }
  }

  // 2. Detect planar 5- and 6-membered conjugated rings -> mark ring bonds as aromatic (order 4)
  //    Skip any cycle that already carries an explicit endocyclic order >= 2 bond (e.g. Kekule CONECT).
  const visitedRings = new Set<string>();
  for (let start = 0; start < ordered.length; start++) {
    if (!['C', 'N', 'O', 'S'].includes(ordered[start].element)) continue;
    const walk = (path: number[], edgePath: number[]) => {
      const cur = path[path.length - 1];
      for (const e of adj[cur]) {
        const next = e.j;
        if (next === start && (path.length === 5 || path.length === 6)) {
          const cycleEdges = [...edgePath, e.bondIdx];
          const sig = [...path].sort((a, b) => a - b).join(',');
          if (visitedRings.has(sig)) continue;
          visitedRings.add(sig);

          const cycleEdgeSet = new Set(cycleEdges);
          const hasEndocyclicMultiBond = cycleEdges.some(
            (bi) => bonds[bi][2] === 2 || bonds[bi][2] === 3,
          );
          if (hasEndocyclicMultiBond) continue;

          const hasOnlyRingEls = path.every((idx) =>
            ['C', 'N', 'O', 'S'].includes(ordered[idx].element),
          );
          const allLowDeg = path.every((idx) => heavyDeg[idx] <= 3 && adj[idx].length <= 3);
          const hasExocyclicMultiBond = path.some((idx) =>
            adj[idx].some(
              (ae) =>
                !cycleEdgeSet.has(ae.bondIdx) &&
                (bonds[ae.bondIdx][2] === 2 || bonds[ae.bondIdx][2] === 3),
            ),
          );
          const maxBondLenEff = Math.max(
            ...cycleEdges.map((bi) => {
              const [u, v] = bonds[bi];
              return (
                Math.hypot(
                  ordered[u].x - ordered[v].x,
                  ordered[u].y - ordered[v].y,
                  ordered[u].z - ordered[v].z,
                ) / scale
              );
            }),
          );
          const hasSulfur = path.some((idx) => ordered[idx].element === 'S');
          if (
            hasOnlyRingEls &&
            allLowDeg &&
            !hasExocyclicMultiBond &&
            maxBondLenEff <= (hasSulfur ? 1.78 : 1.49)
          ) {
            let planar = true;
            const p0 = ordered[path[0]];
            const p1 = ordered[path[1]];
            const p2 = ordered[path[2]];
            const ux = p1.x - p0.x,
              uy = p1.y - p0.y,
              uz = p1.z - p0.z;
            const vx = p2.x - p0.x,
              vy = p2.y - p0.y,
              vz = p2.z - p0.z;
            let nx = uy * vz - uz * vy;
            let ny = uz * vx - ux * vz;
            let nz = ux * vy - uy * vx;
            const nLen = Math.hypot(nx, ny, nz);
            if (nLen > 1e-4) {
              nx /= nLen;
              ny /= nLen;
              nz /= nLen;
              for (let k = 3; k < path.length; k++) {
                const pk = ordered[path[k]];
                const dPlaneEff =
                  Math.abs((pk.x - p0.x) * nx + (pk.y - p0.y) * ny + (pk.z - p0.z) * nz) / scale;
                if (dPlaneEff > 0.18) {
                  planar = false;
                  break;
                }
              }
            }
            if (planar) {
              for (const bi of cycleEdges) bonds[bi][2] = 4;
            }
          }
        } else if (
          path.length < 6 &&
          next > start &&
          !path.includes(next) &&
          ['C', 'N', 'O', 'S'].includes(ordered[next].element)
        ) {
          walk([...path, next], [...edgePath, e.bondIdx]);
        }
      }
    };
    walk([start], []);
  }

  // 3. Perceive short acyclic unsaturated C=C / C=N / C#C / C#N bonds when still marked order 1
  const shortCnCandidates: Array<{ bi: number; u: number; v: number; distEff: number }> = [];
  for (let bi = 0; bi < bonds.length; bi++) {
    if (bonds[bi][2] !== 1) continue;
    const [u, v] = bonds[bi];
    const au = ordered[u];
    const av = ordered[v];
    if (
      (au.element === 'C' || au.element === 'N') &&
      (av.element === 'C' || av.element === 'N')
    ) {
      const distEff = Math.hypot(au.x - av.x, au.y - av.y, au.z - av.z) / scale;
      if (distEff <= 1.37) shortCnCandidates.push({ bi, u, v, distEff });
    }
  }
  shortCnCandidates.sort((a, b) => a.distEff - b.distEff || a.bi - b.bi);

  for (const { bi, u, v, distEff } of shortCnCandidates) {
    if (bonds[bi][2] !== 1) continue;
    const au = ordered[u];
    const av = ordered[v];
    if (
      distEff <= 1.22 &&
      heavyDeg[u] <= 2 &&
      heavyDeg[v] <= 2 &&
      isLinear(u) &&
      isLinear(v) &&
      canAddOrder(u, 2) &&
      canAddOrder(v, 2)
    ) {
      bonds[bi][2] = 3;
      continue;
    }
    if (
      distEff <= 1.37 &&
      heavyDeg[u] <= 3 &&
      heavyDeg[v] <= 3 &&
      isTrigonal(u) &&
      isTrigonal(v) &&
      canAddOrder(u, 1) &&
      canAddOrder(v, 1)
    ) {
      if (au.element !== av.element) {
        const cIdx = au.element === 'C' ? u : v;
        const nIdx = au.element === 'N' ? u : v;
        if (isAmideOrUreidoCarbon(cIdx)) continue;
        const cNitrogenNeighbors = adj[cIdx].filter((e) => ordered[e.j].element === 'N').length;
        if (heavyDeg[nIdx] === 1 && ordered[nIdx].charge === 0 && cNitrogenNeighbors >= 2) continue;
      }
      bonds[bi][2] = 2;
    }
  }

  return serializeSdf(ordered, bonds, bondSource);
}

function matchLigandToCifResidue(
  lig: DetectedLigandInstance,
  cifResidues: ReadonlyArray<CifLigandResidueMeta>,
): CifLigandResidueMeta | null {
  if (cifResidues.length === 0) return null;
  const heavy = lig.atoms.filter((a) => a.element !== 'H');
  if (heavy.length === 0) return null;
  let sx = 0,
    sy = 0,
    sz = 0;
  for (const a of heavy) {
    sx += a.x;
    sy += a.y;
    sz += a.z;
  }
  const cx = sx / heavy.length,
    cy = sy / heavy.length,
    cz = sz / heavy.length;

  let best: CifLigandResidueMeta | null = null;
  let bestScore = -1;
  let bestDist = Infinity;
  for (const cand of cifResidues) {
    const pdbComp = lig.compId.toUpperCase();
    const compMatch =
      pdbComp === cand.fullCompId || (pdbComp.length === 3 && cand.fullCompId.startsWith(pdbComp));
    if (!compMatch) continue;

    let overlap = 0;
    for (const a of heavy) {
      const nm = a.name.trim();
      if (
        cand.heavyAtomNames.has(nm) ||
        cand.authToLabelAtomName.has(nm) ||
        cand.labelToAuthAtomName.has(nm)
      ) {
        overlap++;
      }
    }
    if (overlap < Math.ceil(heavy.length * 0.5)) continue;

    const dist = Math.hypot(cx - cand.centroid[0], cy - cand.centroid[1], cz - cand.centroid[2]);
    const chainMatch = lig.chain === cand.authChain || lig.chain === cand.labelChain;
    const seqMatch = lig.resSeq === cand.authSeq || lig.resSeq === cand.labelSeq;
    // Reject stale cifText when neither coordinates nor chain+seq agree
    if (dist > 1.0 && !(chainMatch && seqMatch)) continue;

    const score = (dist <= 1.0 ? 4 : 0) + (chainMatch && seqMatch ? 2 : 0) + overlap / heavy.length;
    if (score > bestScore || (score === bestScore && dist < bestDist)) {
      bestScore = score;
      bestDist = dist;
      best = cand;
    }
  }
  return best;
}

/**
 * Split a holo PDB or mmCIF structure into a GNINA-ready receptor PDB (protein + modified AAs +
 * optional monoatomic metal cofactors) and detected organic ligand instance(s).
 */
export function extractHoloReceptorAndLigands(
  structText: string,
  opts?: {
    ligandCode?: string;
    includeMetals?: boolean;
    cifText?: string;
    cifOverlay?: CifLigandOverlay;
  },
): GninaComplexPrep {
  const primaryIsCif = isMmcifText(structText);
  const normalizedPdb = normalizeToPdbText(structText);
  const cifOverlay =
    opts?.cifOverlay ??
    (primaryIsCif
      ? parseCifLigandOverlay(structText)
      : opts?.cifText && opts.cifText.trim()
        ? parseCifLigandOverlay(opts.cifText)
        : null);

  const { lines, atoms, conectPairs } = parseAllPdbAtomLines(normalizedPdb);
  const includeMetals = opts?.includeMetals !== false;
  const requestedRaw = (opts?.ligandCode ?? '').trim().toUpperCase();

  const ligandBuckets = new Map<string, ParsedPdbAtom[]>();
  const receptorLines: string[] = [];
  const keptMetalsSet = new Set<string>();

  for (const a of atoms) {
    const isStandardPoly =
      STANDARD_AA_RESNAMES.has(a.resName) || NUCLEIC_ACID_RESNAMES.has(a.resName);
    if (a.record === 'ATOM' || isStandardPoly) {
      receptorLines.push(lines[a.lineIndex]);
      continue;
    }

    const cls = classifyHetatmCompId(a.resName);
    const explicitMatch =
      requestedRaw.length > 0 &&
      (requestedRaw === a.resName ||
        (a.resName.length === 3 && requestedRaw.startsWith(a.resName)) ||
        requestedRaw === `${a.chain}:${a.resSeq}${a.insCode}`.toUpperCase() ||
        requestedRaw === `${a.resName}:${a.chain}:${a.resSeq}${a.insCode}`.toUpperCase());

    if (cls === 'modified-amino-acid' && !explicitMatch) {
      receptorLines.push(lines[a.lineIndex]);
      continue;
    }
    if (cls === 'monoatomic-ion' && !explicitMatch) {
      if (includeMetals) {
        receptorLines.push(lines[a.lineIndex]);
        keptMetalsSet.add(a.resName);
      }
      continue;
    }
    if (cls === 'buffer-or-solvent' && !explicitMatch) {
      continue;
    }

    const instKey = `${a.resName}:${a.chain}:${a.resSeq}${a.insCode}`;
    let bucket = ligandBuckets.get(instKey);
    if (!bucket) {
      bucket = [];
      ligandBuckets.set(instKey, bucket);
    }
    bucket.push(a);
  }

  const detectedLigands: DetectedLigandInstance[] = [];
  const matchedCifByInstKey = new Map<string, CifLigandResidueMeta>();
  for (const [instanceKey, bucket] of ligandBuckets.entries()) {
    const heavy = bucket.filter((a) => a.element !== 'H');
    if (heavy.length === 0) continue;
    const first = bucket[0];
    const inst: DetectedLigandInstance = {
      instanceKey,
      compId: first.resName,
      chain: first.chain,
      resSeq: first.resSeq,
      insCode: first.insCode,
      heavyAtomCount: heavy.length,
      totalAtomCount: bucket.length,
      atoms: bucket,
      heavyLineIndices: heavy.map((a) => a.lineIndex),
      allLineIndices: bucket.map((a) => a.lineIndex),
    };
    if (cifOverlay) {
      const matched = matchLigandToCifResidue(inst, cifOverlay.cifResidues);
      if (matched) {
        inst.compId = matched.fullCompId;
        inst.instanceKey = `${matched.fullCompId}:${inst.chain}:${inst.resSeq}${inst.insCode}`;
        matchedCifByInstKey.set(inst.instanceKey, matched);
        // Overlay `pdbx_formal_charge` where PDB charge column was blank/0
        if (matched.formalChargesByAtomName.size > 0) {
          for (const atom of inst.atoms) {
            if (atom.charge === 0) {
              const nm = atom.name.trim();
              const chg =
                matched.formalChargesByAtomName.get(nm) ??
                matched.formalChargesByAtomName.get(matched.authToLabelAtomName.get(nm) ?? '') ??
                matched.formalChargesByAtomName.get(matched.labelToAuthAtomName.get(nm) ?? '');
              if (chg !== undefined && chg !== 0) atom.charge = chg;
            }
          }
        }
      }
    }
    detectedLigands.push(inst);
  }

  // Largest heavy-atom ligand first (deterministic tie-breaker by original line order)
  detectedLigands.sort(
    (a, b) =>
      b.heavyAtomCount - a.heavyAtomCount ||
      (a.atoms[0]?.lineIndex ?? 0) - (b.atoms[0]?.lineIndex ?? 0),
  );

  let selectedLigand: DetectedLigandInstance | null = null;
  if (requestedRaw) {
    selectedLigand =
      detectedLigands.find(
        (d) =>
          d.compId === requestedRaw ||
          (d.atoms[0]?.resName ?? '') === requestedRaw ||
          `${d.chain}:${d.resSeq}${d.insCode}`.toUpperCase() === requestedRaw ||
          d.instanceKey.toUpperCase() === requestedRaw,
      ) ?? null;
  } else {
    selectedLigand = detectedLigands[0] ?? null;
  }

  const receptorPdb = receptorLines.length > 0 ? `${receptorLines.join('\n')}\nEND\n` : '';
  let selectedLigandSdf: string | null = null;
  let selectedLigandBondSource: LigandBondSource | null = null;
  if (selectedLigand) {
    const matchedCif = matchedCifByInstKey.get(selectedLigand.instanceKey);
    const templateBonds =
      cifOverlay?.componentBonds.get(selectedLigand.compId) ??
      cifOverlay?.componentBonds.get(selectedLigand.atoms[0]?.resName ?? '');
    const aliases = new Map<string, string>();
    if (matchedCif) {
      for (const [k, v] of matchedCif.authToLabelAtomName) aliases.set(k, v);
      for (const [k, v] of matchedCif.labelToAuthAtomName) {
        if (!aliases.has(k)) aliases.set(k, v);
      }
    }
    const built = buildLigandV2000Sdf(
      selectedLigand.atoms,
      conectPairs,
      `${selectedLigand.compId}_${selectedLigand.chain}${selectedLigand.resSeq}${selectedLigand.insCode}`,
      templateBonds
        ? {
            templateBonds,
            atomNameAliases: aliases.size > 0 ? aliases : undefined,
          }
        : undefined,
    );
    selectedLigandSdf = built.sdf;
    selectedLigandBondSource = built.bondSource;
    selectedLigand.heavyLineIndices = built.heavyLineIndices;
  }

  return {
    normalizedPdb,
    receptorPdb,
    detectedLigands,
    selectedLigand,
    selectedLigandSdf,
    selectedLigandBondSource,
    keptMetals: [...keptMetalsSet],
    detectedLigandCodes: [...new Set(detectedLigands.map((d) => d.compId))],
  };
}

/**
 * Find protein receptor residues with at least one heavy atom within `radiusAngstrom` Å
 * of any ligand heavy atom, sorted by closest heavy-atom contact distance.
 */
export function findPocketResidues(
  receptorPdb: string,
  ligandCoords: ReadonlyArray<{ x: number; y: number; z: number; el?: string; element?: string }>,
  radiusAngstrom = 5.0,
  maxResidues = 24,
): PocketResidueEntry[] {
  const ligHeavy = ligandCoords.filter((a) => (a.el ?? a.element ?? 'C').toUpperCase() !== 'H');
  if (ligHeavy.length === 0) return [];

  const { atoms } = parseAllPdbAtomLines(receptorPdb);
  const r2Max = radiusAngstrom * radiusAngstrom;
  const byResidue = new Map<string, PocketResidueEntry>();

  for (const a of atoms) {
    if (a.element === 'H') continue;
    // Only scan polymer protein residues (not monoatomic metal cofactors)
    if (a.record === 'HETATM' && classifyHetatmCompId(a.resName) !== 'modified-amino-acid') {
      continue;
    }
    let minD2 = Infinity;
    for (let k = 0; k < ligHeavy.length; k++) {
      const l = ligHeavy[k];
      const dx = a.x - l.x;
      const dy = a.y - l.y;
      const dz = a.z - l.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < minD2) minD2 = d2;
    }
    if (minD2 > r2Max) continue;

    const gninaKey = [a.chain, a.resSeq, a.insCode, a.resName].filter(Boolean).join('/');
    const dist = Math.sqrt(minD2);
    const prev = byResidue.get(gninaKey);
    if (!prev || dist < prev.minDist) {
      byResidue.set(gninaKey, {
        gninaKey,
        residueId: `${a.chain}${a.resSeq}${a.insCode}`,
        chain: a.chain,
        resSeq: a.resSeq,
        insCode: a.insCode,
        resName: a.resName,
        minDist: dist,
      });
    }
  }

  return [...byResidue.values()]
    .sort((a, b) => a.minDist - b.minDist)
    .slice(0, Math.max(1, maxResidues));
}

function formatPdbCoord8(v: number): string {
  const s = v.toFixed(3);
  return s.length >= 8 ? s.slice(0, 8) : ' '.repeat(8 - s.length) + s;
}

/**
 * Re-inject minimized heavy-atom coordinates from `minimizePose()` back into the original
 * normalized holo PDB text (preserving all receptor atoms, metals, and ligand atom metadata).
 */
export function applyMinimizedHoloLigandCoords(
  normalizedPdb: string,
  selectedLigand: DetectedLigandInstance,
  minimizedHeavyAtoms: ReadonlyArray<{ x: number; y: number; z: number }>,
): string {
  const lines = normalizedPdb.split(/\r?\n/);
  const heavyIndices = selectedLigand.heavyLineIndices;
  const count = Math.min(heavyIndices.length, minimizedHeavyAtoms.length);

  // Drop hydrogens of the selected ligand if their positions would go stale after rigid-body move
  const heavySet = new Set(heavyIndices);
  const dropLineSet = new Set<number>();
  for (const idx of selectedLigand.allLineIndices) {
    if (!heavySet.has(idx)) dropLineSet.add(idx);
  }

  for (let i = 0; i < count; i++) {
    const lineIdx = heavyIndices[i];
    const orig = lines[lineIdx];
    if (!orig || orig.length < 54) continue;
    const atom = minimizedHeavyAtoms[i];
    lines[lineIdx] =
      orig.slice(0, 30) +
      formatPdbCoord8(atom.x) +
      formatPdbCoord8(atom.y) +
      formatPdbCoord8(atom.z) +
      orig.slice(54);
  }

  return lines.filter((_, idx) => !dropLineSet.has(idx)).join('\n');
}

/**
 * Append an external minimized/scored SDF/MOL2 pose as `HETATM` (`LIG`) records onto a receptor PDB
 * so Mol* renders the protein–ligand complex directly in 3D.
 */
export function appendExternalLigandToReceptorPdb(
  receptorPdb: string,
  poseAtoms: ReadonlyArray<{ x: number; y: number; z: number; el?: string; name?: string }>,
  resName = 'LIG',
  chainId = 'L',
): string {
  const baseLines = receptorPdb
    .split(/\r?\n/)
    .filter((l) => !/^(END|ENDMDL)\b/.test(l.trim()));
  const code = resName.slice(0, 3).toUpperCase().padStart(3, ' ');
  const ch = (chainId.trim()[0] ?? 'L').toUpperCase();
  let serial = baseLines.length + 1;

  for (let i = 0; i < poseAtoms.length; i++) {
    const a = poseAtoms[i];
    const el = (a.el ?? 'C').trim();
    const atomLabel = (a.name?.trim() || `${el.toUpperCase()}${i + 1}`).slice(0, 4);
    const nameField = atomLabel.length < 4 ? ` ${atomLabel.padEnd(3, ' ')}` : atomLabel;
    const serialStr = String(Math.min(99999, serial++)).padStart(5, ' ');
    const elemStr = el.toUpperCase().slice(0, 2).padStart(2, ' ');
    baseLines.push(
      `HETATM${serialStr} ${nameField} ${code} ${ch}   1    ` +
        `${formatPdbCoord8(a.x)}${formatPdbCoord8(a.y)}${formatPdbCoord8(a.z)}  1.00  0.00          ${elemStr}`,
    );
  }
  baseLines.push('END');
  return `${baseLines.join('\n')}\n`;
}

/** Compute heavy-atom RMSD (Å) between two identically ordered poses. */
export function computePoseHeavyRmsd(
  before: ReadonlyArray<{ x: number; y: number; z: number }>,
  after: ReadonlyArray<{ x: number; y: number; z: number }>,
): number {
  const n = Math.min(before.length, after.length);
  if (n === 0) return 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const dx = after[i].x - before[i].x;
    const dy = after[i].y - before[i].y;
    const dz = after[i].z - before[i].z;
    sumSq += dx * dx + dy * dy + dz * dz;
  }
  return Math.sqrt(sumSq / n);
}

/** Detect whether pasted external ligand text is MOL2 or SDF. */
export function detectExternalLigandFormat(text: string): 'mol2' | 'sdf' {
  return text.includes('@<TRIPOS>MOLECULE') ? 'mol2' : 'sdf';
}

/**
 * Parse user-friendly pocket residue selectors (e.g. `"A45, A79, A90-94"` or `"45, 79"`)
 * into GNINA's exact `pocketResidues` keys (`[chain, seq, ins, res].filter(Boolean).join('/')`),
 * resolving exact case-sensitive chain IDs first and reporting any unmatched tokens.
 */
export function parsePocketResidueTokensStrict(
  spec: string,
  receptorPdb: string,
): { keys: string[]; unmatched: string[] } {
  const trimmed = spec.trim();
  if (!trimmed) return { keys: [], unmatched: [] };
  const { atoms } = parseAllPdbAtomLines(receptorPdb);
  const resByExact = new Map<string, string>();
  const resByUpper = new Map<string, string[]>();
  const resByNumOnly = new Map<string, string[]>();
  const validKeys = new Set<string>();

  for (const a of atoms) {
    if (a.record === 'HETATM' && classifyHetatmCompId(a.resName) !== 'modified-amino-acid') {
      continue;
    }
    const key = [a.chain, a.resSeq, a.insCode, a.resName].filter(Boolean).join('/');
    if (validKeys.has(key)) continue;
    validKeys.add(key);
    const exactKey = `${a.chain}:${a.resSeq}${a.insCode}`;
    if (!resByExact.has(exactKey)) resByExact.set(exactKey, key);
    const upperKey = `${a.chain.toUpperCase()}:${a.resSeq}${a.insCode.toUpperCase()}`;
    const uList = resByUpper.get(upperKey) ?? [];
    uList.push(key);
    resByUpper.set(upperKey, uList);
    const list = resByNumOnly.get(a.resSeq) ?? [];
    list.push(key);
    resByNumOnly.set(a.resSeq, list);
  }

  const lookupChainRes = (chain: string, resWithIns: string): string | null => {
    const exact = resByExact.get(`${chain}:${resWithIns}`);
    if (exact) return exact;
    const uHits = resByUpper.get(`${chain.toUpperCase()}:${resWithIns.toUpperCase()}`);
    if (uHits && uHits.length === 1) return uHits[0];
    return null;
  };

  const out = new Set<string>();
  const unmatched: string[] = [];
  const tokens = trimmed.split(/[\s,;]+/).filter(Boolean);
  for (const rawTok of tokens) {
    const tok = rawTok.trim();
    if (validKeys.has(tok)) {
      out.add(tok);
      continue;
    }
    // Range with chain prefix: A45-52 or A:45-52 (preserves exact chain case!)
    const rangeMatch = tok.match(/^([A-Za-z]):?(\d+)-(\d+)$/);
    if (rangeMatch) {
      const ch = rangeMatch[1];
      const lo = Number(rangeMatch[2]);
      const hi = Number(rangeMatch[3]);
      let matchedAny = false;
      for (let n = Math.min(lo, hi); n <= Math.max(lo, hi); n++) {
        const hit = lookupChainRes(ch, String(n));
        if (hit) {
          out.add(hit);
          matchedAny = true;
        }
      }
      if (!matchedAny) unmatched.push(tok);
      continue;
    }
    // Single residue with chain prefix: A45, a45, A:45, A100A
    const singleChain = tok.match(/^([A-Za-z]):?(\d+)([A-Za-z]?)$/);
    if (singleChain) {
      const hit = lookupChainRes(singleChain[1], `${singleChain[2]}${singleChain[3]}`);
      if (hit) {
        out.add(hit);
      } else {
        unmatched.push(tok);
      }
      continue;
    }
    // Bare residue number or range (match first/all chains)
    const bareRange = tok.match(/^(\d+)-(\d+)$/);
    if (bareRange) {
      const lo = Number(bareRange[1]);
      const hi = Number(bareRange[2]);
      let matchedAny = false;
      for (let n = Math.min(lo, hi); n <= Math.max(lo, hi); n++) {
        for (const hit of resByNumOnly.get(String(n)) ?? []) {
          out.add(hit);
          matchedAny = true;
        }
      }
      if (!matchedAny) unmatched.push(tok);
      continue;
    }
    if (/^\d+$/.test(tok)) {
      const hits = resByNumOnly.get(tok) ?? [];
      if (hits.length > 0) {
        for (const hit of hits) out.add(hit);
      } else {
        unmatched.push(tok);
      }
      continue;
    }
    unmatched.push(tok);
  }
  return { keys: [...out], unmatched };
}

export function parsePocketResidueTokens(spec: string, receptorPdb: string): string[] {
  return parsePocketResidueTokensStrict(spec, receptorPdb).keys;
}

/**
 * Build a multi-model PDB (`MODEL 1`..`MODEL K`) combining the receptor with each ranked docked pose
 * so Mol*'s frame scrubber lets users inspect Pose #1..#K interactively in 3D.
 */
export type GninaTaskMode = 'score' | 'pocket-scan' | 'minimize' | 'dock';
export type GninaLigandSourceKind = 'external' | 'rcsb-ccd' | 'holo';

export function normalizeGninaTaskMode(raw: unknown): GninaTaskMode {
  const val = String(raw ?? 'score').trim().toLowerCase();
  if (val === 'pocket-scan' || val === 'minimize' || val === 'dock') return val;
  return 'score';
}

/**
 * Single source of truth for GNINA ligand precedence across worker, input preview,
 * bio-capabilities preflight, and canvas summaries:
 *   pasted SDF/MOL2 (`external`) > RCSB CCD (`rcsb-ccd`, dock mode only) > holo HETATM (`holo`).
 * `dockLigandCcd` is returned blank unless `source === 'rcsb-ccd'`.
 */
export function resolveGninaLigandSource(p: Record<string, unknown>): {
  readonly taskMode: GninaTaskMode;
  readonly ligandSdf: string;
  readonly ligandCode: string;
  readonly dockLigandCcd: string;
  readonly source: GninaLigandSourceKind;
} {
  const taskMode = normalizeGninaTaskMode(p.taskMode);
  const ligandSdf = String(p.ligandSdf ?? '').trim();
  const rawCcd = String(p.dockLigandCcd ?? '').trim().toUpperCase();
  const source: GninaLigandSourceKind = ligandSdf
    ? 'external'
    : taskMode === 'dock' && rawCcd
      ? 'rcsb-ccd'
      : 'holo';
  return {
    taskMode,
    ligandSdf,
    ligandCode: String(p.ligandCode ?? '').trim(),
    dockLigandCcd: source === 'rcsb-ccd' ? rawCcd : '',
    source,
  } as const;
}

export function buildMultiPoseDockedPdb(
  receptorPdb: string,
  rankedPoses: ReadonlyArray<ReadonlyArray<{ x: number; y: number; z: number; el?: string; name?: string }>>,
  resName = 'LIG',
  chainId = 'L',
): string {
  const models: string[] = [];
  for (let i = 0; i < rankedPoses.length; i++) {
    const framePdb = appendExternalLigandToReceptorPdb(receptorPdb, rankedPoses[i], resName, chainId)
      .split(/\r?\n/)
      .filter((l) => !/^END\b/.test(l.trim()))
      .join('\n');
    models.push(`MODEL     ${String(i + 1).padStart(4, ' ')}\n${framePdb}\nENDMDL`);
  }
  return `${models.join('\n')}\nEND\n`;
}
