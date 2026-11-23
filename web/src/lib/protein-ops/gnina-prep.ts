// Pure-TS protein–ligand preparation, pocket detection, and pose coordinate re-injection
// helpers for GNINA 1.3 WebGPU docking & affinity scoring.

import {
  classifyHetatmCompId,
  NUCLEIC_ACID_RESNAMES,
  STANDARD_AA_RESNAMES,
} from './hetatm-policy';
import { mmcifToPdb } from './pdb';

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

function normalizeToPdbText(rawText: string): string {
  const trimmed = rawText.trimStart();
  if (trimmed.startsWith('data_') || rawText.includes('_atom_site.')) {
    return mmcifToPdb(rawText, { firstModelOnly: true });
  }
  return rawText;
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

  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (l.startsWith('ENDMDL') && atoms.length > 0) break;
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

/**
 * Synthesize a V2000 SDF block for a set of PDB ligand atoms, inferring covalent bonds
 * and basic geometric bond orders (carbonyl C=O, terminal imine/nitrile, aromatic 5/6 rings)
 * so GNINA's SDF atom typer & polar-hydrogen placement run identically to external SDF poses.
 */
export function buildLigandV2000Sdf(
  atoms: ParsedPdbAtom[],
  conectPairs: ReadonlyArray<[number, number]>,
  title = 'LIG',
): { sdf: string; heavyLineIndices: number[] } {
  const ordered = atoms.slice();
  const serialToIdx = new Map<number, number>();
  ordered.forEach((a, idx) => {
    if (a.serial > 0 && !serialToIdx.has(a.serial)) serialToIdx.set(a.serial, idx);
  });

  const bondMap = new Map<string, [number, number, number]>();
  const addBond = (i: number, j: number, order = 1) => {
    if (i === j || i < 0 || j < 0 || i >= ordered.length || j >= ordered.length) return;
    const a = Math.min(i, j);
    const b = Math.max(i, j);
    const key = `${a}/${b}`;
    if (!bondMap.has(key)) bondMap.set(key, [a, b, order]);
  };

  for (const [s1, s2] of conectPairs) {
    const i = serialToIdx.get(s1);
    const j = serialToIdx.get(s2);
    if (i !== undefined && j !== undefined) addBond(i, j, 1);
  }

  for (let i = 0; i < ordered.length; i++) {
    const a = ordered[i];
    const ra = COVALENT_RADII[a.element] ?? 0.76;
    for (let j = i + 1; j < ordered.length; j++) {
      const b = ordered[j];
      if (a.element === 'H' && b.element === 'H') continue;
      const rb = COVALENT_RADII[b.element] ?? 0.76;
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      const maxD = ra + rb + 0.42;
      if (d2 > 0.16 && d2 < maxD * maxD) {
        addBond(i, j, 1);
      }
    }
  }

  const bonds = [...bondMap.values()];
  const adj: Array<Array<{ j: number; bondIdx: number; dist: number }>> = ordered.map(() => []);
  bonds.forEach(([i, j], bondIdx) => {
    const a = ordered[i];
    const b = ordered[j];
    const dist = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
    adj[i].push({ j, bondIdx, dist });
    adj[j].push({ j: i, bondIdx, dist });
  });

  // Geometric bond-order perception for PDB HETATMs (which lack explicit bond orders):
  // 1. Detect planar 5- and 6-membered conjugated rings -> mark ring bonds as aromatic (order 4)
  const heavyDeg = ordered.map((_, i) => adj[i].filter((e) => ordered[e.j].element !== 'H').length);
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
          // Check approximate planarity & sp2 bond lengths (1.28–1.46 Å for C/N/O, <=1.78 Å with S)
          const hasOnlyRingEls = path.every((idx) => ['C', 'N', 'O', 'S'].includes(ordered[idx].element));
          const maxBondLen = Math.max(...cycleEdges.map((bi) => {
            const [u, v] = bonds[bi];
            return Math.hypot(
              ordered[u].x - ordered[v].x,
              ordered[u].y - ordered[v].y,
              ordered[u].z - ordered[v].z,
            );
          }));
          const hasSulfur = path.some((idx) => ordered[idx].element === 'S');
          const allLowDeg = path.every((idx) => heavyDeg[idx] <= 3);
          if (hasOnlyRingEls && allLowDeg && maxBondLen <= (hasSulfur ? 1.78 : 1.46)) {
            // Check planarity via normal consistency across consecutive triples
            let planar = true;
            const p0 = ordered[path[0]];
            const p1 = ordered[path[1]];
            const p2 = ordered[path[2]];
            const ux = p1.x - p0.x, uy = p1.y - p0.y, uz = p1.z - p0.z;
            const vx = p2.x - p0.x, vy = p2.y - p0.y, vz = p2.z - p0.z;
            let nx = uy * vz - uz * vy;
            let ny = uz * vx - ux * vz;
            let nz = ux * vy - uy * vx;
            const nLen = Math.hypot(nx, ny, nz);
            if (nLen > 1e-4) {
              nx /= nLen; ny /= nLen; nz /= nLen;
              for (let k = 3; k < path.length; k++) {
                const pk = ordered[path[k]];
                const dPlane = Math.abs((pk.x - p0.x) * nx + (pk.y - p0.y) * ny + (pk.z - p0.z) * nz);
                if (dPlane > 0.18) {
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

  // 2. Perceive short exocyclic C=O carbonyls / carboxylates & P=O / S=O
  for (let i = 0; i < ordered.length; i++) {
    const a = ordered[i];
    if (a.element !== 'O' || heavyDeg[i] !== 1) continue;
    const edge = adj[i].find((e) => ordered[e.j].element !== 'H');
    if (!edge) continue;
    const parent = ordered[edge.j];
    if (parent.element === 'C' && edge.dist <= 1.29 && heavyDeg[edge.j] <= 3) {
      // Check if carboxylate (two terminal oxygens on same carbon): mark one C=O and one C-O-
      const termO = adj[edge.j].filter(
        (e) => ordered[e.j].element === 'O' && heavyDeg[e.j] === 1 && e.dist <= 1.32,
      );
      if (termO.length === 2) {
        const sorted = [...termO].sort((x, y) => x.dist - y.dist);
        bonds[sorted[0].bondIdx][2] = 2;
      } else {
        bonds[edge.bondIdx][2] = 2;
      }
    } else if ((parent.element === 'P' || parent.element === 'S') && edge.dist <= 1.52) {
      bonds[edge.bondIdx][2] = 2;
    }
  }

  const padL = (s: string, w: number) => (s.length >= w ? s.slice(0, w) : ' '.repeat(w - s.length) + s);
  const padR = (s: string, w: number) => (s.length >= w ? s.slice(0, w) : s + ' '.repeat(w - s.length));

  const out: string[] = [
    title.slice(0, 60),
    '  denovo.run GNINA 1.3 prep',
    '',
    `${padL(String(ordered.length), 3)}${padL(String(bonds.length), 3)}  0  0  0  0  0  0  0  0999 V2000`,
  ];

  const heavyLineIndices: number[] = [];
  const chargeEntries: Array<[number, number]> = [];
  for (let i = 0; i < ordered.length; i++) {
    const a = ordered[i];
    if (a.element !== 'H') heavyLineIndices.push(a.lineIndex);
    if (a.charge !== 0) chargeEntries.push([i + 1, a.charge]);
    const el = padR(a.element, 3);
    out.push(
      `${padL(a.x.toFixed(4), 10)}${padL(a.y.toFixed(4), 10)}${padL(a.z.toFixed(4), 10)} ${el} 0  0  0  0  0  0  0  0  0  0  0  0`,
    );
  }
  for (const [i, j, order] of bonds) {
    out.push(`${padL(String(i + 1), 3)}${padL(String(j + 1), 3)}${padL(String(order), 3)}  0  0  0  0`);
  }
  if (chargeEntries.length > 0) {
    for (let k = 0; k < chargeEntries.length; k += 8) {
      const chunk = chargeEntries.slice(k, k + 8);
      const body = chunk.map(([idx, chg]) => ` ${padL(String(idx), 3)} ${padL(String(chg), 3)}`).join('');
      out.push(`M  CHG${padL(String(chunk.length), 3)}${body}`);
    }
  }
  out.push('M  END', '$$$$', '');

  return { sdf: out.join('\n'), heavyLineIndices };
}

/**
 * Split a holo PDB or mmCIF structure into a GNINA-ready receptor PDB (protein + modified AAs +
 * optional monoatomic metal cofactors) and detected organic ligand instance(s).
 */
export function extractHoloReceptorAndLigands(
  structText: string,
  opts?: { ligandCode?: string; includeMetals?: boolean },
): GninaComplexPrep {
  const normalizedPdb = normalizeToPdbText(structText);
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
  for (const [instanceKey, bucket] of ligandBuckets.entries()) {
    const heavy = bucket.filter((a) => a.element !== 'H');
    if (heavy.length === 0) continue;
    const first = bucket[0];
    detectedLigands.push({
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
    });
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
          `${d.chain}:${d.resSeq}${d.insCode}`.toUpperCase() === requestedRaw ||
          d.instanceKey.toUpperCase() === requestedRaw,
      ) ?? null;
  } else {
    selectedLigand = detectedLigands[0] ?? null;
  }

  // Keep any *non-selected* organic ligands out of the receptor when they are symmetric copies
  // on other chains, or keep them? Standard docking scores the selected pose in its binding pocket
  // without other copies of the same ligand interfering.
  const receptorPdb = receptorLines.length > 0 ? `${receptorLines.join('\n')}\nEND\n` : '';
  let selectedLigandSdf: string | null = null;
  if (selectedLigand) {
    const built = buildLigandV2000Sdf(
      selectedLigand.atoms,
      conectPairs,
      `${selectedLigand.compId}_${selectedLigand.chain}${selectedLigand.resSeq}${selectedLigand.insCode}`,
    );
    selectedLigandSdf = built.sdf;
    selectedLigand.heavyLineIndices = built.heavyLineIndices;
  }

  return {
    normalizedPdb,
    receptorPdb,
    detectedLigands,
    selectedLigand,
    selectedLigandSdf,
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
 * into GNINA's exact `pocketResidues` keys (`[chain, seq, ins, res].filter(Boolean).join('/')`).
 */
export function parsePocketResidueTokens(spec: string, receptorPdb: string): string[] {
  const trimmed = spec.trim();
  if (!trimmed) return [];
  const { atoms } = parseAllPdbAtomLines(receptorPdb);
  const resByChainNum = new Map<string, string>();
  const resByNumOnly = new Map<string, string[]>();
  const validKeys = new Set<string>();

  for (const a of atoms) {
    if (a.record === 'HETATM' && classifyHetatmCompId(a.resName) !== 'modified-amino-acid') {
      continue;
    }
    const key = [a.chain, a.resSeq, a.insCode, a.resName].filter(Boolean).join('/');
    if (validKeys.has(key)) continue;
    validKeys.add(key);
    resByChainNum.set(`${a.chain.toUpperCase()}:${a.resSeq}${a.insCode.toUpperCase()}`, key);
    const list = resByNumOnly.get(a.resSeq) ?? [];
    list.push(key);
    resByNumOnly.set(a.resSeq, list);
  }

  const out = new Set<string>();
  const tokens = trimmed.split(/[\s,;]+/).filter(Boolean);
  for (const rawTok of tokens) {
    const tok = rawTok.trim();
    if (validKeys.has(tok)) {
      out.add(tok);
      continue;
    }
    // Range with chain prefix: A45-52 or A:45-52
    const rangeMatch = tok.match(/^([A-Za-z]):?(\d+)-(\d+)$/);
    if (rangeMatch) {
      const ch = rangeMatch[1].toUpperCase();
      const lo = Number(rangeMatch[2]);
      const hi = Number(rangeMatch[3]);
      for (let n = Math.min(lo, hi); n <= Math.max(lo, hi); n++) {
        const hit = resByChainNum.get(`${ch}:${n}`);
        if (hit) out.add(hit);
      }
      continue;
    }
    // Single residue with chain prefix: A45, A:45, A100A
    const singleChain = tok.match(/^([A-Za-z]):?(\d+)([A-Za-z]?)$/);
    if (singleChain) {
      const hit = resByChainNum.get(
        `${singleChain[1].toUpperCase()}:${singleChain[2]}${singleChain[3].toUpperCase()}`,
      );
      if (hit) out.add(hit);
      continue;
    }
    // Bare residue number or range (match first/all chains)
    const bareRange = tok.match(/^(\d+)-(\d+)$/);
    if (bareRange) {
      const lo = Number(bareRange[1]);
      const hi = Number(bareRange[2]);
      for (let n = Math.min(lo, hi); n <= Math.max(lo, hi); n++) {
        for (const hit of resByNumOnly.get(String(n)) ?? []) out.add(hit);
      }
      continue;
    }
    if (/^\d+$/.test(tok)) {
      for (const hit of resByNumOnly.get(tok) ?? []) out.add(hit);
    }
  }
  return [...out];
}

/**
 * Build a multi-model PDB (`MODEL 1`..`MODEL K`) combining the receptor with each ranked docked pose
 * so Mol*'s frame scrubber lets users inspect Pose #1..#K interactively in 3D.
 */
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
