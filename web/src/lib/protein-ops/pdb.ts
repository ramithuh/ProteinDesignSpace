// PDB & mmCIF utilities for protein backbones and all-atom structures.
// Supports 3-atom (N,CA,C), 4-atom (N,CA,C,O with deterministic peptide-plane O
// reconstruction), AF2 atom37 all-atom export with pLDDT B-factors, and PDB/mmCIF parsing.

const ATOM_NAMES_3 = ['N', 'CA', 'C'] as const;
const ATOM_NAMES_4 = ['N', 'CA', 'C', 'O'] as const;

export const AA1_TO_AA3: Record<string, string> = {
  A: 'ALA', C: 'CYS', D: 'ASP', E: 'GLU', F: 'PHE',
  G: 'GLY', H: 'HIS', I: 'ILE', K: 'LYS', L: 'LEU',
  M: 'MET', N: 'ASN', P: 'PRO', Q: 'GLN', R: 'ARG',
  S: 'SER', T: 'THR', V: 'VAL', W: 'TRP', Y: 'TYR',
  X: 'UNK',
};

export const AA3_TO_AA1: Record<string, string> = Object.fromEntries(
  Object.entries(AA1_TO_AA3).map(([k, v]) => [v, k]),
);

// Standard AlphaFold2 37-atom order (identical to openfold / alphafold / localfold)
export const AF2_ATOM37_NAMES = [
  'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG',
  'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1',
  'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2',
  'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT',
] as const;

function pad(s: string, n: number): string {
  return s.length >= n ? s : ' '.repeat(n - s.length) + s;
}
function padR(s: string, n: number): string {
  return s.length >= n ? s : s + ' '.repeat(n - s.length);
}

/**
 * Reconstruct planar peptide carbonyl oxygen O from 3-atom [L * 3 * 3] (N, CA, C)
 * coordinates into 4-atom [L * 4 * 3] (N, CA, C, O).
 * Uses the ideal peptide plane defined by (CA_i, C_i, N_{i+1}) with C=O = 1.231 Å
 * and angle CA-C-O = 120.8° (falls back to (N_i, CA_i, C_i) on the C-terminal residue).
 */
export function reconstructBackboneOxygen(ncac: Float32Array, nResidues: number): Float32Array {
  const out = new Float32Array(nResidues * 4 * 3);
  const CO_LEN = 1.231;

  for (let r = 0; r < nResidues; r++) {
    const nOff = (r * 3 + 0) * 3;
    const caOff = (r * 3 + 1) * 3;
    const cOff = (r * 3 + 2) * 3;

    const nx = ncac[nOff], ny = ncac[nOff + 1], nz = ncac[nOff + 2];
    const cax = ncac[caOff], cay = ncac[caOff + 1], caz = ncac[caOff + 2];
    const cx = ncac[cOff], cy = ncac[cOff + 1], cz = ncac[cOff + 2];

    const dst = r * 4 * 3;
    out[dst + 0] = nx; out[dst + 1] = ny; out[dst + 2] = nz;
    out[dst + 3] = cax; out[dst + 4] = cay; out[dst + 5] = caz;
    out[dst + 6] = cx; out[dst + 7] = cy; out[dst + 8] = cz;

    // Unit vector from C -> CA
    let v1x = cax - cx, v1y = cay - cy, v1z = caz - cz;
    const l1 = Math.hypot(v1x, v1y, v1z) || 1;
    v1x /= l1; v1y /= l1; v1z /= l1;

    let ox: number, oy: number, oz: number;
    if (r + 1 < nResidues) {
      // Peptide plane bisector opposite to CA_i and N_{i+1}
      const nextNOff = ((r + 1) * 3 + 0) * 3;
      let v2x = ncac[nextNOff] - cx;
      let v2y = ncac[nextNOff + 1] - cy;
      let v2z = ncac[nextNOff + 2] - cz;
      const l2 = Math.hypot(v2x, v2y, v2z) || 1;
      v2x /= l2; v2y /= l2; v2z /= l2;

      let bx = -(v1x + v2x);
      let by = -(v1y + v2y);
      let bz = -(v1z + v2z);
      const bl = Math.hypot(bx, by, bz) || 1;
      ox = cx + (bx / bl) * CO_LEN;
      oy = cy + (by / bl) * CO_LEN;
      oz = cz + (bz / bl) * CO_LEN;
    } else {
      // C-terminal fallback: place O in (N, CA, C) plane at ~120.8° from C->CA
      let u2x = nx - cax, u2y = ny - cay, u2z = nz - caz;
      const dot = u2x * v1x + u2y * v1y + u2z * v1z;
      u2x -= dot * v1x; u2y -= dot * v1y; u2z -= dot * v1z;
      const ul = Math.hypot(u2x, u2y, u2z) || 1;
      u2x /= ul; u2y /= ul; u2z /= ul;
      const cosA = Math.cos((120.8 * Math.PI) / 180);
      const sinA = Math.sin((120.8 * Math.PI) / 180);
      ox = cx + CO_LEN * (cosA * v1x + sinA * u2x);
      oy = cy + CO_LEN * (cosA * v1y + sinA * u2y);
      oz = cz + CO_LEN * (cosA * v1z + sinA * u2z);
    }

    out[dst + 9] = ox; out[dst + 10] = oy; out[dst + 11] = oz;
  }
  return out;
}

/** Extract Cα coordinates [L * 3] from 3-atom or 4-atom backbone coords. */
export function extractCaFromBackbone(
  coords: Float32Array,
  nResidues: number,
  atomsPerRes: 3 | 4,
): Float32Array {
  const ca = new Float32Array(nResidues * 3);
  for (let r = 0; r < nResidues; r++) {
    const off = (r * atomsPerRes + 1) * 3;
    ca[r * 3 + 0] = coords[off + 0];
    ca[r * 3 + 1] = coords[off + 1];
    ca[r * 3 + 2] = coords[off + 2];
  }
  return ca;
}

/**
 * Convert backbone coords [nResidues * atomsPerRes, 3] into a single-model PDB string.
 */
export function coordsToBackbonePdb(
  coords: Float32Array,
  nResidues: number,
  opts: {
    atomsPerRes?: 3 | 4;
    sequence?: string;
    bFactors?: Float32Array;
    chainId?: string;
  } = {},
): string {
  const atomsPerRes = opts.atomsPerRes ?? 3;
  const names = atomsPerRes === 4 ? ATOM_NAMES_4 : ATOM_NAMES_3;
  const chain = (opts.chainId ?? 'A')[0] ?? 'A';
  const lines: string[] = [];
  let atomIdx = 1;

  for (let r = 0; r < nResidues; r++) {
    const aa1 = opts.sequence?.[r]?.toUpperCase() ?? 'G';
    const resName = AA1_TO_AA3[aa1] ?? 'GLY';
    const bVal = opts.bFactors ? opts.bFactors[r] : 0.0;
    for (let a = 0; a < atomsPerRes; a++) {
      const flatIdx = r * atomsPerRes + a;
      const x = coords[flatIdx * 3 + 0];
      const y = coords[flatIdx * 3 + 1];
      const z = coords[flatIdx * 3 + 2];
      const name = names[a];
      const element = name[0];
      lines.push(
        `ATOM  ${pad(String(atomIdx), 5)}  ${padR(name, 3)} ${resName} ${chain}${pad(String(r + 1), 4)}    ` +
          `${pad(x.toFixed(3), 8)}${pad(y.toFixed(3), 8)}${pad(z.toFixed(3), 8)}` +
          `  1.00${pad(bVal.toFixed(2), 6)}           ${element}  `,
      );
      atomIdx++;
    }
  }
  lines.push('END');
  return lines.join('\n') + '\n';
}

/** Backward-compatible 3-atom poly-GLY PDB writer. */
export function coordsToPolyGlyPdb(coords: Float32Array, nResidues: number): string {
  return coordsToBackbonePdb(coords, nResidues, { atomsPerRes: 3 });
}

/**
 * Convert AlphaFold2 atom37 coordinates [L * 37 * 3] + mask [L * 37] + sequence + pLDDT [L]
 * into an all-atom PDB string (with pLDDT written to the B-factor column in 0..100 scale).
 */
export function atom37ToPdb(
  atom37: Float32Array,
  atom37Mask: Float32Array,
  sequence: string,
  plddt?: Float32Array,
  chainId = 'A',
): { pdbText: string; caCoords: Float32Array; backbone4Coords: Float32Array } {
  const L = sequence.length;
  const chain = chainId[0] ?? 'A';
  const lines: string[] = [];
  const caCoords = new Float32Array(L * 3);
  const backbone4Coords = new Float32Array(L * 4 * 3);
  let atomIdx = 1;

  for (let r = 0; r < L; r++) {
    const aa1 = sequence[r].toUpperCase();
    const resName = AA1_TO_AA3[aa1] ?? 'ALA';
    const bVal = plddt ? plddt[r] : 0.0;

    // Extract N(0), CA(1), C(2), O(4) for canonical backbone4
    const slots = [0, 1, 2, 4];
    for (let k = 0; k < 4; k++) {
      const s = slots[k];
      const src = (r * 37 + s) * 3;
      const dst = (r * 4 + k) * 3;
      backbone4Coords[dst + 0] = atom37[src + 0];
      backbone4Coords[dst + 1] = atom37[src + 1];
      backbone4Coords[dst + 2] = atom37[src + 2];
    }
    caCoords[r * 3 + 0] = atom37[(r * 37 + 1) * 3 + 0];
    caCoords[r * 3 + 1] = atom37[(r * 37 + 1) * 3 + 1];
    caCoords[r * 3 + 2] = atom37[(r * 37 + 1) * 3 + 2];

    for (let a = 0; a < 37; a++) {
      if (atom37Mask[r * 37 + a] < 0.5) continue;
      const name = AF2_ATOM37_NAMES[a];
      const src = (r * 37 + a) * 3;
      const x = atom37[src + 0];
      const y = atom37[src + 1];
      const z = atom37[src + 2];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      const atomCol = name.length === 4 ? name : ` ${padR(name, 3)}`;
      const element = name[0];
      lines.push(
        `ATOM  ${pad(String(atomIdx), 5)} ${atomCol} ${resName} ${chain}${pad(String(r + 1), 4)}    ` +
          `${pad(x.toFixed(3), 8)}${pad(y.toFixed(3), 8)}${pad(z.toFixed(3), 8)}` +
          `  1.00${pad(bVal.toFixed(2), 6)}           ${element}  `,
      );
      atomIdx++;
    }
  }
  lines.push('END');
  return { pdbText: lines.join('\n') + '\n', caCoords, backbone4Coords };
}

/**
 * Build a multi-model PDB from a trajectory of per-frame coord buffers.
 * MODEL 1 (0-based index 0) = FINAL frame for Mol* topology inference.
 * MODELs 2..T+1 (0-based indices 1..T) = frames 0..T-1 in order.
 */
export function coordsToMultiModelPdb(
  trajectory: Float32Array[],
  nResidues: number,
  atomsPerRes: 3 | 4 = 3,
): string {
  const chunks: string[] = [];
  const terSerial = nResidues * atomsPerRes + 1;
  const terLine = `TER   ${pad(String(terSerial), 5)}      GLY A${pad(String(nResidues), 4)}`;
  const T = trajectory.length;
  const order: number[] = [T - 1];
  for (let i = 0; i < T; i++) order.push(i);

  let modelIdx = 1;
  for (const src of order) {
    const body = coordsToBackbonePdb(trajectory[src], nResidues, { atomsPerRes })
      .replace(/END\s*$/, '')
      .replace(/\s+$/, '');
    chunks.push(`MODEL     ${modelIdx}`);
    chunks.push(body);
    chunks.push(terLine);
    chunks.push('ENDMDL');
    modelIdx++;
  }
  chunks.push('END');
  return chunks.join('\n') + '\n';
}

export interface ParsedStructure {
  sequence: string;
  nResidues: number;
  caCoords: Float32Array;
  backbone4Coords: Float32Array;
  chainIds: string[];
  pdbText: string;
  format: 'pdb' | 'mmcif';
}

/**
 * Parse a PDB (or basic mmCIF `_atom_site`) string into canonical backbone arrays
 * (`caCoords [L*3]` and `backbone4Coords [L*4*3]` with missing O reconstructed).
 * For multi-model PDBs, parses MODEL 1.
 */
export function parseStructureText(text: string): ParsedStructure {
  const trimmed = text.trimStart();
  if (trimmed.startsWith('data_') || trimmed.includes('_atom_site.')) {
    return parseMmcifBackbone(text);
  }
  return parsePdbBackbone(text);
}

export function parsePdbBackbone(pdbText: string): ParsedStructure {
  interface ResAccum {
    chain: string;
    resKey: string;
    aa1: string;
    atoms: Record<string, [number, number, number]>;
  }
  const residues: ResAccum[] = [];
  const byKey = new Map<string, ResAccum>();

  const lines = pdbText.split('\n');
  for (const line of lines) {
    if (line.startsWith('ENDMDL') && residues.length > 0) break;
    if (!line.startsWith('ATOM  ') || line.length < 54) continue;
    const altLoc = line[16];
    if (altLoc !== ' ' && altLoc !== 'A' && altLoc !== '1') continue;
    const atomName = line.slice(12, 16).trim();
    if (atomName !== 'N' && atomName !== 'CA' && atomName !== 'C' && atomName !== 'O') continue;

    const resName = line.slice(17, 20).trim();
    const chain = line[21]?.trim() || 'A';
    const resSeq = line.slice(22, 27).trim();
    const key = `${chain}:${resSeq}`;
    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    let entry = byKey.get(key);
    if (!entry) {
      entry = { chain, resKey: key, aa1: AA3_TO_AA1[resName] ?? 'G', atoms: {} };
      byKey.set(key, entry);
      residues.push(entry);
    }
    entry.atoms[atomName] = [x, y, z];
  }

  // Keep residues with at least N, CA, C (or CA alone)
  const valid = residues.filter((r) => r.atoms.CA !== undefined);
  const L = valid.length;
  const caCoords = new Float32Array(L * 3);
  const ncac = new Float32Array(L * 3 * 3);
  let hasAllO = L > 0;
  let seq = '';
  const chainIds: string[] = [];

  for (let i = 0; i < L; i++) {
    const r = valid[i];
    seq += r.aa1;
    chainIds.push(r.chain);
    const ca = r.atoms.CA;
    const n = r.atoms.N ?? ca;
    const c = r.atoms.C ?? ca;
    caCoords[i * 3 + 0] = ca[0]; caCoords[i * 3 + 1] = ca[1]; caCoords[i * 3 + 2] = ca[2];
    ncac[(i * 3 + 0) * 3 + 0] = n[0]; ncac[(i * 3 + 0) * 3 + 1] = n[1]; ncac[(i * 3 + 0) * 3 + 2] = n[2];
    ncac[(i * 3 + 1) * 3 + 0] = ca[0]; ncac[(i * 3 + 1) * 3 + 1] = ca[1]; ncac[(i * 3 + 1) * 3 + 2] = ca[2];
    ncac[(i * 3 + 2) * 3 + 0] = c[0]; ncac[(i * 3 + 2) * 3 + 1] = c[1]; ncac[(i * 3 + 2) * 3 + 2] = c[2];
    if (!r.atoms.O) hasAllO = false;
  }

  let backbone4Coords: Float32Array;
  if (hasAllO) {
    backbone4Coords = new Float32Array(L * 4 * 3);
    for (let i = 0; i < L; i++) {
      const r = valid[i];
      const arr = [r.atoms.N ?? r.atoms.CA, r.atoms.CA, r.atoms.C ?? r.atoms.CA, r.atoms.O];
      for (let a = 0; a < 4; a++) {
        backbone4Coords[(i * 4 + a) * 3 + 0] = arr[a][0];
        backbone4Coords[(i * 4 + a) * 3 + 1] = arr[a][1];
        backbone4Coords[(i * 4 + a) * 3 + 2] = arr[a][2];
      }
    }
  } else {
    backbone4Coords = reconstructBackboneOxygen(ncac, L);
  }

  const normalizedPdb = hasAllO
    ? pdbText
    : coordsToBackbonePdb(backbone4Coords, L, { atomsPerRes: 4, sequence: seq });

  return {
    sequence: seq,
    nResidues: L,
    caCoords,
    backbone4Coords,
    chainIds,
    pdbText: normalizedPdb,
    format: 'pdb',
  };
}

function parseMmcifBackbone(cifText: string): ParsedStructure {
  const lines = cifText.split('\n');
  const cols: string[] = [];
  let inAtomSite = false;
  const residues: Array<{ chain: string; aa1: string; atoms: Record<string, [number, number, number]> }> = [];
  const byKey = new Map<string, { chain: string; aa1: string; atoms: Record<string, [number, number, number]> }>();

  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith('_atom_site.')) {
      inAtomSite = true;
      cols.push(line.slice('_atom_site.'.length));
      continue;
    }
    if (inAtomSite && (line === '' || line.startsWith('#') || line.startsWith('loop_') || line.startsWith('_'))) {
      if (residues.length > 0) break;
      continue;
    }
    if (!inAtomSite) continue;
    const toks = line.split(/\s+/);
    if (toks.length < cols.length) continue;

    const get = (name: string) => {
      const idx = cols.indexOf(name);
      return idx >= 0 ? toks[idx] : '';
    };
    const group = get('group_PDB');
    if (group && group !== 'ATOM') continue;
    const atomName = get('label_atom_id') || get('auth_atom_id');
    if (atomName !== 'N' && atomName !== 'CA' && atomName !== 'C' && atomName !== 'O') continue;
    const chain = get('auth_asym_id') || get('label_asym_id') || 'A';
    const seqId = get('auth_seq_id') || get('label_seq_id');
    const comp = get('label_comp_id') || get('auth_comp_id') || 'GLY';
    const x = parseFloat(get('Cartn_x'));
    const y = parseFloat(get('Cartn_y'));
    const z = parseFloat(get('Cartn_z'));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    const key = `${chain}:${seqId}`;
    let entry = byKey.get(key);
    if (!entry) {
      entry = { chain, aa1: AA3_TO_AA1[comp] ?? 'G', atoms: {} };
      byKey.set(key, entry);
      residues.push(entry);
    }
    entry.atoms[atomName] = [x, y, z];
  }

  const valid = residues.filter((r) => r.atoms.CA !== undefined);
  const L = valid.length;
  const caCoords = new Float32Array(L * 3);
  const ncac = new Float32Array(L * 3 * 3);
  let seq = '';
  const chainIds: string[] = [];
  for (let i = 0; i < L; i++) {
    const r = valid[i];
    seq += r.aa1;
    chainIds.push(r.chain);
    const ca = r.atoms.CA;
    const n = r.atoms.N ?? ca;
    const c = r.atoms.C ?? ca;
    caCoords[i * 3 + 0] = ca[0]; caCoords[i * 3 + 1] = ca[1]; caCoords[i * 3 + 2] = ca[2];
    ncac[(i * 3 + 0) * 3 + 0] = n[0]; ncac[(i * 3 + 0) * 3 + 1] = n[1]; ncac[(i * 3 + 0) * 3 + 2] = n[2];
    ncac[(i * 3 + 1) * 3 + 0] = ca[0]; ncac[(i * 3 + 1) * 3 + 1] = ca[1]; ncac[(i * 3 + 1) * 3 + 2] = ca[2];
    ncac[(i * 3 + 2) * 3 + 0] = c[0]; ncac[(i * 3 + 2) * 3 + 1] = c[1]; ncac[(i * 3 + 2) * 3 + 2] = c[2];
  }
  const backbone4Coords = reconstructBackboneOxygen(ncac, L);
  const pdbText = coordsToBackbonePdb(backbone4Coords, L, { atomsPerRes: 4, sequence: seq });
  return { sequence: seq, nResidues: L, caCoords, backbone4Coords, chainIds, pdbText, format: 'mmcif' };
}

/**
 * Generates an ideal 56-residue de novo 3-helix bundle backbone (clean, designable
 * geometry with proper N/CA/C/O peptide planes) so users can test the full
 * ProteinMPNN -> LocalFold AF2 -> Kabsch scRMSD pipeline in 1 click without
 * waiting for a generator or uploading a file.
 */
export function generateSampleHelixBundlePdb(): string {
  // Build 3 antiparallel ideal α-helices (16 aa each) joined by short 4-aa loops = 56 aa.
  const L = 56;
  const ncac = new Float32Array(L * 3 * 3);

  const helixSpec = [
    { start: 0, len: 16, cx: -5.2, cy: 0.0, z0: -12.0, dir: 1 },
    { start: 20, len: 16, cx: 5.2, cy: 2.2, z0: 12.0, dir: -1 },
    { start: 40, len: 16, cx: 0.0, cy: -5.8, z0: -12.0, dir: 1 },
  ];

  const caPos: Array<[number, number, number]> = new Array(L);
  const radius = 2.3;
  const rise = 1.5;
  const twist = (100 * Math.PI) / 180;

  for (const h of helixSpec) {
    for (let k = 0; k < h.len; k++) {
      const ang = k * twist * h.dir;
      caPos[h.start + k] = [
        h.cx + radius * Math.cos(ang),
        h.cy + radius * Math.sin(ang),
        h.z0 + k * rise * h.dir,
      ];
    }
  }
  // Smooth cubic-like arches for the 4-residue loops (16..19 and 36..39)
  const fillLoop = (s: number, e: number, zBump: number) => {
    const p0 = caPos[s - 1];
    const p1 = caPos[e];
    const n = e - s;
    for (let k = 0; k < n; k++) {
      const u = (k + 1) / (n + 1);
      const arch = Math.sin(Math.PI * u) * zBump;
      caPos[s + k] = [
        p0[0] * (1 - u) + p1[0] * u,
        p0[1] * (1 - u) + p1[1] * u,
        p0[2] * (1 - u) + p1[2] * u + arch,
      ];
    }
  };
  fillLoop(16, 20, 3.2);
  fillLoop(36, 40, -3.2);

  // Place N and C along local backbone tangents with realistic N-CA (1.46Å) and CA-C (1.52Å)
  for (let i = 0; i < L; i++) {
    const ca = caPos[i];
    const prev = caPos[Math.max(0, i - 1)];
    const next = caPos[Math.min(L - 1, i + 1)];
    let tx = next[0] - prev[0], ty = next[1] - prev[1], tz = next[2] - prev[2];
    const tl = Math.hypot(tx, ty, tz) || 1;
    tx /= tl; ty /= tl; tz /= tl;

    // Slight radial offset so N-CA-C angle is ~111° instead of collinear
    let rx = -ty, ry = tx, rz = 0.25;
    const rl = Math.hypot(rx, ry, rz) || 1;
    rx /= rl; ry /= rl; rz /= rl;

    const nx = ca[0] - 1.20 * tx + 0.82 * rx;
    const ny = ca[1] - 1.20 * ty + 0.82 * ry;
    const nz = ca[2] - 1.20 * tz + 0.82 * rz;

    const cx = ca[0] + 1.25 * tx + 0.85 * rx;
    const cy = ca[1] + 1.25 * ty + 0.85 * ry;
    const cz = ca[2] + 1.25 * tz + 0.85 * rz;

    ncac[(i * 3 + 0) * 3 + 0] = nx; ncac[(i * 3 + 0) * 3 + 1] = ny; ncac[(i * 3 + 0) * 3 + 2] = nz;
    ncac[(i * 3 + 1) * 3 + 0] = ca[0]; ncac[(i * 3 + 1) * 3 + 1] = ca[1]; ncac[(i * 3 + 1) * 3 + 2] = ca[2];
    ncac[(i * 3 + 2) * 3 + 0] = cx; ncac[(i * 3 + 2) * 3 + 1] = cy; ncac[(i * 3 + 2) * 3 + 2] = cz;
  }

  const bb4 = reconstructBackboneOxygen(ncac, L);
  return coordsToBackbonePdb(bb4, L, { atomsPerRes: 4, sequence: 'A'.repeat(L) });
}
