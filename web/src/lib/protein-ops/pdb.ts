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

export const AA3_TO_AA1: Record<string, string> = {
  ...Object.fromEntries(Object.entries(AA1_TO_AA3).map(([k, v]) => [v, k])),
  MSE: 'M',
};

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
 * and angle CA-C-O = 120.8° (falls back to (N_i, CA_i, C_i) on C-termini, chain breaks,
 * or whenever d(C_i, N_{i+1}) >= 2.0 Å).
 */
export function reconstructBackboneOxygen(
  ncac: Float32Array,
  nResidues: number,
  chainIds?: string[],
): Float32Array {
  const out = new Float32Array(nResidues * 4 * 3);
  const CO_LEN = 1.231;
  const hasChains = Boolean(chainIds && chainIds.length === nResidues);

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
    let usedPeptidePlane = false;
    if (r + 1 < nResidues && (!hasChains || chainIds![r] === chainIds![r + 1])) {
      // Peptide plane bisector opposite to CA_i and N_{i+1} (only if peptide bond < 2.0 Å)
      const nextNOff = ((r + 1) * 3 + 0) * 3;
      const dx = ncac[nextNOff] - cx;
      const dy = ncac[nextNOff + 1] - cy;
      const dz = ncac[nextNOff + 2] - cz;
      const l2 = Math.hypot(dx, dy, dz);
      if (l2 > 1e-6 && l2 < 2.0) {
        const v2x = dx / l2;
        const v2y = dy / l2;
        const v2z = dz / l2;
        const bx = -(v1x + v2x);
        const by = -(v1y + v2y);
        const bz = -(v1z + v2z);
        const bl = Math.hypot(bx, by, bz);
        if (bl > 1e-6) {
          ox = cx + (bx / bl) * CO_LEN;
          oy = cy + (by / bl) * CO_LEN;
          oz = cz + (bz / bl) * CO_LEN;
          usedPeptidePlane = true;
        }
      }
    }

    if (!usedPeptidePlane) {
      // C-terminal / chain-break fallback: place O in (N, CA, C) plane at ~120.8° from C->CA
      let u2x = nx - cax, u2y = ny - cay, u2z = nz - caz;
      const dot = u2x * v1x + u2y * v1y + u2z * v1z;
      u2x -= dot * v1x; u2y -= dot * v1y; u2z -= dot * v1z;
      let ul = Math.hypot(u2x, u2y, u2z);
      if (ul < 1e-6) {
        // Fallback orthogonal vector to v1
        const bx = Math.abs(v1y) > 0.9 ? 1 : 0;
        const by = Math.abs(v1y) > 0.9 ? 0 : 1;
        u2x = v1y * 0 - v1z * by;
        u2y = v1z * bx - v1x * 0;
        u2z = v1x * by - v1y * bx;
        ul = Math.hypot(u2x, u2y, u2z) || 1;
      }
      u2x /= ul; u2y /= ul; u2z /= ul;
      const cosA = Math.cos((120.8 * Math.PI) / 180);
      const sinA = Math.sin((120.8 * Math.PI) / 180);
      ox = cx + CO_LEN * (cosA * v1x + sinA * u2x);
      oy = cy + CO_LEN * (cosA * v1y + sinA * u2y);
      oz = cz + CO_LEN * (cosA * v1z + sinA * u2z);
    }

    out[dst + 9] = ox!; out[dst + 10] = oy!; out[dst + 11] = oz!;
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

/** Format PDB residue number (cols 23-26) + insertion code (col 27) + 3 padding spaces. */
function formatPdbResSeq(spec: number | string | undefined, fallbackNum: number): string {
  if (spec === undefined || spec === '') {
    return `${pad(String(fallbackNum), 4)}    `;
  }
  const s = String(spec).trim();
  const m = /^(-?\d+)([A-Za-z])?$/.exec(s);
  if (!m) return `${pad(String(fallbackNum), 4)}    `;
  const numPart = m[1].slice(-4);
  const iCode = m[2] ?? ' ';
  return `${pad(numPart, 4)}${iCode}   `;
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
    chainIds?: string[];
    resSeqs?: Array<number | string>;
    caOnly?: boolean;
  } = {},
): string {
  const atomsPerRes = opts.atomsPerRes ?? 3;
  const names = atomsPerRes === 4 ? ATOM_NAMES_4 : ATOM_NAMES_3;
  const defaultChain = (opts.chainId ?? 'A')[0] ?? 'A';
  const lines: string[] = [];
  let atomIdx = 1;
  let prevChain: string | null = null;
  let prevResName = 'GLY';
  let prevResSeqField = `${pad('1', 4)}    `;

  for (let r = 0; r < nResidues; r++) {
    const chain = (opts.chainIds?.[r] ?? defaultChain)[0] ?? defaultChain;
    if (prevChain !== null && chain !== prevChain) {
      lines.push(`TER   ${pad(String(atomIdx), 5)}      ${prevResName} ${prevChain}${prevResSeqField.slice(0, 5)}`);
      atomIdx++;
    }
    const aa1 = opts.sequence?.[r]?.toUpperCase() ?? 'G';
    const resName = AA1_TO_AA3[aa1] ?? 'GLY';
    const bVal = opts.bFactors ? opts.bFactors[r] : 0.0;
    const resSeqField = formatPdbResSeq(opts.resSeqs?.[r], r + 1);
    const atomSlots = opts.caOnly ? [1] : Array.from({ length: atomsPerRes }, (_, i) => i);

    for (const a of atomSlots) {
      const flatIdx = r * atomsPerRes + a;
      const x = coords[flatIdx * 3 + 0];
      const y = coords[flatIdx * 3 + 1];
      const z = coords[flatIdx * 3 + 2];
      const name = names[a];
      const element = name[0];
      lines.push(
        `ATOM  ${pad(String(atomIdx), 5)}  ${padR(name, 3)} ${resName} ${chain}${resSeqField}` +
          `${pad(x.toFixed(3), 8)}${pad(y.toFixed(3), 8)}${pad(z.toFixed(3), 8)}` +
          `  1.00${pad(bVal.toFixed(2), 6)}           ${element}  `,
      );
      atomIdx++;
    }
    prevChain = chain;
    prevResName = resName;
    prevResSeqField = resSeqField;
  }
  lines.push('END');
  return lines.join('\n') + '\n';
}

/** Backward-compatible 3-atom poly-GLY PDB writer. */
export function coordsToPolyGlyPdb(coords: Float32Array, nResidues: number): string {
  return coordsToBackbonePdb(coords, nResidues, { atomsPerRes: 3 });
}

const WATER_RESNAMES = new Set(['HOH', 'WAT', 'DOD', 'H2O', 'TIP', 'TIP3']);
const NUCLEIC_RESNAMES = new Set([
  'DA', 'DC', 'DG', 'DT', 'DU', 'DI',
  'A', 'C', 'G', 'U', 'T', 'I',
  'ADE', 'CYT', 'GUA', 'THY', 'URA',
]);

export interface StructureParseDiagnostics {
  chainCount: number;
  chainLengths: number[];
  hasLigands: boolean;
  ligandNames: string[];
  hasNucleicAcids: boolean;
  nucleicAcidChains: string[];
}

export function computeChainLengthsFromIds(chainIds: readonly string[]): number[] {
  const lengths: number[] = [];
  for (let i = 0; i < chainIds.length; i++) {
    if (i === 0 || chainIds[i] !== chainIds[i - 1]) {
      lengths.push(1);
    } else {
      lengths[lengths.length - 1]++;
    }
  }
  return lengths;
}

export function chainIdsFromLengths(chainLengths: readonly number[]): string[] {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const out: string[] = [];
  for (let c = 0; c < chainLengths.length; c++) {
    const ch = alphabet[c % alphabet.length] ?? 'A';
    for (let k = 0; k < chainLengths[c]; k++) out.push(ch);
  }
  return out;
}

/**
 * Convert AlphaFold2 atom37 coordinates [L * 37 * 3] + mask [L * 37] + sequence + pLDDT [L]
 * into an all-atom PDB string (with pLDDT written to the B-factor column in 0..100 scale).
 * Supports single chainId ('A') or per-residue chainIds[] for multi-chain complexes.
 */
export function atom37ToPdb(
  atom37: Float32Array,
  atom37Mask: Float32Array,
  sequence: string,
  plddt?: Float32Array,
  chainIdOrIds: string | readonly string[] = 'A',
): { pdbText: string; caCoords: Float32Array; backbone4Coords: Float32Array } {
  const L = sequence.length;
  const perResChains =
    Array.isArray(chainIdOrIds) && chainIdOrIds.length === L ? chainIdOrIds : null;
  const defaultChain = typeof chainIdOrIds === 'string' ? (chainIdOrIds[0] ?? 'A') : 'A';
  const lines: string[] = [];
  const caCoords = new Float32Array(L * 3);
  const backbone4Coords = new Float32Array(L * 4 * 3);
  let atomIdx = 1;
  let prevChain: string | null = null;
  let chainResNum = 0;

  for (let r = 0; r < L; r++) {
    const chain = (perResChains ? perResChains[r] : defaultChain)[0] ?? 'A';
    if (prevChain !== null && chain !== prevChain) {
      lines.push(`TER   ${pad(String(atomIdx), 5)}      ${AA1_TO_AA3[sequence[r - 1].toUpperCase()] ?? 'ALA'} ${prevChain}${pad(String(chainResNum), 4)}`);
      atomIdx++;
      chainResNum = 0;
    }
    prevChain = chain;
    chainResNum++;

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
        `ATOM  ${pad(String(atomIdx), 5)} ${atomCol} ${resName} ${chain}${pad(String(chainResNum), 4)}    ` +
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
  if (trajectory.length === 0) return 'END\n';
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
  diagnostics: StructureParseDiagnostics;
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

/** Tokenize a single mmCIF data row while respecting single- and double-quoted strings. */
function tokenizeCifLine(line: string): string[] {
  const toks: string[] = [];
  const n = line.length;
  let i = 0;
  while (i < n) {
    while (i < n && (line[i] === ' ' || line[i] === '\t')) i++;
    if (i >= n) break;
    const ch = line[i];
    if (ch === "'" || ch === '"') {
      const quote = ch;
      i++;
      const start = i;
      while (i < n) {
        if (line[i] === quote && (i + 1 === n || line[i + 1] === ' ' || line[i + 1] === '\t')) {
          break;
        }
        i++;
      }
      toks.push(line.slice(start, i));
      if (i < n && line[i] === quote) i++;
    } else {
      const start = i;
      while (i < n && line[i] !== ' ' && line[i] !== '\t') i++;
      toks.push(line.slice(start, i));
    }
  }
  return toks;
}

export function parsePdbBackbone(pdbText: string): ParsedStructure {
  interface ResAccum {
    chain: string;
    resKey: string;
    aa1: string;
    resName: string;
    resSeqCol: string;
    atoms: Record<string, [number, number, number]>;
  }
  const residues: ResAccum[] = [];
  const byKey = new Map<string, ResAccum>();

  const lines = pdbText.split('\n');
  let maxSerial = 0;
  const ligandSet = new Set<string>();
  const nucleicChains = new Set<string>();
  for (const line of lines) {
    if (line.startsWith('ENDMDL') && residues.length > 0) break;
    if ((line.startsWith('ATOM  ') || line.startsWith('HETATM')) && line.length >= 11) {
      const serial = parseInt(line.slice(6, 11).trim(), 10);
      if (Number.isFinite(serial) && serial > maxSerial) maxSerial = serial;
    }
    if (line.startsWith('HETATM') && line.length >= 22) {
      const hetRes = line.slice(17, 20).trim().toUpperCase();
      if (hetRes && !WATER_RESNAMES.has(hetRes) && !AA3_TO_AA1[hetRes]) {
        if (NUCLEIC_RESNAMES.has(hetRes)) {
          nucleicChains.add(line[21]?.trim() || 'A');
        } else {
          ligandSet.add(hetRes);
        }
      }
      continue;
    }
    if (!line.startsWith('ATOM  ') || line.length < 54) continue;
    const rawResName = line.slice(17, 20).trim().toUpperCase();
    if (NUCLEIC_RESNAMES.has(rawResName)) {
      nucleicChains.add(line[21]?.trim() || 'A');
      continue;
    }
    const altLoc = line[16];
    if (altLoc !== ' ' && altLoc !== 'A' && altLoc !== '1') continue;
    const atomName = line.slice(12, 16).trim();
    if (atomName !== 'N' && atomName !== 'CA' && atomName !== 'C' && atomName !== 'O') continue;

    const resName = line.slice(17, 20).trim();
    const chain = line[21]?.trim() || 'A';
    const resSeqCol = line.slice(22, 27);
    const resSeq = resSeqCol.trim();
    const key = `${chain}:${resSeq}`;
    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    let entry = byKey.get(key);
    if (!entry) {
      entry = {
        chain,
        resKey: key,
        aa1: AA3_TO_AA1[resName] ?? 'X',
        resName: resName || 'UNK',
        resSeqCol,
        atoms: {},
      };
      byKey.set(key, entry);
      residues.push(entry);
    }
    if (!entry.atoms[atomName]) {
      entry.atoms[atomName] = [x, y, z];
    }
  }

  // Keep residues with at least CA
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

  // Reconstruct O with chain-break guard, then preserve every real parsed O coordinate
  const reconstructed = reconstructBackboneOxygen(ncac, L, chainIds);
  const backbone4Coords = reconstructed;
  const missingOByKey = new Map<string, [number, number, number]>();

  for (let i = 0; i < L; i++) {
    const r = valid[i];
    if (r.atoms.O) {
      backbone4Coords[(i * 4 + 3) * 3 + 0] = r.atoms.O[0];
      backbone4Coords[(i * 4 + 3) * 3 + 1] = r.atoms.O[1];
      backbone4Coords[(i * 4 + 3) * 3 + 2] = r.atoms.O[2];
    } else if (r.atoms.N && r.atoms.C) {
      // Only patch O into PDB when N & C are genuinely present (never synthesize for CA-only inputs)
      missingOByKey.set(r.resKey, [
        backbone4Coords[(i * 4 + 3) * 3 + 0],
        backbone4Coords[(i * 4 + 3) * 3 + 1],
        backbone4Coords[(i * 4 + 3) * 3 + 2],
      ]);
    }
  }

  // Patch only missing O atoms directly into the original PDB so multi-chain IDs,
  // residue numbering, insertion codes, sidechains, and HETATM lines are preserved verbatim.
  let normalizedPdb = pdbText;
  if (missingOByKey.size > 0) {
    const patchedLines: string[] = [];
    let nextSerial = maxSerial;
    let inModel1 = true;
    let seenAtoms = false;
    for (const line of lines) {
      patchedLines.push(line);
      if (inModel1 && line.startsWith('ENDMDL') && seenAtoms) {
        inModel1 = false;
        continue;
      }
      if (!inModel1 || !line.startsWith('ATOM  ') || line.length < 54) continue;
      seenAtoms = true;
      const altLoc = line[16];
      if (altLoc !== ' ' && altLoc !== 'A' && altLoc !== '1') continue;
      const atomName = line.slice(12, 16).trim();
      if (atomName !== 'C') continue;
      const chain = line[21]?.trim() || 'A';
      const resSeq = line.slice(22, 27).trim();
      const key = `${chain}:${resSeq}`;
      const oCoord = missingOByKey.get(key);
      if (oCoord) {
        missingOByKey.delete(key);
        nextSerial = Math.min(99999, nextSerial + 1);
        const resMeta = line.slice(17, 27); // resName + space + chain + resSeq + iCode
        const bFactorTail = line.length >= 66 ? line.slice(54, 66) : '  1.00  0.00';
        patchedLines.push(
          `ATOM  ${pad(String(nextSerial), 5)}  ${padR('O', 3)} ${resMeta}   ` +
            `${pad(oCoord[0].toFixed(3), 8)}${pad(oCoord[1].toFixed(3), 8)}${pad(oCoord[2].toFixed(3), 8)}` +
            `${bFactorTail}           O  `,
        );
      }
    }
    normalizedPdb = patchedLines.join('\n');
  }

  const chainLengths = computeChainLengthsFromIds(chainIds);
  return {
    sequence: seq,
    nResidues: L,
    caCoords,
    backbone4Coords,
    chainIds,
    pdbText: normalizedPdb,
    format: 'pdb',
    diagnostics: {
      chainCount: chainLengths.length,
      chainLengths,
      hasLigands: ligandSet.size > 0,
      ligandNames: [...ligandSet],
      hasNucleicAcids: nucleicChains.size > 0,
      nucleicAcidChains: [...nucleicChains],
    },
  };
}

function parseMmcifBackbone(cifText: string): ParsedStructure {
  interface CifRes {
    chain: string;
    seqId: string;
    insCode: string;
    aa1: string;
    atoms: Record<string, [number, number, number]>;
  }
  const lines = cifText.split('\n');
  const cols: string[] = [];
  let inAtomSite = false;
  let firstModelNum: string | null = null;
  const residues: CifRes[] = [];
  const byKey = new Map<string, CifRes>();

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
    const toks = tokenizeCifLine(line);
    if (toks.length < cols.length) continue;

    const get = (name: string) => {
      const idx = cols.indexOf(name);
      return idx >= 0 ? toks[idx] : '';
    };
    const group = get('group_PDB');
    if (group && group !== 'ATOM') continue;

    // Filter to first model only
    const modelNum = get('pdbx_PDB_model_num');
    if (modelNum && modelNum !== '.' && modelNum !== '?') {
      if (firstModelNum === null) firstModelNum = modelNum;
      else if (modelNum !== firstModelNum) continue;
    }

    // Filter to primary altLoc only
    const altId = get('label_alt_id');
    if (altId && altId !== '.' && altId !== '?' && altId !== 'A' && altId !== '1') continue;

    const atomName = get('label_atom_id') || get('auth_atom_id');
    if (atomName !== 'N' && atomName !== 'CA' && atomName !== 'C' && atomName !== 'O') continue;
    const chain = get('auth_asym_id') || get('label_asym_id') || 'A';
    const seqId = get('auth_seq_id') || get('label_seq_id');
    const rawIns = get('pdbx_PDB_ins_code');
    const insCode = rawIns && rawIns !== '.' && rawIns !== '?' ? rawIns : '';
    const comp = get('label_comp_id') || get('auth_comp_id') || 'UNK';
    const x = parseFloat(get('Cartn_x'));
    const y = parseFloat(get('Cartn_y'));
    const z = parseFloat(get('Cartn_z'));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    const key = `${chain}:${seqId}:${insCode}`;
    let entry = byKey.get(key);
    if (!entry) {
      entry = { chain, seqId, insCode, aa1: AA3_TO_AA1[comp] ?? 'X', atoms: {} };
      byKey.set(key, entry);
      residues.push(entry);
    }
    if (!entry.atoms[atomName]) {
      entry.atoms[atomName] = [x, y, z];
    }
  }

  const valid = residues.filter((r) => r.atoms.CA !== undefined);
  const L = valid.length;
  const caCoords = new Float32Array(L * 3);
  const ncac = new Float32Array(L * 3 * 3);
  let seq = '';
  const chainIds: string[] = [];
  const resSeqs: string[] = [];
  let anyHasNC = false;

  for (let i = 0; i < L; i++) {
    const r = valid[i];
    seq += r.aa1;
    chainIds.push(r.chain);
    resSeqs.push(`${r.seqId}${r.insCode}`);
    if (r.atoms.N && r.atoms.C) anyHasNC = true;
    const ca = r.atoms.CA;
    const n = r.atoms.N ?? ca;
    const c = r.atoms.C ?? ca;
    caCoords[i * 3 + 0] = ca[0]; caCoords[i * 3 + 1] = ca[1]; caCoords[i * 3 + 2] = ca[2];
    ncac[(i * 3 + 0) * 3 + 0] = n[0]; ncac[(i * 3 + 0) * 3 + 1] = n[1]; ncac[(i * 3 + 0) * 3 + 2] = n[2];
    ncac[(i * 3 + 1) * 3 + 0] = ca[0]; ncac[(i * 3 + 1) * 3 + 1] = ca[1]; ncac[(i * 3 + 1) * 3 + 2] = ca[2];
    ncac[(i * 3 + 2) * 3 + 0] = c[0]; ncac[(i * 3 + 2) * 3 + 1] = c[1]; ncac[(i * 3 + 2) * 3 + 2] = c[2];
  }

  const backbone4Coords = reconstructBackboneOxygen(ncac, L, chainIds);
  // Preserve real parsed O coordinates wherever present in the mmCIF
  for (let i = 0; i < L; i++) {
    const o = valid[i].atoms.O;
    if (o) {
      backbone4Coords[(i * 4 + 3) * 3 + 0] = o[0];
      backbone4Coords[(i * 4 + 3) * 3 + 1] = o[1];
      backbone4Coords[(i * 4 + 3) * 3 + 2] = o[2];
    }
  }

  const pdbText = coordsToBackbonePdb(backbone4Coords, L, {
    atomsPerRes: 4,
    sequence: seq,
    chainIds,
    resSeqs,
    caOnly: L > 0 && !anyHasNC,
  });
  // Scan mmCIF for HETATM ligands and nucleic acid residues
  const ligandSet = new Set<string>();
  const nucleicChains = new Set<string>();
  for (const rawLine of cifText.split('\n')) {
    const line = rawLine.trim();
    if (line.startsWith('HETATM')) {
      const toks = tokenizeCifLine(line);
      const comp = (toks[5] ?? '').toUpperCase();
      if (comp && !WATER_RESNAMES.has(comp) && !AA3_TO_AA1[comp]) {
        if (NUCLEIC_RESNAMES.has(comp)) nucleicChains.add(toks[6] ?? 'A');
        else ligandSet.add(comp);
      }
    } else if (line.startsWith('ATOM')) {
      const toks = tokenizeCifLine(line);
      const comp = (toks[5] ?? '').toUpperCase();
      if (NUCLEIC_RESNAMES.has(comp)) {
        nucleicChains.add(toks[6] ?? 'A');
      }
    }
  }
  const chainLengths = computeChainLengthsFromIds(chainIds);
  return {
    sequence: seq,
    nResidues: L,
    caCoords,
    backbone4Coords,
    chainIds,
    pdbText,
    format: 'mmcif',
    diagnostics: {
      chainCount: chainLengths.length,
      chainLengths,
      hasLigands: ligandSet.size > 0,
      ligandNames: [...ligandSet],
      hasNucleicAcids: nucleicChains.size > 0,
      nucleicAcidChains: [...nucleicChains],
    },
  };
}

/**
 * Generates an ideal 53-residue de novo 3-helix bundle backbone with exact ~3.80 Å
 * Cα-Cα spacing, 0 chain breaks, 0 clashes, and proper N/CA/C/O peptide planes so
 * users can test ProteinMPNN -> LocalFold AF2 -> Kabsch scRMSD in 1 click.
 */
export function generateSampleHelixBundlePdb(): string {
  // 3 antiparallel ideal α-helices (radius 2.30 Å, rise 1.50 Å, twist 100° => CA-CA = 3.80 Å)
  // spaced ~10.2 Å apart (classic coiled-coil / 3-helix bundle packing) joined by 5-aa loops.
  const radius = 2.3;
  const rise = 1.5;
  const twist = (100 * Math.PI) / 180;

  const makeHelix = (cx: number, cy: number, z0: number, len: number, dir: 1 | -1, phase: number) => {
    const pts: Array<[number, number, number]> = [];
    for (let k = 0; k < len; k++) {
      const ang = phase + k * twist * dir;
      pts.push([cx + radius * Math.cos(ang), cy + radius * Math.sin(ang), z0 + k * rise * dir]);
    }
    return pts;
  };

  // Sample a smooth quadratic Bezier loop with n interior points spaced cleanly (~3.6-4.0 Å)
  const makeLoop = (
    p0: [number, number, number],
    p1: [number, number, number],
    zOut: number,
  ) => {
    const mid: [number, number, number] = [
      0.5 * (p0[0] + p1[0]),
      0.5 * (p0[1] + p1[1]),
      0.5 * (p0[2] + p1[2]) + zOut,
    ];
    // Oversample and resample by arc length for uniform Cα-Cα step
    const dense: Array<[number, number, number]> = [];
    const M = 100;
    for (let i = 0; i <= M; i++) {
      const t = i / M;
      const omt = 1 - t;
      dense.push([
        omt * omt * p0[0] + 2 * omt * t * mid[0] + t * t * p1[0],
        omt * omt * p0[1] + 2 * omt * t * mid[1] + t * t * p1[1],
        omt * omt * p0[2] + 2 * omt * t * mid[2] + t * t * p1[2],
      ]);
    }
    const cum = [0];
    for (let i = 1; i < dense.length; i++) {
      cum.push(
        cum[i - 1] +
          Math.hypot(
            dense[i][0] - dense[i - 1][0],
            dense[i][1] - dense[i - 1][1],
            dense[i][2] - dense[i - 1][2],
          ),
      );
    }
    const total = cum[cum.length - 1];
    const nSteps = Math.max(3, Math.round(total / 3.8));
    const nOut = nSteps - 1;
    const out: Array<[number, number, number]> = [];
    for (let k = 1; k <= nOut; k++) {
      const target = (total * k) / nSteps;
      let idx = 1;
      while (idx < cum.length - 1 && cum[idx] < target) idx++;
      const frac = (target - cum[idx - 1]) / Math.max(1e-6, cum[idx] - cum[idx - 1]);
      out.push([
        dense[idx - 1][0] + frac * (dense[idx][0] - dense[idx - 1][0]),
        dense[idx - 1][1] + frac * (dense[idx][1] - dense[idx - 1][1]),
        dense[idx - 1][2] + frac * (dense[idx][2] - dense[idx - 1][2]),
      ]);
    }
    return out;
  };

  const h1 = makeHelix(-5.2, -2.8, -12.0, 16, 1, 0.0);
  const h2 = makeHelix(5.2, -2.8, 10.5, 16, -1, 1.2);
  const h3 = makeHelix(0.0, 6.2, -12.0, 16, 1, 2.4);

  const l1 = makeLoop(h1[h1.length - 1], h2[0], 6.8);
  const l2 = makeLoop(h2[h2.length - 1], h3[0], -6.8);
  const caPos = [...h1, ...l1, ...h2, ...l2, ...h3];
  const L = caPos.length;
  const ncac = new Float32Array(L * 3 * 3);

  // Place N and C along local backbone tangents with realistic N-CA (1.46Å) and CA-C (1.52Å)
  for (let i = 0; i < L; i++) {
    const ca = caPos[i];
    const prev = caPos[Math.max(0, i - 1)];
    const next = caPos[Math.min(L - 1, i + 1)];
    let tx = next[0] - prev[0], ty = next[1] - prev[1], tz = next[2] - prev[2];
    const tl = Math.hypot(tx, ty, tz) || 1;
    tx /= tl; ty /= tl; tz /= tl;

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
