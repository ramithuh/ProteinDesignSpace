// cg2all input parsing: Cα trace extraction, sequence threading, SSBOND and ligand passthrough.
// Mirrors cg2all's PDB/CalphaBasedModel.read_cg (MDTraj loader, is_all=False) for PDB text input:
// first MODEL only, a new chain at every chain-ID change or TER, one node per residue with a CA atom.

import { AMINO_ACIDS } from './constants.js';

const ONE_TO_THREE = {
  A: 'ALA', C: 'CYS', D: 'ASP', E: 'GLU', F: 'PHE', G: 'GLY', H: 'HSD', I: 'ILE', K: 'LYS', L: 'LEU',
  M: 'MET', N: 'ASN', P: 'PRO', Q: 'GLN', R: 'ARG', S: 'SER', T: 'THR', V: 'VAL', W: 'TRP', Y: 'TYR',
};

// cg2all maps HIS to HSD or HSE at random (libcg.read_cg); this port always uses HSD (residue_constants
// AMINO_ACID_ALT_s) so runs are reproducible. Common force-field protonation aliases map to their parent.
const RESIDUE_ALIASES = {
  HIS: 'HSD', HID: 'HSD', HIP: 'HSD', HSP: 'HSD', HIE: 'HSE', MSE: 'MET', CYX: 'CYS', CYM: 'CYS',
  ASH: 'ASP', GLH: 'GLU', LYN: 'LYS',
};

const WATER_NAMES = new Set(['HOH', 'WAT', 'DOD', 'H2O', 'TIP', 'TIP3', 'SOL']);

/** cg2all residue-type index (0..21) for a residue name (UNK = 21). */
export function residueTypeIndex(resName) {
  const name = RESIDUE_ALIASES[resName] ?? resName;
  const idx = AMINO_ACIDS.indexOf(name);
  return idx >= 0 ? idx : AMINO_ACIDS.length - 1;
}

/** cg2all (CHARMM) residue name for a 3-letter input name; unknown names stay as given. */
export function canonicalResidueName(resName) {
  return RESIDUE_ALIASES[resName] ?? resName;
}

function parseFloatField(line, a, b) {
  const v = Number.parseFloat(line.slice(a, b));
  return Number.isFinite(v) ? v : 0;
}

/**
 * Parse a PDB into the residue list cg2all sees.
 * @param {string} pdbText
 * @returns {{
 *   residues: Array<{chainIndex:number, chainId:string, resNum:number, insCode:string, resName:string,
 *     inputName:string, ca:[number,number,number], bfactor:number, lines:string[]}>,
 *   ssbonds: Array<[number, number]>,
 *   ligandLines: string[],   // non-water HETATM groups and CA-less residues, verbatim
 *   warnings: string[],
 * }}
 */
export function parseCaTrace(pdbText) {
  const residues = [];
  const ligandLines = [];
  const ssbondRecords = [];
  const warnings = [];
  let chainIndex = -1;
  let lastChainId = null;
  let terSeen = true;
  let current = null;
  let currentKey = '';
  let modelCount = 0;

  const flush = () => {
    // Residues without a CA (nucleic acids, partial residues) cannot be graph nodes: pass them through.
    if (current?.ca) residues.push(current);
    else if (current) ligandLines.push(...current.lines);
    current = null;
    currentKey = '';
  };

  for (const rawLine of String(pdbText ?? '').split(/\r?\n/)) {
    const rec = rawLine.slice(0, 6);
    if (rec === 'MODEL ') {
      modelCount++;
      if (modelCount > 1) break;
      continue;
    }
    if (rec === 'ENDMDL') break;
    if (rec.startsWith('TER')) {
      flush();
      terSeen = true;
      continue;
    }
    if (rec === 'SSBOND') {
      ssbondRecords.push([
        [rawLine[15] ?? ' ', rawLine.slice(17, 22).trim()],
        [rawLine[29] ?? ' ', rawLine.slice(31, 36).trim()],
      ]);
      continue;
    }
    if (rec !== 'ATOM  ' && rec !== 'HETATM') continue;
    const line = rawLine.padEnd(80, ' ');
    const atomName = line.slice(12, 16).trim();
    const altLoc = line[16];
    const resName = line.slice(17, 20).trim() || line.slice(17, 21).trim();
    const chainId = line[21];
    const resNum = Number.parseInt(line.slice(22, 26), 10);
    const insCode = line[26] === ' ' ? '' : line[26];
    const isProteinName = rec === 'ATOM  ' || RESIDUE_ALIASES[resName] !== undefined || AMINO_ACIDS.includes(resName);
    if (!isProteinName) {
      if (!WATER_NAMES.has(resName)) ligandLines.push(rawLine);
      continue;
    }
    if (altLoc !== ' ' && altLoc !== 'A' && altLoc !== '1') continue;
    const key = `${chainId}|${resNum}|${insCode}|${resName}`;
    if (key !== currentKey) {
      flush();
      if (terSeen || chainId !== lastChainId) {
        chainIndex++;
        terSeen = false;
        lastChainId = chainId;
      }
      current = {
        chainIndex,
        chainId: chainId === ' ' ? '' : chainId,
        resNum: Number.isFinite(resNum) ? resNum : 0,
        insCode,
        resName: canonicalResidueName(resName),
        inputName: resName,
        ca: null,
        bfactor: 0,
        lines: [],
      };
      currentKey = key;
    }
    current.lines.push(rawLine);
    if (atomName === 'CA' && !current.ca) {
      current.ca = [parseFloatField(line, 30, 38), parseFloatField(line, 38, 46), parseFloatField(line, 46, 54)];
      current.bfactor = parseFloatField(line, 60, 66);
    }
  }
  flush();

  // Chain indices skip chains whose residues all lacked CA; renumber them densely in order.
  const remap = new Map();
  for (const r of residues) {
    if (!remap.has(r.chainIndex)) remap.set(r.chainIndex, remap.size);
    r.chainIndex = remap.get(r.chainIndex);
  }

  const ssbonds = [];
  const used = new Set();
  for (const [a, b] of ssbondRecords) {
    const find = ([ch, seq]) =>
      residues.findIndex((r) => (r.chainId || ' ') === ch && `${r.resNum}${r.insCode}` === seq);
    const i = find(a);
    const j = find(b);
    if (i < 0 || j < 0 || i === j) {
      warnings.push(`Ignored SSBOND ${a.join('')}-${b.join('')}: residue not found in the Cα trace`);
      continue;
    }
    const pair = i < j ? [i, j] : [j, i];
    if (used.has(pair[0]) || used.has(pair[1])) {
      warnings.push(`Ignored SSBOND ${a.join('')}-${b.join('')}: cysteine already bonded`);
      continue;
    }
    used.add(pair[0]);
    used.add(pair[1]);
    ssbonds.push(pair);
  }
  return { residues, ssbonds, ligandLines, warnings };
}

/**
 * Split a threaded sequence over chains: one-letter codes for all chains (concatenated, or separated by
 * '/' or ':'), or one string per chain — the same contract as the PULCHRA plugin.
 */
export function splitSequence(seq, chainLengths) {
  const clean = (s) => String(s).replace(/\s+/g, '').toUpperCase();
  let parts;
  if (typeof seq !== 'string') parts = Array.from(seq, clean);
  else if (/[/:]/.test(seq)) parts = clean(seq).split(/[/:]/);
  else {
    const all = clean(seq);
    const total = chainLengths.reduce((a, b) => a + b, 0);
    if (all.length !== total) throw new Error(`cg2all: sequence has ${all.length} residues but the trace has ${total}`);
    parts = [];
    let o = 0;
    for (const n of chainLengths) {
      parts.push(all.slice(o, o + n));
      o += n;
    }
  }
  if (parts.length !== chainLengths.length) {
    throw new Error(`cg2all: sequence has ${parts.length} chain(s) but the trace has ${chainLengths.length}`);
  }
  parts.forEach((p, i) => {
    if (p.length !== chainLengths[i]) {
      throw new Error(`cg2all: chain ${i + 1} sequence has ${p.length} residues but the trace has ${chainLengths[i]}`);
    }
    const bad = p.match(/[^ACDEFGHIKLMNPQRSTVWY]/);
    if (bad) throw new Error(`cg2all: unsupported residue code '${bad[0]}' in sequence`);
  });
  return parts;
}

/** Replace residue identities on a parsed trace (in place); threaded residues drop their input lines. */
export function threadSequence(trace, seq) {
  const lengths = [];
  for (const r of trace.residues) lengths[r.chainIndex] = (lengths[r.chainIndex] ?? 0) + 1;
  const parts = splitSequence(seq, lengths);
  const pos = new Array(parts.length).fill(0);
  for (const r of trace.residues) {
    const name = ONE_TO_THREE[parts[r.chainIndex][pos[r.chainIndex]++]];
    if (name !== r.resName) {
      r.resName = name;
      r.inputName = name === 'HSD' ? 'HIS' : name;
      r.lines = [];
    }
  }
  return trace;
}
