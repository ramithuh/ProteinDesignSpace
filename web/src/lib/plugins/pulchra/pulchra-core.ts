/**
 * Pure TypeScript port of PULCHRA 3.06 — PowerfUL CHain Restoration Algorithm
 * (Rotkiewicz & Skolnick, J. Comput. Chem. 29:1460, 2008). Copyright (c)
 * 2000-2009 Piotr Rotkiewicz, MIT License — see static/vendor/pulchra/LICENSE.
 *
 * Rebuilds all-atom models from Cα-only, backbone-only or partial traces:
 *   1. optional Cα optimisation (steepest descent + line search, `-c`/`-u`/`-p`)
 *   2. N/C/O from the nco_stat tables (Cα i-2..i+1 distance/chirality bins),
 *      optional peptide-plane H-bond optimisation (`-q`)
 *   3. side chains from the rot_stat rotamer library, excluded-volume clash
 *      fixing (`-o` disables) and chirality repair (`-z` disables)
 *   4. optional backbone amide hydrogens and PDB formatting.
 *
 * The control flow, operation order and floating-point expressions follow
 * pulchra.c line by line (including its macro expansions, glibc `rand()`,
 * float-typed constants and the statically-allocated clash grid) so results
 * match the native reference; tests pin golden coordinates from it.
 *
 * Deliberate differences from the 3.06 executable:
 *   - every chain is rebuilt independently (the C reader stops at the first
 *     TER), and chain IDs / insertion codes / B-factors are kept in the output;
 *   - 3.06 hard-codes `_PRESERVE = 1`, which disables Cα optimisation entirely.
 *     Here `optimizeCa` runs it in PULCHRA's classic non-preserving mode, so all
 *     non-Cα atoms are rebuilt from the optimised trace;
 *   - `-h` in 3.06 cannot enable hydrogens; `outputHydrogens` places amide H
 *     on non-proline residues with PULCHRA's own N - 1.081 Å·û(C→O) rule;
 *   - centering translates every atom (3.06 skips preserved input atoms);
 *   - residues without a Cα and non-amino-acid HETATM groups are skipped, and
 *     UNK residues are written backbone-only instead of being dropped.
 */

export const PULCHRA_VERSION = '3.06';
export const PULCHRA_TABLES_URL = '/vendor/pulchra/pulchra-tables.bin';

// ─── Lookup tables (pulchra-tables.bin, packed by tools/pack-pulchra-tables.mjs) ──

const TABLES_MAGIC = 'PULCHRA\0';
const TABLES_FORMAT_VERSION = 1;
const TABLES_HEADER_BYTES = 64;

export interface PulchraTables {
  /** Fixed-point divisor: a stored int16 q decodes to Math.fround(q / scale). */
  scale: number;
  ncoBins: Int8Array;
  ncoData: Int16Array;
  ncoProBins: Int8Array;
  ncoProData: Int16Array;
  /** rot_stat_idx columns 0-4 (aa type, r13_1, r13_2, r14 bins, rotamer count). */
  rotIdx: Int8Array;
  /** rot_stat_idx column 5: first rot_stat_coords row of the entry. */
  rotPos: Uint32Array;
  rotCoords: Int16Array;
  /** rot_stat_idx rows per amino-acid type, ascending (C scan order). */
  rotRowsByType: Int32Array[];
  byteLength: number;
}

export function decodePulchraTables(input: ArrayBuffer | ArrayBufferView): PulchraTables {
  // Views (e.g. a Node Buffer from a shared pool) are copied so every section
  // offset is aligned relative to a fresh ArrayBuffer.
  const buffer = ArrayBuffer.isView(input)
    ? (new Uint8Array(input.buffer, input.byteOffset, input.byteLength).slice().buffer as ArrayBuffer)
    : input;
  if (buffer.byteLength < TABLES_HEADER_BYTES) throw new Error('PULCHRA tables: file is truncated');
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 8));
  if (magic !== TABLES_MAGIC) throw new Error('PULCHRA tables: bad magic (not a pulchra-tables.bin file)');
  const h = new DataView(buffer);
  const u32 = (i: number) => h.getUint32(8 + 4 * i, true);
  const version = u32(0);
  if (version !== TABLES_FORMAT_VERSION) throw new Error(`PULCHRA tables: unsupported format version ${version}`);
  const [scale, nNco, nPro, nIdx, nRows] = [u32(1), u32(2), u32(3), u32(4), u32(5)];
  const off = [u32(6), u32(7), u32(8), u32(9), u32(10), u32(11), u32(12)];
  if (u32(13) !== buffer.byteLength) {
    throw new Error(`PULCHRA tables: expected ${u32(13)} bytes, got ${buffer.byteLength}`);
  }
  const view = <T>(ctor: new (b: ArrayBuffer, o: number, n: number) => T, o: number, n: number, bpe: number): T => {
    if (o % bpe !== 0 || o + n * bpe > buffer.byteLength) throw new Error('PULCHRA tables: corrupt section layout');
    return new ctor(buffer, o, n);
  };
  const t: PulchraTables = {
    scale,
    ncoBins: view(Int8Array, off[0], nNco * 3, 1),
    ncoData: view(Int16Array, off[1], nNco * 24, 2),
    ncoProBins: view(Int8Array, off[2], nPro * 3, 1),
    ncoProData: view(Int16Array, off[3], nPro * 24, 2),
    rotIdx: view(Int8Array, off[4], nIdx * 5, 1),
    rotPos: view(Uint32Array, off[5], nIdx, 4),
    rotCoords: view(Int16Array, off[6], nRows * 3, 2),
    rotRowsByType: [],
    byteLength: buffer.byteLength,
  };
  const rows: number[][] = Array.from({ length: 20 }, () => []);
  for (let j = 0; t.rotIdx[j * 5] >= 0; j++) rows[t.rotIdx[j * 5]].push(j);
  t.rotRowsByType = rows.map((r) => Int32Array.from(r));
  return t;
}

const decodedTables = new WeakMap<ArrayBuffer | ArrayBufferView, PulchraTables>();

interface InFlightFetch {
  promise: Promise<PulchraTables>;
  controller: AbortController;
  waiters: number;
}

const fetchedTablesByUrl = new Map<string, PulchraTables>();
const inFlightFetchesByUrl = new Map<string, InFlightFetch>();

export interface LoadPulchraTablesOptions {
  /** Pre-loaded pulchra-tables.bin bytes (Node/Vitest: fs.readFileSync). Skips the network. */
  tablesBuffer?: ArrayBuffer | ArrayBufferView;
  /** Override the static asset URL (defaults to PULCHRA_TABLES_URL). */
  url?: string;
  fetchFn?: typeof fetch;
  /** Optional signal to cancel table loading without poisoning shared cache entries. */
  signal?: AbortSignal;
}

function abortError(signal?: AbortSignal): unknown {
  if (signal?.reason !== undefined) return signal.reason;
  return new DOMException('The operation was aborted.', 'AbortError');
}

function awaitWithAbort<T>(promise: Promise<T>, signal?: AbortSignal, onAbort?: () => void): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) {
    onAbort?.();
    return Promise.reject(abortError(signal));
  }
  return new Promise<T>((resolve, reject) => {
    const handleAbort = () => {
      onAbort?.();
      reject(abortError(signal));
    };
    signal.addEventListener('abort', handleAbort, { once: true });
    promise.then(
      (val) => {
        signal.removeEventListener('abort', handleAbort);
        resolve(val);
      },
      (err) => {
        signal.removeEventListener('abort', handleAbort);
        reject(err);
      },
    );
  });
}

/** Decodes `tablesBuffer`, or lazily fetches the 1.6 MB table file once per URL. */
export async function loadPulchraTables(opts: LoadPulchraTablesOptions = {}): Promise<PulchraTables> {
  opts.signal?.throwIfAborted();
  if (opts.tablesBuffer) {
    let t = decodedTables.get(opts.tablesBuffer);
    if (!t) {
      t = decodePulchraTables(opts.tablesBuffer);
      decodedTables.set(opts.tablesBuffer, t);
    }
    return t;
  }
  const url = opts.url ?? PULCHRA_TABLES_URL;
  const cached = fetchedTablesByUrl.get(url);
  if (cached) return cached;

  let entry = inFlightFetchesByUrl.get(url);
  if (!entry) {
    const fetchFn = opts.fetchFn ?? fetch;
    const controller = new AbortController();
    const promise = (async () => {
      const res = await (opts.signal ? fetchFn(url, { signal: controller.signal }) : fetchFn(url));
      if (!res.ok) throw new Error(`PULCHRA tables: GET ${url} failed (HTTP ${res.status})`);
      const tables = decodePulchraTables(await res.arrayBuffer());
      fetchedTablesByUrl.set(url, tables);
      return tables;
    })();
    entry = { promise, controller, waiters: 0 };
    inFlightFetchesByUrl.set(url, entry);
    // Transient network failures or caller aborts must never poison later non-aborted runs.
    promise
      .finally(() => {
        if (inFlightFetchesByUrl.get(url) === entry) {
          inFlightFetchesByUrl.delete(url);
        }
      })
      .catch(() => {});
  }

  entry.waiters++;
  let abortedHere = false;
  try {
    return await awaitWithAbort(entry.promise, opts.signal, () => {
      abortedHere = true;
      entry!.waiters = Math.max(0, entry!.waiters - 1);
      if (entry!.waiters === 0) {
        if (inFlightFetchesByUrl.get(url) === entry) {
          inFlightFetchesByUrl.delete(url);
        }
        entry!.controller.abort(opts.signal?.reason);
      }
    });
  } finally {
    if (!abortedHere) {
      entry.waiters = Math.max(0, entry.waiters - 1);
    }
  }
}

// ─── Constants from pulchra.c ────────────────────────────────────────────────

const FLAG_BACKBONE = 1;
const FLAG_SIDECHAIN = 4;
const FLAG_SCM = 8;
const FLAG_INITIAL = 16;

const CA_K = 10.0;
const CA_ANGLE_K = 20.0;
const CA_START_K = 0.01;
const CA_XVOL_K = 10.0;
const CA_DIST = 3.8;
const CA_DIST_CISPRO = 2.9;
const CA_DIST_CISPRO_TOL = 0.1;
const CA_XVOL_DIST = 3.5;
const CA_ITER = 100;
const SG_XVOL_DIST = 1.6;
const XVOL_ITER = 3;
const GRID_RES = 6.0;
const M_PI = Math.PI;
/** `1.081f`, `0.01F`: float literals promoted to double. */
const H_BOND_LENGTH_F = Math.fround(1.081);
const HB_EPS_F = Math.fround(0.01);

const AA_NAMES = [
  'GLY', 'ALA', 'SER', 'CYS', 'VAL', 'THR', 'ILE', 'PRO', 'MET', 'ASP',
  'ASN', 'LEU', 'LYS', 'GLU', 'GLN', 'ARG', 'HIS', 'PHE', 'TYR', 'TRP', 'UNK',
];
const SHORT_AA_NAMES = 'GASCVTIPMDNLKEQRHFYWX';
const AA_ALIASES: Record<string, string> = {
  // pulchra.c setseq()
  GLX: 'E', ASX: 'D', HID: 'H', MSE: 'M', SEP: 'S', TPO: 'T', PTR: 'Y',
  // common force-field protonation names
  HIE: 'H', HIP: 'H', HSD: 'H', HSE: 'H', HSP: 'H', CYX: 'C',
};
const SOLVENT_WATER_RESNAMES = new Set(['HOH', 'WAT', 'DOD', 'H2O', 'TIP', 'TIP3']);
const PEPTIDE_CAP_RESNAMES = new Set(['ACE', 'NME', 'NH2', 'FOR']);
const NUCLEIC_RESNAMES = new Set([
  'DA', 'DC', 'DG', 'DT', 'DU', 'DI',
  'A', 'C', 'G', 'U', 'T', 'I',
  'ADE', 'CYT', 'GUA', 'THY', 'URA',
]);
const CHAIN_BREAK_CA_DIST = 4.5;

const TYPE_PRO = 7;
const TYPE_MET = 8;
const TYPE_UNK = 20;
const NHEAVY = [0, 1, 2, 2, 3, 3, 4, 3, 4, 4, 4, 4, 5, 5, 5, 7, 6, 7, 8, 10];
const HEAVY_ATOMS: string[][] = [
  [],
  ['CB '],
  ['CB ', 'OG '],
  ['CB ', 'SG '],
  ['CB ', 'CG1', 'CG2'],
  ['CB ', 'OG1', 'CG2'],
  ['CB ', 'CG1', 'CG2', 'CD1'],
  ['CB ', 'CG ', 'CD '],
  ['CB ', 'CG ', 'SD ', 'CE '],
  ['CB ', 'CG ', 'OD1', 'OD2'],
  ['CB ', 'CG ', 'OD1', 'ND2'],
  ['CB ', 'CG ', 'CD1', 'CD2'],
  ['CB ', 'CG ', 'CD ', 'CE ', 'NZ '],
  ['CB ', 'CG ', 'CD ', 'OE1', 'OE2'],
  ['CB ', 'CG ', 'CD ', 'OE1', 'NE2'],
  ['CB ', 'CG ', 'CD ', 'NE ', 'CZ ', 'NH1', 'NH2'],
  ['CB ', 'CG ', 'ND1', 'CD2', 'CE1', 'NE2'],
  ['CB ', 'CG ', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ '],
  ['CB ', 'CG ', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ ', 'OH '],
  ['CB ', 'CG ', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
];

/** pulchra.c setseq(): one-letter code for a residue name ('X' if unknown). */
function oneLetter(resName: string): string {
  const i = AA_NAMES.indexOf(resName);
  if (i >= 0) return SHORT_AA_NAMES[i];
  return AA_ALIASES[resName] ?? 'X';
}

function aaType(letter: string): number {
  const i = SHORT_AA_NAMES.indexOf(letter);
  return i >= 0 && i < 20 ? i : TYPE_UNK;
}

// ─── Atom / residue model (mirrors atom_type / res_type) ────────────────────

interface Atom {
  /** 3-character PULCHRA name (or 4-char if 4-char heavy atom), e.g. 'CA ', 'OXT', 'CD1', 'SE '. */
  name: string;
  /** Uppercase chemical element symbol (1-2 chars), e.g. 'C', 'N', 'O', 'S', 'SE'. */
  element: string;
  x: number;
  y: number;
  z: number;
  flag: number;
  cispro: boolean;
  gx: number;
  gy: number;
  gz: number;
  res: Residue;
}

interface Residue {
  name: string;
  type: number;
  num: number;
  iCode: string;
  chainId: string;
  /** Sequential index within the chain/segment (res->locnum). */
  locnum: number;
  protein: boolean;
  pdbsg: boolean;
  cmx: number;
  cmy: number;
  cmz: number;
  bfactor: number;
  atoms: Atom[];
  prev: Residue | null;
  next: Residue | null;
}

function inferAtomElement(name: string, rawElement = ''): string {
  const cleanElem = rawElement.trim().toUpperCase().replace(/[^A-Z]/g, '');
  if (cleanElem) return cleanElem.slice(0, 2);
  const trimmed = name.trim().toUpperCase();
  if (trimmed === 'SE') return 'SE';
  const alpha = trimmed.replace(/^[^A-Z]+/, '');
  return alpha[0] ?? 'C';
}

function newAtom(
  res: Residue,
  name: string,
  x: number,
  y: number,
  z: number,
  flag: number,
  element = inferAtomElement(name),
): Atom {
  return { name, element, x, y, z, flag, cispro: false, gx: 0, gy: 0, gz: 0, res };
}

const isName = (a: Atom, c0: string, c1: string) => a.name[0] === c0 && a.name[1] === c1;

function findAtom(res: Residue, name: string): Atom | null {
  for (const a of res.atoms) if (a.name === name) return a;
  return null;
}

function parsePdbAtomName(line: string): { name: string; trimmed: string } {
  const raw4 = (line.substring(12, 16) + '    ').slice(0, 4);
  const trimmed = raw4.trim();
  const name = trimmed.length >= 4 ? trimmed.slice(0, 4) : trimmed.padEnd(3, ' ');
  return { name, trimmed };
}

// ─── PDB input ───────────────────────────────────────────────────────────────

interface ParsedChain {
  chainId: string;
  residues: Residue[];
}

interface PreservedRecord {
  line: string;
  chainId: string;
  resName: string;
  x: number;
  y: number;
  z: number;
  altPrio: number;
}

interface ParsedPdbData {
  chains: ParsedChain[];
  preservedNucleic: PreservedRecord[];
  preservedHetatm: PreservedRecord[];
}

function parsePdbChains(pdbText: string, warnings: string[]): ParsedPdbData {
  const chains: ParsedChain[] = [];
  const preservedNucleic: PreservedRecord[] = [];
  const preservedHetatm: PreservedRecord[] = [];
  const preservedKeyToIndex = new Map<string, number>();
  const preservedSiteBest = new Map<string, { prio: number; resName: string }>();
  const waterKeys = new Set<string>();
  const resAltInfo = new WeakMap<Residue, { rawResName: string; prio: number }>();
  const atomAltInfo = new WeakMap<Atom, { prio: number; altLoc: string }>();

  // `as` keeps the declared unions: the close* helpers below reassign both.
  let chain = null as ParsedChain | null;
  let res = null as Residue | null;
  let resKey = '';
  let skippedNoCa = 0;

  const closeResidue = () => {
    if (!res || !chain) return;
    const hasCa = res.atoms.some((a) => a.name === 'CA ');
    if (!hasCa) skippedNoCa++;
    else {
      res.protein = res.type < 20;
      chain.residues.push(res);
    }
    res = null;
  };
  const closeChain = () => {
    closeResidue();
    if (chain?.residues.length) chains.push(chain);
    chain = null;
  };

  const recordPreserved = (
    list: PreservedRecord[],
    kindTag: 'NUC' | 'HET',
    line: string,
    chainId: string,
    resNumStr: string,
    iCode: string,
    resName: string,
    atomTrimmed: string,
    x: number,
    y: number,
    z: number,
    altPrio: number,
  ) => {
    const siteKey = `${kindTag}:${chainId}:${resNumStr}:${iCode}`;
    const prevSite = preservedSiteBest.get(siteKey);
    if (prevSite && prevSite.resName !== resName) {
      if (altPrio >= prevSite.prio) return;
      preservedSiteBest.set(siteKey, { prio: altPrio, resName });
    } else if (!prevSite || altPrio < prevSite.prio) {
      preservedSiteBest.set(siteKey, { prio: altPrio, resName });
    }
    const atomKey = `${siteKey}:${resName}:${atomTrimmed}`;
    const existingIdx = preservedKeyToIndex.get(atomKey);
    const rec: PreservedRecord = { line, chainId, resName, x, y, z, altPrio };
    if (existingIdx === undefined) {
      preservedKeyToIndex.set(atomKey, list.length);
      list.push(rec);
    } else if (altPrio < list[existingIdx].altPrio) {
      list[existingIdx] = rec;
    }
  };

  for (const line of pdbText.split(/\r?\n/)) {
    if (line.startsWith('END')) break; // END / ENDMDL: first model only
    if (line.startsWith('TER')) {
      closeChain();
      continue;
    }
    const isAtom = line.startsWith('ATOM  ');
    const isHet = line.startsWith('HETATM');
    if (!isAtom && !isHet) continue;

    const altLoc = line[16] ?? ' ';
    const altPrio = altLoc === ' ' || altLoc === 'A' || altLoc === '1' ? 0 : 1;
    const { name, trimmed } = parsePdbAtomName(line);
    const rawElement = line.substring(76, 78).trim().toUpperCase();
    const resName = line.substring(17, 20).trim().toUpperCase();
    const chainId = line[21] ?? ' ';
    const resNumStr = line.substring(22, 26).trim();
    const num = Number.parseInt(line.substring(22, 26), 10);
    const iCode = (line[26] ?? ' ').trim() ? line[26] : ' ';
    const x = Number.parseFloat(line.substring(30, 38));
    const y = Number.parseFloat(line.substring(38, 46));
    const z = Number.parseFloat(line.substring(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    // Skip solvent waters (HOH, WAT, DOD, ...).
    if (isHet && SOLVENT_WATER_RESNAMES.has(resName)) {
      waterKeys.add(`${chainId}|${resNumStr}|${iCode}|${resName}`);
      continue;
    }

    // Preserve nucleic-acid ATOM records cleanly after rebuilt protein chains.
    if (isAtom && NUCLEIC_RESNAMES.has(resName)) {
      recordPreserved(preservedNucleic, 'NUC', line, chainId, resNumStr, iCode, resName, trimmed, x, y, z, altPrio);
      continue;
    }

    // Preserve non-water HETATM ligands, ions, cofactors, and modified-residue HETATMs (e.g. SEP)
    // verbatim without shifting protein residue numbering/length vs parsePdbBackbone (only MSE is
    // rebuilt as a protein residue).
    if (isHet && resName !== 'MSE') {
      recordPreserved(preservedHetatm, 'HET', line, chainId, resNumStr, iCode, resName, trimmed, x, y, z, altPrio);
      continue;
    }

    // Skip terminal peptide caps (ACE, NME, NH2, FOR) on ATOM lines, matching parsePdbBackbone.
    if (isAtom && PEPTIDE_CAP_RESNAMES.has(resName)) continue;

    if (rawElement === 'H' || rawElement === 'D' || /^[0-9]?H/.test(trimmed)) continue;
    if (!Number.isFinite(num)) continue;

    const atomElement = inferAtomElement(name, rawElement);
    if (chain && chain.chainId !== chainId) closeChain();
    if (!chain) chain = { chainId, residues: [] };
    const key = `${num}|${iCode}`;

    const openNewResidue = () => {
      closeResidue();
      resKey = key;
      const type = aaType(oneLetter(resName));
      res = {
        name: resName.padStart(3).slice(-3), type, num, iCode, chainId,
        locnum: chain!.residues.length, protein: false, pdbsg: false,
        cmx: 0, cmy: 0, cmz: 0, bfactor: 0, atoms: [], prev: null, next: null,
      };
      resAltInfo.set(res, { rawResName: resName, prio: altPrio });
    };

    if (!res || key !== resKey) {
      openNewResidue();
    } else {
      const resMeta = resAltInfo.get(res)!;
      const existing = res.atoms.find((a) => a.name === name);
      if (existing) {
        const prevAtomMeta = atomAltInfo.get(existing)!;
        // pulchra.c opens a new residue on duplicate backbone N ("fault in numeration"),
        // whereas distinct altLocs on the same residue select the preferred conformer.
        if (name === 'N  ' && prevAtomMeta.altLoc === altLoc) {
          openNewResidue();
        } else if (altPrio < prevAtomMeta.prio) {
          if (resName !== resMeta.rawResName) {
            res.name = resName.padStart(3).slice(-3);
            res.type = aaType(oneLetter(resName));
            res.pdbsg = false;
            res.atoms = [];
            resAltInfo.set(res, { rawResName: resName, prio: altPrio });
          } else {
            existing.x = x;
            existing.y = y;
            existing.z = z;
            existing.element = atomElement;
            atomAltInfo.set(existing, { prio: altPrio, altLoc });
            if ((name[0] === 'S' && name[1] === 'C') || (name[0] === 'C' && name[1] === 'M')) {
              res.cmx = x;
              res.cmy = y;
              res.cmz = z;
              res.pdbsg = true;
            } else if (name[0] === 'C' && name[1] === 'A') {
              res.bfactor = Number.parseFloat(line.substring(60, 66)) || 0;
              if (!res.pdbsg) {
                res.cmx = x;
                res.cmy = y;
                res.cmz = z;
              }
            }
            continue;
          }
        } else {
          continue;
        }
      } else if (resName !== resMeta.rawResName) {
        if (altPrio < resMeta.prio) {
          res.name = resName.padStart(3).slice(-3);
          res.type = aaType(oneLetter(resName));
          res.pdbsg = false;
          res.atoms = [];
          resAltInfo.set(res, { rawResName: resName, prio: altPrio });
        } else {
          continue;
        }
      }
    }

    const r = res!;
    const atom = newAtom(r, name, x, y, z, FLAG_INITIAL, atomElement);
    atomAltInfo.set(atom, { prio: altPrio, altLoc });
    if ((name[0] === 'S' && name[1] === 'C') || (name[0] === 'C' && name[1] === 'M')) {
      r.cmx = x;
      r.cmy = y;
      r.cmz = z;
      r.pdbsg = true;
      atom.flag |= FLAG_SCM;
    } else if (
      !(
        ((name[0] === 'C' || name[0] === 'N' || name[0] === 'O') && name[1] === ' ') ||
        (name[0] === 'C' && name[1] === 'A') ||
        name === 'OXT'
      )
    ) {
      atom.flag |= FLAG_SIDECHAIN;
    } else {
      atom.flag |= FLAG_BACKBONE;
    }
    if (name[0] === 'C' && name[1] === 'A') {
      atom.flag |= FLAG_BACKBONE;
      r.bfactor = Number.parseFloat(line.substring(60, 66)) || 0;
      if (!r.pdbsg) {
        r.cmx = x;
        r.cmy = y;
        r.cmz = z;
      }
    }
    r.atoms.push(atom);
  }
  closeChain();
  if (skippedNoCa) warnings.push(`Skipped ${skippedNoCa} residue(s) without a Cα atom.`);
  if (waterKeys.size) warnings.push(`Skipped ${waterKeys.size} water HETATM group(s).`);
  for (const c of chains) {
    c.residues.forEach((r, i) => {
      r.locnum = i;
      r.prev = c.residues[i - 1] ?? null;
      r.next = c.residues[i + 1] ?? null;
    });
  }
  const filterMicroHet = (list: PreservedRecord[], kindTag: 'NUC' | 'HET') =>
    list.filter((rec) => {
      const resNumStr = rec.line.substring(22, 26).trim();
      const iCode = (rec.line[26] ?? ' ').trim() ? rec.line[26] : ' ';
      const best = preservedSiteBest.get(`${kindTag}:${rec.chainId}:${resNumStr}:${iCode}`);
      return !best || best.resName === rec.resName;
    });
  return {
    chains,
    preservedNucleic: filterMicroHet(preservedNucleic, 'NUC'),
    preservedHetatm: filterMicroHet(preservedHetatm, 'HET'),
  };
}

// ─── Numerics shared by every stage ─────────────────────────────────────────

/** glibc rand() (TYPE_3 additive feedback generator), seeded like srand(seed). */
class GlibcRand {
  private r = new Int32Array(34);
  private i = 0;

  constructor(seed: number) {
    const r = this.r;
    r[0] = seed === 0 ? 1 : seed;
    for (let i = 1; i < 31; i++) {
      // 16807 * r[i-1] % 2147483647 via Schrage's method, as in srandom_r.
      const hi = Math.trunc(r[i - 1] / 127773);
      const lo = r[i - 1] % 127773;
      let word = 16807 * lo - 2836 * hi;
      if (word < 0) word += 2147483647;
      r[i] = word;
    }
    for (let i = 31; i < 34; i++) r[i] = r[i - 31];
    this.i = 0;
    for (let k = 34; k < 344; k++) this.step();
  }

  private step(): number {
    // r[k] = r[k-31] + r[k-3] over a 34-entry ring buffer.
    const r = this.r;
    const k = this.i;
    const v = (r[(k + 34 - 31 + 34) % 34] + r[(k + 34 - 3) % 34]) | 0;
    r[k] = v;
    this.i = (k + 1) % 34;
    return v;
  }

  next(): number {
    return this.step() >>> 1;
  }
}

/** pulchra.c superimpose2(): rotates tpoints so coords2 best fits coords1. */
function superimpose2(
  coords1: number[][],
  coords2: number[][],
  npoints: number,
  tpoints: number[][],
  ntpoints: number,
): number {
  let cx1 = 0, cy1 = 0, cz1 = 0, cx2 = 0, cy2 = 0, cz2 = 0;
  for (let i = 0; i < npoints; i++) {
    cx1 += coords1[i][0];
    cy1 += coords1[i][1];
    cz1 += coords1[i][2];
    cx2 += coords2[i][0];
    cy2 += coords2[i][1];
    cz2 += coords2[i][2];
  }
  cx1 /= npoints;
  cy1 /= npoints;
  cz1 /= npoints;
  cx2 /= npoints;
  cy2 /= npoints;
  cz2 /= npoints;
  for (let i = 0; i < npoints; i++) {
    coords1[i][0] -= cx1;
    coords1[i][1] -= cy1;
    coords1[i][2] -= cz1;
    coords2[i][0] -= cx2;
    coords2[i][1] -= cy2;
    coords2[i][2] -= cz2;
  }
  for (let i = 0; i < ntpoints; i++) {
    tpoints[i][0] -= cx2;
    tpoints[i][1] -= cy2;
    tpoints[i][2] -= cz2;
  }

  const eye = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const zero = () => [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let s = eye();
  const a = eye(), b = eye(), g = eye();
  let u = zero();
  for (let n = 0; n < npoints; n++) {
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) u[i][j] += coords1[n][i] * coords2[n][j];
  }
  // u = u * R^T ; s = R * s   (accumulated from zero, k ascending, as in C)
  const applyRotation = (r: number[][]) => {
    const nu = zero();
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) nu[i][j] += u[i][k] * r[j][k];
    u = nu;
    const ns = zero();
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) ns[i][j] += r[i][k] * s[k][j];
    s = ns;
  };

  let val: number;
  do {
    let d = u[2][1] - u[1][2];
    let alpha = d === 0 ? 0 : Math.atan(d / (u[1][1] + u[2][2]));
    if (Math.cos(alpha) * (u[1][1] + u[2][2]) + Math.sin(alpha) * (u[2][1] - u[1][2]) < 0.0) alpha += M_PI;
    a[1][1] = a[2][2] = Math.cos(alpha);
    a[2][1] = Math.sin(alpha);
    a[1][2] = -a[2][1];
    applyRotation(a);

    d = u[0][2] - u[2][0];
    let beta = d === 0 ? 0 : Math.atan(d / (u[0][0] + u[2][2]));
    if (Math.cos(beta) * (u[0][0] + u[2][2]) + Math.sin(beta) * (u[0][2] - u[2][0]) < 0.0) beta += M_PI;
    b[0][0] = b[2][2] = Math.cos(beta);
    b[0][2] = Math.sin(beta);
    b[2][0] = -b[0][2];
    applyRotation(b);

    d = u[1][0] - u[0][1];
    let gamma = d === 0 ? 0 : Math.atan(d / (u[0][0] + u[1][1]));
    if (Math.cos(gamma) * (u[0][0] + u[1][1]) + Math.sin(gamma) * (u[1][0] - u[0][1]) < 0.0) gamma += M_PI;
    g[0][0] = g[1][1] = Math.cos(gamma);
    g[1][0] = Math.sin(gamma);
    g[0][1] = -g[1][0];
    applyRotation(g);

    val = Math.abs(alpha) + Math.abs(beta) + Math.abs(gamma);
  } while (val > 0.001);

  val = 0;
  for (let i = 0; i < npoints; i++) {
    const x = coords2[i][0], y = coords2[i][1], z = coords2[i][2];
    const tx = x * s[0][0] + y * s[0][1] + z * s[0][2];
    const ty = x * s[1][0] + y * s[1][1] + z * s[1][2];
    const tz = x * s[2][0] + y * s[2][1] + z * s[2][2];
    const dx = coords1[i][0] - tx, dy = coords1[i][1] - ty, dz = coords1[i][2] - tz;
    val += dx * dx + dy * dy + dz * dz;
  }
  for (let i = 0; i < ntpoints; i++) {
    const x = tpoints[i][0], y = tpoints[i][1], z = tpoints[i][2];
    tpoints[i][0] = x * s[0][0] + y * s[0][1] + z * s[0][2];
    tpoints[i][1] = x * s[1][0] + y * s[1][1] + z * s[1][2];
    tpoints[i][2] = x * s[2][0] + y * s[2][1] + z * s[2][2];
  }
  for (let i = 0; i < npoints; i++) {
    coords1[i][0] += cx1;
    coords1[i][1] += cy1;
    coords1[i][2] += cz1;
    coords2[i][0] += cx2;
    coords2[i][1] += cy2;
    coords2[i][2] += cz2;
  }
  for (let i = 0; i < ntpoints; i++) {
    tpoints[i][0] += cx1;
    tpoints[i][1] += cy1;
    tpoints[i][2] += cz1;
  }
  return Math.sqrt(val / npoints);
}

function calcDistance(x1: number, y1: number, z1: number, x2: number, y2: number, z2: number): number {
  const dx = x1 - x2, dy = y1 - y2, dz = z1 - z2;
  // C truthiness: NaN counts as non-zero.
  return dx !== 0 || dy !== 0 || dz !== 0 ? Math.sqrt(dx * dx + dy * dy + dz * dz) : 0.0;
}

/** Signed Cα(i-2)…Cα(i+1) distance; negative for left-handed quadruples. */
function calcR14(
  x1: number, y1: number, z1: number, x2: number, y2: number, z2: number,
  x3: number, y3: number, z3: number, x4: number, y4: number, z4: number,
): number {
  const dx = x4 - x1, dy = y4 - y1, dz = z4 - z1;
  let r = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const vx1 = x2 - x1, vy1 = y2 - y1, vz1 = z2 - z1;
  const vx2 = x3 - x2, vy2 = y3 - y2, vz2 = z3 - z2;
  const vx3 = x4 - x3, vy3 = y4 - y3, vz3 = z4 - z3;
  const hand =
    (vy1 * vz2 - vy2 * vz1) * vx3 + (vz1 * vx2 - vz2 * vx1) * vy3 + (vx1 * vy2 - vx2 * vy1) * vz3;
  if (hand < 0) r = -r;
  return r;
}

function cross(v1: number[], v2: number[], v3: number[]): void {
  v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
  v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
  v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

function norm(v: number[]): void {
  const d = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  v[0] /= d;
  v[1] /= d;
  v[2] /= d;
}

/** C `(int)` conversion: truncation toward zero (NaN → 0 like aarch64 fcvtzs). */
const toInt = (v: number) => v | 0;

const points = (n: number) => Array.from({ length: n }, () => [0, 0, 0]);

// ─── Per-chain reconstruction state ─────────────────────────────────────────

interface Grid {
  nx: number;
  ny: number;
  nz: number;
  cells: Atom[][];
}

interface ChainStats {
  backboneRmsdMean: number | null;
  backboneRmsdMax: number | null;
  caOptimizationSteps: number;
  cisProlines: number;
  hbondEnergyBefore: number | null;
  hbondEnergyAfter: number | null;
  clashesBefore: number | null;
  clashesAfter: number | null;
  dAminoAcidsFixed: number;
}

class ChainBuilder {
  readonly L: number;
  /** X_COORDS: Cα trace padded with 5 slots on each side; C_ALPHA[i] = row i + 5. */
  private xc: Float64Array;
  private rbins: Int32Array | null = null;
  /** SORTED_ROTAMERS[i][l] = (hit, rot_stat_idx row) for the 10 nearest bins. */
  private sorted: Float64Array | null = null;
  private grid: Grid | null = null;
  readonly stats: ChainStats = {
    backboneRmsdMean: null,
    backboneRmsdMax: null,
    caOptimizationSteps: 0,
    cisProlines: 0,
    hbondEnergyBefore: null,
    hbondEnergyAfter: null,
    clashesBefore: null,
    clashesAfter: null,
    dAminoAcidsFixed: 0,
  };

  constructor(
    readonly residues: Residue[],
    private readonly T: PulchraTables,
    private readonly preserve: boolean,
    private readonly emitTerminalOxt = true,
  ) {
    this.L = residues.length;
    this.xc = new Float64Array((this.L + 10) * 3);
  }

  private caX(i: number, k: number): number {
    return this.xc[(i + 5) * 3 + k];
  }

  private rot(row: number, k: number): number {
    return Math.fround(this.T.rotCoords[row * 3 + k] / this.T.scale);
  }

  /** pulchra.c add_replace(). */
  private addReplace(res: Residue, name: string, x: number, y: number, z: number, flags: number): void {
    for (const atom of res.atoms) {
      if (atom.name === name) {
        if (!(this.preserve && atom.flag & FLAG_INITIAL)) {
          atom.x = x;
          atom.y = y;
          atom.z = z;
        }
        atom.flag |= flags;
        return;
      }
    }
    const atom = newAtom(res, name, x, y, z, flags);
    if (name === 'N  ') res.atoms.unshift(atom);
    else res.atoms.push(atom);
  }

  // ── Stage 1: Cα optimisation (ca_optimize / calc_ca_energy) ──────────────

  optimizeCa(maxShift: number, detectCisPro: boolean): void {
    const ca: Atom[] = [];
    for (const res of this.residues) {
      const a = res.atoms.find((at) => isName(at, 'C', 'A'));
      if (a) ca.push(a);
    }
    const L = ca.length;
    if (L === 0) return;
    const init = new Float64Array(L * 3);
    const trial = new Float64Array(L * 3);
    const grad = new Float64Array(L * 3);
    for (let i = 0; i < L; i++) {
      init[i * 3] = ca[i].x;
      init[i * 3 + 1] = ca[i].y;
      init[i * 3 + 2] = ca[i].z;
    }
    if (detectCisPro) {
      for (let i = 1; i < L; i++) {
        const dx = ca[i].x - ca[i - 1].x, dy = ca[i].y - ca[i - 1].y, dz = ca[i].z - ca[i - 1].z;
        const dd = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (
          oneLetter(ca[i].res.name) === 'P' &&
          dd > CA_DIST_CISPRO - 5 * CA_DIST_CISPRO_TOL &&
          dd < CA_DIST_CISPRO + 5 * CA_DIST_CISPRO_TOL
        ) {
          ca[i].cispro = true;
          this.stats.cisProlines++;
        }
      }
    }

    const energy = (alpha: number, calcGradient: boolean): number => {
      for (let i = 0; i < L; i++) {
        trial[i * 3] = ca[i].x + alpha * grad[i * 3];
        trial[i * 3 + 1] = ca[i].y + alpha * grad[i * 3 + 1];
        trial[i * 3 + 2] = ca[i].z + alpha * grad[i * 3 + 2];
      }
      let e = 0.0;
      for (let i = 0; i < L; i++) {
        const i3 = i * 3;
        // restraint to the input trace
        let dx = trial[i3] - init[i3];
        let dy = trial[i3 + 1] - init[i3 + 1];
        let dz = trial[i3 + 2] - init[i3 + 2];
        let dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        let ddist = -dist;
        if (dist > maxShift) {
          e += CA_START_K * (dist * dist);
          if (calcGradient) {
            const gr = (ddist * (-2.0 * CA_START_K)) / dist;
            grad[i3] -= gr * dx;
            grad[i3 + 1] -= gr * dy;
            grad[i3 + 2] -= gr * dz;
          }
        }
        // virtual Cα-Cα bonds
        if (i > 0) {
          dx = trial[i3] - trial[i3 - 3];
          dy = trial[i3 + 1] - trial[i3 - 2];
          dz = trial[i3 + 2] - trial[i3 - 1];
          dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          ddist = (ca[i].cispro ? CA_DIST_CISPRO : CA_DIST) - dist;
          e += CA_K * (ddist * ddist);
          if (calcGradient) {
            const gr = (ddist * (-2.0 * CA_K)) / dist;
            grad[i3] -= gr * dx;
            grad[i3 + 1] -= gr * dy;
            grad[i3 + 2] -= gr * dz;
            grad[i3 - 3] += gr * dx;
            grad[i3 - 2] += gr * dy;
            grad[i3 - 1] += gr * dz;
          }
        }
        // excluded volume (|i-j| > 2)
        for (let j = 0; j < i - 2; j++) {
          const j3 = j * 3;
          dx = trial[i3] - trial[j3];
          dy = trial[i3 + 1] - trial[j3 + 1];
          dz = trial[i3 + 2] - trial[j3 + 2];
          dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          ddist = dist - CA_XVOL_DIST;
          if (dist < CA_XVOL_DIST) {
            e += CA_XVOL_K * (dist * dist);
            if (calcGradient) {
              const gr = (ddist * (8.0 * CA_XVOL_K)) / dist;
              grad[i3] -= gr * dx;
              grad[i3 + 1] -= gr * dy;
              grad[i3 + 2] -= gr * dz;
              grad[j3] += gr * dx;
              grad[j3 + 1] += gr * dy;
              grad[j3 + 2] += gr * dz;
            }
          }
        }
        // Cα pseudo-angles restrained to [80°, 150°]
        if (i > 0 && i < L - 1) {
          const r12x = trial[i3 - 3] - trial[i3], r12y = trial[i3 - 2] - trial[i3 + 1], r12z = trial[i3 - 1] - trial[i3 + 2];
          const r32x = trial[i3 + 3] - trial[i3], r32y = trial[i3 + 4] - trial[i3 + 1], r32z = trial[i3 + 5] - trial[i3 + 2];
          const d12 = Math.sqrt(r12x * r12x + r12y * r12y + r12z * r12z);
          const d32 = Math.sqrt(r32x * r32x + r32y * r32y + r32z * r32z);
          let cosTheta = (r12x * r32x + r12y * r32y + r12z * r32z) / (d12 * d32);
          if (cosTheta > 1.0) cosTheta = 1.0;
          else if (cosTheta < -1.0) cosTheta = -1.0;
          const sinTheta = Math.sqrt(1.0 - cosTheta * cosTheta);
          const theta = Math.acos(cosTheta);
          let diff: number;
          // RADDEG / DEGRAD are unparenthesised macros: 180./M_PI*x and x*M_PI/180.
          if ((180 / M_PI) * theta < 80) diff = theta - (80 * M_PI) / 180;
          else if ((180 / M_PI) * theta > 150) diff = theta - (150 * M_PI) / 180;
          else diff = 0.0;
          e += CA_ANGLE_K * diff * diff;
          if (calcGradient) {
            const d12inv = 1.0 / d12, d32inv = 1.0 / d32;
            diff *= (-2.0 * CA_ANGLE_K) / sinTheta;
            const c1 = diff * d12inv, c2 = diff * d32inv;
            const f1x = c1 * (r12x * (d12inv * cosTheta) - r32x * d32inv);
            const f1y = c1 * (r12y * (d12inv * cosTheta) - r32y * d32inv);
            const f1z = c1 * (r12z * (d12inv * cosTheta) - r32z * d32inv);
            const f3x = c2 * (r32x * (d32inv * cosTheta) - r12x * d12inv);
            const f3y = c2 * (r32y * (d32inv * cosTheta) - r12y * d12inv);
            const f3z = c2 * (r32z * (d32inv * cosTheta) - r12z * d12inv);
            grad[i3 - 3] += f1x;
            grad[i3 - 2] += f1y;
            grad[i3 - 1] += f1z;
            grad[i3] += -f1x - f3x;
            grad[i3 + 1] += -f1y - f3y;
            grad[i3 + 2] += -f1z - f3z;
            grad[i3 + 3] += f3x;
            grad[i3 + 4] += f3y;
            grad[i3 + 5] += f3z;
          }
        }
      }
      return e;
    };

    const rng = new GlibcRand(1234);
    const rnd = () => 0.001 * (rng.next() % 1000);
    let numsteps = 0, fcnt = 0, eps = 0.5, lastGnorm = 1000, gnorm: number;
    do {
      grad.fill(0);
      energy(0.0, true);
      // Bracketing line search along the (negative) gradient.
      let alpha1 = -1.0, alpha2 = 0.0, alpha3 = 1.0;
      let ene1 = energy(alpha1, false), ene2 = energy(alpha2, false), ene3 = energy(alpha3, false);
      let msteps = 0;
      while (ene2 > (ene1 < ene3 ? ene1 : ene3) && msteps < CA_ITER) {
        msteps++;
        alpha1 *= 2.0;
        alpha3 *= 2.0;
        ene1 = energy(alpha1, false);
        ene3 = energy(alpha3, false);
      }
      msteps = 0;
      do {
        // (pulchra.c also probes a0 ± 1e-5 here; those energies are discarded.)
        if (alpha3 - alpha2 > alpha2 - alpha1) {
          const a0 = 0.5 * (alpha2 + alpha3);
          const e0 = energy(a0, false);
          if (e0 < ene2) {
            alpha1 = alpha2;
            alpha2 = a0;
            ene1 = ene2;
            ene2 = e0;
          } else {
            alpha3 = a0;
            ene3 = e0;
          }
        } else {
          const a0 = 0.5 * (alpha1 + alpha2);
          const e0 = energy(a0, false);
          if (e0 < ene2) {
            alpha3 = alpha2;
            alpha2 = a0;
            ene3 = ene2;
            ene2 = e0;
          } else {
            alpha1 = a0;
            ene1 = e0;
          }
        }
        msteps++;
      } while (alpha3 - alpha1 > 1e-6 && msteps < 20);

      const lastAlpha = alpha2;
      for (let i = 0; i < L; i++) {
        ca[i].x = ca[i].x + (lastAlpha + lastAlpha * (rnd() - 0.5) * eps) * grad[i * 3];
        ca[i].y = ca[i].y + (lastAlpha + lastAlpha * (rnd() - 0.5) * eps) * grad[i * 3 + 1];
        ca[i].z = ca[i].z + (lastAlpha + lastAlpha * (rnd() - 0.5) * eps) * grad[i * 3 + 2];
      }
      eps *= 0.75;
      if (eps < 1e-3) eps = 0.0;
      numsteps++;
      gnorm = 0.0;
      for (let i = 0; i < L; i++) {
        gnorm += grad[i * 3] * grad[i * 3] + grad[i * 3 + 1] * grad[i * 3 + 1] + grad[i * 3 + 2] * grad[i * 3 + 2];
      }
      gnorm = Math.sqrt(gnorm / L);
      if (lastGnorm - gnorm < 1e-3) fcnt++;
      lastGnorm = gnorm;
    } while (fcnt < 3 && gnorm > 0.01 && numsteps < CA_ITER);
    this.stats.caOptimizationSteps = numsteps;
  }

  // ── Stage 2: backbone (prepare_rbins / rebuild_backbone) ────────────────

  private prepareRbins(): Int32Array {
    if (this.rbins) return this.rbins;
    const L = this.L;
    const xc = this.xc;
    let n = 5;
    for (const res of this.residues) {
      for (const atom of res.atoms) {
        if (isName(atom, 'C', 'A')) {
          xc[n * 3] = atom.x;
          xc[n * 3 + 1] = atom.y;
          xc[n * 3 + 2] = atom.z;
          n++;
        }
      }
    }
    const getCa = (i: number) => [this.caX(i, 0), this.caX(i, 1), this.caX(i, 2)];
    const setCa = (i: number, p: number[]) => {
      xc[(i + 5) * 3] = p[0];
      xc[(i + 5) * 3 + 1] = p[1];
      xc[(i + 5) * 3 + 2] = p[2];
    };
    const cacoords = points(8), tmpcoords = points(8), tmpstat = points(8);
    // Extend the trace by two virtual Cα at each end ("rebuild ends").
    for (let i = 0, j = 0; i < 5; i++, j++) tmpcoords[j] = getCa(i);
    for (let i = 2, j = 0; i < 5; i++, j++) cacoords[j] = getCa(i);
    for (let i = 0, j = 0; i < 3; i++, j++) tmpstat[j] = getCa(i);
    superimpose2(tmpstat, cacoords, 3, tmpcoords, 5);
    for (let i = -2, j = 0; i < 0; i++, j++) setCa(i, tmpcoords[j]);
    for (let i = L - 5, j = 0; i < L; i++, j++) tmpcoords[j] = getCa(i);
    for (let i = L - 5, j = 0; i < L - 2; i++, j++) cacoords[j] = getCa(i);
    for (let i = L - 3, j = 0; i < L; i++, j++) tmpstat[j] = getCa(i);
    superimpose2(tmpstat, cacoords, 3, tmpcoords, 5);
    for (let i = L - 3, j = 0; i < L; i++, j++) setCa(i + 3, tmpcoords[j + 3]);

    const rbins = new Int32Array((L + 1) * 3);
    for (let i = 0; i < L + 1; i++) {
      const [x1, y1, z1] = getCa(i - 2);
      const [x2, y2, z2] = getCa(i - 1);
      const [x3, y3, z3] = getCa(i);
      const [x4, y4, z4] = getCa(i + 1);
      const r13a = calcDistance(x1, y1, z1, x3, y3, z3);
      const r13b = calcDistance(x2, y2, z2, x4, y4, z4);
      const r14 = calcR14(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4);
      let b1 = toInt((r13a - 4.6) / 0.3);
      let b2 = toInt((r13b - 4.6) / 0.3);
      let b3 = toInt((r14 + 11) / 0.3);
      if (b1 < 0) b1 = 0;
      if (b2 < 0) b2 = 0;
      if (b3 < 0) b3 = 0;
      if (b1 > 9) b1 = 9;
      if (b2 > 9) b2 = 9;
      if (b3 > 73) b3 = 73;
      rbins[i * 3] = b1;
      rbins[i * 3 + 1] = b2;
      rbins[i * 3 + 2] = b3;
    }
    this.rbins = rbins;
    return rbins;
  }

  rebuildBackbone(): void {
    const rbins = this.prepareRbins();
    const T = this.T;
    const cacoords = points(8), tmpcoords = points(8), tmpstat = points(8);
    let prevres: Residue | null = null;
    let res: Residue | null = this.residues[0] ?? null;
    let total = 0, maxrms = 0;
    for (let i = 0; i < this.L + 1; i++) {
      for (let p = 0; p < 4; p++) for (let k = 0; k < 3; k++) cacoords[p][k] = this.caX(i - 2 + p, k);
      const b1 = rbins[i * 3], b2 = rbins[i * 3 + 1], b3 = rbins[i * 3 + 2];
      const pro = prevres !== null && prevres.name === 'PRO';
      const bins = pro ? T.ncoProBins : T.ncoBins;
      const data = pro ? T.ncoProData : T.ncoData;
      let j = 0, besthit = 1000, bestpos = 0, hit: number;
      do {
        hit = Math.abs(bins[j * 3] - b1) + Math.abs(bins[j * 3 + 1] - b2) + 0.2 * Math.abs(bins[j * 3 + 2] - b3);
        if (hit < besthit) {
          besthit = hit;
          bestpos = j;
        }
        j++;
      } while (bins[j * 3] >= 0 && hit > 1e-3);
      for (let p = 0; p < 8; p++) {
        for (let k = 0; k < 3; k++) {
          const v = Math.fround(data[bestpos * 24 + p * 3 + k] / T.scale);
          tmpcoords[p][k] = v;
          if (p < 4) tmpstat[p][k] = v;
        }
      }
      const rmsd = superimpose2(cacoords, tmpstat, 4, tmpcoords, 8);
      total += rmsd;
      if (rmsd > maxrms) maxrms = rmsd;
      if (prevres) {
        this.addReplace(prevres, 'C  ', tmpcoords[4][0], tmpcoords[4][1], tmpcoords[4][2], FLAG_BACKBONE);
        this.addReplace(prevres, 'O  ', tmpcoords[5][0], tmpcoords[5][1], tmpcoords[5][2], FLAG_BACKBONE);
      }
      if (res) this.addReplace(res, 'N  ', tmpcoords[6][0], tmpcoords[6][1], tmpcoords[6][2], FLAG_BACKBONE);
      else if (prevres && this.emitTerminalOxt) {
        this.addReplace(prevres, 'OXT', tmpcoords[6][0], tmpcoords[6][1], tmpcoords[6][2], FLAG_BACKBONE);
      }
      prevres = res;
      if (res) res = res.next;
    }
    this.stats.backboneRmsdMean = total / this.L;
    this.stats.backboneRmsdMax = maxrms;
  }

  // ── Grid shared by clash fixing and H-bond optimisation (allocate_grid) ──

  /** Allocates (or rebuilds) the spatial grid over all atoms currently in the segment. */
  private allocateGrid(forceRebuild = false): Grid {
    if (this.grid && !forceRebuild) return this.grid;
    const first = this.residues[0]?.atoms[0];
    const min = [first?.x ?? 0, first?.y ?? 0, first?.z ?? 0];
    const max = [...min];
    for (const res of this.residues) {
      for (const a of res.atoms) {
        if (a.x < min[0]) min[0] = a.x;
        if (a.y < min[1]) min[1] = a.y;
        if (a.z < min[2]) min[2] = a.z;
        if (a.x > max[0]) max[0] = a.x;
        if (a.y > max[1]) max[1] = a.y;
        if (a.z > max[2]) max[2] = a.z;
      }
    }
    const nx = toInt((max[0] - min[0]) / GRID_RES);
    const ny = toInt((max[1] - min[1]) / GRID_RES);
    const nz = toInt((max[2] - min[2]) / GRID_RES);
    const cells: Atom[][] = Array.from({ length: (nx + 1) * (ny + 1) * (nz + 1) }, () => []);
    for (const res of this.residues) {
      for (const a of res.atoms) {
        const x = toInt((nx * (a.x - min[0])) / (max[0] - min[0]));
        const y = toInt((ny * (a.y - min[1])) / (max[1] - min[1]));
        const z = toInt((nz * (a.z - min[2])) / (max[2] - min[2]));
        a.gx = x;
        a.gy = y;
        a.gz = z;
        cells[(x * (ny + 1) + y) * (nz + 1) + z].push(a);
      }
    }
    this.grid = { nx, ny, nz, cells };
    return this.grid;
  }

  // ── Stage 2b: peptide-plane H-bond optimisation (optimize_backbone) ──────

  /** DSSP electrostatic H-bond energy of res's N-H to the closest carbonyl. */
  private hbEnergy(res: Residue, grid: Grid): number {
    let ene = 1e3;
    if (!res.prev) return ene;
    const Q = -27888.0;
    let c1: Atom | null = null, o1: Atom | null = null, n1: Atom | null = null;
    for (const a of res.prev.atoms) {
      if (isName(a, 'C', ' ')) c1 = a;
      if (isName(a, 'O', ' ')) o1 = a;
    }
    for (const a of res.atoms) {
      if (isName(a, 'N', ' ')) {
        n1 = a;
        break;
      }
    }
    let minDist2 = 1e10;
    let o2: Atom | null = null, c2: Atom | null = null;
    if (n1) {
      const { gx: i, gy: j, gz: k } = n1;
      for (let ii = i - 1; ii <= i + 1; ii++) {
        for (let jj = j - 1; jj <= j + 1; jj++) {
          for (let kk = k - 1; kk <= k + 1; kk++) {
            // `kk <= zgrid` (not `<`) is verbatim from pulchra.c.
            if (ii >= 0 && ii < grid.nx && jj >= 0 && jj < grid.ny && kk >= 0 && kk <= grid.nz) {
              for (const a of grid.cells[(ii * (grid.ny + 1) + jj) * (grid.nz + 1) + kk]) {
                if (isName(a, 'O', ' ') && Math.abs(a.res.locnum - n1.res.locnum) > 2) {
                  const dx = n1.x - a.x, dy = n1.y - a.y, dz = n1.z - a.z;
                  const dist = dx * dx + dy * dy + dz * dz;
                  if (dist < minDist2 && dist < 25.0) {
                    o2 = a;
                    minDist2 = dist;
                  }
                }
              }
            }
          }
        }
      }
    }
    if (n1 && o2 && c1 && o1) {
      for (const a of o2.res.atoms) {
        if (isName(a, 'C', ' ')) {
          c2 = a;
          break;
        }
      }
      if (c2) {
        let hx = o1.x - c1.x, hy = o1.y - c1.y, hz = o1.z - c1.z;
        const dd = -H_BOND_LENGTH_F / Math.sqrt(hx * hx + hy * hy + hz * hz);
        hx *= dd;
        hy *= dd;
        hz *= dd;
        hx += n1.x;
        hy += n1.y;
        hz += n1.z;
        this.addReplace(n1.res, 'H  ', hx, hy, hz, FLAG_BACKBONE);
        let dx = n1.x - o2.x, dy = n1.y - o2.y, dz = n1.z - o2.z;
        const dno = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dx = n1.x - c2.x;
        dy = n1.y - c2.y;
        dz = n1.z - c2.z;
        const dnc = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dx = hx - o2.x;
        dy = hy - o2.y;
        dz = hz - o2.z;
        const dho = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dx = hx - c2.x;
        dy = hy - c2.y;
        dz = hz - c2.z;
        const dhc = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dho < HB_EPS_F || dhc < HB_EPS_F || dnc < HB_EPS_F || dno < HB_EPS_F) ene = -10.0;
        else ene = 0.001 * (Q / dho - Q / dhc + Q / dnc - Q / dno);
      }
    }
    return ene;
  }

  /** Rotates the peptide plate (C, O of res.prev and N of res) about Cα–Cα. */
  private rotPeptide(res: Residue, angle: number): void {
    if (!res.prev) return;
    let c: Atom | null = null, o: Atom | null = null, n: Atom | null = null;
    let ca1: Atom | null = null, ca2: Atom | null = null;
    for (const a of res.prev.atoms) {
      if (isName(a, 'C', 'A')) ca1 = a;
      if (isName(a, 'C', ' ')) c = a;
      if (isName(a, 'O', ' ')) o = a;
    }
    for (const a of res.atoms) {
      if (isName(a, 'C', 'A')) ca2 = a;
      if (isName(a, 'N', ' ')) n = a;
    }
    if (!(c && o && n && ca1 && ca2)) return;
    let u = ca2.x - ca1.x, v = ca2.y - ca1.y, w = ca2.z - ca1.z;
    const dd = 1.0 / Math.sqrt(u * u + v * v + w * w);
    u *= dd;
    v *= dd;
    w *= dd;
    // sinf/cosf of a float argument (angle is in units of 10°).
    const arg = Math.fround((10.0 * M_PI * angle) / 180.0);
    const sa = Math.fround(Math.sin(arg));
    const cs = Math.fround(Math.cos(arg));
    for (const atom of [n, c, o]) {
      const x = atom.x - ca1.x, y = atom.y - ca1.y, z = atom.z - ca1.z;
      const ux = u * x, uy = u * y, uz = u * z;
      const vx = v * x, vy = v * y, vz = v * z;
      const wx = w * x, wy = w * y, wz = w * z;
      const nx = u * (ux + vy + wz) + (x * (v * v + w * w) - u * (vy + wz)) * cs + (-wy + vz) * sa;
      const ny = v * (ux + vy + wz) + (y * (u * u + w * w) - v * (ux + wz)) * cs + (wx - uz) * sa;
      const nz = w * (ux + vy + wz) + (z * (u * u + v * v) - w * (ux + vy)) * cs + (-vx + uy) * sa;
      atom.x = nx + ca1.x;
      atom.y = ny + ca1.y;
      atom.z = nz + ca1.z;
    }
  }

  optimizeHbonds(): void {
    const grid = this.allocateGrid();
    let tot1 = 0, tot2 = 0;
    for (const res of this.residues) {
      const ene = this.hbEnergy(res, grid);
      if (ene < -0.5) tot1 += ene;
    }
    for (const res of this.residues) {
      if (res.type === TYPE_PRO) continue;
      let ene = this.hbEnergy(res, grid);
      if (ene < 1.0) {
        let minEne = ene;
        this.rotPeptide(res, -1.1);
        let best = 0;
        for (let i = -10; i < 10; i++) {
          this.rotPeptide(res, 0.1);
          ene = this.hbEnergy(res, grid);
          if (ene < minEne) {
            best = i;
            minEne = ene;
          }
        }
        this.rotPeptide(res, -0.9);
        ene = this.hbEnergy(res, grid);
        if (minEne < ene) {
          this.rotPeptide(res, 0.1 * best);
          this.hbEnergy(res, grid);
        }
      }
    }
    for (const res of this.residues) {
      const ene = this.hbEnergy(res, grid);
      if (ene < -0.5) tot2 += ene;
    }
    this.stats.hbondEnergyBefore = tot1;
    this.stats.hbondEnergyAfter = tot2;
  }

  // ── Stage 3: side chains (rebuild_sidechains / optimize_exvol) ──────────

  /** Local Cα frame (v1, v2, v3) of residue i, as in rebuild_sidechains. */
  private localFrame(i: number): number[][] {
    const x2 = this.caX(i - 1, 0), y2 = this.caX(i - 1, 1), z2 = this.caX(i - 1, 2);
    const x3 = this.caX(i, 0), y3 = this.caX(i, 1), z3 = this.caX(i, 2);
    const x4 = this.caX(i + 1, 0), y4 = this.caX(i + 1, 1), z4 = this.caX(i + 1, 2);
    const v1 = [x4 - x2, y4 - y2, z4 - z2];
    const v2a = [x4 - x3, y4 - y3, z4 - z3];
    const v2b = [x3 - x2, y3 - y2, z3 - z2];
    const v2 = [0, 0, 0], v3 = [0, 0, 0];
    cross(v2a, v2b, v2);
    cross(v1, v2, v3);
    norm(v1);
    norm(v2);
    norm(v3);
    return [v1, v2, v3];
  }

  /** Rotamer rows pos..pos+nsc-1 placed in residue i's frame (sc[0] = Cα). */
  private placeRotamer(i: number, frame: number[][], pos: number, nsc: number): number[][] {
    const vv = frame.map((v) => [...v]);
    const lsys = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const sc = points(nsc);
    for (let j = 0; j < nsc; j++) for (let k = 0; k < 3; k++) sc[j][k] = this.rot(pos + j, k);
    superimpose2(vv, lsys, 3, sc, nsc);
    const x3 = this.caX(i, 0), y3 = this.caX(i, 1), z3 = this.caX(i, 2);
    for (let j = 0; j < nsc; j++) {
      sc[j][0] += x3;
      sc[j][1] += y3;
      sc[j][2] += z3;
    }
    return sc;
  }

  private applyRotamer(res: Residue, sc: number[][], nsc: number): void {
    const names = HEAVY_ATOMS[res.type];
    for (let j = 1; j < nsc; j++) this.addReplace(res, names[j - 1], sc[j][0], sc[j][1], sc[j][2], FLAG_SIDECHAIN);
  }

  rebuildSidechains(useSidechainCenters: boolean): void {
    const rbins = this.prepareRbins();
    const T = this.T;
    const sorted = new Float64Array((this.L + 1) * 20);
    this.sorted = sorted;
    let res = this.residues[0];
    for (let i = 0; i < this.L; i++) {
      if (res.name === 'GLY' || !res.protein) {
        if (res.next) res = res.next;
        continue;
      }
      const b1 = rbins[i * 3], b2 = rbins[i * 3 + 1], b3 = rbins[i * 3 + 2];
      const frame = this.localFrame(i);
      // Gather the 10 closest rotamer bins (stable insertion sort on hit).
      const base = i * 20;
      for (let j = 0; j < 10; j++) sorted[base + j * 2] = 500;
      for (const j of T.rotRowsByType[res.type]) {
        const hit =
          Math.abs(T.rotIdx[j * 5 + 1] - b1) + Math.abs(T.rotIdx[j * 5 + 2] - b2) + 0.2 * Math.abs(T.rotIdx[j * 5 + 3] - b3);
        if (hit < sorted[base + 18]) {
          let k = 9;
          while (k >= 0 && hit < sorted[base + k * 2]) k--;
          k++;
          for (let l = 9; l > k; l--) {
            sorted[base + l * 2] = sorted[base + (l - 1) * 2];
            sorted[base + l * 2 + 1] = sorted[base + (l - 1) * 2 + 1];
          }
          sorted[base + k * 2] = hit;
          sorted[base + k * 2 + 1] = j;
        }
      }
      const bestpos = sorted[base + 1];
      let pos = T.rotPos[bestpos];
      const nsc = NHEAVY[res.type] + 1;
      if (useSidechainCenters) {
        // PDB-SG mode: of the two closest bins, keep the rotamer whose centre
        // of mass is nearest the input side-chain centre.
        let bestdd = 100;
        let crot = 0;
        for (let l = 0; l < 2; l++) {
          const cpos = sorted[base + l * 2 + 1];
          for (let m = 0; m < T.rotIdx[cpos * 5 + 4]; m++) {
            const p = T.rotPos[cpos] + nsc * m;
            const sc = this.placeRotamer(i, frame, p, nsc);
            let cmx = 0, cmy = 0, cmz = 0;
            for (let j = 0; j < nsc; j++) {
              cmx += sc[j][0];
              cmy += sc[j][1];
              cmz += sc[j][2];
            }
            cmx /= nsc;
            cmy /= nsc;
            cmz /= nsc;
            let ddx = res.cmx - cmx, ddy = res.cmy - cmy, ddz = res.cmz - cmz;
            ddx *= ddx;
            ddy *= ddy;
            ddz *= ddz;
            const ddd = ddx + ddy + ddz;
            if (ddd < bestdd) {
              bestdd = ddd;
              crot = p;
            }
          }
        }
        pos = crot;
      }
      if (res.type === TYPE_MET) {
        // Map selenomethionine SE to MET SD before rotamer placement/preservation so MSE never emits both SE and SD.
        const seIdx = res.atoms.findIndex((a) => a.name === 'SE ');
        if (seIdx >= 0) {
          if (res.atoms.some((a) => a.name === 'SD ')) res.atoms.splice(seIdx, 1);
          else {
            res.atoms[seIdx].name = 'SD ';
            res.atoms[seIdx].element = 'S';
          }
        }
      }
      this.applyRotamer(res, this.placeRotamer(i, frame, pos, nsc), nsc);
      if (res.next) res = res.next;
    }
  }

  /** pulchra.c get_conflicts(): heavy-atom pairs closer than 1.6 Å. */
  private conflicts(res: Residue, grid: Grid): number {
    const lim = SG_XVOL_DIST * SG_XVOL_DIST;
    let con = 0;
    for (const atom of res.atoms) {
      const { gx: i, gy: j, gz: k } = atom;
      for (let ii = i - 2; ii <= i + 2; ii++) {
        if (ii < 0 || ii >= grid.nx) continue;
        for (let jj = j - 2; jj <= j + 2; jj++) {
          if (jj < 0 || jj >= grid.ny) continue;
          for (let kk = k - 2; kk <= k + 2; kk++) {
            if (kk < 0 || kk >= grid.nz) continue;
            for (const atom2 of grid.cells[(ii * (grid.ny + 1) + jj) * (grid.nz + 1) + kk]) {
              let merged = false;
              if (res === atom2.res) {
                if (atom.flag & FLAG_SIDECHAIN && atom2.flag & FLAG_SIDECHAIN) merged = true;
                if (atom.flag & FLAG_BACKBONE && atom2.flag & FLAG_BACKBONE) merged = true;
                if (isName(atom, 'C', 'A') && isName(atom2, 'C', 'B')) merged = true;
                if (isName(atom, 'C', 'B') && isName(atom2, 'C', 'A')) merged = true;
                if (res.name[0] === 'P') {
                  if (isName(atom, 'C', 'D') && isName(atom2, 'N', ' ')) merged = true;
                  if (isName(atom, 'N', ' ') && isName(atom2, 'C', 'D')) merged = true;
                }
              } else if (res.next === atom2.res || res === atom2.res.next) {
                if (isName(atom, 'C', ' ') && isName(atom2, 'N', ' ')) merged = true;
                if (isName(atom, 'N', ' ') && isName(atom2, 'C', ' ')) merged = true;
              }
              if (atom.flag & FLAG_BACKBONE && atom2.flag & FLAG_BACKBONE) merged = true;
              if (atom.flag & FLAG_SCM || atom2.flag & FLAG_SCM) merged = true;
              if (!merged) {
                let dx = atom.x - atom2.x;
                dx *= dx;
                let dy = atom.y - atom2.y;
                dy *= dy;
                let dz = atom.z - atom2.z;
                dz *= dz;
                if (dx + dy + dz < lim) con++;
              }
            }
          }
        }
      }
    }
    return con;
  }

  private totalConflicts(grid: Grid): { total: number; max: number } {
    let total = 0, max = 0;
    for (const res of this.residues) {
      if (!res.protein) continue;
      const con = this.conflicts(res, grid);
      if (con > 0) {
        total += con;
        if (con > max) max = con;
      }
    }
    return { total, max };
  }

  fixClashes(): void {
    const sorted = this.sorted;
    if (!sorted) return;
    const T = this.T;
    // Rebuild the spatial grid right before clash fixing so sidechain atoms are always indexed
    // even when optimizeHbonds() already built an earlier backbone-only grid.
    const grid = this.allocateGrid(true);
    let iter = 0;
    let remaining = 0;
    do {
      const { total, max } = this.totalConflicts(grid);
      if (iter === 0) this.stats.clashesBefore = total;
      remaining = total;
      if (total === 0) break;
      let totcon = 0;
      if (max > 0) {
        let res: Residue | null = this.residues[0];
        for (let i = 0; i < this.L && res; i++) {
          if (res.name === 'GLY' || !res.protein) {
            if (res.next) res = res.next;
            continue;
          }
          const nsc = NHEAVY[res.type] + 1;
          const frame = this.localFrame(i);
          let con = this.conflicts(res, grid);
          if (con > 0) {
            let bestpos = 0;
            let con0 = 100;
            for (let l = 0; l < 10; l++) {
              const cpos = sorted[i * 20 + l * 2 + 1];
              for (let m = 0; m < T.rotIdx[cpos * 5 + 4]; m++) {
                const pos = T.rotPos[cpos] + nsc * m;
                if (res.type === TYPE_MET) {
        // Map selenomethionine SE to MET SD before rotamer placement/preservation so MSE never emits both SE and SD.
        const seIdx = res.atoms.findIndex((a) => a.name === 'SE ');
        if (seIdx >= 0) {
          if (res.atoms.some((a) => a.name === 'SD ')) res.atoms.splice(seIdx, 1);
          else {
            res.atoms[seIdx].name = 'SD ';
            res.atoms[seIdx].element = 'S';
          }
        }
      }
      this.applyRotamer(res, this.placeRotamer(i, frame, pos, nsc), nsc);
                con = this.conflicts(res, grid);
                if (con < con0) {
                  con0 = con;
                  bestpos = pos;
                }
                if (con === 0) break;
              }
              if (con === 0) break;
            }
            totcon += con0;
            this.applyRotamer(res, this.placeRotamer(i, frame, bestpos, nsc), nsc);
          }
          res = res.next;
        }
      }
      remaining = totcon;
      iter++;
    } while (iter < XVOL_ITER);
    this.stats.clashesAfter = remaining;
  }

  /** chirality_check(): mirrors D-amino-acid side chains through the N-Cα-C frame. */
  checkChirality(): void {
    for (const res of this.residues) {
      const ca = findAtom(res, 'CA '), n = findAtom(res, 'N  '), c = findAtom(res, 'C  '), cb = findAtom(res, 'CB ');
      if (!(ca && n && c && cb)) continue;
      if (!(torsion(ca, n, c, cb) < 0)) continue;
      const xx = ca.x - n.x, xy = ca.y - n.y, xz = ca.z - n.z;
      const yx = c.x - ca.x, yy = c.y - ca.y, yz = c.z - ca.z;
      let rx = xx - yx, ry = xy - yy, rz = xz - yz;
      const dd = Math.sqrt(rx * rx + ry * ry + rz * rz);
      rx /= dd;
      ry /= dd;
      rz /= dd;
      const cost = -1, sint = 0;
      for (const atom of res.atoms) {
        if (!(atom.flag & FLAG_SIDECHAIN)) continue;
        const px = atom.x - ca.x, py = atom.y - ca.y, pz = atom.z - ca.z;
        let qx = 0, qy = 0, qz = 0;
        qx += (cost + (1 - cost) * rx * rx) * px;
        qx += ((1 - cost) * rx * ry - rz * sint) * py;
        qx += ((1 - cost) * rx * rz + ry * sint) * pz;
        qy += ((1 - cost) * rx * ry + rz * sint) * px;
        qy += (cost + (1 - cost) * ry * ry) * py;
        qy += ((1 - cost) * ry * rz - rx * sint) * pz;
        qz += ((1 - cost) * rx * rz - ry * sint) * px;
        qz += ((1 - cost) * ry * rz + rx * sint) * py;
        qz += (cost + (1 - cost) * rz * rz) * pz;
        atom.x = qx + ca.x;
        atom.y = qy + ca.y;
        atom.z = qz + ca.z;
      }
      this.stats.dAminoAcidsFixed++;
    }
  }
}

/** calc_torsion() in degrees; 360 flags a degenerate geometry. */
function torsion(a1: Atom, a2: Atom, a3: Atom, a4: Atom): number {
  const v12x = a1.x - a2.x, v12y = a1.y - a2.y, v12z = a1.z - a2.z;
  const v43x = a4.x - a3.x, v43y = a4.y - a3.y, v43z = a4.z - a3.z;
  const zx = a2.x - a3.x, zy = a2.y - a3.y, zz = a2.z - a3.z;
  const px = zy * v12z - v12y * zz, py = zz * v12x - v12z * zx, pz = zx * v12y - v12x * zy;
  const xx = zy * v43z - v43y * zz, xy = zz * v43x - v43z * zx, xz = zx * v43y - v43x * zy;
  const yx = zy * xz - xy * zz, yy = zz * xx - xz * zx, yz = zx * xy - xx * zy;
  let u = xx * xx + xy * xy + xz * xz;
  let v = yx * yx + yy * yy + yz * yz;
  let angle = 360;
  if (u < 0 || v < 0) return angle;
  u = (px * xx + py * xy + pz * xz) / Math.sqrt(u);
  v = (px * yx + py * yy + pz * yz) / Math.sqrt(v);
  if (u !== 0 || v !== 0) angle = (Math.atan2(v, u) * 180) / M_PI;
  return angle;
}

// ─── Public API ──────────────────────────────────────────────────────────────

export interface PulchraOptions {
  /** Refine the Cα trace (bond 3.8 Å, angles 80-150°, Cα clashes) before rebuilding. Default false. */
  optimizeCa?: boolean;
  /** Free Cα displacement before the restraint to the input engages (Å, `-u`). Default 0.5. */
  maxCaShift?: number;
  /** Use 2.9 Å virtual bonds for prolines with short Cα-Cα (`-p`, needs optimizeCa). Default false. */
  detectCisProline?: boolean;
  /** Translate the output so its heavy-atom centroid is at the origin (`-n`). Default false. */
  centerChain?: boolean;
  /** Rebuild N, C, O and terminal OXT (`-b` skips). Default true. */
  reconstructBackbone?: boolean;
  /** Rotate peptide plates to improve backbone H-bonds (`-q`). Default false. */
  optimizeHbonds?: boolean;
  /** Rebuild side-chain heavy atoms (`-s` skips). Default true. */
  reconstructSidechains?: boolean;
  /** Swap rotamers to remove steric clashes (`-o` disables). Default true. */
  fixClashes?: boolean;
  /** Mirror D-configured side chains back to L (`-z` disables). Default true. */
  checkChirality?: boolean;
  /** Add backbone amide hydrogens (H) to non-proline residues. Default false. */
  outputHydrogens?: boolean;
  /** PDB-SG input (`-g`): choose rotamers using SC/CM side-chain-centre pseudo-atoms. Default false. */
  useSidechainCenters?: boolean;
  /**
   * Keep input atoms that PULCHRA would rebuild (3.06 default). Defaults to
   * `!optimizeCa`: once Cα move, the input N/C/O/side chains no longer fit.
   */
  preserveInput?: boolean;
  /**
   * Thread a new sequence onto the trace: one-letter codes for all chains
   * (concatenated, or separated by '/' or ':'), or one string per chain.
   * Residues whose identity changes lose their input side-chain atoms.
   */
  sequence?: string | readonly string[];
}

export interface PulchraAtomRecord {
  name: string;
  element: string;
  x: number;
  y: number;
  z: number;
}

export interface PulchraResidueRecord {
  chainId: string;
  resSeq: number;
  iCode: string;
  resName: string;
  bfactor: number;
  atoms: PulchraAtomRecord[];
}

export interface PulchraChainResult {
  chainId: string;
  residues: PulchraResidueRecord[];
  stats: ChainStats;
}

export interface PulchraResult {
  pdbText: string;
  chains: PulchraChainResult[];
  sequence: string;
  residueCount: number;
  atomCount: number;
  warnings: string[];
}

function splitSequence(seq: string | readonly string[], chainLengths: number[]): string[] {
  const clean = (s: string) => s.replace(/\s+/g, '').toUpperCase();
  let parts: string[];
  if (typeof seq !== 'string') parts = seq.map(clean);
  else if (/[/:]/.test(seq)) parts = clean(seq).split(/[/:]/);
  else {
    const all = clean(seq);
    const total = chainLengths.reduce((a, b) => a + b, 0);
    if (all.length !== total) {
      throw new Error(`PULCHRA: sequence has ${all.length} residues but the trace has ${total}`);
    }
    parts = [];
    let o = 0;
    for (const n of chainLengths) {
      parts.push(all.slice(o, o + n));
      o += n;
    }
  }
  if (parts.length !== chainLengths.length) {
    throw new Error(`PULCHRA: sequence has ${parts.length} chain(s) but the trace has ${chainLengths.length}`);
  }
  parts.forEach((p, i) => {
    if (p.length !== chainLengths[i]) {
      throw new Error(`PULCHRA: chain ${i + 1} sequence has ${p.length} residues but the trace has ${chainLengths[i]}`);
    }
    const bad = p.match(/[^ACDEFGHIKLMNPQRSTVWY]/);
    if (bad) throw new Error(`PULCHRA: unsupported residue code '${bad[0]}' in sequence`);
  });
  return parts;
}

function threadSequence(chains: ParsedChain[], seq: string | readonly string[]): void {
  const parts = splitSequence(seq, chains.map((c) => c.residues.length));
  chains.forEach((chain, ci) => {
    chain.residues.forEach((res, ri) => {
      const type = aaType(parts[ci][ri]);
      if (type === res.type) return;
      res.type = type;
      res.name = AA_NAMES[type];
      res.protein = true;
      res.pdbsg = false;
      res.atoms = res.atoms.filter((a) => a.flag & FLAG_BACKBONE);
      const ca = res.atoms.find((a) => isName(a, 'C', 'A'));
      if (ca) {
        res.cmx = ca.x;
        res.cmy = ca.y;
        res.cmz = ca.z;
      }
    });
  });
}

/** Output atoms of a residue: list order, PULCHRA's write_pdb() filter. */
function outputAtoms(res: Residue, hydrogens: boolean): Atom[] {
  const heavy: Atom[] = [];
  let oxt: Atom | null = null;
  let h: Atom | null = null;
  for (const a of res.atoms) {
    const [c0, c1] = a.name;
    if ((c0 === 'D' && c1 === 'U') || (c0 === 'S' && c1 === 'C') || (c0 === 'C' && c1 === 'M')) continue;
    if (c0 === 'O' && c1 === 'X') oxt = a;
    else if (c0 === 'H') h = a;
    else heavy.push(a);
  }
  if (oxt) heavy.push(oxt);
  if (hydrogens && h) heavy.push(h);
  return heavy;
}

const fmt = (v: number, w: number, d: number) => v.toFixed(d).padStart(w);

function emptyChainStats(): ChainStats {
  return {
    backboneRmsdMean: null,
    backboneRmsdMax: null,
    caOptimizationSteps: 0,
    cisProlines: 0,
    hbondEnergyBefore: null,
    hbondEnergyAfter: null,
    clashesBefore: null,
    clashesAfter: null,
    dAminoAcidsFixed: 0,
  };
}

function splitContiguousSegments(residues: Residue[]): Residue[][] {
  if (residues.length === 0) return [];
  const segments: Residue[][] = [[residues[0]]];
  for (let i = 1; i < residues.length; i++) {
    const prevCa = findAtom(residues[i - 1], 'CA ');
    const currCa = findAtom(residues[i], 'CA ');
    const d =
      prevCa && currCa
        ? calcDistance(prevCa.x, prevCa.y, prevCa.z, currCa.x, currCa.y, currCa.z)
        : 0;
    if (d > CHAIN_BREAK_CA_DIST) segments.push([residues[i]]);
    else segments[segments.length - 1].push(residues[i]);
  }
  return segments;
}

function aggregateSegmentStats(segStats: Array<{ length: number; stats: ChainStats }>): ChainStats {
  if (segStats.length === 1) return segStats[0].stats;
  const out = emptyChainStats();
  let rmsdNum = 0;
  let rmsdDen = 0;
  let maxRmsd: number | null = null;
  let hbBefore: number | null = null;
  let hbAfter: number | null = null;
  let clBefore: number | null = null;
  let clAfter: number | null = null;
  for (const { length, stats } of segStats) {
    if (stats.backboneRmsdMean !== null) {
      rmsdNum += stats.backboneRmsdMean * length;
      rmsdDen += length;
    }
    if (stats.backboneRmsdMax !== null) {
      maxRmsd = maxRmsd === null ? stats.backboneRmsdMax : Math.max(maxRmsd, stats.backboneRmsdMax);
    }
    if (stats.caOptimizationSteps > out.caOptimizationSteps) {
      out.caOptimizationSteps = stats.caOptimizationSteps;
    }
    out.cisProlines += stats.cisProlines;
    if (stats.hbondEnergyBefore !== null) hbBefore = (hbBefore ?? 0) + stats.hbondEnergyBefore;
    if (stats.hbondEnergyAfter !== null) hbAfter = (hbAfter ?? 0) + stats.hbondEnergyAfter;
    if (stats.clashesBefore !== null) clBefore = (clBefore ?? 0) + stats.clashesBefore;
    if (stats.clashesAfter !== null) clAfter = (clAfter ?? 0) + stats.clashesAfter;
    out.dAminoAcidsFixed += stats.dAminoAcidsFixed;
  }
  out.backboneRmsdMean = rmsdDen > 0 ? rmsdNum / rmsdDen : null;
  out.backboneRmsdMax = maxRmsd;
  out.hbondEnergyBefore = hbBefore;
  out.hbondEnergyAfter = hbAfter;
  out.clashesBefore = clBefore;
  out.clashesAfter = clAfter;
  return out;
}

function formatPreservedAtomLine(rec: PreservedRecord, serial: number, dx: number, dy: number, dz: number): string {
  const prefix = rec.line.slice(0, 6);
  const mid = rec.line.slice(11, 30);
  const coords =
    dx === 0 && dy === 0 && dz === 0 && rec.line.length >= 54
      ? rec.line.slice(30, 54)
      : `${fmt(rec.x - dx, 8, 3)}${fmt(rec.y - dy, 8, 3)}${fmt(rec.z - dz, 8, 3)}`;
  const tail = rec.line.length > 54 ? rec.line.slice(54) : '';
  return `${prefix}${String(serial % 100000).padStart(5)}${mid}${coords}${tail}`;
}

/**
 * Rebuilds an all-atom model from PDB text (Cα-only, backbone or partial).
 * Synchronous; pass tables from `loadPulchraTables()`.
 */
export function reconstructPdb(pdbText: string, tables: PulchraTables, options: PulchraOptions = {}): PulchraResult {
  const o = {
    optimizeCa: options.optimizeCa ?? false,
    maxCaShift: options.maxCaShift ?? 0.5,
    detectCisProline: options.detectCisProline ?? false,
    centerChain: options.centerChain ?? false,
    reconstructBackbone: options.reconstructBackbone ?? true,
    optimizeHbonds: options.optimizeHbonds ?? false,
    reconstructSidechains: options.reconstructSidechains ?? true,
    fixClashes: options.fixClashes ?? true,
    checkChirality: options.checkChirality ?? true,
    outputHydrogens: options.outputHydrogens ?? false,
    useSidechainCenters: options.useSidechainCenters ?? false,
  };
  const preserve = options.preserveInput ?? !o.optimizeCa;
  if (!Number.isFinite(o.maxCaShift) || o.maxCaShift < 0) throw new Error('PULCHRA: maxCaShift must be >= 0');
  const warnings: string[] = [];
  const { chains, preservedNucleic, preservedHetatm } = parsePdbChains(pdbText, warnings);
  if (!chains.length) throw new Error('PULCHRA: no protein residues with Cα atoms found in the input');
  if (options.sequence !== undefined && options.sequence !== '') threadSequence(chains, options.sequence);

  const chainStats: ChainStats[] = [];
  for (const chain of chains) {
    const chainLabel = chain.chainId.trim() || '_';
    const segments = splitContiguousSegments(chain.residues);
    if (segments.length > 1) {
      warnings.push(
        `Chain ${chainLabel} split into ${segments.length} contiguous segments at Cα–Cα break(s) (> ${CHAIN_BREAK_CA_DIST} Å).`,
      );
    }
    const segStats: Array<{ length: number; stats: ChainStats }> = [];
    const rebuiltResidues: Residue[] = [];
    segments.forEach((seg, segIdx) => {
      seg.forEach((r, idx) => {
        r.locnum = idx;
        r.prev = seg[idx - 1] ?? null;
        r.next = seg[idx + 1] ?? null;
      });
      const isTerminalSegment = segIdx === segments.length - 1;
      if (seg.length < 5) {
        warnings.push(
          `Chain ${chainLabel} segment (${seg.length} residue(s), L < 5) preserved as-is without PULCHRA reconstruction.`,
        );
        segStats.push({ length: seg.length, stats: emptyChainStats() });
        rebuiltResidues.push(...seg);
        return;
      }
      const b = new ChainBuilder(seg, tables, preserve, isTerminalSegment);
      if (o.optimizeCa) b.optimizeCa(o.maxCaShift, o.detectCisProline);
      if (o.reconstructBackbone) {
        b.rebuildBackbone();
        if (o.optimizeHbonds) b.optimizeHbonds();
      }
      if (o.reconstructSidechains) {
        b.rebuildSidechains(o.useSidechainCenters);
        if (o.fixClashes) b.fixClashes();
        if (o.checkChirality) b.checkChirality();
      }
      segStats.push({ length: seg.length, stats: b.stats });
      rebuiltResidues.push(...seg);
    });
    chain.residues = rebuiltResidues;
    chainStats.push(aggregateSegmentStats(segStats));
  }

  const allResidues = chains.flatMap((c) => c.residues);
  let shiftX = 0;
  let shiftY = 0;
  let shiftZ = 0;
  if (o.centerChain) {
    let cx = 0, cy = 0, cz = 0, n = 0;
    for (const res of allResidues) {
      for (const a of outputAtoms(res, false)) {
        cx += a.x;
        cy += a.y;
        cz += a.z;
        n++;
      }
    }
    cx /= n;
    cy /= n;
    cz /= n;
    shiftX = cx;
    shiftY = cy;
    shiftZ = cz;
    for (const res of allResidues) {
      for (const a of res.atoms) {
        a.x -= cx;
        a.y -= cy;
        a.z -= cz;
      }
    }
  }

  // Amide H from the final geometry: N - 1.081 Å along the previous C→O.
  for (const res of allResidues) res.atoms = res.atoms.filter((a) => a.name[0] !== 'H');
  if (o.outputHydrogens) {
    for (const res of allResidues) {
      if (res.type === TYPE_PRO || !res.prev) continue;
      const n = findAtom(res, 'N  '), c = findAtom(res.prev, 'C  '), ox = findAtom(res.prev, 'O  ');
      if (!n || !c || !ox) continue;
      let hx = ox.x - c.x, hy = ox.y - c.y, hz = ox.z - c.z;
      const dd = -H_BOND_LENGTH_F / Math.sqrt(hx * hx + hy * hy + hz * hz);
      hx = hx * dd + n.x;
      hy = hy * dd + n.y;
      hz = hz * dd + n.z;
      res.atoms.push(newAtom(res, 'H  ', hx, hy, hz, FLAG_BACKBONE, 'H'));
    }
  }

  const lines = [`REMARK 999 REBUILT BY PULCHRA V.${PULCHRA_VERSION} (TypeScript port)`];
  const chainResults: PulchraChainResult[] = [];
  let serial = 1;
  let atomCount = 0;
  let sequence = '';
  chains.forEach((chain, ci) => {
    const residues: PulchraResidueRecord[] = [];
    let last: Residue | null = null;
    for (const res of chain.residues) {
      const atoms = outputAtoms(res, o.outputHydrogens);
      const record: PulchraResidueRecord = {
        chainId: res.chainId, resSeq: res.num, iCode: res.iCode, resName: res.name, bfactor: res.bfactor, atoms: [],
      };
      for (const a of atoms) {
        if (!Number.isFinite(a.x) || !Number.isFinite(a.y) || !Number.isFinite(a.z)) {
          throw new Error(`PULCHRA: non-finite coordinates for ${res.name} ${res.num} ${a.name.trim()} (degenerate Cα trace?)`);
        }
        const name = a.name.trim();
        const element = a.element || inferAtomElement(a.name);
        const atomField4 =
          name.length >= 4 || element.length === 2
            ? name.slice(0, 4).padEnd(4, ' ')
            : ` ${name.padEnd(3, ' ')}`;
        const elem2 = element.slice(0, 2).toUpperCase().padStart(2, ' ');
        record.atoms.push({ name, element, x: a.x, y: a.y, z: a.z });
        lines.push(
          `ATOM  ${String(serial % 100000).padStart(5)} ${atomField4} ${res.name} ${res.chainId}` +
            `${String(res.num).padStart(4)}${res.iCode}   ${fmt(a.x, 8, 3)}${fmt(a.y, 8, 3)}${fmt(a.z, 8, 3)}` +
            `  1.00${fmt(res.bfactor, 6, 2)}          ${elem2}`,
        );
        serial++;
        atomCount++;
      }
      residues.push(record);
      sequence += SHORT_AA_NAMES[res.type];
      last = res;
    }
    if (last) {
      lines.push(`TER   ${String(serial % 100000).padStart(5)}      ${last.name} ${last.chainId}${String(last.num).padStart(4)}${last.iCode}`);
      serial++;
    }
    chainResults.push({ chainId: chain.chainId, residues, stats: chainStats[ci] });
  });

  if (preservedNucleic.length > 0) {
    const nucOrder: string[] = [];
    const nucByChain = new Map<string, PreservedRecord[]>();
    for (const rec of preservedNucleic) {
      let arr = nucByChain.get(rec.chainId);
      if (!arr) {
        arr = [];
        nucByChain.set(rec.chainId, arr);
        nucOrder.push(rec.chainId);
      }
      arr.push(rec);
    }
    for (const ch of nucOrder) {
      const recs = nucByChain.get(ch)!;
      for (const rec of recs) {
        lines.push(formatPreservedAtomLine(rec, serial, shiftX, shiftY, shiftZ));
        serial++;
        atomCount++;
      }
      const lastNuc = recs[recs.length - 1];
      const lastResName = lastNuc.line.slice(17, 20);
      const lastResSeq = lastNuc.line.slice(22, 27);
      lines.push(`TER   ${String(serial % 100000).padStart(5)}      ${lastResName} ${lastNuc.chainId}${lastResSeq}`);
      serial++;
    }
  }

  for (const rec of preservedHetatm) {
    lines.push(formatPreservedAtomLine(rec, serial, shiftX, shiftY, shiftZ));
    serial++;
    atomCount++;
  }

  lines.push('END', '');

  return {
    pdbText: lines.join('\n'),
    chains: chainResults,
    sequence,
    residueCount: allResidues.length,
    atomCount,
    warnings,
  };
}

/** Async convenience wrapper: loads (or reuses) the tables, then reconstructs. */
export async function runPulchra(
  pdbText: string,
  options: PulchraOptions & LoadPulchraTablesOptions = {},
): Promise<PulchraResult> {
  const tables = await loadPulchraTables(options);
  return reconstructPdb(pdbText, tables, options);
}
