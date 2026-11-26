// cg2all structure assembly: StructureModule.output_to_opr_6D + libmodel.build_structure (rigid-group frames from
// the predicted backbone frame and torsion (cos, sin) pairs), libter.patch_termini, and PDB writing with
// MDTraj-standard (IUPAC/PDB) atom names. Lengths are nm until `toPdb`, which writes Å.

import {
  AMINO_ACIDS,
  MAX_ATOM,
  MAX_RIGID,
  MAX_SS,
  MAX_TORSION,
  PDB_ATOM_NAMES,
  RESIDUES,
  RIGID_GROUPS,
  RIGID_GROUPS_DEP,
  RIGID_TRANSFORMS,
  RIGID_TRANSFORMS_DEP,
} from './constants.js';
import { EPS } from './features.js';

const UNK = AMINO_ACIDS.length - 1;
const N_SCALAR_OUT = 2 * MAX_TORSION + MAX_SS;

function normSafe(v, index) {
  const s = Math.hypot(...v);
  if (s > EPS) return v.map((x) => x / s);
  const u = new Array(v.length).fill(0);
  u[index] = 1;
  return u;
}

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Decode the structure-module output (output_to_opr_6D, ss_dep=True).
 * @returns {{ rot: Float64Array (N×3×3, row-major, columns e0 e1 e2), trans: Float64Array (N×3, nm),
 *   torsion: Float64Array (N×8×2 unit cos/sin), ss: Int32Array (argmax of 4 SS logits) }}
 */
export function decodeOutputs(s, v, n, { fixAtom = false } = {}) {
  const rot = new Float64Array(n * 9);
  const trans = new Float64Array(n * 3);
  const torsion = new Float64Array(n * MAX_TORSION * 2);
  const ss = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const vb = i * 9;
    const v0 = [v[vb], v[vb + 1], v[vb + 2]];
    const v1 = [v[vb + 3], v[vb + 4], v[vb + 5]];
    const e0 = normSafe(v0, 0);
    const p = dot(e0, v1);
    const e1 = normSafe([v1[0] - e0[0] * p, v1[1] - e0[1] * p, v1[2] - e0[2] * p], 1);
    const e2 = cross(e0, e1);
    for (let r = 0; r < 3; r++) {
      rot[i * 9 + r * 3] = e0[r];
      rot[i * 9 + r * 3 + 1] = e1[r];
      rot[i * 9 + r * 3 + 2] = e2[r];
    }
    if (!fixAtom) for (let c = 0; c < 3; c++) trans[i * 3 + c] = 0.1 * v[vb + 6 + c];
    const sb = i * N_SCALAR_OUT;
    for (let t = 0; t < MAX_TORSION; t++) {
      const [c, sn] = normSafe([s[sb + 2 * t], s[sb + 2 * t + 1]], 0);
      torsion[(i * MAX_TORSION + t) * 2] = c;
      torsion[(i * MAX_TORSION + t) * 2 + 1] = sn;
    }
    let best = 0;
    for (let k = 1; k < MAX_SS; k++) if (s[sb + 16 + k] > s[sb + 16 + best]) best = k;
    ss[i] = best;
  }
  return { rot, trans, torsion, ss };
}

// A rigid operation is 12 numbers: R (3×3 row-major) then t (3).
function compose(X, xo, Y, yo, out, oo) {
  const r = new Float64Array(12);
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      r[a * 3 + b] = X[xo + a * 3] * Y[yo + b] + X[xo + a * 3 + 1] * Y[yo + 3 + b] + X[xo + a * 3 + 2] * Y[yo + 6 + b];
    }
    r[9 + a] = X[xo + a * 3] * Y[yo + 9] + X[xo + a * 3 + 1] * Y[yo + 10] + X[xo + a * 3 + 2] * Y[yo + 11] + X[xo + 9 + a];
  }
  out.set(r, oo);
}

/**
 * build_structure: all MAX_ATOM atom positions per residue (nm) from frames, torsions and SS class.
 * @param {Int32Array} residueType @param {Float64Array} pos Cα input (nm)
 */
export function buildAtoms(decoded, residueType, pos) {
  const n = residueType.length;
  const { rot, trans, torsion, ss } = decoded;
  const R = new Float64Array(n * MAX_ATOM * 3);
  const opr = new Float64Array(MAX_RIGID * 12);
  for (let i = 0; i < n; i++) {
    const rt = residueType[i];
    const tBase = (ss[i] * AMINO_ACIDS.length + rt) * MAX_RIGID * 12;
    opr.fill(0);
    for (let k = 0; k < 9; k++) opr[k] = rot[i * 9 + k];
    for (let c = 0; c < 3; c++) opr[9 + c] = trans[i * 3 + c] + pos[i * 3 + c];
    for (let g = 1; g < MAX_RIGID; g++) {
      const c = torsion[(i * MAX_TORSION + g - 1) * 2];
      const s = torsion[(i * MAX_TORSION + g - 1) * 2 + 1];
      const o = g * 12;
      opr[o] = 1;
      opr[o + 4] = c;
      opr[o + 5] = -s;
      opr[o + 7] = s;
      opr[o + 8] = c;
    }
    for (let g = 0; g < MAX_RIGID; g++) compose(RIGID_TRANSFORMS, tBase + g * 12, opr, g * 12, opr, g * 12);
    for (let g = 1; g < MAX_RIGID; g++) {
      const dep = RIGID_TRANSFORMS_DEP[rt * MAX_RIGID + g];
      compose(opr, dep * 12, opr, g * 12, opr, g * 12);
    }
    const gBase = (ss[i] * AMINO_ACIDS.length + rt) * MAX_ATOM * 3;
    for (let a = 0; a < MAX_ATOM; a++) {
      const o = RIGID_GROUPS_DEP[rt * MAX_ATOM + a] * 12;
      const x = RIGID_GROUPS[gBase + a * 3];
      const y = RIGID_GROUPS[gBase + a * 3 + 1];
      const z = RIGID_GROUPS[gBase + a * 3 + 2];
      const ro = (i * MAX_ATOM + a) * 3;
      for (let r = 0; r < 3; r++) {
        R[ro + r] = opr[o + r * 3] * x + opr[o + r * 3 + 1] * y + opr[o + r * 3 + 2] * z + opr[o + 9 + r];
      }
    }
  }
  return R;
}

// libter TER_PATCHes (residue_constants_base): bond length Å, angle and torsion offsets in degrees.
const NTER = { bLen: 1.04, bAng: 109.5, tAng: [0, 120, 240], append: ['HT1', 'HT2', 'HT3'], del: 'HN', define: ['N', 'CA', 'C'], ref: 'HN' };
const NTER_PRO = { bLen: 1.006, bAng: 109.5, tAng: [0, -120], append: ['HN1', 'HN2'], del: null, at: 1, define: ['N', 'CA', 'C'], ref: null };
const CTER = { bLen: 1.26, bAng: 118.0, tAng: [0, 180], append: ['OT1', 'OT2'], del: 'O', define: ['C', 'CA', 'N'], ref: 'O' };

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const unit = (v) => {
  const s = Math.hypot(...v);
  return [v[0] / s, v[1] / s, v[2] / s];
};

function torsion4(p0, p1, p2, p3) {
  const b1 = normSafe(sub(p1, p0), 0);
  const b2 = normSafe(sub(p2, p1), 0);
  const b3 = normSafe(sub(p3, p2), 0);
  let c1 = cross(b2, b3);
  if (!(Math.hypot(...c1) > EPS)) c1 = [1, 0, 0];
  return Math.atan2(dot(b1, c1), dot(c1, cross(b1, b2)));
}

/** numpy_basics.internal_to_cartesian: place X bonded to r0, angle X-r0-r1, torsion X-r0-r1-r2. */
export function internalToCartesian(r0, r1, r2, bLen, bAng, tAng) {
  const v1 = sub(r0, r1);
  const v2 = sub(r0, r2);
  const n1 = unit(cross(v1, v2));
  const n2 = unit(cross(v1, n1));
  const a = -Math.sin(tAng);
  const b = Math.cos(tAng);
  const v3 = unit([n1[0] * a + n2[0] * b, n1[1] * a + n2[1] * b, n1[2] * a + n2[2] * b]);
  const s = bLen * Math.sin(bAng);
  const c = bLen * Math.cos(bAng);
  const u1 = unit(v1);
  return [r0[0] + v3[0] * s - u1[0] * c, r0[1] + v3[1] * s - u1[1] * c, r0[2] + v3[2] * s - u1[2] * c];
}

function applyPatch(atoms, patch) {
  const byName = new Map(atoms.map((a) => [a.name, a]));
  const define = patch.define.map((nm) => byName.get(nm)?.xyz);
  if (define.some((p) => !p)) return atoms;
  const ref = patch.ref ? byName.get(patch.ref) : null;
  if (patch.ref && !ref) return atoms;
  const t0 = ref ? torsion4(ref.xyz, define[0], define[1], define[2]) : 0;
  const added = patch.append.map((name, i) => ({
    name,
    xyz: internalToCartesian(define[0], define[1], define[2], patch.bLen * 0.1, (patch.bAng * Math.PI) / 180, (patch.tAng[i] * Math.PI) / 180 + t0),
  }));
  const out = atoms.filter((a) => a.name !== patch.del);
  const at = patch.del ? atoms.findIndex((a) => a.name === patch.del) : patch.at;
  out.splice(at, 0, ...added);
  return out;
}

/**
 * Per-residue output atoms (CHARMM names, nm) in cg2all's topology order with N/C-terminal patches on the
 * first/last modelled residue of each chain. Residues of type UNK produce `null` (cg2all drops them).
 * @param {Array<{chainIndex:number}>} residues @param {Array<[number, number]>} ssbonds
 */
export function residueAtoms(R, residueType, residues, ssbonds) {
  const n = residueType.length;
  const disulfide = new Set(ssbonds.flat());
  const out = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    const rt = residueType[i];
    if (rt === UNK) continue;
    const name = AMINO_ACIDS[rt];
    const ref = RESIDUES[name];
    const atoms = [];
    ref.outputAtoms.forEach((atomName, k) => {
      if (atomName === 'HG1' && disulfide.has(i)) return;
      const a = ref.outputIndex[k];
      const o = (i * MAX_ATOM + a) * 3;
      atoms.push({ name: atomName, xyz: [R[o], R[o + 1], R[o + 2]] });
    });
    out[i] = atoms;
  }
  const chains = new Map();
  for (let i = 0; i < n; i++) {
    if (!out[i]) continue;
    const c = residues[i].chainIndex;
    if (!chains.has(c)) chains.set(c, []);
    chains.get(c).push(i);
  }
  for (const idx of chains.values()) {
    const first = idx[0];
    const last = idx[idx.length - 1];
    out[first] = applyPatch(out[first], AMINO_ACIDS[residueType[first]] === 'PRO' ? NTER_PRO : NTER);
    out[last] = applyPatch(out[last], CTER);
  }
  return out;
}

const pdbResidueName = (name) => (name === 'HSD' || name === 'HSE' ? 'HIS' : name);

function formatAtomName(name) {
  return name.length >= 4 ? name.slice(0, 4) : ` ${name.padEnd(3)}`;
}

function fmt(x, w, p) {
  return x.toFixed(p).padStart(w);
}

function reserial(line, serial) {
  return `${line.slice(0, 6)}${String(serial % 100000).padStart(5)}${line.slice(11)}`;
}

/**
 * Write the all-atom PDB.
 * @param {{ residues: any[], ssbonds: Array<[number, number]>, ligandLines: string[] }} trace
 * @param {Array<Array<{name:string, xyz:number[]}>|null>} atomsByResidue
 * @param {{ hydrogens?: boolean, ligands?: boolean, standardNames?: boolean, remarks?: string[] }} opts
 */
export function toPdb(trace, atomsByResidue, opts = {}) {
  const { hydrogens = false, ligands = true, standardNames = true, remarks = [] } = opts;
  const lines = remarks.map((r) => `REMARK   1 ${r}`.slice(0, 80));
  trace.ssbonds.forEach(([i, j], k) => {
    const a = trace.residues[i];
    const b = trace.residues[j];
    lines.push(
      `SSBOND ${String(k + 1).padStart(3)} CYS ${a.chainId || ' '} ${String(a.resNum).padStart(4)}${a.insCode || ' '}   CYS ${b.chainId || ' '} ${String(b.resNum).padStart(4)}${b.insCode || ' '}`,
    );
  });
  let serial = 1;
  let atomCount = 0;
  let prevChain = null;
  let lastRes = null;
  const ter = () => {
    if (!lastRes) return;
    lines.push(`TER   ${String(serial % 100000).padStart(5)}      ${lastRes.name} ${lastRes.chain}${String(lastRes.num).padStart(4)}${lastRes.ins}`);
    serial++;
    lastRes = null;
  };
  trace.residues.forEach((res, i) => {
    if (prevChain !== null && res.chainIndex !== prevChain) ter();
    prevChain = res.chainIndex;
    const chain = res.chainId || ' ';
    const ins = res.insCode || ' ';
    const atoms = atomsByResidue[i];
    if (!atoms) {
      // Non-standard residue (UNK to cg2all): keep its input records verbatim, renumbered.
      for (const l of res.lines) {
        lines.push(reserial(l, serial++));
        atomCount++;
      }
      if (res.lines.length) lastRes = { name: res.inputName.padStart(3), chain, num: res.resNum, ins };
      return;
    }
    const resName = pdbResidueName(res.resName);
    const table = standardNames ? (PDB_ATOM_NAMES[resName] ?? {}) : {};
    for (const atom of atoms) {
      if (!hydrogens && atom.name[0] === 'H') continue;
      const name = table[atom.name] ?? atom.name;
      lines.push(
        `ATOM  ${String(serial % 100000).padStart(5)} ${formatAtomName(name)} ${resName.padStart(3)} ${chain}${String(res.resNum).padStart(4)}${ins}   ` +
          `${fmt(atom.xyz[0] * 10, 8, 3)}${fmt(atom.xyz[1] * 10, 8, 3)}${fmt(atom.xyz[2] * 10, 8, 3)}${fmt(1, 6, 2)}${fmt(res.bfactor, 6, 2)}          ${name[0].padStart(2)}`,
      );
      serial++;
      atomCount++;
    }
    lastRes = { name: resName.padStart(3), chain, num: res.resNum, ins };
  });
  ter();
  if (ligands) {
    for (const l of trace.ligandLines) {
      lines.push(reserial(l, serial++));
      atomCount++;
    }
  }
  lines.push('END');
  return { pdbText: `${lines.join('\n')}\n`, atomCount };
}
