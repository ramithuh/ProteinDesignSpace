import type {
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
  WorkflowEdge,
  WorkflowGraph,
  WorkflowNode,
} from '$lib/engine/types';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { runPdbInputCpu } from '$lib/plugins/pdb-input';

export const AA1_TO_AA3: Record<string, string> = {
  A: 'ALA',
  C: 'CYS',
  D: 'ASP',
  E: 'GLU',
  F: 'PHE',
  G: 'GLY',
  H: 'HIS',
  I: 'ILE',
  K: 'LYS',
  L: 'LEU',
  M: 'MET',
  N: 'ASN',
  P: 'PRO',
  Q: 'GLN',
  R: 'ARG',
  S: 'SER',
  T: 'THR',
  V: 'VAL',
  W: 'TRP',
  Y: 'TYR',
};

export const ALL_20_LOCALFOLD_MODELS = [
  'monomer',
  'monomer-2',
  'monomer-3',
  'monomer-4',
  'monomer-5',
  'multimer',
  'multimer-2',
  'multimer-3',
  'multimer-4',
  'multimer-5',
  'rosettafold3',
  'boltz2',
  'openbind0',
  'protenix2',
  'opendde',
  'mol1',
  'intellifold2',
  'af3',
  'ef2-fast-300m',
  'ef2-fast-600m',
] as const;

export const TEMPLATE_CAPABLE_AF2_MONO_MODELS = ['monomer', 'monomer-2'] as const;
export const TEMPLATE_CAPABLE_AF2_MULTI_MODELS = [
  'multimer',
  'multimer-2',
  'multimer-3',
  'multimer-4',
  'multimer-5',
] as const;
export const TEMPLATE_CAPABLE_AF3_MODELS = [
  'rosettafold3',
  'boltz2',
  'openbind0',
  'protenix2',
  'opendde',
  'mol1',
  'intellifold2',
  'af3',
] as const;
export const TEMPLATE_FREE_MODELS = [
  'monomer-3',
  'monomer-4',
  'monomer-5',
  'ef2-fast-300m',
  'ef2-fast-600m',
] as const;

export const STANDARD_MONOMER_SEQUENCE = 'MKWVTFISLLFLFSSAYSRG';
export const HOMOLOG_MONOMER_SEQUENCE = 'GSHMKWVTFISAAFLFSSYSRG'; // 3-aa N-tag + 2 mutations + 1 deletion
export const UNRELATED_SEQUENCE = 'WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW';

export const HETERODIMER_CHAIN_A = 'MKWVTFISLLFLFS';
export const HETERODIMER_CHAIN_B = 'SAYSRGVFRRDTHK';
export const HETERODIMER_SEQUENCE = `${HETERODIMER_CHAIN_A}:${HETERODIMER_CHAIN_B}`;

function fmtAtomLine(opts: {
  record?: 'ATOM' | 'HETATM';
  serial: number;
  atomName: string;
  altLoc?: string;
  resName: string;
  chainId: string;
  resSeq: number;
  insCode?: string;
  x: number;
  y: number;
  z: number;
  occ?: number;
  bfac?: number;
  element: string;
}): string {
  const rec = (opts.record ?? 'ATOM').padEnd(6, ' ');
  const ser = String(opts.serial).padStart(5, ' ');
  const name =
    opts.atomName.length >= 4
      ? opts.atomName.slice(0, 4)
      : ` ${opts.atomName}`.padEnd(4, ' ');
  const alt = (opts.altLoc ?? ' ').slice(0, 1);
  const res = opts.resName.padStart(3, ' ');
  const ch = (opts.chainId || 'A').slice(0, 1);
  const rseq = String(opts.resSeq).padStart(4, ' ');
  const ins = (opts.insCode ?? ' ').slice(0, 1);
  const x = opts.x.toFixed(3).padStart(8, ' ');
  const y = opts.y.toFixed(3).padStart(8, ' ');
  const z = opts.z.toFixed(3).padStart(8, ' ');
  const occ = (opts.occ ?? 1.0).toFixed(2).padStart(6, ' ');
  const bfac = (opts.bfac ?? 85.0).toFixed(2).padStart(6, ' ');
  const elem = opts.element.padStart(2, ' ');
  return `${rec}${ser} ${name}${alt}${res} ${ch}${rseq}${ins}   ${x}${y}${z}${occ}${bfac}          ${elem}`;
}

interface SyntheticResidueOpts {
  chainId: string;
  resSeq: number;
  resName: string;
  insCode?: string;
  origin: [number, number, number];
  bfac?: number;
  record?: 'ATOM' | 'HETATM';
  altLoc?: string;
  occ?: number;
  includeSidechain?: boolean;
}

function residueAtomCoords(
  origin: [number, number, number],
  resName: string,
  includeSidechain = true,
): Array<{ name: string; element: string; xyz: [number, number, number] }> {
  const [ox, oy, oz] = origin;
  // Non-collinear N-CA-C geometry (~1.46 A N-CA, ~1.52 A CA-C, ~111 deg angle)
  const atoms: Array<{ name: string; element: string; xyz: [number, number, number] }> = [
    { name: 'N', element: 'N', xyz: [ox - 1.25, oy - 0.75, oz] },
    { name: 'CA', element: 'C', xyz: [ox, oy, oz] },
    { name: 'C', element: 'C', xyz: [ox + 1.35, oy - 0.7, oz] },
    { name: 'O', element: 'O', xyz: [ox + 1.45, oy - 1.92, oz + 0.1] },
  ];
  if (resName !== 'GLY') {
    atoms.push({ name: 'CB', element: 'C', xyz: [ox - 0.1, oy + 0.85, oz + 1.25] });
  }
  if (includeSidechain) {
    if (resName === 'ARG') {
      atoms.push(
        { name: 'CG', element: 'C', xyz: [ox - 0.15, oy + 1.65, oz + 2.55] },
        { name: 'CD', element: 'C', xyz: [ox - 0.2, oy + 2.95, oz + 3.35] },
        { name: 'NE', element: 'N', xyz: [ox - 0.25, oy + 3.95, oz + 4.35] },
        { name: 'CZ', element: 'C', xyz: [ox - 0.3, oy + 5.15, oz + 4.85] },
        { name: 'NH1', element: 'N', xyz: [ox - 0.35, oy + 6.15, oz + 4.15] },
        { name: 'NH2', element: 'N', xyz: [ox - 0.3, oy + 5.35, oz + 6.15] },
      );
    } else if (resName === 'MSE') {
      atoms.push(
        { name: 'CG', element: 'C', xyz: [ox - 0.1, oy + 1.65, oz + 2.55] },
        { name: 'SE', element: 'SE', xyz: [ox - 0.1, oy + 2.95, oz + 3.65] },
        { name: 'CE', element: 'C', xyz: [ox + 0.9, oy + 3.85, oz + 4.45] },
      );
    } else if (resName === 'SEP') {
      atoms.push(
        { name: 'OG', element: 'O', xyz: [ox - 0.1, oy + 1.75, oz + 2.35] },
        { name: 'P', element: 'P', xyz: [ox + 0.5, oy + 2.85, oz + 3.25] },
        { name: 'O1P', element: 'O', xyz: [ox + 1.8, oy + 3.15, oz + 2.85] },
        { name: 'O2P', element: 'O', xyz: [ox - 0.3, oy + 4.05, oz + 3.15] },
        { name: 'O3P', element: 'O', xyz: [ox + 0.5, oy + 2.45, oz + 4.65] },
      );
    } else if (resName === 'SER') {
      atoms.push({ name: 'OG', element: 'O', xyz: [ox - 0.1, oy + 1.75, oz + 2.35] });
    } else if (resName === 'MET') {
      atoms.push(
        { name: 'CG', element: 'C', xyz: [ox - 0.1, oy + 1.65, oz + 2.55] },
        { name: 'SD', element: 'S', xyz: [ox - 0.1, oy + 2.95, oz + 3.65] },
        { name: 'CE', element: 'C', xyz: [ox + 0.9, oy + 3.85, oz + 4.45] },
      );
    }
  }
  return atoms;
}

export function buildSyntheticChainPdbLines(
  sequence: string,
  opts: {
    chainId?: string;
    startSerial?: number;
    startResSeq?: number;
    originOffset?: [number, number, number];
    bfacForIdx?: (idx: number) => number;
    includeSidechains?: boolean;
  } = {},
): { lines: string[]; nextSerial: number } {
  const chainId = opts.chainId ?? 'A';
  let serial = opts.startSerial ?? 1;
  const startRes = opts.startResSeq ?? 1;
  const [dx, dy, dz] = opts.originOffset ?? [0, 0, 0];
  const lines: string[] = [];

  for (let i = 0; i < sequence.length; i++) {
    const aa1 = sequence[i];
    const resName = AA1_TO_AA3[aa1] ?? 'ALA';
    // Helical-like path with ~3.8 A C-alpha spacing
    const angle = i * 1.74;
    const ox = dx + i * 3.8;
    const oy = dy + Math.sin(angle) * 2.3;
    const oz = dz + Math.cos(angle) * 2.3;
    const bfac = opts.bfacForIdx ? opts.bfacForIdx(i) : 88.0;
    const atoms = residueAtomCoords([ox, oy, oz], resName, opts.includeSidechains ?? true);
    for (const atom of atoms) {
      lines.push(
        fmtAtomLine({
          serial: serial++,
          atomName: atom.name,
          resName,
          chainId,
          resSeq: startRes + i,
          x: atom.xyz[0],
          y: atom.xyz[1],
          z: atom.xyz[2],
          bfac,
          element: atom.element,
        }),
      );
    }
  }
  lines.push(`TER   ${String(serial++).padStart(5, ' ')}      ${AA1_TO_AA3[sequence.at(-1) ?? 'A'] ?? 'ALA'} ${chainId}${String(startRes + sequence.length - 1).padStart(4, ' ')}`);
  return { lines, nextSerial: serial };
}

export const STANDARD_MONOMER_PDB = (() => {
  const { lines } = buildSyntheticChainPdbLines(STANDARD_MONOMER_SEQUENCE, { chainId: 'A' });
  return [...lines, 'END'].join('\n') + '\n';
})();

export const HETERODIMER_PDB = (() => {
  const chA = buildSyntheticChainPdbLines(HETERODIMER_CHAIN_A, {
    chainId: 'A',
    originOffset: [0, 0, 0],
  });
  const chB = buildSyntheticChainPdbLines(HETERODIMER_CHAIN_B, {
    chainId: 'B',
    startSerial: chA.nextSerial,
    originOffset: [4, 10, 8], // ~13 A away in same co-crystal coordinate frame
  });
  return [...chA.lines, ...chB.lines, 'END'].join('\n') + '\n';
})();

export const NMR_MULTIMODEL_PDB = (() => {
  const m1 = buildSyntheticChainPdbLines(STANDARD_MONOMER_SEQUENCE, {
    chainId: 'A',
    originOffset: [0, 0, 0],
  });
  const m2 = buildSyntheticChainPdbLines(STANDARD_MONOMER_SEQUENCE + 'AAAA', {
    chainId: 'A',
    originOffset: [500, 500, 500],
  });
  return ['MODEL        1', ...m1.lines, 'ENDMDL', 'MODEL        2', ...m2.lines, 'ENDMDL', 'END'].join('\n') + '\n';
})();

export const ALTLOC_PDB = (() => {
  const lines: string[] = [];
  let serial = 1;
  // Residue 1: ALA with altLoc A (occ 0.30, shifted +12 A) and altLoc B (occ 0.70, normal)
  const r1A = residueAtomCoords([12, 12, 12], 'ALA');
  const r1B = residueAtomCoords([0, 0, 0], 'ALA');
  for (const a of r1A) {
    lines.push(
      fmtAtomLine({
        serial: serial++,
        atomName: a.name,
        altLoc: 'A',
        resName: 'ALA',
        chainId: 'A',
        resSeq: 1,
        x: a.xyz[0],
        y: a.xyz[1],
        z: a.xyz[2],
        occ: 0.3,
        bfac: 82.0,
        element: a.element,
      }),
    );
  }
  for (const a of r1B) {
    lines.push(
      fmtAtomLine({
        serial: serial++,
        atomName: a.name,
        altLoc: 'B',
        resName: 'ALA',
        chainId: 'A',
        resSeq: 1,
        x: a.xyz[0],
        y: a.xyz[1],
        z: a.xyz[2],
        occ: 0.7,
        bfac: 82.0,
        element: a.element,
      }),
    );
  }
  // Residue 2: VAL with tie occupancy (0.50 / 0.50) between altLoc B (listed first) and altLoc A
  const r2B = residueAtomCoords([3.8, 0.5, 0.5], 'VAL');
  const r2A = residueAtomCoords([15.8, 0.5, 0.5], 'VAL');
  for (const a of r2B) {
    lines.push(
      fmtAtomLine({
        serial: serial++,
        atomName: a.name,
        altLoc: 'B',
        resName: 'VAL',
        chainId: 'A',
        resSeq: 2,
        x: a.xyz[0],
        y: a.xyz[1],
        z: a.xyz[2],
        occ: 0.5,
        bfac: 85.0,
        element: a.element,
      }),
    );
  }
  for (const a of r2A) {
    lines.push(
      fmtAtomLine({
        serial: serial++,
        atomName: a.name,
        altLoc: 'A',
        resName: 'VAL',
        chainId: 'A',
        resSeq: 2,
        x: a.xyz[0],
        y: a.xyz[1],
        z: a.xyz[2],
        occ: 0.5,
        bfac: 85.0,
        element: a.element,
      }),
    );
  }
  const rest = buildSyntheticChainPdbLines('DEFGHIKL', {
    chainId: 'A',
    startSerial: serial,
    startResSeq: 3,
    originOffset: [7.6, 0, 0],
  });
  return [...lines, ...rest.lines, 'END'].join('\n') + '\n';
})();

export const MODIFIED_MSE_SEP_PDB = (() => {
  const lines: string[] = [];
  let serial = 1;
  // Res 1: HETATM MSE (selenomethionine -> parent M, SE -> SD)
  for (const a of residueAtomCoords([0, 0, 0], 'MSE', true)) {
    lines.push(
      fmtAtomLine({
        record: 'HETATM',
        serial: serial++,
        atomName: a.name,
        resName: 'MSE',
        chainId: 'A',
        resSeq: 1,
        x: a.xyz[0],
        y: a.xyz[1],
        z: a.xyz[2],
        bfac: 90,
        element: a.element,
      }),
    );
  }
  // Res 2-4: standard K W V
  const mid = buildSyntheticChainPdbLines('KWV', {
    chainId: 'A',
    startSerial: serial,
    startResSeq: 2,
    originOffset: [3.8, 0, 0],
  });
  lines.push(...mid.lines.slice(0, -1)); // drop TER
  serial = mid.nextSerial - 1;
  // Res 5: HETATM SEP (phosphoserine -> parent S)
  for (const a of residueAtomCoords([15.2, 0.8, 0.4], 'SEP', true)) {
    lines.push(
      fmtAtomLine({
        record: 'HETATM',
        serial: serial++,
        atomName: a.name,
        resName: 'SEP',
        chainId: 'A',
        resSeq: 5,
        x: a.xyz[0],
        y: a.xyz[1],
        z: a.xyz[2],
        bfac: 89,
        element: a.element,
      }),
    );
  }
  const tail = buildSyntheticChainPdbLines('TFISLLFL', {
    chainId: 'A',
    startSerial: serial,
    startResSeq: 6,
    originOffset: [19.0, 0, 0],
  });
  return [...lines, ...tail.lines, 'END'].join('\n') + '\n';
})();

export const ARG_FLIPPED_PDB = (() => {
  const lines: string[] = [];
  let serial = 1;
  const pre = buildSyntheticChainPdbLines('AAA', { chainId: 'A', startSerial: serial, startResSeq: 1 });
  lines.push(...pre.lines.slice(0, -1));
  serial = pre.nextSerial - 1;
  // ARG at res 4 where NH2 is closer to CD (d=1.3A) than NH1 (d=3.6A)
  const ox = 11.4;
  const argAtoms: Array<{ name: string; element: string; xyz: [number, number, number] }> = [
    { name: 'N', element: 'N', xyz: [ox - 1.25, -0.75, 0] },
    { name: 'CA', element: 'C', xyz: [ox, 0, 0] },
    { name: 'C', element: 'C', xyz: [ox + 1.35, -0.7, 0] },
    { name: 'O', element: 'O', xyz: [ox + 1.45, -1.92, 0.1] },
    { name: 'CB', element: 'C', xyz: [ox - 0.1, 0.85, 1.25] },
    { name: 'CG', element: 'C', xyz: [ox - 0.15, 1.65, 2.55] },
    { name: 'CD', element: 'C', xyz: [ox - 0.2, 2.95, 3.35] },
    { name: 'NE', element: 'N', xyz: [ox - 0.25, 3.95, 4.35] },
    { name: 'CZ', element: 'C', xyz: [ox - 0.3, 5.15, 4.85] },
    { name: 'NH1', element: 'N', xyz: [ox - 0.3, 6.2, 5.8] }, // farther from CD
    { name: 'NH2', element: 'N', xyz: [ox - 0.2, 4.1, 3.8] }, // closer to CD
  ];
  for (const a of argAtoms) {
    lines.push(
      fmtAtomLine({
        serial: serial++,
        atomName: a.name,
        resName: 'ARG',
        chainId: 'A',
        resSeq: 4,
        x: a.xyz[0],
        y: a.xyz[1],
        z: a.xyz[2],
        element: a.element,
      }),
    );
  }
  const post = buildSyntheticChainPdbLines('AAAA', {
    chainId: 'A',
    startSerial: serial,
    startResSeq: 5,
    originOffset: [15.2, 0, 0],
  });
  return [...lines, ...post.lines, 'END'].join('\n') + '\n';
})();

export const INS_CODE_NEGATIVE_RESNUM_PDB = (() => {
  const seq = 'MKWVTFISLL';
  const resEntries: Array<{ resSeq: number; insCode: string }> = [
    { resSeq: -3, insCode: ' ' },
    { resSeq: -2, insCode: ' ' },
    { resSeq: -1, insCode: ' ' },
    { resSeq: 0, insCode: ' ' },
    { resSeq: 1, insCode: ' ' },
    { resSeq: 100, insCode: ' ' },
    { resSeq: 100, insCode: 'A' },
    { resSeq: 100, insCode: 'B' },
    { resSeq: 100, insCode: 'C' },
    { resSeq: 101, insCode: ' ' },
  ];
  const lines: string[] = [];
  let serial = 1;
  for (let i = 0; i < seq.length; i++) {
    const resName = AA1_TO_AA3[seq[i]];
    const origin: [number, number, number] = [i * 3.8, Math.sin(i) * 1.5, Math.cos(i) * 1.5];
    for (const a of residueAtomCoords(origin, resName, false)) {
      lines.push(
        fmtAtomLine({
          serial: serial++,
          atomName: a.name,
          resName,
          chainId: 'A',
          resSeq: resEntries[i].resSeq,
          insCode: resEntries[i].insCode,
          x: a.xyz[0],
          y: a.xyz[1],
          z: a.xyz[2],
          element: a.element,
        }),
      );
    }
  }
  return [...lines, 'END'].join('\n') + '\n';
})();

export function buildMmcifFromChains(
  chains: Array<{
    authAsymId: string;
    labelAsymId?: string;
    sequence: string;
    originOffset?: [number, number, number];
    modifiedAt?: Record<number, 'MSE' | 'SEP'>;
  }>,
  extraLines: string[] = [],
): string {
  const header = [
    'data_SYNTHETIC_TEMPLATE',
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
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.auth_seq_id',
    '_atom_site.auth_comp_id',
    '_atom_site.auth_asym_id',
    '_atom_site.auth_atom_id',
    '_atom_site.pdbx_PDB_model_num',
  ];
  const rows: string[] = [];
  let serial = 1;
  chains.forEach((chain, chIdx) => {
    const labelAsym = chain.labelAsymId ?? chain.authAsymId;
    const [dx, dy, dz] = chain.originOffset ?? [0, chIdx * 12, chIdx * 6];
    for (let i = 0; i < chain.sequence.length; i++) {
      const mod = chain.modifiedAt?.[i];
      const resName = mod ?? AA1_TO_AA3[chain.sequence[i]] ?? 'ALA';
      const group = mod ? 'HETATM' : 'ATOM';
      const angle = i * 1.74;
      const origin: [number, number, number] = [
        dx + i * 3.8,
        dy + Math.sin(angle) * 2.2,
        dz + Math.cos(angle) * 2.2,
      ];
      for (const a of residueAtomCoords(origin, resName, true)) {
        rows.push(
          [
            group,
            serial++,
            a.element,
            a.name,
            '.',
            resName,
            labelAsym,
            chIdx + 1,
            i + 1,
            '?',
            a.xyz[0].toFixed(3),
            a.xyz[1].toFixed(3),
            a.xyz[2].toFixed(3),
            '1.00',
            '88.50',
            i + 101, // intentional offset between auth_seq_id and label_seq_id
            resName,
            chain.authAsymId,
            a.name,
            1,
          ].join(' '),
        );
      }
    }
  });
  return [...header, ...rows, ...extraLines, '#'].join('\n') + '\n';
}

export const STANDARD_MONOMER_MMCIF = buildMmcifFromChains([
  { authAsymId: 'A', labelAsymId: 'A', sequence: STANDARD_MONOMER_SEQUENCE },
]);

export const MULTICHAR_COMPLEX_MMCIF = buildMmcifFromChains([
  { authAsymId: 'AA', labelAsymId: 'A', sequence: HETERODIMER_CHAIN_A, originOffset: [0, 0, 0] },
  { authAsymId: 'AB', labelAsymId: 'B', sequence: HETERODIMER_CHAIN_B, originOffset: [4, 10, 8] },
]);

export const MODIFIED_MSE_SEP_MMCIF = buildMmcifFromChains([
  {
    authAsymId: 'AA',
    labelAsymId: 'A',
    sequence: 'MKWVSTFISLLFL',
    modifiedAt: { 0: 'MSE', 4: 'SEP' },
  },
]);

export const HOLO_COMPLEX_MMCIF = (() => {
  const base = buildMmcifFromChains([
    { authAsymId: 'A', labelAsymId: 'A', sequence: 'MKWVSTFISLLFL', modifiedAt: { 4: 'SEP' } },
  ]).replace('#\n', '');
  const extraRows = [
    // DNA chain D
    'ATOM 9001 P P . DA D 2 1 ? 25.000 10.000 5.000 1.00 70.00 1 DA D P 1',
    'ATOM 9002 C4* C4* . DA D 2 1 ? 26.200 10.500 5.400 1.00 70.00 1 DA D C4* 1',
    // Ligand ZN
    'HETATM 9010 ZN ZN . ZN Z 3 . ? 8.000 2.000 2.000 1.00 90.00 201 ZN Z ZN 1',
    // Water HOH
    'HETATM 9011 O O . HOH W 4 . ? 30.000 30.000 30.000 1.00 50.00 301 HOH W O 1',
    '#',
  ];
  return `${base}${extraRows.join('\n')}\n`;
})();

export function buildManyChainMmcif(chainCount = 30, chainLen = 5): string {
  const chains = Array.from({ length: chainCount }, (_, idx) => ({
    authAsymId: `C${idx + 1}`,
    labelAsymId: `L${idx + 1}`,
    sequence: 'MKWVT'.slice(0, chainLen),
    originOffset: [0, idx * 8, 0] as [number, number, number],
  }));
  return buildMmcifFromChains(chains);
}

export const EMPTY_TEXT = '';
export const WHITESPACE_TEXT = '   \n\t \r\n';
export const CORRUPT_TEXT = 'HEADER CORRUPTED\nFOO BAR BAZ\nNOT_ATOM 1 2 3\n';

export const WATER_ONLY_PDB = [
  fmtAtomLine({ record: 'HETATM', serial: 1, atomName: 'O', resName: 'HOH', chainId: 'A', resSeq: 1, x: 1, y: 2, z: 3, element: 'O' }),
  fmtAtomLine({ record: 'HETATM', serial: 2, atomName: 'O', resName: 'HOH', chainId: 'A', resSeq: 2, x: 4, y: 5, z: 6, element: 'O' }),
  'END',
].join('\n') + '\n';

export const LIGAND_ONLY_PDB = [
  fmtAtomLine({ record: 'HETATM', serial: 1, atomName: 'ZN', resName: 'ZN', chainId: 'Z', resSeq: 1, x: 1, y: 2, z: 3, element: 'ZN' }),
  fmtAtomLine({ record: 'HETATM', serial: 2, atomName: 'PG', resName: 'ATP', chainId: 'Z', resSeq: 2, x: 4, y: 5, z: 6, element: 'P' }),
  'END',
].join('\n') + '\n';

export const DNA_ONLY_PDB = [
  fmtAtomLine({ serial: 1, atomName: 'P', resName: 'DA', chainId: 'D', resSeq: 1, x: 1, y: 2, z: 3, element: 'P' }),
  fmtAtomLine({ serial: 2, atomName: 'C1*', resName: 'DT', chainId: 'D', resSeq: 2, x: 4, y: 5, z: 6, element: 'C' }),
  'END',
].join('\n') + '\n';

export const SHORT_4AA_PDB = buildSyntheticChainPdbLines('MKWV', { chainId: 'A' }).lines.join('\n') + '\nEND\n';
export const SHORT_2AA_PDB = buildSyntheticChainPdbLines('MK', { chainId: 'A' }).lines.join('\n') + '\nEND\n';
export const SINGLE_RESIDUE_PDB = buildSyntheticChainPdbLines('M', { chainId: 'A' }).lines.join('\n') + '\nEND\n';

export const COINCIDENT_CLASH_PDB = (() => {
  const seq = 'MKWV';
  const lines: string[] = [];
  let serial = 1;
  for (let i = 0; i < seq.length; i++) {
    const resName = AA1_TO_AA3[seq[i]];
    for (const [atomName, element] of [
      ['N', 'N'],
      ['CA', 'C'],
      ['C', 'C'],
      ['O', 'O'],
      ['CB', 'C'],
    ] as const) {
      lines.push(
        fmtAtomLine({
          serial: serial++,
          atomName,
          resName,
          chainId: 'A',
          resSeq: i + 1,
          x: 0,
          y: 0,
          z: 0,
          element,
        }),
      );
    }
  }
  return [...lines, 'END'].join('\n') + '\n';
})();

export const LARGE_TEMPLATE_PDB = (() => {
  // > 2500 chars (>1500 share threshold)
  const { lines } = buildSyntheticChainPdbLines('MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE', { chainId: 'A' });
  return [...lines, 'END'].join('\n') + '\n';
})();

export function makeSequenceSetArtifact(
  sequence: string,
  opts: {
    id?: string;
    chainLengths?: number[];
    chainIds?: string[];
    sourceStructureId?: string;
    lockedChains?: string[];
    chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>;
    extraCandidates?: SequenceCandidate[];
  } = {},
): SequenceSet {
  const id = opts.id ?? 'seqset-test-1';
  const candidate: SequenceCandidate = {
    id: 'cand-1',
    key: 'cand-1',
    sequence,
    score: -1.5,
    globalScore: -1.5,
    sourceModelId: 'sequence-input',
    sourceStructureId: opts.sourceStructureId,
    ...(opts.chainLengths ? { chainLengths: opts.chainLengths } : {}),
    ...(opts.chainIds ? { chainIds: opts.chainIds } : {}),
  };
  return {
    kind: 'SequenceSet',
    id,
    parentStructureId: opts.sourceStructureId ?? 'none',
    sourceStructureId: opts.sourceStructureId ?? 'none',
    sourceModelId: 'sequence-input',
    ...(opts.lockedChains ? { lockedChains: opts.lockedChains } : {}),
    ...(opts.chainRoles ? { chainRoles: opts.chainRoles } : {}),
    candidates: [candidate, ...(opts.extraCandidates ?? [])],
  };
}

export async function makeStructureArtifactFromText(
  text: string,
  opts: { id?: string; filename?: string; cropSegments?: string } = {},
): Promise<StructureArtifact> {
  const node: WorkflowNode = {
    id: opts.id ?? 'pdb_tpl',
    adapterId: 'pdb-input',
    label: 'Template Structure',
    params: {
      mode: 'custom',
      pdbText: text,
      filename: opts.filename ?? (text.includes('_atom_site.') ? 'template.cif' : 'template.pdb'),
      cropSegments: opts.cropSegments ?? '',
    },
  };
  const out = await runPdbInputCpu(node);
  return out.structure as StructureArtifact;
}

export function makeSequenceNode(
  id: string,
  sequence: string,
  label = 'Sequence Input',
): WorkflowNode {
  return {
    id,
    adapterId: 'sequence-input',
    label,
    params: { sequence, label: label.toLowerCase().replace(/\s+/g, '-') },
  };
}

export function makePdbNode(
  id: string,
  pdbText: string,
  opts: { filename?: string; cropSegments?: string; label?: string } = {},
): WorkflowNode {
  return {
    id,
    adapterId: 'pdb-input',
    label: opts.label ?? 'Template PDB',
    params: {
      mode: 'custom',
      pdbText,
      filename: opts.filename ?? (pdbText.includes('_atom_site.') ? 'template.cif' : 'template.pdb'),
      cropSegments: opts.cropSegments ?? '',
    },
  };
}

export function makeLocalFoldNode(
  id: string,
  overrides: Record<string, unknown> = {},
  label = 'LocalFold',
): WorkflowNode {
  return {
    id,
    adapterId: 'localfold-af2',
    label,
    params: {
      ...defaultParamsForAdapter('localfold-af2'),
      ...overrides,
    },
  };
}

export function makeScorerNode(id: string): WorkflowNode {
  return {
    id,
    adapterId: 'designability-scorer',
    label: 'Designability Scorer',
    params: defaultParamsForAdapter('designability-scorer'),
  };
}

export function makeTemplatedLocalFoldGraph(opts: {
  sequence?: string;
  templateText?: string;
  supplyMode?: 'port' | 'inline' | 'both';
  model?: string;
  localFoldOverrides?: Record<string, unknown>;
  includeScorer?: boolean;
  cropSegments?: string;
}): WorkflowGraph {
  const seq = opts.sequence ?? STANDARD_MONOMER_SEQUENCE;
  const tplText = opts.templateText ?? STANDARD_MONOMER_PDB;
  const mode = opts.supplyMode ?? 'port';
  const model = opts.model ?? 'monomer-2';

  const seqNode = makeSequenceNode('seq', seq);
  const foldParams: Record<string, unknown> = {
    model,
    ...opts.localFoldOverrides,
  };
  if (mode === 'inline' || mode === 'both') {
    foldParams.templatePdbText = tplText;
    foldParams.templateFilename = tplText.includes('_atom_site.') ? 'inline.cif' : 'inline.pdb';
  }
  const foldNode = makeLocalFoldNode('fold', foldParams);
  const nodes: WorkflowNode[] = [seqNode, foldNode];
  const edges: WorkflowEdge[] = [
    { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
  ];

  if (mode === 'port' || mode === 'both') {
    const pdbNode = makePdbNode('tpl', tplText, { cropSegments: opts.cropSegments });
    nodes.unshift(pdbNode);
    edges.push({ fromNode: 'tpl', fromPort: 'structure', toNode: 'fold', toPort: 'template' });
  }

  if (opts.includeScorer) {
    const scorer = makeScorerNode('score');
    nodes.push(scorer);
    edges.push({ fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' });
    if (mode === 'port' || mode === 'both') {
      edges.push({ fromNode: 'tpl', fromPort: 'structure', toNode: 'score', toPort: 'structure' });
    } else {
      edges.push({ fromNode: 'fold', fromPort: 'structure', toNode: 'score', toPort: 'structure' });
    }
  }

  return {
    id: `test-templated-${model}-${mode}`,
    name: `Templated ${model} (${mode})`,
    nodes,
    edges,
  };
}

export function makeUntemplatedLocalFoldGraph(opts: {
  sequence?: string;
  model?: string;
  localFoldOverrides?: Record<string, unknown>;
} = {}): WorkflowGraph {
  const seqNode = makeSequenceNode('seq', opts.sequence ?? STANDARD_MONOMER_SEQUENCE);
  const foldNode = makeLocalFoldNode('fold', {
    model: opts.model ?? 'monomer',
    ...opts.localFoldOverrides,
  });
  return {
    id: `test-untemplated-${opts.model ?? 'monomer'}`,
    name: 'Untemplated LocalFold',
    nodes: [seqNode, foldNode],
    edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
  };
}
