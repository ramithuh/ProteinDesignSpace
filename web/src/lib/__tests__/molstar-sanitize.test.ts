import { describe, expect, it } from 'vitest';
import { parsePDB } from 'molstar/lib/mol-io/reader/pdb/parser';
import { trajectoryFromPDB } from 'molstar/lib/mol-model-formats/structure/pdb';
import { Structure, QueryContext, StructureSelection } from 'molstar/lib/mol-model/structure';
import { StructureSelectionQueries } from 'molstar/lib/mol-plugin-state/helpers/structure-selection-query';
import { pruneFalseNucleicMonomers } from '$lib/viewer/molstar-sanitize';

function pdbAtomLine(
  rec: 'ATOM  ' | 'HETATM',
  serial: number,
  name: string,
  resName: string,
  chain: string,
  resSeq: number,
  x: number,
  y: number,
  z: number,
  elem: string,
): string {
  const paddedName = name.length < 4 ? ` ${name}`.padEnd(4) : name.slice(0, 4);
  return (
    rec +
    String(serial).padStart(5) +
    ' ' +
    paddedName +
    ' ' +
    resName.padStart(3).slice(0, 3) +
    ' ' +
    chain +
    String(resSeq).padStart(4) +
    '    ' +
    x.toFixed(3).padStart(8) +
    y.toFixed(3).padStart(8) +
    z.toFixed(3).padStart(8) +
    '  1.00 85.00          ' +
    elem.padStart(2)
  );
}

function makeProteinChain(chain: string, startSerial = 1): { lines: string[]; nextSerial: number } {
  const lines: string[] = [];
  let s = startSerial;
  for (let r = 1; r <= 4; r += 1) {
    const x0 = (r - 1) * 3.8;
    lines.push(pdbAtomLine('ATOM  ', s++, 'N', 'ALA', chain, r, x0, 0, 0, 'N'));
    lines.push(pdbAtomLine('ATOM  ', s++, 'CA', 'ALA', chain, r, x0 + 1.45, 0.2, 0, 'C'));
    lines.push(pdbAtomLine('ATOM  ', s++, 'C', 'ALA', chain, r, x0 + 2.45, -0.8, 0, 'C'));
    lines.push(pdbAtomLine('ATOM  ', s++, 'O', 'ALA', chain, r, x0 + 2.35, -2.0, 0, 'O'));
  }
  lines.push('TER');
  return { lines, nextSerial: s };
}

function makeDhzLigand(
  chain: string,
  resSeq: number,
  startSerial: number,
  offsetX = 10,
): { lines: string[]; nextSerial: number } {
  let s = startSerial;
  const atoms: Array<[string, number, number, number, string]> = [
    ["C4'", 0.0, 0.0, 0.0, 'C'],
    ["C3'", 1.5, 0.0, 0.0, 'C'],
    ["O3'", 2.0, -1.2, 0.0, 'O'],
    ["C2'", 1.9, 1.2, 0.5, 'C'],
    ["O2'", 3.1, 1.6, 0.5, 'O'],
    ["C1'", 0.7, 2.0, 0.8, 'C'],
    ["O4'", -0.3, 1.2, 0.6, 'O'],
    ["C5'", -0.9, -0.6, 1.0, 'C'],
    ["O5'", -2.2, -0.5, 0.6, 'O'],
    ['N1', 0.7, 3.4, 0.4, 'N'],
    ['C2', 1.8, 4.2, 0.2, 'C'],
    ['O2', 2.9, 3.8, 0.4, 'O'],
    ['N3', 1.7, 5.5, -0.2, 'N'],
    ['C4', 0.5, 6.1, -0.4, 'C'],
    ['C5', -0.6, 5.3, -0.2, 'C'],
    ['C6', -0.5, 4.0, 0.2, 'C'],
  ];
  const lines = atoms.map(([name, x, y, z, elem]) =>
    pdbAtomLine('HETATM', s++, name, 'DHZ', chain, resSeq, x + offsetX, y, z, elem),
  );
  return { lines, nextSerial: s };
}

function makeRnaChain(chain: string, startSerial: number, offsetY = 20): { lines: string[]; nextSerial: number } {
  let s = startSerial;
  const lines: string[] = [];
  const bases = ['C', 'G', 'A'];
  for (let i = 0; i < bases.length; i += 1) {
    const x0 = i * 5.5;
    const b = bases[i];
    const r = i + 1;
    lines.push(pdbAtomLine('ATOM  ', s++, 'P', b, chain, r, x0, offsetY, 0, 'P'));
    lines.push(pdbAtomLine('ATOM  ', s++, "O5'", b, chain, r, x0 + 1.4, offsetY, 0, 'O'));
    lines.push(pdbAtomLine('ATOM  ', s++, "C5'", b, chain, r, x0 + 2.2, offsetY + 1.0, 0, 'C'));
    lines.push(pdbAtomLine('ATOM  ', s++, "C4'", b, chain, r, x0 + 3.5, offsetY + 0.6, 0, 'C'));
    lines.push(pdbAtomLine('ATOM  ', s++, "C3'", b, chain, r, x0 + 4.1, offsetY - 0.6, 0, 'C'));
    lines.push(pdbAtomLine('ATOM  ', s++, "O3'", b, chain, r, x0 + 5.2, offsetY - 0.2, 0, 'O'));
    lines.push(pdbAtomLine('ATOM  ', s++, "O2'", b, chain, r, x0 + 3.2, offsetY - 1.5, 0.8, 'O'));
    lines.push(pdbAtomLine('ATOM  ', s++, 'N1', b, chain, r, x0 + 2.5, offsetY + 2.5, 0.5, 'N'));
  }
  lines.push('TER');
  return { lines, nextSerial: s };
}

async function parsePdbToModel(pdbText: string) {
  const parsed = await parsePDB(pdbText, 'test').run();
  if (parsed.isError) throw new Error(parsed.message);
  const traj = await trajectoryFromPDB(parsed.result).run();
  const model = traj.representative;
  return { traj, model };
}

function countNucleotideElements(struct: Structure): number {
  let total = 0;
  for (const u of struct.units) {
    total += (u as { nucleotideElements?: { length: number } }).nucleotideElements?.length ?? 0;
  }
  return total;
}

describe('pruneFalseNucleicMonomers', () => {
  it('removes false RNA classification on single-residue HETATM DHZ while preserving non-standard ball-and-stick selection', async () => {
    const prot = makeProteinChain('A', 1);
    const dhz = makeDhzLigand('B', 1, prot.nextSerial, 10);
    const pdb = [...prot.lines, ...dhz.lines, 'END'].join('\n');

    // Confirm raw Mol* PDB parser indeed misclassifies DHZ as a nucleotide
    const raw = await parsePdbToModel(pdb);
    const rawStruct = Structure.ofModel(raw.model);
    expect(countNucleotideElements(rawStruct)).toBe(1);

    // After sanitizing, DHZ has 0 nucleotideElements (no phantom cyan cartoon stick/ring)
    const sanitized = await parsePdbToModel(pdb);
    const demoted = pruneFalseNucleicMonomers(sanitized.traj, sanitized.model);
    expect(demoted).toBe(1);

    const cleanStruct = Structure.ofModel(sanitized.model);
    expect(countNucleotideElements(cleanStruct)).toBe(0);

    // DHZ must still be matched by Mol*'s non-standard component query so ball-and-stick renders
    const nonStdSel = StructureSelectionQueries.nonStandardPolymer.query(new QueryContext(cleanStruct));
    const nonStdStruct = StructureSelection.unionStructure(nonStdSel);
    expect(nonStdStruct.elementCount).toBe(16);
  });

  it('demotes unlinked duplicate DHZ ligands sharing one chain while preserving a real RNA chain in the same complex', async () => {
    const prot = makeProteinChain('A', 1);
    const rna = makeRnaChain('B', prot.nextSerial, 20);
    const dhz1 = makeDhzLigand('C', 1, rna.nextSerial, 10);
    const dhz2 = makeDhzLigand('C', 2, dhz1.nextSerial, 40);
    const pdb = [...prot.lines, ...rna.lines, ...dhz1.lines, ...dhz2.lines, 'END'].join('\n');

    const { traj, model } = await parsePdbToModel(pdb);
    const demoted = pruneFalseNucleicMonomers(traj, model);
    expect(demoted).toBe(2);

    const struct = Structure.ofModel(model);
    // Only the 3 real RNA nucleotides remain in nucleotideElements (not the 2 DHZ ligands)
    expect(countNucleotideElements(struct)).toBe(3);
  });
});
