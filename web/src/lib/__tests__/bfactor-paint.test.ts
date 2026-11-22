import { describe, expect, it } from 'vitest';
import type { MutationMatrixArtifact, ResidueTrackArtifact } from '$lib/engine/types';
import {
  formatMutationLabel,
  formatMutationMatrixCsv,
  formatResidueTrackCsv,
  paintStructureBfactors,
  rankTopMutations,
} from '$lib/protein-ops/bfactor-paint';

const SAMPLE_PDB = [
  'REMARK   1 TEST ANTIBODY + LIGAND + ION + WATER + DNA',
  'ATOM      1  N   ALA H 100      10.000  20.000  30.000  1.00 12.34           N  ',
  'ATOM      2  CA  ALA H 100      11.000  20.000  30.000  1.00 12.34           C  ',
  'ATOM      3  N   GLY H 100A     12.000  20.000  30.000  1.00 15.00           N  ',
  'ATOM      4  CA  GLY H 100A     13.000  20.000  30.000  1.00 15.00           C  ',
  'ATOM      5  N   TYR H 100B     14.000  20.000  30.000  1.00 18.00           N  ',
  'ATOM      6  CA  TYR H 100B     15.000  20.000  30.000  1.00 18.00           C  ',
  'HETATM    7  CA  MSE H 101      16.000  20.000  30.000  1.00 22.00           C  ',
  'TER       8      MSE H 101',
  'ATOM      9  P   DA  D   1      20.000  21.000  22.000  1.00 55.55           P  ',
  'HETATM   10  C1  ATP L 201      25.000  26.000  27.000  1.00 44.44           C  ',
  'HETATM   11 ZN   ZN  Z 301      28.000  29.000  30.000  1.00 33.33          ZN  ',
  'HETATM   12  O   HOH W 401      31.000  32.000  33.000  1.00 66.66           O  ',
  'END',
  '',
].join('\n');

const SAMPLE_MMCIF = [
  'data_test_complex',
  '#',
  '_entry.id TEST',
  '#',
  'loop_',
  '_atom_site.group_PDB',
  '_atom_site.id',
  '_atom_site.type_symbol',
  '_atom_site.label_atom_id',
  '_atom_site.label_alt_id',
  '_atom_site.label_comp_id',
  '_atom_site.label_asym_id',
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
  '_atom_site.pdbx_PDB_model_num',
  'ATOM 1 N N . ALA A 1 ? 1.000 2.000 3.000 1.00 10.00 100 ALA AA 1',
  'ATOM 2 C CA . ALA A 1 ? 1.500 2.000 3.000 1.00 10.00 100 ALA AA 1',
  'ATOM 3 N N . GLY A 2 A 2.000 2.000 3.000 1.00 11.00 100 GLY AA 1',
  'ATOM 4 C CA . GLY A 2 A 2.500 2.000 3.000 1.00 11.00 100 GLY AA 1',
  'ATOM 5 C CA . VAL B 1 ? 5.000 6.000 7.000 1.00 12.00 1 VAL AB 1',
  'ATOM 6 O "O5\'" . DA D 1 ? 8.000 8.000 8.000 1.00 55.55 1 DA DNA1 1',
  'HETATM 7 C C1 . ATP L . ? 9.000 9.000 9.000 1.00 44.44 201 ATP LIG1 1',
  'HETATM 8 ZN ZN . ZN Z . ? 10.000 10.000 10.000 1.00 33.33 301 ZN ZN1 1',
  '#',
  '',
].join('\n');

describe('paintStructureBfactors (PDB)', () => {
  it('rewrites polymer B-factors in cols 60..66 while preserving insertion codes, HETATM ligands/ions/waters, and DNA', () => {
    const values = new Float32Array([0.12, 0.85, 0.43, 0.99]);
    const painted = paintStructureBfactors(SAMPLE_PDB, values, { scale: 100 });
    const lines = painted.split('\n');

    // Residue 0: H 100 (both N & CA painted to 12.00)
    expect(lines[1].slice(60, 66)).toBe(' 12.00');
    expect(lines[2].slice(60, 66)).toBe(' 12.00');
    expect(lines[1].slice(0, 60)).toBe(SAMPLE_PDB.split('\n')[1].slice(0, 60));
    expect(lines[1].slice(66)).toBe(SAMPLE_PDB.split('\n')[1].slice(66));

    // Residue 1: H 100A (insertion code A -> 85.00)
    expect(lines[3].slice(60, 66)).toBe(' 85.00');
    expect(lines[4].slice(60, 66)).toBe(' 85.00');

    // Residue 2: H 100B (insertion code B -> 43.00)
    expect(lines[5].slice(60, 66)).toBe(' 43.00');
    expect(lines[6].slice(60, 66)).toBe(' 43.00');

    // Residue 3: HETATM MSE (selenomethionine polymer -> 99.00)
    expect(lines[7].slice(60, 66)).toBe(' 99.00');

    // Untouched: TER, DNA (DA), ATP ligand, ZN ion, HOH water, END
    expect(lines[8]).toBe('TER       8      MSE H 101');
    expect(lines[9].slice(60, 66)).toBe(' 55.55');
    expect(lines[10].slice(60, 66)).toBe(' 44.44');
    expect(lines[11].slice(60, 66)).toBe(' 33.33');
    expect(lines[12].slice(60, 66)).toBe(' 66.66');
    expect(lines[13]).toBe('END');
  });

  it('supports negative ΔΔG values, clamping, and skips NaN residues without corrupting columns', () => {
    const values = [-2.456, Number.NaN, 12.5, -150.0];
    const painted = paintStructureBfactors(SAMPLE_PDB, values, {
      clampMin: -5,
      clampMax: 5,
    });
    const lines = painted.split('\n');

    expect(lines[1].slice(60, 66)).toBe(' -2.46');
    // NaN preserves existing B-factor (15.00)
    expect(lines[3].slice(60, 66)).toBe(' 15.00');
    // Clamped to +5.00
    expect(lines[5].slice(60, 66)).toBe('  5.00');
    // Clamped to -5.00
    expect(lines[7].slice(60, 66)).toBe(' -5.00');
  });

  it('paints all models consistently in multi-model PDB trajectories', () => {
    const multi = [
      'MODEL        1',
      'ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  ',
      'ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C  ',
      'ENDMDL',
      'MODEL        2',
      'ATOM      1  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C  ',
      'ATOM      2  CA  GLY A   2       4.500   5.500   6.500  1.00  0.00           C  ',
      'ENDMDL',
      'END',
    ].join('\n');

    const out = paintStructureBfactors(multi, [0.25, 0.75], { scale: 100 });
    const lines = out.split('\n');
    expect(lines[1].slice(60, 66)).toBe(' 25.00');
    expect(lines[2].slice(60, 66)).toBe(' 75.00');
    expect(lines[5].slice(60, 66)).toBe(' 25.00');
    expect(lines[6].slice(60, 66)).toBe(' 75.00');
  });
});

describe('paintStructureBfactors (mmCIF)', () => {
  it('rewrites _atom_site.B_iso_or_equiv per polymer residue across multi-char chains & insertion codes while preserving ligands/DNA', () => {
    const values = new Float32Array([0.1, 0.88, 0.42]);
    const painted = paintStructureBfactors(SAMPLE_MMCIF, values, { scale: 100 });
    const lines = painted.split('\n');

    const row = (prefix: string) => lines.find((l) => l.startsWith(prefix))!;
    // Multi-char chain AA, residue 100 (no insCode) -> 10.00
    expect(row('ATOM 1 N')).toContain('1.00 10.00 100 ALA AA 1');
    expect(row('ATOM 2 C')).toContain('1.00 10.00 100 ALA AA 1');
    // Multi-char chain AA, residue 100A (insCode A) -> 88.00
    expect(row('ATOM 3 N')).toContain('1.00 88.00 100 GLY AA 1');
    expect(row('ATOM 4 C')).toContain('1.00 88.00 100 GLY AA 1');
    // Multi-char chain AB, residue 1 -> 42.00
    expect(row('ATOM 5 C')).toContain('1.00 42.00 1 VAL AB 1');

    // DNA (with quoted atom name "O5'"), ATP ligand, ZN ion remain completely unchanged
    expect(row('ATOM 6 O')).toBe('ATOM 6 O "O5\'" . DA D 1 ? 8.000 8.000 8.000 1.00 55.55 1 DA DNA1 1');
    expect(row('HETATM 7 C')).toBe('HETATM 7 C C1 . ATP L . ? 9.000 9.000 9.000 1.00 44.44 201 ATP LIG1 1');
    expect(row('HETATM 8 ZN')).toBe('HETATM 8 ZN ZN . ZN Z . ? 10.000 10.000 10.000 1.00 33.33 301 ZN ZN1 1');
  });

  it('appends _atom_site.B_iso_or_equiv when absent from an mmCIF header', () => {
    const noBiso = [
      'data_minimal',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM CA ALA AA 1 0.0 0.0 0.0',
      'HETATM ZN ZN Z 2 5.0 5.0 5.0',
    ].join('\n');

    const painted = paintStructureBfactors(noBiso, [-1.25]);
    expect(painted).toContain('_atom_site.B_iso_or_equiv');
    expect(painted).toContain('ATOM CA ALA AA 1 0.0 0.0 0.0 -1.25');
    expect(painted).toContain('HETATM ZN ZN Z 2 5.0 5.0 5.0 0.00');
  });
});

describe('rankTopMutations', () => {
  const ALPHABET = 'ACDEFGHIKLMNPQRSTVWY';

  function makeMatrix(direction: 'lower-is-better' | 'higher-is-better'): MutationMatrixArtifact {
    // 3-residue sequence: A G K
    const L = 3;
    const K = ALPHABET.length;
    const values = new Float32Array(L * K).fill(0);
    const set = (pos: number, aa: string, val: number) => {
      const col = ALPHABET.indexOf(aa);
      values[pos * K + col] = val;
    };

    // Wild-type self-mutations with extreme scores that MUST be skipped
    set(0, 'A', -99.9);
    set(1, 'G', 99.9);
    set(2, 'K', -50.0);

    // Non-finite scores that MUST be skipped
    set(0, 'C', Number.NaN);
    set(1, 'D', Number.POSITIVE_INFINITY);
    set(2, 'E', Number.NEGATIVE_INFINITY);

    // Real mutations
    set(0, 'G', -1.85); // A42G
    set(1, 'A', -2.40); // G43A
    set(2, 'R', 3.10);  // K44R
    set(0, 'V', 2.25);  // A42V
    set(2, 'L', -0.90); // K44L

    return {
      kind: 'MutationMatrix',
      id: 'mut_test',
      sourceModelId: direction === 'lower-is-better' ? 'thermompnn' : 'esmc-300m',
      metric: direction === 'lower-is-better' ? 'ddg-kcal-mol' : 'llr-fitness',
      unit: direction === 'lower-is-better' ? 'kcal/mol' : 'LLR',
      direction,
      residues: 'AGK',
      residueIds: ['A42', 'A43', 'A44'],
      alphabet: ALPHABET,
      values,
    };
  }

  it('ranks lower-is-better (ThermoMPNN ΔΔG) ascending while skipping WT and non-finite entries', () => {
    const ranked = rankTopMutations(makeMatrix('lower-is-better'), 3);
    expect(ranked).toHaveLength(3);
    expect(ranked[0].label).toBe('G43A');
    expect(ranked[0].score).toBeCloseTo(-2.4, 5);
    expect(ranked[0].mutantSequence).toBe('AAK');
    expect(ranked[0].rank).toBe(1);

    expect(ranked[1].label).toBe('A42G');
    expect(ranked[1].score).toBeCloseTo(-1.85, 5);
    expect(ranked[1].mutantSequence).toBe('GGK');

    expect(ranked[2].label).toBe('K44L');
    expect(ranked[2].score).toBeCloseTo(-0.9, 5);
    expect(ranked[2].mutantSequence).toBe('AGL');

    // Ensure no WT self-mutations or NaNs leaked in
    for (const m of ranked) {
      expect(m.mutant).not.toBe(m.wildType);
      expect(Number.isFinite(m.score)).toBe(true);
    }
  });

  it('ranks higher-is-better (ESMC LLR) descending while skipping WT and non-finite entries', () => {
    const ranked = rankTopMutations(makeMatrix('higher-is-better'), 2);
    expect(ranked).toHaveLength(2);
    expect(ranked[0].label).toBe('K44R');
    expect(ranked[0].score).toBeCloseTo(3.1, 5);
    expect(ranked[0].mutantSequence).toBe('AGR');

    expect(ranked[1].label).toBe('A42V');
    expect(ranked[1].score).toBeCloseTo(2.25, 5);
    expect(ranked[1].mutantSequence).toBe('VGK');
  });

  it('prefixes chain IDs on multi-chain matrices and preserves insertion codes', () => {
    expect(formatMutationLabel('G', 'H100A', 'D', true)).toBe('H:G100AD');
    expect(formatMutationLabel('A', 'A42', 'G', false)).toBe('A42G');
  });
});

describe('CSV formatters', () => {
  it('formats MutationMatrixArtifact as clean CSV with blank cells for non-finite values', () => {
    const matrix: MutationMatrixArtifact = {
      kind: 'MutationMatrix',
      id: 'm1',
      sourceModelId: 'thermompnn',
      metric: 'ddg-kcal-mol',
      direction: 'lower-is-better',
      residues: 'AG',
      residueIds: ['H100', 'H100A'],
      alphabet: 'ACG',
      values: new Float32Array([0, -1.25, Number.NaN, 0.5, -0.75, 0]),
      meanPerPosition: new Float32Array([-0.625, -0.125]),
    };

    const csv = formatMutationMatrixCsv(matrix);
    const lines = csv.trim().split('\n');
    expect(lines[0]).toBe('position,residue_id,wild_type,A,C,G,mean');
    expect(lines[1]).toBe('1,H100,A,0.0000,-1.2500,,-0.6250');
    expect(lines[2]).toBe('2,H100A,G,0.5000,-0.7500,0.0000,-0.1250');
  });

  it('formats ResidueTrackArtifact as clean CSV with optional raw_logit column', () => {
    const track: ResidueTrackArtifact = {
      kind: 'ResidueTrack',
      id: 't1',
      sourceModelId: 'dyna1',
      quantity: 'p(ms exchange)',
      unit: 'prob',
      residues: 'AGY',
      residueIds: ['H100', 'H100A', 'H100B'],
      values: new Float32Array([0.125, 0.78, Number.NaN]),
      rawLogits: new Float32Array([-1.9459, 1.2657, 0]),
    };

    const csv = formatResidueTrackCsv(track);
    const lines = csv.trim().split('\n');
    expect(lines[0]).toBe('position,residue_id,residue,value,raw_logit');
    expect(lines[1]).toBe('1,H100,A,0.125000,-1.945900');
    expect(lines[2]).toBe('2,H100A,G,0.780000,1.265700');
    expect(lines[3]).toBe('3,H100B,Y,,0.000000');
  });

  it('MolSuperposeViewer samples polymer CA B-factors so HETATM ACE/HOH crystal B-factors cannot skew Dyna-1 [0,1] domain', async () => {
    const fs = await import('node:fs');
    const path = await import('node:path');
    const viewerSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/lib/viewer/MolSuperposeViewer.svelte'),
      'utf8',
    );
    expect(viewerSrc).toContain('u?.polymerElements');
    expect(viewerSrc).toContain("atomIdCol.value(el) === 'CA'");
  });
});
