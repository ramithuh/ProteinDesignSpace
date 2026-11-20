import { describe, it, expect } from 'vitest';
import {
  AA1_TO_AA3,
  AA3_TO_AA1,
  AF2_ATOM37_NAMES,
  atom37ToPdb,
  coordsToBackbonePdb,
  coordsToMultiModelPdb,
  extractCaFromBackbone,
  generateSampleHelixBundlePdb,
  parsePdbBackbone,
  parseStructureText,
  reconstructBackboneOxygen,
} from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';

function dist3(a: [number, number, number], b: [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function angleDeg(
  a: [number, number, number],
  vertex: [number, number, number],
  b: [number, number, number],
): number {
  const u = [a[0] - vertex[0], a[1] - vertex[1], a[2] - vertex[2]];
  const v = [b[0] - vertex[0], b[1] - vertex[1], b[2] - vertex[2]];
  const lu = Math.hypot(u[0], u[1], u[2]);
  const lv = Math.hypot(v[0], v[1], v[2]);
  const cos = Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function buildIdealPeptideNcac(nRes: number): Float32Array {
  // Idealized trans-peptide zigzag in the XY plane (z = 0) with realistic bond lengths & ~120 deg angles
  const ncac = new Float32Array(nRes * 3 * 3);
  for (let r = 0; r < nRes; r++) {
    const x0 = r * 3.8;
    const sign = r % 2 === 0 ? 1 : -1;
    // N, CA, C
    ncac[(r * 3 + 0) * 3 + 0] = x0 - 1.2;
    ncac[(r * 3 + 0) * 3 + 1] = sign * 0.6;
    ncac[(r * 3 + 0) * 3 + 2] = 0.0;

    ncac[(r * 3 + 1) * 3 + 0] = x0;
    ncac[(r * 3 + 1) * 3 + 1] = sign * 0.1;
    ncac[(r * 3 + 1) * 3 + 2] = 0.0;

    ncac[(r * 3 + 2) * 3 + 0] = x0 + 1.25;
    ncac[(r * 3 + 2) * 3 + 1] = sign * 0.7;
    ncac[(r * 3 + 2) * 3 + 2] = 0.0;
  }
  return ncac;
}

describe('protein-ops/pdb', () => {
  it('reconstructBackboneOxygen produces exact 1.231 A C=O bonds and planar geometry', () => {
    const nRes = 6;
    const ncac = buildIdealPeptideNcac(nRes);
    const bb4 = reconstructBackboneOxygen(ncac, nRes);

    expect(bb4.length).toBe(nRes * 4 * 3);

    for (let r = 0; r < nRes; r++) {
      // N, CA, C preserved bit-for-bit
      for (let a = 0; a < 3; a++) {
        for (let k = 0; k < 3; k++) {
          expect(bb4[(r * 4 + a) * 3 + k]).toBe(ncac[(r * 3 + a) * 3 + k]);
        }
      }

      const ca: [number, number, number] = [
        bb4[(r * 4 + 1) * 3 + 0],
        bb4[(r * 4 + 1) * 3 + 1],
        bb4[(r * 4 + 1) * 3 + 2],
      ];
      const c: [number, number, number] = [
        bb4[(r * 4 + 2) * 3 + 0],
        bb4[(r * 4 + 2) * 3 + 1],
        bb4[(r * 4 + 2) * 3 + 2],
      ];
      const o: [number, number, number] = [
        bb4[(r * 4 + 3) * 3 + 0],
        bb4[(r * 4 + 3) * 3 + 1],
        bb4[(r * 4 + 3) * 3 + 2],
      ];

      // Exact C=O bond length = 1.231 A
      expect(dist3(c, o)).toBeCloseTo(1.231, 4);

      if (r + 1 < nRes) {
        // Interior residue: O lies in the (CA_i, C_i, N_{i+1}) plane (z=0 here)
        expect(Math.abs(o[2])).toBeLessThan(1e-5);
        // Bisector opposite CA_i and N_{i+1}: angles CA-C-O and N_{next}-C-O must be equal
        const nextN: [number, number, number] = [
          bb4[((r + 1) * 4 + 0) * 3 + 0],
          bb4[((r + 1) * 4 + 0) * 3 + 1],
          bb4[((r + 1) * 4 + 0) * 3 + 2],
        ];
        const a1 = angleDeg(ca, c, o);
        const a2 = angleDeg(nextN, c, o);
        expect(Math.abs(a1 - a2)).toBeLessThan(1e-3);
        expect(a1).toBeGreaterThan(95);
        expect(a1).toBeLessThan(150);
      } else {
        // C-terminal fallback: exact 120.8 deg CA-C-O angle in (N, CA, C) plane
        expect(Math.abs(o[2])).toBeLessThan(1e-5);
        const termAngle = angleDeg(ca, c, o);
        expect(termAngle).toBeCloseTo(120.8, 1);
      }
    }
  });

  it('extractCaFromBackbone extracts C-alpha from both 3-atom and 4-atom buffers', () => {
    const ncac = buildIdealPeptideNcac(4);
    const bb4 = reconstructBackboneOxygen(ncac, 4);
    const ca3 = extractCaFromBackbone(ncac, 4, 3);
    const ca4 = extractCaFromBackbone(bb4, 4, 4);
    expect(ca3.length).toBe(12);
    expect(Array.from(ca3)).toEqual(Array.from(ca4));
  });

  it('parsePdbBackbone handles multi-chain structures and reconstructs missing O', () => {
    const ncacA = buildIdealPeptideNcac(3);
    const ncacB = buildIdealPeptideNcac(2);
    // Offset chain B by +20 in Z
    for (let i = 0; i < ncacB.length; i += 3) ncacB[i + 2] += 20;

    const pdbA = coordsToBackbonePdb(ncacA, 3, { atomsPerRes: 3, sequence: 'ACD', chainId: 'A' })
      .replace(/END\s*$/, '');
    const pdbB = coordsToBackbonePdb(ncacB, 2, { atomsPerRes: 3, sequence: 'EF', chainId: 'B' });
    const combined = `${pdbA}TER\n${pdbB}`;

    const parsed = parseStructureText(combined);
    expect(parsed.format).toBe('pdb');
    expect(parsed.nResidues).toBe(5);
    expect(parsed.sequence).toBe('ACDEF');
    expect(parsed.chainIds).toEqual(['A', 'A', 'A', 'B', 'B']);
    expect(parsed.caCoords.length).toBe(15);
    expect(parsed.backbone4Coords.length).toBe(5 * 4 * 3);
    // Reconstructed O atoms should be present in normalized pdbText
    expect(parsed.pdbText).toContain(' O  ');
    expect(parsed.caCoords[3 * 3 + 2]).toBeCloseTo(20, 3);
  });

  it('parsePdbBackbone filters altLocs (accepts blank, A, 1; ignores B, 2)', () => {
    const pdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00           N  ',
      'ATOM      2  CA AALA A   1       1.458   0.000   0.000  0.60 50.00           C  ',
      'ATOM      3  CA BALA A   1      99.000  99.000  99.000  0.40 50.00           C  ',
      'ATOM      4  C   ALA A   1       2.009   1.420   0.000  1.00 50.00           C  ',
      'ATOM      5  O   ALA A   1       1.251   2.390   0.000  1.00 50.00           O  ',
      'ATOM      6  N   GLY A   2       3.320   1.500   0.000  1.00 50.00           N  ',
      'ATOM      7  CA 1GLY A   2       4.000   2.800   0.000  0.70 50.00           C  ',
      'ATOM      8  CA 2GLY A   2     -88.000 -88.000 -88.000  0.30 50.00           C  ',
      'ATOM      9  C   GLY A   2       5.500   2.600   0.000  1.00 50.00           C  ',
      'ATOM     10  O   GLY A   2       6.000   1.500   0.000  1.00 50.00           O  ',
      'END',
    ].join('\n');

    const parsed = parsePdbBackbone(pdb);
    expect(parsed.nResidues).toBe(2);
    expect(parsed.sequence).toBe('AG');
    // Residue 1 CA must be altLoc 'A' (1.458), NOT altLoc 'B' (99.0)
    expect(parsed.caCoords[0]).toBeCloseTo(1.458, 3);
    expect(parsed.caCoords[1]).toBeCloseTo(0.0, 3);
    // Residue 2 CA must be altLoc '1' (4.0), NOT altLoc '2' (-88.0)
    expect(parsed.caCoords[3]).toBeCloseTo(4.0, 3);
    expect(parsed.caCoords[4]).toBeCloseTo(2.8, 3);
  });

  it('parseStructureText parses multi-chain mmCIF _atom_site blocks', () => {
    const cif = [
      'data_test_complex',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 N MET A 1 0.000 0.000 0.000',
      'ATOM 2 CA MET A 1 1.458 0.000 0.000',
      'ATOM 3 C MET A 1 2.009 1.420 0.000',
      'ATOM 4 O MET A 1 1.251 2.390 0.000',
      'ATOM 5 N LYS A 2 3.320 1.500 0.000',
      'ATOM 6 CA LYS A 2 4.000 2.800 0.000',
      'ATOM 7 C LYS A 2 5.500 2.600 0.000',
      'ATOM 8 N TRP B 10 10.000 1.000 2.000',
      'ATOM 9 CA TRP B 10 11.450 1.200 2.100',
      'ATOM 10 C TRP B 10 12.100 2.500 2.200',
      'HETATM 11 C1 LIG C 1 20.000 20.000 20.000',
      '#',
    ].join('\n');

    const parsed = parseStructureText(cif);
    expect(parsed.format).toBe('mmcif');
    expect(parsed.nResidues).toBe(3);
    expect(parsed.sequence).toBe('MKW');
    expect(parsed.chainIds).toEqual(['A', 'A', 'B']);
    expect(parsed.caCoords[0]).toBeCloseTo(1.458, 3);
    expect(parsed.caCoords[6]).toBeCloseTo(11.45, 3);
    expect(parsed.caCoords[8]).toBeCloseTo(2.1, 3);
    expect(parsed.backbone4Coords.length).toBe(3 * 4 * 3);
    // Converted pdbText is parseable back identically
    const reparsed = parsePdbBackbone(parsed.pdbText);
    expect(reparsed.sequence).toBe('MKW');
    expect(reparsed.nResidues).toBe(3);
  });

  it('atom37ToPdb round-trips sequence, coords, pLDDT B-factors, and respects masks', () => {
    const seq = 'ACDEFGHIKLMNPQRSTVWY';
    const L = seq.length;
    expect(AF2_ATOM37_NAMES.length).toBe(37);
    expect(Object.keys(AA1_TO_AA3).length).toBe(21);
    expect(Object.keys(AA3_TO_AA1).length).toBe(22);
    expect(AA3_TO_AA1.MSE).toBe('M');

    const ncac = buildIdealPeptideNcac(L);
    const bb4 = reconstructBackboneOxygen(ncac, L);
    const atom37 = new Float32Array(L * 37 * 3);
    const mask = new Float32Array(L * 37);
    const plddt = new Float32Array(L);

    for (let r = 0; r < L; r++) {
      plddt[r] = 70 + r * 1.2;
      // N(0), CA(1), C(2), O(4)
      const slotForBb = [0, 1, 2, 4];
      for (let k = 0; k < 4; k++) {
        const slot = slotForBb[k];
        mask[r * 37 + slot] = 1;
        for (let d = 0; d < 3; d++) {
          atom37[(r * 37 + slot) * 3 + d] = bb4[(r * 4 + k) * 3 + d];
        }
      }
      // Add CB (slot 3) for non-GLY
      if (seq[r] !== 'G') {
        mask[r * 37 + 3] = 1;
        atom37[(r * 37 + 3) * 3 + 0] = bb4[(r * 4 + 1) * 3 + 0];
        atom37[(r * 37 + 3) * 3 + 1] = bb4[(r * 4 + 1) * 3 + 1];
        atom37[(r * 37 + 3) * 3 + 2] = 1.53;
      }
      // Masked-out slot with NaN must be safely skipped
      mask[r * 37 + 10] = 0;
      atom37[(r * 37 + 10) * 3 + 0] = Number.NaN;
    }

    const { pdbText, caCoords, backbone4Coords } = atom37ToPdb(atom37, mask, seq, plddt, 'C');
    expect(caCoords.length).toBe(L * 3);
    expect(backbone4Coords.length).toBe(L * 4 * 3);
    expect(pdbText).not.toContain('NaN');
    expect(pdbText).toContain('82.00'); // plddt[10] = 70 + 12 = 82.00

    const parsed = parsePdbBackbone(pdbText);
    expect(parsed.nResidues).toBe(L);
    expect(parsed.sequence).toBe(seq);
    expect(parsed.chainIds.every((c) => c === 'C')).toBe(true);
    for (let i = 0; i < caCoords.length; i++) {
      expect(parsed.caCoords[i]).toBeCloseTo(caCoords[i], 2);
    }
    for (let i = 0; i < backbone4Coords.length; i++) {
      expect(parsed.backbone4Coords[i]).toBeCloseTo(backbone4Coords[i], 2);
    }
  });

  it('generateSampleHelixBundlePdb produces a clean 3-helix bundle with 0 breaks, 0 clashes, meanCa ~3.8 A', () => {
    const pdb1 = generateSampleHelixBundlePdb();
    const pdb2 = generateSampleHelixBundlePdb();
    expect(pdb1).toBe(pdb2); // deterministic

    const parsed = parseStructureText(pdb1);
    expect(parsed.format).toBe('pdb');
    expect(parsed.nResidues).toBeGreaterThanOrEqual(50);
    expect(parsed.nResidues).toBeLessThanOrEqual(65);
    expect(parsed.sequence.length).toBe(parsed.nResidues);

    const stats = computeGeometryStats(parsed.caCoords);
    expect(stats.length).toBe(parsed.nResidues);
    expect(stats.chainBreaks).toBe(0);
    expect(stats.clashes).toBe(0);
    expect(stats.meanCaDist).toBeGreaterThan(3.65);
    expect(stats.meanCaDist).toBeLessThan(3.95);
    expect(stats.helixPct).toBeGreaterThan(50);
    expect(stats.rg).toBeGreaterThan(6);
    expect(stats.rg).toBeLessThan(20);
  });

  it('coordsToMultiModelPdb puts FINAL frame in MODEL 1 followed by frames 0..T-1', () => {
    const L = 4;
    const f0 = buildIdealPeptideNcac(L);
    const f1 = new Float32Array(f0.length);
    const f2 = new Float32Array(f0.length);
    for (let i = 0; i < f0.length; i++) {
      f1[i] = f0[i] + 5.0;
      f2[i] = f0[i] + 12.5;
    }

    const multiPdb = coordsToMultiModelPdb([f0, f1, f2], L, 3);
    const modelHeaders = multiPdb.match(/^MODEL\s+\d+/gm) ?? [];
    const endMdls = multiPdb.match(/^ENDMDL$/gm) ?? [];
    expect(modelHeaders.length).toBe(4); // 1 final + 3 frames
    expect(endMdls.length).toBe(4);
    expect(multiPdb.trimEnd().endsWith('END')).toBe(true);

    // parsePdbBackbone reads MODEL 1 only -> must equal final frame f2
    const model1 = parsePdbBackbone(multiPdb);
    expect(model1.nResidues).toBe(L);
    const expectedCaFinal = extractCaFromBackbone(f2, L, 3);
    for (let i = 0; i < expectedCaFinal.length; i++) {
      expect(model1.caCoords[i]).toBeCloseTo(expectedCaFinal[i], 2);
    }

    // Verify MODEL 2 = f0, MODEL 3 = f1, MODEL 4 = f2
    const blocks = multiPdb.split(/^MODEL\s+\d+\n/m).slice(1);
    expect(blocks.length).toBe(4);
    const m2 = parsePdbBackbone(blocks[1]);
    const m3 = parsePdbBackbone(blocks[2]);
    const m4 = parsePdbBackbone(blocks[3]);
    expect(m2.caCoords[0]).toBeCloseTo(f0[3], 2);
    expect(m3.caCoords[0]).toBeCloseTo(f1[3], 2);
    expect(m4.caCoords[0]).toBeCloseTo(f2[3], 2);
  });

  it('coordsToMultiModelPdb returns END\\n for empty trajectory', () => {
    expect(coordsToMultiModelPdb([], 5)).toBe('END\n');
  });

  it('reconstructBackboneOxygen respects chainIds boundaries and >= 2.0 A chain-break guard', () => {
    const ncac = buildIdealPeptideNcac(4);
    // Move residues 2..3 far away (+50 A in Z) to create a break / second chain
    for (let r = 2; r < 4; r++) {
      for (let a = 0; a < 3; a++) {
        ncac[(r * 3 + a) * 3 + 2] += 50;
      }
    }
    const bb4WithChains = reconstructBackboneOxygen(ncac, 4, ['A', 'A', 'B', 'B']);
    const bb4NoChains = reconstructBackboneOxygen(ncac, 4);

    // For residue 1 (last residue of chain A), d(C_1, N_2) > 50 A >= 2.0 A -> must use (N, CA, C) plane fallback (z = 0, angle = 120.8 deg)
    for (const bb4 of [bb4WithChains, bb4NoChains]) {
      const ca: [number, number, number] = [bb4[(1 * 4 + 1) * 3], bb4[(1 * 4 + 1) * 3 + 1], bb4[(1 * 4 + 1) * 3 + 2]];
      const c: [number, number, number] = [bb4[(1 * 4 + 2) * 3], bb4[(1 * 4 + 2) * 3 + 1], bb4[(1 * 4 + 2) * 3 + 2]];
      const o: [number, number, number] = [bb4[(1 * 4 + 3) * 3], bb4[(1 * 4 + 3) * 3 + 1], bb4[(1 * 4 + 3) * 3 + 2]];
      expect(dist3(c, o)).toBeCloseTo(1.231, 4);
      expect(Math.abs(o[2])).toBeLessThan(1e-4); // NOT pulled toward z=+50 of chain B!
      expect(angleDeg(ca, c, o)).toBeCloseTo(120.8, 1);
    }

    // Even when chain B residue 2 is placed close (< 2.0 A) in space, chainIds boundary forces C-term fallback
    const closeNcac = buildIdealPeptideNcac(2);
    // Bring res 1 N close to res 0 C (< 1.5 A) with a Z offset
    closeNcac[(1 * 3 + 0) * 3 + 0] = closeNcac[(0 * 3 + 2) * 3 + 0] + 1.3;
    closeNcac[(1 * 3 + 0) * 3 + 1] = closeNcac[(0 * 3 + 2) * 3 + 1];
    closeNcac[(1 * 3 + 0) * 3 + 2] = 0.5;
    const acrossChains = reconstructBackboneOxygen(closeNcac, 2, ['A', 'B']);
    // O for residue 0 must stay in its own (N,CA,C) z=0 plane despite nearby chain B N at z=0.5
    expect(Math.abs(acrossChains[(0 * 4 + 3) * 3 + 2])).toBeLessThan(1e-4);
  });

  it('parsePdbBackbone patches missing O in-place preserving chains/numbering/sidechains/HETATMs, maps MSE->M & unknown->X, and leaves CA-only untouched', () => {
    const pdb = [
      'ATOM      1  N   MSE A  10       0.000   0.000   0.000  1.00 50.00           N  ',
      'ATOM      2  CA  MSE A  10       1.458   0.000   0.000  1.00 50.00           C  ',
      'ATOM      3  C   MSE A  10       2.009   1.420   0.000  1.00 50.00           C  ',
      'ATOM      4  CB  MSE A  10       1.900  -0.750   1.200  1.00 50.00           C  ',
      'ATOM      5  N   XYZ A  10A      3.320   1.500   0.000  1.00 50.00           N  ',
      'ATOM      6  CA  XYZ A  10A      4.000   2.800   0.000  1.00 50.00           C  ',
      'ATOM      7  C   XYZ A  10A      5.500   2.600   0.000  1.00 50.00           C  ',
      'ATOM      8  O   XYZ A  10A      6.100   1.550   0.250  1.00 50.00           O  ',
      'TER',
      'ATOM      9  N   ALA B  50      20.000   0.000   0.000  1.00 60.00           N  ',
      'ATOM     10  CA  ALA B  50      21.458   0.000   0.000  1.00 60.00           C  ',
      'ATOM     11  C   ALA B  50      22.009   1.420   0.000  1.00 60.00           C  ',
      'HETATM   12  C1  LIG C 200      30.000  30.000  30.000  1.00 20.00           C  ',
      'END',
    ].join('\n');

    const parsed = parsePdbBackbone(pdb);
    expect(parsed.nResidues).toBe(3);
    expect(parsed.sequence).toBe('MXA'); // MSE->M, XYZ->X, ALA->A
    expect(parsed.chainIds).toEqual(['A', 'A', 'B']);
    // Real parsed O of residue 1 (10A) is preserved verbatim (z=0.250)
    expect(parsed.backbone4Coords[(1 * 4 + 3) * 3 + 0]).toBeCloseTo(6.1, 3);
    expect(parsed.backbone4Coords[(1 * 4 + 3) * 3 + 2]).toBeCloseTo(0.25, 3);
    // In-place patched PDB preserves sidechain CB, insertion code 10A, chain B resSeq 50, and HETATM LIG
    expect(parsed.pdbText).toContain(' CB  MSE A  10');
    expect(parsed.pdbText).toContain(' CA  XYZ A  10A');
    expect(parsed.pdbText).toContain(' CA  ALA B  50');
    expect(parsed.pdbText).toContain('HETATM   12  C1  LIG C 200');
    // Missing O atoms were inserted for A:10 and B:50
    expect(parsed.pdbText).toContain(' O   MSE A  10 ');
    expect(parsed.pdbText).toContain(' O   ALA B  50 ');

    // CA-only PDB must NOT have fake O lines synthesized
    const caOnly = [
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C  ',
      'ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 50.00           C  ',
      'ATOM      3  CA  VAL A   3       7.600   0.000   0.000  1.00 50.00           C  ',
      'END',
    ].join('\n');
    const parsedCa = parsePdbBackbone(caOnly);
    expect(parsedCa.pdbText).toBe(caOnly);
    expect(parsedCa.pdbText).not.toContain(' O  ');
  });

  it('parseStructureText mmCIF handles multi-model, altLocs, insCodes, real O preservation, MSE->M, and unknown->X', () => {
    const cif = [
      'data_multi_test',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.pdbx_PDB_model_num',
      // Model 1: res A:10 (MSE, altLoc A kept, altLoc B ignored), res A:10A (XYZ -> X, real O preserved), res B:5 (ALA)
      'ATOM 1 N . MSE A 10 ? 0.000 0.000 0.000 1',
      'ATOM 2 CA A MSE A 10 ? 1.458 0.000 0.000 1',
      'ATOM 3 CA B MSE A 10 ? 99.000 99.000 99.000 1',
      'ATOM 4 C . MSE A 10 ? 2.009 1.420 0.000 1',
      'ATOM 5 O . MSE A 10 ? 1.111 2.222 0.333 1',
      'ATOM 6 N . XYZ A 10 A 3.320 1.500 0.000 1',
      'ATOM 7 CA . XYZ A 10 A 4.000 2.800 0.000 1',
      'ATOM 8 C . XYZ A 10 A 5.500 2.600 0.000 1',
      'ATOM 9 O . XYZ A 10 A 6.200 1.700 -0.400 1',
      'ATOM 10 N . ALA B 5 ? 20.000 0.000 0.000 1',
      'ATOM 11 CA . ALA B 5 ? 21.458 0.000 0.000 1',
      'ATOM 12 C . ALA B 5 ? 22.009 1.420 0.000 1',
      // Model 2 (must be ignored completely!)
      'ATOM 13 CA . GLY A 10 ? -77.000 -77.000 -77.000 2',
      'ATOM 14 CA . GLY B 99 ? -88.000 -88.000 -88.000 2',
      '#',
    ].join('\n');

    const parsed = parseStructureText(cif);
    expect(parsed.format).toBe('mmcif');
    expect(parsed.nResidues).toBe(3);
    expect(parsed.sequence).toBe('MXA');
    expect(parsed.chainIds).toEqual(['A', 'A', 'B']);
    // AltLoc A kept (1.458), not altLoc B (99.0) or Model 2 (-77.0)
    expect(parsed.caCoords[0]).toBeCloseTo(1.458, 3);
    // Real parsed O coordinates preserved verbatim
    expect(parsed.backbone4Coords[(0 * 4 + 3) * 3 + 0]).toBeCloseTo(1.111, 3);
    expect(parsed.backbone4Coords[(0 * 4 + 3) * 3 + 2]).toBeCloseTo(0.333, 3);
    expect(parsed.backbone4Coords[(1 * 4 + 3) * 3 + 2]).toBeCloseTo(-0.4, 3);
    // Exported PDB preserves chain B, insertion code 10A, and TER between chains
    expect(parsed.pdbText).toContain('TER');
    expect(parsed.pdbText).toContain('A  10A');
    expect(parsed.pdbText).toContain('B   5 ');
  });
});
