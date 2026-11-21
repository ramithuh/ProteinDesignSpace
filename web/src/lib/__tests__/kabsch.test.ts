import { describe, it, expect } from 'vitest';
import {
  applyTransformToPdb,
  computeGeometryStats,
  kabschAlign,
  kabschAlignSubstructure,
} from '$lib/protein-ops/kabsch';
import { generateSampleHelixBundlePdb, parsePdbBackbone } from '$lib/protein-ops/pdb';

function det3(M: ArrayLike<number>): number {
  return (
    M[0] * (M[4] * M[8] - M[5] * M[7]) -
    M[1] * (M[3] * M[8] - M[5] * M[6]) +
    M[2] * (M[3] * M[7] - M[4] * M[6])
  );
}

/** Construct exact proper 3x3 rotation matrix from unit axis + angle via Rodrigues' formula. */
function axisAngleRotation(axis: [number, number, number], angleRad: number): Float64Array {
  const norm = Math.hypot(axis[0], axis[1], axis[2]);
  const x = axis[0] / norm;
  const y = axis[1] / norm;
  const z = axis[2] / norm;
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  const omc = 1 - c;
  return new Float64Array([
    c + x * x * omc, x * y * omc - z * s, x * z * omc + y * s,
    y * x * omc + z * s, c + y * y * omc, y * z * omc - x * s,
    z * x * omc - y * s, z * y * omc + x * s, c + z * z * omc,
  ]);
}

function transformCoords(
  coords: Float32Array,
  R: Float64Array,
  t: Float64Array,
): Float32Array {
  const out = new Float32Array(coords.length);
  for (let i = 0; i < coords.length; i += 3) {
    const x = coords[i];
    const y = coords[i + 1];
    const z = coords[i + 2];
    out[i + 0] = R[0] * x + R[1] * y + R[2] * z + t[0];
    out[i + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
    out[i + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
  }
  return out;
}

describe('protein-ops/kabsch', () => {
  const samplePdb = generateSampleHelixBundlePdb();
  const sampleParsed = parsePdbBackbone(samplePdb);
  const sampleCa = sampleParsed.caCoords;

  it('recovers exact 3D rotation + translation to < 1e-5 A', () => {
    const Rtrue = axisAngleRotation([0.37, -0.81, 0.45], 1.73);
    const tTrue = new Float64Array([42.5, -17.3, 108.9]);
    expect(det3(Rtrue)).toBeCloseTo(1.0, 10);

    // target = Rtrue * mobile + tTrue
    const target = transformCoords(sampleCa, Rtrue, tTrue);
    const res = kabschAlign(sampleCa, target);

    expect(res.rmsd).toBeLessThan(1e-5);
    expect(res.tmScore).toBeGreaterThan(0.99999);
    expect(det3(res.rotation)).toBeCloseTo(1.0, 6);

    for (let i = 0; i < 9; i++) {
      expect(res.rotation[i]).toBeCloseTo(Rtrue[i], 5);
    }
    for (let i = 0; i < 3; i++) {
      expect(res.translation[i]).toBeCloseTo(tTrue[i], 4);
    }
    for (let i = 0; i < target.length; i++) {
      expect(Math.abs(res.aligned[i] - target[i])).toBeLessThan(1e-4);
    }

    // Verify column-major 4x4 transform4x4 (Mol* Mat4 convention) matches R * p + t
    const M = res.transform4x4;
    expect(M.length).toBe(16);
    for (let p = 0; p < 5; p++) {
      const x = sampleCa[p * 3 + 0];
      const y = sampleCa[p * 3 + 1];
      const z = sampleCa[p * 3 + 2];
      const mx = M[0] * x + M[4] * y + M[8] * z + M[12];
      const my = M[1] * x + M[5] * y + M[9] * z + M[13];
      const mz = M[2] * x + M[6] * y + M[10] * z + M[14];
      expect(mx).toBeCloseTo(target[p * 3 + 0], 4);
      expect(my).toBeCloseTo(target[p * 3 + 1], 4);
      expect(mz).toBeCloseTo(target[p * 3 + 2], 4);
    }
  });

  it('strictly rejects reflections (det(R) === +1) when aligning chiral mirror images', () => {
    const mirrored = new Float32Array(sampleCa.length);
    for (let i = 0; i < sampleCa.length; i += 3) {
      mirrored[i + 0] = sampleCa[i + 0];
      mirrored[i + 1] = sampleCa[i + 1];
      mirrored[i + 2] = -sampleCa[i + 2]; // improper reflection
    }

    const res = kabschAlign(mirrored, sampleCa);
    expect(det3(res.rotation)).toBeCloseTo(1.0, 6);

    // R^T * R must be identity
    const R = res.rotation;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const dot = R[0 * 3 + i] * R[0 * 3 + j] + R[1 * 3 + i] * R[1 * 3 + j] + R[2 * 3 + i] * R[2 * 3 + j];
        expect(dot).toBeCloseTo(i === j ? 1 : 0, 6);
      }
    }
    // Chiral 3-helix bundle cannot superimpose onto its mirror image with zero RMSD
    expect(res.rmsd).toBeGreaterThan(1.0);
  });

  it('handles < 3 points safely by returning NaN metrics and identity transform (including non-multiple-of-3 buffers)', () => {
    const tiny = new Float32Array([1, 2, 3, 4, 5, 6]);
    const res = kabschAlign(tiny, tiny);
    expect(Number.isNaN(res.rmsd)).toBe(true);
    expect(Number.isNaN(res.tmScore)).toBe(true);
    expect(det3(res.rotation)).toBe(1);

    // 8 floats = floor(8 / 3) = 2 points (< 3)
    const oddLen = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const resOdd = kabschAlign(oddLen, oddLen);
    expect(Number.isNaN(resOdd.rmsd)).toBe(true);
    expect(Number.isNaN(resOdd.tmScore)).toBe(true);
  });

  it('satisfies scRMSD / scTM bounds, symmetry, rigid-motion invariance, and perturbation monotonicity', () => {
    const L = sampleParsed.nResidues;
    const perturb = (amp: number) => {
      const out = new Float32Array(sampleCa.length);
      for (let i = 0; i < L; i++) {
        // Deterministic pseudo-perturbation
        out[i * 3 + 0] = sampleCa[i * 3 + 0] + amp * Math.sin(i * 1.7 + 0.3);
        out[i * 3 + 1] = sampleCa[i * 3 + 1] + amp * Math.cos(i * 2.3 + 1.1);
        out[i * 3 + 2] = sampleCa[i * 3 + 2] + amp * Math.sin(i * 0.9 - 0.7);
      }
      return out;
    };

    const pSmall = perturb(0.5);
    const pMed = perturb(2.5);
    const pLarge = perturb(8.0);

    const rSmall = kabschAlign(pSmall, sampleCa);
    const rMed = kabschAlign(pMed, sampleCa);
    const rLarge = kabschAlign(pLarge, sampleCa);

    // Monotonicity
    expect(rSmall.rmsd).toBeLessThan(rMed.rmsd);
    expect(rMed.rmsd).toBeLessThan(rLarge.rmsd);
    expect(rSmall.tmScore).toBeGreaterThan(rMed.tmScore);
    expect(rMed.tmScore).toBeGreaterThan(rLarge.tmScore);

    // Bounds
    for (const r of [rSmall, rMed, rLarge]) {
      expect(r.tmScore).toBeGreaterThan(0);
      expect(r.tmScore).toBeLessThanOrEqual(1);
      expect(r.rmsd).toBeGreaterThan(0);
    }

    // Symmetry: rmsd(A, B) == rmsd(B, A)
    const rMedRev = kabschAlign(sampleCa, pMed);
    expect(rMedRev.rmsd).toBeCloseTo(rMed.rmsd, 4);
    expect(rMedRev.tmScore).toBeCloseTo(rMed.tmScore, 4);

    // Rigid transform invariance
    const R = axisAngleRotation([-0.5, 0.2, 0.85], 2.41);
    const t = new Float64Array([-31.4, 88.2, -12.7]);
    const pMedMoved = transformCoords(pMed, R, t);
    const rMoved = kabschAlign(pMedMoved, sampleCa);
    expect(rMoved.rmsd).toBeCloseTo(rMed.rmsd, 4);
    expect(rMoved.tmScore).toBeCloseTo(rMed.tmScore, 4);
  });

  it('applyTransformToPdb accurately transforms ATOM & HETATM coordinates and preserves headers', () => {
    const R = axisAngleRotation([0.2, 0.9, -0.35], 1.15);
    const t = new Float64Array([15.25, -8.75, 22.5]);
    const pdbWithHet =
      'HEADER    TEST STRUCTURE\n' +
      samplePdb.replace(
        'END\n',
        'HETATM 9999  O   HOH B   1       1.000   2.000   3.000  1.00  0.00           O  \nEND\n',
      );

    const movedPdb = applyTransformToPdb(pdbWithHet, R, t);
    expect(movedPdb.startsWith('HEADER    TEST STRUCTURE\n')).toBe(true);

    // Verify HETATM line was also transformed
    const hetLine = movedPdb.split('\n').find((l) => l.startsWith('HETATM'))!;
    expect(hetLine).toBeDefined();
    const hx = parseFloat(hetLine.slice(30, 38));
    const hy = parseFloat(hetLine.slice(38, 46));
    const hz = parseFloat(hetLine.slice(46, 54));
    expect(hx).toBeCloseTo(R[0] * 1 + R[1] * 2 + R[2] * 3 + t[0], 2);
    expect(hy).toBeCloseTo(R[3] * 1 + R[4] * 2 + R[5] * 3 + t[1], 2);
    expect(hz).toBeCloseTo(R[6] * 1 + R[7] * 2 + R[8] * 3 + t[2], 2);

    // Re-align moved PDB back onto original via kabschAlign + applyTransformToPdb
    const movedParsed = parsePdbBackbone(movedPdb);
    const back = kabschAlign(movedParsed.caCoords, sampleCa);
    const superposedPdb = applyTransformToPdb(movedPdb, back.rotation, back.translation);
    const superposedParsed = parsePdbBackbone(superposedPdb);

    for (let i = 0; i < sampleCa.length; i++) {
      expect(Math.abs(superposedParsed.caCoords[i] - sampleCa[i])).toBeLessThan(2e-3);
    }
  });

  it('computeGeometryStats accurately classifies alpha-helix, beta-strand, Rg, breaks, and clashes', () => {
    // Empty input
    const emptyStats = computeGeometryStats(new Float32Array(0));
    expect(emptyStats).toEqual({
      length: 0,
      rg: 0,
      meanCaDist: 0,
      chainBreaks: 0,
      clashes: 0,
      helixPct: 0,
      sheetPct: 0,
      coilPct: 100,
    });

    // 1. Ideal 24-aa alpha-helix (radius 2.3 A, rise 1.5 A, twist 100 deg)
    const Lh = 24;
    const helix = new Float32Array(Lh * 3);
    const twist = (100 * Math.PI) / 180;
    for (let i = 0; i < Lh; i++) {
      helix[i * 3 + 0] = 2.3 * Math.cos(i * twist);
      helix[i * 3 + 1] = 2.3 * Math.sin(i * twist);
      helix[i * 3 + 2] = i * 1.5;
    }
    const hStats = computeGeometryStats(helix);
    expect(hStats.length).toBe(Lh);
    expect(hStats.chainBreaks).toBe(0);
    expect(hStats.clashes).toBe(0);
    expect(hStats.meanCaDist).toBeCloseTo(3.8, 1);
    expect(hStats.helixPct).toBeGreaterThan(75);
    expect(hStats.sheetPct).toBe(0);
    expect(hStats.helixPct + hStats.sheetPct + hStats.coilPct).toBeCloseTo(100, 5);

    // 2. Ideal 16-aa extended beta-strand (zigzag along X: stepX=3.35 A, altY=+-0.9 A => CA-CA ~ 3.80 A)
    const Ls = 16;
    const strand = new Float32Array(Ls * 3);
    for (let i = 0; i < Ls; i++) {
      strand[i * 3 + 0] = i * 3.35;
      strand[i * 3 + 1] = (i % 2 === 0 ? 1 : -1) * 0.9;
      strand[i * 3 + 2] = 0;
    }
    const sStats = computeGeometryStats(strand);
    expect(sStats.chainBreaks).toBe(0);
    expect(sStats.clashes).toBe(0);
    expect(sStats.meanCaDist).toBeCloseTo(3.8, 1);
    expect(sStats.sheetPct).toBeGreaterThan(75);
    expect(sStats.helixPct).toBe(0);

    // 3. Analytical Rg check on symmetric 4-point cube corners around origin (all at distance sqrt(3) * 2)
    const box = new Float32Array([
      -2, -2, -2,
      2, 2, -2,
      2, -2, 2,
      -2, 2, 2,
    ]);
    const bStats = computeGeometryStats(box);
    expect(bStats.rg).toBeCloseTo(Math.sqrt(12), 4);

    // 4. Explicit chain break (> 4.5 A) and steric clash (non-adjacent < 3.0 A)
    const broken = new Float32Array([
      0, 0, 0,
      3.8, 0, 0,
      12.0, 0, 0, // break: 8.2 A from res 1
      1.0, 0, 0, // break from res 2 AND clashes (< 3.0 A) with res 0 (dist 1.0) and res 1 (dist 2.8)
    ]);
    const brStats = computeGeometryStats(broken);
    expect(brStats.chainBreaks).toBe(2);
    expect(brStats.clashes).toBe(2);
  });

  it('matches exhaustive clashes for shuffled coordinates, cutoff ties and missing observations', () => {
    const coords = Float32Array.from({ length: 192 }, (_, k) => ((k * 79) % 97 - 48) / 6);
    coords.set([3, 0, 0, 0, 0, 0, 2.99999, 0, 0, NaN, 0, 0, Infinity, 0, 0, 0, NaN, 0, 0, 0, -Infinity]);
    const n = coords.length / 3;
    const chainIds = Array.from({ length: n }, (_, i) => i < 30 ? 'A' : 'B');
    const resNums = Array.from({ length: n }, (_, i) => i < 15 ? i : i + 3);
    let expected = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (j === i + 1 && chainIds[i] === chainIds[j] && resNums[j] === resNums[i] + 1) continue;
        if (Math.hypot(coords[i*3] - coords[j*3], coords[i*3+1] - coords[j*3+1],
          coords[i*3+2] - coords[j*3+2]) < 3) expected++;
      }
    }
    expect(computeGeometryStats(coords, chainIds, resNums).clashes).toBe(expected);
  });

  it('computeGeometryStats with chainIds ignores inter-chain jumps while catching intra-chain breaks & inter-chain clashes', () => {
    // Chain A (3 residues spaced 3.8 A) + Chain B (3 residues spaced 3.8 A, offset by +50 A in Z)
    const twoChains = new Float32Array([
      0, 0, 0,
      3.8, 0, 0,
      7.6, 0, 0,
      0, 0, 50,
      3.8, 0, 50,
      7.6, 0, 50,
    ]);
    const chainIds = ['A', 'A', 'A', 'B', 'B', 'B'];

    const multi = computeGeometryStats(twoChains, chainIds);
    expect(multi.chainBreaks).toBe(0);
    expect(multi.clashes).toBe(0);
    expect(multi.meanCaDist).toBeCloseTo(3.8, 4);

    // Without chainIds, the 50 A jump looks like a chain break
    expect(computeGeometryStats(twoChains).chainBreaks).toBe(1);

    // Intra-chain break inside chain A is still detected when chainIds are passed
    const intraBreak = new Float32Array(twoChains);
    intraBreak[2 * 3 + 0] = 20.0; // 16.2 A jump inside chain A
    expect(computeGeometryStats(intraBreak, chainIds).chainBreaks).toBe(1);

    // Inter-chain clash (< 3.0 A between last residue of chain A and first residue of chain B) is detected
    const interClash = new Float32Array([
      0, 0, 0,
      3.8, 0, 0,
      7.6, 0, 0,
      7.6, 1.5, 0, // 1.5 A from A:3 -> inter-chain clash!
      11.4, 1.5, 0,
      15.2, 1.5, 0,
    ]);
    const clashStats = computeGeometryStats(interClash, chainIds);
    expect(clashStats.chainBreaks).toBe(0);
    expect(clashStats.clashes).toBeGreaterThanOrEqual(1);
  });

  it('computeGeometryStats with resNums ignores motif/missing-density gaps for breaks, meanCaDist, and SS% while still catching true breaks & clashes', () => {
    // Two 10-aa ideal alpha-helices on chain A with a residue-numbering gap (16..25, 52..61) separated by +30 A
    const twist = (100 * Math.PI) / 180;
    const coords = new Float32Array(20 * 3);
    const chainIds: string[] = [];
    const resNums: number[] = [];
    for (let i = 0; i < 10; i++) {
      coords[i * 3 + 0] = 2.3 * Math.cos(i * twist);
      coords[i * 3 + 1] = 2.3 * Math.sin(i * twist);
      coords[i * 3 + 2] = i * 1.5;
      chainIds.push('A');
      resNums.push(16 + i);
    }
    for (let i = 0; i < 10; i++) {
      const idx = 10 + i;
      coords[idx * 3 + 0] = 30 + 2.3 * Math.cos(i * twist);
      coords[idx * 3 + 1] = 2.3 * Math.sin(i * twist);
      coords[idx * 3 + 2] = i * 1.5;
      chainIds.push('A');
      resNums.push(52 + i);
    }

    const withResNums = computeGeometryStats(coords, chainIds, resNums);
    const withoutResNums = computeGeometryStats(coords, chainIds);
    expect(withResNums.chainBreaks).toBe(0);
    expect(withResNums.meanCaDist).toBeCloseTo(3.8, 1);
    expect(withoutResNums.chainBreaks).toBe(1);
    expect(withoutResNums.meanCaDist).toBeGreaterThan(4.5);

    // True intra-segment covalent break (consecutive resNums) is still detected
    const brokenSeg = new Float32Array(coords);
    brokenSeg[4 * 3 + 2] += 12.0; // break between res 19 and 20 (and 20->21)
    expect(computeGeometryStats(brokenSeg, chainIds, resNums).chainBreaks).toBe(2);

    // Steric clash (< 3.0 A) across a residue-number gap IS detected as a clash
    const gapClash = new Float32Array(coords);
    gapClash[10 * 3 + 0] = coords[9 * 3 + 0] + 1.5;
    gapClash[10 * 3 + 1] = coords[9 * 3 + 1];
    gapClash[10 * 3 + 2] = coords[9 * 3 + 2];
    expect(computeGeometryStats(gapClash, chainIds, resNums).clashes).toBeGreaterThanOrEqual(1);
  });

  it('applyTransformToPdb guards 8-char coordinate column width on moderate and extreme overflows', () => {
    const idR = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    // Moderate overflow beyond [-999.999, 9999.999]
    const modT = new Float64Array([12345.678, -1500.25, 99999.12]);
    const modPdb = applyTransformToPdb(samplePdb, idR, modT);
    const firstAtomMod = modPdb.split('\n').find((l) => l.startsWith('ATOM  '))!;
    expect(firstAtomMod.slice(54, 66)).toBe('  1.00  0.00'); // columns 55-66 preserved intact!
    expect(firstAtomMod.slice(30, 38).length).toBe(8);
    expect(firstAtomMod.slice(38, 46).length).toBe(8);
    expect(firstAtomMod.slice(46, 54).length).toBe(8);
    expect(parseFloat(firstAtomMod.slice(30, 38))).toBeCloseTo(
      parseFloat(samplePdb.split('\n')[0].slice(30, 38)) + 12345.678,
      1,
    );

    // Extreme overflow (> 1e8): re-centered around centroid while preserving internal geometry & column alignment
    const extT = new Float64Array([1e9, -2e9, 5e8]);
    const extPdb = applyTransformToPdb(samplePdb, idR, extT);
    const firstAtomExt = extPdb.split('\n').find((l) => l.startsWith('ATOM  '))!;
    expect(firstAtomExt.slice(54, 66)).toBe('  1.00  0.00');
    const reparsedExt = parsePdbBackbone(extPdb);
    expect(reparsedExt.nResidues).toBe(sampleParsed.nResidues);
    expect(kabschAlign(reparsedExt.caCoords, sampleCa).rmsd).toBeLessThan(2e-3);
  });

  it('kabschAlignSubstructure aligns sparse catalytic motifs (1,1,1,1 & 1,1,3), symmetric multi-island motifs, permuted motifs, and knownIndices without collapsing discontinuous islands', () => {
    const Rtrue = axisAngleRotation([0.4, -0.7, 0.6], 1.9);
    const tTrue = new Float64Array([-18.5, 31.2, 44.8]);
    const extractSubset = (src: Float32Array, indices: number[]): Float32Array => {
      const out = new Float32Array(indices.length * 3);
      for (let i = 0; i < indices.length; i++) {
        out[i * 3 + 0] = src[indices[i] * 3 + 0];
        out[i * 3 + 1] = src[indices[i] * 3 + 1];
        out[i * 3 + 2] = src[indices[i] * 3 + 2];
      }
      return out;
    };

    // 1. Sparse 4-residue catalytic motif (1,1,1,1 like 1CTT/1OS7/4XOJ) at discontinuous non-adjacent indices
    const sparseIdx = [3, 17, 31, 47];
    const sparseMob = transformCoords(extractSubset(sampleCa, sparseIdx), Rtrue, tTrue);
    const sparseRes = kabschAlignSubstructure(sparseMob, sampleCa, [102, 104, 129, 202]);
    expect(sparseRes.alignedResidues).toBe(4);
    expect(sparseRes.rmsd).toBeLessThan(1e-3);

    // 2. Permuted sparse motif (reverse order) still aligns via distance-matrix rigid-body consensus
    const permIdx = [47, 17, 3, 31];
    const permMob = transformCoords(extractSubset(sampleCa, permIdx), Rtrue, tTrue);
    const permRes = kabschAlignSubstructure(permMob, sampleCa, [202, 130, 101, 132]);
    expect(permRes.alignedResidues).toBe(4);
    expect(permRes.rmsd).toBeLessThan(1e-3);

    // 3. Mixed sparse motif (1, 1, 3 — previously collapsed into 1 contiguous 5-aa window!)
    const mixedIdx = [4, 22, 38, 39, 40];
    const mixedMob = transformCoords(extractSubset(sampleCa, mixedIdx), Rtrue, tTrue);
    const mixedRes = kabschAlignSubstructure(mixedMob, sampleCa, [10, 50, 100, 101, 102]);
    expect(mixedRes.alignedResidues).toBe(5);
    expect(mixedRes.rmsd).toBeLessThan(1e-3);

    // 4. Symmetric multi-island motif (4 identical 6-aa helices embedded at distinct 3D positions like 1BCF)
    const twist = (100 * Math.PI) / 180;
    const makeHelixPts = (cx: number, cy: number, cz: number, len: number): number[] => {
      const pts: number[] = [];
      for (let i = 0; i < len; i++) {
        pts.push(cx + 2.3 * Math.cos(i * twist), cy + 2.3 * Math.sin(i * twist), cz + i * 1.5);
      }
      return pts;
    };
    const longPts = [
      ...makeHelixPts(0, 0, 0, 10),
      ...makeHelixPts(12, 0, 3, 10),
      ...makeHelixPts(12, 14, -2, 10),
      ...makeHelixPts(-2, 15, 5, 10),
    ];
    const longSym = new Float32Array(longPts);
    // Extract 4 symmetric 6-aa islands in permuted order (helix 2, helix 0, helix 3, helix 1)
    const symIndices = [
      22, 23, 24, 25, 26, 27,
      2, 3, 4, 5, 6, 7,
      32, 33, 34, 35, 36, 37,
      12, 13, 14, 15, 16, 17,
    ];
    const symResNums = [
      18, 19, 20, 21, 22, 23,
      47, 48, 49, 50, 51, 52,
      92, 93, 94, 95, 96, 97,
      123, 124, 125, 126, 127, 128,
    ];
    const symMob = transformCoords(extractSubset(longSym, symIndices), Rtrue, tTrue);
    const symRes = kabschAlignSubstructure(symMob, longSym, symResNums);
    expect(symRes.alignedResidues).toBe(24);
    expect(symRes.rmsd).toBeLessThan(1e-3);

    // 5. Known motif indices option aligns directly
    const directRes = kabschAlignSubstructure(sparseMob, sampleCa, null, {
      knownIndices: sparseIdx,
    });
    expect(directRes.alignedResidues).toBe(4);
    expect(directRes.rmsd).toBeLessThan(1e-4);

    // 6. shortChainIds splits discontinuous chains even if resNums are consecutive (e.g. 1OS7 B/C)
    const multiChainMob = transformCoords(extractSubset(sampleCa, [5, 6, 7, 35, 36, 37]), Rtrue, tTrue);
    const multiChainRes = kabschAlignSubstructure(multiChainMob, sampleCa, [1, 2, 3, 4, 5, 6], {
      shortChainIds: ['B', 'B', 'B', 'C', 'C', 'C'],
    });
    expect(multiChainRes.alignedResidues).toBe(6);
    expect(multiChainRes.rmsd).toBeLessThan(1e-3);
  });
});
