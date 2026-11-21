import { describe, expect, it } from 'vitest';
import { computeIpsae, computeIpsaeD0, sliceProteinPae } from '$lib/protein-ops/ipsae';

describe('Dunbrack-lab ipSAE interface scoring', () => {
  it('computes exact Dunbrack (2025) d0(n0) and ipSAE formula for both n0 <= 27 and n0 > 27', () => {
    // 1. Small interface (n0 = 4 <= 27): d0 = max(1.0, 1.24 * (27 - 15)^(1/3) - 1.8)
    const expectedD0Small = Math.max(1.0, 1.24 * Math.cbrt(27 - 15) - 1.8);
    expect(computeIpsaeD0(4)).toBeCloseTo(expectedD0Small, 10);
    expect(computeIpsaeD0(27)).toBeCloseTo(expectedD0Small, 10);

    // Build 2-chain complex: Chain A (3 res), Chain B (4 res), L = 7
    const chainIds = ['A', 'A', 'A', 'B', 'B', 'B', 'B'];
    const L = chainIds.length;
    const pae = new Float32Array(L * L).fill(20.0);
    // Row 0 (A0 -> B): PAE values [1.0, 2.0, 3.0, 4.0] (all < 10 -> n0 = 4)
    pae[0 * L + 3] = 1.0;
    pae[0 * L + 4] = 2.0;
    pae[0 * L + 5] = 3.0;
    pae[0 * L + 6] = 4.0;
    // Mirror B -> A identically for B0 (row 3 -> A0, A1, A2): [1.0, 2.0, 3.0] (n0 = 3)
    pae[3 * L + 0] = 1.0;
    pae[3 * L + 1] = 2.0;
    pae[3 * L + 2] = 3.0;

    const res = computeIpsae(pae, chainIds, 10);
    expect(res).not.toBeNull();

    const expectedAtoB =
      ([1.0, 2.0, 3.0, 4.0].reduce((s, v) => s + 1 / (1 + (v / expectedD0Small) ** 2), 0)) / 4;
    const expectedBtoA =
      ([1.0, 2.0, 3.0].reduce((s, v) => s + 1 / (1 + (v / expectedD0Small) ** 2), 0)) / 3;

    expect(res!.pairScores['A->B']).toBeCloseTo(expectedAtoB, 6);
    expect(res!.pairScores['B->A']).toBeCloseTo(expectedBtoA, 6);
    expect(res!.ipsaeMax).toBeCloseTo(Math.max(expectedAtoB, expectedBtoA), 6);
    expect(res!.ipsae).toBeCloseTo(res!.ipsaeMax, 10);
    expect(res!.ipsaeMin).toBeCloseTo(Math.min(expectedAtoB, expectedBtoA), 6);
    expect(res!.paeCutoff).toBe(10);
    // Participating residues: A0, A1, A2 (3) + B0, B1, B2, B3 (4) = 7
    expect(res!.interfaceResidues).toBe(7);

    // 2. Large interface (n0 = 40 > 27): d0 = 1.24 * (40 - 15)^(1/3) - 1.8
    const expectedD0Large = Math.max(1.0, 1.24 * Math.cbrt(40 - 15) - 1.8);
    expect(computeIpsaeD0(40)).toBeCloseTo(expectedD0Large, 10);

    const nA = 5;
    const nB = 40;
    const L2 = nA + nB;
    const chains2 = [...Array(nA).fill('A'), ...Array(nB).fill('B')];
    const pae2 = new Float32Array(L2 * L2).fill(25.0);
    for (let j = nA; j < L2; j++) {
      pae2[0 * L2 + j] = 2.5; // A0 -> all 40 B residues at PAE = 2.5 Å
    }
    for (let i = 0; i < nA; i++) {
      pae2[nA * L2 + i] = 2.5; // B0 -> all 5 A residues at PAE = 2.5 Å
    }
    const res2 = computeIpsae(pae2, chains2, 10)!;
    const expectedAtoBLarge = 1 / (1 + (2.5 / expectedD0Large) ** 2);
    const expectedBtoASmall = 1 / (1 + (2.5 / expectedD0Small) ** 2);
    expect(res2.pairScores['A->B']).toBeCloseTo(expectedAtoBLarge, 6);
    expect(res2.pairScores['B->A']).toBeCloseTo(expectedBtoASmall, 6);
    // Larger d0(40) > d0(27) gives higher ipSAE for A->B than B->A at identical 2.5 Å PAE!
    expect(res2.ipsaeMax).toBeCloseTo(expectedAtoBLarge, 6);
    expect(res2.ipsaeMin).toBeCloseTo(expectedBtoASmall, 6);
    expect(res2.ipsaeMax).toBeGreaterThan(res2.ipsaeMin);
  });

  it('returns 0 with 0 interfaceResidues when all cross-chain PAE values are >= cutoff', () => {
    const chainIds = ['A', 'A', 'A', 'B', 'B', 'B'];
    const L = chainIds.length;
    const pae = new Float32Array(L * L);
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        pae[i * L + j] = chainIds[i] === chainIds[j] ? 1.0 : 10.0; // strictly `< cutoff`, so 10.0 is excluded!
      }
    }
    const res = computeIpsae(pae, chainIds, 10);
    expect(res).not.toBeNull();
    expect(res!.ipsae).toBe(0);
    expect(res!.ipsaeMax).toBe(0);
    expect(res!.ipsaeMin).toBe(0);
    expect(res!.interfaceResidues).toBe(0);
  });

  it('captures asymmetric chain-pair directional scores (ipsaeMax vs ipsaeMin) and 3-chain complexes', () => {
    // 3-chain complex: A (3 aa), B (3 aa), C (2 aa)
    // A -> B has tight PAE (0.5 Å), B -> A has moderate PAE (6.0 Å), C is detached (25 Å)
    const chainIds = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'];
    const L = chainIds.length;
    const pae = new Float32Array(L * L).fill(25.0);
    // A0 -> B0, B1 at 0.5 Å
    pae[0 * L + 3] = 0.5;
    pae[0 * L + 4] = 0.5;
    // B0 -> A0, A1 at 6.0 Å
    pae[3 * L + 0] = 6.0;
    pae[3 * L + 1] = 6.0;

    const res = computeIpsae(pae, chainIds, 10)!;
    expect(res.pairScores['A->B']).toBeGreaterThan(0.75);
    expect(res.pairScores['B->A']).toBeGreaterThan(0);
    expect(res.pairScores['B->A']).toBeLessThan(0.1);
    expect(res.pairScores['A->C']).toBe(0);
    expect(res.pairScores['C->A']).toBe(0);
    expect(res.ipsae).toBeCloseTo(res.pairScores['A->B'], 6);
    expect(res.ipsaeMax).toBeCloseTo(res.pairScores['A->B'], 6);
    expect(res.ipsaeMin).toBe(0);
    // Interface residues: A0, A1, B0, B1 = 4 (C0, C1, A2, B2 not participating)
    expect(res.interfaceResidues).toBe(4);
  });

  it('returns null for single-chain, mismatched matrix length, or invalid cutoff, and slices AF3 holo PAE cleanly', () => {
    expect(computeIpsae(new Float32Array(16).fill(1), ['A', 'A', 'A', 'A'])).toBeNull();
    expect(computeIpsae(new Float32Array(10), ['A', 'A', 'B', 'B'])).toBeNull();
    expect(computeIpsae(new Float32Array(16).fill(1), ['A', 'A', 'B', 'B'], 0)).toBeNull();
    expect(computeIpsae(new Float32Array(16).fill(1), ['A', 'A', 'B', 'B'], -5)).toBeNull();

    // AF3 holo PAE: 6 protein residues + 2 ligand tokens -> 8×8 PAE matrix
    const paeDim = 8;
    const proteinLen = 6;
    const fullPae = new Float32Array(paeDim * paeDim).fill(30.0);
    for (let i = 0; i < proteinLen; i++) {
      for (let j = 0; j < proteinLen; j++) {
        fullPae[i * paeDim + j] = i < 3 === j < 3 ? 0.5 : 1.0;
      }
    }
    const sliced = sliceProteinPae(fullPae, proteinLen);
    expect(sliced).not.toBeNull();
    expect(sliced!.length).toBe(proteinLen * proteinLen);
    const ipsae = computeIpsae(sliced!, ['A', 'A', 'A', 'B', 'B', 'B'], 10);
    expect(ipsae).not.toBeNull();
    expect(ipsae!.ipsae).toBeGreaterThan(0.5);
    expect(ipsae!.interfaceResidues).toBe(6);

    // Exact-size matrix returns without copy
    const exact = new Float32Array(16).fill(2);
    expect(sliceProteinPae(exact, 4)).toBe(exact);
    expect(sliceProteinPae(new Float32Array(15), 4)).toBeNull();
    expect(sliceProteinPae(new Float32Array(9), 4)).toBeNull();
  });
});
