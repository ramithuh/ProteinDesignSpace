import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  computeTargetOnlyOffMask,
  parseCandidateProteinChains,
  resolveMsaChainIdsSafely,
} from '$lib/engine/localfold-msa';
import { paintStructureBfactors } from '$lib/protein-ops/bfactor-paint';
import {
  extractHoloReceptorAndLigands,
  parsePocketResidueTokensStrict,
} from '$lib/protein-ops/gnina-prep';
import { computeCandidateIpsae } from '$lib/plugins/localfold-af2/localfold-esmfold2';

describe('Astra 6.2 Audit Wave 1: P0 MSA Leak, LocalFold, ThermoMPNN, ESM-C/Dyna-1, GNINA', () => {
  it('P0: resolveMsaChainIdsSafely uses physical chainIds and fails closed when chainRoles exceed synthetic A,B', () => {
    // Physical chains B (target) and C (binder): with explicit chainIds, target B is searched and binder C is skipped!
    const explicit = resolveMsaChainIdsSafely({
      chainCount: 2,
      candidateChainIds: ['B', 'C'],
      chainRoles: { B: 'target', C: 'binder' },
    });
    expect(explicit.ambiguous).toBe(false);
    expect(explicit.chainIds).toEqual(['B', 'C']);
    expect(
      computeTargetOnlyOffMask(explicit.chainIds, undefined, { B: 'target', C: 'binder' }),
    ).toEqual([false, true]);

    // Without candidateChainIds, physical roles { B: 'target', C: 'binder' } cannot be mapped safely to synthetic ['A','B']!
    // Must flag ambiguous so caller fails CLOSED (never submits binder C to MMseqs2!).
    const missingIds = resolveMsaChainIdsSafely({
      chainCount: 2,
      candidateChainIds: undefined,
      chainRoles: { B: 'target', C: 'binder' },
    });
    expect(missingIds.ambiguous).toBe(true);
    expect(missingIds.chainIds).toEqual([]);
  });

  it('LocalFold parseCandidateProteinChains preserves inline ":" chain separators even when chainLengths is omitted', () => {
    const res = parseCandidateProteinChains('AAAAA:CCCCC', undefined);
    expect(res.cleanSeq).toBe('AAAAACCCCC');
    expect(res.chainLengths).toEqual([5, 5]);
    expect(res.chainSeqs).toEqual(['AAAAA', 'CCCCC']);
  });

  it('LocalFold computeCandidateIpsae distinguishes 3-entity holo interfaces (no LIG pooling)', () => {
    const dim = 7;
    const pae = new Float32Array(dim * dim).fill(25.0);
    for (let i = 0; i < dim; i++) pae[i * dim + i] = 0.5;
    for (let i = 0; i < 3; i++) {
      for (let j = 3; j < 5; j++) {
        pae[i * dim + j] = 1.0;
        pae[j * dim + i] = 1.0;
      }
    }
    const distinct = computeCandidateIpsae(
      pae,
      [3, 2, 2],
      ['A', 'A', 'A', 'DNA1', 'DNA1', 'LIG1', 'LIG1'],
    );
    const pooled = computeCandidateIpsae(pae, [3, 4], ['A', 'A', 'A', 'LIG', 'LIG', 'LIG', 'LIG']);
    expect(distinct).not.toBeNull();
    expect(pooled).not.toBeNull();
    expect(distinct!.ipsaeMin).toBeLessThan(0.05);
    expect(pooled!.ipsaeMin).toBeGreaterThan(0.2);
  });

  it('ThermoMPNN parser stops on ENDMDL, selects a consistent altloc, and normalizes blank chain IDs to A', async () => {
    const modUrl = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/thermompnn/shared/structure.js'),
    ).href;
    const thermoMod = (await import(/* @vite-ignore */ modUrl)) as {
      parseStructure: (
        text: string,
        opts?: { format?: string },
      ) => { residues: Array<{ chain: string }>; X: Float32Array };
    };
    const pdb = [
      'MODEL        1',
      'ATOM      1  N   ALA     1       0.000   0.000   0.000  1.00 87.00           N',
      'ATOM      2  CA AALA     1       1.458   0.000   0.000  0.50 87.00           C',
      'ATOM      3  CA BALA     1      99.000   0.000   0.000  0.50 87.00           C',
      'ATOM      4  C   ALA     1       2.000   1.420   0.000  1.00 87.00           C',
      'ATOM      5  O   ALA     1       1.250   2.400   0.000  1.00 87.00           O',
      'ENDMDL',
      'MODEL        2',
      'ATOM      6  N   ALA     1      50.000   0.000   0.000  1.00 10.00           N',
      'ATOM      7  CA  ALA     1      51.458   0.000   0.000  1.00 10.00           C',
      'ENDMDL',
    ].join('\n');
    const parsed = thermoMod.parseStructure(pdb, { format: 'pdb' });
    expect(parsed.residues).toHaveLength(1);
    expect(parsed.residues[0].chain).toBe('A');
    // X has layout [L, 4, 3] for [N, CA, C, O] -> CA x-coordinate is X[3]
    expect(parsed.X[3]).toBeCloseTo(1.458, 3);

    // Painting with missingValue: 0 overwrites unscored crystal B-factor 87.00 instead of leaving 87.00 under ΔΔG label
    const painted = paintStructureBfactors(pdb, [Number.NaN], { missingValue: 0 });
    expect(painted).not.toContain('87.00');
    expect(painted).toContain('  0.00');
  });

  it('GNINA CONECT preserves double-bond multiplicity (1.34 Å C=C) and strict pocket parser is case-sensitive + reports unmatched tokens', () => {
    const holoPdb = [
      'ATOM      1  N   ALA A  45       0.000   0.000   0.000  1.00 20.00           N',
      'ATOM      2  CA  ALA A  45       1.458   0.000   0.000  1.00 20.00           C',
      'ATOM      3  C   ALA A  45       2.000   1.420   0.000  1.00 20.00           C',
      'ATOM      4  O   ALA A  45       1.250   2.400   0.000  1.00 20.00           O',
      'HETATM   10  C1  ETH L   1      10.000   0.000   0.000  1.00  0.00           C',
      'HETATM   11  C2  ETH L   1      11.340   0.000   0.000  1.00  0.00           C',
      'HETATM   12  C3  ETH L   1      12.840   0.000   0.000  1.00  0.00           C',
      'HETATM   13  C4  ETH L   1      14.340   0.000   0.000  1.00  0.00           C',
      'HETATM   14  C5  ETH L   1      15.840   0.000   0.000  1.00  0.00           C',
      'CONECT   10   11   11',
      'CONECT   11   10   10',
      'END',
    ].join('\n');
    const prep = extractHoloReceptorAndLigands(holoPdb, { ligandCode: 'ETH' });
    expect(prep.selectedLigandSdf).toBeTruthy();
    // Bond line in V2000 SDF: "  1  2  2  0  0  0  0" (bond order 2!)
    expect(prep.selectedLigandSdf!).toMatch(/\s+1\s+2\s+2\s+0/);

    const recPdb = [
      'ATOM      1  CA  ALA A  45       1.000   2.000   3.000  1.00 20.00           C',
      'ATOM      2  CA  GLY a  45       5.000   6.000   7.000  1.00 20.00           C',
      'END',
    ].join('\n');
    const strictRes = parsePocketResidueTokensStrict('A45, A99', recPdb);
    expect(strictRes.keys).toEqual(['A/45/ALA']);
    expect(strictRes.unmatched).toEqual(['A99']);

    const lowerRes = parsePocketResidueTokensStrict('a45', recPdb);
    expect(lowerRes.keys).toEqual(['a/45/GLY']);
    expect(lowerRes.unmatched).toEqual([]);
  });
});
