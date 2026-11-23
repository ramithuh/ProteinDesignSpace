import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  buildGapAwareResidueIndices,
  compileMpnnInputSpec,
  parseFixedPositionsAndChains,
} from '$lib/engine/mpnn-spec';
import { computeStructurePayloadDigest } from '$lib/engine/collections';

type MockStructure = Parameters<typeof buildGapAwareResidueIndices>[0];

function makeMockStructure(opts: {
  chainIds: string[];
  residueNumbers: number[];
  insertionCodes?: string[];
  sequence: string;
}): MockStructure {
  const n = opts.chainIds.length;
  return {
    length: n,
    atomMode: 'backbone4',
    coordinates: new Float32Array(n * 4 * 3),
    residueIndices: new Int32Array(Array.from({ length: n }, (_, i) => i)),
    chainEncoding: new Int32Array(n),
    chainIds: opts.chainIds,
    residueNumbers: opts.residueNumbers,
    insertionCodes: opts.insertionCodes ?? Array(n).fill(''),
    sequence: opts.sequence,
    mask: new Float32Array(Array(n).fill(1)),
  } as unknown as MockStructure;
}

describe('Astra 6.2 Audit Wave 2: Inverse Folding Correctness (MPNN, LigandMPNN, LASErMPNN, AntiFold)', () => {
  it('mpnn-spec resolves case-sensitive chains (a10 vs A10) without uppercasing into wrong chain', () => {
    const ordered = [
      { chain: 'A', number: 10 },
      { chain: 'a', number: 10 },
      { chain: 'a', number: 11 },
    ];
    const parsed = parseFixedPositionsAndChains('a10', 'A', ordered);
    expect(parsed.fixedPositions).toEqual([{ chain: 'a', number: 10 }]);

    const rangeParsed = parseFixedPositionsAndChains('a10-11', 'A', ordered);
    expect(rangeParsed.fixedPositions).toEqual([
      { chain: 'a', number: 10 },
      { chain: 'a', number: 11 },
    ]);
  });

  it('mpnn-spec buildGapAwareResidueIndices preserves author numbering gaps after insertion codes (10,10A,11,15 -> 0,1,2,6)', () => {
    const indices = buildGapAwareResidueIndices(
      makeMockStructure({
        chainIds: ['A', 'A', 'A', 'A'],
        residueNumbers: [10, 10, 11, 15],
        insertionCodes: ['', 'A', '', ''],
        sequence: 'AAAA',
      }),
    );
    expect(Array.from(indices)).toEqual([0, 1, 2, 6]);
  });

  it('mpnn-spec maps encounter-ordered fixedResidueIndices via sourceResidues into sorted prepared structure', () => {
    // Source encounter order: A20 (index 1), A10 (index 2), A30 (index 3)
    // Prepared (sorted) order: A10 (idx 0), A20 (idx 1), A30 (idx 2)
    const compiled = compileMpnnInputSpec(
      {
        pdbText: '',
        sourceResidues: [
          { authChainId: 'A', projectedChainId: 'A', resNum: 20, insCode: '', resName: 'ALA' },
          { authChainId: 'A', projectedChainId: 'A', resNum: 10, insCode: '', resName: 'GLY' },
          { authChainId: 'A', projectedChainId: 'A', resNum: 30, insCode: '', resName: 'VAL' },
        ],
        fixedResidueIndices: [1], // 1-based -> A20!
      } as Parameters<typeof compileMpnnInputSpec>[0],
      {},
      makeMockStructure({
        chainIds: ['A', 'A', 'A'],
        residueNumbers: [10, 20, 30],
        sequence: 'GAV',
      }),
    );
    // Must lock prepared index 1 (A20), NOT prepared index 0 (A10)!
    expect(compiled.fixedPositions).toEqual([1]);
    expect(compiled.residues[0].willDesign).toBe(true); // A10 designable
    expect(compiled.residues[1].willDesign).toBe(false); // A20 locked
  });

  it('mpnn-spec subtracts explicit fixedPositions chain from designChains instead of redesigning a locked chain', () => {
    const compiled = compileMpnnInputSpec(
      { pdbText: '' },
      { designChains: 'A,B', fixedPositions: 'A' },
      makeMockStructure({
        chainIds: ['A', 'B'],
        residueNumbers: [1, 1],
        sequence: 'AG',
      }),
    );
    expect(compiled.designChains).toEqual(['B']);
    expect(compiled.residues[0].willDesign).toBe(false);
    expect(compiled.residues[1].willDesign).toBe(true);
  });

  it('LigandMPNN & AntiFold parsers reject blank PDB coordinates instead of coercing them to 0.0', async () => {
    const blankCoordPdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N',
      'ATOM      2  CA  ALA A   1               2.000   3.000  1.00 20.00           C',
      'ATOM      3  C   ALA A   1       2.000   1.420   0.000  1.00 20.00           C',
      'ATOM      4  O   ALA A   1       1.250   2.400   0.000  1.00 20.00           O',
      'END',
    ].join('\n');

    const ligUrl = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/ligandmpnn/shared/ligandmpnn/parser.js'),
    ).href;
    const ligParser = (await import(/* @vite-ignore */ ligUrl)) as {
      parseStructure: (text: string) => unknown;
    };
    expect(() => ligParser.parseStructure(blankCoordPdb)).toThrow(/Non-finite/i);

    const afUrl = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/antifold-esmif1/shared/structure.js'),
    ).href;
    const afParser = (await import(/* @vite-ignore */ afUrl)) as {
      parseStructure: (text: string) => unknown;
    };
    expect(() => afParser.parseStructure(blankCoordPdb)).toThrow(/Invalid coordinate/i);
  });

  it('LigandMPNN structured constraint tokens distinguish A:12 from A1:2', async () => {
    const featUrl = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/ligandmpnn/shared/ligandmpnn/features.js'),
    ).href;
    const featMod = (await import(/* @vite-ignore */ featUrl)) as {
      constraints: (
        p: { L: number; S: Int32Array; residues: Array<{ chain: string; number: number; insertionCode: string; token: string }> },
        opts: { fixedPositions?: string[] },
      ) => { chain_mask: Float32Array };
    };
    const res = featMod.constraints(
      {
        L: 2,
        S: new Int32Array([0, 1]),
        residues: [
          { chain: 'A', number: 12, insertionCode: '', token: 'A12' },
          { chain: 'A1', number: 2, insertionCode: '', token: 'A12' },
        ],
      },
      { fixedPositions: ['A:12'] },
    );
    expect(Array.from(res.chain_mask)).toEqual([0, 1]);
  });

  it('computeStructurePayloadDigest hashes cifText when present so CIF-consuming workers never collide', () => {
    const base = {
      id: 'struct-1',
      sequence: 'ALA',
      length: 3,
      pdbText: 'END\n',
      caCoords: new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]),
    };
    const d1 = computeStructurePayloadDigest({ ...base, cifText: 'data_1\n_a 1\n' });
    const d2 = computeStructurePayloadDigest({ ...base, cifText: 'data_2\n_a 2\n' });
    expect(d1).not.toBe(d2);
  });
});
