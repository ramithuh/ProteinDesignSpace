import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  clearRcsbCache,
  fetchStructureByPdbId,
  isValidPdbId,
  normalizePdbId,
} from '$lib/protein-ops/rcsb';
import { generateSampleHelixBundlePdb } from '$lib/protein-ops/pdb';
import { validatePdbId, validateNodeParams } from '$lib/engine/param-validation';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import { decodeHashToGraph, encodeGraphToHash } from '$lib/engine/share';
import type { StructureArtifact, WorkflowGraph } from '$lib/engine/types';

describe('RCSB PDB ID input', () => {
  afterEach(() => {
    clearRcsbCache();
    vi.restoreAllMocks();
  });

  it('normalizes and validates standard 4-char and extended 12-char PDB IDs', () => {
    expect(normalizePdbId(' 1prw.cif ')).toBe('1PRW');
    expect(normalizePdbId('5o45.pdb')).toBe('5O45');
    expect(isValidPdbId('1PRW')).toBe(true);
    expect(isValidPdbId('5o45')).toBe(true);
    expect(isValidPdbId('PDB_00001PRW')).toBe(true);
    expect(isValidPdbId('ABCD')).toBe(false);
    expect(isValidPdbId('12')).toBe(false);

    expect(validatePdbId('1PRW').status).toBe('ok');
    expect(validatePdbId('').status).toBe('error');
    expect(validatePdbId('BAD!').status).toBe('error');

    expect(
      validateNodeParams({
        id: 'pdb',
        adapterId: 'pdb-input',
        label: 'PDB Input',
        params: { mode: 'pdb-id', pdbId: '1PRW' },
      }),
    ).toEqual([]);
    expect(
      validateNodeParams({
        id: 'pdb',
        adapterId: 'pdb-input',
        label: 'PDB Input',
        params: { mode: 'pdb-id', pdbId: 'ZZZ' },
      }).length,
    ).toBeGreaterThan(0);
  });

  it('fetches from RCSB .cif and falls back to .pdb when .cif returns 404', async () => {
    const samplePdb = generateSampleHelixBundlePdb();
    const calls: string[] = [];
    const mockFetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      calls.push(url);
      if (url.endsWith('.cif')) {
        return new Response('Not Found', { status: 404 });
      }
      return new Response(samplePdb, { status: 200 });
    }) as unknown as typeof fetch;

    const res = await fetchStructureByPdbId('1prw', mockFetch);
    expect(res.pdbId).toBe('1PRW');
    expect(res.filename).toBe('1PRW.pdb');
    expect(res.nResidues).toBe(53);
    expect(calls).toEqual([
      'https://files.rcsb.org/download/1PRW.cif',
      'https://files.rcsb.org/download/1PRW.pdb',
    ]);

    // Second call hits in-memory cache without re-fetching
    const cached = await fetchStructureByPdbId('1PRW', mockFetch);
    expect(cached.text).toBe(res.text);
    expect(calls).toHaveLength(2);
  });

  it('shares one download and parse across concurrent requests for the same PDB', async () => {
    const sample = generateSampleHelixBundlePdb();
    const mockFetch = vi.fn(async () => new Response(sample));
    const results = await Promise.all(
      ['1prw', '1PRW.cif', ' 1PRW '].map(id => fetchStructureByPdbId(id, mockFetch)),
    );
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(results[0]).toBe(results[1]);
    expect(results[0]).toBe(results[2]);
  });

  it('retries failed lookups and ignores a result superseded by a cache clear', async () => {
    const unavailable = vi.fn(async () => new Response(null, { status: 404 }));
    await expect(fetchStructureByPdbId('1PRW', unavailable)).rejects.toThrow(/Could not fetch/);
    expect(unavailable).toHaveBeenCalledTimes(3);

    const sample = generateSampleHelixBundlePdb();
    let resolveOld!: (r: Response) => void;
    const oldRequest = fetchStructureByPdbId('1PRW', () => new Promise(r => { resolveOld = r; }));
    clearRcsbCache();
    const newResult = await fetchStructureByPdbId('1PRW', async () => new Response(sample));
    resolveOld(new Response(sample + '\n'));
    expect((await oldRequest).text).toBe(sample + '\n');
    expect(await fetchStructureByPdbId('1PRW', unavailable)).toBe(newResult);
    expect(unavailable).toHaveBeenCalledTimes(3);
  });

  it('scheduler auto-fetches PDB ID when pdbText is not yet cached in params', async () => {
    const samplePdb = generateSampleHelixBundlePdb();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(samplePdb, { status: 200 })),
    );

    const scheduler = new ExecutionScheduler();
    const graph: WorkflowGraph = {
      id: 'rcsb-run',
      name: 'RCSB Run',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: 'PDB Input',
          params: { mode: 'pdb-id', pdbId: '1L2Y', pdbText: '' },
        },
      ],
      edges: [],
    };

    const res = await scheduler.run(graph, () => {});
    const art = res.outputsByNode.pdb.structure as StructureArtifact;
    expect(art.kind).toBe('StructureArtifact');
    expect(art.length).toBe(53);
    expect(art.label).toContain('1L2Y');
  });

  it('share links preserve mode: "pdb-id" and pdbId cleanly when stripping large cached pdbText', async () => {
    const bigPdb = generateSampleHelixBundlePdb().repeat(2);
    expect(bigPdb.length).toBeGreaterThan(1500);

    const graph: WorkflowGraph = {
      id: 'share-rcsb',
      name: 'Share RCSB',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: 'PDB Input',
          params: {
            mode: 'pdb-id',
            pdbId: '5O45',
            fetchedPdbId: '5O45',
            filename: '5O45.cif',
            pdbText: bigPdb,
          },
        },
      ],
      edges: [],
    };

    const encoded = await encodeGraphToHash(graph);
    expect(encoded.strippedPdb).toBe(false);
    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes[0].params.mode).toBe('pdb-id');
    expect(decoded!.nodes[0].params.pdbId).toBe('5O45');
    expect(decoded!.nodes[0].params.pdbText).toBe('');
  });

  it('accepts pure-DNA/RNA PDB IDs (e.g. 1BNA, 1EHZ) and ligand-only entries while rejecting water-only responses', async () => {
    const pureDnaCif = [
      'data_1BNA',
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
      'ATOM 1 P DC A 1 1.000 2.000 3.000',
      'ATOM 2 P DG A 2 4.000 5.000 6.000',
      'ATOM 3 P DC B 13 7.000 8.000 9.000',
      '#',
    ].join('\n');

    const dnaFetch = vi.fn(async () => new Response(pureDnaCif, { status: 200 })) as unknown as typeof fetch;
    const dnaRes = await fetchStructureByPdbId('1BNA', dnaFetch);
    expect(dnaRes.pdbId).toBe('1BNA');
    expect(dnaRes.nResidues).toBe(0);
    expect(dnaRes.nucleicAcidChains).toEqual(['A', 'B']);

    const pureRnaPdb = [
      'ATOM      1  P     G A   1       1.000   2.000   3.000  1.00 20.00           P  ',
      'ATOM      2  P     C A   2       4.000   5.000   6.000  1.00 20.00           P  ',
      'END',
    ].join('\n');
    const rnaFetch = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith('.cif')) return new Response('Not Found', { status: 404 });
      return new Response(pureRnaPdb, { status: 200 });
    }) as unknown as typeof fetch;
    const rnaRes = await fetchStructureByPdbId('1EHZ', rnaFetch);
    expect(rnaRes.pdbId).toBe('1EHZ');
    expect(rnaRes.nResidues).toBe(0);
    expect(rnaRes.nucleicAcidChains).toEqual(['A']);

    // Water-only structure with 0 protein/nucleic/ligand atoms is still rejected
    const waterOnlyPdb = [
      'HETATM    1  O   HOH A 101       1.000   2.000   3.000  1.00 10.00           O  ',
      'HETATM    2  O   HOH A 102       4.000   5.000   6.000  1.00 10.00           O  ',
      'END',
    ].join('\n');
    const waterFetch = vi.fn(async () => new Response(waterOnlyPdb, { status: 200 })) as unknown as typeof fetch;
    await expect(fetchStructureByPdbId('9WAT', waterFetch)).rejects.toThrow(/Could not fetch PDB entry "9WAT"/);
  });

  it('aborts hung RCSB/PDBe endpoints via per-endpoint AbortController timeout and falls back cleanly', async () => {
    const validPdb = [
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  ',
      'ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C  ',
      'ATOM      3  CA  VAL A   3       7.600   0.000   0.000  1.00  0.00           C  ',
      'END',
    ].join('\n');
    let calls = 0;
    const hungFirstFetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      calls++;
      if (calls === 1) {
        return await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });
      }
      return new Response(validPdb, { status: 200 });
    }) as unknown as typeof fetch;

    const res = await fetchStructureByPdbId('7TIM', hungFirstFetch, 25);
    expect(res.pdbId).toBe('7TIM');
    expect(res.nResidues).toBe(3);
    expect(calls).toBe(2);
  });
});
