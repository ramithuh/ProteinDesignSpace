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
    expect(encoded.strippedPdb).toBe(true);
    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes[0].params.mode).toBe('pdb-id');
    expect(decoded!.nodes[0].params.pdbId).toBe('5O45');
    expect(decoded!.nodes[0].params.pdbText).toBe('');
  });
});
