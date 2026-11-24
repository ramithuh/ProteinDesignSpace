import { afterEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_ALPHAFOLD_DB_UNIPROT_ID,
  QUICK_ALPHAFOLD_DB_EXAMPLES,
  clearAlphaFoldDbCache,
  fetchStructureByUniProtId,
  isValidUniProtId,
  normalizeUniProtId,
} from '$lib/protein-ops/alphafold-db';
import { validateNodeParams, validateUniProtId } from '$lib/engine/param-validation';
import { filterCatalogCategories } from '$lib/engine/catalog';
import { decodeHashToGraph, encodeGraphToHash } from '$lib/engine/share';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { buildNodePreviewSignature } from '$lib/ui/workbench-controller';
import { formatNodeParamSummary } from '$lib/ui/node-display-helpers';
import { materializePdbInputNode } from '$lib/plugins/pdb-input/cpu-runner';
import type { WorkflowGraph, WorkflowNode } from '$lib/engine/types';

const MINI_AFDB_PDB = [
  'ATOM      1  N   MET A   1       1.000   1.000   1.000  1.00 92.40           N',
  'ATOM      2  CA  MET A   1       2.458   1.000   1.000  1.00 92.40           C',
  'ATOM      3  C   MET A   1       3.000   2.420   1.000  1.00 92.40           C',
  'ATOM      4  O   MET A   1       2.250   3.400   1.000  1.00 92.40           O',
  'ATOM      5  N   ALA A   2       4.320   2.500   1.000  1.00 95.10           N',
  'ATOM      6  CA  ALA A   2       5.000   3.790   1.000  1.00 95.10           C',
  'ATOM      7  C   ALA A   2       6.510   3.650   1.000  1.00 95.10           C',
  'ATOM      8  O   ALA A   2       7.050   2.550   1.000  1.00 95.10           O',
  'ATOM      9  N   GLY A   3       7.200   4.780   1.000  1.00 88.20           N',
  'ATOM     10  CA  GLY A   3       8.650   4.780   1.000  1.00 88.20           C',
  'ATOM     11  C   GLY A   3       9.200   6.190   1.000  1.00 88.20           C',
  'ATOM     12  O   GLY A   3       8.450   7.170   1.000  1.00 88.20           O',
  'END',
].join('\n');

afterEach(() => {
  clearAlphaFoldDbCache();
});

describe('AlphaFold DB input integration', () => {
  it('normalizes and validates UniProt accessions, AF filenames, and AlphaFold DB URLs', () => {
    expect(normalizeUniProtId(' p04637 ')).toBe('P04637');
    expect(normalizeUniProtId('AF-P69905-F1-model_v6.cif')).toBe('P69905');
    expect(normalizeUniProtId('https://alphafold.ebi.ac.uk/entry/Q9NZQ7')).toBe('Q9NZQ7');
    expect(normalizeUniProtId('https://alphafold.ebi.ac.uk/files/AF-P00698-F1-model_v6.pdb')).toBe('P00698');

    for (const ex of QUICK_ALPHAFOLD_DB_EXAMPLES) {
      expect(isValidUniProtId(ex.id)).toBe(true);
      expect(validateUniProtId(ex.id).status).toBe('ok');
    }
    expect(isValidUniProtId('A0A024RBG1')).toBe(true);
    expect(isValidUniProtId('P04637-2')).toBe(true);
    expect(isValidUniProtId('1PRW')).toBe(false);
    expect(isValidUniProtId('')).toBe(false);
    expect(validateUniProtId('1PRW').status).toBe('error');
  });

  it('fetches via the public EMBL-EBI prediction API + structure file URL and caches results', async () => {
    const calls: string[] = [];
    const mockFetch: typeof fetch = (async (input: RequestInfo | URL) => {
      const url = String(input);
      calls.push(url);
      if (url.includes('/api/prediction/P69905')) {
        return new Response(
          JSON.stringify([
            {
              entryId: 'AF-P69905-F1',
              uniprotAccession: 'P69905',
              latestVersion: 6,
              gene: 'HBA1',
              uniprotDescription: 'Hemoglobin subunit alpha',
              organismScientificName: 'Homo sapiens',
              globalMetricValue: 98.06,
              cifUrl: 'https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v6.cif',
              pdbUrl: 'https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v6.pdb',
            },
          ]),
          { status: 200, headers: { 'content-type': 'application/json' } },
        );
      }
      if (url.endsWith('AF-P69905-F1-model_v6.cif')) {
        return new Response('Not Found', { status: 404 });
      }
      if (url.endsWith('AF-P69905-F1-model_v6.pdb')) {
        return new Response(MINI_AFDB_PDB, { status: 200 });
      }
      return new Response('Not Found', { status: 404 });
    }) as typeof fetch;

    const first = await fetchStructureByUniProtId('af-p69905-f1-model_v6.cif', mockFetch);
    expect(first.uniprotId).toBe('P69905');
    expect(first.entryId).toBe('AF-P69905-F1');
    expect(first.version).toBe(6);
    expect(first.gene).toBe('HBA1');
    expect(first.meanPlddt).toBe(98.06);
    expect(first.nResidues).toBe(3);
    expect(first.filename).toBe('AF-P69905-F1-model_v6.pdb');

    // Second call hits memory cache without extra network requests
    const second = await fetchStructureByUniProtId('P69905', mockFetch);
    expect(second).toBe(first);
    expect(calls).toHaveLength(3);
  });

  it('exposes AlphaFold DB in the Inputs catalog and validates, previews, materializes, and shares cleanly', async () => {
    const matches = filterCatalogCategories('alphafold db');
    const inputsMatch = matches.find((m) => m.category.id === 'inputs');
    expect(inputsMatch).toBeDefined();
    const afdbItem = inputsMatch!.matchingSubItems.find((s) => s.key === 'alphafold-db');
    expect(afdbItem).toBeDefined();
    expect(afdbItem!.insert.adapterId).toBe('pdb-input');
    expect(afdbItem!.insert.params?.mode).toBe('alphafold-db');
    expect(afdbItem!.insert.params?.uniprotId).toBe(DEFAULT_ALPHAFOLD_DB_UNIPROT_ID);

    const node: WorkflowNode = {
      id: 'afdb_1',
      adapterId: 'pdb-input',
      label: `AFDB: Hemoglobin α (${DEFAULT_ALPHAFOLD_DB_UNIPROT_ID})`,
      params: {
        mode: 'alphafold-db',
        uniprotId: DEFAULT_ALPHAFOLD_DB_UNIPROT_ID,
        fetchedUniProtId: DEFAULT_ALPHAFOLD_DB_UNIPROT_ID,
        filename: `AF-${DEFAULT_ALPHAFOLD_DB_UNIPROT_ID}-F1-model_v6.pdb`,
        pdbText: MINI_AFDB_PDB,
        cropSegments: '',
      },
    };

    expect(validateNodeParams(node)).toEqual([]);
    expect(formatNodeParamSummary('pdb-input', node.params)).toBe(`AFDB ${DEFAULT_ALPHAFOLD_DB_UNIPROT_ID}`);
    expect(buildNodePreviewSignature(node)).toMatch(/^alphafold-db\|P69905\|/);

    const art = await materializePdbInputNode(node);
    expect(art.length).toBe(3);
    expect(art.sequence).toBe('MAG');

    // Share URL strips cached pdbText and fetchedUniProtId for accession-based AFDB inputs
    const graph: WorkflowGraph = {
      id: 'afdb-share-test',
      name: 'AFDB Share Test',
      nodes: [node],
      edges: [],
    };
    const encoded = await encodeGraphToHash(graph);
    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes[0].params.mode).toBe('alphafold-db');
    expect(decoded!.nodes[0].params.uniprotId).toBe(DEFAULT_ALPHAFOLD_DB_UNIPROT_ID);
    expect(decoded!.nodes[0].params.pdbText ?? '').toBe('');
    expect(decoded!.nodes[0].params.fetchedUniProtId).toBeUndefined();

    // Unfetched AlphaFold DB -> GNINA (score mode) flags apo protein pre-run
    const afdbToGnina: WorkflowGraph = {
      id: 'afdb-gnina-apo',
      name: 'AFDB -> GNINA',
      nodes: [
        {
          id: 'afdb',
          adapterId: 'pdb-input',
          params: { mode: 'alphafold-db', uniprotId: 'P04637', pdbText: '' },
        },
        {
          id: 'gnina',
          adapterId: 'gnina',
          params: { taskMode: 'score' },
        },
      ],
      edges: [{ fromNode: 'afdb', fromPort: 'structure', toNode: 'gnina', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(afdbToGnina).some(
        (i) => i.nodeId === 'gnina' && i.code === 'gnina-no-organic-ligand' && i.severity === 'error',
      ),
    ).toBe(true);
  });
});
