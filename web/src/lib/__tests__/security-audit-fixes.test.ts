import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { computeTargetOnlyOffMask } from '$lib/engine/localfold-msa';
import { estimateRunRequirements } from '$lib/engine/run-preflight';
import { decodeHashToGraph, normalizeWorkflowGraph } from '$lib/engine/share';
import { exportJoinedCollectionCsv } from '$lib/engine/compare';
import type { WorkflowGraph } from '$lib/engine/types';

const WEB_ROOT = resolve(process.cwd());

describe('Security audit hardening regressions', () => {
  it('Finding 1: vite.config.ts enforces strict FS boundary and does not wildcard .ts.net', () => {
    const viteCfg = readFileSync(resolve(WEB_ROOT, 'vite.config.ts'), 'utf8');
    expect(viteCfg).toContain('strict: true');
    expect(viteCfg).not.toContain('strict: false');
    expect(viteCfg).not.toContain("'.ts.net'");
    expect(viteCfg).not.toContain('".ts.net"');
  });

  it('Finding 2: target-only MSA fails closed when no target/locked chain matches, and preflight discloses remote MSA transfers', () => {
    expect(computeTargetOnlyOffMask(['A', 'B'], undefined, undefined)).toEqual([true, true]);
    expect(computeTargetOnlyOffMask(['A', 'B'], ['H'], { H: 'target' })).toEqual([true, true]);
    expect(computeTargetOnlyOffMask(['A', 'B'], ['A'], undefined)).toEqual([false, true]);

    const graphWithMsa: WorkflowGraph = {
      id: 'msa-test',
      name: 'MSA Test',
      nodes: [
        { id: 'seq_1', adapterId: 'sequence-input', label: 'Seq', params: { sequence: 'MKTAYIAKQR' } },
        {
          id: 'fold_1',
          adapterId: 'localfold-af2',
          label: 'AF2',
          params: { model: 'monomer', msaMode: 'target-only' },
        },
      ],
      edges: [{ fromNode: 'seq_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' }],
    };
    const preflight = estimateRunRequirements(graphWithMsa);
    expect(preflight.remoteMsaTransfers).toEqual([
      {
        nodeId: 'fold_1',
        nodeLabel: 'AF2',
        mode: 'target-only',
        destination: 'api.colabfold.com',
      },
    ]);

    const offlineGraph: WorkflowGraph = {
      ...graphWithMsa,
      nodes: [
        graphWithMsa.nodes[0],
        { ...graphWithMsa.nodes[1], params: { model: 'monomer', msaMode: 'single' } },
      ],
    };
    expect(estimateRunRequirements(offlineGraph).remoteMsaTransfers).toBeUndefined();
  });

  it('Finding 3: decodeHashToGraph rejects oversized URL hashes and oversized node/edge counts', async () => {
    await expect(decodeHashToGraph(`#g=${'A'.repeat(70_000)}`)).rejects.toThrow(/too large/i);

    const manyNodes = Array.from({ length: 70 }, (_, i) => [`n${i}`, 'p', {}]);
    const payload = Buffer.from(JSON.stringify({ n: manyNodes, e: [] }), 'utf8')
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
    await expect(decodeHashToGraph(`#g=${payload}`)).rejects.toThrow(/size limits/i);
  });

  it('Finding 6: normalizeWorkflowGraph strips spoofed fetchedPdbId and pdbText on pdb-id imports', () => {
    const imported = normalizeWorkflowGraph({
      id: 'spoof-test',
      name: 'Spoof Test',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          label: 'RCSB 1UBQ',
          params: {
            mode: 'pdb-id',
            pdbId: '1UBQ',
            fetchedPdbId: '1UBQ',
            pdbText: 'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 99.00           C\nEND\n',
          },
        },
      ],
      edges: [],
    });
    const pdbNode = imported.nodes[0];
    expect(pdbNode.params.mode).toBe('pdb-id');
    expect(pdbNode.params.pdbId).toBe('1UBQ');
    expect(pdbNode.params.fetchedPdbId).toBeUndefined();
    expect(pdbNode.params.pdbText).toBe('');
  });

  it('Other findings: CSV export neutralizes spreadsheet formulas while preserving negative numbers, and CI/headers are hardened', () => {
    const csv = exportJoinedCollectionCsv({
      branchIds: ['=CMD|evil', 'safe'],
      keys: ['+SUM(A1:A2)'],
      completeRowCount: 1,
      partialRowCount: 0,
      rows: [
        {
          key: '+SUM(A1:A2)',
          presentCount: 2,
          missingBranchIds: [],
          byBranch: {
            '=CMD|evil': {
              branchId: '=CMD|evil',
              present: true,
              rank: 1,
              score: -2.45,
              confidence: 91.2,
              confidenceAvailable: true,
              ipsae: null,
              item: null,
            },
            safe: {
              branchId: 'safe',
              present: true,
              rank: 2,
              score: 1.12,
              confidence: 85.0,
              confidenceAvailable: true,
              ipsae: null,
              item: null,
            },
          },
        },
      ],
    });
    const lines = csv.split('\n');
    expect(lines[0]).toContain("'=CMD|evil_status");
    expect(lines[1].startsWith("'+SUM(A1:A2),")).toBe(true);
    // Genuine negative float must NOT be quote-prefixed
    expect(lines[1]).toContain(',-2.45,');

    const headers = readFileSync(resolve(WEB_ROOT, 'static/_headers'), 'utf8');
    expect(headers).toContain('X-Frame-Options: DENY');
    expect(headers).toContain('Strict-Transport-Security');
    const ciYml = readFileSync(resolve(WEB_ROOT, '../.github/workflows/ci.yml'), 'utf8');
    expect(ciYml).toContain('contents: read');
    expect(ciYml).toContain('persist-credentials: false');
    expect(ciYml).toContain('SCARF_ANALYTICS');

    // Supply-chain C1: mpnn.worker.ts loads first-party /vendor/ort/ instead of cdn.jsdelivr.net
    const mpnnWorker = readFileSync(resolve(WEB_ROOT, 'src/lib/plugins/web-protein-mpnn/mpnn.worker.ts'), 'utf8');
    expect(mpnnWorker).not.toContain('cdn.jsdelivr.net');
    expect(mpnnWorker).toContain('/vendor/ort/');

    // M2: rfd4.worker.ts rejects arbitrary http/https URLs and .. traversal in fetchExampleCifBytes
    const rfd4Worker = readFileSync(resolve(WEB_ROOT, 'src/lib/workers/rfd4.worker.ts'), 'utf8');
    expect(rfd4Worker).toContain('SAFE_EXAMPLE_CIF_RE');
  });
});
