import { describe, expect, it } from 'vitest';
import { getBenchmarkById } from '$lib/engine/benchmarks';
import {
  buildBenchmarkPdbParams,
  detectQuickPdbBenchmarkMatch,
  isResidueSpecDisjointFromBenchmark,
  reconcileBenchmarkSeeding,
  switchBenchmark,
} from '$lib/engine/benchmark-sync';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import type { WorkflowGraph } from '$lib/engine/types';

describe('benchmark-sync: AME 180-180 defaults, symmetric teardown (WF-02), and quick-PDB matching (WF-03)', () => {
  it('sets AME 1CTT and 1OS7 benchmark defaults to 180-180 with approxTokens <= 300', () => {
    const ctt = getBenchmarkById('ame_1ctt_cda')!;
    const os7 = getBenchmarkById('ame_1os7_taud')!;
    expect(ctt.conditionParams.contig).toBe('180-180');
    expect(ctt.approxTokens).toBe(190);
    expect(ctt.approxTokens).toBeLessThanOrEqual(300);

    expect(os7.conditionParams.contig).toBe('180-180');
    expect(os7.approxTokens).toBe(195);
    expect(os7.approxTokens).toBeLessThanOrEqual(300);
  });

  it('WF-02: symmetrically tears down auto-seeded RFD4 & Genie 3 fields on wire disconnect and benchmark exit, and re-seeds cleanly on re-wire', () => {
    const initial: WorkflowGraph = {
      id: 'sync-sym',
      name: 'Symmetric Benchmark Sync',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '5O45',
          params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' },
        },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          label: 'RFD4',
          params: { ...defaultParamsForAdapter('rfd4-proteina') },
        },
        {
          id: 'g3',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: { ...defaultParamsForAdapter('genie3') },
        },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' },
      ],
    };

    // 1. Forward seed
    const seeded = reconcileBenchmarkSeeding(initial);
    expect(seeded.changedNodeIds.sort()).toEqual(['g3', 'rfd']);
    const rfdSeeded = seeded.graph.nodes.find((n) => n.id === 'rfd')!;
    const g3Seeded = seeded.graph.nodes.find((n) => n.id === 'g3')!;
    expect(rfdSeeded.params.mode).toBe('binder');
    expect(rfdSeeded.params.hotspots).toBe('A54, A56, A66, A115');
    expect(rfdSeeded.params.autoSeededBenchmarkId).toBe('pdl1_5o45');
    expect(rfdSeeded.params.autoSeededLength).toBe(70);
    expect(g3Seeded.params.taskMode).toBe('binder');
    expect(g3Seeded.params.length).toBe(70);
    expect(g3Seeded.params.autoSeededBenchmarkId).toBe('pdl1_5o45');

    // 2. Disconnect wires -> symmetric teardown reverts seeded fields & restores default lengths
    const disconnected: WorkflowGraph = { ...seeded.graph, edges: [] };
    const tornDown = reconcileBenchmarkSeeding(disconnected);
    expect(tornDown.changedNodeIds.sort()).toEqual(['g3', 'rfd']);
    const rfdAfter = tornDown.graph.nodes.find((n) => n.id === 'rfd')!;
    const g3After = tornDown.graph.nodes.find((n) => n.id === 'g3')!;
    expect(rfdAfter.params.mode).toBe('unconditional');
    expect(rfdAfter.params.contig).toBe('');
    expect(rfdAfter.params.hotspots).toBe('');
    expect(rfdAfter.params.lockChain).toBe('');
    expect(rfdAfter.params.motifSpan).toBe('');
    expect(rfdAfter.params.unindexedMotif).toBe(false);
    expect(rfdAfter.params.autoSeededBenchmarkId).toBeUndefined();
    expect(rfdAfter.params.autoSeededLength).toBeUndefined();
    expect(rfdAfter.params.length).toBe(70);

    expect(g3After.params.taskMode).toBe('unconditional');
    expect(g3After.params.contig).toBe('');
    expect(g3After.params.hotspots).toBe('');
    expect(g3After.params.lockChain).toBe('');
    expect(g3After.params.autoSeededBenchmarkId).toBeUndefined();
    expect(g3After.params.autoSeededLength).toBeUndefined();
    expect(g3After.params.length).toBe(80);

    // 3. Re-wiring the same benchmark immediately re-seeds cleanly
    const rewired = reconcileBenchmarkSeeding({ ...tornDown.graph, edges: initial.edges });
    expect(rewired.changedNodeIds.sort()).toEqual(['g3', 'rfd']);
    expect(rewired.graph.nodes.find((n) => n.id === 'rfd')!.params.hotspots).toBe('A54, A56, A66, A115');

    // 4. Exiting benchmark mode on pdb-input (`switchBenchmark(..., null)`) tears down while preserving custom length
    const customLenGraph: WorkflowGraph = {
      ...rewired.graph,
      nodes: rewired.graph.nodes.map((n) =>
        n.id === 'rfd' ? { ...n, params: { ...n.params, length: 95, userEditedLength: true } } : n,
      ),
    };
    const exited = switchBenchmark(customLenGraph, 'pdb', null);
    expect(exited.changedNodeIds.sort()).toEqual(['g3', 'rfd']);
    const rfdExited = exited.graph.nodes.find((n) => n.id === 'rfd')!;
    expect(rfdExited.params.mode).toBe('unconditional');
    expect(rfdExited.params.hotspots).toBe('');
    expect(rfdExited.params.length).toBe(95);
    expect(rfdExited.params.userEditedLength).toBe(true);
    expect(rfdExited.params.autoSeededBenchmarkId).toBeUndefined();
  });

  it('WF-02: respects userEditedTask: true and partial/inverse-folding modes during teardown', () => {
    const graph: WorkflowGraph = {
      id: 'guard-teardown',
      name: 'Guard Teardown',
      nodes: [
        {
          id: 'rfd_edited',
          adapterId: 'rfd4-proteina',
          params: {
            ...defaultParamsForAdapter('rfd4-proteina'),
            mode: 'binder',
            hotspots: 'A54',
            autoSeededBenchmarkId: 'pdl1_5o45',
            userEditedTask: true,
          },
        },
        {
          id: 'rfd_partial',
          adapterId: 'rfd4-proteina',
          params: {
            ...defaultParamsForAdapter('rfd4-proteina'),
            mode: 'partial',
            partialFraction: 0.4,
            autoSeededBenchmarkId: 'pdl1_5o45',
          },
        },
      ],
      edges: [],
    };
    const res = reconcileBenchmarkSeeding(graph);
    expect(res.changedNodeIds).toEqual([]);
    expect(res.graph.nodes[0].params.hotspots).toBe('A54');
    expect(res.graph.nodes[1].params.mode).toBe('partial');
  });

  it('WF-03: detectQuickPdbBenchmarkMatch finds curated benchmarks and preserves pdbText when fetchedPdbId matches', () => {
    expect(detectQuickPdbBenchmarkMatch('5o45')?.id).toBe('pdl1_5o45');
    expect(detectQuickPdbBenchmarkMatch('1PRW')?.id).toBe('rfd_1prw_efhand');
    expect(detectQuickPdbBenchmarkMatch('1YCR', 'binder')?.id).toBe('mdm2_1ycr');
    expect(detectQuickPdbBenchmarkMatch('1YCR', 'motif')?.id).toBe('rfd_1ycr_p53');
    expect(detectQuickPdbBenchmarkMatch('1L2Y')).toBeNull();

    const pdl1 = getBenchmarkById('pdl1_5o45')!;
    const preserved = buildBenchmarkPdbParams(
      {
        mode: 'pdb-id',
        pdbId: '5O45',
        fetchedPdbId: '5O45',
        pdbText: 'data_5o45\nATOM ...',
        filename: '5O45.cif',
        benchmarkId: '',
      },
      pdl1,
    );
    expect(preserved.benchmarkId).toBe('pdl1_5o45');
    expect(preserved.fetchedPdbId).toBe('5O45');
    expect(preserved.pdbText).toBe('data_5o45\nATOM ...');
    expect(preserved.cropSegments).toBe('A17-132');

    // Switching to a different PDB clears stale pdbText / fetchedPdbId
    const efhand = getBenchmarkById('rfd_1prw_efhand')!;
    const cleared = buildBenchmarkPdbParams(preserved, efhand);
    expect(cleared.benchmarkId).toBe('rfd_1prw_efhand');
    expect(cleared.pdbId).toBe('1PRW');
    expect(cleared.fetchedPdbId).toBe('');
    expect(cleared.pdbText).toBe('');
    expect(cleared.cropSegments).toBe('A16-35, A52-71');
  });

  it('detects disjoint hallucinated residue spans against curated benchmarks (1PRW A1-10 vs A16-35, A52-71)', () => {
    const efhand = getBenchmarkById('rfd_1prw_efhand')!;
    expect(isResidueSpecDisjointFromBenchmark('A1-10', efhand)).toBe(true);
    expect(isResidueSpecDisjointFromBenchmark('A16-35', efhand)).toBe(false);
    expect(isResidueSpecDisjointFromBenchmark('A16-35, A52-71', efhand)).toBe(false);
    expect(isResidueSpecDisjointFromBenchmark('20-20,A1-10,20-20', efhand)).toBe(true);
    expect(isResidueSpecDisjointFromBenchmark('10-10,A16-35,16-16,A52-71,10-10', efhand)).toBe(false);
  });
});
