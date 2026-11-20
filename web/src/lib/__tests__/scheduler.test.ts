import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { applyTransformToPdb } from '$lib/protein-ops/kabsch';
import { generateSampleHelixBundlePdb, parsePdbBackbone } from '$lib/protein-ops/pdb';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import type {
  ConditionSpecArtifact,
  DesignabilityMetrics,
  FoldedStructure,
  NodeProgressEvent,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

// Controllable Worker stub so designability-scorer & memoization can be tested end-to-end through ExecutionScheduler
let mockFoldFactory: (() => FoldedStructure) | null = null;
let mockWorkerRunCalls = 0;

class MockWorker {
  private listeners = new Map<string, Set<(ev: MessageEvent) => void>>();
  addEventListener(type: string, cb: (ev: MessageEvent) => void) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(cb);
  }
  removeEventListener(type: string, cb: (ev: MessageEvent) => void) {
    this.listeners.get(type)?.delete(cb);
  }
  postMessage(msg: { id: number; type: string }) {
    if (msg.type === 'dispose') {
      this.emit('message', { data: { id: msg.id, type: 'disposed' } } as MessageEvent);
      return;
    }
    if (msg.type === 'run') {
      mockWorkerRunCalls++;
      const folds = mockFoldFactory ? mockFoldFactory() : {
        kind: 'FoldedStructure' as const,
        id: 'mock-folds',
        parentStructureId: 'none',
        folds: [],
      };
      queueMicrotask(() => {
        this.emit('message', {
          data: { id: msg.id, type: 'result', outputs: { folds }, elapsedMs: 5 },
        } as MessageEvent);
      });
    }
  }
  terminate() {}
  private emit(type: string, ev: MessageEvent) {
    for (const cb of this.listeners.get(type) ?? []) cb(ev);
  }
}

describe('engine/scheduler (CPU nodes & memoization)', () => {
  beforeEach(() => {
    mockFoldFactory = null;
    mockWorkerRunCalls = 0;
    vi.stubGlobal('Worker', MockWorker);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('executes pdb-input in demo-bundle and custom PDB/mmCIF modes, and rejects < 3 residues', async () => {
    const scheduler = new ExecutionScheduler();
    const events: NodeProgressEvent[] = [];

    // Demo bundle
    const resDemo = await scheduler.run(
      {
        id: 'g-demo',
        name: 'Demo',
        nodes: [{ id: 'p', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } }],
        edges: [],
      },
      (ev) => events.push(ev),
    );
    const struct = resDemo.outputsByNode.p.structure as StructureArtifact;
    expect(struct.kind).toBe('StructureArtifact');
    expect(struct.length).toBeGreaterThanOrEqual(50);
    expect(struct.geometry.chainBreaks).toBe(0);
    expect(struct.geometry.clashes).toBe(0);
    expect(struct.cifText).toBeUndefined();
    expect(events.map((e) => e.status)).toEqual(['running', 'done']);

    // Custom mmCIF populates cifText
    const cif = [
      'data_mini',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM CA ALA A 1 0.0 0.0 0.0',
      'ATOM CA GLY A 2 3.8 0.0 0.0',
      'ATOM CA VAL A 3 7.6 0.0 0.0',
      '#',
    ].join('\n');
    const resCif = await scheduler.run(
      {
        id: 'g-cif',
        name: 'CIF',
        nodes: [
          {
            id: 'p',
            adapterId: 'pdb-input',
            params: { mode: 'custom', pdbText: cif, filename: 'mini.cif' },
          },
        ],
        edges: [],
      },
      () => {},
    );
    const cifStruct = resCif.outputsByNode.p.structure as StructureArtifact;
    expect(cifStruct.length).toBe(3);
    expect(cifStruct.sequence).toBe('AGV');
    expect(cifStruct.cifText).toBe(cif);

    // Rejects < 3 residues
    await expect(
      scheduler.run(
        {
          id: 'g-tiny',
          name: 'Tiny',
          nodes: [
            {
              id: 'p',
              adapterId: 'pdb-input',
              params: {
                mode: 'custom',
                pdbText: 'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n',
              },
            },
          ],
          edges: [],
        },
        () => {},
      ),
    ).rejects.toThrow(/at least 3 residues/);
  });

  it('executes sequence-input on plain text, multi-FASTA, and throws on empty / too-short input', async () => {
    const scheduler = new ExecutionScheduler();

    // Plain sequence with whitespace & lowercase
    const plainRes = await scheduler.run(
      {
        id: 'seq-plain',
        name: 'Plain',
        nodes: [
          {
            id: 's',
            adapterId: 'sequence-input',
            params: { label: 'helix-1', sequence: '  mktay iakqr 123 qisfv  ' },
          },
        ],
        edges: [],
      },
      () => {},
    );
    const plainSet = plainRes.outputsByNode.s.sequences as SequenceSet;
    expect(plainSet.candidates).toHaveLength(1);
    expect(plainSet.candidates[0].id).toBe('helix-1');
    expect(plainSet.candidates[0].sequence).toBe('MKTAYIAKQRQISFV');

    // Multi-FASTA (short <5 aa entry is ignored; valid entries kept)
    const fasta = [
      '>binder_A description here',
      'ACDEFGHIKL',
      'MNPQRSTVWY',
      '>too_short',
      'AAA',
      '>binder_B',
      'GGGGGAAAAA',
    ].join('\n');
    const fastaRes = await scheduler.run(
      {
        id: 'seq-fasta',
        name: 'FASTA',
        nodes: [{ id: 's', adapterId: 'sequence-input', params: { sequence: fasta } }],
        edges: [],
      },
      () => {},
    );
    const fastaSet = fastaRes.outputsByNode.s.sequences as SequenceSet;
    expect(fastaSet.candidates.map((c) => [c.id, c.sequence])).toEqual([
      ['binder_A', 'ACDEFGHIKLMNPQRSTVWY'],
      ['binder_B', 'GGGGGAAAAA'],
    ]);

    // Empty input throws
    await expect(
      scheduler.run(
        {
          id: 'seq-empty',
          name: 'Empty',
          nodes: [{ id: 's', adapterId: 'sequence-input', params: { sequence: '   ' } }],
          edges: [],
        },
        () => {},
      ),
    ).rejects.toThrow(/empty/i);

    // Only invalid / < 5 aa input throws
    await expect(
      scheduler.run(
        {
          id: 'seq-short',
          name: 'Short',
          nodes: [{ id: 's', adapterId: 'sequence-input', params: { sequence: 'ABZ12' } }],
          edges: [],
        },
        () => {},
      ),
    ).rejects.toThrow(/No valid amino acid/);
  });

  it('executes rfd4-conditioner for hotspots, indexed/unindexed motifs, H-bonds, CATH, rawJson, and taskPreset', async () => {
    const scheduler = new ExecutionScheduler();

    // 1. Combined custom conditioning (hotspots + unindexed motif + lockChain + H-bond + CATH) with template
    const combinedGraph: WorkflowGraph = {
      id: 'cond-full',
      name: 'Cond Full',
      nodes: [
        { id: 'tpl', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        {
          id: 'cond',
          adapterId: 'rfd4-conditioner',
          params: {
            taskPreset: 'custom',
            contig: 'A1-20,/0,50-50',
            lockChain: 'a',
            hotspots: 'A54, A56, B10-12',
            motifSpan: 'A16-35, A52-71',
            unindexedMotif: true,
            hbondMode: 'ACCEPTOR',
            hbondSelection: 'B/SAM/*/*',
            cathCode: '3.40.50',
            rawJsonOverride: '',
          },
        },
      ],
      edges: [{ fromNode: 'tpl', fromPort: 'structure', toNode: 'cond', toPort: 'template' }],
    };
    const res1 = await scheduler.run(combinedGraph, () => {});
    const c1 = res1.outputsByNode.cond.condition as ConditionSpecArtifact;
    expect(c1.kind).toBe('ConditionSpec');
    expect(c1.contig).toBe('A1-20,/0,50-50');
    expect(c1.specMode).toBe(true);
    expect(c1.inputCif).toBeTruthy();
    expect(c1.conditions.C_CRD).toEqual({ true: [{ select: "sel('A/*/*/*')" }] });
    expect(c1.conditions.C_HOT).toEqual({
      true: [{ select: "sel('A/*/54,56/*') | sel('B/*/10-12/*')" }],
    });
    expect(c1.conditions.C_CRDGRP).toEqual({
      1: [{ select: "sel('A/*/16-35,52-71/*')" }],
    });
    expect(c1.conditions.C_IDX).toEqual({
      false: [{ select: "sel('A/*/16-35,52-71/*')" }],
    });
    expect(c1.conditions.C_HB).toEqual({
      ACCEPTOR: [{ select: "sel('B/SAM/*/*')" }],
    });
    expect(c1.conditions.C_CATH).toEqual({
      '3.40.50': [{ select: "sel('*/*/*/*')" }],
    });

    // 2. Indexed motif (unindexedMotif: false => no C_IDX)
    const resIdx = await scheduler.run(
      {
        id: 'cond-idx',
        name: 'Idx',
        nodes: [
          {
            id: 'cond',
            adapterId: 'rfd4-conditioner',
            params: { taskPreset: 'custom', contig: '5-20,A16-35,5-20', motifSpan: 'A16-35', unindexedMotif: false },
          },
        ],
        edges: [],
      },
      () => {},
    );
    const cIdx = resIdx.outputsByNode.cond.condition as ConditionSpecArtifact;
    expect(cIdx.conditions.C_CRDGRP).toBeDefined();
    expect(cIdx.conditions.C_IDX).toBeUndefined();

    // 3. rawJsonOverride takes precedence
    const resRaw = await scheduler.run(
      {
        id: 'cond-raw',
        name: 'Raw',
        nodes: [
          {
            id: 'cond',
            adapterId: 'rfd4-conditioner',
            params: {
              taskPreset: 'ppi_pdl1',
              rawJsonOverride: JSON.stringify({
                contig: '64-64',
                conditions: { C_CATH: { '1.10.287': [{ select: "sel('*/*/*/*')" }] } },
              }),
            },
          },
        ],
        edges: [],
      },
      () => {},
    );
    const cRaw = resRaw.outputsByNode.cond.condition as ConditionSpecArtifact;
    expect(cRaw.contig).toBe('64-64');
    expect(cRaw.conditions.C_CATH).toBeDefined();
    expect(cRaw.cifUrl).toBeUndefined();

    // 4. taskPreset mode
    const resPreset = await scheduler.run(
      {
        id: 'cond-preset',
        name: 'Preset',
        nodes: [
          {
            id: 'cond',
            adapterId: 'rfd4-conditioner',
            params: { taskPreset: 'ppi_pdl1', contig: 'A18-132,/0,76-76' },
          },
        ],
        edges: [],
      },
      () => {},
    );
    const cPreset = resPreset.outputsByNode.cond.condition as ConditionSpecArtifact;
    expect(cPreset.cifUrl).toBe('preset:ppi_pdl1');
    expect(cPreset.summary).toBe('preset:ppi_pdl1');
  });

  it('executes designability-scorer with PASS/FAIL thresholds, scRMSD sorting, and accurate alignedFoldPdb', async () => {
    const basePdb = generateSampleHelixBundlePdb();
    const baseParsed = parsePdbBackbone(basePdb);
    const L = baseParsed.nResidues;

    // Candidate 1 (PASS): rigid 90-deg rotation around Z + translation [25, -10, 5] => scRMSD ~ 0 A, pLDDT 90
    const R90 = new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 1]);
    const tShift = new Float64Array([25, -10, 5]);
    const passPdb = applyTransformToPdb(basePdb, R90, tShift);
    const passCa = parsePdbBackbone(passPdb).caCoords;

    // Candidate 2 (FAIL by pLDDT): exact same coords (scRMSD ~ 0 A), but meanPlddt = 65 < 70
    const lowPlddtCa = new Float32Array(passCa);

    // Candidate 3 (FAIL by scRMSD): large coordinate distortion (> 4 A RMSD), high pLDDT = 88
    const badCa = new Float32Array(baseParsed.caCoords.length);
    for (let i = 0; i < L; i++) {
      badCa[i * 3 + 0] = baseParsed.caCoords[i * 3 + 0] + (i % 2 === 0 ? 5.5 : -5.5);
      badCa[i * 3 + 1] = baseParsed.caCoords[i * 3 + 1] + (i % 3 === 0 ? 4.0 : -2.0);
      badCa[i * 3 + 2] = baseParsed.caCoords[i * 3 + 2];
    }

    // Intentionally return candidates unsorted (worst first) to verify ascending scRMSD sort
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-test-1',
      parentStructureId: 'demo',
      folds: [
        {
          sequenceId: 'cand-high-rmsd',
          sequence: 'A'.repeat(L),
          foldModelId: 'localfold-af2',
          pdbText: basePdb,
          caCoords: badCa,
          plddt: new Float32Array(L).fill(88),
          meanPlddt: 88,
          ptm: 0.81,
          elapsedMs: 100,
        },
        {
          sequenceId: 'cand-low-plddt',
          sequence: 'G'.repeat(L),
          foldModelId: 'localfold-af2',
          pdbText: passPdb,
          caCoords: lowPlddtCa,
          plddt: new Float32Array(L).fill(65),
          meanPlddt: 65,
          ptm: 0.62,
          elapsedMs: 110,
        },
        {
          sequenceId: 'cand-pass',
          sequence: 'L'.repeat(L),
          foldModelId: 'localfold-af2',
          pdbText: passPdb,
          caCoords: passCa,
          plddt: new Float32Array(L).fill(91),
          meanPlddt: 91,
          ptm: 0.89,
          elapsedMs: 95,
        },
      ],
    });

    const graph: WorkflowGraph = {
      id: 'score-test',
      name: 'Score Test',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'L'.repeat(L) } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 3, seed: 0 } },
        { id: 'score', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };

    const scheduler = new ExecutionScheduler();
    const res = await scheduler.run(graph, () => {});
    const metrics = res.outputsByNode.score.metrics as DesignabilityMetrics;

    expect(metrics.kind).toBe('DesignabilityMetrics');
    expect(metrics.rows).toHaveLength(3);
    // Sorted ascending by scRMSD: worst candidate must be last
    expect(metrics.rows[0].scRmsd).toBeLessThanOrEqual(metrics.rows[1].scRmsd);
    expect(metrics.rows[1].scRmsd).toBeLessThan(metrics.rows[2].scRmsd);
    expect(metrics.rows[2].sequenceId).toBe('cand-high-rmsd');
    expect(metrics.best).toBe(metrics.rows[0]);

    // Exactly 1 of 3 passes (scRmsd < 2.0 && meanPlddt >= 70)
    const byId = new Map(metrics.rows.map((r) => [r.sequenceId, r]));
    expect(byId.get('cand-pass')!.designable).toBe(true);
    expect(byId.get('cand-pass')!.scRmsd).toBeLessThan(2e-3);
    expect(byId.get('cand-pass')!.scTm).toBeGreaterThan(0.999);
    expect(byId.get('cand-low-plddt')!.designable).toBe(false);
    expect(byId.get('cand-high-rmsd')!.designable).toBe(false);
    expect(metrics.passRate).toBeCloseTo(1 / 3, 6);

    // alignedFoldPdb must be rigidly superimposed onto designed backbone
    const alignedParsed = parsePdbBackbone(byId.get('cand-pass')!.alignedFoldPdb);
    expect(alignedParsed.nResidues).toBe(L);
    for (let i = 0; i < baseParsed.caCoords.length; i++) {
      expect(Math.abs(alignedParsed.caCoords[i] - baseParsed.caCoords[i])).toBeLessThan(2e-3);
    }

    await scheduler.releaseActiveGpu();
  });

  it('memoizes node results on cache hit, re-runs on bypassCache / param change / clearMemo, and rejects cycles', async () => {
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'memo-folds',
      parentStructureId: 'none',
      folds: [],
    });

    const graph: WorkflowGraph = {
      id: 'memo-g',
      name: 'Memo',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'ACDEFGHIKLMNPQRSTVWY' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1, seed: 0 } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };

    const scheduler = new ExecutionScheduler();

    // 1st run: cold
    const ev1: NodeProgressEvent[] = [];
    const r1 = await scheduler.run(graph, (e) => ev1.push(e));
    expect(mockWorkerRunCalls).toBe(1);
    expect(ev1.some((e) => e.status === 'cached')).toBe(false);

    // 2nd run: full cache hit
    const ev2: NodeProgressEvent[] = [];
    const r2 = await scheduler.run(graph, (e) => ev2.push(e));
    expect(mockWorkerRunCalls).toBe(1);
    expect(ev2.filter((e) => e.status === 'cached').map((e) => e.nodeId)).toEqual(['seq', 'fold']);
    expect(r2.outputsByNode.seq.sequences).toBe(r1.outputsByNode.seq.sequences);

    // 3rd run with bypassCache: true => executes again
    const ev3: NodeProgressEvent[] = [];
    await scheduler.run(graph, (e) => ev3.push(e), { bypassCache: true });
    expect(mockWorkerRunCalls).toBe(2);
    expect(ev3.some((e) => e.status === 'cached')).toBe(false);

    // 4th run with modified downstream param => upstream seq hits cache, downstream fold re-runs
    const modified = structuredClone(graph);
    modified.nodes[1].params.recycles = 2;
    const ev4: NodeProgressEvent[] = [];
    await scheduler.run(modified, (e) => ev4.push(e));
    expect(mockWorkerRunCalls).toBe(3);
    expect(ev4.find((e) => e.nodeId === 'seq' && e.status === 'cached')).toBeDefined();
    expect(ev4.find((e) => e.nodeId === 'fold' && e.status === 'done')).toBeDefined();

    // 5th run after clearMemo() => cold again
    scheduler.clearMemo();
    await scheduler.run(graph, () => {});
    expect(mockWorkerRunCalls).toBe(4);

    // Cycle rejection in topoSort
    await expect(
      scheduler.run(
        {
          id: 'cyclic',
          name: 'Cyclic',
          nodes: [
            { id: 'a', adapterId: 'rfd4-conditioner', params: {} },
            { id: 'b', adapterId: 'rfd4-conditioner', params: {} },
          ],
          edges: [
            { fromNode: 'a', fromPort: 'condition', toNode: 'b', toPort: 'template' },
            { fromNode: 'b', fromPort: 'condition', toNode: 'a', toPort: 'template' },
          ],
        },
        () => {},
      ),
    ).rejects.toThrow(/cycle/i);

    await scheduler.releaseActiveGpu();
  });
});
