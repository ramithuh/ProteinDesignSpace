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

// Controllable Worker stub so designability-scorer, memoization & releaseActiveGpu can be tested end-to-end
let mockFoldFactory: (() => FoldedStructure) | null = null;
let mockGenStructureFactory: (() => StructureArtifact) | null = null;
let mockWorkerRunCalls = 0;
let mockHangWorker = false;

class MockWorker {
  private listeners = new Map<string, Set<(ev: MessageEvent) => void>>();
  addEventListener(type: string, cb: (ev: MessageEvent) => void) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(cb);
  }
  removeEventListener(type: string, cb: (ev: MessageEvent) => void) {
    this.listeners.get(type)?.delete(cb);
  }
  postMessage(msg: { id: number; type: string; inputs?: Record<string, unknown> }) {
    if (msg.type === 'dispose') {
      this.emit('message', { data: { id: msg.id, type: 'disposed' } } as MessageEvent);
      return;
    }
    if (msg.type === 'run') {
      mockWorkerRunCalls++;
      if (mockHangWorker) return; // intentionally stays in-flight to test mid-run releaseActiveGpu()
      const hasSeqInput = Boolean(msg.inputs && 'sequences' in msg.inputs);
      if (!hasSeqInput && mockGenStructureFactory) {
        const structure = mockGenStructureFactory();
        queueMicrotask(() => {
          this.emit('message', {
            data: { id: msg.id, type: 'result', outputs: { structure }, elapsedMs: 5 },
          } as MessageEvent);
        });
        return;
      }
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
    mockGenStructureFactory = null;
    mockWorkerRunCalls = 0;
    mockHangWorker = false;
    vi.stubGlobal('Worker', MockWorker);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('rejects invalid programmatic runs before emitting progress or launching workers', async () => {
    const scheduler = new ExecutionScheduler();
    const progress = vi.fn();
    const graph: WorkflowGraph = {
      id: 'invalid', name: 'Invalid',
      nodes: [{ id: 'gen', adapterId: 'genie3', params: { length: 80.5 } }], edges: [],
    };
    await expect(scheduler.run(graph, progress)).rejects.toThrow(/integer/i);
    expect(progress).not.toHaveBeenCalled();
    expect(mockWorkerRunCalls).toBe(0);
    graph.nodes = [{ id: 'p', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: '' } }];
    await expect(scheduler.run(graph, progress)).rejects.toThrow(/structure|PDB/i);
    graph.nodes = [{ id: 'p', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } }];
    graph.edges = [{ fromNode: 'missing', fromPort: 'structure', toNode: 'p', toPort: 'structure' }];
    await expect(scheduler.run(graph, progress)).rejects.toThrow(/missing|unknown/i);
    expect(progress).not.toHaveBeenCalled();
    graph.edges = [];
    await expect(scheduler.run(graph, progress)).resolves.toBeDefined();
  });

  it('keeps FASTA candidate IDs unique when explicit headers collide with generated suffixes', async () => {
    const result = await new ExecutionScheduler().run({
      id: 'headers', name: 'Headers',
      nodes: [{ id: 's', adapterId: 'sequence-input', params: {
        sequence: '>dup\nAAAAA\n>dup\nGGGGG\n>dup_2\nLLLLL\n>dup\nVVVVV',
      } }], edges: [],
    }, () => {});
    const candidates = (result.outputsByNode.s.sequences as SequenceSet).candidates;
    expect(new Set(candidates.map(c => c.id)).size).toBe(candidates.length);
    expect(candidates.map(c => c.sequence)).toEqual(['AAAAA', 'GGGGG', 'LLLLL', 'VVVVV']);
  });

  it('invalidates downstream memoization when a sequence candidate is renamed', async () => {
    const scheduler = new ExecutionScheduler();
    const graph: WorkflowGraph = {
      id: 'rename', name: 'Rename', nodes: [
        { id: 's', adapterId: 'sequence-input', params: { label: 'before', sequence: 'ACDEFGHIKL' } },
        { id: 'f', adapterId: 'localfold-af2', params: {} },
      ], edges: [{ fromNode: 's', fromPort: 'sequences', toNode: 'f', toPort: 'sequences' }],
    };
    const before = await scheduler.run(graph, () => {});
    graph.nodes[0].params.label = 'after';
    const after = await scheduler.run(graph, () => {});
    expect(after.outputsByNode.s.sequences.id).not.toBe(before.outputsByNode.s.sequences.id);
    expect(mockWorkerRunCalls).toBe(2);
    await scheduler.releaseActiveGpu();
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
            params: { label: 'helix-1', sequence: '  mktay iakqr qisfv  ' },
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
    ).rejects.toThrow(/Enter at least one amino acid sequence/);

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
    ).rejects.toThrow(/Invalid residue/);
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
    expect(c1.conditions.C_SEQ).toEqual({
      true: [
        { select: "sel('A/*/*/*')" },
        { select: "sel('A/*/[16-35,52-71]/*')" },
      ],
    });
    expect(c1.conditions.C_HOT).toEqual({
      true: [{ select: "sel('A/*/[54,56]/*') | sel('B/*/[10-12]/*')" }],
    });
    expect(c1.conditions.C_CRDGRP).toEqual({
      1: [{ select: "sel('A/*/[16-35,52-71]/*')" }],
    });
    expect(c1.conditions.C_IDX).toEqual({
      false: [{ select: "sel('A/*/[16-35,52-71]/*')", group: 'motif' }],
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

  it('deduplicates duplicate FASTA headers, busts cond-raw cache on template change, handles scorer length mismatch & NaN sorting, and releases GPU mid-run without deadlock', async () => {
    const scheduler = new ExecutionScheduler();

    // 1. Duplicate FASTA headers -> dup, dup_2, dup_3
    const dupFastaRes = await scheduler.run(
      {
        id: 'dup-fasta',
        name: 'Dup FASTA',
        nodes: [
          {
            id: 's',
            adapterId: 'sequence-input',
            params: {
              sequence: '>dup\nAAAAA\n>dup\nGGGGG\n>dup\nLLLLL',
            },
          },
        ],
        edges: [],
      },
      () => {},
    );
    const dupSet = dupFastaRes.outputsByNode.s.sequences as SequenceSet;
    expect(dupSet.candidates.map((c) => c.id)).toEqual(['dup', 'dup_2', 'dup_3']);

    // 2. cond-raw artifact id includes upstream template.id (cache busting when template changes)
    const pdb1 = [
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  ',
      'ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C  ',
      'ATOM      3  CA  ALA A   3       7.600   0.000   0.000  1.00  0.00           C  ',
      'END',
    ].join('\n');
    const pdb2 = pdb1.replace('7.600', '7.700');
    const rawJson = JSON.stringify({ contig: '10-10', conditions: {} });
    const runCondRaw = (pdbText: string) =>
      scheduler.run(
        {
          id: 'cond-raw-tpl',
          name: 'Cond Raw Tpl',
          nodes: [
            { id: 't', adapterId: 'pdb-input', params: { mode: 'custom', pdbText } },
            { id: 'c', adapterId: 'rfd4-conditioner', params: { rawJsonOverride: rawJson } },
          ],
          edges: [{ fromNode: 't', fromPort: 'structure', toNode: 'c', toPort: 'template' }],
        },
        () => {},
      );
    const rCond1 = await runCondRaw(pdb1);
    const rCond2 = await runCondRaw(pdb2);
    const art1 = rCond1.outputsByNode.c.condition as ConditionSpecArtifact;
    const art2 = rCond2.outputsByNode.c.condition as ConditionSpecArtifact;
    expect(art1.id).not.toBe(art2.id);
    expect(art1.inputCif).toBe(pdb1);
    expect(art2.inputCif).toBe(pdb2);

    // 3. designability-scorer rejects length mismatch with clear error
    const demoPdb = generateSampleHelixBundlePdb();
    const demoL = parsePdbBackbone(demoPdb).nResidues;
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-mismatch',
      parentStructureId: 'demo',
      folds: [
        {
          sequenceId: 'wrong-len',
          sequence: 'A'.repeat(10),
          foldModelId: 'localfold-af2',
          pdbText: demoPdb,
          caCoords: new Float32Array(10 * 3),
          plddt: new Float32Array(10).fill(90),
          meanPlddt: 90,
          ptm: 0.9,
          elapsedMs: 1,
        },
      ],
    });
    const scoreGraph = (id: string): WorkflowGraph => ({
      id,
      name: id,
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'A'.repeat(demoL) } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 2, seed: 0 } },
        { id: 'score', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    });
    await expect(scheduler.run(scoreGraph('mismatch'), () => {}, { bypassCache: true })).rejects.toThrow(
      /Length mismatch/,
    );

    // 4. designability-scorer sorts NaN scRMSD last and marks designable = false
    const demoCa = parsePdbBackbone(demoPdb).caCoords;
    const nanCa = new Float32Array(demoCa);
    nanCa[0] = Number.NaN;
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-nan',
      parentStructureId: 'demo',
      folds: [
        {
          sequenceId: 'nan-cand',
          sequence: 'A'.repeat(demoL),
          foldModelId: 'localfold-af2',
          pdbText: demoPdb,
          caCoords: nanCa,
          plddt: new Float32Array(demoL).fill(95),
          meanPlddt: 95,
          ptm: 0.9,
          elapsedMs: 1,
        },
        {
          sequenceId: 'good-cand',
          sequence: 'L'.repeat(demoL),
          foldModelId: 'localfold-af2',
          pdbText: demoPdb,
          caCoords: new Float32Array(demoCa),
          plddt: new Float32Array(demoL).fill(90),
          meanPlddt: 90,
          ptm: 0.9,
          elapsedMs: 1,
        },
      ],
    });
    const nanRes = await scheduler.run(scoreGraph('nan-sort'), () => {}, { bypassCache: true });
    const nanMetrics = nanRes.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(nanMetrics.rows[0].sequenceId).toBe('good-cand');
    expect(nanMetrics.rows[0].designable).toBe(true);
    expect(nanMetrics.rows[1].sequenceId).toBe('nan-cand');
    expect(Number.isNaN(nanMetrics.rows[1].scRmsd)).toBe(true);
    expect(nanMetrics.rows[1].designable).toBe(false);

    // 5. releaseActiveGpu() mid-run rejects the active promise cleanly and leaves scheduler reusable (no deadlock)
    mockHangWorker = true;
    const inflight = scheduler.run(
      {
        id: 'hang-test',
        name: 'Hang',
        nodes: [
          { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'ACDEFGHIKLMNPQRSTVWY' } },
          { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1, seed: 99 } },
        ],
        edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
      },
      () => {},
      { bypassCache: true },
    );
    // Let runGpuNode spawn and postMessage before releasing GPU
    await Promise.resolve();
    await Promise.resolve();
    const releasePromise = scheduler.releaseActiveGpu();
    await expect(inflight).rejects.toThrow(/released while running/i);
    await releasePromise;

    // Scheduler must NOT be deadlocked after mid-run release
    mockHangWorker = false;
    const afterRes = await scheduler.run(
      {
        id: 'after-release',
        name: 'After Release',
        nodes: [{ id: 's', adapterId: 'sequence-input', params: { sequence: 'ACDEFGHIKL' } }],
        edges: [],
      },
      () => {},
    );
    expect((afterRes.outputsByNode.s.sequences as SequenceSet).candidates).toHaveLength(1);
  });

  it('streams intermediate node outputs live via onNodeOutput in topological order', async () => {
    const scheduler = new ExecutionScheduler();
    const streamed: Array<{ nodeId: string; kinds: string[] }> = [];
    const graph: WorkflowGraph = {
      id: 'stream-test',
      name: 'Stream Test',
      nodes: [
        { id: 'cond', adapterId: 'rfd4-conditioner', params: { taskPreset: 'custom', contig: 'A1-53' } },
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
      ],
      edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'cond', toPort: 'template' }],
    };

    await scheduler.run(graph, () => {}, {
      onNodeOutput: (nodeId, outs) => {
        streamed.push({ nodeId, kinds: Object.values(outs).map((a) => a.kind) });
      },
    });

    expect(streamed).toEqual([
      { nodeId: 'src', kinds: ['StructureArtifact'] },
      { nodeId: 'cond', kinds: ['ConditionSpec'] },
    ]);
  });

  it('keeps pdbText empty for pure ligand/DNA mmCIF (#16), preserves resNums for discontinuous motifs (#21), and preserves holo cifText in scorer', async () => {
    const scheduler = new ExecutionScheduler();

    // 1. Pure DNA mmCIF (nResidues === 0) must NOT shove raw mmCIF into pdbText
    const pureDnaCif = [
      'data_7M5W',
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
      'ATOM 3 P DC B 1 7.000 8.000 9.000',
      '#',
    ].join('\n');
    const dnaRes = await scheduler.run(
      {
        id: 'pure-dna',
        name: 'Pure DNA',
        nodes: [{ id: 'p', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: pureDnaCif, filename: '7M5W.cif' } }],
        edges: [],
      },
      () => {},
    );
    const dnaStruct = dnaRes.outputsByNode.p.structure as StructureArtifact;
    expect(dnaStruct.length).toBe(0);
    expect(dnaStruct.pdbText).toBe('');
    expect(dnaStruct.cifText).toBe(pureDnaCif);
    expect(dnaStruct.diagnostics?.hasNucleicAcids).toBe(true);

    // 2. Discontinuous motif PDB (gap A10..12 -> A50..52 separated by 25 Å) reports 0 false chain breaks (#21)
    const gappedPdb = [
      'ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00  0.00           C  ',
      'ATOM      2  CA  GLY A  11       3.800   0.000   0.000  1.00  0.00           C  ',
      'ATOM      3  CA  VAL A  12       7.600   0.000   0.000  1.00  0.00           C  ',
      'ATOM      4  CA  LEU A  50      35.000   0.000   0.000  1.00  0.00           C  ',
      'ATOM      5  CA  ILE A  51      38.800   0.000   0.000  1.00  0.00           C  ',
      'ATOM      6  CA  SER A  52      42.600   0.000   0.000  1.00  0.00           C  ',
      'END',
    ].join('\n');
    const gapRes = await scheduler.run(
      {
        id: 'gapped-motif',
        name: 'Gapped Motif',
        nodes: [{ id: 'p', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: gappedPdb } }],
        edges: [],
      },
      () => {},
    );
    const gapStruct = gapRes.outputsByNode.p.structure as StructureArtifact;
    expect(gapStruct.geometry.chainBreaks).toBe(0);

    // 3. Holo mmCIF (protein + ligand SAM) preserves cifText through designability-scorer
    const holoCif = [
      'data_holo',
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
      'ATOM 1 CA ALA A 1 0.000 0.000 0.000',
      'ATOM 2 CA GLY A 2 3.800 0.000 0.000',
      'ATOM 3 CA VAL A 3 7.600 1.200 0.000',
      'ATOM 4 CA LEU A 4 11.400 0.000 1.000',
      'ATOM 5 CA ILE A 5 15.200 1.500 0.500',
      'HETATM 6 C1 SAM B 101 5.000 6.000 2.000',
      '#',
    ].join('\n');
    const holoCa = new Float32Array([0, 0, 0, 3.8, 0, 0, 7.6, 1.2, 0, 11.4, 0, 1, 15.2, 1.5, 0.5]);
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'holo-folds',
      parentStructureId: 'holo',
      folds: [
        {
          sequenceId: 'holo-1',
          sequence: 'AGVLI',
          foldModelId: 'localfold-af2',
          pdbText: gappedPdb,
          caCoords: holoCa,
          plddt: new Float32Array(5).fill(90),
          meanPlddt: 90,
          ptm: 0.85,
          elapsedMs: 10,
        },
      ],
    });
    const holoRes = await scheduler.run(
      {
        id: 'holo-score',
        name: 'Holo Score',
        nodes: [
          { id: 'p', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: holoCif, filename: 'holo.cif' } },
          { id: 's', adapterId: 'sequence-input', params: { sequence: 'AGVLI' } },
          { id: 'f', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1, seed: 0 } },
          { id: 'sc', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
        ],
        edges: [
          { fromNode: 's', fromPort: 'sequences', toNode: 'f', toPort: 'sequences' },
          { fromNode: 'p', fromPort: 'structure', toNode: 'sc', toPort: 'structure' },
          { fromNode: 'f', fromPort: 'folds', toNode: 'sc', toPort: 'folds' },
        ],
      },
      () => {},
    );
    const holoMetrics = holoRes.outputsByNode.sc.metrics as DesignabilityMetrics;
    expect(holoMetrics.rows[0].designedPdb).toContain('data_holo');
    expect(holoMetrics.rows[0].designedPdb).toContain('SAM');
    await scheduler.releaseActiveGpu();
  });

  it('computes multi-chain binder metrics (iPAE, binder/target pLDDT, binderScRmsd, targetAlignedBinderRmsd) and gates designability (#8, #36)', async () => {
    const scheduler = new ExecutionScheduler();

    // Build a realistic 2-chain complex: Target A (16 aa helix) + Binder B (8 aa helix)
    const makeHelix = (n: number, ox: number, oy: number, oz: number): Float32Array => {
      const c = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const theta = i * 1.745;
        c[i * 3 + 0] = ox + 2.3 * Math.cos(theta);
        c[i * 3 + 1] = oy + 2.3 * Math.sin(theta);
        c[i * 3 + 2] = oz + i * 1.5;
      }
      return c;
    };
    const nTarget = 16;
    const nBinder = 8;
    const L = nTarget + nBinder;
    const targetCa = makeHelix(nTarget, 0, 0, 0);
    const binderCa = makeHelix(nBinder, 10, 0, 4);
    const complexCa = new Float32Array(L * 3);
    complexCa.set(targetCa, 0);
    complexCa.set(binderCa, nTarget * 3);
    const chainIds = [...Array(nTarget).fill('A'), ...Array(nBinder).fill('B')];

    const toPdb = (ca: Float32Array, chains: string[]): string => {
      const lines: string[] = [];
      let serial = 1;
      for (let i = 0; i < chains.length; i++) {
        const ch = chains[i];
        const res = i < nTarget ? i + 1 : i - nTarget + 1;
        const x = ca[i * 3].toFixed(3).padStart(8);
        const y = ca[i * 3 + 1].toFixed(3).padStart(8);
        const z = ca[i * 3 + 2].toFixed(3).padStart(8);
        lines.push(
          `ATOM  ${String(serial++).padStart(5)}  CA  ALA ${ch}${String(res).padStart(4)}    ${x}${y}${z}  1.00 85.00           C  `,
        );
      }
      lines.push('END');
      return lines.join('\n');
    };
    const complexPdb = toPdb(complexCa, chainIds);

    // Candidate 1 (PASS): docked binder (rigidly rotated complex), low iPAE = 4.0, high binder pLDDT = 86
    const R90 = new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 1]);
    const tShift = new Float64Array([12, -5, 3]);
    const dockedPdb = applyTransformToPdb(complexPdb, R90, tShift);
    const dockedCa = parsePdbBackbone(dockedPdb).caCoords;

    // Candidate 2 (FAIL: dissociated binder): target identical, binder keeps internal fold (binderScRmsd ~ 0)
    // but floats +18 Å away in X (targetAlignedBinderRmsd = 18 Å, iPAE = 20 Å)
    const dissocCa = new Float32Array(complexCa);
    for (let i = nTarget; i < L; i++) dissocCa[i * 3 + 0] += 18.0;
    const dissocPdb = toPdb(dissocCa, chainIds);

    // Candidate 3 (FAIL: low binder pLDDT even though large target keeps overall meanPlddt >= 70)
    const lowBinderPlddt = new Float32Array(L);
    for (let i = 0; i < nTarget; i++) lowBinderPlddt[i] = 95;
    for (let i = nTarget; i < L; i++) lowBinderPlddt[i] = 55;
    // Overall mean = (16*95 + 8*55) / 24 = 81.67 >= 70!

    const makePae = (intra: number, inter: number): Float32Array => {
      const m = new Float32Array(L * L);
      for (let i = 0; i < L; i++) {
        for (let j = 0; j < L; j++) {
          const same = (i < nTarget) === (j < nTarget);
          m[i * L + j] = same ? intra : inter;
        }
      }
      return m;
    };

    mockGenStructureFactory = () => ({
      kind: 'StructureArtifact',
      id: 'binder-gen-1',
      label: 'Binder Complex',
      sourceModelId: 'rfd4-proteina',
      length: L,
      sequence: 'A'.repeat(L),
      caCoords: complexCa,
      backbone4Coords: new Float32Array(L * 4 * 3),
      pdbText: complexPdb,
      chainIds,
      chainLengths: [nTarget, nBinder],
      chainRoles: { A: 'target', B: 'binder' },
      lockedChains: ['A'],
      geometry: { length: L, rg: 12, meanCaDist: 3.8, chainBreaks: 0, clashes: 0, helixPct: 80, sheetPct: 0, coilPct: 20 },
    });

    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'binder-folds-1',
      parentStructureId: 'binder-gen-1',
      folds: [
        {
          sequenceId: 'docked-pass',
          sequence: 'A'.repeat(L),
          foldModelId: 'localfold-multimer',
          pdbText: dockedPdb,
          caCoords: dockedCa,
          plddt: new Float32Array([...Array(nTarget).fill(92), ...Array(nBinder).fill(86)]),
          meanPlddt: 90,
          ptm: 0.88,
          iptm: 0.84,
          pae: makePae(2.0, 4.0),
          chainLengths: [nTarget, nBinder],
          chainIds,
          elapsedMs: 50,
        },
        {
          sequenceId: 'dissociated-fail',
          sequence: 'G'.repeat(L),
          foldModelId: 'localfold-multimer',
          pdbText: dissocPdb,
          caCoords: dissocCa,
          plddt: new Float32Array([...Array(nTarget).fill(92), ...Array(nBinder).fill(86)]),
          meanPlddt: 90,
          ptm: 0.72,
          iptm: 0.25,
          pae: makePae(2.0, 20.0),
          chainLengths: [nTarget, nBinder],
          chainIds,
          elapsedMs: 55,
        },
        {
          sequenceId: 'low-binder-plddt-fail',
          sequence: 'L'.repeat(L),
          foldModelId: 'localfold-multimer',
          pdbText: dockedPdb,
          caCoords: dockedCa,
          plddt: lowBinderPlddt,
          meanPlddt: 81.67,
          ptm: 0.78,
          iptm: 0.65,
          pae: makePae(2.0, 5.0),
          chainLengths: [nTarget, nBinder],
          chainIds,
          elapsedMs: 60,
        },
      ],
    });

    const graph: WorkflowGraph = {
      id: 'binder-score-test',
      name: 'Binder Score Test',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 70 } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `${'A'.repeat(nTarget)}:${'A'.repeat(nBinder)}` } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'multimer', recycles: 1, maxSequences: 3, seed: 0 } },
        { id: 'score', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };

    const res = await scheduler.run(graph, () => {});
    const metrics = res.outputsByNode.score.metrics as DesignabilityMetrics;
    const byId = new Map(metrics.rows.map((r) => [r.sequenceId, r]));

    const docked = byId.get('docked-pass')!;
    expect(docked.designable).toBe(true);
    expect(docked.ipae).toBeCloseTo(4.0, 4);
    expect(docked.binderPlddt).toBeCloseTo(86, 4);
    expect(docked.targetPlddt).toBeCloseTo(92, 4);
    expect(docked.binderScRmsd!).toBeLessThan(1e-2);
    expect(docked.targetAlignedBinderRmsd!).toBeLessThan(1e-2);

    const dissoc = byId.get('dissociated-fail')!;
    expect(dissoc.binderScRmsd!).toBeLessThan(1e-2);
    expect(dissoc.targetAlignedBinderRmsd!).toBeCloseTo(18.0, 1);
    expect(dissoc.ipae).toBeCloseTo(20.0, 4);
    expect(dissoc.designable).toBe(false);

    const lowBinder = byId.get('low-binder-plddt-fail')!;
    expect(lowBinder.meanPlddt).toBeGreaterThan(70);
    expect(lowBinder.binderPlddt).toBeCloseTo(55, 4);
    expect(lowBinder.designable).toBe(false);

    // summarizeArtifact surfaces ipTM, iPAE, binderScRmsd, targetAlignedBinderRmsd (#36)
    const refSummary = res.refsByNode.score.metrics.summary;
    expect(refSummary.bestIptm).toBeCloseTo(0.84, 2);
    expect(refSummary.bestIpae).toBeCloseTo(4.0, 1);
    expect(typeof refSummary.bestBinderScRmsd).toBe('number');
    expect(typeof refSummary.bestTargetAlignedBinderRmsd).toBe('number');
    expect(res.refsByNode.fold.folds.summary.bestIptm).toBeCloseTo(0.84, 2);

    await scheduler.releaseActiveGpu();
  });

  it('computes genMotifRmsd & refoldMotifRmsd and gates motif pass status at < 1.0 Å (#10)', async () => {
    const scheduler = new ExecutionScheduler();
    const basePdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(basePdb);
    const L = parsed.nResidues;
    const motifIndices = [5, 6, 7, 8, 9, 10];
    const nativeMotifCoords = motifIndices.map((idx) => [
      parsed.caCoords[idx * 3 + 0],
      parsed.caCoords[idx * 3 + 1],
      parsed.caCoords[idx * 3 + 2],
    ]);

    // Refold 1 (PASS): exact scaffold + motif => refoldMotifRmsd ~ 0
    const goodCa = new Float32Array(parsed.caCoords);
    // Refold 2 (FAIL motif): scaffold mostly intact (full scRMSD < 2.0 Å),
    // but motif residues distorted by ~2.5 Å => refoldMotifRmsd > 1.0 Å -> designable = false!
    const badMotifCa = new Float32Array(parsed.caCoords);
    for (let k = 0; k < motifIndices.length; k++) {
      const idx = motifIndices[k];
      badMotifCa[idx * 3 + 0] += k % 2 === 0 ? 2.5 : -2.5;
      badMotifCa[idx * 3 + 1] += k % 3 === 0 ? 1.8 : -1.8;
    }

    mockGenStructureFactory = () => ({
      kind: 'StructureArtifact',
      id: 'motif-gen-1',
      label: 'Motif Scaffold',
      sourceModelId: 'genie3',
      length: L,
      sequence: parsed.sequence,
      caCoords: parsed.caCoords,
      backbone4Coords: parsed.backbone4Coords,
      pdbText: basePdb,
      chainIds: parsed.chainIds,
      chainLengths: [L],
      motifIndices,
      nativeMotifCoords,
      geometry: { length: L, rg: 12, meanCaDist: 3.8, chainBreaks: 0, clashes: 0, helixPct: 80, sheetPct: 0, coilPct: 20 },
    });

    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'motif-folds-1',
      parentStructureId: 'motif-gen-1',
      folds: [
        {
          sequenceId: 'motif-pass',
          sequence: parsed.sequence,
          foldModelId: 'localfold-af2',
          pdbText: basePdb,
          caCoords: goodCa,
          plddt: new Float32Array(L).fill(92),
          meanPlddt: 92,
          ptm: 0.89,
          elapsedMs: 40,
        },
        {
          sequenceId: 'motif-distorted',
          sequence: parsed.sequence,
          foldModelId: 'localfold-af2',
          pdbText: basePdb,
          caCoords: badMotifCa,
          plddt: new Float32Array(L).fill(92),
          meanPlddt: 92,
          ptm: 0.85,
          elapsedMs: 45,
        },
      ],
    });

    const graph: WorkflowGraph = {
      id: 'motif-score-test',
      name: 'Motif Score Test',
      nodes: [
        { id: 'gen', adapterId: 'genie3', params: { length: L } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: parsed.sequence } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 2, seed: 0 } },
        { id: 'score', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };

    const res = await scheduler.run(graph, () => {});
    const metrics = res.outputsByNode.score.metrics as DesignabilityMetrics;
    const byId = new Map(metrics.rows.map((r) => [r.sequenceId, r]));

    const passRow = byId.get('motif-pass')!;
    expect(passRow.genMotifRmsd!).toBeLessThan(1e-3);
    expect(passRow.refoldMotifRmsd!).toBeLessThan(1e-3);
    expect(passRow.designable).toBe(true);

    const badRow = byId.get('motif-distorted')!;
    expect(badRow.scRmsd).toBeLessThan(2.0);
    expect(badRow.genMotifRmsd!).toBeLessThan(1e-3);
    expect(badRow.refoldMotifRmsd!).toBeGreaterThan(1.0);
    expect(badRow.designable).toBe(false);

    expect(typeof res.refsByNode.score.metrics.summary.bestGenMotifRmsd).toBe('number');
    expect(typeof res.refsByNode.score.metrics.summary.bestRefoldMotifRmsd).toBe('number');
    await scheduler.releaseActiveGpu();
  });
});
