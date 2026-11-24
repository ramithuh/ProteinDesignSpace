import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { applyTransformToPdb } from '$lib/protein-ops/kabsch';
import { generateSampleHelixBundlePdb, parsePdbBackbone } from '$lib/protein-ops/pdb';
import {
  ExecutionScheduler,
  RunCancellationError,
  RunLeaseError,
  attachProgrammaticApi,
  computeInvalidatedNodeIds,
  exportJoinedCollectionCsv,
  joinBranchArtifacts,
  materializePdbInputNode,
  serializeCheckpoint,
  deserializeCheckpoint,
  type ActiveRunState,
} from '$lib/engine/scheduler';
import { createMemoryCheckpointStore } from '$lib/engine/opfs-cache';
import type {
  ConditionSpecArtifact,
  DesignabilityMetrics,
  FoldedStructure,
  NodeProgressEvent,
  SequenceSet,
  StructureArtifact,
  TrajectoryFramesArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

// Controllable Worker stub so designability-scorer, memoization, output contracts & releaseActiveGpu can be tested end-to-end
let mockFoldFactory: (() => FoldedStructure) | null = null;
let mockGenStructureFactory: (() => StructureArtifact) | null = null;
let mockRawOutputsFactory: ((msg: { id: number; params?: Record<string, unknown>; inputs?: Record<string, unknown> }) => unknown) | null = null;
let mockWorkerDelayMs = 0;
let mockDisposeDelayMs = 0;
let mockLiveWorkers = 0;
let mockMaxLiveWorkers = 0;
let mockWorkerRunCalls = 0;
let mockHangWorker = false;

function makePlaceholderStructure(id: string, seq = 'ACDEFGHIKL'): StructureArtifact {
  const L = Math.max(3, seq.length);
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'mock-worker',
    length: L,
    sequence: seq.slice(0, L),
    caCoords: new Float32Array(L * 3),
    backbone4Coords: new Float32Array(L * 4 * 3),
    pdbText: 'END\n',
    geometry: {
      length: L,
      rg: 10,
      meanCaDist: 3.8,
      chainBreaks: 0,
      clashes: 0,
      helixPct: 50,
      sheetPct: 0,
      coilPct: 50,
    },
  };
}

class MockWorker {
  private listeners = new Map<string, Set<(ev: MessageEvent) => void>>();
  private terminated = false;
  constructor() {
    mockLiveWorkers++;
    if (mockLiveWorkers > mockMaxLiveWorkers) mockMaxLiveWorkers = mockLiveWorkers;
  }
  addEventListener(type: string, cb: (ev: MessageEvent) => void) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(cb);
  }
  removeEventListener(type: string, cb: (ev: MessageEvent) => void) {
    this.listeners.get(type)?.delete(cb);
  }
  postMessage(msg: { id: number; type: string; params?: Record<string, unknown>; inputs?: Record<string, unknown> }) {
    if (msg.type === 'dispose') {
      const reply = () => {
        this.emit('message', { data: { id: msg.id, type: 'disposed' } } as MessageEvent);
      };
      if (mockDisposeDelayMs > 0) setTimeout(reply, mockDisposeDelayMs);
      else reply();
      return;
    }
    if (msg.type === 'run') {
      mockWorkerRunCalls++;
      if (mockHangWorker) return; // intentionally stays in-flight to test mid-run lease & releaseActiveGpu()
      const deliver = (outputs: unknown) => {
        const send = () => {
          this.emit('message', {
            data: { id: msg.id, type: 'result', outputs, elapsedMs: 5 },
          } as MessageEvent);
        };
        if (mockWorkerDelayMs > 0) setTimeout(send, mockWorkerDelayMs);
        else queueMicrotask(send);
      };
      if (mockRawOutputsFactory) {
        deliver(mockRawOutputsFactory(msg));
        return;
      }
      const hasSeqInput = Boolean(msg.inputs && 'sequences' in msg.inputs);
      if (!hasSeqInput && mockGenStructureFactory) {
        const structure = mockGenStructureFactory();
        const sequences: SequenceSet = {
          kind: 'SequenceSet',
          id: `mock-seq-${structure.id}`,
          parentStructureId: structure.id,
          sourceStructureId: structure.id,
          sourceModelId: structure.sourceModelId,
          candidates: [
            {
              id: 'mock-native-1',
              key: 'mock-native-1',
              sequence: structure.sequence || 'ACDEFGHIKL',
              score: null,
              globalScore: null,
              sourceModelId: structure.sourceModelId,
            },
          ],
        };
        deliver({ structure, sequences });
        return;
      }
      const folds = mockFoldFactory ? mockFoldFactory() : {
        kind: 'FoldedStructure' as const,
        id: 'mock-folds',
        parentStructureId: 'none',
        folds: [],
      };
      const repFold = folds.folds[0];
      const repStruct = makePlaceholderStructure(
        `mock-foldstruct-${folds.id}`,
        repFold?.sequence || 'ACDEFGHIKL',
      );
      deliver({ folds, structure: repStruct });
    }
  }
  terminate() {
    if (!this.terminated) {
      this.terminated = true;
      mockLiveWorkers = Math.max(0, mockLiveWorkers - 1);
    }
  }
  private emit(type: string, ev: MessageEvent) {
    for (const cb of this.listeners.get(type) ?? []) cb(ev);
  }
}

describe('engine/scheduler (CPU nodes & memoization)', () => {
  beforeEach(() => {
    mockFoldFactory = null;
    mockGenStructureFactory = null;
    mockRawOutputsFactory = null;
    mockWorkerDelayMs = 0;
    mockDisposeDelayMs = 0;
    mockLiveWorkers = 0;
    mockMaxLiveWorkers = 0;
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
    ).rejects.toThrow(/at least 3 (?:Cα )?residues/);
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

  it('rejects removed legacy rfd4-conditioner adapter cleanly at preflight', async () => {
    const scheduler = new ExecutionScheduler();
    await expect(
      scheduler.run(
        {
          id: 'legacy-cond',
          name: 'Legacy Cond',
          nodes: [{ id: 'cond', adapterId: 'rfd4-conditioner', params: { taskPreset: 'custom', contig: '70-70' } }],
          edges: [],
        },
        () => {},
      ),
    ).rejects.toThrow(/Unknown model "rfd4-conditioner"/);
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
    const demoRef = await materializePdbInputNode({
      id: 'src',
      adapterId: 'pdb-input',
      params: { mode: 'demo-bundle' },
    });
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-test-1',
      parentStructureId: demoRef.id,
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

  it('reuses an ESMFold2 worker for the same checkpoint and releases it when the weight pair changes', async () => {
    const scheduler = new ExecutionScheduler();
    const graph: WorkflowGraph = {
      id: 'fold-switch', name: 'Switch folding models',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'ACDEFGHIKL' } },
        { id: 'fold', adapterId: 'localfold-af2', params: {} },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const terminate = vi.spyOn(MockWorker.prototype, 'terminate');
    try {
      for (const [model, releases] of [
        ['monomer', 0], ['ef2-fast-300m', 1], ['ef2-fast-300m', 1], ['ef2-fast-600m', 2], ['monomer', 3],
      ] as const) {
        graph.nodes[1].params.model = model;
        await scheduler.run(graph, () => {}, { bypassCache: true });
        expect(terminate).toHaveBeenCalledTimes(releases);
      }
      expect(mockMaxLiveWorkers).toBe(1);
    } finally {
      await scheduler.releaseActiveGpu();
      terminate.mockRestore();
    }
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
            { id: 'a', adapterId: 'rfd4-proteina', params: {} },
            { id: 'b', adapterId: 'rfd4-proteina', params: {} },
          ],
          edges: [
            { fromNode: 'a', fromPort: 'structure', toNode: 'b', toPort: 'structure' },
            { fromNode: 'b', fromPort: 'structure', toNode: 'a', toPort: 'structure' },
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

    // 2. pdb-input artifact id changes when custom PDB coordinates change (cache busting)
    const pdb1 = [
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  ',
      'ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C  ',
      'ATOM      3  CA  ALA A   3       7.600   0.000   0.000  1.00  0.00           C  ',
      'END',
    ].join('\n');
    const pdb2 = pdb1.replace('7.600', '7.700');
    const runTpl = (pdbText: string) =>
      scheduler.run(
        {
          id: 'tpl-bust',
          name: 'Tpl Bust',
          nodes: [{ id: 't', adapterId: 'pdb-input', params: { mode: 'custom', pdbText } }],
          edges: [],
        },
        () => {},
      );
    const rTpl1 = await runTpl(pdb1);
    const rTpl2 = await runTpl(pdb2);
    const art1 = rTpl1.outputsByNode.t.structure as StructureArtifact;
    const art2 = rTpl2.outputsByNode.t.structure as StructureArtifact;
    expect(art1.id).not.toBe(art2.id);
    expect(art1.pdbText).toBe(pdb1);
    expect(art2.pdbText).toBe(pdb2);

    // 3. designability-scorer rejects length mismatch with clear error
    const demoPdb = generateSampleHelixBundlePdb();
    const demoL = parsePdbBackbone(demoPdb).nResidues;
    const demoRef = await materializePdbInputNode({
      id: 'src',
      adapterId: 'pdb-input',
      params: { mode: 'demo-bundle' },
    });
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-mismatch',
      parentStructureId: demoRef.id,
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
      parentStructureId: demoRef.id,
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

    // 5. Default releaseActiveGpu() respects active run lease; { force: true } releases mid-run cleanly without deadlock
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
    // Let runGpuNode spawn and postMessage before testing lease guard & forced release
    await Promise.resolve();
    await Promise.resolve();
    await expect(scheduler.releaseActiveGpu()).rejects.toThrow(RunLeaseError);
    expect(scheduler.getActiveRunState().leaseHeld).toBe(true);
    const releasePromise = scheduler.releaseActiveGpu({ force: true });
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
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer' } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFS' } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };

    await scheduler.run(graph, () => {}, {
      onNodeOutput: (nodeId, outs) => {
        streamed.push({ nodeId, kinds: Object.values(outs).map((a) => a.kind) });
      },
    });

    expect(streamed).toEqual([
      { nodeId: 'seq', kinds: ['SequenceSet'] },
      { nodeId: 'fold', kinds: ['FoldedStructure', 'StructureArtifact'] },
    ]);
    await scheduler.releaseActiveGpu();
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
    expect(dnaStruct.pdbText).toContain('ATOM');
    expect(dnaStruct.pdbText).not.toContain('data_pure_dna');
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

  it('Issue #3: flags lineage mismatch when fold descended from A is wired to equal-length reference B, and supports explicit residueMapping & allowExternalReference', async () => {
    const scheduler = new ExecutionScheduler();
    const pdbA = generateSampleHelixBundlePdb();
    const parsedA = parsePdbBackbone(pdbA);
    const L = parsedA.nResidues;

    // Equal-length different reference B (shifted coordinates -> distinct artifact id)
    const R90 = new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 1]);
    const pdbB = applyTransformToPdb(pdbA, R90, new Float64Array([15, -8, 4]));
    const structB = await materializePdbInputNode({
      id: 'refB',
      adapterId: 'pdb-input',
      params: { mode: 'custom', pdbText: pdbB, filename: 'refB.pdb' },
    });

    // Fold descended from structure A ("struct-origin-A"), coordinates identical to A/B up to rigid transform
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-from-A',
      parentStructureId: 'struct-origin-A',
      sourceStructureId: 'struct-origin-A',
      folds: [
        {
          sequenceId: 'cand-A1',
          key: 'cand-A1',
          sequence: parsedA.sequence,
          foldModelId: 'localfold-af2',
          pdbText: pdbA,
          caCoords: parsedA.caCoords,
          plddt: new Float32Array(L).fill(94),
          meanPlddt: 94,
          ptm: 0.91,
          elapsedMs: 20,
        },
      ],
    });

    // 1. Mismatched lineage without external-reference override: flagged & non-passing self-consistency!
    const mismatchGraph: WorkflowGraph = {
      id: 'lineage-mismatch',
      name: 'Lineage Mismatch',
      nodes: [
        { id: 'refB', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: pdbB, filename: 'refB.pdb' } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: parsedA.sequence } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1 } },
        { id: 'score', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'refB', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };

    const mmRes = await scheduler.run(mismatchGraph, () => {});
    const mmMetrics = mmRes.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(mmMetrics.referenceStructureId).toBe(structB.id);
    expect(mmMetrics.foldParentStructureId).toBe('struct-origin-A');
    expect(mmMetrics.lineageMismatch).toBe(true);
    expect(mmMetrics.lineageStatus).toBe('mismatch');
    expect(mmMetrics.referenceMode).toBe('lineage-mismatch');
    expect(mmMetrics.lineageWarning).toMatch(/struct-origin-A/);
    expect(mmMetrics.rows[0].lineageMismatch).toBe(true);
    expect(mmMetrics.rows[0].scRmsd).toBeLessThan(1e-2);
    // Must NOT pass self-consistency despite low RMSD and high pLDDT!
    expect(mmMetrics.rows[0].designable).toBe(false);
    expect(mmMetrics.passRate).toBe(0);
    expect(mmRes.refsByNode.score.metrics.summary.lineageMismatch).toBe(true);

    // 2. Intentional external-reference comparison with explicit residueMapping (even a subset of residues!)
    const mappedGraph = structuredClone(mismatchGraph);
    mappedGraph.nodes[3].params.residueMapping = {
      pairs: [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
      ],
      label: 'N-terminal helix mapping',
    };
    const mappedRes = await scheduler.run(mappedGraph, () => {}, { bypassCache: true });
    const mappedMetrics = mappedRes.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(mappedMetrics.lineageMismatch).toBe(false);
    expect(mappedMetrics.lineageStatus).toBe('external-mapped');
    expect(mappedMetrics.referenceMode).toBe('external-reference');
    expect(mappedMetrics.referenceRelation).toBe('external-mapped');
    expect(mappedMetrics.selfConsistencyValid).toBe(false);
    expect(mappedMetrics.referenceMapping?.label).toBe('N-terminal helix mapping');
    expect(mappedMetrics.rows[0].referenceMappingApplied).toBe(true);
    expect(mappedMetrics.rows[0].alignedResidueCount).toBe(6);
    expect(mappedMetrics.rows[0].scRmsd).toBeLessThan(1e-2);
    // Per #81: external-mapped computes RMSD cleanly without claiming self-consistency PASS!
    expect(mappedMetrics.rows[0].designable).toBe(false);
    expect(mappedMetrics.passRate).toBe(0);

    // 3. Explicit allowExternalReference=true without custom mapping still computes RMSD without claiming self-consistency PASS!
    const extGraph = structuredClone(mismatchGraph);
    extGraph.nodes[3].params.allowExternalReference = true;
    const extRes = await scheduler.run(extGraph, () => {}, { bypassCache: true });
    const extMetrics = extRes.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(extMetrics.lineageMismatch).toBe(false);
    expect(extMetrics.lineageStatus).toBe('external-unmapped');
    expect(extMetrics.referenceMode).toBe('external-reference');
    expect(extMetrics.referenceRelation).toBe('external-mismatch');
    expect(extMetrics.selfConsistencyValid).toBe(false);
    expect(extMetrics.rows[0].scRmsd).toBeLessThan(1e-2);
    expect(extMetrics.rows[0].designable).toBe(false);

    // 4. Out-of-bounds, null-coerced, and duplicate residueMappings reject cleanly
    const badMapGraph = structuredClone(mismatchGraph);
    badMapGraph.nodes[3].params.residueMapping = [[0, 0], [1, 1], [999, 2]];
    await expect(scheduler.run(badMapGraph, () => {}, { bypassCache: true })).rejects.toThrow(
      /out of bounds/i,
    );
    const nullMapGraph = structuredClone(mismatchGraph);
    nullMapGraph.nodes[3].params.residueMapping = [[null, 0], [1, 1], [2, 2]];
    await expect(scheduler.run(nullMapGraph, () => {}, { bypassCache: true })).rejects.toThrow(
      /strict finite integer/i,
    );
    const dupMapGraph = structuredClone(mismatchGraph);
    dupMapGraph.nodes[3].params.residueMapping = [[0, 0], [0, 0], [0, 0]];
    await expect(scheduler.run(dupMapGraph, () => {}, { bypassCache: true })).rejects.toThrow(
      /duplicate/i,
    );

    await scheduler.releaseActiveGpu();
  });

  it('Issue #4: validates worker output contract (wrong artifact kind & missing required port) BEFORE memoizing or passing downstream', async () => {
    const scheduler = new ExecutionScheduler();
    const graph: WorkflowGraph = {
      id: 'contract-test',
      name: 'Contract Test',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFS' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1 } },
        { id: 'score', adapterId: 'designability-scorer', params: {} },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };

    // Case 1: Worker returns wrong artifact kind on `folds` port (`SequenceSet` instead of `FoldedStructure`)
    mockRawOutputsFactory = () => ({
      folds: {
        kind: 'SequenceSet',
        id: 'wrong-kind-1',
        parentStructureId: 'none',
        sourceModelId: 'bad-worker',
        candidates: [],
      },
      structure: makePlaceholderStructure('ok-struct'),
    });
    const deliveredNodes1: string[] = [];
    const progressEvents1: NodeProgressEvent[] = [];
    await expect(
      scheduler.run(graph, (ev) => progressEvents1.push(ev), {
        onNodeOutput: (nid) => deliveredNodes1.push(nid),
      }),
    ).rejects.toThrow(/Output contract violation.*expected "FoldedStructure"/i);

    // Must NOT pass bad output downstream and must NOT memoize `fold`!
    expect(deliveredNodes1).not.toContain('fold');
    expect(deliveredNodes1).not.toContain('score');
    expect(progressEvents1.some((e) => e.nodeId === 'fold' && e.status === 'error')).toBe(true);
    const memoAfterBadKind = scheduler.getMemoStats().entryCount;

    // Case 2: Worker omits required output port (`structure` missing on `localfold-af2`)
    mockRawOutputsFactory = () => ({
      folds: {
        kind: 'FoldedStructure',
        id: 'missing-port-folds',
        parentStructureId: 'none',
        folds: [],
      },
    });
    const deliveredNodes2: string[] = [];
    await expect(
      scheduler.run(graph, () => {}, {
        onNodeOutput: (nid) => deliveredNodes2.push(nid),
      }),
    ).rejects.toThrow(/Output contract violation.*missing required output port "structure"/i);

    expect(deliveredNodes2).not.toContain('fold');
    expect(deliveredNodes2).not.toContain('score');
    // Memo count did not increase with invalid `fold` output
    expect(scheduler.getMemoStats().entryCount).toBe(memoAfterBadKind);

    await scheduler.releaseActiveGpu();
  });

  it('Issue #5: handles confidence-less FoldedCandidate honestly in scorer and serializes TrajectoryFrames in checkpoints', async () => {
    const scheduler = new ExecutionScheduler();
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const demoRef = await materializePdbInputNode({
      id: 'src',
      adapterId: 'pdb-input',
      params: { mode: 'demo-bundle' },
    });

    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-no-conf',
      parentStructureId: demoRef.id,
      collectionKind: 'multi-sample',
      folds: [
        {
          sequenceId: 'sample-noconf',
          key: 'sample-noconf',
          sequence: parsed.sequence,
          foldModelId: 'ensemble-sampler',
          pdbText: pdb,
          caCoords: parsed.caCoords,
          plddt: new Float32Array(0),
          meanPlddt: Number.NaN,
          confidence: null,
          confidenceAvailable: false,
          ptm: null,
          elapsedMs: 15,
        },
      ],
    });

    const graph: WorkflowGraph = {
      id: 'no-conf-score',
      name: 'No Conf Score',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: parsed.sequence } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1 } },
        { id: 'score', adapterId: 'designability-scorer', params: {} },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };

    const res = await scheduler.run(graph, () => {});
    const metrics = res.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(metrics.rows[0].confidenceAvailable).toBe(false);
    expect(Number.isNaN(metrics.rows[0].meanPlddt)).toBe(true);
    expect(metrics.rows[0].designable).toBe(false);
    expect(res.refsByNode.fold.folds.summary.confidenceAvailable).toBe(false);
    expect(res.refsByNode.fold.folds.summary.bestPlddt).toBeNull();

    // Verify TrajectoryFramesArtifact round-trips cleanly through checkpoint serialization
    const trajArt: TrajectoryFramesArtifact = {
      kind: 'TrajectoryFrames',
      id: 'traj-rt-1',
      parentStructureId: demoRef.id,
      frameCount: 2,
      ordering: 'ordered-frames',
      collectionKind: 'ordered-frames',
      pdbText: pdb,
      frames: [
        { key: 'f0', frameIndex: 0, caCoords: new Float32Array([1, 2, 3]), confidence: null },
        { key: 'f1', frameIndex: 1, caCoords: new Float32Array([4, 5, 6]), confidence: null },
      ],
    };
    const encoded = serializeCheckpoint({
      schema: 'denovo.checkpoint/v1',
      runId: 'r-traj',
      graphId: 'g-traj',
      createdAt: new Date().toISOString(),
      status: 'completed',
      completedNodeIds: ['t'],
      unfinishedNodeIds: [],
      nodes: {
        t: {
          nodeId: 't',
          adapterId: 'rfd4-proteina',
          cacheKey: 'k1',
          completedAt: new Date().toISOString(),
          outputs: { trajectory: trajArt },
          refs: {},
        },
      },
    });
    const decoded = deserializeCheckpoint(encoded);
    const restoredTraj = decoded.nodes.t.outputs.trajectory as TrajectoryFramesArtifact;
    expect(restoredTraj.kind).toBe('TrajectoryFrames');
    expect(restoredTraj.frameCount).toBe(2);
    expect(restoredTraj.frames?.[0].caCoords).toBeInstanceOf(Float32Array);
    expect(Array.from(restoredTraj.frames![1].caCoords!)).toEqual([4, 5, 6]);

    await scheduler.releaseActiveGpu();
  });

  it('Issue #6: unifies canvas & API run ownership, enforces active run lease on maintenance actions, and reports cancellation cleanly while preserving history', async () => {
    const scheduler = new ExecutionScheduler();
    const observedStates: ActiveRunState[] = [];
    const unsubscribe = scheduler.subscribeRunState((s) => observedStates.push(s));

    const simpleGraph: WorkflowGraph = {
      id: 'unified-ownership',
      name: 'Unified Ownership',
      nodes: [
        { id: 's', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFS' } },
      ],
      edges: [],
    };

    // 1. Canvas run updates canonical state and records history
    const canvasRes = await scheduler.run(simpleGraph, () => {}, { source: 'canvas' });
    expect(canvasRes.source).toBe('canvas');
    expect(canvasRes.runId).toBeTruthy();
    expect(scheduler.getActiveRunState().runId).toBe(canvasRes.runId);
    expect(scheduler.getActiveRunState().status).toBe('completed');
    expect(scheduler.getActiveRunState().leaseHeld).toBe(false);
    expect(scheduler.getRunHistory()).toHaveLength(1);

    // 2. API run via runWorkflowApi updates the EXACT same canonical state & holds lease while in-flight
    mockHangWorker = true;
    const apiGraph: WorkflowGraph = {
      id: 'api-lease-graph',
      name: 'API Lease Graph',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'ACDEFGHIKLMNPQRSTVWY' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1 } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };

    const abortCtrl = new AbortController();
    const apiPromise = scheduler.runWorkflowApi(apiGraph, () => {}, {
      signal: abortCtrl.signal,
      bypassCache: true,
    });
    await Promise.resolve();
    await Promise.resolve();

    const liveState = scheduler.getActiveRunState();
    expect(liveState.status).toBe('running');
    expect(liveState.source).toBe('api');
    expect(liveState.leaseHeld).toBe(true);
    expect(liveState.activeNodeId).toBe('fold');
    expect(liveState.completedNodeIds).toEqual(['seq']);

    // Maintenance actions MUST reject with RunLeaseError instead of killing the in-flight API run!
    await expect(scheduler.releaseGpu()).rejects.toThrow(RunLeaseError);
    await expect(scheduler.releaseActiveGpu()).rejects.toThrow(RunLeaseError);
    expect(() => scheduler.clearMemo()).toThrow(RunLeaseError);
    await expect(scheduler.clearCache()).rejects.toThrow(RunLeaseError);

    // Explicit cancellation via AbortSignal reports exact completed & unfinished nodes cleanly
    abortCtrl.abort();
    let caughtErr: unknown = null;
    try {
      await apiPromise;
    } catch (e) {
      caughtErr = e;
    }
    expect(caughtErr).toBeInstanceOf(RunCancellationError);
    const cancelErr = caughtErr as RunCancellationError;
    expect(cancelErr.completedNodeIds).toEqual(['seq']);
    expect(cancelErr.unfinishedNodeIds).toEqual(['fold']);
    expect(scheduler.getActiveRunState().status).toBe('cancelled');
    expect(scheduler.getActiveRunState().leaseHeld).toBe(false);

    // Prior completed canvas run is preserved in runHistory alongside the cancelled API run!
    const history = scheduler.getRunHistory();
    expect(history).toHaveLength(2);
    expect(history[0].runId).toBe(canvasRes.runId);
    expect(history[0].status).toBe('completed');
    expect(history[1].status).toBe('cancelled');
    expect(history[1].completedNodeIds).toEqual(['seq']);
    expect(history[1].unfinishedNodeIds).toEqual(['fold']);

    // attachProgrammaticApi routes through runWorkflowApi (source: 'api')
    const fakeWin = {} as Record<string, unknown>;
    vi.stubGlobal('window', fakeWin);
    vi.stubGlobal('document', {});
    attachProgrammaticApi(scheduler);
    const denovoApi = fakeWin.__denovo as { runWorkflow: (g: WorkflowGraph) => Promise<{ runId: string }> };
    mockHangWorker = false;
    const apiRun3 = await denovoApi.runWorkflow(simpleGraph);
    expect(apiRun3.runId).toBe(scheduler.getActiveRunState().runId);
    expect(scheduler.getActiveRunState().source).toBe('api');

    // Editing upstream input marks dependent nodes stale while preserving prior runs in history
    const editedApiGraph = structuredClone(apiGraph);
    editedApiGraph.nodes[0].params.sequence = 'GGGGGGGGGGAAAAAAAAAA';
    const inv = scheduler.invalidateEditedGraph(apiGraph, editedApiGraph);
    expect(inv.staleNodeIds).toEqual(['seq', 'fold']);
    expect(computeInvalidatedNodeIds(apiGraph, editedApiGraph).validNodeIds).toEqual([]);
    expect(scheduler.getRunHistory()).toHaveLength(3);

    unsubscribe();
    expect(observedStates.some((s) => s.source === 'api' && s.leaseHeld)).toBe(true);
    await scheduler.releaseActiveGpu();
  });

  it('Issue #6: checkpoints a branched run, reloads into a fresh scheduler, and resumes ONLY unfinished or invalidated nodes', async () => {
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const store = createMemoryCheckpointStore();
    const scheduler1 = new ExecutionScheduler({ checkpointStore: store });
    const demoRef = await materializePdbInputNode({
      id: 'src',
      adapterId: 'pdb-input',
      params: { mode: 'demo-bundle' },
    });

    const executedNodesRun1: string[] = [];
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: `folds-ckpt-${mockWorkerRunCalls}`,
      parentStructureId: demoRef.id,
      folds: [
        {
          sequenceId: 'cand-1',
          key: 'cand-1',
          sequence: parsed.sequence,
          foldModelId: 'localfold-af2',
          pdbText: pdb,
          caCoords: parsed.caCoords,
          plddt: new Float32Array(parsed.nResidues).fill(89),
          meanPlddt: 89,
          ptm: 0.87,
          elapsedMs: 10,
        },
      ],
    });

    // Two-branch workflow: Branch A (fold_a -> score_a) and Branch B (fold_b -> score_b)
    const branchedGraph: WorkflowGraph = {
      id: 'branched-resume',
      name: 'Branched Resume',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: parsed.sequence } },
        { id: 'fold_a', adapterId: 'localfold-af2', params: { model: 'monomer', recycles: 1, maxSequences: 1, seed: 1 } },
        { id: 'score_a', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
        { id: 'fold_b', adapterId: 'localfold-af2', params: { model: 'monomer', recycles: 1, maxSequences: 1, seed: 2 } },
        { id: 'score_b', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold_a', toPort: 'sequences' },
        { fromNode: 'src', fromPort: 'structure', toNode: 'score_a', toPort: 'structure' },
        { fromNode: 'fold_a', fromPort: 'folds', toNode: 'score_a', toPort: 'folds' },
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold_b', toPort: 'sequences' },
        { fromNode: 'src', fromPort: 'structure', toNode: 'score_b', toPort: 'structure' },
        { fromNode: 'fold_b', fromPort: 'folds', toNode: 'score_b', toPort: 'folds' },
      ],
    };

    // Interrupt after fold_a completes (before fold_b starts!)
    const abort1 = new AbortController();
    await expect(
      scheduler1.run(
        branchedGraph,
        (ev) => {
          if (ev.status === 'done') executedNodesRun1.push(ev.nodeId);
        },
        {
          signal: abort1.signal,
          checkpointKey: 'branched-job-1',
          onNodeOutput: (nodeId) => {
            if (nodeId === 'fold_a') abort1.abort();
          },
        },
      ),
    ).rejects.toThrow(RunCancellationError);

    expect(executedNodesRun1).toEqual(['src', 'seq', 'fold_a']);
    const savedJson = scheduler1.exportCheckpoint();
    expect(await store.load('branched-job-1')).toBe(savedJson);

    // Simulate reload: brand-new ExecutionScheduler instance loads checkpoint from store
    const scheduler2 = new ExecutionScheduler({ checkpointStore: store });
    const loadedCkpt = await scheduler2.loadCheckpointFromStore('branched-job-1');
    expect(loadedCkpt?.completedNodeIds).toEqual(['src', 'seq', 'fold_a']);

    const executedNodesRun2: string[] = [];
    const cachedNodesRun2: string[] = [];
    const workerCallsBeforeResume = mockWorkerRunCalls;
    const res2 = await scheduler2.run(
      branchedGraph,
      (ev) => {
        if (ev.status === 'done') executedNodesRun2.push(ev.nodeId);
        if (ev.status === 'cached') cachedNodesRun2.push(ev.nodeId);
      },
      { checkpointKey: 'branched-job-1' },
    );

    // Only unfinished nodes (score_a, fold_b, score_b) executed; src/seq/fold_a hit verified checkpoint!
    expect(cachedNodesRun2).toEqual(['src', 'seq', 'fold_a']);
    expect(executedNodesRun2.sort()).toEqual(['fold_b', 'score_a', 'score_b']);
    expect(mockWorkerRunCalls).toBe(workerCallsBeforeResume + 1); // ONLY fold_b invoked the worker!
    // Verify restored Float32Array coordinates produced accurate scRMSD in score_a!
    expect((res2.outputsByNode.score_a.metrics as DesignabilityMetrics).rows[0].scRmsd).toBeLessThan(1e-2);
    expect((res2.outputsByNode.score_b.metrics as DesignabilityMetrics).rows[0].scRmsd).toBeLessThan(1e-2);

    // Now invalidate ONLY Branch B (change fold_b.seed), reload into scheduler3, and resume:
    // ONLY fold_b and score_b should re-execute!
    const editedGraph = structuredClone(branchedGraph);
    editedGraph.nodes.find((n) => n.id === 'fold_b')!.params.seed = 99;
    const scheduler3 = new ExecutionScheduler({ checkpointStore: store });
    await scheduler3.loadCheckpointFromStore('branched-job-1');
    const executedRun3: string[] = [];
    const cachedRun3: string[] = [];
    const workerCallsBeforeEditResume = mockWorkerRunCalls;
    await scheduler3.run(editedGraph, (ev) => {
      if (ev.status === 'done') executedRun3.push(ev.nodeId);
      if (ev.status === 'cached') cachedRun3.push(ev.nodeId);
    });
    expect(cachedRun3.sort()).toEqual(['fold_a', 'score_a', 'seq', 'src']);
    expect(executedRun3.sort()).toEqual(['fold_b', 'score_b']);
    expect(mockWorkerRunCalls).toBe(workerCallsBeforeEditResume + 1);

    await scheduler1.releaseActiveGpu();
    await scheduler2.releaseActiveGpu();
    await scheduler3.releaseActiveGpu();
  });

  it('Issue #6: enforces configurable result-cache byte budget (maxMemoBytes) with deterministic LRU eviction over 64 distinct runs independent of weight cache', async () => {
    // Stub a weight cache to confirm result-cache eviction never touches model weights
    const weightCacheKeys = new Set(['denovo-run-weights-v1', 'localfold-v1']);
    const cacheDeleteSpy = vi.fn(async (k: string) => weightCacheKeys.delete(k));
    vi.stubGlobal('caches', {
      keys: vi.fn(async () => [...weightCacheKeys]),
      delete: cacheDeleteSpy,
    });

    const budgetBytes = 12 * 1024; // 12 KiB budget
    const scheduler = new ExecutionScheduler({ maxMemoBytes: budgetBytes });

    // Execute 64 distinct CPU sequence-input runs (each ~500-900 bytes in memo)
    const AA20 = 'ACDEFGHIKLMNPQRSTVWY';
    const graphs: WorkflowGraph[] = [];
    for (let i = 0; i < 64; i++) {
      const seq = `MKTAYIAKQR${'ACDEFGHIKL'.repeat(12)}${AA20[i % 20].repeat(4)}${AA20[Math.floor(i / 20) % 20].repeat(4)}`;
      const g: WorkflowGraph = {
        id: `run-64-${i}`,
        name: `Run ${i}`,
        nodes: [
          {
            id: 's',
            adapterId: 'sequence-input',
            params: { label: `cand-${i}`, sequence: seq },
          },
        ],
        edges: [],
      };
      graphs.push(g);
      await scheduler.run(g, () => {});
      expect(scheduler.getMemoStats().usedBytes).toBeLessThanOrEqual(budgetBytes);
    }

    const stats = scheduler.getMemoStats();
    expect(stats.maxBytes).toBe(budgetBytes);
    expect(stats.usedBytes).toBeLessThanOrEqual(budgetBytes);
    expect(stats.entryCount).toBeLessThan(64);
    expect(stats.entryCount).toBeGreaterThan(0);
    expect(stats.evictionCount).toBeGreaterThan(0);

    // Most recent run (#63) MUST be a cache hit (MRU), while oldest run (#0) MUST be evicted (cache miss)!
    const evNewest: NodeProgressEvent[] = [];
    await scheduler.run(graphs[63], (ev) => evNewest.push(ev));
    expect(evNewest.some((e) => e.status === 'cached')).toBe(true);

    const evOldest: NodeProgressEvent[] = [];
    await scheduler.run(graphs[0], (ev) => evOldest.push(ev));
    expect(evOldest.some((e) => e.status === 'cached')).toBe(false);
    expect(evOldest.some((e) => e.status === 'done')).toBe(true);

    // Weight cache was completely untouched by result-cache LRU eviction
    expect(cacheDeleteSpy).not.toHaveBeenCalled();
    expect(weightCacheKeys.has('denovo-run-weights-v1')).toBe(true);
    expect(weightCacheKeys.has('localfold-v1')).toBe(true);
  });

  it('Issue #3 (mandatory identical-coords test): distinguishes self-consistency against A vs external-mismatch against identical-coordinate B vs external-unlinked standalone sequence', async () => {
    const scheduler = new ExecutionScheduler();
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const L = parsed.nResidues;

    // Two distinct StructureArtifacts A and B with IDENTICAL coordinates but different IDs!
    const structA: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'struct-A',
      label: 'Design A',
      sourceModelId: 'genie3',
      length: L,
      sequence: parsed.sequence,
      caCoords: new Float32Array(parsed.caCoords),
      backbone4Coords: new Float32Array(parsed.backbone4Coords),
      pdbText: pdb,
      geometry: { length: L, rg: 11, meanCaDist: 3.8, chainBreaks: 0, clashes: 0, helixPct: 80, sheetPct: 0, coilPct: 20 },
    };
    const structB: StructureArtifact = {
      ...structA,
      id: 'struct-B',
      label: 'Design B',
      caCoords: new Float32Array(parsed.caCoords),
      backbone4Coords: new Float32Array(parsed.backbone4Coords),
    };

    // Route mock generator by node seed so nodeA outputs structA and nodeB outputs structB
    mockRawOutputsFactory = (msg) => {
      if (msg.inputs && 'sequences' in msg.inputs) {
        const folds: FoldedStructure = {
          kind: 'FoldedStructure',
          id: `folds-${String(msg.params?.seed ?? 0)}`,
          parentStructureId: String(msg.params?.seed) === '99' ? 'none' : 'struct-A',
          sourceStructureId: String(msg.params?.seed) === '99' ? 'none' : 'struct-A',
          folds: [
            {
              sequenceId: 'cand-1',
              key: 'cand-1',
              sequence: parsed.sequence,
              foldModelId: 'localfold-af2',
              pdbText: pdb,
              caCoords: new Float32Array(parsed.caCoords),
              plddt: new Float32Array(L).fill(93),
              meanPlddt: 93,
              confidence: 93,
              confidenceAvailable: true,
              ptm: 0.9,
              elapsedMs: 10,
            },
          ],
        };
        return { folds, structure: makePlaceholderStructure(`rep-${folds.id}`, parsed.sequence) };
      }
      const st = Number(msg.params?.seed) === 1 ? structA : structB;
      return {
        structure: st,
        sequences: {
          kind: 'SequenceSet',
          id: `seq-${st.id}`,
          parentStructureId: st.id,
          sourceStructureId: st.id,
          sourceModelId: 'genie3',
          candidates: [
            { id: 'cand-1', key: 'cand-1', sequence: parsed.sequence, score: null, globalScore: null, sourceModelId: 'genie3' },
          ],
        },
      };
    };

    const makeGraph = (wireRefNode: 'genA' | 'genB', foldSeed: number): WorkflowGraph => ({
      id: `ident-coords-${wireRefNode}-${foldSeed}`,
      name: 'Identical Coords Lineage',
      nodes: [
        { id: 'genA', adapterId: 'genie3', params: { length: L, seed: 1 } },
        { id: 'genB', adapterId: 'genie3', params: { length: L, seed: 2 } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: parsed.sequence } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1, seed: foldSeed } },
        { id: 'score', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: wireRefNode, fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    });

    // 1. Fold from A scored against A -> self-consistency PASS!
    const resA = await scheduler.run(makeGraph('genA', 1), () => {}, { bypassCache: true });
    const mA = resA.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(mA.referenceRelation).toBe('self-consistency');
    expect(mA.referenceLabel).toBe('Self-consistency against Design A');
    expect(mA.selfConsistencyValid).toBe(true);
    expect(mA.lineageMismatch).toBe(false);
    expect(mA.rows[0].designable).toBe(true);
    expect(mA.passRate).toBe(1);

    // 2. Change ONLY the wire from A to identical-coordinate B -> identical RMSD, external-mismatch, NOT a self-consistency PASS!
    const resB = await scheduler.run(makeGraph('genB', 1), () => {}, { bypassCache: true });
    const mB = resB.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(mB.rows[0].scRmsd).toBeCloseTo(mA.rows[0].scRmsd, 6);
    expect(mB.referenceRelation).toBe('external-mismatch');
    expect(mB.referenceLabel).toBe('Comparison against Design B · Fold originated from struct-A');
    expect(mB.selfConsistencyValid).toBe(false);
    expect(mB.lineageMismatch).toBe(true);
    expect(mB.rows[0].designable).toBe(false);
    expect(mB.passRate).toBe(0);

    // 3. Standalone sequence (no parentStructureId / 'none') scored against A -> external-unlinked, NOT a self-consistency PASS!
    const resUnlinked = await scheduler.run(makeGraph('genA', 99), () => {}, { bypassCache: true });
    const mU = resUnlinked.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(mU.rows[0].scRmsd).toBeCloseTo(mA.rows[0].scRmsd, 6);
    expect(mU.referenceRelation).toBe('external-unlinked');
    expect(mU.referenceLabel).toBe('Comparison against Design A · No parent structure recorded');
    expect(mU.selfConsistencyValid).toBe(false);
    expect(mU.rows[0].designable).toBe(false);
    expect(mU.passRate).toBe(0);

    await scheduler.releaseActiveGpu();
  });

  it('Issue #4/#6 (Astra regressions): preserves resume snapshot with maxMemoBytes=0, rejects tampered checkpoints, and parses indented FASTA headers', async () => {
    // 1. Indented second FASTA header produces two separate sequences instead of fusing header text into sequence
    const sFasta = new ExecutionScheduler();
    const fastaRes = await sFasta.run(
      {
        id: 'indented-fasta',
        name: 'Indented FASTA',
        nodes: [
          {
            id: 'seq',
            adapterId: 'sequence-input',
            params: { sequence: '>alpha\nAAAAA\n  >second\nCCCCC' },
          },
        ],
        edges: [],
      },
      () => {},
    );
    const fastaSet = fastaRes.outputsByNode.seq.sequences as SequenceSet;
    expect(fastaSet.candidates.map((c) => [c.id, c.sequence])).toEqual([
      ['alpha', 'AAAAA'],
      ['second', 'CCCCC'],
    ]);

    // 2. Astra Finding #1: With maxMemoBytes=0 (memo disabled), resuming a 2-node checkpoint reuses BOTH nodes without re-execution!
    const twoNodeGraph: WorkflowGraph = {
      id: 'zero-memo-resume',
      name: 'Zero Memo Resume',
      nodes: [
        { id: 'n1', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFS' } },
        { id: 'n2', adapterId: 'localfold-af2', params: { model: 'monomer' } },
      ],
      edges: [{ fromNode: 'n1', fromPort: 'sequences', toNode: 'n2', toPort: 'sequences' }],
    };
    const producer = new ExecutionScheduler();
    await producer.run(twoNodeGraph, () => {});
    const exported = producer.exportCheckpoint();
    await producer.releaseActiveGpu();

    const zeroMemoResumer = new ExecutionScheduler({ maxMemoBytes: 0 });
    zeroMemoResumer.importCheckpoint(exported);
    const cachedIds: string[] = [];
    const executedIds: string[] = [];
    await zeroMemoResumer.run(twoNodeGraph, (ev) => {
      if (ev.status === 'cached') cachedIds.push(ev.nodeId);
      if (ev.status === 'done') executedIds.push(ev.nodeId);
    });
    expect(cachedIds).toEqual(['n1', 'n2']);
    expect(executedIds).toEqual([]);

    // 3. Astra Finding #7: Importing a checkpoint with missing required output ports throws Output contract violation!
    const tampered = JSON.stringify({
      schema: 'denovo.checkpoint/v1',
      runId: 'bad-run',
      graphId: 'bad-g',
      createdAt: new Date().toISOString(),
      status: 'completed',
      completedNodeIds: ['seq'],
      unfinishedNodeIds: [],
      nodes: {
        seq: {
          nodeId: 'seq',
          adapterId: 'sequence-input',
          cacheKey: 'fake-key',
          completedAt: new Date().toISOString(),
          outputs: {}, // missing required 'sequences' port!
          refs: {},
        },
      },
    });
    const validatorScheduler = new ExecutionScheduler();
    expect(() => validatorScheduler.importCheckpoint(tampered)).toThrow(
      /Output contract violation.*missing required output port "sequences"/i,
    );
  });

  it('Issue #5 (end-to-end workflow): executes 3 named candidates across two branches in reversed order with 1 omitted member and exports paired join CSV', async () => {
    const scheduler = new ExecutionScheduler();
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const L = parsed.nResidues;

    // Two predictor branches: branch A returns [cand-3, cand-1, cand-2]; branch B returns [cand-2, cand-1] (omits cand-3!)
    mockRawOutputsFactory = (msg) => {
      const model = String(msg.params?.model ?? 'monomer');
      const isBranchA = model === 'monomer';
      const makeCand = (id: string, plddt: number, confAvail = true) => ({
        sequenceId: id,
        key: id,
        sequence: parsed.sequence,
        foldModelId: `localfold-${model}`,
        pdbText: pdb,
        caCoords: new Float32Array(parsed.caCoords),
        plddt: confAvail ? new Float32Array(L).fill(plddt) : new Float32Array(0),
        meanPlddt: confAvail ? plddt : Number.NaN,
        confidence: confAvail ? plddt : null,
        confidenceAvailable: confAvail,
        ptm: confAvail ? 0.85 : null,
        elapsedMs: 10,
      });
      const folds: FoldedStructure = {
        kind: 'FoldedStructure',
        id: `folds-${model}`,
        parentStructureId: 'none',
        collectionKind: 'independent-candidates',
        folds: isBranchA
          ? [makeCand('cand-3', 79.0), makeCand('cand-1', 91.5), makeCand('cand-2', 0, false)]
          : [makeCand('cand-2', 0, false), makeCand('cand-1', 88.0)],
      };
      return { folds, structure: makePlaceholderStructure(`rep-${model}`, parsed.sequence) };
    };

    const graph: WorkflowGraph = {
      id: 'two-branch-join',
      name: 'Two Branch Join',
      nodes: [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          params: {
            sequence: `>cand-1\n${parsed.sequence}\n>cand-2\n${parsed.sequence}\n>cand-3\n${parsed.sequence}`,
          },
        },
        { id: 'branch_a', adapterId: 'localfold-af2', label: 'Predictor A', params: { model: 'monomer', maxSequences: 3 } },
        { id: 'branch_b', adapterId: 'localfold-af2', label: 'Predictor B', params: { model: 'rosettafold3', maxSequences: 3 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'branch_a', toPort: 'sequences' },
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'branch_b', toPort: 'sequences' },
      ],
    };

    const res = await scheduler.run(graph, () => {});
    const joined = joinBranchArtifacts(graph, ['branch_a', 'branch_b'], res.outputsByNode);
    expect(joined.rows).toHaveLength(3);
    expect(joined.completeRowCount).toBe(2);
    expect(joined.partialRowCount).toBe(1);

    const row1 = joined.rows.find((r) => r.key === 'cand-1')!;
    expect(row1.byBranch.branch_a.present).toBe(true);
    expect(row1.byBranch.branch_b.present).toBe(true);
    expect(row1.byBranch.branch_a.confidence).toBeCloseTo(91.5, 3);
    expect(row1.byBranch.branch_b.confidence).toBeCloseTo(88.0, 3);

    const row2 = joined.rows.find((r) => r.key === 'cand-2')!;
    expect(row2.byBranch.branch_a.confidenceAvailable).toBe(false);
    expect(row2.byBranch.branch_b.confidenceAvailable).toBe(false);

    const row3 = joined.rows.find((r) => r.key === 'cand-3')!;
    expect(row3.missingBranchIds).toEqual(['branch_b']);
    expect(row3.byBranch.branch_b.present).toBe(false);

    const csv = exportJoinedCollectionCsv(joined);
    expect(csv).toContain('cand-3,1,branch_b,PRESENT,1,79,79,MISSING,,,');
    expect(csv).toContain('cand-2,2,,PRESENT,3,,N/A,PRESENT,1,,N/A');

    await scheduler.releaseActiveGpu();
  });

  it('Verifier regression suite: conflicting lineage, partial-mapping no-PASS, invalidateEditedGraph memo eviction, clearMemo checkpoint reset, entry integrity, single cancel history, GPU disposal await, and storage-save failure resilience', async () => {
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const L = parsed.nResidues;

    // 1. Conflicting parentStructureId (A) vs sourceStructureId (B) when ref is B -> external-mismatch, NOT self-consistency PASS!
    const sLineage = new ExecutionScheduler();
    const demoRef = await materializePdbInputNode({
      id: 'src',
      adapterId: 'pdb-input',
      params: { mode: 'demo-bundle' },
    });
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-conflict',
      parentStructureId: 'different-parent-A',
      sourceStructureId: demoRef.id, // disagrees with parentStructureId!
      folds: [
        {
          sequenceId: 'c1',
          key: 'c1',
          sequence: parsed.sequence,
          foldModelId: 'localfold-af2',
          pdbText: pdb,
          caCoords: new Float32Array(parsed.caCoords),
          plddt: new Float32Array(L).fill(94),
          meanPlddt: 94,
          ptm: 0.9,
          elapsedMs: 5,
        },
      ],
    });
    const conflictRes = await sLineage.run(
      {
        id: 'conflict-lineage',
        name: 'Conflict Lineage',
        nodes: [
          { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
          { id: 'seq', adapterId: 'sequence-input', params: { sequence: parsed.sequence } },
          { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1 } },
          { id: 'score', adapterId: 'designability-scorer', params: {} },
        ],
        edges: [
          { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
          { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
          { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
        ],
      },
      () => {},
    );
    const conflictMetrics = conflictRes.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(conflictMetrics.referenceRelation).toBe('external-mismatch');
    expect(conflictMetrics.selfConsistencyValid).toBe(false);
    expect(conflictMetrics.lineageMismatch).toBe(true);
    expect(conflictMetrics.rows[0].designable).toBe(false);

    // 2. Partial residueMapping on a DISTORTED same-parent fold must NOT turn it into a self-consistency PASS!
    const distortedCa = new Float32Array(parsed.caCoords);
    for (let i = 6; i < L; i++) {
      distortedCa[i * 3] += 25.0; // severely distorted outside first 6 residues!
    }
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-distorted-same',
      parentStructureId: demoRef.id,
      sourceStructureId: demoRef.id,
      folds: [
        {
          sequenceId: 'distorted',
          key: 'distorted',
          sequence: parsed.sequence,
          foldModelId: 'localfold-af2',
          pdbText: pdb,
          caCoords: distortedCa,
          plddt: new Float32Array(L).fill(95),
          meanPlddt: 95,
          ptm: 0.9,
          elapsedMs: 5,
        },
      ],
    });
    const partialRes = await sLineage.run(
      {
        id: 'partial-map-same-parent',
        name: 'Partial Map Same Parent',
        nodes: [
          { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
          { id: 'seq', adapterId: 'sequence-input', params: { sequence: parsed.sequence } },
          { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1 } },
          {
            id: 'score',
            adapterId: 'designability-scorer',
            params: { residueMapping: [[0, 0], [1, 1], [2, 2], [3, 3]] },
          },
        ],
        edges: [
          { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
          { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
          { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
        ],
      },
      () => {},
      { bypassCache: true },
    );
    const partialMetrics = partialRes.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(partialMetrics.referenceRelation).toBe('external-mapped');
    expect(partialMetrics.selfConsistencyValid).toBe(false);
    expect(partialMetrics.rows[0].scRmsd).toBeLessThan(1e-2);
    expect(partialMetrics.rows[0].designable).toBe(false);

    // 2b. Explicitly forcing allowExternalReference: true even on a same-parent fold sets selfConsistencyValid: false & designable: false!
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-same-parent-ext',
      parentStructureId: demoRef.id,
      sourceStructureId: demoRef.id,
      folds: [
        {
          sequenceId: 'same-ext',
          key: 'same-ext',
          sequence: parsed.sequence,
          foldModelId: 'localfold-af2',
          pdbText: pdb,
          caCoords: new Float32Array(parsed.caCoords),
          plddt: new Float32Array(L).fill(95),
          meanPlddt: 95,
          ptm: 0.92,
          elapsedMs: 5,
        },
      ],
    });
    const forcedExtRes = await sLineage.run(
      {
        id: 'forced-ext-same-parent',
        name: 'Forced Ext Same Parent',
        nodes: [
          { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
          { id: 'seq', adapterId: 'sequence-input', params: { sequence: parsed.sequence } },
          { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1 } },
          {
            id: 'score',
            adapterId: 'designability-scorer',
            params: { allowExternalReference: true },
          },
        ],
        edges: [
          { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
          { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
          { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
        ],
      },
      () => {},
      { bypassCache: true },
    );
    const forcedExtMetrics = forcedExtRes.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(forcedExtMetrics.referenceMode).toBe('external-reference');
    expect(forcedExtMetrics.referenceRelation).toBe('external-unlinked');
    expect(forcedExtMetrics.selfConsistencyValid).toBe(false);
    expect(forcedExtMetrics.rows[0].selfConsistencyValid).toBe(false);
    expect(forcedExtMetrics.rows[0].designable).toBe(false);
    await sLineage.releaseActiveGpu();

    // 3. invalidateEditedGraph evicts stale memo entries AND benchmarkId changes bust StructureArtifact.id (forcing preflight re-check!)
    const sInv = new ExecutionScheduler();
    mockGenStructureFactory = () => makePlaceholderStructure('gen-ok', parsed.sequence);
    const gBefore: WorkflowGraph = {
      id: 'inv-memo',
      name: 'Inv Memo',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'g3', adapterId: 'genie3', params: { length: 53, taskMode: 'motif', motifSpan: 'A16-35' } },
      ],
      edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    };
    const rBefore = await sInv.run(gBefore, () => {});
    const gAfter = structuredClone(gBefore);
    // Switching benchmarkId to a ligand benchmark unsupported by Genie 3 MUST invalidate & fail preflight (not hit stale memo!)
    gAfter.nodes[0].params.benchmarkId = 'rfd4-sm-sam';
    const invRes = sInv.invalidateEditedGraph(gBefore, gAfter);
    expect(invRes.staleNodeIds).toEqual(['src', 'g3']);
    expect(sInv.getMemoStats().entryCount).toBe(0);
    // Verify artifact id changes when benchmarkId changes
    const srcAfter = await materializePdbInputNode(gAfter.nodes[0]);
    expect(srcAfter.id).not.toBe(rBefore.outputsByNode.src.structure.id);
    await sInv.releaseActiveGpu();

    // 4. clearMemo() resets latestCheckpoint so subsequent identical run does NOT hit checkpoint!
    const sClear = new ExecutionScheduler();
    const cpuGraph: WorkflowGraph = {
      id: 'clear-ckpt',
      name: 'Clear Ckpt',
      nodes: [{ id: 's', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFS' } }],
      edges: [],
    };
    await sClear.run(cpuGraph, () => {});
    sClear.clearMemo();
    const clearEvs: NodeProgressEvent[] = [];
    await sClear.run(cpuGraph, (ev) => clearEvs.push(ev));
    expect(clearEvs.some((e) => e.status === 'cached')).toBe(false);
    expect(clearEvs.some((e) => e.status === 'done')).toBe(true);

    // 5. Checkpoint entry integrity rejects swapped valid outputs between two cache keys!
    const sSwapA = new ExecutionScheduler();
    const sSwapB = new ExecutionScheduler();
    await sSwapA.run(
      { id: 'gA', name: 'A', nodes: [{ id: 's', adapterId: 'sequence-input', params: { sequence: 'AAAAA' } }], edges: [] },
      () => {},
    );
    await sSwapB.run(
      { id: 'gB', name: 'B', nodes: [{ id: 's', adapterId: 'sequence-input', params: { sequence: 'CCCCC' } }], edges: [] },
      () => {},
    );
    const ckptA = sSwapA.exportCheckpointObject()!;
    const ckptB = sSwapB.exportCheckpointObject()!;
    // Swap outputs from B into A's cacheKey & integrity tag
    ckptA.nodes.s.outputs = ckptB.nodes.s.outputs;
    expect(() => new ExecutionScheduler().importCheckpoint(ckptA)).toThrow(
      /Checkpoint integrity violation/i,
    );

    // 6. Cancellation finalizes EXACTLY ONCE (no duplicate history entries) AND awaits GPU disposal before clearing lease / spawning next worker!
    mockHangWorker = true;
    mockDisposeDelayMs = 25;
    mockLiveWorkers = 0;
    mockMaxLiveWorkers = 0;
    const sCancel = new ExecutionScheduler();
    const gpuGraph: WorkflowGraph = {
      id: 'cancel-gpu-dispose',
      name: 'Cancel GPU Dispose',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'ACDEFGHIKLMNPQRSTVWY' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1 } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };

    const ac1 = new AbortController();
    const p1 = sCancel.run(gpuGraph, () => {}, { signal: ac1.signal, bypassCache: true });
    await Promise.resolve();
    await Promise.resolve();
    ac1.abort();
    await expect(p1).rejects.toThrow(RunCancellationError);
    // Disposal MUST have finished before run settled (zero live workers left, exactly 1 history entry!)
    expect(mockLiveWorkers).toBe(0);
    expect(sCancel.getRunHistory()).toHaveLength(1);
    expect(sCancel.getRunHistory()[0].status).toBe('cancelled');

    // Starting a second run immediately after cancellation never overlaps two live GPU workers!
    mockHangWorker = false;
    mockDisposeDelayMs = 0;
    await sCancel.run(gpuGraph, () => {}, { bypassCache: true });
    expect(mockMaxLiveWorkers).toBe(1);
    await sCancel.releaseActiveGpu();

    // 7. Checkpoint-store save failure during run failure still transitions activeRun to 'failed', leaseHeld: false, and records history!
    const failingStore = {
      save: vi.fn(async () => {
        throw new Error('IndexedDB quota exceeded');
      }),
      load: vi.fn(async () => null),
    };
    const sFailStore = new ExecutionScheduler({ checkpointStore: failingStore });
    await expect(
      sFailStore.run(cpuGraph, () => {}, { checkpointKey: 'fail-key' }),
    ).rejects.toThrow(/IndexedDB quota exceeded/);
    const postFailState = sFailStore.getActiveRunState();
    expect(postFailState.status).toBe('failed');
    expect(postFailState.leaseHeld).toBe(false);
    expect(sFailStore.getRunHistory()).toHaveLength(1);
    expect(sFailStore.getRunHistory()[0].status).toBe('failed');
  });

  it('Wave-2 P1/P2 & Safari VRAM regressions: demo-bundle ignores cropSegments, metadata propagation, stoichiometry mismatch, homodimer role gate, AF3 holo PAE slice, thresholds in metrics.id, canonical cacheKey, and idle GPU auto-release', async () => {
    // 1. demo-bundle ignores stale cropSegments & propagates sourceResidues/insCodes/chainIdMap/observedBackboneMask
    const demoWithStaleCrop = await materializePdbInputNode({
      id: 'src',
      adapterId: 'pdb-input',
      params: { mode: 'demo-bundle', cropSegments: 'A1-5' },
    });
    expect(demoWithStaleCrop.length).toBe(53);
    expect(demoWithStaleCrop.sourceResidues).toHaveLength(53);
    expect(demoWithStaleCrop.insCodes).toHaveLength(53);
    expect(demoWithStaleCrop.chainIdMap).toEqual({ A: 'A' });
    expect(demoWithStaleCrop.observedBackboneMask).toBeInstanceOf(Uint8Array);
    expect(demoWithStaleCrop.observedBackboneMask!.length).toBe(53);

    // 2. Canonical cacheKey: reordered param keys hit the exact same memoized entry (no redundant worker call)
    const sKey = new ExecutionScheduler();
    const gA: WorkflowGraph = {
      id: 'canon-a',
      name: 'Canon A',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFS' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer', recycles: 1, seed: 7 } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const gB: WorkflowGraph = {
      id: 'canon-b',
      name: 'Canon B',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFS' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { seed: 7, recycles: 1, model: 'monomer' } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    await sKey.run(gA, () => {});
    expect(mockWorkerRunCalls).toBe(1);
    await sKey.run(gB, () => {});
    expect(mockWorkerRunCalls).toBe(1); // memo hit despite different key order
    await sKey.releaseActiveGpu();

    // 3. Scorer: symmetric homodimer WITHOUT chainRoles/lockedChains omits binderPlddt/targetPlddt;
    //    AF3 holo/DNA full-complex PAE matrix (paeDim > L) still computes finite iPAE;
    //    thresholds change metrics.id; chain-stoichiometry mismatch is rejected cleanly.
    const dimerPdb = [
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C  ',
      'ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 50.00           C  ',
      'ATOM      3  CA  VAL A   3       7.600   0.000   0.000  1.00 50.00           C  ',
      'TER',
      'ATOM      4  CA  ALA B   1       0.000  10.000   0.000  1.00 50.00           C  ',
      'ATOM      5  CA  GLY B   2       3.800  10.000   0.000  1.00 50.00           C  ',
      'ATOM      6  CA  VAL B   3       7.600  10.000   0.000  1.00 50.00           C  ',
      'END',
    ].join('\n');
    const dimerParsed = parsePdbBackbone(dimerPdb);
    const dimerRef = await materializePdbInputNode({
      id: 'src',
      adapterId: 'pdb-input',
      params: { mode: 'custom', pdbText: dimerPdb },
    });

    // Full-complex 8x8 PAE (6 protein + 2 ligand/DNA tokens) with low inter-chain PAE (2.5 Å)
    const paeDim = 8;
    const pae = new Float32Array(paeDim * paeDim).fill(2.5);
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-dimer',
      parentStructureId: dimerRef.id,
      folds: [
        {
          sequenceId: 'homodimer-1',
          sequence: dimerParsed.sequence,
          foldModelId: 'localfold-af3-int5',
          pdbText: dimerPdb,
          caCoords: dimerParsed.caCoords,
          chainIds: ['A', 'A', 'A', 'B', 'B', 'B'],
          resNums: [1, 2, 3, 1, 2, 3],
          plddt: new Float32Array(6).fill(90),
          meanPlddt: 90,
          ptm: 0.9,
          pae,
          paeDim,
          elapsedMs: 10,
        },
      ],
    });
    const scoreGraph = (scRmsdThreshold: number): WorkflowGraph => ({
      id: `score-dimer-${scRmsdThreshold}`,
      name: 'Score Dimer',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: dimerPdb } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'AGV:AGV' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'multimer' } },
        { id: 'score', adapterId: 'designability-scorer', params: { scRmsdThreshold, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    });
    const sScore = new ExecutionScheduler();
    const res1 = await sScore.run(scoreGraph(2.0), () => {}, { bypassCache: true });
    const m1 = res1.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(m1.rows[0].binderPlddt).toBeUndefined();
    expect(m1.rows[0].targetPlddt).toBeUndefined();
    expect(m1.rows[0].ipae).toBeCloseTo(2.5, 3);

    const res2 = await sScore.run(scoreGraph(1.5), () => {}, { bypassCache: true });
    const m2 = res2.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(m2.id).not.toBe(m1.id);

    // Chain-stoichiometry mismatch (6-aa single-chain fold vs 3+3 dimer reference) rejects cleanly
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'folds-stoich-mismatch',
      parentStructureId: dimerRef.id,
      folds: [
        {
          sequenceId: 'mono-6',
          sequence: dimerParsed.sequence,
          foldModelId: 'localfold-af2-monomer-int5',
          pdbText: dimerPdb.replace(/ B /g, ' A '),
          caCoords: dimerParsed.caCoords,
          chainIds: ['A', 'A', 'A', 'A', 'A', 'A'],
          resNums: [1, 2, 3, 4, 5, 6],
          plddt: new Float32Array(6).fill(90),
          meanPlddt: 90,
          ptm: 0.9,
          elapsedMs: 10,
        },
      ],
    });
    await expect(sScore.run(scoreGraph(2.0), () => {}, { bypassCache: true })).rejects.toThrow(
      /chain stoichiometry mismatch/i,
    );
    await sScore.releaseActiveGpu();

    // 4. Idle GPU auto-release timer releases active GPU worker automatically after idle timeout
    const sIdle = new ExecutionScheduler({ gpuIdleReleaseMs: 30 });
    await sIdle.run(gA, () => {}, { bypassCache: true });
    expect(mockLiveWorkers).toBe(1);
    await new Promise((r) => setTimeout(r, 90));
    expect(mockLiveWorkers).toBe(0);
    sIdle.destroy();
  });

  it('Wave-3 designability-scorer: computes Dunbrack ipSAE (including AF3 protein-sliced PAE), propagates msaMode/msaDepth, and populates explicit failedGates', async () => {
    const scheduler = new ExecutionScheduler();
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
    const nTarget = 12;
    const nBinder = 8;
    const L = nTarget + nBinder;
    const complexCa = new Float32Array(L * 3);
    complexCa.set(makeHelix(nTarget, 0, 0, 0), 0);
    complexCa.set(makeHelix(nBinder, 9, 0, 3), nTarget * 3);
    const chainIds = [...Array(nTarget).fill('A'), ...Array(nBinder).fill('B')];
    const lines: string[] = [];
    for (let i = 0; i < L; i++) {
      const ch = chainIds[i];
      const res = i < nTarget ? i + 1 : i - nTarget + 1;
      lines.push(
        `ATOM  ${String(i + 1).padStart(5)}  CA  ALA ${ch}${String(res).padStart(4)}    ${complexCa[i * 3].toFixed(3).padStart(8)}${complexCa[i * 3 + 1].toFixed(3).padStart(8)}${complexCa[i * 3 + 2].toFixed(3).padStart(8)}  1.00 90.00           C  `,
      );
    }
    lines.push('END');
    const complexPdb = lines.join('\n');

    // AF3-style 24x24 PAE (20 protein aa + 4 ligand tokens)
    const makeAf3Pae = (interPae: number): Float32Array => {
      const dim = L + 4;
      const m = new Float32Array(dim * dim).fill(28.0);
      for (let i = 0; i < L; i++) {
        for (let j = 0; j < L; j++) {
          const same = (i < nTarget) === (j < nTarget);
          m[i * dim + j] = same ? 0.8 : interPae;
        }
      }
      return m;
    };

    mockGenStructureFactory = () => ({
      kind: 'StructureArtifact',
      id: 'ipsae-gen-1',
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
      geometry: { length: L, rg: 11, meanCaDist: 3.8, chainBreaks: 0, clashes: 0, helixPct: 80, sheetPct: 0, coilPct: 20 },
    });

    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'ipsae-folds-1',
      parentStructureId: 'ipsae-gen-1',
      folds: [
        {
          sequenceId: 'tight-interface',
          sequence: 'A'.repeat(L),
          foldModelId: 'localfold-boltz2',
          pdbText: complexPdb,
          caCoords: complexCa,
          plddt: new Float32Array(L).fill(91),
          meanPlddt: 91,
          ptm: 0.89,
          iptm: 0.85,
          pae: makeAf3Pae(0.6), // high ipSAE (> 0.7)
          chainLengths: [nTarget, nBinder],
          chainIds,
          msaMode: 'target-only',
          msaDepth: 184,
          elapsedMs: 42,
        },
        {
          sequenceId: 'weak-interface',
          sequence: 'G'.repeat(L),
          foldModelId: 'localfold-boltz2',
          pdbText: complexPdb,
          caCoords: complexCa,
          plddt: new Float32Array(L).fill(64.2), // fails pLDDT & binder pLDDT
          meanPlddt: 64.2,
          ptm: 0.65,
          iptm: 0.45,
          pae: makeAf3Pae(3.0), // low iPAE (3.0 < 10) but low ipSAE (~0.10 < 0.60)
          chainLengths: [nTarget, nBinder],
          chainIds,
          msaMode: 'single',
          msaDepth: 1,
          elapsedMs: 48,
        },
      ],
    });

    const graph: WorkflowGraph = {
      id: 'ipsae-scorer-test',
      name: 'ipSAE Scorer',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 70 } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `${'A'.repeat(nTarget)}:${'A'.repeat(nBinder)}` } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'boltz2', recycles: 1, maxSequences: 2, seed: 0 } },
        {
          id: 'score',
          adapterId: 'designability-scorer',
          params: { scRmsdThreshold: 2.0, plddtThreshold: 70, ipsaeThreshold: 0.6 },
        },
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

    const tight = byId.get('tight-interface')!;
    expect(tight.designable).toBe(true);
    expect(tight.failedGates).toEqual([]);
    expect(tight.chainLengths).toEqual([nTarget, nBinder]);
    expect(tight.ipsae!).toBeGreaterThan(0.7);
    expect(tight.ipsaeMin!).toBeCloseTo(tight.ipsae!, 5);
    expect(tight.msaMode).toBe('target-only');
    expect(tight.msaDepth).toBe(184);

    const weak = byId.get('weak-interface')!;
    expect(weak.designable).toBe(false);
    expect(weak.chainLengths).toEqual([nTarget, nBinder]);
    expect(weak.ipsae!).toBeLessThan(0.6);
    expect(weak.msaMode).toBe('single');
    expect(weak.msaDepth).toBe(1);
    expect(weak.failedGates!.some((g) => g.includes('pLDDT 64.2 < 70'))).toBe(true);
    expect(weak.failedGates!.some((g) => g.includes('ipSAE') && g.includes('< 0.60'))).toBe(true);
    expect(typeof res.refsByNode.score.metrics.summary.bestIpsae).toBe('number');

    // External reference comparison explicitly records 'External reference (not self-consistency)' in failedGates
    mockFoldFactory = () => ({
      kind: 'FoldedStructure',
      id: 'ipsae-folds-ext',
      parentStructureId: 'different-origin-struct',
      folds: [
        {
          sequenceId: 'ext-fold',
          sequence: 'A'.repeat(L),
          foldModelId: 'localfold-boltz2',
          pdbText: complexPdb,
          caCoords: complexCa,
          plddt: new Float32Array(L).fill(92),
          meanPlddt: 92,
          ptm: 0.9,
          iptm: 0.88,
          pae: makeAf3Pae(0.6),
          chainLengths: [nTarget, nBinder],
          chainIds,
          elapsedMs: 30,
        },
      ],
    });
    const extRes = await scheduler.run(graph, () => {}, { bypassCache: true });
    const extMetrics = extRes.outputsByNode.score.metrics as DesignabilityMetrics;
    expect(extMetrics.rows[0].designable).toBe(false);
    expect(extMetrics.rows[0].failedGates).toContain('External reference (not self-consistency)');

    await scheduler.releaseActiveGpu();
  });
});
