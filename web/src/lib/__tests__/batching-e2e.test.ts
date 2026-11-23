import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from '$lib/engine/registry';
import { CATALOG_CATEGORIES, CATALOG_ITEMS, filterCatalog, filterCatalogCategories } from '$lib/engine/catalog';
import { validateNodeParams } from '$lib/engine/param-validation';
import { defaultParamsForAdapter, validateGraph } from '$lib/engine/graph-utils';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import type {
  DesignabilityMetrics,
  FoldedStructure,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

function makeHelixCoords(length: number, radius = 2.3, pitch = 1.5, offset = [0, 0, 0]): Float32Array {
  const coords = new Float32Array(length * 3);
  for (let i = 0; i < length; i++) {
    const theta = i * 1.74532925; // ~100 deg per residue
    coords[i * 3 + 0] = offset[0] + radius * Math.cos(theta);
    coords[i * 3 + 1] = offset[1] + radius * Math.sin(theta);
    coords[i * 3 + 2] = offset[2] + i * pitch;
  }
  return coords;
}

function makeBetaStrandCoords(length: number, offset = [0, 0, 0]): Float32Array {
  const coords = new Float32Array(length * 3);
  for (let i = 0; i < length; i++) {
    coords[i * 3 + 0] = offset[0] + i * 3.3;
    coords[i * 3 + 1] = offset[1] + (i % 2 === 0 ? 0.9 : -0.9);
    coords[i * 3 + 2] = offset[2] + Math.sin(i * 0.2) * 1.5;
  }
  return coords;
}

function makeSyntheticPdb(coords: Float32Array, seq: string, chainId = 'A'): string {
  const lines: string[] = [];
  const L = Math.min(seq.length, Math.floor(coords.length / 3));
  for (let i = 0; i < L; i++) {
    const x = coords[i * 3 + 0].toFixed(3).padStart(8);
    const y = coords[i * 3 + 1].toFixed(3).padStart(8);
    const z = coords[i * 3 + 2].toFixed(3).padStart(8);
    const serial = String(i + 1).padStart(5);
    const resSeq = String(i + 1).padStart(4);
    lines.push(
      `ATOM  ${serial}  CA  ALA ${chainId}${resSeq}    ${x}${y}${z}  1.00 85.00           C  `,
    );
  }
  lines.push('END');
  return `${lines.join('\n')}\n`;
}

function makeStructure(
  id: string,
  designKey: string,
  designIndex: number,
  coords: Float32Array,
  seqChar = 'A',
): StructureArtifact {
  const L = Math.floor(coords.length / 3);
  const sequence = seqChar.repeat(L);
  const pdbText = makeSyntheticPdb(coords, sequence);
  return {
    kind: 'StructureArtifact',
    id,
    label: `Design #${designIndex + 1} (L=${L})`,
    sourceModelId: 'genie3',
    designKey,
    designIndex,
    seed: 1000 + designIndex,
    length: L,
    sequence,
    caCoords: coords,
    backbone4Coords: new Float32Array(L * 4 * 3),
    pdbText,
    chainIds: Array.from({ length: L }, () => 'A'),
    chainLengths: [L],
    geometry: computeGeometryStats(coords),
  };
}

let mockStructureOut: StructureArtifact | null = null;
let mockSeqSetOut: SequenceSet | null = null;
let mockFoldOut: FoldedStructure | null = null;

class MockWorker {
  private listeners = new Map<string, Set<(ev: MessageEvent) => void>>();
  addEventListener(type: string, cb: (ev: MessageEvent) => void) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(cb);
  }
  removeEventListener(type: string, cb: (ev: MessageEvent) => void) {
    this.listeners.get(type)?.delete(cb);
  }
  postMessage(msg: { id: number; type: string; params?: Record<string, unknown>; inputs?: Record<string, unknown> }) {
    if (msg.type === 'dispose') {
      queueMicrotask(() => {
        this.emit('message', { data: { id: msg.id, type: 'disposed' } } as MessageEvent);
      });
      return;
    }
    if (msg.type === 'run') {
      queueMicrotask(() => {
        if (msg.inputs?.sequences && mockFoldOut) {
          const firstFold = mockFoldOut.folds[0];
          const repStruct = makeStructure(
            `foldstruct-${firstFold.sequenceId}`,
            firstFold.designKey ?? 'fold-rep',
            firstFold.designIndex ?? 0,
            firstFold.caCoords,
            firstFold.sequence[0] ?? 'A',
          );
          this.emit('message', {
            data: {
              id: msg.id,
              type: 'result',
              outputs: { folds: mockFoldOut, structure: repStruct },
              elapsedMs: 5,
            },
          } as MessageEvent);
          return;
        }
        if (msg.inputs?.structure && mockSeqSetOut && !mockStructureOut) {
          this.emit('message', {
            data: { id: msg.id, type: 'result', outputs: { sequences: mockSeqSetOut }, elapsedMs: 5 },
          } as MessageEvent);
          return;
        }
        if (mockStructureOut) {
          const seqSet: SequenceSet = mockSeqSetOut ?? {
            kind: 'SequenceSet',
            id: `seq-${mockStructureOut.id}`,
            parentStructureId: mockStructureOut.id,
            sourceStructureId: mockStructureOut.id,
            sourceModelId: 'genie3',
            candidates: [
              {
                id: 'native-1',
                sequence: mockStructureOut.sequence,
                score: null,
                globalScore: null,
                sourceModelId: 'genie3',
                sourceStructureId: mockStructureOut.id,
              },
            ],
          };
          this.emit('message', {
            data: {
              id: msg.id,
              type: 'result',
              outputs: { structure: mockStructureOut, sequences: seqSet },
              elapsedMs: 5,
            },
          } as MessageEvent);
        }
      });
    }
  }
  terminate() {
    /* noop */
  }
  private emit(type: string, ev: MessageEvent) {
    for (const cb of this.listeners.get(type) ?? []) cb(ev);
  }
}

vi.mock('$lib/workers/genie3.worker?worker', () => ({
  default: class {
    constructor() {
      return new MockWorker();
    }
  },
}));
vi.mock('$lib/workers/rfd4.worker?worker', () => ({
  default: class {
    constructor() {
      return new MockWorker();
    }
  },
}));
vi.mock('$lib/plugins/web-protein-mpnn/mpnn.worker?worker', () => ({
  default: class {
    constructor() {
      return new MockWorker();
    }
  },
}));
vi.mock('$lib/workers/localfold.worker?worker', () => ({
  default: class {
    constructor() {
      return new MockWorker();
    }
  },
}));

describe('Wave-6B Multi-Design, Batching, Top-K Filter & Exact-Parent Scoring', () => {
  beforeEach(() => {
    mockStructureOut = null;
    mockSeqSetOut = null;
    mockFoldOut = null;
  });

  afterEach(() => {
    mockStructureOut = null;
    mockSeqSetOut = null;
    mockFoldOut = null;
  });

  it('declares and validates numDesigns on genie3 & rfd4-proteina, maxTotalFolds on localfold-af2, and top-k-filter in registry & catalog', () => {
    // 1. Presets remain untouched with genie3-mpnn-af2 first
    expect(WORKFLOW_PRESETS[0].id).toBe('genie3-mpnn-af2');

    // 2. numDesigns schema & bounds on genie3 and rfd4-proteina
    for (const adapterId of ['genie3', 'rfd4-proteina'] as const) {
      const field = ADAPTER_REGISTRY[adapterId].paramSchema?.find((f) => f.key === 'numDesigns');
      expect(field).toBeDefined();
      expect(field!.kind).toBe('integer');
      expect(field!.tier).toBe('primary');
      expect(field!.defaultValue).toBe(1);
      expect(field!.min).toBe(1);
      expect(field!.max).toBe(16);

      expect(validateNodeParams({ id: 'g', adapterId, params: { numDesigns: 1 } })).toEqual([]);
      expect(validateNodeParams({ id: 'g', adapterId, params: { numDesigns: 16 } })).toEqual([]);
      expect(validateNodeParams({ id: 'g', adapterId, params: { numDesigns: 0 } }).length).toBeGreaterThan(0);
      expect(validateNodeParams({ id: 'g', adapterId, params: { numDesigns: 17 } }).length).toBeGreaterThan(0);
      expect(validateNodeParams({ id: 'g', adapterId, params: { numDesigns: 2.5 } }).length).toBeGreaterThan(0);
    }

    // 3. maxTotalFolds on localfold-af2
    const lfField = ADAPTER_REGISTRY['localfold-af2'].paramSchema?.find((f) => f.key === 'maxTotalFolds');
    expect(lfField).toBeDefined();
    expect(lfField!.tier).toBe('advanced');
    expect(lfField!.defaultValue).toBe(16);
    expect(lfField!.min).toBe(1);
    expect(lfField!.max).toBe(64);
    expect(validateNodeParams({ id: 'lf', adapterId: 'localfold-af2', params: { maxTotalFolds: 32 } })).toEqual([]);
    expect(validateNodeParams({ id: 'lf', adapterId: 'localfold-af2', params: { maxTotalFolds: 65 } }).length).toBeGreaterThan(0);

    // 4. top-k-filter registry, catalog & validation
    const topk = ADAPTER_REGISTRY['top-k-filter'];
    expect(topk).toBeDefined();
    expect(topk.runtime).toBe('cpu-ts');
    expect(topk.inputs.sequences.type).toBe('SequenceSet');
    expect(topk.outputs.sequences.type).toBe('SequenceSet');
    expect(defaultParamsForAdapter('top-k-filter')).toEqual({
      mode: 'per-backbone',
      topK: 2,
      maxTotal: 8,
      metric: 'score',
      direction: 'min',
    });

    expect(CATALOG_ITEMS.some((i) => i.id === 'top-k-filter')).toBe(true);
    expect(filterCatalog('top-k').some((r) => r.item.id === 'top-k-filter')).toBe(true);
    expect(filterCatalogCategories('ranker').some((r) => r.matchingSubItems.some((s) => s.key === 'top-k-filter'))).toBe(true);
    expect(CATALOG_CATEGORIES.find((c) => c.id === 'evaluation')?.subGroups[0].items.some((i) => i.key === 'top-k-filter')).toBe(true);

    expect(validateNodeParams({ id: 'tk', adapterId: 'top-k-filter', params: defaultParamsForAdapter('top-k-filter') })).toEqual([]);
    expect(validateNodeParams({ id: 'tk', adapterId: 'top-k-filter', params: { mode: 'invalid' } }).length).toBeGreaterThan(0);
    expect(validateNodeParams({ id: 'tk', adapterId: 'top-k-filter', params: { topK: 0, maxTotal: 99 } })).toHaveLength(2);
    expect(validateNodeParams({ id: 'tk', adapterId: 'top-k-filter', params: { metric: 'bogus', direction: 'up' } })).toHaveLength(2);
  });

  it('executes top-k-filter in both per-backbone (round-robin fair cap) and global modes via ExecutionScheduler', async () => {
    const d1 = makeStructure('gen-d1', 'design:d1', 0, makeHelixCoords(20), 'A');
    const d2 = makeStructure('gen-d2', 'design:d2', 1, makeHelixCoords(20, 2.3, 1.5, [10, 0, 0]), 'C');
    mockStructureOut = {
      ...d1,
      collection: {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id: 'col-2',
        collectionKind: 'independent-candidates',
        requestedCount: 2,
        complete: true,
        items: [d1, d2],
        keys: [d1.designKey!, d2.designKey!],
      },
    };
    mockSeqSetOut = {
      kind: 'SequenceSet',
      id: 'mpnn-multi-1',
      parentStructureId: d1.id,
      sourceStructureId: d1.id,
      sourceStructureIds: [d1.id, d2.id],
      sourceModelId: 'solublempnn/v_48_020',
      lockedChains: ['B'],
      chainRoles: { A: 'binder', B: 'target' },
      collectionKind: 'independent-candidates',
      candidates: [
        // Design 1: 3 sequences (best scores overall: 1.05, 1.10, 1.15)
        { id: 'd1-s1', key: 'k-d1-1', sequenceKey: 'k-d1-1', designKey: 'design:d1', designIndex: 0, sourceStructureId: d1.id, sampleIndex: 0, sequence: 'A'.repeat(20), score: 1.05, globalScore: 1.40, sourceModelId: 'solublempnn/v_48_020' },
        { id: 'd1-s2', key: 'k-d1-2', sequenceKey: 'k-d1-2', designKey: 'design:d1', designIndex: 0, sourceStructureId: d1.id, sampleIndex: 1, sequence: 'D'.repeat(20), score: 1.10, globalScore: 1.30, sourceModelId: 'solublempnn/v_48_020' },
        { id: 'd1-s3', key: 'k-d1-3', sequenceKey: 'k-d1-3', designKey: 'design:d1', designIndex: 0, sourceStructureId: d1.id, sampleIndex: 2, sequence: 'E'.repeat(20), score: 1.15, globalScore: 1.20, sourceModelId: 'solublempnn/v_48_020' },
        // Design 2: 3 sequences (higher scores: 1.50, 1.60, null)
        { id: 'd2-s1', key: 'k-d2-1', sequenceKey: 'k-d2-1', designKey: 'design:d2', designIndex: 1, sourceStructureId: d2.id, sampleIndex: 0, sequence: 'F'.repeat(20), score: 1.50, globalScore: 1.25, sourceModelId: 'solublempnn/v_48_020' },
        { id: 'd2-s2', key: 'k-d2-2', sequenceKey: 'k-d2-2', designKey: 'design:d2', designIndex: 1, sourceStructureId: d2.id, sampleIndex: 1, sequence: 'G'.repeat(20), score: 1.60, globalScore: 1.35, sourceModelId: 'solublempnn/v_48_020' },
        { id: 'd2-s3', key: 'k-d2-3', sequenceKey: 'k-d2-3', designKey: 'design:d2', designIndex: 1, sourceStructureId: d2.id, sampleIndex: 2, sequence: 'H'.repeat(20), score: null, globalScore: null, sourceModelId: 'solublempnn/v_48_020' },
      ],
    };

    const makeGraph = (params: Record<string, unknown>): WorkflowGraph => ({
      id: 'topk-e2e',
      name: 'Top-K E2E',
      nodes: [
        { id: 'gen', adapterId: 'genie3', params: { numDesigns: 2, length: 30, nSampleStep: 10, seed: 42 } },
        { id: 'filter', adapterId: 'top-k-filter', params },
      ],
      edges: [{ fromNode: 'gen', fromPort: 'sequences', toNode: 'filter', toPort: 'sequences' }],
    });

    const scheduler = new ExecutionScheduler({ gpuIdleReleaseMs: 0 });
    try {
      // A. per-backbone mode with topK=2, maxTotal=3: round-robin keeps top-1 from d1, top-1 from d2, then #2 from d1
      const perBackboneGraph = makeGraph({
        mode: 'per-backbone',
        topK: 2,
        maxTotal: 3,
        metric: 'score',
        direction: 'min',
      });
      expect(validateGraph(perBackboneGraph)).toEqual([]);
      const resPer = await scheduler.run(perBackboneGraph, () => {});
      const outPer = resPer.outputsByNode.filter.sequences as SequenceSet;
      expect(outPer.candidates.map((c) => c.id)).toEqual(['d1-s1', 'd2-s1', 'd1-s2']);
      expect(outPer.excludedCandidates).toHaveLength(3);
      expect(outPer.lockedChains).toEqual(['B']);
      expect(outPer.chainRoles).toEqual({ A: 'binder', B: 'target' });
      expect(outPer.parentStructureId).toBe(d1.id);
      expect(outPer.filterMetadata?.mode).toBe('per-backbone');
      expect(outPer.filterMetadata?.selectedCount).toBe(3);
      expect(outPer.filterMetadata?.excludedCount).toBe(3);

      // B. global mode with topK=3, maxTotal=8 by globalScore ascending (lowest first)
      const globalGraph = makeGraph({
        mode: 'global',
        topK: 3,
        maxTotal: 8,
        metric: 'globalScore',
        direction: 'min',
      });
      const resGlob = await scheduler.run(globalGraph, () => {});
      const outGlob = resGlob.outputsByNode.filter.sequences as SequenceSet;
      // Lowest globalScores: d1-s3 (1.20), d2-s1 (1.25), d1-s2 (1.30)
      expect(outGlob.candidates.map((c) => c.id)).toEqual(['d1-s3', 'd2-s1', 'd1-s2']);
      expect(outGlob.excludedCandidates).toHaveLength(3);
      expect(outGlob.filterMetadata?.mode).toBe('global');

      // C. global mode with direction='max' picks highest finite scores, null still sorts last
      const maxGraph = makeGraph({
        mode: 'global',
        topK: 2,
        maxTotal: 8,
        metric: 'score',
        direction: 'max',
      });
      const resMax = await scheduler.run(maxGraph, () => {});
      const outMax = resMax.outputsByNode.filter.sequences as SequenceSet;
      expect(outMax.candidates.map((c) => c.id)).toEqual(['d2-s2', 'd2-s1']);
      expect(outMax.candidates.every((c) => c.score !== null)).toBe(true);
    } finally {
      await scheduler.destroy();
    }
  });

  it('scores variable-length multi-backbone collections (63 aa & 78 aa) against exact declared parents with reordered folds and fails cross-parent mismatches', async () => {
    const coords63 = makeHelixCoords(63);
    const coords78 = makeBetaStrandCoords(78, [25, -10, 5]);
    const designA = makeStructure('rfd4-d1-L63', 'design:len63', 0, coords63, 'A');
    const designB = makeStructure('rfd4-d2-L78', 'design:len78', 1, coords78, 'G');

    mockStructureOut = {
      ...designA,
      label: 'RFD4-Proteina (2 designs)',
      collection: {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id: 'col-varlen',
        collectionKind: 'independent-candidates',
        requestedCount: 2,
        complete: true,
        items: [designA, designB],
        keys: [designA.designKey!, designB.designKey!],
      },
    };

    // Slightly perturb coords (0.2 Å) and deliberately REORDER folds: 78 aa (Design B) FIRST, 63 aa (Design A) SECOND
    const fold78Coords = new Float32Array(coords78.length);
    for (let i = 0; i < coords78.length; i++) fold78Coords[i] = coords78[i] + (i % 3 === 0 ? 0.15 : 0);
    const fold63Coords = new Float32Array(coords63.length);
    for (let i = 0; i < coords63.length; i++) fold63Coords[i] = coords63[i] + (i % 3 === 1 ? 0.25 : 0);

    mockFoldOut = {
      kind: 'FoldedStructure',
      id: 'folds-varlen-reordered',
      parentStructureId: designA.id,
      sourceStructureId: designA.id,
      collectionKind: 'independent-candidates',
      folds: [
        // Deliberately placed FIRST even though designB is index 1 (78 aa)!
        {
          sequenceId: 'd2-seq-1',
          key: 'seq:d2-1',
          sequenceKey: 'seq:d2-1',
          foldKey: 'fold:d2-1',
          designKey: designB.designKey,
          designIndex: 1,
          sourceStructureId: designB.id,
          sequence: designB.sequence,
          foldModelId: 'localfold-af2-monomer-int5',
          pdbText: makeSyntheticPdb(fold78Coords, designB.sequence),
          caCoords: fold78Coords,
          plddt: new Float32Array(78).fill(91),
          meanPlddt: 91,
          confidence: 91,
          confidenceAvailable: true,
          ptm: 0.9,
          elapsedMs: 12,
        },
        // Design A (63 aa) placed SECOND
        {
          sequenceId: 'd1-seq-1',
          key: 'seq:d1-1',
          sequenceKey: 'seq:d1-1',
          foldKey: 'fold:d1-1',
          designKey: designA.designKey,
          designIndex: 0,
          sourceStructureId: designA.id,
          sequence: designA.sequence,
          foldModelId: 'localfold-af2-monomer-int5',
          pdbText: makeSyntheticPdb(fold63Coords, designA.sequence),
          caCoords: fold63Coords,
          plddt: new Float32Array(63).fill(88),
          meanPlddt: 88,
          confidence: 88,
          confidenceAvailable: true,
          ptm: 0.87,
          elapsedMs: 10,
        },
      ],
    };

    const graph: WorkflowGraph = {
      id: 'varlen-score-test',
      name: 'Variable Length Multi-Design Score Test',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', params: { numDesigns: 2, length: 70, nsteps: 60, seed: 42 } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer', recycles: 1, maxSequences: 2, maxTotalFolds: 16, seed: 0 } },
        { id: 'score', adapterId: 'designability-scorer', params: { scRmsdThreshold: 2.0, plddtThreshold: 70 } },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };

    const scheduler = new ExecutionScheduler({ gpuIdleReleaseMs: 0 });
    try {
      const res = await scheduler.run(graph, () => {});
      const metrics = res.outputsByNode.score.metrics as DesignabilityMetrics;
      expect(metrics.multiDesign).toBe(true);
      expect(metrics.evaluatedDesignCount).toBe(2);
      expect(metrics.designHitRate).toBe(1);
      expect(metrics.selfConsistencyValid).toBe(true);
      expect(metrics.lineageMismatch).toBe(false);
      expect(metrics.passRate).toBe(1);
      expect(metrics.rows).toHaveLength(2);

      const row78 = metrics.rows.find((r) => r.sequenceId === 'd2-seq-1')!;
      const row63 = metrics.rows.find((r) => r.sequenceId === 'd1-seq-1')!;
      expect(row78.structureId).toBe(designB.id);
      expect(row78.designKey).toBe(designB.designKey);
      expect(row78.alignedResidueCount).toBe(78);
      expect(row78.scRmsd).toBeLessThan(0.5);
      expect(row78.selfConsistencyValid).toBe(true);
      expect(row78.designable).toBe(true);

      expect(row63.structureId).toBe(designA.id);
      expect(row63.designKey).toBe(designA.designKey);
      expect(row63.alignedResidueCount).toBe(63);
      expect(row63.scRmsd).toBeLessThan(0.5);
      expect(row63.selfConsistencyValid).toBe(true);
      expect(row63.designable).toBe(true);

      // Now inject a cross-parent mismatch (designKey of Design A paired with sourceStructureId of Design B)
      // plus a missing/unlinked fold: both MUST fail selfConsistencyValid without falling back to array[0] or length!
      mockFoldOut = {
        ...mockFoldOut,
        id: 'folds-varlen-mismatch',
        folds: [
          // Valid Design A fold
          mockFoldOut.folds[1],
          // Cross-parent conflict: claims designKey=design:len63 AND sourceStructureId=rfd4-d2-L78
          {
            ...mockFoldOut.folds[0],
            sequenceId: 'cross-parent-conflict-seq',
            designKey: designA.designKey,
            sourceStructureId: designB.id,
          },
          // Missing parent ID (even though length 63 matches Design A!)
          {
            ...mockFoldOut.folds[1],
            sequenceId: 'missing-parent-seq',
            designKey: undefined,
            sourceStructureId: 'rfd4-nonexistent-L63',
          },
        ],
      };

      const resBad = await scheduler.run(graph, () => {}, { bypassCache: true });
      const badMetrics = resBad.outputsByNode.score.metrics as DesignabilityMetrics;
      expect(badMetrics.selfConsistencyValid).toBe(false);
      expect(badMetrics.lineageMismatch).toBe(true);

      const goodRow = badMetrics.rows.find((r) => r.sequenceId === 'd1-seq-1')!;
      const conflictRow = badMetrics.rows.find((r) => r.sequenceId === 'cross-parent-conflict-seq')!;
      const missingRow = badMetrics.rows.find((r) => r.sequenceId === 'missing-parent-seq')!;

      expect(goodRow.selfConsistencyValid).toBe(true);
      expect(goodRow.designable).toBe(true);

      expect(conflictRow.selfConsistencyValid).toBe(false);
      expect(conflictRow.lineageMismatch).toBe(true);
      expect(conflictRow.designable).toBe(false);
      expect(Number.isNaN(conflictRow.scRmsd)).toBe(true);
      expect(conflictRow.failedGates?.join(' ')).toMatch(/Cross-parent conflict/i);

      expect(missingRow.selfConsistencyValid).toBe(false);
      expect(missingRow.lineageMismatch).toBe(true);
      expect(missingRow.designable).toBe(false);
      expect(Number.isNaN(missingRow.scRmsd)).toBe(true);
      expect(missingRow.failedGates?.join(' ')).toMatch(/Missing parent backbone/i);
    } finally {
      await scheduler.destroy();
    }
  });
});
