import { describe, expect, it } from 'vitest';
import {
  computePlannedFlowMap,
  resolveEffectiveMaxSequences,
  resolveSequenceCapMode,
} from '$lib/engine/batch-flow';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { estimateRunRequirements } from '$lib/engine/run-preflight';
import { selectFoldWorkItems } from '$lib/engine/bio-capabilities';
import { decodeHashToGraph, encodeGraphToHash } from '$lib/engine/share';
import { formatNodeParamSummary } from '$lib/ui/node-display-helpers';
import type { SequenceSet, WorkflowGraph } from '$lib/engine/types';

function makeSeqSet(count: number, len = 80): SequenceSet {
  return {
    kind: 'SequenceSet',
    id: 'seq-1',
    parentStructureId: 'pdb-1',
    sourceStructureId: 'pdb-1',
    sourceModelId: 'web-protein-mpnn',
    candidates: Array.from({ length: count }, (_, i) => ({
      id: `cand_${i + 1}`,
      key: `k_${i + 1}`,
      sequence: 'A'.repeat(len),
      score: 1.0 + i * 0.1,
      globalScore: 1.0 + i * 0.1,
      sourceModelId: 'web-protein-mpnn',
      sourceStructureId: 'pdb-1',
      sampleIndex: i,
    })),
  };
}

describe('Batch-flow capMode, PlannedFlowMap, and v1/v2 share compatibility', () => {
  it('defaults newly created LocalFold, ESM-C, and Dyna-1 nodes to follow-upstream while preserving legacy explicit maxSequences', () => {
    expect(defaultParamsForAdapter('localfold-af2').capMode).toBe('follow-upstream');
    expect(defaultParamsForAdapter('esmc').capMode).toBe('follow-upstream');
    expect(defaultParamsForAdapter('dyna1').capMode).toBe('follow-upstream');

    expect(resolveSequenceCapMode('localfold-af2', { capMode: 'follow-upstream', maxSequences: 2 })).toBe(
      'follow-upstream',
    );
    expect(resolveEffectiveMaxSequences('localfold-af2', { capMode: 'follow-upstream', maxSequences: 2 }).limit).toBe(
      8,
    );

    // Legacy object with explicit maxSequences and no capMode stays fixed
    expect(resolveSequenceCapMode('localfold-af2', { maxSequences: 2 })).toBe('fixed');
    expect(resolveEffectiveMaxSequences('localfold-af2', { maxSequences: 2 }).limit).toBe(2);

    // selectFoldWorkItems folds all 8 under follow-upstream, 2 under fixed
    const seqs = makeSeqSet(8);
    expect(selectFoldWorkItems(seqs, { capMode: 'follow-upstream', maxSequences: 2 }).selected).toHaveLength(8);
    expect(selectFoldWorkItems(seqs, { capMode: 'fixed', maxSequences: 2 }).selected).toHaveLength(2);
    expect(selectFoldWorkItems(seqs, { maxSequences: 2 }).selected).toHaveLength(2);
  });

  it('enforces pass-budget cap on esmc masked mode and conservative unknown-length fallback', () => {
    expect(resolveEffectiveMaxSequences('esmc', { capMode: 'follow-upstream', mode: 'wildtype' }, { sequenceLength: 400 }).limit).toBe(8);
    const masked250 = resolveEffectiveMaxSequences('esmc', { capMode: 'follow-upstream', mode: 'masked' }, { sequenceLength: 250 });
    expect(masked250.limit).toBe(4);
    expect(masked250.isPassBudgetCapped).toBe(true);

    const maskedUnknown = resolveEffectiveMaxSequences('esmc', { capMode: 'follow-upstream', mode: 'masked' });
    expect(maskedUnknown.limit).toBe(1);
    expect(maskedUnknown.unknownLengthFallback).toBe(true);
  });

  it('computes pre-run PlannedFlowMap accurately for follow-upstream, fixed cap, multi-backbone campaign cap, and Top-K Filter exemption', () => {
    // 1. MPNN(8) -> LocalFold(follow-upstream): zero truncation, zero warning
    const followGraph: WorkflowGraph = {
      id: 'follow-8',
      name: 'Follow 8',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { numSequences: 8 } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { capMode: 'follow-upstream', maxSequences: 2 } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const followFlow = computePlannedFlowMap(followGraph);
    expect(followFlow.nodes.get('fold')?.processedCount).toBe(8);
    expect(followFlow.nodes.get('fold')?.droppedCount).toBe(0);
    expect(followFlow.nodes.get('fold')?.badgeText).toBeNull();
    expect(followFlow.downstreamImpactsBySource.get('mpnn')).toBeUndefined();
    expect(estimateRunRequirements(followGraph).plannedFolds).toBe(8);
    expect(estimateRunRequirements(followGraph).truncationNotices).toBeUndefined();

    // 2. MPNN(8) -> LocalFold(fixed 2): pre-run warning on fold card AND downstream impact on MPNN
    const fixedGraph: WorkflowGraph = {
      ...followGraph,
      nodes: [
        followGraph.nodes[0],
        followGraph.nodes[1],
        { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { capMode: 'fixed', maxSequences: 2 } },
      ],
    };
    const fixedFlow = computePlannedFlowMap(fixedGraph);
    expect(fixedFlow.nodes.get('fold')?.processedCount).toBe(2);
    expect(fixedFlow.nodes.get('fold')?.droppedCount).toBe(6);
    expect(fixedFlow.nodes.get('fold')?.badgeText).toContain('2 of 8');
    expect(fixedFlow.downstreamImpactsBySource.get('mpnn')?.[0]?.targetNodeId).toBe('fold');
    expect(fixedFlow.edges.get('mpnn:sequences->fold:sequences')?.isTruncated).toBe(true);
    expect(estimateRunRequirements(fixedGraph).truncationNotices).toHaveLength(1);

    // 3. Multi-backbone 4x8=32 -> LocalFold(follow-upstream, maxTotalFolds=16): campaign cap warning
    const multiBbGraph: WorkflowGraph = {
      id: 'multi-bb',
      name: 'Multi BB',
      nodes: [
        { id: 'gen', adapterId: 'genie3', params: { numDesigns: 4, length: 64 } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { numSequences: 8 } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { capMode: 'follow-upstream', maxTotalFolds: 16 } },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const multiFlow = computePlannedFlowMap(multiBbGraph);
    expect(multiFlow.nodes.get('fold')?.dropKind).toBe('cap-campaign');
    expect(multiFlow.nodes.get('fold')?.processedCount).toBe(16);
    expect(multiFlow.nodes.get('fold')?.droppedCount).toBe(16);
    expect(multiFlow.nodes.get('fold')?.badgeText).toContain('16 of 32');

    // 4. MPNN(8) -> Top-K Filter(2) -> LocalFold(follow-upstream): intentional filter produces ZERO warnings
    const filteredGraph: WorkflowGraph = {
      id: 'filtered',
      name: 'Filtered',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { numSequences: 8 } },
        { id: 'tk', adapterId: 'top-k-filter', params: { mode: 'per-backbone', topK: 2, maxTotal: 8 } },
        { id: 'fold', adapterId: 'localfold-af2', params: { capMode: 'follow-upstream' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'tk', toPort: 'sequences' },
        { fromNode: 'tk', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const filteredFlow = computePlannedFlowMap(filteredGraph);
    expect(filteredFlow.nodes.get('tk')?.dropKind).toBe('intentional-filter');
    expect(filteredFlow.nodes.get('tk')?.badgeText).toBeNull();
    expect(filteredFlow.downstreamImpactsBySource.get('mpnn')).toBeUndefined();
    expect(filteredFlow.nodes.get('fold')?.processedCount).toBe(2);
    expect(filteredFlow.nodes.get('fold')?.droppedCount).toBe(0);
    expect(estimateRunRequirements(filteredGraph).truncationNotices).toBeUndefined();
  });

  it('preserves v1 legacy share links as fixed cap while v2 share links round-trip follow-upstream', async () => {
    // User's actual v1 share link (AFDB Q9NZQ7 -> ProteinMPNN 8 seqs -> AF2 Model 1 default cap 2)
    const legacyV1Hash =
      '#g=dZFRS8MwFIX_SjlPCreyVqlb3iZjIKg4fLOEkDapBtK0Nq0yxv67pBvbS_t2yT333O_cHODA8hytKoRx7dCLBIQWdEDdKA0GadtvWTVWxaoAYXCm7Zr-WYFht3r73D2CUBmrnayDfL2NT8_xNomDhRW_2V1pKhwJ6-3miUXvm_gliW5OsltwyvGnCxFstXGibp0bKepAUcna2D0Yzu3QBcEN9Yf-GbQrtQdbHoOJbUppA6mQVTo6VKDDuDaNXgNKlIzblPbmy8nCWNPvhS-bTnej3oNzgp64iO-7oeyHToOmYf1FMJvHX4BpgvXancnSWBUmZ-FPgjA8jz6f_IrPw_-zPM8WlC045fcP5yJLz8UqXVC8WnJ-_Ac';
    const legacyGraph = await decodeHashToGraph(legacyV1Hash);
    expect(legacyGraph).not.toBeNull();
    const legacyFold = legacyGraph!.nodes.find((n) => n.adapterId === 'localfold-af2')!;
    expect(legacyFold.params.capMode).toBe('fixed');
    expect(resolveEffectiveMaxSequences('localfold-af2', legacyFold.params).limit).toBe(2);

    // v2 share link round-trips follow-upstream cleanly
    const v2Graph: WorkflowGraph = {
      id: 'v2-test',
      name: 'V2 Follow',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { numSequences: 8 } },
        { id: 'fold', adapterId: 'localfold-af2', params: defaultParamsForAdapter('localfold-af2') },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const encoded = await encodeGraphToHash(v2Graph);
    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    const decodedFold = decoded!.nodes.find((n) => n.adapterId === 'localfold-af2')!;
    expect(decodedFold.params.capMode).toBe('follow-upstream');
    expect(resolveEffectiveMaxSequences('localfold-af2', decodedFold.params).limit).toBe(8);
    expect(formatNodeParamSummary('localfold-af2', decodedFold.params)).toContain('Auto (≤8)');
  });
});
