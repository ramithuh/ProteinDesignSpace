import { describe, expect, it } from 'vitest';
import { computePlannedFlowMap } from '$lib/engine/batch-flow';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import {
  extractStructureCollectionItems,
  selectBestGninaMember,
  wrapGninaCollection,
} from '$lib/plugins/gnina/collection';
import { runStructureCompareCpu } from '$lib/plugins/structure-compare/cpu-runner';
import { localfoldAf2Plugin } from '$lib/plugins/localfold-af2';
import type { StructureArtifact, WorkflowGraph } from '$lib/engine/types';

function makeScoredStruct(
  id: string,
  designKey: string,
  scores: { cnnScore: number; cnnAffinity: number; cnnVS: number; vinaScore: number },
): StructureArtifact {
  const caCoords = new Float32Array([0, 0, 0, 3.8, 0, 0, 7.6, 0, 0]);
  return {
    kind: 'StructureArtifact',
    id,
    label: `Fold ${designKey}`,
    sourceModelId: 'gnina',
    designKey,
    length: 3,
    sequence: 'AAA',
    caCoords,
    backbone4Coords: new Float32Array(3 * 4 * 3),
    pdbText: 'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C\nEND\n',
    atomGranularity: 'all-atom',
    geometry: computeGeometryStats(caCoords),
    dockingMetrics: {
      engine: 'gnina-1.3',
      taskMode: 'score',
      precision: 'fp16',
      ensemble: ['dense_1.3'],
      receptorName: `Fold ${designKey}`,
      includeMetals: true,
      detectedLigands: ['DHZ'],
      selectedLigand: 'DHZ',
      rows: [
        {
          poseId: 'DHZ',
          ligandCode: 'DHZ',
          ligandSource: 'holo-hetatm',
          heavyAtoms: 15,
          rotors: 2,
          cnnScore: scores.cnnScore,
          cnnAffinity: scores.cnnAffinity,
          cnnVS: scores.cnnVS,
          vinaScore: scores.vinaScore,
          models: { dense_1_3: { cnnScore: scores.cnnScore, cnnAffinity: scores.cnnAffinity } },
        },
      ],
    },
  };
}

describe('Multi-fold & structure collection fan-out (LocalFold -> GNINA / ESM-C / Dyna-1 / Compare)', () => {
  it('unpacks StructureArtifact.collection items and selects the best GNINA fold without nesting collections', () => {
    const f1 = makeScoredStruct('fold-1', 'seq-1', { cnnScore: 0.72, cnnAffinity: 6.5, cnnVS: 4.68, vinaScore: -6.1 });
    const f2 = makeScoredStruct('fold-2', 'seq-2', { cnnScore: 0.91, cnnAffinity: 7.4, cnnVS: 6.73, vinaScore: -5.4 });
    const f3 = makeScoredStruct('fold-3', 'seq-3', { cnnScore: 0.68, cnnAffinity: 7.9, cnnVS: 5.37, vinaScore: -7.8 });

    // Single structure passes through cleanly with no collection wrapper
    expect(extractStructureCollectionItems(f1)).toEqual([f1]);
    const singleWrap = wrapGninaCollection([f1], { taskMode: 'score', sortBy: 'cnnScore' });
    expect(singleWrap.structure.collection).toBeUndefined();
    expect(singleWrap.representativeIndex).toBe(0);

    // Score mode picks highest CNN_VS (f2)
    const wrappedScore = wrapGninaCollection([f1, f2, f3], { taskMode: 'score', sortBy: 'cnnScore' });
    expect(wrappedScore.representativeIndex).toBe(1);
    expect(wrappedScore.structure.designKey).toBe('seq-2');
    expect(wrappedScore.structure.representativeSelection?.criterion).toBe('highest-cnn-vs');
    expect(wrappedScore.structure.collection?.items).toHaveLength(3);
    expect(wrappedScore.structure.collection?.items.every((m) => m.collection === undefined)).toBe(true);

    // Extracting again yields all 3 flat members
    expect(extractStructureCollectionItems(wrappedScore.structure).map((s) => s.id)).toEqual([
      'fold-1',
      'fold-2',
      'fold-3',
    ]);

    // Dock mode respects sortBy = vina (lowest wins -> f3) and cnnAffinity (highest wins -> f3)
    expect(selectBestGninaMember([f1, f2, f3], 'dock', 'vina').index).toBe(2);
    expect(selectBestGninaMember([f1, f2, f3], 'dock', 'cnnAffinity').index).toBe(2);
    expect(selectBestGninaMember([f1, f2, f3], 'dock', 'cnnScore').index).toBe(1);
  });

  it('propagates LocalFold multi-fold structure cardinality into GNINA, ESM-C, Dyna-1, and Structure Compare with zero spurious collapse', () => {
    const graph: WorkflowGraph = {
      id: 'af3-gnina-fanout',
      name: 'AF3 multi-fold -> GNINA / ESM-C / Dyna-1 / Compare',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'lmpnn', adapterId: 'ligandmpnn', params: { numSequences: 4 } },
        { id: 'af3', adapterId: 'localfold-af2', params: { model: 'af3', capMode: 'follow-upstream', maxSequences: 2 } },
        { id: 'gnina', adapterId: 'gnina', params: { taskMode: 'score' } },
        { id: 'esmc', adapterId: 'esmc', params: { capMode: 'follow-upstream', mode: 'wildtype' } },
        { id: 'dyna1', adapterId: 'dyna1', params: { capMode: 'follow-upstream', inputMode: 'struct-only' } },
        { id: 'cmp', adapterId: 'structure-compare', params: { layout: 'overlay' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'lmpnn', toPort: 'structure' },
        { fromNode: 'lmpnn', fromPort: 'sequences', toNode: 'af3', toPort: 'sequences' },
        { fromNode: 'af3', fromPort: 'structure', toNode: 'gnina', toPort: 'structure' },
        { fromNode: 'af3', fromPort: 'structure', toNode: 'esmc', toPort: 'structure' },
        { fromNode: 'af3', fromPort: 'structure', toNode: 'dyna1', toPort: 'structure' },
        { fromNode: 'gnina', fromPort: 'structure', toNode: 'cmp', toPort: 'structures' },
      ],
    };

    const flow = computePlannedFlowMap(graph);
    // AF3 folds all 4 LigandMPNN sequences
    expect(flow.nodes.get('af3')?.processedCount).toBe(4);
    // GNINA no longer collapses to rep #1 — edge flows 4 -> 4 with zero truncation
    expect(flow.nodes.get('gnina')).toBeUndefined();
    const af3ToGnina = flow.edges.get('af3:structure->gnina:structure');
    expect(af3ToGnina?.incomingCount).toBe(4);
    expect(af3ToGnina?.processedCount).toBe(4);
    expect(af3ToGnina?.droppedCount).toBe(0);
    // ESM-C & Dyna-1 on structure port process all 4 under follow-upstream
    expect(flow.nodes.get('esmc')?.processedCount).toBe(4);
    expect(flow.nodes.get('esmc')?.droppedCount).toBe(0);
    expect(flow.nodes.get('dyna1')?.processedCount).toBe(4);
    expect(flow.nodes.get('dyna1')?.droppedCount).toBe(0);
    // Structure Compare receives all 4 scored structures from GNINA
    const gninaToCmp = flow.edges.get('gnina:structure->cmp:structures');
    expect(gninaToCmp?.incomingCount).toBe(4);
    expect(gninaToCmp?.processedCount).toBe(4);

    // LocalFold preflight reports matching structure & folds cardinality
    const lfPre = localfoldAf2Plugin.preflight!({
      params: { model: 'af3', capMode: 'follow-upstream', maxSequences: 2 },
      upstreamProfiles: {
        sequences: { portName: 'sequences', typeId: 'SequenceSet', itemCount: 4 },
      },
      connectedInputs: new Set(['sequences']),
    });
    expect(lfPre.outputCardinality).toEqual({ folds: 4, structure: 4 });
  });

  it('structure-compare unpacks incoming multi-fold collections flatly without nesting', async () => {
    const f1 = makeScoredStruct('rfd3-f1', 'rfd3-1', { cnnScore: 0.8, cnnAffinity: 7.0, cnnVS: 5.6, vinaScore: -6.0 });
    const f2 = makeScoredStruct('rfd3-f2', 'rfd3-2', { cnnScore: 0.85, cnnAffinity: 7.2, cnnVS: 6.1, vinaScore: -6.4 });
    const f3 = makeScoredStruct('rfd4-f1', 'rfd4-1', { cnnScore: 0.78, cnnAffinity: 6.9, cnnVS: 5.4, vinaScore: -5.9 });
    const f4 = makeScoredStruct('rfd4-f2', 'rfd4-2', { cnnScore: 0.88, cnnAffinity: 7.5, cnnVS: 6.6, vinaScore: -6.8 });

    const colA = wrapGninaCollection([f1, f2], { taskMode: 'score', sortBy: 'cnnScore' }).structure;
    const colB = wrapGninaCollection([f3, f4], { taskMode: 'score', sortBy: 'cnnScore' }).structure;

    const res = await runStructureCompareCpu(
      { id: 'cmp', adapterId: 'structure-compare', params: { layout: 'overlay' } },
      {},
      { structures: [colA, colB] },
    );
    const out = res.structure as StructureArtifact;
    expect(out.collection?.items.map((i) => i.id)).toEqual(['rfd3-f1', 'rfd3-f2', 'rfd4-f1', 'rfd4-f2']);
    expect(out.collection?.items.every((i) => i.collection === undefined)).toBe(true);
  });
});
