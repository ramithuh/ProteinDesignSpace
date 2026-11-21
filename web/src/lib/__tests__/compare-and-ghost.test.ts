import { describe, expect, it } from 'vitest';
import { insertBlockSmart, proposeConnectionsForNode } from '$lib/engine/graph-utils';
import { buildComparisonState } from '$lib/engine/compare';
import { generateSampleHelixBundlePdb, parsePdbBackbone } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import type { CanonicalArtifact, DesignabilityMetrics, WorkflowGraph } from '$lib/engine/types';

describe('ghost auto-connection proposals', () => {
  it('autoCommit: false returns proposedEdges without mutating graph.edges', () => {
    const base: WorkflowGraph = {
      id: 'g1',
      name: 'Linear',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4', params: {}, position: { x: 40, y: 80 } },
      ],
      edges: [],
    };
    const res = insertBlockSmart(base, 'web-protein-mpnn', 'gen', undefined, { autoCommit: false });
    expect(res.graph.edges).toEqual([]);
    expect(res.proposedEdges).toEqual([
      { fromNode: 'gen', fromPort: 'structure', toNode: res.nodeId, toPort: 'structure' },
    ]);
  });

  it('proposes BOTH folds and ancestor structure wires when inserting scRMSD/scTM after LocalFold', () => {
    const base: WorkflowGraph = {
      id: 'g2',
      name: 'Chain',
      nodes: [
        { id: 'gen', adapterId: 'genie3', label: 'Genie 3', params: {}, position: { x: 40, y: 80 } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'MPNN', params: {}, position: { x: 320, y: 80 } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'LocalFold', params: {}, position: { x: 600, y: 80 } },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };

    const res = insertBlockSmart(base, 'designability-scorer', 'fold', undefined, { autoCommit: false });
    expect(res.graph.edges).toHaveLength(2);
    expect(res.proposedEdges).toHaveLength(2);
    expect(res.proposedEdges).toContainEqual({
      fromNode: 'fold',
      fromPort: 'folds',
      toNode: res.nodeId,
      toPort: 'folds',
    });
    expect(res.proposedEdges).toContainEqual({
      fromNode: 'gen',
      fromPort: 'structure',
      toNode: res.nodeId,
      toPort: 'structure',
    });
  });

  it('proposes 0 wires when drag-dropping into open space on a complex graph (>= 4 nodes)', () => {
    const complex: WorkflowGraph = {
      id: 'g3',
      name: 'Complex',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4', params: {}, position: { x: 40, y: 60 } },
        { id: 'f1', adapterId: 'localfold-af2', label: 'AF2', params: {}, position: { x: 340, y: 40 } },
        { id: 'f2', adapterId: 'localfold-af2', label: 'RF3', params: {}, position: { x: 340, y: 240 } },
        { id: 's1', adapterId: 'designability-scorer', label: 'Score1', params: {}, position: { x: 640, y: 40 } },
      ],
      edges: [],
    };
    const res = insertBlockSmart(
      complex,
      'web-protein-mpnn',
      'gen',
      { x: 900, y: 700 },
      { autoCommit: false },
    );
    expect(res.proposedEdges).toEqual([]);
    expect(proposeConnectionsForNode(res.graph, res.nodeId, 'gen', { x: 900, y: 700 })).toEqual([]);
  });
});

describe('multi-select 3D comparison state', () => {
  it('builds a shared-backbone overlay with pairwise Cα RMSD for two scorer branches', () => {
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const geom = computeGeometryStats(parsed.caCoords);

    const makeMetrics = (id: string, rmsd: number, plddt: number): DesignabilityMetrics => ({
      kind: 'DesignabilityMetrics',
      id,
      passRate: 1,
      rows: [
        {
          sequenceId: `${id}-seq`,
          structureId: 'shared-gen-1',
          sequence: parsed.sequence,
          scRmsd: rmsd,
          scTm: 0.91,
          meanPlddt: plddt,
          ptm: 0.85,
          designable: true,
          geometry: geom,
          foldElapsedMs: 1200,
          designedPdb: pdb,
          alignedFoldPdb: pdb,
          plddt: new Float32Array(parsed.nResidues).fill(plddt),
        },
      ],
      best: null,
    });

    const graph: WorkflowGraph = {
      id: 'fork',
      name: 'Fork',
      nodes: [
        { id: 'score_af2', adapterId: 'designability-scorer', label: 'Score (AF2)', params: {} },
        { id: 'score_rf3', adapterId: 'designability-scorer', label: 'Score (RF3)', params: {} },
      ],
      edges: [],
    };

    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      score_af2: { metrics: makeMetrics('m-af2', 1.1, 88) },
      score_rf3: { metrics: makeMetrics('m-rf3', 1.4, 84) },
    };

    const cmp = buildComparisonState(graph, ['score_af2', 'score_rf3'], outputsByNode, {
      layout: 'overlay',
      colorMode: 'by-model',
    });

    expect(cmp.effectiveLayout).toBe('overlay');
    expect(cmp.sharedReferenceLabel).toBe('Designed Backbone');
    // 1 shared designed reference + 2 refold branches
    expect(cmp.models).toHaveLength(3);
    expect(cmp.models[0].isReference).toBe(true);
    expect(cmp.models[1].badge).toBe('A');
    expect(cmp.models[2].badge).toBe('B');
    expect(cmp.pairwise).not.toBeNull();
    expect(cmp.pairwise!.rmsd).toBeCloseTo(0, 4);
  });

  it('automatically uses side-by-side layout when comparing structures of different lengths', () => {
    const pdb = generateSampleHelixBundlePdb();
    const lines = pdb.split('\n');
    const shortPdb = lines
      .filter((l) => !l.startsWith('ATOM  ') || Number(l.slice(22, 26)) <= 25)
      .join('\n');
    const pLong = parsePdbBackbone(pdb);
    const pShort = parsePdbBackbone(shortPdb);

    const graph: WorkflowGraph = {
      id: 'diff-len',
      name: 'Diff',
      nodes: [
        { id: 'n1', adapterId: 'pdb-input', label: 'Long', params: {} },
        { id: 'n2', adapterId: 'pdb-input', label: 'Short', params: {} },
      ],
      edges: [],
    };
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      n1: {
        structure: {
          kind: 'StructureArtifact',
          id: 's1',
          label: 'Long',
          sourceModelId: 'pdb-input',
          length: pLong.nResidues,
          sequence: pLong.sequence,
          caCoords: pLong.caCoords,
          backbone4Coords: pLong.backbone4Coords,
          pdbText: pdb,
          geometry: computeGeometryStats(pLong.caCoords),
        },
      },
      n2: {
        structure: {
          kind: 'StructureArtifact',
          id: 's2',
          label: 'Short',
          sourceModelId: 'pdb-input',
          length: pShort.nResidues,
          sequence: pShort.sequence,
          caCoords: pShort.caCoords,
          backbone4Coords: pShort.backbone4Coords,
          pdbText: shortPdb,
          geometry: computeGeometryStats(pShort.caCoords),
        },
      },
    };

    const cmp = buildComparisonState(graph, ['n1', 'n2'], outputsByNode, {
      layout: 'overlay',
      colorMode: 'by-model',
    });
    expect(cmp.forcedSideBySide).toBe(true);
    expect(cmp.effectiveLayout).toBe('side-by-side');
    expect(cmp.pairwise).toBeNull();
    expect(cmp.models).toHaveLength(2);
  });
});
