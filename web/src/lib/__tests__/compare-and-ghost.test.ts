import { describe, expect, it } from 'vitest';
import { insertBlockSmart, proposeConnectionsForNode } from '$lib/engine/graph-utils';
import {
  buildComparisonState,
  exportJoinedCollectionCsv,
  joinBranchArtifacts,
  joinKeyedCollections,
} from '$lib/engine/compare';
import { generateSampleHelixBundlePdb, parsePdbBackbone } from '$lib/protein-ops/pdb';
import { applyTransformToPdb, computeGeometryStats } from '$lib/protein-ops/kabsch';
import type {
  CanonicalArtifact,
  DesignabilityMetrics,
  FoldedStructure,
  StructureArtifact,
  TrajectoryFramesArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

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

  it('supports substructure Kabsch overlay when comparing different-length structures (motif vs scaffold) and respects alignKabsch toggle', () => {
    const pdb = generateSampleHelixBundlePdb();
    const lines = pdb.split('\n');
    const shortPdb = lines
      .filter((l) => !l.startsWith('ATOM  ') || (Number(l.slice(22, 26)) >= 11 && Number(l.slice(22, 26)) <= 35))
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
      alignKabsch: true,
    });
    expect(cmp.forcedSideBySide).toBe(false);
    expect(cmp.effectiveLayout).toBe('overlay');
    expect(cmp.kabschAligned).toBe(true);
    expect(cmp.pairwise).not.toBeNull();
    expect(cmp.pairwise!.isSubstructure).toBe(true);
    expect(cmp.pairwise!.alignedResidues).toBe(25);
    expect(cmp.pairwise!.rmsd).toBeCloseTo(0, 3);
    expect(cmp.models).toHaveLength(2);

    // Turning off Kabsch align keeps raw coordinates unaligned
    const rawOverlay = buildComparisonState(graph, ['n1', 'n2'], outputsByNode, {
      layout: 'overlay',
      colorMode: 'by-model',
      alignKabsch: false,
    });
    expect(rawOverlay.effectiveLayout).toBe('overlay');
    expect(rawOverlay.kabschAligned).toBe(false);
    // Unrelated structures with no shared upstream template get qualified as Substructure Kabsch (#28)
    expect(cmp.pairwise!.hasSharedTemplate).toBe(false);
    expect(cmp.pairwise!.alignmentLabel).toBe('Substructure Kabsch');

    // When connected to a shared upstream template, substructure alignment qualifies as Motif Kabsch (#28)
    const linkedGraph: WorkflowGraph = {
      ...graph,
      edges: [{ fromNode: 'n2', fromPort: 'structure', toNode: 'n1', toPort: 'structure' }],
    };
    const linkedCmp = buildComparisonState(linkedGraph, ['n1', 'n2'], outputsByNode, {
      layout: 'overlay',
      colorMode: 'by-model',
      alignKabsch: true,
    });
    expect(linkedCmp.pairwise!.hasSharedTemplate).toBe(true);
    expect(linkedCmp.pairwise!.alignmentLabel).toBe('Motif Kabsch');
  });

  it('matches opposite binder chain orders (Genie 3 [Binder=A, Target=B] vs RFD4 [Target=A, Binder=B]) before Kabsch alignment (#9)', () => {
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
    const nTarget = 14;
    const nBinder = 10;
    const targetCa = makeHelix(nTarget, 0, 0, 0);
    const binderCa = makeHelix(nBinder, 12, 3, 2);
    const targetSeq = 'MKTAYIAKQRQISF';
    const binderSeq = 'LEEAARLLKK';

    const buildTwoChainPdb = (
      firstCa: Float32Array,
      firstSeq: string,
      secondCa: Float32Array,
      secondSeq: string,
    ): string => {
      const AA1_TO_3: Record<string, string> = {
        A: 'ALA', C: 'CYS', D: 'ASP', E: 'GLU', F: 'PHE', G: 'GLY', H: 'HIS', I: 'ILE', K: 'LYS', L: 'LEU',
        M: 'MET', N: 'ASN', P: 'PRO', Q: 'GLN', R: 'ARG', S: 'SER', T: 'THR', V: 'VAL', W: 'TRP', Y: 'TYR',
      };
      const lines: string[] = [];
      let serial = 1;
      const emit = (ca: Float32Array, seq: string, chain: string) => {
        for (let i = 0; i < seq.length; i++) {
          const x = ca[i * 3].toFixed(3).padStart(8);
          const y = ca[i * 3 + 1].toFixed(3).padStart(8);
          const z = ca[i * 3 + 2].toFixed(3).padStart(8);
          lines.push(
            `ATOM  ${String(serial++).padStart(5)}  CA  ${AA1_TO_3[seq[i]] ?? 'ALA'} ${chain}${String(i + 1).padStart(4)}    ${x}${y}${z}  1.00 90.00           C  `,
          );
        }
        lines.push('TER');
      };
      emit(firstCa, firstSeq, 'A');
      emit(secondCa, secondSeq, 'B');
      lines.push('END');
      return lines.join('\n');
    };

    // RFD4: [Target=A, Binder=B]
    const rfd4Pdb = buildTwoChainPdb(targetCa, targetSeq, binderCa, binderSeq);
    const rfd4Parsed = parsePdbBackbone(rfd4Pdb);

    // Genie 3: [Binder=A, Target=B] rigidly rotated + translated
    const rawGenie3Pdb = buildTwoChainPdb(binderCa, binderSeq, targetCa, targetSeq);
    const R90 = new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 1]);
    const tShift = new Float64Array([20, -15, 7]);
    const genie3Pdb = applyTransformToPdb(rawGenie3Pdb, R90, tShift);
    const genie3Parsed = parsePdbBackbone(genie3Pdb);

    const graph: WorkflowGraph = {
      id: 'cross-gen',
      name: 'Cross Gen Binder',
      nodes: [
        { id: 'rfd4', adapterId: 'rfd4-proteina', label: 'RFD4', params: {} },
        { id: 'g3', adapterId: 'genie3', label: 'Genie 3', params: {} },
      ],
      edges: [],
    };
    const makeStruct = (
      id: string,
      parsed: ReturnType<typeof parsePdbBackbone>,
      pdbText: string,
      chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>,
    ): StructureArtifact => ({
      kind: 'StructureArtifact',
      id,
      label: id,
      sourceModelId: id,
      length: parsed.nResidues,
      sequence: parsed.sequence,
      caCoords: parsed.caCoords,
      backbone4Coords: parsed.backbone4Coords,
      pdbText,
      chainIds: parsed.chainIds,
      chainLengths: parsed.diagnostics.chainLengths,
      chainRoles,
      geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums),
    });

    const cmp = buildComparisonState(
      graph,
      ['rfd4', 'g3'],
      {
        rfd4: { structure: makeStruct('rfd4', rfd4Parsed, rfd4Pdb, { A: 'target', B: 'binder' }) },
        g3: { structure: makeStruct('g3', genie3Parsed, genie3Pdb, { A: 'binder', B: 'target' }) },
      },
      { layout: 'overlay', colorMode: 'by-model', alignKabsch: true },
    );
    expect(cmp.pairwise).not.toBeNull();
    expect(cmp.pairwise!.rmsd).toBeLessThan(1e-2);
    expect(cmp.pairwise!.tmScore).toBeGreaterThan(0.999);

    // Aligned Genie 3 binder (its chain A, first nBinder residues) must now sit on RFD4's binder (chain B)
    const alignedGenie3 = parsePdbBackbone(cmp.models[1].pdbText);
    for (let i = 0; i < nBinder; i++) {
      expect(Math.abs(alignedGenie3.caCoords[i * 3 + 0] - binderCa[i * 3 + 0])).toBeLessThan(2e-2);
      expect(Math.abs(alignedGenie3.caCoords[i * 3 + 1] - binderCa[i * 3 + 1])).toBeLessThan(2e-2);
      expect(Math.abs(alignedGenie3.caCoords[i * 3 + 2] - binderCa[i * 3 + 2])).toBeLessThan(2e-2);
    }

    // Even WITHOUT chainRoles, equal-length swapped chains match accurately via sequence similarity
    const eqTargetSeq = 'MKTAYIAKQRQI';
    const eqBinderSeq = 'LEEAARLLKKGG';
    const eqTgtCa = makeHelix(12, 0, 0, 0);
    const eqBndCa = makeHelix(12, 11, 4, 1);
    const eqRfd4Pdb = buildTwoChainPdb(eqTgtCa, eqTargetSeq, eqBndCa, eqBinderSeq);
    const eqG3Pdb = applyTransformToPdb(buildTwoChainPdb(eqBndCa, eqBinderSeq, eqTgtCa, eqTargetSeq), R90, tShift);
    const eqRfd4Parsed = parsePdbBackbone(eqRfd4Pdb);
    const eqG3Parsed = parsePdbBackbone(eqG3Pdb);
    const seqFallbackCmp = buildComparisonState(
      graph,
      ['rfd4', 'g3'],
      {
        rfd4: { structure: makeStruct('rfd4-eq', eqRfd4Parsed, eqRfd4Pdb, undefined) },
        g3: { structure: makeStruct('g3-eq', eqG3Parsed, eqG3Pdb, undefined) },
      },
      { layout: 'overlay', colorMode: 'by-model', alignKabsch: true },
    );
    expect(seqFallbackCmp.pairwise!.rmsd).toBeLessThan(1e-2);
  });

  it('respects alignKabsch=false on shared backbones, surfaces binder/motif metrics, supports 3-block allPairwise, and renders holo/DNA mmCIF (#16, #28, #36)', () => {
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const geom = computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums);

    const makeMetrics = (id: string, rmsd: number, iptm: number, ipae: number): DesignabilityMetrics => ({
      kind: 'DesignabilityMetrics',
      id,
      passRate: 1,
      rows: [
        {
          sequenceId: `${id}-seq`,
          structureId: 'shared-gen-1',
          sequence: parsed.sequence,
          scRmsd: rmsd,
          scTm: 0.92,
          meanPlddt: 88,
          ptm: 0.85,
          iptm,
          ipae,
          binderPlddt: 84,
          targetPlddt: 91,
          binderScRmsd: rmsd + 0.1,
          targetAlignedBinderRmsd: rmsd + 0.4,
          genMotifRmsd: 0.35,
          refoldMotifRmsd: 0.55,
          designable: true,
          geometry: geom,
          foldElapsedMs: 1000,
          designedPdb: pdb,
          alignedFoldPdb: pdb,
          plddt: new Float32Array(parsed.nResidues).fill(88),
        },
      ],
      best: null,
    });

    const graph: WorkflowGraph = {
      id: 'tri-fork',
      name: 'Tri Fork',
      nodes: [
        { id: 's1', adapterId: 'designability-scorer', label: 'Score A', params: {} },
        { id: 's2', adapterId: 'designability-scorer', label: 'Score B', params: {} },
        { id: 's3', adapterId: 'designability-scorer', label: 'Score C', params: {} },
      ],
      edges: [],
    };
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      s1: { metrics: makeMetrics('m1', 0.9, 0.86, 4.2) },
      s2: { metrics: makeMetrics('m2', 1.2, 0.81, 5.8) },
      s3: { metrics: makeMetrics('m3', 1.5, 0.77, 7.1) },
    };

    const cmpNoKabsch = buildComparisonState(graph, ['s1', 's2', 's3'], outputsByNode, {
      layout: 'overlay',
      colorMode: 'by-model',
      alignKabsch: false,
    });
    expect(cmpNoKabsch.isPreAlignedSharedBackbone).toBe(true);
    expect(cmpNoKabsch.kabschAligned).toBe(false);
    expect(cmpNoKabsch.allPairwise).toHaveLength(3);
    expect(cmpNoKabsch.branches[0].iptm).toBeCloseTo(0.86, 3);
    expect(cmpNoKabsch.branches[0].ipae).toBeCloseTo(4.2, 3);
    expect(cmpNoKabsch.branches[0].binderScRmsd).toBeCloseTo(1.0, 3);
    expect(cmpNoKabsch.branches[0].targetAlignedBinderRmsd).toBeCloseTo(1.3, 3);
    expect(cmpNoKabsch.branches[0].refoldMotifRmsd).toBeCloseTo(0.55, 3);

    // Pure DNA mmCIF (pdbText: '', cifText present) renders in Compare mode both overlay & side-by-side (#16)
    const dnaCif = [
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
      '#',
    ].join('\n');
    const dnaGraph: WorkflowGraph = {
      id: 'dna-cmp',
      name: 'DNA Compare',
      nodes: [
        { id: 'd1', adapterId: 'pdb-input', label: 'DNA 1', params: {} },
        { id: 'p1', adapterId: 'pdb-input', label: 'Protein 1', params: {} },
      ],
      edges: [],
    };
    const dnaOut: Record<string, Record<string, CanonicalArtifact>> = {
      d1: {
        structure: {
          kind: 'StructureArtifact',
          id: 'dna-1',
          label: '7M5W.cif',
          sourceModelId: 'pdb-input',
          length: 0,
          sequence: '',
          caCoords: new Float32Array(0),
          backbone4Coords: new Float32Array(0),
          pdbText: '',
          cifText: dnaCif,
          diagnostics: {
            chainCount: 0,
            chainLengths: [],
            hasLigands: false,
            ligandNames: [],
            hasNucleicAcids: true,
            nucleicAcidChains: ['A'],
          },
          geometry: computeGeometryStats(new Float32Array(0)),
        },
      },
      p1: {
        structure: {
          kind: 'StructureArtifact',
          id: 'prot-1',
          label: 'Demo',
          sourceModelId: 'pdb-input',
          length: parsed.nResidues,
          sequence: parsed.sequence,
          caCoords: parsed.caCoords,
          backbone4Coords: parsed.backbone4Coords,
          pdbText: pdb,
          geometry: geom,
        },
      },
    };
    const dnaOverlay = buildComparisonState(dnaGraph, ['d1', 'p1'], dnaOut, {
      layout: 'overlay',
      colorMode: 'by-model',
    });
    expect(dnaOverlay.models).toHaveLength(2);
    expect(dnaOverlay.models[0].pdbText).toContain('data_7M5W');
    expect(dnaOverlay.models[0].pdbText).toContain('DC');

    const dnaSideBySide = buildComparisonState(dnaGraph, ['d1', 'p1'], dnaOut, {
      layout: 'side-by-side',
      colorMode: 'by-model',
    });
    expect(dnaSideBySide.models).toHaveLength(2);
    expect(dnaSideBySide.models[0].pdbText).toContain('data_7M5W');
    // Side-by-side centered the DNA mmCIF coordinates around origin (from [2.5, 3.5, 4.5] to [-1.5, -1.5, -1.5] / [1.5, 1.5, 1.5])
    expect(dnaSideBySide.models[0].pdbText).toContain('-1.500');
  });

  it('Issue #3: pairs candidates by identity across reversed branch ranks, marks missing members, and labels branch-best mode separately', () => {
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const geom = computeGeometryStats(parsed.caCoords);

    const makeRow = (sequenceId: string, scRmsd: number, meanPlddt: number) => ({
      structureId: 'shared-gen-1',
      sequenceId,
      candidateKey: sequenceId,
      sequence: sequenceId === 'cand-1' ? 'A'.repeat(parsed.nResidues) : sequenceId === 'cand-2' ? 'G'.repeat(parsed.nResidues) : 'L'.repeat(parsed.nResidues),
      scRmsd,
      scTm: 0.9,
      meanPlddt,
      ptm: 0.85,
      designable: scRmsd < 2.0 && meanPlddt >= 70,
      geometry: geom,
      foldElapsedMs: 100,
      designedPdb: pdb,
      alignedFoldPdb: pdb,
      plddt: new Float32Array(parsed.nResidues).fill(meanPlddt),
    });

    // Branch A ranks cand-2 #1, cand-1 #2, cand-3 #3
    const metricsA: DesignabilityMetrics = {
      kind: 'DesignabilityMetrics',
      id: 'm-branch-a',
      rows: [makeRow('cand-2', 0.7, 92), makeRow('cand-1', 1.4, 86), makeRow('cand-3', 1.9, 80)],
      best: null,
      passRate: 1,
    };
    // Branch B reverses rank order (cand-1 #1, cand-2 #2) and omits cand-3
    const metricsB: DesignabilityMetrics = {
      kind: 'DesignabilityMetrics',
      id: 'm-branch-b',
      rows: [makeRow('cand-1', 0.8, 90), makeRow('cand-2', 1.6, 82)],
      best: null,
      passRate: 1,
    };

    const graph: WorkflowGraph = {
      id: 'rank-rev',
      name: 'Rank Reversed',
      nodes: [
        { id: 'score_a', adapterId: 'designability-scorer', label: 'Score A', params: {} },
        { id: 'score_b', adapterId: 'designability-scorer', label: 'Score B', params: {} },
      ],
      edges: [],
    };
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      score_a: { metrics: metricsA },
      score_b: { metrics: metricsB },
    };

    // Default pairingMode is 'by-candidate' when branches share candidate identities
    const byCandDefault = buildComparisonState(graph, ['score_a', 'score_b'], outputsByNode, {
      layout: 'overlay',
      colorMode: 'by-model',
    });
    expect(byCandDefault.pairingMode).toBe('by-candidate');
    expect(byCandDefault.pairingLabel).toContain('Same candidate');
    expect(byCandDefault.selectedCandidateKey).toBe('cand-2');
    // Both branches display cand-2 (rank #1 on A, rank #2 on B!)
    expect(byCandDefault.branches[0].sequenceId).toBe('cand-2');
    expect(byCandDefault.branches[0].branchRank).toBe(1);
    expect(byCandDefault.branches[1].sequenceId).toBe('cand-2');
    expect(byCandDefault.branches[1].branchRank).toBe(2);
    expect(byCandDefault.branches[1].scRmsd).toBeCloseTo(1.6, 4);

    // Selecting cand-1 pairs cand-1 across both branches (rank #2 on A, rank #1 on B)
    const byCand1 = buildComparisonState(graph, ['score_a', 'score_b'], outputsByNode, {
      layout: 'overlay',
      colorMode: 'by-model',
      pairingMode: 'by-candidate',
      selectedCandidateKey: 'cand-1',
    });
    expect(byCand1.branches[0].sequenceId).toBe('cand-1');
    expect(byCand1.branches[0].scRmsd).toBeCloseTo(1.4, 4);
    expect(byCand1.branches[1].sequenceId).toBe('cand-1');
    expect(byCand1.branches[1].scRmsd).toBeCloseTo(0.8, 4);

    // Selecting cand-3 (absent on Branch B) explicitly marks Branch B missing instead of substituting another candidate
    const byCand3 = buildComparisonState(graph, ['score_a', 'score_b'], outputsByNode, {
      layout: 'overlay',
      colorMode: 'by-model',
      pairingMode: 'by-candidate',
      selectedCandidateKey: 'cand-3',
    });
    expect(byCand3.branches[0].sequenceId).toBe('cand-3');
    expect(byCand3.branches[0].missingSelectedCandidate).toBe(false);
    expect(byCand3.branches[1].missingSelectedCandidate).toBe(true);
    expect(byCand3.branches[1].sequenceId).toBeNull();
    expect(byCand3.branches[1].has3D).toBe(false);
    const cand3Row = byCand3.pairedCandidates.find((r) => r.key === 'cand-3');
    expect(cand3Row?.missingBranchIds).toEqual(['score_b']);
    expect(cand3Row?.byBranch.score_b.present).toBe(false);

    // Separately labeled 'branch-best' mode compares independently ranked #1 winners (cand-2 vs cand-1)
    const branchBest = buildComparisonState(graph, ['score_a', 'score_b'], outputsByNode, {
      layout: 'overlay',
      colorMode: 'by-model',
      pairingMode: 'branch-best',
    });
    expect(branchBest.pairingMode).toBe('branch-best');
    expect(branchBest.pairingLabel).toContain('independently ranked winners');
    expect(branchBest.branches[0].sequenceId).toBe('cand-2');
    expect(branchBest.branches[1].sequenceId).toBe('cand-1');
    expect(branchBest.branches[0].branchRank).toBe(1);
    expect(branchBest.branches[1].branchRank).toBe(1);
  });

  it('Issue #5: joins 3-candidate multi-branch collections with reordered & missing members, confidence-less samples, and TrajectoryFrames', () => {
    // 1. Direct joinKeyedCollections on 2 CPU fixture predictors with 3 named candidates, reversed order, 1 missing, and confidence-less sample
    const joined = joinKeyedCollections([
      {
        branchId: 'pred_a',
        label: 'CPU Predictor A',
        collectionKind: 'independent-candidates',
        items: [
          { key: 'cand-alpha', score: 0.9, confidence: 88.5, confidenceAvailable: true, value: { id: 'a1' } },
          { key: 'cand-beta', score: 1.5, confidence: null, confidenceAvailable: false, value: { id: 'a2' } },
          { key: 'cand-gamma', score: 2.1, confidence: 79.0, confidenceAvailable: true, value: { id: 'a3' } },
        ],
      },
      {
        branchId: 'pred_b',
        label: 'CPU Predictor B',
        collectionKind: 'independent-candidates',
        items: [
          { key: 'cand-beta', score: 1.1, confidence: null, confidenceAvailable: false, value: { id: 'b2' } },
          { key: 'cand-alpha', score: 1.3, confidence: 91.2, confidenceAvailable: true, value: { id: 'b1' } },
        ],
      },
    ]);

    expect(joined.keys).toEqual(['cand-alpha', 'cand-beta', 'cand-gamma']);
    expect(joined.rows).toHaveLength(3);
    expect(joined.completeRowCount).toBe(2);
    expect(joined.partialRowCount).toBe(1);

    const rowAlpha = joined.rows.find((r) => r.key === 'cand-alpha')!;
    expect(rowAlpha.byBranch.pred_a.rank).toBe(1);
    expect(rowAlpha.byBranch.pred_b.rank).toBe(2);
    expect(rowAlpha.byBranch.pred_b.confidence).toBeCloseTo(91.2, 3);

    const rowBeta = joined.rows.find((r) => r.key === 'cand-beta')!;
    expect(rowBeta.byBranch.pred_a.present).toBe(true);
    expect(rowBeta.byBranch.pred_a.confidenceAvailable).toBe(false);
    expect(rowBeta.byBranch.pred_a.confidence).toBeNull();
    expect(rowBeta.byBranch.pred_b.confidenceAvailable).toBe(false);
    expect(rowBeta.byBranch.pred_b.confidence).toBeNull();

    const rowGamma = joined.rows.find((r) => r.key === 'cand-gamma')!;
    expect(rowGamma.missingBranchIds).toEqual(['pred_b']);
    expect(rowGamma.byBranch.pred_b.present).toBe(false);
    expect(rowGamma.byBranch.pred_b.missingReason).toContain('cand-gamma');

    // 2. joinBranchArtifacts + buildComparisonState handle confidence-less multi-sample FoldedStructure & TrajectoryFramesArtifact
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const ensembleFolds: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'ens-1',
      parentStructureId: 'ref-1',
      collectionKind: 'multi-sample',
      folds: [
        {
          sequenceId: 'sample-0',
          key: 'sample-0',
          sampleIndex: 0,
          sequence: parsed.sequence,
          foldModelId: 'ensemble-sampler',
          pdbText: pdb,
          caCoords: parsed.caCoords,
          plddt: new Float32Array(0),
          meanPlddt: Number.NaN,
          confidence: null,
          confidenceAvailable: false,
          ptm: null,
          elapsedMs: 12,
        },
      ],
    };
    const trajArt: TrajectoryFramesArtifact = {
      kind: 'TrajectoryFrames',
      id: 'traj-1',
      label: 'Diffusion Trajectory',
      parentStructureId: 'ref-1',
      frameCount: 2,
      ordering: 'ordered-frames',
      collectionKind: 'ordered-frames',
      pdbText: pdb,
      frames: [
        { key: 'frame-0', frameIndex: 0, pdbText: pdb, confidence: null },
        { key: 'frame-1', frameIndex: 1, pdbText: pdb, confidence: null },
      ],
    };

    const g: WorkflowGraph = {
      id: 'ens-traj',
      name: 'Ensemble & Traj',
      nodes: [
        { id: 'f_ens', adapterId: 'localfold-af2', label: 'Ensemble', params: {} },
        { id: 't_node', adapterId: 'rfd4-proteina', label: 'Trajectory', params: {} },
      ],
      edges: [],
    };
    const outs: Record<string, Record<string, CanonicalArtifact>> = {
      f_ens: { folds: ensembleFolds },
      t_node: { trajectory: trajArt },
    };

    const artJoin = joinBranchArtifacts(g, ['f_ens', 't_node'], outs);
    expect(artJoin.keys).toEqual(['sample-0', 'frame-0', 'frame-1']);
    const cmp = buildComparisonState(g, ['f_ens', 't_node'], outs, {
      layout: 'overlay',
      colorMode: 'by-model',
    });
    expect(cmp.branches[0].collectionKind).toBe('multi-sample');
    expect(cmp.branches[0].confidenceAvailable).toBe(false);
    expect(cmp.branches[0].meanPlddt).toBeNull();
    expect(cmp.branches[1].collectionKind).toBe('ordered-frames');
    expect(cmp.branches[1].has3D).toBe(true);

    // 3. exportJoinedCollectionCsv emits paired rows with explicit MISSING and N/A markers
    const csv = exportJoinedCollectionCsv(joined);
    expect(csv).toContain('candidate_key,present_count,missing_branches');
    expect(csv).toContain('cand-beta,2,,PRESENT,2,1.5,N/A,PRESENT,1,1.1,N/A');
    expect(csv).toContain('cand-gamma,1,pred_b,PRESENT,3,2.1,79,MISSING,,,');
  });

  it('Issue #3 (Astra regression): branch-best mode picks the highest-confidence fold winner instead of index 0', () => {
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parsePdbBackbone(pdb);
    const makeFoldBranch = (id: string, scores: [number, number]): FoldedStructure => ({
      kind: 'FoldedStructure',
      id,
      parentStructureId: 'ref-1',
      folds: [
        {
          sequenceId: `${id}-low`,
          key: `${id}-low`,
          sequence: parsed.sequence,
          foldModelId: 'localfold-af2',
          pdbText: pdb,
          caCoords: parsed.caCoords,
          plddt: new Float32Array(parsed.nResidues).fill(scores[0]),
          meanPlddt: scores[0],
          confidence: scores[0],
          confidenceAvailable: true,
          ptm: 0.5,
          elapsedMs: 10,
        },
        {
          sequenceId: `${id}-high`,
          key: `${id}-high`,
          sequence: parsed.sequence,
          foldModelId: 'localfold-af2',
          pdbText: pdb,
          caCoords: parsed.caCoords,
          plddt: new Float32Array(parsed.nResidues).fill(scores[1]),
          meanPlddt: scores[1],
          confidence: scores[1],
          confidenceAvailable: true,
          ptm: 0.92,
          elapsedMs: 10,
        },
      ],
    });

    const g: WorkflowGraph = {
      id: 'fold-branch-best',
      name: 'Fold Branch Best',
      nodes: [
        { id: 'f1', adapterId: 'localfold-af2', label: 'AF2', params: {} },
        { id: 'f2', adapterId: 'localfold-af2', label: 'RF3', params: {} },
      ],
      edges: [],
    };
    const outs: Record<string, Record<string, CanonicalArtifact>> = {
      f1: { folds: makeFoldBranch('af2', [50, 95]) },
      f2: { folds: makeFoldBranch('rf3', [91, 60]) },
    };
    const cmp = buildComparisonState(g, ['f1', 'f2'], outs, {
      layout: 'overlay',
      colorMode: 'by-model',
      pairingMode: 'branch-best',
      // Even if stale per-branch indices point to non-winners, explicit 'branch-best' MUST display the true winner!
      branchCandidateIdx: { f1: 0, f2: 1 },
    });
    expect(cmp.pairingMode).toBe('branch-best');
    expect(cmp.branches[0].sequenceId).toBe('af2-high');
    expect(cmp.branches[0].meanPlddt).toBeCloseTo(95, 3);
    expect(cmp.branches[0].rankingPolicy).toBe('highest-mean-plddt');
    expect(cmp.branches[1].sequenceId).toBe('rf3-low');
    expect(cmp.branches[1].meanPlddt).toBeCloseTo(91, 3);

    // Defect #7 regression: multi-sample FoldedStructure members omitting `key` but carrying distinct `sampleIndex`
    // (or duplicate sequenceId) must NOT collapse into one row during keyed join!
    const multiSampleFolds: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'ms-1',
      parentStructureId: 'ref-1',
      collectionKind: 'multi-sample',
      folds: [
        {
          sequenceId: 'seq-1',
          sampleIndex: 0,
          sequence: parsed.sequence,
          foldModelId: 'sampler',
          pdbText: pdb,
          caCoords: parsed.caCoords,
          plddt: new Float32Array(parsed.nResidues).fill(80),
          meanPlddt: 80,
          ptm: 0.7,
          elapsedMs: 10,
        },
        {
          sequenceId: 'seq-1',
          sampleIndex: 1,
          sequence: parsed.sequence,
          foldModelId: 'sampler',
          pdbText: pdb,
          caCoords: parsed.caCoords,
          plddt: new Float32Array(parsed.nResidues).fill(92),
          meanPlddt: 92,
          ptm: 0.88,
          elapsedMs: 10,
        },
      ],
    };
    const msJoined = joinBranchArtifacts(
      {
        id: 'ms-g',
        name: 'MS',
        nodes: [{ id: 'ms', adapterId: 'localfold-af2', params: {} }],
        edges: [],
      },
      ['ms'],
      { ms: { folds: multiSampleFolds } },
    );
    expect(msJoined.rows).toHaveLength(2);
    expect(msJoined.keys).toEqual(['seq-1#sample-0', 'seq-1#sample-1']);

    // Also disambiguate duplicate keys directly in joinKeyedCollections without silently dropping members
    // or hanging when existing keys collide with generated suffixes (e.g. ['x', 'x#2', 'x#4', 'x'])!
    const dupJoined = joinKeyedCollections([
      {
        branchId: 'b1',
        items: [
          { key: 'x', value: 1 },
          { key: 'x#2', value: 2 },
          { key: 'x#4', value: 3 },
          { key: 'x', value: 4 },
        ],
      },
    ]);
    expect(dupJoined.rows).toHaveLength(4);
    expect(dupJoined.keys).toEqual(['x', 'x#2', 'x#4', 'x#3']);
  });

  it('Wave-2 P1/P2 regressions: symmetric equal-sequence homodimer Kabsch tie-breaker and non-finite Kabsch guard', () => {
    // 1. Symmetric equal-sequence homodimer: chain A & B have identical sequence ('ACDEF') but distinct 3D positions.
    // Model 2 swaps chain A and B in coordinate order; Kabsch tie-breaker must recover ~0 Å RMSD instead of mis-pairing.
    const chainACoords = [
      [0, 0, 0],
      [3.8, 0, 0],
      [7.6, 1.2, 0],
      [11.4, 0, 1.0],
      [15.2, 1.5, 2.0],
    ];
    const chainBCoords = [
      [0, 20, 5],
      [2.5, 22.8, 5],
      [5.5, 20.5, 6.2],
      [8.5, 23.0, 5.0],
      [11.8, 21.0, 7.5],
    ];
    const makeDimer = (id: string, first: number[][], second: number[][]): StructureArtifact => {
      const ca = new Float32Array(10 * 3);
      for (let i = 0; i < 5; i++) {
        ca[i * 3 + 0] = first[i][0];
        ca[i * 3 + 1] = first[i][1];
        ca[i * 3 + 2] = first[i][2];
        ca[(5 + i) * 3 + 0] = second[i][0];
        ca[(5 + i) * 3 + 1] = second[i][1];
        ca[(5 + i) * 3 + 2] = second[i][2];
      }
      const chainIds = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'];
      return {
        kind: 'StructureArtifact',
        id,
        label: id,
        sourceModelId: 'mock',
        length: 10,
        sequence: 'ACDEFACDEF',
        chainIds,
        caCoords: ca,
        backbone4Coords: new Float32Array(10 * 4 * 3),
        pdbText: 'END\n',
        geometry: {
          length: 10,
          rg: 10,
          meanCaDist: 3.8,
          chainBreaks: 0,
          clashes: 0,
          helixPct: 0,
          sheetPct: 0,
          coilPct: 100,
        },
      };
    };

    const s1 = makeDimer('dimer-1', chainACoords, chainBCoords);
    const s2 = makeDimer('dimer-2', chainBCoords, chainACoords); // chains swapped!
    const g: WorkflowGraph = {
      id: 'homodimer-cmp',
      name: 'Homodimer',
      nodes: [
        { id: 'n1', adapterId: 'pdb-input', label: 'Dimer 1', params: {} },
        { id: 'n2', adapterId: 'pdb-input', label: 'Dimer 2', params: {} },
      ],
      edges: [],
    };
    const cmp = buildComparisonState(g, ['n1', 'n2'], { n1: { structure: s1 }, n2: { structure: s2 } }, {
      layout: 'overlay',
      colorMode: 'by-model',
    });
    expect(cmp.allPairwise).toHaveLength(1);
    expect(cmp.allPairwise[0].rmsd).toBeLessThan(1e-3);
    expect(cmp.kabschAligned).toBe(true);

    // 2. Non-finite coordinates in a branch must NOT emit NaN pairwise rows or apply NaN transforms
    const badCa = new Float32Array(s1.caCoords);
    badCa[0] = Number.NaN;
    const sBad: StructureArtifact = { ...s2, id: 'dimer-nan', caCoords: badCa };
    const cmpBad = buildComparisonState(g, ['n1', 'n2'], { n1: { structure: s1 }, n2: { structure: sBad } }, {
      layout: 'overlay',
      colorMode: 'by-model',
    });
    expect(cmpBad.allPairwise).toEqual([]);
    expect(cmpBad.kabschAligned).toBe(false);
  });

  it('Wave-3 ipSAE surfacing in CompareBranchSummary/CompareBranchMetrics and exportJoinedCollectionCsv', () => {
    const pdbLines: string[] = [];
    for (let i = 0; i < 6; i++) {
      const ch = i < 3 ? 'A' : 'B';
      const rn = i < 3 ? i + 1 : i - 2;
      pdbLines.push(
        `ATOM  ${String(i + 1).padStart(5)}  CA  ALA ${ch}${String(rn).padStart(4)}    ${(i * 3.8).toFixed(3).padStart(8)}   0.000   0.000  1.00 90.00           C  `,
      );
    }
    pdbLines.push('END');
    const dimerPdb = pdbLines.join('\n');
    const parsed = parsePdbBackbone(dimerPdb);
    const geom = computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums);
    const pae = new Float32Array(36);
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        pae[i * 6 + j] = (i < 3) === (j < 3) ? 0.5 : 0.8;
      }
    }

    const metricsBranch: DesignabilityMetrics = {
      kind: 'DesignabilityMetrics',
      id: 'm-ipsae',
      passRate: 1,
      rows: [
        {
          structureId: 'ref-dimer',
          sequenceId: 'cand-1',
          candidateKey: 'cand-1',
          sequence: parsed.sequence,
          scRmsd: 0.75,
          scTm: 0.94,
          meanPlddt: 91,
          ptm: 0.88,
          iptm: 0.84,
          ipae: 0.8,
          ipsae: 0.625,
          ipsaeMin: 0.625,
          designable: true,
          failedGates: [],
          alignedFoldPdb: dimerPdb,
          designedPdb: dimerPdb,
          pae,
          plddt: new Float32Array(6).fill(91),
          geometry: geom,
          foldElapsedMs: 20,
        },
      ],
      best: null,
    };

    const foldBranch: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'f-ipsae',
      parentStructureId: 'ref-dimer',
      folds: [
        {
          sequenceId: 'cand-1',
          key: 'cand-1',
          sequence: parsed.sequence,
          foldModelId: 'localfold-multimer',
          pdbText: dimerPdb,
          caCoords: parsed.caCoords,
          chainLengths: [3, 3],
          chainIds: parsed.chainIds,
          pae,
          plddt: new Float32Array(6).fill(89),
          meanPlddt: 89,
          ptm: 0.86,
          iptm: 0.82,
          elapsedMs: 25,
        },
      ],
    };

    const g: WorkflowGraph = {
      id: 'ipsae-cmp',
      name: 'ipSAE Compare',
      nodes: [
        { id: 'sc', adapterId: 'designability-scorer', label: 'Scorer', params: {} },
        { id: 'lf', adapterId: 'localfold-af2', label: 'Fold', params: {} },
      ],
      edges: [],
    };
    const outs: Record<string, Record<string, CanonicalArtifact>> = {
      sc: { metrics: metricsBranch },
      lf: { folds: foldBranch },
    };

    const cmp = buildComparisonState(g, ['sc', 'lf'], outs, {
      layout: 'overlay',
      colorMode: 'by-model',
    });
    expect(cmp.branches[0].ipsae).toBeCloseTo(0.625, 4);
    expect(cmp.branches[0].ipsaeMin).toBeCloseTo(0.625, 4);
    expect(cmp.branches[1].ipsae).not.toBeNull();
    expect(cmp.branches[1].ipsae!).toBeGreaterThan(0.6);
    expect(cmp.models.some((m) => m.metricSummary?.includes('ipSAE'))).toBe(true);

    const joined = joinBranchArtifacts(g, ['sc', 'lf'], outs);
    const csv = exportJoinedCollectionCsv(joined);
    expect(csv).toContain('sc_ipsae');
    expect(csv).toContain('lf_ipsae');
    expect(csv).toContain('0.625');
  });
});
