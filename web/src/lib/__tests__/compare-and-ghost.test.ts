import { describe, expect, it } from 'vitest';
import { insertBlockSmart, proposeConnectionsForNode } from '$lib/engine/graph-utils';
import { buildComparisonState } from '$lib/engine/compare';
import { generateSampleHelixBundlePdb, parsePdbBackbone } from '$lib/protein-ops/pdb';
import { applyTransformToPdb, computeGeometryStats } from '$lib/protein-ops/kabsch';
import type { CanonicalArtifact, DesignabilityMetrics, StructureArtifact, WorkflowGraph } from '$lib/engine/types';

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
});
