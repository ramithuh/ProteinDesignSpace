import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { resolveUncertaintyColorArgFromValues } from '$lib/viewer/bfactor-domain';
import { computeTrackTooltipPosition } from '$lib/viewer/tooltip-position';
import {
  coordsToBackbonePdb,
  cropParsedStructure,
  cropRawAllAtomStructure,
  mmcifToPdb,
  parsePdbBackbone,
  parseStructureText,
} from '$lib/protein-ops/pdb';
import { buildComparisonState, joinKeyedCollections } from '$lib/engine/compare';
import { filterCatalog, filterCatalogCategories } from '$lib/engine/catalog';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import {
  isCompareDisplayOnlyParamChange,
  resolveResidueFocusToken,
} from '$lib/ui/page-state-helpers';
import {
  computeUpstreamLockSummary,
  formatMutationMatrixBadge,
  formatNodeParamSummary,
} from '$lib/ui/node-display-helpers';
import type {
  CanonicalArtifact,
  DesignabilityMetrics,
  FoldedStructure,
  MutationMatrixArtifact,
  ResidueTrackArtifact,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

function readWebFile(relPath: string): string {
  return fs.readFileSync(path.resolve(__dirname, '../../..', relPath), 'utf8');
}

function makeLinearCaCoords(n: number, shiftX = 0, shiftY = 0, shiftZ = 0): Float32Array {
  const ca = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    ca[i * 3 + 0] = shiftX + i * 3.8;
    ca[i * 3 + 1] = shiftY + Math.sin(i * 0.7) * 1.6;
    ca[i * 3 + 2] = shiftZ + Math.cos(i * 0.5) * 1.4;
  }
  return ca;
}

function makeLinearBackbone4(ca: Float32Array): Float32Array {
  const n = ca.length / 3;
  const bb = new Float32Array(n * 4 * 3);
  for (let i = 0; i < n; i++) {
    const x = ca[i * 3 + 0];
    const y = ca[i * 3 + 1];
    const z = ca[i * 3 + 2];
    bb[(i * 4 + 0) * 3 + 0] = x - 1.2;
    bb[(i * 4 + 0) * 3 + 1] = y + 0.5;
    bb[(i * 4 + 0) * 3 + 2] = z;
    bb[(i * 4 + 1) * 3 + 0] = x;
    bb[(i * 4 + 1) * 3 + 1] = y;
    bb[(i * 4 + 1) * 3 + 2] = z;
    bb[(i * 4 + 2) * 3 + 0] = x + 1.2;
    bb[(i * 4 + 2) * 3 + 1] = y + 0.4;
    bb[(i * 4 + 2) * 3 + 2] = z;
    bb[(i * 4 + 3) * 3 + 0] = x + 1.4;
    bb[(i * 4 + 3) * 3 + 1] = y + 1.6;
    bb[(i * 4 + 3) * 3 + 2] = z;
  }
  return bb;
}

function makeStructArtifact(
  id: string,
  seq: string,
  opts: { shiftX?: number; shiftY?: number; shiftZ?: number; plddt?: number } = {},
): StructureArtifact {
  const n = seq.length;
  const ca = makeLinearCaCoords(n, opts.shiftX ?? 0, opts.shiftY ?? 0, opts.shiftZ ?? 0);
  const bb = makeLinearBackbone4(ca);
  const bFactors = new Float32Array(n).fill(opts.plddt ?? 85);
  const pdbText = coordsToBackbonePdb(bb, n, { atomsPerRes: 4, sequence: seq, bFactors });
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'mock',
    length: n,
    sequence: seq,
    chainIds: Array(n).fill('A'),
    caCoords: ca,
    backbone4Coords: bb,
    pdbText,
    geometry: {
      length: n,
      rg: 12,
      meanCaDist: 3.8,
      chainBreaks: 0,
      clashes: 0,
      helixPct: 20,
      sheetPct: 10,
      coilPct: 70,
    },
  };
}

describe('M3 §2.1: Mol* B-factor domain auto-ranging (bfactor-domain.ts)', () => {
  it('auto-ranges ddg-stability without collapsing non-negative values into [0, 100]', () => {
    // Mixed stabilizing/destabilizing (e.g. 1UBQ best-stabilizing) -> tightened [-2, 2] / [-1.5, 1.5]
    expect(resolveUncertaintyColorArgFromValues([-1.36, -0.4, 0.2], 'ddg-stability')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [-1.5, 1.5] },
    });
    expect(resolveUncertaintyColorArgFromValues([-3.5, 0.0, 3.5], 'ddg-stability')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [-2, 2] },
    });

    // Non-negative kcal/mol (mean-ddg / worst-destabilizing): fits [0, hi] instead of [0, 100]
    expect(resolveUncertaintyColorArgFromValues([0.1, 1.4, 2.8], 'ddg-stability')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [0, 2.8] },
    });
    expect(resolveUncertaintyColorArgFromValues([0.5, 8.0], 'ddg-stability')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [0, 4] },
    });
    expect(resolveUncertaintyColorArgFromValues([0, 0, 0], 'ddg-stability')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [0, 1.5] },
    });
    expect(resolveUncertaintyColorArgFromValues([4.5, 9.5], 'ddg-stability')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [4.5, 9.5] },
    });
    // Ignores minority crystal B-factor outliers (>15) on unpainted chains
    expect(
      resolveUncertaintyColorArgFromValues([0.2, 1.1, 2.5, 60.0], 'ddg-stability'),
    ).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [0, 2.5] },
    });
  });

  it('auto-ranges dynamics-prob ignoring minority crystal B-factor outliers', () => {
    expect(resolveUncertaintyColorArgFromValues([0.05, 0.5, 0.95], 'dynamics-prob')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [0, 1] },
    });
    expect(
      resolveUncertaintyColorArgFromValues([0.05, 0.4, 0.92, 48.0], 'dynamics-prob'),
    ).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [0, 1] },
    });
    expect(resolveUncertaintyColorArgFromValues([15, 50, 85], 'dynamics-prob')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [0, 100] },
    });
  });

  it('auto-ranges llr-fitness around 0 and filters crystal B-factor outliers', () => {
    expect(resolveUncertaintyColorArgFromValues([-3.2, -0.5, 0.4], 'llr-fitness')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [-3.2, 3.2] },
    });
    expect(resolveUncertaintyColorArgFromValues([-12.0, 0.2, 55.0], 'llr-fitness')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [-6, 6] },
    });
    expect(resolveUncertaintyColorArgFromValues([0.2, 2.4, 4.1], 'llr-fitness')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [0, 4.1] },
    });
    expect(resolveUncertaintyColorArgFromValues([0.1, 1.2, 3.0, 62.0], 'llr-fitness')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [0, 3.0] },
    });
  });

  it('fits observed [min, max] for crystal-b / undefined and falls back for constant values', () => {
    expect(resolveUncertaintyColorArgFromValues([5, 18, 38], 'crystal-b')).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [5, 38] },
    });
    expect(resolveUncertaintyColorArgFromValues([120, 240], undefined)).toEqual({
      color: 'uncertainty',
      colorParams: { domain: [120, 240] },
    });
    expect(resolveUncertaintyColorArgFromValues([50, 50], 'crystal-b')).toEqual({
      color: 'uncertainty',
    });
    expect(resolveUncertaintyColorArgFromValues([], 'crystal-b')).toEqual({
      color: 'uncertainty',
    });

    const viewerSrc = readWebFile('src/lib/viewer/MolSuperposeViewer.svelte');
    expect(viewerSrc).toContain('resolveUncertaintyColorArgFromValues');
    expect(viewerSrc).toContain("=== 'llr-fitness'");
  });
});

describe('M3 §2.5a & Bio P0-2a/P1-3: mmcifToPdb, peptide caps, and crop chain hints', () => {
  it('mmcifToPdb preserves sidechains, non-water HETATMs, nucleic acids, B-factors, occupancy, altLoc & multi-char chains', () => {
    const cif = [
      'data_full_test',
      ';',
      '_atom_site.group_PDB fake_in_semicolon',
      ';',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.occupancy',
      '_atom_site.B_iso_or_equiv',
      '_atom_site.pdbx_PDB_model_num',
      'ATOM 1 N N . ALA AA 1 ? 0.000 0.000 0.000 1.00 12.50 1',
      'ATOM 2 C CA B ALA AA 1 ? 99.000 0.000 0.000 0.50 88.00 1',
      'ATOM 3 C CA A ALA AA 1 ? 1.458 0.000 0.000 0.50 14.25 1',
      'ATOM 4 C CB . ALA AA 1 ? 2.000 -1.200 0.500 1.00 19.75 1',
      'ATOM 5 O "O5\'" . DA AB 10 A 10.000 2.000 3.000 1.00 22.00 1',
      'HETATM 6 ZN ZN . ZN Z 201 ? 3.500 1.500 1.500 1.00 9.50 1',
      'HETATM 7 C C1 . DHZ Z 202 ? 4.500 2.500 2.500 1.00 18.00 1',
      'HETATM 8 O O . HOH Z 301 ? 15.000 15.000 15.000 1.00 45.00 1',
      'ATOM 9 C CA . GLY AA 2 ? 4.000 0.000 0.000 1.00 11.00 2',
      '#',
    ].join('\n');

    const pdbFirst = mmcifToPdb(cif, { firstModelOnly: true });
    expect(pdbFirst).toContain(' CB  ALA A');
    expect(pdbFirst).toContain('1.458');
    expect(pdbFirst).not.toContain('99.000'); // altLoc A beat altLoc B even when B came first
    expect(pdbFirst).toContain('14.25');
    expect(pdbFirst).toContain('19.75');
    expect(pdbFirst).toContain("O5'");
    expect(pdbFirst).toContain('HETATM');
    expect(pdbFirst).toContain(' ZN ');
    expect(pdbFirst).toContain(' DHZ ');
    expect(pdbFirst).not.toContain(' HOH ');
    expect(pdbFirst).not.toContain('MODEL');
    expect(pdbFirst).toContain('TER');
    expect(pdbFirst).toContain('END');

    const pdbWithWaterAndModels = mmcifToPdb(cif, { includeWaters: true, firstModelOnly: false });
    expect(pdbWithWaterAndModels).toContain(' HOH ');
    expect(pdbWithWaterAndModels).toMatch(/^MODEL\s+1$/m);
    expect(pdbWithWaterAndModels).toMatch(/^MODEL\s+2$/m);
    expect(pdbWithWaterAndModels).toContain('ENDMDL');

    // parseStructureText(cif).pdbText now includes sidechains and HETATMs too
    const parsed = parseStructureText(cif);
    expect(parsed.pdbText).toContain(' CB  ALA A');
    expect(parsed.pdbText).toContain(' DHZ ');
  });

  it('ArtifactPeekPanel wires mmcifToPdb into ↓ PDB download', () => {
    const peekSrc = readWebFile('src/lib/ui/ArtifactPeekPanel.svelte');
    expect(peekSrc).toContain('mmcifToPdb');
    expect(peekSrc).toContain('cifSource ? mmcifToPdb(cifSource) : activeStruct.pdbText');
  });

  it('Bio P0-2a: ignores ACE/NME/NH2/FOR terminal caps in ligandNames and sequence', () => {
    const pdbWithCaps = [
      'HETATM    1  C   ACE A   0       0.000   0.000   0.000  1.00 10.00           C  ',
      'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C  ',
      'ATOM      3  CA  GLY A   2       5.200   0.000   0.000  1.00 10.00           C  ',
      'ATOM      4  CA  SER A   3       9.000   0.000   0.000  1.00 10.00           C  ',
      'HETATM    5  N   NME A   4      10.500   0.000   0.000  1.00 10.00           N  ',
      'ATOM      6  CA  FOR A   5      12.000   0.000   0.000  1.00 10.00           C  ',
      'ATOM      7  NH2 NH2 A   6      13.500   0.000   0.000  1.00 10.00           N  ',
      'END',
    ].join('\n');
    const parsedPdb = parsePdbBackbone(pdbWithCaps);
    expect(parsedPdb.sequence).toBe('AGS');
    expect(parsedPdb.diagnostics.hasLigands).toBe(false);
    expect(parsedPdb.diagnostics.ligandNames).toEqual([]);

    const cifWithCaps = [
      'data_caps',
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
      'HETATM 1 C ACE A 0 0.000 0.000 0.000',
      'ATOM 2 CA ALA A 1 1.458 0.000 0.000',
      'ATOM 3 CA GLY A 2 5.200 0.000 0.000',
      'ATOM 4 CA SER A 3 9.000 0.000 0.000',
      'HETATM 5 N NME A 4 10.500 0.000 0.000',
      'ATOM 6 CA FOR A 5 12.000 0.000 0.000',
      '#',
    ].join('\n');
    const parsedCif = parseStructureText(cifWithCaps);
    expect(parsedCif.sequence).toBe('AGS');
    expect(parsedCif.diagnostics.hasLigands).toBe(false);
    expect(parsedCif.diagnostics.ligandNames).toEqual([]);
  });

  it('Bio P1-3: empty/unmatched segment crop errors list Available chains with residue ranges', () => {
    const pdb3Chain = [
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  ',
      'ATOM      2  CA  GLY A  50       3.800   0.000   0.000  1.00 10.00           C  ',
      'ATOM      3  CA  SER B   5      10.000   0.000   0.000  1.00 10.00           C  ',
      'ATOM      4  CA  VAL B  90      13.800   0.000   0.000  1.00 10.00           C  ',
      'ATOM      5  CA  LEU C 100      20.000   0.000   0.000  1.00 10.00           C  ',
      'END',
    ].join('\n');
    expect(() => cropRawAllAtomStructure(pdb3Chain, 'H')).toThrow(
      /Available chains: A \(1-50\), B \(5-90\), C \(100-100\)\./,
    );
    expect(() => cropRawAllAtomStructure(pdb3Chain, 'A, Z999-1000')).toThrow(
      /Available chains: A \(1-50\), B \(5-90\), C \(100-100\)\./,
    );

    const cif3Chain = [
      'data_3ch',
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
      'ATOM 1 CA ALA A 10 0.000 0.000 0.000',
      'ATOM 2 CA GLY A 20 3.800 0.000 0.000',
      'ATOM 3 CA SER B 1 10.000 0.000 0.000',
      'ATOM 4 CA VAL B 30 13.800 0.000 0.000',
      '#',
    ].join('\n');
    expect(() => cropRawAllAtomStructure(cif3Chain, 'H')).toThrow(
      /Available chains: A \(10-20\), B \(1-30\)\./,
    );
    expect(() => cropRawAllAtomStructure(cif3Chain, 'A, H')).toThrow(
      /Available chains: A \(10-20\), B \(1-30\)\./,
    );

    const parsed = parsePdbBackbone(pdb3Chain);
    expect(() => cropParsedStructure(parsed, 'H')).toThrow(
      /Available chains: A \(1-50\), B \(5-90\), C \(100-100\)\./,
    );
  });
});

describe('M3 §2.2: 3D Superpose / Compare pinned reference & sequence fallback', () => {
  it('pins single-structure referenceNodeId in by-candidate mode, preserves hasSharedDesigned, and aligns external crystal ref', () => {
    const seq1 = 'ACDEFGHIKL';
    const seq2 = 'MNPQRSTVWY';
    const crystalRef = makeStructArtifact('crystal_ref', seq1, { shiftX: 50, shiftY: 30, shiftZ: 20 });
    const designed = makeStructArtifact('designed_1', seq1, { shiftX: 0, shiftY: 0, shiftZ: 0 });
    const foldA1 = makeStructArtifact('fold_a1', seq1, { shiftX: 0.2, plddt: 92 });
    const foldA2 = makeStructArtifact('fold_a2', seq2, { shiftX: 0.3, plddt: 88 });
    const foldB1 = makeStructArtifact('fold_b1', seq1, { shiftY: 0.2, plddt: 90 });
    const foldB2 = makeStructArtifact('fold_b2', seq2, { shiftY: 0.3, plddt: 86 });

    const geom10 = {
      length: 10,
      rg: 12,
      meanCaDist: 3.8,
      chainBreaks: 0,
      clashes: 0,
      helixPct: 20,
      sheetPct: 10,
      coilPct: 70,
    };
    const metricsA: DesignabilityMetrics = {
      kind: 'DesignabilityMetrics',
      id: 'm_a',
      rows: [
        {
          sequenceId: 'cand-1',
          candidateKey: 'cand-1',
          sequence: seq1,
          scRmsd: 0.4,
          scTm: 0.96,
          meanPlddt: 92,
          plddt: new Float32Array(10).fill(92),
          ptm: 0.92,
          designable: true,
          geometry: geom10,
          foldElapsedMs: 10,
          alignedFoldPdb: foldA1.pdbText,
          designedPdb: designed.pdbText,
          structureId: designed.id,
        },
        {
          sequenceId: 'cand-2',
          candidateKey: 'cand-2',
          sequence: seq2,
          scRmsd: 0.6,
          scTm: 0.94,
          meanPlddt: 88,
          plddt: new Float32Array(10).fill(88),
          ptm: 0.88,
          designable: true,
          geometry: geom10,
          foldElapsedMs: 10,
          alignedFoldPdb: foldA2.pdbText,
          designedPdb: designed.pdbText,
          structureId: designed.id,
        },
      ],
      best: null,
      passRate: 1,
    };
    metricsA.best = metricsA.rows[0];

    const metricsB: DesignabilityMetrics = {
      kind: 'DesignabilityMetrics',
      id: 'm_b',
      rows: [
        {
          sequenceId: 'cand-1',
          candidateKey: 'cand-1',
          sequence: seq1,
          scRmsd: 0.5,
          scTm: 0.95,
          meanPlddt: 90,
          plddt: new Float32Array(10).fill(90),
          ptm: 0.90,
          designable: true,
          geometry: geom10,
          foldElapsedMs: 10,
          alignedFoldPdb: foldB1.pdbText,
          designedPdb: designed.pdbText,
          structureId: designed.id,
        },
        {
          sequenceId: 'cand-2',
          candidateKey: 'cand-2',
          sequence: seq2,
          scRmsd: 0.7,
          scTm: 0.92,
          meanPlddt: 86,
          plddt: new Float32Array(10).fill(86),
          ptm: 0.86,
          designable: true,
          geometry: geom10,
          foldElapsedMs: 10,
          alignedFoldPdb: foldB2.pdbText,
          designedPdb: designed.pdbText,
          structureId: designed.id,
        },
      ],
      best: null,
      passRate: 1,
    };
    metricsB.best = metricsB.rows[0];

    const graph: WorkflowGraph = {
      id: 'cmp-ref-test',
      name: 'Compare Ref',
      nodes: [
        { id: 'pdb_ref', adapterId: 'pdb-input', label: 'Crystal Ref', params: {} },
        { id: 'score_a', adapterId: 'designability-scorer', label: 'Scorer A', params: {} },
        { id: 'score_b', adapterId: 'designability-scorer', label: 'Scorer B', params: {} },
        { id: 'cmp_1', adapterId: 'structure-compare', label: '3D Compare', params: {} },
      ],
      edges: [
        { fromNode: 'pdb_ref', fromPort: 'structure', toNode: 'cmp_1', toPort: 'reference' },
        { fromNode: 'score_a', fromPort: 'metrics', toNode: 'cmp_1', toPort: 'structures' },
        { fromNode: 'score_b', fromPort: 'metrics', toNode: 'cmp_1', toPort: 'structures' },
      ],
    };

    const outputs: Record<string, Record<string, CanonicalArtifact>> = {
      pdb_ref: { structure: crystalRef },
      score_a: { metrics: metricsA },
      score_b: { metrics: metricsB },
    };

    const cmp = buildComparisonState(graph, ['pdb_ref', 'score_a', 'score_b'], outputs, {
      layout: 'overlay',
      colorMode: 'by-model',
      alignKabsch: true,
      referenceNodeId: 'pdb_ref',
    });

    expect(cmp.pairingMode).toBe('by-candidate');
    expect(cmp.selectedCandidateKey).toBe('cand-1');
    const refBranch = cmp.branches.find((b) => b.nodeId === 'pdb_ref')!;
    expect(refBranch.missingSelectedCandidate).toBe(false);
    expect(refBranch.has3D).toBe(true);
    expect(cmp.sharedReferenceLabel).toBe('Designed Backbone');
    expect(cmp.models.some((m) => m.id === 'ref-designed')).toBe(true);
    expect(cmp.models.some((m) => m.id === 'branch-pdb_ref')).toBe(true);
    expect(cmp.models.some((m) => m.id === 'branch-score_a')).toBe(true);
    expect(cmp.models.some((m) => m.id === 'branch-score_b')).toBe(true);
    // External crystal ref was Kabsch-aligned from x~50 onto the shared frame near x~0..34
    const refModel = cmp.models.find((m) => m.id === 'branch-pdb_ref')!;
    const alignedRef = parsePdbBackbone(refModel.pdbText);
    expect(alignedRef.caCoords[0]).toBeLessThan(10);

    // Switching to cand-2 keeps pdb_ref pinned & present
    const cmp2 = buildComparisonState(graph, ['pdb_ref', 'score_a', 'score_b'], outputs, {
      layout: 'overlay',
      colorMode: 'by-model',
      alignKabsch: true,
      referenceNodeId: 'pdb_ref',
      selectedCandidateKey: 'cand-2',
    });
    expect(cmp2.branches.find((b) => b.nodeId === 'pdb_ref')?.missingSelectedCandidate).toBe(false);
    expect(cmp2.models.some((m) => m.id === 'branch-pdb_ref')).toBe(true);
  });

  it('does NOT pin multi-candidate referenceNodeId', () => {
    const seq1 = 'ACDEFGHIKL';
    const seq2 = 'MNPQRSTVWY';
    const f1 = makeStructArtifact('f1', seq1);
    const f2 = makeStructArtifact('f2', seq2);
    const multiFold: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'multi_ref',
      parentStructureId: 's_ref',
      folds: [
        { sequenceId: 'cand-1', key: 'cand-1', foldModelId: 'af2', sequence: seq1, pdbText: f1.pdbText, plddt: new Float32Array(10).fill(90), meanPlddt: 90, ptm: 0.9, caCoords: f1.caCoords, elapsedMs: 10 },
        { sequenceId: 'cand-2', key: 'cand-2', foldModelId: 'af2', sequence: seq2, pdbText: f2.pdbText, plddt: new Float32Array(10).fill(85), meanPlddt: 85, ptm: 0.85, caCoords: f2.caCoords, elapsedMs: 10 },
      ],
    };
    const otherFold: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'other',
      parentStructureId: 's_ref',
      folds: [
        { sequenceId: 'cand-1', key: 'cand-1', foldModelId: 'af2', sequence: seq1, pdbText: f1.pdbText, plddt: new Float32Array(10).fill(88), meanPlddt: 88, ptm: 0.88, caCoords: f1.caCoords, elapsedMs: 10 },
      ],
    };
    const graph: WorkflowGraph = {
      id: 'multi-ref',
      name: 'Multi Ref',
      nodes: [
        { id: 'ref_fold', adapterId: 'localfold-af2', label: 'Ref Fold', params: {} },
        { id: 'b_fold', adapterId: 'localfold-af2', label: 'B Fold', params: {} },
      ],
      edges: [],
    };
    const cmp = buildComparisonState(
      graph,
      ['ref_fold', 'b_fold'],
      { ref_fold: { fold: multiFold }, b_fold: { fold: otherFold } },
      { layout: 'overlay', colorMode: 'by-model', referenceNodeId: 'ref_fold', selectedCandidateKey: 'cand-2' },
    );
    expect(cmp.branches.find((b) => b.nodeId === 'ref_fold')?.candidateKey).toBe('cand-2');
    expect(cmp.branches.find((b) => b.nodeId === 'b_fold')?.missingSelectedCandidate).toBe(true);
  });

  it('joinKeyedCollections matches sequence fallback with differing keys without stealing exact-key peers', () => {
    const seq1 = 'ACDEFGHIKL';
    const seq2 = 'MNPQRSTVWY';
    const joined = joinKeyedCollections(
      [
        { branchId: 'a', items: [{ key: 'seq-1', sequence: seq1, value: 1 }] },
        {
          branchId: 'b',
          items: [
            { key: 'seq-1', sequence: seq1, value: 2 },
            { key: 'cand-2', sequence: seq2, value: 20 },
          ],
        },
        {
          branchId: 'c',
          items: [
            { key: 'design-1', sequence: seq1, value: 3 },
            { key: 'cand-2', sequence: seq2, value: 30 },
          ],
        },
      ],
      { matchBySequenceFallback: true },
    );
    const row1 = joined.rows.find((r) => r.key === 'seq-1')!;
    expect(row1.presentCount).toBe(3);
    expect(row1.byBranch.c.item?.key).toBe('design-1');

    // Exact-key peer 'cand-2' shared by b & c is NOT stolen by 'cand-1' (a & b) even when same sequence!
    const joinedNoSteal = joinKeyedCollections(
      [
        { branchId: 'a', items: [{ key: 'cand-1', sequence: seq2, value: 1 }] },
        {
          branchId: 'b',
          items: [
            { key: 'cand-1', sequence: seq2, value: 2 },
            { key: 'cand-2', sequence: seq2, value: 20 },
          ],
        },
        { branchId: 'c', items: [{ key: 'cand-2', sequence: seq2, value: 3 }] },
      ],
      { matchBySequenceFallback: true },
    );
    expect(joinedNoSteal.rows.find((r) => r.key === 'cand-1')?.presentCount).toBe(2);
    expect(joinedNoSteal.rows.find((r) => r.key === 'cand-2')?.presentCount).toBe(2);
  });
});

describe('M3 §2.5b, §2.5c, §2.5e & Svelte UI wiring', () => {
  it('§2.5c: isCompareDisplayOnlyParamChange distinguishes display-only edits on structure-compare', () => {
    expect(
      isCompareDisplayOnlyParamChange(
        { adapterId: 'structure-compare', params: { layout: 'overlay', colorMode: 'by-model', alignKabsch: true } },
        { layout: 'side-by-side', colorMode: 'by-model', alignKabsch: true },
      ),
    ).toBe(true);
    expect(
      isCompareDisplayOnlyParamChange(
        { adapterId: 'structure-compare', params: { layout: 'overlay', colorMode: 'by-model', alignKabsch: true } },
        { layout: 'overlay', colorMode: 'plddt', alignKabsch: false },
      ),
    ).toBe(true);
    expect(
      isCompareDisplayOnlyParamChange(
        { adapterId: 'structure-compare', params: { layout: 'overlay', colorMode: 'by-model', alignKabsch: true } },
        { layout: 'overlay', colorMode: 'by-model', alignKabsch: true },
      ),
    ).toBe(false);
    expect(
      isCompareDisplayOnlyParamChange(
        { adapterId: 'genie3', params: { length: 80 } },
        { layout: 'side-by-side' },
      ),
    ).toBe(false);
  });

  it('§2.5e: resolveResidueFocusToken resolves non-1-based PDBs, multi-chain, insertion codes & multi-char mmCIF', () => {
    const struct5o45 = {
      kind: 'StructureArtifact',
      id: '5o45',
      label: '5O45',
      sourceModelId: 'pdb-input',
      length: 3,
      sequence: 'AGS',
      chainIds: ['A', 'A', 'B'],
      caCoords: new Float32Array(9),
      backbone4Coords: new Float32Array(36),
      pdbText: 'END\n',
      geometry: { length: 3, rg: 5, meanCaDist: 3.8, chainBreaks: 0, clashes: 0, helixPct: 0, sheetPct: 0, coilPct: 100 },
      sourceResidues: [
        { index: 0, authChainId: 'A', resNum: 17, insCode: '', resName: 'ALA', projectedChainId: 'A' },
        { index: 1, authChainId: 'A', resNum: 18, insCode: '', resName: 'GLY', projectedChainId: 'A' },
        { index: 2, authChainId: 'B', resNum: 19, insCode: 'A', resName: 'SER', projectedChainId: 'B' },
      ],
    } satisfies StructureArtifact;

    const esmcMat: MutationMatrixArtifact = {
      kind: 'MutationMatrix',
      id: 'esmc_1',
      sourceModelId: 'esmc',
      metric: 'llr-fitness',
      direction: 'higher-is-better',
      residues: 'AGS',
      residueIds: ['A1', 'A2', 'A3'],
      alphabet: 'ACDEFGHIKLMNPQRSTVWY',
      values: new Float32Array(60),
    };
    expect(resolveResidueFocusToken(0, 'A1', esmcMat, struct5o45)).toBe('A17');
    expect(resolveResidueFocusToken(2, 'A3', esmcMat, struct5o45)).toBe('B19A');

    const thermoMat: MutationMatrixArtifact = {
      ...esmcMat,
      sourceModelId: 'thermompnn',
      metric: 'ddg-kcal-mol',
      direction: 'lower-is-better',
      residueIds: ['A17', 'A18', 'B19A'],
    };
    expect(resolveResidueFocusToken(2, 'B19A', thermoMat, struct5o45)).toBe('B19A');

    const multiCharStruct: StructureArtifact = {
      ...struct5o45,
      cifText: 'data_multi\n#\n',
      sourceResidues: [
        { index: 0, authChainId: 'AA', resNum: 42, insCode: '', resName: 'ALA', projectedChainId: 'A' },
        { index: 1, authChainId: 'a', resNum: 7, insCode: '', resName: 'GLY', projectedChainId: 'a' },
      ],
    };
    const trackMulti: ResidueTrackArtifact = {
      kind: 'ResidueTrack',
      id: 'dyna_1',
      sourceModelId: 'dyna1',
      quantity: 'exchange-prob',
      residues: 'AG',
      residueIds: ['AA42', 'a7'],
      values: new Float32Array([0.5, 0.8]),
    };
    expect(resolveResidueFocusToken(0, 'AA42', trackMulti, multiCharStruct)).toBe('AA:42');
    expect(resolveResidueFocusToken(1, 'a7', trackMulti, multiCharStruct)).toBe('a:7');
    // PDB-only structure uses projectedChainId
    const pdbOnlyMulti: StructureArtifact = { ...multiCharStruct, cifText: undefined };
    expect(resolveResidueFocusToken(0, 'AA42', trackMulti, pdbOnlyMulti)).toBe('A42');
  });

  it('+page.svelte and ArtifactPeekPanel.svelte wire §2.5b, §2.5c, §2.5e, and §2.2 cleanly', () => {
    const pageSrc = readWebFile('src/routes/+page.svelte');
    expect(pageSrc).toContain('Compare (${effectiveCompareNodeIds.length})');
    expect(pageSrc).not.toContain('Compare (${selectedNodeIds.length})');
    expect(pageSrc).toContain('isCompareDisplayOnlyParamChange');
    expect(pageSrc).toContain('checked={effectiveCompareAlignKabsch}');
    expect(pageSrc).toContain('disabled={kabschDisabled}');
    expect(pageSrc).toContain('referenceNodeId: activeCompareRefNodeId ?? undefined');
    expect(pageSrc).toContain('for (const id of effectiveCompareNodeIds)');
    expect(pageSrc).toContain('focusToken={studioOpen ? studioFocusToken : viewerFocusToken}');
    expect(pageSrc).toContain('onSelectResidue={handleSelectPerResidue}');

    const peekSrc = readWebFile('src/lib/ui/ArtifactPeekPanel.svelte');
    expect(peekSrc).toContain('<MutationHeatmap matrix={mutArt} {onDownloadText} {onSelectResidue}');
    expect(peekSrc).toContain('<ResidueTrackPlot track={trackArt} {onDownloadText} {onSelectResidue} />');

    const viewerSrc = readWebFile('src/lib/viewer/MolSuperposeViewer.svelte');
    expect(viewerSrc).toContain('^([A-Za-z0-9_]+):(-?\\d+)([A-Za-z]?)$');
  });
});

describe('M3 §2.5f: ResidueTrackPlot tooltip non-clipping positioning', () => {
  it('snaps left near left edge, right near right edge, and centers inside middle zone', () => {
    const left = computeTrackTooltipPosition({
      svgX: 44,
      viewBoxWidth: 640,
      containerWidthPx: 360,
      textLength: 22,
    });
    expect(left.align).toBe('left');
    expect(left.left).toBe('6px');
    expect(left.transform).toBe('translateX(0)');

    const right = computeTrackTooltipPosition({
      svgX: 626,
      viewBoxWidth: 640,
      containerWidthPx: 360,
      textLength: 22,
    });
    expect(right.align).toBe('right');
    expect(right.right).toBe('6px');
    expect(right.left).toBe('auto');
    expect(right.transform).toBe('translateX(0)');

    const mid = computeTrackTooltipPosition({
      svgX: 320,
      viewBoxWidth: 640,
      containerWidthPx: 800,
      textLength: 22,
    });
    expect(mid.align).toBe('center');
    expect(mid.transform).toBe('translateX(-50%)');

    const trackSrc = readWebFile('src/lib/viewer/ResidueTrackPlot.svelte');
    expect(trackSrc).toContain('computeTrackTooltipPosition');
  });
});

describe('M3 §2.5d, §2.5g, §2.5h & Bio P0-4: Badges, Catalog Search & Upstream Locks', () => {
  it('§2.5d: formatMutationMatrixBadge skips WT 0.00 and handles NaN/Infinity/-0.00/targetedMutations', () => {
    const alphabet = 'ACDEFGHIKLMNPQRSTVWY';
    // 2 positions: pos 0 = 'A', pos 1 = 'G'
    const values = new Float32Array(40).fill(-2.5);
    values[0] = 0.0; // WT A->A
    values[1] = -0.35; // best non-WT mutant (A->C)
    values[20 + 5] = 0.0; // WT G->G
    values[20 + 6] = Number.NaN;
    values[20 + 7] = Number.POSITIVE_INFINITY;

    const llrNeg: MutationMatrixArtifact = {
      kind: 'MutationMatrix',
      id: 'esmc_neg',
      sourceModelId: 'esmc',
      metric: 'llr-fitness',
      direction: 'higher-is-better',
      residues: 'AG',
      residueIds: ['A1', 'A2'],
      alphabet,
      values,
    };
    expect(formatMutationMatrixBadge(llrNeg)).toBe('2 pos · max LLR -0.35');

    values[2] = 1.25;
    expect(formatMutationMatrixBadge(llrNeg)).toBe('2 pos · max LLR 1.25');

    // ddg-kcal-mol all destabilizing (>0), WT = 0.00
    const ddgVals = new Float32Array(40).fill(1.2);
    ddgVals[0] = 0.0; // WT A->A
    ddgVals[20 + 5] = 0.0; // WT G->G
    ddgVals[3] = 0.18; // least destabilizing non-WT
    const ddgDestab: MutationMatrixArtifact = {
      kind: 'MutationMatrix',
      id: 'thermo_destab',
      sourceModelId: 'thermompnn',
      metric: 'ddg-kcal-mol',
      direction: 'lower-is-better',
      residues: 'AG',
      residueIds: ['A1', 'A2'],
      alphabet,
      values: ddgVals,
    };
    expect(formatMutationMatrixBadge(ddgDestab)).toBe('2 pos · min ΔΔG 0.18');

    ddgVals[4] = -1.36;
    expect(formatMutationMatrixBadge(ddgDestab)).toBe('2 pos · min ΔΔG -1.36');

    // -0.0001 formats as 0.00 without negative sign
    const negZeroVals = new Float32Array(20).fill(-0.0001);
    negZeroVals[0] = 0.0;
    expect(
      formatMutationMatrixBadge({
        ...llrNeg,
        residues: 'A',
        residueIds: ['A1'],
        values: negZeroVals,
      }),
    ).toBe('1 pos · max LLR 0.00');

    // Empty values fallback to N×20
    expect(
      formatMutationMatrixBadge({
        ...llrNeg,
        values: new Float32Array([0.0, Number.NaN]),
        residues: 'A',
        residueIds: ['A1'],
      }),
    ).toBe('1×20 LLR');

    // targetedMutations fallback
    expect(
      formatMutationMatrixBadge({
        ...ddgDestab,
        values: new Float32Array(0),
        targetedMutations: [
          { mutation: 'A1A', residueId: 'A1', wildType: 'A', mutant: 'A', score: 0 },
          { mutation: 'A1V', residueId: 'A1', wildType: 'A', mutant: 'V', score: -0.92 },
        ],
      }),
    ).toBe('2 pos · min ΔΔG -0.92');
  });

  it('§2.5g: filterCatalogCategories and filterCatalog rank exact/relevant hits first', () => {
    const genie2Res = filterCatalogCategories('Genie 2');
    expect(genie2Res[0].category.id).toBe('generation');
    expect(genie2Res[0].matchingSubItems[0].key).toBe('genie2');

    const genieRes = filterCatalogCategories('genie');
    expect(genieRes[0].matchingSubItems.map((s) => s.key)).toEqual(['genie3', 'genie2']);

    const rfd4Res = filterCatalogCategories('rfd4');
    expect(rfd4Res[0].category.id).toBe('generation');
    expect(rfd4Res[0].matchingSubItems[0].key).toBe('rfd4-proteina');

    const boltzRes = filterCatalogCategories('boltz');
    expect(boltzRes.map((r) => r.category.id)).toEqual(['prediction', 'generation']);
    expect(boltzRes[0].matchingSubItems[0].key).toBe('boltz2');
    expect(boltzRes[1].matchingSubItems[0].key).toBe('boltzgen');

    const topKRes = filterCatalogCategories('top-k');
    expect(topKRes).toHaveLength(1);
    expect(topKRes[0].matchingSubItems.map((s) => s.key)).toEqual(['top-k-filter']);
    expect(topKRes[0].hasNarrowedSubMatch).toBe(true);

    const flatRfd4 = filterCatalog('rfd4');
    expect(flatRfd4[0].item.id).toBe('rfd4-proteina');
  });

  it('§2.5h: formatNodeParamSummary returns non-empty summaries for all 25 active registry adapters', () => {
    const activeIds = Object.keys(ADAPTER_REGISTRY);
    expect(activeIds.length).toBeGreaterThanOrEqual(25);
    for (const id of activeIds) {
      const summary = formatNodeParamSummary(id, defaultParamsForAdapter(id));
      expect(summary, `Missing param summary for ${id}`).toBeTruthy();
    }
    expect(formatNodeParamSummary('la-proteina', { length: 80, steps: 400, mode: 'sde' })).toContain('80 aa');
    expect(formatNodeParamSummary('thermompnn', { chain: 'A', bfactorAggregate: 'mean-ddg' })).toContain('Chain A');
    expect(formatNodeParamSummary('structure-compare', { layout: 'side-by-side', alignKabsch: false })).toBe('Side-by-side');
  });

  it('Bio P0-4: computeUpstreamLockSummary surfaces pre-run and post-run inherited locks and 0-designable warning', () => {
    const genieBinderGraph: WorkflowGraph = {
      id: 'g-binder',
      name: 'Genie Binder',
      nodes: [
        { id: 'pdb_1', adapterId: 'pdb-input', label: 'PDB', params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45_binder' } },
        { id: 'genie_1', adapterId: 'genie3', label: 'Genie 3', params: { taskMode: 'binder', length: 70 } },
        { id: 'mpnn_1', adapterId: 'web-protein-mpnn', label: 'SolubleMPNN', params: { fixedPositions: '' } },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'genie_1', toPort: 'template' },
        { fromNode: 'genie_1', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' },
      ],
    };

    const gSummary = computeUpstreamLockSummary(genieBinderGraph, 'mpnn_1', {});
    expect(gSummary).not.toBeNull();
    expect(gSummary!.lockedChains).toEqual(['B']);
    expect(gSummary!.designingChains).toEqual(['A']);
    expect(gSummary!.locksAllDesignChainsWarning).toBeNull();
    expect(gSummary!.nodeBadge).toContain('Lock B');

    // Footgun: user types "A" into fixedPositions on Genie 3 binder (which locks B) -> 0 designable residues!
    const footgunGraph: WorkflowGraph = {
      ...genieBinderGraph,
      nodes: genieBinderGraph.nodes.map((n) =>
        n.id === 'mpnn_1' ? { ...n, params: { fixedPositions: 'A' } } : n,
      ),
    };
    const warnSummary = computeUpstreamLockSummary(footgunGraph, 'mpnn_1', {});
    expect(warnSummary!.status).toBe('warning');
    expect(warnSummary!.locksAllDesignChainsWarning).toContain('0 designable residues');
    expect(warnSummary!.nodeBadge).toContain('0 designable aa');

    // RFD4 binder convention locks Chain A (target), designs Chain B (binder)
    const rfd4Graph: WorkflowGraph = {
      id: 'rfd4-binder',
      name: 'RFD4 Binder',
      nodes: [
        { id: 'rfd4_1', adapterId: 'rfd4-proteina', label: 'RFD4', params: { mode: 'binder', length: 70 } },
        { id: 'mpnn_1', adapterId: 'web-protein-mpnn', label: 'SolubleMPNN', params: {} },
      ],
      edges: [{ fromNode: 'rfd4_1', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' }],
    };
    const rSummary = computeUpstreamLockSummary(rfd4Graph, 'mpnn_1', {});
    expect(rSummary!.lockedChains).toEqual(['A']);
    expect(rSummary!.designingChains).toEqual(['B']);

    // Motif mode -> hasFixedMotif, empty lockedChains
    const motifGraph: WorkflowGraph = {
      id: 'motif',
      name: 'Motif',
      nodes: [
        { id: 'genie_1', adapterId: 'genie3', label: 'Genie 3', params: { taskMode: 'motif', motifSpan: 'A10-25' } },
        { id: 'lig_1', adapterId: 'ligandmpnn', label: 'LigandMPNN', params: {} },
      ],
      edges: [{ fromNode: 'genie_1', fromPort: 'structure', toNode: 'lig_1', toPort: 'structure' }],
    };
    const mSummary = computeUpstreamLockSummary(motifGraph, 'lig_1', {});
    expect(mSummary!.hasFixedMotif).toBe(true);
    expect(mSummary!.lockedChains).toEqual([]);

    // unlockInherited clears warning & sets unlocked status
    const unlockedGraph: WorkflowGraph = {
      ...footgunGraph,
      nodes: footgunGraph.nodes.map((n) =>
        n.id === 'mpnn_1' ? { ...n, params: { fixedPositions: 'A', unlockInherited: true } } : n,
      ),
    };
    const uSummary = computeUpstreamLockSummary(unlockedGraph, 'mpnn_1', {});
    expect(uSummary!.status).toBe('unlocked');
    expect(uSummary!.locksAllDesignChainsWarning).toBeNull();
  });
});

describe('M3 Iteration 2 regressions: llr-fitness viewer/legend, tooltip narrow overflow, mmCIF semicolon/chain/element, Shift-select SequenceSet, param NaN guard', () => {
  it('wires llr-fitness into useSemanticUncertainty in both renderScene and renderMultiModels and both legend blocks', () => {
    const viewerSrc = readWebFile('src/lib/viewer/MolSuperposeViewer.svelte');
    const useSemanticMatches = viewerSrc.match(
      /const useSemanticUncertainty =[\s\S]*?styleOpts\.bfactorSemantics === 'ddg-stability' \|\|\s*\(styleOpts\.bfactorSemantics as string\) === 'llr-fitness';/g,
    );
    expect(useSemanticMatches).toHaveLength(2);

    const llrLegendMatches = viewerSrc.match(
      /\{:else if \(bfactorSemantics as string\) === 'llr-fitness'\}[\s\S]*?Deleterious[\s\S]*?Beneficial \(LLR\)/g,
    );
    expect(llrLegendMatches).toHaveLength(2);
    expect(viewerSrc).toContain('LLR fitness scale: Deleterious to Beneficial');
  });

  it('computeTrackTooltipPosition never overflows [0, containerWidthPx] in narrow containers (260-360px) across all svgX', () => {
    for (const boxW of [260, 320, 360]) {
      for (const simulatedWidth of [boxW - 12, undefined]) {
        for (let svgX = 0; svgX <= 640; svgX += 5) {
          const pos = computeTrackTooltipPosition({
            svgX,
            viewBoxWidth: 640,
            containerWidthPx: boxW,
            tooltipWidthPx: simulatedWidth,
            textLength: 50,
          });
          const actualW = Math.min(boxW - 12, simulatedWidth ?? 50 * 6.8 + 18);
          let leftPx: number;
          if (pos.align === 'left') {
            leftPx = parseFloat(pos.left);
          } else if (pos.align === 'right') {
            leftPx = boxW - parseFloat(pos.right) - actualW;
          } else {
            const pct = parseFloat(pos.left);
            leftPx = (pct / 100) * boxW - actualW / 2;
          }
          const rightPx = leftPx + actualW;
          expect(leftPx, `left clip at boxW=${boxW}, svgX=${svgX}`).toBeGreaterThanOrEqual(0);
          expect(rightPx, `right clip at boxW=${boxW}, svgX=${svgX}`).toBeLessThanOrEqual(boxW);
        }
      }
    }
  });

  it('parseMmcifBackbone ignores _atom_site.* inside multiline semicolon blocks and keeps chainIdMap consistent with nucleic-before-protein mmCIFs', () => {
    const semicolonCif = [
      'data_semicolon_guard',
      '_struct_ref.pdbx_seq_one_letter_code',
      ';MKTAYIAKQRQISFVKSHFSRQ',
      '_atom_site.fake_column',
      'ATOM 999 FAKE FAKE Z 999 0.0 0.0 0.0',
      ';',
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
      'ATOM 1 N ALA A 1 0.000 0.000 0.000',
      'ATOM 2 CA ALA A 1 1.458 0.000 0.000',
      'ATOM 3 C ALA A 1 2.009 1.420 0.000',
      '#',
    ].join('\n');

    const parsedSemi = parseStructureText(semicolonCif);
    expect(parsedSemi.nResidues).toBe(1);
    expect(parsedSemi.sequence).toBe('A');
    expect(parsedSemi.chainIds).toEqual(['A']);

    // Multi-char nucleic chain DD before multi-char protein chain AA
    const dnaBeforeProteinCif = [
      'data_dna_before_prot',
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
      'ATOM 1 P DA DD 1 0.000 0.000 0.000',
      'ATOM 2 N ALA AA 1 10.000 0.000 0.000',
      'ATOM 3 CA ALA AA 1 11.458 0.000 0.000',
      'ATOM 4 C ALA AA 1 12.009 1.420 0.000',
      '#',
    ].join('\n');
    const parsedDnaFirst = parseStructureText(dnaBeforeProteinCif);
    expect(parsedDnaFirst.projectedChainIds).toEqual(['A']);
    expect(parsedDnaFirst.chainIdMap.A).toBe('AA');
    expect(parsedDnaFirst.chainIdMap.B).toBe('DD');
    const reparsedDnaFirst = parsePdbBackbone(parsedDnaFirst.pdbText);
    expect(reparsedDnaFirst.chainIds).toEqual(['A']);
    expect(parsedDnaFirst.chainIdMap[reparsedDnaFirst.chainIds[0]]).toBe('AA');

    // Single-char nucleic chain A before multi-char protein chain AA must not collide on 'A'
    const singleDnaBeforeMultiProt = dnaBeforeProteinCif.replace('DA DD 1', 'DA A 1');
    const parsedCollisionSafe = parseStructureText(singleDnaBeforeMultiProt);
    expect(parsedCollisionSafe.projectedChainIds).toEqual(['B']);
    expect(parsedCollisionSafe.chainIdMap).toEqual({ A: 'A', B: 'AA' });
    const reparsedCollisionSafe = parsePdbBackbone(parsedCollisionSafe.pdbText);
    expect(reparsedCollisionSafe.chainIds).toEqual(['B']);
    expect(parsedCollisionSafe.chainIdMap[reparsedCollisionSafe.chainIds[0]]).toBe('AA');
  });

  it('mmcifToPdb infers element C for HETATM MSE/PTR CA without type_symbol while keeping Calcium CA', () => {
    const cifNoTypeSym = [
      'data_het_ca',
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
      'HETATM 1 N MSE A 1 0.000 0.000 0.000',
      'HETATM 2 CA MSE A 1 1.458 0.000 0.000',
      'HETATM 3 SE MSE A 1 3.000 1.000 1.000',
      'HETATM 4 CA PTR A 2 5.000 0.000 0.000',
      'HETATM 5 CA CA B 101 15.000 0.000 0.000',
      '#',
    ].join('\n');

    const pdb = mmcifToPdb(cifNoTypeSym);
    const lines = pdb.split('\n').filter((l) => l.startsWith('HETATM'));
    expect(lines).toHaveLength(5);
    // MSE CA -> atom cols 13-16 ' CA ', element cols 77-78 ' C'
    expect(lines[1].slice(12, 16)).toBe(' CA ');
    expect(lines[1].slice(76, 78)).toBe(' C');
    // MSE SE -> element 'SE'
    expect(lines[2].slice(76, 78)).toBe('SE');
    // PTR CA -> ' CA ', element ' C'
    expect(lines[3].slice(12, 16)).toBe(' CA ');
    expect(lines[3].slice(76, 78)).toBe(' C');
    // Genuine Calcium HETATM CA -> 'CA  ', element 'CA'
    expect(lines[4].slice(12, 16)).toBe('CA  ');
    expect(lines[4].slice(76, 78)).toBe('CA');
  });

  it('auto-pins single-structure branch carrying SequenceSet when Shift-selected without referenceNodeId', () => {
    const seq1 = 'ACDEFGHIKL';
    const seq2 = 'MNPQRSTVWY';
    const crystalRef = makeStructArtifact('crystal_ref', seq1, { shiftX: 50, shiftY: 30, shiftZ: 20 });
    const refSeq: SequenceSet = {
      kind: 'SequenceSet',
      id: 'crystal_seq',
      parentStructureId: crystalRef.id,
      sourceModelId: 'pdb-input',
      candidates: [{ id: 'crystal-1', sequence: seq1, score: 0, globalScore: 0, sourceModelId: 'pdb-input' }],
    };
    const designed = makeStructArtifact('designed_1', seq1);
    const foldA1 = makeStructArtifact('fold_a1', seq1, { shiftX: 0.2, plddt: 92 });
    const foldA2 = makeStructArtifact('fold_a2', seq2, { shiftX: 0.3, plddt: 88 });
    const geom10 = {
      length: 10,
      rg: 12,
      meanCaDist: 3.8,
      chainBreaks: 0,
      clashes: 0,
      helixPct: 20,
      sheetPct: 10,
      coilPct: 70,
    };
    const makeMetrics = (id: string): DesignabilityMetrics => ({
      kind: 'DesignabilityMetrics',
      id,
      rows: [
        {
          sequenceId: 'cand-1',
          candidateKey: 'cand-1',
          sequence: seq1,
          scRmsd: 0.4,
          scTm: 0.96,
          meanPlddt: 92,
          plddt: new Float32Array(10).fill(92),
          ptm: 0.92,
          designable: true,
          geometry: geom10,
          foldElapsedMs: 10,
          alignedFoldPdb: foldA1.pdbText,
          designedPdb: designed.pdbText,
          structureId: designed.id,
        },
        {
          sequenceId: 'cand-2',
          candidateKey: 'cand-2',
          sequence: seq2,
          scRmsd: 0.6,
          scTm: 0.94,
          meanPlddt: 88,
          plddt: new Float32Array(10).fill(88),
          ptm: 0.88,
          designable: true,
          geometry: geom10,
          foldElapsedMs: 10,
          alignedFoldPdb: foldA2.pdbText,
          designedPdb: designed.pdbText,
          structureId: designed.id,
        },
      ],
      best: null,
      passRate: 1,
    });
    const metricsA = makeMetrics('m_a');
    metricsA.best = metricsA.rows[0];
    const metricsB = makeMetrics('m_b');
    metricsB.best = metricsB.rows[0];

    const graph: WorkflowGraph = {
      id: 'shift-select-seqset',
      name: 'Shift Select',
      nodes: [
        { id: 'pdb_ref', adapterId: 'pdb-input', label: 'Crystal Ref', params: {} },
        { id: 'score_a', adapterId: 'designability-scorer', label: 'Scorer A', params: {} },
        { id: 'score_b', adapterId: 'designability-scorer', label: 'Scorer B', params: {} },
      ],
      edges: [],
    };
    const cmp = buildComparisonState(
      graph,
      ['pdb_ref', 'score_a', 'score_b'],
      {
        pdb_ref: { structure: crystalRef, sequences: refSeq },
        score_a: { metrics: metricsA },
        score_b: { metrics: metricsB },
      },
      {
        layout: 'overlay',
        colorMode: 'by-model',
        alignKabsch: true,
        selectedCandidateKey: 'cand-2',
      },
    );
    const refBranch = cmp.branches.find((b) => b.nodeId === 'pdb_ref')!;
    expect(refBranch.missingSelectedCandidate).toBe(false);
    expect(refBranch.has3D).toBe(true);
    expect(cmp.models.some((m) => m.id === 'branch-pdb_ref')).toBe(true);
    expect(cmp.models.some((m) => m.id === 'ref-designed')).toBe(true);
  });

  it('formatNodeParamSummary guards non-numeric strings and non-finite inputs without emitting NaN', () => {
    const cases: Array<[string, Record<string, unknown>]> = [
      ['genie3', { length: 'abc', nSampleStep: '' }],
      ['rfd4-proteina', { mode: 'unconditional', length: Number.NaN, nsteps: 'bad' }],
      ['rfd4-proteina', { mode: 'partial', partialFraction: 'x', nsteps: Infinity }],
      ['la-proteina', { length: 'bad', steps: null, numDesigns: 'two' }],
      ['web-protein-mpnn', { numSequences: 'foo', temperature: 'bar' }],
      ['localfold-af2', { model: 'monomer', recycles: 'bad', diffusionSteps: '' }],
      ['top-k-filter', { topK: 'bad', maxTotal: Number.NaN }],
      ['designability-scorer', { scRmsdThreshold: 'bad', plddtThreshold: Infinity }],
    ];
    for (const [adapterId, params] of cases) {
      const out = formatNodeParamSummary(adapterId, params);
      expect(out).not.toContain('NaN');
      expect(out).not.toContain('Infinity');
      expect(out.length).toBeGreaterThan(0);
    }
  });
});

describe('M3 Iteration 3 regressions: altloc-only HETATM ligand detection & PDB export', () => {
  it('PDB with B-only, C-only, or non-standard altloc HETATM ligand records ligandNames while excluding B-only HOH/ACE', () => {
    const pdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00           N  ',
      'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 50.00           C  ',
      'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 50.00           C  ',
      'ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 50.00           O  ',
      // B-only SAM ligand (no 'A' or ' ' conformer)
      'HETATM    5  N1 BSAM A 501       5.000   5.000   5.000  0.50 20.00           N  ',
      // C-only LIG ligand (no 'A', 'B', or ' ' conformer)
      'HETATM    6  C1 CLIG A 502       7.000   5.000   5.000  0.30 20.00           C  ',
      // Non-standard altloc '*' COF ligand (no 'A'/'B'/' ' conformer)
      'HETATM    7  C1 *COF A 503       8.000   5.000   5.000  0.30 20.00           C  ',
      // B-only water and peptide cap must still be excluded
      'HETATM    8  O  BHOH A 601      10.000  10.000  10.000  0.50 40.00           O  ',
      'HETATM    9  C  BACE A   0      -1.500   0.000   0.000  0.50 20.00           C  ',
      'END',
    ].join('\n');

    const parsed = parseStructureText(pdb);
    expect(parsed.format).toBe('pdb');
    expect(parsed.diagnostics.hasLigands).toBe(true);
    expect(parsed.diagnostics.ligandNames).toEqual(['SAM', 'LIG', 'COF']);
  });

  it('mmCIF with B-only and C-only HETATM ligands records ligandNames and preserves them in mmcifToPdb while excluding B/C-only HOH/ACE', () => {
    const cif = [
      'data_altloc_ligands',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 N N . ALA A 1 0.000 0.000 0.000',
      'ATOM 2 C CA . ALA A 1 1.458 0.000 0.000',
      'ATOM 3 C C . ALA A 1 2.009 1.420 0.000',
      'ATOM 4 O O . ALA A 1 1.251 2.390 0.000',
      // B-only SAM
      'HETATM 5 N N1 B SAM A 501 5.000 5.000 5.000',
      // C-only LIG
      'HETATM 6 C C1 C LIG A 502 7.000 5.000 5.000',
      // Non-standard altloc '*' COF
      'HETATM 7 C C1 * COF A 503 8.000 5.000 5.000',
      // B-only HOH and C-only ACE must be excluded from ligandNames
      'HETATM 8 O O B HOH A 601 10.000 10.000 10.000',
      'HETATM 9 C C C ACE A 0 -1.500 0.000 0.000',
      '#',
    ].join('\n');

    const parsed = parseStructureText(cif);
    expect(parsed.format).toBe('mmcif');
    expect(parsed.diagnostics.hasLigands).toBe(true);
    expect(parsed.diagnostics.ligandNames).toEqual(['SAM', 'LIG', 'COF']);

    const exportedPdb = mmcifToPdb(cif);
    expect(exportedPdb).toContain('HETATM');
    expect(exportedPdb).toContain(' SAM A 501');
    expect(exportedPdb).toContain(' LIG A 502');
    expect(exportedPdb).toContain(' COF A 503');
    expect(exportedPdb).not.toContain(' HOH ');

    const reparsed = parseStructureText(exportedPdb);
    expect(reparsed.diagnostics.hasLigands).toBe(true);
    expect(reparsed.diagnostics.ligandNames).toEqual(['SAM', 'LIG', 'COF']);
  });

  it('disordered ligand with A, B, and C conformers prefers A without duplicating ligandNames or exported coordinates', () => {
    const pdbDisordered = [
      'ATOM      1  CA  ALA A   1       1.458   0.000   0.000  1.00 50.00           C  ',
      // Same ligand SAM in altlocs A and B
      'HETATM    2  N1 ASAM A 501       5.000   5.000   5.000  0.60 20.00           N  ',
      'HETATM    3  N1 BSAM A 501      15.000  15.000  15.000  0.40 20.00           N  ',
      // Microheterogeneous site 502 where B appears before A and C: A (DHZ) must win
      'HETATM    4  C1 BIGN A 502      25.000  25.000  25.000  0.30 20.00           C  ',
      'HETATM    5  C1 ADHZ A 502      35.000  35.000  35.000  0.50 20.00           C  ',
      'HETATM    6  C1 COTH A 502      45.000  45.000  45.000  0.20 20.00           C  ',
      'END',
    ].join('\n');

    const parsedPdb = parseStructureText(pdbDisordered);
    expect(parsedPdb.diagnostics.hasLigands).toBe(true);
    expect(parsedPdb.diagnostics.ligandNames).toEqual(['SAM', 'DHZ']);

    const cifDisordered = [
      'data_disordered_lig',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 C CA . ALA A 1 1.458 0.000 0.000',
      'HETATM 2 N N1 B SAM A 501 15.000 15.000 15.000',
      'HETATM 3 N N1 A SAM A 501 5.000 5.000 5.000',
      'HETATM 4 N N1 C SAM A 501 99.000 99.000 99.000',
      'HETATM 5 C C1 B IGN A 502 25.000 25.000 25.000',
      'HETATM 6 C C1 A DHZ A 502 35.000 35.000 35.000',
      'HETATM 7 C C1 C OTH A 502 45.000 45.000 45.000',
      '#',
    ].join('\n');

    const parsedCif = parseStructureText(cifDisordered);
    expect(parsedCif.diagnostics.hasLigands).toBe(true);
    expect(parsedCif.diagnostics.ligandNames).toEqual(['SAM', 'DHZ']);

    const exported = mmcifToPdb(cifDisordered);
    expect(exported).toContain('5.000');
    expect(exported).not.toContain('15.000');
    expect(exported).not.toContain('99.000');
    expect(exported).toContain(' DHZ A 502');
    expect(exported).not.toContain(' IGN ');
    expect(exported).not.toContain(' OTH ');
  });
});
