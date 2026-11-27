import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  REFERENCE_BACKBONE_COLOR,
  buildComparisonState,
  resolveCompareDisplayModelId,
  type ComparisonState,
} from '$lib/engine/compare';
import {
  TRACK_NOUN,
  branchesRenderIdentically,
  deriveResidueTrackFromStructure,
  expandMergedModelIds,
  findIdenticalTrackGroups,
  granularityCompatible,
  isTrackSemantics,
  type IdentityBranch,
} from '$lib/engine/compare-identity';
import { generateSampleHelixBundlePdb, parseStructureText } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { paintStructureBfactors } from '$lib/protein-ops/bfactor-paint';
import { poolUncertaintyColorArgs, resolveUncertaintyColorArgFromValues } from '$lib/viewer/bfactor-domain';
import {
  COMPARE_DISPLAY_ONLY_PARAM_KEYS,
  isCompareDisplayOnlyParamChange,
} from '$lib/ui/page-state-helpers';
import { formatNodeParamSummary } from '$lib/ui/node-display-helpers';
import { structureComparePlugin } from '$lib/plugins/structure-compare';
import type {
  AtomGranularity,
  BfactorSemantics,
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

const BASE_PDB = generateSampleHelixBundlePdb();
const BASE = parseStructureText(BASE_PDB);
const L = BASE.nResidues;

/** Rewrite each ATOM/HETATM line via `fn(line, resSeq)`; `null` drops the line. */
function mapAtomLines(pdb: string, fn: (line: string, resSeq: number) => string | null): string {
  return pdb
    .split('\n')
    .map((line) => {
      if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) return line;
      return fn(line, Number.parseInt(line.slice(22, 26), 10));
    })
    .filter((line): line is string => line !== null)
    .join('\n');
}

function setCoords(line: string, x: number, y: number, z: number): string {
  return `${line.slice(0, 30)}${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}${line.slice(54)}`;
}

/** Non-rigid per-residue deformation (cannot be undone by Kabsch). */
function perturbPdb(pdb: string, amp: number): string {
  return mapAtomLines(pdb, (line, r) => {
    const x = Number.parseFloat(line.slice(30, 38)) + amp * Math.sin(r * 0.9);
    const y = Number.parseFloat(line.slice(38, 46)) + amp * Math.cos(r * 1.7);
    const z = Number.parseFloat(line.slice(46, 54)) + amp * Math.sin(r * 0.37 + 1);
    return setCoords(line, x, y, z);
  });
}

function caOnlyPdb(pdb: string): string {
  return mapAtomLines(pdb, (line) => (line.slice(12, 16) === ' CA ' ? line : null));
}

function renumberPdb(pdb: string, offset: number): string {
  return mapAtomLines(pdb, (line, r) => `${line.slice(0, 22)}${String(r + offset).padStart(4)}${line.slice(26)}`);
}

function withLigand(pdb: string, resName = 'ATP'): string {
  const het = [
    `HETATM 9001  PG  ${resName} B 901      12.000  10.000  10.000  1.00  0.00           P  `,
    `HETATM 9002  O1G ${resName} B 901      13.200  10.400  10.200  1.00  0.00           O  `,
    `HETATM 9003  C1  ${resName} B 901      11.000   9.200  10.600  1.00  0.00           C  `,
  ];
  const lines = pdb.split('\n');
  const endIdx = lines.findIndex((l) => l.startsWith('END'));
  if (endIdx >= 0) lines.splice(endIdx, 0, ...het);
  else lines.push(...het);
  return lines.join('\n');
}

function paint(pdb: string, fill: (i: number) => number): string {
  return paintStructureBfactors(pdb, Float32Array.from({ length: L }, (_, i) => fill(i)));
}

function makeStruct(
  id: string,
  pdb: string,
  extra: Partial<StructureArtifact> = {},
): StructureArtifact {
  const parsed = parseStructureText(pdb);
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: pdb,
    chainIds: parsed.chainIds,
    geometry: computeGeometryStats(parsed.caCoords),
    diagnostics: parsed.diagnostics,
    ...extra,
  };
}

/** A Dyna-1 / ThermoMPNN / ESM-C / GNINA-style painted copy: same coords, new B-factors, parent lineage. */
function makePainted(
  id: string,
  parent: StructureArtifact,
  semantics: BfactorSemantics,
  fill: (i: number) => number,
  extra: Partial<StructureArtifact> = {},
): StructureArtifact {
  const { collection: _collection, ...base } = parent;
  void _collection;
  return {
    ...base,
    id,
    label: `${parent.label} (${semantics})`,
    parentStructureId: parent.id,
    pdbText: paint(parent.pdbText, fill),
    caCoords: new Float32Array(parent.caCoords),
    bfactorSemantics: semantics,
    ...extra,
  };
}

function graphOf(nodes: Array<[string, string, string]>): WorkflowGraph {
  return {
    id: 'track-overlay',
    name: 'Track overlay',
    nodes: nodes.map(([id, adapterId, label]) => ({ id, adapterId, label, params: {} })),
    edges: [],
  };
}

function compare(
  nodes: Array<[string, string, string]>,
  outputs: Record<string, Record<string, CanonicalArtifact>>,
  opts: Partial<Parameters<typeof buildComparisonState>[3]> = {},
): ComparisonState {
  return buildComparisonState(
    graphOf(nodes),
    nodes.map(([id]) => id),
    outputs,
    { layout: 'overlay', alignKabsch: true, ...opts },
  );
}

const design = makeStruct('design-1', BASE_PDB, { label: 'Design Backbone' });
const dyna = makePainted('dyna1-struct-design-1', design, 'dynamics-prob', (i) => (i % 10) / 10);
const thermo = makePainted('thermo-struct-design-1', design, 'ddg-stability', (i) => ((i % 7) - 3) * 0.4);

const dynaTrack: ResidueTrackArtifact = {
  kind: 'ResidueTrack',
  id: 'dyna1-track',
  sourceModelId: 'dyna1',
  parentStructureId: design.id,
  quantity: 'dynamics-prob',
  residues: design.sequence,
  values: Float32Array.from({ length: L }, (_, i) => (i % 10) / 10),
};

const thermoMatrix: MutationMatrixArtifact = {
  kind: 'MutationMatrix',
  id: 'thermo-mm',
  sourceModelId: 'thermompnn',
  parentStructureId: design.id,
  metric: 'ddg-kcal-mol',
  unit: 'kcal/mol',
  direction: 'lower-is-better',
  residues: design.sequence,
  residueIds: Array.from({ length: L }, (_, i) => `A${i + 1}`),
  alphabet: 'ACDEFGHIKLMNPQRSTVWY',
  values: new Float32Array(L * 20),
  meanPerPosition: Float32Array.from({ length: L }, (_, i) => ((i % 7) - 3) * 0.4),
};

const NODES_DESIGN_DYNA: Array<[string, string, string]> = [
  ['gen', 'pdb-input', 'Design Backbone'],
  ['dyna', 'dyna1', 'Dyna-1'],
];

function designDynaOutputs(): Record<string, Record<string, CanonicalArtifact>> {
  return {
    gen: { structure: design },
    dyna: { structure: dyna, track: dynaTrack },
  };
}

describe('compare-identity: merging identical painted structures (Overlay)', () => {
  it('merges a Dyna-1 painted copy with its parent backbone into ONE model with a lineage caption', () => {
    const cmp = compare(NODES_DESIGN_DYNA, designDynaOutputs());
    expect(cmp.models).toHaveLength(1);
    const m = cmp.models[0];
    expect(m.id).toBe('branch-dyna');
    expect(m.visible).toBe(true);
    expect(m.mergedFrom?.role).toBe('primary');
    expect(m.mergedFrom?.evidence).toBe('lineage');
    expect(m.mergedFrom?.caption).toBe('Dyna-1 flexibility painted on Design Backbone');
    expect(m.mergedFrom?.rmsd).toBeLessThan(0.05);
    expect(cmp.mergeGroups).toHaveLength(1);
    expect(cmp.mergeGroups[0].absorbedNodeIds).toEqual(['gen']);
    expect(cmp.mergedModelCount).toBe(1);
    // allPairwise is left alone so its 0.00 Å row explains the merge
    expect(cmp.allPairwise[0].rmsd).toBeLessThan(0.05);
    expect(cmp.allPairwise[0]).toMatchObject({ nodeIdA: 'gen', nodeIdB: 'dyna' });
    expect(resolveCompareDisplayModelId(cmp, 'gen')).toBe('branch-dyna');
  });

  it('keeps both models with mergeIdentical: false while still reporting the mergeable count', () => {
    const cmp = compare(NODES_DESIGN_DYNA, designDynaOutputs(), { mergeIdentical: false });
    expect(cmp.models.map((m) => m.id)).toEqual(['branch-gen', 'branch-dyna']);
    expect(cmp.mergeGroups).toEqual([]);
    expect(cmp.mergedModelCount).toBe(0);
    expect(cmp.mergeableModelCount).toBe(1);
  });

  it('never merges in Side-by-Side (no z-fighting there)', () => {
    const cmp = compare(NODES_DESIGN_DYNA, designDynaOutputs(), { layout: 'side-by-side' });
    expect(cmp.models).toHaveLength(2);
    expect(cmp.mergeGroups).toEqual([]);
    expect(cmp.mergeableModelCount).toBe(0);
  });

  it('does NOT merge two different designs of the same length and sequence (several Å apart, or 1.2 Å)', () => {
    for (const amp of [3.0, 1.2]) {
      const other = makeStruct('design-2', perturbPdb(BASE_PDB, amp), { label: 'Design 2' });
      const dyna2 = makePainted('dyna1-struct-design-2', other, 'dynamics-prob', () => 0.5);
      const cmp = compare(
        [
          ['gen', 'pdb-input', 'Design Backbone'],
          ['dyna2', 'dyna1', 'Dyna-1 (design 2)'],
        ],
        { gen: { structure: design }, dyna2: { structure: dyna2 } },
      );
      expect(cmp.allPairwise[0].rmsd).toBeGreaterThan(0.05);
      expect(cmp.models).toHaveLength(2);
      expect(cmp.mergeGroups).toEqual([]);
    }
  });

  it('lineage alone never merges a painted child whose coordinates moved', () => {
    const moved = makePainted('dyna1-struct-moved', design, 'dynamics-prob', () => 0.3, {
      pdbText: paint(perturbPdb(BASE_PDB, 1.5), () => 0.3),
      caCoords: parseStructureText(perturbPdb(BASE_PDB, 1.5)).caCoords,
    });
    const cmp = compare(NODES_DESIGN_DYNA, { gen: { structure: design }, dyna: { structure: moved } });
    expect(cmp.models).toHaveLength(2);
  });

  it('does NOT merge identical coordinates with different residue numbering', () => {
    const renum = makeStruct('renum', renumberPdb(BASE_PDB, 100), { label: 'Renumbered' });
    const dynaRenum = makePainted('dyna1-struct-renum', renum, 'dynamics-prob', () => 0.2);
    const cmp = compare(NODES_DESIGN_DYNA, {
      gen: { structure: design },
      dyna: { structure: dynaRenum },
    });
    expect(cmp.allPairwise[0].rmsd).toBeLessThan(0.05);
    expect(cmp.models).toHaveLength(2);
  });

  it('does NOT merge two identical plain (track-less) structures', () => {
    const copy = makeStruct('design-copy', BASE_PDB, { label: 'Same PDB again' });
    const cmp = compare(
      [
        ['p1', 'pdb-input', 'Plain 1'],
        ['p2', 'pdb-input', 'Plain 2'],
      ],
      { p1: { structure: design }, p2: { structure: copy } },
    );
    expect(cmp.models).toHaveLength(2);
    expect(cmp.mergeableModelCount).toBe(0);
  });

  it('never merges ca-only with all-atom (e.g. Dyna-1 on Genie 3 + PULCHRA rebuild of the same trace)', () => {
    const genie = makeStruct('genie-1', caOnlyPdb(BASE_PDB), {
      label: 'Genie 3',
      sourceModelId: 'genie3',
      atomGranularity: 'ca-only',
    });
    const dynaCa = makePainted('dyna1-struct-genie-1', genie, 'dynamics-prob', () => 0.4);
    const pulchra = makeStruct('pulchra-genie-1', BASE_PDB, {
      label: 'PULCHRA',
      sourceModelId: 'pulchra',
      parentStructureId: genie.id,
      atomGranularity: 'all-atom' as AtomGranularity,
    });
    const cmp = compare(
      [
        ['dyna', 'dyna1', 'Dyna-1 (Genie 3)'],
        ['pul', 'pulchra', 'PULCHRA'],
      ],
      { dyna: { structure: dynaCa }, pul: { structure: pulchra } },
    );
    expect(cmp.allPairwise[0].rmsd).toBeLessThan(0.05);
    expect(cmp.branches[0].atomGranularity).toBe('ca-only');
    expect(cmp.models).toHaveLength(2);
    expect(cmp.mergeGroups).toEqual([]);

    // Granularity is also inferred from the rendered text when the artifact omits it
    const genieUntagged = makeStruct('genie-2', caOnlyPdb(BASE_PDB), { label: 'Genie 3 (untagged)' });
    const untagged = compare([['g', 'genie3', 'Genie 3']], { g: { structure: genieUntagged } });
    expect(untagged.branches[0].atomGranularity).toBe('ca-only');
  });

  it('never absorbs a holo / docked pose whose ligand is absent from the kept branch (GNINA dock + ThermoMPNN apo)', () => {
    const holo = makePainted('gnina-dock-design-1', design, 'gnina-contrib', (i) => (i % 5 === 0 ? 0.8 : 0), {
      pdbText: withLigand(paint(BASE_PDB, (i) => (i % 5 === 0 ? 0.8 : 0))),
    });
    const cmp = compare(
      [
        ['gnina', 'gnina', 'GNINA dock'],
        ['thermo', 'thermompnn', 'ThermoMPNN (apo)'],
        ['gen', 'pdb-input', 'Design Backbone'],
      ],
      {
        gnina: { structure: holo },
        thermo: { structure: thermo, mutations: thermoMatrix },
        gen: { structure: design },
      },
    );
    const gninaBranch = cmp.branches.find((b) => b.nodeId === 'gnina')!;
    expect(gninaBranch.ligandNames).toContain('ATP');
    // ThermoMPNN still merges with its apo parent, but GNINA's holo pose stays separate
    expect(cmp.mergeGroups).toHaveLength(1);
    expect(cmp.mergeGroups[0].memberNodeIds.sort()).toEqual(['gen', 'thermo']);
    expect(cmp.models.map((m) => m.id).sort()).toEqual(['branch-gnina', 'branch-thermo']);
  });

  it('Dyna-1 + ThermoMPNN on the same backbone form ONE group with 2 switchable tracks', () => {
    const outputs = {
      dyna: { structure: dyna, track: dynaTrack } as Record<string, CanonicalArtifact>,
      thermo: { structure: thermo, mutations: thermoMatrix } as Record<string, CanonicalArtifact>,
    };
    const nodes: Array<[string, string, string]> = [
      ['dyna', 'dyna1', 'Dyna-1'],
      ['thermo', 'thermompnn', 'ThermoMPNN'],
    ];
    const cmp = compare(nodes, outputs);
    expect(cmp.mergeGroups).toHaveLength(1);
    const g = cmp.mergeGroups[0];
    expect(g.trackNodeIds).toEqual(['dyna', 'thermo']);
    expect(g.absorbedNodeIds).toEqual([]);
    expect(g.activeTrackNodeId).toBe('dyna');
    // Both track models stay in the scene; the alternative is hidden (eye / solo still work)
    expect(cmp.models).toHaveLength(2);
    const primary = cmp.models.find((m) => m.id === 'branch-dyna')!;
    const secondary = cmp.models.find((m) => m.id === 'branch-thermo')!;
    expect(primary.visible).toBe(true);
    expect(primary.mergedFrom?.role).toBe('primary');
    expect(primary.mergedFrom?.tracks.map((t) => t.nodeId)).toEqual(['dyna', 'thermo']);
    expect(secondary.visible).toBe(false);
    expect(secondary.mergedFrom?.role).toBe('secondary');

    // Switching the active track only flips visibility/roles: ids + coordinates are unchanged,
    // so the viewer's structural render key is identical (no scene reload, no camera reset)
    const switched = compare(nodes, outputs, { activeTrackNodeIds: { [g.id]: 'thermo' } });
    expect(switched.models.find((m) => m.id === 'branch-thermo')!.visible).toBe(true);
    expect(switched.models.find((m) => m.id === 'branch-dyna')!.visible).toBe(false);
    const structKey = (s: ComparisonState) => s.models.map((m) => `${m.id}|${m.pdbText}`).join('||');
    expect(structKey(switched)).toBe(structKey(cmp));

    // Hiding the group (all member ids hidden together) keeps it hidden whichever track is active
    const hiddenIds = new Set(expandMergedModelIds('branch-dyna', cmp.mergeGroups));
    expect([...hiddenIds].sort()).toEqual(['branch-dyna', 'branch-thermo']);
    const hiddenSwitched = compare(nodes, outputs, {
      hiddenModelIds: hiddenIds,
      activeTrackNodeIds: { [g.id]: 'thermo' },
    });
    expect(hiddenSwitched.models.every((m) => !m.visible)).toBe(true);
  });

  it('Dyna-1 + ThermoMPNN + parent show 1 visible cartoon with 2 tracks; an AF3 fold stays separate', () => {
    const refoldPdb = perturbPdb(BASE_PDB, 0.8);
    const refoldParsed = parseStructureText(refoldPdb);
    const fold: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'af3-folds',
      parentStructureId: design.id,
      folds: [
        {
          sequenceId: 'af3-0',
          sequence: design.sequence,
          foldModelId: 'af3',
          pdbText: refoldPdb,
          caCoords: refoldParsed.caCoords,
          plddt: new Float32Array(L).fill(85),
          meanPlddt: 85,
          ptm: 0.8,
          elapsedMs: 10,
        },
      ],
    };
    const cmp = compare(
      [
        ['gen', 'pdb-input', 'Design Backbone'],
        ['dyna', 'dyna1', 'Dyna-1'],
        ['thermo', 'thermompnn', 'ThermoMPNN'],
        ['af3', 'localfold-af2', 'AF3 fold'],
      ],
      {
        gen: { structure: design },
        dyna: { structure: dyna, track: dynaTrack },
        thermo: { structure: thermo, mutations: thermoMatrix },
        af3: { folds: fold },
      },
    );
    expect(cmp.mergeGroups).toHaveLength(1);
    expect(cmp.mergeGroups[0].memberNodeIds).toEqual(['gen', 'dyna', 'thermo']);
    expect(cmp.models.map((m) => m.id)).toEqual(['branch-dyna', 'branch-thermo', 'branch-af3']);
    expect(cmp.models.filter((m) => m.visible).map((m) => m.id)).toEqual(['branch-dyna', 'branch-af3']);
    expect(cmp.mergedModelCount).toBe(2);
    expect(cmp.models[0].mergedFrom?.caption).toBe('Dyna-1 flexibility painted on Design Backbone');
  });

  it('Plain1 + Plain2 + Track1 groups cleanly around the track-bearing member', () => {
    const plain2 = makeStruct('design-copy', BASE_PDB, { label: 'Same PDB (2nd load)' });
    const cmp = compare(
      [
        ['p1', 'pdb-input', 'Plain 1'],
        ['p2', 'pdb-input', 'Plain 2'],
        ['dyna', 'dyna1', 'Dyna-1'],
      ],
      { p1: { structure: design }, p2: { structure: plain2 }, dyna: { structure: dyna } },
    );
    expect(cmp.mergeGroups).toHaveLength(1);
    const g = cmp.mergeGroups[0];
    expect(g.baseNodeId).toBe('dyna');
    expect(g.memberNodeIds).toEqual(['p1', 'p2', 'dyna']);
    expect(g.absorbedNodeIds).toEqual(['p1', 'p2']);
    // p2 only matched by geometry (not lineage) → group evidence is 'geometry'
    expect(g.evidence).toBe('geometry');
    expect(cmp.models.map((m) => m.id)).toEqual(['branch-dyna']);
    expect(cmp.models[0].mergedFrom?.caption).toBe('Dyna-1 flexibility painted on Plain 1 + Plain 2');
  });

  it('every group has a track-bearing base and members never chain through a non-identical neighbor', () => {
    const groups = findIdenticalTrackGroups(
      [
        { nodeId: 'a', label: 'A', length: 3, sequence: 'AAA', resNums: [1, 2, 3], structureId: 'a', caCoords: null },
        { nodeId: 'b', label: 'B', length: 3, sequence: 'AAA', resNums: [1, 2, 3], structureId: 'b', caCoords: null, bfactorSemantics: 'dynamics-prob' },
        { nodeId: 'c', label: 'C', length: 3, sequence: 'AAA', resNums: [1, 2, 3], structureId: 'c', caCoords: null },
      ],
      [
        { nodeIdA: 'a', nodeIdB: 'b', rmsd: 0.01, alignedResidues: 3, isSubstructure: false },
        { nodeIdA: 'a', nodeIdB: 'c', rmsd: 0.01, alignedResidues: 3, isSubstructure: false },
        // c is NOT identical to the track base b → must not join via a
        { nodeIdA: 'b', nodeIdB: 'c', rmsd: 0.4, alignedResidues: 3, isSubstructure: false },
      ],
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].memberNodeIds).toEqual(['a', 'b']);
    expect(groups[0].trackNodeIds).toEqual(['b']);
  });

  it('each identity guard rejects on its own (granularity, atom count, ligands, chains, raw coords)', () => {
    expect(granularityCompatible('ca-only', 'all-atom')).toBe(false);
    expect(granularityCompatible('ca-only', 'backbone4')).toBe(false);
    expect(granularityCompatible('ca-only', undefined)).toBe(false);
    expect(granularityCompatible('backbone4', 'all-atom')).toBe(false);
    expect(granularityCompatible('all-atom', 'all-atom')).toBe(true);
    expect(granularityCompatible(undefined, 'all-atom')).toBe(true);

    const ca = new Float32Array([0, 0, 0, 3.8, 0, 0, 7.6, 0, 0]);
    const a: IdentityBranch = {
      nodeId: 'a', label: 'A', length: 3, sequence: 'AAA', resNums: [1, 2, 3], chainIds: ['A', 'A', 'A'],
      structureId: 'a', caCoords: ca, renderedAtomCount: 12, ligandNames: [], nucleicAcidChains: [],
      bfactorSemantics: 'dynamics-prob',
    };
    const pair = { nodeIdA: 'a', nodeIdB: 'b', rmsd: 0, alignedResidues: 3, isSubstructure: false };
    const b: IdentityBranch = { ...a, nodeId: 'b', structureId: 'b', bfactorSemantics: null };
    expect(branchesRenderIdentically(a, b, pair)).toBe(true);
    expect(branchesRenderIdentically(a, { ...b, atomGranularity: 'ca-only' }, pair)).toBe(false);
    expect(branchesRenderIdentically(a, { ...b, renderedAtomCount: 3 }, pair)).toBe(false);
    expect(branchesRenderIdentically(a, { ...b, ligandNames: ['ATP'] }, pair)).toBe(false);
    expect(branchesRenderIdentically(a, { ...b, nucleicAcidChains: ['C'] }, pair)).toBe(false);
    expect(branchesRenderIdentically(a, { ...b, chainIds: ['A', 'A', 'B'] }, pair)).toBe(false);
    expect(branchesRenderIdentically(a, { ...b, sequence: 'AAG' }, pair)).toBe(false);
    expect(branchesRenderIdentically(a, b, { ...pair, rmsd: 0.06 })).toBe(false);
    expect(branchesRenderIdentically(a, b, null)).toBe(false);
    // With Kabsch alignment off, a rigidly moved copy is NOT the same rendered structure
    const moved = { ...b, caCoords: ca.map((v, i) => (i % 3 === 0 ? v + 5 : v)) };
    expect(branchesRenderIdentically(a, moved, pair)).toBe(true);
    expect(branchesRenderIdentically(a, moved, pair, { requireRawCoords: true })).toBe(false);
  });

  it('rejects target-chain-only / substructure alignments as identity evidence', () => {
    const base = { label: 'X', length: 3, sequence: 'AAA', resNums: [1, 2, 3], caCoords: null };
    const groups = findIdenticalTrackGroups(
      [
        { ...base, nodeId: 'a', structureId: 'a' },
        { ...base, nodeId: 'b', structureId: 'b', bfactorSemantics: 'ddg-stability' },
      ],
      [{ nodeIdA: 'a', nodeIdB: 'b', rmsd: 0, alignedResidues: 2, isSubstructure: true }],
    );
    expect(groups).toEqual([]);
  });
});

describe('compare-identity: synthetic ref-designed ghost absorption', () => {
  const makeMetrics = (id: string, amp: number): DesignabilityMetrics => {
    const refold = perturbPdb(BASE_PDB, amp);
    const row = {
      sequenceId: `${id}-seq`,
      candidateKey: 'cand-1',
      structureId: design.id,
      sequence: design.sequence,
      scRmsd: 0.6,
      scTm: 0.92,
      meanPlddt: 88,
      ptm: 0.85,
      designable: true,
      geometry: design.geometry,
      foldElapsedMs: 10,
      designedPdb: BASE_PDB,
      alignedFoldPdb: refold,
      plddt: new Float32Array(L).fill(88),
    };
    return { kind: 'DesignabilityMetrics', id, rows: [row], best: row, passRate: 1 };
  };
  const scorerOutputs = {
    score_a: { metrics: makeMetrics('m_a', 0.6) },
    score_b: { metrics: makeMetrics('m_b', 0.9) },
  };

  it('keeps the ghost when no selected branch draws the designed structure', () => {
    const cmp = compare(
      [
        ['score_a', 'designability-scorer', 'Scorer A'],
        ['score_b', 'designability-scorer', 'Scorer B'],
      ],
      scorerOutputs,
    );
    expect(cmp.models.some((m) => m.id === 'ref-designed')).toBe(true);
    expect(cmp.ghostAbsorbedBy).toBeNull();
  });

  it('absorbs the ghost into the selected design branch with the same structureId (no 0.00 Å z-fight)', () => {
    const nodes: Array<[string, string, string]> = [
      ['gen', 'pdb-input', 'Design Backbone'],
      ['score_a', 'designability-scorer', 'Scorer A'],
      ['score_b', 'designability-scorer', 'Scorer B'],
    ];
    const outputs = { gen: { structure: design }, ...scorerOutputs };
    const cmp = compare(nodes, outputs);
    expect(cmp.sharedReferenceLabel).toBe('Designed Backbone');
    expect(cmp.models.some((m) => m.id === 'ref-designed')).toBe(false);
    expect(cmp.ghostAbsorbedBy).toBe('branch-gen');
    expect(cmp.models.find((m) => m.id === 'branch-gen')?.mergedFrom?.absorbedLabels).toEqual(['Designed']);
    expect(cmp.mergedModelCount).toBe(1);

    // Split restores the ghost
    const split = compare(nodes, outputs, { mergeIdentical: false });
    expect(split.models.some((m) => m.id === 'ref-designed')).toBe(true);
    expect(split.mergeableModelCount).toBe(1);
  });

  it('absorbs the ghost into a Dyna-1 painted child and keeps it in the designed frame', () => {
    const cmp = compare(
      [
        ['gen', 'pdb-input', 'Design Backbone'],
        ['dyna', 'dyna1', 'Dyna-1'],
        ['score_a', 'designability-scorer', 'Scorer A'],
        ['score_b', 'designability-scorer', 'Scorer B'],
      ],
      { gen: { structure: design }, dyna: { structure: dyna }, ...scorerOutputs },
    );
    expect(cmp.models.some((m) => m.id === 'ref-designed')).toBe(false);
    expect(cmp.ghostAbsorbedBy).toBe('branch-dyna');
    expect(cmp.models.map((m) => m.id)).toEqual(['branch-dyna', 'branch-score_a', 'branch-score_b']);
    const dynaModel = cmp.models[0];
    expect(dynaModel.mergedFrom?.absorbedLabels).toEqual(['Design Backbone', 'Designed']);
    // Painted child with unchanged coords is rendered raw (designed frame), not Kabsch-fit to a refold
    const rendered = parseStructureText(dynaModel.pdbText).caCoords;
    for (let i = 0; i < 9; i++) expect(rendered[i]).toBeCloseTo(design.caCoords[i], 2);
    expect(cmp.mergedModelCount).toBe(2);

    // Toggling Split (mergeIdentical: false) must also keep the painted child in the designed frame
    const split = compare(
      [
        ['gen', 'pdb-input', 'Design Backbone'],
        ['dyna', 'dyna1', 'Dyna-1'],
        ['score_a', 'designability-scorer', 'Scorer A'],
        ['score_b', 'designability-scorer', 'Scorer B'],
      ],
      { gen: { structure: design }, dyna: { structure: dyna }, ...scorerOutputs },
      { mergeIdentical: false },
    );
    const splitDyna = split.models.find((m) => m.id === 'branch-dyna')!;
    expect(splitDyna).toBeDefined();
    const splitRendered = parseStructureText(splitDyna.pdbText).caCoords;
    for (let i = 0; i < 9; i++) expect(splitRendered[i]).toBeCloseTo(design.caCoords[i], 2);
  });
});

describe('Track color mode', () => {
  it('auto (null) resolves to Track when a track branch exists: track model painted, companions muted gray', () => {
    const copy = makeStruct('other-design', perturbPdb(BASE_PDB, 2.5), { label: 'Other design' });
    const nodes: Array<[string, string, string]> = [
      ['dyna', 'dyna1', 'Dyna-1'],
      ['other', 'pdb-input', 'Other design'],
    ];
    const outputs = { dyna: { structure: dyna }, other: { structure: copy } };
    for (const colorMode of [null, undefined, 'track'] as const) {
      const cmp = compare(nodes, outputs, { colorMode });
      expect(cmp.effectiveColorMode).toBe('track');
      expect(cmp.trackSemantics).toEqual(['dynamics-prob']);
      const t = cmp.models.find((m) => m.id === 'branch-dyna')!;
      expect(t.colorByTrack).toBe(true);
      expect(t.bfactorSemantics).toBe('dynamics-prob');
      const o = cmp.models.find((m) => m.id === 'branch-other')!;
      expect(o.colorByTrack).toBe(false);
      expect(o.colorHex).toBe(REFERENCE_BACKBONE_COLOR.hex);
      expect(o.alpha).toBe(0.45);
    }
    // Side-by-Side keeps muted companions opaque
    const sbs = compare(nodes, outputs, { colorMode: 'track', layout: 'side-by-side' });
    expect(sbs.models.find((m) => m.id === 'branch-other')!.alpha).toBe(1.0);
    // By Model keeps palette colors but still tags semantics for hover units / Uncertainty style
    const byModel = compare(nodes, outputs, { colorMode: 'by-model' });
    expect(byModel.effectiveColorMode).toBe('by-model');
    const tb = byModel.models.find((m) => m.id === 'branch-dyna')!;
    expect(tb.colorByTrack).toBeFalsy();
    expect(tb.bfactorSemantics).toBe('dynamics-prob');
  });

  it('falls back to by-model when no track branch exists (auto and explicit track)', () => {
    const copy = makeStruct('other-design', perturbPdb(BASE_PDB, 2.5), { label: 'Other design' });
    const nodes: Array<[string, string, string]> = [
      ['gen', 'pdb-input', 'Design'],
      ['other', 'pdb-input', 'Other design'],
    ];
    const outputs = { gen: { structure: design }, other: { structure: copy } };
    for (const colorMode of [null, 'track'] as const) {
      const cmp = compare(nodes, outputs, { colorMode });
      expect(cmp.trackSemantics).toEqual([]);
      expect(cmp.effectiveColorMode).toBe('by-model');
      expect(cmp.models.every((m) => !m.colorByTrack && m.alpha === 1.0)).toBe(true);
    }
  });

  it('isTrackSemantics covers exactly the four painted track kinds', () => {
    for (const sem of ['dynamics-prob', 'ddg-stability', 'llr-fitness', 'gnina-contrib']) {
      expect(isTrackSemantics(sem)).toBe(true);
      expect(TRACK_NOUN[sem as keyof typeof TRACK_NOUN]).toBeTruthy();
    }
    for (const sem of ['plddt-0-100', 'plddt-0-1', 'crystal-b', 'none', undefined, null]) {
      expect(isTrackSemantics(sem)).toBe(false);
    }
  });
});

describe('Pooled track color domains', () => {
  it('Dyna-1 dynamics probabilities always map to [0, 1]', () => {
    const pooled = poolUncertaintyColorArgs([
      { semantics: 'dynamics-prob' as const, values: [0.05, 0.2, 0.31] },
      { semantics: 'dynamics-prob' as const, values: [0.1, 0.9] },
    ]);
    expect(pooled.get('dynamics-prob')?.colorParams?.domain).toEqual([0, 1]);
  });

  it('pools ΔΔG across structures so the same color means the same value in every model', () => {
    const a = [-0.5, -0.2, 0.1, 0.5];
    const b = [-1.8, 0.4, 1.8];
    expect(resolveUncertaintyColorArgFromValues(a, 'ddg-stability').colorParams?.domain).toEqual([-1.5, 1.5]);
    const pooled = poolUncertaintyColorArgs([
      { semantics: 'ddg-stability' as const, values: a },
      { semantics: 'ddg-stability' as const, values: b },
      { semantics: 'dynamics-prob' as const, values: [0.2] },
    ]);
    expect(pooled.get('ddg-stability')?.colorParams?.domain).toEqual([-1.8, 1.8]);
    expect(pooled.get('ddg-stability')).toEqual(
      resolveUncertaintyColorArgFromValues([...a, ...b], 'ddg-stability'),
    );
    expect(pooled.size).toBe(2);
  });
});

describe('structure-compare display-only params', () => {
  it('COMPARE_DISPLAY_ONLY_PARAM_KEYS stays in parity with the plugin displayOnlyParamKeys (incl. mergeIdentical)', () => {
    const pluginKeys = new Set(structureComparePlugin.display?.displayOnlyParamKeys ?? []);
    expect([...COMPARE_DISPLAY_ONLY_PARAM_KEYS].sort()).toEqual([...pluginKeys].sort());
    expect(pluginKeys.has('mergeIdentical')).toBe(true);
    const node = {
      adapterId: 'structure-compare',
      params: { layout: 'overlay', colorMode: 'by-model', alignKabsch: true, mergeIdentical: true },
    };
    expect(isCompareDisplayOnlyParamChange(node, { ...node.params, mergeIdentical: false })).toBe(true);
    expect(isCompareDisplayOnlyParamChange(node, { ...node.params, colorMode: 'track' })).toBe(true);
  });

  it('formatNodeParamSummary reflects Track color mode and the non-default Split state', () => {
    expect(formatNodeParamSummary('structure-compare', { layout: 'overlay', colorMode: 'track' })).toBe(
      'Overlay · Kabsch · Track',
    );
    expect(
      formatNodeParamSummary('structure-compare', { layout: 'overlay', colorMode: 'by-model', mergeIdentical: false }),
    ).toBe('Overlay · Kabsch · Split');
    expect(
      formatNodeParamSummary('structure-compare', { layout: 'side-by-side', mergeIdentical: false }),
    ).toBe('Side-by-side · Kabsch');
  });
});

describe('Compare 1D residue tracks', () => {
  const esmcMatrix: MutationMatrixArtifact = {
    kind: 'MutationMatrix',
    id: 'esmc-mm',
    sourceModelId: 'esmc',
    parentStructureId: design.id,
    metric: 'llr-fitness',
    unit: 'LLR',
    direction: 'higher-is-better',
    residues: design.sequence,
    residueIds: Array.from({ length: L }, (_, i) => `A${i + 1}`),
    alphabet: 'ACDEFGHIKLMNPQRSTVWY',
    values: new Float32Array(L * 20),
    meanPerPosition: Float32Array.from({ length: L }, (_, i) => -3 + (i % 6) * 0.5),
  };
  const esmcWtTrack: ResidueTrackArtifact = {
    kind: 'ResidueTrack',
    id: 'esmc-track',
    sourceModelId: 'esmc',
    parentStructureId: design.id,
    quantity: 'wt-log-prob',
    unit: 'log p',
    residues: design.sequence,
    values: new Float32Array(L).fill(-0.4),
  };

  it('ESM-C prefers MutationMatrix.meanPerPosition (matches the 3D llr-fitness B-factors) over the wt-log-prob track', () => {
    const esmcStruct = makePainted('esmc-struct-design-1', design, 'llr-fitness', (i) => -3 + (i % 6) * 0.5);
    const cmp = compare(
      [
        ['esmc', 'esmc', 'ESM-C'],
        ['gen', 'pdb-input', 'Design Backbone'],
      ],
      {
        esmc: { mutations: esmcMatrix, track: esmcWtTrack, structure: esmcStruct },
        gen: { structure: design },
      },
    );
    const b = cmp.branches.find((x) => x.nodeId === 'esmc')!;
    expect(b.bfactorSemantics).toBe('llr-fitness');
    expect(b.residueTrack?.quantity).toBe('llr-fitness');
    expect(b.residueTrack?.values).toBe(esmcMatrix.meanPerPosition);
    expect(b.residueTrack?.residueIds).toEqual(esmcMatrix.residueIds);
  });

  it('extracts the 1D track for a SequenceSet-only ESM-C branch without enabling 3D Track mode', () => {
    const seqs: SequenceSet = {
      kind: 'SequenceSet',
      id: 'esmc-seqs',
      parentStructureId: 'none',
      sourceModelId: 'esmc',
      candidates: [
        { id: 'wt', sequence: design.sequence, score: -1.2, globalScore: null, sourceModelId: 'esmc' },
      ],
    };
    const cmp = compare(
      [
        ['esmc', 'esmc', 'ESM-C'],
        ['gen', 'pdb-input', 'Design Backbone'],
      ],
      { esmc: { mutations: esmcMatrix, track: esmcWtTrack, sequences: seqs }, gen: { structure: design } },
    );
    const b = cmp.branches.find((x) => x.nodeId === 'esmc')!;
    expect(b.kind).toBe('sequence');
    expect(b.residueTrack?.values).toBe(esmcMatrix.meanPerPosition);
    expect(b.bfactorSemantics).toBeNull();
    expect(cmp.trackSemantics).toEqual([]);
    expect(cmp.effectiveColorMode).toBe('by-model');
  });

  it('uses an emitted ResidueTrack (Dyna-1) and wraps ThermoMPNN meanPerPosition when no track is emitted', () => {
    const cmp = compare(
      [
        ['dyna', 'dyna1', 'Dyna-1'],
        ['thermo', 'thermompnn', 'ThermoMPNN'],
      ],
      {
        dyna: { structure: dyna, track: dynaTrack },
        thermo: { structure: thermo, mutations: thermoMatrix },
      },
    );
    expect(cmp.branches[0].residueTrack).toBe(dynaTrack);
    expect(cmp.branches[1].residueTrack?.quantity).toBe('ddg-kcal-mol');
    expect(cmp.branches[1].residueTrack?.values).toBe(thermoMatrix.meanPerPosition);
  });

  it('drops a track whose length does not match the selected candidate', () => {
    const short: ResidueTrackArtifact = { ...dynaTrack, values: new Float32Array(L - 4) };
    const cmp = compare([['dyna', 'dyna1', 'Dyna-1']], { dyna: { structure: dyna, track: short } });
    expect(cmp.branches[0].residueTrack).toBeNull();
  });
});

describe('Compare track overlay — viewer / page / panel source wiring', () => {
  const viewerSrc = readWebFile('src/lib/viewer/MolSuperposeViewer.svelte');
  const pageSrc = readWebFile('src/routes/+page.svelte');
  const panelSrc = readWebFile('src/lib/ui/ComparePeekPanel.svelte');
  const compareSrc = readWebFile('src/lib/engine/compare.ts');

  it('gates the Track button on trackSemantics and shows the merge toggle only when merging is possible', () => {
    expect(pageSrc).toMatch(
      /\{#if comparisonState\.trackSemantics\.length > 0\}[\s\S]*?setActiveCompareOption\(\{ colorMode: 'track' \}\)[\s\S]*?Track/,
    );
    expect(pageSrc).toContain("class:active={comparisonState.effectiveColorMode === 'track'}");
    expect(pageSrc).toContain('comparisonState.mergeableModelCount > 0');
    expect(pageSrc).toContain('mergeIdentical: effectiveCompareMergeIdentical');
    expect(pageSrc).toContain("comparisonState.mergedModelCount} merged");
  });

  it('keeps color/alpha/semantics out of multiRenderKey and serializes them in the style key (camera preserved)', () => {
    const keyBlock = viewerSrc.match(/let multiRenderKey = \$derived\.by\([\s\S]*?\n {2}\}\);/)?.[0] ?? '';
    expect(keyBlock).toContain('hashStructureSample(m.pdbText)');
    expect(keyBlock).not.toMatch(/colorHex|alpha|colorByPlddt|colorByTrack|bfactorSemantics/);
    expect(viewerSrc).toMatch(
      /let multiStyleKey = \$derived\.by\([\s\S]*?m\.colorHex[\s\S]*?m\.alpha[\s\S]*?m\.colorByPlddt[\s\S]*?m\.colorByTrack[\s\S]*?m\.bfactorSemantics/,
    );
    expect(viewerSrc).toContain('const fullKey = `${mKey}##${styleKey}##${multiStyleKey}`;');
  });

  it('pools track domains in pass 1 for every track model (including hidden merged tracks) into $state', () => {
    expect(viewerSrc).toContain('let trackDomainsBySemantics = $state<Record<string, [number, number]>>({});');
    expect(viewerSrc).toMatch(
      /isTrackSemantics\(m\.bfactorSemantics\) &&\s*\(m\.colorByTrack \|\| styleOpts\.color === 'uncertainty'\)/,
    );
    expect(viewerSrc).toContain('poolUncertaintyColorArgs(pooledEntries)');
    expect(viewerSrc).toContain('trackDomainsBySemantics = nextDomains;');
    // plddt-confidence never misreads a track model's B-factors
    expect(viewerSrc).toMatch(/\(!styleOpts\.plddtAllowed \|\| isTrackSemantics\(m\.bfactorSemantics\)\)/);
  });

  it('gates the multi-model legend on visible painted track models and Auto / Uncertainty color', () => {
    expect(viewerSrc).toMatch(
      /if \(!hasAnyStructure \|\| \(colorScheme !== 'auto' && colorScheme !== 'uncertainty'\)\) return \[\];/,
    );
    expect(viewerSrc).toMatch(/if \(!m\.visible \|\| !isTrackSemantics\(m\.bfactorSemantics\)\) continue;/);
    expect(viewerSrc).toContain("const painted = colorScheme === 'uncertainty' || Boolean(m.colorByTrack);");
  });

  it('resolves hover units per model and recognises plddt-0-100', () => {
    expect(viewerSrc).toContain('const hoveredModelId = loc.unit?.model?.id;');
    expect(viewerSrc).toContain("m.bfactorSemantics ?? (m.colorByPlddt ? 'plddt-0-100' : undefined)");
    expect(viewerSrc).toContain("if (sem === 'plddt' || sem === 'plddt-0-100')");
  });

  it('passes alpha to the sparse (<= 15 aa) motif ball-and-stick so muted companions stay translucent', () => {
    expect(viewerSrc).toMatch(
      /const sparseRepr = await ctx\.builders\.structure\.representation\.addRepresentation\(struct, \{[\s\S]*?typeParams: \{ alpha: opts\.alpha \?\? 1\.0 \}/,
    );
  });

  it('filters secondary merged tracks out of the legend and wires track switch + Split on merged chips', () => {
    expect(viewerSrc).toContain("(models ?? []).filter((m) => m.mergedFrom?.role !== 'secondary')");
    expect(viewerSrc).toContain('{#each legendModels as m (m.id)}');
    expect(viewerSrc).toContain('onSelectTrack?.(merge.groupId, t.nodeId)');
    expect(viewerSrc).toContain('onToggleMergeIdentical?.()');
    expect(viewerSrc).toContain('⧉ merged · {merge.rmsd.toFixed(2)} Å · {merge.evidence}');
  });

  it('applies hide / solo across the whole merge group and keeps it on track switch', () => {
    expect(pageSrc).toMatch(
      /function toggleCompareModelVisibility\(modelId: string\) \{\s*const groupIds = expandMergedModelIds\(modelId, comparisonState\?\.mergeGroups \?\? \[\]\);/,
    );
    expect(pageSrc).toMatch(/function soloCompareModel[\s\S]*?const soloIds = expandMergedModelIds\(modelId, groups\);/);
    expect(pageSrc).toMatch(/function selectCompareTrack[\s\S]*?compareActiveTrackByGroup = \{ \.\.\.compareActiveTrackByGroup, \[groupId\]: nodeId \};/);
    expect(pageSrc).toContain('onSelectTrack={isCompareMode ? selectCompareTrack : undefined}');
  });

  it('targets click-to-focus at the branch’s own model and skips hidden models', () => {
    expect(viewerSrc).toMatch(/async function resolveTokenLoci\(\s*ctx: any,\s*token: string,\s*opts: \{ modelId\?: string \| null \} = \{\},/);
    expect(viewerSrc).toContain('if (specId && hiddenSpecIds.has(specId)) continue;');
    expect(viewerSrc).toContain('if (targetSpecId && specId !== targetSpecId) continue;');
    expect(viewerSrc).toContain('const loci = await resolveTokenLoci(ctx, tok, { modelId: targetModelId });');
    expect(pageSrc).toContain('focusModelId = resolveCompareDisplayModelId(comparisonState, source.nodeId);');
    expect(pageSrc).toContain('handleSelectPerResidue(positionIndex, residueId, { nodeId })');
    expect(pageSrc).toContain('focusModelId={isCompareMode && !studioOpen ? viewerFocusModelId : null}');
  });

  it('mounts Compare ResidueTrackPlots BELOW the metrics table with branch badge headers', () => {
    const tableEnd = panelSrc.indexOf('</table>');
    const plotIdx = panelSrc.indexOf('<ResidueTrackPlot');
    expect(tableEnd).toBeGreaterThan(0);
    expect(plotIdx).toBeGreaterThan(tableEnd);
    expect(panelSrc).toMatch(/<ResidueTrackPlot[\s\S]*?height=\{130\}/);
    expect(panelSrc).toMatch(/\{#each trackBranches as b \(b\.nodeId\)\}[\s\S]*?class="badge-dot sm"[\s\S]*?\{b\.badge\}/);
    expect(panelSrc).toContain('onSelectResidue(b.nodeId, positionIndex, residueId)');
  });

  it('parses the rendered text (cifText || pdbText) for structure identity fields', () => {
    expect(compareSrc).toContain('const parsed = parseStructureText(displayStructText);');
    expect(compareSrc).not.toContain("parseStructureText(struct.pdbText || struct.cifText || '')");
  });
});

describe('Multi-design StructureArtifact collections in Compare & per-design 1D tracks', () => {
  const BASE_LEN = parseStructureText(BASE_PDB).nResidues;

  function makeBatch(
    batchId: string,
    items: StructureArtifact[],
    opts: { repIdx?: number; criterion?: string } = {},
  ): StructureArtifact {
    const repIdx = opts.repIdx ?? 0;
    const rep = items[repIdx] ?? items[0];
    return {
      ...rep,
      id: batchId,
      ...(opts.repIdx !== undefined
        ? {
            representativeSelection: {
              selectedIndex: repIdx,
              selectedKey: rep.designKey ?? rep.id,
              criterion: opts.criterion ?? 'best-cnn-score',
              scoreValue: 0.9,
              totalMembers: items.length,
            },
          }
        : {}),
      collection: {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id: `${batchId}-col`,
        familyKey: batchId,
        collectionKind: 'independent-candidates',
        requestedCount: items.length,
        complete: true,
        items,
        keys: items.map((it) => it.designKey ?? it.id),
      },
    };
  }

  it('switches both 3D structure and per-design 1D track across Design 1..3 in By candidate mode', () => {
    const d0Pdb = BASE_PDB;
    const d1Pdb = perturbPdb(BASE_PDB, 1.8);
    const d2Pdb = perturbPdb(BASE_PDB, 3.4);
    const L = BASE_LEN;

    const rfdItems = [
      { ...makeStruct('rfd-d0', d0Pdb, { label: 'RFD4 #1/3' }), designKey: 'design:abc12345:d0' },
      { ...makeStruct('rfd-d1', d1Pdb, { label: 'RFD4 #2/3' }), designKey: 'design:def67890:d1' },
      { ...makeStruct('rfd-d2', d2Pdb, { label: 'RFD4 #3/3' }), designKey: 'design:99988877:d2' },
    ];
    const dynaBfs = [
      new Float32Array(L).fill(0.2),
      new Float32Array(L).fill(0.5),
      new Float32Array(L).fill(0.85),
    ];
    const dynaItems = rfdItems.map((r, i) => ({
      ...makeStruct(`dyna-d${i}`, paintStructureBfactors(r.pdbText, dynaBfs[i]), {
        label: `Dyna-1 #${i + 1}/3`,
        bfactorSemantics: 'dynamics-prob',
        parentStructureId: r.id,
      }),
      designKey: r.designKey,
    }));

    const rfdBatch = makeBatch('rfd-batch', rfdItems);
    const dynaBatch = makeBatch('dyna-batch', dynaItems);
    const graph: WorkflowGraph = {
      id: 'multi-cmp',
      name: 'Multi Compare',
      description: '',
      nodes: [
        { id: 'rfd', adapterId: 'rfd4-proteina', label: 'RFD4', params: {}, position: { x: 0, y: 0 } },
        { id: 'dyna', adapterId: 'dyna1', label: 'Dyna-1', params: {}, position: { x: 200, y: 0 } },
      ],
      edges: [{ fromNode: 'rfd', fromPort: 'structure', toNode: 'dyna', toPort: 'structure' }],
    };
    const outputs: Record<string, Record<string, CanonicalArtifact>> = {
      rfd: { structure: rfdBatch },
      dyna: { structure: dynaBatch },
    };

    // Select Design 3 (d2) in By candidate mode
    const cmp = buildComparisonState(graph, ['dyna', 'rfd'], outputs, {
      layout: 'overlay',
      pairingMode: 'by-candidate',
      selectedCandidateKey: 'design:99988877:d2',
    });
    expect(cmp.pairingMode).toBe('by-candidate');
    expect(cmp.candidateKeys).toEqual(['design:abc12345:d0', 'design:def67890:d1', 'design:99988877:d2']);
    expect(cmp.branches[0].structureId).toBe('dyna-d2');
    expect(cmp.branches[0].selectedIndex).toBe(2);
    expect(cmp.branches[0].candidateCount).toBe(3);
    expect(cmp.branches[1].structureId).toBe('rfd-d2');
    expect(cmp.branches[1].selectedIndex).toBe(2);
    // Identical painted Design 3 merges cleanly with RFD4 Design 3 via lineage
    expect(cmp.mergeGroups.length).toBe(1);
    expect(cmp.mergeGroups[0].evidence).toBe('lineage');
    // Per-design 1D track reflects Design 3's B-factors (0.85), NOT Design 1's (0.20)
    expect(cmp.branches[0].residueTrack).not.toBeNull();
    expect(cmp.branches[0].residueTrack!.summary?.mean).toBeCloseTo(0.85, 2);

    // Independent per-branch selection in Branch best mode: Dyna-1 Design 3 vs RFD4 Design 1
    const cross = buildComparisonState(graph, ['dyna', 'rfd'], outputs, {
      layout: 'overlay',
      pairingMode: 'branch-best',
      branchCandidateIdx: { dyna: 2, rfd: 0 },
    });
    expect(cross.branches[0].structureId).toBe('dyna-d2');
    expect(cross.branches[1].structureId).toBe('rfd-d0');
    // Different designs do NOT merge and remain separate Kabsch-aligned models
    expect(cross.mergeGroups.length).toBe(0);
    expect(cross.models.length).toBe(2);
  });

  it('defaults Branch best to representativeSelection.selectedIndex (e.g. GNINA best pose)', () => {
    const items = [
      { ...makeStruct('gn-d0', BASE_PDB), designKey: 'design:11111111:d0' },
      { ...makeStruct('gn-d1', perturbPdb(BASE_PDB, 1.5)), designKey: 'design:22222222:d1' },
    ];
    const gnBatch = makeBatch('gn-batch', items, { repIdx: 1, criterion: 'highest-cnn-score' });
    const cmp = compare(
      [
        ['gn', 'gnina', 'GNINA'],
        ['ref', 'pdb-input', 'Ref'],
      ],
      { gn: { structure: gnBatch }, ref: { structure: makeStruct('ref-s', BASE_PDB) } },
      { pairingMode: 'branch-best' },
    );
    expect(cmp.branches[0].structureId).toBe('gn-d1');
    expect(cmp.branches[0].selectedIndex).toBe(1);
    expect(cmp.branches[0].rankingPolicy).toBe('highest-cnn-score');
  });

  it('does not falsely pair unrelated generators with generic d1..d2 keys and identical sequences', () => {
    const genA = makeBatch('gen-a', [
      { ...makeStruct('a1', BASE_PDB), designKey: 'd1' },
      { ...makeStruct('a2', perturbPdb(BASE_PDB, 1.2)), designKey: 'd2' },
    ]);
    const genB = makeBatch('gen-b', [
      { ...makeStruct('b1', perturbPdb(BASE_PDB, 2.0)), designKey: 'd1' },
      { ...makeStruct('b2', perturbPdb(BASE_PDB, 3.0)), designKey: 'd2' },
    ]);
    // Both have distinct sequences? No, identical BASE_PDB sequence, no shared ancestry.
    // Using distinct sequences prevents sequence-fallback pairing too:
    genB.collection!.items[0].sequence = 'G'.repeat(BASE_LEN);
    genB.collection!.items[1].sequence = 'V'.repeat(BASE_LEN);
    genB.sequence = genB.collection!.items[0].sequence;
    const cmp = compare(
      [
        ['ga', 'genie2', 'Genie 2 A'],
        ['gb', 'frameflow', 'FrameFlow B'],
      ],
      { ga: { structure: genA }, gb: { structure: genB } },
    );
    expect(cmp.pairingMode).toBe('branch-best');
  });

  it('deriveResidueTrackFromStructure extracts per-residue B-factor tracks accurately', () => {
    const L = BASE_LEN;
    const vals = new Float32Array(L);
    for (let i = 0; i < L; i++) vals[i] = i / Math.max(1, L - 1);
    const painted = makeStruct('painted-1', paintStructureBfactors(BASE_PDB, vals), {
      bfactorSemantics: 'dynamics-prob',
    });
    const track = deriveResidueTrackFromStructure(painted);
    expect(track).not.toBeNull();
    expect(track!.values.length).toBe(L);
    expect(track!.unit).toBe('p(exch)');
    expect(track!.summary?.mean).toBeCloseTo(0.5, 2);
  });
});
