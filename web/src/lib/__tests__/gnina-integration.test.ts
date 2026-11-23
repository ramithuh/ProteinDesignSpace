import { describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { filterCatalogCategories } from '$lib/engine/catalog';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { compileNodeInputPreview } from '$lib/engine/input-preview';
import { encodeGraphToHash, decodeHashToGraph } from '$lib/engine/share';
import { getModelCardForAdapter } from '$lib/engine/model-cards';
import {
  applyMinimizedHoloLigandCoords,
  buildMultiPoseDockedPdb,
  computePoseHeavyRmsd,
  extractHoloReceptorAndLigands,
  findPocketResidues,
  parsePocketResidueTokens,
} from '$lib/protein-ops/gnina-prep';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { resolveUncertaintyColorArgFromValues } from '$lib/viewer/bfactor-domain';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import type { StructureArtifact, WorkflowGraph } from '$lib/engine/types';

// Minimal holo metalloenzyme complex: 3 protein residues + catalytic ZN + organic ligand DHZ + water HOH
const MINI_HOLO_PDB = [
  'ATOM      1  N   HIS A  10      10.000  10.000  10.000  1.00 20.00           N  ',
  'ATOM      2  CA  HIS A  10      11.458  10.000  10.000  1.00 20.00           C  ',
  'ATOM      3  C   HIS A  10      12.000  11.420  10.000  1.00 20.00           C  ',
  'ATOM      4  O   HIS A  10      11.250  12.400  10.000  1.00 20.00           O  ',
  'ATOM      5  N   ASP A  11      13.320  11.520  10.000  1.00 20.00           N  ',
  'ATOM      6  CA  ASP A  11      13.980  12.820  10.000  1.00 20.00           C  ',
  'ATOM      7  C   ASP A  11      15.490  12.680  10.000  1.00 20.00           C  ',
  'ATOM      8  O   ASP A  11      16.050  11.580  10.000  1.00 20.00           O  ',
  'ATOM      9  N   ALA A  12      16.140  13.840  10.000  1.00 20.00           N  ',
  'ATOM     10  CA  ALA A  12      17.590  13.880  10.000  1.00 20.00           C  ',
  'ATOM     11  C   ALA A  12      18.120  15.300  10.000  1.00 20.00           C  ',
  'ATOM     12  O   ALA A  12      17.380  16.280  10.000  1.00 20.00           O  ',
  'HETATM   13 ZN    ZN A 200      12.500  11.000  12.500  1.00 15.00          ZN  ',
  'HETATM   14  C1  DHZ A 300      13.000  11.000  14.800  1.00 18.00           C  ',
  'HETATM   15  C2  DHZ A 300      14.390  11.000  14.800  1.00 18.00           C  ',
  'HETATM   16  O1  DHZ A 300      12.300  11.000  13.790  1.00 18.00           O  ',
  'HETATM   17  N1  DHZ A 300      15.100  11.000  15.980  1.00 18.00           N  ',
  'HETATM   18  O   HOH A 401      22.000  22.000  22.000  1.00 30.00           O  ',
  'END',
].join('\n');

const MINI_APO_PDB = MINI_HOLO_PDB.split('\n')
  .filter((l) => !l.includes('DHZ'))
  .join('\n');

function makeStructArtifact(pdbText: string, label = 'Mini Complex'): StructureArtifact {
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id: `struct-${label.toLowerCase().replace(/\s+/g, '-')}`,
    label,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    sourceResidues: parsed.sourceResidues,
    diagnostics: parsed.diagnostics,
    atomGranularity: 'all-atom',
    bfactorSemantics: 'crystal-b',
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds),
  };
}

describe('GNINA 1.3 WebGPU integration', () => {
  it('registers gnina adapter, model card, catalog entry, and compact share code', async () => {
    const manifest = ADAPTER_REGISTRY.gnina;
    expect(manifest).toBeDefined();
    expect(manifest.stage).toBe('score');
    expect(manifest.runtime).toBe('wgsl-custom');
    expect(manifest.precisionWeightsMB).toEqual({ int8: 2, fp16: 4, fp32: 7 });
    expect(manifest.modes?.map((m) => m.id)).toEqual(['score', 'dock', 'pocket-scan', 'minimize']);

    const card = getModelCardForAdapter('gnina');
    expect(card?.id).toBe('gnina-webgpu');
    expect(card?.status).toBe('live');
    expect(card?.activePrecisionId).toBe('fp16');

    const search = filterCatalogCategories('gnina');
    expect(search.length).toBeGreaterThanOrEqual(1);
    expect(search[0].matchingSubItems.some((s) => s.key === 'gnina')).toBe(true);

    const graph: WorkflowGraph = {
      id: 'gnina-share-test',
      name: 'GNINA Share Test',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          label: 'PDB Input',
          params: { ...defaultParamsForAdapter('pdb-input'), source: 'rcsb', pdbId: '1STP' },
        },
        {
          id: 'gnina_1',
          adapterId: 'gnina',
          label: 'GNINA 1.3',
          params: { ...defaultParamsForAdapter('gnina'), taskMode: 'pocket-scan', pocketRadius: 5.5 },
        },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'gnina_1', toPort: 'structure' }],
    };
    const encoded = await encodeGraphToHash(graph);
    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes[1].adapterId).toBe('gnina');
    expect(decoded!.nodes[1].params.taskMode).toBe('pocket-scan');
    expect(decoded!.nodes[1].params.pocketRadius).toBe(5.5);
  });

  it('splits holo PDB into receptor (+ catalytic ZN) and V2000 ligand SDF, excluding HOH', () => {
    const prep = extractHoloReceptorAndLigands(MINI_HOLO_PDB, { includeMetals: true });
    expect(prep.detectedLigandCodes).toEqual(['DHZ']);
    expect(prep.selectedLigand?.compId).toBe('DHZ');
    expect(prep.selectedLigand?.heavyAtomCount).toBe(4);
    expect(prep.keptMetals).toEqual(['ZN']);
    expect(prep.receptorPdb).toContain('HETATM   13 ZN');
    expect(prep.receptorPdb).not.toContain('DHZ');
    expect(prep.receptorPdb).not.toContain('HOH');
    expect(prep.selectedLigandSdf).toContain('V2000');
    expect(prep.selectedLigandSdf).toContain('$$$$');
    // Short C1-O1 (1.23 Å) should be perceived as a double bond (order 2)
    expect(prep.selectedLigandSdf).toMatch(/\n\s*1\s+3\s+2\s+0/);

    const noMetals = extractHoloReceptorAndLigands(MINI_HOLO_PDB, { includeMetals: false });
    expect(noMetals.keptMetals).toEqual([]);
    expect(noMetals.receptorPdb).not.toContain('ZN');
  });

  it('finds pocket residues within cutoff and re-injects minimized ligand coordinates cleanly', () => {
    const prep = extractHoloReceptorAndLigands(MINI_HOLO_PDB, { includeMetals: true });
    const pocket = findPocketResidues(prep.receptorPdb, prep.selectedLigand!.atoms, 6.0, 12);
    expect(pocket.length).toBeGreaterThan(0);
    expect(pocket[0].gninaKey).toMatch(/^A\/1[01]\/(?:HIS|ASP)$/);

    const shifted = prep.selectedLigand!.atoms.map((a) => ({
      x: a.x + 0.5,
      y: a.y - 0.25,
      z: a.z + 0.1,
    }));
    const rmsd = computePoseHeavyRmsd(prep.selectedLigand!.atoms, shifted);
    expect(rmsd).toBeCloseTo(Math.hypot(0.5, -0.25, 0.1), 4);

    const updatedPdb = applyMinimizedHoloLigandCoords(prep.normalizedPdb, prep.selectedLigand!, shifted);
    expect(updatedPdb).toContain('13.500');
    expect(updatedPdb).toContain('HETATM   13 ZN');
  });

  it('blocks apo structures without organic ligands (bio-guard) unless external ligandSdf is pasted', () => {
    const apoGraph: WorkflowGraph = {
      id: 'gnina-apo-check',
      name: 'GNINA Apo Check',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          label: 'Apo PDB',
          params: { ...defaultParamsForAdapter('pdb-input'), source: 'paste', pdbText: MINI_APO_PDB },
        },
        {
          id: 'gnina_1',
          adapterId: 'gnina',
          label: 'GNINA 1.3',
          params: { ...defaultParamsForAdapter('gnina') },
        },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'gnina_1', toPort: 'structure' }],
    };

    const apoIssues = validateBioCapabilities(apoGraph);
    expect(apoIssues.some((i) => i.code === 'gnina-no-organic-ligand' && i.severity === 'error')).toBe(true);

    // Providing an external SDF clears the missing-ligand error
    const extGraph: WorkflowGraph = {
      ...apoGraph,
      nodes: [
        apoGraph.nodes[0],
        {
          ...apoGraph.nodes[1],
          params: { ...apoGraph.nodes[1].params, ligandSdf: 'LIG\n\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$' },
        },
      ],
    };
    const extIssues = validateBioCapabilities(extGraph);
    expect(extIssues.some((i) => i.code === 'gnina-no-organic-ligand')).toBe(false);
  });

  it('compiles Node Studio input preview with detected ligand & pocket hotspot highlights and resolves gnina-contrib B-factor domain', () => {
    const holoStruct = makeStructArtifact(MINI_HOLO_PDB, '1CTT Mini');
    const graph: WorkflowGraph = {
      id: 'gnina-preview',
      name: 'GNINA Preview',
      nodes: [
        { id: 'pdb_1', adapterId: 'pdb-input', label: '1CTT Mini', params: defaultParamsForAdapter('pdb-input') },
        {
          id: 'gnina_1',
          adapterId: 'gnina',
          label: 'GNINA 1.3',
          params: { ...defaultParamsForAdapter('gnina'), taskMode: 'pocket-scan', pocketRadius: 6.0 },
        },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'gnina_1', toPort: 'structure' }],
    };

    const preview = compileNodeInputPreview(graph, 'gnina_1', { pdb_1: { structure: holoStruct } });
    expect(preview.availability).toBe('ready');
    expect(preview.payload.headline).toContain('Pocket Scan');
    expect(preview.payload.headline).toContain('DHZ');
    expect(preview.entities.includedLigands).toContain('DHZ');
    expect(preview.entities.includedLigands).toContain('ZN');

    const colorArg = resolveUncertaintyColorArgFromValues([0, 0.45, -0.15, 0.85], 'gnina-contrib');
    expect(colorArg.color).toBe('uncertainty');
    expect(colorArg.colorParams?.domain).toEqual([0.85, -0.85]);
  });

  it('supports flexible-ligand docking pocket tokens, multi-model PDB output, apo docking guard, and dock preview', () => {
    const prep = extractHoloReceptorAndLigands(MINI_HOLO_PDB, { includeMetals: true });
    const pocketKeys = parsePocketResidueTokens('A10, A11-12', prep.receptorPdb);
    expect(pocketKeys).toEqual(['A/10/HIS', 'A/11/ASP', 'A/12/ALA']);

    const pose1 = prep.selectedLigand!.atoms.map((a) => ({ x: a.x, y: a.y, z: a.z, el: a.element }));
    const pose2 = prep.selectedLigand!.atoms.map((a) => ({ x: a.x + 1.0, y: a.y, z: a.z, el: a.element }));
    const multiPdb = buildMultiPoseDockedPdb(prep.receptorPdb, [pose1, pose2], 'DHZ', 'L');
    expect(multiPdb).toContain('MODEL        1');
    expect(multiPdb).toContain('MODEL        2');
    expect(multiPdb).toContain('ENDMDL');
    expect(multiPdb).toContain('DHZ L');

    // Apo docking with CCD code requires pocketResidues
    const apoDockNoPocket: WorkflowGraph = {
      id: 'gnina-apo-dock-no-pocket',
      name: 'Apo Dock Missing Pocket',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          label: 'Apo PDB',
          params: { ...defaultParamsForAdapter('pdb-input'), mode: 'custom', pdbText: MINI_APO_PDB },
        },
        {
          id: 'gnina_1',
          adapterId: 'gnina',
          label: 'GNINA 1.3',
          params: { ...defaultParamsForAdapter('gnina'), taskMode: 'dock', dockLigandCcd: 'BTN' },
        },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'gnina_1', toPort: 'structure' }],
    };
    const missingPocketIssues = validateBioCapabilities(apoDockNoPocket);
    expect(missingPocketIssues.some((i) => i.code === 'gnina-dock-missing-pocket' && i.severity === 'error')).toBe(true);

    // Adding pocketResidues clears the apo docking error
    const apoDockWithPocket: WorkflowGraph = {
      ...apoDockNoPocket,
      nodes: [
        apoDockNoPocket.nodes[0],
        {
          ...apoDockNoPocket.nodes[1],
          params: { ...apoDockNoPocket.nodes[1].params, pocketResidues: 'A10, A11' },
        },
      ],
    };
    const okIssues = validateBioCapabilities(apoDockWithPocket);
    expect(okIssues.some((i) => i.severity === 'error')).toBe(false);

    // Node Studio input preview highlights explicit pocket residues and shows CCD ligand
    const apoStruct = makeStructArtifact(MINI_APO_PDB, 'Apo Mini');
    const preview = compileNodeInputPreview(apoDockWithPocket, 'gnina_1', { pdb_1: { structure: apoStruct } });
    expect(preview.availability).toBe('ready');
    expect(preview.payload.headline).toContain('Dock');
    expect(preview.payload.headline).toContain('BTN');
    expect(preview.budget.designedResidues).toBe(2);
    expect(preview.entities.includedLigands).toContain('BTN');
  });
});
