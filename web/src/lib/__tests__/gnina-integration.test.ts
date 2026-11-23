import { describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { filterCatalogCategories } from '$lib/engine/catalog';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { resolveLocalFoldEffectiveConfig } from '$lib/engine/conditioning';
import { compileNodeInputPreview } from '$lib/engine/input-preview';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import { encodeGraphToHash, decodeHashToGraph, normalizeWorkflowGraph } from '$lib/engine/share';
import { getModelCardForAdapter } from '$lib/engine/model-cards';
import {
  applyMinimizedHoloLigandCoords,
  buildMultiPoseDockedPdb,
  computePoseHeavyRmsd,
  extractHoloReceptorAndLigands,
  findPocketResidues,
  parsePocketResidueTokens,
} from '$lib/protein-ops/gnina-prep';
import { parsePdbBackbone, parseStructureText } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { resolveUncertaintyColorArgFromValues } from '$lib/viewer/bfactor-domain';
import { defaultParamsForAdapter, validateGraph } from '$lib/engine/graph-utils';
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

  it('Issue #89: decoded share-link workflow graph validates with zero errors across gnina_1..4, preserves expected warnings, and passes scheduler pre-run', async () => {
    const issue89Hash =
      '#g=pVRdb5swFP0r6O6lk24mbJMv3tKk3fKQalr7VIQixzYUDQzjQ9pU9b9PNk0DhWTT9mJZh-tzzz0-5hk0-EEAhTzsCSAUgM-Q5VKBb7BJIg0oD1sJPpD1wwMgiDIv7lWcKV1X4MOKuBSdFXE9s9KlWZlB_Mc7s26-PALCQWnxlPHyu2XimdoTUdd7ITm8IKx2N86Vof8IIQZQRtKzcsqOnCyvkwgQUqXj-gl8snCxBe8Lrs8KAYRGJ1qqn0ruLIVfl41CELmuk9iMtXAnZOECAm_q_F4pqeT1Rbnfbjeec7Xa3bRyY51obvXG2gJRnso9jxggRMcJUsNjoTSJuZZrIde5VMZB45NxyRiRFk_8Nk-lwyyTVFUSa35I0qT-ta9EXqrSdqrsZ3NtiS6a-p8v7_09RUmqNM8MhdHzdXPtbE0D5wPtzGpcjTXg8wvC57vt3cohn1i_hI2XtGOlueDpq0vUah_41DfjyF3VZSPqplR7kWcFL5U9LEZI6X-ZfxQ67Gd4RTsW2zjrFu2P7o2P7kEYIqjee3vjN2E_pv4E9h5Dt_YUuXPF6kejtFCVseEUxxN8lvp85HqduhnPU1ldOtkWDAM7nIhe6jMsZ39nwDBufR9GvndajWfuDa0uyRw_W6pIlab9mCdnddOB7vNujue23zcdsv-R4d3UlylOr6FzR6H5T_lB4Lk4c0MM2HEz627obBFiQFxCkbIQg-kcmYUYI6874lKGdOnZE1OckPnconMXJ2TZUhOcLkztbLnE6WJpdpTgfO6F4ctv';

    const decoded = await decodeHashToGraph(issue89Hash);
    expect(decoded).not.toBeNull();
    const graph = normalizeWorkflowGraph(decoded!, decoded!.id, decoded!.name, false);

    expect(validateGraph(graph)).toEqual([]);

    const issues = validateBioCapabilities(graph);
    expect(issues.filter((i) => i.severity === 'error')).toEqual([]);

    // Expected warnings remain intact
    const apoRefold = issues.find(
      (i) => i.nodeId === 'localfold_af2_1' && i.code === 'af3-apo-refold-note',
    );
    expect(apoRefold).toBeDefined();
    expect(apoRefold!.severity).toBe('warning');
    expect(apoRefold!.fix?.paramsPatch).toEqual({ ligandCcdCodes: 'ZN, DHZ' });

    const singleSeqMsa = issues.find(
      (i) => i.nodeId === 'localfold_af2_2' && i.code === 'natural-single-seq-msa',
    );
    expect(singleSeqMsa).toBeDefined();
    expect(singleSeqMsa!.severity).toBe('warning');
    expect(
      issues.some((i) => i.nodeId === 'localfold_af2_2' && i.code === 'af3-apo-refold-note'),
    ).toBe(false);

    // Node Studio previews for PDB Input #2 (partial) and GNINA 1.3 #4 (pending-upstream + cifText fallback)
    const pdb2Preview = compileNodeInputPreview(graph, 'pdb_input_1', {});
    expect(pdb2Preview.availability).toBe('partial');
    expect(pdb2Preview.entities.includedLigands).toEqual(['ZN', 'DHZ']);

    const gnina4Pending = compileNodeInputPreview(graph, 'gnina_4', {});
    expect(gnina4Pending.availability).toBe('pending-upstream');
    expect(gnina4Pending.issues.filter((i) => i.severity === 'error')).toEqual([]);

    const cifOnlyHolo: StructureArtifact = {
      ...makeStructArtifact(MINI_HOLO_PDB, 'CIF-only Holo'),
      pdbText: '',
      cifText: MINI_HOLO_PDB,
    };
    const gnina4CifPreview = compileNodeInputPreview(graph, 'gnina_4', {
      localfold_af2_2: { structure: cifOnlyHolo },
    });
    expect(gnina4CifPreview.availability).toBe('ready');
    expect(gnina4CifPreview.previewStructurePdb).toBe(MINI_HOLO_PDB);

    // ExecutionScheduler.run() pre-run validation passes on normalized issue #89 graph
    const scheduler = new ExecutionScheduler();
    const runRes = await scheduler.run(graph, () => {}, {
      signal: AbortSignal.abort(),
      returnPartialOnCancel: true,
    });
    expect(runRes.status).toBe('cancelled');
  });

  it('catches metal-only AF3, apo AF3, cropped-out pdb-input, and ligandCode mismatch pre-run and in ExecutionScheduler.run()', async () => {
    // 1. Metal-only AF3 (ligandCcdCodes: "ZN") -> GNINA triggers gnina-no-organic-ligand mentioning ZN
    const metalOnlyAf3: WorkflowGraph = {
      id: 'metal-only-af3-gnina',
      name: 'Metal Only AF3 -> GNINA',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFSRQ' } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AlphaFold 3 Metal', params: { model: 'af3', ligandCcdCodes: 'ZN' } },
        { id: 'gnina', adapterId: 'gnina', label: 'GNINA 1.3', params: { ...defaultParamsForAdapter('gnina') } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'fold', fromPort: 'structure', toNode: 'gnina', toPort: 'structure' },
      ],
    };
    const metalIssues = validateBioCapabilities(metalOnlyAf3);
    const metalErr = metalIssues.find((i) => i.code === 'gnina-no-organic-ligand' && i.severity === 'error');
    expect(metalErr).toBeDefined();
    expect(metalErr!.message).toContain('ZN');

    const scheduler = new ExecutionScheduler();
    await expect(
      scheduler.run(metalOnlyAf3, () => {}, {
        signal: AbortSignal.abort(),
        returnPartialOnCancel: true,
      }),
    ).rejects.toThrow(/only ZN/);

    // 2. Apo AF3 -> GNINA is rejected in scheduler.run() pre-run
    const apoAf3: WorkflowGraph = {
      ...metalOnlyAf3,
      nodes: [
        metalOnlyAf3.nodes[0],
        { id: 'fold', adapterId: 'localfold-af2', label: 'AlphaFold 3 Apo', params: { model: 'af3', ligandCcdCodes: '' } },
        metalOnlyAf3.nodes[2],
      ],
    };
    await expect(
      scheduler.run(apoAf3, () => {}, {
        signal: AbortSignal.abort(),
        returnPartialOnCancel: true,
      }),
    ).rejects.toThrow(/no organic small-molecule ligand/);

    // 3. Holo AF3 ("ZN, DHZ") -> GNINA with mismatched ligandCode ("ATP") warns pre-run
    const mismatchGraph: WorkflowGraph = {
      ...metalOnlyAf3,
      nodes: [
        metalOnlyAf3.nodes[0],
        { id: 'fold', adapterId: 'localfold-af2', label: 'AlphaFold 3 Holo', params: { model: 'af3', ligandCcdCodes: 'ZN, DHZ' } },
        { id: 'gnina', adapterId: 'gnina', label: 'GNINA 1.3', params: { ...defaultParamsForAdapter('gnina'), ligandCode: 'ATP' } },
      ],
    };
    const mismatchIssues = validateBioCapabilities(mismatchGraph);
    expect(mismatchIssues.some((i) => i.severity === 'error')).toBe(false);
    const mismatchWarn = mismatchIssues.find((i) => i.code === 'gnina-ligand-code-mismatch');
    expect(mismatchWarn).toBeDefined();
    expect(mismatchWarn!.message).toContain('ATP');
    expect(mismatchWarn!.message).toContain('DHZ');

    // 4. Cached holo pdbText with protein-only ("A") or metal-only ("A, A:ZN") cropSegments -> GNINA blocks
    const cachedCroppedApo: WorkflowGraph = {
      id: 'cached-crop-apo',
      name: 'Cached Crop Apo',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1CTT Custom',
          params: { mode: 'custom', pdbText: MINI_HOLO_PDB, cropSegments: 'A, A:ZN' },
        },
        { id: 'gnina', adapterId: 'gnina', label: 'GNINA 1.3', params: { ...defaultParamsForAdapter('gnina') } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'gnina', toPort: 'structure' }],
    };
    const cachedCropErr = validateBioCapabilities(cachedCroppedApo).find(
      (i) => i.code === 'gnina-no-organic-ligand' && i.severity === 'error',
    );
    expect(cachedCropErr).toBeDefined();
    expect(cachedCropErr!.message).toContain('ZN');

    const cachedProteinOnly: WorkflowGraph = {
      ...cachedCroppedApo,
      nodes: [
        {
          ...cachedCroppedApo.nodes[0],
          params: { mode: 'custom', pdbText: MINI_HOLO_PDB, cropSegments: 'A' },
        },
        cachedCroppedApo.nodes[1],
      ],
    };
    expect(
      validateBioCapabilities(cachedProteinOnly).some(
        (i) => i.code === 'gnina-no-organic-ligand' && i.severity === 'error',
      ),
    ).toBe(true);

    // 5. Unfetched pdb-id with metal-only cropSegments ("A, A:ZN") -> GNINA blocks pre-fetch, while empty cropSegments remains allowed
    const unfetchedMetalCrop: WorkflowGraph = {
      id: 'unfetched-metal-crop',
      name: 'Unfetched Metal Crop',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1CTT',
          params: { mode: 'pdb-id', pdbId: '1CTT', cropSegments: 'A, A:ZN' },
        },
        { id: 'gnina', adapterId: 'gnina', label: 'GNINA 1.3', params: { ...defaultParamsForAdapter('gnina') } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'gnina', toPort: 'structure' }],
    };
    const unfetchedMetalErr = validateBioCapabilities(unfetchedMetalCrop).find(
      (i) => i.code === 'gnina-no-organic-ligand' && i.severity === 'error',
    );
    expect(unfetchedMetalErr).toBeDefined();
    expect(unfetchedMetalErr!.message).toContain('ZN');

    const unfetchedEmptyCrop: WorkflowGraph = {
      ...unfetchedMetalCrop,
      nodes: [
        {
          ...unfetchedMetalCrop.nodes[0],
          params: { mode: 'pdb-id', pdbId: '1STP', cropSegments: '' },
        },
        unfetchedMetalCrop.nodes[1],
      ],
    };
    expect(
      validateBioCapabilities(unfetchedEmptyCrop).some((i) => i.severity === 'error'),
    ).toBe(false);
  });

  it('parses AF3 toPdb format (HETATM ZN + DHZ + CONECT) cleanly into receptor and V2000 SDF via extractHoloReceptorAndLigands', () => {
    // Format matching vendor/localfold/shared/af3/structure/pdb.js output
    const af3ToPdbText = [
      'ATOM      1  N   HIS A   1      10.000  10.000  10.000  1.00 88.50           N',
      'ATOM      2  CA  HIS A   1      11.458  10.000  10.000  1.00 88.50           C',
      'ATOM      3  C   HIS A   1      12.000  11.420  10.000  1.00 88.50           C',
      'ATOM      4  O   HIS A   1      11.250  12.400  10.000  1.00 88.50           O',
      'TER       5      HIS A   1',
      'HETATM    6  ZN  ZN  B   1      12.500  11.000  12.500  1.00 84.00          ZN',
      'TER       7      ZN  B   1',
      'HETATM    8  C1  DHZ C   1      13.000  11.000  14.800  1.00 82.00           C',
      'HETATM    9  C2  DHZ C   1      14.390  11.000  14.800  1.00 82.00           C',
      'HETATM   10  O1  DHZ C   1      12.300  11.000  13.790  1.00 82.00           O',
      'HETATM   11  N1  DHZ C   1      15.100  11.000  15.980  1.00 82.00           N',
      'CONECT    8    9   10',
      'CONECT    9    8   11',
      'CONECT   10    8',
      'CONECT   11    9',
      'END',
    ].join('\n');

    const parsed = parsePdbBackbone(af3ToPdbText);
    expect(parsed.nResidues).toBe(1);
    expect(parsed.diagnostics.hasLigands).toBe(true);
    expect(parsed.diagnostics.ligandNames).toEqual(['ZN', 'DHZ']);

    const prep = extractHoloReceptorAndLigands(af3ToPdbText, { includeMetals: true });
    expect(prep.detectedLigandCodes).toEqual(['DHZ']);
    expect(prep.keptMetals).toEqual(['ZN']);
    expect(prep.selectedLigand?.compId).toBe('DHZ');
    expect(prep.selectedLigand?.heavyAtomCount).toBe(4);
    expect(prep.receptorPdb).toContain('ZN  B');
    expect(prep.receptorPdb).not.toContain('DHZ');
    expect(prep.selectedLigandSdf).toContain('V2000');
    expect(prep.selectedLigandSdf).toContain('$$$$');
  });

  it('normalizes semicolon/punctuation ligandCcdCodes and blocks invalid or metal dockLigandCcd / AF3 CCDs pre-run', () => {
    // 1. resolveLocalFoldEffectiveConfig semicolon & punctuation handling
    expect(
      resolveLocalFoldEffectiveConfig({ modelKey: 'af3', params: { ligandCcdCodes: ' zn ; dhz ' } }).ligandCcdCodes,
    ).toEqual(['ZN', 'DHZ']);
    expect(
      resolveLocalFoldEffectiveConfig({ modelKey: 'af3', params: { ligandCcdCodes: 'ZN;DHZ' } }).ligandCcdCodes,
    ).toEqual(['ZN', 'DHZ']);
    expect(
      resolveLocalFoldEffectiveConfig({ modelKey: 'af3', params: { ligandCcdCodes: ' zn ; ' } }).ligandCcdCodes,
    ).toEqual(['ZN']);
    const emptyPunct = resolveLocalFoldEffectiveConfig({
      modelKey: 'af3',
      params: { ligandCcdCodes: ' , ; : ' },
    });
    expect(emptyPunct.ligandCcdCodes).toEqual([]);
    expect(emptyPunct.hasHoloContext).toBe(false);
    expect(
      resolveLocalFoldEffectiveConfig({ modelKey: 'af3', params: { ligandCcdCodes: ['ZN;DHZ', ' ; '] } })
        .ligandCcdCodes,
    ).toEqual(['ZN', 'DHZ']);

    // 2. AF3 with metal-only + semicolon (" zn ; ") -> GNINA rejects with gnina-no-organic-ligand (no ';' bypass)
    const metalSemicolonGraph: WorkflowGraph = {
      id: 'af3-zn-semicolon-gnina',
      name: 'AF3 Metal Semicolon -> GNINA',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFSRQ' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'af3', ligandCcdCodes: ' zn ; ' } },
        { id: 'gnina', adapterId: 'gnina', params: { ...defaultParamsForAdapter('gnina') } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'fold', fromPort: 'structure', toNode: 'gnina', toPort: 'structure' },
      ],
    };
    const metalSemiIssues = validateBioCapabilities(metalSemicolonGraph);
    expect(
      metalSemiIssues.some((i) => i.nodeId === 'gnina' && i.code === 'gnina-no-organic-ligand' && i.severity === 'error'),
    ).toBe(true);

    // 3. Invalid AF3 CCD code emits localfold-invalid-ligand-ccd
    const badAf3CcdGraph: WorkflowGraph = {
      ...metalSemicolonGraph,
      nodes: [
        metalSemicolonGraph.nodes[0],
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'af3', ligandCcdCodes: 'BAD_CODE' } },
        metalSemicolonGraph.nodes[2],
      ],
    };
    expect(
      validateBioCapabilities(badAf3CcdGraph).some(
        (i) => i.nodeId === 'fold' && i.code === 'localfold-invalid-ligand-ccd' && i.severity === 'error',
      ),
    ).toBe(true);

    // 4. GNINA dockLigandCcd rejects metal ion ('ZN') and invalid CCD syntax ('BAD_CODE')
    const metalDockGraph: WorkflowGraph = {
      id: 'gnina-metal-dock-ccd',
      name: 'GNINA Metal Dock CCD',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: MINI_APO_PDB } },
        {
          id: 'gnina',
          adapterId: 'gnina',
          params: { ...defaultParamsForAdapter('gnina'), taskMode: 'dock', dockLigandCcd: 'ZN', pocketResidues: 'A10' },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'gnina', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(metalDockGraph).some(
        (i) => i.nodeId === 'gnina' && i.code === 'gnina-non-organic-dock-ccd' && i.severity === 'error',
      ),
    ).toBe(true);

    const badDockCcdGraph: WorkflowGraph = {
      ...metalDockGraph,
      nodes: [
        metalDockGraph.nodes[0],
        {
          id: 'gnina',
          adapterId: 'gnina',
          params: { ...defaultParamsForAdapter('gnina'), taskMode: 'dock', dockLigandCcd: 'BAD_CODE', pocketResidues: 'A10' },
        },
      ],
    };
    expect(
      validateBioCapabilities(badDockCcdGraph).some(
        (i) => i.nodeId === 'gnina' && i.code === 'gnina-invalid-dock-ccd' && i.severity === 'error',
      ),
    ).toBe(true);
  });
});
