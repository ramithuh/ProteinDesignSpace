import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { beforeAll, describe, expect, it } from 'vitest';
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

// ---------------------------------------------------------------------------
// Issue #93: Raw-PDB bond-order perception, valence budgets & donor retention
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let gninaParseLigands: (text: string, format?: string) => any[];

beforeAll(async () => {
  const url = pathToFileURL(
    resolve(__dirname, '../../../static/vendor/gnina/shared/structure.js'),
  ).href;
  const mod = await import(/* @vite-ignore */ url);
  gninaParseLigands = mod.parseLigands;
});

interface ParsedSdfV2000 {
  atoms: Array<{ idx: number; element: string; charge: number }>;
  bonds: Array<{ a: number; b: number; order: number }>;
  valences: number[];
  bondOrderBetween: (i0: number, j0: number) => number | undefined;
}

function parseSdfV2000(sdf: string): ParsedSdfV2000 {
  const lines = sdf.split(/\r?\n/);
  const counts = lines[3];
  const nAtoms = parseInt(counts.slice(0, 3).trim(), 10);
  const nBonds = parseInt(counts.slice(3, 6).trim(), 10);
  const atoms: Array<{ idx: number; element: string; charge: number }> = [];
  for (let i = 0; i < nAtoms; i++) {
    const line = lines[4 + i];
    const element = line.slice(31, 34).trim();
    atoms.push({ idx: i, element, charge: 0 });
  }
  const bonds: Array<{ a: number; b: number; order: number }> = [];
  const valences = new Array<number>(nAtoms).fill(0);
  for (let i = 0; i < nBonds; i++) {
    const line = lines[4 + nAtoms + i];
    const a = parseInt(line.slice(0, 3).trim(), 10) - 1;
    const b = parseInt(line.slice(3, 6).trim(), 10) - 1;
    const order = parseInt(line.slice(6, 9).trim(), 10);
    bonds.push({ a, b, order });
    const v = order === 4 ? 1.5 : order;
    valences[a] += v;
    valences[b] += v;
  }
  for (let i = 4 + nAtoms + nBonds; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('M  CHG')) {
      const count = parseInt(line.slice(6, 9).trim(), 10);
      for (let k = 0; k < count; k++) {
        const idx1 = parseInt(line.slice(9 + k * 8, 13 + k * 8).trim(), 10) - 1;
        const chg = parseInt(line.slice(13 + k * 8, 17 + k * 8).trim(), 10);
        if (atoms[idx1]) atoms[idx1].charge = chg;
      }
    }
  }
  const bondOrderBetween = (i0: number, j0: number): number | undefined =>
    bonds.find((b) => (b.a === i0 && b.b === j0) || (b.a === j0 && b.b === i0))?.order;
  return { atoms, bonds, valences, bondOrderBetween };
}

function assertValidSdfValenceBudgets(sdf: string): ParsedSdfV2000 {
  const parsed = parseSdfV2000(sdf);
  for (let i = 0; i < parsed.atoms.length; i++) {
    const { element, charge } = parsed.atoms[i];
    const v = parsed.valences[i];
    if (element === 'C') expect(v).toBeLessThanOrEqual(Math.max(1, 4 - Math.abs(charge)) + 1e-6);
    else if (element === 'N') expect(v).toBeLessThanOrEqual(Math.max(1, 3 + charge) + 1e-6);
    else if (element === 'O') expect(v).toBeLessThanOrEqual(Math.max(0, 2 + charge) + 1e-6);
    else if (element === 'S') expect(v).toBeLessThanOrEqual(Math.max(2, 6 - Math.abs(charge)) + 1e-6);
    else if (element === 'P') expect(v).toBeLessThanOrEqual(Math.max(3, 5 - Math.abs(charge)) + 1e-6);
  }
  return parsed;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function countPolarHydrogensForAtom(mol: any, atomIdx: number): number {
  let n = 0;
  for (let i = 0; i < mol.centerAtoms.length; i++) {
    if (mol.centerAtoms[i].el === 'H' && mol.centerParents[i] === atomIdx) n++;
  }
  return n;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function countPolarHydrogensByEl(mol: any, el: string): number {
  let n = 0;
  for (let i = 0; i < mol.centerAtoms.length; i++) {
    if (mol.centerAtoms[i].el === 'H' && mol.atoms[mol.centerParents[i]]?.el === el) n++;
  }
  return n;
}

function stripConectRecords(pdbText: string): string {
  return pdbText
    .split(/\r?\n/)
    .filter((l) => !l.startsWith('CONECT'))
    .join('\n');
}

function formatSynthHetatm(
  serial: number,
  name: string,
  resName: string,
  x: number,
  y: number,
  z: number,
  el: string,
  charge = '',
): string {
  const ser = String(serial).padStart(5, ' ');
  const nm = name.length < 4 ? ` ${name.padEnd(3, ' ')}` : name.slice(0, 4);
  const res = resName.padStart(3, ' ').slice(0, 3);
  const c = (v: number) => v.toFixed(3).padStart(8, ' ');
  const elem = el.padStart(2, ' ');
  const chg = charge.padEnd(2, ' ').slice(0, 2);
  return `HETATM${ser} ${nm} ${res} A   1    ${c(x)}${c(y)}${c(z)}  1.00 20.00          ${elem}${chg}`;
}

// Verbatim official RCSB HETATM + CONECT records for 1STP (BTN, 16 heavy atoms)
const RCSB_1STP_BTN_PDB = [
  'HETATM  903  C11 BTN A 300       9.051   4.576 -14.459  1.00 21.57           C  ',
  'HETATM  904  O11 BTN A 300       8.147   5.373 -14.172  1.00 21.65           O  ',
  'HETATM  905  O12 BTN A 300       8.697   3.498 -15.132  1.00 22.67           O  ',
  'HETATM  906  C10 BTN A 300      10.321   4.413 -13.702  1.00 20.77           C  ',
  'HETATM  907  C9  BTN A 300      10.215   3.247 -12.658  1.00 20.09           C  ',
  'HETATM  908  C8  BTN A 300      11.487   2.972 -11.964  1.00 19.24           C  ',
  'HETATM  909  C7  BTN A 300      11.627   1.558 -11.402  1.00 18.33           C  ',
  'HETATM  910  C2  BTN A 300      10.923   1.086 -10.195  1.00 17.74           C  ',
  'HETATM  911  S1  BTN A 300      10.884   2.103  -8.662  1.00 16.59           S  ',
  'HETATM  912  C6  BTN A 300      10.299   0.620  -7.760  1.00 17.06           C  ',
  'HETATM  913  C5  BTN A 300      11.142  -0.543  -8.274  1.00 17.47           C  ',
  'HETATM  914  N1  BTN A 300      12.443  -0.631  -7.613  1.00 17.73           N  ',
  'HETATM  915  C3  BTN A 300      13.467  -0.443  -8.453  1.00 17.95           C  ',
  'HETATM  916  O3  BTN A 300      14.692  -0.446  -8.205  1.00 17.93           O  ',
  'HETATM  917  N2  BTN A 300      12.974  -0.237  -9.695  1.00 17.92           N  ',
  'HETATM  918  C4  BTN A 300      11.513  -0.265  -9.727  1.00 17.65           C  ',
  'CONECT  903  904  905  906                                                      ',
  'CONECT  904  903                                                                ',
  'CONECT  905  903                                                                ',
  'CONECT  906  903  907                                                           ',
  'CONECT  907  906  908                                                           ',
  'CONECT  908  907  909                                                           ',
  'CONECT  909  908  910                                                           ',
  'CONECT  910  909  911  918                                                      ',
  'CONECT  911  910  912                                                           ',
  'CONECT  912  911  913                                                           ',
  'CONECT  913  912  914  918                                                      ',
  'CONECT  914  913  915                                                           ',
  'CONECT  915  914  916  917                                                      ',
  'CONECT  916  915                                                                ',
  'CONECT  917  915  918                                                           ',
  'CONECT  918  910  913  917                                                      ',
  'END',
].join('\n');

// Verbatim official RCSB HETATM + CONECT records for 1RX2 (FOL, 32 heavy atoms)
const RCSB_1RX2_FOL_PDB = [
  'HETATM 1271  N1  FOL A 161      30.407  41.127  12.334  1.00  8.73           N  ',
  'HETATM 1272  C2  FOL A 161      31.183  39.960  12.347  1.00  9.79           C  ',
  'HETATM 1273  NA2 FOL A 161      30.747  38.763  12.656  1.00 15.94           N  ',
  'HETATM 1274  N3  FOL A 161      32.605  40.091  12.045  1.00  9.96           N  ',
  'HETATM 1275  C4  FOL A 161      33.168  41.270  11.746  1.00 13.25           C  ',
  'HETATM 1276  O4  FOL A 161      34.374  41.337  11.542  1.00 12.23           O  ',
  'HETATM 1277  C4A FOL A 161      32.264  42.476  11.702  1.00 11.59           C  ',
  'HETATM 1278  N5  FOL A 161      32.764  43.661  11.235  1.00 16.95           N  ',
  'HETATM 1279  C6  FOL A 161      31.895  44.651  11.140  1.00 12.00           C  ',
  'HETATM 1280  C7  FOL A 161      30.608  44.333  11.572  1.00 17.89           C  ',
  'HETATM 1281  N8  FOL A 161      30.128  43.253  12.021  1.00 13.00           N  ',
  'HETATM 1282  C8A FOL A 161      30.979  42.320  12.033  1.00 10.23           C  ',
  'HETATM 1283  C9  FOL A 161      32.328  46.002  10.646  1.00 12.36           C  ',
  'HETATM 1284  N10 FOL A 161      33.705  45.982  10.049  1.00  5.22           N  ',
  'HETATM 1285  C11 FOL A 161      34.306  44.202   6.323  1.00 14.57           C  ',
  'HETATM 1286  C12 FOL A 161      35.321  44.853   7.040  1.00 17.61           C  ',
  'HETATM 1287  C13 FOL A 161      35.083  45.433   8.285  1.00 18.37           C  ',
  'HETATM 1288  C14 FOL A 161      33.813  45.401   8.851  1.00  4.90           C  ',
  'HETATM 1289  C15 FOL A 161      32.788  44.778   8.245  1.00  5.27           C  ',
  'HETATM 1290  C16 FOL A 161      33.054  44.162   6.947  1.00  9.61           C  ',
  'HETATM 1291  C   FOL A 161      34.568  43.576   5.020  1.00 24.98           C  ',
  'HETATM 1292  O   FOL A 161      35.643  43.806   4.468  1.00 23.33           O  ',
  'HETATM 1293  N   FOL A 161      33.743  42.614   4.530  1.00 24.29           N  ',
  'HETATM 1294  CA  FOL A 161      33.724  41.900   3.170  1.00 27.12           C  ',
  'HETATM 1295  CB  FOL A 161      34.090  40.359   3.384  1.00 33.48           C  ',
  'HETATM 1296  CG  FOL A 161      35.338  39.693   2.575  1.00 51.31           C  ',
  'HETATM 1297  CD  FOL A 161      35.035  38.179   2.282  1.00 73.27           C  ',
  'HETATM 1298  OE1 FOL A 161      33.833  38.054   1.926  1.00 81.18           O  ',
  'HETATM 1299  OE2 FOL A 161      35.859  37.227   2.389  1.00 81.20           O  ',
  'HETATM 1300  CT  FOL A 161      32.282  41.861   2.758  1.00 19.52           C  ',
  'HETATM 1301  O1  FOL A 161      31.411  41.707   3.598  1.00 18.61           O  ',
  'HETATM 1302  O2  FOL A 161      32.016  41.926   1.554  1.00 17.81           O  ',
  'CONECT 1271 1272 1282                                                           ',
  'CONECT 1272 1271 1273 1274                                                      ',
  'CONECT 1273 1272                                                                ',
  'CONECT 1274 1272 1275                                                           ',
  'CONECT 1275 1274 1276 1277                                                      ',
  'CONECT 1276 1275                                                                ',
  'CONECT 1277 1275 1278 1282                                                      ',
  'CONECT 1278 1277 1279                                                           ',
  'CONECT 1279 1278 1280 1283                                                      ',
  'CONECT 1280 1279 1281                                                           ',
  'CONECT 1281 1280 1282                                                           ',
  'CONECT 1282 1271 1277 1281                                                      ',
  'CONECT 1283 1279 1284                                                           ',
  'CONECT 1284 1283 1288                                                           ',
  'CONECT 1285 1286 1290 1291                                                      ',
  'CONECT 1286 1285 1287                                                           ',
  'CONECT 1287 1286 1288                                                           ',
  'CONECT 1288 1284 1287 1289                                                      ',
  'CONECT 1289 1288 1290                                                           ',
  'CONECT 1290 1285 1289                                                           ',
  'CONECT 1291 1285 1292 1293                                                      ',
  'CONECT 1292 1291                                                                ',
  'CONECT 1293 1291 1294                                                           ',
  'CONECT 1294 1293 1295 1300                                                      ',
  'CONECT 1295 1294 1296                                                           ',
  'CONECT 1296 1295 1297                                                           ',
  'CONECT 1297 1296 1298 1299                                                      ',
  'CONECT 1298 1297                                                                ',
  'CONECT 1299 1297                                                                ',
  'CONECT 1300 1294 1301 1302                                                      ',
  'CONECT 1301 1300                                                                ',
  'CONECT 1302 1300                                                                ',
  'END',
].join('\n');

// Verbatim official RCSB HETATM + CONECT records for 1HSG (MK1, 45 heavy atoms)
const RCSB_1HSG_MK1_PDB = [
  'HETATM 1517  N1  MK1 B 902       9.280  23.763   3.004  1.00 28.25           N  ',
  'HETATM 1518  C1  MK1 B 902       9.498  23.983   4.459  1.00 30.30           C  ',
  'HETATM 1519  C2  MK1 B 902      10.591  24.905   4.962  1.00 27.27           C  ',
  'HETATM 1520  C3  MK1 B 902      10.591  24.864   6.466  1.00 28.85           C  ',
  'HETATM 1521  O1  MK1 B 902      10.937  23.849   7.057  1.00 29.59           O  ',
  'HETATM 1522  N2  MK1 B 902      10.193  25.953   7.094  1.00 22.29           N  ',
  'HETATM 1523  C4  MK1 B 902      10.145  26.250   8.490  1.00 23.47           C  ',
  'HETATM 1524  C5  MK1 B 902       9.379  27.577   8.641  1.00 27.66           C  ',
  'HETATM 1525  C6  MK1 B 902      11.398  26.347   9.074  1.00 21.71           C  ',
  'HETATM 1526  C7  MK1 B 902       9.364  25.283   9.268  1.00 22.75           C  ',
  'HETATM 1527  N3  MK1 B 902      11.819  24.282   4.355  1.00 28.91           N  ',
  'HETATM 1528  C8  MK1 B 902      11.753  23.776   2.961  1.00 26.24           C  ',
  'HETATM 1529  C9  MK1 B 902      10.440  23.182   2.493  1.00 27.47           C  ',
  'HETATM 1530  C10 MK1 B 902      13.083  24.963   4.552  1.00 20.86           C  ',
  'HETATM 1531  C11 MK1 B 902      14.203  24.064   5.078  1.00 21.68           C  ',
  'HETATM 1532  O2  MK1 B 902      15.242  24.884   4.634  1.00 15.87           O  ',
  'HETATM 1533  C12 MK1 B 902      14.440  23.761   6.569  1.00 21.49           C  ',
  'HETATM 1534  C13 MK1 B 902      15.573  22.821   7.005  1.00 26.89           C  ',
  'HETATM 1535  C14 MK1 B 902      15.644  22.664   8.534  1.00 28.67           C  ',
  'HETATM 1536  C15 MK1 B 902      16.733  21.750   8.961  1.00 26.89           C  ',
  'HETATM 1537  C16 MK1 B 902      18.058  21.916   8.553  1.00 29.22           C  ',
  'HETATM 1538  C17 MK1 B 902      19.037  21.016   8.947  1.00 29.22           C  ',
  'HETATM 1539  C18 MK1 B 902      18.673  19.939   9.758  1.00 30.97           C  ',
  'HETATM 1540  C19 MK1 B 902      17.347  19.773  10.176  1.00 29.25           C  ',
  'HETATM 1541  C20 MK1 B 902      16.374  20.687   9.772  1.00 29.96           C  ',
  'HETATM 1542  C21 MK1 B 902      15.447  21.440   6.373  1.00 29.35           C  ',
  'HETATM 1543  O3  MK1 B 902      14.367  20.831   6.397  1.00 32.66           O  ',
  'HETATM 1544  N4  MK1 B 902      16.583  20.913   5.924  1.00 31.19           N  ',
  'HETATM 1545  C22 MK1 B 902      16.692  19.500   5.604  1.00 29.22           C  ',
  'HETATM 1546  C23 MK1 B 902      18.067  18.945   5.936  1.00 28.82           C  ',
  'HETATM 1547  O4  MK1 B 902      19.061  19.938   5.729  1.00 28.32           O  ',
  'HETATM 1548  C24 MK1 B 902      18.226  17.726   5.057  1.00 32.05           C  ',
  'HETATM 1549  C25 MK1 B 902      17.476  17.904   3.760  1.00 31.29           C  ',
  'HETATM 1550  C26 MK1 B 902      17.500  17.363   2.496  1.00 32.00           C  ',
  'HETATM 1551  C27 MK1 B 902      16.613  17.872   1.541  1.00 28.00           C  ',
  'HETATM 1552  C28 MK1 B 902      15.722  18.906   1.865  1.00 29.01           C  ',
  'HETATM 1553  C29 MK1 B 902      15.683  19.479   3.129  1.00 27.70           C  ',
  'HETATM 1554  C30 MK1 B 902      16.504  19.061   4.128  1.00 31.86           C  ',
  'HETATM 1555  C31 MK1 B 902       8.033  23.100   2.604  1.00 36.25           C  ',
  'HETATM 1556  C32 MK1 B 902       6.666  23.739   2.876  1.00 42.75           C  ',
  'HETATM 1557  C33 MK1 B 902       6.158  24.808   2.124  1.00 47.41           C  ',
  'HETATM 1558  N5  MK1 B 902       4.911  25.430   2.300  1.00 51.38           N  ',
  'HETATM 1559  C34 MK1 B 902       4.207  24.839   3.348  1.00 50.60           C  ',
  'HETATM 1560  C35 MK1 B 902       4.654  23.774   4.136  1.00 49.34           C  ',
  'HETATM 1561  C36 MK1 B 902       5.905  23.211   3.897  1.00 44.71           C  ',
  'CONECT 1517 1518 1529 1555                                                      ',
  'CONECT 1518 1517 1519                                                           ',
  'CONECT 1519 1518 1520 1527                                                      ',
  'CONECT 1520 1519 1521 1522                                                      ',
  'CONECT 1521 1520                                                                ',
  'CONECT 1522 1520 1523                                                           ',
  'CONECT 1523 1522 1524 1525 1526                                                 ',
  'CONECT 1524 1523                                                                ',
  'CONECT 1525 1523                                                                ',
  'CONECT 1526 1523                                                                ',
  'CONECT 1527 1519 1528 1530                                                      ',
  'CONECT 1528 1527 1529                                                           ',
  'CONECT 1529 1517 1528                                                           ',
  'CONECT 1530 1527 1531                                                           ',
  'CONECT 1531 1530 1532 1533                                                      ',
  'CONECT 1532 1531                                                                ',
  'CONECT 1533 1531 1534                                                           ',
  'CONECT 1534 1533 1535 1542                                                      ',
  'CONECT 1535 1534 1536                                                           ',
  'CONECT 1536 1535 1537 1541                                                      ',
  'CONECT 1537 1536 1538                                                           ',
  'CONECT 1538 1537 1539                                                           ',
  'CONECT 1539 1538 1540                                                           ',
  'CONECT 1540 1539 1541                                                           ',
  'CONECT 1541 1536 1540                                                           ',
  'CONECT 1542 1534 1543 1544                                                      ',
  'CONECT 1543 1542                                                                ',
  'CONECT 1544 1542 1545                                                           ',
  'CONECT 1545 1544 1546 1554                                                      ',
  'CONECT 1546 1545 1547 1548                                                      ',
  'CONECT 1547 1546                                                                ',
  'CONECT 1548 1546 1549                                                           ',
  'CONECT 1549 1548 1550 1554                                                      ',
  'CONECT 1550 1549 1551                                                           ',
  'CONECT 1551 1550 1552                                                           ',
  'CONECT 1552 1551 1553                                                           ',
  'CONECT 1553 1552 1554                                                           ',
  'CONECT 1554 1545 1549 1553                                                      ',
  'CONECT 1555 1517 1556                                                           ',
  'CONECT 1556 1555 1557 1561                                                      ',
  'CONECT 1557 1556 1558                                                           ',
  'CONECT 1558 1557 1559                                                           ',
  'CONECT 1559 1558 1560                                                           ',
  'CONECT 1560 1559 1561                                                           ',
  'CONECT 1561 1556 1560                                                           ',
  'END',
].join('\n');

describe('Issue #93: GNINA raw-PDB bond-order heuristic, valence budgets & donor retention', () => {
  it('1STP/BTN: enforces ureido C3 valence=4 and retains both N1/N2 N–H donors (with and without CONECT)', () => {
    for (const pdb of [RCSB_1STP_BTN_PDB, stripConectRecords(RCSB_1STP_BTN_PDB)]) {
      const prep = extractHoloReceptorAndLigands(pdb, { ligandCode: 'BTN' });
      expect(prep.selectedLigand?.heavyAtomCount).toBe(16);
      const sdf = prep.selectedLigandSdf!;
      const parsed = assertValidSdfValenceBudgets(sdf);
      // C11 (idx 0) = 4, N1 (idx 11) = 2, C3 (idx 12) = 4, N2 (idx 14) = 2
      expect(parsed.valences[0]).toBe(4);
      expect(parsed.valences[11]).toBe(2);
      expect(parsed.valences[12]).toBe(4);
      expect(parsed.valences[14]).toBe(2);
      expect(parsed.bondOrderBetween(12, 13)).toBe(2); // C3=O3
      expect(parsed.bondOrderBetween(11, 12)).toBe(1); // N1-C3
      expect(parsed.bondOrderBetween(12, 14)).toBe(1); // C3-N2

      const [mol] = gninaParseLigands(sdf, 'sdf');
      const donorNs = mol.atoms.filter((a: { el: string; donor: boolean }) => a.el === 'N' && a.donor);
      expect(donorNs.map((a: { index: number }) => a.index)).toEqual([11, 14]);
      expect(countPolarHydrogensByEl(mol, 'N')).toBe(2);
      expect(countPolarHydrogensByEl(mol, 'O')).toBe(1);
      expect(mol.atoms.filter((a: { el: string; channel: number }) => a.el === 'C' && (a.channel === 16 || a.channel === 17))).toHaveLength(0);
      expect(mol.rotors).toBe(5);
    }
  });

  it('1RX2/FOL: avoids hypervalent C4/C8A/C14/amide C, keeps C2–NA2 single (2 H), and detects 1.461 Å benzene aromaticity', () => {
    for (const pdb of [RCSB_1RX2_FOL_PDB, stripConectRecords(RCSB_1RX2_FOL_PDB)]) {
      const prep = extractHoloReceptorAndLigands(pdb, { ligandCode: 'FOL' });
      expect(prep.selectedLigand?.heavyAtomCount).toBe(32);
      const sdf = prep.selectedLigandSdf!;
      const parsed = assertValidSdfValenceBudgets(sdf);
      // C2 (idx 1) - NA2 (idx 2) must stay single bond despite 1.311 Å length
      expect(parsed.bondOrderBetween(1, 2)).toBe(1);
      // Amide C (idx 20) - N (idx 22) must stay single bond
      expect(parsed.bondOrderBetween(20, 22)).toBe(1);
      // Both pyrazine (C4A..C8A) and benzene (C11..C16, max 1.461 Å) are marked order 4
      const aromaticBonds = parsed.bonds.filter((b) => b.order === 4);
      expect(aromaticBonds).toHaveLength(12);

      const [mol] = gninaParseLigands(sdf, 'sdf');
      // N1 (0), NA2 (2), N3 (3: C2-N3 is 1.460 Å in 1RX2 raw X-ray vs 1.30 Å in FOL_ideal), N10 (13), amide N (22)
      const donorNs = mol.atoms.filter((a: { el: string; donor: boolean }) => a.el === 'N' && a.donor);
      expect(donorNs.map((a: { index: number }) => a.index)).toEqual([0, 2, 3, 13, 22]);
      expect(countPolarHydrogensForAtom(mol, 2)).toBe(2); // NA2 -NH2
      expect(countPolarHydrogensForAtom(mol, 0)).toBe(1); // N1
      expect(countPolarHydrogensForAtom(mol, 13)).toBe(1); // N10
      expect(countPolarHydrogensForAtom(mol, 22)).toBe(1); // amide N
      expect(countPolarHydrogensByEl(mol, 'N')).toBe(6);
      const aromCarbons = mol.atoms.filter(
        (a: { el: string; channel: number }) => a.el === 'C' && (a.channel === 16 || a.channel === 17),
      );
      expect(aromCarbons).toHaveLength(10); // 4 pyrazine C + 6 benzene C
    }
  });

  it('1HSG/MK1: preserves amide C3–N2 & C21–N4 single bonds, N2/N4 donors, and phenyl + pyridine aromaticity', () => {
    for (const pdb of [RCSB_1HSG_MK1_PDB, stripConectRecords(RCSB_1HSG_MK1_PDB)]) {
      const prep = extractHoloReceptorAndLigands(pdb, { ligandCode: 'MK1' });
      expect(prep.selectedLigand?.heavyAtomCount).toBe(45);
      const sdf = prep.selectedLigandSdf!;
      const parsed = assertValidSdfValenceBudgets(sdf);
      // C3 (idx 3) - N2 (idx 5) and C21 (idx 25) - N4 (idx 27) stay order 1
      expect(parsed.bondOrderBetween(3, 5)).toBe(1);
      expect(parsed.bondOrderBetween(25, 27)).toBe(1);

      const [mol] = gninaParseLigands(sdf, 'sdf');
      const donorNs = mol.atoms.filter((a: { el: string; donor: boolean }) => a.el === 'N' && a.donor);
      expect(donorNs.map((a: { index: number }) => a.index)).toEqual([5, 27]);
      expect(countPolarHydrogensByEl(mol, 'N')).toBe(2);
      expect(countPolarHydrogensByEl(mol, 'O')).toBe(2);
      const aromCarbons = mol.atoms.filter(
        (a: { el: string; channel: number }) => a.el === 'C' && (a.channel === 16 || a.channel === 17),
      );
      expect(aromCarbons).toHaveLength(11); // 6 phenyl + 5 pyridine
    }
  });

  it('distorted X-ray planar 6-ring (1.48 Å) is aromatic while 1.52 Å planar and 1.47 Å non-planar rings stay non-aromatic', () => {
    // 1) Planar 6-ring with 1.48 Å bonds (within 1.49 Å ceiling)
    const planar148 = [
      [0.0, 0.0, 0.0],
      [1.39, 0.0, 0.0],
      [2.085, 1.204, 0.0],
      [1.39, 2.408, 0.0],
      [0.0, 2.408, 0.0],
      // Midpoint y = 1.204, x = -sqrt(1.48^2 - 1.204^2) = -0.8607 -> edges (4,5) & (5,0) = 1.480 Å
      [-0.86, 1.204, 0.0],
    ];
    const dist148Pdb = [
      ...planar148.map(([x, y, z], i) => formatSynthHetatm(i + 1, `C${i + 1}`, 'AR6', x, y, z, 'C')),
      'END',
    ].join('\n');
    const prep148 = extractHoloReceptorAndLigands(dist148Pdb, { ligandCode: 'AR6' });
    const parsed148 = assertValidSdfValenceBudgets(prep148.selectedLigandSdf!);
    expect(parsed148.bonds.filter((b) => b.order === 4)).toHaveLength(6);
    const [mol148] = gninaParseLigands(prep148.selectedLigandSdf!, 'sdf');
    expect(mol148.atoms.filter((a: { el: string; channel: number }) => a.el === 'C' && a.channel === 16)).toHaveLength(6);

    // 2) Planar cyclohexane (1.52 Å bonds) -> NOT aromatic
    const r152 = 1.52;
    const planar152Pdb = [
      ...Array.from({ length: 6 }, (_, k) => {
        const ang = (k * Math.PI) / 3;
        return formatSynthHetatm(k + 1, `C${k + 1}`, 'CY6', r152 * Math.cos(ang), r152 * Math.sin(ang), 0, 'C');
      }),
      'END',
    ].join('\n');
    const prep152 = extractHoloReceptorAndLigands(planar152Pdb, { ligandCode: 'CY6' });
    const parsed152 = assertValidSdfValenceBudgets(prep152.selectedLigandSdf!);
    expect(parsed152.bonds.filter((b) => b.order === 4)).toHaveLength(0);

    // 3) Non-planar puckered 6-ring with <= 1.47 Å bonds (dPlane > 0.18 Å) -> NOT aromatic
    const puckeredCoords: Array<[number, number, number]> = [
      [1.39, 0.0, 0.0],
      [0.695, 1.204, 0.0],
      [-0.695, 1.204, 0.0],
      [-1.39, 0.0, 0.42],
      [-0.695, -1.204, 0.0],
      [0.695, -1.204, 0.0],
    ];
    const puckeredPdb = [
      ...puckeredCoords.map(([x, y, z], i) => formatSynthHetatm(i + 1, `C${i + 1}`, 'PUK', x, y, z, 'C')),
      'END',
    ].join('\n');
    const prepPuk = extractHoloReceptorAndLigands(puckeredPdb, { ligandCode: 'PUK' });
    const parsedPuk = assertValidSdfValenceBudgets(prepPuk.selectedLigandSdf!);
    expect(parsedPuk.bonds.filter((b) => b.order === 4)).toHaveLength(0);
  });

  it('2-pyridone (1.45 Å planar ring with exocyclic C=O) is NOT falsely aromatized', () => {
    const r = 1.45;
    const coords: Array<[number, number, number]> = Array.from({ length: 6 }, (_, k) => {
      const ang = (k * Math.PI) / 3;
      return [r * Math.cos(ang), r * Math.sin(ang), 0];
    });
    const pyridonePdb = [
      formatSynthHetatm(1, 'N1', 'PYD', coords[0][0], coords[0][1], 0, 'N'),
      formatSynthHetatm(2, 'C2', 'PYD', coords[1][0], coords[1][1], 0, 'C'),
      formatSynthHetatm(3, 'C3', 'PYD', coords[2][0], coords[2][1], 0, 'C'),
      formatSynthHetatm(4, 'C4', 'PYD', coords[3][0], coords[3][1], 0, 'C'),
      formatSynthHetatm(5, 'C5', 'PYD', coords[4][0], coords[4][1], 0, 'C'),
      formatSynthHetatm(6, 'C6', 'PYD', coords[5][0], coords[5][1], 0, 'C'),
      // Exocyclic C2=O2 at 1.23 Å radially outward from C2 (60 deg)
      formatSynthHetatm(7, 'O2', 'PYD', (r + 1.23) * Math.cos(Math.PI / 3), (r + 1.23) * Math.sin(Math.PI / 3), 0, 'O'),
      'END',
    ].join('\n');
    const prep = extractHoloReceptorAndLigands(pyridonePdb, { ligandCode: 'PYD' });
    const parsed = assertValidSdfValenceBudgets(prep.selectedLigandSdf!);
    expect(parsed.bondOrderBetween(1, 6)).toBe(2); // C2=O2
    expect(parsed.bonds.filter((b) => b.order === 4)).toHaveLength(0);
  });

  it('perceives true imine (C=N, 1.28 Å), true nitrile (C#N, 1.16 Å), and protects formamide C–N', () => {
    // 1) True terminal imine R2C=NH: C(CH3)2=NH with C=N = 1.28 Å
    const iminePdb = [
      formatSynthHetatm(1, 'C1', 'IMN', 0.0, 0.0, 0.0, 'C'),
      formatSynthHetatm(2, 'C2', 'IMN', -0.75, 1.3, 0.0, 'C'),
      formatSynthHetatm(3, 'C3', 'IMN', -0.75, -1.3, 0.0, 'C'),
      formatSynthHetatm(4, 'N1', 'IMN', 1.28, 0.0, 0.0, 'N'),
      'END',
    ].join('\n');
    const prepImn = extractHoloReceptorAndLigands(iminePdb, { ligandCode: 'IMN' });
    const parsedImn = assertValidSdfValenceBudgets(prepImn.selectedLigandSdf!);
    expect(parsedImn.bondOrderBetween(0, 3)).toBe(2);
    const [molImn] = gninaParseLigands(prepImn.selectedLigandSdf!, 'sdf');
    expect(molImn.atoms[3].donor).toBe(true); // =NH has 1 H
    expect(countPolarHydrogensForAtom(molImn, 3)).toBe(1);

    // Internal imine R2C=N-CH3 (1.28 Å): 0 N–H donors
    const intIminePdb = [
      formatSynthHetatm(1, 'C1', 'IMI', 0.0, 0.0, 0.0, 'C'),
      formatSynthHetatm(2, 'C2', 'IMI', -0.75, 1.3, 0.0, 'C'),
      formatSynthHetatm(3, 'C3', 'IMI', -0.75, -1.3, 0.0, 'C'),
      formatSynthHetatm(4, 'N1', 'IMI', 1.28, 0.0, 0.0, 'N'),
      formatSynthHetatm(5, 'C4', 'IMI', 2.0, 1.25, 0.0, 'C'),
      'END',
    ].join('\n');
    const prepIntImn = extractHoloReceptorAndLigands(intIminePdb, { ligandCode: 'IMI' });
    const parsedIntImn = assertValidSdfValenceBudgets(prepIntImn.selectedLigandSdf!);
    expect(parsedIntImn.bondOrderBetween(0, 3)).toBe(2);
    const [molIntImn] = gninaParseLigands(prepIntImn.selectedLigandSdf!, 'sdf');
    expect(countPolarHydrogensByEl(molIntImn, 'N')).toBe(0);

    // 2) True nitrile CH3-C#N (1.16 Å)
    const nitrilePdb = [
      formatSynthHetatm(1, 'C1', 'NIT', -1.46, 0.0, 0.0, 'C'),
      formatSynthHetatm(2, 'C2', 'NIT', 0.0, 0.0, 0.0, 'C'),
      formatSynthHetatm(3, 'N1', 'NIT', 1.16, 0.0, 0.0, 'N'),
      'END',
    ].join('\n');
    const prepNit = extractHoloReceptorAndLigands(nitrilePdb, { ligandCode: 'NIT' });
    const parsedNit = assertValidSdfValenceBudgets(prepNit.selectedLigandSdf!);
    expect(parsedNit.bondOrderBetween(1, 2)).toBe(3);
    const [molNit] = gninaParseLigands(prepNit.selectedLigandSdf!, 'sdf');
    expect(countPolarHydrogensByEl(molNit, 'N')).toBe(0);
    expect(molNit.atoms[2].acceptor).toBe(true);

    // 3) Formamide H-C(=O)-NH-CH3 (implicit H on C, C=O 1.23 Å, C-N 1.34 Å) -> C-N stays order 1
    const formamidePdb = [
      formatSynthHetatm(1, 'C1', 'FRM', 0.0, 0.0, 0.0, 'C'),
      formatSynthHetatm(2, 'O1', 'FRM', 0.0, 1.23, 0.0, 'O'),
      formatSynthHetatm(3, 'N1', 'FRM', 1.16, -0.67, 0.0, 'N'),
      formatSynthHetatm(4, 'C2', 'FRM', 2.42, 0.05, 0.0, 'C'),
      'END',
    ].join('\n');
    const prepFrm = extractHoloReceptorAndLigands(formamidePdb, { ligandCode: 'FRM' });
    const parsedFrm = assertValidSdfValenceBudgets(prepFrm.selectedLigandSdf!);
    expect(parsedFrm.bondOrderBetween(0, 1)).toBe(2); // C=O
    expect(parsedFrm.bondOrderBetween(0, 2)).toBe(1); // C-N protected
    const [molFrm] = gninaParseLigands(prepFrm.selectedLigandSdf!, 'sdf');
    expect(countPolarHydrogensByEl(molFrm, 'N')).toBe(1);
  });

  it('prioritizes neutral O over shorter -1-charged O in carboxylates and handles sulfonamides cleanly', () => {
    // Acetate where O2 (charge 1-) is 1.24 Å and O1 (charge 0) is 1.26 Å
    const carboxylatePdb = [
      formatSynthHetatm(1, 'C1', 'CBN', -1.5, 0.0, 0.0, 'C'),
      formatSynthHetatm(2, 'C2', 'CBN', 0.0, 0.0, 0.0, 'C'),
      formatSynthHetatm(3, 'O1', 'CBN', 0.63, 1.09, 0.0, 'O'),
      formatSynthHetatm(4, 'O2', 'CBN', 0.62, -1.07, 0.0, 'O', '1-'),
      'END',
    ].join('\n');
    const prepAce = extractHoloReceptorAndLigands(carboxylatePdb, { ligandCode: 'CBN' });
    const parsedAce = assertValidSdfValenceBudgets(prepAce.selectedLigandSdf!);
    expect(parsedAce.atoms[3].charge).toBe(-1);
    expect(parsedAce.bondOrderBetween(1, 2)).toBe(2); // C2=O1 (neutral)
    expect(parsedAce.bondOrderBetween(1, 3)).toBe(1); // C2-O2(-)
    const [molAce] = gninaParseLigands(prepAce.selectedLigandSdf!, 'sdf');
    expect(countPolarHydrogensByEl(molAce, 'O')).toBe(0);
    expect(molAce.atoms[2].acceptor).toBe(true);
    expect(molAce.atoms[3].acceptor).toBe(true);

    // Sulfonamide CH3-SO2-NH2: S valence = 6, two S=O order 2, N retains 2 N–H
    const sulfonamidePdb = [
      formatSynthHetatm(1, 'C1', 'SFA', -1.78, 0.0, 0.0, 'C'),
      formatSynthHetatm(2, 'S1', 'SFA', 0.0, 0.0, 0.0, 'S'),
      formatSynthHetatm(3, 'O1', 'SFA', 0.5, 1.35, 0.0, 'O'),
      formatSynthHetatm(4, 'O2', 'SFA', 0.5, -1.35, 0.0, 'O'),
      formatSynthHetatm(5, 'N1', 'SFA', 0.9, 0.0, 1.38, 'N'),
      'END',
    ].join('\n');
    const prepSfa = extractHoloReceptorAndLigands(sulfonamidePdb, { ligandCode: 'SFA' });
    const parsedSfa = assertValidSdfValenceBudgets(prepSfa.selectedLigandSdf!);
    expect(parsedSfa.valences[1]).toBe(6);
    expect(parsedSfa.bondOrderBetween(1, 2)).toBe(2);
    expect(parsedSfa.bondOrderBetween(1, 3)).toBe(2);
    expect(parsedSfa.bondOrderBetween(1, 4)).toBe(1);
    const [molSfa] = gninaParseLigands(prepSfa.selectedLigandSdf!, 'sdf');
    expect(countPolarHydrogensForAtom(molSfa, 4)).toBe(2);
  });

  it('multi-N resonance: protects 2-aminopyrimidine -NH2 and neutral guanidine while allowing 1+ guanidinium', () => {
    // 2-aminopyrimidine with 1.31 Å exocyclic C2-NH2
    const r = 1.38;
    const ringCoords: Array<[number, number, number]> = Array.from({ length: 6 }, (_, k) => {
      const ang = (k * Math.PI) / 3;
      return [r * Math.cos(ang), r * Math.sin(ang), 0];
    });
    const aminoPyrimPdb = [
      formatSynthHetatm(1, 'N1', '2AP', ringCoords[0][0], ringCoords[0][1], 0, 'N'),
      formatSynthHetatm(2, 'C2', '2AP', ringCoords[1][0], ringCoords[1][1], 0, 'C'),
      formatSynthHetatm(3, 'N3', '2AP', ringCoords[2][0], ringCoords[2][1], 0, 'N'),
      formatSynthHetatm(4, 'C4', '2AP', ringCoords[3][0], ringCoords[3][1], 0, 'C'),
      formatSynthHetatm(5, 'C5', '2AP', ringCoords[4][0], ringCoords[4][1], 0, 'C'),
      formatSynthHetatm(6, 'C6', '2AP', ringCoords[5][0], ringCoords[5][1], 0, 'C'),
      formatSynthHetatm(7, 'NA2', '2AP', (r + 1.31) * Math.cos(Math.PI / 3), (r + 1.31) * Math.sin(Math.PI / 3), 0, 'N'),
      'END',
    ].join('\n');
    const prep2ap = extractHoloReceptorAndLigands(aminoPyrimPdb, { ligandCode: '2AP' });
    const parsed2ap = assertValidSdfValenceBudgets(prep2ap.selectedLigandSdf!);
    expect(parsed2ap.bondOrderBetween(1, 6)).toBe(1); // C2-NA2 stays order 1
    expect(parsed2ap.valences[1]).toBe(4);
    const [mol2ap] = gninaParseLigands(prep2ap.selectedLigandSdf!, 'sdf');
    expect(countPolarHydrogensForAtom(mol2ap, 6)).toBe(2);

    // Neutral guanidine vs protonated 1+ guanidinium (all C-N = 1.33 Å)
    const guanNeutralPdb = [
      formatSynthHetatm(1, 'C1', 'GUD', 0.0, 0.0, 0.0, 'C'),
      formatSynthHetatm(2, 'N1', 'GUD', 1.33, 0.0, 0.0, 'N'),
      formatSynthHetatm(3, 'N2', 'GUD', -0.665, 1.152, 0.0, 'N'),
      formatSynthHetatm(4, 'N3', 'GUD', -0.665, -1.152, 0.0, 'N'),
      'END',
    ].join('\n');
    const prepGuaN = extractHoloReceptorAndLigands(guanNeutralPdb, { ligandCode: 'GUD' });
    const parsedGuaN = assertValidSdfValenceBudgets(prepGuaN.selectedLigandSdf!);
    expect(parsedGuaN.valences[0]).toBeLessThanOrEqual(4);
    const [molGuaN] = gninaParseLigands(prepGuaN.selectedLigandSdf!, 'sdf');
    expect(countPolarHydrogensForAtom(molGuaN, 1)).toBe(2);

    const guanPosPdb = [
      formatSynthHetatm(1, 'C1', 'GUP', 0.0, 0.0, 0.0, 'C'),
      formatSynthHetatm(2, 'N1', 'GUP', 1.33, 0.0, 0.0, 'N', '1+'),
      formatSynthHetatm(3, 'N2', 'GUP', -0.665, 1.152, 0.0, 'N'),
      formatSynthHetatm(4, 'N3', 'GUP', -0.665, -1.152, 0.0, 'N'),
      'END',
    ].join('\n');
    const prepGuaP = extractHoloReceptorAndLigands(guanPosPdb, { ligandCode: 'GUP' });
    const parsedGuaP = assertValidSdfValenceBudgets(prepGuaP.selectedLigandSdf!);
    expect(parsedGuaP.bondOrderBetween(0, 1)).toBe(2); // C1=N1(+)
    expect(parsedGuaP.valences[0]).toBe(4);
    expect(parsedGuaP.valences[1]).toBe(2);
    const [molGuaP] = gninaParseLigands(prepGuaP.selectedLigandSdf!, 'sdf');
    // N1(+) has target 3 + 1 = 4, valence = 2 => 2 N-H hydrogens (=NH2+)
    expect(countPolarHydrogensForAtom(molGuaP, 1)).toBe(2);
    expect(countPolarHydrogensForAtom(molGuaP, 2)).toBe(2);
    expect(countPolarHydrogensForAtom(molGuaP, 3)).toBe(2);
  });
});
