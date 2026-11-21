import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { WORKFLOW_PRESETS } from '$lib/engine/registry';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { decodeHashToGraph, encodeGraphToHash } from '$lib/engine/share';
import {
  selectFoldWorkItems,
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from '$lib/engine/bio-capabilities';
import { compileNodeInputPreview } from '$lib/engine/input-preview';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { generateSampleHelixBundlePdb } from '$lib/protein-ops/pdb';
import type { SequenceSet, StructureArtifact, WorkflowGraph } from '$lib/engine/types';

describe('Codex computer-use regression suite', () => {
  it('preserves numeric numDesigns on #p= preset links and custom node positions on both #p= and #g= share URLs', async () => {
    // 1. Preset link with numDesigns = 3 (schema-declared numeric param omitted from preset defaults)
    const preset = structuredClone(WORKFLOW_PRESETS[0].graph);
    preset.nodes[0].params.numDesigns = 3;
    preset.nodes[0].params.length = 60;
    preset.nodes[0].position = { x: 145, y: 235 };
    preset.nodes[1].position = { x: 465, y: 120 };

    const encPreset = await encodeGraphToHash(preset);
    expect(encPreset.mode).toBe('preset-diff');
    const decPreset = await decodeHashToGraph(encPreset.hash);
    expect(decPreset).not.toBeNull();
    expect(decPreset!.nodes[0].params.numDesigns).toBe(3);
    expect(typeof decPreset!.nodes[0].params.numDesigns).toBe('number');
    expect(decPreset!.nodes[0].params.length).toBe(60);
    expect(typeof decPreset!.nodes[0].params.length).toBe('number');
    expect(decPreset!.nodes[0].position).toEqual({ x: 145, y: 235 });
    expect(decPreset!.nodes[1].position).toEqual({ x: 465, y: 120 });

    // 2. Custom DAG preserves all custom node positions through #g= round-trip
    const custom: WorkflowGraph = {
      id: 'custom-positioned',
      name: 'Custom Positioned',
      nodes: [
        {
          id: 'g1',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: { ...defaultParamsForAdapter('genie3'), length: 64, numDesigns: 2 },
          position: { x: 112, y: 290 },
        },
        {
          id: 'm1',
          adapterId: 'web-protein-mpnn',
          label: 'MPNN',
          params: { ...defaultParamsForAdapter('web-protein-mpnn'), numSequences: 3 },
          position: { x: 478, y: 84 },
        },
        {
          id: 'k1',
          adapterId: 'top-k-filter',
          label: 'Top-K Filter',
          params: { ...defaultParamsForAdapter('top-k-filter'), topK: 1 },
          position: { x: 830, y: 312 },
        },
      ],
      edges: [
        { fromNode: 'g1', fromPort: 'structure', toNode: 'm1', toPort: 'structure' },
        { fromNode: 'm1', fromPort: 'sequences', toNode: 'k1', toPort: 'sequences' },
      ],
    };

    const encCustom = await encodeGraphToHash(custom);
    expect(encCustom.mode).toBe('compressed-dag');
    const decCustom = await decodeHashToGraph(encCustom.hash);
    expect(decCustom).not.toBeNull();
    expect(decCustom!.nodes.map((n) => n.position)).toEqual([
      { x: 112, y: 290 },
      { x: 478, y: 84 },
      { x: 830, y: 312 },
    ]);
    expect(decCustom!.nodes[0].params.numDesigns).toBe(2);
    expect(typeof decCustom!.nodes[0].params.numDesigns).toBe('number');
  });

  it('selectFoldWorkItems matches multi-backbone round-robin fan-out without false exclusions', () => {
    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'mpnn-batch',
      parentStructureId: 'genie3-batch',
      sourceModelId: 'web-protein-mpnn',
      candidates: [
        { id: 'd1-mpnn-1', sequence: 'M'.repeat(60), score: 1.1, globalScore: 1.1, sourceModelId: 'web-protein-mpnn', designKey: 'genie3:abc:0', designIndex: 0 },
        { id: 'd1-mpnn-2', sequence: 'M'.repeat(60), score: 1.2, globalScore: 1.2, sourceModelId: 'web-protein-mpnn', designKey: 'genie3:abc:0', designIndex: 0 },
        { id: 'd2-mpnn-1', sequence: 'M'.repeat(60), score: 1.0, globalScore: 1.0, sourceModelId: 'web-protein-mpnn', designKey: 'genie3:abc:1', designIndex: 1 },
        { id: 'd2-mpnn-2', sequence: 'M'.repeat(60), score: 1.3, globalScore: 1.3, sourceModelId: 'web-protein-mpnn', designKey: 'genie3:abc:1', designIndex: 1 },
        { id: 'd3-mpnn-1', sequence: 'M'.repeat(60), score: 0.9, globalScore: 0.9, sourceModelId: 'web-protein-mpnn', designKey: 'genie3:abc:2', designIndex: 2 },
        { id: 'd3-mpnn-2', sequence: 'M'.repeat(60), score: 1.4, globalScore: 1.4, sourceModelId: 'web-protein-mpnn', designKey: 'genie3:abc:2', designIndex: 2 },
      ],
    };

    // 3 backbones × 2 seqs = 6 folds with maxSequences=2, maxTotalFolds=16 -> 0 excluded
    const allSix = selectFoldWorkItems(seqSet, { maxSequences: 2, maxTotalFolds: 16 });
    expect(allSix.selected).toHaveLength(6);
    expect(allSix.excludedCandidates).toHaveLength(0);

    // Campaign cap maxTotalFolds=4 excludes 2 cleanly
    const capped = selectFoldWorkItems(seqSet, { maxSequences: 2, maxTotalFolds: 4 });
    expect(capped.selected).toHaveLength(4);
    expect(capped.excludedCandidates).toHaveLength(2);

    // Single-backbone legacy behavior preserved
    const singleBackbone: SequenceSet = {
      ...seqSet,
      candidates: seqSet.candidates.map((c) => ({ ...c, designKey: 'genie3:abc:0', designIndex: 0 })),
    };
    const single = selectFoldWorkItems(singleBackbone, { maxSequences: 2 });
    expect(single.selected).toHaveLength(2);
    expect(single.excludedCandidates).toHaveLength(4);
  });

  it('Studio preview in pdb-id mode never inherits stale demo_3helix_53aa.pdb label', () => {
    const samplePdb = generateSampleHelixBundlePdb();
    const graph: WorkflowGraph = {
      id: 'pdb-1igt-test',
      name: '1IGT Test',
      nodes: [
        {
          id: 'pdb1',
          adapterId: 'pdb-input',
          label: 'RCSB 1IGT',
          params: {
            mode: 'pdb-id',
            pdbId: '1IGT',
            fetchedPdbId: '1IGT',
            filename: 'demo_3helix_53aa.pdb', // stale default left over from demo-bundle mode
            pdbText: samplePdb,
            cropSegments: 'A1-20',
          },
        },
      ],
      edges: [],
    };

    const preview = compileNodeInputPreview(graph, 'pdb1', {});
    expect(preview.previewStructureMeta?.label).toContain('1IGT.cif');
    expect(preview.previewStructureMeta?.label).not.toContain('demo_3helix_53aa.pdb');
    expect(preview.payload.headline).not.toContain('demo_3helix_53aa.pdb');
  });

  it('enforces UI regression guards (measured node preservation, ResizeObserver fit, live hover token, ORT log level, canonical design index)', () => {
    const canvasSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/WorkflowCanvas.svelte'), 'utf8');
    expect(canvasSrc).toContain('prevFlow?.measured');
    expect(canvasSrc).toContain('new ResizeObserver');

    const peekSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/ArtifactPeekPanel.svelte'), 'utf8');
    expect(peekSrc).toContain('inferCanonicalDesignIndex');
    expect(peekSrc).toContain('designIndex: r.designIndex');

    const fastaSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/AnnotatedFastaView.svelte'), 'utf8');
    expect(fastaSrc).toContain('localHoveredToken');
    expect(fastaSrc).toContain('residueByToken.get(localHoveredToken)');

    const mpnnSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/mpnn.worker.ts'), 'utf8');
    expect(mpnnSrc).toContain('logSeverityLevel: 3');
    expect(mpnnSrc).toContain("logLevel = 'error'");
  });

  it('flags dropped ligands loudly across Studio preview, Layer-1 bio compatibility (with 1-click fix), and Layer-2 runtime guard', () => {
    const holoPdb = [
      'ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00 20.00           N  ',
      'ATOM      2  CA  ALA A  10       1.458   0.000   0.000  1.00 20.00           C  ',
      'ATOM      3  C   ALA A  10       2.009   1.420   0.000  1.00 20.00           C  ',
      'ATOM      4  O   ALA A  10       1.250   2.390   0.000  1.00 20.00           O  ',
      'ATOM      5  N   GLY A  11       3.320   1.500   0.000  1.00 20.00           N  ',
      'ATOM      6  CA  GLY A  11       4.000   2.800   0.000  1.00 20.00           C  ',
      'ATOM      7  C   GLY A  11       5.500   2.600   0.000  1.00 20.00           C  ',
      'ATOM      8  O   GLY A  11       6.100   1.550   0.000  1.00 20.00           O  ',
      'ATOM      9  N   VAL A  12       6.100   3.800   0.000  1.00 20.00           N  ',
      'ATOM     10  CA  VAL A  12       7.500   4.000   0.000  1.00 20.00           C  ',
      'ATOM     11  C   VAL A  12       8.200   2.700   0.000  1.00 20.00           C  ',
      'ATOM     12  O   VAL A  12       7.700   1.600   0.000  1.00 20.00           O  ',
      'HETATM   13  C1  BEN A 246       3.000   5.000   2.000  1.00 18.00           C  ',
      'HETATM   14 CA    CA A 480       5.000   5.000   2.000  1.00 12.00          CA  ',
      'END',
    ].join('\n');

    const graph: WorkflowGraph = {
      id: 'holo-crop-test',
      name: 'Holo Crop Test',
      nodes: [
        {
          id: 'pdb1',
          adapterId: 'pdb-input',
          label: 'RCSB 3PTB',
          params: {
            mode: 'pdb-id',
            pdbId: '3PTB',
            fetchedPdbId: '3PTB',
            filename: '3PTB.cif',
            pdbText: holoPdb,
            cropSegments: 'A',
          },
        },
        {
          id: 'lig1',
          adapterId: 'ligandmpnn',
          label: 'LigandMPNN',
          params: { ...defaultParamsForAdapter('ligandmpnn'), useLigandContext: true },
        },
      ],
      edges: [{ fromNode: 'pdb1', fromPort: 'structure', toNode: 'lig1', toPort: 'structure' }],
    };

    // 1. Studio pdb-input preview explicitly warns about dropped ligands BEN & CA
    const pdbPrev = compileNodeInputPreview(graph, 'pdb1', {});
    expect(pdbPrev.issues.some((i) => i.severity === 'warning' && /BEN/.test(i.message) && /CA/.test(i.message))).toBe(true);

    // 2. Studio ligandmpnn preview warns about apo fallback instead of silently showing holo
    const ligPrev = compileNodeInputPreview(graph, 'lig1', {});
    expect(ligPrev.issues.some((i) => i.severity === 'warning' && /apo|no non-polymer ligands/i.test(i.message))).toBe(true);
    expect(String(ligPrev.payload.compiledSummary.useLigandContext)).toMatch(/apo/i);

    // 3. Layer-1 bio compatibility surfaces crop-dropped-ligands with 1-click fix restoring BEN, CA
    const bioIssues = validateBioCapabilities(graph, {});
    const dropIssue = bioIssues.find((i) => i.code === 'crop-dropped-ligands');
    expect(dropIssue).toBeDefined();
    expect(dropIssue?.fix?.nodeId).toBe('pdb1');
    expect(String(dropIssue?.fix?.paramsPatch?.cropSegments)).toContain('BEN');
    expect(String(dropIssue?.fix?.paramsPatch?.cropSegments)).toContain('CA');

    // 4. Layer-2 runtime guard warns when LigandMPNN receives an apo StructureArtifact
    const apoCoords = new Float32Array([0, 0, 0, 3.8, 0, 0, 7.6, 0, 0]);
    const apoStruct: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'apo-1',
      label: '3PTB [A]',
      sourceModelId: 'pdb-input',
      length: 3,
      sequence: 'AGV',
      caCoords: apoCoords,
      backbone4Coords: new Float32Array(36),
      pdbText: holoPdb.split('\n').filter((l) => !l.startsWith('HETATM')).join('\n'),
      geometry: computeGeometryStats(apoCoords),
      diagnostics: {
        chainCount: 1,
        chainLengths: [3],
        hasLigands: false,
        ligandNames: [],
        hasNucleicAcids: false,
        nucleicAcidChains: [],
      },
    };
    const rt = validateRuntimeNodeInputs(graph.nodes[1], { structure: apoStruct });
    expect(rt.warnings.some((w) => /0 ligands/i.test(w))).toBe(true);
  });

  it('guards against Svelte 5 flowEdges effect loop, Mol* concurrent clear race, mobile dock overlap, and summary/form a11y issues', () => {
    const canvasSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/WorkflowCanvas.svelte'), 'utf8');
    // Must never read `flowEdges` inside the $effect that writes `flowEdges`
    expect(canvasSrc).not.toContain('flowEdges = [...flowEdges');
    expect(canvasSrc).toContain('nextFlowEdges.push(...ghostFlowEdges)');
    expect(canvasSrc).toContain('getFitViewOpts');

    const molSrc = readFileSync(resolve(process.cwd(), 'src/lib/viewer/MolSuperposeViewer.svelte'), 'utf8');
    expect(molSrc).toContain('enqueueSceneRender');
    expect(molSrc).toContain('liveStructs === 0');

    const studioSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/NodeStudioPopover.svelte'), 'utf8');
    const summaryMatch = studioSrc.match(/<summary>[\s\S]*?<\/summary>/);
    expect(summaryMatch?.[0]).toBeTruthy();
    expect(summaryMatch![0]).not.toContain('<button');

    const schemaFieldsSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/SchemaParamFields.svelte'), 'utf8');
    expect(schemaFieldsSrc).toContain('name={field.key}');
    expect(schemaFieldsSrc).toContain('id={`param-${field.key}`}');
  });
});
