import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { WORKFLOW_PRESETS } from '$lib/engine/registry';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { decodeHashToGraph, encodeGraphToHash } from '$lib/engine/share';
import { selectFoldWorkItems } from '$lib/engine/bio-capabilities';
import { compileNodeInputPreview } from '$lib/engine/input-preview';
import { generateSampleHelixBundlePdb } from '$lib/protein-ops/pdb';
import type { SequenceSet, WorkflowGraph } from '$lib/engine/types';

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
});
