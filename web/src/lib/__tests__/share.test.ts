import { describe, it, expect } from 'vitest';
import { WORKFLOW_PRESETS } from '$lib/engine/registry';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import {
  decodeHashToGraph,
  encodeGraphToHash,
  graphToYamlRecipe,
  parseRecipeToGraph,
} from '$lib/engine/share';
import { generateSampleHelixBundlePdb } from '$lib/protein-ops/pdb';
import type { WorkflowGraph } from '$lib/engine/types';

describe('engine/share', () => {
  it('round-trips every WORKFLOW_PRESETS entry via compact #p= preset-diff URL hash', async () => {
    for (const preset of WORKFLOW_PRESETS) {
      const encoded = await encodeGraphToHash(preset.graph);
      expect(encoded.mode).toBe('preset-diff');
      expect(encoded.strippedPdb).toBe(false);
      expect(encoded.hash).toBe(`#p=${preset.id}`);
      expect(encoded.length).toBe(encoded.hash.length);

      const decoded = await decodeHashToGraph(encoded.hash);
      expect(decoded).not.toBeNull();
      expect(decoded!.nodes.map((n) => [n.id, n.adapterId])).toEqual(
        preset.graph.nodes.map((n) => [n.id, n.adapterId]),
      );
      expect(decoded!.edges).toEqual(preset.graph.edges);
      for (let i = 0; i < preset.graph.nodes.length; i++) {
        expect(decoded!.nodes[i].params).toEqual(preset.graph.nodes[i].params);
        expect(decoded!.nodes[i].position).toBeDefined();
      }
    }
  });

  it('round-trips #p= preset-diff with modified numeric, boolean, and string params', async () => {
    const base = structuredClone(WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-refold-native')!.graph);
    base.nodes[0].params.length = 95;
    base.nodes[0].params.nsteps = 45;
    base.nodes[0].params.earlyStop = false;
    base.nodes[0].params.recycleSchedule = 'medium';
    base.nodes[2].params.scRmsdThreshold = 1.5;
    base.nodes[2].params.plddtThreshold = 82;

    const encoded = await encodeGraphToHash(base);
    expect(encoded.mode).toBe('preset-diff');
    expect(encoded.hash).toContain('p=rfd4-refold-native');
    expect(encoded.hash).toContain('gen.length=95');
    expect(encoded.hash).toContain('gen.earlyStop=false');

    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes[0].params.length).toBe(95);
    expect(typeof decoded!.nodes[0].params.length).toBe('number');
    expect(decoded!.nodes[0].params.nsteps).toBe(45);
    expect(decoded!.nodes[0].params.earlyStop).toBe(false);
    expect(typeof decoded!.nodes[0].params.earlyStop).toBe('boolean');
    expect(decoded!.nodes[0].params.recycleSchedule).toBe('medium');
    expect(decoded!.nodes[2].params.scRmsdThreshold).toBe(1.5);
    expect(decoded!.nodes[2].params.plddtThreshold).toBe(82);
  });

  it('round-trips custom DAGs via #g= deflate-raw base64url encoding', async () => {
    const custom: WorkflowGraph = {
      id: 'custom-seq-fold',
      name: 'Custom Seq Fold',
      nodes: [
        {
          id: 'seq1',
          adapterId: 'sequence-input',
          label: 'My Binder Seq',
          params: {
            ...defaultParamsForAdapter('sequence-input'),
            label: 'binder-01',
            sequence: 'MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVK',
          },
        },
        {
          id: 'fold1',
          adapterId: 'localfold-af2',
          label: 'AF2 3-Recycle',
          params: { recycles: 3, maxSequences: 1, seed: 7 },
        },
      ],
      edges: [
        { fromNode: 'seq1', fromPort: 'sequences', toNode: 'fold1', toPort: 'sequences' },
      ],
    };

    const encoded = await encodeGraphToHash(custom);
    expect(encoded.mode).toBe('compressed-dag');
    expect(encoded.hash.startsWith('#g=')).toBe(true);
    expect(encoded.strippedPdb).toBe(false);

    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes.length).toBe(2);
    expect(decoded!.nodes[0].id).toBe('seq1');
    expect(decoded!.nodes[0].adapterId).toBe('sequence-input');
    expect(decoded!.nodes[0].label).toBe('My Binder Seq');
    expect(decoded!.nodes[0].params.sequence).toBe(custom.nodes[0].params.sequence);
    expect(decoded!.nodes[1].id).toBe('fold1');
    expect(decoded!.nodes[1].label).toBe('AF2 3-Recycle');
    expect(decoded!.nodes[1].params).toEqual({ recycles: 3, maxSequences: 1, seed: 7 });
    expect(decoded!.edges).toEqual(custom.edges);
    expect(decoded!.nodes.every((n) => n.position !== undefined)).toBe(true);
  });

  it('strips huge pdbText (> 1500 chars) cleanly in both #p= and #g= modes while preserving small PDBs', async () => {
    const hugePdb = generateSampleHelixBundlePdb();
    expect(hugePdb.length).toBeGreaterThan(1500);

    // 1. Preset-diff mode
    const presetGraph = structuredClone(WORKFLOW_PRESETS.find((p) => p.id === 'pdb-refold-qc')!.graph);
    presetGraph.nodes[0].params.mode = 'custom';
    presetGraph.nodes[0].params.pdbText = hugePdb;
    presetGraph.nodes[0].params.filename = 'huge_upload.pdb';

    const encPreset = await encodeGraphToHash(presetGraph);
    expect(encPreset.mode).toBe('preset-diff');
    expect(encPreset.strippedPdb).toBe(true);
    expect(encPreset.length).toBeLessThan(200);
    const decPreset = await decodeHashToGraph(encPreset.hash);
    expect(decPreset!.nodes[0].params.filename).toBe('huge_upload.pdb');
    expect(decPreset!.nodes[0].params.pdbText).toBe('');

    // 2. Compressed-DAG (#g=) mode
    const customGraph: WorkflowGraph = {
      id: 'pdb-only',
      name: 'PDB Only',
      nodes: [
        {
          id: 'p1',
          adapterId: 'pdb-input',
          params: { mode: 'custom', pdbText: hugePdb, filename: 'big.pdb' },
        },
      ],
      edges: [],
    };
    const encCustom = await encodeGraphToHash(customGraph);
    expect(encCustom.mode).toBe('compressed-dag');
    expect(encCustom.strippedPdb).toBe(true);
    expect(encCustom.length).toBeLessThan(200);
    const decCustom = await decodeHashToGraph(encCustom.hash);
    expect(decCustom!.nodes[0].params.pdbText).toBe('');
    expect(decCustom!.nodes[0].params.filename).toBe('big.pdb');

    // 3. Small PDB (<= 1500 chars) is preserved
    const smallPdb = 'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n';
    customGraph.nodes[0].params.pdbText = smallPdb;
    const encSmall = await encodeGraphToHash(customGraph);
    expect(encSmall.strippedPdb).toBe(false);
    const decSmall = await decodeHashToGraph(encSmall.hash);
    expect(decSmall!.nodes[0].params.pdbText).toBe(smallPdb);
  });

  it('decodeHashToGraph returns null for empty, unknown preset, or malformed prefixes', async () => {
    expect(await decodeHashToGraph('')).toBeNull();
    expect(await decodeHashToGraph('#')).toBeNull();
    expect(await decodeHashToGraph('#p=does-not-exist')).toBeNull();
    expect(await decodeHashToGraph('#unknown=123')).toBeNull();
  });

  it('graphToYamlRecipe & parseRecipeToGraph round-trip YAML and JSON recipes', () => {
    const src = WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-conditioned-cath')!.graph;
    const yaml = graphToYamlRecipe(src);
    expect(yaml).toContain('denovo: v1');
    expect(yaml).toContain('cond.condition -> gen.condition');

    const parsedYaml = parseRecipeToGraph(yaml);
    expect(parsedYaml.nodes.map((n) => [n.id, n.adapterId])).toEqual(
      src.nodes.map((n) => [n.id, n.adapterId]),
    );
    expect(parsedYaml.edges).toEqual(src.edges);
    expect(parsedYaml.nodes[0].params.cathCode).toBe('3.40.50');

    // JSON input support
    const parsedJson = parseRecipeToGraph(JSON.stringify(src));
    expect(parsedJson.nodes.length).toBe(src.nodes.length);

    // Error handling
    expect(() => parseRecipeToGraph('denovo: v1\nnodes:\n  x: {"model":"nope"}\n')).toThrow(/Unknown model/);
    expect(() => parseRecipeToGraph('denovo: v1\nwires:\n  - bad wire syntax\n')).toThrow(/Invalid wire syntax/);
  });
});
