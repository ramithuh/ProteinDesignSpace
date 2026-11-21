import { describe, it, expect } from 'vitest';
import { LOCALFOLD_MODEL_OPTIONS, WORKFLOW_PRESETS } from '$lib/engine/registry';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import {
  decodeHashToGraph,
  encodeGraphToHash,
  graphToYamlRecipe,
  normalizeWorkflowGraph,
  parseRecipeToGraph,
  serializeWorkflowJson,
} from '$lib/engine/share';
import { generateSampleHelixBundlePdb } from '$lib/protein-ops/pdb';
import type { WorkflowGraph } from '$lib/engine/types';

describe('engine/share', () => {
  it.each(['__proto__', 'constructor', 'toString'])('rejects inherited adapter names: %s', (adapterId) => {
    expect(() => normalizeWorkflowGraph({ nodes: [{ id: 'n', adapterId, params: {} }], edges: [] }))
      .toThrow(/Unknown model/);
  });

  it('rejects unsafe node IDs and invalid wires while allowing unfinished recipes', () => {
    const graph: WorkflowGraph = { id: 'import', name: 'Import', nodes: [
      { id: 'a', adapterId: 'rfd4-proteina', params: {} },
      { id: 'b', adapterId: 'rfd4-proteina', params: {} },
    ], edges: [] };
    expect(normalizeWorkflowGraph(graph).nodes).toHaveLength(2);
    graph.nodes[0].id = '__proto__';
    expect(() => normalizeWorkflowGraph(graph)).toThrow(/reserved|invalid/i);
    graph.nodes[0].id = 'a';
    graph.edges = [{ fromNode: 'a', fromPort: 'toString', toNode: 'b', toPort: 'toString' }];
    expect(() => normalizeWorkflowGraph(graph)).toThrow(/unknown.*port/i);
    graph.edges = [{ fromNode: 'a', fromPort: 'sequences', toNode: 'b', toPort: 'template' }];
    expect(() => normalizeWorkflowGraph(graph)).toThrow(/type mismatch/i);
    graph.edges = [{ fromNode: 'a', fromPort: 'structure', toNode: 'b', toPort: 'template' }];
    graph.edges.push({ ...graph.edges[0] });
    expect(() => normalizeWorkflowGraph(graph)).toThrow(/multiple incoming/i);
    graph.edges[1] = { fromNode: 'b', fromPort: 'structure', toNode: 'a', toPort: 'template' };
    expect(() => normalizeWorkflowGraph(graph)).toThrow(/cycle/i);
  });

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
    expect(decoded!.nodes[1].params).toEqual({
      model: 'monomer',
      recycles: 3,
      diffusionSteps: 25,
      maxSequences: 1,
      seed: 7,
    });
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
    // Oversized pdbText strip resets mode to demo-bundle & default filename so shared link stays runnable
    expect(decPreset!.nodes[0].params.mode).toBe('demo-bundle');
    expect(decPreset!.nodes[0].params.filename).toBe('demo_3helix_53aa.pdb');
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
    expect(decCustom!.nodes[0].params.mode).toBe('demo-bundle');
    expect(decCustom!.nodes[0].params.filename).toBe('demo_3helix_53aa.pdb');
    expect(decCustom!.nodes[0].params.pdbText).toBe('');

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
    expect(yaml).toContain('gen.sequences -> fold.sequences');

    const parsedYaml = parseRecipeToGraph(yaml);
    expect(parsedYaml.nodes.map((n) => [n.id, n.adapterId])).toEqual(
      src.nodes.map((n) => [n.id, n.adapterId]),
    );
    expect(parsedYaml.edges).toEqual(src.edges);
    expect(parsedYaml.nodes[0].params.cathCode).toBe('3.40.50');

    // Legacy YAML containing an rfd4-conditioner + template wire auto-migrates into inline generator fields + structure port
    const legacyYaml = [
      'denovo: v1',
      'name: "Legacy Conditioned"',
      'nodes:',
      '  pdb: {"model":"pdb-input","mode":"pdb-id","pdbId":"5O45"}',
      '  cond: {"model":"rfd4-conditioner","contig":"A17-132,/0,70-70","lockChain":"A","hotspots":"A54, A56"}',
      '  g3: {"model":"genie3"}',
      'wires:',
      '  - pdb.structure -> cond.template',
      '  - cond.condition -> g3.condition',
    ].join('\n');
    const migrated = parseRecipeToGraph(legacyYaml);
    expect(migrated.nodes.map((n) => n.adapterId)).toEqual(['pdb-input', 'genie3']);
    expect(migrated.edges).toEqual([
      { fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' },
    ]);
    expect(migrated.nodes[1].params.taskMode).toBe('binder');
    expect(migrated.nodes[1].params.hotspots).toBe('A54, A56');

    // JSON input support
    const parsedJson = parseRecipeToGraph(JSON.stringify(src));
    expect(parsedJson.nodes.length).toBe(src.nodes.length);

    // Error handling
    expect(() => parseRecipeToGraph('denovo: v1\nnodes:\n  x: {"model":"nope"}\n')).toThrow(/Unknown model/);
    expect(() => parseRecipeToGraph('denovo: v1\nwires:\n  - bad wire syntax\n')).toThrow(/Invalid wire syntax/);
  });

  it('round-trips >100 KB payload via deflateRaw/inflateRaw without backpressure deadlock', async () => {
    // 140 KB deterministic non-trivial payload (exceeds internal stream highWaterMark)
    const bigSeq = Array.from({ length: 140_000 }, (_, i) => 'ACDEFGHIKLMNPQRSTVWY'[(i * 7 + (i >> 3)) % 20]).join('');
    const bigGraph: WorkflowGraph = {
      id: 'big-payload',
      name: 'Big Payload',
      nodes: [
        {
          id: 's1',
          adapterId: 'sequence-input',
          params: { label: 'big', sequence: bigSeq },
        },
      ],
      edges: [],
    };
    const encoded = await encodeGraphToHash(bigGraph);
    expect(encoded.mode).toBe('compressed-dag');
    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes[0].params.sequence).toBe(bigSeq);
  });

  it('handles #p= omitted preset params via adapter defaults and guards against prototype pollution', async () => {
    // `partialFraction` is omitted on rfd4-refold-native preset.graph.nodes[0].params,
    // so decodeHashToGraph must coerce it to a number using defaultParamsForAdapter('rfd4-proteina')
    const decoded = await decodeHashToGraph(
      '#p=rfd4-refold-native&gen.partialFraction=0.75&gen.__proto__=bad&gen.constructor=bad&gen.prototype=bad',
    );
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes[0].params.partialFraction).toBe(0.75);
    expect(typeof decoded!.nodes[0].params.partialFraction).toBe('number');
    expect(Object.prototype.hasOwnProperty.call(decoded!.nodes[0].params, '__proto__')).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(decoded!.nodes[0].params, 'constructor')).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(decoded!.nodes[0].params, 'prototype')).toBe(false);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it('parses YAML node IDs named "name" and "denovo" and validates JSON recipe adapters', () => {
    const yaml = [
      'denovo: v1',
      'name: "Reserved Node IDs"',
      'nodes:',
      '  name: {"model":"pdb-input"}',
      '  denovo: {"model":"web-protein-mpnn"}',
      'wires:',
      '  - name.structure -> denovo.structure',
    ].join('\n');
    const parsed = parseRecipeToGraph(yaml);
    expect(parsed.name).toBe('Reserved Node IDs');
    expect(parsed.nodes.map((n) => [n.id, n.adapterId])).toEqual([
      ['name', 'pdb-input'],
      ['denovo', 'web-protein-mpnn'],
    ]);
    expect(parsed.edges).toEqual([
      { fromNode: 'name', fromPort: 'structure', toNode: 'denovo', toPort: 'structure' },
    ]);

    // JSON recipe rejects unknown adapterId and missing node id
    expect(() =>
      parseRecipeToGraph(JSON.stringify({ nodes: [{ id: 'a', adapterId: 'fake-model', params: {} }], edges: [] })),
    ).toThrow(/Unknown model/);
    expect(() =>
      parseRecipeToGraph(JSON.stringify({ nodes: [{ id: '', adapterId: 'pdb-input', params: {} }], edges: [] })),
    ).toThrow(/missing an id/);
  });

  it('round-trips all 7 WORKFLOW_PRESETS and all 13 LocalFold models through YAML recipes without model key collision', () => {
    for (const preset of WORKFLOW_PRESETS) {
      const yaml = graphToYamlRecipe(preset.graph);
      const parsed = parseRecipeToGraph(yaml);
      expect(parsed.nodes.map((n) => [n.id, n.adapterId])).toEqual(
        preset.graph.nodes.map((n) => [n.id, n.adapterId]),
      );
      expect(parsed.edges).toEqual(preset.graph.edges);
      for (let i = 0; i < preset.graph.nodes.length; i++) {
        expect(parsed.nodes[i].params).toMatchObject(preset.graph.nodes[i].params);
      }
    }

    for (const opt of LOCALFOLD_MODEL_OPTIONS) {
      const graph: WorkflowGraph = {
        id: `lf-${opt.id}`,
        name: `LocalFold ${opt.shortLabel}`,
        nodes: [
          {
            id: 'seq',
            adapterId: 'sequence-input',
            params: defaultParamsForAdapter('sequence-input'),
          },
          {
            id: 'fold',
            adapterId: 'localfold-af2',
            label: `LocalFold (${opt.shortLabel})`,
            params: { ...defaultParamsForAdapter('localfold-af2'), model: opt.id },
          },
        ],
        edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
      };
      const yaml = graphToYamlRecipe(graph);
      const parsed = parseRecipeToGraph(yaml);
      const foldNode = parsed.nodes.find((n) => n.id === 'fold')!;
      expect(foldNode.adapterId).toBe('localfold-af2');
      expect(foldNode.label).toBe(`LocalFold (${opt.shortLabel})`);
      expect(foldNode.params.model).toBe(opt.id);
    }

    // Also accept checkpoint and adapterId+model aliases in hand-written YAML recipes
    const aliasYaml = [
      'denovo: v1',
      'name: "Alias Check"',
      'nodes:',
      '  f1: {"model":"localfold-af2","checkpoint":"boltz2"}',
      '  f2: {"adapterId":"localfold-af2","model":"rosettafold3"}',
      'wires:',
    ].join('\n');
    const aliasParsed = parseRecipeToGraph(aliasYaml);
    expect(aliasParsed.nodes[0].adapterId).toBe('localfold-af2');
    expect(aliasParsed.nodes[0].params.model).toBe('boltz2');
    expect(aliasParsed.nodes[1].adapterId).toBe('localfold-af2');
    expect(aliasParsed.nodes[1].params.model).toBe('rosettafold3');
  });

  it('fixes YAML sequence-input label collision, #p= JSON-string corruption, large-PDB strip, and rejects invalid prose/edges', async () => {
    // 1. sequence-input params.label vs node display label must not collide in YAML
    const seqGraph: WorkflowGraph = {
      id: 'seq-label-test',
      name: 'Seq Label Test',
      nodes: [
        {
          id: 's1',
          adapterId: 'sequence-input',
          label: 'My Sequence Block',
          params: {
            ...defaultParamsForAdapter('sequence-input'),
            label: 'binder-alpha-1',
            sequence: 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ',
          },
        },
      ],
      edges: [],
    };
    const seqYaml = graphToYamlRecipe(seqGraph);
    expect(seqYaml).toContain('"seqLabel":"binder-alpha-1"');
    const seqParsed = parseRecipeToGraph(seqYaml);
    expect(seqParsed.nodes[0].label).toBe('My Sequence Block');
    expect(seqParsed.nodes[0].params.label).toBe('binder-alpha-1');

    // 2. #p= URL params whose template value is a string (e.g. customSpecYaml)
    // must remain strings even when starting with '{' or '['
    const condPreset = structuredClone(WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-conditioned-cath')!.graph);
    const jsonOverride = '{"contig":"60-60","seed":7}';
    condPreset.nodes[0].params.customSpecYaml = jsonOverride;
    const encCond = await encodeGraphToHash(condPreset);
    expect(encCond.mode).toBe('preset-diff');
    const decCond = await decodeHashToGraph(encCond.hash);
    expect(typeof decCond!.nodes[0].params.customSpecYaml).toBe('string');
    expect(decCond!.nodes[0].params.customSpecYaml).toBe(jsonOverride);

    // 3. YAML strips >1500-char pdbText cleanly and resets mode so standalone YAML stays runnable
    const hugePdb = generateSampleHelixBundlePdb();
    const pdbGraph: WorkflowGraph = {
      id: 'pdb-yaml-strip',
      name: 'PDB Strip',
      nodes: [
        {
          id: 'p1',
          adapterId: 'pdb-input',
          params: { mode: 'custom', pdbText: hugePdb, filename: 'big.pdb' },
        },
      ],
      edges: [],
    };
    const pdbYaml = graphToYamlRecipe(pdbGraph);
    expect(pdbYaml).toContain('oversized inline PDB');
    const pdbParsed = parseRecipeToGraph(pdbYaml);
    expect(pdbParsed.nodes[0].params.mode).toBe('demo-bundle');
    expect(pdbParsed.nodes[0].params.pdbText).toBe('');

    // 4. Rejects malformed non-JSON prose, missing nodes: section, duplicate IDs, and dangling wires
    expect(() => parseRecipeToGraph('')).toThrow(/empty/i);
    expect(() => parseRecipeToGraph('hello world this is not a recipe')).toThrow(/Unexpected line/);
    expect(() => parseRecipeToGraph('denovo: v1\nname: "No Nodes"\n')).toThrow(/nodes:/);
    expect(() =>
      normalizeWorkflowGraph({
        nodes: [
          { id: 'a', adapterId: 'pdb-input', params: {} },
          { id: 'a', adapterId: 'pdb-input', params: {} },
        ],
        edges: [],
      }),
    ).toThrow(/Duplicate node id/);
    expect(() =>
      normalizeWorkflowGraph({
        nodes: [{ id: 'a', adapterId: 'pdb-input', params: {} }],
        edges: [{ fromNode: 'a', fromPort: 'structure', toNode: 'ghost', toPort: 'structure' }],
      }),
    ).toThrow(/unknown target node/i);

    // 5. Genie 3 custom DAG round-trips with short alias and serializeWorkflowJson round-trips cleanly
    const genieDag: WorkflowGraph = {
      id: 'genie-custom',
      name: 'Genie Custom',
      nodes: [{ id: 'g1', adapterId: 'genie3', params: { ...defaultParamsForAdapter('genie3'), length: 96 } }],
      edges: [],
    };
    const encG = await encodeGraphToHash(genieDag);
    expect(encG.mode).toBe('compressed-dag');
    const decG = await decodeHashToGraph(encG.hash);
    expect(decG!.nodes[0].adapterId).toBe('genie3');
    expect(decG!.nodes[0].params.length).toBe(96);
    expect(parseRecipeToGraph(serializeWorkflowJson(genieDag)).nodes[0].params.length).toBe(96);
  });
});
