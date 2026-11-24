import { describe, it, expect } from 'vitest';
import { LOCALFOLD_MODEL_OPTIONS, WORKFLOW_PRESETS } from '$lib/engine/registry';
import { defaultParamsForAdapter, validateGraph } from '$lib/engine/graph-utils';
import { computeUploadDigest } from '$lib/engine/param-validation';
import { ExecutionScheduler } from '$lib/engine/scheduler';
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
    graph.edges = [{ fromNode: 'a', fromPort: 'sequences', toNode: 'b', toPort: 'structure' }];
    expect(() => normalizeWorkflowGraph(graph)).toThrow(/type mismatch/i);
    graph.edges = [{ fromNode: 'a', fromPort: 'structure', toNode: 'b', toPort: 'structure' }];
    graph.edges.push({ ...graph.edges[0] });
    expect(() => normalizeWorkflowGraph(graph)).toThrow(/multiple incoming/i);
    graph.edges[1] = { fromNode: 'b', fromPort: 'structure', toNode: 'a', toPort: 'structure' };
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
      capMode: 'follow-upstream',
      maxSequences: 1,
      seed: 7,
    });
    expect(decoded!.edges).toEqual(custom.edges);
    expect(decoded!.nodes.every((n) => n.position !== undefined)).toBe(true);
  });

  it('strips huge pdbText (> 1500 chars) in #p= and #g= without reverting to demo-bundle, records SHA-256 digest, and blocks until matching re-upload', async () => {
    const hugePdb = generateSampleHelixBundlePdb();
    expect(hugePdb.length).toBeGreaterThan(1500);
    const expectedSha = computeUploadDigest(hugePdb);

    // 1. Preset-diff (#p=) mode
    const presetGraph = structuredClone(WORKFLOW_PRESETS.find((p) => p.id === 'pdb-refold-qc')!.graph);
    presetGraph.nodes[0].params.mode = 'custom';
    presetGraph.nodes[0].params.pdbText = hugePdb;
    presetGraph.nodes[0].params.filename = 'huge_upload.pdb';

    const encPreset = await encodeGraphToHash(presetGraph);
    expect(encPreset.mode).toBe('preset-diff');
    expect(encPreset.strippedPdb).toBe(true);
    expect(encPreset.length).toBeLessThan(260);
    const decPreset = await decodeHashToGraph(encPreset.hash);
    expect(decPreset).not.toBeNull();
    // Must NOT silently revert to demo-bundle; keeps custom mode + digest metadata and blocks execution
    expect(decPreset!.nodes[0].params.mode).toBe('custom');
    expect(decPreset!.nodes[0].params.filename).toBe('huge_upload.pdb');
    expect(decPreset!.nodes[0].params.expectedFilename).toBe('huge_upload.pdb');
    expect(decPreset!.nodes[0].params.missingUpload).toBe(true);
    expect(decPreset!.nodes[0].params.expectedPdbSha256).toBe(expectedSha);
    expect(decPreset!.nodes[0].params.pdbText).toBe('');
    const presetErrs = validateGraph(decPreset!);
    expect(presetErrs.length).toBeGreaterThan(0);
    expect(presetErrs.join(' ')).toMatch(/huge_upload\.pdb/);
    expect(presetErrs.join(' ')).toContain(expectedSha.slice(0, 12));

    // ExecutionScheduler must refuse to run dependent downstream pipeline while upload is missing
    const scheduler = new ExecutionScheduler();
    await expect(scheduler.run(decPreset!, () => {})).rejects.toThrow(/huge_upload\.pdb/);

    // Re-uploading a mismatched PDB stays blocked
    const wrongPdb = hugePdb + '\nREMARK modified\n';
    decPreset!.nodes[0].params.pdbText = wrongPdb;
    expect(validateGraph(decPreset!).join(' ')).toMatch(/does not match expected custom upload/i);

    // Re-uploading the exact matching PDB unblocks validation
    decPreset!.nodes[0].params.pdbText = hugePdb;
    expect(validateGraph(decPreset!)).toEqual([]);

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
    expect(encCustom.length).toBeLessThan(260);
    const decCustom = await decodeHashToGraph(encCustom.hash);
    expect(decCustom).not.toBeNull();
    expect(decCustom!.nodes[0].params.mode).toBe('custom');
    expect(decCustom!.nodes[0].params.filename).toBe('big.pdb');
    expect(decCustom!.nodes[0].params.expectedFilename).toBe('big.pdb');
    expect(decCustom!.nodes[0].params.missingUpload).toBe(true);
    expect(decCustom!.nodes[0].params.expectedPdbSha256).toBe(expectedSha);
    expect(decCustom!.nodes[0].params.pdbText).toBe('');
    expect(validateGraph(decCustom!).join(' ')).toMatch(/big\.pdb/);

    // Re-encoding an already-stripped graph preserves missing upload metadata
    const reEncoded = await encodeGraphToHash(decCustom!);
    expect(reEncoded.strippedPdb).toBe(true);
    const reDecoded = await decodeHashToGraph(reEncoded.hash);
    expect(reDecoded!.nodes[0].params.mode).toBe('custom');
    expect(reDecoded!.nodes[0].params.expectedPdbSha256).toBe(expectedSha);

    // Re-sharing a graph with malformed expectedPdbSha256 must NOT silently repair the digest
    // across #p=, #g=, or YAML even when a large upload is present
    const malformedPreset = structuredClone(presetGraph);
    malformedPreset.nodes[0].params.expectedPdbSha256 = 'not-a-valid-sha';
    const encMalformedP = await encodeGraphToHash(malformedPreset);
    const decMalformedP = await decodeHashToGraph(encMalformedP.hash);
    expect(decMalformedP!.nodes[0].params.expectedPdbSha256).toBe('not-a-valid-sha');
    decMalformedP!.nodes[0].params.pdbText = hugePdb;
    expect(validateGraph(decMalformedP!).join(' ')).toMatch(/Invalid expectedPdbSha256/i);

    const malformedCustom = structuredClone(customGraph);
    malformedCustom.nodes[0].params.expectedPdbSha256 = 'not-a-valid-sha';
    const encMalformedG = await encodeGraphToHash(malformedCustom);
    const decMalformedG = await decodeHashToGraph(encMalformedG.hash);
    expect(decMalformedG!.nodes[0].params.expectedPdbSha256).toBe('not-a-valid-sha');
    decMalformedG!.nodes[0].params.pdbText = hugePdb;
    expect(validateGraph(decMalformedG!).join(' ')).toMatch(/Invalid expectedPdbSha256/i);

    const malformedYamlParsed = parseRecipeToGraph(graphToYamlRecipe(malformedCustom));
    expect(malformedYamlParsed.nodes[0].params.expectedPdbSha256).toBe('not-a-valid-sha');
    malformedYamlParsed.nodes[0].params.pdbText = hugePdb;
    expect(validateGraph(malformedYamlParsed).join(' ')).toMatch(/Invalid expectedPdbSha256/i);

    // 3. Small PDB (<= 1500 chars) is preserved directly without missingUpload markers
    const smallPdb = [
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  ',
      'ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C  ',
      'ATOM      3  CA  VAL A   3       7.600   0.000   0.000  1.00  0.00           C  ',
      'END',
      '',
    ].join('\n');
    customGraph.nodes[0].params.pdbText = smallPdb;
    const encSmall = await encodeGraphToHash(customGraph);
    expect(encSmall.strippedPdb).toBe(false);
    const decSmall = await decodeHashToGraph(encSmall.hash);
    expect(decSmall!.nodes[0].params.mode).toBe('custom');
    expect(decSmall!.nodes[0].params.pdbText).toBe(smallPdb);
    expect(decSmall!.nodes[0].params.missingUpload).toBeUndefined();
    expect(decSmall!.nodes[0].params.expectedPdbSha256).toBeUndefined();
    expect(validateGraph(decSmall!)).toEqual([]);
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

    // Removed legacy rfd4-conditioner adapter is cleanly rejected as an unknown model
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
    expect(() => parseRecipeToGraph(legacyYaml)).toThrow(/Unknown model "rfd4-conditioner"/);

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
            label: opt.shortLabel,
            params: { ...defaultParamsForAdapter('localfold-af2'), model: opt.id },
          },
        ],
        edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
      };
      const yaml = graphToYamlRecipe(graph);
      const parsed = parseRecipeToGraph(yaml);
      const foldNode = parsed.nodes.find((n) => n.id === 'fold')!;
      expect(foldNode.adapterId).toBe('localfold-af2');
      expect(foldNode.label).toBe(opt.shortLabel);
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

    // 3. YAML strips >1500-char pdbText cleanly while preserving mode:'custom' + digest metadata,
    // and full-JSON round-trips embedded uploads untouched
    const hugePdb = generateSampleHelixBundlePdb();
    const expectedSha = computeUploadDigest(hugePdb);
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
    expect(pdbParsed.nodes[0].params.mode).toBe('custom');
    expect(pdbParsed.nodes[0].params.filename).toBe('big.pdb');
    expect(pdbParsed.nodes[0].params.expectedFilename).toBe('big.pdb');
    expect(pdbParsed.nodes[0].params.missingUpload).toBe(true);
    expect(pdbParsed.nodes[0].params.expectedPdbSha256).toBe(expectedSha);
    expect(pdbParsed.nodes[0].params.pdbText).toBe('');
    expect(validateGraph(pdbParsed).join(' ')).toMatch(/big\.pdb/);

    // Full-JSON export/import preserves embedded custom upload completely untouched
    const fullJson = serializeWorkflowJson(pdbGraph);
    const jsonParsed = parseRecipeToGraph(fullJson);
    expect(jsonParsed.nodes[0].params.mode).toBe('custom');
    expect(jsonParsed.nodes[0].params.filename).toBe('big.pdb');
    expect(jsonParsed.nodes[0].params.pdbText).toBe(hugePdb);
    expect(jsonParsed.nodes[0].params.missingUpload).toBeUndefined();
    expect(jsonParsed.nodes[0].params.expectedPdbSha256).toBeUndefined();
    expect(validateGraph(jsonParsed)).toEqual([]);

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

  it('parses unknown boolean params (userEditedTask=false), preserves custom labels in #p=, suppresses pdb-id strippedPdb warning, and migrates contig-only conditioner to conditioned mode', async () => {
    // 1. Unknown boolean params in #p= decode to actual booleans, not truthy strings like "false"
    const decodedBool = await decodeHashToGraph(
      '#p=rfd4-refold-native&gen.userEditedTask=false&gen.customFlag=true',
    );
    expect(decodedBool).not.toBeNull();
    expect(decodedBool!.nodes[0].params.userEditedTask).toBe(false);
    expect(typeof decodedBool!.nodes[0].params.userEditedTask).toBe('boolean');
    expect(decodedBool!.nodes[0].params.customFlag).toBe(true);
    expect(typeof decodedBool!.nodes[0].params.customFlag).toBe('boolean');

    // 2. Custom node.label is preserved across #p= roundtrip when changed from preset default
    const renamed = structuredClone(WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-refold-native')!.graph);
    renamed.nodes[1].label = 'Boltz-2 (custom)';
    renamed.nodes[1].params.model = 'boltz2';
    const encRenamed = await encodeGraphToHash(renamed);
    expect(encRenamed.mode).toBe('preset-diff');
    expect(encRenamed.hash).toContain('fold._label=');
    const decRenamed = await decodeHashToGraph(encRenamed.hash);
    expect(decRenamed).not.toBeNull();
    expect(decRenamed!.nodes[1].label).toBe('Boltz-2 (custom)');
    expect(decRenamed!.nodes[1].params.model).toBe('boltz2');
    expect(decRenamed!.nodes[1].params._label).toBeUndefined();

    // 3. Cached >1500-char pdbText on mode === 'pdb-id' does NOT trigger strippedPdb warning in URL or YAML
    const hugePdb = generateSampleHelixBundlePdb();
    const pdbIdGraph = structuredClone(WORKFLOW_PRESETS.find((p) => p.id === 'pdb-refold-qc')!.graph);
    pdbIdGraph.nodes[0].params.mode = 'pdb-id';
    pdbIdGraph.nodes[0].params.pdbId = '1PRW';
    pdbIdGraph.nodes[0].params.pdbText = hugePdb;

    const encPdbId = await encodeGraphToHash(pdbIdGraph);
    expect(encPdbId.strippedPdb).toBe(false);
    const decPdbId = await decodeHashToGraph(encPdbId.hash);
    expect(decPdbId!.nodes[0].params.mode).toBe('pdb-id');
    expect(decPdbId!.nodes[0].params.pdbId).toBe('1PRW');

    const yamlPdbId = graphToYamlRecipe(pdbIdGraph);
    expect(yamlPdbId).not.toContain('oversized inline PDB');
    expect(yamlPdbId).toContain('"mode":"pdb-id"');
    expect(yamlPdbId).toContain('"pdbId":"1PRW"');

    // 4. Inline RFD4 conditioned recipe round-trips cleanly while legacy rfd4-conditioner is rejected
    const inlineContigYaml = [
      'denovo: v1',
      'name: "Inline Contig"',
      'nodes:',
      '  pdb: {"model":"pdb-input","mode":"pdb-id","pdbId":"1PRW"}',
      '  gen: {"model":"rfd4-proteina","mode":"motif","contig":"10-20,A16-35,10-20"}',
      'wires:',
      '  - pdb.structure -> gen.structure',
    ].join('\n');
    const parsedInline = parseRecipeToGraph(inlineContigYaml);
    expect(parsedInline.nodes.map((n) => n.adapterId)).toEqual(['pdb-input', 'rfd4-proteina']);
    expect(parsedInline.nodes[1].params.mode).toBe('motif');
    expect(parsedInline.nodes[1].params.contig).toBe('10-20,A16-35,10-20');
  });

  it('preserves unresolvedImportIssues across #p=, #g=, YAML, and blocks execution until cleared (Issue #1)', async () => {
    const scheduler = new ExecutionScheduler();

    // 1. Direct unresolvedImportIssues survive #g= and YAML re-encoding and block validateGraph + scheduler.run
    const blockedGraph: WorkflowGraph = {
      id: 'blocked-import',
      name: 'Blocked Import',
      nodes: [
        {
          id: 'g3',
          adapterId: 'genie3',
          params: {
            ...defaultParamsForAdapter('genie3'),
            unresolvedImportIssues: ['Unsupported CATH code 3.40.50 in Genie 3'],
          },
        },
      ],
      edges: [],
    };
    expect(validateGraph(blockedGraph).join(' ')).toMatch(/CATH.*3\.40\.50/i);
    await expect(scheduler.run(blockedGraph, () => {})).rejects.toThrow(/CATH.*3\.40\.50/i);

    const enc = await encodeGraphToHash(blockedGraph);
    const dec = await decodeHashToGraph(enc.hash);
    expect(validateGraph(dec!).join(' ')).toMatch(/CATH.*3\.40\.50/i);

    const yamlRoundTrip = parseRecipeToGraph(graphToYamlRecipe(blockedGraph));
    expect(validateGraph(yamlRoundTrip).join(' ')).toMatch(/CATH.*3\.40\.50/i);

    // Clearing unresolvedImportIssues unblocks execution
    yamlRoundTrip.nodes[0].params.unresolvedImportIssues = [];
    expect(validateGraph(yamlRoundTrip)).toEqual([]);

    // YAML schema version validation and dotted node IDs in wires
    expect(() =>
      parseRecipeToGraph('denovo: v99\nnodes:\n  g1: {"model":"genie3"}\nwires:\n'),
    ).toThrow(/Unsupported recipe schema version/);

    const dottedGraph: WorkflowGraph = {
      id: 'dotted-ids',
      name: 'Dotted IDs',
      nodes: [
        { id: 'step.1', adapterId: 'pdb-input', params: defaultParamsForAdapter('pdb-input') },
        { id: 'step.2', adapterId: 'web-protein-mpnn', params: defaultParamsForAdapter('web-protein-mpnn') },
      ],
      edges: [{ fromNode: 'step.1', fromPort: 'structure', toNode: 'step.2', toPort: 'structure' }],
    };
    const dottedParsed = parseRecipeToGraph(graphToYamlRecipe(dottedGraph));
    expect(dottedParsed.edges).toEqual(dottedGraph.edges);

    // Malformed non-array unresolvedImportIssues (e.g. object) must NOT be erased by #p=, #g=, or YAML re-sharing
    const malformedPreset = structuredClone(WORKFLOW_PRESETS.find((p) => p.id === 'genie3-mpnn-af2')!.graph);
    malformedPreset.nodes[0].params.unresolvedImportIssues = { field: 'cathCode', value: '3.40.50' };
    const encMalformedP = await encodeGraphToHash(malformedPreset);
    expect(encMalformedP.mode).toBe('preset-diff');
    const decMalformedP = await decodeHashToGraph(encMalformedP.hash);
    expect(validateGraph(decMalformedP!).join(' ')).toMatch(/cathCode/);

    const malformedDag: WorkflowGraph = {
      id: 'malformed-issues',
      name: 'Malformed Issues',
      nodes: [
        {
          id: 'g1',
          adapterId: 'genie3',
          params: {
            ...defaultParamsForAdapter('genie3'),
            unresolvedImportIssues: { field: 'cathCode', value: '3.40.50' },
          },
        },
      ],
      edges: [],
    };
    const encMalformedG = await encodeGraphToHash(malformedDag);
    expect(encMalformedG.mode).toBe('compressed-dag');
    const decMalformedG = await decodeHashToGraph(encMalformedG.hash);
    expect(validateGraph(decMalformedG!).join(' ')).toMatch(/cathCode/);

    const decMalformedYaml = parseRecipeToGraph(graphToYamlRecipe(malformedDag));
    expect(validateGraph(decMalformedYaml).join(' ')).toMatch(/cathCode/);
  });

  it('Wave-2 P1/P2 regressions: invalid #g= inflateRaw without unhandled rejection, null/array params guard, dotted IDs in #p=/YAML, and empty sequence-input label round-trip', async () => {
    // 1. Corrupt compressed #g= returns null cleanly (no unhandled rejection from inflateRaw)
    expect(await decodeHashToGraph('#g=!!!not-valid-deflate!!!')).toBeNull();
    expect(await decodeHashToGraph('#g=AAAA')).toBeNull();

    // 2. Null or array params in JSON/YAML/normalizeWorkflowGraph are sanitized instead of crashing
    expect(() =>
      normalizeWorkflowGraph({
        nodes: [
          { id: 's1', adapterId: 'sequence-input', params: null as unknown as Record<string, unknown> },
          { id: 'g1', adapterId: 'genie3', params: ['bad'] as unknown as Record<string, unknown> },
        ],
        edges: [],
      }),
    ).not.toThrow();
    const normNull = normalizeWorkflowGraph({
      nodes: [{ id: 'g1', adapterId: 'genie3', params: null as unknown as Record<string, unknown> }],
      edges: [],
    });
    expect(normNull.nodes[0].params.length).toBe(80);

    // 3. Dotted node IDs round-trip across YAML wires and #p= diff overrides
    const dottedGraph: WorkflowGraph = {
      id: 'dotted-roundtrip',
      name: 'Dotted',
      nodes: [
        { id: 'v1.src', adapterId: 'pdb-input', params: defaultParamsForAdapter('pdb-input') },
        { id: 'v1.mpnn', adapterId: 'web-protein-mpnn', params: defaultParamsForAdapter('web-protein-mpnn') },
      ],
      edges: [{ fromNode: 'v1.src', fromPort: 'structure', toNode: 'v1.mpnn', toPort: 'structure' }],
    };
    const yamlBack = parseRecipeToGraph(graphToYamlRecipe(dottedGraph));
    expect(yamlBack.edges).toEqual(dottedGraph.edges);

    // 4. Empty sequence-input params.label ('') survives YAML round-trip without being overwritten by node display label
    const seqEmptyLabel: WorkflowGraph = {
      id: 'empty-seq-label',
      name: 'Empty Seq Label',
      nodes: [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          label: 'My Sequence Node',
          params: { label: '', sequence: 'MKTAYIAKQRQISFVK' },
        },
      ],
      edges: [],
    };
    const seqYamlBack = parseRecipeToGraph(graphToYamlRecipe(seqEmptyLabel));
    expect(seqYamlBack.nodes[0].params.label).toBe('');
  });

  it('never emits #g= share links ending in _ or - and recovers links truncated by Markdown/chat autolinkers', async () => {
    // Real-world 1CTT AME + Boltz-2 + Protenix-v2 + EF2 Fast + 3D Compare hash that previously ended in `_`
    const legacyTrailingUnderscoreHash =
      '#g=lVRdb5swFP0r6D4byTioWf3WJYu2h0zT0qcihBx_MDRsmDFVtCr_fbIpCTTp2r5Y5tx77jnX94onMECzDFqxLxJA0AJ6At0ICdRjcSUAwV4a_ksz-_ubAApMyyLhzhVcMM8Q-wAnq_t7QMBt0-5kqaVxHVC4SzBB0V2CU3-SW38uPEIfvvtz_fUBjiiQo23jKgU5ysAqkQY7dmJHhzAC3hhXlUBhieMlBjQEdi0zr-oBgt5URsiDFIMKdbaXCEyv17KrStMBTY4Ifm7WabTjTKmmFsFKXZWFbo0JduqqZEb4T-_L9Hon__TScBnoPt3zin1Tu7-AQI3ua6AQQO9Es8OUNlZdcbFqhIfAP83zw3z2rJjAqXZrG1c8kpfVPSxNdfiwwI9nYvw4iHDdhlb50H3DWR10mSIBn6lKRWLFOhcvMNbXlHXHtsP0usqUtYT_m_myIdGGdS5aYLyFPEcgZ9vZOdtz11tf57QjZ3C2OtPc2RDn-fPQyTyaT_IceJtzntCcda2J8bWtVNL6VLhYomvpJ6y7thjvYLzaweXAX7R-GX9DLfc_FZplKUaf0hxli_FyM15upxeyJDnKEjLF0huc58d_';

    // 1. Original link decodes cleanly
    const fullDecoded = await decodeHashToGraph(legacyTrailingUnderscoreHash);
    expect(fullDecoded).not.toBeNull();
    expect(fullDecoded!.nodes).toHaveLength(7);

    // 2. Truncated link (GFM/X/Slack autolinker stripped trailing `_`) STILL decodes cleanly!
    const strippedDecoded = await decodeHashToGraph(legacyTrailingUnderscoreHash.slice(0, -1));
    expect(strippedDecoded).not.toBeNull();
    expect(strippedDecoded!.nodes).toHaveLength(7);
    expect(strippedDecoded!.edges).toEqual(fullDecoded!.edges);

    // 3. Trailing prose punctuation (e.g. period or closing paren) is tolerated
    const punctuatedDecoded = await decodeHashToGraph(legacyTrailingUnderscoreHash + '.)');
    expect(punctuatedDecoded).not.toBeNull();
    expect(punctuatedDecoded!.nodes).toHaveLength(7);

    // 4. Re-encoding that exact graph produces a fresh #g= hash that does NOT end in `_` or `-`
    const reencoded = await encodeGraphToHash(fullDecoded!);
    expect(reencoded.mode).toBe('compressed-dag');
    expect(reencoded.hash).not.toMatch(/[_-]$/);
    const roundTripped = await decodeHashToGraph(reencoded.hash);
    expect(roundTripped).not.toBeNull();
    expect(roundTripped!.nodes).toHaveLength(7);
    expect(roundTripped!.edges).toEqual(fullDecoded!.edges);
  });

  it('WF-11: #p= share decode restores numeric/boolean/string types for UI tracking keys while keeping mode-scoped pdb-input filtering intact', async () => {
    const preset = structuredClone(WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-refold-native')!.graph);
    preset.nodes[0].params.autoSeededLength = 180;
    preset.nodes[0].params.autoSeededBenchmarkId = 'ame_1ctt_cda';
    preset.nodes[0].params.autoSeededFromBenchmark = 'ame_1ctt_cda';
    preset.nodes[0].params.userEditedTask = false;
    preset.nodes[0].params.userEditedLength = true;

    const encoded = await encodeGraphToHash(preset);
    expect(encoded.mode).toBe('preset-diff');
    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded).not.toBeNull();
    const gen = decoded!.nodes[0].params;
    expect(gen.autoSeededLength).toBe(180);
    expect(typeof gen.autoSeededLength).toBe('number');
    expect(gen.userEditedTask).toBe(false);
    expect(typeof gen.userEditedTask).toBe('boolean');
    expect(gen.userEditedLength).toBe(true);
    expect(typeof gen.userEditedLength).toBe('boolean');
    expect(gen.autoSeededBenchmarkId).toBe('ame_1ctt_cda');
    expect(typeof gen.autoSeededBenchmarkId).toBe('string');
    expect(gen.autoSeededFromBenchmark).toBe('ame_1ctt_cda');
    expect(typeof gen.autoSeededFromBenchmark).toBe('string');
  });
});
