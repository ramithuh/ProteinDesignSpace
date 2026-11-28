import { describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS, WORKFLOW_PRESETS } from '$lib/engine/registry';
import {
  ADAPTER_TO_SHORT,
  FLOW_MODEL_ALIASES,
  SHORT_TO_ADAPTER,
  decodeFlowQuery,
  decodeWorkflowQuery,
  encodeCompactShareQuery,
  encodeGraphToFlow,
  isMarkdownSafeFlowUrl,
  parseFlowDsl,
  parseRecipeSafe,
  resolveFlowModelAlias,
  type FlowCatalogHooks,
} from '$lib/engine/flow-dsl';
import {
  buildShareableWorkflowUrl,
  encodeGraphToHash,
  graphToYamlRecipe,
  serializeWorkflowJson,
} from '$lib/engine/share';
import { validateGraph } from '$lib/engine/graph-utils';
import type { AdapterManifest } from '$lib/engine/types';

describe('engine/flow-dsl', () => {
  it('resolves all 32 canonical adapter IDs, all 32 immutable short codes, and LLM aliases without collisions', () => {
    const adapterIds = Object.keys(ADAPTER_REGISTRY).sort();
    expect(adapterIds).toHaveLength(32);
    expect(Object.keys(ADAPTER_TO_SHORT).sort()).toEqual(adapterIds);
    expect(new Set(Object.values(ADAPTER_TO_SHORT)).size).toBe(32);

    for (const adapterId of adapterIds) {
      const short = ADAPTER_TO_SHORT[adapterId];
      expect(SHORT_TO_ADAPTER[short]).toBe(adapterId);
      expect(resolveFlowModelAlias(adapterId)?.adapterId).toBe(adapterId);
      expect(resolveFlowModelAlias(short)?.adapterId).toBe(adapterId);
    }

    // Aliases never shadow a short code or canonical adapter ID with a different adapter
    for (const [alias, target] of Object.entries(FLOW_MODEL_ALIASES)) {
      if (Object.hasOwn(SHORT_TO_ADAPTER, alias)) {
        expect(SHORT_TO_ADAPTER[alias]).toBe(target.adapterId);
      }
      if (Object.hasOwn(ADAPTER_REGISTRY, alias)) {
        expect(alias).toBe(target.adapterId);
      }
    }

    // All 20 LocalFold checkpoints resolve to localfold-af2 with matching model
    for (const opt of LOCALFOLD_MODEL_OPTIONS) {
      const res = resolveFlowModelAlias(opt.id);
      expect(res?.adapterId).toBe('localfold-af2');
      expect(res?.presetParams?.model).toBe(opt.id);
    }

    // Common LLM shorthands & variant presets
    expect(resolveFlowModelAlias('rfd4')).toEqual({ adapterId: 'rfd4-proteina' });
    expect(resolveFlowModelAlias('rfdiffusion3')).toEqual({ adapterId: 'rfd3' });
    expect(resolveFlowModelAlias('mpnn')).toEqual({ adapterId: 'web-protein-mpnn' });
    expect(resolveFlowModelAlias('proteinmpnn')).toEqual({
      adapterId: 'web-protein-mpnn',
      presetParams: { family: 'proteinmpnn' },
    });
    expect(resolveFlowModelAlias('campnn')).toEqual({
      adapterId: 'web-protein-mpnn',
      presetParams: { family: 'ca' },
    });
    expect(resolveFlowModelAlias('antifold')).toEqual({
      adapterId: 'antifold-esmif1',
      presetParams: { variant: 'antifold' },
    });
    expect(resolveFlowModelAlias('esmif1')).toEqual({
      adapterId: 'antifold-esmif1',
      presetParams: { variant: 'esmif1' },
    });
    expect(resolveFlowModelAlias('rf3')).toEqual({
      adapterId: 'localfold-af2',
      presetParams: { model: 'rosettafold3' },
    });
    expect(resolveFlowModelAlias('boltz-2')).toEqual({
      adapterId: 'localfold-af2',
      presetParams: { model: 'boltz2' },
    });
    expect(resolveFlowModelAlias('af3')).toEqual({
      adapterId: 'localfold-af2',
      presetParams: { model: 'af3' },
    });
    expect(resolveFlowModelAlias('ef2-fast-300m')).toEqual({
      adapterId: 'localfold-af2',
      presetParams: { model: 'ef2-fast-300m' },
    });
    expect(resolveFlowModelAlias('ef2-fast-600m')).toEqual({
      adapterId: 'localfold-af2',
      presetParams: { model: 'ef2-fast-600m' },
    });
  });

  it('parses linear chains, multi-branch comparisons, fan-out/fan-in, and designability-scorer dual-input lineage', () => {
    // 1. Linear 3-step chain
    const lin = parseFlowDsl('r3:length=80~m~f');
    expect(lin.ok).toBe(true);
    expect(lin.errors).toEqual([]);
    expect(lin.nodes.map((n) => n.adapterId)).toEqual(['rfd3', 'web-protein-mpnn', 'localfold-af2']);
    expect(lin.edges).toEqual([
      { fromNode: 'r3', fromPort: 'structure', toNode: 'm', toPort: 'structure' },
      { fromNode: 'm', fromPort: 'sequences', toNode: 'f', toPort: 'sequences' },
    ]);

    // 2. Multi-branch RFD3 vs RFD4 comparison into structure-compare
    const cmp = parseFlowDsl('src:p:pdbId=1ubq~r3:length=80~cmp;src~r:length=80~cmp');
    expect(cmp.ok).toBe(true);
    expect(cmp.nodes.map((n) => [n.id, n.adapterId])).toEqual([
      ['src', 'pdb-input'],
      ['r3', 'rfd3'],
      ['cmp', 'structure-compare'],
      ['r', 'rfd4-proteina'],
    ]);
    expect(cmp.nodes[0].params.mode).toBe('pdb-id');
    expect(cmp.nodes[0].params.pdbId).toBe('1UBQ');
    expect(cmp.edges).toEqual([
      { fromNode: 'src', fromPort: 'structure', toNode: 'r3', toPort: 'structure' },
      { fromNode: 'r3', fromPort: 'structure', toNode: 'cmp', toPort: 'structures' },
      { fromNode: 'src', fromPort: 'structure', toNode: 'r', toPort: 'structure' },
      { fromNode: 'r', fromPort: 'structure', toNode: 'cmp', toPort: 'structures' },
    ]);

    // 3. Fan-out/fan-in with designability-scorer auto-wired via upstream lineage BFS
    const scorerFlow = parseFlowDsl('r3:length=70~m:numSequences=4~f~s');
    expect(scorerFlow.ok).toBe(true);
    expect(scorerFlow.edges).toEqual([
      { fromNode: 'r3', fromPort: 'structure', toNode: 'm', toPort: 'structure' },
      { fromNode: 'm', fromPort: 'sequences', toNode: 'f', toPort: 'sequences' },
      { fromNode: 'f', fromPort: 'folds', toNode: 's', toPort: 'folds' },
      { fromNode: 'r3', fromPort: 'structure', toNode: 's', toPort: 'structure' },
    ]);

    // 4. Direct RFD4 native sequence -> LocalFold -> designability-scorer lineage
    const nativeScorer = parseFlowDsl('rfd4:length=65~af2~scorer');
    expect(nativeScorer.ok).toBe(true);
    expect(nativeScorer.edges).toEqual([
      { fromNode: 'rfd4', fromPort: 'sequences', toNode: 'af2', toPort: 'sequences' },
      { fromNode: 'af2', fromPort: 'folds', toNode: 'scorer', toPort: 'folds' },
      { fromNode: 'rfd4', fromPort: 'structure', toNode: 'scorer', toPort: 'structure' },
    ]);

    // 5. Parallel branch shorthand without explicit IDs allocates sibling nodes cleanly
    const parallel = parseFlowDsl('rfd3~mpnn~af2;rfd4~mpnn~af2');
    expect(parallel.ok).toBe(true);
    expect(parallel.nodes.map((n) => n.id)).toEqual(['rfd3', 'mpnn', 'af2', 'rfd4', 'mpnn_2', 'af2_2']);
  });

  it('handles genie2.template, Cα MPNN family inference, structure-compare.reference, and explicit from.port~to.port', () => {
    // genie2 input port is `template`
    const g2 = parseFlowDsl('pdb:pdbId=1ubq~genie2:length=90~mpnn~af2~scorer');
    expect(g2.ok).toBe(true);
    expect(g2.edges[0]).toEqual({ fromNode: 'pdb', fromPort: 'structure', toNode: 'genie2', toPort: 'template' });
    // Cα-only generator infers family='ca' on web-protein-mpnn when not explicitly specified
    expect(g2.nodes.find((n) => n.id === 'mpnn')?.params.family).toBe('ca');
    expect(g2.edges).toContainEqual({
      fromNode: 'genie2',
      fromPort: 'structure',
      toNode: 'mpnn',
      toPort: 'structure',
    });

    // Explicit family=solublempnn or solublempnn alias overrides Cα inference
    const g3Explicit = parseFlowDsl('g3:length=80~solublempnn~f~s');
    expect(g3Explicit.ok).toBe(true);
    expect(g3Explicit.nodes.find((n) => n.adapterId === 'web-protein-mpnn')?.params.family).toBe('solublempnn');

    // Explicit from.port~to.port with LASErMPNN dual outputs + pdb-input reference on structure-compare
    const exp = parseFlowDsl(
      'src:pdb:pdbId=1PRW~pack:lasermpnn:numSequences=2~fold:rf3~score:scorer;src.structure~cmp:compare.reference;pack.structure~cmp;fold.structure~cmp',
    );
    expect(exp.ok).toBe(true);
    expect(exp.edges).toContainEqual({
      fromNode: 'src',
      fromPort: 'structure',
      toNode: 'cmp',
      toPort: 'reference',
    });
    expect(exp.edges).toContainEqual({
      fromNode: 'pack',
      fromPort: 'structure',
      toNode: 'cmp',
      toPort: 'structures',
    });
    expect(exp.edges).toContainEqual({
      fromNode: 'fold',
      fromPort: 'structure',
      toNode: 'cmp',
      toPort: 'structures',
    });
  });

  it('context-aware lexing preserves dotted node IDs, decimals, colons, commas, quotes, and sequence-input labels', () => {
    // Dotted node IDs & dotted ports
    const dotted = parseFlowDsl('v1.src:pdb:pdbId=1ubq;v1.mpnn:mpnn:numSequences=2;v1.src.structure~v1.mpnn.structure');
    expect(dotted.ok).toBe(true);
    expect(dotted.nodes.map((n) => n.id)).toEqual(['v1.src', 'v1.mpnn']);
    expect(dotted.edges).toEqual([
      { fromNode: 'v1.src', fromPort: 'structure', toNode: 'v1.mpnn', toPort: 'structure' },
    ]);

    // Colons in sequence, residueMapping, contig + commas in contig, hotspots, cropSegments + numeric decimals
    const complex = parseFlowDsl(
      'src:pdb:pdbId=1prw,cropSegments=A16-35,A52-71~gen:rfd4:cathCode=3.40.50,contig=h1:16-16P,l1:4-4P,_label="My CATH Design"~mpnn:tiedPositions="A1=B25;A2=B26",numSequences=2~fold:af2:maxSequences=2~sc:scorer:scRmsdThreshold=1.5,residueMapping=0:0,1:1',
    );
    expect(complex.ok).toBe(true);
    const src = complex.nodes.find((n) => n.id === 'src')!;
    const gen = complex.nodes.find((n) => n.id === 'gen')!;
    const mpnn = complex.nodes.find((n) => n.id === 'mpnn')!;
    const fold = complex.nodes.find((n) => n.id === 'fold')!;
    const sc = complex.nodes.find((n) => n.id === 'sc')!;
    expect(src.params.cropSegments).toBe('A16-35,A52-71');
    expect(gen.params.cathCode).toBe('3.40.50');
    expect(gen.params.contig).toBe('h1:16-16P,l1:4-4P');
    expect(gen.label).toBe('My CATH Design');
    expect(mpnn.params.tiedPositions).toBe('A1=B25;A2=B26');
    expect(fold.params.capMode).toBe('fixed');
    expect(fold.params.maxSequences).toBe(2);
    expect(sc.params.scRmsdThreshold).toBe(1.5);
    expect(sc.params.residueMapping).toBe('0:0,1:1');

    // sequence-input `label=` / `seqLabel=` sets params.label without colliding with `_label=` node display label
    const seqRes = parseFlowDsl('q1:seq:sequence=MKT:ACDE,label=my-fasta-id,_label="Input Seq"~fold:af2');
    expect(seqRes.ok).toBe(true);
    expect(seqRes.nodes[0].label).toBe('Input Seq');
    expect(seqRes.nodes[0].params.label).toBe('my-fasta-id');
    expect(seqRes.nodes[0].params.sequence).toBe('MKT:ACDE');

    const seqAliasRes = parseFlowDsl('seq:seqLabel=binder-alpha,sequence=ACDEFGHIKL~af2');
    expect(seqAliasRes.ok).toBe(true);
    expect(seqAliasRes.nodes[0].params.label).toBe('binder-alpha');
  });

  it('tolerates ->, node(k=v), newlines, trailing prose punctuation, and extra query parameters', () => {
    const tolerated = parseFlowDsl(
      'src:pdb(pdbId=1ubq) -> rfd3(length=80,hotspots=A54,A56) -> cmp\nsrc -> rfd4(length=80) -> cmp.',
    );
    expect(tolerated.ok).toBe(true);
    expect(tolerated.nodes[1].params.mode).toBe('binder');
    expect(tolerated.nodes[1].params.hotspots).toBe('A54,A56');

    const fromUrl = decodeFlowQuery(
      'https://denovo.run/?utm_source=chatgpt.com&flow=r3(length=80)->mpnn->af2).',
    );
    expect(fromUrl.ok).toBe(true);
    expect(fromUrl.nodes.map((n) => n.adapterId)).toEqual(['rfd3', 'web-protein-mpnn', 'localfold-af2']);
  });

  it('returns structured validation errors for invalid flows, cycles, incompatible ports, missing inputs, and schema violations', () => {
    expect(parseFlowDsl('').errors[0].code).toBe('EMPTY_FLOW');

    const unknown = parseFlowDsl('rfd4-conditioner~genie3');
    expect(unknown.ok).toBe(false);
    expect(unknown.errors[0].code).toBe('UNKNOWN_MODEL');

    const missingStruct = parseFlowDsl('seq~af2~scorer');
    expect(missingStruct.ok).toBe(false);
    expect(missingStruct.errors.some((e) => e.code === 'MISSING_REQUIRED_INPUT' && e.nodeId === 'scorer')).toBe(true);

    const incompatible = parseFlowDsl('q:sequence=MKTAY~mpnn');
    expect(incompatible.ok).toBe(false);
    expect(incompatible.errors[0].code).toBe('INCOMPATIBLE_PORTS');

    // LocalFold accepts optional template:StructureArtifact, so rfdiffusion~af2 wires template
    // but still fails with MISSING_REQUIRED_INPUT for sequences; pairing with seq succeeds.
    const templateOnly = parseFlowDsl('rfdiffusion~af2');
    expect(templateOnly.ok).toBe(false);
    expect(templateOnly.errors.some((e) => e.code === 'MISSING_REQUIRED_INPUT' && e.nodeId === 'af2')).toBe(true);

    const templateFold = parseFlowDsl('seq:sequence=MKTAY~af2;pdb:pdbId=1UBQ~af2');
    expect(templateFold.ok).toBe(true);
    expect(templateFold.graph?.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ fromNode: 'seq', fromPort: 'sequences', toNode: 'af2', toPort: 'sequences' }),
        expect.objectContaining({ fromNode: 'pdb', fromPort: 'structure', toNode: 'af2', toPort: 'template' }),
      ]),
    );

    const unknownPort = parseFlowDsl('r3.bogus~m');
    expect(unknownPort.ok).toBe(false);
    expect(unknownPort.errors[0].code).toBe('UNKNOWN_PORT');

    const cycle = parseFlowDsl('a:r3~b:r3~a');
    expect(cycle.ok).toBe(false);
    expect(cycle.errors.some((e) => e.code === 'CYCLE_DETECTED')).toBe(true);

    const selfLoop = parseFlowDsl('a:r3~a');
    expect(selfLoop.ok).toBe(false);
    expect(selfLoop.errors.some((e) => e.code === 'SELF_LOOP')).toBe(true);

    const badParam = parseFlowDsl('r3:length=5~m~f');
    expect(badParam.ok).toBe(false);
    expect(badParam.errors.some((e) => e.code === 'INVALID_PARAM')).toBe(true);

    const badCross = parseFlowDsl('rfd4:mode=motif');
    expect(badCross.ok).toBe(false);
    expect(badCross.errors.some((e) => e.code === 'CROSS_VALIDATION_ERROR')).toBe(true);

    const badSym = parseFlowDsl('r3:symmetry=c3,cyclic=true');
    expect(badSym.ok).toBe(false);
    expect(badSym.errors.some((e) => e.code === 'CROSS_VALIDATION_ERROR')).toBe(true);

    const tooManyCompare = parseFlowDsl('a:r3~c;b:r3~c;d1:r3~c;d2:r3~c;d3:r3~c;d4:r3~c;d5:r3~c;d6:r3~c');
    expect(tooManyCompare.ok).toBe(false);
    expect(tooManyCompare.errors.some((e) => e.code === 'MAX_MULTI_INPUTS_EXCEEDED')).toBe(true);

    const tooManyStructuresPort = parseFlowDsl(
      'a:r3~c.structures;b:r3~c.structures;d1:r3~c.structures;d2:r3~c.structures;d3:r3~c.structures;d4:r3~c.structures;d5:r3~c.structures',
    );
    expect(tooManyStructuresPort.ok).toBe(false);
    expect(tooManyStructuresPort.errors.some((e) => e.code === 'MAX_MULTI_INPUTS_EXCEEDED')).toBe(true);
  });

  it('encodeGraphToFlow round-trips all 8 WORKFLOW_PRESETS and produces strictly Markdown-safe URLs', () => {
    const sortEdges = (edges: Array<{ fromNode: string; fromPort: string; toNode: string; toPort: string }>) =>
      [...edges].sort((a, b) =>
        `${a.fromNode}.${a.fromPort}->${a.toNode}.${a.toPort}`.localeCompare(
          `${b.fromNode}.${b.fromPort}->${b.toNode}.${b.toPort}`,
        ),
      );
    for (const preset of WORKFLOW_PRESETS) {
      const encoded = encodeGraphToFlow(preset.graph);
      const url = `https://denovo.run/?${encoded.query}`;
      expect(isMarkdownSafeFlowUrl(url), `Unsafe URL for ${preset.id}: ${url}`).toBe(true);
      expect(encoded.query).not.toMatch(/[()[\]<>\s]/);
      expect(encoded.query).not.toMatch(/[.,;:!?_~-]$/);

      const decoded = parseFlowDsl(encoded.dsl);
      expect(decoded.ok, `Failed round-trip for ${preset.id}: ${JSON.stringify(decoded.errors)}`).toBe(true);
      const sortNodes = (nodes: Array<{ id: string; adapterId: string }>) =>
        nodes.map((n) => [n.id, n.adapterId] as [string, string]).sort((a, b) => a[0].localeCompare(b[0]));
      expect(sortNodes(decoded.graph!.nodes)).toEqual(sortNodes(preset.graph.nodes));
      expect(sortEdges(decoded.graph!.edges)).toEqual(sortEdges(preset.graph.edges));
    }

    // Custom label containing parentheses, spaces, and trailing underscore still encodes Markdown-safely
    const tricky = parseFlowDsl('r3:length=90,_label="My Design (v1)_"~m~f').graph!;
    const encTricky = encodeGraphToFlow(tricky);
    const trickyUrl = `https://denovo.run/?${encTricky.query}`;
    expect(isMarkdownSafeFlowUrl(trickyUrl)).toBe(true);
    const decTricky = decodeFlowQuery(trickyUrl);
    expect(decTricky.ok).toBe(true);
    expect(decTricky.nodes[0].label).toBe('My Design (v1)_');
  });

  it('supports edge-safe parseRecipeSafe, encodeCompactShareQuery, decodeWorkflowQuery, and custom catalog hooks', async () => {
    const src = WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-conditioned-cath')!.graph;
    const yaml = graphToYamlRecipe(src);
    const recRes = parseRecipeSafe(yaml);
    expect(recRes.ok).toBe(true);
    expect(recRes.graph?.edges).toEqual(src.edges);

    const compact = await encodeCompactShareQuery(src);
    expect(compact.mode).toBe('preset-diff');
    const decP = await decodeWorkflowQuery(`?${compact.query}`);
    expect(decP.ok).toBe(true);
    expect(decP.edges).toEqual(src.edges);

    const customGraph = parseFlowDsl('r3:length=95~m:numSequences=3~f').graph!;
    const compactG = await encodeCompactShareQuery(customGraph);
    expect(compactG.mode).toBe('compressed-dag');
    const decG = await decodeWorkflowQuery(`?${compactG.query}`);
    expect(decG.ok).toBe(true);
    expect(decG.nodes.map((n) => n.adapterId)).toEqual(['rfd3', 'web-protein-mpnn', 'localfold-af2']);

    // Custom plugin catalog hook (simulating Layer 3 PluginRegistry injection without $lib/)
    const customManifest: AdapterManifest = {
      id: 'custom-dock-plugin',
      displayName: 'Custom Dock',
      shortName: 'Custom Dock',
      stage: 'score',
      runtime: 'cpu-ts',
      estimatedVramMB: 0,
      approxWeightsMB: 0,
      licenseNote: 'MIT',
      description: 'Test plugin',
      inputs: { structure: { type: 'StructureArtifact' } },
      outputs: { metrics: { type: 'DesignabilityMetrics' } },
    };
    const catalog: FlowCatalogHooks = {
      resolveManifest: (id) => (id === 'custom-dock-plugin' ? customManifest : undefined),
      resolveShortCode: (id) => (id === 'custom-dock-plugin' ? 'cd' : undefined),
      resolveFromShortCode: (sc) => (sc === 'cd' ? 'custom-dock-plugin' : undefined),
    };
    const plugRes = parseFlowDsl('r3~cd', { catalog });
    expect(plugRes.ok).toBe(true);
    expect(plugRes.nodes[1].adapterId).toBe('custom-dock-plugin');
    expect(plugRes.edges[0]).toEqual({
      fromNode: 'r3',
      fromPort: 'structure',
      toNode: 'cd',
      toPort: 'structure',
    });
  });

  it('verifies every example URL and character count reported in docs/llm-url-evaluation.md', async () => {
    const ORIGIN = 'https://denovo.run/';
    const benchmarkExamples = [
      {
        flowShort: `${ORIGIN}?flow=r3:length=80~m~f`,
        flowAlias: `${ORIGIN}?flow=rfd3:length=80~mpnn~af2`,
        expectedShortLen: 41,
        expectedAliasLen: 48,
        expectedCanonLen: 31,
        expectedGUrlLen: 142,
        expectedYamlUrlLen: 353,
        expectedJsonUrlLen: 1984,
        prompt:
          'Design an 80-residue protein with RFdiffusion3, design sequences with ProteinMPNN, and fold with AlphaFold2.',
        expectedPromptUrlLen: 165,
      },
      {
        flowShort: `${ORIGIN}?flow=src:p:pdbId=1UBQ~r3:length=80~cmp;src~r:length=80~cmp`,
        flowAlias: `${ORIGIN}?flow=src:pdb:pdbId=1UBQ~rfd3:length=80~compare;src~rfd4:length=80~compare`,
        expectedShortLen: 78,
        expectedAliasLen: 93,
        expectedCanonLen: 82,
        expectedGUrlLen: 221,
        expectedYamlUrlLen: 585,
        expectedJsonUrlLen: 2960,
        prompt:
          'Load PDB 1UBQ, run RFdiffusion3 (length 80) and RFD4-Proteina (length 80) in parallel, and compare both structures in 3D Compare.',
        expectedPromptUrlLen: 198,
      },
      {
        flowShort: `${ORIGIN}?flow=r3:length=70~m:numSequences=4~f~s`,
        flowAlias: `${ORIGIN}?flow=rfd3:length=70~mpnn:numSequences=4~af2~scorer`,
        expectedShortLen: 58,
        expectedAliasLen: 70,
        expectedCanonLen: 63,
        expectedGUrlLen: 205,
        expectedYamlUrlLen: 543,
        expectedJsonUrlLen: 2707,
        prompt:
          'Generate a 70-aa backbone with RFdiffusion3, design 4 sequences with ProteinMPNN, fold with AlphaFold2, and score self-consistency scRMSD against the RFD3 backbone.',
        expectedPromptUrlLen: 239,
      },
      {
        flowShort: `${ORIGIN}?flow=src:p:pdbId=1PRW~motif:r:contig=10-20,A16-35,10-20,_label=Scaffold~pack:la:numSequences=2~fold:f:model=rosettafold3,_label=RF3~score:s;pack.structure~cmp:c.reference;fold.structure~cmp`,
        flowAlias: `${ORIGIN}?flow=src:pdb:pdbId=1PRW~motif:rfd4:contig=10-20,A16-35,10-20,_label=Scaffold~pack:lasermpnn:numSequences=2~fold:rf3,_label=RF3~score:scorer;pack.structure~cmp:compare.reference;fold.structure~cmp`,
        expectedShortLen: 209,
        expectedAliasLen: 215,
        expectedCanonLen: 219,
        expectedGUrlLen: 349,
        expectedYamlUrlLen: 1029,
        expectedJsonUrlLen: 3814,
        prompt:
          'Scaffold motif A16-35 from PDB 1PRW with RFD4-Proteina (contig 10-20,A16-35,10-20), design 2 sequences with LASErMPNN, fold with RoseTTAFold3, score scRMSD, and compare LASErMPNN packed structure (reference) vs RF3 fold in 3D Compare.',
        expectedPromptUrlLen: 335,
      },
    ];

    for (const ex of benchmarkExamples) {
      expect(ex.flowShort.length).toBe(ex.expectedShortLen);
      expect(ex.flowAlias.length).toBe(ex.expectedAliasLen);
      expect(isMarkdownSafeFlowUrl(ex.flowShort)).toBe(true);
      expect(isMarkdownSafeFlowUrl(ex.flowAlias)).toBe(true);

      const parsedShort = decodeFlowQuery(ex.flowShort);
      const parsedAlias = decodeFlowQuery(ex.flowAlias);
      expect(parsedShort.ok).toBe(true);
      expect(parsedAlias.ok).toBe(true);

      const g = parsedShort.graph!;
      const canon = encodeGraphToFlow(g);
      const canonUrl = `${ORIGIN}?${canon.query}`;
      expect(canonUrl.length).toBe(ex.expectedCanonLen);
      expect(isMarkdownSafeFlowUrl(canonUrl)).toBe(true);
      expect(decodeFlowQuery(canonUrl).ok).toBe(true);

      const hash = await encodeGraphToHash(g);
      expect(`${ORIGIN}?${hash.hash.slice(1)}`.length).toBe(ex.expectedGUrlLen);

      const yamlUrl = `${ORIGIN}?recipe=${encodeURIComponent(graphToYamlRecipe(g))}`;
      expect(yamlUrl.length).toBe(ex.expectedYamlUrlLen);

      const jsonUrl = `${ORIGIN}?recipe=${encodeURIComponent(JSON.stringify(g))}`;
      expect(jsonUrl.length).toBe(ex.expectedJsonUrlLen);
      expect(serializeWorkflowJson(g).length).toBeGreaterThan(JSON.stringify(g).length);

      const promptUrl = `${ORIGIN}?prompt=${encodeURIComponent(ex.prompt)}`;
      expect(promptUrl.length).toBe(ex.expectedPromptUrlLen);
    }

    // Preset baseline cited in docs/llm-url-evaluation.md
    const presetGraph = structuredClone(WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-mpnn-refold')!.graph);
    presetGraph.nodes[0].params.length = 90;
    presetGraph.nodes[1].params.numSequences = 4;
    const presetHash = await encodeGraphToHash(presetGraph);
    expect(`${ORIGIN}?${presetHash.hash.slice(1)}`).toBe(
      'https://denovo.run/?p=rfd4-mpnn-refold&gen.length=90&mpnn.numSequences=4',
    );
    expect(`${ORIGIN}?${presetHash.hash.slice(1)}`.length).toBe(72);
    expect(`${ORIGIN}?${encodeGraphToFlow(presetGraph).query}`.length).toBe(150);
  });

  it('enforces edge-vs-client validation parity for PDB, UniProt, sequence, tiedPositions, and customSpecYaml', () => {
    const invalidFlows = [
      'pdb:pdbId=INVALID_PDB~m',
      'pdb:mode=pdb-id~m',
      'pdb:uniprotId=BAD!~m',
      'pdb:mode=alphafold-db~m',
      'seq:sequence=12345_BAD_AA~af2',
      'seq:sequence=""~af2',
      'r3:length=80~m:tiedPositions=NOT_VALID~f',
      'rfd4:customSpecYaml="just plain text without colon"~f',
    ];
    for (const dsl of invalidFlows) {
      const res = parseFlowDsl(dsl);
      expect(res.ok, `Expected ${dsl} to fail`).toBe(false);
      expect(res.errors.some((e) => e.code === 'INVALID_PARAM')).toBe(true);
    }

    const validFlows = [
      'pdb:pdbId=1PRW~m',
      'pdb:uniprotId=P69905~m',
      'seq:sequence=MKTAYIAKQR~af2',
      'r3:length=80~m:tiedPositions="A10=B10"~f',
      'rfd4:customSpecYaml="contig: 70-70"~f',
      'rfd4:customSpecYaml=\'{"contig":"70-70"}\'~f',
    ];
    for (const dsl of validFlows) {
      const res = parseFlowDsl(dsl);
      expect(res.ok, `Expected ${dsl} to pass: ${JSON.stringify(res.errors)}`).toBe(true);
      expect(validateGraph(res.graph!)).toEqual([]);
    }
  });

  it('handles hybrid ?query#hash URLs, id:model,k=v:val lexing, structure-compare 7th input, and ?g= node positions', async () => {
    // 1. Hybrid ?query#hash URLs
    const hybridFlow = decodeFlowQuery('https://denovo.run/?utm_source=chatgpt.com#flow=r3:length=80~m~f');
    expect(hybridFlow.ok).toBe(true);
    expect(hybridFlow.nodes.map((n) => n.adapterId)).toEqual(['rfd3', 'web-protein-mpnn', 'localfold-af2']);

    const hybridWorkflowFlow = await decodeWorkflowQuery(
      'https://denovo.run/?utm_source=chatgpt.com#flow=r3:length=80~m~f',
    );
    expect(hybridWorkflowFlow.ok).toBe(true);

    const hybridPreset = await decodeWorkflowQuery(
      'https://denovo.run/?utm_source=chatgpt.com#p=rfd4-mpnn-refold',
    );
    expect(hybridPreset.ok).toBe(true);
    expect(hybridPreset.nodes.length).toBeGreaterThan(0);

    // 2. id:model,k=v:val lexing when value contains unquoted colon
    const commaColon = parseFlowDsl('q1:seq,sequence=MKT:ACDE~f:af2');
    expect(commaColon.ok).toBe(true);
    expect(commaColon.nodes[0].id).toBe('q1');
    expect(commaColon.nodes[0].adapterId).toBe('sequence-input');
    expect(commaColon.nodes[0].params.sequence).toBe('MKT:ACDE');

    // 3. structure-compare 7th non-reference input fails MAX_MULTI_INPUTS_EXCEEDED (does not hijack reference)
    const sevenGenerators = parseFlowDsl('a:r3~c;b:r3~c;d:r3~c;e:r3~c;f:r3~c;g:r3~c;h:r3~c');
    expect(sevenGenerators.ok).toBe(false);
    expect(sevenGenerators.errors.some((e) => e.code === 'MAX_MULTI_INPUTS_EXCEEDED')).toBe(true);

    // 6 generators + 1 pdb-input still succeeds (pdb-input auto-wires to reference)
    const sixPlusRef = parseFlowDsl('a:r3~c;b:r3~c;d:r3~c;e:r3~c;f:r3~c;g:r3~c;src:pdb:pdbId=1UBQ~c');
    expect(sixPlusRef.ok).toBe(true);
    expect(sixPlusRef.edges.find((e) => e.fromNode === 'src')?.toPort).toBe('reference');

    // 4. ?g= restores custom obj.p node positions and decodes via hybrid ?utm_source=...#g=...
    const customPosGraph = parseFlowDsl('r3:length=90~m~f').graph!;
    customPosGraph.nodes[0].position = { x: 111, y: 222 };
    customPosGraph.nodes[1].position = { x: 333, y: 444 };
    customPosGraph.nodes[2].position = { x: 555, y: 666 };
    const compactG = await encodeCompactShareQuery(customPosGraph);
    const decodedPos = await decodeWorkflowQuery(`https://denovo.run/?utm_source=chatgpt.com#${compactG.query}`);
    expect(decodedPos.ok).toBe(true);
    expect(decodedPos.graph?.nodes[0].position).toEqual({ x: 111, y: 222 });
    expect(decodedPos.graph?.nodes[1].position).toEqual({ x: 333, y: 444 });
    expect(decodedPos.graph?.nodes[2].position).toEqual({ x: 555, y: 666 });

    // 5. ?recipe= unfinished recipe parity (1-node localfold-af2 YAML allowed by default, rejected if requireRunnable: true)
    const incompleteYaml = 'denovo: v1\nname: "Draft"\nnodes:\n  f1: {"model":"localfold-af2"}\nwires:\n';
    const decDraft = await decodeWorkflowQuery(`?recipe=${encodeURIComponent(incompleteYaml)}`);
    expect(decDraft.ok).toBe(true);
    expect(parseRecipeSafe(incompleteYaml, { requireRunnable: true }).ok).toBe(false);
  });

  it('T1-F3-08 & encoding polish: escapes ()[]<>!\'*, ~~, non-BMP emoji, trailing [.,;!?], non-DSL node IDs, and empty graphs', () => {
    // 1. Explicit T1-F3-08 escaping + [label](url) Markdown link + buildShareableWorkflowUrl
    const specialGraph = parseFlowDsl('r3:length=80~m~f:_label="AF2 Model 1 (Native) [v1] <test> !\'*"').graph!;
    const encSpecial = encodeGraphToFlow(specialGraph);
    const builtUrl = buildShareableWorkflowUrl('https://denovo.run', '/', encSpecial.query);
    expect(builtUrl.startsWith('https://denovo.run/?flow=')).toBe(true);
    expect(builtUrl).not.toMatch(/[()[\]<>\s!*']/);
    for (const seq of ['%28', '%29', '%5B', '%5D', '%3C', '%3E', '%21', '%27', '%2A']) {
      expect(builtUrl).toContain(seq);
    }
    expect(isMarkdownSafeFlowUrl(builtUrl)).toBe(true);
    const md = `[Open workflow](${builtUrl})`;
    const match = md.match(/\[[^\]]+\]\(([^)\s]+)\)/);
    expect(match?.[1]).toBe(builtUrl);
    const decSpecial = decodeFlowQuery(builtUrl);
    expect(decSpecial.ok).toBe(true);
    expect(decSpecial.nodes[2].label).toBe("AF2 Model 1 (Native) [v1] <test> !'*");

    // 2. Non-BMP Unicode / emoji surrogate pair in label
    const emojiGraph = parseFlowDsl('r3:length=80,_label="Helix 🧬"~m~f').graph!;
    const encEmoji = encodeGraphToFlow(emojiGraph);
    const emojiUrl = buildShareableWorkflowUrl('https://denovo.run', '/', encEmoji.query);
    expect(isMarkdownSafeFlowUrl(emojiUrl)).toBe(true);
    const decEmoji = decodeFlowQuery(emojiUrl);
    expect(decEmoji.ok).toBe(true);
    expect(decEmoji.nodes[0].label).toBe('Helix 🧬');

    // 3. Values ending in [.,;!?] are quoted so trailing punctuation is preserved on round-trip
    for (const lbl of ['v1.', 'alpha,', 'part;', 'wow!', 'why?']) {
      const g = parseFlowDsl(`r3:_label="${lbl}"~m~f`).graph!;
      const enc = encodeGraphToFlow(g);
      const url = buildShareableWorkflowUrl('https://denovo.run', '/', enc.query);
      expect(isMarkdownSafeFlowUrl(url)).toBe(true);
      const dec = decodeFlowQuery(url);
      expect(dec.ok).toBe(true);
      expect(dec.nodes[0].label).toBe(lbl);
    }

    // 4. `~~` inside label/param is escaped to avoid GFM strikethrough in bare URLs
    const tildeGraph = parseFlowDsl('r3:_label="a~~b~~c"~m~f').graph!;
    const encTilde = encodeGraphToFlow(tildeGraph);
    expect(encTilde.query).not.toContain('~~');
    expect(encTilde.query).toContain('~%7E');
    expect(isMarkdownSafeFlowUrl(`https://denovo.run/?${encTilde.query}`)).toBe(true);
    expect(decodeFlowQuery(`?${encTilde.query}`).nodes[0].label).toBe('a~~b~~c');

    // 5. Node IDs containing reserved DSL delimiters (`node:1`) sanitize cleanly on encode
    const colonIdGraph = parseFlowDsl('r3~m~f').graph!;
    colonIdGraph.nodes[0].id = 'node:1';
    colonIdGraph.edges[0].fromNode = 'node:1';
    const encColon = encodeGraphToFlow(colonIdGraph);
    const decColon = decodeFlowQuery(`?${encColon.query}`);
    expect(decColon.ok).toBe(true);
    expect(decColon.nodes[0].id).toBe('node_1');

    // 6. Empty graph encodes cleanly with empty query
    const emptyEnc = encodeGraphToFlow({ id: 'empty', name: 'Empty', nodes: [], edges: [] });
    expect(emptyEnc.query).toBe('');
    expect(emptyEnc.length).toBe(0);
    expect(buildShareableWorkflowUrl('https://denovo.run', '/', emptyEnc.query)).toBe('https://denovo.run/');
  });

  it('supports %3B, unquoted atom-map semicolons, pipe separators, and universal steps= aliases across generators', async () => {
    // 1. Unquoted multi-residue atom map directly in parseFlowDsl (even across multi-branch RFD3 vs RFD4)
    const rawEnzyme = parseFlowDsl(
      'src:pdb:pdbId=1CTT~r3:unindex=A102,A104,fixedAtoms=A102:ND1,CG,CE1;A104:OE1,CD~m1:ligandmpnn;src~r4:rfd4:motifSpan=A102,A104,unindexedMotif=true,motifAtoms=A102:ND1,CG,CE1;A104:OE1,CD~m2:ligandmpnn',
    );
    expect(rawEnzyme.ok, JSON.stringify(rawEnzyme.errors)).toBe(true);
    expect(rawEnzyme.graph?.nodes.find((n) => n.id === 'r3')?.params.fixedAtoms).toBe('A102:ND1,CG,CE1;A104:OE1,CD');
    expect(rawEnzyme.graph?.nodes.find((n) => n.id === 'r4')?.params.motifAtoms).toBe('A102:ND1,CG,CE1;A104:OE1,CD');

    // 2. Percent-encoded `%3B` preserved across URL query decoding + lossless encodeGraphToFlow round-trip
    const pctUrl =
      'https://denovo.run/?flow=src:pdb:pdbId=1CTT~r3:length=180,steps=50,seed=42,unindex=A102,A104,fixedAtoms=A102:ND1,CG,CE1%3BA104:OE1,CD~lm1:ligandmpnn;src~r4:rfd4:length=180,steps=50,earlyStop=false,seed=42,motifSpan=A102,A104,unindexedMotif=true,motifAtoms=A102:ND1,CG,CE1%3BA104:OE1,CD~lm2:ligandmpnn';
    const pctDecoded = decodeFlowQuery(pctUrl);
    expect(pctDecoded.ok, JSON.stringify(pctDecoded.errors)).toBe(true);
    expect(pctDecoded.graph?.nodes.find((n) => n.id === 'r3')?.params.fixedAtoms).toBe('A102:ND1,CG,CE1;A104:OE1,CD');
    expect(pctDecoded.graph?.nodes.find((n) => n.id === 'r4')?.params.motifAtoms).toBe('A102:ND1,CG,CE1;A104:OE1,CD');
    expect(pctDecoded.graph?.nodes.find((n) => n.id === 'r4')?.params.nsteps).toBe(50);

    const reencoded = encodeGraphToFlow(pctDecoded.graph!);
    const roundTrip = decodeFlowQuery(`https://denovo.run/?${reencoded.query}`);
    expect(roundTrip.ok).toBe(true);
    expect(roundTrip.graph?.nodes.find((n) => n.id === 'r3')?.params.fixedAtoms).toBe('A102:ND1,CG,CE1;A104:OE1,CD');

    // 3. Pipe `|` separator also normalizes cleanly to `; `
    const pipeParsed = parseFlowDsl('pdb:pdbId=1CTT~r3:unindex=A102,A104,fixedAtoms=A102:ND1,CG|A104:OE1,CD');
    expect(pipeParsed.ok).toBe(true);
    expect(pipeParsed.graph?.nodes[1].params.fixedAtoms).toBe('A102:ND1,CG; A104:OE1,CD');

    // 4. Universal `steps=` alias maps cleanly to native step keys (`nsteps`, `numSteps`, `nSampleStep`, `steps`)
    const stepAliases = parseFlowDsl(
      'a:rfd4:steps=55;b:boltzgen:steps=220;c:protpardelle:steps=250;d:rfdiffusion:steps=35;e:genie3:steps=80;f:rfd3:nsteps=60',
    );
    expect(stepAliases.ok, JSON.stringify(stepAliases.errors)).toBe(true);
    const byId = new Map(stepAliases.graph!.nodes.map((n) => [n.id, n.params]));
    expect(byId.get('a')?.nsteps).toBe(55);
    expect(byId.get('b')?.numSteps).toBe(220);
    expect(byId.get('c')?.numSteps).toBe(250);
    expect(byId.get('d')?.numSteps).toBe(35);
    expect(byId.get('e')?.nSampleStep).toBe(80);
    expect(byId.get('f')?.steps).toBe(60);
  });
});
