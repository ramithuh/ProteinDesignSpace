import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { compile, parse } from 'svelte/compiler';
import { afterEach, describe, expect, it } from 'vitest';
import { compileNodeInputPreview, clearInputPreviewCache } from '$lib/engine/input-preview';
import { validateBioCapabilities, validateRuntimeNodeInputs } from '$lib/engine/bio-capabilities';
import { resolveLocalFoldEffectiveConfig } from '$lib/engine/conditioning';
import { isLocalFoldTemplateCapable } from '$lib/engine/localfold-template';
import { computeUploadDigest, validateNodeParams } from '$lib/engine/param-validation';
import { defaultParamsForAdapter, validateGraph } from '$lib/engine/graph-utils';
import { estimateRunRequirements } from '$lib/engine/run-preflight';
import {
  decodeHashToGraph,
  encodeGraphToHash,
  graphToYamlRecipe,
  parseRecipeToGraph,
} from '$lib/engine/share';
import { compileWorkflowPlanToGraph } from '$lib/engine/workflow-planner';
import { createDefaultPluginRegistry } from '$lib/plugins';
import { localfoldAf2Plugin } from '$lib/plugins/localfold-af2';
import { setActivePluginRegistry } from '$lib/kernel';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import {
  buildInspectorTextParamUpdate,
  resolveLocalFoldModelMeta,
} from '$lib/ui/node-display-helpers';
import type { SequenceSet, StructureArtifact, WorkflowGraph } from '$lib/engine/types';

const SEQ_A = 'MKTAYIAKQRQISFVKSHFSRQ';

function makePdb(len = SEQ_A.length, chain = 'A'): string {
  const aa3: Record<string, string> = {
    A: 'ALA', C: 'CYS', D: 'ASP', E: 'GLU', F: 'PHE', G: 'GLY', H: 'HIS',
    I: 'ILE', K: 'LYS', L: 'LEU', M: 'MET', N: 'ASN', P: 'PRO', Q: 'GLN',
    R: 'ARG', S: 'SER', T: 'THR', V: 'VAL', W: 'TRP', Y: 'TYR',
  };
  const lines: string[] = [];
  let serial = 1;
  for (let i = 0; i < len; i++) {
    const ch = SEQ_A[i % SEQ_A.length];
    const r3 = aa3[ch] ?? 'ALA';
    const x = i * 3.8;
    for (const [nm, dx, dy] of [['N', -1.2, -0.7], ['CA', 0, 0], ['C', 1.3, -0.7], ['O', 1.4, -1.9]] as const) {
      lines.push(
        `ATOM  ${String(serial++).padStart(5, ' ')}  ${nm.padEnd(3, ' ')} ${r3} ${chain}${String(i + 1).padStart(4, ' ')}    ${(x + dx).toFixed(3).padStart(8, ' ')}${dy.toFixed(3).padStart(8, ' ')}   0.000  1.00 85.00           ${nm[0]}`,
      );
    }
  }
  lines.push('END');
  return `${lines.join('\n')}\n`;
}

function makeOversizedPdb(): string {
  const pdb = makePdb(30);
  expect(pdb.length).toBeGreaterThan(1500);
  return pdb;
}

function makeStructArtifact(id: string, pdbText: string): StructureArtifact {
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id,
    label: `${id}.pdb`,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    chainIdMap: parsed.chainIdMap,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    geometry: computeGeometryStats(
      parsed.caCoords,
      parsed.chainIds,
      parsed.resNums,
      parsed.insCodes,
    ),
  };
}

describe('LocalFold Template M2 Integration (Share, Validation, Preview, Planner, Preflight, UI)', () => {
  afterEach(() => {
    setActivePluginRegistry(null);
    clearInputPreviewCache();
  });

  it('preserves compact inline templates and strips/restores oversized templates (>1500 chars) in both booted and unbooted registries', async () => {
    const compactPdb = makePdb(4);
    expect(compactPdb.length).toBeLessThanOrEqual(1500);

    const compactGraph: WorkflowGraph = {
      id: 'compact-tpl',
      name: 'Compact Template',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            ...defaultParamsForAdapter('localfold-af2'),
            model: 'boltz2',
            templatePdbText: compactPdb,
            templateFilename: 'compact.pdb',
            templateSpanChains: false,
          },
        },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };

    // Compact roundtrips cleanly unbooted
    setActivePluginRegistry(null);
    const compactEncoded = await encodeGraphToHash(compactGraph);
    expect(compactEncoded.strippedPdb).toBe(false);
    const compactDecoded = await decodeHashToGraph(compactEncoded.hash);
    expect(compactDecoded?.nodes[1].params.templatePdbText).toBe(compactPdb);
    expect(compactDecoded?.nodes[1].params.templateFilename).toBe('compact.pdb');
    expect(compactDecoded?.nodes[1].params.templateSpanChains).toBe(false);

    const bigPdb = makeOversizedPdb();
    const expectedSha = computeUploadDigest(bigPdb);
    const bigGraph: WorkflowGraph = {
      id: 'big-tpl',
      name: 'Oversized Template',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            ...defaultParamsForAdapter('localfold-af2'),
            model: 'monomer-2',
            templatePdbText: bigPdb,
            templateFilename: 'my_template.pdb',
          },
        },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };

    for (const booted of [false, true]) {
      setActivePluginRegistry(booted ? createDefaultPluginRegistry() : null);

      const enc = await encodeGraphToHash(bigGraph);
      expect(enc.strippedPdb).toBe(true);
      const dec = await decodeHashToGraph(enc.hash);
      expect(dec).not.toBeNull();
      const foldNode = dec!.nodes[1];
      expect(foldNode.params.templatePdbText).toBeUndefined();
      expect(foldNode.params.missingTemplateUpload).toBe(true);
      expect(foldNode.params.expectedTemplateSha256).toBe(expectedSha);
      expect(foldNode.params.expectedTemplateFilename).toBe('my_template.pdb');

      // Validation blocks missing shared upload
      expect(validateGraph(dec!).some((e) => /omitted|re-upload/i.test(e))).toBe(true);

      // YAML note & roundtrip
      const yaml = graphToYamlRecipe(bigGraph);
      expect(yaml).toContain('# Note: oversized inline PDB (>1500 chars) omitted');
      const fromYaml = parseRecipeToGraph(yaml);
      expect(fromYaml.nodes[1].params.missingTemplateUpload).toBe(true);
      expect(fromYaml.nodes[1].params.expectedTemplateSha256).toBe(expectedSha);

      // Re-restore via plugin hook restores original text & clears missing metadata
      const restored = localfoldAf2Plugin.share.restoreOmittedInlineAsset!(
        bigGraph.nodes[1].params,
        fromYaml.nodes[1].params,
        yaml,
        'fold',
      );
      expect(restored).not.toBeNull();
      expect(restored!.templatePdbText).toBe(bigPdb);
      expect(restored!.templateFilename).toBe('my_template.pdb');
      expect(restored!.missingTemplateUpload).toBeUndefined();
      expect(restored!.expectedTemplateSha256).toBeUndefined();

      // Tampered digest returns null
      const mismatch = localfoldAf2Plugin.share.restoreOmittedInlineAsset!(
        bigGraph.nodes[1].params,
        { ...fromYaml.nodes[1].params, expectedTemplateSha256: '0'.repeat(64) },
        yaml,
        'fold',
      );
      expect(mismatch).toBeNull();
    }
  });

  it('validates expectedTemplateSha256 format and digest mismatches in validateNodeParams', () => {
    const pdb = makePdb(8);
    const sha = computeUploadDigest(pdb);

    expect(
      validateNodeParams({
        id: 'lf1',
        adapterId: 'localfold-af2',
        label: 'LocalFold',
        params: { model: 'monomer', expectedTemplateSha256: 'not-hex' },
      }),
    ).toEqual([expect.stringMatching(/Invalid expectedTemplateSha256/)]);

    expect(
      validateNodeParams({
        id: 'lf1',
        adapterId: 'localfold-af2',
        label: 'LocalFold',
        params: { model: 'monomer', expectedTemplateSha256: sha, templatePdbText: '' },
      }),
    ).toEqual([expect.stringMatching(/Expected inline template.*is missing/)]);

    expect(
      validateNodeParams({
        id: 'lf1',
        adapterId: 'localfold-af2',
        label: 'LocalFold',
        params: { model: 'monomer', expectedTemplateSha256: 'f'.repeat(64), templatePdbText: pdb },
      }),
    ).toEqual([expect.stringMatching(/does not match expected inline template/)]);

    expect(
      validateNodeParams({
        id: 'lf1',
        adapterId: 'localfold-af2',
        label: 'LocalFold',
        params: { model: 'monomer', expectedTemplateSha256: sha, templatePdbText: pdb },
      }),
    ).toEqual([]);
  });

  it('compiles localfold-batch template preview coverage, pending-upstream template port, and template-free errors', () => {
    const pdb = makePdb(SEQ_A.length);
    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 's1',
      parentStructureId: 'seq',
      sourceModelId: 'sequence-input',
      candidates: [{ id: 'c1', sequence: SEQ_A, sourceModelId: 'sequence-input', score: null, globalScore: null }],
    };
    const g: WorkflowGraph = {
      id: 'preview-g',
      name: 'Preview',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>c1\n${SEQ_A}` } },
        { id: 'tpl', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: pdb, filename: 'ref.pdb' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'tpl', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ],
    };

    // Unmaterialized template port reports partial + missingUpstreamNodeIds
    const pendingPreview = compileNodeInputPreview(g, 'fold', { seq: { sequences: seqSet } });
    expect(pendingPreview.availability).toBe('partial');
    expect(pendingPreview.missingUpstreamNodeIds).toEqual(['tpl']);

    // Materialized template port reports ready + coverage summary + 3D structure
    const tplStruct = makeStructArtifact('tpl', pdb);
    const readyPreview = compileNodeInputPreview(g, 'fold', {
      seq: { sequences: seqSet },
      tpl: { structure: tplStruct },
    });
    expect(readyPreview.availability).toBe('ready');
    expect(readyPreview.payload.headline).toContain('Template:');
    expect(readyPreview.payload.compiledSummary).toMatchObject({
      templateSource: 'port',
      templateFormat: 'pdb',
      templateSpanChains: true,
      templateMatchedResidues: SEQ_A.length,
      templateTotalResidues: SEQ_A.length,
      templateCoverage: `${SEQ_A.length}/${SEQ_A.length} (100%)`,
    });
    expect(readyPreview.previewStructurePdb).toBe(pdb);

    // Template-free model reports error
    const freeGraph: WorkflowGraph = {
      ...g,
      nodes: [
        g.nodes[0],
        g.nodes[1],
        { ...g.nodes[2], params: { ...g.nodes[2].params, model: 'monomer-4' } },
      ],
    };
    const freePreview = compileNodeInputPreview(freeGraph, 'fold', {
      seq: { sequences: seqSet },
      tpl: { structure: tplStruct },
    });
    expect(freePreview.availability).toBe('error');
    expect(freePreview.issues.some((i) => i.severity === 'error')).toBe(true);
  });

  it('wires sequence + template edges to localfold-af2 regardless of plan.edges order while preserving single-edge pdb-input compatibility', () => {
    for (const edges of [
      [['seq', 'fold'], ['tpl', 'fold']] as Array<[string, string]>,
      [['tpl', 'fold'], ['seq', 'fold']] as Array<[string, string]>,
    ]) {
      const res = compileWorkflowPlanToGraph({
        kind: 'block_graph',
        mode: 'new',
        summary: 'Template-conditioned Boltz-2 folding',
        nodes: [
          { id: 'seq', adapter: 'sequence-input', label: 'Sequence', params: { sequence: `>q\n${SEQ_A}` } },
          { id: 'tpl', adapter: 'pdb-input', label: 'Template PDB', params: { mode: 'demo-bundle' } },
          { id: 'fold', adapter: 'localfold-af2', label: 'Boltz-2', params: { model: 'boltz2' } },
        ],
        edges,
      });
      expect(res.graph.edges).toEqual(
        expect.arrayContaining([
          { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
          { fromNode: 'tpl', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
        ]),
      );
      expect(validateGraph(res.graph)).toEqual([]);
    }

    const singlePdb = compileWorkflowPlanToGraph({
      kind: 'block_graph',
      mode: 'new',
      summary: 'Single PDB to AF2',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: 'PDB Input', params: { mode: 'demo-bundle' } },
        { id: 'fold', adapter: 'localfold-af2', label: 'AF2', params: { model: 'monomer' } },
      ],
      edges: [['pdb', 'fold']],
    });
    expect(singlePdb.graph.edges).toEqual([
      { fromNode: 'pdb', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
    ]);
  });

  it('estimates run preflight cleanly for templated LocalFold workflows including AF2 delta deduplication', () => {
    const graph: WorkflowGraph = {
      id: 'preflight-tpl',
      name: 'Preflight Template',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>a\n${SEQ_A}\n>b\n${SEQ_A}` } },
        { id: 'tpl', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'f1', adapterId: 'localfold-af2', params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer' } },
        { id: 'f2', adapterId: 'localfold-af2', params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'f1', toPort: 'sequences' },
        { fromNode: 'tpl', fromPort: 'structure', toNode: 'f1', toPort: 'template' },
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'f2', toPort: 'sequences' },
        { fromNode: 'tpl', fromPort: 'structure', toNode: 'f2', toPort: 'template' },
      ],
    };
    const est = estimateRunRequirements(graph);
    expect(est.totalDownloadMB).toBeLessThan(200);
    expect(est.models.some((m) => m.key === 'localfold:monomer')).toBe(true);
    expect(est.models.some((m) => m.key === 'localfold:monomer-2' && m.approxMB < 50)).toBe(true);
  });

  it('parses NodeInspector.svelte and LocalFoldWidget.svelte cleanly into Svelte 5 ASTs with template controls', () => {
    const widgetPath = resolve(__dirname, '../ui/inspector/LocalFoldWidget.svelte');
    const inspectorPath = resolve(__dirname, '../ui/NodeInspector.svelte');
    const widgetSrc = readFileSync(widgetPath, 'utf8');
    const inspectorSrc = readFileSync(inspectorPath, 'utf8');

    expect(parse(widgetSrc, { modern: true }).fragment).toBeDefined();
    expect(parse(inspectorSrc, { modern: true }).fragment).toBeDefined();

    expect(widgetSrc).toContain('templatePdbText');
    expect(widgetSrc).toContain('templateSpanChains');
    expect(widgetSrc).toContain('isLocalFoldTemplateCapable');
    expect(widgetSrc).toContain('Connected <code>template</code> port');
    expect(widgetSrc).toContain('Fold all');

    const callIdx = inspectorSrc.indexOf('<LocalFoldWidget');
    expect(callIdx).toBeGreaterThan(-1);
    const callSlice = inspectorSrc.slice(callIdx, callIdx + 800);
    expect(callSlice.toLowerCase()).toContain('template');
  });

  it('Fix 1 & Fix 2: wires rfd4-proteina + pdb-input and edit-mode add_edge ops onto localfold-af2 regardless of order', () => {
    // Fix 1: mode === 'new' with multi-output rfd4-proteina + pdb-input in both edge orders
    for (const edges of [
      [['rfd4', 'fold'], ['pdb', 'fold']] as Array<[string, string]>,
      [['pdb', 'fold'], ['rfd4', 'fold']] as Array<[string, string]>,
    ]) {
      const res = compileWorkflowPlanToGraph({
        kind: 'block_graph',
        mode: 'new',
        summary: 'RFD4 design + PDB template -> LocalFold',
        nodes: [
          { id: 'rfd4', adapter: 'rfd4-proteina', label: 'RFD4', params: { mode: 'unconditional', length: 60 } },
          { id: 'pdb', adapter: 'pdb-input', label: 'Template', params: { mode: 'demo-bundle' } },
          { id: 'fold', adapter: 'localfold-af2', label: 'Boltz-2', params: { model: 'boltz2' } },
        ],
        edges,
      });
      expect(res.graph.edges).toEqual(
        expect.arrayContaining([
          { fromNode: 'rfd4', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
          { fromNode: 'pdb', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
        ]),
      );
      expect(validateGraph(res.graph)).toEqual([]);
    }

    // Explicit port hints override default priority when provided
    const explicitHintRes = compileWorkflowPlanToGraph({
      kind: 'block_graph',
      mode: 'new',
      summary: 'Explicit port hint test',
      nodes: [
        { id: 'rfd4', adapter: 'rfd4-proteina', label: 'RFD4', params: { mode: 'unconditional', length: 60 } },
        { id: 'pdb', adapter: 'pdb-input', label: 'Template', params: { mode: 'demo-bundle' } },
        { id: 'fold', adapter: 'localfold-af2', label: 'Boltz-2', params: { model: 'boltz2' } },
      ],
      edges: [
        ['pdb', 'fold', 'sequences', 'sequences'],
        ['rfd4', 'fold', 'structure', 'template'],
      ],
    });
    expect(explicitHintRes.graph.edges).toEqual(
      expect.arrayContaining([
        { fromNode: 'pdb', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'rfd4', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ]),
    );
    expect(validateGraph(explicitHintRes.graph)).toEqual([]);

    // Fix 2: mode === 'edit' with add_edge(pdb -> fold) before add_edge(seq -> fold)
    const baseSeqEditGraph: WorkflowGraph = {
      id: 'edit-seq-tpl',
      name: 'Edit Seq Tpl',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' } },
      ],
      edges: [],
    };
    const editSeqRes = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        summary: 'Wire template then sequence to LocalFold',
        editOps: [
          { action: 'add_edge', fromId: 'pdb', toId: 'fold' },
          { action: 'add_edge', fromId: 'seq', toId: 'fold' },
        ],
      },
      baseSeqEditGraph,
    );
    expect(editSeqRes.graph.edges).toEqual(
      expect.arrayContaining([
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ]),
    );
    expect(validateGraph(editSeqRes.graph)).toEqual([]);

    // Fix 2: mode === 'edit' with add_edge(pdb -> fold) before add_edge(rfd4 -> fold)
    const baseRfdEditGraph: WorkflowGraph = {
      id: 'edit-rfd-tpl',
      name: 'Edit Rfd Tpl',
      nodes: [
        { id: 'rfd4', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 60 } },
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { ...defaultParamsForAdapter('localfold-af2'), model: 'boltz2' } },
      ],
      edges: [],
    };
    const editRfdRes = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        summary: 'Wire template then rfd4 to LocalFold',
        editOps: [
          { action: 'add_edge', fromId: 'pdb', toId: 'fold' },
          { action: 'add_edge', fromId: 'rfd4', toId: 'fold' },
        ],
      },
      baseRfdEditGraph,
    );
    expect(editRfdRes.graph.edges).toEqual(
      expect.arrayContaining([
        { fromNode: 'rfd4', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ]),
    );
    expect(validateGraph(editRfdRes.graph)).toEqual([]);
  });

  it('Fix 3: validateGraph allows valid wired template port even when leftover invalid inline templatePdbText is present', () => {
    const validPdb = makePdb(10);
    const graphWithWiredPortAndStaleInline: WorkflowGraph = {
      id: 'wired-overrides-stale-inline',
      name: 'Wired Overrides Stale Inline',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        { id: 'tpl', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: validPdb, filename: 'ref.pdb' } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            ...defaultParamsForAdapter('localfold-af2'),
            model: 'monomer-2',
            templatePdbText: 'NOT_A_VALID_PDB_LEFTOVER',
          },
        },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'tpl', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ],
    };
    expect(validateGraph(graphWithWiredPortAndStaleInline)).toEqual([]);

    // Without the wired template edge, invalid inline text is still rejected
    const unwiredInvalid: WorkflowGraph = {
      ...graphWithWiredPortAndStaleInline,
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    expect(validateGraph(unwiredInvalid).some((e) => /Invalid inline template structure/.test(e))).toBe(true);

    // Wired template port on a template-free model is still rejected
    const wiredTemplateFree: WorkflowGraph = {
      ...graphWithWiredPortAndStaleInline,
      nodes: [
        graphWithWiredPortAndStaleInline.nodes[0],
        graphWithWiredPortAndStaleInline.nodes[1],
        {
          ...graphWithWiredPortAndStaleInline.nodes[2],
          params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-4' },
        },
      ],
    };
    expect(validateGraph(wiredTemplateFree).some((e) => /template-free/i.test(e))).toBe(true);
  });

  it('Fix 4 & Fix 5: flushes pending text commits on Model switch / 1-click switch / setParam and deletes optional keys in clearInlineTemplate', async () => {
    const widgetPath = resolve(__dirname, '../ui/inspector/LocalFoldWidget.svelte');
    const inspectorPath = resolve(__dirname, '../ui/NodeInspector.svelte');
    const widgetSrc = readFileSync(widgetPath, 'utf8');
    const inspectorSrc = readFileSync(inspectorPath, 'utf8');

    // Fix 4: Model <select> onchange and 1-click template-free switch onclick both call onFlushPendingTextCommit() before sanitizeParamsOnModeSwitch
    const modelSelectBlock = widgetSrc.match(/<select[\s\S]*?value=\{lfModelId\}[\s\S]*?onchange=\{[\s\S]*?\}\}[\s\S]*?>/);
    expect(modelSelectBlock?.[0]).toBeTruthy();
    expect(modelSelectBlock![0]).toMatch(/onFlushPendingTextCommit\(\)[\s\S]*?sanitizeParamsOnModeSwitch/);

    const quickSwitchBlock = widgetSrc.match(/hasAnyTemplate && !lfIsTemplateCapable[\s\S]*?onclick=\{[\s\S]*?Switch to \{suggestedTemplateModel\.label\}/);
    expect(quickSwitchBlock?.[0]).toBeTruthy();
    expect(quickSwitchBlock![0]).toMatch(/onFlushPendingTextCommit\(\)[\s\S]*?sanitizeParamsOnModeSwitch/);

    // Fix 4: setParam in NodeInspector flushes pending text commit before reading node.params
    const setParamBlock = inspectorSrc.match(/function setParam\([\s\S]*?\n  \}/);
    expect(setParamBlock?.[0]).toBeTruthy();
    expect(setParamBlock![0]).toContain('flushPendingTextCommit()');

    // Fix 5: clearInlineTemplate & clearTemplateMissingExpectation & buildTextParamUpdate delete optional template keys instead of leaving ''/false
    const clearInlineBlock = inspectorSrc.match(/function clearInlineTemplate\(\) \{([\s\S]*?)\n  \}/);
    expect(clearInlineBlock?.[1]).toBeTruthy();
    expect(clearInlineBlock![1]).toContain('delete next.templatePdbText');
    expect(clearInlineBlock![1]).toContain('delete next.templateFilename');
    expect(clearInlineBlock![1]).toContain('delete next.missingTemplateUpload');
    expect(clearInlineBlock![1]).toContain('delete next.expectedTemplateSha256');
    expect(clearInlineBlock![1]).toContain('delete next.expectedTemplateFilename');
    expect(clearInlineBlock![1]).not.toContain("next.templatePdbText = ''");
    expect(clearInlineBlock![1]).not.toContain("next.templateFilename = ''");
    expect(clearInlineBlock![1]).not.toContain('next.missingTemplateUpload = false');

    // Verify share hash parity: clearing template keys restores exact baseline untemplated #g= hash
    const baselineGraph: WorkflowGraph = {
      id: 'hash-parity',
      name: 'Hash Parity',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        { id: 'fold', adapterId: 'localfold-af2', params: { ...defaultParamsForAdapter('localfold-af2') } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const clearedParams: Record<string, unknown> = {
      ...defaultParamsForAdapter('localfold-af2'),
      templatePdbText: makePdb(6),
      templateFilename: 'tpl.pdb',
      missingTemplateUpload: true,
      expectedTemplateSha256: 'a'.repeat(64),
      expectedTemplateFilename: 'tpl.pdb',
    };
    delete clearedParams.templatePdbText;
    delete clearedParams.templateFilename;
    delete clearedParams.missingTemplateUpload;
    delete clearedParams.expectedTemplateSha256;
    delete clearedParams.expectedTemplateFilename;

    const clearedGraph: WorkflowGraph = {
      ...baselineGraph,
      nodes: [baselineGraph.nodes[0], { ...baselineGraph.nodes[1], params: clearedParams }],
    };
    const baseHash = await encodeGraphToHash(baselineGraph);
    const clearedHash = await encodeGraphToHash(clearedGraph);
    expect(clearedHash.hash).toBe(baseHash.hash);
  });

  it('Fix 6: compileNodeInputPreview promotes availability to error for un-run template port on template-free model while keeping template-capable partial', () => {
    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 's1',
      parentStructureId: 'seq',
      sourceModelId: 'sequence-input',
      candidates: [{ id: 'c1', sequence: SEQ_A, sourceModelId: 'sequence-input', score: null, globalScore: null }],
    };
    const makeGraph = (model: string, extraParams: Record<string, unknown> = {}): WorkflowGraph => ({
      id: `preview-unrun-${model}`,
      name: 'Preview Unrun Template',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>c1\n${SEQ_A}` } },
        { id: 'tpl', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { ...defaultParamsForAdapter('localfold-af2'), model, ...extraParams } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'tpl', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ],
    });

    // Template-capable un-run port stays 'partial'
    const capablePending = compileNodeInputPreview(makeGraph('monomer-2'), 'fold', { seq: { sequences: seqSet } });
    expect(capablePending.availability).toBe('partial');
    expect(capablePending.missingUpstreamNodeIds).toEqual(['tpl']);

    // Template-capable un-run port with stale invalid inline text stays 'partial' (port overrides inline)
    const capableWithStaleInline = compileNodeInputPreview(
      makeGraph('monomer-2', { templatePdbText: 'NOT_VALID_LEFTOVER' }),
      'fold',
      { seq: { sequences: seqSet } },
    );
    expect(capableWithStaleInline.availability).toBe('partial');

    // Template-free un-run port promotes to 'error'
    const freePending = compileNodeInputPreview(makeGraph('monomer-4'), 'fold', { seq: { sequences: seqSet } });
    expect(freePending.availability).toBe('error');
    expect(freePending.issues.some((i) => i.severity === 'error')).toBe(true);
    expect(freePending.missingUpstreamNodeIds).toEqual(['tpl']);
  });

  it('M2 Iteration 3 Fix 1: edit-mode remove_node preserves deferred localfold-af2 edge ordering while still splicing top-k-filter', () => {
    // 1. add_node(fold) -> add_edge(pdb -> fold) -> remove_node(stale) -> add_edge(seq -> fold)
    const graphWithStale: WorkflowGraph = {
      id: 'edit-stale-node',
      name: 'Edit With Stale Node',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'stale', adapterId: 'sequence-input', params: { sequence: `>stale\n${SEQ_A}` } },
      ],
      edges: [],
    };
    const res1 = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        summary: 'Add fold, wire pdb & seq, remove stale node',
        editOps: [
          { action: 'add_node', nodeId: 'fold', adapter: 'localfold-af2', params: { model: 'monomer-2' } },
          { action: 'add_edge', fromId: 'pdb', toId: 'fold' },
          { action: 'remove_node', nodeId: 'stale' },
          { action: 'add_edge', fromId: 'seq', toId: 'fold' },
        ],
      },
      graphWithStale,
    );
    expect(res1.graph.nodes.some((n) => n.id === 'stale')).toBe(false);
    expect(res1.graph.edges).toEqual(
      expect.arrayContaining([
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ]),
    );
    expect(validateGraph(res1.graph)).toEqual([]);

    // 2. top-k-filter removal splices incoming/outgoing edges cleanly while keeping unrelated localfold-af2 pending edges deferred
    const graphWithTopK: WorkflowGraph = {
      id: 'edit-topk-splice',
      name: 'Edit With Top-K Splice',
      nodes: [
        { id: 'tpl', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        { id: 'topk', adapterId: 'top-k-filter', params: { topK: 2 } },
        { id: 'fold', adapterId: 'localfold-af2', params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'topk', toPort: 'sequences' },
        { fromNode: 'topk', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const res2 = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        summary: 'Add template edge and splice out Top-K',
        editOps: [
          { action: 'add_edge', fromId: 'tpl', toId: 'fold' },
          { action: 'remove_node', nodeId: 'topk' },
        ],
      },
      graphWithTopK,
    );
    expect(res2.graph.nodes.some((n) => n.id === 'topk')).toBe(false);
    expect(res2.graph.edges).toEqual(
      expect.arrayContaining([
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'tpl', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ]),
    );
    expect(validateGraph(res2.graph)).toEqual([]);

    // 3. Pending edges referencing top-k-filter AND pending pdb->fold on localfold-af2 both resolve cleanly
    const graphForPendingTopK: WorkflowGraph = {
      id: 'edit-pending-topk',
      name: 'Pending TopK',
      nodes: [
        { id: 'tpl', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        { id: 'topk', adapterId: 'top-k-filter', params: { topK: 2 } },
        { id: 'fold', adapterId: 'localfold-af2', params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'topk', toPort: 'sequences' }],
    };
    const res3 = compileWorkflowPlanToGraph(
      {
        kind: 'block_graph',
        mode: 'edit',
        summary: 'Wire tpl->fold and topk->fold then remove topk',
        editOps: [
          { action: 'add_edge', fromId: 'tpl', toId: 'fold' },
          { action: 'add_edge', fromId: 'topk', toId: 'fold' },
          { action: 'remove_node', nodeId: 'topk' },
        ],
      },
      graphForPendingTopK,
    );
    expect(res3.graph.edges).toEqual(
      expect.arrayContaining([
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'tpl', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ]),
    );
    expect(validateGraph(res3.graph)).toEqual([]);
  });

  it('M2 Iteration 3 Fix 2: validateGraph ignores leftover expectedTemplateSha256 when template port is wired', () => {
    const validPdb = makePdb(8);
    const makeWiredGraph = (extraFoldParams: Record<string, unknown>): WorkflowGraph => ({
      id: 'wired-overrides-sha',
      name: 'Wired Overrides SHA',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        { id: 'tpl', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: validPdb, filename: 'ref.pdb' } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            ...defaultParamsForAdapter('localfold-af2'),
            model: 'monomer-2',
            ...extraFoldParams,
          },
        },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'tpl', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ],
    });

    // Leftover expectedTemplateSha256 without missingTemplateUpload
    expect(
      validateGraph(
        makeWiredGraph({
          expectedTemplateSha256: 'a'.repeat(64),
          expectedTemplateFilename: 'shared_tpl.pdb',
        }),
      ),
    ).toEqual([]);

    // Leftover mismatched templatePdbText + expectedTemplateSha256 + missingTemplateUpload
    expect(
      validateGraph(
        makeWiredGraph({
          templatePdbText: validPdb,
          expectedTemplateSha256: 'b'.repeat(64),
          expectedTemplateFilename: 'shared_tpl.pdb',
          missingTemplateUpload: true,
        }),
      ),
    ).toEqual([]);

    // Leftover malformed expectedTemplateSha256
    expect(
      validateGraph(
        makeWiredGraph({
          expectedTemplateSha256: 'NOT_HEX_64',
        }),
      ),
    ).toEqual([]);
  });

  it('M2 Iteration 3 Fix 3: LocalFoldWidget & NodeInspector flush pending text commits and read flushed.params in all handlers', () => {
    const widgetPath = resolve(__dirname, '../ui/inspector/LocalFoldWidget.svelte');
    const inspectorPath = resolve(__dirname, '../ui/NodeInspector.svelte');
    const widgetSrc = readFileSync(widgetPath, 'utf8');
    const inspectorSrc = readFileSync(inspectorPath, 'utf8');

    expect(compile(widgetSrc, { generate: 'client' }).js.code).toBeTruthy();
    expect(compile(inspectorSrc, { generate: 'client' }).js.code).toBeTruthy();

    // Zero un-flushed onUpdate(node.id, { ...node.params, ... }) remain in LocalFoldWidget.svelte
    expect(widgetSrc).not.toMatch(/onUpdate\(\s*node\.id\s*,\s*\{\s*\.\.\.node\.params/);
    expect(widgetSrc).toContain('function commitPatch(');
    expect(widgetSrc).toContain('onFlushPendingTextCommit()');
    expect(widgetSrc).toContain('Fold all');

    // NodeInspector template handlers capture flushed.params
    for (const fnName of [
      'clearTemplateMissingExpectation',
      'clearInlineTemplate',
      'handleTemplateFileUpload',
    ]) {
      const fnMatch = inspectorSrc.match(
        new RegExp(`(?:async\\s+)?function ${fnName}\\([^)]*\\)\\s*\\{([\\s\\S]*?)\\n  \\}`),
      );
      expect(fnMatch?.[1]).toBeTruthy();
      expect(fnMatch![1]).toContain('const flushed = flushPendingTextCommit()');
      expect(fnMatch![1]).toContain('flushed.params');
    }
  });

  it('Tier 5 adversarial (Gaps 1, 4, 5, 6, 8): runtime upload guards, modelKey.trim(), inspector flush handlers, static sequence-input cap parity, and stale missingTemplateUpload cleanup', async () => {
    const pdb = makePdb(SEQ_A.length);
    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'seq-t5',
      parentStructureId: 'none',
      sourceModelId: 'sequence-input',
      candidates: [
        { id: 'c1', sequence: SEQ_A, sourceModelId: 'sequence-input', score: null, globalScore: null },
      ],
    };

    // Gap 1: validateRuntimeNodeInputs enforces missingTemplateUpload & expectedTemplateSha256 when port is absent, allows live port override
    expect(() =>
      validateRuntimeNodeInputs(
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            model: 'monomer',
            missingTemplateUpload: true,
            expectedTemplateFilename: 'lost.pdb',
            expectedTemplateSha256: 'a'.repeat(64),
          },
        },
        { sequences: seqSet },
      ),
    ).toThrow(/lost\.pdb|omitted/i);

    expect(() =>
      validateRuntimeNodeInputs(
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            model: 'monomer',
            templatePdbText: pdb,
            expectedTemplateSha256: '0'.repeat(64),
          },
        },
        { sequences: seqSet },
      ),
    ).toThrow(/does not match/i);

    expect(() =>
      validateRuntimeNodeInputs(
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            model: 'monomer',
            missingTemplateUpload: true,
            templatePdbText: pdb,
            expectedTemplateSha256: '0'.repeat(64),
          },
        },
        { sequences: seqSet, template: makeStructArtifact('tpl', pdb) },
      ),
    ).not.toThrow();

    // Gap 4: modelKey.trim() consistency across capability, conditioning, and bio-capabilities
    expect(isLocalFoldTemplateCapable(' monomer ')).toBe(true);
    expect(isLocalFoldTemplateCapable(' boltz2 ')).toBe(true);
    const trimmedMonoCfg = resolveLocalFoldEffectiveConfig({
      modelKey: ' monomer ',
      params: { model: ' monomer ', templatePdbText: pdb },
    });
    expect(trimmedMonoCfg.isAf2Mono).toBe(true);
    expect(trimmedMonoCfg.isTemplateCapable).toBe(true);
    expect(trimmedMonoCfg.templateFamily).toBe('af2-monomer');

    const trimmedBoltzCfg = resolveLocalFoldEffectiveConfig({
      modelKey: ' boltz2 ',
      params: { model: ' boltz2 ', templatePdbText: pdb },
    });
    expect(trimmedBoltzCfg.isAf3).toBe(true);
    expect(trimmedBoltzCfg.templateFamily).toBe('af3');

    // Gap 5: handleA3mFileUpload & handleFileUpload capture flushed.params in NodeInspector.svelte
    const inspectorSrc = readFileSync(resolve(__dirname, '../ui/NodeInspector.svelte'), 'utf8');
    for (const fnName of ['handleA3mFileUpload', 'handleFileUpload']) {
      const fnMatch = inspectorSrc.match(
        new RegExp(`async function ${fnName}\\([^)]*\\)\\s*\\{([\\s\\S]*?)\\n  \\}`),
      );
      expect(fnMatch?.[1]).toBeTruthy();
      expect(fnMatch![1]).toContain('const flushed = flushPendingTextCommit()');
      expect(fnMatch![1]).toContain('flushed.params');
    }

    // Gap 6: validateBioCapabilities static sequence-input check respects maxSequences cap
    const cappedGraph: WorkflowGraph = {
      id: 't5-capped-seq',
      name: 'Capped Sequence Input',
      nodes: [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          params: {
            sequence: `>c1\n${SEQ_A}\n>c2\n${SEQ_A}\n>c3\n${'W'.repeat(25)}`,
          },
        },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            ...defaultParamsForAdapter('localfold-af2'),
            model: 'monomer',
            capMode: 'cap',
            maxSequences: 1,
            templatePdbText: pdb,
          },
        },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    expect(
      validateBioCapabilities(cappedGraph).filter((i) => i.code === 'localfold-template-mismatch'),
    ).toEqual([]);

    // Gap 8: compact inline template + stale missingTemplateUpload drops missingTemplateUpload in share & widget shows Clear inline for filename/missingTemplateUpload
    const compactPdb = makePdb(4);
    expect(compactPdb.length).toBeLessThanOrEqual(1500);
    const prepNoSha = localfoldAf2Plugin.share.prepareForShare!(
      {
        model: 'monomer-2',
        templatePdbText: compactPdb,
        missingTemplateUpload: true,
      },
      { stripLargeUploads: true },
    );
    expect(prepNoSha.stripped).toBe(false);
    expect(prepNoSha.params.missingTemplateUpload).toBeUndefined();

    const prepBadSha = localfoldAf2Plugin.share.prepareForShare!(
      {
        model: 'monomer-2',
        templatePdbText: compactPdb,
        missingTemplateUpload: true,
        expectedTemplateSha256: 'deadbeef',
      },
      { stripLargeUploads: true },
    );
    expect(prepBadSha.stripped).toBe(false);
    expect(prepBadSha.params.missingTemplateUpload).toBeUndefined();
    expect(prepBadSha.params.expectedTemplateSha256).toBe('deadbeef');

    setActivePluginRegistry(null);
    const staleShareGraph: WorkflowGraph = {
      id: 't5-stale-missing',
      name: 'Stale Missing Flag',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            ...defaultParamsForAdapter('localfold-af2'),
            model: 'monomer-2',
            templatePdbText: compactPdb,
            missingTemplateUpload: true,
          },
        },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const encodedStale = await encodeGraphToHash(staleShareGraph);
    expect(encodedStale.strippedPdb).toBe(false);
    const decodedStale = await decodeHashToGraph(encodedStale.hash);
    expect(decodedStale?.nodes[1].params.missingTemplateUpload).toBeUndefined();

    const widgetSrc = readFileSync(
      resolve(__dirname, '../ui/inspector/LocalFoldWidget.svelte'),
      'utf8',
    );
    expect(compile(widgetSrc, { generate: 'client' }).js.code).toBeTruthy();
    expect(widgetSrc).toContain('node.params.templateFilename');
    expect(widgetSrc).toContain('Clear inline');
    expect(widgetSrc).toMatch(/lfTemplateFamily === 'af2-multimer' \|\| lfTemplateFamily === 'af3'/);
  });

  it('Tier 5 R2 adversarial (Findings 3, 4, 5, 6): inspector model trim, pending-upstream error preservation, preflight expectedTemplateSha256 rejection, and missingTemplateUpload re-share/paste digest verification', async () => {
    const compactPdb = makePdb(4);
    const compactSha = computeUploadDigest(compactPdb);
    const otherPdb = makePdb(5);
    const otherSha = computeUploadDigest(otherPdb);
    expect(compactPdb.length).toBeLessThanOrEqual(1500);
    expect(otherSha).not.toBe(compactSha);

    // Finding 3: NodeInspector selectedLocalFoldMeta helper trims padded model IDs
    const boltzMeta = resolveLocalFoldModelMeta({ model: ' boltz2 ' });
    expect(boltzMeta.id).toBe('boltz2');
    expect(boltzMeta.approxMB).toBe(364);
    expect(boltzMeta.family).toBe('af3');

    // Finding 4: compileNodeInputPreview preserves availability='error' when !seqSet && pending sequences edge
    clearInputPreviewCache();
    const pendingTemplateFreeGraph: WorkflowGraph = {
      id: 'r2-preview-err-tplfree',
      name: 'Pending Seq + Template-Free Port',
      nodes: [
        { id: 'gen', adapterId: 'genie3', params: { taskMode: 'unconditional', length: 30 } },
        { id: 'tpl', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-3' },
        },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'structure', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'tpl', fromPort: 'structure', toNode: 'fold', toPort: 'template' },
      ],
    };
    const prevFree = compileNodeInputPreview(pendingTemplateFreeGraph, 'fold', {});
    expect(prevFree.availability).toBe('error');
    expect(prevFree.issues.some((i) => i.severity === 'error')).toBe(true);

    const pendingBadInlineGraph: WorkflowGraph = {
      id: 'r2-preview-err-badinline',
      name: 'Pending Seq + Invalid Inline Template',
      nodes: [
        { id: 'gen', adapterId: 'genie3', params: { taskMode: 'unconditional', length: 30 } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            ...defaultParamsForAdapter('localfold-af2'),
            model: 'monomer',
            templatePdbText: 'not a valid template',
          },
        },
      ],
      edges: [{ fromNode: 'gen', fromPort: 'structure', toNode: 'fold', toPort: 'sequences' }],
    };
    const prevBadInline = compileNodeInputPreview(pendingBadInlineGraph, 'fold', {});
    expect(prevBadInline.availability).toBe('error');
    expect(prevBadInline.issues.some((i) => i.severity === 'error')).toBe(true);

    // Contrast: pending sequences without errors remains 'pending-upstream'
    const pendingCleanGraph: WorkflowGraph = {
      id: 'r2-preview-pending-clean',
      name: 'Pending Seq Clean',
      nodes: [
        { id: 'gen', adapterId: 'genie3', params: { taskMode: 'unconditional', length: 30 } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer' },
        },
      ],
      edges: [{ fromNode: 'gen', fromPort: 'structure', toNode: 'fold', toPort: 'sequences' }],
    };
    expect(compileNodeInputPreview(pendingCleanGraph, 'fold', {}).availability).toBe('pending-upstream');

    // Finding 5: crossValidate & validateBioCapabilities reject orphaned expectedTemplateSha256 on template-free models
    for (const tplFreeModel of ['monomer-3', 'ef2-fast-300m']) {
      const orphanParams = {
        ...defaultParamsForAdapter('localfold-af2'),
        model: tplFreeModel,
        expectedTemplateSha256: 'a'.repeat(64),
      };
      const cvErrs = ADAPTER_REGISTRY['localfold-af2'].crossValidate!(orphanParams);
      expect(cvErrs.some((e) => /template-free/i.test(e))).toBe(true);

      const bioIssues = validateBioCapabilities({
        id: `r2-orphan-${tplFreeModel}`,
        name: 'Orphan Expected SHA',
        nodes: [
          { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
          { id: 'fold', adapterId: 'localfold-af2', params: orphanParams },
        ],
        edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
      });
      const tplUnsupported = bioIssues.find(
        (i) => i.code === 'localfold-template-unsupported-model',
      );
      expect(tplUnsupported?.severity).toBe('error');
      expect(tplUnsupported?.fix?.paramsPatch.model).toBeTruthy();
    }

    // Finding 6a: compact mismatched inline upload with valid 64-hex expectedTemplateSha256 preserves missingTemplateUpload on re-share (booted + unbooted)
    const prepMismatch = localfoldAf2Plugin.share.prepareForShare!(
      {
        model: 'monomer-2',
        templatePdbText: compactPdb,
        missingTemplateUpload: true,
        expectedTemplateSha256: otherSha,
        expectedTemplateFilename: 'expected.pdb',
      },
      { stripLargeUploads: true },
    );
    expect(prepMismatch.params.missingTemplateUpload).toBe(true);
    expect(prepMismatch.params.expectedTemplateSha256).toBe(otherSha);
    expect(prepMismatch.params.expectedTemplateFilename).toBe('expected.pdb');

    // Matching compact upload clears stale expectation metadata cleanly
    const prepMatch = localfoldAf2Plugin.share.prepareForShare!(
      {
        model: 'monomer-2',
        templatePdbText: compactPdb,
        missingTemplateUpload: true,
        expectedTemplateSha256: compactSha,
        expectedTemplateFilename: 'expected.pdb',
      },
      { stripLargeUploads: true },
    );
    expect(prepMatch.params.missingTemplateUpload).toBeUndefined();
    expect(prepMatch.params.expectedTemplateSha256).toBeUndefined();
    expect(prepMatch.params.expectedTemplateFilename).toBeUndefined();

    // Unbooted share.ts fallback parity
    setActivePluginRegistry(null);
    const mismatchShareGraph: WorkflowGraph = {
      id: 'r2-mismatch-reshare',
      name: 'Mismatch Reshare',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: `>q\n${SEQ_A}` } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: {
            ...defaultParamsForAdapter('localfold-af2'),
            model: 'monomer-2',
            templatePdbText: compactPdb,
            missingTemplateUpload: true,
            expectedTemplateSha256: otherSha,
            expectedTemplateFilename: 'expected.pdb',
          },
        },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const encodedMismatch = await encodeGraphToHash(mismatchShareGraph);
    const decodedMismatch = await decodeHashToGraph(encodedMismatch.hash);
    expect(decodedMismatch?.nodes[1].params.missingTemplateUpload).toBe(true);
    expect(decodedMismatch?.nodes[1].params.expectedTemplateSha256).toBe(otherSha);
    expect(decodedMismatch?.nodes[1].params.expectedTemplateFilename).toBe('expected.pdb');

    // Finding 6b: pasting matching templatePdbText into textarea clears missingTemplateUpload & expectedTemplateSha256
    const sharedMissingParams: Record<string, unknown> = {
      model: 'monomer-2',
      missingTemplateUpload: true,
      expectedTemplateSha256: compactSha,
      expectedTemplateFilename: 'expected.pdb',
    };
    const pastedMatch = buildInspectorTextParamUpdate(
      sharedMissingParams,
      'templatePdbText',
      compactPdb,
    );
    expect(pastedMatch.templatePdbText).toBe(compactPdb);
    expect(pastedMatch.missingTemplateUpload).toBeUndefined();
    expect(pastedMatch.expectedTemplateSha256).toBeUndefined();
    expect(pastedMatch.expectedTemplateFilename).toBeUndefined();

    const pastedMismatch = buildInspectorTextParamUpdate(
      sharedMissingParams,
      'templatePdbText',
      otherPdb,
    );
    expect(pastedMismatch.templatePdbText).toBe(otherPdb);
    expect(pastedMismatch.missingTemplateUpload).toBe(true);
    expect(pastedMismatch.expectedTemplateSha256).toBe(compactSha);

    const clearedText = buildInspectorTextParamUpdate(
      { model: 'monomer-2', templatePdbText: compactPdb, templateFilename: 'tpl.pdb' },
      'templatePdbText',
      '',
    );
    expect(clearedText.templatePdbText).toBeUndefined();
    expect(clearedText.templateFilename).toBeUndefined();
  });
});
