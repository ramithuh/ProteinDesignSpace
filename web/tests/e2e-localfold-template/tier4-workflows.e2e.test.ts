import { afterAll, describe, expect, it } from 'vitest';
import { defaultParamsForAdapter, validateGraph } from '$lib/engine/graph-utils';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { clearInputPreviewCache, compileNodeInputPreview } from '$lib/engine/input-preview';
import { estimateRunRequirements } from '$lib/engine/run-preflight';
import {
  decodeHashToGraph,
  encodeGraphToHash,
  graphToYamlRecipe,
  parseRecipeToGraph,
} from '$lib/engine/share';
import { localfoldAf2Plugin } from '$lib/plugins/localfold-af2';
import type { DesignabilityMetrics, WorkflowGraph } from '$lib/engine/types';
import {
  HETERODIMER_CHAIN_A,
  HETERODIMER_CHAIN_B,
  HETERODIMER_SEQUENCE,
  HOMOLOG_MONOMER_SEQUENCE,
  LARGE_TEMPLATE_PDB,
  MODIFIED_MSE_SEP_MMCIF,
  MULTICHAR_COMPLEX_MMCIF,
  STANDARD_MONOMER_PDB,
  STANDARD_MONOMER_SEQUENCE,
  makeLocalFoldNode,
  makeScorerNode,
  makeSequenceNode,
  makeSequenceSetArtifact,
  makeStructureArtifactFromText,
  makeTemplatedLocalFoldGraph,
} from './fixtures';
import {
  cleanupWorkerVendorOrigin,
  diffFloat32Arrays,
  hasLiveWebGpu,
  loadVendorTemplateOracles,
  runAf3TemplateCpuOracle,
  runLocalFoldWorkerInNode,
  runSchedulerWithCapturedLocalFold,
} from './helpers';

afterAll(() => {
  cleanupWorkerVendorOrigin();
});

describe('Tier 4 — Realistic End-User ProteinDesignSpace Workflows', () => {
  it('Workflow 1: Homology-templated monomer folding (pdb-input + sequence-input -> monomer-2 -> designability-scorer)', async () => {
    clearInputPreviewCache();
    const graph = makeTemplatedLocalFoldGraph({
      sequence: HOMOLOG_MONOMER_SEQUENCE,
      templateText: STANDARD_MONOMER_PDB,
      supplyMode: 'port',
      model: 'monomer-2',
      includeScorer: true,
    });
    // Homolog has N-tag + deletion (L=22 vs template L=20), so scorer compares fold self-consistency
    graph.edges = graph.edges.map((e) =>
      e.toNode === 'score' && e.toPort === 'structure'
        ? { ...e, fromNode: 'fold', fromPort: 'structure' }
        : e,
    );

    expect(validateGraph(graph)).toEqual([]);
    const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
    const seqArt = makeSequenceSetArtifact(HOMOLOG_MONOMER_SEQUENCE);
    expect(
      validateBioCapabilities(graph, {
        tpl: { structure: tplArt },
        seq: { sequences: seqArt },
      }).filter((i) => i.severity === 'error'),
    ).toEqual([]);

    const preview = compileNodeInputPreview(graph, 'fold', {
      tpl: { structure: tplArt },
      seq: { sequences: seqArt },
    });
    expect(preview.availability).toBe('ready');
    expect(estimateRunRequirements(graph).models.some((m) => m.key === 'localfold:monomer-2')).toBe(
      true,
    );

    const schedRes = await runSchedulerWithCapturedLocalFold(graph);
    expect(schedRes.workerCalls[0].inputs.template).toBeDefined();
    const metrics = schedRes.outputsByNode.score?.metrics as DesignabilityMetrics | undefined;
    expect(metrics?.kind).toBe('DesignabilityMetrics');

    const oracles = await loadVendorTemplateOracles();
    const templatedRun = await runLocalFoldWorkerInNode({
      params: graph.nodes.find((n) => n.id === 'fold')!.params,
      inputs: { sequences: seqArt, template: tplArt },
    });
    const untemplatedRun = await runLocalFoldWorkerInNode({
      params: graph.nodes.find((n) => n.id === 'fold')!.params,
      inputs: { sequences: seqArt },
    });
    expect(templatedRun.status).toBe('result');
    expect(templatedRun.outputs!.folds.id).not.toBe(untemplatedRun.outputs!.folds.id);
    const tplObj = templatedRun.captures[0]!.predictOpts!.template as { aatype: Int32Array };
    expect(tplObj.aatype[0]).toBe(oracles.GAP_AATYPE); // 3-aa N-tag
  });

  it('Workflow 2: Co-crystal complex interface templating (multi-char mmCIF AA/AB -> AF2-Multimer spanChains true vs false)', async () => {
    const seqSet = makeSequenceSetArtifact(HETERODIMER_SEQUENCE, {
      chainLengths: [HETERODIMER_CHAIN_A.length, HETERODIMER_CHAIN_B.length],
      chainIds: ['A', 'B'],
    });
    const tplArt = await makeStructureArtifactFromText(MULTICHAR_COMPLEX_MMCIF, {
      filename: 'cocrystal.cif',
    });

    const openGraph = makeTemplatedLocalFoldGraph({
      sequence: HETERODIMER_SEQUENCE,
      templateText: MULTICHAR_COMPLEX_MMCIF,
      supplyMode: 'port',
      model: 'multimer',
      localFoldOverrides: { templateSpanChains: true },
      includeScorer: true,
    });
    expect(validateGraph(openGraph)).toEqual([]);

    const schedOpen = await runSchedulerWithCapturedLocalFold(openGraph);
    expect((schedOpen.workerCalls[0].inputs.template as { cifText?: string } | undefined)?.cifText).toContain('AA');

    const wOpen = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'multimer', templateSpanChains: true },
      inputs: { sequences: seqSet, template: tplArt },
    });
    const wClosed = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'multimer', templateSpanChains: false },
      inputs: { sequences: seqSet, template: tplArt },
    });
    expect(wOpen.status).toBe('result');
    expect(wClosed.status).toBe('result');
    const tplOpen = wOpen.captures[0]!.predictOpts!.template as { spanChains?: boolean } | undefined;
    const tplClosed = wClosed.captures[0]!.predictOpts!.template as { spanChains?: boolean } | undefined;
    expect(tplOpen).toBeDefined();
    expect(tplClosed).toBeDefined();
    expect(tplOpen!.spanChains).toBe(true);
    expect(tplClosed!.spanChains).toBe(false);
    const L = HETERODIMER_CHAIN_A.length + HETERODIMER_CHAIN_B.length;
    const mOpen = wOpen.captures[0]!.effectiveMultichainMask2d as Float32Array | undefined;
    const mClosed = wClosed.captures[0]!.effectiveMultichainMask2d as Float32Array | undefined;
    expect(mOpen).toBeInstanceOf(Float32Array);
    expect(mClosed).toBeInstanceOf(Float32Array);
    expect(mOpen![0 * L + HETERODIMER_CHAIN_A.length]).toBe(1);
    expect(mClosed![0 * L + HETERODIMER_CHAIN_A.length]).toBe(0);
    expect(wOpen.outputs!.folds.id).not.toBe(wClosed.outputs!.folds.id);
  });

  it('Workflow 3: Holo AF3 co-folding with modified MSE/SEP template + ZN ligand + DNA + CPU oracle verification', async () => {
    const seq = 'MKWVSTFISLLFL';
    const seqSet = makeSequenceSetArtifact(seq);
    const tplArt = await makeStructureArtifactFromText(MODIFIED_MSE_SEP_MMCIF, {
      filename: 'holo_tpl.cif',
    });

    for (const model of ['boltz2', 'rosettafold3']) {
      const graph = makeTemplatedLocalFoldGraph({
        sequence: seq,
        templateText: MODIFIED_MSE_SEP_MMCIF,
        supplyMode: 'port',
        model,
        localFoldOverrides: { ligandCcdCodes: 'ZN', dnaSequences: 'ACGT' },
      });
      expect(validateGraph(graph)).toEqual([]);
      expect(
        validateBioCapabilities(graph, {
          tpl: { structure: tplArt },
          seq: { sequences: seqSet },
        }).filter((i) => i.severity === 'error'),
      ).toEqual([]);

      const wRes = await runLocalFoldWorkerInNode({
        params: graph.nodes.find((n) => n.id === 'fold')!.params,
        inputs: { sequences: seqSet, template: tplArt },
      });
      expect(wRes.status).toBe('result');
      const batch = wRes.captures[0]!.batch as Record<string, any>;
      const slots = wRes.captures[0]!.af3Options!.templateSlots as Array<any>;
      expect(slots.some(Boolean)).toBe(true);

      const withTpl = await runAf3TemplateCpuOracle({
        modelKey: model,
        tokens: batch.tokens,
        asymId: batch.asymId,
        slots,
      });
      const noTpl = await runAf3TemplateCpuOracle({
        modelKey: model,
        tokens: batch.tokens,
        asymId: batch.asymId,
        slots: [],
      });
      const delta = diffFloat32Arrays(withTpl.output, noTpl.output);
      expect(delta.hasNaN).toBe(false);
      expect(delta.l2).toBeGreaterThan(1e-3);
    }
  });

  it('Workflow 4: Iterative design-fold redesign loop (fold1.structure -> ProteinMPNN -> fold2.template + sequences)', async () => {
    const graph: WorkflowGraph = {
      id: 'iterative-redesign-loop',
      name: 'Iterative Fold -> MPNN -> Templated Refold',
      nodes: [
        makeSequenceNode('seq0', STANDARD_MONOMER_SEQUENCE, 'Initial Sequence'),
        makeLocalFoldNode('fold1', { model: 'monomer' }, 'Initial Fold'),
        {
          id: 'mpnn',
          adapterId: 'web-protein-mpnn',
          label: 'ProteinMPNN',
          params: { ...defaultParamsForAdapter('web-protein-mpnn'), numSequences: 1 },
        },
        makeLocalFoldNode('fold2', { model: 'monomer-2' }, 'Templated Refold'),
        makeScorerNode('score'),
      ],
      edges: [
        { fromNode: 'seq0', fromPort: 'sequences', toNode: 'fold1', toPort: 'sequences' },
        { fromNode: 'fold1', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold2', toPort: 'sequences' },
        { fromNode: 'fold1', fromPort: 'structure', toNode: 'fold2', toPort: 'template' },
        { fromNode: 'fold1', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold2', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      ],
    };

    expect(validateGraph(graph)).toEqual([]);
    expect(validateBioCapabilities(graph).filter((i) => i.severity === 'error')).toEqual([]);
    const preflight = estimateRunRequirements(graph);
    expect(preflight.models.some((m) => m.key === 'localfold:monomer')).toBe(true);
    expect(preflight.models.some((m) => m.key === 'localfold:monomer-2')).toBe(true);

    const res = await runSchedulerWithCapturedLocalFold(graph);
    const fold2Call = res.workerCalls.find((c) => c.nodeParams.model === 'monomer-2');
    expect(fold2Call).toBeDefined();
    expect(fold2Call!.inputs.template).toBeDefined();
    expect(fold2Call!.inputs.template.id).toBe(res.outputsByNode.fold1.structure.id);
    expect(res.outputsByNode.score.metrics.kind).toBe('DesignabilityMetrics');
  });

  it('Workflow 5: Large inline template share & restore roundtrip (#g= + YAML + SHA-256 + worker parity)', async () => {
    const seq = 'MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE';
    const graph = makeTemplatedLocalFoldGraph({
      sequence: seq,
      templateText: LARGE_TEMPLATE_PDB,
      supplyMode: 'inline',
      model: 'af3',
    });
    const origNode = graph.nodes.find((n) => n.id === 'fold')!;
    const seqSet = makeSequenceSetArtifact(seq);

    const encoded = await encodeGraphToHash(graph);
    expect(encoded.strippedPdb).toBe(true);
    const yaml = graphToYamlRecipe(graph);
    expect(yaml).toContain('expectedTemplateSha256');

    const decodedHash = (await decodeHashToGraph(encoded.hash))!;
    const decodedYaml = parseRecipeToGraph(yaml);
    const restore = localfoldAf2Plugin.share.restoreOmittedInlineAsset!;

    const restoredHashNode = restore(
      origNode.params,
      decodedHash.nodes.find((n) => n.id === 'fold')!.params,
      '',
      'fold',
    )!;
    const restoredYamlNode = restore(
      origNode.params,
      decodedYaml.nodes.find((n) => n.id === 'fold')!.params,
      yaml,
      'fold',
    )!;

    expect(restoredHashNode.templatePdbText).toBe(LARGE_TEMPLATE_PDB);
    expect(restoredYamlNode.templatePdbText).toBe(LARGE_TEMPLATE_PDB);

    const wOrig = await runLocalFoldWorkerInNode({
      params: origNode.params,
      inputs: { sequences: seqSet },
    });
    const wRestored = await runLocalFoldWorkerInNode({
      params: restoredYamlNode,
      inputs: { sequences: seqSet },
    });
    expect(wOrig.outputs!.folds.id).toBe(wRestored.outputs!.folds.id);
  });

  it('Workflow 6: Template-free model guard & 1-click BioCompatibilityFix recovery (monomer-4 & ef2-fast-600m)', async () => {
    const seqArt = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
    const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);

    for (const badModel of ['monomer-4', 'ef2-fast-600m']) {
      const brokenGraph = makeTemplatedLocalFoldGraph({
        supplyMode: 'port',
        model: badModel,
      });

      const issues = validateBioCapabilities(brokenGraph, {
        seq: { sequences: seqArt },
        tpl: { structure: tplArt },
      });
      const errIssue = issues.find((i) => i.severity === 'error');
      expect(errIssue).toBeDefined();
      expect(errIssue?.fix).toBeDefined();

      // Scheduler and worker both refuse to run with template on template-free model
      await expect(runSchedulerWithCapturedLocalFold(brokenGraph)).rejects.toThrow(/template/i);
      const workerErr = await runLocalFoldWorkerInNode({
        params: brokenGraph.nodes.find((n) => n.id === 'fold')!.params,
        inputs: { sequences: seqArt, template: tplArt },
      });
      expect(workerErr.status).toBe('error');

      // Applying the 1-click fix heals validation, scheduler, and worker
      const fixedGraph: WorkflowGraph = {
        ...brokenGraph,
        nodes: brokenGraph.nodes.map((n) =>
          n.id === errIssue!.fix!.nodeId
            ? { ...n, params: { ...n.params, ...errIssue!.fix!.paramsPatch } }
            : n,
        ),
      };
      expect(
        validateBioCapabilities(fixedGraph, {
          seq: { sequences: seqArt },
          tpl: { structure: tplArt },
        }).filter((i) => i.severity === 'error'),
      ).toEqual([]);
      const fixedSched = await runSchedulerWithCapturedLocalFold(fixedGraph);
      expect(fixedSched.outputsByNode.fold.folds.kind).toBe('FoldedStructure');
    }
  });

  it.skipIf(!hasLiveWebGpu())(
    'Workflow 7 (Live WebGPU Probe): requests live GPU adapter when WebGPU runtime is present',
    async () => {
      const nav = navigator as Navigator & { gpu: GPU };
      const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
      expect(adapter).not.toBeNull();
    },
  );
});
