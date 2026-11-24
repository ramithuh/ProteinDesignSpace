import { afterAll, describe, expect, it } from 'vitest';
import {
  defaultParamsForAdapter,
  sanitizeParamsOnModeSwitch,
  validateGraph,
} from '$lib/engine/graph-utils';
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
import {
  ALTLOC_PDB,
  HETERODIMER_CHAIN_A,
  HETERODIMER_CHAIN_B,
  HETERODIMER_PDB,
  HETERODIMER_SEQUENCE,
  HOLO_COMPLEX_MMCIF,
  HOMOLOG_MONOMER_SEQUENCE,
  LARGE_TEMPLATE_PDB,
  MODIFIED_MSE_SEP_MMCIF,
  MODIFIED_MSE_SEP_PDB,
  MULTICHAR_COMPLEX_MMCIF,
  NMR_MULTIMODEL_PDB,
  STANDARD_MONOMER_MMCIF,
  STANDARD_MONOMER_PDB,
  STANDARD_MONOMER_SEQUENCE,
  buildSyntheticChainPdbLines,
  makeSequenceSetArtifact,
  makeStructureArtifactFromText,
  makeTemplatedLocalFoldGraph,
} from './fixtures';
import {
  cleanupWorkerVendorOrigin,
  diffFloat32Arrays,
  loadVendorTemplateOracles,
  runAf3TemplateCpuOracle,
  runLocalFoldWorkerInNode,
  runSchedulerWithCapturedLocalFold,
} from './helpers';

afterAll(() => {
  cleanupWorkerVendorOrigin();
});

describe('Tier 3 — Cross-Feature Pairwise Combinatorial Matrix (>= 16 combinations)', () => {
  it('P01: monomer + PDB + port + homolog alignment -> worker GAP_AATYPE & live input preview', async () => {
    clearInputPreviewCache();
    const graph = makeTemplatedLocalFoldGraph({
      sequence: HOMOLOG_MONOMER_SEQUENCE,
      templateText: STANDARD_MONOMER_PDB,
      supplyMode: 'port',
      model: 'monomer',
    });
    const seqArt = makeSequenceSetArtifact(HOMOLOG_MONOMER_SEQUENCE);
    const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
    const preview = compileNodeInputPreview(graph, 'fold', {
      seq: { sequences: seqArt },
      tpl: { structure: tplArt },
    });
    expect(preview.availability).toBe('ready');

    const oracles = await loadVendorTemplateOracles();
    const wRes = await runLocalFoldWorkerInNode({
      params: graph.nodes.find((n) => n.id === 'fold')!.params,
      inputs: { sequences: seqArt, template: tplArt },
    });
    expect(wRes.status).toBe('result');
    const tplObj = wRes.captures[0]!.predictOpts!.template as { aatype: Int32Array };
    expect(tplObj.aatype[0]).toBe(oracles.GAP_AATYPE); // N-tag
  });

  it('P02: monomer-2 + mmCIF + inline + validation + preflight', async () => {
    const graph = makeTemplatedLocalFoldGraph({
      sequence: STANDARD_MONOMER_SEQUENCE,
      templateText: STANDARD_MONOMER_MMCIF,
      supplyMode: 'inline',
      model: 'monomer-2',
    });
    expect(validateGraph(graph)).toEqual([]);
    expect(estimateRunRequirements(graph).models.some((m) => m.key === 'localfold:monomer-2')).toBe(true);
    const wRes = await runLocalFoldWorkerInNode({
      params: graph.nodes.find((n) => n.id === 'fold')!.params,
      inputs: { sequences: makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE) },
    });
    expect(wRes.status).toBe('result');
    expect(wRes.captures[0]!.predictOpts!.template).toBeDefined();
  });

  it('P03: multimer + multi-char mmCIF (AA/AB) + port + spanChains=true', async () => {
    const seqSet = makeSequenceSetArtifact(HETERODIMER_SEQUENCE, {
      chainLengths: [HETERODIMER_CHAIN_A.length, HETERODIMER_CHAIN_B.length],
      chainIds: ['A', 'B'],
    });
    const tplArt = await makeStructureArtifactFromText(MULTICHAR_COMPLEX_MMCIF, { filename: 'complex.cif' });
    const wRes = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'multimer', templateSpanChains: true },
      inputs: { sequences: seqSet, template: tplArt },
    });
    expect(wRes.status).toBe('result');
    const tpl = wRes.captures[0]!.predictOpts!.template as { spanChains?: boolean } | undefined;
    expect(tpl).toBeDefined();
    expect(tpl!.spanChains).toBe(true);
    const mask = wRes.captures[0]!.effectiveMultichainMask2d as Float32Array | undefined;
    expect(mask).toBeInstanceOf(Float32Array);
    const L = HETERODIMER_CHAIN_A.length + HETERODIMER_CHAIN_B.length;
    expect(mask![0 * L + HETERODIMER_CHAIN_A.length]).toBe(1);
  });

  it('P04: multimer-3 + PDB co-crystal + inline + spanChains=false + permuted query chains [B, A]', async () => {
    const permutedSeq = `${HETERODIMER_CHAIN_B}:${HETERODIMER_CHAIN_A}`;
    const seqSet = makeSequenceSetArtifact(permutedSeq, {
      chainLengths: [HETERODIMER_CHAIN_B.length, HETERODIMER_CHAIN_A.length],
      chainIds: ['A', 'B'],
    });
    const wRes = await runLocalFoldWorkerInNode({
      params: {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'multimer-3',
        templatePdbText: HETERODIMER_PDB,
        templateSpanChains: false,
      },
      inputs: { sequences: seqSet },
    });
    expect(wRes.status).toBe('result');
    const tpl = wRes.captures[0]!.predictOpts!.template as { spanChains?: boolean } | undefined;
    expect(tpl).toBeDefined();
    expect(tpl!.spanChains).toBe(false);
    const mask = wRes.captures[0]!.effectiveMultichainMask2d as Float32Array | undefined;
    expect(mask).toBeInstanceOf(Float32Array);
    const L = HETERODIMER_CHAIN_A.length + HETERODIMER_CHAIN_B.length;
    expect(mask![0 * L + HETERODIMER_CHAIN_B.length]).toBe(0);
  });

  it('P05: af3 (9-proj) + modified mmCIF (MSE+SEP) + port + CPU templateEmbedding oracle delta', async () => {
    const seq = 'MKWVSTFISLLFL';
    const seqSet = makeSequenceSetArtifact(seq);
    const tplArt = await makeStructureArtifactFromText(MODIFIED_MSE_SEP_MMCIF, { filename: 'mod.cif' });
    const wRes = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'af3' },
      inputs: { sequences: seqSet, template: tplArt },
    });
    expect(wRes.status).toBe('result');
    const batch = wRes.captures[0]!.batch as Record<string, any>;
    const slots = wRes.captures[0]!.af3Options!.templateSlots as Array<any>;
    const withTpl = await runAf3TemplateCpuOracle({ modelKey: 'af3', tokens: batch.tokens, asymId: batch.asymId, slots });
    const noTpl = await runAf3TemplateCpuOracle({ modelKey: 'af3', tokens: batch.tokens, asymId: batch.asymId, slots: [] });
    expect(diffFloat32Arrays(withTpl.output, noTpl.output).l2).toBeGreaterThan(1e-3);
  });

  it('P06: protenix2 (fused 108) + modified PDB (MSE+SEP) + inline + CPU fusedTemplateEmbedding delta', async () => {
    const seq = 'MKWVSTFISLLFL';
    const seqSet = makeSequenceSetArtifact(seq);
    const wRes = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'protenix2', templatePdbText: MODIFIED_MSE_SEP_PDB },
      inputs: { sequences: seqSet },
    });
    expect(wRes.status).toBe('result');
    const batch = wRes.captures[0]!.batch as Record<string, any>;
    const slots = wRes.captures[0]!.af3Options!.templateSlots as Array<any>;
    const withTpl = await runAf3TemplateCpuOracle({ modelKey: 'protenix2', tokens: batch.tokens, asymId: batch.asymId, slots });
    const noTpl = await runAf3TemplateCpuOracle({ modelKey: 'protenix2', tokens: batch.tokens, asymId: batch.asymId, slots: [] });
    expect(withTpl.featureWidth).toBe(108);
    expect(diffFloat32Arrays(withTpl.output, noTpl.output).l2).toBeGreaterThan(1e-3);
  });

  it('P07: boltz2 (fused 109, altloc=first) vs af3 (occ) selects altLoc A (x > 8) over altLoc B (x < 2)', async () => {
    const seq = 'AVDEFGHIKL';
    const seqSet = makeSequenceSetArtifact(seq);
    const tplArt = await makeStructureArtifactFromText(ALTLOC_PDB);
    const boltzRes = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'boltz2' },
      inputs: { sequences: seqSet, template: tplArt },
    });
    const af3Res = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'af3' },
      inputs: { sequences: seqSet, template: tplArt },
    });
    expect(boltzRes.status).toBe('result');
    expect(af3Res.status).toBe('result');
    const batch = boltzRes.captures[0]!.batch as Record<string, any>;
    const boltzSlots = boltzRes.captures[0]!.af3Options!.templateSlots as Array<any>;
    const af3Slots = af3Res.captures[0]!.af3Options!.templateSlots as Array<any>;
    // Residue 0 (ALA): dense slot 1 = CA. altLoc A (first) has x=12 (>8); altLoc B (occ=0.70) has x=0 (<2)
    expect(boltzSlots[0].atomPositions[(0 * 24 + 1) * 3]).toBeGreaterThan(8);
    expect(af3Slots[0].atomPositions[(0 * 24 + 1) * 3]).toBeLessThan(2);

    const withTpl = await runAf3TemplateCpuOracle({
      modelKey: 'boltz2',
      tokens: batch.tokens,
      asymId: batch.asymId,
      slots: boltzSlots,
    });
    expect(withTpl.featureWidth).toBe(109);
    expect(diffFloat32Arrays(withTpl.output, new Float32Array(withTpl.output.length)).l2).toBeGreaterThan(1e-3);
  });

  it('P08: rosettafold3 (fused 66) + NMR_MULTIMODEL_PDB (ENDMDL) + inline + CPU oracle delta', async () => {
    const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
    const wRes = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'rosettafold3', templatePdbText: NMR_MULTIMODEL_PDB },
      inputs: { sequences: seqSet },
    });
    expect(wRes.status).toBe('result');
    const batch = wRes.captures[0]!.batch as Record<string, any>;
    const slots = wRes.captures[0]!.af3Options!.templateSlots as Array<any>;
    const withTpl = await runAf3TemplateCpuOracle({ modelKey: 'rosettafold3', tokens: batch.tokens, asymId: batch.asymId, slots });
    const noTpl = await runAf3TemplateCpuOracle({ modelKey: 'rosettafold3', tokens: batch.tokens, asymId: batch.asymId, slots: [] });
    expect(withTpl.featureWidth).toBe(66);
    expect(diffFloat32Arrays(withTpl.output, noTpl.output).l2).toBeGreaterThan(1e-3);
  });

  it('P09: openbind0 (unk_restype) + MODIFIED_MSE_SEP_MMCIF + port -> UNK [N,CA,C,CB] slots', async () => {
    const seqSet = makeSequenceSetArtifact('MKWVSTFISLLFL');
    const tplArt = await makeStructureArtifactFromText(MODIFIED_MSE_SEP_MMCIF, { filename: 'mod.cif' });
    const wRes = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'openbind0' },
      inputs: { sequences: seqSet, template: tplArt },
    });
    expect(wRes.status).toBe('result');
    const slot = (wRes.captures[0]!.af3Options!.templateSlots as Array<any>)[0];
    expect(slot.aatype[4]).toBe(20); // SEP -> UNK
  });

  it('P10: mol1 (fused 108) + HETERODIMER_PDB + inline + spanChains=true vs false oracle delta', async () => {
    const seqSet = makeSequenceSetArtifact(HETERODIMER_SEQUENCE, {
      chainLengths: [HETERODIMER_CHAIN_A.length, HETERODIMER_CHAIN_B.length],
      chainIds: ['A', 'B'],
    });
    const rTrue = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'mol1', templatePdbText: HETERODIMER_PDB, templateSpanChains: true },
      inputs: { sequences: seqSet },
    });
    const rFalse = await runLocalFoldWorkerInNode({
      params: { ...defaultParamsForAdapter('localfold-af2'), model: 'mol1', templatePdbText: HETERODIMER_PDB, templateSpanChains: false },
      inputs: { sequences: seqSet },
    });
    const b1 = rTrue.captures[0]!.batch as Record<string, any>;
    const s1 = rTrue.captures[0]!.af3Options!.templateSlots as Array<any>;
    const s2 = rFalse.captures[0]!.af3Options!.templateSlots as Array<any>;
    const o1 = await runAf3TemplateCpuOracle({ modelKey: 'mol1', tokens: b1.tokens, asymId: b1.asymId, slots: s1 });
    const o2 = await runAf3TemplateCpuOracle({ modelKey: 'mol1', tokens: b1.tokens, asymId: b1.asymId, slots: s2 });
    expect(diffFloat32Arrays(o1.output, o2.output).l2).toBeGreaterThan(1e-4);
  });

  it('P11: intellifold2 + HOLO_COMPLEX_MMCIF + ZN + DNA + port + non-identity tokenOf shift', async () => {
    const seq = 'MKWVSTFISLLFL';
    const seqSet = makeSequenceSetArtifact(seq);
    const tplArt = await makeStructureArtifactFromText(HOLO_COMPLEX_MMCIF, { filename: 'holo.cif' });
    const oracles = await loadVendorTemplateOracles();
    const tpl = await (await import('./helpers')).loadLocalFoldTemplateModule();
    const parsed = tpl.parseTemplateStructure(HOLO_COMPLEX_MMCIF);
    const align = tpl.alignQueryToTemplate([{ chainId: 'A', sequence: seq }], parsed);
    // Exercise non-identity residueOfToken where 4 DNA/ligand tokens (-1) precede protein residues 0..L-1
    const rot = new Int32Array(4 + seq.length + 1).fill(-1);
    for (let r = 0; r < seq.length; r++) rot[4 + r] = r;
    const shifted = tpl.buildAf3TemplateSlots(
      { tokens: rot.length, residueOfToken: rot },
      align,
      'intellifold2',
      { spanChains: true },
    );
    expect(shifted[0].aatype[0]).toBe(oracles.GAP_AATYPE);
    expect(shifted[0].aatype[3]).toBe(oracles.GAP_AATYPE);
    expect(shifted[0].aatype[4]).toBe(12); // MET shifted to token 4
    expect(shifted[0].aatype[4 + seq.length]).toBe(oracles.GAP_AATYPE);

    const wRes = await runLocalFoldWorkerInNode({
      params: {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'intellifold2',
        ligandCcdCodes: 'ZN',
        dnaSequences: 'ACGT',
      },
      inputs: { sequences: seqSet, template: tplArt },
    });
    expect(wRes.status).toBe('result');
    const batch = wRes.captures[0]!.batch as Record<string, any>;
    const slots = wRes.captures[0]!.af3Options!.templateSlots as Array<any>;
    expect(slots[0].aatype).toHaveLength(batch.tokens);
    expect(slots[0].covered).toBe(seq.length);
  });

  it('P12: opendde + HETERODIMER_PDB + inline + bio-capabilities + worker templateSlots', async () => {
    const graph = makeTemplatedLocalFoldGraph({
      sequence: HETERODIMER_SEQUENCE,
      templateText: HETERODIMER_PDB,
      supplyMode: 'inline',
      model: 'opendde',
    });
    expect(validateBioCapabilities(graph).filter((i) => i.severity === 'error')).toEqual([]);
    const seqSet = makeSequenceSetArtifact(HETERODIMER_SEQUENCE, {
      chainLengths: [HETERODIMER_CHAIN_A.length, HETERODIMER_CHAIN_B.length],
      chainIds: ['A', 'B'],
    });
    const wRes = await runLocalFoldWorkerInNode({
      params: graph.nodes.find((n) => n.id === 'fold')!.params,
      inputs: { sequences: seqSet },
    });
    expect(wRes.status).toBe('result');
    const slots = wRes.captures[0]!.af3Options!.templateSlots as Array<any>;
    expect(slots[0].covered).toBe(HETERODIMER_CHAIN_A.length + HETERODIMER_CHAIN_B.length);
  });

  it('P13: monomer-4 (template-free) + PDB port -> BioCompatibilityFix -> valid worker run', async () => {
    const graph = makeTemplatedLocalFoldGraph({ supplyMode: 'port', model: 'monomer-4' });
    const seqArt = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
    const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
    const issues = validateBioCapabilities(graph, {
      seq: { sequences: seqArt },
      tpl: { structure: tplArt },
    });
    const fix = issues.find((i) => i.severity === 'error' && i.fix)?.fix;
    expect(fix).toBeDefined();

    const fixedParams = { ...graph.nodes.find((n) => n.id === 'fold')!.params, ...fix!.paramsPatch };
    const wRes = await runLocalFoldWorkerInNode({
      params: fixedParams,
      inputs: { sequences: seqArt, template: tplArt },
    });
    expect(wRes.status).toBe('result');
    expect(wRes.captures[0]!.predictOpts!.template).toBeDefined();
  });

  it('P14: ef2-fast-600m (template-free) + inline mmCIF -> BioCompatibilityFix + sanitizeParamsOnModeSwitch -> worker run', async () => {
    const graph = makeTemplatedLocalFoldGraph({
      supplyMode: 'inline',
      templateText: STANDARD_MONOMER_MMCIF,
      model: 'ef2-fast-600m',
    });
    const fix = validateBioCapabilities(graph).find((i) => i.severity === 'error' && i.fix)?.fix;
    expect(fix).toBeDefined();
    const nextModel = String(fix!.paramsPatch.model);
    const switched = sanitizeParamsOnModeSwitch(
      'localfold-af2',
      graph.nodes.find((n) => n.id === 'fold')!.params,
      nextModel,
    );
    const wRes = await runLocalFoldWorkerInNode({
      params: switched,
      inputs: { sequences: makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE) },
    });
    expect(wRes.status).toBe('result');
  });

  it('P15: large inline template (>1500 chars) + boltz2 -> hash/YAML share strip -> SHA-256 restore -> runHash parity', async () => {
    const seq = 'MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE';
    const graph = makeTemplatedLocalFoldGraph({
      sequence: seq,
      templateText: LARGE_TEMPLATE_PDB,
      supplyMode: 'inline',
      model: 'boltz2',
    });
    const seqSet = makeSequenceSetArtifact(seq);
    const origParams = graph.nodes.find((n) => n.id === 'fold')!.params;
    const origRun = await runLocalFoldWorkerInNode({ params: origParams, inputs: { sequences: seqSet } });

    const { hash } = await encodeGraphToHash(graph);
    const decoded = (await decodeHashToGraph(hash))!;
    const yamlReparsed = parseRecipeToGraph(graphToYamlRecipe(graph));
    const restore = localfoldAf2Plugin.share.restoreOmittedInlineAsset!;
    const restoredFromHash = restore(origParams, decoded.nodes.find((n) => n.id === 'fold')!.params, '', 'fold')!;
    const restoredFromYaml = restore(origParams, yamlReparsed.nodes.find((n) => n.id === 'fold')!.params, '', 'fold')!;

    const hashRun = await runLocalFoldWorkerInNode({ params: restoredFromHash, inputs: { sequences: seqSet } });
    const yamlRun = await runLocalFoldWorkerInNode({ params: restoredFromYaml, inputs: { sequences: seqSet } });
    expect(hashRun.outputs!.folds.id).toBe(origRun.outputs!.folds.id);
    expect(yamlRun.outputs!.folds.id).toBe(origRun.outputs!.folds.id);
  });

  it('P16: pdb-input (cropSegments) -> localfold-af2.template + ExecutionScheduler memoization invalidation on template change', async () => {
    const g1 = makeTemplatedLocalFoldGraph({
      sequence: 'MKWVTFISLL',
      templateText: STANDARD_MONOMER_PDB,
      supplyMode: 'port',
      model: 'monomer-2',
      cropSegments: 'A1-10',
    });
    const altLines = buildSyntheticChainPdbLines('MKWVTFISLL', {
      chainId: 'A',
      originOffset: [10, 10, 10],
    }).lines.join('\n') + '\nEND\n';
    const g2 = makeTemplatedLocalFoldGraph({
      sequence: 'MKWVTFISLL',
      templateText: altLines,
      supplyMode: 'port',
      model: 'monomer-2',
      cropSegments: 'A1-10',
    });

    const r1 = await runSchedulerWithCapturedLocalFold(g1);
    const r2 = await runSchedulerWithCapturedLocalFold(g2);
    expect(r1.workerCalls[0].inputs.template).toBeDefined();
    expect(r2.workerCalls[0].inputs.template).toBeDefined();
    expect(r1.workerCalls[0].inputs.template.id).not.toBe(r2.workerCalls[0].inputs.template.id);
    expect(r1.outputsByNode.fold.folds.id).not.toBe(r2.outputsByNode.fold.folds.id);
  });
});
