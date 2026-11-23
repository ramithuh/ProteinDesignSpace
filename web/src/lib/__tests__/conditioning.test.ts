import { describe, expect, it } from 'vitest';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  CRYSTALLIZATION_BUFFERS,
  buildCanonicalConditionIntent,
  buildDefaultGenie3ConditionFromTemplate,
  checkGenie3ConditionCompatibility,
  compileConditionToGenie3Spec,
  compileConditionToRfd4,
  effectiveGenie3DdimSteps,
  extractRfd4TemplateContext,
  isBareLengthContig,
  isCrystallizationBuffer,
  pdbTextToSplitCifText,
  resolveGenie3PolyAlaSequence,
  resolveMotifSourceResidues,
} from '$lib/engine/conditioning';
import { BENCHMARK_TARGETS, adaptBenchmarkContigForLength, getBenchmarkById } from '$lib/engine/benchmarks';
import { isValidPdbId } from '$lib/protein-ops/rcsb';
import {
  cropParsedStructure,
  cropRawAllAtomStructure,
  generateSampleHelixBundlePdb,
  parseStructureText,
} from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { validateConnection } from '$lib/engine/graph-utils';
import type { ConditionSpecArtifact, StructureArtifact, WorkflowGraph } from '$lib/engine/types';

function makeSampleTemplate(): StructureArtifact {
  const pdbText = generateSampleHelixBundlePdb();
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id: 'tpl-sample',
    label: 'sample_53aa.pdb',
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    diagnostics: parsed.diagnostics,
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds),
  };
}

describe('Model-Agnostic Conditioning & Curated Benchmark Targets', () => {
  it('compiles canonical intent cleanly into RFD4 sel(...) condition dicts', () => {
    const intent = buildCanonicalConditionIntent({
      contig: 'A17-132,/0,70-70',
      lockChain: 'A',
      hotspots: 'A54, A56, A115',
      motifSpan: 'A16-35',
      unindexedMotif: true,
      hbondMode: 'ACCEPTOR',
      hbondSelection: 'B/SAM/*/*',
      cathCode: '3.40.50',
    });

    const rfd4 = compileConditionToRfd4(intent);
    expect(rfd4.contig).toBe('A17-132,/0,70-70');
    expect(rfd4.conditions.C_CRD).toBeDefined();
    expect(rfd4.conditions.C_HOT).toEqual({
      true: [{ select: "sel('A/*/[54,56,115]/*')" }],
    });
    expect(rfd4.conditions.C_CRDGRP).toEqual({
      1: [{ select: "sel('A/*/[16-35]/*')" }],
    });
    expect(rfd4.conditions.C_IDX).toBeDefined();
    expect(rfd4.conditions.C_HB).toEqual({ ACCEPTOR: [{ select: "sel('B/SAM/*/*')" }] });
    expect(rfd4.conditions.C_CATH).toEqual({ '3.40.50': [{ select: "sel('*/*/*/*')" }] });
  });

  it('compiles PPI binder & motif scaffolding intents into valid Genie 3 createFeaturesFromSpec batches', async () => {
    const tpl = makeSampleTemplate();

    // 1. PPI Binder Design: 40-aa binder against target A5-34 (30 aa) with hotspots A10, A15
    const binderIntent = buildCanonicalConditionIntent({
      contig: 'A5-34,/0,40-40',
      lockChain: 'A',
      hotspots: 'A10, A15',
    });
    const binderCond: ConditionSpecArtifact = {
      kind: 'ConditionSpec',
      id: 'cond-binder',
      label: 'Binder',
      contig: binderIntent.contig,
      conditions: {},
      intent: binderIntent,
      summary: 'binder',
    };
    const binderSpec = compileConditionToGenie3Spec(binderCond, tpl, 60);
    expect(binderSpec.mode).toBe('binder');
    expect(binderSpec.length).toBe(70); // 40 binder + 30 target
    expect(binderSpec.chainLengths).toEqual([40, 30]);
    expect(binderSpec.asymId[0]).toBe(0);
    expect(binderSpec.asymId[39]).toBe(0);
    expect(binderSpec.asymId[40]).toBe(1);
    expect(binderSpec.condStructMask[39]).toBe(0);
    expect(binderSpec.condStructMask[40]).toBe(1);
    // Hotspot A10 is offset 5 into A5-34 => index 40 + 5 = 45; A15 => 40 + 10 = 50
    expect(binderSpec.condInterfaceMask[45]).toBe(1);
    expect(binderSpec.condInterfaceMask[50]).toBe(1);
    expect(binderSpec.condInterfaceMask[46]).toBe(0);

    // 2. Motif Scaffolding: 10-10,A10-20,10-10 (10 + 11 + 10 = 31 aa)
    const motifIntent = buildCanonicalConditionIntent({
      contig: '10-10,A10-20,10-10',
      motifSpan: 'A10-20',
    });
    const motifCond: ConditionSpecArtifact = {
      kind: 'ConditionSpec',
      id: 'cond-motif',
      label: 'Motif',
      contig: motifIntent.contig,
      conditions: {},
      intent: motifIntent,
      summary: 'motif',
    };
    const motifSpec = compileConditionToGenie3Spec(motifCond, tpl, 60);
    expect(motifSpec.mode).toBe('motif');
    expect(motifSpec.length).toBe(31);
    expect(motifSpec.condStructMask[9]).toBe(0);
    expect(motifSpec.condStructMask[10]).toBe(1);
    expect(motifSpec.condStructMask[20]).toBe(1);
    expect(motifSpec.condStructMask[21]).toBe(0);

    // 3. Feed both compiled specs straight into upstream Genie 3 `createFeaturesFromSpec`
    const featuresPath = resolve(__dirname, '../../../static/vendor/genie3/shared/genie3/features.js');
    const g3Features = await import(pathToFileURL(featuresPath).href);
    const binderBatch = g3Features.createFeaturesFromSpec(binderSpec);
    expect(binderBatch.length).toBe(70);
    expect(binderBatch.cond_struct_frame_mask[45]).toBe(1);
    expect(binderBatch.cond_interface_mask[45]).toBe(1);

    const motifBatch = g3Features.createFeaturesFromSpec(motifSpec);
    expect(motifBatch.length).toBe(31);
    expect(motifBatch.cond_struct_frame_mask[15]).toBe(1);
  });

  it('rejects RFD4-only constraints (H-bonds, CATH, unindexed motif, RFD4 preset) on Genie 3 and warns in Layer 1', () => {
    const hbondCond: ConditionSpecArtifact = {
      kind: 'ConditionSpec',
      id: 'hb',
      label: 'HB',
      contig: '80-80',
      conditions: {},
      intent: buildCanonicalConditionIntent({
        contig: '80-80',
        hbondMode: 'ACCEPTOR',
        hbondSelection: 'B/SAM/*/*',
      }),
      summary: 'HB',
    };
    expect(checkGenie3ConditionCompatibility(hbondCond).supported).toBe(false);

    const graph: WorkflowGraph = {
      id: 'g3-warn',
      name: 'Genie3 Warn',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda' },
        },
        { id: 'g3', adapterId: 'genie3', params: { length: 80 } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    };
    expect(validateConnection(graph, 'pdb', 'structure', 'g3', 'structure').valid).toBe(true);
    const issues = validateBioCapabilities(graph);
    expect(issues.some((i) => i.code === 'genie3-unsupported-benchmark' && i.nodeId === 'g3')).toBe(true);
  });

  it('curated BENCHMARK_TARGETS all have unique IDs, valid RCSB PDB IDs, and valid WebGPU token budgets', () => {
    expect(BENCHMARK_TARGETS.length).toBeGreaterThanOrEqual(16);
    const ids = new Set(BENCHMARK_TARGETS.map((b) => b.id));
    expect(ids.size).toBe(BENCHMARK_TARGETS.length);
    for (const b of BENCHMARK_TARGETS) {
      expect(isValidPdbId(b.pdbId), `Invalid PDB ID on ${b.id}: ${b.pdbId}`).toBe(true);
      expect(b.approxTokens).toBeGreaterThan(20);
      expect(b.approxTokens).toBeLessThanOrEqual(300);
      expect(b.conditionParams.contig.length).toBeGreaterThan(0);
    }
  });

  it('auto-builds Genie 3 binder/motif specs when a template PDB is wired directly without a Conditioning block', () => {
    const tpl = makeSampleTemplate();

    // 1. Generic PDB -> Genie 3 defaults to de novo binder design on Chain A with requested binder length
    const autoGeneric = buildDefaultGenie3ConditionFromTemplate(tpl, 64);
    const genericCompiled = compileConditionToGenie3Spec(autoGeneric, tpl, 64);
    expect(genericCompiled.mode).toBe('binder');
    expect(genericCompiled.chainLengths).toEqual([64, tpl.length]);

    // 2. Curated Genie 3 benchmark PDB (e.g. 1PRW motif) auto-applies its motif spec
    const motifTpl: StructureArtifact = { ...tpl, benchmarkId: 'rfd_1prw_efhand' };
    const autoMotif = buildDefaultGenie3ConditionFromTemplate(motifTpl, 80);
    expect(autoMotif.intent?.motifSpans).toEqual(['A16-35', 'A52-71']);

    // 3. Curated binder benchmark (PD-L1 5O45) adapts binder length to Genie 3 slider while preserving hotspots
    const pdl1Tpl: StructureArtifact = { ...tpl, benchmarkId: 'pdl1_5o45' };
    const autoPdl1 = buildDefaultGenie3ConditionFromTemplate(pdl1Tpl, 80);
    expect(autoPdl1.intent?.lockChain).toBe('A');
    expect(autoPdl1.intent?.hotspots).toEqual(['A54', 'A56', 'A66', 'A115']);
    expect(autoPdl1.intent?.contig).toBe('A17-132,/0,80-80');

    // 4. Non-Genie 3 benchmark (enzyme/ligand) fails clearly instead of silently running unconditional
    const samTpl: StructureArtifact = { ...tpl, benchmarkId: 'sm_7c7m_sam' };
    expect(() => buildDefaultGenie3ConditionFromTemplate(samTpl, 80)).toThrow(/RFD4-Proteina/i);

    // 5. Layer 1 warns when RFD4 has a wired template in unconditional mode, and infers multimer on direct PDB -> Genie 3 -> AF2
    const directBinderGraph: WorkflowGraph = {
      id: 'direct-binder',
      name: 'Direct Binder',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' } },
        { id: 'g3', adapterId: 'genie3', params: { length: 80 } },
        { id: 'lf', adapterId: 'localfold-af2', params: { model: 'monomer' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' },
        { fromNode: 'g3', fromPort: 'sequences', toNode: 'lf', toPort: 'sequences' },
      ],
    };
    const directIssues = validateBioCapabilities(directBinderGraph);
    expect(directIssues.some((i) => i.code === 'multimer-into-af2-monomer' && i.nodeId === 'lf')).toBe(true);

    const rfd4UnusedTpl: WorkflowGraph = {
      id: 'rfd4-unused-tpl',
      name: 'RFD4 Unused Template',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'rfd4', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 70 } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd4', toPort: 'structure' }],
    };
    const rfd4Issues = validateBioCapabilities(rfd4UnusedTpl);
    expect(rfd4Issues.some((i) => i.code === 'rfd4-unused-template' && i.fix?.paramsPatch?.mode === 'partial')).toBe(true);
  });

  it('cropParsedStructure extracts exact benchmark residue segments while preserving original PDB numbering', () => {
    const rawPdb = generateSampleHelixBundlePdb();
    const parsed = parseStructureText(rawPdb);
    expect(parsed.nResidues).toBe(53);

    const cropped = cropParsedStructure(parsed, 'A16-25, A40-44');
    expect(cropped.nResidues).toBe(15);
    expect(cropped.resNums.slice(0, 10)).toEqual([16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
    expect(cropped.resNums.slice(10)).toEqual([40, 41, 42, 43, 44]);

    // Re-parsing the emitted cropped PDB retains exact original residue numbers
    const reparsed = parseStructureText(cropped.pdbText);
    expect(reparsed.nResidues).toBe(15);
    expect(reparsed.resNums).toEqual(cropped.resNums);
    expect(reparsed.chainIds.every((c) => c === 'A')).toBe(true);

    // Every single curated benchmark (all 16: motifs, PPI binders, AME enzymes, SAM, B-DNA) defines explicit cropSegments
    expect(BENCHMARK_TARGETS.length).toBe(16);
    for (const b of BENCHMARK_TARGETS) {
      expect(b.cropSegments && b.cropSegments.length > 0, `Missing cropSegments on ${b.id}`).toBe(true);
    }

    // All-atom cropper keeps both selected catalytic residues AND matching HETATM ligands/metals
    const hetPdb =
      rawPdb.replace('END\n', '') +
      'HETATM  900 ZN    ZN A 296       1.000   2.000   3.000  1.00 10.00          ZN\n' +
      'HETATM  901 C1   DHZ A 295       2.000   3.000   4.000  1.00 10.00           C\n' +
      'HETATM  902 O    HOH A 500       9.000   9.000   9.000  1.00 10.00           O\n' +
      'END\n';
    const rawCropped = cropRawAllAtomStructure(hetPdb, 'A10-12, A:ZN, A:DHZ');
    const parsedAllAtom = parseStructureText(rawCropped.text);
    expect(parsedAllAtom.nResidues).toBe(3);
    expect(parsedAllAtom.resNums).toEqual([10, 11, 12]);
    expect(parsedAllAtom.diagnostics.hasLigands).toBe(true);
    expect(parsedAllAtom.diagnostics.ligandNames).toEqual(['ZN', 'DHZ']);
  });

  it('injects motif spans & template ligands/DNA into RFD4 contig and C_CRDGRP/C_CRD, and remaps auth_* -> label_* IDs', () => {
    // 1. Bare length AME enzyme intent (1CTT) with shifted auth_seq_id + RCSB ligand chains
    const cttCif = [
      'data_1ctt',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_entity_id',
      '_atom_site.label_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.occupancy',
      '_atom_site.B_iso_or_equiv',
      '_atom_site.auth_seq_id',
      '_atom_site.auth_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_atom_id',
      '_atom_site.pdbx_PDB_model_num',
      'ATOM 1 C CA . HIS A 1 98 ? 0.0 0.0 0.0 1.0 10.0 102 HIS A CA 1',
      'ATOM 2 C CA . GLU A 1 100 ? 3.8 0.0 0.0 1.0 10.0 104 GLU A CA 1',
      'ATOM 3 C CA . CYS A 1 125 ? 0.0 3.8 0.0 1.0 10.0 129 CYS A CA 1',
      'ATOM 4 C CA . CYS A 1 128 ? 3.8 3.8 0.0 1.0 10.0 132 CYS A CA 1',
      'HETATM 5 ZN ZN . ZN B 2 . ? 1.9 1.9 1.0 1.0 10.0 296 ZN A ZN 1',
      'HETATM 6 C C1 . DHZ C 3 . ? 2.5 2.5 2.0 1.0 10.0 295 DHZ A C1 1',
      '#',
    ].join('\n');

    const cttCtx = extractRfd4TemplateContext(cttCif);
    const cttBench = getBenchmarkById('ame_1ctt_cda')!;
    const cttIntent = buildCanonicalConditionIntent(cttBench.conditionParams);
    const cttCompiled = compileConditionToRfd4(cttIntent, cttCtx);

    // Remapped label_seq_ids (98, 100, 125, 128) + ligand chains B296, C295 injected into contig!
    expect(cttCompiled.contig).toBe('180-180,A98,A100,A125,A128,/0,B296,/0,C295');
    expect(cttCompiled.conditions.C_CRDGRP).toEqual({
      1: [{ select: "sel('A/*/[98,100,125,128]/*') | sel('B/*/*/*') | sel('C/*/*/*')" }],
    });
    expect(cttCompiled.conditions.C_IDX).toEqual({
      false: [{ select: "sel('A/*/[98,100,125,128]/*')", group: 'motif' }],
    });

    // 2. 7C7M (SAM) and 7M5W (B-DNA) benchmarks now use wired-template custom mode instead of remote preset
    const samBench = getBenchmarkById('sm_7c7m_sam')!;
    const dnaBench = getBenchmarkById('rfd4_7m5w_bdna')!;
    expect(samBench.conditionParams.taskPreset).toBe('custom');
    expect(dnaBench.conditionParams.taskPreset).toBe('custom');

    const samCif = [
      'data_7c7m',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_entity_id',
      '_atom_site.label_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.occupancy',
      '_atom_site.B_iso_or_equiv',
      '_atom_site.auth_seq_id',
      '_atom_site.auth_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_atom_id',
      '_atom_site.pdbx_PDB_model_num',
      'HETATM 1 C C1 . SAM D 2 . ? 1.0 2.0 3.0 1.0 10.0 301 SAM A C1 1',
      '#',
    ].join('\n');
    const samCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent(samBench.conditionParams),
      extractRfd4TemplateContext(samCif),
    );
    expect(samCompiled.contig).toBe('110-110,/0,D301');
    expect(samCompiled.conditions.C_CRD).toEqual({ true: [{ select: "sel('D/*/*/*')" }] });
    expect(samCompiled.conditions.C_HB).toEqual({ ACCEPTOR: [{ select: "sel('D/SAM/*/*')" }] });

    // 3. Binder contig synthesis for blank / bare-monomer contigs
    expect(isBareLengthContig('')).toBe(true);
    expect(isBareLengthContig('70-70')).toBe(true);
    expect(isBareLengthContig('A17-132,/0,70-70')).toBe(false);
    expect(cttCtx.synthesizeBinderContig('A', 65)).toBe('A98-128,/0,65-65');
  });

  it('pdbTextToSplitCifText splits HETATM ligands/ions onto distinct label_asym_id and label_entity_id', () => {
    const rawPdb =
      generateSampleHelixBundlePdb().replace('END\n', '') +
      'HETATM  900 ZN    ZN A 296       1.000   2.000   3.000  1.00 10.00          ZN\n' +
      'HETATM  901 C1   SAM A 301       2.000   3.000   4.000  1.00 10.00           C\n' +
      'HETATM  902 O    HOH A 500       9.000   9.000   9.000  1.00 10.00           O\n' +
      'END\n';
    const cif = pdbTextToSplitCifText(rawPdb);
    expect(cif).toContain('_entity.type');
    expect(cif).toContain('non-polymer');
    expect(cif).not.toContain('HOH');
    const ctx = extractRfd4TemplateContext(cif);
    expect(ctx.polymerByAuthChain.get('A')?.labelChain).toBe('A');
    expect(ctx.ligands).toHaveLength(2);
    expect(ctx.ligands[0].labelChain).not.toBe('A');
    expect(ctx.ligands[1].labelChain).not.toBe('A');
    expect(ctx.ligands[0].labelChain).not.toBe(ctx.ligands[1].labelChain);
    expect(ctx.ligands.map((l) => l.compId)).toEqual(['ZN', 'SAM']);
  });

  it('unifies Genie 3 binder length cap at 256, resolves poly-ALA + fixed sequence cleanly, and computes effective DDIM steps', () => {
    const tpl = makeSampleTemplate();
    const auto240 = buildDefaultGenie3ConditionFromTemplate(tpl, 240);
    const spec240 = compileConditionToGenie3Spec(auto240, tpl, 240);
    expect(spec240.chainLengths).toEqual([240, tpl.length]);

    // When predictSequence=false, unconditioned binder positions become 'A' while locked target preserves gtRestype
    const polySeq = resolveGenie3PolyAlaSequence(spec240.length, spec240);
    expect(polySeq.slice(0, 240)).toBe('A'.repeat(240));
    expect(polySeq.slice(240)).toBe(tpl.sequence);

    expect(effectiveGenie3DdimSteps(100)).toBe(100);
    expect(effectiveGenie3DdimSteps(60)).toBe(63);
    expect(effectiveGenie3DdimSteps(150)).toBe(167);
  });

  it('links motif and binder selections back to original source residues (multi-char chains + insertion codes)', () => {
    const cif = [
      'data_motif_identity',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_seq_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 CA ALA A 1 AA 1 ? 0.000 0.000 0.000',
      'ATOM 2 CA HIS A 2 AA 10 ? 3.800 0.000 0.000',
      'ATOM 3 CA CYS A 3 AA 10 A 7.600 0.000 0.000',
      'ATOM 4 CA ASP A 4 AA 10 B 11.400 0.000 0.000',
      'ATOM 5 CA GLU A 5 AA 11 ? 15.200 0.000 0.000',
      'ATOM 6 CA GLY B 1 AB 1 ? 25.000 0.000 0.000',
      'ATOM 7 CA TRP B 2 AB 2 ? 28.800 0.000 0.000',
      '#',
    ].join('\n');

    const parsed = parseStructureText(cif);
    expect(parsed.sequence).toBe('AHCDEGW');
    expect(parsed.chainIdMap).toEqual({ A: 'AA', B: 'AB' });

    // 1. Standalone resolveMotifSourceResidues links multi-char + insCode selection back to source
    const resolved = resolveMotifSourceResidues(parsed, 'AA1, AA10A-10B, AB1');
    expect(resolved).toEqual([
      {
        outputIndex: 0,
        templateIndex: 0,
        authChainId: 'AA',
        labelChainId: 'A',
        resNum: 1,
        labelSeqId: 1,
        insCode: '',
        resName: 'ALA',
        aa1: 'A',
        projectedChainId: 'A',
      },
      {
        outputIndex: 1,
        templateIndex: 2,
        authChainId: 'AA',
        labelChainId: 'A',
        resNum: 10,
        labelSeqId: 3,
        insCode: 'A',
        resName: 'CYS',
        aa1: 'C',
        projectedChainId: 'A',
      },
      {
        outputIndex: 2,
        templateIndex: 3,
        authChainId: 'AA',
        labelChainId: 'A',
        resNum: 10,
        labelSeqId: 4,
        insCode: 'B',
        resName: 'ASP',
        aa1: 'D',
        projectedChainId: 'A',
      },
      {
        outputIndex: 3,
        templateIndex: 5,
        authChainId: 'AB',
        labelChainId: 'B',
        resNum: 1,
        labelSeqId: 1,
        insCode: '',
        resName: 'GLY',
        aa1: 'G',
        projectedChainId: 'B',
      },
    ]);

    const tpl: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'tpl-multichar',
      label: 'multichar.cif',
      sourceModelId: 'pdb-input',
      length: parsed.nResidues,
      sequence: parsed.sequence,
      caCoords: parsed.caCoords,
      backbone4Coords: parsed.backbone4Coords,
      pdbText: parsed.pdbText,
      cifText: cif,
      chainIds: parsed.chainIds,
      chainLengths: parsed.diagnostics.chainLengths,
      diagnostics: parsed.diagnostics,
      geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds),
    };

    // 2. Genie 3 motif compilation preserves AA:1, AA:10A, AA:10B, AB:1 with exact outputIndex mapping
    const motifIntent = buildCanonicalConditionIntent({
      contig: '5-5,AA1,AA10A-10B,AB1,5-5',
      motifSpan: 'AA1, AA10A-10B, AB1',
    });
    const motifCond: ConditionSpecArtifact = {
      kind: 'ConditionSpec',
      id: 'cond-motif-id',
      label: 'Motif Identity',
      contig: motifIntent.contig,
      conditions: {},
      intent: motifIntent,
      summary: 'motif identity',
    };
    const motifSpec = compileConditionToGenie3Spec(motifCond, tpl, 20);
    expect(motifSpec.mode).toBe('motif');
    expect(motifSpec.length).toBe(14); // 5 + 4 + 5
    expect(motifSpec.selectedSourceResidues.map((r) => [r.outputIndex, r.authChainId, r.resNum, r.insCode, r.resName])).toEqual([
      [5, 'AA', 1, '', 'ALA'],
      [6, 'AA', 10, 'A', 'CYS'],
      [7, 'AA', 10, 'B', 'ASP'],
      [8, 'AB', 1, '', 'GLY'],
    ]);
    // Verify condStructMask agrees 1:1 with selectedSourceResidues outputIndex
    for (const r of motifSpec.selectedSourceResidues) {
      expect(motifSpec.condStructMask[r.outputIndex]).toBe(1);
    }

    // 3. Range AA10-11 selects 10, 10A, 10B, 11 in template order without clobbering insertion codes
    const rangeIntent = buildCanonicalConditionIntent({
      contig: '20-20',
      motifSpan: 'AA10-11',
    });
    const rangeSpec = compileConditionToGenie3Spec(
      {
        kind: 'ConditionSpec',
        id: 'cond-range-ins',
        label: 'Range Ins',
        contig: rangeIntent.contig,
        conditions: {},
        intent: rangeIntent,
        summary: 'range ins',
      },
      tpl,
      20,
    );
    expect(rangeSpec.selectedSourceResidues.map((r) => `${r.authChainId}:${r.resNum}${r.insCode}:${r.resName}`)).toEqual([
      'AA:10:HIS',
      'AA:10A:CYS',
      'AA:10B:ASP',
      'AA:11:GLU',
    ]);

    // 4. Binder mode against multi-char chain AA preserves exact lockChain & selectedSourceResidues
    const binderIntent = buildCanonicalConditionIntent({
      contig: 'AA1-11,/0,25-25',
      lockChain: 'AA',
      hotspots: 'AA10A',
    });
    expect(binderIntent.lockChain).toBe('AA');
    const binderSpec = compileConditionToGenie3Spec(
      {
        kind: 'ConditionSpec',
        id: 'cond-binder-aa',
        label: 'Binder AA',
        contig: binderIntent.contig,
        conditions: {},
        intent: binderIntent,
        summary: 'binder aa',
      },
      tpl,
      25,
    );
    expect(binderSpec.mode).toBe('binder');
    expect(binderSpec.chainLengths).toEqual([25, 5]);
    expect(binderSpec.selectedSourceResidues.map((r) => [r.outputIndex, r.authChainId, r.resNum, r.insCode])).toEqual([
      [25, 'AA', 1, ''],
      [26, 'AA', 10, ''],
      [27, 'AA', 10, 'A'],
      [28, 'AA', 10, 'B'],
      [29, 'AA', 11, ''],
    ]);
    // Hotspot AA10A is at outputIndex 27 only (NOT AA10 at 26 or AA10B at 28!)
    expect(binderSpec.condInterfaceMask[26]).toBe(0);
    expect(binderSpec.condInterfaceMask[27]).toBe(1);
    expect(binderSpec.condInterfaceMask[28]).toBe(0);

    // 5. RFD4 compilation with templateContext returns selectedSourceResidues with exact compiled outputIndex
    const rfd4Ctx = extractRfd4TemplateContext(cif);
    const rfd4Compiled = compileConditionToRfd4(motifIntent, rfd4Ctx);
    expect(
      rfd4Compiled.selectedSourceResidues?.map((r) => [
        r.outputIndex,
        `${r.authChainId}:${r.resNum}${r.insCode}:${r.resName}`,
      ]),
    ).toEqual([
      [5, 'AA:1:ALA'],
      [6, 'AA:10A:CYS'],
      [7, 'AA:10B:ASP'],
      [8, 'AB:1:GLY'],
    ]);

    // Contig-only RFD4 selection (no motifSpan) also resolves selectedSourceResidues at exact outputIndex
    const contigOnlyRfd4 = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: '5-5,AA10A,5-5' }),
      rfd4Ctx,
    );
    expect(contigOnlyRfd4.selectedSourceResidues?.map((r) => [r.outputIndex, r.authChainId, r.resNum, r.insCode])).toEqual([
      [5, 'AA', 10, 'A'],
    ]);

    // Absent insertion code (e.g. A10Z) must NOT fall back to unrelated label_seq_id (e.g. A19 -> label 10)
    const shiftedRows = Array.from(
      { length: 10 },
      (_, i) => `ATOM ${i + 1} CA ALA A ${i + 1} A ${i + 10} ? ${i * 4} 0 0`,
    );
    const shiftedCif = [
      'data_shifted_ins',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_seq_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      ...shiftedRows,
      'ATOM 11 CA ASP B 1 AA 10 D 50 0 0',
      '#',
    ].join('\n');
    const shiftedCtx = extractRfd4TemplateContext(shiftedCif);
    const absentInsCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: '5-5,A10Z,5-5', motifSpan: 'A10Z' }),
      shiftedCtx,
    );
    expect(absentInsCompiled.contig).toBe('5-5,A10Z,5-5');
    expect(absentInsCompiled.selectedSourceResidues).toBeUndefined();

    // Insertion codes P/R/D/N with colon syntax (e.g. AA:10D) remap cleanly instead of being treated as expandable segments
    expect(shiftedCtx.remapContig('5-5,AA:10D,5-5')).toBe('5-5,B1,5-5');

    // Named expandable segments without trailing polymer suffix (e.g. h1:5-5) advance RFD4 outputIndex accurately
    const namedExpRfd4 = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: 'h1:5-5,AA10A,5-5' }),
      rfd4Ctx,
    );
    expect(namedExpRfd4.selectedSourceResidues?.map((r) => [r.outputIndex, r.authChainId, r.resNum, r.insCode])).toEqual([
      [5, 'AA', 10, 'A'],
    ]);

    // Nucleic acid residues (including T/ADE/CYT/GUA/THY/URA) and CA-less polymer residues
    // preceding protein in mmCIF must NOT shift protein templateIndex off parseStructureText
    const dnaFirstCif = [
      'data_dna_before_protein',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_seq_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 P DA D 1 D 1 ? 0 0 0',
      'ATOM 2 P THY D 2 D 2 ? 3 0 0',
      'ATOM 3 P ADE R 1 R 1 ? 6 0 0',
      'ATOM 4 P T D 3 D 3 ? 9 0 0',
      // CA-less protein residue A:9 (excluded by parseStructureText) must get templateIndex -1
      'ATOM 5 N SER A 1 A 9 ? 10 0 0',
      'ATOM 6 C SER A 1 A 9 ? 11 0 0',
      // Observed-CA protein residues A:10 (N before CA) and A:11 must get templateIndex 0 and 1
      'ATOM 7 N ALA A 2 A 10 ? 12 0 0',
      'ATOM 8 CA ALA A 2 A 10 ? 13.4 0 0',
      'ATOM 9 CA GLY A 3 A 11 ? 17.2 0 0',
      '#',
    ].join('\n');
    const dnaFirstParsed = parseStructureText(dnaFirstCif);
    expect(dnaFirstParsed.sequence).toBe('AG');
    const dnaFirstCtx = extractRfd4TemplateContext(dnaFirstCif);
    expect(dnaFirstCtx.byLabelRes.get('A:1')?.templateIndex).toBe(-1);
    const dnaFirstCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: '5-5,A10-11,5-5' }),
      dnaFirstCtx,
    );
    expect(dnaFirstCompiled.selectedSourceResidues?.map((r) => [r.outputIndex, r.templateIndex, r.resNum])).toEqual([
      [5, 0, 10],
      [6, 1, 11],
    ]);
    for (const sel of dnaFirstCompiled.selectedSourceResidues ?? []) {
      expect(dnaFirstParsed.sourceResidues[sel.templateIndex].resNum).toBe(sel.resNum);
    }

    // Contig-only Genie 3 motif (no explicit motifSpan) compiles as motif with selectedSourceResidues
    const contigOnlyGenieIntent = buildCanonicalConditionIntent({
      contig: '5-5,AA1-10,5-5',
    });
    const contigOnlyGenie = compileConditionToGenie3Spec(
      {
        kind: 'ConditionSpec',
        id: 'cond-contig-only',
        label: 'Contig Only',
        contig: contigOnlyGenieIntent.contig,
        conditions: {},
        intent: contigOnlyGenieIntent,
        summary: 'contig only',
      },
      tpl,
      20,
    );
    expect(contigOnlyGenie.mode).toBe('motif');
    expect(contigOnlyGenie.selectedSourceResidues.map((r) => [r.outputIndex, r.authChainId, r.resNum, r.insCode])).toEqual([
      [5, 'AA', 1, ''],
      [6, 'AA', 10, ''],
      [7, 'AA', 10, 'A'],
      [8, 'AA', 10, 'B'],
    ]);

    // pdbTextToSplitCifText must NOT emit parser-reconstructed O atoms as observed in mmCIF
    const ncacOnlyPdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00           N  ',
      'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 50.00           C  ',
      'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 50.00           C  ',
      'END',
    ].join('\n');
    const parsedNcacOnly = parseStructureText(ncacOnlyPdb);
    const splitCif = pdbTextToSplitCifText(parsedNcacOnly.pdbText);
    const reparsedSplitCif = parseStructureText(splitCif);
    expect(reparsedSplitCif.atomPresence[0]).toEqual({
      hasN: true,
      hasCA: true,
      hasC: true,
      hasO: false,
      oReconstructed: true,
    });

    // H-bond remapping preserves case-sensitive lowercase chains ('a' vs 'A')
    const caseCif = [
      'data_case_hbond',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_seq_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 CA ALA X 1 A 1 ? 0 0 0',
      'ATOM 2 CA GLY Y 1 a 1 ? 10 0 0',
      '#',
    ].join('\n');
    const caseCtx = extractRfd4TemplateContext(caseCif);
    expect(caseCtx.remapHbondSelection("sel('a/*/1/*')")).toBe("sel('Y/*/1/*')");
    expect(caseCtx.remapHbondSelection("sel('A/*/1/*')")).toBe("sel('X/*/1/*')");
  });

  it('excludes crystallization buffers (SO4, GOL, EDO, PEG, etc.) from RFD4 auto-injection while retaining real metals (ZN, MG, FE2), cofactors (SAM), and explicitly named buffers', () => {
    for (const buf of ['SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'CL', 'NA', 'DMS', 'MPD', 'HOH']) {
      expect(CRYSTALLIZATION_BUFFERS.has(buf), buf).toBe(true);
      expect(isCrystallizationBuffer(buf), buf).toBe(true);
    }
    for (const real of ['ZN', 'MG', 'MN', 'CA', 'FE', 'FE2', 'CU', 'CO', 'NI', 'SAM', 'DHZ']) {
      expect(isCrystallizationBuffer(real), real).toBe(false);
    }

    const mixedCif = [
      'data_crystal_buffers',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_entity_id',
      '_atom_site.label_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.occupancy',
      '_atom_site.B_iso_or_equiv',
      '_atom_site.auth_seq_id',
      '_atom_site.auth_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_atom_id',
      '_atom_site.pdbx_PDB_model_num',
      'ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.0 10.0 1 ALA A CA 1',
      'ATOM 2 C CA . GLY A 1 2 ? 3.8 0.0 0.0 1.0 10.0 2 GLY A CA 1',
      'ATOM 3 C CA . VAL A 1 3 ? 7.6 0.0 0.0 1.0 10.0 3 VAL A CA 1',
      'HETATM 4 S S . SO4 B 2 . ? 10.0 0.0 0.0 1.0 20.0 401 SO4 A S 1',
      'HETATM 5 C C1 . GOL C 3 . ? 12.0 0.0 0.0 1.0 20.0 402 GOL A C1 1',
      'HETATM 6 ZN ZN . ZN D 4 . ? 14.0 0.0 0.0 1.0 10.0 403 ZN A ZN 1',
      'HETATM 7 C C1 . SAM E 5 . ? 16.0 0.0 0.0 1.0 10.0 404 SAM A C1 1',
      '#',
    ].join('\n');

    const ctx = extractRfd4TemplateContext(mixedCif);

    // 1. Default binder design: SO4 & GOL excluded; ZN & SAM retained!
    const binderCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: 'A1-3,/0,50-50', lockChain: 'A' }),
      ctx,
    );
    expect(binderCompiled.contig).toBe('A1-3,/0,50-50,/0,D403,/0,E404');
    expect(binderCompiled.contig).not.toContain('B401');
    expect(binderCompiled.contig).not.toContain('C402');
    const crdStr = JSON.stringify(binderCompiled.conditions.C_CRD);
    expect(crdStr).toContain("sel('D/*/*/*')");
    expect(crdStr).toContain("sel('E/*/*/*')");
    expect(crdStr).not.toContain("sel('B/*/*/*')");
    expect(crdStr).not.toContain("sel('C/*/*/*')");

    // 2. Explicitly naming SO4 in contig retains SO4 while still excluding GOL
    const explicitContigCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: 'A1-3,/0,50-50,/0,A:SO4', lockChain: 'A' }),
      ctx,
    );
    expect(explicitContigCompiled.contig).toContain('B401');
    expect(explicitContigCompiled.contig).not.toContain('C402');
    expect(JSON.stringify(explicitContigCompiled.conditions.C_CRD)).toContain("sel('B/*/*/*')");

    // 3. Explicitly naming GOL in cropSegments retains GOL while still excluding SO4
    const explicitCropCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: 'A1-3,/0,50-50', lockChain: 'A' }),
      ctx,
      { cropSegments: 'A1-3, A:GOL' },
    );
    expect(explicitCropCompiled.contig).toContain('C402');
    expect(explicitCropCompiled.contig).not.toContain('B401');
  });

  it('R3: adaptBenchmarkContigForLength adapts 5O45 binder (65aa) and 1YCR motif (80aa) contigs consistently for Genie 3 and RFD4', () => {
    // 1. Binder 5O45 at 65 aa
    const adapted5o45 = adaptBenchmarkContigForLength('A17-132,/0,70-70', 65, 'binder');
    expect(adapted5o45).toBe('A17-132,/0,65-65');

    // Feed into Genie 3 binder compiler with a 116-aa chain A template (A17..A132)
    const n5o45 = 116;
    const pdb5o45 =
      Array.from({ length: n5o45 }, (_, i) =>
        `ATOM  ${String(i + 1).padStart(5)}  CA  ALA A${String(17 + i).padStart(4)}       0.000   0.000   0.000  1.00 10.00           C  `,
      ).join('\n') + '\nEND\n';
    const parsed5o45 = parseStructureText(pdb5o45);
    const tpl5o45: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'tpl-5o45',
      label: '5O45',
      sourceModelId: 'pdb-input',
      length: parsed5o45.nResidues,
      sequence: parsed5o45.sequence,
      caCoords: parsed5o45.caCoords,
      backbone4Coords: parsed5o45.backbone4Coords,
      pdbText: parsed5o45.pdbText,
      chainIds: parsed5o45.chainIds,
      sourceResidues: parsed5o45.sourceResidues,
      geometry: computeGeometryStats(parsed5o45.caCoords, parsed5o45.chainIds, parsed5o45.resNums),
    };
    const cond5o45: ConditionSpecArtifact = {
      kind: 'ConditionSpec',
      id: 'cond-5o45',
      label: '5O45',
      contig: adapted5o45,
      conditions: {},
      summary: '5O45 65aa binder',
      intent: buildCanonicalConditionIntent({ contig: adapted5o45, lockChain: 'A' }),
    };
    const g3Binder = compileConditionToGenie3Spec(cond5o45, tpl5o45, 65);
    expect(g3Binder.chainLengths[0]).toBe(65);
    expect(g3Binder.chainLengths[1]).toBe(116);

    const rfd4Binder = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: adapted5o45, lockChain: 'A', length: 65 }),
      extractRfd4TemplateContext(pdb5o45),
    );
    expect(rfd4Binder.contig).toBe('A1-116,/0,65-65');

    // 2. Motif 1YCR B19-27 at 80 aa (35 + 9 + 36 = 80)
    const adapted1ycr = adaptBenchmarkContigForLength('28-28,B19-27,28-28', 80, 'motif');
    expect(adapted1ycr).toBe('35-35,B19-27,36-36');

    const n1ycr = 9;
    const pdb1ycr =
      Array.from({ length: n1ycr }, (_, i) =>
        `ATOM  ${String(i + 1).padStart(5)}  CA  ALA B${String(19 + i).padStart(4)}       0.000   0.000   0.000  1.00 10.00           C  `,
      ).join('\n') + '\nEND\n';
    const parsed1ycr = parseStructureText(pdb1ycr);
    const tpl1ycr: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'tpl-1ycr',
      label: '1YCR',
      sourceModelId: 'pdb-input',
      length: parsed1ycr.nResidues,
      sequence: parsed1ycr.sequence,
      caCoords: parsed1ycr.caCoords,
      backbone4Coords: parsed1ycr.backbone4Coords,
      pdbText: parsed1ycr.pdbText,
      chainIds: parsed1ycr.chainIds,
      sourceResidues: parsed1ycr.sourceResidues,
      geometry: computeGeometryStats(parsed1ycr.caCoords, parsed1ycr.chainIds, parsed1ycr.resNums),
    };
    const cond1ycr: ConditionSpecArtifact = {
      kind: 'ConditionSpec',
      id: 'cond-1ycr',
      label: '1YCR',
      contig: adapted1ycr,
      conditions: {},
      summary: '1YCR 80aa motif',
      intent: buildCanonicalConditionIntent({ contig: adapted1ycr, motifSpan: 'B19-27' }),
    };
    const g3Motif = compileConditionToGenie3Spec(cond1ycr, tpl1ycr, 80);
    expect(g3Motif.length).toBe(80);
    expect(g3Motif.chainLengths).toEqual([80]);

    const rfd4Motif = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: adapted1ycr, motifSpan: 'B19-27', length: 80 }),
      extractRfd4TemplateContext(pdb1ycr),
    );
    expect(rfd4Motif.contig).toBe('35-35,B1-9,36-36');

    // 3. Multi-island motif & enzyme-ligand & invalid targetLength guard
    expect(adaptBenchmarkContigForLength('10-10,A16-35,16-16,A52-71,10-10', 100, 'motif')).toMatch(/A16-35.*A52-71/);
    expect(adaptBenchmarkContigForLength('150-150', 120, 'enzyme-ligand')).toBe('120-120');
    expect(adaptBenchmarkContigForLength('28-28,B19-27,28-28', 0, 'motif')).toBe('28-28,B19-27,28-28');
  });
});
