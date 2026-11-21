import { describe, expect, it } from 'vitest';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  buildCanonicalConditionIntent,
  buildDefaultGenie3ConditionFromTemplate,
  checkGenie3ConditionCompatibility,
  compileConditionToGenie3Spec,
  compileConditionToRfd4,
  effectiveGenie3DdimSteps,
  extractRfd4TemplateContext,
  isBareLengthContig,
  pdbTextToSplitCifText,
  resolveGenie3PolyAlaSequence,
} from '$lib/engine/conditioning';
import { BENCHMARK_TARGETS, getBenchmarkById } from '$lib/engine/benchmarks';
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
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'template' },
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
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd4', toPort: 'template' }],
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
    expect(cttCompiled.contig).toBe('150-150,A98,A100,A125,A128,/0,B296,/0,C295');
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
});
