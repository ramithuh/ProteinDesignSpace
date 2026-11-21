import { describe, expect, it } from 'vitest';
import { MODEL_CATALOG } from 'proteinmpnn-web';
import {
  ADAPTER_REGISTRY,
  LOCALFOLD_MODEL_OPTIONS,
  MPNN_VARIANT_OPTIONS,
  RFD4_BENCHMARK_OPTIONS,
  SAMPLE_CAMPAIGN,
  WORKFLOW_PRESETS,
} from '$lib/engine/registry';
import {
  CATALOG_CATEGORIES,
  CATALOG_ITEMS,
  CATALOG_STAGE_GROUPS,
  filterCatalog,
  filterCatalogCategories,
} from '$lib/engine/catalog';
import {
  PORT_COLORS,
  STAGE_LABELS,
  defaultParamsForAdapter,
  insertBlockSmart,
  validateGraph,
} from '$lib/engine/graph-utils';
import { validateNodeParams } from '$lib/engine/param-validation';
import type { WorkflowGraph } from '$lib/engine/types';

describe('engine/catalog & biologist terminology', () => {
  it('exposes complete registry metadata for all 18 LocalFold models, 3 MPNN variants, 9 RFD4 tasks, and 7 presets', () => {
    expect(LOCALFOLD_MODEL_OPTIONS).toHaveLength(18);
    expect(LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af2')).toHaveLength(5);
    expect(LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af2-multimer')).toHaveLength(5);
    expect(LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3')).toHaveLength(8);
    expect(new Set(LOCALFOLD_MODEL_OPTIONS.map((m) => m.id)).size).toBe(18);
    for (const m of LOCALFOLD_MODEL_OPTIONS) {
      expect(m.approxMB).toBeGreaterThan(0);
      expect(m.license.length).toBeGreaterThan(0);
    }

    expect(MPNN_VARIANT_OPTIONS).toHaveLength(3);
    expect(MPNN_VARIANT_OPTIONS.map((v) => v.family)).toEqual(['solublempnn', 'proteinmpnn', 'ca']);
    for (const v of MPNN_VARIANT_OPTIONS) {
      expect(MODEL_CATALOG[v.family]).toContain(v.modelName);
    }

    expect(RFD4_BENCHMARK_OPTIONS).toHaveLength(9);
    expect(WORKFLOW_PRESETS).toHaveLength(7);

    for (const preset of WORKFLOW_PRESETS) {
      expect(validateGraph(preset.graph)).toEqual([]);
      for (const node of preset.graph.nodes) {
        expect(validateNodeParams(node)).toEqual([]);
      }
    }
  });

  it('organizes all 8 blocks into workflow groups with structured sub-items for LocalFold and ProteinMPNN', () => {
    const groupedIds = CATALOG_STAGE_GROUPS.flatMap((g) => g.ids);
    expect(new Set(groupedIds)).toEqual(new Set(Object.keys(ADAPTER_REGISTRY)));
    expect(CATALOG_ITEMS).toHaveLength(Object.keys(ADAPTER_REGISTRY).length);
    expect(CATALOG_ITEMS).toHaveLength(8);
    expect(ADAPTER_REGISTRY['top-k-filter']).toBeDefined();
    expect(ADAPTER_REGISTRY['rfd4-conditioner']).toBeUndefined();

    const lf = CATALOG_ITEMS.find((i) => i.id === 'localfold-af2')!;
    expect(lf.subGroups.map((g) => g.title)).toEqual([
      'AF2-Class (Monomer)',
      'AF2-Multimer v3 (Complexes)',
      'AF3-Class (All-Atom)',
    ]);
    expect(lf.subGroups[0].items).toHaveLength(5);
    expect(lf.subGroups[1].items).toHaveLength(5);
    expect(lf.subGroups[2].items).toHaveLength(8);
    const af3Sub = lf.subGroups[2].items.find((s) => s.key === 'af3')!;
    expect(af3Sub.nonCommercial).toBe(true);
    expect(af3Sub.badge).toBe('Non-Commercial');

    const mpnn = CATALOG_ITEMS.find((i) => i.id === 'web-protein-mpnn')!;
    expect(mpnn.subGroups).toHaveLength(1);
    expect(mpnn.subGroups[0].items.map((s) => s.insert.params?.family)).toEqual([
      'solublempnn',
      'proteinmpnn',
      'ca',
    ]);

    // Genie 3 preset defaults to Cα-ProteinMPNN (family: 'ca') for honest Cα-only backbones
    const g3Preset = WORKFLOW_PRESETS.find((p) => p.id === 'genie3-mpnn-af2')!;
    const g3Mpnn = g3Preset.graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn')!;
    expect(g3Mpnn.params.family).toBe('ca');
    expect(g3Mpnn.label).toBe('Cα-ProteinMPNN');
  });

  it('filterCatalog matches parent fields and narrows multi-variant sub-items for boltz, soluble, motif, and af3', () => {
    const boltzRes = filterCatalog('boltz');
    expect(boltzRes).toHaveLength(1);
    expect(boltzRes[0].item.id).toBe('localfold-af2');
    expect(boltzRes[0].hasNarrowedSubMatch).toBe(true);
    expect(boltzRes[0].matchingSubItems.map((s) => s.key)).toEqual(['boltz2']);

    const solRes = filterCatalog('soluble');
    expect(solRes.some((r) => r.item.id === 'web-protein-mpnn')).toBe(true);
    const mpnnMatch = solRes.find((r) => r.item.id === 'web-protein-mpnn')!;
    expect(mpnnMatch.hasNarrowedSubMatch).toBe(true);
    expect(mpnnMatch.matchingSubItems.map((s) => s.key)).toEqual(['solublempnn']);

    const motifRes = filterCatalog('motif');
    expect(motifRes.some((r) => r.item.id === 'rfd4-proteina')).toBe(true);
    expect(motifRes.some((r) => r.item.id === 'genie3')).toBe(true);

    const af3Res = filterCatalog('af3');
    const lfAf3 = af3Res.find((r) => r.item.id === 'localfold-af2')!;
    expect(lfAf3).toBeDefined();
    expect(lfAf3.hasNarrowedSubMatch).toBe(true);
    expect(lfAf3.matchingSubItems).toHaveLength(8);
    expect(lfAf3.matchingSubItems.some((s) => s.key === 'boltz2')).toBe(true);
    expect(lfAf3.matchingSubItems.some((s) => s.key === 'monomer')).toBe(false);

    // Cross-boundary multi-term queries narrow to matching sub-items
    const lfBoltz = filterCatalog('localfold boltz').find((r) => r.item.id === 'localfold-af2')!;
    expect(lfBoltz.hasNarrowedSubMatch).toBe(true);
    expect(lfBoltz.matchingSubItems.map((s) => s.key)).toEqual(['boltz2']);

    const mpnnSol = filterCatalog('proteinmpnn soluble').find((r) => r.item.id === 'web-protein-mpnn')!;
    expect(mpnnSol.hasNarrowedSubMatch).toBe(true);
    expect(mpnnSol.matchingSubItems.map((s) => s.key)).toEqual(['solublempnn']);

    // Pure parent identity queries ("rfd4", "proteinmpnn", "localfold") must NOT mis-narrow
    const rfd4Bare = filterCatalog('rfd4').find((r) => r.item.id === 'rfd4-proteina')!;
    expect(rfd4Bare).toBeDefined();
    expect(rfd4Bare.hasNarrowedSubMatch).toBe(false);

    const mpnnBare = filterCatalog('proteinmpnn').find((r) => r.item.id === 'web-protein-mpnn')!;
    expect(mpnnBare.hasNarrowedSubMatch).toBe(false);
    expect(mpnnBare.matchingSubItems).toHaveLength(3);

    // "af2" narrows LocalFold to the 10 AF2-class models (5 monomer + 5 multimer v3)
    const af2Res = filterCatalog('af2').find((r) => r.item.id === 'localfold-af2')!;
    expect(af2Res.hasNarrowedSubMatch).toBe(true);
    expect(af2Res.matchingSubItems).toHaveLength(10);

    // "multimer" narrows LocalFold to the 5 AF2-Multimer v3 models
    const multiRes = filterCatalog('multimer').find((r) => r.item.id === 'localfold-af2')!;
    expect(multiRes.hasNarrowedSubMatch).toBe(true);
    expect(multiRes.matchingSubItems).toHaveLength(5);
  });

  it('exposes 5 compact top-level workflow categories with direct model flyouts and LocalFold footnote', () => {
    expect(CATALOG_CATEGORIES.map((c) => c.title)).toEqual([
      'Inputs',
      'Structure Generation',
      'Sequence Design',
      'Structure Prediction',
      'Evaluation',
    ]);

    const pred = CATALOG_CATEGORIES.find((c) => c.id === 'prediction')!;
    expect(pred.footnote).toContain('LocalFold');
    expect(pred.subGroups.flatMap((g) => g.items)).toHaveLength(18);

    const boltz = filterCatalogCategories('boltz');
    expect(boltz).toHaveLength(1);
    expect(boltz[0].category.id).toBe('prediction');
    expect(boltz[0].matchingSubItems.map((s) => s.key)).toEqual(['boltz2']);

    const genie = filterCatalogCategories('genie');
    expect(genie).toHaveLength(1);
    expect(genie[0].category.id).toBe('generation');
    expect(genie[0].matchingSubItems.map((s) => s.key)).toEqual(['genie3']);
  });

  it('insertBlockSmart supports pre-configured params, label deduplication, default param preservation, and auto-wire proposals', () => {
    const base: WorkflowGraph = {
      id: 'preconfig-test',
      name: 'Preconfig Test',
      nodes: [
        {
          id: 'seq_1',
          adapterId: 'sequence-input',
          label: 'Sequence Input',
          params: defaultParamsForAdapter('sequence-input'),
          position: { x: 60, y: 60 },
        },
      ],
      edges: [],
    };

    // 1. Insert LocalFold pre-configured with Boltz-2
    const lfSub = filterCatalog('boltz')[0].matchingSubItems[0];
    const rBoltz = insertBlockSmart(base, lfSub.insert.adapterId, 'seq_1', undefined, {
      autoCommit: true,
      params: lfSub.insert.params,
      label: lfSub.insert.label,
    });
    const boltzNode = rBoltz.graph.nodes.find((n) => n.id === rBoltz.nodeId)!;
    expect(boltzNode.label).toBe('Boltz-2');
    expect(boltzNode.params.model).toBe('boltz2');
    expect(boltzNode.params.recycles).toBe(1);
    expect(boltzNode.params.diffusionSteps).toBe(25);
    expect(boltzNode.params.maxSequences).toBe(2);
    expect(rBoltz.proposedEdges).toEqual([
      { fromNode: 'seq_1', fromPort: 'sequences', toNode: rBoltz.nodeId, toPort: 'sequences' },
    ]);
    expect(validateGraph(rBoltz.graph)).toEqual([]);

    // 2. Label deduplication: second Boltz-2 gets #2, distinct RoseTTAFold 3 stays unnumbered
    const rBoltz2 = insertBlockSmart(rBoltz.graph, 'localfold-af2', 'seq_1', undefined, {
      autoCommit: false,
      params: { model: 'boltz2' },
      label: 'Boltz-2',
    });
    expect(rBoltz2.graph.nodes.find((n) => n.id === rBoltz2.nodeId)!.label).toBe('Boltz-2 #2');

    const rRf3 = insertBlockSmart(rBoltz2.graph, 'localfold-af2', 'seq_1', undefined, {
      autoCommit: false,
      params: { model: 'rosettafold3' },
      label: 'RoseTTAFold 3',
    });
    expect(rRf3.graph.nodes.find((n) => n.id === rRf3.nodeId)!.label).toBe('RoseTTAFold 3');

    // 3. Insert ProteinMPNN variants (SolubleMPNN & CA-only)
    const rSol = insertBlockSmart(base, 'web-protein-mpnn', undefined, { x: 300, y: 200 }, {
      autoCommit: false,
      params: { family: 'solublempnn', modelName: 'v_48_020' },
      label: 'SolubleMPNN',
    });
    const solNode = rSol.graph.nodes.find((n) => n.id === rSol.nodeId)!;
    expect(solNode.params.family).toBe('solublempnn');
    expect(solNode.params.numSequences).toBe(2);
    expect(solNode.params.temperature).toBe(0.1);

    const rCa = insertBlockSmart(rSol.graph, 'web-protein-mpnn', undefined, { x: 300, y: 360 }, {
      autoCommit: false,
      params: { family: 'ca', modelName: 'v_48_020' },
      label: 'ProteinMPNN (Cα-only)',
    });
    const caNode = rCa.graph.nodes.find((n) => n.id === rCa.nodeId)!;
    expect(caNode.params.family).toBe('ca');
    expect(caNode.label).toBe('ProteinMPNN (Cα-only)');

    // 4. Insert RFD4-Proteina with inline preset mode
    const rRfd = insertBlockSmart(base, 'rfd4-proteina', undefined, { x: 60, y: 240 }, {
      autoCommit: false,
      params: { mode: 'preset', presetId: 'ppi_pdl1' },
      label: 'RFD4 Binder',
    });
    const rfdNode = rRfd.graph.nodes.find((n) => n.id === rRfd.nodeId)!;
    expect(rfdNode.params.mode).toBe('preset');
    expect(rfdNode.params.presetId).toBe('ppi_pdl1');
    expect(rfdNode.params.length).toBe(70);
  });

  it('contains zero forbidden internal or ML implementation tokens in user-facing registry, catalog, port, and preset strings', () => {
    const FORBIDDEN =
      /\b(WGSL|wgsl-custom|ort-webgpu|cpu-ts|INT8|INT5|FP16|ONNX|C_HOT|C_CRDGRP|C_IDX|C_HB|C_CATH|condition_spec|vf_tsr|OPFS|LocalFold AF2|Refold \(AF2\)|StructureArtifact|ConditionSpec|FoldedStructure)\b|SE\(3\)/;

    const userFacingStrings: string[] = [];

    for (const m of Object.values(ADAPTER_REGISTRY)) {
      userFacingStrings.push(m.displayName, m.shortName, m.description, m.licenseNote);
    }
    for (const opt of RFD4_BENCHMARK_OPTIONS) {
      userFacingStrings.push(opt.label, opt.category);
    }
    for (const v of MPNN_VARIANT_OPTIONS) {
      userFacingStrings.push(v.label, v.shortLabel, v.description);
    }
    for (const lf of LOCALFOLD_MODEL_OPTIONS) {
      userFacingStrings.push(lf.label, lf.shortLabel);
    }
    for (const pc of Object.values(PORT_COLORS)) {
      userFacingStrings.push(pc.label, pc.short);
    }
    for (const sl of Object.values(STAGE_LABELS)) {
      userFacingStrings.push(sl);
    }
    for (const item of CATALOG_ITEMS) {
      userFacingStrings.push(item.group, item.displayName, item.shortName, item.role);
      for (const sg of item.subGroups) {
        userFacingStrings.push(sg.title);
        for (const sub of sg.items) {
          userFacingStrings.push(sub.label, sub.subtitle ?? '', sub.meta ?? '', sub.badge ?? '');
        }
      }
    }
    for (const preset of WORKFLOW_PRESETS) {
      userFacingStrings.push(preset.name, preset.badge, preset.summary);
      for (const n of preset.graph.nodes) {
        if (n.label) userFacingStrings.push(n.label);
      }
    }
    userFacingStrings.push(SAMPLE_CAMPAIGN.assay.targetName ?? '');
    for (const r of SAMPLE_CAMPAIGN.rounds) {
      userFacingStrings.push(r.generator, r.folder, r.conditioningSummary ?? '');
    }

    for (const s of userFacingStrings) {
      expect(s, `Forbidden token found in user-facing string: "${s}"`).not.toMatch(FORBIDDEN);
    }
  });

  it('uses model-first labels for Structure Prediction blocks and qualifies SolubleMPNN recommendation', () => {
    const predCat = CATALOG_CATEGORIES.find((c) => c.id === 'prediction')!;
    const predItems = predCat.subGroups.flatMap((g) => g.items);
    for (const item of predItems) {
      expect(item.insert.label).toBeTruthy();
      expect(item.insert.label).not.toMatch(/^LocalFold\b/);
    }
    expect(predItems.find((i) => i.key === 'monomer')?.insert.label).toBe('AF2 Model 1');
    expect(predItems.find((i) => i.key === 'multimer')?.insert.label).toBe('AF2-Multimer 1');
    expect(predItems.find((i) => i.key === 'rosettafold3')?.insert.label).toBe('RoseTTAFold 3');
    expect(predItems.find((i) => i.key === 'boltz2')?.insert.label).toBe('Boltz-2');
    expect(predItems.find((i) => i.key === 'af3')?.insert.label).toBe('AlphaFold 3');
    expect(predItems.find((i) => i.key === 'openbind0')?.insert.label).toBe('OpenBind-0');

    const branchPreset = WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-branch-compare')!;
    expect(branchPreset.graph.nodes.find((n) => n.id === 'fold_mpnn')?.label).toBe('AF2 Model 1 (MPNN)');
    expect(branchPreset.graph.nodes.find((n) => n.id === 'fold_native')?.label).toBe('AF2 Model 1 (Native)');

    const crossFoldPreset = WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-af2-vs-rf3')!;
    expect(crossFoldPreset.graph.nodes.find((n) => n.id === 'fold_af2')?.label).toBe('AF2 Model 1');
    expect(crossFoldPreset.graph.nodes.find((n) => n.id === 'fold_rf3')?.label).toBe('RoseTTAFold 3');

    const solOpt = MPNN_VARIANT_OPTIONS.find((v) => v.id === 'solublempnn')!;
    expect(solOpt.description).toContain('recommended for soluble proteins');
    expect(solOpt.description).not.toMatch(/\(recommended\)/);
  });
});
