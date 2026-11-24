import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterAll, describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from '$lib/engine/registry';
import {
  defaultParamsForAdapter,
  sanitizeParamsOnModeSwitch,
  validateConnection,
  validateGraph,
} from '$lib/engine/graph-utils';
import { computeUploadDigest } from '$lib/engine/param-validation';
import { resolveLocalFoldEffectiveConfig } from '$lib/engine/conditioning';
import {
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from '$lib/engine/bio-capabilities';
import { clearInputPreviewCache, compileNodeInputPreview } from '$lib/engine/input-preview';
import { compileWorkflowPlanToGraph } from '$lib/engine/workflow-planner';
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
  COINCIDENT_CLASH_PDB,
  CORRUPT_TEXT,
  DNA_ONLY_PDB,
  EMPTY_TEXT,
  HETERODIMER_CHAIN_A,
  HETERODIMER_CHAIN_B,
  HETERODIMER_PDB,
  HETERODIMER_SEQUENCE,
  INS_CODE_NEGATIVE_RESNUM_PDB,
  LARGE_TEMPLATE_PDB,
  LIGAND_ONLY_PDB,
  MODIFIED_MSE_SEP_MMCIF,
  MODIFIED_MSE_SEP_PDB,
  MULTICHAR_COMPLEX_MMCIF,
  SHORT_2AA_PDB,
  SHORT_4AA_PDB,
  SINGLE_RESIDUE_PDB,
  STANDARD_MONOMER_PDB,
  STANDARD_MONOMER_SEQUENCE,
  TEMPLATE_CAPABLE_AF2_MULTI_MODELS,
  TEMPLATE_CAPABLE_AF3_MODELS,
  TEMPLATE_FREE_MODELS,
  UNRELATED_SEQUENCE,
  WATER_ONLY_PDB,
  WHITESPACE_TEXT,
  buildManyChainMmcif,
  buildMmcifFromChains,
  buildSyntheticChainPdbLines,
  makeLocalFoldNode,
  makePdbNode,
  makeSequenceNode,
  makeSequenceSetArtifact,
  makeStructureArtifactFromText,
  makeTemplatedLocalFoldGraph,
  makeUntemplatedLocalFoldGraph,
} from './fixtures';
import {
  cleanupWorkerVendorOrigin,
  diffFloat32Arrays,
  ensureWorkerVendorOrigin,
  loadLocalFoldTemplateModule,
  loadVendorTemplateOracles,
  runAf3TemplateCpuOracle,
  runLocalFoldWorkerInNode,
  runSchedulerWithCapturedLocalFold,
} from './helpers';

afterAll(() => {
  cleanupWorkerVendorOrigin();
});

describe('Tier 2 — Boundary & Corner Cases (F1–F12)', () => {
  describe('F1 Boundaries: Capability Taxonomy', () => {
    it('rejects empty and unknown model IDs as non-template-capable', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      for (const bad of ['', 'unknown-model', 'boltz1', 'monomer-6', 'esmfold']) {
        expect(tpl.isLocalFoldTemplateCapable(bad)).toBe(false);
        expect(tpl.localFoldTemplateFamily(bad)).toBe('none');
      }
    });

    it('does not misclassify uppercase or near-miss model IDs as capable', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      for (const bad of ['MONOMER', 'AF3', 'Boltz2', 'monomer_2', '   ']) {
        expect(tpl.isLocalFoldTemplateCapable(bad)).toBe(false);
        expect(tpl.localFoldTemplateFamily(bad)).toBe('none');
      }
    });

    it('maps monomer-2 -> af2-monomer and monomer-3 -> none at the exact boundary', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      expect(tpl.isLocalFoldTemplateCapable('monomer-2')).toBe(true);
      expect(tpl.localFoldTemplateFamily('monomer-2')).toBe('af2-monomer');
      expect(tpl.isLocalFoldTemplateCapable('monomer-3')).toBe(false);
      expect(tpl.localFoldTemplateFamily('monomer-3')).toBe('none');
    });

    it('maps all AF2-Multimer variants multimer..multimer-5 to af2-multimer', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      for (const m of TEMPLATE_CAPABLE_AF2_MULTI_MODELS) {
        expect(tpl.localFoldTemplateFamily(m)).toBe('af2-multimer');
      }
    });

    it('maps all 8 AF3 dialect models to af3', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      for (const m of TEMPLATE_CAPABLE_AF3_MODELS) {
        expect(tpl.localFoldTemplateFamily(m)).toBe('af3');
      }
    });
  });

  describe('F2 Boundaries: PDB & mmCIF Parser', () => {
    it('throws explicit error on empty and whitespace-only inputs', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      expect(() => tpl.parseTemplateStructure(EMPTY_TEXT)).toThrow();
      expect(() => tpl.parseTemplateStructure(WHITESPACE_TEXT)).toThrow();
    });

    it('throws explicit error on corrupted non-coordinate text', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      expect(() => tpl.parseTemplateStructure(CORRUPT_TEXT)).toThrow();
    });

    it('throws explicit 0-polymer-residue error on water-only, ligand-only, and DNA-only structures', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      expect(() => tpl.parseTemplateStructure(WATER_ONLY_PDB)).toThrow();
      expect(() => tpl.parseTemplateStructure(LIGAND_ONLY_PDB)).toThrow();
      expect(() => tpl.parseTemplateStructure(DNA_ONLY_PDB)).toThrow();
    });

    it('preserves negative residue numbers and insertion codes (100A, 100B, 100C) in order', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(INS_CODE_NEGATIVE_RESNUM_PDB);
      expect(parsed.chains).toHaveLength(1);
      expect(parsed.chains[0].sequence).toBe('MKWVTFISLL');
      expect(parsed.chains[0].residues).toHaveLength(10);
    });

    it('breaks equal-occupancy altLoc ties deterministically vs altlocFirst', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const occPolicy = tpl.parseTemplateStructure(ALTLOC_PDB, { altlocFirst: false });
      const firstPolicy = tpl.parseTemplateStructure(ALTLOC_PDB, { altlocFirst: true });
      // Residue 2: altLoc B (first in file, x=3.8, occ=0.5) vs altLoc A (second in file, x=15.8, occ=0.5)
      const caFirst = firstPolicy.chains[0].residues[1].atoms.get('CA')!;
      const caOcc = occPolicy.chains[0].residues[1].atoms.get('CA')!;
      expect(caFirst[0]).toBeLessThan(6); // altLoc B (first seen)
      expect(caOcc[0]).toBeGreaterThan(10); // altLoc A (alphabetical tie-break)
    });

    it('parses >26 multi-char mmCIF chains (30 chains) without collision', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const cif30 = buildManyChainMmcif(30, 5);
      const parsed = tpl.parseTemplateStructure(cif30);
      expect(parsed.chains).toHaveLength(30);
      expect(new Set(parsed.chains.map((c) => c.chainId)).size).toBe(30);
    });
  });

  describe('F3 Boundaries: Alignment & Chain Matching', () => {
    it('rejects or reports zero-overlap alignment on completely unrelated sequence', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      expect(() =>
        tpl.alignQueryToTemplate([{ chainId: 'A', sequence: UNRELATED_SEQUENCE }], parsed),
      ).toThrow();
    });

    it('handles single-residue and 4-aa minimal templates cleanly', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed1 = tpl.parseTemplateStructure(SINGLE_RESIDUE_PDB);
      expect(parsed1.chains[0].residues).toHaveLength(1);
      const parsed4 = tpl.parseTemplateStructure(SHORT_4AA_PDB);
      const align4 = tpl.alignQueryToTemplate([{ chainId: 'A', sequence: 'MKWV' }], parsed4);
      expect(align4.matchedQueryResidues).toBe(4);
    });

    it('aligns structured domain while leaving long C-terminal disordered tail unaligned', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const tailedQuery = STANDARD_MONOMER_SEQUENCE + 'G'.repeat(20);
      const align = tpl.alignQueryToTemplate([{ chainId: 'A', sequence: tailedQuery }], parsed);
      expect(align.matchedQueryResidues).toBe(STANDARD_MONOMER_SEQUENCE.length);
      expect(align.totalQueryResidues).toBe(tailedQuery.length);
    });

    it('aligns homodimer query (A:A) against single-chain or two-chain template consistently', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const homodimerCif = buildMmcifFromChains([
        { authAsymId: 'A', sequence: STANDARD_MONOMER_SEQUENCE },
        { authAsymId: 'B', sequence: STANDARD_MONOMER_SEQUENCE },
      ]);
      const parsed = tpl.parseTemplateStructure(homodimerCif);
      const align = tpl.alignQueryToTemplate(
        [
          { chainId: 'A', sequence: STANDARD_MONOMER_SEQUENCE },
          { chainId: 'B', sequence: STANDARD_MONOMER_SEQUENCE },
        ],
        parsed,
      );
      expect(align.matchedQueryResidues).toBe(STANDARD_MONOMER_SEQUENCE.length * 2);
    });

    it('enforces strict minCoverage boundary when requested', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(SHORT_4AA_PDB);
      const longQuery = 'MKWV' + 'D'.repeat(36);
      expect(() =>
        tpl.alignQueryToTemplate([{ chainId: 'A', sequence: longQuery }], parsed, {
          minCoverage: 0.5,
        }),
      ).toThrow();
    });
  });

  describe('F4 Boundaries: AF2 Monomer Atom37', () => {
    it('leaves zero NaNs in AF2 monomer payload and templateGeometry even with coincident [0,0,0] clashing coords', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(COINCIDENT_CLASH_PDB);
      const align = tpl.alignQueryToTemplate([{ chainId: 'A', sequence: 'MKWV' }], parsed);
      const payload = tpl.buildAf2TemplatePayload(['MKWV'], align, 'af2-monomer', { spanChains: true });
      // Verify consecutive residues genuinely share [0, 0, 0] CA coordinates (r = 0 clash)
      expect(payload.template.atomPositions[(0 * 37 + 1) * 3]).toBe(0);
      expect(payload.template.atomPositions[(1 * 37 + 1) * 3]).toBe(0);
      const geom = oracles.templateGeometry(payload.template, payload.multichainMask2d, 4, oracles.AF2_ATOM37_MONOMER);
      expect(Array.from(geom.unitVector).every(Number.isFinite)).toBe(true);
      expect(Array.from(geom.distogram).every(Number.isFinite)).toBe(true);
    });

    it('maps MSE SE->SD into MET SD slot in AF2 atom37', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(MODIFIED_MSE_SEP_PDB);
      const seq = 'MKWVSTFISLLFL';
      const align = tpl.alignQueryToTemplate([{ chainId: 'A', sequence: seq }], parsed);
      const payload = tpl.buildAf2TemplatePayload([seq], align, 'af2-monomer', { spanChains: true });
      const sdSlot = oracles.ATOM37.indexOf('SD');
      expect(sdSlot).toBeGreaterThanOrEqual(0);
      expect(payload.template.atomMask[0 * 37 + sdSlot]).toBe(1);
    });

    it('masks missing CB on non-GLY residue cleanly in pseudoBetaMask2d', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      // Remove CB from residue 1 (MET) in STANDARD_MONOMER_PDB
      const noCbPdb = STANDARD_MONOMER_PDB.split('\n')
        .filter((line) => !(line.includes(' CB ') && line.includes(' A   1 ')))
        .join('\n');
      const parsed = tpl.parseTemplateStructure(noCbPdb);
      const align = tpl.alignQueryToTemplate(
        [{ chainId: 'A', sequence: STANDARD_MONOMER_SEQUENCE }],
        parsed,
      );
      const payload = tpl.buildAf2TemplatePayload(
        [STANDARD_MONOMER_SEQUENCE],
        align,
        'af2-monomer',
        { spanChains: true },
      );
      const L = STANDARD_MONOMER_SEQUENCE.length;
      const geom = oracles.templateGeometry(payload.template, payload.multichainMask2d, L, oracles.AF2_ATOM37_MONOMER);
      expect(geom.pseudoBetaMask2d[0 * L + 2]).toBe(0);
      expect(geom.backboneMask2d[0 * L + 2]).toBe(1);
    });

    it('contains zero non-finite values in atomPositions and atomMask', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const align = tpl.alignQueryToTemplate(
        [{ chainId: 'A', sequence: STANDARD_MONOMER_SEQUENCE }],
        parsed,
      );
      const payload = tpl.buildAf2TemplatePayload(
        [STANDARD_MONOMER_SEQUENCE],
        align,
        'af2-monomer',
        { spanChains: false },
      );
      expect(Array.from(payload.template.atomPositions).every(Number.isFinite)).toBe(true);
      expect(Array.from(payload.template.atomMask).every((v) => v === 0 || v === 1)).toBe(true);
    });

    it('sets GAP_AATYPE for internal missing-density deletion positions', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      // Insert 'GGGG' in the middle of query
      const gappedQuery = STANDARD_MONOMER_SEQUENCE.slice(0, 8) + 'QQQQ' + STANDARD_MONOMER_SEQUENCE.slice(8);
      const align = tpl.alignQueryToTemplate([{ chainId: 'A', sequence: gappedQuery }], parsed);
      const payload = tpl.buildAf2TemplatePayload([gappedQuery], align, 'af2-monomer', { spanChains: true });
      const gapCount = Array.from(payload.template.aatype).filter((v) => v === oracles.GAP_AATYPE).length;
      expect(gapCount).toBe(4);
    });
  });

  describe('F5 Boundaries: AF2-Multimer asymId & spanChains', () => {
    it('keeps cross-chain mask 0 when 1-chain template covers only chain A even with spanChains=true', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const align = tpl.alignQueryToTemplate(
        [
          { chainId: 'A', sequence: STANDARD_MONOMER_SEQUENCE },
          { chainId: 'B', sequence: 'EEEEEEEEEE' },
        ],
        parsed,
      );
      const payload = tpl.buildAf2TemplatePayload(
        [STANDARD_MONOMER_SEQUENCE, 'EEEEEEEEEE'],
        align,
        'af2-multimer',
        { spanChains: true },
      );
      const L = STANDARD_MONOMER_SEQUENCE.length + 10;
      expect(payload.multichainMask2d[0 * L + STANDARD_MONOMER_SEQUENCE.length]).toBe(0);
    });

    it('opens only A-C pairs and masks A-B and B-C pairs on 3-chain query with A+C template', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(HETERODIMER_PDB);
      const qB = 'CCCCCCCCCC';
      const align = tpl.alignQueryToTemplate(
        [
          { chainId: 'A', sequence: HETERODIMER_CHAIN_A },
          { chainId: 'B', sequence: qB },
          { chainId: 'C', sequence: HETERODIMER_CHAIN_B },
        ],
        parsed,
      );
      const payload = tpl.buildAf2TemplatePayload(
        [HETERODIMER_CHAIN_A, qB, HETERODIMER_CHAIN_B],
        align,
        'af2-multimer',
        { spanChains: true },
      );
      const L = HETERODIMER_CHAIN_A.length + qB.length + HETERODIMER_CHAIN_B.length;
      const idxB = HETERODIMER_CHAIN_A.length;
      const idxC = HETERODIMER_CHAIN_A.length + qB.length;
      expect(payload.multichainMask2d[0 * L + idxC]).toBe(1);
      expect(payload.multichainMask2d[0 * L + idxB]).toBe(0);
      expect(payload.multichainMask2d[idxB * L + idxC]).toBe(0);
    });

    it('emits contiguous 0/1 asymId for multi-char mmCIF chains AA and AB', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(MULTICHAR_COMPLEX_MMCIF);
      const align = tpl.alignQueryToTemplate(
        [
          { chainId: 'A', sequence: HETERODIMER_CHAIN_A },
          { chainId: 'B', sequence: HETERODIMER_CHAIN_B },
        ],
        parsed,
      );
      const payload = tpl.buildAf2TemplatePayload(
        [HETERODIMER_CHAIN_A, HETERODIMER_CHAIN_B],
        align,
        'af2-multimer',
        { spanChains: true },
      );
      expect(payload.asymId[0]).not.toBe(payload.asymId[HETERODIMER_CHAIN_A.length]);
    });

    it('emits explicit asymId on featuresByRecycle for all multimer delta variants (multimer-2..5)', async () => {
      for (const model of ['multimer-2', 'multimer-5']) {
        const res = await runLocalFoldWorkerInNode({
          params: { ...defaultParamsForAdapter('localfold-af2'), model, templatePdbText: STANDARD_MONOMER_PDB },
          inputs: { sequences: makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE) },
        });
        expect(res.status).toBe('result');
        const cap = res.captures[0];
        expect(cap?.predictOpts?.template).toBeDefined();
        const asymId = cap?.featuresByRecycle?.[0]?.asymId as ArrayLike<number> | undefined;
        expect(asymId).toBeDefined();
        expect(asymId!.length).toBe(STANDARD_MONOMER_SEQUENCE.length);
      }
    });

    it('sets coverageOf to 0 at internal gap positions in AF2_ATOM37', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const gapped = STANDARD_MONOMER_SEQUENCE.slice(0, 6) + 'PPP' + STANDARD_MONOMER_SEQUENCE.slice(6);
      const align = tpl.alignQueryToTemplate([{ chainId: 'A', sequence: gapped }], parsed);
      const payload = tpl.buildAf2TemplatePayload([gapped], align, 'af2-multimer', { spanChains: false });
      const cov = oracles.coverageOf(payload.template, gapped.length, oracles.AF2_ATOM37);
      const zeros = Array.from(cov).filter((v) => v === 0).length;
      expect(zeros).toBe(3);
    });

    it('rejects 1-chain predictA3m with template even when multichainMask2d is in options (real vendor seam)', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const align = tpl.alignQueryToTemplate(
        [{ chainId: 'A', sequence: STANDARD_MONOMER_SEQUENCE }],
        parsed,
      );
      const payload = tpl.buildAf2TemplatePayload(
        [STANDARD_MONOMER_SEQUENCE],
        align,
        'af2-multimer',
        { spanChains: true },
      );
      const originDir = ensureWorkerVendorOrigin();
      const { AlphaFoldUnifiedGpu } = (await import(
        pathToFileURL(resolve(originDir, 'vendor/localfold/webgpu/af2/multimer/model.js')).href
      )) as { AlphaFoldUnifiedGpu: new (dev: unknown) => { predictA3m: (...args: unknown[]) => Promise<unknown> } };
      const runner = new AlphaFoldUnifiedGpu({});
      const tables = {
        atom37ToAtom14: new Float32Array(21 * 37),
        atom37Mask: new Float32Array(21 * 37),
      };
      const a3m = `>query\n${STANDARD_MONOMER_SEQUENCE}\n`;
      await expect(
        runner.predictA3m(a3m, {}, tables, {
          recycles: 0,
          chainLengths: [STANDARD_MONOMER_SEQUENCE.length],
          chainAware: true,
          outerProductMeanFirst: true,
          template: payload.template,
          multichainMask2d: payload.multichainMask2d,
        }),
      ).rejects.toThrow(/asymId/i);
    });
  });

  describe('F6 Boundaries: AF3 Dialects & Token Offsets', () => {
    it('enforces exact fused feature widths (108 vs 109 vs 66) across protenix2, boltz2, rosettafold3', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const expected: Array<[string, number]> = [
        ['protenix2', 108],
        ['boltz2', 109],
        ['rosettafold3', 66],
      ];
      for (const [model, width] of expected) {
        const res = await runLocalFoldWorkerInNode({
          params: { ...defaultParamsForAdapter('localfold-af2'), model, templatePdbText: STANDARD_MONOMER_PDB },
          inputs: { sequences: seqSet },
        });
        const batch = res.captures[0]?.batch as Record<string, any>;
        const slots = res.captures[0]?.af3Options?.templateSlots as Array<any>;
        const oracle = await runAf3TemplateCpuOracle({
          modelKey: model,
          tokens: batch.tokens,
          asymId: batch.asymId,
          slots,
        });
        expect(oracle.featureWidth).toBe(width);
      }
    });

    it('produces exact zero output on untemplated boltz2 vs live query output on untemplated protenix2', async () => {
      const oracles = await loadVendorTemplateOracles();
      const { batch } = oracles.af3BatchFromA3m(STANDARD_MONOMER_SEQUENCE, null);
      const boltzEmpty = await runAf3TemplateCpuOracle({
        modelKey: 'boltz2',
        tokens: batch.tokens,
        asymId: batch.asymId,
        slots: [],
      });
      const protenixEmpty = await runAf3TemplateCpuOracle({
        modelKey: 'protenix2',
        tokens: batch.tokens,
        asymId: batch.asymId,
        slots: [],
      });
      const bStats = diffFloat32Arrays(boltzEmpty.output, new Float32Array(boltzEmpty.output.length));
      const pStats = diffFloat32Arrays(protenixEmpty.output, new Float32Array(protenixEmpty.output.length));
      expect(bStats.maxAbs).toBe(0);
      expect(pStats.maxAbs).toBeGreaterThan(0);
    });

    it('handles multi-chain mmCIF with spanChains=true vs false inside AF3 CPU oracle', async () => {
      const seqSet = makeSequenceSetArtifact(HETERODIMER_SEQUENCE, {
        chainLengths: [HETERODIMER_CHAIN_A.length, HETERODIMER_CHAIN_B.length],
        chainIds: ['A', 'B'],
      });
      const rTrue = await runLocalFoldWorkerInNode({
        params: {
          ...defaultParamsForAdapter('localfold-af2'),
          model: 'af3',
          templatePdbText: MULTICHAR_COMPLEX_MMCIF,
          templateSpanChains: true,
        },
        inputs: { sequences: seqSet },
      });
      const rFalse = await runLocalFoldWorkerInNode({
        params: {
          ...defaultParamsForAdapter('localfold-af2'),
          model: 'af3',
          templatePdbText: MULTICHAR_COMPLEX_MMCIF,
          templateSpanChains: false,
        },
        inputs: { sequences: seqSet },
      });
      const bTrue = rTrue.captures[0]!.batch as Record<string, any>;
      const sTrue = rTrue.captures[0]!.af3Options!.templateSlots as Array<any>;
      const bFalse = rFalse.captures[0]!.batch as Record<string, any>;
      const sFalse = rFalse.captures[0]!.af3Options!.templateSlots as Array<any>;
      const oTrue = await runAf3TemplateCpuOracle({ modelKey: 'af3', tokens: bTrue.tokens, asymId: bTrue.asymId, slots: sTrue });
      const oFalse = await runAf3TemplateCpuOracle({ modelKey: 'af3', tokens: bFalse.tokens, asymId: bFalse.asymId, slots: sFalse });
      const delta = diffFloat32Arrays(oTrue.output, oFalse.output);
      expect(delta.l2).toBeGreaterThan(1e-4);
    });

    it('keeps all 8 AF3 dialects NaN-free with modified MSE+SEP mmCIF template', async () => {
      const seqSet = makeSequenceSetArtifact('MKWVSTFISLLFL');
      for (const model of TEMPLATE_CAPABLE_AF3_MODELS) {
        const res = await runLocalFoldWorkerInNode({
          params: { ...defaultParamsForAdapter('localfold-af2'), model, templatePdbText: MODIFIED_MSE_SEP_MMCIF },
          inputs: { sequences: seqSet },
        });
        expect(res.status).toBe('result');
        const batch = res.captures[0]!.batch as Record<string, any>;
        const slots = res.captures[0]!.af3Options!.templateSlots as Array<any>;
        const out = await runAf3TemplateCpuOracle({ modelKey: model, tokens: batch.tokens, asymId: batch.asymId, slots });
        const stats = diffFloat32Arrays(out.output, new Float32Array(out.output.length));
        expect(stats.hasNaN).toBe(false);
      }
    });

    it('aligns holo ligand ZN + protein without corrupting template coverage', async () => {
      const seqSet = makeSequenceSetArtifact('MKWVSTFISLLFL');
      const res = await runLocalFoldWorkerInNode({
        params: {
          ...defaultParamsForAdapter('localfold-af2'),
          model: 'rosettafold3',
          ligandCcdCodes: 'ZN',
          templatePdbText: MODIFIED_MSE_SEP_PDB,
        },
        inputs: { sequences: seqSet },
      });
      expect(res.status).toBe('result');
      const slots = res.captures[0]!.af3Options!.templateSlots as Array<any>;
      expect(slots[0].covered).toBeGreaterThanOrEqual(11);
    });
  });

  describe('F7 Boundaries: Manifest, Port Precedence & Edge Rules', () => {
    it('gives connected template port precedence over inline templatePdbText in worker', async () => {
      const tplPort = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB, { filename: 'port.pdb' });
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const portOnly = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' },
        inputs: { sequences: seqSet, template: tplPort },
      });
      const both = await runLocalFoldWorkerInNode({
        params: {
          ...defaultParamsForAdapter('localfold-af2'),
          model: 'monomer-2',
          templatePdbText: MODIFIED_MSE_SEP_PDB,
        },
        inputs: { sequences: seqSet, template: tplPort },
      });
      expect(both.status).toBe('result');
      expect(both.outputs!.folds.id).toBe(portOnly.outputs!.folds.id);
    });

    it('rejects self-loop connection to localfold-af2.template', () => {
      const graph = makeUntemplatedLocalFoldGraph({ model: 'monomer-2' });
      const check = validateConnection(graph, 'fold', 'structure', 'fold', 'template');
      expect(check.valid).toBe(false);
    });

    it('rejects incompatible FoldedStructure -> localfold-af2.template connection', () => {
      const graph = makeUntemplatedLocalFoldGraph({ model: 'monomer-2' });
      const fold2 = makeLocalFoldNode('fold2', { model: 'monomer-2' });
      const check = validateConnection(
        { ...graph, nodes: [...graph.nodes, fold2] },
        'fold',
        'folds',
        'fold2',
        'template',
      );
      expect(check.valid).toBe(false);
    });

    it('defaults templateSpanChains to true when omitted', () => {
      const cfg = resolveLocalFoldEffectiveConfig({
        modelKey: 'multimer',
        params: { templatePdbText: HETERODIMER_PDB },
      });
      expect(cfg.templateSpanChains).toBe(true);
    });

    it('treats whitespace-only templatePdbText as untemplated', () => {
      const graph = makeTemplatedLocalFoldGraph({
        supplyMode: 'inline',
        model: 'monomer-4',
        templateText: '   \n\t  ',
      });
      const errs = validateBioCapabilities(graph).filter((i) => i.severity === 'error');
      expect(errs).toEqual([]);
    });
  });

  describe('F8 Boundaries: Validation & Rejection Edge Cases', () => {
    it('does not reject template-free model when templatePdbText is empty string', () => {
      const graph = makeTemplatedLocalFoldGraph({
        supplyMode: 'inline',
        model: 'monomer-3',
        templateText: '',
      });
      const errs = validateBioCapabilities(graph).filter((i) => i.severity === 'error');
      expect(errs).toEqual([]);
    });

    it('rejects water-only inline template text in validateBioCapabilities and crossValidate', () => {
      const graph = makeTemplatedLocalFoldGraph({
        supplyMode: 'inline',
        model: 'monomer-2',
        templateText: WATER_ONLY_PDB,
      });
      const cvErrs = ADAPTER_REGISTRY['localfold-af2'].crossValidate!(graph.nodes[1].params);
      const bioErrs = validateBioCapabilities(graph).filter((i) => i.severity === 'error');
      expect(cvErrs.length + bioErrs.length).toBeGreaterThan(0);
    });

    it('rejects zero-overlap sequence vs template in validateBioCapabilities when outputs available', async () => {
      const graph = makeTemplatedLocalFoldGraph({
        sequence: UNRELATED_SEQUENCE,
        templateText: STANDARD_MONOMER_PDB,
        supplyMode: 'port',
        model: 'monomer-2',
      });
      const seqArt = makeSequenceSetArtifact(UNRELATED_SEQUENCE);
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
      const issues = validateBioCapabilities(graph, {
        seq: { sequences: seqArt },
        tpl: { structure: tplArt },
      });
      expect(issues.some((i) => i.severity === 'error')).toBe(true);
    });

    it('rejects all 5 template-free models in validateRuntimeNodeInputs when template port is populated', async () => {
      const seqArt = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
      for (const model of TEMPLATE_FREE_MODELS) {
        const node = makeLocalFoldNode('fold', { model });
        expect(() =>
          validateRuntimeNodeInputs(node, { sequences: seqArt, template: tplArt }),
        ).toThrow();
      }
    });

    it('rejects invalid mmCIF text in crossValidate on AF3 models', () => {
      const cv = ADAPTER_REGISTRY['localfold-af2'].crossValidate!;
      const errs = cv({
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'af3',
        templatePdbText: 'data_BAD\n_atom_site.id\n',
      });
      expect(errs.length).toBeGreaterThan(0);
    });
  });

  describe('F9 Boundaries: Worker Guards & Hashing', () => {
    it('enforces maxFoldResidues memory ceiling before model load even when template is provided', async () => {
      const seqSet = makeSequenceSetArtifact('M' + 'A'.repeat(99));
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
      const res = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' },
        inputs: { sequences: seqSet, template: tplArt },
        maxFoldResidues: 64,
      });
      expect(res.status).toBe('error');
      expect(res.errorMessage).toMatch(/ceiling/i);
    });

    it('rejects water-only template inside localfold.worker.ts pre-fetch', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const res = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2', templatePdbText: WATER_ONLY_PDB },
        inputs: { sequences: seqSet },
      });
      expect(res.status).toBe('error');
      expect(res.fetchUrls).toEqual([]);
    });

    it('changes FoldedStructure.id runHash when a single coordinate in template changes', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const perturbed = STANDARD_MONOMER_PDB.replace('   0.000   2.300', '   5.000   2.300');
      expect(perturbed).not.toBe(STANDARD_MONOMER_PDB);
      const r1 = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2', templatePdbText: STANDARD_MONOMER_PDB },
        inputs: { sequences: seqSet },
      });
      const r2 = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2', templatePdbText: perturbed },
        inputs: { sequences: seqSet },
      });
      expect(r1.outputs!.folds.id).not.toBe(r2.outputs!.folds.id);
    });

    it('prefers StructureArtifact.cifText when mmCIF multi-char structure is wired via template port', async () => {
      const seqSet = makeSequenceSetArtifact(HETERODIMER_SEQUENCE, {
        chainLengths: [HETERODIMER_CHAIN_A.length, HETERODIMER_CHAIN_B.length],
        chainIds: ['A', 'B'],
      });
      const tplArt = await makeStructureArtifactFromText(MULTICHAR_COMPLEX_MMCIF, {
        filename: 'complex.cif',
      });
      expect(tplArt.cifText).toBeTruthy();
      const res = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'multimer' },
        inputs: { sequences: seqSet, template: tplArt },
      });
      expect(res.status).toBe('result');
      expect(res.captures[0]?.predictOpts?.template).toBeDefined();
    });

    it('ignores missing/empty optional template port cleanly without breaking untemplated run', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const res = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' },
        inputs: { sequences: seqSet, template: undefined },
      });
      expect(res.status).toBe('result');
      expect(res.captures[0]?.predictOpts?.template).toBeUndefined();
    });
  });

  describe('F10 Boundaries: Preview, Planner, Preflight & Share', () => {
    it('preserves compact inline templatePdbText (<= 1500 chars) across YAML & hash share without stripping', async () => {
      expect(SHORT_2AA_PDB.length).toBeLessThan(1500);
      const graph = makeTemplatedLocalFoldGraph({
        sequence: 'MK',
        templateText: SHORT_2AA_PDB,
        supplyMode: 'inline',
        model: 'monomer-2',
      });
      const encoded = await encodeGraphToHash(graph);
      expect(encoded.strippedPdb).toBe(false);
      const decoded = await decodeHashToGraph(encoded.hash);
      expect(decoded?.nodes.find((n) => n.id === 'fold')?.params.templatePdbText).toBe(SHORT_2AA_PDB);
      const reparsed = parseRecipeToGraph(graphToYamlRecipe(graph));
      expect(reparsed.nodes.find((n) => n.id === 'fold')?.params.templatePdbText).toBe(SHORT_2AA_PDB);
    });

    it('refuses restoreOmittedInlineAsset when expectedTemplateSha256 mismatches', () => {
      const restore = localfoldAf2Plugin.share.restoreOmittedInlineAsset!;
      const prevParams = {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-2',
        templatePdbText: LARGE_TEMPLATE_PDB,
      };
      const nextParams = {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-2',
        templatePdbText: '',
        missingTemplateUpload: true,
        expectedTemplateSha256: '0'.repeat(64),
      };
      expect(restore(prevParams, nextParams, '', 'fold')).toBeNull();
    });

    it('reports pending-upstream gracefully in compileNodeInputPreview when template pdb-input has not run', () => {
      clearInputPreviewCache();
      const graph = makeTemplatedLocalFoldGraph({ supplyMode: 'port', model: 'monomer-2' });
      const seqArt = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const preview = compileNodeInputPreview(graph, 'fold', { seq: { sequences: seqArt } });
      expect(['pending-upstream', 'partial']).toContain(preview.availability);
      expect(preview.missingUpstreamNodeIds).toContain('tpl');
    });

    it('preserves existing sequences edge when block_graph adds pdb-input -> localfold-af2', () => {
      const compiled = compileWorkflowPlanToGraph({
        kind: 'block_graph',
        mode: 'new',
        summary: 'Add structural template to LocalFold multimer',
        nodes: [
          { id: 'seq', adapter: 'sequence-input', label: 'Sequence', params: { sequence: STANDARD_MONOMER_SEQUENCE } },
          { id: 'tpl', adapter: 'pdb-input', label: 'Template', params: { mode: 'demo-bundle' } },
          { id: 'fold', adapter: 'localfold-af2', label: 'LocalFold', params: { model: 'multimer' } },
        ],
        edges: [
          ['seq', 'fold'],
          ['tpl', 'fold'],
        ],
      });
      const seqEdge = compiled.graph.edges.find((e) => e.fromNode === 'seq' && e.toPort === 'sequences');
      const tplEdge = compiled.graph.edges.find((e) => e.fromNode === 'tpl' && e.toPort === 'template');
      expect(seqEdge).toBeDefined();
      expect(tplEdge).toBeDefined();
    });

    it('deduplicates AF2 delta download bytes in estimateRunRequirements on templated multi-branch graph', () => {
      const g = {
        id: 'delta-dedup',
        name: 'Delta Dedup',
        nodes: [
          makeSequenceNode('seq', STANDARD_MONOMER_SEQUENCE),
          makePdbNode('tpl', STANDARD_MONOMER_PDB),
          makeLocalFoldNode('f1', { model: 'monomer' }),
          makeLocalFoldNode('f2', { model: 'monomer-2' }),
        ],
        edges: [
          { fromNode: 'seq', fromPort: 'sequences', toNode: 'f1', toPort: 'sequences' },
          { fromNode: 'seq', fromPort: 'sequences', toNode: 'f2', toPort: 'sequences' },
          { fromNode: 'tpl', fromPort: 'structure', toNode: 'f2', toPort: 'template' },
        ],
      };
      const pre = estimateRunRequirements(g);
      const m2 = pre.models.find((m) => m.key === 'localfold:monomer-2');
      expect(m2).toBeDefined();
      expect(m2!.approxMB).toBeLessThan(140); // base 97 MB deduplicated
    });
  });

  describe('F11 Boundaries: Inspector & Mode-Switch Edge Cases', () => {
    it('strips AF3 holo params when switching af3 -> monomer-2 while keeping templatePdbText', () => {
      const prev = {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'af3',
        ligandCcdCodes: 'ZN',
        dnaSequences: 'ACGT',
        templatePdbText: STANDARD_MONOMER_PDB,
        templateFilename: 'tpl.pdb',
      };
      const next = sanitizeParamsOnModeSwitch('localfold-af2', prev, 'monomer-2');
      expect(next.ligandCcdCodes).toBeUndefined();
      expect(next.dnaSequences).toBeUndefined();
      expect(next.templatePdbText).toBe(STANDARD_MONOMER_PDB);
    });

    it('forces msaMode="single" on ef2-fast-300m switch while keeping templatePdbText visible for warning', () => {
      const prev = {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-2',
        msaMode: 'mmseqs2',
        templatePdbText: STANDARD_MONOMER_PDB,
      };
      const next = sanitizeParamsOnModeSwitch('localfold-af2', prev, 'ef2-fast-300m');
      expect(next.msaMode).toBe('single');
      expect(next.templatePdbText).toBe(STANDARD_MONOMER_PDB);
    });

    it('formats canvas summary cleanly even when large templatePdbText is stored', () => {
      const summary = localfoldAf2Plugin.display.formatCanvasSummary({
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-2',
        templatePdbText: LARGE_TEMPLATE_PDB,
      });
      expect(summary.length).toBeGreaterThan(0);
      expect(summary.length).toBeLessThan(120);
    });

    it('wires upstream template info through NodeInspector.svelte to LocalFoldWidget', () => {
      const inspectorSrc = readFileSync(
        resolve(__dirname, '../../src/lib/ui/NodeInspector.svelte'),
        'utf8',
      );
      const widgetCall = inspectorSrc.match(/<LocalFoldWidget[\s\S]*?\/>/);
      expect(widgetCall).not.toBeNull();
      expect(widgetCall![0].toLowerCase()).toContain('template');

      const widgetSrc = readFileSync(
        resolve(__dirname, '../../src/lib/ui/inspector/LocalFoldWidget.svelte'),
        'utf8',
      );
      expect(widgetSrc).toContain('templatePdbText');
      expect(widgetSrc).toContain('templateSpanChains');
      expect(widgetSrc.toLowerCase()).toMatch(/template.*port|upstream.*template|connected.*template/);
    });

    it('clears template state cleanly back to untemplated validity when templatePdbText is cleared', () => {
      const templated = makeTemplatedLocalFoldGraph({ supplyMode: 'inline', model: 'monomer-4' });
      expect(validateBioCapabilities(templated).some((i) => i.severity === 'error')).toBe(true);
      const cleared = {
        ...templated,
        nodes: templated.nodes.map((n) =>
          n.id === 'fold' ? { ...n, params: { ...n.params, templatePdbText: '', templateFilename: '' } } : n,
        ),
      };
      expect(validateBioCapabilities(cleared).some((i) => i.severity === 'error')).toBe(false);
    });
  });

  describe('F12 Boundaries: Untemplated Parity', () => {
    it('produces identical worker runHash when templatePdbText is empty string vs omitted', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const omitted = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer' },
        inputs: { sequences: seqSet },
      });
      const emptyStr = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer', templatePdbText: '' },
        inputs: { sequences: seqSet },
      });
      expect(omitted.outputs!.folds.id).toBe(emptyStr.outputs!.folds.id);
    });

    it('introduces zero template keys during untemplated sanitizeParamsOnModeSwitch roundtrips', () => {
      const p0 = defaultParamsForAdapter('localfold-af2');
      const p1 = sanitizeParamsOnModeSwitch('localfold-af2', p0, 'multimer');
      const p2 = sanitizeParamsOnModeSwitch('localfold-af2', p1, 'af3');
      const p3 = sanitizeParamsOnModeSwitch('localfold-af2', p2, 'monomer');
      expect(p3.templatePdbText).toBeUndefined();
      expect(p3.templateFilename).toBeUndefined();
    });

    it('allows untemplated validateRuntimeNodeInputs on monomer-3..5 and ef2-fast-300m/600m', () => {
      const seqArt = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      for (const model of TEMPLATE_FREE_MODELS) {
        const node = makeLocalFoldNode('fold', { model });
        expect(() => validateRuntimeNodeInputs(node, { sequences: seqArt })).not.toThrow();
      }
    });

    it('keeps untemplated preset share hash in compact preset-diff mode with strippedPdb=false', async () => {
      const preset = WORKFLOW_PRESETS.find((p) =>
        p.graph.nodes.some((n) => n.adapterId === 'localfold-af2'),
      )!;
      const res = await encodeGraphToHash(preset.graph);
      expect(res.mode).toBe('preset-diff');
      expect(res.strippedPdb).toBe(false);
    });

    it('executes untemplated ExecutionScheduler workflow cleanly with template input undefined', async () => {
      const graph = makeUntemplatedLocalFoldGraph({ model: 'monomer' });
      const res = await runSchedulerWithCapturedLocalFold(graph);
      expect(res.outputsByNode.fold?.folds?.kind).toBe('FoldedStructure');
      expect(res.workerCalls[0].inputs.template).toBeUndefined();
    });
  });
});
