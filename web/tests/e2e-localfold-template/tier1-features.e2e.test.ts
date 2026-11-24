import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import {
  ADAPTER_REGISTRY,
  LOCALFOLD_MODEL_OPTIONS,
  WORKFLOW_PRESETS,
} from '$lib/engine/registry';
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
import type { CanonicalArtifact } from '$lib/engine/types';
import {
  ALL_20_LOCALFOLD_MODELS,
  ALTLOC_PDB,
  ARG_FLIPPED_PDB,
  HETERODIMER_CHAIN_A,
  HETERODIMER_CHAIN_B,
  HETERODIMER_PDB,
  HETERODIMER_SEQUENCE,
  HOLO_COMPLEX_MMCIF,
  HOMOLOG_MONOMER_SEQUENCE,
  LARGE_TEMPLATE_PDB,
  MODIFIED_MSE_SEP_PDB,
  MULTICHAR_COMPLEX_MMCIF,
  NMR_MULTIMODEL_PDB,
  STANDARD_MONOMER_MMCIF,
  STANDARD_MONOMER_PDB,
  STANDARD_MONOMER_SEQUENCE,
  TEMPLATE_CAPABLE_AF2_MONO_MODELS,
  TEMPLATE_CAPABLE_AF2_MULTI_MODELS,
  TEMPLATE_CAPABLE_AF3_MODELS,
  TEMPLATE_FREE_MODELS,
  makeLocalFoldNode,
  makeSequenceNode,
  makeSequenceSetArtifact,
  makeStructureArtifactFromText,
  makeTemplatedLocalFoldGraph,
  makeUntemplatedLocalFoldGraph,
} from './fixtures';
import {
  cleanupWorkerVendorOrigin,
  diffFloat32Arrays,
  loadLocalFoldTemplateModule,
  loadVendorTemplateOracles,
  runAf3TemplateCpuOracle,
  runLocalFoldWorkerInNode,
} from './helpers';

afterAll(() => {
  cleanupWorkerVendorOrigin();
});

describe('Tier 1 — Feature Coverage (F1–F12)', () => {
  describe('F1: Template Capability Taxonomy', () => {
    it('classifies AF2 monomer models 1 and 2 as af2-monomer template-capable', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      for (const m of TEMPLATE_CAPABLE_AF2_MONO_MODELS) {
        expect(tpl.isLocalFoldTemplateCapable(m)).toBe(true);
        expect(tpl.localFoldTemplateFamily(m)).toBe('af2-monomer');
      }
    });

    it('classifies AF2 monomer models 3–5 as template-free', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      for (const m of ['monomer-3', 'monomer-4', 'monomer-5']) {
        expect(tpl.isLocalFoldTemplateCapable(m)).toBe(false);
        expect(tpl.localFoldTemplateFamily(m)).toBe('none');
      }
    });

    it('classifies all 5 AF2-Multimer checkpoints as af2-multimer template-capable', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      for (const m of TEMPLATE_CAPABLE_AF2_MULTI_MODELS) {
        expect(tpl.isLocalFoldTemplateCapable(m)).toBe(true);
        expect(tpl.localFoldTemplateFamily(m)).toBe('af2-multimer');
      }
    });

    it('classifies all 8 AF3-family models as af3 template-capable', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      for (const m of TEMPLATE_CAPABLE_AF3_MODELS) {
        expect(tpl.isLocalFoldTemplateCapable(m)).toBe(true);
        expect(tpl.localFoldTemplateFamily(m)).toBe('af3');
      }
    });

    it('classifies ESMFold2 models as template-free', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      for (const m of ['ef2-fast-300m', 'ef2-fast-600m']) {
        expect(tpl.isLocalFoldTemplateCapable(m)).toBe(false);
        expect(tpl.localFoldTemplateFamily(m)).toBe('none');
      }
    });

    it('partitions all 20 LOCALFOLD_MODEL_OPTIONS consistently (15 capable, 5 template-free)', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      expect(LOCALFOLD_MODEL_OPTIONS.map((m) => m.id)).toEqual([...ALL_20_LOCALFOLD_MODELS]);
      const capable = ALL_20_LOCALFOLD_MODELS.filter((m) => tpl.isLocalFoldTemplateCapable(m));
      const free = ALL_20_LOCALFOLD_MODELS.filter((m) => !tpl.isLocalFoldTemplateCapable(m));
      expect(capable).toHaveLength(15);
      expect(free).toEqual([...TEMPLATE_FREE_MODELS]);
    });
  });

  describe('F2: PDB & mmCIF All-Atom Template Parser', () => {
    it('parses standard PDB with heavy atoms (N, CA, C, O, CB, sidechains) and B-factors', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      expect(parsed.format).toBe('pdb');
      expect(parsed.chains).toHaveLength(1);
      expect(parsed.chains[0].chainId).toBe('A');
      expect(parsed.chains[0].sequence).toBe(STANDARD_MONOMER_SEQUENCE);
      expect(parsed.chains[0].residues).toHaveLength(STANDARD_MONOMER_SEQUENCE.length);
      const argIdx = STANDARD_MONOMER_SEQUENCE.indexOf('R');
      const argAtoms = parsed.chains[0].residues[argIdx].atoms;
      expect(argAtoms.has('N')).toBe(true);
      expect(argAtoms.has('CA')).toBe(true);
      expect(argAtoms.has('C')).toBe(true);
      expect(argAtoms.has('CB')).toBe(true);
      expect(argAtoms.has('NH1')).toBe(true);
      expect(parsed.chains[0].residues[0].confidence).toBeCloseTo(88.0, 1);
    });

    it('parses mmCIF _atom_site preserving multi-character chain IDs (AA, AB)', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(MULTICHAR_COMPLEX_MMCIF);
      expect(parsed.format).toBe('mmcif');
      expect(parsed.chains.map((c) => c.chainId)).toEqual(['AA', 'AB']);
      expect(parsed.chains[0].sequence).toBe(HETERODIMER_CHAIN_A);
      expect(parsed.chains[1].sequence).toBe(HETERODIMER_CHAIN_B);
    });

    it('stops at ENDMDL on NMR multi-model PDB and reads only MODEL 1', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(NMR_MULTIMODEL_PDB);
      expect(parsed.chains[0].residues).toHaveLength(STANDARD_MONOMER_SEQUENCE.length);
      const ca0 = parsed.chains[0].residues[0].atoms.get('CA')!;
      expect(Math.abs(ca0[0])).toBeLessThan(20); // MODEL 2 is shifted by +500
    });

    it('parses MSE (SE -> SD) and SEP modified residues via HETATM', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(MODIFIED_MSE_SEP_PDB);
      const seq = parsed.chains[0].sequence;
      expect(seq[0]).toBe('M');
      expect(seq[4]).toBe('S');
      expect(parsed.chains[0].residues[0].resName).toBe('MSE');
      expect(parsed.chains[0].residues[0].atoms.has('SD')).toBe(true);
      expect(parsed.chains[0].residues[4].resName).toBe('SEP');
    });

    it('selects highest-occupancy altLoc by default and respects altlocFirst option', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const byOcc = tpl.parseTemplateStructure(ALTLOC_PDB, { altlocFirst: false });
      const firstPolicy = tpl.parseTemplateStructure(ALTLOC_PDB, { altlocFirst: true });
      // Residue 1: altLoc A has occ 0.30 at x=12, altLoc B has occ 0.70 at x=0
      const caOcc = byOcc.chains[0].residues[0].atoms.get('CA')!;
      const caFirst = firstPolicy.chains[0].residues[0].atoms.get('CA')!;
      expect(Math.abs(caOcc[0])).toBeLessThan(2);
      expect(caFirst[0]).toBeGreaterThan(8);
    });

    it('parses holo mmCIF (protein + SEP + DNA + ZN + HOH) keeping only polymer protein residues', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(HOLO_COMPLEX_MMCIF);
      expect(parsed.format).toBe('mmcif');
      expect(parsed.chains).toHaveLength(1);
      expect(parsed.chains[0].chainId).toBe('A');
      expect(parsed.chains[0].sequence).toBe('MKWVSTFISLLFL');
      expect(parsed.chains[0].residues[4].resName).toBe('SEP');
    });

    it('normalizes ARG NH1/NH2 ordering relative to CD when flipped', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(ARG_FLIPPED_PDB);
      const argRes = parsed.chains[0].residues[3];
      const cd = argRes.atoms.get('CD')!;
      const nh1 = argRes.atoms.get('NH1')!;
      const nh2 = argRes.atoms.get('NH2')!;
      const d1 = Math.hypot(nh1[0] - cd[0], nh1[1] - cd[1], nh1[2] - cd[2]);
      const d2 = Math.hypot(nh2[0] - cd[0], nh2[1] - cd[1], nh2[2] - cd[2]);
      expect(d1).toBeLessThanOrEqual(d2 + 1e-5);
    });
  });

  describe('F3: Chain Matching & Sequence-to-Template Alignment', () => {
    it('achieves 100% identity and 100% coverage on exact monomer match', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const res = tpl.alignQueryToTemplate(
        [{ chainId: 'A', sequence: STANDARD_MONOMER_SEQUENCE }],
        parsed,
      );
      expect(res.matchedQueryResidues).toBe(STANDARD_MONOMER_SEQUENCE.length);
      expect(res.totalQueryResidues).toBe(STANDARD_MONOMER_SEQUENCE.length);
      expect(res.chainAlignments[0].coverage).toBeCloseTo(1.0, 4);
      expect(res.chainAlignments[0].identity).toBeCloseTo(1.0, 4);
    });

    it('aligns homolog with N-terminal tag, point mutations, and deletion accurately', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const res = tpl.alignQueryToTemplate(
        [{ chainId: 'A', sequence: HOMOLOG_MONOMER_SEQUENCE }],
        parsed,
      );
      expect(res.matchedQueryResidues).toBeGreaterThanOrEqual(17);
      expect(res.chainAlignments[0].coverage).toBeGreaterThan(0.75);
      expect(res.chainAlignments[0].coverage).toBeLessThan(0.95);
    });

    it('matches multi-chain heterodimer query chains to template chains A and B', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(HETERODIMER_PDB);
      const res = tpl.alignQueryToTemplate(
        [
          { chainId: 'A', sequence: HETERODIMER_CHAIN_A },
          { chainId: 'B', sequence: HETERODIMER_CHAIN_B },
        ],
        parsed,
      );
      expect(res.chainAlignments).toHaveLength(2);
      expect(res.chainAlignments[0].templateChainId).toBe('A');
      expect(res.chainAlignments[1].templateChainId).toBe('B');
      expect(res.matchedQueryResidues).toBe(
        HETERODIMER_CHAIN_A.length + HETERODIMER_CHAIN_B.length,
      );
    });

    it('assigns bipartite chain match when query chain order is permuted vs template', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(HETERODIMER_PDB);
      const res = tpl.alignQueryToTemplate(
        [
          { chainId: 'A', sequence: HETERODIMER_CHAIN_B },
          { chainId: 'B', sequence: HETERODIMER_CHAIN_A },
        ],
        parsed,
      );
      expect(res.chainAlignments[0].templateChainId).toBe('B');
      expect(res.chainAlignments[1].templateChainId).toBe('A');
      expect(res.matchedQueryResidues).toBe(
        HETERODIMER_CHAIN_A.length + HETERODIMER_CHAIN_B.length,
      );
    });

    it('supports subset matching (1-chain template covering 1 chain of a 2-chain query)', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const res = tpl.alignQueryToTemplate(
        [
          { chainId: 'A', sequence: STANDARD_MONOMER_SEQUENCE },
          { chainId: 'B', sequence: 'DDDDDDDDDDDD' },
        ],
        parsed,
      );
      expect(res.matchedQueryResidues).toBe(STANDARD_MONOMER_SEQUENCE.length);
      expect(res.totalQueryResidues).toBe(STANDARD_MONOMER_SEQUENCE.length + 12);
    });
  });

  describe('F4: AF2 Monomer (monomer, monomer-2) Atom37 Slot Builder', () => {
    it('builds AF2 atom37 payload with exact dimensions [L, L*37*3, L*37]', async () => {
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
        { spanChains: true },
      );
      const L = STANDARD_MONOMER_SEQUENCE.length;
      expect(payload.template.aatype).toHaveLength(L);
      expect(payload.template.atomPositions).toHaveLength(L * 37 * 3);
      expect(payload.template.atomMask).toHaveLength(L * 37);
    });

    it('sets GAP_AATYPE (21) and zero atomMask on unaligned N-terminal tag residues', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const align = tpl.alignQueryToTemplate(
        [{ chainId: 'A', sequence: HOMOLOG_MONOMER_SEQUENCE }],
        parsed,
      );
      const payload = tpl.buildAf2TemplatePayload(
        [HOMOLOG_MONOMER_SEQUENCE],
        align,
        'af2-monomer',
        { spanChains: true },
      );
      // First 3 residues ('GSH') are N-terminal tag not in template
      for (let i = 0; i < 3; i++) {
        expect(payload.template.aatype[i]).toBe(oracles.GAP_AATYPE);
        let maskSum = 0;
        for (let s = 0; s < 37; s++) maskSum += payload.template.atomMask[i * 37 + s];
        expect(maskSum).toBe(0);
      }
      // Aligned core position has valid standard aatype < 20
      expect(payload.template.aatype[4]).toBeLessThan(20);
    });

    it('produces valid non-zero distogram & backboneMask2d in vendored templateGeometry (AF2_ATOM37_MONOMER)', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const align = tpl.alignQueryToTemplate(
        [{ chainId: 'A', sequence: HOMOLOG_MONOMER_SEQUENCE }],
        parsed,
      );
      const payload = tpl.buildAf2TemplatePayload(
        [HOMOLOG_MONOMER_SEQUENCE],
        align,
        'af2-monomer',
        { spanChains: true },
      );
      const L = HOMOLOG_MONOMER_SEQUENCE.length;
      const geom = oracles.templateGeometry(
        payload.template,
        payload.multichainMask2d,
        L,
        oracles.AF2_ATOM37_MONOMER,
      );
      // Tag residue 0 vs core residue 5 has backboneMask2d = 0
      expect(geom.backboneMask2d[0 * L + 5]).toBe(0);
      // Core pair (4, 8) has backboneMask2d = 1 and a non-zero distogram bin
      expect(geom.backboneMask2d[4 * L + 8]).toBe(1);
      let binSum = 0;
      for (let b = 0; b < 39; b++) binSum += geom.distogram[(4 * L + 8) * 39 + b];
      expect(binSum).toBe(1);
    });

    it('uses CA as pseudo-beta for GLY residues in AF2 monomer geometry', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
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
      const glyIdx = STANDARD_MONOMER_SEQUENCE.indexOf('G');
      expect(glyIdx).toBeGreaterThan(0);
      const geom = oracles.templateGeometry(
        payload.template,
        payload.multichainMask2d,
        L,
        oracles.AF2_ATOM37_MONOMER,
      );
      expect(geom.pseudoBetaMask2d[0 * L + glyIdx]).toBe(1);
    });

    it('packs AF2 monomer geometry with packTemplateGeometry yielding bin+1 > 0 for covered pairs', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
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
      const geom = oracles.templateGeometry(
        payload.template,
        payload.multichainMask2d,
        L,
        oracles.AF2_ATOM37_MONOMER,
      );
      const packed = oracles.packTemplateGeometry(geom, L);
      expect(packed).toHaveLength(L * L * 6);
      expect(packed[(0 * L + 3) * 6]).toBeGreaterThan(0);
    });
  });

  describe('F5: AF2-Multimer (multimer..multimer-5) Atom37 & asymId Guard', () => {
    it('synthesizes explicit asymId and multichainMask2d on 1-chain multimer runs to prevent crash', async () => {
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
        { spanChains: false },
      );
      const L = STANDARD_MONOMER_SEQUENCE.length;
      expect(payload.asymId).toHaveLength(L);
      expect(payload.multichainMask2d).toHaveLength(L * L);
      expect(payload.multichainMask2d[0]).toBe(1);
    });

    it('opens cross-chain multichainMask2d pairs when spanChains is true on co-crystal', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(HETERODIMER_PDB);
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
      const L = HETERODIMER_CHAIN_A.length + HETERODIMER_CHAIN_B.length;
      const crossIdx = 0 * L + HETERODIMER_CHAIN_A.length;
      expect(payload.multichainMask2d[crossIdx]).toBe(1);
    });

    it('masks cross-chain multichainMask2d pairs when spanChains is false', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(HETERODIMER_PDB);
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
        { spanChains: false },
      );
      const L = HETERODIMER_CHAIN_A.length + HETERODIMER_CHAIN_B.length;
      const intraIdx = 0 * L + 2;
      const crossIdx = 0 * L + HETERODIMER_CHAIN_A.length;
      expect(payload.multichainMask2d[intraIdx]).toBe(1);
      expect(payload.multichainMask2d[crossIdx]).toBe(0);
    });

    it('preserves vs zeroes cross-chain distogram in vendored AF2_ATOM37 templateGeometry', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(HETERODIMER_PDB);
      const align = tpl.alignQueryToTemplate(
        [
          { chainId: 'A', sequence: HETERODIMER_CHAIN_A },
          { chainId: 'B', sequence: HETERODIMER_CHAIN_B },
        ],
        parsed,
      );
      const openPayload = tpl.buildAf2TemplatePayload(
        [HETERODIMER_CHAIN_A, HETERODIMER_CHAIN_B],
        align,
        'af2-multimer',
        { spanChains: true },
      );
      const maskedPayload = tpl.buildAf2TemplatePayload(
        [HETERODIMER_CHAIN_A, HETERODIMER_CHAIN_B],
        align,
        'af2-multimer',
        { spanChains: false },
      );
      const L = HETERODIMER_CHAIN_A.length + HETERODIMER_CHAIN_B.length;
      const gOpen = oracles.templateGeometry(openPayload.template, openPayload.multichainMask2d, L, oracles.AF2_ATOM37);
      const gMasked = oracles.templateGeometry(maskedPayload.template, maskedPayload.multichainMask2d, L, oracles.AF2_ATOM37);
      const crossPair = 0 * L + HETERODIMER_CHAIN_A.length;
      expect(gOpen.backboneMask2d[crossPair]).toBe(1);
      expect(gMasked.backboneMask2d[crossPair]).toBe(0);
    });

    it('keeps uncovered cross-chain pairs masked even when spanChains is true', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB); // 1 chain only
      const align = tpl.alignQueryToTemplate(
        [
          { chainId: 'A', sequence: STANDARD_MONOMER_SEQUENCE },
          { chainId: 'B', sequence: 'DDDDDDDDDD' }, // uncovered
        ],
        parsed,
      );
      const payload = tpl.buildAf2TemplatePayload(
        [STANDARD_MONOMER_SEQUENCE, 'DDDDDDDDDD'],
        align,
        'af2-multimer',
        { spanChains: true },
      );
      const L = STANDARD_MONOMER_SEQUENCE.length + 10;
      const crossPair = 0 * L + STANDARD_MONOMER_SEQUENCE.length + 1;
      expect(payload.multichainMask2d[crossPair]).toBe(0);
    });
  });

  describe('F6: AF3-Family (dense-24) Token-Space templateSlots Builder across 4 Dialect Regimes', () => {
    it('builds valid 9-projection templateSlots for af3 and intellifold2 producing non-zero CPU oracle delta', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      expect(tpl.localFoldTemplateFamily('af3')).toBe('af3');
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      for (const model of ['af3', 'intellifold2']) {
        const res = await runLocalFoldWorkerInNode({
          params: { ...defaultParamsForAdapter('localfold-af2'), model, templatePdbText: STANDARD_MONOMER_PDB },
          inputs: { sequences: seqSet },
        });
        expect(res.status).toBe('result');
        const slots = res.captures[0]?.af3Options?.templateSlots as Array<any> | undefined;
        const batch = res.captures[0]?.batch as Record<string, any>;
        expect(slots?.some(Boolean)).toBe(true);
        const withTpl = await runAf3TemplateCpuOracle({
          modelKey: model,
          tokens: batch.tokens,
          asymId: batch.asymId,
          slots: slots!,
        });
        const withoutTpl = await runAf3TemplateCpuOracle({
          modelKey: model,
          tokens: batch.tokens,
          asymId: batch.asymId,
          slots: [],
        });
        const delta = diffFloat32Arrays(withTpl.output, withoutTpl.output);
        expect(delta.hasNaN).toBe(false);
        expect(delta.l2).toBeGreaterThan(1e-3);
      }
    });

    it('builds valid fused 108-col templateSlots for protenix2 and mol1 with non-zero CPU oracle delta', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      for (const model of ['protenix2', 'mol1']) {
        const res = await runLocalFoldWorkerInNode({
          params: { ...defaultParamsForAdapter('localfold-af2'), model, templatePdbText: STANDARD_MONOMER_PDB },
          inputs: { sequences: seqSet },
        });
        expect(res.status).toBe('result');
        const slots = res.captures[0]?.af3Options?.templateSlots as Array<any>;
        const batch = res.captures[0]?.batch as Record<string, any>;
        expect(slots?.some(Boolean)).toBe(true);
        const withTpl = await runAf3TemplateCpuOracle({
          modelKey: model,
          tokens: batch.tokens,
          asymId: batch.asymId,
          slots,
        });
        const withoutTpl = await runAf3TemplateCpuOracle({
          modelKey: model,
          tokens: batch.tokens,
          asymId: batch.asymId,
          slots: [],
        });
        expect(withTpl.featureWidth).toBe(108);
        const delta = diffFloat32Arrays(withTpl.output, withoutTpl.output);
        expect(delta.hasNaN).toBe(false);
        expect(delta.l2).toBeGreaterThan(1e-3);
      }
    });

    it('builds valid fused 109-col templateSlots for boltz2 with non-zero CPU oracle delta', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const res = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'boltz2', templatePdbText: STANDARD_MONOMER_PDB },
        inputs: { sequences: seqSet },
      });
      expect(res.status).toBe('result');
      const slots = res.captures[0]?.af3Options?.templateSlots as Array<any>;
      const batch = res.captures[0]?.batch as Record<string, any>;
      expect(slots?.some(Boolean)).toBe(true);
      const withTpl = await runAf3TemplateCpuOracle({
        modelKey: 'boltz2',
        tokens: batch.tokens,
        asymId: batch.asymId,
        slots,
      });
      const withoutTpl = await runAf3TemplateCpuOracle({
        modelKey: 'boltz2',
        tokens: batch.tokens,
        asymId: batch.asymId,
        slots: [],
      });
      expect(withTpl.featureWidth).toBe(109);
      const delta = diffFloat32Arrays(withTpl.output, withoutTpl.output);
      expect(delta.hasNaN).toBe(false);
      expect(delta.l2).toBeGreaterThan(1e-3);
    });

    it('builds valid fused 66-col templateSlots for rosettafold3 with non-zero CPU oracle delta', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const res = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'rosettafold3', templatePdbText: STANDARD_MONOMER_PDB },
        inputs: { sequences: seqSet },
      });
      expect(res.status).toBe('result');
      const slots = res.captures[0]?.af3Options?.templateSlots as Array<any>;
      const batch = res.captures[0]?.batch as Record<string, any>;
      expect(slots?.some(Boolean)).toBe(true);
      const withTpl = await runAf3TemplateCpuOracle({
        modelKey: 'rosettafold3',
        tokens: batch.tokens,
        asymId: batch.asymId,
        slots,
      });
      const withoutTpl = await runAf3TemplateCpuOracle({
        modelKey: 'rosettafold3',
        tokens: batch.tokens,
        asymId: batch.asymId,
        slots: [],
      });
      expect(withTpl.featureWidth).toBe(66);
      const delta = diffFloat32Arrays(withTpl.output, withoutTpl.output);
      expect(delta.hasNaN).toBe(false);
      expect(delta.l2).toBeGreaterThan(1e-3);
    });

    it('honors dialect modified-residue policies for openbind0 (unk_restype) and opendde (unk_n)', async () => {
      const seq = 'MKWVSTFISLLFL';
      const seqSet = makeSequenceSetArtifact(seq);
      const obRes = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'openbind0', templatePdbText: MODIFIED_MSE_SEP_PDB },
        inputs: { sequences: seqSet },
      });
      const odRes = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'opendde', templatePdbText: MODIFIED_MSE_SEP_PDB },
        inputs: { sequences: seqSet },
      });
      expect(obRes.status).toBe('result');
      expect(odRes.status).toBe('result');
      const obSlot = (obRes.captures[0]?.af3Options?.templateSlots as Array<any>)[0];
      const odSlot = (odRes.captures[0]?.af3Options?.templateSlots as Array<any>)[0];
      expect(obSlot.aatype[0]).toBe(20); // openbind0 unk_restype: MSE -> UNK
      expect(odSlot.aatype[0]).toBe(12); // opendde unk_n: MSE -> MET (MSE exempt from unk_n)
      expect(obSlot.aatype[4]).toBe(20); // SEP -> UNK under unk_restype
      expect(odSlot.aatype[4]).toBe(20); // SEP -> UNK under unk_n
      // unk_n fills all 24 dense slots with N for SEP; unk_restype fills only [0,1,2,4]
      let obMaskCount = 0;
      let odMaskCount = 0;
      for (let s = 0; s < 24; s++) {
        obMaskCount += obSlot.atomMask[4 * 24 + s];
        odMaskCount += odSlot.atomMask[4 * 24 + s];
      }
      expect(obMaskCount).toBe(4);
      expect(odMaskCount).toBe(24);
    });

    it('indexes templateSlots over batch.tokens in holo AF3 runs and honors non-identity tokenOf shifts', async () => {
      const tpl = await loadLocalFoldTemplateModule();
      const oracles = await loadVendorTemplateOracles();
      const parsed = tpl.parseTemplateStructure(STANDARD_MONOMER_PDB);
      const align = tpl.alignQueryToTemplate(
        [{ chainId: 'A', sequence: STANDARD_MONOMER_SEQUENCE }],
        parsed,
      );
      // Preceding DNA chain shifts protein tokens so tokenOf(r) !== r
      const { batch: dnaFirstBatch } = oracles.af3BatchFromA3m(
        `ACGT:${STANDARD_MONOMER_SEQUENCE}`,
        null,
        { ...oracles.featuriserDialect(oracles.dialectFor('boltz2')), chainKinds: ['dna', 'protein'] },
      );
      const dnaTokenOffset = 4;
      const shiftedSlots = tpl.buildAf3TemplateSlots(
        {
          tokens: dnaFirstBatch.tokens,
          tokenOf: (r: number) => r + dnaTokenOffset,
        },
        align,
        'boltz2',
        { spanChains: true },
      );
      expect(shiftedSlots[0].aatype).toHaveLength(dnaFirstBatch.tokens);
      for (let t = 0; t < dnaTokenOffset; t++) {
        expect(shiftedSlots[0].aatype[t]).toBe(oracles.GAP_AATYPE);
        expect(shiftedSlots[0].atomMask[t * 24 + 1]).toBe(0);
      }
      expect(shiftedSlots[0].aatype[dnaTokenOffset]).toBe(12); // MET shifted to token 4
      expect(shiftedSlots[0].atomMask[dnaTokenOffset * 24 + 1]).toBe(1);

      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const res = await runLocalFoldWorkerInNode({
        params: {
          ...defaultParamsForAdapter('localfold-af2'),
          model: 'boltz2',
          ligandCcdCodes: 'ZN',
          dnaSequences: 'ACGT',
          templatePdbText: STANDARD_MONOMER_PDB,
        },
        inputs: { sequences: seqSet },
      });
      expect(res.status).toBe('result');
      const batch = res.captures[0]?.batch as Record<string, any>;
      const slots = res.captures[0]?.af3Options?.templateSlots as Array<any>;
      const slot = slots[0];
      expect(batch.tokens).toBeGreaterThan(STANDARD_MONOMER_SEQUENCE.length);
      expect(slot.aatype).toHaveLength(batch.tokens);
      // Non-protein tokens (DNA / ligand beyond protein length) must remain GAP_AATYPE with 0 mask
      for (let t = STANDARD_MONOMER_SEQUENCE.length; t < batch.tokens; t++) {
        expect(slot.aatype[t]).toBe(oracles.GAP_AATYPE);
      }
    });
  });

  describe('F7: Node Manifest, Port & Config Contract', () => {
    it('declares optional template: StructureArtifact input port on localfold-af2 manifest', () => {
      const manifest = ADAPTER_REGISTRY['localfold-af2'];
      expect(manifest.inputs.template).toBeDefined();
      expect(manifest.inputs.template.type).toBe('StructureArtifact');
      expect(manifest.inputs.template.required).toBe(false);
    });

    it('allows pdb-input.structure -> localfold-af2.template and rejects sequence-input.sequences -> template', () => {
      const graph = makeTemplatedLocalFoldGraph({ supplyMode: 'port', model: 'monomer-2' });
      const okCheck = validateConnection(
        { ...graph, edges: [] },
        'tpl',
        'structure',
        'fold',
        'template',
      );
      expect(okCheck.valid).toBe(true);
      const badCheck = validateConnection(
        { ...graph, edges: [] },
        'seq',
        'sequences',
        'fold',
        'template',
      );
      expect(badCheck.valid).toBe(false);
    });

    it('validates both port-templated and inline-templated graphs via validateGraph', () => {
      const portGraph = makeTemplatedLocalFoldGraph({ supplyMode: 'port', model: 'monomer-2' });
      const inlineGraph = makeTemplatedLocalFoldGraph({ supplyMode: 'inline', model: 'af3' });
      expect(validateGraph(portGraph)).toEqual([]);
      expect(validateGraph(inlineGraph)).toEqual([]);
    });

    it('resolves template config with port precedence over inline template in conditioning', async () => {
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB, {
        filename: '1ubq_ref.pdb',
      });
      const cfg = resolveLocalFoldEffectiveConfig({
        modelKey: 'monomer-2',
        templateInput: tplArt,
        params: {
          templatePdbText: MODIFIED_MSE_SEP_PDB,
          templateFilename: 'inline.pdb',
          templateSpanChains: false,
        },
      });
      expect(cfg.templateSource).not.toBeNull();
      expect(cfg.templateSource?.sourceKind).toBe('port');
      expect(cfg.templateSource?.bothProvided).toBe(true);
      expect(cfg.templateSource?.text).toBe(STANDARD_MONOMER_PDB);
    });

    it('preserves minimal defaultParamsForAdapter baseline keys without injecting template params', () => {
      const defaults = defaultParamsForAdapter('localfold-af2');
      expect(Object.keys(defaults).sort()).toEqual([
        'capMode',
        'diffusionSteps',
        'maxSequences',
        'model',
        'recycles',
        'seed',
      ]);
    });
  });

  describe('F8: Validation & Template-Free Rejection Rules', () => {
    it('rejects wired template port on template-free models with 1-click BioCompatibilityFix', async () => {
      for (const model of TEMPLATE_FREE_MODELS) {
        const graph = makeTemplatedLocalFoldGraph({ supplyMode: 'port', model });
        const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
        const seqArt = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
        const issues = validateBioCapabilities(graph, {
          tpl: { structure: tplArt },
          seq: { sequences: seqArt },
        });
        const tplErrors = issues.filter((i) => i.severity === 'error');
        expect(tplErrors.length).toBeGreaterThan(0);
        expect(tplErrors.some((i) => Boolean(i.fix?.paramsPatch?.model))).toBe(true);
      }
    });

    it('rejects inline templatePdbText on template-free models in validateBioCapabilities', () => {
      const graph = makeTemplatedLocalFoldGraph({ supplyMode: 'inline', model: 'monomer-4' });
      const issues = validateBioCapabilities(graph);
      const tplErrors = issues.filter((i) => i.severity === 'error');
      expect(tplErrors.length).toBeGreaterThan(0);
      expect(tplErrors[0].fix?.paramsPatch?.model).toBeTruthy();
    });

    it('throws explicit error in validateRuntimeNodeInputs when template is supplied to template-free model', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
      const node = makeLocalFoldNode('fold', { model: 'ef2-fast-300m' });
      expect(() =>
        validateRuntimeNodeInputs(node, { sequences: seqSet, template: tplArt }),
      ).toThrow(/template/i);
    });

    it('reports crossValidate errors for inline template on template-free model or corrupt inline text', () => {
      const cv = ADAPTER_REGISTRY['localfold-af2'].crossValidate!;
      const freeErrs = cv({
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-5',
        templatePdbText: STANDARD_MONOMER_PDB,
      });
      expect(freeErrs.length).toBeGreaterThan(0);
      const corruptErrs = cv({
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-2',
        templatePdbText: 'NOT A VALID STRUCTURE',
      });
      expect(corruptErrs.length).toBeGreaterThan(0);
    });

    it('clears template-free errors after applying BioCompatibilityFix paramsPatch', () => {
      const graph = makeTemplatedLocalFoldGraph({ supplyMode: 'inline', model: 'monomer-3' });
      const issues = validateBioCapabilities(graph);
      const fix = issues.find((i) => i.severity === 'error' && i.fix)?.fix;
      expect(fix).toBeDefined();
      const patchedGraph = {
        ...graph,
        nodes: graph.nodes.map((n) =>
          n.id === fix!.nodeId ? { ...n, params: { ...n.params, ...fix!.paramsPatch } } : n,
        ),
      };
      const afterIssues = validateBioCapabilities(patchedGraph).filter((i) => i.severity === 'error');
      expect(afterIssues).toEqual([]);
    });
  });

  describe('F9: Worker Inference Wiring & Cache Hashing', () => {
    it('passes non-empty AF2_ATOM37_MONOMER options.template to AlphaFoldMonomerGpu on monomer-2', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
      const res = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' },
        inputs: { sequences: seqSet, template: tplArt },
      });
      expect(res.status).toBe('result');
      const call = res.captures.find((c) => c.family === 'af2-monomer');
      expect(call?.predictOpts?.template).toBeDefined();
      const tplObj = call!.predictOpts!.template as { aatype: Int32Array; atomMask: Float32Array };
      expect(tplObj.aatype).toHaveLength(STANDARD_MONOMER_SEQUENCE.length);
      expect(Array.from(tplObj.atomMask).some((v) => v > 0)).toBe(true);
    });

    it('passes template and explicit asymId on 1-chain AF2-Multimer run via predict()', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
      const res = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'multimer' },
        inputs: { sequences: seqSet, template: tplArt },
      });
      expect(res.status).toBe('result');
      const call = res.captures.find((c) => c.family === 'af2-multimer');
      expect(call?.predictOpts?.template).toBeDefined();
      const asymId = call?.featuresByRecycle?.[0]?.asymId as ArrayLike<number> | undefined;
      expect(asymId).toBeDefined();
      expect(asymId!.length).toBe(STANDARD_MONOMER_SEQUENCE.length);
    });

    it('passes populated templateSlots to foldBatch on AF3 inline mmCIF run', async () => {
      const seqSet = makeSequenceSetArtifact(HETERODIMER_SEQUENCE, {
        chainLengths: [HETERODIMER_CHAIN_A.length, HETERODIMER_CHAIN_B.length],
        chainIds: ['A', 'B'],
      });
      const res = await runLocalFoldWorkerInNode({
        params: {
          ...defaultParamsForAdapter('localfold-af2'),
          model: 'boltz2',
          templatePdbText: MULTICHAR_COMPLEX_MMCIF,
          templateFilename: 'complex.cif',
        },
        inputs: { sequences: seqSet },
      });
      expect(res.status).toBe('result');
      const call = res.captures.find((c) => c.family === 'af3');
      const slots = call?.af3Options?.templateSlots as Array<any> | undefined;
      expect(Array.isArray(slots) && slots.some(Boolean)).toBe(true);
    });

    it('rejects template on template-free models pre-fetch without network weight downloads', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB);
      for (const model of ['monomer-3', 'ef2-fast-300m']) {
        const res = await runLocalFoldWorkerInNode({
          params: { ...defaultParamsForAdapter('localfold-af2'), model },
          inputs: { sequences: seqSet, template: tplArt },
        });
        expect(res.status).toBe('error');
        expect(res.errorMessage).toMatch(/template/i);
        expect(res.fetchUrls).toEqual([]);
      }
    });

    it('produces distinct FoldedStructure.id (runHash) between templated, untemplated, and spanChains=false runs', async () => {
      const seqSet = makeSequenceSetArtifact(HETERODIMER_SEQUENCE, {
        chainLengths: [HETERODIMER_CHAIN_A.length, HETERODIMER_CHAIN_B.length],
        chainIds: ['A', 'B'],
      });
      const tplArt = await makeStructureArtifactFromText(HETERODIMER_PDB);
      const untemplated = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'multimer' },
        inputs: { sequences: seqSet },
      });
      const spanned = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'multimer', templateSpanChains: true },
        inputs: { sequences: seqSet, template: tplArt },
      });
      const unspanned = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'multimer', templateSpanChains: false },
        inputs: { sequences: seqSet, template: tplArt },
      });
      expect(spanned.outputs!.folds.id).not.toBe(untemplated.outputs!.folds.id);
      expect(spanned.outputs!.folds.id).not.toBe(unspanned.outputs!.folds.id);
    });

    it('records structured template metadata in FoldedCandidate.provenance for templated runs', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB, {
        filename: '1ubq_ref.pdb',
      });
      const res = await runLocalFoldWorkerInNode({
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'monomer-2' },
        inputs: { sequences: seqSet, template: tplArt },
      });
      expect(res.status).toBe('result');
      const prov = res.outputs?.folds.folds[0]?.provenance;
      expect(prov).toBeDefined();
      expect(prov?.templateUsed).toBe(true);
      expect(prov?.templateFormat).toBe('pdb');
      expect(typeof prov?.templateDigest).toBe('string');
      expect(prov?.templateDigest?.length).toBeGreaterThan(0);
      expect(prov?.templateMatchedResidues).toBe(STANDARD_MONOMER_SEQUENCE.length);
      expect(prov?.templateTotalResidues).toBe(STANDARD_MONOMER_SEQUENCE.length);
      expect(prov?.templateSpanChains).toBe(true);
    });
  });

  describe('F10: Live Input Preview, Planner, Preflight & Share Integration', () => {
    it('surfaces template source and alignment coverage in compileNodeInputPreview', async () => {
      clearInputPreviewCache();
      const graph = makeTemplatedLocalFoldGraph({ supplyMode: 'port', model: 'monomer-2' });
      graph.nodes = graph.nodes.map((n) =>
        n.id === 'tpl'
          ? {
              ...n,
              label: 'Reference PDB',
              params: { ...n.params, filename: '1ubq_ref.pdb' },
            }
          : n,
      );
      const seqArt = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const tplArt = await makeStructureArtifactFromText(STANDARD_MONOMER_PDB, {
        filename: '1ubq_ref.pdb',
      });
      const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
        seq: { sequences: seqArt },
        tpl: { structure: tplArt },
      };
      const preview = compileNodeInputPreview(graph, 'fold', outputsByNode);
      expect(preview.availability).toBe('ready');
      const summaryObj = preview.payload.compiledSummary as Record<string, unknown>;
      const hasTemplateKey = Object.keys(summaryObj).some((k) =>
        k.toLowerCase().includes('template'),
      );
      const hasHeadlineTemplate = preview.payload.headline.toLowerCase().includes('template');
      expect(hasTemplateKey || hasHeadlineTemplate).toBe(true);
      const payloadJson = JSON.stringify({
        headline: preview.payload.headline,
        compiledSummary: preview.payload.compiledSummary,
      }).toLowerCase();
      expect(payloadJson).toMatch(/coverage|matched|aligned|100%|20\/20/);
    });

    it('surfaces template error issue in compileNodeInputPreview when template supplied to template-free model', async () => {
      clearInputPreviewCache();
      const graph = makeTemplatedLocalFoldGraph({ supplyMode: 'inline', model: 'monomer-4' });
      const seqArt = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const preview = compileNodeInputPreview(graph, 'fold', { seq: { sequences: seqArt } });
      expect(preview.issues.some((i) => i.severity === 'error')).toBe(true);
    });

    it('wires pdb-input.structure -> localfold-af2.template in compileWorkflowPlanToGraph when sequences is occupied', () => {
      const compiled = compileWorkflowPlanToGraph({
        kind: 'block_graph',
        mode: 'new',
        summary: 'Fold sequence with structural template',
        nodes: [
          { id: 'seq', adapter: 'sequence-input', label: 'Sequence', params: { sequence: STANDARD_MONOMER_SEQUENCE } },
          { id: 'tpl', adapter: 'pdb-input', label: 'Template', params: { mode: 'demo-bundle' } },
          { id: 'fold', adapter: 'localfold-af2', label: 'LocalFold', params: { model: 'monomer-2' } },
        ],
        edges: [
          ['seq', 'fold'],
          ['tpl', 'fold'],
        ],
      });
      const tplEdge = compiled.graph.edges.find(
        (e) => e.fromNode === 'tpl' && e.toNode === 'fold',
      );
      expect(tplEdge).toBeDefined();
      expect(tplEdge!.fromPort).toBe('structure');
      expect(tplEdge!.toPort).toBe('template');
    });

    it('computes preflight estimateRunRequirements cleanly for port and inline templated graphs', () => {
      const portGraph = makeTemplatedLocalFoldGraph({ supplyMode: 'port', model: 'monomer-2' });
      const inlineGraph = makeTemplatedLocalFoldGraph({ supplyMode: 'inline', model: 'boltz2' });
      const pre1 = estimateRunRequirements(portGraph);
      const pre2 = estimateRunRequirements(inlineGraph);
      expect(pre1.models.some((m) => m.key === 'localfold:monomer-2')).toBe(true);
      expect(pre2.models.some((m) => m.key === 'localfold:boltz2')).toBe(true);
    });

    it('strips oversized inline templatePdbText (>1500 chars) on YAML/hash share with SHA-256 digest', async () => {
      expect(LARGE_TEMPLATE_PDB.length).toBeGreaterThan(1500);
      const graph = makeTemplatedLocalFoldGraph({
        supplyMode: 'inline',
        model: 'monomer-2',
        templateText: LARGE_TEMPLATE_PDB,
      });
      const encoded = await encodeGraphToHash(graph);
      expect(encoded.strippedPdb).toBe(true);
      const decoded = await decodeHashToGraph(encoded.hash);
      expect(decoded).not.toBeNull();
      const foldDecoded = decoded!.nodes.find((n) => n.id === 'fold')!;
      expect(foldDecoded.params.missingTemplateUpload).toBe(true);
      expect(foldDecoded.params.expectedTemplateSha256).toBe(
        computeUploadDigest(LARGE_TEMPLATE_PDB),
      );
      const yaml = graphToYamlRecipe(graph);
      const parsed = parseRecipeToGraph(yaml);
      const foldYaml = parsed.nodes.find((n) => n.id === 'fold')!;
      expect(foldYaml.params.expectedTemplateSha256).toBe(
        computeUploadDigest(LARGE_TEMPLATE_PDB),
      );
    });

    it('restores stripped inline templatePdbText via localfoldAf2Plugin.share.restoreOmittedInlineAsset', () => {
      const restore = localfoldAf2Plugin.share.restoreOmittedInlineAsset;
      expect(restore).toBeTypeOf('function');
      const prevParams = {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-2',
        templatePdbText: LARGE_TEMPLATE_PDB,
        templateFilename: 'large.pdb',
      };
      const sha = computeUploadDigest(LARGE_TEMPLATE_PDB);
      const strippedParams = {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-2',
        templatePdbText: '',
        missingTemplateUpload: true,
        expectedTemplateSha256: sha,
        expectedTemplateFilename: 'large.pdb',
      };
      const restored = restore!(prevParams, strippedParams, '', 'fold');
      expect(restored).not.toBeNull();
      expect(restored!.templatePdbText).toBe(LARGE_TEMPLATE_PDB);
      expect(restored!.missingTemplateUpload).toBeUndefined();
    });
  });

  describe('F11: LocalFold Inspector UI / Widget & Mode-Switch Contract', () => {
    it('declares templatePdbText, templateFilename, and templateSpanChains in paramSchema', () => {
      const keys = new Set(ADAPTER_REGISTRY['localfold-af2'].paramSchema?.map((f) => f.key) ?? []);
      expect(keys.has('templatePdbText')).toBe(true);
      expect(keys.has('templateFilename')).toBe(true);
      expect(keys.has('templateSpanChains')).toBe(true);
    });

    it('preserves template parameters across template-capable model switches', () => {
      const prev = {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-2',
        templatePdbText: STANDARD_MONOMER_PDB,
        templateFilename: 'tpl.pdb',
        templateSpanChains: false,
      };
      const toMultimer = sanitizeParamsOnModeSwitch('localfold-af2', prev, 'multimer');
      expect(toMultimer.templatePdbText).toBe(STANDARD_MONOMER_PDB);
      expect(toMultimer.templateSpanChains).toBe(false);
      const toAf3 = sanitizeParamsOnModeSwitch('localfold-af2', toMultimer, 'boltz2');
      expect(toAf3.templatePdbText).toBe(STANDARD_MONOMER_PDB);
    });

    it('retains or warns explicitly rather than silently discarding template state on mode switch to template-free', () => {
      const prev = {
        ...defaultParamsForAdapter('localfold-af2'),
        model: 'monomer-2',
        templatePdbText: STANDARD_MONOMER_PDB,
      };
      const toFree = sanitizeParamsOnModeSwitch('localfold-af2', prev, 'monomer-4');
      // Either templatePdbText stays attached (so validation surfaces the actionable warning + quick-switch)
      // or validation explicitly rejects it when still present
      expect(toFree.model).toBe('monomer-4');
      const graph = {
        id: 'mode-switch-check',
        name: 'Mode Switch',
        nodes: [makeSequenceNode('seq', STANDARD_MONOMER_SEQUENCE), makeLocalFoldNode('fold', toFree)],
        edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
      };
      if (String(toFree.templatePdbText ?? '').trim()) {
        const errs = validateBioCapabilities(graph).filter((i) => i.severity === 'error');
        expect(errs.length).toBeGreaterThan(0);
      } else {
        throw new Error('Expected templatePdbText to remain inspectable for explicit template-free warning rather than silent drop');
      }
    });

    it('exposes template controls and template-free warning in LocalFoldWidget.svelte', () => {
      const widgetPath = resolve(
        __dirname,
        '../../src/lib/ui/inspector/LocalFoldWidget.svelte',
      );
      const src = readFileSync(widgetPath, 'utf8');
      expect(src).toContain('templatePdbText');
      expect(src).toContain('templateSpanChains');
      expect(src).toMatch(
        /isLocalFoldTemplateCapable|localFoldTemplateFamily|templateCapable|template[- ]free|does not (?:support|accept|use)[^.\n]*template|unsupported[^.\n]*template|template[^.\n]*(?:unsupported|not support|ignored|capable)|switch[^.\n]*(?:monomer-2|template-capable)/i,
      );
    });

    it('supports quick-switch recovery from template-free model inside inspector parameter workflow', () => {
      const initial = makeTemplatedLocalFoldGraph({ supplyMode: 'inline', model: 'ef2-fast-600m' });
      const issue = validateBioCapabilities(initial).find((i) => i.severity === 'error' && i.fix);
      expect(issue?.fix).toBeDefined();
      const switched = sanitizeParamsOnModeSwitch(
        'localfold-af2',
        initial.nodes[1].params,
        String(issue!.fix!.paramsPatch.model),
      );
      expect(switched.templatePdbText).toBe(STANDARD_MONOMER_PDB);
    });
  });

  describe('F12: Backward Compatibility & Untemplated Parity', () => {
    it('keeps defaultParamsForAdapter("localfold-af2") identical to baseline 6 keys', () => {
      expect(defaultParamsForAdapter('localfold-af2')).toEqual({
        model: 'monomer',
        recycles: 1,
        diffusionSteps: 25,
        capMode: 'follow-upstream',
        maxSequences: 2,
        seed: 0,
      });
    });

    it('keeps all existing WORKFLOW_PRESETS valid in graph, bio-capabilities, and share roundtrips', async () => {
      const lfPresets = WORKFLOW_PRESETS.filter((p) =>
        p.graph.nodes.some((n) => n.adapterId === 'localfold-af2'),
      );
      expect(lfPresets.length).toBeGreaterThan(0);
      for (const preset of lfPresets) {
        expect(validateGraph(preset.graph)).toEqual([]);
        const errs = validateBioCapabilities(preset.graph).filter((i) => i.severity === 'error');
        expect(errs).toEqual([]);
        const encoded = await encodeGraphToHash(preset.graph);
        expect(encoded.mode).toBe('preset-diff');
        const decoded = await decodeHashToGraph(encoded.hash);
        expect(decoded?.nodes.length).toBe(preset.graph.nodes.length);
        const yaml = graphToYamlRecipe(preset.graph);
        const reparsed = parseRecipeToGraph(yaml);
        expect(reparsed.nodes.length).toBe(preset.graph.nodes.length);
      }
    });

    it('runs untemplated localfold.worker.ts cleanly across AF2 monomer, multimer, AF3, and ESMFold2', async () => {
      const seqSet = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      for (const model of ['monomer', 'multimer', 'af3', 'ef2-fast-300m']) {
        const res = await runLocalFoldWorkerInNode({
          params: { ...defaultParamsForAdapter('localfold-af2'), model },
          inputs: { sequences: seqSet },
        });
        expect(res.errorMessage).toBeUndefined();
        expect(res.status).toBe('result');
        expect(res.outputs?.folds.folds).toHaveLength(1);
        const af2Call = res.captures.find(
          (c) => c.family === 'af2-monomer' || c.family === 'af2-multimer',
        );
        if (af2Call) {
          expect(af2Call.predictOpts?.template).toBeUndefined();
        }
        const af3Call = res.captures.find((c) => c.family === 'af3');
        if (af3Call) {
          const slots = af3Call.af3Options?.templateSlots as Array<unknown> | undefined;
          expect(!slots || slots.every((s) => !s)).toBe(true);
        }
      }
    });

    it('keeps untemplated compileNodeInputPreview and estimateRunRequirements regression-free', () => {
      clearInputPreviewCache();
      const graph = makeUntemplatedLocalFoldGraph({ model: 'monomer' });
      const seqArt = makeSequenceSetArtifact(STANDARD_MONOMER_SEQUENCE);
      const preview = compileNodeInputPreview(graph, 'fold', { seq: { sequences: seqArt } });
      expect(preview.availability).toBe('ready');
      expect(preview.issues.filter((i) => i.severity === 'error')).toEqual([]);
      const preflight = estimateRunRequirements(graph);
      expect(preflight.models.some((m) => m.key === 'localfold:monomer')).toBe(true);
    });

    it('runs untemplated AF3 CPU templateEmbedding oracle cleanly with finite non-NaN output', async () => {
      const oracles = await loadVendorTemplateOracles();
      const { batch } = oracles.af3BatchFromA3m(STANDARD_MONOMER_SEQUENCE, null);
      const out = await runAf3TemplateCpuOracle({
        modelKey: 'af3',
        tokens: batch.tokens,
        asymId: batch.asymId,
        slots: [],
      });
      const stats = diffFloat32Arrays(out.output, new Float32Array(out.output.length));
      expect(stats.hasNaN).toBe(false);
      expect(stats.nonZeroA).toBeGreaterThan(0);
    });
  });
});
