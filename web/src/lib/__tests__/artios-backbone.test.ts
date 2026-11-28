// artios-backbone: ArtiosBackbone all-atom EDM diffusion (Ligo-Biosciences/artios, Apache-2.0;
// WebGPU/CPU port ramithuh/artios-webgpu). Covers the plugin manifest & registration contract,
// the bundled CCD ideal-conformer table, PDB atom parsing, motif/ligand token-layout planning,
// AF3-style feature assembly against the vendored featurizer, PDB serialization, StructureArtifact
// assembly, and an end-to-end tiny-config CPU EDM sampling run (Karras schedule + Kabsch-aligned
// self-conditioning) through the vendored `/vendor/artios-backbone/` modules.

import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getDefaultParamsFromSchema, validateParamsWithSchema } from '$lib/engine/param-schema';
import { getModelCardForAdapter } from '$lib/engine/model-cards';
import type { AdapterManifest, StructureArtifact } from '$lib/engine/types';
import { createDefaultPluginRegistry, NATIVE_PLUGINS } from '$lib/plugins';
import { artiosBackbonePlugin, estimateArtiosBackboneVramMB } from '$lib/plugins/artios-backbone';
import {
  ARTIOS_BACKBONE_MANIFEST,
  ARTIOS_BACKBONE_WEIGHTS_MB,
} from '$lib/plugins/artios-backbone/manifest';
import {
  buildStandardComponents,
  componentFromRows,
  molTypeFromCompType,
  SCAFFOLD_RESIDUE,
  STANDARD_CCD_ROWS,
} from '$lib/plugins/artios-backbone/ccd-standard';
import {
  alignPrevPredBatch,
  artiosBackboneOptionsFromParams,
  artiosGpuBatchLimit,
  buildArtiosBackboneArtifact,
  buildArtiosFeatures,
  buildMotifPlan,
  buildUnconditionalPlan,
  centreAugmentBatch,
  componentFromInputLigand,
  coordsToArtiosPdb,
  groupResidues,
  normalizeArtiosBackbonePrecision,
  parseMotifContig,
  parsePdbAtomRecords,
  rcsbCcdUrl,
  runArtiosEdmSampler,
  solveMinCostAssignment,
  stitchUnindexedMotifCoords,
  wrapArtiosBackboneOutputs,
  type ArtiosSharedModule,
  type ArtiosTokenPlan,
} from '$lib/plugins/artios-backbone/jobs';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { formatNodeParamSummary } from '$lib/ui/node-display-helpers';

/* eslint-disable @typescript-eslint/no-explicit-any */
const VENDOR = resolve(process.cwd(), 'static/vendor/artios-backbone');
const load = async (rel: string): Promise<any> =>
  import(/* @vite-ignore */ pathToFileURL(join(VENDOR, rel)).href);
const shared = (await load('shared/index.js')) as ArtiosSharedModule & Record<string, any>;
const cpu = await load('cpu/index.js');
const gpu = await load('webgpu/index.js');

/** Registry that contains the plugin both before and after the coordinator applies the wiring. */
function registryWithPlugin() {
  const wiredIn = NATIVE_PLUGINS.some((p) => p.id === 'artios-backbone');
  return createDefaultPluginRegistry(wiredIn ? undefined : { extraPlugins: [artiosBackbonePlugin] });
}
const WIRED = Boolean(ADAPTER_REGISTRY['artios-backbone']);

// ===========================================================================
// fixtures
// ===========================================================================

const FIXTURE_SEQ = ['ALA', 'GLY', 'SER', 'ALA', 'VAL', 'ALA', 'LEU', 'ALA', 'GLY', 'ALA', 'THR', 'ALA'];

/** Heavy side-chain atoms past CB, using the CCD atom names of each residue. */
const SIDE_CHAINS: Record<string, Array<[string, string]>> = {
  SER: [['OG', 'O']],
  VAL: [['CG1', 'C'], ['CG2', 'C']],
  LEU: [['CG', 'C'], ['CD1', 'C'], ['CD2', 'C']],
  THR: [['OG1', 'O'], ['CG2', 'C']],
};

/** 12-residue ideal α-helix on chain A, a ZN HETATM on chain B, one water and one altLoc pair. */
function makeFixturePdb(): string {
  const lines: string[] = [];
  let serial = 1;
  const push = (
    record: 'ATOM  ' | 'HETATM',
    name: string,
    alt: string,
    resName: string,
    chain: string,
    resId: number,
    [x, y, z]: [number, number, number],
    element: string,
  ) => {
    lines.push(
      `${record}${String(serial++).padStart(5, ' ')} ${name.length >= 4 ? name : ` ${name.padEnd(3, ' ')}`}` +
        `${alt}${resName.padStart(3, ' ')} ${chain}${String(resId).padStart(4, ' ')}    ` +
        `${x.toFixed(3).padStart(8, ' ')}${y.toFixed(3).padStart(8, ' ')}${z.toFixed(3).padStart(8, ' ')}` +
        `  1.00 20.00          ${element.padStart(2, ' ')}`,
    );
  };
  const radius = 2.3;
  const rise = 1.5;
  const twist = (100 * Math.PI) / 180;
  for (let i = 0; i < FIXTURE_SEQ.length; i++) {
    const resName = FIXTURE_SEQ[i];
    const ang = i * twist;
    const ca: [number, number, number] = [radius * Math.cos(ang), radius * Math.sin(ang), i * rise];
    const off = (dx: number, dy: number, dz: number): [number, number, number] => [
      ca[0] + dx,
      ca[1] + dy,
      ca[2] + dz,
    ];
    push('ATOM  ', 'N', ' ', resName, 'A', i + 1, off(-1.2, 0.3, -0.5), 'N');
    push('ATOM  ', 'CA', ' ', resName, 'A', i + 1, ca, 'C');
    push('ATOM  ', 'C', ' ', resName, 'A', i + 1, off(1.1, 0.5, 0.2), 'C');
    push('ATOM  ', 'O', ' ', resName, 'A', i + 1, off(1.7, 1.5, 0.3), 'O');
    if (resName !== 'GLY') {
      // Residue 5 carries an A/B altLoc pair on CB so the parser's first-altLoc rule is exercised.
      const altPair = i === 4;
      push('ATOM  ', 'CB', altPair ? 'A' : ' ', resName, 'A', i + 1, off(0.6, -1.2, 0.9), 'C');
      if (altPair) push('ATOM  ', 'CB', 'B', resName, 'A', i + 1, off(0.9, -1.4, 1.2), 'C');
      // Real side-chain atoms (CCD names) so motif all-atom preservation is observable.
      SIDE_CHAINS[resName]?.forEach(([name, element], k) => {
        push('ATOM  ', name, ' ', resName, 'A', i + 1, off(1.0 + 0.4 * k, -2.0 - 0.3 * k, 1.4 + 0.5 * k), element);
      });
    }
    // One explicit hydrogen that the parser must drop.
    push('ATOM  ', 'HA', ' ', resName, 'A', i + 1, off(0.1, 0.9, -0.4), 'H');
  }
  push('HETATM', 'ZN', ' ', 'ZN', 'B', 101, [0.4, 0.2, 8.5], 'ZN');
  push('HETATM', 'O', ' ', 'HOH', 'C', 201, [8.0, 8.0, 8.0], 'O');
  lines.push('END');
  return `${lines.join('\n')}\n`;
}
const FIXTURE_PDB = makeFixturePdb();

/** `StructureArtifact` wrapper around the fixture, as a pdb-input block would emit it. */
function fixtureStructure(): StructureArtifact {
  const parsed = parseStructureText(FIXTURE_PDB);
  return {
    kind: 'StructureArtifact',
    id: 'fixture-helix',
    label: 'Fixture helix + ZN',
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: FIXTURE_PDB,
    chainIds: parsed.chainIds,
    diagnostics: parsed.diagnostics,
    geometry: { length: parsed.nResidues, rg: 0, meanCaDist: 0, chainBreaks: 0, clashes: 0, helixPct: 0, sheetPct: 0, coilPct: 100 },
  } as unknown as StructureArtifact;
}

const TINY_CONFIG = {
  cToken: 8,
  cAtom: 6,
  cS: 4,
  cZ: 2,
  numBlocks: 2,
  numHeads: 2,
  cTimestepEmbedding: 5,
  sigmaData: 16.0,
  transitionExpansion: 2,
};

/**
 * Deterministic synthetic `DiffusionModule` state dict at an arbitrary (tiny) architecture,
 * ported from `artios-webgpu`'s `test/backbone-cpu.test.mjs :: syntheticWeights` so the port's
 * structural invariants can be exercised without the 131 MiB checkpoint.
 */
function syntheticWeights(config: typeof TINY_CONFIG, rng: { gaussian(): number }) {
  const { cToken, cAtom, cS, cZ, numBlocks, numHeads, cTimestepEmbedding, transitionExpansion: n } = config;
  const out: Record<string, { shape: number[]; data: Float32Array }> = {};
  const put = (name: string, shape: number[], scale = 0.1) => {
    const size = shape.reduce((a, b) => a * b, 1);
    out[name] = { shape, data: Float32Array.from({ length: size }, () => rng.gaussian() * scale) };
  };
  const fill = (name: string, dim: number, value: number) => {
    out[name] = { shape: [dim], data: new Float32Array(dim).fill(value) };
  };
  const layernorm = (prefix: string, dim: number) => {
    fill(`${prefix}.weight`, dim, 1);
    fill(`${prefix}.bias`, dim, 0);
  };
  const transition = (prefix: string, dim: number) => {
    layernorm(`${prefix}.layernorm`, dim);
    put(`${prefix}.swiglu.linear.weight`, [2 * n * dim, dim]);
    put(`${prefix}.linear_no_bias.weight`, [dim, n * dim]);
  };

  const atomCondDim = 1 + 1 + 128 + 256;
  const atomReadDim = 1 + 128 + 256;

  put('linear_no_bias_r_embed.weight', [cToken, 3]);
  put('linear_no_bias_prev_pred_embed.weight', [cToken, 3]);
  put('self_cond_merge.weight', [cToken, 2 * cToken]);

  const c = 'diffusion_conditioning';
  put(`${c}.embed_atom_ref_pos.weight`, [cS, 3]);
  put(`${c}.embed_atom_pair_offsets.weight`, [cZ, 3]);
  put(`${c}.embed_atom_pair_invd.weight`, [cZ, 1]);
  put(`${c}.embed_atom_pair_offsets_valid.weight`, [cZ, 1]);
  put(`${c}.atom_condition_embed.weight`, [cS, atomCondDim], 0.02);
  put(`${c}.embed_motif_restype.weight`, [cS, 32]);
  put(`${c}.embed_motif_coord.weight`, [cS, 3]);
  put(`${c}.embed_motif_pair_dist.weight`, [cZ, 56]);
  put(`${c}.embed_motif_depth.weight`, [cS, 42]);
  put(`${c}.embed_rasa.weight`, [cS, 22]);
  put(`${c}.relpos.embed_relpos_features.weight`, [cZ, 139]);
  layernorm(`${c}.layernorm_z`, 2 * cZ);
  put(`${c}.linear_no_bias_z.weight`, [cZ, 2 * cZ]);
  out[`${c}.fourier_embedding.w`] = {
    shape: [cTimestepEmbedding],
    data: Float32Array.from({ length: cTimestepEmbedding }, () => rng.gaussian()),
  };
  out[`${c}.fourier_embedding.b`] = {
    shape: [cTimestepEmbedding],
    data: Float32Array.from({ length: cTimestepEmbedding }, () => rng.gaussian()),
  };
  layernorm(`${c}.layernorm_t`, cTimestepEmbedding);
  put(`${c}.linear_t.weight`, [cS, cTimestepEmbedding]);
  put(`${c}.entity_type_embed.weight`, [cS, 4]);
  put(`${c}.motif_entity_type_embed.weight`, [cS, 2]);
  transition(`${c}.transition_z`, cZ);
  transition(`${c}.transition_s1`, cS);
  transition(`${c}.transition_s2`, cS);

  layernorm('layernorm_s', cS);
  put('linear_no_bias_s.weight', [cToken, cS]);

  const d = 'diffusion_transformer';
  const fused = 2 * numBlocks * cToken;
  layernorm(`${d}.condition_layernorm`, cS);
  put(`${d}.scale_proj.0.weight`, [fused, cS]);
  fill(`${d}.scale_proj.0.bias`, fused, 1);
  put(`${d}.shift_proj.weight`, [fused, cS]);
  put(`${d}.gate_proj.0.weight`, [fused, cS]);
  fill(`${d}.gate_proj.0.bias`, fused, -2);
  layernorm(`${d}.z_layernorm`, cZ);
  put(`${d}.z_proj.weight`, [numHeads * numBlocks, cZ]);
  for (let b = 0; b < numBlocks; b++) {
    const p = `${d}.blocks.${b}`;
    layernorm(`${p}.attn_layernorm`, cToken);
    put(`${p}.attention.linear_q.weight`, [cToken, cToken]);
    fill(`${p}.attention.linear_q.bias`, cToken, 0);
    put(`${p}.attention.linear_k.weight`, [cToken, cToken]);
    put(`${p}.attention.linear_v.weight`, [cToken, cToken]);
    put(`${p}.attention.gate_linear.0.weight`, [cToken, cToken]);
    fill(`${p}.attention.gate_linear.0.bias`, cToken, 0);
    put(`${p}.attention.linear_output.weight`, [cToken, cToken]);
    layernorm(`${p}.mlp_layernorm`, cToken);
    put(`${p}.mlp.0.linear.weight`, [4 * cToken, cToken]);
    put(`${p}.mlp.1.weight`, [cToken, 2 * cToken]);
  }

  layernorm('layernorm_a', cToken);
  put('linear_no_bias_a.weight', [cAtom, cToken]);
  put('linear_no_bias_f.weight', [cAtom, atomReadDim], 0.02);
  transition('readout_transition', cAtom);
  layernorm('readout_layernorm', cAtom);
  put('readout_linear.weight', [3, cAtom]);
  return out;
}

const DEFAULT_OPTIONS = artiosBackboneOptionsFromParams({});

// ===========================================================================
// 1. vendored runtime
// ===========================================================================

describe('artios-backbone vendored runtime (static/vendor/artios-backbone)', () => {
  it('ships the Apache-2.0 LICENSE, the Ligo Biosciences NOTICE and the shared/cpu/webgpu entrypoints', () => {
    expect(readFileSync(join(VENDOR, 'LICENSE'), 'utf8')).toContain('Apache License');
    const notice = readFileSync(join(VENDOR, 'NOTICE'), 'utf8');
    expect(notice).toContain('Ligo Biosciences');
    expect(notice).toContain('Apache License, Version 2.0');

    for (const name of [
      'loadWeightBundle',
      'parseSafetensors',
      'parseChemComp',
      'parseCif',
      'buildReferenceFeatures',
      'buildTokenization',
      'buildMotifFeatures',
      'buildMotifDepthFeature',
      'buildRasaFeature',
      'computeMotifDepthNorm',
      'edmNoiseSchedule',
      'edmPreconditioning',
      'kabschAlign',
      'centreRandomAugmentation',
      'createRng',
      'encodeElementOneHot',
      'encodeRestypeOneHot',
      'encodeAtomNameChars',
    ]) {
      expect(typeof shared[name], `shared.${name}`).toBe('function');
    }
    expect(typeof cpu.createArtiosBackboneCPU).toBe('function');
    expect(typeof cpu.prepareFeatures).toBe('function');
    expect(typeof gpu.createArtiosBackboneGpu).toBe('function');
    expect(typeof gpu.edmNoiseSchedule).toBe('function');
  });

  it('keeps the manifest EDM defaults in lockstep with the upstream sampler constants', () => {
    expect(shared.SIGMA_DATA).toBe(16.0);
    expect(shared.EDM_SCHEDULE).toEqual({ sMax: 160.0, sMin: 4e-4, rho: 7.0 });
    expect(shared.EDM_SAMPLER).toEqual({
      gamma0: 0.8,
      gammaMin: 1.0,
      noiseScaleLambda: 1.003,
      stepScaleEta: 1.5,
    });
    const byKey = new Map(
      (ARTIOS_BACKBONE_MANIFEST.paramSchema ?? []).map((f) => [f.key, f.defaultValue]),
    );
    expect(byKey.get('gamma0')).toBe(shared.EDM_SAMPLER.gamma0);
    expect(byKey.get('noiseScaleLambda')).toBe(shared.EDM_SAMPLER.noiseScaleLambda);
    expect(byKey.get('stepScaleEta')).toBe(shared.EDM_SAMPLER.stepScaleEta);
    // The architecture the CPU denoiser defaults to is the released 136.7M checkpoint.
    expect(cpu.DEFAULT_CONFIG).toMatchObject({
      cToken: 768,
      cAtom: 256,
      cS: 384,
      cZ: 128,
      numBlocks: 16,
      numHeads: 16,
      cTimestepEmbedding: 256,
      sigmaData: 16.0,
    });
  });

  it('produces the Karras rho=7 schedule the worker integrates (descending, final sigma exactly 0)', () => {
    const schedule = shared.edmNoiseSchedule(20) as Float64Array;
    expect(schedule).toHaveLength(21);
    expect(schedule[0]).toBeCloseTo(16.0 * 160.0, 6);
    expect(schedule[20]).toBe(0);
    for (let i = 1; i < schedule.length; i++) expect(schedule[i]).toBeLessThan(schedule[i - 1]);
    // The WebGPU helper and the shared helper agree to float precision.
    const viaGpu = gpu.edmNoiseSchedule(20) as Float32Array;
    for (let i = 0; i < schedule.length; i++) expect(viaGpu[i]).toBeCloseTo(schedule[i], 3);
  });
});

// ===========================================================================
// 2. bundled CCD conformers
// ===========================================================================

describe('artios-backbone bundled CCD ideal conformers', () => {
  it('covers the 20 standard amino acids plus UNK with wwPDB heavy-atom ideal coordinates', () => {
    const codes = Object.keys(STANDARD_CCD_ROWS);
    expect(codes).toHaveLength(21);
    expect(codes).toContain('UNK');
    expect(codes).toContain(SCAFFOLD_RESIDUE);
    for (const code of codes) {
      expect(shared.RESIDUE_VOCAB[code], `${code} in RESIDUE_VOCAB`).toBeTypeOf('number');
      expect(molTypeFromCompType(STANDARD_CCD_ROWS[code].type)).toBe('protein');
      const names = STANDARD_CCD_ROWS[code].atoms.map((a) => a[0]);
      expect(new Set(names).size, `${code} atom names unique`).toBe(names.length);
      expect(names.slice(0, 4)).toEqual(['N', 'CA', 'C', 'O']);
      // Leaving atoms (OXT) and hydrogens are dropped, matching parseChemComp's defaults.
      expect(names).not.toContain('OXT');
      for (const [, element, , x, y, z] of STANDARD_CCD_ROWS[code].atoms) {
        expect(shared.ELEMENT_SYMBOLS).toContain(element);
        expect(element).not.toBe('H');
        for (const v of [x, y, z]) expect(Number.isFinite(v)).toBe(true);
      }
    }
    // Spot-check against the wwPDB CCD entry for ALANINE.
    expect(STANDARD_CCD_ROWS.ALA.atoms).toEqual([
      ['N', 'N', 0, -0.966, 0.493, 1.5],
      ['CA', 'C', 0, 0.257, 0.418, 0.692],
      ['C', 'C', 0, -0.094, 0.017, -0.716],
      ['O', 'O', 0, -1.056, -0.682, -0.923],
      ['CB', 'C', 0, 1.204, -0.62, 1.296],
    ]);
    expect(STANDARD_CCD_ROWS.GLY.atoms.map((a) => a[0])).toEqual(['N', 'CA', 'C', 'O']);
    expect(STANDARD_CCD_ROWS.ARG.atoms.find((a) => a[0] === 'NH2')?.[2]).toBe(1);
  });

  it('expands rows into the vendored parseChemComp component shape', () => {
    const components = buildStandardComponents();
    expect(components.size).toBe(21);
    const ala = componentFromRows('ALA', STANDARD_CCD_ROWS.ALA);
    expect(ala).toMatchObject({ id: 'ALA', molType: 'protein', coordSource: 'ideal' });
    expect(ala.atomIndex.get('CB')).toBe(4);
    expect(ala.atoms[4]).toMatchObject({ atomId: 'CB', element: 'C', hasCoord: true, leaving: false });
    expect(ala.atoms[4].coord).toEqual([1.204, -0.62, 1.296]);
    expect(molTypeFromCompType('NON-POLYMER')).toBe('ligand');
    expect(molTypeFromCompType('PEPTIDE-LIKE')).toBe('ligand');
    expect(molTypeFromCompType('DNA LINKING')).toBe('dna');
    expect(molTypeFromCompType('RNA LINKING')).toBe('rna');
  });

  it('parses a runtime-fetched CCD CIF through the vendored parser into the same shape', () => {
    const cif = `data_LIG
_chem_comp.id LIG
_chem_comp.type NON-POLYMER
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
LIG C1 C1 C 0 N 1.0 2.0 3.0 0.100 0.200 0.300
LIG O1 O1 O -1 N 2.0 3.0 4.0 1.100 1.200 1.300
LIG H1 H1 H 0 N 3.0 4.0 5.0 2.100 2.200 2.300
`;
    const comp = shared.parseChemComp(cif);
    expect(comp.molType).toBe('ligand');
    expect(comp.atoms.map((a: any) => a.atomId)).toEqual(['C1', 'O1']); // hydrogen dropped
    expect(comp.atoms[0].coord).toEqual([0.1, 0.2, 0.3]);
    expect(comp.atoms[1].charge).toBe(-1);
    expect(comp.coordSource).toBe('ideal');
    expect(rcsbCcdUrl('atp')).toBe('https://files.rcsb.org/ligands/download/ATP.cif');
    expect(() => rcsbCcdUrl('TOOLONG')).toThrow(/Invalid CCD component code/);
  });
});

// ===========================================================================
// 3. plugin manifest, ports, params & registration
// ===========================================================================

describe('artios-backbone plugin manifest & registration', () => {
  it('declares the manifest, ports and runtime contract', () => {
    const m = ARTIOS_BACKBONE_MANIFEST as AdapterManifest;
    expect(m.id).toBe('artios-backbone');
    expect(m.stage).toBe('generate');
    expect(m.runtime).toBe('wgsl-custom');
    expect(m.shortName).toBe('ArtiosBackbone');
    expect(m.nonCommercial).toBeUndefined();
    expect(m.licenseNote).toContain('Apache-2.0');
    expect(m.licenseNote).toContain('Ligo-Biosciences/artios');
    expect(m.description).toContain('136.7M');
    expect(Object.keys(m.inputs)).toEqual(['structure']);
    expect(m.inputs.structure).toEqual({ type: 'StructureArtifact', required: false });
    expect(Object.keys(m.outputs)).toEqual(['structure']);
    expect(m.outputs.structure).toEqual({ type: 'StructureArtifact' });
    expect(m.approxWeightsMB).toBe(ARTIOS_BACKBONE_WEIGHTS_MB.int8);
    expect(m.precisionWeightsMB).toEqual({ int8: 131.3, fp16: 260.7, fp32: 521.4 });
    expect(artiosBackbonePlugin.manifest.id).toBe('artios-backbone');
  });

  it('exposes every documented parameter with sensible defaults and ranges', () => {
    const schema = ARTIOS_BACKBONE_MANIFEST.paramSchema ?? [];
    expect(schema.map((f) => f.key)).toEqual([
      'precision',
      'length',
      'numSteps',
      'numDesigns',
      'motifContig',
      'unindexedMotif',
      'includeLigand',
      'ligandRasa',
      'motifDepth',
      'stepScaleEta',
      'noiseScaleLambda',
      'gamma0',
      'selfConditioning',
      'seed',
    ]);
    for (const field of schema) {
      expect(field.label.length, `${field.key}.label`).toBeGreaterThan(0);
      expect(['primary', 'advanced']).toContain(field.tier);
      expect(field.description, `${field.key}.description`).toBeTruthy();
    }
    const byKey = new Map(schema.map((f) => [f.key, f]));
    expect(byKey.get('precision')!.options?.map((o) => o.value)).toEqual(['int8', 'fp16', 'fp32']);
    expect(byKey.get('length')).toMatchObject({ kind: 'integer', min: 30, max: 512, defaultValue: 100 });
    expect(byKey.get('numDesigns')).toMatchObject({ kind: 'integer', min: 1, max: 16, defaultValue: 1 });
    expect(byKey.get('numSteps')).toMatchObject({ kind: 'integer', min: 20, max: 400, defaultValue: 200 });
    expect(byKey.get('numSteps')!.warnBelow?.fixValue).toBe(200);
    expect(byKey.get('motifContig')).toMatchObject({ kind: 'text', defaultValue: '' });
    expect(byKey.get('unindexedMotif')).toMatchObject({ kind: 'boolean', defaultValue: true });
    expect(byKey.get('includeLigand')).toMatchObject({ kind: 'boolean', defaultValue: true });
    expect(byKey.get('ligandRasa')).toMatchObject({ kind: 'number', min: -1, max: 1, defaultValue: -1 });
    expect(byKey.get('motifDepth')).toMatchObject({ kind: 'number', min: -1, max: 2, defaultValue: -1 });
    expect(byKey.get('selfConditioning')).toMatchObject({ kind: 'boolean', defaultValue: true });
    expect(byKey.get('seed')).toMatchObject({ kind: 'integer', defaultValue: 42 });

    expect(getDefaultParamsFromSchema(ARTIOS_BACKBONE_MANIFEST as AdapterManifest)).toEqual({
      precision: 'int8',
      length: 100,
      numSteps: 200,
      numDesigns: 1,
      motifContig: '',
      unindexedMotif: true,
      includeLigand: true,
      ligandRasa: -1,
      motifDepth: -1,
      stepScaleEta: 1.5,
      noiseScaleLambda: 1.003,
      gamma0: 0.8,
      selfConditioning: true,
      seed: 42,
    });
  });

  it('validates params through the declarative schema and the crossValidate hook', () => {
    const manifest = ARTIOS_BACKBONE_MANIFEST as AdapterManifest;
    const defaults = getDefaultParamsFromSchema(manifest);
    expect(validateParamsWithSchema(manifest, defaults)).toEqual([]);
    expect(validateParamsWithSchema(manifest, { ...defaults, length: 1024 }).join(' ')).toMatch(/length/i);
    expect(validateParamsWithSchema(manifest, { ...defaults, numSteps: 5 }).join(' ')).toMatch(/steps/i);
    expect(validateParamsWithSchema(manifest, { ...defaults, precision: 'int4' }).length).toBeGreaterThan(0);

    const cross = manifest.crossValidate!;
    expect(cross({ ...defaults }, { connectedInputs: new Set() })).toEqual([]);
    expect(cross({ ...defaults, motifContig: 'A10-25' }, { connectedInputs: new Set(['structure']) })).toEqual([]);
    expect(
      cross({ ...defaults, motifContig: 'A10-25' }, { connectedInputs: new Set() }).join(' '),
    ).toMatch(/requires a StructureArtifact/);
    expect(cross({ ...defaults, motifContig: 'nonsense' }, { connectedInputs: new Set(['structure']) }).join(' ')).toMatch(
      /Invalid ArtiosBackbone motifContig/,
    );
    expect(cross({ ...defaults, motifContig: 'A30-10' }, { connectedInputs: new Set(['structure']) }).length).toBe(1);
    expect(cross({ ...defaults, ligandRasa: -0.5 }, {}).join(' ')).toMatch(/ligandRasa/);
    expect(cross({ ...defaults, motifDepth: -0.2 }, {}).join(' ')).toMatch(/motifDepth/);
  });

  it('registers in the PluginRegistry as a generate-stage generator with short code "ab"', () => {
    const reg = registryWithPlugin();
    expect(reg.getPlugin('artios-backbone')).toBe(artiosBackbonePlugin);
    expect(reg.resolveShortCode('ab')).toBe('artios-backbone');
    expect(reg.getShortCode('artios-backbone')).toBe('ab');
    expect(reg.findPluginsByStage('generate').map((p) => p.id)).toContain('artios-backbone');
    expect(reg.findPluginsByPlannerRole('generator').map((p) => p.id)).toContain('artios-backbone');
    expect(artiosBackbonePlugin.share.shortCode).toBe('ab');
    expect(artiosBackbonePlugin.planner?.plannerRole).toBe('generator');
    expect(artiosBackbonePlugin.planner?.capabilities).toContain('motif-scaffolding');
    expect(artiosBackbonePlugin.planner?.capabilities).toContain('ligand-conditioned-generation');
    expect(artiosBackbonePlugin.catalog.categoryId).toBe('generate');
    expect(artiosBackbonePlugin.catalog.searchKeywords).toContain('artios');
    expect(artiosBackbonePlugin.catalog.searchKeywords).toContain('ligand');
    expect(artiosBackbonePlugin.cachePolicy.usesSharedOpfsStore).toBe(true);
    expect(artiosBackbonePlugin.execution.runtime).toBe('wgsl-custom');
    expect(artiosBackbonePlugin.execution.vramLeaseMode).toBe('exclusive');
    expect(artiosBackbonePlugin.execution.supportsCooperativeAbort).toBe(true);
    expect(artiosBackbonePlugin.execution.fallback).toEqual({
      kind: 'cpu-js',
      moduleUrl: '/vendor/artios-backbone/cpu/index.js',
      degradedWarning: expect.stringContaining('CPU JS denoiser'),
    });
    expect(typeof artiosBackbonePlugin.execution.createWorker).toBe('function');
    // No other registered plugin may own "ab".
    const owners = reg.listPlugins().filter((p) => p.share.shortCode === 'ab');
    expect(owners.map((p) => p.id)).toEqual(['artios-backbone']);
  });

  it('summarizes canvas state and estimates download / VRAM per precision', () => {
    const summary = (params: Record<string, unknown>) =>
      artiosBackbonePlugin.display.formatCanvasSummary(params);
    expect(summary({})).toBe('100 aa · 200 steps · INT8');
    expect(summary({ length: 256, numSteps: 150, precision: 'fp16', numDesigns: 4 })).toBe(
      '256 aa · 150 steps · FP16 · ×4',
    );
    expect(summary({ motifContig: 'A10-25', length: 120, numSteps: 200 })).toBe(
      'Motif (A10-25) · 120 aa · 200 steps',
    );
    expect(summary({ motifContig: 'A10-25', includeLigand: false, numDesigns: 2 })).toBe(
      'Motif (A10-25) · 100 aa · 200 steps · no ligand · ×2',
    );

    const est = (params: Record<string, unknown>) =>
      artiosBackbonePlugin.preflight({
        params,
        upstreamProfiles: {},
        connectedInputs: new Set<string>(),
      });
    const base = est({});
    expect(base.estimatedDownloadMB).toBe(131.3);
    expect(base.workItems).toBe(1);
    expect(base.outputCardinality).toEqual({ structure: 1 });
    expect(base.stageBreakdown).toEqual({ generate: 1 });
    expect(base.estimatedVramMB).toBe(estimateArtiosBackboneVramMB(100, 1, 'int8'));
    expect(base.estimatedVramMB).toBe(ARTIOS_BACKBONE_MANIFEST.estimatedVramMB);
    expect(est({ precision: 'fp32' }).estimatedDownloadMB).toBe(521.4);
    expect(est({ numDesigns: 7 }).workItems).toBe(7);
    expect(est({ numDesigns: 99 }).workItems).toBe(16);
    expect(Object.keys(base.precisionOptions ?? {})).toEqual(['int8', 'fp16', 'fp32']);
    // VRAM grows with tokens² (pair bias) and with the batched design count.
    expect(estimateArtiosBackboneVramMB(512, 1, 'int8')).toBeGreaterThan(
      estimateArtiosBackboneVramMB(100, 1, 'int8'),
    );
    expect(estimateArtiosBackboneVramMB(100, 8, 'int8')).toBeGreaterThan(
      estimateArtiosBackboneVramMB(100, 1, 'int8'),
    );
    expect(artiosGpuBatchLimit(100)).toBe(8);
    expect(artiosGpuBatchLimit(256)).toBe(4);
    expect(artiosGpuBatchLimit(512)).toBe(2);
  });

  it('wires the worker to the vendored runtime, the resilient device helpers and the weight cache', () => {
    const src = readFileSync(
      resolve(process.cwd(), 'src/lib/plugins/artios-backbone/artios-backbone.worker.ts'),
      'utf8',
    );
    for (const needle of [
      'createWorkerHarness',
      'requestResilientWebGpuDevice',
      'createDeviceLostGuard',
      'isWebGpuDeviceLostError',
      'makeCachedFetch',
      'evictCachedUrls',
      'loadWeightBundle',
      'createArtiosBackboneCPU',
      'conditioningCache',
      'createArtiosBackboneGpu',
      'prepareConditioning',
      'denoiseStep',
      'runArtiosEdmSampler',
      'collectOwnedTransferables',
      '/vendor/artios-backbone',
      'VITE_ARTIOS_BACKBONE_WEIGHTS_URL',
      'https://huggingface.co/ramithuh/artios-webgpu/resolve/main/backbone',
    ]) {
      expect(src, `worker references ${needle}`).toContain(needle);
    }
    // The CPU denoiser is both the conditioning-cache path and the fallback denoiser.
    expect(src).toContain('cpuModel.denoise(');
    expect(src).toContain("fallbackReason = 'device-lost'");
  });

  it.skipIf(!WIRED)('keeps the shared node summary identical to the plugin canvas summary once wired', () => {
    const cases: Array<Record<string, unknown>> = [
      {},
      { length: 256, numSteps: 150, precision: 'fp16', numDesigns: 4 },
      { motifContig: 'A10-25', length: 120 },
      { motifContig: 'A10-25', includeLigand: false, numDesigns: 2 },
      getDefaultParamsFromSchema(ARTIOS_BACKBONE_MANIFEST as AdapterManifest),
    ];
    for (const params of cases) {
      const expected = artiosBackbonePlugin.display.formatCanvasSummary(params);
      expect(formatNodeParamSummary('artios-backbone', params), JSON.stringify(params)).toBe(expected);
    }
  });

  it.skipIf(!WIRED)('matches the shared ADAPTER_REGISTRY entry and model card once wired', () => {
    expect(ADAPTER_REGISTRY['artios-backbone']).toBe(ARTIOS_BACKBONE_MANIFEST);
    const card = getModelCardForAdapter('artios-backbone', { precision: 'int8' })!;
    expect(card.adapterId).toBe('artios-backbone');
    expect(card.activePrecisionId).toBe('int8');
    expect(card.weightsRepo).toBe('https://huggingface.co/ramithuh/artios-webgpu');
    expect(card.upstreamRepo).toBe('https://github.com/Ligo-Biosciences/artios');
    expect(card.license).toContain('Apache-2.0');
    expect(card.precisions.map((p) => p.id)).toEqual(['int8', 'fp16', 'fp32']);
    expect(card.precisions.find((p) => p.id === 'int8')!.approxSizeMB).toBe(131.3);
    expect(card.limitations.length).toBeGreaterThan(0);
  });
});

// ===========================================================================
// 4. parameter normalization
// ===========================================================================

describe('artios-backbone option normalization', () => {
  it('applies upstream defaults and clamps every numeric range', () => {
    expect(DEFAULT_OPTIONS).toEqual({
      length: 100,
      numDesigns: 1,
      numSteps: 200,
      motifContig: '',
      unindexedMotif: true,
      includeLigand: true,
      ligandRasa: -1,
      motifDepth: -1,
      stepScaleEta: 1.5,
      noiseScaleLambda: 1.003,
      gamma0: 0.8,
      selfConditioning: true,
      seed: 42,
      precision: 'int8',
    });
    const clamped = artiosBackboneOptionsFromParams({
      length: 9000,
      numDesigns: 0,
      numSteps: 1,
      stepScaleEta: 99,
      noiseScaleLambda: 0,
      gamma0: -5,
      seed: '7',
      motifContig: '  A10-25  ',
      unindexedMotif: false,
      includeLigand: false,
      selfConditioning: false,
      precision: 'FP16',
    });
    expect(clamped).toMatchObject({
      length: 512,
      numDesigns: 1,
      numSteps: 20,
      stepScaleEta: 3,
      noiseScaleLambda: 0.5,
      gamma0: 0,
      seed: 7,
      motifContig: 'A10-25',
      unindexedMotif: false,
      includeLigand: false,
      selfConditioning: false,
      precision: 'fp16',
    });
    // Blank strings fall back to the schema default rather than becoming NaN/0.
    expect(artiosBackboneOptionsFromParams({ length: '', numSteps: null }).length).toBe(100);
    expect(artiosBackboneOptionsFromParams({ numSteps: null }).numSteps).toBe(200);
  });

  it('treats -1 as the unconditioned null bin and rejects other negatives', () => {
    expect(artiosBackboneOptionsFromParams({ ligandRasa: 0.4, motifDepth: 1.25 })).toMatchObject({
      ligandRasa: 0.4,
      motifDepth: 1.25,
    });
    expect(artiosBackboneOptionsFromParams({ ligandRasa: 5, motifDepth: 5 })).toMatchObject({
      ligandRasa: 1,
      motifDepth: 2,
    });
    expect(() => artiosBackboneOptionsFromParams({ ligandRasa: -0.5 })).toThrow(/ligandRasa/);
    expect(() => artiosBackboneOptionsFromParams({ motifDepth: -3 })).toThrow(/motifDepth/);
    expect(normalizeArtiosBackbonePrecision(undefined)).toBe('int8');
    expect(normalizeArtiosBackbonePrecision(' Fp32 ')).toBe('fp32');
    expect(() => normalizeArtiosBackbonePrecision('int4')).toThrow(/Unsupported ArtiosBackbone precision/);
  });
});

// ===========================================================================
// 5. PDB parsing & motif contigs
// ===========================================================================

describe('artios-backbone PDB atom parsing', () => {
  it('drops hydrogens, waters and non-first altLocs, and resolves elements and charges', () => {
    const atoms = parsePdbAtomRecords(FIXTURE_PDB);
    expect(atoms.some((a) => a.element === 'H')).toBe(false);
    expect(atoms.some((a) => a.resName === 'HOH')).toBe(false);
    const res5 = atoms.filter((a) => a.resId === 5 && a.chainId === 'A');
    expect(res5.map((a) => a.atomName)).toEqual(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2']);
    expect(res5.find((a) => a.atomName === 'CB')!.coord[0]).toBeCloseTo(
      2.3 * Math.cos(4 * ((100 * Math.PI) / 180)) + 0.6,
      3,
    );
    const zn = atoms.find((a) => a.resName === 'ZN')!;
    expect(zn).toMatchObject({ element: 'ZN', chainId: 'B', resId: 101, hetero: true });
    expect(atoms.filter((a) => a.hetero)).toHaveLength(1);
    expect(parsePdbAtomRecords(FIXTURE_PDB, { keepWaters: true }).some((a) => a.resName === 'HOH')).toBe(true);
    expect(parsePdbAtomRecords(FIXTURE_PDB, { keepHydrogens: true }).some((a) => a.element === 'H')).toBe(true);
  });

  it('infers two-letter elements from column alignment and parses formal charges', () => {
    const pdb = [
      'ATOM      1  N   ALA A   1      11.000  12.000  13.000  1.00  0.00           N',
      'HETATM    2 ZN    ZN B 101       1.000   2.000   3.000  1.00  0.00          ZN2+',
      'HETATM    3 CL    CL B 102       4.000   5.000   6.000  1.00  0.00          CL1-',
      'HETATM    4  C1  LIG B 103       7.000   8.000   9.000  1.00  0.00',
      'END',
    ].join('\n');
    const atoms = parsePdbAtomRecords(pdb);
    expect(atoms.map((a) => [a.atomName, a.element, a.charge])).toEqual([
      ['N', 'N', 0],
      ['ZN', 'ZN', 2],
      ['CL', 'CL', -1],
      ['C1', 'C', 0],
    ]);
  });

  it('reads only the first MODEL and rejects non-finite coordinates', () => {
    const multi = [
      'MODEL        1',
      'ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C',
      'ENDMDL',
      'MODEL        2',
      'ATOM      2  CA  ALA A   1       9.000   9.000   9.000  1.00  0.00           C',
      'ENDMDL',
      'END',
    ].join('\n');
    const atoms = parsePdbAtomRecords(multi);
    expect(atoms).toHaveLength(1);
    expect(atoms[0].coord).toEqual([1, 2, 3]);
    const broken = 'ATOM      1  CA  ALA A   1         NaN   2.000   3.000  1.00  0.00           C\nEND';
    expect(() => parsePdbAtomRecords(broken)).toThrow(/non-finite coordinates/);
  });

  it('groups residues in file order and parses motif contigs', () => {
    const residues = groupResidues(parsePdbAtomRecords(FIXTURE_PDB));
    expect(residues).toHaveLength(FIXTURE_SEQ.length + 1); // 12 polymer residues + ZN
    expect(residues[0]).toMatchObject({ chainId: 'A', resId: 1, resName: 'ALA' });
    expect(residues.at(-1)).toMatchObject({ chainId: 'B', resId: 101, resName: 'ZN' });

    expect(parseMotifContig('')).toEqual([]);
    expect(parseMotifContig('A10-25')).toEqual([{ chain: 'A', start: 10, end: 25 }]);
    expect(parseMotifContig('A3-5, B77; 9')).toEqual([
      { chain: 'A', start: 3, end: 5 },
      { chain: 'B', start: 77, end: 77 },
      { chain: null, start: 9, end: 9 },
    ]);
    expect(() => parseMotifContig('A10-')).toThrow(/Invalid ArtiosBackbone motifContig token/);
    expect(() => parseMotifContig('A25-10')).toThrow(/end precedes start/);
  });
});

// ===========================================================================
// 6. token-layout planning
// ===========================================================================

describe('artios-backbone token-layout planning', () => {
  it('builds an unconditional poly-glycine backbone layout of the requested length', () => {
    const plan = buildUnconditionalPlan(40);
    expect(plan.designedLength).toBe(40);
    expect(plan.atoms).toHaveLength(40 * 4);
    expect(plan.atoms.slice(0, 4).map((a) => a.atomName)).toEqual(['N', 'CA', 'C', 'O']);
    expect(new Set(plan.atoms.map((a) => a.resName))).toEqual(new Set(['GLY']));
    expect(new Set(plan.atoms.map((a) => a.chainId))).toEqual(new Set(['A']));
    expect(plan.atoms[4].resId).toBe(2);
    expect(plan.atoms.at(-1)!.resId).toBe(40);
    expect([...plan.isMotifAtom].every((v) => v === 0)).toBe(true);
    expect([...plan.isHetAtom].every((v) => v === 0)).toBe(true);
    expect(plan.motifIndices).toEqual([]);
    expect(plan.ligandResNames).toEqual([]);
    expect(plan.missingComponentCodes).toEqual([]);
    expect(plan.warnings).toEqual([]);
  });

  it('places a motif at its native numbering (indexed mode), keeps its side chains and conditions on the ligand', () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 60, motifContig: 'A3-6', includeLigand: true, unindexedMotif: false },
    });
    expect(plan.designedLength).toBe(60);
    expect(plan.motifIndices).toEqual([2, 3, 4, 5]); // residues 3..6 -> 0-based 2..5
    expect(plan.fixedResidueIndices).toEqual([3, 4, 5, 6]);
    expect(plan.unindexedMotifResidues).toEqual([]);
    expect(plan.warnings).toEqual([]);

    // Motif residues keep their identity and every heavy atom; scaffold positions are GLY N/CA/C/O.
    const byRes = new Map<number, string[]>();
    for (const atom of plan.atoms.filter((a) => a.chainId === 'A')) {
      if (!byRes.has(atom.resId)) byRes.set(atom.resId, []);
      byRes.get(atom.resId)!.push(atom.atomName);
    }
    expect(plan.atoms.find((a) => a.resId === 3)!.resName).toBe('SER');
    expect(byRes.get(3)).toEqual(['N', 'CA', 'C', 'O', 'CB', 'OG']);
    expect(plan.atoms.find((a) => a.resId === 5)!.resName).toBe('VAL');
    expect(byRes.get(5)).toEqual(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2']);
    expect(byRes.get(1)).toEqual(['N', 'CA', 'C', 'O']);
    expect(plan.atoms.find((a) => a.resId === 1)!.resName).toBe('GLY');

    // nativeMotifCoords are the untouched input Cα positions.
    const inputCa = parsePdbAtomRecords(FIXTURE_PDB).filter(
      (a) => a.atomName === 'CA' && a.resId >= 3 && a.resId <= 6,
    );
    expect(plan.nativeMotifCoords).toEqual(inputCa.map((a) => [...a.coord]));

    // The ligand becomes its own chain of motif-flagged HETATM tokens with resId = 1.
    expect(plan.ligandResNames).toEqual(['ZN']);
    expect(plan.ligandChainIds).toEqual(['B']);
    const lig = plan.atoms.filter((a, i) => plan.isHetAtom[i] === 1);
    expect(lig).toHaveLength(1);
    expect(lig[0]).toMatchObject({ resName: 'ZN', chainId: 'B', resId: 1, insCode: '', element: 'ZN' });
    expect(plan.missingComponentCodes).toEqual(['ZN']); // fetched from the RCSB CCD at run time

    const motifAtomCount = plan.atoms.filter((_, i) => plan.isMotifAtom[i] === 1).length;
    expect(motifAtomCount).toBe(
      plan.atoms.filter((a) => a.chainId === 'A' && a.resId >= 3 && a.resId <= 6).length + 1,
    );
  });

  it('allocates separate output chains (B, C, ...) and resId=1 for multiple ligands sharing an input chain', () => {
    const multiLigPdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N',
      'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C',
      'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C',
      'ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O',
      'HETATM    5 FE    FE A 501       5.000   5.000   5.000  1.00  0.00          FE',
      'HETATM    6  C1  FPY A 502       7.000   5.000   5.000  1.00  0.00           C',
      'HETATM    7  O1  FPY A 502       8.200   5.000   5.000  1.00  0.00           O',
      'END',
    ].join('\n');
    const plan = buildMotifPlan({
      pdbText: multiLigPdb,
      options: { length: 10, motifContig: '', includeLigand: true },
    });
    expect(plan.ligandResNames).toEqual(['FE', 'FPY']);
    expect(plan.ligandChainIds).toEqual(['B', 'C']);
    const hetAtoms = plan.atoms.filter((_, i) => plan.isHetAtom[i] === 1);
    expect(hetAtoms).toHaveLength(3);
    expect(hetAtoms[0]).toMatchObject({ resName: 'FE', chainId: 'B', resId: 1, insCode: '' });
    expect(hetAtoms[1]).toMatchObject({ resName: 'FPY', chainId: 'C', resId: 1, insCode: '' });
    expect(hetAtoms[2]).toMatchObject({ resName: 'FPY', chainId: 'C', resId: 1, insCode: '' });

    // Tokenization assigns distinct asym_id and entity_id to FE vs FPY.
    const { result } = featuresFor(plan);
    const f = result.features;
    const nTok = f.nToken;
    // Last 3 tokens are FE (chain B), FPY C1 (chain C), FPY O1 (chain C).
    expect(f.asym_id[nTok - 3]).not.toBe(f.asym_id[nTok - 2]);
    expect(f.entity_id[nTok - 3]).not.toBe(f.entity_id[nTok - 2]);
    expect(f.asym_id[nTok - 2]).toBe(f.asym_id[nTok - 1]);
    expect(f.residue_index[nTok - 3]).toBe(1);
    expect(f.residue_index[nTok - 2]).toBe(1);
  });

  it('builds native unindexed motif layout (chain A full GLY + M0/M1... chains + ligands)', () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 20, motifContig: 'A3, A5', includeLigand: true, unindexedMotif: true },
    });
    expect(plan.designedLength).toBe(20);
    expect(plan.motifIndices).toEqual([]);
    expect(plan.fixedResidueIndices).toEqual([]);
    expect(plan.unindexedMotifResidues).toHaveLength(2);
    expect(plan.unindexedMotifResidues[0]).toMatchObject({
      resName: 'SER',
      sourceChainId: 'A',
      sourceResId: 3,
      atomStart: 20 * 4,
      atomCount: 6,
      caOffset: 1,
    });
    expect(plan.unindexedMotifResidues[1]).toMatchObject({
      resName: 'VAL',
      sourceChainId: 'A',
      sourceResId: 5,
      atomStart: 20 * 4 + 6,
      atomCount: 7,
      caOffset: 1,
    });

    // Chain A is pure 20-residue GLY scaffold (isMotifAtom = 0).
    const chainAAtoms = plan.atoms.filter((a) => a.chainId === 'A');
    expect(chainAAtoms).toHaveLength(20 * 4);
    expect(new Set(chainAAtoms.map((a) => a.resName))).toEqual(new Set(['GLY']));
    expect([...plan.isMotifAtom.slice(0, 80)].every((v) => v === 0)).toBe(true);

    // M0 and M1 carry the motif atoms with resId = 1, hetero = false, isMotifAtom = 1, isHetAtom = 0.
    const m0 = plan.atoms.filter((a) => a.chainId === 'M0');
    const m1 = plan.atoms.filter((a) => a.chainId === 'M1');
    expect(m0).toHaveLength(6);
    expect(m1).toHaveLength(7);
    expect(m0.every((a) => a.resId === 1 && !a.hetero)).toBe(true);
    expect(m1.every((a) => a.resId === 1 && !a.hetero)).toBe(true);
    expect([...plan.isMotifAtom.slice(80, 93)].every((v) => v === 1)).toBe(true);
    expect([...plan.isHetAtom.slice(80, 93)].every((v) => v === 0)).toBe(true);

    // Ligand ZN is appended on chain B.
    expect(plan.ligandResNames).toEqual(['ZN']);
    expect(plan.ligandChainIds).toEqual(['B']);
    expect(plan.atoms[93]).toMatchObject({ resName: 'ZN', chainId: 'B', resId: 1, hetero: true });

    // Tokenization gives 20 scaffold tokens + 6 M0 tokens + 7 M1 tokens + 1 ZN token = 34 tokens,
    // with distinct asym_id for A, M0, M1, B.
    const { result } = featuresFor(plan);
    const f = result.features;
    expect(f.nToken).toBe(20 + 6 + 7 + 1);
    expect(f.asym_id[0]).toBe(0); // chain A
    expect(f.asym_id[20]).toBe(1); // chain M0
    expect(f.asym_id[26]).toBe(2); // chain M1
    expect(f.asym_id[33]).toBe(3); // chain B (ZN)
    expect([...f.is_motif_token.slice(0, 20)].every((v) => v === 0)).toBe(true);
    expect([...f.is_motif_token.slice(20)].every((v) => v === 1)).toBe(true);
    expect([...f.is_protein_token.slice(20, 33)].every((v) => v === 1)).toBe(true);
    expect(f.is_ligand_token[33]).toBe(1);
  });

  it('stitches unindexed motif coordinates via exact 1-to-1 Hungarian matching and round-trips through PDB', () => {
    // Direct Hungarian test where greedy matching fails:
    // Row 0 -> [1, 2], Row 1 -> [0, 100]. Greedy picks (0->0, 1->1) = 101; Hungarian picks (0->1, 1->0) = 2.
    expect([...solveMinCostAssignment(Float64Array.from([1, 2, 0, 100]), 2, 2)]).toEqual([1, 0]);

    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 16, motifContig: 'A3, A5', includeLigand: true, unindexedMotif: true },
    });
    const coords = new Float32Array(plan.atoms.length * 3);
    // Place scaffold residues i = 0..15 along the x-axis at x = i * 4.0.
    for (let i = 0; i < 16; i++) {
      for (let a = 0; a < 4; a++) {
        const idx = (i * 4 + a) * 3;
        coords[idx] = i * 4.0 + a * 0.1;
        coords[idx + 1] = 0;
        coords[idx + 2] = 0;
      }
    }
    // Place M0 (SER) near scaffold slot 12 (x = 48.0), M1 (VAL) near scaffold slot 4 (x = 16.0).
    const [m0, m1] = plan.unindexedMotifResidues;
    for (let k = 0; k < m0.atomCount; k++) {
      const idx = (m0.atomStart + k) * 3;
      coords[idx] = 48.05 + k * 0.2;
      coords[idx + 1] = 0.3;
      coords[idx + 2] = -0.1;
    }
    for (let k = 0; k < m1.atomCount; k++) {
      const idx = (m1.atomStart + k) * 3;
      coords[idx] = 16.05 + k * 0.2;
      coords[idx + 1] = -0.2;
      coords[idx + 2] = 0.4;
    }
    // Place ligand ZN at [30, 5, 2].
    const znIdx = (plan.atoms.length - 1) * 3;
    coords[znIdx] = 30.0;
    coords[znIdx + 1] = 5.0;
    coords[znIdx + 2] = 2.0;

    const stitched = stitchUnindexedMotifCoords(coords, plan);
    expect(stitched.plan.designedLength).toBe(16);
    expect(stitched.plan.unindexedMotifResidues).toEqual([]);
    // Sorted stitched 0-based indices: slot 4 (M1 = VAL) and slot 12 (M0 = SER).
    expect(stitched.plan.motifIndices).toEqual([4, 12]);
    expect(stitched.plan.fixedResidueIndices).toEqual([5, 13]);
    // nativeMotifCoords aligns with sorted motifIndices: M1 first, then M0!
    expect(stitched.plan.nativeMotifCoords).toEqual([m1.nativeCaCoord, m0.nativeCaCoord]);
    // Ligand preserved at the end with its sampled coordinates.
    expect(stitched.plan.ligandResNames).toEqual(['ZN']);
    const outZnIdx = (stitched.plan.atoms.length - 1) * 3;
    expect([stitched.coords[outZnIdx], stitched.coords[outZnIdx + 1], stitched.coords[outZnIdx + 2]]).toEqual([
      30, 5, 2,
    ]);

    // PDB & StructureArtifact round-trip.
    const pdbText = coordsToArtiosPdb(stitched.coords, stitched.plan);
    const parsed = parseStructureText(pdbText);
    expect(parsed.nResidues).toBe(16);
    expect(parsed.sequence[4]).toBe('V'); // M1 (VAL) stitched onto slot 4
    expect(parsed.sequence[12]).toBe('S'); // M0 (SER) stitched onto slot 12
    expect(parsed.diagnostics.ligandNames).toContain('ZN');

    const artifact = buildArtiosBackboneArtifact(
      { pdbText, designIndex: 0, seed: 42, stepsRun: 20, warnings: [] },
      { options: { ...DEFAULT_OPTIONS, length: 16 }, plan: stitched.plan, backend: 'webgpu' },
    );
    expect(artifact.length).toBe(16);
    expect(artifact.motifIndices).toEqual([4, 12]);
    expect(artifact.fixedResidueIndices).toEqual([5, 13]);
    expect(artifact.nativeMotifCoords).toEqual([m1.nativeCaCoord, m0.nativeCaCoord]);
  });

  it('keeps the ligand out of the layout when includeLigand is false', () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 40, motifContig: 'A2', includeLigand: false, unindexedMotif: false },
    });
    expect(plan.ligandResNames).toEqual([]);
    expect([...plan.isHetAtom].every((v) => v === 0)).toBe(true);
    expect(plan.missingComponentCodes).toEqual([]);
    expect(plan.motifIndices).toEqual([1]);
  });

  it('supports ligand-only conditioning with no motif span', () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 35, motifContig: '', includeLigand: true },
    });
    expect(plan.motifIndices).toEqual([]);
    expect(plan.ligandResNames).toEqual(['ZN']);
    expect(plan.atoms).toHaveLength(35 * 4 + 1);
    expect(plan.isMotifAtom[plan.atoms.length - 1]).toBe(1);
  });

  it('center-shifts overflowing indexed spans preserving spacing, evenly distributes wide spans, and rejects impossible requests', () => {
    const fits = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 30, motifContig: 'A9-12', includeLigand: false, unindexedMotif: false },
    });
    expect(fits.motifIndices).toEqual([8, 9, 10, 11]);
    expect(fits.warnings).toEqual([]);

    // Overflowing single span A9-12 (width 4) in L=10 is rigidly center-shifted to [3, 4, 5, 6].
    const centered = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 10, motifContig: 'A9-12', includeLigand: false, unindexedMotif: false },
    });
    expect(centered.motifIndices).toEqual([3, 4, 5, 6]);
    expect(centered.warnings.join(' ')).toMatch(/was shifted to design position 4/);

    // Multi-span rigid center-shift (1RA0-style A156, A217 in L=180): preserves exact 61-aa spacing!
    const makeHighResPdb = (resIds: number[]) =>
      [
        ...resIds.flatMap((id, k) => [
          `ATOM  ${String(k * 4 + 1).padStart(5, ' ')}  N   ALA A${String(id).padStart(4, ' ')}       0.000   0.000   0.000  1.00  0.00           N`,
          `ATOM  ${String(k * 4 + 2).padStart(5, ' ')}  CA  ALA A${String(id).padStart(4, ' ')}       1.458   0.000   0.000  1.00  0.00           C`,
          `ATOM  ${String(k * 4 + 3).padStart(5, ' ')}  C   ALA A${String(id).padStart(4, ' ')}       2.009   1.420   0.000  1.00  0.00           C`,
          `ATOM  ${String(k * 4 + 4).padStart(5, ' ')}  O   ALA A${String(id).padStart(4, ' ')}       1.251   2.390   0.000  1.00  0.00           O`,
        ]),
        'END',
      ].join('\n');

    const ra0 = buildMotifPlan({
      pdbText: makeHighResPdb([156, 217]),
      options: { length: 180, motifContig: 'A156, A217', includeLigand: false, unindexedMotif: false },
    });
    expect(ra0.motifIndices).toEqual([59, 120]);
    expect(ra0.motifIndices[1] - ra0.motifIndices[0]).toBe(61);

    // Wide multi-span (native width > L) falls back to even distribution, never crushing onto C-terminus.
    const wide = buildMotifPlan({
      pdbText: makeHighResPdb([5, 6, 50, 51]),
      options: { length: 20, motifContig: 'A5-6, A50-51', includeLigand: false, unindexedMotif: false },
    });
    expect(wide.motifIndices).toEqual([6, 7, 13, 14]);
    expect(wide.motifIndices.at(-1)!).toBeLessThan(19);

    expect(() =>
      buildMotifPlan({
        pdbText: FIXTURE_PDB,
        options: { length: 3, motifContig: 'A1-12', includeLigand: false, unindexedMotif: false },
      }),
    ).toThrow(/motif needs 12 residues but length is 3/);
    expect(() =>
      buildMotifPlan({
        pdbText: FIXTURE_PDB,
        options: { length: 40, motifContig: 'Z1-5', includeLigand: false, unindexedMotif: false },
      }),
    ).toThrow(/matched 0 polymer residues/);
  });

  it('places multiple spans in order without overlap', () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 40, motifContig: 'A2-3, A7-8', includeLigand: false, unindexedMotif: false },
    });
    expect(plan.motifIndices).toEqual([1, 2, 6, 7]);
    const resNames = plan.motifIndices.map(
      (i) => plan.atoms.find((a) => a.chainId === 'A' && a.resId === i + 1)!.resName,
    );
    expect(resNames).toEqual([FIXTURE_SEQ[1], FIXTURE_SEQ[2], FIXTURE_SEQ[6], FIXTURE_SEQ[7]]);
  });
});

// ===========================================================================
// 7. feature assembly
// ===========================================================================

function featuresFor(
  plan: ArtiosTokenPlan,
  overrides: Partial<typeof DEFAULT_OPTIONS> & { augment?: boolean } = {},
) {
  const components = buildStandardComponents();
  for (const code of plan.missingComponentCodes) {
    // Stand in for the run-time RCSB fetch with the structure's own conformer.
    const atoms = plan.atoms.filter((a) => a.resName === code);
    components.set(code, componentFromInputLigand(code, atoms));
  }
  return {
    components,
    result: buildArtiosFeatures(shared, plan, { ...DEFAULT_OPTIONS, ...overrides }, components),
  };
}

describe('artios-backbone feature assembly', () => {
  it('emits a feature block the vendored prepareFeatures accepts (unconditional)', () => {
    const plan = buildUnconditionalPlan(16);
    const { result } = featuresFor(plan);
    const f = result.features;
    expect(f.nAtom).toBe(64);
    expect(f.nToken).toBe(16);
    const prepared = cpu.prepareFeatures(f);
    expect(prepared.nAtom).toBe(64);
    expect(prepared.nToken).toBe(16);
    expect(prepared.__prepared).toBe(true);

    expect(f.ref_pos).toHaveLength(64 * 3);
    expect(f.ref_element).toHaveLength(64 * 128);
    expect(f.ref_atom_name_chars).toHaveLength(64 * 256);
    expect(f.motif_restype).toHaveLength(16 * 32);
    expect([...f.atom_mask].every((v) => v === 1)).toBe(true);
    expect([...f.token_mask].every((v) => v === 1)).toBe(true);
    expect([...f.is_protein].every((v) => v === 1)).toBe(true);
    expect([...f.is_ligand].every((v) => v === 0)).toBe(true);
    expect([...f.is_protein_token].every((v) => v === 1)).toBe(true);
    expect([...f.is_motif].every((v) => v === 0)).toBe(true);
    expect([...f.motif_coordinate].every((v) => v === 0)).toBe(true);
    expect([...f.motif_depth_norm].every((v) => v === -1)).toBe(true);
    expect([...f.rasa].every((v) => v === -1)).toBe(true);
    expect([...f.residue_index]).toEqual(Array.from({ length: 16 }, (_, i) => i + 1));
    expect([...f.token_index]).toEqual(Array.from({ length: 16 }, (_, i) => i));
    expect([...f.asym_id].every((v) => v === 0)).toBe(true);
    // Reference conformers are per-residue augmented CCD ideal coordinates by default (augment: true).
    expect([...f.ref_space_uid.slice(0, 8)]).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(Math.abs(f.ref_pos[0] - f.ref_pos[12])).toBeGreaterThan(1e-4);
    // Intra-residue bond lengths are preserved by the rigid per-residue augmentation.
    const dist = (a: number, b: number) =>
      Math.hypot(
        f.ref_pos[a * 3] - f.ref_pos[b * 3],
        f.ref_pos[a * 3 + 1] - f.ref_pos[b * 3 + 1],
        f.ref_pos[a * 3 + 2] - f.ref_pos[b * 3 + 2],
      );
    expect(dist(0, 1)).toBeCloseTo(dist(4, 5), 4);
    expect([...f.ref_mask].every((v) => v === 1)).toBe(true);
    expect(result.unresolvedAtomCount).toBe(0);
    expect(result.missingComponents).toEqual([]);
    expect(result.warnings).toEqual([]);

    // Deterministic for a given seed, diverges across seeds, and unaugmented repeats identical conformers.
    const sameSeed = featuresFor(plan, { seed: 42 }).result.features.ref_pos;
    const diffSeed = featuresFor(plan, { seed: 99 }).result.features.ref_pos;
    const unaug = featuresFor(plan, { augment: false }).result.features.ref_pos;
    expect([...sameSeed]).toEqual([...f.ref_pos]);
    expect([...diffSeed]).not.toEqual([...f.ref_pos]);
    expect(unaug[0]).toBeCloseTo(unaug[12], 6);
  });

  it('conditions motif tokens on their native coordinates and the ligand on RASA', () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 24, motifContig: 'A3-6', includeLigand: true, unindexedMotif: false },
    });
    const { result } = featuresFor(plan, { ligandRasa: 0.35, motifDepth: 1.1 });
    const f = result.features;
    expect(cpu.prepareFeatures(f).nToken).toBe(f.nToken);
    // Upstream Artios tokenizes non-motif standard residues 1/residue (20 tokens),
    // motif residues 1/atom (each motif heavy atom gets its own token), and ligand 1/atom (1 Zn).
    const motifAtomCount = Array.from(plan.isMotifAtom).reduce(
      (sum, v, idx) => sum + (v === 1 && plan.isHetAtom[idx] === 0 ? 1 : 0),
      0,
    );
    const expectedTokens = 20 + motifAtomCount + 1;
    expect(f.nToken).toBe(expectedTokens);
    expect(f.is_ligand_token[expectedTokens - 1]).toBe(1);
    expect([...f.is_ligand_token.slice(0, expectedTokens - 1)].every((v) => v === 0)).toBe(true);
    expect(f.is_ligand[f.nAtom - 1]).toBe(1);
    expect(f.is_protein[f.nAtom - 1]).toBe(0);
    expect([...f.is_dna].every((v) => v === 0)).toBe(true);
    expect([...f.is_rna_token].every((v) => v === 0)).toBe(true);

    // Every motif atom token (tokens 2 .. 2+motifAtomCount-1) plus the ligand token carry motif conditioning.
    expect([...f.is_motif_token]).toEqual(
      Array.from(
        { length: expectedTokens },
        (_, t) => (t >= 2 && t < 2 + motifAtomCount) || t === expectedTokens - 1 ? 1 : 0,
      ),
    );
    expect([...f.is_motif_related_token]).toEqual([...f.is_motif_token]);
    const firstMotifAtom = plan.atoms.find((_, idx) => plan.isMotifAtom[idx] === 1 && plan.isHetAtom[idx] === 0)!;
    expect([f.motif_coordinate[2 * 3], f.motif_coordinate[2 * 3 + 1], f.motif_coordinate[2 * 3 + 2]]).toEqual(
      firstMotifAtom.coord.map((v) => Math.fround(v)),
    );
    expect([...f.motif_coordinate.slice(0, 6)].every((v) => v === 0)).toBe(true);

    // motif_restype one-hots the real residue identity of motif tokens only.
    const restypeOf = (t: number) => [...f.motif_restype.slice(t * 32, (t + 1) * 32)].indexOf(1);
    expect(restypeOf(2)).toBe(shared.RESIDUE_VOCAB.SER);
    expect(restypeOf(2 + motifAtomCount - 1)).toBe(shared.RESIDUE_VOCAB.ALA);
    expect(restypeOf(expectedTokens - 1)).toBe(shared.RESIDUE_VOCAB.UNK); // ZN is not in the restype vocabulary
    expect(restypeOf(0)).toBe(-1);

    // Conditioning targets land on the ligand / every token respectively.
    expect(f.rasa[expectedTokens - 1]).toBeCloseTo(0.35, 6);
    expect([...f.rasa.slice(0, expectedTokens - 1)].every((v) => v === -1)).toBe(true);
    expect([...f.motif_depth_norm].every((v) => Math.abs(v - 1.1) < 1e-6)).toBe(true);

    // The ligand's atom name is anonymized to <ELEMENT><n> (ContextGenerator.get_renamed_atom_names).
    const nameChars = f.ref_atom_name_chars.slice((f.nAtom - 1) * 256, f.nAtom * 256);
    const decoded = [0, 1, 2, 3]
      .map((j) => String.fromCharCode([...nameChars.slice(j * 64, (j + 1) * 64)].indexOf(1) + 32))
      .join('')
      .trim();
    expect(decoded).toBe('ZN1');
  });

  it('reports atoms without a reference conformer instead of failing', () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 20, motifContig: '', includeLigand: true },
    });
    // No component registered for ZN: the vendored featurizer zeroes its ref_pos and ref_mask.
    const result = buildArtiosFeatures(shared, plan, DEFAULT_OPTIONS, buildStandardComponents());
    expect(result.unresolvedAtomCount).toBe(1);
    expect(result.missingComponents).toEqual(['ZN']);
    expect(result.warnings.join(' ')).toMatch(/no CCD reference conformer \(ZN\)/);
    const f = result.features;
    expect(f.ref_mask[f.nAtom - 1]).toBe(0);
    expect([...f.ref_pos.slice((f.nAtom - 1) * 3)]).toEqual([0, 0, 0]);
    // Unknown components still tokenize as ligand atoms.
    expect(f.is_ligand[f.nAtom - 1]).toBe(1);
  });

  it('rejects unknown elements and empty layouts', () => {
    const plan = buildUnconditionalPlan(2);
    plan.atoms[0] = { ...plan.atoms[0], element: 'XX' };
    expect(() => buildArtiosFeatures(shared, plan, DEFAULT_OPTIONS, buildStandardComponents())).toThrow(
      /unknown element "XX"/,
    );
    const empty = { ...buildUnconditionalPlan(1), atoms: [] } as ArtiosTokenPlan;
    expect(() => buildArtiosFeatures(shared, empty, DEFAULT_OPTIONS, buildStandardComponents())).toThrow(
      /empty token layout/,
    );
  });
});

// ===========================================================================
// 8. PDB serialization & artifact assembly
// ===========================================================================

describe('artios-backbone PDB serialization & artifact assembly', () => {
  it('writes ATOM/HETATM records, chain IDs and TER that round-trip through parseStructureText', () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 20, motifContig: 'A3-6', includeLigand: true, unindexedMotif: false },
    });
    const coords = Float32Array.from({ length: plan.atoms.length * 3 }, (_, i) => (i % 97) * 0.37 - 10);
    const pdbText = coordsToArtiosPdb(coords, plan, { remarks: ['unit test remark'] });
    expect(pdbText.startsWith('REMARK   1 unit test remark')).toBe(true);
    const atomLines = pdbText.split('\n').filter((l) => l.startsWith('ATOM  '));
    const hetLines = pdbText.split('\n').filter((l) => l.startsWith('HETATM'));
    expect(atomLines).toHaveLength(plan.atoms.length - 1);
    expect(hetLines).toHaveLength(1);
    expect(hetLines[0].slice(17, 20)).toBe(' ZN');
    expect(hetLines[0].slice(21, 22)).toBe('B');
    expect(hetLines[0].slice(76, 78)).toBe('ZN');
    expect(atomLines[0].slice(12, 16)).toBe(' N  ');
    expect(atomLines[0].slice(17, 20)).toBe('GLY');
    expect(pdbText).toContain('\nTER\n');
    expect(pdbText.trimEnd().endsWith('END')).toBe(true);

    const parsed = parseStructureText(pdbText);
    expect(parsed.nResidues).toBe(20);
    expect(parsed.diagnostics.ligandNames).toContain('ZN');
    expect([...parsed.caCoords].every(Number.isFinite)).toBe(true);
    expect(parsed.sequence).toHaveLength(20);
    expect(parsed.sequence[2]).toBe('S'); // motif residue 3 is SER

    expect(() => coordsToArtiosPdb(new Float32Array(3), plan)).toThrow(/expected \d+ coordinates/);
    const bad = coords.slice();
    bad[1] = Number.NaN;
    expect(() => coordsToArtiosPdb(bad, plan)).toThrow(/non-finite coordinate/);
  });

  it('builds a StructureArtifact with motif lineage, provenance and multi-design collections', () => {
    const parent = fixtureStructure();
    // `length` below the schema minimum (30) is clamped by artiosBackboneOptionsFromParams.
    expect(artiosBackboneOptionsFromParams({ length: 20 }).length).toBe(30);
    const options = artiosBackboneOptionsFromParams({
      length: 40,
      motifContig: 'A3-6',
      unindexedMotif: false,
      numDesigns: 2,
    });
    const plan = buildMotifPlan({ pdbText: FIXTURE_PDB, options });
    const makeArtifact = (designIndex: number, backend: 'webgpu' | 'cpu') => {
      const coords = Float32Array.from(
        { length: plan.atoms.length * 3 },
        (_, i) => Math.sin(i * 0.7 + designIndex) * 12,
      );
      return buildArtiosBackboneArtifact(
        {
          pdbText: coordsToArtiosPdb(coords, plan),
          designIndex,
          seed: options.seed + designIndex,
          stepsRun: options.numSteps,
          warnings: [],
        },
        { options, plan, backend, parent, fallbackReason: backend === 'cpu' ? 'device-lost' : undefined },
      );
    };

    const first = makeArtifact(0, 'webgpu');
    expect(first.kind).toBe('StructureArtifact');
    expect(first.sourceModelId).toBe('artios-backbone/int8');
    expect(first.parentStructureId).toBe('fixture-helix');
    expect(first.length).toBe(40);
    expect(first.id).toMatch(/^ab-40aa-s42-/);
    expect(first.label).toContain('motif');
    expect(first.designKey).toBe('design-1');
    expect(first.designIndex).toBe(0);
    expect(first.seed).toBe(42);
    expect(first.atomGranularity).toBe('all-atom'); // motif side chains are kept
    expect(first.bfactorSemantics).toBe('none');
    expect(first.chainRoles).toEqual({ A: 'scaffold' });
    expect(first.molecularContext).toEqual({ ligands: ['ZN'], chainTypes: { A: 'protein', B: 'ligand' } });
    expect(first.motifIndices).toEqual([2, 3, 4, 5]);
    expect(first.fixedResidueIndices).toEqual([3, 4, 5, 6]);
    expect(first.nativeMotifCoords).toEqual(plan.nativeMotifCoords);
    expect(first.provenance).toMatchObject({
      quantization: 'int8',
      diffusionStepsScheduled: 200,
      diffusionStepsRun: 200,
    });
    expect(first.provenance?.degradedFallback).toBeUndefined();
    expect(first.geometry.length).toBe(40);
    expect([...first.caCoords].every(Number.isFinite)).toBe(true);

    const second = makeArtifact(1, 'cpu');
    expect(second.provenance?.degradedFallback).toBe('cpu-js');
    expect(second.provenance?.warnings?.join(' ')).toMatch(/WebGPU device loss/);
    expect(second.id).not.toBe(first.id);

    const wrapped = wrapArtiosBackboneOutputs([first, second]);
    expect(wrapped.id).toMatch(/^ab-batch-2-/);
    expect(wrapped.collection?.items).toHaveLength(2);
    expect(wrapped.collection?.keys).toEqual(['design-1', 'design-2']);
    expect(wrapped.collection?.collectionKind).toBe('independent-candidates');
    expect(wrapped.collection?.requestedCount).toBe(2);
    expect(wrapped.representativeSelection?.selectedKey).toBe('design-1');
    expect(wrapArtiosBackboneOutputs([first])).toBe(first);
    expect(() => wrapArtiosBackboneOutputs([])).toThrow(/produced no designs/);
  });

  it('emits a backbone-only artifact for unconditional designs', () => {
    const options = artiosBackboneOptionsFromParams({ length: 30 });
    const plan = buildUnconditionalPlan(options.length);
    const coords = Float32Array.from({ length: plan.atoms.length * 3 }, (_, i) => Math.cos(i * 0.31) * 14);
    const artifact = buildArtiosBackboneArtifact(
      { pdbText: coordsToArtiosPdb(coords, plan), designIndex: 0, seed: 42, stepsRun: 200, warnings: [] },
      { options, plan, backend: 'webgpu' },
    );
    expect(artifact.atomGranularity).toBe('backbone4');
    expect(artifact.molecularContext).toBeUndefined();
    expect(artifact.motifIndices).toBeUndefined();
    expect(artifact.fixedResidueIndices).toBeUndefined();
    expect(artifact.parentStructureId).toBeUndefined();
    expect(artifact.label).toContain('de novo');
    expect(artifact.sequence).toBe('G'.repeat(30));
    expect(artifact.backbone4Coords).toHaveLength(30 * 4 * 3);
  });
});

// ===========================================================================
// 9. end-to-end tiny-config CPU EDM sampling
// ===========================================================================

describe('artios-backbone end-to-end CPU EDM sampling (tiny config, synthetic weights)', () => {
  const weights = syntheticWeights(TINY_CONFIG, shared.createRng(123));

  function run(
    plan: ArtiosTokenPlan,
    overrides: Partial<typeof DEFAULT_OPTIONS> = {},
    seed = 7,
  ) {
    const { result } = featuresFor(plan, overrides);
    const features = cpu.prepareFeatures(result.features);
    const model = cpu.createArtiosBackboneCPU({ weights, config: TINY_CONFIG });
    const cache = model.conditioningCache(features, {
      motifAugmentation: { rng: shared.createRng(seed) },
    });
    const options = { ...DEFAULT_OPTIONS, numSteps: 3, ...overrides };
    const prevPredSeen: Array<Float32Array | null> = [];
    const steps: number[] = [];
    return {
      features,
      options,
      prevPredSeen,
      steps,
      coords: runArtiosEdmSampler(shared, {
        denoise: async (xNoisy, tHat, prevPred) => {
          prevPredSeen.push(prevPred);
          return model.denoise({
            features,
            rNoisy: xNoisy,
            noiseLevel: tHat,
            cache,
            prevPred,
          });
        },
        nAtom: features.nAtom,
        nSample: 1,
        atomMask: features.atom_mask,
        options,
        seed,
        onStep: (step) => steps.push(step),
      }),
    };
  }

  it('integrates the Karras schedule into finite coordinates and a valid PDB', async () => {
    const plan = buildUnconditionalPlan(16);
    const session = run(plan);
    const coords = await session.coords;
    expect(coords).toHaveLength(plan.atoms.length * 3);
    expect([...coords].every(Number.isFinite)).toBe(true);
    expect(session.steps).toEqual([1, 2, 3]);

    const pdbText = coordsToArtiosPdb(coords, plan, { remarks: ['tiny-config smoke test'] });
    const parsed = parseStructureText(pdbText);
    expect(parsed.nResidues).toBe(16);
    expect(parsed.sequence).toBe('G'.repeat(16));
    expect([...parsed.caCoords].every(Number.isFinite)).toBe(true);
    expect([...parsed.backbone4Coords].every(Number.isFinite)).toBe(true);

    const artifact = buildArtiosBackboneArtifact(
      { pdbText, designIndex: 0, seed: 7, stepsRun: 3, warnings: [] },
      { options: { ...session.options, length: 16 }, plan, backend: 'cpu' },
    );
    expect(artifact.length).toBe(16);
    expect(Number.isFinite(artifact.geometry.rg)).toBe(true);
    expect(artifact.provenance?.degradedFallback).toBe('cpu-js');
  });

  it('feeds back Kabsch-aligned self-conditioning after the first step', async () => {
    const plan = buildUnconditionalPlan(12);
    const withSelfCond = run(plan, { selfConditioning: true });
    await withSelfCond.coords;
    expect(withSelfCond.prevPredSeen).toHaveLength(3);
    expect(withSelfCond.prevPredSeen[0]).toBeNull();
    expect(withSelfCond.prevPredSeen[1]).not.toBeNull();
    expect([...withSelfCond.prevPredSeen[1]!].every(Number.isFinite)).toBe(true);

    const without = run(plan, { selfConditioning: false });
    const coordsWithout = await without.coords;
    expect(without.prevPredSeen.every((p) => p === null)).toBe(true);
    // Self-conditioning changes the trajectory (the network consumes prev_pred_embed).
    const coordsWith = await run(plan, { selfConditioning: true }).coords;
    let maxDiff = 0;
    for (let i = 0; i < coordsWith.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(coordsWith[i] - coordsWithout[i]));
    }
    expect(maxDiff).toBeGreaterThan(1e-6);
  });

  it('is deterministic for a seed and diverges across seeds', async () => {
    const plan = buildUnconditionalPlan(12);
    const a = await run(plan, {}, 11).coords;
    const b = await run(plan, {}, 11).coords;
    expect([...a]).toEqual([...b]);
    const c = await run(plan, {}, 12).coords;
    expect([...c]).not.toEqual([...a]);
  });

  it('samples a motif + ligand layout (unindexed + post-diffusion stitch) and keeps every token finite', async () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 14, motifContig: 'A3-6', includeLigand: true, unindexedMotif: true },
    });
    const rawCoords = await run(plan, { length: 14, ligandRasa: 0.4, motifDepth: 0.9 }).coords;
    expect(rawCoords).toHaveLength(plan.atoms.length * 3);
    expect([...rawCoords].every(Number.isFinite)).toBe(true);
    const { coords, plan: stitchedPlan } = stitchUnindexedMotifCoords(rawCoords, plan);
    const parsed = parseStructureText(coordsToArtiosPdb(coords, stitchedPlan));
    expect(parsed.nResidues).toBe(14);
    expect(stitchedPlan.motifIndices).toHaveLength(4);
    expect(parsed.diagnostics.ligandNames).toContain('ZN');
  });

  it('batches several designs in one sampler pass', async () => {
    const plan = buildUnconditionalPlan(10);
    const { result } = featuresFor(plan);
    const features = cpu.prepareFeatures(result.features);
    const model = cpu.createArtiosBackboneCPU({ weights, config: TINY_CONFIG });
    const cache = model.conditioningCache(features, {
      motifAugmentation: { rng: shared.createRng(3) },
    });
    const nSample = 3;
    const perSample = features.nAtom * 3;
    const coords = await runArtiosEdmSampler(shared, {
      // Batched denoiser: the real GPU engine takes [S * A * 3] in one call.
      denoise: async (xNoisy, tHat, prevPred) => {
        expect(xNoisy).toHaveLength(nSample * perSample);
        const out = new Float32Array(xNoisy.length);
        for (let s = 0; s < nSample; s++) {
          const slice = model.denoise({
            features,
            rNoisy: xNoisy.slice(s * perSample, (s + 1) * perSample),
            noiseLevel: tHat,
            cache,
            prevPred: prevPred ? prevPred.slice(s * perSample, (s + 1) * perSample) : null,
          });
          out.set(slice, s * perSample);
        }
        return out;
      },
      nAtom: features.nAtom,
      nSample,
      atomMask: features.atom_mask,
      options: { ...DEFAULT_OPTIONS, numSteps: 2 },
      seed: 5,
    });
    expect(coords).toHaveLength(nSample * perSample);
    expect([...coords].every(Number.isFinite)).toBe(true);
    // Independent per-sample augmentation and noise, so the designs differ.
    const first = coords.slice(0, perSample);
    const second = coords.slice(perSample, 2 * perSample);
    let diff = 0;
    for (let i = 0; i < perSample; i++) diff = Math.max(diff, Math.abs(first[i] - second[i]));
    expect(diff).toBeGreaterThan(1e-3);
  });

  it('produces a conditioning cache the WebGPU engine can consume directly', () => {
    const plan = buildMotifPlan({
      pdbText: FIXTURE_PDB,
      options: { length: 12, motifContig: 'A3-4', includeLigand: true },
    });
    const { result } = featuresFor(plan);
    const features = cpu.prepareFeatures(result.features);
    const model = cpu.createArtiosBackboneCPU({ weights, config: TINY_CONFIG });
    const cache = model.conditioningCache(features, {
      motifAugmentation: { rng: shared.createRng(1) },
    });

    // `_prepareConditioning` merges `{ ...cache.features, ...cache }` and picks these keys.
    expect(Object.keys(cache).sort()).toEqual(['features', 'nToken', 'sBase', 'z']);
    expect(cache.nToken).toBe(features.nToken);
    expect(cache.sBase).toHaveLength(features.nToken * TINY_CONFIG.cS);
    expect(cache.z).toHaveLength(features.nToken * features.nToken * TINY_CONFIG.cZ);
    expect([...cache.sBase].every(Number.isFinite)).toBe(true);
    expect([...cache.z].every(Number.isFinite)).toBe(true);

    const merged = { ...cache.features, ...cache } as Record<string, any>;
    expect(merged.atom_to_token_idx).toHaveLength(features.nAtom);
    expect(merged.token_mask).toHaveLength(features.nToken);
    expect(merged.atom_mask).toHaveLength(features.nAtom);
    // refFeatureMatrix: ref_charge [A] + ref_element [A, 128] + ref_atom_name_chars [A, 256].
    expect(merged.ref_charge).toHaveLength(features.nAtom);
    expect(merged.ref_element).toHaveLength(features.nAtom * 128);
    expect(merged.ref_atom_name_chars).toHaveLength(features.nAtom * 256);
    expect(1 + 128 + 256).toBe(gpu.N_REF_FEATURES);
  });

  it('centres and Kabsch-aligns each sample of a batch independently', () => {
    const nSample = 2;
    const nAtom = 4;
    const mask = new Uint8Array(nAtom).fill(1);
    const x = Float32Array.from({ length: nSample * nAtom * 3 }, (_, i) => i + 1);
    const centred = centreAugmentBatch(shared, x, nSample, mask, shared.createRng(2));
    const pairDist = (buf: Float32Array, base: number, i: number, j: number) =>
      Math.hypot(
        buf[base + i * 3] - buf[base + j * 3],
        buf[base + i * 3 + 1] - buf[base + j * 3 + 1],
        buf[base + i * 3 + 2] - buf[base + j * 3 + 2],
      );
    for (let s = 0; s < nSample; s++) {
      const base = s * nAtom * 3;
      // Centring + a proper rotation is rigid: every intra-sample distance survives.
      for (let i = 0; i < nAtom; i++) {
        for (let j = i + 1; j < nAtom; j++) {
          expect(pairDist(centred, base, i, j)).toBeCloseTo(pairDist(x, base, i, j), 4);
        }
      }
    }
    // Each sample draws its own rotation/translation, so the two samples are not a shared transform.
    const delta = (s: number) => centred[s * nAtom * 3] - x[s * nAtom * 3];
    expect(Math.abs(delta(0) - delta(1))).toBeGreaterThan(1e-6);
    // Aligning a rigidly transformed copy onto the target recovers the target.
    const target = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 3, 0, 0, 2, 1, 0, 2, 0, 1]);
    const moved = new Float32Array(target.length);
    for (let i = 0; i < target.length; i += 3) {
      // 90 degree rotation about z plus a translation.
      moved[i] = -target[i + 1] + 5;
      moved[i + 1] = target[i] - 3;
      moved[i + 2] = target[i + 2] + 1;
    }
    const aligned = alignPrevPredBatch(shared, moved, target, nSample, mask);
    for (let i = 0; i < target.length; i++) expect(aligned[i]).toBeCloseTo(target[i], 4);
  });
});
