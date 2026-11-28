// artios-potts (ArtiosPotts, Ligo-Biosciences/artios, Apache-2.0): plugin manifest & registration,
// PDB/mmCIF -> featurizer-spec compilation (protein-only, holo, nucleic acid, insertion codes),
// constraint/mask resolution (fixedPositions, inherited locks, ligand cutoff, omit alphabet),
// SequenceSet / ResidueTrack assembly, and an end-to-end CPU smoke test that runs the vendored
// runtime in `static/vendor/artios-potts/` (featurize -> Potts head -> score -> Gibbs/DLMC MCMC)
// on a deterministic synthetic weight bundle, plus bit-exactness of the row-restricted Gibbs sweep
// and count-based complexity penalty against the reference-shaped `legacyPath`.

import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getDefaultParamsFromSchema, validateParamsWithSchema } from '$lib/engine/param-schema';
import { createDefaultPluginRegistry } from '$lib/plugins';
import { artiosPottsPlugin, formatArtiosPottsSummary } from '$lib/plugins/artios-potts';
import { ARTIOS_POTTS_MANIFEST, ARTIOS_POTTS_WEIGHTS_MB } from '$lib/plugins/artios-potts/manifest';
import {
  artiosPottsOptionsFromParams,
  buildArtiosPottsCandidate,
  buildArtiosPottsInput,
  buildArtiosPottsSequenceSet,
  buildArtiosPottsTrack,
  buildFixedTokenMask,
  countDesignablePositions,
  flattenChainSequences,
  inputPdbText,
  planArtiosPottsJobs,
  proteinTokenValues,
  residueIdsOf,
  resolveArtiosPottsAlphabet,
  resolveArtiosPottsConstraints,
  sanitizeOmitAAs,
  structureItems,
  type ArtiosPottsSample,
} from '$lib/plugins/artios-potts/jobs';
import type { AdapterParamField } from '$lib/engine/types';
import type { StructureArtifact } from '$lib/engine/types';

/* eslint-disable @typescript-eslint/no-explicit-any */
const VENDOR = resolve(process.cwd(), 'static/vendor/artios-potts');
const load = async (rel: string): Promise<any> =>
  import(/* @vite-ignore */ pathToFileURL(join(VENDOR, rel)).href);

const shared = await load('shared/index.js');
const weightsMod = await load('webgpu/weights.js');
const cpuMod = await load('cpu/index.js');

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const pdbLine = (
  record: 'ATOM' | 'HETATM',
  serial: number,
  atom: string,
  resName: string,
  chain: string,
  resSeq: number,
  xyz: readonly [number, number, number],
  element: string,
  insCode = ' ',
  altLoc = ' ',
): string => {
  const name = atom.length >= 4 ? atom : ` ${atom.padEnd(3)}`;
  return (
    `${record.padEnd(6)}${String(serial).padStart(5)} ${name}${altLoc}${resName.padStart(3)} ${chain}` +
    `${String(resSeq).padStart(4)}${insCode}   ` +
    xyz.map((v) => v.toFixed(3).padStart(8)).join('') +
    `  1.00  0.00          ${element.padStart(2)}  `
  );
};

/** Ideal-ish α-helix backbone, enough for a real kNN graph and a virtual CB per residue. */
function helixPdb(
  nResidues: number,
  chain = 'A',
  options: { resNames?: string[]; startResSeq?: number; serial?: number; offset?: number } = {},
): { text: string; nextSerial: number } {
  const { resNames, startResSeq = 1, serial: startSerial = 1, offset = 0 } = options;
  const lines: string[] = [];
  let serial = startSerial;
  const rise = 1.5;
  const radius = 2.3;
  const turn = (100 * Math.PI) / 180;
  for (let i = 0; i < nResidues; i++) {
    const resName = resNames?.[i] ?? ['ALA', 'LEU', 'GLU', 'LYS', 'SER', 'THR', 'VAL', 'ILE'][i % 8];
    const a = i * turn;
    const z = i * rise + offset;
    const ca: [number, number, number] = [radius * Math.cos(a), radius * Math.sin(a), z];
    const n: [number, number, number] = [
      radius * Math.cos(a - 0.35),
      radius * Math.sin(a - 0.35),
      z - 0.6,
    ];
    const c: [number, number, number] = [
      radius * Math.cos(a + 0.35),
      radius * Math.sin(a + 0.35),
      z + 0.6,
    ];
    const o: [number, number, number] = [
      (radius + 1.1) * Math.cos(a + 0.5),
      (radius + 1.1) * Math.sin(a + 0.5),
      z + 0.9,
    ];
    const resSeq = startResSeq + i;
    lines.push(pdbLine('ATOM', serial++, 'N', resName, chain, resSeq, n, 'N'));
    lines.push(pdbLine('ATOM', serial++, 'CA', resName, chain, resSeq, ca, 'C'));
    lines.push(pdbLine('ATOM', serial++, 'C', resName, chain, resSeq, c, 'C'));
    lines.push(pdbLine('ATOM', serial++, 'O', resName, chain, resSeq, o, 'O'));
  }
  return { text: lines.join('\n'), nextSerial: serial };
}

const PROTEIN_ONLY_PDB = `${helixPdb(10).text}\nEND\n`;

/** 10-residue helix + a 3-atom zinc-like ligand near residues 1-3, a water, and hydrogens. */
const HOLO_PDB = (() => {
  const helix = helixPdb(10);
  let serial = helix.nextSerial;
  const het = [
    pdbLine('HETATM', serial++, 'ZN', ' ZN', 'B', 1, [4.0, 0.5, 1.5], 'ZN'),
    pdbLine('HETATM', serial++, 'C1', 'NAG', 'C', 2, [4.5, 1.2, 3.0], 'C'),
    pdbLine('HETATM', serial++, 'O1', 'NAG', 'C', 2, [5.2, 2.0, 3.4], 'O'),
    // Water and hydrogens must be ignored.
    pdbLine('HETATM', serial++, 'O', 'HOH', 'W', 9, [20.0, 20.0, 20.0], 'O'),
    pdbLine('HETATM', serial++, 'H1', 'NAG', 'C', 2, [4.9, 1.9, 3.9], 'H'),
  ];
  return `${helix.text}\n${het.join('\n')}\nEND\n`;
})();

const stub = (over: Partial<StructureArtifact> = {}): StructureArtifact =>
  ({
    kind: 'StructureArtifact',
    id: 'struct-1',
    label: 'Test backbone',
    sourceModelId: 'pdb-input',
    length: 10,
    sequence: 'ALEKSTVIAL',
    pdbText: PROTEIN_ONLY_PDB,
    ...over,
  }) as StructureArtifact;

const sampleOf = (sequence: string, chainIds: string[], chainLengths: number[]): ArtiosPottsSample => ({
  flat: { sequence, chainIds, chainLengths },
  score: {
    proteinEnergy: -12.5,
    totalEnergy: -13.25,
    pseudoLogLikelihood: -1.75,
    perResidueLogLikelihood: Float32Array.from(
      Array.from({ length: sequence.length }, (_, i) => -1 - i * 0.1),
    ),
    perResidueEnergy: Float32Array.from(Array.from({ length: sequence.length }, (_, i) => -i)),
  },
  sampleIndex: 0,
  seed: 42,
  acceptedSweeps: null,
});

// ---------------------------------------------------------------------------
// 1. Plugin manifest, ports, params, share code & planner role
// ---------------------------------------------------------------------------

describe('artios-potts plugin manifest & registration', () => {
  it('declares the ArtiosPotts identity, stage, runtime and weight budget', () => {
    const m = ARTIOS_POTTS_MANIFEST;
    expect(m.id).toBe('artios-potts');
    expect(m.displayName).toBe('ArtiosPotts');
    expect(m.shortName).toBe('ArtiosPotts');
    // `inverse-fold` is the Sequence Design stage of the protein-biopolymer domain.
    expect(m.stage).toBe('inverse-fold');
    expect(m.runtime).toBe('wgsl-custom');
    expect(m.licenseNote).toContain('Apache-2.0');
    expect(m.licenseNote).toContain('Ligo-Biosciences/artios');
    expect(m.description.length).toBeGreaterThan(80);
    expect(m.approxWeightsMB).toBe(ARTIOS_POTTS_WEIGHTS_MB.fp16);
    expect(m.precisionWeightsMB).toEqual({ int8: 2.7, fp16: 5.2, fp32: 10.4 });
    expect(m.estimatedVramMB).toBeGreaterThan(0);
  });

  it('wires a single StructureArtifact input to a SequenceSet plus an optional ResidueTrack', () => {
    expect(ARTIOS_POTTS_MANIFEST.inputs).toEqual({ structure: { type: 'StructureArtifact' } });
    expect(ARTIOS_POTTS_MANIFEST.outputs).toEqual({
      sequences: { type: 'SequenceSet' },
      track: { type: 'ResidueTrack', required: false },
    });
  });

  it('exposes design and score modes keyed by `mode`', () => {
    expect(ARTIOS_POTTS_MANIFEST.modeParamKey).toBe('mode');
    expect(ARTIOS_POTTS_MANIFEST.modes?.map((mode) => mode.id)).toEqual(['design', 'score']);
    for (const mode of ARTIOS_POTTS_MANIFEST.modes ?? []) {
      expect(mode.label.length).toBeGreaterThan(0);
      expect(mode.description.length).toBeGreaterThan(20);
      // Both modes read the structure's own coordinates; neither needs extra ports.
      expect(mode.requiredInputs ?? []).toEqual([]);
    }
  });

  it('declares every documented parameter with upstream-faithful defaults and bounds', () => {
    const schema = ARTIOS_POTTS_MANIFEST.paramSchema as AdapterParamField[];
    const byKey = new Map(schema.map((f) => [f.key, f]));
    expect([...byKey.keys()].sort()).toEqual(
      [
        'cfgScale',
        'complexityWeight',
        'fixedPositions',
        'ligandCutoff',
        'mode',
        'numSequences',
        'numSteps',
        'omitAAs',
        'precision',
        'sampler',
        'seed',
        'temperature',
        'temperatureInit',
      ].sort(),
    );

    const expectField = (
      key: string,
      kind: AdapterParamField['kind'],
      defaultValue: unknown,
      bounds?: [number, number],
    ) => {
      const f = byKey.get(key)!;
      expect(f, `missing field ${key}`).toBeDefined();
      expect(f.kind, key).toBe(kind);
      expect(f.defaultValue, key).toEqual(defaultValue);
      if (bounds) {
        expect(f.min, key).toBe(bounds[0]);
        expect(f.max, key).toBe(bounds[1]);
      }
      expect(f.description?.length ?? 0, key).toBeGreaterThan(10);
    };

    expectField('mode', 'select', 'design');
    expectField('numSequences', 'integer', 8, [1, 64]);
    expectField('sampler', 'select', 'gibbs');
    expectField('temperature', 'number', 0.1, [0.01, 2.0]);
    expectField('temperatureInit', 'number', 1.0, [0.01, 5.0]);
    expectField('numSteps', 'integer', 150, [10, 500]);
    expectField('cfgScale', 'number', 0, [0, 5.0]);
    expectField('ligandCutoff', 'number', 0, [0, 20]);
    expectField('omitAAs', 'aa-set', 'CM');
    expectField('complexityWeight', 'number', 0.6, [0, 5.0]);
    expectField('fixedPositions', 'residue-list', '');
    expectField('seed', 'integer', 42, [0, 2_147_483_647]);
    expectField('precision', 'select', 'fp16');

    expect(byKey.get('precision')!.options?.map((o) => o.value)).toEqual(['fp16', 'int8', 'fp32']);
    expect(byKey.get('sampler')!.options?.map((o) => o.value)).toEqual(['gibbs', 'dlmc']);
    expect(byKey.get('fixedPositions')!.studioRole).toBe('fixed');
    expect(byKey.get('fixedPositions')!.allowBareChain).toBe(true);
    // Sampling-only knobs stay hidden in score mode; precision and the cutoff apply to both.
    expect(byKey.get('numSequences')!.modes).toEqual(['design']);
    expect(byKey.get('precision')!.modes).toBeUndefined();
    expect(byKey.get('ligandCutoff')!.modes).toBeUndefined();
    expect(byKey.get('mode')!.modes).toBeUndefined();
  });

  it('round-trips schema defaults through declarative validation and rejects bad values', () => {
    const defaults = getDefaultParamsFromSchema(ARTIOS_POTTS_MANIFEST);
    expect(defaults).toMatchObject({
      mode: 'design',
      numSequences: 8,
      sampler: 'gibbs',
      temperature: 0.1,
      numSteps: 150,
      cfgScale: 0,
      ligandCutoff: 0,
      omitAAs: 'CM',
      complexityWeight: 0.6,
      precision: 'fp16',
      seed: 42,
    });
    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, defaults)).toEqual([]);
    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, mode: 'score' })).toEqual([]);

    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, numSequences: 65 })).toHaveLength(1);
    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, temperature: 0 })).toHaveLength(1);
    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, numSteps: 7 })).toHaveLength(1);
    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, ligandCutoff: 25 })).toHaveLength(1);
    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, precision: 'bf16' })).toHaveLength(1);
    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, sampler: 'metropolis' })).toHaveLength(1);
    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, mode: 'refine' })).toHaveLength(1);
    expect(validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, omitAAs: 'CZ' })).toHaveLength(1);
    // Sampling knobs are not validated in score mode, where they are hidden.
    expect(
      validateParamsWithSchema(ARTIOS_POTTS_MANIFEST, { ...defaults, mode: 'score', numSequences: 999 }),
    ).toEqual([]);
  });

  it('registers as plugin "artios-potts" with share code "ap" and both planner roles', () => {
    const registry = createDefaultPluginRegistry({ freeze: true });
    expect(registry.hasPlugin('artios-potts')).toBe(true);
    const plugin = registry.getPlugin('artios-potts')!;
    expect(plugin.share.shortCode).toBe('ap');
    expect(plugin.domainId).toBe('protein-biopolymer');
    expect(plugin.execution.runtime).toBe('wgsl-custom');
    expect(plugin.execution.supportsCooperativeAbort).toBe(true);
    expect(plugin.execution.fallback).toEqual({
      kind: 'cpu-js',
      moduleUrl: '/vendor/artios-potts/cpu/index.js',
      degradedWarning: expect.stringContaining('CPU JS fallback'),
    });
    expect(plugin.cachePolicy.usesSharedOpfsStore).toBe(true);
    expect(plugin.catalog.categoryId).toBe('sequence-design');
    expect(registry.getPluginPlannerRoles(plugin)).toEqual(['sequence-designer', 'scorer']);
    expect(plugin.planner?.capabilities).toContain('ligand-aware-sequence-design');
    expect(plugin.planner?.capabilities).toContain('sequence-scoring');
    // "ap" must not collide with any of the 29 pre-existing short codes.
    expect(registry.listPlugins().filter((p) => p.share.shortCode === 'ap')).toHaveLength(1);
  });

  it('estimates preflight download and output cardinality per mode', () => {
    const ctx = (params: Record<string, unknown>) => ({
      params,
      upstreamProfiles: { structure: { itemCount: 3 } as any },
      connectedInputs: new Set(['structure']),
    });
    const design = artiosPottsPlugin.preflight(ctx({ mode: 'design', numSequences: 4, precision: 'fp16' }));
    expect(design.estimatedDownloadMB).toBe(5.2);
    expect(design.workItems).toBe(12);
    expect(design.outputCardinality).toEqual({ sequences: 12, track: 1 });
    expect(design.stageBreakdown).toEqual({ 'inverse-fold': 12 });

    const score = artiosPottsPlugin.preflight(ctx({ mode: 'score', numSequences: 4, precision: 'int8' }));
    expect(score.estimatedDownloadMB).toBe(2.7);
    expect(score.workItems).toBe(3);
  });

  it('summarizes design and score nodes for the canvas', () => {
    expect(formatArtiosPottsSummary({})).toBe('8 seqs · T=0.1 · Gibbs ×150 · FP16');
    expect(
      formatArtiosPottsSummary({ numSequences: 1, sampler: 'dlmc', numSteps: 300, ligandCutoff: 8, cfgScale: 2, precision: 'int8' }),
    ).toBe('1 seq · T=0.1 · DLMC ×300 · ≤8 Å of ligand · CFG 2 · INT8');
    expect(formatArtiosPottsSummary({ mode: 'score' })).toBe('Score native · Energy + PLL · FP16');
    expect(formatArtiosPottsSummary({ mode: 'score', ligandCutoff: 6 })).toBe(
      'Score native · Energy + PLL · 6 Å pocket · FP16',
    );
    // The plugin's own display hook must agree with the exported formatter.
    expect(artiosPottsPlugin.display.formatCanvasSummary({ mode: 'score' })).toBe(
      formatArtiosPottsSummary({ mode: 'score' }),
    );
  });

  it('stays byte-identical to the shared ADAPTER_REGISTRY entry once wired', () => {
    const registered = (ADAPTER_REGISTRY as Record<string, unknown>)['artios-potts'];
    if (!registered) {
      // Registry wiring is applied by the coordinator; the plugin runs off its local manifest until then.
      expect(artiosPottsPlugin.manifest.id).toBe('artios-potts');
      return;
    }
    expect(JSON.parse(JSON.stringify(registered))).toEqual(
      JSON.parse(JSON.stringify(ARTIOS_POTTS_MANIFEST)),
    );
  });
});

// ---------------------------------------------------------------------------
// 2. Param normalization
// ---------------------------------------------------------------------------

describe('artiosPottsOptionsFromParams', () => {
  it('applies upstream ArtiosPotts defaults for an empty param object', () => {
    expect(artiosPottsOptionsFromParams({})).toEqual({
      mode: 'design',
      numSequences: 8,
      sampler: 'gibbs',
      temperature: 0.1,
      temperatureInit: 1.0,
      numSteps: 150,
      cfgScale: 0,
      ligandCutoff: 0,
      omitAAs: 'CM',
      complexityWeight: 0.6,
      fixedPositions: '',
      seed: 42,
      precision: 'fp16',
    });
  });

  it('clamps out-of-range numbers and rounds integers instead of failing the run', () => {
    const o = artiosPottsOptionsFromParams({
      numSequences: 500,
      numSteps: 2,
      temperature: 99,
      temperatureInit: -3,
      cfgScale: 12,
      ligandCutoff: -5,
      complexityWeight: 50,
      seed: 7.6,
    });
    expect(o.numSequences).toBe(64);
    expect(o.numSteps).toBe(10);
    expect(o.temperature).toBe(2.0);
    expect(o.temperatureInit).toBe(0.01);
    expect(o.cfgScale).toBe(5.0);
    expect(o.ligandCutoff).toBe(0);
    expect(o.complexityWeight).toBe(5.0);
    expect(o.seed).toBe(8);
  });

  it('falls back to upstream defaults for unknown enums and non-numeric input', () => {
    const o = artiosPottsOptionsFromParams({
      mode: 'nonsense',
      sampler: 'metropolis',
      precision: 'bf16',
      numSequences: 'many',
      temperature: null,
    });
    expect(o.mode).toBe('design');
    expect(o.sampler).toBe('gibbs');
    expect(o.precision).toBe('fp16');
    expect(o.numSequences).toBe(8);
    expect(o.temperature).toBe(0.1);
  });

  it('accepts the score mode, dlmc sampler and both non-default precisions', () => {
    expect(artiosPottsOptionsFromParams({ mode: 'SCORE' }).mode).toBe('score');
    expect(artiosPottsOptionsFromParams({ sampler: 'DLMC' }).sampler).toBe('dlmc');
    expect(artiosPottsOptionsFromParams({ precision: 'INT8' }).precision).toBe('int8');
    expect(artiosPottsOptionsFromParams({ precision: 'fp32' }).precision).toBe('fp32');
  });

  it('canonicalizes the omit-amino-acid set', () => {
    expect(sanitizeOmitAAs('c, m')).toBe('CM');
    expect(sanitizeOmitAAs('CCMM')).toBe('CM');
    expect(sanitizeOmitAAs('C X M z')).toBe('CM');
    expect(sanitizeOmitAAs('')).toBe('');
    expect(artiosPottsOptionsFromParams({ omitAAs: '' }).omitAAs).toBe('');
  });
});

describe('resolveArtiosPottsAlphabet', () => {
  it('maps the upstream Cys/Met exclusion onto the build_alphabet_mask flags', () => {
    const a = resolveArtiosPottsAlphabet('CM');
    expect(a.excludeCys).toBe(true);
    expect(a.excludeMet).toBe(true);
    expect(a.energyBiasCodes).toEqual([]);
    expect(a.designable).toHaveLength(18);
    expect(a.designable).not.toContain('C');
    expect(a.designable).not.toContain('M');
  });

  it('routes any other omitted letter through the energy_bias channel', () => {
    const a = resolveArtiosPottsAlphabet('CWH');
    expect(a.excludeCys).toBe(true);
    expect(a.excludeMet).toBe(false);
    expect(a.energyBiasCodes).toEqual(['H', 'W']);
    expect(a.designable).toHaveLength(17);
  });

  it('keeps the full alphabet when nothing is omitted', () => {
    const a = resolveArtiosPottsAlphabet('');
    expect(a).toMatchObject({ excludeCys: false, excludeMet: false, energyBiasCodes: [] });
    expect(a.designable).toHaveLength(20);
  });

  it('refuses an omit set that leaves no amino acid', () => {
    expect(() => resolveArtiosPottsAlphabet('ACDEFGHIKLMNPQRSTVWY')).toThrow(/all 20 amino acids/);
  });
});

// ---------------------------------------------------------------------------
// 3. PDB / mmCIF -> featurizer spec
// ---------------------------------------------------------------------------

describe('buildArtiosPottsInput', () => {
  it('compiles a protein-only PDB into one protein chain with backbone atoms', () => {
    const input = buildArtiosPottsInput(PROTEIN_ONLY_PDB, 'helix10');
    expect(input.spec.name).toBe('helix10');
    expect(input.spec.chains).toHaveLength(1);
    const chain = input.spec.chains[0];
    expect(chain).toMatchObject({ id: 'A', kind: 'protein' });
    expect(chain.residues).toHaveLength(10);
    expect(chain.residues[0]).toMatchObject({ resName: 'ALA', resSeq: 1 });
    for (const r of chain.residues as any[]) {
      for (const atom of ['N', 'CA', 'C', 'O']) expect(r[atom]).toHaveLength(3);
    }
    expect(input.proteinResidues).toHaveLength(10);
    expect(input.proteinResidues[0]).toEqual({
      chain: 'A',
      number: 1,
      insertionCode: '',
      resName: 'ALA',
    });
    expect(input.diagnostics).toMatchObject({
      proteinResidueCount: 10,
      contextAtomCount: 0,
      contextGroupNames: [],
      nucleicAcidChains: [],
      skippedNoCa: 0,
      warnings: [],
    });
  });

  it('routes HETATM ligands, cofactors and metals into per-atom context tokens, dropping water and H', () => {
    const input = buildArtiosPottsInput(HOLO_PDB);
    const protein = input.spec.chains.filter((c) => c.kind === 'protein');
    const ligand = input.spec.chains.filter((c) => c.kind === 'ligand');
    expect(protein).toHaveLength(1);
    // One spec chain per context group (matching the featurizer's per-group asym IDs).
    expect(ligand).toHaveLength(2);
    expect(input.diagnostics.contextAtomCount).toBe(3);
    expect(input.diagnostics.contextGroupNames.sort()).toEqual(['NAG', 'ZN']);
    const atomNames = ligand.flatMap((c) => (c.residues as any[]).flatMap((r) => r.atoms.map((a: any) => a.name)));
    expect(atomNames.sort()).toEqual(['C1', 'O1', 'ZN']);
    expect(atomNames).not.toContain('H1');
    expect(input.diagnostics.contextGroupNames).not.toContain('HOH');
    expect(input.diagnostics.nucleicAcidChains).toEqual([]);
  });

  it('keeps insertion-coded residues distinct instead of collapsing them onto one resSeq', () => {
    const helix = helixPdb(3, 'H', { startResSeq: 100 });
    // Re-stamp residues 100/101/102 as 100 / 100A / 100B.
    const text = helix.text
      .split('\n')
      .map((line, idx) => {
        const resSeq = Number.parseInt(line.slice(22, 26), 10);
        if (resSeq === 100) return line;
        const ins = resSeq === 101 ? 'A' : 'B';
        return `${line.slice(0, 22)}${String(100).padStart(4)}${ins}${line.slice(27)}`;
      })
      .join('\n');
    const input = buildArtiosPottsInput(text);
    expect(input.proteinResidues).toHaveLength(3);
    expect(input.proteinResidues.map((r) => `${r.number}${r.insertionCode}`)).toEqual([
      '100',
      '100A',
      '100B',
    ]);
    expect(residueIdsOf(input.proteinResidues)).toEqual(['H100', 'H100A', 'H100B']);
  });

  it('conditions on nucleic-acid chains as context atoms and says so', () => {
    const dna = [
      pdbLine('ATOM', 500, "P", ' DA', 'D', 1, [6.0, 0.0, 1.0], 'P'),
      pdbLine('ATOM', 501, "C1'", ' DA', 'D', 1, [6.8, 0.6, 1.8], 'C'),
      pdbLine('ATOM', 502, "N1", ' DT', 'D', 2, [7.2, 1.4, 3.2], 'N'),
    ].join('\n');
    const input = buildArtiosPottsInput(`${helixPdb(6).text}\n${dna}\nEND\n`);
    expect(input.diagnostics.nucleicAcidChains).toEqual(['D']);
    expect(input.diagnostics.contextAtomCount).toBe(3);
    expect(input.diagnostics.contextGroupNames.sort()).toEqual(['DA', 'DT']);
    expect(input.diagnostics.warnings.join(' ')).toMatch(/DNA\/RNA chain\(s\) D/);
    // Protein residues are unaffected.
    expect(input.proteinResidues).toHaveLength(6);
  });

  it('keeps altLoc A, drops other conformers, and skips residues without a Cα', () => {
    const base = helixPdb(3);
    const extra = [
      pdbLine('ATOM', 900, 'CA', 'SER', 'A', 2, [9.9, 9.9, 9.9], 'C', ' ', 'B'),
      // A residue with only N/C/O and no CA cannot be tokenized.
      pdbLine('ATOM', 901, 'N', 'GLY', 'A', 50, [0, 0, 30], 'N'),
      pdbLine('ATOM', 902, 'C', 'GLY', 'A', 50, [0, 0, 31], 'C'),
    ].join('\n');
    const input = buildArtiosPottsInput(`${base.text}\n${extra}\n`);
    expect(input.proteinResidues).toHaveLength(3);
    expect(input.diagnostics.skippedNoCa).toBe(1);
    expect(input.diagnostics.warnings.join(' ')).toMatch(/without a Cα were dropped/);
    // altLoc B was ignored, so residue 2 keeps its altLoc-free Cα.
    const ca = (input.spec.chains[0].residues as any[])[1].CA;
    expect(ca[0]).not.toBeCloseTo(9.9, 5);
  });

  it('preserves multi-chain order and refuses a structure with no protein', () => {
    const a = helixPdb(4, 'A');
    const b = helixPdb(3, 'B', { serial: a.nextSerial, offset: 25 });
    const input = buildArtiosPottsInput(`${a.text}\n${b.text}\n`);
    expect(input.spec.chains.map((c) => c.id)).toEqual(['A', 'B']);
    expect(input.proteinResidues.map((r) => r.chain)).toEqual(['A', 'A', 'A', 'A', 'B', 'B', 'B']);

    expect(() => buildArtiosPottsInput('HETATM    1 ZN    ZN B   1       0.000   0.000   0.000  1.00  0.00          ZN  \n')).toThrow(
      /no protein residue/,
    );
  });

  it('reads PDB text directly and converts mmCIF-only structures', () => {
    expect(inputPdbText(stub())).toBe(PROTEIN_ONLY_PDB);
    const cif = [
      'data_test',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.auth_seq_id',
      '_atom_site.auth_asym_id',
      'ATOM 1 N N ALA A 1 0.000 0.000 0.000 1 A',
      'ATOM 2 C CA ALA A 1 1.458 0.000 0.000 1 A',
      'ATOM 3 C C ALA A 1 2.009 1.420 0.000 1 A',
      'ATOM 4 O O ALA A 1 1.251 2.390 0.000 1 A',
      'ATOM 5 N N GLY A 2 3.332 1.541 0.000 2 A',
      'ATOM 6 C CA GLY A 2 4.013 2.828 0.000 2 A',
      'ATOM 7 C C GLY A 2 5.520 2.638 0.000 2 A',
      'ATOM 8 O O GLY A 2 6.012 1.511 0.000 2 A',
      '#',
    ].join('\n');
    const converted = inputPdbText(stub({ pdbText: undefined, cifText: cif }));
    expect(converted).toMatch(/^ATOM/m);
    const input = buildArtiosPottsInput(converted);
    expect(input.proteinResidues.map((r) => r.resName)).toEqual(['ALA', 'GLY']);

    expect(() => inputPdbText(stub({ pdbText: undefined, cifText: undefined }))).toThrow(
      /no coordinate text/,
    );
  });
});

// ---------------------------------------------------------------------------
// 4. Constraints & masks
// ---------------------------------------------------------------------------

describe('fixed positions, inherited locks & ligand cutoff', () => {
  const twoChain = (() => {
    const a = helixPdb(4, 'A');
    const b = helixPdb(3, 'B', { serial: a.nextSerial, offset: 25 });
    return buildArtiosPottsInput(`${a.text}\n${b.text}\n`);
  })();

  it('resolves explicit residues, ranges and bare-chain selectors', () => {
    const { fixedProteinIndices, lockedChains } = resolveArtiosPottsConstraints(
      twoChain.proteinResidues,
      'A2-3',
    );
    expect([...fixedProteinIndices].sort((x, y) => x - y)).toEqual([1, 2]);
    expect(lockedChains).toEqual([]);

    const whole = resolveArtiosPottsConstraints(twoChain.proteinResidues, 'B');
    expect([...whole.fixedProteinIndices].sort((x, y) => x - y)).toEqual([4, 5, 6]);
    expect(whole.lockedChains).toEqual(['B']);
  });

  it('inherits locked chains and 1-based motif residue indices from the upstream generator', () => {
    const inherited = resolveArtiosPottsConstraints(twoChain.proteinResidues, '', {
      lockedChains: ['B'],
      fixedResidueIndices: [1, 4],
    });
    expect([...inherited.fixedProteinIndices].sort((x, y) => x - y)).toEqual([0, 3, 4, 5, 6]);
    expect(inherited.lockedChains).toEqual(['B']);
  });

  it('ignores selectors for chains and residues that are not present', () => {
    const { fixedProteinIndices, lockedChains } = resolveArtiosPottsConstraints(
      twoChain.proteinResidues,
      'Z',
      { lockedChains: ['Q'] },
    );
    expect(fixedProteinIndices.size).toBe(0);
    expect(lockedChains).toEqual([]);
  });

  it('builds a token-level fixed mask that skips context tokens', () => {
    const context = shared.featurize(buildArtiosPottsInput(HOLO_PDB).spec);
    expect(context.nToken).toBe(13); // 10 protein residues + 3 context atoms
    const fixed = buildFixedTokenMask(context, new Set([0, 9]));
    expect(fixed).toHaveLength(13);
    expect([...fixed]).toEqual([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
    expect(countDesignablePositions(context, fixed)).toBe(8);
    expect(countDesignablePositions(context, new Uint8Array(13))).toBe(10);
  });

  it('fixes every residue outside the ligand cutoff and leaves the pocket designable', () => {
    const input = buildArtiosPottsInput(HOLO_PDB);
    const context = shared.featurize(input.spec);
    const near = shared.buildLigandProximityMask(context, 5.0);
    const far = shared.buildLigandProximityMask(context, 0.5);

    const pocket = buildFixedTokenMask(context, new Set(), near);
    const pocketCount = countDesignablePositions(context, pocket);
    expect(pocketCount).toBeGreaterThan(0);
    expect(pocketCount).toBeLessThan(10);
    // The residues the mask keeps designable are exactly the ones within the cutoff.
    for (let i = 0; i < context.nToken; i++) {
      if (!context.isProteinToken[i]) continue;
      expect(pocket[i] === 0).toBe(Boolean(near[i]));
    }

    // A sub-Ångström cutoff reaches nothing, so every residue is fixed.
    expect(countDesignablePositions(context, buildFixedTokenMask(context, new Set(), far))).toBe(0);
    // A cutoff of 0 is expressed as "no proximity mask", which designs everything.
    expect(countDesignablePositions(context, buildFixedTokenMask(context, new Set(), null))).toBe(10);
  });

  it('returns an all-zero proximity mask for apo inputs so the cutoff can be skipped', () => {
    const context = shared.featurize(buildArtiosPottsInput(PROTEIN_ONLY_PDB).spec);
    const mask = shared.buildLigandProximityMask(context, 10.0);
    expect([...mask].every((v: number) => v === 0)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 5. Output artifact assembly
// ---------------------------------------------------------------------------

describe('SequenceSet, candidate & ResidueTrack assembly', () => {
  it('flattens per-chain sampler output in spec chain order', () => {
    const a = helixPdb(4, 'A');
    const b = helixPdb(3, 'B', { serial: a.nextSerial, offset: 25 });
    const input = buildArtiosPottsInput(`${a.text}\n${b.text}\n`);
    const flat = flattenChainSequences(input.proteinResidues, { A: 'WXYZ', B: 'PQR' });
    expect(flat).toEqual({ sequence: 'WXYZPQR', chainIds: ['A', 'B'], chainLengths: [4, 3] });
  });

  it('rejects a chain whose sampled sequence length disagrees with the structure', () => {
    const input = buildArtiosPottsInput(PROTEIN_ONLY_PDB);
    expect(() => flattenChainSequences(input.proteinResidues, { A: 'SHORT' })).toThrow(
      /chain A has 10 residues but the sampler returned 5/,
    );
  });

  it('builds a candidate whose score is the negated PLL so lower sorts better', () => {
    const options = artiosPottsOptionsFromParams({ temperature: 0.2 });
    const candidate = buildArtiosPottsCandidate(
      sampleOf('ALEKSTVIAL', ['A'], [10]),
      stub(),
      options,
    );
    expect(candidate).toMatchObject({
      id: 'artios-potts-1',
      sourceStructureId: 'struct-1',
      sampleIndex: 0,
      sequence: 'ALEKSTVIAL',
      sourceModelId: 'artios-potts',
      temperature: 0.2,
      seed: 42,
      chainIds: ['A'],
      chainLengths: [10],
    });
    expect(candidate.score).toBeCloseTo(1.75, 6);
    expect(candidate.globalScore).toBeCloseTo(-1.25, 6);
  });

  it('carries design lineage through for multi-backbone collections', () => {
    const candidate = buildArtiosPottsCandidate(
      sampleOf('ALEK', ['A'], [4]),
      stub(),
      artiosPottsOptionsFromParams({}),
      'd2-artios-potts-d',
      { designKey: 'design-2', designIndex: 1 },
    );
    expect(candidate.id).toBe('d2-artios-potts-d-1');
    expect(candidate).toMatchObject({ designKey: 'design-2', designIndex: 1 });
  });

  it('emits a SequenceSet with lineage, locks and chain roles', () => {
    const parent = stub({ chainRoles: { A: 'binder', B: 'target' } });
    const options = artiosPottsOptionsFromParams({ seed: 7 });
    const candidates = [
      buildArtiosPottsCandidate(sampleOf('ALEKSTVIAL', ['A'], [10]), parent, options),
    ];
    const set = buildArtiosPottsSequenceSet(parent, candidates, options, ['B'], 'deadbeef');
    expect(set.kind).toBe('SequenceSet');
    expect(set.id).toBe('artios-potts-seq-struct-1-7-deadbeef');
    expect(set).toMatchObject({
      parentStructureId: 'struct-1',
      sourceStructureId: 'struct-1',
      sourceModelId: 'artios-potts',
      lockedChains: ['B'],
      chainRoles: { A: 'binder', B: 'target' },
    });
    expect(set.candidates).toHaveLength(1);

    const noLocks = buildArtiosPottsSequenceSet(stub(), candidates, options, [], 'cafe');
    expect(noLocks.lockedChains).toBeUndefined();
    expect(noLocks.chainRoles).toBeUndefined();
  });

  it('labels the ResidueTrack differently for design and score runs', () => {
    const input = buildArtiosPottsInput(PROTEIN_ONLY_PDB);
    const sample = sampleOf('ALEKSTVIAL', ['A'], [10]);
    const design = buildArtiosPottsTrack(
      stub(),
      input.proteinResidues,
      sample,
      artiosPottsOptionsFromParams({}),
      'hash1',
    );
    expect(design).toMatchObject({
      kind: 'ResidueTrack',
      id: 'artios-potts-track-design-fp16-hash1',
      sourceModelId: 'artios-potts',
      parentStructureId: 'struct-1',
      quantity: 'potts-pll',
      unit: 'log p',
      residues: 'ALEKSTVIAL',
    });
    expect(design.label).toMatch(/best design/);
    expect(design.residueIds).toEqual(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']);
    expect(design.values).toHaveLength(10);
    expect(design.summary?.mean).toBeCloseTo(-1.45, 5);
    expect(design.summary?.max).toBeCloseTo(-1.0, 5);
    expect(design.summary?.min).toBeCloseTo(-1.9, 5);

    const scored = buildArtiosPottsTrack(
      stub(),
      input.proteinResidues,
      sample,
      artiosPottsOptionsFromParams({ mode: 'score', precision: 'int8' }),
      'hash2',
    );
    expect(scored.id).toBe('artios-potts-track-score-int8-hash2');
    expect(scored.label).toMatch(/native/);
  });

  it('restricts per-residue values to protein tokens, in token order', () => {
    const context = shared.featurize(buildArtiosPottsInput(HOLO_PDB).spec);
    const all = Float32Array.from({ length: context.nToken }, (_, i) => i);
    expect([...proteinTokenValues(context, all)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});

describe('structure-collection fan-out', () => {
  const member = (id: string, designKey: string, designIndex: number): StructureArtifact =>
    stub({ id, designKey, designIndex }) as StructureArtifact;

  it('treats a single structure as one job', () => {
    const one = stub();
    expect(structureItems(one)).toEqual([one]);
    expect(planArtiosPottsJobs(one)).toEqual([{ parent: one, designIndex: 0 }]);
  });

  it('emits one job per collection item, preserving design keys and ordinals', () => {
    const items = [member('s1', 'design-1', 0), member('s2', 'design-2', 1)];
    const collection = stub({
      id: 'batch',
      collection: {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id: 'col-1',
        collectionKind: 'independent-candidates',
        requestedCount: 2,
        complete: true,
        items,
        keys: ['design-1', 'design-2'],
      },
    } as Partial<StructureArtifact>);
    expect(structureItems(collection)).toHaveLength(2);
    expect(planArtiosPottsJobs(collection)).toEqual([
      { parent: items[0], designKey: 'design-1', designIndex: 0 },
      { parent: items[1], designKey: 'design-2', designIndex: 1 },
    ]);
  });

  it('falls back to collection keys when items carry no designKey', () => {
    const items = [stub({ id: 's1' }), stub({ id: 's2' })];
    const collection = stub({
      collection: {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id: 'col-2',
        collectionKind: 'independent-candidates',
        requestedCount: 2,
        complete: true,
        items,
        keys: ['k1', 'k2'],
      },
    } as Partial<StructureArtifact>);
    expect(planArtiosPottsJobs(collection).map((j) => j.designKey)).toEqual(['k1', 'k2']);
  });
});

// ---------------------------------------------------------------------------
// 6. End-to-end CPU smoke test over the vendored runtime
// ---------------------------------------------------------------------------

/** Minimal little-endian safetensors writer (F32 only) for the synthetic bundle. */
function writeSafetensors(tensors: Map<string, { shape: number[]; data: Float32Array }>): Uint8Array {
  const header: Record<string, unknown> = {};
  let offset = 0;
  for (const [name, t] of tensors) {
    const bytes = t.data.byteLength;
    header[name] = { dtype: 'F32', shape: t.shape, data_offsets: [offset, offset + bytes] };
    offset += bytes;
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  // safetensors requires the data block to start 8-byte aligned.
  const pad = (8 - (headerBytes.byteLength % 8)) % 8;
  const headerLength = headerBytes.byteLength + pad;
  const out = new Uint8Array(8 + headerLength + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerLength), true);
  out.set(headerBytes, 8);
  out.fill(0x20, 8 + headerBytes.byteLength, 8 + headerLength);
  let cursor = 8 + headerLength;
  for (const [, t] of tensors) {
    out.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), cursor);
    cursor += t.data.byteLength;
  }
  return out;
}

const CONFIG = {
  num_residue_types: 32,
  node_dim: 128,
  edge_dim: 128,
  num_encoder_layers: 3,
  num_decoder_layers: 3,
  num_neighbours: 32,
  positional_encoding_dim: 16,
  positional_encoding_max_relative_feats: 32,
  num_rbf_kernels: 16,
  rbf_min_distance: 2.0,
  rbf_max_distance: 22.0,
  relpos_r_max: 32,
  relpos_s_max: 2,
  num_backbone_atoms: 5,
  layer_norm_eps: 1e-5,
  mask_token_index: 32,
  potts: { parameterisation: 'factor', num_factors: 32, symmetric_J: true, zero_sum_gauge: true },
};

/** Deterministic weights with the checkpoint's exact tensor names and shapes. */
function syntheticBundle(seed = 11): { manifest: unknown; file: Uint8Array } {
  let s = seed >>> 0;
  const rand = (): number => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
  const tensors = new Map<string, { shape: number[]; data: Float32Array }>();
  for (const spec of weightsMod.weightSpec(CONFIG) as Array<{ name: string; shape: number[] }>) {
    const shape = spec.shape.length === 0 ? [1] : spec.shape;
    const count = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(count);
    if (spec.name === 'mpnn.featurizer.rbf.centers') {
      for (let i = 0; i < count; i++) data[i] = 2 + (20 * i) / (count - 1);
    } else if (spec.name === 'mpnn.featurizer.rbf.gamma') {
      data[0] = 1 / ((20 / (CONFIG.num_rbf_kernels - 1)) ** 2);
    } else if (spec.name === 'potts.log_scale') {
      data[0] = 0;
    } else if (/norm\d?\.weight$|_norm\.weight$/.test(spec.name)) {
      for (let i = 0; i < count; i++) data[i] = 1 + 0.1 * rand();
    } else if (/\.bias$/.test(spec.name)) {
      for (let i = 0; i < count; i++) data[i] = 0.05 * rand();
    } else {
      const fanIn = shape.length === 2 ? shape[1] : 1;
      for (let i = 0; i < count; i++) data[i] = (2 * rand() * Math.sqrt(3)) / Math.sqrt(fanIn);
    }
    tensors.set(spec.name, { shape, data });
  }
  const file = writeSafetensors(tensors);
  return {
    manifest: {
      format: 'artios-potts-weights/1',
      model: 'ArtiosPotts',
      config: CONFIG,
      precisions: {
        fp32: {
          quantization: { scheme: 'none' },
          files: [{ path: 'fp32/model.safetensors', bytes: file.byteLength, sha256: '' }],
        },
      },
    },
    file,
  };
}

describe('end-to-end CPU smoke test on the vendored ArtiosPotts runtime', () => {
  const bundle = syntheticBundle();
  const read = async (url: string): Promise<Uint8Array> => {
    if (url.endsWith('manifest.json')) {
      return new TextEncoder().encode(JSON.stringify(bundle.manifest));
    }
    if (url.endsWith('fp32/model.safetensors')) return bundle.file;
    throw new Error(`unexpected bundle request ${url}`);
  };

  it('exports the symbols the worker imports from the vendored modules', () => {
    for (const name of [
      'featurize',
      'withoutLigand',
      'buildLigandProximityMask',
      'sequencesByChain',
      'scoreSequence',
      'prepareSampleFromParams',
      'sampleFromParams',
      'loadWeightBundle',
    ]) {
      expect(typeof shared[name], name).toBe('function');
    }
    expect(shared.ALPHABET_BIAS).toBe(1e4);
    expect(typeof cpuMod.createArtiosPottsCPU).toBe('function');
  });

  it('featurizes a holo pocket into the exact tensors the Potts head consumes', () => {
    const input = buildArtiosPottsInput(HOLO_PDB);
    const context = shared.featurize(input.spec);
    expect(context.nToken).toBe(13);
    expect(context.numResidueTypes).toBe(32);
    // 10 residues x 4 backbone atoms + 3 context atoms.
    expect(context.nAtom).toBe(43);
    expect([...context.isProteinToken].filter(Boolean)).toHaveLength(10);
    expect([...context.isLigandToken].filter(Boolean)).toHaveLength(3);
    // The virtual CB is derived from N/CA/C, so all 5 backbone slots are masked in.
    expect([...context.backboneMask.slice(0, 5)]).toEqual([1, 1, 1, 1, 1]);
    const native = shared.sequencesByChain(context);
    expect(native.A).toBe('ALEKSTVIAL');
    // Ligand-free context drops context tokens from the graph without changing the token count.
    const apo = shared.withoutLigand(context);
    expect(apo.nToken).toBe(13);
    expect([...apo.tokenMask].filter(Boolean)).toHaveLength(10);
  });

  it('scores and designs end to end: Potts head -> PLL -> Gibbs & DLMC MCMC', async () => {
    const input = buildArtiosPottsInput(HOLO_PDB);
    const engine = await cpuMod.createArtiosPottsCPU({ baseUrl: '/weights', precision: 'fp32', read });
    const context = engine.featurize(input.spec);
    const params = engine.computeParams(context);

    // Shapes match the factorized GraphPotts head: h [N, R], J [N, k, R, R].
    const R = 32;
    const k = Math.min(32, context.nToken);
    expect(params.nToken).toBe(13);
    expect(params.k).toBe(k);
    expect(params.numResidueTypes).toBe(R);
    expect(params.h).toHaveLength(13 * R);
    expect(params.J).toHaveLength(13 * k * R * R);
    expect([...params.h].every(Number.isFinite)).toBe(true);
    expect([...params.h].some((v) => v !== 0)).toBe(true);

    // Zero-sum gauge: every h row sums to ~0 for valid tokens.
    for (let i = 0; i < params.nToken; i++) {
      let sum = 0;
      for (let a = 0; a < R; a++) sum += params.h[i * R + a];
      expect(Math.abs(sum)).toBeLessThan(1e-2);
    }

    // --- score mode -------------------------------------------------------
    const native = Int32Array.from(context.restypeIndex);
    const scored = shared.scoreSequence(params, native, { isProteinToken: context.isProteinToken });
    expect(Number.isFinite(scored.proteinEnergy)).toBe(true);
    expect(Number.isFinite(scored.totalEnergy)).toBe(true);
    expect(Number.isFinite(scored.pseudoLogLikelihood)).toBe(true);
    // A pseudo-log-likelihood is a mean log-probability, so it must be <= 0.
    expect(scored.pseudoLogLikelihood).toBeLessThanOrEqual(0);
    const perResidue = proteinTokenValues(context, scored.logLikelihood);
    expect(perResidue).toHaveLength(10);
    expect([...perResidue].every((v) => Number.isFinite(v) && v <= 0)).toBe(true);
    const flatNative = flattenChainSequences(
      input.proteinResidues,
      shared.sequencesByChain(context, native),
    );
    expect(flatNative.sequence).toBe('ALEKSTVIAL');

    // --- design mode (chromatic Gibbs) ------------------------------------
    const alphabet = resolveArtiosPottsAlphabet('CM');
    const baseOptions = {
      context,
      nSweeps: 12,
      temperature: 0.1,
      temperatureInit: 1.0,
      annealingFraction: 0.8,
      excludeCys: alphabet.excludeCys,
      excludeMet: alphabet.excludeMet,
      complexityPenaltyWeight: 0.6,
      complexityWindowSize: 30,
      complexityEntropyMin: 2.32,
    };
    const gibbs = shared.sampleFromParams(params, { ...baseOptions, proposal: 'gibbs', seed: 1 });
    expect(gibbs.sequence).toHaveLength(13);
    expect(gibbs.acceptedSweeps).toBeNull();
    const designed = flattenChainSequences(
      input.proteinResidues,
      shared.sequencesByChain(context, gibbs.sequence),
    );
    expect(designed.sequence).toHaveLength(10);
    expect(designed.sequence).toMatch(/^[ARNDCQEGHILKMFPSTWYVX]{10}$/);
    // Cys and Met are excluded by the upstream default alphabet mask.
    expect(designed.sequence).not.toMatch(/[CM]/);
    // Context atoms keep their own identity.
    for (let i = 0; i < context.nToken; i++) {
      if (!context.isLigandToken[i]) continue;
      expect(gibbs.sequence[i]).toBe(context.restypeIndex[i]);
    }
    // The same seed reproduces the same design; a different seed is free to differ.
    const repeat = shared.sampleFromParams(params, { ...baseOptions, proposal: 'gibbs', seed: 1 });
    expect([...repeat.sequence]).toEqual([...gibbs.sequence]);

    // --- fixed positions are honoured ------------------------------------
    const fixedMask = buildFixedTokenMask(context, new Set([0, 1, 2]));
    const constrained = shared.sampleFromParams(params, {
      ...baseOptions,
      proposal: 'gibbs',
      seed: 3,
      fixedMask,
    });
    const constrainedSeq = flattenChainSequences(
      input.proteinResidues,
      shared.sequencesByChain(context, constrained.sequence),
    ).sequence;
    expect(constrainedSeq.slice(0, 3)).toBe('ALE');
    expect([...constrained.mutableMask.slice(0, 3)]).toEqual([0, 0, 0]);

    // --- ligand cutoff restricts design to the pocket ---------------------
    const near = shared.buildLigandProximityMask(context, 5.0);
    const pocketMask = buildFixedTokenMask(context, new Set(), near);
    const pocketDesign = shared.sampleFromParams(params, {
      ...baseOptions,
      proposal: 'gibbs',
      seed: 4,
      fixedMask: pocketMask,
    });
    const pocketSeq = flattenChainSequences(
      input.proteinResidues,
      shared.sequencesByChain(context, pocketDesign.sequence),
    ).sequence;
    let proteinOrdinal = 0;
    for (let i = 0; i < context.nToken; i++) {
      if (!context.isProteinToken[i]) continue;
      const ordinal = proteinOrdinal++;
      if (!near[i]) expect(pocketSeq[ordinal]).toBe(flatNative.sequence[ordinal]);
    }

    // --- DLMC + Metropolis-Hastings ---------------------------------------
    const dlmc = shared.sampleFromParams(params, {
      ...baseOptions,
      proposal: 'dlmc',
      rejectionStep: true,
      dt: 0.1,
      seed: 2,
    });
    expect(dlmc.sequence).toHaveLength(13);
    expect(dlmc.acceptedSweeps).toBeGreaterThanOrEqual(0);
    expect(dlmc.acceptedSweeps).toBeLessThanOrEqual(12);

    // --- classifier-free guidance on the ligand-free pathway ---------------
    const uncond = engine.computeParams(shared.withoutLigand(context));
    expect(uncond.h).toHaveLength(13 * R);
    expect([...uncond.h].some((v, i) => Math.abs(v - params.h[i]) > 1e-6)).toBe(true);
    const guided = shared.sampleFromParams(params, {
      ...baseOptions,
      proposal: 'gibbs',
      seed: 5,
      uncond,
      cfgScale: 2.0,
    });
    expect(guided.sequence).toHaveLength(13);
    expect(
      flattenChainSequences(input.proteinResidues, shared.sequencesByChain(context, guided.sequence))
        .sequence,
    ).toHaveLength(10);

    // --- a designed sequence scores on the same parameters ------------------
    const designScore = shared.scoreSequence(params, gibbs.sequence, {
      isProteinToken: context.isProteinToken,
    });
    expect(Number.isFinite(designScore.pseudoLogLikelihood)).toBe(true);
    const sample: ArtiosPottsSample = {
      flat: designed,
      score: {
        proteinEnergy: designScore.proteinEnergy,
        totalEnergy: designScore.totalEnergy,
        pseudoLogLikelihood: designScore.pseudoLogLikelihood,
        perResidueLogLikelihood: proteinTokenValues(context, designScore.logLikelihood),
        perResidueEnergy: proteinTokenValues(context, designScore.perNodeEnergy),
      },
      sampleIndex: 0,
      seed: 1,
      acceptedSweeps: null,
    };
    const options = artiosPottsOptionsFromParams({ numSequences: 1, numSteps: 12 });
    const candidate = buildArtiosPottsCandidate(sample, stub(), options);
    expect(candidate.sequence).toBe(designed.sequence);
    expect(candidate.score).toBeCloseTo(-designScore.pseudoLogLikelihood, 5);
    const track = buildArtiosPottsTrack(stub(), input.proteinResidues, sample, options, 'smoke');
    expect(track.values).toHaveLength(10);
    expect(track.residueIds).toHaveLength(10);
    expect(Number.isFinite(track.summary!.mean)).toBe(true);
  }, 120_000);

  it('designs a protein-only input with the same code path', async () => {
    const input = buildArtiosPottsInput(PROTEIN_ONLY_PDB);
    const engine = await cpuMod.createArtiosPottsCPU({ baseUrl: '/weights', precision: 'fp32', read });
    const context = engine.featurize(input.spec);
    expect(context.nToken).toBe(10);
    expect([...context.isLigandToken].filter(Boolean)).toHaveLength(0);
    const params = engine.computeParams(context);
    const result = shared.sampleFromParams(params, {
      context,
      proposal: 'gibbs',
      nSweeps: 8,
      temperature: 0.1,
      excludeCys: true,
      excludeMet: true,
      seed: 9,
    });
    const flat = flattenChainSequences(
      input.proteinResidues,
      shared.sequencesByChain(context, result.sequence),
    );
    expect(flat).toMatchObject({ chainIds: ['A'], chainLengths: [10] });
    expect(flat.sequence).toHaveLength(10);
    expect(flat.sequence).not.toMatch(/[CM]/);
  }, 120_000);

  it('rejects a bundle that does not carry the requested precision', async () => {
    await expect(
      cpuMod.createArtiosPottsCPU({ baseUrl: '/weights', precision: 'fp16', read }),
    ).rejects.toThrow(/no "fp16" precision/);
  });
});

// ---------------------------------------------------------------------------
// 7. Row-restricted Gibbs & count-based LCP are bit-identical to the legacy path
// ---------------------------------------------------------------------------

const POTTS_R = 32;

/** Seeded uniform in [-0.5, 0.5). */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
}

/** A compact synthetic backbone (+ optional ligand atoms) with a real kNN graph and random `h` / `J`. */
function syntheticPotts(
  nResidues: number,
  nLigand: number,
  options: { seed?: number; scale?: number; neighbours?: number } = {},
): { context: any; params: any } {
  const { seed = 7, scale = 1, neighbours = 32 } = options;
  const N = nResidues + nLigand;
  const rand = lcg(seed);
  const centre = new Float32Array(N * 3);
  const pos = [0, 0, 0];
  for (let i = 0; i < nResidues; i++) {
    const r = Math.hypot(pos[0], pos[1], pos[2]) || 1;
    const pull = Math.min(0.9, r / (2.2 * Math.cbrt(nResidues)));
    const step = [0, 1, 2].map((d) => rand() - (pull * pos[d]) / r);
    const norm = Math.hypot(step[0], step[1], step[2]) || 1;
    for (let d = 0; d < 3; d++) {
      pos[d] += (3.8 * step[d]) / norm;
      centre[i * 3 + d] = pos[d];
    }
  }
  const anchor = Math.floor(nResidues / 2);
  for (let a = 0; a < nLigand; a++) {
    centre[(nResidues + a) * 3] = centre[anchor * 3] + (a % 3) * 1.4;
    centre[(nResidues + a) * 3 + 1] = centre[anchor * 3 + 1] + Math.floor(a / 3) * 1.4;
    centre[(nResidues + a) * 3 + 2] = centre[anchor * 3 + 2] + 2;
  }
  const tokenMask = new Uint8Array(N).fill(1);
  const isProteinToken = new Uint8Array(N).fill(1, 0, nResidues);
  const isLigandToken = new Uint8Array(N).fill(1, nResidues, N);
  const restypeIndex = Int32Array.from({ length: N }, (_, i) => (i < nResidues ? (i * 7) % 20 : 25));
  // Five backbone slots per residue, as the ligand proximity mask reads them.
  const backboneCoordinate = new Float32Array(N * 5 * 3);
  const backboneMask = new Uint8Array(N * 5);
  for (let i = 0; i < N; i++) {
    for (let slot = 0; slot < 5; slot++) {
      backboneMask[i * 5 + slot] = i < nResidues ? 1 : 0;
      for (let d = 0; d < 3; d++) backboneCoordinate[(i * 5 + slot) * 3 + d] = centre[i * 3 + d] + slot * 0.4;
    }
  }
  const context = {
    nToken: N,
    nAtom: N,
    tokenMask,
    isProteinToken,
    isLigandToken,
    restypeIndex,
    asymId: new Int32Array(N),
    centreCoordinate: centre,
    backboneCoordinate,
    backboneMask,
  };
  const graph = shared.buildNeighbourGraph({ nToken: N, centreCoordinate: centre, tokenMask }, neighbours);
  const h = Float32Array.from({ length: N * POTTS_R }, () => rand() * scale);
  const J = Float32Array.from({ length: N * graph.k * POTTS_R * POTTS_R }, () => rand() * 0.1 * scale);
  return {
    context,
    params: {
      h,
      J,
      edgeIndex: graph.edgeIndex,
      edgeMask: graph.edgeMask,
      nToken: N,
      k: graph.k,
      numResidueTypes: POTTS_R,
    },
  };
}

function sameBits(a: ArrayBufferView, b: ArrayBufferView): boolean {
  if (a.byteLength !== b.byteLength) return false;
  const ua = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const ub = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
  return true;
}

/** Sampler options as `artios-potts.worker.ts` passes them with manifest defaults. */
const WORKER_SAMPLER_DEFAULTS = {
  proposal: 'gibbs',
  nSweeps: 8,
  temperature: 0.1,
  temperatureInit: 1.0,
  annealingFraction: 0.8,
  dt: 0.1,
  rejectionStep: false,
  excludeCys: true,
  excludeMet: true,
  complexityPenaltyWeight: 0.6,
  complexityWindowSize: 30,
  complexityEntropyMin: 2.32,
  activeSiteDistanceCutoff: 10.0,
  granularActiveSiteColoring: true,
  trackTrajectory: true,
};

describe('ArtiosPotts fast sampler is bit-identical to the legacy path', () => {
  /** Every per-sweep sequence, the final state and the energies must carry the same bits. */
  function expectIdenticalRuns(fast: any, legacy: any): void {
    expect(fast.trajectory.sequences).toHaveLength(legacy.trajectory.sequences.length);
    for (let t = 0; t < legacy.trajectory.sequences.length; t++) {
      expect([...fast.trajectory.sequences[t]], `sweep ${t}`).toEqual([...legacy.trajectory.sequences[t]]);
    }
    expect(sameBits(fast.sequence, legacy.sequence)).toBe(true);
    expect(sameBits(fast.oneHot, legacy.oneHot)).toBe(true);
    expect(sameBits(fast.energyPerNode, legacy.energyPerNode)).toBe(true);
    expect(sameBits(fast.temperatureSchedule, legacy.temperatureSchedule)).toBe(true);
    expect(Object.is(fast.totalEnergy, legacy.totalEnergy)).toBe(true);
    expect(fast.trajectory.energies).toEqual(legacy.trajectory.energies);
  }

  /**
   * Runs the legacy path once per seed and checks the fast path against it three ways: built per
   * call, with the per-backbone `prepared` setup the worker hoists, and without the transposed J.
   */
  function compareAgainstLegacy(
    built: { context: any; params: any },
    extra: Record<string, unknown>,
    expected: { columns: number; seeds?: number[] },
  ): void {
    const options = { ...WORKER_SAMPLER_DEFAULTS, context: built.context, ...extra };
    const prepared = shared.prepareSampleFromParams(built.params, options);
    for (const seed of expected.seeds ?? [42, 43]) {
      const legacy = shared.sampleFromParams(built.params, { ...options, seed, legacyPath: true });
      expect(legacy.gibbsPath).toBe('legacy');
      const fast = shared.sampleFromParams(built.params, { ...options, seed });
      expect(fast.gibbsPath).toBe('row-restricted');
      expect(fast.gibbsColumns).toBe(expected.columns);
      expectIdenticalRuns(fast, legacy);
      const hoisted = shared.sampleFromParams(built.params, { ...options, seed, prepared });
      expect(hoisted.gibbsPath).toBe('row-restricted');
      expect(hoisted.gibbsColumns).toBe(expected.columns);
      expectIdenticalRuns(hoisted, legacy);
      const inPlace = shared.sampleFromParams(built.params, { ...options, seed, transposeJ: false });
      expectIdenticalRuns(inPlace, legacy);
    }
  }

  it('count-based complexityLcp returns the legacy gradient and value bits', () => {
    const N = 64;
    const rand = lcg(3);
    const chainId = Int32Array.from({ length: N }, (_, i) => (i < 40 ? 0 : 1));
    const tokenMask = new Uint8Array(N).fill(1);
    tokenMask[17] = 0;
    for (const [windowSize, entropyMin, alphabet] of [
      [30, 2.32, 20],
      [30, 2.32, 3],
      [7, 1.6, 6],
    ]) {
      const lcpOptions = { nToken: N, numResidueTypes: POTTS_R, chainId, tokenMask, windowSize, entropyMin };
      const evaluate = shared.createComplexityLcp(lcpOptions);
      let nonZero = 0;
      for (let rep = 0; rep < 4; rep++) {
        const S = new Float32Array(N * POTTS_R);
        for (let i = 0; i < N; i++) {
          // Mostly a low-complexity alphabet, with the odd non-amino-acid state.
          const state = rand() > 0.42 ? 20 + (i % 12) : Math.floor((rand() + 0.5) * alphabet);
          S[i * POTTS_R + state] = 1;
        }
        const legacy = shared.complexityLcpLegacy(S, lcpOptions);
        for (const got of [evaluate(S), shared.complexityLcp(S, lcpOptions)]) {
          expect(sameBits(got.grad, legacy.grad)).toBe(true);
          expect(Object.is(got.value, legacy.value)).toBe(true);
        }
        nonZero += legacy.grad.filter((v: number) => v !== 0).length;
      }
      // A full alphabet sits above the perplexity floor (all-zero gradient); for the reduced
      // ones the comparison is vacuous unless the penalty actually fires.
      if (alphabet < 20) expect(nonZero, `window ${windowSize}`).toBeGreaterThan(0);
    }
    // A soft state has no integer counts: it must take the brute-force path, not approximate.
    const soft = new Float32Array(N * POTTS_R);
    for (let i = 0; i < N; i++) {
      soft[i * POTTS_R + (i % 4)] = 0.75;
      soft[i * POTTS_R + 5] = 0.25;
    }
    const softOptions = { nToken: N, numResidueTypes: POTTS_R, chainId, tokenMask };
    const softLegacy = shared.complexityLcpLegacy(soft, softOptions);
    const softGot = shared.complexityLcp(soft, softOptions);
    expect(sameBits(softGot.grad, softLegacy.grad)).toBe(true);
    expect(Object.is(softGot.value, softLegacy.value)).toBe(true);
  });

  it('apo backbone: identical sequences at every sweep, 20 live columns', () => {
    compareAgainstLegacy(syntheticPotts(40, 0), {}, { columns: 20 });
  }, 120_000);

  it('sparse graph: identical when a colour group resamples many positions at once', () => {
    // With 8 neighbours the colouring needs few colours, so groups are large and every member's
    // conditional must come from the pre-group state, as in a 300-residue production run.
    const built = syntheticPotts(60, 0, { neighbours: 8 });
    const groups: Uint8Array[] = shared.colorGroups(built.params);
    const largest = Math.max(...groups.map((g) => g.reduce((a, b) => a + b, 0)));
    expect(largest).toBeGreaterThanOrEqual(4);
    compareAgainstLegacy(built, {}, { columns: 20 });
  }, 120_000);

  it('low-complexity regime: identical while the complexity penalty is steering', () => {
    const built = syntheticPotts(40, 0);
    // Six letters left: every window is below the perplexity floor, so no LCP term can be skipped.
    const energyBias = Object.fromEntries([...'ADEFGHIKLNPQ'].map((c) => [c, shared.ALPHABET_BIAS]));
    const extra = { energyBias, complexityPenaltyWeight: 2 };
    compareAgainstLegacy(built, extra, { columns: 20, seeds: [42] });
    // The penalty must change the outcome here, or this case would not exercise it at all.
    const options = { ...WORKER_SAMPLER_DEFAULTS, context: built.context, energyBias, seed: 42 };
    const steered = shared.sampleFromParams(built.params, { ...options, complexityPenaltyWeight: 2 });
    const unsteered = shared.sampleFromParams(built.params, { ...options, complexityPenaltyWeight: 0 });
    expect([...steered.sequence]).not.toEqual([...unsteered.sequence]);
  }, 120_000);

  it('ligand + granular active-site colouring: identical with singleton colours', () => {
    const built = syntheticPotts(36, 6);
    const groups = shared.colorGroups(built.params, {
      activeSiteMask: shared.buildLigandProximityMask(built.context, 10.0),
      granular: true,
    });
    // Granular colouring gives each active-site token its own colour, ligand atoms included.
    expect(groups.length).toBeGreaterThan(6 + 20);
    compareAgainstLegacy(built, {}, { columns: 20 });
  }, 120_000);

  it('classifier-free guidance: identical when blending against ligand-free parameters', () => {
    const built = syntheticPotts(36, 6);
    const uncond = syntheticPotts(36, 6, { seed: 11 }).params;
    compareAgainstLegacy(built, { uncond, cfgScale: 2.0 }, { columns: 20, seeds: [42] });
    // Active-site-only CFG over a partial pocket, so both guided and unguided rows are exercised.
    const activeSiteMask = Uint8Array.from({ length: built.context.nToken }, (_, i) => (i % 3 === 0 ? 1 : 0));
    compareAgainstLegacy(
      built,
      { uncond, cfgScale: 0.5, activeSiteOnlyCfg: true, activeSiteMask },
      { columns: 20, seeds: [43] },
    );
  }, 120_000);

  it('fixed positions: identical, and fixed tokens keep their identity (incl. C, M and an omitted letter)', () => {
    const built = syntheticPotts(36, 6);
    const fixedMask = new Uint8Array(built.context.nToken);
    for (let i = 0; i < 36; i += 3) {
      fixedMask[i] = 1;
      built.context.restypeIndex[i] = [4, 12, 0][(i / 3) % 3]; // CYS, MET, ALA
    }
    const extra = { fixedMask, energyBias: { A: shared.ALPHABET_BIAS } };
    compareAgainstLegacy(built, extra, { columns: 20 });
    const run = shared.sampleFromParams(built.params, {
      ...WORKER_SAMPLER_DEFAULTS,
      context: built.context,
      ...extra,
      seed: 42,
    });
    for (let i = 0; i < built.context.nToken; i++) {
      if (fixedMask[i] || i >= 36) expect(run.sequence[i]).toBe(built.context.restypeIndex[i]);
      else expect([0, 4, 12]).not.toContain(run.sequence[i]);
    }
  }, 120_000);

  it('dead-state guard refusal: falls back to all 32 columns and stays identical', () => {
    const built = syntheticPotts(36, 6);
    // A hot start, or a weak alphabet bias, leaves non-amino-acid states reachable in principle.
    compareAgainstLegacy(built, { temperatureInit: 20 }, { columns: 32, seeds: [42] });
    compareAgainstLegacy(built, { alphabetBias: 50 }, { columns: 32, seeds: [43] });
  }, 120_000);

  it('keeps the legacy sweep when another penalty is active, and DLMC stays identical', () => {
    const built = syntheticPotts(30, 4);
    const options = { ...WORKER_SAMPLER_DEFAULTS, context: built.context, nSweeps: 4, seed: 5 };
    const mixed = shared.sampleFromParams(built.params, { ...options, netChargePenaltyWeight: 0.5 });
    expect(mixed.gibbsPath).toBe('legacy');
    expect(mixed.penaltyNames).toEqual(['lcp', 'net_charge']);
    const mixedLegacy = shared.sampleFromParams(built.params, {
      ...options,
      netChargePenaltyWeight: 0.5,
      legacyPath: true,
    });
    expectIdenticalRuns(mixed, mixedLegacy);

    // DLMC needs the full gradient every sweep, so only the penalty itself is faster there.
    const dlmc = { ...options, proposal: 'dlmc', rejectionStep: true, nSweeps: 12 };
    const dlmcFast = shared.sampleFromParams(built.params, dlmc);
    const dlmcLegacy = shared.sampleFromParams(built.params, { ...dlmc, legacyPath: true });
    expect(dlmcFast.gibbsPath).toBeNull();
    expect(dlmcFast.acceptedSweeps).toBe(dlmcLegacy.acceptedSweeps);
    expectIdenticalRuns(dlmcFast, dlmcLegacy);
  }, 120_000);

  it('refuses a `prepared` setup built for other parameters', () => {
    const built = syntheticPotts(30, 0);
    const other = syntheticPotts(30, 0, { seed: 9 });
    const options = { ...WORKER_SAMPLER_DEFAULTS, context: built.context };
    const prepared = shared.prepareSampleFromParams(built.params, options);
    expect(() =>
      shared.sampleFromParams(other.params, { ...options, context: other.context, prepared }),
    ).toThrow(/prepared/);
  });
});
