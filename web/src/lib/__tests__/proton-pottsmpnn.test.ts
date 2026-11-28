// proton-pottsmpnn (Proton-PottsMPNN; christian-creator/ProtonPottsMPNN, MIT, with PottsMPNN from
// IPD rc-foundry under BSD-3-Clause): plugin manifest & registration, PDB/mmCIF -> backbone residue
// compilation checked position-for-position against the vendored parser, protonation-override and
// pH-centre resolution, fixed positions & designable-set resolution, the v6 forbidden-token
// alphabet, SequenceSet / ResidueTrack assembly, collection fan-out, and an end-to-end CPU smoke
// test that runs the vendored runtime in `static/vendor/proton-pottsmpnn/` (featurize -> Potts head
// -> energies -> pH-selective block descent) on a deterministic synthetic weight bundle.

import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { CONCISE_BLOCK_ROLES, CATALOG_STAGE_GROUPS } from '$lib/engine/catalog';
import { ADAPTER_TO_SHORT, FLOW_MODEL_ALIASES, SHORT_TO_ADAPTER } from '$lib/engine/flow-dsl';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { getDefaultParamsFromSchema, validateParamsWithSchema } from '$lib/engine/param-schema';
import { createDefaultPluginRegistry } from '$lib/plugins';
import { formatProtonPottsMpnnSummary, protonPottsMpnnPlugin } from '$lib/plugins/proton-pottsmpnn';
import {
  PROTON_POTTSMPNN_CENTRE_TOKENS,
  PROTON_POTTSMPNN_MANIFEST,
  PROTON_POTTSMPNN_STATE_TOKENS,
  PROTON_POTTSMPNN_WEIGHTS_BASE_URL,
  PROTON_POTTSMPNN_WEIGHTS_MB,
} from '$lib/plugins/proton-pottsmpnn/manifest';
import {
  BACKBONE_ATOMS,
  DEFAULT_FORBIDDEN_TOKENS,
  PHYSIOLOGICAL_STATE,
  STANDARD_AA,
  THREE_TO_ONE,
  TITRATABLE_PARENTS,
  V6_DEP_MAP,
  V6_PROTONATION_TOKENS,
  V6_TOKENS,
  buildCentreReports,
  buildProtonPottsCandidate,
  buildProtonPottsInput,
  buildProtonPottsSequenceSet,
  buildProtonPottsTrack,
  chainLayoutOf,
  formatCentreReport,
  inputPdbText,
  orderProtonPottsCandidates,
  parseProtonationEntry,
  parseProtonationSpec,
  pinCentres,
  planProtonPottsJobs,
  protonPottsObjective,
  protonPottsOptionsFromParams,
  protonPottsRunHash,
  residueIdsOf,
  resolveDesignableSet,
  resolveProtonPottsAlphabet,
  resolveProtonPottsCentres,
  resolveProtonPottsConstraints,
  resolveProtonationOverrides,
  resolveResiduePosition,
  sanitizeOmitAAs,
  siteEnergies,
  splitProtonationEntries,
  structureItems,
  tokenParent,
  type ProtonPottsSample,
  type ProtonPottsScore,
} from '$lib/plugins/proton-pottsmpnn/jobs';
import type { AdapterParamField, StructureArtifact } from '$lib/engine/types';

/* eslint-disable @typescript-eslint/no-explicit-any */
const VENDOR = resolve(process.cwd(), 'static/vendor/proton-pottsmpnn');
const load = async (rel: string): Promise<any> =>
  import(/* @vite-ignore */ pathToFileURL(join(VENDOR, rel)).href);

const shared = await load('shared/index.js');
const cpuMod = await load('cpu/index.js');

// `webgpu/engine.js` reads GPUBufferUsage at module scope; stub the enums so the export surface of
// the WebGPU entry point can still be contract-checked in Node (no device is ever requested).
const gpuMod = await (async () => {
  const g = globalThis as Record<string, unknown>;
  g.GPUBufferUsage ??= { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4, UNIFORM: 64, MAP_READ: 1 };
  g.GPUMapMode ??= { READ: 1 };
  return load('webgpu/index.js');
})();

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
  options: {
    resNames?: readonly string[];
    startResSeq?: number;
    serial?: number;
    offset?: number;
    insCodes?: readonly string[];
    resSeqs?: readonly number[];
    dropAtoms?: ReadonlySet<string>;
  } = {},
): { text: string; nextSerial: number } {
  const {
    resNames,
    startResSeq = 1,
    serial: startSerial = 1,
    offset = 0,
    insCodes,
    resSeqs,
    dropAtoms,
  } = options;
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
    const resSeq = resSeqs?.[i] ?? startResSeq + i;
    const ins = insCodes?.[i] ?? ' ';
    const atoms: Array<[string, [number, number, number], string]> = [
      ['N', n, 'N'],
      ['CA', ca, 'C'],
      ['C', c, 'C'],
      ['O', o, 'O'],
    ];
    for (const [name, xyz, element] of atoms) {
      if (dropAtoms?.has(`${resSeq}${ins.trim()}:${name}`)) continue;
      lines.push(pdbLine('ATOM', serial++, name, resName, chain, resSeq, xyz, element, ins));
    }
  }
  return { text: lines.join('\n'), nextSerial: serial };
}

/** 14 residues with two histidines, one aspartate and two glutamates for the pH machinery. */
const TITRATABLE_RESNAMES = [
  'ALA', 'LEU', 'GLU', 'LYS', 'SER', 'HIS', 'VAL', 'ILE',
  'ALA', 'ASP', 'LEU', 'GLU', 'SER', 'HIS',
] as const;

const TITRATABLE_PDB = `${helixPdb(TITRATABLE_RESNAMES.length, 'A', { resNames: TITRATABLE_RESNAMES }).text}\nEND\n`;
const PROTEIN_ONLY_PDB = `${helixPdb(10).text}\nEND\n`;

const stub = (over: Partial<StructureArtifact> = {}): StructureArtifact =>
  ({
    kind: 'StructureArtifact',
    id: 'struct-1',
    sourceModelId: 'pdb-input',
    length: 14,
    pdbText: TITRATABLE_PDB,
    ...over,
  }) as StructureArtifact;

// ---------------------------------------------------------------------------
// 1. Manifest, plugin & registration
// ---------------------------------------------------------------------------

describe('proton-pottsmpnn plugin manifest & registration', () => {
  it('declares the Proton-PottsMPNN identity, stage, runtime and weight budget', () => {
    const m = PROTON_POTTSMPNN_MANIFEST;
    expect(m.id).toBe('proton-pottsmpnn');
    expect(m.displayName).toBe('Proton-PottsMPNN');
    expect(m.shortName).toBe('Proton-PottsMPNN');
    // `inverse-fold` is the Sequence Design stage of the protein-biopolymer domain.
    expect(m.stage).toBe('inverse-fold');
    expect(m.runtime).toBe('wgsl-custom');
    expect(m.licenseNote).toContain('MIT');
    expect(m.licenseNote).toContain('BSD-3-Clause');
    expect(m.licenseNote).toContain('christian-creator/ProtonPottsMPNN');
    expect(m.description.length).toBeGreaterThan(80);
    expect(m.approxWeightsMB).toBe(PROTON_POTTSMPNN_WEIGHTS_MB.fp16);
    expect(m.precisionWeightsMB).toEqual({ int8: 1.87, fp16: 3.6, fp32: 7.1 });
    expect(m.estimatedVramMB).toBeGreaterThan(0);
    expect(PROTON_POTTSMPNN_WEIGHTS_BASE_URL).toBe(
      'https://huggingface.co/ramithuh/proton-pottsmpnn-webgpu/resolve/main',
    );
  });

  it('wires a single StructureArtifact input to a SequenceSet plus an optional ResidueTrack', () => {
    expect(PROTON_POTTSMPNN_MANIFEST.inputs).toEqual({ structure: { type: 'StructureArtifact' } });
    expect(PROTON_POTTSMPNN_MANIFEST.outputs).toEqual({
      sequences: { type: 'SequenceSet' },
      track: { type: 'ResidueTrack', required: false },
    });
  });

  it('exposes design and score modes keyed by `mode`', () => {
    expect(PROTON_POTTSMPNN_MANIFEST.modeParamKey).toBe('mode');
    expect(PROTON_POTTSMPNN_MANIFEST.modes?.map((mode) => mode.id)).toEqual(['design', 'score']);
    for (const mode of PROTON_POTTSMPNN_MANIFEST.modes ?? []) {
      expect(mode.label.length).toBeGreaterThan(0);
      expect(mode.description.length).toBeGreaterThan(20);
      // Both modes read the structure's own coordinates; neither needs extra ports.
      expect(mode.requiredInputs ?? []).toEqual([]);
    }
  });

  it('declares every parameter the vendored runtime actually takes, with upstream-faithful bounds', () => {
    const schema = PROTON_POTTSMPNN_MANIFEST.paramSchema as AdapterParamField[];
    const byKey = new Map(schema.map((f) => [f.key, f]));
    expect([...byKey.keys()]).toEqual([
      'mode',
      'protonationCentres',
      'protonationOverrides',
      'lambda',
      'numSequences',
      'temperature',
      'fixedPositions',
      'designScope',
      'shellPerCentre',
      'maxDesignable',
      'blockSize',
      'maxRounds',
      'sweepOrder',
      'zscaleMode',
      'omitAAs',
      'titratableTokens',
      'precision',
      'seed',
    ]);
    for (const field of schema) {
      expect(field.label.length, `${field.key} needs a label`).toBeGreaterThan(0);
      expect(field.description?.length ?? 0, `${field.key} needs a description`).toBeGreaterThan(20);
      expect(['primary', 'advanced']).toContain(field.tier);
    }

    expect(byKey.get('mode')!.defaultValue).toBe('design');
    expect(byKey.get('lambda')).toMatchObject({ defaultValue: 0.5, min: 0, max: 1, modes: ['design'] });
    expect(byKey.get('numSequences')).toMatchObject({ defaultValue: 4, min: 1, max: 32 });
    // Upstream `blockDescent` defaults: temperature 0 (exact argmin), blockSize 2, maxRounds 10.
    expect(byKey.get('temperature')).toMatchObject({ defaultValue: 0, min: 0, max: 2 });
    expect(byKey.get('blockSize')).toMatchObject({ defaultValue: 2, min: 1, max: 3 });
    expect(byKey.get('maxRounds')).toMatchObject({ defaultValue: 10, min: 1, max: 50 });
    expect(byKey.get('blockSize')!.upstreamDefaultHint).toContain('2');
    expect(byKey.get('fixedPositions')).toMatchObject({
      kind: 'residue-list',
      studioRole: 'fixed',
      allowBareChain: true,
    });
    expect(byKey.get('omitAAs')).toMatchObject({ kind: 'aa-set', defaultValue: '' });
    expect(byKey.get('designScope')!.options?.map((o) => o.value)).toEqual([
      'centre-shell',
      'all-positions',
    ]);
    expect(byKey.get('sweepOrder')!.options?.map((o) => o.value)).toEqual([
      'position',
      'knn',
      'energy',
    ]);
    expect(byKey.get('zscaleMode')!.options?.map((o) => o.value)).toEqual([
      'block',
      'single-mutation',
    ]);
    expect(byKey.get('titratableTokens')!.options?.map((o) => o.value)).toEqual([
      'upstream',
      'explicit-states',
    ]);
    expect(byKey.get('precision')!.options?.map((o) => o.value)).toEqual(['fp16', 'int8', 'fp32']);
    expect(byKey.get('precision')!.defaultValue).toBe('fp16');

    // Score mode only needs the two pinning controls plus the precision tier.
    const scoreOnly = schema.filter((f) => !f.modes || f.modes.includes('score')).map((f) => f.key);
    expect(scoreOnly).toEqual([
      'mode',
      'protonationCentres',
      'protonationOverrides',
      'precision',
    ]);
  });

  it('round-trips schema defaults through declarative validation and rejects bad values', () => {
    const defaults = getDefaultParamsFromSchema(PROTON_POTTSMPNN_MANIFEST);
    expect(defaults).toMatchObject({
      mode: 'design',
      lambda: 0.5,
      numSequences: 4,
      temperature: 0,
      blockSize: 2,
      precision: 'fp16',
      seed: 42,
    });
    expect(validateParamsWithSchema(PROTON_POTTSMPNN_MANIFEST, defaults)).toEqual([]);
    expect(
      validateParamsWithSchema(PROTON_POTTSMPNN_MANIFEST, { ...defaults, mode: 'refine' }),
    ).toHaveLength(1);
    expect(
      validateParamsWithSchema(PROTON_POTTSMPNN_MANIFEST, { ...defaults, lambda: 1.5 }),
    ).toHaveLength(1);
    expect(
      validateParamsWithSchema(PROTON_POTTSMPNN_MANIFEST, { ...defaults, blockSize: 2.5 }),
    ).toHaveLength(1);
    expect(
      validateParamsWithSchema(PROTON_POTTSMPNN_MANIFEST, { ...defaults, omitAAs: 'CZ' }),
    ).toHaveLength(1);
    expect(
      validateParamsWithSchema(PROTON_POTTSMPNN_MANIFEST, { ...defaults, fixedPositions: 'A999-1' }),
    ).toHaveLength(1);
    // Design-only knobs are not validated in score mode, where they are hidden.
    expect(
      validateParamsWithSchema(PROTON_POTTSMPNN_MANIFEST, {
        ...defaults,
        mode: 'score',
        numSequences: 999,
      }),
    ).toEqual([]);
  });

  it('cross-validates the two protonation specs and the degenerate λ = 1 case', () => {
    const cross = PROTON_POTTSMPNN_MANIFEST.crossValidate!;
    expect(cross({ mode: 'design', lambda: 0.5, protonationCentres: 'A42=HIS-P' })).toEqual([]);
    // Whitespace around the separator is folded away, exactly as `jobs.ts` does.
    expect(cross({ mode: 'design', lambda: 0.5, protonationCentres: 'A42 = HIS-P, A58 = GLU-P' })).toEqual([]);
    expect(cross({ mode: 'score', protonationOverrides: 'A42=HIS-P; A9=ASP-P A11=GLU-D' })).toEqual([]);

    expect(cross({ mode: 'design', protonationCentres: 'A42' })).toEqual([
      expect.stringContaining('Invalid protonationCentres entry "A42"'),
    ]);
    // Only the three protonated microstates with a deprotonated partner can be a centre.
    expect(cross({ mode: 'design', protonationCentres: 'A42=HIS-S' })).toEqual([
      expect.stringContaining('Invalid protonationCentres token "HIS-S"'),
    ]);
    expect(cross({ mode: 'score', protonationOverrides: 'A42=HIS-Q' })).toEqual([
      expect.stringContaining('Invalid protonationOverrides token "HIS-Q"'),
    ]);
    // An override may be any v6 token, including a plain parent.
    expect(cross({ mode: 'score', protonationOverrides: 'A42=ALA' })).toEqual([]);

    expect(cross({ mode: 'design', lambda: 1, protonationCentres: '' })).toEqual([
      expect.stringContaining('λ = 1 is pure pH selectivity'),
    ]);
    expect(cross({ mode: 'design', lambda: 1, protonationCentres: 'A42=HIS-P' })).toEqual([]);
    // λ = 1 is only degenerate while designing; score mode never runs the descent.
    expect(cross({ mode: 'score', lambda: 1, protonationCentres: '' })).toEqual([]);
  });

  it('exposes the plugin with share code "pm", both planner roles and a CPU fallback', () => {
    expect(protonPottsMpnnPlugin.id).toBe('proton-pottsmpnn');
    expect(protonPottsMpnnPlugin.domainId).toBe('protein-biopolymer');
    expect(protonPottsMpnnPlugin.share.shortCode).toBe('pm');
    expect(protonPottsMpnnPlugin.execution.runtime).toBe('wgsl-custom');
    expect(protonPottsMpnnPlugin.execution.supportsCooperativeAbort).toBe(true);
    expect(protonPottsMpnnPlugin.execution.abortGraceMs ?? 0).toBeGreaterThanOrEqual(1500);
    expect(protonPottsMpnnPlugin.execution.vramLeaseMode).toBe('exclusive');
    expect(protonPottsMpnnPlugin.execution.fallback).toEqual({
      kind: 'cpu-js',
      moduleUrl: '/vendor/proton-pottsmpnn/cpu/index.js',
      degradedWarning: expect.stringContaining('CPU JS fallback'),
    });
    expect(protonPottsMpnnPlugin.cachePolicy.usesSharedOpfsStore).toBe(true);
    expect(protonPottsMpnnPlugin.catalog.categoryId).toBe('sequence-design');
    expect(protonPottsMpnnPlugin.catalog.conciseRole).toContain('pH');
    expect(protonPottsMpnnPlugin.catalog.searchKeywords).toContain('protonation');
    expect(protonPottsMpnnPlugin.planner?.plannerRole).toBe('sequence-designer');
    expect(protonPottsMpnnPlugin.planner?.plannerRoles).toEqual(['sequence-designer', 'scorer']);
    expect(protonPottsMpnnPlugin.planner?.capabilities).toContain('ph-selective-sequence-design');
    expect(protonPottsMpnnPlugin.planner?.capabilities).toContain('sequence-scoring');
    expect(protonPottsMpnnPlugin.display.compareEligible).toBe(true);
    expect(protonPottsMpnnPlugin.manifest.id).toBe('proton-pottsmpnn');
  });

  it('estimates preflight download and output cardinality per mode', () => {
    const ctx = (params: Record<string, unknown>) => ({
      params,
      upstreamProfiles: { structure: { itemCount: 3 } as any },
      connectedInputs: new Set(['structure']),
    });
    const sampled = protonPottsMpnnPlugin.preflight(
      ctx({ mode: 'design', numSequences: 4, temperature: 0.5, precision: 'fp16' }),
    );
    expect(sampled.estimatedDownloadMB).toBe(3.6);
    expect(sampled.workItems).toBe(12);
    expect(sampled.outputCardinality).toEqual({ sequences: 12, track: 1 });
    expect(sampled.stageBreakdown).toEqual({ 'inverse-fold': 12 });

    // At Temp 0 the descent is deterministic, so only one design per backbone is produced.
    const argmin = protonPottsMpnnPlugin.preflight(
      ctx({ mode: 'design', numSequences: 4, temperature: 0, precision: 'int8' }),
    );
    expect(argmin.estimatedDownloadMB).toBe(1.87);
    expect(argmin.workItems).toBe(3);

    const score = protonPottsMpnnPlugin.preflight(
      ctx({ mode: 'score', numSequences: 4, precision: 'fp32' }),
    );
    expect(score.estimatedDownloadMB).toBe(7.1);
    expect(score.workItems).toBe(3);
  });

  it('summarizes design and score nodes for the canvas', () => {
    expect(formatProtonPottsMpnnSummary({})).toBe('1 seq · stability only · B=2 · argmin · FP16');
    expect(
      formatProtonPottsMpnnSummary({ protonationCentres: 'A6=HIS-P, A10=ASP-P', lambda: 0.8 }),
    ).toBe('1 seq · λ=0.8 · 2 pH centres · B=2 · argmin · FP16');
    expect(
      formatProtonPottsMpnnSummary({
        protonationCentres: 'A6=HIS-P',
        numSequences: 8,
        temperature: 0.4,
        blockSize: 1,
        precision: 'int8',
      }),
    ).toBe('8 seqs · λ=0.5 · 1 pH centre · B=1 · T=0.4 · INT8');
    expect(formatProtonPottsMpnnSummary({ mode: 'score' })).toBe(
      'Score native · Potts energy · no pH centre · FP16',
    );
    expect(formatProtonPottsMpnnSummary({ mode: 'score', protonationCentres: 'A6=HIS-P' })).toBe(
      'Score native · Potts energy · 1 pH centre · ΔE · FP16',
    );
    // The plugin's own display hook must agree with the exported formatter.
    expect(protonPottsMpnnPlugin.display.formatCanvasSummary({ mode: 'score' })).toBe(
      formatProtonPottsMpnnSummary({ mode: 'score' }),
    );
  });

  it('mirrors the vendored v6 vocabulary tables exactly', () => {
    expect(V6_TOKENS).toEqual([...shared.V6_TOKENS]);
    expect(V6_TOKENS).toHaveLength(30);
    expect(STANDARD_AA).toEqual([...shared.STANDARD_AA]);
    expect(V6_PROTONATION_TOKENS).toEqual([...shared.V6_PROTONATION_TOKENS]);
    expect(PHYSIOLOGICAL_STATE).toEqual({ ...shared.PHYSIOLOGICAL_STATE });
    expect(DEFAULT_FORBIDDEN_TOKENS).toEqual([...shared.DEFAULT_FORBIDDEN_TOKENS]);
    expect(TITRATABLE_PARENTS).toEqual([...shared.TITRATABLE_PARENTS]);
    expect(Object.keys(V6_DEP_MAP)).toEqual(Object.keys(shared.V6_DEP_MAP));
    for (const [token, partners] of Object.entries(V6_DEP_MAP)) {
      expect(partners).toEqual([...shared.V6_DEP_MAP[token]]);
    }
    for (const [three, one] of Object.entries(THREE_TO_ONE)) {
      expect(shared.THREE_TO_ONE[three]).toBe(one);
    }
    expect(BACKBONE_ATOMS).toEqual(['N', 'CA', 'C', 'O']);
    // The manifest's own copies (used by `crossValidate`) must agree with the vendored tables.
    expect([...PROTON_POTTSMPNN_STATE_TOKENS]).toEqual([...shared.V6_PROTONATION_TOKENS]);
    expect([...PROTON_POTTSMPNN_CENTRE_TOKENS]).toEqual(Object.keys(shared.V6_DEP_MAP));
    expect(tokenParent('HIS-P')).toBe('HIS');
    expect(tokenParent('ALA')).toBe('ALA');
  });

  it('matches the shared registry, catalog, share-code and model-card wiring once applied', () => {
    const registered = (ADAPTER_REGISTRY as Record<string, unknown>)['proton-pottsmpnn'];
    if (registered) {
      expect(JSON.parse(JSON.stringify(registered))).toEqual(
        JSON.parse(JSON.stringify(PROTON_POTTSMPNN_MANIFEST)),
      );
      expect(CATALOG_STAGE_GROUPS.find((g) => g.title === 'Sequence Design')!.ids).toContain(
        'proton-pottsmpnn',
      );
      expect(CONCISE_BLOCK_ROLES['proton-pottsmpnn']).toBeTruthy();
      expect(getLegacyModelCardForAdapter('proton-pottsmpnn')?.adapterId).toBe('proton-pottsmpnn');
    } else {
      // Shared-file wiring is applied by the coordinator; until then the plugin runs off its own
      // manifest literal, and nothing else may have claimed the `pm` short code.
      expect(protonPottsMpnnPlugin.manifest).toBe(PROTON_POTTSMPNN_MANIFEST);
    }

    // `pm` is either unclaimed or ours — never someone else's.
    expect(SHORT_TO_ADAPTER.pm ?? 'proton-pottsmpnn').toBe('proton-pottsmpnn');
    expect(ADAPTER_TO_SHORT['proton-pottsmpnn'] ?? 'pm').toBe('pm');
    for (const alias of ['proton-potts', 'protonpotts', 'proton-mpnn']) {
      const resolved = FLOW_MODEL_ALIASES[alias]?.adapterId;
      expect(resolved ?? 'proton-pottsmpnn').toBe('proton-pottsmpnn');
    }

    const registry = createDefaultPluginRegistry({ freeze: true });
    if (registry.hasPlugin('proton-pottsmpnn')) {
      const plugin = registry.getPlugin('proton-pottsmpnn')!;
      expect(plugin.share.shortCode).toBe('pm');
      expect(registry.getPluginPlannerRoles(plugin)).toEqual(['sequence-designer', 'scorer']);
      expect(registry.listPlugins().filter((p) => p.share.shortCode === 'pm')).toHaveLength(1);
    } else {
      expect(registry.listPlugins().some((p) => p.share.shortCode === 'pm')).toBe(false);
    }
  });
});

// ---------------------------------------------------------------------------
// 2. Param normalization & alphabet
// ---------------------------------------------------------------------------

describe('protonPottsOptionsFromParams', () => {
  it('applies the schema defaults for an empty param object', () => {
    expect(protonPottsOptionsFromParams({})).toEqual({
      mode: 'design',
      protonationCentres: '',
      protonationOverrides: '',
      lambda: 0.5,
      numSequences: 4,
      temperature: 0,
      fixedPositions: '',
      designScope: 'centre-shell',
      shellPerCentre: 8,
      maxDesignable: 12,
      blockSize: 2,
      maxRounds: 10,
      sweepOrder: 'position',
      zscaleMode: 'block',
      omitAAs: '',
      titratableTokens: 'upstream',
      precision: 'fp16',
      seed: 42,
    });
    // Defaults must be the schema's defaults, not a second hard-coded copy.
    const schemaDefaults = getDefaultParamsFromSchema(PROTON_POTTSMPNN_MANIFEST);
    const normalized = protonPottsOptionsFromParams(schemaDefaults) as unknown as Record<string, unknown>;
    for (const [key, value] of Object.entries(schemaDefaults)) {
      expect(normalized[key], `schema default for ${key}`).toEqual(value);
    }
  });

  it('clamps out-of-range numbers and rounds integers instead of failing the run', () => {
    const opts = protonPottsOptionsFromParams({
      lambda: 7,
      numSequences: 999,
      temperature: -3,
      blockSize: 9,
      maxRounds: 0,
      shellPerCentre: 0,
      maxDesignable: -5,
      seed: -1,
    });
    expect(opts.lambda).toBe(1);
    expect(opts.numSequences).toBe(32);
    expect(opts.temperature).toBe(0);
    expect(opts.blockSize).toBe(3);
    expect(opts.maxRounds).toBe(1);
    expect(opts.shellPerCentre).toBe(1);
    expect(opts.maxDesignable).toBe(0);
    expect(opts.seed).toBe(0);
    expect(protonPottsOptionsFromParams({ blockSize: 1.6, numSequences: 2.4 })).toMatchObject({
      blockSize: 2,
      numSequences: 2,
    });
  });

  it('falls back to defaults for unknown enums and non-numeric input', () => {
    const opts = protonPottsOptionsFromParams({
      mode: 'refine',
      designScope: 'pocket',
      sweepOrder: 'random',
      zscaleMode: 'global',
      titratableTokens: 'strict',
      precision: 'bf16',
      lambda: 'hot',
      numSequences: null,
    });
    expect(opts).toMatchObject({
      mode: 'design',
      designScope: 'centre-shell',
      sweepOrder: 'position',
      zscaleMode: 'block',
      titratableTokens: 'upstream',
      precision: 'fp16',
      lambda: 0.5,
      numSequences: 4,
    });
    // Enum values are matched case-insensitively.
    expect(protonPottsOptionsFromParams({ mode: 'SCORE', precision: 'INT8' })).toMatchObject({
      mode: 'score',
      precision: 'int8',
    });
  });

  it('trims the two protonation specs and canonicalizes the omit set', () => {
    expect(
      protonPottsOptionsFromParams({
        protonationCentres: '  A6=HIS-P ',
        protonationOverrides: ' A10=ASP-P ',
        omitAAs: 'c,m,c;x',
      }),
    ).toMatchObject({
      protonationCentres: 'A6=HIS-P',
      protonationOverrides: 'A10=ASP-P',
      omitAAs: 'CM',
    });
    expect(sanitizeOmitAAs('cmz')).toBe('CM');
    expect(sanitizeOmitAAs(undefined)).toBe('');
  });
});

describe('resolveProtonPottsAlphabet', () => {
  it('always re-applies the upstream ambiguous-state mask', () => {
    const alphabet = resolveProtonPottsAlphabet('');
    expect(alphabet.forbiddenTokens).toEqual(['HIS-A', 'ASP-A', 'GLU-A']);
    expect(alphabet.allowedTokens).toHaveLength(26); // 30 tokens - UNK - 3 ambiguous states
    // Letters follow the checkpoint's own (3-letter alphabetical) parent order.
    expect(alphabet.designableLetters).toBe('ARNDCQEGHILKMFPSTWYV');
    // A caller-passed list replaces `validTokenMask`'s own default, so the defaults must be inside.
    const mask = shared.validTokenMask(alphabet.forbiddenTokens);
    expect(mask[shared.TOKEN_TO_IDX.UNK]).toBe(0);
    for (const token of ['HIS-A', 'ASP-A', 'GLU-A']) {
      expect(mask[shared.TOKEN_TO_IDX[token]]).toBe(0);
    }
    for (const token of ['ALA', 'HIS-P', 'HIS-S', 'ASP-D', 'HIS']) {
      expect(mask[shared.TOKEN_TO_IDX[token]]).toBe(1);
    }
  });

  it('forbids every v6 token of an omitted parent', () => {
    const alphabet = resolveProtonPottsAlphabet('CH');
    expect(alphabet.forbiddenTokens).toEqual([
      'CYS',
      'HIS',
      'HIS-P',
      'HIS-S',
      'HIS-A',
      'ASP-A',
      'GLU-A',
    ].filter((t) => V6_TOKENS.includes(t)).sort((a, b) => V6_TOKENS.indexOf(a) - V6_TOKENS.indexOf(b)));
    expect(alphabet.allowedTokens).not.toContain('HIS-P');
    expect(alphabet.allowedTokens).not.toContain('CYS');
    expect(alphabet.designableLetters).not.toContain('C');
    expect(alphabet.designableLetters).not.toContain('H');
    const mask = shared.validTokenMask(alphabet.forbiddenTokens);
    expect(mask[shared.TOKEN_TO_IDX['HIS-S']]).toBe(0);
    expect(mask[shared.TOKEN_TO_IDX.CYS]).toBe(0);
    expect(mask[shared.TOKEN_TO_IDX.ALA]).toBe(1);
  });

  it('can forbid the state-unspecified titratable parents', () => {
    const upstream = resolveProtonPottsAlphabet('', 'upstream');
    expect(upstream.allowedTokens).toContain('HIS');
    expect(upstream.allowedTokens).toContain('ASP');

    const strict = resolveProtonPottsAlphabet('', 'explicit-states');
    expect(strict.forbiddenTokens).toEqual(['ASP', 'GLU', 'HIS', 'HIS-A', 'ASP-A', 'GLU-A']);
    expect(strict.allowedTokens).not.toContain('HIS');
    expect(strict.allowedTokens).toContain('HIS-P');
    expect(strict.allowedTokens).toContain('ASP-D');
    // The parents stay reachable as letters through their state tokens.
    expect(strict.designableLetters).toContain('H');
    expect(strict.designableLetters).toContain('D');
  });

  it('refuses an omit set that leaves nothing to design', () => {
    expect(() => resolveProtonPottsAlphabet('ACDEFGHIKLMNPQRSTVWY')).toThrow(
      /excludes every amino acid/,
    );
  });
});

// ---------------------------------------------------------------------------
// 3. PDB -> residues (checked against the vendored parser)
// ---------------------------------------------------------------------------

/** The vendored parser's view of a PDB, in its own emission order. */
const vendorResidueKeys = (pdbText: string): string[] =>
  (shared.parseBackbonePDB(pdbText) as Array<{ chain: string; resSeq: number; icode: string; resName: string }>)
    .map((r) => `${r.chain.trim() || '_'}${r.resSeq}${r.icode.trim()}:${r.resName}`);

const pluginResidueKeys = (pdbText: string): string[] =>
  buildProtonPottsInput(pdbText).residues.map(
    (r) => `${r.chain}${r.number}${r.insertionCode}:${r.resName}`,
  );

describe('buildProtonPottsInput', () => {
  it('reproduces the vendored parseBackbonePDB order position for position', () => {
    const multiChain = (() => {
      const a = helixPdb(4, 'A', { resNames: ['ALA', 'HIS', 'GLU', 'ASP'] });
      const b = helixPdb(3, 'B', { serial: a.nextSerial, offset: 30, startResSeq: 50 });
      return `${a.text}\nTER\n${b.text}\nEND\n`;
    })();
    // Kabat-style insertion codes: 100, 100A, 100B then 101.
    const insertionCoded = helixPdb(4, 'H', {
      resSeqs: [100, 100, 100, 101],
      insCodes: [' ', 'A', 'B', ' '],
      resNames: ['TYR', 'HIS', 'TRP', 'GLY'],
    }).text;
    const withGaps = helixPdb(5, 'A', { dropAtoms: new Set(['3:O']) }).text;
    const noisy = [
      'HEADER    SYNTHETIC',
      pdbLine('HETATM', 900, 'ZN', ' ZN', 'B', 1, [4, 0.5, 1.5], 'ZN'),
      pdbLine('ATOM', 901, 'N', 'MSE', 'A', 77, [1, 1, 1], 'N'),
      pdbLine('ATOM', 902, 'CA', 'MSE', 'A', 77, [2, 1, 1], 'C'),
      pdbLine('ATOM', 903, 'O', 'HOH', 'W', 1, [9, 9, 9], 'O'),
      helixPdb(3, 'A', { serial: 1000 }).text,
      'END',
    ].join('\n');

    for (const [name, text] of [
      ['titratable helix', TITRATABLE_PDB],
      ['multi chain', multiChain],
      ['insertion codes', insertionCoded],
      ['incomplete backbone', withGaps],
      ['hetatm / non-standard / water', noisy],
    ] as const) {
      expect(pluginResidueKeys(text), name).toEqual(vendorResidueKeys(text));
    }
  });

  it('keeps the first altLoc of an atom, exactly as upstream does', () => {
    const base = helixPdb(3, 'A').text.split('\n');
    // Duplicate residue 2's CA with altLoc B and absurd coordinates; the first one must win.
    const withAlt = [
      ...base,
      pdbLine('ATOM', 500, 'CA', 'LEU', 'A', 2, [99, 99, 99], 'C', ' ', 'B'),
    ].join('\n');
    expect(pluginResidueKeys(withAlt)).toEqual(vendorResidueKeys(withAlt));
    expect(buildProtonPottsInput(withAlt).residues).toHaveLength(3);
  });

  it('labels residues, chains and default protonation states', () => {
    const input = buildProtonPottsInput(TITRATABLE_PDB);
    expect(input.residues).toHaveLength(14);
    expect(input.diagnostics.residueCount).toBe(14);
    expect(input.diagnostics.chainIds).toEqual(['A']);
    expect(input.diagnostics.titratable).toEqual({ HIS: 2, ASP: 1, GLU: 2 });
    expect(residueIdsOf(input.residues).slice(0, 3)).toEqual(['A1', 'A2', 'A3']);
    expect(input.residues[5]).toMatchObject({
      chain: 'A',
      number: 6,
      insertionCode: '',
      resName: 'HIS',
      defaultToken: 'HIS-S',
    });
    expect(input.residues[9].defaultToken).toBe('ASP-D');
    expect(input.residues[2].defaultToken).toBe('GLU-D');
    expect(input.residues[0].defaultToken).toBe('ALA');
    expect(input.diagnostics.warnings.join(' ')).toMatch(/5 titratable residue\(s\) default/);
    expect(input.diagnostics.warnings.join(' ')).toMatch(/no HBPLUS\/propka labeller is bundled/);
  });

  it('reports dropped non-standard residues and incomplete backbones', () => {
    const text = [
      pdbLine('ATOM', 1, 'N', 'MSE', 'A', 1, [0, 0, 0], 'N'),
      pdbLine('ATOM', 2, 'CA', 'MSE', 'A', 1, [1, 0, 0], 'C'),
      helixPdb(4, 'A', { startResSeq: 10, serial: 100, dropAtoms: new Set(['12:C']) }).text,
    ].join('\n');
    const input = buildProtonPottsInput(text);
    expect(input.residues).toHaveLength(3);
    expect(input.diagnostics.skippedNonStandard).toEqual(['MSE']);
    expect(input.diagnostics.skippedIncompleteBackbone).toBe(1);
    expect(input.diagnostics.warnings.join(' ')).toMatch(/without a complete N\/CA\/C\/O backbone/);
    expect(input.diagnostics.warnings.join(' ')).toMatch(/Non-standard residues MSE were dropped/);
  });

  it('refuses structures with no usable residue and non-finite coordinates', () => {
    expect(() =>
      buildProtonPottsInput(pdbLine('HETATM', 1, 'ZN', ' ZN', 'B', 1, [0, 0, 0], 'ZN')),
    ).toThrow(/no standard amino acid/);
    const broken = helixPdb(2, 'A')
      .text.split('\n')
      .map((l, i) => (i === 1 ? `${l.slice(0, 30)}     nan${l.slice(38)}` : l))
      .join('\n');
    expect(() => buildProtonPottsInput(broken)).toThrow(/non-finite backbone coordinate/);
  });

  it('reads PDB text directly and converts mmCIF-only structures', () => {
    expect(inputPdbText(stub())).toBe(TITRATABLE_PDB);
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
      'ATOM 1 N N HIS A 1 0.000 0.000 0.000 1 A',
      'ATOM 2 C CA HIS A 1 1.458 0.000 0.000 1 A',
      'ATOM 3 C C HIS A 1 2.009 1.420 0.000 1 A',
      'ATOM 4 O O HIS A 1 1.251 2.390 0.000 1 A',
      'ATOM 5 N N GLY A 2 3.332 1.541 0.000 2 A',
      'ATOM 6 C CA GLY A 2 4.013 2.828 0.000 2 A',
      'ATOM 7 C C GLY A 2 5.520 2.638 0.000 2 A',
      'ATOM 8 O O GLY A 2 6.012 1.511 0.000 2 A',
      '#',
    ].join('\n');
    const converted = inputPdbText(stub({ pdbText: undefined, cifText: cif }));
    expect(converted).toMatch(/^ATOM/m);
    const input = buildProtonPottsInput(converted);
    expect(input.residues.map((r) => r.resName)).toEqual(['HIS', 'GLY']);
    expect(input.residues[0].defaultToken).toBe('HIS-S');
    // The converted text must still parse identically on both sides.
    expect(pluginResidueKeys(converted)).toEqual(vendorResidueKeys(converted));

    expect(() => inputPdbText(stub({ pdbText: undefined, cifText: undefined }))).toThrow(
      /no coordinate text/,
    );
  });

  it('derives the chain layout downstream folders need', () => {
    const multi = (() => {
      const a = helixPdb(3, 'A');
      const b = helixPdb(2, 'B', { serial: a.nextSerial, offset: 30 });
      return `${a.text}\n${b.text}\n`;
    })();
    const residues = buildProtonPottsInput(multi).residues;
    expect(chainLayoutOf(residues)).toEqual({ chainIds: ['A', 'B'], chainLengths: [3, 2] });
    expect(chainLayoutOf(buildProtonPottsInput(TITRATABLE_PDB).residues)).toEqual({
      chainIds: ['A'],
      chainLengths: [14],
    });
  });
});

// ---------------------------------------------------------------------------
// 4. Protonation specs: overrides & pH centres
// ---------------------------------------------------------------------------

describe('protonation specs', () => {
  const residues = buildProtonPottsInput(TITRATABLE_PDB).residues;

  it('splits and parses `<residue>=<token>` entries tolerantly', () => {
    expect(splitProtonationEntries('A6=HIS-P, A10 = ASP-P;A12=GLU-P A3=GLU-D')).toEqual([
      'A6=HIS-P',
      'A10=ASP-P',
      'A12=GLU-P',
      'A3=GLU-D',
    ]);
    expect(splitProtonationEntries('   ')).toEqual([]);
    expect(splitProtonationEntries(undefined)).toEqual([]);
    expect(parseProtonationEntry('A6=HIS-P')).toEqual({ selector: 'A6', token: 'HIS-P' });
    expect(parseProtonationEntry('A6:his-p')).toEqual({ selector: 'A6', token: 'HIS-P' });
    // A multi-char chain selector keeps its own colon: the LAST separator wins.
    expect(parseProtonationEntry('AA:42:HIS-P')).toEqual({ selector: 'AA:42', token: 'HIS-P' });
    expect(parseProtonationEntry('A6')).toBeNull();
    expect(parseProtonationEntry('HIS-P')).toBeNull();
  });

  it('validates tokens against the vocabulary it is given', () => {
    expect(parseProtonationSpec('A6=HIS-P', V6_TOKENS)).toEqual([
      { selector: 'A6', token: 'HIS-P' },
    ]);
    expect(() => parseProtonationSpec('A6=HIS-Z', V6_TOKENS, 'protonationOverrides')).toThrow(
      /invalid protonationOverrides token "HIS-Z"/,
    );
    expect(() => parseProtonationSpec('A6', V6_TOKENS, 'protonationOverrides')).toThrow(
      /invalid protonationOverrides entry "A6"/,
    );
    expect(() => parseProtonationSpec('A6=HIS-S', Object.keys(V6_DEP_MAP), 'protonationCentres')).toThrow(
      /invalid protonationCentres token "HIS-S"/,
    );
  });

  it('resolves single-residue selectors, insertion codes and multi-char chains', () => {
    expect(resolveResiduePosition(residues, 'A1')).toBe(0);
    expect(resolveResiduePosition(residues, 'A6')).toBe(5);
    expect(resolveResiduePosition(residues, 'a6')).toBe(5); // case-insensitive chain fallback
    expect(resolveResiduePosition(residues, '6')).toBe(5); // bare number -> first chain
    expect(() => resolveResiduePosition(residues, 'A99')).toThrow(/is not in the structure/);
    expect(() => resolveResiduePosition(residues, 'Z1')).toThrow(/is not in the structure/);
    expect(() => resolveResiduePosition(residues, 'nonsense')).toThrow(/invalid residue selector/);

    const insertionCoded = buildProtonPottsInput(
      helixPdb(3, 'H', {
        resSeqs: [100, 100, 100],
        insCodes: [' ', 'A', 'B'],
        resNames: ['HIS', 'HIS', 'HIS'],
      }).text,
    ).residues;
    expect(resolveResiduePosition(insertionCoded, 'H100')).toBe(0);
    expect(resolveResiduePosition(insertionCoded, 'H100A')).toBe(1);
    expect(resolveResiduePosition(insertionCoded, 'H100B')).toBe(2);
  });

  it('resolves protonation overrides into positional buildStructure keys', () => {
    const resolved = resolveProtonationOverrides(residues, 'A6=HIS-P, A10=ASP-P');
    expect(resolved.overrides).toEqual({ 5: 'HIS-P', 9: 'ASP-P' });
    expect(resolved.entries).toEqual([
      { position: 5, residueId: 'A6', resName: 'HIS', token: 'HIS-P' },
      { position: 9, residueId: 'A10', resName: 'ASP', token: 'ASP-P' },
    ]);
    expect(resolved.warnings).toEqual([]);
    expect(resolveProtonationOverrides(residues, '').overrides).toEqual({});
  });

  it('warns when an override replaces a different parent or repeats a residue', () => {
    const mismatched = resolveProtonationOverrides(residues, 'A1=GLU-P');
    expect(mismatched.overrides).toEqual({ 0: 'GLU-P' });
    expect(mismatched.warnings.join(' ')).toMatch(/replaces a ALA input token/);

    const repeated = resolveProtonationOverrides(residues, 'A6=HIS-P, A6=HIS-S');
    expect(repeated.overrides).toEqual({ 5: 'HIS-S' });
    expect(repeated.warnings.join(' ')).toMatch(/was given twice/);
  });

  it('resolves pH centres with their deprotonated contrast partners', () => {
    const resolved = resolveProtonPottsCentres(residues, 'A6=HIS-P, A10=ASP-P, A12=GLU-P');
    expect(resolved.centres).toEqual([
      {
        position: 5,
        residueId: 'A6',
        resName: 'HIS',
        protonationType: 'HIS-P',
        contrastTokens: ['HIS-S'],
      },
      {
        position: 9,
        residueId: 'A10',
        resName: 'ASP',
        protonationType: 'ASP-P',
        contrastTokens: ['ASP-D'],
      },
      {
        position: 11,
        residueId: 'A12',
        resName: 'GLU',
        protonationType: 'GLU-P',
        contrastTokens: ['GLU-D'],
      },
    ]);
    expect(resolved.specs).toEqual([
      { position: 5, protonationType: 'HIS-P' },
      { position: 9, protonationType: 'ASP-P' },
      { position: 11, protonationType: 'GLU-P' },
    ]);
    expect(resolved.warnings).toEqual([]);
    expect(resolveProtonPottsCentres(residues, '').centres).toEqual([]);
  });

  it('drops duplicate centres and discloses a centre on a different parent', () => {
    const dup = resolveProtonPottsCentres(residues, 'A6=HIS-P A6=HIS-P');
    expect(dup.centres).toHaveLength(1);
    expect(dup.warnings.join(' ')).toMatch(/listed twice/);

    const grafted = resolveProtonPottsCentres(residues, 'A1=HIS-P');
    expect(grafted.centres[0]).toMatchObject({ position: 0, resName: 'ALA', protonationType: 'HIS-P' });
    expect(grafted.warnings.join(' ')).toMatch(/sits on a ALA/);
  });
});

// ---------------------------------------------------------------------------
// 5. Fixed positions & the designable set
// ---------------------------------------------------------------------------

describe('fixed positions, inherited locks & the designable set', () => {
  const twoChain = (() => {
    const a = helixPdb(4, 'A', { resNames: ['ALA', 'HIS', 'GLU', 'ASP'] });
    const b = helixPdb(3, 'B', { serial: a.nextSerial, offset: 30 });
    return `${a.text}\n${b.text}\n`;
  })();
  const residues = buildProtonPottsInput(twoChain).residues;

  it('resolves explicit residues, ranges and bare-chain selectors', () => {
    expect([...resolveProtonPottsConstraints(residues, 'A2').fixedPositions]).toEqual([1]);
    expect([...resolveProtonPottsConstraints(residues, 'A2-3').fixedPositions]).toEqual([1, 2]);
    const chain = resolveProtonPottsConstraints(residues, 'B');
    expect([...chain.fixedPositions]).toEqual([4, 5, 6]);
    expect(chain.lockedChains).toEqual(['B']);
    expect(resolveProtonPottsConstraints(residues, '').fixedPositions.size).toBe(0);
  });

  it('inherits locked chains and 1-based motif residue indices from upstream generators', () => {
    const inherited = resolveProtonPottsConstraints(residues, '', {
      lockedChains: ['B'],
      fixedResidueIndices: [1, 3],
    } as Pick<StructureArtifact, 'lockedChains' | 'fixedResidueIndices'>);
    expect([...inherited.fixedPositions].sort((a, b) => a - b)).toEqual([0, 2, 4, 5, 6]);
    expect(inherited.lockedChains).toEqual(['B']);
  });

  it('ignores selectors for chains and residues that are not present', () => {
    const ghost = resolveProtonPottsConstraints(residues, 'Z1 C', {
      lockedChains: ['Q'],
      fixedResidueIndices: [99],
    } as Pick<StructureArtifact, 'lockedChains' | 'fixedResidueIndices'>);
    expect(ghost.fixedPositions.size).toBe(0);
    expect(ghost.lockedChains).toEqual([]);
  });

  it('subtracts fixed positions and pH centres from the designable set', () => {
    expect(
      resolveDesignableSet({
        candidates: [3, 1, 2, 0, 4],
        fixed: new Set([1]),
        centres: [4],
        max: 0,
      }),
    ).toEqual([0, 2, 3]);
    // Duplicates and non-integers never reach the runtime.
    expect(
      resolveDesignableSet({
        candidates: [2, 2, 1.5, 0],
        fixed: new Set(),
        centres: [],
        max: 0,
      }),
    ).toEqual([0, 2]);
  });

  it('caps the designable set after exclusions, preferring the ranking hook', () => {
    const candidates = [0, 1, 2, 3, 4, 5, 6];
    // Without a hook the cap keeps the lowest sequence positions.
    expect(
      resolveDesignableSet({ candidates, fixed: new Set([0]), centres: [6], max: 3 }),
    ).toEqual([1, 2, 3]);
    // With the runtime's kNN ranking the cap keeps the positions the ranking puts first,
    // re-sorted into sequence order, and the hook never sees an excluded position.
    const seen: number[][] = [];
    expect(
      resolveDesignableSet({
        candidates,
        fixed: new Set([0]),
        centres: [6],
        max: 3,
        rank: (positions) => {
          seen.push([...positions]);
          return [5, 2, 4, 1, 3];
        },
      }),
    ).toEqual([2, 4, 5]);
    expect(seen).toEqual([[1, 2, 3, 4, 5]]);
    // A cap of 0 (or one the set does not reach) leaves the set untouched.
    expect(resolveDesignableSet({ candidates, fixed: new Set(), centres: [], max: 0 })).toEqual(candidates);
    expect(resolveDesignableSet({ candidates, fixed: new Set(), centres: [], max: 99 })).toEqual(candidates);
  });
});

// ---------------------------------------------------------------------------
// 6. Score decomposition, candidates, SequenceSet & ResidueTrack
// ---------------------------------------------------------------------------

describe('score decomposition & artifact assembly', () => {
  const residues = buildProtonPottsInput(TITRATABLE_PDB).residues;
  const layout = chainLayoutOf(residues);

  const scoreOf = (over: Partial<ProtonPottsScore> = {}): ProtonPottsScore => ({
    globalEnergy: -140,
    energyPerResidue: -10,
    selectiveGapSum: -4.5,
    centres: [],
    perResidueEnergy: Float32Array.from({ length: 14 }, (_, i) => -1 - i / 10),
    ...over,
  });

  const sampleOf = (over: Partial<ProtonPottsSample> = {}): ProtonPottsSample => ({
    sequence: 'ALEKSHVIADLESH',
    tokenNames: residues.map((r) => r.defaultToken),
    score: scoreOf(),
    sampleIndex: 0,
    seed: 42,
    rounds: 3,
    converged: true,
    nChanges: 7,
    objective: -12.5,
    mutations: 5,
    chainIds: layout.chainIds,
    chainLengths: layout.chainLengths,
    ...over,
  });

  it('pins centres and splits each gap into its protonated and deprotonated halves', () => {
    const seq = Int32Array.from({ length: 4 }, () => 0);
    const centres = [
      { position: 1, protonationType: 'HIS-P', protIdx: 21, depIdxs: [22] },
      { position: 3, protonationType: 'ASP-P', protIdx: 24, depIdxs: [25, 26] },
    ];
    expect([...pinCentres(seq, centres)]).toEqual([0, 21, 0, 24]);

    const rows: Record<number, number[]> = {
      1: Object.assign(new Array(30).fill(0), { 21: -3, 22: 1 }),
      3: Object.assign(new Array(30).fill(0), { 24: -1, 25: 2, 26: 4 }),
    };
    const reports = buildCentreReports(
      [
        { position: 1, residueId: 'A2', resName: 'HIS', protonationType: 'HIS-P', contrastTokens: ['HIS-S'] },
        { position: 3, residueId: 'A4', resName: 'ASP', protonationType: 'ASP-P', contrastTokens: ['ASP-D'] },
      ],
      centres,
      (position) => rows[position],
    );
    expect(reports[0]).toEqual({
      position: 1,
      residueId: 'A2',
      resName: 'HIS',
      protonationType: 'HIS-P',
      contrastTokens: ['HIS-S'],
      energyProtonated: -3,
      energyDeprotonated: 1,
      gap: -4,
    });
    // Two contrast partners are averaged, as `selectiveGapOf` does.
    expect(reports[1]).toMatchObject({ energyProtonated: -1, energyDeprotonated: 3, gap: -4 });
    expect(formatCentreReport(reports[0])).toBe('A2 HIS-P ΔE -4.000');
  });

  it('pulls per-residue site energies out of an [L, V] candidate table', () => {
    const L = 3;
    const V = 4;
    const table = Float32Array.from([
      0, 1, 2, 3, //
      10, 11, 12, 13, //
      20, 21, 22, 23,
    ]);
    expect([...siteEnergies(table, [3, 0, 2], L, V)]).toEqual([3, 10, 22]);
  });

  it('scores candidates by per-residue energy and carries ΔE as the global score', () => {
    const parent = stub();
    const options = protonPottsOptionsFromParams({ temperature: 0.3 });
    const candidate = buildProtonPottsCandidate(sampleOf(), parent, options);
    expect(candidate).toMatchObject({
      id: 'proton-pottsmpnn-1',
      sourceStructureId: 'struct-1',
      sourceModelId: 'proton-pottsmpnn',
      sequence: 'ALEKSHVIADLESH',
      score: -10,
      globalScore: -4.5,
      temperature: 0.3,
      seed: 42,
      chainIds: ['A'],
      chainLengths: [14],
    });
    // No pH centre declared -> no selective gap to report.
    expect(
      buildProtonPottsCandidate(
        sampleOf({ score: scoreOf({ selectiveGapSum: null }) }),
        parent,
        options,
      ).globalScore,
    ).toBeNull();
    // Scores are rounded, never left at full float noise.
    expect(
      buildProtonPottsCandidate(
        sampleOf({ score: scoreOf({ energyPerResidue: -1 / 3 }) }),
        parent,
        options,
      ).score,
    ).toBe(-0.333333);
  });

  it('carries design lineage through for multi-backbone collections', () => {
    const candidate = buildProtonPottsCandidate(
      sampleOf({ sampleIndex: 2 }),
      stub(),
      protonPottsOptionsFromParams({}),
      'd3-proton-pottsmpnn-d',
      { designKey: 'design-3', designIndex: 2 },
    );
    expect(candidate.id).toBe('d3-proton-pottsmpnn-d-3');
    expect(candidate).toMatchObject({ designKey: 'design-3', designIndex: 2, sampleIndex: 2 });
  });

  it('orders candidates by the frozen-z descent objective', () => {
    expect(protonPottsObjective({ pottsEnergy: -100, selectiveEnergy: -20 }, { wH: 0.5, wSel: 2 })).toBe(
      -90,
    );
    // λ = 0 freezes the selectivity weight at 0, so the objective is pure stability.
    expect(protonPottsObjective({ pottsEnergy: -100, selectiveEnergy: 7 }, { wH: 0.1, wSel: 0 })).toBe(-10);

    const ordered = orderProtonPottsCandidates([
      { objective: -1, sampleIndex: 0 },
      { objective: -9, sampleIndex: 1 },
      { objective: null, sampleIndex: 2 },
      { objective: -9, sampleIndex: 3 },
      { objective: Number.NaN, sampleIndex: 4 },
    ]);
    expect(ordered.map((s) => s.sampleIndex)).toEqual([1, 3, 0, 2, 4]);
  });

  it('emits a SequenceSet with lineage, locks and inherited chain roles', () => {
    const parent = stub({ chainRoles: { A: 'binder' } });
    const options = protonPottsOptionsFromParams({ seed: 7 });
    const hash = protonPottsRunHash(parent, options, new Set([1, 0]), [2, 3]);
    expect(hash).toMatch(/^[0-9a-f]+$/);
    // The hash covers the resolved constraints, not just the raw params.
    expect(protonPottsRunHash(parent, options, new Set([0, 1]), [2, 3])).toBe(hash);
    expect(protonPottsRunHash(parent, options, new Set([0]), [2, 3])).not.toBe(hash);
    expect(protonPottsRunHash(parent, options, new Set([0, 1]), [2, 4])).not.toBe(hash);

    const set = buildProtonPottsSequenceSet(
      parent,
      [buildProtonPottsCandidate(sampleOf(), parent, options)],
      options,
      ['B'],
      hash,
    );
    expect(set).toMatchObject({
      kind: 'SequenceSet',
      id: `proton-pottsmpnn-seq-struct-1-7-${hash}`,
      parentStructureId: 'struct-1',
      sourceStructureId: 'struct-1',
      sourceModelId: 'proton-pottsmpnn',
      lockedChains: ['B'],
      chainRoles: { A: 'binder' },
    });
    expect(set.candidates).toHaveLength(1);

    const multi = buildProtonPottsSequenceSet(parent, [], options, [], hash, {
      sourceStructureIds: ['a', 'b'],
      collectionKind: 'independent-candidates',
    });
    expect(multi).toMatchObject({
      sourceStructureIds: ['a', 'b'],
      collectionKind: 'independent-candidates',
    });
    expect(multi.lockedChains).toBeUndefined();
  });

  it('labels the ResidueTrack differently for design and score runs', () => {
    const parent = stub();
    const design = buildProtonPottsTrack(
      parent,
      residues,
      sampleOf(),
      protonPottsOptionsFromParams({ precision: 'int8' }),
      'abc123',
    );
    expect(design).toMatchObject({
      kind: 'ResidueTrack',
      id: 'proton-pottsmpnn-track-design-int8-abc123',
      sourceModelId: 'proton-pottsmpnn',
      parentStructureId: 'struct-1',
      quantity: 'potts-site-energy',
      unit: 'a.u.',
      residues: 'ALEKSHVIADLESH',
    });
    expect(design.label).toMatch(/^Proton-PottsMPNN site energy \(best design · mean -1\.650\)$/);
    expect(design.residueIds).toEqual(residueIdsOf(residues));
    expect(design.values).toHaveLength(14);
    expect(design.summary!.mean).toBeCloseTo(-1.65, 6);
    expect(design.summary!.min!).toBeCloseTo(-2.3, 6);
    expect(design.summary!.max).toBeCloseTo(-1, 6);

    const score = buildProtonPottsTrack(
      parent,
      residues,
      sampleOf(),
      protonPottsOptionsFromParams({ mode: 'score' }),
      'abc123',
    );
    expect(score.id).toBe('proton-pottsmpnn-track-score-fp16-abc123');
    expect(score.label).toContain('native');

    // A fully non-finite track still produces a well-formed artifact.
    const empty = buildProtonPottsTrack(
      parent,
      residues,
      sampleOf({ score: scoreOf({ perResidueEnergy: new Float32Array(14).fill(Number.NaN) }) }),
      protonPottsOptionsFromParams({}),
      'abc123',
    );
    expect(Number.isNaN(empty.summary!.mean)).toBe(true);
    expect(Number.isNaN(empty.summary!.min!)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 7. Structure-collection fan-out
// ---------------------------------------------------------------------------

describe('structure-collection fan-out', () => {
  const item = (id: string, over: Partial<StructureArtifact> = {}): StructureArtifact =>
    stub({ id, ...over });

  it('treats a single structure as one job', () => {
    const single = item('s1');
    expect(structureItems(single)).toEqual([single]);
    expect(planProtonPottsJobs(single)).toEqual([{ parent: single, designIndex: 0 }]);
  });

  it('emits one job per collection item, preserving design keys and ordinals', () => {
    const items = [
      item('s1', { designKey: 'design-1', designIndex: 0 }),
      item('s2', { designKey: 'design-2', designIndex: 1 }),
      item('s3', { designKey: 'design-3', designIndex: 2 }),
    ];
    const collection = stub({
      id: 'coll',
      collection: { items, keys: ['design-1', 'design-2', 'design-3'] },
    } as Partial<StructureArtifact>);
    const jobs = planProtonPottsJobs(collection);
    expect(jobs).toHaveLength(3);
    expect(jobs.map((j) => j.parent.id)).toEqual(['s1', 's2', 's3']);
    expect(jobs.map((j) => j.designKey)).toEqual(['design-1', 'design-2', 'design-3']);
    expect(jobs.map((j) => j.designIndex)).toEqual([0, 1, 2]);
  });

  it('falls back to collection keys when items carry no designKey', () => {
    const items = [item('s1'), item('s2')];
    const collection = stub({ id: 'coll', collection: { items, keys: ['k1', 'k2'] } } as Partial<StructureArtifact>);
    const jobs = planProtonPottsJobs(collection);
    expect(jobs.map((j) => j.designKey)).toEqual(['k1', 'k2']);
    expect(jobs.map((j) => j.designIndex)).toEqual([0, 1]);

    const unkeyed = stub({ id: 'coll2', collection: { items } } as Partial<StructureArtifact>);
    expect(planProtonPottsJobs(unkeyed).map((j) => j.designKey)).toEqual(['s1', 's2']);
  });

  it('collapses a one-item collection onto the parent structure', () => {
    const only = item('s1');
    const collection = stub({ id: 'coll', collection: { items: [only], keys: ['k1'] } } as Partial<StructureArtifact>);
    expect(planProtonPottsJobs(collection)).toEqual([{ parent: collection, designIndex: 0 }]);
  });
});

// ---------------------------------------------------------------------------
// 8. End-to-end CPU smoke test on the vendored runtime
// ---------------------------------------------------------------------------

/** Minimal little-endian safetensors writer (F32 only) for the synthetic bundle. */
function writeSafetensors(tensors: Array<{ name: string; shape: number[]; data: Float32Array }>): Uint8Array {
  const header: Record<string, unknown> = {};
  let offset = 0;
  for (const t of tensors) {
    header[t.name] = { dtype: 'F32', shape: t.shape, data_offsets: [offset, offset + t.data.byteLength] };
    offset += t.data.byteLength;
  }
  let json = new TextEncoder().encode(JSON.stringify(header));
  // safetensors requires the data block to start 8-byte aligned.
  const pad = (-(json.length + 8) % 8 + 8) % 8;
  if (pad > 0) {
    const padded = new Uint8Array(json.length + pad);
    padded.set(json);
    padded.fill(0x20, json.length);
    json = padded;
  }
  const out = new Uint8Array(8 + json.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  let cursor = 8 + json.length;
  for (const t of tensors) {
    out.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), cursor);
    cursor += t.data.byteLength;
  }
  return out;
}

const SYNTH = { H: 8, FF: 16, V: 30, K: 8 } as const;

/** A tiny but architecturally valid v6 bundle: 1 encoder layer, no decoder, real 30-token head. */
function syntheticBundle(seed = 7): { manifest: Record<string, unknown>; file: Uint8Array } {
  let state = seed >>> 0;
  const rand = (): number => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const { H, FF, V } = SYNTH;
  const tensors: Array<{ name: string; shape: number[]; data: Float32Array }> = [];
  const add = (name: string, shape: number[], scale = 0.3, fill?: number): void => {
    const n = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(n);
    if (fill !== undefined) data.fill(fill);
    else for (let i = 0; i < n; i++) data[i] = (rand() * 2 - 1) * scale;
    tensors.push({ name, shape, data });
  };
  add('graph_featurization_module.positional_embedding.embed_positional_features.weight', [16, 66]);
  add('graph_featurization_module.positional_embedding.embed_positional_features.bias', [16], 0.1);
  add('graph_featurization_module.edge_embedding.weight', [H, 416], 0.1);
  add('graph_featurization_module.edge_norm.weight', [H], 0, 1);
  add('graph_featurization_module.edge_norm.bias', [H], 0, 0);
  add('W_e.weight', [H, H]);
  add('W_e.bias', [H], 0.05);
  for (const norm of ['norm1', 'norm2', 'norm3']) {
    add(`encoder_layers.0.${norm}.weight`, [H], 0, 1);
    add(`encoder_layers.0.${norm}.bias`, [H], 0, 0);
  }
  for (const w of ['W1', 'W11']) {
    add(`encoder_layers.0.${w}.weight`, [H, 3 * H]);
    add(`encoder_layers.0.${w}.bias`, [H], 0.05);
  }
  for (const w of ['W2', 'W3', 'W12', 'W13']) {
    add(`encoder_layers.0.${w}.weight`, [H, H]);
    add(`encoder_layers.0.${w}.bias`, [H], 0.05);
  }
  add('encoder_layers.0.dense.W_in.weight', [FF, H]);
  add('encoder_layers.0.dense.W_in.bias', [FF], 0.05);
  add('encoder_layers.0.dense.W_out.weight', [H, FF]);
  add('encoder_layers.0.dense.W_out.bias', [H], 0.05);
  add('etab_out.weight', [V * V, H], 1.0);
  add('etab_out.bias', [V * V], 0.5);

  const file = writeSafetensors(tensors);
  return {
    manifest: {
      model: 'proton-pottsmpnn-v6-synthetic',
      format: 'safetensors',
      arch: {
        hidden_dim: H,
        vocab_size: V,
        potts_vocab_size: V,
        num_encoder_layers: 1,
        num_decoder_layers: 0,
        num_neighbors: SYNTH.K,
        num_edge_input_features: 416,
        num_positional_embeddings: 16,
        num_positional_features: 66,
        max_relative_feature: 32,
        num_rbf: 16,
        min_rbf_mean: 2.0,
        max_rbf_mean: 22.0,
        ffn_dim: FF,
        message_scale: 30.0,
        etab_source: 'edge',
        field_source: 'self_edge',
        has_decoder: false,
      },
      vocab: { name: 'v6', tokens: [...V6_TOKENS] },
      precisions: {
        fp32: {
          file: 'potts_v6-fp32.safetensors',
          bytes: file.byteLength,
          sha256: '',
          n_tensors: tensors.length,
        },
      },
      n_parameters: tensors.reduce((a, t) => a + t.data.length, 0),
      tensors: tensors.map((t) => ({ name: t.name, shape: t.shape })),
    },
    file,
  };
}

describe('end-to-end CPU smoke test on the vendored Proton-PottsMPNN runtime', () => {
  const bundle = syntheticBundle();
  const requested: string[] = [];
  const read = async (url: string): Promise<Uint8Array> => {
    requested.push(url);
    if (url.endsWith('manifest.json')) {
      return new TextEncoder().encode(JSON.stringify(bundle.manifest));
    }
    if (url.endsWith('potts_v6-fp32.safetensors')) return bundle.file;
    throw new Error(`unexpected weight URL ${url}`);
  };

  it('exports every symbol the worker imports from the vendored modules', () => {
    for (const name of [
      'structureFromPDB',
      'parseBackbonePDB',
      'buildStructure',
      'featurize',
      'pottsEnergy',
      'pottsCandidateEnergies',
      'pottsCandidateEnergiesAt',
      'makePottsTables',
      'resolveCentres',
      'selectiveReport',
      'selectiveGapSum',
      'centreNeighbourhood',
      'blockDescent',
      'greedyCoordinateDescent',
      'knnRank',
      'validTokenMask',
      'canonicalSequence',
      'tokenNames',
      'makeRng',
      'V6_TOKENS',
      'TOKEN_TO_IDX',
      'V6_DEP_MAP',
      'PHYSIOLOGICAL_STATE',
      'DEFAULT_FORBIDDEN_TOKENS',
    ]) {
      expect(shared[name], `shared/index.js must export ${name}`).toBeDefined();
    }
    expect(cpuMod.createProtonPottsCPU).toBeTypeOf('function');
    expect(cpuMod.loadBundle).toBeTypeOf('function');
    expect([...cpuMod.PRECISIONS]).toEqual(['fp32', 'fp16', 'int8']);
    // The plugin's precision tiers are exactly the runtime's bundles.
    expect(Object.keys(PROTON_POTTSMPNN_WEIGHTS_MB).sort()).toEqual([...cpuMod.PRECISIONS].sort());
    expect(gpuMod.createProtonPottsWebGPU).toBeTypeOf('function');
    expect(gpuMod.requestDevice).toBeTypeOf('function');
  });

  it('loads a bundle through the injected reader and exposes the worker’s backend surface', async () => {
    const model = await cpuMod.createProtonPottsCPU({ baseUrl: 'mem://w', precision: 'fp32', read });
    try {
      expect(requested).toEqual(['mem://w/manifest.json', 'mem://w/potts_v6-fp32.safetensors']);
      expect(model.backend).toBe('cpu');
      expect(model.precision).toBe('fp32');
      expect(model.arch.numNeighbors).toBe(SYNTH.K);
      expect(model.arch.pottsVocabSize).toBe(30);
      // Exactly the methods `proton-pottsmpnn.worker.ts :: PottsBackend` calls.
      for (const method of [
        'structureFromPDB',
        'run',
        'resolveCentres',
        'selectiveGap',
        'candidateEnergies',
        'centreNeighbourhood',
        'optimize',
        'destroy',
      ]) {
        expect(model[method], `handle must expose ${method}`).toBeTypeOf('function');
      }
      await expect(
        cpuMod.createProtonPottsCPU({ baseUrl: 'mem://w', precision: 'bf16', read }),
      ).rejects.toThrow(/precision must be one of/);
    } finally {
      model.destroy();
    }
  });

  it('featurizes a backbone into the exact tensors the Potts head consumes', async () => {
    const model = await cpuMod.createProtonPottsCPU({ baseUrl: 'mem://w', precision: 'fp32', read });
    try {
      const input = buildProtonPottsInput(TITRATABLE_PDB);
      const structure = model.structureFromPDB(TITRATABLE_PDB);

      // The plugin's residue list and the runtime's tensors must agree position for position —
      // the invariant `proton-pottsmpnn.worker.ts` asserts before it designs anything.
      expect(structure.L).toBe(input.residues.length);
      expect([...structure.resSeq]).toEqual(input.residues.map((r) => r.number));
      expect(structure.resName).toEqual(input.residues.map((r) => r.resName));
      expect(structure.tokenNames).toEqual(input.residues.map((r) => r.defaultToken));
      expect(structure.tokenNames[5]).toBe('HIS-S');

      // Positional overrides from `protonationOverrides` land on exactly those residues.
      const overrides = resolveProtonationOverrides(input.residues, 'A6=HIS-P, A10=ASP-P').overrides;
      const pinned = model.structureFromPDB(TITRATABLE_PDB, { overrides });
      expect(pinned.tokenNames[5]).toBe('HIS-P');
      expect(pinned.tokenNames[9]).toBe('ASP-P');
      expect(pinned.tokenNames[13]).toBe('HIS-S');
      expect(() => model.structureFromPDB(TITRATABLE_PDB, { overrides: { 5: 'HIS-Q' } })).toThrow(
        /not a v6 token/,
      );

      const out = model.run(structure);
      expect(out.K).toBe(Math.min(SYNTH.K, structure.L));
      expect(out.V).toBe(30);
      expect(out.J).toHaveLength(structure.L * out.K * 30 * 30);
      expect(out.h).toHaveLength(structure.L * 30);
      expect(Number.isFinite(out.energy)).toBe(true);
      for (let i = 0; i < structure.L; i++) {
        expect(out.edgeIdx[i * out.K], 'slot 0 is the self edge').toBe(i);
        // The self edge is diagonal and its diagonal is the single-body field.
        const base = i * out.K * 900;
        for (let a = 0; a < 30; a++) {
          expect(out.J[base + a * 30 + a]).toBe(out.h[i * 30 + a]);
          if (a > 0) expect(out.J[base + a * 30 + (a - 1)]).toBe(0);
        }
      }
      // `E_global` really is the Hamiltonian of the structure's own tokens.
      expect(model.pottsEnergy(out, structure.tokens)).toBeCloseTo(out.energy, 10);
    } finally {
      model.destroy();
    }
  });

  it('scores the native sequence: E_global, per-centre E_P/E_D/ΔE and site energies', async () => {
    const model = await cpuMod.createProtonPottsCPU({ baseUrl: 'mem://w', precision: 'fp32', read });
    try {
      const input = buildProtonPottsInput(TITRATABLE_PDB);
      const resolved = resolveProtonPottsCentres(input.residues, 'A6=HIS-P, A10=ASP-P');
      const structure = model.structureFromPDB(TITRATABLE_PDB);
      const out = model.run(structure);

      const centres = model.resolveCentres(structure, resolved.specs);
      expect(centres.map((c: any) => c.position)).toEqual([5, 9]);
      expect(centres[0].protIdx).toBe(shared.TOKEN_TO_IDX['HIS-P']);
      expect(centres[0].depIdxs).toEqual([shared.TOKEN_TO_IDX['HIS-S']]);
      expect(centres[1].depIdxs).toEqual([shared.TOKEN_TO_IDX['ASP-D']]);

      // The plugin's E_P / E_D decomposition must reproduce the runtime's own gap exactly.
      const vendorReport = model.selectiveGap(out, centres, structure.tokens);
      const pinnedSeq = pinCentres(structure.tokens, centres);
      const row = new Float64Array(out.V);
      const reports = buildCentreReports(resolved.centres, centres, (position) =>
        shared.pottsCandidateEnergiesAt(out.tables, pinnedSeq, position, row),
      );
      expect(reports.map((r) => r.residueId)).toEqual(['A6', 'A10']);
      expect(reports.map((r) => r.protonationType)).toEqual(['HIS-P', 'ASP-P']);
      expect(reports.map((r) => r.contrastTokens)).toEqual([['HIS-S'], ['ASP-D']]);
      reports.forEach((r, i) => {
        expect(r.gap).toBe(vendorReport.centres[i].gap);
        expect(r.energyProtonated - r.energyDeprotonated).toBeCloseTo(r.gap, 12);
      });
      expect(reports.reduce((a, r) => a + r.gap, 0)).toBeCloseTo(vendorReport.gapSum, 12);

      // Site energies obey the runtime's own conditional-energy identity.
      const candidates = model.candidateEnergies(out, structure.tokens);
      const site = siteEnergies(candidates, structure.tokens, out.L, out.V);
      expect(site).toHaveLength(out.L);
      for (const v of site) expect(Number.isFinite(v)).toBe(true);
      const probe = 4;
      const mutated = Int32Array.from(structure.tokens);
      mutated[probe] = shared.TOKEN_TO_IDX.TRP;
      const deltaH = model.pottsEnergy(out, mutated) - model.pottsEnergy(out, structure.tokens);
      expect(deltaH).toBeCloseTo(
        candidates[probe * out.V + shared.TOKEN_TO_IDX.TRP] - site[probe],
        4,
      );

      // With no centre there is simply no gap to report.
      expect(model.resolveCentres(structure, [])).toEqual([]);
      expect(buildCentreReports([], [], () => [])).toEqual([]);
    } finally {
      model.destroy();
    }
  });

  it('designs with pH-selective block descent: pinned centres, honoured masks, frozen z-scales', async () => {
    const model = await cpuMod.createProtonPottsCPU({ baseUrl: 'mem://w', precision: 'fp32', read });
    try {
      const input = buildProtonPottsInput(TITRATABLE_PDB);
      const options = protonPottsOptionsFromParams({
        protonationCentres: 'A6=HIS-P, A10=ASP-P',
        lambda: 0.5,
        blockSize: 2,
        maxDesignable: 6,
        shellPerCentre: 4,
        omitAAs: 'C',
        titratableTokens: 'explicit-states',
      });
      const resolved = resolveProtonPottsCentres(input.residues, options.protonationCentres);
      const alphabet = resolveProtonPottsAlphabet(options.omitAAs, options.titratableTokens);
      const { fixedPositions } = resolveProtonPottsConstraints(input.residues, 'A1-2');

      const structure = model.structureFromPDB(TITRATABLE_PDB);
      const out = model.run(structure);
      const centres = model.resolveCentres(structure, resolved.specs);

      const shell = model.centreNeighbourhood(out, centres, {
        perCentre: options.shellPerCentre,
        max: 0,
      });
      expect(shell.length).toBeGreaterThan(0);
      expect(shell).not.toContain(5); // a centre is never designable
      const designable = resolveDesignableSet({
        candidates: shell,
        fixed: fixedPositions,
        centres: centres.map((c: any) => c.position),
        max: options.maxDesignable,
        rank: (positions) => shared.knnRank(out.tables, centres, [...positions]),
      });
      expect(designable.length).toBeGreaterThan(0);
      expect(designable.length).toBeLessThanOrEqual(options.maxDesignable);
      expect(designable).not.toContain(0);
      expect(designable).not.toContain(1);
      expect(designable).not.toContain(5);
      expect(designable).not.toContain(9);

      const descentOptions = {
        centres,
        designable,
        lambda: options.lambda,
        blockSize: options.blockSize,
        zscaleMode: 'block',
        sweepOrder: options.sweepOrder,
        temperature: 0,
        maxRounds: options.maxRounds,
        forbiddenTokens: [...alphabet.forbiddenTokens],
      };
      const steps: number[] = [];
      const design = model.optimize(out, {
        ...descentOptions,
        rng: shared.makeRng(options.seed),
        onStep: (step: { round: number }) => steps.push(step.round),
      });

      expect(design.tokens).toHaveLength(structure.L);
      expect(design.sequence).toHaveLength(structure.L);
      expect(design.tokenNames).toHaveLength(structure.L);
      expect(design.rounds).toBeGreaterThan(0);
      expect(design.rounds).toBeLessThanOrEqual(options.maxRounds);
      expect(design.converged).toBe(true);
      expect(design.nChanges).toBeGreaterThan(0);
      expect(steps.length).toBe(design.nChanges); // the worker's only cancellation hook fires

      // Centres stay pinned protonated; everything outside the designable set is untouched.
      for (const c of centres) expect(design.tokens[c.position]).toBe(c.protIdx);
      for (let i = 0; i < structure.L; i++) {
        if (designable.includes(i) || centres.some((c: any) => c.position === i)) continue;
        expect(design.tokens[i], `position ${i} must be untouched`).toBe(structure.tokens[i]);
      }
      // The forbidden-token mask is honoured at every designable position.
      const forbidden = new Set([...alphabet.forbiddenTokens, 'UNK']);
      for (const p of designable) {
        expect(forbidden.has(V6_TOKENS[design.tokens[p]])).toBe(false);
      }

      // The frozen z-scales are what make λ a relative weight; both weights must be live.
      expect(design.zscales.sdH).toBeGreaterThan(0);
      expect(design.zscales.sdSel).toBeGreaterThan(0);
      expect(design.weights.wH).toBeCloseTo((1 - options.lambda) / design.zscales.sdH, 12);
      expect(design.weights.wSel).toBeCloseTo(options.lambda / design.zscales.sdSel, 12);
      expect(
        protonPottsObjective(
          { pottsEnergy: design.pottsEnergy, selectiveEnergy: design.selectiveEnergy },
          design.weights,
        ),
      ).toBeCloseTo(
        design.weights.wH * design.pottsEnergy + design.weights.wSel * design.selectiveEnergy,
        12,
      );

      // Temp 0 is an exact argmin, so the seed cannot matter — which is why the worker runs one
      // descent instead of `numSequences` identical ones.
      const repeat = model.optimize(out, { ...descentOptions, rng: shared.makeRng(options.seed + 123) });
      expect(repeat.sequence).toBe(design.sequence);
      expect([...repeat.tokens]).toEqual([...design.tokens]);

      // Above Temp 0 the block is sampled: reproducible per seed, different across seeds.
      const sampled = (seed: number) =>
        model.optimize(out, {
          ...descentOptions,
          blockSize: 1,
          temperature: 1.0,
          maxRounds: 3,
          rng: shared.makeRng(seed),
        });
      expect(sampled(1).sequence).toBe(sampled(1).sequence);
      expect(sampled(1).sequence).not.toBe(sampled(2).sequence);

      // Pure selectivity may not worsen the gap it is optimizing.
      const before = shared.selectiveGapSum(out.tables, structure.tokens, centres);
      const pure = model.optimize(out, {
        ...descentOptions,
        lambda: 1,
        blockSize: 1,
        maxRounds: 4,
        rng: shared.makeRng(0),
      });
      expect(pure.selectiveEnergy).toBeLessThanOrEqual(before + 1e-6);
      expect(pure.weights.wH).toBe(0);

      // Pure stability needs no centre at all, and then the selectivity term is exactly zero.
      const stability = model.optimize(out, {
        centres: [],
        designable,
        lambda: 0,
        blockSize: 1,
        zscaleMode: 'block',
        temperature: 0,
        maxRounds: 4,
        forbiddenTokens: [...resolveProtonPottsAlphabet('').forbiddenTokens],
        rng: shared.makeRng(options.seed),
      });
      expect(stability.selectiveEnergy).toBe(0);
      expect(stability.weights.wSel).toBe(0);
      expect(stability.pottsEnergy).toBeLessThanOrEqual(out.energy + 1e-9);

      // `greedyCoordinateDescent` is `blockSize: 1`, the same knob the plugin exposes.
      const greedy = model.optimizeGreedy(out, { ...descentOptions, rng: shared.makeRng(5) });
      const block1 = model.optimize(out, { ...descentOptions, blockSize: 1, rng: shared.makeRng(5) });
      expect([...greedy.tokens]).toEqual([...block1.tokens]);
    } finally {
      model.destroy();
    }
  });

  it('assembles real artifacts from a real descent, in both modes', async () => {
    const model = await cpuMod.createProtonPottsCPU({ baseUrl: 'mem://w', precision: 'fp32', read });
    try {
      const parent = stub();
      const input = buildProtonPottsInput(inputPdbText(parent));
      const layout = chainLayoutOf(input.residues);
      const resolved = resolveProtonPottsCentres(input.residues, 'A6=HIS-P');
      const structure = model.structureFromPDB(TITRATABLE_PDB);
      const out = model.run(structure);
      const centres = model.resolveCentres(structure, resolved.specs);
      const designable = resolveDesignableSet({
        candidates: model.centreNeighbourhood(out, centres, { perCentre: 4, max: 0 }),
        fixed: new Set<number>(),
        centres: centres.map((c: any) => c.position),
        max: 5,
        rank: (positions) => shared.knnRank(out.tables, centres, [...positions]),
      });

      const makeScore = (seq: ArrayLike<number>, globalEnergy: number): ProtonPottsScore => {
        const pinnedSeq = pinCentres(seq, centres);
        const row = new Float64Array(out.V);
        const reports = buildCentreReports(resolved.centres, centres, (position) =>
          shared.pottsCandidateEnergiesAt(out.tables, pinnedSeq, position, row),
        );
        return {
          globalEnergy,
          energyPerResidue: globalEnergy / out.L,
          selectiveGapSum: model.selectiveGap(out, centres, seq).gapSum,
          centres: reports,
          perResidueEnergy: siteEnergies(model.candidateEnergies(out, seq), seq, out.L, out.V),
        };
      };

      // --- score mode ---
      const scoreOptions = protonPottsOptionsFromParams({
        mode: 'score',
        protonationCentres: 'A6=HIS-P',
      });
      const nativeSample: ProtonPottsSample = {
        sequence: shared.canonicalSequence(structure.tokens),
        tokenNames: shared.tokenNames(structure.tokens),
        score: makeScore(structure.tokens, out.energy),
        sampleIndex: 0,
        seed: scoreOptions.seed,
        rounds: null,
        converged: null,
        nChanges: null,
        objective: null,
        mutations: 0,
        chainIds: layout.chainIds,
        chainLengths: layout.chainLengths,
      };
      expect(nativeSample.sequence).toBe('ALEKSHVIADLESH');
      expect(nativeSample.tokenNames[5]).toBe('HIS-S');
      const scoreCandidate = buildProtonPottsCandidate(
        nativeSample,
        parent,
        scoreOptions,
        'proton-pottsmpnn-native',
      );
      expect(scoreCandidate.id).toBe('proton-pottsmpnn-native-1');
      expect(scoreCandidate.score).toBeCloseTo(out.energy / out.L, 5);
      expect(scoreCandidate.globalScore).toBeCloseTo(
        model.selectiveGap(out, centres, structure.tokens).gapSum,
        5,
      );
      const scoreTrack = buildProtonPottsTrack(parent, input.residues, nativeSample, scoreOptions, 'h1');
      expect(scoreTrack.values).toHaveLength(out.L);
      expect(scoreTrack.residues).toBe(nativeSample.sequence);
      expect(scoreTrack.residueIds).toEqual(residueIdsOf(input.residues));
      expect(scoreTrack.label).toContain('native');

      // --- design mode ---
      const designOptions = protonPottsOptionsFromParams({
        protonationCentres: 'A6=HIS-P',
        lambda: 0.6,
        temperature: 0.8,
        numSequences: 3,
        blockSize: 1,
        maxRounds: 3,
      });
      const samples: ProtonPottsSample[] = [];
      for (let i = 0; i < designOptions.numSequences; i++) {
        const result = model.optimize(out, {
          centres,
          designable,
          lambda: designOptions.lambda,
          blockSize: designOptions.blockSize,
          zscaleMode: 'block',
          sweepOrder: designOptions.sweepOrder,
          temperature: designOptions.temperature,
          maxRounds: designOptions.maxRounds,
          forbiddenTokens: [...resolveProtonPottsAlphabet('').forbiddenTokens],
          rng: shared.makeRng(designOptions.seed + i),
        });
        let mutations = 0;
        for (let q = 0; q < out.L; q++) if (result.tokens[q] !== structure.tokens[q]) mutations++;
        samples.push({
          sequence: result.sequence,
          tokenNames: result.tokenNames,
          score: makeScore(result.tokens, result.pottsEnergy),
          sampleIndex: i,
          seed: designOptions.seed + i,
          rounds: result.rounds,
          converged: result.converged,
          nChanges: result.nChanges,
          objective: protonPottsObjective(
            { pottsEnergy: result.pottsEnergy, selectiveEnergy: result.selectiveEnergy },
            result.weights,
          ),
          mutations,
          chainIds: layout.chainIds,
          chainLengths: layout.chainLengths,
        });
      }
      expect(samples).toHaveLength(3);
      // Every sample keeps the centre pinned, so the designed token there is always HIS-P.
      for (const sample of samples) {
        expect(sample.tokenNames[5]).toBe('HIS-P');
        expect(sample.sequence[5]).toBe('H');
        expect(sample.mutations).toBeGreaterThan(0);
        expect(sample.score.centres).toHaveLength(1);
        expect(sample.score.selectiveGapSum).not.toBeNull();
      }

      const ordered = orderProtonPottsCandidates(samples);
      expect(ordered[0].objective).toBeLessThanOrEqual(ordered[ordered.length - 1].objective!);
      const candidates = ordered.map((sample, idx) =>
        buildProtonPottsCandidate(
          sample,
          parent,
          designOptions,
          'proton-pottsmpnn-d',
          idx === 0 ? {} : {},
        ),
      );
      const runHash = protonPottsRunHash(parent, designOptions, new Set<number>(), designable);
      const set = buildProtonPottsSequenceSet(parent, candidates, designOptions, [], runHash);
      expect(set.candidates).toHaveLength(3);
      expect(new Set(set.candidates.map((c) => c.id)).size).toBe(3);
      for (const candidate of set.candidates) {
        expect(candidate.sequence).toHaveLength(out.L);
        expect(candidate.chainLengths).toEqual([out.L]);
        expect(Number.isFinite(candidate.score!)).toBe(true);
        expect(Number.isFinite(candidate.globalScore!)).toBe(true);
      }
      const designTrack = buildProtonPottsTrack(
        parent,
        input.residues,
        ordered[0],
        designOptions,
        runHash,
      );
      expect(designTrack.label).toContain('best design');
      expect(designTrack.values).toHaveLength(out.L);
      expect(designTrack.summary!.mean).toBeLessThan(0);
      expect(designTrack.id).toBe(`proton-pottsmpnn-track-design-fp16-${runHash}`);
    } finally {
      model.destroy();
    }
  });
});
