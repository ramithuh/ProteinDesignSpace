import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  ADAPTER_REGISTRY,
  MPNN_CHECKPOINTS_BY_FAMILY,
  WORKFLOW_PRESETS,
} from '../engine/registry';
import {
  fieldToNumericSpec,
  getActiveModeSpec,
  getDefaultParamsFromSchema,
  getVisibleParamFields,
  isValidResidueSelectorToken,
  validateParamsWithSchema,
  validateResidueMappingSyntax,
} from '../engine/param-schema';
import {
  defaultParamsForAdapter,
  insertBlockSmart,
  sanitizeParamsOnModeSwitch,
} from '../engine/graph-utils';
import { NUMERIC_SPECS } from '../engine/param-validation';
import { formatNodeParamSummary } from '../ui/node-display-helpers';
import type { AdapterManifest } from '../engine/types';

/**
 * Extract every `p.<key>`, `params.<key>`, or `msg.params.<key>` property read
 * inside a worker's `msg.type === 'run'` handler.
 */
function extractWorkerRunParamKeys(workerRelPath: string): string[] {
  const src = readFileSync(resolve(process.cwd(), workerRelPath), 'utf8');
  let runIdx = src.indexOf("msg.type === 'run'");
  if (runIdx === -1) {
    runIdx = src.search(/\basync\s+run\s*\(/);
  }
  expect(
    runIdx,
    `Expected msg.type === 'run' or createWorkerHarness async run(ctx) handler in ${workerRelPath}`,
  ).toBeGreaterThan(-1);
  const runSlice = src.slice(runIdx);
  const keys = new Set<string>();
  for (const m of runSlice.matchAll(/\b(?:p|params|msg\.params|ctx\.params)\??\.([A-Za-z_][A-Za-z0-9_]*)\b/g)) {
    keys.add(m[1]);
  }
  return [...keys].sort();
}

describe('declarative ParamSchema & modes single source of truth', () => {
  it('declares non-empty unique paramSchema on every ADAPTER_REGISTRY entry and modes on multi-mode adapters', () => {
    for (const [id, manifest] of Object.entries(ADAPTER_REGISTRY)) {
      expect(manifest.paramSchema, `${id} must declare paramSchema`).toBeDefined();
      expect(manifest.paramSchema!.length, `${id}.paramSchema must be non-empty`).toBeGreaterThan(0);

      const keys = manifest.paramSchema!.map((f) => f.key);
      expect(new Set(keys).size, `${id}.paramSchema has duplicate keys`).toBe(keys.length);

      // Every numeric spec in NUMERIC_SPECS must align cleanly with paramSchema bounds & integer kind
      const numSpecs = NUMERIC_SPECS[id];
      if (numSpecs) {
        for (const [key, spec] of Object.entries(numSpecs)) {
          const field = manifest.paramSchema!.find((f) => f.key === key);
          expect(field, `${id}.paramSchema missing NUMERIC_SPECS key "${key}"`).toBeDefined();
          expect(field!.kind).toBe(spec.integer ? 'integer' : 'number');
          expect(field!.min).toBe(spec.min);
          expect(field!.max).toBe(spec.max);
          expect(field!.defaultValue).toBe(spec.defaultValue);
        }
      }
    }

    // Multi-mode adapters must declare modeParamKey + modes with requiredInputs
    for (const id of ['rfd4-proteina', 'genie3', 'localfold-af2', 'pdb-input']) {
      const m = ADAPTER_REGISTRY[id];
      expect(m.modeParamKey, `${id} missing modeParamKey`).toBeTruthy();
      expect(m.modes?.length, `${id} missing modes`).toBeGreaterThan(0);
    }
  });

  it('resolves active AdapterModeSpec via getActiveModeSpec for default, explicit, and unknown modes', () => {
    const rfd4 = ADAPTER_REGISTRY['rfd4-proteina'];
    expect(getActiveModeSpec(rfd4, {})?.id).toBe('unconditional');
    expect(getActiveModeSpec(rfd4, {})?.requiredInputs).toEqual([]);
    expect(getActiveModeSpec(rfd4, { mode: 'binder' })?.id).toBe('binder');
    expect(getActiveModeSpec(rfd4, { mode: 'binder' })?.requiredInputs).toEqual(['structure']);
    expect(getActiveModeSpec(rfd4, { mode: 'nonexistent' })).toBeNull();

    const genie3 = ADAPTER_REGISTRY.genie3;
    expect(getActiveModeSpec(genie3, {})?.id).toBe('unconditional');
    expect(getActiveModeSpec(genie3, { taskMode: 'motif' })?.requiredInputs).toEqual(['structure']);

    const lf = ADAPTER_REGISTRY['localfold-af2'];
    expect(getActiveModeSpec(lf, {})?.id).toBe('single');
    expect(getActiveModeSpec(lf, { msaMode: 'a3m' })?.requiredInputs).toEqual(['sequences']);

    const pdb = ADAPTER_REGISTRY['pdb-input'];
    expect(getActiveModeSpec(pdb, {})?.id).toBe('demo-bundle');
    expect(getActiveModeSpec(pdb, { mode: 'pdb-id' })?.id).toBe('pdb-id');

    // Adapters without modes return null
    expect(getActiveModeSpec(ADAPTER_REGISTRY['web-protein-mpnn'], {})).toBeNull();
    expect(getActiveModeSpec(undefined, {})).toBeNull();
  });

  it('filters visible fields accurately by active mode and tier via getVisibleParamFields', () => {
    const rfd4 = ADAPTER_REGISTRY['rfd4-proteina'];
    const uncondPrimary = getVisibleParamFields(rfd4, { mode: 'unconditional' }, 'primary').map((f) => f.key);
    expect(uncondPrimary).toContain('length');
    expect(uncondPrimary).toContain('cathCode');
    expect(uncondPrimary).not.toContain('partialFraction');
    expect(uncondPrimary).not.toContain('hotspots');
    expect(uncondPrimary).not.toContain('presetId');

    const partialPrimary = getVisibleParamFields(rfd4, { mode: 'partial' }, 'primary').map((f) => f.key);
    expect(partialPrimary).toContain('partialFraction');
    expect(partialPrimary).not.toContain('length');

    const binderPrimary = getVisibleParamFields(rfd4, { mode: 'binder' }, 'primary').map((f) => f.key);
    expect(binderPrimary).toContain('lockChain');
    expect(binderPrimary).toContain('hotspots');
    expect(binderPrimary).toContain('length');

    const lf = ADAPTER_REGISTRY['localfold-af2'];
    expect(getVisibleParamFields(lf, { msaMode: 'single' }, 'primary').map((f) => f.key)).not.toContain('msaA3mText');
    expect(getVisibleParamFields(lf, { msaMode: 'a3m' }, 'primary').map((f) => f.key)).toContain('msaA3mText');

    // web-protein-mpnn exposes all previously hidden advanced controls via paramSchema
    const mpnnAdvKeys = getVisibleParamFields(ADAPTER_REGISTRY['web-protein-mpnn'], {}, 'advanced').map(
      (f) => f.key,
    );
    expect(mpnnAdvKeys).toEqual(
      expect.arrayContaining([
        'modelName',
        'omitAminoAcids',
        'tiedPositions',
        'designChains',
        'backboneNoise',
        'unlockInherited',
        'seed',
      ]),
    );
    expect(MPNN_CHECKPOINTS_BY_FAMILY.proteinmpnn.map((o) => o.value)).toEqual([
      'v_48_020',
      'v_48_010',
      'v_48_002',
      'v_48_030',
    ]);
  });

  it('validates bounds, enums, modes, requiredInputs, aa-set, and residue-list via validateParamsWithSchema', () => {
    const rfd4 = ADAPTER_REGISTRY['rfd4-proteina'];
    // Valid defaults pass
    expect(validateParamsWithSchema(rfd4, getDefaultParamsFromSchema(rfd4), { connectedInputs: [] })).toEqual([]);
    // Mode requiring structure fails when connectedInputs omits 'structure'
    expect(
      validateParamsWithSchema(rfd4, { mode: 'binder', length: 70 }, { connectedInputs: [] }).join(' '),
    ).toMatch(/requires a connected "structure" input/);
    // Passes once 'structure' is connected
    expect(
      validateParamsWithSchema(rfd4, { mode: 'binder', length: 70 }, { connectedInputs: ['structure'] }),
    ).toEqual([]);
    // Mode-irrelevant field (partialFraction in unconditional mode) is skipped
    expect(validateParamsWithSchema(rfd4, { mode: 'unconditional', partialFraction: 99 })).toEqual([]);
    // Unknown mode reported cleanly without duplicate select error
    const badMode = validateParamsWithSchema(rfd4, { mode: 'teleport' });
    expect(badMode).toHaveLength(1);
    expect(badMode[0]).toMatch(/Unknown mode "teleport"/);

    // Plug-and-play future adapter (e.g. ThermoMPNN) works out of the box with zero hand-coded validation
    const thermoMpnn: AdapterManifest = {
      id: 'thermompnn',
      displayName: 'ThermoMPNN',
      shortName: 'ThermoMPNN',
      stage: 'score',
      runtime: 'ort-webgpu',
      estimatedVramMB: 180,
      approxWeightsMB: 20,
      licenseNote: 'MIT',
      description: 'Predict ΔΔG stability landscape.',
      inputs: { structure: { type: 'StructureArtifact' } },
      outputs: { mutations: { type: 'MutationMatrix' } },
      modeParamKey: 'scanMode',
      modes: [
        { id: 'ssm', label: 'Full Site-Saturation (L×20)', description: 'Score all point mutations.', requiredInputs: ['structure'] },
        { id: 'targeted', label: 'Targeted Mutations', description: 'Score explicit positions.', requiredInputs: ['structure'] },
      ],
      paramSchema: [
        {
          key: 'scanMode',
          label: 'Scan Mode',
          kind: 'select',
          tier: 'primary',
          defaultValue: 'ssm',
          options: [
            { value: 'ssm', label: 'Full SSM' },
            { value: 'targeted', label: 'Targeted' },
          ],
        },
        {
          key: 'positions',
          label: 'Target Positions',
          kind: 'residue-list',
          tier: 'primary',
          defaultValue: '',
          modes: ['targeted'],
        },
        {
          key: 'omitAAs',
          label: 'Omit AAs',
          kind: 'aa-set',
          tier: 'advanced',
          defaultValue: 'C',
        },
        {
          key: 'batchSize',
          label: 'Batch Size',
          kind: 'integer',
          tier: 'advanced',
          defaultValue: 32,
          min: 1,
          max: 256,
          step: 1,
        },
      ],
    };

    const defaults = getDefaultParamsFromSchema(thermoMpnn);
    expect(defaults).toEqual({ scanMode: 'ssm', positions: '', omitAAs: 'C', batchSize: 32 });
    expect(validateParamsWithSchema(thermoMpnn, defaults, { connectedInputs: ['structure'] })).toEqual([]);
    expect(
      validateParamsWithSchema(
        thermoMpnn,
        { scanMode: 'targeted', positions: 'A999-1', omitAAs: 'C5!', batchSize: 4.5 },
        { connectedInputs: [] },
      ),
    ).toHaveLength(4);

    // fieldToNumericSpec produces NumericInput-compatible spec
    const batchField = thermoMpnn.paramSchema!.find((f) => f.key === 'batchSize')!;
    expect(fieldToNumericSpec(batchField)).toEqual({
      min: 1,
      max: 256,
      step: 1,
      integer: true,
      defaultValue: 32,
    });
  });

  it('Worker-Param Coverage Guard: every p.<key> / params.<key> read by GPU workers is declared in ADAPTER_REGISTRY[id].paramSchema', () => {
    /**
     * Strictly minimal allowlist ONLY for programmatic/alias keys that are not standalone UI controls:
     * - `mpnn.worker.ts`: `ignoreInheritedLocks` is a programmatic alias of declared `unlockInherited`.
     * - `localfold.worker.ts`: `lockedChains` and `chainRoles` are structural lineage arrays/maps normally
     *   inherited from `SequenceSet` (kept as programmatic fallbacks only).
     */
    const PROGRAMMATIC_OR_ALIAS_KEYS: Record<string, ReadonlySet<string>> = {
      'web-protein-mpnn': new Set(['ignoreInheritedLocks']),
      'localfold-af2': new Set(['lockedChains', 'chainRoles']),
    };

    const WORKER_TARGETS: Array<{ adapterId: string; file: string; extraHelperKeys?: string[] }> = [
      { adapterId: 'rfd4-proteina', file: 'src/lib/plugins/rfd4-proteina/rfd4.worker.ts' },
      { adapterId: 'genie3', file: 'src/lib/plugins/genie3/genie3.worker.ts' },
      { adapterId: 'web-protein-mpnn', file: 'src/lib/plugins/web-protein-mpnn/mpnn.worker.ts' },
      {
        adapterId: 'localfold-af2',
        file: 'src/lib/plugins/localfold-af2/localfold.worker.ts',
        extraHelperKeys: ['maxSequences'], // read inside selectFoldWorkItems(seqSet, p)
      },
      { adapterId: 'thermompnn', file: 'src/lib/plugins/thermompnn/thermompnn.worker.ts' },
      { adapterId: 'ligandmpnn', file: 'src/lib/plugins/ligandmpnn/ligandmpnn.worker.ts' },
      { adapterId: 'lasermpnn', file: 'src/lib/plugins/lasermpnn/lasermpnn.worker.ts' },
      { adapterId: 'antifold-esmif1', file: 'src/lib/plugins/antifold-esmif1/antifold.worker.ts' },
      { adapterId: 'protpardelle', file: 'src/lib/plugins/protpardelle/protpardelle.worker.ts' },
      { adapterId: 'esmc', file: 'src/lib/plugins/esmc/esmc.worker.ts' },
      { adapterId: 'dyna1', file: 'src/lib/plugins/dyna1/dyna1.worker.ts' },
    ];

    for (const target of WORKER_TARGETS) {
      const manifest = ADAPTER_REGISTRY[target.adapterId];
      expect(manifest, `Missing manifest for ${target.adapterId}`).toBeDefined();
      const declared = new Set((manifest.paramSchema ?? []).map((f) => f.key));
      const allowedAliases = PROGRAMMATIC_OR_ALIAS_KEYS[target.adapterId] ?? new Set<string>();

      const workerKeys = new Set([
        ...extractWorkerRunParamKeys(target.file),
        ...(target.extraHelperKeys ?? []),
      ]);
      expect(workerKeys.size, `No param keys extracted from ${target.file}`).toBeGreaterThan(0);

      const missing = [...workerKeys].filter((k) => !declared.has(k) && !allowedAliases.has(k));
      expect(
        missing,
        `Hidden/unexposed worker parameter(s) in ${target.file} missing from ADAPTER_REGISTRY['${target.adapterId}'].paramSchema: ${missing.join(', ')}`,
      ).toEqual([]);

      // Reverse dead-param guard (#19): every declared paramSchema key must actually be read by the worker
      const DEAD_PARAM_ALLOWLIST: Record<string, ReadonlySet<string>> = {
        // `rfd4-proteina` reads `mode` / `hotspots` / `motifSpan` / etc. cleanly; only UI-only metadata keys allowed if any
      };
      const allowedUiOnly = DEAD_PARAM_ALLOWLIST[target.adapterId] ?? new Set<string>();
      const deadDeclared = [...declared].filter((k) => !workerKeys.has(k) && !allowedUiOnly.has(k));
      expect(
        deadDeclared,
        `Declared paramSchema key(s) in ADAPTER_REGISTRY['${target.adapterId}'] are never read in ${target.file}: ${deadDeclared.join(', ')}`,
      ).toEqual([]);
    }

    // Confirm unlockInherited (the canonical key for mpnn's ignoreInheritedLocks alias) is explicitly declared
    expect(
      ADAPTER_REGISTRY['web-protein-mpnn'].paramSchema!.some((f) => f.key === 'unlockInherited'),
    ).toBe(true);
  });

  it('CPU-Node Param Coverage Guard: migrated CPU plugin runners (pdb-input, sequence-input, top-k-filter, structure-compare, designability-scorer) params are declared in paramSchema', () => {
    const extractFileKeys = (relPath: string): string[] => {
      const src = readFileSync(resolve(process.cwd(), relPath), 'utf8');
      const keys = new Set<string>();
      for (const m of src.matchAll(/\bnode\.params\.([A-Za-z_][A-Za-z0-9_]*)\b/g)) {
        keys.add(m[1]);
      }
      return [...keys].sort();
    };

    const cpuChecks: Array<{
      adapterId: string;
      file: string;
      aliases?: ReadonlySet<string>;
    }> = [
      { adapterId: 'pdb-input', file: 'src/lib/plugins/pdb-input/cpu-runner.ts' },
      { adapterId: 'sequence-input', file: 'src/lib/plugins/sequence-input/cpu-runner.ts' },
      { adapterId: 'top-k-filter', file: 'src/lib/plugins/top-k-filter/cpu-runner.ts' },
      { adapterId: 'structure-compare', file: 'src/lib/plugins/structure-compare/cpu-runner.ts' },
      { adapterId: 'designability-scorer', file: 'src/lib/plugins/designability-scorer/cpu-runner.ts' },
    ];

    for (const check of cpuChecks) {
      const declared = new Set((ADAPTER_REGISTRY[check.adapterId].paramSchema ?? []).map((f) => f.key));
      const readKeys = extractFileKeys(check.file);
      expect(readKeys.length).toBeGreaterThan(0);
      const missing = readKeys.filter((k) => !declared.has(k) && !check.aliases?.has(k));
      expect(
        missing,
        `CPU plugin "${check.adapterId}" (${check.file}) reads params missing from paramSchema: ${missing.join(', ')}`,
      ).toEqual([]);
    }
  });

  it('Wave 1B Primitive 1: unified residue tokens (UX-04), mode-scoped warnBelow (#78), sanitizeParamsOnModeSwitch (WF-07/WF-09), preset fix (CT-07) & Inspector parity (CT-02/CT-04)', () => {
    // 1. Unified residue-selector token predicate (UX-04)
    expect(isValidResidueSelectorToken('A', { allowBareChain: true })).toBe(true);
    expect(isValidResidueSelectorToken('AA', { allowBareChain: true })).toBe(true);
    expect(isValidResidueSelectorToken('HL', { allowBareChain: true })).toBe(true);
    expect(isValidResidueSelectorToken('A', { allowBareChain: false })).toBe(false);
    expect(isValidResidueSelectorToken('AA', { allowBareChain: false })).toBe(false);
    expect(isValidResidueSelectorToken('123', { allowBareChain: true })).toBe(true); // numeric residue index
    expect(isValidResidueSelectorToken('A:SAM', { allowBareChain: true, allowLigandRef: true })).toBe(true);
    expect(isValidResidueSelectorToken('A:1PE', { allowBareChain: true, allowLigandRef: true })).toBe(true);
    expect(isValidResidueSelectorToken('A:SAM', { allowBareChain: true, allowLigandRef: false })).toBe(false);
    expect(
      validateParamsWithSchema(ADAPTER_REGISTRY['pdb-input'], {
        mode: 'pdb-id',
        pdbId: '1CTT',
        cropSegments: 'A, A:1PE',
      }),
    ).toEqual([]);
    expect(isValidResidueSelectorToken('AA:1-5', { allowBareChain: false })).toBe(true);
    expect(isValidResidueSelectorToken('A100A-100C', { allowBareChain: false })).toBe(true);
    expect(isValidResidueSelectorToken('A-5--2', { allowBareChain: false })).toBe(true);
    expect(isValidResidueSelectorToken('A999-1', { allowBareChain: true })).toBe(false);
    expect(isValidResidueSelectorToken('A1-B5', { allowBareChain: true })).toBe(false);

    // Schema allowBareChain flags match design intent
    expect(ADAPTER_REGISTRY['web-protein-mpnn'].paramSchema!.find((f) => f.key === 'fixedPositions')?.allowBareChain).toBe(true);
    expect(ADAPTER_REGISTRY.ligandmpnn.paramSchema!.find((f) => f.key === 'fixedPositions')?.allowBareChain).toBe(true);
    expect(ADAPTER_REGISTRY.lasermpnn.paramSchema!.find((f) => f.key === 'fixedPositions')?.allowBareChain).toBe(true);
    expect(ADAPTER_REGISTRY['antifold-esmif1'].paramSchema!.find((f) => f.key === 'fixedPositions')?.allowBareChain).toBe(true);
    expect(ADAPTER_REGISTRY['pdb-input'].paramSchema!.find((f) => f.key === 'cropSegments')?.allowBareChain).toBe(true);
    expect(ADAPTER_REGISTRY['rfd4-proteina'].paramSchema!.find((f) => f.key === 'hotspots')?.allowBareChain).toBe(false);
    expect(ADAPTER_REGISTRY['rfd4-proteina'].paramSchema!.find((f) => f.key === 'motifSpan')?.allowBareChain).toBe(false);
    expect(ADAPTER_REGISTRY.genie3.paramSchema!.find((f) => f.key === 'hotspots')?.allowBareChain).toBe(false);
    expect(ADAPTER_REGISTRY.genie3.paramSchema!.find((f) => f.key === 'motifSpan')?.allowBareChain).toBe(false);
    expect(ADAPTER_REGISTRY.boltzgen.paramSchema!.find((f) => f.key === 'hotspots')?.allowBareChain).toBe(false);

    // 2. Schema-driven #78 step guardrails & mode-scoped warnBelow
    const genie2Steps = ADAPTER_REGISTRY.genie2.paramSchema!.find((f) => f.key === 'steps')!;
    expect(genie2Steps.min).toBe(1000);
    expect(genie2Steps.max).toBe(1000);
    expect(genie2Steps.defaultValue).toBe(1000);
    expect(fieldToNumericSpec(genie2Steps).warnBelow).toMatchObject({ threshold: 1000, fixValue: 1000 });

    const laSteps = ADAPTER_REGISTRY['la-proteina'].paramSchema!.find((f) => f.key === 'steps')!;
    expect(laSteps.min).toBe(50);
    expect(laSteps.max).toBe(400);
    expect(fieldToNumericSpec(laSteps, 'sde').warnBelow).toMatchObject({ threshold: 350, fixValue: 400 });
    expect(fieldToNumericSpec(laSteps, 'ode').warnBelow).toBeUndefined();

    const proSteps = ADAPTER_REGISTRY.proteina.paramSchema!.find((f) => f.key === 'steps')!;
    expect(proSteps.min).toBe(50);
    expect(proSteps.max).toBe(400);
    expect(fieldToNumericSpec(proSteps, 'sc').warnBelow).toMatchObject({ threshold: 350, fixValue: 400 });
    expect(fieldToNumericSpec(proSteps, 'vf').warnBelow).toBeUndefined();

    const dynaModeField = ADAPTER_REGISTRY.dyna1.paramSchema!.find((f) => f.key === 'inputMode')!;
    expect(dynaModeField.description).toMatch(/sequence-only/i);
    expect(dynaModeField.options?.find((o) => o.value === 'seq-only')?.label).toMatch(/3D/i);

    // 3. CT-07: rfd4-conditioned-cath preset uses mode: 'conditioned'
    const cathPreset = WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-conditioned-cath')!;
    expect(cathPreset.graph.nodes[0].params.mode).toBe('conditioned');

    // 4. WF-10 / WF-13: defaultParamsForAdapter delegates to schema & removes dead rfd4-conditioner
    expect(defaultParamsForAdapter('rfd4-conditioner')).toEqual({});
    expect(defaultParamsForAdapter('rfd4-proteina').numDesigns).toBe(1);
    expect(defaultParamsForAdapter('designability-scorer').targetAlignedBinderRmsdThreshold).toBe(4.0);
    expect(defaultParamsForAdapter('web-protein-mpnn')).toMatchObject({
      omitAminoAcids: '',
      tiedPositions: '',
      designChains: '',
      backboneNoise: 0,
      unlockInherited: false,
    });

    // 5. WF-07 / WF-09: sanitizeParamsOnModeSwitch clears incompatible mode constraints
    const rfdBinderToMotif = sanitizeParamsOnModeSwitch(
      'rfd4-proteina',
      {
        mode: 'binder',
        lockChain: 'A',
        hotspots: 'A54, A56',
        contig: 'A17-132,/0,70-70',
        cathCode: '3.40.50',
        length: 70,
      },
      'motif',
    );
    expect(rfdBinderToMotif.mode).toBe('motif');
    expect(rfdBinderToMotif.hotspots).toBe('');
    expect(rfdBinderToMotif.lockChain).toBe('');
    expect(rfdBinderToMotif.contig).toBe('');
    expect(rfdBinderToMotif.cathCode).toBe('');

    const rfdBinderToUncond = sanitizeParamsOnModeSwitch(
      'rfd4-proteina',
      {
        mode: 'binder',
        lockChain: 'A',
        hotspots: 'A54',
        hbondMode: 'ACCEPTOR',
        hbondSelection: 'B/SAM/*/*',
        contig: 'A17-132,/0,70-70',
      },
      'unconditional',
    );
    expect(rfdBinderToUncond.hotspots).toBe('');
    expect(rfdBinderToUncond.lockChain).toBe('');
    expect(rfdBinderToUncond.contig).toBe('');
    expect(rfdBinderToUncond.hbondMode).toBe('none');
    expect(rfdBinderToUncond.hbondSelection).toBe('');

    const lfToEsmfold2 = sanitizeParamsOnModeSwitch(
      'localfold-af2',
      { model: 'boltz2', msaMode: 'mmseqs2', msaA3mText: '>q\nACDEF', diffusionSteps: 25, recycles: 4 },
      'ef2-fast-300m',
    );
    expect(lfToEsmfold2.model).toBe('ef2-fast-300m');
    expect(lfToEsmfold2.msaMode).toBe('single');
    expect(lfToEsmfold2.msaA3mText).toBeUndefined();
    expect(lfToEsmfold2.diffusionSteps).toBeUndefined();

    const lfEsmfoldToMono = sanitizeParamsOnModeSwitch(
      'localfold-af2',
      { model: 'ef2-fast-300m', esmfold2Steps: 32, ligandCcdCodes: 'SAM', dnaSequences: 'ACGT', recycles: 15 },
      'monomer',
    );
    expect(lfEsmfoldToMono.esmfold2Steps).toBeUndefined();
    expect(lfEsmfoldToMono.ligandCcdCodes).toBeUndefined();
    expect(lfEsmfoldToMono.dnaSequences).toBeUndefined();
    expect(lfEsmfoldToMono.recycles).toBe(10);

    // 6. CT-02 / CT-04 / UX-04 Inspector & helper parity
    expect(validateResidueMappingSyntax('0:0, 1:1, 2:2')).toBeNull();
    expect(validateResidueMappingSyntax('bad_token')).not.toBeNull();
    expect(
      formatNodeParamSummary('rfd4-proteina', { mode: 'preset', presetId: 'yaml_5o45_binder', nsteps: 60 }),
    ).toContain('5O45 PD-L1 Binder');

    const inspectorSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/NodeInspector.svelte'), 'utf8');
    expect(inspectorSrc).toContain('value="conditioned"');
    expect(inspectorSrc).toContain('rfd4PresetOptions');
    expect(inspectorSrc).toContain('sanitizeParamsOnModeSwitch');
    expect(inspectorSrc).toContain('isValidResidueSelectorToken');
    expect(inspectorSrc).toContain('Auto (${inferredTemplateChain})');
    expect(inspectorSrc).not.toContain("node.params.lockChain ?? 'A'");
    expect(inspectorSrc).toContain('scorerAdvFields');
    expect(inspectorSrc).not.toContain('updateBinderLengthInContig(prevContig, v, false)');

    const schemaFieldsSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/SchemaParamFields.svelte'), 'utf8');
    expect(schemaFieldsSrc).toContain('isValidResidueSelectorToken');
    expect(schemaFieldsSrc).toContain('validateResidueMappingSyntax');
    expect(schemaFieldsSrc).toContain('fieldToNumericSpec(field, resolveCurrentMode())');
    expect(schemaFieldsSrc).toContain('warn-fix-btn');

    // 7. WF-08: insertBlockSmart places upstream sources -280px to the left and wires parallel sibling generators
    const rfdOnlyGraph = {
      id: 'wf08-upstream',
      name: 'WF-08 Upstream',
      nodes: [
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          label: 'RFD4-Proteina',
          params: defaultParamsForAdapter('rfd4-proteina'),
          position: { x: 340, y: 60 },
        },
      ],
      edges: [],
    };
    const upRes = insertBlockSmart(rfdOnlyGraph, 'pdb-input', 'rfd', undefined, { autoCommit: true });
    const pdbNode = upRes.graph.nodes.find((n) => n.id === upRes.nodeId)!;
    expect(pdbNode.position!.x).toBe(60); // 340 - 280 (to the left of RFD4)
    expect(upRes.proposedEdges).toEqual([
      { fromNode: upRes.nodeId, fromPort: 'structure', toNode: 'rfd', toPort: 'structure' },
    ]);

    const sibRes = insertBlockSmart(upRes.graph, 'genie3', 'rfd', undefined, { autoCommit: false });
    const genieNode = sibRes.graph.nodes.find((n) => n.id === sibRes.nodeId)!;
    expect(genieNode.position!.x).toBe(340); // parallel sibling in same column below RFD4
    expect(genieNode.position!.y).toBeGreaterThan(60);
    expect(sibRes.proposedEdges).toEqual([
      { fromNode: upRes.nodeId, fromPort: 'structure', toNode: sibRes.nodeId, toPort: 'structure' },
    ]);
    expect(sibRes.graph.edges).toEqual(upRes.graph.edges); // autoCommit: false respected
  });
});
