import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  ADAPTER_REGISTRY,
  MPNN_CHECKPOINTS_BY_FAMILY,
} from '../engine/registry';
import {
  fieldToNumericSpec,
  getActiveModeSpec,
  getDefaultParamsFromSchema,
  getVisibleParamFields,
  validateParamsWithSchema,
} from '../engine/param-schema';
import { NUMERIC_SPECS } from '../engine/param-validation';
import type { AdapterManifest } from '../engine/types';

/**
 * Extract every `p.<key>`, `params.<key>`, or `msg.params.<key>` property read
 * inside a worker's `msg.type === 'run'` handler.
 */
function extractWorkerRunParamKeys(workerRelPath: string): string[] {
  const src = readFileSync(resolve(process.cwd(), workerRelPath), 'utf8');
  const runIdx = src.indexOf("msg.type === 'run'");
  expect(runIdx, `Expected msg.type === 'run' handler in ${workerRelPath}`).toBeGreaterThan(-1);
  const runSlice = src.slice(runIdx);
  const keys = new Set<string>();
  for (const m of runSlice.matchAll(/\b(?:p|params|msg\.params)\??\.([A-Za-z_][A-Za-z0-9_]*)\b/g)) {
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
      { adapterId: 'rfd4-proteina', file: 'src/lib/workers/rfd4.worker.ts' },
      { adapterId: 'genie3', file: 'src/lib/workers/genie3.worker.ts' },
      { adapterId: 'web-protein-mpnn', file: 'src/lib/workers/mpnn.worker.ts' },
      {
        adapterId: 'localfold-af2',
        file: 'src/lib/workers/localfold.worker.ts',
        extraHelperKeys: ['maxSequences'], // read inside selectFoldWorkItems(seqSet, p)
      },
      { adapterId: 'thermompnn', file: 'src/lib/workers/thermompnn.worker.ts' },
      { adapterId: 'ligandmpnn', file: 'src/lib/workers/ligandmpnn.worker.ts' },
      { adapterId: 'lasermpnn', file: 'src/lib/workers/lasermpnn.worker.ts' },
      { adapterId: 'antifold-esmif1', file: 'src/lib/workers/antifold.worker.ts' },
      { adapterId: 'protpardelle', file: 'src/lib/workers/protpardelle.worker.ts' },
      { adapterId: 'esmc', file: 'src/lib/workers/esmc.worker.ts' },
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

  it('CPU-Node Param Coverage Guard: scheduler.ts CPU nodes (pdb-input, sequence-input, designability-scorer) params are declared in paramSchema', () => {
    const schedSrc = readFileSync(resolve(process.cwd(), 'src/lib/engine/scheduler.ts'), 'utf8');

    const extractSectionKeys = (startMarker: string, endMarker: string): string[] => {
      const s = schedSrc.indexOf(startMarker);
      const e = schedSrc.indexOf(endMarker, s + startMarker.length);
      expect(s).toBeGreaterThan(-1);
      expect(e).toBeGreaterThan(s);
      const slice = schedSrc.slice(s, e);
      const keys = new Set<string>();
      for (const m of slice.matchAll(/\bnode\.params\.([A-Za-z_][A-Za-z0-9_]*)\b/g)) {
        keys.add(m[1]);
      }
      return [...keys].sort();
    };

    const cpuChecks: Array<{
      adapterId: string;
      start: string;
      end: string;
      aliases?: ReadonlySet<string>;
    }> = [
      {
        adapterId: 'pdb-input',
        start: 'export async function materializePdbInputNode(',
        end: 'export function materializeSequenceInputNode(',
      },
      {
        adapterId: 'sequence-input',
        start: 'export function materializeSequenceInputNode(',
        end: 'function topoSort(',
      },
      {
        adapterId: 'top-k-filter',
        start: "if (node.adapterId === 'top-k-filter') {",
        end: "if (node.adapterId === 'designability-scorer') {",
      },
      {
        adapterId: 'designability-scorer',
        start: "if (node.adapterId === 'designability-scorer') {",
        end: 'throw new Error(`Unknown CPU adapter',
        aliases: new Set(['referenceMapping']), // alias of declared `residueMapping`
      },
    ];

    for (const check of cpuChecks) {
      const declared = new Set((ADAPTER_REGISTRY[check.adapterId].paramSchema ?? []).map((f) => f.key));
      const readKeys = extractSectionKeys(check.start, check.end);
      expect(readKeys.length).toBeGreaterThan(0);
      const missing = readKeys.filter((k) => !declared.has(k) && !check.aliases?.has(k));
      expect(
        missing,
        `CPU node "${check.adapterId}" reads params missing from paramSchema: ${missing.join(', ')}`,
      ).toEqual([]);
    }
  });
});
