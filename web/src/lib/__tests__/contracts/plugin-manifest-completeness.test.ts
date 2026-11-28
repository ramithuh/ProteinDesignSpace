import { describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from '$lib/engine/registry';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { getModelCardForAdapter } from '$lib/engine/model-cards';

const EXPECTED_ADAPTER_IDS = [
  'pdb-input',
  'sequence-input',
  'rfd4-proteina',
  'genie3',
  'protpardelle',
  'rfdiffusion',
  'rfd3',
  'boltzgen',
  'evodiff',
  'la-proteina',
  'proteina',
  'sid-protein',
  'genie2',
  'frameflow',
  'foldflow2',
  'artios-backbone',
  'web-protein-mpnn',
  'ligandmpnn',
  'lasermpnn',
  'artios-potts',
  'antifold-esmif1',
  'localfold-af2',
  'thermompnn',
  'esmc',
  'dyna1',
  'gnina',
  'top-k-filter',
  'designability-scorer',
  'structure-compare',
  'pulchra',
  'cg2all',
] as const;

describe('Phase 0 contract: plugin manifest & model-card completeness (31 adapters)', () => {
  it('registers all 31 baseline adapters with valid manifests and default params', () => {
    const keys = Object.keys(ADAPTER_REGISTRY).sort();
    expect(keys).toEqual([...EXPECTED_ADAPTER_IDS].sort());
    expect(keys).toHaveLength(31);

    let cpuCount = 0;
    let ortCount = 0;
    let wgslCount = 0;

    for (const id of EXPECTED_ADAPTER_IDS) {
      const m = ADAPTER_REGISTRY[id];
      expect(m, `Missing manifest for ${id}`).toBeDefined();
      expect(m.id).toBe(id);
      expect(m.displayName.trim().length).toBeGreaterThan(0);
      expect(m.shortName.trim().length).toBeGreaterThan(0);
      expect(m.description.trim().length).toBeGreaterThan(0);
      expect(m.licenseNote.trim().length).toBeGreaterThan(0);
      expect(m.inputs).toBeTypeOf('object');
      expect(m.outputs).toBeTypeOf('object');
      expect(Number.isFinite(m.estimatedVramMB) && m.estimatedVramMB >= 0).toBe(true);
      expect(Number.isFinite(m.approxWeightsMB) && m.approxWeightsMB >= 0).toBe(true);

      const defaults = defaultParamsForAdapter(id);
      expect(defaults).toBeTypeOf('object');

      if (m.runtime === 'cpu-ts') cpuCount++;
      else if (m.runtime === 'ort-webgpu') ortCount++;
      else if (m.runtime === 'wgsl-custom') wgslCount++;
    }

    expect(cpuCount).toBe(6);
    expect(ortCount).toBe(1);
    expect(wgslCount).toBe(24);
  });

  it('provides model cards for all 25 ML adapters and none for the 6 pure CPU utilities', () => {
    const mlAdapters = EXPECTED_ADAPTER_IDS.filter((id) => ADAPTER_REGISTRY[id].runtime !== 'cpu-ts');
    const cpuAdapters = EXPECTED_ADAPTER_IDS.filter((id) => ADAPTER_REGISTRY[id].runtime === 'cpu-ts');

    expect(mlAdapters).toHaveLength(25);
    expect(cpuAdapters).toHaveLength(6);

    for (const id of mlAdapters) {
      const card = getModelCardForAdapter(id, defaultParamsForAdapter(id));
      expect(card, `Expected model card for ML adapter "${id}"`).not.toBeNull();
      expect(card!.precisions.length).toBeGreaterThan(0);
    }

    for (const id of cpuAdapters) {
      expect(getModelCardForAdapter(id)).toBeNull();
    }
  });

  it('keeps all preset node adapterIds strictly inside the registered adapter set', () => {
    expect(WORKFLOW_PRESETS.length).toBeGreaterThan(0);
    for (const preset of WORKFLOW_PRESETS) {
      for (const node of preset.graph.nodes) {
        expect(ADAPTER_REGISTRY[node.adapterId], `Unknown adapter "${node.adapterId}" in preset "${preset.id}"`).toBeDefined();
      }
    }
  });
});
