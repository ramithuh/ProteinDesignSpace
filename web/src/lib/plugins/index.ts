// Layer 3 Model Plugins Barrel — Single Registration Entrypoint.
// Adding model #27 = 1 cohesive folder under `src/lib/plugins/<id>/` + 1 entry in `NATIVE_PLUGINS`.

import { PluginRegistry, setActivePluginRegistry, type DomainPack, type ModelPlugin } from '$lib/kernel';
import { proteinBiopolymerDomainPack } from '$lib/domains/protein-biopolymer';
import { buildLegacyBridgedPlugins } from './legacy-bridge';
import { pdbInputPlugin } from './pdb-input';
import { sequenceInputPlugin } from './sequence-input';
import { topKFilterPlugin } from './top-k-filter';
import { structureComparePlugin } from './structure-compare';
import { designabilityScorerPlugin } from './designability-scorer';

/**
 * Native Layer 3 ModelPlugins extracted out of legacy switches (populated across Phase 3 Batches 3A-3F).
 */
export const NATIVE_PLUGINS: readonly ModelPlugin[] = [
  // Batch 3A: cpu-ts utilities (5/5)
  pdbInputPlugin,
  sequenceInputPlugin,
  topKFilterPlugin,
  structureComparePlugin,
  designabilityScorerPlugin,
];

/**
 * Build a fresh PluginRegistry booted with `@domain/protein-biopolymer`, all native + bridged
 * plugins, and any optional extra DomainPacks / ModelPlugins (e.g., future small-molecule or
 * property-prediction packs, or test fixtures).
 */
export function createDefaultPluginRegistry(opts?: {
  extraDomainPacks?: readonly DomainPack[];
  extraPlugins?: readonly ModelPlugin[];
  freeze?: boolean;
}): PluginRegistry {
  const registry = new PluginRegistry();
  registry.registerDomainPack(proteinBiopolymerDomainPack);
  for (const pack of opts?.extraDomainPacks ?? []) {
    registry.registerDomainPack(pack);
  }

  const nativeIds = new Set<string>();
  for (const plugin of NATIVE_PLUGINS) {
    nativeIds.add(plugin.id);
    registry.registerPlugin(plugin);
  }

  for (const bridged of buildLegacyBridgedPlugins(nativeIds)) {
    registry.registerPlugin(bridged);
  }

  for (const extra of opts?.extraPlugins ?? []) {
    registry.registerPlugin(extra);
  }

  if (opts?.freeze !== false) {
    registry.freeze();
  }
  return registry;
}

/** Frozen singleton PluginRegistry for production runtime boot. */
export const defaultPluginRegistry: PluginRegistry = createDefaultPluginRegistry({ freeze: true });
setActivePluginRegistry(defaultPluginRegistry);
