// Layer 3 Model Plugins Barrel — Single Registration Entrypoint (ALL 32 ADAPTERS NATIVE!).
// Adding model #33 = 1 cohesive folder under `src/lib/plugins/<id>/` + 1 entry in `NATIVE_PLUGINS`.

import { PluginRegistry, setActivePluginRegistry, type DomainPack, type ModelPlugin } from '$lib/kernel';
import { proteinBiopolymerDomainPack } from '$lib/domains/protein-biopolymer';

// Batch 3A: cpu-ts utilities (6)
import { pdbInputPlugin } from './pdb-input';
import { sequenceInputPlugin } from './sequence-input';
import { topKFilterPlugin } from './top-k-filter';
import { structureComparePlugin } from './structure-compare';
import { designabilityScorerPlugin } from './designability-scorer';
import { pulchraPlugin } from './pulchra';

// Batch 3B: ort-webgpu (1)
import { webProteinMpnnPlugin } from './web-protein-mpnn';

// Batch 3C: standard wgsl-custom generators & inverse-folders (16)
import { evodiffPlugin } from './evodiff';
import { laProteinaPlugin } from './la-proteina';
import { proteinaPlugin } from './proteina';
import { sidProteinPlugin } from './sid-protein';
import { genie2Plugin } from './genie2';
import { frameflowPlugin } from './frameflow';
import { foldflow2Plugin } from './foldflow2';
import { protpardellePlugin } from './protpardelle';
import { rfdiffusionPlugin } from './rfdiffusion';
import { rfd3Plugin } from './rfd3';
import { boltzgenPlugin } from './boltzgen';
import { artiosBackbonePlugin } from './artios-backbone';
import { ligandmpnnPlugin } from './ligandmpnn';
import { lasermpnnPlugin } from './lasermpnn';
import { artiosPottsPlugin } from './artios-potts';
import { protonPottsMpnnPlugin } from './proton-pottsmpnn';
import { antifoldEsmif1Plugin } from './antifold-esmif1';

// Batch 3D: scorers & dynamics (3)
import { thermompnnPlugin } from './thermompnn';
import { esmcPlugin } from './esmc';
import { dyna1Plugin } from './dyna1';

// Batch 3E: docking & scoring (1)
import { gninaPlugin } from './gnina';

// Batch 3G: structure reconstruction (1)
import { cg2allPlugin } from './cg2all';

// Batch 3F: conditioned generators & multi-checkpoint folder (3)
import { rfd4ProteinaPlugin } from './rfd4-proteina';
import { genie3Plugin } from './genie3';
import { localfoldAf2Plugin } from './localfold-af2';

/**
 * All 32 native Layer 3 ModelPlugins (zero legacy-bridged adapters remain!).
 */
export const NATIVE_PLUGINS: readonly ModelPlugin[] = [
  // 6 cpu-ts utilities
  pdbInputPlugin,
  sequenceInputPlugin,
  topKFilterPlugin,
  structureComparePlugin,
  designabilityScorerPlugin,
  pulchraPlugin,
  // 1 ort-webgpu
  webProteinMpnnPlugin,
  // 17 wgsl-custom generators & inverse-folders
  protpardellePlugin,
  rfdiffusionPlugin,
  rfd3Plugin,
  boltzgenPlugin,
  artiosBackbonePlugin,
  laProteinaPlugin,
  proteinaPlugin,
  sidProteinPlugin,
  genie2Plugin,
  frameflowPlugin,
  foldflow2Plugin,
  evodiffPlugin,
  ligandmpnnPlugin,
  lasermpnnPlugin,
  artiosPottsPlugin,
  protonPottsMpnnPlugin,
  antifoldEsmif1Plugin,
  // 3 scorers & dynamics
  thermompnnPlugin,
  esmcPlugin,
  dyna1Plugin,
  // 1 docking
  gninaPlugin,
  // 1 Cα → all-atom backmapping
  cg2allPlugin,
  // 3 complex conditioned generators & multi-checkpoint folder
  rfd4ProteinaPlugin,
  genie3Plugin,
  localfoldAf2Plugin,
];

/**
 * Build a fresh PluginRegistry booted with `@domain/protein-biopolymer`, all native plugins,
 * and any optional extra DomainPacks / ModelPlugins (e.g., future small-molecule or
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

  for (const plugin of NATIVE_PLUGINS) {
    registry.registerPlugin(plugin);
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
