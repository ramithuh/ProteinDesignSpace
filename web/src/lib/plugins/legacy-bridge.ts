// Phase 2-3 Transitional Bridge — Wraps not-yet-migrated baseline adapters into strict Layer 3 ModelPlugins.
// As Phase 3 batches (3A-3F) migrate each adapter to `src/lib/plugins/<adapter-id>/`,
// its entry is removed from this bridge and registered natively in `src/lib/plugins/index.ts`.

import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from '$lib/engine/registry';
import { CONCISE_BLOCK_ROLES } from '$lib/engine/catalog';
import { getModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type {
  FallbackPolicy,
  KernelAdapterManifest,
  KernelModelCardSpec,
  ModelPlugin,
  PluginCachePolicy,
  PluginShareSpec,
} from '$lib/kernel';

/** Immutable #g= URL short codes locked by Phase 0 golden tests. */
export const LEGACY_SHORT_CODES: Readonly<Record<string, string>> = {
  'pdb-input': 'p',
  'sequence-input': 'q',
  'rfd4-proteina': 'r',
  genie3: 'g',
  protpardelle: 'pp',
  rfdiffusion: 'd',
  boltzgen: 'b',
  evodiff: 'e',
  'la-proteina': 'lp',
  proteina: 'pr',
  'sid-protein': 'sp',
  genie2: 'g2',
  frameflow: 'ff',
  foldflow2: 'f2',
  'web-protein-mpnn': 'm',
  ligandmpnn: 'lm',
  lasermpnn: 'la',
  'antifold-esmif1': 'af',
  'localfold-af2': 'f',
  thermompnn: 'tm',
  esmc: 'ec',
  dyna1: 'dy',
  gnina: 'gn',
  'top-k-filter': 'tk',
  'designability-scorer': 's',
  'structure-compare': 'c',
};

const COOPERATIVE_ABORT_ADAPTERS = new Set([
  'evodiff',
  'foldflow2',
  'frameflow',
  'genie2',
  'la-proteina',
  'proteina',
  'sid-protein',
]);

const LEGACY_CACHE_POLICIES: Readonly<Record<string, PluginCachePolicy>> = {
  'web-protein-mpnn': { usesSharedOpfsStore: true, opfsDirs: ['denovo-weights'], cacheStoragePrefixes: ['denovo-run'] },
  'localfold-af2': { cacheStoragePrefixes: ['localfold-'] },
  'rfd4-proteina': { cacheStoragePrefixes: ['rfd4p-'] },
  genie3: { cacheStoragePrefixes: ['genie3-weights-'] },
  ligandmpnn: { cacheStorageNames: ['ligandmpnn-v1'], usesSharedOpfsStore: true },
  esmc: { cacheStorageNames: ['esmc-v1'], opfsDirs: ['esmc-v1'] },
  lasermpnn: { cacheStorageNames: ['lasermpnn-v1'], usesSharedOpfsStore: true },
  thermompnn: { cacheStorageNames: ['thermompnn-v1'], opfsDirs: ['thermompnn-v1'] },
  protpardelle: { cacheStorageNames: ['protpardelle-v1'], opfsDirs: ['protpardelle-v1'] },
  rfdiffusion: { cacheStorageNames: ['rfdiffusion-v1'], opfsDirs: ['rfdiffusion-v1'] },
  boltzgen: { cacheStorageNames: ['boltzgen-v1'], opfsDirs: ['boltzgen-v1'] },
  evodiff: { cacheStorageNames: ['evodiff-v1'], opfsDirs: ['evodiff-v1'] },
  'la-proteina': { cacheStorageNames: ['la-proteina-v1'], opfsDirs: ['la-proteina-v1'] },
  proteina: { cacheStorageNames: ['proteina-v1'], opfsDirs: ['proteina-v1'] },
  genie2: { cacheStorageNames: ['genie2-v1'], opfsDirs: ['genie2-v1'] },
  frameflow: { cacheStorageNames: ['frameflow-v1'], opfsDirs: ['frameflow-v1'] },
  foldflow2: { cacheStorageNames: ['foldflow2-v1'], opfsDirs: ['foldflow2-v1'] },
  'sid-protein': { cacheStorageNames: ['sid-protein-v1'], opfsDirs: ['sid-protein-v1'] },
  dyna1: { cacheStorageNames: ['dyna1-v1'], opfsDirs: ['dyna1-v1'] },
  'antifold-esmif1': { cacheStorageNames: ['gvp-transformer-v1'], opfsDirs: ['gvp-transformer-v1'] },
  gnina: { cacheStorageNames: ['gnina-v1'], opfsDirs: ['gnina-v1'] },
};

const LAZY_WORKER_LOADERS: Readonly<Record<string, () => Promise<Worker>>> = {
  'web-protein-mpnn': async () => new (await import('$lib/workers/mpnn.worker?worker')).default(),
  'localfold-af2': async () => new (await import('$lib/workers/localfold.worker?worker')).default(),
  'rfd4-proteina': async () => new (await import('$lib/workers/rfd4.worker?worker')).default(),
  genie3: async () => new (await import('$lib/workers/genie3.worker?worker')).default(),
  thermompnn: async () => new (await import('$lib/workers/thermompnn.worker?worker')).default(),
  ligandmpnn: async () => new (await import('$lib/workers/ligandmpnn.worker?worker')).default(),
  lasermpnn: async () => new (await import('$lib/workers/lasermpnn.worker?worker')).default(),
  'antifold-esmif1': async () => new (await import('$lib/workers/antifold.worker?worker')).default(),
  protpardelle: async () => new (await import('$lib/workers/protpardelle.worker?worker')).default(),
  rfdiffusion: async () => new (await import('$lib/workers/rfdiffusion.worker?worker')).default(),
  boltzgen: async () => new (await import('$lib/workers/boltzgen.worker?worker')).default(),
  evodiff: async () => new (await import('$lib/workers/evodiff.worker?worker')).default(),
  'la-proteina': async () => new (await import('$lib/workers/laproteina.worker?worker')).default(),
  proteina: async () => new (await import('$lib/workers/proteina.worker?worker')).default(),
  genie2: async () => new (await import('$lib/workers/genie2.worker?worker')).default(),
  frameflow: async () => new (await import('$lib/workers/frameflow.worker?worker')).default(),
  foldflow2: async () => new (await import('$lib/workers/foldflow2.worker?worker')).default(),
  'sid-protein': async () => new (await import('$lib/workers/sidprotein.worker?worker')).default(),
  esmc: async () => new (await import('$lib/workers/esmc.worker?worker')).default(),
  dyna1: async () => new (await import('$lib/workers/dyna1.worker?worker')).default(),
  gnina: async () => new (await import('$lib/workers/gnina.worker?worker')).default(),
};

function resolveLegacyFallback(adapterId: string, runtime: string): FallbackPolicy {
  if (runtime === 'cpu-ts') {
    return { kind: 'none', reason: 'Pure TypeScript CPU adapter' };
  }
  if (adapterId === 'web-protein-mpnn') {
    return { kind: 'ort-wasm', wasmAssetDir: '/vendor/ort/' };
  }
  if (adapterId === 'thermompnn') {
    return {
      kind: 'cpu-js',
      moduleUrl: '/vendor/thermompnn/cpu/index.js',
      degradedWarning: 'Running ThermoMPNN on CPU JS fallback (slower than WebGPU).',
    };
  }
  return { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' };
}

function buildShareSpec(adapterId: string): PluginShareSpec {
  const shortCode = LEGACY_SHORT_CODES[adapterId];
  if (!shortCode) {
    throw new Error(`Missing immutable shortCode for legacy adapter "${adapterId}".`);
  }
  if (adapterId === 'localfold-af2') {
    return { shortCode, yamlKeyAliases: { model: 'foldModel' } };
  }
  if (adapterId === 'sequence-input') {
    return { shortCode, yamlKeyAliases: { label: 'seqLabel' } };
  }
  return { shortCode };
}

/** Build strict Layer 3 `ModelPlugin` wrappers for all baseline adapters not yet natively migrated. */
export function buildLegacyBridgedPlugins(excludeIds: ReadonlySet<string> = new Set()): ModelPlugin[] {
  const plugins: ModelPlugin[] = [];

  for (const [id, legacyManifest] of Object.entries(ADAPTER_REGISTRY)) {
    if (excludeIds.has(id)) continue;

    const manifest = legacyManifest as unknown as KernelAdapterManifest;
    const isGpu = manifest.runtime !== 'cpu-ts';
    const workerLoader = LAZY_WORKER_LOADERS[id];

    const plugin: ModelPlugin = {
      id,
      domainId: PROTEIN_DOMAIN_ID,
      manifest,
      execution: {
        runtime: manifest.runtime,
        fallback: resolveLegacyFallback(id, manifest.runtime),
        ...(isGpu && workerLoader ? { createWorker: workerLoader } : {}),
        ...(COOPERATIVE_ABORT_ADAPTERS.has(id)
          ? { supportsCooperativeAbort: true, abortGraceMs: 1500 }
          : {}),
        vramLeaseMode: 'exclusive',
        ...(id === 'localfold-af2'
          ? {
              getWorkerIsolationKey: (params) =>
                LOCALFOLD_MODEL_OPTIONS.find(
                  (m) => m.id === params.model && m.family === 'esmfold2',
                )?.id,
            }
          : {}),
      },
      cachePolicy: LEGACY_CACHE_POLICIES[id] ?? {},
      share: buildShareSpec(id),
      catalog: {
        categoryId: manifest.stage,
        conciseRole: CONCISE_BLOCK_ROLES[id] ?? manifest.description,
      },
      ...(isGpu
        ? {
            modelCard: {
              resolveCard: (params) =>
                (getModelCardForAdapter(id, params) as unknown as KernelModelCardSpec) ?? {
                  adapterId: id,
                  displayName: manifest.displayName,
                  license: manifest.licenseNote,
                  precisions: [],
                  limitations: [],
                },
            },
          }
        : {}),
      display: {
        formatCanvasSummary: () => manifest.shortName,
        ...(id === 'structure-compare'
          ? { displayOnlyParamKeys: ['layout', 'alignToReference', 'showAllChains'] }
          : {}),
        ...(id === 'rfd4-proteina' || id === 'genie3' ? { isSelfGenerator: true } : {}),
      },
      preflight: () => ({
        estimatedVramMB: manifest.estimatedVramMB,
        estimatedDownloadMB: manifest.approxWeightsMB,
        workItems: 1,
        stageBreakdown: { [manifest.stage]: 1 },
      }),
    };

    plugins.push(plugin);
  }

  return plugins;
}
