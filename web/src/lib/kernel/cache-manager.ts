// Layer 1 Core Kernel — Static PluginCachePolicy Aggregator & Multi-Shard Recovery Helper.
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

import type { PluginCachePolicy } from './plugin-types';
import type { PluginRegistry } from './plugin-registry';

export interface AggregatedCacheNamespaces {
  cacheStorageNames: ReadonlySet<string>;
  cacheStoragePrefixes: readonly string[];
  opfsDirs: readonly string[];
  customStores: ReadonlyArray<{
    pluginId: string;
    store: NonNullable<PluginCachePolicy['customCacheStore']>;
  }>;
}

/**
 * Merge static `PluginCachePolicy` declarations from a `PluginRegistry` with optional
 * legacy baseline namespaces so main-thread cache inspection/eviction never has to spawn
 * a worker first and preserves existing user caches across migrations.
 */
export function aggregatePluginCachePolicies(
  registry: PluginRegistry | null | undefined,
  baseline?: {
    cacheStorageNames?: Iterable<string>;
    cacheStoragePrefixes?: Iterable<string>;
    opfsDirs?: Iterable<string>;
  },
): AggregatedCacheNamespaces {
  const names = new Set<string>(baseline?.cacheStorageNames ?? []);
  const prefixes = new Set<string>(baseline?.cacheStoragePrefixes ?? []);
  const opfsDirs = new Set<string>(baseline?.opfsDirs ?? []);
  const customStores: Array<{
    pluginId: string;
    store: NonNullable<PluginCachePolicy['customCacheStore']>;
  }> = [];

  if (registry) {
    for (const { pluginId, policy } of registry.collectCachePolicies()) {
      for (const n of policy.cacheStorageNames ?? []) names.add(n);
      for (const p of policy.cacheStoragePrefixes ?? []) prefixes.add(p);
      for (const d of policy.opfsDirs ?? []) opfsDirs.add(d);
      if (policy.usesSharedOpfsStore) {
        opfsDirs.add('denovo-weights');
        prefixes.add('denovo-run');
      }
      if (policy.customCacheStore) {
        customStores.push({ pluginId, store: policy.customCacheStore });
      }
    }
  }

  return {
    cacheStorageNames: names,
    cacheStoragePrefixes: [...prefixes],
    opfsDirs: [...opfsDirs],
    customStores,
  };
}

/** Check whether a CacheStorage name belongs to any declared plugin or baseline namespace. */
export function isManagedCacheStorageName(
  cacheName: string,
  namespaces: AggregatedCacheNamespaces,
): boolean {
  if (namespaces.cacheStorageNames.has(cacheName)) return true;
  return namespaces.cacheStoragePrefixes.some((p) => cacheName.startsWith(p));
}

/** Invoke targeted partial-shard corruption / quota recovery for a specific plugin if supported. */
export async function recoverPluginCacheStore(
  registry: PluginRegistry,
  pluginId: string,
  keys?: readonly string[],
): Promise<boolean> {
  const plugin = registry.getPlugin(pluginId);
  const recover = plugin?.cachePolicy?.customCacheStore?.recoverFromCorruptOrQuota;
  if (!recover) return false;
  await recover(keys);
  return true;
}
