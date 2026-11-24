// Layer 1 Core Kernel — Active PluginRegistry Slot (cycle-free runtime lookup).
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

import type { PluginRegistry } from './plugin-registry';

let activeRegistry: PluginRegistry | null = null;

/** Register the active booted PluginRegistry for kernel/engine subsystems. */
export function setActivePluginRegistry(registry: PluginRegistry | null): void {
  activeRegistry = registry;
}

/** Return the active booted PluginRegistry if one has been registered. */
export function getActivePluginRegistry(): PluginRegistry | null {
  return activeRegistry;
}
