// Layer 1 Core Kernel — Single-Source-of-Truth PluginRegistry with Boot Startup Invariants.
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

import { PortTypeRegistry } from './artifacts';
import { DomainPackRegistry, type DomainPack } from './domain-pack';
import type { KernelPlannerRole, ModelPlugin, PluginCachePolicy, PortSpec } from './plugin-types';

function normalizePortTypes(spec: PortSpec): readonly string[] {
  if (typeof spec.type === 'string') return spec.type ? [spec.type] : [];
  return spec.type;
}

function hasStaticCacheDeclaration(policy: PluginCachePolicy | undefined): boolean {
  if (!policy) return false;
  return Boolean(
    policy.usesSharedOpfsStore ||
      (policy.cacheStorageNames && policy.cacheStorageNames.length > 0) ||
      (policy.cacheStoragePrefixes && policy.cacheStoragePrefixes.length > 0) ||
      (policy.opfsDirs && policy.opfsDirs.length > 0) ||
      policy.customCacheStore,
  );
}

/**
 * Central Layer 1 registry unifying DomainPacks, PortTypes, and ModelPlugins.
 * Enforces startup invariants at registration time and freezes after app boot.
 */
export class PluginRegistry {
  readonly portTypes = new PortTypeRegistry();
  readonly domains = new DomainPackRegistry();

  private readonly plugins = new Map<string, ModelPlugin>();
  private readonly shortCodeToPluginId = new Map<string, string>();
  private readonly pluginIdToShortCode = new Map<string, string>();
  private frozen = false;

  registerDomainPack(pack: DomainPack): this {
    if (this.frozen) {
      throw new Error(`PluginRegistry is frozen; cannot register DomainPack "${pack.id}".`);
    }
    this.domains.register(pack);
    for (const pt of pack.portTypes) {
      this.portTypes.register(pt);
    }
    return this;
  }

  registerPlugin(plugin: ModelPlugin): this {
    if (this.frozen) {
      throw new Error(`PluginRegistry is frozen; cannot register plugin "${plugin.id}".`);
    }

    const id = plugin.id.trim();
    if (!id || id !== plugin.manifest.id) {
      throw new Error(
        `Plugin id "${plugin.id}" must be non-empty and match manifest.id ("${plugin.manifest.id}").`,
      );
    }
    if (this.plugins.has(id)) {
      throw new Error(`Duplicate ModelPlugin registration for "${id}".`);
    }

    // Invariant 1: domainId resolves in DomainPackRegistry & stage resolves in DomainPack.stages
    if (!this.domains.has(plugin.domainId)) {
      throw new Error(
        `Plugin "${id}" references unregistered domainId "${plugin.domainId}". Register the DomainPack first.`,
      );
    }
    if (!this.domains.hasStage(plugin.domainId, plugin.manifest.stage)) {
      throw new Error(
        `Plugin "${id}" references stage "${plugin.manifest.stage}" not declared by domain "${plugin.domainId}".`,
      );
    }
    for (const secStage of plugin.manifest.secondaryStages ?? []) {
      const stageExists = this.domains
        .list()
        .some((d) => this.domains.hasStage(d.id, secStage));
      if (!stageExists) {
        throw new Error(
          `Plugin "${id}" references secondary stage "${secStage}" not declared by any registered DomainPack.`,
        );
      }
    }

    // Invariant 2: globally unique share.shortCode (and non-colliding legacyShortCodes)
    const shortCode = plugin.share.shortCode.trim();
    if (!shortCode || shortCode.length > 4) {
      throw new Error(`Plugin "${id}" must provide a 1–4 char share.shortCode.`);
    }
    const existingOwner = this.shortCodeToPluginId.get(shortCode);
    if (existingOwner) {
      throw new Error(
        `Share shortCode "${shortCode}" collision between "${existingOwner}" and "${id}".`,
      );
    }
    for (const legacyCode of plugin.share.legacyShortCodes ?? []) {
      const legacyOwner = this.shortCodeToPluginId.get(legacyCode);
      if (legacyOwner && legacyOwner !== id) {
        throw new Error(
          `Legacy share shortCode "${legacyCode}" collision between "${legacyOwner}" and "${id}".`,
        );
      }
    }

    // Invariant 3: ports resolve in PortTypeRegistry (except soft-wired optional cross-domain ports)
    for (const [portName, spec] of Object.entries(plugin.manifest.inputs)) {
      const declared = normalizePortTypes(spec);
      const hasOptionalCrossDomain =
        spec.required === false && (spec.acceptsOptionalPortTypes?.length ?? 0) > 0;
      if (declared.length === 0 && !hasOptionalCrossDomain) {
        throw new Error(
          `Plugin "${id}" input port "${portName}" must declare at least one PortType.`,
        );
      }
      for (const pt of declared) {
        if (!this.portTypes.has(pt)) {
          throw new Error(
            `Plugin "${id}" input port "${portName}" references unregistered PortType "${pt}".`,
          );
        }
      }
    }
    for (const [portName, spec] of Object.entries(plugin.manifest.outputs)) {
      const declared = normalizePortTypes(spec);
      if (declared.length === 0) {
        throw new Error(
          `Plugin "${id}" output port "${portName}" must declare at least one PortType.`,
        );
      }
      for (const pt of declared) {
        if (!this.portTypes.has(pt)) {
          throw new Error(
            `Plugin "${id}" output port "${portName}" references unregistered PortType "${pt}".`,
          );
        }
      }
    }

    // Invariant 4: GPU/WASM plugins declare createWorker + static cachePolicy
    if (plugin.execution.runtime !== 'cpu-ts') {
      if (typeof plugin.execution.createWorker !== 'function') {
        throw new Error(
          `Plugin "${id}" has runtime "${plugin.execution.runtime}" and must provide execution.createWorker().`,
        );
      }
      if (!hasStaticCacheDeclaration(plugin.cachePolicy)) {
        throw new Error(
          `Plugin "${id}" has runtime "${plugin.execution.runtime}" and must declare static cachePolicy.`,
        );
      }
    }

    this.plugins.set(id, plugin);
    this.pluginIdToShortCode.set(id, shortCode);
    this.shortCodeToPluginId.set(shortCode, id);
    for (const legacyCode of plugin.share.legacyShortCodes ?? []) {
      if (!this.shortCodeToPluginId.has(legacyCode)) {
        this.shortCodeToPluginId.set(legacyCode, id);
      }
    }
    return this;
  }

  freeze(): this {
    this.frozen = true;
    this.portTypes.freeze();
    this.domains.freeze();
    return this;
  }

  isFrozen(): boolean {
    return this.frozen;
  }

  hasPlugin(id: string): boolean {
    return this.plugins.has(id);
  }

  getPlugin(id: string): ModelPlugin | undefined {
    return this.plugins.get(id);
  }

  listPlugins(): ModelPlugin[] {
    return [...this.plugins.values()];
  }

  /** Return all plugins matching a primary or secondary stage (supports multi-task / omni models). */
  findPluginsByStage(stageId: string): ModelPlugin[] {
    return [...this.plugins.values()].filter(
      (p) => p.manifest.stage === stageId || Boolean(p.manifest.secondaryStages?.includes(stageId)),
    );
  }

  /** Return all planner roles declared by a plugin (merges legacy single `plannerRole` + `plannerRoles`). */
  getPluginPlannerRoles(plugin: ModelPlugin): KernelPlannerRole[] {
    const roles: KernelPlannerRole[] = [];
    if (plugin.planner?.plannerRole) roles.push(plugin.planner.plannerRole);
    for (const r of plugin.planner?.plannerRoles ?? []) {
      if (!roles.includes(r)) roles.push(r);
    }
    return roles;
  }

  /** Find all plugins advertising a given planner role (multi-role omni-models match every role they declare). */
  findPluginsByPlannerRole(role: KernelPlannerRole): ModelPlugin[] {
    return [...this.plugins.values()].filter((p) => this.getPluginPlannerRoles(p).includes(role));
  }

  resolveShortCode(shortCode: string): string | undefined {
    return this.shortCodeToPluginId.get(shortCode);
  }

  getShortCode(pluginId: string): string | undefined {
    return this.pluginIdToShortCode.get(pluginId);
  }

  getShortCodeMaps(): {
    adapterToShort: Record<string, string>;
    shortToAdapter: Record<string, string>;
  } {
    return {
      adapterToShort: Object.fromEntries(this.pluginIdToShortCode.entries()),
      shortToAdapter: Object.fromEntries(this.shortCodeToPluginId.entries()),
    };
  }

  /** Resolve all currently-loaded accepted PortTypes for a PortSpec (including loaded optional cross-domain types). */
  resolveAcceptedPortTypes(spec: PortSpec): string[] {
    const out: string[] = [...normalizePortTypes(spec)];
    for (const optType of spec.acceptsOptionalPortTypes ?? []) {
      if (this.portTypes.has(optType) && !out.includes(optType)) {
        out.push(optType);
      }
    }
    return out;
  }

  /** Check whether an output PortSpec can connect to an input PortSpec. */
  arePortsCompatible(fromSpec: PortSpec, toSpec: PortSpec): boolean {
    const outTypes = normalizePortTypes(fromSpec);
    const inTypes = new Set(this.resolveAcceptedPortTypes(toSpec));
    return outTypes.some((t) => inTypes.has(t));
  }

  collectCachePolicies(): Array<{ pluginId: string; policy: PluginCachePolicy }> {
    return [...this.plugins.values()].map((p) => ({
      pluginId: p.id,
      policy: p.cachePolicy,
    }));
  }
}
