// Layer 1 Core Kernel — Dynamic PlannerCatalogIndex built from PluginRegistry.
// Eliminates closed adapter unions (`GeneratorId`, `SUPPORTED_BLOCK_NAMES`, `PASS2_GENERATOR_ADAPTERS`)
// so registering plugin #27+ automatically makes it discoverable to the planner & NL repair passes.
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

import type { PluginRegistry } from '../plugin-registry';
import type {
  KernelCompatibilityIssue,
  KernelPlannerRole,
  KernelWorkflowNode,
  ModelPlugin,
} from '../plugin-types';

export function transliterateBlockName(raw: string): string {
  return raw.toLowerCase().replace(/[αΑ]/g, 'a');
}

export function normalizeBlockToken(raw: string): string {
  return transliterateBlockName(raw)
    .replace(/\(.*?\)/g, '')
    .replace(/[^a-z0-9]+/g, '');
}

const STAGE_TO_DEFAULT_ROLE: Readonly<Record<string, KernelPlannerRole>> = {
  source: 'source',
  condition: 'conditioner',
  generate: 'generator',
  'inverse-fold': 'sequence-designer',
  fold: 'predictor',
  score: 'scorer',
};

export interface PlannerPluginSummary {
  id: string;
  domainId: string;
  displayName: string;
  shortName: string;
  stage: string;
  secondaryStages: readonly string[];
  roles: readonly KernelPlannerRole[];
  keywords: readonly string[];
  capabilities: readonly string[];
}

export interface PlannerCatalogIndex {
  readonly plugins: readonly PlannerPluginSummary[];
  readonly supportedBlockTokens: ReadonlySet<string>;
  getPlugin(id: string): ModelPlugin | undefined;
  findByRole(role: KernelPlannerRole): PlannerPluginSummary[];
  hasRole(adapterId: string, role: KernelPlannerRole): boolean;
  findByKeyword(query: string): PlannerPluginSummary[];
  isSupportedBlockName(raw: string): boolean;
  getGeneratorIds(): string[];
  getSequenceDesignerIds(): string[];
  getPredictorIds(): string[];
  applyRepairRules(
    node: KernelWorkflowNode,
    issue: KernelCompatibilityIssue,
  ): Partial<KernelWorkflowNode> | null;
}

function inferRolesForPlugin(plugin: ModelPlugin, registry: PluginRegistry): KernelPlannerRole[] {
  const explicit = registry.getPluginPlannerRoles(plugin);
  if (explicit.length > 0) return explicit;
  const inferred: KernelPlannerRole[] = [];
  const primary = STAGE_TO_DEFAULT_ROLE[plugin.manifest.stage];
  if (primary) inferred.push(primary);
  for (const sec of plugin.manifest.secondaryStages ?? []) {
    const r = STAGE_TO_DEFAULT_ROLE[sec];
    if (r && !inferred.includes(r)) inferred.push(r);
  }
  return inferred;
}

/**
 * Build an immutable `PlannerCatalogIndex` directly from any booted `PluginRegistry`.
 * Supports multi-role omni-models (`plannerRoles`), custom keywords, and plugin-owned repair rules.
 */
export function buildPlannerCatalogIndex(registry: PluginRegistry): PlannerCatalogIndex {
  const summaries: PlannerPluginSummary[] = [];
  const byId = new Map<string, PlannerPluginSummary>();
  const byRole = new Map<KernelPlannerRole, PlannerPluginSummary[]>();
  const tokenToPlugins = new Map<string, Set<string>>();
  const supportedBlockTokens = new Set<string>();

  const registerToken = (pluginId: string, rawToken: string) => {
    const full = transliterateBlockName(rawToken).replace(/[^a-z0-9]+/g, '');
    const stripped = normalizeBlockToken(rawToken);
    for (const tok of [full, stripped]) {
      if (!tok) continue;
      supportedBlockTokens.add(tok);
      let set = tokenToPlugins.get(tok);
      if (!set) {
        set = new Set();
        tokenToPlugins.set(tok, set);
      }
      set.add(pluginId);
    }
  };

  for (const plugin of registry.listPlugins()) {
    const roles = inferRolesForPlugin(plugin, registry);
    const keywords = [
      ...(plugin.planner?.keywords ?? []),
      ...(plugin.catalog.searchKeywords ?? []),
    ];
    const capabilities = [...(plugin.planner?.capabilities ?? [])];

    const summary: PlannerPluginSummary = {
      id: plugin.id,
      domainId: plugin.domainId,
      displayName: plugin.manifest.displayName,
      shortName: plugin.manifest.shortName,
      stage: plugin.manifest.stage,
      secondaryStages: [...(plugin.manifest.secondaryStages ?? [])],
      roles,
      keywords,
      capabilities,
    };

    summaries.push(summary);
    byId.set(plugin.id, summary);

    for (const role of roles) {
      let arr = byRole.get(role);
      if (!arr) {
        arr = [];
        byRole.set(role, arr);
      }
      arr.push(summary);
    }

    registerToken(plugin.id, plugin.id);
    registerToken(plugin.id, plugin.manifest.shortName);
    registerToken(plugin.id, plugin.manifest.displayName);
    for (const kw of keywords) {
      registerToken(plugin.id, kw);
    }
  }

  return {
    plugins: summaries,
    supportedBlockTokens,
    getPlugin(id: string) {
      return registry.getPlugin(id);
    },
    findByRole(role: KernelPlannerRole) {
      return [...(byRole.get(role) ?? [])];
    },
    hasRole(adapterId: string, role: KernelPlannerRole) {
      return Boolean(byId.get(adapterId)?.roles.includes(role));
    },
    findByKeyword(query: string) {
      const full = transliterateBlockName(query).replace(/[^a-z0-9]+/g, '');
      const stripped = normalizeBlockToken(query);
      const matchedIds = new Set<string>();
      for (const tok of [full, stripped]) {
        if (!tok) continue;
        const exact = tokenToPlugins.get(tok);
        if (exact) {
          for (const id of exact) matchedIds.add(id);
        }
        // Also match substring containment on normalized keywords/ids
        for (const [k, ids] of tokenToPlugins.entries()) {
          if (tok.length >= 3 && (k.includes(tok) || tok.includes(k))) {
            for (const id of ids) matchedIds.add(id);
          }
        }
      }
      return [...matchedIds]
        .map((id) => byId.get(id))
        .filter((s): s is PlannerPluginSummary => Boolean(s));
    },
    isSupportedBlockName(raw: string) {
      const full = transliterateBlockName(raw).replace(/[^a-z0-9]+/g, '');
      const stripped = normalizeBlockToken(raw);
      return Boolean(
        (stripped && supportedBlockTokens.has(stripped)) ||
          (full && supportedBlockTokens.has(full)),
      );
    },
    getGeneratorIds() {
      return (byRole.get('generator') ?? []).map((p) => p.id);
    },
    getSequenceDesignerIds() {
      return (byRole.get('sequence-designer') ?? []).map((p) => p.id);
    },
    getPredictorIds() {
      return (byRole.get('predictor') ?? []).map((p) => p.id);
    },
    applyRepairRules(node: KernelWorkflowNode, issue: KernelCompatibilityIssue) {
      const plugin = registry.getPlugin(node.adapterId);
      const rules = plugin?.planner?.repairRules;
      if (!rules || rules.length === 0) return null;
      for (const rule of rules) {
        if (rule.issueCode === issue.code || rule.issueCode === '*') {
          const patch = rule.apply(node, issue);
          if (patch) return patch;
        }
      }
      return null;
    },
  };
}
