// Layer 1 Core Kernel — DomainPack & DomainPackRegistry Contracts.
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

import type {
  ArtifactEnvelope,
  ArtifactViewerDescriptor,
  ParamWidgetDescriptor,
  PortTypeDescriptor,
  StageViewportDescriptor,
  ViewerModelSpec,
} from './artifacts';
import type {
  AdapterParamField,
  GuardRuntimeResult,
  KernelBenchmarkTargetMeta,
  KernelCompatibilityIssue,
  KernelWorkflowGraph,
  KernelWorkflowNode,
  PluginCapabilityProfile,
  PortProfileSummary,
} from './plugin-types';

export interface DomainStageDescriptor {
  readonly id: string;
  readonly label: string;
  readonly order: number;
}

export interface DomainParamFieldKindDescriptor {
  readonly kind: string;
  readonly validateValue?: (value: unknown, field: AdapterParamField) => string | null;
  readonly widgetLoader?: () => Promise<{ default: unknown }>;
}

export interface DomainGuardSuite {
  readonly id: string;
  readonly domainId: string;
  validateGraph?: (ctx: {
    graph: KernelWorkflowGraph;
    outputsByNode: Readonly<Record<string, Readonly<Record<string, ArtifactEnvelope>>>>;
    capabilitiesByNode: Readonly<Record<string, PluginCapabilityProfile>>;
  }) => KernelCompatibilityIssue[];
  validateNodeCapabilities?: (ctx: {
    node: KernelWorkflowNode;
    capabilities: PluginCapabilityProfile;
    upstreamProfiles: Record<string, PortProfileSummary>;
  }) => KernelCompatibilityIssue[];
  validateRuntime?: (ctx: {
    node: KernelWorkflowNode;
    capabilities: PluginCapabilityProfile;
    inputs: Record<string, ArtifactEnvelope>;
  }) => GuardRuntimeResult;
}

export interface DomainBenchmarkProvider {
  readonly targets: readonly KernelBenchmarkTargetMeta[];
  getTargetById: (id: string) => KernelBenchmarkTargetMeta | undefined;
}

export interface DomainOnboardingStep {
  readonly id: string;
  readonly title: string;
  readonly body: string;
  readonly targetCategoryId: string;
  readonly targetSubKey: string;
  readonly autoAddPluginId: string;
}

export interface KeyedComparisonRow<T = unknown> {
  key: string;
  label?: string;
  ordinal?: number;
  metrics: Record<string, number | string | boolean | null>;
  viewerData?: T;
}

export interface CompareCsvColumn {
  key: string;
  header: string;
  format?: (val: unknown) => string;
}

export interface DomainCompareAdapter<TArtifact extends ArtifactEnvelope = ArtifactEnvelope> {
  readonly id: string;
  readonly domainId: string;
  readonly supportedPortTypes: readonly string[];
  extractKeyedRows: (
    artifact: TArtifact,
    ctx: { branchId: string; isSelfGenerator?: boolean },
  ) => KeyedComparisonRow[];
  exportCsvColumns: readonly CompareCsvColumn[];
  buildComparisonView?: (
    branches: ReadonlyArray<{
      branchId: string;
      label: string;
      artifact: TArtifact;
      colorHex: number;
      colorCss: string;
    }>,
    opts: { layout: 'overlay' | 'side-by-side'; pairingMode: 'by-candidate' | 'branch-best' },
  ) => { viewerModels?: ViewerModelSpec[]; summaryMetrics?: Record<string, unknown> };
}

export interface DomainPack {
  readonly id: string;
  readonly displayName: string;
  readonly stages: readonly DomainStageDescriptor[];
  readonly portTypes: readonly PortTypeDescriptor[];
  readonly paramFieldKinds?: readonly DomainParamFieldKindDescriptor[];
  readonly paramWidgets?: readonly ParamWidgetDescriptor[];
  readonly domainGuardSuites?: readonly DomainGuardSuite[];
  readonly compareAdapters?: readonly DomainCompareAdapter[];
  readonly benchmarkProvider?: DomainBenchmarkProvider;
  readonly formatPreflightSummary?: (stageTotals: Record<string, number>) => string | undefined;
  readonly onboardingTour?: readonly DomainOnboardingStep[];
  readonly artifactViewers: readonly ArtifactViewerDescriptor[];
  readonly stageViewports?: readonly StageViewportDescriptor[];
}

export class DomainPackRegistry {
  private readonly packs = new Map<string, DomainPack>();
  private readonly stagesByDomain = new Map<string, Map<string, DomainStageDescriptor>>();
  private readonly paramFieldKinds = new Map<string, DomainParamFieldKindDescriptor>();
  private readonly paramWidgets = new Map<string, ParamWidgetDescriptor>();
  private readonly artifactViewers = new Map<string, ArtifactViewerDescriptor>();
  private readonly stageViewports = new Map<string, StageViewportDescriptor>();
  private readonly compareAdapters: DomainCompareAdapter[] = [];
  private frozen = false;

  register(pack: DomainPack): void {
    if (this.frozen) {
      throw new Error(`DomainPackRegistry is frozen; cannot register "${pack.id}".`);
    }
    if (!pack.id.trim()) {
      throw new Error('DomainPack.id must be non-empty.');
    }
    if (this.packs.has(pack.id)) {
      throw new Error(`Duplicate DomainPack "${pack.id}".`);
    }
    this.packs.set(pack.id, pack);

    const stageMap = new Map<string, DomainStageDescriptor>();
    for (const s of pack.stages) {
      stageMap.set(s.id, s);
    }
    this.stagesByDomain.set(pack.id, stageMap);

    for (const fk of pack.paramFieldKinds ?? []) {
      this.paramFieldKinds.set(fk.kind, fk);
    }
    for (const w of pack.paramWidgets ?? []) {
      this.paramWidgets.set(w.id, w);
    }
    for (const v of pack.artifactViewers) {
      this.artifactViewers.set(v.id, v);
    }
    for (const sv of pack.stageViewports ?? []) {
      this.stageViewports.set(sv.id, sv);
    }
    for (const ca of pack.compareAdapters ?? []) {
      this.compareAdapters.push(ca);
    }
  }

  has(domainId: string): boolean {
    return this.packs.has(domainId);
  }

  get(domainId: string): DomainPack | undefined {
    return this.packs.get(domainId);
  }

  list(): DomainPack[] {
    return [...this.packs.values()];
  }

  hasStage(domainId: string, stageId: string): boolean {
    return Boolean(this.stagesByDomain.get(domainId)?.has(stageId));
  }

  listStages(domainId?: string): DomainStageDescriptor[] {
    if (domainId) {
      return [...(this.stagesByDomain.get(domainId)?.values() ?? [])].sort(
        (a, b) => a.order - b.order,
      );
    }
    const merged = new Map<string, DomainStageDescriptor>();
    for (const map of this.stagesByDomain.values()) {
      for (const s of map.values()) {
        if (!merged.has(s.id)) merged.set(s.id, s);
      }
    }
    return [...merged.values()].sort((a, b) => a.order - b.order);
  }

  getParamFieldKind(kind: string): DomainParamFieldKindDescriptor | undefined {
    return this.paramFieldKinds.get(kind);
  }

  getParamWidget(widgetId: string): ParamWidgetDescriptor | undefined {
    return this.paramWidgets.get(widgetId);
  }

  getArtifactViewer(viewerId: string): ArtifactViewerDescriptor | undefined {
    return this.artifactViewers.get(viewerId);
  }

  getStageViewport(viewportId: string): StageViewportDescriptor | undefined {
    return this.stageViewports.get(viewportId);
  }

  getCompareAdapterForPort(portTypeId: string): DomainCompareAdapter | undefined {
    return this.compareAdapters.find((a) => a.supportedPortTypes.includes(portTypeId));
  }

  freeze(): void {
    this.frozen = true;
  }

  isFrozen(): boolean {
    return this.frozen;
  }
}
