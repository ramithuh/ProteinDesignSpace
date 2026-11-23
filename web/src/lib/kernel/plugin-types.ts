// Layer 1 Core Kernel — Single-Source-of-Truth ModelPlugin & Open Manifest Contracts.
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

import type { ArtifactEnvelope } from './artifacts';
import type { ConditionIntent, ConditionTargetCompiler } from './condition-intent';

export type { ConditionIntent, ConditionTargetCompiler };

export type RuntimeKind = 'wgsl-custom' | 'ort-webgpu' | 'cpu-ts' | 'wasm-cpu';

export type FallbackPolicy =
  | { kind: 'none'; reason: string }
  | { kind: 'ort-wasm'; wasmAssetDir: string }
  | { kind: 'cpu-js'; moduleUrl: string; degradedWarning: string };

export interface PortSpec {
  /** Single port type ID or polymorphic union of compatible port type IDs. */
  type: string | readonly string[];
  required?: boolean;
  /** When true, this input port accepts multiple incoming wires (fan-in, e.g. structure-compare). */
  multiple?: boolean;
  /** Optional cross-domain port types accepted when the providing DomainPack is loaded (does not fail boot if absent). */
  acceptsOptionalPortTypes?: readonly string[];
}

export interface AdapterModeSpec {
  id: string;
  label: string;
  description: string;
  /** Conjunctive required input port names when this mode is active. */
  requiredInputs?: readonly string[];
  /** Disjunctive Normal Form (OR-of-ANDs) input groups, e.g. `[['structure'], ['receptor', 'ligands']]`. */
  requiredInputGroups?: readonly (readonly string[])[];
}

export type CoreParamFieldKind =
  | 'number'
  | 'integer'
  | 'boolean'
  | 'select'
  | 'multi-select'
  | 'text'
  | 'textarea'
  | 'file-upload'
  | 'custom';

export interface AdapterParamOption {
  value: string;
  label: string;
  hint?: string;
}

export interface AdapterParamField {
  key: string;
  label: string;
  /** Open field kind: core primitives or domain-registered kinds (e.g. 'residue-list', 'aa-set'). */
  kind: CoreParamFieldKind | (string & {});
  tier: 'primary' | 'advanced';
  defaultValue: unknown;
  description?: string;
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: AdapterParamOption[];
  modes?: readonly string[];
  upstreamDefaultHint?: string;
  warnBelow?: {
    threshold: number;
    message: string;
    modes?: readonly string[];
    fixValue?: number;
  };
  /** Optional per-field widget ID resolved by ParamWidgetRegistry when kind === 'custom' or domain-specialized. */
  fieldWidgetId?: string;
  /** Domain-specific field options validated by DomainPack.paramFieldKinds. */
  fieldProps?: Record<string, unknown>;
}

export interface KernelAdapterManifest {
  id: string;
  displayName: string;
  shortName: string;
  /** Primary stage ID validated against DomainPack.stages (no closed protein enum in Layer 1). */
  stage: string;
  /** Optional secondary stage IDs for multi-task / omni foundation models (e.g. Boltz-2 folding+affinity, ESM3). */
  secondaryStages?: readonly string[];
  runtime: RuntimeKind;
  estimatedVramMB: number;
  approxWeightsMB: number;
  precisionWeightsMB?: Record<string, number>;
  nonCommercial?: boolean;
  licenseNote: string;
  description: string;
  inputs: Record<string, PortSpec>;
  outputs: Record<string, PortSpec>;
  modeParamKey?: string;
  modes?: readonly AdapterModeSpec[];
  paramSchema?: readonly AdapterParamField[];
  crossValidate?: (
    params: Record<string, unknown>,
    ctx?: { connectedInputs?: ReadonlySet<string> },
  ) => string[];
}

export interface KernelWorkflowNode<TParams extends Record<string, unknown> = Record<string, unknown>> {
  id: string;
  adapterId: string;
  label?: string;
  params: TParams;
  position?: { x: number; y: number };
}

export interface KernelWorkflowEdge {
  fromNode: string;
  fromPort: string;
  toNode: string;
  toPort: string;
}

export interface KernelWorkflowGraph {
  id: string;
  name: string;
  description?: string;
  nodes: KernelWorkflowNode[];
  edges: KernelWorkflowEdge[];
}

export interface KernelProgressEvent {
  nodeId: string;
  status: 'idle' | 'cached' | 'loading' | 'running' | 'done' | 'error';
  phase?: string;
  progress?: number;
  detail?: string;
  error?: string;
  elapsedMs?: number;
}

export interface KernelCompatibilityFix {
  label: string;
  nodeId: string;
  paramsPatch?: Record<string, unknown>;
  nodeLabel?: string;
}

export interface KernelCompatibilityIssue {
  severity: 'error' | 'warning';
  code: string;
  nodeId: string;
  message: string;
  fix?: KernelCompatibilityFix;
}

export interface KernelExcludedItemRecord {
  key: string;
  id: string;
  index: number;
  reason: string;
}

export interface PluginCachePolicy {
  /** CacheStorage names owned by this plugin (replaces VENDOR_CACHE_NAMES). */
  cacheStorageNames?: readonly string[];
  /** CacheStorage prefixes owned by this plugin (replaces VENDOR_CACHE_PREFIXES). */
  cacheStoragePrefixes?: readonly string[];
  /** OPFS directories owned by this plugin (replaces OPFS_WEIGHT_DIRS). */
  opfsDirs?: readonly string[];
  /** Shared default OPFS namespace when using standard fetchCachedBytes/makeCachedFetch. */
  usesSharedOpfsStore?: boolean;
  /** Optional main-thread-accessible store hooks for sharded/tensor stores (rfd4, localfold). */
  customCacheStore?: {
    inspectBytes?: () => Promise<number>;
    clear?: () => Promise<void>;
    /** Targeted recovery on partial multi-shard corruption or quota exhaustion. */
    recoverFromCorruptOrQuota?: (keys?: readonly string[]) => Promise<void>;
  };
}

export interface UpstreamGraphQuery {
  getIncomingEdges: (
    nodeId: string,
    toPort?: string,
  ) => ReadonlyArray<{ fromNode: string; fromPort: string; toPort: string }>;
  getNode: (nodeId: string) => KernelWorkflowNode | undefined;
  getNodeOutputs: (nodeId: string) => Record<string, ArtifactEnvelope> | undefined;
}

export interface CpuExecutionContext<
  TParams extends Record<string, unknown> = Record<string, unknown>,
  TInputs extends Record<string, ArtifactEnvelope> = Record<string, ArtifactEnvelope>,
> {
  node: KernelWorkflowNode<TParams>;
  inputs: TInputs;
  /** Populated for ports declared with `multiple: true` (fan-in, e.g. structure-compare). */
  multiInputs: { [K in keyof TInputs]?: readonly TInputs[K][] };
  /** Read-only upstream DAG & artifact query. */
  upstreamGraph: UpstreamGraphQuery;
  /** All materialized upstream node outputs indexed by upstream nodeId -> portName. */
  upstreamArtifacts: Readonly<Record<string, Readonly<Record<string, ArtifactEnvelope>>>>;
  onProgress: (ev: Omit<KernelProgressEvent, 'nodeId'>) => void;
  abortSignal: AbortSignal;
}

export interface PostProcessContext<
  TParams extends Record<string, unknown> = Record<string, unknown>,
  TInputs extends Record<string, ArtifactEnvelope> = Record<string, ArtifactEnvelope>,
  TOutputs extends Record<string, ArtifactEnvelope> = Record<string, ArtifactEnvelope>,
> {
  node: KernelWorkflowNode<TParams>;
  inputs: TInputs;
  outputs: TOutputs;
  upstreamGraph: UpstreamGraphQuery;
}

export interface PluginExecutionSpec<
  TParams extends Record<string, unknown> = Record<string, unknown>,
  TInputs extends Record<string, ArtifactEnvelope> = Record<string, ArtifactEnvelope>,
  TOutputs extends Record<string, ArtifactEnvelope> = Record<string, ArtifactEnvelope>,
> {
  runtime: RuntimeKind;
  fallback: FallbackPolicy;
  /** GPU worker factory (lazy Vite import `() => new Worker(...)` or `import(...)`). */
  createWorker?: () => Worker | Promise<Worker>;
  /** Cooperative abort support declaration; when true, scheduler sends `{type:'abort'}` before dispose. */
  supportsCooperativeAbort?: boolean;
  /** Minimum grace window (default 1500 ms, >= 1500 ms) to await `{type:'aborted'}` after in-flight step settlement. */
  abortGraceMs?: number;
  /** GPU resource lease policy (default `'exclusive'`). */
  vramLeaseMode?: 'exclusive' | 'shared-if-fits';
  /** Maximum batch items per run. */
  maxBatchItems?: number;
  /** Optional worker isolation key (e.g. ESMFold2 checkpoints retaining weights for worker lifetime). */
  getWorkerIsolationKey?: (params: TParams) => string | undefined;
  /** Pure CPU execution path for `cpu-ts` plugins. */
  executeCpu?: (ctx: CpuExecutionContext<TParams, TInputs>) => Promise<TOutputs>;
  /** Optional upstream DAG lineage resolver. */
  resolveUpstreamLineage?: (ctx: {
    node: KernelWorkflowNode<TParams>;
    upstreamGraph: UpstreamGraphQuery;
  }) => Record<string, unknown> | string | undefined;
  /** Optional post-validation output normalization/lineage hook. */
  postProcessOutputs?: (ctx: PostProcessContext<TParams, TInputs, TOutputs>) => TOutputs;
  /** Optional live input materialization hook for source nodes. */
  materializePreview?: (params: TParams) => Promise<TOutputs>;
}

export interface PluginShareSpec<TParams extends Record<string, unknown> = Record<string, unknown>> {
  /** Unique 1–3 char code for `#g=` URL compression (validated unique at registry boot). */
  shortCode: string;
  /** Legacy aliases still accepted on `#g=` decode. */
  legacyShortCodes?: readonly string[];
  /** Param keys unconditionally stripped from URL hashes / recipes (transient UI tracking flags). */
  stripOnShareKeys?: readonly string[];
  /** Export-time conditional stripping & upload digest stamping. */
  prepareForShare?: (
    params: TParams,
    opts: { stripLargeUploads: boolean },
  ) => { params: TParams; stripped?: boolean; uploadDigest?: string; expectedFilename?: string };
  /** Optional custom sanitize hook on share/import. */
  sanitizeOnImport?: (params: TParams) => TParams;
  /** Restore omitted inline upload from prior canvas node on YAML apply. */
  restoreOmittedInlineAsset?: (
    prevParams: TParams,
    nextParams: TParams,
    rawRecipeText?: string,
  ) => TParams | null;
  /** Bidirectional YAML key remapping (e.g. `model` <-> `foldModel`). */
  yamlKeyAliases?: Readonly<Record<string, string>>;
}

export interface PortProfileSummary {
  portName: string;
  typeId: string;
  itemCount: number;
  entityKinds?: readonly string[];
  domainMeta?: Record<string, unknown>;
}

export interface PluginPreflightContext {
  params: Record<string, unknown>;
  upstreamProfiles: Record<string, PortProfileSummary>;
  connectedInputs: ReadonlySet<string>;
}

export interface PluginPreflightEstimate {
  estimatedVramMB: number;
  estimatedDownloadMB: number;
  /** Sequential GPU work dispatches. */
  workItems: number;
  /** Produced item counts per output port propagated to downstream nodes during topological DAG preflight. */
  outputCardinality?: Record<string, number>;
  /** Domain-agnostic counter increments by stage ID. */
  stageBreakdown?: Record<string, number>;
  tokenBudget?: number;
  precisionOptions?: Record<string, { weightsMB: number; vramMB: number }>;
  remoteTransfers?: Array<{ destination: string; purpose: string; mode: string }>;
  notices?: string[];
}

export interface PluginSlotDescriptor {
  portName: string;
  direction: 'in' | 'out';
  availability: 'active' | 'inactive-in-mode';
  inactiveReason?: string;
  viewportRole?: string;
}

export interface PluginDisplaySpec<TParams extends Record<string, unknown> = Record<string, unknown>> {
  formatCanvasSummary: (params: TParams, ctx?: { upstreamProfile?: PortProfileSummary }) => string;
  formatActiveBadge?: (params: TParams) => string;
  resolveSlots?: (params: TParams, connectedInputs: ReadonlySet<string>) => PluginSlotDescriptor[];
  displayOnlyParamKeys?: readonly string[];
  taskTrackedParamKeys?: readonly string[];
  onModeSwitch?: (prevParams: TParams, nextMode: string) => TParams;
  compareEligible?: boolean;
  isSelfGenerator?: boolean;
  autoFocusPriority?: number;
  sourceRole?: string;
  pickRoles?: readonly string[];
}

export interface KernelCatalogSubGroup {
  id: string;
  label: string;
  description?: string;
  paramsPatch: Record<string, unknown>;
  nodeLabel?: string;
  approxMB?: number;
  badge?: string;
}

export interface PluginCatalogSpec {
  categoryId: string;
  conciseRole: string;
  buildSubGroups?: (ctx: { defaultParams: Record<string, unknown> }) => KernelCatalogSubGroup[];
  quickInsertRole?: string;
  searchKeywords?: readonly string[];
}

export interface KernelModelCardSpec {
  adapterId: string;
  displayName: string;
  license: string;
  citation?: string;
  weightsRevision?: string;
  activePrecisionId?: string;
  precisions: Array<{
    id: string;
    label: string;
    approxSizeMB: number;
    quantizationDetail: string;
    parityVsPytorchFp32?: Record<string, unknown> | null;
  }>;
  limitations: string[];
  [key: string]: unknown;
}

export interface PluginModelCardSpec<TParams extends Record<string, unknown> = Record<string, unknown>> {
  staticCard?: KernelModelCardSpec;
  resolveCard?: (params: TParams) => KernelModelCardSpec;
  hostedPortCardId?: string;
}

export interface PluginCapabilityProfile {
  supportedEntities: readonly string[];
  requiredGranularity?: string;
  supportsLigands?: 'none' | 'organic-only' | 'all';
  supportsMultimer?: boolean;
  supportsCyclic?: boolean;
  supportsAntibodyNumbering?: boolean;
  maxItems?: number;
  maxResidues?: number;
  domainFlags?: Record<string, unknown>;
}

export interface GuardGraphContext<TParams extends Record<string, unknown> = Record<string, unknown>> {
  node: KernelWorkflowNode<TParams>;
  graph: KernelWorkflowGraph;
  upstreamProfiles: Record<string, PortProfileSummary>;
  upstreamArtifacts: Readonly<Record<string, Readonly<Record<string, ArtifactEnvelope>>>>;
}

export interface GuardRuntimeContext<TParams extends Record<string, unknown> = Record<string, unknown>> {
  node: KernelWorkflowNode<TParams>;
  inputs: Record<string, ArtifactEnvelope>;
}

export interface GuardRuntimeResult {
  warnings: string[];
  excludedCandidates?: KernelExcludedItemRecord[];
  selectedCandidateIds?: string[];
  issues?: KernelCompatibilityIssue[];
}

export interface PluginGuardContributor<TParams extends Record<string, unknown> = Record<string, unknown>> {
  resolveCapabilities: (
    params: TParams,
    ctx?: { connectedInputs?: ReadonlySet<string> },
  ) => PluginCapabilityProfile;
  validateGraphNode?: (ctx: GuardGraphContext<TParams>) => KernelCompatibilityIssue[];
  validateRuntimeInputs?: (ctx: GuardRuntimeContext<TParams>) => GuardRuntimeResult;
}

export interface PreviewMetricItem {
  label: string;
  value: string | number;
  unit?: string;
}

export interface PreviewIssue {
  severity: 'error' | 'warning';
  source: 'schema' | 'domain' | 'runtime' | 'compiler';
  message: string;
  fix?: KernelCompatibilityFix;
}

export interface NodePreviewEnvelope {
  nodeId: string;
  adapterId: string;
  availability: 'ready' | 'partial' | 'pending-upstream' | 'error';
  missingUpstreamNodeIds: string[];
  metrics: PreviewMetricItem[];
  payload: {
    format: string;
    headline: string;
    compiledSummary: Record<string, string | number | boolean | string[]>;
    rawSpecText: string;
  };
  budget: {
    metrics: PreviewMetricItem[];
    nodeWeightsMB: number;
    nodeVramMB: number;
    subgraphDownloadMB: number;
    subgraphPeakVramMB: number;
  };
  issues: PreviewIssue[];
  viewerPayload?: unknown;
  studioPayload?: unknown;
  previewDigest: string;
}

export interface PreviewCompileContext<TParams extends Record<string, unknown> = Record<string, unknown>> {
  node: KernelWorkflowNode<TParams>;
  graph: KernelWorkflowGraph;
  outputsByNode: Readonly<Record<string, Readonly<Record<string, ArtifactEnvelope>>>>;
  abortSignal?: AbortSignal;
}

export interface PluginPreviewContributor<TParams extends Record<string, unknown> = Record<string, unknown>> {
  compilePreview: (
    ctx: PreviewCompileContext<TParams>,
  ) => NodePreviewEnvelope | Promise<NodePreviewEnvelope>;
}

export interface PlannerIntentContext {
  prompt: string;
  domainHints?: readonly string[];
  existingGraph?: KernelWorkflowGraph;
  meta?: Record<string, unknown>;
}

export interface PlannerRepairRule {
  issueCode: string;
  description: string;
  apply: (node: KernelWorkflowNode, issue: KernelCompatibilityIssue) => Partial<KernelWorkflowNode> | null;
}

export type KernelPlannerRole =
  | 'source'
  | 'conditioner'
  | 'generator'
  | 'sequence-designer'
  | 'predictor'
  | 'scorer'
  | 'filter'
  | 'comparator'
  | 'optimizer'
  | 'simulator';

export interface PluginPlannerContributor {
  /** Primary planner role (backwards-compatible single role). */
  plannerRole?: KernelPlannerRole;
  /** Multi-role declaration for omni / multi-task foundation models (e.g. Boltz-2: ['predictor', 'scorer'], ESM3: ['generator', 'sequence-designer', 'predictor']). */
  plannerRoles?: readonly KernelPlannerRole[];
  keywords: readonly string[];
  capabilities: readonly string[];
  extractParamsFromPrompt?: (prompt: string, context: PlannerIntentContext) => Record<string, unknown>;
  repairRules?: readonly PlannerRepairRule[];
}

export interface KernelBenchmarkTargetMeta {
  id: string;
  label: string;
  domainId?: string;
  [key: string]: unknown;
}

export interface PluginBenchmarkContributor<TParams extends Record<string, unknown> = Record<string, unknown>> {
  supportsTarget?: (target: KernelBenchmarkTargetMeta) => boolean;
  toPluginParams: (
    target: KernelBenchmarkTargetMeta,
    prevParams: TParams,
    opts?: { targetLength?: number },
  ) => Partial<TParams>;
  resetParams: (prevParams: TParams) => Partial<TParams>;
}

export interface InspectorSectionDescriptor {
  id: string;
  title: string;
  tier: 'primary' | 'advanced';
  widgetId: string;
}

export interface PluginInspectorContribution<TParams extends Record<string, unknown> = Record<string, unknown>> {
  customWidgetId?: string;
  customSections?: readonly InspectorSectionDescriptor[];
  summaryHeaderExtras?: (params: TParams) => Array<{ label: string; value: string }>;
}

export interface ModelPlugin<
  TParams extends Record<string, unknown> = Record<string, unknown>,
  TInputs extends Record<string, ArtifactEnvelope> = Record<string, ArtifactEnvelope>,
  TOutputs extends Record<string, ArtifactEnvelope> = Record<string, ArtifactEnvelope>,
> {
  readonly id: string;
  readonly domainId: string;
  readonly manifest: KernelAdapterManifest;
  readonly execution: PluginExecutionSpec<TParams, TInputs, TOutputs>;
  readonly cachePolicy: PluginCachePolicy;
  readonly share: PluginShareSpec<TParams>;
  readonly catalog: PluginCatalogSpec;
  readonly modelCard?: PluginModelCardSpec<TParams>;
  readonly display: PluginDisplaySpec<TParams>;
  readonly preflight: (ctx: PluginPreflightContext) => PluginPreflightEstimate;
  readonly guards?: PluginGuardContributor<TParams>;
  readonly preview?: PluginPreviewContributor<TParams>;
  readonly planner?: PluginPlannerContributor;
  readonly benchmark?: PluginBenchmarkContributor<TParams>;
  readonly inspector?: PluginInspectorContribution<TParams>;
  readonly conditioningCompiler?: ConditionTargetCompiler;
}
