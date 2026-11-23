// Layer 1 Core Kernel — Canonical ArtifactEnvelope, PortTypeRegistry, & 'DNV1' Compatibility Boundary.
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

export const ARTIFACT_ENVELOPE_SCHEMA_VERSION = 'denovo.artifact/v2';

export type KernelCollectionKind =
  | 'independent-candidates'
  | 'multi-backbone-design-set'
  | 'paired-refolds'
  | 'trajectory'
  | 'ensemble'
  | 'conformational-ensemble'
  | 'md-trajectory'
  | (string & {});

export type ThermodynamicEnergyUnit = 'kcal/mol' | 'kJ/mol' | 'eV' | 'kT';

/**
 * First-class metadata for conformational ensembles, MD trajectories, and MLIP/Boltzmann state distributions
 * (BioEmu, AlphaFlow, MDGen, Dyna-1, MACE/UMA relaxation, docking pose Boltzmann weighting).
 */
export interface ThermodynamicEnsembleMeta {
  ensembleKind:
    | 'conformational-ensemble'
    | 'md-trajectory'
    | 'denoising-trajectory'
    | 'independent-designs';
  frameCount: number;
  /** Optional normalized Boltzmann / state probability weights (length === frameCount). */
  weights?: Float32Array;
  /** Optional per-frame potential or free energies (length === frameCount). */
  energies?: Float32Array;
  energyUnit?: ThermodynamicEnergyUnit;
  temperatureK?: number;
  timestepFs?: number;
}

/**
 * Compute numerically stable normalized Boltzmann weights `p_i ∝ exp(-E_i / k_B T)`
 * from per-frame energies without floating-point overflow or NaN.
 */
export function computeBoltzmannWeights(
  energies: Float32Array,
  temperatureK = 300,
  unit: ThermodynamicEnergyUnit = 'kcal/mol',
): Float32Array {
  const n = energies.length;
  const out = new Float32Array(n);
  if (n === 0) return out;
  const T = Number.isFinite(temperatureK) && temperatureK > 0 ? temperatureK : 300;
  let kT: number;
  switch (unit) {
    case 'kcal/mol':
      kT = 0.00198720425864083 * T;
      break;
    case 'kJ/mol':
      kT = 0.00831446261815324 * T;
      break;
    case 'eV':
      kT = 8.617333262e-5 * T;
      break;
    case 'kT':
      kT = 1;
      break;
  }
  let minBetaE = Infinity;
  const betaE = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const e = energies[i];
    if (!Number.isFinite(e)) {
      betaE[i] = Infinity;
      continue;
    }
    const v = e / kT;
    betaE[i] = v;
    if (v < minBetaE) minBetaE = v;
  }
  if (!Number.isFinite(minBetaE)) {
    out.fill(1 / n);
    return out;
  }
  let sum = 0;
  const raw = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const w = Number.isFinite(betaE[i]) ? Math.exp(-(betaE[i] - minBetaE)) : 0;
    raw[i] = w;
    sum += w;
  }
  if (sum <= 0 || !Number.isFinite(sum)) {
    out.fill(1 / n);
    return out;
  }
  for (let i = 0; i < n; i++) {
    out[i] = raw[i] / sum;
  }
  return out;
}

export interface KernelRepresentativeSelection {
  strategy: string;
  selectedKey?: string;
  reason?: string;
}

export interface ArtifactLineage {
  parentArtifactIds?: readonly string[];
  /** Domain-agnostic immutable member/candidate/molecule key. */
  itemKey?: string;
  /** Immutable parent item keys across DAG stages. */
  parentKeys?: readonly string[];
  /** Optional grouping/design-family key (e.g. backbone designKey or scaffold series ID). */
  groupKey?: string;
  /** 0-based creation ordinal within a collection (never score rank). */
  ordinal?: number;
  /** 0-based round index for closed-loop iterative optimization / active learning. */
  roundIndex?: number;
  /** Parent round index when derived from a prior optimization round. */
  parentRoundIndex?: number;
  seed?: number;
  collectionKind?: KernelCollectionKind;
  representativeSelection?: KernelRepresentativeSelection;
  /** Optional thermodynamic/conformational ensemble metadata (Boltzmann weights, energies, temperature). */
  ensembleMeta?: ThermodynamicEnsembleMeta;
  /** Domain-specific lineage metadata (e.g. sourceStructureId, sequenceKey, foldKey). */
  domainMeta?: Record<string, unknown>;
}

export interface ArtifactProvenance {
  quantization?: 'fp32' | 'fp16' | 'int8' | 'int5' | string;
  degradedFallback?: string;
  warnings?: string[];
  elapsedMs?: number;
  runtimeDigest?: string;
  weightsDigest?: string;
  /** Domain-specific provenance (e.g. msaMode, msaDepth, recyclesRun). */
  domainMeta?: Record<string, unknown>;
}

export interface ArtifactEnvelope<TKind extends string = string, TPayload = unknown> {
  /** Extensible port/artifact kind identifier (e.g. 'StructureArtifact', 'PropertyTable'). */
  kind: TKind;
  /** Semantic schema version for forward-compatible blob hydration. */
  schemaVersion: string;
  id: string;
  label?: string;
  sourceModelId: string;
  domainId?: string;
  lineage?: ArtifactLineage;
  provenance?: ArtifactProvenance;
  /** Canonical domain-typed payload (single runtime shape). */
  payload: TPayload;
}

export interface ArtifactExporter<TArtifact extends ArtifactEnvelope = ArtifactEnvelope> {
  readonly id: string;
  readonly label: string;
  readonly fileExtension: string;
  readonly mimeType: string;
  exportBlob: (artifact: TArtifact) => Uint8Array | string;
}

export interface StageViewportDescriptor {
  readonly id: string;
  readonly domainId: string;
  readonly label: string;
  readonly componentLoader: () => Promise<{ default: unknown }>;
  readonly supportedFormats?: readonly string[];
  readonly supportsMultiModelOverlay?: boolean;
}

export interface ArtifactViewerDescriptor {
  readonly id: string;
  readonly domainId: string;
  readonly label: string;
  readonly componentLoader: () => Promise<{ default: unknown }>;
  readonly supportedPortTypes?: readonly string[];
}

export interface ParamWidgetDescriptor {
  readonly id: string;
  readonly domainId?: string;
  readonly scope: 'field' | 'node';
  readonly componentLoader: () => Promise<{ default: unknown }>;
}

export interface ViewerModelSpec {
  id: string;
  nodeId: string | null;
  badge?: string;
  label: string;
  /** Multi-domain structure/molecule payload format. */
  format: 'pdb' | 'mmcif' | 'sdf' | 'mol2';
  text: string;
  colorHex: number;
  colorCss: string;
  alpha: number;
  visible: boolean;
  isReference?: boolean;
  metricSummary?: string;
  domainMeta?: Record<string, unknown>;
}

export interface PortTypeDescriptor<TArtifact extends ArtifactEnvelope = ArtifactEnvelope> {
  typeId: string;
  domainId: string;
  label: string;
  color: { bg: string; fg: string; border: string };
  /** Higher priority wins default Peek slot selection. */
  peekPriority: number;
  /** True if this port renders in the 3D/2D main stage viewport. */
  hasStageViewport?: boolean;
  /** StageViewportDescriptor ID mounted by StageViewportHost when inspecting this port. */
  stageViewportId?: string;
  /** Related context input/output port names combined on the same stage. */
  stageContextPorts?: readonly string[];
  /** Lightweight summary for Agent / WebMCP / Compare / Node footer. */
  summarize: (artifact: TArtifact) => Record<string, unknown>;
  /** One-line KPI badge shown on completed Node footer. */
  formatNodeFooterKpi?: (artifact: TArtifact) => { text: string; tone?: 'ok' | 'warn' | 'neutral' } | null;
  /** Runtime output sanity validator. */
  validateOutput?: (artifact: TArtifact) => string[];
  /** Supported file export handlers. */
  exporters?: readonly ArtifactExporter<TArtifact>[];
  /** Viewer component key resolved by ArtifactViewerRegistry (for Peek outputs). */
  viewerId: string;
}

const FORBIDDEN_ARTIFACT_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

const ENVELOPE_CORE_KEYS = new Set([
  'kind',
  'schemaVersion',
  'id',
  'label',
  'sourceModelId',
  'domainId',
  'lineage',
  'provenance',
  'payload',
]);

const LEGACY_LINEAGE_KEYS = new Set([
  'parentStructureId',
  'sourceStructureId',
  'designKey',
  'sequenceKey',
  'foldKey',
  'collectionKind',
  'representativeSelection',
]);

/** Type guard verifying whether an unknown object is already a canonical `ArtifactEnvelope`. */
export function isArtifactEnvelope(value: unknown): value is ArtifactEnvelope {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.kind === 'string' &&
    typeof obj.schemaVersion === 'string' &&
    typeof obj.id === 'string' &&
    typeof obj.sourceModelId === 'string' &&
    'payload' in obj
  );
}

/**
 * Lossless 'DNV1' hydration boundary: converts either a canonical `ArtifactEnvelope`
 * or a legacy flat `CanonicalArtifact` record (types.ts:293-634) into an `ArtifactEnvelope`.
 * Never mutates the input object or alters underlying typed arrays (`Float32Array`, `Uint8Array`, etc.).
 */
export function hydrateLegacyArtifactToEnvelope<TKind extends string = string>(
  raw: Record<string, unknown>,
  defaultDomainId = 'protein-biopolymer',
): ArtifactEnvelope<TKind, Record<string, unknown>> {
  if (isArtifactEnvelope(raw)) {
    return raw as unknown as ArtifactEnvelope<TKind, Record<string, unknown>>;
  }

  const kind = String(raw.kind ?? 'UnknownArtifact') as TKind;
  const id = String(raw.id ?? `art_${Date.now().toString(36)}`);
  const label = typeof raw.label === 'string' ? raw.label : undefined;
  const sourceModelId = String(raw.sourceModelId ?? 'unknown');

  const payload: Record<string, unknown> = {};
  const lineageDomainMeta: Record<string, unknown> = {};
  const parentArtifactIds: string[] = [];

  for (const [k, v] of Object.entries(raw)) {
    if (
      FORBIDDEN_ARTIFACT_KEYS.has(k) ||
      k === 'kind' ||
      k === 'id' ||
      k === 'label' ||
      k === 'sourceModelId' ||
      k === 'domainId'
    ) {
      continue;
    }
    Object.defineProperty(payload, k, {
      value: v,
      enumerable: true,
      writable: true,
      configurable: true,
    });
    if (LEGACY_LINEAGE_KEYS.has(k) && v !== undefined) {
      Object.defineProperty(lineageDomainMeta, k, {
        value: v,
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }
  }

  if (typeof raw.sourceStructureId === 'string' && raw.sourceStructureId && raw.sourceStructureId !== 'none') {
    parentArtifactIds.push(raw.sourceStructureId);
  }
  if (
    typeof raw.parentStructureId === 'string' &&
    raw.parentStructureId &&
    raw.parentStructureId !== 'none' &&
    !parentArtifactIds.includes(raw.parentStructureId)
  ) {
    parentArtifactIds.push(raw.parentStructureId);
  }

  const itemKey =
    typeof raw.foldKey === 'string'
      ? raw.foldKey
      : typeof raw.sequenceKey === 'string'
        ? raw.sequenceKey
        : typeof raw.designKey === 'string'
          ? raw.designKey
          : undefined;
  const groupKey = typeof raw.designKey === 'string' ? raw.designKey : undefined;

  const lineage: ArtifactLineage | undefined =
    parentArtifactIds.length > 0 ||
    itemKey !== undefined ||
    groupKey !== undefined ||
    Object.keys(lineageDomainMeta).length > 0
      ? {
          ...(parentArtifactIds.length > 0 ? { parentArtifactIds } : {}),
          ...(itemKey !== undefined ? { itemKey } : {}),
          ...(groupKey !== undefined ? { groupKey } : {}),
          ...(typeof raw.collectionKind === 'string'
            ? { collectionKind: raw.collectionKind as KernelCollectionKind }
            : {}),
          ...(raw.representativeSelection && typeof raw.representativeSelection === 'object'
            ? {
                representativeSelection:
                  raw.representativeSelection as KernelRepresentativeSelection,
              }
            : {}),
          ...(Object.keys(lineageDomainMeta).length > 0 ? { domainMeta: lineageDomainMeta } : {}),
        }
      : undefined;

  const rawWarnings = Array.isArray(raw.warnings)
    ? raw.warnings.filter((w): w is string => typeof w === 'string')
    : undefined;
  const rawElapsed = typeof raw.elapsedMs === 'number' ? raw.elapsedMs : undefined;
  const provenance: ArtifactProvenance | undefined =
    rawWarnings?.length || rawElapsed !== undefined
      ? {
          ...(rawWarnings?.length ? { warnings: rawWarnings } : {}),
          ...(rawElapsed !== undefined ? { elapsedMs: rawElapsed } : {}),
        }
      : undefined;

  return {
    kind,
    schemaVersion: ARTIFACT_ENVELOPE_SCHEMA_VERSION,
    id,
    ...(label !== undefined ? { label } : {}),
    sourceModelId,
    domainId: typeof raw.domainId === 'string' ? raw.domainId : defaultDomainId,
    ...(lineage ? { lineage } : {}),
    ...(provenance ? { provenance } : {}),
    payload,
  };
}

/**
 * Lossless 'DNV1' serialization boundary: writes canonical `ArtifactEnvelope` directly
 * (or projects back to legacy flat shape during Phase 1–3 transitions when `legacyFlat: true`).
 */
export function serializeEnvelopeForDnv1(
  env: ArtifactEnvelope,
  opts?: { legacyFlat?: boolean },
): Record<string, unknown> {
  if (!opts?.legacyFlat) {
    return { ...env };
  }
  const flat: Record<string, unknown> = {
    kind: env.kind,
    id: env.id,
    ...(env.label !== undefined ? { label: env.label } : {}),
    sourceModelId: env.sourceModelId,
  };
  if (env.payload && typeof env.payload === 'object' && !Array.isArray(env.payload)) {
    for (const [k, v] of Object.entries(env.payload as Record<string, unknown>)) {
      if (!ENVELOPE_CORE_KEYS.has(k) && !FORBIDDEN_ARTIFACT_KEYS.has(k)) {
        Object.defineProperty(flat, k, {
          value: v,
          enumerable: true,
          writable: true,
          configurable: true,
        });
      }
    }
  }
  if (env.lineage?.domainMeta) {
    for (const [k, v] of Object.entries(env.lineage.domainMeta)) {
      if (!(k in flat) && !FORBIDDEN_ARTIFACT_KEYS.has(k)) {
        Object.defineProperty(flat, k, {
          value: v,
          enumerable: true,
          writable: true,
          configurable: true,
        });
      }
    }
  }
  return flat;
}

/** Open registry of all DomainPack-provided PortTypes. */
export class PortTypeRegistry {
  private readonly descriptors = new Map<string, PortTypeDescriptor>();
  private frozen = false;

  register(descriptor: PortTypeDescriptor): void {
    if (this.frozen) {
      throw new Error(`PortTypeRegistry is frozen; cannot register "${descriptor.typeId}".`);
    }
    if (!descriptor.typeId.trim()) {
      throw new Error('PortTypeDescriptor.typeId must be non-empty.');
    }
    const existing = this.descriptors.get(descriptor.typeId);
    if (existing) {
      if (existing.domainId === descriptor.domainId) {
        throw new Error(
          `Duplicate PortType "${descriptor.typeId}" in domain "${descriptor.domainId}".`,
        );
      }
      throw new Error(
        `PortType "${descriptor.typeId}" collision between domains "${existing.domainId}" and "${descriptor.domainId}".`,
      );
    }
    this.descriptors.set(descriptor.typeId, descriptor);
  }

  has(typeId: string): boolean {
    return this.descriptors.has(typeId);
  }

  get(typeId: string): PortTypeDescriptor | undefined {
    return this.descriptors.get(typeId);
  }

  list(): PortTypeDescriptor[] {
    return [...this.descriptors.values()];
  }

  resolveColor(typeId: string): { bg: string; fg: string; border: string } | undefined {
    return this.descriptors.get(typeId)?.color;
  }

  resolvePeekPriority(typeId: string): number {
    return this.descriptors.get(typeId)?.peekPriority ?? 0;
  }

  summarize(artifact: ArtifactEnvelope): Record<string, unknown> {
    const desc = this.descriptors.get(artifact.kind);
    if (desc) return desc.summarize(artifact);
    return {
      kind: artifact.kind,
      id: artifact.id,
      ...(artifact.label ? { label: artifact.label } : {}),
    };
  }

  validateOutput(artifact: ArtifactEnvelope): string[] {
    const desc = this.descriptors.get(artifact.kind);
    return desc?.validateOutput ? desc.validateOutput(artifact) : [];
  }

  freeze(): void {
    this.frozen = true;
  }

  isFrozen(): boolean {
    return this.frozen;
  }
}
