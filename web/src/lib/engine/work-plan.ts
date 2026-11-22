// Wave-6A Multi-Design & Batching Core: Deterministic Work Planner,
// Stable Domain-Separated Seed Derivation, and Pure Top-K Filter / Ranker.
//
// Authoritative pure planner shared identically by UI, preflight, workers, and export
// (RFC §4.2, §5.1, Gates #2, #7, #8, #10).

import {
  canonicalJson,
  computeStructurePayloadDigest,
  isDesignCollection,
  makeDesignKey,
  makeFoldKey,
  makeSequenceKey,
  normalizeToCollection,
  sha256Hex,
  type CollectionMember,
  type DesignCollection,
  type Digest,
  type ItemLineage,
  type MemberKey,
  type SequenceCollection,
  type StructureCollection,
  type StructureIdentity,
  type WorkItemOutcome,
} from './collections';
import type { SequenceCandidate, StructureArtifact } from './types';

export const SEED_DOMAIN_SPAN = 2 ** 28; // 268,435,456 (fits in signed 31-bit int across 8 domains)

export const SEED_DOMAINS = {
  conditioning: 0,
  generation: 1,
  mpnn: 2,
  fold: 3,
  ensemble: 4,
  residueTrack: 5,
  reserved6: 6,
  reserved7: 7,
} as const;

export type SeedDomain = keyof typeof SEED_DOMAINS;

export const MAX_DESIGNS = 16;
export const MAX_SEQUENCES_PER_DESIGN = 8;
export const DEFAULT_PER_DESIGN_TOP_K = 2;
export const DEFAULT_MAX_FOLDS = 32;
export const HARD_MAX_FOLDS = 256;
export const MAX_SAMPLES_PER_SEQUENCE = 16;

// ============================================================================
// Stable Domain-Separated Seed Derivation (RFC §4.2, Gate #7)
// ============================================================================

function resolveDomainIndex(domain: SeedDomain | number): number {
  const idx = typeof domain === 'number' ? domain : SEED_DOMAINS[domain];
  if (!Number.isInteger(idx) || idx < 0 || idx > 7) {
    throw new RangeError(`Seed domain must be an integer 0..7 or valid SeedDomain; got ${String(domain)}`);
  }
  return idx;
}

function normalizeBaseSeed(baseSeed: number): number {
  if (!Number.isFinite(baseSeed)) {
    throw new RangeError(`baseSeed must be a finite number; got ${String(baseSeed)}`);
  }
  return Math.trunc(baseSeed) >>> 0;
}

/**
 * Derive a deterministic, collision-resistant seed for a single work item.
 *
 * Properties (RFC Gate #7):
 * - Disjoint numeric ranges across domains (`domainIndex * 2^28 + offset`).
 * - Depends ONLY on `(baseSeed, domain, itemKey)` — invariant to MPNN score sorting,
 *   Top-K policy changes, cache hits/skips, scheduling order, or campaign extension.
 * - Always returns a non-negative 31-bit integer in `[0, 2^31 - 1]` safe for JS, WebGPU, and ONNX RNGs.
 */
export function deriveStableItemSeed(
  baseSeed: number,
  domain: SeedDomain | number,
  itemKey: MemberKey,
): number {
  const normalizedBase = normalizeBaseSeed(baseSeed);
  const domainIndex = resolveDomainIndex(domain);
  if (typeof itemKey !== 'string' || itemKey.trim().length === 0) {
    throw new RangeError('itemKey must be a non-empty string');
  }
  const digest = sha256Hex(`denovo-seed/v2|${normalizedBase}|${domainIndex}|${itemKey}`);
  const offset = parseInt(digest.slice(0, 8), 16) & (SEED_DOMAIN_SPAN - 1);
  return domainIndex * SEED_DOMAIN_SPAN + offset;
}

/**
 * Injective ordinal stage seed helper matching RFC §4.2.
 * Rejects unsupported ordinals rather than silently wrapping.
 */
export function deriveOrdinalStageSeed(
  domain: SeedDomain | number,
  stageOffset: number,
  ordinal: number,
): number {
  const domainIndex = resolveDomainIndex(domain);
  if (!Number.isInteger(ordinal) || ordinal < 0 || ordinal >= SEED_DOMAIN_SPAN) {
    throw new RangeError(`ordinal must be an integer in [0, ${SEED_DOMAIN_SPAN - 1}]; got ${ordinal}`);
  }
  if (!Number.isInteger(stageOffset) || stageOffset < 0) {
    throw new RangeError(`stageOffset must be a non-negative integer; got ${stageOffset}`);
  }
  return domainIndex * SEED_DOMAIN_SPAN + ((stageOffset + ordinal) % SEED_DOMAIN_SPAN);
}

/**
 * Strip non-scientific parameters that must NEVER alter item identity or family keys
 * (RFC §4.1: exclude numDesigns, numSequences, maxFolds, Top-K, runId, timestamps, labels).
 */
const EXCLUDED_IDENTITY_PARAM_KEYS = new Set([
  'numDesigns',
  'numSequences',
  'sequencesPerBackbone',
  'perDesignTopK',
  'maxSequences',
  'maxFolds',
  'maxTotal',
  'topK',
  'runId',
  'nodeLabel',
  'label',
  'timestamp',
  'selectedKey',
]);

function normalizeScientificParams(params?: Record<string, unknown>): Record<string, unknown> {
  if (!params) return {};
  const cleaned: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(params)) {
    if (EXCLUDED_IDENTITY_PARAM_KEYS.has(k) || v === undefined) continue;
    cleaned[k] = v;
  }
  return cleaned;
}

// ============================================================================
// 1. Generator Multi-Design Work Plan (RFC Gate #2)
// ============================================================================

export interface GeneratorPlanParams {
  modelId: string;
  /** Number of independent backbones to generate (1..16). */
  numDesigns: number;
  baseSeed: number;
  /** Scientific parameters (conditioning, steps, temperature, etc.); excludes numDesigns. */
  scientificParams?: Record<string, unknown>;
  inputDigest?: Digest;
  runtimeDigest?: Digest;
  weightsDigest?: Digest;
  /** Optional set of already-cached designKeys (never alters keys or seeds). */
  cachedKeys?: ReadonlySet<MemberKey>;
}

export interface PlannedDesignItem {
  key: MemberKey;
  ordinal: number;
  seed: number;
  seedScheme: 'domain-v2';
  lineage: ItemLineage;
  cached: boolean;
}

export interface GeneratorDesignPlan {
  familyKey: Digest;
  planDigest: Digest;
  modelId: string;
  numDesigns: number;
  baseSeed: number;
  items: readonly PlannedDesignItem[];
  cachedCount: number;
  newCount: number;
  summary: string;
}

/**
 * Plan independent backbone designs for a generator (numDesigns 1..16).
 * Extension stability (RFC Gate #2): extending 4 -> 8 keeps identical familyKey,
 * designKeys, and seeds for items 0..3.
 */
export function planGeneratorDesigns(params: GeneratorPlanParams): GeneratorDesignPlan {
  if (typeof params.modelId !== 'string' || params.modelId.trim().length === 0) {
    throw new RangeError('modelId must be a non-empty string');
  }
  if (!Number.isInteger(params.numDesigns) || params.numDesigns < 1 || params.numDesigns > MAX_DESIGNS) {
    throw new RangeError(`numDesigns must be an integer between 1 and ${MAX_DESIGNS}; got ${String(params.numDesigns)}`);
  }
  const baseSeed = normalizeBaseSeed(params.baseSeed);
  const scientific = normalizeScientificParams(params.scientificParams);

  const familyKey = sha256Hex(
    canonicalJson({
      schema: 'denovo.family-generator/v1',
      modelId: params.modelId.trim(),
      seedScheme: 'domain-v2',
      baseSeed,
      scientificParams: scientific,
      inputDigest: params.inputDigest ?? '',
      runtimeDigest: params.runtimeDigest ?? '',
      weightsDigest: params.weightsDigest ?? '',
    }),
  );

  const items: PlannedDesignItem[] = [];
  let cachedCount = 0;
  for (let designIndex = 0; designIndex < params.numDesigns; designIndex++) {
    const key = makeDesignKey(familyKey, designIndex);
    const seed = deriveStableItemSeed(baseSeed, 'generation', key);
    const cached = params.cachedKeys?.has(key) ?? false;
    if (cached) cachedCount++;
    items.push({
      key,
      ordinal: designIndex,
      seed,
      seedScheme: 'domain-v2',
      lineage: {
        designKey: key,
        designIndex,
        parentKeys: [],
      },
      cached,
    });
  }

  const newCount = params.numDesigns - cachedCount;
  const planDigest = sha256Hex(
    canonicalJson({
      schema: 'denovo.plan-generator/v1',
      familyKey,
      numDesigns: params.numDesigns,
      keys: items.map((i) => i.key),
    }),
  );

  const unit = params.numDesigns === 1 ? 'backbone' : 'backbones';
  const summary =
    cachedCount > 0
      ? `${params.numDesigns} ${unit} (${cachedCount} cached · ${newCount} to run)`
      : `${params.numDesigns} ${unit}`;

  return {
    familyKey,
    planDigest,
    modelId: params.modelId.trim(),
    numDesigns: params.numDesigns,
    baseSeed,
    items,
    cachedCount,
    newCount,
    summary,
  };
}

// ============================================================================
// 2. Inverse-Folding (MPNN) Fan-out Plan (N backbones x K sequences, RFC Gate #10)
// ============================================================================

export interface MpnnFanoutParams {
  modelId?: string;
  /** Number of sequences to sample per backbone (1..8). */
  sequencesPerBackbone: number;
  baseSeed: number;
  temperature?: number;
  scientificParams?: Record<string, unknown>;
  runtimeDigest?: Digest;
  weightsDigest?: Digest;
  cachedKeys?: ReadonlySet<MemberKey>;
}

export interface PlannedMpnnItem {
  key: MemberKey;
  designKey: MemberKey;
  designIndex: number;
  sequenceIndex: number;
  ordinal: number;
  seed: number;
  seedScheme: 'domain-v2';
  sourceStructure: StructureIdentity;
  lineage: ItemLineage;
  /** Design-specific constraints copied defensively per backbone (RFC Gate #10). */
  lockedChains: readonly string[];
  fixedResidueIndices: readonly number[];
  motifIndices: readonly number[];
  tiedPositions?: string;
  chainRoles?: Readonly<Record<string, 'target' | 'binder' | 'scaffold'>>;
  backboneLength: number;
  cached: boolean;
}

export interface BackboneMpnnSubplan {
  designKey: MemberKey;
  designIndex: number;
  sourceStructure: StructureIdentity;
  lockedChains: readonly string[];
  fixedResidueIndices: readonly number[];
  motifIndices: readonly number[];
  tiedPositions?: string;
  chainRoles?: Readonly<Record<string, 'target' | 'binder' | 'scaffold'>>;
  backboneLength: number;
  items: readonly PlannedMpnnItem[];
}

export interface MpnnFanoutPlan {
  familyKey: Digest;
  planDigest: Digest;
  modelId: string;
  backboneCount: number;
  sequencesPerBackbone: number;
  totalPlannedSequences: number;
  perBackbonePlans: readonly BackboneMpnnSubplan[];
  items: readonly PlannedMpnnItem[];
  cachedCount: number;
  newCount: number;
  summary: string;
}

function normalizeStructureInputs(
  structures:
    | StructureCollection
    | readonly CollectionMember<StructureArtifact>[]
    | readonly StructureArtifact[],
): CollectionMember<StructureArtifact>[] {
  if (isDesignCollection(structures)) {
    return [...structures.members];
  }
  const list: CollectionMember<StructureArtifact>[] = [];
  for (let i = 0; i < structures.length; i++) {
    const item = structures[i];
    if ('kind' in item && item.kind === 'StructureArtifact') {
      const col = normalizeToCollection(item);
      const member = col.members[0];
      list.push({
        ...member,
        ordinal: i,
        lineage: {
          ...member.lineage,
          designIndex: i,
        },
      });
    } else {
      list.push(item as CollectionMember<StructureArtifact>);
    }
  }
  return list;
}

/**
 * Plan N backbones x K sequences MPNN inverse-folding fan-out.
 * Preserves each backbone's exact designKey, sourceStructure identity, lockedChains,
 * fixedResidueIndices, motifIndices, and chainRoles without cross-backbone leakage.
 */
export function planMpnnFanout(
  structures:
    | StructureCollection
    | readonly CollectionMember<StructureArtifact>[]
    | readonly StructureArtifact[],
  params: MpnnFanoutParams,
): MpnnFanoutPlan {
  const modelId = (params.modelId ?? 'proteinmpnn').trim() || 'proteinmpnn';
  if (
    !Number.isInteger(params.sequencesPerBackbone) ||
    params.sequencesPerBackbone < 1 ||
    params.sequencesPerBackbone > MAX_SEQUENCES_PER_DESIGN
  ) {
    throw new RangeError(
      `sequencesPerBackbone must be an integer between 1 and ${MAX_SEQUENCES_PER_DESIGN}; got ${String(params.sequencesPerBackbone)}`,
    );
  }
  const baseSeed = normalizeBaseSeed(params.baseSeed);
  const scientific = normalizeScientificParams({
    ...params.scientificParams,
    ...(params.temperature !== undefined ? { temperature: params.temperature } : {}),
  });

  const normalizedMembers = normalizeStructureInputs(structures).sort((a, b) =>
    a.ordinal !== b.ordinal ? a.ordinal - b.ordinal : a.key < b.key ? -1 : a.key > b.key ? 1 : 0,
  );

  const familyKey = sha256Hex(
    canonicalJson({
      schema: 'denovo.family-mpnn/v1',
      modelId,
      seedScheme: 'domain-v2',
      baseSeed,
      scientificParams: scientific,
      runtimeDigest: params.runtimeDigest ?? '',
      weightsDigest: params.weightsDigest ?? '',
    }),
  );

  const perBackbonePlans: BackboneMpnnSubplan[] = [];
  const allItems: PlannedMpnnItem[] = [];
  let cachedCount = 0;

  for (let bIdx = 0; bIdx < normalizedMembers.length; bIdx++) {
    const member = normalizedMembers[bIdx];
    const struct = member.payload;
    if (!struct) {
      throw new Error(`Structure member "${member.key}" has no hydrated StructureArtifact payload for MPNN planning.`);
    }
    const designKey = member.lineage.designKey ?? member.key;
    const designIndex = member.lineage.designIndex ?? member.ordinal ?? bIdx;
    const payloadDigest =
      member.lineage.sourceStructure?.payloadDigest || computeStructurePayloadDigest(struct);
    const sourceStructure: StructureIdentity = {
      artifactId: struct.id,
      payloadDigest,
      designKey,
    };

    // Defensive per-backbone copies (RFC Gate #10)
    const lockedChains = Object.freeze([...(struct.lockedChains ?? [])]);
    const fixedResidueIndices = Object.freeze([...(struct.fixedResidueIndices ?? [])]);
    const motifIndices = Object.freeze([...(struct.motifIndices ?? [])]);
    const tiedPositions =
      typeof struct.tiedPositions === 'string' && struct.tiedPositions.trim()
        ? struct.tiedPositions
        : undefined;
    const chainRoles = struct.chainRoles ? Object.freeze({ ...struct.chainRoles }) : undefined;

    // Bind design-specific constraints into the per-backbone family anchor so changing
    // a design's locks or fixed residues creates fresh sequence keys for that design only.
    const backboneFamilyKey = sha256Hex(
      canonicalJson({
        familyKey,
        designKey,
        sourceStructure,
        lockedChains,
        fixedResidueIndices,
        motifIndices,
        ...(tiedPositions ? { tiedPositions } : {}),
      }),
    );

    const subItems: PlannedMpnnItem[] = [];
    for (let sequenceIndex = 0; sequenceIndex < params.sequencesPerBackbone; sequenceIndex++) {
      const key = makeSequenceKey(backboneFamilyKey, designKey, sequenceIndex);
      const seed = deriveStableItemSeed(baseSeed, 'mpnn', key);
      const cached = params.cachedKeys?.has(key) ?? false;
      if (cached) cachedCount++;
      const lineage: ItemLineage = {
        designKey,
        designIndex,
        sequenceKey: key,
        sequenceIndex,
        sourceStructure,
        parentKeys: [designKey],
      };
      const item: PlannedMpnnItem = {
        key,
        designKey,
        designIndex,
        sequenceIndex,
        ordinal: sequenceIndex,
        seed,
        seedScheme: 'domain-v2',
        sourceStructure,
        lineage,
        lockedChains,
        fixedResidueIndices,
        motifIndices,
        ...(tiedPositions ? { tiedPositions } : {}),
        ...(chainRoles ? { chainRoles } : {}),
        backboneLength: struct.length,
        cached,
      };
      subItems.push(item);
      allItems.push(item);
    }

    perBackbonePlans.push({
      designKey,
      designIndex,
      sourceStructure,
      lockedChains,
      fixedResidueIndices,
      motifIndices,
      ...(tiedPositions ? { tiedPositions } : {}),
      ...(chainRoles ? { chainRoles } : {}),
      backboneLength: struct.length,
      items: subItems,
    });
  }

  const totalPlannedSequences = allItems.length;
  const newCount = totalPlannedSequences - cachedCount;
  const planDigest = sha256Hex(
    canonicalJson({
      schema: 'denovo.plan-mpnn/v1',
      familyKey,
      sequencesPerBackbone: params.sequencesPerBackbone,
      keys: allItems.map((i) => i.key),
    }),
  );

  const bWord = normalizedMembers.length === 1 ? 'backbone' : 'backbones';
  const sWord = totalPlannedSequences === 1 ? 'sequence' : 'sequences';
  const summary = `${normalizedMembers.length} ${bWord} → ${totalPlannedSequences} ${sWord}`;

  return {
    familyKey,
    planDigest,
    modelId,
    backboneCount: normalizedMembers.length,
    sequencesPerBackbone: params.sequencesPerBackbone,
    totalPlannedSequences,
    perBackbonePlans,
    items: allItems,
    cachedCount,
    newCount,
    summary,
  };
}

// ============================================================================
// 3. Pure Top-K Filter / Ranker (RFC §5.1, Gate #8)
// ============================================================================

export type TopKMode = 'per-backbone' | 'global';
export type TopKMetric = 'score' | 'globalScore' | 'meanPlddt' | 'ipsae' | 'scRmsd';
export type TopKDirection = 'asc' | 'desc';

export interface TopKFilterPolicy {
  mode: TopKMode;
  topK: number;
  maxTotal?: number;
  metric: TopKMetric;
  /** Defaults: 'asc' for score/globalScore/scRmsd; 'desc' for meanPlddt/ipsae. */
  direction?: TopKDirection;
}

export interface TopKSelectedEntry<T> {
  member: CollectionMember<T>;
  key: MemberKey;
  designKey: MemberKey | null;
  groupKey: string;
  rankInGroup: number;
  globalRank: number;
  metricValue: number | null;
}

export interface TopKFilterSummary {
  mode: TopKMode;
  metric: TopKMetric;
  direction: TopKDirection;
  topK: number;
  maxTotal: number | null;
  totalInput: number;
  groupCount: number;
  selectedCount: number;
  excludedCount: number;
  description: string;
}

export interface TopKFilterResult<T> {
  policy: Required<Pick<TopKFilterPolicy, 'mode' | 'topK' | 'metric' | 'direction'>> & {
    maxTotal: number | null;
  };
  planDigest: Digest;
  selected: readonly TopKSelectedEntry<T>[];
  excluded: readonly WorkItemOutcome[];
  summary: TopKFilterSummary;
}

export function defaultDirectionForMetric(metric: TopKMetric): TopKDirection {
  switch (metric) {
    case 'meanPlddt':
    case 'ipsae':
      return 'desc';
    case 'score':
    case 'globalScore':
    case 'scRmsd':
      return 'asc';
  }
}

function extractMetricValue<T>(member: CollectionMember<T>, metric: TopKMetric): number | null {
  const payload = member.payload as Record<string, unknown> | undefined;
  const { summary } = member;

  const pickFinite = (...candidates: unknown[]): number | null => {
    for (const c of candidates) {
      if (typeof c === 'number' && Number.isFinite(c)) return c;
    }
    return null;
  };

  switch (metric) {
    case 'score':
      return pickFinite(summary.score, payload?.score);
    case 'globalScore':
      return pickFinite(summary.globalScore, payload?.globalScore);
    case 'meanPlddt':
      if (summary.confidenceAvailable === false || payload?.confidenceAvailable === false) {
        return null;
      }
      return pickFinite(summary.confidence, payload?.meanPlddt, payload?.confidence);
    case 'ipsae':
      return pickFinite(summary.ipsae, payload?.ipsae);
    case 'scRmsd':
      return pickFinite(summary.scRmsd, payload?.scRmsd);
  }
}

function compareMetricEntries<T>(
  a: { member: CollectionMember<T>; metricValue: number | null },
  b: { member: CollectionMember<T>; metricValue: number | null },
  direction: TopKDirection,
): number {
  const aVal = a.metricValue;
  const bVal = b.metricValue;
  // Null / missing metrics ALWAYS sort last regardless of direction
  if (aVal !== null && bVal === null) return -1;
  if (aVal === null && bVal !== null) return 1;
  if (aVal !== null && bVal !== null && aVal !== bVal) {
    return direction === 'asc' ? aVal - bVal : bVal - aVal;
  }
  // Deterministic tie-breaking: creation ordinal ascending, then stable member key ascending
  if (a.member.ordinal !== b.member.ordinal) return a.member.ordinal - b.member.ordinal;
  return a.member.key < b.member.key ? -1 : a.member.key > b.member.key ? 1 : 0;
}

const STANDALONE_GROUP_KEY = '__standalone__';

function groupByDesignInCreationOrder<T>(
  members: readonly CollectionMember<T>[],
): Array<{ groupKey: string; designKey: MemberKey | null; designOrder: number; members: CollectionMember<T>[] }> {
  const groups = new Map<
    string,
    { groupKey: string; designKey: MemberKey | null; designOrder: number; members: CollectionMember<T>[] }
  >();

  for (const m of members) {
    const designKey = m.lineage.designKey?.trim() || null;
    const groupKey = designKey ?? STANDALONE_GROUP_KEY;
    const existing = groups.get(groupKey);
    const designOrder =
      typeof m.lineage.designIndex === 'number' && Number.isInteger(m.lineage.designIndex)
        ? m.lineage.designIndex
        : Number.MAX_SAFE_INTEGER;
    if (existing) {
      existing.members.push(m);
      if (designOrder < existing.designOrder) existing.designOrder = designOrder;
    } else {
      groups.set(groupKey, {
        groupKey,
        designKey,
        designOrder,
        members: [m],
      });
    }
  }

  return [...groups.values()].sort((a, b) => {
    if (a.designOrder !== b.designOrder) return a.designOrder - b.designOrder;
    return a.groupKey < b.groupKey ? -1 : a.groupKey > b.groupKey ? 1 : 0;
  });
}

/**
 * Pure Top-K Filter / Ranker supporting 'per-backbone' (with round-robin maxTotal fairness)
 * and 'global' modes. Null metrics sort last; ties break deterministically by creation
 * ordinal then stable sequence/member key.
 */
export function planTopKFilter<T = SequenceCandidate>(
  input: DesignCollection<T> | readonly CollectionMember<T>[],
  policy: TopKFilterPolicy,
): TopKFilterResult<T> {
  if (!Number.isInteger(policy.topK) || policy.topK < 1) {
    throw new RangeError(`topK must be a positive integer; got ${String(policy.topK)}`);
  }
  if (policy.maxTotal !== undefined && (!Number.isInteger(policy.maxTotal) || policy.maxTotal < 1)) {
    throw new RangeError(`maxTotal must be a positive integer when provided; got ${String(policy.maxTotal)}`);
  }

  const members = isDesignCollection(input) ? input.members : input;
  const direction = policy.direction ?? defaultDirectionForMetric(policy.metric);
  const maxTotal = policy.maxTotal ?? null;

  const resolvedPolicy = {
    mode: policy.mode,
    topK: policy.topK,
    metric: policy.metric,
    direction,
    maxTotal,
  };

  const groups = groupByDesignInCreationOrder(members);
  const excluded: WorkItemOutcome[] = [];
  const selected: TopKSelectedEntry<T>[] = [];

  if (policy.mode === 'global') {
    const scored = members
      .map((member) => ({
        member,
        metricValue: extractMetricValue(member, policy.metric),
      }))
      .sort((a, b) => compareMetricEntries(a, b, direction));

    const limit = maxTotal !== null ? Math.min(policy.topK, maxTotal) : policy.topK;
    const rankWithinGroupCounter = new Map<string, number>();

    scored.forEach((entry, idx) => {
      const designKey = entry.member.lineage.designKey?.trim() || null;
      const groupKey = designKey ?? STANDALONE_GROUP_KEY;
      const groupRank = rankWithinGroupCounter.get(groupKey) ?? 0;
      rankWithinGroupCounter.set(groupKey, groupRank + 1);

      if (idx < limit) {
        selected.push({
          member: entry.member,
          key: entry.member.key,
          designKey,
          groupKey,
          rankInGroup: groupRank,
          globalRank: idx,
          metricValue: entry.metricValue,
        });
      } else {
        const capReason =
          maxTotal !== null && maxTotal < policy.topK
            ? `Excluded by global maxTotal=${maxTotal} (rank #${idx + 1} of ${scored.length} by ${policy.metric} ${direction})`
            : `Excluded by global topK=${policy.topK} (rank #${idx + 1} of ${scored.length} by ${policy.metric} ${direction})`;
        excluded.push({
          key: entry.member.key,
          status: 'excluded',
          attempt: 0,
          reason: capReason,
          lineage: entry.member.lineage,
        });
      }
    });
  } else {
    // 'per-backbone' mode: rank within each design, keep topK per design, then apply round-robin maxTotal
    const eligiblePerGroup: Array<
      Array<{
        member: CollectionMember<T>;
        designKey: MemberKey | null;
        groupKey: string;
        rankInGroup: number;
        metricValue: number | null;
      }>
    > = [];

    for (const group of groups) {
      const scored = group.members
        .map((member) => ({
          member,
          metricValue: extractMetricValue(member, policy.metric),
        }))
        .sort((a, b) => compareMetricEntries(a, b, direction));

      const kept: Array<{
        member: CollectionMember<T>;
        designKey: MemberKey | null;
        groupKey: string;
        rankInGroup: number;
        metricValue: number | null;
      }> = [];

      scored.forEach((entry, rankIdx) => {
        if (rankIdx < policy.topK) {
          kept.push({
            member: entry.member,
            designKey: group.designKey,
            groupKey: group.groupKey,
            rankInGroup: rankIdx,
            metricValue: entry.metricValue,
          });
        } else {
          excluded.push({
            key: entry.member.key,
            status: 'excluded',
            attempt: 0,
            reason: `Excluded by per-backbone topK=${policy.topK} (rank #${rankIdx + 1} of ${scored.length} for ${group.groupKey} by ${policy.metric} ${direction})`,
            lineage: entry.member.lineage,
          });
        }
      });

      eligiblePerGroup.push(kept);
    }

    // Round-robin across backbones by rankInGroup so maxTotal is allocated fairly across designs
    const maxRank = eligiblePerGroup.reduce((acc, g) => Math.max(acc, g.length), 0);
    const totalEligible = eligiblePerGroup.reduce((acc, g) => acc + g.length, 0);
    const cap = maxTotal ?? totalEligible;

    for (let rank = 0; rank < maxRank; rank++) {
      for (const groupList of eligiblePerGroup) {
        if (rank >= groupList.length) continue;
        const item = groupList[rank];
        if (selected.length < cap) {
          selected.push({
            member: item.member,
            key: item.member.key,
            designKey: item.designKey,
            groupKey: item.groupKey,
            rankInGroup: item.rankInGroup,
            globalRank: selected.length,
            metricValue: item.metricValue,
          });
        } else {
          excluded.push({
            key: item.member.key,
            status: 'excluded',
            attempt: 0,
            reason: `Excluded by maxTotal=${cap} after round-robin allocation across ${groups.length} backbones`,
            lineage: item.member.lineage,
          });
        }
      }
    }
  }

  const planDigest = sha256Hex(
    canonicalJson({
      schema: 'denovo.plan-topk/v1',
      policy: resolvedPolicy,
      selectedKeys: selected.map((s) => s.key),
      excludedKeys: excluded.map((e) => e.key),
    }),
  );

  const summary: TopKFilterSummary = {
    mode: policy.mode,
    metric: policy.metric,
    direction,
    topK: policy.topK,
    maxTotal,
    totalInput: members.length,
    groupCount: groups.length,
    selectedCount: selected.length,
    excludedCount: excluded.length,
    description: `${members.length} candidates across ${groups.length} ${groups.length === 1 ? 'group' : 'groups'} → ${selected.length} selected (${excluded.length} excluded)`,
  };

  return {
    policy: resolvedPolicy,
    planDigest,
    selected,
    excluded,
    summary,
  };
}

// ============================================================================
// 4. Authoritative LocalFold Fan-out Plan (RFC §5.1, Gates #7 & #8)
// ============================================================================

export interface LocalFoldFanoutParams {
  foldModelId: string;
  baseSeed: number;
  /** Maximum sequences selected per backbone (default 2). */
  perDesignTopK?: number;
  /** Number of fold samples per selected sequence (default 1, max 16). */
  samplesPerSequence?: number;
  /** Total desired fold cap across the campaign (default 32, hard cap 256). */
  maxFolds?: number;
  /** Ranking within each backbone (default 'mpnn-score'). */
  ranking?: 'creation-order' | 'mpnn-score';
  /** Optional explicit metric override when ranking === 'mpnn-score' (default 'score'). */
  metric?: TopKMetric;
  direction?: TopKDirection;
  scientificParams?: Record<string, unknown>;
  runtimeDigest?: Digest;
  weightsDigest?: Digest;
  /** Already-cached foldKeys; resolved AFTER scientific selection so cache never changes selection. */
  cachedFoldKeys?: ReadonlySet<MemberKey>;
}

export interface PlannedFoldItem {
  key: MemberKey;
  foldKey: MemberKey;
  sequenceKey: MemberKey;
  designKey?: MemberKey;
  designIndex?: number;
  sampleIndex: number;
  rankInDesign: number;
  foldModelId: string;
  seed: number;
  seedScheme: 'domain-v2';
  lineage: ItemLineage;
  sequenceMember: CollectionMember<SequenceCandidate>;
  cached: boolean;
}

export interface LocalFoldFanoutPlan {
  familyKey: Digest;
  planDigest: Digest;
  foldModelId: string;
  policy: {
    perDesignTopK: number;
    samplesPerSequence: number;
    maxFolds: number;
    ranking: 'creation-order' | 'mpnn-score';
    allocation: 'round-robin-by-design';
  };
  backboneCount: number;
  totalSequenceCount: number;
  selectedSequenceCount: number;
  eligibleFoldCount: number;
  desiredFoldCount: number;
  cachedFoldCount: number;
  newFoldCount: number;
  selected: readonly PlannedFoldItem[];
  excluded: readonly WorkItemOutcome[];
  /** Human-readable path summary, e.g. "4 backbones → 8 sequences → 4 selected → 4 refolds". */
  pathSummary: string;
  /** Cache delta summary, e.g. "2 refolds cached · 2 to run". */
  cacheSummary: string;
}

/**
 * Authoritative LocalFold fan-out planner shared by UI, preflight, workers, and export.
 *
 * Implements RFC §5.1 verbatim:
 * 1. Group sequences by designKey (standalone inputs use standalone group).
 * 2. Sort within each group by finite MPNN score ascending (or creation-order), tie-break by ordinal then sequenceKey.
 * 3. Keep at most perDesignTopK sequences per group.
 * 4. Expand each selected sequence into samplesPerSequence fold identities (model-specific foldKeys & stable seeds).
 * 5. Allocate round-robin across designs, then sequence rank, then sample ordinal.
 * 6. Apply maxFolds cap (hard cap 256).
 * 7. Resolve cache hits AFTER scientific selection.
 * 8. Record an explicit outcome reason for every excluded sequence/fold.
 */
export function planLocalFoldFanout(
  sequences: SequenceCollection | readonly CollectionMember<SequenceCandidate>[],
  params: LocalFoldFanoutParams,
): LocalFoldFanoutPlan {
  const foldModelId = params.foldModelId.trim();
  if (!foldModelId) {
    throw new RangeError('foldModelId must be a non-empty string');
  }
  const baseSeed = normalizeBaseSeed(params.baseSeed);
  const perDesignTopK = params.perDesignTopK ?? DEFAULT_PER_DESIGN_TOP_K;
  const samplesPerSequence = params.samplesPerSequence ?? 1;
  const maxFolds = params.maxFolds ?? DEFAULT_MAX_FOLDS;
  const ranking = params.ranking ?? 'mpnn-score';

  if (!Number.isInteger(perDesignTopK) || perDesignTopK < 1) {
    throw new RangeError(`perDesignTopK must be a positive integer; got ${String(perDesignTopK)}`);
  }
  if (
    !Number.isInteger(samplesPerSequence) ||
    samplesPerSequence < 1 ||
    samplesPerSequence > MAX_SAMPLES_PER_SEQUENCE
  ) {
    throw new RangeError(
      `samplesPerSequence must be an integer in 1..${MAX_SAMPLES_PER_SEQUENCE}; got ${String(samplesPerSequence)}`,
    );
  }
  if (!Number.isInteger(maxFolds) || maxFolds < 1 || maxFolds > HARD_MAX_FOLDS) {
    throw new RangeError(`maxFolds must be an integer in 1..${HARD_MAX_FOLDS}; got ${String(maxFolds)}`);
  }

  const members = isDesignCollection(sequences) ? sequences.members : sequences;
  const scientific = normalizeScientificParams(params.scientificParams);

  const familyKey = sha256Hex(
    canonicalJson({
      schema: 'denovo.family-fold/v1',
      foldModelId,
      seedScheme: 'domain-v2',
      baseSeed,
      scientificParams: scientific,
      runtimeDigest: params.runtimeDigest ?? '',
      weightsDigest: params.weightsDigest ?? '',
    }),
  );

  const groups = groupByDesignInCreationOrder(members);
  const metric = params.metric ?? 'score';
  const direction = params.direction ?? defaultDirectionForMetric(metric);

  const excluded: WorkItemOutcome[] = [];
  const selectedSeqsPerGroup: Array<
    Array<{
      member: CollectionMember<SequenceCandidate>;
      rankInDesign: number;
      designKey?: MemberKey;
      designIndex?: number;
    }>
  > = [];

  for (const group of groups) {
    const ordered =
      ranking === 'creation-order'
        ? [...group.members].sort((a, b) =>
            a.ordinal !== b.ordinal ? a.ordinal - b.ordinal : a.key < b.key ? -1 : a.key > b.key ? 1 : 0,
          )
        : group.members
            .map((member) => ({
              member,
              metricValue: extractMetricValue(member, metric),
            }))
            .sort((a, b) => compareMetricEntries(a, b, direction))
            .map((e) => e.member);

    const kept: Array<{
      member: CollectionMember<SequenceCandidate>;
      rankInDesign: number;
      designKey?: MemberKey;
      designIndex?: number;
    }> = [];

    ordered.forEach((seqMember, rankIdx) => {
      if (rankIdx < perDesignTopK) {
        kept.push({
          member: seqMember,
          rankInDesign: rankIdx,
          designKey: seqMember.lineage.designKey,
          designIndex: seqMember.lineage.designIndex,
        });
      } else {
        const seqKey = seqMember.lineage.sequenceKey ?? seqMember.key;
        excluded.push({
          key: seqKey,
          status: 'excluded',
          attempt: 0,
          reason: `Excluded by perDesignTopK=${perDesignTopK} (rank #${rankIdx + 1} of ${ordered.length} in ${group.groupKey})`,
          lineage: seqMember.lineage,
        });
      }
    });

    selectedSeqsPerGroup.push(kept);
  }

  // Round-robin across designs -> sequence rank -> sample ordinal (RFC §5.1 step 5)
  const maxRankInAnyGroup = selectedSeqsPerGroup.reduce((m, g) => Math.max(m, g.length), 0);
  const selected: PlannedFoldItem[] = [];
  const selectedSequenceKeys = new Set<MemberKey>();
  let eligibleFoldCount = 0;

  for (let sampleIndex = 0; sampleIndex < samplesPerSequence; sampleIndex++) {
    for (let rank = 0; rank < maxRankInAnyGroup; rank++) {
      for (const groupList of selectedSeqsPerGroup) {
        if (rank >= groupList.length) continue;
        const entry = groupList[rank];
        const seqMember = entry.member;
        const sequenceKey = seqMember.lineage.sequenceKey ?? seqMember.key;
        const foldKey = makeFoldKey(familyKey, sequenceKey, sampleIndex, foldModelId);
        const seed = deriveStableItemSeed(baseSeed, 'fold', foldKey);
        const lineage: ItemLineage = {
          ...(seqMember.lineage.designKey ? { designKey: seqMember.lineage.designKey } : {}),
          ...(seqMember.lineage.designIndex !== undefined
            ? { designIndex: seqMember.lineage.designIndex }
            : {}),
          sequenceKey,
          ...(seqMember.lineage.sequenceIndex !== undefined
            ? { sequenceIndex: seqMember.lineage.sequenceIndex }
            : {}),
          foldKey,
          foldIndex: sampleIndex,
          ...(seqMember.lineage.sourceStructure
            ? { sourceStructure: seqMember.lineage.sourceStructure }
            : {}),
          parentKeys: [sequenceKey],
        };

        eligibleFoldCount++;
        if (selected.length < maxFolds) {
          const cached = params.cachedFoldKeys?.has(foldKey) ?? false;
          selectedSequenceKeys.add(sequenceKey);
          selected.push({
            key: foldKey,
            foldKey,
            sequenceKey,
            designKey: seqMember.lineage.designKey,
            designIndex: seqMember.lineage.designIndex,
            sampleIndex,
            rankInDesign: entry.rankInDesign,
            foldModelId,
            seed,
            seedScheme: 'domain-v2',
            lineage,
            sequenceMember: seqMember,
            cached,
          });
        } else {
          excluded.push({
            key: foldKey,
            status: 'excluded',
            attempt: 0,
            reason: `Excluded by maxFolds=${maxFolds} cap after round-robin allocation across ${groups.length} ${groups.length === 1 ? 'design' : 'designs'}`,
            lineage,
          });
        }
      }
    }
  }

  const cachedFoldCount = selected.reduce((acc, item) => acc + (item.cached ? 1 : 0), 0);
  const newFoldCount = selected.length - cachedFoldCount;
  const policy = {
    perDesignTopK,
    samplesPerSequence,
    maxFolds,
    ranking,
    allocation: 'round-robin-by-design' as const,
  };

  const planDigest = sha256Hex(
    canonicalJson({
      schema: 'denovo.plan-fold/v1',
      familyKey,
      policy,
      selectedKeys: selected.map((s) => s.foldKey),
      excludedKeys: excluded.map((e) => e.key),
    }),
  );

  const backboneCount = groups.length;
  const totalSequenceCount = members.length;
  const selectedSequenceCount = selectedSequenceKeys.size;

  const bWord = backboneCount === 1 ? 'backbone' : 'backbones';
  const sWord = totalSequenceCount === 1 ? 'sequence' : 'sequences';
  const fWord = selected.length === 1 ? 'refold' : 'refolds';
  const pathSummary = `${backboneCount} ${bWord} → ${totalSequenceCount} ${sWord} → ${selectedSequenceCount} selected → ${selected.length} ${fWord}`;
  const cacheSummary = `${cachedFoldCount} ${cachedFoldCount === 1 ? 'refold' : 'refolds'} cached · ${newFoldCount} to run`;

  return {
    familyKey,
    planDigest,
    foldModelId,
    policy,
    backboneCount,
    totalSequenceCount,
    selectedSequenceCount,
    eligibleFoldCount,
    desiredFoldCount: selected.length,
    cachedFoldCount,
    newFoldCount,
    selected,
    excluded,
    pathSummary,
    cacheSummary,
  };
}

/** Pre-generation cardinality upper-bound helper (RFC §5.1). */
export function estimateFanoutBounds(input: {
  numDesigns: number;
  sequencesPerBackbone: number;
  perDesignTopK: number;
  samplesPerSequence?: number;
  maxFolds?: number;
}): {
  maxSequences: number;
  maxSelectedSequences: number;
  maxFolds: number;
  summary: string;
} {
  const N = Math.max(1, Math.min(MAX_DESIGNS, Math.round(input.numDesigns)));
  const K = Math.max(1, Math.min(MAX_SEQUENCES_PER_DESIGN, Math.round(input.sequencesPerBackbone)));
  const topK = Math.max(1, Math.round(input.perDesignTopK));
  const S = Math.max(1, Math.min(MAX_SAMPLES_PER_SEQUENCE, Math.round(input.samplesPerSequence ?? 1)));
  const cap = Math.max(1, Math.min(HARD_MAX_FOLDS, Math.round(input.maxFolds ?? DEFAULT_MAX_FOLDS)));

  const maxSequences = N * K;
  const maxSelectedSequences = N * Math.min(K, topK);
  const maxFolds = Math.min(cap, maxSelectedSequences * S);
  const bWord = N === 1 ? 'backbone' : 'backbones';
  const sWord = maxSequences === 1 ? 'sequence' : 'sequences';
  const fWord = maxFolds === 1 ? 'refold' : 'refolds';

  return {
    maxSequences,
    maxSelectedSequences,
    maxFolds,
    summary: `${N} ${bWord} → ${maxSequences} ${sWord} → ${maxSelectedSequences} selected → ${maxFolds} ${fWord}`,
  };
}
