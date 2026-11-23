// Wave-6A Multi-Design & Batching Core: Canonical Collection Envelope, Lineage,
// Exact Parent Resolution, Representative Selection, and Campaign Aggregation.
//
// Pure TypeScript module (zero DOM/GPU dependencies; runs identically in main thread,
// Dedicated Web Workers, and Node/Vitest tests).

import type {
  CanonicalArtifact,
  CollectionKind,
  DesignabilityMetrics,
  DesignabilityRow,
  FoldedCandidate,
  FoldedStructure,
  RepresentativeSelection,
  ResidueTrackArtifact,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from './types';

/** Lowercase hexadecimal SHA-256 digest. */
export type Digest = string;

/** Immutable deterministic member identity (never derived from score rank or array index). */
export type MemberKey = string;

export type ItemStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'excluded'
  | 'failed'
  | 'cancelled';

export interface BlobRef {
  digest: Digest;
  byteLength: number;
  mediaType: string;
}

export interface StructureIdentity {
  artifactId: string;
  payloadDigest: Digest;
  designKey?: MemberKey;
}

export interface ItemLineage {
  /** Absent for a sequence-only source with no designed backbone. */
  designKey?: MemberKey;
  designIndex?: number;

  sequenceKey?: MemberKey;
  sequenceIndex?: number;

  foldKey?: MemberKey;
  foldIndex?: number;

  ensembleKey?: MemberKey;
  ensembleSampleIndex?: number;
  ensembleFrameKey?: MemberKey;
  frameIndex?: number;

  /** Exact backbone used to design this sequence / fold. */
  sourceStructure?: StructureIdentity;

  /** General derivation edges, including redesign from an earlier fold. */
  parentKeys: readonly MemberKey[];
}

export interface SamplingProvenance {
  seedScheme: 'legacy-v1' | 'domain-v2';
  seed: number;
  runtimeDigest?: Digest;
  weightsDigest?: Digest;
  scientificParamsDigest?: Digest;
  resolvedInputDigest?: Digest;
}

export interface WorkItemOutcome {
  key: MemberKey;
  status: ItemStatus;
  attempt: number;
  reason?: string;
  lineage?: ItemLineage;
}

export type CollectionMemberKind =
  | 'StructureArtifact'
  | 'SequenceCandidate'
  | 'FoldedCandidate'
  | 'DesignabilityRow'
  | 'ResidueTrack';

export interface CollectionMemberSummary {
  label: string;
  residueCount?: number;
  atomCount?: number;
  sequenceDigest?: Digest;
  score?: number | null;
  globalScore?: number | null;
  confidence?: number | null;
  confidenceAvailable?: boolean;
  designable?: boolean | null;
  scRmsd?: number | null;
  ipsae?: number | null;
}

export interface CollectionMember<T = unknown> {
  key: MemberKey;
  /** Creation ordinal (0-based); never score rank. */
  ordinal: number;
  memberKind: CollectionMemberKind;
  lineage: ItemLineage;
  provenance?: SamplingProvenance;
  /** Durable payload reference for hydrated/dehydrated storage. */
  payloadRef?: BlobRef;
  /** Optional hydrated payload (may be omitted when stored dehydrated in ArtifactBlobStore). */
  payload?: T;
  molecularContextRef?: BlobRef;
  summary: CollectionMemberSummary;
}

export interface CollectionRepresentativeSelection extends RepresentativeSelection {
  collectionId: string;
  criterionVersion: number;
}

export interface DesignCollection<T = unknown> {
  kind: 'Collection';
  schema: 'denovo.collection/v1';
  /** Immutable manifest snapshot digest. */
  id: string;
  /** Stable across incremental extension (e.g., 4 -> 8 designs). */
  familyKey: string;
  revision: number;
  memberKind: CollectionMemberKind;
  collectionKind: CollectionKind;
  label: string;
  sourceModelId: string;

  /** Committed members only. */
  members: readonly CollectionMember<T>[];
  outcomes: readonly WorkItemOutcome[];
  requestedCount: number;
  complete: boolean;

  representativeSelection?: CollectionRepresentativeSelection;
  legacySourceKind?: CanonicalArtifact['kind'];
  legacySourceId?: string;
  aggregationPolicy?: string;
}

export type StructureCollection = DesignCollection<StructureArtifact>;
export type SequenceCollection = DesignCollection<SequenceCandidate>;
export type FoldCollection = DesignCollection<FoldedCandidate>;
export type DesignabilityCollection = DesignCollection<DesignabilityRow>;
export type ResidueTrackCollection = DesignCollection<ResidueTrackArtifact>;

export interface DesignPassSummary {
  designKey: MemberKey;
  plannedFolds: number;
  completedFolds: number;
  evaluableFolds: number;
  passedFolds: number;
  failedFolds: number;
  excludedFolds: number;
  pendingFolds: number;
  ineligibleFolds: number;
  /** passedFolds / evaluableFolds; null when evaluableFolds === 0. */
  foldPassRate: number | null;
  hasPassingSequence: boolean;
  complete: boolean;
}

export interface CampaignCoverageSummary {
  plannedFolds: number;
  completedFolds: number;
  evaluableFolds: number;
  /** completedFolds / plannedFolds; null when plannedFolds === 0. */
  completedFraction: number | null;
  /** evaluableFolds / plannedFolds; null when plannedFolds === 0. */
  evaluableFraction: number | null;
}

export interface CampaignSummary {
  aggregationPolicy: string;
  samplesPerSequence: number;
  predictorModelId?: string;
  requestedDesigns: number;
  evaluatedDesigns: number;
  passingDesigns: number;
  plannedFolds: number;
  completedFolds: number;
  evaluableFolds: number;
  passedFolds: number;
  failedFolds: number;
  excludedFolds: number;
  pendingFolds: number;
  ineligibleFolds: number;
  /** passing folds / evaluable folds; null when evaluableFolds === 0. */
  foldPassRate: number | null;
  /** designs with at least one passing sequence / evaluated designs; null when evaluatedDesigns === 0. */
  designHitRate: number | null;
  /** Equal-weight mean of per-design foldPassRate across evaluable designs; null when none. */
  meanPerDesignFoldPassRate: number | null;
  coverage: CampaignCoverageSummary;
  /** passingDesigns / requestedDesigns (explicitly includes unfinished/failed designs); null when requestedDesigns === 0. */
  conservativeCampaignYield: number | null;
  perDesign: readonly DesignPassSummary[];
  complete: boolean;
}

export interface CollectionInvariantViolation {
  code:
    | 'duplicate-member-key'
    | 'invalid-member-key'
    | 'invalid-ordinal'
    | 'member-kind-mismatch'
    | 'missing-design-key'
    | 'duplicate-design-key'
    | 'missing-sequence-key'
    | 'inconsistent-sequence-parent'
    | 'missing-fold-key'
    | 'fold-reused-sequence-key'
    | 'missing-parent-edge'
    | 'source-structure-conflict'
    | 'missing-parent-member'
    | 'cross-parent-aliasing';
  memberKey?: MemberKey;
  message: string;
}

export interface CollectionValidationContext {
  /** Known upstream structures keyed by designKey or artifactId for cross-parent verification. */
  knownStructures?:
    | StructureCollection
    | readonly CollectionMember<StructureArtifact>[]
    | readonly StructureArtifact[];
  /** Optional parent collection to verify every member's parent key exists. */
  parentCollection?: DesignCollection<unknown>;
  /** Verify payload digests against knownStructures when available (default true). */
  verifyPayloadDigests?: boolean;
}

export interface CollectionValidationResult {
  valid: boolean;
  errors: readonly CollectionInvariantViolation[];
}

export type ParentResolutionErrorCode =
  | 'unlinked-parent'
  | 'parent-not-found'
  | 'sequence-not-found'
  | 'lineage-conflict'
  | 'cross-parent-conflict'
  | 'digest-mismatch'
  | 'length-mismatch';

export type ParentStructureResolution =
  | {
      ok: true;
      structure: StructureArtifact;
      member?: CollectionMember<StructureArtifact>;
      designKey?: MemberKey;
      structureIdentity: StructureIdentity;
    }
  | {
      ok: false;
      code: ParentResolutionErrorCode;
      reason: string;
    };

export interface ResolveParentOptions {
  /** Optional sequence collection used to cross-check fold -> sequence -> design lineage. */
  sequenceCollection?: SequenceCollection | readonly CollectionMember<SequenceCandidate>[];
  /** Verify sourceStructure.payloadDigest against candidate structure payload (default true). */
  verifyPayloadDigest?: boolean;
  /** Optional expected residue length (never truncates on mismatch). */
  expectedLength?: number;
}

export interface FoldEvaluationRecord {
  foldKey: MemberKey;
  designKey: MemberKey;
  sequenceKey: MemberKey;
  status: ItemStatus;
  /** True if scored and passed gates; false if scored and failed; null/undefined if unscored or ineligible. */
  designable?: boolean | null;
  /** False when the predictor has no confidence head (ineligible under confidence-dependent criteria). */
  confidenceAvailable?: boolean;
  /** Optional explicit evaluability override (e.g., invalid reference or ineligible sample). */
  evaluable?: boolean;
  reason?: string;
}

export interface DesignPassSummaryOptions {
  /** Explicit ordered list of designKeys to include (even if 0 folds have been planned/run). */
  designKeys?: readonly MemberKey[];
  /** Require confidence availability for a completed fold to be evaluable (default true). */
  requireConfidence?: boolean;
  /** Rule determining whether a sequence passes (default 'any-sample-passes'). */
  sequenceAggregationRule?: 'any-sample-passes' | 'all-samples-pass';
}

export interface CampaignSummaryOptions extends DesignPassSummaryOptions {
  /** Total requested designs for conservative campaign yield (defaults to discovered design count). */
  requestedDesigns?: number;
  /** Declared number of fold samples per sequence (default 1). */
  samplesPerSequence?: number;
  predictorModelId?: string;
}

// ============================================================================
// Pure Synchronous FIPS 180-4 SHA-256 & Canonical Serialization
// ============================================================================

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const textEncoder = new TextEncoder();

/**
 * Pure synchronous SHA-256 returning a 64-character lowercase hex digest.
 * Works identically in browsers (including non-secure contexts & workers) and Node.
 */
export function sha256Hex(input: string | Uint8Array): Digest {
  const bytes = typeof input === 'string' ? textEncoder.encode(input) : input;
  const bitLenHi = Math.floor((bytes.byteLength * 8) / 0x100000000);
  const bitLenLo = (bytes.byteLength * 8) >>> 0;

  // Pad to multiple of 64 bytes: message + 0x80 + zeros + 8-byte big-endian bit length.
  const paddedLen = (((bytes.byteLength + 9) + 63) >>> 6) << 6;
  const padded = new Uint8Array(paddedLen);
  padded.set(bytes);
  padded[bytes.byteLength] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLen - 8, bitLenHi, false);
  view.setUint32(paddedLen - 4, bitLenLo, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);

  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i++) {
      w[i] = view.getUint32(offset + i * 4, false);
    }
    for (let i = 16; i < 64; i++) {
      const w15 = w[i - 15];
      const w2 = w[i - 2];
      const s0 = ((w15 >>> 7) | (w15 << 25)) ^ ((w15 >>> 18) | (w15 << 14)) ^ (w15 >>> 3);
      const s1 = ((w2 >>> 17) | (w2 << 15)) ^ ((w2 >>> 19) | (w2 << 13)) ^ (w2 >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return (
    h0.toString(16).padStart(8, '0') +
    h1.toString(16).padStart(8, '0') +
    h2.toString(16).padStart(8, '0') +
    h3.toString(16).padStart(8, '0') +
    h4.toString(16).padStart(8, '0') +
    h5.toString(16).padStart(8, '0') +
    h6.toString(16).padStart(8, '0') +
    h7.toString(16).padStart(8, '0')
  );
}

/** Deterministic canonical JSON stringification with sorted object keys. */
export function canonicalJson(value: unknown): string {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return 'null';
    return Object.is(value, -0) ? '0' : String(value);
  }
  if (typeof value === 'boolean' || typeof value === 'string') {
    return JSON.stringify(value);
  }
  if (ArrayBuffer.isView(value)) {
    const view = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    return JSON.stringify({
      __typed: value.constructor.name,
      len: value.byteLength,
      sha256: sha256Hex(view),
    });
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(',')}]`;
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(',')}}`;
  }
  return 'null';
}

/** Compute a deterministic content digest for a StructureArtifact's identity. */
export function computeStructurePayloadDigest(
  structure: Pick<StructureArtifact, 'id' | 'sequence' | 'length' | 'pdbText' | 'caCoords'>,
): Digest {
  const caBytes =
    structure.caCoords instanceof Float32Array
      ? new Uint8Array(
          structure.caCoords.buffer,
          structure.caCoords.byteOffset,
          structure.caCoords.byteLength,
        )
      : new Uint8Array(0);
  const caDigest = sha256Hex(caBytes);
  return sha256Hex(
    canonicalJson({
      schema: 'denovo.structure-digest/v1',
      id: structure.id,
      length: structure.length,
      sequence: structure.sequence,
      caDigest,
      pdbDigest: sha256Hex(structure.pdbText ?? ''),
    }),
  );
}

/** Compute an immutable manifest snapshot ID from familyKey, members, and outcomes. */
export function computeCollectionSnapshotId<T>(
  familyKey: string,
  members: readonly CollectionMember<T>[],
  outcomes: readonly WorkItemOutcome[],
  selectionPolicyDigest = '',
): Digest {
  return sha256Hex(
    canonicalJson({
      schema: 'denovo.collection-snapshot/v1',
      familyKey,
      selectionPolicyDigest,
      members: members.map((m) => ({
        key: m.key,
        ordinal: m.ordinal,
        memberKind: m.memberKind,
        payloadDigest: m.payloadRef?.digest ?? m.summary.sequenceDigest ?? null,
        lineage: m.lineage,
      })),
      outcomes: outcomes.map((o) => ({
        key: o.key,
        status: o.status,
        attempt: o.attempt,
        reason: o.reason ?? null,
      })),
    }),
  );
}

// ============================================================================
// Deterministic Member Key Derivation (Independent of Sorting & Top-K)
// ============================================================================

function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0 || !Number.isSafeInteger(value)) {
    throw new RangeError(`${name} must be a non-negative integer; got ${String(value)}`);
  }
}

function assertNonEmptyString(value: string, name: string): void {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new RangeError(`${name} must be a non-empty string`);
  }
}

/**
 * Derive a deterministic, immutable backbone design key.
 * Independent of numDesigns, score sorting, Top-K changes, or run timestamps.
 */
export function makeDesignKey(familyKey: string, designIndex: number): MemberKey {
  assertNonEmptyString(familyKey, 'familyKey');
  assertNonNegativeInteger(designIndex, 'designIndex');
  const hash = sha256Hex(`design|v1|${familyKey}|${designIndex}`).slice(0, 16);
  return `design:${hash}:d${designIndex}`;
}

/**
 * Derive a deterministic, immutable sequence candidate key.
 * Includes parent designKey (or explicit standalone marker) and sampling ordinal;
 * completely independent of MPNN score sorting, Top-K selection, or numSequences.
 */
export function makeSequenceKey(
  familyKey: string,
  designKey: MemberKey | null | undefined,
  sequenceIndex: number,
): MemberKey {
  assertNonEmptyString(familyKey, 'familyKey');
  assertNonNegativeInteger(sequenceIndex, 'sequenceIndex');
  const anchor = designKey && designKey.trim().length > 0 ? designKey : 'standalone';
  const hash = sha256Hex(`seq|v1|${familyKey}|${anchor}|${sequenceIndex}`).slice(0, 16);
  return `seq:${hash}:q${sequenceIndex}`;
}

/**
 * Derive a deterministic, model-specific fold key.
 * Never reuses sequenceKey as foldKey; independent of ranking, Top-K, or cache hits.
 */
export function makeFoldKey(
  familyKey: string,
  sequenceKey: MemberKey,
  sampleIndex: number,
  foldModelId: string,
): MemberKey {
  assertNonEmptyString(familyKey, 'familyKey');
  assertNonEmptyString(sequenceKey, 'sequenceKey');
  assertNonEmptyString(foldModelId, 'foldModelId');
  assertNonNegativeInteger(sampleIndex, 'sampleIndex');
  const hash = sha256Hex(
    `fold|v1|${familyKey}|${sequenceKey}|${foldModelId.trim()}|${sampleIndex}`,
  ).slice(0, 16);
  return `fold:${hash}:s${sampleIndex}`;
}

// ============================================================================
// Legacy Artifact Normalization
// ============================================================================

export function isDesignCollection(value: unknown): value is DesignCollection<unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as { kind?: unknown }).kind === 'Collection' &&
    (value as { schema?: unknown }).schema === 'denovo.collection/v1'
  );
}

export function normalizeToCollection(
  artifact: StructureArtifact | StructureCollection,
): StructureCollection;
export function normalizeToCollection(
  artifact: SequenceSet | SequenceCollection,
): SequenceCollection;
export function normalizeToCollection(
  artifact: FoldedStructure | FoldCollection,
): FoldCollection;
export function normalizeToCollection(
  artifact: DesignabilityMetrics | DesignabilityCollection,
): DesignabilityCollection;
export function normalizeToCollection<T>(artifact: DesignCollection<T>): DesignCollection<T>;
export function normalizeToCollection(
  artifact:
    | StructureArtifact
    | SequenceSet
    | FoldedStructure
    | DesignabilityMetrics
    | DesignCollection<unknown>,
): DesignCollection<unknown> {
  if (isDesignCollection(artifact)) {
    return artifact;
  }

  switch (artifact.kind) {
    case 'StructureArtifact': {
      const payloadDigest = computeStructurePayloadDigest(artifact);
      const familyKey = sha256Hex(
        canonicalJson({
          schema: 'denovo.legacy-family/v1',
          kind: 'StructureArtifact',
          id: artifact.id,
          sourceModelId: artifact.sourceModelId,
        }),
      );
      const designKey = makeDesignKey(familyKey, 0);
      const sourceStructure: StructureIdentity = {
        artifactId: artifact.id,
        payloadDigest,
        designKey,
      };
      const member: CollectionMember<StructureArtifact> = {
        key: designKey,
        ordinal: 0,
        memberKind: 'StructureArtifact',
        lineage: {
          designKey,
          designIndex: 0,
          sourceStructure,
          parentKeys: artifact.parentStructureId ? [artifact.parentStructureId] : [],
        },
        payload: artifact,
        summary: {
          label: artifact.label || artifact.id,
          residueCount: artifact.length,
          sequenceDigest: sha256Hex(artifact.sequence),
        },
      };
      const outcomes: WorkItemOutcome[] = [
        { key: designKey, status: 'completed', attempt: 1, lineage: member.lineage },
      ];
      const id = computeCollectionSnapshotId(familyKey, [member], outcomes);
      const collection: StructureCollection = {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id,
        familyKey,
        revision: 1,
        memberKind: 'StructureArtifact',
        collectionKind: 'independent-candidates',
        label: artifact.label || artifact.id,
        sourceModelId: artifact.sourceModelId,
        members: [member],
        outcomes,
        requestedCount: 1,
        complete: true,
        legacySourceKind: 'StructureArtifact',
        legacySourceId: artifact.id,
      };
      const rep = selectRepresentativeMember(collection);
      if (rep) {
        return { ...collection, representativeSelection: rep.selection };
      }
      return collection;
    }

    case 'SequenceSet': {
      const familyKey = sha256Hex(
        canonicalJson({
          schema: 'denovo.legacy-family/v1',
          kind: 'SequenceSet',
          id: artifact.id,
          sourceModelId: artifact.sourceModelId,
          parentStructureId: artifact.sourceStructureId ?? artifact.parentStructureId,
        }),
      );
      const setParentId = (artifact.sourceStructureId ?? artifact.parentStructureId ?? '').trim();
      const usedKeys = new Set<string>();
      const members: CollectionMember<SequenceCandidate>[] = artifact.candidates.map((cand, i) => {
        const ordinal =
          typeof cand.sampleIndex === 'number' && Number.isInteger(cand.sampleIndex) && cand.sampleIndex >= 0
            ? cand.sampleIndex
            : i;
        const candParentId = (cand.sourceStructureId ?? setParentId).trim();
        const explicitDesignKey = (cand.designKey ?? '').trim();
        const designKey = explicitDesignKey || (candParentId.length > 0 ? candParentId : undefined);
        const hasParent = candParentId.length > 0;
        let sequenceKey = (cand.sequenceKey ?? cand.key ?? cand.id ?? '').trim();
        if (!sequenceKey || usedKeys.has(sequenceKey)) {
          sequenceKey = makeSequenceKey(familyKey, designKey, ordinal);
        }
        usedKeys.add(sequenceKey);
        const lineage: ItemLineage = {
          ...(designKey
            ? {
                designKey,
                ...(typeof cand.designIndex === 'number' && Number.isInteger(cand.designIndex) && cand.designIndex >= 0
                  ? { designIndex: cand.designIndex }
                  : {}),
              }
            : {}),
          ...(hasParent
            ? {
                sourceStructure: {
                  artifactId: candParentId,
                  payloadDigest: '',
                  ...(designKey ? { designKey } : {}),
                },
              }
            : {}),
          sequenceKey,
          sequenceIndex: ordinal,
          parentKeys: designKey ? [designKey] : [],
        };
        return {
          key: sequenceKey,
          ordinal,
          memberKind: 'SequenceCandidate',
          lineage,
          ...(typeof cand.seed === 'number'
            ? {
                provenance: {
                  seedScheme: 'legacy-v1' as const,
                  seed: cand.seed,
                },
              }
            : {}),
          payload: cand,
          summary: {
            label: cand.id || sequenceKey,
            residueCount: cand.sequence.length,
            sequenceDigest: sha256Hex(cand.sequence),
            score: cand.score,
            globalScore: cand.globalScore,
          },
        };
      });
      const outcomes: WorkItemOutcome[] = members.map((m) => ({
        key: m.key,
        status: 'completed',
        attempt: 1,
        lineage: m.lineage,
      }));
      const id = computeCollectionSnapshotId(familyKey, members, outcomes);
      const collection: SequenceCollection = {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id,
        familyKey,
        revision: 1,
        memberKind: 'SequenceCandidate',
        collectionKind: artifact.collectionKind ?? 'independent-candidates',
        label: artifact.id,
        sourceModelId: artifact.sourceModelId,
        members,
        outcomes,
        requestedCount: members.length,
        complete: true,
        legacySourceKind: 'SequenceSet',
        legacySourceId: artifact.id,
      };
      const rep = selectRepresentativeMember(collection);
      if (rep) {
        return { ...collection, representativeSelection: rep.selection };
      }
      return collection;
    }

    case 'FoldedStructure': {
      const familyKey = sha256Hex(
        canonicalJson({
          schema: 'denovo.legacy-family/v1',
          kind: 'FoldedStructure',
          id: artifact.id,
          parentStructureId: artifact.sourceStructureId ?? artifact.parentStructureId,
        }),
      );
      const setParentId = (artifact.sourceStructureId ?? artifact.parentStructureId ?? '').trim();
      const sourceModelId = artifact.folds[0]?.foldModelId ?? 'localfold';

      const usedFoldKeys = new Set<string>();
      const members: CollectionMember<FoldedCandidate>[] = artifact.folds.map((fold, i) => {
        const sampleOrdinal =
          typeof fold.sampleIndex === 'number' && Number.isInteger(fold.sampleIndex) && fold.sampleIndex >= 0
            ? fold.sampleIndex
            : i;
        const sequenceKey = (fold.sequenceKey ?? fold.key ?? fold.sequenceId).trim() || `seq-legacy-${i}`;
        const explicitFoldKey = (fold.foldKey ?? '').trim();
        let foldKey =
          explicitFoldKey && explicitFoldKey !== sequenceKey && !usedFoldKeys.has(explicitFoldKey)
            ? explicitFoldKey
            : makeFoldKey(familyKey, sequenceKey, sampleOrdinal, fold.foldModelId || sourceModelId);
        if (usedFoldKeys.has(foldKey)) {
          foldKey = makeFoldKey(familyKey, `${sequenceKey}#${i}`, sampleOrdinal, fold.foldModelId || sourceModelId);
        }
        usedFoldKeys.add(foldKey);
        const candParentId = (fold.sourceStructureId ?? setParentId).trim();
        const explicitDesignKey = (fold.designKey ?? '').trim();
        const designKey = explicitDesignKey || (candParentId.length > 0 ? candParentId : undefined);
        const hasParent = candParentId.length > 0;
        const confidenceAvailable =
          fold.confidenceAvailable !== false &&
          fold.confidence !== null &&
          (typeof fold.confidence === 'number'
            ? Number.isFinite(fold.confidence)
            : Number.isFinite(fold.meanPlddt));
        const confidence = !confidenceAvailable
          ? null
          : typeof fold.confidence === 'number'
            ? fold.confidence
            : fold.meanPlddt;

        const lineage: ItemLineage = {
          ...(designKey
            ? {
                designKey,
                ...(typeof fold.designIndex === 'number' && Number.isInteger(fold.designIndex) && fold.designIndex >= 0
                  ? { designIndex: fold.designIndex }
                  : {}),
              }
            : {}),
          ...(hasParent
            ? {
                sourceStructure: {
                  artifactId: candParentId,
                  payloadDigest: '',
                  ...(designKey ? { designKey } : {}),
                },
              }
            : {}),
          sequenceKey,
          foldKey,
          foldIndex: sampleOrdinal,
          parentKeys: [sequenceKey],
        };
        return {
          key: foldKey,
          ordinal: sampleOrdinal,
          memberKind: 'FoldedCandidate',
          lineage,
          ...(typeof fold.seed === 'number' && Number.isFinite(fold.seed)
            ? {
                provenance: {
                  seedScheme: 'legacy-v1' as const,
                  seed: fold.seed,
                },
              }
            : {}),
          payload: fold,
          summary: {
            label: `${fold.sequenceId} (${fold.foldModelId})`,
            residueCount: fold.sequence.length,
            sequenceDigest: sha256Hex(fold.sequence),
            confidence,
            confidenceAvailable,
            ipsae: typeof fold.ipsae === 'number' && Number.isFinite(fold.ipsae) ? fold.ipsae : null,
          },
        };
      });

      const outcomes: WorkItemOutcome[] = members.map((m) => ({
        key: m.key,
        status: 'completed',
        attempt: 1,
        lineage: m.lineage,
      }));
      const completedKeys = new Set<string>();
      for (const m of members) {
        completedKeys.add(m.key);
        if (m.lineage.foldKey) completedKeys.add(m.lineage.foldKey);
        if (m.lineage.sequenceKey) completedKeys.add(m.lineage.sequenceKey);
        if (m.payload?.foldKey) completedKeys.add(m.payload.foldKey);
        if (m.payload?.sequenceKey) completedKeys.add(m.payload.sequenceKey);
        if (m.payload?.key) completedKeys.add(m.payload.key);
        if (m.payload?.sequenceId) completedKeys.add(m.payload.sequenceId);
      }
      const seenNonCompletedKeys = new Set<string>();
      for (const excl of artifact.excludedCandidates ?? []) {
        const exclKey = (excl.key || excl.id || '').trim();
        if (!exclKey || seenNonCompletedKeys.has(exclKey)) continue;
        seenNonCompletedKeys.add(exclKey);
        const extExcl = excl as typeof excl & { designKey?: string; sourceStructureId?: string };
        const exclDesignKey = (extExcl.designKey ?? extExcl.sourceStructureId ?? '').trim() || undefined;
        outcomes.push({
          key: exclKey,
          status: 'excluded',
          attempt: 0,
          reason: excl.reason,
          lineage: {
            ...(exclDesignKey ? { designKey: exclDesignKey } : {}),
            sequenceKey: exclKey,
            parentKeys: exclDesignKey ? [exclDesignKey] : [],
          },
        });
      }
      for (const itemOutcome of artifact.itemOutcomes ?? []) {
        const k = (itemOutcome.key ?? '').trim();
        if (!k) continue;
        if (itemOutcome.status === 'completed') {
          // Completed folds are already committed in `members`; never duplicate them as
          // trailing blank phantom rows (`fold:...:s0`) during CSV export.
          if (completedKeys.has(k) || members.length > 0) continue;
        }
        if (seenNonCompletedKeys.has(k) || outcomes.some((o) => o.key === k)) continue;
        seenNonCompletedKeys.add(k);
        outcomes.push({
          key: k,
          status: itemOutcome.status,
          attempt: itemOutcome.status === 'excluded' ? 0 : 1,
          reason: itemOutcome.reason,
        });
      }

      const id = computeCollectionSnapshotId(familyKey, members, outcomes);
      const collection: FoldCollection = {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id,
        familyKey,
        revision: 1,
        memberKind: 'FoldedCandidate',
        collectionKind: artifact.collectionKind ?? 'multi-sample',
        label: artifact.id,
        sourceModelId,
        members,
        outcomes,
        requestedCount: members.length,
        complete: true,
        legacySourceKind: 'FoldedStructure',
        legacySourceId: artifact.id,
      };
      const explicitKey = artifact.representativeSelection
        ? members.find(
            (m) =>
              m.key === artifact.representativeSelection?.selectedKey ||
              m.lineage.foldKey === artifact.representativeSelection?.selectedKey ||
              m.lineage.sequenceKey === artifact.representativeSelection?.selectedKey ||
              m.payload?.key === artifact.representativeSelection?.selectedKey ||
              m.payload?.sequenceId === artifact.representativeSelection?.selectedKey,
          )?.key
        : undefined;
      const rep = selectRepresentativeMember(collection, { explicitKey });
      if (rep) {
        return { ...collection, representativeSelection: rep.selection };
      }
      return collection;
    }

    case 'DesignabilityMetrics': {
      const familyKey = sha256Hex(
        canonicalJson({
          schema: 'denovo.legacy-family/v1',
          kind: 'DesignabilityMetrics',
          id: artifact.id,
          referenceStructureId: artifact.referenceStructureId ?? null,
        }),
      );
      const usedFoldKeys = new Set<string>();
      const members: CollectionMember<DesignabilityRow>[] = artifact.rows.map((row, i) => {
        const sequenceKey = (row.sequenceKey ?? row.candidateKey ?? row.sequenceId).trim() || `seq-row-${i}`;
        const explicitFoldKey = (row.foldKey ?? '').trim();
        let foldKey =
          explicitFoldKey && explicitFoldKey !== sequenceKey && !usedFoldKeys.has(explicitFoldKey)
            ? explicitFoldKey
            : makeFoldKey(familyKey, sequenceKey, i, 'scorer');
        if (usedFoldKeys.has(foldKey)) {
          foldKey = makeFoldKey(familyKey, `${sequenceKey}#${i}`, i, 'scorer');
        }
        usedFoldKeys.add(foldKey);
        const refId = (row.sourceStructureId ?? row.structureId ?? '').trim();
        const explicitDesignKey = (row.designKey ?? '').trim();
        const designKey = explicitDesignKey || (refId.length > 0 ? refId : undefined);
        const hasParent = refId.length > 0;
        const confidenceAvailable = row.confidenceAvailable !== false && Number.isFinite(row.meanPlddt);
        const lineage: ItemLineage = {
          ...(designKey
            ? {
                designKey,
                ...(typeof row.designIndex === 'number' && Number.isInteger(row.designIndex) && row.designIndex >= 0
                  ? { designIndex: row.designIndex }
                  : {}),
              }
            : {}),
          ...(hasParent
            ? {
                sourceStructure: {
                  artifactId: refId,
                  payloadDigest: '',
                  ...(designKey ? { designKey } : {}),
                },
              }
            : {}),
          sequenceKey,
          foldKey,
          foldIndex: i,
          parentKeys: [sequenceKey],
        };
        return {
          key: foldKey,
          ordinal: i,
          memberKind: 'DesignabilityRow',
          lineage,
          payload: row,
          summary: {
            label: `${row.sequenceId} vs ${row.structureId}`,
            residueCount: row.sequence.length,
            sequenceDigest: sha256Hex(row.sequence),
            confidence: confidenceAvailable ? row.meanPlddt : null,
            confidenceAvailable,
            designable: row.designable,
            scRmsd: Number.isFinite(row.scRmsd) ? row.scRmsd : null,
            ipsae: typeof row.ipsae === 'number' && Number.isFinite(row.ipsae) ? row.ipsae : null,
          },
        };
      });
      const outcomes: WorkItemOutcome[] = members.map((m) => ({
        key: m.key,
        status: 'completed',
        attempt: 1,
        lineage: m.lineage,
      }));
      const id = computeCollectionSnapshotId(familyKey, members, outcomes);
      const collection: DesignabilityCollection = {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id,
        familyKey,
        revision: 1,
        memberKind: 'DesignabilityRow',
        collectionKind: artifact.collectionKind ?? 'multi-sample',
        label: artifact.id,
        sourceModelId: 'scorer',
        members,
        outcomes,
        requestedCount: members.length,
        complete: true,
        legacySourceKind: 'DesignabilityMetrics',
        legacySourceId: artifact.id,
      };
      const rep = selectRepresentativeMember(collection);
      if (rep) {
        return { ...collection, representativeSelection: rep.selection };
      }
      return collection;
    }
  }
}

// ============================================================================
// Collection Invariant Validator (RFC §2.1)
// ============================================================================

interface IndexedStructureEntry {
  structure: StructureArtifact;
  member?: CollectionMember<StructureArtifact>;
  designKey?: MemberKey;
  digest: Digest;
}

function indexStructurePool(
  pool:
    | StructureCollection
    | readonly CollectionMember<StructureArtifact>[]
    | ReadonlyMap<MemberKey, CollectionMember<StructureArtifact>>
    | readonly StructureArtifact[]
    | undefined,
): {
  byDesignKey: Map<MemberKey, IndexedStructureEntry>;
  byArtifactId: Map<string, IndexedStructureEntry>;
} {
  const byDesignKey = new Map<MemberKey, IndexedStructureEntry>();
  const byArtifactId = new Map<string, IndexedStructureEntry>();
  if (!pool) return { byDesignKey, byArtifactId };

  const addMember = (member: CollectionMember<StructureArtifact>) => {
    if (!member.payload) return;
    const structure = member.payload;
    const digest =
      member.lineage.sourceStructure?.payloadDigest ||
      computeStructurePayloadDigest(structure);
    const designKey = member.lineage.designKey ?? member.key;
    const entry: IndexedStructureEntry = { structure, member, designKey, digest };
    byDesignKey.set(designKey, entry);
    byDesignKey.set(member.key, entry);
    byArtifactId.set(structure.id, entry);
  };

  if (isDesignCollection(pool)) {
    for (const m of pool.members) addMember(m);
  } else if (pool instanceof Map) {
    for (const m of pool.values()) addMember(m);
  } else if (Array.isArray(pool)) {
    for (const item of pool) {
      if (item && typeof item === 'object' && 'kind' in item && item.kind === 'StructureArtifact') {
        const struct = item as StructureArtifact;
        const digest = computeStructurePayloadDigest(struct);
        const entry: IndexedStructureEntry = { structure: struct, digest };
        byArtifactId.set(struct.id, entry);
      } else if (item && typeof item === 'object' && 'memberKind' in item) {
        addMember(item as CollectionMember<StructureArtifact>);
      }
    }
  }

  return { byDesignKey, byArtifactId };
}

/**
 * Validate collection invariants (RFC §2.1):
 * - Unique member keys (never silently repaired with suffixes).
 * - Valid ordinals and memberKind consistency.
 * - Exact parent linkage (designKey / sequenceKey / foldKey / sourceStructure).
 * - Distinct foldKey !== sequenceKey.
 * - No cross-parent aliasing or artifactId/payloadDigest conflicts.
 */
export function validateCollectionInvariants<T>(
  collection: DesignCollection<T>,
  context: CollectionValidationContext = {},
): CollectionValidationResult {
  const errors: CollectionInvariantViolation[] = [];
  const seenKeys = new Set<MemberKey>();
  const seenDesignKeys = new Set<MemberKey>();
  const verifyDigests = context.verifyPayloadDigests !== false;

  const knownStructIndex = indexStructurePool(context.knownStructures);
  const parentKeySet = context.parentCollection
    ? new Set(context.parentCollection.members.map((m) => m.key))
    : null;

  for (const member of collection.members) {
    if (typeof member.key !== 'string' || member.key.trim().length === 0) {
      errors.push({
        code: 'invalid-member-key',
        memberKey: member.key,
        message: 'Collection member has an empty or non-string key.',
      });
      continue;
    }

    if (seenKeys.has(member.key)) {
      errors.push({
        code: 'duplicate-member-key',
        memberKey: member.key,
        message: `Duplicate member key "${member.key}" in collection "${collection.id}".`,
      });
    } else {
      seenKeys.add(member.key);
    }

    if (!Number.isInteger(member.ordinal) || member.ordinal < 0) {
      errors.push({
        code: 'invalid-ordinal',
        memberKey: member.key,
        message: `Member "${member.key}" has invalid ordinal ${String(member.ordinal)}.`,
      });
    }

    if (member.memberKind !== collection.memberKind) {
      errors.push({
        code: 'member-kind-mismatch',
        memberKey: member.key,
        message: `Member "${member.key}" has kind "${member.memberKind}" but collection expects "${collection.memberKind}".`,
      });
    }

    const { lineage } = member;

    if (member.memberKind === 'StructureArtifact') {
      if (!lineage.designKey || lineage.designKey.trim().length === 0) {
        errors.push({
          code: 'missing-design-key',
          memberKey: member.key,
          message: `Structure member "${member.key}" must declare a unique lineage.designKey.`,
        });
      } else if (seenDesignKeys.has(lineage.designKey)) {
        errors.push({
          code: 'duplicate-design-key',
          memberKey: member.key,
          message: `Duplicate designKey "${lineage.designKey}" in structure collection.`,
        });
      } else {
        seenDesignKeys.add(lineage.designKey);
      }
    }

    if (member.memberKind === 'SequenceCandidate') {
      if (!lineage.sequenceKey || lineage.sequenceKey.trim().length === 0) {
        errors.push({
          code: 'missing-sequence-key',
          memberKey: member.key,
          message: `Sequence member "${member.key}" must declare lineage.sequenceKey.`,
        });
      }
      const hasDesign = Boolean(lineage.designKey && lineage.designKey.trim().length > 0);
      const hasSourceStruct = Boolean(lineage.sourceStructure?.artifactId);
      if (hasDesign !== hasSourceStruct) {
        errors.push({
          code: 'inconsistent-sequence-parent',
          memberKey: member.key,
          message: `Sequence member "${member.key}" must have both designKey and sourceStructure (or neither if standalone).`,
        });
      }
      if (hasDesign && lineage.designKey && !lineage.parentKeys.includes(lineage.designKey)) {
        errors.push({
          code: 'missing-parent-edge',
          memberKey: member.key,
          message: `Sequence member "${member.key}" is missing parentKey "${lineage.designKey}".`,
        });
      }
      if (parentKeySet && hasDesign && lineage.designKey && !parentKeySet.has(lineage.designKey)) {
        errors.push({
          code: 'missing-parent-member',
          memberKey: member.key,
          message: `Sequence member "${member.key}" references unknown parent design "${lineage.designKey}".`,
        });
      }
    }

    if (member.memberKind === 'FoldedCandidate' || member.memberKind === 'DesignabilityRow') {
      if (!lineage.foldKey || lineage.foldKey.trim().length === 0) {
        errors.push({
          code: 'missing-fold-key',
          memberKey: member.key,
          message: `Fold member "${member.key}" must declare lineage.foldKey.`,
        });
      }
      if (!lineage.sequenceKey || lineage.sequenceKey.trim().length === 0) {
        errors.push({
          code: 'missing-sequence-key',
          memberKey: member.key,
          message: `Fold member "${member.key}" must retain its parent lineage.sequenceKey.`,
        });
      }
      if (lineage.foldKey && lineage.sequenceKey && lineage.foldKey === lineage.sequenceKey) {
        errors.push({
          code: 'fold-reused-sequence-key',
          memberKey: member.key,
          message: `Fold member "${member.key}" reused sequenceKey "${lineage.sequenceKey}" as foldKey.`,
        });
      }
      if (lineage.sequenceKey && !lineage.parentKeys.includes(lineage.sequenceKey)) {
        errors.push({
          code: 'missing-parent-edge',
          memberKey: member.key,
          message: `Fold member "${member.key}" is missing parentKey "${lineage.sequenceKey}".`,
        });
      }
      if (parentKeySet && lineage.sequenceKey && !parentKeySet.has(lineage.sequenceKey)) {
        errors.push({
          code: 'missing-parent-member',
          memberKey: member.key,
          message: `Fold member "${member.key}" references unknown parent sequence "${lineage.sequenceKey}".`,
        });
      }
    }

    // Check payload sourceStructureId agreement with lineage.sourceStructure.artifactId
    if (member.payload && typeof member.payload === 'object' && lineage.sourceStructure) {
      const payloadStructId = (member.payload as { sourceStructureId?: unknown }).sourceStructureId;
      if (
        typeof payloadStructId === 'string' &&
        payloadStructId.trim().length > 0 &&
        payloadStructId !== lineage.sourceStructure.artifactId
      ) {
        errors.push({
          code: 'source-structure-conflict',
          memberKey: member.key,
          message: `Member "${member.key}" payload sourceStructureId "${payloadStructId}" conflicts with lineage sourceStructure "${lineage.sourceStructure.artifactId}".`,
        });
      }
    }

    // Cross-parent aliasing check against knownStructures
    if (knownStructIndex.byDesignKey.size > 0 || knownStructIndex.byArtifactId.size > 0) {
      const byDesign = lineage.designKey ? knownStructIndex.byDesignKey.get(lineage.designKey) : undefined;
      const byArt = lineage.sourceStructure?.artifactId
        ? knownStructIndex.byArtifactId.get(lineage.sourceStructure.artifactId)
        : undefined;

      if (byDesign && byArt && byDesign.structure.id !== byArt.structure.id) {
        errors.push({
          code: 'cross-parent-aliasing',
          memberKey: member.key,
          message: `Member "${member.key}" aliases designKey "${lineage.designKey}" (${byDesign.structure.id}) with sourceStructure "${byArt.structure.id}".`,
        });
      }

      const resolved = byDesign ?? byArt;
      if (
        resolved &&
        verifyDigests &&
        lineage.sourceStructure?.payloadDigest &&
        lineage.sourceStructure.payloadDigest.length > 0 &&
        lineage.sourceStructure.payloadDigest !== resolved.digest
      ) {
        errors.push({
          code: 'source-structure-conflict',
          memberKey: member.key,
          message: `Member "${member.key}" sourceStructure payloadDigest does not match structure "${resolved.structure.id}".`,
        });
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

// ============================================================================
// Exact Parent Structure Resolution (RFC Gates #5 & #6)
// ============================================================================

export type ParentResolvableChild =
  | CollectionMember<SequenceCandidate | FoldedCandidate | DesignabilityRow | unknown>
  | {
      key?: MemberKey;
      lineage: ItemLineage;
      expectedLength?: number;
    };

/**
 * Resolve the exact parent StructureArtifact for a sequence, fold, or score row.
 *
 * STRICT INVARIANTS (RFC §6.1, Gates #5 & #6):
 * - Uses ONLY exact declared lineage (designKey and/or sourceStructure.artifactId + payloadDigest).
 * - NEVER falls back to a collection representative, array[0], nearest upstream structure,
 *   or an equal-length heuristic.
 * - Rejects cross-parent conflicts, digest mismatches, and length mismatches (never truncates).
 */
export function resolveExactParentStructure(
  child: ParentResolvableChild,
  structurePool:
    | StructureCollection
    | readonly CollectionMember<StructureArtifact>[]
    | ReadonlyMap<MemberKey, CollectionMember<StructureArtifact>>
    | readonly StructureArtifact[],
  options: ResolveParentOptions = {},
): ParentStructureResolution {
  const verifyDigest = options.verifyPayloadDigest !== false;
  let effectiveDesignKey = child.lineage.designKey?.trim() || undefined;
  let effectiveSourceStruct = child.lineage.sourceStructure;

  // Step 1: If a sequenceCollection is provided and child has a sequenceKey, verify fold -> sequence agreement.
  if (options.sequenceCollection && child.lineage.sequenceKey) {
    const seqMembers = isDesignCollection(options.sequenceCollection)
      ? options.sequenceCollection.members
      : options.sequenceCollection;
    const seqMember = seqMembers.find(
      (m) => m.key === child.lineage.sequenceKey || m.lineage.sequenceKey === child.lineage.sequenceKey,
    );
    if (!seqMember) {
      return {
        ok: false,
        code: 'sequence-not-found',
        reason: `Parent sequence "${child.lineage.sequenceKey}" was not found in the sequence collection.`,
      };
    }

    const seqDesignKey = seqMember.lineage.designKey?.trim() || undefined;
    const seqSource = seqMember.lineage.sourceStructure;

    if (effectiveDesignKey && seqDesignKey && effectiveDesignKey !== seqDesignKey) {
      return {
        ok: false,
        code: 'lineage-conflict',
        reason: `Child designKey "${effectiveDesignKey}" conflicts with parent sequence "${seqMember.key}" designKey "${seqDesignKey}".`,
      };
    }
    if (
      effectiveSourceStruct?.artifactId &&
      seqSource?.artifactId &&
      effectiveSourceStruct.artifactId !== seqSource.artifactId
    ) {
      return {
        ok: false,
        code: 'lineage-conflict',
        reason: `Child sourceStructure "${effectiveSourceStruct.artifactId}" conflicts with parent sequence "${seqMember.key}" sourceStructure "${seqSource.artifactId}".`,
      };
    }
    if (
      effectiveSourceStruct?.payloadDigest &&
      seqSource?.payloadDigest &&
      effectiveSourceStruct.payloadDigest !== seqSource.payloadDigest
    ) {
      return {
        ok: false,
        code: 'digest-mismatch',
        reason: `Child payloadDigest conflicts with parent sequence "${seqMember.key}" sourceStructure digest.`,
      };
    }

    effectiveDesignKey = effectiveDesignKey ?? seqDesignKey;
    effectiveSourceStruct = effectiveSourceStruct ?? seqSource;
  }

  // Step 2: Require explicit parent anchor (standalone sequences have no parent structure).
  const declaredArtifactId = effectiveSourceStruct?.artifactId?.trim() || undefined;
  if (!effectiveDesignKey && !declaredArtifactId) {
    return {
      ok: false,
      code: 'unlinked-parent',
      reason: 'Child has no declared designKey or sourceStructure; refusing implicit fallback.',
    };
  }

  const index = indexStructurePool(structurePool);
  const matchByDesign = effectiveDesignKey ? index.byDesignKey.get(effectiveDesignKey) : undefined;
  const matchByArtifact = declaredArtifactId ? index.byArtifactId.get(declaredArtifactId) : undefined;

  // Step 3: Reject cross-parent conflict if both anchors are declared and disagree.
  if (effectiveDesignKey && declaredArtifactId) {
    if (matchByDesign && matchByDesign.structure.id !== declaredArtifactId) {
      return {
        ok: false,
        code: 'cross-parent-conflict',
        reason: `Declared designKey "${effectiveDesignKey}" resolved to structure "${matchByDesign.structure.id}", which conflicts with declared sourceStructure "${declaredArtifactId}".`,
      };
    }
    if (matchByArtifact && matchByArtifact.designKey && matchByArtifact.designKey !== effectiveDesignKey) {
      return {
        ok: false,
        code: 'cross-parent-conflict',
        reason: `Declared sourceStructure "${declaredArtifactId}" belongs to design "${matchByArtifact.designKey}", which conflicts with declared designKey "${effectiveDesignKey}".`,
      };
    }
    if (!matchByDesign && !matchByArtifact) {
      return {
        ok: false,
        code: 'parent-not-found',
        reason: `Neither designKey "${effectiveDesignKey}" nor sourceStructure "${declaredArtifactId}" exists in the structure pool.`,
      };
    }
    if (!matchByDesign && matchByArtifact && matchByArtifact.designKey) {
      return {
        ok: false,
        code: 'cross-parent-conflict',
        reason: `Declared designKey "${effectiveDesignKey}" is absent; refusing to alias to design "${matchByArtifact.designKey}".`,
      };
    }
  }

  const resolved = matchByDesign ?? matchByArtifact;
  if (!resolved) {
    return {
      ok: false,
      code: 'parent-not-found',
      reason: `Parent structure (${effectiveDesignKey ?? declaredArtifactId}) was not found in the structure pool.`,
    };
  }

  // Step 4: Verify payloadDigest when declared.
  if (
    verifyDigest &&
    effectiveSourceStruct?.payloadDigest &&
    effectiveSourceStruct.payloadDigest.length > 0
  ) {
    const actualDigest = computeStructurePayloadDigest(resolved.structure);
    if (effectiveSourceStruct.payloadDigest !== actualDigest) {
      return {
        ok: false,
        code: 'digest-mismatch',
        reason: `Parent structure "${resolved.structure.id}" payloadDigest (${actualDigest.slice(0, 12)}) does not match declared digest (${effectiveSourceStruct.payloadDigest.slice(0, 12)}).`,
      };
    }
  }

  // Step 5: Verify exact residue length when provided (never truncate to min(length)).
  const expectedLength =
    options.expectedLength ??
    ('expectedLength' in child ? child.expectedLength : undefined) ??
    ('summary' in child ? child.summary.residueCount : undefined);

  if (
    typeof expectedLength === 'number' &&
    Number.isInteger(expectedLength) &&
    expectedLength > 0 &&
    resolved.structure.length !== expectedLength
  ) {
    return {
      ok: false,
      code: 'length-mismatch',
      reason: `Parent structure "${resolved.structure.id}" has ${resolved.structure.length} residues, which mismatches child length ${expectedLength}.`,
    };
  }

  const actualDigest = resolved.digest || computeStructurePayloadDigest(resolved.structure);
  return {
    ok: true,
    structure: resolved.structure,
    member: resolved.member,
    designKey: resolved.designKey ?? effectiveDesignKey,
    structureIdentity: {
      artifactId: resolved.structure.id,
      payloadDigest: actualDigest,
      designKey: resolved.designKey ?? effectiveDesignKey,
    },
  };
}

// ============================================================================
// Lineage-Aware Deterministic Representative Selection (RFC §2.3)
// ============================================================================

export interface SelectRepresentativeOptions {
  explicitKey?: MemberKey;
  criterionVersion?: number;
}

function resolveMemberConfidence<T>(member: CollectionMember<T>): number | null {
  if (member.summary.confidenceAvailable === false) return null;
  if (member.summary.confidence === null || member.summary.confidence === undefined) return null;
  return Number.isFinite(member.summary.confidence) ? member.summary.confidence : null;
}

function compareByOrdinalThenKey<T>(a: CollectionMember<T>, b: CollectionMember<T>): number {
  if (a.ordinal !== b.ordinal) return a.ordinal - b.ordinal;
  return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
}

/**
 * Lineage-aware deterministic representative selector (RFC §2.3):
 * - Empty collection -> null.
 * - Explicit selected key wins when present in members.
 * - Ordered trajectories ('ordered-frames') -> final committed frame.
 * - Multi-sample / fold / designability collections -> highest available confidence,
 *   tie-broken by creation ordinal then key; confidence-less -> lowest creation ordinal.
 * - Independent candidates (structures / sequences) -> lowest creation ordinal, tie-broken by key.
 */
export function selectRepresentativeMember<T>(
  collection: DesignCollection<T>,
  options: SelectRepresentativeOptions = {},
): { member: CollectionMember<T>; selection: CollectionRepresentativeSelection } | null {
  const { members } = collection;
  if (members.length === 0) return null;

  const criterionVersion = options.criterionVersion ?? 1;

  // 1. Explicit key wins
  if (options.explicitKey) {
    const idx = members.findIndex((m) => m.key === options.explicitKey);
    if (idx >= 0) {
      const chosen = members[idx];
      return {
        member: chosen,
        selection: {
          collectionId: collection.id,
          criterionVersion,
          selectedKey: chosen.key,
          criterion: 'explicit-selection',
          totalMembers: members.length,
          selectedIndex: idx,
          scoreValue: resolveMemberConfidence(chosen) ?? chosen.summary.score ?? null,
        },
      };
    }
  }

  // 2. Ordered trajectories: final committed frame (highest ordinal)
  if (collection.collectionKind === 'ordered-frames') {
    let bestIdx = 0;
    for (let i = 1; i < members.length; i++) {
      const cur = members[i];
      const best = members[bestIdx];
      if (cur.ordinal > best.ordinal || (cur.ordinal === best.ordinal && cur.key > best.key)) {
        bestIdx = i;
      }
    }
    const chosen = members[bestIdx];
    return {
      member: chosen,
      selection: {
        collectionId: collection.id,
        criterionVersion,
        selectedKey: chosen.key,
        criterion: 'final-committed-frame',
        totalMembers: members.length,
        selectedIndex: bestIdx,
        scoreValue: resolveMemberConfidence(chosen),
      },
    };
  }

  // 3. Confidence-bearing multi-sample / fold / designability collections
  const isConfidenceDriven =
    collection.collectionKind === 'multi-sample' ||
    collection.memberKind === 'FoldedCandidate' ||
    collection.memberKind === 'DesignabilityRow';

  if (isConfidenceDriven) {
    let bestIdx = -1;
    let bestConf = -Infinity;
    for (let i = 0; i < members.length; i++) {
      const conf = resolveMemberConfidence(members[i]);
      if (conf === null) continue;
      if (
        bestIdx === -1 ||
        conf > bestConf ||
        (conf === bestConf && compareByOrdinalThenKey(members[i], members[bestIdx]) < 0)
      ) {
        bestConf = conf;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) {
      const chosen = members[bestIdx];
      return {
        member: chosen,
        selection: {
          collectionId: collection.id,
          criterionVersion,
          selectedKey: chosen.key,
          criterion: 'highest-available-confidence',
          totalMembers: members.length,
          selectedIndex: bestIdx,
          scoreValue: bestConf,
        },
      };
    }

    // Confidence-less fallback: lowest creation ordinal
    let fallbackIdx = 0;
    for (let i = 1; i < members.length; i++) {
      if (compareByOrdinalThenKey(members[i], members[fallbackIdx]) < 0) {
        fallbackIdx = i;
      }
    }
    const chosen = members[fallbackIdx];
    return {
      member: chosen,
      selection: {
        collectionId: collection.id,
        criterionVersion,
        selectedKey: chosen.key,
        criterion: 'lowest-ordinal-fallback (no confidence score)',
        totalMembers: members.length,
        selectedIndex: fallbackIdx,
        scoreValue: null,
      },
    };
  }

  // 4. Independent unscored designs / candidates: lowest creation ordinal
  let firstIdx = 0;
  for (let i = 1; i < members.length; i++) {
    if (compareByOrdinalThenKey(members[i], members[firstIdx]) < 0) {
      firstIdx = i;
    }
  }
  const chosen = members[firstIdx];
  return {
    member: chosen,
    selection: {
      collectionId: collection.id,
      criterionVersion,
      selectedKey: chosen.key,
      criterion: 'lowest-creation-ordinal',
      totalMembers: members.length,
      selectedIndex: firstIdx,
      scoreValue: chosen.summary.score ?? null,
    },
  };
}

// ============================================================================
// Per-Design & Campaign Aggregation (RFC §6.2, Gate #9)
// ============================================================================

function isRecordEvaluable(record: FoldEvaluationRecord, requireConfidence: boolean): boolean {
  if (record.status !== 'completed') return false;
  if (record.evaluable === false) return false;
  if (requireConfidence && record.confidenceAvailable === false) return false;
  return typeof record.designable === 'boolean';
}

/**
 * Compute per-design pass summaries from fold/scoring records.
 * - Zero denominators ALWAYS yield `null` (never fabricated 0).
 * - Excluded folds are tracked separately from plannedFolds.
 * - Confidence-less samples under a confidence-dependent criterion are counted as
 *   `ineligibleFolds`, never fabricated failures.
 */
export function computeDesignPassSummaries(
  records: readonly FoldEvaluationRecord[],
  options: DesignPassSummaryOptions = {},
): DesignPassSummary[] {
  const requireConfidence = options.requireConfidence !== false;
  const seqRule = options.sequenceAggregationRule ?? 'any-sample-passes';

  const orderedDesignKeys: MemberKey[] = [];
  const seenDesigns = new Set<MemberKey>();
  const grouped = new Map<MemberKey, FoldEvaluationRecord[]>();

  const registerDesign = (designKey: MemberKey) => {
    if (!seenDesigns.has(designKey)) {
      seenDesigns.add(designKey);
      orderedDesignKeys.push(designKey);
      grouped.set(designKey, []);
    }
  };

  for (const dk of options.designKeys ?? []) {
    registerDesign(dk);
  }
  for (const rec of records) {
    registerDesign(rec.designKey);
    grouped.get(rec.designKey)!.push(rec);
  }

  return orderedDesignKeys.map((designKey) => {
    const items = grouped.get(designKey) ?? [];
    let plannedFolds = 0;
    let completedFolds = 0;
    let evaluableFolds = 0;
    let passedFolds = 0;
    let failedFolds = 0;
    let excludedFolds = 0;
    let pendingFolds = 0;
    let ineligibleFolds = 0;

    // Group evaluable outcomes per sequence to compute hasPassingSequence
    const seqStats = new Map<MemberKey, { evaluable: number; passed: number }>();

    for (const rec of items) {
      if (rec.status === 'excluded') {
        excludedFolds++;
        continue;
      }
      plannedFolds++;
      if (rec.status === 'failed') {
        failedFolds++;
        continue;
      }
      if (rec.status === 'pending' || rec.status === 'running' || rec.status === 'cancelled') {
        pendingFolds++;
        continue;
      }
      // status === 'completed'
      completedFolds++;
      if (!isRecordEvaluable(rec, requireConfidence)) {
        ineligibleFolds++;
        continue;
      }
      evaluableFolds++;
      const entry = seqStats.get(rec.sequenceKey) ?? { evaluable: 0, passed: 0 };
      entry.evaluable++;
      if (rec.designable === true) {
        passedFolds++;
        entry.passed++;
      }
      seqStats.set(rec.sequenceKey, entry);
    }

    let hasPassingSequence = false;
    for (const stat of seqStats.values()) {
      if (seqRule === 'all-samples-pass') {
        if (stat.evaluable > 0 && stat.passed === stat.evaluable) {
          hasPassingSequence = true;
          break;
        }
      } else {
        if (stat.passed > 0) {
          hasPassingSequence = true;
          break;
        }
      }
    }

    const foldPassRate = evaluableFolds > 0 ? passedFolds / evaluableFolds : null;
    const complete = plannedFolds > 0 && pendingFolds === 0;

    return {
      designKey,
      plannedFolds,
      completedFolds,
      evaluableFolds,
      passedFolds,
      failedFolds,
      excludedFolds,
      pendingFolds,
      ineligibleFolds,
      foldPassRate,
      hasPassingSequence,
      complete,
    };
  });
}

/**
 * Compute campaign-level summary metrics (RFC §6.2, Gate #9):
 * - foldPassRate = passing folds / evaluable folds (null if 0 evaluable)
 * - designHitRate = designs with >=1 passing sequence / evaluated designs (null if 0 evaluated)
 * - meanPerDesignFoldPassRate = equal-weight mean across designs with evaluableFolds > 0 (null if none)
 * - coverage = completed & evaluable folds / planned folds (null if 0 planned)
 * - conservativeCampaignYield = passing designs / requested designs (null if 0 requested)
 */
export function computeCampaignSummary(
  records: readonly FoldEvaluationRecord[],
  options: CampaignSummaryOptions = {},
): CampaignSummary {
  const samplesPerSequence = Math.max(1, Math.round(options.samplesPerSequence ?? 1));
  const seqRule = options.sequenceAggregationRule ?? 'any-sample-passes';
  const aggregationPolicy =
    seqRule === 'all-samples-pass'
      ? `all-of-${samplesPerSequence}-pass`
      : `at-least-one-of-${samplesPerSequence}-passes`;

  const perDesign = computeDesignPassSummaries(records, options);
  const requestedDesigns = Math.max(
    0,
    options.requestedDesigns !== undefined ? Math.round(options.requestedDesigns) : perDesign.length,
  );

  let evaluatedDesigns = 0;
  let passingDesigns = 0;
  let plannedFolds = 0;
  let completedFolds = 0;
  let evaluableFolds = 0;
  let passedFolds = 0;
  let failedFolds = 0;
  let excludedFolds = 0;
  let pendingFolds = 0;
  let ineligibleFolds = 0;

  let perDesignRateSum = 0;
  let perDesignRateCount = 0;

  for (const d of perDesign) {
    if (d.evaluableFolds > 0) evaluatedDesigns++;
    if (d.hasPassingSequence) passingDesigns++;
    plannedFolds += d.plannedFolds;
    completedFolds += d.completedFolds;
    evaluableFolds += d.evaluableFolds;
    passedFolds += d.passedFolds;
    failedFolds += d.failedFolds;
    excludedFolds += d.excludedFolds;
    pendingFolds += d.pendingFolds;
    ineligibleFolds += d.ineligibleFolds;
    if (d.foldPassRate !== null) {
      perDesignRateSum += d.foldPassRate;
      perDesignRateCount++;
    }
  }

  const foldPassRate = evaluableFolds > 0 ? passedFolds / evaluableFolds : null;
  const designHitRate = evaluatedDesigns > 0 ? passingDesigns / evaluatedDesigns : null;
  const meanPerDesignFoldPassRate =
    perDesignRateCount > 0 ? perDesignRateSum / perDesignRateCount : null;
  const conservativeCampaignYield =
    requestedDesigns > 0 ? passingDesigns / requestedDesigns : null;

  const coverage: CampaignCoverageSummary = {
    plannedFolds,
    completedFolds,
    evaluableFolds,
    completedFraction: plannedFolds > 0 ? completedFolds / plannedFolds : null,
    evaluableFraction: plannedFolds > 0 ? evaluableFolds / plannedFolds : null,
  };

  const complete =
    requestedDesigns > 0 &&
    perDesign.length >= requestedDesigns &&
    perDesign.every((d) => d.complete);

  return {
    aggregationPolicy,
    samplesPerSequence,
    ...(options.predictorModelId ? { predictorModelId: options.predictorModelId } : {}),
    requestedDesigns,
    evaluatedDesigns,
    passingDesigns,
    plannedFolds,
    completedFolds,
    evaluableFolds,
    passedFolds,
    failedFolds,
    excludedFolds,
    pendingFolds,
    ineligibleFolds,
    foldPassRate,
    designHitRate,
    meanPerDesignFoldPassRate,
    coverage,
    conservativeCampaignYield,
    perDesign,
    complete,
  };
}
