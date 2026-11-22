// Wave-6A Multi-Design & Batching Core: Snapshot-Consistent Streaming Collection Exporter.
//
// Implements RFC §7 & Gate #13:
// - FASTA export with stable design/sequence keys, model, seed, and chain boundaries.
// - CSV export with one row per fold/score + explicit exclusions, lineage keys,
//   exact reference ID, status, thresholds, and confidence availability.
// - Multi-MODEL PDB export with explicit selected-member order + manifest, REFUSING
//   to silently flatten heterogeneous atom topologies or variable-length designs.
// - Streaming bundle entry generator (`buildCollectionBundleEntries`) yielding one
//   entry at a time (PDB/mmCIF + FASTA + scores CSV + residue tracks + versioned manifest.json)
//   without assembling all PDBs simultaneously in memory.

import {
  isDesignCollection,
  sha256Hex,
  type CampaignSummary,
  type CollectionMember,
  type DesignabilityCollection,
  type DesignCollection,
  type Digest,
  type FoldCollection,
  type ItemLineage,
  type ItemStatus,
  type MemberKey,
  type ResidueTrackCollection,
  type SequenceCollection,
  type StructureCollection,
  type WorkItemOutcome,
} from './collections';
import type {
  DesignabilityRow,
  FoldedCandidate,
  ResidueTrackArtifact,
  SequenceCandidate,
  StructureArtifact,
} from './types';

// ============================================================================
// Typed Errors
// ============================================================================

export interface TopologySignature {
  memberKey: MemberKey;
  atomCount: number;
  residueCount: number;
  chainAtomCounts: readonly string[];
  atomNamesDigest: Digest;
}

export class HeterogeneousTopologyError extends Error {
  readonly code = 'HETEROGENEOUS_TOPOLOGY' as const;
  readonly signatures: readonly TopologySignature[];

  constructor(message: string, signatures: readonly TopologySignature[]) {
    super(message);
    this.name = 'HeterogeneousTopologyError';
    this.signatures = signatures;
  }
}

// ============================================================================
// 1. FASTA Export
// ============================================================================

export interface ExportFastaOptions {
  /** Optional explicit ordered subset of member keys to export. */
  selectedKeys?: readonly MemberKey[];
  /** Optional line wrap width (default 80; 0 disables wrapping). */
  lineWidth?: number;
}

function formatChainSequence(sequence: string, chainLengths?: readonly number[]): string {
  if (!chainLengths || chainLengths.length <= 1) return sequence;
  const total = chainLengths.reduce((acc, n) => acc + n, 0);
  if (total !== sequence.length) return sequence;
  const parts: string[] = [];
  let offset = 0;
  for (const len of chainLengths) {
    parts.push(sequence.slice(offset, offset + len));
    offset += len;
  }
  return parts.join(':');
}

function wrapLines(text: string, width: number): string {
  if (width <= 0 || text.length <= width) return text;
  const out: string[] = [];
  for (let i = 0; i < text.length; i += width) {
    out.push(text.slice(i, i + width));
  }
  return out.join('\n');
}

/**
 * Export a frozen sequence/fold collection to FASTA with deterministic headers carrying
 * stable sequenceKey, designKey, model, seed, score, and chain boundaries.
 */
export function exportCollectionFasta(
  collection:
    | SequenceCollection
    | FoldCollection
    | DesignCollection<SequenceCandidate | FoldedCandidate | StructureArtifact>,
  options: ExportFastaOptions = {},
): string {
  const lineWidth = options.lineWidth ?? 80;
  const orderedMembers = selectOrderedMembers(collection.members, options.selectedKeys);

  const records: string[] = [];
  for (const member of orderedMembers) {
    const payload = member.payload as
      | (Partial<SequenceCandidate> & Partial<FoldedCandidate> & Partial<StructureArtifact>)
      | undefined;
    const rawSeq = payload?.sequence ?? '';
    if (!rawSeq) continue;

    const seqKey = member.lineage.sequenceKey ?? member.key;
    const designKey = member.lineage.designKey ?? 'standalone';
    const model =
      payload?.sourceModelId ??
      payload?.foldModelId ??
      collection.sourceModelId ??
      'unknown';
    const seed = member.provenance?.seed ?? payload?.seed;
    const chainLengths = payload?.chainLengths;
    const formattedSeq = formatChainSequence(rawSeq, chainLengths);
    const chainBoundarySpec =
      chainLengths && chainLengths.length > 0
        ? chainLengths.join(',')
        : String(rawSeq.length);

    const headerParts: string[] = [
      seqKey,
      `design=${designKey}`,
      `model=${model}`,
      `ordinal=${member.ordinal}`,
      `chains=${chainBoundarySpec}`,
    ];
    if (member.lineage.foldKey) {
      headerParts.push(`fold=${member.lineage.foldKey}`);
    }
    if (typeof seed === 'number' && Number.isFinite(seed)) {
      headerParts.push(`seed=${seed}`);
    }
    const score = member.summary.score ?? payload?.score;
    if (typeof score === 'number' && Number.isFinite(score)) {
      headerParts.push(`score=${score.toFixed(4)}`);
    }

    records.push(`>${headerParts.join(' | ')}\n${wrapLines(formattedSeq, lineWidth)}`);
  }

  return records.length > 0 ? `${records.join('\n')}\n` : '';
}

// ============================================================================
// 2. CSV Export (Folds, Scores, Exclusions, Thresholds, Confidence Availability)
// ============================================================================

export interface ExportCsvThresholds {
  scRmsd?: number;
  plddt?: number;
  ipsae?: number;
}

export interface CollectionCsvRowInput {
  designKey?: MemberKey;
  sequenceKey?: MemberKey;
  foldKey?: MemberKey;
  referenceStructureId?: string;
  referenceDigest?: Digest;
  referenceMode?: string;
  status: ItemStatus;
  exclusionReason?: string;
  modelId?: string;
  seed?: number;
  length?: number;
  designable?: boolean | null;
  confidenceAvailable?: boolean;
  meanPlddt?: number | null;
  scRmsd?: number | null;
  scTm?: number | null;
  ptm?: number | null;
  iptm?: number | null;
  ipsae?: number | null;
  mpnnScore?: number | null;
  failedGates?: readonly string[];
}

export interface ExportCsvOptions {
  snapshotId?: string;
  outcomes?: readonly WorkItemOutcome[];
  thresholds?: ExportCsvThresholds;
  aggregationPolicy?: string;
}

const CSV_COLUMNS = [
  'snapshot_id',
  'design_key',
  'sequence_key',
  'fold_key',
  'reference_structure_id',
  'reference_digest',
  'reference_mode',
  'status',
  'exclusion_reason',
  'model_id',
  'seed',
  'length',
  'designable',
  'confidence_available',
  'mean_plddt',
  'sc_rmsd',
  'sc_tm',
  'ptm',
  'iptm',
  'ipsae',
  'mpnn_score',
  'sc_rmsd_threshold',
  'plddt_threshold',
  'ipsae_threshold',
  'aggregation_policy',
  'failed_gates',
] as const;

function escapeCsvCell(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '';
    return String(value);
  }
  const str = String(value);
  const isNumeric = /^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(str);
  const safe = !isNumeric && /^[=+\-@\t\r]/.test(str) ? `'${str}` : str;
  if (safe.includes(',') || safe.includes('"') || safe.includes('\n') || safe.includes('\r')) {
    return `"${safe.replace(/"/g, '""')}"`;
  }
  return safe;
}

function formatNum(value: number | null | undefined, digits = 4): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '';
  return Number(value.toFixed(digits)).toString();
}

function rowsFromCollection(
  input: DesignabilityCollection | FoldCollection | readonly CollectionCsvRowInput[],
): { rows: CollectionCsvRowInput[]; outcomes: readonly WorkItemOutcome[]; snapshotId?: string; aggregationPolicy?: string } {
  if (!isDesignCollection(input)) {
    return { rows: [...input], outcomes: [] };
  }

  const rows: CollectionCsvRowInput[] = [];
  for (const member of input.members) {
    if (member.memberKind === 'DesignabilityRow') {
      const row = member.payload as DesignabilityRow | undefined;
      const confAvail = member.summary.confidenceAvailable ?? row?.confidenceAvailable ?? true;
      rows.push({
        designKey: member.lineage.designKey,
        sequenceKey: member.lineage.sequenceKey,
        foldKey: member.lineage.foldKey ?? member.key,
        referenceStructureId:
          member.lineage.sourceStructure?.artifactId ?? row?.structureId ?? row?.sourceStructureId,
        referenceDigest: member.lineage.sourceStructure?.payloadDigest,
        referenceMode: row?.referenceMode ?? row?.referenceRelation,
        status: 'completed',
        modelId: input.sourceModelId,
        seed: member.provenance?.seed,
        length: member.summary.residueCount ?? row?.sequence.length,
        designable: row?.designable ?? member.summary.designable ?? null,
        confidenceAvailable: confAvail,
        meanPlddt: confAvail ? (row?.meanPlddt ?? member.summary.confidence ?? null) : null,
        scRmsd: row?.scRmsd ?? member.summary.scRmsd ?? null,
        scTm: row?.scTm ?? null,
        ptm: row?.ptm ?? null,
        iptm: row?.iptm ?? null,
        ipsae: row?.ipsae ?? member.summary.ipsae ?? null,
        mpnnScore: member.summary.score ?? null,
        failedGates: row?.failedGates,
      });
    } else if (member.memberKind === 'FoldedCandidate') {
      const fold = member.payload as FoldedCandidate | undefined;
      const confAvail = member.summary.confidenceAvailable ?? fold?.confidenceAvailable ?? true;
      rows.push({
        designKey: member.lineage.designKey,
        sequenceKey: member.lineage.sequenceKey,
        foldKey: member.lineage.foldKey ?? member.key,
        referenceStructureId:
          member.lineage.sourceStructure?.artifactId ?? fold?.sourceStructureId,
        referenceDigest: member.lineage.sourceStructure?.payloadDigest,
        status: 'completed',
        modelId: fold?.foldModelId ?? input.sourceModelId,
        seed: member.provenance?.seed,
        length: member.summary.residueCount ?? fold?.sequence.length,
        designable: member.summary.designable ?? null,
        confidenceAvailable: confAvail,
        meanPlddt: confAvail ? (fold?.meanPlddt ?? member.summary.confidence ?? null) : null,
        ptm: fold?.ptm ?? null,
        iptm: fold?.iptm ?? null,
        ipsae: fold?.ipsae ?? member.summary.ipsae ?? null,
        mpnnScore: member.summary.score ?? null,
      });
    }
  }

  return {
    rows,
    outcomes: input.outcomes,
    snapshotId: input.id,
    aggregationPolicy: input.aggregationPolicy,
  };
}

/**
 * Export fold/score rows and non-completed/excluded outcomes to RFC-4180 CSV.
 * Missing numeric values emit empty cells (never fabricated 0 or NaN).
 */
export function exportCollectionCsv(
  input: DesignabilityCollection | FoldCollection | readonly CollectionCsvRowInput[],
  options: ExportCsvOptions = {},
): string {
  const extracted = rowsFromCollection(input);
  const snapshotId = options.snapshotId ?? extracted.snapshotId ?? '';
  const aggregationPolicy = options.aggregationPolicy ?? extracted.aggregationPolicy ?? '';
  const thresholds = options.thresholds ?? {};

  const allRows: CollectionCsvRowInput[] = [...extracted.rows];
  const coveredKeys = new Set<string>();
  for (const r of allRows) {
    if (r.foldKey) coveredKeys.add(r.foldKey);
    if (r.sequenceKey && !r.foldKey) coveredKeys.add(r.sequenceKey);
  }

  const combinedOutcomes = [...extracted.outcomes, ...(options.outcomes ?? [])];
  const seenOutcomeKeys = new Set<string>();
  for (const outcome of combinedOutcomes) {
    if (outcome.status === 'completed' && coveredKeys.has(outcome.key)) continue;
    if (seenOutcomeKeys.has(outcome.key)) continue;
    seenOutcomeKeys.add(outcome.key);
    allRows.push({
      designKey: outcome.lineage?.designKey,
      sequenceKey: outcome.lineage?.sequenceKey,
      foldKey: outcome.lineage?.foldKey ?? (outcome.lineage?.sequenceKey ? undefined : outcome.key),
      referenceStructureId: outcome.lineage?.sourceStructure?.artifactId,
      referenceDigest: outcome.lineage?.sourceStructure?.payloadDigest,
      status: outcome.status,
      exclusionReason: outcome.reason,
      designable: null,
    });
  }

  const lines: string[] = [CSV_COLUMNS.join(',')];
  for (const row of allRows) {
    const cells = [
      snapshotId,
      row.designKey ?? '',
      row.sequenceKey ?? '',
      row.foldKey ?? '',
      row.referenceStructureId ?? '',
      row.referenceDigest ?? '',
      row.referenceMode ?? '',
      row.status,
      row.exclusionReason ?? '',
      row.modelId ?? '',
      row.seed !== undefined && Number.isFinite(row.seed) ? String(row.seed) : '',
      row.length !== undefined && Number.isFinite(row.length) ? String(row.length) : '',
      row.designable === true ? 'true' : row.designable === false ? 'false' : '',
      row.confidenceAvailable === true
        ? 'true'
        : row.confidenceAvailable === false
          ? 'false'
          : '',
      formatNum(row.meanPlddt, 2),
      formatNum(row.scRmsd, 3),
      formatNum(row.scTm, 4),
      formatNum(row.ptm, 4),
      formatNum(row.iptm, 4),
      formatNum(row.ipsae, 4),
      formatNum(row.mpnnScore, 4),
      formatNum(thresholds.scRmsd, 3),
      formatNum(thresholds.plddt, 2),
      formatNum(thresholds.ipsae, 4),
      aggregationPolicy,
      row.failedGates && row.failedGates.length > 0 ? row.failedGates.join('; ') : '',
    ];
    lines.push(cells.map(escapeCsvCell).join(','));
  }

  return `${lines.join('\n')}\n`;
}

// ============================================================================
// 3. Multi-MODEL PDB Export with Heterogeneous Topology Guard (RFC §7, Gate #13)
// ============================================================================

export interface MultiModelManifestEntry {
  modelNumber: number;
  key: MemberKey;
  designKey?: MemberKey;
  sequenceKey?: MemberKey;
  foldKey?: MemberKey;
  atomCount: number;
  residueCount: number;
}

export interface MultiModelPdbManifest {
  schema: 'denovo.multimodel-pdb-manifest/v1';
  modelCount: number;
  homogeneousTopology: true;
  atomNamesDigest: Digest;
  models: readonly MultiModelManifestEntry[];
}

export interface ExportMultiModelPdbOptions {
  /** Explicit member order; if omitted, uses member order as supplied. */
  selectedKeys?: readonly MemberKey[];
  /** Optional map of memberKey -> PDB text (when members are dehydrated). */
  pdbByKey?: ReadonlyMap<MemberKey, string>;
  title?: string;
}

export interface ExportMultiModelPdbResult {
  pdbText: string;
  manifest: MultiModelPdbManifest;
}

function selectOrderedMembers<T>(
  members: readonly CollectionMember<T>[],
  selectedKeys?: readonly MemberKey[],
): CollectionMember<T>[] {
  if (!selectedKeys) return [...members];
  const byKey = new Map<MemberKey, CollectionMember<T>>();
  for (const m of members) byKey.set(m.key, m);
  const out: CollectionMember<T>[] = [];
  for (const key of selectedKeys) {
    const hit = byKey.get(key);
    if (hit) out.push(hit);
  }
  return out;
}

/** Extract atom topology signature from PDB coordinate records. */
export function inspectPdbTopologySignature(
  memberKey: MemberKey,
  pdbText: string,
): TopologySignature {
  const lines = pdbText.split(/\r?\n/);
  const atomDescriptors: string[] = [];
  const residueSet = new Set<string>();
  const chainAtomCountsMap = new Map<string, number>();

  for (const line of lines) {
    if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) continue;
    const record = line.slice(0, 6).trim();
    const atomName = line.slice(12, 16).trim();
    const resName = line.slice(17, 20).trim();
    const chainId = line.slice(21, 22).trim() || '_';
    const resSeq = line.slice(22, 27).trim();
    atomDescriptors.push(`${record}:${chainId}:${resSeq}:${resName}:${atomName}`);
    residueSet.add(`${chainId}:${resSeq}`);
    chainAtomCountsMap.set(chainId, (chainAtomCountsMap.get(chainId) ?? 0) + 1);
  }

  const chainAtomCounts = [...chainAtomCountsMap.entries()].map(([ch, n]) => `${ch}:${n}`);
  return {
    memberKey,
    atomCount: atomDescriptors.length,
    residueCount: residueSet.size,
    chainAtomCounts,
    atomNamesDigest: sha256Hex(atomDescriptors.join('|')),
  };
}

function stripOuterModelRecords(pdbText: string): string {
  const lines = pdbText.split(/\r?\n/);
  const kept: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (
      line.startsWith('MODEL ') ||
      trimmed === 'MODEL' ||
      trimmed === 'ENDMDL' ||
      trimmed === 'END'
    ) {
      continue;
    }
    kept.push(line);
  }
  return kept.join('\n');
}

/**
 * Export selected structure/fold members into a multi-MODEL PDB with an explicit
 * model-to-key manifest.
 *
 * REFUSES to silently flatten heterogeneous atom topologies or variable-length designs
 * into a misleading trajectory — throws `HeterogeneousTopologyError` instead (RFC Gate #13).
 */
export function exportCollectionMultiModelPdb(
  input:
    | StructureCollection
    | FoldCollection
    | readonly CollectionMember<StructureArtifact | FoldedCandidate>[],
  options: ExportMultiModelPdbOptions = {},
): ExportMultiModelPdbResult {
  const rawMembers = isDesignCollection(input) ? input.members : input;
  const members = selectOrderedMembers(rawMembers, options.selectedKeys);

  if (members.length === 0) {
    throw new RangeError('Cannot export multi-MODEL PDB from zero selected members.');
  }

  const resolvedRecords: Array<{
    member: CollectionMember<StructureArtifact | FoldedCandidate>;
    pdbText: string;
    signature: TopologySignature;
  }> = [];

  for (const member of members) {
    const payload = member.payload as { pdbText?: string } | undefined;
    const pdbText = options.pdbByKey?.get(member.key) ?? payload?.pdbText ?? '';
    if (!pdbText.trim()) {
      throw new Error(`Member "${member.key}" has no PDB text available for multi-MODEL export.`);
    }
    const signature = inspectPdbTopologySignature(member.key, pdbText);
    resolvedRecords.push({ member, pdbText, signature });
  }

  // Strictly enforce homogeneous atom topology across all selected models
  const baseline = resolvedRecords[0].signature;
  const mismatched = resolvedRecords.filter(
    (r) =>
      r.signature.atomCount !== baseline.atomCount ||
      r.signature.residueCount !== baseline.residueCount ||
      r.signature.atomNamesDigest !== baseline.atomNamesDigest,
  );

  if (mismatched.length > 0) {
    const allSignatures = resolvedRecords.map((r) => r.signature);
    const details = allSignatures
      .map((s) => `${s.memberKey} (${s.residueCount} res, ${s.atomCount} atoms)`)
      .join(', ');
    throw new HeterogeneousTopologyError(
      `Refusing to flatten heterogeneous topologies into a multi-MODEL PDB: ${details}. Export separate files via buildCollectionBundleEntries instead.`,
      allSignatures,
    );
  }

  const models: MultiModelManifestEntry[] = [];
  const chunks: string[] = [];

  if (options.title) {
    chunks.push(`TITLE     ${options.title}`);
  }
  chunks.push(`REMARK   1 DENOVO.RUN MULTI-MODEL EXPORT (${resolvedRecords.length} MODELS)`);

  resolvedRecords.forEach(({ member, pdbText, signature }, idx) => {
    const modelNumber = idx + 1;
    models.push({
      modelNumber,
      key: member.key,
      ...(member.lineage.designKey ? { designKey: member.lineage.designKey } : {}),
      ...(member.lineage.sequenceKey ? { sequenceKey: member.lineage.sequenceKey } : {}),
      ...(member.lineage.foldKey ? { foldKey: member.lineage.foldKey } : {}),
      atomCount: signature.atomCount,
      residueCount: signature.residueCount,
    });
    chunks.push(
      `REMARK   2 MODEL ${modelNumber}: KEY=${member.key}` +
        (member.lineage.designKey ? ` DESIGN=${member.lineage.designKey}` : '') +
        (member.lineage.sequenceKey ? ` SEQ=${member.lineage.sequenceKey}` : ''),
    );
  });

  resolvedRecords.forEach(({ pdbText }, idx) => {
    const modelNumber = idx + 1;
    chunks.push(`MODEL     ${String(modelNumber).padStart(4, ' ')}`);
    const body = stripOuterModelRecords(pdbText);
    if (body) chunks.push(body);
    chunks.push('ENDMDL');
  });

  chunks.push('END');

  const manifest: MultiModelPdbManifest = {
    schema: 'denovo.multimodel-pdb-manifest/v1',
    modelCount: models.length,
    homogeneousTopology: true,
    atomNamesDigest: baseline.atomNamesDigest,
    models,
  };

  return {
    pdbText: `${chunks.join('\n')}\n`,
    manifest,
  };
}

// ============================================================================
// 4. Streaming Snapshot Bundle Entry Generator (RFC §7, Gate #13)
// ============================================================================

export interface CollectionBundleEntry {
  path: string;
  mediaType: string;
  content: string | Uint8Array;
  byteLength: number;
  digest: Digest;
  memberKey?: MemberKey;
}

export interface CollectionBundleManifestEntry {
  path: string;
  mediaType: string;
  byteLength: number;
  digest: Digest;
  memberKey?: MemberKey;
  lineage?: ItemLineage;
  format?: 'pdb' | 'mmcif' | 'fasta' | 'csv' | 'json';
}

export interface CollectionBundleManifest {
  schema: 'denovo.export-manifest/v1';
  snapshotId: string;
  familyKey: string;
  createdAt: string;
  heterogeneousTopology: boolean;
  counts: {
    structures: number;
    sequences: number;
    folds: number;
    scores: number;
    residueTracks: number;
    excluded: number;
    failed: number;
  };
  thresholds?: ExportCsvThresholds;
  campaignSummary?: CampaignSummary;
  entries: readonly CollectionBundleManifestEntry[];
}

export interface CollectionExportSnapshot {
  snapshotId: string;
  familyKey: string;
  createdAt?: string;
  structures?: StructureCollection;
  sequences?: SequenceCollection;
  folds?: FoldCollection;
  scores?: DesignabilityCollection;
  residueTracks?: ResidueTrackCollection;
  outcomes?: readonly WorkItemOutcome[];
  thresholds?: ExportCsvThresholds;
  campaignSummary?: CampaignSummary;
}

export interface BuildBundleEntriesOptions {
  /**
   * Optional lazy payload loader (e.g. `(m) => store.hydratePayload(m)`).
   * Called strictly one member at a time so the exporter never holds all PDBs
   * in memory simultaneously (RFC §7, Gate #11 & #13).
   */
  loadPayload?: <T>(member: CollectionMember<T>) => Promise<T | null> | T | null;
  /** Prefer mmCIF over PDB when a structure carries both `cifText` and `pdbText` (default true). */
  preferMmcif?: boolean;
  /** Also emit `.pdb` alongside `.cif` when both exist (default false). */
  emitBothPdbAndCif?: boolean;
}

const encoder = new TextEncoder();

function sanitizeFileSegment(key: string): string {
  return key.replace(/[^a-zA-Z0-9._-]/g, '_');
}

function makeTextEntry(
  path: string,
  mediaType: string,
  content: string,
  memberKey?: MemberKey,
): CollectionBundleEntry {
  const bytes = encoder.encode(content);
  return {
    path,
    mediaType,
    content,
    byteLength: bytes.byteLength,
    digest: sha256Hex(bytes),
    ...(memberKey ? { memberKey } : {}),
  };
}

function exportResidueTrackCsv(
  member: CollectionMember<ResidueTrackArtifact>,
  track: ResidueTrackArtifact,
): string {
  const lines = ['member_key,residue_index,residue_id,residue_aa,quantity,unit,value'];
  for (let i = 0; i < track.values.length; i++) {
    const val = track.values[i];
    const resId = track.residueIds?.[i] ?? String(i + 1);
    const aa = track.residues[i] ?? '';
    lines.push(
      [
        member.key,
        String(i),
        resId,
        aa,
        track.quantity,
        track.unit ?? '',
        Number.isFinite(val) ? String(val) : '',
      ]
        .map(escapeCsvCell)
        .join(','),
    );
  }
  return `${lines.join('\n')}\n`;
}

/**
 * Async generator yielding one export bundle entry at a time over a frozen collection snapshot.
 *
 * - Never assembles all PDB/mmCIF strings simultaneously in RAM.
 * - Keeps heterogeneous variable-length structures in separate files (preserving mmCIF when present).
 * - Emits `sequences.fasta`, `scores.csv`, optional residue-track CSVs with units,
 *   and a final versioned `manifest.json` indexing every emitted entry with SHA-256 digests.
 */
export async function* buildCollectionBundleEntries(
  snapshot: CollectionExportSnapshot,
  options: BuildBundleEntriesOptions = {},
): AsyncGenerator<CollectionBundleEntry, CollectionBundleManifest, void> {
  const preferMmcif = options.preferMmcif !== false;
  const emitBoth = options.emitBothPdbAndCif === true;
  const manifestEntries: CollectionBundleManifestEntry[] = [];
  const observedTopologyDigests = new Set<string>();

  const loadMemberPayload = async <T>(member: CollectionMember<T>): Promise<T | null> => {
    if (options.loadPayload) {
      return await options.loadPayload(member);
    }
    return member.payload ?? null;
  };

  // 1. Structures (one at a time, preserving mmCIF for all-atom/holo context)
  const structureMembers = snapshot.structures?.members ?? [];
  for (const member of structureMembers) {
    const struct = await loadMemberPayload(member);
    if (!struct) continue;
    const safeName = sanitizeFileSegment(member.key);

    if (struct.pdbText) {
      const sig = inspectPdbTopologySignature(member.key, struct.pdbText);
      observedTopologyDigests.add(`${sig.residueCount}:${sig.atomCount}:${sig.atomNamesDigest}`);
    } else {
      observedTopologyDigests.add(`len:${struct.length}`);
    }

    const hasCif = typeof struct.cifText === 'string' && struct.cifText.trim().length > 0;
    const hasPdb = typeof struct.pdbText === 'string' && struct.pdbText.trim().length > 0;

    if (hasCif && (preferMmcif || emitBoth || !hasPdb)) {
      const entry = makeTextEntry(
        `structures/${safeName}.cif`,
        'chemical/x-mmcif',
        struct.cifText!,
        member.key,
      );
      manifestEntries.push({
        path: entry.path,
        mediaType: entry.mediaType,
        byteLength: entry.byteLength,
        digest: entry.digest,
        memberKey: member.key,
        lineage: member.lineage,
        format: 'mmcif',
      });
      yield entry;
    }

    if (hasPdb && (!hasCif || !preferMmcif || emitBoth)) {
      const entry = makeTextEntry(
        `structures/${safeName}.pdb`,
        'chemical/x-pdb',
        struct.pdbText,
        member.key,
      );
      manifestEntries.push({
        path: entry.path,
        mediaType: entry.mediaType,
        byteLength: entry.byteLength,
        digest: entry.digest,
        memberKey: member.key,
        lineage: member.lineage,
        format: 'pdb',
      });
      yield entry;
    }
  }

  // 2. Folds (one at a time)
  const foldMembers = snapshot.folds?.members ?? [];
  for (const member of foldMembers) {
    const fold = await loadMemberPayload(member);
    if (!fold || !fold.pdbText) continue;
    const safeName = sanitizeFileSegment(member.key);
    const sig = inspectPdbTopologySignature(member.key, fold.pdbText);
    observedTopologyDigests.add(`${sig.residueCount}:${sig.atomCount}:${sig.atomNamesDigest}`);

    const entry = makeTextEntry(
      `folds/${safeName}.pdb`,
      'chemical/x-pdb',
      fold.pdbText,
      member.key,
    );
    manifestEntries.push({
      path: entry.path,
      mediaType: entry.mediaType,
      byteLength: entry.byteLength,
      digest: entry.digest,
      memberKey: member.key,
      lineage: member.lineage,
      format: 'pdb',
    });
    yield entry;
  }

  // 3. Sequences FASTA
  if (snapshot.sequences && snapshot.sequences.members.length > 0) {
    // Hydrate sequence payloads lazily if needed (sequences are tiny strings)
    const hydratedSeqMembers: CollectionMember<SequenceCandidate>[] = [];
    for (const m of snapshot.sequences.members) {
      const payload = await loadMemberPayload(m);
      hydratedSeqMembers.push(payload ? { ...m, payload } : m);
    }
    const fastaText = exportCollectionFasta({
      ...snapshot.sequences,
      members: hydratedSeqMembers,
    });
    if (fastaText.length > 0) {
      const entry = makeTextEntry('sequences.fasta', 'text/plain;charset=utf-8', fastaText);
      manifestEntries.push({
        path: entry.path,
        mediaType: entry.mediaType,
        byteLength: entry.byteLength,
        digest: entry.digest,
        format: 'fasta',
      });
      yield entry;
    }
  }

  // 4. Scores / Folds CSV
  const csvSource = snapshot.scores ?? snapshot.folds;
  const extraOutcomes = snapshot.outcomes ?? [];
  if (csvSource || extraOutcomes.length > 0) {
    const csvText = csvSource
      ? exportCollectionCsv(csvSource, {
          snapshotId: snapshot.snapshotId,
          outcomes: extraOutcomes,
          thresholds: snapshot.thresholds,
          aggregationPolicy: snapshot.campaignSummary?.aggregationPolicy ?? csvSource.aggregationPolicy,
        })
      : exportCollectionCsv([], {
          snapshotId: snapshot.snapshotId,
          outcomes: extraOutcomes,
          thresholds: snapshot.thresholds,
          aggregationPolicy: snapshot.campaignSummary?.aggregationPolicy,
        });
    const entry = makeTextEntry('scores.csv', 'text/csv;charset=utf-8', csvText);
    manifestEntries.push({
      path: entry.path,
      mediaType: entry.mediaType,
      byteLength: entry.byteLength,
      digest: entry.digest,
      format: 'csv',
    });
    yield entry;
  }

  // 5. Residue tracks (one at a time, preserving units)
  const trackMembers = snapshot.residueTracks?.members ?? [];
  for (const member of trackMembers) {
    const track = await loadMemberPayload(member);
    if (!track) continue;
    const safeName = sanitizeFileSegment(member.key);
    const csvText = exportResidueTrackCsv(member, track);
    const entry = makeTextEntry(
      `tracks/${safeName}.csv`,
      'text/csv;charset=utf-8',
      csvText,
      member.key,
    );
    manifestEntries.push({
      path: entry.path,
      mediaType: entry.mediaType,
      byteLength: entry.byteLength,
      digest: entry.digest,
      memberKey: member.key,
      lineage: member.lineage,
      format: 'csv',
    });
    yield entry;
  }

  // 6. Versioned manifest.json
  const allOutcomes = [
    ...(snapshot.structures?.outcomes ?? []),
    ...(snapshot.sequences?.outcomes ?? []),
    ...(snapshot.folds?.outcomes ?? []),
    ...(snapshot.scores?.outcomes ?? []),
    ...extraOutcomes,
  ];
  const excludedCount = allOutcomes.filter((o) => o.status === 'excluded').length;
  const failedCount = allOutcomes.filter((o) => o.status === 'failed').length;

  const manifest: CollectionBundleManifest = {
    schema: 'denovo.export-manifest/v1',
    snapshotId: snapshot.snapshotId,
    familyKey: snapshot.familyKey,
    createdAt: snapshot.createdAt ?? new Date().toISOString(),
    heterogeneousTopology: observedTopologyDigests.size > 1,
    counts: {
      structures: structureMembers.length,
      sequences: snapshot.sequences?.members.length ?? 0,
      folds: foldMembers.length,
      scores: snapshot.scores?.members.length ?? 0,
      residueTracks: trackMembers.length,
      excluded: excludedCount,
      failed: failedCount,
    },
    ...(snapshot.thresholds ? { thresholds: snapshot.thresholds } : {}),
    ...(snapshot.campaignSummary ? { campaignSummary: snapshot.campaignSummary } : {}),
    entries: manifestEntries,
  };

  const manifestJson = `${JSON.stringify(manifest, null, 2)}\n`;
  const manifestEntry = makeTextEntry(
    'manifest.json',
    'application/json;charset=utf-8',
    manifestJson,
  );
  yield manifestEntry;

  return manifest;
}
