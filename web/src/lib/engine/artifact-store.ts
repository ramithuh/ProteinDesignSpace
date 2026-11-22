// Wave-6A Multi-Design & Batching Core: Durable IndexedDB BlobRef Payload Store,
// Bounded Hydrated LRU + Pin Ownership, Quota Resilience, and v2 Collection Checkpoints.
//
// Implements RFC §4.4, §4.5, Gates #3, #4, #11, #12.
// Works seamlessly in browsers (IndexedDB) and falls back cleanly to in-memory storage
// in Node/Vitest or Safari Private Mode / denied storage (explicitly flagged `isDurable: false`).

import {
  computeCollectionSnapshotId,
  sha256Hex,
  type BlobRef,
  type CollectionMember,
  type DesignCollection,
  type Digest,
  type MemberKey,
  type WorkItemOutcome,
} from './collections';

export const ARTIFACT_DB_NAME = 'denovo-run-artifacts-v2';
export const ARTIFACT_DB_VERSION = 1;
export const PAYLOAD_MEDIA_TYPE = 'application/x-denovo-payload+bin;v=1';

const STORE_PAYLOADS = 'payloads';
const STORE_ITEMS = 'items';
const STORE_COLLECTION_MEMBERS = 'collectionMembers';
const STORE_RUNS = 'runs';

const MAGIC_BYTES = [0x44, 0x4e, 0x56, 0x31] as const; // 'DNV1'
const CODEC_VERSION = 1;

// ============================================================================
// Typed Errors (RFC Gates #11 & #12)
// ============================================================================

export class StorageQuotaError extends Error {
  readonly code = 'QUOTA_EXCEEDED' as const;
  readonly attemptedBytes: number;
  readonly causeError?: unknown;

  constructor(message: string, attemptedBytes = 0, causeError?: unknown) {
    super(message);
    this.name = 'StorageQuotaError';
    this.attemptedBytes = attemptedBytes;
    this.causeError = causeError;
  }
}

export class PinBudgetExceededError extends Error {
  readonly code = 'PIN_BUDGET_EXCEEDED' as const;
  readonly pinnedBytes: number;
  readonly maxTotalBytes: number;

  constructor(message: string, pinnedBytes: number, maxTotalBytes: number) {
    super(message);
    this.name = 'PinBudgetExceededError';
    this.pinnedBytes = pinnedBytes;
    this.maxTotalBytes = maxTotalBytes;
  }
}

export function isQuotaExceededError(err: unknown): boolean {
  if (err instanceof StorageQuotaError) return true;
  if (!err || typeof err !== 'object') return false;
  const e = err as { name?: unknown; code?: unknown; message?: unknown };
  if (e.name === 'QuotaExceededError' || e.name === 'NS_ERROR_DOM_QUOTA_REACHED') return true;
  if (e.code === 22 || e.code === 1014) return true;
  if (typeof e.message === 'string' && /quota/i.test(e.message)) return true;
  return false;
}

// ============================================================================
// Typed-Array-Preserving Binary Payload Codec (No JSON Stringification of Arrays)
// ============================================================================

type SupportedTypedArrayCtorName =
  | 'Float32Array'
  | 'Float64Array'
  | 'Int8Array'
  | 'Uint8Array'
  | 'Uint8ClampedArray'
  | 'Int16Array'
  | 'Uint16Array'
  | 'Int32Array'
  | 'Uint32Array';

interface TypedArrayMarker {
  __denovoTypedArray: SupportedTypedArrayCtorName;
  chunkIndex: number;
  length: number;
  byteLength: number;
}

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

function constructTypedArray(
  ctorName: SupportedTypedArrayCtorName,
  buffer: ArrayBuffer,
): ArrayBufferView {
  switch (ctorName) {
    case 'Float32Array':
      return new Float32Array(buffer);
    case 'Float64Array':
      return new Float64Array(buffer);
    case 'Int8Array':
      return new Int8Array(buffer);
    case 'Uint8Array':
      return new Uint8Array(buffer);
    case 'Uint8ClampedArray':
      return new Uint8ClampedArray(buffer);
    case 'Int16Array':
      return new Int16Array(buffer);
    case 'Uint16Array':
      return new Uint16Array(buffer);
    case 'Int32Array':
      return new Int32Array(buffer);
    case 'Uint32Array':
      return new Uint32Array(buffer);
  }
}

function isSupportedTypedArrayName(name: string): name is SupportedTypedArrayCtorName {
  return (
    name === 'Float32Array' ||
    name === 'Float64Array' ||
    name === 'Int8Array' ||
    name === 'Uint8Array' ||
    name === 'Uint8ClampedArray' ||
    name === 'Int16Array' ||
    name === 'Uint16Array' ||
    name === 'Int32Array' ||
    name === 'Uint32Array'
  );
}

/**
 * Encode an arbitrary payload (including nested Float32Array/Uint8Array/etc.) into
 * a compact, versioned binary frame without JSON-stringifying typed array numbers.
 */
export function encodePayloadBytes(payload: unknown): Uint8Array {
  const chunks: Uint8Array[] = [];

  const dehydrate = (value: unknown): unknown => {
    if (value === null || value === undefined) return value;
    if (typeof value !== 'object') return value;

    if (ArrayBuffer.isView(value)) {
      const ctorName = value.constructor.name;
      if (!isSupportedTypedArrayName(ctorName)) {
        throw new Error(`Unsupported ArrayBufferView type: ${ctorName}`);
      }
      const chunkIndex = chunks.length;
      const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
      chunks.push(bytes);
      const marker: TypedArrayMarker = {
        __denovoTypedArray: ctorName,
        chunkIndex,
        length: 'length' in value && typeof value.length === 'number' ? value.length : value.byteLength,
        byteLength: value.byteLength,
      };
      return marker;
    }

    if (Array.isArray(value)) {
      return value.map((item) => dehydrate(item));
    }

    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (v !== undefined) {
        out[k] = dehydrate(v);
      }
    }
    return out;
  };

  const dehydrated = dehydrate(payload);
  const headerJson = JSON.stringify({
    root: dehydrated,
    chunkLengths: chunks.map((c) => c.byteLength),
  });
  const jsonBytes = textEncoder.encode(headerJson);

  // Align each binary chunk to 8-byte boundaries for direct TypedArray alignment
  const headerSize = 12 + jsonBytes.byteLength;
  let totalBytes = (headerSize + 7) & ~7;
  const chunkOffsets: number[] = [];
  for (const chunk of chunks) {
    chunkOffsets.push(totalBytes);
    totalBytes = (totalBytes + chunk.byteLength + 7) & ~7;
  }

  const out = new Uint8Array(totalBytes);
  out[0] = MAGIC_BYTES[0];
  out[1] = MAGIC_BYTES[1];
  out[2] = MAGIC_BYTES[2];
  out[3] = MAGIC_BYTES[3];
  const view = new DataView(out.buffer);
  view.setUint32(4, CODEC_VERSION, false);
  view.setUint32(8, jsonBytes.byteLength, false);
  out.set(jsonBytes, 12);

  for (let i = 0; i < chunks.length; i++) {
    out.set(chunks[i], chunkOffsets[i]);
  }
  return out;
}

/** Decode a versioned binary payload frame back into its original objects & typed arrays. */
export function decodePayloadBytes<T = unknown>(bytes: Uint8Array): T {
  if (
    bytes.byteLength < 12 ||
    bytes[0] !== MAGIC_BYTES[0] ||
    bytes[1] !== MAGIC_BYTES[1] ||
    bytes[2] !== MAGIC_BYTES[2] ||
    bytes[3] !== MAGIC_BYTES[3]
  ) {
    throw new Error('Invalid denovo binary payload header');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, false);
  if (version !== CODEC_VERSION) {
    throw new Error(`Unsupported denovo binary payload version: ${version}`);
  }
  const jsonLen = view.getUint32(8, false);
  if (12 + jsonLen > bytes.byteLength) {
    throw new Error('Truncated denovo binary payload metadata');
  }
  const jsonText = textDecoder.decode(bytes.subarray(12, 12 + jsonLen));
  const parsed = JSON.parse(jsonText) as { root: unknown; chunkLengths: number[] };

  let cursor = (12 + jsonLen + 7) & ~7;
  const buffers: ArrayBuffer[] = [];
  for (const len of parsed.chunkLengths ?? []) {
    if (cursor + len > bytes.byteLength) {
      throw new Error('Truncated denovo binary payload typed-array chunk');
    }
    const copy = new Uint8Array(len);
    copy.set(bytes.subarray(cursor, cursor + len));
    buffers.push(copy.buffer);
    cursor = (cursor + len + 7) & ~7;
  }

  const rehydrate = (value: unknown): unknown => {
    if (value === null || value === undefined || typeof value !== 'object') return value;
    if (Array.isArray(value)) return value.map((item) => rehydrate(item));
    const rec = value as Record<string, unknown>;
    if (
      typeof rec.__denovoTypedArray === 'string' &&
      isSupportedTypedArrayName(rec.__denovoTypedArray) &&
      typeof rec.chunkIndex === 'number'
    ) {
      const buf = buffers[rec.chunkIndex];
      if (!buf) throw new Error(`Missing typed-array chunk ${rec.chunkIndex}`);
      return constructTypedArray(rec.__denovoTypedArray, buf);
    }
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(rec)) {
      out[k] = rehydrate(v);
    }
    return out;
  };

  return rehydrate(parsed.root) as T;
}

// ============================================================================
// Backing-Buffer-Aware Memory Estimation (Aligned with result-memo.ts)
// ============================================================================

/**
 * Estimate retained RAM bytes for a hydrated payload, charging each unique
 * backing ArrayBuffer once (including the whole buffer retained by a view).
 */
export function estimatePayloadMemoryBytes(value: unknown, seen = new Set<unknown>()): number {
  if (typeof value === 'string') {
    if (value.length > 256) {
      if (seen.has(value)) return 0;
      seen.add(value);
    }
    return 16 + 2 * value.length;
  }
  if (value === null || typeof value !== 'object') return 8;
  if (seen.has(value)) return 0;
  seen.add(value);
  if (ArrayBuffer.isView(value)) {
    return 64 + estimatePayloadMemoryBytes(value.buffer, seen);
  }
  if (
    value instanceof ArrayBuffer ||
    (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer)
  ) {
    return 32 + value.byteLength;
  }
  let bytes = 64;
  if (Array.isArray(value)) {
    for (const item of value) bytes += 8 + estimatePayloadMemoryBytes(item, seen);
  } else {
    for (const [key, item] of Object.entries(value)) {
      bytes += 16 + 2 * key.length + estimatePayloadMemoryBytes(item, seen);
    }
  }
  return bytes;
}

// ============================================================================
// v2 Checkpoint & Atomic Commit Contracts
// ============================================================================

export interface CollectionCheckpointRecord {
  schema: 'denovo.checkpoint/v2';
  runId: string;
  nodeId: string;
  familyKey: string;
  revision: number;
  updatedAt: string;
  /** Dehydrated manifest (members carry BlobRef without heavy inline typed arrays). */
  collection: DesignCollection<unknown>;
  committedKeys: readonly MemberKey[];
  pendingKeys: readonly MemberKey[];
  cancelled: boolean;
}

export interface StoredItemIndexRecord {
  memberKey: MemberKey;
  familyKey: string;
  digest: Digest;
  byteLength: number;
  mediaType: string;
  committedAt: string;
}

export interface AtomicItemCommit {
  digest: Digest;
  bytes: Uint8Array;
  mediaType: string;
  itemRecord: StoredItemIndexRecord;
  membershipKey: [string, MemberKey];
  ordinal: number;
  checkpoint: CollectionCheckpointRecord;
}

export interface BlobStorageBackend {
  readonly isDurable: boolean;
  putBlobBytes(digest: Digest, bytes: Uint8Array, mediaType: string): Promise<void>;
  getBlobBytes(digest: Digest): Promise<{ bytes: Uint8Array; mediaType: string } | null>;
  hasBlobBytes(digest: Digest): Promise<boolean>;
  deleteBlobBytes(digest: Digest): Promise<void>;
  listBlobDigests(): Promise<Digest[]>;
  putCheckpoint(runId: string, record: CollectionCheckpointRecord): Promise<void>;
  getCheckpoint(runId: string): Promise<CollectionCheckpointRecord | null>;
  deleteCheckpoint(runId: string): Promise<void>;
  listCheckpointIds(): Promise<string[]>;
  commitItemAtomic(commit: AtomicItemCommit): Promise<void>;
}

// ============================================================================
// In-Memory Fallback Backend (Tests & Safari Private Mode / Denied Storage)
// ============================================================================

export interface MemoryBackendOptions {
  /** Optional simulated storage quota in bytes (for deterministic Gate #12 testing). */
  maxQuotaBytes?: number;
  /** Explicitly false by default (RFC §4.4: in-memory fallback must never be labelled durable). */
  isDurable?: boolean;
}

export class MemoryArtifactBackend implements BlobStorageBackend {
  readonly isDurable: boolean;
  private readonly maxQuotaBytes: number;
  private readonly payloads = new Map<Digest, { bytes: Uint8Array; mediaType: string }>();
  private readonly items = new Map<MemberKey, StoredItemIndexRecord>();
  private readonly memberships = new Map<string, number>();
  private readonly checkpoints = new Map<string, CollectionCheckpointRecord>();
  private usedPayloadBytes = 0;

  constructor(options: MemoryBackendOptions = {}) {
    this.isDurable = options.isDurable ?? false;
    this.maxQuotaBytes = options.maxQuotaBytes ?? Number.POSITIVE_INFINITY;
  }

  private assertQuota(additionalBytes: number, replacingBytes = 0): void {
    const projected = this.usedPayloadBytes - replacingBytes + additionalBytes;
    if (projected > this.maxQuotaBytes) {
      throw new StorageQuotaError(
        `Artifact storage quota exceeded (${projected} > ${this.maxQuotaBytes} bytes)`,
        additionalBytes,
      );
    }
  }

  async putBlobBytes(digest: Digest, bytes: Uint8Array, mediaType: string): Promise<void> {
    const existing = this.payloads.get(digest);
    const existingLen = existing?.bytes.byteLength ?? 0;
    this.assertQuota(bytes.byteLength, existingLen);
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    this.payloads.set(digest, { bytes: copy, mediaType });
    this.usedPayloadBytes = this.usedPayloadBytes - existingLen + copy.byteLength;
  }

  async getBlobBytes(digest: Digest): Promise<{ bytes: Uint8Array; mediaType: string } | null> {
    const hit = this.payloads.get(digest);
    if (!hit) return null;
    const copy = new Uint8Array(hit.bytes.byteLength);
    copy.set(hit.bytes);
    return { bytes: copy, mediaType: hit.mediaType };
  }

  async hasBlobBytes(digest: Digest): Promise<boolean> {
    return this.payloads.has(digest);
  }

  async deleteBlobBytes(digest: Digest): Promise<void> {
    const existing = this.payloads.get(digest);
    if (existing) {
      this.usedPayloadBytes -= existing.bytes.byteLength;
      this.payloads.delete(digest);
    }
  }

  async listBlobDigests(): Promise<Digest[]> {
    return [...this.payloads.keys()];
  }

  async putCheckpoint(runId: string, record: CollectionCheckpointRecord): Promise<void> {
    const serializedLen = textEncoder.encode(JSON.stringify(record)).byteLength;
    this.assertQuota(serializedLen, 0);
    this.checkpoints.set(runId, structuredClone(record));
  }

  async getCheckpoint(runId: string): Promise<CollectionCheckpointRecord | null> {
    const hit = this.checkpoints.get(runId);
    return hit ? structuredClone(hit) : null;
  }

  async deleteCheckpoint(runId: string): Promise<void> {
    this.checkpoints.delete(runId);
  }

  async listCheckpointIds(): Promise<string[]> {
    return [...this.checkpoints.keys()];
  }

  async commitItemAtomic(commit: AtomicItemCommit): Promise<void> {
    const existing = this.payloads.get(commit.digest);
    const existingLen = existing?.bytes.byteLength ?? 0;
    const checkpointBytes = textEncoder.encode(JSON.stringify(commit.checkpoint)).byteLength;
    // Validate quota BEFORE mutating any store so failed writes stay atomic
    this.assertQuota(commit.bytes.byteLength + checkpointBytes, existingLen);

    const copy = new Uint8Array(commit.bytes.byteLength);
    copy.set(commit.bytes);
    this.payloads.set(commit.digest, { bytes: copy, mediaType: commit.mediaType });
    this.usedPayloadBytes = this.usedPayloadBytes - existingLen + copy.byteLength;
    this.items.set(commit.itemRecord.memberKey, { ...commit.itemRecord });
    this.memberships.set(`${commit.membershipKey[0]}::${commit.membershipKey[1]}`, commit.ordinal);
    this.checkpoints.set(commit.checkpoint.runId, structuredClone(commit.checkpoint));
  }

  getUsedBytes(): number {
    return this.usedPayloadBytes;
  }
}

// ============================================================================
// Browser IndexedDB v2 Backend (RFC §4.4)
// ============================================================================

const openArtifactDbs = new Set<IDBDatabase>();

export function closeAllOpenArtifactDbs(): void {
  for (const db of openArtifactDbs) {
    try {
      db.close();
    } catch {
      /* ignore */
    }
  }
  openArtifactDbs.clear();
}

export async function clearArtifactDatabase(dbName = ARTIFACT_DB_NAME): Promise<void> {
  closeAllOpenArtifactDbs();
  if (typeof indexedDB === 'undefined' || typeof indexedDB.deleteDatabase !== 'function') {
    return;
  }
  await new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(finish, 2000);
    try {
      const req = indexedDB.deleteDatabase(dbName);
      if (!req || typeof req !== 'object') {
        finish();
        return;
      }
      req.onsuccess = finish;
      req.onerror = finish;
      // Do not settle immediately onblocked; wait for onversionchange -> db.close() or timeout
    } catch {
      finish();
    }
  });
}

class IndexedDbArtifactBackend implements BlobStorageBackend {
  readonly isDurable = true;

  constructor(private readonly dbName = ARTIFACT_DB_NAME) {}

  private closeDb(db: IDBDatabase): void {
    openArtifactDbs.delete(db);
    try {
      db.close();
    } catch {
      /* ignore */
    }
  }

  private openDb(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      try {
        const req = indexedDB.open(this.dbName, ARTIFACT_DB_VERSION);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(STORE_PAYLOADS)) {
            db.createObjectStore(STORE_PAYLOADS);
          }
          if (!db.objectStoreNames.contains(STORE_ITEMS)) {
            db.createObjectStore(STORE_ITEMS);
          }
          if (!db.objectStoreNames.contains(STORE_COLLECTION_MEMBERS)) {
            db.createObjectStore(STORE_COLLECTION_MEMBERS);
          }
          if (!db.objectStoreNames.contains(STORE_RUNS)) {
            db.createObjectStore(STORE_RUNS);
          }
        };
        req.onsuccess = () => {
          const db = req.result;
          openArtifactDbs.add(db);
          db.onversionchange = () => {
            openArtifactDbs.delete(db);
            try {
              db.close();
            } catch {
              /* ignore */
            }
          };
          db.onclose = () => {
            openArtifactDbs.delete(db);
          };
          resolve(db);
        };
        req.onerror = () => reject(req.error ?? new Error('Failed to open artifact IndexedDB'));
      } catch (err) {
        reject(err);
      }
    });
  }

  private wrapError(err: unknown, attemptedBytes = 0): Error {
    if (isQuotaExceededError(err)) {
      return new StorageQuotaError(
        err instanceof Error ? err.message : 'IndexedDB storage quota exceeded',
        attemptedBytes,
        err,
      );
    }
    return err instanceof Error ? err : new Error(String(err));
  }

  async putBlobBytes(digest: Digest, bytes: Uint8Array, mediaType: string): Promise<void> {
    const db = await this.openDb();
    try {
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_PAYLOADS, 'readwrite');
        tx.objectStore(STORE_PAYLOADS).put({ bytes, mediaType }, digest);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(this.wrapError(tx.error, bytes.byteLength));
        tx.onabort = () => reject(this.wrapError(tx.error, bytes.byteLength));
      });
    } finally {
      this.closeDb(db);
    }
  }

  async getBlobBytes(digest: Digest): Promise<{ bytes: Uint8Array; mediaType: string } | null> {
    const db = await this.openDb();
    try {
      return await new Promise<{ bytes: Uint8Array; mediaType: string } | null>((resolve, reject) => {
        const tx = db.transaction(STORE_PAYLOADS, 'readonly');
        const req = tx.objectStore(STORE_PAYLOADS).get(digest);
        req.onsuccess = () => {
          const val = req.result as { bytes?: Uint8Array; mediaType?: string } | undefined;
          if (!val || !(val.bytes instanceof Uint8Array)) {
            resolve(null);
            return;
          }
          resolve({ bytes: val.bytes, mediaType: val.mediaType ?? PAYLOAD_MEDIA_TYPE });
        };
        req.onerror = () => reject(this.wrapError(req.error));
      });
    } finally {
      this.closeDb(db);
    }
  }

  async hasBlobBytes(digest: Digest): Promise<boolean> {
    const db = await this.openDb();
    try {
      return await new Promise<boolean>((resolve, reject) => {
        const tx = db.transaction(STORE_PAYLOADS, 'readonly');
        const req = tx.objectStore(STORE_PAYLOADS).count(digest);
        req.onsuccess = () => resolve((req.result ?? 0) > 0);
        req.onerror = () => reject(this.wrapError(req.error));
      });
    } finally {
      this.closeDb(db);
    }
  }

  async deleteBlobBytes(digest: Digest): Promise<void> {
    const db = await this.openDb();
    try {
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_PAYLOADS, 'readwrite');
        tx.objectStore(STORE_PAYLOADS).delete(digest);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(this.wrapError(tx.error));
      });
    } finally {
      this.closeDb(db);
    }
  }

  async listBlobDigests(): Promise<Digest[]> {
    const db = await this.openDb();
    try {
      return await new Promise<Digest[]>((resolve, reject) => {
        const tx = db.transaction(STORE_PAYLOADS, 'readonly');
        const req = tx.objectStore(STORE_PAYLOADS).getAllKeys();
        req.onsuccess = () => resolve((req.result ?? []).map((k) => String(k)));
        req.onerror = () => reject(this.wrapError(req.error));
      });
    } finally {
      this.closeDb(db);
    }
  }

  async putCheckpoint(runId: string, record: CollectionCheckpointRecord): Promise<void> {
    const db = await this.openDb();
    try {
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_RUNS, 'readwrite');
        tx.objectStore(STORE_RUNS).put(record, runId);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(this.wrapError(tx.error));
        tx.onabort = () => reject(this.wrapError(tx.error));
      });
    } finally {
      this.closeDb(db);
    }
  }

  async getCheckpoint(runId: string): Promise<CollectionCheckpointRecord | null> {
    const db = await this.openDb();
    try {
      return await new Promise<CollectionCheckpointRecord | null>((resolve, reject) => {
        const tx = db.transaction(STORE_RUNS, 'readonly');
        const req = tx.objectStore(STORE_RUNS).get(runId);
        req.onsuccess = () => {
          const val = req.result as CollectionCheckpointRecord | undefined;
          resolve(val && val.schema === 'denovo.checkpoint/v2' ? val : null);
        };
        req.onerror = () => reject(this.wrapError(req.error));
      });
    } finally {
      this.closeDb(db);
    }
  }

  async deleteCheckpoint(runId: string): Promise<void> {
    const db = await this.openDb();
    try {
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_RUNS, 'readwrite');
        tx.objectStore(STORE_RUNS).delete(runId);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(this.wrapError(tx.error));
      });
    } finally {
      this.closeDb(db);
    }
  }

  async listCheckpointIds(): Promise<string[]> {
    const db = await this.openDb();
    try {
      return await new Promise<string[]>((resolve, reject) => {
        const tx = db.transaction(STORE_RUNS, 'readonly');
        const req = tx.objectStore(STORE_RUNS).getAllKeys();
        req.onsuccess = () => resolve((req.result ?? []).map((k) => String(k)));
        req.onerror = () => reject(this.wrapError(req.error));
      });
    } finally {
      this.closeDb(db);
    }
  }

  async commitItemAtomic(commit: AtomicItemCommit): Promise<void> {
    const db = await this.openDb();
    try {
      await new Promise<void>((resolve, reject) => {
        // RFC §4.4: write payload, item, membership, and run checkpoint in ONE IndexedDB transaction.
        const tx = db.transaction(
          [STORE_PAYLOADS, STORE_ITEMS, STORE_COLLECTION_MEMBERS, STORE_RUNS],
          'readwrite',
        );
        tx.objectStore(STORE_PAYLOADS).put(
          { bytes: commit.bytes, mediaType: commit.mediaType },
          commit.digest,
        );
        tx.objectStore(STORE_ITEMS).put(commit.itemRecord, commit.itemRecord.memberKey);
        tx.objectStore(STORE_COLLECTION_MEMBERS).put(
          { ordinal: commit.ordinal, digest: commit.digest },
          `${commit.membershipKey[0]}::${commit.membershipKey[1]}`,
        );
        tx.objectStore(STORE_RUNS).put(commit.checkpoint, commit.checkpoint.runId);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(this.wrapError(tx.error, commit.bytes.byteLength));
        tx.onabort = () => reject(this.wrapError(tx.error, commit.bytes.byteLength));
      });
    } finally {
      this.closeDb(db);
    }
  }
}

/**
 * Create the default backend: IndexedDB when available and accessible,
 * gracefully degrading to MemoryArtifactBackend (`isDurable: false`) in Node
 * or Safari Private Mode / storage-denied environments.
 */
export function createDefaultArtifactBackend(dbName = ARTIFACT_DB_NAME): BlobStorageBackend {
  if (typeof indexedDB === 'undefined') {
    return new MemoryArtifactBackend({ isDurable: false });
  }
  return new IndexedDbArtifactBackend(dbName);
}

// ============================================================================
// ArtifactBlobStore: Bounded Hydrated LRU + Pinning + Quota Resilience
// ============================================================================

export interface ArtifactBlobStoreOptions {
  backend?: BlobStorageBackend;
  /** Maximum unpinned hydrated payload bytes kept in memory (default 32 MiB). */
  maxHydratedBytes?: number;
  /** Maximum combined pinned + unpinned hydrated bytes in memory (default 64 MiB). */
  maxTotalBytes?: number;
  /** Maximum number of hydrated LRU entries kept in memory (default 64). */
  maxHydratedEntries?: number;
  /** Optional callback invoked when storage quota is exceeded. */
  onQuotaExceeded?: (error: StorageQuotaError) => void;
}

export interface ArtifactStoreMemoryStats {
  hydratedCount: number;
  pinnedCount: number;
  hydratedBytes: number;
  pinnedBytes: number;
  unpinnedBytes: number;
  maxHydratedBytes: number;
  maxTotalBytes: number;
  evictionCount: number;
  isDurable: boolean;
  quotaExceeded: boolean;
}

interface HydratedCacheEntry {
  digest: Digest;
  value: unknown;
  bytes: number;
}

/**
 * Durable BlobRef payload store with bounded hydrated-memory LRU and explicit pin ownership.
 *
 * - Keeps manifests lightweight while storing heavy coordinates/PAE/PDB blobs durably.
 * - Evicts unpinned hydrated payloads automatically to honor memory budgets (RFC Gate #11).
 * - Protects pinned keys (active execution inputs & visible viewer models) while bounding total pinned bytes.
 * - Falls back cleanly when IndexedDB is unavailable/denied (marking `isDurable: false`).
 * - Surfaces StorageQuotaError cleanly without corrupting committed state (RFC Gate #12).
 */
export class ArtifactBlobStore {
  private backend: BlobStorageBackend;
  private readonly maxHydratedBytes: number;
  private readonly maxTotalBytes: number;
  private readonly maxHydratedEntries: number;
  private readonly onQuotaExceeded?: (error: StorageQuotaError) => void;

  private readonly hydrated = new Map<Digest, HydratedCacheEntry>();
  private readonly pinnedDigests = new Set<Digest>();
  private readonly memberKeyToDigest = new Map<MemberKey, Digest>();
  private readonly pinnedMemberKeys = new Set<MemberKey>();
  private readonly commitQueues = new Map<string, Promise<unknown>>();

  private evictionCount = 0;
  private _quotaExceeded = false;

  constructor(options: ArtifactBlobStoreOptions = {}) {
    this.backend = options.backend ?? createDefaultArtifactBackend();
    this.maxHydratedBytes = Math.max(1024, options.maxHydratedBytes ?? 32 * 1024 * 1024);
    this.maxTotalBytes = Math.max(
      this.maxHydratedBytes,
      options.maxTotalBytes ?? 64 * 1024 * 1024,
    );
    this.maxHydratedEntries = Math.max(1, options.maxHydratedEntries ?? 64);
    this.onQuotaExceeded = options.onQuotaExceeded;
  }

  get isDurable(): boolean {
    return this.backend.isDurable;
  }

  get quotaExceeded(): boolean {
    return this._quotaExceeded;
  }

  /** Expose backend for checkpoint helpers. */
  getBackend(): BlobStorageBackend {
    return this.backend;
  }

  /** Serialize per-run async operations to prevent commit/cancellation races (RFC Gate #4). */
  enqueueRunOp<T>(runId: string, op: () => Promise<T>): Promise<T> {
    const prev = this.commitQueues.get(runId) ?? Promise.resolve();
    const next = prev.catch(() => undefined).then(op);
    const tracked = next.finally(() => {
      if (this.commitQueues.get(runId) === tracked) {
        this.commitQueues.delete(runId);
      }
    });
    this.commitQueues.set(runId, tracked);
    return next;
  }

  private async runWithFallback<T>(
    op: (backend: BlobStorageBackend) => Promise<T>,
    attemptedBytes = 0,
  ): Promise<T> {
    try {
      return await op(this.backend);
    } catch (err) {
      if (isQuotaExceededError(err)) {
        const quotaErr =
          err instanceof StorageQuotaError
            ? err
            : new StorageQuotaError(
                err instanceof Error ? err.message : 'Storage quota exceeded',
                attemptedBytes,
                err,
              );
        this._quotaExceeded = true;
        this.onQuotaExceeded?.(quotaErr);
        throw quotaErr;
      }
      // If IndexedDB fails to open (e.g. Safari Private Mode / SecurityError), fall back to non-durable memory
      if (this.backend instanceof IndexedDbArtifactBackend) {
        this.backend = new MemoryArtifactBackend({ isDurable: false });
        return await op(this.backend);
      }
      throw err;
    }
  }

  private isDigestPinned(digest: Digest): boolean {
    if (this.pinnedDigests.has(digest)) return true;
    for (const mk of this.pinnedMemberKeys) {
      if (this.memberKeyToDigest.get(mk) === digest) return true;
    }
    return false;
  }

  private computeMemoryTotals(): { pinnedBytes: number; unpinnedBytes: number; pinnedCount: number } {
    let pinnedBytes = 0;
    let unpinnedBytes = 0;
    let pinnedCount = 0;
    for (const entry of this.hydrated.values()) {
      if (this.isDigestPinned(entry.digest)) {
        pinnedBytes += entry.bytes;
        pinnedCount++;
      } else {
        unpinnedBytes += entry.bytes;
      }
    }
    return { pinnedBytes, unpinnedBytes, pinnedCount };
  }

  private enforceMemoryBudget(): void {
    let totals = this.computeMemoryTotals();
    while (
      totals.unpinnedBytes > this.maxHydratedBytes ||
      totals.pinnedBytes + totals.unpinnedBytes > this.maxTotalBytes ||
      this.hydrated.size > this.maxHydratedEntries
    ) {
      let evictedKey: Digest | null = null;
      for (const digest of this.hydrated.keys()) {
        if (!this.isDigestPinned(digest)) {
          evictedKey = digest;
          break;
        }
      }
      if (!evictedKey) break;
      const entry = this.hydrated.get(evictedKey);
      if (entry) {
        this.hydrated.delete(evictedKey);
        this.evictionCount++;
        totals = this.computeMemoryTotals();
      } else {
        break;
      }
    }
  }

  private cacheHydrated(digest: Digest, value: unknown, pinRequested = false): void {
    const bytes = estimatePayloadMemoryBytes(value);
    const alreadyPinned = pinRequested || this.isDigestPinned(digest);

    if (alreadyPinned) {
      const existing = this.hydrated.get(digest);
      const currentTotals = this.computeMemoryTotals();
      const projectedPinned =
        currentTotals.pinnedBytes - (existing && this.isDigestPinned(digest) ? existing.bytes : 0) + bytes;
      if (projectedPinned > this.maxTotalBytes) {
        throw new PinBudgetExceededError(
          `Pinning payload ${digest.slice(0, 12)} (${bytes} B) would exceed maxTotalBytes (${this.maxTotalBytes} B)`,
          projectedPinned,
          this.maxTotalBytes,
        );
      }
    } else if (bytes > this.maxHydratedBytes) {
      // Too large for unpinned cache: still return to caller without retaining in LRU
      return;
    }

    this.hydrated.delete(digest);
    this.hydrated.set(digest, { digest, value, bytes });
    this.enforceMemoryBudget();
  }

  /** Persist a payload durably and return its content-addressed BlobRef (without bloating live RAM). */
  async putBlob(payload: unknown, mediaType = PAYLOAD_MEDIA_TYPE): Promise<BlobRef> {
    const encoded = encodePayloadBytes(payload);
    const digest = sha256Hex(encoded);
    await this.runWithFallback(
      (b) => b.putBlobBytes(digest, encoded, mediaType),
      encoded.byteLength,
    );
    return {
      digest,
      byteLength: encoded.byteLength,
      mediaType,
    };
  }

  /** Retrieve and decode a payload by BlobRef or Digest, populating the bounded LRU cache. */
  async getBlob<T = unknown>(ref: BlobRef | Digest): Promise<T | null> {
    const digest = typeof ref === 'string' ? ref : ref.digest;
    const hit = this.hydrated.get(digest);
    if (hit) {
      // Refresh LRU recency
      this.hydrated.delete(digest);
      this.hydrated.set(digest, hit);
      return hit.value as T;
    }
    const raw = await this.runWithFallback((b) => b.getBlobBytes(digest));
    if (!raw) return null;
    const decoded = decodePayloadBytes<T>(raw.bytes);
    this.cacheHydrated(digest, decoded, false);
    return decoded;
  }

  async hasBlob(ref: BlobRef | Digest): Promise<boolean> {
    const digest = typeof ref === 'string' ? ref : ref.digest;
    if (this.hydrated.has(digest)) return true;
    return this.runWithFallback((b) => b.hasBlobBytes(digest));
  }

  async deleteBlob(ref: BlobRef | Digest): Promise<void> {
    const digest = typeof ref === 'string' ? ref : ref.digest;
    this.pinnedDigests.delete(digest);
    this.hydrated.delete(digest);
    await this.runWithFallback((b) => b.deleteBlobBytes(digest));
  }

  /**
   * Hydrate a CollectionMember's payload on demand from its BlobRef (or return its inline payload).
   * Never mutates the manifest member to hold unbounded strong references.
   */
  async hydratePayload<T = unknown>(
    member: CollectionMember<T>,
    options: { pin?: boolean } = {},
  ): Promise<T | null> {
    const digest = member.payloadRef?.digest;
    if (digest) {
      this.memberKeyToDigest.set(member.key, digest);
    }

    if (member.payload !== undefined) {
      if (digest) {
        if (options.pin) {
          this.pinnedMemberKeys.add(member.key);
          this.pinnedDigests.add(digest);
        }
        this.cacheHydrated(digest, member.payload, Boolean(options.pin));
      }
      return member.payload;
    }

    if (!digest) return null;

    if (options.pin) {
      this.pinnedMemberKeys.add(member.key);
      this.pinnedDigests.add(digest);
    }

    const hit = this.hydrated.get(digest);
    if (hit) {
      this.hydrated.delete(digest);
      this.hydrated.set(digest, hit);
      return hit.value as T;
    }

    const raw = await this.runWithFallback((b) => b.getBlobBytes(digest));
    if (!raw) return null;
    const decoded = decodePayloadBytes<T>(raw.bytes);
    try {
      this.cacheHydrated(digest, decoded, Boolean(options.pin));
    } catch (err) {
      if (options.pin) {
        this.pinnedMemberKeys.delete(member.key);
        this.pinnedDigests.delete(digest);
      }
      throw err;
    }
    return decoded;
  }

  /** Pin one or more member keys or payload digests against LRU eviction. */
  pinKeys(keys: Iterable<MemberKey | Digest>): void {
    for (const k of keys) {
      this.pinnedMemberKeys.add(k);
      this.pinnedDigests.add(k);
      const mapped = this.memberKeyToDigest.get(k);
      if (mapped) this.pinnedDigests.add(mapped);
    }
    const { pinnedBytes } = this.computeMemoryTotals();
    if (pinnedBytes > this.maxTotalBytes) {
      for (const k of keys) {
        this.pinnedMemberKeys.delete(k);
        this.pinnedDigests.delete(k);
        const mapped = this.memberKeyToDigest.get(k);
        if (mapped) this.pinnedDigests.delete(mapped);
      }
      throw new PinBudgetExceededError(
        `Pinned payloads (${pinnedBytes} B) exceed maxTotalBytes (${this.maxTotalBytes} B)`,
        pinnedBytes,
        this.maxTotalBytes,
      );
    }
    this.enforceMemoryBudget();
  }

  /** Release pinned member keys or payload digests and immediately reclaim excess unpinned cache. */
  unpinKeys(keys: Iterable<MemberKey | Digest>): void {
    for (const k of keys) {
      this.pinnedMemberKeys.delete(k);
      this.pinnedDigests.delete(k);
      const mapped = this.memberKeyToDigest.get(k);
      if (mapped) this.pinnedDigests.delete(mapped);
    }
    this.enforceMemoryBudget();
  }

  /** Register a memberKey -> digest association so pinKeys(memberKey) works even before hydration. */
  registerMemberBlob(memberKey: MemberKey, ref: BlobRef): void {
    this.memberKeyToDigest.set(memberKey, ref.digest);
  }

  /** Check whether a digest or memberKey is currently resident in the hydrated RAM cache. */
  isHydrated(keyOrDigest: MemberKey | Digest): boolean {
    if (this.hydrated.has(keyOrDigest)) return true;
    const mapped = this.memberKeyToDigest.get(keyOrDigest);
    return mapped ? this.hydrated.has(mapped) : false;
  }

  clearHydratedCache(): void {
    this.hydrated.clear();
    this.pinnedDigests.clear();
    this.pinnedMemberKeys.clear();
  }

  getMemoryStats(): ArtifactStoreMemoryStats {
    const { pinnedBytes, unpinnedBytes, pinnedCount } = this.computeMemoryTotals();
    return {
      hydratedCount: this.hydrated.size,
      pinnedCount,
      hydratedBytes: pinnedBytes + unpinnedBytes,
      pinnedBytes,
      unpinnedBytes,
      maxHydratedBytes: this.maxHydratedBytes,
      maxTotalBytes: this.maxTotalBytes,
      evictionCount: this.evictionCount,
      isDurable: this.isDurable,
      quotaExceeded: this._quotaExceeded,
    };
  }
}

// ============================================================================
// Resumable v2 Collection Checkpoints & Atomic Per-Item Commits (RFC Gates #3 & #4)
// ============================================================================

function dehydrateMemberForCheckpoint<T>(
  member: CollectionMember<T>,
  payloadRef: BlobRef,
): CollectionMember<unknown> {
  const { payload: _dropped, ...rest } = member;
  return {
    ...rest,
    payloadRef,
  };
}

/**
 * Save a v2 collection checkpoint, dehydrating any inline member payloads into BlobRefs
 * so checkpoints remain compact and never JSON-stringify typed arrays.
 */
export async function saveCollectionCheckpoint(
  store: ArtifactBlobStore,
  checkpoint: CollectionCheckpointRecord,
): Promise<CollectionCheckpointRecord> {
  return store.enqueueRunOp(checkpoint.runId, async () => {
    const dehydratedMembers: CollectionMember<unknown>[] = [];
    for (const member of checkpoint.collection.members) {
      let ref = member.payloadRef;
      if (!ref && member.payload !== undefined) {
        ref = await store.putBlob(member.payload);
      }
      if (ref) {
        store.registerMemberBlob(member.key, ref);
        dehydratedMembers.push(dehydrateMemberForCheckpoint(member, ref));
      } else {
        const { payload: _dropped, ...rest } = member;
        dehydratedMembers.push(rest);
      }
    }

    const updatedCollection: DesignCollection<unknown> = {
      ...checkpoint.collection,
      id: computeCollectionSnapshotId(
        checkpoint.collection.familyKey,
        dehydratedMembers,
        checkpoint.collection.outcomes,
      ),
      members: dehydratedMembers,
    };

    const record: CollectionCheckpointRecord = {
      ...checkpoint,
      schema: 'denovo.checkpoint/v2',
      updatedAt: checkpoint.updatedAt || new Date().toISOString(),
      collection: updatedCollection,
    };

    try {
      await store.getBackend().putCheckpoint(record.runId, record);
    } catch (err) {
      if (isQuotaExceededError(err)) {
        throw err instanceof StorageQuotaError
          ? err
          : new StorageQuotaError('Storage quota exceeded saving checkpoint', 0, err);
      }
      throw err;
    }

    return record;
  });
}

/** Load a resumable v2 collection checkpoint for `runId` (or `null` if absent). */
export async function loadCollectionCheckpoint(
  store: ArtifactBlobStore,
  runId: string,
): Promise<CollectionCheckpointRecord | null> {
  const record = await store.getBackend().getCheckpoint(runId);
  if (!record) return null;
  for (const member of record.collection.members) {
    if (member.payloadRef) {
      store.registerMemberBlob(member.key, member.payloadRef);
    }
  }
  return record;
}

/** Clear one run's checkpoint or all collection checkpoints. */
export async function clearCollectionCheckpoints(
  store: ArtifactBlobStore,
  runId?: string,
): Promise<void> {
  const backend = store.getBackend();
  if (runId !== undefined) {
    await backend.deleteCheckpoint(runId);
    return;
  }
  const ids = await backend.listCheckpointIds();
  await Promise.all(ids.map((id) => backend.deleteCheckpoint(id)));
}

export interface CommitCollectionItemParams<T> {
  runId: string;
  nodeId: string;
  /** Base collection manifest (used if no checkpoint exists yet for this runId). */
  baseCollection: DesignCollection<T>;
  member: CollectionMember<T>;
  payload: T;
  /** All planned member keys for this run so pendingKeys are tracked accurately. */
  plannedKeys: readonly MemberKey[];
  cancelled?: boolean;
}

export interface CommitCollectionItemResult {
  committed: boolean;
  alreadyCommitted: boolean;
  blobRef: BlobRef;
  revision: number;
  checkpoint: CollectionCheckpointRecord;
}

/**
 * Atomically and idempotently commit one completed collection item (RFC Gates #3 & #4).
 *
 * - Serialized per `runId` so concurrent commits or cancellation cannot race.
 * - Prepares/hashes binary bytes BEFORE opening the storage transaction.
 * - If `member.key` is already committed in the checkpoint, returns `alreadyCommitted: true`
 *   without duplicating the member or corrupting the manifest.
 */
export async function commitCollectionItemAtomic<T>(
  store: ArtifactBlobStore,
  params: CommitCollectionItemParams<T>,
): Promise<CommitCollectionItemResult> {
  // Encode & hash BEFORE entering the storage transaction (RFC §4.4)
  const encodedBytes = encodePayloadBytes(params.payload);
  const digest = sha256Hex(encodedBytes);
  const blobRef: BlobRef = {
    digest,
    byteLength: encodedBytes.byteLength,
    mediaType: PAYLOAD_MEDIA_TYPE,
  };

  return store.enqueueRunOp(params.runId, async () => {
    const backend = store.getBackend();
    const existing = await backend.getCheckpoint(params.runId);

    const existingMembers: CollectionMember<unknown>[] = existing
      ? [...existing.collection.members]
      : params.baseCollection.members.map((m) => {
          const { payload: _drop, ...rest } = m;
          return rest;
        });

    // Idempotency guard: never commit the same memberKey twice
    const existingMember = existingMembers.find((m) => m.key === params.member.key);
    if (existing && existingMember) {
      return {
        committed: false,
        alreadyCommitted: true,
        blobRef: existingMember.payloadRef ?? blobRef,
        revision: existing.revision,
        checkpoint: existing,
      };
    }

    const dehydratedMember = dehydrateMemberForCheckpoint(params.member, blobRef);
    const nextMembers = [...existingMembers, dehydratedMember].sort((a, b) =>
      a.ordinal !== b.ordinal ? a.ordinal - b.ordinal : a.key < b.key ? -1 : a.key > b.key ? 1 : 0,
    );

    const committedKeySet = new Set<MemberKey>(nextMembers.map((m) => m.key));
    const priorOutcomes = existing ? existing.collection.outcomes : params.baseCollection.outcomes;
    const nextOutcomeMap = new Map<MemberKey, WorkItemOutcome>();
    for (const o of priorOutcomes) {
      nextOutcomeMap.set(o.key, o);
    }
    const prevOutcome = nextOutcomeMap.get(params.member.key);
    nextOutcomeMap.set(params.member.key, {
      key: params.member.key,
      status: 'completed',
      attempt: (prevOutcome?.attempt ?? 0) + 1,
      lineage: params.member.lineage,
    });
    const nextOutcomes = [...nextOutcomeMap.values()];

    const pendingKeys = params.plannedKeys.filter((k) => {
      if (committedKeySet.has(k)) return false;
      const outcome = nextOutcomeMap.get(k);
      if (outcome && (outcome.status === 'failed' || outcome.status === 'excluded')) {
        return false;
      }
      return true;
    });

    const nextRevision = (existing?.revision ?? params.baseCollection.revision ?? 0) + 1;
    const familyKey = existing?.familyKey ?? params.baseCollection.familyKey;
    const complete = pendingKeys.length === 0 && !params.cancelled;

    const nextCollection: DesignCollection<unknown> = {
      ...(existing?.collection ?? params.baseCollection),
      revision: nextRevision,
      id: computeCollectionSnapshotId(familyKey, nextMembers, nextOutcomes),
      members: nextMembers,
      outcomes: nextOutcomes,
      requestedCount: params.plannedKeys.length,
      complete,
    };

    const nextCheckpoint: CollectionCheckpointRecord = {
      schema: 'denovo.checkpoint/v2',
      runId: params.runId,
      nodeId: params.nodeId,
      familyKey,
      revision: nextRevision,
      updatedAt: new Date().toISOString(),
      collection: nextCollection,
      committedKeys: [...committedKeySet],
      pendingKeys,
      cancelled: Boolean(params.cancelled),
    };

    const commit: AtomicItemCommit = {
      digest,
      bytes: encodedBytes,
      mediaType: PAYLOAD_MEDIA_TYPE,
      itemRecord: {
        memberKey: params.member.key,
        familyKey,
        digest,
        byteLength: encodedBytes.byteLength,
        mediaType: PAYLOAD_MEDIA_TYPE,
        committedAt: nextCheckpoint.updatedAt,
      },
      membershipKey: [familyKey, params.member.key],
      ordinal: params.member.ordinal,
      checkpoint: nextCheckpoint,
    };

    try {
      await backend.commitItemAtomic(commit);
    } catch (err) {
      if (isQuotaExceededError(err)) {
        throw err instanceof StorageQuotaError
          ? err
          : new StorageQuotaError('Storage quota exceeded during atomic item commit', encodedBytes.byteLength, err);
      }
      throw err;
    }

    store.registerMemberBlob(params.member.key, blobRef);
    return {
      committed: true,
      alreadyCommitted: false,
      blobRef,
      revision: nextRevision,
      checkpoint: nextCheckpoint,
    };
  });
}
