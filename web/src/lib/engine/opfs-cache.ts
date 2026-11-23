// OPFS + CacheStorage weight caching utilities with streaming progress.
// Runs in both Dedicated Web Workers (where OPFS is fastest) and main thread.

import { ARTIFACT_DB_NAME, clearArtifactDatabase } from './artifact-store';

const CACHE_NAME = 'denovo-run-weights-v1';
const CACHE_PUT_TIMEOUT_MS = 5000;

const VENDOR_CACHE_NAMES = new Set([
  'ligandmpnn-v1',
  'esmc-v1',
  'lasermpnn-v1',
  'thermompnn-v1',
  'protpardelle-v1',
  'rfdiffusion-v1',
  'boltzgen-v1',
  'evodiff-v1',
  'la-proteina-v1',
  'proteina-v1',
  'genie2-v1',
  'frameflow-v1',
  'foldflow2-v1',
  'sid-protein-v1',
  'dyna1-v1',
  'gvp-transformer-v1',
  'gnina-v1',
]);

const VENDOR_CACHE_PREFIXES = [
  'denovo-run',
  'localfold-',
  'rfd4p-',
  'genie3-weights-',
];

const OPFS_WEIGHT_DIRS = [
  'denovo-weights',
  'esmc-v1',
  'thermompnn-v1',
  'protpardelle-v1',
  'rfdiffusion-v1',
  'boltzgen-v1',
  'evodiff-v1',
  'la-proteina-v1',
  'proteina-v1',
  'genie2-v1',
  'frameflow-v1',
  'foldflow2-v1',
  'sid-protein-v1',
  'dyna1-v1',
  'gvp-transformer-v1',
  'gnina-v1',
];

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

export interface FetchProgress {
  loaded: number;
  total: number;
  fromCache: boolean;
}

/**
 * Fetch a binary artifact with streaming progress, persisting it in OPFS
 * (falling back gracefully to Cache API / direct network in private mode).
 */
export async function fetchCachedBytes(
  cacheKey: string,
  url: string,
  onProgress?: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  // 1. Try OPFS read first
  const opfsHit = await readFromOpfs(cacheKey);
  if (opfsHit) {
    onProgress?.({ loaded: opfsHit.byteLength, total: opfsHit.byteLength, fromCache: true });
    return opfsHit;
  }

  // 2. Try Cache API fallback
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(CACHE_NAME);
      const hit = await cache.match(url);
      if (hit) {
        const buf = new Uint8Array(await hit.arrayBuffer());
        onProgress?.({ loaded: buf.byteLength, total: buf.byteLength, fromCache: true });
        return buf;
      }
    } catch {
      /* ignore cache errors */
    }
  }

  // 3. Stream from network
  const resp = await fetch(url, { signal });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  const contentEncoding = resp.headers.get('content-encoding');
  const rawTotal = Number(resp.headers.get('content-length') ?? 0);
  const total = Number.isFinite(rawTotal) && rawTotal > 0 ? rawTotal : 0;
  // On cross-origin (CORS) responses, `Content-Encoding` is not a CORS-safelisted header and
  // reads as `null` even when a CDN (e.g. GitHub Pages / Fastly) gzips an already-compressed
  // `.cif.zst` binary—making the encoded `Content-Length` slightly LARGER than the decoded body.
  // Fetch already errors on true HTTP stream truncation; only enforce exact length when non-CORS.
  const canTrustContentLength = !contentEncoding && resp.type !== 'cors';

  let bytes: Uint8Array;
  if (resp.body) {
    const reader = resp.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      // Keep reported total >= loaded so callers computing loaded / total stay <= 1 (e.g. gzipped bodies)
      onProgress?.({ loaded, total: Math.max(total, loaded), fromCache: false });
    }
    if (total > 0 && (loaded === 0 || (canTrustContentLength && loaded < total))) {
      chunks.length = 0;
      throw new Error(`Truncated download for ${url}: received ${loaded} of ${total} bytes`);
    }
    bytes = new Uint8Array(loaded);
    let off = 0;
    for (let i = 0; i < chunks.length; i++) {
      bytes.set(chunks[i], off);
      off += chunks[i].byteLength;
    }
    // Release chunk references before persisting to OPFS to avoid holding ~2x RAM during write
    chunks.length = 0;
  } else {
    bytes = new Uint8Array(await resp.arrayBuffer());
    if (total > 0 && (bytes.byteLength === 0 || (canTrustContentLength && bytes.byteLength < total))) {
      throw new Error(
        `Truncated download for ${url}: received ${bytes.byteLength} of ${total} bytes`,
      );
    }
    onProgress?.({ loaded: bytes.byteLength, total: bytes.byteLength, fromCache: false });
  }

  // 4. Persist to OPFS (best-effort)
  const storedOpfs = await writeToOpfs(cacheKey, bytes);
  if (!storedOpfs && typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(CACHE_NAME);
      await cache.put(url, new Response(bytes as unknown as BodyInit, { status: 200 }));
    } catch {
      /* ignore quota / private-mode errors */
    }
  }

  return bytes;
}

async function cachePutSafe(cache: Cache, url: string, resp: Response): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const putPromise = cache.put(url, resp).catch(() => {});
    await Promise.race([
      putPromise,
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, CACHE_PUT_TIMEOUT_MS);
      }),
    ]);
  } catch {
    /* ignore quota / private-mode errors */
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/** Evict one or more URLs from the Cache API weight cache (e.g. after integrity mismatch). */
export async function evictCachedUrls(urls: Iterable<string>): Promise<void> {
  if (typeof caches === 'undefined') return;
  try {
    const cache = await caches.open(CACHE_NAME);
    const unique = new Set(urls);
    await Promise.all([...unique].map((u) => cache.delete(u).catch(() => false)));
  } catch {
    /* ignore cache errors */
  }
}

/**
 * Drop-in `fetch` wrapper backed by Cache API (for libraries like proteinmpnn-web
 * that fetch small ONNX files & JSON manifests directly).
 */
export function makeCachedFetch(
  onFile?: (url: string, fromCache: boolean) => void,
): typeof globalThis.fetch {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    let cache: Cache | undefined;
    let hit: Response | undefined;
    if (typeof caches !== 'undefined') {
      try {
        cache = await caches.open(CACHE_NAME);
        hit = await cache.match(url);
      } catch {
        /* storage unavailable; fetch normally */
      }
    }
    if (hit) {
      onFile?.(url, true);
      return hit; // Cache.match already returns an independently consumable response.
    }

    const resp = await fetch(input, init);
    if (resp.ok && cache) {
      await cachePutSafe(cache, url, resp.clone());
    }
    onFile?.(url, false);
    return resp;
  };
}

function sanitizeKey(key: string): string {
  let safe = key.replace(/[^a-zA-Z0-9._-]/g, '_');
  if (!safe || safe === '.' || safe === '..') {
    safe = `k_${fnv1a(key)}`;
  }
  if (safe.length > 200) {
    safe = `${safe.slice(0, 191)}_${fnv1a(key)}`;
  }
  return safe;
}

async function getWeightsDir(create: boolean): Promise<FileSystemDirectoryHandle | null> {
  if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return null;
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle('denovo-weights', { create });
  } catch {
    return null;
  }
}

async function readFromOpfs(key: string): Promise<Uint8Array | null> {
  const dir = await getWeightsDir(false);
  if (!dir) return null;
  try {
    const fh = await dir.getFileHandle(sanitizeKey(key));
    const file = await fh.getFile();
    if (file.size === 0) return null;
    return new Uint8Array(await file.arrayBuffer());
  } catch {
    return null;
  }
}

async function writeToOpfs(key: string, bytes: Uint8Array): Promise<boolean> {
  const dir = await getWeightsDir(true);
  if (!dir) return false;
  const safeName = sanitizeKey(key);
  let writable: FileSystemWritableFileStream | null = null;
  try {
    const fh = await dir.getFileHandle(safeName, { create: true });
    // Prefer createWritable when available (works on main thread & modern workers)
    if (typeof fh.createWritable === 'function') {
      writable = await fh.createWritable();
      await writable.write(bytes as unknown as BufferSource);
      await writable.close();
      writable = null;
      return true;
    }
    // Worker-only sync access handle fallback
    const syncHandle = await (
      fh as unknown as {
        createSyncAccessHandle?: () => Promise<{
          truncate(size: number): void;
          write(buf: BufferSource, opts?: { at: number }): number;
          flush(): void;
          close(): void;
        }>;
      }
    ).createSyncAccessHandle?.();
    if (syncHandle) {
      try {
        syncHandle.truncate(0);
        const written = syncHandle.write(bytes as unknown as BufferSource, { at: 0 });
        if (written !== bytes.byteLength) {
          throw new Error(`Short OPFS write: ${written}/${bytes.byteLength} bytes`);
        }
        syncHandle.flush();
      } finally {
        syncHandle.close();
      }
      return true;
    }
    return false;
  } catch {
    if (writable) {
      await writable.abort().catch(() => {});
    }
    await dir.removeEntry(safeName).catch(() => {});
    return false;
  }
}

/** Return storage estimate + cached entry summary for the UI status bar. */
export async function getCacheStats(): Promise<{
  usedBytes: number;
  quotaBytes: number;
  opfsFiles: Array<{ name: string; bytes: number }>;
}> {
  let usedBytes = 0;
  let quotaBytes = 0;
  if (typeof navigator !== 'undefined' && navigator.storage?.estimate) {
    try {
      const est = await navigator.storage.estimate();
      usedBytes = est.usage ?? 0;
      quotaBytes = est.quota ?? 0;
    } catch {
      /* ignore */
    }
  }
  const opfsFiles: Array<{ name: string; bytes: number }> = [];
  if (typeof navigator !== 'undefined' && navigator.storage?.getDirectory) {
    try {
      const root = await navigator.storage.getDirectory();
      for (const dirName of OPFS_WEIGHT_DIRS) {
        const dir = await root.getDirectoryHandle(dirName, { create: false }).catch(() => null);
        if (!dir) continue;
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          for await (const [name, handle] of (dir as any).entries()) {
            if (handle.kind === 'file') {
              const f = await handle.getFile();
              opfsFiles.push({
                name: dirName === 'denovo-weights' ? name : `${dirName}/${name}`,
                bytes: f.size,
              });
            }
          }
        } catch {
          /* ignore */
        }
      }
    } catch {
      /* ignore */
    }
  }
  return { usedBytes, quotaBytes, opfsFiles };
}

const openCheckpointDbs = new Set<IDBDatabase>();

function closeAllOpenCheckpointDbs(): void {
  for (const db of openCheckpointDbs) {
    try {
      db.close();
    } catch {
      /* ignore */
    }
  }
  openCheckpointDbs.clear();
}

async function deleteIndexedDbSafe(dbName: string): Promise<void> {
  if (dbName === CHECKPOINT_DB_NAME) {
    closeAllOpenCheckpointDbs();
  }
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
    } catch {
      finish();
    }
  });
}

/** Clear all denovo.run OPFS, CacheStorage, and IndexedDB weight/artifact caches across all vendor models.
 *  Note: persistent weight caches are managed independently of ExecutionScheduler's in-memory result-cache byte budget. */
export async function clearAllModelCaches(options?: { preserveCheckpoints?: boolean }): Promise<void> {
  if (typeof navigator !== 'undefined' && navigator.storage?.getDirectory) {
    try {
      const root = await navigator.storage.getDirectory();
      await Promise.all(
        OPFS_WEIGHT_DIRS.map((dirName) =>
          root.removeEntry(dirName, { recursive: true }).catch(() => {}),
        ),
      );
    } catch {
      /* ignore */
    }
  }
  if (typeof caches !== 'undefined') {
    try {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter(
            (k) =>
              VENDOR_CACHE_NAMES.has(k) ||
              VENDOR_CACHE_PREFIXES.some((prefix) => k.startsWith(prefix)),
          )
          .map((k) => caches.delete(k)),
      );
    } catch {
      /* ignore */
    }
  }

  const preserveCheckpoints = Boolean(options?.preserveCheckpoints);
  const idbPromises: Promise<void>[] = [];
  const handledDbNames = new Set<string>();
  if (!preserveCheckpoints) {
    idbPromises.push(clearArtifactDatabase(ARTIFACT_DB_NAME));
    handledDbNames.add(ARTIFACT_DB_NAME);
    idbPromises.push(deleteIndexedDbSafe(CHECKPOINT_DB_NAME));
    handledDbNames.add(CHECKPOINT_DB_NAME);
  }
  if (typeof indexedDB !== 'undefined' && typeof indexedDB.databases === 'function') {
    try {
      const dbs = await indexedDB.databases();
      for (const dbInfo of dbs ?? []) {
        const name = dbInfo?.name;
        if (
          name &&
          name.startsWith('denovo-run') &&
          !handledDbNames.has(name) &&
          !(preserveCheckpoints && (name === CHECKPOINT_DB_NAME || name === ARTIFACT_DB_NAME))
        ) {
          handledDbNames.add(name);
          idbPromises.push(deleteIndexedDbSafe(name));
        }
      }
    } catch {
      /* ignore */
    }
  }
  await Promise.all(idbPromises);
}

/** Storage adapter interface for durable workflow run checkpoints (independent of model weight caches). */
export interface CheckpointStorageAdapter {
  save(key: string, serialized: string): Promise<void> | void;
  load(key: string): Promise<string | null> | string | null;
  delete?(key: string): Promise<void> | void;
  listKeys?(): Promise<string[]> | string[];
}

/** In-memory checkpoint storage adapter for tests and headless sessions. */
export function createMemoryCheckpointStore(
  initial?: Record<string, string>,
): CheckpointStorageAdapter {
  const store = new Map<string, string>(Object.entries(initial ?? {}));
  return {
    save(key: string, serialized: string): void {
      store.set(key, serialized);
    },
    load(key: string): string | null {
      return store.get(key) ?? null;
    },
    delete(key: string): void {
      store.delete(key);
    },
    listKeys(): string[] {
      return [...store.keys()];
    },
  };
}

const CHECKPOINT_DB_NAME = 'denovo-run-checkpoints-v1';
const CHECKPOINT_STORE_NAME = 'checkpoints';

/** Browser IndexedDB-backed checkpoint storage adapter (falls back to in-memory map if IndexedDB is unavailable). */
export function createIndexedDbCheckpointStore(): CheckpointStorageAdapter {
  if (typeof indexedDB === 'undefined') {
    return createMemoryCheckpointStore();
  }

  const closeDb = (db: IDBDatabase) => {
    openCheckpointDbs.delete(db);
    try {
      db.close();
    } catch {
      /* ignore */
    }
  };

  const openDb = (): Promise<IDBDatabase> =>
    new Promise((resolve, reject) => {
      const req = indexedDB.open(CHECKPOINT_DB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(CHECKPOINT_STORE_NAME)) {
          db.createObjectStore(CHECKPOINT_STORE_NAME);
        }
      };
      req.onsuccess = () => {
        const db = req.result;
        openCheckpointDbs.add(db);
        db.onversionchange = () => {
          openCheckpointDbs.delete(db);
          try {
            db.close();
          } catch {
            /* ignore */
          }
        };
        db.onclose = () => {
          openCheckpointDbs.delete(db);
        };
        resolve(db);
      };
      req.onerror = () => reject(req.error ?? new Error('Failed to open checkpoint IndexedDB'));
    });

  return {
    async save(key: string, serialized: string): Promise<void> {
      const db = await openDb();
      try {
        await new Promise<void>((resolve, reject) => {
          const tx = db.transaction(CHECKPOINT_STORE_NAME, 'readwrite');
          tx.objectStore(CHECKPOINT_STORE_NAME).put(serialized, key);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error ?? new Error('Failed to save checkpoint'));
        });
      } finally {
        closeDb(db);
      }
    },
    async load(key: string): Promise<string | null> {
      const db = await openDb();
      try {
        return await new Promise<string | null>((resolve, reject) => {
          const tx = db.transaction(CHECKPOINT_STORE_NAME, 'readonly');
          const req = tx.objectStore(CHECKPOINT_STORE_NAME).get(key);
          req.onsuccess = () => resolve(typeof req.result === 'string' ? req.result : null);
          req.onerror = () => reject(req.error ?? new Error('Failed to load checkpoint'));
        });
      } finally {
        closeDb(db);
      }
    },
    async delete(key: string): Promise<void> {
      const db = await openDb();
      try {
        await new Promise<void>((resolve, reject) => {
          const tx = db.transaction(CHECKPOINT_STORE_NAME, 'readwrite');
          tx.objectStore(CHECKPOINT_STORE_NAME).delete(key);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error ?? new Error('Failed to delete checkpoint'));
        });
      } finally {
        closeDb(db);
      }
    },
    async listKeys(): Promise<string[]> {
      const db = await openDb();
      try {
        return await new Promise<string[]>((resolve, reject) => {
          const tx = db.transaction(CHECKPOINT_STORE_NAME, 'readonly');
          const req = tx.objectStore(CHECKPOINT_STORE_NAME).getAllKeys();
          req.onsuccess = () => resolve((req.result ?? []).map((k) => String(k)));
          req.onerror = () => reject(req.error ?? new Error('Failed to list checkpoint keys'));
        });
      } finally {
        closeDb(db);
      }
    },
  };
}
