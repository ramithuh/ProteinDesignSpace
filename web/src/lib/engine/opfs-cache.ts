// OPFS + CacheStorage weight caching utilities with streaming progress.
// Runs in both Dedicated Web Workers (where OPFS is fastest) and main thread.

const CACHE_NAME = 'denovo-run-weights-v1';
const CACHE_PUT_TIMEOUT_MS = 5000;

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
    if (!contentEncoding && total > 0 && loaded !== total) {
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
    if (!contentEncoding && total > 0 && bytes.byteLength !== total) {
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
    if (typeof caches !== 'undefined') {
      try {
        const cache = await caches.open(CACHE_NAME);
        const hit = await cache.match(url);
        if (hit) {
          onFile?.(url, true);
          return hit.clone();
        }
        const resp = await fetch(input, init);
        if (resp.ok) {
          await cachePutSafe(cache, url, resp.clone());
        }
        onFile?.(url, false);
        return resp;
      } catch {
        /* fall through to plain fetch */
      }
    }
    onFile?.(url, false);
    return fetch(input, init);
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
  const dir = await getWeightsDir(false);
  if (dir) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      for await (const [name, handle] of (dir as any).entries()) {
        if (handle.kind === 'file') {
          const f = await handle.getFile();
          opfsFiles.push({ name, bytes: f.size });
        }
      }
    } catch {
      /* ignore */
    }
  }
  return { usedBytes, quotaBytes, opfsFiles };
}

/** Clear all denovo.run OPFS and CacheStorage caches (including localfold & rfd4 shard caches). */
export async function clearAllModelCaches(): Promise<void> {
  try {
    const root = await navigator.storage?.getDirectory?.();
    await root?.removeEntry('denovo-weights', { recursive: true }).catch(() => {});
  } catch {
    /* ignore */
  }
  if (typeof caches !== 'undefined') {
    try {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter(
            (k) =>
              k.startsWith('denovo-run') ||
              k.startsWith('localfold-') ||
              k.startsWith('rfd4p-'),
          )
          .map((k) => caches.delete(k)),
      );
    } catch {
      /* ignore */
    }
  }
}
