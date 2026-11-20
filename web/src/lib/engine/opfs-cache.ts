// OPFS + CacheStorage weight caching utilities with streaming progress.
// Runs in both Dedicated Web Workers (where OPFS is fastest) and main thread.

const CACHE_NAME = 'denovo-run-weights-v1';

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
  const total = Number(resp.headers.get('content-length') ?? 0);

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
      onProgress?.({ loaded, total: total || loaded, fromCache: false });
    }
    bytes = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) {
      bytes.set(c, off);
      off += c.byteLength;
    }
  } else {
    bytes = new Uint8Array(await resp.arrayBuffer());
    onProgress?.({ loaded: bytes.byteLength, total: bytes.byteLength, fromCache: false });
  }

  // 4. Persist to OPFS (best-effort)
  const storedOpfs = await writeToOpfs(cacheKey, bytes);
  if (!storedOpfs && typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(CACHE_NAME);
      const body = bytes.slice().buffer;
      await cache.put(url, new Response(body, { status: 200 }));
    } catch {
      /* ignore quota / private-mode errors */
    }
  }

  return bytes;
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
          void cache.put(url, resp.clone()).catch(() => {});
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
  return key.replace(/[^a-zA-Z0-9._-]/g, '_');
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
  try {
    const fh = await dir.getFileHandle(sanitizeKey(key), { create: true });
    // Prefer createWritable when available (works on main thread & modern workers)
    if (typeof fh.createWritable === 'function') {
      const writable = await fh.createWritable();
      await writable.write(bytes.slice());
      await writable.close();
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
        syncHandle.write(bytes.slice(), { at: 0 });
        syncHandle.flush();
      } finally {
        syncHandle.close();
      }
      return true;
    }
    return false;
  } catch {
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
