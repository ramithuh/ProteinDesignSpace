import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  clearAllModelCaches,
  evictCachedUrls,
  fetchCachedBytes,
  makeCachedFetch,
  type FetchProgress,
} from '$lib/engine/opfs-cache';

function makeStreamResponse(
  chunks: Uint8Array[],
  headers: Record<string, string>,
  status = 200,
): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(c);
      controller.close();
    },
  });
  return new Response(stream, { status, headers });
}

describe('engine/opfs-cache', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('rejects truncated streaming downloads when loaded < content-length without content-encoding', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        makeStreamResponse(
          [new Uint8Array([1, 2, 3]), new Uint8Array([4, 5])],
          { 'content-length': '10' },
        ),
      ),
    );

    await expect(fetchCachedBytes('trunc-stream', 'https://example.test/weights.bin')).rejects.toThrow(
      /Truncated download/,
    );
  });

  it('rejects truncated non-streaming (body === null) downloads when bytes < content-length', async () => {
    const fakeResp = {
      ok: true,
      status: 200,
      headers: new Headers({ 'content-length': '16' }),
      body: null,
      arrayBuffer: async () => new Uint8Array([1, 2, 3, 4]).buffer,
    } as unknown as Response;

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fakeResp));

    await expect(fetchCachedBytes('trunc-buf', 'https://example.test/weights2.bin')).rejects.toThrow(
      /Truncated download/,
    );
  });

  it('accepts CORS-compressed/decompressed downloads where loaded > content-length and clamps progress <= 1', async () => {
    // Simulate a cross-origin gzipped response where CORS exposes compressed Content-Length=4
    // without exposing Content-Encoding, while browser transparently streams 10 decompressed bytes.
    const c1 = new Uint8Array([10, 20, 30, 40, 50]);
    const c2 = new Uint8Array([60, 70, 80, 90, 100]);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(makeStreamResponse([c1, c2], { 'content-length': '4' })),
    );

    const progressEvents: FetchProgress[] = [];
    const bytes = await fetchCachedBytes(
      'cors-gzip-key',
      'https://example.test/model.bin',
      (p) => progressEvents.push(p),
    );

    expect(Array.from(bytes)).toEqual([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    expect(progressEvents.length).toBe(2);
    for (const p of progressEvents) {
      expect(p.fromCache).toBe(false);
      expect(p.total).toBeGreaterThanOrEqual(p.loaded);
      expect(p.loaded / p.total).toBeLessThanOrEqual(1);
    }
  });

  it('accepts CORS responses where hidden gzip expands Content-Length above decoded bytes (e.g. .cif.zst)', async () => {
    const payload = new Uint8Array(3802).fill(42);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(payload);
        controller.close();
      },
    });
    const corsResp = {
      ok: true,
      status: 200,
      type: 'cors',
      headers: new Headers({ 'content-length': '3830' }),
      body: stream,
    } as unknown as Response;

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(corsResp));

    const bytes = await fetchCachedBytes('cors-zst-key', 'https://web.rfd4-proteina.org/web/examples/sm_sam.cif.zst');
    expect(bytes.byteLength).toBe(3802);
  });

  it('accepts exact-length downloads and rejects HTTP error responses', async () => {
    const payload = new Uint8Array([7, 8, 9, 10]);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(makeStreamResponse([payload], { 'content-length': '4' })),
    );
    const okBytes = await fetchCachedBytes('ok-key', 'https://example.test/ok.bin');
    expect(Array.from(okBytes)).toEqual([7, 8, 9, 10]);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('not found', { status: 404 })));
    await expect(fetchCachedBytes('err-key', 'https://example.test/missing.bin')).rejects.toThrow(
      /HTTP 404/,
    );
  });

  it('propagates a network failure after a cache miss without fetching twice', async () => {
    vi.stubGlobal('caches', { open: async () => ({ match: async () => undefined }) });
    const network = vi.fn().mockRejectedValue(new Error('connection interrupted'));
    vi.stubGlobal('fetch', network);
    await expect(makeCachedFetch()('https://example.test/model')).rejects.toThrow('connection interrupted');
    expect(network).toHaveBeenCalledTimes(1);
  });

  it('returns the independently consumable cache match without cloning its body', async () => {
    const hit = new Response('cached bytes');
    vi.stubGlobal('caches', { open: async () => ({ match: async () => hit }) });
    expect(await makeCachedFetch()('https://example.test/model')).toBe(hit);
  });

  it('clears vendored Genie weight caches and preserves unrelated caches', async () => {
    const remove = vi.fn().mockResolvedValue(true);
    vi.stubGlobal('caches', {
      keys: async () => ['genie3-weights-fp16-1', 'genie3-unrelated', 'other-cache'],
      delete: remove,
    });
    await clearAllModelCaches();
    expect(remove.mock.calls).toEqual([['genie3-weights-fp16-1']]);
  });

  it('makeCachedFetch caches on miss, hits on repeat, and evictCachedUrls removes cached entries', async () => {
    const store = new Map<string, Response>();
    const fakeCache = {
      match: vi.fn(async (url: string) => store.get(url)),
      put: vi.fn(async (url: string, resp: Response) => {
        store.set(url, resp);
      }),
      delete: vi.fn(async (url: string) => store.delete(url)),
    };
    vi.stubGlobal('caches', {
      open: vi.fn(async () => fakeCache),
    });

    const netFetch = vi.fn(async () => new Response('model-bytes', { status: 200 }));
    vi.stubGlobal('fetch', netFetch);

    const seen: Array<[string, boolean]> = [];
    const cachedFetch = makeCachedFetch((url, fromCache) => seen.push([url, fromCache]));

    // 1st call: miss -> fetches from network and populates cache
    const r1 = await cachedFetch('https://example.test/mpnn.onnx');
    expect(await r1.text()).toBe('model-bytes');
    expect(netFetch).toHaveBeenCalledTimes(1);
    expect(seen[0]).toEqual(['https://example.test/mpnn.onnx', false]);

    // 2nd call: cache hit -> zero network calls
    const r2 = await cachedFetch('https://example.test/mpnn.onnx');
    expect(await r2.text()).toBe('model-bytes');
    expect(netFetch).toHaveBeenCalledTimes(1);
    expect(seen[1]).toEqual(['https://example.test/mpnn.onnx', true]);

    // Evict -> 3rd call hits network again
    await evictCachedUrls(['https://example.test/mpnn.onnx']);
    const r3 = await cachedFetch('https://example.test/mpnn.onnx');
    expect(await r3.text()).toBe('model-bytes');
    expect(netFetch).toHaveBeenCalledTimes(2);
    expect(seen[2]).toEqual(['https://example.test/mpnn.onnx', false]);
  });
});
