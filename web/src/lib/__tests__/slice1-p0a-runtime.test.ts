// Slice 1 (P0a): ORT 1.31.1 MPNN tuning, streaming OPFS + in-flight dedup, device-lost guards,
// non-poisoning worker control chain, LocalFold device retention, and RFD4 post-upload shard release.
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { existsSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fetchCachedBytes, getCacheStats, type FetchProgress } from '$lib/engine/opfs-cache';
import {
  createDeviceLostGuard,
  requestResilientWebGpuDevice,
  tryCreateBuffer,
  WebGpuOutOfMemoryError,
} from '$lib/kernel/webgpu-device';
import { createWorkerHarness, type ModelRunner, type WorkerScopeLike } from '$lib/kernel/worker-harness';
import {
  releaseRfd4UploadedShardBytes,
  Rfd4ReleasedWeightError,
  type Rfd4ShardedBundleLike,
} from '$lib/plugins/rfd4-proteina/lifecycle';
import {
  MPNN_WEBGPU_EP_OPTIONS,
  ORT_WEBGPU_ADAPTER_LIMITS,
  resolveOrtWasmThreads,
  upgradeWebGpuExecutionProviders,
  withWebGpuExecutionProviderOptions,
} from '$lib/plugins/web-protein-mpnn/ort-session-tuning';
import type { SequenceSet } from '$lib/engine/types';

const WEB_ROOT = resolve(import.meta.dirname, '../../..');
const readSrc = (rel: string) => readFileSync(resolve(WEB_ROOT, rel), 'utf8');

const flush = () => new Promise<void>((r) => setTimeout(r, 0));

function streamResponse(chunks: number[][], headers: Record<string, string> = {}): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(new Uint8Array(c));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers });
}

// ---------------------------------------------------------------------------------------------
// In-memory OPFS (directory + worker-only sync access handles)
// ---------------------------------------------------------------------------------------------

interface MemFile {
  data: Uint8Array;
  locked: boolean;
}

interface MemOpfsOptions {
  /** `createSyncAccessHandle`: available, throws NoModificationAllowedError, or absent (main thread). */
  syncAccess?: 'ok' | 'busy' | 'none';
  /** `createWritable`: available, throws NoModificationAllowedError, or absent. */
  writable?: 'ok' | 'busy' | 'none';
  /** 1-based sync-handle write call that throws QuotaExceededError. */
  failWriteOnCall?: number;
}

function createMemOpfs(opts: MemOpfsOptions = {}) {
  const files = new Map<string, MemFile>();
  const log = {
    syncHandles: [] as Array<{ name: string; closed: boolean; writes: Array<{ at: number; len: number }> }>,
    writableCalls: [] as string[],
  };
  let writeCalls = 0;
  const notFound = () => new DOMException('missing', 'NotFoundError');
  const busy = () => new DOMException('locked', 'NoModificationAllowedError');

  const fileHandle = (name: string) => {
    const handle: Record<string, unknown> = {
      kind: 'file',
      getFile: async () => {
        const f = files.get(name);
        if (!f) throw notFound();
        const copy = f.data.slice();
        return { size: copy.byteLength, arrayBuffer: async () => copy.buffer };
      },
    };
    if (opts.writable !== 'none') {
      handle.createWritable = async () => {
        if (opts.writable === 'busy' || files.get(name)?.locked) throw busy();
        log.writableCalls.push(name);
        let staged = new Uint8Array(0);
        return {
          write: async (buf: BufferSource) => {
            staged = new Uint8Array(buf as ArrayBuffer).slice();
          },
          close: async () => {
            files.set(name, { data: staged, locked: false });
          },
          abort: async () => {},
        };
      };
    }
    if (opts.syncAccess !== 'none') {
      handle.createSyncAccessHandle = async () => {
        const f = files.get(name);
        if (!f || opts.syncAccess === 'busy' || f.locked) throw busy();
        f.locked = true;
        const entry = { name, closed: false, writes: [] as Array<{ at: number; len: number }> };
        log.syncHandles.push(entry);
        return {
          read(buf: Uint8Array, o?: { at: number }) {
            const at = o?.at ?? 0;
            const n = Math.max(0, Math.min(buf.byteLength, f.data.byteLength - at));
            buf.set(f.data.subarray(at, at + n));
            return n;
          },
          write(buf: Uint8Array, o?: { at: number }) {
            writeCalls++;
            if (opts.failWriteOnCall === writeCalls) throw new DOMException('full', 'QuotaExceededError');
            const at = o?.at ?? 0;
            if (at + buf.byteLength > f.data.byteLength) {
              const grown = new Uint8Array(at + buf.byteLength);
              grown.set(f.data);
              f.data = grown;
            }
            f.data.set(buf, at);
            entry.writes.push({ at, len: buf.byteLength });
            return buf.byteLength;
          },
          truncate(size: number) {
            const next = new Uint8Array(size);
            next.set(f.data.subarray(0, Math.min(size, f.data.byteLength)));
            f.data = next;
          },
          getSize: () => f.data.byteLength,
          flush() {},
          close() {
            entry.closed = true;
            f.locked = false;
          },
        };
      };
    }
    return handle;
  };

  const dir = {
    async getFileHandle(name: string, o?: { create?: boolean }) {
      if (!files.has(name)) {
        if (!o?.create) throw notFound();
        files.set(name, { data: new Uint8Array(0), locked: false });
      }
      return fileHandle(name);
    },
    async removeEntry(name: string) {
      const f = files.get(name);
      if (!f) throw notFound();
      if (f.locked) throw busy();
      files.delete(name);
    },
    entries: async function* () {
      for (const name of [...files.keys()]) yield [name, fileHandle(name)] as const;
    },
  };
  const root = {
    async getDirectoryHandle(name: string) {
      if (name !== 'denovo-weights') throw notFound();
      return dir;
    },
    async removeEntry() {},
  };
  vi.stubGlobal('navigator', { storage: { getDirectory: async () => root } });
  return { files, log, read: (name: string) => files.get(name)?.data };
}

describe('Slice 1 P0a — engine/opfs-cache INFLIGHT_LOADS dedup', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('coalesces concurrent identical loads into exactly one network request and fans progress out', async () => {
    let release!: (r: Response) => void;
    const network = vi.fn(() => new Promise<Response>((r) => (release = r)));
    vi.stubGlobal('fetch', network);

    const seen: FetchProgress[][] = [[], [], []];
    const url = 'https://example.test/dedup.bin';
    const loads = seen.map((events) => fetchCachedBytes('dedup-key', url, (p) => events.push(p)));
    await vi.waitFor(() => expect(network).toHaveBeenCalledTimes(1));
    release(streamResponse([[1, 2, 3], [4, 5]], { 'content-length': '5' }));

    const [a, b, c] = await Promise.all(loads);
    expect(network).toHaveBeenCalledTimes(1);
    expect(Array.from(a)).toEqual([1, 2, 3, 4, 5]);
    expect(b).toBe(a);
    expect(c).toBe(a);
    for (const events of seen) {
      expect(events.at(-1)).toEqual({ loaded: 5, total: 5, fromCache: false });
    }

    // Settled loads leave the map: a later call starts a fresh load (no OPFS / Cache API here).
    network.mockImplementationOnce(async () => streamResponse([[9]], { 'content-length': '1' }));
    expect(Array.from(await fetchCachedBytes('dedup-key', url))).toEqual([9]);
    expect(network).toHaveBeenCalledTimes(2);
  });

  it('keeps distinct keys independent', async () => {
    const network = vi.fn(async (u: string) => streamResponse([[u.endsWith('a.bin') ? 1 : 2]]));
    vi.stubGlobal('fetch', network);
    const [a, b] = await Promise.all([
      fetchCachedBytes('key-a', 'https://example.test/a.bin'),
      fetchCachedBytes('key-b', 'https://example.test/b.bin'),
    ]);
    expect(network).toHaveBeenCalledTimes(2);
    expect(Array.from(a)).toEqual([1]);
    expect(Array.from(b)).toEqual([2]);
  });

  it('aborts only the cancelled caller, and the shared download only once every caller has aborted', async () => {
    let release!: (r: Response) => void;
    const signals: AbortSignal[] = [];
    const network = vi.fn((_u: string, init?: RequestInit) => {
      signals.push(init!.signal!);
      return new Promise<Response>((r) => (release = r));
    });
    vi.stubGlobal('fetch', network);
    const url = 'https://example.test/abort.bin';

    const ctrlA = new AbortController();
    const pA = fetchCachedBytes('abort-key', url, undefined, ctrlA.signal);
    const pB = fetchCachedBytes('abort-key', url);
    await vi.waitFor(() => expect(network).toHaveBeenCalledTimes(1));
    ctrlA.abort(new Error('user cancelled A'));
    await expect(pA).rejects.toThrow('user cancelled A');
    expect(signals[0].aborted).toBe(false);
    release(streamResponse([[7, 7]]));
    expect(Array.from(await pB)).toEqual([7, 7]);

    const c1 = new AbortController();
    const c2 = new AbortController();
    const q1 = fetchCachedBytes('abort-all', url, undefined, c1.signal);
    const q2 = fetchCachedBytes('abort-all', url, undefined, c2.signal);
    await vi.waitFor(() => expect(network).toHaveBeenCalledTimes(2));
    c1.abort(new Error('one'));
    await expect(q1).rejects.toThrow('one');
    expect(signals[1].aborted).toBe(false);
    c2.abort(new Error('two'));
    await expect(q2).rejects.toThrow('two');
    expect(signals[1].aborted).toBe(true);

    // The aborted load is gone from the map immediately: a new caller starts over.
    void fetchCachedBytes('abort-all', url).catch(() => {});
    await vi.waitFor(() => expect(network).toHaveBeenCalledTimes(3));
  });
});

describe('Slice 1 P0a — engine/opfs-cache streaming createSyncAccessHandle writes', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('streams each chunk to OPFS at its offset, verifies the size, clears the marker, and serves later reads from OPFS', async () => {
    const opfs = createMemOpfs();
    const network = vi.fn(async () =>
      streamResponse([[1, 2, 3, 4, 5], [6, 7, 8, 9], [10]], { 'content-length': '10' }),
    );
    vi.stubGlobal('fetch', network);

    const progress: FetchProgress[] = [];
    const bytes = await fetchCachedBytes('stream-key', 'https://example.test/s.bin', (p) => progress.push(p));
    expect(Array.from(bytes)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(progress.map((p) => p.loaded)).toEqual([5, 9, 10]);

    expect(opfs.log.syncHandles).toHaveLength(1);
    expect(opfs.log.syncHandles[0].writes).toEqual([
      { at: 0, len: 5 },
      { at: 5, len: 4 },
      { at: 9, len: 1 },
    ]);
    expect(opfs.log.syncHandles[0].closed).toBe(true);
    expect(opfs.log.writableCalls).toEqual([]);
    expect(Array.from(opfs.read('stream-key') ?? [])).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(opfs.files.has('stream-key.incomplete')).toBe(false);

    const again: FetchProgress[] = [];
    const cached = await fetchCachedBytes('stream-key', 'https://example.test/s.bin', (p) => again.push(p));
    expect(Array.from(cached)).toEqual(Array.from(bytes));
    expect(again).toEqual([{ loaded: 10, total: 10, fromCache: true }]);
    expect(network).toHaveBeenCalledTimes(1);
  });

  it('treats an entry left with an .incomplete marker (interrupted stream) as a miss and rewrites it', async () => {
    const opfs = createMemOpfs();
    opfs.files.set('crash-key', { data: new Uint8Array([1, 2]), locked: false });
    opfs.files.set('crash-key.incomplete', { data: new Uint8Array(0), locked: false });
    const network = vi.fn(async () => streamResponse([[1, 2, 3, 4]], { 'content-length': '4' }));
    vi.stubGlobal('fetch', network);

    const bytes = await fetchCachedBytes('crash-key', 'https://example.test/c.bin');
    expect(network).toHaveBeenCalledTimes(1);
    expect(Array.from(bytes)).toEqual([1, 2, 3, 4]);
    expect(Array.from(opfs.read('crash-key') ?? [])).toEqual([1, 2, 3, 4]);
    expect(opfs.files.has('crash-key.incomplete')).toBe(false);
  });

  it('keeps legacy entries (written before markers existed) as cache hits', async () => {
    const opfs = createMemOpfs();
    opfs.files.set('legacy-key', { data: new Uint8Array([4, 2]), locked: false });
    const network = vi.fn();
    vi.stubGlobal('fetch', network);
    expect(Array.from(await fetchCachedBytes('legacy-key', 'https://example.test/l.bin'))).toEqual([4, 2]);
    expect(network).not.toHaveBeenCalled();
  });

  it('falls back to buffered createWritable() when the sync handle is busy (NoModificationAllowedError)', async () => {
    const opfs = createMemOpfs({ syncAccess: 'busy' });
    vi.stubGlobal('fetch', vi.fn(async () => streamResponse([[5, 6], [7]], { 'content-length': '3' })));

    const bytes = await fetchCachedBytes('busy-key', 'https://example.test/b.bin');
    expect(Array.from(bytes)).toEqual([5, 6, 7]);
    expect(opfs.log.syncHandles).toEqual([]);
    expect(opfs.log.writableCalls).toEqual(['busy-key']);
    expect(Array.from(opfs.read('busy-key') ?? [])).toEqual([5, 6, 7]);
    expect(opfs.files.has('busy-key.incomplete')).toBe(false);
  });

  it("never touches another context's in-progress stream, and falls back to the Cache API", async () => {
    const opfs = createMemOpfs({ writable: 'busy' });
    // Another worker is mid-stream: its marker exists and it holds the file's sync handle.
    opfs.files.set('shared-key', { data: new Uint8Array([1]), locked: true });
    opfs.files.set('shared-key.incomplete', { data: new Uint8Array(0), locked: false });
    const put = vi.fn(async () => {});
    vi.stubGlobal('caches', { open: async () => ({ match: async () => undefined, put }) });
    vi.stubGlobal('fetch', vi.fn(async () => streamResponse([[1, 2, 3]], { 'content-length': '3' })));

    const bytes = await fetchCachedBytes('shared-key', 'https://example.test/shared.bin');
    expect(Array.from(bytes)).toEqual([1, 2, 3]);
    expect(opfs.files.has('shared-key.incomplete')).toBe(true);
    expect(Array.from(opfs.read('shared-key') ?? [])).toEqual([1]);
    expect(put).toHaveBeenCalledTimes(1);
  });

  it('spills to memory when an OPFS write fails mid-stream (quota) and still returns and persists every byte', async () => {
    const opfs = createMemOpfs({ failWriteOnCall: 2 });
    vi.stubGlobal('fetch', vi.fn(async () => streamResponse([[1, 2], [3, 4], [5]], { 'content-length': '5' })));

    const bytes = await fetchCachedBytes('quota-key', 'https://example.test/q.bin');
    expect(Array.from(bytes)).toEqual([1, 2, 3, 4, 5]);
    expect(opfs.log.syncHandles[0].closed).toBe(true);
    // The partial stream was discarded and the full buffer re-persisted through createWritable().
    expect(opfs.log.writableCalls).toEqual(['quota-key']);
    expect(Array.from(opfs.read('quota-key') ?? [])).toEqual([1, 2, 3, 4, 5]);
    expect(opfs.files.has('quota-key.incomplete')).toBe(false);
  });

  it('removes the partial entry and its marker when a streamed download is truncated', async () => {
    const opfs = createMemOpfs();
    vi.stubGlobal('fetch', vi.fn(async () => streamResponse([[1, 2, 3, 4]], { 'content-length': '10' })));
    await expect(fetchCachedBytes('trunc-key', 'https://example.test/t.bin')).rejects.toThrow(/Truncated download/);
    expect(opfs.files.has('trunc-key')).toBe(false);
    expect(opfs.files.has('trunc-key.incomplete')).toBe(false);
    expect(opfs.log.syncHandles[0].closed).toBe(true);
  });

  it('stays on the buffered path on the main thread (no createSyncAccessHandle)', async () => {
    const opfs = createMemOpfs({ syncAccess: 'none' });
    vi.stubGlobal('fetch', vi.fn(async () => streamResponse([[8, 9]])));
    expect(Array.from(await fetchCachedBytes('main-key', 'https://example.test/m.bin'))).toEqual([8, 9]);
    expect(opfs.log.writableCalls).toEqual(['main-key']);
    expect(opfs.files.has('main-key.incomplete')).toBe(false);
  });

  it('hides .incomplete markers from cache stats', async () => {
    const opfs = createMemOpfs();
    opfs.files.set('done.bin', { data: new Uint8Array(3), locked: false });
    opfs.files.set('half.bin.incomplete', { data: new Uint8Array(0), locked: false });
    const stats = await getCacheStats();
    expect(stats.opfsFiles).toEqual([{ name: 'done.bin', bytes: 3 }]);
  });
});

describe('Slice 1 P0a — kernel/webgpu-device tryCreateBuffer + device-lost guard', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function mockDevice(popResult: unknown, createThrows = false) {
    const scopes: string[] = [];
    const buffer = { destroy: vi.fn() };
    const device = {
      pushErrorScope: vi.fn((filter: string) => {
        scopes.push(filter);
      }),
      popErrorScope: vi.fn(async () => {
        scopes.pop();
        return popResult;
      }),
      createBuffer: vi.fn(() => {
        if (createThrows) throw new TypeError('bad descriptor');
        return buffer;
      }),
    };
    return { device: device as unknown as GPUDevice, raw: device, scopes, buffer };
  }

  it('returns the buffer inside a balanced out-of-memory error scope', async () => {
    const m = mockDevice(null);
    const buf = await tryCreateBuffer(m.device, { size: 64, usage: 0x80 });
    expect(buf).toBe(m.buffer);
    expect(m.raw.pushErrorScope).toHaveBeenCalledWith('out-of-memory');
    expect(m.raw.popErrorScope).toHaveBeenCalledTimes(1);
    expect(m.scopes).toEqual([]);
    expect(m.buffer.destroy).not.toHaveBeenCalled();
  });

  it('destroys the invalid buffer and rejects with WebGpuOutOfMemoryError on OOM', async () => {
    const m = mockDevice({ message: 'Not enough memory left' });
    const err = await tryCreateBuffer(m.device, { size: 1 << 30, usage: 0x80, label: 'pair' }).catch((e) => e);
    expect(err).toBeInstanceOf(WebGpuOutOfMemoryError);
    expect(err.message).toContain('1073741824 bytes');
    expect(err.message).toContain('"pair"');
    expect(err.message).toContain('Not enough memory left');
    expect(m.buffer.destroy).toHaveBeenCalledTimes(1);
    expect(m.scopes).toEqual([]);
  });

  it('pops the scope even when createBuffer throws synchronously', async () => {
    const m = mockDevice(null, true);
    await expect(tryCreateBuffer(m.device, { size: 4, usage: 0x80 })).rejects.toThrow('bad descriptor');
    expect(m.raw.popErrorScope).toHaveBeenCalledTimes(1);
    expect(m.scopes).toEqual([]);
  });

  function lostDevice(lostAsync: boolean) {
    let resolveLost!: (info: GPUDeviceLostInfo) => void;
    const lost = new Promise<GPUDeviceLostInfo>((r) => (resolveLost = r));
    const destroyed = { reason: 'destroyed', message: '' } as GPUDeviceLostInfo;
    const device = {
      lost,
      destroy: vi.fn(() => {
        // Browsers resolve `lost` after destroy() returns; model that by deferring a task.
        if (lostAsync) setTimeout(() => resolveLost(destroyed), 0);
        else resolveLost(destroyed);
      }),
    };
    return { device: device as unknown as GPUDevice, raw: device, lose: (info: GPUDeviceLostInfo) => resolveLost(info) };
  }

  it('suppresses device-lost handling for intentional destroy(), even when `lost` resolves later', async () => {
    const guard = createDeviceLostGuard();
    const d = lostDevice(true);
    const onLost = vi.fn();
    guard.watch(d.device, onLost);
    guard.destroy(d.device);
    expect(d.raw.destroy).toHaveBeenCalledTimes(1);
    expect(guard.deviceLostIsError).toBe(true);
    await flush();
    await flush();
    expect(onLost).not.toHaveBeenCalled();
    guard.destroy(null);
  });

  it('still reports real device loss', async () => {
    const guard = createDeviceLostGuard();
    const d = lostDevice(false);
    const onLost = vi.fn();
    guard.watch(d.device, onLost);
    d.lose({ reason: 'unknown', message: 'GPU process crashed' } as GPUDeviceLostInfo);
    await flush();
    expect(onLost).toHaveBeenCalledWith(expect.objectContaining({ reason: 'unknown' }));
  });

  it('raises requested adapter limits (ORT set) and optional features on the resilient device', async () => {
    const requestDevice = vi.fn(async () => ({ lost: new Promise(() => {}) }));
    const adapter = {
      features: new Set(['shader-f16', 'subgroups']),
      limits: {
        maxBufferSize: 4e9,
        maxStorageBufferBindingSize: 2e9,
        maxStorageBuffersPerShaderStage: 10,
        maxComputeWorkgroupStorageSize: 32768,
        maxComputeInvocationsPerWorkgroup: 1024,
      },
      info: { vendor: 'nvidia', architecture: '', device: '', description: 'RTX' },
      requestDevice,
    };
    vi.stubGlobal('navigator', { gpu: { requestAdapter: vi.fn(async () => adapter) } });
    await requestResilientWebGpuDevice({
      optionalFeatures: ['subgroups'],
      adapterLimits: ORT_WEBGPU_ADAPTER_LIMITS,
    });
    expect(requestDevice).toHaveBeenCalledWith({
      requiredFeatures: ['shader-f16', 'subgroups'],
      requiredLimits: {
        maxBufferSize: 4e9,
        maxStorageBufferBindingSize: 2e9,
        maxStorageBuffersPerShaderStage: 10,
        maxComputeWorkgroupStorageSize: 32768,
        maxComputeInvocationsPerWorkgroup: 1024,
      },
    });
  });
});

describe('Slice 1 P0a — kernel/worker-harness non-poisoning controlChain', () => {
  function harness(runner: ModelRunner) {
    const posted: Array<Record<string, unknown>> = [];
    const scope: WorkerScopeLike = { onmessage: null, postMessage: (m) => posted.push(m as Record<string, unknown>) };
    const h = createWorkerHarness(runner, { scope });
    const send = (data: Record<string, unknown>) => h.handleMessage({ data } as MessageEvent);
    return { h, posted, send };
  }

  const waitForAbort = (signal: AbortSignal) =>
    new Promise<never>((_, reject) => signal.addEventListener('abort', () => reject(new Error('stopped'))));

  it("still posts 'aborted' when runner.abort() rejects, and accepts the next run", async () => {
    const runner = {
      run: vi.fn(async (ctx: { signal: AbortSignal }) => {
        await waitForAbort(ctx.signal);
        return { outputs: {} };
      }),
      abort: vi.fn(async () => {
        throw new Error('abort hook exploded');
      }),
      dispose: vi.fn(),
    };
    const { h, posted, send } = harness(runner as unknown as ModelRunner);
    const running = send({ id: 1, type: 'run' });
    await send({ id: 2, type: 'abort' });
    await running;
    expect(posted).toContainEqual({ id: 2, type: 'aborted' });
    expect(h.isAborting()).toBe(false);
    expect(h.isRunning()).toBe(false);

    runner.run.mockImplementationOnce(async () => ({ outputs: { ok: true } }));
    await send({ id: 3, type: 'run' });
    expect(posted).toContainEqual(expect.objectContaining({ id: 3, type: 'result', outputs: { ok: true } }));
  });

  it("still posts 'disposed' when runner.dispose() rejects, and re-initializes on the next run", async () => {
    const runner = {
      init: vi.fn(async () => {}),
      run: vi.fn(async () => ({ outputs: {} })),
      dispose: vi.fn(async () => {
        throw new Error('dispose exploded');
      }),
    };
    const { posted, send } = harness(runner as unknown as ModelRunner);
    await send({ id: 1, type: 'run' });
    await send({ id: 2, type: 'dispose' });
    expect(posted).toContainEqual({ id: 2, type: 'disposed' });
    await send({ id: 3, type: 'run' });
    expect(runner.init).toHaveBeenCalledTimes(2);
    expect(posted).toContainEqual(expect.objectContaining({ id: 3, type: 'result' }));
  });

  it('runs control steps queued behind a rejected step, in order', async () => {
    const runner = {
      run: vi.fn(async (ctx: { signal: AbortSignal }) => {
        await waitForAbort(ctx.signal);
        return { outputs: {} };
      }),
      abort: vi.fn(() => {
        throw new Error('sync abort failure');
      }),
      dispose: vi.fn(async () => {}),
    };
    const { h, posted, send } = harness(runner as unknown as ModelRunner);
    const running = send({ id: 1, type: 'run' });
    const abort = send({ id: 2, type: 'abort' });
    const dispose = send({ id: 3, type: 'dispose' });
    await Promise.all([running, abort, dispose]);
    const replies = posted.filter((m) => m.type === 'aborted' || m.type === 'disposed');
    expect(replies).toEqual([
      { id: 2, type: 'aborted' },
      { id: 3, type: 'disposed' },
    ]);
    expect(runner.dispose).toHaveBeenCalledTimes(1);
    expect(h.isAborting()).toBe(false);
  });
});

describe('Slice 1 P0a — web-protein-mpnn ORT 1.31.1 session tuning', () => {
  it('pins ORT 1.31.1 WebGPU EP options and the ORT device-limit set', () => {
    expect(MPNN_WEBGPU_EP_OPTIONS).toEqual({ storageBufferCacheMode: 'simple', validationMode: 'wgpuOnly' });
    expect(ORT_WEBGPU_ADAPTER_LIMITS).toContain('maxStorageBuffersPerShaderStage');
    expect(ORT_WEBGPU_ADAPTER_LIMITS).toContain('maxComputeWorkgroupStorageSize');
  });

  it('uses 1 WASM thread unless cross-origin isolated, then ORT’s min(4, ceil(cores / 2))', () => {
    expect(resolveOrtWasmThreads(false, 16)).toBe(1);
    expect(resolveOrtWasmThreads(true, 16)).toBe(4);
    expect(resolveOrtWasmThreads(true, 4)).toBe(2);
    expect(resolveOrtWasmThreads(true, 1)).toBe(1);
    expect(resolveOrtWasmThreads(true, undefined)).toBe(1);
  });

  it("rewrites only the bare 'webgpu' EP name into a configured option object", () => {
    const device = { label: 'resilient' } as unknown as GPUDevice;
    expect(upgradeWebGpuExecutionProviders(['webgpu', 'wasm'], { ...MPNN_WEBGPU_EP_OPTIONS, device })).toEqual([
      { name: 'webgpu', storageBufferCacheMode: 'simple', validationMode: 'wgpuOnly', device },
      'wasm',
    ]);
    expect(upgradeWebGpuExecutionProviders(['wasm'], MPNN_WEBGPU_EP_OPTIONS)).toEqual(['wasm']);
  });

  it('injects the device + EP options into InferenceSession.create only while loading, then restores it', async () => {
    const calls: unknown[][] = [];
    class FakeSession {
      static async create(...args: unknown[]) {
        calls.push(args);
        return new FakeSession();
      }
    }
    const ort = { InferenceSession: FakeSession };
    const original = FakeSession.create;
    const device = { label: 'resilient' } as unknown as GPUDevice;
    const bytes = new Uint8Array([1, 2, 3]);

    await withWebGpuExecutionProviderOptions(ort, { ...MPNN_WEBGPU_EP_OPTIONS, device }, async () => {
      // proteinmpnn-web OnnxBackend.create: `{...sessionOptions, executionProviders: ['webgpu','wasm']}`
      await ort.InferenceSession.create(bytes, {
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3,
        executionProviders: ['webgpu', 'wasm'],
      });
      // ... and its own WebGPU-to-WASM fallback.
      await ort.InferenceSession.create(bytes, { logSeverityLevel: 3, executionProviders: ['wasm'] });
    });
    expect(calls[0][0]).toBe(bytes);
    expect(calls[0][1]).toEqual({
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
      executionProviders: [
        { name: 'webgpu', storageBufferCacheMode: 'simple', validationMode: 'wgpuOnly', device },
        'wasm',
      ],
    });
    expect(calls[1][1]).toEqual({ logSeverityLevel: 3, executionProviders: ['wasm'] });
    expect(FakeSession.create).toBe(original);

    await expect(
      withWebGpuExecutionProviderOptions(ort, MPNN_WEBGPU_EP_OPTIONS, async () => {
        throw new Error('load failed');
      }),
    ).rejects.toThrow('load failed');
    expect(FakeSession.create).toBe(original);
  });

  it('patches the real onnxruntime-web 1.31.1 InferenceSession factory (writable static create)', async () => {
    const ort = await import('onnxruntime-web');
    expect(ort.env.versions.web).toBe('1.31.1');
    expect(ort.env.versions.common).toBe('1.31.1');
    const desc = Object.getOwnPropertyDescriptor(ort.InferenceSession, 'create');
    expect(desc?.writable).toBe(true);
  });

  it('mpnn.worker.ts injects a resilient GPUDevice and ORT 1.31.1 options with a clean WASM fallback', () => {
    const src = readSrc('src/lib/plugins/web-protein-mpnn/mpnn.worker.ts');
    expect(src).toContain('ONNX Runtime 1.31.1');
    expect(src).toContain('requestResilientWebGpuDevice({');
    expect(src).toContain('adapterLimits: ORT_WEBGPU_ADAPTER_LIMITS');
    expect(src).toContain("if (!webGpuDevice) effectiveBackend = 'wasm';");
    expect(src).toContain('ortMod.env.webgpu.device = webGpuDevice');
    expect(src).toContain('withWebGpuExecutionProviderOptions(');
    expect(src).toContain('{ ...MPNN_WEBGPU_EP_OPTIONS, device: webGpuDevice }');
    expect(src).toContain("sessionOptions: { graphOptimizationLevel: 'all', logSeverityLevel: 3 }");
    expect(src).toContain('ortMod.env.wasm.simd = true');
    expect(src).toContain('numThreads: WASM_NUM_THREADS');
    expect(src).not.toMatch(/numThreads:\s*1\b/);
    expect(src).toContain('deviceLostGuard.destroy(ortWebGpuDevice)');
    expect(src).not.toContain('setTimeout(');
  });
});

describe('Slice 1 P0a — rfd4-proteina post-upload shard release', () => {
  function makeBundle() {
    const coreShard = new ArrayBuffer(4096);
    const encoderShard = new ArrayBuffer(4096);
    new Float32Array(coreShard).fill(1.5);
    new Float32Array(encoderShard).fill(2.5);
    const lazy = (buf: ArrayBuffer, offsetFloats: number, len: number) => ({
      shape: [len],
      dtype: 'F32',
      get data() {
        return new Float32Array(buf, offsetFloats * 4, len);
      },
    });
    const bundle: Rfd4ShardedBundleLike = {
      tensors: {
        'nn.init_repr_factory.feat_creators.2.embedding.weight': lazy(coreShard, 0, 224),
        'nn.cond_factory.feat_creators.2.embedding_T.weight': lazy(coreShard, 224, 64),
        'nn.transformer_layers.0.mhba.qkv.weight': lazy(coreShard, 288, 512),
        'autoencoder.decoder.logit_linear.1.weight': lazy(coreShard, 800, 200),
        'autoencoder.encoder.init_repr_factory.feat_creators.0.embedding.weight': lazy(encoderShard, 0, 224),
        'autoencoder.encoder.transformer_layers.0.w': lazy(encoderShard, 224, 512),
      },
      manifest: {
        shards: [
          { file: 'shard_000.bin', role: 'core' },
          { file: 'encoder_shard_000.bin', role: 'encoder' },
        ],
        tensors: {
          'nn.init_repr_factory.feat_creators.2.embedding.weight': { file: 'shard_000.bin' },
          'nn.cond_factory.feat_creators.2.embedding_T.weight': { file: 'shard_000.bin' },
          'nn.transformer_layers.0.mhba.qkv.weight': { file: 'shard_000.bin' },
          'autoencoder.decoder.logit_linear.1.weight': { file: 'shard_000.bin' },
          'autoencoder.encoder.init_repr_factory.feat_creators.0.embedding.weight': { file: 'encoder_shard_000.bin' },
          'autoencoder.encoder.transformer_layers.0.w': { file: 'encoder_shard_000.bin' },
        },
      },
    };
    return { bundle, coreShard, encoderShard };
  }

  it('detaches CPU-read feat_creators tables, stubs uploaded tensors, and keeps encoder tensors until uploaded', () => {
    const { bundle, coreShard, encoderShard } = makeBundle();
    const t = bundle.tensors;
    const tablesBefore = t['nn.cond_factory.feat_creators.2.embedding_T.weight'];

    const first = releaseRfd4UploadedShardBytes(bundle, { encoderUploaded: false });
    expect(first).toEqual({ released: 2, detached: 3, pendingEncoder: 1 });
    // Same object identity: the generator keeps `bundle.tensors` as `this.weights`.
    expect(bundle.tensors).toBe(t);

    const table = t['nn.cond_factory.feat_creators.2.embedding_T.weight'];
    expect(table).not.toBe(tablesBefore);
    expect(table.shape).toEqual([64]);
    expect(table.data.buffer).not.toBe(coreShard);
    expect(Array.from(table.data.slice(0, 2))).toEqual([1.5, 1.5]);

    expect(() => t['nn.transformer_layers.0.mhba.qkv.weight'].data).toThrow(Rfd4ReleasedWeightError);
    expect(() => t['autoencoder.decoder.logit_linear.1.weight'].data).toThrow(/released from the JS heap/);
    expect(t['nn.transformer_layers.0.mhba.qkv.weight'].shape).toEqual([512]);

    // Lazily-built encoder (getEncoder) can still upload from its shard.
    expect(t['autoencoder.encoder.transformer_layers.0.w'].data.buffer).toBe(encoderShard);
    expect(t['autoencoder.encoder.init_repr_factory.feat_creators.0.embedding.weight'].data.buffer).not.toBe(
      encoderShard,
    );

    const second = releaseRfd4UploadedShardBytes(bundle, { encoderUploaded: true });
    expect(second).toEqual({ released: 1, detached: 0, pendingEncoder: 0 });
    expect(() => t['autoencoder.encoder.transformer_layers.0.w'].data).toThrow(Rfd4ReleasedWeightError);

    // No surviving record still reaches a shard buffer.
    for (const rec of Object.values(t)) {
      let buf: ArrayBufferLike | null = null;
      try {
        buf = rec.data.buffer;
      } catch {
        continue;
      }
      expect(buf).not.toBe(coreShard);
      expect(buf).not.toBe(encoderShard);
    }
    expect(releaseRfd4UploadedShardBytes(bundle, { encoderUploaded: true })).toEqual({
      released: 0,
      detached: 0,
      pendingEncoder: 0,
    });
    expect(releaseRfd4UploadedShardBytes(null, { encoderUploaded: true }).released).toBe(0);
  });

  it('falls back to the autoencoder.encoder. prefix when the manifest is missing', () => {
    const { bundle } = makeBundle();
    bundle.manifest = null;
    expect(releaseRfd4UploadedShardBytes(bundle, { encoderUploaded: false }).pendingEncoder).toBe(1);
  });

  it('rfd4.worker.ts releases after GPU upload and after encoder runs, and re-opens the bundle for a new generator', () => {
    const src = readSrc('src/lib/plugins/rfd4-proteina/rfd4.worker.ts');
    const ensureLoaded = src.slice(src.indexOf('async function ensureLoaded('), src.indexOf('const SAFE_EXAMPLE_CIF_RE'));
    expect(ensureLoaded.indexOf('Rfd4pGeneratorGpu.create(')).toBeLessThan(ensureLoaded.lastIndexOf('releaseUploadedShardBytes();'));
    expect(ensureLoaded).toMatch(/if \(!generator && shardBytesReleased\) \{\s*\/\/[^\n]*\n\s*shardedBundle = null;/);
    expect(src).toContain('generator.weights !== shardedBundle.tensors');
    expect(src).toContain('encoderUploaded: Boolean(generator.encoder)');
    expect(src).toMatch(/candidates\.push\(candidate\);\s*\}\s*\/\/[^\n]*\n\s*releaseUploadedShardBytes\(\);/);
    expect(src).toContain('createWorkerHarness(withShardReleaseRecovery({');
    expect(src).toContain('err instanceof Rfd4ReleasedWeightError');
  });

  it('vendor tripwire: after upload, RFD4 reads only feat_creators tables on the CPU and the encoder only its own shards', () => {
    const vendor = (rel: string) => readSrc(`static/vendor/rfd4-proteina/${rel}`);
    const features = vendor('shared/rfd4p/features.js');
    const weightCalls = features.split('\n').filter((l) => l.includes('weight(weights, `'));
    expect(weightCalls.length).toBeGreaterThan(0);
    for (const line of weightCalls) expect(line).toMatch(/feat_creators|\$\{p\}\.embedding_/);
    expect(features).toContain('const p = `${prefix}.feat_creators.${idx}`;');
    expect(features.match(/weights\[/g)).toHaveLength(1);

    for (const [file, fn] of [
      ['cpu/rfd4p/decoder.js', 'decoderSeqFeatures'],
      ['cpu/rfd4p/encoder.js', 'encoderSeqFeatures'],
    ] as const) {
      const src = vendor(file);
      const start = src.indexOf(`export function ${fn}(`);
      const body = src.slice(start, src.indexOf('\nexport function', start + 1));
      const reads = body.match(/W\(`[^`]*`\)/g) ?? [];
      expect(reads.length).toBeGreaterThan(0);
      for (const r of reads) expect(r).toContain('feat_creators');
    }

    const generate = vendor('webgpu/rfd4p/generate.js');
    for (const line of generate.split('\n').filter((l) => l.includes('this.weights'))) {
      expect(line).toMatch(/this\.weights = resolvedWeights|weights: this\.weights|^\s*this\.weights,\s*$/);
    }
    for (const file of ['webgpu/rfd4p/decoder.js', 'webgpu/rfd4p/encoder.js']) {
      for (const line of vendor(file).split('\n').filter((l) => l.includes('rawWeights'))) {
        expect(line).toMatch(/this\.rawWeights = weights|weights: this\.rawWeights/);
      }
    }
    expect(vendor('webgpu/rfd4p/encoder.js')).toContain("const PREFIX = 'autoencoder.encoder';");
  });
});

describe('Slice 1 P0a — localfold-af2 keeps its GPUDevice across checkpoint switches', () => {
  let originDir = '';
  let originUrl = '';

  beforeAll(() => {
    originDir = resolve(WEB_ROOT, `.vitest-p0a-lf-origin-${process.pid}`);
    rmSync(originDir, { recursive: true, force: true });
    const lfRoot = resolve(originDir, 'vendor/localfold');
    mkdirSync(lfRoot, { recursive: true });
    writeFileSync(resolve(originDir, 'package.json'), '{"type":"module"}\n', 'utf8');
    const inputLink = resolve(lfRoot, 'shared/input');
    mkdirSync(dirname(inputLink), { recursive: true });
    if (!existsSync(inputLink)) symlinkSync(resolve(WEB_ROOT, 'static/vendor/localfold/shared/input'), inputLink);
    const stub = (rel: string, code: string) => {
      const p = resolve(lfRoot, rel);
      mkdirSync(dirname(p), { recursive: true });
      writeFileSync(p, code, 'utf8');
    };
    stub(
      'webgpu/runtime/device.js',
      `export async function requestAlphaFoldDevice(adapter) {
        const g = globalThis.__P0A_LF__;
        let lose;
        const lost = new Promise((r) => { lose = r; });
        const id = g.devices.length + 1;
        const dev = { id, adapter, limits: { maxStorageBufferBindingSize: 1 << 30 }, lost,
          destroy() { g.destroyed.push(id); lose({ reason: 'destroyed', message: '' }); } };
        g.devices.push({ dev, lose });
        return dev;
      }\n`,
    );
    stub(
      'webgpu/runtime/resident.js',
      `export function releaseAllWeights(device) { globalThis.__P0A_LF__.released.push(device.id); }
      export function releaseResidentWeights() {}\n`,
    );
    stub(
      'shared/bundles/manifests/index.js',
      `export const MODEL_BUNDLES = { 'monomer-2': { delta: { base: 'monomer' } } };
      export function graphFamily() { return 'monomer'; }
      export function bundleBaseUrl(m) { return 'https://stub.invalid/bundles/' + m; }
      export async function loadManifest(m) { return { model: m, tensors: {} }; }\n`,
    );
    stub(
      'shared/bundles/http-tensor-store.js',
      `export class HttpTensorStore {
        static async fromManifest() { return new HttpTensorStore(); }
        prefetch() {}
        releaseDecoded() {}
      }\n`,
    );
    stub(
      'shared/bundles/delta-tensor-store.js',
      `export class DeltaTensorStore { constructor() {} releaseDecoded() {} }\n`,
    );
    stub(
      'shared/bundles/alphafold-fixture.js',
      `export class AlphaFoldFixture {
        static fromStore() { return new AlphaFoldFixture(); }
        async embeddingWeights() { return {}; }
        async templateWeights() { return {}; }
        async templateEmbeddingWeights() { return {}; }
        async extraStackWeights() { return {}; }
        async mainStackWeights() { return {}; }
        async structureWeights() { return {}; }
        async confidenceWeights() { return { lddt: {}, pae: {} }; }
        async geometryTables() { return {}; }
        async queryOnlyFeatureTables() { return {}; }
        async tensor() { return new Float32Array(64); }
      }\n`,
    );
    stub(
      'webgpu/input/a3m-features.js',
      `export async function makeA3mFeaturesFor(device, a3mText, tables, options = {}) {
        const lines = String(a3mText).split(/\\r?\\n/).filter((l) => l && !l.startsWith('>'));
        const len = (lines[0] || '').replace(/[^A-Za-z]/g, '').length || 10;
        return Array.from({ length: Number(options.recycles ?? 1) + 1 }, () => ({ length: len }));
      }\n`,
    );
    stub(
      'webgpu/af2/model/monomer.js',
      `export class AlphaFoldMonomerGpu {
        constructor(device) { this.device = device; }
        async predictA3m(a3mText, weights, tables, options = {}) {
          globalThis.__P0A_LF__.predictDevices.push(this.device.id);
          const len = (options.chainLengths || [22]).reduce((a, b) => a + b, 0);
          const atom37 = new Float32Array(len * 37 * 3);
          const atom37Mask = new Float32Array(len * 37);
          for (let i = 0; i < len; i++) { atom37[(i * 37 + 1) * 3] = i * 3.8; atom37Mask[i * 37 + 1] = 1; }
          return {
            final: {
              structure: { atom37, atom37Mask },
              confidence: { plddt: new Float32Array(len).fill(80), ptm: 0.8, predictedAlignedError: new Float32Array(len * len).fill(2) },
            },
            recycles: [{}],
          };
        }
      }\n`,
    );
    stub('webgpu/af2/multimer/model.js', `export class AlphaFoldUnifiedGpu { constructor(device) { this.device = device; } }\n`);
    originUrl = pathToFileURL(originDir).href.replace(/\/+$/, '');
  });

  afterAll(() => {
    vi.unstubAllGlobals();
    delete (globalThis as Record<string, unknown>).__P0A_LF__;
    if (originDir && existsSync(originDir)) rmSync(originDir, { recursive: true, force: true });
  });

  it('switches monomer -> monomer-2 on the same device, recovers from real loss, and destroys only on dispose', async () => {
    const state = {
      devices: [] as Array<{ dev: { id: number }; lose: (info: unknown) => void }>,
      destroyed: [] as number[],
      released: [] as number[],
      predictDevices: [] as number[],
    };
    (globalThis as Record<string, unknown>).__P0A_LF__ = state;
    const messages: Array<Record<string, unknown>> = [];
    const scope = {
      postMessage: (m: Record<string, unknown>) => messages.push(m),
      location: { origin: originUrl },
      onmessage: null as null | ((ev: { data: unknown }) => Promise<void>),
    };
    vi.stubGlobal('self', scope);
    vi.stubGlobal('navigator', {
      userAgent: 'Linux',
      hardwareConcurrency: 8,
      gpu: {
        requestAdapter: vi.fn(async () => ({
          features: new Set<string>(),
          limits: { maxStorageBufferBindingSize: 1 << 30 },
          info: { vendor: 'stub', architecture: '', device: 'stub', description: 'stub' },
        })),
      },
    });
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })));
    vi.resetModules();
    await import('$lib/plugins/localfold-af2/localfold.worker');

    const sequences: SequenceSet = {
      kind: 'SequenceSet',
      id: 'p0a-seqs',
      parentStructureId: 'seq',
      sourceModelId: 'sequence-input',
      candidates: [
        { id: 'c1', sequence: 'MKTAYIAKQRQISFVKSHFSRQ', sourceModelId: 'sequence-input', score: null, globalScore: null },
      ],
    };
    const run = async (id: number, model: string) => {
      await scope.onmessage!({ data: { id, type: 'run', params: { model, msaMode: 'single', recycles: 1 }, inputs: { sequences } } });
      const reply = messages.find((m) => m.id === id && (m.type === 'result' || m.type === 'error'));
      expect(reply?.type, String(reply?.message ?? '')).toBe('result');
    };

    await run(1, 'monomer');
    await run(2, 'monomer-2');
    expect(state.devices).toHaveLength(1);
    expect(state.destroyed).toEqual([]);
    expect(state.released).toContain(1);
    expect(state.predictDevices).toEqual([1, 1]);

    // Real loss (not our destroy): the worker drops the device and acquires a fresh one next run.
    state.devices[0].lose({ reason: 'unknown', message: 'GPU reset' });
    await flush();
    await run(3, 'monomer-2');
    expect(state.devices).toHaveLength(2);
    expect(state.predictDevices).toEqual([1, 1, 2]);

    await scope.onmessage!({ data: { id: 4, type: 'dispose' } });
    expect(messages).toContainEqual({ id: 4, type: 'disposed' });
    expect(state.destroyed).toEqual([2]);
    expect(messages.filter((m) => m.type === 'error')).toEqual([]);
  });

  it('source: no gpuDevice.destroy() on switch; dispose goes through the device-lost guard', () => {
    const src = readSrc('src/lib/plugins/localfold-af2/localfold.worker.ts');
    const ensure = src.slice(src.indexOf('async function ensureModelLoaded('), src.indexOf('await ensureDevice();'));
    expect(ensure).toContain('releaseCurrentModel();');
    expect(ensure).not.toContain('destroy()');
    expect(ensure).not.toContain('gpuDevice = null');
    expect(src).toContain('deviceLostGuard.watch(dev, () => {');
    expect(src).toMatch(/async dispose\(\) \{\s*releaseCurrentModel\(\);\s*deviceLostGuard\.destroy\(gpuDevice\);\s*gpuDevice = null;/);
    const code = src
      .split('\n')
      .filter((l) => !l.trim().startsWith('//'))
      .join('\n');
    expect(code.match(/gpuDevice\??\.destroy\(\)/g)).toBeNull();
  });
});
