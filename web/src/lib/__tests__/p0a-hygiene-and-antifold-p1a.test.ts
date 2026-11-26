import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchCachedBytes, type FetchProgress } from '$lib/engine/opfs-cache';
import { Rfd4ReleasedWeightError, type Rfd4ShardedBundleLike } from '$lib/plugins/rfd4-proteina/lifecycle';
import type { StructureArtifact } from '$lib/engine/types';

const WEB_ROOT = resolve(import.meta.dirname, '../../..');
const vendorUrl = (rel: string) => pathToFileURL(resolve(WEB_ROOT, 'static/vendor', rel)).href;
const flush = () => new Promise<void>((r) => setTimeout(r, 0));

const BUFFER_USAGE = {
  MAP_READ: 1,
  MAP_WRITE: 2,
  COPY_SRC: 4,
  COPY_DST: 8,
  UNIFORM: 64,
  STORAGE: 128,
};

// -----------------------------------------------------------------------------
// Hoisted mocks for worker tests (rfd4, mpnn, antifold)
// -----------------------------------------------------------------------------

const ORIGIN = 'http://p0a-test.local';

const workerMocks = vi.hoisted(() => {
  const rfd4State = {
    bundle: null as Rfd4ShardedBundleLike | null,
    generator: null as Record<string, unknown> | null,
    createCalls: 0,
    generateCalls: 0,
  };
  const mpnnState = {
    loadCalls: 0,
    loadGate: null as Promise<void> | null,
    loadedSessions: [] as Array<{ id: number; disposed: number; backend: string }>,
    sampleCalls: 0,
  };
  const antifoldState = {
    createdEngines: [] as Array<{
      engine: Record<string, unknown>;
      prepareContextCalls: number;
      destroyedContexts: number;
    }>,
  };
  return { rfd4State, mpnnState, antifoldState };
});

// rfd4-proteina worker vendor mocks
vi.mock('http://p0a-test.local/vendor/rfd4-proteina/webgpu/index.js', () => ({
  RFD4P_PRESETS: [],
  UNIFIED_ATOM37_TOKENS: [
    'UNK',
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL',
  ],
  async requestWebGpuDevice() {
    let resolveLost!: (info: GPUDeviceLostInfo) => void;
    const lost = new Promise<GPUDeviceLostInfo>((r) => (resolveLost = r));
    return {
      device: {
        lost,
        pushErrorScope() {},
        async popErrorScope() {
          return null;
        },
        destroy() {
          resolveLost({ reason: 'destroyed', message: '' } as GPUDeviceLostInfo);
        },
      },
    };
  },
  async fetchShardedWeights() {
    const coreShard = new ArrayBuffer(4096);
    const encShard = new ArrayBuffer(4096);
    new Float32Array(coreShard).fill(1.0);
    new Float32Array(encShard).fill(2.0);
    const lazy = (buf: ArrayBuffer, off: number, len: number) => ({
      shape: [len],
      dtype: 'F32',
      get data() {
        return new Float32Array(buf, off * 4, len);
      },
    });
    const bundle: Rfd4ShardedBundleLike = {
      tensors: {
        'nn.init_repr_factory.feat_creators.0.embedding.weight': lazy(coreShard, 0, 64),
        'nn.transformer_layers.0.mhba.qkv.weight': lazy(coreShard, 64, 128),
        'autoencoder.encoder.init_repr_factory.feat_creators.0.embedding.weight': lazy(encShard, 0, 64),
        'autoencoder.encoder.transformer_layers.0.w': lazy(encShard, 64, 128),
      },
      manifest: {
        shards: [
          { file: 'shard_000.bin', role: 'core' },
          { file: 'encoder_shard_000.bin', role: 'encoder' },
        ],
        tensors: {
          'nn.init_repr_factory.feat_creators.0.embedding.weight': { file: 'shard_000.bin' },
          'nn.transformer_layers.0.mhba.qkv.weight': { file: 'shard_000.bin' },
          'autoencoder.encoder.init_repr_factory.feat_creators.0.embedding.weight': {
            file: 'encoder_shard_000.bin',
          },
          'autoencoder.encoder.transformer_layers.0.w': { file: 'encoder_shard_000.bin' },
        },
      },
    };
    workerMocks.rfd4State.bundle = bundle;
    return {
      ...bundle,
      config: {},
      encoderLoaded: true,
      async loadEncoder() {},
    };
  },
  Rfd4pGeneratorGpu: {
    async create(_dev: unknown, tensors: Record<string, unknown>) {
      workerMocks.rfd4State.createCalls++;
      const aas = ['ALA', 'GLY', 'VAL', 'LEU', 'ILE', 'SER', 'THR', 'ASP', 'GLU', 'LYS', 'ARG', 'PHE', 'TYR', 'TRP', 'PRO', 'ASN'];
      const pdbLines = aas.map((aa, i) => {
        const res = String(i + 1).padStart(4);
        const x = (i * 3.8).toFixed(3).padStart(8);
        return [
          `ATOM  ${String(i * 3 + 1).padStart(5)}  N   ${aa} A${res}    ${x}   0.000   0.000  1.00  0.00           N  `,
          `ATOM  ${String(i * 3 + 2).padStart(5)}  CA  ${aa} A${res}    ${x}   1.450   0.000  1.00  0.00           C  `,
          `ATOM  ${String(i * 3 + 3).padStart(5)}  C   ${aa} A${res}    ${x}   1.450   1.500  1.00  0.00           C  `,
        ].join('\n');
      });
      const pdb = [...pdbLines, 'END'].join('\n');
      const gen: Record<string, unknown> = {
        weights: tensors,
        encoder: null as Record<string, unknown> | null,
        trunk: { n: 16, condUniform: true, engine: { releaseActivations() {} } },
        decoder: { na: 16, fdim: 64, engine: { releaseActivations() {} } },
        releaseActivations() {},
        async generate() {
          workerMocks.rfd4State.generateCalls++;
          // Simulate lazy encoder upload on first partial / inverse-folding generate()
          gen.encoder = { na: 16, fdim: 64, engine: { releaseActivations() {} } };
          return {
            n: 16,
            stepsRun: 60,
            nsteps: 60,
            logits: new Float32Array(16 * 20).fill(1),
            residueType: Int32Array.from({ length: 16 }, (_, i) => (i % 15) + 1),
            structure: { atoms: [] },
            pdb,
            cif: '',
          };
        },
        destroy() {},
      };
      workerMocks.rfd4State.generator = gen;
      return gen;
    },
  },
  buildBatch() {
    return { batch: { n: 16 }, atoms: [] };
  },
  parseCifToAtomArray() {
    return {
      res_name: ['ALA', 'GLY'],
      chain_id: ['A', 'A'],
      res_id: [1, 2],
      ins_code: ['', ''],
      atom_name: ['CA', 'CA'],
      coord: [[0, 0, 0], [3.8, 0, 0]],
    };
  },
  decodeTextOrZstd: (b: Uint8Array) => new TextDecoder().decode(b),
  parseConditionSpecYaml: () => ({}),
  formatSampleLocalLatents: () => ({}),
  processSampleToAtoms: () => ({ atoms: [] }),
  toCifString: () => '',
  toPdbString: () => '',
}));

vi.mock('http://p0a-test.local/vendor/rfd4-proteina/shared/rfd4p/pipeline/cif.js', () => ({
  parseCifBlock: () => ({ atom_site: {} }),
  buildBondsForAtomArray: () => {},
}));

vi.mock('http://p0a-test.local/vendor/rfd4-proteina/shared/rfd4p/pipeline/ccd.js', () => ({
  componentsNeedingCcd: () => [],
  fetchCcdComponents: async () => ({ blocks: new Map() }),
}));

vi.mock('http://p0a-test.local/vendor/rfd4-proteina/shared/rfd4p/pipeline/prepare.js', () => ({
  convertMseToMet: () => {},
  assignIonCharges: () => {},
}));

vi.mock('http://p0a-test.local/vendor/rfd4-proteina/shared/rfd4p/pipeline/spec.js', () => ({
  selectConditionAtoms: () => new Uint8Array(0),
}));

// web-protein-mpnn worker mocks
vi.mock('proteinmpnn-web', () => ({
  modelManifestURL: (family: string, name: string) => `https://mpnn.test/${family}/${name}/manifest.json`,
  parsePDB: () => ({
    coordinates: new Float32Array(2 * 4 * 3),
    sequence: 'AG',
    chainIds: ['A', 'A'],
    residueIndices: [1, 2],
    insertionCodes: ['', ''],
  }),
  ProteinMPNN: {
    async load(_url: string, opts: { backend?: string }) {
      workerMocks.mpnnState.loadCalls++;
      await workerMocks.mpnnState.loadGate;
      const rec = { id: workerMocks.mpnnState.loadCalls, disposed: 0, backend: opts.backend ?? 'auto' };
      workerMocks.mpnnState.loadedSessions.push(rec);
      return {
        manifest: { atomMode: 'backbone' },
        async prepare(parsed: Record<string, unknown>) {
          return {
            structure: parsed,
            async sample() {
              workerMocks.mpnnState.sampleCalls++;
              return { sequence: 'AG', score: 1.1, globalScore: 1.1 };
            },
          };
        },
        async dispose() {
          rec.disposed++;
        },
      };
    },
  },
}));

vi.mock('onnxruntime-web', () => ({
  env: { logLevel: 'warning', wasm: { simd: false }, webgpu: { device: null } },
  InferenceSession: { create: async () => ({}) },
}));

vi.mock('onnxruntime-web/webgpu', () => ({
  env: { logLevel: 'warning', wasm: { simd: false }, webgpu: { device: null } },
  InferenceSession: { create: async () => ({}) },
}));

// antifold-esmif1 worker vendor bridge to REAL static/vendor/antifold-esmif1/ files
vi.mock('http://p0a-test.local/vendor/antifold-esmif1/webgpu/index.js?v=3', async () => {
  const real = await import(vendorUrl('antifold-esmif1/webgpu/index.js'));
  return {
    ...real,
    async createGVPTransformerEngine(opts: Record<string, unknown>) {
      const engine = await real.createGVPTransformerEngine(opts);
      const record = { engine, prepareContextCalls: 0, destroyedContexts: 0 };
      const origPrepare = engine.prepareContext.bind(engine);
      engine.prepareContext = async (...args: unknown[]) => {
        record.prepareContextCalls++;
        const ctx = await origPrepare(...args);
        const origDestroy = ctx.destroy.bind(ctx);
        ctx.destroy = () => {
          if (!ctx.destroyed) record.destroyedContexts++;
          origDestroy();
        };
        return ctx;
      };
      workerMocks.antifoldState.createdEngines.push(record);
      return engine;
    },
  };
});

vi.mock('http://p0a-test.local/vendor/antifold-esmif1/shared/index.js', async () => {
  return await import(vendorUrl('antifold-esmif1/shared/index.js'));
});

// -----------------------------------------------------------------------------
// Part A.1 & A.2: opfs-cache.ts abort-then-retry marker race + late-joiner guard
// -----------------------------------------------------------------------------

interface MemFile {
  data: Uint8Array;
  locked: boolean;
}

function createControllableOpfs(opts: { onRemoveEntry?: (name: string) => Promise<void> | void } = {}) {
  const files = new Map<string, MemFile>();
  const opsLog: string[] = [];
  const notFound = () => new DOMException('missing', 'NotFoundError');
  const busy = () => new DOMException('locked', 'NoModificationAllowedError');

  const fileHandle = (name: string) => ({
    kind: 'file',
    async getFile() {
      const f = files.get(name);
      if (!f) throw notFound();
      const copy = f.data.slice();
      return { size: copy.byteLength, arrayBuffer: async () => copy.buffer };
    },
    async createSyncAccessHandle() {
      const f = files.get(name);
      if (!f || f.locked) throw busy();
      f.locked = true;
      opsLog.push(`lock:${name}`);
      return {
        read(buf: Uint8Array, o?: { at: number }) {
          const at = o?.at ?? 0;
          const n = Math.max(0, Math.min(buf.byteLength, f.data.byteLength - at));
          buf.set(f.data.subarray(at, at + n));
          return n;
        },
        write(buf: Uint8Array, o?: { at: number }) {
          const at = o?.at ?? 0;
          if (at + buf.byteLength > f.data.byteLength) {
            const grown = new Uint8Array(at + buf.byteLength);
            grown.set(f.data);
            f.data = grown;
          }
          f.data.set(buf, at);
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
          f.locked = false;
          opsLog.push(`close:${name}`);
        },
      };
    },
  });

  const dir = {
    async getFileHandle(name: string, o?: { create?: boolean }) {
      opsLog.push(`open:${name}:create=${Boolean(o?.create)}`);
      if (!files.has(name)) {
        if (!o?.create) throw notFound();
        files.set(name, { data: new Uint8Array(0), locked: false });
      }
      return fileHandle(name);
    },
    async removeEntry(name: string) {
      opsLog.push(`remove:start:${name}`);
      await opts.onRemoveEntry?.(name);
      const f = files.get(name);
      if (!f) throw notFound();
      if (f.locked) throw busy();
      files.delete(name);
      opsLog.push(`remove:done:${name}`);
    },
  };

  const root = {
    async getDirectoryHandle(name: string) {
      if (name !== 'denovo-weights') throw notFound();
      return dir;
    },
  };
  vi.stubGlobal('navigator', { storage: { getDirectory: async () => root } });
  return { files, opsLog };
}

describe('P0a hygiene — engine/opfs-cache.ts', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('serializes per-cacheKey OPFS sink teardown so an aborted stream cannot delete a retry .incomplete marker', async () => {
    let releaseMarkerRemove!: () => void;
    let markerRemoveStarted!: () => void;
    const markerRemovePromise = new Promise<void>((r) => (releaseMarkerRemove = r));
    const markerStartedPromise = new Promise<void>((r) => (markerRemoveStarted = r));
    let markerRemovals = 0;

    const opfs = createControllableOpfs({
      async onRemoveEntry(name) {
        if (name === 'race-key.incomplete') {
          markerRemovals++;
          if (markerRemovals === 1) {
            markerRemoveStarted();
            await markerRemovePromise;
          }
        }
      },
    });

    let stream1Ctrl!: ReadableStreamDefaultController<Uint8Array>;
    let stream2Ctrl!: ReadableStreamDefaultController<Uint8Array>;
    let fetchCount = 0;

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        fetchCount++;
        if (fetchCount === 1) {
          return new Response(
            new ReadableStream<Uint8Array>({
              start(c) {
                stream1Ctrl = c;
                c.enqueue(new Uint8Array([1, 2]));
              },
            }),
            { status: 200, headers: { 'content-length': '5' } },
          );
        }
        return new Response(
          new ReadableStream<Uint8Array>({
            start(c) {
              stream2Ctrl = c;
              c.enqueue(new Uint8Array([10, 20]));
            },
          }),
          { status: 200, headers: { 'content-length': '3' } },
        );
      }),
    );

    const ctrl1 = new AbortController();
    const p1 = fetchCachedBytes('race-key', 'https://example.test/race.bin', undefined, ctrl1.signal);
    await vi.waitFor(() => expect(opfs.files.get('race-key')?.data.byteLength).toBe(2));

    // Abort last waiter and immediately retry the same cacheKey + URL.
    ctrl1.abort(new Error('retry now'));
    const p2 = fetchCachedBytes('race-key', 'https://example.test/race.bin');
    await expect(p1).rejects.toThrow('retry now');

    // Sink 1's marker removal is in flight and paused.
    await markerStartedPromise;
    // Give p2 enough ticks to reach openOpfsStreamSink: it MUST wait for sink 1's teardown.
    await flush();
    await flush();
    expect(opfs.opsLog).not.toContain('open:race-key.incomplete:create=true' + ':after-remove');
    expect(opfs.files.get('race-key')?.locked ?? false).toBe(false);

    // Let sink 1 finish removing its marker, then let sink 2 stream chunk 1 and hold before commit.
    releaseMarkerRemove();
    await vi.waitFor(() => expect(opfs.files.get('race-key')?.data.byteLength).toBe(2));
    // Sink 2's .incomplete marker MUST still be intact mid-stream!
    expect(opfs.files.has('race-key.incomplete')).toBe(true);
    expect(Array.from(opfs.files.get('race-key')?.data ?? [])).toEqual([10, 20]);

    // Verify sink 1's remove:done happened strictly before sink 2 opened the file for streaming.
    const sink1DoneIdx = opfs.opsLog.indexOf('remove:done:race-key.incomplete');
    const sink2LockIdx = opfs.opsLog.lastIndexOf('lock:race-key');
    expect(sink1DoneIdx).toBeGreaterThan(-1);
    expect(sink2LockIdx).toBeGreaterThan(sink1DoneIdx);

    // Finish stream 2.
    stream2Ctrl.enqueue(new Uint8Array([30]));
    stream2Ctrl.close();
    const bytes = await p2;
    expect(Array.from(bytes)).toEqual([10, 20, 30]);
    expect(opfs.files.has('race-key.incomplete')).toBe(false);
    expect(Array.from(opfs.files.get('race-key')?.data ?? [])).toEqual([10, 20, 30]);
    void stream1Ctrl;
  });

  it('guards throwing late-joiner onProgress(load.last) so leave() and load.waiting never leak', async () => {
    let streamCtrl!: ReadableStreamDefaultController<Uint8Array>;
    const abortSignals: AbortSignal[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init?: RequestInit) => {
        abortSignals.push(init!.signal!);
        return new Response(
          new ReadableStream<Uint8Array>({
            start(c) {
              streamCtrl = c;
              c.enqueue(new Uint8Array([1, 2]));
            },
          }),
          { status: 200, headers: { 'content-length': '4' } },
        );
      }),
    );

    const progress1: FetchProgress[] = [];
    const ctrl1 = new AbortController();
    const p1 = fetchCachedBytes('late-join-throw', 'https://example.test/lj.bin', (p) => progress1.push(p), ctrl1.signal);
    await vi.waitFor(() => expect(progress1.length).toBe(1));

    // Join mid-download with a throwing callback AND an abort signal.
    const ctrl2 = new AbortController();
    const throwingProgress = vi.fn(() => {
      throw new Error('stale UI component unmounted');
    });
    const p2 = fetchCachedBytes('late-join-throw', 'https://example.test/lj.bin', throwingProgress, ctrl2.signal);
    expect(throwingProgress).toHaveBeenCalledTimes(1);

    // Aborting caller 1 keeps shared download alive while caller 2 is still waiting.
    ctrl1.abort(new Error('cancel 1'));
    await expect(p1).rejects.toThrow('cancel 1');
    expect(abortSignals[0].aborted).toBe(false);

    // Aborting caller 2 MUST abort the shared download (proving load.waiting did not leak).
    ctrl2.abort(new Error('cancel 2'));
    await expect(p2).rejects.toThrow('cancel 2');
    expect(abortSignals[0].aborted).toBe(true);

    // Also verify throwing late-joiner still receives final bytes when neither caller aborts.
    let streamCtrl2!: ReadableStreamDefaultController<Uint8Array>;
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        return new Response(
          new ReadableStream<Uint8Array>({
            start(c) {
              streamCtrl2 = c;
              c.enqueue(new Uint8Array([5, 6]));
            },
          }),
          { status: 200, headers: { 'content-length': '4' } },
        );
      }),
    );
    const seenA: FetchProgress[] = [];
    const a = fetchCachedBytes('late-join-ok', 'https://example.test/lj2.bin', (p) => seenA.push(p));
    await vi.waitFor(() => expect(seenA.length).toBe(1));
    const b = fetchCachedBytes('late-join-ok', 'https://example.test/lj2.bin', () => {
      throw new Error('boom');
    });
    streamCtrl2.enqueue(new Uint8Array([7, 8]));
    streamCtrl2.close();
    const [resA, resB] = await Promise.all([a, b]);
    expect(Array.from(resA)).toEqual([5, 6, 7, 8]);
    expect(resB).toBe(resA);
    void streamCtrl;
  });
});

// -----------------------------------------------------------------------------
// Part A.3 & A.4: rfd4.worker.ts numDesigns===1 shard release & mpnn.worker.ts live device
// -----------------------------------------------------------------------------

describe('P0a hygiene — rfd4.worker.ts and mpnn.worker.ts', () => {
  beforeEach(() => {
    workerMocks.rfd4State.bundle = null;
    workerMocks.rfd4State.generator = null;
    workerMocks.rfd4State.createCalls = 0;
    workerMocks.rfd4State.generateCalls = 0;
    workerMocks.mpnnState.loadCalls = 0;
    workerMocks.mpnnState.loadGate = null;
    workerMocks.mpnnState.loadedSessions = [];
    workerMocks.mpnnState.sampleCalls = 0;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('rfd4.worker.ts releases lazily-loaded encoder shard bytes after numDesigns === 1 inverse-folding / partial run', async () => {
    const messages: Array<Record<string, unknown>> = [];
    const scope = {
      postMessage: (m: Record<string, unknown>) => messages.push(m),
      location: { origin: ORIGIN },
      onmessage: null as null | ((ev: { data: unknown }) => Promise<void>),
    };
    vi.stubGlobal('self', scope);
    vi.stubGlobal('navigator', { userAgent: 'Linux', hardwareConcurrency: 4 });
    vi.resetModules();
    await import('$lib/plugins/rfd4-proteina/rfd4.worker');

    const template = {
      kind: 'StructureArtifact',
      id: 'tpl-1',
      label: 'tpl',
      sourceModelId: 'pdb-input',
      length: 16,
      sequence: 'AGVLISSTDEKRFYWP',
      pdbText:
        'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n' +
        'ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C  \n' +
        'ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00  0.00           C  \nEND\n',
      caCoords: new Float32Array(48),
      backbone4Coords: new Float32Array(192),
    } as unknown as StructureArtifact;

    await scope.onmessage!({
      data: {
        id: 1,
        type: 'run',
        params: { mode: 'inverse-folding', numDesigns: 1, nsteps: 60, seed: 42 },
        inputs: { structure: template },
      },
    });

    const reply = messages.find((m) => m.id === 1 && (m.type === 'result' || m.type === 'error'));
    expect(reply?.type, String(reply?.message ?? '')).toBe('result');
    expect(workerMocks.rfd4State.createCalls).toBe(1);
    expect(workerMocks.rfd4State.generateCalls).toBe(1);

    // Core non-CPU tensors AND lazily-built encoder tensors MUST both be released in numDesigns===1!
    const tensors = workerMocks.rfd4State.bundle!.tensors;
    expect(() => tensors['nn.transformer_layers.0.mhba.qkv.weight'].data).toThrow(
      /was released from the JS heap after GPU upload/,
    );
    expect(() => tensors['autoencoder.encoder.transformer_layers.0.w'].data).toThrow(
      /was released from the JS heap after GPU upload/,
    );
    void Rfd4ReleasedWeightError;
  });

  it('mpnn.worker.ts does not reuse a cached session whose WebGPU device was lost mid-load', async () => {
    const devices: Array<{ lose: (msg: string) => void; destroyCalls: number }> = [];
    const adapter = {
      features: new Set<string>(['subgroups']),
      limits: {
        maxBufferSize: 1 << 30,
        maxStorageBufferBindingSize: 1 << 30,
        maxStorageBuffersPerShaderStage: 10,
        maxComputeWorkgroupStorageSize: 32768,
        maxComputeInvocationsPerWorkgroup: 1024,
      },
      info: { vendor: 'stub', architecture: '', device: 'stub', description: 'stub' },
      async requestDevice() {
        let resolveLost!: (info: GPUDeviceLostInfo) => void;
        const lost = new Promise<GPUDeviceLostInfo>((r) => (resolveLost = r));
        const rec = {
          destroyCalls: 0,
          lose: (message: string) => resolveLost({ reason: 'unknown', message } as GPUDeviceLostInfo),
        };
        devices.push(rec);
        return {
          features: new Set(['subgroups']),
          limits: { ...adapter.limits },
          lost,
          destroy() {
            rec.destroyCalls++;
          },
        };
      },
    };

    const messages: Array<Record<string, unknown>> = [];
    const scope = {
      postMessage: (m: Record<string, unknown>) => messages.push(m),
      location: { origin: ORIGIN },
      onmessage: null as null | ((ev: { data: unknown }) => Promise<void>),
    };
    vi.stubGlobal('self', scope);
    vi.stubGlobal('crossOriginIsolated', false);
    vi.stubGlobal('navigator', {
      userAgent: 'Linux',
      hardwareConcurrency: 4,
      gpu: { requestAdapter: vi.fn(async () => adapter) },
    });
    vi.stubGlobal('WebAssembly', { Suspending: function () {} });
    vi.resetModules();
    await import('$lib/plugins/web-protein-mpnn/mpnn.worker');

    const structure = {
      kind: 'StructureArtifact',
      id: 'mpnn-s',
      label: 'mpnn',
      sourceModelId: 'pdb-input',
      length: 2,
      sequence: 'AG',
      pdbText:
        'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n' +
        'ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C  \n' +
        'ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00  0.00           C  \nEND\n',
      caCoords: new Float32Array(6),
      backbone4Coords: new Float32Array(24),
    } as unknown as StructureArtifact;

    let releaseLoad!: () => void;
    workerMocks.mpnnState.loadGate = new Promise<void>((r) => (releaseLoad = r));

    const run1 = scope.onmessage!({
      data: {
        id: 1,
        type: 'run',
        params: { family: 'solublempnn', modelName: 'v_48_020', numSequences: 1 },
        inputs: { structure },
      },
    });
    await vi.waitFor(() => expect(devices).toHaveLength(1));

    // Lose device1 MID-LOAD while ProteinMPNN.load is still pending!
    devices[0].lose('GPU reset mid-load');
    await flush();
    releaseLoad();
    await run1;
    workerMocks.mpnnState.loadGate = null;
    expect(workerMocks.mpnnState.loadCalls).toBe(1);

    // Run 2 MUST acquire device2 and reload a fresh model (never reuse dead device1's session)!
    await scope.onmessage!({
      data: {
        id: 2,
        type: 'run',
        params: { family: 'solublempnn', modelName: 'v_48_020', numSequences: 1 },
        inputs: { structure },
      },
    });
    expect(devices).toHaveLength(2);
    expect(workerMocks.mpnnState.loadCalls).toBe(2);
    expect(workerMocks.mpnnState.loadedSessions[0].disposed).toBe(1);

    // Run 3 with live device2 DOES reuse cached session #2.
    await scope.onmessage!({
      data: {
        id: 3,
        type: 'run',
        params: { family: 'solublempnn', modelName: 'v_48_020', numSequences: 1 },
        inputs: { structure },
      },
    });
    expect(devices).toHaveLength(2);
    expect(workerMocks.mpnnState.loadCalls).toBe(2);

    // Run 4 & 5 on WASM fallback (no navigator.gpu, ortWebGpuDevice === null): WASM session is cached and reused.
    vi.stubGlobal('navigator', { userAgent: 'Linux', hardwareConcurrency: 4 });
    await scope.onmessage!({
      data: {
        id: 4,
        type: 'run',
        params: { family: 'solublempnn', modelName: 'v_48_020', numSequences: 1 },
        inputs: { structure },
      },
    });
    expect(workerMocks.mpnnState.loadCalls).toBe(3);
    await scope.onmessage!({
      data: {
        id: 5,
        type: 'run',
        params: { family: 'solublempnn', modelName: 'v_48_020', numSequences: 1 },
        inputs: { structure },
      },
    });
    expect(workerMocks.mpnnState.loadCalls).toBe(3);
  });
});

// -----------------------------------------------------------------------------
// Part B: Functional WebGPU emulator + synthetic safetensors bundle for antifold-esmif1
// -----------------------------------------------------------------------------

function buildSyntheticAntiFoldBundle(variant: 'antifold' | 'esmif1' = 'antifold') {
  const S = 8;
  const V = 4;
  const Se = 8;
  const Ve = 2;
  const Sm = 8;
  const Vm = 4;
  const Sf = 8;
  const Vf = 4;
  const D = 16;
  const F = 32;

  const tensors: Record<string, { shape: number[]; data: Float32Array }> = {};
  let seed = 0x12345678;
  const nextVal = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return ((seed / 4294967296) - 0.5) * 0.25;
  };
  const add = (name: string, shape: number[], biasVal?: number) => {
    const n = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = biasVal ?? nextVal();
    tensors[name] = { shape, data };
  };
  const addLinear = (prefix: string, outDim: number, inDim: number, bias = true) => {
    add(`${prefix}.weight`, [outDim, inDim]);
    if (bias) add(`${prefix}.bias`, [outDim]);
  };
  const addNorm = (prefix: string, dim: number, gain = false) => {
    add(`${prefix}.${gain ? 'gain' : 'weight'}`, [dim], 1.0);
    add(`${prefix}.bias`, [dim], 0.0);
  };
  const addGvp = (prefix: string, sOut: number, sIn: number, vOut: number, vIn: number, gate: boolean) => {
    addLinear(`${prefix}.wh`, vOut, vIn, false);
    addLinear(`${prefix}.ws`, sOut, sIn + vOut, true);
    addLinear(`${prefix}.wv`, vOut, vOut, false);
    if (gate) addLinear(`${prefix}.wg`, vOut, sOut, true);
  };

  // GVP encoder
  addGvp('encoder.gvp_encoder.embed_graph.embed_node.0', S, 7, V, 3, false);
  addNorm('encoder.gvp_encoder.embed_graph.embed_node.1.scalar_norm', S);
  addLinear('encoder.gvp_encoder.embed_graph.embed_confidence', S, 16, true);
  addGvp('encoder.gvp_encoder.embed_graph.embed_edge.0', Se, 34, Ve, 1, false);
  addNorm('encoder.gvp_encoder.embed_graph.embed_edge.1.scalar_norm', Se);
  for (let i = 0; i < 4; i++) {
    const b = `encoder.gvp_encoder.encoder_layers.${i}`;
    addGvp(`${b}.conv.message_func.0`, Sm, 2 * S + Se, Vm, 2 * V + Ve, true);
    addGvp(`${b}.conv.message_func.1`, Sm, Sm, Vm, Vm, true);
    addGvp(`${b}.conv.message_func.2`, S, Sm, V, Vm, false);
    addNorm(`${b}.norm.0.scalar_norm`, S);
    addGvp(`${b}.ff_func.0`, Sf, S, Vf, V, true);
    addGvp(`${b}.ff_func.1`, S, Sf, V, Vf, false);
    addNorm(`${b}.norm.1.scalar_norm`, S);
  }

  // Encoder transformer
  add('encoder.embed_tokens.weight', [35, D]);
  addLinear('encoder.embed_dihedrals.node_embedding', D, 6, true);
  addNorm('encoder.embed_dihedrals.norm_nodes', D, true);
  addLinear('encoder.embed_gvp_output', D, S + V * 3, true);
  addLinear('encoder.embed_confidence', D, 16, true);
  addLinear('encoder.embed_gvp_input_features', D, 15, true);
  for (let i = 0; i < 8; i++) {
    const p = `encoder.layers.${i}`;
    addNorm(`${p}.self_attn_layer_norm`, D);
    for (const k of ['q_proj', 'k_proj', 'v_proj', 'out_proj']) addLinear(`${p}.self_attn.${k}`, D, D);
    addNorm(`${p}.final_layer_norm`, D);
    addLinear(`${p}.fc1`, F, D);
    addLinear(`${p}.fc2`, D, F);
  }
  addNorm('encoder.layer_norm', D);

  // Decoder transformer
  add('decoder.embed_tokens.weight', [35, D]);
  for (let i = 0; i < 8; i++) {
    const p = `decoder.layers.${i}`;
    addNorm(`${p}.self_attn_layer_norm`, D);
    for (const k of ['q_proj', 'k_proj', 'v_proj', 'out_proj']) addLinear(`${p}.self_attn.${k}`, D, D);
    addNorm(`${p}.encoder_attn_layer_norm`, D);
    for (const k of ['q_proj', 'k_proj', 'v_proj', 'out_proj']) addLinear(`${p}.encoder_attn.${k}`, D, D);
    addNorm(`${p}.final_layer_norm`, D);
    addLinear(`${p}.fc1`, F, D);
    addLinear(`${p}.fc2`, D, F);
  }
  addNorm('decoder.layer_norm', D);
  addLinear('decoder.output_projection', 35, D);

  // Serialize to safetensors
  const header: Record<string, unknown> = {};
  let byteOffset = 0;
  const chunks: Uint8Array[] = [];
  for (const [name, t] of Object.entries(tensors)) {
    const bytes = new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength);
    header[name] = { dtype: 'F32', shape: t.shape, data_offsets: [byteOffset, byteOffset + bytes.byteLength] };
    chunks.push(bytes);
    byteOffset += bytes.byteLength;
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const safetensors = new Uint8Array(8 + headerBytes.byteLength + byteOffset);
  new DataView(safetensors.buffer).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  safetensors.set(headerBytes, 8);
  let pos = 8 + headerBytes.byteLength;
  for (const c of chunks) {
    safetensors.set(c, pos);
    pos += c.byteLength;
  }

  const sha = createHash('sha256').update(safetensors).digest('hex');
  const manifestTensors: Record<string, unknown> = {};
  for (const [name, t] of Object.entries(tensors)) {
    manifestTensors[name] = { dtype: 'F32', shape: t.shape, shard: 'shard_000.safetensors' };
  }
  const manifest = {
    format: 'gvp-transformer-safetensors-v1',
    model: variant,
    precision: 'fp16',
    num_tensors: Object.keys(tensors).length,
    total_bytes: safetensors.byteLength,
    shards: [{ file: 'shard_000.safetensors', byteLength: safetensors.byteLength, sha256: sha }],
    tensors: manifestTensors,
  };

  return {
    tensorOrder: Object.keys(tensors),
    manifestJson: JSON.stringify(manifest),
    shardBuffer: safetensors.buffer,
  };
}

function createFunctionalWebGpuDevice(weightTensorNames: string[]) {
  interface FakeBuffer {
    id: number;
    size: number;
    data: ArrayBuffer;
    destroyed: boolean;
    mappedRange: ArrayBuffer | null;
    destroy(): void;
    mapAsync(): Promise<void>;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
  }

  let nextBufId = 1;
  const allBuffers: FakeBuffer[] = [];
  const weightNameByBuffer = new Map<FakeBuffer, string>();
  const matmulWeightCounts = new Map<string, number>();
  let uploadIdx = 0;
  let readbackCount = 0;
  let submitCount = 0;

  const execShader = (code: string, buffers: FakeBuffer[], groups: [number, number, number]) => {
    const f = (idx: number) => new Float32Array(buffers[idx].data);
    const i32 = (idx: number) => new Int32Array(buffers[idx].data);

    if (code.includes('var<workgroup> at:array<f32,256>;')) {
      const mMatch = /if\(r<(\d+)u&&base/.exec(code)!;
      const nMatch = /if\(c<(\d+)u&&base/.exec(code)!;
      const kMatch = /base<(\d+)u;base/.exec(code)!;
      const M = Number(mMatch[1]);
      const N = Number(nMatch[1]);
      const K = Number(kMatch[1]);
      const hasBias = code.includes('+b[c]');
      const wName = weightNameByBuffer.get(buffers[1]);
      if (wName) matmulWeightCounts.set(wName, (matmulWeightCounts.get(wName) ?? 0) + 1);
      const a = f(0);
      const w = f(1);
      const b = f(2);
      const y = f(4);
      for (let r = 0; r < M; r++) {
        for (let c = 0; c < N; c++) {
          let s = 0;
          for (let k = 0; k < K; k++) s += a[r * K + k] * w[c * K + k];
          y[r * N + c] = hasBias ? s + b[c] : s;
        }
      }
      return;
    }

    if (code.includes('var<workgroup> tmp:array<f32,256>;')) {
      const D = Number(/let row=g\.x\*(\d+)u;/.exec(code)![1]);
      const dihedral = code.includes(`/${D - 1}.0+0.000001`);
      const M = groups[0];
      const x = f(0);
      const w = f(1);
      const b = f(2);
      const y = f(3);
      for (let r = 0; r < M; r++) {
        const off = r * D;
        let sum = 0;
        for (let j = 0; j < D; j++) sum += x[off + j];
        const mean = sum / D;
        let v = 0;
        for (let j = 0; j < D; j++) {
          const d = x[off + j] - mean;
          v += d * d;
        }
        const den = dihedral
          ? Math.sqrt(v / (D - 1) + 0.000001) + 0.000001
          : Math.sqrt(v / D + 0.00001);
        for (let j = 0; j < D; j++) y[off + j] = ((x[off + j] - mean) / den) * w[j] + b[j];
      }
      return;
    }

    if (code.includes('let mask=select(0.0,1.0,dot(v,v)+0.0001>0.0002)')) {
      const M = Number(/if\(r>=(\d+)u\)/.exec(code)![1]);
      const D = Number(/let i=r\*(\d+)u\+j\*3u;/.exec(code)![1]);
      const V = D / 3;
      const x = f(0);
      const y = f(1);
      for (let r = 0; r < M; r++) {
        let sum = 0;
        let count = 0;
        for (let j = 0; j < V; j++) {
          const idx = r * D + j * 3;
          const n = x[idx] * x[idx] + x[idx + 1] * x[idx + 1] + x[idx + 2] * x[idx + 2] + 0.0001;
          if (n > 0.0002) {
            sum += n;
            count += 1;
          }
        }
        const den = Math.sqrt(sum / (0.0001 + count) + 0.0001);
        for (let j = 0; j < V; j++) {
          const idx = r * D + j * 3;
          const n = x[idx] * x[idx] + x[idx + 1] * x[idx + 1] + x[idx + 2] * x[idx + 2] + 0.0001;
          const m = n > 0.0002 ? 1 : 0;
          y[idx] = (m * x[idx]) / den;
          y[idx + 1] = (m * x[idx + 1]) / den;
          y[idx + 2] = (m * x[idx + 2]) / den;
        }
      }
      return;
    }

    if (code.includes('x2[col]!=0')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const S = Number(/let col=i%(\d+)u;/.exec(code)![1]);
      const M = Number(/%\s*(\d+)u;let h=/.exec(code)![1]);
      const D = Number(/x0\[r\*(\d+)u/.exec(code)![1]);
      const C = D / 8;
      const causal = code.includes('||col>r');
      const q = f(0);
      const k = f(1);
      const pad = i32(2);
      const y = f(3);
      const scale = 1 / Math.sqrt(C);
      for (let idx = 0; idx < n; idx++) {
        const col = idx % S;
        const r = Math.floor(idx / S) % M;
        const h = Math.floor(idx / (M * S));
        let s = 0;
        for (let j = 0; j < C; j++) s += q[r * D + h * C + j] * k[col * D + h * C + j];
        y[idx] = pad[col] !== 0 || (causal && col > r) ? -3.402823e38 : s * scale;
      }
      return;
    }

    if (code.includes('var<workgroup> t:array<f32,256>;')) {
      const S = Number(/let off=g\.x\*(\d+)u;/.exec(code)![1]);
      const R = groups[0];
      const x = f(0);
      const y = f(1);
      for (let r = 0; r < R; r++) {
        const off = r * S;
        let m = -3.402823e38;
        for (let k = 0; k < S; k++) m = Math.max(m, x[off + k]);
        let s = 0;
        for (let k = 0; k < S; k++) s += Math.exp(x[off + k] - m);
        for (let k = 0; k < S; k++) y[off + k] = Math.exp(x[off + k] - m) / s;
      }
      return;
    }

    if (code.includes('sum+=x0[(h*')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const D = Number(/let r=i\/(\d+)u;/.exec(code)![1]);
      const C = D / 8;
      const M = Number(/\*(\d+)u\+r\)/.exec(code)![1]);
      const S = Number(/var j=0u;j<(\d+)u;/.exec(code)![1]);
      const p = f(0);
      const v = f(1);
      const y = f(2);
      for (let idx = 0; idx < n; idx++) {
        const r = Math.floor(idx / D);
        const d = idx % D;
        const h = Math.floor(d / C);
        let sum = 0;
        for (let j = 0; j < S; j++) sum += p[(h * M + r) * S + j] * v[j * D + d];
        y[idx] = sum;
      }
      return;
    }

    if (code.includes('let off=r*47u;')) {
      const N = Number(/if\(r>=(\d+)u\)/.exec(code)![1]);
      const coords = f(0);
      const mask = i32(1);
      const conf = f(2);
      const y = f(3);
      for (let r = 0; r < N; r++) {
        const off = r * 47;
        for (let j = 0; j < 6; j++) y[off + j] = (r + j) * 0.05;
        y[off + 6] = mask[r];
        for (let j = 7; j < 31; j++) {
          const v = coords[r * 9 + (j % 9)];
          const fv = Number.isFinite(v) ? v : 0;
          y[off + j] = (fv % 10) * 0.1 + 0.1;
        }
        for (let j = 0; j < 16; j++) {
          const z = (conf[r] - j / 15) * 16;
          y[off + 31 + j] = Math.exp(-z * z);
        }
      }
      return;
    }

    if (code.includes('var ds:array<f32,30>;')) {
      const N = Number(/if\(src>=(\d+)u\)/.exec(code)![1]);
      const K = Number(/j<(\d+)u;j\+\+\)\{ds/.exec(code)![1]);
      const coords = f(0);
      const mask = i32(1);
      const yI32 = i32(2);
      for (let src = 0; src < N; src++) {
        const dists: Array<[number, number]> = [];
        if (mask[src] !== 0) {
          for (let dst = 0; dst < N; dst++) {
            if (mask[dst] === 0) continue;
            const dx = (Number.isFinite(coords[src * 9 + 3]) ? coords[src * 9 + 3] : 0) - (Number.isFinite(coords[dst * 9 + 3]) ? coords[dst * 9 + 3] : 0);
            const dy = (Number.isFinite(coords[src * 9 + 4]) ? coords[src * 9 + 4] : 0) - (Number.isFinite(coords[dst * 9 + 4]) ? coords[dst * 9 + 4] : 0);
            const dz = (Number.isFinite(coords[src * 9 + 5]) ? coords[src * 9 + 5] : 0) - (Number.isFinite(coords[dst * 9 + 5]) ? coords[dst * 9 + 5] : 0);
            dists.push([Math.sqrt(dx * dx + dy * dy + dz * dz + 1e-8), dst]);
          }
          dists.sort((a, b) => a[0] - b[0]);
        }
        for (let j = 0; j < K; j++) {
          const off = (src * K + j) * 2;
          const hit = dists[j];
          yI32[off] = hit ? src : -1;
          yI32[off + 1] = hit ? hit[1] : -1;
        }
      }
      return;
    }

    if (code.includes('let off=e*37u;')) {
      const E = Number(/if\(e>=(\d+)u\)/.exec(code)![1]);
      const idx = i32(1);
      const y = f(2);
      for (let e = 0; e < E; e++) {
        const off = e * 37;
        const src = idx[e * 2];
        const dst = idx[e * 2 + 1];
        if (src < 0 || dst < 0) {
          y.fill(0, off, off + 37);
          continue;
        }
        for (let j = 0; j < 34; j++) y[off + j] = Math.exp(-0.1 * j) * 0.5;
        y[off + 34] = 0.6;
        y[off + 35] = 0.8;
        y[off + 36] = 0.0;
      }
      return;
    }

    if (code.includes('y[i]=x0[i]+x1[i];')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const a = f(0);
      const b = f(1);
      const y = f(2);
      for (let i = 0; i < n; i++) y[i] = a[i] + b[i];
      return;
    }

    if (code.includes('y[i]=max(x0[i],0.0);')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const a = f(0);
      const y = f(1);
      for (let i = 0; i < n; i++) y[i] = Math.max(a[i], 0);
      return;
    }

    if (code.includes('if(j<')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const D = Number(/let r=i\/(\d+)u;/.exec(code)![1]);
      const widths = [...code.matchAll(/r\*(\d+)u\+j/g)].map((m) => Number(m[1]));
      const y = f(widths.length);
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / D);
        const j = i % D;
        let off = 0;
        for (let k = 0; k < widths.length; k++) {
          if (j < off + widths[k]) {
            y[i] = f(k)[r * widths[k] + j - off];
            break;
          }
          off += widths[k];
        }
      }
      return;
    }

    if (code.includes('y[i]=x0[(i/')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const m = /\(i\/(\d+)u\)\*(\d+)u\+(\d+)u\+i%/.exec(code)!;
      const width = Number(m[1]);
      const D = Number(m[2]);
      const start = Number(m[3]);
      const x0 = f(0);
      const y = f(1);
      for (let i = 0; i < n; i++) y[i] = x0[Math.floor(i / width) * D + start + (i % width)];
      return;
    }

    if (code.includes('let v=(i/3u)%')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const V = Number(/let v=\(i\/3u\)%(\d+)u;/.exec(code)![1]);
      const x0 = f(0);
      const y = f(1);
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / (V * 3));
        const v = Math.floor(i / 3) % V;
        const k = i % 3;
        y[i] = x0[(r * 3 + k) * V + v];
      }
      return;
    }

    if (code.includes('let k=(i/')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const V = Number(/let k=\(i\/(\d+)u\)%3u;/.exec(code)![1]);
      const x0 = f(0);
      const y = f(1);
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / (V * 3));
        const k = Math.floor(i / V) % 3;
        const v = i % V;
        y[i] = x0[(r * V + v) * 3 + k];
      }
      return;
    }

    if (code.includes('sqrt(dot(v,v)+0.00000001);')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const x0 = f(0);
      const y = f(1);
      for (let i = 0; i < n; i++) {
        const a = x0[i * 3];
        const b = x0[i * 3 + 1];
        const c = x0[i * 3 + 2];
        y[i] = Math.sqrt(a * a + b * b + c * c + 1e-8);
      }
      return;
    }

    if (code.includes('/(1.0+exp(-x1[i/3u]));')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const v = f(0);
      const g = f(1);
      const y = f(2);
      for (let i = 0; i < n; i++) y[i] = v[i] / (1 + Math.exp(-g[Math.floor(i / 3)]));
      return;
    }

    if (code.includes('let a=r*9u+k*3u;')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const D = Number(/let r=i\/(\d+)u;/.exec(code)![1]);
      const x0 = f(0);
      const x1 = f(1);
      const y = f(2);
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / D);
        const j = Math.floor((i % D) / 3);
        const k = i % 3;
        const base = r * D + j * 3;
        const a = r * 9 + k * 3;
        y[i] = x0[base] * x1[a] + x0[base + 1] * x1[a + 1] + x0[base + 2] * x1[a + 2];
      }
      return;
    }

    if (code.includes('let r=x1[(i/')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const D = Number(/x1\[\(i\/(\d+)u\)\*2u\+/.exec(code)![1]);
      const which = Number(/\*2u\+(\d+)u\];/.exec(code)![1]);
      const x0 = f(0);
      const x1 = i32(1);
      const y = f(2);
      for (let i = 0; i < n; i++) {
        const r = x1[Math.floor(i / D) * 2 + which];
        y[i] = r >= 0 ? x0[r * D + (i % D)] : 0;
      }
      return;
    }

    if (code.includes('var c=0u;for(var e=0u;')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const E = Number(/e<(\d+)u;/.exec(code)![1]);
      const x0 = i32(0);
      const y = f(1);
      for (let i = 0; i < n; i++) {
        let c = 0;
        for (let e = 0; e < E; e++) if (x0[e * 2 + 1] >= 0 && x0[e * 2 + 1] < i) c++;
        y[i] = c;
      }
      return;
    }

    if (code.includes('let a=u32(x1[i]);var k=0u;')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const E = Number(/e<(\d+)u;/.exec(code)![1]);
      const x0 = i32(0);
      const x1 = f(1);
      const y = f(2);
      for (let i = 0; i < n; i++) {
        const a = Math.trunc(x1[i]);
        let k = 0;
        for (let e = 0; e < E; e++) {
          if (x0[e * 2 + 1] === i) {
            y[a + k] = e;
            k++;
          }
        }
      }
      return;
    }

    if (code.includes('let a=u32(x2[r]);let b=u32(x2[r+1u]);')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const D = Number(/let r=i\/(\d+)u;/.exec(code)![1]);
      const x0 = f(0);
      const x1 = f(1);
      const x2 = f(2);
      const y = f(3);
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / D);
        const j = i % D;
        const a = Math.trunc(x2[r]);
        const b = Math.trunc(x2[r + 1]);
        let sum = 0;
        for (let k = a; k < b; k++) sum += x0[Math.trunc(x1[k]) * D + j];
        y[i] = sum / Math.max(b - a, 1);
      }
      return;
    }

    if (code.includes('y[i]=x1[u32(x0[i/')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const D = Number(/x0\[i\/(\d+)u\]/.exec(code)![1]);
      const scale = Math.sqrt(D);
      const x0 = i32(0);
      const x1 = f(1);
      const y = f(2);
      for (let i = 0; i < n; i++) y[i] = x1[x0[Math.floor(i / D)] * D + (i % D)] * scale;
      return;
    }

    if (code.includes('let a=f32(r+2u)*exp(')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const D = Number(/let r=i\/(\d+)u;/.exec(code)![1]);
      const half = D / 2;
      const zeroPad = code.includes('select(x0[i]+pos,0.0,x1[r]!=0)');
      const x0 = f(0);
      const x1 = i32(1);
      const y = f(2);
      const step = -Math.log(10000) / (half - 1);
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / D);
        const j = i % D;
        const a = (r + 2) * Math.exp((j % half) * step);
        const pos = x1[r] !== 0 ? 0 : j < half ? Math.sin(a) : Math.cos(a);
        y[i] = zeroPad && x1[r] !== 0 ? 0 : x0[i] + pos;
      }
      return;
    }

    if (code.includes('let r=i/22u;let j=i%22u;')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const aa = /array<u32,20>\(([^)]+)\)/.exec(code)![1].split(',').map((s) => parseInt(s, 10));
      const x0 = f(0);
      const x1 = i32(1);
      const y = f(2);
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / 22);
        const j = i % 22;
        let m = -3.402823e38;
        for (let t = 0; t < 35; t++) m = Math.max(m, x0[r * 35 + t]);
        let s = 0;
        for (let t = 0; t < 35; t++) s += Math.exp(x0[r * 35 + t] - m);
        if (j === 0) {
          y[i] = x0[r * 35 + x1[r]] - m - Math.log(s);
          continue;
        }
        let cm = -3.402823e38;
        for (let t = 0; t < 20; t++) cm = Math.max(cm, x0[r * 35 + aa[t]]);
        let cs = 0;
        for (let t = 0; t < 20; t++) cs += Math.exp(x0[r * 35 + aa[t]] - cm);
        if (j >= 2) {
          y[i] = Math.exp(x0[r * 35 + aa[j - 2]] - cm) / cs;
          continue;
        }
        let entropy = 0;
        for (let t = 0; t < 20; t++) {
          const prob = Math.exp(x0[r * 35 + aa[t]] - cm) / cs;
          entropy -= prob * Math.log(Math.max(prob, 1e-30));
        }
        y[i] = Math.exp(entropy);
      }
      return;
    }

    if (code.includes('var sum=0.0;var count=0.0;for(var r=0u;')) {
      const L = Number(/r<(\d+)u;r\+\+/.exec(code)![1]);
      const x0 = f(0);
      const x1 = i32(1);
      const x2 = i32(2);
      const y = f(3);
      for (let i = 0; i < 8; i++) {
        let sum = 0;
        let count = 0;
        for (let r = 0; r < L; r++) {
          if (x1[r] !== 0 && (i % 4 < 2 || x2[r] !== 0)) {
            sum += x0[r * 22];
            count += 1;
          }
        }
        const mean = sum / Math.max(count, 1);
        y[i] = i >= 4 ? count : i % 2 === 1 ? Math.exp(-mean) : mean;
      }
      return;
    }

    if (code.includes('let base=u32(x1[0]);let temp=x1[1];let uDraw=x1[2];')) {
      const aa = /array<u32,20>\(([^)]+)\)/.exec(code)![1].split(',').map((s) => parseInt(s, 10));
      const x0 = f(0);
      const x1 = f(1);
      const y = f(2);
      const base = Math.trunc(x1[0]);
      const temp = x1[1];
      const uDraw = x1[2];
      let m = -3.402823e38;
      for (let j = 0; j < 20; j++) m = Math.max(m, x0[base + aa[j]]);
      let sum = 0;
      for (let j = 0; j < 20; j++) sum += Math.exp((x0[base + aa[j]] - m) / temp);
      let c = 0;
      let token = aa[19];
      for (let j = 0; j < 20; j++) {
        c += Math.exp((x0[base + aa[j]] - m) / temp) / sum;
        if (uDraw < c) {
          token = aa[j];
          break;
        }
      }
      y[0] = token;
      return;
    }

    if (code.includes('if(i%2u==1u){y[i]=f32(arg);return;}')) {
      const n = Number(/if\(i<(\d+)u\)/.exec(code)![1]);
      const aa = /array<u32,20>\(([^)]+)\)/.exec(code)![1].split(',').map((s) => parseInt(s, 10));
      const temp = Number(/\/([^)]+)\)\/sum;if\(x1/.exec(code)![1]);
      const x0 = f(0);
      const x1 = f(1);
      const y = f(2);
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / 2);
        let m = -3.402823e38;
        let arg = 0;
        for (let j = 0; j < 20; j++) {
          const v = x0[r * 35 + aa[j]];
          if (v > m) {
            m = v;
            arg = j;
          }
        }
        if (i % 2 === 1) {
          y[i] = arg;
          continue;
        }
        let sum = 0;
        for (let j = 0; j < 20; j++) sum += Math.exp((x0[r * 35 + aa[j]] - m) / temp);
        let c = 0;
        let token = 19;
        for (let j = 0; j < 20; j++) {
          c += Math.exp((x0[r * 35 + aa[j]] - m) / temp) / sum;
          if (x1[r] < c) {
            token = j;
            break;
          }
        }
        y[i] = token;
      }
      return;
    }

    throw new Error(`Unhandled WGSL shader in test emulator:\n${code}`);
  };

  let lostResolve!: (info: GPUDeviceLostInfo) => void;
  const lost = new Promise<GPUDeviceLostInfo>((r) => (lostResolve = r));
  const device = {
    features: new Set<string>(),
    limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 30 },
    lost,
    destroy() {
      lostResolve({ reason: 'destroyed', message: '' } as GPUDeviceLostInfo);
    },
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    },
    createBuffer({ size, usage }: { size: number; usage: number }) {
      const buf: FakeBuffer = {
        id: nextBufId++,
        size,
        data: new ArrayBuffer(size),
        destroyed: false,
        mappedRange: null,
        destroy() {
          buf.destroyed = true;
        },
        async mapAsync() {
          readbackCount++;
          buf.mappedRange = buf.data.slice(0);
        },
        getMappedRange() {
          return buf.mappedRange!;
        },
        unmap() {
          buf.mappedRange = null;
        },
      };
      allBuffers.push(buf);
      void usage;
      return buf;
    },
    createShaderModule({ code }: { code: string }) {
      return { code, getCompilationInfo: async () => ({ messages: [] }) };
    },
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    async createComputePipelineAsync({ compute }: { compute: { module: { code: string } } }) {
      return { code: compute.module.code };
    },
    createBindGroup({ entries }: { entries: Array<{ binding: number; resource: { buffer: FakeBuffer } }> }) {
      return entries.slice().sort((a, b) => a.binding - b.binding).map((e) => e.resource.buffer);
    },
    createCommandEncoder() {
      const cmds: Array<
        | { kind: 'dispatch'; code: string; buffers: FakeBuffer[]; groups: [number, number, number] }
        | { kind: 'copy'; src: FakeBuffer; srcOff: number; dst: FakeBuffer; dstOff: number; size: number }
      > = [];
      return {
        beginComputePass() {
          let code = '';
          let buffers: FakeBuffer[] = [];
          return {
            setPipeline(p: { code: string }) {
              code = p.code;
            },
            setBindGroup(_idx: number, bg: FakeBuffer[]) {
              buffers = bg;
            },
            dispatchWorkgroups(x = 1, y = 1, z = 1) {
              cmds.push({ kind: 'dispatch', code, buffers, groups: [x, y, z] });
            },
            end() {},
          };
        },
        copyBufferToBuffer(src: FakeBuffer, srcOff: number, dst: FakeBuffer, dstOff: number, size: number) {
          cmds.push({ kind: 'copy', src, srcOff, dst, dstOff, size });
        },
        finish: () => cmds,
      };
    },
    queue: {
      writeBuffer(buffer: FakeBuffer, offset: number, data: Uint8Array) {
        new Uint8Array(buffer.data, offset, data.byteLength).set(data);
        // Track uploaded weight buffers (skip initial ops.zero buffer id=1, and skip temporary buffers during inference).
        if (buffer.id > 1 && uploadIdx < weightTensorNames.length && submitCount === 0) {
          weightNameByBuffer.set(buffer, weightTensorNames[uploadIdx++]);
        }
      },
      submit(batches: Array<Array<Record<string, unknown>>>) {
        submitCount++;
        for (const cmds of batches) {
          for (const cmd of cmds) {
            if (cmd.kind === 'dispatch') {
              execShader(
                cmd.code as string,
                cmd.buffers as FakeBuffer[],
                cmd.groups as [number, number, number],
              );
            } else if (cmd.kind === 'copy') {
              const srcBytes = new Uint8Array((cmd.src as FakeBuffer).data, cmd.srcOff as number, cmd.size as number);
              new Uint8Array((cmd.dst as FakeBuffer).data, cmd.dstOff as number, cmd.size as number).set(srcBytes);
            }
          }
        }
      },
    },
  };

  return {
    device,
    allBuffers,
    matmulCount: (name: string) => matmulWeightCounts.get(name) ?? 0,
    sumMatmulsMatching: (re: RegExp) => {
      let total = 0;
      for (const [k, v] of matmulWeightCounts) if (re.test(k)) total += v;
      return total;
    },
    resetCounters: () => {
      matmulWeightCounts.clear();
      readbackCount = 0;
      submitCount = 1; // ensure subsequent writeBuffer calls aren't mistaken for weight uploads
    },
    getReadbackCount: () => readbackCount,
    getSubmitCount: () => submitCount,
  };
}

const SAMPLE_PDB_SINGLE = [
  'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  ',
  'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  ',
  'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C  ',
  'ATOM      4  N   GLY A   2       3.320   1.510   0.000  1.00 20.00           N  ',
  'ATOM      5  CA  GLY A   2       3.960   2.810   0.000  1.00 20.00           C  ',
  'ATOM      6  C   GLY A   2       5.470   2.700   0.000  1.00 20.00           C  ',
  'ATOM      7  N   VAL A   3       6.100   3.850   0.200  1.00 20.00           N  ',
  'ATOM      8  CA  VAL A   3       7.520   3.950   0.450  1.00 20.00           C  ',
  'ATOM      9  C   VAL A   3       8.200   2.650   0.150  1.00 20.00           C  ',
  'ATOM     10  N   LEU A   4       9.450   2.600   0.550  1.00 20.00           N  ',
  'ATOM     11  CA  LEU A   4      10.200   1.400   0.850  1.00 20.00           C  ',
  'ATOM     12  C   LEU A   4      11.650   1.650   0.500  1.00 20.00           C  ',
  'ATOM     13  N   SER A   5      12.400   0.600   0.750  1.00 20.00           N  ',
  'ATOM     14  CA  SER A   5      13.800   0.800   0.400  1.00 20.00           C  ',
  'ATOM     15  C   SER A   5      14.500  -0.500   0.650  1.00 20.00           C  ',
  'ATOM     16  N   GLU A   6      15.800  -0.350   0.350  1.00 20.00           N  ',
  'ATOM     17  CA  GLU A   6      16.650  -1.500   0.600  1.00 20.00           C  ',
  'ATOM     18  C   GLU A   6      18.000  -1.100   0.100  1.00 20.00           C  ',
  'END',
].join('\n');

const ANTIBODY_PDB = [
  // Heavy chain H: IMGT 25..26 FW1, 27..28 CDR1, 105..106 CDR3
  'ATOM      1  N   GLU H  25       0.000   0.000   0.000  1.00 20.00           N  ',
  'ATOM      2  CA  GLU H  25       1.458   0.000   0.000  1.00 20.00           C  ',
  'ATOM      3  C   GLU H  25       2.009   1.420   0.000  1.00 20.00           C  ',
  'ATOM      4  N   VAL H  26       3.320   1.510   0.000  1.00 20.00           N  ',
  'ATOM      5  CA  VAL H  26       3.960   2.810   0.000  1.00 20.00           C  ',
  'ATOM      6  C   VAL H  26       5.470   2.700   0.000  1.00 20.00           C  ',
  'ATOM      7  N   GLY H  27       6.100   3.850   0.200  1.00 20.00           N  ',
  'ATOM      8  CA  GLY H  27       7.520   3.950   0.450  1.00 20.00           C  ',
  'ATOM      9  C   GLY H  27       8.200   2.650   0.150  1.00 20.00           C  ',
  'ATOM     10  N   PHE H  28       9.450   2.600   0.550  1.00 20.00           N  ',
  'ATOM     11  CA  PHE H  28      10.200   1.400   0.850  1.00 20.00           C  ',
  'ATOM     12  C   PHE H  28      11.650   1.650   0.500  1.00 20.00           C  ',
  'ATOM     13  N   ALA H 105      12.400   0.600   0.750  1.00 20.00           N  ',
  'ATOM     14  CA  ALA H 105      13.800   0.800   0.400  1.00 20.00           C  ',
  'ATOM     15  C   ALA H 105      14.500  -0.500   0.650  1.00 20.00           C  ',
  'ATOM     16  N   ARG H 106      15.800  -0.350   0.350  1.00 20.00           N  ',
  'ATOM     17  CA  ARG H 106      16.650  -1.500   0.600  1.00 20.00           C  ',
  'ATOM     18  C   ARG H 106      18.000  -1.100   0.100  1.00 20.00           C  ',
  // Light chain L: 2 FW residues
  'ATOM     19  N   ASP L   1      20.000   0.000   0.000  1.00 20.00           N  ',
  'ATOM     20  CA  ASP L   1      21.458   0.000   0.000  1.00 20.00           C  ',
  'ATOM     21  C   ASP L   1      22.009   1.420   0.000  1.00 20.00           C  ',
  'ATOM     22  N   ILE L   2      23.320   1.510   0.000  1.00 20.00           N  ',
  'ATOM     23  CA  ILE L   2      23.960   2.810   0.000  1.00 20.00           C  ',
  'ATOM     24  C   ILE L   2      25.470   2.700   0.000  1.00 20.00           C  ',
  'END',
].join('\n');

describe('antifold-esmif1 Phase 1A — encoder & cross-K/V hoisting + fixed-position skip', () => {
  beforeEach(() => {
    workerMocks.antifoldState.createdEngines = [];
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  async function createTestEngine(variant: 'antifold' | 'esmif1' = 'antifold') {
    vi.stubGlobal('GPUBufferUsage', BUFFER_USAGE);
    vi.stubGlobal('GPUMapMode', { READ: 1, WRITE: 2 });
    vi.stubGlobal('GPUShaderStage', { COMPUTE: 4 });
    const bundle = buildSyntheticAntiFoldBundle(variant);
    const gpu = createFunctionalWebGpuDevice(bundle.tensorOrder);
    const { createGVPTransformerEngine } = await import(vendorUrl('antifold-esmif1/webgpu/index.js'));
    const engine = await createGVPTransformerEngine({
      baseUrl: 'https://antifold.test',
      variant,
      precision: 'fp16',
      device: gpu.device,
      read: async (url: string) =>
        url.endsWith('manifest.json')
          ? new TextEncoder().encode(bundle.manifestJson).buffer
          : bundle.shardBuffer.slice(0),
    });
    gpu.resetCounters();
    return { engine, gpu, baseBuffers: engine.memoryUsage().bufferCount };
  }

  it('hoists encode() + all 8 encoder_attn K,V projections across sampleIndependent/sample + score with exact numeric parity', async () => {
    const { engine, gpu, baseBuffers } = await createTestEngine('esmif1');
    const uniforms1 = new Float32Array([0.11, 0.37, 0.72, 0.25, 0.84, 0.55]);
    const uniforms2 = new Float32Array([0.62, 0.19, 0.44, 0.81, 0.09, 0.91]);

    // 1. Reference outputs WITHOUT context
    const refPred = await engine.predict(SAMPLE_PDB_SINGLE);
    const refIndep = await engine.sampleIndependent(SAMPLE_PDB_SINGLE, {
      temperature: 0.3,
      uniforms: uniforms1,
    });
    const refAr = await engine.sample(SAMPLE_PDB_SINGLE, {
      temperature: 0.3,
      uniforms: uniforms2,
    });
    const refScore = await engine.score(SAMPLE_PDB_SINGLE, { sequence: refAr.sequence });
    expect(engine.memoryUsage().transientBuffers).toBe(0);
    expect(engine.memoryUsage().bufferCount).toBe(baseBuffers);

    // 2. Prepare reusable context ONCE
    gpu.resetCounters();
    const ctx = await engine.prepareContext(SAMPLE_PDB_SINGLE);
    expect(ctx.destroyed).toBe(false);
    // Encoder ran once (7 GVP embed + 72 GVP layers + 4 embed + 48 transformer = 131) + 16 cross-attention K/V projections.
    expect(gpu.sumMatmulsMatching(/^encoder\./)).toBe(131);
    expect(gpu.matmulCount('encoder.layers.0.self_attn.q_proj.weight')).toBe(1);
    expect(gpu.matmulCount('encoder.layers.7.fc2.weight')).toBe(1);
    for (let l = 0; l < 8; l++) {
      expect(gpu.matmulCount(`decoder.layers.${l}.encoder_attn.k_proj.weight`)).toBe(1);
      expect(gpu.matmulCount(`decoder.layers.${l}.encoder_attn.v_proj.weight`)).toBe(1);
    }
    expect(gpu.sumMatmulsMatching(/^decoder\.layers\.\d+\.self_attn\./)).toBe(0);
    // Persistent context tensors survive ops.finish(): 1 padding + 16 K/V = 17 buffers, 0 transient!
    expect(engine.memoryUsage().transientBuffers).toBe(0);
    expect(engine.memoryUsage().bufferCount).toBe(baseBuffers + 17);

    // 3. Reuse context across predict, sampleIndependent x2, sample x2, score x2
    gpu.resetCounters();
    const ctxPred = await engine.predict(SAMPLE_PDB_SINGLE, { context: ctx });
    const ctxIndep1 = await engine.sampleIndependent(SAMPLE_PDB_SINGLE, {
      temperature: 0.3,
      uniforms: uniforms1,
      context: ctx,
    });
    const ctxIndep2 = await engine.sampleIndependent(SAMPLE_PDB_SINGLE, {
      temperature: 0.3,
      uniforms: uniforms2,
      context: ctx,
    });
    const ctxAr = await engine.sample(SAMPLE_PDB_SINGLE, {
      temperature: 0.3,
      uniforms: uniforms2,
      context: ctx,
    });
    const ctxScore1 = await engine.score(SAMPLE_PDB_SINGLE, {
      sequence: ctxAr.sequence,
      context: ctx,
    });
    const ctxScore2 = await engine.score(SAMPLE_PDB_SINGLE, {
      sequence: ctxIndep1.sequence,
      context: ctx,
    });
    void ctxIndep2;
    void ctxScore2;

    // ZERO encoder calls and ZERO encoder_attn K/V projections during all 6 reused runs!
    expect(gpu.sumMatmulsMatching(/^encoder\./)).toBe(0);
    expect(gpu.sumMatmulsMatching(/^decoder\.layers\.\d+\.encoder_attn\.(k_proj|v_proj)\.weight$/)).toBe(0);
    // Cross-attention Q + out_proj still ran (each call did decode passes).
    expect(gpu.sumMatmulsMatching(/^decoder\.layers\.\d+\.encoder_attn\.q_proj\.weight$/)).toBeGreaterThan(0);

    // Exact bit-for-bit parity with uncached runs
    expect(Array.from(ctxPred.logits)).toEqual(Array.from(refPred.logits));
    expect(ctxPred.logLikelihood).toBe(refPred.logLikelihood);
    expect(ctxPred.perplexity).toBe(refPred.perplexity);
    expect(ctxIndep1.sequence).toBe(refIndep.sequence);
    expect(Array.from(ctxIndep1.tokens)).toEqual(Array.from(refIndep.tokens));
    expect(ctxAr.sequence).toBe(refAr.sequence);
    expect(Array.from(ctxAr.tokens)).toEqual(Array.from(refAr.tokens));
    expect(ctxScore1.logLikelihood).toBe(refScore.logLikelihood);
    expect(Array.from(ctxScore1.logits)).toEqual(Array.from(refScore.logits));

    // VRAM never grew across runs (still exact baseBuffers + 17, 0 transient)
    expect(engine.memoryUsage().transientBuffers).toBe(0);
    expect(engine.memoryUsage().bufferCount).toBe(baseBuffers + 17);

    // returnIntermediates: true with context still populates encoder intermediates cleanly
    const ctxIntermediates = await engine.predict(SAMPLE_PDB_SINGLE, {
      returnIntermediates: true,
      context: ctx,
    });
    expect(ctxIntermediates.intermediates?.['encoder.output']).toBeDefined();
    expect(Array.from(ctxIntermediates.logits)).toEqual(Array.from(refPred.logits));

    // Destroy context -> frees all 17 persistent GPU buffers!
    ctx.destroy();
    expect(ctx.destroyed).toBe(true);
    expect(engine.memoryUsage().bufferCount).toBe(baseBuffers);
    ctx.destroy(); // idempotent

    // Using destroyed context rejects cleanly
    await expect(engine.predict(SAMPLE_PDB_SINGLE, { context: ctx })).rejects.toThrow(/destroyed/i);

    // Foreign context from another engine rejects cleanly
    const other = await createTestEngine('esmif1');
    const foreignCtx = await other.engine.prepareContext(SAMPLE_PDB_SINGLE);
    await expect(engine.predict(SAMPLE_PDB_SINGLE, { context: foreignCtx })).rejects.toThrow(
      /Invalid encoder context/i,
    );
    other.engine.destroy();

    // Context created for a different structure rejects cleanly
    const ctx2 = await engine.prepareContext(SAMPLE_PDB_SINGLE);
    const alteredCoordsPdb = SAMPLE_PDB_SINGLE.replace('18.000', '19.500');
    await expect(engine.predict(alteredCoordsPdb, { context: ctx2 })).rejects.toThrow(/match/i);

    // engine.destroy() also destroys owned un-disposed contexts
    engine.destroy();
    expect(ctx2.destroyed).toBe(true);
    expect(engine.memoryUsage().bufferCount).toBe(0);
  });

  it('skips decoder + arSample + readback on fixed[i] when !returnIntermediates while preserving exact parity and returnIntermediates', async () => {
    const { engine, gpu, baseBuffers } = await createTestEngine('esmif1');
    const uniforms = new Float32Array([0.12, 0.45, 0.78, 0.31, 0.66, 0.9]);
    const fixed = { 0: 'M', 2: 'W', 4: 'K' };

    // 1. returnIntermediates: true -> must decode ALL 6 positions and return 6*35 logits
    gpu.resetCounters();
    const withIntermediates = await engine.sample(SAMPLE_PDB_SINGLE, {
      temperature: 0.4,
      uniforms,
      fixed,
      returnIntermediates: true,
    });
    expect(gpu.matmulCount('decoder.output_projection.weight')).toBe(6);
    expect(gpu.getReadbackCount()).toBe(6);
    expect(withIntermediates.logits?.length).toBe(6 * 35);
    // Even in non-context sample(), cross-K/V is precomputed once per layer instead of 6 times!
    expect(gpu.matmulCount('decoder.layers.0.encoder_attn.k_proj.weight')).toBe(1);
    expect(gpu.matmulCount('decoder.layers.7.encoder_attn.v_proj.weight')).toBe(1);

    // 2. !returnIntermediates -> skips decode + readback for fixed positions 0, 2, 4 (runs ONLY 3 times for 1, 3, 5!)
    gpu.resetCounters();
    const skipped = await engine.sample(SAMPLE_PDB_SINGLE, {
      temperature: 0.4,
      uniforms,
      fixed,
      returnIntermediates: false,
    });
    expect(gpu.matmulCount('decoder.output_projection.weight')).toBe(3);
    expect(gpu.getReadbackCount()).toBe(3);
    expect(skipped.logits).toBeUndefined();
    expect(skipped.sequence).toBe(withIntermediates.sequence);
    expect(Array.from(skipped.tokens)).toEqual(Array.from(withIntermediates.tokens));
    expect(skipped.sequence[0]).toBe('M');
    expect(skipped.sequence[2]).toBe('W');
    expect(skipped.sequence[4]).toBe('K');

    // 3. Same exact parity with hoisted context
    const ctx = await engine.prepareContext(SAMPLE_PDB_SINGLE);
    gpu.resetCounters();
    const skippedWithCtx = await engine.sample(SAMPLE_PDB_SINGLE, {
      temperature: 0.4,
      uniforms,
      fixed,
      context: ctx,
    });
    expect(gpu.sumMatmulsMatching(/^encoder\./)).toBe(0);
    expect(gpu.sumMatmulsMatching(/^decoder\.layers\.\d+\.encoder_attn\.(k_proj|v_proj)\.weight$/)).toBe(0);
    expect(gpu.matmulCount('decoder.output_projection.weight')).toBe(3);
    expect(gpu.getReadbackCount()).toBe(3);
    expect(skippedWithCtx.sequence).toBe(withIntermediates.sequence);
    expect(Array.from(skippedWithCtx.tokens)).toEqual(Array.from(withIntermediates.tokens));

    // 4. All positions fixed + !returnIntermediates -> 0 decode calls
    gpu.resetCounters();
    const allFixed = await engine.sample(SAMPLE_PDB_SINGLE, {
      temperature: 0.4,
      uniforms,
      fixed: { 0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G' },
      context: ctx,
    });
    expect(gpu.matmulCount('decoder.output_projection.weight')).toBe(0);
    expect(gpu.getReadbackCount()).toBe(0);
    expect(allFixed.sequence).toBe('ACDEFG');

    ctx.destroy();
    expect(engine.memoryUsage().bufferCount).toBe(baseBuffers);
    engine.destroy();
  });

  it('antifold.worker.ts prepares encoder context once per backbone across numSequences and multi-backbone collections and destroys in finally', async () => {
    const bundle = buildSyntheticAntiFoldBundle('antifold');
    const createdGpus: Array<ReturnType<typeof createFunctionalWebGpuDevice>> = [];

    const adapter = {
      features: new Set<string>(),
      limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 30 },
      info: { vendor: 'stub', architecture: '', device: 'stub', description: 'stub' },
      async requestDevice() {
        const g = createFunctionalWebGpuDevice(bundle.tensorOrder);
        createdGpus.push(g);
        return g.device;
      },
    };

    const messages: Array<Record<string, unknown>> = [];
    const scope = {
      postMessage: (m: Record<string, unknown>) => messages.push(m),
      location: { origin: ORIGIN },
      onmessage: null as null | ((ev: { data: unknown }) => Promise<void>),
    };

    vi.stubGlobal('self', scope);
    vi.stubGlobal('GPUBufferUsage', BUFFER_USAGE);
    vi.stubGlobal('GPUMapMode', { READ: 1, WRITE: 2 });
    vi.stubGlobal('GPUShaderStage', { COMPUTE: 4 });
    vi.stubGlobal('navigator', {
      userAgent: 'Linux',
      hardwareConcurrency: 4,
      gpu: { requestAdapter: vi.fn(async () => adapter) },
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (url.endsWith('manifest.json')) return new Response(bundle.manifestJson, { status: 200 });
        if (url.endsWith('shard_000.safetensors')) return new Response(bundle.shardBuffer.slice(0), { status: 200 });
        return new Response('', { status: 404 });
      }),
    );

    vi.resetModules();
    await import('$lib/plugins/antifold-esmif1/antifold.worker');

    const abStruct = {
      kind: 'StructureArtifact',
      id: 'ab-1',
      label: 'ab-1',
      sourceModelId: 'pdb-input',
      length: 8,
      sequence: 'EVGFARDI',
      pdbText: ANTIBODY_PDB,
      caCoords: new Float32Array(24),
      backbone4Coords: new Float32Array(96),
      molecularContext: { chainTypes: { H: 'heavy', L: 'light' } },
    } as unknown as StructureArtifact;

    // 1. Single backbone, independent mode, numSequences=3 -> 1 prepareContext (1 encode), not 6 encodes!
    await scope.onmessage!({
      data: {
        id: 1,
        type: 'run',
        params: {
          variant: 'antifold',
          precision: 'fp16',
          samplingMode: 'independent',
          numSequences: 3,
          targetChain: 'H',
          regions: 'CDR1',
          numbering: 'imgt',
          seed: 42,
        },
        inputs: { structure: abStruct },
      },
    });
    const res1 = messages.find((m) => m.id === 1 && (m.type === 'result' || m.type === 'error'));
    expect(res1?.type, String(res1?.message ?? '')).toBe('result');
    expect(workerMocks.antifoldState.createdEngines).toHaveLength(1);
    const rec = workerMocks.antifoldState.createdEngines[0];
    expect(rec.prepareContextCalls).toBe(1);
    expect(rec.destroyedContexts).toBe(1);
    expect(createdGpus[0].matmulCount('encoder.layers.0.self_attn.q_proj.weight')).toBe(1);
    expect(createdGpus[0].matmulCount('decoder.layers.0.encoder_attn.k_proj.weight')).toBe(1);
    // 3 sampleIndependent + 3 score = 6 decode passes
    expect(createdGpus[0].matmulCount('decoder.output_projection.weight')).toBe(6);
    const seqSet1 = (res1 as { outputs: { sequences: { candidates: Array<{ sequence: string; score: number }> } } })
      .outputs.sequences;
    expect(seqSet1.candidates).toHaveLength(3);
    for (const c of seqSet1.candidates) {
      expect(c.sequence).toHaveLength(8);
      // Framework residues H25(E), H26(V), H105(A), H106(R), L1(D), L2(I) remain untouched!
      expect(c.sequence.slice(0, 2)).toBe('EV');
      expect(c.sequence.slice(4, 8)).toBe('ARDI');
      expect(Number.isFinite(c.score)).toBe(true);
    }

    // 2. Single backbone, autoregressive mode, numSequences=2, targetChain=H (6 residues: 4 fixed FW/CDR3, 2 CDR1)
    // -> 1 prepareContext + only 2 unfixed positions decoded per sequence (+ 2 score passes = 2*2 + 2 = 6)!
    createdGpus[0].resetCounters();
    await scope.onmessage!({
      data: {
        id: 2,
        type: 'run',
        params: {
          variant: 'antifold',
          precision: 'fp16',
          samplingMode: 'autoregressive',
          numSequences: 2,
          targetChain: 'H',
          regions: 'CDR1',
          numbering: 'imgt',
          seed: 99,
        },
        inputs: { structure: abStruct },
      },
    });
    const res2 = messages.find((m) => m.id === 2 && (m.type === 'result' || m.type === 'error'));
    expect(res2?.type, String(res2?.message ?? '')).toBe('result');
    expect(rec.prepareContextCalls).toBe(2);
    expect(rec.destroyedContexts).toBe(2);
    expect(createdGpus[0].matmulCount('encoder.layers.0.self_attn.q_proj.weight')).toBe(1);
    expect(createdGpus[0].matmulCount('decoder.layers.0.encoder_attn.k_proj.weight')).toBe(1);
    expect(createdGpus[0].matmulCount('decoder.output_projection.weight')).toBe(6);
    const seqSet2 = (res2 as { outputs: { sequences: { candidates: Array<{ sequence: string; score: number }> } } })
      .outputs.sequences;
    expect(seqSet2.candidates).toHaveLength(2);
    for (const c of seqSet2.candidates) {
      expect(c.sequence).toHaveLength(8);
      expect(c.sequence.slice(0, 2)).toBe('EV');
      expect(c.sequence.slice(4, 8)).toBe('ARDI');
      expect(Number.isFinite(c.score)).toBe(true);
    }

    // 3. Multi-backbone collection (2 backbones x 2 sequences = 2 prepareContext calls, both destroyed)
    createdGpus[0].resetCounters();
    const ab2 = { ...abStruct, id: 'ab-2', designKey: 'd2', designIndex: 1 };
    const colStruct = {
      ...abStruct,
      designKey: 'd1',
      designIndex: 0,
      collection: {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id: 'col-1',
        familyKey: 'fam-1',
        collectionKind: 'independent-candidates',
        requestedCount: 2,
        complete: true,
        items: [
          { ...abStruct, designKey: 'd1', designIndex: 0 },
          ab2,
        ],
        keys: ['d1', 'd2'],
      },
    } as unknown as StructureArtifact;

    await scope.onmessage!({
      data: {
        id: 3,
        type: 'run',
        params: {
          variant: 'antifold',
          precision: 'fp16',
          samplingMode: 'independent',
          numSequences: 2,
          targetChain: 'H',
          regions: 'CDR1',
          numbering: 'imgt',
          seed: 7,
        },
        inputs: { structure: colStruct },
      },
    });
    const res3 = messages.find((m) => m.id === 3 && (m.type === 'result' || m.type === 'error'));
    expect(res3?.type, String(res3?.message ?? '')).toBe('result');
    expect(rec.prepareContextCalls).toBe(4);
    expect(rec.destroyedContexts).toBe(4);
    expect(createdGpus[0].matmulCount('encoder.layers.0.self_attn.q_proj.weight')).toBe(2);
    expect(createdGpus[0].matmulCount('decoder.layers.0.encoder_attn.k_proj.weight')).toBe(2);
    // 2 backbones * 2 sequences * (1 sampleIndependent + 1 score) = 8
    expect(createdGpus[0].matmulCount('decoder.output_projection.weight')).toBe(8);

    await scope.onmessage!({ data: { id: 4, type: 'dispose' } });
    expect(messages).toContainEqual({ id: 4, type: 'disposed' });
  });
});
