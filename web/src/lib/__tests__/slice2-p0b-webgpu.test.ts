// Slice 2 (P0b): foldflow2 queue pacing, protpardelle workgroup arrays under 8 KiB, device-gated shader-f16 and
// device-lost resilience for the antifold-esmif1, esmc, protpardelle, lasermpnn and ligandmpnn workers.
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { isWebGpuDeviceLostError } from '$lib/kernel/webgpu-device';
import type { SequenceSet, StructureArtifact } from '$lib/engine/types';

const WEB_ROOT = resolve(import.meta.dirname, '../../..');
const readSrc = (rel: string) => readFileSync(resolve(WEB_ROOT, rel), 'utf8');
const vendorUrl = (rel: string) => pathToFileURL(resolve(WEB_ROOT, 'static/vendor', rel)).href;
const flush = () => new Promise<void>((r) => setTimeout(r, 0));

const BUFFER_USAGE = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };

// Vendor engines behind the workers' `new URL('/vendor/...', self.location.origin)` imports. Each records how it was
// built (the injected device, the f16 flag) and fails its first real call, which is past everything under test.
const stubs = vi.hoisted(() => {
  const state = {
    created: [] as Array<{ kind: string; opts: Record<string, unknown>; destroyed: number }>,
    designError: null as null | (() => Error),
    /** While set, engine factories wait on it: the window in which a device can be lost mid-load. */
    loading: null as Promise<void> | null,
  };
  const stop = () => {
    throw new Error('stub-stop');
  };
  const engine = (kind: string, opts: Record<string, unknown>) => {
    const rec = { kind, opts, destroyed: 0 };
    state.created.push(rec);
    return {
      run: stop,
      sample: stop,
      sampleIndependent: stop,
      score: stop,
      predict: stop,
      design: async () => {
        throw state.designError?.() ?? new Error('stub-stop');
      },
      destroy() {
        rec.destroyed++;
      },
    };
  };
  const factory = (kind: string) => async (opts: Record<string, unknown>) => {
    await state.loading;
    return engine(kind, opts);
  };
  return { state, stop, factory };
});

vi.mock('http://pds.test/vendor/antifold-esmif1/webgpu/index.js?v=3', () => ({
  createGVPTransformerEngine: stubs.factory('antifold'),
}));
vi.mock('http://pds.test/vendor/antifold-esmif1/shared/index.js', () => ({
  parseStructure: stubs.stop,
  prepareInput: stubs.stop,
}));
vi.mock('http://pds.test/vendor/esmc/webgpu/index.js?v=2', () => ({
  createEsmcEngine: stubs.factory('esmc'),
}));
vi.mock('http://pds.test/vendor/protpardelle/webgpu/index.js', () => ({
  createProtpardelleEngine: stubs.factory('protpardelle'),
}));
vi.mock('http://pds.test/vendor/protpardelle/shared/inputs.js', () => ({ chainLayout: stubs.stop, selectPositions: stubs.stop }));
vi.mock('http://pds.test/vendor/protpardelle/shared/structure.js', () => ({ parseStructure: stubs.stop, toPDB: stubs.stop }));
vi.mock('http://pds.test/vendor/protpardelle/shared/sampler.js', () => ({ Replay: class {} }));
vi.mock('http://pds.test/vendor/lasermpnn/webgpu/index.js', () => ({
  createLaserMpnnEngine: stubs.factory('lasermpnn'),
}));
vi.mock('http://pds.test/vendor/lasermpnn/shared/lasermpnn/weights.js', () => ({
  loadWeights: async () => ({ manifest: {}, tensors: {} }),
}));
vi.mock('http://pds.test/vendor/ligandmpnn/webgpu/index.js?v=4', () => ({
  createLigandMpnnEngine: stubs.factory('ligandmpnn'),
}));
vi.mock('http://pds.test/vendor/ligandmpnn/cpu/index.js?v=2', () => ({
  createLigandMpnnCpuEngine: stubs.factory('ligandmpnn-cpu'),
}));

// ---------------------------------------------------------------------------------------------
// 1. foldflow2 GPUAlgebra queue pacing
// ---------------------------------------------------------------------------------------------

describe('Slice 2 P0b — foldflow2 GPUAlgebra queue pacing', () => {
  // A fake GPU on a simulated clock: each dispatch costs COST_MS, and a fence is heard FENCE_MS after the work ahead of it.
  const COST_MS = 4;
  const FENCE_MS = 0.5;
  const SUBMISSION_MS = 15;
  let clock = 0;
  const yields = { n: 0 };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let GPUAlgebra: any;

  function fakeDevice() {
    let busyUntil = 0;
    const submissions: number[] = [];
    let fences = 0;
    let gate: Promise<void> | null = null;
    const device = {
      features: new Set<string>(),
      limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 30 },
      addEventListener() {},
      removeEventListener() {},
      createBuffer: ({ size }: { size: number }) => ({
        size,
        destroy() {},
        async mapAsync() {
          clock = Math.max(clock, busyUntil);
        },
        getMappedRange: () => new ArrayBuffer(size),
        unmap() {},
      }),
      createShaderModule: ({ code }: { code: string }) => ({ code }),
      createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
      createBindGroup: () => ({}),
      createCommandEncoder() {
        let dispatches = 0;
        return {
          beginComputePass: () => ({
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {
              dispatches++;
            },
            end() {},
          }),
          copyBufferToBuffer() {},
          finish: () => ({ dispatches }),
        };
      },
      queue: {
        writeBuffer() {},
        submit(cbs: Array<{ dispatches: number }>) {
          const n = cbs.reduce((s, c) => s + c.dispatches, 0);
          submissions.push(n);
          busyUntil = Math.max(busyUntil, clock) + n * COST_MS;
        },
        // The clock moves to the fence's time in the same microtask its continuation runs, as a real fence would be heard.
        onSubmittedWorkDone() {
          fences++;
          const at = Math.max(clock, busyUntil) + FENCE_MS;
          const wait = gate;
          return {
            then(onDone?: (v: void) => unknown, onFail?: (e: unknown) => unknown) {
              return (wait ?? Promise.resolve()).then(() => {
                clock = Math.max(clock, at);
                return onDone?.();
              }, onFail);
            },
          };
        },
      },
    };
    return {
      device,
      submissions,
      fences: () => fences,
      hold() {
        let release!: () => void;
        gate = new Promise<void>((r) => (release = r));
        return () => {
          gate = null;
          release();
        };
      },
    };
  }

  const block = async (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    o: any,
    phase: string,
    dispatches: number,
  ) => {
    o.beginPacedBlock(phase);
    for (let i = 0; i < dispatches; i++) o.constant([64], i);
    await o.checkpoint([]);
  };

  beforeAll(async () => {
    // yieldToBrowser makes its MessageChannel on first use; count what it posts to see every yield.
    class CountingChannel extends MessageChannel {
      constructor() {
        super();
        const port = this.port2;
        const post = port.postMessage.bind(port);
        port.postMessage = (m: unknown) => {
          yields.n++;
          post(m);
        };
      }
    }
    vi.stubGlobal('MessageChannel', CountingChannel);
    vi.stubGlobal('GPUBufferUsage', BUFFER_USAGE);
    vi.stubGlobal('GPUMapMode', { READ: 1, WRITE: 2 });
    ({ GPUAlgebra } = await import(vendorUrl('foldflow2/webgpu/algebra.js')));
  });

  beforeEach(() => {
    clock = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => clock);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  afterAll(() => {
    vi.unstubAllGlobals();
  });

  it('times a phase’s first block piece by piece, then cuts its blocks into submissions under SUBMISSION_MS', async () => {
    const gpu = fakeDevice();
    const o = new GPUAlgebra(gpu.device, { useF16: false });
    const before = yields.n;

    await block(o, 'esm', 20);
    expect(gpu.submissions.splice(0)).toEqual(new Array(20).fill(1));

    for (let b = 0; b < 4; b++) await block(o, 'esm', 20);
    const later = gpu.submissions.splice(0);
    expect(later.reduce((a, n) => a + n, 0)).toBe(80);
    expect(Math.max(...later) * COST_MS).toBeLessThanOrEqual(SUBMISSION_MS);
    expect(later.length).toBeLessThan(80);
    expect(later.slice(0, 7)).toEqual([3, 3, 3, 3, 3, 3, 2]);

    // One real MessageChannel yield per closed block, and nothing left open or encoded.
    expect(yields.n - before).toBe(5);
    expect(o.encoder).toBeNull();
    expect(o.activePacer).toBeNull();
    expect(o.dispatches).toBe(0);
    expect(o.uUsed).toHaveLength(0);
  });

  it('keeps one pacer per phase: a new phase is timed on its own', async () => {
    const gpu = fakeDevice();
    const o = new GPUAlgebra(gpu.device, { useF16: false });
    await block(o, 'esm', 6);
    await block(o, 'esm', 6);
    gpu.submissions.length = 0;
    await block(o, 'trunk.0', 6);
    expect(gpu.submissions).toEqual(new Array(6).fill(1));
    expect([...o.pacers.keys()]).toEqual(['esm', 'trunk.0']);
  });

  it('checkpoint holds the run until the queue has room (back-pressure), then yields', async () => {
    const gpu = fakeDevice();
    const o = new GPUAlgebra(gpu.device, { useF16: false });
    const release = gpu.hold();
    o.beginPacedBlock('esm');
    o.constant([64], 1);
    let done = false;
    const pending = o.checkpoint([]).then(() => {
      done = true;
    });
    await flush();
    await flush();
    expect(done).toBe(false);
    release();
    await pending;
    expect(done).toBe(true);
  });

  it('leaves unpaced work as before: the 512-dispatch cap, and no flush or yield at a plain checkpoint', async () => {
    const gpu = fakeDevice();
    const o = new GPUAlgebra(gpu.device, { useF16: false });
    const before = yields.n;
    for (let i = 0; i < 600; i++) o.constant([4], 0);
    expect(gpu.submissions).toEqual([512]);
    await o.checkpoint([]);
    expect(gpu.submissions).toEqual([512]);
    expect(yields.n).toBe(before);
    // Opening a block sends the pending work first, on its own.
    o.beginPacedBlock('esm');
    expect(gpu.submissions).toEqual([512, 88]);
    await o.checkpoint([]);
  });

  it('stepBoundary yields every ODE step and drains the queue every 16th', async () => {
    const gpu = fakeDevice();
    const o = new GPUAlgebra(gpu.device, { useF16: false });
    const before = yields.n;
    for (let step = 0; step < 16; step++) {
      o.constant([4], step);
      await o.stepBoundary([], step);
    }
    expect(gpu.submissions).toEqual(new Array(16).fill(1));
    expect(yields.n - before).toBe(16);
    expect(gpu.fences()).toBe(1);
  });

  it('clear() after a run that threw mid-block drops that phase’s pacer, so the next block is timed afresh', async () => {
    const gpu = fakeDevice();
    const o = new GPUAlgebra(gpu.device, { useF16: false });
    await block(o, 'esm', 6);
    await block(o, 'esm', 6);
    o.beginPacedBlock('esm');
    for (let i = 0; i < 3; i++) o.constant([4], i);
    o.clear();
    expect(o.activePacer).toBeNull();
    expect(o.pacers.has('esm')).toBe(false);
    gpu.submissions.length = 0;
    await block(o, 'esm', 6);
    expect(gpu.submissions).toEqual(new Array(6).fill(1));
    o.destroy();
    expect(o.pacers.size).toBe(0);
  });

  it('source: LocalFold QueuePacer + MessageChannel yield, and every network stack opens a paced block', () => {
    const algebra = readSrc('static/vendor/foldflow2/webgpu/algebra.js');
    expect(algebra).toContain("import { QueuePacer } from '../../localfold/webgpu/runtime/queue-pacing.js';");
    expect(algebra).toContain("import { yieldToBrowser } from '../../localfold/webgpu/runtime/yield.js';");
    expect(algebra).not.toContain('if (++this.dispatches >= 512) this.flush();');
    const network = readSrc('static/vendor/foldflow2/shared/network.js');
    expect(network).toContain('o.beginPacedBlock?.("esm");');
    expect(network).toContain('o.beginPacedBlock?.(`${p}.${b}`);');
    expect(network).toContain('o.beginPacedBlock?.(`trunk.${b}`);');
    // The pacer's own constants are LocalFold's 15 ms submissions and ~1 s of queued work.
    const pacing = readSrc('static/vendor/localfold/webgpu/runtime/queue-pacing.js');
    expect(pacing).toMatch(/const SUBMISSION_MS = 15;/);
    expect(pacing).toMatch(/const QUEUED_WORK_MS = 1000;/);
    expect(readSrc('static/vendor/localfold/webgpu/runtime/yield.js')).toContain('new MessageChannel()');
  });
});

// ---------------------------------------------------------------------------------------------
// 2. protpardelle workgroup arrays
// ---------------------------------------------------------------------------------------------

describe('Slice 2 P0b — protpardelle workgroup arrays stay under 8 KiB', () => {
  const ELEMENT_BYTES: Record<string, number> = {
    f32: 4,
    u32: 4,
    i32: 4,
    f16: 2,
    'vec2<f32>': 8,
    'vec4<f32>': 16,
    'vec4<f16>': 8,
  };
  let tmpDir = '';
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let gen: any;
  const sources: Record<string, string> = {};

  beforeAll(async () => {
    // The WGSL generators are module-private: load ops.js with them exported (imports pinned to the real vendor tree).
    tmpDir = resolve(WEB_ROOT, `.vitest-p0b-pp-${process.pid}`);
    mkdirSync(tmpDir, { recursive: true });
    const src = readSrc('static/vendor/protpardelle/webgpu/ops.js')
      .replace('from "./graph.js"', `from "${vendorUrl('protpardelle/webgpu/graph.js')}"`)
      .replace('from "../shared/constants.js"', `from "${vendorUrl('protpardelle/shared/constants.js')}"`);
    const file = resolve(tmpDir, 'ops.mjs');
    writeFileSync(file, `${src}\nexport { linearTiledSource, NORM_SOURCE, SOFTMAX_SOURCE };\n`);
    gen = await import(pathToFileURL(file).href);
    sources.NORM = gen.NORM_SOURCE;
    sources.SOFTMAX = gen.SOFTMAX_SOURCE;
    for (const dtype of ['F32', 'F16', 'I8'])
      for (const large of [true, false])
        for (const hasF16 of [false, true])
          sources[`linear:${dtype}:${large ? 'large' : 'small'}:${hasF16 ? 'f16' : 'f32'}`] = gen.linearTiledSource(dtype, large, hasF16);
  });

  afterAll(() => {
    if (tmpDir) rmSync(tmpDir, { recursive: true, force: true });
  });

  const workgroupArrays = (wgsl: string) =>
    [...wgsl.matchAll(/var<workgroup>\s+(\w+)\s*:\s*array<\s*((?:vec[234]<\w+>)|\w+)\s*,\s*(\d+)\s*>\s*;/g)].map(
      ([, name, type, count]) => ({ name, type, bytes: ELEMENT_BYTES[type] * Number(count) }),
    );

  it('declares no workgroup array of 8192 B or more in any generated kernel, and fits the 16 KiB default', () => {
    expect(Object.keys(sources)).toHaveLength(14);
    for (const [name, wgsl] of Object.entries(sources)) {
      const arrays = workgroupArrays(wgsl);
      expect(arrays.length, name).toBeGreaterThan(0);
      for (const a of arrays) {
        expect(Number.isFinite(a.bytes), `${name} ${a.name}: ${a.type}`).toBe(true);
        expect(a.bytes, `${name} ${a.name}`).toBeLessThan(8192);
      }
      expect(arrays.reduce((s, a) => s + a.bytes, 0), name).toBeLessThanOrEqual(16384);
    }
    const file = readSrc('static/vendor/protpardelle/webgpu/ops.js');
    expect(file).not.toMatch(/array<\s*f32\s*,\s*2048\s*>/);
    expect(file).not.toMatch(/array<\s*f32\s*,\s*2176\s*>/);
    expect(file).not.toMatch(/vec4<\$\{EB\}>\s*,\s*528/);
  });

  it('indexes only declared arrays: no leftover rowTile[ / smTile[ / large-tile tb[', () => {
    for (const [name, wgsl] of Object.entries(sources)) {
      const declared = new Set([...wgsl.matchAll(/var(?:<[^>]*>)?\s+(\w+)\s*:\s*array</g)].map((m) => m[1]));
      const indexed = new Set([...wgsl.matchAll(/\b([A-Za-z_]\w*)\s*\[/g)].map((m) => m[1]));
      for (const id of indexed) expect(declared.has(id), `${name} indexes undeclared ${id}`).toBe(true);
    }
    expect(sources.NORM).toContain('var<workgroup> rowTile0:array<f32,1024>;');
    expect(sources.NORM).toContain('var<workgroup> rowTile1:array<f32,1024>;');
    expect(sources.SOFTMAX).toContain('var<workgroup> smTile0:array<f32,1024>;');
    expect(sources.SOFTMAX).toContain('var<workgroup> smTile1:array<f32,1024>;');
    expect(sources.NORM).toContain('fn rowTileGet(i:u32)->f32{if(i<1024u){return rowTile0[i];}return rowTile1[i-1024u];}');
    expect(sources.NORM).toContain('fn rowTileSet(i:u32,v:f32){if(i<1024u){rowTile0[i]=v;}else{rowTile1[i-1024u]=v;}}');
    expect(sources.SOFTMAX).toContain('fn smTileGet(i:u32)->f32{if(i<1024u){return smTile0[i];}return smTile1[i-1024u];}');
    expect(sources.SOFTMAX).toContain('fn smTileSet(i:u32,v:f32){if(i<1024u){smTile0[i]=v;}else{smTile1[i-1024u]=v;}}');
    // The bounds that pick the tiled paths are unchanged.
    expect(sources.NORM).toContain('if(d<=2048u&&(d&3u)==0u){');
    expect(sources.SOFTMAX.match(/if\(n<=2048u\)\{/g)).toHaveLength(3);
  });

  it('large-tile static B reads address exactly the slot each dynamic tbSet wrote', () => {
    // Model tbSet from the generated text (its split point) and check every unrolled read resolves to the same
    // (half, offset) as the write of that logical slot: vec4 slots kk*33+r4, I8 slots row*17+k.
    for (const [name, wgsl] of Object.entries(sources)) {
      if (!name.includes(':large:')) continue;
      const half = Number(/fn tbSet\(i:u32,v:[^)]+\)\{if\(i<(\d+)u\)\{tb0\[i\]=v;\}else\{tb1\[i-\1u\]=v;\}\}/.exec(wgsl)?.[1]);
      const write = (slot: number) => (slot < half ? `tb0:${slot}` : `tb1:${slot - half}`);
      const isI8 = name.startsWith('linear:I8');
      expect(half).toBe(isI8 ? 1088 : 264);
      expect(wgsl).toContain(`var<workgroup> tb0:array<${isI8 ? 'f32' : name.includes(':F16:') && name.endsWith(':f16') ? 'vec4<f16>' : 'vec4<f32>'},${half}>;`);
      if (isI8) {
        for (const [, part, j, kk] of wgsl.matchAll(/(tb[01])\[\(tn0\+(\d)u\)\*17u\+(\d+)u\]/g)) {
          for (let tx = 0; tx < 16; tx++) {
            const row = tx * 4 + Number(j) + (part === 'tb1' ? 64 : 0);
            expect(write(row * 17 + Number(kk)), name).toBe(`${part}:${(tx * 4 + Number(j)) * 17 + Number(kk)}`);
          }
        }
        expect(wgsl.match(/tb[01]\[\(tn0\+\du\)\*17u\+\d+u\]/g)).toHaveLength(16 * 8);
      } else {
        const reads = [...wgsl.matchAll(/(tb[01])\[(\d+)u\+tx\]/g)];
        expect(reads).toHaveLength(32);
        reads.forEach(([, part, off], i) => {
          const kk = i >> 1;
          for (let tx = 0; tx < 16; tx++) {
            const r4 = (i & 1 ? 16 : 0) + tx;
            expect(write(kk * 33 + r4), name).toBe(`${part}:${Number(off) + tx}`);
          }
        });
      }
    }
  });

  it('protpardelle engine fails fast after device loss on every entry point, not only denoise()', () => {
    const index = readSrc('static/vendor/protpardelle/webgpu/index.js');
    expect(index.match(/if \(lost\) throw new Error\(lost\);/g)).toHaveLength(4);
  });
});

// ---------------------------------------------------------------------------------------------
// 3. Worker device acquisition, shader-f16 gating, device-lost recovery, dispose
// ---------------------------------------------------------------------------------------------

interface FakeDevice {
  id: number;
  features: Set<string>;
  limits: Record<string, number>;
  lost: Promise<GPUDeviceLostInfo>;
  destroyCalls: number;
  destroy(): void;
  lose(message: string): void;
}

function fakeNavigatorGpu(hasF16: boolean) {
  const devices: FakeDevice[] = [];
  const adapter = {
    features: new Set(hasF16 ? ['shader-f16'] : []),
    limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 30, maxComputeWorkgroupStorageSize: 16384 },
    info: { vendor: 'stub', architecture: '', device: 'stub', description: 'stub gpu' },
    async requestDevice(desc: { requiredFeatures?: string[] }) {
      let resolveLost!: (info: GPUDeviceLostInfo) => void;
      const lost = new Promise<GPUDeviceLostInfo>((r) => (resolveLost = r));
      const dev: FakeDevice = {
        id: devices.length + 1,
        features: new Set(desc.requiredFeatures ?? []),
        limits: { ...adapter.limits },
        lost,
        destroyCalls: 0,
        destroy() {
          dev.destroyCalls++;
          // Browsers resolve `lost` after destroy() returns.
          setTimeout(() => resolveLost({ reason: 'destroyed', message: '' } as GPUDeviceLostInfo), 0);
        },
        lose(message) {
          resolveLost({ reason: 'unknown', message } as GPUDeviceLostInfo);
        },
      };
      devices.push(dev);
      return dev;
    },
  };
  return { gpu: { requestAdapter: vi.fn(async () => adapter) }, devices };
}

const PDB = [
  'ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N  ',
  'ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C  ',
  'ATOM      3  C   ALA A   1      13.153   5.856  -5.172  1.00  0.00           C  ',
  'ATOM      4  O   ALA A   1      13.746   5.537  -6.205  1.00  0.00           O  ',
  'ATOM      5  N   GLY A   2      13.775   6.026  -4.010  1.00  0.00           N  ',
  'ATOM      6  CA  GLY A   2      15.220   5.869  -3.923  1.00  0.00           C  ',
  'ATOM      7  C   GLY A   2      15.711   5.578  -2.517  1.00  0.00           C  ',
  'ATOM      8  O   GLY A   2      14.928   5.524  -1.574  1.00  0.00           O  ',
  'END',
].join('\n');

const structure = {
  kind: 'StructureArtifact',
  id: 'p0b-struct',
  label: 'p0b',
  sourceModelId: 'pdb-input',
  pdbText: PDB,
  sequence: 'AG',
  length: 2,
  caCoords: new Float32Array(6),
  backbone4Coords: new Float32Array(24),
} as unknown as StructureArtifact;

const sequences: SequenceSet = {
  kind: 'SequenceSet',
  id: 'p0b-seqs',
  parentStructureId: 'seq',
  sourceModelId: 'sequence-input',
  candidates: [{ id: 'c1', sequence: 'MKTAYIAKQRQISFVKSHFSRQ', sourceModelId: 'sequence-input', score: null, globalScore: null }],
};

interface WorkerCase {
  name: string;
  kind: string;
  load: () => Promise<unknown>;
  params: Record<string, unknown>;
  inputs: Record<string, unknown>;
  /** The engine option that turns native f16 on. */
  f16: (opts: Record<string, unknown>) => unknown;
  source: string;
}

const WORKERS: WorkerCase[] = [
  {
    name: 'antifold-esmif1',
    kind: 'antifold',
    load: () => import('$lib/plugins/antifold-esmif1/antifold.worker'),
    params: { variant: 'esmif1', precision: 'fp16' },
    inputs: { structure },
    f16: (o) => o.useF16,
    source: 'src/lib/plugins/antifold-esmif1/antifold.worker.ts',
  },
  {
    name: 'esmc',
    kind: 'esmc',
    load: () => import('$lib/plugins/esmc/esmc.worker'),
    params: { precision: 'fp16' },
    inputs: { sequences },
    f16: (o) => o.useF16,
    source: 'src/lib/plugins/esmc/esmc.worker.ts',
  },
  {
    name: 'protpardelle',
    kind: 'protpardelle',
    load: () => import('$lib/plugins/protpardelle/protpardelle.worker'),
    params: { precision: 'fp16' },
    inputs: {},
    f16: (o) => o.useF16,
    source: 'src/lib/plugins/protpardelle/protpardelle.worker.ts',
  },
  {
    name: 'lasermpnn',
    kind: 'lasermpnn',
    load: () => import('$lib/plugins/lasermpnn/lasermpnn.worker'),
    params: { precision: 'fp16' },
    inputs: { structure },
    f16: (o) => o.shaderF16,
    source: 'src/lib/plugins/lasermpnn/lasermpnn.worker.ts',
  },
  {
    name: 'ligandmpnn',
    kind: 'ligandmpnn',
    load: () => import('$lib/plugins/ligandmpnn/ligandmpnn.worker'),
    params: { precision: 'fp16' },
    inputs: { structure },
    f16: (o) => o.useF16,
    source: 'src/lib/plugins/ligandmpnn/ligandmpnn.worker.ts',
  },
];

describe('Slice 2 P0b — resilient device, shader-f16 gating and device-lost recovery in five workers', () => {
  async function bootWorker(w: WorkerCase, hasF16: boolean) {
    const nav = fakeNavigatorGpu(hasF16);
    const messages: Array<Record<string, unknown>> = [];
    const scope = {
      postMessage: (m: Record<string, unknown>) => messages.push(m),
      location: { origin: 'http://pds.test' },
      onmessage: null as null | ((ev: { data: unknown }) => Promise<void>),
    };
    vi.stubGlobal('self', scope);
    vi.stubGlobal('navigator', { gpu: nav.gpu, userAgent: 'Linux', hardwareConcurrency: 8 });
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })));
    vi.stubGlobal('GPUBufferUsage', BUFFER_USAGE);
    vi.resetModules();
    await w.load();
    let nextId = 1;
    const send = async (data: Record<string, unknown>) => {
      const id = nextId++;
      await scope.onmessage!({ data: { id, ...data } });
      return messages.filter((m) => m.id === id);
    };
    const run = (params: Record<string, unknown> = w.params) => send({ type: 'run', params, inputs: w.inputs });
    const engines = () => stubs.state.created.filter((c) => c.kind === w.kind);
    return { nav, messages, send, run, engines };
  }

  beforeEach(() => {
    stubs.state.created.length = 0;
    stubs.state.designError = null;
    stubs.state.loading = null;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  for (const w of WORKERS) {
    it(`${w.name}: injects one resilient device with shader-f16, recovers from real loss, and destroys it only on dispose`, async () => {
      const { nav, messages, send, run, engines } = await bootWorker(w, true);

      await run();
      expect(engines()).toHaveLength(1);
      expect(nav.devices).toHaveLength(1);
      expect(engines()[0].opts.device).toBe(nav.devices[0]);
      expect(nav.devices[0].features.has('shader-f16')).toBe(true);
      expect(w.f16(engines()[0].opts)).toBe(true);

      // Same key: the engine and its device are reused.
      await run();
      expect(engines()).toHaveLength(1);
      expect(nav.devices).toHaveLength(1);

      // Real loss: the cached engine is invalidated and the next run rebuilds on a fresh device.
      nav.devices[0].lose('GPU process crashed');
      await flush();
      await run();
      expect(nav.devices).toHaveLength(2);
      expect(engines()).toHaveLength(2);
      expect(engines()[0].destroyed).toBe(1);
      expect(engines()[1].opts.device).toBe(nav.devices[1]);
      expect(nav.devices[0].destroyCalls).toBe(0);

      // Dispose tears down the engine and destroys the device once, through the guard.
      const disposed = await send({ type: 'dispose' });
      expect(disposed).toContainEqual(expect.objectContaining({ type: 'disposed' }));
      expect(engines()[1].destroyed).toBe(1);
      expect(nav.devices[1].destroyCalls).toBe(1);
      await flush();
      await flush();

      // The destroyed device's `lost` resolving changes nothing; a later run starts clean on a new device.
      await run();
      expect(nav.devices).toHaveLength(3);
      expect(engines()).toHaveLength(3);
      expect(engines()[2].opts.device).toBe(nav.devices[2]);
      expect(nav.devices.map((d) => d.destroyCalls)).toEqual([0, 1, 0]);
      // Every run got past device and engine setup to the stubbed engine; none saw a device-lost error.
      const errors = messages.filter((m) => m.type === 'error').map((m) => String(m.message));
      expect(errors).toEqual(new Array(4).fill('stub-stop'));
      expect(errors.some((e) => isWebGpuDeviceLostError(new Error(e)))).toBe(false);
    });

    it(`${w.name}: an engine whose device was lost while it loaded is not reused`, async () => {
      const { nav, run, engines } = await bootWorker(w, true);
      let finishLoading!: () => void;
      stubs.state.loading = new Promise<void>((r) => (finishLoading = r));
      const first = run();
      await vi.waitFor(() => expect(nav.devices).toHaveLength(1));
      nav.devices[0].lose('GPU reset during load');
      await flush();
      finishLoading();
      await first;
      stubs.state.loading = null;
      expect(engines()).toHaveLength(1);
      await run();
      expect(nav.devices).toHaveLength(2);
      expect(engines()).toHaveLength(2);
      expect(engines()[1].opts.device).toBe(nav.devices[1]);
    });

    it(`${w.name}: leaves native f16 off on an adapter without shader-f16`, async () => {
      const { nav, run, engines } = await bootWorker(w, false);
      await run();
      expect(engines()).toHaveLength(1);
      expect(nav.devices[0].features.has('shader-f16')).toBe(false);
      expect(w.f16(engines()[0].opts)).toBe(false);
    });
  }

  it('lasermpnn: shader-f16 GEMMs only on the fp16 bundle', async () => {
    const w = WORKERS.find((c) => c.name === 'lasermpnn')!;
    const { run, engines } = await bootWorker(w, true);
    await run({ precision: 'int8' });
    expect(engines()[0].opts.shaderF16).toBe(false);
  });

  it('ligandmpnn: a mid-run loss still falls back to CPU, and the CPU engine survives the guard', async () => {
    const w = WORKERS.find((c) => c.name === 'ligandmpnn')!;
    const { nav, run } = await bootWorker(w, true);
    stubs.state.designError = () => new Error('WebGPU device lost: GPU reset');
    await run();
    nav.devices[0].lose('GPU reset');
    await flush();
    const kinds = () => stubs.state.created.map((c) => c.kind);
    expect(kinds()).toEqual(['ligandmpnn', 'ligandmpnn-cpu']);
    expect(stubs.state.created[0].destroyed).toBe(1);
    stubs.state.designError = null;
    await run();
    expect(kinds()).toEqual(['ligandmpnn', 'ligandmpnn-cpu']);
  });

  it('source: every worker acquires through requestResilientWebGpuDevice and tracks it with createDeviceLostGuard', () => {
    for (const w of WORKERS) {
      const src = readSrc(w.source);
      expect(src, w.name).toContain("requestResilientWebGpuDevice({ optionalFeatures: ['shader-f16'] })");
      expect(src, w.name).toContain('const deviceLostGuard = createDeviceLostGuard();');
      expect(src, w.name).toContain('deviceLostGuard.watch(device, () => {');
      expect(src, w.name).toMatch(/deviceLostGuard\.destroy\(gpu\?\.device\);\s*gpu = null;/);
      expect(src, w.name).not.toMatch(/navigator\.gpu\.requestAdapter/);
    }
    expect(readSrc('src/lib/plugins/esmc/esmc.worker.ts')).not.toContain('useF16: false');
    expect(readSrc('src/lib/plugins/lasermpnn/lasermpnn.worker.ts')).toContain("shaderF16: precision === 'fp16' && hasF16");
  });
});

// ---------------------------------------------------------------------------------------------
// 4. Vendor GPU backends after device loss
// ---------------------------------------------------------------------------------------------

describe('Slice 2 P0b — lasermpnn GpuBackend / ligandmpnn GpuOps after device loss', () => {
  function lossyDevice() {
    let lose!: (info: GPUDeviceLostInfo) => void;
    const lost = new Promise<GPUDeviceLostInfo>((r) => (lose = r));
    const submit = vi.fn();
    const device = {
      lost,
      features: new Set<string>(),
      limits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30 },
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      createBuffer: ({ size }: { size: number }) => ({
        size,
        destroy: vi.fn(),
        mapAsync: async () => {},
        getMappedRange: () => new ArrayBuffer(size),
      }),
      createShaderModule: () => ({}),
      createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
      createBindGroup: () => ({}),
      createCommandEncoder: () => ({
        beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
        copyBufferToBuffer() {},
        finish: () => ({}),
      }),
      queue: { submit, writeBuffer: vi.fn(), onSubmittedWorkDone: async () => {} },
    };
    return { device, submit, lose: (message: string) => lose({ reason: 'unknown', message } as GPUDeviceLostInfo) };
  }

  beforeAll(() => {
    vi.stubGlobal('GPUBufferUsage', BUFFER_USAGE);
    vi.stubGlobal('GPUMapMode', { READ: 1, WRITE: 2 });
  });

  afterAll(() => {
    vi.unstubAllGlobals();
  });

  it('ligandmpnn GpuOps drops recorded work and fails fast with an error the CPU fallback recognises', async () => {
    const { GpuOps } = await import(vendorUrl('ligandmpnn/webgpu/ligandmpnn/ops.js'));
    const d = lossyDevice();
    const ops = new GpuOps(d.device, {});
    const t = ops.tensor(new Float32Array(4), [4]);
    ops.dispatch('fn main(){}', [t], ops.zeros([4]), [1]);
    expect(ops.encoder).not.toBeNull();
    d.lose('GPU reset');
    await flush();
    expect(ops.lost).toBe('WebGPU device lost: GPU reset');
    ops.flush();
    expect(d.submit).not.toHaveBeenCalled();
    const thrown = (() => {
      try {
        ops.tensor(null, [4]);
      } catch (e) {
        return e;
      }
    })();
    expect(isWebGpuDeviceLostError(thrown)).toBe(true);
    expect(() => ops.dispatch('fn main(){}', [t], t, [1])).toThrow(/device lost/);
    await expect(ops.read(t)).rejects.toThrow(/device lost/);
    ops.destroy();
    expect(d.submit).not.toHaveBeenCalled();
    expect(d.device.removeEventListener).toHaveBeenCalledWith('uncapturederror', ops.listener);
  });

  it('lasermpnn GpuBackend drops recorded work and fails fast after loss', async () => {
    const { GpuBackend } = await import(vendorUrl('lasermpnn/webgpu/lasermpnn/backend.js'));
    const d = lossyDevice();
    const b = new GpuBackend(d.device, {}, {});
    b.pipelines.set('k', { getBindGroupLayout: () => ({}) });
    const x = b.upload(new Float32Array(4), 1, 4);
    b.dispatch('k', [x], new Uint32Array(4), [1]);
    expect(b.encoder).not.toBeNull();
    d.lose('');
    await flush();
    expect(b.lost).toBe('WebGPU device lost');
    expect(b.encoder).toBeNull();
    b.flush();
    expect(d.submit).not.toHaveBeenCalled();
    expect(() => b.dispatch('k', [x], new Uint32Array(4), [1])).toThrow(/device lost/);
    await expect(b.read(x)).rejects.toThrow(/device lost/);
    await expect(b.readMany([x])).rejects.toThrow(/device lost/);
    b.destroy();
    expect(d.submit).not.toHaveBeenCalled();
  });
});
