import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  installRfd4QueuePacing,
  paceRfd4Engine,
  QUEUED_WORK_MS,
  SUBMISSION_MS,
} from '$lib/plugins/rfd4-proteina/queue-pacing';

// The real upstream engine (synced into static/vendor) on a fake device whose queue runs on a simulated clock: each
// dispatch costs its pipeline's `ms`, fences are heard FENCE_MS after the work ahead of them finishes, and a fence's
// time only passes when someone waits on it.

const FENCE_MS = 0.5;

type Cmd = { kind: 'dispatch'; id: string } | { kind: 'copy' };

interface FakeGpu {
  device: GPUDevice;
  submissions: Cmd[][];
  maxQueuedMs: number;
}

let clock = 0;

function fakeGpu(): FakeGpu {
  let busyUntil = 0;
  const gpu: FakeGpu = { device: null as unknown as GPUDevice, submissions: [], maxQueuedMs: 0 };
  const fence = () => {
    const at = Math.max(clock, busyUntil) + FENCE_MS;
    return {
      then(onDone: (v: void) => unknown, onFail?: (e: unknown) => unknown) {
        return Promise.resolve().then(() => {
          clock = Math.max(clock, at);
        }).then(onDone, onFail);
      },
      catch() {
        return this;
      },
    };
  };
  const device = {
    features: new Set<string>(['subgroups']),
    limits: { minStorageBufferOffsetAlignment: 256 },
    createBuffer: ({ size }: { size: number }) => ({
      size,
      destroy() {},
      unmap() {},
      async mapAsync() {
        clock = Math.max(clock, busyUntil);
      },
      getMappedRange: () => new ArrayBuffer(size),
    }),
    createCommandEncoder() {
      const cmds: Cmd[] = [];
      return {
        beginComputePass() {
          let pipeline: { id: string; ms: number } | null = null;
          return {
            setPipeline(p: { id: string; ms: number }) {
              pipeline = p;
            },
            setBindGroup() {},
            dispatchWorkgroups() {
              cmds.push({ kind: 'dispatch', id: pipeline!.id });
            },
            end() {},
          };
        },
        copyBufferToBuffer() {
          cmds.push({ kind: 'copy' });
        },
        finish: () => ({ cmds }),
      };
    },
    queue: {
      submit(buffers: Array<{ cmds: Cmd[] }>) {
        const cmds = buffers.flatMap((b) => b.cmds);
        gpu.submissions.push(cmds);
        const ms = cmds.reduce((sum, c) => sum + (c.kind === 'dispatch' ? costs.get(c.id)! : 0), 0);
        busyUntil = Math.max(busyUntil, clock) + ms;
        gpu.maxQueuedMs = Math.max(gpu.maxQueuedMs, busyUntil - clock);
      },
      onSubmittedWorkDone: fence,
    },
  };
  gpu.device = device as unknown as GPUDevice;
  return gpu;
}

const costs = new Map<string, number>();

function dispatch(id: string, ms: number, x = 1) {
  costs.set(id, ms);
  return { pipeline: { id, ms }, bindGroup: {}, x, y: 1, label: id };
}

const dispatchIds = (subs: Cmd[][]) =>
  subs.flat().flatMap((c) => (c.kind === 'dispatch' ? [c.id] : []));
const dispatchCounts = (subs: Cmd[][]) =>
  subs.map((s) => s.filter((c) => c.kind === 'dispatch').length).filter((n) => n > 0);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let Rfd4pEngine: any;

beforeAll(async () => {
  const enginePath = resolve(process.cwd(), 'static/vendor/rfd4-proteina/webgpu/rfd4p/engine.js');
  ({ Rfd4pEngine } = await import(pathToFileURL(enginePath).href));
});

beforeEach(() => {
  clock = 0;
  costs.clear();
  vi.spyOn(performance, 'now').mockImplementation(() => clock);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('RFD4-Proteina queue pacing', () => {
  it('replays upstream dispatches in the same order, with the readback copy after all of them', async () => {
    const steps = Array.from({ length: 12 }, (_, i) => dispatch(`k${i}`, 1 + (i % 5) * 3));
    const reference = fakeGpu();
    const paced = fakeGpu();
    const refEngine = new Rfd4pEngine(reference.device);
    const engine = new Rfd4pEngine(paced.device);
    paceRfd4Engine(engine);
    const programs = (e: typeof engine) => [
      e.program(steps.slice(0, 5)),
      [e.program(steps.slice(5, 9)), e.program(steps.slice(9))],
    ];
    const out = { floats: 8 } as unknown as GPUBuffer;

    await refEngine.runAndRead(programs(refEngine), out);
    expect(reference.submissions).toHaveLength(1);

    for (let pass = 0; pass < 2; pass++) {
      paced.submissions = [];
      const result = await engine.runAndRead(programs(engine), out);
      expect(result).toBeInstanceOf(Float32Array);
      expect(result).toHaveLength(8);
      expect(dispatchIds(paced.submissions)).toEqual(dispatchIds(reference.submissions));
      const all = paced.submissions.flat();
      expect(all.filter((c) => c.kind === 'copy')).toHaveLength(1);
      expect(all[all.length - 1]).toEqual({ kind: 'copy' });
    }
  });

  it('times each new dispatch alone, then fills submissions up to SUBMISSION_MS', async () => {
    const gpu = fakeGpu();
    const engine = new Rfd4pEngine(gpu.device);
    paceRfd4Engine(engine);
    const program = engine.program(Array.from({ length: 10 }, (_, i) => dispatch(`d${i}`, 4)));
    const out = { floats: 4 } as unknown as GPUBuffer;

    await engine.runAndRead([program], out);
    expect(dispatchCounts(gpu.submissions)).toEqual(new Array(10).fill(1));

    gpu.submissions = [];
    await engine.runAndRead([program], out);
    // 4 ms each: three fit in SUBMISSION_MS (15 ms), a fourth would not.
    expect(dispatchCounts(gpu.submissions)).toEqual([3, 3, 3, 1]);
  });

  it('sends a dispatch longer than SUBMISSION_MS alone', async () => {
    const gpu = fakeGpu();
    const engine = new Rfd4pEngine(gpu.device);
    paceRfd4Engine(engine);
    const program = engine.program([dispatch('a', 2), dispatch('b', 2), dispatch('big', 40), dispatch('c', 2)]);
    const out = { floats: 4 } as unknown as GPUBuffer;

    await engine.runAndRead([program], out);
    gpu.submissions = [];
    await engine.runAndRead([program], out);
    const subs = gpu.submissions.filter((s) => s.some((c) => c.kind === 'dispatch'));
    expect(subs.map((s) => dispatchIds([s]))).toEqual([['a', 'b'], ['big'], ['c']]);
  });

  it('keeps about QUEUED_WORK_MS queued however long a step is', async () => {
    const gpu = fakeGpu();
    const engine = new Rfd4pEngine(gpu.device);
    paceRfd4Engine(engine);
    // Three kinds of layer repeated: 600 dispatches, 3.6 s of work in one runAndRead.
    const layer = [dispatch('gemm', 8), dispatch('norm', 1), dispatch('attn', 9)];
    const program = engine.program(Array.from({ length: 200 }, () => layer).flat());
    const out = { floats: 4 } as unknown as GPUBuffer;

    await engine.runAndRead([program], out);
    gpu.maxQueuedMs = 0;
    gpu.submissions = [];
    await engine.runAndRead([program], out);
    expect(dispatchIds(gpu.submissions)).toHaveLength(600);
    expect(gpu.maxQueuedMs).toBeGreaterThan(QUEUED_WORK_MS / 2);
    expect(gpu.maxQueuedMs).toBeLessThanOrEqual(QUEUED_WORK_MS + 2 * SUBMISSION_MS);
  });

  it('keeps upstream run() synchronous: its work is submitted before it returns, in order', async () => {
    const gpu = fakeGpu();
    const engine = new Rfd4pEngine(gpu.device);
    paceRfd4Engine(engine);
    const steps = [dispatch('p', 5), dispatch('q', 5), dispatch('r', 5), dispatch('s', 5)];
    const out = { floats: 4 } as unknown as GPUBuffer;

    engine.run(engine.program(steps));
    expect(dispatchCounts(gpu.submissions)).toEqual([1, 1, 1, 1]);

    await engine.runAndRead([engine.program(steps)], out);
    gpu.submissions = [];
    engine.run([engine.program(steps.slice(0, 2)), engine.program(steps.slice(2))]);
    expect(dispatchIds(gpu.submissions)).toEqual(['p', 'q', 'r', 's']);
    expect(dispatchCounts(gpu.submissions)).toEqual([3, 1]);
  });

  it('installs once per engine and paces an encoder the generator creates later', () => {
    const gpu = fakeGpu();
    const trunk = new Rfd4pEngine(gpu.device);
    const lateEncoder = new Rfd4pEngine(gpu.device);
    const generator = {
      trunk: { engine: trunk },
      decoder: { engine: new Rfd4pEngine(gpu.device) },
      encoder: null as { engine: typeof lateEncoder } | null,
      getEncoder() {
        this.encoder ??= { engine: lateEncoder };
        return this.encoder;
      },
    };

    installRfd4QueuePacing(generator);
    const pacedRun = trunk.run;
    expect(pacedRun).not.toBe(Rfd4pEngine.prototype.run);
    installRfd4QueuePacing(generator);
    expect(trunk.run).toBe(pacedRun);

    expect(lateEncoder.run).toBe(Rfd4pEngine.prototype.run);
    expect(generator.getEncoder().engine).toBe(lateEncoder);
    expect(lateEncoder.run).not.toBe(Rfd4pEngine.prototype.run);
  });
});
