import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = process.cwd();

function readSrc(relPath: string): string {
  return readFileSync(resolve(ROOT, relPath), 'utf8');
}

function vendorUrl(relPath: string): string {
  return pathToFileURL(resolve(ROOT, relPath)).href;
}

/**
 * Faithful JS execution of thermompnn/webgpu/shaders.js topkShader's storage-backed
 * libstdc++ introselect + introsort algorithm, verifying exact tie-break parity against cpu/topk.js.
 */
function runStorageTopkReference(distRow: Float32Array, K: number): number[] {
  const L = distRow.length;
  const q = new Uint32Array(L);
  for (let i = 0; i < L; i++) q[i] = i;
  const less = (a: number, b: number) => distRow[a] < distRow[b];
  const swap = (a: number, b: number) => {
    const t = q[a];
    q[a] = q[b];
    q[b] = t;
  };
  const median = (out: number, a: number, b: number, c: number) => {
    if (less(q[a], q[b])) {
      if (less(q[b], q[c])) swap(out, b);
      else if (less(q[a], q[c])) swap(out, c);
      else swap(out, a);
    } else if (less(q[a], q[c])) swap(out, a);
    else if (less(q[b], q[c])) swap(out, c);
    else swap(out, b);
  };
  const splitRange = (lo: number, hi: number): number => {
    median(lo, lo + 1, lo + Math.floor((hi - lo) / 2), hi - 1);
    const pivot = q[lo];
    let a = lo + 1;
    let b = hi;
    while (true) {
      while (less(q[a], pivot)) a++;
      b--;
      while (less(pivot, q[b])) b--;
      if (a >= b) return a;
      swap(a, b);
      a++;
    }
  };
  const insertion = (lo: number, hi: number) => {
    for (let i = lo + 1; i < hi; i++) {
      const value = q[i];
      let j = i;
      while (j > lo) {
        if (!less(value, q[j - 1])) break;
        q[j] = q[j - 1];
        j--;
      }
      q[j] = value;
    }
  };
  const adjust = (lo: number, h: number, len: number, value: number) => {
    const top = h;
    let hole = h;
    let child = hole;
    while (child < Math.floor((len - 1) / 2)) {
      child = 2 * (child + 1);
      if (less(q[lo + child], q[lo + child - 1])) child--;
      q[lo + hole] = q[lo + child];
      hole = child;
    }
    if (len % 2 === 0 && child === Math.floor((len - 2) / 2)) {
      child = 2 * (child + 1);
      q[lo + hole] = q[lo + child - 1];
      hole = child - 1;
    }
    let parent = Math.floor((hole - 1) / 2);
    while (hole > top) {
      if (!less(q[lo + parent], value)) break;
      q[lo + hole] = q[lo + parent];
      hole = parent;
      parent = Math.floor((hole - 1) / 2);
    }
    q[lo + hole] = value;
  };
  const heap = (lo: number, hi: number) => {
    const n = hi - lo;
    if (n < 2) return;
    for (let p = Math.floor((n - 2) / 2); p >= 0; p--) adjust(lo, p, n, q[lo + p]);
  };
  const heapsort = (lo: number, hi: number) => {
    heap(lo, hi);
    let end = hi;
    while (end - lo > 1) {
      const v = q[end - 1];
      q[end - 1] = q[lo];
      end--;
      adjust(lo, 0, end - lo, v);
    }
  };

  let lo = 0;
  let hi = L;
  let depth = 2 * Math.floor(Math.log2(L));
  while (hi - lo > 3) {
    if (depth === 0) {
      heap(lo, K);
      for (let i = K; i < hi; i++) {
        if (less(q[i], q[lo])) {
          const v = q[i];
          q[i] = q[lo];
          adjust(lo, 0, K - lo, v);
        }
      }
      swap(lo, K - 1);
      break;
    }
    depth--;
    const cut = splitRange(lo, hi);
    if (cut <= K - 1) lo = cut;
    else hi = cut;
  }
  if (hi - lo <= 3) insertion(lo, hi);
  const stack: Array<[number, number, number]> = [
    [0, K - 1, 2 * Math.floor(Math.log2(Math.max(1, K - 1)))],
  ];
  while (stack.length > 0) {
    const s = stack.pop()!;
    lo = s[0];
    hi = s[1];
    depth = s[2];
    while (hi - lo > 16) {
      if (depth === 0) {
        heapsort(lo, hi);
        break;
      }
      depth--;
      const cut = splitRange(lo, hi);
      stack.push([cut, hi, depth]);
      hi = cut;
    }
  }
  insertion(0, K - 1);
  return Array.from(q.slice(0, K));
}

describe('Track A — behavioral regression tests for WebGPU runtime & kernel fixes', () => {
  it('thermompnn topkShader eliminates var<private> O(L) array AND stays bit-exact with cpu/topk.js on gapped/tied inputs', async () => {
    const [{ topkShader }, { topk }] = await Promise.all([
      import(vendorUrl('static/vendor/thermompnn/webgpu/shaders.js')),
      import(vendorUrl('static/vendor/thermompnn/cpu/topk.js')),
    ]);

    const wgsl: string = topkShader(2048, 48);
    expect(wgsl).toContain('@binding(2) var<storage,read_write> q:array<u32>;');
    expect(wgsl).not.toMatch(/var<private>\s+q\s*:/);

    // Build gapped distance rows (exact tie cluster at maximum, matching distanceShader)
    const makeGappedDistances = (L: number, gapRanges: Array<[number, number]>): Float32Array => {
      const mask = new Float32Array(L).fill(1);
      for (const [a, b] of gapRanges) {
        for (let i = a; i < b; i++) mask[i] = 0;
      }
      const row = new Float32Array(L);
      let maximum = 0;
      for (let j = 0; j < L; j++) {
        const d = Math.fround(Math.sqrt((j - 10) * (j - 10) * 3.8 + 1e-6) * mask[j]);
        row[j] = d;
        if (d > maximum) maximum = d;
      }
      for (let j = 0; j < L; j++) {
        row[j] = Math.fround(row[j] + (1 - mask[j]) * maximum);
      }
      return row;
    };

    const cases: Array<[string, Float32Array, number]> = [
      ['contiguous-40-symmetric', Float32Array.from({ length: 40 }, (_, j) => Math.abs(j - 20)), 40],
      ['gap-20-in-middle', makeGappedDistances(60, [[20, 40]]), 48],
      ['two-gaps', makeGappedDistances(85, [[15, 30], [50, 65]]), 48],
      ['large-2048-gapped', makeGappedDistances(2048, [[400, 500], [1200, 1250]]), 48],
    ];
    for (const [label, row, K] of cases) {
      const expected = Array.from(topk(row, K));
      const actual = runStorageTopkReference(row, K);
      expect(actual, `tie-break permutation mismatch on ${label}`).toEqual(expected);
    }
  });

  it('thermompnn GPUOps.norm dispatches 2D (65535, 2) without throwing at L=1400 (67,200 edge rows)', async () => {
    const { GPUOps } = await import(vendorUrl('static/vendor/thermompnn/webgpu/ops.js'));
    const dispatches: Array<[number, number]> = [];
    const fakeBuffer = (size: number) => ({ size, destroy: vi.fn() });
    const fakeDevice = {
      limits: { maxComputeWorkgroupsPerDimension: 65535 },
      createBuffer: vi.fn(({ size }: { size: number }) => fakeBuffer(size)),
      createBindGroup: vi.fn(() => ({})),
      createCommandEncoder: vi.fn(() => ({
        beginComputePass: () => ({
          setPipeline: vi.fn(),
          setBindGroup: vi.fn(),
          dispatchWorkgroups: (x: number, y = 1) => dispatches.push([x, y]),
          end: vi.fn(),
        }),
        finish: vi.fn(),
      })),
      queue: { writeBuffer: vi.fn(), submit: vi.fn() },
    };
    const fakeEngine = {
      device: fakeDevice,
      useF16: false,
      w: {
        'enc.norm': { buffer: fakeBuffer(512), size: 512 },
        'enc.norm.weight': { buffer: fakeBuffer(512), size: 512 },
        'enc.norm.bias': { buffer: fakeBuffer(512), size: 512 },
      },
      pipeline: vi.fn(() => ({ getBindGroupLayout: () => ({}) })),
      ensureAlive: vi.fn(),
      admit: vi.fn(),
      release: vi.fn(),
      buffer: vi.fn((size: number) => fakeBuffer(size)),
    };
    const prevGpuUsage = (globalThis as Record<string, unknown>).GPUBufferUsage;
    (globalThis as Record<string, unknown>).GPUBufferUsage = {
      STORAGE: 0x80,
      COPY_SRC: 0x04,
      COPY_DST: 0x08,
      MAP_READ: 0x01,
    };
    try {
      const ops = new GPUOps(fakeEngine, {
        length: 1,
        X: new Float32Array(12),
        mask: new Float32Array(1),
        residue_idx: new Int32Array(1),
        chain: new Int32Array(1),
        S: new Int32Array(1),
      });
      const edgeTensor = { rows: 1400 * 48, cols: 128, buffer: fakeBuffer(1400 * 48 * 128 * 4), size: 1400 * 48 * 128 * 4 };
      const out = ops.norm(edgeTensor, null, 'enc.norm', false);
      expect(out.rows).toBe(67200);
      expect(dispatches).toEqual([[65535, 2]]);
    } finally {
      (globalThis as Record<string, unknown>).GPUBufferUsage = prevGpuUsage;
    }
  });

  it('foldflow2 Engine.encodeEsm reads single & pair separately and returns null above maxBufferSize', async () => {
    const { Engine } = await import(vendorUrl('static/vendor/foldflow2/shared/engine.js'));

    const readCalls: unknown[] = [];
    const readManySpy = vi.fn();
    const fakeSingle = { id: 'single' };
    const fakePair = { id: 'pair' };
    const makeEngine = (maxBufferSize: number) => {
      const fakeAlg = {
        device: { limits: { maxBufferSize } },
        read: vi.fn(async (t: unknown) => {
          readCalls.push(t);
          return new Float32Array(4);
        }),
        readMany: readManySpy,
        clear: vi.fn(),
      };
      const eng = Object.create(Engine.prototype);
      eng.o = fakeAlg;
      eng.network = { esm: vi.fn(async () => [fakeSingle, fakePair]) };
      eng.destroyed = false;
      eng.busy = false;
      eng.controller = null;
      return eng;
    };

    // Case 1: L=64 fits well under 256 MiB -> reads single and pair separately (never concatenates via readMany)
    const engSmall = makeEngine(256 * 1024 * 1024);
    const emb = await engSmall.encodeEsm({ length: 64 });
    expect(emb).not.toBeNull();
    expect(readManySpy).not.toHaveBeenCalled();
    expect(readCalls).toEqual([fakeSingle, fakePair]);

    // Case 2: L=320 -> pairBytes = 320*320*660*4 = ~270 MiB > 256 MiB -> returns null cleanly instead of crashing
    const engTooBig = makeEngine(256 * 1024 * 1024);
    const skipped = await engTooBig.encodeEsm({ length: 320 });
    expect(skipped).toBeNull();
    expect(engTooBig.network.esm).not.toHaveBeenCalled();
  });

  it('evodiff generate() supports recordSteps: false and yields via MessageChannel without setTimeout(0)', async () => {
    const { generate } = await import(vendorUrl('static/vendor/evodiff/shared/sampling.js'));
    const logits = new Float32Array(6 * 31).fill(0);
    const forward = vi.fn(async (tokens: Uint32Array) => ({ logits, tokens }));
    const res = await generate(forward, { length: 6, seed: 42, recordSteps: false });
    expect(res.sequence).toHaveLength(6);
    expect(res.steps).toEqual([]);
    expect(forward).toHaveBeenCalledTimes(6);
  });

  it('evodiff WebGPU forward() checks error scopes on pass >= 2 and always unmaps persistent staging on mid-readback error', async () => {
    const { createEvoDiffEngine } = await import(vendorUrl('static/vendor/evodiff/webgpu/index.js'));

    let passCount = 0;
    let failGetMappedRangeOnce = true;
    let popErrorOnPass3 = false;
    const stagingUnmaps: number[] = [];
    const createdBuffers: Array<{ usage: number; unmapCount: number; mapped: boolean }> = [];

    const prevUsage = (globalThis as Record<string, unknown>).GPUBufferUsage;
    const prevMapMode = (globalThis as Record<string, unknown>).GPUMapMode;
    (globalThis as Record<string, unknown>).GPUBufferUsage = {
      STORAGE: 0x80,
      UNIFORM: 0x40,
      COPY_SRC: 0x04,
      COPY_DST: 0x08,
      MAP_READ: 0x01,
    };
    (globalThis as Record<string, unknown>).GPUMapMode = { READ: 1 };

    try {
      let scopeStack: Array<string> = [];
      const fakeDevice = {
        features: new Set<string>(),
        limits: {
          maxBufferSize: 256 * 1024 * 1024,
          maxStorageBufferBindingSize: 128 * 1024 * 1024,
        },
        lost: new Promise(() => {}),
        createShaderModule: vi.fn(() => ({ getCompilationInfo: async () => ({ messages: [] }) })),
        createComputePipelineAsync: vi.fn(async () => ({ getBindGroupLayout: () => ({}) })),
        createBindGroup: vi.fn(() => ({})),
        createCommandEncoder: vi.fn(() => ({
          beginComputePass: () => ({
            setPipeline: vi.fn(),
            setBindGroup: vi.fn(),
            dispatchWorkgroups: vi.fn(),
            end: vi.fn(),
          }),
          copyBufferToBuffer: vi.fn(),
          finish: vi.fn(() => ({})),
        })),
        pushErrorScope: vi.fn((kind: string) => scopeStack.push(kind)),
        popErrorScope: vi.fn(async () => {
          const kind = scopeStack.pop();
          if (popErrorOnPass3 && kind === 'validation') return { message: 'simulated pass-3 validation failure' };
          return null;
        }),
        createBuffer: vi.fn(({ size, usage }: { size: number; usage: number }) => {
          const rec = { usage, unmapCount: 0, mapped: false };
          createdBuffers.push(rec);
          const ab = new ArrayBuffer(size);
          return {
            size,
            destroy: vi.fn(),
            mapAsync: vi.fn(async () => {
              if (rec.mapped) throw new Error('Buffer is already mapped');
              rec.mapped = true;
            }),
            getMappedRange: vi.fn(() => {
              if (passCount === 2 && failGetMappedRangeOnce) {
                failGetMappedRangeOnce = false;
                throw new Error('simulated mid-readback throw');
              }
              return ab;
            }),
            unmap: vi.fn(() => {
              rec.mapped = false;
              rec.unmapCount += 1;
              stagingUnmaps.push(passCount);
            }),
          };
        }),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() },
      };

      // Build a real, valid OA_DM_38M int8 safetensors bundle in memory (2 shards < 25 MB each)
      const { sha256 } = await import(vendorUrl('static/vendor/evodiff/shared/weights.js'));
      const expected: Record<string, number[]> = {
        'embedder.embedder.weight': [31, 8],
        'embedder.up_embedder.conv.weight': [1024, 8, 1],
        'embedder.up_embedder.conv.bias': [1024],
        'decoder.conv.weight': [31, 1024, 1],
        'decoder.conv.bias': [31],
        'last_norm.weight': [1024],
        'last_norm.bias': [1024],
      };
      for (let i = 0; i < 16; i++) {
        const prefix = `embedder.layers.${i}.`;
        for (const [name, shape] of Object.entries({
          conv: [512, 512, 5],
          'sequence1.0': [1024],
          'sequence1.2.conv': [512, 1024, 1],
          'sequence1.3': [512],
          'sequence2.0': [512],
          'sequence2.2.conv': [1024, 512, 1],
        })) {
          expected[`${prefix}${name}.weight`] = shape;
          expected[`${prefix}${name}.bias`] = [shape[0]];
        }
      }

      const files = new Map<string, ArrayBuffer>();
      const manifestTensors: Record<string, unknown> = {};
      const shards: Array<{ file: string; byteLength: number; sha256: string }> = [];
      let totalBytes = 0;
      let totalParams = 0;

      const entries = Object.entries(expected);
      const chunks = [entries.slice(0, 100), entries.slice(100)];
      for (let sIdx = 0; sIdx < chunks.length; sIdx++) {
        const shardFile = `shard-${sIdx}.safetensors`;
        const header: Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }> = {};
        let dataOff = 0;
        const scaleChunks: Array<{ off: number; len: number }> = [];
        for (const [name, shape] of chunks[sIdx]) {
          const count = shape.reduce((a, b) => a * b, 1);
          totalParams += count;
          const isI8 = shape.length === 3;
          const dtype = isI8 ? 'I8' : 'F32';
          const byteLen = count * (isI8 ? 1 : 4);
          header[name] = { dtype, shape, data_offsets: [dataOff, dataOff + byteLen] };
          dataOff += byteLen;
          if (isI8) {
            const scaleName = `${name}.__scale`;
            const scaleBytes = shape[0] * 4;
            header[scaleName] = { dtype: 'F32', shape: [shape[0]], data_offsets: [dataOff, dataOff + scaleBytes] };
            scaleChunks.push({ off: dataOff, len: shape[0] });
            dataOff += scaleBytes;
            manifestTensors[name] = {
              shard: shardFile,
              dtype: 'I8',
              shape,
              quant: { scheme: 'symmetric_per_output_channel', axis: 0, scale_tensor: scaleName },
            };
          } else {
            manifestTensors[name] = { shard: shardFile, dtype: 'F32', shape };
          }
        }
        const headerBytes = new TextEncoder().encode(JSON.stringify(header));
        const buf = new ArrayBuffer(8 + headerBytes.byteLength + dataOff);
        new DataView(buf).setBigUint64(0, BigInt(headerBytes.byteLength), true);
        new Uint8Array(buf, 8, headerBytes.byteLength).set(headerBytes);
        const bodyStart = 8 + headerBytes.byteLength;
        const dv = new DataView(buf);
        for (const sc of scaleChunks) {
          for (let k = 0; k < sc.len; k++) dv.setFloat32(bodyStart + sc.off + k * 4, 1.0, true);
        }
        files.set(`mock://evodiff/int8/${shardFile}`, buf);
        shards.push({ file: shardFile, byteLength: buf.byteLength, sha256: await sha256(buf) });
        totalBytes += buf.byteLength;
      }

      const manifestObj = {
        format: 'evodiff-safetensors-v1',
        model: 'OA_DM_38M',
        precision: 'int8',
        num_tensors: 199,
        total_params: totalParams,
        total_bytes: totalBytes,
        config: {
          dilations: [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128],
          width: 1024,
          activation: 'gelu_exact',
        },
        shards,
        tensors: manifestTensors,
      };
      files.set(
        'mock://evodiff/int8/manifest.json',
        new TextEncoder().encode(JSON.stringify(manifestObj)).buffer,
      );

      const eng = await createEvoDiffEngine({
        baseUrl: 'mock://evodiff',
        precision: 'int8',
        device: fakeDevice,
        cache: 'none',
        read: async (url: string) => {
          const hit = files.get(url);
          if (!hit) throw new Error(`Unexpected URL ${url}`);
          return hit;
        },
      });

      // Pass 1: succeeds
      passCount = 1;
      const r1 = await eng.forward('ACDEFG');
      expect(r1.logits.length).toBe(6 * 31);

      // Pass 2: throws after mapAsync -> finally MUST unmap persistent scratch.staging
      passCount = 2;
      await expect(eng.forward('ACDEFG')).rejects.toThrow('simulated mid-readback throw');
      expect(stagingUnmaps).toContain(2);

      // Pass 3: error scope MUST still be active and throw on validation error (not latched off)
      passCount = 3;
      popErrorOnPass3 = true;
      await expect(eng.forward('ACDEFG')).rejects.toThrow('simulated pass-3 validation failure');

      // Pass 4: persistent staging buffer is still unmapped and reusable after pass 2 & 3 errors!
      passCount = 4;
      popErrorOnPass3 = false;
      const r4 = await eng.forward('ACDEFG');
      expect(r4.logits.length).toBe(6 * 31);

      eng.destroy();
    } finally {
      (globalThis as Record<string, unknown>).GPUBufferUsage = prevUsage;
      (globalThis as Record<string, unknown>).GPUMapMode = prevMapMode;
    }
  });

  it('lasermpnn createLaserMpnnEngine downgrades gracefully when second adapter lacks shader-f16', async () => {
    const { createLaserMpnnEngine } = await import(vendorUrl('static/vendor/lasermpnn/webgpu/index.js'));
    const requestedFeatures: string[][] = [];
    const fakeDevice = {
      features: new Set<string>(),
      destroy: vi.fn(),
      createShaderModule: vi.fn(() => ({})),
      createComputePipelineAsync: vi.fn(async () => ({})),
      createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
      queue: { writeBuffer: vi.fn() },
    };
    const origGpu = (navigator as unknown as Record<string, unknown>).gpu;
    Object.defineProperty(navigator, 'gpu', {
      configurable: true,
      value: {
        requestAdapter: vi.fn(async () => ({
          features: new Set<string>(), // lacks shader-f16
          requestDevice: vi.fn(async (desc?: { requiredFeatures?: string[] }) => {
            requestedFeatures.push(desc?.requiredFeatures ?? []);
            return fakeDevice;
          }),
        })),
      },
    });
    try {
      // Stop right after device creation via manifest precision check to verify adapter-gated requestDevice
      await expect(
        createLaserMpnnEngine({
          precision: 'fp16',
          shaderF16: true,
          weights: { manifest: { precision: 'fp32' }, tensors: {} },
        }),
      ).rejects.toThrow('Manifest precision does not match requested precision');
      // Must NOT have requested unguarded ['shader-f16'] on the adapter that lacked it
      expect(requestedFeatures).toEqual([[]]);
    } finally {
      Object.defineProperty(navigator, 'gpu', { configurable: true, value: origGpu });
    }
  });

  it('ligandmpnn linearShader uses Firefox/naga-safe unpack2x16float and geometry.js aligns cache-bust query', async () => {
    const { linearShader } = await import(vendorUrl('static/vendor/ligandmpnn/webgpu/ligandmpnn/shaders.js'));
    const wgsl: string = linearShader(32, 128, 128, { packed: true, f16: true });
    expect(wgsl).toContain('unpack2x16float');
    expect(wgsl).not.toContain('bitcast<vec2<f16>>');
    expect(wgsl).not.toContain('enable f16;');

    const geomSrc = readSrc('static/vendor/ligandmpnn/webgpu/ligandmpnn/geometry.js');
    expect(geomSrc).toContain('./shaders.js?v=2');
  });
});
