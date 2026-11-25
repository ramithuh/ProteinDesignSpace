import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = process.cwd();

function readSrc(relPath: string): string {
  return readFileSync(resolve(ROOT, relPath), 'utf8');
}

const WG_ELEM_BYTES: Record<string, number> = {
  f32: 4,
  u32: 4,
  i32: 4,
  f16: 2,
  vec2f: 8,
  vec3f: 16,
  vec4f: 16,
  'vec2<f32>': 8,
  'vec3<f32>': 16,
  'vec4<f32>': 16,
  'vec2<f16>': 4,
  'vec4<f16>': 8,
};

function extractWorkgroupArrayBytes(wgsl: string): Array<{ name: string; bytes: number }> {
  const out: Array<{ name: string; bytes: number }> = [];
  const re = /var<workgroup>\s+(\w+)\s*:\s*array<\s*([^,>]+(?:<[^>]+>)?)\s*,\s*(\d+)u?\s*>/g;
  for (const m of wgsl.matchAll(re)) {
    const [, name, rawType, countStr] = m;
    const elemBytes = WG_ELEM_BYTES[rawType.replace(/\s+/g, '')];
    expect(elemBytes, `Unknown workgroup element type ${rawType}`).toBeDefined();
    out.push({ name, bytes: elemBytes * Number(countStr) });
  }
  return out;
}

describe('Track C — LocalFold WebGPU kernels, pacing & buffer pooling (rfd3, boltzgen, dyna1, gnina)', () => {
  it('rfd3: negotiates optional shader-f16, wires QueuePacer + completionPump + compilerText, and uses uniform-driven Naga-safe 64x128 / 32x64 GEMMs', async () => {
    const workerSrc = readSrc('src/lib/plugins/rfd3/rfd3.worker.ts');
    expect(workerSrc).toContain("adapter.features.has('shader-f16')");
    expect(workerSrc).toContain('requiredFeatures');
    expect(workerSrc).toContain('webgpu/index.js?v=2');

    const resSrc = readSrc('static/vendor/rfd3/webgpu/resources.js');
    expect(resSrc).toContain('shader-f16');
    expect(resSrc).toContain('startCompletionPump');
    expect(resSrc).not.toContain('recordCompilerSupport');

    const modelSrc = readSrc('static/vendor/rfd3/webgpu/model.js');
    expect(modelSrc).toContain('this.o.piece()');
    expect(modelSrc).toContain('o.piece()');

    const opsMod = await import(
      pathToFileURL(resolve(ROOT, 'static/vendor/rfd3/webgpu/ops.js')).href
    );
    const { buildRfd3LinearGemmShader, GPUOps } = opsMod;
    const { createGpuResources } = await import(
      pathToFileURL(resolve(ROOT, 'static/vendor/rfd3/webgpu/resources.js')).href
    );

    // Check all generated GEMM variants (32x64 & 64x128, F32/F16/I8, f16 panel, unaligned K, reluInput)
    for (const large of [false, true]) {
      for (const dtype of [0, 1, 2]) {
        for (const alignedB of [false, true]) {
          if (dtype === 0 && alignedB) continue;
          const useF16Panel = dtype === 1 && alignedB;
          const { code } = buildRfd3LinearGemmShader({
            large,
            dtype,
            hasBias: true,
            reluInput: large,
            alignedB,
            useF16Panel,
          });
          expect(code).toContain('struct GemmDims');
          expect(code).not.toContain('bitcast<vec2<f16>>');
          expect(code).not.toContain('bitcast<vec4<f16>>');
          if (useF16Panel) {
            expect(code).toContain('enable f16;');
            expect(code).toContain('vec4<f16>');
          }
          const wgArrays = extractWorkgroupArrayBytes(code);
          expect(wgArrays.length).toBeGreaterThanOrEqual(2);
          for (const arr of wgArrays) {
            expect(
              arr.bytes,
              `rfd3 workgroup array ${arr.name} (${arr.bytes}B) must stay <= 8192B for Firefox/Naga`,
            ).toBeLessThanOrEqual(8192);
          }
        }
      }
    }

    // Verify zero per-(m, n, k) shader recompilation across varying sequence lengths & clean trim()
    const compiledShaders: string[] = [];
    let submitCount = 0;
    let workDoneCalls = 0;
    const mockDevice = {
      adapterInfo: { vendor: 'apple', architecture: 'metal-3', isFallbackAdapter: false },
      features: new Set(['shader-f16']),
      lost: new Promise(() => {}),
      limits: {
        maxBufferSize: 512 * 1024 * 1024,
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        minStorageBufferOffsetAlignment: 16,
      },
      createBuffer: (desc: { size: number; usage: number; label?: string }) => ({
        ...desc,
        destroy() {},
      }),
      createShaderModule: ({ code }: { code: string }) => {
        compiledShaders.push(code);
        return { code };
      },
      createComputePipeline: ({ compute }: { compute: { module: { code: string } } }) => ({
        code: compute.module.code,
        getBindGroupLayout: () => ({}),
      }),
      createBindGroup: () => ({}),
      createCommandEncoder: () => ({
        beginComputePass: () => ({
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {},
        }),
        finish: () => ({}),
      }),
      queue: {
        writeBuffer() {},
        submit() {
          submitCount++;
        },
        onSubmittedWorkDone: async () => {
          workDoneCalls++;
        },
      },
      destroy() {},
    };

    const resources = await createGpuResources({ device: mockDevice });
    expect(resources.hasF16).toBe(true);
    const ops = new GPUOps(resources);

    ops.begin();
    const wBuffer = resources.allocate(128 * 128 * 2, 128 | 8, 'w16');
    const bBuffer = resources.allocate(128 * 4, 128 | 8, 'b32');
    const weight = { shape: [128, 128], dtype: 'F16', buffer: wBuffer };
    const bias = { buffer: bBuffer, length: 128 };

    // Three distinct M lengths (16, 28, 44) with same 32x64 F16+bias kernel -> ONLY 1 shader module compiled!
    const x1 = ops.tensor(16 * 128);
    const x2 = ops.tensor(28 * 128);
    const x3 = ops.tensor(44 * 128);
    const y1 = ops.linear(x1, weight, bias);
    ops.piece();
    const y2 = ops.linear(x2, weight, bias);
    ops.piece();
    const y3 = ops.linear(x3, weight, bias);
    ops.piece();
    await ops.submit();

    expect(compiledShaders).toHaveLength(1);
    expect(compiledShaders[0]).toContain('enable f16;');
    expect(submitCount).toBeGreaterThanOrEqual(3); // QueuePacer timed each piece on pass 0
    expect(workDoneCalls).toBeGreaterThanOrEqual(1);

    for (const t of [x1, x2, x3, y1, y2, y3]) ops.free(t);
    resources.release(wBuffer);
    resources.release(bBuffer);
    ops.trim();
    expect(resources.memoryUsage()).toBe(0);
  });

  it('boltzgen: paces all 64 Pairformer blocks + diffusion steps via QueuePacer and compiles shape-invariant 64x128 / 32x64 batched matmuls', async () => {
    const modelSrc = readSrc('static/vendor/boltzgen/shared/model.js');
    expect(modelSrc).toContain('o.backend.beginPacedBlock?.("trunk")');
    expect(modelSrc).toContain('await o.backend.paceBlock("trunk"');

    const samplerSrc = readSrc('static/vendor/boltzgen/shared/sampler.js');
    expect(samplerSrc).toContain('o.backend.beginPacedBlock?.("diffusion")');
    expect(samplerSrc).toContain('await o.backend.paceBlock("diffusion"');

    const backendSrc = readSrc('static/vendor/boltzgen/webgpu/backend.js');
    expect(backendSrc).toContain('QueuePacer');
    expect(backendSrc).toContain('this.pending >= 512');
    expect(backendSrc).toContain('yieldToBrowser');

    const indexSrc = readSrc('static/vendor/boltzgen/webgpu/index.js');
    expect(indexSrc).toContain('startCompletionPump');
    expect(indexSrc).not.toContain('recordCompilerSupport');

    const { shader } = await import(
      pathToFileURL(resolve(ROOT, 'static/vendor/boltzgen/webgpu/shaders.js')).href
    );
    const { tensor } = await import(
      pathToFileURL(resolve(ROOT, 'static/vendor/boltzgen/shared/tensor.js')).href
    );

    const mkTensor = (shape: number[], stride?: number[], encoding?: string) =>
      tensor({ buffer: {}, bytes: 4096, ...(encoding ? { encoding } : {}) }, shape, stride);

    // 1. Batched triangleMult shape [1, 128, N, N] @ [1, 128, N, N] -> 64x128 kernel, identical WGSL across N=96 and N=144!
    const tri96 = shader(
      'matmul',
      [mkTensor([1, 128, 96, 96]), mkTensor([1, 128, 96, 96])],
      [mkTensor([1, 128, 96, 96])],
      { hasF16: true },
    );
    const tri144 = shader(
      'matmul',
      [mkTensor([1, 128, 144, 144]), mkTensor([1, 128, 144, 144])],
      [mkTensor([1, 128, 144, 144])],
      { hasF16: true },
    );
    expect(tri96.code).toBe(tri144.code);
    expect(tri96.code).toContain('@workgroup_size(256)');
    expect(tri96.code).toContain('tb0:array<vec4<f32>,272>');
    expect(tri96.code).toContain('tb1:array<vec4<f32>,272>');
    for (const arr of extractWorkgroupArrayBytes(tri96.code)) {
      expect(arr.bytes).toBeLessThanOrEqual(8192);
    }

    // 2. Batched pair-bias attention Q @ K^T (transposed B: sBk=1) -> 32x64 kernel, identical WGSL across L=40 and L=72!
    const attn40 = shader(
      'matmul',
      [mkTensor([1, 16, 40, 32]), mkTensor([1, 16, 32, 40], [20480, 1280, 1, 32])],
      [mkTensor([1, 16, 40, 40])],
      { hasF16: true },
    );
    const attn72 = shader(
      'matmul',
      [mkTensor([1, 16, 72, 32]), mkTensor([1, 16, 32, 72], [36864, 2304, 1, 32])],
      [mkTensor([1, 16, 72, 72])],
      { hasF16: true },
    );
    expect(attn40.code).toBe(attn72.code);
    expect(attn40.code).toContain('@workgroup_size(256)');
    expect(attn40.code).toContain('ta:array<vec4<f32>,144>');
    expect(attn40.code).toContain('tb:array<vec4<f32>,272>');
    for (const arr of extractWorkgroupArrayBytes(attn40.code)) {
      expect(arr.bytes).toBeLessThanOrEqual(8192);
    }

    // 3. F16-encoded batched B stages tb in vec4<f16> when hasF16 is true
    const f16Batched = shader(
      'matmul',
      [mkTensor([1, 8, 32, 64]), mkTensor([1, 8, 64, 64], undefined, 'F16')],
      [mkTensor([1, 8, 32, 64])],
      { hasF16: true },
    );
    expect(f16Batched.code).toContain('enable f16;');
    expect(f16Batched.code).toContain('vec4<f16>');

    // 4. Linear 64x128 GEMM keeps every workgroup array <= 8192B even in INT8 / FP32 mode
    const linInt8 = shader(
      'matmul',
      [
        mkTensor([256, 384]),
        tensor(
          { buffer: {}, bytes: 4096, encoding: 'I8', columns: 384, scale: { buffer: {} } },
          [384, 768],
          [1, 384],
        ),
      ],
      [mkTensor([256, 768])],
      { hasF16: false },
    );
    for (const arr of extractWorkgroupArrayBytes(linInt8.code)) {
      expect(arr.bytes).toBeLessThanOrEqual(8192);
    }
  });

  it('dyna1: splits StructureTokenEncoder chunks, 23 ESM-3 blocks, and 12 head blocks via QueuePacer + MessageChannel yield', async () => {
    const engineSrc = readSrc('static/vendor/dyna1/webgpu/dyna1/engine.js');
    expect(engineSrc).toContain('QueuePacer');
    expect(engineSrc).toContain('startCompletionPump');
    expect(engineSrc).toContain('startCompletionPump');
    expect(engineSrc).not.toContain('recordCompilerSupport');
    expect(engineSrc).toContain('yieldToBrowser');
    expect(engineSrc).toContain('this.structPacer.pieceEnded');
    expect(engineSrc).toContain('this.backbonePacer.pieceEnded');
    expect(engineSrc).toContain('this.headPacer.pieceEnded');
    expect(engineSrc).toContain('await this._paceBlock(this.backbonePacer');
    expect(engineSrc).toContain('await this._paceBlock(this.headPacer');

    const { matmulSource } = await import(
      pathToFileURL(resolve(ROOT, 'static/vendor/dyna1/webgpu/dyna1/shaders.js')).href
    );
    for (const M of [64, 256]) {
      for (const dtype of ['F32', 'F16', 'I8']) {
        for (const useF16 of [false, true]) {
          const wgsl = matmulSource(M, 1536, 1536, dtype, true, useF16);
          expect(wgsl).not.toContain('bitcast<vec2<f16>>');
          for (const arr of extractWorkgroupArrayBytes(wgsl)) {
            expect(
              arr.bytes,
              `dyna1 M=${M} dtype=${dtype} workgroup array ${arr.name} (${arr.bytes}B) > 8192B`,
            ).toBeLessThanOrEqual(8192);
          }
        }
      }
    }
  });

  it('gnina: reuses persistent GPU buffers with 0 per-pose allocations after warmup, batches multi-pose scoring, and preserves exact BatchNorm3D+Conv3D parity', async () => {
    const [{ Resources }, { GPUOps }, { shaders }, { createScorer }, { parseReceptor, parseLigands }] =
      await Promise.all([
        import(pathToFileURL(resolve(ROOT, 'static/vendor/gnina/webgpu/device.js')).href),
        import(pathToFileURL(resolve(ROOT, 'static/vendor/gnina/webgpu/ops.js')).href),
        import(pathToFileURL(resolve(ROOT, 'static/vendor/gnina/webgpu/shaders.js')).href),
        import(pathToFileURL(resolve(ROOT, 'static/vendor/gnina/shared/api.js')).href),
        import(pathToFileURL(resolve(ROOT, 'static/vendor/gnina/shared/structure.js')).href),
      ]);

    // 1. Verify all GNINA shaders stay <= 8192B per workgroup array (Firefox/Naga safe)
    for (const [name, wgsl] of Object.entries(shaders as Record<string, string>)) {
      for (const arr of extractWorkgroupArrayBytes(wgsl)) {
        expect(arr.bytes, `gnina.${name} array ${arr.name} > 8192B`).toBeLessThanOrEqual(8192);
      }
    }

    // 2. Verify exact IEEE-754 parity of fused BatchNorm3D + Conv3D across interior & zero-padded 3D boundary voxels!
    const d = 5;
    const v = d * d * d;
    const cin = 3;
    const cout = 4;
    const k = 3;
    const k3 = k * k * k;
    const pad = 1;
    const f32 = Math.fround;

    const x = Float32Array.from({ length: cin * v }, (_, i) => f32(Math.sin(i * 0.37) * 2.5 - 0.4));
    const bnW = Float32Array.from({ length: cin }, (_, c) => f32(0.75 + 0.3 * c));
    const bnB = Float32Array.from({ length: cin }, (_, c) => f32(-0.25 + 0.4 * c));
    const bnM = Float32Array.from({ length: cin }, (_, c) => f32(0.15 - 0.2 * c));
    const bnV = Float32Array.from({ length: cin }, (_, c) => f32(0.4 + 0.5 * c));
    const convW = Float32Array.from({ length: cout * cin * k3 }, (_, i) =>
      f32(Math.cos(i * 0.19) * 0.35),
    );
    const convBias = Float32Array.from({ length: cout }, (_, c) => f32(0.1 * c - 0.15));

    // Unfused reference: BatchNorm3D over [cin, d, d, d], then zero-padded 3x3x3 Conv3D + ReLU
    const normOut = new Float32Array(cin * v);
    for (let c = 0; c < cin; c++) {
      const invStd = f32(1 / Math.sqrt(f32(bnV[c] + 1e-5)));
      for (let pos = 0; pos < v; pos++) {
        const idx = c * v + pos;
        normOut[idx] = f32(f32(f32(x[idx] - bnM[c]) * invStd) * bnW[c] + bnB[c]);
      }
    }
    const expectedOut = new Float32Array(cout * v);
    for (let oc = 0; oc < cout; oc++) {
      for (let pos = 0; pos < v; pos++) {
        const qx = Math.floor(pos / (d * d));
        const qy = Math.floor(pos / d) % d;
        const qz = pos % d;
        let acc = 0;
        for (let kidx = 0; kidx < cin * k3; kidx++) {
          const ch = Math.floor(kidx / k3);
          const rem = kidx % k3;
          const ax = qx + Math.floor(rem / (k * k)) - pad;
          const ay = qy + (Math.floor(rem / k) % k) - pad;
          const az = qz + (rem % k) - pad;
          let fv = 0;
          if (ax >= 0 && ax < d && ay >= 0 && ay < d && az >= 0 && az < d) {
            fv = normOut[((ch * d + ax) * d + ay) * d + az];
          }
          acc = f32(acc + f32(fv * convW[oc * cin * k3 + kidx]));
        }
        const raw = f32(acc + convBias[oc]);
        expectedOut[oc * v + pos] = raw >= 0 ? raw : 0;
      }
    }

    // Fused bnConv mirror matching shaders.conv applyBn() only on in-bounds coordinates
    const fusedOut = new Float32Array(cout * v);
    for (let oc = 0; oc < cout; oc++) {
      for (let pos = 0; pos < v; pos++) {
        const qx = Math.floor(pos / (d * d));
        const qy = Math.floor(pos / d) % d;
        const qz = pos % d;
        let acc = 0;
        for (let kidx = 0; kidx < cin * k3; kidx++) {
          const ch = Math.floor(kidx / k3);
          const rem = kidx % k3;
          const ax = qx + Math.floor(rem / (k * k)) - pad;
          const ay = qy + (Math.floor(rem / k) % k) - pad;
          const az = qz + (rem % k) - pad;
          let fv = 0;
          if (ax >= 0 && ax < d && ay >= 0 && ay < d && az >= 0 && az < d) {
            const rawX = x[((ch * d + ax) * d + ay) * d + az];
            const invStd = f32(1 / Math.sqrt(f32(bnV[ch] + 1e-5)));
            fv = f32(f32(f32(rawX - bnM[ch]) * invStd) * bnW[ch] + bnB[ch]);
          }
          acc = f32(acc + f32(fv * convW[oc * cin * k3 + kidx]));
        }
        const raw = f32(acc + convBias[oc]);
        fusedOut[oc * v + pos] = raw >= 0 ? raw : 0;
      }
    }
    expect(Array.from(fusedOut)).toEqual(Array.from(expectedOut));

    // 3. Verify persistent cross-pose buffer reuse, 0 norm dispatches, and 4-pose GPU batching
    let createBufferCalls = 0;
    let destroyBufferCalls = 0;
    let submitCalls = 0;
    const dispatchedKernels: string[] = [];

    const mockDevice = {
      adapterInfo: { vendor: 'nvidia', architecture: 'ampere', isFallbackAdapter: false },
      features: new Set(['shader-f16']),
      lost: new Promise(() => {}),
      limits: {
        maxBufferSize: 512 * 1024 * 1024,
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        maxComputeWorkgroupsPerDimension: 65535,
      },
      createBuffer: ({ size, usage }: { size: number; usage: number }) => {
        createBufferCalls++;
        const backing = new ArrayBuffer(size);
        // Seed deterministic finite cnnScore / cnnAffinity values in readback
        const fView = new Float32Array(backing);
        for (let i = 0; i < fView.length; i += 2) {
          fView[i] = 0.85;
          if (i + 1 < fView.length) fView[i + 1] = 6.2;
        }
        return {
          size,
          usage,
          getMappedRange: () => backing,
          unmap() {},
          mapAsync: async () => {},
          destroy() {
            destroyBufferCalls++;
          },
        };
      },
      createShaderModule: ({ label, code }: { label: string; code: string }) => ({ label, code }),
      createComputePipelineAsync: async ({ label }: { label: string }) => ({
        label,
        getBindGroupLayout: () => ({}),
      }),
      createBindGroup: () => ({}),
      createCommandEncoder: () => ({
        beginComputePass: () => {
          let currentLabel = '';
          return {
            setPipeline(p: { label: string }) {
              currentLabel = p.label;
            },
            setBindGroup() {},
            dispatchWorkgroups() {
              dispatchedKernels.push(currentLabel);
            },
            end() {},
          };
        },
        copyBufferToBuffer() {},
        finish: () => ({}),
      }),
      queue: {
        writeBuffer() {},
        submit() {
          submitCalls++;
        },
      },
      destroy() {},
    };

    // Build synthetic weights layout for dense_1.3 so Resources.initialize succeeds
    const dummyTensor = (shape: number[]) => ({
      shape,
      dtype: 'F32',
      bytes: new Uint8Array(shape.reduce((a, b) => a * b, 1) * 4),
    });
    const tensors: Record<string, ReturnType<typeof dummyTensor>> = {
      'dense_1.3/data_enc_init_conv.weight': dummyTensor([32, 28, 3, 3, 3]),
      'dense_1.3/data_enc_init_conv.bias': dummyTensor([32]),
      'dense_1.3/pose_output.weight': dummyTensor([2, 160]),
      'dense_1.3/pose_output.bias': dummyTensor([2]),
      'dense_1.3/affinity_output.weight': dummyTensor([1, 160]),
      'dense_1.3/affinity_output.bias': dummyTensor([1]),
    };
    let ch = 32;
    for (let level = 0; level < 3; level++) {
      const p = `dense_1.3/dense_block_${level}.data_enc_level${level}`;
      for (let i = 0; i < 4; i++) {
        for (const kName of ['weight', 'bias', 'running_mean', 'running_var']) {
          tensors[`${p}_batchnorm_conv${i}.${kName}`] = dummyTensor([ch]);
        }
        tensors[`${p}_conv${i}.weight`] = dummyTensor([16, ch, 3, 3, 3]);
        tensors[`${p}_conv${i}.bias`] = dummyTensor([16]);
        ch += 16;
      }
      if (level < 2) {
        tensors[`dense_1.3/data_enc_level${level}_bottleneck.weight`] = dummyTensor([ch, ch, 1, 1, 1]);
        tensors[`dense_1.3/data_enc_level${level}_bottleneck.bias`] = dummyTensor([ch]);
      }
    }

    const resources = new Resources({
      device: mockDevice,
      owns: false,
      adapterInfo: mockDevice.adapterInfo,
      isFallbackAdapter: false,
    });
    await resources.initialize(tensors, ['dense_1.3']);
    resources.begin = (check: () => void) => new GPUOps(resources, check);
    const scorer = createScorer(resources, ['dense_1.3']);

    const receptor = parseReceptor(
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n',
      'pdb',
    );
    const [basePose] = parseLigands(
      'LIG\n\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$',
      'sdf',
    );

    // Warmup single pose: zero standalone 'norm' dispatches (all 12 fused into 'conv'!)
    dispatchedKernels.length = 0;
    await scorer.scorePose(receptor, basePose);
    expect(dispatchedKernels.filter((k) => k === 'norm')).toHaveLength(0);
    expect(dispatchedKernels.filter((k) => k === 'conv')).toHaveLength(15); // 1 stem + 12 bnConv + 2 bottleneck

    // Sequential single-pose scoring across 5 more poses -> ZERO new createBuffer or destroyBuffer calls!
    const buffersAfterWarmup = createBufferCalls;
    const destroysAfterWarmup = destroyBufferCalls;
    for (let i = 0; i < 5; i++) {
      await scorer.scorePose(receptor, basePose);
    }
    expect(createBufferCalls).toBe(buffersAfterWarmup);
    expect(destroyBufferCalls).toBe(destroysAfterWarmup);
    expect(resources.memoryUsage().poolHits).toBeGreaterThan(20);

    // Multi-pose scorePoses(6 poses) batches into ceil(6 / 4) = 2 submits!
    const submitsBeforeBatch = submitCalls;
    const sixPoses = Array.from({ length: 6 }, () => basePose);
    const batchRes = await scorer.scorePoses(receptor, sixPoses);
    expect(batchRes).toHaveLength(6);
    expect(submitCalls - submitsBeforeBatch).toBe(2);
    expect(destroyBufferCalls).toBe(destroysAfterWarmup);

    // Clean teardown frees 100% of resident + pooled buffers
    scorer.destroy();
    expect(resources.memoryUsage().buffers).toBe(0);
    expect(resources.memoryUsage().totalBytes).toBe(0);
  });

  it('gnina: low maxPoolBytes (16 MiB) defers evicted activation destroy() until after queue.submit() (no use-after-destroy)', async () => {
    const [{ Resources }, { GPUOps }, { createScorer }, { parseReceptor, parseLigands }] =
      await Promise.all([
        import(pathToFileURL(resolve(ROOT, 'static/vendor/gnina/webgpu/device.js')).href),
        import(pathToFileURL(resolve(ROOT, 'static/vendor/gnina/webgpu/ops.js')).href),
        import(pathToFileURL(resolve(ROOT, 'static/vendor/gnina/shared/api.js')).href),
        import(pathToFileURL(resolve(ROOT, 'static/vendor/gnina/shared/structure.js')).href),
      ]);

    const destroyedSet = new Set<object>();
    let referencedInPendingEncoder = new Set<object>();
    let usedDestroyedInSubmit = 0;

    const mockDevice = {
      adapterInfo: { vendor: 'nvidia', architecture: 'blackwell', isFallbackAdapter: false },
      features: new Set(['shader-f16']),
      lost: new Promise(() => {}),
      limits: {
        maxBufferSize: 512 * 1024 * 1024,
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        maxComputeWorkgroupsPerDimension: 65535,
      },
      createBuffer: (desc: { size: number; usage: number }) => {
        const ab = new ArrayBuffer(desc.size);
        const buf = {
          size: desc.size,
          usage: desc.usage,
          getMappedRange: () => ab,
          mapAsync: async () => {},
          unmap() {},
          destroy() {
            destroyedSet.add(buf);
          },
        };
        return buf;
      },
      createShaderModule: ({ code }: { code: string }) => ({ code }),
      createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
      createBindGroup: ({ entries }: { entries: Array<{ resource: { buffer: object } }> }) => {
        for (const e of entries) referencedInPendingEncoder.add(e.resource.buffer);
        return { entries };
      },
      createCommandEncoder: () => ({
        beginComputePass: () => ({
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {},
        }),
        copyBufferToBuffer(src: object, _so: number, dst: object) {
          referencedInPendingEncoder.add(src);
          referencedInPendingEncoder.add(dst);
        },
        finish: () => {
          const snapshot = referencedInPendingEncoder;
          referencedInPendingEncoder = new Set();
          return { snapshot };
        },
      }),
      queue: {
        writeBuffer() {},
        submit(cmds: Array<{ snapshot: Set<object> }>) {
          for (const c of cmds) {
            for (const b of c.snapshot) {
              if (destroyedSet.has(b)) usedDestroyedInSubmit++;
            }
          }
        },
      },
      destroy() {},
    };

    const dummyTensor = (shape: number[]) => ({
      shape,
      dtype: 'F32',
      bytes: new Uint8Array(shape.reduce((a, b) => a * b, 1) * 4),
    });
    const tensors: Record<string, ReturnType<typeof dummyTensor>> = {
      'dense_1.3/data_enc_init_conv.weight': dummyTensor([32, 28, 3, 3, 3]),
      'dense_1.3/data_enc_init_conv.bias': dummyTensor([32]),
      'dense_1.3/pose_output.weight': dummyTensor([2, 160]),
      'dense_1.3/pose_output.bias': dummyTensor([2]),
      'dense_1.3/affinity_output.weight': dummyTensor([1, 160]),
      'dense_1.3/affinity_output.bias': dummyTensor([1]),
    };
    let ch = 32;
    for (let level = 0; level < 3; level++) {
      const p = `dense_1.3/dense_block_${level}.data_enc_level${level}`;
      for (let i = 0; i < 4; i++) {
        for (const kName of ['weight', 'bias', 'running_mean', 'running_var']) {
          tensors[`${p}_batchnorm_conv${i}.${kName}`] = dummyTensor([ch]);
        }
        tensors[`${p}_conv${i}.weight`] = dummyTensor([16, ch, 3, 3, 3]);
        tensors[`${p}_conv${i}.bias`] = dummyTensor([16]);
        ch += 16;
      }
      if (level < 2) {
        tensors[`dense_1.3/data_enc_level${level}_bottleneck.weight`] = dummyTensor([ch, ch, 1, 1, 1]);
        tensors[`dense_1.3/data_enc_level${level}_bottleneck.bias`] = dummyTensor([ch]);
      }
    }

    // Force heavy mid-batch activation evictions via maxPoolBytes = 16 MiB
    const resources = new Resources(
      {
        device: mockDevice,
        owns: false,
        adapterInfo: mockDevice.adapterInfo,
        isFallbackAdapter: false,
      },
      { maxPoolBytes: 16 * 1024 * 1024 },
    );
    await resources.initialize(tensors, ['dense_1.3']);
    resources.begin = (check: () => void) => new GPUOps(resources, check);
    const scorer = createScorer(resources, ['dense_1.3']);
    const receptor = parseReceptor(
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n',
      'pdb',
    );
    const [basePose] = parseLigands(
      'LIG\n\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$',
      'sdf',
    );

    await scorer.scorePoses(receptor, [basePose, basePose, basePose, basePose]);
    expect(usedDestroyedInSubmit).toBe(0);
    expect(resources.deferredDestroy).toHaveLength(0);
    scorer.destroy();
  });

  it('rfd3: narrow-N attention-bias (N=16) selects 32x64 tile (not 64x128) and GPUOps.abort() resets pacer state', async () => {
    const { GPUOps } = await import(
      pathToFileURL(resolve(ROOT, 'static/vendor/rfd3/webgpu/ops.js')).href
    );
    const { createGpuResources } = await import(
      pathToFileURL(resolve(ROOT, 'static/vendor/rfd3/webgpu/resources.js')).href
    );
    const compiled: string[] = [];
    const mockDevice = {
      adapterInfo: { vendor: 'nvidia', architecture: 'blackwell', isFallbackAdapter: false },
      features: new Set(['shader-f16']),
      lost: new Promise(() => {}),
      limits: {
        maxBufferSize: 512 * 1024 * 1024,
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        minStorageBufferOffsetAlignment: 16,
      },
      createBuffer: (desc: { size: number; usage: number }) => ({ ...desc, destroy() {} }),
      createShaderModule: ({ code }: { code: string }) => {
        compiled.push(code);
        return { code };
      },
      createComputePipeline: ({ compute }: { compute: { module: { code: string } } }) => ({
        code: compute.module.code,
        getBindGroupLayout: () => ({}),
      }),
      createBindGroup: () => ({}),
      createCommandEncoder: () => ({
        beginComputePass: () => ({
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {},
        }),
        finish: () => ({}),
      }),
      queue: { writeBuffer() {}, submit() {}, onSubmittedWorkDone: async () => {} },
      destroy() {},
    };
    const res = await createGpuResources({ device: mockDevice });
    const ops = new GPUOps(res);
    ops.begin();
    // to_b shape: m = 256*256 = 65536 (lots of tiles), n = 16 attention heads -> MUST use 32x64 (no tb1!)
    const x = ops.tensor(4096 * 128);
    const wBuf = res.allocate(16 * 128 * 4, 128 | 8, 'to_b');
    const y = ops.linear(x, { shape: [16, 128], dtype: 'F32', buffer: wBuf });
    expect(compiled).toHaveLength(1);
    expect(compiled[0]).not.toContain('var<workgroup> tb1:'); // 32x64 has single tb, 64x128 has tb0/tb1

    // abort() must reset pieceUsed and replace pacer cleanly
    ops.piece();
    expect(ops.pieceUsed).toBe(true);
    const prevPacer = ops.pacer;
    ops.abort();
    expect(ops.pieceUsed).toBe(false);
    expect(ops.pacer).not.toBe(prevPacer);

    ops.free(x);
    ops.free(y);
    res.release(wBuf);
    ops.trim();
  });

  it('boltzgen: GPUBackend enforces 512-dispatch hard cap even when activePacer is active', async () => {
    const { GPUBackend } = await import(
      pathToFileURL(resolve(ROOT, 'static/vendor/boltzgen/webgpu/backend.js')).href
    );
    let submits = 0;
    const mockDevice = {
      features: new Set<string>(),
      limits: {
        maxBufferSize: 256 * 1024 * 1024,
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
      },
      addEventListener() {},
      removeEventListener() {},
      createBuffer: (desc: { size: number; usage: number }) => ({ ...desc, destroy() {} }),
      createShaderModule: ({ code }: { code: string }) => ({ code }),
      createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
      createBindGroup: () => ({}),
      createCommandEncoder: () => ({
        beginComputePass: () => ({
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {},
        }),
        finish: () => ({}),
      }),
      queue: {
        writeBuffer() {},
        submit() {
          submits++;
        },
        onSubmittedWorkDone: async () => {},
      },
    };
    const prevUsage = (globalThis as Record<string, unknown>).GPUBufferUsage;
    (globalThis as Record<string, unknown>).GPUBufferUsage = {
      STORAGE: 0x80,
      UNIFORM: 0x40,
      COPY_SRC: 0x04,
      COPY_DST: 0x08,
      MAP_READ: 0x01,
    };
    try {
      const { tensor } = await import(
        pathToFileURL(resolve(ROOT, 'static/vendor/boltzgen/shared/tensor.js')).href
      );
      const backend = new GPUBackend(mockDevice);
      backend.beginPacedBlock('trunk');
      expect(backend.activePacer).not.toBeNull();
      const x = tensor(backend.storage(16), [4]);
      for (let i = 0; i < 512; i++) {
        const y = backend.compute('copy', [x], [[4]]);
        backend.release(y);
      }
      expect(submits).toBeGreaterThanOrEqual(1);
      backend.release(x);
      backend.destroy();
    } finally {
      (globalThis as Record<string, unknown>).GPUBufferUsage = prevUsage;
    }
  });
});
