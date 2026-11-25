import { createGpuResources } from './resources.js';

const code = `
struct Dims { m: u32, n: u32, k: u32, dtype: u32, bias: u32, tilesN: u32, totalTiles: u32, pad0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> d: Dims;
var<workgroup> ta: array<vec4<f32>, 144>;
var<workgroup> tb: array<vec4<f32>, 272>;
fn weight(index: u32, row: u32) -> f32 {
  if (d.dtype == 0u) { return bitcast<f32>(w[index]); }
  if (d.dtype == 1u) { let v = unpack2x16float(w[index >> 1u]); return select(v.x, v.y, (index & 1u) != 0u); }
  let byte = (w[index >> 2u] >> (8u * (index & 3u))) & 255u;
  return f32(bitcast<i32>(byte << 24u) >> 24u) * scales[row];
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let flat = wid.x + wid.y * 65535u;
  let inTile = flat < d.totalTiles;
  let tile = min(flat, d.totalTiles - 1u);
  let m0 = (tile / d.tilesN) * 32u;
  let n0 = (tile % d.tilesN) * 64u;
  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * 2u;
  let r4Read = tm4 >> 1u;
  let hiPair = (tm4 & 1u) != 0u;
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  let kkA = li % 16u;
  let r4A = li / 16u;
  let rowA0 = m0 + r4A * 4u;
  let fullA = rowA0 + 3u < d.m;
  let kkB = li % 16u;
  let r4B = li / 16u;
  let rowB0 = n0 + r4B * 4u;
  let fullB = rowB0 + 3u < d.n;
  for (var k0 = 0u; k0 < d.k; k0 += 16u) {
    if (li < 128u) {
      var av = vec4<f32>(0.0);
      if (inTile && k0 + kkA < d.k) {
        let aOff = rowA0 * d.k + k0 + kkA;
        if (fullA) {
          av = vec4<f32>(x[aOff], x[aOff + d.k], x[aOff + 2u * d.k], x[aOff + 3u * d.k]);
        } else {
          if (rowA0 < d.m) { av.x = x[aOff]; }
          if (rowA0 + 1u < d.m) { av.y = x[aOff + d.k]; }
          if (rowA0 + 2u < d.m) { av.z = x[aOff + 2u * d.k]; }
          if (rowA0 + 3u < d.m) { av.w = x[aOff + 3u * d.k]; }
        }
      }
      ta[kkA * 9u + r4A] = av;
    }
    {
      var bv = vec4<f32>(0.0);
      let gk = k0 + kkB;
      if (inTile && gk < d.k) {
        let bOff = rowB0 * d.k + gk;
        if (fullB) {
          bv = vec4<f32>(weight(bOff, rowB0), weight(bOff + d.k, rowB0 + 1u), weight(bOff + 2u * d.k, rowB0 + 2u), weight(bOff + 3u * d.k, rowB0 + 3u));
        } else {
          if (rowB0 < d.n) { bv.x = weight(bOff, rowB0); }
          if (rowB0 + 1u < d.n) { bv.y = weight(bOff + d.k, rowB0 + 1u); }
          if (rowB0 + 2u < d.n) { bv.z = weight(bOff + 2u * d.k, rowB0 + 2u); }
          if (rowB0 + 3u < d.n) { bv.w = weight(bOff + 3u * d.k, rowB0 + 3u); }
        }
      }
      tb[kkB * 17u + r4B] = bv;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let av4 = ta[kk * 9u + r4Read];
      let av = select(av4.xy, av4.zw, hiPair);
      let bv = tb[kk * 17u + tx];
      acc0 = fma(vec4<f32>(av.x), bv, acc0);
      acc1 = fma(vec4<f32>(av.y), bv, acc1);
    }
    workgroupBarrier();
  }
  if (!inTile) { return; }
  let col0 = n0 + tx * 4u;
  var bVec = vec4<f32>(0.0);
  if (d.bias != 0u) {
    if (col0 < d.n) { bVec.x = bias[col0]; }
    if (col0 + 1u < d.n) { bVec.y = bias[col0 + 1u]; }
    if (col0 + 2u < d.n) { bVec.z = bias[col0 + 2u]; }
    if (col0 + 3u < d.n) { bVec.w = bias[col0 + 3u]; }
  }
  for (var ri = 0u; ri < 2u; ri++) {
    let m = m0 + tm + ri;
    if (m < d.m) {
      let res = select(acc0, acc1, ri == 1u) + bVec;
      let rowOut = m * d.n;
      if (col0 < d.n) { out[rowOut + col0] = res.x; }
      if (col0 + 1u < d.n) { out[rowOut + col0 + 1u] = res.y; }
      if (col0 + 2u < d.n) { out[rowOut + col0 + 2u] = res.z; }
      if (col0 + 3u < d.n) { out[rowOut + col0 + 3u] = res.w; }
    }
  }
}`;

export async function createPackedLinearRuntime(options = {}) {
  const arena = options.resources ?? await createGpuResources(options);
  const { device } = arena, weights = new Set();
  let destroyed = false;
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code, label: 'RFD3 packed linear' });
  let pipeline;
  try {
    pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  } catch (error) {
    await device.popErrorScope(); if (!options.resources) arena.destroy(); throw error;
  }
  const error = await device.popErrorScope();
  if (error) { if (!options.resources) arena.destroy(); throw new Error(error.message); }
  function upload(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const buffer = arena.allocate(bytes.byteLength || 4, usage);
    // The final word is padded only on upload. Narrow weights never expand to FP32.
    const padded = new Uint8Array(buffer.size); padded.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    device.queue.writeBuffer(buffer, 0, padded); return buffer;
  }
  const zero = upload(new Float32Array(1));
  return {
    resources: arena,
    uploadWeight(tensor) {
      if (destroyed || arena.closed) throw new Error('Runtime is closed');
      if (!['F32', 'F16', 'I8'].includes(tensor.dtype) || tensor.shape.length !== 2 ||
          !tensor.shape.every(n => Number.isSafeInteger(n) && n > 0)) throw new Error('Invalid linear weight');
      const expected = tensor.shape[0] * tensor.shape[1] * ({ F32: 4, F16: 2, I8: 1 }[tensor.dtype]);
      if (tensor.bytes.byteLength !== expected || tensor.dtype === 'I8' && tensor.scales?.byteLength !== tensor.shape[0] * 4)
        throw new Error('Invalid packed weight bytes');
      let buffer, scales;
      try {
        buffer = upload(tensor.bytes);
        scales = tensor.scales ? upload(tensor.scales) : zero;
        const weight = { shape: [...tensor.shape], dtype: { F32: 0, F16: 1, I8: 2 }[tensor.dtype], buffer, scales };
        weights.add(weight); return weight;
      } catch (error) { if (buffer) arena.release(buffer); if (scales && scales !== zero) arena.release(scales); throw error; }
    },
    releaseWeight(weight) {
      if (!weights.delete(weight)) return;
      arena.release(weight.buffer); if (weight.scales !== zero) arena.release(weight.scales);
    },
    async linear(input, weight, bias, { signal } = {}) {
      signal?.throwIfAborted();
      if (destroyed || !weights.has(weight)) throw new Error('Weight is not live in this runtime');
      const [n, k] = weight.shape, m = input.length / k;
      if (!(input instanceof Float32Array) || !Number.isSafeInteger(m) || m < 1 || bias && bias.length !== n)
        throw new Error('Invalid linear input');
      const local = [];
      const keep = b => { local.push(b); return b; };
      device.pushErrorScope('validation');
      let popped = false;
      try {
        const x = keep(upload(input)), biasBuffer = bias ? keep(upload(bias)) : zero;
        const output = keep(arena.allocate(m * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC));
        const readback = keep(arena.allocate(m * n * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ));
        const tilesM = Math.ceil(m / 32), tilesN = Math.ceil(n / 64), totalTiles = tilesM * tilesN;
        const gx = totalTiles <= 65535 ? Math.max(1, totalTiles) : 65535;
        const gy = totalTiles <= 65535 ? 1 : Math.ceil(totalTiles / 65535);
        const uniform = keep(upload(new Uint32Array([m, n, k, weight.dtype, bias ? 1 : 0, tilesN, totalTiles, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
        const entries = [x, weight.buffer, weight.scales, biasBuffer, output, uniform].map((buffer, binding) =>
          ({ binding, resource: binding === 5 ? { buffer } : arena.storageBinding(buffer) }));
        const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(gx, gy); pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, m * n * 4);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        signal?.throwIfAborted();
        const result = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap();
        const validation = await device.popErrorScope(); popped = true;
        if (validation) throw new Error(validation.message);
        return result;
      } finally {
        if (!popped) await device.popErrorScope();
        for (const buffer of local) arena.release(buffer);
      }
    },
    destroy() {
      if (destroyed) return; destroyed = true;
      for (const weight of [...weights]) this.releaseWeight(weight);
      arena.release(zero); if (!options.resources) arena.destroy();
    },
  };
}
