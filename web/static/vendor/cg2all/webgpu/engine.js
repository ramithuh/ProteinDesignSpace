// cg2all SE(3)-Transformer forward pass on WebGPU (f32 compute; fp16/int8 bundles are dequantized on load).
// Six shape-independent pipelines (shaders.js) run off 256-byte uniform slots with dynamic offsets into one
// activation arena, so a forward pass for any (N, E) is one bind group plus ~150 recorded dispatches.

import { inputFiber, NUM_HEADS } from '../shared/model.js';
import { KERNELS } from './shaders.js';

const SLOT_BYTES = 256;
const NONE = 0xffffffff;
/** Edges per conv dispatch: bounds a single dispatch to 256 workgroups on large proteins. */
const EDGE_TILE = 4096;
const MLP_HEAD_FLOATS = 128 + 32 * 3 + 1024 + 32 * 3; // w0 b0 g0 be0 w1 b1 g1 be1 → w2 rows follow
/** Node-slot channel capacity per degree (largest fiber: pass degree 0 = 64, hidden = 32 / 32). */
const NODE_CAP = [64, 32, 32];
const EDGE_CAP = [64, 32, 32];
const dimOf = (d) => 2 * d + 1;
const degreesOf = (fiber) => Object.keys(fiber).map(Number).sort((a, b) => a - b);

export function yieldToBrowser() {
  if (typeof MessageChannel === 'undefined') return new Promise((resolve) => setTimeout(resolve, 0));
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => resolve();
    ch.port2.postMessage(undefined);
  });
}

/** Bounded submission queue: at most `maxInFlight` command buffers queued, yielding to the event loop between. */
export class QueuePacer {
  constructor(device, maxInFlight = 2) {
    this.device = device;
    this.maxInFlight = maxInFlight;
    this.inFlight = [];
  }

  async submit(commandBuffer) {
    this.device.queue.submit([commandBuffer]);
    this.inFlight.push(this.device.queue.onSubmittedWorkDone());
    while (this.inFlight.length > this.maxInFlight) await this.inFlight.shift();
    await yieldToBrowser();
  }

  async drain() {
    await Promise.all(this.inFlight.splice(0));
  }
}

/** Concatenate every weight array the kernels read into one Float32Array (rows 16-byte aligned). */
function packWeights(net) {
  const offsets = new Map();
  const parts = [];
  let size = 0;
  const put = (arr) => {
    if (offsets.has(arr)) return offsets.get(arr);
    size = Math.ceil(size / 4) * 4;
    offsets.set(arr, size);
    parts.push([size, arr]);
    size += arr.length;
    return offsets.get(arr);
  };
  const mlpOffsets = new Map();
  const putMlp = (m) => {
    size = Math.ceil(size / 4) * 4;
    const base = size;
    for (const a of [m.w0, m.b0, m.g0, m.be0, m.w1, m.b1, m.g1, m.be1]) {
      parts.push([size, a]);
      size += a.length;
    }
    if (size - base !== MLP_HEAD_FLOATS) throw new Error('cg2all: unexpected radial MLP layout');
    parts.push([size, m.w2]);
    mlpOffsets.set(m, { mlpOff: base, w2Off: size });
    size += m.w2.length;
  };
  const seqs = [...net.init, ...net.struct, ...net.layers.flatMap((l) => [l.query, l.project, l.norm])];
  for (const op of seqs) {
    for (const p of Object.values(op.degrees)) {
      if (op.kind === 'norm') {
        put(p.gamma);
        put(p.beta);
      } else put(p.w);
    }
  }
  for (const conv of [...net.layers.map((l) => l.kvConv), net.final]) {
    for (const m of conv.mlps) putMlp(m);
    for (const k of Object.values(conv.self ?? {})) put(k);
  }
  const data = new Float32Array(Math.ceil(size / 4) * 4);
  for (const [o, a] of parts) data.set(a, o);
  return { data, off: (arr) => offsets.get(arr), mlp: (m) => mlpOffsets.get(m) };
}

class UniformWriter {
  constructor() {
    this.buf = new ArrayBuffer(SLOT_BYTES * 64);
    this.count = 0;
  }

  slot(values) {
    if ((this.count + 1) * SLOT_BYTES > this.buf.byteLength) {
      const next = new ArrayBuffer(this.buf.byteLength * 2);
      new Uint8Array(next).set(new Uint8Array(this.buf));
      this.buf = next;
    }
    const view = new DataView(this.buf, this.count * SLOT_BYTES, SLOT_BYTES);
    values.forEach((v, i) => {
      if (typeof v === 'object') view.setFloat32(i * 4, v.f32, true);
      else view.setUint32(i * 4, v >>> 0, true);
    });
    return this.count++;
  }
}

function dispatchSize(workgroups) {
  const x = Math.min(workgroups, 65535);
  return [Math.max(1, x), Math.max(1, Math.ceil(workgroups / x))];
}

export async function createGpuEngine({ device, net }) {
  if (!device) throw new Error('cg2all WebGPU engine requires a GPUDevice');
  let lost = null;
  device.lost.then(
    (info) => {
      lost = info;
    },
    () => {},
  );
  const packed = packWeights(net);
  const weightBuffer = device.createBuffer({
    label: 'cg2all-weights',
    size: packed.data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuffer, 0, packed.data);

  const layout = device.createBindGroupLayout({
    label: 'cg2all-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const names = Object.keys(KERNELS);
  const compiled = await Promise.all(
    names.map((name) =>
      device.createComputePipelineAsync({
        label: `cg2all-${name}`,
        layout: pipelineLayout,
        compute: { module: device.createShaderModule({ label: `cg2all-${name}`, code: KERNELS[name] }), entryPoint: 'main' },
      }),
    ),
  );
  const pipelines = Object.fromEntries(names.map((n, i) => [n, compiled[i]]));
  const limits = device.limits;
  const owned = new Set([weightBuffer]);
  let arena = null;
  let destroyed = false;

  const createBuffer = async (desc) => {
    device.pushErrorScope('out-of-memory');
    const buf = device.createBuffer(desc);
    const err = await device.popErrorScope();
    if (err) {
      buf.destroy();
      throw new Error(`WebGPU out of memory allocating ${desc.size} bytes for ${desc.label}: ${err.message}`);
    }
    owned.add(buf);
    return buf;
  };
  const release = (buf) => {
    if (!buf) return;
    owned.delete(buf);
    buf.destroy();
  };
  const assertAlive = () => {
    if (destroyed) throw new Error('cg2all WebGPU engine was destroyed');
    if (lost) throw new Error(`WebGPU device lost (${lost.reason ?? 'unknown'}): ${lost.message ?? ''}`);
  };

  /** Arena + uniform plan of every dispatch for one prepared graph. */
  function plan(prep) {
    const { n, e } = prep;
    let size = 0;
    const alloc = (count) => {
      size = Math.ceil(size / 4) * 4;
      const o = size;
      size += Math.max(count, 4);
      return o;
    };
    const nodeSlot = () => {
      const base = alloc(n * NODE_CAP.reduce((s, c, d) => s + c * dimOf(d), 0));
      let o = base;
      return NODE_CAP.map((c, d) => {
        const at = o;
        o += n * c * dimOf(d);
        return at;
      });
    };
    const slots = [nodeSlot(), nodeSlot(), nodeSlot()];
    const qSlot = nodeSlot();
    const zSlot = nodeSlot();
    const kvBase = alloc(e * EDGE_CAP.reduce((s, c, d) => s + c * dimOf(d), 0));
    const kv = [kvBase, kvBase + e * EDGE_CAP[0], kvBase + e * (EDGE_CAP[0] + EDGE_CAP[1] * 3)];
    const csrOff = e;
    const csrEdges = e + n + 1;
    const featOff = prep.basis.data.length;
    const pairOf = new Map(prep.basis.pairs.map((p) => [`${p.dIn},${p.dOut}`, p]));

    const u = new UniformWriter();
    const stages = [];
    let cur = [];
    const emit = (kernel, slotIndex, workgroups) => cur.push({ kernel, slot: slotIndex, wg: dispatchSize(workgroups) });
    const endStage = () => {
      stages.push(cur);
      cur = [];
    };
    const runSeq = (seq, from, to, other) => {
      // ping-pong `from` → (`to`, `other`) ... returns the slot holding the result
      let a = from;
      let b = to;
      for (const op of seq) {
        for (const [dStr, p] of Object.entries(op.degrees)) {
          const d = Number(dStr);
          const M = dimOf(d);
          if (op.kind === 'norm') {
            emit('norm', u.slot([n, p.C, M, a[d], b[d], packed.off(p.gamma), packed.off(p.beta)]), Math.ceil(n / 64));
          } else {
            emit('linear', u.slot([n * p.cOut * M, p.cOut, p.cIn, 0, M, a[d], 0, b[d], packed.off(p.w)]), Math.ceil((n * p.cOut * M) / 64));
          }
        }
        const next = a === from ? other : a;
        a = b;
        b = next;
      }
      return a;
    };
    const conv = (spec, h) => {
      for (const unit of spec.units) {
        const K = dimOf(unit.dOut);
        const blocks = unit.blocks.map((b) => {
          const pair = b.dIn === 0 && unit.dOut === 0 ? null : pairOf.get(`${b.dIn},${unit.dOut}`);
          const { mlpOff, w2Off } = packed.mlp(spec.mlps[b.mlp]);
          return [b.dIn, b.cIn, dimOf(b.dIn), b.F, b.fOff, b.fCount, h[b.dIn], pair ? pair.offset : NONE, pair ? pair.size : 0, mlpOff, w2Off, 0];
        });
        while (blocks.length < 3) blocks.push(new Array(12).fill(0));
        for (let base = 0; base < e; base += EDGE_TILE) {
          const tile = Math.min(EDGE_TILE, e - base);
          const s = u.slot([e, base, unit.cOut, K, kv[unit.dOut], unit.blocks.length, featOff, 0, ...blocks.flat()]);
          emit('conv', s, Math.ceil(tile / 16));
        }
      }
    };

    // input fiber lives in slots[0]; initialization module ping-pongs through slots[1] / slots[2]
    const x0 = runSeq(net.init, slots[0], slots[1], slots[2]);
    endStage();
    const free = slots.filter((s) => s !== x0);
    let h = x0;
    for (const layer of net.layers) {
      conv(layer.kvConv, h);
      for (const [dStr, p] of Object.entries(layer.query.degrees)) {
        const d = Number(dStr);
        const M = dimOf(d);
        emit('linear', u.slot([n * p.cOut * M, p.cOut, p.cIn, 0, M, h[d], 0, qSlot[d], packed.off(p.w)]), Math.ceil((n * p.cOut * M) / 64));
      }
      const deg = [0, 1, 2];
      const cv = deg.map((d) => layer.value[d] ?? 0);
      const ckv = deg.map((d) => layer.kvConv.fiberOut[d] ?? 0);
      const hasKey = deg.map((d) => (layer.key[d] ? 1 : 0));
      emit(
        'attn',
        u.slot([n, NUM_HEADS, csrOff, csrEdges, ...cv, 0, ...ckv, 0, ...hasKey, 0, ...kv, 0, ...qSlot, 0, ...zSlot, 0, { f32: 1 / Math.sqrt(layer.keyFeatures) }, 3, 0, 0]),
        Math.ceil((n * NUM_HEADS) / 64),
      );
      const out = free.find((s) => s !== h);
      for (const [dStr, p] of Object.entries(layer.project.degrees)) {
        const d = Number(dStr);
        const M = dimOf(d);
        const cb = layer.fiberIn[d] ?? 0;
        emit(
          'linear',
          u.slot([n * p.cOut * M, p.cOut, cv[d], cb, M, zSlot[d], cb ? h[d] : 0, out[d], packed.off(p.w)]),
          Math.ceil((n * p.cOut * M) / 64),
        );
      }
      const normed = free.find((s) => s !== out);
      for (const [dStr, p] of Object.entries(layer.norm.degrees)) {
        const d = Number(dStr);
        emit('norm', u.slot([n, p.C, dimOf(d), out[d], normed[d], packed.off(p.gamma), packed.off(p.beta)]), Math.ceil(n / 64));
      }
      h = normed;
      endStage();
    }
    conv(net.final, h);
    const pooled = free.find((s) => s !== h);
    for (const unit of net.final.units) {
      const K = dimOf(unit.dOut);
      const ks = net.final.self[unit.dOut];
      const cIn = net.final.fiberIn[unit.dOut] ?? 0;
      emit(
        'pool',
        u.slot([n * unit.cOut * K, unit.cOut, K, cIn, kv[unit.dOut], h[unit.dOut], pooled[unit.dOut], ks ? packed.off(ks) : 0, ks ? 1 : 0, csrOff, csrEdges]),
        Math.ceil((n * unit.cOut * K) / 64),
      );
    }
    for (const d of degreesOf(net.fibers.pass)) {
      const total = n * net.fibers.pass[d] * dimOf(d);
      emit('add', u.slot([total, pooled[d], x0[d]]), Math.ceil(total / 64));
    }
    const result = runSeq(net.struct, pooled, h, free.find((s) => s !== pooled && s !== h) ?? x0);
    endStage();
    return { size, stages, uniforms: new Uint8Array(u.buf, 0, u.count * SLOT_BYTES), input: slots[0], result };
  }

  async function predict(prep) {
    assertAlive();
    const { n, e } = prep;
    const p = plan(prep);
    const maxBinding = limits.maxStorageBufferBindingSize;
    const arenaBytes = p.size * 4;
    const basisBytes = (prep.basis.data.length + e * 4) * 4;
    if (arenaBytes > maxBinding || basisBytes > maxBinding) {
      throw new Error(
        `cg2all: ${n} residues / ${e} edges need ${(Math.max(arenaBytes, basisBytes) / 2 ** 20).toFixed(0)} MiB buffers, above this GPU's ${(maxBinding / 2 ** 20).toFixed(0)} MiB binding limit`,
      );
    }
    if (!arena || arena.bytes < arenaBytes) {
      release(arena?.buffer);
      arena = { buffer: await createBuffer({ label: 'cg2all-arena', size: arenaBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }), bytes: arenaBytes };
    }
    const graph = new Uint32Array(Math.max(4, e + n + 1 + e));
    graph.set(prep.src, 0);
    graph.set(prep.csr.offsets, e);
    graph.set(prep.csr.edges, e + n + 1);
    const bData = new Float32Array(Math.max(4, prep.basis.data.length + e * 4));
    bData.set(prep.basis.data, 0);
    bData.set(prep.edgeFeat, prep.basis.data.length);
    const fiber = inputFiber(net, prep);
    const transient = [];
    try {
      const graphBuf = await createBuffer({ label: 'cg2all-graph', size: graph.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const basisBuf = await createBuffer({ label: 'cg2all-basis', size: bData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const uniformBuf = await createBuffer({ label: 'cg2all-uniforms', size: Math.max(SLOT_BYTES, p.uniforms.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const outFloats = n * (20 + 9);
      const staging = await createBuffer({ label: 'cg2all-readback', size: outFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      transient.push(graphBuf, basisBuf, uniformBuf, staging);
      device.queue.writeBuffer(graphBuf, 0, graph);
      device.queue.writeBuffer(basisBuf, 0, bData);
      device.queue.writeBuffer(uniformBuf, 0, p.uniforms);
      device.queue.writeBuffer(arena.buffer, p.input[0] * 4, Float32Array.from(fiber[0].data));
      device.queue.writeBuffer(arena.buffer, p.input[1] * 4, Float32Array.from(fiber[1].data));
      const bindGroup = device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf, offset: 0, size: SLOT_BYTES } },
          { binding: 1, resource: { buffer: weightBuffer } },
          { binding: 2, resource: { buffer: arena.buffer, size: arenaBytes } },
          { binding: 3, resource: { buffer: graphBuf } },
          { binding: 4, resource: { buffer: basisBuf } },
          { binding: 5, resource: { buffer: weightBuffer } },
        ],
      });
      const pacer = new QueuePacer(device, 2);
      for (const [si, stage] of p.stages.entries()) {
        assertAlive();
        const enc = device.createCommandEncoder({ label: `cg2all-stage-${si}` });
        const pass = enc.beginComputePass();
        for (const d of stage) {
          pass.setPipeline(pipelines[d.kernel]);
          pass.setBindGroup(0, bindGroup, [d.slot * SLOT_BYTES]);
          pass.dispatchWorkgroups(d.wg[0], d.wg[1]);
        }
        pass.end();
        if (si === p.stages.length - 1) {
          enc.copyBufferToBuffer(arena.buffer, p.result[0] * 4, staging, 0, n * 20 * 4);
          enc.copyBufferToBuffer(arena.buffer, p.result[1] * 4, staging, n * 20 * 4, n * 9 * 4);
        }
        await pacer.submit(enc.finish());
      }
      await pacer.drain();
      assertAlive();
      try {
        await staging.mapAsync(GPUMapMode.READ);
      } catch (err) {
        assertAlive();
        throw err;
      }
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return { s: Float64Array.from(out.subarray(0, n * 20)), v: Float64Array.from(out.subarray(n * 20)) };
    } finally {
      for (const b of transient) release(b);
    }
  }

  return {
    backend: 'webgpu',
    device,
    weightBytes: packed.data.byteLength,
    predict,
    get lost() {
      return lost;
    },
    destroy() {
      destroyed = true;
      for (const b of [...owned]) release(b);
      arena = null;
    },
  };
}
