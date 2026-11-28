/**
 * WebGPU engine: GPU-resident weights, compiled per-shape plans, and the
 * dispatch sequence for the Proton-PottsMPNN encoder, Potts head and decoder.
 *
 * The weights are uploaded once, in the precision they were exported in — FP32
 * as `f32`, FP16 as packed half pairs, INT8 as packed bytes plus a per-row
 * scale — and never re-uploaded. A *plan* is the set of intermediate buffers and
 * bind groups for one `(L, K)` shape; re-running the same structure reuses it,
 * which is what makes the warm timing meaningful.
 */

import {
  ADD_WGSL,
  DEC_MSG_INPUT_WGSL,
  ELEMENTWISE_WORKGROUP,
  ENC_AGGREGATE_WGSL,
  ENC_MSG_INPUT_WGSL,
  EXTRACT_FIELD_WGSL,
  GEMM_FLAG_BIAS,
  GEMM_FLAG_GELU,
  LAYERNORM_WGSL,
  MASK_NODES_WGSL,
  POTTS_MASK_DIAG_WGSL,
  POTTS_MERGE_WGSL,
  TILE,
  gemmSource,
} from './shaders.js';

const STORAGE = GPUBufferUsage.STORAGE;
const COPY_DST = GPUBufferUsage.COPY_DST;
const COPY_SRC = GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM;

/** `ceil(a / b)` for positive integers. */
const divUp = (a, b) => Math.ceil(a / b);

/**
 * Request an adapter and a device with the limits this engine needs.
 * Returns `{ device, adapter, info }` or throws with a diagnosable message.
 */
export async function requestDevice(navigatorGpu = globalThis.navigator?.gpu, options = {}) {
  if (!navigatorGpu) throw new Error('WebGPU is not available (navigator.gpu is undefined)');
  const adapter = await navigatorGpu.requestAdapter({
    powerPreference: options.powerPreference ?? 'high-performance',
  });
  if (!adapter) throw new Error('no WebGPU adapter available');
  // Ask only for the two limits that actually bind us: the Potts coupling
  // tensor is L*K*V*V floats, which is ~40 MB at L=229 — above the 128 MiB
  // default only for very large complexes, but worth raising where we can.
  const want = {
    maxStorageBufferBindingSize: options.maxStorageBufferBindingSize ?? (1 << 28),
    maxBufferSize: options.maxBufferSize ?? (1 << 29),
  };
  const requiredLimits = {};
  for (const [k, v] of Object.entries(want)) {
    const supported = adapter.limits?.[k];
    if (typeof supported === 'number') requiredLimits[k] = Math.min(v, supported);
  }
  const device = await adapter.requestDevice({ requiredLimits });
  const info = {
    vendor: adapter.info?.vendor ?? '',
    architecture: adapter.info?.architecture ?? '',
    device: adapter.info?.device ?? '',
    description: adapter.info?.description ?? '',
    limits: { ...requiredLimits },
  };
  return { device, adapter, info };
}

/** Pack the raw bytes of a typed array into a 4-byte-aligned `Uint32Array`. */
function packToU32(view) {
  const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  const padded = new Uint8Array(Math.ceil(bytes.length / 4) * 4);
  padded.set(bytes);
  return new Uint32Array(padded.buffer);
}

/**
 * Dequantize one stored tensor to FP32 on the host.
 *
 * Exported because the WebGPU backend still runs the featurizer on the CPU: its
 * positional-embedding Linear has to be widened the same way, and reading an
 * INT8 bundle's raw bytes as if they were weights is a silent, large error.
 */
export function tensorToFloat32(bundle, name, halfToFloat) {
  const t = bundle.tensor(name);
  if (t.dtype === 'F32') return t.view;
  if (t.dtype === 'F16') return Float32Array.from(t.view, halfToFloat);
  if (t.dtype === 'I8') {
    const scale = bundle.tensor(`${name}.scale`).view;
    const [rows, cols] = t.shape;
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) out[r * cols + c] = t.view[r * cols + c] * scale[r];
    }
    return out;
  }
  throw new Error(`cannot widen dtype ${t.dtype}`);
}

/**
 * Upload a bundle's tensors to the GPU and build the kernel pipelines.
 *
 * `bundle` is the parsed safetensors object from `shared/parseSafetensors`.
 * Tensors stay in their stored precision — that is the point of the FP16/INT8
 * bundles: the device holds 3.6 MB or 1.9 MB instead of 7.1 MB, and the
 * unpacking happens inside the GEMM.
 */
export function createEngine(device, { bundle, arch, precision, halfToFloat }) {
  const weightDtype = { fp32: 'f32', fp16: 'f16', int8: 'i8' }[precision];
  if (!weightDtype) throw new Error(`unsupported precision "${precision}"`);
  const needsScale = weightDtype === 'i8';

  const owned = [];
  const dummy = device.createBuffer({ size: 4, usage: STORAGE });
  owned.push(dummy);
  let destroyed = false;

  const upload = (data, usage = STORAGE | COPY_DST) => {
    const bytes = data.byteLength;
    const buf = device.createBuffer({ size: Math.max(4, Math.ceil(bytes / 4) * 4), usage });
    device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, bytes);
    owned.push(buf);
    return buf;
  };

  // --- weights -----------------------------------------------------------
  let weightBytes = 0;
  const weightMeta = new Map();
  for (const name of bundle.names) {
    if (name.endsWith('.scale')) continue;
    const t = bundle.tensor(name);
    let data;
    if (t.dtype === 'F32') data = t.view;
    else if (t.dtype === 'F16' || t.dtype === 'I8') data = packToU32(t.view);
    else throw new Error(`unsupported stored dtype ${t.dtype}`);
    const buffer = upload(data);
    weightBytes += buffer.size;
    let scale = dummy;
    if (t.dtype === 'I8') {
      scale = upload(bundle.tensor(`${name}.scale`).view);
      weightBytes += scale.size;
    }
    weightMeta.set(name, { buffer, scale, shape: t.shape, dtype: t.dtype });
  }
  // The sequence embedding is a lookup, not a GEMM: the host keeps an FP32 copy
  // and uploads only the rows the decoder gathers.
  const WsHost = tensorToFloat32(bundle, 'W_s.weight', halfToFloat);

  // --- pipelines ---------------------------------------------------------
  const pipeline = (code, label) => device.createComputePipeline({
    label,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code, label }), entryPoint: 'main' },
  });
  const pipelines = {
    gemm: pipeline(gemmSource(weightDtype), `gemm-${weightDtype}`),
    layernorm: pipeline(LAYERNORM_WGSL, 'layernorm'),
    encMsgInput: pipeline(ENC_MSG_INPUT_WGSL, 'enc-msg-input'),
    encAggregate: pipeline(ENC_AGGREGATE_WGSL, 'enc-aggregate'),
    add: pipeline(ADD_WGSL, 'add'),
    maskNodes: pipeline(MASK_NODES_WGSL, 'mask-nodes'),
    decMsgInput: pipeline(DEC_MSG_INPUT_WGSL, 'dec-msg-input'),
    pottsMaskDiag: pipeline(POTTS_MASK_DIAG_WGSL, 'potts-mask-diag'),
    pottsMerge: pipeline(POTTS_MERGE_WGSL, 'potts-merge'),
    extractField: pipeline(EXTRACT_FIELD_WGSL, 'extract-field'),
  };

  // --- 16-byte uniform blocks, cached by content -------------------------
  const uniforms = new Map();
  const uniformBuffer = (key, fill, bytes = 16) => {
    const hit = uniforms.get(key);
    if (hit) return hit;
    const data = new ArrayBuffer(bytes);
    fill(new DataView(data));
    const buf = device.createBuffer({ size: bytes, usage: UNIFORM | COPY_DST });
    device.queue.writeBuffer(buf, 0, data);
    uniforms.set(key, buf);
    return buf;
  };
  const gemmUniform = (m, kin, n, flags) => uniformBuffer(
    `gemm:${m}:${kin}:${n}:${flags}`,
    (dv) => {
      dv.setUint32(0, m, true); dv.setUint32(4, kin, true);
      dv.setUint32(8, n, true); dv.setUint32(12, flags, true);
    },
  );
  /**
   * The elementwise kernels' shared 32-byte `EDims` block:
   * `(a, b, c, strideX, scale)`. `strideX` is how a kernel folds a 2-D dispatch
   * back into one linear thread index.
   */
  const eDims = (a, b, c, strideX, scale) => uniformBuffer(
    `e:${a}:${b}:${c}:${strideX}:${scale}`,
    (dv) => {
      dv.setUint32(0, a, true);
      dv.setUint32(4, b, true);
      dv.setUint32(8, c, true);
      dv.setUint32(12, strideX, true);
      dv.setFloat32(16, scale, true);
    },
    32,
  );

  const bind = (pipe, resources) => device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

  /** `out = linear(A, W[name])`, optionally with bias and/or a fused GELU. */
  const gemm = (pass, { A, name, M, Kin, N, out, bias = true, gelu = false }) => {
    const w = weightMeta.get(`${name}.weight`);
    if (!w) throw new Error(`no weight "${name}.weight" on the device`);
    if (w.shape[0] !== N || w.shape[1] !== Kin) {
      throw new Error(`${name}.weight is [${w.shape}] but the plan wants [${N}, ${Kin}]`);
    }
    const biasBuf = bias ? weightMeta.get(`${name}.bias`)?.buffer : undefined;
    if (bias && !biasBuf) throw new Error(`no bias for "${name}"`);
    const flags = (bias ? GEMM_FLAG_BIAS : 0) | (gelu ? GEMM_FLAG_GELU : 0);
    // `layout: 'auto'` derives the layout from the bindings the shader actually
    // uses, so the per-row scale is only bound for the INT8 pipeline.
    const resources = [gemmUniform(M, Kin, N, flags), A, w.buffer, biasBuf ?? dummy, out];
    if (needsScale) resources.push(w.scale);
    pass.setPipeline(pipelines.gemm);
    pass.setBindGroup(0, bind(pipelines.gemm, resources));
    pass.dispatchWorkgroups(divUp(N, TILE), divUp(M, TILE), 1);
  };

  const layernorm = (pass, { X, rows, dim, name }) => {
    pass.setPipeline(pipelines.layernorm);
    pass.setBindGroup(0, bind(pipelines.layernorm, [
      uniformBuffer(`ln:${rows}:${dim}`, (dv) => {
        dv.setUint32(0, rows, true); dv.setUint32(4, dim, true);
        dv.setFloat32(8, 1e-5, true); dv.setUint32(12, 0, true);
      }),
      X, weightMeta.get(`${name}.weight`).buffer, weightMeta.get(`${name}.bias`).buffer,
    ]));
    pass.dispatchWorkgroups(rows, 1, 1);
  };

  /**
   * Dispatch an elementwise kernel over `threads` linear elements.
   *
   * The grid is split across X and Y because the Potts tensor runs to millions
   * of elements and `maxComputeWorkgroupsPerDimension` is 65,535; the kernel
   * recombines the two with the `strideX` it is handed.
   */
  const MAX_WG_PER_DIM = 65535;
  const elementwise = (pass, pipe, { a, b = 0, c = 0, scale = 0 }, resources, threads) => {
    const wgTotal = divUp(threads, ELEMENTWISE_WORKGROUP);
    const wgX = Math.min(wgTotal, MAX_WG_PER_DIM);
    const wgY = divUp(wgTotal, wgX);
    const strideX = wgX * ELEMENTWISE_WORKGROUP;
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind(pipe, [eDims(a, b, c, strideX, scale), ...resources]));
    pass.dispatchWorkgroups(wgX, wgY, 1);
  };

  /* -------------------------------------------------------------------- *
   * Plans
   * -------------------------------------------------------------------- */

  let planBuffers = null;
  let planShape = null;
  let planOwned = [];

  const allocate = ({ L, K }) => {
    const H = arch.hiddenDim;
    const V = arch.pottsVocabSize;
    const Vv = arch.vocabSize;
    const FF = arch.ffnDim;
    const rowsE = L * K;
    const VV = V * V;
    const mk = (floats, extra = 0) => {
      const buf = device.createBuffer({
        size: Math.max(4, floats * 4),
        usage: STORAGE | COPY_SRC | COPY_DST | extra,
      });
      planOwned.push(buf);
      return buf;
    };
    return {
      L, K, H, V, Vv, FF, rowsE, VV,
      Eraw: mk(rowsE * arch.numEdgeInputFeatures),
      edgeIdx: mk(rowsE),
      maskE: mk(rowsE),
      residueMask: mk(L),
      reverseK: mk(rowsE),
      mergeValid: mk(rowsE),
      cmask: mk(rowsE * 2),
      hS: mk(L * H),
      E: mk(rowsE * H),
      hE: mk(rowsE * H),
      // Two node buffers ping-pong through the layers; `hV0` is always the
      // encoder's input (zeros) so a second run starts from the same state.
      hV0: mk(L * H),
      hV1: mk(L * H),
      hVdec0: mk(L * H),
      hVdec1: mk(L * H),
      msgIn: mk(rowsE * 4 * H),
      msgA: mk(rowsE * H),
      msgB: mk(rowsE * H),
      ffn: mk(L * FF),
      ffnOut: mk(L * H),
      J: mk(rowsE * VV),
      Jmerged: mk(rowsE * VV),
      h: mk(L * V),
      logits: mk(L * Vv),
    };
  };

  /**
   * Compile (or reuse) the buffer set for one `(L, K)` shape.
   *
   * A new shape reallocates, which invalidates any previously returned plan —
   * the caller keeps at most one alive.
   */
  const compile = ({ L, K }) => {
    if (planShape && planShape.L === L && planShape.K === K) return planBuffers;
    // A new shape replaces the old intermediates rather than accumulating them:
    // the Potts buffers alone are ~80 MB at L=229, so leaking a plan per shape
    // would exhaust the device in a session that varies its input.
    for (const buf of planOwned) buf.destroy();
    planOwned = [];
    planBuffers = allocate({ L, K });
    planShape = { L, K };
    const b = planBuffers;
    const { H, V, Vv, FF, rowsE, VV } = b;

    /** One EncLayer: node update, feed-forward, then edge update. */
    const encoderLayer = (pass, layer, hVin, hVout) => {
      const p = `encoder_layers.${layer}`;
      elementwise(pass, pipelines.encMsgInput, { a: L, b: K, c: H },
        [hVin, b.hE, b.edgeIdx, b.msgIn], rowsE * H);
      gemm(pass, { A: b.msgIn, name: `${p}.W1`, M: rowsE, Kin: 3 * H, N: H, out: b.msgA, gelu: true });
      gemm(pass, { A: b.msgA, name: `${p}.W2`, M: rowsE, Kin: H, N: H, out: b.msgB, gelu: true });
      gemm(pass, { A: b.msgB, name: `${p}.W3`, M: rowsE, Kin: H, N: H, out: b.msgA });
      elementwise(pass, pipelines.encAggregate, { a: L, b: K, c: H, scale: arch.messageScale },
        [hVin, b.msgA, b.maskE, hVout], L * H);
      layernorm(pass, { X: hVout, rows: L, dim: H, name: `${p}.norm1` });
      gemm(pass, { A: hVout, name: `${p}.dense.W_in`, M: L, Kin: H, N: FF, out: b.ffn, gelu: true });
      gemm(pass, { A: b.ffn, name: `${p}.dense.W_out`, M: L, Kin: FF, N: H, out: b.ffnOut });
      elementwise(pass, pipelines.add, { a: L * H }, [hVout, b.ffnOut], L * H);
      layernorm(pass, { X: hVout, rows: L, dim: H, name: `${p}.norm2` });
      elementwise(pass, pipelines.maskNodes, { a: L, b: H },
        [hVout, b.residueMask], L * H);
      // Edge update, from the UPDATED node states; h_E is refreshed in place.
      elementwise(pass, pipelines.encMsgInput, { a: L, b: K, c: H },
        [hVout, b.hE, b.edgeIdx, b.msgIn], rowsE * H);
      gemm(pass, { A: b.msgIn, name: `${p}.W11`, M: rowsE, Kin: 3 * H, N: H, out: b.msgA, gelu: true });
      gemm(pass, { A: b.msgA, name: `${p}.W12`, M: rowsE, Kin: H, N: H, out: b.msgB, gelu: true });
      gemm(pass, { A: b.msgB, name: `${p}.W13`, M: rowsE, Kin: H, N: H, out: b.msgA });
      elementwise(pass, pipelines.add, { a: rowsE * H }, [b.hE, b.msgA], rowsE * H);
      layernorm(pass, { X: b.hE, rows: rowsE, dim: H, name: `${p}.norm3` });
    };

    // The encoder's final node buffer is fixed by the (constant) layer count,
    // so the decoder can name it without tracking runtime state.
    const hVFinal = arch.numEncoderLayers % 2 === 0 ? b.hV0 : b.hV1;

    const encodeAndHead = (encoder) => {
      // h_V starts at zero (ProteinMPNN has no node features); clear outside
      // the compute pass, where copy/clear commands are legal.
      encoder.clearBuffer(b.hV0, 0, L * H * 4);
      const pass = encoder.beginComputePass({ label: 'encoder+potts-head' });
      gemm(pass, {
        A: b.Eraw, name: 'graph_featurization_module.edge_embedding',
        M: rowsE, Kin: arch.numEdgeInputFeatures, N: arch.numEdgeFeatures,
        out: b.E, bias: false,
      });
      layernorm(pass, {
        X: b.E, rows: rowsE, dim: arch.numEdgeFeatures,
        name: 'graph_featurization_module.edge_norm',
      });
      gemm(pass, { A: b.E, name: 'W_e', M: rowsE, Kin: arch.numEdgeFeatures, N: H, out: b.hE });
      let cur = b.hV0;
      for (let l = 0; l < arch.numEncoderLayers; l += 1) {
        const next = cur === b.hV0 ? b.hV1 : b.hV0;
        encoderLayer(pass, l, cur, next);
        cur = next;
      }
      if (cur !== hVFinal) throw new Error('encoder node ping-pong ended on the wrong buffer');
      if (arch.etabSource === 'node_edge_node') {
        elementwise(pass, pipelines.encMsgInput, { a: L, b: K, c: H },
          [cur, b.hE, b.edgeIdx, b.msgIn], rowsE * H);
        gemm(pass, { A: b.msgIn, name: 'etab_out', M: rowsE, Kin: 3 * H, N: VV, out: b.J });
      } else {
        gemm(pass, { A: b.hE, name: 'etab_out', M: rowsE, Kin: H, N: VV, out: b.J });
      }
      elementwise(pass, pipelines.pottsMaskDiag, { a: L, b: K, c: V },
        [b.J, b.maskE], rowsE * VV);
      elementwise(pass, pipelines.pottsMerge, { a: L, b: K, c: V },
        [b.J, b.Jmerged, b.edgeIdx, b.reverseK, b.mergeValid], rowsE * VV);
      elementwise(pass, pipelines.extractField, { a: L, b: K, c: V },
        [b.Jmerged, b.h], L * V);
      pass.end();
    };

    const decode = (encoder) => {
      encoder.copyBufferToBuffer(hVFinal, 0, b.hVdec0, 0, L * H * 4);
      const pass = encoder.beginComputePass({ label: 'decoder' });
      let cur = b.hVdec0;
      for (let l = 0; l < arch.numDecoderLayers; l += 1) {
        const next = cur === b.hVdec0 ? b.hVdec1 : b.hVdec0;
        const p = `decoder_layers.${l}`;
        elementwise(pass, pipelines.decMsgInput, { a: L, b: K, c: H },
          [cur, b.hE, b.hS, hVFinal, b.edgeIdx, b.cmask, b.msgIn], rowsE * H);
        gemm(pass, { A: b.msgIn, name: `${p}.W1`, M: rowsE, Kin: 4 * H, N: H, out: b.msgA, gelu: true });
        gemm(pass, { A: b.msgA, name: `${p}.W2`, M: rowsE, Kin: H, N: H, out: b.msgB, gelu: true });
        gemm(pass, { A: b.msgB, name: `${p}.W3`, M: rowsE, Kin: H, N: H, out: b.msgA });
        elementwise(pass, pipelines.encAggregate, { a: L, b: K, c: H, scale: arch.messageScale },
          [cur, b.msgA, b.maskE, next], L * H);
        layernorm(pass, { X: next, rows: L, dim: H, name: `${p}.norm1` });
        gemm(pass, { A: next, name: `${p}.dense.W_in`, M: L, Kin: H, N: FF, out: b.ffn, gelu: true });
        gemm(pass, { A: b.ffn, name: `${p}.dense.W_out`, M: L, Kin: FF, N: H, out: b.ffnOut });
        elementwise(pass, pipelines.add, { a: L * H }, [next, b.ffnOut], L * H);
        layernorm(pass, { X: next, rows: L, dim: H, name: `${p}.norm2` });
        elementwise(pass, pipelines.maskNodes, { a: L, b: H },
          [next, b.residueMask], L * H);
        cur = next;
      }
      gemm(pass, { A: cur, name: 'W_out', M: L, Kin: H, N: Vv, out: b.logits });
      pass.end();
    };

    planBuffers.encodeAndHead = encodeAndHead;
    planBuffers.decode = decode;
    planBuffers.hVFinal = hVFinal;
    return planBuffers;
  };

  /** Read a GPU buffer back as a `Float32Array` of `floats` elements. */
  const readFloats = async (buffer, floats) => {
    const size = floats * 4;
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  };

  return {
    device,
    precision,
    weightDtype,
    weightBytes,
    pipelines,
    WsHost,
    compile,
    readFloats,
    write: (buffer, data) => device.queue.writeBuffer(
      buffer, 0, data.buffer, data.byteOffset, data.byteLength,
    ),

    /** Release every GPU buffer this engine owns. Idempotent. */
    destroy() {
      if (destroyed) return;
      destroyed = true;
      for (const buf of owned) buf.destroy();
      owned.length = 0;
      for (const buf of planOwned) buf.destroy();
      planOwned = [];
      for (const buf of uniforms.values()) buf.destroy();
      uniforms.clear();
      planBuffers = null;
      planShape = null;
    },
  };
}
