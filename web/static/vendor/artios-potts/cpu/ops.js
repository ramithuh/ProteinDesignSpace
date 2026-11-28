/**
 * Float32 dense primitives for the CPU ArtiosPotts reference.
 *
 * Everything works on row-major `Float32Array`s with explicit row counts, so the
 * same code serves `[nToken, d]` node tensors and `[nToken * k, d]` edge tensors.
 * Accumulators are float64 (JS numbers), which is what PyTorch's CPU float32
 * GEMM effectively does too, so the two agree to float32 rounding.
 */

/**
 * `y = x @ W^T + b`, with `W` stored `[outDim, inDim]` as PyTorch does.
 *
 * @param {Float32Array} x `[rows, inDim]`
 * @param {number} rows
 * @param {number} inDim
 * @param {Float32Array} weight `[outDim, inDim]`
 * @param {number} outDim
 * @param {Float32Array|null} bias `[outDim]` or null
 * @param {Float32Array} [out] `[rows, outDim]`
 */
export function linear(x, rows, inDim, weight, outDim, bias = null, out = null) {
  const y = out ?? new Float32Array(rows * outDim);
  for (let r = 0; r < rows; r += 1) {
    const xBase = r * inDim;
    const yBase = r * outDim;
    for (let o = 0; o < outDim; o += 1) {
      const wBase = o * inDim;
      let acc = bias ? bias[o] : 0;
      for (let c = 0; c < inDim; c += 1) acc += x[xBase + c] * weight[wBase + c];
      y[yBase + o] = acc;
    }
  }
  return y;
}

/**
 * LayerNorm over the last dimension, matching
 * `artios.model.modules.primitives.LayerNorm` (biased variance, learned weight,
 * bias is a zero buffer unless the checkpoint says otherwise).
 */
export function layerNorm(x, rows, dim, weight, bias, eps = 1e-5, out = null) {
  const y = out ?? new Float32Array(rows * dim);
  for (let r = 0; r < rows; r += 1) {
    const base = r * dim;
    let mean = 0;
    for (let c = 0; c < dim; c += 1) mean += x[base + c];
    mean /= dim;
    let variance = 0;
    for (let c = 0; c < dim; c += 1) {
      const d = x[base + c] - mean;
      variance += d * d;
    }
    variance /= dim;
    const inverseStd = 1 / Math.sqrt(variance + eps);
    for (let c = 0; c < dim; c += 1) {
      y[base + c] = (x[base + c] - mean) * inverseStd * weight[c] + (bias ? bias[c] : 0);
    }
  }
  return y;
}

/** `x * sigmoid(x)`, PyTorch's `F.silu`. */
export function silu(x) {
  return x / (1 + Math.exp(-x));
}

/**
 * `SwiGLU_MLP`: `W_out(a * silu(b))` where `[a, b] = W_in(x).chunk(2, -1)`.
 *
 * `W_in` is `[2 * hidden * n, inDim]` and `W_out` is `[hidden, hidden * n]`.
 */
export function swiGluMlp(x, rows, inDim, wIn, wOut, hidden, out = null) {
  const inner = wIn.length / inDim; // 2 * hidden * n
  const half = inner / 2;
  const projected = linear(x, rows, inDim, wIn, inner, null);
  const gated = new Float32Array(rows * half);
  for (let r = 0; r < rows; r += 1) {
    const src = r * inner;
    const dst = r * half;
    for (let c = 0; c < half; c += 1) {
      gated[dst + c] = projected[src + c] * silu(projected[src + half + c]);
    }
  }
  return linear(gated, rows, half, wOut, hidden, null, out);
}

/**
 * `gather_nodes`: `out[i, s, :] = nodes[edgeIndex[i, s], :]`.
 */
export function gatherNodes(nodes, dim, edgeIndex, nToken, k, out = null) {
  const y = out ?? new Float32Array(nToken * k * dim);
  for (let i = 0; i < nToken; i += 1) {
    for (let s = 0; s < k; s += 1) {
      const src = edgeIndex[i * k + s] * dim;
      const dst = (i * k + s) * dim;
      for (let c = 0; c < dim; c += 1) y[dst + c] = nodes[src + c];
    }
  }
  return y;
}

/** Concatenate several `[rows, d_i]` blocks along the last dimension. */
export function concatLast(blocks, rows) {
  const total = blocks.reduce((sum, [, dim]) => sum + dim, 0);
  const out = new Float32Array(rows * total);
  let offset = 0;
  for (const [data, dim] of blocks) {
    for (let r = 0; r < rows; r += 1) {
      const src = r * dim;
      const dst = r * total + offset;
      for (let c = 0; c < dim; c += 1) out[dst + c] = data[src + c];
    }
    offset += dim;
  }
  return out;
}

/** Multiply every row of `x` by the matching entry of `mask`, in place. */
export function maskRows(x, rows, dim, mask) {
  for (let r = 0; r < rows; r += 1) {
    if (mask[r]) continue;
    x.fill(0, r * dim, r * dim + dim);
  }
  return x;
}
