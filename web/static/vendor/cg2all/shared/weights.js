// cg2all weight bundles (tools/cg2all/export_weights.py): manifest.json + weights.bin, fp16 or int8 (+fp16 group scales);
// fp32 reference bundles are accepted too.
// Tensors keep their PyTorch state_dict names; everything is dequantized to Float32 on load.

export const BUNDLE_FORMAT = 'cg2all-webgpu/v1';

const HALF = (() => {
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    table[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return table;
})();

function view16(bytes, offset, count) {
  if ((bytes.byteOffset + offset) % 2 === 0) return new Uint16Array(bytes.buffer, bytes.byteOffset + offset, count);
  return new Uint16Array(bytes.slice(offset, offset + count * 2).buffer);
}

/** Decode every tensor of a bundle into `Map<name, { shape, data: Float32Array }>`. */
export function decodeTensors(manifest, bytes) {
  const tensors = new Map();
  for (const t of manifest.tensors) {
    const count = t.shape.reduce((a, b) => a * b, 1);
    if (t.offset + t.bytes > bytes.length) throw new Error(`cg2all weights: tensor ${t.name} truncated`);
    const data = new Float32Array(count);
    if (t.dtype === 'float32') {
      data.set(new Float32Array(bytes.slice(t.offset, t.offset + count * 4).buffer));
    } else if (t.dtype === 'float16') {
      const h = view16(bytes, t.offset, count);
      for (let i = 0; i < count; i++) data[i] = HALF[h[i]];
    } else if (t.dtype === 'int8') {
      const q = new Int8Array(bytes.buffer, bytes.byteOffset + t.offset, count);
      const g = t.scale.groupSize;
      const scales = view16(bytes, t.scale.offset, count / g);
      for (let i = 0; i < count; i++) data[i] = q[i] * HALF[scales[(i / g) | 0]];
    } else {
      throw new Error(`cg2all weights: unsupported dtype ${t.dtype} for ${t.name}`);
    }
    tensors.set(t.name, { shape: t.shape, data });
  }
  return tensors;
}

async function sha256Hex(bytes) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const digest = await subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Load a bundle through `read(name) -> Uint8Array` (fetch/OPFS/fs supplied by the caller).
 * Verifies byte length and SHA-256 of weights.bin against the manifest.
 */
export async function loadWeights(baseUrl, { read, onProgress, verify = true } = {}) {
  if (typeof read !== 'function') throw new Error('cg2all loadWeights: a read(name) function is required');
  const manifest = JSON.parse(new TextDecoder().decode(await read('manifest.json')));
  if (manifest.format !== BUNDLE_FORMAT) {
    throw new Error(`cg2all manifest mismatch: format ${manifest.format} (expected ${BUNDLE_FORMAT}) at ${baseUrl}`);
  }
  const file = manifest.files?.[0];
  if (!file) throw new Error('cg2all manifest mismatch: no weight file listed');
  onProgress?.({ loaded: 0, total: file.bytes });
  const bytes = await read(file.name);
  if (bytes.length !== file.bytes) {
    throw new Error(`cg2all weights truncated: ${file.name} has ${bytes.length} bytes, manifest says ${file.bytes}`);
  }
  if (verify) {
    const hex = await sha256Hex(bytes);
    if (hex !== null && hex !== file.sha256) throw new Error(`cg2all weights checksum mismatch for ${file.name}`);
  }
  onProgress?.({ loaded: file.bytes, total: file.bytes });
  return { manifest, tensors: decodeTensors(manifest, bytes), bytes: file.bytes };
}
