export interface Rfd4RecordableTrunk {
  n?: number;
  condUniform?: boolean | null;
  lastPairSrc?: unknown;
  buf?: unknown;
  engine?: { releaseActivations?: () => void } | null;
}

export interface Rfd4RecordableAutoencoder {
  na?: number;
  fdim?: number;
  buf?: unknown;
  engine?: { releaseActivations?: () => void } | null;
}

export interface Rfd4GeneratorLike {
  releaseActivations?: () => void;
  trunk?: Rfd4RecordableTrunk | null;
  decoder?: Rfd4RecordableAutoencoder | null;
  encoder?: Rfd4RecordableAutoencoder | null;
}

/**
 * Release RFD4-Proteina's transient activation GPU buffers AND invalidate the
 * recorded-size caches (`trunk.n`, `decoder.na`, `encoder.na`) on every sub-module.
 *
 * Without resetting those cached sizes, the next `generate()` / `decode()` / `encode()`
 * call with the same token count `n` hits `if (this.n === n && this.condUniform === condUniform) return;`
 * in `trunk._record` (or `if (this.na === na && this.fdim === fdim) return;` in decoder/encoder)
 * and writes into the already-destroyed `rfd4p.seqF` / `rfd4p.dec.seqF` GPUBuffer (issue #96).
 */
export function releaseRfd4GeneratorActivations(
  generator: Rfd4GeneratorLike | null | undefined,
): void {
  if (!generator) return;
  try {
    generator.releaseActivations?.();
  } catch {
    /* ignore */
  }
  try {
    generator.trunk?.engine?.releaseActivations?.();
  } catch {
    /* ignore */
  } finally {
    if (generator.trunk) {
      generator.trunk.n = 0;
      generator.trunk.condUniform = null;
      generator.trunk.lastPairSrc = null;
      generator.trunk.buf = null;
    }
  }
  try {
    generator.decoder?.engine?.releaseActivations?.();
  } catch {
    /* ignore */
  } finally {
    if (generator.decoder) {
      generator.decoder.na = 0;
      generator.decoder.fdim = 0;
      generator.decoder.buf = null;
    }
  }
  try {
    generator.encoder?.engine?.releaseActivations?.();
  } catch {
    /* ignore */
  } finally {
    if (generator.encoder) {
      generator.encoder.na = 0;
      generator.encoder.fdim = 0;
      generator.encoder.buf = null;
    }
  }
}

/** A sharded-bundle tensor record (`shared/rfd4p/safetensors.js` `lazyTensor`). */
export interface Rfd4TensorRecord {
  shape: number[];
  dtype?: string;
  readonly data: Float32Array;
}

export interface Rfd4ShardedBundleLike {
  tensors: Record<string, Rfd4TensorRecord>;
  manifest?: {
    shards?: Array<{ file: string; role?: string }>;
    tensors?: Record<string, { file?: string }>;
  } | null;
}

/**
 * The only tensors the WebGPU generator reads on the CPU after upload: the feature-creator embedding
 * tables used by `trunkSeqFeatures` / `trunkCondFeatures` (`shared/rfd4p/features.js`) and by
 * `decoderSeqFeatures` / `encoderSeqFeatures` (`cpu/rfd4p/{decoder,encoder}.js`, via `rawWeights`).
 */
const RFD4_CPU_READ_TENSOR_RE = /\.feat_creators\./;

const RELEASED = Symbol('rfd4-released-shard-bytes');

/** Thrown if anything reads a tensor whose shard bytes were released after GPU upload. */
export class Rfd4ReleasedWeightError extends Error {
  constructor(name: string) {
    super(`RFD4-Proteina weight "${name}" was released from the JS heap after GPU upload`);
    this.name = 'Rfd4ReleasedWeightError';
  }
}

export interface Rfd4ShardReleaseStats {
  /** Tensors replaced by a released stub (their bytes now live only on the GPU). */
  released: number;
  /** CPU-read embedding tables copied out of their shard so the shard itself can be collected. */
  detached: number;
  /** Encoder tensors kept because the lazily-built encoder has not uploaded them yet. */
  pendingEncoder: number;
}

function isEncoderTensor(bundle: Rfd4ShardedBundleLike, name: string): boolean {
  const file = bundle.manifest?.tensors?.[name]?.file;
  const shard = file ? bundle.manifest?.shards?.find((s) => s.file === file) : undefined;
  return shard ? shard.role === 'encoder' : name.startsWith('autoencoder.encoder.');
}

/**
 * Release the sharded bundle's JS-heap shard bytes (~316 MB core + ~131 MB encoder) once the generator
 * has uploaded them to the GPU. The generator keeps `bundle.tensors` as `this.weights`, and every lazy
 * tensor closes over its whole shard `ArrayBuffer`, so the records are rewritten in place:
 * - CPU-read feature-creator tables become small detached copies (a view would pin its ~31 MB shard);
 * - encoder tensors stay lazy until `encoderUploaded`, because `getEncoder()` builds the encoder on first use;
 * - everything else becomes a stub that throws `Rfd4ReleasedWeightError` if read.
 * Idempotent. A new generator (e.g. after device loss) needs a freshly fetched bundle.
 */
export function releaseRfd4UploadedShardBytes(
  bundle: Rfd4ShardedBundleLike | null | undefined,
  opts: { encoderUploaded: boolean },
): Rfd4ShardReleaseStats {
  const stats: Rfd4ShardReleaseStats = { released: 0, detached: 0, pendingEncoder: 0 };
  const tensors = bundle?.tensors;
  if (!bundle || !tensors) return stats;
  for (const name of Object.keys(tensors)) {
    const rec = tensors[name] as Rfd4TensorRecord & { [RELEASED]?: true };
    if (!rec || rec[RELEASED]) continue;
    if (RFD4_CPU_READ_TENSOR_RE.test(name)) {
      tensors[name] = Object.assign(
        { shape: rec.shape, dtype: rec.dtype ?? 'F32', data: rec.data.slice() },
        { [RELEASED]: true as const },
      );
      stats.detached++;
      continue;
    }
    if (!opts.encoderUploaded && isEncoderTensor(bundle, name)) {
      stats.pendingEncoder++;
      continue;
    }
    tensors[name] = {
      shape: rec.shape,
      dtype: rec.dtype ?? 'F32',
      [RELEASED]: true,
      get data(): Float32Array {
        throw new Rfd4ReleasedWeightError(name);
      },
    } as Rfd4TensorRecord;
    stats.released++;
  }
  return stats;
}
