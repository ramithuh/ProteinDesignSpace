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
