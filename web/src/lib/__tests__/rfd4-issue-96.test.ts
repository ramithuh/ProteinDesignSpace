import { describe, expect, it, vi } from 'vitest';
import {
  releaseRfd4GeneratorActivations,
  type Rfd4GeneratorLike,
} from '$lib/plugins/rfd4-proteina/lifecycle';

describe('RFD4-Proteina activation release lifecycle (issue #96)', () => {
  it('invalidates trunk.n, decoder.na, and encoder.na after releasing GPU activation buffers so repeated same-length runs re-record instead of writing to destroyed rfd4p.seqF', () => {
    const trunkRelease = vi.fn();
    const decoderRelease = vi.fn();
    const encoderRelease = vi.fn();

    const gen: Rfd4GeneratorLike = {
      trunk: {
        n: 186,
        condUniform: true,
        lastPairSrc: new Float32Array(4),
        buf: { seqF: { label: 'rfd4p.seqF', destroyed: false } },
        engine: { releaseActivations: trunkRelease },
      },
      decoder: {
        na: 186,
        fdim: 64,
        buf: { seqF: { label: 'rfd4p.dec.seqF', destroyed: false } },
        engine: { releaseActivations: decoderRelease },
      },
      encoder: {
        na: 186,
        fdim: 64,
        buf: { seqF: { label: 'rfd4p.enc.seqF', destroyed: false } },
        engine: { releaseActivations: encoderRelease },
      },
    };

    releaseRfd4GeneratorActivations(gen);

    expect(trunkRelease).toHaveBeenCalledTimes(1);
    expect(decoderRelease).toHaveBeenCalledTimes(1);
    expect(encoderRelease).toHaveBeenCalledTimes(1);

    // Every cached recording guard must be invalidated so the next run with n=186 re-records fresh buffers
    expect(gen.trunk?.n).toBe(0);
    expect(gen.trunk?.condUniform).toBeNull();
    expect(gen.trunk?.lastPairSrc).toBeNull();
    expect(gen.trunk?.buf).toBeNull();

    expect(gen.decoder?.na).toBe(0);
    expect(gen.decoder?.fdim).toBe(0);
    expect(gen.decoder?.buf).toBeNull();

    expect(gen.encoder?.na).toBe(0);
    expect(gen.encoder?.fdim).toBe(0);
    expect(gen.encoder?.buf).toBeNull();
  });

  it('handles null/partial generators and throwing release hooks without skipping subsequent cache invalidations', () => {
    expect(() => releaseRfd4GeneratorActivations(null)).not.toThrow();
    expect(() => releaseRfd4GeneratorActivations(undefined)).not.toThrow();

    const decoderRelease = vi.fn();
    const gen: Rfd4GeneratorLike = {
      trunk: {
        n: 100,
        engine: {
          releaseActivations: () => {
            throw new Error('device lost');
          },
        },
      },
      decoder: {
        na: 100,
        fdim: 32,
        buf: { seqF: {} },
        engine: { releaseActivations: decoderRelease },
      },
      encoder: null,
    };

    expect(() => releaseRfd4GeneratorActivations(gen)).not.toThrow();
    expect(decoderRelease).toHaveBeenCalledTimes(1);
    expect(gen.decoder?.na).toBe(0);
    expect(gen.decoder?.fdim).toBe(0);
    expect(gen.decoder?.buf).toBeNull();
  });
});
