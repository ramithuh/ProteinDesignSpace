import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const EXPECTED_VENDOR_CACHE_NAMES = [
  'ligandmpnn-v1',
  'esmc-v1',
  'lasermpnn-v1',
  'thermompnn-v1',
  'protpardelle-v1',
  'rfdiffusion-v1',
  'boltzgen-v1',
  'evodiff-v1',
  'la-proteina-v1',
  'proteina-v1',
  'genie2-v1',
  'frameflow-v1',
  'foldflow2-v1',
  'sid-protein-v1',
  'dyna1-v1',
  'gvp-transformer-v1',
  'gnina-v1',
] as const;

const EXPECTED_VENDOR_CACHE_PREFIXES = [
  'denovo-run',
  'localfold-',
  'rfd4p-',
  'genie3-weights-',
] as const;

const EXPECTED_OPFS_WEIGHT_DIRS = [
  'denovo-weights',
  'esmc-v1',
  'thermompnn-v1',
  'protpardelle-v1',
  'rfdiffusion-v1',
  'boltzgen-v1',
  'evodiff-v1',
  'la-proteina-v1',
  'proteina-v1',
  'genie2-v1',
  'frameflow-v1',
  'foldflow2-v1',
  'sid-protein-v1',
  'dyna1-v1',
  'gvp-transformer-v1',
  'gnina-v1',
] as const;

describe('Phase 0 contract: golden CacheStorage & OPFS weight namespaces', () => {
  it('locks all 17 VENDOR_CACHE_NAMES, 4 VENDOR_CACHE_PREFIXES, and 16 OPFS_WEIGHT_DIRS so users never re-download weights', () => {
    const src = readFileSync(resolve(process.cwd(), 'src/lib/engine/opfs-cache.ts'), 'utf8');

    for (const name of EXPECTED_VENDOR_CACHE_NAMES) {
      expect(src, `Missing CacheStorage namespace "${name}"`).toContain(`'${name}'`);
    }
    for (const prefix of EXPECTED_VENDOR_CACHE_PREFIXES) {
      expect(src, `Missing CacheStorage prefix "${prefix}"`).toContain(`'${prefix}'`);
    }
    for (const dir of EXPECTED_OPFS_WEIGHT_DIRS) {
      expect(src, `Missing OPFS weight dir "${dir}"`).toContain(`'${dir}'`);
    }

    expect(EXPECTED_VENDOR_CACHE_NAMES).toHaveLength(17);
    expect(EXPECTED_VENDOR_CACHE_PREFIXES).toHaveLength(4);
    expect(EXPECTED_OPFS_WEIGHT_DIRS).toHaveLength(16);
  });
});
