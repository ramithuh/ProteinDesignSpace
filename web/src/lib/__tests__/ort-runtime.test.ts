import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const WEB_ROOT = resolve(import.meta.dirname, '../../..');
const ORT_VENDOR_DIR = resolve(WEB_ROOT, 'static/vendor/ort');
const CF_MAX_FILE_BYTES = 25 * 1024 * 1024;

describe('ONNX Runtime Web first-party asset integrity (issue #59)', () => {
  it('routes onnxruntime-web/webgpu to JSPI C++ WebGPU EP and onnxruntime-web to lean WASM in vite.config.ts', () => {
    const viteCfg = readFileSync(resolve(WEB_ROOT, 'vite.config.ts'), 'utf8');
    expect(viteCfg).toContain('ort.jspi.bundle.min.mjs');
    expect(viteCfg).toContain('ort.wasm.bundle.min.mjs');
  });

  it('ships all required JSPI WebGPU & WASM runtime assets under Cloudflare Pages 25 MiB limit', () => {
    const required = [
      'ort-wasm-simd-threaded.jspi.mjs',
      'ort-wasm-simd-threaded.jspi.wasm',
      'ort-wasm-simd-threaded.mjs',
      'ort-wasm-simd-threaded.wasm',
    ];
    for (const file of required) {
      const fullPath = resolve(ORT_VENDOR_DIR, file);
      expect(existsSync(fullPath), `missing ${file} in web/static/vendor/ort/`).toBe(true);
      const size = statSync(fullPath).size;
      expect(size).toBeGreaterThan(0);
      expect(size, `${file} exceeds 25 MiB Cloudflare Pages limit`).toBeLessThan(CF_MAX_FILE_BYTES);
    }
    expect(existsSync(resolve(ORT_VENDOR_DIR, 'ort-wasm-simd-threaded.asyncify.wasm'))).toBe(false);
    expect(existsSync(resolve(ORT_VENDOR_DIR, 'ort-wasm-simd-threaded.jsep.wasm'))).toBe(false);
  });

  it('pins onnxruntime-web 1.31.1 and dedupes proteinmpnn-web onto the same ORT copy', () => {
    const pkg = JSON.parse(readFileSync(resolve(WEB_ROOT, 'package.json'), 'utf8'));
    expect(pkg.dependencies['onnxruntime-web']).toBe('1.31.1');
    expect(pkg.overrides?.['proteinmpnn-web']?.['onnxruntime-web']).toBe('$onnxruntime-web');
    const lock = JSON.parse(readFileSync(resolve(WEB_ROOT, 'package-lock.json'), 'utf8'));
    expect(lock.packages['node_modules/onnxruntime-web'].version).toBe('1.31.1');
    expect(lock.packages['node_modules/onnxruntime-common'].version).toBe('1.31.1');
    expect(lock.packages['node_modules/proteinmpnn-web/node_modules/onnxruntime-web']).toBeUndefined();
    const installed = JSON.parse(
      readFileSync(resolve(WEB_ROOT, 'node_modules/onnxruntime-web/package.json'), 'utf8'),
    );
    expect(installed.version).toBe('1.31.1');
  });

  it('uses ORT bundles whose referenced WASM/MJS filenames are all present in vendor/ort/', () => {
    for (const bundleName of ['ort.jspi.bundle.min.mjs', 'ort.wasm.bundle.min.mjs']) {
      const ortBundlePath = resolve(WEB_ROOT, 'node_modules/onnxruntime-web/dist', bundleName);
      const bundle = readFileSync(ortBundlePath, 'utf8');
      const refs = [...new Set(bundle.match(/ort-wasm-[a-z0-9.-]+\.(?:mjs|wasm)\b/g) || [])];
      expect(refs.length).toBeGreaterThan(0);
      expect(refs).not.toContain('ort-wasm-simd-threaded.asyncify.wasm');
      expect(refs).not.toContain('ort-wasm-simd-threaded.jsep.wasm');
      for (const ref of refs) {
        expect(
          existsSync(resolve(ORT_VENDOR_DIR, ref)),
          `${bundleName} references missing vendor/ort/${ref}`,
        ).toBe(true);
      }
    }
  });
});
