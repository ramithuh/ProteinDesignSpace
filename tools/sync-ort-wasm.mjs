#!/usr/bin/env node
/**
 * Copy the exact needed ONNX Runtime Web WASM + MJS runtime files from the
 * npm-locked `onnxruntime-web` package into `web/static/vendor/ort/` so
 * `mpnn.worker.ts` loads 100% first-party same-origin binaries with zero
 * third-party CDN execution (and stays under Cloudflare Pages' 25 MiB limit).
 */
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEB_DIR = resolve(__dirname, '..', 'web');
const OUT_DIR = resolve(WEB_DIR, 'static', 'vendor', 'ort');

const CANDIDATE_DISTS = [
  resolve(WEB_DIR, 'node_modules', 'onnxruntime-web', 'dist'),
  resolve(WEB_DIR, 'node_modules', 'proteinmpnn-web', 'node_modules', 'onnxruntime-web', 'dist'),
];

const REQUIRED_FILES = [
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
];

const distDir = CANDIDATE_DISTS.find((d) => existsSync(resolve(d, REQUIRED_FILES[0])));
if (!distDir) {
  console.error('[sync-ort-wasm] Could not find onnxruntime-web/dist in node_modules.');
  process.exit(1);
}

mkdirSync(OUT_DIR, { recursive: true });
for (const file of REQUIRED_FILES) {
  const src = resolve(distDir, file);
  const dst = resolve(OUT_DIR, file);
  const srcStat = statSync(src);
  if (existsSync(dst) && statSync(dst).size === srcStat.size) {
    continue;
  }
  copyFileSync(src, dst);
}
console.log(`[sync-ort-wasm] Synced ${REQUIRED_FILES.length} verified ORT files to web/static/vendor/ort/`);
