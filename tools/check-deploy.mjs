#!/usr/bin/env node
/**
 * Cloudflare Pages pre-deploy verification script.
 * Ensures web/build exists, required vendor engines are present, and NO single
 * asset exceeds Cloudflare Pages' 25 MiB per-file limit.
 */

import { existsSync, readdirSync, statSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const buildDir = resolve(__dirname, '../web/build');
const CF_MAX_FILE_BYTES = 25 * 1024 * 1024; // 25 MiB

if (!existsSync(resolve(buildDir, 'index.html'))) {
  console.error('[check:deploy] FAIL: web/build/index.html not found. Run `npm run build` first.');
  process.exit(1);
}

for (const req of ['_headers', 'vendor/localfold/webgpu/index.js', 'vendor/rfd4-proteina/webgpu/index.js']) {
  if (!existsSync(resolve(buildDir, req))) {
    console.error(`[check:deploy] FAIL: required build asset missing: ${req}`);
    process.exit(1);
  }
}

/** @type {Array<{ path: string; bytes: number }>} */
const files = [];
let totalBytes = 0;

function walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = resolve(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full);
    } else if (entry.isFile()) {
      const size = statSync(full).size;
      totalBytes += size;
      files.push({ path: relative(buildDir, full), bytes: size });
    }
  }
}

walk(buildDir);
files.sort((a, b) => b.bytes - a.bytes);

const oversized = files.filter((f) => f.bytes > CF_MAX_FILE_BYTES);
if (oversized.length > 0) {
  console.error('[check:deploy] FAIL: Files exceed Cloudflare Pages 25 MiB limit:');
  for (const f of oversized) {
    console.error(`  - ${f.path}: ${(f.bytes / 1024 / 1024).toFixed(2)} MiB`);
  }
  process.exit(1);
}

console.log(
  `[check:deploy] PASS: ${files.length} files, ${(totalBytes / 1024 / 1024).toFixed(2)} MiB total (max ${(files[0].bytes / 1024 / 1024).toFixed(2)} MiB < 25 MiB)`,
);
console.log('[check:deploy] Largest 5 assets:');
for (const f of files.slice(0, 5)) {
  console.log(`  ${(f.bytes / 1024 / 1024).toFixed(2).padStart(6)} MiB  ${f.path}`);
}
