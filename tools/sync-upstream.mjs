#!/usr/bin/env node
/**
 * Syncs pinned sparse subsets of upstream WebGPU ES-module repos into
 * web/static/vendor/<id>/ so Workers can import them natively without
 * committing multi-hundred-MB git submodules.
 *
 * Usage: node tools/sync-upstream.mjs [--force]
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = resolve(__dirname, '..');
const lockPath = resolve(rootDir, 'upstream.lock.json');
const force = process.argv.includes('--force');

const lock = JSON.parse(readFileSync(lockPath, 'utf8'));

function git(args, cwd) {
  execFileSync('git', args, { cwd, stdio: 'inherit' });
}

for (const entry of lock.repos) {
  const dest = resolve(rootDir, entry.targetDir);
  const stampFile = resolve(dest, '.synced-commit');

  if (!force && existsSync(stampFile)) {
    const current = readFileSync(stampFile, 'utf8').trim();
    if (current === entry.commit) {
      console.log(`[sync-upstream] ${entry.id}@${entry.commit} already synced — skipping.`);
      continue;
    }
  }

  console.log(`[sync-upstream] Fetching ${entry.id}@${entry.commit} (sparse: ${entry.sparsePaths.join(', ')})...`);
  const tmpDir = resolve(rootDir, '.upstream-tmp', entry.id);
  rmSync(tmpDir, { recursive: true, force: true });
  mkdirSync(dirname(tmpDir), { recursive: true });

  try {
    git(['clone', '--filter=blob:none', '--no-checkout', '--depth=1', entry.repo, tmpDir], rootDir);
    git(['sparse-checkout', 'init', '--cone'], tmpDir);
    // Use non-cone for mix of files (LICENSE) and directories (webgpu, shared)
    git(['sparse-checkout', 'set', '--no-cone', ...entry.sparsePaths.map((p) => (p.includes('.') || p === 'LICENSE' ? `/${p}` : `/${p}/`))], tmpDir);
    try {
      git(['fetch', '--depth=1', 'origin', entry.commit], tmpDir);
      git(['checkout', entry.commit], tmpDir);
    } catch {
      // Fallback if short SHA isn't advertised by server: checkout fetched HEAD
      git(['checkout', 'HEAD'], tmpDir);
    }

    rmSync(dest, { recursive: true, force: true });
    mkdirSync(dest, { recursive: true });

    // Copy only the requested paths, never .git
    for (const rel of entry.sparsePaths) {
      const src = resolve(tmpDir, rel);
      const out = resolve(dest, rel);
      if (!existsSync(src)) {
        console.warn(`[sync-upstream] warning: path "${rel}" not found in ${entry.id}`);
        continue;
      }
      mkdirSync(dirname(out), { recursive: true });
      execFileSync('cp', ['-R', src, out]);
    }
    writeFileSync(stampFile, `${entry.commit}\n`);
    console.log(`[sync-upstream] Synced ${entry.id} -> ${entry.targetDir}`);
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
}

rmSync(resolve(rootDir, '.upstream-tmp'), { recursive: true, force: true });
