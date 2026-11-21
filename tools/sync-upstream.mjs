#!/usr/bin/env node
/**
 * Syncs pinned sparse subsets of upstream WebGPU ES-module repos into
 * web/static/vendor/<id>/ so Workers can import them natively without
 * committing multi-hundred-MB git submodules.
 *
 * Security & reproducibility guarantees:
 * - Never falls back to `HEAD` when a pinned commit is unavailable.
 * - Verifies the checked-out full 40-char commit SHA matches the requested pin.
 * - Verifies every required `sparsePaths` entry exists in the checked-out tree.
 * - Preserves relative symlinks verbatim and verifies no staged symlink is
 *   dangling or escapes the staged vendor tree.
 * - Stages into a sibling temp directory and atomically swaps into place ONLY
 *   after all verification passes, preserving any previous verified vendor tree
 *   intact on failure (including rolling back multi-repo publish failures).
 * - Uses per-invocation temp directories so concurrent runs do not clobber
 *   each other's clones.
 * - Records the verified full 40-char commit SHA in `.synced-commit` and
 *   file-level SHA-256 provenance metadata in `.synced-provenance.json` so
 *   cache damage or legacy unverified stamps trigger re-verification.
 *
 * Usage: node tools/sync-upstream.mjs [--force] [--lock <path>] [--root <path>]
 */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  realpathSync,
  renameSync,
  rmdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const HEX_PIN_RE = /^[0-9a-f]{7,40}$/i;
const FULL_SHA_RE = /^[0-9a-f]{40}$/;
const RESERVED_STAMP_NAMES = new Set(['.synced-commit', '.synced-provenance.json']);

function pathExistsOrSymlink(targetPath) {
  try {
    lstatSync(targetPath);
    return true;
  } catch {
    return false;
  }
}

function ensureWithin(baseDir, candidatePath, label) {
  const rel = relative(baseDir, candidatePath);
  if (!rel || rel.startsWith('..') || isAbsolute(rel)) {
    throw new Error(`[sync-upstream] Unsafe ${label} path "${candidatePath}" escapes "${baseDir}"`);
  }
}

function ensureWithinOrEqual(baseDir, candidatePath, label) {
  const rel = relative(baseDir, candidatePath);
  if (rel.startsWith('..') || isAbsolute(rel)) {
    throw new Error(`[sync-upstream] Unsafe ${label} path "${candidatePath}" escapes "${baseDir}"`);
  }
}

function ensurePathDoesNotEscapeRoot(rootDir, candidatePath, label) {
  ensureWithin(rootDir, candidatePath, label);
  const rootReal = realpathSync(rootDir);
  const rel = relative(rootDir, candidatePath);
  const parts = rel.split(sep).filter(Boolean);
  let cur = rootDir;
  for (const part of parts) {
    cur = resolve(cur, part);
    if (!pathExistsOrSymlink(cur)) {
      break;
    }
    const st = lstatSync(cur);
    if (st.isSymbolicLink()) {
      let resolved;
      try {
        resolved = realpathSync(cur);
      } catch {
        throw new Error(`[sync-upstream] Unsafe ${label} traverses dangling symlink "${cur}"`);
      }
      ensureWithinOrEqual(rootReal, resolved, `${label} symlink "${cur}"`);
    }
  }
}

function normalizeSparsePath(rawPath) {
  if (typeof rawPath !== 'string') {
    throw new Error(`[sync-upstream] Invalid sparsePath entry: ${String(rawPath)}`);
  }
  const cleaned = rawPath.replace(/\\/g, '/').replace(/\/+$/, '').trim();
  if (!cleaned || cleaned.startsWith('/')) {
    throw new Error(`[sync-upstream] Invalid sparsePath "${rawPath}" (must be a non-empty relative path)`);
  }
  const segments = cleaned.split('/');
  if (
    segments.some(
      (seg) =>
        seg === '' || seg === '.' || seg === '..' || seg === '.git' || RESERVED_STAMP_NAMES.has(seg),
    )
  ) {
    throw new Error(`[sync-upstream] Unsafe or reserved sparsePath "${rawPath}"`);
  }
  return cleaned;
}

function validateEntry(entry, rootDir) {
  if (!entry || typeof entry !== 'object') {
    throw new Error('[sync-upstream] Lockfile repo entry must be an object');
  }
  const id = String(entry.id ?? '').trim();
  if (!id || !/^[a-zA-Z0-9._-]+$/.test(id)) {
    throw new Error(`[sync-upstream] Invalid repo id "${id}"`);
  }
  const repo = String(entry.repo ?? '').trim();
  if (!repo || repo.startsWith('-')) {
    throw new Error(`[sync-upstream] Invalid repo URL/path "${repo}" for "${id}"`);
  }
  const commit = String(entry.commit ?? '').trim().toLowerCase();
  if (!HEX_PIN_RE.test(commit)) {
    throw new Error(
      `[sync-upstream] Invalid commit pin "${String(entry.commit ?? '')}" for "${id}" (expected 7–40 hex SHA)`,
    );
  }
  const targetDir = String(entry.targetDir ?? '').trim();
  if (!targetDir) {
    throw new Error(`[sync-upstream] Missing targetDir for "${id}"`);
  }
  const dest = resolve(rootDir, targetDir);
  ensurePathDoesNotEscapeRoot(rootDir, dest, 'targetDir');

  if (!Array.isArray(entry.sparsePaths) || entry.sparsePaths.length === 0) {
    throw new Error(`[sync-upstream] Missing sparsePaths for "${id}"`);
  }
  const sparsePaths = entry.sparsePaths.map(normalizeSparsePath);

  return { id, repo, commit, targetDir, dest, sparsePaths };
}

function gitRun(args, cwd, { quiet = false } = {}) {
  execFileSync('git', args, {
    cwd,
    stdio: quiet ? ['ignore', 'pipe', 'pipe'] : 'inherit',
  });
}

function gitOutput(args, cwd) {
  return execFileSync('git', args, {
    cwd,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim();
}

function resolveCommitInClone(cloneDir, requestedPin) {
  try {
    const resolved = gitOutput(['rev-parse', '--verify', '--quiet', `${requestedPin}^{commit}`], cloneDir)
      .trim()
      .toLowerCase();
    if (FULL_SHA_RE.test(resolved) && resolved.startsWith(requestedPin)) {
      return resolved;
    }
    return null;
  } catch {
    return null;
  }
}

function isShallowRepo(cloneDir) {
  try {
    return gitOutput(['rev-parse', '--is-shallow-repository'], cloneDir) === 'true';
  } catch {
    return false;
  }
}

function sha256File(filePath) {
  return createHash('sha256').update(readFileSync(filePath)).digest('hex');
}

/**
 * Walk a staged/published vendor directory, verify every symlink resolves inside
 * the tree (never dangling or escaping), reject reserved stamp names in upstream
 * content, and build a deterministic file manifest.
 */
function buildAndVerifyManifest(rootDir, { allowRootStamps = false } = {}) {
  const rootReal = realpathSync(rootDir);
  const manifest = [];

  function walk(dir) {
    const entries = readdirSync(dir, { withFileTypes: true }).sort((a, b) =>
      a.name < b.name ? -1 : a.name > b.name ? 1 : 0,
    );
    for (const dirent of entries) {
      const fullPath = resolve(dir, dirent.name);
      const relPath = relative(rootDir, fullPath).replace(/\\/g, '/');
      if (RESERVED_STAMP_NAMES.has(dirent.name)) {
        if (allowRootStamps && dir === rootDir && dirent.isFile() && !dirent.isSymbolicLink()) {
          continue;
        }
        throw new Error(
          `[sync-upstream] Upstream/staged tree contains reserved metadata entry "${relPath}"`,
        );
      }
      if (dirent.isSymbolicLink()) {
        const linkTarget = readlinkSync(fullPath);
        if (isAbsolute(linkTarget)) {
          throw new Error(
            `[sync-upstream] Refusing absolute symlink "${relPath}" -> "${linkTarget}" inside vendor tree`,
          );
        }
        if (!existsSync(fullPath)) {
          throw new Error(
            `[sync-upstream] Broken/dangling symlink "${relPath}" -> "${linkTarget}" inside staged vendor tree`,
          );
        }
        const resolvedTarget = realpathSync(fullPath);
        ensureWithinOrEqual(rootReal, resolvedTarget, `symlink target for "${relPath}"`);
        manifest.push({ path: relPath, type: 'symlink', target: linkTarget });
      } else if (dirent.isDirectory()) {
        walk(fullPath);
      } else if (dirent.isFile()) {
        manifest.push({ path: relPath, type: 'file', sha256: sha256File(fullPath) });
      }
    }
  }

  walk(rootDir);
  return manifest;
}

function isAlreadyVerified(entry) {
  const stampFile = resolve(entry.dest, '.synced-commit');
  const provenanceFile = resolve(entry.dest, '.synced-provenance.json');
  if (!pathExistsOrSymlink(stampFile) || !pathExistsOrSymlink(provenanceFile)) return false;
  if (!lstatSync(stampFile).isFile() || !lstatSync(provenanceFile).isFile()) return false;

  const recordedSha = readFileSync(stampFile, 'utf8').trim().toLowerCase();
  if (!FULL_SHA_RE.test(recordedSha) || !recordedSha.startsWith(entry.commit)) {
    return false;
  }

  let prov;
  try {
    prov = JSON.parse(readFileSync(provenanceFile, 'utf8'));
    if (!prov || typeof prov !== 'object') return false;
    if (prov.id !== entry.id || prov.repo !== entry.repo) return false;
    const provResolved = String(prov.resolvedCommit ?? '').trim().toLowerCase();
    if (provResolved !== recordedSha || !provResolved.startsWith(entry.commit)) return false;
    if (!Array.isArray(prov.sparsePaths)) return false;
    const provPaths = prov.sparsePaths.map(normalizeSparsePath);
    if (provPaths.length !== entry.sparsePaths.length) return false;
    for (let i = 0; i < provPaths.length; i++) {
      if (provPaths[i] !== entry.sparsePaths[i]) return false;
    }
    if (!Array.isArray(prov.files) || prov.files.length === 0) return false;
  } catch {
    return false;
  }

  for (const rel of entry.sparsePaths) {
    const target = resolve(entry.dest, rel);
    if (!existsSync(target)) return false;
  }

  // Verify file-level integrity (catches deleted/modified files inside sparse directories or broken symlinks).
  try {
    const actualFiles = buildAndVerifyManifest(entry.dest, { allowRootStamps: true });
    if (actualFiles.length !== prov.files.length) return false;
    for (let i = 0; i < actualFiles.length; i++) {
      const a = actualFiles[i];
      const b = prov.files[i];
      if (!b || a.path !== b.path || a.type !== b.type) return false;
      if (a.type === 'file' && a.sha256 !== b.sha256) return false;
      if (a.type === 'symlink' && a.target !== b.target) return false;
    }
  } catch {
    return false;
  }

  return true;
}

function stageVerifiedEntry(entry, runTmpDir, runTag) {
  const cloneDir = resolve(runTmpDir, entry.id);
  const destParent = dirname(entry.dest);
  mkdirSync(destParent, { recursive: true });
  const stageDir = resolve(destParent, `.${entry.id}.stage-${runTag}`);

  rmSync(cloneDir, { recursive: true, force: true });
  rmSync(stageDir, { recursive: true, force: true });
  mkdirSync(runTmpDir, { recursive: true });

  try {
    // Prefer a blobless clone for speed; if a transport/local repo rejects --filter, fall back to plain --no-checkout.
    try {
      gitRun(['clone', '--filter=blob:none', '--no-checkout', '--depth=1', '--', entry.repo, cloneDir], runTmpDir, {
        quiet: true,
      });
    } catch {
      rmSync(cloneDir, { recursive: true, force: true });
      gitRun(['clone', '--no-checkout', '--depth=1', '--', entry.repo, cloneDir], runTmpDir, {
        quiet: true,
      });
    }

    gitRun(['sparse-checkout', 'init', '--no-cone'], cloneDir, { quiet: true });
    gitRun(
      ['sparse-checkout', 'set', '--no-cone', ...entry.sparsePaths.map((p) => `/${p}`)],
      cloneDir,
      { quiet: true },
    );

    // Resolve the exact pinned commit. NEVER fall back to HEAD!
    let resolvedSha = resolveCommitInClone(cloneDir, entry.commit);

    if (!resolvedSha) {
      // Try fetching the exact pin (works for full SHAs advertised or allowed by uploadpack.allowReachableSHA1InWant).
      try {
        gitRun(['fetch', '--depth=1', 'origin', entry.commit], cloneDir, { quiet: true });
      } catch {
        // Proceed to unshallow/full commit-graph fetch for older or short SHAs.
      }
      resolvedSha = resolveCommitInClone(cloneDir, entry.commit);
    }

    if (!resolvedSha) {
      // Fetch commit history (blobless if clone used --filter=blob:none) so short or historical SHAs can resolve.
      try {
        if (isShallowRepo(cloneDir)) {
          gitRun(['fetch', '--unshallow', 'origin'], cloneDir, { quiet: true });
        } else {
          gitRun(['fetch', 'origin'], cloneDir, { quiet: true });
        }
      } catch {
        // Keep checking in case fetch partially populated refs; final check below enforces strict match.
      }
      resolvedSha = resolveCommitInClone(cloneDir, entry.commit);
    }

    if (!resolvedSha) {
      throw new Error(
        `[sync-upstream] Requested commit "${entry.commit}" is unavailable in ${entry.repo} for "${entry.id}". Refusing to fall back to HEAD.`,
      );
    }

    gitRun(['checkout', '--detach', resolvedSha], cloneDir, { quiet: true });

    const checkedOutSha = gitOutput(['rev-parse', 'HEAD'], cloneDir).trim().toLowerCase();
    if (!FULL_SHA_RE.test(checkedOutSha) || checkedOutSha !== resolvedSha || !checkedOutSha.startsWith(entry.commit)) {
      throw new Error(
        `[sync-upstream] Checked-out commit "${checkedOutSha}" does not match requested pin "${entry.commit}" (resolved "${resolvedSha}") for "${entry.id}".`,
      );
    }

    // Verify every required sparsePath exists before staging anything.
    const missingPaths = [];
    for (const rel of entry.sparsePaths) {
      const src = resolve(cloneDir, rel);
      ensureWithin(cloneDir, src, 'sparsePath');
      if (!existsSync(src)) {
        missingPaths.push(rel);
      }
    }
    if (missingPaths.length > 0) {
      throw new Error(
        `[sync-upstream] Required sparsePaths missing in ${entry.id}@${checkedOutSha}: ${missingPaths.join(', ')}`,
      );
    }

    // Stage into sibling temp directory (same filesystem as dest for atomic swap).
    mkdirSync(stageDir, { recursive: true });
    for (const rel of entry.sparsePaths) {
      const src = resolve(cloneDir, rel);
      const out = resolve(stageDir, rel);
      ensureWithin(stageDir, out, 'staged sparsePath');
      mkdirSync(dirname(out), { recursive: true });
      cpSync(src, out, {
        recursive: true,
        dereference: false,
        verbatimSymlinks: true,
        filter: (srcPath) => {
          const parts = srcPath.split(sep);
          return !parts.includes('.git');
        },
      });
      if (!pathExistsOrSymlink(out)) {
        throw new Error(`[sync-upstream] Failed to stage required path "${rel}" for "${entry.id}"`);
      }
    }

    // After ALL sparsePaths are copied, verify every required entry resolves (including symlinks listed before their targets).
    for (const rel of entry.sparsePaths) {
      const out = resolve(stageDir, rel);
      if (!existsSync(out)) {
        throw new Error(`[sync-upstream] Staged path "${rel}" is broken or missing after copy for "${entry.id}"`);
      }
    }

    // Verify every staged symlink is valid & contained within stageDir, reject any reserved
    // stamp filenames (.synced-commit / .synced-provenance.json) from upstream, and compute
    // the file-level SHA-256 manifest BEFORE writing our own verified stamps.
    const files = buildAndVerifyManifest(stageDir, { allowRootStamps: false });

    // Write verified full revision stamp + provenance metadata inside stageDir before publish.
    writeFileSync(resolve(stageDir, '.synced-commit'), `${checkedOutSha}\n`);
    writeFileSync(
      resolve(stageDir, '.synced-provenance.json'),
      `${JSON.stringify(
        {
          schema: 1,
          id: entry.id,
          repo: entry.repo,
          requestedCommit: entry.commit,
          resolvedCommit: checkedOutSha,
          sparsePaths: entry.sparsePaths,
          files,
          syncedAt: new Date().toISOString(),
        },
        null,
        2,
      )}\n`,
    );

    return { entry, stageDir, checkedOutSha };
  } catch (err) {
    rmSync(stageDir, { recursive: true, force: true });
    throw err;
  } finally {
    rmSync(cloneDir, { recursive: true, force: true });
  }
}

function acquirePublishLock(tmpRoot) {
  mkdirSync(tmpRoot, { recursive: true });
  const lockDir = resolve(tmpRoot, '.publish.lock');
  const maxAttempts = 50;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      mkdirSync(lockDir);
      return lockDir;
    } catch (err) {
      if (err && err.code === 'EEXIST') {
        // Brief spin-sleep via Atomics.wait to serialize concurrent publish swaps
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 50);
        continue;
      }
      throw err;
    }
  }
  throw new Error(`[sync-upstream] Timed out acquiring publish lock "${lockDir}"`);
}

function releasePublishLock(lockDir) {
  try {
    rmdirSync(lockDir);
  } catch {
    // Ignore if already removed
  }
}

function publishStagedEntries(staged, runTag, tmpRoot, onBeforeSwap) {
  const lockDir = acquirePublishLock(tmpRoot);
  const swapped = [];
  try {
    for (const item of staged) {
      const { entry, stageDir } = item;
      const destParent = dirname(entry.dest);
      const backupDir = resolve(destParent, `.${entry.id}.backup-${runTag}`);
      rmSync(backupDir, { recursive: true, force: true });

      if (onBeforeSwap) {
        onBeforeSwap(entry);
      }

      const hadExisting = pathExistsOrSymlink(entry.dest);
      if (hadExisting) {
        renameSync(entry.dest, backupDir);
      }
      try {
        renameSync(stageDir, entry.dest);
      } catch (err) {
        if (hadExisting && pathExistsOrSymlink(backupDir) && !pathExistsOrSymlink(entry.dest)) {
          renameSync(backupDir, entry.dest);
        }
        throw err;
      }
      swapped.push({ entry, backupDir: hadExisting ? backupDir : null });
    }
  } catch (err) {
    // Roll back any already-swapped entries to preserve previous verified vendor trees intact.
    for (const done of swapped.reverse()) {
      try {
        if (pathExistsOrSymlink(done.entry.dest)) {
          rmSync(done.entry.dest, { recursive: true, force: true });
        }
        if (done.backupDir && pathExistsOrSymlink(done.backupDir)) {
          renameSync(done.backupDir, done.entry.dest);
        }
      } catch {
        // Best-effort rollback
      }
    }
    throw err;
  } finally {
    releasePublishLock(lockDir);
  }

  for (const done of swapped) {
    if (done.backupDir) {
      rmSync(done.backupDir, { recursive: true, force: true });
    }
  }
}

export function syncUpstream({
  rootDir = resolve(__dirname, '..'),
  lockPath = resolve(rootDir, 'upstream.lock.json'),
  force = false,
  onBeforeSwap = undefined,
} = {}) {
  const resolvedRoot = resolve(rootDir);
  const resolvedLock = resolve(lockPath);
  const lock = JSON.parse(readFileSync(resolvedLock, 'utf8'));
  if (!lock || !Array.isArray(lock.repos)) {
    throw new Error(`[sync-upstream] Invalid lockfile "${resolvedLock}": expected repos[]`);
  }

  const entries = lock.repos.map((raw) => validateEntry(raw, resolvedRoot));
  const runTag = `${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  const tmpRoot = resolve(resolvedRoot, '.upstream-tmp');
  const runTmpDir = resolve(tmpRoot, `run-${runTag}`);
  const staged = [];
  const results = [];

  try {
    for (const entry of entries) {
      if (!force && isAlreadyVerified(entry)) {
        const recorded = readFileSync(resolve(entry.dest, '.synced-commit'), 'utf8').trim();
        console.log(`[sync-upstream] ${entry.id}@${recorded} already verified — skipping.`);
        results.push({ id: entry.id, status: 'skipped', resolvedCommit: recorded });
        continue;
      }

      console.log(
        `[sync-upstream] Fetching ${entry.id}@${entry.commit} (sparse: ${entry.sparsePaths.join(', ')})...`,
      );
      const item = stageVerifiedEntry(entry, runTmpDir, runTag);
      staged.push(item);
    }

    if (staged.length > 0) {
      publishStagedEntries(staged, runTag, tmpRoot, onBeforeSwap);
      for (const item of staged) {
        console.log(
          `[sync-upstream] Synced ${item.entry.id}@${item.checkedOutSha} -> ${item.entry.targetDir}`,
        );
        results.push({
          id: item.entry.id,
          status: 'synced',
          resolvedCommit: item.checkedOutSha,
        });
      }
    }
  } catch (err) {
    for (const item of staged) {
      rmSync(item.stageDir, { recursive: true, force: true });
    }
    throw err;
  } finally {
    // Only delete this invocation's temp directory so concurrent syncs are never clobbered.
    rmSync(runTmpDir, { recursive: true, force: true });
    try {
      rmdirSync(tmpRoot);
    } catch {
      // Leave .upstream-tmp in place if another concurrent sync still uses it.
    }
  }

  return results;
}

function parseCliArgs(argv) {
  let force = false;
  let rootDir = resolve(__dirname, '..');
  let lockPath = null;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--force') {
      force = true;
    } else if (arg === '--root') {
      const next = argv[++i];
      if (!next) throw new Error('[sync-upstream] --root requires a directory path');
      rootDir = resolve(next);
    } else if (arg === '--lock') {
      const next = argv[++i];
      if (!next) throw new Error('[sync-upstream] --lock requires a file path');
      lockPath = resolve(next);
    } else {
      throw new Error(`[sync-upstream] Unknown argument "${arg}"`);
    }
  }

  return {
    force,
    rootDir,
    lockPath: lockPath ?? resolve(rootDir, 'upstream.lock.json'),
  };
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  try {
    const opts = parseCliArgs(process.argv.slice(2));
    syncUpstream(opts);
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  }
}
