import { afterEach, describe, expect, it } from 'vitest';
import { execFileSync, spawnSync } from 'node:child_process';
import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const SCRIPT_PATH = resolve(process.cwd(), '../tools/sync-upstream.mjs');
const tempDirs: string[] = [];

function makeTempDir(prefix: string): string {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function git(args: string[], cwd: string): string {
  return execFileSync('git', args, {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      GIT_AUTHOR_NAME: 'Test Author',
      GIT_AUTHOR_EMAIL: 'test@example.com',
      GIT_COMMITTER_NAME: 'Test Author',
      GIT_COMMITTER_EMAIL: 'test@example.com',
    },
  }).trim();
}

function createFixtureRepo(): {
  repoDir: string;
  repoUrl: string;
  commitA: string;
  commitB: string;
} {
  const repoDir = makeTempDir('sync-upstream-repo-');
  git(['init', '-b', 'main'], repoDir);

  mkdirSync(join(repoDir, 'webgpu'), { recursive: true });
  mkdirSync(join(repoDir, 'shared'), { recursive: true });
  mkdirSync(join(repoDir, 'ignored-dir'), { recursive: true });
  writeFileSync(join(repoDir, 'LICENSE'), 'MIT License v1\n');
  writeFileSync(join(repoDir, 'webgpu/index.js'), 'export const VERSION = "v1";\n');
  symlinkSync('index.js', join(repoDir, 'webgpu/alias.js'));
  writeFileSync(join(repoDir, 'shared/util.js'), 'export const UTIL = 1;\n');
  writeFileSync(join(repoDir, 'ignored-dir/heavy.txt'), 'not sparse\n');
  git(['add', '.'], repoDir);
  git(['commit', '-m', 'commit v1'], repoDir);
  const commitA = git(['rev-parse', 'HEAD'], repoDir);

  writeFileSync(join(repoDir, 'LICENSE'), 'MIT License v2\n');
  writeFileSync(join(repoDir, 'webgpu/index.js'), 'export const VERSION = "v2";\n');
  writeFileSync(join(repoDir, 'shared/util.js'), 'export const UTIL = 2;\n');
  git(['add', '.'], repoDir);
  git(['commit', '-m', 'commit v2'], repoDir);
  const commitB = git(['rev-parse', 'HEAD'], repoDir);

  return { repoDir, repoUrl: pathToFileURL(repoDir).href, commitA, commitB };
}

function runSyncCli(rootDir: string, lockPath: string, extraArgs: string[] = []) {
  return spawnSync(process.execPath, [SCRIPT_PATH, '--root', rootDir, '--lock', lockPath, ...extraArgs], {
    encoding: 'utf8',
  });
}

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop()!;
    rmSync(dir, { recursive: true, force: true });
  }
});

describe('tools/sync-upstream.mjs (Issue #7)', () => {
  it('syncs a pinned historical non-HEAD short SHA and full SHA over file:// (real shallow fetch), preserves relative symlinks, and records full provenance', () => {
    const { repoUrl, commitA, commitB } = createFixtureRepo();
    const workspace = makeTempDir('sync-upstream-ws-');
    const lockPath = join(workspace, 'upstream.lock.json');
    const vendorDest = join(workspace, 'web/static/vendor/engine-a');

    // Pin the older non-HEAD commit using a 10-char short SHA over file:// so --depth=1 genuinely enforces shallow fetch + unshallow
    const shortPinA = commitA.slice(0, 10);
    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'engine-a',
            repo: repoUrl,
            commit: shortPinA,
            targetDir: 'web/static/vendor/engine-a',
            sparsePaths: ['LICENSE', 'webgpu', 'shared'],
          },
        ],
      }),
    );

    const res1 = runSyncCli(workspace, lockPath);
    expect(res1.status, res1.stderr).toBe(0);
    expect(readFileSync(join(vendorDest, 'webgpu/index.js'), 'utf8')).toContain('VERSION = "v1"');
    // Relative symlink must remain verbatim & readable after clone cleanup (not dangling into deleted cloneDir!)
    const aliasPath = join(vendorDest, 'webgpu/alias.js');
    expect(lstatSync(aliasPath).isSymbolicLink()).toBe(true);
    expect(readlinkSync(aliasPath)).toBe('index.js');
    expect(readFileSync(aliasPath, 'utf8')).toContain('VERSION = "v1"');
    expect(readFileSync(join(vendorDest, 'shared/util.js'), 'utf8')).toContain('UTIL = 1');
    expect(readFileSync(join(vendorDest, 'LICENSE'), 'utf8')).toContain('v1');
    expect(existsSync(join(vendorDest, '.git'))).toBe(false);
    expect(existsSync(join(vendorDest, 'ignored-dir'))).toBe(false);

    // Records the verified full 40-char SHA + structured file-level provenance
    expect(readFileSync(join(vendorDest, '.synced-commit'), 'utf8').trim()).toBe(commitA);
    const provA = JSON.parse(readFileSync(join(vendorDest, '.synced-provenance.json'), 'utf8'));
    expect(provA).toMatchObject({
      schema: 1,
      id: 'engine-a',
      repo: repoUrl,
      requestedCommit: shortPinA,
      resolvedCommit: commitA,
      sparsePaths: ['LICENSE', 'webgpu', 'shared'],
    });
    expect(Array.isArray(provA.files)).toBe(true);
    expect(provA.files.some((f: { path: string }) => f.path === 'webgpu/index.js')).toBe(true);
    expect(provA.files.some((f: { path: string; type: string; target?: string }) => f.path === 'webgpu/alias.js' && f.type === 'symlink' && f.target === 'index.js')).toBe(true);
    expect(typeof provA.syncedAt).toBe('string');

    // Subsequent run recognizes the verified full SHA and skips cleanly
    const resSkip = runSyncCli(workspace, lockPath);
    expect(resSkip.status, resSkip.stderr).toBe(0);
    expect(resSkip.stdout).toContain('already verified — skipping');

    // --force re-syncs even when already verified
    const resForce = runSyncCli(workspace, lockPath, ['--force']);
    expect(resForce.status, resForce.stderr).toBe(0);
    expect(resForce.stdout).toContain(`Synced engine-a@${commitA}`);

    // Updating lock to full 40-char commitB atomically publishes v2
    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'engine-a',
            repo: repoUrl,
            commit: commitB,
            targetDir: 'web/static/vendor/engine-a',
            sparsePaths: ['LICENSE', 'webgpu', 'shared'],
          },
        ],
      }),
    );
    const res2 = runSyncCli(workspace, lockPath);
    expect(res2.status, res2.stderr).toBe(0);
    expect(readFileSync(join(vendorDest, 'webgpu/index.js'), 'utf8')).toContain('VERSION = "v2"');
    expect(readFileSync(aliasPath, 'utf8')).toContain('VERSION = "v2"');
    expect(readFileSync(join(vendorDest, '.synced-commit'), 'utf8').trim()).toBe(commitB);
    const provB = JSON.parse(readFileSync(join(vendorDest, '.synced-provenance.json'), 'utf8'));
    expect(provB.resolvedCommit).toBe(commitB);
    expect(provB.requestedCommit).toBe(commitB);

    // Symlink listed BEFORE its target in sparsePaths must still stage & verify cleanly
    const orderedLock = join(workspace, 'ordered-symlink.lock.json');
    const orderedDest = join(workspace, 'web/static/vendor/engine-symlink-order');
    writeFileSync(
      orderedLock,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'engine-symlink-order',
            repo: repoUrl,
            commit: commitB,
            targetDir: 'web/static/vendor/engine-symlink-order',
            sparsePaths: ['webgpu/alias.js', 'webgpu/index.js'],
          },
        ],
      }),
    );
    const resOrder = runSyncCli(workspace, orderedLock);
    expect(resOrder.status, resOrder.stderr).toBe(0);
    expect(readFileSync(join(orderedDest, 'webgpu/alias.js'), 'utf8')).toContain('VERSION = "v2"');
  });

  it('exits non-zero on unavailable pin (0000000000), preserves previous verified vendor tree intact, and writes no success stamp', () => {
    const { repoUrl, commitA } = createFixtureRepo();
    const workspace = makeTempDir('sync-upstream-fail-');
    const lockPath = join(workspace, 'upstream.lock.json');
    const vendorParent = join(workspace, 'web/static/vendor');
    const vendorDest = join(vendorParent, 'engine-a');

    // 1. Seed a verified vendor tree at commitA
    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'engine-a',
            repo: repoUrl,
            commit: commitA,
            targetDir: 'web/static/vendor/engine-a',
            sparsePaths: ['LICENSE', 'webgpu'],
          },
        ],
      }),
    );
    const seed = runSyncCli(workspace, lockPath);
    expect(seed.status, seed.stderr).toBe(0);
    const originalStamp = readFileSync(join(vendorDest, '.synced-commit'), 'utf8');
    const originalProv = readFileSync(join(vendorDest, '.synced-provenance.json'), 'utf8');
    const originalCode = readFileSync(join(vendorDest, 'webgpu/index.js'), 'utf8');

    // 2. Request unavailable pin 0000000000
    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'engine-a',
            repo: repoUrl,
            commit: '0000000000',
            targetDir: 'web/static/vendor/engine-a',
            sparsePaths: ['LICENSE', 'webgpu'],
          },
        ],
      }),
    );

    const failed = runSyncCli(workspace, lockPath);
    expect(failed.status).not.toBe(0);
    expect(failed.stderr).toMatch(/unavailable|Refusing to fall back to HEAD/i);

    // Previous verified vendor tree & stamps must remain completely intact
    expect(readFileSync(join(vendorDest, '.synced-commit'), 'utf8')).toBe(originalStamp);
    expect(readFileSync(join(vendorDest, '.synced-provenance.json'), 'utf8')).toBe(originalProv);
    expect(readFileSync(join(vendorDest, 'webgpu/index.js'), 'utf8')).toBe(originalCode);
    expect(existsSync(join(workspace, '.upstream-tmp'))).toBe(false);
    expect(readdirSync(vendorParent)).toEqual(['engine-a']);

    // 3. On a fresh targetDir with no prior tree, unavailable pin leaves no directory and no stamp
    const freshLockPath = join(workspace, 'fresh.lock.json');
    const freshDest = join(vendorParent, 'fresh-engine');
    writeFileSync(
      freshLockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'fresh-engine',
            repo: repoUrl,
            commit: '0000000000',
            targetDir: 'web/static/vendor/fresh-engine',
            sparsePaths: ['LICENSE', 'webgpu'],
          },
        ],
      }),
    );
    const freshFailed = runSyncCli(workspace, freshLockPath);
    expect(freshFailed.status).not.toBe(0);
    expect(existsSync(freshDest)).toBe(false);
    expect(existsSync(join(freshDest, '.synced-commit'))).toBe(false);
  });

  it('detects cache damage inside sparse directories and invalidates legacy stamps lacking provenance', () => {
    const { repoUrl, commitA } = createFixtureRepo();
    const workspace = makeTempDir('sync-upstream-cache-');
    const lockPath = join(workspace, 'upstream.lock.json');
    const vendorDest = join(workspace, 'web/static/vendor/engine-a');

    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'engine-a',
            repo: repoUrl,
            commit: commitA.slice(0, 10),
            targetDir: 'web/static/vendor/engine-a',
            sparsePaths: ['LICENSE', 'webgpu'],
          },
        ],
      }),
    );

    expect(runSyncCli(workspace, lockPath).status).toBe(0);

    // 1. Delete webgpu/index.js while leaving webgpu/ directory intact -> must detect damage and re-sync
    unlinkSync(join(vendorDest, 'webgpu/index.js'));
    const repairRes = runSyncCli(workspace, lockPath);
    expect(repairRes.status, repairRes.stderr).toBe(0);
    expect(repairRes.stdout).toContain(`Synced engine-a@${commitA}`);
    expect(readFileSync(join(vendorDest, 'webgpu/index.js'), 'utf8')).toContain('VERSION = "v1"');

    // 2. Legacy short-SHA .synced-commit without .synced-provenance.json must NOT skip
    unlinkSync(join(vendorDest, '.synced-provenance.json'));
    writeFileSync(join(vendorDest, '.synced-commit'), `${commitA.slice(0, 10)}\n`);
    const upgradeRes = runSyncCli(workspace, lockPath);
    expect(upgradeRes.status, upgradeRes.stderr).toBe(0);
    expect(upgradeRes.stdout).toContain(`Synced engine-a@${commitA}`);
    expect(readFileSync(join(vendorDest, '.synced-commit'), 'utf8').trim()).toBe(commitA);
    expect(existsSync(join(vendorDest, '.synced-provenance.json'))).toBe(true);
  });

  it('fails and preserves previous vendor tree when a required sparsePaths entry is missing or pin is non-SHA', () => {
    const { repoUrl, commitA, commitB } = createFixtureRepo();
    const workspace = makeTempDir('sync-upstream-sparse-');
    const lockPath = join(workspace, 'upstream.lock.json');
    const vendorDest = join(workspace, 'web/static/vendor/engine-a');

    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'engine-a',
            repo: repoUrl,
            commit: commitA,
            targetDir: 'web/static/vendor/engine-a',
            sparsePaths: ['LICENSE', 'webgpu'],
          },
        ],
      }),
    );
    expect(runSyncCli(workspace, lockPath).status).toBe(0);

    // Missing required sparsePath at valid commitB must fail without overwriting commitA
    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'engine-a',
            repo: repoUrl,
            commit: commitB,
            targetDir: 'web/static/vendor/engine-a',
            sparsePaths: ['LICENSE', 'webgpu', 'missing-runtime/index.js'],
          },
        ],
      }),
    );
    const missingRes = runSyncCli(workspace, lockPath);
    expect(missingRes.status).not.toBe(0);
    expect(missingRes.stderr).toMatch(/missing-runtime\/index\.js/);
    expect(readFileSync(join(vendorDest, '.synced-commit'), 'utf8').trim()).toBe(commitA);
    expect(readFileSync(join(vendorDest, 'webgpu/index.js'), 'utf8')).toContain('VERSION = "v1"');

    // Non-hex symbolic ref like HEAD must be rejected
    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'engine-a',
            repo: repoUrl,
            commit: 'HEAD',
            targetDir: 'web/static/vendor/engine-a',
            sparsePaths: ['LICENSE'],
          },
        ],
      }),
    );
    const headRes = runSyncCli(workspace, lockPath);
    expect(headRes.status).not.toBe(0);
    expect(headRes.stderr).toMatch(/Invalid commit pin/);
    expect(readFileSync(join(vendorDest, '.synced-commit'), 'utf8').trim()).toBe(commitA);
  });

  it('rolls back already-swapped trees on publish failure and leaves sibling concurrent temp dirs untouched', () => {
    const { repoUrl, commitA, commitB } = createFixtureRepo();
    const workspace = makeTempDir('sync-upstream-rollback-');
    const lockPath = join(workspace, 'upstream.lock.json');
    const dest1 = join(workspace, 'web/static/vendor/engine-1');
    const dest2 = join(workspace, 'web/static/vendor/engine-2');

    // Seed both engines at commitA
    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          { id: 'engine-1', repo: repoUrl, commit: commitA, targetDir: 'web/static/vendor/engine-1', sparsePaths: ['webgpu'] },
          { id: 'engine-2', repo: repoUrl, commit: commitA, targetDir: 'web/static/vendor/engine-2', sparsePaths: ['webgpu'] },
        ],
      }),
    );
    expect(runSyncCli(workspace, lockPath).status).toBe(0);
    const origProv1 = readFileSync(join(dest1, '.synced-provenance.json'), 'utf8');
    const origProv2 = readFileSync(join(dest2, '.synced-provenance.json'), 'utf8');

    // Simulate a concurrent sync holding its own run dir inside .upstream-tmp
    const siblingTmp = join(workspace, '.upstream-tmp', 'run-concurrent-sibling');
    mkdirSync(siblingTmp, { recursive: true });
    writeFileSync(join(siblingTmp, 'keep.txt'), 'concurrent-run');

    // Upgrade both to commitB, then inject a publish-phase failure right before engine-2 swaps (after engine-1 already swapped!)
    writeFileSync(
      lockPath,
      JSON.stringify({
        schema: 1,
        repos: [
          { id: 'engine-1', repo: repoUrl, commit: commitB, targetDir: 'web/static/vendor/engine-1', sparsePaths: ['webgpu'] },
          { id: 'engine-2', repo: repoUrl, commit: commitB, targetDir: 'web/static/vendor/engine-2', sparsePaths: ['webgpu'] },
        ],
      }),
    );

    const probe = spawnSync(
      process.execPath,
      [
        '--input-type=module',
        '-e',
        `import { syncUpstream } from ${JSON.stringify(pathToFileURL(SCRIPT_PATH).href)};
         syncUpstream({
           rootDir: ${JSON.stringify(workspace)},
           lockPath: ${JSON.stringify(lockPath)},
           onBeforeSwap(entry) {
             if (entry.id === 'engine-2') throw new Error('Injected publish swap failure on engine-2');
           },
         });`,
      ],
      { encoding: 'utf8' },
    );
    expect(probe.status).not.toBe(0);
    expect(probe.stderr).toMatch(/Injected publish swap failure/);

    // engine-1 MUST be rolled back to its exact previous verified commitA tree & provenance
    expect(readFileSync(join(dest1, '.synced-commit'), 'utf8').trim()).toBe(commitA);
    expect(readFileSync(join(dest1, '.synced-provenance.json'), 'utf8')).toBe(origProv1);
    expect(readFileSync(join(dest1, 'webgpu/index.js'), 'utf8')).toContain('VERSION = "v1"');

    // engine-2 MUST remain completely untouched
    expect(readFileSync(join(dest2, '.synced-commit'), 'utf8').trim()).toBe(commitA);
    expect(readFileSync(join(dest2, '.synced-provenance.json'), 'utf8')).toBe(origProv2);
    expect(readFileSync(join(dest2, 'webgpu/index.js'), 'utf8')).toContain('VERSION = "v1"');

    // No leftover stage/backup directories in vendor/, AND sibling concurrent temp dir was NOT deleted
    expect(readdirSync(join(workspace, 'web/static/vendor')).sort()).toEqual(['engine-1', 'engine-2']);
    expect(readFileSync(join(siblingTmp, 'keep.txt'), 'utf8')).toBe('concurrent-run');
  });

  it('rejects upstream reserved stamp files/symlinks (.synced-commit / .synced-provenance.json) and symlinked targetDir parents escaping --root', () => {
    const workspace = makeTempDir('sync-upstream-sec-');
    const outsideDir = makeTempDir('sync-upstream-outside-');
    const victimFile = join(outsideDir, 'victim.txt');
    writeFileSync(victimFile, 'DO_NOT_OVERWRITE');

    // 1. Upstream repo containing a nested .synced-commit symlink pointing outside
    const maliciousRepo = makeTempDir('sync-upstream-malrepo-');
    git(['init', '-b', 'main'], maliciousRepo);
    mkdirSync(join(maliciousRepo, 'webgpu'), { recursive: true });
    writeFileSync(join(maliciousRepo, 'webgpu/index.js'), 'export const X = 1;\n');
    symlinkSync(victimFile, join(maliciousRepo, 'webgpu/.synced-commit'));
    git(['add', '.'], maliciousRepo);
    git(['commit', '-m', 'add nested reserved symlink'], maliciousRepo);
    const badSha = git(['rev-parse', 'HEAD'], maliciousRepo);

    const lock1 = join(workspace, 'mal1.lock.json');
    writeFileSync(
      lock1,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'mal-engine',
            repo: pathToFileURL(maliciousRepo).href,
            commit: badSha,
            targetDir: 'web/static/vendor/mal-engine',
            sparsePaths: ['webgpu'],
          },
        ],
      }),
    );
    const res1 = runSyncCli(workspace, lock1);
    expect(res1.status).not.toBe(0);
    expect(res1.stderr).toMatch(/reserved metadata entry/i);
    expect(readFileSync(victimFile, 'utf8')).toBe('DO_NOT_OVERWRITE');
    expect(existsSync(join(workspace, 'web/static/vendor/mal-engine'))).toBe(false);

    // 2. Listing .synced-commit directly in sparsePaths is rejected upfront
    const lock2 = join(workspace, 'mal2.lock.json');
    writeFileSync(
      lock2,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'mal-root',
            repo: pathToFileURL(maliciousRepo).href,
            commit: badSha,
            targetDir: 'web/static/vendor/mal-root',
            sparsePaths: ['.synced-commit'],
          },
        ],
      }),
    );
    const res2 = runSyncCli(workspace, lock2);
    expect(res2.status).not.toBe(0);
    expect(res2.stderr).toMatch(/reserved sparsePath/i);
    expect(readFileSync(victimFile, 'utf8')).toBe('DO_NOT_OVERWRITE');

    // 3. Symlinked parent directory escaping --root (root/linked -> outsideDir) must be rejected
    const { repoUrl, commitA } = createFixtureRepo();
    symlinkSync(outsideDir, join(workspace, 'linked'));
    const lock3 = join(workspace, 'escape.lock.json');
    writeFileSync(
      lock3,
      JSON.stringify({
        schema: 1,
        repos: [
          {
            id: 'escape-engine',
            repo: repoUrl,
            commit: commitA,
            targetDir: 'linked/engine',
            sparsePaths: ['webgpu'],
          },
        ],
      }),
    );
    const res3 = runSyncCli(workspace, lock3);
    expect(res3.status).not.toBe(0);
    expect(res3.stderr).toMatch(/escapes/i);
    expect(existsSync(join(outsideDir, 'engine'))).toBe(false);
  });
});
