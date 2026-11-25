#!/usr/bin/env node
/**
 * Bundle Cloudflare Pages Advanced Mode `_worker.js` and `_routes.json` into `web/build/`,
 * and ensure the static default OG image `web/static/og-default.png` + `web/build/og-default.png` exist.
 *
 * Placing `_worker.js` directly inside `web/build/` works identically regardless of whether
 * Cloudflare Pages has Root Directory set to `/` or `web/`, and works with `wrangler pages deploy web/build`.
 */

import { existsSync, mkdirSync, writeFileSync, statSync, copyFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');
const webDir = resolve(repoRoot, 'web');
const buildDir = resolve(webDir, 'build');
const staticDir = resolve(webDir, 'static');
const entryPath = resolve(webDir, 'src/lib/og/cf-worker.ts');
const outWorkerPath = resolve(buildDir, '_worker.js');
const outRoutesPath = resolve(buildDir, '_routes.json');

const requireFromWeb = createRequire(resolve(webDir, 'package.json'));
const esbuild = requireFromWeb('esbuild');

if (!existsSync(buildDir)) {
  mkdirSync(buildDir, { recursive: true });
}

await esbuild.build({
  entryPoints: [entryPath],
  outfile: outWorkerPath,
  bundle: true,
  format: 'esm',
  target: 'es2022',
  platform: 'browser',
  minify: true,
  legalComments: 'none',
});

const routesConfig = {
  version: 1,
  include: ['/*'],
  exclude: ['/_app/*', '/vendor/*', '/favicon.svg', '/og-default.png', '/demo-welcome*', '/llms.txt', '/robots.txt'],
};
writeFileSync(outRoutesPath, JSON.stringify(routesConfig, null, 2) + '\n', 'utf8');

// Also generate/refresh static fallback og-default.png using the exact same bundled worker module
const staticOgPath = resolve(staticDir, 'og-default.png');
const buildOgPath = resolve(buildDir, 'og-default.png');
try {
  const mod = await import(`${pathToFileURL(outWorkerPath).href}?t=${Date.now()}`);
  const res = await mod.default.fetch(new Request('https://denovo.run/api/og'), {
    ASSETS: { fetch: async () => new Response('Not found', { status: 404 }) },
  });
  if (res.ok) {
    const buf = Buffer.from(await res.arrayBuffer());
    writeFileSync(staticOgPath, buf);
    writeFileSync(buildOgPath, buf);
  }
} catch (err) {
  if (existsSync(staticOgPath) && !existsSync(buildOgPath)) {
    copyFileSync(staticOgPath, buildOgPath);
  } else {
    console.warn('[build-cf-worker] Warning: could not refresh og-default.png:', err);
  }
}

let buildGitSha = null;
try {
  const rawSha = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 2000,
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim();
  if (/^[0-9a-f]{7,40}$/i.test(rawSha)) {
    buildGitSha = rawSha.toLowerCase();
  }
} catch {
  const envSha = String(
    process.env.CF_PAGES_COMMIT_SHA || process.env.GITHUB_SHA || process.env.DENOVO_BUILD_SHA || '',
  ).trim();
  if (/^[0-9a-f]{7,40}$/i.test(envSha)) {
    buildGitSha = envSha.toLowerCase();
  }
}
writeFileSync(
  resolve(buildDir, '.build-meta.json'),
  JSON.stringify({ gitSha: buildGitSha, builtAt: new Date().toISOString() }, null, 2) + '\n',
  'utf8',
);

const workerKb = (statSync(outWorkerPath).size / 1024).toFixed(1);
console.log(
  `[build-cf-worker] Built web/build/_worker.js (${workerKb} KB) + _routes.json + og-default.png + .build-meta.json (${buildGitSha ? buildGitSha.slice(0, 7) : 'no-git'})`,
);
