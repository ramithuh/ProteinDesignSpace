#!/usr/bin/env node
/**
 * Zero-Python Self-Hosted WebGPU Server for denovo.run (ProteinDesignSpace).
 *
 * Runs ALL 22 existing browser-native JS/WebGPU workers on the host machine's GPU
 * inside a single persistent headless Chromium instance — zero Python, zero uv/conda,
 * zero CUDA wheel conflicts — while serving the static denovo.run web app and
 * exposing a clean HTTP + SSE execution API for LAN / Tailscale / iPad / thin clients.
 *
 * Usage:
 *   node server/selfhost-server.mjs
 *
 * Environment variables:
 *   PORT=8787                         HTTP listen port (default: 8787)
 *   HOST=0.0.0.0                      Bind address (default: 0.0.0.0)
 *   DENOVO_API_TOKEN=...              Optional Bearer token for /api/* endpoints
 *   DENOVO_ALLOWED_ORIGINS=*          Comma-separated CORS origins or * (default: *)
 *   DENOVO_DATA_DIR=...               Persistent Chromium profile / OPFS weight cache dir
 *   DENOVO_CHROME_PATH=...            Explicit Chromium/Chrome binary path (auto-detected if omitted)
 *   DENOVO_ALLOW_SOFTWARE_GPU=0       Set to 1 to permit SwiftShader/software WebGPU fallback
 *   DENOVO_MAX_BODY_MB=128            Max JSON request payload size in MB (default: 128)
 */

import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const BUILD_DIR = path.resolve(REPO_ROOT, 'web', 'build');

const PORT = Number(process.env.PORT || 8787);
const HOST = process.env.HOST || '0.0.0.0';
const API_TOKEN = (process.env.DENOVO_API_TOKEN || '').trim();
const ALLOWED_ORIGINS = (process.env.DENOVO_ALLOWED_ORIGINS || '*').trim();
const DATA_DIR = path.resolve(
  process.env.DENOVO_DATA_DIR || path.join(os.homedir(), '.cache', 'denovo-selfhost', 'profile'),
);
const ALLOW_SOFTWARE_GPU = process.env.DENOVO_ALLOW_SOFTWARE_GPU === '1';
const MAX_BODY_BYTES = Math.max(4, Number(process.env.DENOVO_MAX_BODY_MB || 128)) * 1024 * 1024;
const MAX_FOLD_RESIDUES = Math.max(
  512,
  Math.min(4096, Math.round(Number(process.env.DENOVO_MAX_FOLD_RESIDUES || 2048) || 2048)),
);

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.mp4': 'video/mp4',
  '.txt': 'text/plain; charset=utf-8',
  '.pdb': 'text/plain; charset=utf-8',
  '.cif': 'text/plain; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
};

function loadPlaywright() {
  const candidates = [
    path.join(REPO_ROOT, 'server', 'package.json'),
    path.join(REPO_ROOT, 'web', 'package.json'),
    path.join(REPO_ROOT, 'package.json'),
    process.env.DENOVO_PLAYWRIGHT_PKG || '',
  ].filter(Boolean);

  for (const pkg of candidates) {
    if (!fs.existsSync(pkg)) continue;
    try {
      const req = createRequire(pkg);
      try {
        return req('playwright-core');
      } catch {
        return req('playwright');
      }
    } catch {
      /* try next */
    }
  }
  // Fallback: check sibling denovo-webgpu-models or global node_modules
  const siblingPkg = path.resolve(REPO_ROOT, '..', 'denovo-webgpu-models', 'models', 'rfdiffusion', 'package.json');
  if (fs.existsSync(siblingPkg)) {
    try {
      return createRequire(siblingPkg)('playwright-core');
    } catch {
      /* ignore */
    }
  }
  throw new Error(
    'Playwright is not installed. Run `npm install playwright-core` (and `npx playwright-core install chromium`) in server/ or set DENOVO_PLAYWRIGHT_PKG.',
  );
}

function findChromeExecutable() {
  if (process.env.DENOVO_CHROME_PATH && fs.existsSync(process.env.DENOVO_CHROME_PATH)) {
    return process.env.DENOVO_CHROME_PATH;
  }
  // Check Playwright's local chromium cache first
  const pwCacheDir = path.join(os.homedir(), '.cache', 'ms-playwright');
  if (fs.existsSync(pwCacheDir)) {
    const dirs = fs
      .readdirSync(pwCacheDir)
      .filter((d) => d.startsWith('chromium-'))
      .sort()
      .reverse();
    for (const d of dirs) {
      for (const rel of [
        'chrome-linux64/chrome',
        'chrome-linux/chrome',
        'chrome-linux-arm64/chrome',
        'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
        'chrome-mac/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
      ]) {
        const full = path.join(pwCacheDir, d, rel);
        if (fs.existsSync(full)) return full;
      }
    }
  }
  for (const sysPath of [
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ]) {
    if (fs.existsSync(sysPath)) return sysPath;
  }
  return undefined;
}

function detectLinuxVulkanIcd() {
  if (process.platform !== 'linux' || process.env.VK_ICD_FILENAMES) return process.env.VK_ICD_FILENAMES;
  for (const icd of [
    '/usr/share/vulkan/icd.d/nvidia_icd.json',
    '/usr/share/vulkan/icd.d/radeon_icd.x86_64.json',
    '/usr/share/vulkan/icd.d/intel_icd.x86_64.json',
  ]) {
    if (fs.existsSync(icd)) return icd;
  }
  return undefined;
}

function formatShortGpuName(rawName) {
  const cleaned = String(rawName || '')
    .replace(/^NVIDIA\s+GeForce\s+/i, '')
    .replace(/^NVIDIA\s+/i, '')
    .replace(/^AMD\s+Radeon\(TM\)\s+/i, '')
    .replace(/^AMD\s+Radeon\s+/i, '')
    .replace(/^Intel\(R\)\s+/i, '')
    .replace(/\s+Laptop\s+GPU$/i, '')
    .trim();
  return cleaned.length > 22 ? `${cleaned.slice(0, 21)}…` : cleaned;
}

function detectHostGpuHardware() {
  const customLabel = (process.env.DENOVO_GPU_LABEL || '').trim();
  if (customLabel) {
    return { name: customLabel, shortName: formatShortGpuName(customLabel), vramMB: null };
  }
  // 1. NVIDIA (Linux / Windows) via nvidia-smi
  try {
    const out = execFileSync(
      'nvidia-smi',
      ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
      { encoding: 'utf8', timeout: 1500, stdio: ['ignore', 'pipe', 'ignore'] },
    ).trim();
    const firstLine = out.split(/\r?\n/)[0]?.trim();
    if (firstLine) {
      const [rawName, rawMem] = firstLine.split(',').map((s) => s.trim());
      if (rawName) {
        const memNum = Number(rawMem);
        const vramMB = Number.isFinite(memNum) && memNum > 0 ? Math.round(memNum) : null;
        return { name: rawName, shortName: formatShortGpuName(rawName), vramMB };
      }
    }
  } catch {
    /* nvidia-smi unavailable */
  }
  // 2. macOS Apple Silicon / discrete GPU via system_profiler
  if (process.platform === 'darwin') {
    try {
      const out = execFileSync('system_profiler', ['SPDisplaysDataType'], {
        encoding: 'utf8',
        timeout: 2000,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      const m = out.match(/Chipset Model:\s*(.+)/);
      if (m?.[1]?.trim()) {
        const name = m[1].trim();
        return { name, shortName: formatShortGpuName(name), vramMB: null };
      }
    } catch {
      /* ignore */
    }
  }
  // 3. Linux AMD / Intel fallback via lspci
  if (process.platform === 'linux') {
    try {
      const out = execFileSync('lspci', ['-mm'], {
        encoding: 'utf8',
        timeout: 1500,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      const line = out
        .split(/\r?\n/)
        .find((l) => /VGA compatible controller|3D controller/i.test(l));
      if (line) {
        const cols = [...line.matchAll(/"([^"]*)"/g)].map((x) => x[1]);
        const name = [cols[1], cols[2]].filter(Boolean).join(' ').trim();
        if (name) return { name, shortName: formatShortGpuName(name), vramMB: null };
      }
    } catch {
      /* ignore */
    }
  }
  return null;
}

const HOST_GPU_HW = detectHostGpuHardware();

function enrichGpuInfo(gpu) {
  if (!gpu) return null;
  if (!HOST_GPU_HW || gpu.fallback) return gpu;
  const vramLabel =
    HOST_GPU_HW.vramMB && HOST_GPU_HW.vramMB >= 1024
      ? `${Math.round(HOST_GPU_HW.vramMB / 1024)} GB`
      : null;
  return {
    ...gpu,
    name: HOST_GPU_HW.name,
    shortName: HOST_GPU_HW.shortName,
    vramMB: HOST_GPU_HW.vramMB,
    description: vramLabel ? `${HOST_GPU_HW.name} (${vramLabel})` : HOST_GPU_HW.name,
  };
}

function setCorsHeaders(req, res) {
  const origin = req.headers.origin;
  if (ALLOWED_ORIGINS === '*') {
    res.setHeader('Access-Control-Allow-Origin', origin || '*');
  } else if (origin) {
    const list = ALLOWED_ORIGINS.split(',').map((s) => s.trim()).filter(Boolean);
    if (list.includes(origin)) {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Vary', 'Origin');
    }
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept');
  if (req.headers['access-control-request-private-network'] === 'true') {
    res.setHeader('Access-Control-Allow-Private-Network', 'true');
  }
}

function isAuthorized(req) {
  if (!API_TOKEN) return true;
  const auth = String(req.headers.authorization || '');
  return auth === `Bearer ${API_TOKEN}`;
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    req.on('data', (chunk) => {
      total += chunk.length;
      if (total > MAX_BODY_BYTES) {
        reject(new Error(`Payload exceeds limit (${Math.round(MAX_BODY_BYTES / 1024 / 1024)} MB)`));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      try {
        const text = Buffer.concat(chunks).toString('utf8');
        resolve(text ? JSON.parse(text) : {});
      } catch (err) {
        reject(new Error(`Invalid JSON body: ${err instanceof Error ? err.message : String(err)}`));
      }
    });
    req.on('error', reject);
  });
}

function sendSse(res, event, data) {
  if (res.writableEnded || res.destroyed) return;
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function serveStaticFile(req, res, pathname) {
  let cleanPath = decodeURIComponent(pathname.split('?')[0]);
  if (cleanPath === '/' || cleanPath === '') cleanPath = '/index.html';

  const tryFiles = [
    path.join(BUILD_DIR, cleanPath),
    path.join(BUILD_DIR, `${cleanPath}.html`),
    path.join(BUILD_DIR, cleanPath, 'index.html'),
  ];

  let target = null;
  for (const cand of tryFiles) {
    const resolved = path.resolve(cand);
    if (!resolved.startsWith(BUILD_DIR)) continue;
    if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) {
      target = resolved;
      break;
    }
  }

  // SPA fallback to index.html for client-side routes without file extension
  if (!target && !path.extname(cleanPath)) {
    const fallback = path.join(BUILD_DIR, 'index.html');
    if (fs.existsSync(fallback)) target = fallback;
  }

  if (!target) {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
    return;
  }

  const ext = path.extname(target).toLowerCase();
  res.writeHead(200, {
    'Content-Type': MIME_TYPES[ext] || 'application/octet-stream',
    'Cache-Control': target.includes('/_app/immutable/') ? 'public, max-age=31536000, immutable' : 'no-cache',
  });
  fs.createReadStream(target).pipe(res);
}

// ---- Headless Host-GPU Chromium Runner Manager ----
let runnerContext = null;
let runnerPage = null;
let runnerHealth = { ready: false, gpu: null, activeAdapter: null, error: 'Starting host GPU runner...' };

const jobQueue = [];
let runningJob = null;

function notifyQueuePositions() {
  for (let i = 0; i < jobQueue.length; i++) {
    const item = jobQueue[i];
    sendSse(item.res, 'progress', {
      status: 'loading',
      phase: 'queue',
      progress: 0.01,
      detail: `Queued on host GPU (position #${i + 1})...`,
    });
  }
}

async function pumpQueue() {
  if (runningJob || jobQueue.length === 0) return;
  const next = jobQueue.shift();
  notifyQueuePositions();

  if (next.cancelled || next.res.writableEnded || next.res.destroyed) {
    void pumpQueue();
    return;
  }

  runningJob = next;
  const heartbeat = setInterval(() => {
    if (!next.res.writableEnded && !next.res.destroyed) {
      next.res.write(': ping\n\n');
    }
  }, 15_000);

  try {
    if (!runnerPage || !runnerHealth.ready) {
      throw new Error(runnerHealth.error || 'Host GPU runner is still starting.');
    }
    if (runnerHealth.error) {
      throw new Error(runnerHealth.error);
    }

    const result = await runnerPage.evaluate(
      async ({ jobId, node, inputs, maxFoldResidues }) => {
        if (!window.__denovoHostRunner) {
          throw new Error('Host runner bridge not mounted.');
        }
        return await window.__denovoHostRunner.runNode(jobId, { node, inputs, maxFoldResidues });
      },
      { jobId: next.jobId, node: next.node, inputs: next.inputs, maxFoldResidues: MAX_FOLD_RESIDUES },
    );

    if (!next.cancelled && !next.res.writableEnded) {
      sendSse(next.res, 'result', result);
      next.res.end();
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (!next.cancelled && !next.res.writableEnded) {
      sendSse(next.res, 'error', { message });
      next.res.end();
    }
  } finally {
    clearInterval(heartbeat);
    if (runningJob === next) runningJob = null;
    try {
      if (runnerPage) {
        const h = await runnerPage.evaluate(() => window.__denovoHostRunner?.getHealth?.() ?? null);
        if (h) runnerHealth = h;
      }
    } catch {
      /* ignore */
    }
    void pumpQueue();
  }
}

async function cancelJob(jobId) {
  const qIdx = jobQueue.findIndex((j) => j.jobId === jobId);
  if (qIdx !== -1) {
    const [removed] = jobQueue.splice(qIdx, 1);
    removed.cancelled = true;
    sendSse(removed.res, 'error', { message: 'Workflow run cancelled' });
    removed.res.end();
    notifyQueuePositions();
    return true;
  }
  if (runningJob && runningJob.jobId === jobId) {
    runningJob.cancelled = true;
    try {
      await runnerPage?.evaluate((id) => window.__denovoHostRunner?.cancel(id), jobId);
    } catch {
      /* ignore */
    }
    return true;
  }
  return false;
}

async function startHostGpuRunner(localOrigin) {
  const { chromium } = loadPlaywright();
  const executablePath = findChromeExecutable();
  fs.mkdirSync(DATA_DIR, { recursive: true });
  for (const lf of ['SingletonLock', 'SingletonSocket', 'SingletonCookie']) {
    try {
      fs.rmSync(path.join(DATA_DIR, lf), { force: true });
    } catch {
      /* ignore */
    }
  }

  const env = { ...process.env };
  const vkIcd = detectLinuxVulkanIcd();
  if (process.platform === 'linux') {
    delete env.DISPLAY;
    delete env.WAYLAND_DISPLAY;
    if (vkIcd) env.VK_ICD_FILENAMES = vkIcd;
  }

  const args = [
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu-sandbox',
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
  ];
  if (process.platform === 'linux') {
    args.push(
      '--ozone-platform=headless',
      '--enable-features=Vulkan',
      '--use-angle=vulkan',
      '--use-vulkan=native',
      '--disable-vulkan-surface',
    );
  }

  console.log(`[denovo-selfhost] Launching host WebGPU Chromium (${executablePath || 'bundled'})...`);
  if (vkIcd) console.log(`[denovo-selfhost] Vulkan ICD: ${vkIcd}`);

  runnerContext = await chromium.launchPersistentContext(DATA_DIR, {
    ...(executablePath ? { executablePath } : {}),
    headless: false,
    env,
    args,
  });

  runnerPage = runnerContext.pages()[0] || (await runnerContext.newPage());
  runnerPage.on('pageerror', (err) => {
    console.error('[denovo-selfhost] Host runner pageerror:', err.message);
  });
  runnerPage.on('console', (msg) => {
    if (msg.type() === 'error') {
      console.error('[denovo-selfhost] Host runner console.error:', msg.text());
    }
  });

  await runnerPage.exposeFunction('__denovoHostEmit', (ev) => {
    if (!ev || typeof ev !== 'object') return;
    if (ev.type === 'progress' && runningJob && ev.jobId === runningJob.jobId) {
      sendSse(runningJob.res, 'progress', {
        status: ev.status,
        phase: ev.phase,
        progress: ev.progress,
        detail: ev.detail,
      });
    }
  });

  await runnerPage.goto(`${localOrigin}/runner`, { waitUntil: 'domcontentloaded' });
  await runnerPage.waitForFunction(() => Boolean(window.__denovoHostRunner?.getHealth()?.ready), {
    timeout: 30_000,
  });

  runnerHealth = await runnerPage.evaluate(() => window.__denovoHostRunner.getHealth());
  if (runnerHealth.gpu?.fallback && !ALLOW_SOFTWARE_GPU) {
    runnerHealth = {
      ...runnerHealth,
      error:
        'Host Chromium fell back to software WebGPU (SwiftShader). Install Vulkan/GPU drivers or set DENOVO_ALLOW_SOFTWARE_GPU=1 to override.',
    };
    console.warn(`[denovo-selfhost] WARNING: ${runnerHealth.error}`);
  } else if (runnerHealth.gpu) {
    console.log(
      `[denovo-selfhost] Host GPU ready: ${runnerHealth.gpu.vendor} ${runnerHealth.gpu.description} (fallback=${runnerHealth.gpu.fallback})`,
    );
  } else {
    console.warn(`[denovo-selfhost] Host GPU unavailable: ${runnerHealth.error}`);
  }
}

async function main() {
  if (!fs.existsSync(path.join(BUILD_DIR, 'index.html'))) {
    console.error(`[denovo-selfhost] Missing web/build/index.html. Run "npm --prefix web run build" first.`);
    process.exit(1);
  }

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
    setCorsHeaders(req, res);

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    if (url.pathname.startsWith('/api/')) {
      if (!isAuthorized(req)) {
        res.writeHead(401, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false, error: 'Unauthorized: invalid or missing Bearer token.' }));
        return;
      }

      if (req.method === 'GET' && url.pathname === '/api/health') {
        try {
          if (runnerPage && !runningJob) {
            const h = await runnerPage.evaluate(() => window.__denovoHostRunner?.getHealth?.() ?? null);
            if (h) runnerHealth = { ...h, error: runnerHealth.error || h.error };
          }
        } catch {
          /* ignore */
        }
        const ok = Boolean(runnerHealth.ready && runnerHealth.gpu && !runnerHealth.error);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify({
            ok,
            service: 'denovo-selfhost',
            version: '0.1.0',
            gpu: enrichGpuInfo(runnerHealth.gpu),
            activeAdapter: runnerHealth.activeAdapter,
            queueDepth: jobQueue.length + (runningJob ? 1 : 0),
            runningJobId: runningJob?.jobId ?? null,
            maxFoldResidues: MAX_FOLD_RESIDUES,
            ...(runnerHealth.error ? { error: runnerHealth.error } : {}),
          }),
        );
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/cancel') {
        try {
          const body = await readJsonBody(req);
          const cancelled = await cancelJob(String(body.jobId || ''));
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: true, cancelled }));
        } catch (err) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: err instanceof Error ? err.message : String(err) }));
        }
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/release-gpu') {
        try {
          if (!runningJob && runnerPage) {
            await runnerPage.evaluate(() => window.__denovoHostRunner?.releaseGpu?.());
            runnerHealth.activeAdapter = null;
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: true }));
        } catch (err) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: err instanceof Error ? err.message : String(err) }));
        }
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/run-node') {
        let body;
        try {
          body = await readJsonBody(req);
        } catch (err) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: err instanceof Error ? err.message : String(err) }));
          return;
        }
        if (!body?.node?.adapterId) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: 'Missing node.adapterId in request body.' }));
          return;
        }

        const jobId = String(body.jobId || `job_${Date.now().toString(36)}`);
        res.writeHead(200, {
          'Content-Type': 'text/event-stream; charset=utf-8',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
          'X-Accel-Buffering': 'no',
        });

        const item = {
          jobId,
          node: body.node,
          inputs: body.inputs ?? {},
          res,
          cancelled: false,
        };

        req.on('close', () => {
          if (!res.writableEnded) {
            void cancelJob(jobId);
          }
        });

        jobQueue.push(item);
        if (runningJob) {
          notifyQueuePositions();
        }
        void pumpQueue();
        return;
      }

      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, error: 'Unknown API endpoint' }));
      return;
    }

    serveStaticFile(req, res, url.pathname);
  });

  await new Promise((resolve) => server.listen(PORT, HOST, resolve));
  const localOrigin = `http://127.0.0.1:${PORT}`;
  console.log(`[denovo-selfhost] Listening on http://${HOST}:${PORT} (serving ${BUILD_DIR})`);
  console.log(`[denovo-selfhost] Persistent weight cache: ${DATA_DIR}`);
  if (API_TOKEN) {
    console.log('[denovo-selfhost] Bearer API token auth: ENABLED');
  }

  await startHostGpuRunner(localOrigin);

  const shutdown = async () => {
    console.log('\n[denovo-selfhost] Shutting down...');
    try {
      await runnerContext?.close();
    } catch {
      /* ignore */
    }
    server.close(() => process.exit(0));
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch((err) => {
  console.error('[denovo-selfhost] Fatal startup error:', err);
  process.exit(1);
});
