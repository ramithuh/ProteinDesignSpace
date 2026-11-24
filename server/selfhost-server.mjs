#!/usr/bin/env node
/**
 * Zero-Python Self-Hosted WebGPU Server for denovo.run (ProteinDesignSpace).
 *
 * Hardened host-GPU execution server:
 *   - Defaults to loopback (127.0.0.1) and same-origin only (no wildcard CORS)
 *   - Refuses non-loopback binds unless DENOVO_API_TOKEN is configured
 *   - Constant-time Bearer token verification (SHA-256 + crypto.timingSafeEqual)
 *   - Server-side Origin allowlist + Content-Type: application/json enforcement
 *   - Canonical realpath + path.relative static file containment & browser security headers
 *   - Allowlisted GPU adapter validation + bounded upload/queue/job-timeout budgets
 *   - Per-job random cancel capability tokens (prevents cross-user job cancellation)
 *   - Minimal allowlisted environment + Chromium sandbox enabled by default
 */

import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_REPO_ROOT = path.resolve(__dirname, '..');
const DEFAULT_BUILD_DIR = path.resolve(DEFAULT_REPO_ROOT, 'web', 'build');

/** Explicit allowlist of GPU-backed denovo.run adapters permitted on /api/run-node (H4). */
export const ALLOWED_GPU_ADAPTER_IDS = new Set([
  'web-protein-mpnn',
  'localfold-af2',
  'rfd4-proteina',
  'genie3',
  'thermompnn',
  'ligandmpnn',
  'lasermpnn',
  'antifold-esmif1',
  'protpardelle',
  'rfdiffusion',
  'rfd3',
  'boltzgen',
  'evodiff',
  'la-proteina',
  'proteina',
  'genie2',
  'frameflow',
  'foldflow2',
  'sid-protein',
  'esmc',
  'dyna1',
  'gnina',
]);

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

const VALID_HOST_HEADER_RE = /^[a-zA-Z0-9._\-:[\]]{1,255}$/;
const MAX_CONTROL_BODY_BYTES = 64 * 1024;
const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
const SESSION_COOKIE_NAME = 'denovo_session';
const SESSION_COOKIE_MAX_AGE_SEC = 30 * 24 * 60 * 60;
const AUTH_RATE_LIMIT_WINDOW_MS = 60_000;
const AUTH_RATE_LIMIT_MAX_FAILURES = 10;

function parseFiniteInt(raw, fallback, min, max) {
  if (raw === undefined || raw === null || raw === '') return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.round(n)));
}

function isLoopbackHost(host) {
  const h = String(host || '').trim().toLowerCase();
  return h === '127.0.0.1' || h === '::1' || h === 'localhost';
}

function isLoopbackRemoteAddress(addr) {
  const a = String(addr || '').trim().toLowerCase();
  return a === '127.0.0.1' || a === '::1' || a === '::ffff:127.0.0.1';
}

function sha256Buf(str) {
  return crypto.createHash('sha256').update(String(str), 'utf8').digest();
}

function constantTimeEquals(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string' || a.length === 0 || b.length === 0) {
    return false;
  }
  return crypto.timingSafeEqual(sha256Buf(a), sha256Buf(b));
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
  try {
    const out = execFileSync(
      'nvidia-smi',
      ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
      { encoding: 'utf8', timeout: 1500, maxBuffer: 64 * 1024, stdio: ['ignore', 'pipe', 'ignore'] },
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
  if (process.platform === 'darwin') {
    try {
      const out = execFileSync('system_profiler', ['SPDisplaysDataType'], {
        encoding: 'utf8',
        timeout: 2000,
        maxBuffer: 256 * 1024,
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
  if (process.platform === 'linux') {
    try {
      const out = execFileSync('lspci', ['-mm'], {
        encoding: 'utf8',
        timeout: 1500,
        maxBuffer: 256 * 1024,
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

function enrichGpuInfo(gpu, hostHw) {
  if (!gpu) return null;
  if (!hostHw || gpu.fallback) return gpu;
  const vramLabel =
    hostHw.vramMB && hostHw.vramMB >= 1024 ? `${Math.round(hostHw.vramMB / 1024)} GB` : null;
  return {
    ...gpu,
    name: hostHw.name,
    shortName: hostHw.shortName,
    vramMB: hostHw.vramMB,
    description: vramLabel ? `${hostHw.name} (${vramLabel})` : hostHw.name,
  };
}

function loadPlaywright(repoRoot) {
  const candidates = [
    path.join(repoRoot, 'server', 'package.json'),
    path.join(repoRoot, 'web', 'package.json'),
    path.join(repoRoot, 'package.json'),
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
  throw new Error(
    'Playwright is not installed. Run `npm --prefix server install playwright-core` (and install Chromium) or set DENOVO_PLAYWRIGHT_PKG.',
  );
}

function findChromeExecutable() {
  if (process.env.DENOVO_CHROME_PATH && fs.existsSync(process.env.DENOVO_CHROME_PATH)) {
    return process.env.DENOVO_CHROME_PATH;
  }
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

/** Build a minimal allowlisted env for headless Chromium (never leaking DENOVO_API_TOKEN or secrets, H6). */
function buildMinimalChromiumEnv(vulkanIcd) {
  const allow = [
    'PATH',
    'HOME',
    'USER',
    'LOGNAME',
    'TMPDIR',
    'TEMP',
    'TMP',
    'LANG',
    'LC_ALL',
    'TZ',
    'XDG_RUNTIME_DIR',
    'XDG_CONFIG_HOME',
    'XDG_CACHE_HOME',
    'DISPLAY',
    'WAYLAND_DISPLAY',
    'DBUS_SESSION_BUS_ADDRESS',
    'LD_LIBRARY_PATH',
    'VK_ICD_FILENAMES',
    'VK_DRIVER_FILES',
    '__GLX_VENDOR_LIBRARY_NAME',
    'GBM_BACKEND',
  ];
  const env = {};
  for (const k of allow) {
    if (typeof process.env[k] === 'string' && process.env[k].length > 0) {
      env[k] = process.env[k];
    }
  }
  if (process.platform === 'linux') {
    delete env.DISPLAY;
    delete env.WAYLAND_DISPLAY;
  }
  if (vulkanIcd) {
    env.VK_ICD_FILENAMES = vulkanIcd;
  }
  return env;
}

function sanitizeJsonObject(val, depth = 0) {
  if (depth > 32) {
    throw new Error('Request JSON nesting exceeds maximum depth (32).');
  }
  if (val === null || typeof val !== 'object') return val;
  if (Array.isArray(val)) {
    return val.map((item) => sanitizeJsonObject(item, depth + 1));
  }
  const out = Object.create(null);
  for (const [k, v] of Object.entries(val)) {
    if (FORBIDDEN_KEYS.has(k)) {
      throw new Error(`Forbidden property key "${k}" in request payload.`);
    }
    out[k] = sanitizeJsonObject(v, depth + 1);
  }
  return out;
}

function sendJson(res, status, payload) {
  if (res.writableEnded || res.destroyed) return;
  if (!res.headersSent) {
    res.writeHead(status, {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
      ...(status === 413 || status === 429 ? { Connection: 'close' } : {}),
    });
  }
  res.end(JSON.stringify(payload));
}

function setSecurityHeaders(res, isHtml) {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'no-referrer');
  if (isHtml) {
    res.setHeader('Content-Security-Policy', "frame-ancestors 'none'; base-uri 'self'; object-src 'none'");
  }
}

function canonicalOrigin(raw) {
  try {
    const u = new URL(String(raw).trim());
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
    return u.origin.toLowerCase();
  } catch {
    return null;
  }
}

/**
 * Create the hardened self-hosted HTTP server + job scheduler (testable directly in Node/Vitest
 * with a mock runner bridge, or wired to live headless WebGPU Chromium in CLI mode).
 */
export function createSelfhostServer(options = {}) {
  const repoRoot = path.resolve(options.repoRoot || DEFAULT_REPO_ROOT);
  const buildDir = path.resolve(options.buildDir || DEFAULT_BUILD_DIR);
  const realBuildDir = fs.existsSync(buildDir) ? fs.realpathSync(buildDir) : buildDir;

  const host = String(options.host ?? process.env.HOST ?? '127.0.0.1').trim() || '127.0.0.1';
  const port = parseFiniteInt(options.port ?? process.env.PORT, 8787, 0, 65535);
  const apiToken = String(options.apiToken ?? process.env.DENOVO_API_TOKEN ?? '').trim();
  const allowUnauthenticatedRemote =
    Boolean(options.allowUnauthenticatedRemote) ||
    process.env.DENOVO_ALLOW_UNAUTHENTICATED_REMOTE === '1';

  // H1: Refuse non-loopback bind without Bearer authentication unless explicitly overridden
  if (!isLoopbackHost(host) && !apiToken && !allowUnauthenticatedRemote) {
    throw new Error(
      `Refusing to bind non-loopback host "${host}" without DENOVO_API_TOKEN. Set DENOVO_API_TOKEN or bind HOST=127.0.0.1.`,
    );
  }

  const rawAllowedOrigins = String(
    options.allowedOrigins ?? process.env.DENOVO_ALLOWED_ORIGINS ?? '',
  ).trim();
  const allowedOriginSet = new Set(
    rawAllowedOrigins
      .split(',')
      .map((s) => canonicalOrigin(s))
      .filter(Boolean),
  );

  const allowSoftwareGpu =
    options.allowSoftwareGpu !== undefined
      ? Boolean(options.allowSoftwareGpu)
      : process.env.DENOVO_ALLOW_SOFTWARE_GPU === '1';
  const maxBodyBytes =
    parseFiniteInt(options.maxBodyMB ?? process.env.DENOVO_MAX_BODY_MB, 64, 1, 256) * 1024 * 1024;
  const maxFoldResidues = parseFiniteInt(
    options.maxFoldResidues ?? process.env.DENOVO_MAX_FOLD_RESIDUES,
    2048,
    512,
    4096,
  );
  const maxQueueDepth = parseFiniteInt(
    options.maxQueueDepth ?? process.env.DENOVO_MAX_QUEUE_DEPTH,
    8,
    1,
    64,
  );
  const maxConcurrentUploads = parseFiniteInt(
    options.maxConcurrentUploads ?? process.env.DENOVO_MAX_CONCURRENT_UPLOADS,
    4,
    1,
    16,
  );
  const maxInflightUploadBytes = maxBodyBytes * 2;
  const jobTimeoutMs = parseFiniteInt(
    options.jobTimeoutMs ?? process.env.DENOVO_JOB_TIMEOUT_MS,
    600_000,
    1_000,
    3_600_000,
  );
  const hostGpuHw = options.hostGpuHw !== undefined ? options.hostGpuHw : detectHostGpuHardware();

  // Runner bridge (replaced by real Playwright page in CLI mode or mock bridge in integration tests)
  let runnerBridge = options.runnerBridge ?? null;
  let observedHealth = {
    ready: false,
    gpu: null,
    activeAdapter: null,
    error: 'Starting host GPU runner...',
  };

  /** M8: Separate raw runner health from operator policy; recompute admission on every check/dispatch. */
  function evaluateAdmission(raw) {
    const h = raw || observedHealth;
    if (!h.ready) {
      return { ok: false, error: h.error || 'Host GPU runner is still starting.', gpu: null };
    }
    if (!h.gpu) {
      return { ok: false, error: h.error || 'Host GPU unavailable.', gpu: null };
    }
    if (h.gpu.fallback && !allowSoftwareGpu) {
      return {
        ok: false,
        error:
          'Host Chromium fell back to software WebGPU (SwiftShader). Install Vulkan/GPU drivers or set DENOVO_ALLOW_SOFTWARE_GPU=1 to override.',
        gpu: h.gpu,
      };
    }
    if (h.error) {
      return { ok: false, error: h.error, gpu: h.gpu };
    }
    return { ok: true, error: null, gpu: enrichGpuInfo(h.gpu, hostGpuHw) };
  }

  function extractHostname(hostHeader) {
    const raw = String(hostHeader || '').trim();
    if (!raw) return '';
    if (raw.startsWith('[')) {
      const end = raw.indexOf(']');
      return end > 1 ? raw.slice(1, end).toLowerCase() : '';
    }
    return raw.replace(/:\d+$/, '').toLowerCase();
  }

  function isSameOrigin(originHeader, hostHeader, req) {
    const canon = canonicalOrigin(originHeader);
    if (!canon || !hostHeader) return false;
    try {
      const parsed = new URL(canon);
      if (parsed.host.toLowerCase() !== hostHeader.toLowerCase()) return false;
      const proto = String(req?.headers?.['x-forwarded-proto'] || '').split(',')[0].trim().toLowerCase();
      if ((proto === 'https' || req?.socket?.encrypted) && parsed.protocol !== 'https:') {
        return false;
      }
      // Prevent DNS rebinding on tokenless loopback servers: only literal loopback origins are same-origin
      if (!apiToken && !isLoopbackHost(parsed.hostname)) {
        return false;
      }
      return true;
    } catch {
      return false;
    }
  }

  /** H1: Server-side Origin & loopback-Host admission check (never reflects arbitrary origins). */
  function checkOriginPolicy(req, res) {
    const originHeader = typeof req.headers.origin === 'string' ? req.headers.origin.trim() : '';
    const hostHeader = typeof req.headers.host === 'string' ? req.headers.host.trim() : '';
    // Prevent DNS rebinding when running tokenless on loopback: reject non-loopback Host headers
    if (!apiToken && hostHeader && !isLoopbackHost(extractHostname(hostHeader))) {
      return { allowed: false, grantedOrigin: null };
    }
    if (!originHeader) {
      return { allowed: true, grantedOrigin: null };
    }
    const canon = canonicalOrigin(originHeader);
    if (!canon) {
      return { allowed: false, grantedOrigin: null };
    }
    const sameOrig = isSameOrigin(originHeader, hostHeader, req);
    const explicitlyAllowed = allowedOriginSet.has(canon);
    if (!sameOrig && !explicitlyAllowed) {
      return { allowed: false, grantedOrigin: null };
    }
    res.setHeader('Access-Control-Allow-Origin', originHeader);
    res.setHeader('Vary', 'Origin');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept');
    if (explicitlyAllowed && req.headers['access-control-request-private-network'] === 'true') {
      res.setHeader('Access-Control-Allow-Private-Network', 'true');
    }
    return { allowed: true, grantedOrigin: originHeader };
  }

  /** Derive a non-reversible HMAC session cookie value from apiToken (rotates automatically if apiToken changes). */
  const expectedSessionCookie = apiToken
    ? crypto.createHmac('sha256', apiToken).update('denovo-selfhost-session/v1').digest('hex')
    : '';

  /** Per-client-IP sliding-window failed auth tracker (prevents brute-force on /api/auth). */
  const authFailuresByIp = new Map();

  function getClientIpKey(req) {
    const xff = typeof req.headers['x-forwarded-for'] === 'string' ? req.headers['x-forwarded-for'].split(',')[0].trim() : '';
    return xff || String(req.socket?.remoteAddress || 'unknown');
  }

  function isAuthRateLimited(ipKey, now = Date.now()) {
    const list = authFailuresByIp.get(ipKey);
    if (!list) return false;
    const recent = list.filter((ts) => now - ts < AUTH_RATE_LIMIT_WINDOW_MS);
    if (recent.length === 0) {
      authFailuresByIp.delete(ipKey);
      return false;
    }
    authFailuresByIp.set(ipKey, recent);
    return recent.length >= AUTH_RATE_LIMIT_MAX_FAILURES;
  }

  function recordAuthFailure(ipKey, now = Date.now()) {
    const prev = authFailuresByIp.get(ipKey) || [];
    const recent = prev.filter((ts) => now - ts < AUTH_RATE_LIMIT_WINDOW_MS);
    recent.push(now);
    authFailuresByIp.set(ipKey, recent);
  }

  function parseCookieValue(req, name) {
    const raw = typeof req.headers.cookie === 'string' ? req.headers.cookie : '';
    if (!raw) return '';
    for (const part of raw.split(';')) {
      const eq = part.indexOf('=');
      if (eq <= 0) continue;
      const k = part.slice(0, eq).trim();
      if (k === name) return part.slice(eq + 1).trim();
    }
    return '';
  }

  function shouldUseSecureCookie(req) {
    const proto = String(req.headers['x-forwarded-proto'] || '').split(',')[0].trim().toLowerCase();
    if (proto === 'https' || req.socket?.encrypted) return true;
    const rawHost = typeof req.headers.host === 'string' ? req.headers.host.trim() : '';
    const hostname = rawHost.replace(/:\d+$/, '').replace(/^\[|\]$/g, '');
    return Boolean(hostname) && !isLoopbackHost(hostname);
  }

  function buildSessionCookieHeader(value, maxAgeSec, secure) {
    const parts = [
      `${SESSION_COOKIE_NAME}=${value}`,
      'HttpOnly',
      'SameSite=Strict',
      'Path=/api',
      `Max-Age=${maxAgeSec}`,
    ];
    if (secure) parts.push('Secure');
    return parts.join('; ');
  }

  /** L1: Constant-time Bearer token OR HttpOnly HMAC session cookie verification. */
  function getAuthMethod(req) {
    if (!apiToken) return 'none';
    const auth = String(req.headers.authorization || '');
    if (auth.startsWith('Bearer ')) {
      const provided = auth.slice(7).trim();
      if (constantTimeEquals(provided, apiToken)) return 'bearer';
    }
    const cookieVal = parseCookieValue(req, SESSION_COOKIE_NAME);
    if (cookieVal && constantTimeEquals(cookieVal, expectedSessionCookie)) {
      return 'cookie';
    }
    return null;
  }

  function isAuthorized(req) {
    return getAuthMethod(req) !== null;
  }

  let activeUploads = 0;
  let inflightUploadBytes = 0;

  /** H5: Bounded JSON body reader enforcing concurrent upload & aggregate byte budgets. */
  function readJsonBody(req, limitBytes) {
    return new Promise((resolve, reject) => {
      if (activeUploads >= maxConcurrentUploads) {
        const err = new Error('Too many concurrent uploads.');
        err.statusCode = 429;
        reject(err);
        return;
      }
      const contentLen = Number(req.headers['content-length']);
      if (Number.isFinite(contentLen) && contentLen > limitBytes) {
        const err = new Error(`Payload exceeds limit (${Math.round(limitBytes / 1024 / 1024) || 1} MB)`);
        err.statusCode = 413;
        reject(err);
        return;
      }
      if (Number.isFinite(contentLen) && inflightUploadBytes + contentLen > maxInflightUploadBytes) {
        const err = new Error('Server upload memory budget temporarily full.');
        err.statusCode = 429;
        reject(err);
        return;
      }

      activeUploads++;
      const chunks = [];
      let total = 0;
      let done = false;

      const release = () => {
        if (done) return;
        done = true;
        activeUploads = Math.max(0, activeUploads - 1);
        inflightUploadBytes = Math.max(0, inflightUploadBytes - total);
      };

      req.on('data', (chunk) => {
        if (done) return;
        if (total + chunk.length > limitBytes) {
          release();
          const err = new Error(`Payload exceeds limit (${Math.max(1, Math.round(limitBytes / 1024 / 1024))} MB)`);
          err.statusCode = 413;
          req.pause();
          reject(err);
          return;
        }
        if (inflightUploadBytes + chunk.length > maxInflightUploadBytes) {
          release();
          const err = new Error('Server upload memory budget exceeded.');
          err.statusCode = 429;
          req.pause();
          reject(err);
          return;
        }
        total += chunk.length;
        inflightUploadBytes += chunk.length;
        chunks.push(chunk);
      });
      req.on('end', () => {
        if (done) return;
        try {
          const text = Buffer.concat(chunks).toString('utf8');
          const parsed = text ? JSON.parse(text) : {};
          const safe = sanitizeJsonObject(parsed);
          release();
          resolve(safe);
        } catch (err) {
          release();
          const parseErr = new Error(
            `Invalid JSON body: ${err instanceof Error ? err.message : String(err)}`,
          );
          parseErr.statusCode = 400;
          reject(parseErr);
        }
      });
      req.on('error', (err) => {
        release();
        reject(err);
      });
      req.on('aborted', () => {
        release();
        const err = new Error('Client aborted upload');
        err.statusCode = 400;
        reject(err);
      });
    });
  }

  function sendSse(res, event, data, allowDropIfBackpressured = false) {
    if (res.writableEnded || res.destroyed) return;
    if (allowDropIfBackpressured && res.writableNeedDrain) return;
    try {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    } catch {
      /* client disconnected */
    }
  }

  // ---- FIFO GPU Job Queue with M1 cancel capabilities, M2 res.close finalizer, H5 deadlines ----
  const jobQueue = [];
  let runningJob = null;

  /** M2: Single idempotent finalizer ensuring SSE streams always close cleanly on every terminal path. */
  function finalizeJob(item, event, payload) {
    if (item.finalized) return;
    item.finalized = true;
    if (item.heartbeat) {
      clearInterval(item.heartbeat);
      item.heartbeat = null;
    }
    if (item.deadlineTimer) {
      clearTimeout(item.deadlineTimer);
      item.deadlineTimer = null;
    }
    if (!item.res.writableEnded && !item.res.destroyed) {
      if (event) sendSse(item.res, event, payload, false);
      try {
        item.res.end();
      } catch {
        /* ignore */
      }
    }
  }

  function notifyQueuePositions() {
    for (let i = 0; i < jobQueue.length; i++) {
      const item = jobQueue[i];
      if (!item.finalized) {
        sendSse(
          item.res,
          'progress',
          {
            status: 'loading',
            phase: 'queue',
            progress: 0.01,
            detail: `Queued on host GPU (position #${i + 1})...`,
          },
          true,
        );
      }
    }
  }

  async function refreshObservedHealth() {
    if (!runnerBridge || runningJob) return;
    try {
      const h = await runnerBridge.getHealth();
      if (h) {
        observedHealth = h;
      } else {
        observedHealth = {
          ready: false,
          gpu: null,
          activeAdapter: null,
          error: 'Host GPU runner health unavailable (runner disconnected).',
        };
      }
    } catch (err) {
      observedHealth = {
        ready: false,
        gpu: null,
        activeAdapter: null,
        error: `Host GPU runner unreachable: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  }

  async function pumpQueue() {
    if (runningJob || jobQueue.length === 0) return;
    const next = jobQueue.shift();
    notifyQueuePositions();

    if (next.cancelled || next.finalized || next.res.writableEnded || next.res.destroyed) {
      finalizeJob(next, null, null);
      void pumpQueue();
      return;
    }

    runningJob = next;
    next.heartbeat = setInterval(() => {
      if (!next.res.writableEnded && !next.res.destroyed && !next.res.writableNeedDrain) {
        try {
          next.res.write(': ping\n\n');
        } catch {
          /* ignore */
        }
      }
    }, 15_000);

    try {
      const admission = evaluateAdmission(observedHealth);
      if (!runnerBridge || !admission.ok) {
        throw new Error(admission.error || 'Host GPU runner is unavailable.');
      }

      const timeoutPromise = new Promise((_, reject) => {
        next.deadlineTimer = setTimeout(() => {
          next.timedOut = true;
          next.cancelled = true;
          void runnerBridge?.cancel?.(next.jobId).catch(() => {});
          reject(new Error(`Host GPU job exceeded deadline (${Math.round(jobTimeoutMs / 1000)}s).`));
        }, jobTimeoutMs);
      });

      const result = await Promise.race([
        runnerBridge.runNode(next.jobId, {
          node: next.node,
          inputs: next.inputs,
          maxFoldResidues,
        }),
        timeoutPromise,
      ]);

      if (next.timedOut) {
        finalizeJob(next, 'error', {
          message: `Host GPU job exceeded deadline (${Math.round(jobTimeoutMs / 1000)}s).`,
        });
      } else if (next.cancelled) {
        finalizeJob(next, 'error', { message: 'Workflow run cancelled' });
      } else {
        finalizeJob(next, 'result', result);
      }
    } catch (err) {
      const message = next.timedOut
        ? `Host GPU job exceeded deadline (${Math.round(jobTimeoutMs / 1000)}s).`
        : next.cancelled
          ? 'Workflow run cancelled'
          : err instanceof Error
            ? err.message
            : String(err);
      finalizeJob(next, 'error', { message });
    } finally {
      if (next.heartbeat) {
        clearInterval(next.heartbeat);
        next.heartbeat = null;
      }
      if (next.deadlineTimer) {
        clearTimeout(next.deadlineTimer);
        next.deadlineTimer = null;
      }
      if (runningJob === next) runningJob = null;
      await refreshObservedHealth();
      void pumpQueue();
    }
  }

  async function cancelJobAuthorized(jobId, cancelToken) {
    const qIdx = jobQueue.findIndex((j) => j.jobId === jobId);
    if (qIdx !== -1) {
      const item = jobQueue[qIdx];
      if (!constantTimeEquals(cancelToken, item.cancelToken)) {
        return { ok: false, status: 403, error: 'Invalid cancellation token for job.' };
      }
      jobQueue.splice(qIdx, 1);
      item.cancelled = true;
      finalizeJob(item, 'error', { message: 'Workflow run cancelled' });
      notifyQueuePositions();
      return { ok: true, status: 200, cancelled: true };
    }
    if (runningJob && runningJob.jobId === jobId) {
      if (!constantTimeEquals(cancelToken, runningJob.cancelToken)) {
        return { ok: false, status: 403, error: 'Invalid cancellation token for job.' };
      }
      const target = runningJob;
      target.cancelled = true;
      try {
        await runnerBridge?.cancel?.(jobId);
      } catch {
        /* ignore */
      }
      finalizeJob(target, 'error', { message: 'Workflow run cancelled' });
      return { ok: true, status: 200, cancelled: true };
    }
    return { ok: true, status: 200, cancelled: false };
  }

  function handleRunnerEmit(ev) {
    if (!ev || typeof ev !== 'object') return;
    if (ev.type === 'runner-ready') {
      observedHealth = {
        ready: true,
        gpu: ev.gpu || null,
        activeAdapter: null,
        error: ev.error || null,
      };
      return;
    }
    if (ev.type === 'runner-error') {
      observedHealth = {
        ready: false,
        gpu: null,
        activeAdapter: null,
        error: typeof ev.error === 'string' && ev.error ? ev.error : 'Host GPU runner crashed or closed.',
      };
      return;
    }
    if (ev.type === 'progress' && runningJob && ev.jobId === runningJob.jobId && !runningJob.cancelled) {
      sendSse(
        runningJob.res,
        'progress',
        {
          status: ev.status,
          phase: ev.phase,
          progress: ev.progress,
          detail: ev.detail,
        },
        true,
      );
    }
  }

  /** H3: Canonical realpath + path.relative containment with symlink escape protection. */
  function resolveSafeStaticFile(rawPathname) {
    let decoded;
    try {
      decoded = decodeURIComponent(rawPathname);
    } catch {
      return { error: 400 };
    }
    if (decoded.includes('\0')) return { error: 400 };
    const cleanPath = decoded === '/' || decoded === '' ? '/index.html' : decoded;
    const relRequested = cleanPath.replace(/^\/+/, '');

    const candidates = [
      path.resolve(realBuildDir, relRequested),
      path.resolve(realBuildDir, `${relRequested}.html`),
      path.resolve(realBuildDir, relRequested, 'index.html'),
    ];
    if (!path.extname(cleanPath)) {
      candidates.push(path.resolve(realBuildDir, 'index.html'));
    }

    for (const cand of candidates) {
      let realCand;
      let stat;
      try {
        realCand = fs.realpathSync(cand);
        stat = fs.statSync(realCand);
      } catch {
        continue;
      }
      if (!stat.isFile()) continue;
      const rel = path.relative(realBuildDir, realCand);
      if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) {
        continue;
      }
      return { filePath: realCand };
    }
    return { error: 404 };
  }

  function serveStaticFile(req, res, pathname) {
    // Restrict internal /runner harness strictly to direct loopback Chromium (never external visitors)
    const lowerPath = pathname.toLowerCase();
    if (
      (lowerPath === '/runner' || lowerPath === '/runner.html' || lowerPath.startsWith('/runner/')) &&
      (!isLoopbackRemoteAddress(req.socket?.remoteAddress) || Boolean(req.headers['x-forwarded-for']))
    ) {
      res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8', 'X-Content-Type-Options': 'nosniff' });
      res.end('Forbidden');
      return;
    }

    const resolved = resolveSafeStaticFile(pathname);
    if (resolved.error) {
      res.writeHead(resolved.error, {
        'Content-Type': 'text/plain; charset=utf-8',
        'X-Content-Type-Options': 'nosniff',
      });
      res.end(resolved.error === 400 ? 'Bad Request' : 'Not Found');
      return;
    }

    const target = resolved.filePath;
    const ext = path.extname(target).toLowerCase();
    const isHtml = ext === '.html';
    const stream = fs.createReadStream(target);

    stream.on('open', () => {
      if (res.writableEnded || res.destroyed) {
        stream.destroy();
        return;
      }
      setSecurityHeaders(res, isHtml);
      res.writeHead(200, {
        'Content-Type': MIME_TYPES[ext] || 'application/octet-stream',
        'Cache-Control': target.includes(`${path.sep}_app${path.sep}immutable${path.sep}`)
          ? 'public, max-age=31536000, immutable'
          : 'no-cache',
      });
      stream.pipe(res);
    });
    stream.on('error', () => {
      if (!res.headersSent) {
        res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Static file read error');
      } else {
        res.destroy();
      }
    });
    res.on('close', () => {
      if (!stream.destroyed) stream.destroy();
    });
  }

  const server = http.createServer((req, res) => {
    void (async () => {
      // H1/H2: Validate Host header syntax & parse request target against fixed internal base
      const rawHost = typeof req.headers.host === 'string' ? req.headers.host.trim() : '';
      if (rawHost && !VALID_HOST_HEADER_RE.test(rawHost)) {
        sendJson(res, 400, { ok: false, error: 'Malformed Host header.' });
        return;
      }

      let url;
      try {
        url = new URL(req.url || '/', 'http://selfhost.internal');
      } catch {
        sendJson(res, 400, { ok: false, error: 'Malformed request URL.' });
        return;
      }

      if (url.pathname.startsWith('/api/')) {
        const originCheck = checkOriginPolicy(req, res);
        if (!originCheck.allowed) {
          sendJson(res, 403, { ok: false, error: 'Forbidden origin.' });
          return;
        }

        if (req.method === 'OPTIONS') {
          res.writeHead(204);
          res.end();
          return;
        }

        if (req.method === 'POST') {
          const ct = String(req.headers['content-type'] || '').toLowerCase();
          if (!ct.includes('application/json')) {
            sendJson(res, 415, { ok: false, error: 'Content-Type must be application/json.' });
            return;
          }
        }

        if (req.method === 'POST' && url.pathname === '/api/auth') {
          const ipKey = getClientIpKey(req);
          if (isAuthRateLimited(ipKey)) {
            res.setHeader('Retry-After', '60');
            sendJson(res, 429, {
              ok: false,
              error: 'Too many failed authentication attempts. Wait 60 seconds and try again.',
            });
            return;
          }
          let body;
          try {
            body = await readJsonBody(req, MAX_CONTROL_BODY_BYTES);
          } catch (err) {
            const code = typeof err?.statusCode === 'number' ? err.statusCode : 400;
            sendJson(res, code, { ok: false, error: err instanceof Error ? err.message : String(err) });
            return;
          }
          if (!apiToken) {
            sendJson(res, 200, { ok: true, authRequired: false, authMode: 'none' });
            return;
          }
          const bearer = String(req.headers.authorization || '').startsWith('Bearer ')
            ? String(req.headers.authorization || '').slice(7).trim()
            : '';
          const provided = typeof body?.token === 'string' && body.token.trim() ? body.token.trim() : bearer;
          if (!constantTimeEquals(provided, apiToken)) {
            recordAuthFailure(ipKey);
            sendJson(res, 401, { ok: false, error: 'Unauthorized: invalid API token.' });
            return;
          }
          authFailuresByIp.delete(ipKey);
          const secure = shouldUseSecureCookie(req);
          res.setHeader('Set-Cookie', buildSessionCookieHeader(expectedSessionCookie, SESSION_COOKIE_MAX_AGE_SEC, secure));
          sendJson(res, 200, { ok: true, authRequired: true, authMode: 'cookie' });
          return;
        }

        if (req.method === 'POST' && url.pathname === '/api/logout') {
          try {
            await readJsonBody(req, MAX_CONTROL_BODY_BYTES);
          } catch {
            /* ignore empty/invalid logout body */
          }
          const secure = shouldUseSecureCookie(req);
          res.setHeader('Set-Cookie', buildSessionCookieHeader('', 0, secure));
          sendJson(res, 200, { ok: true });
          return;
        }

        const authMethod = getAuthMethod(req);
        if (!authMethod) {
          sendJson(res, 401, { ok: false, error: 'Unauthorized: invalid or missing Bearer token.' });
          return;
        }

        if (req.method === 'GET' && url.pathname === '/api/health') {
          await refreshObservedHealth();
          const admission = evaluateAdmission(observedHealth);
          // M1: Never leak runningJobId in shared /api/health
          sendJson(res, 200, {
            ok: admission.ok,
            service: 'denovo-selfhost',
            version: '0.2.0',
            gpu: admission.gpu,
            activeAdapter: observedHealth.activeAdapter ?? null,
            queueDepth: jobQueue.length + (runningJob ? 1 : 0),
            busy: Boolean(runningJob),
            maxFoldResidues,
            authRequired: Boolean(apiToken),
            authenticatedBy: authMethod,
            ...(admission.error ? { error: admission.error } : {}),
          });
          return;
        }

        if (req.method === 'POST' && url.pathname === '/api/cancel') {
          try {
            const body = await readJsonBody(req, MAX_CONTROL_BODY_BYTES);
            const jobId = String(body.jobId || '').trim();
            const cancelToken = String(body.cancelToken || '').trim();
            if (!jobId || !cancelToken) {
              sendJson(res, 400, { ok: false, error: 'Both jobId and cancelToken are required.' });
              return;
            }
            const out = await cancelJobAuthorized(jobId, cancelToken);
            sendJson(res, out.status, out.ok ? { ok: true, cancelled: out.cancelled } : { ok: false, error: out.error });
          } catch (err) {
            const code = typeof err?.statusCode === 'number' ? err.statusCode : 400;
            sendJson(res, code, { ok: false, error: err instanceof Error ? err.message : String(err) });
          }
          return;
        }

        if (req.method === 'POST' && url.pathname === '/api/release-gpu') {
          try {
            await readJsonBody(req, MAX_CONTROL_BODY_BYTES);
            if (runningJob) {
              sendJson(res, 409, { ok: false, error: 'Host GPU is currently busy running a job.' });
              return;
            }
            if (runnerBridge) {
              await runnerBridge.releaseGpu();
              observedHealth = { ...observedHealth, activeAdapter: null };
            }
            sendJson(res, 200, { ok: true });
          } catch (err) {
            const code = typeof err?.statusCode === 'number' ? err.statusCode : 500;
            sendJson(res, code, { ok: false, error: err instanceof Error ? err.message : String(err) });
          }
          return;
        }

        if (req.method === 'POST' && url.pathname === '/api/run-node') {
          // H5: Reserve queue capacity BEFORE reading request body
          if (jobQueue.length >= maxQueueDepth) {
            sendJson(res, 429, {
              ok: false,
              error: `Host GPU queue is full (max ${maxQueueDepth} queued jobs). Try again shortly.`,
            });
            return;
          }

          let body;
          try {
            body = await readJsonBody(req, maxBodyBytes);
          } catch (err) {
            const code = typeof err?.statusCode === 'number' ? err.statusCode : 400;
            sendJson(res, code, { ok: false, error: err instanceof Error ? err.message : String(err) });
            return;
          }

          const node = body?.node;
          const adapterId = typeof node?.adapterId === 'string' ? node.adapterId.trim() : '';
          // H4: Reject unknown or CPU adapters before queueing
          if (!adapterId || !ALLOWED_GPU_ADAPTER_IDS.has(adapterId)) {
            sendJson(res, 400, {
              ok: false,
              error: `Unsupported or disallowed GPU adapter "${adapterId || '(missing)'}".`,
            });
            return;
          }
          if (node.params !== undefined && (typeof node.params !== 'object' || node.params === null || Array.isArray(node.params))) {
            sendJson(res, 400, { ok: false, error: 'node.params must be a JSON object.' });
            return;
          }
          if (body.inputs !== undefined && (typeof body.inputs !== 'object' || body.inputs === null || Array.isArray(body.inputs))) {
            sendJson(res, 400, { ok: false, error: 'inputs must be a JSON object.' });
            return;
          }

          // Re-check queue depth after async body read
          if (jobQueue.length >= maxQueueDepth) {
            sendJson(res, 429, {
              ok: false,
              error: `Host GPU queue is full (max ${maxQueueDepth} queued jobs).`,
            });
            return;
          }

          // M1: Server-generated jobId & random cancel capability token
          const jobId = `job_${crypto.randomBytes(12).toString('hex')}`;
          const cancelToken = crypto.randomBytes(18).toString('hex');

          res.writeHead(200, {
            'Content-Type': 'text/event-stream; charset=utf-8',
            'Cache-Control': 'no-cache, no-transform',
            Connection: 'keep-alive',
            'X-Accel-Buffering': 'no',
            'X-Content-Type-Options': 'nosniff',
          });

          const item = {
            jobId,
            cancelToken,
            node: {
              id: typeof node.id === 'string' && node.id.length <= 128 ? node.id : jobId,
              adapterId,
              ...(typeof node.label === 'string' ? { label: node.label.slice(0, 128) } : {}),
              params: node.params ?? {},
            },
            inputs: body.inputs ?? {},
            res,
            cancelled: false,
            finalized: false,
            heartbeat: null,
            deadlineTimer: null,
          };

          // Emit server-assigned jobId + cancelToken immediately so only this owner can cancel
          sendSse(res, 'started', { jobId, cancelToken }, false);

          // M2: Listen on res.close (not req.close) to detect SSE client disconnects accurately
          res.on('close', () => {
            if (!item.finalized && !res.writableEnded) {
              void cancelJobAuthorized(jobId, cancelToken);
            }
          });

          jobQueue.push(item);
          if (runningJob) {
            notifyQueuePositions();
          }
          void pumpQueue();
          return;
        }

        sendJson(res, 404, { ok: false, error: 'Unknown API endpoint.' });
        return;
      }

      serveStaticFile(req, res, url.pathname);
    })().catch((err) => {
      // H2: Never allow unhandled rejections to crash the Node server
      console.error('[denovo-selfhost] Request handler error:', err);
      if (!res.headersSent) {
        sendJson(res, 500, { ok: false, error: 'Internal server error.' });
      } else if (!res.destroyed) {
        res.destroy();
      }
    });
  });

  // H5: Bound HTTP header/keep-alive timeouts against slowloris clients
  server.headersTimeout = 15_000;
  server.keepAliveTimeout = 65_000;

  return {
    server,
    config: {
      host,
      port,
      apiTokenEnabled: Boolean(apiToken),
      allowedOrigins: [...allowedOriginSet],
      maxBodyBytes,
      maxFoldResidues,
      maxQueueDepth,
      jobTimeoutMs,
      allowSoftwareGpu,
      realBuildDir,
    },
    setRunnerBridge(bridge, initialHealth) {
      runnerBridge = bridge;
      if (initialHealth) observedHealth = initialHealth;
    },
    handleRunnerEmit,
    getState() {
      return {
        queueDepth: jobQueue.length,
        runningJobId: runningJob?.jobId ?? null,
        activeUploads,
        inflightUploadBytes,
      };
    },
  };
}

async function startPlaywrightRunner(localOrigin, { repoRoot, dataDir, disableSandbox, onEmit }) {
  const { chromium } = loadPlaywright(repoRoot);
  const executablePath = findChromeExecutable();
  fs.mkdirSync(dataDir, { recursive: true });

  // H6: Only clear stale lockfiles if not actively owned by a live process; use a lockfile guard
  const serverLockPath = path.join(dataDir, '.denovo-server.pid');
  if (fs.existsSync(serverLockPath)) {
    const prevPid = Number(fs.readFileSync(serverLockPath, 'utf8').trim());
    if (Number.isInteger(prevPid) && prevPid > 0 && prevPid !== process.pid) {
      let alive = false;
      try {
        process.kill(prevPid, 0);
        alive = true;
      } catch {
        alive = false;
      }
      if (alive) {
        throw new Error(
          `Another denovo-selfhost instance (PID ${prevPid}) is already using DENOVO_DATA_DIR="${dataDir}".`,
        );
      }
    }
  }
  fs.writeFileSync(serverLockPath, `${process.pid}\n`, { mode: 0o600 });

  for (const lf of ['SingletonLock', 'SingletonSocket', 'SingletonCookie']) {
    try {
      fs.rmSync(path.join(dataDir, lf), { force: true });
    } catch {
      /* ignore */
    }
  }

  const vulkanIcd = detectLinuxVulkanIcd();
  const minimalEnv = buildMinimalChromiumEnv(vulkanIcd);
  const disableGpuSandbox = disableSandbox || process.env.DENOVO_DISABLE_GPU_SANDBOX === '1';

  const launchArgs = [
    '--headless=new',
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    '--disable-dev-shm-usage',
  ];
  if (process.platform === 'linux') {
    launchArgs.push(
      '--ozone-platform=headless',
      '--enable-features=Vulkan',
      '--use-angle=vulkan',
      '--use-vulkan=native',
      '--disable-vulkan-surface',
    );
  }
  if (disableSandbox) {
    launchArgs.push('--no-sandbox');
  }
  if (disableGpuSandbox) {
    launchArgs.push('--disable-gpu-sandbox');
  }

  console.log(
    `[denovo-selfhost] Launching host WebGPU Chromium (${executablePath || 'bundled'}, sandbox=${!disableSandbox}, gpuSandbox=${!disableGpuSandbox})...`,
  );
  if (vulkanIcd) {
    console.log(`[denovo-selfhost] Vulkan ICD: ${vulkanIcd}`);
  }

  let context;
  try {
    context = await chromium.launchPersistentContext(dataDir, {
      headless: false,
      chromiumSandbox: !disableSandbox,
      ...(executablePath ? { executablePath } : {}),
      args: launchArgs,
      env: minimalEnv,
      viewport: { width: 800, height: 600 },
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (/No usable sandbox|SUID sandbox|AppArmor|zygote/i.test(msg) && !disableSandbox) {
      throw new Error(
        `Chromium sandbox failed to start (common on Ubuntu 24.04 AppArmor or containers). ` +
          `Retry with DENOVO_CHROME_PATH=/usr/bin/google-chrome DENOVO_DISABLE_GPU_SANDBOX=1 ` +
          `or DENOVO_DISABLE_SANDBOX=1.\nOriginal error: ${msg}`,
      );
    }
    throw err;
  }

  const page = context.pages()[0] || (await context.newPage());
  await page.exposeFunction('__denovoHostEmit', onEmit);

  page.on('crash', () => {
    console.error('[denovo-selfhost] Host Chromium runner page crashed.');
    onEmit({ type: 'runner-error', error: 'Host Chromium runner page crashed.' });
  });
  page.on('close', () => {
    onEmit({ type: 'runner-error', error: 'Host Chromium runner page closed.' });
  });
  context.on('close', () => {
    onEmit({ type: 'runner-error', error: 'Host Chromium browser context closed.' });
  });

  page.on('console', (msg) => {
    const t = msg.type();
    if (t === 'error' || t === 'warning') {
      console.log(`[host-chromium:${t}] ${msg.text()}`);
    }
  });
  page.on('pageerror', (err) => {
    console.error(`[host-chromium:pageerror] ${err.message}`);
  });

  await page.goto(`${localOrigin}/runner`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => Boolean(window.__denovoHostRunner?.getHealth()?.ready), {
    timeout: 30_000,
  });

  const initialHealth = await page.evaluate(() => window.__denovoHostRunner.getHealth());
  const bridge = {
    getHealth: () => page.evaluate(() => window.__denovoHostRunner?.getHealth?.() ?? null),
    runNode: (jobId, payload) =>
      page.evaluate(
        async ({ jobId: id, node, inputs, maxFoldResidues }) => {
          if (!window.__denovoHostRunner) {
            throw new Error('Host runner bridge not mounted.');
          }
          return await window.__denovoHostRunner.runNode(id, { node, inputs, maxFoldResidues });
        },
        { jobId, ...payload },
      ),
    cancel: (jobId) => page.evaluate((id) => window.__denovoHostRunner?.cancel(id), jobId),
    releaseGpu: () => page.evaluate(() => window.__denovoHostRunner?.releaseGpu?.()),
  };

  return {
    context,
    bridge,
    initialHealth,
    cleanupLock: () => {
      try {
        fs.rmSync(serverLockPath, { force: true });
      } catch {
        /* ignore */
      }
    },
  };
}

async function main() {
  if (!fs.existsSync(path.join(DEFAULT_BUILD_DIR, 'index.html'))) {
    console.error('[denovo-selfhost] Missing web/build/index.html. Run "npm --prefix web run build" first.');
    process.exit(1);
  }

  const dataDir = path.resolve(
    process.env.DENOVO_DATA_DIR || path.join(os.homedir(), '.cache', 'denovo-selfhost', 'profile'),
  );
  const disableSandbox = process.env.DENOVO_DISABLE_SANDBOX === '1';

  const instance = createSelfhostServer({
    repoRoot: DEFAULT_REPO_ROOT,
    buildDir: DEFAULT_BUILD_DIR,
  });

  await new Promise((resolve) =>
    instance.server.listen(instance.config.port, instance.config.host, resolve),
  );
  const addr = instance.server.address();
  const actualPort = typeof addr === 'object' && addr ? addr.port : instance.config.port;
  const localOrigin = `http://127.0.0.1:${actualPort}`;

  console.log(
    `[denovo-selfhost] Listening on http://${instance.config.host}:${actualPort} (serving ${instance.config.realBuildDir})`,
  );
  console.log(`[denovo-selfhost] Persistent weight cache: ${dataDir}`);
  console.log(
    `[denovo-selfhost] Bearer API token auth: ${instance.config.apiTokenEnabled ? 'ENABLED' : 'DISABLED (loopback only)'}`,
  );
  if (instance.config.allowedOrigins.length > 0) {
    console.log(`[denovo-selfhost] Allowed CORS origins: ${instance.config.allowedOrigins.join(', ')}`);
  }

  const runner = await startPlaywrightRunner(localOrigin, {
    repoRoot: DEFAULT_REPO_ROOT,
    dataDir,
    disableSandbox,
    onEmit: instance.handleRunnerEmit,
  });
  instance.setRunnerBridge(runner.bridge, runner.initialHealth);

  if (runner.initialHealth?.gpu?.fallback && !instance.config.allowSoftwareGpu) {
    console.warn('[denovo-selfhost] WARNING: Host Chromium fell back to software WebGPU (SwiftShader).');
  } else if (runner.initialHealth?.gpu) {
    const hw = detectHostGpuHardware();
    const enriched = enrichGpuInfo(runner.initialHealth.gpu, hw);
    console.log(
      `[denovo-selfhost] Host GPU ready: ${enriched.description} (fallback=${enriched.fallback})`,
    );
  }

  const shutdown = async () => {
    console.log('\n[denovo-selfhost] Shutting down...');
    try {
      await runner.context?.close();
    } catch {
      /* ignore */
    }
    runner.cleanupLock();
    instance.server.close(() => process.exit(0));
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

const isMainEntry =
  process.argv[1] && path.resolve(process.argv[1]) === path.resolve(__filename);
if (isMainEntry) {
  main().catch((err) => {
    console.error('[denovo-selfhost] Fatal startup error:', err);
    process.exit(1);
  });
}
