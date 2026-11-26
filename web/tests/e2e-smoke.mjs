#!/usr/bin/env node
/**
 * Real-browser E2E smoke suite for denovo.run (Chromium & WebKit).
 * Boots `vite preview` against `web/build` and verifies:
 *  1. Clean SPA hydration with 0 pageerrors and 0 failed same-origin asset loads
 *  2. Welcome modal renders & dismisses cleanly
 *  3. Straight-edge click selection off-center (Issue #104 Safari SVG hit-test regression)
 *  4. Canvas panning works even with Shift held, never drawing .svelte-flow__selection (Issue #104)
 *  5. `?flow=` URL hydration + Model Card Recommended Operating Envelope + 1-click Reset recovery
 *  6. Unsafe `?flow=` step count degrades gracefully with 0 uncaught errors
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createServer } from 'node:net';
import { resolve } from 'node:path';
import { chromium, webkit } from 'playwright';

const WEB_DIR = resolve(import.meta.dirname, '..');
const BUILD_INDEX = resolve(WEB_DIR, 'build', 'index.html');

if (!existsSync(BUILD_INDEX)) {
  console.error('[e2e] Missing web/build/index.html — run `npm run build` first.');
  process.exit(1);
}

async function getFreePort() {
  return new Promise((resolvePort, reject) => {
    const srv = createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = typeof addr === 'object' && addr ? addr.port : 4185;
      srv.close(() => resolvePort(port));
    });
  });
}

async function waitForServer(url, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      /* server starting */
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`Preview server did not become ready at ${url} within ${timeoutMs}ms`);
}

function assert(cond, msg) {
  if (!cond) throw new Error(`Assertion failed: ${msg}`);
}

const BROWSER_TYPES = { chromium, webkit };
const requestedBrowsers = (process.env.E2E_BROWSERS || 'chromium,webkit')
  .split(',')
  .map((s) => s.trim().toLowerCase())
  .filter(Boolean);

for (const name of requestedBrowsers) {
  if (!BROWSER_TYPES[name]) {
    console.error(`[e2e] Unknown browser "${name}" (supported: chromium, webkit)`);
    process.exit(1);
  }
}

async function runSuiteForBrowser(browserName, baseUrl) {
  const launchOpts = { headless: true };
  if (browserName === 'chromium' && process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH) {
    launchOpts.executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  }
  if (browserName === 'webkit' && process.env.PLAYWRIGHT_WEBKIT_EXECUTABLE_PATH) {
    launchOpts.executablePath = process.env.PLAYWRIGHT_WEBKIT_EXECUTABLE_PATH;
  }

  const browser = await BROWSER_TYPES[browserName].launch(launchOpts);
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    const page = await context.newPage();
    page.setDefaultTimeout(15000);

    const pageErrors = [];
    const failedAssets = [];
    page.on('pageerror', (err) => pageErrors.push(String(err)));
    page.on('response', (res) => {
      if (res.url().startsWith(baseUrl) && res.status() >= 400 && !res.url().includes('/api/')) {
        failedAssets.push(`${res.status()} ${res.url()}`);
      }
    });

    // 1. Clean SPA boot & welcome modal
    console.log(`[e2e] (${browserName}) 1/5 boot & welcome modal...`);
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 20000 });
    await page.waitForSelector('.svelte-flow__edge', { timeout: 15000 });
    await page.waitForSelector('.onb-backdrop', { timeout: 10000 });
    await page.waitForTimeout(300);

    const welcomeText = await page.evaluate(
      () => document.querySelector('.onb-backdrop')?.textContent ?? '',
    );
    assert(
      welcomeText.includes('Prototype protein design, prediction, and docking workflows'),
      `[${browserName}] Welcome modal missing expected subtitle`,
    );

    // Dismiss welcome modal via the explicit "Skip & explore" button
    await page.locator('.onb-backdrop .btn-skip').click();
    await page.waitForSelector('.onb-backdrop', { state: 'detached', timeout: 5000 });
    await page.waitForTimeout(250);

    // 2. Straight-edge click selection off-center (Issue #104 Safari SVG hit-test regression)
    console.log(`[e2e] (${browserName}) 2/5 straight-edge click selection...`);
    const straightEdge = await page.evaluate(() => {
      const edges = [...document.querySelectorAll('.svelte-flow__edge')];
      for (const g of edges) {
        const rectEl = g.querySelector('.smart-edge-hit-rect');
        if (!rectEl) continue;
        const r = rectEl.getBoundingClientRect();
        if (r.width >= 10 && r.height >= 10) {
          return {
            id: g.getAttribute('data-id'),
            cx: r.left + r.width / 2,
            cy: r.top + r.height / 2,
          };
        }
      }
      return null;
    });
    assert(straightEdge !== null, `[${browserName}] Expected at least one straight edge with .smart-edge-hit-rect`);

    await page.mouse.click(straightEdge.cx, straightEdge.cy + 6);
    await page.waitForTimeout(250);
    const edgeSelection = await page.evaluate((id) => {
      const g = document.querySelector(`.svelte-flow__edge[data-id="${CSS.escape(id)}"]`);
      return {
        selected: Boolean(g?.classList.contains('selected')),
        hasDeleteBtn: Boolean(document.querySelector('.edge-delete-btn')),
      };
    }, straightEdge.id);
    assert(
      edgeSelection.selected && edgeSelection.hasDeleteBtn,
      `[${browserName}] Straight edge click (+6px) failed: ${JSON.stringify(edgeSelection)}`,
    );

    // 3. Canvas pan drag with Shift held (Issue #104 stuck marquee selection regression)
    console.log(`[e2e] (${browserName}) 3/5 canvas pan with Shift held...`);
    await page.mouse.click(120, 680);
    await page.waitForTimeout(150);
    const vpBefore = await page.evaluate(
      () => document.querySelector('.svelte-flow__viewport')?.getAttribute('style') ?? '',
    );
    await page.keyboard.down('Shift');
    await page.mouse.move(260, 620);
    await page.mouse.down();
    await page.mouse.move(420, 560, { steps: 10 });
    const selectionBoxVisible = await page.evaluate(() =>
      Boolean(document.querySelector('.svelte-flow__selection')),
    );
    await page.mouse.up();
    await page.keyboard.up('Shift');
    const vpAfter = await page.evaluate(
      () => document.querySelector('.svelte-flow__viewport')?.getAttribute('style') ?? '',
    );
    assert(!selectionBoxVisible, `[${browserName}] Marquee .svelte-flow__selection appeared during Shift+drag`);
    assert(vpBefore !== vpAfter, `[${browserName}] Canvas failed to pan during Shift+drag (${vpBefore})`);

    // 4. ?flow= hydration + Model Card Recommended Operating Envelope + Inspector 1-click Reset
    console.log(`[e2e] (${browserName}) 4/5 ?flow= hydration + Model Card envelope + reset...`);
    const flowPage = await context.newPage();
    flowPage.setDefaultTimeout(15000);
    flowPage.on('pageerror', (err) => pageErrors.push(String(err)));
    flowPage.on('response', (res) => {
      if (res.url().startsWith(baseUrl) && res.status() >= 400 && !res.url().includes('/api/')) {
        failedAssets.push(`${res.status()} ${res.url()}`);
      }
    });
    await flowPage.goto(`${baseUrl}?flow=laproteina:mode=sde,steps=250~mpnn~af2`, {
      waitUntil: 'domcontentloaded',
      timeout: 20000,
    });
    await flowPage.waitForSelector('.svelte-flow__node', { timeout: 15000 });
    await flowPage.waitForFunction(
      () => document.querySelectorAll('.svelte-flow__node').length === 3,
      { timeout: 10000 },
    );
    // Dismiss welcome modal if shown on fresh navigation
    if (await flowPage.locator('.onb-backdrop .btn-skip').count()) {
      await flowPage.locator('.onb-backdrop .btn-skip').click();
      await flowPage.waitForSelector('.onb-backdrop', { state: 'detached', timeout: 5000 });
    }
    const flowNodeCount = await flowPage.evaluate(
      () => document.querySelectorAll('.svelte-flow__node').length,
    );
    assert(flowNodeCount === 3, `[${browserName}] Expected 3 nodes from ?flow=, got ${flowNodeCount}`);

    // Select La-Proteina node (first node) and inspect Model Card pill + modal
    await flowPage.locator('.svelte-flow__node').first().click();
    await flowPage.waitForTimeout(250);
    const mcPill = flowPage.locator('.mc-pill').first();
    assert((await mcPill.count()) > 0, `[${browserName}] Expected Model Card pill in Inspector`);
    await mcPill.click();
    await flowPage.waitForSelector('.mc-modal', { timeout: 5000 });
    const modalText = await flowPage.evaluate(
      () => document.querySelector('.mc-modal')?.textContent ?? '',
    );
    assert(
      modalText.includes('Recommended Operating Envelope') && modalText.includes('BELOW RECOMMENDED'),
      `[${browserName}] Model Card modal missing Operating Envelope or BELOW RECOMMENDED banner`,
    );
    await flowPage.keyboard.press('Escape');
    await flowPage.waitForSelector('.mc-modal', { state: 'detached', timeout: 5000 });

    // Click 1-click Reset to 400 in Inspector
    const resetBtn = flowPage.locator('.warn-fix-btn').first();
    assert((await resetBtn.count()) > 0, `[${browserName}] Expected .warn-fix-btn for steps=250 in SDE mode`);
    await resetBtn.click();
    await flowPage.waitForTimeout(200);
    const stepsVal = await flowPage.locator('#param-steps').inputValue();
    assert(stepsVal === '400', `[${browserName}] Expected Reset button to set steps=400, got ${stepsVal}`);
    await flowPage.close();

    // 5. Unsafe ?flow= URL degrades gracefully without crashing
    console.log(`[e2e] (${browserName}) 5/5 unsafe ?flow= graceful handling...`);
    const badFlowPage = await context.newPage();
    badFlowPage.setDefaultTimeout(15000);
    badFlowPage.on('pageerror', (err) => pageErrors.push(String(err)));
    await badFlowPage.goto(`${baseUrl}?flow=boltzgen:numSteps=50`, {
      waitUntil: 'domcontentloaded',
      timeout: 20000,
    });
    await badFlowPage.waitForSelector('.svelte-flow__node', { timeout: 15000 });
    await badFlowPage.waitForTimeout(300);
    await badFlowPage.close();

    assert(pageErrors.length === 0, `[${browserName}] Uncaught pageerrors: ${pageErrors.join(' | ')}`);
    assert(failedAssets.length === 0, `[${browserName}] Failed same-origin assets: ${failedAssets.join(' | ')}`);

    console.log(`[e2e] PASS (${browserName}): boot, straight-edge click, Shift+drag pan, ?flow= + Model Card envelope + reset`);
  } finally {
    await browser.close();
  }
}

const port = await getFreePort();
const baseUrl = `http://127.0.0.1:${port}/`;
const viteBin = resolve(WEB_DIR, 'node_modules', 'vite', 'bin', 'vite.js');
const previewProc = spawn(
  process.execPath,
  [viteBin, 'preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
  { cwd: WEB_DIR, stdio: ['ignore', 'pipe', 'pipe'] },
);

let previewLogs = '';
previewProc.stdout.on('data', (d) => (previewLogs += String(d)));
previewProc.stderr.on('data', (d) => (previewLogs += String(d)));

let exitCode = 0;
try {
  await waitForServer(baseUrl);
  for (const name of requestedBrowsers) {
    await runSuiteForBrowser(name, baseUrl);
  }
} catch (err) {
  console.error('[e2e] FAILED:', err);
  if (previewLogs.trim()) {
    console.error('[e2e] vite preview output:\n' + previewLogs.slice(-1500));
  }
  exitCode = 1;
} finally {
  previewProc.kill('SIGTERM');
  process.exit(exitCode);
}
