import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  BUILD_DIR,
  TOOLS_DIR,
  WEB_DIR,
  assertMarkdownAutolinkerSafe,
  assertValidOgPngBytes,
  cfWorker,
  createMockCfEnv,
  encodeGraphToFlowQuery,
  extractWorkflowExampleUrls,
  generateLiveLlmsTxt,
  graphUtilsNs,
  ogGraphNs,
  registryNs,
  shareNs,
} from './helpers';

describe('Tier 3 — Pairwise Cross-Feature Combinations (14 cases)', () => {
  it('T3-01 (F3 x F4): multi-branch ?flow= pdb-input -> rfd3 & rfd4-proteina -> structure-compare auto-wires structures[]', async () => {
    const flow = 'src:p:mode=demo-bundle;src~b3:r3:length=80~cmp:c;src~b4:r:length=80~cmp';
    const g = await shareNs.decodeHashToGraph(`?flow=${flow}`);
    expect(g).not.toBeNull();
    const cmpEdges = g!.edges.filter((e) => e.toNode === 'cmp');
    expect(cmpEdges).toHaveLength(2);
    expect(cmpEdges.every((e) => e.toPort === 'structures')).toBe(true);
    expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
  });

  it('T3-02 (F3 x F5): preset-matching ?flow= encodes to compact #p= or #g= via encodeGraphToHash and round-trips', async () => {
    const g = await shareNs.decodeHashToGraph('?flow=gen:r:length=65~fold:f~score:s;gen~score');
    expect(g).not.toBeNull();
    const enc = await shareNs.encodeGraphToHash(g!);
    const dec = await shareNs.decodeHashToGraph(enc.hash);
    expect(dec).not.toBeNull();
    expect(dec!.nodes.map((n) => n.adapterId)).toEqual(g!.nodes.map((n) => n.adapterId));
    expect(graphUtilsNs.validateGraph(dec!)).toEqual([]);
  });

  it('T3-03 (F3 x F6): every ?flow= example in generated /llms.txt decodes and re-encodes stably', async () => {
    const { implemented, text } = await generateLiveLlmsTxt();
    expect(implemented).toBe(true);
    const flowUrls = extractWorkflowExampleUrls(text!).filter((u) => u.includes('?flow='));
    expect(flowUrls.length).toBeGreaterThanOrEqual(2);

    for (const rawUrl of flowUrls) {
      const query = `?${rawUrl.split('?')[1]}`;
      const g1 = await shareNs.decodeHashToGraph(query);
      expect(g1).not.toBeNull();
      const reEncoded = await encodeGraphToFlowQuery(g1!);
      expect(reEncoded.implemented).toBe(true);
      assertMarkdownAutolinkerSafe(reEncoded.url!);
      const g2 = await shareNs.decodeHashToGraph(`?flow=${reEncoded.dsl}`);
      expect(g2).not.toBeNull();
      expect(g2!.nodes.map((n) => n.adapterId)).toEqual(g1!.nodes.map((n) => n.adapterId));
      expect(g2!.edges).toHaveLength(g1!.edges.length);
    }
  });

  it('T3-04 (F3 x F8): multi-branch ?flow= comparison renders 1200x630 PNG and injects matching og:* meta tags', async () => {
    const flow = 'src:p:mode=demo-bundle;src~b3:r3:length=80~cmp:c;src~b4:r:length=80~cmp';
    const env = createMockCfEnv();

    const ogRes = await cfWorker.fetch(
      new Request(`https://denovo.run/api/og?flow=${encodeURIComponent(flow)}`),
      env,
    );
    expect(ogRes.status).toBe(200);
    assertValidOgPngBytes(new Uint8Array(await ogRes.arrayBuffer()), 1200, 630);

    const pageRes = await cfWorker.fetch(
      new Request(`https://denovo.run/?flow=${encodeURIComponent(flow)}`),
      env,
    );
    expect(pageRes.status).toBe(200);
    const html = await pageRes.text();
    expect(html).toContain('property="og:title"');
    expect(html).toContain('/api/og?flow=');
  });

  it('T3-05 (F3 x F9): POST /api/link with tolerated -> and (...) returns canonical Markdown-safe url + stateless shortUrl', async () => {
    const env = createMockCfEnv();
    const res = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow: 'r3(length=80)->m(numSequences=2)->f' }),
      }),
      env,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      ok: boolean;
      url: string;
      shortUrl: string;
      errors: unknown[];
    };
    expect(body.ok).toBe(true);
    expect(body.errors).toEqual([]);
    expect(body.url).toContain('?flow=');
    assertMarkdownAutolinkerSafe(body.url);
    expect(body.shortUrl).toMatch(/\?[gp]=/);
    assertMarkdownAutolinkerSafe(body.shortUrl);
  });

  it('T3-06 (F4 x F9): GET /api/link?flow=r~m~f~s returns all 4 auto-wired edges including designability-scorer dual inputs', async () => {
    const env = createMockCfEnv();
    const res = await cfWorker.fetch(
      new Request('https://denovo.run/api/link?flow=r~m~f~s'),
      env,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      ok: boolean;
      edges: Array<{ fromNode: string; fromPort: string; toNode: string; toPort: string }>;
    };
    expect(body.ok).toBe(true);
    expect(body.edges).toHaveLength(4);
    expect(body.edges.some((e) => e.fromPort === 'structure' && e.toPort === 'structure')).toBe(true);
    expect(body.edges.some((e) => e.fromPort === 'sequences' && e.toPort === 'sequences')).toBe(true);
    expect(body.edges.some((e) => e.fromPort === 'folds' && e.toPort === 'folds')).toBe(true);
  });

  it('T3-07 (F5 x F8): ?recipe= YAML with pdb-id=1PRW extracts rcsb structureTarget in og-graph and renders 1200x630 PNG', async () => {
    const yaml = [
      'denovo: v1',
      'name: "1PRW OG Test"',
      'nodes:',
      '  src: {"model":"pdb-input","mode":"pdb-id","pdbId":"1PRW"}',
      '  thermo: {"model":"thermompnn"}',
      'wires:',
      '  - src.structure -> thermo.structure',
    ].join('\n');
    const ogWf = await ogGraphNs.decodeOgWorkflowFromQuery(`?recipe=${encodeURIComponent(yaml)}`);
    expect(ogWf).not.toBeNull();
    expect(ogWf!.structureTarget).toEqual({ kind: 'rcsb', pdbId: '1PRW', cropSegments: '' });

    const env = createMockCfEnv();
    const res = await cfWorker.fetch(
      new Request(`https://denovo.run/api/og?recipe=${encodeURIComponent(yaml)}`),
      env,
    );
    expect(res.status).toBe(200);
    assertValidOgPngBytes(new Uint8Array(await res.arrayBuffer()), 1200, 630);
  });

  it('T3-08 (F5 x F9): POST /api/link with text/plain denovo: v1 YAML converts to shortUrl matching parseRecipeToGraph', async () => {
    const yaml = shareNs.graphToYamlRecipe(registryNs.WORKFLOW_PRESETS[7].graph);
    const expected = shareNs.parseRecipeToGraph(yaml);
    const env = createMockCfEnv();
    const res = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: yaml,
      }),
      env,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as { ok: boolean; shortUrl: string };
    expect(body.ok).toBe(true);
    const decoded = await shareNs.decodeHashToGraph(
      `?${new URL(body.shortUrl).searchParams.toString()}`,
    );
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes.map((n) => n.adapterId)).toEqual(expected.nodes.map((n) => n.adapterId));
    expect(decoded!.edges).toEqual(expected.edges);
  });

  it('T3-09 (F6 x F7): href from <link rel="llms"> in built index.html resolves to emitted web/build/llms.txt', () => {
    const indexPath = resolve(BUILD_DIR, 'index.html');
    expect(existsSync(indexPath), 'Expected web/build/index.html to exist').toBe(true);
    const html = readFileSync(indexPath, 'utf8');
    const m = html.match(/<link\b[^>]*\brel="llms"[^>]*\bhref="([^"]+)"/i);
    expect(m).not.toBeNull();
    const relPath = m![1].replace(/^\//, '');
    expect(existsSync(resolve(BUILD_DIR, relPath))).toBe(true);
  });

  it('T3-10 (F6 x F9): every example ?flow= URL in generated /llms.txt returns 200 ok:true on GET /api/link', async () => {
    const { implemented, text } = await generateLiveLlmsTxt();
    expect(implemented).toBe(true);
    const flowUrls = extractWorkflowExampleUrls(text!).filter((u) => u.includes('?flow='));
    expect(flowUrls.length).toBeGreaterThanOrEqual(2);
    const env = createMockCfEnv();

    for (const rawUrl of flowUrls) {
      const search = `?${rawUrl.split('?')[1]}`;
      const res = await cfWorker.fetch(new Request(`https://denovo.run/api/link${search}`), env);
      expect(res.status, `GET /api/link failed for ${rawUrl}`).toBe(200);
      const body = (await res.json()) as { ok: boolean; errors: unknown[] };
      expect(body.ok).toBe(true);
      expect(body.errors).toEqual([]);
    }
  });

  it('T3-11 (F7 x F8): GET /?flow=... injects dynamic og:* meta tags AND preserves <link rel="llms">', async () => {
    const appHtml = readFileSync(resolve(WEB_DIR, 'src/app.html'), 'utf8');
    const env = createMockCfEnv({ html: appHtml });
    const res = await cfWorker.fetch(new Request('https://denovo.run/?flow=r3~m~f~s'), env);
    expect(res.status).toBe(200);
    const out = await res.text();
    expect(out).toContain('/api/og?flow=r3');
    expect(out).toMatch(/<link\b[^>]*\brel="llms"[^>]*\bhref="\/llms\.txt"[^>]*>/i);
  });

  it('T3-12 (F8 x F9): shortUrl returned by POST /api/link renders 1200x630 PNG via GET /api/og with matching node count', async () => {
    const env = createMockCfEnv();
    const linkRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow: 'g:length=80~m~tk~f~s' }),
      }),
      env,
    );
    expect(linkRes.status).toBe(200);
    const linkBody = (await linkRes.json()) as {
      ok: boolean;
      shortUrl: string;
      nodes: unknown[];
    };
    expect(linkBody.ok).toBe(true);
    const shortSearch = `?${new URL(linkBody.shortUrl).searchParams.toString()}`;

    const ogWf = await ogGraphNs.decodeOgWorkflowFromQuery(shortSearch);
    expect(ogWf).not.toBeNull();
    expect(ogWf!.nodes).toHaveLength(linkBody.nodes.length);

    const ogRes = await cfWorker.fetch(
      new Request(`https://denovo.run/api/og${shortSearch}`),
      env,
    );
    expect(ogRes.status).toBe(200);
    assertValidOgPngBytes(new Uint8Array(await ogRes.arrayBuffer()), 1200, 630);
  });

  it('T3-13 (F1 x F6 x F7): built web/build/ has llms.txt, index.html with <link rel="llms">, _worker.js, _routes.json, and passes check:deploy', () => {
    for (const rel of ['llms.txt', 'index.html', '_worker.js', '_routes.json']) {
      expect(existsSync(resolve(BUILD_DIR, rel)), `Missing web/build/${rel}`).toBe(true);
    }
    const html = readFileSync(resolve(BUILD_DIR, 'index.html'), 'utf8');
    expect(html).toMatch(/<link\b[^>]*\brel="llms"[^>]*>/i);

    const out = execFileSync('node', [resolve(TOOLS_DIR, 'check-deploy.mjs')], {
      cwd: WEB_DIR,
      encoding: 'utf8',
    });
    expect(out).toContain('[check:deploy] PASS');
  });

  it('T3-14 (F3 x F10): generated url & shortUrl from /api/link survive realistic ChatGPT Markdown responses intact', async () => {
    const env = createMockCfEnv();
    const res = await cfWorker.fetch(
      new Request(
        'https://denovo.run/api/link?flow=src:p:mode=pdb-id,pdbId=5O45~r:mode=binder,length=75~m~f~s',
      ),
      env,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as { ok: boolean; url: string; shortUrl: string };
    expect(body.ok).toBe(true);

    assertMarkdownAutolinkerSafe(body.url);
    assertMarkdownAutolinkerSafe(body.shortUrl);

    const decodedMain = await shareNs.decodeHashToGraph(
      `?${new URL(body.url).searchParams.toString()}`,
    );
    const decodedShort = await shareNs.decodeHashToGraph(
      `?${new URL(body.shortUrl).searchParams.toString()}`,
    );
    expect(decodedMain).not.toBeNull();
    expect(decodedShort).not.toBeNull();
    expect(decodedMain!.nodes.map((n) => n.adapterId)).toEqual(
      decodedShort!.nodes.map((n) => n.adapterId),
    );
  });
});
