import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { deflateRawSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

import * as flowDslNs from '$lib/engine/flow-dsl';
import {
  BUILD_DIR,
  TOOLS_DIR,
  WEB_DIR,
  assertMarkdownAutolinkerSafe,
  assertValidOgPngBytes,
  cfWorker,
  cfWorkerNs,
  createMockCfEnv,
  generateLiveLlmsTxt,
  graphUtilsNs,
  ogGraphNs,
  pageHelpersNs,
  registryNs,
  renderOgNs,
  shareNs,
} from './helpers';

function toBase64Url(buf: Buffer): string {
  return buf.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

describe('Tier 5 — Adversarial Security, Boundary & Edge-Case Hardening (11 cases)', () => {
  it('T5-01 (HIGH-01): enforces MAX_HASH_CHARS on POST /api/link & encoded output and bounds error.token/error.message', async () => {
    const env = createMockCfEnv();
    const overLimit = flowDslNs.MAX_HASH_CHARS + 100;

    // 1. POST JSON { flow } > MAX_HASH_CHARS -> 400 HASH_TOO_LARGE
    const jsonTooBig = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow: `badmodel_${'a'.repeat(overLimit)}` }),
      }),
      env,
    );
    expect(jsonTooBig.status).toBe(400);
    const jsonTooBigBody = (await jsonTooBig.json()) as {
      ok: boolean;
      errors: Array<{ code: string; message: string; token?: string }>;
    };
    expect(jsonTooBigBody.ok).toBe(false);
    expect(jsonTooBigBody.errors[0].code).toBe('HASH_TOO_LARGE');

    // 2. POST text/plain > MAX_HASH_CHARS -> 400 HASH_TOO_LARGE
    const textTooBig = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: `badmodel_${'a'.repeat(overLimit)}`,
      }),
      env,
    );
    expect(textTooBig.status).toBe(400);
    const textTooBigBody = (await textTooBig.json()) as {
      ok: boolean;
      errors: Array<{ code: string; message: string; token?: string }>;
    };
    expect(textTooBigBody.ok).toBe(false);
    expect(textTooBigBody.errors[0].code).toBe('HASH_TOO_LARGE');

    // 3. Encoded ?flow= query > MAX_HASH_CHARS via { flow } (percent expansion), { graph }, and GET ?recipe=
    const expandedFlow = `r3:_label="${'!'.repeat(25_000)}"~m~f`;
    expect(expandedFlow.length).toBeLessThan(flowDslNs.MAX_HASH_CHARS);
    const expandedRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow: expandedFlow }),
      }),
      env,
    );
    expect(expandedRes.status).toBe(400);
    const expandedBody = (await expandedRes.json()) as {
      ok: boolean;
      errors: Array<{ code: string }>;
    };
    expect(expandedBody.ok).toBe(false);
    expect(expandedBody.errors[0].code).toBe('HASH_TOO_LARGE');

    const hugeGraph = {
      id: 'big',
      name: 'Big',
      nodes: [
        { id: 'r3', adapterId: 'rfd3', label: 'A'.repeat(70_000), params: {} },
        { id: 'm', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN', params: {} },
      ],
      edges: [{ fromNode: 'r3', fromPort: 'structure', toNode: 'm', toPort: 'structure' }],
    };
    const graphTooBigRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: hugeGraph }),
      }),
      env,
    );
    expect(graphTooBigRes.status).toBe(400);
    const graphTooBigBody = (await graphTooBigRes.json()) as {
      ok: boolean;
      errors: Array<{ code: string }>;
    };
    expect(graphTooBigBody.ok).toBe(false);
    expect(graphTooBigBody.errors[0].code).toBe('HASH_TOO_LARGE');

    const recipeGetRes = await cfWorker.fetch(
      new Request(
        `https://denovo.run/api/link?recipe=${encodeURIComponent(JSON.stringify(hugeGraph))}`,
      ),
      env,
    );
    expect(recipeGetRes.status).toBe(400);
    const recipeGetBody = (await recipeGetRes.json()) as {
      ok: boolean;
      errors: Array<{ code: string }>;
    };
    expect(recipeGetBody.ok).toBe(false);
    expect(recipeGetBody.errors[0].code).toBe('HASH_TOO_LARGE');

    // 4. Error token (<= 160) & message (<= 512) are capped on 65,000-char invalid inputs
    const hugeInvalid = `bad_${'a'.repeat(65_000)}`;
    const probes: Request[] = [
      new Request(`https://denovo.run/api/link?flow=${encodeURIComponent(hugeInvalid)}`),
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow: hugeInvalid }),
      }),
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: hugeInvalid,
      }),
    ];
    for (const req of probes) {
      const res = await cfWorker.fetch(req, env);
      expect(res.status).toBe(400);
      const body = (await res.json()) as {
        ok: boolean;
        errors: Array<{ code: string; message: string; token?: string; nodeId?: string }>;
      };
      expect(body.ok).toBe(false);
      expect(body.errors.length).toBeGreaterThan(0);
      for (const err of body.errors) {
        expect(err.message.length).toBeLessThanOrEqual(160 + 352);
        expect(err.message.length).toBeLessThanOrEqual(512);
        if (err.token !== undefined) expect(err.token.length).toBeLessThanOrEqual(160);
        if (err.nodeId !== undefined) expect(err.nodeId.length).toBeLessThanOrEqual(160);
      }
    }
  });

  it('T5-02 (HIGH-02): POST /api/link text/plain preserves quoted &p=, &flow=, ?g=, #p= inside _label while keeping query/URL support', async () => {
    const env = createMockCfEnv();
    const quotedLabels = ['A&p=B', 'Test&flow=r~m', 'x?g=1', 'x#p=1'];
    for (const label of quotedLabels) {
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'text/plain' },
          body: `r3:_label="${label}"~m~f`,
        }),
        env,
      );
      expect(res.status, label).toBe(200);
      const body = (await res.json()) as {
        ok: boolean;
        nodes: Array<{ label: string }>;
        url: string;
      };
      expect(body.ok, label).toBe(true);
      expect(body.nodes[0].label).toBe(label);
      const roundTrip = await flowDslNs.decodeFlowQuery(body.url);
      expect(roundTrip.ok).toBe(true);
      expect(roundTrip.nodes[0].label).toBe(label);
    }

    const queryInputs = [
      'flow=r3~m~f',
      '?utm_source=chatgpt.com#flow=r3~m~f',
      '/?utm_source=chatgpt.com#flow=r3~m~f',
    ];
    for (const qInput of queryInputs) {
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'text/plain' },
          body: qInput,
        }),
        env,
      );
      expect(res.status, qInput).toBe(200);
      const body = (await res.json()) as { ok: boolean; nodes: Array<{ adapterId: string }> };
      expect(body.ok, qInput).toBe(true);
      expect(body.nodes.map((n) => n.adapterId)).toEqual([
        'rfd3',
        'web-protein-mpnn',
        'localfold-af2',
      ]);
    }
  });

  it('T5-03 (MEDIUM-01): relative-path /?... and /index.html?...#flow= URLs work across all parsers while non-workflow paths are rejected', async () => {
    const validRelativeUrls = [
      '/?flow=r3~m~f',
      '/?utm_source=chatgpt.com#flow=r3~m~f',
      '/index.html?flow=r3~m~f',
    ];
    for (const rel of validRelativeUrls) {
      expect(flowDslNs.extractWorkflowSearchOrHash(rel).clean, rel).toBe('flow=r3~m~f');
      expect(pageHelpersNs.isSharedWorkflowHash(rel), rel).toBe(true);
      expect(flowDslNs.isMarkdownSafeFlowUrl(rel), rel).toBe(true);

      const decFlow = flowDslNs.decodeFlowQuery(rel);
      expect(decFlow.ok, rel).toBe(true);
      expect(decFlow.nodes.map((n) => n.adapterId)).toEqual([
        'rfd3',
        'web-protein-mpnn',
        'localfold-af2',
      ]);

      const ogWf = await ogGraphNs.decodeOgWorkflowFromQuery(rel);
      expect(ogWf, rel).not.toBeNull();
      expect(ogWf!.nodes).toHaveLength(3);

      const clientGraph = await shareNs.decodeHashToGraph(rel);
      expect(clientGraph, rel).not.toBeNull();
      expect(clientGraph!.nodes).toHaveLength(3);
    }

    const nonWorkflowPaths = ['/?utm_source=chatgpt.com#section', '//evil.example/?flow=r3~m~f'];
    for (const badRel of nonWorkflowPaths) {
      expect(pageHelpersNs.isSharedWorkflowHash(badRel), badRel).toBe(false);
      expect(flowDslNs.decodeFlowQuery(badRel).ok, badRel).toBe(false);
      expect(await ogGraphNs.decodeOgWorkflowFromQuery(badRel), badRel).toBeNull();
      expect(await shareNs.decodeHashToGraph(badRel), badRel).toBeNull();
    }
    expect(flowDslNs.isMarkdownSafeFlowUrl('/?utm_source=chatgpt.com#section')).toBe(false);
  });

  it('T5-04 (MEDIUM-02): preserves balanced braces/brackets in values while still stripping unbalanced trailing prose punctuation', () => {
    expect(flowDslNs.stripTrailingProsePunctuation('r3:_label=Domain{1}')).toBe(
      'r3:_label=Domain{1}',
    );
    expect(flowDslNs.stripTrailingProsePunctuation('r3~m~f}')).toBe('r3~m~f');
    expect(flowDslNs.stripTrailingProsePunctuation('r3~m~f]')).toBe('r3~m~f');
    expect(flowDslNs.stripTrailingProsePunctuation('r3~m~f)')).toBe('r3~m~f');

    for (const label of ['Domain{1}', '{braces}']) {
      const base = flowDslNs.parseFlowDsl('r3~m~f').graph!;
      base.nodes[base.nodes.length - 1].label = label;
      const enc = flowDslNs.encodeGraphToFlow(base);
      const url = `https://denovo.run/?${enc.query}`;
      expect(flowDslNs.isMarkdownSafeFlowUrl(url)).toBe(true);
      const dec = flowDslNs.decodeFlowQuery(url);
      expect(dec.ok, label).toBe(true);
      expect(dec.nodes[dec.nodes.length - 1].label).toBe(label);
    }

    const jsonSpec = flowDslNs.parseFlowDsl('rfd4:customSpecYaml={"contig":"70-70"}');
    expect(jsonSpec.ok).toBe(true);
    expect(jsonSpec.nodes[0].params.customSpecYaml).toBe('{"contig":"70-70"}');
  });

  it('T5-05 (MEDIUM-03 + security): resists prototype pollution, >128 edges, deflate bombs, XSS, and 64KB ReDoS', async () => {
    // 1. Prototype pollution in parseFlowDsl
    for (const bad of [
      '__proto__:r3~m~f',
      'constructor:r3~m~f',
      'prototype:r3~m~f',
      'r3:__proto__=1~m~f',
      'r3:constructor=1~m~f',
      'r3:prototype=1~m~f',
      '__proto__~m',
      'constructor~m',
    ]) {
      const r = flowDslNs.parseFlowDsl(bad);
      expect(r.ok, bad).toBe(false);
    }

    // 2. Prototype pollution in parseRecipeSafe (including depth > 8)
    let deep: Record<string, unknown> = JSON.parse('{"__proto__":{"polluted":"yes"},"constructor":"bad"}');
    for (let i = 0; i < 12; i++) {
      deep = { nested: deep };
    }
    const recipePayload = JSON.stringify({
      id: 'pp-test',
      name: 'PP',
      nodes: [
        {
          id: 'r3',
          adapterId: 'rfd3',
          params: {
            length: 80,
            ...JSON.parse('{"__proto__":{"polluted":"yes"},"constructor":"bad"}'),
            deep,
          },
        },
        { id: 'm', adapterId: 'web-protein-mpnn', params: {} },
      ],
      edges: [{ fromNode: 'r3', fromPort: 'structure', toNode: 'm', toPort: 'structure' }],
    });
    const recRes = flowDslNs.parseRecipeSafe(recipePayload);
    expect(recRes.ok).toBe(true);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    const r3Params = recRes.graph!.nodes[0].params;
    expect(Object.hasOwn(r3Params, 'constructor')).toBe(false);
    expect(Object.hasOwn(r3Params, '__proto__')).toBe(false);
    let cursor: unknown = r3Params.deep;
    for (let i = 0; i < 12 && cursor && typeof cursor === 'object'; i++) {
      expect(Object.hasOwn(cursor, '__proto__')).toBe(false);
      expect(Object.hasOwn(cursor, 'constructor')).toBe(false);
      cursor = (cursor as Record<string, unknown>).nested;
    }

    // 3. og-graph.ts ?g= prototype-key lookups & parameter sanitization
    for (const protoShortA of ['constructor', '__proto__', 'prototype']) {
      const gProto = toBase64Url(
        deflateRawSync(JSON.stringify({ v: 2, n: [['n1', protoShortA]], e: [] })),
      );
      expect(await ogGraphNs.decodeOgWorkflowFromQuery(`?g=${gProto}`)).toBeNull();
    }
    const gToString = toBase64Url(
      deflateRawSync(
        JSON.stringify({
          v: 2,
          n: [['n1', 'toString', JSON.parse('{"__proto__":{"p":1},"constructor":"bad","length":80}')]],
          e: [],
        }),
      ),
    );
    const ogToStr = await ogGraphNs.decodeOgWorkflowFromQuery(`?g=${gToString}`);
    expect(ogToStr).not.toBeNull();
    expect(Object.hasOwn(ogToStr!.nodes[0].params, 'constructor')).toBe(false);
    expect(Object.hasOwn(ogToStr!.nodes[0].params, '__proto__')).toBe(false);
    expect(ogToStr!.nodes[0].params.length).toBe(80);
    expect(({} as Record<string, unknown>).p).toBeUndefined();

    // 4. og-graph.ts ?g= rejects > 128 edges cleanly
    const manyEdges = Array.from({ length: 129 }, () => ['n1', 'structure', 'n2', 'structure']);
    const gTooManyEdges = toBase64Url(
      deflateRawSync(JSON.stringify({ v: 2, n: [['n1', 'r3'], ['n2', 'm']], e: manyEdges })),
    );
    expect(await ogGraphNs.decodeOgWorkflowFromQuery(`?g=${gTooManyEdges}`)).toBeNull();

    // 5. 2 MiB deflate-raw bomb cleanly rejected across all decoders
    const bomb = toBase64Url(deflateRawSync('A'.repeat(2 * 1024 * 1024)));
    await expect(shareNs.decodeHashToGraph(`?g=${bomb}`)).rejects.toThrow();
    const bombDq = await flowDslNs.decodeWorkflowQuery(`?g=${bomb}`);
    expect(bombDq.ok).toBe(false);
    expect(bombDq.errors[0].code).toBe('INVALID_COMPRESSED_DAG');
    expect(await ogGraphNs.decodeOgWorkflowFromQuery(`?g=${bomb}`)).toBeNull();
    const env = createMockCfEnv();
    const bombApi = await cfWorker.fetch(new Request(`https://denovo.run/api/link?g=${bomb}`), env);
    expect(bombApi.status).toBe(400);

    // 6. HTML/XSS payloads escaped in injectOgMetaIntoHtml & rendered safely to PNG
    const xssTitle = '</title><script>alert(1)</script>';
    const xssDesc = '" onerror="alert(1) $100 $&';
    const injected = cfWorkerNs.injectOgMetaIntoHtml(
      '<html><head><title>Default</title><meta property="og:title" content="Old" /></head><body></body></html>',
      {
        title: xssTitle,
        description: xssDesc,
        imageUrl: 'https://denovo.run/api/og?flow=r3~m~f',
        pageUrl: 'https://denovo.run/?flow=r3~m~f',
      },
    );
    expect(injected).not.toContain('<script>alert(1)</script>');
    expect(injected).toContain('&lt;/title&gt;&lt;script&gt;alert(1)&lt;/script&gt;');
    expect(injected).toContain('&quot; onerror=&quot;alert(1) $100 $&amp;');
    const xssWf = await ogGraphNs.decodeOgWorkflowFromQuery(
      `?flow=${encodeURIComponent(`r3:_label="${xssTitle}"~m~f`)}`,
    );
    expect(xssWf).not.toBeNull();
    assertValidOgPngBytes(await renderOgNs.renderWorkflowOgPng(xssWf!, null), 1200, 630);

    // 7. 64 KB ReDoS & stress inputs complete rapidly (< 1000 ms)
    const t0 = performance.now();
    flowDslNs.parseFlowDsl(`r3:contig=${'10,'.repeat(21_000)}10~m~f`);
    const manyNodesRes = flowDslNs.parseFlowDsl(
      Array.from({ length: 8_000 }, (_, i) => `n${i}:r3`).join(';'),
    );
    expect(manyNodesRes.ok).toBe(false);
    const denseEdges = Array.from({ length: 2_000 }, (_, i) => `n${i % 64}~n${(i + 1) % 64}`).join(
      ';',
    );
    const denseRes = flowDslNs.parseFlowDsl(
      `${Array.from({ length: 64 }, (_, i) => `n${i}:r3`).join(';')};${denseEdges}`,
    );
    expect(denseRes.ok).toBe(false);
    expect(performance.now() - t0).toBeLessThan(1000);
  });

  it('T5-06 (LOW-03 + Markdown/Unicode): round-trips special characters safely and keeps preset shortUrl trailing-char safe', async () => {
    const trickyLabels = [
      'Parens (v1)',
      'Brackets [v1]',
      'Angles <v1>',
      'Braces {v1}',
      'Tilde ~~double~~',
      'Triple ~~~tilde',
      'Trail_',
      'Trail-',
      'Trail.',
      'Trail,',
      'Trail:',
      'Trail;',
      'Trail!',
      'Trail?',
      'a&b=c,d;e~f:g',
      '100% pure #1 + good',
      'quote"and\\slash\'tick`',
      '🧬 Protein α-helix 中文 café',
    ];

    for (const label of trickyLabels) {
      const g = flowDslNs.parseFlowDsl('r3~m~f').graph!;
      g.nodes[2].label = label;
      const enc = flowDslNs.encodeGraphToFlow(g);
      const url = `https://denovo.run/?${enc.query}`;
      expect(flowDslNs.isMarkdownSafeFlowUrl(url), label).toBe(true);
      assertMarkdownAutolinkerSafe(url);
      const dec = flowDslNs.decodeFlowQuery(url);
      expect(dec.ok, label).toBe(true);
      expect(dec.nodes[2].label, label).toBe(label);
    }

    // LOW-03: Preset shortUrl with custom label ending in `_` or `.` never ends in bare `_` or `.`
    const preset = registryNs.WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-mpnn-refold')!;
    expect(preset).toBeDefined();
    for (const suffix of ['_', '.', '-', ':', ';', ',']) {
      const customLabel = `Scorer${suffix}`;
      const g = structuredClone(preset.graph);
      g.nodes[g.nodes.length - 1].label = customLabel;
      const compactEdge = await flowDslNs.encodeCompactShareQuery(g);
      const compactClient = await shareNs.encodeGraphToHash(g);
      expect(compactEdge.query, suffix).not.toMatch(/[._,;\-:]$/);
      expect(compactClient.hash, suffix).not.toMatch(/[._,;\-:]$/);

      const env = createMockCfEnv();
      const linkRes = await cfWorker.fetch(
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ graph: g }),
        }),
        env,
      );
      expect(linkRes.status).toBe(200);
      const linkBody = (await linkRes.json()) as { shortUrl: string };
      expect(linkBody.shortUrl, suffix).not.toMatch(/[._,;\-:]$/);

      const decClient = await shareNs.decodeHashToGraph(linkBody.shortUrl);
      expect(decClient).not.toBeNull();
      expect(decClient!.nodes[decClient!.nodes.length - 1].label).toBe(customLabel);

      const decOg = await ogGraphNs.decodeOgWorkflowFromQuery(linkBody.shortUrl);
      expect(decOg).not.toBeNull();
      expect(decOg!.nodes[decOg!.nodes.length - 1].label).toBe(customLabel);
    }
  });

  it('T5-07 (MEDIUM-04 + vite.config.ts): check-deploy.mjs enforces llms.txt > 500 bytes and vite.config.ts guards /llms.txt', async () => {
    const checkDeployPath = resolve(TOOLS_DIR, 'check-deploy.mjs');
    const checkDeploySrc = readFileSync(checkDeployPath, 'utf8');
    expect(checkDeploySrc).toContain("'llms.txt'");

    const { implemented, text } = await generateLiveLlmsTxt();
    expect(implemented).toBe(true);
    expect(Buffer.byteLength(text!, 'utf8')).toBeGreaterThan(500);

    if (existsSync(resolve(BUILD_DIR, 'index.html'))) {
      expect(existsSync(resolve(BUILD_DIR, 'llms.txt'))).toBe(true);
      expect(statSync(resolve(BUILD_DIR, 'llms.txt')).size).toBeGreaterThan(500);
      const out = execFileSync('node', [checkDeployPath], {
        cwd: WEB_DIR,
        encoding: 'utf8',
      });
      expect(out).toContain('[check:deploy] PASS:');
    }

    const viteCfg = readFileSync(resolve(WEB_DIR, 'vite.config.ts'), 'utf8');
    expect(viteCfg).toContain("server.middlewares.use('/llms.txt'");
    expect(viteCfg).toContain("req.method !== 'GET' && req.method !== 'HEAD'");
    expect(viteCfg).toContain("reqPath !== '/' && reqPath !== ''");
    expect(viteCfg).toContain("fileName: 'llms.txt'");
  });

  it('T5-08 (NEW-HIGH-01): rejects { toString: 1, valueOf: 1 } non-primitive objects cleanly without TypeError across /api/link, flow-dsl, og-graph, and share', async () => {
    const env = createMockCfEnv();
    const evilObj = { toString: 1, valueOf: 1 };

    const evilGraphs = [
      {
        nodes: [{ id: 'n1', adapterId: 'rfd4-proteina', params: { customSpecYaml: evilObj } }],
        edges: [],
      },
      {
        nodes: [{ id: 'n1', adapterId: 'rfd4-proteina', params: { mode: evilObj } }],
        edges: [],
      },
      {
        nodes: [{ id: 'n1', adapterId: 'rfd3', params: { length: evilObj } }],
        edges: [],
      },
      {
        nodes: [{ id: 'n1', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: evilObj } }],
        edges: [],
      },
      {
        nodes: [
          { id: 'n1', adapterId: 'pdb-input', params: { mode: 'alphafold-db', uniprotId: evilObj } },
        ],
        edges: [],
      },
      {
        nodes: [{ id: 'n1', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: evilObj } }],
        edges: [],
      },
      {
        nodes: [{ id: 'n1', adapterId: 'sequence-input', params: { sequence: evilObj } }],
        edges: [],
      },
      {
        nodes: [{ id: evilObj, adapterId: 'rfd3', params: {} }],
        edges: [],
      },
      {
        nodes: [{ id: 'n1', adapterId: evilObj, params: {} }],
        edges: [],
      },
    ];

    for (const g of evilGraphs) {
      const jsonStr = JSON.stringify(g);

      const recipeRes = flowDslNs.parseRecipeSafe(jsonStr);
      expect(recipeRes.ok, jsonStr).toBe(false);
      expect(recipeRes.errors.length).toBeGreaterThan(0);

      const postGraphRes = await cfWorker.fetch(
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ graph: g }),
        }),
        env,
      );
      expect(postGraphRes.status, jsonStr).toBe(400);
      const postBody = (await postGraphRes.json()) as { ok: boolean; errors: unknown[] };
      expect(postBody.ok).toBe(false);
      expect(postBody.errors.length).toBeGreaterThan(0);

      const getRecipeRes = await cfWorker.fetch(
        new Request(`https://denovo.run/api/link?recipe=${encodeURIComponent(jsonStr)}`),
        env,
      );
      expect(getRecipeRes.status, jsonStr).toBe(400);

      expect(await ogGraphNs.decodeOgWorkflowFromQuery(`?recipe=${encodeURIComponent(jsonStr)}`)).toBeNull();
    }

    // YAML recipe with { toString: 1, valueOf: 1 } param
    const evilYaml = [
      'denovo: v1',
      'nodes:',
      '  n1: {"model":"rfd4-proteina","customSpecYaml":{"toString":1,"valueOf":1}}',
    ].join('\n');
    const yamlPostRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: evilYaml,
      }),
      env,
    );
    expect(yamlPostRes.status).toBe(400);
    expect(((await yamlPostRes.json()) as { ok: boolean }).ok).toBe(false);

    // Compressed ?g= payloads with { toString: 1, valueOf: 1 } across og-graph, share, flow-dsl, /api/link
    const gInvalidParamPayloads = [
      { v: 2, n: [['n1', 'p', { mode: evilObj }]], e: [] },
      { v: 2, n: [['n1', 'p', { mode: 'pdb-id', pdbId: evilObj }]], e: [] },
      { v: 2, n: [['n1', 'r', { customSpecYaml: evilObj }]], e: [] },
      { v: 2, n: [['n1', 'r3', { length: evilObj }]], e: [] },
    ];
    for (const payload of gInvalidParamPayloads) {
      const gStr = toBase64Url(deflateRawSync(JSON.stringify(payload)));
      const ogOut = await ogGraphNs.decodeOgWorkflowFromQuery(`?g=${gStr}`);
      expect(ogOut).not.toBeNull();
      const dqOut = await flowDslNs.decodeWorkflowQuery(`?g=${gStr}`);
      expect(dqOut.ok).toBe(false);
    }

    const gAuxEvilPayloads = [
      { v: 2, n: [['n1', 'p', { mode: 'custom', filename: evilObj }]], e: [] },
      { v: 2, n: [['n1', 'r3']], e: [], p: [[evilObj, 0]] },
    ];
    for (const payload of gAuxEvilPayloads) {
      const gStr = toBase64Url(deflateRawSync(JSON.stringify(payload)));
      const ogOut = await ogGraphNs.decodeOgWorkflowFromQuery(`?g=${gStr}`);
      expect(ogOut).not.toBeNull();
      const dqOut = await flowDslNs.decodeWorkflowQuery(`?g=${gStr}`);
      expect(typeof dqOut.ok).toBe('boolean');
    }

    const gBadTuplePayloads = [
      { v: 2, n: [[evilObj, 'r3']], e: [] },
      { v: 2, n: [['n1', evilObj]], e: [] },
      { v: 2, n: [['n1', 'r3'], ['n2', 'm']], e: [[evilObj, 'structure', 'n2', 'structure']] },
    ];
    for (const payload of gBadTuplePayloads) {
      const gStr = toBase64Url(deflateRawSync(JSON.stringify(payload)));
      expect(await ogGraphNs.decodeOgWorkflowFromQuery(`?g=${gStr}`)).toBeNull();
      expect(await shareNs.decodeHashToGraph(`?g=${gStr}`)).toBeNull();
      const dqOut = await flowDslNs.decodeWorkflowQuery(`?g=${gStr}`);
      expect(dqOut.ok).toBe(false);
      const apiOut = await cfWorker.fetch(new Request(`https://denovo.run/api/link?g=${gStr}`), env);
      expect(apiOut.status).toBe(400);
    }
  });

  it('T5-09 (NEW-MEDIUM-01): keeps POST /api/link shortUrl <= MAX_HASH_CHARS via ?g= fallback when preset-diff URLSearchParams expands past limit', async () => {
    const env = createMockCfEnv();
    const preset = registryNs.WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-mpnn-refold')!;
    expect(preset).toBeDefined();
    const g = structuredClone(preset.graph);
    g.nodes[0].label = ';'.repeat(22_000);

    const res = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: g }),
      }),
      env,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as { ok: boolean; url: string; shortUrl: string };
    expect(body.ok).toBe(true);
    const urlQuery = new URL(body.url).search.slice(1);
    const shortQuery = new URL(body.shortUrl).search.slice(1);
    expect(urlQuery.length).toBeLessThanOrEqual(flowDslNs.MAX_HASH_CHARS);
    expect(shortQuery.length).toBeLessThanOrEqual(flowDslNs.MAX_HASH_CHARS);
    expect(shortQuery.startsWith('g=')).toBe(true);

    const decFlow = await flowDslNs.decodeWorkflowQuery(body.shortUrl);
    expect(decFlow.ok).toBe(true);
    expect(decFlow.nodes[0].label).toBe(';'.repeat(22_000));
    const decClient = await shareNs.decodeHashToGraph(body.shortUrl);
    expect(decClient).not.toBeNull();
    expect(decClient!.nodes[0].label).toBe(';'.repeat(22_000));
    const decOg = await ogGraphNs.decodeOgWorkflowFromQuery(body.shortUrl);
    expect(decOg).not.toBeNull();
    expect(decOg!.nodes[0].label).toBe(';'.repeat(22_000));
  });

  it('T5-10 (NEW-MEDIUM-02): POST /api/link text/plain classifies nodes:/denovo: DSL, quoted multiline labels, multi-param queries, and real YAML recipes accurately', async () => {
    const env = createMockCfEnv();

    const validDslBodies: Array<{ body: string; expectedAdapters: string[]; firstLabel?: string }> = [
      { body: 'nodes:r3~m~f', expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'] },
      {
        body: 'src:p\nnodes:r3\nsrc~nodes~m~f',
        expectedAdapters: ['pdb-input', 'rfd3', 'web-protein-mpnn', 'localfold-af2'],
      },
      { body: 'denovo:r3~m~f', expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'] },
      {
        body: 'r3:_label="hello\nnodes:world"~m~f',
        expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'],
        firstLabel: 'hello\nnodes:world',
      },
      {
        body: 'r3:_label="hello\nnodes:\nworld"~m~f',
        expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'],
        firstLabel: 'hello\nnodes:\nworld',
      },
      { body: 'v=2&flow=r3~m~f', expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'] },
      {
        body: 'ref=chatgpt&flow=r3~m~f',
        expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'],
      },
      {
        body: 'r3:_label="A&p=B"~m~f',
        expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'],
        firstLabel: 'A&p=B',
      },
      {
        body: 'r3:_label="Test&flow=r~m"~m~f',
        expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'],
        firstLabel: 'Test&flow=r~m',
      },
      {
        body: 'r3:_label="x?g=1"~m~f',
        expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'],
        firstLabel: 'x?g=1',
      },
      {
        body: 'r3:_label="x#p=1"~m~f',
        expectedAdapters: ['rfd3', 'web-protein-mpnn', 'localfold-af2'],
        firstLabel: 'x#p=1',
      },
    ];

    for (const tc of validDslBodies) {
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'text/plain' },
          body: tc.body,
        }),
        env,
      );
      expect(res.status, tc.body).toBe(200);
      const json = (await res.json()) as {
        ok: boolean;
        nodes: Array<{ adapterId: string; label: string }>;
      };
      expect(json.ok, tc.body).toBe(true);
      expect(json.nodes.map((n) => n.adapterId), tc.body).toEqual(tc.expectedAdapters);
      if (tc.firstLabel !== undefined) {
        expect(json.nodes[0].label, tc.body).toBe(tc.firstLabel);
      }
    }

    // Real YAML recipes still route to parseRecipeSafe
    const validYaml = 'denovo: v1\nnodes:\n  r: {"model":"rfd3","length":80}\n';
    const validYamlRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: validYaml,
      }),
      env,
    );
    expect(validYamlRes.status).toBe(200);
    expect(((await validYamlRes.json()) as { ok: boolean }).ok).toBe(true);

    const standaloneNodesYaml = 'nodes:\n  r: {"model":"rfd3","length":80}\n';
    const standaloneYamlRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: standaloneNodesYaml,
      }),
      env,
    );
    expect(standaloneYamlRes.status).toBe(200);
    expect(((await standaloneYamlRes.json()) as { ok: boolean }).ok).toBe(true);

    const badVerYaml = 'denovo: v2\nnodes:\n  r: {"model":"rfd3"}\n';
    const badVerRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: badVerYaml,
      }),
      env,
    );
    expect(badVerRes.status).toBe(400);
    const badVerBody = (await badVerRes.json()) as {
      ok: boolean;
      errors: Array<{ code: string }>;
    };
    expect(badVerBody.ok).toBe(false);
    expect(badVerBody.errors[0].code).toBe('INVALID_RECIPE');
  });

  it('T5-11 (NEW-MEDIUM-03, LOW-A, LOW-B): validateGraph handles decoded ?g=/?recipe= {toString:1,valueOf:1} params without TypeError, encodeGraphToHash falls back to #g=, and encodeGraphToFlow tolerates null-proto IDs', async () => {
    const evilObj = { toString: 1, valueOf: 1 };

    // 1. NEW-MEDIUM-03: ?g= and ?recipe= decoded via share.decodeHashToGraph -> graphUtils.validateGraph(decoded)
    const gPayloads = [
      { v: 2, n: [['n1', 'r', { customSpecYaml: evilObj }]], e: [] },
      { v: 2, n: [['n1', 'r', { mode: 'motif', contig: evilObj }]], e: [] },
      { v: 2, n: [['n1', 'p', { mode: 'pdb-id', pdbId: evilObj }]], e: [] },
      { v: 2, n: [['n1', 'p', { mode: 'alphafold-db', uniprotId: evilObj }]], e: [] },
      { v: 2, n: [['n1', 'q', { sequence: evilObj }]], e: [] },
      {
        v: 2,
        n: [
          [
            'n1',
            'p',
            {
              mode: 'custom',
              pdbText: evilObj,
              expectedPdbSha256: evilObj,
              expectedFilename: evilObj,
              filename: evilObj,
            },
          ],
        ],
        e: [],
      },
    ];
    for (const payload of gPayloads) {
      const gStr = toBase64Url(deflateRawSync(JSON.stringify(payload)));
      const decoded = await shareNs.decodeHashToGraph(`?g=${gStr}`);
      expect(decoded).not.toBeNull();
      const errs = graphUtilsNs.validateGraph(decoded!);
      expect(errs.length).toBeGreaterThan(0);
    }

    const recipePayloads = [
      { nodes: [{ id: 'n1', adapterId: 'rfd4-proteina', params: { customSpecYaml: evilObj } }], edges: [] },
      {
        nodes: [{ id: 'n1', adapterId: 'rfd4-proteina', params: { mode: 'motif', contig: evilObj } }],
        edges: [],
      },
      { nodes: [{ id: 'n1', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: evilObj } }], edges: [] },
      {
        nodes: [{ id: 'n1', adapterId: 'pdb-input', params: { mode: 'alphafold-db', uniprotId: evilObj } }],
        edges: [],
      },
      { nodes: [{ id: 'n1', adapterId: 'sequence-input', params: { sequence: evilObj } }], edges: [] },
    ];
    for (const recipe of recipePayloads) {
      const decoded = await shareNs.decodeHashToGraph(
        `?recipe=${encodeURIComponent(JSON.stringify(recipe))}`,
      );
      expect(decoded).not.toBeNull();
      const errs = graphUtilsNs.validateGraph(decoded!);
      expect(errs.length).toBeGreaterThan(0);
    }

    // 2. LOW-A: share.encodeGraphToHash falls back to #g= when `;` expansion exceeds MAX_HASH_CHARS
    const preset = registryNs.WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-mpnn-refold')!;
    expect(preset).toBeDefined();
    const bigLabelGraph = structuredClone(preset.graph);
    bigLabelGraph.nodes[0].label = ';'.repeat(22_000);
    const encodedHash = await shareNs.encodeGraphToHash(bigLabelGraph);
    expect(encodedHash.mode).toBe('compressed-dag');
    expect(encodedHash.hash.startsWith('#g=')).toBe(true);
    expect(encodedHash.length).toBeLessThanOrEqual(flowDslNs.MAX_HASH_CHARS);
    const roundTripped = await shareNs.decodeHashToGraph(encodedHash.hash);
    expect(roundTripped).not.toBeNull();
    expect(roundTripped!.nodes[0].label).toBe(';'.repeat(22_000));

    // 3. LOW-B: encodeGraphToFlow with Object.create(null) node/edge IDs & ports does not throw TypeError
    const nullProto = Object.create(null) as unknown as string;
    const directNullProtoGraph = {
      id: 'null-proto-test',
      name: 'Null Proto Test',
      nodes: [
        { id: nullProto, adapterId: 'rfd3', label: nullProto, params: {} },
        { id: 'n2', adapterId: 'web-protein-mpnn', label: 'MPNN', params: {} },
      ],
      edges: [
        {
          fromNode: nullProto,
          fromPort: nullProto,
          toNode: 'n2',
          toPort: nullProto,
        },
      ],
    };
    const flowRes = flowDslNs.encodeGraphToFlow(directNullProtoGraph);
    expect(flowRes.dsl.length).toBeGreaterThan(0);
    expect(flowRes.query.startsWith('flow=')).toBe(true);
  });
});
