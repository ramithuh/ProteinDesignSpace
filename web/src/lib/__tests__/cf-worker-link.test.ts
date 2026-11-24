import { describe, expect, it } from 'vitest';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import esbuild from 'esbuild';
import cfWorker, { handleLinkRequest, type LinkApiResponse } from '$lib/og/cf-worker';
import { decodeOgWorkflowFromQuery } from '$lib/og/og-graph';
import { MAX_HASH_CHARS, MAX_INFLATED_BYTES, isMarkdownSafeFlowUrl } from '$lib/engine/flow-dsl';
import { WORKFLOW_PRESETS } from '$lib/engine/registry';
import { decodeHashToGraph } from '$lib/engine/share';

const MOCK_ENV = {
  ASSETS: {
    fetch: async () => new Response('Not found', { status: 404 }),
  },
};

async function invokeLink(input: string | Request, init?: RequestInit): Promise<{ res: Response; body: LinkApiResponse }> {
  const req = typeof input === 'string' ? new Request(input, init) : input;
  const res = await cfWorker.fetch(req, MOCK_ENV);
  const body = (await res.json()) as LinkApiResponse;
  return { res, body };
}

function assertCorsAndNoStoreHeaders(res: Response) {
  expect(res.headers.get('access-control-allow-origin')).toBe('*');
  expect(res.headers.get('access-control-allow-methods')).toContain('GET');
  expect(res.headers.get('access-control-allow-methods')).toContain('POST');
  expect(res.headers.get('access-control-allow-methods')).toContain('OPTIONS');
  expect(res.headers.get('cache-control')).toBe('no-store');
  expect(res.headers.get('x-content-type-options')).toBe('nosniff');
}

describe('Cloudflare Worker stateless /api/link endpoint', () => {
  const USER_6FIZ_G =
    'HYxBC4JAEEb_inzn6RCU4BwTCqFTnnJYJJspFtx1KT2J_z2263u8tyKCRZB06H1My9zvQUigFWFSA2ez85qhDo2CUZ6bDoTnZ0qtvYPF-QtGTcXlegfh5UeLj5BTbIRb3Z6Kf-IcwcDi8p9FDlVF5dG57Qc';

  it('handles GET /api/link for linear, multi-branch RFD3 vs RFD4, and fan-out/fan-in workflows with round-trip verification', async () => {
    // 1. Linear chain
    const linear = await invokeLink('https://denovo.run/api/link?flow=r3:length=80~m~f');
    expect(linear.res.status).toBe(200);
    assertCorsAndNoStoreHeaders(linear.res);
    expect(linear.body.ok).toBe(true);
    expect(linear.body.errors).toEqual([]);
    expect(linear.body.nodes.map((n) => n.adapterId)).toEqual([
      'rfd3',
      'web-protein-mpnn',
      'localfold-af2',
    ]);
    expect(linear.body.edges).toHaveLength(2);
    expect(linear.body.url).toMatch(/^https:\/\/denovo\.run\/\?flow=/);
    expect(linear.body.shortUrl).toMatch(/^https:\/\/denovo\.run\/\?g=/);
    expect(isMarkdownSafeFlowUrl(linear.body.url!)).toBe(true);

    // Round-trip both url and shortUrl through share.ts & og-graph.ts
    for (const shareUrl of [linear.body.url!, linear.body.shortUrl!]) {
      const clientGraph = await decodeHashToGraph(shareUrl);
      expect(clientGraph).not.toBeNull();
      expect(clientGraph!.nodes).toHaveLength(3);
      expect(clientGraph!.edges).toHaveLength(2);

      const ogWorkflow = await decodeOgWorkflowFromQuery(shareUrl);
      expect(ogWorkflow).not.toBeNull();
      expect(ogWorkflow!.nodes).toHaveLength(3);
      expect(ogWorkflow!.edges).toHaveLength(2);
    }

    // 2. RFD3 vs RFD4 multi-branch comparison
    const branchFlow = encodeURIComponent('src:p:pdbId=1ubq~r3:length=80~cmp;src~r:length=80~cmp');
    const branch = await invokeLink(`https://denovo.run/api/link?flow=${branchFlow}`);
    expect(branch.res.status).toBe(200);
    expect(branch.body.ok).toBe(true);
    expect(branch.body.nodes.map((n) => n.adapterId)).toEqual([
      'pdb-input',
      'rfd3',
      'structure-compare',
      'rfd4-proteina',
    ]);
    expect(branch.body.edges).toHaveLength(4);
    expect(isMarkdownSafeFlowUrl(branch.body.url!)).toBe(true);
    const branchClient = await decodeHashToGraph(branch.body.url!);
    expect(branchClient!.nodes).toHaveLength(4);
    expect(branchClient!.edges).toHaveLength(4);
    const branchOg = await decodeOgWorkflowFromQuery(branch.body.shortUrl!);
    expect(branchOg!.structureTarget).toEqual({ kind: 'rcsb', pdbId: '1UBQ', cropSegments: '' });

    // 3. Fan-out/fan-in with designability-scorer
    const fanFlow = encodeURIComponent('r3:length=70~m:numSequences=4~f~s');
    const fan = await invokeLink(`https://denovo.run/api/link?flow=${fanFlow}`);
    expect(fan.res.status).toBe(200);
    expect(fan.body.ok).toBe(true);
    expect(fan.body.edges.some((e) => e.toNode === 's' && e.toPort === 'structure')).toBe(true);
    expect(fan.body.edges.some((e) => e.toNode === 's' && e.toPort === 'folds')).toBe(true);
    expect(isMarkdownSafeFlowUrl(fan.body.url!)).toBe(true);
  });

  it('handles GET /api/link for ?recipe=, ?g=, ?p= (preserving overrides), and ?url= (including hybrid #flow=)', async () => {
    // ?recipe= YAML
    const yamlRecipe = [
      'denovo: v1',
      'nodes:',
      '  gen: {"model":"rfd3","length":90}',
      '  mpnn: {"model":"web-protein-mpnn","numSequences":3}',
      'wires:',
      '  - gen.structure -> mpnn.structure',
    ].join('\n');
    const yamlRes = await invokeLink(
      `https://denovo.run/api/link?recipe=${encodeURIComponent(yamlRecipe)}`,
    );
    expect(yamlRes.res.status).toBe(200);
    expect(yamlRes.body.ok).toBe(true);
    expect(yamlRes.body.nodes).toHaveLength(2);
    expect(isMarkdownSafeFlowUrl(yamlRes.body.url!)).toBe(true);

    // ?recipe= JSON
    const jsonRecipe = JSON.stringify({
      id: 'r-json',
      name: 'JSON Workflow',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4', params: { length: 100 } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'LocalFold', params: {} },
      ],
      edges: [{ fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    });
    const jsonRecipeRes = await invokeLink(
      `https://denovo.run/api/link?recipe=${encodeURIComponent(jsonRecipe)}`,
    );
    expect(jsonRecipeRes.res.status).toBe(200);
    expect(jsonRecipeRes.body.ok).toBe(true);
    expect(jsonRecipeRes.body.nodes).toHaveLength(2);

    // ?g= compressed DAG
    const gRes = await invokeLink(`https://denovo.run/api/link?g=${USER_6FIZ_G}`);
    expect(gRes.res.status).toBe(200);
    expect(gRes.body.ok).toBe(true);
    expect(gRes.body.nodes[0].params.pdbId).toBe('6FIZ');
    expect(gRes.body.shortUrl).toMatch(/^https:\/\/denovo\.run\/\?g=/);
    expect(isMarkdownSafeFlowUrl(gRes.body.url!)).toBe(true);

    // ?p= preset with overrides -> shortUrl keeps ?p= preset-diff encoding
    const pRes = await invokeLink(
      'https://denovo.run/api/link?p=rfd4-mpnn-refold&gen.length=90&mpnn.numSequences=5',
    );
    expect(pRes.res.status).toBe(200);
    expect(pRes.body.ok).toBe(true);
    expect(pRes.body.shortUrl).toContain('?p=rfd4-mpnn-refold');
    expect(pRes.body.shortUrl).toContain('gen.length=90');
    expect(pRes.body.nodes.find((n) => n.id === 'gen')?.params.length).toBe(90);
    expect(pRes.body.nodes.find((n) => n.id === 'mpnn')?.params.numSequences).toBe(5);
    const pRoundTrip = await decodeHashToGraph(pRes.body.shortUrl!);
    expect(pRoundTrip!.nodes.find((n) => n.id === 'gen')?.params.length).toBe(90);

    // ?url= full hybrid URL
    const hybridUrl = 'https://denovo.run/?utm_source=chatgpt.com#flow=r3:length=75~m~f';
    const urlRes = await invokeLink(
      `https://denovo.run/api/link?url=${encodeURIComponent(hybridUrl)}`,
    );
    expect(urlRes.res.status).toBe(200);
    expect(urlRes.body.ok).toBe(true);
    expect(urlRes.body.nodes[0].params.length).toBe(75);
    expect(urlRes.body.url).toMatch(/^https:\/\/denovo\.run\/\?flow=/);
  });

  it('handles POST /api/link for JSON ({ flow, recipe, url, graph }), form-urlencoded, and plain text/YAML (with -> and (...) tolerance)', async () => {
    // JSON { flow } with -> and (...) tolerance normalized to Markdown-safe ?flow=
    const jsonFlow = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ flow: 'r3(length=80)->mpnn->af2' }),
    });
    expect(jsonFlow.res.status).toBe(200);
    expect(jsonFlow.body.ok).toBe(true);
    expect(isMarkdownSafeFlowUrl(jsonFlow.body.url!)).toBe(true);
    expect(jsonFlow.body.url).not.toContain('(');
    expect(jsonFlow.body.url).not.toContain('->');

    // JSON { recipe } (YAML string)
    const yamlRecipe = 'denovo: v1\nnodes:\n  r: {"model":"rfd4-proteina","length":70}\n  f: {"model":"localfold-af2"}\nwires:\n  - r.sequences -> f.sequences\n';
    const jsonRecipe = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipe: yamlRecipe }),
    });
    expect(jsonRecipe.res.status).toBe(200);
    expect(jsonRecipe.body.ok).toBe(true);
    expect(jsonRecipe.body.nodes).toHaveLength(2);

    // JSON { url }
    const jsonUrl = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: 'https://denovo.run/?p=pdb-thermompnn' }),
    });
    expect(jsonUrl.res.status).toBe(200);
    expect(jsonUrl.body.ok).toBe(true);
    expect(jsonUrl.body.shortUrl).toContain('?p=pdb-thermompnn');

    // JSON { graph }
    const jsonGraph = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        graph: {
          id: 'g1',
          name: 'Graph Payload',
          nodes: [
            { id: 'gen', adapterId: 'genie3', label: 'Genie 3', params: { length: 65 } },
            { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN', params: {} },
          ],
          edges: [{ fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' }],
        },
      }),
    });
    expect(jsonGraph.res.status).toBe(200);
    expect(jsonGraph.body.ok).toBe(true);
    expect(jsonGraph.body.nodes).toHaveLength(2);

    // application/x-www-form-urlencoded
    const formRes = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'p=rfd4-mpnn-refold&gen.length=110',
    });
    expect(formRes.res.status).toBe(200);
    expect(formRes.body.ok).toBe(true);
    expect(formRes.body.nodes.find((n) => n.id === 'gen')?.params.length).toBe(110);

    // Plain text flow DSL
    const textFlowRes = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body: 'r3:length=80~m~f',
    });
    expect(textFlowRes.res.status).toBe(200);
    expect(textFlowRes.body.ok).toBe(true);
    expect(textFlowRes.body.nodes).toHaveLength(3);

    // Plain text / YAML denovo: v1 recipe
    const textYamlRes = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'text/yaml' },
      body: yamlRecipe,
    });
    expect(textYamlRes.res.status).toBe(200);
    expect(textYamlRes.body.ok).toBe(true);
    expect(textYamlRes.body.nodes).toHaveLength(2);
  });

  it('returns structured 400, 405, 413, 415 errors and 204 OPTIONS CORS responses', async () => {
    // OPTIONS preflight
    const optRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', { method: 'OPTIONS' }),
      MOCK_ENV,
    );
    expect(optRes.status).toBe(204);
    assertCorsAndNoStoreHeaders(optRes);
    expect(optRes.headers.get('access-control-max-age')).toBe('86400');

    // 405 Method Not Allowed
    for (const method of ['PUT', 'DELETE', 'PATCH']) {
      const mRes = await invokeLink('https://denovo.run/api/link', { method });
      expect(mRes.res.status).toBe(405);
      expect(mRes.res.headers.get('allow')).toBe('GET, POST, OPTIONS');
      assertCorsAndNoStoreHeaders(mRes.res);
      expect(mRes.body).toEqual({
        ok: false,
        url: null,
        shortUrl: null,
        nodes: [],
        edges: [],
        errors: [{ code: 'METHOD_NOT_ALLOWED', message: expect.any(String) }],
      });
    }

    // 400 GET missing input
    const missingGet = await invokeLink('https://denovo.run/api/link');
    expect(missingGet.res.status).toBe(400);
    expect(missingGet.body.ok).toBe(false);
    expect(missingGet.body.url).toBeNull();
    expect(missingGet.body.shortUrl).toBeNull();
    expect(missingGet.body.nodes).toEqual([]);
    expect(missingGet.body.edges).toEqual([]);
    expect(missingGet.body.errors[0].code).toBe('MISSING_INPUT');

    // 400 Domain validation errors via GET
    const domainErrorCases: Array<[string, string]> = [
      ['?flow=nonexistent-model', 'UNKNOWN_MODEL'],
      ['?flow=r3~s', 'MISSING_REQUIRED_INPUT'],
      ['?flow=a:r3~b:r3~a', 'CYCLE_DETECTED'],
      ['?flow=rfdiffusion~af2', 'INCOMPATIBLE_PORTS'],
      ['?flow=p:pdbId=BAD', 'INVALID_PARAM'],
      ['?recipe=not-a-valid-recipe', 'INVALID_RECIPE'],
      ['?g=invalid_compressed_dag_payload', 'INVALID_COMPRESSED_DAG'],
    ];
    for (const [qs, expectedCode] of domainErrorCases) {
      const errRes = await invokeLink(`https://denovo.run/api/link${qs}`);
      expect(errRes.res.status, qs).toBe(400);
      assertCorsAndNoStoreHeaders(errRes.res);
      expect(errRes.body.ok).toBe(false);
      expect(errRes.body.url).toBeNull();
      expect(errRes.body.shortUrl).toBeNull();
      expect(errRes.body.nodes).toEqual([]);
      expect(errRes.body.edges).toEqual([]);
      expect(errRes.body.errors.some((e) => e.code === expectedCode), `${qs} errors: ${JSON.stringify(errRes.body.errors)}`).toBe(true);
    }

    // 400 POST malformed JSON
    const badJson = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{"flow": ',
    });
    expect(badJson.res.status).toBe(400);
    expect(badJson.body.errors[0].code).toBe('INVALID_JSON');

    // 400 POST empty body
    const emptyBody = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '   ',
    });
    expect(emptyBody.res.status).toBe(400);
    expect(emptyBody.body.errors[0].code).toBe('EMPTY_BODY');

    // 400 POST empty graph ({ graph: { nodes: [], edges: [] } })
    const emptyGraph = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: { nodes: [], edges: [] } }),
    });
    expect(emptyGraph.res.status).toBe(400);
    expect(emptyGraph.body.errors[0].code).toBe('EMPTY_FLOW');

    // 400 POST missing JSON fields & invalid field types
    const missingJsonKey = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ unrelated: true }),
    });
    expect(missingJsonKey.res.status).toBe(400);
    expect(missingJsonKey.body.errors[0].code).toBe('MISSING_INPUT');

    const badType = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ flow: 42 }),
    });
    expect(badType.res.status).toBe(400);
    expect(badType.body.errors[0].code).toBe('INVALID_INPUT');

    // 415 Unsupported Media Type (application/xml & multipart/form-data)
    for (const ct of ['application/xml', 'multipart/form-data; boundary=abc']) {
      const unsupported = await invokeLink('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': ct },
        body: '<flow>r3~m~f</flow>',
      });
      expect(unsupported.res.status, ct).toBe(415);
      assertCorsAndNoStoreHeaders(unsupported.res);
      expect(unsupported.body.errors[0].code).toBe('UNSUPPORTED_MEDIA_TYPE');
    }

    // 413 Payload Too Large (actual oversized body)
    const oversized = await invokeLink(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: 'a'.repeat(MAX_INFLATED_BYTES + 10),
      }),
    );
    expect(oversized.res.status).toBe(413);
    assertCorsAndNoStoreHeaders(oversized.res);
    expect(oversized.body.errors[0].code).toBe('PAYLOAD_TOO_LARGE');

    // Direct handleLinkRequest call works standalone as well
    const directRes = await handleLinkRequest(new Request('https://denovo.run/api/link?flow=r3~m~f'));
    expect(directRes.status).toBe(200);
    const directBody = (await directRes.json()) as LinkApiResponse;
    expect(directBody.ok).toBe(true);
  });

  it('bundles cf-worker.ts cleanly with esbuild --platform=browser (zero $lib/ or Node builtins)', () => {
    const here = path.dirname(fileURLToPath(import.meta.url));
    const entry = path.resolve(here, '../og/cf-worker.ts');
    const result = esbuild.buildSync({
      entryPoints: [entry],
      bundle: true,
      write: false,
      format: 'esm',
      target: 'es2022',
      platform: 'browser',
      minify: true,
      legalComments: 'none',
      metafile: true,
    });
    expect(result.errors).toEqual([]);
    expect(result.outputFiles[0].contents.length).toBeGreaterThan(10_000);
    const inputs = Object.keys(result.metafile?.inputs ?? {});
    expect(inputs.some((p) => p.endsWith('src/lib/og/cf-worker.ts'))).toBe(true);
    expect(inputs.some((p) => p.endsWith('src/lib/engine/flow-dsl.ts'))).toBe(true);
    expect(inputs.some((p) => p.endsWith('src/lib/engine/share.ts'))).toBe(false);
  });

  it('round-trips all 8 WORKFLOW_PRESETS via /api/link shortUrl, rejects oversized inline pdbText, handles text/plain hybrid queries, and enforces GET flow MAX_HASH_CHARS', async () => {
    // 1. All 8 WORKFLOW_PRESETS round-trip identically from /api/link?p=<id> shortUrl through share.ts & og-graph.ts
    expect(WORKFLOW_PRESETS).toHaveLength(8);
    for (const preset of WORKFLOW_PRESETS) {
      const { res, body } = await invokeLink(`https://denovo.run/api/link?p=${preset.id}`);
      expect(res.status, preset.id).toBe(200);
      expect(body.ok, preset.id).toBe(true);
      expect(body.shortUrl, preset.id).toBe(`https://denovo.run/?p=${preset.id}`);

      const clientGraph = await decodeHashToGraph(body.shortUrl!);
      const ogWf = await decodeOgWorkflowFromQuery(body.shortUrl!);
      expect(clientGraph, preset.id).not.toBeNull();
      expect(ogWf, preset.id).not.toBeNull();

      expect(ogWf!.nodes.map((n) => ({ id: n.id, adapterId: n.adapterId, label: n.label }))).toEqual(
        clientGraph!.nodes.map((n) => ({ id: n.id, adapterId: n.adapterId, label: n.label })),
      );
      expect(ogWf!.edges).toEqual(clientGraph!.edges);
      expect(body.edges).toEqual(clientGraph!.edges);
      for (let i = 0; i < clientGraph!.nodes.length; i++) {
        expect(ogWf!.nodes[i].params, `${preset.id}.${clientGraph!.nodes[i].id}`).toEqual(
          clientGraph!.nodes[i].params,
        );
      }

      if (preset.id === 'rfd4-branch-compare') {
        expect(body.edges).toHaveLength(7);
        expect(ogWf!.edges).toHaveLength(7);
      }
      if (preset.id === 'rfd4-af2-vs-rf3') {
        expect(body.edges).toHaveLength(6);
        expect(ogWf!.edges).toHaveLength(6);
      }
    }

    // 2. POST /api/link with inline pdbText > 1500 chars returns HTTP 400 INLINE_PDB_TOO_LARGE
    const largePdb =
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n'.repeat(30);
    expect(largePdb.length).toBeGreaterThan(1500);
    const largePdbRes = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        graph: {
          nodes: [
            {
              id: 'p1',
              adapterId: 'pdb-input',
              label: 'Structure Input',
              params: { mode: 'custom', pdbText: largePdb, filename: 'big.pdb' },
            },
            {
              id: 'm1',
              adapterId: 'web-protein-mpnn',
              label: 'ProteinMPNN',
              params: { numSequences: 2 },
            },
          ],
          edges: [{ fromNode: 'p1', fromPort: 'structure', toNode: 'm1', toPort: 'structure' }],
        },
      }),
    });
    expect(largePdbRes.res.status).toBe(400);
    assertCorsAndNoStoreHeaders(largePdbRes.res);
    expect(largePdbRes.body.ok).toBe(false);
    expect(largePdbRes.body.url).toBeNull();
    expect(largePdbRes.body.shortUrl).toBeNull();
    expect(largePdbRes.body.nodes).toEqual([]);
    expect(largePdbRes.body.edges).toEqual([]);
    expect(largePdbRes.body.errors.some((e) => e.code === 'INLINE_PDB_TOO_LARGE')).toBe(true);

    // 3. POST /api/link text/plain relative hybrid query (?utm_source=chatgpt.com#flow=r3~m~f) -> 200 ok:true
    const relativeHybrid = await invokeLink('https://denovo.run/api/link', {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body: '?utm_source=chatgpt.com#flow=r3~m~f',
    });
    expect(relativeHybrid.res.status).toBe(200);
    expect(relativeHybrid.body.ok).toBe(true);
    expect(relativeHybrid.body.nodes.map((n) => n.adapterId)).toEqual([
      'rfd3',
      'web-protein-mpnn',
      'localfold-af2',
    ]);

    // 4. GET /api/link?flow=... exceeding MAX_HASH_CHARS -> 400 HASH_TOO_LARGE
    const hugeFlow = 'r3:_label="' + 'a'.repeat(MAX_HASH_CHARS + 10) + '"~m~f';
    const hugeRes = await invokeLink(
      `https://denovo.run/api/link?flow=${encodeURIComponent(hugeFlow)}`,
    );
    expect(hugeRes.res.status).toBe(400);
    assertCorsAndNoStoreHeaders(hugeRes.res);
    expect(hugeRes.body.ok).toBe(false);
    expect(hugeRes.body.errors.some((e) => e.code === 'HASH_TOO_LARGE')).toBe(true);
  });
});
