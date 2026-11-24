import { describe, expect, it } from 'vitest';
import { decodeOgWorkflowFromQuery } from '$lib/og/og-graph';
import { parseCifForOgPreview, parsePdbForOgPreview } from '$lib/og/edge-structure';
import { OG_HEIGHT, OG_WIDTH, renderDefaultOgPng, renderWorkflowOgPng } from '$lib/og/render-og-card';
import cfWorker, { injectOgMetaIntoHtml } from '$lib/og/cf-worker';
import { WORKFLOW_PRESETS } from '$lib/engine/registry';
import { buildShareableWorkflowUrl, decodeHashToGraph } from '$lib/engine/share';
import { isSharedWorkflowHash } from '$lib/ui/page-state-helpers';

const PNG_SIG = [137, 80, 78, 71, 13, 10, 26, 10];

function assertValidOgPng(bytes: Uint8Array) {
  expect(bytes.length).toBeGreaterThan(2000);
  expect([...bytes.subarray(0, 8)]).toEqual(PNG_SIG);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  expect(dv.getUint32(16, false)).toBe(OG_WIDTH);
  expect(dv.getUint32(20, false)).toBe(OG_HEIGHT);
}

describe('Social media OG preview cards & edge worker', () => {
  const USER_6FIZ_G =
    'HYxBC4JAEEb_inzn6RCU4BwTCqFTnnJYJJspFtx1KT2J_z2263u8tyKCRZB06H1My9zvQUigFWFSA2ez85qhDo2CUZ6bDoTnZ0qtvYPF-QtGTcXlegfh5UeLj5BTbIRb3Z6Kf-IcwcDi8p9FDlVF5dG57Qc';

  const USER_AFDB_PIPELINE_G =
    'dZFRS8MwFIX_SjlPCreyVqlb3iZjIKg4fLOEkDapBtK0Nq0yxv67pBvbS_t2yT333O_cHODA8hytKoRx7dCLBIQWdEDdKA0GadtvWTVWxaoAYXCm7Zr-WYFht3r73D2CUBmrnayDfL2NT8_xNomDhRW_2V1pKhwJ6-3miUXvm_gliW5OsltwyvGnCxFstXGibp0bKepAUcna2D0Yzu3QBcEN9Yf-GbQrtQdbHoOJbUppA6mQVTo6VKDDuDaNXgNKlIzblPbmy8nCWNPvhS-bTnej3oNzgp64iO-7oeyHToOmYf1FMJvHX4BpgvXancnSWBUmZ-FPgjA8jz6f_IrPw_-zPM8WlC045fcP5yJLz8UqXVC8WnJ-_Ac';

  it('decodes ?g= and ?p= links in both client share.ts and zero-dep edge og-graph.ts', async () => {
    expect(isSharedWorkflowHash(`?g=${USER_6FIZ_G}`)).toBe(true);
    expect(isSharedWorkflowHash('?p=genie3-mpnn-af2')).toBe(true);
    expect(buildShareableWorkflowUrl('https://denovo.run', '/', '#p=genie3-mpnn-af2')).toBe(
      'https://denovo.run/?p=genie3-mpnn-af2',
    );

    const clientGraph = await decodeHashToGraph(`?g=${USER_6FIZ_G}`);
    expect(clientGraph).not.toBeNull();
    expect(clientGraph!.nodes[0].params.pdbId).toBe('6FIZ');
    expect(clientGraph!.nodes[0].params.cropSegments).toBe('C, GLY');

    const edgeSingle = await decodeOgWorkflowFromQuery(`?g=${USER_6FIZ_G}`);
    expect(edgeSingle).not.toBeNull();
    expect(edgeSingle!.nodes).toHaveLength(1);
    expect(edgeSingle!.title).toContain('RCSB 6FIZ');
    expect(edgeSingle!.structureTarget).toEqual({
      kind: 'rcsb',
      pdbId: '6FIZ',
      cropSegments: 'C, GLY',
    });

    const edgePipe = await decodeOgWorkflowFromQuery(`?g=${USER_AFDB_PIPELINE_G}`);
    expect(edgePipe).not.toBeNull();
    expect(edgePipe!.nodes).toHaveLength(4);
    expect(edgePipe!.edges.length).toBeGreaterThanOrEqual(3);
    expect(edgePipe!.structureTarget).toEqual({
      kind: 'afdb',
      uniprotId: 'Q9NZQ7',
      cropSegments: '',
    });

    const edgePreset = await decodeOgWorkflowFromQuery('?p=genie3-mpnn-af2&mpnn.numSequences=8');
    expect(edgePreset).not.toBeNull();
    expect(edgePreset!.nodes.find((n) => n.id === 'mpnn')?.paramChip).toContain('8 seqs');
  });

  it('parses PDB/mmCIF crops (including ligands & numeric chains) and renders valid 1200x630 PNGs', async () => {
    const samplePdb = [
      'ATOM      1  CA  ALA C   1       0.000   0.000   0.000  1.00 92.00           C',
      'ATOM      2  CA  GLY C   2       3.800   0.500   0.200  1.00 91.00           C',
      'ATOM      3  CA  VAL C   3       7.200   1.900   1.100  1.00 88.00           C',
      'ATOM      4  CA  LEU C   4       9.800   4.100   2.500  1.00 84.00           C',
      'ATOM      5  CA  MET A   1      25.000   0.000   0.000  1.00 50.00           C',
      'HETATM    6  N   GLY C 301       5.000   2.000   0.500  1.00 20.00           N',
      'HETATM    7  CA  GLY C 301       6.200   2.400   0.900  1.00 20.00           C',
      'END',
    ].join('\n');

    const parsed = parsePdbForOgPreview(samplePdb, 'RCSB 6FIZ', 'C, GLY', false);
    expect(parsed).not.toBeNull();
    expect(parsed!.residueCount).toBe(4);
    expect(parsed!.chains.map((c) => c.id)).toEqual(['C']);
    expect(parsed!.ligands).toHaveLength(1);
    expect(parsed!.ligands[0].name).toBe('GLY');

    const numericCif = [
      'data_6GRG',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM CA . ALA 2 13 0.0 0.0 0.0',
      'ATOM CA . GLY 2 14 3.8 0.2 0.1',
      'ATOM CA . VAL 2 15 7.5 1.0 0.4',
    ].join('\n');
    const parsedCif = parseCifForOgPreview(numericCif, 'RCSB 6GRG', '2', false);
    expect(parsedCif).not.toBeNull();
    expect(parsedCif!.residueCount).toBe(3);

    const wfSingle = (await decodeOgWorkflowFromQuery(`?g=${USER_6FIZ_G}`))!;
    assertValidOgPng(await renderWorkflowOgPng(wfSingle, parsed));

    const wfPipe = (await decodeOgWorkflowFromQuery(`?g=${USER_AFDB_PIPELINE_G}`))!;
    assertValidOgPng(await renderWorkflowOgPng(wfPipe, parsed));
    assertValidOgPng(await renderDefaultOgPng());
  });

  it('injects dynamic og:title, og:description, og:image, and twitter:card in Cloudflare Pages worker', async () => {
    const baseHtml = `<!doctype html><html><head><meta property="og:title" content="Default" /><meta property="og:image" content="https://denovo.run/og-default.png" /></head><body>ok</body></html>`;
    const rewritten = injectOgMetaIntoHtml(baseHtml, {
      title: 'RCSB 6FIZ (RCSB 6FIZ · C, GLY) — denovo.run',
      description: 'Test description',
      imageUrl: `https://denovo.run/api/og?g=${USER_6FIZ_G}`,
      pageUrl: `https://denovo.run/?g=${USER_6FIZ_G}`,
    });
    expect(rewritten).toContain(`<meta property="og:image" content="https://denovo.run/api/og?g=${USER_6FIZ_G}" />`);
    expect(rewritten).toContain('<meta name="twitter:card" content="summary_large_image" />');
    expect(rewritten).not.toContain('content="Default"');

    const res = await cfWorker.fetch(
      new Request(`https://denovo.run/?g=${USER_6FIZ_G}`),
      {
        ASSETS: {
          fetch: async (input) => {
            const reqUrl = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url);
            // Simulate real Cloudflare Pages ASSETS redirecting /index.html -> /
            if (reqUrl.pathname === '/index.html') {
              return new Response(null, { status: 308, headers: { Location: '/' } });
            }
            return new Response(baseHtml, { status: 200, headers: { 'Content-Type': 'text/html; charset=utf-8' } });
          },
        },
      },
    );
    const html = await res.text();
    expect(res.status).toBe(200);
    expect(html).toContain('RCSB 6FIZ');
    expect(html).toContain(`/api/og?g=${USER_6FIZ_G}`);

    const ogRes = await cfWorker.fetch(new Request('https://denovo.run/api/og?p=genie3-mpnn-af2'), {
      ASSETS: { fetch: async () => new Response('nf', { status: 404 }) },
    });
    expect(ogRes.status).toBe(200);
    expect(ogRes.headers.get('content-type')).toBe('image/png');
    assertValidOgPng(new Uint8Array(await ogRes.arrayBuffer()));

    // Scanner probes return crisp 404 instead of SPA 200 index.html
    for (const probe of ['/.env', '/.env.bak', '/.git/config', '/sites/default/settings.php.old', '/config.js']) {
      const probeRes = await cfWorker.fetch(new Request(`https://denovo.run${probe}`), {
        ASSETS: { fetch: async () => new Response('<html>SPA</html>', { status: 200 }) },
      });
      expect(probeRes.status, probe).toBe(404);
    }
  });

  it('decodes ?flow= and ?recipe= (including hybrid URLs, RFD3 vs RFD4, fan-out/fan-in, AFDB, and $-labels) and renders 1200x630 OG PNGs', async () => {
    // 1. Linear ?flow=
    const linear = await decodeOgWorkflowFromQuery('?flow=r3:length=80~m~f');
    expect(linear).not.toBeNull();
    expect(linear!.nodes).toHaveLength(3);
    expect(linear!.edges).toHaveLength(2);
    expect(linear!.structureTarget).toBeNull();
    assertValidOgPng(await renderWorkflowOgPng(linear!));

    // 2. Multi-branch RFD3 vs RFD4 with 1UBQ structure target
    const branchQuery = '?flow=src:p:pdbId=1UBQ~r3:length=80~cmp;src~r:length=80~cmp';
    const branch = await decodeOgWorkflowFromQuery(branchQuery);
    expect(branch).not.toBeNull();
    expect(branch!.nodes).toHaveLength(4);
    expect(branch!.edges).toHaveLength(4);
    expect(branch!.structureTarget).toEqual({
      kind: 'rcsb',
      pdbId: '1UBQ',
      cropSegments: '',
    });
    assertValidOgPng(await renderWorkflowOgPng(branch!));

    // 3. Fan-out/fan-in + custom _label with $ special replacement chars
    const fanQuery = '?flow=' + encodeURIComponent('r3:length=70,_label="Backbone $100 $& Gen"~m:numSequences=4~f~s');
    const fan = await decodeOgWorkflowFromQuery(fanQuery);
    expect(fan).not.toBeNull();
    expect(fan!.nodes).toHaveLength(4);
    expect(fan!.nodes[0].label).toBe('Backbone $100 $& Gen');
    // designability-scorer receives both folds and structure (fan-out/fan-in)
    expect(fan!.edges.some((e) => e.toNode === 's' && e.toPort === 'structure')).toBe(true);
    expect(fan!.edges.some((e) => e.toNode === 's' && e.toPort === 'folds')).toBe(true);
    assertValidOgPng(await renderWorkflowOgPng(fan!));

    // 4. AFDB target with cropSegments
    const afdb = await decodeOgWorkflowFromQuery('?flow=src:p:uniprotId=P69905,cropSegments=A10-50~m');
    expect(afdb).not.toBeNull();
    expect(afdb!.structureTarget).toEqual({
      kind: 'afdb',
      uniprotId: 'P69905',
      cropSegments: 'A10-50',
    });
    assertValidOgPng(await renderWorkflowOgPng(afdb!));

    // 5. ?recipe= YAML & JSON
    const yamlRecipe = [
      'denovo: v1',
      'name: "Custom YAML Pipeline"',
      'nodes:',
      '  gen: {"model":"rfd3","length":90}',
      '  mpnn: {"model":"web-protein-mpnn","numSequences":3}',
      'wires:',
      '  - gen.structure -> mpnn.structure',
    ].join('\n');
    const decodedYaml = await decodeOgWorkflowFromQuery(`?recipe=${encodeURIComponent(yamlRecipe)}`);
    expect(decodedYaml).not.toBeNull();
    expect(decodedYaml!.title).toContain('Custom YAML Pipeline');
    expect(decodedYaml!.nodes).toHaveLength(2);
    expect(decodedYaml!.edges).toHaveLength(1);
    assertValidOgPng(await renderWorkflowOgPng(decodedYaml!));

    const jsonRecipe = JSON.stringify({
      id: 'j1',
      name: 'JSON Recipe',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4', params: { length: 85 } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AF2', params: {} },
      ],
      edges: [{ fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    });
    const decodedJson = await decodeOgWorkflowFromQuery(`?recipe=${encodeURIComponent(jsonRecipe)}`);
    expect(decodedJson).not.toBeNull();
    expect(decodedJson!.title).toContain('JSON Recipe');
    expect(decodedJson!.nodes).toHaveLength(2);
    assertValidOgPng(await renderWorkflowOgPng(decodedJson!));

    // 6. Hybrid ?utm_...#flow= and full URLs
    const hybridFlow = await decodeOgWorkflowFromQuery('?utm_source=chatgpt.com#flow=r3:length=80~m~f');
    expect(hybridFlow).not.toBeNull();
    expect(hybridFlow!.nodes).toHaveLength(3);

    const fullHybridUrl = await decodeOgWorkflowFromQuery(
      `https://denovo.run/?utm_source=chatgpt.com#recipe=${encodeURIComponent(yamlRecipe)}`,
    );
    expect(fullHybridUrl).not.toBeNull();
    expect(fullHybridUrl!.title).toContain('Custom YAML Pipeline');

    // 7. Invalid ?flow= and ?recipe= return null (so /api/og falls back cleanly)
    expect(await decodeOgWorkflowFromQuery('?flow=')).toBeNull();
    expect(await decodeOgWorkflowFromQuery('?flow=unknown-model-xyz')).toBeNull();
    expect(await decodeOgWorkflowFromQuery('?recipe=invalid-yaml')).toBeNull();
    expect(await decodeOgWorkflowFromQuery('?recipe=' + encodeURIComponent('{"nodes":[],"edges":[]}'))).toBeNull();
  });

  it('serves /api/og PNG and injects OG meta tags on GET / and /index.html for ?flow= and ?recipe= (including $ labels)', async () => {
    const baseHtml =
      '<!doctype html><html><head><title>denovo.run</title><meta property="og:title" content="Default" /><meta name="twitter:card" content="summary" /></head><body>ok</body></html>';
    const assetsEnv = {
      ASSETS: {
        fetch: async (input: RequestInfo | URL) => {
          const reqUrl = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url);
          if (reqUrl.pathname === '/index.html') {
            return new Response(null, { status: 308, headers: { Location: '/' } });
          }
          return new Response(baseHtml, { status: 200, headers: { 'Content-Type': 'text/html; charset=utf-8' } });
        },
      },
    };

    // /api/og?flow=...
    const ogFlowRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/og?flow=r3:length=80~m~f'),
      assetsEnv,
    );
    expect(ogFlowRes.status).toBe(200);
    expect(ogFlowRes.headers.get('content-type')).toBe('image/png');
    assertValidOgPng(new Uint8Array(await ogFlowRes.arrayBuffer()));

    // /api/og?recipe=...
    const yamlRecipe = 'denovo: v1\nnodes:\n  g: {"model":"rfd3","length":75}\n';
    const ogRecipeRes = await cfWorker.fetch(
      new Request(`https://denovo.run/api/og?recipe=${encodeURIComponent(yamlRecipe)}`),
      assetsEnv,
    );
    expect(ogRecipeRes.status).toBe(200);
    expect(ogRecipeRes.headers.get('content-type')).toBe('image/png');
    assertValidOgPng(new Uint8Array(await ogRecipeRes.arrayBuffer()));

    // /api/og with invalid ?flow= falls open to default 1200x630 PNG
    const ogBadFlowRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/og?flow=not-a-real-model'),
      assetsEnv,
    );
    expect(ogBadFlowRes.status).toBe(200);
    expect(ogBadFlowRes.headers.get('content-type')).toBe('image/png');
    assertValidOgPng(new Uint8Array(await ogBadFlowRes.arrayBuffer()));

    // GET /?utm_source=chatgpt.com&flow=... with $ and ' in _label
    const trickyFlow = encodeURIComponent("r3:length=80,_label=\"A's $100 $& Test\"~m~f");
    const trickyReqUrl = new URL(`https://denovo.run/?utm_source=chatgpt.com&flow=${trickyFlow}`);
    const pageRes = await cfWorker.fetch(new Request(trickyReqUrl.toString()), assetsEnv);
    expect(pageRes.status).toBe(200);
    const html = await pageRes.text();
    expect(html).toContain('A&#39;s $100 $&amp; Test');
    expect(html).toContain(
      `<meta property="og:image" content="https://denovo.run/api/og${trickyReqUrl.search.replace(/&/g, '&amp;')}" />`,
    );
    expect(html).toContain('<meta name="twitter:card" content="summary_large_image" />');
    expect(html).not.toContain('content="Default"');
    // Ensure </head> was not duplicated or corrupted by $&
    expect(html.match(/<\/head>/g)).toHaveLength(1);

    // GET /index.html?recipe=...
    const indexRecipeRes = await cfWorker.fetch(
      new Request(`https://denovo.run/index.html?recipe=${encodeURIComponent(yamlRecipe)}`),
      assetsEnv,
    );
    expect(indexRecipeRes.status).toBe(200);
    const recipeHtml = await indexRecipeRes.text();
    expect(recipeHtml).toContain(`/api/og?recipe=${encodeURIComponent(yamlRecipe)}`);
    expect(recipeHtml).toContain('RFdiffusion3');
  });

  it('decodes all 8 WORKFLOW_PRESETS in parity with share.ts, infers capMode=fixed on maxSequences override, and blocks prototype keys', async () => {
    expect(WORKFLOW_PRESETS).toHaveLength(8);

    for (const preset of WORKFLOW_PRESETS) {
      const query = `?p=${preset.id}`;
      const clientGraph = await decodeHashToGraph(query);
      const ogWf = await decodeOgWorkflowFromQuery(query);
      expect(clientGraph, preset.id).not.toBeNull();
      expect(ogWf, preset.id).not.toBeNull();

      expect(ogWf!.nodes.map((n) => ({ id: n.id, adapterId: n.adapterId, label: n.label }))).toEqual(
        clientGraph!.nodes.map((n) => ({ id: n.id, adapterId: n.adapterId, label: n.label })),
      );
      expect(ogWf!.edges).toEqual(clientGraph!.edges);
      for (let i = 0; i < clientGraph!.nodes.length; i++) {
        expect(ogWf!.nodes[i].params, `${preset.id}.${clientGraph!.nodes[i].id}`).toEqual(
          clientGraph!.nodes[i].params,
        );
      }

      if (preset.id === 'rfd4-branch-compare') {
        expect(ogWf!.edges).toHaveLength(7);
      }
      if (preset.id === 'rfd4-af2-vs-rf3') {
        expect(ogWf!.edges).toHaveLength(6);
      }
    }

    // maxSequences override without explicit capMode infers capMode='fixed' and renders 'Cap 4'
    const capQuery = '?p=rfd4-mpnn-refold&fold.maxSequences=4';
    const capClient = await decodeHashToGraph(capQuery);
    const capOg = await decodeOgWorkflowFromQuery(capQuery);
    expect(capOg).not.toBeNull();
    const foldOg = capOg!.nodes.find((n) => n.id === 'fold');
    const foldClient = capClient!.nodes.find((n) => n.id === 'fold');
    expect(foldOg?.params.capMode).toBe('fixed');
    expect(foldOg?.params.maxSequences).toBe(4);
    expect(foldOg?.params).toEqual(foldClient?.params);
    expect(foldOg?.paramChip).toContain('Cap 4');

    // __proto__ / constructor / prototype overrides must be ignored without polluting Object.prototype
    const protoQuery =
      '?p=rfd4-mpnn-refold&gen.prototype=polluted&gen.__proto__=polluted&gen.constructor=polluted';
    const protoOg = await decodeOgWorkflowFromQuery(protoQuery);
    expect(protoOg).not.toBeNull();
    const genParams = protoOg!.nodes.find((n) => n.id === 'gen')!.params;
    expect(Object.hasOwn(genParams, 'prototype')).toBe(false);
    expect(Object.hasOwn(genParams, 'constructor')).toBe(false);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });
});
