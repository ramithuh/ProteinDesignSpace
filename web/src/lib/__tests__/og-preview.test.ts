import { describe, expect, it } from 'vitest';
import { decodeOgWorkflowFromQuery } from '$lib/og/og-graph';
import { parseCifForOgPreview, parsePdbForOgPreview } from '$lib/og/edge-structure';
import { OG_HEIGHT, OG_WIDTH, renderDefaultOgPng, renderWorkflowOgPng } from '$lib/og/render-og-card';
import cfWorker, { injectOgMetaIntoHtml } from '$lib/og/cf-worker';
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
});
