import { describe, expect, it } from 'vitest';

import {
  assertMarkdownAutolinkerSafe,
  assertValidOgPngBytes,
  cfWorker,
  createMockCfEnv,
  extractWorkflowExampleUrls,
  generateLiveLlmsTxt,
  graphUtilsNs,
  ogGraphNs,
  shareNs,
} from './helpers';

describe('Tier 4 — Realistic End-to-End Protein Design Scenarios (7 cases)', () => {
  it('T4-01: Scenario 1 — De Novo Monomer Benchmark (Genie 3 vs RFD4-Proteina)', async () => {
    const flow =
      'g3:g:length=80~m1:m:family=ca,numSequences=4~f1:f:model=monomer~s1:s;' +
      'r4:r:length=80~f2:f:model=monomer~s2:s;' +
      'g3~cmp:c;r4~cmp';

    const env = createMockCfEnv();
    const linkRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow }),
      }),
      env,
    );
    expect(linkRes.status).toBe(200);
    const link = (await linkRes.json()) as {
      ok: boolean;
      url: string;
      shortUrl: string;
      nodes: unknown[];
      errors: unknown[];
    };
    expect(link.ok).toBe(true);
    expect(link.errors).toEqual([]);
    expect(link.nodes).toHaveLength(8);
    assertMarkdownAutolinkerSafe(link.url);

    const pageRes = await cfWorker.fetch(new Request(link.url), env);
    expect(pageRes.status).toBe(200);
    const pageHtml = await pageRes.text();
    expect(pageHtml).toContain('/api/og?flow=');

    const ogRes = await cfWorker.fetch(
      new Request(`https://denovo.run/api/og${new URL(link.url).search}`),
      env,
    );
    expect(ogRes.status).toBe(200);
    assertValidOgPngBytes(new Uint8Array(await ogRes.arrayBuffer()), 1200, 630);

    const clientGraph = await shareNs.decodeHashToGraph(`?${new URL(link.url).searchParams.toString()}`);
    expect(clientGraph).not.toBeNull();
    expect(clientGraph!.nodes).toHaveLength(8);
    expect(graphUtilsNs.validateGraph(clientGraph!)).toEqual([]);
  });

  it('T4-02: Scenario 2 — PD-L1 PPI Binder Design with Hotspots & Cross-Folder Validation (5O45)', async () => {
    const flow =
      'src:p:mode=pdb-id,pdbId=5O45,cropSegments=A17-132~' +
      'gen:r:mode=binder,lockChain=A,hotspots=A54-56,length=75~' +
      'mpnn:m:family=solublempnn,numSequences=4~' +
      'tk:tk:topK=2~' +
      'f_af2:f:model=multimer~s_af2:s;' +
      'tk~f_b2:f:model=boltz2~s_b2:s;' +
      'gen~s_af2;gen~s_b2;gen~cmp:c;f_af2~cmp;f_b2~cmp';

    const env = createMockCfEnv();
    const linkRes = await cfWorker.fetch(
      new Request(`https://denovo.run/api/link?flow=${encodeURIComponent(flow)}`),
      env,
    );
    expect(linkRes.status).toBe(200);
    const link = (await linkRes.json()) as {
      ok: boolean;
      shortUrl: string;
      nodes: unknown[];
    };
    expect(link.ok).toBe(true);
    expect(link.nodes).toHaveLength(9);

    const ogWf = await ogGraphNs.decodeOgWorkflowFromQuery(`?flow=${encodeURIComponent(flow)}`);
    expect(ogWf?.structureTarget).toEqual({
      kind: 'rcsb',
      pdbId: '5O45',
      cropSegments: 'A17-132',
    });

    const decoded = await shareNs.decodeHashToGraph(
      `?${new URL(link.shortUrl).searchParams.toString()}`,
    );
    expect(decoded).not.toBeNull();
    expect(graphUtilsNs.validateGraph(decoded!)).toEqual([]);
  });

  it('T4-03: Scenario 3 — All-Atom Catalytic Motif Scaffolding (rfd3 + ligandmpnn + gnina + boltz2)', async () => {
    const flow =
      'src:p:mode=pdb-id,pdbId=1PRW~' +
      'gen:r3:mode=motif,contig=10-20/A16-35/10-20,steps=50~' +
      'lm:lm:useLigandContext=true,numSequences=2~' +
      'fold:f:model=boltz2~' +
      'sc:s;gen~sc;gen~dock:gn:taskMode=score';

    const env = createMockCfEnv();
    const res = await cfWorker.fetch(
      new Request('https://denovo.run/api/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow }),
      }),
      env,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      ok: boolean;
      url: string;
      nodes: Array<{ id: string; adapterId: string; params: Record<string, unknown> }>;
    };
    expect(body.ok).toBe(true);
    expect(body.nodes).toHaveLength(6);
    const linkGen = body.nodes.find((n) => n.id === 'gen')!;
    expect(linkGen.params.contig).toBe('10-20/A16-35/10-20');

    const g = await shareNs.decodeHashToGraph(`?flow=${encodeURIComponent(flow)}`);
    expect(g).not.toBeNull();
    const genNode = g!.nodes.find((n) => n.id === 'gen')!;
    expect(genNode.params.contig).toBe('10-20/A16-35/10-20');
    expect(g!.nodes.map((n) => n.adapterId)).toEqual(
      expect.arrayContaining([
        'pdb-input',
        'rfd3',
        'ligandmpnn',
        'localfold-af2',
        'designability-scorer',
        'gnina',
      ]),
    );
    expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
  });

  it('T4-04: Scenario 4 — Antibody CDR Redesign & Sequence Naturalness (antifold-esmif1 + esmc + dyna1 + ef2-fast-300m)', async () => {
    const flow =
      'src:p:mode=pdb-id,pdbId=4JHW~' +
      'ab:af:variant=antifold,numbering=imgt,regions="CDR1 CDR2 CDR3"~' +
      'fold:f:model=ef2-fast-300m,msaMode=single~' +
      'sc:s;ab~ec:ec:mode=masked;ab~dy:dy:inputMode=seq-only;src~sc';

    const g = await shareNs.decodeHashToGraph(`?flow=${encodeURIComponent(flow)}`);
    expect(g).not.toBeNull();
    expect(g!.nodes).toHaveLength(6);
    const abNode = g!.nodes.find((n) => n.id === 'ab')!;
    expect(abNode.params.regions).toBe('CDR1 CDR2 CDR3');
    expect(graphUtilsNs.validateGraph(g!)).toEqual([]);

    const ogWf = await ogGraphNs.decodeOgWorkflowFromQuery(`?flow=${encodeURIComponent(flow)}`);
    const ogAbNode = ogWf?.nodes.find((n) => n.id === 'ab');
    expect(ogAbNode?.params.regions).toBe('CDR1 CDR2 CDR3');

    const env = createMockCfEnv();
    const ogRes = await cfWorker.fetch(
      new Request(`https://denovo.run/api/og?flow=${encodeURIComponent(flow)}`),
      env,
    );
    expect(ogRes.status).toBe(200);
    assertValidOgPngBytes(new Uint8Array(await ogRes.arrayBuffer()), 1200, 630);
  });

  it('T4-05: Scenario 5 — Stability Engineering & Multi-Structure Comparison (thermompnn + lasermpnn + protpardelle on AFDB P69905)', async () => {
    const flow =
      'src:p:mode=alphafold-db,uniprotId=P69905~' +
      'tm:tm:bfactorAggregate=best-stabilizing;' +
      'src~la:la:numSequences=2~cmp:c:layout=side-by-side;' +
      'src~pp:pp:taskMode=partial,numSteps=100,rewindSteps=40~cmp';

    const env = createMockCfEnv();
    const linkRes = await cfWorker.fetch(
      new Request(`https://denovo.run/api/link?flow=${encodeURIComponent(flow)}`),
      env,
    );
    expect(linkRes.status).toBe(200);
    const link = (await linkRes.json()) as { ok: boolean; nodes: unknown[] };
    expect(link.ok).toBe(true);
    expect(link.nodes).toHaveLength(5);

    const ogWf = await ogGraphNs.decodeOgWorkflowFromQuery(`?flow=${encodeURIComponent(flow)}`);
    expect(ogWf?.structureTarget).toEqual({
      kind: 'afdb',
      uniprotId: 'P69905',
      cropSegments: '',
    });

    const g = await shareNs.decodeHashToGraph(`?flow=${encodeURIComponent(flow)}`);
    expect(g).not.toBeNull();
    expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
  });

  it('T4-06: Scenario 6 — Peptide / Multi-Chain Sequence Evaluation (sequence-input label vs _label + evodiff + foldflow2)', async () => {
    const seq = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ:ACDEFGHIKLMNPQRSTVWY';
    const flow =
      `q:q:sequence=${seq},label=dimer-lead,_label="Lead Complex"~` +
      `evo:e:taskMode=unconditional~fold:f:model=protenix2;evo~f2:f2`;

    const g = await shareNs.decodeHashToGraph(`?flow=${encodeURIComponent(flow)}`);
    expect(g).not.toBeNull();
    const qNode = g!.nodes.find((n) => n.adapterId === 'sequence-input')!;
    expect(qNode.label).toBe('Lead Complex');
    expect(qNode.params.label).toBe('dimer-lead');
    expect(qNode.params.sequence).toBe(seq);
    expect(graphUtilsNs.validateGraph(g!)).toEqual([]);

    const env = createMockCfEnv();
    const linkRes = await cfWorker.fetch(
      new Request(`https://denovo.run/api/link?flow=${encodeURIComponent(flow)}`),
      env,
    );
    expect(linkRes.status).toBe(200);
    const linkBody = (await linkRes.json()) as {
      ok: boolean;
      nodes: Array<{ adapterId: string; label: string; params: Record<string, unknown> }>;
    };
    expect(linkBody.ok).toBe(true);
    const linkQ = linkBody.nodes.find((n) => n.adapterId === 'sequence-input')!;
    expect(linkQ.label).toBe('Lead Complex');
    expect(linkQ.params.label).toBe('dimer-lead');
    expect(linkQ.params.sequence).toBe(seq);
  });

  it('T4-07: Scenario 7 — LLM Self-Correction Loop via /llms.txt + /api/link Validation', async () => {
    // 1. LLM discovers capabilities from /llms.txt
    const { implemented, text } = await generateLiveLlmsTxt();
    expect(implemented).toBe(true);
    expect(text!).toContain('rfd4-proteina');
    expect(extractWorkflowExampleUrls(text!).length).toBeGreaterThanOrEqual(2);

    const env = createMockCfEnv();

    // 2. LLM submits a hallucinated draft URL to /api/link -> gets 400 structured error
    const badRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/link?flow=rfdiffusion4:length=999~mpnn~af2'),
      env,
    );
    expect(badRes.status).toBe(400);
    const badBody = (await badRes.json()) as {
      ok: boolean;
      errors: Array<{ code?: string; message: string }>;
    };
    expect(badBody.ok).toBe(false);
    expect(badBody.errors.length).toBeGreaterThan(0);

    // 3. LLM consults /llms.txt, repairs to valid model IDs & bounds, and re-validates via /api/link
    const goodFlow =
      'rfd4-proteina:length=80~web-protein-mpnn:numSequences=2~localfold-af2:model=monomer~designability-scorer';
    const goodRes = await cfWorker.fetch(
      new Request(`https://denovo.run/api/link?flow=${encodeURIComponent(goodFlow)}`),
      env,
    );
    expect(goodRes.status).toBe(200);
    const goodBody = (await goodRes.json()) as {
      ok: boolean;
      url: string;
      shortUrl: string;
      errors: unknown[];
    };
    expect(goodBody.ok).toBe(true);
    expect(goodBody.errors).toEqual([]);
    assertMarkdownAutolinkerSafe(goodBody.url);

    // 4. Resulting shortUrl decodes & validates cleanly in client engine
    const finalGraph = await shareNs.decodeHashToGraph(
      `?${new URL(goodBody.shortUrl).searchParams.toString()}`,
    );
    expect(finalGraph).not.toBeNull();
    expect(graphUtilsNs.validateGraph(finalGraph!)).toEqual([]);
  });
});
