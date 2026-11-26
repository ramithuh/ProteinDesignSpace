import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  ALL_28_ADAPTER_IDS,
  ALL_28_SHORT_CODES,
  BUILD_DIR,
  REPO_ROOT,
  TOOLS_DIR,
  WEB_DIR,
  assertMarkdownAutolinkerSafe,
  assertValidOgPngBytes,
  buildMinimalRunnableGraphForAdapter,
  cfWorker,
  createMockCfEnv,
  encodeGraphToFlowQuery,
  extractWorkflowExampleUrls,
  findEvaluationDocPath,
  generateLiveLlmsTxt,
  graphUtilsNs,
  ogGraphNs,
  pageHelpersNs,
  parseFlowStructured,
  registryNs,
  shareNs,
} from './helpers';

describe('Tier 1 — Core Feature Coverage (55 cases across F1–F10)', () => {
  // =========================================================================
  // F1: Clean Worktree & Verification Gate (5 cases)
  // =========================================================================
  describe('F1: Clean git branch/worktree & verification gate', () => {
    it('T1-F1-01: repository HEAD is a valid commit descended from 361ad44', () => {
      const headSha = execFileSync('git', ['-C', REPO_ROOT, 'rev-parse', 'HEAD'], {
        encoding: 'utf8',
      }).trim();
      expect(headSha).toMatch(/^[0-9a-f]{40}$/);

      const ancestorCheck = execFileSync(
        'git',
        ['-C', REPO_ROOT, 'merge-base', '--is-ancestor', '361ad4410cb009d393ec75ac738a7d8704f9bb32', 'HEAD'],
        { encoding: 'utf8' },
      );
      expect(ancestorCheck).toBe('');
    });

    it('T1-F1-02: E2E harness is isolated in vitest.e2e-llm.config.ts without altering unit vitest.config.ts', () => {
      const unitCfg = readFileSync(resolve(WEB_DIR, 'vitest.config.ts'), 'utf8');
      const e2eCfg = readFileSync(resolve(WEB_DIR, 'vitest.e2e-llm.config.ts'), 'utf8');
      expect(unitCfg).toContain("include: ['src/**/*.test.ts']");
      expect(e2eCfg).toContain("include: ['tests/e2e-llm-urls/**/*.test.ts']");
      expect(e2eCfg).toContain('$lib');
      expect(e2eCfg).toContain('proteinmpnn-web');
    });

    it('T1-F1-03: web/package.json exposes test, check, build, postbuild, and check:deploy scripts', () => {
      const pkg = JSON.parse(readFileSync(resolve(WEB_DIR, 'package.json'), 'utf8')) as {
        scripts?: Record<string, string>;
      };
      expect(pkg.scripts?.test).toContain('vitest run');
      expect(pkg.scripts?.check).toContain('svelte-check');
      expect(pkg.scripts?.build).toContain('vite build');
      expect(pkg.scripts?.postbuild).toContain('build-cf-worker.mjs');
      expect(pkg.scripts?.['check:deploy']).toContain('check-deploy.mjs');
    });

    it('T1-F1-04: tools/build-cf-worker.mjs configures esbuild browser bundle for cf-worker.ts', () => {
      const script = readFileSync(resolve(TOOLS_DIR, 'build-cf-worker.mjs'), 'utf8');
      expect(script).toContain('cf-worker.ts');
      expect(script).toContain("platform: 'browser'");
      expect(script).toContain("format: 'esm'");
      expect(script).toContain('_worker.js');
      expect(script).toContain('_routes.json');
    });

    it('T1-F1-05: tools/check-deploy.mjs enforces Cloudflare 25 MiB limit and forbids bundled weights', () => {
      const script = readFileSync(resolve(TOOLS_DIR, 'check-deploy.mjs'), 'utf8');
      expect(script).toContain('25 * 1024 * 1024');
      expect(script).toContain('weights');
      expect(script).toContain('_worker.js');
      expect(script).toContain('_routes.json');
      expect(script).toContain('og-default.png');
    });
  });

  // =========================================================================
  // F2: Quantitative URL Representation Evaluation (5 cases)
  // =========================================================================
  describe('F2: Quantitative URL representation evaluation', () => {
    it('T1-F2-01: evaluation document exists on the clean branch and is non-empty Markdown', () => {
      const docPath = findEvaluationDocPath();
      expect(docPath, 'Expected evaluation markdown doc in src/lib/engine/ or docs/').not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      expect(text.length).toBeGreaterThan(500);
      expect(text).toMatch(/^#\s+/m);
    });

    it('T1-F2-02: evaluation explicitly analyzes all 4 candidate representations', () => {
      const docPath = findEvaluationDocPath();
      expect(docPath).not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      expect(text).toContain('?flow=');
      expect(text).toMatch(/shorthand|\?g=|\?p=/i);
      expect(text).toContain('?recipe=');
      expect(text).toContain('?prompt=');
    });

    it('T1-F2-03: evaluation provides quantitative character/token efficiency numbers', () => {
      const docPath = findEvaluationDocPath();
      expect(docPath).not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      const tableRows = text.split('\n').filter((line) => line.trim().startsWith('|'));
      expect(tableRows.length).toBeGreaterThanOrEqual(4);
      expect(text).toMatch(/\b\d{2,4}\s*(?:chars?|characters?|tokens?|bytes?)\b/i);
    });

    it('T1-F2-04: evaluation explains Markdown autolinker safety of ~ : = , ; over () [] >', () => {
      const docPath = findEvaluationDocPath();
      expect(docPath).not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      expect(text).toMatch(/autolink|markdown|GFM|Slack|Discord/i);
      expect(text).toContain('~');
      expect(text).toContain(':');
      expect(text).toMatch(/\(\s*\)|parenthes/i);
    });

    it('T1-F2-05: evaluation compares DAG expressiveness & error recovery justifying ?flow= primary + ?recipe= fallback', () => {
      const docPath = findEvaluationDocPath();
      expect(docPath).not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      expect(text).toMatch(/DAG|branch|fan-out|fan-in/i);
      expect(text).toMatch(/error|recovery|validation|deterministic/i);
      expect(text).toMatch(/fallback/i);
    });
  });

  // =========================================================================
  // F3: Markdown-Safe ?flow= DSL & Encoder/Decoder (8 cases)
  // =========================================================================
  describe('F3: Markdown-safe ?flow= DSL & encoder/decoder', () => {
    it('T1-F3-01: decodes linear 3-node chain r3:length=80~m~f using short codes', async () => {
      const res = await parseFlowStructured('r3:length=80~m~f');
      expect(res.implemented).toBe(true);
      expect(res.ok).toBe(true);
      expect(res.errors).toEqual([]);

      const g = await shareNs.decodeHashToGraph('?flow=r3:length=80~m~f');
      expect(g).not.toBeNull();
      expect(g!.nodes.map((n) => n.adapterId)).toEqual([
        'rfd3',
        'web-protein-mpnn',
        'localfold-af2',
      ]);
      expect(g!.nodes[0].params.length).toBe(80);
      expect(g!.edges).toHaveLength(2);
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F3-02: decodes linear chain using full adapter IDs and typed parameters', async () => {
      const dsl =
        'rfd4-proteina:length=70~web-protein-mpnn:numSequences=4~localfold-af2:model=boltz2';
      const g = await shareNs.decodeHashToGraph(`?flow=${dsl}`);
      expect(g).not.toBeNull();
      expect(g!.nodes.map((n) => n.adapterId)).toEqual([
        'rfd4-proteina',
        'web-protein-mpnn',
        'localfold-af2',
      ]);
      expect(g!.nodes[0].params.length).toBe(70);
      expect(g!.nodes[1].params.numSequences).toBe(4);
      expect(g!.nodes[2].params.model).toBe('boltz2');
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F3-03: decodes multi-branch RFD3 vs RFD4 comparison from shared pdb-input into structure-compare', async () => {
      const dsl = 'src:p:mode=demo-bundle;src~b3:r3:length=80~cmp:c;src~b4:r:length=80~cmp';
      const g = await shareNs.decodeHashToGraph(`?flow=${dsl}`);
      expect(g).not.toBeNull();
      expect(g!.nodes.map((n) => n.adapterId)).toEqual(
        expect.arrayContaining(['pdb-input', 'rfd3', 'rfd4-proteina', 'structure-compare']),
      );
      expect(g!.nodes).toHaveLength(4);
      expect(g!.edges).toHaveLength(4);
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F3-04: decodes fan-out/fan-in merge (gen:r~m~f;gen~s;f~s)', async () => {
      const dsl = 'gen:r~m~f;gen~s;f~s';
      const g = await shareNs.decodeHashToGraph(`?flow=${dsl}`);
      expect(g).not.toBeNull();
      expect(g!.nodes).toHaveLength(4);
      expect(g!.edges.length).toBeGreaterThanOrEqual(4);
      const genToScorer = g!.edges.find(
        (e) => e.fromNode === 'gen' && e.toNode === 's' && e.toPort === 'structure',
      );
      const foldToScorer = g!.edges.find(
        (e) => e.fromNode === 'f' && e.toNode === 's' && e.toPort === 'folds',
      );
      expect(genToScorer).toBeDefined();
      expect(foldToScorer).toBeDefined();
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F3-05: decodes custom node labels (_label=Hero_Gen) and explicit from.port~to.port', async () => {
      const dsl = 'gen:r.structure:_label=Hero_Gen~mpnn:m.structure;mpnn.sequences~fold:f.sequences';
      const g = await shareNs.decodeHashToGraph(`?flow=${dsl}`);
      expect(g).not.toBeNull();
      const gen = g!.nodes.find((n) => n.id === 'gen');
      expect(gen?.label).toBe('Hero_Gen');
      expect(g!.edges).toEqual(
        expect.arrayContaining([
          { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
          { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        ]),
      );
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F3-06: round-trips WorkflowGraph -> encodeGraphToFlow -> decodeHashToGraph across all 28 adapter IDs', async () => {
      expect(Object.keys(registryNs.ADAPTER_REGISTRY)).toHaveLength(28);
      for (const adapterId of ALL_28_ADAPTER_IDS) {
        const orig = buildMinimalRunnableGraphForAdapter(adapterId);
        expect(graphUtilsNs.validateGraph(orig)).toEqual([]);

        const encoded = await encodeGraphToFlowQuery(orig);
        expect(encoded.implemented, `encodeGraphToFlow missing for ${adapterId}`).toBe(true);
        expect(encoded.url).toBeTruthy();
        assertMarkdownAutolinkerSafe(encoded.url!);

        const decoded = await shareNs.decodeHashToGraph(`?flow=${encoded.dsl}`);
        expect(decoded, `decodeHashToGraph returned null for ${adapterId}`).not.toBeNull();
        expect(decoded!.nodes.map((n) => n.adapterId)).toEqual(orig.nodes.map((n) => n.adapterId));
        expect(decoded!.edges).toHaveLength(orig.edges.length);
        expect(graphUtilsNs.validateGraph(decoded!)).toEqual([]);
      }
    });

    it('T1-F3-07: tolerates LLM-emitted -> and (...) syntax identically to ~ and :', async () => {
      const canonical = await shareNs.decodeHashToGraph('?flow=r3:length=80~m:numSequences=2~f');
      const tolerated = await shareNs.decodeHashToGraph(
        '?flow=r3(length=80)->m(numSequences=2)->f',
      );
      expect(canonical).not.toBeNull();
      expect(tolerated).not.toBeNull();
      expect(tolerated!.nodes.map((n) => n.adapterId)).toEqual(
        canonical!.nodes.map((n) => n.adapterId),
      );
      expect(tolerated!.nodes[0].params.length).toBe(80);
      expect(tolerated!.nodes[1].params.numSequences).toBe(2);
      expect(tolerated!.edges).toEqual(canonical!.edges);
    });

    it('T1-F3-08: primary generated ?flow= URLs contain zero Markdown-unsafe characters and pass autolinker checks', async () => {
      const preset = registryNs.WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-branch-compare')!;
      const encoded = await encodeGraphToFlowQuery(preset.graph);
      expect(encoded.implemented).toBe(true);
      expect(encoded.url).toBeTruthy();
      assertMarkdownAutolinkerSafe(encoded.url!);
      expect(encoded.dsl).not.toMatch(/[()[\]<>]/);
    });
  });

  // =========================================================================
  // F4: Compatible-Port Auto-Wiring & designability-scorer (6 cases)
  // =========================================================================
  describe('F4: Compatible-port auto-wiring & designability-scorer', () => {
    it('T1-F4-01: auto-wires designability-scorer dual inputs (structure + folds) in r~m~f~s', async () => {
      const g = await shareNs.decodeHashToGraph('?flow=r~m~f~s');
      expect(g).not.toBeNull();
      const scorer = g!.nodes.find((n) => n.adapterId === 'designability-scorer')!;
      const gen = g!.nodes.find((n) => n.adapterId === 'rfd4-proteina')!;
      const fold = g!.nodes.find((n) => n.adapterId === 'localfold-af2')!;
      expect(g!.edges).toEqual(
        expect.arrayContaining([
          { fromNode: gen.id, fromPort: 'structure', toNode: scorer.id, toPort: 'structure' },
          { fromNode: fold.id, fromPort: 'folds', toNode: scorer.id, toPort: 'folds' },
        ]),
      );
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F4-02: auto-wires native co-design r~f~s (r.sequences->f.sequences, r.structure->s.structure, f.folds->s.folds)', async () => {
      const g = await shareNs.decodeHashToGraph('?flow=r~f~s');
      expect(g).not.toBeNull();
      expect(g!.edges).toHaveLength(3);
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F4-03: auto-wires pdb-input~genie2 into non-standard input port genie2.template', async () => {
      const g = await shareNs.decodeHashToGraph('?flow=p~g2');
      expect(g).not.toBeNull();
      expect(g!.edges).toEqual([
        {
          fromNode: g!.nodes[0].id,
          fromPort: 'structure',
          toNode: g!.nodes[1].id,
          toPort: 'template',
        },
      ]);
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F4-04: auto-wires 3 parallel generators into structure-compare.structures (multiple: true)', async () => {
      const g = await shareNs.decodeHashToGraph('?flow=a:r~cmp:c;b:g~cmp;d:r3~cmp');
      expect(g).not.toBeNull();
      const cmpEdges = g!.edges.filter((e) => e.toNode === 'cmp');
      expect(cmpEdges).toHaveLength(3);
      expect(cmpEdges.every((e) => e.toPort === 'structures')).toBe(true);
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F4-05: auto-wires esmc and dyna1 from either SequenceSet (m~ec) or StructureArtifact (r~dy)', async () => {
      const seqBranch = await shareNs.decodeHashToGraph('?flow=r~m~ec');
      expect(seqBranch).not.toBeNull();
      const ecEdge = seqBranch!.edges.find((e) => e.toNode === seqBranch!.nodes[2].id);
      expect(ecEdge?.toPort).toBe('sequences');
      expect(graphUtilsNs.validateGraph(seqBranch!)).toEqual([]);

      const structBranch = await shareNs.decodeHashToGraph('?flow=r~dy');
      expect(structBranch).not.toBeNull();
      const dyEdge = structBranch!.edges.find((e) => e.toNode === structBranch!.nodes[1].id);
      expect(dyEdge?.toPort).toBe('structure');
      expect(graphUtilsNs.validateGraph(structBranch!)).toEqual([]);
    });

    it('T1-F4-06: auto-wires top-k-filter between web-protein-mpnn and localfold-af2 (r~m~tk~f~s)', async () => {
      const g = await shareNs.decodeHashToGraph('?flow=r~m~tk~f~s');
      expect(g).not.toBeNull();
      expect(g!.nodes.map((n) => n.adapterId)).toEqual([
        'rfd4-proteina',
        'web-protein-mpnn',
        'top-k-filter',
        'localfold-af2',
        'designability-scorer',
      ]);
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });
  });

  // =========================================================================
  // F5: ?recipe= Fallback & Legacy ?g=/?p=/#g=/#p= Preservation (5 cases)
  // =========================================================================
  describe('F5: ?recipe= fallback & legacy share link preservation', () => {
    it('T1-F5-01: decodes ?recipe=<urlencoded YAML> (denovo: v1) via decodeHashToGraph', async () => {
      const yaml = shareNs.graphToYamlRecipe(registryNs.WORKFLOW_PRESETS[0].graph);
      const g = await shareNs.decodeHashToGraph(`?recipe=${encodeURIComponent(yaml)}`);
      expect(g).not.toBeNull();
      expect(g!.nodes.length).toBe(registryNs.WORKFLOW_PRESETS[0].graph.nodes.length);
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F5-02: decodes ?recipe=<urlencoded JSON> (WorkflowGraph JSON) via decodeHashToGraph', async () => {
      const json = shareNs.serializeWorkflowJson(registryNs.WORKFLOW_PRESETS[7].graph);
      const g = await shareNs.decodeHashToGraph(`?recipe=${encodeURIComponent(json)}`);
      expect(g).not.toBeNull();
      expect(g!.nodes.map((n) => n.adapterId)).toEqual(['pdb-input', 'thermompnn']);
      expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
    });

    it('T1-F5-03: preserves #p= and ?p= round-trip for all 8 WORKFLOW_PRESETS', async () => {
      expect(registryNs.WORKFLOW_PRESETS).toHaveLength(8);
      for (const preset of registryNs.WORKFLOW_PRESETS) {
        const enc = await shareNs.encodeGraphToHash(preset.graph);
        expect(enc.mode).toBe('preset-diff');
        expect(enc.hash).toBe(`#p=${preset.id}`);
        const fromHash = await shareNs.decodeHashToGraph(enc.hash);
        const fromQuery = await shareNs.decodeHashToGraph(enc.hash.replace(/^#/, '?'));
        expect(fromHash?.nodes.length).toBe(preset.graph.nodes.length);
        expect(fromQuery?.edges.length).toBe(preset.graph.edges.length);
      }
    });

    it('T1-F5-04: preserves #g= and ?g= v2 compressed-DAG round-trip for custom multi-node DAG', async () => {
      const custom = buildMinimalRunnableGraphForAdapter('ligandmpnn');
      custom.nodes[1].label = 'CustomLigMPNN';
      custom.nodes[1].params.numSequences = 4;
      const enc = await shareNs.encodeGraphToHash(custom);
      expect(enc.mode).toBe('compressed-dag');
      expect(enc.hash.startsWith('#g=')).toBe(true);
      const dec = await shareNs.decodeHashToGraph(enc.hash.replace(/^#/, '?'));
      expect(dec).not.toBeNull();
      expect(dec!.nodes[1].label).toBe('CustomLigMPNN');
      expect(dec!.nodes[1].params.numSequences).toBe(4);
      expect(graphUtilsNs.validateGraph(dec!)).toEqual([]);
    });

    it('T1-F5-05: isSharedWorkflowHash recognizes ?flow=, #flow=, ?recipe=, #recipe=, ?g=, #g=, ?p=, #p=', () => {
      for (const prefix of [
        '?flow=r~m~f',
        '#flow=r~m~f',
        '?recipe=denovo',
        '#recipe=denovo',
        '?g=abc',
        '#g=abc',
        '?p=genie3-mpnn-af2',
        '#p=genie3-mpnn-af2',
      ]) {
        expect(pageHelpersNs.isSharedWorkflowHash(prefix), `Expected true for ${prefix}`).toBe(true);
      }
      expect(pageHelpersNs.isSharedWorkflowHash('?foo=bar')).toBe(false);
      expect(pageHelpersNs.isSharedWorkflowHash('#')).toBe(false);
      expect(pageHelpersNs.isSharedWorkflowHash('')).toBe(false);
    });
  });

  // =========================================================================
  // F6: Build-Generated /llms.txt from ADAPTER_REGISTRY (5 cases)
  // =========================================================================
  describe('F6: Build-generated /llms.txt from ADAPTER_REGISTRY', () => {
    it('T1-F6-01: generateLlmsTxt() lists all 28 adapter IDs and all 28 short codes', async () => {
      const { implemented, text } = await generateLiveLlmsTxt();
      expect(implemented).toBe(true);
      expect(text).toBeTruthy();
      for (const id of ALL_28_ADAPTER_IDS) {
        expect(text!, `Missing adapter ID ${id}`).toContain(id);
        const code = ALL_28_SHORT_CODES[id];
        expect(text!, `Missing short code ${code} for ${id}`).toMatch(
          new RegExp(`(?:^|[^a-z0-9])${code}(?:$|[^a-z0-9])`, 'm'),
        );
      }
    });

    it('T1-F6-02: /llms.txt documents all 20 LOCALFOLD_MODEL_OPTIONS and all 3 MPNN_VARIANT_OPTIONS', async () => {
      const { implemented, text } = await generateLiveLlmsTxt();
      expect(implemented).toBe(true);
      expect(registryNs.LOCALFOLD_MODEL_OPTIONS).toHaveLength(20);
      for (const opt of registryNs.LOCALFOLD_MODEL_OPTIONS) {
        expect(text!, `Missing LocalFold model ${opt.id}`).toContain(opt.id);
      }
      expect(registryNs.MPNN_VARIANT_OPTIONS).toHaveLength(3);
      for (const opt of registryNs.MPNN_VARIANT_OPTIONS) {
        expect(text!, `Missing MPNN variant ${opt.id}`).toContain(opt.id);
      }
    });

    it('T1-F6-03: /llms.txt documents port types and Markdown-safe ?flow= syntax rules', async () => {
      const { implemented, text } = await generateLiveLlmsTxt();
      expect(implemented).toBe(true);
      for (const portType of [
        'StructureArtifact',
        'SequenceSet',
        'FoldedStructure',
        'DesignabilityMetrics',
      ]) {
        expect(text!).toContain(portType);
      }
      for (const sym of ['~', ':', '=', ',', ';', '.']) {
        expect(text!).toContain(sym);
      }
      expect(text!).toContain('?flow=');
    });

    it('T1-F6-04: every example URL extracted from /llms.txt parses and validates with 0 errors', async () => {
      const { implemented, text } = await generateLiveLlmsTxt();
      expect(implemented).toBe(true);
      const urls = extractWorkflowExampleUrls(text!);
      expect(urls.length).toBeGreaterThanOrEqual(3);
      for (const rawUrl of urls) {
        const query = rawUrl.includes('?') ? `?${rawUrl.split('?')[1]}` : rawUrl;
        const g = await shareNs.decodeHashToGraph(query);
        expect(g, `Example URL failed to decode: ${rawUrl}`).not.toBeNull();
        expect(graphUtilsNs.validateGraph(g!), `Example URL invalid graph: ${rawUrl}`).toEqual([]);
      }
    });

    it('T1-F6-05: built web/build/llms.txt exists and matches live generateLlmsTxt() output', async () => {
      const { implemented, text } = await generateLiveLlmsTxt();
      expect(implemented).toBe(true);
      const builtPath = resolve(BUILD_DIR, 'llms.txt');
      expect(existsSync(builtPath), 'Expected web/build/llms.txt to exist after npm run build').toBe(
        true,
      );
      const builtText = readFileSync(builtPath, 'utf8');
      expect(builtText).toBe(text);

      // Also verify vite.config.ts wires dev middleware / build plugin for /llms.txt
      const viteCfg = readFileSync(resolve(WEB_DIR, 'vite.config.ts'), 'utf8');
      expect(viteCfg).toContain('/llms.txt');
    });
  });

  // =========================================================================
  // F7: HTML Discovery <link rel="llms"> (5 cases)
  // =========================================================================
  describe('F7: HTML discovery <link rel="llms">', () => {
    it('T1-F7-01: web/src/app.html includes <link rel="llms" ... href="/llms.txt">', () => {
      const appHtml = readFileSync(resolve(WEB_DIR, 'src/app.html'), 'utf8');
      expect(appHtml).toMatch(/<link\b[^>]*\brel="llms"[^>]*\bhref="\/llms\.txt"[^>]*>/i);
    });

    it('T1-F7-02: built web/build/index.html preserves <link rel="llms" ... href="/llms.txt">', () => {
      const indexPath = resolve(BUILD_DIR, 'index.html');
      expect(existsSync(indexPath), 'Expected web/build/index.html to exist').toBe(true);
      const html = readFileSync(indexPath, 'utf8');
      expect(html).toMatch(/<link\b[^>]*\brel="llms"[^>]*\bhref="\/llms\.txt"[^>]*>/i);
    });

    it('T1-F7-03: built web/build/runner.html (or runner/index.html) preserves <link rel="llms">', () => {
      const runnerHtml = existsSync(resolve(BUILD_DIR, 'runner.html'))
        ? resolve(BUILD_DIR, 'runner.html')
        : resolve(BUILD_DIR, 'runner/index.html');
      expect(existsSync(runnerHtml), 'Expected built runner HTML to exist').toBe(true);
      const html = readFileSync(runnerHtml, 'utf8');
      expect(html).toMatch(/<link\b[^>]*\brel="llms"[^>]*\bhref="\/llms\.txt"[^>]*>/i);
    });

    it('T1-F7-04: existing og:* meta tags and favicon link remain intact in app.html alongside <link rel="llms">', () => {
      const appHtml = readFileSync(resolve(WEB_DIR, 'src/app.html'), 'utf8');
      expect(appHtml).toContain('<link rel="icon"');
      expect(appHtml).toContain('property="og:title"');
      expect(appHtml).toContain('property="og:image"');
      expect(appHtml).toMatch(/<link\b[^>]*\brel="llms"/i);
    });

    it('T1-F7-05: cfWorker HTML OG meta rewrite preserves <link rel="llms"> when serving /?flow=...', async () => {
      const appHtml = readFileSync(resolve(WEB_DIR, 'src/app.html'), 'utf8');
      const env = createMockCfEnv({ html: appHtml });
      const res = await cfWorker.fetch(new Request('https://denovo.run/?flow=r:length=80~m~f~s'), env);
      expect(res.status).toBe(200);
      const out = await res.text();
      expect(out).toMatch(/<link\b[^>]*\brel="llms"[^>]*\bhref="\/llms\.txt"[^>]*>/i);
      expect(out).toContain('/api/og?flow=');
    });
  });

  // =========================================================================
  // F8: Dynamic OG PNG (1200x630) & Meta Tag Injection (5 cases)
  // =========================================================================
  describe('F8: Dynamic OG PNG & meta tag injection for ?flow= and ?recipe=', () => {
    it('T1-F8-01: decodeOgWorkflowFromQuery decodes ?flow=r:length=80~m~f~s into 4 OgNodes and 4 OgEdges', async () => {
      const wf = await ogGraphNs.decodeOgWorkflowFromQuery('?flow=r:length=80~m~f~s');
      expect(wf).not.toBeNull();
      expect(wf!.nodes).toHaveLength(4);
      expect(wf!.edges).toHaveLength(4);
      expect(wf!.title).toContain('denovo.run');
      expect(wf!.description.length).toBeGreaterThan(20);
    });

    it('T1-F8-02: decodeOgWorkflowFromQuery decodes ?recipe=<YAML/JSON> including structureTarget', async () => {
      const yaml = [
        'denovo: v1',
        'name: "1PRW Motif"',
        'nodes:',
        '  src: {"model":"pdb-input","mode":"pdb-id","pdbId":"1PRW"}',
        '  mpnn: {"model":"web-protein-mpnn"}',
        'wires:',
        '  - src.structure -> mpnn.structure',
      ].join('\n');
      const wf = await ogGraphNs.decodeOgWorkflowFromQuery(`?recipe=${encodeURIComponent(yaml)}`);
      expect(wf).not.toBeNull();
      expect(wf!.nodes).toHaveLength(2);
      expect(wf!.structureTarget).toEqual({ kind: 'rcsb', pdbId: '1PRW', cropSegments: '' });
    });

    it('T1-F8-03: GET /api/og?flow=... returns 200 image/png with valid 1200x630 IHDR dimensions', async () => {
      const env = createMockCfEnv();
      const defRes = await cfWorker.fetch(new Request('https://denovo.run/api/og'), env);
      const defBytes = new Uint8Array(await defRes.arrayBuffer());
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/og?flow=r:length=80~m~f~s'),
        env,
      );
      expect(res.status).toBe(200);
      expect(res.headers.get('content-type')).toBe('image/png');
      const pngBytes = new Uint8Array(await res.arrayBuffer());
      assertValidOgPngBytes(pngBytes, 1200, 630);
      expect(Buffer.from(pngBytes).equals(Buffer.from(defBytes))).toBe(false);
    });

    it('T1-F8-04: GET /api/og?recipe=... returns 200 image/png with valid 1200x630 IHDR dimensions', async () => {
      const yaml = shareNs.graphToYamlRecipe(registryNs.WORKFLOW_PRESETS[0].graph);
      const env = createMockCfEnv();
      const defRes = await cfWorker.fetch(new Request('https://denovo.run/api/og'), env);
      const defBytes = new Uint8Array(await defRes.arrayBuffer());
      const res = await cfWorker.fetch(
        new Request(`https://denovo.run/api/og?recipe=${encodeURIComponent(yaml)}`),
        env,
      );
      expect(res.status).toBe(200);
      expect(res.headers.get('content-type')).toBe('image/png');
      const pngBytes = new Uint8Array(await res.arrayBuffer());
      assertValidOgPngBytes(pngBytes, 1200, 630);
      expect(Buffer.from(pngBytes).equals(Buffer.from(defBytes))).toBe(false);
    });

    it('T1-F8-05: GET /?flow=... and GET /?recipe=... inject dynamic OG meta tags and strip og-default.png', async () => {
      const env = createMockCfEnv();
      const resFlow = await cfWorker.fetch(
        new Request('https://denovo.run/?flow=r3:length=80~m~f~s'),
        env,
      );
      expect(resFlow.status).toBe(200);
      const htmlFlow = await resFlow.text();
      expect(htmlFlow).toContain('/api/og?flow=r3');
      expect(htmlFlow).toContain('property="og:title"');
      expect(htmlFlow).toContain('name="twitter:card" content="summary_large_image"');
      expect(htmlFlow).not.toContain('https://denovo.run/og-default.png');

      const yaml = shareNs.graphToYamlRecipe(registryNs.WORKFLOW_PRESETS[7].graph);
      const resRecipe = await cfWorker.fetch(
        new Request(`https://denovo.run/?recipe=${encodeURIComponent(yaml)}`),
        env,
      );
      expect(resRecipe.status).toBe(200);
      const htmlRecipe = await resRecipe.text();
      expect(htmlRecipe).toContain('/api/og?recipe=');
      expect(htmlRecipe).not.toContain('https://denovo.run/og-default.png');
    });
  });

  // =========================================================================
  // F9: Stateless GET/POST /api/link Edge Endpoint (5 cases)
  // =========================================================================
  describe('F9: Stateless GET/POST /api/link edge endpoint', () => {
    it('T1-F9-01: GET /api/link?flow=r3:length=80~m~f~s returns 200 JSON with valid shortUrl (?g= or ?p=)', async () => {
      const env = createMockCfEnv();
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/link?flow=r3:length=80~m~f~s'),
        env,
      );
      expect(res.status).toBe(200);
      const body = (await res.json()) as {
        ok: boolean;
        url: string;
        shortUrl: string;
        nodes: unknown[];
        edges: unknown[];
        errors: unknown[];
      };
      expect(body.ok).toBe(true);
      expect(body.errors).toEqual([]);
      expect(body.nodes).toHaveLength(4);
      expect(body.edges).toHaveLength(4);
      expect(body.url).toContain('?flow=');
      expect(body.shortUrl).toMatch(/\?[gp]=/);

      const shortQuery = `?${new URL(body.shortUrl).searchParams.toString()}`;
      const decoded = await shareNs.decodeHashToGraph(shortQuery);
      expect(decoded).not.toBeNull();
      expect(decoded!.nodes).toHaveLength(4);
    });

    it('T1-F9-02: GET /api/link?recipe=<yaml> returns 200 JSON with ok: true and valid shortUrl', async () => {
      const yaml = shareNs.graphToYamlRecipe(registryNs.WORKFLOW_PRESETS[7].graph);
      const env = createMockCfEnv();
      const res = await cfWorker.fetch(
        new Request(`https://denovo.run/api/link?recipe=${encodeURIComponent(yaml)}`),
        env,
      );
      expect(res.status).toBe(200);
      const body = (await res.json()) as { ok: boolean; shortUrl: string; errors: unknown[] };
      expect(body.ok).toBe(true);
      expect(body.errors).toEqual([]);
      const decoded = await shareNs.decodeHashToGraph(`?${new URL(body.shortUrl).searchParams.toString()}`);
      expect(decoded).not.toBeNull();
      expect(decoded!.nodes.map((n) => n.adapterId)).toEqual(['pdb-input', 'thermompnn']);
    });

    it('T1-F9-03: POST /api/link with JSON { flow } returns 200 JSON with nodes, edges, and shortUrl', async () => {
      const env = createMockCfEnv();
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ flow: 'r:length=70~m~f~s' }),
        }),
        env,
      );
      expect(res.status).toBe(200);
      const body = (await res.json()) as {
        ok: boolean;
        nodes: unknown[];
        edges: unknown[];
        shortUrl: string;
        errors: unknown[];
      };
      expect(body.ok).toBe(true);
      expect(body.nodes).toHaveLength(4);
      expect(body.edges).toHaveLength(4);
      expect(body.errors).toEqual([]);
      expect(body.shortUrl).toMatch(/\?[gp]=/);
    });

    it('T1-F9-04: POST /api/link supports JSON { recipe }, JSON { graph }, and text/plain body', async () => {
      const env = createMockCfEnv();
      const preset = registryNs.WORKFLOW_PRESETS[7].graph;
      const yaml = shareNs.graphToYamlRecipe(preset);

      for (const req of [
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ recipe: yaml }),
        }),
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ graph: preset }),
        }),
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'text/plain' },
          body: 'r:length=70~m~f',
        }),
      ]) {
        const res = await cfWorker.fetch(req, env);
        expect(res.status).toBe(200);
        const body = (await res.json()) as { ok: boolean; errors: unknown[] };
        expect(body.ok).toBe(true);
        expect(body.errors).toEqual([]);
      }
    });

    it('T1-F9-05: GET /api/link?g=... and ?p=genie3-mpnn-af2 validate and normalize existing share links', async () => {
      const env = createMockCfEnv();
      const presetRes = await cfWorker.fetch(
        new Request('https://denovo.run/api/link?p=genie3-mpnn-af2'),
        env,
      );
      expect(presetRes.status).toBe(200);
      const presetBody = (await presetRes.json()) as { ok: boolean; nodes: unknown[] };
      expect(presetBody.ok).toBe(true);
      expect(presetBody.nodes).toHaveLength(4);

      const custom = buildMinimalRunnableGraphForAdapter('thermompnn');
      const enc = await shareNs.encodeGraphToHash(custom);
      const gRes = await cfWorker.fetch(
        new Request(`https://denovo.run/api/link?${enc.hash.slice(1)}`),
        env,
      );
      expect(gRes.status).toBe(200);
      const gBody = (await gRes.json()) as { ok: boolean; nodes: unknown[] };
      expect(gBody.ok).toBe(true);
      expect(gBody.nodes).toHaveLength(2);
    });
  });

  // =========================================================================
  // F10: Full-Stack E2E Workflows & Build/Deploy Contracts (6 cases)
  // =========================================================================
  describe('F10: Full-stack E2E workflows & deploy contracts', () => {
    it('T1-F10-01: cross-layer parity: flow -> /api/link -> shortUrl -> client decode matches direct flow decode', async () => {
      const flow = 'p:mode=pdb-id,pdbId=1PRW~r3:mode=motif,steps=50~lm~f:model=boltz2~s';
      const direct = await shareNs.decodeHashToGraph(`?flow=${flow}`);
      expect(direct).not.toBeNull();

      const env = createMockCfEnv();
      const linkRes = await cfWorker.fetch(
        new Request(`https://denovo.run/api/link?flow=${encodeURIComponent(flow)}`),
        env,
      );
      expect(linkRes.status).toBe(200);
      const linkBody = (await linkRes.json()) as { ok: boolean; shortUrl: string };
      expect(linkBody.ok).toBe(true);

      const viaShort = await shareNs.decodeHashToGraph(
        `?${new URL(linkBody.shortUrl).searchParams.toString()}`,
      );
      expect(viaShort).not.toBeNull();
      expect(viaShort!.nodes.map((n) => n.adapterId)).toEqual(
        direct!.nodes.map((n) => n.adapterId),
      );
      expect(viaShort!.edges).toHaveLength(direct!.edges.length);
    });

    it('T1-F10-02: all cf-worker routes respond with expected HTTP status & content-type', async () => {
      const env = createMockCfEnv();
      const home = await cfWorker.fetch(new Request('https://denovo.run/?flow=r~m~f'), env);
      expect(home.status).toBe(200);
      expect(home.headers.get('content-type')).toContain('text/html');

      const og = await cfWorker.fetch(new Request('https://denovo.run/api/og?flow=r~m~f'), env);
      expect(og.status).toBe(200);
      expect(og.headers.get('content-type')).toBe('image/png');

      const link = await cfWorker.fetch(new Request('https://denovo.run/api/link?flow=r~m~f'), env);
      expect(link.status).toBe(200);
      expect(link.headers.get('content-type')).toContain('application/json');

      const unknownApi = await cfWorker.fetch(new Request('https://denovo.run/api/unknown'), env);
      expect(unknownApi.status).toBe(404);

      const probe = await cfWorker.fetch(new Request('https://denovo.run/.env'), env);
      expect(probe.status).toBe(404);
    });

    it('T1-F10-03: built web/build/_worker.js executes cleanly in isolated ESM import for /api/link and /api/og', async () => {
      const workerPath = resolve(BUILD_DIR, '_worker.js');
      expect(existsSync(workerPath), 'Expected web/build/_worker.js to exist after build').toBe(true);
      const mod = (await import(workerPath)) as { default: typeof cfWorker };
      const env = createMockCfEnv();
      const res = await mod.default.fetch(
        new Request('https://denovo.run/api/link?flow=r~m~f~s'),
        env,
      );
      expect(res.status).toBe(200);
      const body = (await res.json()) as { ok: boolean };
      expect(body.ok).toBe(true);
    });

    it('T1-F10-04: web/build/_routes.json matches Cloudflare Pages v1 schema and routes /api/* cleanly', () => {
      const routesPath = resolve(BUILD_DIR, '_routes.json');
      expect(existsSync(routesPath), 'Expected web/build/_routes.json to exist after build').toBe(true);
      const routes = JSON.parse(readFileSync(routesPath, 'utf8')) as {
        version: number;
        include: string[];
        exclude: string[];
      };
      expect(routes.version).toBe(1);
      expect(routes.include).toContain('/*');
      expect(Array.isArray(routes.exclude)).toBe(true);
      expect(routes.exclude).not.toContain('/api/*');
    });

    it('T1-F10-05: tools/check-deploy.mjs passes with exit code 0 against built web/build/', () => {
      const out = execFileSync('node', [resolve(TOOLS_DIR, 'check-deploy.mjs')], {
        cwd: WEB_DIR,
        encoding: 'utf8',
      });
      expect(out).toContain('[check:deploy] PASS');
    });

    it('T1-F10-06: built web/build/og-default.png is valid 1200x630 PNG and _headers enforces nosniff', () => {
      const ogDefaultPath = resolve(BUILD_DIR, 'og-default.png');
      expect(existsSync(ogDefaultPath), 'Expected web/build/og-default.png to exist').toBe(true);
      assertValidOgPngBytes(new Uint8Array(readFileSync(ogDefaultPath)), 1200, 630);

      const headersPath = resolve(BUILD_DIR, '_headers');
      expect(existsSync(headersPath), 'Expected web/build/_headers to exist').toBe(true);
      const headersTxt = readFileSync(headersPath, 'utf8');
      expect(headersTxt).toContain('X-Content-Type-Options: nosniff');
    });
  });
});
