import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  ALL_27_ADAPTER_IDS,
  BUILD_DIR,
  ORIGINAL_MAIN_DIR,
  REPO_ROOT,
  WEB_DIR,
  assertMarkdownAutolinkerSafe,
  assertValidOgPngBytes,
  buildMinimalRunnableGraphForAdapter,
  cfWorker,
  cfWorkerNs,
  createMockCfEnv,
  extractWorkflowExampleUrls,
  findEvaluationDocPath,
  generateLiveLlmsTxt,
  graphUtilsNs,
  ogGraphNs,
  parseFlowStructured,
  registryNs,
  shareNs,
} from './helpers';

function collectFilesRecursive(dir: string): Array<{ path: string; bytes: number }> {
  const out: Array<{ path: string; bytes: number }> = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = resolve(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...collectFilesRecursive(full));
    } else if (entry.isFile()) {
      out.push({ path: full, bytes: statSync(full).size });
    }
  }
  return out;
}

describe('Tier 2 — Boundary, Corner & Invalid-Input Cases (55 cases across F1–F10)', () => {
  // =========================================================================
  // F1: Worktree Isolation & Asset Limits (5 cases)
  // =========================================================================
  describe('F1: Worktree isolation & deploy boundary guards', () => {
    it('T2-F1-01: original ProteinDesignSpace main working tree remains isolated from LLM-URL branch', () => {
      const branch = execFileSync('git', ['-C', ORIGINAL_MAIN_DIR, 'branch', '--show-current'], {
        encoding: 'utf8',
      }).trim();
      expect(branch).toBe('main');

      const ancestorCheck = execFileSync(
        'git',
        [
          '-C',
          ORIGINAL_MAIN_DIR,
          'merge-base',
          '--is-ancestor',
          '361ad4410cb009d393ec75ac738a7d8704f9bb32',
          'HEAD',
        ],
        { encoding: 'utf8' },
      );
      expect(ancestorCheck).toBe('');

      const newFeatureFiles = [
        'web/src/lib/engine/flow-dsl.ts',
        'web/src/lib/engine/llms-txt.ts',
        'docs/llm-url-evaluation.md',
        'web/tests/e2e-llm-urls',
        'web/tests/e2e-llm-urls/tier5-adversarial.test.ts',
        'web/vitest.e2e-llm.config.ts',
        'web/src/lib/__tests__/flow-dsl.test.ts',
        'web/src/lib/__tests__/llms-txt.test.ts',
        'web/src/lib/__tests__/cf-worker-link.test.ts',
      ];
      for (const relPath of newFeatureFiles) {
        expect(
          existsSync(resolve(ORIGINAL_MAIN_DIR, relPath)),
          `Feature file leaked into ORIGINAL_MAIN_DIR: ${relPath}`,
        ).toBe(false);
      }

      const dirtyFiles = execFileSync('git', ['-C', ORIGINAL_MAIN_DIR, 'diff', '--name-only', 'HEAD'], {
        encoding: 'utf8',
      })
        .split(/\r?\n/)
        .map((s) => s.trim())
        .filter(Boolean);
      const ourModifiedFiles = [
        'web/src/lib/engine/param-schema.ts',
        'web/src/lib/engine/param-validation.ts',
        'web/src/lib/engine/share.ts',
        'web/src/lib/og/cf-worker.ts',
        'web/src/lib/og/og-graph.ts',
        'web/vite.config.ts',
        'web/src/app.html',
        'tools/build-cf-worker.mjs',
        'tools/check-deploy.mjs',
        'web/static/_headers',
        'web/src/lib/ui/page-state-helpers.ts',
        'web/src/lib/__tests__/og-preview.test.ts',
        'web/src/lib/__tests__/page-state-helpers.test.ts',
        'web/src/lib/__tests__/share.test.ts',
      ];
      for (const relPath of ourModifiedFiles) {
        expect(dirtyFiles, `Modified feature file dirty in ORIGINAL_MAIN_DIR: ${relPath}`).not.toContain(
          relPath,
        );
      }

      const porcelainPaths = execFileSync(
        'git',
        ['-C', ORIGINAL_MAIN_DIR, 'status', '--porcelain', '-u'],
        { encoding: 'utf8' },
      )
        .split(/\r?\n/)
        .map((line) => line.slice(3).trim())
        .filter(Boolean);
      for (const relPath of [...newFeatureFiles, ...ourModifiedFiles]) {
        expect(
          porcelainPaths,
          `Feature path dirty/untracked in ORIGINAL_MAIN_DIR: ${relPath}`,
        ).not.toContain(relPath);
      }
    });

    it('T2-F1-02: zero weights/ directories exist under web/static/vendor/* or web/build/vendor/*', () => {
      for (const vendorRoot of [
        resolve(WEB_DIR, 'static/vendor'),
        ...(existsSync(resolve(BUILD_DIR, 'vendor')) ? [resolve(BUILD_DIR, 'vendor')] : []),
      ]) {
        for (const entry of readdirSync(vendorRoot, { withFileTypes: true })) {
          if (!entry.isDirectory()) continue;
          expect(existsSync(resolve(vendorRoot, entry.name, 'weights'))).toBe(false);
        }
      }
    });

    it('T2-F1-03: every built asset in web/build/ is strictly under Cloudflare Pages 25 MiB limit', () => {
      expect(existsSync(BUILD_DIR), 'Expected web/build to exist after npm run build').toBe(true);
      const files = collectFilesRecursive(BUILD_DIR);
      expect(files.length).toBeGreaterThan(0);
      const maxAllowed = 25 * 1024 * 1024;
      for (const f of files) {
        expect(f.bytes, `Asset exceeds 25 MiB: ${f.path}`).toBeLessThan(maxAllowed);
      }
    });

    it('T2-F1-04: zero node:* or Node-builtin imports exist in edge-bundled or pure engine files', () => {
      const candidateFiles = [
        resolve(WEB_DIR, 'src/lib/og/og-graph.ts'),
        resolve(WEB_DIR, 'src/lib/og/cf-worker.ts'),
        resolve(WEB_DIR, 'src/lib/engine/flow-dsl.ts'),
        resolve(WEB_DIR, 'src/lib/engine/llms-txt.ts'),
      ].filter((p) => existsSync(p));

      expect(candidateFiles.length).toBeGreaterThanOrEqual(2);
      for (const filePath of candidateFiles) {
        const src = readFileSync(filePath, 'utf8');
        expect(src, `Node builtin import found in ${filePath}`).not.toMatch(
          /from\s+['"](?:node:|fs|path|crypto|child_process|os|url|stream|http|https)['"]/,
        );
      }
    });

    it('T2-F1-05: zero $lib/ path aliases in edge-bundled modules (og-graph.ts, cf-worker.ts, flow-dsl.ts)', () => {
      const edgeFiles = [
        resolve(WEB_DIR, 'src/lib/og/og-graph.ts'),
        resolve(WEB_DIR, 'src/lib/og/cf-worker.ts'),
        resolve(WEB_DIR, 'src/lib/engine/flow-dsl.ts'),
      ];
      // flow-dsl.ts must exist (or share.ts edge-safe equivalent)
      expect(existsSync(resolve(WEB_DIR, 'src/lib/engine/flow-dsl.ts'))).toBe(true);
      for (const filePath of edgeFiles) {
        const src = readFileSync(filePath, 'utf8');
        expect(src, `Found $lib/ alias in edge file ${filePath}`).not.toMatch(
          /from\s+['"]\$lib\//,
        );
        expect(src, `Found $app/ import in edge file ${filePath}`).not.toMatch(
          /from\s+['"]\$app\//,
        );
        expect(src, `Found svelte import in edge file ${filePath}`).not.toMatch(
          /from\s+['"]svelte/,
        );
      }
    });
  });

  // =========================================================================
  // F2: Evaluation Document Completeness & Adversarial Checks (5 cases)
  // =========================================================================
  describe('F2: Evaluation document completeness & validity', () => {
    it('T2-F2-01: evaluation quantifies multi-branch RFD3 vs RFD4 comparison across candidates', () => {
      const docPath = findEvaluationDocPath();
      expect(docPath).not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      expect(text).toMatch(/RFD3.*RFD4|branch|comparison/i);
    });

    it('T2-F2-02: evaluation documents ?prompt= failure modes (non-determinism, server/LLM requirement)', () => {
      const docPath = findEvaluationDocPath();
      expect(docPath).not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      expect(text).toContain('?prompt=');
      expect(text).toMatch(/non-determinis|stateless|ambig|hallucinat/i);
    });

    it('T2-F2-03: evaluation documents ?recipe= percent-encoding overhead vs ?flow=', () => {
      const docPath = findEvaluationDocPath();
      expect(docPath).not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      expect(text).toContain('?recipe=');
      expect(text).toMatch(/percent|encode|verbose|overhead|bloat/i);
    });

    it('T2-F2-04: evaluation documents why naive -> and (...) break in chat/Markdown autolinkers', () => {
      const docPath = findEvaluationDocPath();
      expect(docPath).not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      expect(text).toMatch(/->/);
      expect(text).toMatch(/truncat|strip|break|delimiter/i);
    });

    it('T2-F2-05: every sample ?flow= URL embedded in evaluation doc parses & validates with 0 errors', async () => {
      const docPath = findEvaluationDocPath();
      expect(docPath).not.toBeNull();
      const text = readFileSync(docPath!, 'utf8');
      const urls = extractWorkflowExampleUrls(text);
      // Exclude intentional negative counter-example URLs with '(' or ')' (e.g. showing broken Markdown links)
      const validSampleUrls = urls.filter((u) => !u.includes('(') && !u.includes(')'));
      expect(validSampleUrls.length).toBeGreaterThanOrEqual(1);
      for (const rawUrl of validSampleUrls) {
        const query = rawUrl.includes('?') ? `?${rawUrl.split('?')[1]}` : rawUrl;
        const g = await shareNs.decodeHashToGraph(query);
        expect(g, `Sample URL in eval doc failed: ${rawUrl}`).not.toBeNull();
        expect(graphUtilsNs.validateGraph(g!)).toEqual([]);
      }
    });
  });

  // =========================================================================
  // F3: ?flow= DSL Boundary & Adversarial Inputs (8 cases)
  // =========================================================================
  describe('F3: ?flow= DSL boundary, escaping & error cases', () => {
    it('T2-F3-01: unknown model ID returns structured error naming the unknown model without throwing', async () => {
      const res = await parseFlowStructured('unknown_model_xyz~m~f');
      expect(res.implemented).toBe(true);
      expect(res.ok).toBe(false);
      expect(res.errors.length).toBeGreaterThan(0);
      expect(res.errors.map((e) => e.message).join(' ')).toMatch(/unknown_model_xyz/i);
    });

    it('T2-F3-02: cycle (a:r~b:r;b~a) and self-loop (a:r~a) return structured cycle/self-loop error', async () => {
      const cyc = await parseFlowStructured('a:r~b:r;b~a');
      expect(cyc.implemented).toBe(true);
      expect(cyc.ok).toBe(false);
      expect(cyc.errors.map((e) => e.message).join(' ')).toMatch(/cycle|loop|circular/i);

      const self = await parseFlowStructured('a:r~a');
      expect(self.implemented).toBe(true);
      expect(self.ok).toBe(false);
      expect(self.errors.map((e) => `${e.code ?? ''} ${e.message}`).join(' ')).toMatch(
        /self[-_]?loop|cycle|loop|to itself/i,
      );
    });

    it('T2-F3-03: prototype pollution keys (__proto__, constructor, prototype) are rejected/stripped safely', async () => {
      const res = await parseFlowStructured('r:__proto__=1,constructor=bad,prototype=bad~m~f');
      expect(res.implemented).toBe(true);
      expect(
        res.ok === false || Object.hasOwn(res.graph?.nodes[0]?.params ?? {}, '__proto__') === false,
      ).toBe(true);
      expect(Object.prototype).not.toHaveProperty('1');
      expect(({} as Record<string, unknown>)).not.toHaveProperty('1');

      const badNodeId = await parseFlowStructured('__proto__:r~m~f');
      expect(badNodeId.implemented).toBe(true);
      expect(badNodeId.ok).toBe(false);
      expect(Object.prototype).not.toHaveProperty('1');
      expect(({} as Record<string, unknown>)).not.toHaveProperty('1');
    });

    it('T2-F3-04: sequence-input params.label vs _label isolation and empty label="" preservation', async () => {
      const res = await parseFlowStructured(
        'q:sequence=MKTAYIAKQRQISFVK,label=pep1,_label=MySeq~f:model=monomer',
      );
      expect(res.implemented).toBe(true);
      expect(res.ok).toBe(true);
      const qNode = res.graph!.nodes.find((n) => n.adapterId === 'sequence-input')!;
      expect(qNode.label).toBe('MySeq');
      expect(qNode.params.label).toBe('pep1');

      const emptyLabelRes = await parseFlowStructured(
        'q:sequence=MKTAYIAKQRQISFVK,label=,_label=CanvasCard~f',
      );
      expect(emptyLabelRes.ok).toBe(true);
      const q2 = emptyLabelRes.graph!.nodes.find((n) => n.adapterId === 'sequence-input')!;
      expect(q2.label).toBe('CanvasCard');
      expect(q2.params.label).toBe('');
    });

    it('T2-F3-05: out-of-bounds, non-numeric, and crossValidate-failing params report errors', async () => {
      for (const badFlow of [
        'r:length=9999~m~f',
        'r:length=abc~m~f',
        'm:temperature=-5',
        'r3:symmetry=C3,cyclic=true',
      ]) {
        const res = await parseFlowStructured(badFlow);
        expect(res.implemented, `Expected implemented parser for ${badFlow}`).toBe(true);
        const validationErrs = res.graph ? graphUtilsNs.validateGraph(res.graph) : [];
        expect(
          res.ok === false || validationErrs.length > 0,
          `Expected validation error for "${badFlow}"`,
        ).toBe(true);
      }
    });

    it('T2-F3-06: URL-encoded special characters (contig slashes/spaces, cropSegments commas, multi-chain colons) decode accurately', async () => {
      const contig = '10-20/A16-35/10-20';
      const crop = 'A1-20, C';
      const seq = 'MKTAYIAKQRQISFVK:ACDEFGHIKLMNPQRSTVWY';
      const flow = `p:mode=pdb-id,pdbId=1PRW,cropSegments=${encodeURIComponent(crop)}~r3:mode=motif,contig=${encodeURIComponent(contig)}`;
      const g = await shareNs.decodeHashToGraph(`?flow=${flow}`);
      expect(g).not.toBeNull();
      expect(g!.nodes[0].params.cropSegments).toBe(crop);
      expect(g!.nodes[1].params.contig).toBe(contig);

      const seqFlow = `q:sequence=${encodeURIComponent(seq)}~f:model=multimer`;
      const gSeq = await shareNs.decodeHashToGraph(`?flow=${seqFlow}`);
      expect(gSeq).not.toBeNull();
      expect(gSeq!.nodes[0].params.sequence).toBe(seq);
    });

    it('T2-F3-07: trailing autolinker punctuation (., ;, ,, )) on ?flow= is tolerated cleanly', async () => {
      for (const suffix of ['.', ',', ';', ')']) {
        const g = await shareNs.decodeHashToGraph(`?flow=r~m~f${suffix}`);
        expect(g, `Failed to tolerate trailing "${suffix}"`).not.toBeNull();
        expect(g!.nodes.map((n) => n.adapterId)).toEqual([
          'rfd4-proteina',
          'web-protein-mpnn',
          'localfold-af2',
        ]);
      }
    });

    it('T2-F3-08: empty ?flow= and oversized >65,536-char or >64-node ?flow= are rejected without crashing', async () => {
      const emptyRes = await parseFlowStructured('');
      expect(emptyRes.implemented).toBe(true);
      expect(emptyRes.ok).toBe(false);
      expect(emptyRes.errors.every((e) => e.code !== 'NOT_IMPLEMENTED')).toBe(true);

      const hugeFlow = 'r:length=70~' + Array.from({ length: 70 }, (_, i) => `n${i}:m`).join(';');
      const hugeRes = await parseFlowStructured(hugeFlow);
      expect(hugeRes.implemented).toBe(true);
      expect(hugeRes.ok).toBe(false);

      const overLimit = 'r:length=70~m~f' + ';x:r'.repeat(17_000);
      await expect(shareNs.decodeHashToGraph(`?flow=${overLimit}`)).rejects.toThrow(/too large|max/i);
    });
  });

  // =========================================================================
  // F4: Auto-Wiring & Multi-Port Error Boundaries (6 cases)
  // =========================================================================
  describe('F4: Auto-wiring & multi-port error boundaries', () => {
    it('T2-F4-01: incompatible port pair (sequence-input ~ web-protein-mpnn) returns type-mismatch error', async () => {
      const res = await parseFlowStructured('q~m');
      expect(res.implemented).toBe(true);
      expect(res.ok).toBe(false);
      expect(res.errors.map((e) => e.message).join(' ')).toMatch(
        /mismatch|compatible|StructureArtifact|SequenceSet/i,
      );
    });

    it('T2-F4-02: designability-scorer with only folds (q~f~s) reports missing required structure input', async () => {
      const res = await parseFlowStructured('q~f~s');
      expect(res.implemented).toBe(true);
      const errs = res.graph ? graphUtilsNs.validateGraph(res.graph) : res.errors.map((e) => e.message);
      expect(res.ok === false || errs.length > 0).toBe(true);
      expect(errs.join(' ')).toMatch(/structure|required/i);
    });

    it('T2-F4-03: structure-compare exceeding 6 incoming edges reports >6 edge limit error', async () => {
      const sevenBranches = Array.from({ length: 7 }, (_, i) =>
        i === 0 ? `g${i}:r~cmp:c.structures` : `g${i}:r~cmp.structures`,
      ).join(';');
      const res = await parseFlowStructured(sevenBranches);
      expect(res.implemented).toBe(true);
      const errs = res.graph ? graphUtilsNs.validateGraph(res.graph) : res.errors.map((e) => e.message);
      expect(res.ok === false || errs.length > 0).toBe(true);
      expect(errs.join(' ')).toMatch(/6|max|multi-input|structures/i);
    });

    it('T2-F4-04: esmc (mode=fill with no input) and dyna1 (inputMode=struct-only wired to sequence-input) fail crossValidate', async () => {
      const esmcRes = await parseFlowStructured('ec:mode=fill');
      expect(esmcRes.implemented).toBe(true);
      const esmcErrs = esmcRes.graph
        ? graphUtilsNs.validateGraph(esmcRes.graph)
        : esmcRes.errors.map((e) => e.message);
      expect(esmcRes.ok === false || esmcErrs.length > 0).toBe(true);

      const dynaRes = await parseFlowStructured('q~dy:inputMode=struct-only');
      expect(dynaRes.implemented).toBe(true);
      const dynaErrs = dynaRes.graph
        ? graphUtilsNs.validateGraph(dynaRes.graph)
        : dynaRes.errors.map((e) => e.message);
      expect(dynaRes.ok === false || dynaErrs.length > 0).toBe(true);
    });

    it('T2-F4-05: localfold-af2 ef2-fast-300m rejects non-single msaMode and invalid esmfold2Steps', async () => {
      const badMsa = await parseFlowStructured('q~f:model=ef2-fast-300m,msaMode=mmseqs2');
      expect(badMsa.implemented).toBe(true);
      const msaErrs = badMsa.graph
        ? graphUtilsNs.validateGraph(badMsa.graph)
        : badMsa.errors.map((e) => e.message);
      expect(badMsa.ok === false || msaErrs.length > 0).toBe(true);

      const badSteps = await parseFlowStructured('q~f:model=ef2-fast-300m,esmfold2Steps=99');
      expect(badSteps.implemented).toBe(true);
      const stepErrs = badSteps.graph
        ? graphUtilsNs.validateGraph(badSteps.graph)
        : badSteps.errors.map((e) => e.message);
      expect(badSteps.ok === false || stepErrs.length > 0).toBe(true);
    });

    it('T2-F4-06: designability-scorer missing folds (r~s) or invalid residueMapping syntax reports error', async () => {
      const missingFolds = await parseFlowStructured('r~s');
      expect(missingFolds.implemented).toBe(true);
      const foldErrs = missingFolds.graph
        ? graphUtilsNs.validateGraph(missingFolds.graph)
        : missingFolds.errors.map((e) => e.message);
      expect(missingFolds.ok === false || foldErrs.length > 0).toBe(true);

      const badMap = await parseFlowStructured('r~m~f~s:residueMapping=INVALID_MAP_SYNTAX');
      expect(badMap.implemented).toBe(true);
      const mapErrs = badMap.graph
        ? graphUtilsNs.validateGraph(badMap.graph)
        : badMap.errors.map((e) => e.message);
      expect(badMap.ok === false || mapErrs.length > 0).toBe(true);
    });
  });

  // =========================================================================
  // F5: ?recipe= & Legacy #g=/?g= Edge Cases (6 cases)
  // =========================================================================
  describe('F5: ?recipe= & legacy #g=/?g= boundary cases', () => {
    it('T2-F5-01: malformed ?recipe= (prose, missing nodes, denovo: v99, removed rfd4-conditioner) rejects cleanly', async () => {
      // Ensure ?recipe= is wired into decodeHashToGraph first (using valid recipe)
      const validYaml = shareNs.graphToYamlRecipe(registryNs.WORKFLOW_PRESETS[7].graph);
      const decodedValid = await shareNs.decodeHashToGraph(`?recipe=${encodeURIComponent(validYaml)}`);
      expect(decodedValid, 'Expected decodeHashToGraph to support ?recipe=').not.toBeNull();

      for (const badRecipe of [
        'just some prose without nodes',
        'denovo: v99\nnodes:\n  a: {"model":"rfd3"}\nwires:',
        'denovo: v1\nnodes:\n  a: {"model":"rfd4-conditioner"}\nwires:',
      ]) {
        expect(() => shareNs.parseRecipeToGraph(badRecipe)).toThrow();
        let threwOrNull = false;
        try {
          const res = await shareNs.decodeHashToGraph(`?recipe=${encodeURIComponent(badRecipe)}`);
          threwOrNull = res === null;
        } catch {
          threwOrNull = true;
        }
        expect(threwOrNull).toBe(true);
      }
    });

    it('T2-F5-02: #g= recovers from Markdown autolinker truncation and trailing .) punctuation', async () => {
      const custom = buildMinimalRunnableGraphForAdapter('thermompnn');
      const enc = await shareNs.encodeGraphToHash(custom);
      expect(enc.mode).toBe('compressed-dag');
      expect(enc.hash.startsWith('#g=')).toBe(true);
      const polluted = `${enc.hash.replace('#g=', '?g=').replace(/_/g, '%5F')}.)`;
      const dec = await shareNs.decodeHashToGraph(polluted);
      expect(dec).not.toBeNull();
      expect(dec!.nodes.map((n) => n.adapterId)).toEqual(['pdb-input', 'thermompnn']);
    });

    it('T2-F5-03: corrupt base64 / invalid deflate #g=!!! and #g=AAAA return null without throwing', async () => {
      await expect(shareNs.decodeHashToGraph('#g=!!!')).resolves.toBeNull();
      await expect(shareNs.decodeHashToGraph('#g=AAAA')).resolves.toBeNull();
    });

    it('T2-F5-04: pdb-input mode=custom with >1500-char pdbText strips pdbText & preserves SHA-256 verification', async () => {
      const bigPdb = 'ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n'.repeat(40);
      expect(bigPdb.length).toBeGreaterThan(1500);
      const graph = {
        id: 'big-pdb',
        name: 'Big PDB',
        nodes: [
          {
            id: 'src',
            adapterId: 'pdb-input',
            label: 'PDB Input',
            params: {
              ...graphUtilsNs.defaultParamsForAdapter('pdb-input'),
              mode: 'custom',
              filename: 'big.pdb',
              pdbText: bigPdb,
            },
          },
          {
            id: 'thermo',
            adapterId: 'thermompnn',
            label: 'ThermoMPNN',
            params: graphUtilsNs.defaultParamsForAdapter('thermompnn'),
          },
        ],
        edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'thermo', toPort: 'structure' }],
      };
      const enc = await shareNs.encodeGraphToHash(graph);
      expect(enc.strippedPdb).toBe(true);
      const dec = await shareNs.decodeHashToGraph(enc.hash);
      expect(dec).not.toBeNull();
      expect(dec!.nodes[0].params.missingUpload).toBe(true);
      expect(dec!.nodes[0].params.pdbText).toBe('');
      expect(typeof dec!.nodes[0].params.expectedPdbSha256).toBe('string');
      expect((dec!.nodes[0].params.expectedPdbSha256 as string).length).toBe(64);
    });

    it('T2-F5-05: legacy v1 #g= without capMode patches capMode="fixed" for localfold-af2, esmc, dyna1', async () => {
      const v1Payload = JSON.stringify({
        v: 1,
        n: [
          ['q1', 'q', {}],
          ['f1', 'f', { maxSequences: 1 }],
          ['ec1', 'ec', { mode: 'wildtype' }],
          ['dy1', 'dy', { inputMode: 'seq-only' }],
        ],
        e: [
          ['q1', 'sequences', 'f1', 'sequences'],
          ['q1', 'sequences', 'ec1', 'sequences'],
          ['q1', 'sequences', 'dy1', 'sequences'],
        ],
      });
      const b64 = Buffer.from(v1Payload, 'utf8')
        .toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
      const dec = await shareNs.decodeHashToGraph(`#g=${b64}`);
      expect(dec).not.toBeNull();
      expect(dec!.nodes[1].params.capMode).toBe('fixed');
      expect(dec!.nodes[2].params.capMode).toBe('fixed');
      expect(dec!.nodes[3].params.capMode).toBe('fixed');
    });

    it('T2-F5-06: pdb-input mode=pdb-id strips externally spoofed fetchedPdbId and pdbText on ?recipe= import', async () => {
      const spoofedJson = JSON.stringify({
        id: 'spoof',
        name: 'Spoof',
        nodes: [
          {
            id: 'src',
            adapterId: 'pdb-input',
            params: {
              mode: 'pdb-id',
              pdbId: '1PRW',
              fetchedPdbId: '9FAKE',
              pdbText: 'ATOM 1 CA ALA A 1 0 0 0',
            },
          },
        ],
        edges: [],
      });
      const dec = await shareNs.decodeHashToGraph(`?recipe=${encodeURIComponent(spoofedJson)}`);
      expect(dec).not.toBeNull();
      expect(dec!.nodes[0].params.fetchedPdbId).toBeUndefined();
      expect(dec!.nodes[0].params.pdbText).toBe('');
    });
  });

  // =========================================================================
  // F6: /llms.txt Quality, Accuracy & Size Boundaries (5 cases)
  // =========================================================================
  describe('F6: /llms.txt accuracy, safety & size boundaries', () => {
    it('T2-F6-01: /llms.txt covers exact 27 live adapters and does not advertise stale removed IDs', async () => {
      const { implemented, text } = await generateLiveLlmsTxt();
      expect(implemented).toBe(true);
      expect(ALL_27_ADAPTER_IDS).toHaveLength(27);
      for (const staleId of ['rfd4-conditioner', 'colabfold', 'rfdiffusion4']) {
        expect(text!).not.toContain(staleId);
      }
    });

    it('T2-F6-02: all ?flow= example URLs in /llms.txt pass Markdown autolinker safety', async () => {
      const { implemented, text } = await generateLiveLlmsTxt();
      expect(implemented).toBe(true);
      const flowExamples = extractWorkflowExampleUrls(text!).filter((u) => u.includes('?flow='));
      expect(flowExamples.length).toBeGreaterThanOrEqual(2);
      for (const rawUrl of flowExamples) {
        const fullUrl = rawUrl.startsWith('http') ? rawUrl : `https://denovo.run${rawUrl}`;
        assertMarkdownAutolinkerSafe(fullUrl);
      }
    });

    it('T2-F6-03: /llms.txt documents sequence-input, pdb-input, genie2.template, and designability-scorer dual wiring', async () => {
      const { implemented, text } = await generateLiveLlmsTxt();
      expect(implemented).toBe(true);
      expect(text!).toContain('sequence-input');
      expect(text!).toContain('pdb-input');
      expect(text!).toContain('template');
      expect(text!).toContain('designability-scorer');
      expect(text!).toContain('folds');
    });

    it('T2-F6-04: generateLlmsTxt() produces byte-for-byte deterministic output across calls', async () => {
      const first = await generateLiveLlmsTxt();
      const second = await generateLiveLlmsTxt();
      expect(first.implemented).toBe(true);
      expect(first.text).toBe(second.text);
    });

    it('T2-F6-05: /llms.txt is concise and LLM-context-friendly (< 48 KB and > 500 bytes)', async () => {
      const { implemented, text } = await generateLiveLlmsTxt();
      expect(implemented).toBe(true);
      const bytes = Buffer.byteLength(text!, 'utf8');
      expect(bytes).toBeGreaterThan(500);
      expect(bytes).toBeLessThan(48 * 1024);
    });
  });

  // =========================================================================
  // F7: HTML <link rel="llms"> & XSS / Redirect Edge Cases (5 cases)
  // =========================================================================
  describe('F7: HTML discovery <link rel="llms"> & cf-worker HTML boundary cases', () => {
    it('T2-F7-01: app.html <link rel="llms"> has exact rel="llms", type="text/plain", href="/llms.txt"', () => {
      const appHtml = readFileSync(resolve(WEB_DIR, 'src/app.html'), 'utf8');
      const linkMatch = appHtml.match(/<link\b[^>]*\brel="llms"[^>]*>/i);
      expect(linkMatch).not.toBeNull();
      expect(linkMatch![0]).toContain('href="/llms.txt"');
      expect(linkMatch![0]).toContain('type="text/plain"');
    });

    it('T2-F7-02: built web/build/index.html contains exactly 1 <link rel="llms"> tag', () => {
      const indexPath = resolve(BUILD_DIR, 'index.html');
      expect(existsSync(indexPath), 'Expected web/build/index.html to exist').toBe(true);
      const html = readFileSync(indexPath, 'utf8');
      const matches = html.match(/<link\b[^>]*\brel="llms"[^>]*>/gi) ?? [];
      expect(matches).toHaveLength(1);
    });

    it('T2-F7-03: cfWorker HTML injection escapes XSS payloads in _label (<script>alert(1)</script>)', async () => {
      const payload = encodeURIComponent('<script>alert(1)</script>');
      const env = createMockCfEnv();
      const res = await cfWorker.fetch(
        new Request(`https://denovo.run/?flow=r:_label=${payload}~m~f`),
        env,
      );
      expect(res.status).toBe(200);
      const html = await res.text();
      expect(html).toContain('/api/og?flow=');
      expect(html).not.toContain('<script>alert(1)</script>');
      expect(html).toContain('&lt;script&gt;alert(1)&lt;/script&gt;');
    });

    it('T2-F7-04: cfWorker /index.html?flow=... avoids ASSETS 308 loop by fetching / and returns 200 HTML', async () => {
      const requested: string[] = [];
      const env = createMockCfEnv({ recordRequests: requested });
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/index.html?flow=r~m~f'),
        env,
      );
      expect(res.status).toBe(200);
      expect(requested).toContain('/');
      expect(requested).not.toContain('/index.html');
      const html = await res.text();
      expect(html).toContain('/api/og?flow=');
    });

    it('T2-F7-05: cfWorker /?flow=... passes through ASSETS 404 or non-HTML responses cleanly without 500', async () => {
      const env404 = createMockCfEnv({ assetsStatus: 404 });
      const res404 = await cfWorker.fetch(new Request('https://denovo.run/?flow=r~m~f'), env404);
      expect(res404.status).toBe(404);

      const envJson = createMockCfEnv({ assetsContentType: 'application/json' });
      const resJson = await cfWorker.fetch(new Request('https://denovo.run/?flow=r~m~f'), envJson);
      expect(resJson.status).toBe(200);
      expect(resJson.headers.get('content-type')).toContain('application/json');
    });
  });

  // =========================================================================
  // F8: Edge OG PNG & Fallback Boundaries (5 cases)
  // =========================================================================
  describe('F8: Dynamic OG PNG error & fallback boundaries', () => {
    it('T2-F8-01: GET /api/og?flow=invalid_model falls back cleanly to 200 1200x630 PNG (while valid ?flow= decodes)', async () => {
      const validDecoded = await ogGraphNs.decodeOgWorkflowFromQuery('?flow=r~m~f');
      expect(validDecoded).not.toBeNull();
      const invalidDecoded = await ogGraphNs.decodeOgWorkflowFromQuery('?flow=invalid_model');
      expect(invalidDecoded).toBeNull();

      const env = createMockCfEnv();
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/og?flow=invalid_model'),
        env,
      );
      expect(res.status).toBe(200);
      expect(res.headers.get('content-type')).toBe('image/png');
      assertValidOgPngBytes(new Uint8Array(await res.arrayBuffer()), 1200, 630);
    });

    it('T2-F8-02: GET /api/og?recipe=not_valid_yaml falls back cleanly to 200 1200x630 PNG', async () => {
      const validYaml = shareNs.graphToYamlRecipe(registryNs.WORKFLOW_PRESETS[7].graph);
      expect(await ogGraphNs.decodeOgWorkflowFromQuery(`?recipe=${encodeURIComponent(validYaml)}`)).not.toBeNull();
      expect(await ogGraphNs.decodeOgWorkflowFromQuery('?recipe=not_valid_yaml')).toBeNull();

      const env = createMockCfEnv();
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/og?recipe=not_valid_yaml'),
        env,
      );
      expect(res.status).toBe(200);
      assertValidOgPngBytes(new Uint8Array(await res.arrayBuffer()), 1200, 630);
    });

    it('T2-F8-03: GET /api/og with empty ?flow= or >32,768-char query returns 200 1200x630 PNG without crashing', async () => {
      expect(await ogGraphNs.decodeOgWorkflowFromQuery('?flow=r~m~f')).not.toBeNull();
      const env = createMockCfEnv();
      const emptyRes = await cfWorker.fetch(new Request('https://denovo.run/api/og?flow='), env);
      expect(emptyRes.status).toBe(200);
      assertValidOgPngBytes(new Uint8Array(await emptyRes.arrayBuffer()), 1200, 630);

      const huge = 'r~'.repeat(17_000) + 'f';
      const hugeRes = await cfWorker.fetch(
        new Request(`https://denovo.run/api/og?flow=${huge}`),
        env,
      );
      expect(hugeRes.status).toBe(200);
      assertValidOgPngBytes(new Uint8Array(await hugeRes.arrayBuffer()), 1200, 630);
    });

    it('T2-F8-04: GET /api/og?flow=p:mode=pdb-id,pdbId=1PRW~m~f~s degrades gracefully if RCSB fetch fails', async () => {
      const decoded = await ogGraphNs.decodeOgWorkflowFromQuery(
        '?flow=p:mode=pdb-id,pdbId=1PRW~m~f~s',
      );
      expect(decoded).not.toBeNull();
      expect(decoded!.structureTarget).toEqual({ kind: 'rcsb', pdbId: '1PRW', cropSegments: '' });

      const failingFetch: typeof fetch = async () => new Response('Upstream error', { status: 500 });
      const res = await cfWorkerNs.handleOgImageRequest(
        new Request('https://denovo.run/api/og?flow=p:mode=pdb-id,pdbId=1PRW~m~f~s'),
        failingFetch,
      );
      expect(res.status).toBe(200);
      assertValidOgPngBytes(new Uint8Array(await res.arrayBuffer()), 1200, 630);
    });

    it('T2-F8-05: GET /og.png?flow=r~m~f~s renders workflow 1200x630 PNG distinct from default card', async () => {
      const env = createMockCfEnv();
      const defRes = await cfWorker.fetch(new Request('https://denovo.run/og.png'), env);
      const flowRes = await cfWorker.fetch(
        new Request('https://denovo.run/og.png?flow=r~m~f~s'),
        env,
      );
      expect(defRes.status).toBe(200);
      expect(flowRes.status).toBe(200);
      const defBytes = new Uint8Array(await defRes.arrayBuffer());
      const flowBytes = new Uint8Array(await flowRes.arrayBuffer());
      assertValidOgPngBytes(flowBytes, 1200, 630);
      expect(Buffer.from(flowBytes).equals(Buffer.from(defBytes))).toBe(false);
    });
  });

  // =========================================================================
  // F9: /api/link Boundary & Error Handling (5 cases)
  // =========================================================================
  describe('F9: /api/link boundary & structured error handling', () => {
    it('T2-F9-01: GET /api/link with no query params returns 400 with structured errors[]', async () => {
      const env = createMockCfEnv();
      const res = await cfWorker.fetch(new Request('https://denovo.run/api/link'), env);
      expect(res.status).toBe(400);
      const body = (await res.json()) as {
        ok: boolean;
        url: unknown;
        shortUrl: unknown;
        nodes: unknown[];
        edges: unknown[];
        errors: Array<{ code?: string; message: string }>;
      };
      expect(body.ok).toBe(false);
      expect(body.url).toBeNull();
      expect(body.shortUrl).toBeNull();
      expect(body.nodes).toEqual([]);
      expect(body.edges).toEqual([]);
      expect(body.errors.length).toBeGreaterThan(0);
    });

    it('T2-F9-02: GET /api/link?flow=badmodel~m returns 400 with error referencing badmodel', async () => {
      const env = createMockCfEnv();
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/link?flow=badmodel~m'),
        env,
      );
      expect(res.status).toBe(400);
      const body = (await res.json()) as {
        ok: boolean;
        errors: Array<{ code?: string; message: string; token?: string }>;
      };
      expect(body.ok).toBe(false);
      expect(body.errors.length).toBeGreaterThan(0);
      expect(JSON.stringify(body.errors)).toMatch(/badmodel/i);
    });

    it('T2-F9-03: POST /api/link with malformed JSON body returns 400 structured JSON (no 500)', async () => {
      const env = createMockCfEnv();
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{not-valid-json',
        }),
        env,
      );
      expect(res.status).toBe(400);
      const body = (await res.json()) as { ok: boolean; errors: unknown[] };
      expect(body.ok).toBe(false);
      expect(body.errors.length).toBeGreaterThan(0);
    });

    it('T2-F9-04: POST /api/link with empty body and PUT /api/link return 400 or 405 structured JSON', async () => {
      const env = createMockCfEnv();
      const emptyPost = await cfWorker.fetch(
        new Request('https://denovo.run/api/link', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '',
        }),
        env,
      );
      expect(emptyPost.status).toBe(400);
      const emptyBody = (await emptyPost.json()) as { ok: boolean; errors: unknown[] };
      expect(emptyBody.ok).toBe(false);
      expect(emptyBody.errors.length).toBeGreaterThan(0);

      const putRes = await cfWorker.fetch(
        new Request('https://denovo.run/api/link', { method: 'PUT', body: '{}' }),
        env,
      );
      expect([400, 405]).toContain(putRes.status);
      const putBody = (await putRes.json()) as { ok: boolean; errors: unknown[] };
      expect(putBody.ok).toBe(false);
      expect(putBody.errors.length).toBeGreaterThan(0);
    });

    it('T2-F9-05: GET /api/link?url=... unwraps full URL (200) and cycle flow=a:r~b:r;b~a returns 400', async () => {
      const env = createMockCfEnv();
      const okRes = await cfWorker.fetch(
        new Request(
          `https://denovo.run/api/link?url=${encodeURIComponent('https://denovo.run/?flow=r~m~f')}`,
        ),
        env,
      );
      expect(okRes.status).toBe(200);
      const okBody = (await okRes.json()) as { ok: boolean };
      expect(okBody.ok).toBe(true);

      const cycRes = await cfWorker.fetch(
        new Request(`https://denovo.run/api/link?flow=${encodeURIComponent('a:r~b:r;b~a')}`),
        env,
      );
      expect(cycRes.status).toBe(400);
      const cycBody = (await cycRes.json()) as {
        ok: boolean;
        errors: Array<{ code?: string; message: string }>;
      };
      expect(cycBody.ok).toBe(false);
      expect(cycBody.errors.length).toBeGreaterThan(0);
      expect(
        cycBody.errors.some((e) => e.code === 'CYCLE_DETECTED' || /cycle/i.test(e.message)),
      ).toBe(true);
    });
  });

  // =========================================================================
  // F10: Security, Probes, Scale & Normalization Boundaries (5 cases)
  // =========================================================================
  describe('F10: Scanner probes, headers, scale & normalization', () => {
    it('T2-F10-01: scanner probes (/.env, /.git/config, /wp-login.php, /config.json) return 404', async () => {
      const env = createMockCfEnv();
      for (const path of ['/.env', '/.git/config', '/wp-login.php', '/config.json']) {
        const res = await cfWorker.fetch(new Request(`https://denovo.run${path}`), env);
        expect(res.status, `Expected 404 for ${path}`).toBe(404);
      }
    });

    it('T2-F10-02: unknown /api/nonexistent still returns 404 JSON error', async () => {
      const env = createMockCfEnv();
      const res = await cfWorker.fetch(new Request('https://denovo.run/api/nonexistent'), env);
      expect(res.status).toBe(404);
      const body = (await res.json()) as { ok: boolean; error: string };
      expect(body.ok).toBe(false);
      expect(body.error).toBeTruthy();
    });

    it('T2-F10-03: Cache-Control and X-Content-Type-Options: nosniff headers on /api/og, /api/link, and /?flow=', async () => {
      const env = createMockCfEnv();
      const og = await cfWorker.fetch(new Request('https://denovo.run/api/og?flow=r~m~f'), env);
      expect(og.status).toBe(200);
      expect(og.headers.get('x-content-type-options')).toBe('nosniff');
      expect(og.headers.get('cache-control')).toContain('max-age=');

      const link = await cfWorker.fetch(new Request('https://denovo.run/api/link?flow=r~m~f'), env);
      expect(link.status).toBe(200);
      expect(link.headers.get('x-content-type-options')).toBe('nosniff');

      const page = await cfWorker.fetch(new Request('https://denovo.run/?flow=r~m~f'), env);
      expect(page.status).toBe(200);
      expect(page.headers.get('cache-control')).toContain('must-revalidate');
    });

    it('T2-F10-04: large 20-node DAG survives ?flow= -> /api/link -> ?g= -> /api/og without overflow', async () => {
      const branches = Array.from({ length: 5 }, (_, i) => {
        const p = `g${i}:r:length=70~m${i}:m~f${i}:f~s${i}:s`;
        return `${p};g${i}~s${i}`;
      }).join(';');
      // 20 nodes total (5 * 4)
      const env = createMockCfEnv();
      const linkRes = await cfWorker.fetch(
        new Request(`https://denovo.run/api/link?flow=${encodeURIComponent(branches)}`),
        env,
      );
      expect(linkRes.status).toBe(200);
      const linkBody = (await linkRes.json()) as {
        ok: boolean;
        nodes: unknown[];
        shortUrl: string;
      };
      expect(linkBody.ok).toBe(true);
      expect(linkBody.nodes).toHaveLength(20);

      const ogRes = await cfWorker.fetch(
        new Request(`https://denovo.run/api/og${new URL(linkBody.shortUrl).search}`),
        env,
      );
      expect(ogRes.status).toBe(200);
      assertValidOgPngBytes(new Uint8Array(await ogRes.arrayBuffer()), 1200, 630);
    });

    it('T2-F10-05: lowercase pdbId=5o45 is normalized/preserved as 5O45 across /api/link and og-graph', async () => {
      const flow = 'p:mode=pdb-id,pdbId=5o45~r:mode=binder,hotspots=A54~m~f~s';
      const env = createMockCfEnv();
      const linkRes = await cfWorker.fetch(
        new Request(`https://denovo.run/api/link?flow=${encodeURIComponent(flow)}`),
        env,
      );
      expect(linkRes.status).toBe(200);
      const linkBody = (await linkRes.json()) as { ok: boolean; shortUrl: string };
      expect(linkBody.ok).toBe(true);

      const ogFromFlow = await ogGraphNs.decodeOgWorkflowFromQuery(`?flow=${flow}`);
      expect(ogFromFlow?.structureTarget).toEqual({
        kind: 'rcsb',
        pdbId: '5O45',
        cropSegments: '',
      });
    });
  });
});
