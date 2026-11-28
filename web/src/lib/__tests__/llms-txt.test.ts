import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { BENCHMARK_TARGETS } from '$lib/engine/benchmarks';
import {
  ADAPTER_REGISTRY,
  ESMFOLD2_STEPS,
  LOCALFOLD_MODEL_OPTIONS,
  MPNN_VARIANT_OPTIONS,
  WORKFLOW_PRESETS,
} from '$lib/engine/registry';
import {
  ADAPTER_TO_SHORT,
  FLOW_MODEL_ALIASES,
  LOCALFOLD_MODEL_ALIASES,
  decodeFlowQuery,
  encodeGraphToFlow,
  isMarkdownSafeFlowUrl,
} from '$lib/engine/flow-dsl';
import { validateGraph } from '$lib/engine/graph-utils';
import { generateLlmsTxt } from '$lib/engine/llms-txt';
import { decodeHashToGraph } from '$lib/engine/share';
import type { PortType } from '$lib/engine/types';

const WEB_ROOT = resolve(__dirname, '../../..');

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function extractExampleUrls(text: string): string[] {
  const matches = text.match(/(?:https:\/\/denovo\.run)?\/\?(?:flow|recipe)=[^\s`"'<>)\]]+/g) ?? [];
  return matches.map((m) => m.replace(/[.,;:!?]+$/, ''));
}

describe('engine/llms-txt', () => {
  it('covers 100% of live ADAPTER_REGISTRY models, short codes, ports, PortTypes, and primary parameters with zero stale IDs', () => {
    const text = generateLlmsTxt();
    const adapterIds = Object.keys(ADAPTER_REGISTRY);
    expect(adapterIds).toHaveLength(32);

    for (const id of adapterIds) {
      expect(text).toContain(id);
      const shortCode = ADAPTER_TO_SHORT[id];
      expect(shortCode).toBeTruthy();
      expect(text).toMatch(new RegExp(`(?:^|[^a-z0-9])${escapeRegExp(shortCode)}(?:$|[^a-z0-9])`));

      const manifest = ADAPTER_REGISTRY[id];
      for (const [portName, portSpec] of Object.entries(manifest.inputs)) {
        expect(text).toContain(portName);
        expect(text).toContain(portSpec.type);
      }
      for (const [portName, portSpec] of Object.entries(manifest.outputs)) {
        expect(text).toContain(portName);
        expect(text).toContain(portSpec.type);
      }
      for (const field of manifest.paramSchema ?? []) {
        if (field.tier === 'primary' || field.tier === 'advanced') {
          expect(text, `Missing ${field.tier} param ${id}.${field.key}`).toContain(field.key);
        }
      }
    }

    expect(text).toContain('nsteps');
    expect(text).toContain('seed');
    expect(text).toContain('earlyStop');

    const allPortTypes: PortType[] = [
      'StructureArtifact',
      'TrajectoryFrames',
      'SequenceSet',
      'FoldedStructure',
      'DesignabilityMetrics',
      'MutationMatrix',
      'ResidueTrack',
      'SequenceEmbedding',
      'ConditionSpec',
    ];
    for (const portType of allPortTypes) {
      expect(text).toContain(portType);
    }

    for (const staleId of ['rfd4-conditioner', 'colabfold', 'rfdiffusion4']) {
      expect(text).not.toContain(staleId);
    }
  });

  it('documents all FLOW_MODEL_ALIASES, LOCALFOLD_MODEL_ALIASES, 20 LocalFold checkpoints, 3 MPNN variants, and 8 presets', () => {
    const text = generateLlmsTxt();

    for (const alias of Object.keys(FLOW_MODEL_ALIASES)) {
      expect(text).toContain(alias);
    }
    for (const alias of Object.keys(LOCALFOLD_MODEL_ALIASES)) {
      expect(text).toContain(alias);
    }

    expect(LOCALFOLD_MODEL_OPTIONS).toHaveLength(20);
    for (const opt of LOCALFOLD_MODEL_OPTIONS) {
      expect(text).toContain(opt.id);
    }
    for (const step of ESMFOLD2_STEPS) {
      expect(text).toContain(String(step));
    }

    expect(MPNN_VARIANT_OPTIONS).toHaveLength(3);
    for (const variant of MPNN_VARIANT_OPTIONS) {
      expect(text).toContain(variant.id);
    }

    expect(WORKFLOW_PRESETS).toHaveLength(8);
    for (const preset of WORKFLOW_PRESETS) {
      expect(text).toContain(preset.id);
    }

    expect(BENCHMARK_TARGETS).toHaveLength(80);
    for (const bench of BENCHMARK_TARGETS) {
      expect(text, `Missing benchmark target ${bench.id}`).toContain(bench.id);
      expect(text).toContain(bench.pdbId);
    }

    // Host-GPU headless runner API + remote-codec + recipe schema + scorer gates
    for (const hostToken of [
      '/api/health',
      '/api/run-node',
      '/api/cancel',
      '__denovoTypedArray',
      '__denovoNum',
      'scRmsd < 2.0',
      'plddt >= 70',
      'denovo: v1',
      '%3B',
    ]) {
      expect(text).toContain(hostToken);
    }
  });

  it('documents ?flow= syntax, auto-wiring semantics, fallback formats, and edge APIs without broken template slashes', () => {
    const text = generateLlmsTxt();

    for (const token of [
      '?flow=',
      '?recipe=',
      '?p=',
      '?g=',
      '_label=',
      '~',
      ':',
      '=',
      ',',
      ';',
      '.',
      '/api/link',
      '/api/og',
      '<link rel="llms"',
      'designability-scorer',
      'folds',
      'structure',
      'genie2',
      'template',
      'structure-compare',
      'reference',
      'structures',
      'esmc',
      'dyna1',
      'sequence-input',
      'pdb-input',
    ]) {
      expect(text).toContain(token);
    }

    // Never emit broken template URLs starting with `/?flow=<` or `/?recipe=<`
    expect(text).not.toMatch(/\/\?(?:flow|recipe)=</);
  });

  it('extracts every workflow URL, decodes & validates with 0 errors, and verifies Markdown autolinker safety + round-trip', async () => {
    const text = generateLlmsTxt();
    const urls = extractExampleUrls(text);
    expect(urls.length).toBeGreaterThanOrEqual(21);

    for (const url of urls) {
      expect(isMarkdownSafeFlowUrl(url)).toBe(true);

      // Markdown autolinker / wrapper safety checks
      expect(url).not.toMatch(/[\s()[\]<>{}"'`\\]/);
      expect(url).not.toMatch(/[.,;:!?_~-]$/);
      expect(url).not.toContain('~~');

      const parsed = decodeFlowQuery(url);
      expect(parsed.ok, `Expected ok=true for ${url}: ${JSON.stringify(parsed.errors)}`).toBe(true);
      expect(parsed.errors).toEqual([]);
      expect(parsed.graph).not.toBeNull();

      const validationErrors = validateGraph(parsed.graph!);
      expect(validationErrors, `Validation errors for ${url}`).toEqual([]);

      const queryPart = '?' + url.split('?')[1];
      const decoded = await decodeHashToGraph(queryPart);
      expect(decoded).not.toBeNull();
      expect(validateGraph(decoded!)).toEqual([]);

      const reencoded = encodeGraphToFlow(parsed.graph!);
      const roundTripUrl = `https://denovo.run/?${reencoded.query}`;
      expect(isMarkdownSafeFlowUrl(roundTripUrl)).toBe(true);

      const reparsed = decodeFlowQuery(roundTripUrl);
      expect(reparsed.ok).toBe(true);
      expect(reparsed.errors).toEqual([]);
      expect(reparsed.graph?.nodes.map((n) => n.adapterId).sort()).toEqual(
        parsed.graph!.nodes.map((n) => n.adapterId).sort(),
      );
      expect(reparsed.graph?.edges.length).toBe(parsed.graph!.edges.length);
    }
  });

  it('is byte-for-byte deterministic, within size bounds, and respects generation options', () => {
    const first = generateLlmsTxt();
    const second = generateLlmsTxt();
    expect(first).toBe(second);

    const bytes = Buffer.byteLength(first, 'utf8');
    expect(bytes).toBeGreaterThan(500);
    expect(bytes).toBeLessThan(64 * 1024);
    expect(first).toContain('Advanced params:');

    const compact = generateLlmsTxt({ includeAdvancedParams: false });
    expect(compact).not.toContain('Advanced params:');

    const staging = generateLlmsTxt({
      origin: 'https://staging.denovo.run/',
      includePresets: false,
      includeAdvancedParams: true,
    });
    expect(staging).toContain('https://staging.denovo.run/?flow=');
    expect(staging).not.toContain('https://denovo.run/?flow=');
    expect(staging).not.toContain('## Built-In Workflow Presets');
    expect(staging).toContain('Advanced params:');
  });

  it('includes HTML <link rel="llms"> + body WebFetch discovery, Link header, and robots.txt (and matches built artifacts when present)', () => {
    const appHtml = readFileSync(resolve(WEB_ROOT, 'src/app.html'), 'utf8');
    expect(appHtml).toContain('<link rel="icon"');
    expect(appHtml).toContain('og:title');
    expect(appHtml).toContain('<link rel="llms" type="text/plain" href="/llms.txt" />');
    expect(appHtml).toContain('denovo-llm-discovery');
    expect(appHtml).toContain('https://denovo.run/llms.txt');

    const headersFile = readFileSync(resolve(WEB_ROOT, 'static/_headers'), 'utf8');
    expect(headersFile).toContain('/llms.txt');
    expect(headersFile).toContain('Content-Type: text/plain; charset=utf-8');
    expect(headersFile).toContain('Link: </llms.txt>; rel="llms"');

    const robotsFile = readFileSync(resolve(WEB_ROOT, 'static/robots.txt'), 'utf8');
    expect(robotsFile).toContain('https://denovo.run/llms.txt');

    const buildDir = resolve(WEB_ROOT, 'build');
    const buildLlms = resolve(buildDir, 'llms.txt');
    const buildIndex = resolve(buildDir, 'index.html');
    const buildRunner = resolve(buildDir, 'runner.html');
    const buildRoutes = resolve(buildDir, '_routes.json');
    const buildHeaders = resolve(buildDir, '_headers');

    if (existsSync(buildLlms)) {
      expect(readFileSync(buildLlms, 'utf8')).toBe(generateLlmsTxt());
    }
    if (existsSync(buildIndex)) {
      expect(readFileSync(buildIndex, 'utf8')).toContain('<link rel="llms" type="text/plain" href="/llms.txt" />');
    }
    if (existsSync(buildRunner)) {
      expect(readFileSync(buildRunner, 'utf8')).toContain('<link rel="llms" type="text/plain" href="/llms.txt" />');
    }
    if (existsSync(buildRoutes)) {
      const routesJson = JSON.parse(readFileSync(buildRoutes, 'utf8')) as { exclude?: string[] };
      expect(routesJson.exclude).toContain('/llms.txt');
    }
    if (existsSync(buildHeaders)) {
      const builtHeaders = readFileSync(buildHeaders, 'utf8');
      expect(builtHeaders).toContain('/llms.txt');
      expect(builtHeaders).toContain('Content-Type: text/plain; charset=utf-8');
    }
  });
});
