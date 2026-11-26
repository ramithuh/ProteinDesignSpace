import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect } from 'vitest';

import * as shareNs from '$lib/engine/share';
import * as registryNs from '$lib/engine/registry';
import * as graphUtilsNs from '$lib/engine/graph-utils';
import * as ogGraphNs from '$lib/og/og-graph';
import cfWorker, * as cfWorkerNs from '$lib/og/cf-worker';
import * as renderOgNs from '$lib/og/render-og-card';
import * as pageHelpersNs from '$lib/ui/page-state-helpers';

import type { WorkflowEdge, WorkflowGraph, WorkflowNode } from '$lib/engine/types';
import type { CfPagesEnv } from '$lib/og/cf-worker';
import type { DecodedOgWorkflow } from '$lib/og/og-graph';

export { shareNs, registryNs, graphUtilsNs, ogGraphNs, cfWorker, cfWorkerNs, renderOgNs, pageHelpersNs };
export type { WorkflowEdge, WorkflowGraph, WorkflowNode, CfPagesEnv, DecodedOgWorkflow };

export const WEB_DIR = resolve(__dirname, '../..');
export const REPO_ROOT = resolve(WEB_DIR, '..');
export const ORIGINAL_MAIN_DIR = '/home/ruh/ramith/personal_projects/ProteinDesignSpace';
export const BUILD_DIR = resolve(WEB_DIR, 'build');
export const STATIC_DIR = resolve(WEB_DIR, 'static');
export const TOOLS_DIR = resolve(REPO_ROOT, 'tools');

export const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10] as const;

export const ALL_29_ADAPTER_IDS = [
  'pdb-input',
  'sequence-input',
  'rfd4-proteina',
  'genie3',
  'protpardelle',
  'rfdiffusion',
  'rfd3',
  'boltzgen',
  'evodiff',
  'la-proteina',
  'proteina',
  'sid-protein',
  'genie2',
  'frameflow',
  'foldflow2',
  'web-protein-mpnn',
  'ligandmpnn',
  'lasermpnn',
  'antifold-esmif1',
  'localfold-af2',
  'thermompnn',
  'esmc',
  'dyna1',
  'gnina',
  'top-k-filter',
  'structure-compare',
  'designability-scorer',
  'pulchra',
  'cg2all',
] as const;

export const ALL_29_SHORT_CODES: Record<string, string> = {
  'pdb-input': 'p',
  'sequence-input': 'q',
  'rfd4-proteina': 'r',
  genie3: 'g',
  protpardelle: 'pp',
  rfdiffusion: 'd',
  rfd3: 'r3',
  boltzgen: 'b',
  evodiff: 'e',
  'la-proteina': 'lp',
  proteina: 'pr',
  'sid-protein': 'sp',
  genie2: 'g2',
  frameflow: 'ff',
  foldflow2: 'f2',
  'web-protein-mpnn': 'm',
  ligandmpnn: 'lm',
  lasermpnn: 'la',
  'antifold-esmif1': 'af',
  'localfold-af2': 'f',
  thermompnn: 'tm',
  esmc: 'ec',
  dyna1: 'dy',
  gnina: 'gn',
  'top-k-filter': 'tk',
  'designability-scorer': 's',
  'structure-compare': 'c',
  pulchra: 'pu',
  cg2all: 'cg',
};

// Type-safe dynamic loader for uncreated M1/M2 engine modules (`flow-dsl.ts`, `llms-txt.ts`)
// Using import.meta.glob avoids TS2307 in `svelte-check` before M1/M2 land.
const engineModules = import.meta.glob<Record<string, unknown>>('/src/lib/engine/*.ts');

export async function loadOptionalEngineModule(
  filename: 'flow-dsl.ts' | 'llms-txt.ts',
): Promise<Record<string, unknown> | null> {
  const loader = engineModules[`/src/lib/engine/${filename}`];
  if (!loader) return null;
  return await loader();
}

export interface FlowValidationErrorLike {
  code?: string;
  message: string;
  token?: string;
  nodeId?: string;
}

export interface StructuredFlowResult {
  implemented: boolean;
  ok: boolean;
  graph: WorkflowGraph | null;
  nodes: unknown[];
  edges: unknown[];
  errors: FlowValidationErrorLike[];
}

export async function getFlowDslModuleOrShare(): Promise<{
  flowMod: Record<string, unknown> | null;
  shareAny: Record<string, unknown>;
}> {
  const flowMod = await loadOptionalEngineModule('flow-dsl.ts');
  const shareAny = shareNs as unknown as Record<string, unknown>;
  return { flowMod, shareAny };
}

function normalizeErrors(rawErrors: unknown): FlowValidationErrorLike[] {
  if (!Array.isArray(rawErrors)) return [];
  return rawErrors.map((err) => {
    if (typeof err === 'string') return { message: err };
    if (err && typeof err === 'object') {
      const e = err as Record<string, unknown>;
      return {
        code: typeof e.code === 'string' ? e.code : undefined,
        message: typeof e.message === 'string' ? e.message : JSON.stringify(e),
        token: typeof e.token === 'string' ? e.token : undefined,
        nodeId: typeof e.nodeId === 'string' ? e.nodeId : undefined,
      };
    }
    return { message: String(err) };
  });
}

function isGraphLike(v: unknown): v is WorkflowGraph {
  return Boolean(
    v &&
      typeof v === 'object' &&
      Array.isArray((v as WorkflowGraph).nodes) &&
      Array.isArray((v as WorkflowGraph).edges),
  );
}

/**
 * Opaque-box adapter to invoke M1 structured `?flow=` parser/validator without hardcoding
 * internal symbol names. Checks `flow-dsl.ts` first, then `share.ts`.
 */
export async function parseFlowStructured(rawFlowOrQuery: string): Promise<StructuredFlowResult> {
  const { flowMod, shareAny } = await getFlowDslModuleOrShare();
  const candidates = [
    'parseFlowDsl',
    'decodeFlowDsl',
    'parseFlowToGraph',
    'parseFlow',
    'validateFlowDsl',
    'decodeFlowToGraph',
    'flowToGraph',
  ];

  for (const mod of [flowMod, shareAny]) {
    if (!mod) continue;
    for (const fnName of candidates) {
      const fn = mod[fnName];
      if (typeof fn !== 'function') continue;
      try {
        const res = await (fn as (input: string) => unknown)(rawFlowOrQuery);
        if (res && typeof res === 'object' && 'ok' in res) {
          const obj = res as Record<string, unknown>;
          const graph = isGraphLike(obj.graph) ? obj.graph : null;
          const nodes = Array.isArray(obj.nodes) ? obj.nodes : graph?.nodes ?? [];
          const edges = Array.isArray(obj.edges) ? obj.edges : graph?.edges ?? [];
          const errors = normalizeErrors(obj.errors);
          return {
            implemented: true,
            ok: Boolean(obj.ok),
            graph,
            nodes,
            edges,
            errors,
          };
        }
        if (isGraphLike(res)) {
          return {
            implemented: true,
            ok: true,
            graph: res,
            nodes: res.nodes,
            edges: res.edges,
            errors: [],
          };
        }
      } catch (err) {
        return {
          implemented: true,
          ok: false,
          graph: null,
          nodes: [],
          edges: [],
          errors: [{ code: 'PARSE_ERROR', message: err instanceof Error ? err.message : String(err) }],
        };
      }
    }
  }

  // Fallback check via decodeHashToGraph if `?flow=` is handled there without separate structured export
  const normalizedQuery =
    rawFlowOrQuery.startsWith('?flow=') || rawFlowOrQuery.startsWith('#flow=')
      ? rawFlowOrQuery
      : `?flow=${rawFlowOrQuery.replace(/^flow=/, '')}`;

  try {
    const graph = await shareNs.decodeHashToGraph(normalizedQuery);
    if (graph) {
      return {
        implemented: true,
        ok: true,
        graph,
        nodes: graph.nodes,
        edges: graph.edges,
        errors: [],
      };
    }
  } catch (err) {
    return {
      implemented: flowMod !== null,
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [{ code: 'DECODE_ERROR', message: err instanceof Error ? err.message : String(err) }],
    };
  }

  return {
    implemented: flowMod !== null,
    ok: false,
    graph: null,
    nodes: [],
    edges: [],
    errors: [
      {
        code: flowMod ? 'INVALID_FLOW' : 'NOT_IMPLEMENTED',
        message: flowMod
          ? 'Flow DSL returned null / invalid'
          : 'M1 flow-dsl / ?flow= parser is not implemented yet',
      },
    ],
  };
}

/**
 * Opaque-box adapter to encode a `WorkflowGraph` into a primary `?flow=` DSL / URL string.
 */
export async function encodeGraphToFlowQuery(graph: WorkflowGraph): Promise<{
  implemented: boolean;
  raw: string | null;
  dsl: string | null;
  url: string | null;
}> {
  const { flowMod, shareAny } = await getFlowDslModuleOrShare();
  const candidates = [
    'encodeGraphToFlow',
    'graphToFlowDsl',
    'encodeFlowDsl',
    'graphToFlow',
    'encodeWorkflowToFlow',
  ];

  for (const mod of [shareAny, flowMod]) {
    if (!mod) continue;
    for (const fnName of candidates) {
      const fn = mod[fnName];
      if (typeof fn !== 'function') continue;
      const res = await (fn as (g: WorkflowGraph) => unknown)(graph);
      let rawStr: string | null = null;
      if (typeof res === 'string') {
        rawStr = res;
      } else if (res && typeof res === 'object') {
        const obj = res as Record<string, unknown>;
        for (const k of ['flow', 'dsl', 'query', 'hash', 'url']) {
          if (typeof obj[k] === 'string') {
            rawStr = obj[k] as string;
            break;
          }
        }
      }
      if (rawStr !== null) {
        let dsl = rawStr;
        if (rawStr.startsWith('http://') || rawStr.startsWith('https://')) {
          const u = new URL(rawStr);
          dsl = u.searchParams.get('flow') ?? rawStr;
        } else if (/^[#?]flow=/.test(rawStr)) {
          dsl = rawStr.replace(/^[#?]flow=/, '');
        } else if (rawStr.startsWith('flow=')) {
          dsl = rawStr.slice(5);
        }
        const url =
          rawStr.startsWith('http://') || rawStr.startsWith('https://')
            ? rawStr
            : shareNs.buildShareableWorkflowUrl('https://denovo.run', '/', `?flow=${dsl}`);
        return { implemented: true, raw: rawStr, dsl, url };
      }
    }
  }

  return { implemented: false, raw: null, dsl: null, url: null };
}

/**
 * Opaque-box loader for M2 `llms-txt.ts` `generateLlmsTxt()`.
 */
export async function generateLiveLlmsTxt(): Promise<{ implemented: boolean; text: string | null }> {
  const mod = await loadOptionalEngineModule('llms-txt.ts');
  if (!mod) return { implemented: false, text: null };
  for (const name of ['generateLlmsTxt', 'buildLlmsTxt', 'renderLlmsTxt']) {
    const fn = mod[name];
    if (typeof fn === 'function') {
      const out = await (fn as () => unknown)();
      if (typeof out === 'string') return { implemented: true, text: out };
    }
  }
  return { implemented: false, text: null };
}

/**
 * Extract all `https://denovo.run/?flow=...` or `https://denovo.run/?recipe=...` (and relative `/?flow=...`)
 * example URLs from `/llms.txt` or evaluation markdown.
 */
export function extractWorkflowExampleUrls(markdownOrText: string): string[] {
  const found = new Set<string>();
  const urlRe = /(?:https:\/\/denovo\.run)?\/\?(?:flow|recipe)=[^\s`"'<>)\]]+/g;
  for (const match of markdownOrText.matchAll(urlRe)) {
    const cleaned = match[0].replace(/[.,;:!?]+$/, '');
    if (cleaned.length > 7) found.add(cleaned);
  }
  return [...found];
}

/**
 * Simulates GFM / Slack / Discord bare-URL & Markdown-link autolinkers.
 * Primary `?flow=` URLs must survive all 5 embedding contexts verbatim.
 */
export function assertMarkdownAutolinkerSafe(url: string): void {
  expect(url.startsWith('https://denovo.run/') || url.startsWith('http://') || url.startsWith('https://')).toBe(
    true,
  );
  const parsed = new URL(url);
  const queryPart = parsed.search;

  // Primary generated share URLs must NOT contain Markdown-unsafe structural chars or whitespace
  expect(queryPart).not.toMatch(/[()[\]<>\s]/);
  expect(url).not.toMatch(/[.,;!?]$/);

  // Simulate Slack/GFM bare URL extraction: stops at whitespace or ()[]<> or trailing . , ; ! ?
  const extractBareUrl = (text: string): string | null => {
    const m = text.match(/https?:\/\/[^\s()[\]<>"]+/);
    if (!m) return null;
    return m[0].replace(/[.,;:!?]+$/, '');
  };

  // Context 1: standalone line
  expect(extractBareUrl(url)).toBe(url);
  // Context 2: followed by period in prose
  expect(extractBareUrl(`Open ${url}.`)).toBe(url);
  // Context 3: followed by comma in prose
  expect(extractBareUrl(`Compare via ${url}, then inspect scRMSD.`)).toBe(url);
  // Context 4: wrapped in parentheses
  expect(extractBareUrl(`(workflow: ${url})`)).toBe(url);
  // Context 5: inside Markdown `[label](url)` link
  const mdLink = `[Run on denovo.run](${url})`;
  const mdMatch = mdLink.match(/\[[^\]]+\]\(([^)\s]+)\)/);
  expect(mdMatch?.[1]).toBe(url);
}

/**
 * Verifies PNG 8-byte signature and IHDR width x height (default 1200x630).
 */
export function assertValidOgPngBytes(bytes: Uint8Array, width = 1200, height = 630): void {
  expect(bytes.length).toBeGreaterThan(2000);
  expect([...bytes.subarray(0, 8)]).toEqual([...PNG_SIGNATURE]);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // IHDR chunk starts at byte 8 (len 4 bytes, 'IHDR' 4 bytes, width at offset 16, height at offset 20)
  expect(String.fromCharCode(bytes[12], bytes[13], bytes[14], bytes[15])).toBe('IHDR');
  expect(dv.getUint32(16, false)).toBe(width);
  expect(dv.getUint32(20, false)).toBe(height);
}

/**
 * Creates a Cloudflare Pages ASSETS mock simulating the real `/index.html` -> `/` 308 redirect
 * and serving `web/build/index.html` (if built) or `web/src/app.html`.
 */
export function createMockCfEnv(opts?: {
  html?: string;
  llmsTxt?: string;
  assetsStatus?: number;
  assetsContentType?: string;
  recordRequests?: string[];
}): CfPagesEnv {
  const defaultHtmlPath = existsSync(resolve(BUILD_DIR, 'index.html'))
    ? resolve(BUILD_DIR, 'index.html')
    : resolve(WEB_DIR, 'src/app.html');
  const html = opts?.html ?? readFileSync(defaultHtmlPath, 'utf8');

  return {
    ASSETS: {
      fetch: async (input) => {
        const reqUrl = new URL(
          typeof input === 'string' ? input : input instanceof URL ? input.href : input.url,
        );
        opts?.recordRequests?.push(reqUrl.pathname + reqUrl.search);

        // Real Cloudflare Pages ASSETS issues a 308 redirect on `/index.html` -> `/`
        if (reqUrl.pathname === '/index.html') {
          return new Response(null, { status: 308, headers: { Location: '/' } });
        }
        if (reqUrl.pathname === '/llms.txt' && opts?.llmsTxt !== undefined) {
          return new Response(opts.llmsTxt, {
            status: 200,
            headers: { 'Content-Type': 'text/plain; charset=utf-8' },
          });
        }
        if (opts?.assetsStatus && opts.assetsStatus !== 200) {
          return new Response('Mock error', {
            status: opts.assetsStatus,
            headers: { 'Content-Type': opts.assetsContentType ?? 'text/plain; charset=utf-8' },
          });
        }
        return new Response(html, {
          status: 200,
          headers: { 'Content-Type': opts?.assetsContentType ?? 'text/html; charset=utf-8' },
        });
      },
    },
  };
}

/**
 * Helper to find the quantitative evaluation markdown document on the clean branch.
 */
export function findEvaluationDocPath(): string | null {
  const candidates = [
    resolve(WEB_DIR, 'src/lib/engine/llm-url-evaluation.md'),
    resolve(REPO_ROOT, 'docs/llm-url-evaluation.md'),
    resolve(REPO_ROOT, 'docs/LLM_URL_EVALUATION.md'),
  ];
  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  return null;
}

/**
 * Helper to construct a minimal valid `WorkflowGraph` for a single adapter (plus required upstream if needed)
 * so all 29 adapters can be exercised in round-trip tests.
 */
export function buildMinimalRunnableGraphForAdapter(adapterId: string): WorkflowGraph {
  const manifest = registryNs.ADAPTER_REGISTRY[adapterId];
  expect(manifest, `Adapter "${adapterId}" must exist in ADAPTER_REGISTRY`).toBeDefined();

  const nodes: WorkflowNode[] = [];
  const edges: WorkflowEdge[] = [];

  const needStructure = Object.values(manifest.inputs).some(
    (spec) => spec.required !== false && spec.type === 'StructureArtifact',
  );
  const needSequences = Object.values(manifest.inputs).some(
    (spec) => spec.required !== false && spec.type === 'SequenceSet',
  );
  const needFolds = Object.values(manifest.inputs).some(
    (spec) => spec.required !== false && spec.type === 'FoldedStructure',
  );

  // Special case: esmc / dyna1 have all-optional ports in manifest, but crossValidate requires >=1 connected input
  const needsOptionalInputForCrossValidate = adapterId === 'esmc' || adapterId === 'dyna1';

  if (needStructure || needFolds || needsOptionalInputForCrossValidate) {
    nodes.push({
      id: 'src',
      adapterId: 'pdb-input',
      label: 'PDB Input',
      params: { ...graphUtilsNs.defaultParamsForAdapter('pdb-input'), mode: 'demo-bundle' },
    });
  }

  if (needSequences && !needFolds) {
    nodes.push({
      id: 'seq',
      adapterId: 'sequence-input',
      label: 'Sequence Input',
      params: { ...graphUtilsNs.defaultParamsForAdapter('sequence-input') },
    });
  }

  if (needFolds) {
    nodes.push({
      id: 'mpnn',
      adapterId: 'web-protein-mpnn',
      label: 'ProteinMPNN',
      params: { ...graphUtilsNs.defaultParamsForAdapter('web-protein-mpnn') },
    });
    nodes.push({
      id: 'fold',
      adapterId: 'localfold-af2',
      label: 'LocalFold',
      params: { ...graphUtilsNs.defaultParamsForAdapter('localfold-af2') },
    });
    edges.push(
      { fromNode: 'src', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
      { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
    );
  }

  const targetId = 'n1';
  nodes.push({
    id: targetId,
    adapterId,
    label: manifest.shortName,
    params: { ...graphUtilsNs.defaultParamsForAdapter(adapterId) },
  });

  if (needFolds) {
    edges.push(
      { fromNode: 'src', fromPort: 'structure', toNode: targetId, toPort: 'structure' },
      { fromNode: 'fold', fromPort: 'folds', toNode: targetId, toPort: 'folds' },
    );
  } else {
    for (const [portName, spec] of Object.entries(manifest.inputs)) {
      if (spec.required !== false && spec.type === 'StructureArtifact') {
        edges.push({ fromNode: 'src', fromPort: 'structure', toNode: targetId, toPort: portName });
      } else if (spec.required !== false && spec.type === 'SequenceSet') {
        edges.push({ fromNode: 'seq', fromPort: 'sequences', toNode: targetId, toPort: portName });
      }
    }
    if (needsOptionalInputForCrossValidate) {
      edges.push({ fromNode: 'src', fromPort: 'structure', toNode: targetId, toPort: 'structure' });
    }
  }

  return {
    id: `test-${adapterId}`,
    name: `Test ${adapterId}`,
    nodes,
    edges,
  };
}
