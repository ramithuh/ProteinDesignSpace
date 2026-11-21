// Compact shareable workflow URLs (#p= preset diff OR #g= deflate-raw base64url)
// + clean human/LLM-readable YAML/JSON workflow recipe serializer.

import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from './registry';
import { autoLayoutGraph, defaultParamsForAdapter } from './graph-utils';
import type { WorkflowEdge, WorkflowGraph, WorkflowNode } from './types';

const ADAPTER_TO_SHORT: Record<string, string> = {
  'pdb-input': 'p',
  'sequence-input': 'q',
  'rfd4-conditioner': 'c',
  'rfd4-proteina': 'r',
  genie3: 'g',
  'web-protein-mpnn': 'm',
  'localfold-af2': 'f',
  'designability-scorer': 's',
};

const SHORT_TO_ADAPTER: Record<string, string> = Object.fromEntries(
  Object.entries(ADAPTER_TO_SHORT).map(([k, v]) => [v, k]),
);

function toBase64Url(bytes: Uint8Array): string {
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(str: string): Uint8Array {
  const b64 = str.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((str.length + 3) % 4);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

function sanitizeParams(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (FORBIDDEN_KEYS.has(k)) continue;
    out[k] = v;
  }
  return out;
}

async function deflateRaw(text: string): Promise<Uint8Array> {
  const input = new TextEncoder().encode(text);
  if (typeof CompressionStream === 'undefined') return input;
  const cs = new CompressionStream('deflate-raw');
  const bufPromise = new Response(cs.readable).arrayBuffer();
  const writer = cs.writable.getWriter();
  await writer.write(input);
  await writer.close();
  const buf = await bufPromise;
  return new Uint8Array(buf);
}

async function inflateRaw(bytes: Uint8Array): Promise<string> {
  if (typeof DecompressionStream === 'undefined') {
    return new TextDecoder().decode(bytes);
  }
  const ds = new DecompressionStream('deflate-raw');
  const bufPromise = new Response(ds.readable).arrayBuffer();
  const writer = ds.writable.getWriter();
  await writer.write(bytes.slice());
  await writer.close();
  const buf = await bufPromise;
  return new TextDecoder().decode(buf);
}

function diffParams(
  adapterId: string,
  params: Record<string, unknown>,
  base?: Record<string, unknown>,
  resetModeOnStrip = false,
): { diff: Record<string, unknown>; strippedPdb: boolean } {
  const defaults = base ?? defaultParamsForAdapter(adapterId);
  const diff: Record<string, unknown> = {};
  let strippedPdb = false;
  for (const [k, v] of Object.entries(params)) {
    if (FORBIDDEN_KEYS.has(k)) continue;
    if (k === 'pdbText' && typeof v === 'string' && v.length > 1500) {
      strippedPdb = true;
      continue;
    }
    if (adapterId === 'pdb-input' && k === 'fetchedPdbId') {
      continue;
    }
    if (JSON.stringify(v) !== JSON.stringify(defaults[k])) {
      diff[k] = v;
    }
  }
  if (strippedPdb && resetModeOnStrip && adapterId === 'pdb-input' && params.mode !== 'pdb-id') {
    // Do not leave a broken mode:'custom' with an empty pdbText in shared links
    if (defaults.mode === 'demo-bundle') {
      delete diff.mode;
      delete diff.filename;
    } else {
      diff.mode = 'demo-bundle';
      delete diff.filename;
    }
  }
  return { diff, strippedPdb };
}

function sameTopology(a: WorkflowGraph, b: WorkflowGraph): boolean {
  if (a.nodes.length !== b.nodes.length || a.edges.length !== b.edges.length) return false;
  for (let i = 0; i < a.nodes.length; i++) {
    if (a.nodes[i].id !== b.nodes[i].id || a.nodes[i].adapterId !== b.nodes[i].adapterId) return false;
  }
  const edgeSet = new Set(a.edges.map((e) => `${e.fromNode}.${e.fromPort}->${e.toNode}.${e.toPort}`));
  for (const e of b.edges) {
    if (!edgeSet.has(`${e.fromNode}.${e.fromPort}->${e.toNode}.${e.toPort}`)) return false;
  }
  return true;
}

export interface EncodedHashResult {
  hash: string;
  length: number;
  mode: 'preset-diff' | 'compressed-dag';
  strippedPdb: boolean;
}

/**
 * Encode a WorkflowGraph into the shortest possible URL hash:
 * - If topology matches a built-in preset: `#p=<presetId>&<node>.<param>=<val>...` (~30-70 chars!)
 * - Otherwise: compact alias JSON -> deflate-raw -> base64url `#g=<payload>` (~70-150 chars!)
 */
export async function encodeGraphToHash(graph: WorkflowGraph): Promise<EncodedHashResult> {
  let strippedPdb = false;

  // 1. Check for exact preset topology match -> ultra-short human-readable hash!
  for (const preset of WORKFLOW_PRESETS) {
    if (!sameTopology(graph, preset.graph)) continue;
    const qs = new URLSearchParams();
    qs.set('p', preset.id);
    for (let i = 0; i < graph.nodes.length; i++) {
      const n = graph.nodes[i];
      const baseN = preset.graph.nodes[i];
      const { diff, strippedPdb: s } = diffParams(n.adapterId, n.params, baseN.params, true);
      if (s) strippedPdb = true;
      for (const [k, v] of Object.entries(diff)) {
        qs.set(`${n.id}.${k}`, typeof v === 'string' ? v : JSON.stringify(v));
      }
    }
    const hash = `#${qs.toString()}`;
    return { hash, length: hash.length, mode: 'preset-diff', strippedPdb };
  }

  // 2. General DAG -> short-alias tuple representation -> deflate-raw -> base64url
  const compactNodes = graph.nodes.map((n) => {
    const a = ADAPTER_TO_SHORT[n.adapterId] ?? n.adapterId;
    const { diff, strippedPdb: s } = diffParams(n.adapterId, n.params, undefined, true);
    if (s) strippedPdb = true;
    const tuple: unknown[] = [n.id, a];
    const hasDiff = Object.keys(diff).length > 0;
    const customLabel =
      n.label && n.label !== ADAPTER_REGISTRY[n.adapterId]?.shortName ? n.label : undefined;
    if (hasDiff || customLabel) tuple.push(diff);
    if (customLabel) tuple.push(customLabel);
    return tuple;
  });

  const compactEdges = graph.edges.map((e) => [e.fromNode, e.fromPort, e.toNode, e.toPort]);
  const payload = JSON.stringify({ n: compactNodes, e: compactEdges });
  const compressed = await deflateRaw(payload);
  const hash = `#g=${toBase64Url(compressed)}`;
  return { hash, length: hash.length, mode: 'compressed-dag', strippedPdb };
}

function parseParamValue(raw: string, templateVal: unknown): unknown {
  if (typeof templateVal === 'number') {
    const n = Number(raw);
    return Number.isFinite(n) ? n : templateVal;
  }
  if (typeof templateVal === 'boolean') return raw === 'true';
  if (typeof templateVal === 'string') return raw;
  try {
    if (raw.startsWith('{') || raw.startsWith('[') || raw === 'null') return JSON.parse(raw);
  } catch {
    /* keep as string */
  }
  return raw;
}

/**
 * Canonical graph validator & normalizer shared across URL hash decoding,
 * YAML/JSON recipe parsing, and JSON file import.
 */
export function normalizeWorkflowGraph(
  raw: unknown,
  fallbackId = 'custom-recipe',
  fallbackName = 'Custom Recipe',
  fillDefaults = true,
): WorkflowGraph {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Workflow must be a JSON object with nodes[] and edges[]');
  }
  const parsed = raw as Partial<WorkflowGraph>;
  if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
    throw new Error('Workflow must contain nodes[] and edges[]');
  }

  const seenIds = new Set<string>();
  const validatedNodes: WorkflowNode[] = parsed.nodes.map((n, idx) => {
    if (!n || typeof n !== 'object') {
      throw new Error(`Invalid node at index ${idx}`);
    }
    const id = String(n.id ?? '').trim();
    if (!id) throw new Error(`Node at index ${idx} is missing an id`);
    if (seenIds.has(id)) throw new Error(`Duplicate node id "${id}"`);
    seenIds.add(id);

    const adapterId = String(n.adapterId ?? '');
    if (!ADAPTER_REGISTRY[adapterId]) {
      throw new Error(`Unknown model "${adapterId}" on node "${id}"`);
    }
    const rawParams = n.params && typeof n.params === 'object' ? sanitizeParams(n.params) : {};
    const pos =
      n.position &&
      typeof n.position === 'object' &&
      typeof n.position.x === 'number' &&
      Number.isFinite(n.position.x) &&
      typeof n.position.y === 'number' &&
      Number.isFinite(n.position.y)
        ? { x: n.position.x, y: n.position.y }
        : undefined;

    return {
      id,
      adapterId,
      label: typeof n.label === 'string' && n.label.trim() ? n.label : ADAPTER_REGISTRY[adapterId].shortName,
      params: fillDefaults ? { ...defaultParamsForAdapter(adapterId), ...rawParams } : rawParams,
      ...(pos ? { position: pos } : {}),
    };
  });

  const validatedEdges: WorkflowEdge[] = parsed.edges.map((e, idx) => {
    if (!e || typeof e !== 'object') {
      throw new Error(`Invalid wire at index ${idx}`);
    }
    const fromNode = String(e.fromNode ?? '').trim();
    const fromPort = String(e.fromPort ?? '').trim();
    const toNode = String(e.toNode ?? '').trim();
    const toPort = String(e.toPort ?? '').trim();
    if (!fromNode || !fromPort || !toNode || !toPort) {
      throw new Error(`Wire at index ${idx} is missing node or port name`);
    }
    if (!seenIds.has(fromNode)) {
      throw new Error(`Wire references unknown source node "${fromNode}"`);
    }
    if (!seenIds.has(toNode)) {
      throw new Error(`Wire references unknown target node "${toNode}"`);
    }
    return { fromNode, fromPort, toNode, toPort };
  });

  const normalized: WorkflowGraph = {
    id: typeof parsed.id === 'string' && parsed.id.trim() ? parsed.id.trim() : fallbackId,
    name: typeof parsed.name === 'string' && parsed.name.trim() ? parsed.name.trim() : fallbackName,
    ...(typeof parsed.description === 'string' ? { description: parsed.description } : {}),
    nodes: validatedNodes,
    edges: validatedEdges,
  };

  return normalized.nodes.some((n) => !n.position) ? autoLayoutGraph(normalized) : normalized;
}

/** Serialize a WorkflowGraph cleanly as pretty-printed canonical JSON. */
export function serializeWorkflowJson(graph: WorkflowGraph): string {
  return JSON.stringify(graph, null, 2);
}

/** Decode `#p=...` or `#g=...` back into a full WorkflowGraph (with auto-layout). */
export async function decodeHashToGraph(hash: string): Promise<WorkflowGraph | null> {
  const clean = hash.replace(/^#/, '').trim();
  if (!clean) return null;

  if (clean.startsWith('p=')) {
    const qs = new URLSearchParams(clean);
    const presetId = qs.get('p');
    const preset = WORKFLOW_PRESETS.find((p) => p.id === presetId);
    if (!preset) return null;
    const g = structuredClone(preset.graph);
    const byId = new Map(g.nodes.map((n) => [n.id, n]));
    for (const [k, v] of qs.entries()) {
      if (k === 'p') continue;
      const dot = k.indexOf('.');
      if (dot <= 0) continue;
      const nodeId = k.slice(0, dot);
      const paramKey = k.slice(dot + 1);
      if (FORBIDDEN_KEYS.has(paramKey)) continue;
      const node = byId.get(nodeId);
      if (!node) continue;
      const defaults = defaultParamsForAdapter(node.adapterId);
      const templateVal = node.params[paramKey] !== undefined ? node.params[paramKey] : defaults[paramKey];
      node.params[paramKey] = parseParamValue(v, templateVal);
    }
    return normalizeWorkflowGraph(g, preset.id, preset.name, false);
  }

  if (clean.startsWith('g=')) {
    const encoded = clean.slice(2);
    const rawBytes = fromBase64Url(encoded);
    let jsonText: string;
    try {
      jsonText = await inflateRaw(rawBytes);
    } catch {
      jsonText = new TextDecoder().decode(rawBytes);
    }
    const obj = JSON.parse(jsonText) as { n: unknown[][]; e: string[][] };
    if (!Array.isArray(obj.n) || !Array.isArray(obj.e)) return null;

    const nodes: WorkflowNode[] = obj.n.map((tuple) => {
      const id = String(tuple[0]);
      const shortA = String(tuple[1]);
      const adapterId = SHORT_TO_ADAPTER[shortA] ?? shortA;
      const rawOverrides = (tuple[2] && typeof tuple[2] === 'object' ? tuple[2] : {}) as Record<string, unknown>;
      const overrides = sanitizeParams(rawOverrides);
      const label = typeof tuple[3] === 'string' ? tuple[3] : ADAPTER_REGISTRY[adapterId]?.shortName ?? adapterId;
      return {
        id,
        adapterId,
        label,
        params: { ...defaultParamsForAdapter(adapterId), ...overrides },
      };
    });
    const edges: WorkflowEdge[] = obj.e.map((e) => ({
      fromNode: e[0],
      fromPort: e[1],
      toNode: e[2],
      toPort: e[3],
    }));
    try {
      return normalizeWorkflowGraph(
        { id: 'shared-workflow', name: 'Shared Workflow', nodes, edges },
        'shared-workflow',
        'Shared Workflow',
      );
    } catch {
      return null;
    }
  }

  return null;
}

/** Format a WorkflowGraph as a concise, human- & LLM-readable YAML recipe. */
export function graphToYamlRecipe(graph: WorkflowGraph): string {
  const lines: string[] = [
    `denovo: v1`,
    `name: ${JSON.stringify(graph.name || graph.id)}`,
  ];
  let anyStrippedPdb = false;
  const nodeLines: string[] = [];
  for (const n of graph.nodes) {
    const { diff, strippedPdb } = diffParams(n.adapterId, n.params, undefined, true);
    if (strippedPdb) anyStrippedPdb = true;
    const kv: Record<string, unknown> = { model: n.adapterId };
    if (n.label && n.label !== ADAPTER_REGISTRY[n.adapterId]?.shortName) kv.label = n.label;
    if (n.adapterId === 'localfold-af2' && 'model' in diff) {
      diff.foldModel = diff.model;
      delete diff.model;
    }
    if (n.adapterId === 'sequence-input' && 'label' in diff) {
      diff.seqLabel = diff.label;
      delete diff.label;
    }
    Object.assign(kv, diff);
    nodeLines.push(`  ${n.id}: ${JSON.stringify(kv)}`);
  }
  if (anyStrippedPdb) {
    lines.push(`# Note: oversized inline PDB (>1500 chars) omitted; use Full JSON or Export JSON to preserve it`);
  }
  lines.push(`nodes:`);
  lines.push(...nodeLines);
  lines.push(`wires:`);
  for (const e of graph.edges) {
    lines.push(`  - ${e.fromNode}.${e.fromPort} -> ${e.toNode}.${e.toPort}`);
  }
  return lines.join('\n') + '\n';
}

/** Parse either the concise YAML recipe format or full WorkflowGraph JSON back into a graph. */
export function parseRecipeToGraph(text: string): WorkflowGraph {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error('Recipe is empty');
  }
  if (trimmed.startsWith('{')) {
    const parsed = JSON.parse(trimmed) as unknown;
    return normalizeWorkflowGraph(parsed, 'custom-recipe', 'Custom Recipe');
  }

  let name = 'Custom Recipe';
  let section: 'none' | 'nodes' | 'wires' = 'none';
  let sawNodesSection = false;
  const nodes: WorkflowNode[] = [];
  const edges: WorkflowEdge[] = [];

  for (const rawLine of trimmed.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    if (line === 'nodes:') {
      section = 'nodes';
      sawNodesSection = true;
      continue;
    }
    if (line === 'wires:') {
      section = 'wires';
      continue;
    }
    if (section === 'none') {
      if (line.startsWith('denovo:')) continue;
      if (line.startsWith('name:')) {
        const val = line.slice(5).trim();
        try {
          name = String(JSON.parse(val));
        } catch {
          name = val.replace(/^['"]|['"]$/g, '');
        }
        continue;
      }
      throw new Error(`Unexpected line before "nodes:": "${line}"`);
    }

    if (section === 'nodes') {
      const colon = line.indexOf(':');
      if (colon <= 0) throw new Error(`Invalid node line: "${line}"`);
      const id = line.slice(0, colon).trim();
      const bodyStr = line.slice(colon + 1).trim();
      const obj = JSON.parse(bodyStr) as Record<string, unknown>;
      const adapterId = String(obj.adapterId ?? obj.model ?? '');
      if (!ADAPTER_REGISTRY[adapterId]) {
        throw new Error(`Unknown model "${adapterId}" on node "${id}"`);
      }
      const label = typeof obj.label === 'string' ? obj.label : ADAPTER_REGISTRY[adapterId].shortName;
      const rest = sanitizeParams(obj);
      delete rest.model;
      delete rest.adapterId;
      delete rest.label;
      if (adapterId === 'localfold-af2') {
        const foldModel = obj.foldModel ?? obj.checkpoint;
        if (typeof foldModel === 'string' && foldModel) {
          rest.model = foldModel;
        } else if (obj.adapterId && typeof obj.model === 'string' && obj.model && obj.model !== adapterId) {
          rest.model = obj.model;
        }
        delete rest.foldModel;
        delete rest.checkpoint;
      }
      if (adapterId === 'sequence-input') {
        const seqLabel = obj.seqLabel ?? obj.sequenceLabel;
        if (typeof seqLabel === 'string' && seqLabel) {
          rest.label = seqLabel;
        }
        delete rest.seqLabel;
        delete rest.sequenceLabel;
      }
      nodes.push({
        id,
        adapterId,
        label,
        params: { ...defaultParamsForAdapter(adapterId), ...rest },
      });
    } else if (section === 'wires') {
      const clean = line.replace(/^-\s*/, '').trim();
      const m = /^([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_-]+)\s*->\s*([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_-]+)$/.exec(clean);
      if (!m) throw new Error(`Invalid wire syntax: "${line}" (expected "a.port -> b.port")`);
      edges.push({ fromNode: m[1], fromPort: m[2], toNode: m[3], toPort: m[4] });
    }
  }

  if (!sawNodesSection) {
    throw new Error('Invalid recipe: expected JSON or "denovo: v1" YAML with a "nodes:" section');
  }

  return normalizeWorkflowGraph(
    { id: 'custom-recipe', name, nodes, edges },
    'custom-recipe',
    name,
  );
}
