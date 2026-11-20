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
  'legacy-teacher': 'x',
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

async function deflateRaw(text: string): Promise<Uint8Array> {
  const input = new TextEncoder().encode(text);
  if (typeof CompressionStream === 'undefined') return input;
  const cs = new CompressionStream('deflate-raw');
  const writer = cs.writable.getWriter();
  await writer.write(input);
  await writer.close();
  const buf = await new Response(cs.readable).arrayBuffer();
  return new Uint8Array(buf);
}

async function inflateRaw(bytes: Uint8Array): Promise<string> {
  if (typeof DecompressionStream === 'undefined') {
    return new TextDecoder().decode(bytes);
  }
  const ds = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  await writer.write(bytes.slice());
  await writer.close();
  const buf = await new Response(ds.readable).arrayBuffer();
  return new TextDecoder().decode(buf);
}

function diffParams(
  adapterId: string,
  params: Record<string, unknown>,
  base?: Record<string, unknown>,
): { diff: Record<string, unknown>; strippedPdb: boolean } {
  const defaults = base ?? defaultParamsForAdapter(adapterId);
  const diff: Record<string, unknown> = {};
  let strippedPdb = false;
  for (const [k, v] of Object.entries(params)) {
    if (k === 'pdbText' && typeof v === 'string' && v.length > 1500) {
      strippedPdb = true;
      continue;
    }
    if (JSON.stringify(v) !== JSON.stringify(defaults[k])) {
      diff[k] = v;
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
      const { diff, strippedPdb: s } = diffParams(n.adapterId, n.params, baseN.params);
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
    const { diff, strippedPdb: s } = diffParams(n.adapterId, n.params);
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
  try {
    if (raw.startsWith('{') || raw.startsWith('[') || raw === 'null') return JSON.parse(raw);
  } catch {
    /* keep as string */
  }
  return raw;
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
      const node = byId.get(nodeId);
      if (!node) continue;
      node.params[paramKey] = parseParamValue(v, node.params[paramKey]);
    }
    return g.nodes.some((n) => !n.position) ? autoLayoutGraph(g) : g;
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
      const overrides = (tuple[2] && typeof tuple[2] === 'object' ? tuple[2] : {}) as Record<string, unknown>;
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
    return autoLayoutGraph({
      id: 'shared-workflow',
      name: 'Shared Workflow',
      nodes,
      edges,
    });
  }

  return null;
}

/** Format a WorkflowGraph as a concise, human- & LLM-readable YAML recipe. */
export function graphToYamlRecipe(graph: WorkflowGraph): string {
  const lines: string[] = [
    `denovo: v1`,
    `name: ${JSON.stringify(graph.name || graph.id)}`,
    `nodes:`,
  ];
  for (const n of graph.nodes) {
    const { diff } = diffParams(n.adapterId, n.params);
    const kv: Record<string, unknown> = { model: n.adapterId };
    if (n.label && n.label !== ADAPTER_REGISTRY[n.adapterId]?.shortName) kv.label = n.label;
    Object.assign(kv, diff);
    lines.push(`  ${n.id}: ${JSON.stringify(kv)}`);
  }
  lines.push(`wires:`);
  for (const e of graph.edges) {
    lines.push(`  - ${e.fromNode}.${e.fromPort} -> ${e.toNode}.${e.toPort}`);
  }
  return lines.join('\n') + '\n';
}

/** Parse either the concise YAML recipe format or full WorkflowGraph JSON back into a graph. */
export function parseRecipeToGraph(text: string): WorkflowGraph {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) {
    const parsed = JSON.parse(trimmed) as WorkflowGraph;
    if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
      throw new Error('JSON must contain nodes[] and edges[]');
    }
    return parsed.nodes.some((n) => !n.position) ? autoLayoutGraph(parsed) : parsed;
  }

  let name = 'Custom Recipe';
  let section: 'none' | 'nodes' | 'wires' = 'none';
  const nodes: WorkflowNode[] = [];
  const edges: WorkflowEdge[] = [];

  for (const rawLine of trimmed.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
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
    if (line === 'nodes:') {
      section = 'nodes';
      continue;
    }
    if (line === 'wires:') {
      section = 'wires';
      continue;
    }

    if (section === 'nodes') {
      const colon = line.indexOf(':');
      if (colon <= 0) throw new Error(`Invalid node line: "${line}"`);
      const id = line.slice(0, colon).trim();
      const bodyStr = line.slice(colon + 1).trim();
      const obj = JSON.parse(bodyStr) as Record<string, unknown>;
      const adapterId = String(obj.model ?? obj.adapterId ?? '');
      if (!ADAPTER_REGISTRY[adapterId]) {
        throw new Error(`Unknown model "${adapterId}" on node "${id}"`);
      }
      const label = typeof obj.label === 'string' ? obj.label : ADAPTER_REGISTRY[adapterId].shortName;
      const rest = { ...obj };
      delete rest.model;
      delete rest.adapterId;
      delete rest.label;
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

  return autoLayoutGraph({
    id: 'custom-recipe',
    name,
    nodes,
    edges,
  });
}
