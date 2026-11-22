// Compact shareable workflow URLs (#p= preset diff OR #g= deflate-raw base64url)
// + clean human/LLM-readable YAML/JSON workflow recipe serializer.

import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from './registry';
import { autoLayoutGraph, defaultParamsForAdapter, validateGraph } from './graph-utils';
import { computeUploadDigest, isValidSha256Hex } from './param-validation';
import type { WorkflowEdge, WorkflowGraph, WorkflowNode } from './types';

const ADAPTER_TO_SHORT: Record<string, string> = {
  'pdb-input': 'p',
  'sequence-input': 'q',
  'rfd4-proteina': 'r',
  genie3: 'g',
  protpardelle: 'pp',
  rfdiffusion: 'd',
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
  'top-k-filter': 'tk',
  'designability-scorer': 's',
  'structure-compare': 'c',
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

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return Boolean(v) && typeof v === 'object' && !Array.isArray(v);
}

function sanitizeValue(val: unknown, depth = 0): unknown {
  if (depth > 8 || val === null || typeof val !== 'object') return val;
  if (Array.isArray(val)) return val.map((item) => sanitizeValue(item, depth + 1));
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(val as Record<string, unknown>)) {
    if (FORBIDDEN_KEYS.has(k)) continue;
    out[k] = sanitizeValue(v, depth + 1);
  }
  return out;
}

function sanitizeParams(obj: Record<string, unknown>): Record<string, unknown> {
  return sanitizeValue(obj, 0) as Record<string, unknown>;
}

async function deflateRaw(text: string): Promise<Uint8Array> {
  const input = new TextEncoder().encode(text);
  if (typeof CompressionStream === 'undefined') return input;
  const cs = new CompressionStream('deflate-raw');
  const bufPromise = new Response(cs.readable).arrayBuffer();
  bufPromise.catch(() => {});
  const writer = cs.writable.getWriter();
  const writePromise = (async () => {
    await writer.write(input);
    await writer.close();
  })();
  writePromise.catch(() => {});
  const [buf] = await Promise.all([bufPromise, writePromise]);
  return new Uint8Array(buf);
}

const MAX_HASH_CHARS = 65_536;
const MAX_INFLATED_BYTES = 512 * 1024;
const MAX_IMPORT_NODES = 64;
const MAX_IMPORT_EDGES = 128;

async function inflateRaw(bytes: Uint8Array): Promise<string> {
  if (bytes.byteLength > MAX_INFLATED_BYTES) {
    throw new Error('Shared workflow payload exceeds maximum allowed size.');
  }
  if (typeof DecompressionStream === 'undefined') {
    return new TextDecoder().decode(bytes);
  }
  const ds = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  const writePromise = (async () => {
    await writer.write(bytes.slice());
    await writer.close();
  })();
  writePromise.catch(() => {});

  const reader = ds.readable.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) {
        totalBytes += value.byteLength;
        if (totalBytes > MAX_INFLATED_BYTES) {
          await reader.cancel('Shared workflow exceeds maximum decompressed size').catch(() => {});
          throw new Error('Shared workflow exceeds maximum decompressed size.');
        }
        chunks.push(value);
      }
    }
  } finally {
    reader.releaseLock();
  }
  await writePromise;

  const out = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(out);
}

const UPLOAD_META_KEYS = new Set(['missingUpload', 'expectedPdbSha256', 'expectedFilename']);
const PARAM_TYPE_HINTS: Record<string, unknown> = {
  expectedPdbSha256: '',
  expectedFilename: '',
  missingUpload: false,
};

function formatUnresolvedIssueEntry(raw: unknown): string {
  if (typeof raw === 'string') return raw.trim();
  if (raw === undefined || raw === null) return '';
  if (typeof raw === 'object') {
    try {
      return JSON.stringify(raw);
    } catch {
      return String(raw).trim();
    }
  }
  return String(raw).trim();
}

function normalizeUnresolvedIssues(raw: unknown): string[] {
  if (raw === undefined || raw === null) return [];
  if (!Array.isArray(raw)) {
    const single = formatUnresolvedIssueEntry(raw);
    return single ? [single] : [];
  }
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of raw) {
    const text = formatUnresolvedIssueEntry(item);
    if (text && !seen.has(text)) {
      seen.add(text);
      out.push(text);
    }
  }
  return out;
}

function diffParams(
  adapterId: string,
  params: Record<string, unknown>,
  base?: Record<string, unknown>,
  stripLargePdb = false,
): { diff: Record<string, unknown>; strippedPdb: boolean } {
  const defaults = base ?? defaultParamsForAdapter(adapterId);
  const diff: Record<string, unknown> = {};
  let strippedPdb = false;
  let strippedPdbText: string | null = null;
  const effectivePdbMode =
    adapterId === 'pdb-input' ? String(params.mode ?? defaults.mode ?? 'demo-bundle') : '';

  for (const [k, v] of Object.entries(params)) {
    if (FORBIDDEN_KEYS.has(k)) continue;
    if (adapterId === 'pdb-input') {
      if (k === 'fetchedPdbId') continue;
      if (effectivePdbMode === 'pdb-id' || effectivePdbMode === 'demo-bundle') {
        if (k === 'pdbText' || UPLOAD_META_KEYS.has(k)) continue;
      } else if (effectivePdbMode === 'custom') {
        if (UPLOAD_META_KEYS.has(k)) continue;
        if (k === 'pdbText' && typeof v === 'string' && v.length > 1500 && stripLargePdb) {
          strippedPdb = true;
          strippedPdbText = v;
          continue;
        }
      }
    } else if (k === 'pdbText' && typeof v === 'string' && v.length > 1500 && stripLargePdb) {
      strippedPdb = true;
      continue;
    }

    if (k === 'unresolvedImportIssues') {
      const normalized = normalizeUnresolvedIssues(v);
      const defaultIssues = normalizeUnresolvedIssues(defaults[k]);
      if (JSON.stringify(normalized) !== JSON.stringify(defaultIssues)) {
        diff[k] = normalized;
      }
      continue;
    }

    if (JSON.stringify(v) !== JSON.stringify(defaults[k])) {
      diff[k] = v;
    }
  }

  if (adapterId === 'pdb-input' && effectivePdbMode === 'custom') {
    const existingShaRaw = String(params.expectedPdbSha256 ?? '').trim().toLowerCase();
    const existingExpName = String(params.expectedFilename ?? '').trim();
    const curFilename = String(params.filename ?? '').trim();

    if (strippedPdb && stripLargePdb && strippedPdbText !== null) {
      // Preserve mode:'custom' and record a deterministic SHA-256 digest so recipients
      // cannot silently execute on demo-bundle coordinates.
      if (defaults.mode !== 'custom') {
        diff.mode = 'custom';
      }
      if (typeof defaults.pdbText === 'string' && defaults.pdbText.length > 0) {
        diff.pdbText = '';
      }
      const computedSha = computeUploadDigest(strippedPdbText);
      // Preserve any existing non-empty expectation (even if malformed or mismatched)
      // so re-sharing cannot silently repair or overwrite an unresolved digest check.
      const expectedSha = existingShaRaw ? existingShaRaw : computedSha;
      diff.missingUpload = true;
      diff.expectedPdbSha256 = expectedSha;
      const expName = existingExpName || curFilename;
      if (expName) {
        diff.expectedFilename = expName;
      }
    } else {
      const rawPdb = typeof params.pdbText === 'string' ? params.pdbText : '';
      const hasMissingFlag = Boolean(params.missingUpload);
      if (!rawPdb.trim() && (hasMissingFlag || existingShaRaw)) {
        strippedPdb = true;
        if (defaults.mode !== 'custom') {
          diff.mode = 'custom';
        }
        if (hasMissingFlag) diff.missingUpload = true;
        if (existingShaRaw) diff.expectedPdbSha256 = existingShaRaw;
        const expName = existingExpName || curFilename;
        if (expName) diff.expectedFilename = expName;
      } else if (rawPdb.trim() && existingShaRaw) {
        // Keep mismatched digest metadata if a small inline upload does not match the recorded digest.
        const actualSha = computeUploadDigest(rawPdb);
        if (!isValidSha256Hex(existingShaRaw) || actualSha !== existingShaRaw) {
          if (hasMissingFlag) diff.missingUpload = true;
          diff.expectedPdbSha256 = existingShaRaw;
          if (existingExpName) diff.expectedFilename = existingExpName;
        }
      }
    }
  }

  return { diff, strippedPdb };
}

function sameTopology(a: WorkflowGraph, b: WorkflowGraph): boolean {
  if (a.nodes.length !== b.nodes.length || a.edges.length !== b.edges.length) return false;
  for (let i = 0; i < a.nodes.length; i++) {
    if (a.nodes[i].id !== b.nodes[i].id || a.nodes[i].adapterId !== b.nodes[i].adapterId) return false;
  }
  const edgeSet = new Set(a.edges.map((e) => `${e.fromNode}\0${e.fromPort}->${e.toNode}\0${e.toPort}`));
  for (const e of b.edges) {
    if (!edgeSet.has(`${e.fromNode}\0${e.fromPort}->${e.toNode}\0${e.toPort}`)) return false;
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

  // 1. Check for exact preset topology match -> pick the preset with the shortest diff hash!
  let bestPresetMatch: EncodedHashResult | null = null;
  for (const preset of WORKFLOW_PRESETS) {
    if (!sameTopology(graph, preset.graph)) continue;
    const qs = new URLSearchParams();
    qs.set('p', preset.id);
    let presetStripped = false;
    const presetBaseline = preset.graph.nodes.some((n) => !n.position)
      ? autoLayoutGraph(structuredClone(preset.graph))
      : preset.graph;
    for (let i = 0; i < graph.nodes.length; i++) {
      const n = graph.nodes[i];
      const baseN = preset.graph.nodes[i];
      const laidN = presetBaseline.nodes[i];
      const { diff, strippedPdb: s } = diffParams(n.adapterId, n.params, baseN.params, true);
      if (s) presetStripped = true;
      for (const [k, v] of Object.entries(diff)) {
        qs.set(`${n.id}.${k}`, typeof v === 'string' ? v : JSON.stringify(v));
      }
      const baseLabel = baseN.label || ADAPTER_REGISTRY[baseN.adapterId]?.shortName || '';
      const curLabel = n.label?.trim() ? n.label : baseLabel;
      if (curLabel && curLabel !== baseLabel) {
        qs.set(`${n.id}._label`, curLabel);
      }
      const basePos = baseN.position ?? laidN?.position;
      if (
        n.position &&
        Number.isFinite(n.position.x) &&
        Number.isFinite(n.position.y) &&
        basePos
      ) {
        const rx = Math.round(n.position.x);
        const ry = Math.round(n.position.y);
        const lx = Math.round(basePos.x);
        const ly = Math.round(basePos.y);
        if (Math.abs(rx - lx) > 2 || Math.abs(ry - ly) > 2) {
          qs.set(`${n.id}._pos`, `${rx},${ry}`);
        }
      }
    }
    const hash = `#${qs.toString()}`;
    const candidate: EncodedHashResult = {
      hash,
      length: hash.length,
      mode: 'preset-diff',
      strippedPdb: presetStripped,
    };
    if (!bestPresetMatch || candidate.length < bestPresetMatch.length) {
      bestPresetMatch = candidate;
    }
  }
  if (bestPresetMatch) return bestPresetMatch;

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
  const hasAllPositions =
    graph.nodes.length > 0 &&
    graph.nodes.every(
      (n) =>
        n.position !== undefined &&
        Number.isFinite(n.position.x) &&
        Number.isFinite(n.position.y),
    );
  const compactPositions = hasAllPositions
    ? graph.nodes.map((n) => [Math.round(n.position!.x), Math.round(n.position!.y)])
    : undefined;
  const payload = JSON.stringify(
    compactPositions
      ? { n: compactNodes, e: compactEdges, p: compactPositions }
      : { n: compactNodes, e: compactEdges },
  );
  // Avoid trailing `_` or `-` in base64url payloads: GFM/chat/social autolinkers
  // strip trailing `_` from bare URLs. Trailing whitespace is valid JSON and 100%
  // backwards-compatible with all decoders.
  let b64 = toBase64Url(await deflateRaw(payload));
  for (let pad = 1; pad <= 8 && /[_-]$/.test(b64); pad++) {
    b64 = toBase64Url(await deflateRaw(payload + ' '.repeat(pad)));
  }
  const hash = `#g=${b64}`;
  return { hash, length: hash.length, mode: 'compressed-dag', strippedPdb };
}

function getSchemaTypeTemplate(adapterId: string, paramKey: string): unknown {
  const schema = ADAPTER_REGISTRY[adapterId]?.paramSchema;
  if (!schema) return undefined;
  const field = schema.find((f) => f.key === paramKey);
  if (!field) return undefined;
  if (field.kind === 'integer' || field.kind === 'number') {
    return typeof field.defaultValue === 'number' ? field.defaultValue : 0;
  }
  if (field.kind === 'boolean') {
    return typeof field.defaultValue === 'boolean' ? field.defaultValue : false;
  }
  return typeof field.defaultValue === 'string' ? field.defaultValue : '';
}

function parseParamValue(raw: string, templateVal: unknown): unknown {
  if (typeof templateVal === 'number') {
    const n = Number(raw);
    return Number.isFinite(n) ? n : templateVal;
  }
  if (typeof templateVal === 'boolean') return raw === 'true';
  if (typeof templateVal === 'string') return raw;
  if (raw === 'true') return true;
  if (raw === 'false') return false;
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
  if (!isPlainObject(raw)) {
    throw new Error('Workflow must be a JSON object with nodes[] and edges[]');
  }
  const parsed = raw as Partial<WorkflowGraph>;
  if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
    throw new Error('Workflow must contain nodes[] and edges[]');
  }
  if (parsed.nodes.length > MAX_IMPORT_NODES) {
    throw new Error(`Workflow exceeds maximum allowed block count (${MAX_IMPORT_NODES}).`);
  }
  if (parsed.edges.length > MAX_IMPORT_EDGES) {
    throw new Error(`Workflow exceeds maximum allowed wire count (${MAX_IMPORT_EDGES}).`);
  }

  const seenIds = new Set<string>();
  const validatedNodes: WorkflowNode[] = parsed.nodes.map((n, idx) => {
    if (!isPlainObject(n)) {
      throw new Error(`Invalid node at index ${idx}`);
    }
    const id = String(n.id ?? '').trim();
    if (!id) throw new Error(`Node at index ${idx} is missing an id`);
    if (seenIds.has(id)) throw new Error(`Duplicate node id "${id}"`);
    seenIds.add(id);

    const adapterId = String(n.adapterId ?? '');
    if (!Object.hasOwn(ADAPTER_REGISTRY, adapterId)) {
      throw new Error(`Unknown model "${adapterId}" on node "${id}"`);
    }
    const rawParams = isPlainObject(n.params) ? sanitizeParams(n.params) : {};
    const mergedParams = fillDefaults ? { ...defaultParamsForAdapter(adapterId), ...rawParams } : { ...rawParams };
    // Security: discard externally supplied fetch provenance / cached coordinates for accession-based
    // PDB inputs so an imported recipe cannot spoof an RCSB accession with arbitrary coordinates.
    if (adapterId === 'pdb-input') {
      delete mergedParams.fetchedPdbId;
      const mode = String(mergedParams.mode ?? 'demo-bundle');
      if (mode === 'pdb-id' || mode === 'demo-bundle') {
        if ('pdbText' in mergedParams) {
          mergedParams.pdbText = '';
        }
      }
    }
    const posObj = isPlainObject(n.position) ? n.position : null;
    const pos =
      posObj &&
      typeof posObj.x === 'number' &&
      Number.isFinite(posObj.x) &&
      typeof posObj.y === 'number' &&
      Number.isFinite(posObj.y)
        ? { x: posObj.x, y: posObj.y }
        : undefined;

    return {
      id,
      adapterId,
      label: typeof n.label === 'string' && n.label.trim() ? n.label : ADAPTER_REGISTRY[adapterId].shortName,
      params: mergedParams,
      ...(pos ? { position: pos } : {}),
    };
  });

  const validatedEdges: WorkflowEdge[] = parsed.edges.map((e, idx) => {
    if (!isPlainObject(e)) {
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

  // Recipes may be unfinished, but their existing wires must be valid before layout/rendering.
  const errors = validateGraph(normalized, { requireRunnable: false });
  if (errors.length) throw new Error(errors.join('\n'));
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
  if (clean.length > MAX_HASH_CHARS) {
    throw new Error(`Shared workflow link is too large (max ${MAX_HASH_CHARS} chars).`);
  }

  if (clean.startsWith('p=')) {
    const qs = new URLSearchParams(clean);
    const presetId = qs.get('p');
    const preset = WORKFLOW_PRESETS.find((p) => p.id === presetId);
    if (!preset) return null;
    const g = structuredClone(preset.graph);
    const byId = new Map(g.nodes.map((n) => [n.id, n]));
    let hasCustomPositions = false;
    for (const [k, v] of qs.entries()) {
      if (k === 'p') continue;
      const dot = k.lastIndexOf('.');
      if (dot <= 0) continue;
      const nodeId = k.slice(0, dot);
      const paramKey = k.slice(dot + 1);
      if (FORBIDDEN_KEYS.has(paramKey)) continue;
      const node = byId.get(nodeId);
      if (!node) continue;
      if (paramKey === '_label') {
        if (v.trim()) node.label = v;
        continue;
      }
      if (paramKey === '_pos') {
        const parts = v.split(',');
        if (parts.length === 2) {
          const x = Number(parts[0]);
          const y = Number(parts[1]);
          if (Number.isFinite(x) && Number.isFinite(y)) {
            node.position = { x, y };
            hasCustomPositions = true;
          }
        }
        continue;
      }
      const defaults = defaultParamsForAdapter(node.adapterId);
      const schemaVal = getSchemaTypeTemplate(node.adapterId, paramKey);
      const templateVal =
        node.params[paramKey] !== undefined
          ? node.params[paramKey]
          : defaults[paramKey] !== undefined
            ? defaults[paramKey]
            : schemaVal !== undefined
              ? schemaVal
              : PARAM_TYPE_HINTS[paramKey];
      node.params[paramKey] = parseParamValue(v, templateVal);
    }
    if (hasCustomPositions) {
      const laid = preset.graph.nodes.some((n) => !n.position)
        ? autoLayoutGraph(structuredClone(preset.graph))
        : preset.graph;
      for (let i = 0; i < g.nodes.length; i++) {
        if (!g.nodes[i].position && laid.nodes[i]?.position) {
          g.nodes[i].position = { ...laid.nodes[i].position! };
        }
      }
    }
    return normalizeWorkflowGraph(g, preset.id, preset.name, false);
  }

  if (clean.startsWith('g=')) {
    const normalizedEncoded = clean
      .slice(2)
      .replace(/%5[fF]/g, '_')
      .replace(/%2[dD]/g, '-')
      .replace(/\\_/g, '_')
      .replace(/[.,;:!?)\]}>'"]+$/, '');

    const tryDecodeCompressedObj = async (
      candidate: string,
    ): Promise<{ n: unknown[]; e: unknown[]; p?: unknown } | null> => {
      let rawBytes: Uint8Array;
      try {
        rawBytes = fromBase64Url(candidate);
      } catch {
        return null;
      }
      let jsonText: string;
      try {
        jsonText = await inflateRaw(rawBytes);
      } catch (inflateErr) {
        if (inflateErr instanceof Error && /maximum (decompressed|allowed) size/i.test(inflateErr.message)) {
          throw inflateErr;
        }
        if (rawBytes.byteLength > MAX_INFLATED_BYTES) {
          throw new Error('Shared workflow payload exceeds maximum allowed size.');
        }
        jsonText = new TextDecoder().decode(rawBytes);
      }
      let parsedObj: unknown;
      try {
        parsedObj = JSON.parse(jsonText);
      } catch {
        return null;
      }
      if (!isPlainObject(parsedObj) || !Array.isArray(parsedObj.n) || !Array.isArray(parsedObj.e)) {
        return null;
      }
      return { n: parsedObj.n, e: parsedObj.e, p: parsedObj.p };
    };

    let obj = await tryDecodeCompressedObj(normalizedEncoded);
    if (!obj) {
      // Recover links whose trailing `_` or `-` was stripped by Markdown/chat/social autolinkers.
      for (const suffix of ['_', '-', '__']) {
        obj = await tryDecodeCompressedObj(normalizedEncoded + suffix);
        if (obj) break;
      }
    }
    if (!obj) return null;
    if (obj.n.length > MAX_IMPORT_NODES || obj.e.length > MAX_IMPORT_EDGES) {
      throw new Error(
        `Shared workflow exceeds size limits (max ${MAX_IMPORT_NODES} blocks, ${MAX_IMPORT_EDGES} wires).`,
      );
    }

    const rawPositions = Array.isArray(obj.p) ? obj.p : null;
    const nodes: WorkflowNode[] = [];
    for (let i = 0; i < obj.n.length; i++) {
      const tuple = obj.n[i];
      if (!Array.isArray(tuple) || tuple.length < 2) return null;
      const id = String(tuple[0]);
      const shortA = String(tuple[1]);
      const adapterId = Object.hasOwn(SHORT_TO_ADAPTER, shortA) ? SHORT_TO_ADAPTER[shortA] : shortA;
      const rawOverrides = isPlainObject(tuple[2]) ? tuple[2] : {};
      const overrides = sanitizeParams(rawOverrides);
      const label = typeof tuple[3] === 'string' ? tuple[3] : ADAPTER_REGISTRY[adapterId]?.shortName ?? adapterId;
      const posTuple = rawPositions?.[i];
      const position =
        Array.isArray(posTuple) &&
        posTuple.length === 2 &&
        typeof posTuple[0] === 'number' &&
        Number.isFinite(posTuple[0]) &&
        typeof posTuple[1] === 'number' &&
        Number.isFinite(posTuple[1])
          ? { x: posTuple[0], y: posTuple[1] }
          : undefined;
      nodes.push({
        id,
        adapterId,
        label,
        params: { ...defaultParamsForAdapter(adapterId), ...overrides },
        ...(position ? { position } : {}),
      });
    }
    const edges: WorkflowEdge[] = [];
    for (const e of obj.e) {
      if (!Array.isArray(e) || e.length < 4) return null;
      edges.push({
        fromNode: String(e[0]),
        fromPort: String(e[1]),
        toNode: String(e[2]),
        toPort: String(e[3]),
      });
    }
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
    lines.push(
      `# Note: oversized inline PDB (>1500 chars) omitted; re-upload matching structure or use Full JSON to preserve it`,
    );
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
      if (line.startsWith('denovo:')) {
        const ver = line.slice(7).trim().replace(/^['"]|['"]$/g, '');
        if (ver !== 'v1') {
          throw new Error(`Unsupported recipe schema version "${ver || '(empty)'}" (supported: v1)`);
        }
        continue;
      }
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
      const parsedBody = JSON.parse(bodyStr) as unknown;
      if (!isPlainObject(parsedBody)) {
        throw new Error(`Invalid node body on "${id}": expected a JSON object`);
      }
      const obj = parsedBody;
      const adapterId = String(obj.adapterId ?? obj.model ?? '');
      if (!Object.hasOwn(ADAPTER_REGISTRY, adapterId)) {
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
        if (typeof seqLabel === 'string') {
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
      const arrowIdx = clean.indexOf('->');
      if (arrowIdx <= 0 || clean.indexOf('->', arrowIdx + 2) !== -1) {
        throw new Error(`Invalid wire syntax: "${line}" (expected "a.port -> b.port")`);
      }
      const lhs = clean.slice(0, arrowIdx).trim();
      const rhs = clean.slice(arrowIdx + 2).trim();
      const lhsDot = lhs.lastIndexOf('.');
      const rhsDot = rhs.lastIndexOf('.');
      const fromNode = lhsDot > 0 ? lhs.slice(0, lhsDot).trim() : '';
      const fromPort = lhsDot > 0 ? lhs.slice(lhsDot + 1).trim() : '';
      const toNode = rhsDot > 0 ? rhs.slice(0, rhsDot).trim() : '';
      const toPort = rhsDot > 0 ? rhs.slice(rhsDot + 1).trim() : '';
      if (
        !fromNode ||
        !toNode ||
        !/^[a-zA-Z0-9_-]+$/.test(fromPort) ||
        !/^[a-zA-Z0-9_-]+$/.test(toPort)
      ) {
        throw new Error(`Invalid wire syntax: "${line}" (expected "a.port -> b.port")`);
      }
      edges.push({ fromNode, fromPort, toNode, toPort });
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
