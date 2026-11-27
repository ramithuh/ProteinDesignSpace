// Zero-auth Save / Load Workflow Session via GitHub Secret Gists.
// Pure-TS serializer, re-hydrator, strict reference parser, and host-allowlisted loader.

import { decodeWireValue, encodeWireValue } from './remote-codec';
import { validateGraph } from './graph-utils';
import { normalizeWorkflowGraph } from './share';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  CanonicalArtifact,
  FoldedCandidate,
  FoldedStructure,
  StructureArtifact,
  WorkflowGraph,
} from './types';

export const GIST_SESSION_SCHEMA = 'denovo.gist-session/v1' as const;
export const GIST_SESSION_FILENAME = 'denovo-session.json' as const;
export const DEFAULT_MAX_GIST_BYTES = 15 * 1024 * 1024; // 15 MiB

export const ALLOWED_GIST_HOSTS = new Set([
  'api.github.com',
  'gist.githubusercontent.com',
]);

const GIST_ID_RE = /^[a-f0-9]{20,64}$/i;
const GIST_OWNER_RE = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$/;
const GIST_REV_RE = /^[a-f0-9]{7,40}$/i;
const SAFE_RAW_FILENAME_RE = /^[A-Za-z0-9._-]+$/;
const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

export interface GistReference {
  id: string;
  owner?: string;
  rev?: string;
}

export interface GistSessionPayload {
  schema: typeof GIST_SESSION_SCHEMA;
  createdAt: string;
  graph: WorkflowGraph;
  outputs: Record<string, Record<string, CanonicalArtifact>>;
}

export interface SerializedGistSession {
  json: string;
  bytes: number;
  nodeCount: number;
  resultCount: number;
  warnings: string[];
}

export interface LoadGistOptions {
  fetchFn?: typeof fetch;
  signal?: AbortSignal;
  maxBytes?: number;
}

export interface LoadedGistSession {
  session: GistSessionPayload;
  ref: GistReference;
  sourceUrl: string;
  bytes: number;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return Boolean(v) && typeof v === 'object' && !Array.isArray(v);
}

function assertAllowedGistUrl(rawUrl: string): URL {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    throw new Error(`Invalid Gist fetch URL.`);
  }
  if (url.protocol !== 'https:') {
    throw new Error(`Blocked non-HTTPS Gist URL protocol "${url.protocol}".`);
  }
  if (url.username || url.password || url.port) {
    throw new Error(`Blocked Gist URL with credentials or non-standard port.`);
  }
  const host = url.hostname.toLowerCase();
  if (!ALLOWED_GIST_HOSTS.has(host)) {
    throw new Error(`Blocked disallowed Gist host "${host}".`);
  }
  return url;
}

function hasUnsafeRawFilename(segment: string): boolean {
  return (
    !segment ||
    segment === '.' ||
    segment === '..' ||
    segment.includes('..') ||
    !SAFE_RAW_FILENAME_RE.test(segment)
  );
}

/**
 * Parse and strictly validate a GitHub Gist URL or shorthand reference:
 * - `https://gist.github.com/<owner>/<id>`
 * - `https://gist.github.com/<owner>/<id>/<rev>`
 * - `https://gist.github.com/<id>`
 * - `https://gist.githubusercontent.com/<owner>/<id>/raw/...`
 * - `<owner>/<id>`, `<owner>/<id>@<rev>`, or `<id>`
 */
export function parseGistReference(input: string): GistReference | null {
  if (typeof input !== 'string') return null;
  const trimmed = input.trim();
  if (!trimmed || trimmed.length > 2048) return null;
  if (/[\s<>"'`\\]/.test(trimmed)) return null;

  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)) {
    let url: URL;
    try {
      url = new URL(trimmed);
    } catch {
      return null;
    }
    if (url.protocol !== 'https:') return null;
    if (url.username || url.password || url.port || url.search) return null;

    const host = url.hostname.toLowerCase();
    const rawSegments = url.pathname.replace(/^\/+|\/+$/g, '').split('/').filter(Boolean);
    if (rawSegments.some((s) => s === '.' || s === '..' || s.includes('%'))) {
      return null;
    }

    if (host === 'gist.github.com') {
      if (rawSegments.length === 1) {
        const [id] = rawSegments;
        return GIST_ID_RE.test(id) ? { id: id.toLowerCase() } : null;
      }
      if (rawSegments.length === 2) {
        const [owner, id] = rawSegments;
        if (!GIST_OWNER_RE.test(owner) || !GIST_ID_RE.test(id)) return null;
        return { owner, id: id.toLowerCase() };
      }
      if (rawSegments.length === 3) {
        const [owner, id, rev] = rawSegments;
        if (!GIST_OWNER_RE.test(owner) || !GIST_ID_RE.test(id) || !GIST_REV_RE.test(rev)) {
          return null;
        }
        return { owner, id: id.toLowerCase(), rev: rev.toLowerCase() };
      }
      return null;
    }

    if (host === 'gist.githubusercontent.com') {
      if (rawSegments.length < 3 || rawSegments.length > 5) return null;
      const [owner, id, rawSeg, seg3, seg4] = rawSegments;
      if (!GIST_OWNER_RE.test(owner) || !GIST_ID_RE.test(id) || rawSeg !== 'raw') {
        return null;
      }
      if (rawSegments.length === 3) {
        return { owner, id: id.toLowerCase() };
      }
      if (rawSegments.length === 4) {
        if (GIST_REV_RE.test(seg3)) {
          return { owner, id: id.toLowerCase(), rev: seg3.toLowerCase() };
        }
        if (hasUnsafeRawFilename(seg3)) return null;
        return { owner, id: id.toLowerCase() };
      }
      // length === 5: /<owner>/<id>/raw/<rev>/<filename>
      if (!GIST_REV_RE.test(seg3) || hasUnsafeRawFilename(seg4)) return null;
      return { owner, id: id.toLowerCase(), rev: seg3.toLowerCase() };
    }

    return null;
  }

  // Shorthand forms: <owner>/<id>, <owner>/<id>@<rev>, or <id>
  if (trimmed.includes('?') || trimmed.includes('#') || trimmed.includes(':')) {
    return null;
  }
  if (trimmed.startsWith('/') || trimmed.endsWith('/')) return null;

  if (trimmed.includes('/')) {
    const slashParts = trimmed.split('/');
    if (slashParts.length !== 2) return null;
    const [owner, rest] = slashParts;
    if (!GIST_OWNER_RE.test(owner) || !rest) return null;

    if (rest.includes('@')) {
      const atParts = rest.split('@');
      if (atParts.length !== 2) return null;
      const [id, rev] = atParts;
      if (!GIST_ID_RE.test(id) || !GIST_REV_RE.test(rev)) return null;
      return { owner, id: id.toLowerCase(), rev: rev.toLowerCase() };
    }

    return GIST_ID_RE.test(rest) ? { owner, id: rest.toLowerCase() } : null;
  }

  if (trimmed.includes('@')) return null;
  return GIST_ID_RE.test(trimmed) ? { id: trimmed.toLowerCase() } : null;
}

/**
 * Extract a valid Gist reference string from a direct Gist URL/shorthand,
 * a `?gist=...` query string, or a full denovo.run page URL containing `?gist=...`.
 */
export function extractGistQueryParam(input: string): string | null {
  if (typeof input !== 'string') return null;
  const trimmed = input.trim();
  if (!trimmed) return null;

  if (parseGistReference(trimmed)) return trimmed;

  try {
    let search = '';
    if (trimmed.startsWith('?') || trimmed.startsWith('&')) {
      search = trimmed.replace(/^&/, '?');
    } else if (/^https?:\/\//i.test(trimmed)) {
      search = new URL(trimmed).search;
    } else if (trimmed.startsWith('gist=')) {
      search = `?${trimmed}`;
    }
    if (!search) return null;
    const raw = new URLSearchParams(search).get('gist')?.trim() ?? '';
    return raw && parseGistReference(raw) ? raw : null;
  } catch {
    return null;
  }
}

export function formatGistShareParam(ref: GistReference): string {
  const id = ref.id.trim().toLowerCase();
  if (!GIST_ID_RE.test(id)) {
    throw new Error(`Invalid Gist ID "${ref.id}".`);
  }
  if (ref.owner !== undefined) {
    if (!GIST_OWNER_RE.test(ref.owner)) {
      throw new Error(`Invalid Gist owner "${ref.owner}".`);
    }
    if (ref.rev !== undefined) {
      const rev = ref.rev.trim().toLowerCase();
      if (!GIST_REV_RE.test(rev)) {
        throw new Error(`Invalid Gist revision "${ref.rev}".`);
      }
      return `${ref.owner}/${id}@${rev}`;
    }
    return `${ref.owner}/${id}`;
  }
  return id;
}

export function buildGistShareUrl(origin: string, ref: GistReference, pathname = '/'): string {
  const cleanOrigin = origin.replace(/\/+$/, '');
  const cleanPath = pathname ? (pathname.startsWith('/') ? pathname : `/${pathname}`) : '/';
  return `${cleanOrigin}${cleanPath}?gist=${formatGistShareParam(ref)}`;
}

function hasStructureText(art: { pdbText?: unknown; cifText?: unknown }): boolean {
  return (
    (typeof art.pdbText === 'string' && art.pdbText.trim().length > 0) ||
    (typeof art.cifText === 'string' && art.cifText.trim().length > 0)
  );
}

const GIST_COMPRESSION_ENCODING = 'gzip+base64' as const;

function stripStructureArtifact(art: StructureArtifact): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const hasPdb = typeof art.pdbText === 'string' && art.pdbText.trim().length > 0;
  const hasCif = typeof art.cifText === 'string' && art.cifText.trim().length > 0;
  const canRederive = hasPdb || hasCif;
  const hasCollectionItems =
    isPlainObject(art.collection) &&
    Array.isArray(art.collection.items) &&
    art.collection.items.length > 0;

  for (const [key, val] of Object.entries(art as unknown as Record<string, unknown>)) {
    if (FORBIDDEN_KEYS.has(key) || val === undefined) continue;
    if (
      canRederive &&
      (key === 'caCoords' ||
        key === 'backbone4Coords' ||
        key === 'sourceResidues' ||
        key === 'atomPresence' ||
        key === 'chainIds' ||
        key === 'insCodes')
    ) {
      continue;
    }
    // Drop redundant mmCIF text (which carries large RCSB dictionary headers) whenever pdbText is present
    if (hasPdb && key === 'cifText') {
      continue;
    }
    // Avoid duplicating item[0]'s coordinate text at the collection wrapper top level
    if (hasCollectionItems && (key === 'pdbText' || key === 'cifText')) {
      continue;
    }
    if (key === 'collection' && isPlainObject(val) && Array.isArray(val.items)) {
      out.collection = {
        ...val,
        items: val.items.map((item) =>
          isPlainObject(item) && item.kind === 'StructureArtifact'
            ? stripStructureArtifact(item as unknown as StructureArtifact)
            : item,
        ),
      };
      continue;
    }
    out[key] = val;
  }
  return out;
}

function stripFoldedStructure(art: FoldedStructure): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, val] of Object.entries(art as unknown as Record<string, unknown>)) {
    if (FORBIDDEN_KEYS.has(key) || val === undefined) continue;
    if (key === 'folds' && Array.isArray(val)) {
      out.folds = val.map((fold) => {
        if (!isPlainObject(fold)) return fold;
        const hasPdb = typeof fold.pdbText === 'string' && fold.pdbText.trim().length > 0;
        const canRederive = hasStructureText(fold as { pdbText?: unknown; cifText?: unknown });
        const nextFold: Record<string, unknown> = {};
        for (const [fk, fv] of Object.entries(fold)) {
          if (FORBIDDEN_KEYS.has(fk) || fv === undefined) continue;
          if (canRederive && (fk === 'caCoords' || fk === 'backbone4Coords')) {
            continue;
          }
          if (hasPdb && fk === 'cifText') {
            continue;
          }
          nextFold[fk] = fv;
        }
        return nextFold;
      });
      continue;
    }
    out[key] = val;
  }
  return out;
}

function stripArtifactForGist(art: CanonicalArtifact): unknown {
  if (art.kind === 'StructureArtifact') {
    return stripStructureArtifact(art);
  }
  if (art.kind === 'FoldedStructure') {
    return stripFoldedStructure(art);
  }
  return art;
}

/**
 * Serialize a WorkflowGraph + computed node outputs into a compact `denovo.gist-session/v1` JSON string.
 * Strips re-derivable coordinate arrays (`caCoords`, `backbone4Coords`) from `StructureArtifact`,
 * `ArtifactCollection`, and `FoldedStructure`, preserves TypedArrays via `encodeWireValue`,
 * and skips `TrajectoryFrames` with a clear warning.
 */
export function serializeGistSession(
  graph: WorkflowGraph,
  nodeOutputs: Record<string, Record<string, CanonicalArtifact>>,
): SerializedGistSession {
  const warnings: string[] = [];
  const compactOutputs: Record<string, Record<string, unknown>> = {};

  for (const node of graph.nodes) {
    if (FORBIDDEN_KEYS.has(node.id)) continue;
    const rawOuts = nodeOutputs?.[node.id];
    if (!isPlainObject(rawOuts)) continue;

    const nodePorts: Record<string, unknown> = {};
    for (const [portName, rawArt] of Object.entries(rawOuts)) {
      if (FORBIDDEN_KEYS.has(portName) || !isPlainObject(rawArt)) continue;
      const art = rawArt as unknown as CanonicalArtifact;
      if (art.kind === 'TrajectoryFrames') {
        const nodeLabel = node.label || node.id;
        const frameNote =
          typeof art.frameCount === 'number' && Number.isFinite(art.frameCount)
            ? ` (${art.frameCount} frames)`
            : '';
        warnings.push(
          `Skipped trajectory frames on "${nodeLabel}.${portName}"${frameNote} to keep the Gist session compact.`,
        );
        continue;
      }
      Object.defineProperty(nodePorts, portName, {
        value: stripArtifactForGist(art),
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }

    if (Object.keys(nodePorts).length > 0) {
      Object.defineProperty(compactOutputs, node.id, {
        value: nodePorts,
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }
  }

  const wirePayload = encodeWireValue({
    schema: GIST_SESSION_SCHEMA,
    createdAt: new Date().toISOString(),
    graph,
    outputs: compactOutputs,
  });

  const json = JSON.stringify(wirePayload);
  const bytes = new TextEncoder().encode(json).byteLength;

  return {
    json,
    bytes,
    nodeCount: graph.nodes.length,
    resultCount: Object.keys(compactOutputs).length,
    warnings,
  };
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

async function compressGzipBase64(text: string): Promise<string | null> {
  if (typeof CompressionStream === 'undefined') return null;
  const rawBytes = new TextEncoder().encode(text);
  const cs = new CompressionStream('gzip');
  const writer = cs.writable.getWriter();
  // Drain readable side concurrently with write/close to avoid TransformStream backpressure deadlock on >64 KB inputs
  const writePromise = writer.write(rawBytes).then(() => writer.close());
  const reader = cs.readable.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        total += value.byteLength;
      }
    }
    await writePromise;
  } finally {
    reader.releaseLock();
  }
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytesToBase64(merged);
}

async function decompressGzipBase64(b64: string, maxBytes: number): Promise<string> {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This browser does not support DecompressionStream("gzip") to load compressed Gist sessions.');
  }
  const compressed = base64ToBytes(b64);
  const ds = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  const writePromise = writer.write(compressed as unknown as BufferSource).then(() => writer.close());
  const reader = ds.readable.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) {
        total += value.byteLength;
        if (total > maxBytes) {
          await reader.cancel('Decompressed Gist payload exceeds maximum allowed size').catch(() => {});
          throw new Error(
            `Decompressed Gist payload exceeds maximum allowed size (${(maxBytes / 1024 / 1024).toFixed(1)} MiB).`,
          );
        }
        chunks.push(value);
      }
    }
    await writePromise;
  } finally {
    reader.releaseLock();
  }
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(merged);
}

/**
 * Serialize a WorkflowGraph + computed node outputs into a compact gzip+base64
 * `denovo.gist-session/v1` JSON envelope so multi-design sessions stay well under
 * GitHub's `gist.github.com` web form size limit.
 */
export async function serializeGistSessionCompressed(
  graph: WorkflowGraph,
  nodeOutputs: Record<string, Record<string, CanonicalArtifact>>,
): Promise<SerializedGistSession> {
  const base = serializeGistSession(graph, nodeOutputs);
  try {
    const gzB64 = await compressGzipBase64(base.json);
    if (gzB64) {
      const envelope = JSON.stringify({
        schema: GIST_SESSION_SCHEMA,
        encoding: GIST_COMPRESSION_ENCODING,
        data: gzB64,
      });
      const envBytes = new TextEncoder().encode(envelope).byteLength;
      if (envBytes < base.bytes) {
        return {
          ...base,
          json: envelope,
          bytes: envBytes,
        };
      }
    }
  } catch {
    // Fall back cleanly to uncompressed JSON if CompressionStream fails
  }
  return base;
}

function parseCoordinatesFromText(pdbText?: string, cifText?: string) {
  if (typeof pdbText === 'string' && pdbText.trim().length > 0) {
    return parseStructureText(pdbText);
  }
  if (typeof cifText === 'string' && cifText.trim().length > 0) {
    return parseStructureText(cifText);
  }
  return null;
}

function rehydrateStructureArtifact(raw: Record<string, unknown>): StructureArtifact {
  const art = { ...raw } as unknown as StructureArtifact;

  if (art.collection && isPlainObject(art.collection) && Array.isArray(art.collection.items)) {
    art.collection = {
      ...art.collection,
      items: art.collection.items.map((item) =>
        isPlainObject(item) && item.kind === 'StructureArtifact'
          ? rehydrateStructureArtifact(item)
          : (item as StructureArtifact),
      ),
    };
    const first = art.collection.items[0];
    if (first && !hasStructureText(art)) {
      if (typeof first.pdbText === 'string' && first.pdbText) art.pdbText = first.pdbText;
      if (typeof first.cifText === 'string' && first.cifText) art.cifText = first.cifText;
    }
  }

  const parsed = parseCoordinatesFromText(art.pdbText, art.cifText);
  if (parsed) {
    art.caCoords = parsed.caCoords;
    art.backbone4Coords = parsed.backbone4Coords;
    if (!art.sourceResidues) art.sourceResidues = parsed.sourceResidues;
    if (!art.atomPresence) art.atomPresence = parsed.atomPresence;
    if (!art.chainIds) art.chainIds = parsed.chainIds;
    if (!art.insCodes) art.insCodes = parsed.insCodes;
  } else {
    if (!(art.caCoords instanceof Float32Array)) {
      art.caCoords = new Float32Array(0);
    }
    if (!(art.backbone4Coords instanceof Float32Array)) {
      art.backbone4Coords = new Float32Array(0);
    }
  }

  return art;
}

function rehydrateFoldedStructure(raw: Record<string, unknown>): FoldedStructure {
  if (!Array.isArray(raw.folds)) {
    throw new Error('Malformed FoldedStructure artifact in Gist session: missing folds[].');
  }
  const art = { ...raw } as unknown as FoldedStructure;
  art.folds = raw.folds.map((item): FoldedCandidate => {
    if (!isPlainObject(item)) {
      throw new Error('Malformed fold candidate in Gist session.');
    }
    const fold = { ...item } as unknown as FoldedCandidate & { cifText?: string };
    const parsed = parseCoordinatesFromText(fold.pdbText, fold.cifText);
    if (parsed) {
      fold.caCoords = parsed.caCoords;
    } else if (!(fold.caCoords instanceof Float32Array)) {
      fold.caCoords = new Float32Array(0);
    }
    return fold;
  });
  return art;
}

function rehydrateArtifactFromGist(raw: unknown): CanonicalArtifact {
  if (!isPlainObject(raw) || typeof raw.kind !== 'string' || typeof raw.id !== 'string' || !raw.id.trim()) {
    throw new Error('Malformed artifact entry in Gist session.');
  }
  if (raw.kind === 'StructureArtifact') {
    return rehydrateStructureArtifact(raw);
  }
  if (raw.kind === 'FoldedStructure') {
    return rehydrateFoldedStructure(raw);
  }
  return raw as unknown as CanonicalArtifact;
}

/**
 * Parse and validate a serialized `denovo.gist-session/v1` JSON document:
 * - Decodes via prototype-pollution-hardened `decodeWireValue`
 * - Validates schema version and workflow DAG via `validateGraph` / `normalizeWorkflowGraph`
 * - Re-derives `caCoords` / `backbone4Coords` from `pdbText`/`cifText` via `parsePdbBackbone`
 */
export function deserializeGistSession(rawText: string): GistSessionPayload {
  if (typeof rawText !== 'string' || !rawText.trim()) {
    throw new Error('Gist session payload is empty.');
  }

  let parsedJson: unknown;
  try {
    parsedJson = JSON.parse(rawText);
  } catch {
    throw new Error('Invalid JSON in Gist session payload.');
  }

  const decoded = decodeWireValue<unknown>(parsedJson);
  if (!isPlainObject(decoded)) {
    throw new Error('Gist session must be a JSON object.');
  }
  if (decoded.schema !== GIST_SESSION_SCHEMA) {
    throw new Error(
      `Unsupported Gist session schema "${String(decoded.schema ?? 'missing')}" (expected "${GIST_SESSION_SCHEMA}").`,
    );
  }

  const graph = normalizeWorkflowGraph(
    decoded.graph,
    'gist-session',
    'Shared Gist Session',
    true,
  );
  const graphErrors = validateGraph(graph, { requireRunnable: false });
  if (graphErrors.length > 0) {
    throw new Error(`Invalid workflow graph in Gist session: ${graphErrors.join('; ')}`);
  }

  if (decoded.outputs !== undefined && !isPlainObject(decoded.outputs)) {
    throw new Error('Malformed outputs map in Gist session.');
  }

  const validNodeIds = new Set(graph.nodes.map((n) => n.id));
  const outputs: Record<string, Record<string, CanonicalArtifact>> = {};

  if (isPlainObject(decoded.outputs)) {
    for (const [nodeId, rawPorts] of Object.entries(decoded.outputs)) {
      if (FORBIDDEN_KEYS.has(nodeId) || !validNodeIds.has(nodeId)) continue;
      if (!isPlainObject(rawPorts)) {
        throw new Error(`Malformed outputs entry for node "${nodeId}" in Gist session.`);
      }
      const hydratedPorts: Record<string, CanonicalArtifact> = {};
      for (const [portName, rawArt] of Object.entries(rawPorts)) {
        if (FORBIDDEN_KEYS.has(portName)) continue;
        Object.defineProperty(hydratedPorts, portName, {
          value: rehydrateArtifactFromGist(rawArt),
          enumerable: true,
          writable: true,
          configurable: true,
        });
      }
      if (Object.keys(hydratedPorts).length > 0) {
        Object.defineProperty(outputs, nodeId, {
          value: hydratedPorts,
          enumerable: true,
          writable: true,
          configurable: true,
        });
      }
    }
  }

  return {
    schema: GIST_SESSION_SCHEMA,
    createdAt: typeof decoded.createdAt === 'string' ? decoded.createdAt : new Date(0).toISOString(),
    graph,
    outputs,
  };
}

/**
 * Async deserializer supporting both plain JSON `denovo.gist-session/v1` payloads and
 * `encoding: "gzip+base64"` compressed envelopes (with bounded decompression against zip bombs).
 */
export async function deserializeGistSessionAsync(
  rawText: string,
  maxDecompressedBytes = DEFAULT_MAX_GIST_BYTES * 4,
): Promise<GistSessionPayload> {
  if (typeof rawText !== 'string' || !rawText.trim()) {
    throw new Error('Gist session payload is empty.');
  }
  let outer: unknown;
  try {
    outer = JSON.parse(rawText);
  } catch {
    throw new Error('Invalid JSON in Gist session payload.');
  }
  if (
    isPlainObject(outer) &&
    outer.schema === GIST_SESSION_SCHEMA &&
    outer.encoding === GIST_COMPRESSION_ENCODING
  ) {
    if (typeof outer.data !== 'string' || !outer.data.trim()) {
      throw new Error('Malformed compressed Gist session payload.');
    }
    const innerText = await decompressGzipBase64(outer.data, maxDecompressedBytes);
    return deserializeGistSession(innerText);
  }
  return deserializeGistSession(rawText);
}

async function readBoundedResponseText(
  resp: Response,
  maxBytes: number,
  signal?: AbortSignal,
): Promise<{ text: string; bytes: number }> {
  const rawLenHeader = resp.headers?.get?.('content-length');
  if (rawLenHeader) {
    const cl = Number(rawLenHeader);
    if (Number.isFinite(cl) && cl > maxBytes) {
      try {
        await resp.body?.cancel?.();
      } catch {
        /* ignore cancel error */
      }
      throw new Error(
        `Gist payload (${(cl / 1024 / 1024).toFixed(1)} MiB) exceeds maximum allowed size (${(maxBytes / 1024 / 1024).toFixed(1)} MiB).`,
      );
    }
  }

  if (resp.body && typeof resp.body.getReader === 'function') {
    const reader = resp.body.getReader();
    const chunks: Uint8Array[] = [];
    let totalBytes = 0;
    try {
      while (true) {
        if (signal?.aborted) {
          await reader.cancel(signal.reason).catch(() => {});
          throw signal.reason instanceof Error ? signal.reason : new Error('Aborted');
        }
        const { value, done } = await reader.read();
        if (done) break;
        if (value) {
          totalBytes += value.byteLength;
          if (totalBytes > maxBytes) {
            await reader.cancel('Gist payload exceeds maximum allowed size').catch(() => {});
            throw new Error(
              `Gist payload exceeds maximum allowed size (${(maxBytes / 1024 / 1024).toFixed(1)} MiB).`,
            );
          }
          chunks.push(value);
        }
      }
    } finally {
      reader.releaseLock();
    }
    const merged = new Uint8Array(totalBytes);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return {
      text: new TextDecoder().decode(merged),
      bytes: totalBytes,
    };
  }

  const text = await resp.text();
  const bytes = new TextEncoder().encode(text).byteLength;
  if (bytes > maxBytes) {
    throw new Error(
      `Gist payload (${(bytes / 1024 / 1024).toFixed(1)} MiB) exceeds maximum allowed size (${(maxBytes / 1024 / 1024).toFixed(1)} MiB).`,
    );
  }
  return { text, bytes };
}

interface GitHubGistFileEntry {
  filename?: string;
  content?: string;
  truncated?: boolean;
  raw_url?: string;
  size?: number;
}

function selectGistJsonFile(files: Record<string, unknown>): { filename: string; entry: GitHubGistFileEntry } {
  if (Object.hasOwn(files, GIST_SESSION_FILENAME) && isPlainObject(files[GIST_SESSION_FILENAME])) {
    return {
      filename: GIST_SESSION_FILENAME,
      entry: files[GIST_SESSION_FILENAME] as unknown as GitHubGistFileEntry,
    };
  }
  for (const [name, val] of Object.entries(files)) {
    if (FORBIDDEN_KEYS.has(name) || !isPlainObject(val)) continue;
    if (name.toLowerCase().endsWith('.json')) {
      return { filename: name, entry: val as unknown as GitHubGistFileEntry };
    }
  }
  throw new Error(
    `No "${GIST_SESSION_FILENAME}" or .json file found inside this Gist.`,
  );
}

/**
 * Fetch, enforce host allowlist & max-bytes limits, and deserialize a Gist session.
 * When `ref.owner` is present, tries direct `gist.githubusercontent.com` raw fetch first
 * (avoiding GitHub API rate limits), then falls back to `api.github.com/gists/<id>`.
 */
export async function loadGistSession(
  refInput: string,
  opts: LoadGistOptions = {},
): Promise<LoadedGistSession> {
  const ref = parseGistReference(refInput);
  if (!ref) {
    throw new Error('Invalid GitHub Gist URL or ID.');
  }

  const fetchFn = opts.fetchFn ?? globalThis.fetch;
  if (typeof fetchFn !== 'function') {
    throw new Error('Fetch API is unavailable in this environment.');
  }
  const maxBytes =
    typeof opts.maxBytes === 'number' && Number.isFinite(opts.maxBytes) && opts.maxBytes > 0
      ? Math.floor(opts.maxBytes)
      : DEFAULT_MAX_GIST_BYTES;

  // 1. Direct raw fast path when owner is known (zero GitHub API rate-limit hit)
  if (ref.owner) {
    const revPart = ref.rev ? `${ref.rev}/` : '';
    const rawUrl = `https://gist.githubusercontent.com/${ref.owner}/${ref.id}/raw/${revPart}${GIST_SESSION_FILENAME}`;
    assertAllowedGistUrl(rawUrl);

    try {
      const rawResp = await fetchFn(rawUrl, { signal: opts.signal });
      if (rawResp.url) {
        assertAllowedGistUrl(rawResp.url);
      }
      if (rawResp.ok) {
        const { text, bytes } = await readBoundedResponseText(rawResp, maxBytes, opts.signal);
        const session = await deserializeGistSessionAsync(text, Math.max(maxBytes * 4, DEFAULT_MAX_GIST_BYTES * 2));
        return { session, ref, sourceUrl: rawUrl, bytes };
      }
      // Fall back to GitHub API only when raw URL isn't found (e.g. non-standard filename)
      if (rawResp.status !== 404 && rawResp.status !== 403 && rawResp.status !== 400) {
        throw new Error(`HTTP ${rawResp.status} fetching raw Gist.`);
      }
    } catch (err) {
      if (opts.signal?.aborted) throw err;
      if (
        err instanceof Error &&
        /exceeds maximum|Blocked|Invalid JSON|Unsupported Gist|Invalid workflow graph|Malformed|Forbidden property/i.test(
          err.message,
        )
      ) {
        throw err;
      }
      // Network or 404 fallback to api.github.com below
    }
  }

  // 2. Fallback / owner-less path via api.github.com
  const apiUrl = ref.rev
    ? `https://api.github.com/gists/${ref.id}/${ref.rev}`
    : `https://api.github.com/gists/${ref.id}`;
  assertAllowedGistUrl(apiUrl);

  const apiResp = await fetchFn(apiUrl, {
    signal: opts.signal,
    headers: { Accept: 'application/vnd.github+json' },
  });
  if (apiResp.url) {
    assertAllowedGistUrl(apiResp.url);
  }
  if (!apiResp.ok) {
    if (apiResp.status === 404) {
      throw new Error('Gist not found (or deleted/private).');
    }
    if (apiResp.status === 403 || apiResp.status === 429) {
      throw new Error(
        'GitHub API rate limit reached — include the Gist owner in the URL (e.g. owner/id) to load directly.',
      );
    }
    throw new Error(`HTTP ${apiResp.status} fetching Gist metadata.`);
  }

  const { text: metaText } = await readBoundedResponseText(apiResp, maxBytes, opts.signal);
  let metaObj: unknown;
  try {
    metaObj = JSON.parse(metaText);
  } catch {
    throw new Error('Invalid JSON returned from GitHub Gist API.');
  }
  if (!isPlainObject(metaObj) || !isPlainObject(metaObj.files)) {
    throw new Error('Unexpected GitHub Gist API response (missing files).');
  }

  const { entry } = selectGistJsonFile(metaObj.files);

  if (entry.truncated === true || typeof entry.content !== 'string') {
    if (typeof entry.raw_url !== 'string' || !entry.raw_url) {
      throw new Error('Gist file is truncated and missing raw_url.');
    }
    assertAllowedGistUrl(entry.raw_url);
    const rawResp = await fetchFn(entry.raw_url, { signal: opts.signal });
    if (rawResp.url) {
      assertAllowedGistUrl(rawResp.url);
    }
    if (!rawResp.ok) {
      throw new Error(`HTTP ${rawResp.status} fetching truncated Gist raw_url.`);
    }
    const { text: rawText, bytes } = await readBoundedResponseText(rawResp, maxBytes, opts.signal);
    const session = await deserializeGistSessionAsync(rawText, Math.max(maxBytes * 4, DEFAULT_MAX_GIST_BYTES * 2));
    return { session, ref, sourceUrl: entry.raw_url, bytes };
  }

  const contentBytes = new TextEncoder().encode(entry.content).byteLength;
  if (contentBytes > maxBytes) {
    throw new Error(
      `Gist payload (${(contentBytes / 1024 / 1024).toFixed(1)} MiB) exceeds maximum allowed size (${(maxBytes / 1024 / 1024).toFixed(1)} MiB).`,
    );
  }
  const session = await deserializeGistSessionAsync(entry.content, Math.max(maxBytes * 4, DEFAULT_MAX_GIST_BYTES * 2));
  return { session, ref, sourceUrl: apiUrl, bytes: contentBytes };
}
