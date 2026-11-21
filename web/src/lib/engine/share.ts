// Compact shareable workflow URLs (#p= preset diff OR #g= deflate-raw base64url)
// + clean human/LLM-readable YAML/JSON workflow recipe serializer.

import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from './registry';
import { autoLayoutGraph, defaultParamsForAdapter, validateGraph } from './graph-utils';
import { computeUploadDigest, isValidSha256Hex } from './param-validation';
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

function parseFixedBareProteinContig(contig: string): number | null {
  const m = /^(\d+)(?:-(\d+))?$/.exec(contig.trim());
  if (!m) return null;
  const lo = Number(m[1]);
  const hi = m[2] !== undefined ? Number(m[2]) : lo;
  if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo !== hi) return null;
  return lo;
}

function hasRfd4OnlyContigSyntax(contig: string): boolean {
  for (const rawTok of contig.split(/[,\s;]+/)) {
    const tok = rawTok.trim();
    if (!tok) continue;
    // Secondary-structure labeled contig segment like "h1:16-16P"
    if (/^[A-Za-z0-9_]+:\d+(?:-\d+)?[PRDN]$/i.test(tok)) return true;
    // RFD4 polymer-typed contig suffix like "120-120P", "70-70D", "20-20R", "12-12N"
    if (/^\d+(?:-\d+)?[PRDN]$/i.test(tok)) return true;
    // Explicit ligand selector like "A:SAM" or "A:ZN"
    if (/^[A-Za-z]+:[A-Za-z][A-Za-z0-9]{0,4}$/.test(tok)) return true;
  }
  return false;
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

  // 1. Check for exact preset topology match -> pick the preset with the shortest diff hash!
  let bestPresetMatch: EncodedHashResult | null = null;
  for (const preset of WORKFLOW_PRESETS) {
    if (!sameTopology(graph, preset.graph)) continue;
    const qs = new URLSearchParams();
    qs.set('p', preset.id);
    let presetStripped = false;
    for (let i = 0; i < graph.nodes.length; i++) {
      const n = graph.nodes[i];
      const baseN = preset.graph.nodes[i];
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
    if (!Object.hasOwn(ADAPTER_REGISTRY, adapterId)) {
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

  const normalizedRaw: WorkflowGraph = {
    id: typeof parsed.id === 'string' && parsed.id.trim() ? parsed.id.trim() : fallbackId,
    name: typeof parsed.name === 'string' && parsed.name.trim() ? parsed.name.trim() : fallbackName,
    ...(typeof parsed.description === 'string' ? { description: parsed.description } : {}),
    nodes: validatedNodes,
    edges: validatedEdges,
  };

  // Transparently migrate legacy graphs:
  // 1) Remap legacy `template` input port on genie3/rfd4-proteina -> unified `structure` port.
  // 2) Fold legacy `rfd4-conditioner` nodes wired into genie3/rfd4-proteina into inline task fields.
  const normalized = migrateLegacyConditioningGraph(normalizedRaw);

  // Recipes may be unfinished, but their existing wires must be valid before layout/rendering.
  const errors = validateGraph(normalized, { requireRunnable: false });
  if (errors.length) throw new Error(errors.join('\n'));
  return normalized.nodes.some((n) => !n.position) ? autoLayoutGraph(normalized) : normalized;
}

function migrateLegacyConditioningGraph(graph: WorkflowGraph): WorkflowGraph {
  const nodeById = new Map(graph.nodes.map((n) => [n.id, { ...n, params: { ...n.params } }]));
  let edges = graph.edges.map((e) => {
    const dst = nodeById.get(e.toNode);
    if (
      dst &&
      (dst.adapterId === 'genie3' || dst.adapterId === 'rfd4-proteina') &&
      e.toPort === 'template'
    ) {
      return { ...e, toPort: 'structure' };
    }
    return { ...e };
  });

  const condNodesToDrop = new Set<string>();
  for (const node of [...nodeById.values()]) {
    if (node.adapterId !== 'rfd4-conditioner') continue;
    const outCondEdges = edges.filter((e) => e.fromNode === node.id && e.fromPort === 'condition');
    if (outCondEdges.length === 0) continue;

    const inTplEdge = edges.find((e) => e.toNode === node.id && (e.toPort === 'template' || e.toPort === 'structure'));
    const cp = node.params ?? {};
    const tp = String(cp.taskPreset ?? 'custom');
    let allFolded = true;

    for (const ce of outCondEdges) {
      const dst = nodeById.get(ce.toNode);
      if (!dst || (dst.adapterId !== 'rfd4-proteina' && dst.adapterId !== 'genie3')) {
        allFolded = false;
        continue;
      }
      const lockChain = String(cp.lockChain ?? '').trim();
      const motifSpan = String(cp.motifSpan ?? '').trim();
      const hotspots = String(cp.hotspots ?? '').trim();
      const contig = String(cp.contig ?? '').trim();
      const cathCode = String(cp.cathCode ?? '').trim();
      const hbondMode = String(cp.hbondMode ?? 'none').trim() || 'none';
      const hbondSelection = String(cp.hbondSelection ?? '').trim();
      const unindexedMotif = Boolean(cp.unindexedMotif);
      const rawJsonOverride = String(cp.rawJsonOverride ?? '').trim();
      const customSpecYaml = String(cp.customSpecYaml ?? '').trim();
      const rawOverride = rawJsonOverride || customSpecYaml;
      const legacyMode = String(cp.mode ?? '').trim();
      const hasPartialFraction =
        cp.partialFraction !== undefined &&
        cp.partialFraction !== null &&
        String(cp.partialFraction).trim() !== '';
      const isPartial =
        legacyMode === 'partial' || Boolean(cp.partialDiffusion) || hasPartialFraction;
      const isInverseFolding =
        legacyMode === 'inverse-folding' || Boolean(cp.inverseFolding);
      const sourceCondIssues = normalizeUnresolvedIssues(cp.unresolvedImportIssues);

      if (dst.adapterId === 'rfd4-proteina') {
        const mergedIssues = normalizeUnresolvedIssues(dst.params.unresolvedImportIssues);
        for (const issue of sourceCondIssues) {
          if (!mergedIssues.includes(issue)) mergedIssues.push(issue);
        }
        const inferredMode =
          tp !== 'custom'
            ? 'preset'
            : isPartial
              ? 'partial'
              : isInverseFolding
                ? 'inverse-folding'
                : lockChain
                  ? 'binder'
                  : motifSpan
                    ? 'motif'
                    : hbondMode !== 'none' || cathCode || rawOverride || contig
                      ? 'conditioned'
                      : String(dst.params.mode ?? 'unconditional') === 'conditioned'
                        ? 'conditioned'
                        : String(dst.params.mode ?? 'unconditional');
        dst.params = {
          ...dst.params,
          mode: inferredMode,
          ...(tp !== 'custom' ? { presetId: tp } : {}),
          ...(hasPartialFraction ? { partialFraction: Number(cp.partialFraction) } : {}),
          ...(contig ? { contig } : {}),
          ...(lockChain ? { lockChain } : {}),
          ...(hotspots ? { hotspots } : {}),
          ...(motifSpan ? { motifSpan } : {}),
          ...(unindexedMotif ? { unindexedMotif: true } : {}),
          ...(hbondMode !== 'none' ? { hbondMode, hbondSelection } : {}),
          ...(cathCode ? { cathCode } : {}),
          ...(rawOverride ? { customSpecYaml: rawOverride } : {}),
          ...(mergedIssues.length > 0 ? { unresolvedImportIssues: mergedIssues } : {}),
        };
      } else {
        const inferredTaskMode = lockChain ? 'binder' : motifSpan ? 'motif' : 'unconditional';
        const importIssues = normalizeUnresolvedIssues(dst.params.unresolvedImportIssues);
        const addIssue = (msg: string) => {
          if (!importIssues.includes(msg)) importIssues.push(msg);
        };
        for (const issue of sourceCondIssues) {
          addIssue(issue);
        }

        if (tp !== 'custom') {
          addIssue(
            `Legacy taskPreset "${tp}" is not supported by Genie 3 (switch to RFD4-Proteina or configure a Genie 3 binder/motif task).`,
          );
        }
        if (lockChain && motifSpan) {
          addIssue(
            `Legacy combined binder (lockChain="${lockChain}") and motif (motifSpan="${motifSpan}") conditioning cannot be run simultaneously in Genie 3 (choose binder or motif mode, or use RFD4-Proteina).`,
          );
        }
        if (cathCode) {
          addIssue(`Legacy CATH fold conditioning (cathCode="${cathCode}") is not supported by Genie 3.`);
        }
        if (hbondMode !== 'none' || hbondSelection) {
          const detail = [
            hbondMode !== 'none' ? `hbondMode="${hbondMode}"` : '',
            hbondSelection ? `hbondSelection="${hbondSelection}"` : '',
          ]
            .filter(Boolean)
            .join(', ');
          addIssue(`Legacy H-bond conditioning (${detail}) is not supported by Genie 3.`);
        }
        if (isPartial) {
          const detail = hasPartialFraction
            ? `partialFraction=${String(cp.partialFraction)}`
            : 'partial diffusion';
          addIssue(`Legacy partial diffusion (${detail}) is not supported by Genie 3.`);
        }
        if (isInverseFolding) {
          addIssue('Legacy inverse folding setting is not supported by Genie 3.');
        }
        if (unindexedMotif) {
          addIssue('Legacy unindexed motif placement (unindexedMotif=true) is not supported by Genie 3.');
        }
        if (rawOverride) {
          const fieldName = rawJsonOverride ? 'rawJsonOverride' : 'customSpecYaml';
          addIssue(`Legacy custom spec override (${fieldName}) is not supported by Genie 3.`);
        }
        if (hotspots && inferredTaskMode !== 'binder') {
          addIssue(
            `Legacy hotspots "${hotspots}" require binder mode (lockChain) in Genie 3 and cannot be represented in "${inferredTaskMode}" mode.`,
          );
        }

        let translatedLength: number | undefined;
        const hasTemplateContext =
          Boolean(inTplEdge) || edges.some((e) => e.toNode === dst.id && e.toPort === 'structure');
        if (contig) {
          if (hasRfd4OnlyContigSyntax(contig)) {
            addIssue(
              `Legacy contig "${contig}" uses RFD4 polymer-suffix, secondary-structure, or ligand syntax not supported by Genie 3.`,
            );
          } else if (inferredTaskMode === 'unconditional') {
            const fixedLen = parseFixedBareProteinContig(contig);
            if (fixedLen !== null) {
              if (hasTemplateContext) {
                // Wiring a template into unconditional Genie 3 triggers automatic binder conditioning in genie3.worker.ts.
                addIssue(
                  `Legacy bare contig "${contig}" with a wired template structure would silently switch Genie 3 into automatic binder mode (disconnect the template for unconditional length ${fixedLen}, or set binder lockChain / motifSpan).`,
                );
              } else {
                // Translate fixed bare monomer contigs (e.g. "120-120") into Genie 3's effective length parameter.
                translatedLength = fixedLen;
              }
            } else {
              addIssue(
                `Legacy contig "${contig}" cannot be mapped automatically to an unconditional Genie 3 task (set binder lockChain or motifSpan, pick a fixed length, or use RFD4-Proteina).`,
              );
            }
          }
        }

        dst.params = {
          ...dst.params,
          taskMode: inferredTaskMode,
          ...(translatedLength !== undefined ? { length: translatedLength } : {}),
          ...(contig ? { contig } : {}),
          ...(lockChain ? { lockChain } : {}),
          ...(hotspots ? { hotspots } : {}),
          ...(motifSpan ? { motifSpan } : {}),
          ...(importIssues.length > 0 ? { unresolvedImportIssues: importIssues } : {}),
        };
      }

      if (inTplEdge) {
        const alreadyHasStruct = edges.some((e) => e.toNode === dst.id && e.toPort === 'structure');
        if (!alreadyHasStruct) {
          edges.push({
            fromNode: inTplEdge.fromNode,
            fromPort: inTplEdge.fromPort,
            toNode: dst.id,
            toPort: 'structure',
          });
        }
      }
    }

    if (allFolded) {
      condNodesToDrop.add(node.id);
    }
  }

  if (condNodesToDrop.size > 0) {
    edges = edges.filter((e) => !condNodesToDrop.has(e.fromNode) && !condNodesToDrop.has(e.toNode));
    for (const id of condNodesToDrop) nodeById.delete(id);
  }

  return {
    ...graph,
    nodes: [...nodeById.values()],
    edges,
  };
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
      if (paramKey === '_label') {
        if (v.trim()) node.label = v;
        continue;
      }
      const defaults = defaultParamsForAdapter(node.adapterId);
      const templateVal =
        node.params[paramKey] !== undefined
          ? node.params[paramKey]
          : defaults[paramKey] !== undefined
            ? defaults[paramKey]
            : PARAM_TYPE_HINTS[paramKey];
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
      const adapterId = Object.hasOwn(SHORT_TO_ADAPTER, shortA) ? SHORT_TO_ADAPTER[shortA] : shortA;
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
      const obj = JSON.parse(bodyStr) as Record<string, unknown>;
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
      const m = /^([a-zA-Z0-9_.-]+)\.([a-zA-Z0-9_-]+)\s*->\s*([a-zA-Z0-9_.-]+)\.([a-zA-Z0-9_-]+)$/.exec(clean);
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
