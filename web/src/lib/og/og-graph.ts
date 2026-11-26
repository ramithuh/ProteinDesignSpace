// Lightweight, zero-Svelte workflow decoder & metadata extractor for Cloudflare Pages Edge OG previews.

import {
  MAX_HASH_CHARS,
  MAX_IMPORT_EDGES,
  MAX_IMPORT_NODES,
  SHORT_TO_ADAPTER,
  decodeWorkflowQuery,
  extractWorkflowSearchOrHash,
} from '../engine/flow-dsl';
import { ADAPTER_REGISTRY } from '../engine/registry';

const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

function safeStr(v: unknown): string {
  return typeof v === 'string'
    ? v
    : typeof v === 'number' || typeof v === 'boolean'
      ? String(v)
      : '';
}

function sanitizeOgValue(val: unknown, depth = 0): unknown {
  if (val === null || typeof val !== 'object') return val;
  if (depth > 32) return undefined;
  if (Array.isArray(val)) return val.map((item) => sanitizeOgValue(item, depth + 1));
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(val as Record<string, unknown>)) {
    if (FORBIDDEN_KEYS.has(k)) continue;
    out[k] = sanitizeOgValue(v, depth + 1);
  }
  return out;
}

function sanitizeOgParams(obj: Record<string, unknown>): Record<string, unknown> {
  return sanitizeOgValue(obj, 0) as Record<string, unknown>;
}

export type OgNodeCategory = 'input' | 'generation' | 'sequence' | 'prediction' | 'evaluation';

export interface OgNode {
  id: string;
  adapterId: string;
  label: string;
  category: OgNodeCategory;
  paramChip: string;
  params: Record<string, unknown>;
  position?: { x: number; y: number };
}

export interface OgEdge {
  fromNode: string;
  fromPort: string;
  toNode: string;
  toPort: string;
}

export type OgStructureTarget =
  | { kind: 'rcsb'; pdbId: string; cropSegments: string }
  | { kind: 'afdb'; uniprotId: string; cropSegments: string };

export interface DecodedOgWorkflow {
  title: string;
  description: string;
  nodes: OgNode[];
  edges: OgEdge[];
  structureTarget: OgStructureTarget | null;
}

interface AdapterMeta {
  shortName: string;
  category: OgNodeCategory;
}

const ADAPTER_META: Record<string, AdapterMeta> = {
  'pdb-input': { shortName: 'Structure Input', category: 'input' },
  'sequence-input': { shortName: 'Sequence Input', category: 'input' },
  'rfd4-proteina': { shortName: 'RFD4-Proteina', category: 'generation' },
  genie3: { shortName: 'Genie 3', category: 'generation' },
  protpardelle: { shortName: 'Protpardelle', category: 'generation' },
  rfdiffusion: { shortName: 'RFdiffusion', category: 'generation' },
  rfd3: { shortName: 'RFdiffusion3', category: 'generation' },
  boltzgen: { shortName: 'BoltzGen', category: 'generation' },
  evodiff: { shortName: 'EvoDiff', category: 'generation' },
  'la-proteina': { shortName: 'La-Proteina', category: 'generation' },
  proteina: { shortName: 'Proteina', category: 'generation' },
  'sid-protein': { shortName: 'SiD-Protein', category: 'generation' },
  genie2: { shortName: 'Genie 2', category: 'generation' },
  frameflow: { shortName: 'FrameFlow', category: 'generation' },
  foldflow2: { shortName: 'FoldFlow 2', category: 'generation' },
  'web-protein-mpnn': { shortName: 'ProteinMPNN', category: 'sequence' },
  ligandmpnn: { shortName: 'LigandMPNN', category: 'sequence' },
  lasermpnn: { shortName: 'LASErMPNN', category: 'sequence' },
  'antifold-esmif1': { shortName: 'AntiFold / ESM-IF1', category: 'sequence' },
  'localfold-af2': { shortName: 'LocalFold', category: 'prediction' },
  thermompnn: { shortName: 'ThermoMPNN', category: 'evaluation' },
  esmc: { shortName: 'ESM-C 300M', category: 'evaluation' },
  dyna1: { shortName: 'Dyna-1', category: 'evaluation' },
  gnina: { shortName: 'GNINA', category: 'evaluation' },
  cg2all: { shortName: 'cg2all', category: 'evaluation' },
  'top-k-filter': { shortName: 'Top-K Filter', category: 'evaluation' },
  'designability-scorer': { shortName: 'scRMSD / scTM', category: 'evaluation' },
  'structure-compare': { shortName: '3D Superpose', category: 'evaluation' },
};

export const CATEGORY_COLORS: Record<
  OgNodeCategory,
  { accent: string; badgeBg: string; badgeText: string; label: string }
> = {
  input: { accent: '#0d9488', badgeBg: '#ccfbf1', badgeText: '#0f766e', label: 'INPUT' },
  generation: { accent: '#2563eb', badgeBg: '#dbeafe', badgeText: '#1d4ed8', label: 'GENERATION' },
  sequence: { accent: '#7c3aed', badgeBg: '#ede9fe', badgeText: '#6d28d9', label: 'SEQUENCE DESIGN' },
  prediction: { accent: '#059669', badgeBg: '#d1fae5', badgeText: '#047857', label: 'PREDICTION' },
  evaluation: { accent: '#d97706', badgeBg: '#fef3c7', badgeText: '#b45309', label: 'EVALUATION' },
};

function formatCompactParamChip(adapterId: string, params: Record<string, unknown>): string {
  const str = (k: string, fb = '') => {
    if (!Object.hasOwn(params, k)) return fb;
    const v = params[k];
    if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
      return String(v).trim();
    }
    return fb;
  };
  const num = (k: string, fb: number) => {
    if (!Object.hasOwn(params, k)) return fb;
    const raw = params[k];
    const n =
      typeof raw === 'number'
        ? raw
        : typeof raw === 'string' && raw.trim() !== ''
          ? Number(raw)
          : NaN;
    return Number.isFinite(n) ? n : fb;
  };
  switch (adapterId) {
    case 'pdb-input': {
      const mode = str('mode', 'demo-bundle');
      const crop = str('cropSegments');
      const cropTag = crop ? ` · ${crop}` : '';
      if (mode === 'pdb-id') return `RCSB ${str('pdbId', '1LVM').toUpperCase()}${cropTag}`;
      if (mode === 'alphafold-db') return `AFDB ${str('uniprotId', 'P69905').toUpperCase()}${cropTag}`;
      if (mode === 'custom') return `${str('filename', 'Custom PDB')}${cropTag}`;
      return 'Demo Helix Bundle';
    }
    case 'sequence-input': {
      const seq = str('sequence').replace(/[^A-Za-z]/g, '');
      return seq ? `${seq.length} aa sequence` : 'FASTA Sequence';
    }
    case 'web-protein-mpnn':
    case 'ligandmpnn':
    case 'lasermpnn':
    case 'antifold-esmif1': {
      const n = num('numSequences', 2);
      const t = num('temperature', 0.1);
      return `${n} seq${n === 1 ? '' : 's'} · T=${t}`;
    }
    case 'localfold-af2': {
      const rec = num('recycles', 1);
      const capMode = str('capMode', 'follow-upstream');
      const maxSeq = num('maxSequences', 2);
      const capTag = capMode === 'fixed' ? `Cap ${maxSeq}` : 'Auto (≤8)';
      return `${rec} rec · ${capTag}`;
    }
    case 'genie3':
    case 'rfd4-proteina':
    case 'rfdiffusion':
    case 'rfd3':
    case 'protpardelle':
    case 'boltzgen':
    case 'frameflow':
    case 'foldflow2':
    case 'la-proteina':
    case 'proteina':
    case 'genie2': {
      const len = num('length', 80);
      const n = num('numDesigns', 1);
      return `${len} aa${n > 1 ? ` · ×${n}` : ''}`;
    }
    case 'designability-scorer':
      return 'scRMSD < 2.0 Å · pLDDT ≥ 70';
    case 'esmc':
      return str('mode', 'wildtype') === 'masked' ? 'Masked LLR' : 'Wildtype LLR';
    case 'thermompnn':
      return 'Site-saturation ΔΔG';
    case 'dyna1':
      return 'Conformational dynamics';
    case 'gnina':
      return str('taskMode', 'score') === 'dock' ? 'CNN Docking' : 'Pose Scoring';
    case 'cg2all':
      return str('variant', 'ca') === 'ca-fix' ? 'Fixed-Cα all-atom' : 'Cα → all-atom';
    case 'top-k-filter':
      return `Top-${num('topK', 2)} filter`;
    default:
      return resolveOgAdapterMeta(adapterId).shortName;
  }
}

function fromBase64Url(str: string): Uint8Array {
  const b64 = str.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((str.length + 3) % 4);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function inflateRawEdge(bytes: Uint8Array): Promise<string> {
  if (bytes.byteLength > 128 * 1024) throw new Error('Payload too large');
  const ds = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  const writePromise = (async () => {
    await writer.write(bytes.slice());
    await writer.close();
  })();
  writePromise.catch(() => {});

  const reader = ds.readable.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) {
        total += value.byteLength;
        if (total > 256 * 1024) {
          await reader.cancel('Decompressed payload too large').catch(() => {});
          throw new Error('Decompressed payload too large');
        }
        chunks.push(value);
      }
    }
  } finally {
    reader.releaseLock();
  }
  await writePromise;
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return new TextDecoder().decode(out);
}

function extractStructureTarget(nodes: OgNode[]): OgStructureTarget | null {
  for (const n of nodes) {
    if (n.adapterId !== 'pdb-input') continue;
    const mode = safeStr(n.params.mode ?? 'demo-bundle') || 'demo-bundle';
    const cropSegments = safeStr(n.params.cropSegments).trim();
    if (mode === 'pdb-id') {
      const pdbId = safeStr(n.params.pdbId).trim().toUpperCase();
      if (/^[0-9A-Z]{4}$/.test(pdbId)) {
        return { kind: 'rcsb', pdbId, cropSegments };
      }
    } else if (mode === 'alphafold-db') {
      const uniprotId = safeStr(n.params.uniprotId).trim().toUpperCase();
      if (/^[A-Z0-9]{6,10}$/.test(uniprotId)) {
        return { kind: 'afdb', uniprotId, cropSegments };
      }
    }
  }
  return null;
}

function buildTitleAndDesc(
  nodes: OgNode[],
  presetName?: string,
): { title: string; description: string } {
  if (nodes.length === 0) {
    return {
      title: 'denovo.run — Interactive WebGPU Molecular Design Workspace',
      description:
        'Have a molecular design idea you want to quickly prototype in your browser? Interactive browser-native WebGPU workspace for structure, design, folding & docking.',
    };
  }
  if (nodes.length === 1) {
    const n = nodes[0];
    let cleanChip = n.paramChip;
    if (cleanChip.toLowerCase().startsWith(n.label.toLowerCase())) {
      cleanChip = cleanChip.slice(n.label.length).replace(/^\s*[·:\-]\s*/, '').trim();
    }
    const chip = cleanChip && cleanChip !== n.label ? ` (${cleanChip})` : '';
    return {
      title: `${n.label}${chip} — denovo.run`,
      description: `Interactive browser-native WebGPU molecular design workspace · ${n.label}${chip}.`,
    };
  }
  const chainNames = nodes
    .slice(0, 4)
    .map((n) => n.label)
    .join(' → ');
  const suffix = nodes.length > 4 ? ` (+${nodes.length - 4})` : '';
  const baseTitle = presetName || `${chainNames}${suffix}`;
  const summaryParts = nodes
    .slice(0, 4)
    .map((n) => (n.paramChip ? `${n.label} (${n.paramChip})` : n.label));
  return {
    title: `${baseTitle} — denovo.run`,
    description: `${nodes.length}-block WebGPU molecular design workflow: ${summaryParts.join(' → ')}. Prototype locally in your browser.`,
  };
}

function stageToCategory(stage: string): OgNodeCategory {
  switch (stage) {
    case 'source':
      return 'input';
    case 'generate':
      return 'generation';
    case 'inverse-fold':
      return 'sequence';
    case 'fold':
      return 'prediction';
    case 'condition':
    case 'score':
    default:
      return 'evaluation';
  }
}

function resolveOgAdapterMeta(
  adapterId: string,
  fallbackLabel?: string,
  fallbackStage?: string,
): AdapterMeta {
  if (Object.hasOwn(ADAPTER_META, adapterId)) {
    return ADAPTER_META[adapterId];
  }
  const manifest = Object.hasOwn(ADAPTER_REGISTRY, adapterId)
    ? ADAPTER_REGISTRY[adapterId]
    : undefined;
  return {
    shortName: fallbackLabel ?? manifest?.displayName ?? adapterId,
    category: stageToCategory(fallbackStage ?? manifest?.stage ?? 'score'),
  };
}

/**
 * Decode a `?flow=...`, `?recipe=...`, `?g=...`, or `?p=...` (or `#...` / full URL / hybrid `?utm_...#flow=`)
 * share query string into lightweight edge OG metadata and graph structure.
 */
export async function decodeOgWorkflowFromQuery(
  rawQueryOrHash: string,
): Promise<DecodedOgWorkflow | null> {
  try {
    const { clean, invalidUrl } = extractWorkflowSearchOrHash(rawQueryOrHash);
    if (invalidUrl || !clean || clean.length > MAX_HASH_CHARS) return null;

    const qs = new URLSearchParams(clean);

    if (qs.has('flow') || qs.has('recipe') || qs.has('p')) {
      const res = await decodeWorkflowQuery(clean);
      if (!res.ok || !res.graph || res.graph.nodes.length === 0) return null;

      const summaryById = new Map(res.nodes.map((s) => [s.id, s]));
      const nodes: OgNode[] = res.graph.nodes.map((n) => {
        const summary = summaryById.get(n.id);
        const meta = resolveOgAdapterMeta(n.adapterId, summary?.label, summary?.stage);
        const label = typeof n.label === 'string' && n.label.trim() ? n.label.trim() : meta.shortName;
        const params = { ...n.params };
        return {
          id: n.id,
          adapterId: n.adapterId,
          label,
          category: meta.category,
          paramChip: formatCompactParamChip(n.adapterId, params),
          params,
          ...(n.position ? { position: n.position } : {}),
        };
      });

      const edges: OgEdge[] = res.graph.edges.map((e) => ({
        fromNode: e.fromNode,
        fromPort: e.fromPort,
        toNode: e.toNode,
        toPort: e.toPort,
      }));

      const customTitle =
        res.graph.name && res.graph.name !== 'Shared Workflow' && res.graph.name !== 'Custom Recipe'
          ? res.graph.name
          : undefined;
      const { title, description } = buildTitleAndDesc(nodes, customTitle);
      return {
        title,
        description,
        nodes,
        edges,
        structureTarget: extractStructureTarget(nodes),
      };
    }

    const gVal = qs.get('g');

    if (gVal) {
      const normalized = gVal
        .replace(/%5[fF]/g, '_')
        .replace(/%2[dD]/g, '-')
        .replace(/\\_/g, '_')
        .replace(/[.,;:!?)\]}>'"]+$/, '');

      let parsed: { v?: number; n: unknown[]; e: unknown[]; p?: unknown } | null = null;
      for (const candidate of [normalized, normalized + '_', normalized + '-', normalized + '__']) {
        try {
          const rawBytes = fromBase64Url(candidate);
          let text: string;
          try {
            text = await inflateRawEdge(rawBytes);
          } catch {
            text = new TextDecoder().decode(rawBytes);
          }
          const obj = JSON.parse(text);
          if (obj && typeof obj === 'object' && Array.isArray(obj.n) && Array.isArray(obj.e)) {
            parsed = obj;
            break;
          }
        } catch {
          /* try next */
        }
      }
      if (
        !parsed ||
        parsed.n.length === 0 ||
        parsed.n.length > MAX_IMPORT_NODES ||
        parsed.e.length > MAX_IMPORT_EDGES
      ) {
        return null;
      }

      const isLegacyV1 = !parsed.v || parsed.v < 2;
      const rawPos = Array.isArray(parsed.p) ? parsed.p : null;
      const nodes: OgNode[] = [];
      for (let i = 0; i < parsed.n.length; i++) {
        const tuple = parsed.n[i];
        if (!Array.isArray(tuple) || tuple.length < 2) return null;
        if (typeof tuple[0] !== 'string' || typeof tuple[1] !== 'string') return null;
        const id = tuple[0];
        const shortA = tuple[1];
        if (!id || FORBIDDEN_KEYS.has(id) || !shortA || FORBIDDEN_KEYS.has(shortA)) {
          return null;
        }
        const adapterId = Object.hasOwn(SHORT_TO_ADAPTER, shortA)
          ? SHORT_TO_ADAPTER[shortA]
          : shortA;
        if (FORBIDDEN_KEYS.has(adapterId)) return null;
        const meta = resolveOgAdapterMeta(adapterId);
        const params: Record<string, unknown> =
          tuple[2] && typeof tuple[2] === 'object' && !Array.isArray(tuple[2])
            ? sanitizeOgParams(tuple[2] as Record<string, unknown>)
            : {};
        if (
          isLegacyV1 &&
          (adapterId === 'localfold-af2' || adapterId === 'esmc' || adapterId === 'dyna1') &&
          !('capMode' in params)
        ) {
          params.capMode = 'fixed';
        }
        const label =
          typeof tuple[3] === 'string' && tuple[3].trim() ? tuple[3].trim() : meta.shortName;
        const posTuple = rawPos?.[i];
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
          category: meta.category,
          paramChip: formatCompactParamChip(adapterId, params),
          params,
          ...(position ? { position } : {}),
        });
      }

      const edges: OgEdge[] = [];
      for (const rawE of parsed.e) {
        if (!Array.isArray(rawE)) continue;
        if (rawE.length >= 4) {
          if (
            typeof rawE[0] !== 'string' ||
            typeof rawE[2] !== 'string' ||
            (rawE[1] !== undefined && rawE[1] !== null && typeof rawE[1] !== 'string') ||
            (rawE[3] !== undefined && rawE[3] !== null && typeof rawE[3] !== 'string')
          ) {
            return null;
          }
          const fromNode = rawE[0].trim();
          const fromPort = (typeof rawE[1] === 'string' ? rawE[1] : 'out').trim();
          const toNode = rawE[2].trim();
          const toPort = (typeof rawE[3] === 'string' ? rawE[3] : 'in').trim();
          if (fromNode && toNode) edges.push({ fromNode, fromPort, toNode, toPort });
        } else if (rawE.length >= 2) {
          const [src, dst] = rawE;
          if (typeof src !== 'string' || typeof dst !== 'string') continue;
          const [fromNode, fromPort = 'out'] = src.split('.');
          const [toNode, toPort = 'in'] = dst.split('.');
          if (fromNode && toNode) edges.push({ fromNode, fromPort, toNode, toPort });
        }
      }

      const { title, description } = buildTitleAndDesc(nodes);
      return {
        title,
        description,
        nodes,
        edges,
        structureTarget: extractStructureTarget(nodes),
      };
    }

    return null;
  } catch {
    return null;
  }
}
