// Lightweight, zero-Svelte workflow decoder & metadata extractor for Cloudflare Pages Edge OG previews.

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

const SHORT_TO_ADAPTER: Record<string, string> = {
  p: 'pdb-input',
  q: 'sequence-input',
  r: 'rfd4-proteina',
  g: 'genie3',
  pp: 'protpardelle',
  d: 'rfdiffusion',
  r3: 'rfd3',
  b: 'boltzgen',
  e: 'evodiff',
  lp: 'la-proteina',
  pr: 'proteina',
  sp: 'sid-protein',
  g2: 'genie2',
  ff: 'frameflow',
  f2: 'foldflow2',
  m: 'web-protein-mpnn',
  lm: 'ligandmpnn',
  la: 'lasermpnn',
  af: 'antifold-esmif1',
  f: 'localfold-af2',
  tm: 'thermompnn',
  ec: 'esmc',
  dy: 'dyna1',
  gn: 'gnina',
  tk: 'top-k-filter',
  s: 'designability-scorer',
  c: 'structure-compare',
};

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

interface PresetDef {
  name: string;
  nodes: Array<{ id: string; adapterId: string; label: string; params: Record<string, unknown> }>;
  edges: OgEdge[];
}

const PRESET_SNAPSHOTS: Record<string, PresetDef> = {
  'genie3-mpnn-af2': {
    name: 'Genie 3 → Cα-ProteinMPNN → LocalFold',
    nodes: [
      { id: 'gen', adapterId: 'genie3', label: 'Genie 3', params: { length: 80, nSampleStep: 100 } },
      { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'Cα-ProteinMPNN', params: { family: 'ca', numSequences: 2 } },
      { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { capMode: 'follow-upstream', recycles: 1 } },
      { id: 'score', adapterId: 'designability-scorer', label: 'scRMSD / scTM', params: {} },
    ],
    edges: [
      { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
      { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
      { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
    ],
  },
  'rfd4-refold-native': {
    name: 'RFD4-Proteina → LocalFold → scRMSD',
    nodes: [
      { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4-Proteina', params: { length: 80, numSteps: 50 } },
      { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { capMode: 'follow-upstream', recycles: 1 } },
      { id: 'score', adapterId: 'designability-scorer', label: 'scRMSD / scTM', params: {} },
    ],
    edges: [
      { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
      { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
    ],
  },
  'pdb-refold-qc': {
    name: 'Backbone Input → SolubleMPNN → LocalFold',
    nodes: [
      { id: 'src', adapterId: 'pdb-input', label: 'Backbone Input', params: { mode: 'demo-bundle' } },
      { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'SolubleMPNN', params: { family: 'soluble', numSequences: 2 } },
      { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { capMode: 'follow-upstream', recycles: 1 } },
      { id: 'score', adapterId: 'designability-scorer', label: 'scRMSD / scTM', params: {} },
    ],
    edges: [
      { fromNode: 'src', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
      { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      { fromNode: 'src', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
      { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
    ],
  },
  'rfd4-mpnn-refold': {
    name: 'RFD4-Proteina → SolubleMPNN → LocalFold',
    nodes: [
      { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4-Proteina', params: { length: 80, numSteps: 50 } },
      { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'SolubleMPNN', params: { family: 'soluble', numSequences: 2 } },
      { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { capMode: 'follow-upstream', recycles: 1 } },
      { id: 'score', adapterId: 'designability-scorer', label: 'scRMSD / scTM', params: {} },
    ],
    edges: [
      { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
      { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
      { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
    ],
  },
  'rfd4-branch-compare': {
    name: 'RFD4-Proteina: SolubleMPNN vs Native',
    nodes: [
      { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4-Proteina', params: { length: 80 } },
      { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'SolubleMPNN', params: { numSequences: 2 } },
      { id: 'fold_mpnn', adapterId: 'localfold-af2', label: 'AF2 (MPNN)', params: { capMode: 'follow-upstream' } },
      { id: 'score_mpnn', adapterId: 'designability-scorer', label: 'Score (MPNN)', params: {} },
      { id: 'fold_native', adapterId: 'localfold-af2', label: 'AF2 (Native)', params: { capMode: 'follow-upstream' } },
      { id: 'score_native', adapterId: 'designability-scorer', label: 'Score (Native)', params: {} },
    ],
    edges: [
      { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
      { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold_mpnn', toPort: 'sequences' },
      { fromNode: 'fold_mpnn', fromPort: 'folds', toNode: 'score_mpnn', toPort: 'folds' },
      { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold_native', toPort: 'sequences' },
      { fromNode: 'fold_native', fromPort: 'folds', toNode: 'score_native', toPort: 'folds' },
    ],
  },
  'rfd4-conditioned-cath': {
    name: 'CATH-Conditioned RFD4-Proteina → LocalFold',
    nodes: [
      { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4-Proteina', params: { length: 80, cathCode: '1.10.287' } },
      { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { capMode: 'follow-upstream' } },
      { id: 'score', adapterId: 'designability-scorer', label: 'scRMSD / scTM', params: {} },
    ],
    edges: [
      { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
      { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
    ],
  },
  'rfd4-af2-vs-rf3': {
    name: 'RFD4-Proteina: AF2 vs RoseTTAFold 3',
    nodes: [
      { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4-Proteina', params: { length: 80 } },
      { id: 'fold_af2', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { capMode: 'follow-upstream' } },
      { id: 'score_af2', adapterId: 'designability-scorer', label: 'Score (AF2)', params: {} },
      { id: 'fold_rf3', adapterId: 'localfold-af2', label: 'RoseTTAFold 3', params: { model: 'rf3', capMode: 'follow-upstream' } },
      { id: 'score_rf3', adapterId: 'designability-scorer', label: 'Score (RF3)', params: {} },
    ],
    edges: [
      { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold_af2', toPort: 'sequences' },
      { fromNode: 'fold_af2', fromPort: 'folds', toNode: 'score_af2', toPort: 'folds' },
      { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold_rf3', toPort: 'sequences' },
      { fromNode: 'fold_rf3', fromPort: 'folds', toNode: 'score_rf3', toPort: 'folds' },
    ],
  },
  'pdb-thermompnn': {
    name: 'Backbone Input → ThermoMPNN ΔΔG',
    nodes: [
      { id: 'src', adapterId: 'pdb-input', label: 'Backbone Input', params: { mode: 'demo-bundle' } },
      { id: 'thermo', adapterId: 'thermompnn', label: 'ThermoMPNN', params: {} },
    ],
    edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'thermo', toPort: 'structure' }],
  },
};

function formatCompactParamChip(adapterId: string, params: Record<string, unknown>): string {
  const str = (k: string, fb = '') => (params[k] !== undefined && params[k] !== null ? String(params[k]).trim() : fb);
  const num = (k: string, fb: number) => {
    const n = Number(params[k]);
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
    case 'top-k-filter':
      return `Top-${num('topK', 2)} filter`;
    default:
      return ADAPTER_META[adapterId]?.shortName ?? adapterId;
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
        if (total > 256 * 1024) throw new Error('Decompressed payload too large');
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
    const mode = String(n.params.mode ?? 'demo-bundle');
    const cropSegments = String(n.params.cropSegments ?? '').trim();
    if (mode === 'pdb-id') {
      const pdbId = String(n.params.pdbId ?? '').trim().toUpperCase();
      if (/^[0-9A-Z]{4}$/.test(pdbId)) {
        return { kind: 'rcsb', pdbId, cropSegments };
      }
    } else if (mode === 'alphafold-db') {
      const uniprotId = String(n.params.uniprotId ?? '').trim().toUpperCase();
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

/**
 * Decode a `?g=...` / `?p=...` (or `#g=...` / `#p=...`) share query string into
 * lightweight edge OG metadata and graph structure.
 */
export async function decodeOgWorkflowFromQuery(
  rawQueryOrHash: string,
): Promise<DecodedOgWorkflow | null> {
  const clean = rawQueryOrHash.replace(/^[#?]/, '').trim();
  if (!clean || clean.length > 32_768) return null;

  const qs = new URLSearchParams(clean);
  const pVal = qs.get('p');
  const gVal = qs.get('g');

  if (pVal) {
    const preset = PRESET_SNAPSHOTS[pVal];
    if (!preset) return null;
    const nodeMap = new Map<string, OgNode>();
    const nodes: OgNode[] = preset.nodes.map((pn) => {
      const meta = ADAPTER_META[pn.adapterId] ?? { shortName: pn.adapterId, category: 'evaluation' as const };
      const params = { ...pn.params };
      const node: OgNode = {
        id: pn.id,
        adapterId: pn.adapterId,
        label: pn.label || meta.shortName,
        category: meta.category,
        paramChip: '',
        params,
      };
      nodeMap.set(pn.id, node);
      return node;
    });
    for (const [k, v] of qs.entries()) {
      if (k === 'p' || k === 'v') continue;
      const dot = k.lastIndexOf('.');
      if (dot <= 0) continue;
      const node = nodeMap.get(k.slice(0, dot));
      const paramKey = k.slice(dot + 1);
      if (!node || !paramKey || paramKey === '__proto__' || paramKey === 'constructor') continue;
      if (paramKey === '_label' && v.trim()) {
        node.label = v.trim();
      } else if (paramKey === '_pos') {
        const [xs, ys] = v.split(',');
        const x = Number(xs);
        const y = Number(ys);
        if (Number.isFinite(x) && Number.isFinite(y)) node.position = { x, y };
      } else {
        node.params[paramKey] = v;
      }
    }
    for (const n of nodes) {
      n.paramChip = formatCompactParamChip(n.adapterId, n.params);
    }
    const { title, description } = buildTitleAndDesc(nodes, preset.name);
    return {
      title,
      description,
      nodes,
      edges: preset.edges.map((e) => ({ ...e })),
      structureTarget: extractStructureTarget(nodes),
    };
  }

  if (gVal) {
    const normalized = gVal
      .replace(/%5[fF]/g, '_')
      .replace(/%2[dD]/g, '-')
      .replace(/\\_/g, '_')
      .replace(/[.,;:!?)\]}>'"]+$/, '');

    let parsed: { v?: number; n: unknown[]; e: unknown[]; p?: unknown } | null = null;
    for (const candidate of [normalized, normalized + '_', normalized + '-']) {
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
    if (!parsed || parsed.n.length === 0 || parsed.n.length > 64) return null;

    const isLegacyV1 = !parsed.v || parsed.v < 2;
    const rawPos = Array.isArray(parsed.p) ? parsed.p : null;
    const nodes: OgNode[] = [];
    for (let i = 0; i < parsed.n.length; i++) {
      const tuple = parsed.n[i];
      if (!Array.isArray(tuple) || tuple.length < 2) return null;
      const id = String(tuple[0]);
      const shortA = String(tuple[1]);
      const adapterId = SHORT_TO_ADAPTER[shortA] ?? shortA;
      const meta = ADAPTER_META[adapterId] ?? { shortName: adapterId, category: 'evaluation' as const };
      const params: Record<string, unknown> =
        tuple[2] && typeof tuple[2] === 'object' && !Array.isArray(tuple[2])
          ? { ...(tuple[2] as Record<string, unknown>) }
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
        Number.isFinite(Number(posTuple[0])) &&
        Number.isFinite(Number(posTuple[1]))
          ? { x: Number(posTuple[0]), y: Number(posTuple[1]) }
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
        const fromNode = String(rawE[0] ?? '').trim();
        const fromPort = String(rawE[1] ?? 'out').trim();
        const toNode = String(rawE[2] ?? '').trim();
        const toPort = String(rawE[3] ?? 'in').trim();
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
}
