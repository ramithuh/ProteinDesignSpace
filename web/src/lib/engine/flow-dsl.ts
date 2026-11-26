// Pure-TS, edge-safe Markdown-friendly workflow DSL (?flow=), YAML/JSON recipe fallback (?recipe=),
// and compact share helpers (?p= / ?g=).
// STRICT IMPORT RULE: Relative imports ONLY (./registry, ./param-schema, and `import type` from ./types).
// ZERO $lib/ runtime imports, ZERO Svelte imports, ZERO Node builtins so tools/build-cf-worker.mjs
// bundles this cleanly with `esbuild --platform=browser`.

import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS, WORKFLOW_PRESETS } from './registry';
import { getDefaultParamsFromSchema, validateParamsWithSchema } from './param-schema';
import type {
  AdapterManifest,
  PortSpec,
  PortType,
  WorkflowEdge,
  WorkflowGraph,
  WorkflowNode,
} from './types';

export const MAX_HASH_CHARS = 65_536;
export const MAX_INFLATED_BYTES = 512 * 1024;
export const MAX_IMPORT_NODES = 64;
export const MAX_IMPORT_EDGES = 128;
export const MAX_ERROR_TOKEN_CHARS = 160;
export const MAX_ERROR_MESSAGE_CHARS = 512;
const MAX_MULTI_INPUT_EDGES = 6;

const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

function safeStr(v: unknown): string {
  return typeof v === 'string'
    ? v
    : typeof v === 'number' || typeof v === 'boolean'
      ? String(v)
      : '';
}

export function truncateErrorField(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  if (maxLen <= 1) return s.slice(0, maxLen);
  return `${s.slice(0, maxLen - 1)}…`;
}

function sanitizeFlowValidationError(err: FlowValidationError): FlowValidationError {
  return {
    ...err,
    message: truncateErrorField(err.message, MAX_ERROR_MESSAGE_CHARS),
    ...(err.token !== undefined ? { token: truncateErrorField(err.token, MAX_ERROR_TOKEN_CHARS) } : {}),
    ...(err.nodeId !== undefined ? { nodeId: truncateErrorField(err.nodeId, MAX_ERROR_TOKEN_CHARS) } : {}),
  };
}

function sanitizeFlowErrors(errors: FlowValidationError[]): FlowValidationError[] {
  return errors.map(sanitizeFlowValidationError);
}

/** Immutable 1-2 char short codes matching `share.ts` and `share-roundtrip-golden.test.ts`. */
export const ADAPTER_TO_SHORT: Record<string, string> = {
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

export const SHORT_TO_ADAPTER: Record<string, string> = Object.fromEntries(
  Object.entries(ADAPTER_TO_SHORT).map(([k, v]) => [v, k]),
);

export const LOCALFOLD_DEFAULT_BASELINE_KEYS = new Set([
  'model',
  'recycles',
  'diffusionSteps',
  'capMode',
  'maxSequences',
  'seed',
]);

export const LOCALFOLD_MODEL_ALIASES: Record<string, string> = {
  af2: 'monomer',
  alphafold2: 'monomer',
  'af2-1': 'monomer',
  'af2-2': 'monomer-2',
  'af2-3': 'monomer-3',
  'af2-4': 'monomer-4',
  'af2-5': 'monomer-5',
  'af2-multimer': 'multimer',
  'alphafold2-multimer': 'multimer',
  rf3: 'rosettafold3',
  'rosettafold-3': 'rosettafold3',
  'boltz-2': 'boltz2',
  'openbind-0': 'openbind0',
  openfold3: 'openbind0',
  'protenix-v2': 'protenix2',
  'mol-1': 'mol1',
  'intellifold-2': 'intellifold2',
  alphafold3: 'af3',
  esmfold2: 'ef2-fast-300m',
  ef2: 'ef2-fast-300m',
  'ef2-300m': 'ef2-fast-300m',
  'ef2-600m': 'ef2-fast-600m',
};

export interface FlowAliasResolution {
  adapterId: string;
  presetParams?: Record<string, unknown>;
}

/** Human/LLM-friendly model aliases -> canonical adapterId (+ optional preset params). */
export const FLOW_MODEL_ALIASES: Record<string, FlowAliasResolution> = {
  // Inputs
  pdb: { adapterId: 'pdb-input' },
  'structure-input': { adapterId: 'pdb-input' },
  pdbinput: { adapterId: 'pdb-input' },
  seq: { adapterId: 'sequence-input' },
  sequence: { adapterId: 'sequence-input' },
  fasta: { adapterId: 'sequence-input' },
  sequenceinput: { adapterId: 'sequence-input' },
  // Generators
  rfd4: { adapterId: 'rfd4-proteina' },
  r4: { adapterId: 'rfd4-proteina' },
  'proteina-rfd4': { adapterId: 'rfd4-proteina' },
  rfd4proteina: { adapterId: 'rfd4-proteina' },
  g3: { adapterId: 'genie3' },
  'genie-3': { adapterId: 'genie3' },
  'protpardelle-1c': { adapterId: 'protpardelle' },
  pp1c: { adapterId: 'protpardelle' },
  rfd: { adapterId: 'rfdiffusion' },
  rfd1: { adapterId: 'rfdiffusion' },
  rfdiffusion1: { adapterId: 'rfdiffusion' },
  rfdiffusion3: { adapterId: 'rfd3' },
  'rfdiffusion-3': { adapterId: 'rfd3' },
  bg: { adapterId: 'boltzgen' },
  'boltzgen-1': { adapterId: 'boltzgen' },
  evo: { adapterId: 'evodiff' },
  'evodiff-38m': { adapterId: 'evodiff' },
  laproteina: { adapterId: 'la-proteina' },
  'proteina-200m': { adapterId: 'proteina' },
  sid: { adapterId: 'sid-protein' },
  sidprotein: { adapterId: 'sid-protein' },
  g2: { adapterId: 'genie2' },
  'genie-2': { adapterId: 'genie2' },
  'frame-flow': { adapterId: 'frameflow' },
  'foldflow-2': { adapterId: 'foldflow2' },
  'fold-flow-2': { adapterId: 'foldflow2' },
  // Inverse folders
  mpnn: { adapterId: 'web-protein-mpnn' },
  proteinmpnn: { adapterId: 'web-protein-mpnn', presetParams: { family: 'proteinmpnn' } },
  'protein-mpnn': { adapterId: 'web-protein-mpnn', presetParams: { family: 'proteinmpnn' } },
  solublempnn: { adapterId: 'web-protein-mpnn', presetParams: { family: 'solublempnn' } },
  'soluble-mpnn': { adapterId: 'web-protein-mpnn', presetParams: { family: 'solublempnn' } },
  campnn: { adapterId: 'web-protein-mpnn', presetParams: { family: 'ca' } },
  'ca-mpnn': { adapterId: 'web-protein-mpnn', presetParams: { family: 'ca' } },
  'ca-proteinmpnn': { adapterId: 'web-protein-mpnn', presetParams: { family: 'ca' } },
  'ligand-mpnn': { adapterId: 'ligandmpnn' },
  'laser-mpnn': { adapterId: 'lasermpnn' },
  antifold: { adapterId: 'antifold-esmif1', presetParams: { variant: 'antifold' } },
  esmif1: { adapterId: 'antifold-esmif1', presetParams: { variant: 'esmif1' } },
  'esm-if1': { adapterId: 'antifold-esmif1', presetParams: { variant: 'esmif1' } },
  // Folders (LocalFold checkpoints)
  localfold: { adapterId: 'localfold-af2' },
  fold: { adapterId: 'localfold-af2' },
  af2: { adapterId: 'localfold-af2', presetParams: { model: 'monomer' } },
  alphafold2: { adapterId: 'localfold-af2', presetParams: { model: 'monomer' } },
  'af2-multimer': { adapterId: 'localfold-af2', presetParams: { model: 'multimer' } },
  'alphafold2-multimer': { adapterId: 'localfold-af2', presetParams: { model: 'multimer' } },
  multimer: { adapterId: 'localfold-af2', presetParams: { model: 'multimer' } },
  rf3: { adapterId: 'localfold-af2', presetParams: { model: 'rosettafold3' } },
  rosettafold3: { adapterId: 'localfold-af2', presetParams: { model: 'rosettafold3' } },
  'rosettafold-3': { adapterId: 'localfold-af2', presetParams: { model: 'rosettafold3' } },
  boltz2: { adapterId: 'localfold-af2', presetParams: { model: 'boltz2' } },
  'boltz-2': { adapterId: 'localfold-af2', presetParams: { model: 'boltz2' } },
  openbind0: { adapterId: 'localfold-af2', presetParams: { model: 'openbind0' } },
  'openbind-0': { adapterId: 'localfold-af2', presetParams: { model: 'openbind0' } },
  openfold3: { adapterId: 'localfold-af2', presetParams: { model: 'openbind0' } },
  protenix2: { adapterId: 'localfold-af2', presetParams: { model: 'protenix2' } },
  'protenix-v2': { adapterId: 'localfold-af2', presetParams: { model: 'protenix2' } },
  opendde: { adapterId: 'localfold-af2', presetParams: { model: 'opendde' } },
  mol1: { adapterId: 'localfold-af2', presetParams: { model: 'mol1' } },
  'mol-1': { adapterId: 'localfold-af2', presetParams: { model: 'mol1' } },
  intellifold2: { adapterId: 'localfold-af2', presetParams: { model: 'intellifold2' } },
  'intellifold-2': { adapterId: 'localfold-af2', presetParams: { model: 'intellifold2' } },
  af3: { adapterId: 'localfold-af2', presetParams: { model: 'af3' } },
  alphafold3: { adapterId: 'localfold-af2', presetParams: { model: 'af3' } },
  'ef2-fast-300m': { adapterId: 'localfold-af2', presetParams: { model: 'ef2-fast-300m' } },
  'ef2-fast-600m': { adapterId: 'localfold-af2', presetParams: { model: 'ef2-fast-600m' } },
  esmfold2: { adapterId: 'localfold-af2', presetParams: { model: 'ef2-fast-300m' } },
  ef2: { adapterId: 'localfold-af2', presetParams: { model: 'ef2-fast-300m' } },
  'ef2-300m': { adapterId: 'localfold-af2', presetParams: { model: 'ef2-fast-300m' } },
  'ef2-600m': { adapterId: 'localfold-af2', presetParams: { model: 'ef2-fast-600m' } },
  // Evaluators
  'thermo-mpnn': { adapterId: 'thermompnn' },
  'esm-c': { adapterId: 'esmc' },
  'esmc-300m': { adapterId: 'esmc' },
  'dyna-1': { adapterId: 'dyna1' },
  'gnina1.3': { adapterId: 'gnina' },
  'gnina-1.3': { adapterId: 'gnina' },
  topk: { adapterId: 'top-k-filter' },
  'top-k': { adapterId: 'top-k-filter' },
  topkfilter: { adapterId: 'top-k-filter' },
  scorer: { adapterId: 'designability-scorer' },
  scrmsd: { adapterId: 'designability-scorer' },
  sctm: { adapterId: 'designability-scorer' },
  designability: { adapterId: 'designability-scorer' },
  compare: { adapterId: 'structure-compare' },
  cmp: { adapterId: 'structure-compare' },
  superpose: { adapterId: 'structure-compare' },
  '3d-compare': { adapterId: 'structure-compare' },
  'pulchra-3.06': { adapterId: 'pulchra' },
  allatom: { adapterId: 'pulchra' },
  'cg2all-ca': { adapterId: 'cg2all' },
  'cg2all-fix': { adapterId: 'cg2all', presetParams: { variant: 'ca-fix' } },
  backmap: { adapterId: 'cg2all' },
};

for (const opt of LOCALFOLD_MODEL_OPTIONS) {
  const key = opt.id.toLowerCase();
  if (!Object.hasOwn(FLOW_MODEL_ALIASES, key)) {
    FLOW_MODEL_ALIASES[key] = { adapterId: 'localfold-af2', presetParams: { model: opt.id } };
  }
}

const CA_ONLY_GENERATORS = new Set(['genie3', 'genie2', 'proteina', 'sid-protein']);

export interface FlowCatalogHooks {
  resolveManifest?: (adapterId: string) => AdapterManifest | undefined;
  resolveShortCode?: (adapterId: string) => string | undefined;
  resolveFromShortCode?: (shortCode: string) => string | undefined;
  yamlKeyAliases?: (adapterId: string) => Record<string, string> | undefined;
  prepareForShare?: (
    adapterId: string,
    params: Record<string, unknown>,
    opts: { stripLargeUploads: boolean },
  ) => { params: Record<string, unknown>; stripped?: boolean } | undefined;
  stripOnShareKeys?: (adapterId: string) => readonly string[] | undefined;
  sanitizeOnImport?: (adapterId: string, params: Record<string, unknown>) => Record<string, unknown> | undefined;
}

export interface FlowValidationError {
  code: string;
  message: string;
  token?: string;
  nodeId?: string;
}

export interface LinkNodeSummary {
  id: string;
  adapterId: string;
  label: string;
  stage: AdapterManifest['stage'];
  params: Record<string, unknown>;
}

export interface LinkEdgeSummary {
  fromNode: string;
  fromPort: string;
  toNode: string;
  toPort: string;
}

export interface FlowParseResult {
  ok: boolean;
  graph: WorkflowGraph | null;
  nodes: LinkNodeSummary[];
  edges: LinkEdgeSummary[];
  errors: FlowValidationError[];
}

export interface FlowParseOptions {
  catalog?: FlowCatalogHooks;
  requireRunnable?: boolean;
  id?: string;
  name?: string;
}

export interface FlowEncodeOptions {
  catalog?: FlowCatalogHooks;
  stripLargeUploads?: boolean;
}

export interface FlowEncodeResult {
  dsl: string;
  flow: string;
  query: string;
  length: number;
  strippedPdb: boolean;
  warnings: string[];
}

const PORT_TYPE_LABELS: Record<PortType, string> = {
  StructureArtifact: 'Structure (3D)',
  TrajectoryFrames: 'Trajectory',
  SequenceSet: 'Sequences (AA)',
  FoldedStructure: 'Predicted Structure',
  DesignabilityMetrics: 'Designability Scores',
  MutationMatrix: 'Mutation Matrix (L×20)',
  ResidueTrack: 'Residue Profile',
  SequenceEmbedding: 'Embeddings',
  ConditionSpec: 'Design Constraints',
};

const SHA256_HEX_RE = /^[0-9a-f]{64}$/i;
const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function computeUploadDigestEdge(text: string): string {
  const bytes = new TextEncoder().encode(text);
  const bitLenHi = Math.floor((bytes.length * 8) / 0x100000000);
  const bitLenLo = (bytes.length * 8) >>> 0;
  const paddedLen = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLen);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLen - 8, bitLenHi, false);
  view.setUint32(paddedLen - 4, bitLenLo, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(offset + i * 4, false);
    for (let i = 16; i < 64; i++) {
      const s0 = ((w[i - 15] >>> 7) | (w[i - 15] << 25)) ^ ((w[i - 15] >>> 18) | (w[i - 15] << 14)) ^ (w[i - 15] >>> 3);
      const s1 = ((w[i - 2] >>> 17) | (w[i - 2] << 15)) ^ ((w[i - 2] >>> 19) | (w[i - 2] << 13)) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = [h0, h1, h2, h3, h4, h5, h6, h7];
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;
      h = g; g = f; f = e; e = (d + temp1) >>> 0;
      d = c; c = b; b = a; a = (temp1 + temp2) >>> 0;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0; h5 = (h5 + f) >>> 0; h6 = (h6 + g) >>> 0; h7 = (h7 + h) >>> 0;
  }
  return [h0, h1, h2, h3, h4, h5, h6, h7].map((n) => n.toString(16).padStart(8, '0')).join('');
}

const VALID_AA_SET_EDGE = new Set('ACDEFGHIKLMNPQRSTVWY'.split(''));
const UNIPROT_ACCESSION_RE_EDGE =
  /^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$/;

function normalizePdbIdEdge(raw: string): string {
  return raw
    .trim()
    .replace(/\.(cif|mmcif|pdb|ent)$/i, '')
    .trim()
    .toUpperCase();
}

function isValidPdbIdEdge(raw: string): boolean {
  const id = normalizePdbIdEdge(raw);
  return /^[0-9][A-Z0-9]{3}$/.test(id) || /^PDB_[0-9A-Z]{8}$/.test(id);
}

function validatePdbIdEdge(raw: string): string | null {
  const norm = normalizePdbIdEdge(raw);
  if (!norm) {
    return 'Enter a 4-character RCSB PDB ID (e.g. 1PRW, 5O45).';
  }
  if (!isValidPdbIdEdge(norm)) {
    return `"${norm}" is not a valid PDB ID (expected 4 characters starting with a digit, e.g. 1PRW).`;
  }
  return null;
}

function normalizeUniProtIdEdge(raw: string): string {
  let s = raw.trim();
  if (!s) return '';
  const urlMatch = s.match(/alphafold\.ebi\.ac\.uk\/(?:entry|files)\/([^/?#]+)/i);
  if (urlMatch) {
    s = urlMatch[1];
  }
  s = s.replace(/\.(bcif|cif|mmcif|pdb|ent)$/i, '').trim();
  const afMatch = s.match(/^AF-([A-Za-z0-9]+(?:-\d+)?)-F\d+(?:-model_v\d+)?$/i);
  if (afMatch) {
    s = afMatch[1];
  }
  return s.toUpperCase();
}

function validateUniProtIdEdge(raw: string): string | null {
  const norm = normalizeUniProtIdEdge(raw);
  if (!norm) {
    return 'Enter a UniProt accession (e.g. P69905, P04637, Q9NZQ7).';
  }
  if (!UNIPROT_ACCESSION_RE_EDGE.test(norm)) {
    return `"${norm}" is not a valid UniProt accession (expected e.g. P69905 or Q9NZQ7).`;
  }
  return null;
}

function validateSequenceTextEdge(raw: string): string | null {
  const text = raw.trim();
  if (!text) {
    return 'Enter at least one amino acid sequence.';
  }
  const lines = text.split(/\r?\n/);
  const entries: string[] = [];
  let cur = '';
  let activeHeader: string | null = null;
  for (const line of lines) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith('>')) {
      if (cur) entries.push(cur);
      activeHeader = t.slice(1).trim().split(/\s+/)[0] || '(unnamed)';
      cur = '';
    } else {
      cur += t.replace(/\s+/g, '').toUpperCase();
    }
  }
  if (cur) entries.push(cur);
  void activeHeader;

  if (entries.length === 0) {
    return 'FASTA header found, but no sequence residues.';
  }

  const illegal = new Set<string>();
  let totalResidues = 0;
  let maxLen = 0;
  for (const seq of entries) {
    let aaCount = 0;
    for (const ch of seq) {
      if (ch === ':' || ch === '/') continue;
      if (!VALID_AA_SET_EDGE.has(ch)) {
        illegal.add(ch);
      } else {
        aaCount++;
      }
    }
    totalResidues += aaCount;
    if (aaCount > maxLen) maxLen = aaCount;
  }

  if (illegal.size > 0) {
    const shown = [...illegal].slice(0, 6).join(', ');
    const list = illegal.size > 6 ? `${shown} (+${illegal.size - 6} more)` : shown;
    return `Invalid residue character${illegal.size > 1 ? 's' : ''}: ${list} (use standard 20 AAs, or ':' between chains).`;
  }
  if (totalResidues === 0 || maxLen === 0) {
    return 'Sequence contains no amino acid residues (minimum 5 aa required).';
  }
  if (maxLen < 5) {
    return `Sequence too short (${maxLen} aa) — minimum 5 amino acids required.`;
  }
  return null;
}

function validateTiedPositionsEdge(raw: string): string | null {
  const text = (raw ?? '').trim();
  if (!text) return null;
  const rawGroups = text.split(/[;,]+/);
  let groupCount = 0;
  const seenResidues = new Set<string>();
  for (const rawGroup of rawGroups) {
    const groupText = rawGroup.trim();
    if (!groupText) continue;
    groupCount++;
    if (groupText.includes('=') && groupText.split('=').some((part) => !part.trim())) {
      return `Invalid tiedPositions group "${groupText}" (empty side around "="; expected e.g. "A10=B10").`;
    }
    const tokens = groupText.split(/[=\s]+/).filter(Boolean);
    if (tokens.length < 2) {
      return `Invalid tiedPositions group "${groupText}" (expected at least 2 residues, e.g. "A10=B10").`;
    }
    for (const tok of tokens) {
      const m = /^([A-Za-z_]):?(-?\d+)([A-Za-z])?$/.exec(tok);
      if (!m) {
        return `Invalid tiedPositions token "${tok}" in "${groupText}" (expected chain ID + residue number, e.g. "A10=B10" or "A100A=B100A").`;
      }
      const key = `${m[1].toUpperCase()}:${Number(m[2])}${m[3] ?? ''}`;
      if (seenResidues.has(key)) {
        return `Duplicate or overlapping tiedPositions residue "${tok}" in "${groupText}" (each residue may appear in at most one tie group).`;
      }
      seenResidues.add(key);
    }
  }
  if (groupCount === 0) {
    return `Invalid tiedPositions "${text}" (expected at least 2 residues, e.g. "A10=B10").`;
  }
  return null;
}

function validateJsonOverrideEdge(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return 'Override must be a JSON object (e.g. {"contig": "70-70"}).';
    }
    return null;
  } catch (err) {
    return `Invalid JSON: ${err instanceof Error ? err.message : safeStr(err)}`;
  }
}

function validateCustomSpecYamlEdge(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    return validateJsonOverrideEdge(trimmed);
  }
  let braceDepth = 0;
  let bracketDepth = 0;
  let hasMappingKey = false;
  for (const rawLine of trimmed.split(/\r?\n/)) {
    let inSingle = false;
    let inDouble = false;
    let clean = '';
    for (let i = 0; i < rawLine.length; i++) {
      const ch = rawLine[i];
      if (ch === "'" && !inDouble) {
        inSingle = !inSingle;
      } else if (ch === '"' && !inSingle && rawLine[i - 1] !== '\\') {
        inDouble = !inDouble;
      } else if (!inSingle && !inDouble) {
        if (ch === '#') break;
        if (ch === '{') braceDepth++;
        else if (ch === '}') braceDepth--;
        else if (ch === '[') bracketDepth++;
        else if (ch === ']') bracketDepth--;
        if (braceDepth < 0 || bracketDepth < 0) {
          return 'Invalid customSpecYaml: unmatched closing bracket or brace.';
        }
      }
      clean += ch;
    }
    if (inSingle || inDouble) {
      return 'Invalid customSpecYaml: unclosed quote string.';
    }
    const stripped = clean.trim();
    if (!stripped) continue;
    if (/^[A-Za-z_][A-Za-z0-9_-]*\s*:/.test(stripped)) {
      hasMappingKey = true;
    }
  }
  if (braceDepth !== 0 || bracketDepth !== 0) {
    return 'Invalid customSpecYaml: unclosed "{" or "[".';
  }
  if (!hasMappingKey) {
    return 'Invalid customSpecYaml: expected a YAML mapping (e.g. "contig: 70-70") or JSON object.';
  }
  return null;
}

function validateStructureTextEdge(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return 'Paste a PDB / mmCIF structure or upload a file.';
  }
  const isCif = /_atom_site\./.test(trimmed);
  const hasAtom = /^(ATOM|HETATM)\s+/m.test(trimmed);
  if (!isCif && !hasAtom) {
    return 'No PDB ATOM lines or mmCIF _atom_site records found.';
  }
  return null;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return Boolean(v) && typeof v === 'object' && !Array.isArray(v);
}

function sanitizeValue(val: unknown, depth = 0): unknown {
  if (val === null || typeof val !== 'object') return val;
  if (depth > 32) return undefined;
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

function resolveManifest(adapterId: string, catalog?: FlowCatalogHooks): AdapterManifest | undefined {
  const fromCatalog = catalog?.resolveManifest?.(adapterId);
  if (fromCatalog) return fromCatalog;
  return Object.hasOwn(ADAPTER_REGISTRY, adapterId) ? ADAPTER_REGISTRY[adapterId] : undefined;
}

function resolveShortCode(adapterId: string, catalog?: FlowCatalogHooks): string {
  return catalog?.resolveShortCode?.(adapterId) ?? (Object.hasOwn(ADAPTER_TO_SHORT, adapterId) ? ADAPTER_TO_SHORT[adapterId] : adapterId);
}

export function resolveFlowModelAlias(token: string, catalog?: FlowCatalogHooks): FlowAliasResolution | null {
  const raw = token.trim();
  if (!raw || FORBIDDEN_KEYS.has(raw)) return null;

  if (resolveManifest(raw, catalog)) {
    return { adapterId: raw };
  }
  const lower = raw.toLowerCase();
  if (resolveManifest(lower, catalog)) {
    return { adapterId: lower };
  }
  const fromShort =
    catalog?.resolveFromShortCode?.(raw) ??
    catalog?.resolveFromShortCode?.(lower) ??
    (Object.hasOwn(SHORT_TO_ADAPTER, lower) ? SHORT_TO_ADAPTER[lower] : undefined);
  if (fromShort && resolveManifest(fromShort, catalog)) {
    return { adapterId: fromShort };
  }
  if (Object.hasOwn(FLOW_MODEL_ALIASES, lower)) {
    const alias = FLOW_MODEL_ALIASES[lower];
    return {
      adapterId: alias.adapterId,
      ...(alias.presetParams ? { presetParams: { ...alias.presetParams } } : {}),
    };
  }
  return null;
}

export function defaultParamsForAdapterEdge(
  adapterId: string,
  catalog?: FlowCatalogHooks,
): Record<string, unknown> {
  const manifest = resolveManifest(adapterId, catalog);
  if (!manifest?.paramSchema) return {};
  const full = getDefaultParamsFromSchema(manifest);
  if (adapterId === 'localfold-af2') {
    const trimmed: Record<string, unknown> = {};
    for (const key of LOCALFOLD_DEFAULT_BASELINE_KEYS) {
      if (key in full) trimmed[key] = full[key];
    }
    return trimmed;
  }
  return full;
}

function arePortSpecsCompatible(outSpec: PortSpec, inSpec: PortSpec): boolean {
  const outTypes = Array.isArray(outSpec.type) ? outSpec.type : [outSpec.type];
  const inTypes = new Set(Array.isArray(inSpec.type) ? inSpec.type : [inSpec.type]);
  return outTypes.some((t) => inTypes.has(t));
}

function wouldCreateCycle(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  fromNode: string,
  toNode: string,
): boolean {
  if (fromNode === toNode) return true;
  const adj = new Map<string, string[]>();
  for (const n of nodes) adj.set(n.id, []);
  for (const e of edges) adj.get(e.fromNode)?.push(e.toNode);

  const visited = new Set<string>([toNode]);
  const stack = [toNode];
  while (stack.length > 0) {
    const cur = stack.pop()!;
    if (cur === fromNode) return true;
    for (const nxt of adj.get(cur) ?? []) {
      if (!visited.has(nxt)) {
        visited.add(nxt);
        stack.push(nxt);
      }
    }
  }
  return false;
}

/** Pure, edge-safe left-to-right auto-layout matching `graph-utils.ts:autoLayoutGraph`. */
export function autoLayoutGraphEdge(
  graph: WorkflowGraph,
  catalog?: FlowCatalogHooks,
): WorkflowGraph {
  if (graph.nodes.length === 0) return graph;
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const depth = new Map<string, number>();
  for (const n of graph.nodes) depth.set(n.id, 0);

  for (let iter = 0; iter < graph.nodes.length; iter++) {
    let changed = false;
    for (const e of graph.edges) {
      const dSrc = depth.get(e.fromNode) ?? 0;
      const dDst = depth.get(e.toNode) ?? 0;
      if (dSrc + 1 > dDst) {
        depth.set(e.toNode, dSrc + 1);
        changed = true;
      }
    }
    if (!changed) break;
  }

  const byCol = new Map<number, WorkflowNode[]>();
  let maxCol = 0;
  for (const n of graph.nodes) {
    const col = depth.get(n.id) ?? 0;
    if (col > maxCol) maxCol = col;
    const list = byCol.get(col) ?? [];
    list.push(n);
    byCol.set(col, list);
  }

  const PORT0_Y = 70;
  const PORT_STEP = 26;
  const portIdx = (nodeId: string, dir: 'inputs' | 'outputs', portName: string): number => {
    const node = byId.get(nodeId);
    if (!node) return 0;
    const keys = Object.keys(resolveManifest(node.adapterId, catalog)?.[dir] ?? {});
    const idx = keys.indexOf(portName);
    return idx >= 0 ? idx : 0;
  };
  const cardHeight = (node: WorkflowNode): number => {
    const m = resolveManifest(node.adapterId, catalog);
    const maxP = Math.max(1, m ? Object.keys(m.inputs).length : 1, m ? Object.keys(m.outputs).length : 1);
    return 132 + maxP * PORT_STEP;
  };

  const spannedCols = new Set<number>();
  for (const e of graph.edges) {
    const c1 = depth.get(e.fromNode) ?? 0;
    const c2 = depth.get(e.toNode) ?? 0;
    if (c2 - c1 >= 2) {
      for (let c = c1 + 1; c < c2; c++) spannedCols.add(c);
    }
  }

  const X_START = 40;
  const X_GAP = 300;
  const TOP_Y = 24;
  const MAIN_Y = spannedCols.size > 0 ? 84 : TOP_Y + 36;
  const ROW_MARGIN = 24;
  const posMap = new Map<string, { x: number; y: number }>();

  for (let col = 0; col <= maxCol; col++) {
    const list = byCol.get(col);
    if (!list || list.length === 0) continue;

    if (col > 0 && list.length > 1) {
      const barycenter = (node: WorkflowNode): number => {
        const inc = graph.edges.filter((e) => e.toNode === node.id);
        if (inc.length === 0) return 0;
        let sum = 0;
        for (const e of inc) {
          const upY = posMap.get(e.fromNode)?.y ?? MAIN_Y;
          sum += upY + PORT0_Y + portIdx(e.fromNode, 'outputs', e.fromPort) * PORT_STEP;
        }
        return sum / inc.length;
      };
      list.sort((a, b) => barycenter(a) - barycenter(b));
    }

    const x = X_START + col * X_GAP;
    if (list.length === 1) {
      const n = list[0];
      const inc = graph.edges.filter((e) => e.toNode === n.id);
      let y = MAIN_Y;
      const primary = inc.find((e) => (depth.get(e.fromNode) ?? 0) === col - 1) ?? inc[0];
      if (primary) {
        const upY = posMap.get(primary.fromNode)?.y ?? MAIN_Y;
        const outI = portIdx(primary.fromNode, 'outputs', primary.fromPort);
        const inI = portIdx(n.id, 'inputs', primary.toPort);
        y = Math.max(TOP_Y, upY + (outI - inI) * PORT_STEP);
      }
      posMap.set(n.id, { x, y: Math.round(y) });
    } else {
      let cursorY = MAIN_Y;
      for (const n of list) {
        const inc = graph.edges.filter((e) => e.toNode === n.id);
        let idealY = cursorY;
        const primary = inc.find((e) => (depth.get(e.fromNode) ?? 0) === col - 1) ?? inc[0];
        if (primary) {
          const upY = posMap.get(primary.fromNode)?.y;
          if (upY !== undefined) {
            const outI = portIdx(primary.fromNode, 'outputs', primary.fromPort);
            const inI = portIdx(n.id, 'inputs', primary.toPort);
            idealY = Math.max(cursorY, upY + (outI - inI) * PORT_STEP);
          }
        }
        posMap.set(n.id, { x, y: Math.round(idealY) });
        cursorY = idealY + cardHeight(n) + ROW_MARGIN;
      }
    }
  }

  return {
    ...graph,
    nodes: graph.nodes.map((n) => ({
      ...n,
      position: posMap.get(n.id) ?? { x: X_START, y: TOP_Y },
    })),
  };
}

/** Edge-safe structural + schema + crossValidate validation returning structured errors. */
export function validateGraphEdge(
  graph: WorkflowGraph,
  opts: { requireRunnable?: boolean; catalog?: FlowCatalogHooks } = {},
): FlowValidationError[] {
  const requireRunnable = opts.requireRunnable !== false;
  const catalog = opts.catalog;
  const errors: FlowValidationError[] = [];

  if (graph.nodes.length === 0) {
    if (requireRunnable) {
      errors.push({ code: 'EMPTY_FLOW', message: 'Workflow contains no nodes.' });
    }
    return errors;
  }
  if (graph.nodes.length > MAX_IMPORT_NODES) {
    errors.push({
      code: 'GRAPH_TOO_LARGE',
      message: `Workflow exceeds maximum allowed block count (${MAX_IMPORT_NODES}).`,
    });
  }
  if (graph.edges.length > MAX_IMPORT_EDGES) {
    errors.push({
      code: 'GRAPH_TOO_LARGE',
      message: `Workflow exceeds maximum allowed wire count (${MAX_IMPORT_EDGES}).`,
    });
  }

  const byId = new Map<string, WorkflowNode>();
  for (const node of graph.nodes) {
    if (typeof node.id !== 'string' || !node.id || FORBIDDEN_KEYS.has(node.id)) {
      const idLabel = safeStr(node.id);
      errors.push({
        code: 'INVALID_NODE_ID',
        message: `Invalid or reserved node ID "${idLabel}".`,
        nodeId: idLabel,
      });
      continue;
    }
    if (byId.has(node.id)) {
      errors.push({
        code: 'DUPLICATE_NODE_ID',
        message: `Duplicate node ID "${node.id}".`,
        nodeId: node.id,
      });
    } else {
      byId.set(node.id, node);
    }
    if (typeof node.adapterId !== 'string' || !resolveManifest(node.adapterId, catalog)) {
      const adapterLabel = safeStr(node.adapterId);
      errors.push({
        code: 'UNKNOWN_MODEL',
        message: `Node "${node.id}" references unknown model "${adapterLabel}".`,
        nodeId: node.id,
        token: adapterLabel,
      });
    }
  }

  const targetPortCounts = new Map<string, number>();
  const seenExactEdges = new Set<string>();
  const validConnectedTargets = new Set<string>();
  const indeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const id of byId.keys()) {
    indeg.set(id, 0);
    adj.set(id, []);
  }

  for (const e of graph.edges) {
    if (
      typeof e.fromNode !== 'string' ||
      typeof e.fromPort !== 'string' ||
      typeof e.toNode !== 'string' ||
      typeof e.toPort !== 'string'
    ) {
      errors.push({
        code: 'INVALID_EDGE',
        message: 'Edge endpoints and port names must be strings.',
      });
      continue;
    }
    const src = byId.get(e.fromNode);
    const dst = byId.get(e.toNode);
    if (!src) {
      errors.push({
        code: 'MISSING_NODE',
        message: `Edge references missing source node "${e.fromNode}".`,
        nodeId: e.fromNode,
      });
      continue;
    }
    if (!dst) {
      errors.push({
        code: 'MISSING_NODE',
        message: `Edge references missing target node "${e.toNode}".`,
        nodeId: e.toNode,
      });
      continue;
    }
    if (e.fromNode === e.toNode) {
      errors.push({
        code: 'SELF_LOOP',
        message: `Self-loop detected on node "${e.fromNode}".`,
        nodeId: e.fromNode,
      });
    }

    const srcManifest = resolveManifest(src.adapterId, catalog);
    const dstManifest = resolveManifest(dst.adapterId, catalog);
    const inSpec =
      dstManifest && Object.hasOwn(dstManifest.inputs, e.toPort)
        ? dstManifest.inputs[e.toPort]
        : undefined;
    let edgePortsOk = Boolean(srcManifest && dstManifest);
    if (srcManifest && dstManifest) {
      const outSpec = Object.hasOwn(srcManifest.outputs, e.fromPort)
        ? srcManifest.outputs[e.fromPort]
        : undefined;
      if (!outSpec) {
        errors.push({
          code: 'UNKNOWN_PORT',
          message: `Node "${src.id}" (${srcManifest.shortName}) has unknown output port "${e.fromPort}".`,
          nodeId: src.id,
          token: e.fromPort,
        });
        edgePortsOk = false;
      }
      if (!inSpec) {
        errors.push({
          code: 'UNKNOWN_PORT',
          message: `Node "${dst.id}" (${dstManifest.shortName}) has unknown input port "${e.toPort}".`,
          nodeId: dst.id,
          token: e.toPort,
        });
        edgePortsOk = false;
      }
      if (outSpec && inSpec && !arePortSpecsCompatible(outSpec as PortSpec, inSpec as PortSpec)) {
        const outPrimary = (Array.isArray(outSpec.type) ? outSpec.type[0] : outSpec.type) as PortType;
        const inPrimary = (Array.isArray(inSpec.type) ? inSpec.type[0] : inSpec.type) as PortType;
        errors.push({
          code: 'INCOMPATIBLE_PORTS',
          message: `Type mismatch on connection ${src.id}.${e.fromPort} -> ${dst.id}.${e.toPort} (${PORT_TYPE_LABELS[outPrimary] ?? outPrimary} vs ${PORT_TYPE_LABELS[inPrimary] ?? inPrimary}).`,
          nodeId: dst.id,
        });
        edgePortsOk = false;
      }
    }

    const exactKey = `${e.fromNode}:${e.fromPort}->${e.toNode}:${e.toPort}`;
    if (seenExactEdges.has(exactKey)) {
      errors.push({
        code: 'DUPLICATE_EDGE',
        message: `Duplicate edge "${e.fromNode}.${e.fromPort} -> ${e.toNode}.${e.toPort}".`,
        nodeId: dst.id,
      });
    } else {
      seenExactEdges.add(exactKey);
    }

    const targetKey = `${e.toNode}:${e.toPort}`;
    const prevCount = targetPortCounts.get(targetKey) ?? 0;
    if (inSpec?.multiple) {
      if (prevCount >= MAX_MULTI_INPUT_EDGES) {
        errors.push({
          code: 'MAX_MULTI_INPUTS_EXCEEDED',
          message: `Multi-input port "${e.toNode}.${e.toPort}" exceeds maximum ${MAX_MULTI_INPUT_EDGES} incoming edges.`,
          nodeId: dst.id,
        });
      }
      targetPortCounts.set(targetKey, prevCount + 1);
    } else if (prevCount >= 1) {
      errors.push({
        code: 'INPUT_PORT_CONFLICT',
        message: `Input port "${e.toNode}.${e.toPort}" has multiple incoming edges.`,
        nodeId: dst.id,
      });
      targetPortCounts.set(targetKey, prevCount + 1);
    } else {
      targetPortCounts.set(targetKey, 1);
    }

    if (edgePortsOk) {
      validConnectedTargets.add(targetKey);
    }
    adj.get(e.fromNode)!.push(e.toNode);
    indeg.set(e.toNode, (indeg.get(e.toNode) ?? 0) + 1);
  }

  for (const node of byId.values()) {
    const manifest = resolveManifest(node.adapterId, catalog);
    if (!manifest) continue;
    const connectedInputs = new Set<string>();
    for (const portName of Object.keys(manifest.inputs)) {
      if (validConnectedTargets.has(`${node.id}:${portName}`)) {
        connectedInputs.add(portName);
      }
    }

    const nodeLabel = typeof node.label === 'string' && node.label ? node.label : manifest.shortName;
    const p = isPlainObject(node.params) ? node.params : {};

    if (requireRunnable) {
      for (const [portName, spec] of Object.entries(manifest.inputs)) {
        if (spec.required === false) continue;
        if (!connectedInputs.has(portName)) {
          const pType = (Array.isArray(spec.type) ? spec.type[0] : spec.type) as PortType;
          errors.push({
            code: 'MISSING_REQUIRED_INPUT',
            message: `"${nodeLabel}" (${node.id}) is missing required input "${portName}" (${PORT_TYPE_LABELS[pType] ?? pType}).`,
            nodeId: node.id,
            token: portName,
          });
        }
      }
    }

    const schemaErrs = validateParamsWithSchema(manifest, p, {
      label: nodeLabel,
      ...(requireRunnable ? { connectedInputs } : {}),
    });
    for (const msg of schemaErrs) {
      const code = msg.includes('requires a connected') ? 'MISSING_REQUIRED_INPUT' : 'INVALID_PARAM';
      errors.push({ code, message: msg, nodeId: node.id });
    }

    if (manifest.crossValidate) {
      const safeCrossParams: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(p)) {
        if (v === null || typeof v !== 'object' || k === 'unresolvedImportIssues' || k === 'residueMapping') {
          safeCrossParams[k] = v;
        }
      }
      try {
        const crossErrs = manifest.crossValidate(
          safeCrossParams,
          requireRunnable ? { connectedInputs } : undefined,
        );
        for (const msg of crossErrs) {
          errors.push({ code: 'CROSS_VALIDATION_ERROR', message: msg, nodeId: node.id });
        }
      } catch (err) {
        errors.push({
          code: 'CROSS_VALIDATION_ERROR',
          message: err instanceof Error ? err.message : 'Cross-validation failed.',
          nodeId: node.id,
        });
      }
    }

    if (node.adapterId === 'rfd4-proteina') {
      if (p.customSpecYaml !== undefined && p.customSpecYaml !== null && typeof p.customSpecYaml !== 'string') {
        errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": customSpecYaml must be a string.`, nodeId: node.id, token: 'customSpecYaml' });
      } else {
        const customYaml = safeStr(p.customSpecYaml).trim();
        if (customYaml) {
          const yamlErr = validateCustomSpecYamlEdge(customYaml);
          if (yamlErr) {
            errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": ${yamlErr}`, nodeId: node.id, token: 'customSpecYaml' });
          }
        }
      }
    }

    if (node.adapterId === 'web-protein-mpnn' || node.adapterId === 'ligandmpnn') {
      if (typeof p.tiedPositions === 'string' && p.tiedPositions.trim()) {
        const tiedErr = validateTiedPositionsEdge(p.tiedPositions);
        if (tiedErr) {
          errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": ${tiedErr}`, nodeId: node.id, token: 'tiedPositions' });
        }
      }
    }

    if (requireRunnable && p.unresolvedImportIssues !== undefined && p.unresolvedImportIssues !== null) {
      const formatIssue = (raw: unknown): string => {
        if (typeof raw === 'string') return raw.trim();
        if (raw === undefined || raw === null) return '';
        if (typeof raw === 'object') {
          try {
            return JSON.stringify(raw);
          } catch {
            return '';
          }
        }
        return safeStr(raw).trim();
      };
      if (Array.isArray(p.unresolvedImportIssues)) {
        for (const rawIssue of p.unresolvedImportIssues) {
          const issue = formatIssue(rawIssue);
          if (issue) {
            errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": Unresolved imported setting — ${issue}`, nodeId: node.id });
          }
        }
      } else {
        const issue = formatIssue(p.unresolvedImportIssues);
        if (issue) {
          errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": Unresolved imported setting — ${issue}`, nodeId: node.id });
        }
      }
    }

    if (node.adapterId === 'pdb-input') {
      if (p.mode !== undefined && p.mode !== null && typeof p.mode !== 'string') {
        errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": mode must be a string.`, nodeId: node.id, token: 'mode' });
      }
      const mode = typeof p.mode === 'string' ? p.mode : 'demo-bundle';
      if (mode === 'pdb-id') {
        if (p.pdbId !== undefined && p.pdbId !== null && typeof p.pdbId !== 'string') {
          errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": pdbId must be a string.`, nodeId: node.id, token: 'pdbId' });
        } else {
          const pdbErr = validatePdbIdEdge(safeStr(p.pdbId));
          if (pdbErr) {
            errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": ${pdbErr}`, nodeId: node.id, token: 'pdbId' });
          }
        }
      } else if (mode === 'alphafold-db') {
        if (p.uniprotId !== undefined && p.uniprotId !== null && typeof p.uniprotId !== 'string') {
          errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": uniprotId must be a string.`, nodeId: node.id, token: 'uniprotId' });
        } else {
          const uniErr = validateUniProtIdEdge(safeStr(p.uniprotId));
          if (uniErr) {
            errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": ${uniErr}`, nodeId: node.id, token: 'uniprotId' });
          }
        }
      } else if (mode === 'custom') {
        if (p.pdbText !== undefined && p.pdbText !== null && typeof p.pdbText !== 'string') {
          errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": pdbText must be a string.`, nodeId: node.id, token: 'pdbText' });
        }
        const rawPdb = safeStr(p.pdbText);
        const missingUpload = Boolean(p.missingUpload);
        const expectedShaRaw = safeStr(p.expectedPdbSha256).trim();
        const expectedSha = expectedShaRaw.toLowerCase();
        const expectedFilename = safeStr(p.expectedFilename ?? p.filename).trim();
        if (expectedSha && !SHA256_HEX_RE.test(expectedSha)) {
          errors.push({
            code: 'INVALID_PARAM',
            message: `"${nodeLabel}": Invalid expectedPdbSha256 digest "${expectedShaRaw}" (expected a 64-character SHA-256 hex digest).`,
            nodeId: node.id,
            token: 'expectedPdbSha256',
          });
        } else if (requireRunnable) {
          if (!rawPdb.trim()) {
            if (missingUpload || expectedSha) {
              const fileNote = expectedFilename ? ` "${expectedFilename}"` : '';
              const digestNote = expectedSha ? ` (expected SHA-256 ${expectedSha.slice(0, 12)}…)` : '';
              errors.push({
                code: 'INVALID_PARAM',
                message: `"${nodeLabel}": Custom upload${fileNote} was omitted from this shared recipe${digestNote}. Re-upload the matching PDB/mmCIF file, or switch Source mode / clear expectedPdbSha256.`,
                nodeId: node.id,
              });
            } else {
              const structErr = validateStructureTextEdge(rawPdb);
              if (structErr) {
                errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": ${structErr}`, nodeId: node.id, token: 'pdbText' });
              }
            }
          } else {
            const structErr = validateStructureTextEdge(rawPdb);
            if (structErr) {
              errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": ${structErr}`, nodeId: node.id, token: 'pdbText' });
            } else if (expectedSha) {
              const actualSha = computeUploadDigestEdge(rawPdb);
              if (actualSha !== expectedSha) {
                const fileNote = expectedFilename ? ` "${expectedFilename}"` : '';
                errors.push({
                  code: 'INVALID_PARAM',
                  message: `"${nodeLabel}": Uploaded structure digest (${actualSha.slice(0, 12)}…) does not match expected custom upload${fileNote} (SHA-256 ${expectedSha.slice(0, 12)}…).`,
                  nodeId: node.id,
                });
              }
            }
          }
        }
      }
    }

    if (node.adapterId === 'sequence-input') {
      if (p.sequence !== undefined && p.sequence !== null && typeof p.sequence !== 'string') {
        errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": sequence must be a string.`, nodeId: node.id, token: 'sequence' });
      } else {
        const seqErr = validateSequenceTextEdge(safeStr(p.sequence));
        if (seqErr) {
          errors.push({ code: 'INVALID_PARAM', message: `"${nodeLabel}": ${seqErr}`, nodeId: node.id, token: 'sequence' });
        }
      }
    }
  }

  // Cycle detection via Kahn's topological sort
  const queue: string[] = [];
  for (const [id, d] of indeg) {
    if (d === 0) queue.push(id);
  }
  let visited = 0;
  while (queue.length > 0) {
    const cur = queue.shift()!;
    visited++;
    for (const nxt of adj.get(cur) ?? []) {
      const nd = (indeg.get(nxt) ?? 1) - 1;
      indeg.set(nxt, nd);
      if (nd === 0) queue.push(nxt);
    }
  }
  if (visited !== byId.size) {
    errors.push({
      code: 'CYCLE_DETECTED',
      message: 'Workflow graph contains a cycle (must be a directed acyclic graph).',
    });
  }

  return errors;
}

function summarizeNodes(nodes: WorkflowNode[], catalog?: FlowCatalogHooks): LinkNodeSummary[] {
  return nodes.map((n) => ({
    id: n.id,
    adapterId: n.adapterId,
    label: n.label ?? resolveManifest(n.adapterId, catalog)?.shortName ?? n.adapterId,
    stage: resolveManifest(n.adapterId, catalog)?.stage ?? 'score',
    params: n.params,
  }));
}

function summarizeEdges(edges: WorkflowEdge[]): LinkEdgeSummary[] {
  return edges.map((e) => ({
    fromNode: e.fromNode,
    fromPort: e.fromPort,
    toNode: e.toNode,
    toPort: e.toPort,
  }));
}

// ---------------------------------------------------------------------------
// Context-aware ?flow= Lexer & Parser
// ---------------------------------------------------------------------------

/** Strip trailing prose punctuation added by chat/Markdown wrappers without breaking `node(k=v)` or `"quoted"`. */
export function stripTrailingProsePunctuation(raw: string): string {
  let s = raw.trim();
  while (s.length > 0) {
    const last = s[s.length - 1];
    if (last === '.' || last === ',' || last === ';' || last === '!' || last === '?' || last === '>') {
      s = s.slice(0, -1).trimEnd();
      continue;
    }
    if (last === ')' || last === ']' || last === '}') {
      const openCh = last === ')' ? '(' : last === ']' ? '[' : '{';
      let depth = 0;
      let quote: string | null = null;
      for (let i = 0; i < s.length; i++) {
        const ch = s[i];
        if (quote) {
          if (ch === '\\') { i++; continue; }
          if (ch === quote) quote = null;
          continue;
        }
        if (ch === '"' || ch === "'") { quote = ch; continue; }
        if (ch === openCh) depth++;
        else if (ch === last) depth--;
      }
      if (depth < 0) {
        s = s.slice(0, -1).trimEnd();
        continue;
      }
    }
    if (last === '"' || last === "'") {
      let count = 0;
      for (let i = 0; i < s.length; i++) {
        if (s[i] === '\\') { i++; continue; }
        if (s[i] === last) count++;
      }
      if (count % 2 === 1) {
        s = s.slice(0, -1).trimEnd();
        continue;
      }
    }
    break;
  }
  return s;
}

const SEMICOLON_CONTINUATION_PARAM_KEYS = new Set([
  'fixedAtoms',
  'motifAtoms',
  'tiedPositions',
  'hbondSelection',
  'hbondDonorSelection',
  'hbondAcceptorSelection',
  'hbondDonor',
  'hbondAcceptor',
  'rasaBuriedSelection',
  'piStackSelection',
  'unfixedSequence',
]);

const PIPE_SEPARABLE_PARAM_KEYS = new Set([
  'fixedAtoms',
  'motifAtoms',
  'tiedPositions',
  'unfixedSequence',
]);

const STEP_ALIAS_KEYS = new Set(['steps', 'nsteps', 'numSteps', 'nSampleStep']);

function getActiveUnquotedParamKey(curStmt: string): string | null {
  // Isolate the trailing node expression (after last unquoted `~` or `->`)
  const chainParts = splitEdgeChain(curStmt);
  const lastExpr = chainParts[chainParts.length - 1] ?? '';
  const segs = splitOutsideQuotes(lastExpr, ':');
  if (segs.length < 2) return null;

  let paramTail = '';
  const seg1FirstEq = segs[1].indexOf('=');
  const seg1FirstComma = segs[1].indexOf(',');
  const seg1StartsWithParam = seg1FirstEq > 0 && (seg1FirstComma === -1 || seg1FirstEq < seg1FirstComma);
  if (seg1StartsWithParam) {
    paramTail = segs.slice(1).join(':');
  } else if (seg1FirstComma > 0 && (seg1FirstEq === -1 || seg1FirstComma < seg1FirstEq)) {
    paramTail = segs.slice(1).join(':').slice(seg1FirstComma + 1);
  } else if (segs.length >= 3) {
    paramTail = segs.slice(2).join(':');
  }
  if (!paramTail) return null;

  const pairs = splitParamPairs(paramTail);
  if (pairs.length === 0) return null;
  const lastPair = pairs[pairs.length - 1];
  const eqIdx = lastPair.indexOf('=');
  if (eqIdx <= 0) return null;
  return lastPair.slice(0, eqIdx).trim();
}

function isIntraParamSemicolon(
  curStmt: string,
  restAfterSemi: string,
  catalog?: FlowCatalogHooks,
): boolean {
  const activeKey = getActiveUnquotedParamKey(curStmt);
  if (!activeKey || !SEMICOLON_CONTINUATION_PARAM_KEYS.has(activeKey)) return false;

  const nextSemiChunk = restAfterSemi.split(/[;\r\n]/)[0] ?? '';
  const nextNodeHead = (splitEdgeChain(nextSemiChunk)[0] ?? '').trim();
  if (!nextNodeHead) return false;

  if (activeKey === 'tiedPositions') {
    const firstPair = splitParamPairs(nextNodeHead)[0]?.trim() ?? '';
    return /^[A-Za-z0-9]+-?\d+(?:=[A-Za-z0-9]+-?\d+)+$/.test(firstPair);
  }

  // Isolate up to any `,nextKey=` parameter boundary on the same node
  const firstParamChunk = splitParamPairs(nextNodeHead)[0]?.trim() ?? '';
  const atomMapMatch =
    /^([A-Za-z0-9]+-?\d+(?:--?\d+)?)\s*:\s*([A-Za-z0-9,'*]+(?:\s*,\s*[A-Za-z0-9,'*]+)*)$/.exec(
      firstParamChunk,
    );
  if (atomMapMatch) {
    const firstAtomTok = atomMapMatch[2].split(',')[0].trim();
    if (!resolveFlowModelAlias(firstAtomTok, catalog)) {
      return true;
    }
  }
  if (/^[A-Za-z0-9_]+\/[^=~:]+$/.test(firstParamChunk) || /^sel\(/i.test(firstParamChunk)) {
    return true;
  }
  return false;
}

/** Split input on `;` or `\n` outside quotes, parentheses, and unquoted atom-map / tiedPositions values. */
function splitStatements(input: string, catalog?: FlowCatalogHooks): string[] {
  const out: string[] = [];
  let cur = '';
  let quote: string | null = null;
  let parenDepth = 0;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (quote) {
      cur += ch;
      if (ch === '\\' && i + 1 < input.length) {
        cur += input[++i];
        continue;
      }
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      cur += ch;
      continue;
    }
    if (ch === '(') {
      parenDepth++;
      cur += ch;
      continue;
    }
    if (ch === ')') {
      if (parenDepth > 0) parenDepth--;
      cur += ch;
      continue;
    }
    if ((ch === ';' || ch === '\n' || ch === '\r') && parenDepth === 0) {
      if (ch === ';' && isIntraParamSemicolon(cur, input.slice(i + 1), catalog)) {
        cur += ch;
        continue;
      }
      out.push(cur);
      cur = '';
      continue;
    }
    cur += ch;
  }
  if (cur.length > 0) out.push(cur);
  return out;
}

/** Split a statement on `~` or `->` outside quotes and parentheses. */
function splitEdgeChain(stmt: string): string[] {
  const out: string[] = [];
  let cur = '';
  let quote: string | null = null;
  let parenDepth = 0;

  for (let i = 0; i < stmt.length; i++) {
    const ch = stmt[i];
    if (quote) {
      cur += ch;
      if (ch === '\\' && i + 1 < stmt.length) {
        cur += stmt[++i];
        continue;
      }
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      cur += ch;
      continue;
    }
    if (ch === '(') {
      parenDepth++;
      cur += ch;
      continue;
    }
    if (ch === ')') {
      if (parenDepth > 0) parenDepth--;
      cur += ch;
      continue;
    }
    if (parenDepth === 0) {
      if (ch === '~') {
        out.push(cur);
        cur = '';
        continue;
      }
      if (ch === '-' && stmt[i + 1] === '>') {
        out.push(cur);
        cur = '';
        i++;
        continue;
      }
    }
    cur += ch;
  }
  out.push(cur);
  return out;
}

function splitOutsideQuotes(str: string, delimiter: string): string[] {
  const out: string[] = [];
  let cur = '';
  let quote: string | null = null;
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (quote) {
      cur += ch;
      if (ch === '\\' && i + 1 < str.length) {
        cur += str[++i];
        continue;
      }
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      cur += ch;
      continue;
    }
    if (ch === delimiter) {
      out.push(cur);
      cur = '';
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out;
}

/**
 * Split parameter list on unquoted `,` ONLY when the remainder starts with `key=`.
 * Preserves commas inside `contig=10-20,A16-35,10-20`, `hotspots=A54, A56`, `cropSegments=A16-35,A52-71`.
 */
function splitParamPairs(paramStr: string): string[] {
  const out: string[] = [];
  let cur = '';
  let quote: string | null = null;

  for (let i = 0; i < paramStr.length; i++) {
    const ch = paramStr[i];
    if (quote) {
      cur += ch;
      if (ch === '\\' && i + 1 < paramStr.length) {
        cur += paramStr[++i];
        continue;
      }
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      cur += ch;
      continue;
    }
    if (ch === ',') {
      const rest = paramStr.slice(i + 1);
      if (/^\s*[A-Za-z_][A-Za-z0-9_]*\s*=/.test(rest)) {
        out.push(cur);
        cur = '';
        continue;
      }
    }
    cur += ch;
  }
  if (cur.trim().length > 0) out.push(cur);
  return out;
}

function unquoteValue(raw: string): { value: string; wasQuoted: boolean } {
  const s = raw.trim();
  if (s.length >= 2 && ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'")))) {
    const q = s[0];
    const inner = s.slice(1, -1);
    let unescaped = '';
    for (let i = 0; i < inner.length; i++) {
      if (inner[i] === '\\' && i + 1 < inner.length) {
        const next = inner[i + 1];
        if (next === q || next === '\\') {
          unescaped += next;
          i++;
          continue;
        }
      }
      unescaped += inner[i];
    }
    return { value: unescaped, wasQuoted: true };
  }
  return { value: s, wasQuoted: false };
}

interface ParsedNodeExpr {
  explicitId?: string;
  modelToken?: string;
  refToken: string;
  port?: string;
  rawParams: Array<{ key: string; rawVal: string; wasQuoted: boolean }>;
  rawExpr: string;
}

function extractTrailingParenParams(expr: string): { head: string; parenParams: string } | null {
  const s = expr.trim();
  if (!s.endsWith(')')) return { head: s, parenParams: '' };

  let quote: string | null = null;
  let openIdx = -1;
  let depth = 0;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (quote) {
      if (ch === '\\') { i++; continue; }
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (ch === '(') {
      if (depth === 0) openIdx = i;
      depth++;
    } else if (ch === ')') {
      depth--;
      if (depth < 0) return null;
      if (depth === 0 && i < s.length - 1) {
        return null;
      }
    }
  }
  if (depth !== 0 || openIdx <= 0) return null;
  return {
    head: s.slice(0, openIdx).trim(),
    parenParams: s.slice(openIdx + 1, -1).trim(),
  };
}

function splitPortSuffix(
  token: string,
  existingNodeIds: ReadonlySet<string>,
  catalog?: FlowCatalogHooks,
): { base: string; port?: string } {
  if (!token) return { base: token };
  if (existingNodeIds.has(token) || resolveFlowModelAlias(token, catalog)) {
    return { base: token };
  }
  const dot = token.lastIndexOf('.');
  if (dot <= 0) return { base: token };
  const base = token.slice(0, dot);
  const suffix = token.slice(dot + 1);
  if (/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(suffix) && (existingNodeIds.has(base) || resolveFlowModelAlias(base, catalog))) {
    return { base, port: suffix };
  }
  return { base: token };
}

function parseNodeExpression(
  rawExpr: string,
  existingNodeIds: ReadonlySet<string>,
  catalog?: FlowCatalogHooks,
): { parsed?: ParsedNodeExpr; error?: FlowValidationError } {
  const extracted = extractTrailingParenParams(rawExpr);
  if (!extracted) {
    return {
      error: {
        code: 'INVALID_NODE_SYNTAX',
        message: `Unbalanced parentheses in node expression "${rawExpr.trim()}".`,
        token: rawExpr.trim(),
      },
    };
  }

  const { head, parenParams } = extracted;
  if (!head) {
    return {
      error: {
        code: 'INVALID_NODE_SYNTAX',
        message: `Missing node or model identifier in "${rawExpr.trim()}".`,
        token: rawExpr.trim(),
      },
    };
  }

  const segs = splitOutsideQuotes(head, ':');
  let explicitId: string | undefined;
  let modelToken: string | undefined;
  let refToken = '';
  let port: string | undefined;
  let colonParams = '';

  if (segs.length === 1) {
    const split = splitPortSuffix(segs[0].trim(), existingNodeIds, catalog);
    refToken = split.base;
    port = split.port;
  } else {
    if (!segs[0].trim()) {
      return {
        error: {
          code: 'INVALID_NODE_SYNTAX',
          message: `Empty identifier before ":" in "${rawExpr.trim()}".`,
          token: rawExpr.trim(),
        },
      };
    }
    if (!segs[1].trim()) {
      return {
        error: {
          code: 'INVALID_NODE_SYNTAX',
          message: `Empty model/parameter segment after ":" in "${rawExpr.trim()}".`,
          token: rawExpr.trim(),
        },
      };
    }
    // Distinguish `model:k=v,...` from `id:model:k=v,...` and `id:model,k=v,...`
    const seg1FirstEq = segs[1].indexOf('=');
    const seg1FirstComma = segs[1].indexOf(',');
    const seg1StartsWithParam = seg1FirstEq > 0 && (seg1FirstComma === -1 || seg1FirstEq < seg1FirstComma);
    if (seg1StartsWithParam) {
      const split = splitPortSuffix(segs[0].trim(), existingNodeIds, catalog);
      refToken = split.base;
      port = split.port;
      colonParams = segs.slice(1).join(':').trim();
    } else if (seg1FirstComma > 0 && (seg1FirstEq === -1 || seg1FirstComma < seg1FirstEq)) {
      explicitId = segs[0].trim();
      const restAfterId = segs.slice(1).join(':');
      const modelPart = restAfterId.slice(0, seg1FirstComma).trim();
      const split = splitPortSuffix(modelPart, new Set<string>(), catalog);
      modelToken = split.base;
      port = split.port;
      refToken = explicitId;
      colonParams = restAfterId.slice(seg1FirstComma + 1).trim();
      if (!colonParams) {
        return {
          error: {
            code: 'INVALID_NODE_SYNTAX',
            message: `Empty parameter segment after "," in "${rawExpr.trim()}".`,
            token: rawExpr.trim(),
          },
        };
      }
    } else {
      explicitId = segs[0].trim();
      const split = splitPortSuffix(segs[1].trim(), new Set<string>(), catalog);
      modelToken = split.base;
      port = split.port;
      refToken = explicitId;
      colonParams = segs.slice(2).join(':').trim();
      if (segs.length >= 3 && !colonParams) {
        return {
          error: {
            code: 'INVALID_NODE_SYNTAX',
            message: `Empty parameter segment after ":" in "${rawExpr.trim()}".`,
            token: rawExpr.trim(),
          },
        };
      }
    }
  }

  const combinedParamStr = [colonParams, parenParams].filter(Boolean).join(',');
  const rawParams: Array<{ key: string; rawVal: string; wasQuoted: boolean }> = [];
  if (combinedParamStr) {
    for (const pair of splitParamPairs(combinedParamStr)) {
      const eqIdx = pair.indexOf('=');
      if (eqIdx <= 0) {
        return {
          error: {
            code: 'INVALID_PARAM_SYNTAX',
            message: `Expected "key=value" parameter syntax in "${pair.trim()}" (on "${rawExpr.trim()}").`,
            token: pair.trim(),
          },
        };
      }
      const key = pair.slice(0, eqIdx).trim();
      const rawRhs = pair.slice(eqIdx + 1);
      const { value: unquotedVal, wasQuoted } = unquoteValue(rawRhs);
      const value =
        PIPE_SEPARABLE_PARAM_KEYS.has(key) && unquotedVal.includes('|')
          ? unquotedVal.replace(/\s*\|\s*/g, '; ')
          : unquotedVal;
      if (FORBIDDEN_KEYS.has(key)) {
        return {
          error: {
            code: 'FORBIDDEN_PARAM_KEY',
            message: `Forbidden parameter key "${key}".`,
            token: key,
          },
        };
      }
      rawParams.push({ key, rawVal: value, wasQuoted });
    }
  }

  return {
    parsed: {
      explicitId,
      modelToken,
      refToken,
      port,
      rawParams,
      rawExpr: rawExpr.trim(),
    },
  };
}

function coerceParamValue(
  manifest: AdapterManifest,
  key: string,
  rawVal: string,
): unknown {
  const field = manifest.paramSchema?.find((f) => f.key === key);
  if (field) {
    if (field.kind === 'integer' || field.kind === 'number') {
      if (rawVal === '') return rawVal;
      const n = Number(rawVal);
      return Number.isFinite(n) ? n : rawVal;
    }
    if (field.kind === 'boolean') {
      const lower = rawVal.toLowerCase();
      if (lower === 'true' || lower === '1') return true;
      if (lower === 'false' || lower === '0') return false;
      return rawVal;
    }
    if (manifest.id === 'localfold-af2' && key === 'model') {
      const lower = rawVal.toLowerCase();
      return Object.hasOwn(LOCALFOLD_MODEL_ALIASES, lower) ? LOCALFOLD_MODEL_ALIASES[lower] : rawVal;
    }
    return rawVal;
  }
  if (rawVal === 'true') return true;
  if (rawVal === 'false') return false;
  return rawVal;
}

function buildParamsAndLabel(
  manifest: AdapterManifest,
  presetParams: Record<string, unknown> | undefined,
  rawParams: Array<{ key: string; rawVal: string }>,
  catalog?: FlowCatalogHooks,
): { params: Record<string, unknown>; label?: string; explicitKeys: Set<string>; errors: FlowValidationError[] } {
  const params: Record<string, unknown> = {
    ...defaultParamsForAdapterEdge(manifest.id, catalog),
    ...(presetParams ?? {}),
  };
  const explicitKeys = new Set<string>(Object.keys(presetParams ?? {}));
  const errors: FlowValidationError[] = [];
  let label: string | undefined;

  const hasFieldNamedMode = Boolean(manifest.paramSchema?.some((f) => f.key === 'mode'));

  for (const { key: rawKey, rawVal } of rawParams) {
    if (rawKey === '_label') {
      if (rawVal.trim()) label = rawVal.trim();
      continue;
    }
    if (rawKey === 'seqLabel' || rawKey === 'sequenceLabel') {
      if (manifest.id === 'sequence-input') {
        params.label = rawVal;
        explicitKeys.add('label');
      } else {
        errors.push({
          code: 'INVALID_PARAM',
          message: `"${rawKey}" is only supported on sequence-input (use "_label=" or "label=" for node display labels).`,
          token: rawKey,
        });
      }
      continue;
    }
    if (rawKey === 'label') {
      if (manifest.id === 'sequence-input') {
        params.label = rawVal;
        explicitKeys.add('label');
      } else if (rawVal.trim()) {
        label = rawVal.trim();
      }
      continue;
    }

    let key = rawKey;
    if (manifest.id === 'localfold-af2' && (rawKey === 'foldModel' || rawKey === 'checkpoint')) {
      key = 'model';
    } else if (rawKey === 'mode' && manifest.modeParamKey && !hasFieldNamedMode) {
      key = manifest.modeParamKey;
    } else if (manifest.stage === 'generate' && STEP_ALIAS_KEYS.has(rawKey)) {
      const hasExactStep = manifest.paramSchema?.some((f) => f.key === rawKey);
      if (!hasExactStep) {
        const canonicalStep = manifest.paramSchema?.find((f) => STEP_ALIAS_KEYS.has(f.key));
        if (canonicalStep) key = canonicalStep.key;
      }
    }

    const val = coerceParamValue(manifest, key, rawVal);
    params[key] = val;
    explicitKeys.add(key);
  }

  // Mode & capMode inference
  if (manifest.id === 'pdb-input') {
    if (typeof params.pdbId === 'string' && params.pdbId.trim()) {
      params.pdbId = params.pdbId.trim().toUpperCase();
      if (!explicitKeys.has('mode')) params.mode = 'pdb-id';
    } else if (typeof params.uniprotId === 'string' && params.uniprotId.trim()) {
      params.uniprotId = params.uniprotId.trim().toUpperCase();
      if (!explicitKeys.has('mode')) params.mode = 'alphafold-db';
    }
  }

  if (manifest.stage === 'generate' && manifest.modeParamKey && manifest.modes?.length) {
    const modeKey = manifest.modeParamKey;
    if (!explicitKeys.has(modeKey)) {
      const modeIds = new Set(manifest.modes.map((m) => m.id));
      const hasNonEmpty = (k: string) => typeof params[k] === 'string' && safeStr(params[k]).trim().length > 0;
      const contigStr = safeStr(params.contig).trim();
      if (
        modeIds.has('binder') &&
        (hasNonEmpty('hotspots') || hasNonEmpty('lockChain') || hasNonEmpty('target') || hasNonEmpty('ligand') || contigStr.includes('/0'))
      ) {
        params[modeKey] = 'binder';
      } else if (
        modeIds.has('partial') &&
        (explicitKeys.has('partialFraction') || explicitKeys.has('partialT') || explicitKeys.has('rewindSteps'))
      ) {
        params[modeKey] = 'partial';
      } else if (
        modeIds.has('conditioned') &&
        (hasNonEmpty('cathCode') || hasNonEmpty('customSpecYaml'))
      ) {
        params[modeKey] = 'conditioned';
      } else if (
        modeIds.has('motif') &&
        (hasNonEmpty('motifSpan') ||
          hasNonEmpty('motifSelection') ||
          hasNonEmpty('motifAtoms') ||
          hasNonEmpty('unindex') ||
          params.unindexedMotif === true ||
          hasNonEmpty('fixedAtoms') ||
          contigStr.length > 0)
      ) {
        params[modeKey] = 'motif';
      } else if (
        modeIds.has('symmetry') &&
        explicitKeys.has('symmetry') &&
        safeStr(params.symmetry ?? 'none').toLowerCase() !== 'none'
      ) {
        params[modeKey] = 'symmetry';
      } else if (modeIds.has('preset') && explicitKeys.has('presetId')) {
        params[modeKey] = 'preset';
      }
    }
  }

  if (
    (manifest.id === 'localfold-af2' || manifest.id === 'esmc' || manifest.id === 'dyna1') &&
    explicitKeys.has('maxSequences') &&
    !explicitKeys.has('capMode')
  ) {
    params.capMode = 'fixed';
  }

  return { params, label, explicitKeys, errors };
}

function sanitizeNodeIdBase(token: string): string {
  const cleaned = token.replace(/[^A-Za-z0-9_.-]/g, '_');
  return cleaned || 'node';
}

function allocateUniqueNodeId(base: string, existingIds: ReadonlySet<string>): string {
  const cleanBase = sanitizeNodeIdBase(base);
  if (!existingIds.has(cleanBase) && !FORBIDDEN_KEYS.has(cleanBase)) return cleanBase;
  let idx = 2;
  while (existingIds.has(`${cleanBase}_${idx}`)) idx++;
  return `${cleanBase}_${idx}`;
}

function resolveFlowEdgePorts(
  fromNode: WorkflowNode,
  toNode: WorkflowNode,
  explicitFromPort: string | undefined,
  explicitToPort: string | undefined,
  edges: WorkflowEdge[],
  catalog?: FlowCatalogHooks,
): { edge?: WorkflowEdge; error?: FlowValidationError } {
  const fromManifest = resolveManifest(fromNode.adapterId, catalog);
  const toManifest = resolveManifest(toNode.adapterId, catalog);
  if (!fromManifest || !toManifest) {
    return {
      error: {
        code: 'UNKNOWN_MODEL',
        message: `Cannot wire "${fromNode.id}" -> "${toNode.id}": unknown model.`,
      },
    };
  }

  if (explicitFromPort && !Object.hasOwn(fromManifest.outputs, explicitFromPort)) {
    return {
      error: {
        code: 'UNKNOWN_PORT',
        message: `Node "${fromNode.id}" (${fromManifest.shortName}) has no output port "${explicitFromPort}" (available: ${Object.keys(fromManifest.outputs).join(', ') || 'none'}).`,
        nodeId: fromNode.id,
        token: explicitFromPort,
      },
    };
  }
  if (explicitToPort && !Object.hasOwn(toManifest.inputs, explicitToPort)) {
    return {
      error: {
        code: 'UNKNOWN_PORT',
        message: `Node "${toNode.id}" (${toManifest.shortName}) has no input port "${explicitToPort}" (available: ${Object.keys(toManifest.inputs).join(', ') || 'none'}).`,
        nodeId: toNode.id,
        token: explicitToPort,
      },
    };
  }

  const isTargetFree = (toPort: string): boolean => {
    const spec = toManifest.inputs[toPort];
    if (!spec) return false;
    const count = edges.filter((e) => e.toNode === toNode.id && e.toPort === toPort).length;
    return spec.multiple ? count < MAX_MULTI_INPUT_EDGES : count === 0;
  };

  if (explicitFromPort && explicitToPort) {
    const outSpec = fromManifest.outputs[explicitFromPort];
    const inSpec = toManifest.inputs[explicitToPort];
    if (!arePortSpecsCompatible(outSpec, inSpec)) {
      return {
        error: {
          code: 'INCOMPATIBLE_PORTS',
          message: `Incompatible ports: "${fromNode.id}.${explicitFromPort}" (${outSpec.type}) cannot connect to "${toNode.id}.${explicitToPort}" (${inSpec.type}).`,
          nodeId: toNode.id,
        },
      };
    }
    return {
      edge: {
        fromNode: fromNode.id,
        fromPort: explicitFromPort,
        toNode: toNode.id,
        toPort: explicitToPort,
      },
    };
  }

  // Build ordered candidate pairs honoring semantic preferences
  const outKeys = explicitFromPort ? [explicitFromPort] : Object.keys(fromManifest.outputs);
  const inKeys = explicitToPort ? [explicitToPort] : Object.keys(toManifest.inputs);

  const preferredPairs: Array<[string, string]> = [];
  if (!explicitFromPort && !explicitToPort) {
    if (toNode.adapterId === 'designability-scorer') {
      if (fromNode.adapterId === 'localfold-af2') preferredPairs.push(['folds', 'folds']);
      preferredPairs.push(['folds', 'folds'], ['structure', 'structure']);
    } else if (toNode.adapterId === 'structure-compare') {
      if (fromNode.adapterId === 'pdb-input' && isTargetFree('reference')) {
        preferredPairs.push(['structure', 'reference']);
      }
      preferredPairs.push(['structure', 'structures']);
    } else if (toNode.adapterId === 'genie2') {
      preferredPairs.push(['structure', 'template']);
    } else if (
      toNode.adapterId === 'localfold-af2' ||
      toNode.adapterId === 'evodiff' ||
      toNode.adapterId === 'foldflow2' ||
      toNode.adapterId === 'top-k-filter'
    ) {
      preferredPairs.push(['sequences', 'sequences']);
    } else if (toNode.adapterId === 'esmc') {
      if (fromNode.adapterId === 'pdb-input') {
        preferredPairs.push(['structure', 'structure'], ['sequences', 'sequences']);
      } else {
        preferredPairs.push(['sequences', 'sequences'], ['structure', 'structure']);
      }
    } else if (toNode.adapterId === 'dyna1') {
      preferredPairs.push(['structure', 'structure'], ['sequences', 'sequences']);
    } else {
      preferredPairs.push(['structure', 'structure'], ['sequences', 'sequences']);
    }
  }

  // Append all compatible fallback pairs (required inputs first)
  const filteredInKeys =
    toNode.adapterId === 'structure-compare' && !explicitToPort && fromNode.adapterId !== 'pdb-input'
      ? inKeys.filter((k) => k !== 'reference')
      : inKeys;
  const orderedInKeys = [...filteredInKeys].sort((a, b) => {
    const reqA = toManifest.inputs[a]?.required === false ? 1 : 0;
    const reqB = toManifest.inputs[b]?.required === false ? 1 : 0;
    return reqA - reqB;
  });
  for (const ok of outKeys) {
    for (const ik of orderedInKeys) {
      preferredPairs.push([ok, ik]);
    }
  }

  const seen = new Set<string>();
  const compatiblePairs: Array<[string, string]> = [];
  for (const [fp, tp] of preferredPairs) {
    const k = `${fp}->${tp}`;
    if (seen.has(k)) continue;
    seen.add(k);
    const outSpec = fromManifest.outputs[fp];
    const inSpec = toManifest.inputs[tp];
    if (outSpec && inSpec && arePortSpecsCompatible(outSpec, inSpec)) {
      compatiblePairs.push([fp, tp]);
    }
  }

  if (compatiblePairs.length === 0) {
    const outSummary = outKeys.map((k) => `${k}:${fromManifest.outputs[k]?.type}`).join(', ') || 'none';
    const inSummary = inKeys.map((k) => `${k}:${toManifest.inputs[k]?.type}`).join(', ') || 'none';
    const hint =
      fromManifest.stage === 'generate' && !outKeys.some((k) => fromManifest.outputs[k]?.type === 'SequenceSet') && toNode.adapterId === 'localfold-af2'
        ? ' Hint: insert an inverse-folding block (e.g. "~mpnn~") before LocalFold.'
        : '';
    return {
      error: {
        code: 'INCOMPATIBLE_PORTS',
        message: `Cannot auto-wire "${fromNode.id}" (${fromManifest.shortName}; outputs [${outSummary}]) to "${toNode.id}" (${toManifest.shortName}; inputs [${inSummary}]).${hint}`,
        nodeId: toNode.id,
      },
    };
  }

  // Prefer a pair whose target port is free; otherwise return the top compatible pair so conflict validation reports clearly
  const freePair = compatiblePairs.find(([, tp]) => isTargetFree(tp)) ?? compatiblePairs[0];
  return {
    edge: {
      fromNode: fromNode.id,
      fromPort: freePair[0],
      toNode: toNode.id,
      toPort: freePair[1],
    },
  };
}

function findUpstreamScorerStructureSource(
  scorerId: string,
  nodesById: ReadonlyMap<string, WorkflowNode>,
  edges: WorkflowEdge[],
  catalog?: FlowCatalogHooks,
): string | null {
  const visited = new Set<string>();
  const queue = edges.filter((e) => e.toNode === scorerId).map((e) => e.fromNode);
  let fallbackPdb: string | null = null;
  let fallbackOther: string | null = null;

  while (queue.length > 0) {
    const cur = queue.shift()!;
    if (visited.has(cur)) continue;
    visited.add(cur);
    const cn = nodesById.get(cur);
    if (cn) {
      const m = resolveManifest(cn.adapterId, catalog);
      if (m && Object.hasOwn(m.outputs, 'structure') && cn.adapterId !== 'localfold-af2') {
        if (m.stage === 'generate') {
          return cur;
        }
        if (cn.adapterId === 'pdb-input' && !fallbackPdb) {
          fallbackPdb = cur;
        } else if (!fallbackOther && m.stage !== 'score') {
          fallbackOther = cur;
        }
      }
    }
    for (const pe of edges) {
      if (pe.toNode === cur && !visited.has(pe.fromNode)) {
        queue.push(pe.fromNode);
      }
    }
  }
  return fallbackPdb ?? fallbackOther;
}

/** Parse a Markdown-safe `?flow=` DSL string into a validated, auto-wired, laid-out WorkflowGraph. */
export function parseFlowDsl(dsl: string, opts: FlowParseOptions = {}): FlowParseResult {
  const catalog = opts.catalog;
  const trimmed = stripTrailingProsePunctuation(dsl);
  if (!trimmed) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [{ code: 'EMPTY_FLOW', message: 'Flow DSL is empty.' }],
    };
  }

  const statements = splitStatements(trimmed, catalog)
    .map((s) => s.trim())
    .filter(Boolean);

  if (statements.length === 0) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [{ code: 'EMPTY_FLOW', message: 'Flow DSL contains no statements.' }],
    };
  }

  const nodes: WorkflowNode[] = [];
  const nodesById = new Map<string, WorkflowNode>();
  const existingNodeIds = new Set<string>();
  const explicitParamNodes = new Set<string>();
  const explicitMpnnFamilyNodes = new Set<string>();
  const explicitIdNodes = new Set<string>();
  const edges: WorkflowEdge[] = [];
  const errors: FlowValidationError[] = [];

  for (const stmt of statements) {
    const chain = splitEdgeChain(stmt);
    if (chain.some((part) => !part.trim())) {
      errors.push({
        code: 'INVALID_EDGE_SYNTAX',
        message: `Empty node expression around edge operator in "${stmt}".`,
        token: stmt,
      });
      continue;
    }

    let prevNode: WorkflowNode | null = null;
    let prevOutPort: string | undefined;

    for (let idx = 0; idx < chain.length; idx++) {
      const rawToken = chain[idx].trim();
      const { parsed, error } = parseNodeExpression(rawToken, existingNodeIds, catalog);
      if (error || !parsed) {
        errors.push(error!);
        prevNode = null;
        prevOutPort = undefined;
        break;
      }

      const isTarget = idx > 0;
      const hasParams = parsed.rawParams.length > 0;
      let curNode: WorkflowNode | undefined;

      if (parsed.explicitId) {
        if (FORBIDDEN_KEYS.has(parsed.explicitId)) {
          errors.push({
            code: 'INVALID_NODE_ID',
            message: `Reserved node ID "${parsed.explicitId}".`,
            nodeId: parsed.explicitId,
          });
          break;
        }
        const resolved = resolveFlowModelAlias(parsed.modelToken!, catalog);
        if (!resolved) {
          errors.push({
            code: 'UNKNOWN_MODEL',
            message: `Unknown model "${parsed.modelToken}" on node "${parsed.explicitId}".`,
            token: parsed.modelToken,
            nodeId: parsed.explicitId,
          });
          break;
        }
        if (nodesById.has(parsed.explicitId)) {
          errors.push({
            code: 'DUPLICATE_NODE_ID',
            message: `Duplicate node ID "${parsed.explicitId}".`,
            nodeId: parsed.explicitId,
          });
          break;
        }
        const manifest = resolveManifest(resolved.adapterId, catalog)!;
        const { params, label, explicitKeys, errors: paramErrs } = buildParamsAndLabel(
          manifest,
          resolved.presetParams,
          parsed.rawParams,
          catalog,
        );
        errors.push(...paramErrs);
        curNode = {
          id: parsed.explicitId,
          adapterId: manifest.id,
          label: label ?? manifest.shortName,
          params,
        };
        nodes.push(curNode);
        nodesById.set(curNode.id, curNode);
        existingNodeIds.add(curNode.id);
        explicitIdNodes.add(curNode.id);
        if (hasParams) explicitParamNodes.add(curNode.id);
        if (explicitKeys.has('family')) explicitMpnnFamilyNodes.add(curNode.id);
      } else {
        // No explicit `id:` prefix: either reference an existing node or instantiate a model
        const existingByRef = nodesById.get(parsed.refToken);
        const resolved = resolveFlowModelAlias(parsed.refToken, catalog);

        if (existingByRef && !hasParams) {
          // If in target position and existing node was auto-created from a model alias
          // (not an explicit custom ID) and cannot accept another incoming wire on its compatible port,
          // allocate a fresh sibling node (e.g. `rfd3~mpnn;rfd4~mpnn` creates `mpnn_2`).
          if (isTarget && prevNode && resolved && !explicitIdNodes.has(existingByRef.id)) {
            const trial = resolveFlowEdgePorts(
              prevNode,
              existingByRef,
              prevOutPort,
              parsed.port,
              edges,
              catalog,
            );
            if (trial.edge) {
              const dstManifest = resolveManifest(existingByRef.adapterId, catalog);
              const inSpec = dstManifest?.inputs[trial.edge.toPort];
              const exactExists = edges.some(
                (e) =>
                  e.fromNode === trial.edge!.fromNode &&
                  e.fromPort === trial.edge!.fromPort &&
                  e.toNode === trial.edge!.toNode &&
                  e.toPort === trial.edge!.toPort,
              );
              const occupied = edges.filter(
                (e) => e.toNode === existingByRef.id && e.toPort === trial.edge!.toPort,
              ).length;
              if (!exactExists && !inSpec?.multiple && occupied >= 1) {
                const manifest = resolveManifest(resolved.adapterId, catalog)!;
                const newId = allocateUniqueNodeId(parsed.refToken, existingNodeIds);
                const { params, label, explicitKeys, errors: paramErrs } = buildParamsAndLabel(
                  manifest,
                  resolved.presetParams,
                  [],
                  catalog,
                );
                errors.push(...paramErrs);
                curNode = {
                  id: newId,
                  adapterId: manifest.id,
                  label: label ?? manifest.shortName,
                  params,
                };
                nodes.push(curNode);
                nodesById.set(curNode.id, curNode);
                existingNodeIds.add(curNode.id);
                if (explicitKeys.has('family')) explicitMpnnFamilyNodes.add(curNode.id);
              } else {
                curNode = existingByRef;
              }
            } else {
              curNode = existingByRef;
            }
          } else {
            curNode = existingByRef;
          }
        } else if (existingByRef && hasParams && !resolved) {
          errors.push({
            code: 'DUPLICATE_NODE_ID',
            message: `Node "${parsed.refToken}" is already defined; omit parameters when referencing it in downstream edges.`,
            nodeId: parsed.refToken,
          });
          break;
        } else if (resolved) {
          const manifest = resolveManifest(resolved.adapterId, catalog)!;
          const newId = allocateUniqueNodeId(parsed.refToken, existingNodeIds);
          const { params, label, explicitKeys, errors: paramErrs } = buildParamsAndLabel(
            manifest,
            resolved.presetParams,
            parsed.rawParams,
            catalog,
          );
          errors.push(...paramErrs);
          curNode = {
            id: newId,
            adapterId: manifest.id,
            label: label ?? manifest.shortName,
            params,
          };
          nodes.push(curNode);
          nodesById.set(curNode.id, curNode);
          existingNodeIds.add(curNode.id);
          if (hasParams) explicitParamNodes.add(curNode.id);
          if (explicitKeys.has('family')) explicitMpnnFamilyNodes.add(curNode.id);
        } else {
          errors.push({
            code: 'UNKNOWN_MODEL',
            message: `Unknown model or undefined node "${parsed.refToken}" in "${rawToken}".`,
            token: parsed.refToken,
          });
          break;
        }
      }

      if (isTarget && prevNode && curNode) {
        if (prevNode.id === curNode.id) {
          errors.push({
            code: 'SELF_LOOP',
            message: `Cannot connect node "${curNode.id}" to itself.`,
            nodeId: curNode.id,
          });
        } else {
          const resolvedEdge = resolveFlowEdgePorts(
            prevNode,
            curNode,
            prevOutPort,
            parsed.port,
            edges,
            catalog,
          );
          if (resolvedEdge.error || !resolvedEdge.edge) {
            errors.push(resolvedEdge.error!);
          } else {
            const e = resolvedEdge.edge;
            const exactExists = edges.some(
              (x) =>
                x.fromNode === e.fromNode &&
                x.fromPort === e.fromPort &&
                x.toNode === e.toNode &&
                x.toPort === e.toPort,
            );
            if (!exactExists) {
              if (wouldCreateCycle(nodes, edges, e.fromNode, e.toNode)) {
                errors.push({
                  code: 'CYCLE_DETECTED',
                  message: `Connection ${e.fromNode}.${e.fromPort} -> ${e.toNode}.${e.toPort} introduces a cycle.`,
                  nodeId: e.toNode,
                });
              } else {
                edges.push(e);
              }
            }
          }
        }
        // Only treat `parsed.port` as the input port for the incoming edge;
        // if this node continues in a chain (`a~b.in~c`), `b`'s outgoing port defaults unless a chain node overrides it.
        prevOutPort = undefined;
      } else {
        prevOutPort = parsed.port;
      }

      prevNode = curNode ?? null;
    }
  }

  // Post-edge pass 1: Cα-only generators -> web-protein-mpnn family inference
  for (const e of edges) {
    const src = nodesById.get(e.fromNode);
    const dst = nodesById.get(e.toNode);
    if (
      src &&
      dst &&
      dst.adapterId === 'web-protein-mpnn' &&
      CA_ONLY_GENERATORS.has(src.adapterId) &&
      !explicitMpnnFamilyNodes.has(dst.id)
    ) {
      dst.params.family = 'ca';
    }
  }

  // Post-edge pass 2: designability-scorer dual-input lineage pass
  for (const node of nodes) {
    if (node.adapterId !== 'designability-scorer') continue;
    const hasStruct = edges.some((e) => e.toNode === node.id && e.toPort === 'structure');
    if (hasStruct) continue;
    const hasFolds = edges.some((e) => e.toNode === node.id && e.toPort === 'folds');
    if (!hasFolds) continue;
    const srcId = findUpstreamScorerStructureSource(node.id, nodesById, edges, catalog);
    if (srcId) {
      edges.push({
        fromNode: srcId,
        fromPort: 'structure',
        toNode: node.id,
        toPort: 'structure',
      });
    }
  }

  const rawGraph: WorkflowGraph = {
    id: opts.id ?? 'shared-workflow',
    name: opts.name ?? 'Shared Workflow',
    nodes,
    edges,
  };

  const validationErrors = validateGraphEdge(rawGraph, {
    requireRunnable: opts.requireRunnable !== false,
    catalog,
  });
  errors.push(...validationErrors);

  if (errors.length > 0) {
    return {
      ok: false,
      graph: null,
      nodes: summarizeNodes(nodes, catalog),
      edges: summarizeEdges(edges),
      errors: sanitizeFlowErrors(errors),
    };
  }

  const laidOut = autoLayoutGraphEdge(rawGraph, catalog);
  return {
    ok: true,
    graph: laidOut,
    nodes: summarizeNodes(laidOut.nodes, catalog),
    edges: summarizeEdges(laidOut.edges),
    errors: [],
  };
}

export const parseFlowToGraph = parseFlowDsl;

const WORKFLOW_QUERY_KEYS = ['flow', 'recipe', 'p', 'g'] as const;

function hasAnyWorkflowKey(cleanQuery: string, keys: readonly string[] = WORKFLOW_QUERY_KEYS): boolean {
  if (!cleanQuery || !cleanQuery.includes('=')) return false;
  const qs = new URLSearchParams(cleanQuery);
  return keys.some((k) => qs.has(k));
}

/** Extract the active workflow query or hash parameter payload, supporting hybrid `?utm_...#flow=` and relative `/?...` URLs. */
export function extractWorkflowSearchOrHash(
  rawInput: string,
  keys: readonly string[] = WORKFLOW_QUERY_KEYS,
): { clean: string; invalidUrl?: boolean } {
  const trimmed = rawInput.trim();
  if (!trimmed) return { clean: '' };
  if (/^https?:\/\//i.test(trimmed) || (trimmed.startsWith('/') && !trimmed.startsWith('//'))) {
    try {
      const u = trimmed.startsWith('/')
        ? new URL(trimmed, 'https://denovo.run')
        : new URL(trimmed);
      const searchClean = u.search.replace(/^\?/, '').trim();
      const hashClean = u.hash.replace(/^#/, '').trim();
      if (hasAnyWorkflowKey(searchClean, keys)) {
        return { clean: searchClean };
      }
      if (hasAnyWorkflowKey(hashClean, keys)) {
        return { clean: hashClean };
      }
      return { clean: searchClean || hashClean };
    } catch {
      return { clean: '', invalidUrl: true };
    }
  }
  if (trimmed.startsWith('?') && trimmed.includes('#')) {
    const hashIdx = trimmed.indexOf('#');
    const searchClean = trimmed.slice(1, hashIdx).trim();
    const hashClean = trimmed.slice(hashIdx + 1).trim();
    if (hasAnyWorkflowKey(searchClean, keys)) {
      return { clean: searchClean };
    }
    if (hasAnyWorkflowKey(hashClean, keys)) {
      return { clean: hashClean };
    }
    return { clean: searchClean || hashClean };
  }
  return { clean: trimmed.replace(/^[#?]/, '').trim() };
}

function extractRawQueryParam(clean: string, targetKey: string): string | null {
  const qs = new URLSearchParams(clean.replace(/^[?#]/, ''));
  if (!qs.has(targetKey)) return null;
  return qs.get(targetKey) ?? '';
}

/** Decode a `?flow=...`, `#flow=...`, full URL, or multi-param query string via `parseFlowDsl`. */
export function decodeFlowQuery(rawQueryOrUrl: string, opts: FlowParseOptions = {}): FlowParseResult {
  const trimmed = rawQueryOrUrl.trim();
  if (!trimmed) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [{ code: 'EMPTY_FLOW', message: 'Flow query is empty.' }],
    };
  }

  const { clean, invalidUrl } = extractWorkflowSearchOrHash(trimmed, ['flow']);
  if (invalidUrl) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [{ code: 'INVALID_URL', message: 'Invalid workflow URL.' }],
    };
  }
  if (clean.length > MAX_HASH_CHARS) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [
        {
          code: 'FLOW_TOO_LARGE',
          message: `Shared workflow link is too large (max ${MAX_HASH_CHARS} chars).`,
        },
      ],
    };
  }

  const rawFlow = extractRawQueryParam(clean, 'flow');
  if (rawFlow === null) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [{ code: 'MISSING_FLOW_PARAM', message: 'URL does not contain a "flow" parameter.' }],
    };
  }

  return parseFlowDsl(rawFlow, opts);
}

// ---------------------------------------------------------------------------
// Markdown-Safe ?flow= Encoder & Autolinker Safety Validator
// ---------------------------------------------------------------------------

function diffParamsEdge(
  adapterId: string,
  params: Record<string, unknown>,
  base?: Record<string, unknown>,
  stripLargePdb = false,
  catalog?: FlowCatalogHooks,
): { diff: Record<string, unknown>; strippedPdb: boolean } {
  const defaults = base ?? defaultParamsForAdapterEdge(adapterId, catalog);
  const stripKeys = catalog?.stripOnShareKeys?.(adapterId);
  const stripKeySet = stripKeys ? new Set(stripKeys) : null;
  let effectiveParams = params;
  let pluginStripped = false;
  if (catalog?.prepareForShare) {
    const prep = catalog.prepareForShare(adapterId, params, { stripLargeUploads: stripLargePdb });
    if (prep) {
      effectiveParams = prep.params;
      pluginStripped = Boolean(prep.stripped);
    }
  }
  const diff: Record<string, unknown> = {};
  let strippedPdb = pluginStripped;
  let strippedPdbText: string | null = null;
  const effectivePdbMode =
    adapterId === 'pdb-input' ? safeStr(effectiveParams.mode ?? defaults.mode ?? 'demo-bundle') || 'demo-bundle' : '';
  const uploadMetaKeys = new Set(['missingUpload', 'expectedPdbSha256', 'expectedFilename']);

  for (const [k, v] of Object.entries(effectiveParams)) {
    if (FORBIDDEN_KEYS.has(k)) continue;
    if (stripKeySet?.has(k)) continue;
    if (adapterId === 'pdb-input' && !catalog?.prepareForShare) {
      if (k === 'fetchedPdbId' || k === 'fetchedUniProtId') continue;
      if (effectivePdbMode === 'pdb-id' || effectivePdbMode === 'alphafold-db' || effectivePdbMode === 'demo-bundle') {
        if (k === 'pdbText' || uploadMetaKeys.has(k)) continue;
      } else if (effectivePdbMode === 'custom') {
        if (uploadMetaKeys.has(k)) continue;
        if (k === 'pdbText' && typeof v === 'string' && v.length > 1500 && stripLargePdb) {
          strippedPdb = true;
          strippedPdbText = v;
          continue;
        }
      }
    } else if (!catalog?.prepareForShare && k === 'pdbText' && typeof v === 'string' && v.length > 1500 && stripLargePdb) {
      strippedPdb = true;
      continue;
    }

    if (JSON.stringify(v) !== JSON.stringify(defaults[k])) {
      diff[k] = v;
    }
  }

  if (adapterId === 'pdb-input' && !catalog?.prepareForShare && effectivePdbMode === 'custom') {
    const existingShaRaw = safeStr(params.expectedPdbSha256).trim().toLowerCase();
    const existingExpName = safeStr(params.expectedFilename).trim();
    const curFilename = safeStr(params.filename).trim();
    if (strippedPdb && stripLargePdb && strippedPdbText !== null) {
      if (defaults.mode !== 'custom') diff.mode = 'custom';
      if (typeof defaults.pdbText === 'string' && defaults.pdbText.length > 0) diff.pdbText = '';
      diff.missingUpload = true;
      diff.expectedPdbSha256 = existingShaRaw || computeUploadDigestEdge(strippedPdbText);
      const expName = existingExpName || curFilename;
      if (expName) diff.expectedFilename = expName;
    } else {
      const rawPdb = typeof params.pdbText === 'string' ? params.pdbText : '';
      const hasMissingFlag = Boolean(params.missingUpload);
      if (!rawPdb.trim() && (hasMissingFlag || existingShaRaw)) {
        strippedPdb = true;
        if (defaults.mode !== 'custom') diff.mode = 'custom';
        if (hasMissingFlag) diff.missingUpload = true;
        if (existingShaRaw) diff.expectedPdbSha256 = existingShaRaw;
        const expName = existingExpName || curFilename;
        if (expName) diff.expectedFilename = expName;
      } else if (rawPdb.trim() && existingShaRaw) {
        const actualSha = computeUploadDigestEdge(rawPdb);
        if (!SHA256_HEX_RE.test(existingShaRaw) || actualSha !== existingShaRaw) {
          if (hasMissingFlag) diff.missingUpload = true;
          diff.expectedPdbSha256 = existingShaRaw;
          if (existingExpName) diff.expectedFilename = existingExpName;
        }
      }
    }
  }

  return { diff, strippedPdb };
}

function formatDslParamValue(val: unknown): string {
  const raw =
    typeof val === 'string'
      ? val
      : typeof val === 'number' || typeof val === 'boolean'
        ? String(val)
        : val === undefined
          ? ''
          : (JSON.stringify(val) ?? '');
  const needsQuotes =
    raw === '' ||
    /[\s;"'()[\]{}<>~&=+#%?\\]|->/.test(raw) ||
    /[.,;!?]$/.test(raw) ||
    /,\s*[A-Za-z_][A-Za-z0-9_]*\s*=/.test(raw);
  if (!needsQuotes) return raw;
  return `"${raw.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
}

function encodeFlowQueryValue(dsl: string): string {
  let out = '';
  for (const ch of dsl) {
    if (ch === '~' && out.endsWith('~')) {
      // Avoid `~~` runs inside values that GFM could interpret as strikethrough
      out += '%7E';
    } else if (/^[A-Za-z0-9\-._~:;=,]$/.test(ch)) {
      out += ch;
    } else if ((ch.codePointAt(0) ?? 0) < 128) {
      // Explicitly percent-encode ASCII punctuation like `( ) ' ! *` that encodeURIComponent leaves unencoded
      out += '%' + ch.charCodeAt(0).toString(16).toUpperCase().padStart(2, '0');
    } else {
      out += encodeURIComponent(ch);
    }
  }
  // Never end a bare URL in trailing `_`, `-`, `.`, `,`, `;`, `:` (autolinkers strip them).
  if (out.length > 0 && /[._,;\-:]/.test(out[out.length - 1])) {
    const last = out[out.length - 1];
    out = out.slice(0, -1) + '%' + last.charCodeAt(0).toString(16).toUpperCase().padStart(2, '0');
  }
  return out;
}

/**
 * Encode any WorkflowGraph into a deterministic, Markdown-safe `?flow=...` representation.
 * Emits ONLY `~`, `:`, `=`, `,`, `;`, `.` — ZERO `(`, `)`, `[`, `]`, `<`, `>`, unencoded whitespace,
 * or trailing `_`/`-`/`.`/`,`/`;`/`:`.
 */
export function encodeGraphToFlow(
  graph: WorkflowGraph,
  opts: FlowEncodeOptions = {},
): FlowEncodeResult {
  const catalog = opts.catalog;
  const stripLarge = opts.stripLargeUploads !== false;
  const warnings: string[] = [];
  let strippedPdb = false;

  if (graph.nodes.length === 0) {
    warnings.push('Workflow contains no nodes.');
    return {
      dsl: '',
      flow: '',
      query: '',
      length: 0,
      strippedPdb: false,
      warnings,
    };
  }

  const safeNodes: WorkflowNode[] = graph.nodes.map((n) => ({
    ...n,
    id: safeStr(n.id),
    adapterId: safeStr(n.adapterId),
    label: safeStr(n.label),
    params: n.params && typeof n.params === 'object' && !Array.isArray(n.params) ? n.params : {},
  }));
  const safeEdges: WorkflowEdge[] = graph.edges.map((e) => ({
    ...e,
    fromNode: safeStr(e.fromNode),
    fromPort: safeStr(e.fromPort),
    toNode: safeStr(e.toNode),
    toPort: safeStr(e.toPort),
  }));

  const usedMappedIds = new Set<string>();
  const nodeIdMap = new Map<string, string>();
  for (const n of safeNodes) {
    const nid = safeStr(n.id);
    const cleanId =
      /^[A-Za-z0-9_.-]+$/.test(nid) && !FORBIDDEN_KEYS.has(nid) && !usedMappedIds.has(nid)
        ? nid
        : allocateUniqueNodeId(nid, usedMappedIds);
    usedMappedIds.add(cleanId);
    nodeIdMap.set(nid, cleanId);
  }

  const nodesById = new Map(safeNodes.map((n) => [safeStr(n.id), n]));
  const emittedDefs = new Set<string>();

  const formatNodeDef = (node: WorkflowNode, port?: string): string => {
    const shortCode = resolveShortCode(node.adapterId, catalog);
    const { diff, strippedPdb: s } = diffParamsEdge(node.adapterId, node.params, undefined, stripLarge, catalog);
    if (s) strippedPdb = true;
    const manifest = resolveManifest(node.adapterId, catalog);

    const kvParts: string[] = [];
    const nodeLabel = safeStr(node.label).trim();
    if (nodeLabel && nodeLabel !== manifest?.shortName) {
      kvParts.push(`_label=${formatDslParamValue(nodeLabel)}`);
    }
    for (const [k, v] of Object.entries(diff)) {
      if (v === undefined) continue;
      kvParts.push(`${k}=${formatDslParamValue(v)}`);
    }

    const nid = safeStr(node.id);
    const safeId = nodeIdMap.get(nid) ?? nid;
    const head = safeId === shortCode ? shortCode : `${safeId}:${shortCode}`;
    const headWithPort = port ? `${head}.${port}` : head;
    return kvParts.length > 0 ? `${headWithPort}:${kvParts.join(',')}` : headWithPort;
  };

  const renderEndpoint = (nodeId: string, port: string | undefined): string => {
    const nid = safeStr(nodeId);
    const node = nodesById.get(nid)!;
    const safeId = nodeIdMap.get(nid) ?? nid;
    if (emittedDefs.has(nid)) {
      return port ? `${safeId}.${port}` : safeId;
    }
    emittedDefs.add(nid);
    return formatNodeDef(node, port);
  };

  // Determine which edges require explicit port suffixes or can omit them safely
  const pairEdgeCount = new Map<string, number>();
  for (const e of safeEdges) {
    const k = `${safeStr(e.fromNode)}->${safeStr(e.toNode)}`;
    pairEdgeCount.set(k, (pairEdgeCount.get(k) ?? 0) + 1);
  }

  // Topological edge ordering so linear/branch chains flow cleanly left-to-right
  const indeg = new Map<string, number>();
  const outAdj = new Map<string, WorkflowEdge[]>();
  for (const n of safeNodes) {
    const nid = safeStr(n.id);
    indeg.set(nid, 0);
    outAdj.set(nid, []);
  }
  for (const e of safeEdges) {
    const fromId = safeStr(e.fromNode);
    const toId = safeStr(e.toNode);
    indeg.set(toId, (indeg.get(toId) ?? 0) + 1);
    outAdj.get(fromId)?.push(e);
  }
  const topoOrder: string[] = [];
  const q: string[] = [];
  for (const n of safeNodes) {
    const nid = safeStr(n.id);
    if ((indeg.get(nid) ?? 0) === 0) q.push(nid);
  }
  while (q.length > 0) {
    const cur = q.shift()!;
    topoOrder.push(cur);
    for (const e of outAdj.get(cur) ?? []) {
      const toId = safeStr(e.toNode);
      const nd = (indeg.get(toId) ?? 1) - 1;
      indeg.set(toId, nd);
      if (nd === 0) q.push(toId);
    }
  }
  for (const n of safeNodes) {
    const nid = safeStr(n.id);
    if (!topoOrder.includes(nid)) topoOrder.push(nid);
  }

  const remainingEdges: WorkflowEdge[] = [];
  for (const nid of topoOrder) {
    for (const e of outAdj.get(nid) ?? []) remainingEdges.push(e);
  }

  // Decide port omission per edge using a simulated edge list
  const simulatedEdges: WorkflowEdge[] = [];
  const edgePortFlags = new Map<WorkflowEdge, { showFrom: boolean; showTo: boolean }>();

  for (const e of remainingEdges) {
    const fromId = safeStr(e.fromNode);
    const toId = safeStr(e.toNode);
    const fromPort = safeStr(e.fromPort);
    const toPort = safeStr(e.toPort);
    const src = nodesById.get(fromId);
    const dst = nodesById.get(toId);
    if (!src || !dst) continue;
    const multiBetweenSamePair = (pairEdgeCount.get(`${fromId}->${toId}`) ?? 0) > 1;
    if (multiBetweenSamePair) {
      edgePortFlags.set(e, { showFrom: true, showTo: true });
      simulatedEdges.push(e);
      continue;
    }
    const trial = resolveFlowEdgePorts(src, dst, undefined, undefined, simulatedEdges, catalog);
    const matchesDefault =
      trial.edge && trial.edge.fromPort === fromPort && trial.edge.toPort === toPort;
    edgePortFlags.set(e, { showFrom: !matchesDefault, showTo: !matchesDefault });
    simulatedEdges.push(e);
  }

  // Build greedy chains covering every edge once
  const usedEdge = new Set<WorkflowEdge>();
  const statements: string[] = [];

  for (const startEdge of remainingEdges) {
    if (usedEdge.has(startEdge)) continue;
    usedEdge.add(startEdge);
    const startFlags = edgePortFlags.get(startEdge);
    if (!startFlags) continue;
    let chain = renderEndpoint(
      safeStr(startEdge.fromNode),
      startFlags.showFrom ? safeStr(startEdge.fromPort) : undefined,
    );
    let curEdge: WorkflowEdge = startEdge;

    while (true) {
      const flags = edgePortFlags.get(curEdge);
      if (!flags) break;
      const curToId = safeStr(curEdge.toNode);
      const nextOut = remainingEdges.find(
        (candidate) =>
          !usedEdge.has(candidate) &&
          edgePortFlags.has(candidate) &&
          safeStr(candidate.fromNode) === curToId,
      );
      const nextOutFlags = nextOut ? edgePortFlags.get(nextOut) : undefined;
      const toPort = flags.showTo ? safeStr(curEdge.toPort) : undefined;

      // Only chain seamlessly if nextOut doesn't need a conflicting fromPort on the same node token
      const canChain = Boolean(nextOut && (!nextOutFlags?.showFrom || !toPort));
      if (!canChain || !nextOut) {
        chain += `~${renderEndpoint(curToId, toPort)}`;
        break;
      }
      // If nextOut has showFrom, emit curEdge.toNode without outPort here and start a fresh statement? No:
      // if toPort is undefined and nextOutFlags.showFrom is true, `node.outPort` in middle of chain sets prevOutPort!
      if (nextOutFlags?.showFrom) {
        chain += `~${renderEndpoint(curToId, toPort)}`;
        break;
      }
      chain += `~${renderEndpoint(curToId, toPort)}`;
      usedEdge.add(nextOut);
      curEdge = nextOut;
    }
    statements.push(chain);
  }

  // Emit any isolated nodes not touched by edges
  for (const n of safeNodes) {
    const nid = safeStr(n.id);
    if (!emittedDefs.has(nid)) {
      emittedDefs.add(nid);
      statements.push(formatNodeDef(n));
    }
  }

  if (strippedPdb) {
    warnings.push('Large inline PDB (>1500 chars) omitted; SHA-256 digest metadata preserved.');
  }

  const dsl = statements.join(';');
  const flow = encodeFlowQueryValue(dsl);
  const query = `flow=${flow}`;
  return {
    dsl,
    flow,
    query,
    length: query.length,
    strippedPdb,
    warnings,
  };
}

/** Verify a `?flow=` URL or query string satisfies strict Markdown autolinker safety (GFM / ChatGPT / Slack). */
export function isMarkdownSafeFlowUrl(urlOrQuery: string): boolean {
  const s = urlOrQuery.trim();
  if (!s) return false;
  if (/[\s()[\]<>{}"'`\\]/.test(s)) return false;
  if (/[.,;:!?_~-]$/.test(s)) return false;
  if (/~~/.test(s)) return false;
  if (!/^[A-Za-z0-9:/?#&=~;.,_%+-]+$/.test(s)) return false;

  const { clean: queryPart, invalidUrl } = extractWorkflowSearchOrHash(s, ['flow']);
  if (invalidUrl || !queryPart) return false;
  const qs = new URLSearchParams(queryPart);
  const flowVal = qs.get('flow');
  return Boolean(flowVal && flowVal.trim().length > 0);
}

// ---------------------------------------------------------------------------
// Edge-Safe Recipe Parser (?recipe=) & Compact Share Helpers (?p= / ?g=)
// ---------------------------------------------------------------------------

/** Edge-safe `denovo: v1` YAML or full `WorkflowGraph` JSON parser returning structured FlowParseResult. */
export function parseRecipeSafe(
  text: string,
  opts: { catalog?: FlowCatalogHooks; requireRunnable?: boolean } = {},
): FlowParseResult {
  const catalog = opts.catalog;
  const trimmed = text.trim();
  if (!trimmed) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [{ code: 'EMPTY_RECIPE', message: 'Recipe is empty.' }],
    };
  }
  if (trimmed.length > MAX_INFLATED_BYTES) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [{ code: 'RECIPE_TOO_LARGE', message: 'Recipe exceeds maximum allowed size.' }],
    };
  }

  let rawNodes: WorkflowNode[] = [];
  let rawEdges: WorkflowEdge[] = [];
  let graphId = 'custom-recipe';
  let graphName = 'Custom Recipe';
  let graphDesc: string | undefined;
  let fillDefaults = true;

  try {
    if (trimmed.startsWith('{')) {
      const parsed = JSON.parse(trimmed) as unknown;
      if (!isPlainObject(parsed) || !Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
        return {
          ok: false,
          graph: null,
          nodes: [],
          edges: [],
          errors: [{ code: 'INVALID_RECIPE', message: 'Workflow JSON must contain nodes[] and edges[].' }],
        };
      }
      if (typeof parsed.id === 'string' && parsed.id.trim()) graphId = parsed.id.trim();
      if (typeof parsed.name === 'string' && parsed.name.trim()) graphName = parsed.name.trim();
      if (typeof parsed.description === 'string') graphDesc = parsed.description;

      for (let i = 0; i < parsed.nodes.length; i++) {
        const rn = parsed.nodes[i];
        if (!isPlainObject(rn)) {
          throw new Error(`Invalid node at index ${i}`);
        }
        if (typeof rn.id !== 'string' || !rn.id.trim()) {
          throw new Error(`Node at index ${i} is missing a string id`);
        }
        const id = rn.id.trim();
        if (typeof rn.adapterId !== 'string' || !rn.adapterId) {
          throw new Error(`Node "${id}" is missing a string adapterId`);
        }
        const adapterId = rn.adapterId;
        const manifest = resolveManifest(adapterId, catalog);
        if (!manifest) throw new Error(`Unknown model "${adapterId}" on node "${id}"`);
        const cleaned = isPlainObject(rn.params) ? sanitizeParams(rn.params) : {};
        let merged = { ...defaultParamsForAdapterEdge(adapterId, catalog), ...cleaned };
        if (catalog?.sanitizeOnImport) {
          merged = catalog.sanitizeOnImport(adapterId, merged) ?? merged;
        } else if (adapterId === 'pdb-input') {
          delete merged.fetchedPdbId;
          delete merged.fetchedUniProtId;
          const mode = safeStr(merged.mode ?? 'demo-bundle');
          if ((mode === 'pdb-id' || mode === 'alphafold-db' || mode === 'demo-bundle') && 'pdbText' in merged) {
            merged.pdbText = '';
          }
        }
        const pos =
          isPlainObject(rn.position) &&
          typeof rn.position.x === 'number' &&
          Number.isFinite(rn.position.x) &&
          typeof rn.position.y === 'number' &&
          Number.isFinite(rn.position.y)
            ? { x: rn.position.x, y: rn.position.y }
            : undefined;
        rawNodes.push({
          id,
          adapterId,
          label: typeof rn.label === 'string' && rn.label.trim() ? rn.label : manifest.shortName,
          params: merged,
          ...(pos ? { position: pos } : {}),
        });
      }

      for (let i = 0; i < parsed.edges.length; i++) {
        const re = parsed.edges[i];
        if (!isPlainObject(re)) throw new Error(`Invalid wire at index ${i}`);
        if (
          typeof re.fromNode !== 'string' ||
          typeof re.fromPort !== 'string' ||
          typeof re.toNode !== 'string' ||
          typeof re.toPort !== 'string'
        ) {
          throw new Error(`Wire at index ${i} must have string node and port names`);
        }
        const fromNode = re.fromNode.trim();
        const fromPort = re.fromPort.trim();
        const toNode = re.toNode.trim();
        const toPort = re.toPort.trim();
        if (!fromNode || !fromPort || !toNode || !toPort) {
          throw new Error(`Wire at index ${i} is missing node or port name`);
        }
        rawEdges.push({ fromNode, fromPort, toNode, toPort });
      }
      fillDefaults = false;
    } else {
      let section: 'none' | 'nodes' | 'wires' = 'none';
      let sawNodesSection = false;
      for (const rawLine of trimmed.split('\n')) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#')) continue;
        if (line === 'nodes:') { section = 'nodes'; sawNodesSection = true; continue; }
        if (line === 'wires:') { section = 'wires'; continue; }
        if (section === 'none') {
          if (line.startsWith('denovo:')) {
            const ver = line.slice(7).trim().replace(/^['"]|['"]$/g, '');
            if (ver !== 'v1') throw new Error(`Unsupported recipe schema version "${ver || '(empty)'}" (supported: v1)`);
            continue;
          }
          if (line.startsWith('name:')) {
            const val = line.slice(5).trim();
            try {
              const parsedName = JSON.parse(val);
              graphName = typeof parsedName === 'string' ? parsedName : safeStr(parsedName) || val.replace(/^['"]|['"]$/g, '');
            } catch {
              graphName = val.replace(/^['"]|['"]$/g, '');
            }
            continue;
          }
          throw new Error(`Unexpected line before "nodes:": "${line}"`);
        }
        if (section === 'nodes') {
          const colon = line.indexOf(':');
          if (colon <= 0) throw new Error(`Invalid node line: "${line}"`);
          const id = line.slice(0, colon).trim();
          const parsedBody = JSON.parse(line.slice(colon + 1).trim()) as unknown;
          if (!isPlainObject(parsedBody)) throw new Error(`Invalid node body on "${id}": expected a JSON object`);
          const rawAdapter = parsedBody.adapterId ?? parsedBody.model ?? '';
          if (typeof rawAdapter !== 'string') {
            throw new Error(`Invalid model on node "${id}": expected a string`);
          }
          const adapterId = rawAdapter;
          const manifest = resolveManifest(adapterId, catalog);
          if (!manifest) throw new Error(`Unknown model "${adapterId}" on node "${id}"`);
          const label = typeof parsedBody.label === 'string' && parsedBody.label.trim() ? parsedBody.label : manifest.shortName;
          const rest = sanitizeParams(parsedBody);
          delete rest.model; delete rest.adapterId; delete rest.label;
          const aliases = catalog?.yamlKeyAliases?.(adapterId);
          if (aliases) {
            for (const [paramKey, yamlKey] of Object.entries(aliases)) {
              if (yamlKey in rest) {
                rest[paramKey] = rest[yamlKey];
                delete rest[yamlKey];
              }
            }
          }
          if (adapterId === 'localfold-af2') {
            const foldModel = parsedBody.foldModel ?? parsedBody.checkpoint;
            if (typeof foldModel === 'string' && foldModel) rest.model = foldModel;
            else if (parsedBody.adapterId && typeof parsedBody.model === 'string' && parsedBody.model && parsedBody.model !== adapterId) {
              rest.model = parsedBody.model;
            }
            delete rest.foldModel; delete rest.checkpoint;
          }
          if (adapterId === 'sequence-input') {
            const seqLabel = parsedBody.seqLabel ?? parsedBody.sequenceLabel;
            if (typeof seqLabel === 'string') rest.label = seqLabel;
            delete rest.seqLabel; delete rest.sequenceLabel;
          }
          let merged = { ...defaultParamsForAdapterEdge(adapterId, catalog), ...rest };
          if (catalog?.sanitizeOnImport) {
            merged = catalog.sanitizeOnImport(adapterId, merged) ?? merged;
          } else if (adapterId === 'pdb-input') {
            delete merged.fetchedPdbId;
            delete merged.fetchedUniProtId;
            const mode = safeStr(merged.mode ?? 'demo-bundle');
            if ((mode === 'pdb-id' || mode === 'alphafold-db' || mode === 'demo-bundle') && 'pdbText' in merged) {
              merged.pdbText = '';
            }
          }
          rawNodes.push({ id, adapterId, label, params: merged });
        } else {
          const cleanLine = line.replace(/^-\s*/, '').trim();
          const arrowIdx = cleanLine.indexOf('->');
          if (arrowIdx <= 0 || cleanLine.indexOf('->', arrowIdx + 2) !== -1) {
            throw new Error(`Invalid wire syntax: "${line}" (expected "a.port -> b.port")`);
          }
          const lhs = cleanLine.slice(0, arrowIdx).trim();
          const rhs = cleanLine.slice(arrowIdx + 2).trim();
          const lhsDot = lhs.lastIndexOf('.');
          const rhsDot = rhs.lastIndexOf('.');
          const fromNode = lhsDot > 0 ? lhs.slice(0, lhsDot).trim() : '';
          const fromPort = lhsDot > 0 ? lhs.slice(lhsDot + 1).trim() : '';
          const toNode = rhsDot > 0 ? rhs.slice(0, rhsDot).trim() : '';
          const toPort = rhsDot > 0 ? rhs.slice(rhsDot + 1).trim() : '';
          if (!fromNode || !toNode || !/^[a-zA-Z0-9_-]+$/.test(fromPort) || !/^[a-zA-Z0-9_-]+$/.test(toPort)) {
            throw new Error(`Invalid wire syntax: "${line}" (expected "a.port -> b.port")`);
          }
          rawEdges.push({ fromNode, fromPort, toNode, toPort });
        }
      }
      if (!sawNodesSection) {
        throw new Error('Invalid recipe: expected JSON or "denovo: v1" YAML with a "nodes:" section');
      }
    }

    void fillDefaults;
    const graph: WorkflowGraph = {
      id: graphId,
      name: graphName,
      ...(graphDesc !== undefined ? { description: graphDesc } : {}),
      nodes: rawNodes,
      edges: rawEdges,
    };

    const errs = validateGraphEdge(graph, {
      requireRunnable: opts.requireRunnable ?? false,
      catalog,
    });
    if (errs.length > 0) {
      return {
        ok: false,
        graph: null,
        nodes: summarizeNodes(rawNodes, catalog),
        edges: summarizeEdges(rawEdges),
        errors: sanitizeFlowErrors(errs),
      };
    }
    const laidOut = graph.nodes.some((n) => !n.position)
      ? autoLayoutGraphEdge(graph, catalog)
      : graph;
    return {
      ok: true,
      graph: laidOut,
      nodes: summarizeNodes(laidOut.nodes, catalog),
      edges: summarizeEdges(laidOut.edges),
      errors: [],
    };
  } catch (err) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: sanitizeFlowErrors([
        {
          code: 'INVALID_RECIPE',
          message: err instanceof Error ? err.message : 'Failed to parse recipe.',
        },
      ]),
    };
  }
}

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

async function deflateRawEdge(text: string): Promise<Uint8Array> {
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

async function inflateRawEdge(bytes: Uint8Array): Promise<string> {
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
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return new TextDecoder().decode(out);
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

/** Edge-safe stateless compact `?p=` / `?g=` encoder (for Cloudflare Worker `/api/link` `shortUrl`). */
export async function encodeCompactShareQuery(
  graph: WorkflowGraph,
  opts: { catalog?: FlowCatalogHooks; forceCompressedDag?: boolean } = {},
): Promise<{
  query: string;
  hash: string;
  length: number;
  mode: 'preset-diff' | 'compressed-dag';
  strippedPdb: boolean;
}> {
  const catalog = opts.catalog;
  let bestPreset: {
    query: string;
    hash: string;
    length: number;
    mode: 'preset-diff';
    strippedPdb: boolean;
  } | null = null;

  if (!opts.forceCompressedDag) {
    for (const preset of WORKFLOW_PRESETS) {
      if (!sameTopology(graph, preset.graph)) continue;
      const qs = new URLSearchParams();
      qs.set('p', preset.id);
      let presetStripped = false;
      const presetBaseline = preset.graph.nodes.some((n) => !n.position)
        ? autoLayoutGraphEdge(structuredClone(preset.graph), catalog)
        : preset.graph;
      for (let i = 0; i < graph.nodes.length; i++) {
        const n = graph.nodes[i];
        const baseN = preset.graph.nodes[i];
        const laidN = presetBaseline.nodes[i];
        const { diff, strippedPdb: s } = diffParamsEdge(n.adapterId, n.params, baseN.params, true, catalog);
        if (s) presetStripped = true;
        for (const [k, v] of Object.entries(diff)) {
          qs.set(`${n.id}.${k}`, typeof v === 'string' ? v : JSON.stringify(v));
        }
        const baseLabel = baseN.label || resolveManifest(baseN.adapterId, catalog)?.shortName || '';
        const curLabel = n.label?.trim() ? n.label : baseLabel;
        if (curLabel && curLabel !== baseLabel) {
          qs.set(`${n.id}._label`, curLabel);
        }
        const basePos = baseN.position ?? laidN?.position;
        if (n.position && Number.isFinite(n.position.x) && Number.isFinite(n.position.y) && basePos) {
          const rx = Math.round(n.position.x);
          const ry = Math.round(n.position.y);
          if (Math.abs(rx - Math.round(basePos.x)) > 2 || Math.abs(ry - Math.round(basePos.y)) > 2) {
            qs.set(`${n.id}._pos`, `${rx},${ry}`);
          }
        }
      }
      const query = qs.toString().replace(/[._,;\-:]$/, (ch) =>
        '%' + ch.charCodeAt(0).toString(16).toUpperCase().padStart(2, '0'),
      );
      const hash = `#${query}`;
      const candidate = { query, hash, length: hash.length, mode: 'preset-diff' as const, strippedPdb: presetStripped };
      if (!bestPreset || candidate.length < bestPreset.length) bestPreset = candidate;
    }
  }
  if (bestPreset && bestPreset.query.length <= MAX_HASH_CHARS) return bestPreset;

  let strippedPdb = false;
  const compactNodes = graph.nodes.map((n) => {
    const a = resolveShortCode(n.adapterId, catalog);
    const { diff, strippedPdb: s } = diffParamsEdge(n.adapterId, n.params, undefined, true, catalog);
    if (s) strippedPdb = true;
    const tuple: unknown[] = [n.id, a];
    const hasDiff = Object.keys(diff).length > 0;
    const customLabel =
      n.label && n.label !== resolveManifest(n.adapterId, catalog)?.shortName ? n.label : undefined;
    if (hasDiff || customLabel) tuple.push(diff);
    if (customLabel) tuple.push(customLabel);
    return tuple;
  });
  const compactEdges = graph.edges.map((e) => [e.fromNode, e.fromPort, e.toNode, e.toPort]);
  const hasAllPos =
    graph.nodes.length > 0 &&
    graph.nodes.every((n) => n.position !== undefined && Number.isFinite(n.position.x) && Number.isFinite(n.position.y));
  const compactPositions = hasAllPos
    ? graph.nodes.map((n) => [Math.round(n.position!.x), Math.round(n.position!.y)])
    : undefined;
  const payload = JSON.stringify(
    compactPositions ? { v: 2, n: compactNodes, e: compactEdges, p: compactPositions } : { v: 2, n: compactNodes, e: compactEdges },
  );
  let b64 = toBase64Url(await deflateRawEdge(payload));
  for (let pad = 1; pad <= 8 && /[_-]$/.test(b64); pad++) {
    b64 = toBase64Url(await deflateRawEdge(payload + ' '.repeat(pad)));
  }
  const query = `g=${b64}`;
  const hash = `#${query}`;
  return { query, hash, length: hash.length, mode: 'compressed-dag', strippedPdb };
}

/** Edge-safe unified decoder for `?flow=`, `?recipe=`, `?p=`, `?g=` (and `#` equivalents). */
export async function decodeWorkflowQuery(
  rawQueryOrUrl: string,
  opts: FlowParseOptions = {},
): Promise<FlowParseResult> {
  const catalog = opts.catalog;
  const trimmed = rawQueryOrUrl.trim();
  if (!trimmed) {
    return { ok: false, graph: null, nodes: [], edges: [], errors: [{ code: 'EMPTY_QUERY', message: 'Query is empty.' }] };
  }

  const { clean, invalidUrl } = extractWorkflowSearchOrHash(trimmed);
  if (invalidUrl) {
    return { ok: false, graph: null, nodes: [], edges: [], errors: [{ code: 'INVALID_URL', message: 'Invalid URL.' }] };
  }
  if (!clean) {
    return { ok: false, graph: null, nodes: [], edges: [], errors: [{ code: 'EMPTY_QUERY', message: 'Query is empty.' }] };
  }
  if (clean.length > MAX_HASH_CHARS) {
    return {
      ok: false,
      graph: null,
      nodes: [],
      edges: [],
      errors: [{ code: 'QUERY_TOO_LARGE', message: `Shared workflow link is too large (max ${MAX_HASH_CHARS} chars).` }],
    };
  }

  const qs = new URLSearchParams(clean);
  if (qs.has('flow')) {
    return parseFlowDsl(extractRawQueryParam(clean, 'flow') ?? qs.get('flow') ?? '', opts);
  }
  if (qs.has('recipe')) {
    return parseRecipeSafe(qs.get('recipe') ?? '', opts);
  }
  if (qs.has('p')) {
    const presetId = qs.get('p');
    const preset = WORKFLOW_PRESETS.find((p) => p.id === presetId);
    if (!preset) {
      return {
        ok: false,
        graph: null,
        nodes: [],
        edges: [],
        errors: sanitizeFlowErrors([{ code: 'UNKNOWN_PRESET', message: `Unknown workflow preset "${presetId ?? ''}".`, token: presetId ?? '' }]),
      };
    }
    const g = structuredClone(preset.graph);
    const byId = new Map(g.nodes.map((n) => [n.id, n]));
    const nodesWithCap = new Set<string>();
    const nodesWithMaxSeq = new Set<string>();
    for (const [k, v] of qs.entries()) {
      if (k === 'p' || k === 'v') continue;
      const dot = k.lastIndexOf('.');
      if (dot <= 0) continue;
      const nodeId = k.slice(0, dot);
      const paramKey = k.slice(dot + 1);
      if (FORBIDDEN_KEYS.has(paramKey)) continue;
      const node = byId.get(nodeId);
      if (!node) continue;
      if (paramKey === 'capMode') nodesWithCap.add(nodeId);
      if (paramKey === 'maxSequences') nodesWithMaxSeq.add(nodeId);
      if (paramKey === '_label') {
        if (v.trim()) node.label = v;
        continue;
      }
      if (paramKey === '_pos') {
        const parts = v.split(',');
        if (parts.length === 2 && Number.isFinite(Number(parts[0])) && Number.isFinite(Number(parts[1]))) {
          node.position = { x: Number(parts[0]), y: Number(parts[1]) };
        }
        continue;
      }
      const manifest = resolveManifest(node.adapterId, catalog);
      if (manifest) {
        node.params[paramKey] = coerceParamValue(manifest, paramKey, v);
      } else {
        node.params[paramKey] = v;
      }
    }
    for (const nid of nodesWithMaxSeq) {
      if (!nodesWithCap.has(nid)) {
        const n = byId.get(nid);
        if (n && (n.adapterId === 'localfold-af2' || n.adapterId === 'esmc' || n.adapterId === 'dyna1')) {
          n.params.capMode = 'fixed';
        }
      }
    }
    const errs = validateGraphEdge(g, { requireRunnable: false, catalog });
    if (errs.length > 0) {
      return { ok: false, graph: null, nodes: summarizeNodes(g.nodes, catalog), edges: summarizeEdges(g.edges), errors: sanitizeFlowErrors(errs) };
    }
    const laidOut = g.nodes.some((n) => !n.position) ? autoLayoutGraphEdge(g, catalog) : g;
    return { ok: true, graph: laidOut, nodes: summarizeNodes(laidOut.nodes, catalog), edges: summarizeEdges(laidOut.edges), errors: [] };
  }
  if (qs.has('g')) {
    const rawG = (qs.get('g') ?? '')
      .replace(/%5[fF]/g, '_')
      .replace(/%2[dD]/g, '-')
      .replace(/\\_/g, '_')
      .replace(/[.,;:!?)\]}>'"]+$/, '');

    const tryDecode = async (cand: string): Promise<{ v?: number; n: unknown[]; e: unknown[]; p?: unknown } | null> => {
      let bytes: Uint8Array;
      try { bytes = fromBase64Url(cand); } catch { return null; }
      let jsonText: string;
      try {
        jsonText = await inflateRawEdge(bytes);
      } catch (err) {
        if (err instanceof Error && /maximum (decompressed|allowed) size/i.test(err.message)) throw err;
        if (bytes.byteLength > MAX_INFLATED_BYTES) throw err;
        jsonText = new TextDecoder().decode(bytes);
      }
      let parsed: unknown;
      try { parsed = JSON.parse(jsonText); } catch { return null; }
      if (!isPlainObject(parsed) || !Array.isArray(parsed.n) || !Array.isArray(parsed.e)) return null;
      return { v: typeof parsed.v === 'number' ? parsed.v : undefined, n: parsed.n, e: parsed.e, p: parsed.p };
    };

    try {
      let obj = await tryDecode(rawG);
      if (!obj) {
        for (const suffix of ['_', '-', '__']) {
          obj = await tryDecode(rawG + suffix);
          if (obj) break;
        }
      }
      if (!obj) {
        return { ok: false, graph: null, nodes: [], edges: [], errors: [{ code: 'INVALID_COMPRESSED_DAG', message: 'Could not decode compressed workflow payload.' }] };
      }
      const rawPositions = Array.isArray(obj.p) ? obj.p : null;
      const nodes: WorkflowNode[] = [];
      for (let i = 0; i < obj.n.length; i++) {
        const tuple = obj.n[i];
        if (!Array.isArray(tuple) || tuple.length < 2 || typeof tuple[0] !== 'string' || typeof tuple[1] !== 'string') {
          return { ok: false, graph: null, nodes: [], edges: [], errors: [{ code: 'INVALID_COMPRESSED_DAG', message: 'Malformed node tuple.' }] };
        }
        const id = tuple[0];
        const shortA = tuple[1];
        const adapterId = catalog?.resolveFromShortCode?.(shortA) ?? (Object.hasOwn(SHORT_TO_ADAPTER, shortA) ? SHORT_TO_ADAPTER[shortA] : shortA);
        const rawOverrides = isPlainObject(tuple[2]) ? sanitizeParams(tuple[2]) : {};
        const manifest = resolveManifest(adapterId, catalog);
        const label = typeof tuple[3] === 'string' ? tuple[3] : manifest?.shortName ?? adapterId;
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
        const isLegacyV1 = !obj.v || obj.v < 2;
        const isSeqCapped = adapterId === 'localfold-af2' || adapterId === 'esmc' || adapterId === 'dyna1';
        const legacyCapPatch = isLegacyV1 && isSeqCapped && !('capMode' in rawOverrides) ? { capMode: 'fixed' } : {};
        nodes.push({
          id,
          adapterId,
          label,
          params: { ...defaultParamsForAdapterEdge(adapterId, catalog), ...legacyCapPatch, ...rawOverrides },
          ...(position ? { position } : {}),
        });
      }
      const edges: WorkflowEdge[] = [];
      for (const e of obj.e) {
        if (
          !Array.isArray(e) ||
          e.length < 4 ||
          typeof e[0] !== 'string' ||
          typeof e[1] !== 'string' ||
          typeof e[2] !== 'string' ||
          typeof e[3] !== 'string'
        ) {
          return { ok: false, graph: null, nodes: [], edges: [], errors: [{ code: 'INVALID_COMPRESSED_DAG', message: 'Malformed edge tuple.' }] };
        }
        edges.push({ fromNode: e[0], fromPort: e[1], toNode: e[2], toPort: e[3] });
      }
      const g: WorkflowGraph = { id: 'shared-workflow', name: 'Shared Workflow', nodes, edges };
      const errs = validateGraphEdge(g, { requireRunnable: false, catalog });
      if (errs.length > 0) {
        return { ok: false, graph: null, nodes: summarizeNodes(nodes, catalog), edges: summarizeEdges(edges), errors: sanitizeFlowErrors(errs) };
      }
      const laidOut = g.nodes.some((n) => !n.position) ? autoLayoutGraphEdge(g, catalog) : g;
      return { ok: true, graph: laidOut, nodes: summarizeNodes(laidOut.nodes, catalog), edges: summarizeEdges(laidOut.edges), errors: [] };
    } catch (err) {
      return {
        ok: false,
        graph: null,
        nodes: [],
        edges: [],
        errors: sanitizeFlowErrors([{ code: 'INVALID_COMPRESSED_DAG', message: err instanceof Error ? err.message : 'Decode failed.' }]),
      };
    }
  }

  return {
    ok: false,
    graph: null,
    nodes: [],
    edges: [],
    errors: [{ code: 'UNSUPPORTED_QUERY', message: 'Expected flow, recipe, p, or g parameter.' }],
  };
}
