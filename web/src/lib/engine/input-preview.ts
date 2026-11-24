// Single-source-of-truth Compiled Input Preview engine for all 7 denovo.run adapters.
// Pure synchronous compilation sharing the exact parsers, lock precedence, MSA masking,
// budget estimation, and validation rules used by execution workers and scheduler.ts.

import { parsePDB, type Structure as MpnnStructure } from 'proteinmpnn-web';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import {
  AA1_TO_AA3,
  AA3_TO_AA1,
  chainIdsFromLengths,
  cropRawAllAtomStructure,
  generateSampleHelixBundlePdb,
  parseStructureText,
  type BackboneAtomPresence,
  type ParsedStructure,
} from '$lib/protein-ops/pdb';
import { normalizePdbId } from '$lib/protein-ops/rcsb';
import { normalizeUniProtId } from '$lib/protein-ops/alphafold-db';
import { evaluateProteinOnlyStructurePolicy } from '$lib/protein-ops/hetatm-policy';
import {
  extractHoloReceptorAndLigands,
  findPocketResidues,
  parsePocketResidueTokens,
} from '$lib/protein-ops/gnina-prep';
import { getBenchmarkById } from './benchmarks';
import { sanitizeEvoDiffFixedSequence, validateSequenceText } from './param-validation';
import {
  selectFoldWorkItems,
  splitMultiChainSequence,
  summarizeCropSegments,
  validateBioCapabilities,
  validateRuntimeNodeInputs,
  type BioFoldLimitOptions,
} from './bio-capabilities';
import {
  extractRfd4TemplateContext,
  isCrystallizationBuffer,
  parseResidueRangeTokens,
  resolveGenie3CompiledInput,
  resolveLocalFoldEffectiveConfig,
  resolveRfd4CompiledInput,
  type Rfd4TemplateContext,
} from './conditioning';
import {
  computeAf2MsaBudgets,
  computeTargetOnlyOffMask,
  validateAndParseA3mText,
} from './localfold-msa';
import { esmfold2Chains } from './localfold-esmfold2';
import {
  compileAntiFoldDesignPlan,
  compileMpnnInputSpec,
  normalizeAntiFoldNumberingParam,
  parseFixedPositionsAndChains,
  type NumberedChainInfo,
  type OrderedStructureResidue,
} from './mpnn-spec';
import {
  getActiveModeSpec,
  getVisibleParamFields,
  validateParamsWithSchema,
} from './param-schema';
import { extractSubgraphUpTo, findContextStructure } from './peek';
import {
  ADAPTER_REGISTRY,
  LOCALFOLD_MODEL_OPTIONS,
  RFD4_BENCHMARK_OPTIONS,
} from './registry';
import { formatResidueToken } from './residue-tokens';
import { estimateRunRequirements } from './run-preflight';
import type {
  CompiledModelPayloadView,
  FastaPreviewEntry,
  NodeInputPreview,
  PreviewChainTrack,
  PreviewIssue,
  PreviewResidue,
  ResidueHighlightLayer,
  ResidueRole,
} from './studio-types';
import type {
  AdapterManifest,
  CanonicalArtifact,
  ConditionSpecArtifact,
  FoldedStructure,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
  WorkflowNode,
} from './types';

// ---------------------------------------------------------------------------
// Bounded LRU structure-parse cache keyed by artifact.id (or raw text digest)
// ---------------------------------------------------------------------------

interface CachedStructureEntry {
  parsed: ParsedStructure;
  bfactors?: Float32Array;
  rfd4Context?: Rfd4TemplateContext;
  mpnnStructures: Partial<Record<'backbone' | 'ca', MpnnStructure>>;
}

const PREVIEW_CACHE_LIMIT = 24;
const structurePreviewCache = new Map<string, CachedStructureEntry>();

export function clearInputPreviewCache(): void {
  structurePreviewCache.clear();
}

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function fastTextKey(prefix: string, text: string): string {
  const len = text.length;
  if (len <= 131_072) return `${prefix}:${len}:${fnv1a(text)}`;
  const edge = 4096;
  let h = 0x811c9dc5;
  for (let i = 0; i < edge; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  const step = Math.max(1, Math.floor((len - edge * 2) / 16_384));
  for (let i = edge; i < len - edge; i += step) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  for (let i = len - edge; i < len; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return `${prefix}:${len}:${(h >>> 0).toString(16).padStart(8, '0')}`;
}

function touchCache(key: string, entry: CachedStructureEntry): CachedStructureEntry {
  structurePreviewCache.delete(key);
  structurePreviewCache.set(key, entry);
  while (structurePreviewCache.size > PREVIEW_CACHE_LIMIT) {
    const oldest = structurePreviewCache.keys().next().value as string | undefined;
    if (oldest === undefined) break;
    structurePreviewCache.delete(oldest);
  }
  return entry;
}

function tokenizeCifRow(line: string): string[] {
  const toks: string[] = [];
  let i = 0;
  const n = line.length;
  while (i < n) {
    while (i < n && (line[i] === ' ' || line[i] === '\t')) i++;
    if (i >= n) break;
    const ch = line[i];
    if (ch === "'" || ch === '"') {
      const q = ch;
      i++;
      const start = i;
      while (i < n && !(line[i] === q && (i + 1 >= n || line[i + 1] === ' ' || line[i + 1] === '\t'))) {
        i++;
      }
      toks.push(line.slice(start, i));
      if (i < n && line[i] === q) i++;
    } else {
      const start = i;
      while (i < n && line[i] !== ' ' && line[i] !== '\t') i++;
      toks.push(line.slice(start, i));
    }
  }
  return toks;
}

/** Extract per-CA B-factor values from PDB or mmCIF text when present and non-zero. */
function extractCaBfactors(rawText: string, expectedLen: number): Float32Array | undefined {
  if (expectedLen <= 0 || !rawText) return undefined;
  const isCif = /^\s*(data_|loop_|_atom_site\.)/m.test(rawText);
  const values: number[] = [];
  const seen = new Set<string>();
  let anyNonZero = false;

  if (!isCif) {
    for (const line of rawText.split(/\r?\n/)) {
      if (line.startsWith('ENDMDL') && values.length > 0) break;
      const isAtom = line.startsWith('ATOM  ');
      const isHet = line.startsWith('HETATM');
      if (!isAtom && !isHet) continue;
      if (line.length < 66) continue;
      const resName = line.slice(17, 20).trim().toUpperCase();
      if (isHet && resName !== 'MSE') continue;
      const atomName = line.slice(12, 16).trim();
      if (atomName !== 'CA') continue;
      const alt = line[16];
      if (alt !== ' ' && alt !== 'A' && alt !== '1') continue;
      const key = `${line[21]?.trim() || 'A'}:${line.slice(22, 26).trim()}:${(line[26] ?? ' ').trim()}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const b = parseFloat(line.slice(60, 66));
      if (Number.isFinite(b)) {
        if (b !== 0) anyNonZero = true;
        values.push(b);
      } else {
        values.push(0);
      }
    }
  } else {
    const cols: string[] = [];
    let inAtomSite = false;
    let firstModel: string | null = null;
    for (const raw of rawText.split(/\r?\n/)) {
      const tr = raw.trim();
      if (tr.startsWith('_atom_site.')) {
        inAtomSite = true;
        cols.push(tr.slice('_atom_site.'.length));
        continue;
      }
      if (!inAtomSite) continue;
      if (tr === '' || tr.startsWith('#') || tr.startsWith('loop_') || tr.startsWith('_')) {
        inAtomSite = false;
        continue;
      }
      const toks = tokenizeCifRow(tr);
      if (toks.length < cols.length) continue;
      const get = (name: string) => {
        const idx = cols.indexOf(name);
        return idx >= 0 ? toks[idx] : '';
      };
      const modelNum = get('pdbx_PDB_model_num');
      if (modelNum && modelNum !== '.' && modelNum !== '?') {
        if (firstModel === null) firstModel = modelNum;
        else if (modelNum !== firstModel) continue;
      }
      const altId = get('label_alt_id');
      if (altId && altId !== '.' && altId !== '?' && altId !== 'A' && altId !== '1') continue;
      const grp = (get('group_PDB') || 'ATOM').toUpperCase();
      const comp = (get('label_comp_id') || get('auth_comp_id') || '').toUpperCase();
      if (grp === 'HETATM' && comp !== 'MSE') continue;
      const atomName = get('label_atom_id') || get('auth_atom_id');
      if (atomName !== 'CA') continue;
      const ch = get('auth_asym_id') || get('label_asym_id') || 'A';
      const seq = get('auth_seq_id') || get('label_seq_id') || '';
      const rawIns = get('pdbx_PDB_ins_code');
      const ins = rawIns && rawIns !== '.' && rawIns !== '?' ? rawIns : '';
      const key = `${ch}:${seq}:${ins}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const b = parseFloat(get('B_iso_or_equiv'));
      if (Number.isFinite(b)) {
        if (b !== 0) anyNonZero = true;
        values.push(b);
      } else {
        values.push(0);
      }
    }
  }

  if (!anyNonZero || values.length !== expectedLen) return undefined;
  return Float32Array.from(values);
}

function getOrParseCachedStructure(cacheKey: string, rawText: string): CachedStructureEntry {
  const existing = structurePreviewCache.get(cacheKey);
  if (existing) {
    return touchCache(cacheKey, existing);
  }
  const parsed = parseStructureText(rawText);
  const bfactors = extractCaBfactors(rawText, parsed.nResidues);
  const entry: CachedStructureEntry = {
    parsed,
    ...(bfactors ? { bfactors } : {}),
    mpnnStructures: {},
  };
  return touchCache(cacheKey, entry);
}

function getCachedRfd4Context(entry: CachedStructureEntry, rawText: string): Rfd4TemplateContext {
  if (!entry.rfd4Context) {
    entry.rfd4Context = extractRfd4TemplateContext(rawText);
  }
  return entry.rfd4Context;
}

function getCachedMpnnStructure(
  entry: CachedStructureEntry,
  pdbText: string,
  atomMode: 'backbone' | 'ca',
): MpnnStructure {
  let cached = entry.mpnnStructures[atomMode];
  if (!cached) {
    cached = parsePDB(pdbText, { fillGaps: false, atomMode });
    entry.mpnnStructures[atomMode] = cached;
  }
  return cached;
}

// ---------------------------------------------------------------------------
// Formatting & Highlight Layer Helpers
// ---------------------------------------------------------------------------

function formatBackboneAtoms(
  presence: BackboneAtomPresence | undefined,
  atomGranularity?: StructureArtifact['atomGranularity'],
): string | undefined {
  if (presence) {
    const hasO = presence.hasO || Boolean(presence.oReconstructed);
    if (presence.hasN && presence.hasCA && presence.hasC && hasO) return 'N/CA/C/O';
    if (!presence.hasN && presence.hasCA && !presence.hasC && !presence.hasO) return 'CA-only';
    const parts: string[] = [];
    if (presence.hasN) parts.push('N');
    if (presence.hasCA) parts.push('CA');
    if (presence.hasC) parts.push('C');
    if (hasO) parts.push('O');
    return parts.length > 0 ? parts.join('/') : 'none';
  }
  if (atomGranularity === 'ca-only') return 'CA-only';
  if (atomGranularity === 'backbone4' || atomGranularity === 'all-atom') return 'N/CA/C/O';
  return undefined;
}

const HIGHLIGHT_LAYER_SPECS: ReadonlyArray<{
  id: string;
  role: ResidueRole;
  matchRoles: readonly ResidueRole[];
  label: string;
  colorHex: number;
  accentSticks?: boolean;
  alpha?: number;
}> = [
  {
    id: 'layer-cropped-out',
    role: 'cropped-out',
    matchRoles: ['cropped-out'],
    label: 'Cropped out',
    colorHex: 0x94a3b8,
    alpha: 0.22,
  },
  {
    id: 'layer-fixed-backbone',
    role: 'fixed-backbone',
    matchRoles: ['fixed-backbone'],
    label: 'Fixed backbone',
    colorHex: 0x64748b,
  },
  {
    id: 'layer-fixed-sequence',
    role: 'fixed-sequence',
    matchRoles: ['fixed-sequence'],
    label: 'Fixed sequence',
    colorHex: 0xeab308,
  },
  {
    id: 'layer-tied',
    role: 'tied',
    matchRoles: ['tied'],
    label: 'Tied (symmetry)',
    colorHex: 0xa855f7,
  },
  {
    id: 'layer-motif',
    role: 'motif',
    matchRoles: ['motif', 'unindexed-motif'],
    label: 'Motif',
    colorHex: 0x14b8a6,
    accentSticks: true,
  },
  {
    id: 'layer-hotspot',
    role: 'hotspot',
    matchRoles: ['hotspot'],
    label: 'Hotspot',
    colorHex: 0xf97316,
    accentSticks: true,
  },
];

function buildHighlightLayersFromResidues(
  residues: readonly PreviewResidue[],
  extraByRole?: Partial<Record<ResidueRole, Array<{ chain: string; seqId: number; insCode?: string }>>>,
): ResidueHighlightLayer[] {
  const layers: ResidueHighlightLayer[] = [];

  for (const spec of HIGHLIGHT_LAYER_SPECS) {
    const seen = new Set<string>();
    const tokens: string[] = [];
    const selectors: Array<{ chain: string; seqId: number; insCode?: string }> = [];

    for (const r of residues) {
      if (r.aa1 === '-') continue; // Skip de novo placeholder spans with no 3D template atoms
      if (!spec.matchRoles.some((mr) => r.roles.includes(mr))) continue;
      const ins = r.insCode ?? '';
      const key = `${r.authChain}:${r.authSeqId}:${ins}`;
      if (seen.has(key)) continue;
      seen.add(key);
      tokens.push(r.token);
      selectors.push({
        chain: r.authChain,
        seqId: r.authSeqId,
        ...(ins ? { insCode: ins } : {}),
      });
    }

    if (extraByRole) {
      for (const mr of spec.matchRoles) {
        const extras = extraByRole[mr];
        if (!extras) continue;
        for (const e of extras) {
          const ins = e.insCode ?? '';
          const key = `${e.chain}:${e.seqId}:${ins}`;
          if (seen.has(key)) continue;
          seen.add(key);
          tokens.push(formatResidueToken(e.chain, e.seqId, ins));
          selectors.push({
            chain: e.chain,
            seqId: e.seqId,
            ...(ins ? { insCode: ins } : {}),
          });
        }
      }
    }

    if (tokens.length > 0) {
      layers.push({
        id: spec.id,
        role: spec.role,
        label: spec.label,
        colorHex: spec.colorHex,
        tokens,
        residues: selectors,
        ...(spec.accentSticks ? { accentSticks: true } : {}),
        ...(spec.alpha !== undefined ? { alpha: spec.alpha } : {}),
      });
    }
  }

  return layers;
}

function classifyLigands(ligandNames: readonly string[]): {
  includedLigands: string[];
  filteredBuffers: string[];
} {
  const included: string[] = [];
  const filtered: string[] = [];
  const seenInc = new Set<string>();
  const seenFilt = new Set<string>();
  for (const raw of ligandNames) {
    const code = raw.trim().toUpperCase();
    if (!code) continue;
    if (isCrystallizationBuffer(code)) {
      if (!seenFilt.has(code)) {
        seenFilt.add(code);
        filtered.push(code);
      }
    } else if (!seenInc.has(code)) {
      seenInc.add(code);
      included.push(code);
    }
  }
  return { includedLigands: included, filteredBuffers: filtered };
}

function nextChainLetter(used: ReadonlySet<string>): string {
  const alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  for (let i = 0; i < alpha.length; i++) {
    if (!used.has(alpha[i])) return alpha[i];
  }
  return 'Z';
}

function parseContigLengthMid(tok: string): number | null {
  const m = /^(\d+)(?:-(\d+))?$/.exec(tok.trim());
  if (!m) return null;
  const lo = parseInt(m[1], 10);
  const hi = m[2] ? parseInt(m[2], 10) : lo;
  return Math.max(1, Math.round((lo + hi) / 2));
}

function buildSyntheticDesignedChain(
  chainId: string,
  length: number,
  startFlatIndex: number,
  roleTag: PreviewChainTrack['roleTag'],
  labelChainId?: string,
): PreviewChainTrack {
  const residues: PreviewResidue[] = [];
  for (let i = 0; i < length; i++) {
    residues.push({
      flatIndex: startFlatIndex + i,
      authChain: chainId,
      ...(labelChainId ? { labelChain: labelChainId } : {}),
      authSeqId: i + 1,
      insCode: '',
      ...(labelChainId ? { labelSeqId: i + 1 } : {}),
      token: formatResidueToken(chainId, i + 1),
      aa1: '-',
      resName: 'UNK',
      roles: ['designable'],
    });
  }
  return {
    chainId,
    ...(labelChainId ? { labelChainId } : {}),
    entityKind: 'protein',
    ...(roleTag ? { roleTag } : {}),
    sequence: '-'.repeat(length),
    residues,
  };
}

// ---------------------------------------------------------------------------
// Synchronous CPU source materialization for live un-run upstream preview
// ---------------------------------------------------------------------------

function tryMaterializeLocalPdbInput(node: WorkflowNode): StructureArtifact | null {
  const mode = String(node.params.mode ?? 'demo-bundle');
  let rawText = '';
  let filename = '';
  if (mode === 'demo-bundle') {
    rawText = generateSampleHelixBundlePdb();
    filename = 'demo_3helix_53aa.pdb';
  } else if (mode === 'custom') {
    rawText = String(node.params.pdbText ?? '').trim();
    if (!rawText) return null;
    filename = String(node.params.filename || 'uploaded.pdb');
  } else if (mode === 'pdb-id') {
    const normId = normalizePdbId(String(node.params.pdbId ?? ''));
    const fetchedFor = normalizePdbId(String(node.params.fetchedPdbId ?? ''));
    const cachedText = String(node.params.pdbText ?? '').trim();
    if (!normId || !cachedText || (fetchedFor && fetchedFor !== normId)) return null;
    rawText = cachedText;
    const rawFn = String(node.params.filename ?? '').trim();
    filename = rawFn && rawFn !== 'demo_3helix_53aa.pdb' ? rawFn : `${normId}.cif`;
  } else if (mode === 'alphafold-db') {
    const normUniProt = normalizeUniProtId(String(node.params.uniprotId ?? ''));
    const fetchedFor = normalizeUniProtId(String(node.params.fetchedUniProtId ?? ''));
    const cachedText = String(node.params.pdbText ?? '').trim();
    if (!normUniProt || !cachedText || (fetchedFor && fetchedFor !== normUniProt)) return null;
    rawText = cachedText;
    const rawFn = String(node.params.filename ?? '').trim();
    filename = rawFn && rawFn !== 'demo_3helix_53aa.pdb' ? rawFn : `AF-${normUniProt}-F1.cif`;
  } else {
    return null;
  }

  const cropSpec = mode === 'demo-bundle' ? '' : String(node.params.cropSegments ?? '').trim();
  const effectiveRawText = cropSpec ? cropRawAllAtomStructure(rawText, cropSpec).text : rawText;
  const cacheKey = fastTextKey('pdb-src', effectiveRawText);
  const cached = getOrParseCachedStructure(cacheKey, effectiveRawText);
  const parsed = cached.parsed;
  const hasNonProtein = parsed.diagnostics.hasLigands || parsed.diagnostics.hasNucleicAcids;
  if (parsed.nResidues < 3 && !hasNonProtein) return null;
  const benchmarkId = String(node.params.benchmarkId ?? '').trim() || undefined;

  return {
    kind: 'StructureArtifact',
    id: `pdb-${fnv1a(`${rawText}::${cropSpec}${benchmarkId ? `::${benchmarkId}` : ''}`)}`,
    label: cropSpec ? `${filename} [${cropSpec}]` : filename,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.nResidues > 0 ? parsed.pdbText : parsed.format === 'mmcif' ? '' : effectiveRawText,
    cifText: parsed.format === 'mmcif' ? effectiveRawText : undefined,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    chainIdMap: parsed.chainIdMap,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    ...(benchmarkId ? { benchmarkId } : {}),
    geometry: computeGeometryStats(
      parsed.caCoords,
      parsed.chainIds,
      parsed.resNums,
      parsed.insCodes,
    ),
  };
}

function tryMaterializeLocalPdbInputSequences(
  node: WorkflowNode,
  existingStruct?: StructureArtifact,
): SequenceSet | null {
  const struct = existingStruct ?? tryMaterializeLocalPdbInput(node);
  if (!struct || struct.length <= 0) return null;
  const seq = struct.sequence.trim().toUpperCase();
  if (!seq) return null;
  const candidateId = struct.label.replace(/\s+\([^)]*\)$/, '').trim() || struct.id;
  const chainLengths =
    struct.chainLengths && struct.chainLengths.length > 0 ? [...struct.chainLengths] : [seq.length];
  return {
    kind: 'SequenceSet',
    id: `pdbseq-${fnv1a(`${struct.id}:${seq}`)}`,
    parentStructureId: struct.id,
    sourceStructureId: struct.id,
    sourceModelId: 'pdb-input',
    collectionKind: 'independent-candidates',
    candidates: [
      {
        id: candidateId,
        key: candidateId,
        sequence: seq,
        ...(chainLengths.length > 1 ? { chainLengths } : {}),
        score: null,
        globalScore: null,
        sourceModelId: 'pdb-input',
        sourceStructureId: struct.id,
      },
    ],
  };
}

function tryMaterializeLocalSequenceInput(node: WorkflowNode): SequenceSet | null {
  const raw = String(node.params.sequence ?? '').trim();
  if (!raw) return null;
  if (validateSequenceText(raw).status === 'error') return null;
  const candidates: SequenceSet['candidates'] = [];
  if (raw.includes('>')) {
    const blocks = raw.split(/^\s*>/m).filter((b) => b.trim().length > 0);
    const seenIds = new Set<string>();
    blocks.forEach((b, idx) => {
      const lines = b.split('\n');
      const baseHeader = lines[0].trim().split(/\s+/)[0] || `seq-${idx + 1}`;
      const split = splitMultiChainSequence(lines.slice(1).join(''));
      if (split.concatenated.length >= 5) {
        let id = baseHeader;
        let suffix = 2;
        while (seenIds.has(id)) id = `${baseHeader}_${suffix++}`;
        seenIds.add(id);
        candidates.push({
          id,
          key: id,
          sequence: split.concatenated,
          ...(split.chains.length > 1 ? { chainLengths: split.chainLengths } : {}),
          score: null,
          globalScore: null,
          sourceModelId: 'sequence-input',
          sourceStructureId: 'none',
        });
      }
    });
  } else {
    const split = splitMultiChainSequence(raw);
    if (split.concatenated.length >= 5) {
      const id = String(node.params.label || 'custom-seq-1');
      candidates.push({
        id,
        key: id,
        sequence: split.concatenated,
        ...(split.chains.length > 1 ? { chainLengths: split.chainLengths } : {}),
        score: null,
        globalScore: null,
        sourceModelId: 'sequence-input',
        sourceStructureId: 'none',
      });
    }
  }
  if (candidates.length === 0) return null;
  return {
    kind: 'SequenceSet',
    id: `seqin-${fnv1a(JSON.stringify(candidates))}`,
    parentStructureId: 'none',
    sourceStructureId: 'none',
    sourceModelId: 'sequence-input',
    collectionKind: 'independent-candidates',
    candidates,
  };
}

// ---------------------------------------------------------------------------
// Residue mapping validator (mirrors scheduler.ts parseReferenceResidueMapping)
// ---------------------------------------------------------------------------

function parseScorerMappingPreview(
  raw: unknown,
  foldLen: number,
  refLen: number,
): Array<[number, number]> | null {
  if (raw === undefined || raw === null || raw === '') return null;
  const pairs: Array<[number, number]> = [];
  const assertStrictInt = (v: unknown, role: string): number => {
    if (typeof v !== 'number' || !Number.isInteger(v)) {
      throw new Error(`Invalid residueMapping: ${role} must be a strict finite integer (got ${String(v)}).`);
    }
    return v;
  };
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    for (const tok of trimmed.split(/[,\s;]+/).filter(Boolean)) {
      const m = /^(\d+)\s*(?::|->|=)\s*(\d+)$/.exec(tok);
      if (!m) {
        throw new Error(
          `Invalid residueMapping entry "${tok}". Expected 0-based "foldIdx:refIdx" pairs (e.g. "0:0,1:1,2:2").`,
        );
      }
      pairs.push([Number(m[1]), Number(m[2])]);
    }
  } else if (Array.isArray(raw)) {
    for (const entry of raw) {
      if (!Array.isArray(entry) || entry.length < 2) {
        throw new Error('Invalid residueMapping array: every entry must be a [foldIdx, refIdx] pair.');
      }
      pairs.push([assertStrictInt(entry[0], 'fold index'), assertStrictInt(entry[1], 'reference index')]);
    }
  } else if (typeof raw === 'object') {
    const obj = raw as { pairs?: unknown[]; foldIndices?: unknown[]; referenceIndices?: unknown[] };
    if (Array.isArray(obj.pairs)) {
      for (const entry of obj.pairs) {
        if (!Array.isArray(entry) || entry.length < 2) {
          throw new Error('Invalid residueMapping.pairs: every entry must be a [foldIdx, refIdx] pair.');
        }
        pairs.push([assertStrictInt(entry[0], 'fold index'), assertStrictInt(entry[1], 'reference index')]);
      }
    } else if (Array.isArray(obj.foldIndices) && Array.isArray(obj.referenceIndices)) {
      if (obj.foldIndices.length !== obj.referenceIndices.length) {
        throw new Error(
          `Invalid residueMapping: foldIndices (${obj.foldIndices.length}) and referenceIndices (${obj.referenceIndices.length}) must have equal length.`,
        );
      }
      for (let i = 0; i < obj.foldIndices.length; i++) {
        pairs.push([
          assertStrictInt(obj.foldIndices[i], 'fold index'),
          assertStrictInt(obj.referenceIndices[i], 'reference index'),
        ]);
      }
    } else {
      throw new Error(
        'Invalid residueMapping object: provide either "pairs" or both "foldIndices" and "referenceIndices".',
      );
    }
  } else {
    throw new Error('Invalid residueMapping parameter type.');
  }
  if (pairs.length < 3) {
    throw new Error(
      `Invalid residueMapping: at least 3 mapped residue pairs are required for Kabsch superposition (got ${pairs.length}).`,
    );
  }
  const seenFold = new Set<number>();
  const seenRef = new Set<number>();
  for (const [fi, ri] of pairs) {
    if (fi < 0 || fi >= foldLen) {
      throw new Error(`Invalid residueMapping: fold index ${fi} is out of bounds for fold length ${foldLen}.`);
    }
    if (ri < 0 || ri >= refLen) {
      throw new Error(`Invalid residueMapping: reference index ${ri} is out of bounds for reference length ${refLen}.`);
    }
    if (seenFold.has(fi)) {
      throw new Error(`Invalid residueMapping: duplicate fold index ${fi}.`);
    }
    if (seenRef.has(ri)) {
      throw new Error(`Invalid residueMapping: duplicate reference index ${ri}.`);
    }
    seenFold.add(fi);
    seenRef.add(ri);
  }
  return pairs;
}

// ---------------------------------------------------------------------------
// Main entry point: compileNodeInputPreview
// ---------------------------------------------------------------------------

export interface SchemaFallbackPreviewContext {
  node: WorkflowNode;
  nodeLabel: string;
  manifest: AdapterManifest | undefined;
  resolvedInputs: Record<string, CanonicalArtifact>;
  resolvedInputLists?: Record<string, CanonicalArtifact[]>;
  missingUpstreamByPort: Map<string, string>;
  allMissingUpstream: Set<string>;
  runRuntimeValidation: () => boolean;
}

export interface SchemaFallbackPreviewResult {
  availability: NodeInputPreview['availability'];
  missingUpstreamNodeIds: string[];
  chains: PreviewChainTrack[];
  entities: NodeInputPreview['entities'];
  fastaEntries: FastaPreviewEntry[];
  previewStructurePdb: string | null;
  previewStructureMeta?: NodeInputPreview['previewStructureMeta'];
  highlightLayers: ResidueHighlightLayer[];
  payload: CompiledModelPayloadView;
  totalResidues: number;
  designedResidues: number;
  fixedResidues: number;
  estimatedTokens: number;
}

/**
 * Schema-driven compiled input preview for adapters without a dedicated branch (`CT-08`:
 * ProtPardelle, RFdiffusion, BoltzGen, EvoDiff, La-Proteina, Proteina, SiD-Protein, Genie 2,
 * FrameFlow, FoldFlow 2, ESM-C, Dyna-1, Top-K Filter, Structure Compare).
 * Evaluates active mode & port requirements, resolves upstream structures/sequences, runs
 * runtime validation, and emits honest `ready` / `pending-upstream` / `error` availability.
 */
export function compileSchemaFallbackPreview(
  ctx: SchemaFallbackPreviewContext,
): SchemaFallbackPreviewResult {
  const {
    node,
    nodeLabel,
    manifest,
    resolvedInputs,
    missingUpstreamByPort,
    allMissingUpstream,
    runRuntimeValidation,
  } = ctx;

  const activeMode = manifest ? getActiveModeSpec(manifest, node.params) : null;
  const requiredPorts = new Set<string>();
  if (manifest) {
    for (const [port, spec] of Object.entries(manifest.inputs)) {
      if (spec.required !== false) requiredPorts.add(port);
    }
    if (activeMode?.requiredInputs) {
      for (const port of activeMode.requiredInputs) requiredPorts.add(port);
    }
  }

  const missingUpstreamIds = new Set<string>();
  const unconnectedRequiredPorts: string[] = [];
  for (const port of requiredPorts) {
    if (!resolvedInputs[port]) {
      const upId = missingUpstreamByPort.get(port);
      if (upId) missingUpstreamIds.add(upId);
      else unconnectedRequiredPorts.push(port);
    }
  }

  // Handle cross-input adapters (e.g. esmc, dyna1) where at least one of sequences/structure is required
  if (
    (node.adapterId === 'esmc' || node.adapterId === 'dyna1') &&
    !resolvedInputs.sequences &&
    !resolvedInputs.structure
  ) {
    const upSeq = missingUpstreamByPort.get('sequences');
    const upStruct = missingUpstreamByPort.get('structure');
    if (upSeq) missingUpstreamIds.add(upSeq);
    if (upStruct) missingUpstreamIds.add(upStruct);
    if (!upSeq && !upStruct) {
      unconnectedRequiredPorts.push('sequences|structure');
    }
  }

  // Handle multi-input comparison nodes wired to un-run upstreams
  if (node.adapterId === 'structure-compare' && Object.keys(resolvedInputs).length === 0) {
    if (allMissingUpstream.size > 0) {
      for (const id of allMissingUpstream) missingUpstreamIds.add(id);
    } else {
      unconnectedRequiredPorts.push('structure');
    }
  }

  // Build schema-driven compiledSummary from visible fields
  const compiledSummary: CompiledModelPayloadView['compiledSummary'] = {
    adapterId: node.adapterId,
  };
  if (manifest?.modeParamKey && activeMode) {
    compiledSummary[manifest.modeParamKey] = activeMode.id;
  }
  const visibleFields = manifest ? getVisibleParamFields(manifest, node.params) : [];
  for (const field of visibleFields) {
    const raw = node.params[field.key] ?? field.defaultValue;
    if (raw === undefined || raw === null || raw === '') continue;
    if (typeof raw === 'string' || typeof raw === 'number' || typeof raw === 'boolean') {
      compiledSummary[field.key] = raw;
    } else if (Array.isArray(raw)) {
      compiledSummary[field.key] = raw.map((x) => String(x));
    }
  }

  let availability: NodeInputPreview['availability'] = 'ready';
  if (unconnectedRequiredPorts.length > 0) {
    availability = 'error';
  } else if (missingUpstreamIds.size > 0) {
    availability = 'pending-upstream';
  }

  let chains: PreviewChainTrack[] = [];
  let entities: NodeInputPreview['entities'] = {
    includedLigands: [],
    filteredBuffers: [],
    nucleicChains: [],
  };
  let fastaEntries: FastaPreviewEntry[] = [];
  let previewStructurePdb: string | null = null;
  let previewStructureMeta: NodeInputPreview['previewStructureMeta'];
  const highlightLayers: ResidueHighlightLayer[] = [];
  let totalResidues = 0;
  let designedResidues = 0;
  let fixedResidues = 0;
  let estimatedTokens = 0;

  if (availability === 'ready') {
    runRuntimeValidation();
  }

  const allCompareStructArts: StructureArtifact[] = [];
  if (node.adapterId === 'structure-compare' && ctx.resolvedInputLists) {
    const seenIds = new Set<string>();
    const ordered = [
      ...(ctx.resolvedInputLists.reference ?? []),
      ...(ctx.resolvedInputLists.structures ?? []),
      ...Object.values(ctx.resolvedInputLists).flat(),
    ];
    for (const art of ordered) {
      if (art?.kind === 'StructureArtifact' && !seenIds.has(art.id)) {
        seenIds.add(art.id);
        allCompareStructArts.push(art);
      }
    }
  }
  const structArt =
    allCompareStructArts[0] ??
    (resolvedInputs.structure as StructureArtifact | undefined) ??
    (Object.values(resolvedInputs).find((a) => a?.kind === 'StructureArtifact') as
      | StructureArtifact
      | undefined);
  const seqArt = resolvedInputs.sequences as SequenceSet | undefined;

  if (structArt && structArt.kind === 'StructureArtifact') {
    const rawText = structArt.pdbText || structArt.cifText || '';
    if (rawText) {
      try {
        const entry = getOrParseCachedStructure(structArt.id, rawText);
        const parsed = entry.parsed;
        const chainMap = new Map<string, PreviewChainTrack>();
        for (let i = 0; i < parsed.nResidues; i++) {
          const src = parsed.sourceResidues[i];
          let track = chainMap.get(src.authChainId);
          if (!track) {
            track = {
              chainId: src.authChainId,
              ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
              entityKind: 'protein',
              sequence: '',
              residues: [],
            };
            chainMap.set(src.authChainId, track);
          }
          const aa1 = parsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
          track.residues.push({
            flatIndex: i,
            authChain: src.authChainId,
            ...(src.labelChainId ? { labelChain: src.labelChainId } : {}),
            authSeqId: src.resNum,
            insCode: src.insCode,
            token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
            aa1,
            resName: src.resName,
            roles: ['fixed-backbone'],
            backboneAtoms: formatBackboneAtoms(parsed.atomPresence[i], structArt.atomGranularity),
          });
          track.sequence += aa1;
        }
        chains = [...chainMap.values()];
        const tryParseStruct = (s: StructureArtifact) => {
          const t = s.pdbText || s.cifText || '';
          if (!t) return null;
          try {
            return getOrParseCachedStructure(s.id, t).parsed;
          } catch {
            return null;
          }
        };
        const rawLigNames =
          allCompareStructArts.length > 1
            ? allCompareStructArts.flatMap((s) => {
                const p = !s.diagnostics ? tryParseStruct(s) : null;
                return s.diagnostics?.ligandNames ?? p?.diagnostics.ligandNames ?? [];
              })
            : (structArt.diagnostics?.ligandNames ?? parsed.diagnostics.ligandNames);
        const rawNucleicChains =
          allCompareStructArts.length > 1
            ? [
                ...new Set(
                  allCompareStructArts.flatMap((s) => {
                    const p = !s.diagnostics ? tryParseStruct(s) : null;
                    return s.diagnostics?.nucleicAcidChains ?? p?.diagnostics.nucleicAcidChains ?? [];
                  }),
                ),
              ]
            : [...(structArt.diagnostics?.nucleicAcidChains ?? parsed.diagnostics.nucleicAcidChains)];
        const ligClass = classifyLigands(rawLigNames);
        entities = {
          includedLigands: ligClass.includedLigands,
          filteredBuffers: ligClass.filteredBuffers,
          nucleicChains: rawNucleicChains,
        };
        totalResidues = parsed.nResidues;
        fixedResidues = parsed.nResidues;
        estimatedTokens = parsed.nResidues;
        const hasPrimaryNonProt = Boolean(
          ligClass.includedLigands.length > 0 ||
            rawNucleicChains.length > 0 ||
            structArt.diagnostics?.hasLigands ||
            structArt.diagnostics?.hasNucleicAcids,
        );
        previewStructurePdb =
          structArt.cifText && (hasPrimaryNonProt || !structArt.pdbText)
            ? structArt.cifText
            : structArt.pdbText || structArt.cifText || null;
        previewStructureMeta = {
          label: structArt.label,
          atomGranularity: structArt.atomGranularity ?? 'backbone4',
          bfactorSemantics: structArt.bfactorSemantics,
          length: parsed.nResidues,
        };
        fastaEntries =
          allCompareStructArts.length > 1
            ? allCompareStructArts.map((s) => {
                const t = s.pdbText || s.cifText || '';
                const p = t ? getOrParseCachedStructure(s.id, t).parsed : null;
                const seq = p?.sequence || s.sequence || '';
                const nRes = p?.nResidues ?? s.length;
                return {
                  id: s.id,
                  header: `>${s.label} (${nRes} aa)`,
                  sequence: seq,
                  ...(p?.diagnostics.chainLengths
                    ? { chainLengths: p.diagnostics.chainLengths }
                    : s.chainLengths
                      ? { chainLengths: s.chainLengths }
                      : {}),
                  selected: true,
                };
              })
            : [
                {
                  id: structArt.id,
                  header: `>${structArt.label} (${parsed.nResidues} aa)`,
                  sequence: parsed.sequence,
                  chainLengths: parsed.diagnostics.chainLengths,
                  selected: true,
                },
              ];
      } catch {
        totalResidues = structArt.length;
        estimatedTokens = structArt.length;
        previewStructurePdb = structArt.pdbText || structArt.cifText || null;
      }
    } else {
      totalResidues = structArt.length;
      estimatedTokens = structArt.length;
    }
  } else if (seqArt && seqArt.kind === 'SequenceSet') {
    fastaEntries = seqArt.candidates.map((cand) => ({
      id: cand.id,
      header: `>${cand.id}${typeof cand.score === 'number' && Number.isFinite(cand.score) ? ` | score ${cand.score.toFixed(2)}` : ''}`,
      sequence: cand.sequence,
      ...(cand.chainLengths ? { chainLengths: cand.chainLengths } : {}),
      selected: true,
    }));
    const primaryLen = seqArt.candidates[0]?.sequence.replace(/[\s/:,]+/g, '').length ?? 0;
    totalResidues = primaryLen;
    estimatedTokens = primaryLen;
  } else {
    const lengthField = manifest?.paramSchema?.find((f) => f.key === 'length');
    const rawLen = Number(node.params.length ?? lengthField?.defaultValue ?? 0);
    if (Number.isFinite(rawLen) && rawLen > 0) {
      totalResidues = Math.round(rawLen);
      designedResidues = totalResidues;
      estimatedTokens = totalResidues;
    }
  }

  const modeTag = activeMode ? ` (${activeMode.label})` : '';
  const statusTag =
    availability === 'pending-upstream'
      ? 'waiting for upstream input'
      : availability === 'error' && unconnectedRequiredPorts.length > 0
        ? `connect ${unconnectedRequiredPorts.join(', ')}`
        : totalResidues > 0
          ? `L=${totalResidues} aa`
          : 'ready';
  const payload: CompiledModelPayloadView = {
    format: 'raw-input',
    headline: `${nodeLabel}${modeTag} · ${statusTag}`,
    compiledSummary,
    rawSpecText: JSON.stringify(compiledSummary, null, 2),
  };

  return {
    availability,
    missingUpstreamNodeIds: [...missingUpstreamIds],
    chains,
    entities,
    fastaEntries,
    previewStructurePdb,
    previewStructureMeta,
    highlightLayers,
    payload,
    totalResidues,
    designedResidues,
    fixedResidues,
    estimatedTokens,
  };
}

/**
 * Pure synchronous compiler producing the exact `NodeInputPreview` for any workflow node.
 * Covers all 25 denovo.run adapters (dedicated compilers + schema-driven fallback) with zero side effects.
 */
export function compileNodeInputPreview(
  graph: WorkflowGraph,
  nodeId: string,
  outputsByNode: Readonly<Record<string, Record<string, CanonicalArtifact>>>,
  foldLimitOpts?: BioFoldLimitOptions,
): NodeInputPreview {
  const node = graph.nodes.find((n) => n.id === nodeId);
  if (!node) {
    return {
      nodeId,
      adapterId: 'unknown',
      availability: 'error',
      missingUpstreamNodeIds: [],
      chains: [],
      entities: { includedLigands: [], filteredBuffers: [], nucleicChains: [] },
      fastaEntries: [],
      previewStructurePdb: null,
      highlightLayers: [],
      payload: {
        format: 'raw-input',
        headline: `Node "${nodeId}" not found`,
        compiledSummary: {},
        rawSpecText: '',
      },
      budget: {
        totalResidues: 0,
        designedResidues: 0,
        fixedResidues: 0,
        estimatedTokens: 0,
        nodeWeightsMB: 0,
        nodeVramMB: 0,
        subgraphDownloadMB: 0,
        subgraphPeakVramMB: 0,
      },
      issues: [{ severity: 'error', source: 'compiler', message: `Node "${nodeId}" not found in graph.` }],
      previewDigest: fnv1a(`missing:${nodeId}`),
    };
  }

  const manifest = ADAPTER_REGISTRY[node.adapterId];
  const nodeLabel = node.label || manifest?.shortName || node.adapterId;
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
  const incomingEdges = graph.edges.filter((e) => e.toNode === nodeId);
  const connectedInputs = incomingEdges.map((e) => e.toPort);

  // Resolve incoming port artifacts (materializing local CPU source nodes on the fly when un-run)
  const resolvedInputs: Record<string, CanonicalArtifact> = {};
  const resolvedInputLists: Record<string, CanonicalArtifact[]> = {};
  const missingUpstreamByPort = new Map<string, string>();
  const allMissingUpstream = new Set<string>();

  for (const edge of incomingEdges) {
    let art: CanonicalArtifact | undefined = outputsByNode[edge.fromNode]?.[edge.fromPort];
    if (!art) {
      const srcNode = nodeById.get(edge.fromNode);
      if (srcNode?.adapterId === 'pdb-input' && edge.fromPort === 'structure') {
        art = tryMaterializeLocalPdbInput(srcNode) ?? undefined;
      } else if (srcNode?.adapterId === 'pdb-input' && edge.fromPort === 'sequences') {
        const cachedStruct = outputsByNode[edge.fromNode]?.structure as
          | StructureArtifact
          | undefined;
        art = tryMaterializeLocalPdbInputSequences(srcNode, cachedStruct) ?? undefined;
      } else if (srcNode?.adapterId === 'sequence-input' && edge.fromPort === 'sequences') {
        art = tryMaterializeLocalSequenceInput(srcNode) ?? undefined;
      }
    }
    if (art) {
      resolvedInputs[edge.toPort] = art;
      if (!resolvedInputLists[edge.toPort]) resolvedInputLists[edge.toPort] = [];
      resolvedInputLists[edge.toPort].push(art);
    } else {
      missingUpstreamByPort.set(edge.toPort, edge.fromNode);
      allMissingUpstream.add(edge.fromNode);
    }
  }

  // Merge schema + bio issues
  const issues: PreviewIssue[] = [];
  const seenIssues = new Set<string>();
  const pushIssue = (issue: PreviewIssue) => {
    const key = `${issue.severity}::${issue.message}`;
    if (seenIssues.has(key)) return;
    seenIssues.add(key);
    issues.push(issue);
  };

  if (manifest) {
    for (const msg of validateParamsWithSchema(manifest, node.params, {
      label: nodeLabel,
      connectedInputs,
    })) {
      pushIssue({ severity: 'error', source: 'schema', message: msg });
    }
  }

  const mutableOutputs = outputsByNode as Record<string, Record<string, CanonicalArtifact>>;
  for (const bio of validateBioCapabilities(graph, mutableOutputs, foldLimitOpts)) {
    if (bio.nodeId === nodeId) {
      pushIssue({
        severity: bio.severity,
        source: 'bio',
        message: bio.message,
        ...(bio.fix ? { fix: bio.fix } : {}),
      });
    }
  }

  // Compute node & ancestor subgraph VRAM/weight budgets
  const ancestorSubgraph = extractSubgraphUpTo(graph, nodeId);
  const subReq = estimateRunRequirements(ancestorSubgraph);
  const singleNodeGraph: WorkflowGraph = {
    ...graph,
    nodes: [node],
    edges: incomingEdges,
  };
  const nodeReq = estimateRunRequirements(singleNodeGraph);

  let availability: NodeInputPreview['availability'] = 'ready';
  let missingUpstreamNodeIds: string[] = [];
  let chains: PreviewChainTrack[] = [];
  let entities: NodeInputPreview['entities'] = {
    includedLigands: [],
    filteredBuffers: [],
    nucleicChains: [],
  };
  let fastaEntries: FastaPreviewEntry[] = [];
  let previewStructurePdb: string | null = null;
  let previewStructureMeta: NodeInputPreview['previewStructureMeta'];
  let highlightLayers: ResidueHighlightLayer[] = [];
  let payload: CompiledModelPayloadView = {
    format: 'raw-input',
    headline: nodeLabel,
    compiledSummary: {},
    rawSpecText: '',
  };
  let totalResidues = 0;
  let designedResidues = 0;
  let fixedResidues = 0;
  let estimatedTokens = 0;

  const runRuntimeValidation = () => {
    try {
      const rt = validateRuntimeNodeInputs(node, resolvedInputs, foldLimitOpts);
      for (const w of rt.warnings) {
        pushIssue({ severity: 'warning', source: 'runtime', message: w });
      }
      return true;
    } catch (err) {
      pushIssue({
        severity: 'error',
        source: 'runtime',
        message: err instanceof Error ? err.message : String(err),
      });
      return false;
    }
  };

  // -------------------------------------------------------------------------
  // Adapter-specific compilation
  // -------------------------------------------------------------------------

  if (node.adapterId === 'pdb-input') {
    const mode = String(node.params.mode ?? 'demo-bundle');
    const cropSpec = mode === 'demo-bundle' ? '' : String(node.params.cropSegments ?? '').trim();
    const outStruct = outputsByNode[nodeId]?.structure as StructureArtifact | undefined;

    let rawText = '';
    let filename = String(node.params.filename ?? '');
    if (mode === 'demo-bundle') {
      rawText = generateSampleHelixBundlePdb();
      filename = 'demo_3helix_53aa.pdb';
    } else if (mode === 'custom') {
      rawText = String(node.params.pdbText ?? '').trim() || outStruct?.cifText || outStruct?.pdbText || '';
      filename = filename || 'uploaded.pdb';
    } else if (mode === 'pdb-id') {
      const normId = normalizePdbId(String(node.params.pdbId ?? ''));
      const fetchedFor = normalizePdbId(String(node.params.fetchedPdbId ?? ''));
      const cachedText = String(node.params.pdbText ?? '').trim();
      const nonDemoFilename =
        filename.trim() && filename.trim() !== 'demo_3helix_53aa.pdb' ? filename.trim() : '';
      if (normId && cachedText && (!fetchedFor || fetchedFor === normId)) {
        rawText = cachedText;
        filename = nonDemoFilename || `${normId}.cif`;
      } else if (outStruct) {
        rawText = outStruct.cifText ?? outStruct.pdbText;
        filename = nonDemoFilename || outStruct.label || (normId ? `${normId}.cif` : 'structure.cif');
      } else {
        availability = 'partial';
        const benchId = String(node.params.benchmarkId ?? '').trim();
        const bench = benchId ? getBenchmarkById(benchId) : undefined;
        const effectiveCropHints = cropSpec || String(bench?.cropSegments ?? '').trim();
        const nucleicChains = bench?.entities.includes('dna') ? ['DNA'] : [];
        if (effectiveCropHints) {
          const cropHints = summarizeCropSegments(effectiveCropHints);
          if (cropHints.ligandNames.length > 0 || nucleicChains.length > 0) {
            const ligClass = classifyLigands(cropHints.ligandNames);
            entities = {
              includedLigands: ligClass.includedLigands,
              filteredBuffers: ligClass.filteredBuffers,
              nucleicChains,
            };
          }
        } else if (nucleicChains.length > 0) {
          entities = {
            includedLigands: [],
            filteredBuffers: [],
            nucleicChains,
          };
        }
        payload = {
          format: 'raw-input',
          headline: normId
            ? `RCSB PDB ${normId} (click Fetch or Run to download)`
            : 'PDB Input · enter a 4-char RCSB PDB ID',
          compiledSummary: {
            mode,
            pdbId: normId || '(unset)',
            cropSegments: cropSpec || '(none)',
          },
          rawSpecText: JSON.stringify({ mode, pdbId: normId, cropSegments: cropSpec }, null, 2),
        };
      }
    }

    if (rawText) {
      try {
        const fullKey = outStruct && !cropSpec ? outStruct.id : fastTextKey('pdb-full', rawText);
        const fullEntry = getOrParseCachedStructure(fullKey, rawText);
        const fullParsed = fullEntry.parsed;

        let keptParsed = fullParsed;
        let keptEntry = fullEntry;
        let keptRawText = rawText;
        let keptKeySet: Set<string> | null = null;
        if (cropSpec) {
          const cropped = cropRawAllAtomStructure(rawText, cropSpec);
          keptRawText = cropped.text;
          keptEntry = getOrParseCachedStructure(
            fastTextKey(`pdb-crop:${cropSpec}`, cropped.text),
            cropped.text,
          );
          keptParsed = keptEntry.parsed;
          keptKeySet = new Set<string>();
          for (let i = 0; i < keptParsed.nResidues; i++) {
            const s = keptParsed.sourceResidues[i];
            keptKeySet.add(`${s.authChainId}:${s.resNum}:${s.insCode}`);
          }
        }

        const keptCtx = getCachedRfd4Context(keptEntry, keptRawText);
        const ligClass = classifyLigands([
          ...keptParsed.diagnostics.ligandNames,
          ...keptCtx.ligands.map((l) => l.compId),
        ]);
        if (cropSpec) {
          const fullLigClass = classifyLigands(fullParsed.diagnostics.ligandNames);
          const keptLigSet = new Set(ligClass.includedLigands.map((l) => l.toUpperCase()));
          const droppedFromFull = fullLigClass.includedLigands.filter(
            (l) => !keptLigSet.has(l.toUpperCase()),
          );
          const droppedLigands =
            droppedFromFull.length > 0
              ? droppedFromFull
              : (outStruct?.diagnostics?.omittedLigands ?? []).filter(
                  (l) => !keptLigSet.has(l.toUpperCase()),
                );
          if (droppedLigands.length > 0) {
            pushIssue({
              severity: 'warning',
              source: 'compiler',
              message: `Crop "${cropSpec}" omits ligand/ion context (${droppedLigands.join(', ')}). Include them explicitly (e.g. ${cropSpec}, ${droppedLigands.join(', ')}) or clear the crop to keep holo ligands.`,
            });
          }
        }
        entities = {
          includedLigands: ligClass.includedLigands,
          filteredBuffers: ligClass.filteredBuffers,
          nucleicChains: [
            ...new Set([
              ...keptParsed.diagnostics.nucleicAcidChains,
              ...keptCtx.nucleicChains.map((c) => c.authChain),
            ]),
          ],
        };

        const chainMap = new Map<string, PreviewChainTrack>();
        const allRes: PreviewResidue[] = [];
        let keptIdx = 0;
        let croppedOutCount = 0;

        for (let i = 0; i < fullParsed.nResidues; i++) {
          const src = fullParsed.sourceResidues[i];
          const key = `${src.authChainId}:${src.resNum}:${src.insCode}`;
          const isKept = keptKeySet ? keptKeySet.has(key) : true;
          const flatIndex = isKept ? keptIdx++ : -1;
          if (!isKept) croppedOutCount++;

          let track = chainMap.get(src.authChainId);
          if (!track) {
            track = {
              chainId: src.authChainId,
              ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
              entityKind: 'protein',
              sequence: '',
              residues: [],
            };
            chainMap.set(src.authChainId, track);
          }

          let gapBefore = 0;
          if (track.residues.length > 0) {
            const prev = track.residues[track.residues.length - 1];
            const delta = src.resNum - prev.authSeqId - 1;
            if (delta > 0) gapBefore = delta;
          }

          const aa1 = fullParsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
          const res: PreviewResidue = {
            flatIndex,
            authChain: src.authChainId,
            ...(src.labelChainId ? { labelChain: src.labelChainId } : {}),
            authSeqId: src.resNum,
            insCode: src.insCode,
            ...(src.labelSeqId !== undefined ? { labelSeqId: src.labelSeqId } : {}),
            token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
            aa1,
            resName: src.resName,
            roles: isKept ? ['designable'] : ['cropped-out'],
            ...(gapBefore > 0 ? { gapBefore } : {}),
            ...(fullEntry.bfactors ? { bfactor: fullEntry.bfactors[i] } : {}),
            backboneAtoms: formatBackboneAtoms(fullParsed.atomPresence[i]),
          };
          track.residues.push(res);
          if (isKept) track.sequence += aa1;
          allRes.push(res);
        }

        chains = [...chainMap.values()];
        highlightLayers = buildHighlightLayersFromResidues(allRes);
        totalResidues = keptParsed.nResidues;
        fixedResidues = keptParsed.nResidues;
        designedResidues = 0;
        estimatedTokens = keptParsed.nResidues + entities.includedLigands.length;

        const hasNonProt = fullParsed.diagnostics.hasLigands || fullParsed.diagnostics.hasNucleicAcids;
        previewStructurePdb =
          fullParsed.format === 'mmcif' && hasNonProt
            ? rawText
            : fullParsed.nResidues > 0
              ? fullParsed.pdbText
              : rawText;
        previewStructureMeta = {
          label: cropSpec ? `${filename} [${cropSpec}]` : filename,
          atomGranularity: fullParsed.atomPresence.every((p) => p.hasCA && !p.hasN && !p.hasC)
            ? 'ca-only'
            : 'all-atom',
          bfactorSemantics: 'crystal-b',
          length: keptParsed.nResidues,
        };

        if (keptParsed.nResidues > 0) {
          fastaEntries = [
            {
              id: filename || 'structure',
              header: `>${filename || 'structure'} (${keptParsed.nResidues} aa)`,
              sequence: keptParsed.sequence,
              chainLengths: keptParsed.diagnostics.chainLengths,
              selected: true,
            },
          ];
        }

        payload = {
          format: 'raw-input',
          headline: cropSpec
            ? `PDB Input · ${filename} · ${keptParsed.nResidues} aa kept (${croppedOutCount} cropped)`
            : `PDB Input · ${filename} · ${keptParsed.nResidues} aa (${chains.length} chain${chains.length === 1 ? '' : 's'})`,
          compiledSummary: {
            mode,
            format: fullParsed.format,
            filename,
            keptResidues: keptParsed.nResidues,
            ...(cropSpec ? { croppedOutResidues: croppedOutCount, cropSegments: cropSpec } : {}),
            chains: chains.map((c) => c.chainId),
            ...(entities.includedLigands.length > 0 ? { ligands: entities.includedLigands } : {}),
            ...(entities.filteredBuffers.length > 0 ? { buffers: entities.filteredBuffers } : {}),
            ...(entities.nucleicChains.length > 0 ? { nucleicChains: entities.nucleicChains } : {}),
          },
          rawSpecText: JSON.stringify(
            {
              mode,
              filename,
              format: fullParsed.format,
              cropSegments: cropSpec || null,
              keptResidues: keptParsed.nResidues,
              croppedOutResidues: croppedOutCount,
              chains: chains.map((c) => ({ chainId: c.chainId, length: c.sequence.length })),
              entities,
            },
            null,
            2,
          ),
        };
      } catch (err) {
        availability = 'error';
        pushIssue({
          severity: 'error',
          source: 'compiler',
          message: err instanceof Error ? err.message : String(err),
        });
      }
    } else if (mode === 'custom') {
      availability = 'error';
      pushIssue({
        severity: 'error',
        source: 'compiler',
        message: 'PDB Input (custom mode) has no PDB/mmCIF structure text yet.',
      });
    }
  } else if (node.adapterId === 'sequence-input') {
    const seqSet = tryMaterializeLocalSequenceInput(node);
    if (!seqSet) {
      availability = 'error';
      pushIssue({
        severity: 'error',
        source: 'compiler',
        message: 'Sequence Input requires at least one valid amino acid sequence (>= 5 aa).',
      });
      payload = {
        format: 'raw-input',
        headline: 'Sequence Input · empty or invalid sequence',
        compiledSummary: { sequences: 0 },
        rawSpecText: '',
      };
    } else {
      const first = seqSet.candidates[0];
      const chainLengths = first.chainLengths ?? [first.sequence.length];
      const chainLetters = [...new Set(chainIdsFromLengths(chainLengths))];
      let offset = 0;
      chains = chainLengths.map((len, idx) => {
        const chId = chainLetters[idx] ?? String.fromCharCode(65 + idx);
        const subSeq = first.sequence.slice(offset, offset + len);
        const residues: PreviewResidue[] = [];
        for (let k = 0; k < subSeq.length; k++) {
          const aa1 = subSeq[k];
          residues.push({
            flatIndex: offset + k,
            authChain: chId,
            authSeqId: k + 1,
            insCode: '',
            token: formatResidueToken(chId, k + 1),
            aa1,
            resName: AA1_TO_AA3[aa1] ?? 'UNK',
            roles: ['designable'],
          });
        }
        offset += len;
        return {
          chainId: chId,
          entityKind: 'protein',
          sequence: subSeq,
          residues,
        };
      });

      fastaEntries = seqSet.candidates.map((c, idx) => ({
        id: c.id,
        header: `>${c.id}`,
        sequence: c.sequence,
        ...(c.chainLengths ? { chainLengths: c.chainLengths } : {}),
        selected: idx === 0,
      }));

      totalResidues = first.sequence.length;
      designedResidues = totalResidues;
      fixedResidues = 0;
      estimatedTokens = totalResidues;

      const rawFasta = seqSet.candidates.map((c) => `>${c.id}\n${c.sequence}`).join('\n');
      payload = {
        format: 'raw-input',
        headline: `Sequence Input · ${seqSet.candidates.length} sequence${seqSet.candidates.length === 1 ? '' : 's'} · ${first.sequence.length} aa`,
        compiledSummary: {
          sequences: seqSet.candidates.length,
          firstLength: first.sequence.length,
          chains: chainLengths.length,
          chainLengths: chainLengths.map(String),
        },
        rawSpecText: rawFasta,
      };
    }
  } else if (node.adapterId === 'web-protein-mpnn') {
    const structure = resolvedInputs.structure as StructureArtifact | undefined;
    if (!structure) {
      const missingUp = missingUpstreamByPort.get('structure');
      if (missingUp) {
        availability = 'pending-upstream';
        missingUpstreamNodeIds = [missingUp];
      } else {
        availability = 'error';
      }
      payload = {
        format: 'mpnn-masks',
        headline: missingUp
          ? 'ProteinMPNN · waiting for upstream structure'
          : 'ProteinMPNN · connect a StructureArtifact input',
        compiledSummary: {
          family: String(node.params.family ?? 'solublempnn'),
          modelName: String(node.params.modelName ?? 'v_48_020'),
          numSequences: Number(node.params.numSequences ?? 2),
          temperature: Number(node.params.temperature ?? 0.1),
        },
        rawSpecText: JSON.stringify(node.params, null, 2),
      };
    } else {
      runRuntimeValidation();
      try {
        const rawText = structure.pdbText;
        const entry = getOrParseCachedStructure(structure.id, rawText);
        const family = String(node.params.family ?? 'solublempnn');
        const atomMode: 'backbone' | 'ca' = family === 'ca' ? 'ca' : 'backbone';
        const mpnnStruct = getCachedMpnnStructure(entry, rawText, atomMode);
        const compiled = compileMpnnInputSpec(structure, node.params, mpnnStruct);

        for (const w of compiled.warnings) {
          pushIssue({ severity: 'warning', source: 'compiler', message: w });
        }
        for (const e of compiled.errors) {
          pushIssue({ severity: 'error', source: 'compiler', message: e });
        }

        // Index parsed atomPresence & bfactors by (chain, resNum, insCode) for exact matching
        const parsedLookup = new Map<
          string,
          { presence?: BackboneAtomPresence; bfactor?: number; labelChain?: string; labelSeqId?: number; resName?: string }
        >();
        for (let i = 0; i < entry.parsed.nResidues; i++) {
          const src = entry.parsed.sourceResidues[i];
          const k = `${src.authChainId}:${src.resNum}:${src.insCode}`;
          parsedLookup.set(k, {
            presence: entry.parsed.atomPresence[i],
            bfactor: entry.bfactors?.[i],
            labelChain: src.labelChainId,
            labelSeqId: src.labelSeqId,
            resName: src.resName,
          });
        }

        const chainMap = new Map<string, PreviewChainTrack>();
        const allRes: PreviewResidue[] = [];

        for (const r of compiled.residues) {
          let track = chainMap.get(r.chain);
          if (!track) {
            const roleTag = structure.chainRoles?.[r.chain];
            track = {
              chainId: r.chain,
              entityKind: 'protein',
              ...(roleTag ? { roleTag } : {}),
              sequence: '',
              residues: [],
            };
            chainMap.set(r.chain, track);
          }

          const roles: ResidueRole[] = [];
          if (!r.validBackbone) {
            roles.push('missing-backbone');
          } else if (r.willDesign) {
            roles.push('designable');
          } else {
            roles.push('fixed-sequence');
          }
          if (r.tieGroupId !== undefined) {
            roles.push('tied');
          }

          const meta = parsedLookup.get(`${r.chain}:${r.resNum}:${r.insCode}`);
          const pr: PreviewResidue = {
            flatIndex: r.flatIndex,
            authChain: r.chain,
            ...(meta?.labelChain ? { labelChain: meta.labelChain } : {}),
            authSeqId: r.resNum,
            insCode: r.insCode,
            ...(meta?.labelSeqId !== undefined ? { labelSeqId: meta.labelSeqId } : {}),
            token: formatResidueToken(r.chain, r.resNum, r.insCode),
            aa1: r.aa1,
            resName: meta?.resName ?? AA1_TO_AA3[r.aa1] ?? 'UNK',
            roles,
            ...(r.tieGroupId !== undefined ? { tieGroupId: r.tieGroupId } : {}),
            ...(r.inheritedLock ? { inheritedLock: true } : {}),
            ...(r.gapBefore > 0 ? { gapBefore: r.gapBefore } : {}),
            ...(meta?.bfactor !== undefined ? { bfactor: meta.bfactor } : {}),
            backboneAtoms: formatBackboneAtoms(meta?.presence, structure.atomGranularity),
          };
          track.residues.push(pr);
          track.sequence += r.aa1;
          allRes.push(pr);
        }

        chains = [...chainMap.values()];
        highlightLayers = buildHighlightLayersFromResidues(allRes);

        const mpnnCtx = getCachedRfd4Context(entry, structure.cifText ?? rawText);
        const ligClass = classifyLigands([
          ...(structure.diagnostics?.ligandNames ?? entry.parsed.diagnostics.ligandNames),
          ...mpnnCtx.ligands.map((l) => l.compId),
        ]);
        entities = {
          includedLigands: ligClass.includedLigands,
          filteredBuffers: ligClass.filteredBuffers,
          nucleicChains: [
            ...new Set([
              ...(structure.diagnostics?.nucleicAcidChains ??
                entry.parsed.diagnostics.nucleicAcidChains),
              ...mpnnCtx.nucleicChains.map((c) => c.authChain),
            ]),
          ],
        };

        totalResidues = compiled.residues.length;
        designedResidues = compiled.designedCount;
        fixedResidues = compiled.fixedCount;
        estimatedTokens = totalResidues;

        previewStructurePdb = structure.pdbText;
        previewStructureMeta = {
          label: structure.label,
          atomGranularity: structure.atomGranularity ?? (atomMode === 'ca' ? 'ca-only' : 'backbone4'),
          bfactorSemantics: structure.bfactorSemantics,
          length: compiled.residues.length,
        };

        fastaEntries = [
          {
            id: structure.id,
            header: `>${structure.label} (template · ${compiled.designedCount}/${compiled.residues.length} designable)`,
            sequence: compiled.structure.sequence,
            chainLengths: compiled.chainLengths,
            selected: true,
          },
        ];

        const familyLabel =
          compiled.family === 'solublempnn'
            ? 'SolubleMPNN'
            : compiled.family === 'ca'
              ? 'Cα-ProteinMPNN'
              : 'ProteinMPNN';

        payload = {
          format: 'mpnn-masks',
          headline: `${familyLabel} (${compiled.modelName}) · ${compiled.designedCount} design / ${compiled.fixedCount} fixed`,
          compiledSummary: {
            family: compiled.family,
            modelName: compiled.modelName,
            atomMode: compiled.atomMode,
            numSequences: compiled.numSequences,
            temperature: compiled.temperature,
            backboneNoise: compiled.backboneNoise,
            seed: compiled.seed,
            designChains: compiled.designChains ?? compiled.allChains,
            lockedChains: compiled.lockedChains,
            omitAminoAcids: compiled.omitAminoAcids ?? 'X',
            tiedGroups: compiled.tiedPositions.length,
            ignoreInheritedLocks: compiled.ignoreInheritedLocks,
          },
          rawSpecText: JSON.stringify(
            {
              family: compiled.family,
              modelName: compiled.modelName,
              atomMode: compiled.atomMode,
              numSequences: compiled.numSequences,
              temperature: compiled.temperature,
              backboneNoise: compiled.backboneNoise,
              seed: compiled.seed,
              designChains: compiled.designChains ?? null,
              lockedChains: compiled.lockedChains,
              inheritedLockedChains: compiled.inheritedLockedChains,
              fixedPositions: compiled.fixedPositions,
              tiedPositions: compiled.tiedPositions,
              omitAminoAcids: compiled.omitAminoAcids ?? 'X',
              ignoreInheritedLocks: compiled.ignoreInheritedLocks,
              designedCount: compiled.designedCount,
              fixedCount: compiled.fixedCount,
              missingBackboneCount: compiled.missingBackboneCount,
            },
            null,
            2,
          ),
        };
      } catch (err) {
        availability = 'error';
        pushIssue({
          severity: 'error',
          source: 'compiler',
          message: err instanceof Error ? err.message : String(err),
        });
      }
    }
  } else if (node.adapterId === 'rfd4-proteina') {
    const mode = String(node.params.mode ?? 'unconditional');
    const template = resolvedInputs.structure as StructureArtifact | undefined;
    const condArtifact = resolvedInputs.condition as ConditionSpecArtifact | undefined;
    const modeNeedsTemplate =
      mode === 'motif' || mode === 'binder' || mode === 'partial' || mode === 'inverse-folding';
    const missingStructUp = missingUpstreamByPort.get('structure');

    if (modeNeedsTemplate && !template && !condArtifact?.inputCif && !condArtifact?.cifUrl && missingStructUp) {
      availability = 'pending-upstream';
      missingUpstreamNodeIds = [missingStructUp];
      payload = {
        format: 'rfd4-spec',
        headline: `RFD4-Proteina (${mode}) · waiting for upstream structure`,
        compiledSummary: {
          mode,
          nsteps: Number(node.params.nsteps ?? 60),
          recycleSchedule: String(node.params.recycleSchedule ?? 'low'),
          seed: Number(node.params.seed ?? 42),
        },
        rawSpecText: JSON.stringify(node.params, null, 2),
      };
    } else {
      if (template || !modeNeedsTemplate) {
        runRuntimeValidation();
      }
      try {
        const rawTplText =
          mode === 'partial' || mode === 'inverse-folding'
            ? (template?.cifText ?? template?.pdbText)
            : (condArtifact?.inputCif ?? template?.cifText ?? template?.pdbText);

        let tplEntry: CachedStructureEntry | undefined;
        let tplContext: Rfd4TemplateContext | undefined;
        if (rawTplText) {
          const cacheKey = template?.id ?? fastTextKey('rfd4-tpl', rawTplText);
          tplEntry = getOrParseCachedStructure(cacheKey, rawTplText);
          tplContext = getCachedRfd4Context(tplEntry, rawTplText);
        }

        const compiled = resolveRfd4CompiledInput(node.params, template, condArtifact, tplContext);

        // Classify template entities (included non-buffer / explicitly named ligands vs filtered buffers)
        if (compiled.templateContext) {
          const ctx = compiled.templateContext;
          const explicitNamed = new Set<string>();
          const collectRefs = (raw: string | undefined) => {
            if (!raw) return;
            for (const tok of raw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean)) {
              if (tok === '/0') continue;
              const m = /^([A-Za-z0-9_]+):([A-Za-z0-9]{1,5})$/.exec(tok);
              if (m && !/^\d+[A-Za-z]?$/.test(m[2])) {
                explicitNamed.add(m[2].toUpperCase());
              } else if (/^[A-Za-z0-9]{1,5}$/.test(tok) && !/^\d+$/.test(tok)) {
                explicitNamed.add(tok.toUpperCase());
              }
            }
          };
          collectRefs(compiled.intent?.contig);
          if (compiled.intent?.hbondSelection) {
            for (const seg of compiled.intent.hbondSelection.split(/[,|)\s'"]+/).filter(Boolean)) {
              const parts = seg.split('/');
              if (parts.length >= 2 && parts[1] && parts[1] !== '*') {
                explicitNamed.add(parts[1].trim().toUpperCase());
              }
            }
          }
          const incLig = new Set<string>();
          const filtBuf = new Set<string>();
          for (const lig of ctx.ligands) {
            const comp = lig.compId.toUpperCase();
            if (isCrystallizationBuffer(comp) && !explicitNamed.has(comp)) {
              filtBuf.add(comp);
            } else {
              incLig.add(comp);
            }
          }
          entities = {
            includedLigands: [...incLig],
            filteredBuffers: [...filtBuf],
            nucleicChains: ctx.nucleicChains.map((c) => c.authChain),
          };
        }

        if (compiled.kind === 'preset') {
          const presetMeta = RFD4_BENCHMARK_OPTIONS.find((b) => b.id === compiled.activePresetId);
          payload = {
            format: 'rfd4-spec',
            headline: `RFD4 Preset · ${presetMeta?.label ?? compiled.activePresetId}`,
            compiledSummary: {
              mode: compiled.mode,
              presetId: compiled.activePresetId ?? '',
              needsEncoder: compiled.needEncoder,
              nsteps: compiled.nsteps,
              recycleSchedule: compiled.recycleSchedule,
              earlyStop: compiled.earlyStop,
              seed: compiled.seed,
            },
            rawSpecText: JSON.stringify(
              {
                presetId: compiled.activePresetId,
                needsEncoder: compiled.needEncoder,
                nsteps: compiled.nsteps,
                recycleSchedule: compiled.recycleSchedule,
                earlyStop: compiled.earlyStop,
                seed: compiled.seed,
              },
              null,
              2,
            ),
          };
          estimatedTokens = 100;
        } else if (!compiled.usesTemplate || !tplEntry || !compiled.templateContext) {
          // Unconditional or CATH monomer without template
          const track = buildSyntheticDesignedChain('A', compiled.length, 0, 'designed-span', 'A');
          chains = [track];
          totalResidues = compiled.length;
          designedResidues = compiled.length;
          fixedResidues = 0;
          estimatedTokens = compiled.length;
          fastaEntries = [
            {
              id: 'rfd4-designed',
              header: `>RFD4_designed (${compiled.length} aa)`,
              sequence: track.sequence,
              chainLengths: [compiled.length],
              selected: true,
            },
          ];
          payload = {
            format: 'rfd4-spec',
            headline: compiled.runLabel,
            compiledSummary: {
              mode: compiled.mode,
              contig: compiled.contig ?? '(custom)',
              conditions: Object.keys(compiled.conditions),
              nsteps: compiled.nsteps,
              recycleSchedule: compiled.recycleSchedule,
              earlyStop: compiled.earlyStop,
              seed: compiled.seed,
              needsEncoder: compiled.needEncoder,
            },
            rawSpecText:
              compiled.kind === 'custom-yaml'
                ? compiled.customSpecYaml
                : JSON.stringify(
                    {
                      contig: compiled.contig,
                      conditions: compiled.conditions,
                      nsteps: compiled.nsteps,
                      recycleSchedule: compiled.recycleSchedule,
                      earlyStop: compiled.earlyStop,
                      seed: compiled.seed,
                    },
                    null,
                    2,
                  ),
          };
        } else {
          // Template-conditioned RFD4 modes: binder, motif, conditioned, partial, inverse-folding
          const parsed = tplEntry.parsed;
          const ctx = compiled.templateContext;
          const intent = compiled.intent;

          // Determine which template residues are kept in compiled.contig (label_* space)
          const keptLabelKeys = new Set<string>();
          const designedSpans: Array<{ len: number; afterBreak: boolean }> = [];
          const preserveAllTemplate =
            compiled.kind === 'template-spec' ||
            compiled.kind === 'template-custom-yaml' ||
            compiled.contig === null;

          if (!preserveAllTemplate && compiled.contig) {
            let sawBreak = false;
            for (const rawTok of compiled.contig.split(',').map((s) => s.trim()).filter(Boolean)) {
              if (rawTok === '/0') {
                sawBreak = true;
                continue;
              }
              if (/^\d+(?:-\d+)?[PRDN]?$/i.test(rawTok)) {
                const mid = parseContigLengthMid(rawTok.replace(/[PRDN]$/i, ''));
                if (mid !== null) designedSpans.push({ len: mid, afterBreak: sawBreak });
                sawBreak = false;
                continue;
              }
              const namedExp = /^([A-Za-z0-9_]+):(\d+(?:-\d+)?)[PRDN]?$/i.exec(rawTok);
              if (namedExp && !ctx.polymerByLabelChain.has(namedExp[1]) && !ctx.polymerByAuthChain.has(namedExp[1])) {
                const mid = parseContigLengthMid(namedExp[2]);
                if (mid !== null) designedSpans.push({ len: mid, afterBreak: sawBreak });
                sawBreak = false;
                continue;
              }
              for (const r of parseResidueRangeTokens(rawTok)) {
                const poly =
                  ctx.polymerByLabelChain.get(r.chain) ??
                  ctx.polymerByLabelChain.get(r.chain.toUpperCase());
                if (!poly || poly.isNucleic) continue;
                for (const entry of poly.residues) {
                  if (entry.labelSeqId >= r.start && entry.labelSeqId <= r.end) {
                    keptLabelKeys.add(`${entry.labelChain}:${entry.labelSeqId}`);
                  }
                }
              }
              sawBreak = false;
            }
          }

          // Resolve locked chain, hotspots, and motif spans in label_* space
          const lockedLabelChain = intent?.lockChain
            ? ctx.remapPolymerChain(intent.lockChain)
            : undefined;
          const hotspotLabelKeys = new Set<string>();
          if (intent?.hotspots?.length) {
            for (const tok of ctx.remapResidueList(intent.hotspots)) {
              for (const r of parseResidueRangeTokens(tok)) {
                for (let s = r.start; s <= r.end; s++) hotspotLabelKeys.add(`${r.chain}:${s}`);
              }
            }
          }
          const motifLabelKeys = new Set<string>();
          if (intent?.motifSpans?.length) {
            for (const tok of ctx.remapResidueList(intent.motifSpans)) {
              for (const r of parseResidueRangeTokens(tok)) {
                for (let s = r.start; s <= r.end; s++) motifLabelKeys.add(`${r.chain}:${s}`);
              }
            }
          }

          const outputIdxByTplIdx = new Map<number, number>();
          for (const sel of compiled.selectedSourceResidues ?? []) {
            if (sel.templateIndex >= 0 && !outputIdxByTplIdx.has(sel.templateIndex)) {
              outputIdxByTplIdx.set(sel.templateIndex, sel.outputIndex);
            }
          }

          const chainMap = new Map<string, PreviewChainTrack>();
          const allRes: PreviewResidue[] = [];
          let fixedCount = 0;
          let tplDesignedCount = 0;
          let fallbackIdx = 0;

          for (let i = 0; i < parsed.nResidues; i++) {
            const src = parsed.sourceResidues[i];
            const authKey = `${src.authChainId}:${src.resNum}:${src.insCode}`;
            const polyEntry =
              ctx.byAuthRes.get(authKey) ?? ctx.byAuthRes.get(`${src.authChainId}:${src.resNum}`);
            const labelChain = polyEntry?.labelChain ?? src.labelChainId ?? src.authChainId;
            const labelSeqId = polyEntry?.labelSeqId ?? src.labelSeqId;
            const labelKey = labelSeqId !== undefined ? `${labelChain}:${labelSeqId}` : '';

            const isKept = preserveAllTemplate || (labelKey ? keptLabelKeys.has(labelKey) : false);
            const isLockedChain = Boolean(lockedLabelChain && labelChain === lockedLabelChain);
            const isHotspot = Boolean(labelKey && hotspotLabelKeys.has(labelKey));
            const isMotif = Boolean(labelKey && motifLabelKeys.has(labelKey));

            const roles: ResidueRole[] = [];
            if (!isKept) {
              roles.push('cropped-out');
            } else {
              if (compiled.mode === 'inverse-folding') {
                roles.push('fixed-backbone');
                if (isLockedChain || isMotif) {
                  roles.push('fixed-sequence');
                  fixedCount++;
                } else {
                  roles.push('designable');
                  tplDesignedCount++;
                }
              } else if (compiled.mode === 'partial') {
                if (isLockedChain || isMotif) {
                  roles.push('fixed-backbone', 'fixed-sequence');
                  fixedCount++;
                } else {
                  roles.push('designable');
                  tplDesignedCount++;
                }
              } else {
                if (isLockedChain) {
                  roles.push('fixed-backbone', 'fixed-sequence');
                  fixedCount++;
                } else if (isMotif) {
                  roles.push(
                    intent?.unindexedMotif ? 'unindexed-motif' : 'motif',
                    'fixed-backbone',
                    'fixed-sequence',
                  );
                  fixedCount++;
                } else {
                  roles.push('fixed-backbone', 'fixed-sequence');
                  fixedCount++;
                }
              }
              if (isHotspot) roles.push('hotspot');
            }

            let track = chainMap.get(src.authChainId);
            if (!track) {
              track = {
                chainId: src.authChainId,
                labelChainId: labelChain,
                entityKind: 'protein',
                ...(isLockedChain ? { roleTag: 'target' } : {}),
                sequence: '',
                residues: [],
              };
              chainMap.set(src.authChainId, track);
            }

            let gapBefore = 0;
            if (track.residues.length > 0) {
              const prev = track.residues[track.residues.length - 1];
              const delta = src.resNum - prev.authSeqId - 1;
              if (delta > 0) gapBefore = delta;
            }

            const aa1 = parsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
            const flatIndex = isKept
              ? (outputIdxByTplIdx.get(i) ?? fallbackIdx++)
              : -1;

            const pr: PreviewResidue = {
              flatIndex,
              authChain: src.authChainId,
              labelChain,
              authSeqId: src.resNum,
              insCode: src.insCode,
              ...(labelSeqId !== undefined ? { labelSeqId } : {}),
              token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
              aa1,
              resName: src.resName,
              roles,
              ...(gapBefore > 0 ? { gapBefore } : {}),
              ...(tplEntry.bfactors ? { bfactor: tplEntry.bfactors[i] } : {}),
              backboneAtoms: formatBackboneAtoms(parsed.atomPresence[i], template?.atomGranularity),
            };
            track.residues.push(pr);
            if (isKept) track.sequence += aa1;
            allRes.push(pr);
          }

          const tracks: PreviewChainTrack[] = [...chainMap.values()];
          const usedChainIds = new Set(tracks.map((t) => t.chainId));
          const designedSpanTotal = designedSpans.reduce((s, d) => s + d.len, 0);

          if (designedSpanTotal > 0) {
            const binderChainId = nextChainLetter(usedChainIds);
            const roleTag = compiled.mode === 'binder' ? 'binder' : 'designed-span';
            const designedTrack = buildSyntheticDesignedChain(
              binderChainId,
              designedSpanTotal,
              fixedCount,
              roleTag,
              binderChainId,
            );
            tracks.push(designedTrack);
          }

          chains = tracks;
          highlightLayers = buildHighlightLayersFromResidues(allRes);
          designedResidues = designedSpanTotal + tplDesignedCount;
          fixedResidues = fixedCount;
          totalResidues = designedResidues + fixedResidues;
          estimatedTokens =
            totalResidues +
            entities.includedLigands.length +
            ctx.nucleicChains.reduce((s, c) => s + c.residues.length, 0);

          const hasNonProt =
            entities.includedLigands.length > 0 ||
            entities.nucleicChains.length > 0 ||
            Boolean(template?.diagnostics?.hasLigands || template?.diagnostics?.hasNucleicAcids);
          previewStructurePdb =
            template?.cifText && (hasNonProt || !template.pdbText)
              ? template.cifText
              : (template?.pdbText ?? rawTplText ?? null);
          previewStructureMeta = template
            ? {
                label: template.label,
                atomGranularity: template.atomGranularity ?? 'all-atom',
                bfactorSemantics: template.bfactorSemantics,
                length: template.length,
              }
            : undefined;

          const activeSeqs = tracks.filter((t) => t.sequence.length > 0);
          if (activeSeqs.length > 0) {
            fastaEntries = [
              {
                id: 'rfd4-input',
                header: `>${compiled.runLabel}`,
                sequence: activeSeqs.map((t) => t.sequence).join(''),
                chainLengths: activeSeqs.map((t) => t.sequence.length),
                selected: true,
              },
            ];
          }

          payload = {
            format: 'rfd4-spec',
            headline: compiled.runLabel,
            compiledSummary: {
              mode: compiled.mode,
              contig: compiled.contig ?? 'input:template',
              conditions: Object.keys(compiled.conditions),
              nsteps: compiled.nsteps,
              recycleSchedule: compiled.recycleSchedule,
              earlyStop: compiled.earlyStop,
              seed: compiled.seed,
              needsEncoder: compiled.needEncoder,
              ...(compiled.effectivePartialFraction != null
                ? { partialFraction: compiled.effectivePartialFraction }
                : {}),
              ...(entities.includedLigands.length > 0 ? { includedLigands: entities.includedLigands } : {}),
              ...(entities.filteredBuffers.length > 0 ? { filteredBuffers: entities.filteredBuffers } : {}),
              ...(entities.nucleicChains.length > 0 ? { nucleicChains: entities.nucleicChains } : {}),
            },
            rawSpecText:
              compiled.kind === 'custom-yaml' || compiled.kind === 'template-custom-yaml'
                ? compiled.customSpecYaml
                : JSON.stringify(
                    {
                      ...(compiled.contig ? { contig: compiled.contig } : { input: 'template' }),
                      conditions: compiled.conditions,
                      nsteps: compiled.nsteps,
                      recycleSchedule: compiled.recycleSchedule,
                      earlyStop: compiled.earlyStop,
                      seed: compiled.seed,
                      ...(compiled.effectivePartialFraction != null
                        ? { partialTrajectoryFractionSteps: compiled.effectivePartialFraction }
                        : {}),
                      ...(compiled.inverseFolding ? { inverseFolding: true } : {}),
                    },
                    null,
                    2,
                  ),
          };
        }
      } catch (err) {
        availability = 'error';
        pushIssue({
          severity: 'error',
          source: 'compiler',
          message: err instanceof Error ? err.message : String(err),
        });
      }
    }
  } else if (node.adapterId === 'genie3') {
    const taskMode = String(node.params.taskMode ?? 'unconditional');
    const template = resolvedInputs.structure as StructureArtifact | undefined;
    const condArtifact = resolvedInputs.condition as ConditionSpecArtifact | undefined;
    const needsTemplate =
      taskMode === 'motif' ||
      taskMode === 'binder' ||
      Boolean(String(node.params.lockChain ?? '').trim()) ||
      Boolean(String(node.params.motifSpan ?? '').trim());
    const missingStructUp = missingUpstreamByPort.get('structure');

    if (needsTemplate && !template && !condArtifact?.inputCif && missingStructUp) {
      availability = 'pending-upstream';
      missingUpstreamNodeIds = [missingStructUp];
      payload = {
        format: 'genie3-spec',
        headline: `Genie 3 (${taskMode}) · waiting for upstream structure`,
        compiledSummary: {
          taskMode,
          length: Number(node.params.length ?? 80),
          nSampleStep: Number(node.params.nSampleStep ?? 100),
          atomGranularity: 'ca-only',
        },
        rawSpecText: JSON.stringify(node.params, null, 2),
      };
    } else {
      if (template || !needsTemplate) {
        runRuntimeValidation();
      }
      try {
        const resolved = resolveGenie3CompiledInput(node.params, template, condArtifact);
        const spec = resolved.compiledSpec;

        if (!spec || spec.mode === 'unconditional') {
          const track = buildSyntheticDesignedChain('A', resolved.totalLength, 0, 'scaffold', 'A');
          chains = [track];
          totalResidues = resolved.totalLength;
          designedResidues = resolved.totalLength;
          fixedResidues = 0;
          estimatedTokens = resolved.totalLength;
          fastaEntries = [
            {
              id: 'genie3-uncond',
              header: `>Genie3_unconditional (${resolved.totalLength} aa)`,
              sequence: track.sequence,
              chainLengths: [resolved.totalLength],
              selected: true,
            },
          ];
        } else if (spec.mode === 'binder') {
          // Genie 3 binder-first convention: Chain A (asym 0) = binder, Chain B (asym 1) = fixed target
          const binderLen = spec.chainLengths[0] ?? resolved.length;
          const binderTrack = buildSyntheticDesignedChain('A', binderLen, 0, 'binder', 'A');

          const rawTplText = template?.cifText ?? template?.pdbText ?? resolved.effectiveCond?.inputCif ?? '';
          const tplEntry = getOrParseCachedStructure(
            template?.id ?? fastTextKey('g3-tpl', rawTplText),
            rawTplText,
          );
          const parsed = tplEntry.parsed;

          const selByTplIdx = new Map<number, { outputIndex: number; isHotspot: boolean }>();
          for (const sel of spec.selectedSourceResidues) {
            selByTplIdx.set(sel.templateIndex, {
              outputIndex: sel.outputIndex,
              isHotspot: spec.condInterfaceMask[sel.outputIndex] > 0.5,
            });
          }

          const tplChainMap = new Map<string, PreviewChainTrack>();
          const allTplRes: PreviewResidue[] = [];
          let targetCount = 0;

          for (let i = 0; i < parsed.nResidues; i++) {
            const src = parsed.sourceResidues[i];
            const hit = selByTplIdx.get(i);
            const isTarget = Boolean(hit);
            if (isTarget) targetCount++;

            let track = tplChainMap.get(src.authChainId);
            if (!track) {
              track = {
                chainId: src.authChainId,
                ...(isTarget ? { labelChainId: 'B', roleTag: 'target' } : {}),
                entityKind: 'protein',
                sequence: '',
                residues: [],
              };
              tplChainMap.set(src.authChainId, track);
            } else if (isTarget && !track.labelChainId) {
              track.labelChainId = 'B';
              track.roleTag = 'target';
            }

            let gapBefore = 0;
            if (track.residues.length > 0) {
              const prev = track.residues[track.residues.length - 1];
              const delta = src.resNum - prev.authSeqId - 1;
              if (delta > 0) gapBefore = delta;
            }

            const aa1 = parsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
            const roles: ResidueRole[] = isTarget
              ? hit!.isHotspot
                ? ['fixed-backbone', 'fixed-sequence', 'hotspot']
                : ['fixed-backbone', 'fixed-sequence']
              : ['cropped-out'];

            const pr: PreviewResidue = {
              flatIndex: hit ? hit.outputIndex : -1,
              authChain: src.authChainId,
              ...(isTarget ? { labelChain: 'B', labelSeqId: spec.residueIndex[hit!.outputIndex] } : {}),
              authSeqId: src.resNum,
              insCode: src.insCode,
              token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
              aa1,
              resName: src.resName,
              roles,
              ...(gapBefore > 0 ? { gapBefore } : {}),
              ...(tplEntry.bfactors ? { bfactor: tplEntry.bfactors[i] } : {}),
              backboneAtoms: formatBackboneAtoms(parsed.atomPresence[i], template?.atomGranularity),
            };
            track.residues.push(pr);
            if (isTarget) track.sequence += aa1;
            allTplRes.push(pr);
          }

          chains = [binderTrack, ...tplChainMap.values()];
          highlightLayers = buildHighlightLayersFromResidues(allTplRes);
          totalResidues = spec.length;
          designedResidues = binderLen;
          fixedResidues = targetCount;
          estimatedTokens = spec.length;

          const targetSeq = [...tplChainMap.values()].map((t) => t.sequence).join('');
          fastaEntries = [
            {
              id: 'genie3-binder',
              header: `>Genie3_binder (Chain A=${binderLen} aa binder, Chain B=${targetCount} aa target)`,
              sequence: binderTrack.sequence + targetSeq,
              chainLengths: spec.chainLengths,
              selected: true,
            },
          ];

          if (template) {
            previewStructurePdb = template.pdbText || template.cifText || null;
            previewStructureMeta = {
              label: template.label,
              atomGranularity: template.atomGranularity,
              bfactorSemantics: template.bfactorSemantics,
              length: template.length,
            };
          }
        } else {
          // Motif scaffolding mode (midpoint contig resolution)
          const rawTplText = template?.cifText ?? template?.pdbText ?? resolved.effectiveCond?.inputCif ?? '';
          const tplEntry = rawTplText
            ? getOrParseCachedStructure(template?.id ?? fastTextKey('g3-motif-tpl', rawTplText), rawTplText)
            : undefined;

          const selByOutIdx = new Map(spec.selectedSourceResidues.map((s) => [s.outputIndex, s]));
          const selectedTplIndices = new Set(spec.selectedSourceResidues.map((s) => s.templateIndex));

          const scaffoldResidues: PreviewResidue[] = [];
          let seq = '';
          let motifCount = 0;

          for (let i = 0; i < spec.length; i++) {
            const sel = selByOutIdx.get(i);
            if (sel) {
              motifCount++;
              seq += sel.aa1;
              scaffoldResidues.push({
                flatIndex: i,
                authChain: sel.authChainId,
                labelChain: 'A',
                authSeqId: sel.resNum,
                insCode: sel.insCode,
                labelSeqId: i + 1,
                token: formatResidueToken(sel.authChainId, sel.resNum, sel.insCode),
                aa1: sel.aa1,
                resName: sel.resName,
                roles: ['motif', 'fixed-backbone', 'fixed-sequence'],
                ...(tplEntry?.bfactors && sel.templateIndex >= 0
                  ? { bfactor: tplEntry.bfactors[sel.templateIndex] }
                  : {}),
                backboneAtoms:
                  tplEntry && sel.templateIndex >= 0
                    ? formatBackboneAtoms(tplEntry.parsed.atomPresence[sel.templateIndex], template?.atomGranularity)
                    : 'CA-only',
              });
            } else {
              seq += '-';
              scaffoldResidues.push({
                flatIndex: i,
                authChain: 'A',
                labelChain: 'A',
                authSeqId: i + 1,
                insCode: '',
                labelSeqId: i + 1,
                token: formatResidueToken('A', i + 1),
                aa1: '-',
                resName: 'UNK',
                roles: ['designable'],
              });
            }
          }

          chains = [
            {
              chainId: 'A',
              labelChainId: 'A',
              entityKind: 'protein',
              roleTag: 'scaffold',
              sequence: seq,
              residues: scaffoldResidues,
            },
          ];

          const croppedExtras: Array<{ chain: string; seqId: number; insCode?: string }> = [];
          if (tplEntry) {
            for (let i = 0; i < tplEntry.parsed.nResidues; i++) {
              if (!selectedTplIndices.has(i)) {
                const src = tplEntry.parsed.sourceResidues[i];
                croppedExtras.push({
                  chain: src.authChainId,
                  seqId: src.resNum,
                  ...(src.insCode ? { insCode: src.insCode } : {}),
                });
              }
            }
          }
          highlightLayers = buildHighlightLayersFromResidues(scaffoldResidues, {
            'cropped-out': croppedExtras,
          });

          totalResidues = spec.length;
          fixedResidues = motifCount;
          designedResidues = spec.length - motifCount;
          estimatedTokens = spec.length;

          fastaEntries = [
            {
              id: 'genie3-motif',
              header: `>Genie3_motif_scaffold (${spec.length} aa · ${motifCount} motif aa)`,
              sequence: seq,
              chainLengths: spec.chainLengths,
              selected: true,
            },
          ];

          if (template) {
            previewStructurePdb = template.pdbText || template.cifText || null;
            previewStructureMeta = {
              label: template.label,
              atomGranularity: template.atomGranularity,
              bfactorSemantics: template.bfactorSemantics,
              length: template.length,
            };
          }
        }

        if (template?.diagnostics) {
          const ligClass = classifyLigands(template.diagnostics.ligandNames);
          entities = {
            includedLigands: ligClass.includedLigands,
            filteredBuffers: ligClass.filteredBuffers,
            nucleicChains: [...template.diagnostics.nucleicAcidChains],
          };
        }

        const hotspotCount = spec
          ? spec.condInterfaceMask.reduce((acc, v) => acc + (v > 0.5 ? 1 : 0), 0)
          : 0;

        payload = {
          format: 'genie3-spec',
          headline: `Genie 3 · ${resolved.summary}`,
          compiledSummary: {
            mode: resolved.mode,
            totalLength: resolved.totalLength,
            chainLengths: resolved.chainLengths.map(String),
            chainConvention:
              resolved.mode === 'binder'
                ? 'Chain A = binder, Chain B = fixed target'
                : 'Chain A = scaffold',
            atomGranularity: 'ca-only',
            nSampleStep: resolved.nSampleStep,
            effectiveSteps: resolved.effectiveSteps,
            directionScale: resolved.directionScale,
            noiseScale: resolved.noiseScale,
            predictSequence: resolved.predictSequence,
            hotspotCount,
            seed: resolved.seed,
          },
          rawSpecText: JSON.stringify(
            {
              mode: resolved.mode,
              totalLength: resolved.totalLength,
              chainLengths: resolved.chainLengths,
              contig: resolved.effectiveCond?.contig ?? `${resolved.length}-${resolved.length}`,
              designedResidues,
              fixedResidues,
              hotspotCount,
              nSampleStep: resolved.nSampleStep,
              effectiveSteps: resolved.effectiveSteps,
              directionScale: resolved.directionScale,
              noiseScale: resolved.noiseScale,
              predictSequence: resolved.predictSequence,
              caOnly: true,
              seed: resolved.seed,
            },
            null,
            2,
          ),
        };
      } catch (err) {
        availability = 'error';
        pushIssue({
          severity: 'error',
          source: 'compiler',
          message: err instanceof Error ? err.message : String(err),
        });
      }
    }
  } else if (node.adapterId === 'localfold-af2') {
    const seqSet = resolvedInputs.sequences as SequenceSet | undefined;
    const modelKey = String(node.params.model ?? 'monomer');
    const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelKey) ?? LOCALFOLD_MODEL_OPTIONS[0];
    const isEsmFold = meta.family === 'esmfold2';
    const baseCfg = resolveLocalFoldEffectiveConfig({ modelKey, params: node.params });
    const {
      recycles,
      tolerance,
      diffusionSteps,
      msaMode,
      msaA3mText,
      ligandCcdCodes,
      dnaSeqs,
      rnaSeqs,
    } = baseCfg;
    const foldSettings: CompiledModelPayloadView['compiledSummary'] = {
      model: modelKey,
      family: meta.family,
      msaMode,
    };
    if (isEsmFold) {
      foldSettings.esmfold2Steps = baseCfg.effectiveEsmfold2Steps;
      if (baseCfg.esmfold2StepsAdjusted) {
        foldSettings.requestedEsmfold2Steps = baseCfg.requestedEsmfold2Steps;
      }
    } else {
      foldSettings.recycles = recycles;
      foldSettings.tolerance = tolerance;
      if (meta.family === 'af3') foldSettings.diffusionSteps = diffusionSteps;
    }

    if (!seqSet) {
      const missingUp = missingUpstreamByPort.get('sequences');
      if (missingUp) {
        availability = 'pending-upstream';
        missingUpstreamNodeIds = [missingUp];
      } else {
        availability = 'error';
      }
      payload = {
        format: 'localfold-batch',
        headline: missingUp
          ? `LocalFold (${meta.shortLabel}) · waiting for upstream sequences`
          : `LocalFold (${meta.shortLabel}) · connect a SequenceSet input`,
        compiledSummary: foldSettings,
        rawSpecText: JSON.stringify(node.params, null, 2),
      };
    } else {
      const validInputs = runRuntimeValidation();
      const work = selectFoldWorkItems(seqSet, node.params);
      if (isEsmFold) {
        const maxResidueLength = work.selected.reduce(
          (mx, c) => Math.max(mx, c.sequence.replace(/[\s/:,]+/g, '').length),
          0,
        );
        const effCfg = resolveLocalFoldEffectiveConfig({
          modelKey,
          params: node.params,
          maxResidueLength,
        });
        foldSettings.esmfold2Steps = effCfg.effectiveEsmfold2Steps;
        if (effCfg.esmfold2StepsAdjusted) {
          foldSettings.requestedEsmfold2Steps = effCfg.requestedEsmfold2Steps;
        }
      }
      const lockedChains: string[] | undefined = Array.isArray(seqSet.lockedChains)
        ? seqSet.lockedChains
        : Array.isArray(node.params.lockedChains)
          ? (node.params.lockedChains as unknown[]).map((c) => String(c).trim()).filter(Boolean)
          : undefined;
      const chainRoles: Record<string, 'target' | 'binder' | 'scaffold'> | undefined =
        seqSet.chainRoles ??
        (node.params.chainRoles && typeof node.params.chainRoles === 'object'
          ? (node.params.chainRoles as Record<string, 'target' | 'binder' | 'scaffold'>)
          : undefined);

      const selectedSet = new Set(work.selected.map((c) => c.key ?? c.id));
      fastaEntries = seqSet.candidates.map((cand) => {
        const isSelected = selectedSet.has(cand.key ?? cand.id);
        const scorePart =
          typeof cand.score === 'number' && Number.isFinite(cand.score)
            ? `score ${cand.score.toFixed(2)}`
            : undefined;
        return {
          id: cand.id,
          header: `>${cand.id}${scorePart ? ` | ${scorePart}` : ''}`,
          sequence: cand.sequence,
          ...(cand.chainLengths ? { chainLengths: cand.chainLengths } : {}),
          selected: isSelected,
          ...(isSelected
            ? scorePart
              ? { scoreLabel: scorePart }
              : {}
            : { scoreLabel: `excluded (maxSequences=${work.maxSequences})` }),
        };
      });

      const rep = work.selected[0] ?? seqSet.candidates[0];
      let offMask: boolean[] = [];
      let uniqueChains: string[] = [];
      let a3mDepth: number | null = null;
      let a3mDistinctDepth: number | null = null;
      let chainLengths: number[] = [];

      if (rep && (!isEsmFold || validInputs)) {
        const cleanSeq = rep.sequence
          .toUpperCase()
          .replace(/[\s/:,]+/g, '')
          .replace(/[^ACDEFGHIKLMNPQRSTVWYX]/g, 'X');
        const resolvedEf2Limit =
          foldLimitOpts?.resolveNodeFoldLimits?.(node)?.maxFoldResidues ??
          foldLimitOpts?.maxFoldResidues ??
          undefined;
        chainLengths = isEsmFold
          ? esmfold2Chains(rep, resolvedEf2Limit ?? 512).map((chain) => chain.length)
          : Array.isArray(rep.chainLengths) &&
          rep.chainLengths.length > 0 &&
          rep.chainLengths.every((n) => Number.isInteger(n) && n > 0) &&
          rep.chainLengths.reduce((a, b) => a + b, 0) === cleanSeq.length
            ? rep.chainLengths
            : [cleanSeq.length];
        const chainSeqs: string[] = [];
        let cursor = 0;
        for (const len of chainLengths) {
          chainSeqs.push(cleanSeq.slice(cursor, cursor + len));
          cursor += len;
        }
        uniqueChains = [...new Set(chainIdsFromLengths(chainLengths))];

        if (msaMode === 'target-only') {
          offMask = computeTargetOnlyOffMask(uniqueChains, lockedChains, chainRoles);
        } else if (msaMode === 'single') {
          offMask = uniqueChains.map(() => true);
        } else {
          offMask = uniqueChains.map(() => false);
        }

        if (msaMode === 'a3m') {
          try {
            const validated = validateAndParseA3mText(msaA3mText, chainSeqs);
            a3mDepth = validated.depth;
            a3mDistinctDepth = validated.distinctDepth;
          } catch (err) {
            availability = 'error';
            pushIssue({
              severity: 'error',
              source: 'compiler',
              message: err instanceof Error ? err.message : String(err),
            });
          }
        }

        const lockedSet = new Set((lockedChains ?? []).map((c) => c.trim().toUpperCase()));
        let flatOffset = 0;
        const tracks: PreviewChainTrack[] = [];

        for (let c = 0; c < chainSeqs.length; c++) {
          const chId = uniqueChains[c] ?? String.fromCharCode(65 + c);
          const seq = chainSeqs[c];
          const isLocked = lockedSet.has(chId) || chainRoles?.[chId] === 'target';
          const msaStatus: PreviewChainTrack['msaStatus'] =
            msaMode === 'single'
              ? 'single'
              : msaMode === 'a3m'
                ? 'a3m-provided'
                : msaMode === 'target-only'
                  ? offMask[c]
                    ? 'target-only-masked'
                    : 'mmseqs2-search'
                  : 'mmseqs2-search';
          const msaDepth =
            msaStatus === 'single' || msaStatus === 'target-only-masked'
              ? 1
              : msaStatus === 'a3m-provided'
                ? a3mDepth
                : null;

          const residues: PreviewResidue[] = [];
          for (let k = 0; k < seq.length; k++) {
            const aa1 = seq[k];
            residues.push({
              flatIndex: flatOffset + k,
              authChain: chId,
              authSeqId: k + 1,
              insCode: '',
              token: formatResidueToken(chId, k + 1),
              aa1,
              resName: AA1_TO_AA3[aa1] ?? 'UNK',
              roles: isLocked ? ['fixed-sequence'] : ['designable'],
            });
          }
          flatOffset += seq.length;

          tracks.push({
            chainId: chId,
            entityKind: 'protein',
            ...(chainRoles?.[chId] ? { roleTag: chainRoles[chId] } : {}),
            sequence: seq,
            residues,
            msaStatus,
            msaDepth,
          });
        }

        const nucleicChainIds: string[] = [];
        const supportsHolo = meta.family === 'af3' || meta.family === 'esmfold2';
        if (supportsHolo) {
          const used = new Set(tracks.map((t) => t.chainId));
          for (const dSeq of dnaSeqs) {
            const chId = nextChainLetter(used);
            used.add(chId);
            nucleicChainIds.push(chId);
            const residues: PreviewResidue[] = Array.from(dSeq, (nt, k) => ({
              flatIndex: flatOffset + k,
              authChain: chId,
              authSeqId: k + 1,
              insCode: '',
              token: formatResidueToken(chId, k + 1),
              aa1: nt,
              resName: `D${nt}`,
              roles: ['nucleic'],
            }));
            flatOffset += dSeq.length;
            tracks.push({ chainId: chId, entityKind: 'dna', sequence: dSeq, residues });
          }
          for (const rSeq of rnaSeqs) {
            const chId = nextChainLetter(used);
            used.add(chId);
            nucleicChainIds.push(chId);
            const residues: PreviewResidue[] = Array.from(rSeq, (nt, k) => ({
              flatIndex: flatOffset + k,
              authChain: chId,
              authSeqId: k + 1,
              insCode: '',
              token: formatResidueToken(chId, k + 1),
              aa1: nt,
              resName: nt,
              roles: ['nucleic'],
            }));
            flatOffset += rSeq.length;
            tracks.push({ chainId: chId, entityKind: 'rna', sequence: rSeq, residues });
          }
          entities = {
            includedLigands: ligandCcdCodes,
            filteredBuffers: [],
            nucleicChains: nucleicChainIds,
          };
        }

        chains = tracks;
        totalResidues = flatOffset;
        fixedResidues = tracks.reduce(
          (sum, t) => sum + t.residues.filter((r) => r.roles.includes('fixed-sequence')).length,
          0,
        );
        designedResidues = totalResidues - fixedResidues;
        estimatedTokens = totalResidues + (meta.family === 'af3' || meta.family === 'esmfold2' ? ligandCcdCodes.length : 0);
      }

      const ctxStruct = findContextStructure(graph, nodeId, mutableOutputs);
      if (ctxStruct) {
        previewStructurePdb = ctxStruct.pdbText || ctxStruct.cifText || '';
        previewStructureMeta = {
          label: ctxStruct.label,
          atomGranularity: ctxStruct.atomGranularity,
          bfactorSemantics:
            ctxStruct.bfactorSemantics ??
            (ctxStruct.sourceModelId === 'pdb-input' ? 'crystal-b' : undefined),
          length: ctxStruct.length,
        };
      }

      const af2Budgets =
        meta.family === 'af2' || meta.family === 'af2-multimer'
          ? computeAf2MsaBudgets(a3mDepth ?? 1, Math.max(1, uniqueChains.length), a3mDistinctDepth ?? 1)
          : undefined;

      payload = {
        format: 'localfold-batch',
        headline: `LocalFold (${meta.shortLabel}) · ${work.selected.length}/${work.totalCandidates} seqs · ${isEsmFold ? 'ESM-C language model' : `MSA: ${msaMode}`}`,
        compiledSummary: {
          ...foldSettings,
          selectedCount: work.selected.length,
          excludedCount: work.excludedCandidates.length,
          totalCandidates: work.totalCandidates,
          chainLengths: chainLengths.map(String),
          chainMsaPolicy: uniqueChains.map(
            (ch, idx) =>
              `${ch}:${msaMode === 'a3m' ? `a3m(${a3mDepth ?? '?'})` : offMask[idx] ? 'query-only' : 'mmseqs2'}`,
          ),
          ...(a3mDepth !== null ? { a3mDepth } : {}),
          ...(af2Budgets ? { maxMsaSequences: af2Budgets.maxMsaSequences } : {}),
        },
        rawSpecText: JSON.stringify(
          {
            ...foldSettings,
            ...(meta.family === 'af3' || meta.family === 'esmfold2'
              ? { ligandCcdCodes, dnaSequences: dnaSeqs, rnaSequences: rnaSeqs }
              : { af2Budgets }),
            maxSequences: work.maxSequences,
            selectedCandidates: work.selected.map((c) => c.id),
            excludedCandidates: work.excludedCandidates.map((c) => c.id),
            chainMsaOffMask: Object.fromEntries(uniqueChains.map((ch, i) => [ch, offMask[i]])),
            ...(a3mDepth !== null ? { a3mDepth, a3mDistinctDepth } : {}),
          },
          null,
          2,
        ),
      };
    }
  } else if (node.adapterId === 'designability-scorer') {
    const structure = resolvedInputs.structure as StructureArtifact | undefined;
    const folds = resolvedInputs.folds as FoldedStructure | undefined;

    const scThresh = Number(node.params.scRmsdThreshold ?? 2.0);
    const plddtThresh = Number(node.params.plddtThreshold ?? 70);
    const motifThresh = Number(node.params.motifRmsdThreshold ?? 1.0);
    const ipaeThresh = Number(node.params.ipaeThreshold ?? 10.0);
    const ipsaeThresh = Number(node.params.ipsaeThreshold ?? 0);
    const targetAlignedBinderThresh = Number(
      node.params.targetAlignedBinderRmsdThreshold ?? Math.max(scThresh * 2, 4.0),
    );
    const allowExternalReference = Boolean(node.params.allowExternalReference);
    const strictLineage = Boolean(node.params.strictLineage);
    const rawMappingParam = node.params.residueMapping ?? node.params.referenceMapping;

    const missingList = [...allMissingUpstream];
    if (!structure || !folds) {
      if (missingList.length > 0) {
        availability = 'pending-upstream';
        missingUpstreamNodeIds = missingList;
      } else {
        availability = 'error';
      }
    }

    let mappedRefSet: Set<number> | null = null;
    let mappedPairsCount = 0;
    let referenceRelation = 'pending';

    if (structure) {
      const rawStructText = structure.pdbText || structure.cifText || '';
      const structEntry = rawStructText
        ? getOrParseCachedStructure(structure.id, rawStructText)
        : null;
      const parsed = structEntry?.parsed;

      if (folds) {
        const normalizeLineageId = (val: string | undefined | null): string => {
          const s = String(val ?? '').trim();
          return s && s.toLowerCase() !== 'none' ? s : '';
        };
        const lineageSet = new Set<string>(
          [
            normalizeLineageId(folds.parentStructureId),
            normalizeLineageId(folds.sourceStructureId),
          ].filter(Boolean),
        );
        const hasConflict = lineageSet.size > 1;
        const nonRef = [...lineageSet].find((id) => id !== structure.id);
        const hasDisagreement = hasConflict || Boolean(nonRef);

        const firstFold = folds.folds[0];
        const foldLen = firstFold ? Math.floor(firstFold.caCoords.length / 3) : 0;
        const structLen = Math.floor(structure.caCoords.length / 3);

        if (rawMappingParam !== undefined && rawMappingParam !== null && String(rawMappingParam).trim() !== '') {
          try {
            const pairs = parseScorerMappingPreview(rawMappingParam, foldLen, structLen);
            if (pairs) {
              mappedPairsCount = pairs.length;
              mappedRefSet = new Set(pairs.map(([, ri]) => ri));
            }
          } catch (err) {
            availability = 'error';
            pushIssue({
              severity: 'error',
              source: 'compiler',
              message: err instanceof Error ? err.message : String(err),
            });
          }
        } else if (firstFold && foldLen !== structLen) {
          availability = 'error';
          pushIssue({
            severity: 'error',
            source: 'compiler',
            message: `Length mismatch: refolded "${firstFold.sequenceId}" has ${foldLen} aa while reference "${structure.label}" has ${structLen} aa. Provide explicit residueMapping or match structures.`,
          });
        }

        const selfConsistencyValid =
          !mappedRefSet &&
          !allowExternalReference &&
          !hasConflict &&
          lineageSet.size === 1 &&
          !nonRef;
        referenceRelation = mappedRefSet
          ? 'external-mapped'
          : selfConsistencyValid
            ? 'self-consistency'
            : hasDisagreement
              ? 'external-mismatch'
              : 'external-unlinked';

        if (hasDisagreement && !allowExternalReference && !mappedRefSet) {
          pushIssue({
            severity: strictLineage ? 'error' : 'warning',
            source: 'compiler',
            message: `Fold lineage (${[...lineageSet].join(' / ')}) differs from reference "${structure.id}".`,
          });
        }

        fastaEntries = folds.folds.map((f, idx) => ({
          id: f.sequenceId,
          header: `>${f.sequenceId} | pLDDT ${f.meanPlddt.toFixed(1)}`,
          sequence: f.sequence,
          ...(f.chainLengths ? { chainLengths: f.chainLengths } : {}),
          selected: idx === 0,
          scoreLabel: `pLDDT ${f.meanPlddt.toFixed(1)}`,
        }));
      }

      const motifSet = new Set(structure.motifIndices ?? []);
      const lockedSet = new Set((structure.lockedChains ?? []).map((c) => c.toUpperCase()));
      const chainMap = new Map<string, PreviewChainTrack>();
      const allRes: PreviewResidue[] = [];

      for (let i = 0; i < (parsed?.nResidues ?? 0); i++) {
        const src = parsed!.sourceResidues[i];
        let track = chainMap.get(src.authChainId);
        if (!track) {
          const roleTag = structure.chainRoles?.[src.authChainId];
          track = {
            chainId: src.authChainId,
            ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
            entityKind: 'protein',
            ...(roleTag ? { roleTag } : {}),
            sequence: '',
            residues: [],
          };
          chainMap.set(src.authChainId, track);
        }

        let gapBefore = 0;
        if (track.residues.length > 0) {
          const prev = track.residues[track.residues.length - 1];
          const delta = src.resNum - prev.authSeqId - 1;
          if (delta > 0) gapBefore = delta;
        }

        const roles: ResidueRole[] = [];
        if (mappedRefSet) {
          roles.push(mappedRefSet.has(i) ? 'motif' : 'cropped-out');
        } else {
          if (motifSet.has(i)) roles.push('motif');
          if (lockedSet.has(src.authChainId.toUpperCase())) {
            roles.push('fixed-backbone', 'fixed-sequence');
          } else if (roles.length === 0) {
            roles.push('designable');
          }
        }

        const aa1 = parsed!.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
        const pr: PreviewResidue = {
          flatIndex: mappedRefSet && !mappedRefSet.has(i) ? -1 : i,
          authChain: src.authChainId,
          ...(src.labelChainId ? { labelChain: src.labelChainId } : {}),
          authSeqId: src.resNum,
          insCode: src.insCode,
          ...(src.labelSeqId !== undefined ? { labelSeqId: src.labelSeqId } : {}),
          token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
          aa1,
          resName: src.resName,
          roles,
          ...(gapBefore > 0 ? { gapBefore } : {}),
          ...(structEntry?.bfactors ? { bfactor: structEntry.bfactors[i] } : {}),
          backboneAtoms: formatBackboneAtoms(parsed!.atomPresence[i], structure.atomGranularity),
        };
        track.residues.push(pr);
        track.sequence += aa1;
        allRes.push(pr);
      }

      chains = [...chainMap.values()];
      highlightLayers = buildHighlightLayersFromResidues(allRes);

      const ligClass = classifyLigands(
        structure.diagnostics?.ligandNames ?? parsed?.diagnostics.ligandNames ?? [],
      );
      entities = {
        includedLigands: ligClass.includedLigands,
        filteredBuffers: ligClass.filteredBuffers,
        nucleicChains: [
          ...(structure.diagnostics?.nucleicAcidChains ?? parsed?.diagnostics.nucleicAcidChains ?? []),
        ],
      };

      const hasNonProt = Boolean(
        structure.diagnostics?.hasLigands ||
          structure.diagnostics?.hasNucleicAcids ||
          parsed?.diagnostics.hasLigands ||
          parsed?.diagnostics.hasNucleicAcids,
      );
      previewStructurePdb =
        structure.cifText && (hasNonProt || !structure.pdbText)
          ? structure.cifText
          : structure.pdbText || structure.cifText || null;
      previewStructureMeta = {
        label: structure.label,
        atomGranularity: structure.atomGranularity,
        bfactorSemantics: structure.bfactorSemantics,
        length: structure.length,
      };
      totalResidues = structure.length;
      fixedResidues = mappedRefSet ? mappedPairsCount : structure.length;
      designedResidues = 0;
      estimatedTokens = structure.length;
    }

    payload = {
      format: 'scorer-mapping',
      headline:
        structure && folds
          ? `Scorer · ${folds.folds.length} fold${folds.folds.length === 1 ? '' : 's'} vs ${structure.label} (${referenceRelation})`
          : `Designability Scorer · waiting for upstream ${!structure && !folds ? 'structure & folds' : !structure ? 'structure' : 'folds'}`,
      compiledSummary: {
        referenceRelation,
        foldCount: folds?.folds.length ?? 0,
        scRmsdThreshold: scThresh,
        plddtThreshold: plddtThresh,
        motifRmsdThreshold: motifThresh,
        ipaeThreshold: ipaeThresh,
        ipsaeThreshold: ipsaeThresh,
        targetAlignedBinderRmsdThreshold: targetAlignedBinderThresh,
        allowExternalReference,
        strictLineage,
        ...(mappedPairsCount > 0 ? { mappedPairs: mappedPairsCount } : {}),
      },
      rawSpecText: JSON.stringify(
        {
          referenceStructureId: structure?.id ?? null,
          foldParentStructureId: folds?.parentStructureId ?? null,
          referenceRelation,
          thresholds: {
            scRmsdThreshold: scThresh,
            plddtThreshold: plddtThresh,
            motifRmsdThreshold: motifThresh,
            ipaeThreshold: ipaeThresh,
            ipsaeThreshold: ipsaeThresh,
            targetAlignedBinderRmsdThreshold: targetAlignedBinderThresh,
          },
          allowExternalReference,
          strictLineage,
          mappedPairsCount,
        },
        null,
        2,
      ),
    };
  } else if (node.adapterId === 'gnina') {
    const structure = resolvedInputs.structure as StructureArtifact | undefined;
    const taskMode = String(node.params.taskMode ?? 'score');
    const precision = String(node.params.precision ?? 'fp16');
    const ligandCode = String(node.params.ligandCode ?? '').trim();
    const ligandSdf = String(node.params.ligandSdf ?? '').trim();
    const dockLigandCcd = String(node.params.dockLigandCcd ?? '').trim().toUpperCase();
    const pocketResiduesSpec = String(node.params.pocketResidues ?? '').trim();
    const exhaustiveness = Math.max(1, Math.min(16, Math.round(Number(node.params.exhaustiveness ?? 8))));
    const numModes = Math.max(1, Math.min(9, Math.round(Number(node.params.numModes ?? 5))));
    const sortBy = String(node.params.sortBy ?? 'cnnScore').trim();
    const autoboxAdd = Math.max(2.0, Math.min(12.0, Number(node.params.autoboxAdd ?? 4.0)));
    const seed = Math.max(0, Math.min(999999, Math.round(Number(node.params.seed ?? 42))));
    const includeMetals = node.params.includeMetals !== false;
    const pocketRadius = Math.max(3.0, Math.min(8.0, Number(node.params.pocketRadius ?? 5.0)));
    const maxPocketResidues = Math.max(4, Math.min(48, Math.round(Number(node.params.maxPocketResidues ?? 24))));
    const minimizeIterations = Math.max(1, Math.min(12, Math.round(Number(node.params.minimizeIterations ?? 4))));
    const modeLabel =
      taskMode === 'dock'
        ? `Dock Ligand (exh=${exhaustiveness}, ${numModes}p)`
        : taskMode === 'pocket-scan'
          ? 'Pocket Scan'
          : taskMode === 'minimize'
            ? `Minimize (${minimizeIterations} iters)`
            : 'Score Pose';

    if (!structure) {
      const missingUp = missingUpstreamByPort.get('structure');
      if (missingUp) {
        availability = 'pending-upstream';
        missingUpstreamNodeIds = [missingUp];
      } else {
        availability = 'error';
      }
      payload = {
        format: 'scorer-mapping',
        headline: missingUp
          ? `GNINA 1.3 · ${modeLabel} (${precision.toUpperCase()}) · waiting for upstream structure`
          : `GNINA 1.3 · ${modeLabel} (${precision.toUpperCase()}) · connect a StructureArtifact`,
        compiledSummary: {
          taskMode,
          precision,
          ligandCode: dockLigandCcd || ligandCode || 'auto',
          externalLigand: ligandSdf.length > 0,
          ...(taskMode === 'dock'
            ? {
                exhaustiveness,
                numModes,
                sortBy,
                pocketBox: pocketResiduesSpec || 'holo autobox',
              }
            : {}),
          includeMetals,
        },
        rawSpecText: JSON.stringify(
          {
            taskMode,
            precision,
            ligandCode: ligandCode || 'auto',
            ...(taskMode === 'dock'
              ? { dockLigandCcd: dockLigandCcd || null, pocketResidues: pocketResiduesSpec || null, exhaustiveness, numModes, sortBy, autoboxAdd, seed }
              : {}),
            includeMetals,
          },
          null,
          2,
        ),
      };
    } else {
      runRuntimeValidation();
      const rawText = structure.pdbText || structure.cifText || '';
      const entry = getOrParseCachedStructure(structure.id, rawText);
      const parsed = entry.parsed;
      const prep = extractHoloReceptorAndLigands(rawText, {
        ligandCode: ligandCode || undefined,
        includeMetals,
      });

      const pocketSet = new Set<string>();
      let explicitPocketKeys: string[] = [];
      if (taskMode === 'dock') {
        if (pocketResiduesSpec && prep.receptorPdb.trim()) {
          explicitPocketKeys = parsePocketResidueTokens(pocketResiduesSpec, prep.receptorPdb);
          for (const key of explicitPocketKeys) {
            const parts = key.split('/');
            if (parts.length >= 2) {
              const ins = parts.length === 4 ? parts[2] : '';
              pocketSet.add(`${parts[0]}${parts[1]}${ins}`);
            }
          }
          if (explicitPocketKeys.length === 0) {
            pushIssue({
              severity: 'error',
              source: 'compiler',
              message: `Pocket Box Residues "${pocketResiduesSpec}" matched 0 residues in "${structure.label}". Use tokens like "A45, A79, A90".`,
            });
          }
        } else if (prep.selectedLigand && prep.receptorPdb.trim()) {
          const pocket = findPocketResidues(prep.receptorPdb, prep.selectedLigand.atoms, 5.0, 24);
          for (const r of pocket) pocketSet.add(r.residueId);
        }
      } else if (taskMode === 'pocket-scan' && prep.selectedLigand && prep.receptorPdb.trim()) {
        const pocket = findPocketResidues(
          prep.receptorPdb,
          prep.selectedLigand.atoms,
          pocketRadius,
          maxPocketResidues,
        );
        for (const r of pocket) pocketSet.add(r.residueId);
      }

      const hasDockSource = taskMode === 'dock' && Boolean(dockLigandCcd);
      if (!ligandSdf && !hasDockSource && !prep.selectedLigand) {
        pushIssue({
          severity: 'error',
          source: 'compiler',
          message:
            ligandCode && prep.detectedLigandCodes.length > 0
              ? `Target ligand "${ligandCode}" not found in "${structure.label}" (detected: ${prep.detectedLigandCodes.join(', ')}).`
              : taskMode === 'dock'
                ? `No ligand specified for docking into "${structure.label}". Enter a Ligand CCD Code (e.g. BTN, ATP, I4B), paste an SDF/MOL2, or wire a holo complex.`
                : `No organic small-molecule ligand found in "${structure.label}". Wire a holo complex or paste an SDF/MOL2 in Advanced parameters.`,
        });
      } else if (
        taskMode === 'dock' &&
        (ligandSdf || hasDockSource) &&
        !pocketResiduesSpec &&
        !prep.selectedLigand
      ) {
        pushIssue({
          severity: 'error',
          source: 'compiler',
          message: `Docking into apo structure "${structure.label}" requires Pocket Box Residues (e.g. "A45, A79, A90") to define the binding box.`,
        });
      }

      const chainMap = new Map<string, PreviewChainTrack>();
      const allRes: PreviewResidue[] = [];
      for (let i = 0; i < (parsed?.nResidues ?? 0); i++) {
        const src = parsed!.sourceResidues[i];
        let track = chainMap.get(src.authChainId);
        if (!track) {
          const roleTag = structure.chainRoles?.[src.authChainId];
          track = {
            chainId: src.authChainId,
            ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
            entityKind: 'protein',
            ...(roleTag ? { roleTag } : {}),
            sequence: '',
            residues: [],
          };
          chainMap.set(src.authChainId, track);
        }
        let gapBefore = 0;
        if (track.residues.length > 0) {
          const prev = track.residues[track.residues.length - 1];
          const delta = src.resNum - prev.authSeqId - 1;
          if (delta > 0) gapBefore = delta;
        }
        const rid = `${src.authChainId}${src.resNum}${src.insCode ?? ''}`;
        const isPocket = pocketSet.has(rid);
        const aa1 = parsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
        const pr: PreviewResidue = {
          flatIndex: i,
          authChain: src.authChainId,
          ...(src.labelChainId ? { labelChain: src.labelChainId } : {}),
          authSeqId: src.resNum,
          insCode: src.insCode,
          ...(src.labelSeqId !== undefined ? { labelSeqId: src.labelSeqId } : {}),
          token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
          aa1,
          resName: src.resName,
          roles: isPocket ? ['hotspot'] : ['fixed-backbone'],
          ...(gapBefore > 0 ? { gapBefore } : {}),
          ...(entry.bfactors ? { bfactor: entry.bfactors[i] } : {}),
          backboneAtoms: formatBackboneAtoms(parsed.atomPresence[i], structure.atomGranularity),
        };
        track.residues.push(pr);
        track.sequence += aa1;
        allRes.push(pr);
      }

      chains = [...chainMap.values()];
      highlightLayers = buildHighlightLayersFromResidues(allRes);
      const ligClass = classifyLigands(structure.diagnostics?.ligandNames ?? parsed.diagnostics.ligandNames);
      const previewLigands = [
        ...(hasDockSource ? [dockLigandCcd] : []),
        ...(prep.selectedLigand ? [prep.selectedLigand.compId] : prep.detectedLigandCodes),
        ...prep.keptMetals,
      ];
      entities = {
        includedLigands: [...new Set(previewLigands)],
        filteredBuffers: ligClass.filteredBuffers,
        nucleicChains: [...(structure.diagnostics?.nucleicAcidChains ?? parsed.diagnostics.nucleicAcidChains)],
      };

      totalResidues = parsed.nResidues;
      designedResidues = pocketSet.size;
      fixedResidues = Math.max(0, parsed.nResidues - pocketSet.size);
      estimatedTokens = parsed.nResidues;

      previewStructurePdb = structure.pdbText || structure.cifText || '';
      previewStructureMeta = {
        label: structure.label,
        atomGranularity: structure.atomGranularity ?? 'all-atom',
        bfactorSemantics: structure.bfactorSemantics,
        length: parsed.nResidues,
      };

      const targetLigLabel = ligandSdf
        ? 'external SDF/MOL2'
        : hasDockSource
          ? `RCSB CCD ${dockLigandCcd}`
          : prep.selectedLigand
            ? `${prep.selectedLigand.compId} (${prep.selectedLigand.heavyAtomCount} heavy atoms)`
            : 'none detected';

      const dockBoxDesc =
        explicitPocketKeys.length > 0
          ? `${explicitPocketKeys.length} pocket res (+${autoboxAdd.toFixed(1)} Å)`
          : prep.selectedLigand
            ? `autobox ${prep.selectedLigand.compId} (+${autoboxAdd.toFixed(1)} Å)`
            : 'unboxed';

      payload = {
        format: 'scorer-mapping',
        headline:
          taskMode === 'dock'
            ? `GNINA 1.3 · Dock ${targetLigLabel} · ${dockBoxDesc} · exh=${exhaustiveness}`
            : taskMode === 'pocket-scan'
              ? `GNINA 1.3 · Pocket Scan (${pocketSet.size} res ≤ ${pocketRadius.toFixed(1)} Å) · ${targetLigLabel}`
              : `GNINA 1.3 · ${modeLabel} (${precision.toUpperCase()}) · ${targetLigLabel}`,
        compiledSummary: {
          taskMode,
          precision,
          selectedLigand: targetLigLabel,
          detectedLigands: prep.detectedLigandCodes,
          keptMetals: prep.keptMetals,
          ...(taskMode === 'dock'
            ? {
                pocketBox: dockBoxDesc,
                exhaustiveness,
                numModes,
                sortBy,
              }
            : {}),
          ...(taskMode === 'pocket-scan' ? { pocketResidues: pocketSet.size, pocketRadius } : {}),
        },
        rawSpecText: JSON.stringify(
          {
            taskMode,
            precision,
            selectedLigand: ligandSdf
              ? 'external'
              : hasDockSource
                ? `rcsb:${dockLigandCcd}`
                : prep.selectedLigand?.instanceKey ?? null,
            detectedLigands: prep.detectedLigandCodes,
            keptMetals: prep.keptMetals,
            ...(taskMode === 'dock'
              ? {
                  pocketBox: dockBoxDesc,
                  explicitPocketKeys,
                  autoboxAdd,
                  exhaustiveness,
                  numModes,
                  sortBy,
                  seed,
                }
              : {}),
            ...(taskMode === 'pocket-scan'
              ? { pocketRadius, maxPocketResidues, pocketResidues: [...pocketSet] }
              : {}),
            ...(taskMode === 'minimize' ? { minimizeIterations } : {}),
          },
          null,
          2,
        ),
      };
    }
  } else if (node.adapterId === 'thermompnn') {
    const structure = resolvedInputs.structure as StructureArtifact | undefined;
    const precision = String(node.params.precision ?? 'fp16');
    const chainFilter = String(node.params.chain ?? '').trim();
    const bfactorAggregate = String(node.params.bfactorAggregate ?? 'best-stabilizing');

    if (!structure) {
      const missingUp = missingUpstreamByPort.get('structure');
      if (missingUp) {
        availability = 'pending-upstream';
        missingUpstreamNodeIds = [missingUp];
      } else {
        availability = 'error';
      }
      payload = {
        format: 'scorer-mapping',
        headline: missingUp
          ? `ThermoMPNN (${precision.toUpperCase()}) · waiting for upstream structure`
          : `ThermoMPNN (${precision.toUpperCase()}) · connect a StructureArtifact input`,
        compiledSummary: {
          precision,
          chain: chainFilter || 'all',
          bfactorAggregate,
        },
        rawSpecText: JSON.stringify({ precision, chain: chainFilter, bfactorAggregate }, null, 2),
      };
    } else {
      runRuntimeValidation();
      const rawText = structure.pdbText || structure.cifText || '';
      const entry = getOrParseCachedStructure(structure.id, rawText);
      const parsed = entry.parsed;
      const availableChains = [...new Set(parsed.sourceResidues.map((s) => s.authChainId))];
      const requestedChains = chainFilter
        .split(/[,\s]+/)
        .map((c) => c.trim().toUpperCase())
        .filter(Boolean);
      const requestedChainSet = new Set(requestedChains);
      const missingChains = requestedChains.filter(
        (rc) => !availableChains.some((c) => c.toUpperCase() === rc),
      );
      if (missingChains.length > 0) {
        pushIssue({
          severity: 'warning',
          source: 'compiler',
          message: `Target chain(s) "${missingChains.join(', ')}" not found in structure (available: ${availableChains.join(', ')}).`,
        });
      }

      const chainMap = new Map<string, PreviewChainTrack>();
      const allRes: PreviewResidue[] = [];
      let scannedCount = 0;

      for (let i = 0; i < (parsed?.nResidues ?? 0); i++) {
        const src = parsed!.sourceResidues[i];
        let track = chainMap.get(src.authChainId);
        if (!track) {
          const roleTag = structure.chainRoles?.[src.authChainId];
          track = {
            chainId: src.authChainId,
            ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
            entityKind: 'protein',
            ...(roleTag ? { roleTag } : {}),
            sequence: '',
            residues: [],
          };
          chainMap.set(src.authChainId, track);
        }

        let gapBefore = 0;
        if (track.residues.length > 0) {
          const prev = track.residues[track.residues.length - 1];
          const delta = src.resNum - prev.authSeqId - 1;
          if (delta > 0) gapBefore = delta;
        }

        const inScope =
          requestedChainSet.size === 0 || requestedChainSet.has(src.authChainId.toUpperCase());
        if (inScope) scannedCount++;
        const aa1 = parsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
        const pr: PreviewResidue = {
          flatIndex: inScope ? i : -1,
          authChain: src.authChainId,
          ...(src.labelChainId ? { labelChain: src.labelChainId } : {}),
          authSeqId: src.resNum,
          insCode: src.insCode,
          ...(src.labelSeqId !== undefined ? { labelSeqId: src.labelSeqId } : {}),
          token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
          aa1,
          resName: src.resName,
          roles: inScope ? ['designable'] : ['cropped-out'],
          ...(gapBefore > 0 ? { gapBefore } : {}),
          ...(entry.bfactors ? { bfactor: entry.bfactors[i] } : {}),
          backboneAtoms: formatBackboneAtoms(parsed.atomPresence[i], structure.atomGranularity),
        };
        track.residues.push(pr);
        track.sequence += aa1;
        allRes.push(pr);
      }

      chains = [...chainMap.values()];
      highlightLayers = buildHighlightLayersFromResidues(allRes);
      const ligClass = classifyLigands(structure.diagnostics?.ligandNames ?? parsed.diagnostics.ligandNames);
      entities = {
        includedLigands: ligClass.includedLigands,
        filteredBuffers: ligClass.filteredBuffers,
        nucleicChains: [...(structure.diagnostics?.nucleicAcidChains ?? parsed.diagnostics.nucleicAcidChains)],
      };

      totalResidues = parsed.nResidues;
      designedResidues = scannedCount;
      fixedResidues = Math.max(0, parsed.nResidues - scannedCount);
      estimatedTokens = scannedCount;

      previewStructurePdb = structure.pdbText || structure.cifText || null;
      previewStructureMeta = {
        label: structure.label,
        atomGranularity: structure.atomGranularity ?? 'backbone4',
        bfactorSemantics: structure.bfactorSemantics,
        length: parsed.nResidues,
      };
      const scannedSeq = allRes
        .filter((r) => r.flatIndex >= 0)
        .map((r) => r.aa1)
        .join('');
      if (scannedSeq.length > 0) {
        fastaEntries = [
          {
            id: structure.id,
            header: `>${structure.label} (SSM scan · ${scannedCount} pos × 20 aa = ${scannedCount * 20} ΔΔG)`,
            sequence: scannedSeq,
            selected: true,
          },
        ];
      }

      payload = {
        format: 'scorer-mapping',
        headline: `ThermoMPNN (${precision.toUpperCase()}) · ${scannedCount} pos (${scannedCount * 20} ΔΔG mutations)`,
        compiledSummary: {
          precision,
          chain: chainFilter || 'all',
          bfactorAggregate,
          scannedResidues: scannedCount,
          totalMutations: scannedCount * 20,
          chains: availableChains,
        },
        rawSpecText: JSON.stringify(
          {
            precision,
            chain: chainFilter || null,
            bfactorAggregate,
            scannedResidues: scannedCount,
            totalMutations: scannedCount * 20,
            chains: availableChains,
          },
          null,
          2,
        ),
      };
    }
  } else if (node.adapterId === 'ligandmpnn' || node.adapterId === 'lasermpnn') {
    const isLigand = node.adapterId === 'ligandmpnn';
    const modelName = isLigand ? 'LigandMPNN' : 'LASErMPNN';
    const structure = resolvedInputs.structure as StructureArtifact | undefined;
    const precision = String(node.params.precision ?? 'fp16');
    const checkpoint = isLigand ? String(node.params.checkpoint ?? 'ligandmpnn_v_32_020_25') : undefined;
    const numSequences = Math.max(1, Math.min(8, Math.round(Number(node.params.numSequences ?? 2))));
    const temperature = Number(node.params.temperature ?? 0.1);
    const fixedSpec = String(node.params.fixedPositions ?? '').trim();
    const omitAminoAcids = String(node.params.omitAminoAcids ?? '')
      .trim()
      .toUpperCase()
      .replace(/[^ACDEFGHIKLMNPQRSTVWYX]/g, '');
    const useLigandContext = isLigand ? node.params.useLigandContext !== false : true;
    const unlockInherited = Boolean(node.params.unlockInherited);
    const seed = Number(node.params.seed ?? 42);

    if (!structure) {
      const missingUp = missingUpstreamByPort.get('structure');
      if (missingUp) {
        availability = 'pending-upstream';
        missingUpstreamNodeIds = [missingUp];
      } else {
        availability = 'error';
      }
      payload = {
        format: 'mpnn-masks',
        headline: missingUp
          ? `${modelName} · waiting for upstream structure`
          : `${modelName} · connect a StructureArtifact input`,
        compiledSummary: {
          ...(checkpoint ? { checkpoint } : {}),
          precision,
          numSequences,
          temperature,
          ...(isLigand ? { useLigandContext } : { allAtomRotamers: true }),
          seed,
        },
        rawSpecText: JSON.stringify(node.params, null, 2),
      };
    } else {
      runRuntimeValidation();
      try {
        const rawAllAtom = structure.cifText ?? structure.pdbText;
        const entry = getOrParseCachedStructure(structure.id, rawAllAtom);
        const parsed = entry.parsed;
        const ctx = getCachedRfd4Context(entry, rawAllAtom);
        const ligClass = classifyLigands([
          ...(structure.diagnostics?.ligandNames ?? parsed.diagnostics.ligandNames),
          ...ctx.ligands.map((l) => l.compId),
        ]);
        const nucleicChains = [
          ...new Set([
            ...(structure.diagnostics?.nucleicAcidChains ?? parsed.diagnostics.nucleicAcidChains),
            ...ctx.nucleicChains.map((c) => c.authChain),
          ]),
        ];
        entities = {
          includedLigands: ligClass.includedLigands,
          filteredBuffers: ligClass.filteredBuffers,
          nucleicChains,
        };

        const allChains = [...new Set(parsed.sourceResidues.map((s) => s.authChainId))];
        const defaultChain = allChains[0] ?? 'A';
        const orderedResidues: OrderedStructureResidue[] = parsed.sourceResidues.map((sr) => ({
          chain: sr.authChainId,
          number: sr.resNum,
          insertionCode: sr.insCode ?? '',
        }));
        const { fixedPositions: userFixed, lockedChains: userLockedRaw } =
          parseFixedPositionsAndChains(fixedSpec, defaultChain, orderedResidues);
        const userLockedChainSet = new Set(userLockedRaw.map((c) => c.toUpperCase()));
        const inheritedLockedChainSet = new Set(
          !unlockInherited && Array.isArray(structure.lockedChains)
            ? structure.lockedChains.map((c) => String(c).trim().toUpperCase())
            : [],
        );
        const userFixedKeys = new Set<string>();
        for (const ref of userFixed) {
          if (typeof ref !== 'number') {
            userFixedKeys.add(`${ref.chain.toUpperCase()}:${ref.number}:${ref.insertionCode ?? ''}`);
          }
        }
        const inheritedFixedIndices = new Set<number>(
          !unlockInherited && Array.isArray(structure.fixedResidueIndices)
            ? structure.fixedResidueIndices.map((idx) => Number(idx) - 1)
            : [],
        );

        const chainMap = new Map<string, PreviewChainTrack>();
        const allRes: PreviewResidue[] = [];
        let designedCount = 0;
        let fixedCount = 0;

        for (let i = 0; i < parsed.nResidues; i++) {
          const src = parsed.sourceResidues[i];
          const chUpper = src.authChainId.toUpperCase();
          let track = chainMap.get(src.authChainId);
          if (!track) {
            const roleTag = structure.chainRoles?.[src.authChainId];
            track = {
              chainId: src.authChainId,
              ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
              entityKind: 'protein',
              ...(roleTag ? { roleTag } : {}),
              sequence: '',
              residues: [],
            };
            chainMap.set(src.authChainId, track);
          }

          let gapBefore = 0;
          if (track.residues.length > 0) {
            const prev = track.residues[track.residues.length - 1];
            const delta = src.resNum - prev.authSeqId - 1;
            if (delta > 0) gapBefore = delta;
          }

          const resKey = `${chUpper}:${src.resNum}:${src.insCode}`;
          const isUserFixed = userLockedChainSet.has(chUpper) || userFixedKeys.has(resKey);
          const isInheritedFixed =
            !isUserFixed && (inheritedLockedChainSet.has(chUpper) || inheritedFixedIndices.has(i));
          const isFixed = isUserFixed || isInheritedFixed;
          if (isFixed) fixedCount++;
          else designedCount++;

          const aa1 = parsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
          const pr: PreviewResidue = {
            flatIndex: i,
            authChain: src.authChainId,
            ...(src.labelChainId ? { labelChain: src.labelChainId } : {}),
            authSeqId: src.resNum,
            insCode: src.insCode,
            ...(src.labelSeqId !== undefined ? { labelSeqId: src.labelSeqId } : {}),
            token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
            aa1,
            resName: src.resName,
            roles: isFixed ? ['fixed-sequence'] : ['designable'],
            ...(isInheritedFixed ? { inheritedLock: true } : {}),
            ...(gapBefore > 0 ? { gapBefore } : {}),
            ...(entry.bfactors ? { bfactor: entry.bfactors[i] } : {}),
            backboneAtoms: formatBackboneAtoms(parsed.atomPresence[i], structure.atomGranularity),
          };
          track.residues.push(pr);
          track.sequence += aa1;
          allRes.push(pr);
        }

        chains = [...chainMap.values()];
        highlightLayers = buildHighlightLayersFromResidues(allRes);
        totalResidues = parsed.nResidues;
        designedResidues = designedCount;
        fixedResidues = fixedCount;
        estimatedTokens =
          totalResidues + (useLigandContext ? entities.includedLigands.length : 0);

        const hasNonProt = entities.includedLigands.length > 0 || entities.nucleicChains.length > 0;
        if (useLigandContext && !hasNonProt) {
          pushIssue({
            severity: 'warning',
            source: 'compiler',
            message: `${modelName} has Use Ligand Context enabled, but the input structure contains no non-polymer ligands, ions, or nucleic acids — running protein-only (apo). If an upstream PDB Input is cropped, add ligand codes (e.g. A:BEN, A:CA) or clear the crop to retain holo context.`,
          });
        }
        previewStructurePdb = structure.cifText || structure.pdbText;
        previewStructureMeta = {
          label: structure.label,
          atomGranularity: structure.atomGranularity ?? 'all-atom',
          bfactorSemantics: structure.bfactorSemantics,
          length: parsed.nResidues,
        };
        fastaEntries = [
          {
            id: structure.id,
            header: `>${structure.label} (template · ${designedCount}/${parsed.nResidues} designable)`,
            sequence: parsed.sequence,
            chainLengths: parsed.diagnostics.chainLengths,
            selected: true,
          },
        ];

        const contextNote =
          useLigandContext && hasNonProt
            ? ` · retains ${[
                ...(entities.includedLigands.length > 0
                  ? [`${entities.includedLigands.length} ligand${entities.includedLigands.length === 1 ? '' : 's'}`]
                  : []),
                ...(entities.nucleicChains.length > 0
                  ? [`${entities.nucleicChains.length} NA chain${entities.nucleicChains.length === 1 ? '' : 's'}`]
                  : []),
              ].join(' + ')}`
            : useLigandContext
              ? ' · apo (no ligands in input)'
              : '';

        payload = {
          format: 'mpnn-masks',
          headline: `${modelName} (${precision.toUpperCase()}) · ${designedCount} design / ${fixedCount} fixed${contextNote}`,
          compiledSummary: {
            ...(checkpoint ? { checkpoint } : {}),
            precision,
            numSequences,
            temperature,
            useLigandContext: useLigandContext ? (hasNonProt ? 'Yes (holo)' : 'Yes (apo — 0 ligands in input)') : 'No',
            ...(isLigand ? {} : { allAtomRotamers: true }),
            retainedLigands: useLigandContext ? entities.includedLigands : [],
            retainedNucleicChains: useLigandContext ? entities.nucleicChains : [],
            lockedChains: [...new Set([...userLockedChainSet, ...inheritedLockedChainSet])],
            omitAminoAcids: omitAminoAcids || 'none',
            unlockInherited,
            seed,
          },
          rawSpecText: JSON.stringify(
            {
              adapter: node.adapterId,
              ...(checkpoint ? { checkpoint } : {}),
              precision,
              numSequences,
              temperature,
              useLigandContext,
              retainedLigands: useLigandContext ? entities.includedLigands : [],
              retainedNucleicChains: useLigandContext ? entities.nucleicChains : [],
              fixedPositions: fixedSpec || null,
              omitAminoAcids: omitAminoAcids || null,
              unlockInherited,
              designedCount,
              fixedCount,
              seed,
            },
            null,
            2,
          ),
        };
      } catch (err) {
        availability = 'error';
        pushIssue({
          severity: 'error',
          source: 'compiler',
          message: err instanceof Error ? err.message : String(err),
        });
      }
    }
  } else if (node.adapterId === 'antifold-esmif1') {
    const structure = resolvedInputs.structure as StructureArtifact | undefined;
    const variant = String(node.params.variant ?? 'antifold') === 'esmif1' ? 'esmif1' : 'antifold';
    const variantLabel = variant === 'esmif1' ? 'ESM-IF1' : 'AntiFold';
    const precision = String(node.params.precision ?? 'fp16');
    const numSequences = Math.max(1, Math.min(8, Math.round(Number(node.params.numSequences ?? 2))));
    const temperature = Number(node.params.temperature ?? 0.2);
    const samplingMode =
      String(node.params.samplingMode ?? 'independent') === 'autoregressive'
        ? 'autoregressive'
        : 'independent';
    const targetChainParam = String(node.params.targetChain ?? '').trim();
    const numberingParam = normalizeAntiFoldNumberingParam(node.params.numbering);
    const regions = variant === 'antifold' ? String(node.params.regions ?? 'CDR1 CDR2 CDR3').trim() : '';
    const fixedSpec = String(node.params.fixedPositions ?? '').trim();
    const unlockInherited = Boolean(node.params.unlockInherited);
    const seed = Number(node.params.seed ?? 42);

    if (!structure) {
      const missingUp = missingUpstreamByPort.get('structure');
      if (missingUp) {
        availability = 'pending-upstream';
        missingUpstreamNodeIds = [missingUp];
      } else {
        availability = 'error';
      }
      payload = {
        format: 'mpnn-masks',
        headline: missingUp
          ? `${variantLabel} · waiting for upstream structure`
          : `${variantLabel} · connect a StructureArtifact input`,
        compiledSummary: {
          variant,
          precision,
          numSequences,
          temperature,
          samplingMode,
          ...(targetChainParam ? { targetChain: targetChainParam } : {}),
          ...(regions ? { regions } : {}),
          seed,
        },
        rawSpecText: JSON.stringify(node.params, null, 2),
      };
    } else {
      runRuntimeValidation();
      try {
        const rawText = structure.pdbText || structure.cifText || '';
        const entry = getOrParseCachedStructure(structure.id, rawText);
        const parsed = entry.parsed;

        const orderedResidues: OrderedStructureResidue[] = [];
        const chainInfoMap = new Map<string, NumberedChainInfo>();
        for (let i = 0; i < parsed.nResidues; i++) {
          const sr = parsed.sourceResidues[i];
          const aa1 = parsed.sequence[i] ?? AA3_TO_AA1[sr.resName] ?? 'X';
          orderedResidues.push({
            chain: sr.authChainId,
            number: sr.resNum,
            insertionCode: sr.insCode ?? '',
          });
          let info = chainInfoMap.get(sr.authChainId);
          if (!info) {
            info = { id: sr.authChainId, residues: [] };
            chainInfoMap.set(sr.authChainId, info);
          }
          (info.residues as Array<{ number: number; insertion: string; aa1: string }>).push({
            number: sr.resNum,
            insertion: sr.insCode ?? '',
            aa1,
          });
        }
        const chainInfos = [...chainInfoMap.values()];

        // Compile via single-source-of-truth plan shared 1:1 with antifold.worker.ts (#77)
        const designPlan = compileAntiFoldDesignPlan({
          chainInfos,
          orderedResidues,
          chainTypes: structure.molecularContext?.chainTypes,
          inheritedLockedChains: structure.lockedChains,
          inheritedFixedResidueIndices: structure.fixedResidueIndices,
          variant,
          samplingMode,
          targetChainSpec: targetChainParam,
          numberingParam,
          regionsSpec: regions,
          fixedPositionsSpec: fixedSpec,
          unlockInherited,
          strictTargetChain: false,
        });
        for (const w of designPlan.warnings) {
          pushIssue({ severity: 'warning', source: 'compiler', message: w });
        }

        const {
          resolvedTargetChain,
          resolvedNumbering,
          effectiveRegions,
          remappedChains,
          designedCount,
          fixedCount,
        } = designPlan;

        const chainMap = new Map<string, PreviewChainTrack>();
        const allRes: PreviewResidue[] = [];

        for (let i = 0; i < parsed.nResidues; i++) {
          const src = parsed.sourceResidues[i];
          let track = chainMap.get(src.authChainId);
          if (!track) {
            const roleTag = structure.chainRoles?.[src.authChainId];
            track = {
              chainId: src.authChainId,
              ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
              entityKind: 'protein',
              ...(roleTag ? { roleTag } : {}),
              sequence: '',
              residues: [],
            };
            chainMap.set(src.authChainId, track);
          }

          let gapBefore = 0;
          if (track.residues.length > 0) {
            const prev = track.residues[track.residues.length - 1];
            const delta = src.resNum - prev.authSeqId - 1;
            if (delta > 0) gapBefore = delta;
          }

          const state = designPlan.residueStates[i];
          const roles: ResidueRole[] = state.lockedByTargetChain
            ? ['fixed-backbone', 'fixed-sequence']
            : !state.willDesign
              ? ['fixed-sequence']
              : ['designable'];

          const aa1 = parsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
          const pr: PreviewResidue = {
            flatIndex: i,
            authChain: src.authChainId,
            ...(src.labelChainId ? { labelChain: src.labelChainId } : {}),
            authSeqId: src.resNum,
            insCode: src.insCode,
            ...(src.labelSeqId !== undefined ? { labelSeqId: src.labelSeqId } : {}),
            token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
            aa1,
            resName: src.resName,
            roles,
            ...(state.isInheritedFixed ? { inheritedLock: true } : {}),
            ...(gapBefore > 0 ? { gapBefore } : {}),
            ...(entry.bfactors ? { bfactor: entry.bfactors[i] } : {}),
            backboneAtoms: formatBackboneAtoms(parsed.atomPresence[i], structure.atomGranularity),
          };
          track.residues.push(pr);
          track.sequence += aa1;
          allRes.push(pr);
        }

        chains = [...chainMap.values()];
        highlightLayers = buildHighlightLayersFromResidues(allRes);
        const ligClass = classifyLigands(
          structure.diagnostics?.ligandNames ?? parsed.diagnostics.ligandNames,
        );
        entities = {
          includedLigands: ligClass.includedLigands,
          filteredBuffers: ligClass.filteredBuffers,
          nucleicChains: [
            ...(structure.diagnostics?.nucleicAcidChains ?? parsed.diagnostics.nucleicAcidChains),
          ],
        };

        totalResidues = parsed.nResidues;
        designedResidues = designedCount;
        fixedResidues = fixedCount;
        estimatedTokens = totalResidues;

        previewStructurePdb = structure.pdbText || structure.cifText || null;
        previewStructureMeta = {
          label: structure.label,
          atomGranularity: structure.atomGranularity ?? 'backbone4',
          bfactorSemantics: structure.bfactorSemantics,
          length: parsed.nResidues,
        };
        fastaEntries = [
          {
            id: structure.id,
            header: `>${structure.label} (${variantLabel} · target ${resolvedTargetChain} · ${designedCount} designable)`,
            sequence: parsed.sequence,
            chainLengths: parsed.diagnostics.chainLengths,
            selected: true,
          },
        ];

        payload = {
          format: 'mpnn-masks',
          headline: `${variantLabel} (${precision.toUpperCase()} · ${samplingMode}) · chain ${resolvedTargetChain} · ${designedCount} design / ${fixedCount} fixed`,
          compiledSummary: {
            variant,
            precision,
            samplingMode,
            targetChain: resolvedTargetChain,
            ...(variant === 'antifold' ? { numbering: resolvedNumbering } : {}),
            ...(regions ? { regions } : {}),
            ...(remappedChains.length > 0 ? { effectiveRegions } : {}),
            numSequences,
            temperature,
            unlockInherited,
            seed,
          },
          rawSpecText: JSON.stringify(
            {
              variant,
              precision,
              samplingMode,
              targetChain: resolvedTargetChain,
              ...(variant === 'antifold'
                ? {
                    numbering: resolvedNumbering,
                    regions: regions || null,
                    ...(remappedChains.length > 0 ? { effectiveRegions, remappedChains } : {}),
                  }
                : {}),
              numSequences,
              temperature,
              fixedPositions: fixedSpec || null,
              unlockInherited,
              designedCount,
              fixedCount,
              seed,
            },
            null,
            2,
          ),
        };
      } catch (err) {
        availability = 'error';
        pushIssue({
          severity: 'error',
          source: 'compiler',
          message: err instanceof Error ? err.message : String(err),
        });
      }
    }
  } else if (node.adapterId === 'genie2') {
    const length = Math.max(20, Math.min(256, Math.round(Number(node.params.length ?? 80))));
    const steps = Math.max(10, Math.min(1000, Math.round(Number(node.params.steps ?? 1000))));
    const scale = Math.max(0.1, Math.min(1.0, Number(node.params.scale ?? 0.6)));
    const precision = String(node.params.precision ?? 'fp16').trim().toLowerCase() === 'int8' ? 'int8' : 'fp16';
    const motifRange = String(node.params.motifRange ?? '').trim();
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(node.params.numDesigns ?? 1))));
    const seed = Math.round(Number(node.params.seed ?? 42));
    const template = resolvedInputs.template as StructureArtifact | undefined;
    const missingTplUp = missingUpstreamByPort.get('template');

    if (motifRange && !template) {
      if (missingTplUp) {
        availability = 'pending-upstream';
        missingUpstreamNodeIds = [missingTplUp];
      } else {
        availability = 'error';
      }
      payload = {
        format: 'genie3-spec',
        headline: missingTplUp
          ? `Genie 2 (Motif ${motifRange}) · waiting for upstream template`
          : `Genie 2 (Motif ${motifRange}) · connect a template structure`,
        compiledSummary: {
          precision,
          length,
          steps,
          scale,
          motifRange,
          numDesigns,
          atomGranularity: 'ca-only',
          seed,
        },
        rawSpecText: JSON.stringify(
          { precision, length, steps, scale, motifRange, numDesigns, seed },
          null,
          2,
        ),
      };
    } else if (template) {
      runRuntimeValidation();
      const rawText = template.pdbText || template.cifText || '';
      const tplEntry = rawText ? getOrParseCachedStructure(template.id, rawText) : undefined;
      const parsed = tplEntry?.parsed;
      const rawLigands = [
        ...new Set(
          [
            ...(template.diagnostics?.ligandNames ?? []),
            ...(parsed?.diagnostics.ligandNames ?? []),
          ]
            .map((l) => l.trim().toUpperCase())
            .filter(Boolean),
        ),
      ];
      const rawNucleic = [
        ...new Set(
          [
            ...(template.diagnostics?.nucleicAcidChains ?? []),
            ...(parsed?.diagnostics.nucleicAcidChains ?? []),
          ]
            .map((c) => c.trim())
            .filter(Boolean),
        ),
      ];
      const bench = template.benchmarkId ? getBenchmarkById(template.benchmarkId) : undefined;
      const inferredPdbId =
        bench?.pdbId ??
        (/\b([0-9][A-Za-z0-9]{3})\b/.exec(template.label ?? '')?.[1]?.toUpperCase() || undefined);
      const policy = evaluateProteinOnlyStructurePolicy({
        modelName: nodeLabel,
        benchmark: bench,
        pdbId: inferredPdbId,
        proteinResidueCount: parsed?.nResidues ?? template.length,
        ligandNames: rawLigands,
        nucleicAcidChains: rawNucleic,
        hasLigands: template.diagnostics?.hasLigands ?? rawLigands.length > 0,
        hasNucleicAcids: template.diagnostics?.hasNucleicAcids ?? rawNucleic.length > 0,
      });
      if (policy.action === 'hard-block' && policy.message) {
        availability = 'error';
        pushIssue({ severity: 'error', source: 'compiler', message: policy.message });
      } else if (policy.action === 'warn' && policy.message) {
        pushIssue({ severity: 'warning', source: 'compiler', message: policy.message });
      }

      // Genie 2 is strictly protein-only Cα diffusion: ALL crystal HETATMs (M3L, CA, ligands, buffers)
      // and nucleic chains are stripped/ignored — NEVER reported as retained ligands.
      entities = {
        includedLigands: [],
        filteredBuffers: rawLigands,
        nucleicChains: [],
      };

      previewStructurePdb = template.pdbText || template.cifText || null;
      previewStructureMeta = {
        label: template.label,
        atomGranularity: 'ca-only',
        bfactorSemantics: template.bfactorSemantics,
        length: parsed?.nResidues ?? template.length,
      };

      if (motifRange && parsed) {
        const rangeMatch = motifRange.match(/^(?:([A-Za-z]):?)?(-?\d+)(?:-(-?\d+))?$/);
        if (!rangeMatch) {
          availability = 'error';
          pushIssue({
            severity: 'error',
            source: 'compiler',
            message: `Invalid Genie 2 motifRange "${motifRange}" (expected e.g. A16-35 or 16-35).`,
          });
        } else {
          const chainFilter = rangeMatch[1]?.toUpperCase();
          const startRes = Number(rangeMatch[2]);
          const endRes = rangeMatch[3] ? Number(rangeMatch[3]) : startRes;
          const lo = Math.min(startRes, endRes);
          const hi = Math.max(startRes, endRes);
          const picked: number[] = [];
          for (let i = 0; i < parsed.nResidues; i++) {
            if (chainFilter && parsed.chainIds[i]?.toUpperCase() !== chainFilter) continue;
            const r = parsed.resNums[i];
            if (r >= lo && r <= hi) picked.push(i);
          }
          if (picked.length < 3) {
            availability = 'error';
            pushIssue({
              severity: 'error',
              source: 'compiler',
              message: `Genie 2 motifRange "${motifRange}" matched only ${picked.length} residue(s) (minimum 3 required).`,
            });
          } else if (picked.length >= length) {
            availability = 'error';
            pushIssue({
              severity: 'error',
              source: 'compiler',
              message: `Genie 2 motif length (${picked.length} aa) must be smaller than target length (${length} aa).`,
            });
          } else {
            const startOffset = Math.max(0, Math.floor((length - picked.length) / 2));
            const pickedSet = new Set(picked);
            const scaffoldResidues: PreviewResidue[] = [];
            let seq = '';
            for (let i = 0; i < length; i++) {
              const motifOffset = i - startOffset;
              if (motifOffset >= 0 && motifOffset < picked.length) {
                const tplIdx = picked[motifOffset];
                const src = parsed.sourceResidues[tplIdx];
                const aa1 = parsed.sequence[tplIdx] ?? AA3_TO_AA1[src.resName] ?? 'X';
                seq += aa1;
                scaffoldResidues.push({
                  flatIndex: i,
                  authChain: src.authChainId,
                  labelChain: 'A',
                  authSeqId: src.resNum,
                  insCode: src.insCode,
                  labelSeqId: i + 1,
                  token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
                  aa1,
                  resName: src.resName,
                  roles: ['motif', 'fixed-backbone', 'fixed-sequence'],
                  ...(tplEntry?.bfactors ? { bfactor: tplEntry.bfactors[tplIdx] } : {}),
                  backboneAtoms: 'CA-only',
                });
              } else {
                seq += '-';
                scaffoldResidues.push({
                  flatIndex: i,
                  authChain: 'A',
                  labelChain: 'A',
                  authSeqId: i + 1,
                  insCode: '',
                  labelSeqId: i + 1,
                  token: formatResidueToken('A', i + 1),
                  aa1: '-',
                  resName: 'UNK',
                  roles: ['designable'],
                });
              }
            }
            chains = [
              {
                chainId: 'A',
                labelChainId: 'A',
                entityKind: 'protein',
                roleTag: 'scaffold',
                sequence: seq,
                residues: scaffoldResidues,
              },
            ];
            const croppedExtras: Array<{ chain: string; seqId: number; insCode?: string }> = [];
            for (let i = 0; i < parsed.nResidues; i++) {
              if (!pickedSet.has(i)) {
                const src = parsed.sourceResidues[i];
                croppedExtras.push({
                  chain: src.authChainId,
                  seqId: src.resNum,
                  ...(src.insCode ? { insCode: src.insCode } : {}),
                });
              }
            }
            highlightLayers = buildHighlightLayersFromResidues(scaffoldResidues, {
              'cropped-out': croppedExtras,
            });
            totalResidues = length;
            fixedResidues = picked.length;
            designedResidues = length - picked.length;
            estimatedTokens = length;
            fastaEntries = [
              {
                id: 'genie2-motif',
                header: `>Genie2_motif (${length} aa · ${picked.length} motif / ${length - picked.length} design)`,
                sequence: seq,
                chainLengths: [length],
                selected: true,
              },
            ];
          }
        }
      } else {
        // Template wired without motifRange: Genie 2 generates unconditionally at `length`
        const track = buildSyntheticDesignedChain('A', length, 0, 'scaffold', 'A');
        chains = [track];
        totalResidues = length;
        designedResidues = length;
        fixedResidues = 0;
        estimatedTokens = length;
        fastaEntries = [
          {
            id: 'genie2-uncond',
            header: `>Genie2_unconditional (${length} aa)`,
            sequence: track.sequence,
            chainLengths: [length],
            selected: true,
          },
        ];
      }

      payload = {
        format: 'genie3-spec',
        headline: motifRange
          ? `Genie 2 (Motif ${motifRange}) · ${designedResidues} design / ${fixedResidues} fixed · L=${length} aa`
          : `Genie 2 (Unconditional) · L=${length} aa (${steps} steps)`,
        compiledSummary: {
          mode: motifRange ? 'motif' : 'unconditional',
          precision,
          length,
          designedResidues,
          fixedResidues,
          steps,
          scale,
          ...(motifRange ? { motifRange } : {}),
          numDesigns,
          atomGranularity: 'ca-only',
          ...(rawLigands.length > 0 ? { strippedHetatms: rawLigands } : {}),
          ...(rawNucleic.length > 0 ? { strippedNucleicChains: rawNucleic } : {}),
          seed,
        },
        rawSpecText: JSON.stringify(
          {
            adapter: 'genie2',
            mode: motifRange ? 'motif' : 'unconditional',
            precision,
            length,
            designedResidues,
            fixedResidues,
            steps,
            scale,
            motifRange: motifRange || null,
            strippedHetatms: rawLigands,
            strippedNucleicChains: rawNucleic,
            numDesigns,
            seed,
          },
          null,
          2,
        ),
      };
    } else {
      runRuntimeValidation();
      const track = buildSyntheticDesignedChain('A', length, 0, 'scaffold', 'A');
      chains = [track];
      totalResidues = length;
      designedResidues = length;
      fixedResidues = 0;
      estimatedTokens = length;
      fastaEntries = [
        {
          id: 'genie2-uncond',
          header: `>Genie2_unconditional (${length} aa)`,
          sequence: track.sequence,
          chainLengths: [length],
          selected: true,
        },
      ];
      payload = {
        format: 'genie3-spec',
        headline: `Genie 2 (Unconditional) · L=${length} aa (${steps} steps)`,
        compiledSummary: {
          mode: 'unconditional',
          precision,
          length,
          steps,
          scale,
          numDesigns,
          atomGranularity: 'ca-only',
          seed,
        },
        rawSpecText: JSON.stringify(
          { adapter: 'genie2', mode: 'unconditional', precision, length, steps, scale, numDesigns, seed },
          null,
          2,
        ),
      };
    }
  } else if (node.adapterId === 'protpardelle') {
    const PROTPARDELLE_ALLATOM_CHECKPOINTS = new Set([
      'cc91_epoch383',
      'cc91_tip_epoch480',
      'cc94_epoch3100',
    ]);
    const rawTask = String(node.params.taskMode ?? 'unconditional').trim().toLowerCase();
    const taskMode: 'unconditional' | 'motif' | 'partial' =
      rawTask === 'motif' || rawTask === 'partial' ? rawTask : 'unconditional';
    const checkpoint = String(node.params.checkpoint ?? 'cc91_epoch383').trim();
    const precision = String(node.params.precision ?? 'int8').trim().toLowerCase() === 'fp16' ? 'fp16' : 'int8';
    const length = Math.max(20, Math.min(256, Math.round(Number(node.params.length ?? 80))));
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(node.params.numDesigns ?? 1))));
    const numSteps = Math.max(20, Math.min(500, Math.round(Number(node.params.numSteps ?? 100))));
    const isAllAtomCkpt = PROTPARDELLE_ALLATOM_CHECKPOINTS.has(checkpoint);
    const sidechainMode = Boolean(node.params.sidechainMode ?? true) && isAllAtomCkpt;
    const motifSelection = String(node.params.motifSelection ?? '').trim();
    const motifGuidance = String(node.params.motifGuidance ?? 'replacement').trim();
    const rewindSteps = Math.max(5, Math.min(numSteps - 1, Math.round(Number(node.params.rewindSteps ?? 40))));
    const stepScale = Number(node.params.stepScale ?? 1.2);
    const temperature = Number(node.params.temperature ?? 0.1);
    const seed = Math.max(0, Math.round(Number(node.params.seed ?? 42)));
    const miniMpnnNote = sidechainMode
      ? `MiniMPNN sidechain/sequence co-design enabled (~${precision === 'fp16' ? '28' : '14'} MB weights loaded on run)`
      : undefined;

    const structure = resolvedInputs.structure as StructureArtifact | undefined;
    const missingStructUp = missingUpstreamByPort.get('structure');

    if ((taskMode === 'motif' || taskMode === 'partial') && !structure) {
      if (missingStructUp) {
        availability = 'pending-upstream';
        missingUpstreamNodeIds = [missingStructUp];
      } else {
        availability = 'error';
      }
      payload = {
        format: 'raw-input',
        headline: missingStructUp
          ? `ProtPardelle-1c (${taskMode}) · waiting for upstream structure`
          : `ProtPardelle-1c (${taskMode}) · connect a StructureArtifact input`,
        compiledSummary: {
          taskMode,
          checkpoint,
          precision,
          ...(taskMode === 'partial' ? { rewindSteps } : { length }),
          numSteps,
          sidechainMode,
          ...(miniMpnnNote ? { miniMpnnNote } : {}),
          numDesigns,
          seed,
        },
        rawSpecText: JSON.stringify(
          { taskMode, checkpoint, precision, length, numSteps, rewindSteps, sidechainMode, miniMpnnNote, numDesigns, seed },
          null,
          2,
        ),
      };
    } else if (taskMode === 'partial' && structure) {
      runRuntimeValidation();
      const rawText = structure.pdbText || structure.cifText || '';
      const entry = rawText ? getOrParseCachedStructure(structure.id, rawText) : undefined;
      const parsed = entry?.parsed;
      const rawLigands = [
        ...new Set(
          [
            ...(structure.diagnostics?.ligandNames ?? []),
            ...(parsed?.diagnostics.ligandNames ?? []),
          ]
            .map((l) => l.trim().toUpperCase())
            .filter(Boolean),
        ),
      ];
      entities = {
        includedLigands: [],
        filteredBuffers: rawLigands,
        nucleicChains: [],
      };
      if (parsed) {
        const chainMap = new Map<string, PreviewChainTrack>();
        const allRes: PreviewResidue[] = [];
        for (let i = 0; i < parsed.nResidues; i++) {
          const src = parsed.sourceResidues[i];
          let track = chainMap.get(src.authChainId);
          if (!track) {
            track = {
              chainId: src.authChainId,
              ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
              entityKind: 'protein',
              roleTag: 'scaffold',
              sequence: '',
              residues: [],
            };
            chainMap.set(src.authChainId, track);
          }
          const aa1 = parsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X';
          const pr: PreviewResidue = {
            flatIndex: i,
            authChain: src.authChainId,
            ...(src.labelChainId ? { labelChain: src.labelChainId } : {}),
            authSeqId: src.resNum,
            insCode: src.insCode,
            token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
            aa1,
            resName: src.resName,
            roles: ['designable'],
            ...(entry?.bfactors ? { bfactor: entry.bfactors[i] } : {}),
            backboneAtoms: formatBackboneAtoms(parsed.atomPresence[i], structure.atomGranularity),
          };
          track.residues.push(pr);
          track.sequence += aa1;
          allRes.push(pr);
        }
        chains = [...chainMap.values()];
        highlightLayers = buildHighlightLayersFromResidues(allRes);
        totalResidues = parsed.nResidues;
        designedResidues = parsed.nResidues;
        fixedResidues = 0;
        estimatedTokens = parsed.nResidues;
        previewStructurePdb = structure.pdbText || structure.cifText || null;
        previewStructureMeta = {
          label: structure.label,
          atomGranularity: structure.atomGranularity ?? 'all-atom',
          bfactorSemantics: structure.bfactorSemantics,
          length: parsed.nResidues,
        };
        fastaEntries = [
          {
            id: structure.id,
            header: `>${structure.label} (partial diffusion · rewind ${rewindSteps}/${numSteps} · ${parsed.nResidues} aa refined)`,
            sequence: parsed.sequence,
            chainLengths: parsed.diagnostics.chainLengths,
            selected: true,
          },
        ];
      } else {
        totalResidues = structure.length;
        designedResidues = structure.length;
        fixedResidues = 0;
        estimatedTokens = structure.length;
      }
      payload = {
        format: 'raw-input',
        headline: `ProtPardelle-1c (Partial Diffusion · rewind ${rewindSteps}/${numSteps}) · ${designedResidues} aa refined${sidechainMode ? ' · + MiniMPNN' : ''}`,
        compiledSummary: {
          taskMode: 'partial',
          checkpoint,
          precision,
          totalResidues,
          designedResidues,
          fixedResidues: 0,
          rewindSteps,
          numSteps,
          stepScale,
          temperature,
          sidechainMode,
          ...(miniMpnnNote ? { miniMpnnNote } : {}),
          ...(rawLigands.length > 0 ? { strippedHetatms: rawLigands } : {}),
          numDesigns,
          seed,
        },
        rawSpecText: JSON.stringify(
          {
            adapter: 'protpardelle',
            taskMode: 'partial',
            checkpoint,
            precision,
            totalResidues,
            designedResidues,
            fixedResidues: 0,
            rewindSteps,
            numSteps,
            stepScale,
            temperature,
            sidechainMode,
            ...(miniMpnnNote ? { miniMpnnNote } : {}),
            numDesigns,
            seed,
          },
          null,
          2,
        ),
      };
    } else if (taskMode === 'motif' && structure) {
      runRuntimeValidation();
      const rawText = structure.pdbText || structure.cifText || '';
      const entry = rawText ? getOrParseCachedStructure(structure.id, rawText) : undefined;
      const parsed = entry?.parsed;
      const rawLigands = [
        ...new Set(
          [
            ...(structure.diagnostics?.ligandNames ?? []),
            ...(parsed?.diagnostics.ligandNames ?? []),
          ]
            .map((l) => l.trim().toUpperCase())
            .filter(Boolean),
        ),
      ];
      entities = {
        includedLigands: [],
        filteredBuffers: rawLigands,
        nucleicChains: [],
      };
      if (parsed) {
        let sourceIndices: number[] = [];
        if (!motifSelection) {
          sourceIndices = Array.from({ length: parsed.nResidues }, (_, i) => i);
        } else {
          const spans = parseResidueRangeTokens(motifSelection);
          const wholeChains = new Set(
            motifSelection
              .split(/[,\s]+/)
              .map((t) => t.trim())
              .filter((t) => /^[A-Za-z]$/.test(t))
              .map((t) => t.toUpperCase()),
          );
          for (let i = 0; i < parsed.nResidues; i++) {
            const src = parsed.sourceResidues[i];
            const ch = src.authChainId.toUpperCase();
            if (
              wholeChains.has(ch) ||
              spans.some(
                (s) =>
                  s.chain.toUpperCase() === ch &&
                  src.resNum >= Math.min(s.start, s.end) &&
                  src.resNum <= Math.max(s.start, s.end),
              )
            ) {
              sourceIndices.push(i);
            }
          }
        }
        if (sourceIndices.length === 0) {
          availability = 'error';
          pushIssue({
            severity: 'error',
            source: 'compiler',
            message: `Motif selection "${motifSelection}" matched 0 residues in "${structure.label}".`,
          });
        } else {
          const motifLen = sourceIndices.length;
          const sampleLen = Math.max(length, motifLen + 10);
          const padLeft = Math.floor((sampleLen - motifLen) / 2);
          const selectedSet = new Set(sourceIndices);
          const scaffoldResidues: PreviewResidue[] = [];
          let seq = '';
          for (let i = 0; i < sampleLen; i++) {
            const motifOffset = i - padLeft;
            if (motifOffset >= 0 && motifOffset < motifLen) {
              const tplIdx = sourceIndices[motifOffset];
              const src = parsed.sourceResidues[tplIdx];
              const aa1 = parsed.sequence[tplIdx] ?? AA3_TO_AA1[src.resName] ?? 'X';
              seq += aa1;
              scaffoldResidues.push({
                flatIndex: i,
                authChain: src.authChainId,
                labelChain: 'A',
                authSeqId: src.resNum,
                insCode: src.insCode,
                labelSeqId: i + 1,
                token: formatResidueToken(src.authChainId, src.resNum, src.insCode),
                aa1,
                resName: src.resName,
                roles: ['motif', 'fixed-backbone', 'fixed-sequence'],
                ...(entry?.bfactors ? { bfactor: entry.bfactors[tplIdx] } : {}),
                backboneAtoms: formatBackboneAtoms(parsed.atomPresence[tplIdx], structure.atomGranularity),
              });
            } else {
              seq += '-';
              scaffoldResidues.push({
                flatIndex: i,
                authChain: 'A',
                labelChain: 'A',
                authSeqId: i + 1,
                insCode: '',
                labelSeqId: i + 1,
                token: formatResidueToken('A', i + 1),
                aa1: '-',
                resName: 'UNK',
                roles: ['designable'],
              });
            }
          }
          chains = [
            {
              chainId: 'A',
              labelChainId: 'A',
              entityKind: 'protein',
              roleTag: 'scaffold',
              sequence: seq,
              residues: scaffoldResidues,
            },
          ];
          const croppedExtras: Array<{ chain: string; seqId: number; insCode?: string }> = [];
          for (let i = 0; i < parsed.nResidues; i++) {
            if (!selectedSet.has(i)) {
              const src = parsed.sourceResidues[i];
              croppedExtras.push({
                chain: src.authChainId,
                seqId: src.resNum,
                ...(src.insCode ? { insCode: src.insCode } : {}),
              });
            }
          }
          highlightLayers = buildHighlightLayersFromResidues(scaffoldResidues, {
            'cropped-out': croppedExtras,
          });
          totalResidues = sampleLen;
          fixedResidues = motifLen;
          designedResidues = sampleLen - motifLen;
          estimatedTokens = sampleLen;
          previewStructurePdb = structure.pdbText || structure.cifText || null;
          previewStructureMeta = {
            label: structure.label,
            atomGranularity: structure.atomGranularity ?? 'all-atom',
            bfactorSemantics: structure.bfactorSemantics,
            length: parsed.nResidues,
          };
          fastaEntries = [
            {
              id: 'protpardelle-motif',
              header: `>ProtPardelle_motif (${sampleLen} aa · ${motifLen} motif / ${sampleLen - motifLen} design)`,
              sequence: seq,
              chainLengths: [sampleLen],
              selected: true,
            },
          ];
        }
      }
      payload = {
        format: 'raw-input',
        headline: `ProtPardelle-1c (Motif) · ${designedResidues} design / ${fixedResidues} motif · L=${totalResidues} aa${sidechainMode ? ' · + MiniMPNN' : ''}`,
        compiledSummary: {
          taskMode: 'motif',
          checkpoint,
          precision,
          length: totalResidues || length,
          designedResidues,
          fixedResidues,
          motifSelection: motifSelection || '(all cropped residues)',
          motifGuidance,
          numSteps,
          sidechainMode,
          ...(miniMpnnNote ? { miniMpnnNote } : {}),
          ...(rawLigands.length > 0 ? { strippedHetatms: rawLigands } : {}),
          numDesigns,
          seed,
        },
        rawSpecText: JSON.stringify(
          {
            adapter: 'protpardelle',
            taskMode: 'motif',
            checkpoint,
            precision,
            length: totalResidues || length,
            designedResidues,
            fixedResidues,
            motifSelection: motifSelection || null,
            motifGuidance,
            numSteps,
            sidechainMode,
            ...(miniMpnnNote ? { miniMpnnNote } : {}),
            numDesigns,
            seed,
          },
          null,
          2,
        ),
      };
    } else {
      runRuntimeValidation();
      const track = buildSyntheticDesignedChain('A', length, 0, 'scaffold', 'A');
      chains = [track];
      totalResidues = length;
      designedResidues = length;
      fixedResidues = 0;
      estimatedTokens = length;
      fastaEntries = [
        {
          id: 'protpardelle-uncond',
          header: `>ProtPardelle-1c (${length} aa · ${checkpoint})`,
          sequence: track.sequence,
          chainLengths: [length],
          selected: true,
        },
      ];
      payload = {
        format: 'raw-input',
        headline: `ProtPardelle-1c (Unconditional) · L=${length} aa${sidechainMode ? ' · + MiniMPNN' : ''}`,
        compiledSummary: {
          taskMode: 'unconditional',
          checkpoint,
          precision,
          length,
          numSteps,
          sidechainMode,
          ...(miniMpnnNote ? { miniMpnnNote } : {}),
          numDesigns,
          seed,
        },
        rawSpecText: JSON.stringify(
          {
            adapter: 'protpardelle',
            taskMode: 'unconditional',
            checkpoint,
            precision,
            length,
            numSteps,
            sidechainMode,
            ...(miniMpnnNote ? { miniMpnnNote } : {}),
            numDesigns,
            seed,
          },
          null,
          2,
        ),
      };
    }
  } else if (node.adapterId === 'evodiff') {
    const taskMode = String(node.params.taskMode ?? 'unconditional').trim().toLowerCase() === 'infill'
      ? 'infill'
      : 'unconditional';
    const precision = String(node.params.precision ?? 'int8').trim().toLowerCase() === 'fp16' ? 'fp16' : 'int8';
    const length = Math.max(12, Math.min(512, Math.round(Number(node.params.length ?? 64))));
    const numSequences = Math.max(1, Math.min(8, Math.round(Number(node.params.numSequences ?? 2))));
    const rawFixedSeq = sanitizeEvoDiffFixedSequence(node.params.fixedSequence);
    const rawMaskPositions = String(node.params.maskPositions ?? '').trim();
    const seed = Math.max(0, Math.round(Number(node.params.seed ?? 42)));
    const seqArt = resolvedInputs.sequences as SequenceSet | undefined;
    const missingSeqUp = missingUpstreamByPort.get('sequences');

    if (taskMode === 'infill') {
      let templateSeq = rawFixedSeq;
      let templateSource = 'inline fixedSequence';
      const usedUpstream = !templateSeq && Boolean(seqArt?.candidates?.length);
      if (usedUpstream && seqArt?.candidates?.length) {
        const cand0 = seqArt.candidates[0];
        templateSeq = sanitizeEvoDiffFixedSequence(cand0.sequence).replace(/[:/,-]+/g, '');
        templateSource = cand0.id || 'upstream sequence';
      }

      if (!templateSeq) {
        if (missingSeqUp) {
          availability = 'pending-upstream';
          missingUpstreamNodeIds = [missingSeqUp];
        } else {
          availability = 'error';
          pushIssue({
            severity: 'error',
            source: 'compiler',
            message:
              'EvoDiff in "infill" mode requires either a connected SequenceSet or a Fixed Sequence template (use # for masked positions).',
          });
        }
        payload = {
          format: 'mpnn-masks',
          headline: missingSeqUp
            ? 'EvoDiff-38M (Infill) · waiting for upstream sequence'
            : 'EvoDiff-38M (Infill) · provide fixedSequence or connect SequenceSet',
          compiledSummary: {
            taskMode: 'infill',
            precision,
            numSequences,
            ...(rawMaskPositions ? { maskPositions: rawMaskPositions } : {}),
            seed,
          },
          rawSpecText: JSON.stringify(
            { adapter: 'evodiff', taskMode: 'infill', precision, numSequences, maskPositions: rawMaskPositions || null, seed },
            null,
            2,
          ),
        };
      } else {
        runRuntimeValidation();
        const maskSet = new Set<number>();
        for (let i = 0; i < templateSeq.length; i++) {
          if (templateSeq[i] === '#') maskSet.add(i);
        }
        if (rawMaskPositions) {
          for (const tok of rawMaskPositions.split(/[,\s]+/).filter(Boolean)) {
            const m = /^(\d+)(?:-(\d+))?$/.exec(tok);
            if (!m) {
              availability = 'error';
              pushIssue({
                severity: 'error',
                source: 'compiler',
                message: `Invalid mask position "${tok}". Use 1-based numbers or ranges like "1-15, 45-60".`,
              });
              continue;
            }
            const lo = Number(m[1]);
            const hi = m[2] ? Number(m[2]) : lo;
            if (lo < 1 || hi < lo || hi > templateSeq.length) {
              availability = 'error';
              pushIssue({
                severity: 'error',
                source: 'compiler',
                message: `Mask position "${tok}" is out of range for a ${templateSeq.length}-aa sequence.`,
              });
              continue;
            }
            for (let pos = lo; pos <= hi; pos++) {
              maskSet.add(pos - 1);
            }
          }
        }
        if (maskSet.size === 0 && availability === 'ready') {
          availability = 'error';
          pushIssue({
            severity: 'error',
            source: 'compiler',
            message:
              'EvoDiff "infill" mode requires "#" mask characters in the template or 1-based Mask Positions (e.g. "1-12, 40-50").',
          });
        }

        const residues: PreviewResidue[] = [];
        let trackSeq = '';
        for (let i = 0; i < templateSeq.length; i++) {
          const isMasked = maskSet.has(i);
          const ch = templateSeq[i];
          const aa1 = ch === '#' ? '-' : ch;
          trackSeq += aa1;
          residues.push({
            flatIndex: i,
            authChain: 'A',
            authSeqId: i + 1,
            insCode: '',
            token: formatResidueToken('A', i + 1),
            aa1,
            resName: aa1 === '-' ? 'UNK' : (AA1_TO_AA3[aa1] ?? 'UNK'),
            roles: isMasked ? ['designable'] : ['fixed-sequence'],
          });
        }

        chains = [
          {
            chainId: 'A',
            entityKind: 'protein',
            roleTag: 'scaffold',
            sequence: trackSeq,
            residues,
          },
        ];
        highlightLayers = buildHighlightLayersFromResidues(residues);
        totalResidues = templateSeq.length;
        designedResidues = maskSet.size;
        fixedResidues = templateSeq.length - maskSet.size;
        estimatedTokens = templateSeq.length;

        if (usedUpstream && seqArt?.candidates?.length) {
          fastaEntries = seqArt.candidates.map((cand, idx) => ({
            id: cand.id,
            header: `>${cand.id} (${idx === 0 ? `${designedResidues} masked / ${fixedResidues} fixed` : 'ignored'})`,
            sequence: idx === 0 ? templateSeq : cand.sequence,
            ...(cand.chainLengths ? { chainLengths: cand.chainLengths } : {}),
            selected: idx === 0,
            ...(idx > 0 ? { scoreLabel: 'ignored (candidate 0 only)' } : {}),
          }));
        } else {
          fastaEntries = [
            {
              id: 'evodiff-infill',
              header: `>EvoDiff_infill (${templateSeq.length} aa · ${designedResidues} masked / ${fixedResidues} fixed)`,
              sequence: templateSeq,
              chainLengths: [templateSeq.length],
              selected: true,
            },
          ];
        }

        payload = {
          format: 'mpnn-masks',
          headline: `EvoDiff-38M (Infill) · ${designedResidues} design / ${fixedResidues} fixed · L=${totalResidues} aa`,
          compiledSummary: {
            taskMode: 'infill',
            precision,
            templateSource,
            totalResidues,
            designedResidues,
            fixedResidues,
            ...(rawMaskPositions ? { maskPositions: rawMaskPositions } : {}),
            numSequences,
            seed,
          },
          rawSpecText: JSON.stringify(
            {
              adapter: 'evodiff',
              taskMode: 'infill',
              precision,
              templateSource,
              totalResidues,
              designedResidues,
              fixedResidues,
              maskPositions: rawMaskPositions || null,
              numSequences,
              seed,
            },
            null,
            2,
          ),
        };
      }
    } else {
      runRuntimeValidation();
      const track = buildSyntheticDesignedChain('A', length, 0, 'scaffold', 'A');
      chains = [track];
      totalResidues = length;
      designedResidues = length;
      fixedResidues = 0;
      estimatedTokens = length;
      fastaEntries = [
        {
          id: 'evodiff-uncond',
          header: `>EvoDiff-38M unconditional (${length} aa × ${numSequences} seq${numSequences === 1 ? '' : 's'})`,
          sequence: track.sequence,
          chainLengths: [length],
          selected: true,
        },
      ];
      payload = {
        format: 'raw-input',
        headline: `EvoDiff-38M (Unconditional) · L=${length} aa × ${numSequences} seq${numSequences === 1 ? '' : 's'}`,
        compiledSummary: {
          taskMode: 'unconditional',
          precision,
          length,
          numSequences,
          seed,
        },
        rawSpecText: JSON.stringify(
          { adapter: 'evodiff', taskMode: 'unconditional', precision, length, numSequences, seed },
          null,
          2,
        ),
      };
    }
  } else if (node.adapterId === 'foldflow2') {
    const length = Math.max(20, Math.min(256, Math.round(Number(node.params.length ?? 80))));
    const steps = Math.max(5, Math.min(100, Math.round(Number(node.params.steps ?? 50))));
    const rotationScaling = Math.max(1, Math.min(20, Number(node.params.rotationScaling ?? 10)));
    const selfConditioning = node.params.selfConditioning !== false;
    const precision = String(node.params.precision ?? 'int8').trim().toLowerCase() === 'fp16' ? 'fp16' : 'int8';
    const numDesigns = Math.max(1, Math.min(6, Math.round(Number(node.params.numDesigns ?? 1))));
    const seed = Math.round(Number(node.params.seed ?? 42));
    const seqArt = resolvedInputs.sequences as SequenceSet | undefined;
    const missingSeqUp = missingUpstreamByPort.get('sequences');

    if (!seqArt && missingSeqUp) {
      availability = 'pending-upstream';
      missingUpstreamNodeIds = [missingSeqUp];
      payload = {
        format: 'raw-input',
        headline: 'FoldFlow 2 (Sequence-Conditioned) · waiting for upstream sequence',
        compiledSummary: {
          mode: 'sequence-conditioned',
          precision,
          steps,
          rotationScaling,
          selfConditioning,
          numDesigns,
          seed,
        },
        rawSpecText: JSON.stringify(
          { adapter: 'foldflow2', mode: 'sequence-conditioned', precision, steps, rotationScaling, selfConditioning, numDesigns, seed },
          null,
          2,
        ),
      };
    } else if (seqArt && seqArt.candidates.length > 0) {
      runRuntimeValidation();
      const cand0 = seqArt.candidates[0];
      const rawSeq = cand0.sequence.trim();
      const rawHasColonOrSlash = rawSeq.includes(':') || rawSeq.includes('/');
      const hasMultiChainLengths = Array.isArray(cand0.chainLengths) && cand0.chainLengths.length > 1;
      const strippedChainBreaks = rawHasColonOrSlash || hasMultiChainLengths;
      const condSeq = rawSeq.toUpperCase().replace(/[^A-Z]/g, '');

      const residues: PreviewResidue[] = [];
      for (let i = 0; i < condSeq.length; i++) {
        const aa1 = condSeq[i];
        residues.push({
          flatIndex: i,
          authChain: 'A',
          authSeqId: i + 1,
          insCode: '',
          token: formatResidueToken('A', i + 1),
          aa1,
          resName: AA1_TO_AA3[aa1] ?? 'UNK',
          roles: ['designable'],
        });
      }

      chains = [
        {
          chainId: 'A',
          entityKind: 'protein',
          roleTag: 'scaffold',
          sequence: condSeq,
          residues,
        },
      ];
      totalResidues = condSeq.length;
      designedResidues = condSeq.length;
      fixedResidues = 0;
      estimatedTokens = condSeq.length;

      fastaEntries = seqArt.candidates.map((cand, idx) => ({
        id: cand.id,
        header:
          idx === 0
            ? `>${cand.id} (${condSeq.length} aa conditioned${strippedChainBreaks ? ' · ":" chain breaks stripped' : ''})`
            : `>${cand.id} (ignored)`,
        sequence: idx === 0 ? condSeq : cand.sequence,
        ...(idx === 0 ? { chainLengths: [condSeq.length] } : cand.chainLengths ? { chainLengths: cand.chainLengths } : {}),
        selected: idx === 0,
        ...(idx > 0 ? { scoreLabel: 'ignored (candidate 0 only)' } : {}),
      }));

      if (strippedChainBreaks && !issues.some((i) => i.message.includes('chain breaks'))) {
        pushIssue({
          severity: 'warning',
          source: 'compiler',
          message: `${nodeLabel} strips ":" chain breaks from multi-chain input "${cand0.id}" and collapses it into a single ${condSeq.length}-aa monomer chain.`,
        });
      }

      payload = {
        format: 'raw-input',
        headline: `FoldFlow 2 (Sequence-Conditioned · ${condSeq.length} aa${strippedChainBreaks ? ' · ":" chain breaks stripped' : ''}) · ${steps} steps`,
        compiledSummary: {
          mode: 'sequence-conditioned',
          precision,
          conditionedCandidate: cand0.id,
          conditionedLength: condSeq.length,
          ...(strippedChainBreaks
            ? {
                strippedChainBreaks: true,
                ...(hasMultiChainLengths ? { originalChainLengths: cand0.chainLengths!.map(String) } : {}),
              }
            : {}),
          ...(seqArt.candidates.length > 1 ? { ignoredCandidates: seqArt.candidates.length - 1 } : {}),
          steps,
          rotationScaling,
          selfConditioning,
          numDesigns,
          seed,
        },
        rawSpecText: JSON.stringify(
          {
            adapter: 'foldflow2',
            mode: 'sequence-conditioned',
            precision,
            conditionedCandidate: cand0.id,
            conditionedLength: condSeq.length,
            strippedChainBreaks,
            ...(hasMultiChainLengths ? { originalChainLengths: cand0.chainLengths } : {}),
            ignoredCandidates: Math.max(0, seqArt.candidates.length - 1),
            steps,
            rotationScaling,
            selfConditioning,
            numDesigns,
            seed,
          },
          null,
          2,
        ),
      };
    } else {
      runRuntimeValidation();
      const track = buildSyntheticDesignedChain('A', length, 0, 'scaffold', 'A');
      chains = [track];
      totalResidues = length;
      designedResidues = length;
      fixedResidues = 0;
      estimatedTokens = length;
      fastaEntries = [
        {
          id: 'foldflow2-uncond',
          header: `>FoldFlow2_unconditional (${length} aa)`,
          sequence: track.sequence,
          chainLengths: [length],
          selected: true,
        },
      ];
      payload = {
        format: 'raw-input',
        headline: `FoldFlow 2 (Unconditional) · L=${length} aa (${steps} steps)`,
        compiledSummary: {
          mode: 'unconditional',
          precision,
          length,
          steps,
          rotationScaling,
          selfConditioning,
          numDesigns,
          seed,
        },
        rawSpecText: JSON.stringify(
          { adapter: 'foldflow2', mode: 'unconditional', precision, length, steps, rotationScaling, selfConditioning, numDesigns, seed },
          null,
          2,
        ),
      };
    }
  } else {
    const fallback = compileSchemaFallbackPreview({
      node,
      nodeLabel,
      manifest,
      resolvedInputs,
      resolvedInputLists,
      missingUpstreamByPort,
      allMissingUpstream,
      runRuntimeValidation,
    });
    availability = fallback.availability;
    missingUpstreamNodeIds = fallback.missingUpstreamNodeIds;
    chains = fallback.chains;
    entities = fallback.entities;
    fastaEntries = fallback.fastaEntries;
    previewStructurePdb = fallback.previewStructurePdb;
    previewStructureMeta = fallback.previewStructureMeta;
    highlightLayers = fallback.highlightLayers;
    payload = fallback.payload;
    totalResidues = fallback.totalResidues;
    designedResidues = fallback.designedResidues;
    fixedResidues = fallback.fixedResidues;
    estimatedTokens = fallback.estimatedTokens;
  }

  // Promote error availability if schema/runtime/compiler emitted blocking errors while not pending-upstream (CT-08)
  if (
    availability === 'ready' &&
    issues.some(
      (i) =>
        i.severity === 'error' &&
        (i.source === 'schema' || i.source === 'compiler' || i.source === 'runtime'),
    )
  ) {
    availability = 'error';
  }

  const budget = {
    totalResidues,
    designedResidues,
    fixedResidues,
    estimatedTokens,
    nodeWeightsMB: nodeReq.totalDownloadMB,
    nodeVramMB: nodeReq.peakVramMB,
    subgraphDownloadMB: subReq.totalDownloadMB,
    subgraphPeakVramMB: subReq.peakVramMB,
  };

  const digestInput = JSON.stringify({
    nodeId,
    adapterId: node.adapterId,
    availability,
    missingUpstreamNodeIds,
    params: node.params,
    inputs: Object.fromEntries(Object.entries(resolvedInputs).map(([k, v]) => [k, v.id])),
    payload: { format: payload.format, headline: payload.headline, rawSpecText: payload.rawSpecText },
    chains: chains.map((c) => [
      c.chainId,
      c.labelChainId ?? '',
      c.msaStatus ?? '',
      c.msaDepth ?? null,
      c.sequence,
      c.residues.map((r) => [r.flatIndex, r.token, r.roles, r.tieGroupId ?? 0, Boolean(r.inheritedLock), r.gapBefore ?? 0]),
    ]),
    entities,
    issues: issues.map((i) => [i.severity, i.source, i.message]),
  });
  const previewDigest = `${fnv1a(digestInput)}${fnv1a(`v2:${digestInput}`)}`;

  return {
    nodeId,
    adapterId: node.adapterId,
    availability,
    missingUpstreamNodeIds,
    chains,
    entities,
    fastaEntries,
    previewStructurePdb,
    ...(previewStructureMeta ? { previewStructureMeta } : {}),
    highlightLayers,
    payload,
    budget,
    issues,
    previewDigest,
  };
}
