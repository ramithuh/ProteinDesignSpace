// Pure-TS multi-block 3D comparison & structural alignment helper.
// Builds color-coded Mol* multi-model overlays (Kabsch-aligned when same length or
// sharing a designed backbone, with chain-role/sequence matching for multi-chain complexes,
// or clean +X side-by-side offset when Side-by-Side layout is toggled) + pairwise inter-model
// Cα RMSD/TM-score and all-atom mmCIF preservation for holo/DNA structures.

import {
  applyTransformToPdb,
  computeGeometryStats,
  kabschAlign,
  kabschAlignSubstructure,
  type GeometryStats,
  type KabschResult,
} from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { computeIpsae, sliceProteinPae } from '$lib/protein-ops/ipsae';
import { ADAPTER_REGISTRY } from './registry';
import { resolveFoldConfidence, selectFoldRepresentative } from './bio-capabilities';
import { analyzeSequence, type SequenceAnalysis } from './peek';
import type {
  CanonicalArtifact,
  CollectionKind,
  DesignabilityMetrics,
  FoldedStructure,
  SequenceSet,
  StructureArtifact,
  TrajectoryFramesArtifact,
  WorkflowGraph,
} from './types';

export type ComparisonPairingMode = 'by-candidate' | 'branch-best';

export interface KeyedCollectionItem<T = unknown> {
  key: string;
  sequence?: string;
  rank?: number;
  score?: number | null;
  confidence?: number | null;
  confidenceAvailable?: boolean;
  ipsae?: number | null;
  value: T;
}

export interface KeyedCollectionBranch<T = unknown> {
  branchId: string;
  label?: string;
  collectionKind?: CollectionKind;
  items: readonly KeyedCollectionItem<T>[];
}

export interface JoinedBranchEntry<T = unknown> {
  branchId: string;
  present: boolean;
  missingReason?: string;
  rank: number | null;
  score: number | null;
  confidence: number | null;
  confidenceAvailable: boolean;
  ipsae?: number | null;
  item: KeyedCollectionItem<T> | null;
}

export interface JoinedCollectionRow<T = unknown> {
  key: string;
  sequence?: string;
  presentCount: number;
  missingBranchIds: string[];
  byBranch: Record<string, JoinedBranchEntry<T>>;
}

export interface KeyedCollectionJoinResult<T = unknown> {
  keys: string[];
  branchIds: string[];
  rows: JoinedCollectionRow<T>[];
  completeRowCount: number;
  partialRowCount: number;
}

export interface BranchCandidateRef {
  nodeId: string;
  kind: 'metrics' | 'fold' | 'structure' | 'trajectory' | 'sequence';
  index: number;
  key: string;
  sequenceId: string;
  sequence: string;
}

export interface CompareColorSlot {
  badge: string;
  css: string;
  hex: number;
}

export const COMPARE_PALETTE: CompareColorSlot[] = [
  { badge: 'A', css: '#38bdf8', hex: 0x38bdf8 }, // Sky
  { badge: 'B', css: '#f59e0b', hex: 0xf59e0b }, // Amber
  { badge: 'C', css: '#c084fc', hex: 0xc084fc }, // Violet
  { badge: 'D', css: '#34d399', hex: 0x34d399 }, // Emerald
  { badge: 'E', css: '#fb7185', hex: 0xfb7185 }, // Rose
  { badge: 'F', css: '#22d3ee', hex: 0x22d3ee }, // Cyan
];

export const REFERENCE_BACKBONE_COLOR = {
  css: '#94a3b8',
  hex: 0x94a3b8,
};

export interface ViewerModelSpec {
  id: string;
  nodeId: string | null;
  badge?: string;
  label: string;
  pdbText: string;
  colorHex: number;
  colorCss: string;
  alpha: number;
  colorByPlddt?: boolean;
  isReference?: boolean;
  visible: boolean;
  metricSummary?: string;
}

export interface CompareBranchSummary {
  nodeId: string;
  label: string;
  adapterId: string;
  badge: string;
  colorCss: string;
  colorHex: number;
  hasOutputs: boolean;
  has3D: boolean;
  kind: 'metrics' | 'fold' | 'structure' | 'sequence' | 'none';
  structureId: string | null;
  designedPdb: string | null;
  pdbText: string | null;
  cifText?: string | null;
  caCoords: Float32Array | null;
  resNums: number[] | null;
  chainIds?: string[] | null;
  chainLengths?: number[] | null;
  chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'> | null;
  sequence?: string | null;
  motifIndices?: number[] | null;
  benchmarkId?: string | null;
  hasNonProtein3D?: boolean;
  length: number | null;
  sequenceId: string | null;
  candidateKey?: string | null;
  missingSelectedCandidate?: boolean;
  rankingPolicy?: string;
  branchRank?: number | null;
  confidenceAvailable?: boolean;
  collectionKind?: CollectionKind;
  candidateCount: number;
  selectedIndex: number;
  scRmsd: number | null;
  scTm: number | null;
  meanPlddt: number | null;
  ptm: number | null;
  iptm: number | null;
  ipae: number | null;
  ipsae: number | null;
  ipsaeMin: number | null;
  binderPlddt: number | null;
  targetPlddt: number | null;
  binderScRmsd: number | null;
  targetAlignedBinderRmsd: number | null;
  genMotifRmsd: number | null;
  refoldMotifRmsd: number | null;
  designable: boolean | null;
  foldElapsedMs: number | null;
  geometry: GeometryStats | null;
  pae: Float32Array | null;
  mpnnScore: number | null;
  globalScore: number | null;
  seqRecoveryPct: number | null;
  seqAnalysis: SequenceAnalysis | null;
}

export type CompareBranchMetrics = CompareBranchSummary;

export interface PairwiseAgreement {
  labelA: string;
  labelB: string;
  badgeA: string;
  badgeB: string;
  rmsd: number;
  tmScore: number;
  alignedResidues?: number;
  isSubstructure?: boolean;
  /** True when both compared branches share an upstream template/structure ancestor. */
  hasSharedTemplate?: boolean;
  /** Qualified alignment method label (e.g. 'Motif Kabsch' vs 'Substructure Kabsch'). */
  alignmentLabel?: string;
}

export interface ComparisonState {
  branches: CompareBranchSummary[];
  models: ViewerModelSpec[];
  effectiveLayout: 'overlay' | 'side-by-side';
  forcedSideBySide: boolean;
  kabschAligned: boolean;
  /** True when all compared branches are scorer outputs sharing the exact same pre-aligned designed backbone. */
  isPreAlignedSharedBackbone: boolean;
  sharedReferenceLabel: string | null;
  pairwise: PairwiseAgreement | null;
  /** Pairwise agreements across all alignable pairs (for 3–4 block comparisons). */
  allPairwise: PairwiseAgreement[];
  /** Active candidate pairing mode across compared branches. */
  pairingMode: ComparisonPairingMode;
  /** Human-readable label distinguishing candidate-identity pairing vs branch-best ranking. */
  pairingLabel: string;
  /** Currently paired candidate key when in `'by-candidate'` mode (or `null` when unavailable). */
  selectedCandidateKey: string | null;
  /** Ordered union of candidate keys across compared branches. */
  candidateKeys: string[];
  /** Keyed join rows across all compared branches, explicitly flagging missing members. */
  pairedCandidates: JoinedCollectionRow<BranchCandidateRef>[];
}

interface ChainSegment {
  chainId: string;
  start: number;
  length: number;
  role?: 'target' | 'binder' | 'scaffold';
  seq: string;
}

const IDENTITY_ROT = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

function isMmcifText(text: string): boolean {
  return /^\s*(data_|loop_|_atom_site\.)/m.test(text);
}

function centroidOf(coords: Float32Array): [number, number, number] {
  const n = Math.floor(coords.length / 3);
  if (n === 0) return [0, 0, 0];
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < n; i++) {
    cx += coords[i * 3];
    cy += coords[i * 3 + 1];
    cz += coords[i * 3 + 2];
  }
  return [cx / n, cy / n, cz / n];
}

interface CifTokSpan {
  value: string;
  start: number;
  end: number;
}

function tokenizeCifLineWithSpans(line: string): CifTokSpan[] {
  const out: CifTokSpan[] = [];
  let i = 0;
  const len = line.length;
  while (i < len) {
    while (i < len && (line[i] === ' ' || line[i] === '\t' || line[i] === '\r')) i++;
    if (i >= len || line[i] === '#') break;
    const start = i;
    if (line[i] === "'" || line[i] === '"') {
      const q = line[i++];
      while (
        i < len &&
        !(
          line[i] === q &&
          (i + 1 === len || line[i + 1] === ' ' || line[i + 1] === '\t' || line[i + 1] === '\r')
        )
      ) {
        i++;
      }
      if (i < len) i++;
      out.push({ value: line.slice(start + 1, i - 1), start, end: i });
    } else {
      while (i < len && line[i] !== ' ' && line[i] !== '\t' && line[i] !== '\r') i++;
      out.push({ value: line.slice(start, i), start, end: i });
    }
  }
  return out;
}

/** Apply rigid rotation R + translation t to every `_atom_site` row of an mmCIF string. */
function applyTransformToCif(cifText: string, R: Float64Array, t: Float64Array): string {
  const lines = cifText.split('\n');
  const out = lines.slice();
  let inLoop = false;
  let cols: string[] = [];
  let inAtomSiteData = false;
  let iX = -1;
  let iY = -1;
  let iZ = -1;
  let maxCoordCol = -1;

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (trimmed === 'loop_') {
      inLoop = true;
      cols = [];
      inAtomSiteData = false;
      iX = iY = iZ = maxCoordCol = -1;
      continue;
    }
    if (inLoop && trimmed.startsWith('_')) {
      cols.push(trimmed.split(/\s+/)[0]);
      continue;
    }
    if (inLoop && cols.length > 0 && !inAtomSiteData) {
      if (cols[0].startsWith('_atom_site.')) {
        iX = cols.indexOf('_atom_site.Cartn_x');
        iY = cols.indexOf('_atom_site.Cartn_y');
        iZ = cols.indexOf('_atom_site.Cartn_z');
        maxCoordCol = Math.max(iX, iY, iZ);
        inAtomSiteData = iX >= 0 && iY >= 0 && iZ >= 0;
      }
      inLoop = false;
    }
    if (inAtomSiteData) {
      if (
        trimmed === '' ||
        trimmed.startsWith('#') ||
        trimmed.startsWith('loop_') ||
        trimmed.startsWith('data_') ||
        trimmed.startsWith('_')
      ) {
        inAtomSiteData = false;
        if (trimmed === 'loop_') {
          inLoop = true;
          cols = [];
        }
        continue;
      }
      const toks = tokenizeCifLineWithSpans(lines[i]);
      if (toks.length <= maxCoordCol) continue;
      const x = parseFloat(toks[iX].value);
      const y = parseFloat(toks[iY].value);
      const z = parseFloat(toks[iZ].value);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      const nx = R[0] * x + R[1] * y + R[2] * z + t[0];
      const ny = R[3] * x + R[4] * y + R[5] * z + t[1];
      const nz = R[6] * x + R[7] * y + R[8] * z + t[2];

      const replacements = [
        { span: toks[iX], text: nx.toFixed(3) },
        { span: toks[iY], text: ny.toFixed(3) },
        { span: toks[iZ], text: nz.toFixed(3) },
      ].sort((a, b) => b.span.start - a.span.start);

      let line = lines[i];
      for (const rep of replacements) {
        line = line.slice(0, rep.span.start) + rep.text + line.slice(rep.span.end);
      }
      out[i] = line;
    }
  }
  return out.join('\n');
}

function applyTransformToStructureText(text: string, R: Float64Array, t: Float64Array): string {
  return isMmcifText(text) ? applyTransformToCif(text, R, t) : applyTransformToPdb(text, R, t);
}

function centroidOfStructureText(text: string): [number, number, number] {
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let count = 0;
  if (isMmcifText(text)) {
    const lines = text.split('\n');
    let inLoop = false;
    let cols: string[] = [];
    let inData = false;
    let iX = -1;
    let iY = -1;
    let iZ = -1;
    let maxCol = -1;
    for (const raw of lines) {
      const trimmed = raw.trim();
      if (trimmed === 'loop_') {
        inLoop = true;
        cols = [];
        inData = false;
        continue;
      }
      if (inLoop && trimmed.startsWith('_')) {
        cols.push(trimmed.split(/\s+/)[0]);
        continue;
      }
      if (inLoop && cols.length > 0 && !inData) {
        if (cols[0].startsWith('_atom_site.')) {
          iX = cols.indexOf('_atom_site.Cartn_x');
          iY = cols.indexOf('_atom_site.Cartn_y');
          iZ = cols.indexOf('_atom_site.Cartn_z');
          maxCol = Math.max(iX, iY, iZ);
          inData = iX >= 0 && iY >= 0 && iZ >= 0;
        }
        inLoop = false;
      }
      if (inData) {
        if (
          trimmed === '' ||
          trimmed.startsWith('#') ||
          trimmed.startsWith('loop_') ||
          trimmed.startsWith('data_') ||
          trimmed.startsWith('_')
        ) {
          inData = false;
          continue;
        }
        const toks = tokenizeCifLineWithSpans(raw);
        if (toks.length <= maxCol) continue;
        const x = parseFloat(toks[iX].value);
        const y = parseFloat(toks[iY].value);
        const z = parseFloat(toks[iZ].value);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          sx += x;
          sy += y;
          sz += z;
          count++;
        }
      }
    }
  } else {
    for (const line of text.split('\n')) {
      if ((line.startsWith('ATOM  ') || line.startsWith('HETATM')) && line.length >= 54) {
        const x = parseFloat(line.slice(30, 38));
        const y = parseFloat(line.slice(38, 46));
        const z = parseFloat(line.slice(46, 54));
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          sx += x;
          sy += y;
          sz += z;
          count++;
        }
      }
    }
  }
  return count > 0 ? [sx / count, sy / count, sz / count] : [0, 0, 0];
}

function findStructureArtifactById(
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
  structureId: string | null | undefined,
): StructureArtifact | null {
  if (!structureId) return null;
  for (const outs of Object.values(outputsByNode)) {
    for (const art of Object.values(outs)) {
      if (art.kind === 'StructureArtifact' && art.id === structureId) {
        return art;
      }
    }
  }
  return null;
}

function findUpstreamStructureArtifact(
  graph: WorkflowGraph,
  nodeId: string,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
): StructureArtifact | null {
  const visited = new Set<string>([nodeId]);
  const queue = [nodeId];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    const incoming = graph.edges
      .filter((e) => e.toNode === cur)
      .sort((a, b) => (a.toPort === 'structure' ? -1 : b.toPort === 'structure' ? 1 : 0));
    for (const e of incoming) {
      if (visited.has(e.fromNode)) continue;
      visited.add(e.fromNode);
      const outs = outputsByNode[e.fromNode];
      if (outs) {
        const st = Object.values(outs).find(
          (a): a is StructureArtifact => a.kind === 'StructureArtifact',
        );
        if (st) return st;
      }
      queue.push(e.fromNode);
    }
  }
  return null;
}

function getAncestorNodeIds(graph: WorkflowGraph, nodeId: string): Set<string> {
  const visited = new Set<string>([nodeId]);
  const queue = [nodeId];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    for (const e of graph.edges) {
      if (e.toNode === cur && !visited.has(e.fromNode)) {
        visited.add(e.fromNode);
        queue.push(e.fromNode);
      }
    }
  }
  return visited;
}

function computeIpaeFromMatrix(pae: Float32Array, chainLengths: readonly number[]): number | null {
  if (chainLengths.length < 2) return null;
  const L = chainLengths.reduce((a, b) => a + b, 0);
  if (L <= 0 || pae.length < L * L) return null;
  const paeDim = Math.round(Math.sqrt(pae.length));
  if (paeDim < L || paeDim * paeDim !== pae.length) return null;
  const chainOf = new Int32Array(L);
  let off = 0;
  for (let c = 0; c < chainLengths.length; c++) {
    for (let k = 0; k < chainLengths[c]; k++) chainOf[off++] = c;
  }
  let sum = 0;
  let count = 0;
  for (let i = 0; i < L; i++) {
    const ci = chainOf[i];
    for (let j = 0; j < L; j++) {
      if (ci === chainOf[j]) continue;
      const v = pae[i * paeDim + j];
      if (Number.isFinite(v)) {
        sum += v;
        count++;
      }
    }
  }
  return count > 0 ? sum / count : null;
}

function computeIpsaeFromFoldMatrix(
  pae: Float32Array,
  chainLengths: readonly number[],
  chainIds?: readonly string[],
): { ipsae: number; ipsaeMin: number } | null {
  const L =
    chainIds && chainIds.length > 0
      ? chainIds.length
      : chainLengths.reduce((a, b) => a + b, 0);
  if (L <= 0) return null;
  let perResChains: string[];
  if (chainIds && chainIds.length === L) {
    perResChains = [...chainIds];
  } else if (chainLengths.length > 1 && chainLengths.reduce((a, b) => a + b, 0) === L) {
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    perResChains = [];
    for (let c = 0; c < chainLengths.length; c++) {
      const cid = alphabet[c % alphabet.length];
      for (let k = 0; k < chainLengths[c]; k++) perResChains.push(cid);
    }
  } else {
    return null;
  }
  if (new Set(perResChains).size < 2) return null;
  const sliced = sliceProteinPae(pae, L);
  if (!sliced) return null;
  const res = computeIpsae(sliced, perResChains, 10);
  return res ? { ipsae: res.ipsae, ipsaeMin: res.ipsaeMin } : null;
}

/**
 * General keyed multi-branch collection join helper (Issue #5 / Issue #3).
 * Pairs items across arbitrary branches by canonical candidate identity `key`
 * (regardless of branch rank order), explicitly marking missing members on any branch
 * and preserving confidence-less samples without fabricating confidence scores.
 */
export function joinKeyedCollections<T = unknown>(
  branches: readonly KeyedCollectionBranch<T>[],
  opts: { matchBySequenceFallback?: boolean } = {},
): KeyedCollectionJoinResult<T> {
  const branchIds = branches.map((b) => b.branchId);
  const matchBySeq = Boolean(opts.matchBySequenceFallback);

  // Disambiguate duplicate keys within the same branch deterministically so multi-sample
  // members sharing a base key are never silently dropped or collapsed.
  const normalizedBranches: KeyedCollectionBranch<T>[] = branches.map((br) => {
    const usedKeys = new Set<string>();
    const baseCounts = new Map<string, number>();
    const items = br.items.map((it) => {
      const base = it.key;
      const count = (baseCounts.get(base) ?? 0) + 1;
      baseCounts.set(base, count);
      if (!usedKeys.has(base)) {
        usedKeys.add(base);
        return it;
      }
      let suffix = Math.max(2, count);
      let disambig = `${base}#${suffix}`;
      while (usedKeys.has(disambig)) {
        suffix++;
        disambig = `${base}#${suffix}`;
      }
      usedKeys.add(disambig);
      return { ...it, key: disambig };
    });
    return { ...br, items };
  });

  // Track which item indices in each branch have been matched so sequence fallback never double-binds.
  const consumedByBranch = new Map<string, Set<number>>();
  const keyIndexByBranch = new Map<string, Map<string, number>>();
  for (const br of normalizedBranches) {
    consumedByBranch.set(br.branchId, new Set());
    const idxMap = new Map<string, number>();
    br.items.forEach((it, idx) => {
      if (!idxMap.has(it.key)) idxMap.set(it.key, idx);
    });
    keyIndexByBranch.set(br.branchId, idxMap);
  }

  // First determine whether any branch shares exact keys; sequence fallback is only used
  // for items that have no exact-key peer in another branch.
  const rows: JoinedCollectionRow<T>[] = [];
  const keys: string[] = [];
  const seenKeys = new Set<string>();

  for (const sourceBranch of normalizedBranches) {
    const srcConsumed = consumedByBranch.get(sourceBranch.branchId)!;
    for (let srcIdx = 0; srcIdx < sourceBranch.items.length; srcIdx++) {
      if (srcConsumed.has(srcIdx)) continue;
      const anchorItem = sourceBranch.items[srcIdx];
      const key = anchorItem.key;
      if (seenKeys.has(key)) continue;
      seenKeys.add(key);
      keys.push(key);

      const normAnchorSeq = anchorItem.sequence?.trim().toUpperCase() || '';
      const byBranch: Record<string, JoinedBranchEntry<T>> = {};
      const missingBranchIds: string[] = [];
      let presentCount = 0;

      for (const br of normalizedBranches) {
        const consumed = consumedByBranch.get(br.branchId)!;
        let matchIdx: number | undefined = keyIndexByBranch.get(br.branchId)!.get(key);
        if (matchIdx !== undefined && consumed.has(matchIdx)) {
          matchIdx = undefined;
        }
        if (matchIdx === undefined && matchBySeq && normAnchorSeq.length > 0) {
          // Only fallback-match by exact sequence when unique & unconsumed in this branch
          const seqMatches: number[] = [];
          for (let j = 0; j < br.items.length; j++) {
            if (consumed.has(j)) continue;
            const candSeq = br.items[j].sequence?.trim().toUpperCase() || '';
            if (candSeq === normAnchorSeq) seqMatches.push(j);
          }
          if (seqMatches.length === 1) {
            matchIdx = seqMatches[0];
          }
        }

        if (matchIdx !== undefined) {
          consumed.add(matchIdx);
          const item = br.items[matchIdx];
          const hasConf =
            item.confidenceAvailable !== false &&
            typeof item.confidence === 'number' &&
            Number.isFinite(item.confidence);
          byBranch[br.branchId] = {
            branchId: br.branchId,
            present: true,
            rank: item.rank ?? matchIdx + 1,
            score: typeof item.score === 'number' && Number.isFinite(item.score) ? item.score : null,
            confidence: hasConf ? (item.confidence as number) : null,
            confidenceAvailable: hasConf,
            ...(typeof item.ipsae === 'number' && Number.isFinite(item.ipsae)
              ? { ipsae: item.ipsae }
              : { ipsae: null }),
            item,
          };
          presentCount++;
        } else {
          missingBranchIds.push(br.branchId);
          byBranch[br.branchId] = {
            branchId: br.branchId,
            present: false,
            missingReason: `Candidate "${key}" absent in branch "${br.label ?? br.branchId}"`,
            rank: null,
            score: null,
            confidence: null,
            confidenceAvailable: false,
            ipsae: null,
            item: null,
          };
        }
      }

      rows.push({
        key,
        ...(normAnchorSeq ? { sequence: normAnchorSeq } : {}),
        presentCount,
        missingBranchIds,
        byBranch,
      });
    }
  }

  let completeRowCount = 0;
  let partialRowCount = 0;
  for (const r of rows) {
    if (r.missingBranchIds.length === 0) completeRowCount++;
    else partialRowCount++;
  }

  return {
    keys,
    branchIds,
    rows,
    completeRowCount,
    partialRowCount,
  };
}

function extractBranchCandidateItems(
  graph: WorkflowGraph,
  nodeId: string,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
): KeyedCollectionBranch<BranchCandidateRef> {
  const node = graph.nodes.find((n) => n.id === nodeId);
  const label =
    node?.label || (node ? ADAPTER_REGISTRY[node.adapterId]?.shortName : undefined) || nodeId;
  const outs = outputsByNode[nodeId];
  if (!outs) return { branchId: nodeId, label, items: [] };
  const values = Object.values(outs);

  const metrics = values.find((a): a is DesignabilityMetrics => a.kind === 'DesignabilityMetrics');
  if (metrics && metrics.rows.length > 0) {
    return {
      branchId: nodeId,
      label,
      collectionKind: metrics.collectionKind ?? 'independent-candidates',
      items: metrics.rows.map((row, idx) => {
        const key = row.candidateKey ?? row.sequenceId;
        const hasConf = row.confidenceAvailable !== false && Number.isFinite(row.meanPlddt);
        return {
          key,
          sequence: row.sequence,
          rank: idx + 1,
          score: Number.isFinite(row.scRmsd) ? row.scRmsd : null,
          confidence: hasConf ? row.meanPlddt : null,
          confidenceAvailable: hasConf,
          ipsae: typeof row.ipsae === 'number' && Number.isFinite(row.ipsae) ? row.ipsae : null,
          value: {
            nodeId,
            kind: 'metrics' as const,
            index: idx,
            key,
            sequenceId: row.sequenceId,
            sequence: row.sequence,
          },
        };
      }),
    };
  }

  const folds = values.find((a): a is FoldedStructure => a.kind === 'FoldedStructure');
  if (folds && folds.folds.length > 0) {
    return {
      branchId: nodeId,
      label,
      collectionKind: folds.collectionKind ?? 'independent-candidates',
      items: folds.folds.map((f, idx) => {
        const key =
          f.key ??
          (f.sampleIndex !== undefined && f.sampleIndex !== null
            ? `${f.sequenceId}#sample-${f.sampleIndex}`
            : f.sequenceId);
        const conf = resolveFoldConfidence(f);
        const foldIpsae =
          typeof f.ipsae === 'number' && Number.isFinite(f.ipsae)
            ? f.ipsae
            : f.pae && ((f.chainLengths && f.chainLengths.length > 1) || (f.chainIds && new Set(f.chainIds).size > 1))
              ? (computeIpsaeFromFoldMatrix(f.pae, f.chainLengths ?? [], f.chainIds)?.ipsae ?? null)
              : null;
        return {
          key,
          sequence: f.sequence,
          rank: idx + 1,
          score: conf,
          confidence: conf,
          confidenceAvailable: conf !== null,
          ipsae: foldIpsae,
          value: {
            nodeId,
            kind: 'fold' as const,
            index: idx,
            key,
            sequenceId: f.sequenceId,
            sequence: f.sequence,
          },
        };
      }),
    };
  }

  const seqSet = values.find((a): a is SequenceSet => a.kind === 'SequenceSet');
  if (seqSet && seqSet.candidates.length > 0) {
    return {
      branchId: nodeId,
      label,
      collectionKind: seqSet.collectionKind ?? 'independent-candidates',
      items: seqSet.candidates.map((c, idx) => {
        const key =
          c.key ??
          (c.sampleIndex !== undefined && c.sampleIndex !== null
            ? `${c.id}#sample-${c.sampleIndex}`
            : c.id);
        return {
          key,
          sequence: c.sequence,
          rank: idx + 1,
          score: c.score,
          confidence: null,
          confidenceAvailable: false,
          value: {
            nodeId,
            kind: 'sequence' as const,
            index: idx,
            key,
            sequenceId: c.id,
            sequence: c.sequence,
          },
        };
      }),
    };
  }

  const traj = values.find((a): a is TrajectoryFramesArtifact => a.kind === 'TrajectoryFrames');
  if (traj) {
    const frames = traj.frames ?? [];
    if (frames.length > 0) {
      return {
        branchId: nodeId,
        label,
        collectionKind: 'ordered-frames',
        items: frames.map((fr, idx) => ({
          key: fr.key,
          rank: fr.frameIndex + 1,
          score: null,
          confidence: typeof fr.confidence === 'number' && Number.isFinite(fr.confidence) ? fr.confidence : null,
          confidenceAvailable: typeof fr.confidence === 'number' && Number.isFinite(fr.confidence),
          value: {
            nodeId,
            kind: 'trajectory' as const,
            index: idx,
            key: fr.key,
            sequenceId: fr.key,
            sequence: '',
          },
        })),
      };
    }
    return {
      branchId: nodeId,
      label,
      collectionKind: 'ordered-frames',
      items: [
        {
          key: traj.id,
          rank: 1,
          score: null,
          confidence: null,
          confidenceAvailable: false,
          value: {
            nodeId,
            kind: 'trajectory',
            index: 0,
            key: traj.id,
            sequenceId: traj.label ?? traj.id,
            sequence: '',
          },
        },
      ],
    };
  }

  const struct = values.find((a): a is StructureArtifact => a.kind === 'StructureArtifact');
  if (struct) {
    const key = struct.representativeSelection?.selectedKey ?? struct.id;
    return {
      branchId: nodeId,
      label,
      items: [
        {
          key,
          sequence: struct.sequence,
          rank: 1,
          score: null,
          confidence: struct.representativeSelection?.scoreValue ?? null,
          confidenceAvailable:
            typeof struct.representativeSelection?.scoreValue === 'number' &&
            Number.isFinite(struct.representativeSelection.scoreValue),
          value: {
            nodeId,
            kind: 'structure',
            index: 0,
            key,
            sequenceId: struct.label,
            sequence: struct.sequence,
          },
        },
      ],
    };
  }

  return { branchId: nodeId, label, items: [] };
}

/**
 * Join multiple workflow branch outputs (metrics, folds, sequence sets, trajectory frames, or structures)
 * into candidate-keyed comparison rows with explicit missing-member markers.
 */
export function joinBranchArtifacts(
  graph: WorkflowGraph,
  branchNodeIds: readonly string[],
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
  opts: { matchBySequenceFallback?: boolean } = {},
): KeyedCollectionJoinResult<BranchCandidateRef> {
  const branches = branchNodeIds.map((id) => extractBranchCandidateItems(graph, id, outputsByNode));
  return joinKeyedCollections(branches, opts);
}

/**
 * Export a keyed multi-branch join result as a clean CSV table with explicit
 * `MISSING` markers for absent branch members and `N/A` for confidence-less samples.
 */
export function exportJoinedCollectionCsv<T = unknown>(
  joined: KeyedCollectionJoinResult<T>,
): string {
  const esc = (v: string): string => {
    // Neutralize spreadsheet formula injection on non-numeric text cells while preserving negative numbers.
    const isNumeric = /^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(v);
    const safe = !isNumeric && /^[=+\-@\t\r]/.test(v) ? `'${v}` : v;
    return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
  };
  const includeIpsae = joined.rows.some((r) =>
    joined.branchIds.some((bid) => {
      const val = r.byBranch[bid]?.ipsae;
      return typeof val === 'number' && Number.isFinite(val);
    }),
  );
  const headers = [
    'candidate_key',
    'present_count',
    'missing_branches',
    ...joined.branchIds.flatMap((id) => [
      `${id}_status`,
      `${id}_rank`,
      `${id}_score`,
      `${id}_confidence`,
      ...(includeIpsae ? [`${id}_ipsae`] : []),
    ]),
  ];
  const lines = [headers.map(esc).join(',')];
  for (const row of joined.rows) {
    const cols: string[] = [
      row.key,
      String(row.presentCount),
      row.missingBranchIds.join(';'),
    ];
    for (const bid of joined.branchIds) {
      const entry = row.byBranch[bid];
      if (!entry || !entry.present) {
        cols.push('MISSING', '', '', '');
        if (includeIpsae) cols.push('');
      } else {
        cols.push(
          'PRESENT',
          entry.rank !== null ? String(entry.rank) : '',
          entry.score !== null ? String(entry.score) : '',
          entry.confidenceAvailable && entry.confidence !== null ? String(entry.confidence) : 'N/A',
        );
        if (includeIpsae) {
          cols.push(
            typeof entry.ipsae === 'number' && Number.isFinite(entry.ipsae)
              ? String(entry.ipsae)
              : 'N/A',
          );
        }
      }
    }
    lines.push(cols.map(esc).join(','));
  }
  return lines.join('\n');
}

function branchesShareCandidateIdentity(
  branchItems: readonly KeyedCollectionBranch<BranchCandidateRef>[],
): { hasSharedKey: boolean; hasSharedSequence: boolean } {
  const keyBranchCount = new Map<string, number>();
  const seqBranchCount = new Map<string, number>();
  for (const br of branchItems) {
    const localKeys = new Set<string>();
    const localSeqs = new Set<string>();
    for (const item of br.items) {
      if (item.key) localKeys.add(item.key);
      const seq = item.sequence?.trim().toUpperCase();
      if (seq && seq.length >= 3) localSeqs.add(seq);
    }
    for (const k of localKeys) keyBranchCount.set(k, (keyBranchCount.get(k) ?? 0) + 1);
    for (const s of localSeqs) seqBranchCount.set(s, (seqBranchCount.get(s) ?? 0) + 1);
  }
  let hasSharedKey = false;
  for (const count of keyBranchCount.values()) {
    if (count >= 2) {
      hasSharedKey = true;
      break;
    }
  }
  let hasSharedSequence = false;
  for (const count of seqBranchCount.values()) {
    if (count >= 2) {
      hasSharedSequence = true;
      break;
    }
  }
  return { hasSharedKey, hasSharedSequence };
}

function extractBranch(
  graph: WorkflowGraph,
  nodeId: string,
  slotIdx: number,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
  branchCandidateIdx: Record<string, number>,
  overrideSelection?: { index: number | null; candidateKey: string | null; missing?: boolean },
): CompareBranchSummary {
  const node = graph.nodes.find((n) => n.id === nodeId);
  const slot = COMPARE_PALETTE[slotIdx % COMPARE_PALETTE.length];
  const label =
    node?.label ||
    (node ? ADAPTER_REGISTRY[node.adapterId]?.shortName : undefined) ||
    nodeId;
  const adapterId = node?.adapterId ?? 'unknown';

  const empty: CompareBranchSummary = {
    nodeId,
    label,
    adapterId,
    badge: slot.badge,
    colorCss: slot.css,
    colorHex: slot.hex,
    hasOutputs: false,
    has3D: false,
    kind: 'none',
    structureId: null,
    designedPdb: null,
    pdbText: null,
    cifText: null,
    caCoords: null,
    resNums: null,
    chainIds: null,
    chainLengths: null,
    chainRoles: null,
    sequence: null,
    motifIndices: null,
    benchmarkId: null,
    hasNonProtein3D: false,
    length: null,
    sequenceId: null,
    candidateKey: overrideSelection?.candidateKey ?? null,
    missingSelectedCandidate: Boolean(overrideSelection?.missing),
    branchRank: null,
    candidateCount: 0,
    selectedIndex: 0,
    scRmsd: null,
    scTm: null,
    meanPlddt: null,
    ptm: null,
    iptm: null,
    ipae: null,
    ipsae: null,
    ipsaeMin: null,
    binderPlddt: null,
    targetPlddt: null,
    binderScRmsd: null,
    targetAlignedBinderRmsd: null,
    genMotifRmsd: null,
    refoldMotifRmsd: null,
    designable: null,
    foldElapsedMs: null,
    geometry: null,
    pae: null,
    mpnnScore: null,
    globalScore: null,
    seqRecoveryPct: null,
    seqAnalysis: null,
  };

  const outs = outputsByNode[nodeId];
  if (!outs) return empty;
  const values = Object.values(outs);
  if (values.length === 0) return empty;

  const metrics = values.find((a): a is DesignabilityMetrics => a.kind === 'DesignabilityMetrics');
  if (metrics && metrics.rows.length > 0) {
    if (overrideSelection?.missing) {
      return {
        ...empty,
        hasOutputs: true,
        kind: 'metrics',
        candidateCount: metrics.rows.length,
        selectedIndex: -1,
        candidateKey: overrideSelection.candidateKey,
        missingSelectedCandidate: true,
        rankingPolicy: 'lowest-scRmsd',
        collectionKind: metrics.collectionKind ?? 'independent-candidates',
      };
    }
    const rawIdx =
      overrideSelection?.index !== undefined && overrideSelection.index !== null
        ? overrideSelection.index
        : (branchCandidateIdx[nodeId] ?? 0);
    const idx = Math.max(0, Math.min(metrics.rows.length - 1, rawIdx));
    const row = metrics.rows[idx] ?? metrics.best ?? metrics.rows[0];
    const parsed = parseStructureText(row.alignedFoldPdb);
    const upstreamStruct =
      findStructureArtifactById(outputsByNode, row.structureId) ??
      findUpstreamStructureArtifact(graph, nodeId, outputsByNode);
    const designedHasNonProt = Boolean(
      upstreamStruct?.diagnostics?.hasLigands || upstreamStruct?.diagnostics?.hasNucleicAcids,
    );
    const designedDisplay =
      designedHasNonProt && upstreamStruct?.cifText ? upstreamStruct.cifText : row.designedPdb;
    const confAvail = row.confidenceAvailable !== false && Number.isFinite(row.meanPlddt);
    return {
      ...empty,
      hasOutputs: true,
      has3D: parsed.nResidues > 0,
      kind: 'metrics',
      structureId: row.structureId,
      designedPdb: designedDisplay,
      pdbText: row.alignedFoldPdb,
      cifText: null,
      caCoords: parsed.caCoords,
      resNums: parsed.resNums,
      chainIds: parsed.chainIds,
      chainLengths: upstreamStruct?.chainLengths ?? parsed.diagnostics.chainLengths,
      chainRoles: upstreamStruct?.chainRoles ?? null,
      sequence: row.sequence || parsed.sequence,
      motifIndices: upstreamStruct?.motifIndices ?? null,
      benchmarkId: upstreamStruct?.benchmarkId ?? null,
      length: parsed.nResidues,
      sequenceId: row.sequenceId,
      candidateKey: row.candidateKey ?? row.sequenceId,
      missingSelectedCandidate: false,
      rankingPolicy: 'lowest-scRmsd',
      branchRank: idx + 1,
      confidenceAvailable: confAvail,
      collectionKind: metrics.collectionKind ?? 'independent-candidates',
      candidateCount: metrics.rows.length,
      selectedIndex: idx,
      scRmsd: row.scRmsd,
      scTm: row.scTm,
      meanPlddt: confAvail ? row.meanPlddt : null,
      ptm: row.ptm,
      iptm: row.iptm ?? null,
      ipae: row.ipae ?? null,
      ipsae: typeof row.ipsae === 'number' && Number.isFinite(row.ipsae) ? row.ipsae : null,
      ipsaeMin: typeof row.ipsaeMin === 'number' && Number.isFinite(row.ipsaeMin) ? row.ipsaeMin : null,
      binderPlddt: row.binderPlddt ?? null,
      targetPlddt: row.targetPlddt ?? null,
      binderScRmsd: row.binderScRmsd ?? null,
      targetAlignedBinderRmsd: row.targetAlignedBinderRmsd ?? null,
      genMotifRmsd: row.genMotifRmsd ?? null,
      refoldMotifRmsd: row.refoldMotifRmsd ?? null,
      designable: row.designable,
      foldElapsedMs: row.foldElapsedMs,
      geometry: row.geometry,
      pae: row.pae ?? null,
    };
  }

  const folds = values.find((a): a is FoldedStructure => a.kind === 'FoldedStructure');
  if (folds && folds.folds.length > 0) {
    if (overrideSelection?.missing) {
      return {
        ...empty,
        hasOutputs: true,
        kind: 'fold',
        candidateCount: folds.folds.length,
        selectedIndex: -1,
        candidateKey: overrideSelection.candidateKey,
        missingSelectedCandidate: true,
        rankingPolicy: 'branch-order',
        collectionKind: folds.collectionKind ?? 'independent-candidates',
      };
    }
    const hasExplicitIdx =
      overrideSelection?.index !== undefined && overrideSelection.index !== null
        ? true
        : branchCandidateIdx[nodeId] !== undefined;
    let rawIdx: number;
    let rankingPolicy = 'branch-order';
    if (overrideSelection?.index !== undefined && overrideSelection.index !== null) {
      rawIdx = overrideSelection.index;
    } else if (branchCandidateIdx[nodeId] !== undefined) {
      rawIdx = branchCandidateIdx[nodeId];
    } else {
      // Branch-best default for FoldedStructure: pick the true representative winner
      // (highest confidence when available, deterministic fallback when confidence-less)
      const rep = selectFoldRepresentative(folds.folds, {
        collectionKind: folds.collectionKind,
      });
      rawIdx = rep?.selection.selectedIndex ?? 0;
      rankingPolicy = rep?.selection.criterion ?? 'branch-order';
    }
    const idx = Math.max(0, Math.min(folds.folds.length - 1, rawIdx));
    const f = folds.folds[idx] ?? folds.folds[0];
    const parsed = parseStructureText(f.pdbText);
    const geom =
      parsed.nResidues > 0
        ? computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums)
        : null;
    const upstreamStruct =
      findStructureArtifactById(outputsByNode, folds.parentStructureId) ??
      findUpstreamStructureArtifact(graph, nodeId, outputsByNode);
    const chainLengths = f.chainLengths ?? parsed.diagnostics.chainLengths;
    const effectiveChainIds = f.chainIds ?? parsed.chainIds;
    const ipae = f.pae && chainLengths.length > 1 ? computeIpaeFromMatrix(f.pae, chainLengths) : null;
    const computedIpsae =
      typeof f.ipsae === 'number' && Number.isFinite(f.ipsae)
        ? {
            ipsae: f.ipsae,
            ipsaeMin: typeof f.ipsaeMin === 'number' && Number.isFinite(f.ipsaeMin) ? f.ipsaeMin : null,
          }
        : f.pae && chainLengths.length > 1
          ? computeIpsaeFromFoldMatrix(f.pae, chainLengths, effectiveChainIds)
          : null;
    const conf = resolveFoldConfidence(f);
    // Compute confidence rank (1 = highest confidence) when not explicitly overridden
    const branchRank =
      !hasExplicitIdx && rankingPolicy === 'highest-mean-plddt' ? 1 : idx + 1;
    return {
      ...empty,
      hasOutputs: true,
      has3D: parsed.nResidues > 0,
      kind: 'fold',
      structureId: null,
      designedPdb: null,
      pdbText: f.pdbText,
      cifText: null,
      caCoords: parsed.caCoords,
      resNums: parsed.resNums,
      chainIds: effectiveChainIds,
      chainLengths,
      chainRoles: upstreamStruct?.chainRoles ?? null,
      sequence: f.sequence || parsed.sequence,
      motifIndices: upstreamStruct?.motifIndices ?? null,
      benchmarkId: upstreamStruct?.benchmarkId ?? null,
      length: parsed.nResidues,
      sequenceId: f.sequenceId,
      candidateKey: f.key ?? f.sequenceId,
      missingSelectedCandidate: false,
      rankingPolicy,
      branchRank,
      confidenceAvailable: conf !== null,
      collectionKind: folds.collectionKind ?? 'independent-candidates',
      candidateCount: folds.folds.length,
      selectedIndex: idx,
      meanPlddt: conf,
      ptm: f.ptm,
      iptm: f.iptm ?? null,
      ipae,
      ipsae: computedIpsae?.ipsae ?? null,
      ipsaeMin: computedIpsae?.ipsaeMin ?? null,
      foldElapsedMs: f.elapsedMs,
      geometry: geom,
      pae: f.pae ?? null,
    };
  }

  const traj = values.find((a): a is TrajectoryFramesArtifact => a.kind === 'TrajectoryFrames');
  if (traj) {
    if (overrideSelection?.missing) {
      return {
        ...empty,
        hasOutputs: true,
        kind: 'structure',
        candidateCount: traj.frames?.length || traj.frameCount,
        selectedIndex: -1,
        candidateKey: overrideSelection.candidateKey,
        missingSelectedCandidate: true,
        rankingPolicy: 'ordered-frames',
        collectionKind: 'ordered-frames',
      };
    }
    const frames = traj.frames ?? [];
    const rawIdx =
      overrideSelection?.index !== undefined && overrideSelection.index !== null
        ? overrideSelection.index
        : (branchCandidateIdx[nodeId] ?? Math.max(0, frames.length - 1));
    const idx = frames.length > 0 ? Math.max(0, Math.min(frames.length - 1, rawIdx)) : 0;
    const frame = frames[idx];
    const text = frame?.cifText || frame?.pdbText || traj.cifText || traj.pdbText || '';
    if (text) {
      const parsed = parseStructureText(text);
      const geom =
        parsed.nResidues > 0
          ? computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums)
          : null;
      return {
        ...empty,
        hasOutputs: true,
        has3D: parsed.nResidues > 0,
        kind: 'structure',
        structureId: traj.id,
        pdbText: parsed.nResidues > 0 ? parsed.pdbText : text,
        cifText: isMmcifText(text) ? text : null,
        caCoords: parsed.caCoords,
        resNums: parsed.resNums,
        chainIds: parsed.chainIds,
        chainLengths: parsed.diagnostics.chainLengths,
        sequence: parsed.sequence,
        length: parsed.nResidues,
        sequenceId: frame?.key ?? traj.label ?? traj.id,
        candidateKey: frame?.key ?? traj.id,
        missingSelectedCandidate: false,
        rankingPolicy: 'ordered-frames',
        branchRank: idx + 1,
        confidenceAvailable: typeof frame?.confidence === 'number' && Number.isFinite(frame.confidence),
        meanPlddt:
          typeof frame?.confidence === 'number' && Number.isFinite(frame.confidence)
            ? frame.confidence
            : null,
        collectionKind: 'ordered-frames',
        candidateCount: frames.length || traj.frameCount,
        selectedIndex: idx,
        geometry: geom,
      };
    }
  }

  const struct = values.find((a): a is StructureArtifact => a.kind === 'StructureArtifact');
  const displayStructText = struct ? struct.cifText || struct.pdbText : '';
  if (struct && displayStructText) {
    if (overrideSelection?.missing) {
      return {
        ...empty,
        hasOutputs: true,
        kind: 'structure',
        candidateCount: 1,
        selectedIndex: -1,
        candidateKey: overrideSelection.candidateKey,
        missingSelectedCandidate: true,
        rankingPolicy: 'single-structure',
      };
    }
    let resNums: number[] | null = null;
    let parsedChainIds: string[] | null = struct.chainIds ?? null;
    let parsedChainLengths: number[] | null = struct.chainLengths ?? null;
    try {
      const parsed = parseStructureText(struct.pdbText || struct.cifText || '');
      resNums = parsed.resNums;
      if (!parsedChainIds || parsedChainIds.length !== struct.length) {
        parsedChainIds = parsed.chainIds;
      }
      if (!parsedChainLengths || parsedChainLengths.length === 0) {
        parsedChainLengths = parsed.diagnostics.chainLengths;
      }
    } catch {
      resNums = null;
    }
    const hasNonProtein3D = Boolean(
      struct.diagnostics?.hasLigands || struct.diagnostics?.hasNucleicAcids,
    );
    return {
      ...empty,
      hasOutputs: true,
      has3D: struct.length > 0 || hasNonProtein3D,
      kind: 'structure',
      structureId: struct.id,
      designedPdb: null,
      pdbText: struct.pdbText || null,
      cifText: struct.cifText ?? null,
      caCoords: struct.caCoords,
      resNums,
      chainIds: parsedChainIds,
      chainLengths: parsedChainLengths,
      chainRoles: struct.chainRoles ?? null,
      sequence: struct.sequence || null,
      motifIndices: struct.motifIndices ?? null,
      benchmarkId: struct.benchmarkId ?? null,
      hasNonProtein3D,
      length: struct.length,
      sequenceId: struct.label,
      candidateKey: struct.representativeSelection?.selectedKey ?? struct.id,
      missingSelectedCandidate: false,
      rankingPolicy: 'single-structure',
      branchRank: 1,
      candidateCount: 1,
      selectedIndex: 0,
      geometry: struct.geometry,
    };
  }

  const seqSet = values.find((a): a is SequenceSet => a.kind === 'SequenceSet');
  if (seqSet && seqSet.candidates.length > 0) {
    if (overrideSelection?.missing) {
      return {
        ...empty,
        hasOutputs: true,
        kind: 'sequence',
        candidateCount: seqSet.candidates.length,
        selectedIndex: -1,
        candidateKey: overrideSelection.candidateKey,
        missingSelectedCandidate: true,
        rankingPolicy: 'sequence-score',
        collectionKind: seqSet.collectionKind ?? 'independent-candidates',
      };
    }
    const rawIdx =
      overrideSelection?.index !== undefined && overrideSelection.index !== null
        ? overrideSelection.index
        : (branchCandidateIdx[nodeId] ?? 0);
    const idx = Math.max(0, Math.min(seqSet.candidates.length - 1, rawIdx));
    const cand = seqSet.candidates[idx] ?? seqSet.candidates[0];
    const upstreamStruct =
      findStructureArtifactById(outputsByNode, cand.sourceStructureId ?? seqSet.parentStructureId) ??
      findUpstreamStructureArtifact(graph, nodeId, outputsByNode);
    const isSelfGen =
      adapterId.startsWith('rfd4-proteina') ||
      adapterId.startsWith('genie3') ||
      cand.sourceModelId.startsWith('rfd4-proteina') ||
      cand.sourceModelId.startsWith('genie3');
    const trimmedCandSeq = cand.sequence.trim();
    const trimmedRefSeq = upstreamStruct?.sequence?.trim() ?? '';
    const refSeq =
      !isSelfGen && trimmedRefSeq && trimmedRefSeq.length === trimmedCandSeq.length
        ? trimmedRefSeq
        : undefined;
    const seqAnalysis = analyzeSequence(trimmedCandSeq, refSeq);
    const candKey =
      cand.key ??
      (cand.sampleIndex !== undefined && cand.sampleIndex !== null
        ? `${cand.id}#sample-${cand.sampleIndex}`
        : cand.id);
    const mpnnScore =
      typeof cand.score === 'number' && Number.isFinite(cand.score) ? cand.score : null;
    const globalScore =
      typeof cand.globalScore === 'number' && Number.isFinite(cand.globalScore)
        ? cand.globalScore
        : null;
    return {
      ...empty,
      hasOutputs: true,
      has3D: false,
      kind: 'sequence',
      structureId: cand.sourceStructureId ?? seqSet.parentStructureId ?? null,
      chainLengths: cand.chainLengths ?? upstreamStruct?.chainLengths ?? null,
      chainRoles: upstreamStruct?.chainRoles ?? null,
      sequence: trimmedCandSeq,
      length: trimmedCandSeq.length,
      sequenceId: cand.id,
      candidateKey: candKey,
      missingSelectedCandidate: false,
      rankingPolicy: 'sequence-score',
      branchRank: idx + 1,
      confidenceAvailable: false,
      collectionKind: seqSet.collectionKind ?? 'independent-candidates',
      candidateCount: seqSet.candidates.length,
      selectedIndex: idx,
      mpnnScore,
      globalScore,
      seqRecoveryPct: seqAnalysis.recoveryPct,
      seqAnalysis,
    };
  }

  return { ...empty, hasOutputs: true };
}

function getBranchChainSegments(b: CompareBranchSummary): ChainSegment[] {
  const L = b.length ?? 0;
  if (L <= 0) return [];
  const seq = b.sequence ?? '';
  const roles = b.chainRoles ?? undefined;

  if (b.chainIds && b.chainIds.length === L) {
    const segs: ChainSegment[] = [];
    let start = 0;
    for (let i = 1; i <= L; i++) {
      if (i === L || b.chainIds[i] !== b.chainIds[start]) {
        const cid = b.chainIds[start];
        segs.push({
          chainId: cid,
          start,
          length: i - start,
          role: roles?.[cid],
          seq: seq.slice(start, i),
        });
        start = i;
      }
    }
    if (segs.length > 0) return segs;
  }

  if (b.chainLengths && b.chainLengths.length > 0 && b.chainLengths.reduce((x, y) => x + y, 0) === L) {
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let start = 0;
    return b.chainLengths.map((len, idx) => {
      const cid = alphabet[idx % alphabet.length];
      const seg: ChainSegment = {
        chainId: cid,
        start,
        length: len,
        role: roles?.[cid],
        seq: seq.slice(start, start + len),
      };
      start += len;
      return seg;
    });
  }

  return [{ chainId: 'A', start: 0, length: L, role: roles?.A, seq: seq.slice(0, L) }];
}

function sequenceIdentity(a: string, b: string): number {
  if (!a || !b || a.length !== b.length) return 0;
  let matches = 0;
  let valid = 0;
  for (let i = 0; i < a.length; i++) {
    const ca = a[i];
    const cb = b[i];
    if (ca === 'X' || cb === 'X') continue;
    valid++;
    if (ca === cb) matches++;
  }
  return valid > 0 ? matches / valid : 0;
}

function isValidKabschResult(res: KabschResult): boolean {
  if (!Number.isFinite(res.rmsd) || !Number.isFinite(res.tmScore)) return false;
  for (let i = 0; i < 9; i++) {
    if (!Number.isFinite(res.rotation[i])) return false;
  }
  for (let i = 0; i < 3; i++) {
    if (!Number.isFinite(res.translation[i])) return false;
  }
  return true;
}

function scoreChainPairCore(anchorSeg: ChainSegment, mobileSeg: ChainSegment): number {
  let score = 0;
  if (anchorSeg.role && mobileSeg.role) {
    score += anchorSeg.role === mobileSeg.role ? 1000 : -1000;
  }
  if (anchorSeg.length === mobileSeg.length) {
    score += 50;
    const id = sequenceIdentity(anchorSeg.seq, mobileSeg.seq);
    score += id * 100;
  } else {
    score -= Math.abs(anchorSeg.length - mobileSeg.length);
  }
  return score;
}

function scoreChainPair(anchorSeg: ChainSegment, mobileSeg: ChainSegment, sameIndex: boolean): number {
  return scoreChainPairCore(anchorSeg, mobileSeg) + (sameIndex ? 1 : 0);
}

/**
 * Return all chain permutations achieving the maximum intrinsic (role + length + sequence) score
 * so symmetric homodimers / homomers can break ties by lowest 3D Kabsch RMSD rather than
 * arbitrarily locking chain indices.
 */
function topChainPermutations(
  anchorSegs: ChainSegment[],
  mobileSegs: ChainSegment[],
  maxTied = 24,
): number[][] {
  const C = anchorSegs.length;
  if (C <= 6) {
    let bestScore = -Infinity;
    let bestPerms: Array<{ perm: number[]; sameCount: number }> = [];
    const used = new Uint8Array(C);
    const cur = new Array<number>(C);
    const dfs = (aIdx: number, acc: number, sameCount: number) => {
      if (aIdx === C) {
        if (acc > bestScore + 1e-6) {
          bestScore = acc;
          bestPerms = [{ perm: cur.slice(), sameCount }];
        } else if (Math.abs(acc - bestScore) <= 1e-6 && bestPerms.length < maxTied) {
          bestPerms.push({ perm: cur.slice(), sameCount });
        }
        return;
      }
      for (let mIdx = 0; mIdx < C; mIdx++) {
        if (used[mIdx]) continue;
        used[mIdx] = 1;
        cur[aIdx] = mIdx;
        dfs(
          aIdx + 1,
          acc + scoreChainPairCore(anchorSegs[aIdx], mobileSegs[mIdx]),
          sameCount + (aIdx === mIdx ? 1 : 0),
        );
        used[mIdx] = 0;
      }
    };
    dfs(0, 0, 0);
    bestPerms.sort((a, b) => b.sameCount - a.sameCount);
    return bestPerms.map((p) => p.perm);
  }

  // Greedy fallback for very large complexes (>6 chains)
  const used = new Uint8Array(C);
  const perm = new Array<number>(C);
  for (let aIdx = 0; aIdx < C; aIdx++) {
    let bestM = 0;
    let bestS = -Infinity;
    for (let mIdx = 0; mIdx < C; mIdx++) {
      if (used[mIdx]) continue;
      const s = scoreChainPair(anchorSegs[aIdx], mobileSegs[mIdx], aIdx === mIdx);
      if (s > bestS) {
        bestS = s;
        bestM = mIdx;
      }
    }
    used[bestM] = 1;
    perm[aIdx] = bestM;
  }
  return [perm];
}

interface BranchAlignOutcome {
  result: KabschResult;
  alignedResidues: number;
  isSubstructure: boolean;
}

/**
 * Align `mobile` onto `anchor`, matching multi-chain complexes by `chainRoles` / sequence
 * similarity / chain length (#9) with a Kabsch RMSD tie-breaker for symmetric equal-sequence
 * homodimers/homomers so opposite chain ordering superimposes cleanly.
 */
function alignBranchPair(
  mobile: CompareBranchSummary,
  anchor: CompareBranchSummary,
  hasSharedTemplate: boolean,
): BranchAlignOutcome {
  const mLen = mobile.length ?? 0;
  const aLen = anchor.length ?? 0;
  const mobileCoords = mobile.caCoords!;
  const anchorCoords = anchor.caCoords!;

  const mobileSegs = getBranchChainSegments(mobile);
  const anchorSegs = getBranchChainSegments(anchor);

  // Case 1: Both are multi-chain with matching chain count and equal per-matched-chain lengths
  if (mobileSegs.length > 1 && mobileSegs.length === anchorSegs.length) {
    const candidatePerms = topChainPermutations(anchorSegs, mobileSegs);
    const matchingLenPerms =
      mLen === aLen
        ? candidatePerms.filter((perm) =>
            anchorSegs.every((aSeg, aIdx) => mobileSegs[perm[aIdx]].length === aSeg.length),
          )
        : [];

    if (matchingLenPerms.length > 0) {
      let bestResult: KabschResult | null = null;
      const reorderedMobile = new Float32Array(aLen * 3);
      for (const perm of matchingLenPerms) {
        for (let aIdx = 0; aIdx < anchorSegs.length; aIdx++) {
          const aSeg = anchorSegs[aIdx];
          const mSeg = mobileSegs[perm[aIdx]];
          reorderedMobile.set(
            mobileCoords.subarray(mSeg.start * 3, (mSeg.start + mSeg.length) * 3),
            aSeg.start * 3,
          );
        }
        const cand = kabschAlign(reorderedMobile, anchorCoords);
        if (!bestResult) {
          bestResult = cand;
        } else if (
          Number.isFinite(cand.rmsd) &&
          (!Number.isFinite(bestResult.rmsd) || cand.rmsd < bestResult.rmsd - 1e-6)
        ) {
          bestResult = cand;
        }
      }
      return {
        result: bestResult!,
        alignedResidues: aLen,
        isSubstructure: false,
      };
    }

    const perm = candidatePerms[0];

    // Different binder lengths on a shared multi-chain target: align on matched equal-length target chain(s)
    const matchedPairs: Array<{ aSeg: ChainSegment; mSeg: ChainSegment }> = [];
    for (let aIdx = 0; aIdx < anchorSegs.length; aIdx++) {
      const aSeg = anchorSegs[aIdx];
      const mSeg = mobileSegs[perm[aIdx]];
      if (aSeg.length !== mSeg.length || aSeg.length < 3) continue;
      const bothTarget = aSeg.role === 'target' && mSeg.role === 'target';
      const highSeqId = aSeg.length >= 10 && sequenceIdentity(aSeg.seq, mSeg.seq) >= 0.8;
      if (bothTarget || highSeqId) {
        matchedPairs.push({ aSeg, mSeg });
      }
    }
    if (matchedPairs.length > 0) {
      const totalMatched = matchedPairs.reduce((sum, p) => sum + p.aSeg.length, 0);
      const subMob = new Float32Array(totalMatched * 3);
      const subTgt = new Float32Array(totalMatched * 3);
      let cursor = 0;
      for (const { aSeg, mSeg } of matchedPairs) {
        subMob.set(
          mobileCoords.subarray(mSeg.start * 3, (mSeg.start + mSeg.length) * 3),
          cursor * 3,
        );
        subTgt.set(
          anchorCoords.subarray(aSeg.start * 3, (aSeg.start + aSeg.length) * 3),
          cursor * 3,
        );
        cursor += aSeg.length;
      }
      return {
        result: kabschAlign(subMob, subTgt),
        alignedResidues: totalMatched,
        isSubstructure: true,
      };
    }
  }

  // Case 2: Equal-length single-chain (or non-bijective multichain) direct Kabsch
  if (mLen === aLen) {
    return {
      result: kabschAlign(mobileCoords, anchorCoords),
      alignedResidues: aLen,
      isSubstructure: false,
    };
  }

  // Case 3: Different lengths -> substructure / motif Kabsch
  if (mLen <= aLen) {
    const knownIndices =
      hasSharedTemplate && anchor.motifIndices && anchor.motifIndices.length === mLen
        ? anchor.motifIndices
        : null;
    const sub = kabschAlignSubstructure(mobileCoords, anchorCoords, mobile.resNums, {
      shortChainIds: mobile.chainIds,
      longChainIds: anchor.chainIds,
      knownIndices,
    });
    return { result: sub, alignedResidues: sub.alignedResidues, isSubstructure: true };
  }

  // mobile is longer than anchor: invert transform from aligning anchor -> mobile
  const knownIndices =
    hasSharedTemplate && mobile.motifIndices && mobile.motifIndices.length === aLen
      ? mobile.motifIndices
      : null;
  const rev = kabschAlignSubstructure(anchorCoords, mobileCoords, anchor.resNums, {
    shortChainIds: anchor.chainIds,
    longChainIds: mobile.chainIds,
    knownIndices,
  });
  const R = rev.rotation;
  const t = rev.translation;
  // R_inv = R^T, t_inv = -R^T * t
  const Rinv = new Float64Array([
    R[0], R[3], R[6],
    R[1], R[4], R[7],
    R[2], R[5], R[8],
  ]);
  const tinv = new Float64Array([
    -(Rinv[0] * t[0] + Rinv[1] * t[1] + Rinv[2] * t[2]),
    -(Rinv[3] * t[0] + Rinv[4] * t[1] + Rinv[5] * t[2]),
    -(Rinv[6] * t[0] + Rinv[7] * t[1] + Rinv[8] * t[2]),
  ]);
  const transform4x4 = new Float64Array([
    Rinv[0], Rinv[3], Rinv[6], 0,
    Rinv[1], Rinv[4], Rinv[7], 0,
    Rinv[2], Rinv[5], Rinv[8], 0,
    tinv[0], tinv[1], tinv[2], 1,
  ]);
  return {
    result: {
      rmsd: rev.rmsd,
      tmScore: rev.tmScore,
      rotation: Rinv,
      translation: tinv,
      transform4x4,
      aligned: mobileCoords,
    },
    alignedResidues: rev.alignedResidues,
    isSubstructure: true,
  };
}

function branchesShareUpstreamTemplate(
  a: CompareBranchSummary,
  b: CompareBranchSummary,
  ancestorsByNode: Map<string, Set<string>>,
): boolean {
  if (a.structureId && b.structureId && a.structureId === b.structureId) return true;
  if (a.benchmarkId && b.benchmarkId && a.benchmarkId === b.benchmarkId) return true;
  const ancA = ancestorsByNode.get(a.nodeId);
  const ancB = ancestorsByNode.get(b.nodeId);
  if (!ancA || !ancB) return false;
  if (ancA.has(b.nodeId) || ancB.has(a.nodeId)) return true;
  for (const id of ancA) {
    if (ancB.has(id)) return true;
  }
  return false;
}

const MULTI_DESIGN_PALETTE: ReadonlyArray<{ hex: number; css: string }> = [
  { hex: 0x3b82f6, css: '#3b82f6' }, // blue
  { hex: 0x10b981, css: '#10b981' }, // emerald
  { hex: 0xf59e0b, css: '#f59e0b' }, // amber
  { hex: 0xa855f7, css: '#a855f7' }, // purple
  { hex: 0xef4444, css: '#ef4444' }, // red
  { hex: 0x06b6d4, css: '#06b6d4' }, // cyan
  { hex: 0xec4899, css: '#ec4899' }, // pink
  { hex: 0x84cc16, css: '#84cc16' }, // lime
];

/**
 * Build multi-model `ViewerModelSpec[]` for rendering all generated backbones from a
 * `StructureArtifact.collection` simultaneously in a single Mol* canvas — either arranged
 * in a centered 2D X-Y gallery (`grid`) or centered/superposed together (`overlay`).
 */
export function buildMultiDesignViewerModels(
  items: readonly StructureArtifact[],
  mode: 'grid' | 'overlay' = 'grid',
  hiddenIds: ReadonlySet<string> = new Set(),
): ViewerModelSpec[] {
  if (items.length === 0) return [];
  const cols = mode === 'grid' ? Math.ceil(Math.sqrt(items.length)) : 1;
  const rows = mode === 'grid' ? Math.ceil(items.length / cols) : 1;
  let maxRg = 12;
  for (const item of items) {
    if (item.geometry?.rg && Number.isFinite(item.geometry.rg)) {
      maxRg = Math.max(maxRg, item.geometry.rg);
    }
  }
  const spacing = maxRg * 3.1 + 12;

  const anchor = items[0];
  const anchorText = anchor.cifText || anchor.pdbText;
  const anchorCenter =
    anchor.caCoords && anchor.length > 0
      ? centroidOf(anchor.caCoords)
      : centroidOfStructureText(anchorText);

  const out: ViewerModelSpec[] = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const rawText = item.cifText || item.pdbText;
    if (!rawText) continue;
    const pal = MULTI_DESIGN_PALETTE[i % MULTI_DESIGN_PALETTE.length];
    const [cx, cy, cz] =
      item.caCoords && item.length > 0
        ? centroidOf(item.caCoords)
        : centroidOfStructureText(rawText);

    let transformedText: string;
    if (mode === 'grid') {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const tx = (col - (cols - 1) / 2) * spacing;
      const ty = ((rows - 1) / 2 - row) * spacing;
      const shift = new Float64Array([tx - cx, ty - cy, -cz]);
      transformedText = applyTransformToStructureText(rawText, IDENTITY_ROT, shift);
    } else if (
      i > 0 &&
      anchor.caCoords &&
      item.caCoords &&
      anchor.length === item.length &&
      item.length >= 3
    ) {
      // Kabsch-align equal-length backbones onto Design 1 centered at origin
      try {
        const centeredAnchor = new Float32Array(anchor.caCoords.length);
        for (let k = 0; k < anchor.length; k++) {
          centeredAnchor[k * 3 + 0] = anchor.caCoords[k * 3 + 0] - anchorCenter[0];
          centeredAnchor[k * 3 + 1] = anchor.caCoords[k * 3 + 1] - anchorCenter[1];
          centeredAnchor[k * 3 + 2] = anchor.caCoords[k * 3 + 2] - anchorCenter[2];
        }
        const fit = kabschAlign(item.caCoords, centeredAnchor);
        if (isValidKabschResult(fit)) {
          transformedText = applyTransformToStructureText(rawText, fit.rotation, fit.translation);
        } else {
          transformedText = applyTransformToStructureText(
            rawText,
            IDENTITY_ROT,
            new Float64Array([-cx, -cy, -cz]),
          );
        }
      } catch {
        transformedText = applyTransformToStructureText(
          rawText,
          IDENTITY_ROT,
          new Float64Array([-cx, -cy, -cz]),
        );
      }
    } else {
      const shift = new Float64Array([-cx, -cy, -cz]);
      transformedText = applyTransformToStructureText(rawText, IDENTITY_ROT, shift);
    }

    const id = `design-${item.designKey ?? item.id ?? i}`;
    const rgTxt = item.geometry?.rg ? `Rg ${item.geometry.rg.toFixed(1)} Å` : '';
    out.push({
      id,
      nodeId: null,
      badge: `D${i + 1}`,
      label: `Design ${i + 1}`,
      pdbText: transformedText,
      colorHex: pal.hex,
      colorCss: pal.css,
      alpha: mode === 'overlay' ? 0.85 : 1.0,
      colorByPlddt: false,
      visible: !hiddenIds.has(id),
      metricSummary: rgTxt ? `${item.length} aa · ${rgTxt}` : `${item.length} aa`,
    });
  }
  return out;
}

export function buildComparisonState(
  graph: WorkflowGraph,
  selectedNodeIds: readonly string[],
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
  opts: {
    layout: 'overlay' | 'side-by-side';
    colorMode: 'by-model' | 'plddt';
    alignKabsch?: boolean;
    branchCandidateIdx?: Record<string, number>;
    hiddenModelIds?: ReadonlySet<string>;
    pairingMode?: ComparisonPairingMode;
    selectedCandidateKey?: string | null;
  },
): ComparisonState {
  const capped = selectedNodeIds.slice(0, COMPARE_PALETTE.length);
  const candIdx = opts.branchCandidateIdx ?? {};
  const hidden = opts.hiddenModelIds ?? new Set<string>();
  const wantKabsch = opts.alignKabsch !== false;

  const branchCollections = capped.map((id) => extractBranchCandidateItems(graph, id, outputsByNode));
  const { hasSharedKey, hasSharedSequence } = branchesShareCandidateIdentity(branchCollections);
  const hasSharedIdentity = hasSharedKey || hasSharedSequence;
  const pairingMode: ComparisonPairingMode =
    opts.pairingMode ?? (hasSharedIdentity ? 'by-candidate' : 'branch-best');
  const pairingLabel =
    pairingMode === 'by-candidate'
      ? 'Same candidate (by identity)'
      : 'Branch best (independently ranked winners)';

  // Use sequence fallback only when branches do not share explicit candidate keys
  // (e.g., comparing branches with different per-branch auto-labels for the same sequence).
  const joined = joinKeyedCollections(branchCollections, {
    matchBySequenceFallback: !hasSharedKey && hasSharedSequence,
  });

  let selectedCandidateKey: string | null = null;
  if (pairingMode === 'by-candidate') {
    if (opts.selectedCandidateKey && joined.keys.includes(opts.selectedCandidateKey)) {
      selectedCandidateKey = opts.selectedCandidateKey;
    } else if (opts.selectedCandidateKey) {
      // Caller explicitly requested a key (even if missing in some or all branches)
      selectedCandidateKey = opts.selectedCandidateKey;
    } else {
      const firstBranch = branchCollections[0];
      const firstExplicitIdx = firstBranch ? candIdx[firstBranch.branchId] : undefined;
      if (
        firstBranch &&
        firstExplicitIdx !== undefined &&
        firstExplicitIdx >= 0 &&
        firstExplicitIdx < firstBranch.items.length
      ) {
        selectedCandidateKey = firstBranch.items[firstExplicitIdx].key;
      } else {
        selectedCandidateKey = joined.keys[0] ?? null;
      }
    }
  }

  const selectedRow =
    selectedCandidateKey !== null
      ? joined.rows.find((r) => r.key === selectedCandidateKey) ?? null
      : null;

  const branches = capped.map((id, idx) => {
    if (pairingMode === 'by-candidate' && selectedCandidateKey !== null) {
      const entry = selectedRow?.byBranch[id];
      if (entry?.present && entry.item) {
        return extractBranch(graph, id, idx, outputsByNode, candIdx, {
          index: entry.item.value.index,
          candidateKey: selectedCandidateKey,
          missing: false,
        });
      }
      return extractBranch(graph, id, idx, outputsByNode, candIdx, {
        index: null,
        candidateKey: selectedCandidateKey,
        missing: true,
      });
    }
    return extractBranch(graph, id, idx, outputsByNode, candIdx);
  });

  // All branches with renderable 3D (protein >= 3 aa OR pure ligand/DNA mmCIF/PDB)
  const renderable = branches.filter(
    (b) =>
      b.has3D &&
      Boolean(b.cifText || b.pdbText) &&
      ((b.length ?? 0) >= 3 || Boolean(b.hasNonProtein3D)),
  );
  // Branches eligible for Cα Kabsch superposition & pairwise RMSD
  const alignable = renderable.filter((b) => b.caCoords && (b.length ?? 0) >= 3);

  const ancestorsByNode = new Map<string, Set<string>>();
  for (const b of branches) {
    ancestorsByNode.set(b.nodeId, getAncestorNodeIds(graph, b.nodeId));
  }

  // Check if 2+ scorer branches share the exact same upstream designed backbone
  const metricBranches = alignable.filter((b) => b.kind === 'metrics' && b.structureId && b.designedPdb);
  const sharedStructIds = new Set(metricBranches.map((b) => b.structureId));
  const hasSharedDesigned =
    metricBranches.length >= 2 &&
    metricBranches.length === renderable.length &&
    sharedStructIds.size === 1;

  // Check if all alignable 3D models share the same residue count
  const lengths = new Set(alignable.map((b) => b.length));
  const sameLength = alignable.length >= 2 && lengths.size === 1;
  const forcedSideBySide = false;
  const effectiveLayout: 'overlay' | 'side-by-side' =
    opts.layout === 'side-by-side' ? 'side-by-side' : 'overlay';

  // Pick anchor frame: for same-length use model 0; for different-length use the longest model
  // so shorter motifs/targets snap onto the larger scaffold/complex cleanly.
  let anchorBranch: CompareBranchSummary | null = alignable[0] ?? null;
  if (!sameLength && alignable.length >= 2) {
    let maxLen = alignable[0].length ?? 0;
    for (let i = 1; i < alignable.length; i++) {
      if ((alignable[i].length ?? 0) > maxLen) {
        maxLen = alignable[i].length ?? 0;
        anchorBranch = alignable[i];
      }
    }
  }

  // Compute pairwise agreements across all alignable pairs (A↔B primary, plus A↔C, B↔C, etc. for 3–4 blocks)
  // Cache alignments targeting `anchorBranch` so the overlay loop reuses them without duplicating Kabsch work.
  const allPairwise: PairwiseAgreement[] = [];
  const cachedAlignToAnchor = new Map<string, BranchAlignOutcome>();
  for (let i = 0; i < alignable.length; i++) {
    for (let j = i + 1; j < alignable.length; j++) {
      const a = alignable[i];
      const b = alignable[j];
      const sharedTpl = branchesShareUpstreamTemplate(a, b, ancestorsByNode);
      try {
        // Align directly onto `anchorBranch` whenever either endpoint is the anchor so the
        // exact (rotation, translation) transform is ready for reuse in the overlay loop.
        const bIsAnchor = anchorBranch !== null && b.nodeId === anchorBranch.nodeId;
        const aIsAnchor = anchorBranch !== null && a.nodeId === anchorBranch.nodeId;
        const outcome = bIsAnchor
          ? alignBranchPair(a, b, sharedTpl)
          : alignBranchPair(b, a, sharedTpl);
        if (!isValidKabschResult(outcome.result)) continue;
        if (aIsAnchor) {
          cachedAlignToAnchor.set(b.nodeId, outcome);
        } else if (bIsAnchor) {
          cachedAlignToAnchor.set(a.nodeId, outcome);
        }
        allPairwise.push({
          labelA: a.label,
          labelB: b.label,
          badgeA: a.badge,
          badgeB: b.badge,
          rmsd: outcome.result.rmsd,
          tmScore: outcome.result.tmScore,
          alignedResidues: outcome.alignedResidues,
          isSubstructure: outcome.isSubstructure,
          hasSharedTemplate: sharedTpl,
          alignmentLabel: outcome.isSubstructure
            ? sharedTpl
              ? 'Motif Kabsch'
              : 'Substructure Kabsch'
            : 'Inter-model Kabsch',
        });
      } catch {
        /* skip degenerate pair */
      }
    }
  }
  const pairwise = allPairwise[0] ?? null;

  const models: ViewerModelSpec[] = [];
  let sharedReferenceLabel: string | null = null;
  let kabschAligned = false;
  const isPreAlignedSharedBackbone = hasSharedDesigned;

  const formatMetricBits = (b: CompareBranchSummary): string | undefined => {
    const metricBits: string[] = [];
    if (b.scRmsd !== null) metricBits.push(`${b.scRmsd.toFixed(2)} Å`);
    if (b.meanPlddt !== null) metricBits.push(`pLDDT ${b.meanPlddt.toFixed(0)}`);
    if (b.iptm !== null) metricBits.push(`ipTM ${b.iptm.toFixed(2)}`);
    if (b.ipsae !== null) metricBits.push(`ipSAE ${b.ipsae.toFixed(2)}`);
    if (metricBits.length > 0) return metricBits.join(' · ');
    if (b.length && b.length > 0) return `${b.length} aa`;
    if (b.hasNonProtein3D) return 'ligand/DNA';
    return undefined;
  };

  if (effectiveLayout === 'overlay') {
    // 1. Optional ghosted shared designed backbone (only claim kabschAligned when wantKabsch is enabled)
    if (hasSharedDesigned && metricBranches[0].designedPdb) {
      sharedReferenceLabel = 'Designed Backbone';
      if (wantKabsch) {
        kabschAligned = true;
      }
      models.push({
        id: 'ref-designed',
        nodeId: null,
        label: 'Designed',
        pdbText: metricBranches[0].designedPdb,
        colorHex: REFERENCE_BACKBONE_COLOR.hex,
        colorCss: REFERENCE_BACKBONE_COLOR.css,
        alpha: 0.35,
        isReference: true,
        visible: !hidden.has('ref-designed'),
      });
    }

    for (const b of renderable) {
      let structText = b.cifText || b.pdbText!;
      const canAlign = Boolean(b.caCoords && (b.length ?? 0) >= 3);
      // If Kabsch align is enabled and NOT already in a shared pre-aligned frame, align non-anchor models onto anchor
      if (
        wantKabsch &&
        !hasSharedDesigned &&
        canAlign &&
        anchorBranch &&
        b.nodeId !== anchorBranch.nodeId
      ) {
        try {
          const outcome =
            cachedAlignToAnchor.get(b.nodeId) ??
            alignBranchPair(
              b,
              anchorBranch,
              branchesShareUpstreamTemplate(b, anchorBranch, ancestorsByNode),
            );
          if (isValidKabschResult(outcome.result)) {
            structText = applyTransformToStructureText(
              structText,
              outcome.result.rotation,
              outcome.result.translation,
            );
            kabschAligned = true;
          }
        } catch {
          /* keep original coordinates on degenerate alignment */
        }
      }
      models.push({
        id: `branch-${b.nodeId}`,
        nodeId: b.nodeId,
        badge: b.badge,
        label: b.label,
        pdbText: structText,
        colorHex: b.colorHex,
        colorCss: b.colorCss,
        alpha: 1.0,
        colorByPlddt: opts.colorMode === 'plddt' && (b.kind === 'metrics' || b.kind === 'fold'),
        visible: !hidden.has(`branch-${b.nodeId}`),
        metricSummary: formatMetricBits(b),
      });
    }
  } else {
    // Side-by-side inside a single Mol* scene via centroid-centered +X translation
    let xCursor = 0;
    let prevRg = 0;
    for (let i = 0; i < renderable.length; i++) {
      const b = renderable[i];
      const rawText = b.cifText || b.pdbText!;
      const rg = Math.max(8, b.geometry?.rg ?? 14);
      if (i > 0) {
        xCursor += (prevRg + rg) * 1.65 + 10;
      }
      prevRg = rg;

      const [cx, cy, cz] =
        b.caCoords && (b.length ?? 0) > 0
          ? centroidOf(b.caCoords)
          : centroidOfStructureText(rawText);
      const shift = new Float64Array([xCursor - cx, -cy, -cz]);
      const shiftedText = applyTransformToStructureText(rawText, IDENTITY_ROT, shift);

      models.push({
        id: `branch-${b.nodeId}`,
        nodeId: b.nodeId,
        badge: b.badge,
        label: b.label,
        pdbText: shiftedText,
        colorHex: b.colorHex,
        colorCss: b.colorCss,
        alpha: 1.0,
        colorByPlddt: opts.colorMode === 'plddt' && (b.kind === 'metrics' || b.kind === 'fold'),
        visible: !hidden.has(`branch-${b.nodeId}`),
        metricSummary: formatMetricBits(b),
      });
    }
  }

  return {
    branches,
    models,
    effectiveLayout,
    forcedSideBySide,
    kabschAligned,
    isPreAlignedSharedBackbone,
    sharedReferenceLabel,
    pairwise,
    allPairwise,
    pairingMode,
    pairingLabel,
    selectedCandidateKey,
    candidateKeys: joined.keys,
    pairedCandidates: joined.rows,
  };
}

export interface SequencePairDiff {
  badgeA: string;
  badgeB: string;
  labelA: string;
  labelB: string;
  seqA: string;
  seqB: string;
  comparableLength: number;
  identicalCount: number;
  identityPct: number;
  mismatchIndices: number[];
  lengthDiffers: boolean;
}

export function computePairwiseSequenceDiff(
  branches: readonly CompareBranchSummary[],
): SequencePairDiff[] {
  const withSeq = branches.filter((b) => Boolean(b.sequence && b.sequence.trim().length > 0));
  const diffs: SequencePairDiff[] = [];
  for (let i = 0; i < withSeq.length; i++) {
    for (let j = i + 1; j < withSeq.length; j++) {
      const a = withSeq[i];
      const b = withSeq[j];
      const seqA = a.sequence!.trim().toUpperCase();
      const seqB = b.sequence!.trim().toUpperCase();
      const comparableLength = Math.min(seqA.length, seqB.length);
      if (comparableLength === 0) continue;
      const mismatchIndices: number[] = [];
      let identicalCount = 0;
      for (let k = 0; k < comparableLength; k++) {
        if (seqA[k] === seqB[k]) identicalCount++;
        else mismatchIndices.push(k);
      }
      diffs.push({
        badgeA: a.badge,
        badgeB: b.badge,
        labelA: a.label,
        labelB: b.label,
        seqA,
        seqB,
        comparableLength,
        identicalCount,
        identityPct: (identicalCount / comparableLength) * 100,
        mismatchIndices,
        lengthDiffers: seqA.length !== seqB.length,
      });
    }
  }
  return diffs;
}
