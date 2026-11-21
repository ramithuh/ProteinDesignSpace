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
import { ADAPTER_REGISTRY } from './registry';
import type {
  CanonicalArtifact,
  DesignabilityMetrics,
  FoldedStructure,
  StructureArtifact,
  WorkflowGraph,
} from './types';

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
  kind: 'metrics' | 'fold' | 'structure' | 'none';
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
  candidateCount: number;
  selectedIndex: number;
  scRmsd: number | null;
  scTm: number | null;
  meanPlddt: number | null;
  ptm: number | null;
  iptm: number | null;
  ipae: number | null;
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
}

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
  if (L <= 0 || pae.length !== L * L) return null;
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
      const v = pae[i * L + j];
      if (Number.isFinite(v)) {
        sum += v;
        count++;
      }
    }
  }
  return count > 0 ? sum / count : null;
}

function extractBranch(
  graph: WorkflowGraph,
  nodeId: string,
  slotIdx: number,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
  branchCandidateIdx: Record<string, number>,
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
    candidateCount: 0,
    selectedIndex: 0,
    scRmsd: null,
    scTm: null,
    meanPlddt: null,
    ptm: null,
    iptm: null,
    ipae: null,
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
  };

  const outs = outputsByNode[nodeId];
  if (!outs) return empty;
  const values = Object.values(outs);
  if (values.length === 0) return empty;

  const metrics = values.find((a): a is DesignabilityMetrics => a.kind === 'DesignabilityMetrics');
  if (metrics && metrics.rows.length > 0) {
    const idx = Math.max(0, Math.min(metrics.rows.length - 1, branchCandidateIdx[nodeId] ?? 0));
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
      candidateCount: metrics.rows.length,
      selectedIndex: idx,
      scRmsd: row.scRmsd,
      scTm: row.scTm,
      meanPlddt: row.meanPlddt,
      ptm: row.ptm,
      iptm: row.iptm ?? null,
      ipae: row.ipae ?? null,
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
    const idx = Math.max(0, Math.min(folds.folds.length - 1, branchCandidateIdx[nodeId] ?? 0));
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
    const ipae = f.pae && chainLengths.length > 1 ? computeIpaeFromMatrix(f.pae, chainLengths) : null;
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
      chainIds: f.chainIds ?? parsed.chainIds,
      chainLengths,
      chainRoles: upstreamStruct?.chainRoles ?? null,
      sequence: f.sequence || parsed.sequence,
      motifIndices: upstreamStruct?.motifIndices ?? null,
      benchmarkId: upstreamStruct?.benchmarkId ?? null,
      length: parsed.nResidues,
      sequenceId: f.sequenceId,
      candidateCount: folds.folds.length,
      selectedIndex: idx,
      meanPlddt: f.meanPlddt,
      ptm: f.ptm,
      iptm: f.iptm ?? null,
      ipae,
      foldElapsedMs: f.elapsedMs,
      geometry: geom,
      pae: f.pae ?? null,
    };
  }

  const struct = values.find((a): a is StructureArtifact => a.kind === 'StructureArtifact');
  const displayStructText = struct ? struct.cifText || struct.pdbText : '';
  if (struct && displayStructText) {
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
      candidateCount: 1,
      selectedIndex: 0,
      geometry: struct.geometry,
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

function scoreChainPair(anchorSeg: ChainSegment, mobileSeg: ChainSegment, sameIndex: boolean): number {
  let score = sameIndex ? 1 : 0;
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

function bestChainPermutation(anchorSegs: ChainSegment[], mobileSegs: ChainSegment[]): number[] {
  const C = anchorSegs.length;
  if (C <= 6) {
    let bestScore = -Infinity;
    let bestPerm = Array.from({ length: C }, (_, i) => i);
    const used = new Uint8Array(C);
    const cur = new Array<number>(C);
    const dfs = (aIdx: number, acc: number) => {
      if (aIdx === C) {
        if (acc > bestScore) {
          bestScore = acc;
          bestPerm = cur.slice();
        }
        return;
      }
      for (let mIdx = 0; mIdx < C; mIdx++) {
        if (used[mIdx]) continue;
        used[mIdx] = 1;
        cur[aIdx] = mIdx;
        dfs(aIdx + 1, acc + scoreChainPair(anchorSegs[aIdx], mobileSegs[mIdx], aIdx === mIdx));
        used[mIdx] = 0;
      }
    };
    dfs(0, 0);
    return bestPerm;
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
  return perm;
}

interface BranchAlignOutcome {
  result: KabschResult;
  alignedResidues: number;
  isSubstructure: boolean;
}

/**
 * Align `mobile` onto `anchor`, matching multi-chain complexes by `chainRoles` / sequence
 * similarity / chain length (#9) so opposite chain ordering (e.g. Genie 3 [Binder=A, Target=B]
 * vs RFD4 [Target=A, Binder=B]) superimposes binder-to-binder and target-to-target.
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
    const perm = bestChainPermutation(anchorSegs, mobileSegs);
    const allLengthsMatch = anchorSegs.every((aSeg, aIdx) => mobileSegs[perm[aIdx]].length === aSeg.length);
    if (allLengthsMatch && mLen === aLen) {
      const reorderedMobile = new Float32Array(aLen * 3);
      for (let aIdx = 0; aIdx < anchorSegs.length; aIdx++) {
        const aSeg = anchorSegs[aIdx];
        const mSeg = mobileSegs[perm[aIdx]];
        reorderedMobile.set(
          mobileCoords.subarray(mSeg.start * 3, (mSeg.start + mSeg.length) * 3),
          aSeg.start * 3,
        );
      }
      return {
        result: kabschAlign(reorderedMobile, anchorCoords),
        alignedResidues: aLen,
        isSubstructure: false,
      };
    }

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
  },
): ComparisonState {
  const capped = selectedNodeIds.slice(0, COMPARE_PALETTE.length);
  const candIdx = opts.branchCandidateIdx ?? {};
  const hidden = opts.hiddenModelIds ?? new Set<string>();
  const wantKabsch = opts.alignKabsch !== false;

  const branches = capped.map((id, idx) =>
    extractBranch(graph, id, idx, outputsByNode, candIdx),
  );

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
  const allPairwise: PairwiseAgreement[] = [];
  for (let i = 0; i < alignable.length; i++) {
    for (let j = i + 1; j < alignable.length; j++) {
      const a = alignable[i];
      const b = alignable[j];
      const sharedTpl = branchesShareUpstreamTemplate(a, b, ancestorsByNode);
      try {
        const outcome = alignBranchPair(b, a, sharedTpl);
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
          const sharedTpl = branchesShareUpstreamTemplate(b, anchorBranch, ancestorsByNode);
          const outcome = alignBranchPair(b, anchorBranch, sharedTpl);
          structText = applyTransformToStructureText(
            structText,
            outcome.result.rotation,
            outcome.result.translation,
          );
          kabschAligned = true;
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
  };
}
