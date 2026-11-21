// Pure-TS multi-block 3D comparison & structural alignment helper.
// Builds color-coded Mol* multi-model overlays (Kabsch-aligned when same length or
// sharing a designed backbone, or clean +X side-by-side offset when lengths differ
// or Side-by-Side layout is toggled) + pairwise inter-model Cα RMSD/TM-score.

import {
  applyTransformToPdb,
  computeGeometryStats,
  kabschAlign,
  kabschAlignSubstructure,
  type GeometryStats,
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
  caCoords: Float32Array | null;
  resNums: number[] | null;
  length: number | null;
  sequenceId: string | null;
  candidateCount: number;
  selectedIndex: number;
  scRmsd: number | null;
  scTm: number | null;
  meanPlddt: number | null;
  ptm: number | null;
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
}

export interface ComparisonState {
  branches: CompareBranchSummary[];
  models: ViewerModelSpec[];
  effectiveLayout: 'overlay' | 'side-by-side';
  forcedSideBySide: boolean;
  kabschAligned: boolean;
  sharedReferenceLabel: string | null;
  pairwise: PairwiseAgreement | null;
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

const IDENTITY_ROT = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

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
    caCoords: null,
    resNums: null,
    length: null,
    sequenceId: null,
    candidateCount: 0,
    selectedIndex: 0,
    scRmsd: null,
    scTm: null,
    meanPlddt: null,
    ptm: null,
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
    return {
      ...empty,
      hasOutputs: true,
      has3D: parsed.nResidues > 0,
      kind: 'metrics',
      structureId: row.structureId,
      designedPdb: row.designedPdb,
      pdbText: row.alignedFoldPdb,
      caCoords: parsed.caCoords,
      resNums: parsed.resNums,
      length: parsed.nResidues,
      sequenceId: row.sequenceId,
      candidateCount: metrics.rows.length,
      selectedIndex: idx,
      scRmsd: row.scRmsd,
      scTm: row.scTm,
      meanPlddt: row.meanPlddt,
      ptm: row.ptm,
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
    const geom = parsed.nResidues > 0 ? computeGeometryStats(parsed.caCoords, parsed.chainIds) : null;
    return {
      ...empty,
      hasOutputs: true,
      has3D: parsed.nResidues > 0,
      kind: 'fold',
      structureId: null,
      designedPdb: null,
      pdbText: f.pdbText,
      caCoords: parsed.caCoords,
      resNums: parsed.resNums,
      length: parsed.nResidues,
      sequenceId: f.sequenceId,
      candidateCount: folds.folds.length,
      selectedIndex: idx,
      meanPlddt: f.meanPlddt,
      ptm: f.ptm,
      foldElapsedMs: f.elapsedMs,
      geometry: geom,
      pae: f.pae ?? null,
    };
  }

  const struct = values.find((a): a is StructureArtifact => a.kind === 'StructureArtifact');
  if (struct && struct.pdbText) {
    let resNums: number[] | null = null;
    try {
      resNums = parseStructureText(struct.pdbText).resNums;
    } catch {
      resNums = null;
    }
    return {
      ...empty,
      hasOutputs: true,
      has3D: struct.length > 0,
      kind: 'structure',
      structureId: struct.id,
      designedPdb: null,
      pdbText: struct.pdbText,
      caCoords: struct.caCoords,
      resNums,
      length: struct.length,
      sequenceId: struct.label,
      candidateCount: 1,
      selectedIndex: 0,
      geometry: struct.geometry,
    };
  }

  return { ...empty, hasOutputs: true };
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

  const with3d = branches.filter((b) => b.has3D && b.pdbText && b.caCoords && (b.length ?? 0) >= 3);

  // Check if 2+ scorer branches share the exact same upstream designed backbone
  const metricBranches = with3d.filter((b) => b.kind === 'metrics' && b.structureId && b.designedPdb);
  const sharedStructIds = new Set(metricBranches.map((b) => b.structureId));
  const hasSharedDesigned =
    metricBranches.length >= 2 &&
    metricBranches.length === with3d.length &&
    sharedStructIds.size === 1;

  // Check if all 3D models share the same residue count
  const lengths = new Set(with3d.map((b) => b.length));
  const sameLength = with3d.length >= 2 && lengths.size === 1;
  // Substructure Kabsch alignment now supports overlaying different-length models (e.g. motif vs scaffold)!
  const forcedSideBySide = false;
  const effectiveLayout: 'overlay' | 'side-by-side' =
    opts.layout === 'side-by-side' ? 'side-by-side' : 'overlay';

  // Pick anchor frame: for same-length use model 0; for different-length use the longest model
  // so shorter motifs/targets snap onto the larger scaffold/complex cleanly.
  let anchorIdx = 0;
  if (!sameLength && with3d.length >= 2) {
    let maxLen = with3d[0].length ?? 0;
    for (let i = 1; i < with3d.length; i++) {
      if ((with3d[i].length ?? 0) > maxLen) {
        maxLen = with3d[i].length ?? 0;
        anchorIdx = i;
      }
    }
  }

  // Compute pairwise agreement between first 2 3D models (full Kabsch if same length, substructure if different)
  let pairwise: PairwiseAgreement | null = null;
  if (with3d.length >= 2) {
    const a = with3d[0];
    const b = with3d[1];
    try {
      if (sameLength) {
        const align = kabschAlign(b.caCoords!, a.caCoords!);
        pairwise = {
          labelA: a.label,
          labelB: b.label,
          badgeA: a.badge,
          badgeB: b.badge,
          rmsd: align.rmsd,
          tmScore: align.tmScore,
          alignedResidues: a.length ?? undefined,
          isSubstructure: false,
        };
      } else {
        const shorter = (a.length ?? 0) <= (b.length ?? 0) ? a : b;
        const longer = shorter === a ? b : a;
        const align = kabschAlignSubstructure(shorter.caCoords!, longer.caCoords!, shorter.resNums);
        pairwise = {
          labelA: a.label,
          labelB: b.label,
          badgeA: a.badge,
          badgeB: b.badge,
          rmsd: align.rmsd,
          tmScore: align.tmScore,
          alignedResidues: align.alignedResidues,
          isSubstructure: true,
        };
      }
    } catch {
      pairwise = null;
    }
  }

  const models: ViewerModelSpec[] = [];
  let sharedReferenceLabel: string | null = null;
  let kabschAligned = false;

  if (effectiveLayout === 'overlay') {
    // 1. Optional ghosted shared designed backbone
    if (hasSharedDesigned && metricBranches[0].designedPdb) {
      sharedReferenceLabel = 'Designed Backbone';
      kabschAligned = true;
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

    const anchor = with3d[anchorIdx];
    for (let i = 0; i < with3d.length; i++) {
      const b = with3d[i];
      let pdb = b.pdbText!;
      // If Kabsch align is enabled and NOT already in a shared pre-aligned frame, align non-anchor models onto anchor
      if (wantKabsch && !hasSharedDesigned && i !== anchorIdx && anchor?.caCoords && b.caCoords) {
        try {
          const res =
            (b.length ?? 0) === (anchor.length ?? 0)
              ? kabschAlign(b.caCoords, anchor.caCoords)
              : kabschAlignSubstructure(b.caCoords, anchor.caCoords, b.resNums);
          pdb = applyTransformToPdb(pdb, res.rotation, res.translation);
          kabschAligned = true;
        } catch {
          /* keep original coordinates on degenerate alignment */
        }
      }
      const metricBits: string[] = [];
      if (b.scRmsd !== null) metricBits.push(`${b.scRmsd.toFixed(2)} Å`);
      if (b.meanPlddt !== null) metricBits.push(`pLDDT ${b.meanPlddt.toFixed(0)}`);
      models.push({
        id: `branch-${b.nodeId}`,
        nodeId: b.nodeId,
        badge: b.badge,
        label: b.label,
        pdbText: pdb,
        colorHex: b.colorHex,
        colorCss: b.colorCss,
        alpha: 1.0,
        colorByPlddt: opts.colorMode === 'plddt' && (b.kind === 'metrics' || b.kind === 'fold'),
        visible: !hidden.has(`branch-${b.nodeId}`),
        metricSummary: metricBits.join(' · ') || (b.length ? `${b.length} aa` : undefined),
      });
    }
  } else {
    // Side-by-side inside a single Mol* scene via centroid-centered +X translation
    let xCursor = 0;
    let prevRg = 0;
    for (let i = 0; i < with3d.length; i++) {
      const b = with3d[i];
      const rg = Math.max(8, b.geometry?.rg ?? 14);
      if (i > 0) {
        xCursor += (prevRg + rg) * 1.65 + 10;
      }
      prevRg = rg;

      const [cx, cy, cz] = centroidOf(b.caCoords!);
      const shift = new Float64Array([xCursor - cx, -cy, -cz]);
      const shiftedPdb = applyTransformToPdb(b.pdbText!, IDENTITY_ROT, shift);

      const metricBits: string[] = [];
      if (b.scRmsd !== null) metricBits.push(`${b.scRmsd.toFixed(2)} Å`);
      if (b.meanPlddt !== null) metricBits.push(`pLDDT ${b.meanPlddt.toFixed(0)}`);
      models.push({
        id: `branch-${b.nodeId}`,
        nodeId: b.nodeId,
        badge: b.badge,
        label: b.label,
        pdbText: shiftedPdb,
        colorHex: b.colorHex,
        colorCss: b.colorCss,
        alpha: 1.0,
        colorByPlddt: opts.colorMode === 'plddt' && (b.kind === 'metrics' || b.kind === 'fold'),
        visible: !hidden.has(`branch-${b.nodeId}`),
        metricSummary: metricBits.join(' · ') || (b.length ? `${b.length} aa` : undefined),
      });
    }
  }

  return {
    branches,
    models,
    effectiveLayout,
    forcedSideBySide,
    kabschAligned,
    sharedReferenceLabel,
    pairwise,
  };
}
