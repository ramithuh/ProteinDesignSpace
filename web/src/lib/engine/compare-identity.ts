// Pure-TS identity detection for 3D Compare: find branches that render the exact same
// structure (e.g. a Dyna-1 / ThermoMPNN / ESM-C / GNINA painted copy of its parent backbone)
// so the overlay can show ONE cartoon carrying the per-residue track instead of z-fighting
// duplicates at 0.00 Å.

import type {
  AtomGranularity,
  BfactorSemantics,
  ResidueTrackArtifact,
  StructureArtifact,
} from './types';

/** B-factor semantics that encode a per-residue scientific track (not pLDDT / crystal B). */
export type TrackSemantics = Extract<
  BfactorSemantics,
  'dynamics-prob' | 'ddg-stability' | 'llr-fitness' | 'gnina-contrib'
>;

const TRACK_SEMANTICS: ReadonlySet<string> = new Set<TrackSemantics>([
  'dynamics-prob',
  'ddg-stability',
  'llr-fitness',
  'gnina-contrib',
]);

export function isTrackSemantics(sem: string | null | undefined): sem is TrackSemantics {
  return typeof sem === 'string' && TRACK_SEMANTICS.has(sem);
}

/** Human-readable noun for a painted track (used in merged-model captions). */
export const TRACK_NOUN: Record<TrackSemantics, string> = {
  'dynamics-prob': 'Dyna-1 flexibility',
  'ddg-stability': 'ΔΔG stability',
  'llr-fitness': 'ESM-C LLR fitness',
  'gnina-contrib': 'GNINA pocket contribution',
};

/** Maximum Cα RMSD (Å) for two branches to count as the same rendered structure. */
export const IDENTICAL_RMSD_TOL_A = 0.05;

/** Minimal branch shape needed for identity checks (a subset of `CompareBranchSummary`). */
export interface IdentityBranch {
  nodeId: string;
  label: string;
  length: number | null;
  sequence?: string | null;
  resNums: number[] | null;
  chainIds?: string[] | null;
  /** Chain IDs parsed from the text that is actually rendered (cifText || pdbText). */
  renderedChainIds?: string[] | null;
  structureId: string | null;
  parentStructureId?: string | null;
  bfactorSemantics?: BfactorSemantics | null;
  atomGranularity?: AtomGranularity | null;
  ligandNames?: string[] | null;
  nucleicAcidChains?: string[] | null;
  /** Number of polymer `ATOM` records in the rendered text. */
  renderedAtomCount?: number | null;
  caCoords: Float32Array | null;
}

/** Minimal pairwise alignment shape (a subset of `PairwiseAgreement`). */
export interface IdentityPairwise {
  nodeIdA?: string;
  nodeIdB?: string;
  rmsd: number;
  alignedResidues?: number;
  isSubstructure?: boolean;
}

export interface TrackMergeGroup {
  /** Stable group key (`merge-<first track nodeId>`), used to remember the active track. */
  id: string;
  /** First track-bearing member in selection order; every other member matches it. */
  baseNodeId: string;
  /** All members in selection order. */
  memberNodeIds: string[];
  /** Track-bearing members in selection order (>= 1). */
  trackNodeIds: string[];
  /** Plain (track-less) members that are dropped from the scene. */
  absorbedNodeIds: string[];
  /** Track currently shown on the shared cartoon. */
  activeTrackNodeId: string;
  /** Largest Cα RMSD between the base and any member (Å). */
  rmsd: number;
  /** `lineage` when every member is linked to the base by structure ID / parent; else `geometry`. */
  evidence: 'lineage' | 'geometry';
}

function normSeq(seq: string | null | undefined): string {
  return (seq ?? '').replace(/\s+/g, '').toUpperCase();
}

function arraysEqual<T>(a: readonly T[], b: readonly T[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function sortedUnique(values: readonly string[] | null | undefined): string {
  return [...new Set((values ?? []).map((v) => v.trim().toUpperCase()).filter(Boolean))]
    .sort()
    .join(',');
}

/** Raw (un-superposed) Cα RMSD, or `Infinity` when coordinates are missing / mismatched. */
export function rawCaRmsd(a: Float32Array | null, b: Float32Array | null): number {
  if (!a || !b || a.length !== b.length || a.length < 3) return Infinity;
  let sum = 0;
  const n = Math.floor(a.length / 3);
  for (let i = 0; i < n * 3; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return n > 0 ? Math.sqrt(sum / n) : Infinity;
}

/** `ca-only` never matches `backbone4` / `all-atom` (e.g. Genie 3 vs PULCHRA / cg2all rebuilds). */
export function granularityCompatible(
  a: AtomGranularity | null | undefined,
  b: AtomGranularity | null | undefined,
): boolean {
  const ga = a ?? null;
  const gb = b ?? null;
  if (ga === gb) return true;
  if (ga === 'ca-only' || gb === 'ca-only') return false;
  return !(ga && gb);
}

/** Ligand and nucleic-acid content must match exactly so a holo / docked pose is never hidden. */
export function nonProteinSignatureEqual(a: IdentityBranch, b: IdentityBranch): boolean {
  return (
    sortedUnique(a.ligandNames) === sortedUnique(b.ligandNames) &&
    sortedUnique(a.nucleicAcidChains) === sortedUnique(b.nucleicAcidChains)
  );
}

/** Two branches are linked by lineage: same artifact, child → parent, or shared parent. */
export function shareLineage(a: IdentityBranch, b: IdentityBranch): boolean {
  if (a.structureId && a.structureId === b.structureId) return true;
  if (a.parentStructureId && a.parentStructureId === b.structureId) return true;
  if (b.parentStructureId && b.parentStructureId === a.structureId) return true;
  return Boolean(a.parentStructureId && a.parentStructureId === b.parentStructureId);
}

export function findPairwise<P extends IdentityPairwise>(
  allPairwise: readonly P[],
  nodeIdA: string,
  nodeIdB: string,
): P | null {
  for (const p of allPairwise) {
    if (
      (p.nodeIdA === nodeIdA && p.nodeIdB === nodeIdB) ||
      (p.nodeIdA === nodeIdB && p.nodeIdB === nodeIdA)
    ) {
      return p;
    }
  }
  return null;
}

/**
 * True when `a` and `b` render the same structure: equal length, sequence, rendered chain IDs,
 * residue numbering, atom granularity, polymer atom count and ligand / nucleic-acid content,
 * plus a full-length (non-substructure) Kabsch fit below `tolA`. Lineage is never sufficient.
 */
export function branchesRenderIdentically(
  a: IdentityBranch,
  b: IdentityBranch,
  pair: IdentityPairwise | null,
  opts: { tolA?: number; requireRawCoords?: boolean } = {},
): boolean {
  const tol = opts.tolA ?? IDENTICAL_RMSD_TOL_A;
  const L = a.length ?? 0;
  if (L < 3 || b.length !== L) return false;
  const seqA = normSeq(a.sequence);
  if (!seqA || seqA !== normSeq(b.sequence)) return false;
  if (!a.resNums || !b.resNums || !arraysEqual(a.resNums, b.resNums)) return false;
  const chainsA = a.renderedChainIds ?? a.chainIds ?? null;
  const chainsB = b.renderedChainIds ?? b.chainIds ?? null;
  if ((chainsA === null) !== (chainsB === null)) return false;
  if (chainsA && chainsB && !arraysEqual(chainsA, chainsB)) return false;
  if (!granularityCompatible(a.atomGranularity, b.atomGranularity)) return false;
  if (
    typeof a.renderedAtomCount === 'number' &&
    typeof b.renderedAtomCount === 'number' &&
    a.renderedAtomCount !== b.renderedAtomCount
  ) {
    return false;
  }
  if (!nonProteinSignatureEqual(a, b)) return false;
  if (!pair || pair.isSubstructure || pair.alignedResidues !== L) return false;
  if (!Number.isFinite(pair.rmsd) || pair.rmsd >= tol) return false;
  if (opts.requireRawCoords && !(rawCaRmsd(a.caCoords, b.caCoords) < tol)) return false;
  return true;
}

/**
 * Group branches that render identically around a track-bearing base so the overlay can keep
 * one cartoon per backbone. Each group is seeded by the first ungrouped track-bearing branch
 * (selection order); any ungrouped branch — plain or track — joins only if it matches that base,
 * so matches never chain. Plain-only duplicates are never grouped.
 */
export function findIdenticalTrackGroups(
  branches: readonly IdentityBranch[],
  allPairwise: readonly IdentityPairwise[],
  opts: {
    tolA?: number;
    requireRawCoords?: boolean;
    activeTrackNodeIds?: Readonly<Record<string, string>>;
  } = {},
): TrackMergeGroup[] {
  const grouped = new Set<string>();
  const groups: TrackMergeGroup[] = [];
  for (const base of branches) {
    if (grouped.has(base.nodeId) || !isTrackSemantics(base.bfactorSemantics)) continue;
    const members: IdentityBranch[] = [];
    let maxRmsd = 0;
    let allLineage = true;
    for (const cand of branches) {
      if (cand.nodeId === base.nodeId) {
        members.push(cand);
        continue;
      }
      if (grouped.has(cand.nodeId)) continue;
      const pair = findPairwise(allPairwise, base.nodeId, cand.nodeId);
      if (!branchesRenderIdentically(base, cand, pair, opts)) continue;
      members.push(cand);
      maxRmsd = Math.max(maxRmsd, pair!.rmsd);
      if (!shareLineage(base, cand)) allLineage = false;
    }
    if (members.length < 2) continue;
    for (const m of members) grouped.add(m.nodeId);
    const id = `merge-${base.nodeId}`;
    const trackNodeIds = members
      .filter((m) => isTrackSemantics(m.bfactorSemantics))
      .map((m) => m.nodeId);
    const requested = opts.activeTrackNodeIds?.[id];
    groups.push({
      id,
      baseNodeId: base.nodeId,
      memberNodeIds: members.map((m) => m.nodeId),
      trackNodeIds,
      absorbedNodeIds: members
        .filter((m) => !isTrackSemantics(m.bfactorSemantics))
        .map((m) => m.nodeId),
      activeTrackNodeId:
        requested && trackNodeIds.includes(requested) ? requested : trackNodeIds[0],
      rmsd: maxRmsd,
      evidence: allLineage ? 'lineage' : 'geometry',
    });
  }
  return groups;
}

/**
 * How `branch` represents the synthetic shared designed backbone (`ref-designed` ghost):
 * `'id'` for the same artifact, `'lineage'` for a painted child whose raw Cα coordinates are
 * unchanged, or `null` when the ghost is still needed.
 */
export function branchRepresentsStructure(
  branch: IdentityBranch,
  target: { structureId: string | null; caCoords: Float32Array | null; length: number | null },
  tolA = IDENTICAL_RMSD_TOL_A,
): 'id' | 'lineage' | null {
  if (!target.structureId) return null;
  if (branch.structureId && branch.structureId === target.structureId) return 'id';
  if (
    branch.parentStructureId === target.structureId &&
    branch.length !== null &&
    branch.length === target.length &&
    rawCaRmsd(branch.caCoords, target.caCoords) < tolA
  ) {
    return 'lineage';
  }
  return null;
}

/** Expand a model ID to every model ID of its merge group (hide / solo act on the whole group). */
export function expandMergedModelIds(
  modelId: string,
  groups: ReadonlyArray<{ memberModelIds: readonly string[] }>,
): string[] {
  for (const g of groups) {
    if (g.memberModelIds.includes(modelId)) return [...g.memberModelIds];
  }
  return [modelId];
}

const TRACK_META_BY_SEMANTICS: Record<
  TrackSemantics,
  {
    quantity: ResidueTrackArtifact['quantity'];
    unit: string;
    range?: [number, number];
    threshold?: number;
  }
> = {
  'dynamics-prob': {
    quantity: 'dynamics-prob',
    unit: 'p(exch)',
    range: [0, 1],
    threshold: 0.5,
  },
  'ddg-stability': {
    quantity: 'ddg-stability',
    unit: 'kcal/mol',
    threshold: 0,
  },
  'llr-fitness': {
    quantity: 'llr-fitness',
    unit: 'LLR',
    threshold: 0,
  },
  'gnina-contrib': {
    quantity: 'gnina-contrib',
    unit: 'ΔCNN_VS',
    threshold: 0,
  },
};

function tokenizeCifRow(line: string): string[] {
  const out: string[] = [];
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
      while (i < n && !(line[i] === q && (i + 1 === n || line[i + 1] === ' ' || line[i + 1] === '\t'))) {
        i++;
      }
      out.push(line.slice(start, i));
      if (i < n) i++;
    } else {
      const start = i;
      while (i < n && line[i] !== ' ' && line[i] !== '\t') i++;
      out.push(line.slice(start, i));
    }
  }
  return out;
}

function extractPerResidueCaBfactors(
  rawText: string,
): { values: Float32Array; residueIds: string[] } | null {
  if (!rawText) return null;
  const isCif = /^\s*(data_|loop_|_atom_site\.)/m.test(rawText);
  const values: number[] = [];
  const residueIds: string[] = [];
  const seen = new Set<string>();

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
      const chain = line[21]?.trim() || 'A';
      const resSeq = line.slice(22, 26).trim();
      const ins = (line[26] ?? ' ').trim();
      const key = `${chain}:${resSeq}:${ins}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const b = parseFloat(line.slice(60, 66));
      values.push(Number.isFinite(b) ? b : 0);
      residueIds.push(`${chain}${resSeq}${ins}`);
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
      const idxGroup = cols.indexOf('group_PDB');
      const idxAtom = cols.indexOf('label_atom_id') >= 0 ? cols.indexOf('label_atom_id') : cols.indexOf('auth_atom_id');
      const idxComp = cols.indexOf('label_comp_id') >= 0 ? cols.indexOf('label_comp_id') : cols.indexOf('auth_comp_id');
      const idxAlt = cols.indexOf('label_alt_id');
      const idxChain = cols.indexOf('auth_asym_id') >= 0 ? cols.indexOf('auth_asym_id') : cols.indexOf('label_asym_id');
      const idxSeq = cols.indexOf('auth_seq_id') >= 0 ? cols.indexOf('auth_seq_id') : cols.indexOf('label_seq_id');
      const idxIns = cols.indexOf('pdbx_PDB_ins_code');
      const idxB = cols.indexOf('B_iso_or_equiv');
      const idxModel = cols.indexOf('pdbx_PDB_model_num');
      if (idxAtom < 0 || idxB < 0) continue;
      if (idxModel >= 0) {
        const m = toks[idxModel];
        if (firstModel === null) firstModel = m;
        else if (m !== firstModel) break;
      }
      const grp = idxGroup >= 0 ? toks[idxGroup] : 'ATOM';
      const comp = idxComp >= 0 ? toks[idxComp].toUpperCase() : '';
      if (grp !== 'ATOM' && !(grp === 'HETATM' && comp === 'MSE')) continue;
      if (toks[idxAtom] !== 'CA') continue;
      const alt = idxAlt >= 0 ? toks[idxAlt] : '.';
      if (alt !== '.' && alt !== '?' && alt !== 'A' && alt !== '1') continue;
      const chain = idxChain >= 0 && toks[idxChain] !== '.' && toks[idxChain] !== '?' ? toks[idxChain] : 'A';
      const seq = idxSeq >= 0 ? toks[idxSeq] : String(values.length + 1);
      const ins = idxIns >= 0 && toks[idxIns] !== '.' && toks[idxIns] !== '?' ? toks[idxIns] : '';
      const key = `${chain}:${seq}:${ins}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const b = parseFloat(toks[idxB]);
      values.push(Number.isFinite(b) ? b : 0);
      residueIds.push(`${chain}${seq}${ins}`);
    }
  }

  if (values.length === 0) return null;
  return { values: new Float32Array(values), residueIds };
}

const DERIVED_TRACK_CACHE = new WeakMap<StructureArtifact, ResidueTrackArtifact | null>();

/**
 * Derive a per-residue `ResidueTrackArtifact` from a painted `StructureArtifact`'s Cα B-factors
 * when `bfactorSemantics` is a scientific track (`dynamics-prob`, `ddg-stability`, `llr-fitness`,
 * `gnina-contrib`). Memoized per `StructureArtifact` reference so repeated renders are free.
 */
export function deriveResidueTrackFromStructure(
  struct: StructureArtifact | null | undefined,
): ResidueTrackArtifact | null {
  if (!struct || !isTrackSemantics(struct.bfactorSemantics)) return null;
  if (DERIVED_TRACK_CACHE.has(struct)) {
    return DERIVED_TRACK_CACHE.get(struct) ?? null;
  }
  const text = struct.cifText || struct.pdbText;
  if (!text) {
    DERIVED_TRACK_CACHE.set(struct, null);
    return null;
  }
  const extracted = extractPerResidueCaBfactors(text);
  if (!extracted || extracted.values.length === 0) {
    DERIVED_TRACK_CACHE.set(struct, null);
    return null;
  }
  const vals = extracted.values;
  const meta = TRACK_META_BY_SEMANTICS[struct.bfactorSemantics];
  let sum = 0;
  let minV = Infinity;
  let maxV = -Infinity;
  let above = 0;
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    sum += v;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
    if (meta.threshold !== undefined && v >= meta.threshold) above++;
  }
  const mean = sum / vals.length;
  const residueIds =
    struct.sourceResidues && struct.sourceResidues.length === vals.length
      ? struct.sourceResidues.map((r) => `${r.authChainId}${r.resNum}${r.insCode ?? ''}`)
      : extracted.residueIds;
  const seq =
    struct.sequence && struct.sequence.length === vals.length
      ? struct.sequence
      : 'X'.repeat(vals.length);

  const track: ResidueTrackArtifact = {
    kind: 'ResidueTrack',
    id: `${struct.id}-track`,
    label: `${TRACK_NOUN[struct.bfactorSemantics]} (mean ${mean.toFixed(2)})`,
    sourceModelId: struct.sourceModelId,
    ...(struct.parentStructureId ? { parentStructureId: struct.parentStructureId } : {}),
    quantity: meta.quantity,
    unit: meta.unit,
    ...(meta.range ? { range: meta.range } : {}),
    ...(meta.threshold !== undefined ? { threshold: meta.threshold } : {}),
    residues: seq,
    residueIds,
    values: vals,
    summary: {
      mean,
      min: Number.isFinite(minV) ? minV : 0,
      max: Number.isFinite(maxV) ? maxV : 0,
      ...(meta.threshold !== undefined ? { fractionAboveThreshold: above / vals.length } : {}),
    },
  };
  DERIVED_TRACK_CACHE.set(struct, track);
  return track;
}
