// Layer 3 Plugin CPU Runner — designability-scorer
// Self-consistency & external-reference scorer: Kabsch alignment, scRMSD, scTM,
// motif RMSD, binder/target interface metrics (iPAE, ipSAE, binder scRMSD, target-aligned RMSD),
// and strict multi-backbone lineage enforcement.

import { parseStructureText } from '$lib/protein-ops/pdb';
import { applyTransformToPdb, computeGeometryStats, kabschAlign } from '$lib/protein-ops/kabsch';
import { computeIpsae, sliceProteinPae } from '$lib/protein-ops/ipsae';
import { NUCLEIC_ACID_RESNAMES, tokenizeCifLine } from '$lib/protein-ops/hetatm-policy';
import { resolveFoldConfidence } from '$lib/engine/bio-capabilities';
import { canonicalJsonStringify, fnv1a } from '$lib/kernel/hash-utils';
import type { UpstreamGraphQuery } from '$lib/kernel/plugin-types';
import type {
  CanonicalArtifact,
  DesignabilityMetrics,
  DesignabilityRow,
  FoldedCandidate,
  FoldedStructure,
  ReferenceRelation,
  ReferenceResidueMapping,
  SequenceSet,
  StructureArtifact,
  WorkflowNode,
} from '$lib/engine/types';

interface ScorerChainRange {
  chainId: string;
  start: number;
  length: number;
}

export interface NativeMotifAtomCoord {
  resIdx: number;
  atomName: string;
  x: number;
  y: number;
  z: number;
}

type StructureWithMotifAtoms = StructureArtifact & {
  nativeMotifAtomCoords?: NativeMotifAtomCoord[];
};

type DesignabilityRowWithMotifAtoms = DesignabilityRow & {
  genMotifAtomRmsd?: number | null;
  refoldMotifAtomRmsd?: number | null;
};

type DesignabilityMetricsWithMotifAtoms = DesignabilityMetrics & {
  genMotifAtomRmsd?: number | null;
  refoldMotifAtomRmsd?: number | null;
};

const SCORER_WATER_RESNAMES = new Set(['HOH', 'WAT', 'DOD', 'H2O', 'TIP', 'TIP3']);
const SCORER_PEPTIDE_CAP_RESNAMES = new Set(['ACE', 'NME', 'NH2', 'FOR']);

function scorerAltLocPriority(alt: string): number {
  const ch = alt.trim();
  if (ch === '' || ch === 'A' || ch === '1') return 0;
  if (ch === 'B' || ch === '2') return 1;
  return 2;
}

function isProteinHydrogen(elemRaw: string, atomNameRaw: string): boolean {
  const elem = elemRaw.trim().toUpperCase();
  if (elem && elem !== '.' && elem !== '?') {
    return elem === 'H' || elem === 'D';
  }
  const atom = atomNameRaw.trim().toUpperCase();
  return /^[0-9]*[HD]/.test(atom);
}

/**
 * Parse all heavy atoms of protein residues from PDB or mmCIF text, indexed by 0-based protein
 * residue index matching `parseStructureText` (`caCoords` ordering: only residues with observed
 * `CA`, skipping waters, non-MSE HETATM ligands, nucleic acids, peptide caps, and hydrogens).
 */
export function extractProteinHeavyAtomsByResidueIndex(
  structureText: string,
): Array<Map<string, [number, number, number]>> {
  if (!structureText || !structureText.trim()) return [];
  interface ResEntry {
    atoms: Map<string, [number, number, number]>;
    atomAltPrio: Map<string, number>;
  }
  const residues: ResEntry[] = [];
  const byKey = new Map<string, ResEntry>();

  const addAtom = (
    resKey: string,
    atomNameRaw: string,
    prio: number,
    x: number,
    y: number,
    z: number,
  ) => {
    const atomName = atomNameRaw.trim().toUpperCase();
    if (!atomName || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
    let entry = byKey.get(resKey);
    if (!entry) {
      entry = { atoms: new Map(), atomAltPrio: new Map() };
      byKey.set(resKey, entry);
      residues.push(entry);
    }
    const prevPrio = entry.atomAltPrio.get(atomName);
    if (prevPrio === undefined || prio < prevPrio) {
      entry.atoms.set(atomName, [x, y, z]);
      entry.atomAltPrio.set(atomName, prio);
    }
  };

  const trimmed = structureText.trimStart();
  const isMmcif = trimmed.startsWith('data_') || structureText.includes('_atom_site.');

  if (isMmcif) {
    const lines = structureText.split(/\r?\n/);
    let inLoop = false;
    let inAtomSite = false;
    let cols: string[] = [];
    let iGroup = -1;
    let iModel = -1;
    let iAlt = -1;
    let iLabelAtom = -1;
    let iAuthAtom = -1;
    let iAuthAsym = -1;
    let iLabelAsym = -1;
    let iAuthSeq = -1;
    let iLabelSeq = -1;
    let iIns = -1;
    let iLabelComp = -1;
    let iAuthComp = -1;
    let iElem = -1;
    let iX = -1;
    let iY = -1;
    let iZ = -1;
    let firstModel: string | null = null;
    let pendingRow: string[] = [];

    const cifVal = (row: string[], idx: number): string => {
      if (idx < 0 || idx >= row.length) return '';
      const v = row[idx];
      return v === '.' || v === '?' ? '' : v;
    };

    const flushRow = (row: string[]) => {
      const modelNum = cifVal(row, iModel);
      if (modelNum) {
        if (firstModel === null) firstModel = modelNum;
        else if (modelNum !== firstModel) return;
      }
      const prio = scorerAltLocPriority(cifVal(row, iAlt));
      if (prio > 1) return;
      const group = (cifVal(row, iGroup) || 'ATOM').toUpperCase();
      const comp = (cifVal(row, iLabelComp) || cifVal(row, iAuthComp) || 'UNK').toUpperCase();
      if (
        SCORER_PEPTIDE_CAP_RESNAMES.has(comp) ||
        NUCLEIC_ACID_RESNAMES.has(comp) ||
        SCORER_WATER_RESNAMES.has(comp)
      ) {
        return;
      }
      if (group === 'HETATM') {
        if (comp !== 'MSE') return;
      } else if (group !== 'ATOM') {
        return;
      }
      const atomName = cifVal(row, iLabelAtom) || cifVal(row, iAuthAtom);
      if (!atomName || isProteinHydrogen(cifVal(row, iElem), atomName)) return;
      const chain = cifVal(row, iAuthAsym) || cifVal(row, iLabelAsym) || 'A';
      const seqId = cifVal(row, iAuthSeq) || cifVal(row, iLabelSeq);
      const ins = cifVal(row, iIns);
      const x = parseFloat(iX >= 0 ? row[iX] : '');
      const y = parseFloat(iY >= 0 ? row[iY] : '');
      const z = parseFloat(iZ >= 0 ? row[iZ] : '');
      addAtom(`${chain}:${seqId}:${ins}`, atomName, prio, x, y, z);
    };

    for (let li = 0; li < lines.length; li++) {
      const tr = lines[li].trim();
      if (!tr || tr.startsWith('#')) continue;
      if (tr === 'loop_') {
        if (inAtomSite) break;
        inLoop = true;
        inAtomSite = false;
        cols = [];
        pendingRow = [];
        continue;
      }
      if (inLoop && tr.startsWith('_')) {
        const colName = tr.split(/\s+/)[0];
        if (colName.startsWith('_atom_site.')) {
          inAtomSite = true;
          cols.push(colName);
          const idx = cols.length - 1;
          if (colName === '_atom_site.group_PDB') iGroup = idx;
          else if (colName === '_atom_site.pdbx_PDB_model_num') iModel = idx;
          else if (colName === '_atom_site.label_alt_id') iAlt = idx;
          else if (colName === '_atom_site.label_atom_id') iLabelAtom = idx;
          else if (colName === '_atom_site.auth_atom_id') iAuthAtom = idx;
          else if (colName === '_atom_site.auth_asym_id') iAuthAsym = idx;
          else if (colName === '_atom_site.label_asym_id') iLabelAsym = idx;
          else if (colName === '_atom_site.auth_seq_id') iAuthSeq = idx;
          else if (colName === '_atom_site.label_seq_id') iLabelSeq = idx;
          else if (colName === '_atom_site.pdbx_PDB_ins_code') iIns = idx;
          else if (colName === '_atom_site.label_comp_id') iLabelComp = idx;
          else if (colName === '_atom_site.auth_comp_id') iAuthComp = idx;
          else if (colName === '_atom_site.type_symbol') iElem = idx;
          else if (colName === '_atom_site.Cartn_x') iX = idx;
          else if (colName === '_atom_site.Cartn_y') iY = idx;
          else if (colName === '_atom_site.Cartn_z') iZ = idx;
        } else {
          inAtomSite = false;
        }
        continue;
      }
      if (inAtomSite) {
        if (
          tr.startsWith('_') ||
          tr.startsWith('data_') ||
          tr.startsWith('save_') ||
          tr === 'stop_'
        ) {
          break;
        }
        inLoop = false;
        const toks = tokenizeCifLine(tr);
        if (toks.length === 0) continue;
        if (pendingRow.length === 0 && toks.length >= cols.length) {
          flushRow(toks);
        } else {
          pendingRow.push(...toks);
          while (cols.length > 0 && pendingRow.length >= cols.length) {
            flushRow(pendingRow.slice(0, cols.length));
            pendingRow = pendingRow.slice(cols.length);
          }
        }
      } else {
        inLoop = false;
      }
    }
  } else {
    const lines = structureText.split(/\r?\n/);
    let seenAtoms = false;
    let seenModel = false;
    for (const line of lines) {
      if (line.startsWith('MODEL ')) {
        if (seenModel) break;
        seenModel = true;
        continue;
      }
      if (line.startsWith('ENDMDL')) {
        if (seenModel || seenAtoms) break;
        continue;
      }
      if (line.length < 54) continue;
      const isAtom = line.startsWith('ATOM  ');
      const isHet = line.startsWith('HETATM');
      if (!isAtom && !isHet) continue;
      const comp = line.slice(17, 20).trim().toUpperCase();
      if (
        SCORER_PEPTIDE_CAP_RESNAMES.has(comp) ||
        NUCLEIC_ACID_RESNAMES.has(comp) ||
        SCORER_WATER_RESNAMES.has(comp)
      ) {
        continue;
      }
      if (isHet && comp !== 'MSE') continue;
      seenAtoms = true;
      const prio = scorerAltLocPriority(line[16] ?? ' ');
      if (prio > 1) continue;
      const atomName = line.slice(12, 16).trim();
      const elem = line.length >= 78 ? line.slice(76, 78) : '';
      if (!atomName || isProteinHydrogen(elem, atomName)) continue;
      const chain = line[21]?.trim() || 'A';
      const resNum = line.slice(22, 26).trim();
      const ins = (line[26] ?? ' ').trim();
      const x = parseFloat(line.slice(30, 38));
      const y = parseFloat(line.slice(38, 46));
      const z = parseFloat(line.slice(46, 54));
      addAtom(`${chain}:${resNum}:${ins}`, atomName, prio, x, y, z);
    }
  }

  return residues.filter((r) => r.atoms.has('CA')).map((r) => r.atoms);
}

function computeMotifHeavyAtomRmsd(
  heavyAtomsByRes: readonly Map<string, [number, number, number]>[],
  nativeAtoms: readonly NativeMotifAtomCoord[],
): number | undefined {
  if (heavyAtomsByRes.length === 0 || nativeAtoms.length < 3) return undefined;
  const matchedMob: Array<[number, number, number]> = [];
  const matchedNat: Array<[number, number, number]> = [];
  for (const a of nativeAtoms) {
    const resMap = heavyAtomsByRes[a.resIdx];
    if (!resMap) continue;
    const coord = resMap.get(a.atomName);
    if (!coord) continue;
    matchedMob.push(coord);
    matchedNat.push([a.x, a.y, a.z]);
  }
  if (matchedMob.length < 3) return undefined;
  const mobBuf = new Float32Array(matchedMob.length * 3);
  const natBuf = new Float32Array(matchedNat.length * 3);
  for (let i = 0; i < matchedMob.length; i++) {
    mobBuf[i * 3 + 0] = matchedMob[i][0];
    mobBuf[i * 3 + 1] = matchedMob[i][1];
    mobBuf[i * 3 + 2] = matchedMob[i][2];
    natBuf[i * 3 + 0] = matchedNat[i][0];
    natBuf[i * 3 + 1] = matchedNat[i][1];
    natBuf[i * 3 + 2] = matchedNat[i][2];
  }
  const res = kabschAlign(mobBuf, natBuf);
  return Number.isFinite(res.rmsd) ? res.rmsd : undefined;
}

function resolveScorerChainRanges(
  L: number,
  chainLengths?: readonly number[],
  chainIds?: readonly string[],
): ScorerChainRange[] {
  if (L <= 0) return [];
  if (chainIds && chainIds.length === L) {
    const ranges: ScorerChainRange[] = [];
    let start = 0;
    for (let i = 1; i <= L; i++) {
      if (i === L || chainIds[i] !== chainIds[start]) {
        ranges.push({ chainId: chainIds[start], start, length: i - start });
        start = i;
      }
    }
    if (ranges.length > 0) return ranges;
  }
  if (chainLengths && chainLengths.length > 0 && chainLengths.reduce((a, b) => a + b, 0) === L) {
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let start = 0;
    return chainLengths.map((len, idx) => {
      const r = { chainId: alphabet[idx % alphabet.length], start, length: len };
      start += len;
      return r;
    });
  }
  return [{ chainId: 'A', start: 0, length: L }];
}

function expandChainRangesToIds(ranges: readonly ScorerChainRange[]): string[] {
  const out: string[] = [];
  for (const r of ranges) {
    for (let i = 0; i < r.length; i++) out.push(r.chainId);
  }
  return out;
}

function splitBinderAndTargetIndices(
  ranges: readonly ScorerChainRange[],
  chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>,
  lockedChains?: readonly string[],
): { binderIndices: number[]; targetIndices: number[]; explicitRoleSplit: boolean } {
  let binderRanges: ScorerChainRange[] = [];
  let targetRanges: ScorerChainRange[] = [];
  let explicitRoleSplit = false;

  if (chainRoles) {
    binderRanges = ranges.filter((r) => chainRoles[r.chainId] === 'binder');
    targetRanges = ranges.filter((r) => chainRoles[r.chainId] === 'target');
    if (binderRanges.length > 0 && targetRanges.length > 0) {
      explicitRoleSplit = true;
    }
  }

  if (!explicitRoleSplit && lockedChains && lockedChains.length > 0) {
    const locked = new Set(lockedChains.map((c) => c.trim().toUpperCase()));
    targetRanges = ranges.filter((r) => locked.has(r.chainId.toUpperCase()));
    binderRanges = ranges.filter((r) => !locked.has(r.chainId.toUpperCase()));
    if (binderRanges.length > 0 && targetRanges.length > 0) {
      explicitRoleSplit = true;
    }
  }

  if (binderRanges.length === 0 || targetRanges.length === 0) {
    // Fallback heuristic: shorter chain is the designed binder (tie-breaker: later chain, matching Target=A, Binder=B)
    let binderIdx = ranges.length - 1;
    let minLen = ranges[binderIdx].length;
    for (let i = ranges.length - 2; i >= 0; i--) {
      if (ranges[i].length < minLen) {
        minLen = ranges[i].length;
        binderIdx = i;
      }
    }
    binderRanges = [ranges[binderIdx]];
    targetRanges = ranges.filter((_, idx) => idx !== binderIdx);
  }

  const expand = (rs: readonly ScorerChainRange[]): number[] => {
    const out: number[] = [];
    for (const r of rs) {
      for (let i = r.start; i < r.start + r.length; i++) out.push(i);
    }
    return out;
  };

  return {
    binderIndices: expand(binderRanges),
    targetIndices: expand(targetRanges),
    explicitRoleSplit,
  };
}

function gatherCoordsByIndices(coords: Float32Array, indices: readonly number[]): Float32Array {
  const out = new Float32Array(indices.length * 3);
  for (let i = 0; i < indices.length; i++) {
    const src = indices[i] * 3;
    out[i * 3 + 0] = coords[src + 0];
    out[i * 3 + 1] = coords[src + 1];
    out[i * 3 + 2] = coords[src + 2];
  }
  return out;
}

function computeInterfacePae(
  pae: Float32Array,
  paeDim: number,
  proteinLen: number,
  ranges: readonly ScorerChainRange[],
  binderIndices?: readonly number[],
  targetIndices?: readonly number[],
): number | undefined {
  let sum = 0;
  let count = 0;
  if (binderIndices && targetIndices && binderIndices.length > 0 && targetIndices.length > 0) {
    for (const bi of binderIndices) {
      for (const ti of targetIndices) {
        const v1 = pae[bi * paeDim + ti];
        if (Number.isFinite(v1)) {
          sum += v1;
          count++;
        }
        const v2 = pae[ti * paeDim + bi];
        if (Number.isFinite(v2)) {
          sum += v2;
          count++;
        }
      }
    }
  } else {
    const chainOf = new Int32Array(proteinLen);
    for (let c = 0; c < ranges.length; c++) {
      const r = ranges[c];
      for (let i = r.start; i < r.start + r.length; i++) chainOf[i] = c;
    }
    for (let i = 0; i < proteinLen; i++) {
      const ci = chainOf[i];
      for (let j = 0; j < proteinLen; j++) {
        if (ci === chainOf[j]) continue;
        const v = pae[i * paeDim + j];
        if (Number.isFinite(v)) {
          sum += v;
          count++;
        }
      }
    }
  }
  return count > 0 ? sum / count : undefined;
}

function parseReferenceResidueMapping(
  raw: unknown,
  foldLen: number,
  refLen: number,
): { pairs: Array<[number, number]>; spec: ReferenceResidueMapping } | null {
  if (raw === undefined || raw === null || raw === '') return null;
  const pairs: Array<[number, number]> = [];
  let label: string | undefined;

  const assertStrictInt = (v: unknown, role: string): number => {
    if (typeof v !== 'number' || !Number.isInteger(v)) {
      throw new Error(
        `Invalid residueMapping: ${role} must be a strict finite integer (got ${String(v)}).`,
      );
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
      pairs.push([
        assertStrictInt(entry[0], 'fold index'),
        assertStrictInt(entry[1], 'reference index'),
      ]);
    }
  } else if (typeof raw === 'object') {
    const obj = raw as ReferenceResidueMapping;
    label = obj.label;
    if (Array.isArray(obj.pairs)) {
      for (const entry of obj.pairs) {
        if (!Array.isArray(entry) || entry.length < 2) {
          throw new Error('Invalid residueMapping.pairs: every entry must be a [foldIdx, refIdx] pair.');
        }
        pairs.push([
          assertStrictInt(entry[0], 'fold index'),
          assertStrictInt(entry[1], 'reference index'),
        ]);
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
      throw new Error(
        `Invalid residueMapping: fold index ${fi} is out of bounds for fold length ${foldLen}.`,
      );
    }
    if (ri < 0 || ri >= refLen) {
      throw new Error(
        `Invalid residueMapping: reference index ${ri} is out of bounds for reference length ${refLen}.`,
      );
    }
    if (seenFold.has(fi)) {
      throw new Error(
        `Invalid residueMapping: duplicate fold index ${fi} (mapping must be 1-to-1 with >= 3 distinct pairs).`,
      );
    }
    if (seenRef.has(ri)) {
      throw new Error(
        `Invalid residueMapping: duplicate reference index ${ri} (mapping must be 1-to-1 with >= 3 distinct pairs).`,
      );
    }
    seenFold.add(fi);
    seenRef.add(ri);
  }

  return {
    pairs,
    spec: {
      pairs,
      foldIndices: pairs.map(([fi]) => fi),
      referenceIndices: pairs.map(([, ri]) => ri),
      ...(label ? { label } : {}),
    },
  };
}

/** Trace SequenceSet lineage feeding the upstream LocalFold node for designability-scorer. */
export function resolveDesignabilityUpstreamLineage(
  nodeId: string,
  upstreamGraph: UpstreamGraphQuery,
): string | undefined {
  for (const edge of upstreamGraph.getIncomingEdges(nodeId, 'folds')) {
    for (const seqEdge of upstreamGraph.getIncomingEdges(edge.fromNode, 'sequences')) {
      const seqArt = upstreamGraph.getNodeOutputs(seqEdge.fromNode)?.[
        seqEdge.fromPort
      ] as unknown as SequenceSet | undefined;
      if (seqArt && seqArt.kind === 'SequenceSet') {
        const candidateLineage = seqArt.sourceStructureId ?? seqArt.parentStructureId;
        if (candidateLineage && candidateLineage.toLowerCase() !== 'none') {
          return candidateLineage;
        }
      }
    }
  }
  return undefined;
}

export async function runDesignabilityScorerCpu(
  node: WorkflowNode,
  inputs: Record<string, CanonicalArtifact>,
  upstreamLineageHint?: string,
): Promise<Record<string, CanonicalArtifact>> {
  const structure = inputs.structure as StructureArtifact | undefined;
  const folds = inputs.folds as FoldedStructure | undefined;
  if (!structure || !folds) {
    throw new Error('designability-scorer requires both structure and folds inputs');
  }
  const scThresh = Number(node.params.scRmsdThreshold ?? 2.0);
  const plddtThresh = Number(node.params.plddtThreshold ?? 70);
  const motifThresh = Number(node.params.motifRmsdThreshold ?? 1.0);
  const ipaeThresh = Number(node.params.ipaeThreshold ?? 10.0);
  const ipsaeThresh = Number(node.params.ipsaeThreshold ?? 0);
  const targetAlignedBinderThresh = Number(
    node.params.targetAlignedBinderRmsdThreshold ?? 4.0,
  );

  const allowExternalReference = Boolean(node.params.allowExternalReference);
  const strictLineage = Boolean(node.params.strictLineage);
  const rawMappingParam = node.params.residueMapping;

  const normalizeLineageId = (val: string | undefined | null): string => {
    const s = String(val ?? '').trim();
    return s && s.toLowerCase() !== 'none' ? s : '';
  };

  // Helper: precompute motif pairs, genMotifRmsd, and designedStructureText for a specific reference backbone
  const prepareRefScoringContext = (ref: StructureArtifact) => {
    const refLen = Math.floor(ref.caCoords.length / 3);
    const motifPairs: Array<{ idx: number; coord: [number, number, number] }> = [];
    if (
      Array.isArray(ref.motifIndices) &&
      Array.isArray(ref.nativeMotifCoords) &&
      ref.motifIndices.length > 0 &&
      ref.nativeMotifCoords.length > 0
    ) {
      const limit = Math.min(ref.motifIndices.length, ref.nativeMotifCoords.length);
      for (let k = 0; k < limit; k++) {
        const idx = ref.motifIndices[k];
        const nc = ref.nativeMotifCoords[k];
        if (
          Number.isInteger(idx) &&
          idx >= 0 &&
          idx < refLen &&
          Array.isArray(nc) &&
          nc.length >= 3 &&
          Number.isFinite(nc[0]) &&
          Number.isFinite(nc[1]) &&
          Number.isFinite(nc[2])
        ) {
          motifPairs.push({ idx, coord: [nc[0], nc[1], nc[2]] });
        }
      }
    }
    let nativeMotifCa: Float32Array | null = null;
    let genMotifRmsd: number | undefined;
    if (motifPairs.length >= 3) {
      nativeMotifCa = new Float32Array(motifPairs.length * 3);
      const genMotifCa = new Float32Array(motifPairs.length * 3);
      for (let k = 0; k < motifPairs.length; k++) {
        const { idx, coord } = motifPairs[k];
        nativeMotifCa[k * 3 + 0] = coord[0];
        nativeMotifCa[k * 3 + 1] = coord[1];
        nativeMotifCa[k * 3 + 2] = coord[2];
        genMotifCa[k * 3 + 0] = ref.caCoords[idx * 3 + 0];
        genMotifCa[k * 3 + 1] = ref.caCoords[idx * 3 + 1];
        genMotifCa[k * 3 + 2] = ref.caCoords[idx * 3 + 2];
      }
      const genAlign = kabschAlign(genMotifCa, nativeMotifCa);
      if (Number.isFinite(genAlign.rmsd)) {
        genMotifRmsd = genAlign.rmsd;
      }
    }
    const hasNonProtein = Boolean(
      ref.diagnostics?.hasLigands || ref.diagnostics?.hasNucleicAcids,
    );
    const designedStructureText =
      ref.cifText && (hasNonProtein || !ref.pdbText) ? ref.cifText : ref.pdbText;

    const rawNativeAtoms = (ref as StructureWithMotifAtoms).nativeMotifAtomCoords;
    const validNativeAtoms: NativeMotifAtomCoord[] = [];
    if (Array.isArray(rawNativeAtoms) && rawNativeAtoms.length >= 3) {
      for (const entry of rawNativeAtoms) {
        const atomName = typeof entry?.atomName === 'string' ? entry.atomName.trim().toUpperCase() : '';
        if (
          entry &&
          Number.isInteger(entry.resIdx) &&
          entry.resIdx >= 0 &&
          entry.resIdx < refLen &&
          atomName.length > 0 &&
          Number.isFinite(entry.x) &&
          Number.isFinite(entry.y) &&
          Number.isFinite(entry.z)
        ) {
          validNativeAtoms.push({
            resIdx: entry.resIdx,
            atomName,
            x: entry.x,
            y: entry.y,
            z: entry.z,
          });
        }
      }
    }
    let genMotifAtomRmsd: number | undefined;
    if (validNativeAtoms.length >= 3) {
      const refTextForAtoms = designedStructureText || ref.cifText || ref.pdbText || '';
      if (refTextForAtoms) {
        const designedHeavy = extractProteinHeavyAtomsByResidueIndex(refTextForAtoms);
        genMotifAtomRmsd = computeMotifHeavyAtomRmsd(designedHeavy, validNativeAtoms);
      }
    }

    return {
      ref,
      refLen,
      refDisplayName: ref.label || ref.id,
      motifPairs,
      nativeMotifCa,
      genMotifRmsd,
      validNativeAtoms,
      genMotifAtomRmsd,
      designedStructureText,
    };
  };

  // Helper: score a single FoldedCandidate against an already-resolved StructureArtifact reference
  const scoreFoldAgainstRef = (
    f: FoldedCandidate,
    ctx: ReturnType<typeof prepareRefScoringContext>,
    lineageOpts: {
      rowLineage: string;
      rowSelfConsistencyValid: boolean;
      rowLineageMismatch: boolean;
      rowReferenceRelation: ReferenceRelation;
      rowReferenceLabel: string;
      rowReferenceMode: 'self-consistency' | 'external-reference' | 'lineage-mismatch';
      parsedMapping: ReturnType<typeof parseReferenceResidueMapping>;
    },
  ): DesignabilityRowWithMotifAtoms => {
    const {
      ref,
      refLen,
      motifPairs,
      nativeMotifCa,
      genMotifRmsd,
      validNativeAtoms,
      genMotifAtomRmsd,
      designedStructureText,
    } = ctx;
    const {
      rowLineage,
      rowSelfConsistencyValid,
      rowLineageMismatch,
      rowReferenceRelation,
      rowReferenceLabel,
      rowReferenceMode,
      parsedMapping,
    } = lineageOpts;
    const foldLen = Math.floor(f.caCoords.length / 3);

    let align;
    let alignedResidueCount: number;
    if (parsedMapping) {
      const nMapped = parsedMapping.pairs.length;
      const foldMappedCa = new Float32Array(nMapped * 3);
      const refMappedCa = new Float32Array(nMapped * 3);
      for (let k = 0; k < nMapped; k++) {
        const [fi, ri] = parsedMapping.pairs[k];
        foldMappedCa[k * 3 + 0] = f.caCoords[fi * 3 + 0];
        foldMappedCa[k * 3 + 1] = f.caCoords[fi * 3 + 1];
        foldMappedCa[k * 3 + 2] = f.caCoords[fi * 3 + 2];
        refMappedCa[k * 3 + 0] = ref.caCoords[ri * 3 + 0];
        refMappedCa[k * 3 + 1] = ref.caCoords[ri * 3 + 1];
        refMappedCa[k * 3 + 2] = ref.caCoords[ri * 3 + 2];
      }
      align = kabschAlign(foldMappedCa, refMappedCa);
      alignedResidueCount = nMapped;
    } else {
      if (f.caCoords.length !== ref.caCoords.length) {
        throw new Error(
          `Length mismatch in designability-scorer: refolded "${f.sequenceId}" has ${foldLen} residues, ` +
            `but designed structure "${ref.label}" has ${refLen} residues.`,
        );
      }
      align = kabschAlign(f.caCoords, ref.caCoords);
      alignedResidueCount = refLen;
    }
    const alignedFoldPdb = applyTransformToPdb(f.pdbText, align.rotation, align.translation);

    let foldResNums: number[] | undefined =
      f.resNums && f.resNums.length === foldLen ? f.resNums : undefined;
    let foldChainIds = f.chainIds && f.chainIds.length === foldLen ? f.chainIds : undefined;
    let foldInsCodes: string[] | undefined;
    if ((!foldResNums || !foldChainIds) && f.pdbText) {
      try {
        const parsedFold = parseStructureText(f.pdbText);
        if (parsedFold.nResidues === foldLen) {
          if (!foldResNums) foldResNums = parsedFold.resNums;
          if (!foldChainIds) foldChainIds = parsedFold.chainIds;
          foldInsCodes = parsedFold.insCodes;
        }
      } catch {
        /* keep f.chainIds without resNums if pdbText is synthetic */
      }
    }
    const geom = computeGeometryStats(f.caCoords, foldChainIds, foldResNums, foldInsCodes);

    // Multi-chain binder & interface evaluation (#8) — only applicable for full 1:1 equal-length complexes
    const isFullOneToOne = !parsedMapping && foldLen === refLen;
    if (isFullOneToOne) {
      const hasStructChainInfo =
        (ref.chainLengths && ref.chainLengths.length > 0) ||
        (ref.chainIds && ref.chainIds.length === refLen);
      const hasFoldChainInfo =
        (f.chainLengths && f.chainLengths.length > 0) ||
        (foldChainIds && foldChainIds.length === foldLen);
      if (hasStructChainInfo && hasFoldChainInfo) {
        const structRanges = resolveScorerChainRanges(
          refLen,
          ref.chainLengths,
          ref.chainIds,
        );
        const foldRanges = resolveScorerChainRanges(foldLen, f.chainLengths, foldChainIds);
        const stoichMatches =
          structRanges.length === foldRanges.length &&
          structRanges.every((sr, idx) => sr.length === foldRanges[idx].length);
        if (!stoichMatches) {
          throw new Error(
            `Chain stoichiometry mismatch in designability-scorer: refolded "${f.sequenceId}" has ${foldRanges.length} chain(s) [${foldRanges.map((r) => r.length).join(', ')}], ` +
              `but reference structure "${ref.label}" has ${structRanges.length} chain(s) [${structRanges.map((r) => r.length).join(', ')}].`,
          );
        }
      }
    }
    const chainRanges = isFullOneToOne
      ? resolveScorerChainRanges(
          refLen,
          ref.chainLengths ?? f.chainLengths,
          ref.chainIds ?? foldChainIds,
        )
      : [];
    let ipae: number | undefined;
    let ipsae: number | undefined;
    let ipsaeMin: number | undefined;
    let binderPlddt: number | undefined;
    let targetPlddt: number | undefined;
    let binderScRmsd: number | undefined;
    let targetAlignedBinderRmsd: number | undefined;

    if (isFullOneToOne && chainRanges.length > 1) {
      const split = splitBinderAndTargetIndices(
        chainRanges,
        ref.chainRoles,
        ref.lockedChains,
      );
      const { binderIndices, targetIndices, explicitRoleSplit } = split;

      if (f.pae && f.pae.length >= refLen * refLen) {
        const paeDim = Math.round(Math.sqrt(f.pae.length));
        if (paeDim >= refLen && paeDim * paeDim === f.pae.length) {
          ipae = computeInterfacePae(
            f.pae,
            paeDim,
            refLen,
            chainRanges,
            explicitRoleSplit ? binderIndices : undefined,
            explicitRoleSplit ? targetIndices : undefined,
          );
        }
      }

      if (explicitRoleSplit) {
        if (f.plddt.length === refLen) {
          if (binderIndices.length > 0) {
            let sum = 0;
            for (const idx of binderIndices) sum += f.plddt[idx];
            const mean = sum / binderIndices.length;
            if (Number.isFinite(mean)) binderPlddt = mean;
          }
          if (targetIndices.length > 0) {
            let sum = 0;
            for (const idx of targetIndices) sum += f.plddt[idx];
            const mean = sum / targetIndices.length;
            if (Number.isFinite(mean)) targetPlddt = mean;
          }
        }

        if (binderIndices.length >= 3) {
          const foldBinderCa = gatherCoordsByIndices(f.caCoords, binderIndices);
          const structBinderCa = gatherCoordsByIndices(ref.caCoords, binderIndices);
          const bAlign = kabschAlign(foldBinderCa, structBinderCa);
          if (Number.isFinite(bAlign.rmsd)) {
            binderScRmsd = bAlign.rmsd;
          }
        }

        if (targetIndices.length >= 3 && binderIndices.length >= 1) {
          const foldTargetCa = gatherCoordsByIndices(f.caCoords, targetIndices);
          const structTargetCa = gatherCoordsByIndices(ref.caCoords, targetIndices);
          const tAlign = kabschAlign(foldTargetCa, structTargetCa);
          if (Number.isFinite(tAlign.rmsd)) {
            const { rotation: R, translation: t } = tAlign;
            let sqSum = 0;
            for (let k = 0; k < binderIndices.length; k++) {
              const off = binderIndices[k] * 3;
              const x = f.caCoords[off + 0];
              const y = f.caCoords[off + 1];
              const z = f.caCoords[off + 2];
              const dx = R[0] * x + R[1] * y + R[2] * z + t[0] - ref.caCoords[off + 0];
              const dy = R[3] * x + R[4] * y + R[5] * z + t[1] - ref.caCoords[off + 1];
              const dz = R[6] * x + R[7] * y + R[8] * z + t[2] - ref.caCoords[off + 2];
              sqSum += dx * dx + dy * dy + dz * dz;
            }
            const tabRmsd = Math.sqrt(sqSum / binderIndices.length);
            if (Number.isFinite(tabRmsd)) {
              targetAlignedBinderRmsd = tabRmsd;
            }
          }
        }
      }
    } else if (
      isFullOneToOne &&
      chainRanges.length === 1 &&
      f.pae &&
      f.pae.length > refLen * refLen
    ) {
      // Holo protein–ligand / protein–DNA interface PAE: average off-diagonal protein↔non-protein blocks
      const paeDim = Math.round(Math.sqrt(f.pae.length));
      if (paeDim > refLen && paeDim * paeDim === f.pae.length) {
        let sum = 0;
        let count = 0;
        for (let i = 0; i < refLen; i++) {
          for (let j = refLen; j < paeDim; j++) {
            const v1 = f.pae[i * paeDim + j];
            if (Number.isFinite(v1)) {
              sum += v1;
              count++;
            }
            const v2 = f.pae[j * paeDim + i];
            if (Number.isFinite(v2)) {
              sum += v2;
              count++;
            }
          }
        }
        if (count > 0) {
          ipae = sum / count;
        }
      }
    }

    const ipsaeChainRanges =
      chainRanges.length > 1
        ? chainRanges
        : resolveScorerChainRanges(foldLen, f.chainLengths, foldChainIds);
    if (ipsaeChainRanges.length > 1 && f.pae && f.pae.length >= foldLen * foldLen) {
      const proteinPae = sliceProteinPae(f.pae, foldLen);
      if (proteinPae) {
        const perResChains = expandChainRangesToIds(ipsaeChainRanges);
        const ipsaeRes = computeIpsae(proteinPae, perResChains, 10);
        if (ipsaeRes) {
          ipsae = ipsaeRes.ipsae;
          ipsaeMin = ipsaeRes.ipsaeMin;
        }
      }
    }
    if (ipsae === undefined && typeof f.ipsae === 'number' && Number.isFinite(f.ipsae)) {
      ipsae = f.ipsae;
      if (typeof f.ipsaeMin === 'number' && Number.isFinite(f.ipsaeMin)) {
        ipsaeMin = f.ipsaeMin;
      }
    }

    let refoldMotifRmsd: number | undefined;
    if (isFullOneToOne && nativeMotifCa && motifPairs.length >= 3) {
      const refoldMotifCa = new Float32Array(motifPairs.length * 3);
      for (let k = 0; k < motifPairs.length; k++) {
        const off = motifPairs[k].idx * 3;
        refoldMotifCa[k * 3 + 0] = f.caCoords[off + 0];
        refoldMotifCa[k * 3 + 1] = f.caCoords[off + 1];
        refoldMotifCa[k * 3 + 2] = f.caCoords[off + 2];
      }
      const refoldAlign = kabschAlign(refoldMotifCa, nativeMotifCa);
      if (Number.isFinite(refoldAlign.rmsd)) {
        refoldMotifRmsd = refoldAlign.rmsd;
      }
    }

    let refoldMotifAtomRmsd: number | undefined;
    if (isFullOneToOne && validNativeAtoms.length >= 3) {
      const foldStructureText =
        f.pdbText || (f as FoldedCandidate & { cifText?: string }).cifText || '';
      if (foldStructureText) {
        const refoldHeavy = extractProteinHeavyAtomsByResidueIndex(foldStructureText);
        refoldMotifAtomRmsd = computeMotifHeavyAtomRmsd(refoldHeavy, validNativeAtoms);
      }
    }

    const confVal = resolveFoldConfidence(f);
    const confidenceAvailable = confVal !== null;
    const effectiveMeanPlddt = confVal !== null ? confVal : Number.NaN;

    const failedGates: string[] = [];
    if (!rowSelfConsistencyValid || rowLineageMismatch) {
      failedGates.push('External reference (not self-consistency)');
    }
    if (!confidenceAvailable) {
      failedGates.push('Confidence unavailable (no pLDDT)');
    } else if (effectiveMeanPlddt < plddtThresh) {
      failedGates.push(`pLDDT ${effectiveMeanPlddt.toFixed(1)} < ${plddtThresh}`);
    }
    if (!Number.isFinite(align.rmsd)) {
      failedGates.push('scRMSD unavailable');
    } else if (align.rmsd >= scThresh) {
      failedGates.push(`scRMSD ${align.rmsd.toFixed(2)} Å ≥ ${scThresh.toFixed(1)} Å`);
    }
    if (binderScRmsd !== undefined && binderScRmsd >= scThresh) {
      failedGates.push(`Binder scRMSD ${binderScRmsd.toFixed(2)} Å ≥ ${scThresh.toFixed(1)} Å`);
    }
    if (binderPlddt !== undefined && binderPlddt < plddtThresh) {
      failedGates.push(`Binder pLDDT ${binderPlddt.toFixed(1)} < ${plddtThresh}`);
    }
    if (targetAlignedBinderRmsd !== undefined && targetAlignedBinderRmsd >= targetAlignedBinderThresh) {
      failedGates.push(
        `Binder RMSD (target-aligned) ${targetAlignedBinderRmsd.toFixed(2)} Å ≥ ${targetAlignedBinderThresh.toFixed(1)} Å`,
      );
    }
    if (ipae !== undefined && ipae >= ipaeThresh) {
      failedGates.push(`iPAE ${ipae.toFixed(1)} Å ≥ ${ipaeThresh.toFixed(1)} Å`);
    }
    if (
      Number.isFinite(ipsaeThresh) &&
      ipsaeThresh > 0 &&
      ipsae !== undefined &&
      ipsae < ipsaeThresh
    ) {
      failedGates.push(`ipSAE ${ipsae.toFixed(2)} < ${ipsaeThresh.toFixed(2)}`);
    }
    if (refoldMotifRmsd !== undefined && refoldMotifRmsd >= motifThresh) {
      failedGates.push(`Motif RMSD ${refoldMotifRmsd.toFixed(2)} Å ≥ ${motifThresh.toFixed(1)} Å`);
    }
    const designable = failedGates.length === 0;

    const designKey = ref.designKey ?? f.designKey;
    const designIndex = ref.designIndex ?? f.designIndex;
    const rowChainLengths =
      f.chainLengths && f.chainLengths.length > 0
        ? f.chainLengths
        : ipsaeChainRanges.length > 1
          ? ipsaeChainRanges.map((r) => r.length)
          : undefined;

    return {
      structureId: ref.id,
      sequenceId: f.sequenceId,
      candidateKey: f.key ?? f.sequenceId,
      ...(designKey ? { designKey } : {}),
      ...(designIndex !== undefined ? { designIndex } : {}),
      ...(f.sequenceKey ? { sequenceKey: f.sequenceKey } : {}),
      ...(f.foldKey ? { foldKey: f.foldKey } : {}),
      ...(rowLineage ? { sourceStructureId: rowLineage } : {}),
      sequence: f.sequence,
      ...(rowChainLengths ? { chainLengths: rowChainLengths } : {}),
      scRmsd: align.rmsd,
      scTm: align.tmScore,
      meanPlddt: effectiveMeanPlddt,
      confidenceAvailable,
      ptm: f.ptm,
      iptm: f.iptm ?? null,
      ...(ipae !== undefined ? { ipae } : {}),
      ...(ipsae !== undefined ? { ipsae } : {}),
      ...(ipsaeMin !== undefined ? { ipsaeMin } : {}),
      ...(binderPlddt !== undefined ? { binderPlddt } : {}),
      ...(targetPlddt !== undefined ? { targetPlddt } : {}),
      ...(binderScRmsd !== undefined ? { binderScRmsd } : {}),
      ...(targetAlignedBinderRmsd !== undefined ? { targetAlignedBinderRmsd } : {}),
      ...(genMotifRmsd !== undefined ? { genMotifRmsd } : {}),
      ...(refoldMotifRmsd !== undefined ? { refoldMotifRmsd } : {}),
      ...(genMotifAtomRmsd !== undefined ? { genMotifAtomRmsd } : {}),
      ...(refoldMotifAtomRmsd !== undefined ? { refoldMotifAtomRmsd } : {}),
      referenceMode: rowReferenceMode,
      referenceRelation: rowReferenceRelation,
      referenceLabel: rowReferenceLabel,
      selfConsistencyValid: rowSelfConsistencyValid,
      lineageMismatch: rowLineageMismatch,
      ...(parsedMapping ? { referenceMappingApplied: true } : {}),
      alignedResidueCount,
      designable,
      failedGates,
      ...(f.msaMode ? { msaMode: f.msaMode } : {}),
      ...(typeof f.msaDepth === 'number' && Number.isFinite(f.msaDepth) ? { msaDepth: f.msaDepth } : {}),
      alignedFoldPdb,
      designedPdb: designedStructureText,
      pae: f.pae,
      plddt: f.plddt,
      geometry: geom,
      foldElapsedMs: f.elapsedMs,
    };
  };

  const collectionItems =
    structure.collection?.items && structure.collection.items.length > 1
      ? structure.collection.items
      : null;

  if (collectionItems) {
    // Multi-backbone collection scoring: resolve EACH fold strictly against its exact declared parent backbone!
    // NEVER fall back to collection.items[0], representative, or length matching.
    const ctxById = new Map<string, ReturnType<typeof prepareRefScoringContext>>();
    const ctxByDesignKey = new Map<string, ReturnType<typeof prepareRefScoringContext>>();
    for (const item of collectionItems) {
      const ctx = prepareRefScoringContext(item);
      ctxById.set(item.id, ctx);
      if (item.designKey) ctxByDesignKey.set(item.designKey, ctx);
    }

    let anyLineageMismatch = false;
    let mismatchedDetail: string | null = null;
    let appliedMappingSpec: ReferenceResidueMapping | undefined;
    const evaluatedDesignIds = new Set<string>();
    const passingDesignIds = new Set<string>();

    const rows: DesignabilityRow[] = folds.folds.map((f) => {
      const foldLen = Math.floor(f.caCoords.length / 3);
      const foldSourceId = normalizeLineageId(f.sourceStructureId);
      const foldDesignKey = normalizeLineageId(f.designKey);
      const byIdCtx = foldSourceId ? ctxById.get(foldSourceId) : undefined;
      const byKeyCtx = foldDesignKey ? ctxByDesignKey.get(foldDesignKey) : undefined;

      let parentError: string | null = null;
      let resolvedCtx: ReturnType<typeof prepareRefScoringContext> | undefined;

      if (!foldSourceId && !foldDesignKey) {
        parentError = `Unlinked parent reference for "${f.sequenceId}": fold declares neither sourceStructureId nor designKey (refusing implicit fallback).`;
      } else if (foldSourceId && foldDesignKey) {
        if (byKeyCtx && byKeyCtx.ref.id !== foldSourceId) {
          parentError = `Cross-parent conflict for "${f.sequenceId}": designKey "${foldDesignKey}" belongs to "${byKeyCtx.ref.id}", conflicting with sourceStructureId "${foldSourceId}".`;
        } else if (byIdCtx && byIdCtx.ref.designKey && byIdCtx.ref.designKey !== foldDesignKey) {
          parentError = `Cross-parent conflict for "${f.sequenceId}": sourceStructureId "${foldSourceId}" belongs to designKey "${byIdCtx.ref.designKey}", conflicting with declared designKey "${foldDesignKey}".`;
        } else if (byIdCtx && byKeyCtx && byIdCtx.ref.id !== byKeyCtx.ref.id) {
          parentError = `Cross-parent conflict for "${f.sequenceId}": sourceStructureId "${foldSourceId}" and designKey "${foldDesignKey}" resolved to different backbones.`;
        } else if (!byIdCtx && !byKeyCtx) {
          parentError = `Missing parent backbone for "${f.sequenceId}": neither sourceStructureId "${foldSourceId}" nor designKey "${foldDesignKey}" exists in the structure collection.`;
        } else if (!byKeyCtx && byIdCtx && byIdCtx.ref.designKey) {
          parentError = `Cross-parent conflict for "${f.sequenceId}": declared designKey "${foldDesignKey}" is absent; refusing to alias to "${byIdCtx.ref.designKey}".`;
        } else {
          resolvedCtx = byKeyCtx ?? byIdCtx;
        }
      } else {
        resolvedCtx = byKeyCtx ?? byIdCtx;
        if (!resolvedCtx) {
          parentError = `Missing parent backbone for "${f.sequenceId}": declared parent "${foldSourceId || foldDesignKey}" was not found in the structure collection.`;
        }
      }

      if (parentError || !resolvedCtx) {
        const errMsg = parentError ?? `Missing parent backbone for "${f.sequenceId}".`;
        anyLineageMismatch = true;
        if (!mismatchedDetail) mismatchedDetail = foldSourceId || foldDesignKey || 'unlinked';
        if (strictLineage) {
          throw new Error(errMsg);
        }
        const confVal = resolveFoldConfidence(f);
        const confidenceAvailable = confVal !== null;
        const effectiveMeanPlddt = confVal !== null ? confVal : Number.NaN;
        const geom = computeGeometryStats(
          f.caCoords,
          f.chainIds && f.chainIds.length === foldLen ? f.chainIds : undefined,
          f.resNums && f.resNums.length === foldLen ? f.resNums : undefined,
        );
        const failedGates: string[] = [errMsg];
        if (!confidenceAvailable) {
          failedGates.push('Confidence unavailable (no pLDDT)');
        } else if (effectiveMeanPlddt < plddtThresh) {
          failedGates.push(`pLDDT ${effectiveMeanPlddt.toFixed(1)} < ${plddtThresh}`);
        }
        return {
          structureId: foldSourceId || structure.id,
          sequenceId: f.sequenceId,
          candidateKey: f.key ?? f.sequenceId,
          ...(foldDesignKey ? { designKey: foldDesignKey } : {}),
          ...(f.designIndex !== undefined ? { designIndex: f.designIndex } : {}),
          ...(f.sequenceKey ? { sequenceKey: f.sequenceKey } : {}),
          ...(f.foldKey ? { foldKey: f.foldKey } : {}),
          ...(foldSourceId ? { sourceStructureId: foldSourceId } : {}),
          sequence: f.sequence,
          ...(f.chainLengths && f.chainLengths.length > 0 ? { chainLengths: f.chainLengths } : {}),
          scRmsd: Number.NaN,
          scTm: Number.NaN,
          meanPlddt: effectiveMeanPlddt,
          confidenceAvailable,
          ptm: f.ptm,
          iptm: f.iptm ?? null,
          ...(typeof f.ipsae === 'number' && Number.isFinite(f.ipsae) ? { ipsae: f.ipsae } : {}),
          ...(typeof f.ipsaeMin === 'number' && Number.isFinite(f.ipsaeMin) ? { ipsaeMin: f.ipsaeMin } : {}),
          referenceMode: 'lineage-mismatch',
          referenceRelation: foldSourceId || foldDesignKey ? 'external-mismatch' : 'external-unlinked',
          referenceLabel: `Invalid parent reference · ${errMsg}`,
          selfConsistencyValid: false,
          lineageMismatch: true,
          alignedResidueCount: 0,
          designable: false,
          failedGates,
          ...(f.msaMode ? { msaMode: f.msaMode } : {}),
          ...(typeof f.msaDepth === 'number' && Number.isFinite(f.msaDepth) ? { msaDepth: f.msaDepth } : {}),
          alignedFoldPdb: f.pdbText,
          designedPdb: '',
          pae: f.pae,
          plddt: f.plddt,
          geometry: geom,
          foldElapsedMs: f.elapsedMs,
        };
      }

      const parsedMapping = parseReferenceResidueMapping(rawMappingParam, foldLen, resolvedCtx.refLen);
      if (parsedMapping && !appliedMappingSpec) {
        appliedMappingSpec = parsedMapping.spec;
      }
      const rowSelfConsistencyValid = !parsedMapping && !allowExternalReference;
      const rowReferenceRelation: ReferenceRelation = parsedMapping
        ? 'external-mapped'
        : rowSelfConsistencyValid
          ? 'self-consistency'
          : 'external-unlinked';
      const rowReferenceMode: 'self-consistency' | 'external-reference' | 'lineage-mismatch' =
        rowSelfConsistencyValid ? 'self-consistency' : 'external-reference';
      const rowReferenceLabel =
        rowReferenceRelation === 'self-consistency'
          ? `Self-consistency against ${resolvedCtx.refDisplayName}`
          : parsedMapping
            ? `Comparison against ${resolvedCtx.refDisplayName} · Mapped ${parsedMapping.pairs.length} residues`
            : `Comparison against ${resolvedCtx.refDisplayName} · External reference mode enabled`;

      const row = scoreFoldAgainstRef(f, resolvedCtx, {
        rowLineage: resolvedCtx.ref.id,
        rowSelfConsistencyValid,
        rowLineageMismatch: false,
        rowReferenceRelation,
        rowReferenceLabel,
        rowReferenceMode,
        parsedMapping,
      });
      evaluatedDesignIds.add(resolvedCtx.ref.id);
      if (row.designable) passingDesignIds.add(resolvedCtx.ref.id);
      return row;
    });

    rows.sort((a, b) => {
      const af = Number.isFinite(a.scRmsd);
      const bf = Number.isFinite(b.scRmsd);
      if (af && bf) return a.scRmsd - b.scRmsd;
      if (af) return -1;
      if (bf) return 1;
      return 0;
    });

    const passed = rows.filter((r) => r.designable).length;
    const topSelfConsistencyValid =
      !appliedMappingSpec &&
      !allowExternalReference &&
      !anyLineageMismatch &&
      rows.length > 0 &&
      rows.every((r) => r.selfConsistencyValid === true);

    let lineageStatus: DesignabilityMetrics['lineageStatus'];
    let referenceMode: DesignabilityMetrics['referenceMode'];
    let referenceRelation: ReferenceRelation;
    let referenceLabel: string;
    let lineageWarning: string | undefined;

    if (appliedMappingSpec) {
      lineageStatus = 'external-mapped';
      referenceMode = 'external-reference';
      referenceRelation = 'external-mapped';
      referenceLabel = `Comparison across ${evaluatedDesignIds.size} backbones · Mapped ${appliedMappingSpec.pairs?.length ?? 0} residues`;
    } else if (anyLineageMismatch) {
      lineageStatus = 'mismatch';
      referenceMode = 'lineage-mismatch';
      referenceRelation = 'external-mismatch';
      referenceLabel = `Multi-design parent mismatch (${mismatchedDetail ?? 'unknown'})`;
      lineageWarning =
        `Multi-design parent mismatch: one or more folds referenced a missing, unlinked, or conflicting parent backbone (${mismatchedDetail ?? 'unknown'}). ` +
        `Self-consistency pass status is disabled on mismatched rows.`;
    } else if (allowExternalReference) {
      lineageStatus = 'external-unmapped';
      referenceMode = 'external-reference';
      referenceRelation = 'external-unlinked';
      referenceLabel = `External reference comparison across ${evaluatedDesignIds.size} backbones`;
    } else {
      lineageStatus = 'matched';
      referenceMode = 'self-consistency';
      referenceRelation = 'self-consistency';
      referenceLabel = `Self-consistency across ${evaluatedDesignIds.size} backbones`;
    }

    const threshHash = fnv1a(
      canonicalJsonStringify({
        scThresh,
        plddtThresh,
        motifThresh,
        ipaeThresh,
        ipsaeThresh,
        targetAlignedBinderThresh,
        allowExternalReference,
        strictLineage,
        mapping: appliedMappingSpec ?? null,
        collectionId: structure.collection?.id ?? null,
      }),
    );
    const bestRow = (rows[0] ?? null) as DesignabilityRowWithMotifAtoms | null;
    const metrics: DesignabilityMetricsWithMotifAtoms = {
      kind: 'DesignabilityMetrics',
      id: `metrics-${structure.id}-${folds.id}-${threshHash}`,
      rows,
      best: bestRow,
      passRate: rows.length > 0 ? passed / rows.length : 0,
      referenceStructureId: structure.id,
      foldParentStructureId: folds.parentStructureId || 'multi-design',
      referenceMode,
      referenceRelation,
      referenceLabel,
      selfConsistencyValid: topSelfConsistencyValid,
      lineageStatus,
      lineageMismatch: anyLineageMismatch,
      multiDesign: true,
      evaluatedDesignCount: evaluatedDesignIds.size,
      designHitRate:
        evaluatedDesignIds.size > 0
          ? passingDesignIds.size / evaluatedDesignIds.size
          : null,
      ...(bestRow?.genMotifAtomRmsd !== undefined
        ? { genMotifAtomRmsd: bestRow.genMotifAtomRmsd }
        : {}),
      ...(bestRow?.refoldMotifAtomRmsd !== undefined
        ? { refoldMotifAtomRmsd: bestRow.refoldMotifAtomRmsd }
        : {}),
      ...(lineageWarning ? { lineageWarning } : {}),
      ...(appliedMappingSpec ? { referenceMapping: appliedMappingSpec } : {}),
      ...(folds.collectionKind ? { collectionKind: folds.collectionKind } : {}),
    };
    return { metrics };
  }

  // Single-backbone legacy path (100% preserved semantics)
  const refCtx = prepareRefScoringContext(structure);
  const structLen = refCtx.refLen;

  const foldSetLineages = new Set<string>(
    [
      normalizeLineageId(folds.parentStructureId),
      normalizeLineageId(folds.sourceStructureId),
      normalizeLineageId(upstreamLineageHint),
    ].filter(Boolean),
  );
  const foldSetHasConflict = foldSetLineages.size > 1;
  const foldSetNonRef = [...foldSetLineages].find((id) => id !== structure.id);
  const foldSetLineage = foldSetNonRef ?? [...foldSetLineages][0] ?? '';

  let anyLineageMismatch = false;
  let mismatchedSourceId: string | null = null;
  let appliedMappingSpec: ReferenceResidueMapping | undefined;
  const refDisplayName = refCtx.refDisplayName;

  const rows: DesignabilityRow[] = folds.folds.map((f) => {
    const foldLen = Math.floor(f.caCoords.length / 3);
    const parsedMapping = parseReferenceResidueMapping(rawMappingParam, foldLen, structLen);
    if (parsedMapping && !appliedMappingSpec) {
      appliedMappingSpec = parsedMapping.spec;
    }

    const rowLineageSet = new Set<string>(foldSetLineages);
    const perFoldLineage = normalizeLineageId(f.sourceStructureId);
    if (perFoldLineage) rowLineageSet.add(perFoldLineage);
    const rowHasConflict = rowLineageSet.size > 1;
    const rowNonRef = [...rowLineageSet].find((id) => id !== structure.id);
    const rowLineage = rowNonRef ?? [...rowLineageSet][0] ?? '';

    const hasExplicitExternal = Boolean(parsedMapping) || allowExternalReference;
    const rowSelfConsistencyValid =
      !parsedMapping &&
      !allowExternalReference &&
      !rowHasConflict &&
      rowLineageSet.size === 1 &&
      rowLineage === structure.id;
    const rowHasLineageDisagreement =
      rowHasConflict || (Boolean(rowLineage) && rowLineage !== structure.id);
    const rowLineageMismatch = !hasExplicitExternal && rowHasLineageDisagreement;
    if (rowLineageMismatch) {
      anyLineageMismatch = true;
      if (!mismatchedSourceId) {
        mismatchedSourceId = rowHasConflict ? [...rowLineageSet].join(' vs ') : rowLineage;
      }
    }

    const rowReferenceRelation: ReferenceRelation = parsedMapping
      ? 'external-mapped'
      : rowSelfConsistencyValid
        ? 'self-consistency'
        : rowHasLineageDisagreement
          ? 'external-mismatch'
          : 'external-unlinked';

    const rowOriginLabel = rowHasConflict ? [...rowLineageSet].join(' / ') : rowLineage;
    const rowReferenceLabel =
      rowReferenceRelation === 'self-consistency'
        ? `Self-consistency against ${refDisplayName}`
        : rowReferenceRelation === 'external-mapped'
          ? `Comparison against ${refDisplayName} · Mapped ${parsedMapping!.pairs.length} residues${rowOriginLabel ? ` · Fold originated from ${rowOriginLabel}` : ''}`
          : rowReferenceRelation === 'external-mismatch'
            ? `Comparison against ${refDisplayName} · Fold originated from ${rowOriginLabel}`
            : allowExternalReference && rowOriginLabel
              ? `Comparison against ${refDisplayName} · External reference mode enabled`
              : `Comparison against ${refDisplayName} · No parent structure recorded`;

    const rowReferenceMode: 'self-consistency' | 'external-reference' | 'lineage-mismatch' =
      rowLineageMismatch
        ? 'lineage-mismatch'
        : rowSelfConsistencyValid
          ? 'self-consistency'
          : 'external-reference';

    return scoreFoldAgainstRef(f, refCtx, {
      rowLineage,
      rowSelfConsistencyValid,
      rowLineageMismatch,
      rowReferenceRelation,
      rowReferenceLabel,
      rowReferenceMode,
      parsedMapping,
    });
  });

  // Also detect top-level lineage mismatch even if `folds.folds` is empty
  const topHasLineageDisagreement =
    foldSetHasConflict || (Boolean(foldSetLineage) && foldSetLineage !== structure.id);
  if (
    !anyLineageMismatch &&
    !appliedMappingSpec &&
    !allowExternalReference &&
    topHasLineageDisagreement
  ) {
    anyLineageMismatch = true;
    mismatchedSourceId = foldSetHasConflict
      ? [...foldSetLineages].join(' vs ')
      : foldSetLineage;
  }

  const topSelfConsistencyValid =
    !appliedMappingSpec &&
    !allowExternalReference &&
    !anyLineageMismatch &&
    !foldSetHasConflict &&
    foldSetLineages.size === 1 &&
    foldSetLineage === structure.id &&
    rows.every((r) => r.selfConsistencyValid === true);

  let lineageStatus: DesignabilityMetrics['lineageStatus'];
  let referenceMode: DesignabilityMetrics['referenceMode'];
  let referenceRelation: ReferenceRelation;
  let referenceLabel: string;
  let lineageWarning: string | undefined;

  if (appliedMappingSpec) {
    lineageStatus = 'external-mapped';
    referenceMode = 'external-reference';
    referenceRelation = 'external-mapped';
    referenceLabel = `Comparison against ${refDisplayName} · Mapped ${appliedMappingSpec.pairs?.length ?? 0} residues${foldSetLineage ? ` · Fold originated from ${foldSetLineage}` : ''}`;
  } else if (anyLineageMismatch) {
    lineageStatus = 'mismatch';
    referenceMode = 'lineage-mismatch';
    referenceRelation = 'external-mismatch';
    referenceLabel = `Comparison against ${refDisplayName} · Fold originated from ${mismatchedSourceId}`;
    lineageWarning =
      `Reference lineage mismatch: folds descended from structure "${mismatchedSourceId}" ` +
      `were wired to different reference structure "${structure.id}" ("${structure.label}"). ` +
      `Self-consistency pass status is disabled; wire the originating structure or configure allowExternalReference / residueMapping for external-reference comparison.`;
    if (strictLineage) {
      throw new Error(lineageWarning);
    }
  } else if (allowExternalReference) {
    lineageStatus = 'external-unmapped';
    referenceMode = 'external-reference';
    referenceRelation = topHasLineageDisagreement
      ? 'external-mismatch'
      : 'external-unlinked';
    referenceLabel = topHasLineageDisagreement
      ? `Comparison against ${refDisplayName} · Fold originated from ${foldSetHasConflict ? [...foldSetLineages].join(' / ') : foldSetLineage}`
      : foldSetLineage
        ? `Comparison against ${refDisplayName} · External reference mode enabled`
        : `Comparison against ${refDisplayName} · No parent structure recorded`;
    if (foldSetLineage && foldSetLineage !== structure.id) {
      lineageWarning = `External-reference comparison enabled without custom residueMapping: comparing folds from "${foldSetLineage}" against "${structure.id}" by positional index.`;
    }
  } else if (!foldSetLineage) {
    lineageStatus = 'untracked';
    referenceMode = 'external-reference';
    referenceRelation = 'external-unlinked';
    referenceLabel = `Comparison against ${refDisplayName} · No parent structure recorded`;
    lineageWarning =
      `Unverified reference correspondence: incoming folds have no recorded parent structure (sequence-only source). ` +
      `Self-consistency pass status is disabled; supply residueMapping or allowExternalReference to make external-reference correspondence explicit.`;
  } else {
    lineageStatus = 'matched';
    referenceMode = 'self-consistency';
    referenceRelation = 'self-consistency';
    referenceLabel = `Self-consistency against ${refDisplayName}`;
  }

  rows.sort((a, b) => {
    const af = Number.isFinite(a.scRmsd);
    const bf = Number.isFinite(b.scRmsd);
    if (af && bf) return a.scRmsd - b.scRmsd;
    if (af) return -1;
    if (bf) return 1;
    return 0;
  });
  const passed = rows.filter((r) => r.designable).length;
  const threshHash = fnv1a(
    canonicalJsonStringify({
      scThresh,
      plddtThresh,
      motifThresh,
      ipaeThresh,
      ipsaeThresh,
      targetAlignedBinderThresh,
      allowExternalReference,
      strictLineage,
      mapping: appliedMappingSpec ?? null,
    }),
  );
  const bestRow = (rows[0] ?? null) as DesignabilityRowWithMotifAtoms | null;
  const metrics: DesignabilityMetricsWithMotifAtoms = {
    kind: 'DesignabilityMetrics',
    id: `metrics-${structure.id}-${folds.id}-${threshHash}`,
    rows,
    best: bestRow,
    passRate: rows.length > 0 ? passed / rows.length : 0,
    referenceStructureId: structure.id,
    foldParentStructureId: foldSetLineage || folds.parentStructureId || 'none',
    referenceMode,
    referenceRelation,
    referenceLabel,
    selfConsistencyValid: topSelfConsistencyValid,
    lineageStatus,
    lineageMismatch: anyLineageMismatch,
    ...(bestRow?.genMotifAtomRmsd !== undefined
      ? { genMotifAtomRmsd: bestRow.genMotifAtomRmsd }
      : {}),
    ...(bestRow?.refoldMotifAtomRmsd !== undefined
      ? { refoldMotifAtomRmsd: bestRow.refoldMotifAtomRmsd }
      : {}),
    ...(lineageWarning ? { lineageWarning } : {}),
    ...(appliedMappingSpec ? { referenceMapping: appliedMappingSpec } : {}),
    ...(folds.collectionKind ? { collectionKind: folds.collectionKind } : {}),
  };
  return { metrics };
}
