// Two-Layer Biological Capability Schema:
// Layer 1: Static graph/wire capability contracts (catches monomer-vs-multimer, ligand/DNA stripping,
//          pure-DNA sequence input, and token-ceiling risks before clicking Run).
// Layer 2: Runtime pre-dispatch artifact validation (called in scheduler.ts before launching GPU workers).

import { resolveEffectiveMaxSequences } from './batch-flow';
import { esmfold2Chains, esmfold2ParamIssue } from './localfold-esmfold2';
import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from './registry';
import { BENCHMARK_TARGETS, getBenchmarkById, type BenchmarkTargetMeta } from './benchmarks';
import { planLocalFoldFanout } from './work-plan';
import {
  isBareLengthContig,
  isCrystallizationBuffer,
  parseResidueRangeTokens,
  resolveLocalFoldEffectiveConfig,
} from './conditioning';
import {
  BUFFER_OR_SOLVENT_RESNAMES,
  classifyHetatmCompId,
  evaluateProteinOnlyStructurePolicy,
  inspectUnprotonatedOrganicLigands,
  isSilentProteinOnlyHetatm,
  MODIFIED_AMINO_ACID_RESNAMES,
  MONOATOMIC_ION_RESNAMES,
  NUCLEIC_ACID_RESNAMES,
  STANDARD_AA_RESNAMES,
  tokenizeCifLine,
} from '$lib/protein-ops/hetatm-policy';
import type { NumberedChainInfo } from './mpnn-spec';
import type { CollectionMember } from './collections';
import {
  BACKBONE_MASK_CA,
  BACKBONE_MASK_C,
  BACKBONE_MASK_N,
  cropRawAllAtomStructure,
  getMissingBackboneResidues,
  parseStructureText,
  type StructureParseDiagnostics,
} from '$lib/protein-ops/pdb';
import type {
  BioCompatibilityIssue,
  BioEntityKind,
  CanonicalArtifact,
  CollectionKind,
  ExcludedCandidateRecord,
  FoldedCandidate,
  RepresentativeSelection,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
  WorkflowNode,
} from './types';

interface CachedStructureBioSummary {
  nResidues: number;
  diagnostics: StructureParseDiagnostics;
  missingBackboneCount: number;
  isCaOnly: boolean;
  chainIds: string[];
  resNums: number[];
  numberedChains: NumberedChainInfo[];
  headerMetaText: string;
}

const KNOWN_NON_ANTIBODY_PDBS = new Set(['1UBQ', '1CTT', '1LYZ']);

function extractStructureHeaderMetaText(rawText: string): string {
  if (!rawText) return '';
  const atomSiteIdx = rawText.indexOf('_atom_site.');
  if (atomSiteIdx > 0) {
    return rawText.slice(0, Math.min(atomSiteIdx, 32_768));
  }
  return rawText.slice(0, 8192);
}

function buildNumberedChains(
  sourceResidues: ReadonlyArray<{
    chainId?: string;
    projectedChainId?: string;
    authChainId?: string;
    resNum: number;
    insCode?: string;
  }> | undefined,
  sequence: string | undefined,
): NumberedChainInfo[] {
  if (!sourceResidues?.length || !sequence) return [];
  const byChain = new Map<string, Array<{ number: number; insertion: string; aa1: string }>>();
  for (let i = 0; i < sourceResidues.length; i++) {
    const r = sourceResidues[i];
    const id = String(r.projectedChainId ?? r.chainId ?? r.authChainId ?? 'A').trim() || 'A';
    let residues = byChain.get(id);
    if (!residues) {
      residues = [];
      byChain.set(id, residues);
    }
    residues.push({
      number: r.resNum,
      insertion: String(r.insCode ?? '').trim(),
      aa1: (sequence[i] ?? 'X').toUpperCase(),
    });
  }
  return [...byChain.entries()].map(([id, residues]) => ({ id, residues }));
}

function scoreChainAntibodyHallmarksLocal(chain: NumberedChainInfo): {
  kabatVH: number;
  kabatVL: number;
  imgt: number;
} {
  const byNum = new Map<number, string>();
  let hasKabatHIns = false;
  let hasKabatLIns = false;
  let hasImgtIns = false;
  for (const r of chain.residues) {
    const ins = r.insertion.toUpperCase();
    if (!ins && !byNum.has(r.number)) byNum.set(r.number, r.aa1.toUpperCase());
    if (ins) {
      if ((r.number === 82 || r.number === 52 || r.number === 100) && ins >= 'A' && ins <= 'Z') {
        hasKabatHIns = true;
      }
      if ((r.number === 27 || r.number === 95) && ins >= 'A' && ins <= 'Z') {
        hasKabatLIns = true;
      }
      if ((r.number === 111 || r.number === 112) && ins >= 'A' && ins <= 'Z') {
        hasImgtIns = true;
      }
    }
  }
  const kabatVH =
    (byNum.get(22) === 'C' ? 1 : 0) +
    (byNum.get(36) === 'W' ? 1 : 0) +
    (byNum.get(92) === 'C' ? 1 : 0) +
    (byNum.get(103) === 'W' ? 1 : 0) +
    (hasKabatHIns ? 2 : 0);
  const kabatVL =
    (byNum.get(23) === 'C' ? 1 : 0) +
    (byNum.get(35) === 'W' ? 1 : 0) +
    (byNum.get(88) === 'C' ? 1 : 0) +
    (byNum.get(98) === 'F' || byNum.get(98) === 'W' ? 1 : 0) +
    (hasKabatLIns ? 2 : 0);
  const imgt =
    (byNum.get(23) === 'C' ? 1 : 0) +
    (byNum.get(41) === 'W' ? 1 : 0) +
    (byNum.get(104) === 'C' ? 1 : 0) +
    (byNum.get(118) === 'W' || byNum.get(118) === 'F' ? 1 : 0) +
    (hasImgtIns ? 2 : 0);
  return { kabatVH, kabatVL, imgt };
}

export function hasAntibodyChainHallmarks(chains: readonly NumberedChainInfo[]): boolean {
  for (const chain of chains) {
    const s = scoreChainAntibodyHallmarksLocal(chain);
    if (Math.max(s.kabatVH, s.kabatVL, s.imgt) >= 3) {
      return true;
    }
  }
  return false;
}

const STRUCTURE_DIAG_CACHE_LIMIT = 32;
const structureDiagCache = new Map<string, CachedStructureBioSummary>();

function fastTextHash(text: string): string {
  const len = text.length;
  let h = 0x811c9dc5;
  if (len <= 262_144) {
    for (let i = 0; i < len; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
  } else {
    // Sample head, tail, and uniform stride for multi-MB structures to keep main-thread O(1)
    const edge = 4096;
    for (let i = 0; i < edge; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    const step = Math.max(1, Math.floor((len - edge * 2) / 32_768));
    for (let i = edge; i < len - edge; i += step) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    for (let i = len - edge; i < len; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function getCachedStructureBioSummary(rawText: string, cropSpec = ''): CachedStructureBioSummary {
  const key = `${rawText.length}:${fastTextHash(rawText)}:${cropSpec}`;
  const hit = structureDiagCache.get(key);
  if (hit) {
    // Refresh LRU order
    structureDiagCache.delete(key);
    structureDiagCache.set(key, hit);
    return hit;
  }
  const effectiveText = cropSpec ? cropRawAllAtomStructure(rawText, cropSpec).text : rawText;
  const parsed = parseStructureText(effectiveText);
  const missing = getMissingBackboneResidues(parsed, ['N', 'CA', 'C', 'O'], {
    allowReconstructedO: true,
  });
  const isCaOnly =
    parsed.nResidues > 0 &&
    parsed.atomPresence.every((p) => p.hasCA && !p.hasN && !p.hasC);
  const summary: CachedStructureBioSummary = {
    nResidues: parsed.nResidues,
    diagnostics: parsed.diagnostics,
    missingBackboneCount: missing.length,
    isCaOnly,
    chainIds: parsed.chainIds,
    resNums: parsed.resNums,
    numberedChains: buildNumberedChains(parsed.sourceResidues, parsed.sequence),
    headerMetaText: extractStructureHeaderMetaText(rawText),
  };
  if (structureDiagCache.size >= STRUCTURE_DIAG_CACHE_LIMIT) {
    const oldest = structureDiagCache.keys().next().value as string | undefined;
    if (oldest !== undefined) structureDiagCache.delete(oldest);
  }
  structureDiagCache.set(key, summary);
  return summary;
}

function inspectArtifactBackboneCoverage(struct: StructureArtifact): {
  missingBackboneCount: number;
  isCaOnly: boolean;
  completeBackboneCount: number;
} {
  if (struct.atomPresence && struct.atomPresence.length === struct.length) {
    let missingCount = 0;
    let completeCount = 0;
    let allCaOnly = struct.length > 0;
    for (let i = 0; i < struct.length; i++) {
      const p = struct.atomPresence[i];
      if (!p.hasN || !p.hasCA || !p.hasC || (!p.hasO && !p.oReconstructed)) {
        missingCount++;
      } else {
        completeCount++;
      }
      if (!p.hasCA || p.hasN || p.hasC) {
        allCaOnly = false;
      }
    }
    const isCaOnly =
      allCaOnly || (struct.atomGranularity === 'ca-only' && completeCount === 0 && struct.length > 0);
    return {
      missingBackboneCount: missingCount,
      isCaOnly,
      completeBackboneCount: completeCount,
    };
  }
  if (struct.observedBackboneMask && struct.observedBackboneMask.length === struct.length) {
    const required = BACKBONE_MASK_N | BACKBONE_MASK_CA | BACKBONE_MASK_C;
    let missingCount = 0;
    let completeCount = 0;
    let allCaOnly = struct.length > 0;
    for (let i = 0; i < struct.length; i++) {
      const m = struct.observedBackboneMask[i];
      if ((m & required) !== required) {
        missingCount++;
      } else {
        completeCount++;
      }
      if ((m & required) !== BACKBONE_MASK_CA) allCaOnly = false;
    }
    const isCaOnly =
      allCaOnly || (struct.atomGranularity === 'ca-only' && completeCount === 0 && struct.length > 0);
    return {
      missingBackboneCount: missingCount,
      isCaOnly,
      completeBackboneCount: completeCount,
    };
  }
  const rawStructureText = struct.pdbText || struct.cifText || '';
  if (rawStructureText && struct.length > 0) {
    try {
      const summary = getCachedStructureBioSummary(rawStructureText);
      if (summary.nResidues === struct.length) {
        const completeBackboneCount = Math.max(0, summary.nResidues - summary.missingBackboneCount);
        const isCaOnly =
          summary.isCaOnly ||
          (struct.atomGranularity === 'ca-only' && completeBackboneCount === 0);
        return {
          missingBackboneCount: summary.missingBackboneCount,
          isCaOnly,
          completeBackboneCount,
        };
      }
    } catch {
      /* ignore synthetic pdbText */
    }
  }
  if (struct.atomGranularity === 'ca-only' && struct.length > 0) {
    return {
      missingBackboneCount: struct.length,
      isCaOnly: true,
      completeBackboneCount: 0,
    };
  }
  return {
    missingBackboneCount: 0,
    isCaOnly: false,
    completeBackboneCount: struct.length,
  };
}

export interface FoldWorkItemSelection {
  maxSequences: number;
  maxTotalFolds: number;
  selected: SequenceCandidate[];
  excludedCandidates: ExcludedCandidateRecord[];
  totalCandidates: number;
}

/**
 * Pure work-item resolver shared by preflight validation (`validateRuntimeNodeInputs`),
 * Studio input preview, and `localfold.worker.ts` so all three inspect the EXACT
 * identical candidate slice (including multi-backbone round-robin fan-out) and
 * surface excluded candidates accurately.
 */
export function selectFoldWorkItems(
  seqSet: SequenceSet | undefined | null,
  params?: Record<string, unknown>,
): FoldWorkItemSelection {
  const resolvedCap = resolveEffectiveMaxSequences('localfold-af2', params);
  const maxSequences = resolvedCap.limit;
  const rawTotal = Number(params?.maxTotalFolds ?? 16);
  const maxTotalFolds = Math.max(1, Math.min(64, Math.round(Number.isFinite(rawTotal) ? rawTotal : 16)));
  const all = seqSet?.candidates ?? [];

  // Mirror localfold.worker.ts multi-backbone detection so preflight/preview never
  // falsely warn about excluded candidates on multi-backbone campaigns.
  const distinctAnchors = new Set<string>();
  for (const cand of all) {
    const anchor = (cand.designKey ?? cand.sourceStructureId ?? '').trim();
    if (anchor && anchor.toLowerCase() !== 'none') {
      distinctAnchors.add(anchor);
    }
  }

  if (distinctAnchors.size > 1) {
    const groupOrder = new Map<string, number>();
    const origIndexByKey = new Map<string, number>();
    const origIdByKey = new Map<string, string>();
    const usedKeys = new Set<string>();
    const members: CollectionMember<SequenceCandidate>[] = all.map((cand, idx) => {
      const anchor = (cand.designKey ?? cand.sourceStructureId ?? '').trim();
      const hasDesign = anchor.length > 0 && anchor.toLowerCase() !== 'none';
      const designKey = hasDesign ? anchor : undefined;
      let designIndex = cand.designIndex;
      if (designKey !== undefined && designIndex === undefined) {
        if (!groupOrder.has(designKey)) groupOrder.set(designKey, groupOrder.size);
        designIndex = groupOrder.get(designKey);
      }
      let seqKey = (cand.sequenceKey ?? cand.key ?? cand.id).trim() || `seq-${idx}`;
      if (usedKeys.has(seqKey)) seqKey = `${seqKey}::${idx}`;
      usedKeys.add(seqKey);
      origIndexByKey.set(seqKey, idx);
      origIdByKey.set(seqKey, cand.id);
      const ordinal =
        typeof cand.sampleIndex === 'number' && Number.isInteger(cand.sampleIndex) && cand.sampleIndex >= 0
          ? cand.sampleIndex
          : idx;
      return {
        key: seqKey,
        ordinal,
        memberKind: 'SequenceCandidate',
        lineage: {
          ...(designKey ? { designKey, designIndex } : {}),
          sequenceKey: seqKey,
          sequenceIndex: ordinal,
          parentKeys: designKey ? [designKey] : [],
        },
        payload: cand,
        summary: {
          label: cand.id,
          residueCount: cand.sequence.length,
          score: cand.score,
          globalScore: cand.globalScore,
        },
      };
    });

    const plan = planLocalFoldFanout(members, {
      foldModelId: String(params?.model ?? 'monomer'),
      baseSeed: Number(params?.seed ?? 0),
      perDesignTopK: maxSequences,
      samplesPerSequence: 1,
      maxFolds: maxTotalFolds,
      ranking: 'mpnn-score',
    });

    const selected = plan.selected.map((p) => p.sequenceMember.payload!);
    const excludedCandidates: ExcludedCandidateRecord[] = plan.excluded.map((ex, exIdx) => {
      const refKey = ex.lineage?.sequenceKey ?? ex.key;
      return {
        key: ex.key,
        id: origIdByKey.get(refKey) ?? ex.key,
        index: origIndexByKey.get(refKey) ?? selected.length + exIdx,
        reason:
          ex.reason ??
          `Excluded by LocalFold fan-out policy (maxSequences=${maxSequences}, maxTotalFolds=${maxTotalFolds})`,
      };
    });
    return {
      maxSequences,
      maxTotalFolds,
      selected,
      excludedCandidates,
      totalCandidates: all.length,
    };
  }

  const selected = all.slice(0, maxSequences);
  const excludedCandidates: ExcludedCandidateRecord[] = all
    .slice(maxSequences)
    .map((cand, offset) => {
      const index = maxSequences + offset;
      return {
        key: cand.key ?? cand.id,
        id: cand.id,
        index,
        reason: `Excluded by maxSequences=${maxSequences} (candidate #${index + 1} of ${all.length})`,
      };
    });
  return {
    maxSequences,
    maxTotalFolds,
    selected,
    excludedCandidates,
    totalCandidates: all.length,
  };
}

/**
 * Resolve the effective numeric confidence for a folded candidate, returning `null`
 * when the predictor emitted a confidence-less sample.
 */
export function resolveFoldConfidence(cand: FoldedCandidate): number | null {
  if (cand.confidenceAvailable === false) return null;
  if (cand.confidence === null) return null;
  if (typeof cand.confidence === 'number') {
    return Number.isFinite(cand.confidence) ? cand.confidence : null;
  }
  return Number.isFinite(cand.meanPlddt) ? cand.meanPlddt : null;
}

/**
 * Deterministic representative selector for publishing a single `StructureArtifact`
 * from a multi-member `FoldedStructure` collection. Records the exact key, criterion,
 * and total member count, and falls back deterministically when confidence is unavailable.
 */
export function selectFoldRepresentative(
  folds: readonly FoldedCandidate[],
  opts: { collectionKind?: CollectionKind } = {},
): { fold: FoldedCandidate; selection: RepresentativeSelection } | null {
  if (folds.length === 0) return null;

  let bestIdx = -1;
  let bestScore = -Infinity;
  for (let i = 0; i < folds.length; i++) {
    const score = resolveFoldConfidence(folds[i]);
    if (score !== null && score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }

  if (bestIdx >= 0) {
    const best = folds[bestIdx];
    return {
      fold: best,
      selection: {
        selectedKey: best.key ?? best.sequenceId,
        criterion: 'highest-mean-plddt',
        totalMembers: folds.length,
        selectedIndex: bestIdx,
        scoreValue: bestScore,
      },
    };
  }

  // Deterministic confidence-less fallback
  const fallbackIdx = opts.collectionKind === 'ordered-frames' ? folds.length - 1 : 0;
  const fallback = folds[fallbackIdx];
  const criterion =
    opts.collectionKind === 'ordered-frames'
      ? 'final-frame-fallback (no confidence score)'
      : 'first-member-fallback (no confidence score)';
  return {
    fold: fallback,
    selection: {
      selectedKey: fallback.key ?? fallback.sequenceId,
      criterion,
      totalMembers: folds.length,
      selectedIndex: fallbackIdx,
      scoreValue: null,
    },
  };
}

export interface NodeBioProfile {
  entities: Set<BioEntityKind>;
  stoichiometry: 'monomer' | 'multimer';
  chainCount: number;
  /** Number of protein chains only (excluding DNA/RNA chains stripped by inverse-folding stages). */
  proteinChainCount?: number;
  estimatedTokens: number;
  looksLikeDnaSequence?: boolean;
  /** Upstream non-protein entities stripped by an intermediate protein-only stage (e.g. ProteinMPNN). */
  strippedNonProtein?: BioEntityKind[];
  /** Number of protein residues missing required N/CA/C backbone atoms. */
  missingBackboneCount?: number;
  /** True when the upstream structure is Cα-only (e.g. Genie 3 or a Cα-trace PDB). */
  isCaOnly?: boolean;
  /** True when the upstream generator is Genie 3 (for tailored Cα-only guidance). */
  fromGenie3?: boolean;
  /** True when the complex contains a locked target chain (e.g. de novo binder design). */
  hasLockedTarget?: boolean;
  /** True when the upstream sequence/structure originates from a natural/uploaded protein rather than a de novo generator. */
  fromNaturalProtein?: boolean;
}

const MULTICHAIN_RFD4_PRESETS = new Set(['ppi_pdl1', 'yaml_5o45_binder']);
const LIGAND_RFD4_PRESETS = new Set(['sm_sam']);
const DNA_RFD4_PRESETS = new Set(['dna_7m5w']);

/** Split a FASTA or raw sequence entry on `:` or `/` into per-chain amino acid strings. */
export function splitMultiChainSequence(rawSeq: string): {
  concatenated: string;
  chains: string[];
  chainLengths: number[];
} {
  const parts = rawSeq
    .split(/[:/]+/)
    .map((p) => p.replace(/[^ACDEFGHIKLMNPQRSTVWY]/gi, '').toUpperCase())
    .filter((p) => p.length > 0);
  const concatenated = parts.join('');
  return {
    concatenated,
    chains: parts,
    chainLengths: parts.map((p) => p.length),
  };
}

function looksLikePureDna(seq: string): boolean {
  if (seq.length < 12) return false;
  const acgt = (seq.match(/[ACGT]/g) ?? []).length;
  return acgt === seq.length;
}

function parseLengthMidpoint(tok: string): number | null {
  const m = /^(\d+)(?:-(\d+))?[PRDN]?$/i.exec(tok.trim());
  if (!m) return null;
  const lo = Number(m[1]);
  const hi = m[2] ? Number(m[2]) : lo;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  return Math.max(1, Math.round((lo + hi) / 2));
}

function estimateBinderBenchmarkTargetTokens(bench: BenchmarkTargetMeta): number | null {
  if (bench.cropSegments?.trim()) {
    const spans = parseResidueRangeTokens(bench.cropSegments);
    if (spans.length > 0) {
      return spans.reduce((s, r) => s + Math.max(1, r.end - r.start + 1), 0);
    }
  }
  if (bench.conditionParams.contig.includes('/0')) {
    const targetPart = bench.conditionParams.contig.split('/0')[0]?.trim();
    if (targetPart) {
      const spans = parseResidueRangeTokens(targetPart);
      if (spans.length > 0) {
        return spans.reduce((s, r) => s + Math.max(1, r.end - r.start + 1), 0);
      }
    }
  }
  return null;
}

function estimateContigProteinMetrics(opts: {
  contig: string;
  motifSpan: string;
  unindexedMotif: boolean;
  length: number;
  knownProteinChains?: ReadonlySet<string>;
}): { proteinTokens: number; proteinChains: number } {
  const contig = opts.contig.trim();
  const baseLen = Math.max(
    1,
    Math.round(Number.isFinite(opts.length) && opts.length > 0 ? opts.length : 70),
  );
  if (contig && !isBareLengthContig(contig)) {
    const normalized = contig.replace(/\/0(?=$|[\s,/])/g, ',__CHAIN_BREAK__,');
    const rawTokens = normalized
      .split(/[,\s/]+/)
      .map((t) => t.trim())
      .filter(Boolean);
    let proteinTokens = 0;
    let proteinChains = 0;
    let currentChainHasProtein = false;
    for (const tok of rawTokens) {
      if (tok === '__CHAIN_BREAK__') {
        if (currentChainHasProtein) {
          proteinChains++;
          currentChainHasProtein = false;
        }
        continue;
      }
      const lenMid = parseLengthMidpoint(tok);
      if (lenMid !== null) {
        proteinTokens += lenMid;
        currentChainHasProtein = true;
        continue;
      }
      // Named expandable segment with [PRDN] suffix (e.g. "h1:200-200P", "seg:10-20R") — check polymer match first
      const namedExp = /^([A-Za-z0-9_]+):(\d+-\d+)[PRDN]$/i.exec(tok);
      if (namedExp) {
        const prefix = namedExp[1].toUpperCase();
        if (!opts.knownProteinChains?.has(prefix)) {
          const namedMid = parseLengthMidpoint(namedExp[2]);
          if (namedMid !== null) {
            proteinTokens += namedMid;
            currentChainHasProtein = true;
            continue;
          }
        }
      }
      const resRange = /^(?:([A-Za-z0-9_]+):|([A-Za-z_]+))(\d+)(?:-(\d+))?$/.exec(tok);
      if (resRange) {
        const ch = (resRange[1] ?? resRange[2]).toUpperCase();
        if (
          opts.knownProteinChains &&
          opts.knownProteinChains.size > 0 &&
          !opts.knownProteinChains.has(ch)
        ) {
          // Non-protein chain (ligand/DNA) residue token — handled separately
          continue;
        }
        const start = Number(resRange[3]);
        const end = resRange[4] ? Number(resRange[4]) : start;
        if (Number.isFinite(start) && Number.isFinite(end) && end >= start) {
          proteinTokens += end - start + 1;
          currentChainHasProtein = true;
        }
      }
    }
    if (currentChainHasProtein) proteinChains++;
    return {
      proteinTokens: proteinTokens > 0 ? proteinTokens : baseLen,
      proteinChains: Math.max(1, proteinChains),
    };
  }

  const effectiveBase = (contig ? parseLengthMidpoint(contig) : null) ?? baseLen;
  const spans = opts.motifSpan.trim() ? parseResidueRangeTokens(opts.motifSpan) : [];
  if (spans.length === 0) {
    return { proteinTokens: effectiveBase, proteinChains: 1 };
  }
  const motifRes = spans.reduce((s, r) => s + Math.max(1, r.end - r.start + 1), 0);
  const motifChains = new Set(spans.map((r) => r.chain)).size;
  if (opts.unindexedMotif) {
    return {
      proteinTokens: effectiveBase + motifRes,
      proteinChains: motifChains > 1 ? 1 + motifChains : 1,
    };
  }
  return {
    proteinTokens: Math.max(effectiveBase, motifRes + 4 * (spans.length + 1)),
    proteinChains: Math.max(1, motifChains),
  };
}

function collectExplicitLigandsFromRfd4(
  contig: string,
  cropSegments: string,
  hbondSelection: string,
): Set<string> {
  const out = new Set<string>();
  const collect = (raw: string) => {
    for (const rawTok of raw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean)) {
      if (rawTok === '/0') continue;
      const chainComp = /^([A-Za-z0-9_]+):([A-Za-z0-9]{1,5})$/.exec(rawTok);
      if (chainComp && !/^\d+[A-Za-z]?$/.test(chainComp[2])) {
        out.add(chainComp[2].toUpperCase());
        continue;
      }
      if (/^[A-Za-z0-9]{1,5}$/.test(rawTok) && !/^\d+$/.test(rawTok)) {
        out.add(rawTok.toUpperCase());
      }
    }
  };
  collect(contig);
  collect(cropSegments);
  if (hbondSelection) {
    for (const seg of hbondSelection.split(/[,|)\s'"]+/).filter(Boolean)) {
      const parts = seg.split('/');
      if (parts.length >= 2 && parts[1] && parts[1] !== '*') {
        out.add(parts[1].trim().toUpperCase());
      }
    }
  }
  return out;
}

function resolveTemplateArtifactsAndEntities(opts: {
  contig: string;
  hbondSelection: string;
  tplNode: WorkflowNode | null | undefined;
  outputsByNode?: Record<string, Record<string, CanonicalArtifact>>;
}): {
  knownProteinChains: Set<string>;
  retainedLigands: string[];
  nucleicAcidChains: string[];
  fallbackNonProteinEntities: BioEntityKind[];
} {
  const tplNode = opts.tplNode;
  const tplStruct = tplNode
    ? (opts.outputsByNode?.[tplNode.id]?.structure as StructureArtifact | undefined)
    : undefined;
  const tplMode = tplNode?.adapterId === 'pdb-input' ? String(tplNode.params.mode ?? 'demo-bundle') : '';
  const tplCrop =
    tplNode?.adapterId === 'pdb-input' && tplMode !== 'demo-bundle'
      ? String(tplNode.params.cropSegments ?? '').trim()
      : '';
  const tplPdbText = tplNode?.adapterId === 'pdb-input' ? String(tplNode.params.pdbText ?? '').trim() : '';
  let tplSummary: CachedStructureBioSummary | null = null;
  if (!tplStruct?.diagnostics && tplPdbText) {
    try {
      tplSummary = getCachedStructureBioSummary(tplPdbText, tplCrop);
    } catch {
      tplSummary = null;
    }
  }
  const knownProteinChains = new Set<string>(
    (tplStruct?.chainIds ?? tplSummary?.chainIds ?? []).map((c) => c.toUpperCase()),
  );
  const explicitlyNamed = collectExplicitLigandsFromRfd4(
    opts.contig,
    tplCrop,
    opts.hbondSelection,
  );
  const diag = tplStruct?.diagnostics ?? tplSummary?.diagnostics;
  if (diag) {
    const retainedLigands = diag.ligandNames.filter(
      (lig) => !isCrystallizationBuffer(lig) || explicitlyNamed.has(lig.toUpperCase()),
    );
    return {
      knownProteinChains,
      retainedLigands,
      nucleicAcidChains: diag.nucleicAcidChains,
      fallbackNonProteinEntities: [],
    };
  }
  if (tplMode === 'demo-bundle') {
    return {
      knownProteinChains,
      retainedLigands: [],
      nucleicAcidChains: [],
      fallbackNonProteinEntities: [],
    };
  }
  const benchId = tplNode?.adapterId === 'pdb-input' ? String(tplNode.params.benchmarkId ?? '').trim() : '';
  const bench = benchId ? getBenchmarkById(benchId) : undefined;
  if (tplCrop) {
    const cropSummary = summarizeCropSegments(tplCrop);
    if (cropSummary.valid && cropSummary.hasExplicitTokens) {
      const retainedLigands = cropSummary.ligandNames.filter(
        (lig) => !isCrystallizationBuffer(lig) || explicitlyNamed.has(lig.toUpperCase()),
      );
      const fallbackNonProteinEntities: BioEntityKind[] =
        bench && bench.entities.includes('dna') ? ['dna'] : [];
      return {
        knownProteinChains,
        retainedLigands,
        nucleicAcidChains: [],
        fallbackNonProteinEntities,
      };
    }
  }
  const fallbackNonProteinEntities: BioEntityKind[] = bench
    ? bench.entities.filter((e) => e !== 'protein')
    : [];
  return {
    knownProteinChains,
    retainedLigands: [],
    nucleicAcidChains: [],
    fallbackNonProteinEntities,
  };
}

function getSchemaDefaultLength(adapterId: string, fallback = 80): number {
  const field = ADAPTER_REGISTRY[adapterId]?.paramSchema?.find((f) => f.key === 'length');
  return typeof field?.defaultValue === 'number' && Number.isFinite(field.defaultValue)
    ? field.defaultValue
    : fallback;
}

interface SimpleResidueSpan {
  chain: string;
  start: number;
  end: number;
}

/**
 * Parse protein residue spans (and whole-chain single-letter tokens like `A`, `B`) from a
 * `cropSegments` spec, ignoring ligand crop tokens such as `A:ZN`, `A:DHZ`, or `SAM`.
 */
function parseProteinCropSpans(cropSpec: string): SimpleResidueSpan[] {
  const out: SimpleResidueSpan[] = [];
  const trimmed = cropSpec.trim();
  if (!trimmed) return out;
  for (const raw of trimmed.split(/[,\s]+/)) {
    const tok = raw.trim();
    if (!tok) continue;
    // Skip explicit ligand tokens like A:ZN or A:DHZ (chain + non-numeric CCD code)
    if (/^[A-Za-z0-9_]+:[A-Za-z][A-Za-z0-9]*$/.test(tok)) continue;
    // Single-character whole-chain crop token e.g. "A", "B", "H", "L", "1", "2"
    if (/^[A-Za-z0-9_]$/.test(tok)) {
      out.push({
        chain: tok.toUpperCase(),
        start: Number.NEGATIVE_INFINITY,
        end: Number.POSITIVE_INFINITY,
      });
      continue;
    }
    const parsed = parseResidueRangeTokens(tok);
    if (parsed.length > 0) {
      for (const s of parsed) {
        out.push({
          chain: s.chain.toUpperCase(),
          start: Math.min(s.start, s.end),
          end: Math.max(s.start, s.end),
        });
      }
      continue;
    }
  }
  return out;
}

/**
 * Extract ONLY explicit chain-prefixed template residue spans from a contig string
 * (e.g. `A16-35`, `B19-27`, `A55-55`, `AA:10-20`), ignoring de novo numeric spacer lengths
 * such as `10-10`, `16-16`, `180-180`, and chain-break tokens `/0`.
 */
function parseContigTemplateSpans(contig: string): SimpleResidueSpan[] {
  const out: SimpleResidueSpan[] = [];
  const trimmed = contig.trim();
  if (!trimmed) return out;
  for (const raw of trimmed.split(/[,\s/]+/)) {
    const tok = raw.trim();
    if (!tok || /^\d+(?:-\d+)?$/.test(tok)) continue;
    // Skip RFD4 named de novo segment tokens like `h1:16-16P` or bare `/0`
    if (/^[A-Za-z][A-Za-z0-9_]*:\d+-\d+[A-Za-z]$/.test(tok)) continue;
    // Require explicit chain prefix (single letter like A16-35 / A55 or colon form AA:10-20)
    const m =
      /^(?:([A-Za-z0-9_]+):|([A-Za-z_]))(-?\d+)[A-Za-z]?(?:-(?:[A-Za-z0-9_]+:|[A-Za-z_])?(-?\d+)[A-Za-z]?)?$/.exec(
        tok,
      );
    if (!m) continue;
    const chain = (m[1] ?? m[2] ?? '').toUpperCase();
    const a = Number(m[3]);
    const b = m[4] !== undefined ? Number(m[4]) : a;
    if (!chain || !Number.isFinite(a) || !Number.isFinite(b)) continue;
    out.push({ chain, start: Math.min(a, b), end: Math.max(a, b) });
  }
  return out;
}

function residueSpansOverlapAny(
  aSpans: readonly SimpleResidueSpan[],
  bSpans: readonly SimpleResidueSpan[],
): boolean {
  for (const a of aSpans) {
    const aChain = a.chain.toUpperCase();
    for (const b of bSpans) {
      if (aChain !== b.chain.toUpperCase()) continue;
      if (a.start <= b.end && b.start <= a.end) return true;
    }
  }
  return false;
}

const SOLVENT_WATER_RESNAMES = new Set(['HOH', 'WAT', 'DOD', 'H2O', 'TIP', 'TIP3']);

export interface CropSegmentsSummary {
  proteinChains: Set<string>;
  proteinRangeTokens: number;
  hasWholeChainProtein: boolean;
  ligandNames: string[];
  hasExplicitTokens: boolean;
  valid: boolean;
}

function parseCropRangeSpecLocal(tok: string): {
  chain: string;
  start: number;
  end: number;
} | null {
  const colonMatch = /^([A-Za-z0-9_]+):(-?\d+)([A-Za-z])?(?:-(-?\d+)([A-Za-z])?)?$/.exec(tok);
  const bareMatch = !colonMatch
    ? /^([A-Za-z]+)?(-?\d+)([A-Za-z])?(?:-(-?\d+)([A-Za-z])?)?$/.exec(tok)
    : null;
  const m = colonMatch ?? bareMatch;
  if (!m) return null;
  const chain = m[1] || 'A';
  const start = parseInt(m[2], 10);
  const startIns = m[3] || undefined;
  const hasEnd = m[4] !== undefined;
  const end = hasEnd ? parseInt(m[4], 10) : start;
  const endIns = hasEnd ? m[5] || undefined : startIns;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  if (end < start) return null;
  if (end === start && startIns && endIns && endIns < startIns) return null;
  return { chain, start, end };
}

/**
 * Summarize explicit protein chain/range tokens and non-water/non-buffer/non-modified-AA
 * ligand CCD names from a `cropSegments` specification, consistent with `pdb.ts`
 * `parseRawCropRules`/`parseCropRangeSpec` and `hetatm-policy.ts`.
 */
export function summarizeCropSegments(cropSpec: string): CropSegmentsSummary {
  const tokens = cropSpec
    .trim()
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const proteinChains = new Set<string>();
  let proteinRangeTokens = 0;
  let hasWholeChainProtein = false;
  const ligands: string[] = [];
  const seenLigands = new Set<string>();
  let valid = true;

  const maybeAddLigand = (rawCode: string) => {
    const comp = rawCode.trim().toUpperCase();
    if (!comp) return;
    if (
      SOLVENT_WATER_RESNAMES.has(comp) ||
      BUFFER_OR_SOLVENT_RESNAMES.has(comp) ||
      MODIFIED_AMINO_ACID_RESNAMES.has(comp) ||
      STANDARD_AA_RESNAMES.has(comp) ||
      NUCLEIC_ACID_RESNAMES.has(comp)
    ) {
      return;
    }
    if (!seenLigands.has(comp)) {
      seenLigands.add(comp);
      ligands.push(comp);
    }
  };

  for (const tok of tokens) {
    if (tok.includes(':')) {
      const range = parseCropRangeSpecLocal(tok);
      if (range) {
        proteinChains.add(range.chain);
        proteinRangeTokens += Math.max(1, range.end - range.start + 1);
        continue;
      }
      const chainLig = /^([A-Za-z0-9_]+):([A-Za-z0-9]{1,5})$/.exec(tok);
      if (chainLig && !/^-?\d+$/.test(chainLig[2])) {
        maybeAddLigand(chainLig[2]);
        continue;
      }
      valid = false;
      continue;
    }

    if (/^[A-Za-z0-9]$/.test(tok)) {
      proteinChains.add(tok);
      hasWholeChainProtein = true;
      continue;
    }

    const range = parseCropRangeSpecLocal(tok);
    if (range) {
      const upper = tok.toUpperCase();
      if (
        !tok.includes('-') &&
        /^[A-Za-z][A-Za-z0-9]{1,4}$/.test(tok) &&
        (MONOATOMIC_ION_RESNAMES.has(upper) || BUFFER_OR_SOLVENT_RESNAMES.has(upper))
      ) {
        maybeAddLigand(upper);
        continue;
      }
      proteinChains.add(range.chain);
      proteinRangeTokens += Math.max(1, range.end - range.start + 1);
      continue;
    }

    if (/^[A-Za-z0-9]{2,5}$/.test(tok) && /[A-Za-z]/.test(tok)) {
      const upper = tok.toUpperCase();
      if (
        MONOATOMIC_ION_RESNAMES.has(upper) ||
        BUFFER_OR_SOLVENT_RESNAMES.has(upper) ||
        MODIFIED_AMINO_ACID_RESNAMES.has(upper) ||
        STANDARD_AA_RESNAMES.has(upper) ||
        NUCLEIC_ACID_RESNAMES.has(upper) ||
        tok.length >= 3
      ) {
        maybeAddLigand(upper);
      } else {
        // 2-char alphanumeric token (e.g. "AA", "AB") is treated as multi-char chain ID, not a ligand
        proteinChains.add(tok);
        hasWholeChainProtein = true;
      }
      continue;
    }

    valid = false;
  }

  return {
    proteinChains,
    proteinRangeTokens,
    hasWholeChainProtein,
    ligandNames: ligands,
    hasExplicitTokens: tokens.length > 0,
    valid,
  };
}

function inferNodeOutputProfile(
  node: WorkflowNode,
  graph: WorkflowGraph,
  cache: Map<string, NodeBioProfile>,
  visiting: Set<string>,
  outputsByNode?: Record<string, Record<string, CanonicalArtifact>>,
): NodeBioProfile {
  const cached = cache.get(node.id);
  if (cached) return cached;
  if (visiting.has(node.id)) {
    return { entities: new Set(['protein']), stoichiometry: 'monomer', chainCount: 1, estimatedTokens: 80 };
  }
  visiting.add(node.id);

  const upstreamProfileForPort = (port: string): NodeBioProfile | null => {
    const edge = graph.edges.find((e) => e.toNode === node.id && e.toPort === port);
    if (!edge) return null;
    const src = graph.nodes.find((n) => n.id === edge.fromNode);
    if (!src) return null;
    return inferNodeOutputProfile(src, graph, cache, visiting, outputsByNode);
  };

  let profile: NodeBioProfile = {
    entities: new Set<BioEntityKind>(['protein']),
    stoichiometry: 'monomer',
    chainCount: 1,
    estimatedTokens: 80,
  };

  if (node.adapterId === 'pdb-input') {
    const mode = String(node.params.mode ?? 'demo-bundle');
    const isNaturalPdb = mode === 'custom' || mode === 'pdb-id' || mode === 'alphafold-db';
    const pdbText = String(node.params.pdbText ?? '').trim();
    const cropSpec = mode === 'demo-bundle' ? '' : String(node.params.cropSegments ?? '').trim();
    const benchId = String(node.params.benchmarkId ?? '').trim();
    const bench = benchId ? getBenchmarkById(benchId) : undefined;
    const liveStruct = outputsByNode?.[node.id]?.structure as StructureArtifact | undefined;
    const benchCropSpec = String(bench?.cropSegments ?? '').trim();
    const hasCustomCrop = Boolean(bench && cropSpec.length > 0 && cropSpec !== benchCropSpec);

    if (liveStruct && liveStruct.kind === 'StructureArtifact' && liveStruct.diagnostics) {
      const diag = liveStruct.diagnostics;
      const entities = new Set<BioEntityKind>();
      if (liveStruct.length > 0 || (!diag.hasLigands && !diag.hasNucleicAcids)) {
        entities.add('protein');
      }
      if (diag.hasLigands) entities.add('ligand');
      if (diag.hasNucleicAcids) entities.add('dna');
      if (bench && !hasCustomCrop) {
        for (const e of bench.entities) entities.add(e);
      }
      const totalChains = Math.max(
        1,
        diag.chainCount + (diag.hasNucleicAcids ? diag.nucleicAcidChains.length : 0),
      );
      const coverage = inspectArtifactBackboneCoverage(liveStruct);
      profile = {
        entities,
        stoichiometry: totalChains > 1 ? 'multimer' : 'monomer',
        chainCount: totalChains,
        proteinChainCount: diag.chainCount,
        estimatedTokens: liveStruct.length > 0 ? liveStruct.length : (bench?.approxTokens ?? 80),
        missingBackboneCount: coverage.missingBackboneCount,
        isCaOnly: coverage.isCaOnly,
        ...(isNaturalPdb ? { fromNaturalProtein: true } : {}),
      };
    } else if ((mode === 'custom' || mode === 'pdb-id' || mode === 'alphafold-db') && pdbText.length > 0) {
      try {
        const summary = getCachedStructureBioSummary(pdbText, cropSpec);
        const entities = new Set<BioEntityKind>();
        if (summary.nResidues > 0 || (!summary.diagnostics.hasLigands && !summary.diagnostics.hasNucleicAcids)) {
          entities.add('protein');
        }
        if (summary.diagnostics.hasLigands) entities.add('ligand');
        if (summary.diagnostics.hasNucleicAcids) entities.add('dna');
        if (bench && !hasCustomCrop) {
          for (const e of bench.entities) entities.add(e);
        }
        const totalChains = Math.max(
          1,
          summary.diagnostics.chainCount +
            (summary.diagnostics.hasNucleicAcids ? summary.diagnostics.nucleicAcidChains.length : 0),
        );
        profile = {
          entities,
          stoichiometry: totalChains > 1 ? 'multimer' : 'monomer',
          chainCount: totalChains,
          proteinChainCount: summary.diagnostics.chainCount,
          estimatedTokens: summary.nResidues > 0 ? summary.nResidues : (bench?.approxTokens ?? 80),
          missingBackboneCount: summary.missingBackboneCount,
          isCaOnly: summary.isCaOnly,
          fromNaturalProtein: true,
        };
      } catch {
        profile.estimatedTokens = bench?.approxTokens ?? 80;
        profile.fromNaturalProtein = true;
        if (bench) {
          profile.entities = new Set<BioEntityKind>(bench.entities);
        }
      }
    } else if (bench) {
      const contigChains = bench.conditionParams.contig.includes('/0')
        ? bench.conditionParams.contig.split('/0').filter((s) => s.trim().length > 0).length
        : 1;
      if (hasCustomCrop) {
        const cropSummary = summarizeCropSegments(cropSpec);
        const entities = new Set<BioEntityKind>(['protein']);
        if (bench.entities.includes('dna')) entities.add('dna');
        if (cropSummary.ligandNames.length > 0) entities.add('ligand');
        const chainCount = Math.max(1, cropSummary.proteinChains.size);
        const estimatedTokens =
          !cropSummary.hasWholeChainProtein && cropSummary.proteinRangeTokens > 0
            ? cropSummary.proteinRangeTokens + cropSummary.ligandNames.length
            : bench.approxTokens;
        profile = {
          entities,
          stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
          chainCount,
          proteinChainCount: bench.id === 'rfd4_7m5w_bdna' ? 0 : chainCount,
          estimatedTokens,
          ...(isNaturalPdb ? { fromNaturalProtein: true } : {}),
        };
      } else {
        const cropChains = bench.cropSegments
          ? Math.max(1, summarizeCropSegments(bench.cropSegments).proteinChains.size)
          : 1;
        const chainCount = Math.max(1, contigChains, cropChains);
        const proteinChainCount =
          bench.id === 'rfd4_7m5w_bdna'
            ? 0
            : bench.entities.includes('dna') && !bench.conditionParams.contig.includes('/0')
              ? 1
              : chainCount;
        profile = {
          entities: new Set<BioEntityKind>(bench.entities),
          stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
          chainCount,
          proteinChainCount,
          estimatedTokens: bench.approxTokens,
          ...(isNaturalPdb ? { fromNaturalProtein: true } : {}),
        };
      }
    } else if (cropSpec) {
      const cropSummary = summarizeCropSegments(cropSpec);
      const entities = new Set<BioEntityKind>(['protein']);
      if (cropSummary.ligandNames.length > 0) entities.add('ligand');
      const chainCount = Math.max(1, cropSummary.proteinChains.size);
      const baseProteinTokens =
        !cropSummary.hasWholeChainProtein && cropSummary.proteinRangeTokens > 0
          ? cropSummary.proteinRangeTokens
          : 53;
      profile = {
        entities,
        stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
        chainCount,
        estimatedTokens: baseProteinTokens + cropSummary.ligandNames.length,
        ...(isNaturalPdb ? { fromNaturalProtein: true } : {}),
      };
    } else {
      profile.estimatedTokens = 53;
      if (isNaturalPdb) profile.fromNaturalProtein = true;
    }
  } else if (node.adapterId === 'sequence-input') {
    const raw = String(node.params.sequence ?? '').trim();
    const blocks = raw.includes('>')
      ? raw.split(/^>/m).filter((b) => b.trim().length > 0).map((b) => b.split('\n').slice(1).join(''))
      : [raw];
    let maxChains = 1;
    let maxLen = 0;
    let dnaFlag = false;
    for (const blk of blocks) {
      const split = splitMultiChainSequence(blk);
      if (split.chains.length > maxChains) maxChains = split.chains.length;
      if (split.concatenated.length > maxLen) maxLen = split.concatenated.length;
      if (looksLikePureDna(split.concatenated)) dnaFlag = true;
    }
    profile = {
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: maxChains > 1 ? 'multimer' : 'monomer',
      chainCount: maxChains,
      estimatedTokens: maxLen,
      looksLikeDnaSequence: dnaFlag,
      fromNaturalProtein: true,
    };
  } else if (node.adapterId === 'rfd4-proteina') {
    const mode = String(node.params.mode ?? 'unconditional');
    const tpl = upstreamProfileForPort('structure');
    const tplEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
    const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;
    const entities = new Set<BioEntityKind>(['protein']);
    let chainCount = 1;
    let proteinChainCount = 1;
    const rawLength = Number(node.params.length ?? 70);
    let estimatedTokens = Number.isFinite(rawLength) && rawLength > 0 ? rawLength : 70;
    const contig = String(node.params.contig ?? '').trim();
    const motifSpan = String(node.params.motifSpan ?? '').trim();
    const unindexedMotif = Boolean(node.params.unindexedMotif);
    const hbondSelection = String(node.params.hbondSelection ?? '').trim();
    const lockChain = String(node.params.lockChain ?? '').trim();
    let hasLockedTarget = false;

    const tplInfo = tplNode
      ? resolveTemplateArtifactsAndEntities({
          contig,
          hbondSelection,
          tplNode,
          outputsByNode,
        })
      : null;

    const applyRetainedNonProtein = () => {
      if (!tplInfo) return;
      if (tplInfo.retainedLigands.length > 0) {
        entities.add('ligand');
        estimatedTokens += tplInfo.retainedLigands.length;
      }
      if (tplInfo.nucleicAcidChains.length > 0) {
        entities.add('dna');
        chainCount += tplInfo.nucleicAcidChains.length;
        estimatedTokens += 16 * tplInfo.nucleicAcidChains.length;
      }
      for (const e of tplInfo.fallbackNonProteinEntities) {
        entities.add(e);
      }
    };

    if (mode === 'preset') {
      const presetId = String(node.params.presetId ?? '');
      if (MULTICHAIN_RFD4_PRESETS.has(presetId)) {
        chainCount = 2;
        proteinChainCount = 2;
        estimatedTokens = 208;
        hasLockedTarget = true;
      } else if (LIGAND_RFD4_PRESETS.has(presetId)) {
        entities.add('ligand');
        proteinChainCount = 1;
        estimatedTokens = 143;
      } else if (DNA_RFD4_PRESETS.has(presetId)) {
        entities.add('dna');
        chainCount = 3;
        proteinChainCount = 1;
        estimatedTokens = 128;
      }
    } else if (mode === 'partial' || mode === 'inverse-folding') {
      if (tpl) {
        for (const e of tpl.entities) entities.add(e);
        chainCount = Math.max(chainCount, tpl.chainCount);
        proteinChainCount = Math.max(1, tpl.proteinChainCount ?? tpl.chainCount);
        estimatedTokens = Math.max(estimatedTokens, tpl.estimatedTokens);
      }
    } else if (mode === 'motif') {
      const metrics = estimateContigProteinMetrics({
        contig,
        motifSpan,
        unindexedMotif,
        length: estimatedTokens,
        knownProteinChains: tplInfo?.knownProteinChains,
      });
      chainCount = metrics.proteinChains;
      proteinChainCount = metrics.proteinChains;
      estimatedTokens = metrics.proteinTokens;
      applyRetainedNonProtein();
    } else if (mode === 'binder' || Boolean(lockChain) || contig.includes('/0')) {
      hasLockedTarget = mode === 'binder' || Boolean(lockChain);
      if (contig && !isBareLengthContig(contig)) {
        const metrics = estimateContigProteinMetrics({
          contig,
          motifSpan,
          unindexedMotif,
          length: estimatedTokens,
          knownProteinChains: tplInfo?.knownProteinChains,
        });
        chainCount = Math.max(2, metrics.proteinChains);
        proteinChainCount = chainCount;
        estimatedTokens = metrics.proteinTokens;
      } else {
        chainCount = Math.max(2, tpl?.proteinChainCount ?? tpl?.chainCount ?? 2);
        proteinChainCount = chainCount;
        const tplBenchId =
          tplNode?.adapterId === 'pdb-input' ? String(tplNode.params.benchmarkId ?? '').trim() : '';
        const tplBench = tplBenchId ? getBenchmarkById(tplBenchId) : undefined;
        const tplHasLiveOrText =
          tplNode?.adapterId === 'pdb-input' &&
          (Boolean(outputsByNode?.[tplNode.id]?.structure) ||
            Boolean(String(tplNode.params.pdbText ?? '').trim()));
        const benchTargetOnly =
          !tplHasLiveOrText && tplBench && tplBench.category === 'binder'
            ? estimateBinderBenchmarkTargetTokens(tplBench)
            : null;
        estimatedTokens = (benchTargetOnly ?? tpl?.estimatedTokens ?? 100) + estimatedTokens;
      }
      applyRetainedNonProtein();
    } else if (mode === 'conditioned') {
      if (contig || motifSpan) {
        const metrics = estimateContigProteinMetrics({
          contig,
          motifSpan,
          unindexedMotif,
          length: estimatedTokens,
          knownProteinChains: tplInfo?.knownProteinChains,
        });
        chainCount = metrics.proteinChains;
        proteinChainCount = metrics.proteinChains;
        estimatedTokens = metrics.proteinTokens;
        applyRetainedNonProtein();
      } else if (tpl) {
        for (const e of tpl.entities) entities.add(e);
        chainCount = Math.max(chainCount, tpl.chainCount);
        proteinChainCount = Math.max(1, tpl.proteinChainCount ?? tpl.chainCount);
        estimatedTokens = Math.max(estimatedTokens, tpl.estimatedTokens);
      }
    }
    const liveStruct = outputsByNode?.[node.id]?.structure as StructureArtifact | undefined;
    if (liveStruct && liveStruct.kind === 'StructureArtifact') {
      if (liveStruct.diagnostics) {
        if (liveStruct.diagnostics.hasLigands) entities.add('ligand');
        if (liveStruct.diagnostics.hasNucleicAcids) entities.add('dna');
        const liveProtChains = Math.max(1, liveStruct.diagnostics.chainCount);
        const liveNaChains = liveStruct.diagnostics.hasNucleicAcids
          ? liveStruct.diagnostics.nucleicAcidChains.length
          : 0;
        proteinChainCount = liveProtChains;
        chainCount = Math.max(chainCount, liveProtChains + liveNaChains);
      }
      if (liveStruct.length > 0) estimatedTokens = liveStruct.length;
      if (
        (liveStruct.lockedChains && liveStruct.lockedChains.length > 0) ||
        Object.values(liveStruct.chainRoles ?? {}).includes('target')
      ) {
        hasLockedTarget = true;
      }
    }
    profile = {
      entities,
      stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
      chainCount,
      proteinChainCount,
      estimatedTokens,
      ...(hasLockedTarget ? { hasLockedTarget: true } : {}),
    };
  } else if (node.adapterId === 'genie3') {
    const tpl = upstreamProfileForPort('structure');
    const tplEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
    const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;
    const tplBenchId =
      tplNode?.adapterId === 'pdb-input' ? String(tplNode.params.benchmarkId ?? '').trim() : '';
    const tplBench = tplBenchId ? getBenchmarkById(tplBenchId) : undefined;
    const taskMode = String(node.params.taskMode ?? 'unconditional');
    const inlineLock = String(node.params.lockChain ?? '').trim();
    const inlineContig = String(node.params.contig ?? '').trim();
    const inlineMotif = String(node.params.motifSpan ?? '').trim();
    const isBinderTask =
      taskMode === 'binder' ||
      Boolean(inlineLock) ||
      inlineContig.includes('/0') ||
      (Boolean(tpl) && taskMode !== 'motif' && tplBench?.category !== 'motif');
    const binderLen = Number(node.params.length ?? 80);
    let chainCount = 1;
    let estimatedTokens = Number.isFinite(binderLen) && binderLen > 0 ? binderLen : 80;
    if (isBinderTask) {
      chainCount = 2;
      estimatedTokens = tpl
        ? tplBench
          ? tplBench.approxTokens
          : estimatedTokens + tpl.estimatedTokens
        : estimatedTokens;
    } else if (taskMode === 'motif' || inlineContig || inlineMotif) {
      const metrics = estimateContigProteinMetrics({
        contig: inlineContig,
        motifSpan: inlineMotif,
        unindexedMotif: false,
        length: estimatedTokens,
      });
      chainCount = metrics.proteinChains;
      estimatedTokens = metrics.proteinTokens;
    }
    profile = {
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
      chainCount,
      estimatedTokens,
      isCaOnly: true,
      missingBackboneCount: estimatedTokens,
      fromGenie3: true,
      ...(isBinderTask ? { hasLockedTarget: true } : {}),
    };
  } else if (node.adapterId === 'rfdiffusion') {
    const mode = String(node.params.taskMode ?? node.params.mode ?? 'unconditional');
    const contig = String(node.params.contig ?? '').trim();
    const defaultLen = getSchemaDefaultLength('rfdiffusion', 80);
    const rawLen = Number(node.params.length ?? defaultLen);
    const baseLen = Number.isFinite(rawLen) && rawLen > 0 ? rawLen : defaultLen;
    let chainCount = 1;
    let estimatedTokens = baseLen;
    let hasLockedTarget = false;
    if (mode === 'symmetry') {
      const sym = String(node.params.symmetry ?? 'c3').trim().toLowerCase();
      const cMatch = /^c(\d+)$/.exec(sym);
      const dMatch = /^d(\d+)$/.exec(sym);
      const copies = cMatch ? Number(cMatch[1]) : dMatch ? Number(dMatch[1]) * 2 : 3;
      chainCount = Math.max(2, copies);
      // RFdiffusion's `length` parameter is total oligomer length across all symmetric subunits
      estimatedTokens = baseLen;
    } else if (mode === 'motif' || mode === 'binder' || contig.length > 0) {
      const metrics = estimateContigProteinMetrics({
        contig,
        motifSpan: String(node.params.motifSpan ?? ''),
        unindexedMotif: false,
        length: baseLen,
      });
      chainCount = mode === 'binder' ? Math.max(2, metrics.proteinChains) : metrics.proteinChains;
      estimatedTokens = metrics.proteinTokens;
      if (mode === 'binder' || contig.includes('/0')) hasLockedTarget = true;
    }
    profile = {
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
      chainCount,
      estimatedTokens,
      ...(hasLockedTarget ? { hasLockedTarget: true } : {}),
    };
  } else if (node.adapterId === 'rfd3') {
    const mode = String(node.params.mode ?? 'unconditional').trim().toLowerCase();
    const contig = String(node.params.contig ?? '').trim();
    const unindex = String(node.params.unindex ?? '').trim();
    const ligandParam = String(node.params.ligand ?? '').trim();
    const sym = String(node.params.symmetry ?? 'none').trim().toUpperCase();
    const tpl = upstreamProfileForPort('structure');
    const defaultLen = getSchemaDefaultLength('rfd3', 80);
    const rawLen = Number(node.params.length ?? defaultLen);
    const baseLen = Number.isFinite(rawLen) && rawLen > 0 ? rawLen : defaultLen;

    const entities = new Set<BioEntityKind>(['protein']);
    if (mode !== 'unconditional') {
      if (ligandParam) entities.add('ligand');
      if (tpl) {
        if (tpl.entities.has('ligand')) entities.add('ligand');
        if (tpl.entities.has('dna')) entities.add('dna');
        if (tpl.entities.has('rna')) entities.add('rna');
      }
    }

    let chainCount = 1;
    let estimatedTokens = baseLen;
    let hasLockedTarget = false;

    if (mode === 'unconditional' && sym && sym !== 'NONE') {
      const m = /^C(\d+)$/.exec(sym);
      const copies = m ? Math.max(2, Number(m[1])) : 2;
      chainCount = copies;
      // RFD3 symmetry length is per asymmetric unit
      estimatedTokens = baseLen * copies;
    } else if (mode === 'binder') {
      hasLockedTarget = true;
      if (contig) {
        const metrics = estimateContigProteinMetrics({
          contig,
          motifSpan: '',
          unindexedMotif: false,
          length: baseLen,
        });
        chainCount = Math.max(2, metrics.proteinChains);
        estimatedTokens = metrics.proteinTokens;
      } else {
        chainCount = Math.max(2, (tpl?.chainCount ?? 1) + 1);
        estimatedTokens = baseLen + (tpl?.estimatedTokens ?? 0);
      }
    } else if (mode === 'partial') {
      chainCount = tpl?.chainCount ?? 1;
      estimatedTokens = tpl?.estimatedTokens ?? baseLen;
    } else if (mode === 'motif') {
      if (contig) {
        const metrics = estimateContigProteinMetrics({
          contig,
          motifSpan: '',
          unindexedMotif: false,
          length: baseLen,
        });
        chainCount = metrics.proteinChains;
        estimatedTokens = metrics.proteinTokens;
      } else if (unindex) {
        const unidxSpans = parseResidueRangeTokens(unindex);
        const unidxRes = unidxSpans.reduce((s, r) => s + Math.max(1, r.end - r.start + 1), 0);
        estimatedTokens = baseLen + unidxRes;
      }
    }

    profile = {
      entities,
      stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
      chainCount,
      estimatedTokens,
      ...(hasLockedTarget ? { hasLockedTarget: true } : {}),
    };
  } else if (node.adapterId === 'protpardelle') {
    const taskMode = String(node.params.taskMode ?? 'unconditional');
    const tpl = upstreamProfileForPort('structure');
    const defaultLen = getSchemaDefaultLength('protpardelle', 80);
    const rawLen = Number(node.params.length ?? defaultLen);
    const baseLen = Number.isFinite(rawLen) && rawLen > 0 ? rawLen : defaultLen;
    if (taskMode === 'partial' && tpl) {
      const strippedNonProtein = new Set<BioEntityKind>(tpl.strippedNonProtein ?? []);
      for (const e of tpl.entities) {
        if (e !== 'protein') strippedNonProtein.add(e);
      }
      profile = {
        entities: new Set<BioEntityKind>(['protein']),
        stoichiometry: tpl.stoichiometry,
        chainCount: tpl.chainCount,
        estimatedTokens: tpl.estimatedTokens,
        ...(strippedNonProtein.size > 0 ? { strippedNonProtein: [...strippedNonProtein] } : {}),
      };
    } else if (taskMode === 'motif') {
      const motifSelection = String(node.params.motifSelection ?? node.params.motifSpan ?? '').trim();
      const spans = motifSelection ? parseResidueRangeTokens(motifSelection) : [];
      const motifRes = spans.reduce((s, r) => s + Math.max(1, r.end - r.start + 1), 0);
      const estimatedTokens = motifRes > 0 ? Math.max(baseLen, motifRes + 10) : baseLen;
      profile = {
        entities: new Set<BioEntityKind>(['protein']),
        stoichiometry: 'monomer',
        chainCount: 1,
        estimatedTokens,
      };
    } else {
      profile = {
        entities: new Set<BioEntityKind>(['protein']),
        stoichiometry: 'monomer',
        chainCount: 1,
        estimatedTokens: baseLen,
      };
    }
  } else if (node.adapterId === 'boltzgen') {
    const taskMode = String(node.params.taskMode ?? 'unconditional');
    const tpl = upstreamProfileForPort('structure');
    const defaultLen = getSchemaDefaultLength('boltzgen', 48);
    const rawLen = Number(node.params.length ?? defaultLen);
    const baseLen = Number.isFinite(rawLen) && rawLen > 0 ? rawLen : defaultLen;
    if (taskMode === 'binder' && tpl) {
      const chainCount = Math.max(2, tpl.chainCount + 1);
      profile = {
        entities: new Set<BioEntityKind>(['protein']),
        stoichiometry: 'multimer',
        chainCount,
        estimatedTokens: baseLen + tpl.estimatedTokens,
        hasLockedTarget: true,
      };
    } else if (taskMode === 'motif') {
      const rawOff = Number(node.params.motifOffset ?? 10);
      const motifOffset = Number.isFinite(rawOff) && rawOff >= 0 ? rawOff : 10;
      const estimatedTokens = tpl ? Math.max(baseLen, tpl.estimatedTokens + motifOffset + 4) : baseLen;
      profile = {
        entities: new Set<BioEntityKind>(['protein']),
        stoichiometry: 'monomer',
        chainCount: 1,
        estimatedTokens,
      };
    } else {
      profile = {
        entities: new Set<BioEntityKind>(['protein']),
        stoichiometry: 'monomer',
        chainCount: 1,
        estimatedTokens: baseLen,
      };
    }
  } else if (
    node.adapterId === 'la-proteina' ||
    node.adapterId === 'frameflow' ||
    node.adapterId === 'foldflow2'
  ) {
    const defaultLen = getSchemaDefaultLength(node.adapterId, 80);
    const rawLen = Number(node.params.length ?? defaultLen);
    const est = Number.isFinite(rawLen) && rawLen > 0 ? rawLen : defaultLen;
    profile = {
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: 'monomer',
      chainCount: 1,
      estimatedTokens: est,
    };
  } else if (
    node.adapterId === 'proteina' ||
    node.adapterId === 'sid-protein' ||
    node.adapterId === 'genie2'
  ) {
    const defaultLen = getSchemaDefaultLength(node.adapterId, 80);
    const rawLen = Number(node.params.length ?? defaultLen);
    const est = Number.isFinite(rawLen) && rawLen > 0 ? rawLen : defaultLen;
    profile = {
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: 'monomer',
      chainCount: 1,
      estimatedTokens: est,
      isCaOnly: true,
      missingBackboneCount: est,
    };
  } else if (
    node.adapterId === 'web-protein-mpnn' ||
    node.adapterId === 'ligandmpnn' ||
    node.adapterId === 'lasermpnn' ||
    node.adapterId === 'antifold-esmif1'
  ) {
    const st = upstreamProfileForPort('structure');
    const strippedSet = new Set<BioEntityKind>(st?.strippedNonProtein ?? []);
    if (st) {
      for (const e of st.entities) {
        if (e !== 'protein') strippedSet.add(e);
      }
    }
    const strippedNonProtein = strippedSet.size > 0 ? [...strippedSet] : undefined;
    // MPNN / inverse folding outputs protein sequences only — DNA/RNA chains are stripped,
    // so downstream stoichiometry MUST reflect protein chains only (not DNA duplex chains!).
    const proteinChainCount = Math.max(1, st?.proteinChainCount ?? st?.chainCount ?? 1);
    const proteinStoichiometry: 'monomer' | 'multimer' =
      proteinChainCount > 1 ? 'multimer' : 'monomer';
    const hasLockedTarget =
      Boolean(st?.hasLockedTarget) ||
      (proteinStoichiometry === 'multimer' && Boolean(String(node.params.fixedPositions ?? '').trim()));
    profile = {
      // Inverse folding emits designed protein sequences (not natural sequences — do not
      // propagate fromNaturalProtein, since designed sequences should fold single-seq unless
      // paired with a locked natural target chain).
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: proteinStoichiometry,
      chainCount: proteinChainCount,
      proteinChainCount,
      estimatedTokens: st?.estimatedTokens ?? 80,
      ...(strippedNonProtein ? { strippedNonProtein } : {}),
      ...(hasLockedTarget ? { hasLockedTarget: true } : {}),
    };
  } else if (node.adapterId === 'top-k-filter') {
    const seq = upstreamProfileForPort('sequences');
    profile = {
      entities: new Set<BioEntityKind>(seq ? [...seq.entities] : ['protein']),
      stoichiometry: seq?.stoichiometry ?? 'monomer',
      chainCount: seq?.chainCount ?? 1,
      ...(seq?.proteinChainCount !== undefined ? { proteinChainCount: seq.proteinChainCount } : {}),
      estimatedTokens: seq?.estimatedTokens ?? 80,
      ...(seq?.strippedNonProtein ? { strippedNonProtein: seq.strippedNonProtein } : {}),
      ...(seq?.hasLockedTarget ? { hasLockedTarget: true } : {}),
      ...(seq?.fromNaturalProtein ? { fromNaturalProtein: true } : {}),
    };
  } else if (node.adapterId === 'localfold-af2') {
    const seq = upstreamProfileForPort('sequences');
    const modelId = String(node.params.model ?? 'monomer');
    const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
    const supportsHolo = meta.family === 'af3' || meta.family === 'esmfold2';
    const liveStruct = outputsByNode?.[node.id]?.structure as StructureArtifact | undefined;

    if (liveStruct && liveStruct.kind === 'StructureArtifact' && liveStruct.diagnostics) {
      const diag = liveStruct.diagnostics;
      const cfg = supportsHolo
        ? resolveLocalFoldEffectiveConfig({ modelKey: modelId, params: node.params })
        : null;
      const entities = new Set<BioEntityKind>();
      if (liveStruct.length > 0 || (!diag.hasLigands && !diag.hasNucleicAcids)) {
        entities.add('protein');
      }
      if (diag.hasLigands) entities.add('ligand');
      if (diag.hasNucleicAcids) {
        entities.add(cfg && cfg.rnaSeqs.length > 0 && cfg.dnaSeqs.length === 0 ? 'rna' : 'dna');
      }
      const totalChains = Math.max(
        1,
        diag.chainCount + (diag.hasNucleicAcids ? diag.nucleicAcidChains.length : 0),
      );
      const coverage = inspectArtifactBackboneCoverage(liveStruct);
      const strippedSet = new Set<BioEntityKind>(seq?.strippedNonProtein ?? []);
      if (seq) {
        for (const e of seq.entities) {
          if (e !== 'protein' && !entities.has(e)) strippedSet.add(e);
        }
      }
      for (const e of entities) strippedSet.delete(e);
      const strippedNonProtein = strippedSet.size > 0 ? [...strippedSet] : undefined;
      profile = {
        entities,
        stoichiometry: totalChains > 1 ? 'multimer' : 'monomer',
        chainCount: totalChains,
        proteinChainCount: Math.max(1, diag.chainCount),
        estimatedTokens:
          (liveStruct.length > 0 ? liveStruct.length : (seq?.estimatedTokens ?? 80)) +
          diag.ligandNames.length,
        missingBackboneCount: coverage.missingBackboneCount,
        isCaOnly: coverage.isCaOnly,
        ...(strippedNonProtein ? { strippedNonProtein } : {}),
        ...(seq?.hasLockedTarget ? { hasLockedTarget: true } : {}),
        ...(seq?.fromNaturalProtein ? { fromNaturalProtein: true } : {}),
      };
    } else {
      const entities = new Set<BioEntityKind>(['protein']);
      const proteinChainCount = Math.max(1, seq?.proteinChainCount ?? seq?.chainCount ?? 1);
      let chainCount = proteinChainCount;
      let estimatedTokens = seq?.estimatedTokens ?? 80;
      if (supportsHolo) {
        const cfg = resolveLocalFoldEffectiveConfig({ modelKey: modelId, params: node.params });
        if (cfg.ligandCcdCodes.length > 0) {
          entities.add('ligand');
          estimatedTokens += cfg.ligandCcdCodes.length;
        }
        if (cfg.dnaSeqs.length > 0) {
          entities.add('dna');
          chainCount += cfg.dnaSeqs.length;
          estimatedTokens += cfg.dnaSeqs.reduce((s, d) => s + d.length, 0);
        }
        if (cfg.rnaSeqs.length > 0) {
          entities.add('rna');
          chainCount += cfg.rnaSeqs.length;
          estimatedTokens += cfg.rnaSeqs.reduce((s, r) => s + r.length, 0);
        }
      }
      const strippedSet = new Set<BioEntityKind>(seq?.strippedNonProtein ?? []);
      if (seq) {
        for (const e of seq.entities) {
          if (e !== 'protein' && !entities.has(e)) strippedSet.add(e);
        }
      }
      for (const e of entities) strippedSet.delete(e);
      const strippedNonProtein = strippedSet.size > 0 ? [...strippedSet] : undefined;
      profile = {
        entities,
        stoichiometry: chainCount > 1 ? 'multimer' : (seq?.stoichiometry ?? 'monomer'),
        chainCount,
        proteinChainCount,
        estimatedTokens,
        ...(strippedNonProtein ? { strippedNonProtein } : {}),
        ...(seq?.hasLockedTarget ? { hasLockedTarget: true } : {}),
        ...(seq?.fromNaturalProtein ? { fromNaturalProtein: true } : {}),
      };
    }
  }

  visiting.delete(node.id);
  cache.set(node.id, profile);
  return profile;
}

const NUCLEIC_RESNAME_TO_BASE: Readonly<Record<string, string>> = {
  DA: 'A',
  DC: 'C',
  DG: 'G',
  DT: 'T',
  DU: 'U',
  A: 'A',
  C: 'C',
  G: 'G',
  T: 'T',
  U: 'U',
  ADE: 'A',
  CYT: 'C',
  GUA: 'G',
  THY: 'T',
  URA: 'U',
};

/** Exact 18-bp RCSB 7M5W B-DNA duplex (chains B and C matching `cropSegments: 'B, C'`). */
const RCSB_7M5W_DNA_DUPLEX: readonly [string, string] = [
  'GGTTATGAATGAAAAAGC',
  'GCTTTTTCATTCATAACC',
];

/**
 * Extract per-chain DNA/RNA strand sequences (`DA/DC/DG/DT` -> `A/C/G/T`, `A/C/G/U` -> `A/C/G/U`)
 * from PDB (`ATOM`/`HETATM`) or mmCIF (`_atom_site`) structure text (first model only, in chain order).
 */
export function extractNucleicAcidSequencesFromStructureText(text: string): string[] {
  if (!text || !text.trim()) return [];
  const chainSeqs = new Map<string, string[]>();
  const seenResidues = new Set<string>();

  const recordResidue = (chainRaw: string, seqRaw: string, insRaw: string, compRaw: string) => {
    const comp = compRaw.trim().toUpperCase();
    const base = NUCLEIC_RESNAME_TO_BASE[comp];
    if (!base) return;
    const chain = chainRaw.trim() || 'A';
    const seqId = seqRaw.trim();
    const ins = insRaw.trim();
    const resKey = `${chain}:${seqId}:${ins}`;
    if (seenResidues.has(resKey)) return;
    seenResidues.add(resKey);
    let arr = chainSeqs.get(chain);
    if (!arr) {
      arr = [];
      chainSeqs.set(chain, arr);
    }
    arr.push(base);
  };

  const trimmed = text.trimStart();
  const isMmcif = trimmed.startsWith('data_') || text.includes('_atom_site.');

  if (isMmcif) {
    const lines = text.split(/\r?\n/);
    let inLoop = false;
    let inAtomSite = false;
    let cols: string[] = [];
    let iGroup = -1;
    let iModel = -1;
    let iLabelComp = -1;
    let iAuthComp = -1;
    let iAuthAsym = -1;
    let iLabelAsym = -1;
    let iAuthSeq = -1;
    let iLabelSeq = -1;
    let iIns = -1;
    let firstModel: string | null = null;
    let pendingRow: string[] = [];

    const cifVal = (row: string[], idx: number): string => {
      if (idx < 0 || idx >= row.length) return '';
      const v = row[idx];
      return v === '.' || v === '?' ? '' : v;
    };

    const flushRow = (row: string[]) => {
      const grp = (cifVal(row, iGroup) || 'ATOM').toUpperCase();
      if (grp !== 'ATOM' && grp !== 'HETATM') return;
      const modelNum = cifVal(row, iModel);
      if (modelNum) {
        if (firstModel === null) firstModel = modelNum;
        else if (modelNum !== firstModel) return;
      }
      const comp = cifVal(row, iLabelComp) || cifVal(row, iAuthComp);
      if (!comp) return;
      const chain = cifVal(row, iAuthAsym) || cifVal(row, iLabelAsym) || 'A';
      const seqId = cifVal(row, iAuthSeq) || cifVal(row, iLabelSeq);
      const ins = cifVal(row, iIns);
      recordResidue(chain, seqId, ins, comp);
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
          else if (colName === '_atom_site.label_comp_id') iLabelComp = idx;
          else if (colName === '_atom_site.auth_comp_id') iAuthComp = idx;
          else if (colName === '_atom_site.auth_asym_id') iAuthAsym = idx;
          else if (colName === '_atom_site.label_asym_id') iLabelAsym = idx;
          else if (colName === '_atom_site.auth_seq_id') iAuthSeq = idx;
          else if (colName === '_atom_site.label_seq_id') iLabelSeq = idx;
          else if (colName === '_atom_site.pdbx_PDB_ins_code') iIns = idx;
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
    const lines = text.split(/\r?\n/);
    let seenModel = false;
    let seenAtoms = false;
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
      if ((!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) || line.length < 26) continue;
      seenAtoms = true;
      const comp = line.slice(17, 20);
      const chain = line[21] ?? 'A';
      const resNum = line.slice(22, 26);
      const ins = line[26] ?? '';
      recordResidue(chain, resNum, ins, comp);
    }
  }

  const out: string[] = [];
  for (const bases of chainSeqs.values()) {
    if (bases.length > 0) out.push(bases.join(''));
  }
  return out;
}

function sanitizeHoloLigandHint(
  rawLig: string,
  opts?: { pdbId?: string; benchmarkId?: string },
): string | null {
  const clean = String(rawLig).trim().toUpperCase();
  if (!/^[A-Z0-9]{1,5}$/.test(clean)) return null;
  if (
    STANDARD_AA_RESNAMES.has(clean) ||
    NUCLEIC_ACID_RESNAMES.has(clean) ||
    isCrystallizationBuffer(clean) ||
    isSilentProteinOnlyHetatm(clean, opts)
  ) {
    return null;
  }
  return clean;
}

const STRUCTURE_ORIGIN_ADAPTERS = new Set([
  'pdb-input',
  'rfd4-proteina',
  'rfd3',
  'genie3',
  'rfdiffusion',
  'protpardelle',
  'boltzgen',
  'la-proteina',
  'frameflow',
  'foldflow2',
  'proteina',
  'sid-protein',
  'genie2',
  'localfold-af2',
]);

function hasUpstreamStructureOrigin(graph: WorkflowGraph, targetNodeId: string): boolean {
  const visited = new Set<string>([targetNodeId]);
  const queue: string[] = [];
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
  for (const e of graph.edges) {
    if (e.toNode === targetNodeId && !visited.has(e.fromNode)) {
      queue.push(e.fromNode);
    }
  }
  while (queue.length > 0) {
    const cur = queue.shift()!;
    if (visited.has(cur)) continue;
    visited.add(cur);
    const n = nodeById.get(cur);
    if (n && STRUCTURE_ORIGIN_ADAPTERS.has(n.adapterId)) return true;
    for (const e of graph.edges) {
      if (e.toNode === cur && !visited.has(e.fromNode)) {
        queue.push(e.fromNode);
      }
    }
  }
  return false;
}

/**
 * Walk upstream of `targetNodeId` to recover holo ligand CCD codes and DNA sequences
 * from curated benchmarks, explicit `cropSegments` (e.g. `A:ZN, A:DHZ`), un-run `pdbText`,
 * RFD4 preset/live outputs, or live StructureArtifacts (filtering buffers and respecting
 * the 1PRW crystal Ca²⁺ exemption).
 */
export function extractUpstreamHoloHints(
  graph: WorkflowGraph,
  targetNodeId: string,
  outputsByNode?: Record<string, Record<string, CanonicalArtifact>>,
): { ligandCcdCodes: string[]; dnaSequences: string[] } {
  const ligands = new Set<string>();
  const dnas = new Set<string>();
  const visited = new Set<string>();
  const queue = [targetNodeId];
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
  let needs7m5wFallback = false;

  const addFilteredLigands = (
    rawList: readonly string[] | undefined,
    opts?: { pdbId?: string; benchmarkId?: string },
  ) => {
    if (!rawList?.length) return;
    for (const lig of rawList) {
      const ok = sanitizeHoloLigandHint(lig, opts);
      if (ok) ligands.add(ok);
    }
  };

  while (queue.length > 0) {
    const cur = queue.shift()!;
    if (visited.has(cur)) continue;
    visited.add(cur);
    const n = nodeById.get(cur);
    let skipUpstreamEdges = false;

    if (n && cur !== targetNodeId) {
      if (n.adapterId === 'pdb-input') {
        const mode = String(n.params.mode ?? 'demo-bundle');
        const bmId = String(n.params.benchmarkId ?? '').trim();
        const bm = bmId ? getBenchmarkById(bmId) : undefined;
        const pdbId = String(n.params.pdbId ?? '').trim().toUpperCase() || bm?.pdbId || '';
        const filterOpts = { pdbId: pdbId || undefined, benchmarkId: bm?.id };
        const rawCrop = mode === 'demo-bundle' ? '' : String(n.params.cropSegments ?? '').trim();
        const rawPdbText = String(n.params.pdbText ?? '').trim();
        const liveStruct = outputsByNode?.[n.id]?.structure as StructureArtifact | undefined;

        if (liveStruct && liveStruct.kind === 'StructureArtifact') {
          addFilteredLigands(liveStruct.diagnostics?.ligandNames, filterOpts);
          const liveText = liveStruct.cifText || liveStruct.pdbText || '';
          const liveDnas = liveText ? extractNucleicAcidSequencesFromStructureText(liveText) : [];
          for (const seq of liveDnas) dnas.add(seq);
          if (
            bm?.id === 'rfd4_7m5w_bdna' &&
            liveDnas.length === 0 &&
            (!liveStruct.diagnostics || liveStruct.diagnostics.hasNucleicAcids)
          ) {
            needs7m5wFallback = true;
          }
        } else if (mode !== 'demo-bundle' && rawPdbText.length > 0) {
          let effectiveText = rawPdbText;
          if (rawCrop) {
            try {
              effectiveText = cropRawAllAtomStructure(rawPdbText, rawCrop).text;
            } catch {
              effectiveText = rawPdbText;
            }
          }
          try {
            const summary = getCachedStructureBioSummary(rawPdbText, rawCrop);
            addFilteredLigands(summary.diagnostics.ligandNames, filterOpts);
          } catch {
            /* ignore malformed un-run pdbText */
          }
          const textDnas = extractNucleicAcidSequencesFromStructureText(effectiveText);
          for (const seq of textDnas) dnas.add(seq);
          if (bm?.id === 'rfd4_7m5w_bdna' && textDnas.length === 0) {
            needs7m5wFallback = true;
          }
        } else if (mode !== 'demo-bundle') {
          const crop = rawCrop || String(bm?.cropSegments ?? '').trim();
          if (crop) {
            addFilteredLigands(summarizeCropSegments(crop).ligandNames, filterOpts);
          }
          if (bm?.id === 'rfd4_7m5w_bdna') {
            needs7m5wFallback = true;
          }
        }
      } else if (n.adapterId === 'rfd4-proteina') {
        const mode = String(n.params.mode ?? 'unconditional');
        const liveStruct = outputsByNode?.[n.id]?.structure as StructureArtifact | undefined;
        const liveBenchId = liveStruct?.benchmarkId || String(n.params.autoSeededBenchmarkId ?? '').trim();
        const liveBench = liveBenchId ? getBenchmarkById(liveBenchId) : undefined;
        const livePdbId =
          liveBench?.pdbId ||
          (/\b([0-9][A-Za-z0-9]{3})\b/.exec(liveStruct?.label ?? '')?.[1]?.toUpperCase() ?? '');
        const filterOpts = { pdbId: livePdbId || undefined, benchmarkId: liveBench?.id };

        if (liveStruct && liveStruct.kind === 'StructureArtifact') {
          addFilteredLigands(liveStruct.diagnostics?.ligandNames, filterOpts);
          const liveText = liveStruct.cifText || liveStruct.pdbText || '';
          const liveDnas = liveText ? extractNucleicAcidSequencesFromStructureText(liveText) : [];
          for (const seq of liveDnas) dnas.add(seq);
          // When RFD4 has a live output with diagnostics, prefer its live entities over ignored/older templates
          if (liveStruct.diagnostics && mode === 'preset') {
            skipUpstreamEdges = true;
          }
        }
        if (mode === 'preset') {
          skipUpstreamEdges = true;
          const presetId = String(n.params.presetId ?? '');
          if (LIGAND_RFD4_PRESETS.has(presetId)) {
            ligands.add('SAM');
          } else if (DNA_RFD4_PRESETS.has(presetId)) {
            needs7m5wFallback = true;
          }
        } else if (mode === 'unconditional') {
          skipUpstreamEdges = true;
        }
      }
    }

    if (!skipUpstreamEdges) {
      for (const e of graph.edges) {
        if (e.toNode === cur && !visited.has(e.fromNode)) {
          queue.push(e.fromNode);
        }
      }
    }
  }

  if (needs7m5wFallback && dnas.size === 0) {
    for (const seq of RCSB_7M5W_DNA_DUPLEX) dnas.add(seq);
  }

  return { ligandCcdCodes: [...ligands], dnaSequences: [...dnas] };
}

export interface BioFoldLimitOptions {
  /** Override maximum residue count for LocalFold (AF3 / Boltz / AF2 / ESMFold2). Omit/null for browser defaults (512 AF3, 768 AF2). */
  maxFoldResidues?: number | null;
  /** True when the folding node executes on a self-hosted Host GPU rather than local browser WebGPU. */
  isHostGpu?: boolean;
  /** Optional per-node resolver (used in Hybrid mode where only some nodes route to Host GPU). */
  resolveNodeFoldLimits?: (node: WorkflowNode) => { maxFoldResidues?: number | null; isHostGpu?: boolean } | undefined;
}

/**
 * Layer 1: Static biological capability validation across the workflow graph.
 * Accepts optional materialized `outputsByNode` to inspect live `StructureArtifact.diagnostics`.
 * Returns actionable warnings/errors with 1-click fixes for biological modality mismatches.
 */
export function validateBioCapabilities(
  graph: WorkflowGraph,
  outputsByNode?: Record<string, Record<string, CanonicalArtifact>>,
  foldLimitOpts?: BioFoldLimitOptions,
): BioCompatibilityIssue[] {
  const issues: BioCompatibilityIssue[] = [];
  const cache = new Map<string, NodeBioProfile>();
  const visiting = new Set<string>();

  for (const node of graph.nodes) {
    const ownProfile = inferNodeOutputProfile(node, graph, cache, visiting, outputsByNode);

    // Generic schema-driven `field.warnBelow` guardrails (respecting `field.modes` & `warnBelow.modes`)
    const manifest = ADAPTER_REGISTRY[node.adapterId];
    if (manifest?.paramSchema) {
      const modeKey =
        manifest.modeParamKey ??
        (manifest.paramSchema.some((f) => f.key === 'mode') ? 'mode' : undefined);
      const modeField = modeKey
        ? manifest.paramSchema.find((f) => f.key === modeKey)
        : undefined;
      const activeMode = modeKey
        ? String(node.params[modeKey] ?? modeField?.defaultValue ?? '')
        : undefined;
      for (const field of manifest.paramSchema) {
        if (!field.warnBelow) continue;
        if (
          field.modes &&
          field.modes.length > 0 &&
          activeMode !== undefined &&
          !field.modes.includes(activeMode)
        ) {
          continue;
        }
        if (
          field.warnBelow.modes &&
          field.warnBelow.modes.length > 0 &&
          (activeMode === undefined || !field.warnBelow.modes.includes(activeMode))
        ) {
          continue;
        }
        const rawVal = node.params[field.key];
        const numVal =
          rawVal !== undefined && rawVal !== null && rawVal !== ''
            ? Number(rawVal)
            : typeof field.defaultValue === 'number'
              ? field.defaultValue
              : Number.NaN;
        if (Number.isFinite(numVal) && numVal < field.warnBelow.threshold) {
          issues.push({
            severity: 'warning',
            code: `param-warn-below-${node.adapterId}-${field.key}`,
            nodeId: node.id,
            message: `${node.label ?? manifest.shortName}: ${field.warnBelow.message}`,
            ...(field.warnBelow.fixValue !== undefined
              ? {
                  fix: {
                    label: `Set ${field.warnBelow.fixValue}`,
                    nodeId: node.id,
                    paramsPatch: { [field.key]: field.warnBelow.fixValue },
                  },
                }
              : {}),
          });
        }
      }
    }

    if (node.adapterId === 'sequence-input' && ownProfile.looksLikeDnaSequence) {
      issues.push({
        severity: 'warning',
        code: 'dna-in-protein-seq',
        nodeId: node.id,
        message: `${node.label ?? 'Sequence Input'} contains only A/C/G/T characters — make sure this is a protein sequence, not raw DNA.`,
      });
    }

    if (node.adapterId === 'genie3') {
      const taskMode = String(node.params.taskMode ?? 'unconditional');
      const inlineLock = String(node.params.lockChain ?? '').trim();
      const inlineMotif = String(node.params.motifSpan ?? '').trim();
      const inlineContig = String(node.params.contig ?? '').trim();
      const tplEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
      const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;

      if ((taskMode === 'motif' || taskMode === 'binder' || Boolean(inlineLock) || Boolean(inlineMotif)) && !tplNode) {
        issues.push({
          severity: 'error',
          code: 'genie3-missing-template',
          nodeId: node.id,
          message: `${node.label ?? 'Genie 3'} (${taskMode} mode) requires a wired input structure (e.g. from PDB Input) on its "structure" port.`,
        });
      }
      if (taskMode === 'motif' && !inlineMotif && !inlineContig) {
        issues.push({
          severity: 'error',
          code: 'genie3-empty-motif',
          nodeId: node.id,
          message: `${node.label ?? 'Genie 3'} is set to motif mode, but both motifSpan and contig are empty. Specify a motif span (e.g. A16-35) or contig.`,
        });
      }

      if (tplNode) {
        const benchId =
          tplNode.adapterId === 'pdb-input' ? String(tplNode.params.benchmarkId ?? '').trim() : '';
        const bench = benchId ? getBenchmarkById(benchId) : undefined;
        const tplProfile = inferNodeOutputProfile(tplNode, graph, cache, visiting, outputsByNode);
        const liveStruct = outputsByNode?.[tplNode.id]?.structure as StructureArtifact | undefined;
        const rawPdbText =
          tplNode.adapterId === 'pdb-input' ? String(tplNode.params.pdbText ?? '').trim() : '';
        const cropSpec =
          tplNode.adapterId === 'pdb-input' ? String(tplNode.params.cropSegments ?? '').trim() : '';
        let parsedSummary: CachedStructureBioSummary | null = null;
        if (!liveStruct && rawPdbText) {
          try {
            parsedSummary = getCachedStructureBioSummary(rawPdbText, cropSpec);
          } catch {
            parsedSummary = null;
          }
        }
        const pdbId =
          (tplNode.adapterId === 'pdb-input' ? String(tplNode.params.pdbId ?? '').trim() : '') ||
          bench?.pdbId ||
          (/\b([0-9][A-Za-z0-9]{3})\b/.exec(liveStruct?.label ?? '')?.[1] ?? '');
        const diag = liveStruct?.diagnostics ?? parsedSummary?.diagnostics;
        const proteinResidueCount = liveStruct
          ? liveStruct.length
          : parsedSummary
            ? parsedSummary.nResidues
            : undefined;
        const policy = evaluateProteinOnlyStructurePolicy({
          modelName: node.label ?? 'Genie 3',
          benchmark: bench,
          pdbId,
          proteinResidueCount,
          ligandNames: diag?.ligandNames,
          nucleicAcidChains: diag?.nucleicAcidChains,
          hasLigands: diag ? diag.hasLigands : tplProfile.entities.has('ligand'),
          hasNucleicAcids: diag
            ? diag.hasNucleicAcids
            : tplProfile.entities.has('dna') || tplProfile.entities.has('rna'),
        });
        if (policy.action === 'hard-block') {
          issues.push({
            severity: policy.reason === 'insufficient-protein' ? 'error' : 'warning',
            code:
              policy.reason === 'insufficient-protein'
                ? 'genie3-insufficient-protein'
                : 'genie3-unsupported-benchmark',
            nodeId: node.id,
            message: policy.message!,
          });
        } else if (policy.action === 'warn') {
          const proteinChains = [
            ...new Set(
              (liveStruct?.chainIds ?? parsedSummary?.chainIds ?? []).filter((c) => Boolean(c)),
            ),
          ];
          const cropFixSpec = proteinChains.length > 0 ? proteinChains.join(', ') : 'A';
          issues.push({
            severity: 'warning',
            code: 'genie3-unsupported-entities',
            nodeId: node.id,
            message: policy.message!,
            ...(tplNode.adapterId === 'pdb-input'
              ? {
                  fix: {
                    label: 'Crop protein chain',
                    nodeId: tplNode.id,
                    paramsPatch: { cropSegments: cropFixSpec },
                  },
                }
              : {}),
          });
        }
      }
    }

    if (node.adapterId === 'rfd4-proteina') {
      const mode = String(node.params.mode ?? 'unconditional');
      const hasTpl = graph.edges.some((e) => e.toNode === node.id && e.toPort === 'structure');
      if (
        (mode === 'motif' || mode === 'binder' || mode === 'partial' || mode === 'inverse-folding') &&
        !hasTpl
      ) {
        issues.push({
          severity: 'error',
          code: 'rfd4-missing-template',
          nodeId: node.id,
          message: `${node.label ?? 'RFD4-Proteina'} (${mode} mode) requires a wired input structure (e.g. from PDB Input) on its "structure" port.`,
        });
      } else if (hasTpl && mode === 'preset') {
        issues.push({
          severity: 'warning',
          code: 'rfd4-preset-ignores-template',
          nodeId: node.id,
          message: `${node.label ?? 'RFD4-Proteina'} is set to built-in Preset mode ("${String(node.params.presetId ?? 'custom_cath_monomer')}"), which loads a remote example archive and ignores the wired input structure. Switch to Motif / Context Mode to condition on your wired template.`,
          fix: {
            label: 'Switch to Motif / Context Mode',
            nodeId: node.id,
            paramsPatch: { mode: 'motif' },
          },
        });
      }
      if (mode === 'motif') {
        const rawHotspots = String(node.params.hotspots ?? '').trim();
        if (rawHotspots.length > 0) {
          const hotspotTokens = rawHotspots
            .split(/[,\s]+/)
            .map((t) => t.trim())
            .filter((t) => /^[A-Za-z]?\d+$/.test(t));
          const rawMotif = String(node.params.motifSpan ?? '').trim();
          const paramsPatch: Record<string, unknown> = { hotspots: '' };
          let detail = 'Clear binder hotspots';
          if (hotspotTokens.length >= 2) {
            const nums = hotspotTokens.map((t) => Number(t.replace(/^[A-Za-z]/, '')));
            const minNum = Math.min(...nums);
            const maxNum = Math.max(...nums);
            const rangeMatch = /^([A-Za-z])?(\d+)-(\d+)$/.exec(rawMotif);
            const isCollapsedRange =
              !rawMotif ||
              (rangeMatch !== null &&
                Number(rangeMatch[2]) === minNum &&
                Number(rangeMatch[3]) === maxNum &&
                maxNum - minNum > hotspotTokens.length);
            if (isCollapsedRange) {
              const recovered = hotspotTokens
                .map((t) => (/^\d+$/.test(t) ? `A${t}` : t.toUpperCase()))
                .join(', ');
              paramsPatch.motifSpan = recovered;
              paramsPatch.unindexedMotif = true;
              detail = `Keep discrete motif (${recovered}, unindexed)`;
            }
          }
          issues.push({
            severity: 'warning',
            code: 'rfd4-hotspots-in-motif-mode',
            nodeId: node.id,
            message: `${node.label ?? 'RFD4-Proteina'} is in Motif mode, but binder hotspots ("${rawHotspots}") are set${paramsPatch.motifSpan ? ` and motifSpan "${rawMotif}" collapses all residues ${rawMotif} instead of just ${String(paramsPatch.motifSpan)}` : ''}.`,
            fix: {
              label: detail,
              nodeId: node.id,
              paramsPatch,
            },
          });
        }
      }
    }

    // #63: Generic wired-template unconditional guard covering all multi-mode template generators
    // (`rfd4-proteina`, `rfd3`, `genie3`, `protpardelle`, `boltzgen`, `rfdiffusion`).
    if (
      node.adapterId === 'rfd4-proteina' ||
      node.adapterId === 'rfd3' ||
      node.adapterId === 'genie3' ||
      node.adapterId === 'protpardelle' ||
      node.adapterId === 'boltzgen' ||
      node.adapterId === 'rfdiffusion'
    ) {
      const modeKey = node.adapterId === 'rfd4-proteina' || node.adapterId === 'rfd3' ? 'mode' : 'taskMode';
      const activeMode = String(node.params[modeKey] ?? 'unconditional').trim().toLowerCase();
      const tplEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
      const hasYaml =
        node.adapterId === 'rfd4-proteina' &&
        String(node.params.customSpecYaml ?? '').trim().length > 0;
      if (tplEdge && activeMode === 'unconditional' && !hasYaml) {
        const displayName = node.label ?? manifest?.shortName ?? node.adapterId;
        const tplNode = graph.nodes.find((n) => n.id === tplEdge.fromNode);
        const tplBenchId =
          tplNode?.adapterId === 'pdb-input' ? String(tplNode.params.benchmarkId ?? '').trim() : '';
        const tplBench = tplBenchId ? getBenchmarkById(tplBenchId) : undefined;

        let code: string;
        let availableTasks: string;
        let targetMode: string;
        let fixLabel: string;

        if (node.adapterId === 'rfd4-proteina') {
          code = 'rfd4-unused-template';
          availableTasks = 'Motif, Binder, or Partial Diffusion';
          targetMode = 'partial';
          fixLabel = 'Switch to Partial Diffusion';
        } else if (node.adapterId === 'rfd3') {
          code = 'rfd3-unused-template';
          availableTasks = 'Motif / All-Atom Scaffolding, Binder, or Partial Diffusion';
          const preferBinder =
            String(node.params.hotspots ?? '').trim().length > 0 ||
            tplBench?.category === 'binder';
          const preferMotif =
            String(node.params.contig ?? '').trim().length > 0 ||
            String(node.params.unindex ?? '').trim().length > 0 ||
            String(node.params.ligand ?? '').trim().length > 0 ||
            tplBench?.category === 'motif' ||
            tplBench?.category === 'enzyme-ligand';
          targetMode = preferBinder ? 'binder' : preferMotif ? 'motif' : 'partial';
          fixLabel = preferBinder
            ? 'Switch to Binder Design'
            : preferMotif
              ? 'Switch to Motif / All-Atom Scaffolding'
              : 'Switch to Partial Diffusion';
        } else if (node.adapterId === 'protpardelle') {
          code = 'protpardelle-unused-template';
          availableTasks = 'Motif Scaffolding or Partial Diffusion';
          const hasMotifSel =
            String(node.params.motifSelection ?? '').trim().length > 0 ||
            tplBench?.category === 'motif';
          targetMode = hasMotifSel ? 'motif' : 'partial';
          fixLabel = hasMotifSel ? 'Switch to Motif Scaffolding' : 'Switch to Partial Diffusion';
        } else if (node.adapterId === 'rfdiffusion') {
          code = 'rfdiffusion-unused-template';
          availableTasks = 'Motif Scaffolding';
          targetMode = 'motif';
          fixLabel = 'Switch to Motif Scaffolding';
        } else if (node.adapterId === 'boltzgen') {
          code = 'boltzgen-unused-template';
          availableTasks = 'Binder Design or Motif Scaffolding';
          const preferBinder =
            String(node.params.hotspots ?? '').trim().length > 0 ||
            tplBench?.category === 'binder';
          targetMode = preferBinder ? 'binder' : 'motif';
          fixLabel = preferBinder ? 'Switch to Binder Design' : 'Switch to Motif Scaffolding';
        } else {
          // genie3
          code = 'genie3-unused-template';
          availableTasks = 'Motif Scaffolding or Binder Design';
          const hasMotifCue =
            (String(node.params.motifSpan ?? '').trim().length > 0 ||
              String(node.params.contig ?? '').trim().length > 0) &&
            !String(node.params.contig ?? '').includes('/0') &&
            !String(node.params.lockChain ?? '').trim();
          targetMode = hasMotifCue ? 'motif' : 'binder';
          fixLabel = hasMotifCue ? 'Switch to Motif Scaffolding' : 'Switch to Binder Design';
        }

        issues.push({
          severity: 'warning',
          code,
          nodeId: node.id,
          message: `${displayName} has a wired input structure but is set to Unconditional mode (the structure will be ignored). Pick a conditioned task (${availableTasks}).`,
          fix: {
            label: fixLabel,
            nodeId: node.id,
            paramsPatch: { [modeKey]: targetMode },
          },
        });
      }
    }

    // Disjoint motifSpan check: warn when a motif node's motifSpan/motifSelection has 0 overlap
    // with its own contig residue spans or upstream PDB Input's cropSegments (catches AI hallucinations & typos).
    const isMotifNode =
      (node.adapterId === 'rfd4-proteina' && String(node.params.mode ?? 'unconditional') === 'motif') ||
      (node.adapterId === 'genie3' && String(node.params.taskMode ?? 'unconditional') === 'motif') ||
      (node.adapterId === 'protpardelle' && String(node.params.taskMode ?? 'unconditional') === 'motif') ||
      (node.adapterId === 'rfdiffusion' &&
        String(node.params.taskMode ?? node.params.mode ?? 'unconditional') === 'motif');
    if (isMotifNode) {
      const motifParamKey = node.adapterId === 'protpardelle' ? 'motifSelection' : 'motifSpan';
      const rawMotif = String(
        node.adapterId === 'protpardelle'
          ? (node.params.motifSelection ?? node.params.motifSpan ?? '')
          : (node.params.motifSpan ?? ''),
      ).trim();
      const motifSpans = rawMotif
        ? parseResidueRangeTokens(rawMotif).map((s) => ({
            chain: s.chain.toUpperCase(),
            start: Math.min(s.start, s.end),
            end: Math.max(s.start, s.end),
          }))
        : [];
      if (motifSpans.length > 0) {
        const rawContig = String(node.params.contig ?? '').trim();
        const contigSpans = rawContig ? parseContigTemplateSpans(rawContig) : [];
        if (contigSpans.length > 0 && !residueSpansOverlapAny(motifSpans, contigSpans)) {
          issues.push({
            severity: 'warning',
            code: 'motif-span-disjoint-from-contig',
            nodeId: node.id,
            message: `${node.label ?? node.adapterId} motifSpan "${rawMotif}" does not overlap any template residue in contig "${rawContig}". Check for a residue-numbering typo or stale motif span.`,
          });
        }
        const tplEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
        const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;
        if (tplNode?.adapterId === 'pdb-input') {
          const cropSpec = String(tplNode.params.cropSegments ?? '').trim();
          const cropSpans = cropSpec ? parseProteinCropSpans(cropSpec) : [];
          if (cropSpans.length > 0 && !residueSpansOverlapAny(motifSpans, cropSpans)) {
            const benchId = String(tplNode.params.benchmarkId ?? '').trim();
            const bench = benchId ? getBenchmarkById(benchId) : undefined;
            const fixMotif = bench?.conditionParams.motifSpan;
            issues.push({
              severity: 'warning',
              code: 'motif-span-disjoint-from-crop',
              nodeId: node.id,
              message: `${node.label ?? node.adapterId} ${motifParamKey} "${rawMotif}" is completely disjoint from upstream cropSegments "${cropSpec}" — 0 motif residues will match.`,
              ...(fixMotif
                ? {
                    fix: {
                      label: `Use ${bench!.shortLabel} motif (${fixMotif})`,
                      nodeId: node.id,
                      paramsPatch: { [motifParamKey]: fixMotif },
                    },
                  }
                : {}),
            });
          }
        }
      }
    }

    const isFullBackboneConsumer =
      (node.adapterId === 'web-protein-mpnn' &&
        String(node.params.family ?? 'solublempnn') !== 'ca') ||
      node.adapterId === 'ligandmpnn' ||
      node.adapterId === 'lasermpnn' ||
      node.adapterId === 'antifold-esmif1' ||
      node.adapterId === 'thermompnn' ||
      node.adapterId === 'gnina';

    if (isFullBackboneConsumer) {
      const edge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
      const src = edge ? graph.nodes.find((n) => n.id === edge.fromNode) : null;
      if (src) {
        const up = inferNodeOutputProfile(src, graph, cache, visiting, outputsByNode);
        if (up.isCaOnly || (up.missingBackboneCount ?? 0) > 0) {
          const consumerName =
            node.label ??
            (node.adapterId === 'web-protein-mpnn'
              ? 'ProteinMPNN'
              : node.adapterId === 'antifold-esmif1'
                ? String(node.params.variant ?? 'antifold') === 'esmif1'
                  ? 'ESM-IF1'
                  : 'AntiFold'
                : node.adapterId);
          const srcName = up.fromGenie3 ? 'Genie 3' : 'Upstream structure';
          let msg: string;
          if (node.adapterId === 'web-protein-mpnn') {
            msg = up.fromGenie3
              ? `Genie 3 outputs Cα-only coordinates, whereas ${consumerName} requires full backbone coordinates. Switch to Cα-ProteinMPNN.`
              : up.isCaOnly
                ? `Upstream structure is Cα-only, whereas ${consumerName} requires full backbone coordinates. Switch to Cα-ProteinMPNN.`
                : `Upstream structure has ${up.missingBackboneCount} residue(s) with incomplete backbone coordinates — ${consumerName} requires full backbone coordinates. Switch to Cα-ProteinMPNN.`;
          } else if (node.adapterId === 'thermompnn') {
            msg = up.isCaOnly
              ? `${srcName} outputs Cα-only coordinates, whereas ${consumerName} requires full N/CA/C/O backbone atoms. Use a full-backbone generator or refold via LocalFold first.`
              : `Upstream structure has ${up.missingBackboneCount} residue(s) with incomplete backbone coordinates — ${consumerName} requires full N/CA/C/O backbone coordinates.`;
          } else {
            msg = up.isCaOnly
              ? `${srcName} outputs Cα-only coordinates, whereas ${consumerName} requires full N/CA/C/O backbone atoms. Switch upstream to a full-backbone generator (RFD4-Proteina, RFdiffusion, ProtPardelle-1c, BoltzGen-1, or La-Proteina) or replace ${consumerName} with Cα-ProteinMPNN.`
              : `Upstream structure has ${up.missingBackboneCount} residue(s) with incomplete backbone coordinates — ${consumerName} requires full N/CA/C/O backbone coordinates.`;
          }
          issues.push({
            severity: up.isCaOnly ? 'error' : 'warning',
            code: 'mpnn-ca-only-backbone',
            nodeId: node.id,
            message: msg,
            ...(node.adapterId === 'web-protein-mpnn'
              ? {
                  fix: {
                    label: 'Switch to Cα-ProteinMPNN',
                    nodeId: node.id,
                    paramsPatch: { family: 'ca', modelName: 'v_48_020' },
                    nodeLabel: 'Cα-ProteinMPNN',
                  },
                }
              : {}),
          });
        }
      }
    }

    if (node.adapterId === 'web-protein-mpnn') {
      const edge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
      const src = edge ? graph.nodes.find((n) => n.id === edge.fromNode) : null;
      if (src) {
        const up = inferNodeOutputProfile(src, graph, cache, visiting, outputsByNode);
        if (up.entities.has('ligand') || up.entities.has('dna') || up.entities.has('rna')) {
          const nonProt = [...up.entities].filter((e) => e !== 'protein').join('/');
          issues.push({
            severity: 'warning',
            code: 'mpnn-non-protein-stripped',
            nodeId: node.id,
            message: `${node.label ?? 'ProteinMPNN'} designs protein chains only — upstream ${nonProt.toUpperCase()} coordinates will be ignored during inverse folding.`,
          });
        }
      }
    }

    if (node.adapterId === 'antifold-esmif1') {
      const variant = String(node.params.variant ?? 'antifold');
      if (variant === 'antifold') {
        const edge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
        const src = edge ? graph.nodes.find((n) => n.id === edge.fromNode) : null;
        if (src) {
          const liveStruct = outputsByNode?.[src.id]?.structure as StructureArtifact | undefined;
          const pdbText = src.adapterId === 'pdb-input' ? String(src.params.pdbText ?? '').trim() : '';
          let parsedSummary: CachedStructureBioSummary | null = null;
          if (pdbText) {
            try {
              const cropSpec = String(src.params.cropSegments ?? '').trim();
              parsedSummary = getCachedStructureBioSummary(pdbText, cropSpec);
            } catch {
              parsedSummary = null;
            }
          }
          const benchId =
            src.adapterId === 'pdb-input' ? String(src.params.benchmarkId ?? '').trim() : '';
          const bench = benchId ? getBenchmarkById(benchId) : undefined;
          const pdbIdUpper =
            src.adapterId === 'pdb-input' ? String(src.params.pdbId ?? '').trim().toUpperCase() : '';
          const isDeNovoGenerator =
            src.adapterId === 'rfd4-proteina' ||
            src.adapterId === 'rfd3' ||
            src.adapterId === 'protpardelle' ||
            src.adapterId === 'rfdiffusion' ||
            src.adapterId === 'boltzgen' ||
            src.adapterId === 'la-proteina' ||
            src.adapterId === 'proteina' ||
            src.adapterId === 'sid-protein' ||
            src.adapterId === 'genie2' ||
            src.adapterId === 'frameflow' ||
            src.adapterId === 'foldflow2' ||
            src.adapterId === 'genie3';
          const isDemoBundle =
            src.adapterId === 'pdb-input' &&
            String(src.params.mode ?? 'demo-bundle') === 'demo-bundle';
          const isKnownNonAntibodyPdb =
            KNOWN_NON_ANTIBODY_PDBS.has(pdbIdUpper) ||
            BENCHMARK_TARGETS.some((b) => b.pdbId === pdbIdUpper);
          const targetChain = String(node.params.targetChain ?? '').trim().toUpperCase();
          const chainTypes = liveStruct?.molecularContext?.chainTypes
            ? Object.values(liveStruct.molecularContext.chainTypes)
            : [];
          const headerText = [
            pdbText.slice(0, 8192),
            parsedSummary?.headerMetaText ?? '',
            liveStruct?.cifText ? extractStructureHeaderMetaText(liveStruct.cifText) : '',
            liveStruct?.pdbText ? liveStruct.pdbText.slice(0, 8192) : '',
            liveStruct?.label ?? '',
            String(src.params.filename ?? ''),
          ].join('\n');
          const hasAntibodyHeaderCue =
            !isDeNovoGenerator &&
            !isDemoBundle &&
            !bench &&
            !isKnownNonAntibodyPdb &&
            /\b(?:ANTIBODY|IMMUNOGLOBULIN(?!-LIKE)|FAB|SCFV|VHH|NANOBODY|IGG|IMGT|KABAT|SINGLE[- ]DOMAIN)\b/i.test(
              headerText,
            );
          const hasParsedStructure =
            (liveStruct !== undefined && liveStruct.length >= 3) ||
            (parsedSummary !== null && parsedSummary.nResidues >= 3);
          const availableChains = new Set<string>([
            ...(liveStruct?.chainIds ?? []).map((c) => c.toUpperCase()),
            ...(parsedSummary?.chainIds ?? []).map((c) => c.toUpperCase()),
          ]);
          const targetChainValidForStructure =
            (targetChain === 'H' || targetChain === 'L') &&
            (hasParsedStructure
              ? availableChains.size === 0 || availableChains.has(targetChain)
              : !isDeNovoGenerator && !isDemoBundle && !bench && !isKnownNonAntibodyPdb);
          const numberedChains =
            liveStruct?.sourceResidues?.length && liveStruct.sequence
              ? buildNumberedChains(liveStruct.sourceResidues, liveStruct.sequence)
              : parsedSummary?.numberedChains ?? [];
          const hasAntibodyContext =
            targetChainValidForStructure ||
            chainTypes.some(
              (ct) => ct === 'antibody-H' || ct === 'antibody-L' || ct === 'nanobody',
            ) ||
            Boolean(liveStruct?.chainIds?.some((c) => c === 'H' || c === 'L')) ||
            Boolean(parsedSummary?.chainIds?.some((c) => c === 'H' || c === 'L')) ||
            (!isDeNovoGenerator &&
              !isDemoBundle &&
              !bench &&
              !isKnownNonAntibodyPdb &&
              hasAntibodyChainHallmarks(numberedChains)) ||
            hasAntibodyHeaderCue;

          if (!hasAntibodyContext) {
            if (
              isDeNovoGenerator ||
              isDemoBundle ||
              Boolean(bench) ||
              isKnownNonAntibodyPdb ||
              hasParsedStructure
            ) {
              const regionsLabel = String(node.params.regions ?? '').trim();
              const regionNote = regionsLabel ? ` (${regionsLabel} may match 0 residues)` : '';
              issues.push({
                severity: 'warning',
                code: 'antifold-non-antibody',
                nodeId: node.id,
                message: `AntiFold is trained for IMGT/Kabat-numbered antibody/nanobody CDRs, whereas upstream appears to be a non-antibody structure${regionNote}. Switch to ESM-IF1 (or replace with SolubleMPNN).`,
                fix: {
                  label: 'Switch to ESM-IF1',
                  nodeId: node.id,
                  paramsPatch: { variant: 'esmif1' },
                  nodeLabel: 'ESM-IF1',
                },
              });
            }
          }
        }
      }
    }

    if (node.adapterId === 'ligandmpnn' || node.adapterId === 'lasermpnn') {
      const useLigCtx =
        node.adapterId === 'lasermpnn' ? true : node.params.useLigandContext !== false;
      if (useLigCtx) {
        const edge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
        const src = edge ? graph.nodes.find((n) => n.id === edge.fromNode) : null;
        if (src) {
          const up = inferNodeOutputProfile(src, graph, cache, visiting, outputsByNode);
          const hasNonProt = up.entities.has('ligand') || up.entities.has('dna') || up.entities.has('rna');
          if (!hasNonProt && src.adapterId === 'pdb-input') {
            const cropSpec = String(src.params.cropSegments ?? '').trim();
            if (cropSpec) {
              const liveStruct = outputsByNode?.[src.id]?.structure as StructureArtifact | undefined;
              let droppedLigands = liveStruct?.diagnostics?.omittedLigands ?? [];
              if (droppedLigands.length === 0) {
                const rawText = String(src.params.pdbText ?? '').trim();
                if (rawText) {
                  try {
                    droppedLigands = getCachedStructureBioSummary(rawText).diagnostics.ligandNames;
                  } catch {
                    droppedLigands = [];
                  }
                }
              }
              if (droppedLigands.length > 0) {
                const keepSpec = `${cropSpec}, ${droppedLigands.join(', ')}`;
                issues.push({
                  severity: 'warning',
                  code: 'crop-dropped-ligands',
                  nodeId: node.id,
                  message: `${node.label ?? node.adapterId} has ligand context enabled, but upstream crop "${cropSpec}" dropped ${droppedLigands.join(', ')} (running apo protein-only).`,
                  fix: {
                    label: `Keep ${droppedLigands.join(', ')} in crop`,
                    nodeId: src.id,
                    paramsPatch: { cropSegments: keepSpec },
                  },
                });
              }
            }
          }

          // #82: Warn LASErMPNN strictly when an organic-ligand HETATM lacks explicit hydrogens
          // (never false-positive on monoatomic ions like ZN/CA/MG/FE2, buffers like SO4, or MSE).
          if (node.adapterId === 'lasermpnn') {
            const liveStruct = outputsByNode?.[src.id]?.structure as StructureArtifact | undefined;
            let inspectText = liveStruct?.cifText || liveStruct?.pdbText || '';
            if (!inspectText && src.adapterId === 'pdb-input') {
              const rawText = String(src.params.pdbText ?? '').trim();
              if (rawText) {
                const cropSpec = String(src.params.cropSegments ?? '').trim();
                try {
                  inspectText = cropSpec
                    ? cropRawAllAtomStructure(rawText, cropSpec).text
                    : rawText;
                } catch {
                  inspectText = rawText;
                }
              }
            }
            if (inspectText) {
              const unprot = inspectUnprotonatedOrganicLigands(inspectText);
              if (unprot.hasUnprotonated) {
                issues.push({
                  severity: 'warning',
                  code: 'lasermpnn-unprotonated-ligand',
                  nodeId: node.id,
                  message: `${node.label ?? 'LASErMPNN'} expects explicit hydrogens on small-molecule organic ligands (${unprot.unprotonatedLigands.join(', ')} has 0 H atoms). Standard X-ray PDB/mmCIF files omit hydrogens, which can degrade ligand-contact packing — protonate externally or use LigandMPNN.`,
                });
              }
            }
          }
        }
      }
    }

    // Wave 2 / CT-10: RFdiffusion symmetry rounding & protein-only HETATM policy notices
    if (node.adapterId === 'rfdiffusion') {
      const taskMode = String(node.params.taskMode ?? 'unconditional').trim().toLowerCase();
      if (taskMode === 'symmetry') {
        const SYM_COPIES: Record<string, number> = { c2: 2, c3: 3, c4: 4, c6: 6, d2: 4, d3: 6 };
        const sym = String(node.params.symmetry ?? 'c3').trim().toLowerCase();
        const copies = SYM_COPIES[sym] ?? 3;
        const rawLen = Math.max(16, Math.min(256, Math.round(Number(node.params.length ?? 80))));
        const maxSymLen = Math.floor(256 / copies) * copies;
        const effectiveLen = Math.min(
          maxSymLen,
          Math.max(copies * 8, Math.round(rawLen / copies) * copies),
        );
        if (effectiveLen !== rawLen) {
          issues.push({
            severity: 'warning',
            code: 'rfdiffusion-symmetry-length-adjust',
            nodeId: node.id,
            message: `${node.label ?? 'RFdiffusion'} (${sym.toUpperCase()}, ${copies} copies) adjusts length ${rawLen} aa → ${effectiveLen} aa (minimum ${copies * 8} aa, multiple of ${copies}).`,
            fix: {
              label: `Set length to ${effectiveLen} aa`,
              nodeId: node.id,
              paramsPatch: { length: effectiveLen },
            },
          });
        }
      }
      if (taskMode === 'motif') {
        const tplEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
        const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;
        if (tplNode) {
          const tplProfile = inferNodeOutputProfile(tplNode, graph, cache, visiting, outputsByNode);
          const bench =
            tplNode.adapterId === 'pdb-input' && tplNode.params.benchmarkId
              ? getBenchmarkById(String(tplNode.params.benchmarkId))
              : undefined;
          const liveStruct = outputsByNode?.[tplNode.id]?.structure as StructureArtifact | undefined;
          const rawPdbText =
            tplNode.adapterId === 'pdb-input' ? String(tplNode.params.pdbText ?? '').trim() : '';
          const cropSpec =
            tplNode.adapterId === 'pdb-input' ? String(tplNode.params.cropSegments ?? '').trim() : '';
          let parsedSummary: CachedStructureBioSummary | null = null;
          if (!liveStruct && rawPdbText) {
            try {
              parsedSummary = getCachedStructureBioSummary(rawPdbText, cropSpec);
            } catch {
              parsedSummary = null;
            }
          }
          const pdbId =
            (tplNode.adapterId === 'pdb-input' ? String(tplNode.params.pdbId ?? '').trim() : '') ||
            bench?.pdbId ||
            (/\b([0-9][A-Za-z0-9]{3})\b/.exec(liveStruct?.label ?? '')?.[1] ?? '');
          const diag = liveStruct?.diagnostics ?? parsedSummary?.diagnostics;
          const policy = evaluateProteinOnlyStructurePolicy({
            modelName: node.label ?? 'RFdiffusion',
            benchmark: bench,
            pdbId,
            proteinResidueCount: liveStruct
              ? liveStruct.length
              : parsedSummary
                ? parsedSummary.nResidues
                : undefined,
            ligandNames: diag?.ligandNames,
            nucleicAcidChains: diag?.nucleicAcidChains,
            hasLigands: diag ? diag.hasLigands : tplProfile.entities.has('ligand'),
            hasNucleicAcids: diag
              ? diag.hasNucleicAcids
              : tplProfile.entities.has('dna') || tplProfile.entities.has('rna'),
          });
          if (policy.action === 'hard-block') {
            issues.push({
              severity: policy.reason === 'insufficient-protein' ? 'error' : 'warning',
              code: 'rfdiffusion-unsupported-entities',
              nodeId: node.id,
              message: policy.message!,
            });
          } else if (policy.action === 'warn') {
            issues.push({
              severity: 'warning',
              code: 'rfdiffusion-unsupported-entities',
              nodeId: node.id,
              message: policy.message!,
            });
          }
        }
      }
    }

    // Wave 2 / CT-10: Genie 2 motifRange template requirement & protein-only HETATM policy notice
    if (node.adapterId === 'genie2') {
      const motifRange = String(node.params.motifRange ?? '').trim();
      const tplEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'template');
      const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;
      if (motifRange && !tplNode) {
        issues.push({
          severity: 'error',
          code: 'genie2-missing-template',
          nodeId: node.id,
          message: `${node.label ?? 'Genie 2'} has motifRange "${motifRange}" set but no connected "template" structure.`,
        });
      } else if (tplNode) {
        const tplProfile = inferNodeOutputProfile(tplNode, graph, cache, visiting, outputsByNode);
        const bench =
          tplNode.adapterId === 'pdb-input' && tplNode.params.benchmarkId
            ? getBenchmarkById(String(tplNode.params.benchmarkId))
            : undefined;
        const liveStruct = outputsByNode?.[tplNode.id]?.structure as StructureArtifact | undefined;
        const rawPdbText =
          tplNode.adapterId === 'pdb-input' ? String(tplNode.params.pdbText ?? '').trim() : '';
        const cropSpec =
          tplNode.adapterId === 'pdb-input' ? String(tplNode.params.cropSegments ?? '').trim() : '';
        let parsedSummary: CachedStructureBioSummary | null = null;
        if (!liveStruct && rawPdbText) {
          try {
            parsedSummary = getCachedStructureBioSummary(rawPdbText, cropSpec);
          } catch {
            parsedSummary = null;
          }
        }
        const pdbId =
          (tplNode.adapterId === 'pdb-input' ? String(tplNode.params.pdbId ?? '').trim() : '') ||
          bench?.pdbId ||
          (/\b([0-9][A-Za-z0-9]{3})\b/.exec(liveStruct?.label ?? '')?.[1] ?? '');
        const diag = liveStruct?.diagnostics ?? parsedSummary?.diagnostics;
        const policy = evaluateProteinOnlyStructurePolicy({
          modelName: node.label ?? 'Genie 2',
          benchmark: bench,
          pdbId,
          proteinResidueCount: liveStruct
            ? liveStruct.length
            : parsedSummary
              ? parsedSummary.nResidues
              : undefined,
          ligandNames: diag?.ligandNames,
          nucleicAcidChains: diag?.nucleicAcidChains,
          hasLigands: diag ? diag.hasLigands : tplProfile.entities.has('ligand'),
          hasNucleicAcids: diag
            ? diag.hasNucleicAcids
            : tplProfile.entities.has('dna') || tplProfile.entities.has('rna'),
        });
        if (policy.action === 'hard-block') {
          issues.push({
            severity: policy.reason === 'insufficient-protein' ? 'error' : 'warning',
            code: 'genie2-unsupported-entities',
            nodeId: node.id,
            message: policy.message!,
          });
        } else if (policy.action === 'warn') {
          issues.push({
            severity: 'warning',
            code: 'genie2-unsupported-entities',
            nodeId: node.id,
            message: policy.message!,
          });
        }
      }
    }

    // Wave 2 / CT-10: FoldFlow 2 single-candidate & single-chain conditioning notices
    if (node.adapterId === 'foldflow2') {
      const seqEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'sequences');
      const seqSrc = seqEdge ? graph.nodes.find((n) => n.id === seqEdge.fromNode) : null;
      if (seqSrc) {
        const up = inferNodeOutputProfile(seqSrc, graph, cache, visiting, outputsByNode);
        const liveSeqs = outputsByNode?.[seqSrc.id]?.sequences as SequenceSet | undefined;
        const candCount = liveSeqs?.candidates?.length
          ?? (seqSrc.adapterId === 'web-protein-mpnn' ||
              seqSrc.adapterId === 'ligandmpnn' ||
              seqSrc.adapterId === 'lasermpnn' ||
              seqSrc.adapterId === 'antifold-esmif1' ||
              seqSrc.adapterId === 'evodiff'
            ? Math.max(1, Math.round(Number(seqSrc.params.numSequences ?? 1)))
            : seqSrc.adapterId === 'sequence-input'
              ? Math.max(
                  1,
                  String(seqSrc.params.sequence ?? '')
                    .split(/\r?\n/)
                    .filter((l) => l.trim().startsWith('>')).length || 1,
                )
              : 1);
        if (candCount > 1) {
          issues.push({
            severity: 'warning',
            code: 'foldflow2-first-candidate-only',
            nodeId: node.id,
            message: `${node.label ?? 'FoldFlow 2'} conditions only on the first sequence candidate from "${seqSrc.label ?? seqSrc.id}" (${candCount} candidates upstream; remaining ${candCount - 1} are ignored).`,
          });
        }
        const firstSeqStr =
          liveSeqs?.candidates?.[0]?.sequence ??
          (seqSrc.adapterId === 'sequence-input' ? String(seqSrc.params.sequence ?? '') : '');
        if (up.chainCount > 1 || firstSeqStr.includes(':')) {
          issues.push({
            severity: 'warning',
            code: 'foldflow2-multichain-collapsed',
            nodeId: node.id,
            message: `${node.label ?? 'FoldFlow 2'} is a monomer SE(3) generator and strips ":" chain breaks from multi-chain sequence inputs, collapsing them into a single continuous chain.`,
          });
        }
      }
    }

    // Wave 2 / CT-10: ESM-C requires at least one of sequences or structure
    if (node.adapterId === 'esmc') {
      const hasInput = graph.edges.some(
        (e) => e.toNode === node.id && (e.toPort === 'sequences' || e.toPort === 'structure'),
      );
      if (!hasInput) {
        issues.push({
          severity: 'error',
          code: 'esmc-missing-input',
          nodeId: node.id,
          message: `${node.label ?? 'ESM-C 300M'} requires a wired "sequences" or "structure" input.`,
        });
      }
    }

    // #82 + Wave 2 / CT-10: Dyna-1 required input presence & seq-only calibration warning
    if (node.adapterId === 'dyna1') {
      const inputMode = String(node.params.inputMode ?? 'auto').trim().toLowerCase();
      const structEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
      const seqEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'sequences');
      if (!structEdge && !seqEdge) {
        issues.push({
          severity: 'error',
          code: 'dyna1-missing-input',
          nodeId: node.id,
          message: `${node.label ?? 'Dyna-1'} requires a wired "structure" or "sequences" input.`,
        });
      }
      const structSrc = structEdge ? graph.nodes.find((n) => n.id === structEdge.fromNode) : null;
      const structProf = structSrc
        ? inferNodeOutputProfile(structSrc, graph, cache, visiting, outputsByNode)
        : null;
      const willRunSeqOnly =
        (Boolean(structEdge) || Boolean(seqEdge)) &&
        (inputMode === 'seq-only' ||
          (inputMode === 'auto' && Boolean(seqEdge) && (!structSrc || Boolean(structProf?.isCaOnly))));
      if (willRunSeqOnly) {
        issues.push({
          severity: 'warning',
          code: 'dyna1-seq-only-calibration',
          nodeId: node.id,
          message: `${node.label ?? 'Dyna-1'} is configured to run in sequence-only mode (without 3D full-backbone structure tokens). Sequence-only exchange probabilities are less calibrated than seq+struct mode — wire a full-backbone 3D structure when available.`,
        });
      }
    }

    // GNINA 1.3: requires a wired all-atom structure and either a holo organic ligand, RCSB CCD (dock), or pasted SDF/MOL2
    if (node.adapterId === 'gnina') {
      const structEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
      const taskMode = String(node.params.taskMode ?? 'score');
      const hasExternalLig = String(node.params.ligandSdf ?? '').trim().length > 0;
      const dockCcd = taskMode === 'dock' ? String(node.params.dockLigandCcd ?? '').trim().toUpperCase() : '';
      const hasDockCcd = dockCcd.length > 0;
      const hasPocketSpec = String(node.params.pocketResidues ?? '').trim().length > 0;
      const requestedCode = String(node.params.ligandCode ?? '').trim().toUpperCase();
      if (taskMode === 'dock' && !hasExternalLig && hasDockCcd) {
        if (!/^[A-Z0-9]{1,5}$/.test(dockCcd)) {
          issues.push({
            severity: 'error',
            code: 'gnina-invalid-dock-ccd',
            nodeId: node.id,
            message: `${node.label ?? 'GNINA 1.3'} has invalid Ligand CCD code "${dockCcd}" (expected 1–5 alphanumeric characters, e.g. BTN, ATP, I4B).`,
          });
        } else if (
          STANDARD_AA_RESNAMES.has(dockCcd) ||
          NUCLEIC_ACID_RESNAMES.has(dockCcd) ||
          classifyHetatmCompId(dockCcd) !== 'organic-ligand'
        ) {
          issues.push({
            severity: 'error',
            code: 'gnina-non-organic-dock-ccd',
            nodeId: node.id,
            message: `${node.label ?? 'GNINA 1.3'} Ligand CCD "${dockCcd}" is not an organic small-molecule ligand (monoatomic ions, waters, buffers, and standard residues cannot be docked as small molecules).`,
          });
        }
      }
      if (!structEdge) {
        issues.push({
          severity: 'error',
          code: 'gnina-missing-input',
          nodeId: node.id,
          message: `${node.label ?? 'GNINA 1.3'} requires a wired "structure" input (holo protein–ligand complex or receptor).`,
        });
      } else {
        const src = graph.nodes.find((n) => n.id === structEdge.fromNode);
        if (src) {
          const liveStruct = outputsByNode?.[src.id]?.[structEdge.fromPort] as StructureArtifact | undefined;
          let ligandNames: string[] | null = null;
          let knownApo = false;
          if (liveStruct?.diagnostics) {
            ligandNames = liveStruct.diagnostics.ligandNames ?? [];
            knownApo = true;
          } else if (src.adapterId === 'pdb-input') {
            const srcMode = String(src.params.mode ?? 'demo-bundle');
            const rawText = String(src.params.pdbText ?? '').trim();
            const cropSpec = srcMode === 'demo-bundle' ? '' : String(src.params.cropSegments ?? '').trim();
            if (srcMode === 'demo-bundle') {
              ligandNames = [];
              knownApo = true;
            } else if (rawText) {
              try {
                ligandNames = getCachedStructureBioSummary(rawText, cropSpec).diagnostics.ligandNames;
                knownApo = true;
              } catch {
                ligandNames = null;
              }
            } else if (cropSpec) {
              const cropHints = summarizeCropSegments(cropSpec);
              if (cropHints.valid && cropHints.hasExplicitTokens) {
                ligandNames = cropHints.ligandNames;
                knownApo = true;
              }
            } else if (srcMode === 'alphafold-db') {
              // AlphaFold DB entries are predicted apo protein monomers (no organic small-molecule ligands)
              ligandNames = [];
              knownApo = true;
            } else if (srcMode === 'pdb-id') {
              const srcBenchId = String(src.params.benchmarkId ?? '').trim();
              const srcBench = srcBenchId ? getBenchmarkById(srcBenchId) : undefined;
              if (srcBench && !srcBench.entities.includes('ligand')) {
                ligandNames = [];
                knownApo = true;
              } else if (srcBench?.cropSegments) {
                const benchHints = summarizeCropSegments(srcBench.cropSegments);
                if (benchHints.valid && benchHints.hasExplicitTokens) {
                  ligandNames = benchHints.ligandNames;
                  knownApo = true;
                }
              }
            }
          } else if (src.adapterId === 'localfold-af2') {
            const modelId = String(src.params.model ?? 'monomer');
            const meta =
              LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
            const supportsHolo = meta.family === 'af3' || meta.family === 'esmfold2';
            ligandNames = supportsHolo
              ? resolveLocalFoldEffectiveConfig({ modelKey: modelId, params: src.params }).ligandCcdCodes
              : [];
            knownApo = true;
          } else if (src.adapterId === 'rfd4-proteina') {
            const up = inferNodeOutputProfile(src, graph, cache, visiting, outputsByNode);
            if (!up.entities.has('ligand')) {
              ligandNames = [];
              knownApo = true;
            } else if (
              String(src.params.mode ?? 'unconditional') === 'preset' &&
              LIGAND_RFD4_PRESETS.has(String(src.params.presetId ?? ''))
            ) {
              ligandNames = ['SAM'];
              knownApo = true;
            } else if (!String(src.params.customSpecYaml ?? '').trim()) {
              const tplEdge = graph.edges.find((e) => e.toNode === src.id && e.toPort === 'structure');
              const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;
              if (tplNode) {
                const tplInfo = resolveTemplateArtifactsAndEntities({
                  contig: String(src.params.contig ?? '').trim(),
                  hbondSelection: String(src.params.hbondSelection ?? '').trim(),
                  tplNode,
                  outputsByNode,
                });
                const tplHasLiveOrText =
                  Boolean((outputsByNode?.[tplNode.id]?.structure as StructureArtifact | undefined)?.diagnostics) ||
                  (tplNode.adapterId === 'pdb-input' && Boolean(String(tplNode.params.pdbText ?? '').trim()));
                if (tplHasLiveOrText) {
                  ligandNames = tplInfo.retainedLigands;
                  knownApo = true;
                } else if (tplNode.adapterId === 'pdb-input') {
                  const tplMode = String(tplNode.params.mode ?? 'demo-bundle');
                  const tplCrop =
                    tplMode === 'demo-bundle' ? '' : String(tplNode.params.cropSegments ?? '').trim();
                  if (tplCrop) {
                    const cropHints = summarizeCropSegments(tplCrop);
                    if (cropHints.valid && cropHints.hasExplicitTokens) {
                      ligandNames = cropHints.ligandNames;
                      knownApo = true;
                    }
                  } else {
                    const tplBenchId = String(tplNode.params.benchmarkId ?? '').trim();
                    const tplBench = tplBenchId ? getBenchmarkById(tplBenchId) : undefined;
                    if (tplBench?.cropSegments) {
                      const benchHints = summarizeCropSegments(tplBench.cropSegments);
                      if (benchHints.valid && benchHints.hasExplicitTokens) {
                        ligandNames = benchHints.ligandNames;
                        knownApo = true;
                      }
                    }
                  }
                }
              }
            }
          } else {
            const up = inferNodeOutputProfile(src, graph, cache, visiting, outputsByNode);
            if (!up.entities.has('ligand')) {
              ligandNames = [];
              knownApo = true;
            }
          }
          if (knownApo && ligandNames !== null) {
            const organic = ligandNames.filter((c) => {
              const code = c.trim().toUpperCase();
              return (
                /^[A-Z0-9]{1,5}$/.test(code) &&
                !STANDARD_AA_RESNAMES.has(code) &&
                !NUCLEIC_ACID_RESNAMES.has(code) &&
                classifyHetatmCompId(code) === 'organic-ligand'
              );
            });
            if (hasExternalLig || hasDockCcd) {
              if (taskMode === 'dock' && organic.length === 0 && !hasPocketSpec) {
                issues.push({
                  severity: 'error',
                  code: 'gnina-dock-missing-pocket',
                  nodeId: node.id,
                  message: `${node.label ?? 'GNINA 1.3'} is docking ${hasDockCcd ? `CCD "${dockCcd}"` : 'an external ligand'} into apo structure "${src.label ?? src.id}" with no autobox ligand. Specify Pocket Box Residues (e.g. A45, A79, A90) or wire a holo complex.`,
                });
              }
            } else if (organic.length === 0) {
              issues.push({
                severity: 'error',
                code: 'gnina-no-organic-ligand',
                nodeId: node.id,
                message:
                  taskMode === 'dock'
                    ? `${node.label ?? 'GNINA 1.3'} found no organic ligand in "${src.label ?? src.id}"${ligandNames.length > 0 ? ` (only ${ligandNames.join(', ')})` : ''}. Enter a Ligand CCD Code (e.g. BTN, ATP, I4B) + Pocket Box Residues, paste a ligand SDF/MOL2, or wire a holo complex.`
                    : `${node.label ?? 'GNINA 1.3'} found no organic small-molecule ligand in "${src.label ?? src.id}"${ligandNames.length > 0 ? ` (only ${ligandNames.join(', ')})` : ''}. Wire a holo protein–ligand complex (e.g. 1STP, 1CTT, 184L) or paste a ligand SDF/MOL2 in Advanced parameters.`,
              });
            } else if (
              requestedCode &&
              !requestedCode.includes(':') &&
              !organic.some((c) => c.toUpperCase() === requestedCode)
            ) {
              issues.push({
                severity: 'warning',
                code: 'gnina-ligand-code-mismatch',
                nodeId: node.id,
                message: `${node.label ?? 'GNINA 1.3'} target ligand "${requestedCode}" was not found in "${src.label ?? src.id}" (detected organic ligands: ${organic.join(', ')}).`,
              });
            }
          }
        }
      }
    }

    if (node.adapterId === 'localfold-af2') {
      const modelId = String(node.params.model ?? 'monomer');
      const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
      const nodeName = node.label ?? 'LocalFold';
      const modelDisplay = nodeName.toLowerCase().includes(meta.shortLabel.toLowerCase())
        ? nodeName
        : `${nodeName} (${meta.shortLabel})`;
      const supportsHolo = meta.family === 'af3' || meta.family === 'esmfold2';
      const lfCfg = supportsHolo
        ? resolveLocalFoldEffectiveConfig({ modelKey: modelId, params: node.params })
        : null;
      const hasExplicitAf3NonProt = Boolean(lfCfg?.hasHoloContext);
      if (lfCfg) {
        const invalidCcds = lfCfg.ligandCcdCodes.filter((c) => !/^[A-Z0-9]{1,5}$/.test(c));
        if (invalidCcds.length > 0) {
          issues.push({
            severity: 'error',
            code: 'localfold-invalid-ligand-ccd',
            nodeId: node.id,
            message: `${modelDisplay} has invalid Ligand CCD code(s): ${invalidCcds.join(', ')} (expected 1–5 alphanumeric characters, e.g. ZN, DHZ, ATP).`,
          });
        }
      }
      const edge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'sequences');
      const src = edge ? graph.nodes.find((n) => n.id === edge.fromNode) : null;
      if (!src) continue;
      const up = inferNodeOutputProfile(src, graph, cache, visiting, outputsByNode);
      const nonProtEntities = new Set<BioEntityKind>([
        ...[...up.entities].filter((e) => e !== 'protein'),
        ...(up.strippedNonProtein ?? []),
      ]);

      if (nonProtEntities.size > 0 && !hasExplicitAf3NonProt) {
        const nonProt = [...nonProtEntities].join('/');
        const holoHints = extractUpstreamHoloHints(graph, node.id, outputsByNode);
        const holoPatch: Record<string, unknown> = {};
        if (holoHints.ligandCcdCodes.length > 0) {
          holoPatch.ligandCcdCodes = holoHints.ligandCcdCodes.join(', ');
        }
        if (holoHints.dnaSequences.length > 0) {
          holoPatch.dnaSequences = holoHints.dnaSequences.join(':');
        }
        const holoParts = [
          holoHints.ligandCcdCodes.length > 0 ? holoHints.ligandCcdCodes.join(', ') : '',
          holoHints.dnaSequences.length > 0 ? 'DNA' : '',
        ].filter(Boolean);
        const holoSummary = holoParts.join(' + ');

        if (!supportsHolo) {
          issues.push({
            severity: 'warning',
            code: 'af2-non-protein-entity',
            nodeId: node.id,
            message: `${modelDisplay} is protein-only, while upstream involves ${nonProt.toUpperCase()}. Switch to RoseTTAFold3 or Boltz-2${holoSummary ? ` with ${holoSummary}` : ''} for holo co-folding.`,
            fix: {
              label: holoSummary ? `Switch to RoseTTAFold3 (${holoSummary})` : 'Switch to RoseTTAFold3',
              nodeId: node.id,
              paramsPatch: { model: 'rosettafold3', ...holoPatch },
              nodeLabel: 'RoseTTAFold 3',
            },
          });
        } else {
          issues.push({
            severity: 'warning',
            code: 'af3-apo-refold-note',
            nodeId: node.id,
            message: holoSummary
              ? `${modelDisplay} is set to apo (protein-only) while upstream includes ${nonProt.toUpperCase()} (${holoSummary}) — add holo context to co-fold with the ligand/nucleic acid.`
              : `${modelDisplay} is set to apo (protein-only) — add ligand CCD codes or DNA/RNA sequences in Parameters to co-fold with upstream ${nonProt.toUpperCase()}.`,
            ...(Object.keys(holoPatch).length > 0
              ? {
                  fix: {
                    label: `Co-fold with ${holoSummary}`,
                    nodeId: node.id,
                    paramsPatch: holoPatch,
                  },
                }
              : {}),
          });
        }
      } else if (supportsHolo && hasExplicitAf3NonProt && lfCfg) {
        const holoHints = extractUpstreamHoloHints(graph, node.id, outputsByNode);
        const hintLigands = holoHints.ligandCcdCodes;
        const hintDnas = holoHints.dnaSequences;
        const hasUpstreamHoloHints = hintLigands.length > 0 || hintDnas.length > 0;
        const cfgLigands = lfCfg.ligandCcdCodes;
        const cfgDnas = lfCfg.dnaSeqs;

        if (hasUpstreamHoloHints) {
          const cfgLigSet = new Set(cfgLigands.map((c) => c.toUpperCase()));
          const hintLigSet = new Set(hintLigands.map((c) => c.toUpperCase()));
          const ligandsMatch =
            cfgLigSet.size === hintLigSet.size &&
            [...hintLigSet].every((c) => cfgLigSet.has(c));
          const cfgDnaSet = new Set(cfgDnas.map((s) => s.toUpperCase()));
          const hintDnaSet = new Set(hintDnas.map((s) => s.toUpperCase()));
          const dnasMatch =
            cfgDnaSet.size === hintDnaSet.size &&
            [...hintDnaSet].every((s) => cfgDnaSet.has(s));

          if (!ligandsMatch || !dnasMatch) {
            const cfgDesc = [
              cfgLigands.length > 0 ? cfgLigands.join(', ') : '',
              cfgDnas.length > 0 ? `${cfgDnas.length} DNA strand${cfgDnas.length === 1 ? '' : 's'}` : '',
            ]
              .filter(Boolean)
              .join(' + ');
            const upDesc = [
              hintLigands.length > 0 ? hintLigands.join(', ') : '',
              hintDnas.length > 0 ? `${hintDnas.length} DNA strand${hintDnas.length === 1 ? '' : 's'}` : '',
            ]
              .filter(Boolean)
              .join(' + ');
            const syncPatch: Record<string, unknown> = {
              ligandCcdCodes: hintLigands.join(', '),
              dnaSequences: hintDnas.join(':'),
            };
            issues.push({
              severity: 'warning',
              code: 'af3-holo-mismatch',
              nodeId: node.id,
              message: `${modelDisplay} holo context (${cfgDesc || 'none'}) does not match upstream holo entities (${upDesc}). Sync ligand CCD codes and DNA sequences to match upstream.`,
              fix: {
                label: `Sync holo context (${upDesc})`,
                nodeId: node.id,
                paramsPatch: syncPatch,
              },
            });
          }
        } else if (
          nonProtEntities.size === 0 &&
          hasUpstreamStructureOrigin(graph, node.id) &&
          (cfgLigands.length > 0 || cfgDnas.length > 0)
        ) {
          const cfgDesc = [
            cfgLigands.length > 0 ? cfgLigands.join(', ') : '',
            cfgDnas.length > 0 ? `${cfgDnas.length} DNA strand${cfgDnas.length === 1 ? '' : 's'}` : '',
          ]
            .filter(Boolean)
            .join(' + ');
          issues.push({
            severity: 'warning',
            code: 'af3-stale-holo-on-apo',
            nodeId: node.id,
            message: `${modelDisplay} has stale holo params (${cfgDesc}) while upstream is strictly apo (protein-only). Clear holo ligands/DNA to avoid phantom co-folding.`,
            fix: {
              label: 'Clear stale holo params',
              nodeId: node.id,
              paramsPatch: { ligandCcdCodes: '', dnaSequences: '' },
            },
          });
        }
      } else if (up.stoichiometry === 'multimer' && meta.family === 'af2') {
        const monoDesc = nodeName.toLowerCase().includes(meta.shortLabel.toLowerCase())
          ? `${nodeName} (monomer)`
          : `${nodeName} is set to ${meta.shortLabel} (monomer)`;
        issues.push({
          severity: 'warning',
          code: 'multimer-into-af2-monomer',
          nodeId: node.id,
          message: `Upstream produces a ${up.chainCount}-chain complex, but ${monoDesc}. Switch to AlphaFold2-Multimer v3 or Boltz-2 for proper complex folding & ipTM.`,
          fix: {
            label: 'Switch to AF2-Multimer v3',
            nodeId: node.id,
            paramsPatch: { model: 'multimer' },
            nodeLabel: 'AF2-Multimer 1',
          },
        });
      }

      const msaMode = String(node.params.msaMode ?? 'single');
      if (msaMode === 'single' && meta.family !== 'esmfold2') {
        if (up.stoichiometry === 'multimer' && up.hasLockedTarget) {
          issues.push({
            severity: 'warning',
            code: 'binder-single-seq-msa',
            nodeId: node.id,
            message: `${modelDisplay} is folding a multi-chain binder complex in single-sequence mode. Enable Target-Only MSA so the locked target chain uses evolutionary alignments while the de novo binder stays query-only.`,
            fix: {
              label: 'Enable Target-Only MSA',
              nodeId: node.id,
              paramsPatch: { msaMode: 'target-only' },
            },
          });
        } else if (up.fromNaturalProtein) {
          issues.push({
            severity: 'warning',
            code: 'natural-single-seq-msa',
            nodeId: node.id,
            message: `${modelDisplay} is folding an uploaded/natural protein in single-sequence mode. Enable MMseqs2 MSA (ColabFold API) for accurate structure prediction.`,
            fix: {
              label: 'Enable MMseqs2 MSA',
              nodeId: node.id,
              paramsPatch: { msaMode: 'mmseqs2' },
            },
          });
        }
      }

      const nodeLimit = foldLimitOpts?.resolveNodeFoldLimits?.(node) ?? foldLimitOpts;
      const customMax =
        typeof nodeLimit?.maxFoldResidues === 'number' &&
        Number.isFinite(nodeLimit.maxFoldResidues) &&
        nodeLimit.maxFoldResidues >= 64
          ? Math.round(nodeLimit.maxFoldResidues)
          : null;
      const isHostGpu = Boolean(nodeLimit?.isHostGpu);
      const defaultTokenLimit = meta.family === 'af3' || meta.family === 'esmfold2' ? 384 : 512;
      const tokenLimit =
        customMax !== null
          ? isHostGpu
            ? customMax
            : customMax < defaultTokenLimit
              ? customMax
              : Math.max(defaultTokenLimit, Math.round(customMax * 0.85))
          : defaultTokenLimit;
      if (up.estimatedTokens > tokenLimit) {
        const foldCeilingTiers = [768, 1024, 1536, 2048, 3072, 4096] as const;
        const suggestedTier =
          !isHostGpu && up.estimatedTokens <= 4096
            ? (foldCeilingTiers.find(
                (t) => t > (customMax ?? 0) && Math.round(t * 0.85) >= up.estimatedTokens,
              ) ??
              foldCeilingTiers.find((t) => t > (customMax ?? 0) && t >= up.estimatedTokens))
            : undefined;
        issues.push({
          severity: 'warning',
          code: 'webgpu-token-ceiling',
          nodeId: node.id,
          message: isHostGpu
            ? `Estimated length (~${up.estimatedTokens} tokens) exceeds the configured Host GPU ceiling (${tokenLimit}) for ${meta.shortLabel}. Raise "Max fold len" in the GPU menu if your host VRAM permits.`
            : `Estimated length (~${up.estimatedTokens} tokens) exceeds the recommended browser WebGPU ceiling (${tokenLimit}) for ${meta.shortLabel} and may hit GPU memory limits.`,
          ...(suggestedTier !== undefined
            ? {
                fix: {
                  label: `Raise Max fold len to ${suggestedTier} aa`,
                  nodeId: node.id,
                  paramsPatch: {},
                  computeBackendPatch: { maxFoldResidues: suggestedTier },
                },
              }
            : {}),
        });
      }
    }
  }

  return issues;
}

/**
 * Layer 2: Pre-dispatch runtime input guard executed in scheduler.ts BEFORE
 * spawning a WebGPU worker or downloading 97–612 MB of weights.
 */
export function validateRuntimeNodeInputs(
  node: WorkflowNode,
  inputs: Record<string, CanonicalArtifact>,
  foldLimitOpts?: BioFoldLimitOptions,
): {
  warnings: string[];
  excludedCandidates?: ExcludedCandidateRecord[];
  selectedCandidateIds?: string[];
} {
  const warnings: string[] = [];
  let excludedCandidates: ExcludedCandidateRecord[] | undefined;
  let selectedCandidateIds: string[] | undefined;

  if (node.adapterId === 'localfold-af2') {
    const seqSet = inputs.sequences as SequenceSet | undefined;
    if (!seqSet?.candidates?.length) {
      throw new Error('LocalFold requires an upstream SequenceSet with at least one sequence.');
    }
    const modelId = String(node.params.model ?? 'monomer');
    const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
    const defaultHardLimit = meta.family === 'af3' || meta.family === 'esmfold2' ? 512 : 768;
    const nodeLimit = foldLimitOpts?.resolveNodeFoldLimits?.(node) ?? foldLimitOpts;
    const customMax =
      typeof nodeLimit?.maxFoldResidues === 'number' &&
      Number.isFinite(nodeLimit.maxFoldResidues) &&
      nodeLimit.maxFoldResidues >= 64
        ? Math.round(nodeLimit.maxFoldResidues)
        : null;
    const isHostGpu = Boolean(nodeLimit?.isHostGpu);
    const hardLimit = customMax !== null ? customMax : defaultHardLimit;
    if (meta.family === 'esmfold2') {
      const issue = esmfold2ParamIssue(node.params);
      if (issue) throw new Error(issue);
    }

    // Resolve the EXACT same candidate subset that localfold.worker.ts will execute.
    const work = selectFoldWorkItems(seqSet, node.params);
    selectedCandidateIds = work.selected.map((c) => c.id);
    if (work.excludedCandidates.length > 0) {
      excludedCandidates = work.excludedCandidates;
      warnings.push(
        `LocalFold maxSequences=${work.maxSequences} excludes ${work.excludedCandidates.length} candidate(s): ${work.excludedCandidates.map((c) => c.id).join(', ')}.`,
      );
    }

    const extraPolymerTokens =
      meta.family === 'af3' || meta.family === 'esmfold2'
        ? (Array.isArray(node.params.dnaSequences)
            ? (node.params.dnaSequences as unknown[]).map((s) => String(s ?? ''))
            : String(node.params.dnaSequences ?? '').split(/[:,;\s]+/)
          ).reduce((sum, s) => sum + s.replace(/[^A-Za-z]/g, '').length, 0) +
          (Array.isArray(node.params.rnaSequences)
            ? (node.params.rnaSequences as unknown[]).map((s) => String(s ?? ''))
            : String(node.params.rnaSequences ?? '').split(/[:,;\s]+/)
          ).reduce((sum, s) => sum + s.replace(/[^A-Za-z]/g, '').length, 0) +
          (Array.isArray(node.params.ligandCcdCodes)
            ? (node.params.ligandCcdCodes as unknown[]).map((s) => String(s ?? '').trim()).filter(Boolean).length
            : String(node.params.ligandCcdCodes ?? '')
                .split(/[,;\s]+/)
                .map((s) => s.trim())
                .filter(Boolean).length)
        : 0;

    for (const cand of work.selected) {
      const len = cand.sequence.replace(/[\s/:,]+/g, '').length;
      if (len < 5) {
        throw new Error(`Refusing to fold "${cand.id}": sequence is too short (${len} aa; minimum 5 aa).`);
      }
      const totalTokens = len + extraPolymerTokens;
      if (totalTokens > hardLimit) {
        const sizeDesc =
          extraPolymerTokens > 0
            ? `${len} aa + ${extraPolymerTokens} nucleic/ligand = ${totalTokens} tokens`
            : `${len} aa`;
        throw new Error(
          `Refusing to fold "${cand.id}" (${sizeDesc}) with ${meta.shortLabel}: exceeds ${isHostGpu ? 'host' : 'browser'} WebGPU memory ceiling (${hardLimit} residues). Raise "Max fold len" in the GPU / Compute menu${isHostGpu ? '' : ', switch to Self-Hosted,'} or crop the target domain before running.`,
        );
      }
      if (meta.family === 'esmfold2') esmfold2Chains(cand, hardLimit);
      if (cand.chainLengths && cand.chainLengths.length > 1 && meta.family === 'af2') {
        warnings.push(
          `Folding ${cand.chainLengths.length}-chain complex (${cand.chainLengths.join(' + ')} aa) with AF2 monomer (using unpaired multi-chain A3M); consider AF2-Multimer v3 or Boltz-2 for accurate interface scoring.`,
        );
      }
    }
  }

  if (node.adapterId === 'ligandmpnn' || node.adapterId === 'lasermpnn') {
    const useLigCtx =
      node.adapterId === 'lasermpnn' ? true : node.params.useLigandContext !== false;
    const tpl = inputs.structure as StructureArtifact | undefined;
    if (useLigCtx && tpl) {
      const hasLig = Boolean(tpl.diagnostics?.hasLigands || (tpl.diagnostics?.ligandNames?.length ?? 0) > 0);
      const hasNa = Boolean(
        tpl.diagnostics?.hasNucleicAcids || (tpl.diagnostics?.nucleicAcidChains?.length ?? 0) > 0,
      );
      if (!hasLig && !hasNa) {
        warnings.push(
          `${node.label ?? node.adapterId} has ligand context enabled, but the input structure contains 0 ligands/ions/nucleic acids — running protein-only (apo).`,
        );
      } else if (node.adapterId === 'lasermpnn') {
        const rawText = tpl.cifText || tpl.pdbText || '';
        const unprot = inspectUnprotonatedOrganicLigands(rawText);
        if (unprot.hasUnprotonated) {
          warnings.push(
            `${node.label ?? 'LASErMPNN'} expects explicit hydrogens on small-molecule organic ligands (${unprot.unprotonatedLigands.join(', ')} has 0 H atoms).`,
          );
        }
      }
    }
  }

  if (node.adapterId === 'rfd4-proteina') {
    const mode = String(node.params.mode ?? 'unconditional');
    const tpl = inputs.structure as StructureArtifact | undefined;
    if (
      (mode === 'motif' || mode === 'binder' || mode === 'partial' || mode === 'inverse-folding') &&
      !tpl?.cifText &&
      !tpl?.pdbText
    ) {
      throw new Error(
        `RFD4-Proteina (${mode} mode) requires a wired structure input (e.g. from PDB Input).`,
      );
    }
  }

  if (node.adapterId === 'rfd3') {
    const mode = String(node.params.mode ?? 'unconditional').trim().toLowerCase();
    const tpl = inputs.structure as StructureArtifact | undefined;
    if ((mode === 'motif' || mode === 'binder' || mode === 'partial') && !tpl?.cifText && !tpl?.pdbText) {
      throw new Error(
        `RFdiffusion3 (${mode} mode) requires a wired structure input (e.g. from PDB Input).`,
      );
    }
  }

  if (node.adapterId === 'genie3') {
    const taskMode = String(node.params.taskMode ?? 'unconditional');
    const tpl = inputs.structure as StructureArtifact | undefined;
    if (tpl && taskMode !== 'unconditional') {
      const bench = tpl.benchmarkId ? getBenchmarkById(tpl.benchmarkId) : undefined;
      const inferredPdbId =
        bench?.pdbId ??
        (/\b([0-9][A-Za-z0-9]{3})\b/.exec(tpl.label ?? '')?.[1]?.toUpperCase() || undefined);
      const policy = evaluateProteinOnlyStructurePolicy({
        modelName: node.label ?? 'Genie 3',
        benchmark: bench,
        pdbId: inferredPdbId,
        proteinResidueCount: tpl.length,
        ligandNames: tpl.diagnostics?.ligandNames,
        nucleicAcidChains: tpl.diagnostics?.nucleicAcidChains,
        hasLigands: tpl.diagnostics?.hasLigands,
        hasNucleicAcids: tpl.diagnostics?.hasNucleicAcids,
      });
      if (policy.action === 'hard-block') {
        throw new Error(policy.message);
      }
      if (policy.action === 'warn' && policy.message) {
        warnings.push(policy.message);
      }
    }
    const inlineLock = String(node.params.lockChain ?? '').trim();
    const inlineMotif = String(node.params.motifSpan ?? '').trim();
    const inlineContig = String(node.params.contig ?? '').trim();
    if (taskMode === 'motif' && !inlineMotif && !inlineContig) {
      throw new Error(
        'Genie 3 motif mode requires a motifSpan (e.g. A16-35) or contig specifying motif residues.',
      );
    }
    if (
      (taskMode === 'motif' || taskMode === 'binder' || Boolean(inlineLock) || Boolean(inlineMotif)) &&
      !tpl?.cifText &&
      !tpl?.pdbText
    ) {
      throw new Error(
        `Genie 3 (${taskMode} mode) requires a wired structure input (e.g. from PDB Input).`,
      );
    }
  }

  if (node.adapterId === 'web-protein-mpnn') {
    const struct = inputs.structure as StructureArtifact | undefined;
    if (!struct?.pdbText) {
      throw new Error('ProteinMPNN requires an upstream 3D structure.');
    }
    if (struct.length < 3) {
      throw new Error(
        `ProteinMPNN requires at least 3 protein residues (received ${struct.length}). Upstream non-protein ligand/DNA-only structures cannot be inverse-folded directly.`,
      );
    }
    const family = String(node.params.family ?? 'solublempnn');
    if (family !== 'ca') {
      const cov = inspectArtifactBackboneCoverage(struct);
      if (cov.isCaOnly || (struct.length >= 3 && cov.completeBackboneCount < 3)) {
        throw new Error(
          `Upstream structure is Cα-only (${cov.completeBackboneCount}/${struct.length} full-backbone residues), whereas ${node.label ?? 'ProteinMPNN'} requires full backbone N/CA/C/O coordinates. Switch ProteinMPNN to Cα-ProteinMPNN (family: "ca").`,
        );
      }
      if (cov.missingBackboneCount > 0) {
        warnings.push(
          `Upstream structure has ${cov.missingBackboneCount} residue(s) missing N/C backbone atoms; consider switching to Cα-ProteinMPNN.`,
        );
      }
    }
    if (struct.diagnostics?.hasLigands || struct.diagnostics?.hasNucleicAcids) {
      const extras = [
        ...(struct.diagnostics.hasLigands ? [`ligands (${struct.diagnostics.ligandNames.join(', ')})`] : []),
        ...(struct.diagnostics.hasNucleicAcids ? [`nucleic acid chains (${struct.diagnostics.nucleicAcidChains.join(', ')})`] : []),
      ];
      warnings.push(`ProteinMPNN ignores non-protein ${extras.join(' and ')} and designs protein residues only.`);
    }
  }

  if (
    node.adapterId === 'ligandmpnn' ||
    node.adapterId === 'lasermpnn' ||
    node.adapterId === 'antifold-esmif1' ||
    node.adapterId === 'thermompnn' ||
    node.adapterId === 'gnina'
  ) {
    const struct = inputs.structure as StructureArtifact | undefined;
    if (struct) {
      const cov = inspectArtifactBackboneCoverage(struct);
      const consumerName =
        node.label ??
        (node.adapterId === 'antifold-esmif1'
          ? String(node.params.variant ?? 'antifold') === 'esmif1'
            ? 'ESM-IF1'
            : 'AntiFold'
          : node.adapterId);
      if (cov.isCaOnly || (struct.length >= 3 && cov.completeBackboneCount < 3)) {
        throw new Error(
          `Upstream structure is Cα-only (${cov.completeBackboneCount}/${struct.length} full-backbone residues), whereas ${consumerName} requires full N/CA/C/O backbone atoms. Use a full-backbone generator or Cα-ProteinMPNN.`,
        );
      }
      if (cov.missingBackboneCount > 0) {
        warnings.push(
          `Upstream structure has ${cov.missingBackboneCount} residue(s) missing backbone atoms; ${consumerName} requires full backbone coordinates.`,
        );
      }
    }
  }

  if (node.adapterId === 'rfdiffusion' || node.adapterId === 'genie2') {
    const tpl = (
      node.adapterId === 'genie2' ? inputs.template : inputs.structure
    ) as StructureArtifact | undefined;
    if (tpl) {
      const bench = tpl.benchmarkId ? getBenchmarkById(tpl.benchmarkId) : undefined;
      const inferredPdbId =
        bench?.pdbId ??
        (/\b([0-9][A-Za-z0-9]{3})\b/.exec(tpl.label ?? '')?.[1]?.toUpperCase() || undefined);
      const policy = evaluateProteinOnlyStructurePolicy({
        modelName: node.label ?? (node.adapterId === 'genie2' ? 'Genie 2' : 'RFdiffusion'),
        benchmark: bench,
        pdbId: inferredPdbId,
        proteinResidueCount: tpl.length,
        ligandNames: tpl.diagnostics?.ligandNames,
        nucleicAcidChains: tpl.diagnostics?.nucleicAcidChains,
        hasLigands: tpl.diagnostics?.hasLigands,
        hasNucleicAcids: tpl.diagnostics?.hasNucleicAcids,
      });
      if (policy.action === 'hard-block') {
        throw new Error(policy.message);
      }
      if (policy.action === 'warn' && policy.message) {
        warnings.push(policy.message);
      }
    }
  }

  return {
    warnings,
    ...(excludedCandidates ? { excludedCandidates } : {}),
    ...(selectedCandidateIds ? { selectedCandidateIds } : {}),
  };
}
