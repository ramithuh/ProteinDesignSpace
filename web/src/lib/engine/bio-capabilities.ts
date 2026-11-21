// Two-Layer Biological Capability Schema:
// Layer 1: Static graph/wire capability contracts (catches monomer-vs-multimer, ligand/DNA stripping,
//          pure-DNA sequence input, and token-ceiling risks before clicking Run).
// Layer 2: Runtime pre-dispatch artifact validation (called in scheduler.ts before launching GPU workers).

import { LOCALFOLD_MODEL_OPTIONS } from './registry';
import { getBenchmarkById } from './benchmarks';
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
} {
  if (struct.atomPresence && struct.atomPresence.length === struct.length) {
    let missingCount = 0;
    let allCaOnly = struct.length > 0;
    for (let i = 0; i < struct.length; i++) {
      const p = struct.atomPresence[i];
      if (!p.hasN || !p.hasCA || !p.hasC || (!p.hasO && !p.oReconstructed)) {
        missingCount++;
      }
      if (!p.hasCA || p.hasN || p.hasC) {
        allCaOnly = false;
      }
    }
    return { missingBackboneCount: missingCount, isCaOnly: allCaOnly };
  }
  if (struct.observedBackboneMask && struct.observedBackboneMask.length === struct.length) {
    const required = BACKBONE_MASK_N | BACKBONE_MASK_CA | BACKBONE_MASK_C;
    let missingCount = 0;
    let allCaOnly = struct.length > 0;
    for (let i = 0; i < struct.length; i++) {
      const m = struct.observedBackboneMask[i];
      if ((m & required) !== required) missingCount++;
      if ((m & required) !== BACKBONE_MASK_CA) allCaOnly = false;
    }
    return { missingBackboneCount: missingCount, isCaOnly: allCaOnly };
  }
  if (struct.pdbText && struct.length > 0) {
    try {
      const summary = getCachedStructureBioSummary(struct.pdbText);
      if (summary.nResidues === struct.length) {
        return {
          missingBackboneCount: summary.missingBackboneCount,
          isCaOnly: summary.isCaOnly,
        };
      }
    } catch {
      /* ignore synthetic pdbText */
    }
  }
  return { missingBackboneCount: 0, isCaOnly: false };
}

export interface FoldWorkItemSelection {
  maxSequences: number;
  selected: SequenceCandidate[];
  excludedCandidates: ExcludedCandidateRecord[];
  totalCandidates: number;
}

/**
 * Pure work-item resolver shared by preflight validation (`validateRuntimeNodeInputs`)
 * and `localfold.worker.ts` so both inspect the EXACT identical candidate slice and
 * surface excluded candidates explicitly.
 */
export function selectFoldWorkItems(
  seqSet: SequenceSet | undefined | null,
  params?: Record<string, unknown>,
): FoldWorkItemSelection {
  const rawMax = Number(params?.maxSequences ?? 2);
  const maxSequences = Math.max(1, Math.min(8, Math.round(Number.isFinite(rawMax) ? rawMax : 2)));
  const all = seqSet?.candidates ?? [];
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
    const pdbText = String(node.params.pdbText ?? '').trim();
    const benchId = String(node.params.benchmarkId ?? '').trim();
    const bench = benchId ? getBenchmarkById(benchId) : undefined;
    const liveStruct = outputsByNode?.[node.id]?.structure as StructureArtifact | undefined;

    if (liveStruct && liveStruct.kind === 'StructureArtifact' && liveStruct.diagnostics) {
      const diag = liveStruct.diagnostics;
      const entities = new Set<BioEntityKind>();
      if (liveStruct.length > 0 || (!diag.hasLigands && !diag.hasNucleicAcids)) {
        entities.add('protein');
      }
      if (diag.hasLigands) entities.add('ligand');
      if (diag.hasNucleicAcids) entities.add('dna');
      if (bench) {
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
        estimatedTokens: liveStruct.length > 0 ? liveStruct.length : (bench?.approxTokens ?? 80),
        missingBackboneCount: coverage.missingBackboneCount,
        isCaOnly: coverage.isCaOnly,
      };
    } else if ((mode === 'custom' || mode === 'pdb-id') && pdbText.length > 0) {
      try {
        const cropSpec = String(node.params.cropSegments ?? '').trim();
        const summary = getCachedStructureBioSummary(pdbText, cropSpec);
        const entities = new Set<BioEntityKind>();
        if (summary.nResidues > 0 || (!summary.diagnostics.hasLigands && !summary.diagnostics.hasNucleicAcids)) {
          entities.add('protein');
        }
        if (summary.diagnostics.hasLigands) entities.add('ligand');
        if (summary.diagnostics.hasNucleicAcids) entities.add('dna');
        if (bench) {
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
          estimatedTokens: summary.nResidues > 0 ? summary.nResidues : (bench?.approxTokens ?? 80),
          missingBackboneCount: summary.missingBackboneCount,
          isCaOnly: summary.isCaOnly,
        };
      } catch {
        profile.estimatedTokens = bench?.approxTokens ?? 80;
        if (bench) {
          profile.entities = new Set<BioEntityKind>(bench.entities);
        }
      }
    } else if (bench) {
      const contigChains = bench.conditionParams.contig.includes('/0')
        ? bench.conditionParams.contig.split('/0').filter((s) => s.trim().length > 0).length
        : 1;
      const cropChains = bench.cropSegments
        ? new Set(
            bench.cropSegments
              .split(/[,\s]+/)
              .map((tok) => /^([A-Za-z])(?:$|\d)/.exec(tok.trim())?.[1]?.toUpperCase())
              .filter(Boolean),
          ).size
        : 1;
      const chainCount = Math.max(1, contigChains, cropChains);
      profile = {
        entities: new Set<BioEntityKind>(bench.entities),
        stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
        chainCount,
        estimatedTokens: bench.approxTokens,
      };
    } else {
      profile.estimatedTokens = 53;
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
    };
  } else if (node.adapterId === 'rfd4-proteina') {
    const mode = String(node.params.mode ?? 'unconditional');
    const tpl = upstreamProfileForPort('structure');
    const entities = new Set<BioEntityKind>(['protein']);
    let chainCount = 1;
    let estimatedTokens = Number(node.params.length ?? 70);
    const contig = String(node.params.contig ?? '');
    const lockChain = String(node.params.lockChain ?? '').trim();

    if (mode === 'preset') {
      const presetId = String(node.params.presetId ?? '');
      if (MULTICHAIN_RFD4_PRESETS.has(presetId)) {
        chainCount = 2;
        estimatedTokens = 208;
      } else if (LIGAND_RFD4_PRESETS.has(presetId)) {
        entities.add('ligand');
        estimatedTokens = 143;
      } else if (DNA_RFD4_PRESETS.has(presetId)) {
        entities.add('dna');
        chainCount = 3;
        estimatedTokens = 128;
      }
    }
    if (mode === 'binder' || Boolean(lockChain) || contig.includes('/0')) {
      chainCount = Math.max(chainCount, 2);
      estimatedTokens = Math.max(estimatedTokens, (tpl?.estimatedTokens ?? 100) + Number(node.params.length ?? 70));
    }
    if (tpl && (mode === 'partial' || mode === 'inverse-folding' || mode === 'motif' || mode === 'binder' || mode === 'conditioned')) {
      for (const e of tpl.entities) entities.add(e);
      chainCount = Math.max(chainCount, tpl.chainCount);
      estimatedTokens = Math.max(estimatedTokens, tpl.estimatedTokens);
    }
    const liveStruct = outputsByNode?.[node.id]?.structure as StructureArtifact | undefined;
    if (liveStruct && liveStruct.kind === 'StructureArtifact' && liveStruct.diagnostics) {
      if (liveStruct.diagnostics.hasLigands) entities.add('ligand');
      if (liveStruct.diagnostics.hasNucleicAcids) entities.add('dna');
      chainCount = Math.max(chainCount, liveStruct.diagnostics.chainCount);
      if (liveStruct.length > 0) estimatedTokens = liveStruct.length;
    }
    profile = {
      entities,
      stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
      chainCount,
      estimatedTokens,
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
    const inlineContig = String(node.params.contig ?? '');
    const isBinderTask =
      taskMode === 'binder' ||
      Boolean(inlineLock) ||
      inlineContig.includes('/0') ||
      (Boolean(tpl) && taskMode !== 'motif' && tplBench?.category !== 'motif');
    const chainCount = Math.max(1, isBinderTask ? 2 : 1);
    const binderLen = Number(node.params.length ?? 80);
    const estimatedTokens = tpl
      ? tplBench
        ? tplBench.approxTokens
        : binderLen + tpl.estimatedTokens
      : binderLen;
    profile = {
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
      chainCount,
      estimatedTokens,
      isCaOnly: true,
      missingBackboneCount: estimatedTokens,
      fromGenie3: true,
    };
  } else if (node.adapterId === 'web-protein-mpnn') {
    const st = upstreamProfileForPort('structure');
    const strippedSet = new Set<BioEntityKind>(st?.strippedNonProtein ?? []);
    if (st) {
      for (const e of st.entities) {
        if (e !== 'protein') strippedSet.add(e);
      }
    }
    const strippedNonProtein = strippedSet.size > 0 ? [...strippedSet] : undefined;
    profile = {
      // ProteinMPNN emits protein sequences only (strips ligands/DNA), but records stripped context for downstream apo warnings
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: st?.stoichiometry ?? 'monomer',
      chainCount: st?.chainCount ?? 1,
      estimatedTokens: st?.estimatedTokens ?? 80,
      ...(strippedNonProtein ? { strippedNonProtein } : {}),
    };
  } else if (node.adapterId === 'localfold-af2') {
    const seq = upstreamProfileForPort('sequences');
    profile = {
      entities: new Set<BioEntityKind>(seq ? [...seq.entities] : ['protein']),
      stoichiometry: seq?.stoichiometry ?? 'monomer',
      chainCount: seq?.chainCount ?? 1,
      estimatedTokens: seq?.estimatedTokens ?? 80,
      ...(seq?.strippedNonProtein ? { strippedNonProtein: seq.strippedNonProtein } : {}),
    };
  }

  visiting.delete(node.id);
  cache.set(node.id, profile);
  return profile;
}

/**
 * Layer 1: Static biological capability validation across the workflow graph.
 * Accepts optional materialized `outputsByNode` to inspect live `StructureArtifact.diagnostics`.
 * Returns actionable warnings/errors with 1-click fixes for biological modality mismatches.
 */
export function validateBioCapabilities(
  graph: WorkflowGraph,
  outputsByNode?: Record<string, Record<string, CanonicalArtifact>>,
): BioCompatibilityIssue[] {
  const issues: BioCompatibilityIssue[] = [];
  const cache = new Map<string, NodeBioProfile>();
  const visiting = new Set<string>();

  for (const node of graph.nodes) {
    const ownProfile = inferNodeOutputProfile(node, graph, cache, visiting, outputsByNode);

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
        if (bench && !bench.supportsGenie3) {
          issues.push({
            severity: 'warning',
            code: 'genie3-unsupported-benchmark',
            nodeId: node.id,
            message: `${bench.shortLabel} requires all-atom enzyme/ligand/DNA conditioning, which is supported by RFD4-Proteina rather than ${node.label ?? 'Genie 3'}.`,
          });
        } else if (tplProfile.entities.has('ligand') || tplProfile.entities.has('dna') || tplProfile.entities.has('rna')) {
          const nonProt = [...tplProfile.entities].filter((e) => e !== 'protein').join('/');
          issues.push({
            severity: 'warning',
            code: 'genie3-unsupported-entities',
            nodeId: node.id,
            message: `Upstream template contains ${nonProt.toUpperCase()} entities, which ${node.label ?? 'Genie 3'} (protein-only Cα diffusion) does not support. Use RFD4-Proteina for all-atom ligand/DNA conditioning.`,
          });
        }
      }
    }

    if (node.adapterId === 'rfd4-proteina') {
      const mode = String(node.params.mode ?? 'unconditional');
      const hasTpl = graph.edges.some((e) => e.toNode === node.id && e.toPort === 'structure');
      const hasYaml = String(node.params.customSpecYaml ?? '').trim().length > 0;
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
      } else if (hasTpl && mode === 'unconditional' && !hasYaml) {
        issues.push({
          severity: 'warning',
          code: 'rfd4-unused-template',
          nodeId: node.id,
          message: `${node.label ?? 'RFD4-Proteina'} has a wired input structure but is set to Unconditional mode (the structure will be ignored). Pick a task in RFD4 Parameters (Motif, Binder, or Partial Diffusion).`,
          fix: {
            label: 'Switch to Partial Diffusion',
            nodeId: node.id,
            paramsPatch: { mode: 'partial' },
          },
        });
      } else if (hasTpl && mode === 'preset') {
        issues.push({
          severity: 'warning',
          code: 'rfd4-preset-ignores-template',
          nodeId: node.id,
          message: `${node.label ?? 'RFD4-Proteina'} is set to built-in Preset mode ("${String(node.params.presetId ?? 'custom_cath_monomer')}"), which loads a remote example archive and ignores the wired input structure. Switch to Motif / Binder / Partial mode to condition on your wired template.`,
          fix: {
            label: 'Switch to Motif / Context Mode',
            nodeId: node.id,
            paramsPatch: { mode: 'motif' },
          },
        });
      }
    }

    if (node.adapterId === 'web-protein-mpnn') {
      const edge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
      const src = edge ? graph.nodes.find((n) => n.id === edge.fromNode) : null;
      if (src) {
        const up = inferNodeOutputProfile(src, graph, cache, visiting, outputsByNode);
        const family = String(node.params.family ?? 'solublempnn');
        if (family !== 'ca' && (up.isCaOnly || (up.missingBackboneCount ?? 0) > 0)) {
          const mpnnName = node.label ?? 'ProteinMPNN';
          const msg = up.fromGenie3
            ? `Genie 3 outputs Cα-only coordinates (no N/C/O backbone atoms), which ${mpnnName} requires. Switch to Cα-ProteinMPNN for faithful sequence design.`
            : up.isCaOnly
              ? `Upstream structure is Cα-only (missing N/C/O backbone atoms), which ${mpnnName} requires. Switch to Cα-ProteinMPNN.`
              : `Upstream structure has ${up.missingBackboneCount} residue(s) missing N/C backbone atoms — ${mpnnName} requires complete backbone atoms. Switch to Cα-ProteinMPNN.`;
          issues.push({
            severity: 'warning',
            code: 'mpnn-ca-only-backbone',
            nodeId: node.id,
            message: msg,
            fix: {
              label: 'Switch to Cα-ProteinMPNN',
              nodeId: node.id,
              paramsPatch: { family: 'ca', modelName: 'v_48_020' },
              nodeLabel: 'Cα-ProteinMPNN',
            },
          });
        }
        if (up.entities.has('ligand') || up.entities.has('dna')) {
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

    if (node.adapterId === 'localfold-af2') {
      const edge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'sequences');
      const src = edge ? graph.nodes.find((n) => n.id === edge.fromNode) : null;
      if (!src) continue;
      const up = inferNodeOutputProfile(src, graph, cache, visiting, outputsByNode);
      const modelId = String(node.params.model ?? 'monomer');
      const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
      const nodeName = node.label ?? 'LocalFold';
      const modelDisplay = nodeName.toLowerCase().includes(meta.shortLabel.toLowerCase())
        ? nodeName
        : `${nodeName} (${meta.shortLabel})`;
      const hasExplicitAf3NonProt =
        meta.family === 'af3' &&
        (String(node.params.ligandCcdCodes ?? '').trim().length > 0 ||
          String(node.params.dnaSequences ?? '').trim().length > 0 ||
          String(node.params.rnaSequences ?? '').trim().length > 0);
      const nonProtEntities = new Set<BioEntityKind>([
        ...[...up.entities].filter((e) => e !== 'protein'),
        ...(up.strippedNonProtein ?? []),
      ]);

      if (nonProtEntities.size > 0 && !hasExplicitAf3NonProt) {
        const nonProt = [...nonProtEntities].join('/');
        if (meta.family !== 'af3') {
          issues.push({
            severity: 'warning',
            code: 'af2-non-protein-entity',
            nodeId: node.id,
            message: `${modelDisplay} is protein-only, while upstream involves ${nonProt.toUpperCase()}. Switch to RoseTTAFold3 or Boltz-2 (note: current browser refolding folds the protein chain apo unless ligand/DNA context is configured).`,
            fix: {
              label: 'Switch to RoseTTAFold3',
              nodeId: node.id,
              paramsPatch: { model: 'rosettafold3' },
              nodeLabel: 'RoseTTAFold 3',
            },
          });
        } else {
          issues.push({
            severity: 'warning',
            code: 'af3-apo-refold-note',
            nodeId: node.id,
            message: `${modelDisplay} currently receives protein sequences only — upstream ${nonProt.toUpperCase()} coordinates were not forwarded into structure prediction (apo protein refold).`,
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

      const tokenLimit = meta.family === 'af3' ? 384 : 512;
      if (up.estimatedTokens > tokenLimit) {
        issues.push({
          severity: 'warning',
          code: 'webgpu-token-ceiling',
          nodeId: node.id,
          message: `Estimated length (~${up.estimatedTokens} tokens) exceeds the recommended browser WebGPU ceiling (${tokenLimit}) for ${meta.shortLabel} and may hit GPU memory limits.`,
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
    const hardLimit = meta.family === 'af3' ? 512 : 768;

    // Resolve the EXACT same candidate subset that localfold.worker.ts will execute.
    const work = selectFoldWorkItems(seqSet, node.params);
    selectedCandidateIds = work.selected.map((c) => c.id);
    if (work.excludedCandidates.length > 0) {
      excludedCandidates = work.excludedCandidates;
      warnings.push(
        `LocalFold maxSequences=${work.maxSequences} excludes ${work.excludedCandidates.length} candidate(s): ${work.excludedCandidates.map((c) => c.id).join(', ')}.`,
      );
    }

    for (const cand of work.selected) {
      const len = cand.sequence.replace(/[\s/:,]+/g, '').length;
      if (len < 5) {
        throw new Error(`Refusing to fold "${cand.id}": sequence is too short (${len} aa; minimum 5 aa).`);
      }
      if (len > hardLimit) {
        throw new Error(
          `Refusing to fold "${cand.id}" (${len} aa) with ${meta.shortLabel}: exceeds browser WebGPU memory ceiling (${hardLimit} residues). Crop the target domain or shorten the sequence before running.`,
        );
      }
      if (cand.chainLengths && cand.chainLengths.length > 1 && meta.family === 'af2') {
        warnings.push(
          `Folding ${cand.chainLengths.length}-chain complex (${cand.chainLengths.join(' + ')} aa) with AF2 monomer (using unpaired multi-chain A3M); consider AF2-Multimer v3 or Boltz-2 for accurate interface scoring.`,
        );
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

  if (node.adapterId === 'genie3') {
    const tpl = inputs.structure as StructureArtifact | undefined;
    if (tpl) {
      if (tpl.benchmarkId) {
        const bench = getBenchmarkById(tpl.benchmarkId);
        if (bench && !bench.supportsGenie3) {
          throw new Error(
            `${bench.shortLabel} requires all-atom enzyme/ligand/DNA conditioning, which is supported by RFD4-Proteina rather than Genie 3.`,
          );
        }
      }
      if (tpl.diagnostics?.hasLigands || tpl.diagnostics?.hasNucleicAcids) {
        const extras = [
          ...(tpl.diagnostics.hasLigands ? [`ligands (${tpl.diagnostics.ligandNames.join(', ')})`] : []),
          ...(tpl.diagnostics.hasNucleicAcids ? [`nucleic acid chains (${tpl.diagnostics.nucleicAcidChains.join(', ')})`] : []),
        ];
        throw new Error(
          `Genie 3 is a protein-only Cα backbone diffusion model and cannot condition on ${extras.join(' and ')}. Use RFD4-Proteina for all-atom ligand/DNA conditioning.`,
        );
      }
    }
    const taskMode = String(node.params.taskMode ?? 'unconditional');
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
      if (cov.isCaOnly || cov.missingBackboneCount > 0) {
        warnings.push(
          cov.isCaOnly
            ? `Upstream structure is Cα-only; switch ProteinMPNN to Cα-ProteinMPNN (family: "ca") for faithful sequence design.`
            : `Upstream structure has ${cov.missingBackboneCount} residue(s) missing N/C backbone atoms; consider switching to Cα-ProteinMPNN.`,
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

  return {
    warnings,
    ...(excludedCandidates ? { excludedCandidates } : {}),
    ...(selectedCandidateIds ? { selectedCandidateIds } : {}),
  };
}
