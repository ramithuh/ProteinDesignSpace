// Two-Layer Biological Capability Schema:
// Layer 1: Static graph/wire capability contracts (catches monomer-vs-multimer, ligand/DNA stripping,
//          pure-DNA sequence input, and token-ceiling risks before clicking Run).
// Layer 2: Runtime pre-dispatch artifact validation (called in scheduler.ts before launching GPU workers).

import { LOCALFOLD_MODEL_OPTIONS } from './registry';
import { getBenchmarkById } from './benchmarks';
import { cropRawAllAtomStructure, parseStructureText } from '$lib/protein-ops/pdb';
import { checkGenie3ConditionCompatibility } from './conditioning';
import type {
  BioCompatibilityIssue,
  BioEntityKind,
  CanonicalArtifact,
  CollectionKind,
  ConditionSpecArtifact,
  ExcludedCandidateRecord,
  FoldedCandidate,
  RepresentativeSelection,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
  WorkflowNode,
} from './types';

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

function inferConditionProfile(node: WorkflowNode): NodeBioProfile {
  const preset = String(node.params.taskPreset ?? 'custom');
  const contig = String(node.params.contig ?? '');
  const entities = new Set<BioEntityKind>(['protein']);
  let chainCount = 1;
  let estimatedTokens = 80;

  if (MULTICHAIN_RFD4_PRESETS.has(preset)) {
    chainCount = 2;
    estimatedTokens = 208;
  } else if (LIGAND_RFD4_PRESETS.has(preset)) {
    entities.add('ligand');
    estimatedTokens = 143;
  } else if (DNA_RFD4_PRESETS.has(preset)) {
    entities.add('dna');
    chainCount = 3;
    estimatedTokens = 128;
  } else if (contig.includes('/0')) {
    chainCount = contig.split('/0').filter((s) => s.trim().length > 0).length;
    estimatedTokens = 160;
  }

  return {
    entities,
    stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
    chainCount,
    estimatedTokens,
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
      profile = {
        entities,
        stoichiometry: totalChains > 1 ? 'multimer' : 'monomer',
        chainCount: totalChains,
        estimatedTokens: liveStruct.length > 0 ? liveStruct.length : (bench?.approxTokens ?? 80),
      };
    } else if ((mode === 'custom' || mode === 'pdb-id') && pdbText.length > 0) {
      try {
        const cropSpec = String(node.params.cropSegments ?? '').trim();
        const rawText = cropSpec ? cropRawAllAtomStructure(pdbText, cropSpec).text : pdbText;
        const parsed = parseStructureText(rawText);
        const entities = new Set<BioEntityKind>();
        if (parsed.nResidues > 0 || (!parsed.diagnostics.hasLigands && !parsed.diagnostics.hasNucleicAcids)) {
          entities.add('protein');
        }
        if (parsed.diagnostics.hasLigands) entities.add('ligand');
        if (parsed.diagnostics.hasNucleicAcids) entities.add('dna');
        if (bench) {
          for (const e of bench.entities) entities.add(e);
        }
        const totalChains = Math.max(
          1,
          parsed.diagnostics.chainCount +
            (parsed.diagnostics.hasNucleicAcids ? parsed.diagnostics.nucleicAcidChains.length : 0),
        );
        profile = {
          entities,
          stoichiometry: totalChains > 1 ? 'multimer' : 'monomer',
          chainCount: totalChains,
          estimatedTokens: parsed.nResidues > 0 ? parsed.nResidues : (bench?.approxTokens ?? 80),
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
  } else if (node.adapterId === 'rfd4-conditioner') {
    profile = inferConditionProfile(node);
    const tpl = upstreamProfileForPort('template');
    if (tpl) {
      for (const e of tpl.entities) profile.entities.add(e);
      profile.chainCount = Math.max(profile.chainCount, tpl.chainCount);
      profile.stoichiometry = profile.chainCount > 1 ? 'multimer' : 'monomer';
    }
  } else if (node.adapterId === 'rfd4-proteina') {
    const mode = String(node.params.mode ?? 'unconditional');
    const cond = upstreamProfileForPort('condition');
    const tpl = upstreamProfileForPort('structure') ?? upstreamProfileForPort('template');
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
    if (cond) {
      for (const e of cond.entities) entities.add(e);
      chainCount = Math.max(chainCount, cond.chainCount);
      estimatedTokens = Math.max(estimatedTokens, cond.estimatedTokens);
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
    const cond = upstreamProfileForPort('condition');
    const tpl = upstreamProfileForPort('structure') ?? upstreamProfileForPort('template');
    const tplEdge = graph.edges.find(
      (e) => e.toNode === node.id && (e.toPort === 'structure' || e.toPort === 'template'),
    );
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
      (Boolean(tpl) && !cond && taskMode !== 'motif' && tplBench?.category !== 'motif');
    const chainCount = Math.max(1, cond?.chainCount ?? 1, isBinderTask ? 2 : 1);
    const binderLen = Number(node.params.length ?? 80);
    const estimatedTokens =
      tpl && !cond
        ? tplBench
          ? tplBench.approxTokens
          : binderLen + tpl.estimatedTokens
        : Math.max(binderLen, cond?.estimatedTokens ?? 0, tpl?.estimatedTokens ?? 0);
    profile = {
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
      chainCount,
      estimatedTokens,
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
      const condEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'condition');
      const condNode = condEdge ? graph.nodes.find((n) => n.id === condEdge.fromNode) : null;
      const tplEdge = graph.edges.find(
        (e) => e.toNode === node.id && (e.toPort === 'structure' || e.toPort === 'template'),
      );
      const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;

      if (!condNode && tplNode) {
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

      if (condNode && condNode.adapterId === 'rfd4-conditioner') {
        const tp = String(condNode.params.taskPreset ?? 'custom');
        const hbond = String(condNode.params.hbondMode ?? 'none');
        const cath = String(condNode.params.cathCode ?? '').trim();
        const unidx = Boolean(condNode.params.unindexedMotif);
        const upCond = inferNodeOutputProfile(condNode, graph, cache, visiting, outputsByNode);
        const reasons: string[] = [];
        if (tp !== 'custom') reasons.push(`RFD4 preset "${tp}"`);
        if (upCond.entities.has('ligand') || upCond.entities.has('dna')) reasons.push('ligand/DNA targets');
        if (hbond !== 'none') reasons.push('H-bond atom constraints');
        if (cath) reasons.push(`CATH fold "${cath}"`);
        if (unidx) reasons.push('unindexed motif placement');
        if (reasons.length > 0) {
          issues.push({
            severity: 'warning',
            code: 'genie3-unsupported-condition',
            nodeId: node.id,
            message: `${node.label ?? 'Genie 3'} supports protein motif scaffolding and PPI binder design, not ${reasons.join(', ')}. Use RFD4-Proteina for those constraints.`,
          });
        }
      }
    }

    if (node.adapterId === 'rfd4-proteina') {
      const mode = String(node.params.mode ?? 'unconditional');
      const hasTpl = graph.edges.some(
        (e) => e.toNode === node.id && (e.toPort === 'structure' || e.toPort === 'template'),
      );
      const hasCond = graph.edges.some((e) => e.toNode === node.id && e.toPort === 'condition');
      const hasYaml = String(node.params.customSpecYaml ?? '').trim().length > 0;
      if (hasTpl && !hasCond && mode === 'unconditional' && !hasYaml) {
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
            message: `${nodeName} (${meta.shortLabel}) is protein-only, while upstream involves ${nonProt.toUpperCase()}. Switch to RoseTTAFold3 or Boltz-2 (note: current browser refolding folds the protein chain apo unless ligand/DNA context is configured).`,
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
            message: `${nodeName} (${meta.shortLabel}) currently receives protein sequences only — upstream ${nonProt.toUpperCase()} coordinates were not forwarded into structure prediction (apo protein refold).`,
          });
        }
      } else if (up.stoichiometry === 'multimer' && meta.family === 'af2') {
        issues.push({
          severity: 'warning',
          code: 'multimer-into-af2-monomer',
          nodeId: node.id,
          message: `Upstream produces a ${up.chainCount}-chain complex, but ${nodeName} is set to ${meta.shortLabel} (monomer). Switch to AlphaFold2-Multimer v3 or Boltz-2 for proper complex folding & ipTM.`,
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

  if (node.adapterId === 'genie3') {
    const cond = inputs.condition as ConditionSpecArtifact | undefined;
    if (cond) {
      const compat = checkGenie3ConditionCompatibility(cond);
      if (!compat.supported) {
        throw new Error(compat.reason ?? 'Conditioning spec is not supported by Genie 3.');
      }
    }
    const tpl = (inputs.structure ?? inputs.template) as StructureArtifact | undefined;
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
