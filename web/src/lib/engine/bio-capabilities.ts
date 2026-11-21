// Two-Layer Biological Capability Schema:
// Layer 1: Static graph/wire capability contracts (catches monomer-vs-multimer, ligand/DNA stripping,
//          pure-DNA sequence input, and token-ceiling risks before clicking Run).
// Layer 2: Runtime pre-dispatch artifact validation (called in scheduler.ts before launching GPU workers).

import { LOCALFOLD_MODEL_OPTIONS } from './registry';
import { getBenchmarkById } from './benchmarks';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { checkGenie3ConditionCompatibility } from './conditioning';
import type {
  BioCompatibilityIssue,
  BioEntityKind,
  CanonicalArtifact,
  ConditionSpecArtifact,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
  WorkflowNode,
} from './types';

export interface NodeBioProfile {
  entities: Set<BioEntityKind>;
  stoichiometry: 'monomer' | 'multimer';
  chainCount: number;
  estimatedTokens: number;
  looksLikeDnaSequence?: boolean;
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
    return inferNodeOutputProfile(src, graph, cache, visiting);
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
    if ((mode === 'custom' || mode === 'pdb-id') && pdbText.length > 0) {
      try {
        const parsed = parseStructureText(pdbText);
        const entities = new Set<BioEntityKind>(['protein']);
        if (bench) {
          for (const e of bench.entities) entities.add(e);
        } else {
          if (parsed.diagnostics.hasLigands) entities.add('ligand');
          if (parsed.diagnostics.hasNucleicAcids) entities.add('dna');
        }
        const chainCount = Math.max(1, parsed.diagnostics.chainCount);
        profile = {
          entities,
          stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
          chainCount,
          estimatedTokens: parsed.nResidues,
        };
      } catch {
        profile.estimatedTokens = bench?.approxTokens ?? 80;
      }
    } else if (bench) {
      profile = {
        entities: new Set<BioEntityKind>(bench.entities),
        stoichiometry: 'monomer',
        chainCount: 1,
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
    const tpl = upstreamProfileForPort('template');
    const entities = new Set<BioEntityKind>(['protein']);
    let chainCount = 1;
    let estimatedTokens = Number(node.params.length ?? 70);

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
    if (cond) {
      for (const e of cond.entities) entities.add(e);
      chainCount = Math.max(chainCount, cond.chainCount);
      estimatedTokens = Math.max(estimatedTokens, cond.estimatedTokens);
    }
    if (tpl && (mode === 'partial' || mode === 'inverse-folding')) {
      for (const e of tpl.entities) entities.add(e);
      chainCount = Math.max(chainCount, tpl.chainCount);
      estimatedTokens = Math.max(estimatedTokens, tpl.estimatedTokens);
    }
    profile = {
      entities,
      stoichiometry: chainCount > 1 ? 'multimer' : 'monomer',
      chainCount,
      estimatedTokens,
    };
  } else if (node.adapterId === 'genie3') {
    const cond = upstreamProfileForPort('condition');
    const tpl = upstreamProfileForPort('template');
    const tplEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'template');
    const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;
    const tplBenchId =
      tplNode?.adapterId === 'pdb-input' ? String(tplNode.params.benchmarkId ?? '').trim() : '';
    const tplBench = tplBenchId ? getBenchmarkById(tplBenchId) : undefined;
    // Direct PDB -> Genie 3 (no explicit Conditioning block) defaults to 2-chain binder design
    // unless the benchmark target is a single-chain motif scaffold.
    const autoTemplateChains = tpl && !cond ? (tplBench?.category === 'motif' ? 1 : 2) : 1;
    const chainCount = Math.max(1, cond?.chainCount ?? 1, autoTemplateChains);
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
    profile = {
      // ProteinMPNN emits protein sequences only (strips ligands/DNA)
      entities: new Set<BioEntityKind>(['protein']),
      stoichiometry: st?.stoichiometry ?? 'monomer',
      chainCount: st?.chainCount ?? 1,
      estimatedTokens: st?.estimatedTokens ?? 80,
    };
  } else if (node.adapterId === 'localfold-af2') {
    const seq = upstreamProfileForPort('sequences');
    profile = {
      entities: new Set<BioEntityKind>(seq ? [...seq.entities] : ['protein']),
      stoichiometry: seq?.stoichiometry ?? 'monomer',
      chainCount: seq?.chainCount ?? 1,
      estimatedTokens: seq?.estimatedTokens ?? 80,
    };
  }

  visiting.delete(node.id);
  cache.set(node.id, profile);
  return profile;
}

/**
 * Layer 1: Static biological capability validation across the workflow graph.
 * Returns actionable warnings/errors with 1-click fixes for biological modality mismatches.
 */
export function validateBioCapabilities(graph: WorkflowGraph): BioCompatibilityIssue[] {
  const issues: BioCompatibilityIssue[] = [];
  const cache = new Map<string, NodeBioProfile>();
  const visiting = new Set<string>();

  for (const node of graph.nodes) {
    const ownProfile = inferNodeOutputProfile(node, graph, cache, visiting);

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
      const tplEdge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'template');
      const tplNode = tplEdge ? graph.nodes.find((n) => n.id === tplEdge.fromNode) : null;

      if (!condNode && tplNode?.adapterId === 'pdb-input') {
        const benchId = String(tplNode.params.benchmarkId ?? '').trim();
        const bench = benchId ? getBenchmarkById(benchId) : undefined;
        if (bench && !bench.supportsGenie3) {
          issues.push({
            severity: 'warning',
            code: 'genie3-unsupported-benchmark',
            nodeId: node.id,
            message: `${bench.shortLabel} requires all-atom enzyme/ligand/DNA conditioning, which is supported by RFD4-Proteina rather than ${node.label ?? 'Genie 3'}.`,
          });
        }
      }

      if (condNode && condNode.adapterId === 'rfd4-conditioner') {
        const tp = String(condNode.params.taskPreset ?? 'custom');
        const hbond = String(condNode.params.hbondMode ?? 'none');
        const cath = String(condNode.params.cathCode ?? '').trim();
        const unidx = Boolean(condNode.params.unindexedMotif);
        const upCond = inferNodeOutputProfile(condNode, graph, cache, visiting);
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
      const hasTpl = graph.edges.some((e) => e.toNode === node.id && e.toPort === 'template');
      const hasCond = graph.edges.some((e) => e.toNode === node.id && e.toPort === 'condition');
      const hasYaml = String(node.params.customSpecYaml ?? '').trim().length > 0;
      if (hasTpl && !hasCond && mode === 'unconditional' && !hasYaml) {
        issues.push({
          severity: 'warning',
          code: 'rfd4-unused-template',
          nodeId: node.id,
          message: `${node.label ?? 'RFD4-Proteina'} has a wired template structure but is set to Unconditional mode (the template will be ignored). Switch to Partial Diffusion or wire a Conditioning block.`,
          fix: {
            label: 'Switch to Partial Diffusion',
            nodeId: node.id,
            paramsPatch: { mode: 'partial' },
          },
        });
      }
    }

    if (node.adapterId === 'web-protein-mpnn') {
      const edge = graph.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
      const src = edge ? graph.nodes.find((n) => n.id === edge.fromNode) : null;
      if (src) {
        const up = inferNodeOutputProfile(src, graph, cache, visiting);
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
      const up = inferNodeOutputProfile(src, graph, cache, visiting);
      const modelId = String(node.params.model ?? 'monomer');
      const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
      const nodeName = node.label ?? 'LocalFold';

      if (up.entities.has('ligand') || up.entities.has('dna')) {
        const nonProt = [...up.entities].filter((e) => e !== 'protein').join('/');
        if (meta.family !== 'af3') {
          issues.push({
            severity: 'warning',
            code: 'af2-non-protein-entity',
            nodeId: node.id,
            message: `${nodeName} (${meta.shortLabel}) is protein-only, while upstream emits ${nonProt.toUpperCase()}. Switch to RoseTTAFold3 or Boltz-2 (note: current browser refolding folds the protein chain apo).`,
            fix: {
              label: 'Switch to RoseTTAFold3',
              nodeId: node.id,
              paramsPatch: { model: 'rosettafold3' },
              nodeLabel: 'LocalFold (RF3)',
            },
          });
        } else {
          issues.push({
            severity: 'warning',
            code: 'af3-apo-refold-note',
            nodeId: node.id,
            message: `${nodeName} (${meta.shortLabel}) currently receives protein sequences only — upstream ${nonProt.toUpperCase()} coordinates are not yet forwarded into structure prediction (apo protein refold).`,
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
            nodeLabel: 'LocalFold (AF2-Multimer 1)',
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
): { warnings: string[] } {
  const warnings: string[] = [];

  if (node.adapterId === 'localfold-af2') {
    const seqSet = inputs.sequences as SequenceSet | undefined;
    if (!seqSet?.candidates?.length) {
      throw new Error('LocalFold requires an upstream SequenceSet with at least one sequence.');
    }
    const modelId = String(node.params.model ?? 'monomer');
    const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
    const hardLimit = meta.family === 'af3' ? 512 : 768;

    for (const cand of seqSet.candidates) {
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
  }

  if (node.adapterId === 'web-protein-mpnn') {
    const struct = inputs.structure as StructureArtifact | undefined;
    if (!struct?.pdbText) {
      throw new Error('ProteinMPNN requires an upstream 3D structure.');
    }
    if (struct.diagnostics?.hasLigands || struct.diagnostics?.hasNucleicAcids) {
      const extras = [
        ...(struct.diagnostics.hasLigands ? [`ligands (${struct.diagnostics.ligandNames.join(', ')})`] : []),
        ...(struct.diagnostics.hasNucleicAcids ? [`nucleic acid chains (${struct.diagnostics.nucleicAcidChains.join(', ')})`] : []),
      ];
      warnings.push(`ProteinMPNN ignores non-protein ${extras.join(' and ')} and designs protein residues only.`);
    }
  }

  return { warnings };
}
