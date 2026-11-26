// Layer 3 Plugin CPU Runner — pulchra
// PULCHRA 3.06 all-atom reconstruction over a structure (or each member of a
// structure collection), optionally threading designed sequences first.

import { fnv1a } from '$lib/kernel';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  AtomGranularity,
  CanonicalArtifact,
  NodeProgressEvent,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
  WorkflowNode,
} from '$lib/engine/types';
import {
  loadPulchraTables,
  reconstructPdb,
  type LoadPulchraTablesOptions,
  type PulchraOptions,
  type PulchraResult,
} from './pulchra-core';

export interface PulchraRunnerOptions extends LoadPulchraTablesOptions {
  signal?: AbortSignal;
}

interface PulchraJob {
  parent: StructureArtifact;
  candidate?: SequenceCandidate;
}

export function pulchraOptionsFromParams(node: WorkflowNode): PulchraOptions {
  const maxCaShift = Number(node.params.maxCaShift ?? 0.5);
  return {
    reconstructBackbone: node.params.reconstructBackbone !== false,
    reconstructSidechains: node.params.reconstructSidechains !== false,
    optimizeCa: node.params.optimizeCa === true,
    optimizeHbonds: node.params.optimizeHbonds === true,
    fixClashes: node.params.fixClashes !== false,
    maxCaShift: Number.isFinite(maxCaShift) && maxCaShift >= 0 ? maxCaShift : 0.5,
    detectCisProline: node.params.detectCisProline === true,
    outputHydrogens: node.params.outputHydrogens === true,
  };
}

function structureItems(structure: StructureArtifact): StructureArtifact[] {
  const items = structure.collection?.items;
  return items && items.length > 1 ? items : [structure];
}

/** Pairs each sequence candidate with its parent backbone (lineage first, then position). */
function planJobs(items: StructureArtifact[], sequences: SequenceSet | undefined): PulchraJob[] {
  const candidates = sequences?.candidates ?? [];
  if (!candidates.length) return items.map((parent) => ({ parent }));
  const byId = new Map(items.map((it) => [it.id, it]));
  const byDesignKey = new Map(items.filter((it) => it.designKey).map((it) => [it.designKey as string, it]));
  return candidates.map((candidate, idx) => {
    const parent =
      (candidate.sourceStructureId ? byId.get(candidate.sourceStructureId) : undefined) ??
      (candidate.designKey ? byDesignKey.get(candidate.designKey) : undefined) ??
      (items.length === 1 ? items[0] : undefined) ??
      (sequences?.parentStructureId ? byId.get(sequences.parentStructureId) : undefined) ??
      (items.length === candidates.length ? items[idx] : undefined);
    if (!parent) {
      throw new Error(
        `PULCHRA cannot match sequence ${candidate.id} to one of ${items.length} upstream structures (no lineage link).`,
      );
    }
    return { parent, candidate };
  });
}

function inputPdbText(parent: StructureArtifact): string {
  if (parent.pdbText?.trim()) return parent.pdbText;
  if (parent.cifText?.trim()) return parseStructureText(parent.cifText).pdbText;
  throw new Error(`PULCHRA: upstream structure ${parent.id} has no coordinates text`);
}

function granularity(result: PulchraResult): AtomGranularity {
  const atoms = result.chains.flatMap((c) => c.residues.flatMap((r) => r.atoms.map((a) => a.name)));
  if (atoms.some((n) => !['N', 'CA', 'C', 'O', 'OXT', 'H'].includes(n))) return 'all-atom';
  return atoms.includes('N') && atoms.includes('C') ? 'backbone4' : 'ca-only';
}

function buildArtifact(job: PulchraJob, result: PulchraResult, options: PulchraOptions): StructureArtifact {
  const { parent, candidate } = job;
  const parsed = parseStructureText(result.pdbText);
  const geometry = computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes);
  const sameLength = parsed.nResidues === parent.length;
  const id = `pulchra-${fnv1a(`${parent.id}|${candidate?.id ?? ''}|${candidate?.sequence ?? ''}|${JSON.stringify(options)}`)}`;
  const label = candidate ? `${parent.label} · ${candidate.id} (PULCHRA)` : `${parent.label} (PULCHRA)`;
  return {
    kind: 'StructureArtifact',
    id,
    label,
    sourceModelId: 'pulchra',
    parentStructureId: parent.id,
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: result.pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    chainIdMap: parent.chainIdMap ?? parsed.chainIdMap,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    atomGranularity: granularity(result),
    bfactorSemantics: parent.bfactorSemantics ?? 'none',
    ...(parent.molecularContext?.chainTypes ? { molecularContext: { chainTypes: { ...parent.molecularContext.chainTypes } } } : {}),
    ...(parent.designKey !== undefined ? { designKey: parent.designKey } : {}),
    ...(parent.designIndex !== undefined ? { designIndex: parent.designIndex } : {}),
    ...(parent.seed !== undefined ? { seed: parent.seed } : {}),
    ...(parent.benchmarkId ? { benchmarkId: parent.benchmarkId } : {}),
    ...(parent.lockedChains ? { lockedChains: [...parent.lockedChains] } : {}),
    ...(parent.chainRoles ? { chainRoles: { ...parent.chainRoles } } : {}),
    // Residue-index annotations stay valid only when no residue was dropped.
    ...(sameLength && parent.fixedResidueIndices ? { fixedResidueIndices: [...parent.fixedResidueIndices] } : {}),
    ...(sameLength && parent.motifIndices ? { motifIndices: [...parent.motifIndices] } : {}),
    ...(sameLength && parent.nativeMotifCoords ? { nativeMotifCoords: parent.nativeMotifCoords } : {}),
    ...(sameLength && parent.nativeMotifAtomCoords ? { nativeMotifAtomCoords: parent.nativeMotifAtomCoords } : {}),
    ...(result.warnings.length ? { provenance: { warnings: [...result.warnings] } } : { provenance: {} }),
    geometry,
  };
}

function wrapOutputs(outputs: StructureArtifact[], upstream: StructureArtifact): StructureArtifact {
  if (outputs.length === 1) return outputs[0];
  const primary = outputs[0];
  const designKeys = outputs.map((it) => it.designKey);
  const keys =
    designKeys.every(Boolean) && new Set(designKeys).size === outputs.length
      ? (designKeys as string[])
      : outputs.map((it) => it.id);
  const batchHash = fnv1a(outputs.map((it) => it.id).join('|'));
  return {
    ...primary,
    id: `pulchra-batch-${outputs.length}-${batchHash}`,
    collection: {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: `pulchra-col-${batchHash}`,
      collectionKind: upstream.collection?.collectionKind ?? 'independent-candidates',
      requestedCount: outputs.length,
      complete: true,
      items: outputs,
      keys,
    },
  };
}

export async function runPulchraCpu(
  node: WorkflowNode,
  inputs: Record<string, CanonicalArtifact>,
  onProgress?: (ev: NodeProgressEvent) => void,
  opts: PulchraRunnerOptions = {},
): Promise<Record<string, CanonicalArtifact>> {
  const structure = inputs.structure as StructureArtifact | undefined;
  if (!structure || structure.kind !== 'StructureArtifact') {
    throw new Error('PULCHRA requires an upstream StructureArtifact on its structure input');
  }
  const seqInput = inputs.sequences as SequenceSet | undefined;
  const sequences = seqInput?.kind === 'SequenceSet' ? seqInput : undefined;
  const options = pulchraOptionsFromParams(node);
  const jobs = planJobs(structureItems(structure), sequences);

  onProgress?.({ nodeId: node.id, status: 'running', phase: 'load', progress: 0.02, detail: 'Loading PULCHRA tables…' });
  const tables = await loadPulchraTables(opts);

  const outputs: StructureArtifact[] = [];
  let atoms = 0;
  for (const [i, job] of jobs.entries()) {
    opts.signal?.throwIfAborted();
    onProgress?.({
      nodeId: node.id,
      status: 'running',
      phase: 'rebuild',
      progress: 0.05 + (0.9 * i) / jobs.length,
      detail: `Rebuilding all-atom model ${i + 1}/${jobs.length}…`,
    });
    const result = reconstructPdb(inputPdbText(job.parent), tables, {
      ...options,
      ...(job.candidate ? { sequence: job.candidate.sequence } : {}),
    });
    outputs.push(buildArtifact(job, result, options));
    atoms += result.atomCount;
    // Yield between models so long batches keep the page responsive.
    if (i + 1 < jobs.length) await new Promise((r) => setTimeout(r, 0));
  }
  onProgress?.({
    nodeId: node.id,
    status: 'running',
    phase: 'rebuild',
    progress: 1,
    detail: `Rebuilt ${outputs.length} model(s), ${atoms} atoms`,
  });
  return { structure: wrapOutputs(outputs, structure) };
}
