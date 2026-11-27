// Layer 3 Plugin helpers — cg2all
// Job planning (structure collections × optional SequenceSet candidates) and StructureArtifact assembly,
// shared by the worker and unit tests. Mirrors the PULCHRA runner so both rebuild tools fan out identically.

import { fnv1a } from '$lib/kernel';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  AtomGranularity,
  ChainMolecularKind,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
  StructureMolecularContext,
} from '$lib/engine/types';

export type Cg2allVariant = 'ca' | 'ca-fix';
export type Cg2allPrecision = 'fp16' | 'int8';
export type Cg2allCpuFallbackReason = 'unavailable' | 'device-lost';

export interface Cg2allOptions {
  variant: Cg2allVariant;
  precision: Cg2allPrecision;
  outputHydrogens: boolean;
  preserveLigands: boolean;
}

export interface Cg2allJob {
  parent: StructureArtifact;
  candidate?: SequenceCandidate;
}

/** Result of the vendored `convert()` (static/vendor/cg2all/shared/index.js) for one job. */
export interface Cg2allConversion {
  pdbText: string;
  atomCount: number;
  warnings: string[];
  prepared: { n: number; e: number };
}

export function cg2allOptionsFromParams(params: Record<string, unknown>): Cg2allOptions {
  const variant = String(params.variant ?? 'ca').trim().toLowerCase();
  if (variant !== 'ca' && variant !== 'ca-fix') {
    throw new Error(`Unsupported cg2all variant "${String(params.variant)}". Expected "ca" or "ca-fix".`);
  }
  const precision = String(params.precision ?? 'fp16').trim().toLowerCase() || 'fp16';
  if (precision !== 'fp16' && precision !== 'int8') {
    throw new Error(`Unsupported cg2all precision "${String(params.precision)}". Expected "fp16" or "int8".`);
  }
  return {
    variant,
    precision,
    outputHydrogens: params.outputHydrogens === true,
    preserveLigands: params.preserveLigands !== false,
  };
}

export function structureItems(structure: StructureArtifact): StructureArtifact[] {
  const items = structure.collection?.items;
  return items && items.length > 1 ? items : [structure];
}

/** Pairs each sequence candidate with its parent backbone (lineage first, then position). */
export function planCg2allJobs(items: StructureArtifact[], sequences: SequenceSet | undefined): Cg2allJob[] {
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
        `cg2all cannot match sequence ${candidate.id} to one of ${items.length} upstream structures (no lineage link).`,
      );
    }
    return { parent, candidate };
  });
}

export function inputPdbText(parent: StructureArtifact): string {
  if (parent.pdbText?.trim()) return parent.pdbText;
  if (parent.cifText?.trim()) return parseStructureText(parent.cifText).pdbText;
  throw new Error(`cg2all: upstream structure ${parent.id} has no coordinates text`);
}

function granularity(pdbText: string): AtomGranularity {
  for (const line of pdbText.split('\n')) {
    if (!line.startsWith('ATOM')) continue;
    const name = line.slice(12, 16).trim();
    if (!['N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'HA', 'HA2', 'HA3'].includes(name)) return 'all-atom';
  }
  return 'backbone4';
}

function buildCg2allMolecularContext(
  parent: StructureArtifact,
  parsed: ReturnType<typeof parseStructureText>,
  pdbText: string,
  options: Cg2allOptions,
): StructureMolecularContext | undefined {
  const preservedHetResNames = new Set<string>();
  const hetChainSet = new Set<string>();
  let hasOutputProteinH = false;
  for (const line of pdbText.split('\n')) {
    if (line.startsWith('HETATM') && line.length >= 20) {
      const resName = line.slice(17, 20).trim().toUpperCase();
      if (resName) preservedHetResNames.add(resName);
      if (line.length >= 22) hetChainSet.add((line[21] ?? '').trim() || 'A');
    } else if (options.outputHydrogens && line.startsWith('ATOM  ') && line.length >= 16) {
      const elem = line.slice(76, 78).trim().toUpperCase();
      const atomName = line.slice(12, 16).trim().toUpperCase();
      if (elem === 'H' || elem === 'D' || /^[0-9]?H/.test(atomName)) {
        hasOutputProteinH = true;
      }
    }
  }

  const ligandSet = new Set<string>(options.preserveLigands ? parsed.diagnostics.ligandNames : []);
  if (options.preserveLigands && parent.molecularContext?.ligands?.length) {
    for (const lig of parent.molecularContext.ligands) {
      if (preservedHetResNames.has(lig.toUpperCase())) ligandSet.add(lig);
    }
  }
  const ligands = ligandSet.size > 0 ? [...ligandSet] : undefined;
  const nucleicAcidChains =
    options.preserveLigands && parsed.diagnostics.hasNucleicAcids
      ? [...parsed.diagnostics.nucleicAcidChains]
      : undefined;

  const proteinChainSet = new Set(parsed.chainIds);
  const nucleicChainSet = new Set(nucleicAcidChains ?? []);
  let chainTypes: Record<string, ChainMolecularKind> | undefined;
  if (parent.molecularContext?.chainTypes) {
    const kept: Record<string, ChainMolecularKind> = {};
    for (const [ch, kind] of Object.entries(parent.molecularContext.chainTypes)) {
      if (kind === 'dna' || kind === 'rna') {
        if (options.preserveLigands && nucleicChainSet.has(ch)) kept[ch] = kind;
      } else if (kind === 'ligand') {
        if (options.preserveLigands && hetChainSet.has(ch)) kept[ch] = kind;
      } else if (
        proteinChainSet.has(ch) ||
        Object.entries(parent.chainIdMap ?? {}).some(([proj, orig]) => orig === ch && proteinChainSet.has(proj))
      ) {
        kept[ch] = kind;
      }
    }
    if (Object.keys(kept).length > 0) chainTypes = kept;
  }

  const hasExplicitHydrogens =
    hasOutputProteinH ||
    (Boolean(parent.molecularContext?.hasExplicitHydrogens) &&
      Boolean(ligands?.length || nucleicAcidChains?.length));

  if (!chainTypes && !ligands && !nucleicAcidChains && !hasExplicitHydrogens) return undefined;
  return {
    ...(chainTypes ? { chainTypes } : {}),
    ...(ligands ? { ligands } : {}),
    ...(nucleicAcidChains ? { nucleicAcidChains } : {}),
    ...(hasExplicitHydrogens ? { hasExplicitHydrogens: true } : {}),
  };
}

export function buildCg2allArtifact(
  job: Cg2allJob,
  result: Cg2allConversion,
  options: Cg2allOptions,
  backend: 'webgpu' | 'cpu',
  fallbackReason: Cg2allCpuFallbackReason = 'unavailable',
): StructureArtifact {
  const { parent, candidate } = job;
  const parsed = parseStructureText(result.pdbText);
  const geometry = computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes);
  const sameLength = parsed.nResidues === parent.length;
  const id = `cg2all-${fnv1a(`${parent.id}|${candidate?.id ?? ''}|${candidate?.sequence ?? ''}|${JSON.stringify(options)}`)}`;
  const tag = options.variant === 'ca-fix' ? 'cg2all FIX' : 'cg2all';
  const label = candidate ? `${parent.label} · ${candidate.id} (${tag})` : `${parent.label} (${tag})`;
  const warnings = [...result.warnings];
  if (backend === 'cpu') {
    warnings.push(
      fallbackReason === 'device-lost'
        ? 'cg2all fell back to the CPU JS engine after a WebGPU device loss.'
        : 'cg2all ran on the CPU JS fallback (WebGPU unavailable).',
    );
  }
  const molecularContext = buildCg2allMolecularContext(parent, parsed, result.pdbText, options);
  return {
    kind: 'StructureArtifact',
    id,
    label,
    sourceModelId: 'cg2all',
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
    atomGranularity: granularity(result.pdbText),
    bfactorSemantics: parent.bfactorSemantics ?? 'none',
    ...(molecularContext ? { molecularContext } : {}),
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
    provenance: {
      quantization: options.precision,
      ...(backend === 'cpu' ? { degradedFallback: 'cpu-js' } : {}),
      ...(warnings.length ? { warnings } : {}),
    },
    geometry,
  };
}

export function wrapCg2allOutputs(outputs: StructureArtifact[], upstream: StructureArtifact): StructureArtifact {
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
    id: `cg2all-batch-${outputs.length}-${batchHash}`,
    collection: {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: `cg2all-col-${batchHash}`,
      collectionKind: upstream.collection?.collectionKind ?? 'independent-candidates',
      requestedCount: outputs.length,
      complete: true,
      items: outputs,
      keys,
    },
  };
}
