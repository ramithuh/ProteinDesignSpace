// Layer 3 Plugin CPU Runner — pdb-input
// Pure CPU/fetch materializer shared by ExecutionScheduler and instant live 3D preview.

import {
  cropRawAllAtomStructure,
  generateSampleHelixBundlePdb,
  parseStructureText,
} from '$lib/protein-ops/pdb';
import { fetchStructureByPdbId, normalizePdbId } from '$lib/protein-ops/rcsb';
import { fetchStructureByUniProtId, normalizeUniProtId } from '$lib/protein-ops/alphafold-db';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { fnv1a } from '$lib/kernel/hash-utils';
import type {
  CanonicalArtifact,
  NodeProgressEvent,
  SequenceSet,
  StructureArtifact,
  WorkflowNode,
} from '$lib/engine/types';

/**
 * Pure CPU/fetch materializer for a `pdb-input` node.
 * Shared by both ExecutionScheduler and instant live 3D preview (no Run needed).
 */
export async function materializePdbInputNode(
  node: WorkflowNode,
  onProgress?: (ev: NodeProgressEvent) => void,
): Promise<StructureArtifact> {
  const mode = String(node.params.mode ?? 'demo-bundle');
  let rawText: string;
  let filename: string;
  if (mode === 'pdb-id') {
    const normId = normalizePdbId(String(node.params.pdbId ?? ''));
    if (!normId) {
      throw new Error('PDB Input is in PDB ID mode, but no PDB ID was entered.');
    }
    const cachedText = String(node.params.pdbText ?? '').trim();
    const fetchedFor = normalizePdbId(String(node.params.fetchedPdbId ?? ''));
    if (cachedText.length > 0 && fetchedFor === normId) {
      rawText = cachedText;
      const rawFn = String(node.params.filename ?? '').trim();
      filename = rawFn && rawFn !== 'demo_3helix_53aa.pdb' ? rawFn : `${normId}.cif`;
    } else {
      onProgress?.({
        nodeId: node.id,
        status: 'loading',
        phase: 'fetch',
        progress: 0.2,
        detail: `Fetching ${normId} from RCSB PDB...`,
      });
      const fetched = await fetchStructureByPdbId(normId);
      rawText = fetched.text;
      filename = fetched.filename;
    }
  } else if (mode === 'alphafold-db') {
    const normUniProt = normalizeUniProtId(String(node.params.uniprotId ?? ''));
    if (!normUniProt) {
      throw new Error('AlphaFold DB input is missing a UniProt accession (e.g. P69905, P04637).');
    }
    const cachedText = String(node.params.pdbText ?? '').trim();
    const fetchedFor = normalizeUniProtId(String(node.params.fetchedUniProtId ?? ''));
    if (cachedText.length > 0 && fetchedFor === normUniProt) {
      rawText = cachedText;
      const rawFn = String(node.params.filename ?? '').trim();
      filename = rawFn && rawFn !== 'demo_3helix_53aa.pdb' ? rawFn : `AF-${normUniProt}-F1.cif`;
    } else {
      onProgress?.({
        nodeId: node.id,
        status: 'loading',
        phase: 'fetch',
        progress: 0.2,
        detail: `Fetching ${normUniProt} from AlphaFold DB...`,
      });
      const fetched = await fetchStructureByUniProtId(normUniProt);
      rawText = fetched.text;
      filename = fetched.filename;
    }
  } else if (mode === 'custom' && String(node.params.pdbText ?? '').trim().length > 0) {
    rawText = String(node.params.pdbText);
    filename = String(node.params.filename || 'uploaded.pdb');
  } else {
    rawText = generateSampleHelixBundlePdb();
    filename = 'demo_3helix_53aa.pdb';
  }
  const cropSpec = mode === 'demo-bundle' ? '' : String(node.params.cropSegments ?? '').trim();
  const isCropped = cropSpec.length > 0;
  let effectiveRawText = rawText;
  let omittedLigands: string[] = [];
  if (isCropped) {
    effectiveRawText = cropRawAllAtomStructure(rawText, cropSpec).text;
  }
  const parsed = parseStructureText(effectiveRawText);
  if (isCropped) {
    try {
      const fullLigands = parseStructureText(rawText).diagnostics.ligandNames;
      if (fullLigands.length > 0) {
        const keptSet = new Set(parsed.diagnostics.ligandNames.map((l) => l.toUpperCase()));
        omittedLigands = fullLigands.filter((l) => !keptSet.has(l.toUpperCase()));
      }
    } catch {
      omittedLigands = [];
    }
  }
  const diagnostics =
    omittedLigands.length > 0
      ? { ...parsed.diagnostics, omittedLigands }
      : parsed.diagnostics;
  const hasNonProteinContext = parsed.diagnostics.hasLigands || parsed.diagnostics.hasNucleicAcids;
  if (parsed.nResidues < 3 && !hasNonProteinContext) {
    throw new Error('Input PDB/mmCIF must contain at least 3 residues with Cα atoms');
  }
  const geometry = computeGeometryStats(
    parsed.caCoords,
    parsed.chainIds,
    parsed.resNums,
    parsed.insCodes,
  );
  const benchmarkId = String(node.params.benchmarkId ?? '').trim() || undefined;

  const summaryParts: string[] = [];
  if (parsed.nResidues > 0) summaryParts.push(`L=${parsed.nResidues}`);
  if (parsed.diagnostics.ligandNames.length > 0) summaryParts.push(parsed.diagnostics.ligandNames.join('+'));
  if (parsed.diagnostics.nucleicAcidChains.length > 0) {
    summaryParts.push(`DNA(${parsed.diagnostics.nucleicAcidChains.join(',')})`);
  }
  const summaryTag = summaryParts.join(' · ') || 'context';

  return {
    kind: 'StructureArtifact',
    id: `pdb-${fnv1a(`${rawText}::${cropSpec}${benchmarkId ? `::${benchmarkId}` : ''}`)}`,
    label: isCropped
      ? `${filename} [${cropSpec}] (${summaryTag})`
      : `${filename} (${summaryTag})`,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText:
      parsed.pdbText.trim().length > 0
        ? parsed.pdbText
        : parsed.format === 'mmcif'
          ? ''
          : effectiveRawText,
    cifText: parsed.format === 'mmcif' ? effectiveRawText : undefined,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    chainIdMap: parsed.chainIdMap,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics,
    ...(benchmarkId ? { benchmarkId } : {}),
    geometry,
  };
}

/**
 * Derive an optional `SequenceSet` from a materialized `pdb-input` `StructureArtifact`
 * so `pdb-input` can directly feed sequence-consuming blocks (`localfold-af2`, `evodiff`, `foldflow2`).
 * Returns `undefined` for non-protein structures (`length <= 0` or empty sequence, e.g. pure DNA).
 */
export function deriveSequenceSetFromStructure(art: StructureArtifact): SequenceSet | undefined {
  const seq = art.sequence.trim().toUpperCase();
  if (art.length <= 0 || !seq) return undefined;
  const candidateId = art.label.replace(/\s+\([^)]*\)$/, '').trim() || art.id;
  const chainLengths =
    art.chainLengths && art.chainLengths.length > 0 ? [...art.chainLengths] : [seq.length];
  const uniqueChainIds =
    art.chainIds && art.chainIds.length > 0
      ? (() => {
          const out: string[] = [];
          for (const c of art.chainIds) {
            if (out.length === 0 || out[out.length - 1] !== c) out.push(c);
          }
          return out.length === chainLengths.length ? out : undefined;
        })()
      : undefined;
  return {
    kind: 'SequenceSet',
    id: `pdb-seq-${art.id}`,
    parentStructureId: art.id,
    sourceStructureId: art.id,
    sourceModelId: art.sourceModelId || 'pdb-input',
    collectionKind: 'independent-candidates',
    ...(art.lockedChains?.length ? { lockedChains: art.lockedChains } : {}),
    ...(art.chainRoles ? { chainRoles: art.chainRoles } : {}),
    candidates: [
      {
        id: candidateId,
        key: art.id,
        sequence: seq,
        score: null,
        globalScore: null,
        sourceModelId: art.sourceModelId || 'pdb-input',
        sourceStructureId: art.id,
        sampleIndex: 0,
        chainLengths,
        ...(uniqueChainIds ? { chainIds: uniqueChainIds } : {}),
      },
    ],
  };
}

export async function runPdbInputCpu(
  node: WorkflowNode,
  onProgress?: (ev: NodeProgressEvent) => void,
): Promise<Record<string, CanonicalArtifact>> {
  const art = await materializePdbInputNode(node, onProgress);
  const sequences = deriveSequenceSetFromStructure(art);
  return sequences ? { structure: art, sequences } : { structure: art };
}
