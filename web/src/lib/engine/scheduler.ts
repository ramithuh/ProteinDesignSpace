// Headless DAG ExecutionScheduler (zero Svelte/DOM dependencies).
// Handles topological execution, content-addressed memoization, Single-Active-GPU-Worker
// VRAM mutex (dispose -> terminate), CPU nodes, and WebMCP / window.__denovo hooks.

import MpnnWorker from '$lib/workers/mpnn.worker?worker';
import LocalFoldWorker from '$lib/workers/localfold.worker?worker';
import Rfd4Worker from '$lib/workers/rfd4.worker?worker';
import Genie3Worker from '$lib/workers/genie3.worker?worker';
import {
  cropRawAllAtomStructure,
  generateSampleHelixBundlePdb,
  parseStructureText,
} from '$lib/protein-ops/pdb';
import { fetchStructureByPdbId, normalizePdbId } from '$lib/protein-ops/rcsb';
import { applyTransformToPdb, computeGeometryStats, kabschAlign } from '$lib/protein-ops/kabsch';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from './registry';
import { validateGraph } from './graph-utils';
import { ResultMemo } from './result-memo';
import { normalizeWorkflowGraph } from './share';
import { splitMultiChainSequence, validateRuntimeNodeInputs } from './bio-capabilities';
import { buildCanonicalConditionIntent, compileConditionToRfd4 } from './conditioning';
import type {
  ArtifactRef,
  CanonicalArtifact,
  ConditionSpecArtifact,
  DesignabilityMetrics,
  DesignabilityRow,
  FoldedStructure,
  NodeProgressEvent,
  StructureArtifact,
  WorkflowGraph,
  WorkflowNode,
} from './types';

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

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
      filename = String(node.params.filename || `${normId}.cif`);
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
  } else if (mode === 'custom' && String(node.params.pdbText ?? '').trim().length > 0) {
    rawText = String(node.params.pdbText);
    filename = String(node.params.filename || 'uploaded.pdb');
  } else {
    rawText = generateSampleHelixBundlePdb();
    filename = 'demo_3helix_53aa.pdb';
  }
  const cropSpec = String(node.params.cropSegments ?? '').trim();
  const isCropped = cropSpec.length > 0;
  let effectiveRawText = rawText;
  if (isCropped) {
    effectiveRawText = cropRawAllAtomStructure(rawText, cropSpec).text;
  }
  const parsed = parseStructureText(effectiveRawText);
  const hasNonProteinContext = parsed.diagnostics.hasLigands || parsed.diagnostics.hasNucleicAcids;
  if (parsed.nResidues < 3 && !hasNonProteinContext) {
    throw new Error('Input PDB/mmCIF must contain at least 3 residues with Cα atoms');
  }
  const geometry = computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums);
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
    id: `pdb-${fnv1a(`${rawText}::${cropSpec}`)}`,
    label: isCropped
      ? `${filename} [${cropSpec}] (${summaryTag})`
      : `${filename} (${summaryTag})`,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.nResidues > 0 ? parsed.pdbText : parsed.format === 'mmcif' ? '' : effectiveRawText,
    cifText: parsed.format === 'mmcif' ? effectiveRawText : undefined,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    diagnostics: parsed.diagnostics,
    ...(benchmarkId ? { benchmarkId } : {}),
    geometry,
  };
}

/**
 * Pure CPU materializer for a `sequence-input` node.
 * Shared by both ExecutionScheduler and instant live preview (no Run needed).
 */
export function materializeSequenceInputNode(node: WorkflowNode): {
  kind: 'SequenceSet';
  id: string;
  parentStructureId: string;
  sourceModelId: string;
  candidates: Array<{
    id: string;
    sequence: string;
    chainLengths?: number[];
    score: null;
    globalScore: null;
    sourceModelId: string;
  }>;
} {
  const raw = String(node.params.sequence ?? '').trim();
  if (!raw) throw new Error('Sequence Input is empty');
  const candidates: Array<{
    id: string;
    sequence: string;
    chainLengths?: number[];
    score: null;
    globalScore: null;
    sourceModelId: string;
  }> = [];
  if (raw.includes('>')) {
    const blocks = raw.split(/^>/m).filter((b) => b.trim().length > 0);
    const seenIds = new Set<string>();
    blocks.forEach((b, idx) => {
      const lines = b.split('\n');
      const baseHeader = lines[0].trim().split(/\s+/)[0] || `seq-${idx + 1}`;
      const split = splitMultiChainSequence(lines.slice(1).join(''));
      if (split.concatenated.length >= 5) {
        let id = baseHeader;
        let suffix = 2;
        while (seenIds.has(id)) id = `${baseHeader}_${suffix++}`;
        seenIds.add(id);
        candidates.push({
          id,
          sequence: split.concatenated,
          ...(split.chains.length > 1 ? { chainLengths: split.chainLengths } : {}),
          score: null,
          globalScore: null,
          sourceModelId: 'sequence-input',
        });
      }
    });
  } else {
    const split = splitMultiChainSequence(raw);
    if (split.concatenated.length >= 5) {
      candidates.push({
        id: String(node.params.label || 'custom-seq-1'),
        sequence: split.concatenated,
        ...(split.chains.length > 1 ? { chainLengths: split.chainLengths } : {}),
        score: null,
        globalScore: null,
        sourceModelId: 'sequence-input',
      });
    }
  }
  if (candidates.length === 0) throw new Error('No valid amino acid sequences (>= 5 aa) found');
  return {
    kind: 'SequenceSet',
    id: `seqin-${fnv1a(JSON.stringify(candidates))}`,
    parentStructureId: 'none',
    sourceModelId: 'sequence-input',
    candidates,
  };
}

function topoSort(graph: WorkflowGraph): WorkflowNode[] {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const indeg = new Map<string, number>(graph.nodes.map((n) => [n.id, 0]));
  const adj = new Map<string, string[]>(graph.nodes.map((n) => [n.id, []]));

  for (const e of graph.edges) {
    if (!byId.has(e.fromNode) || !byId.has(e.toNode)) continue;
    adj.get(e.fromNode)!.push(e.toNode);
    indeg.set(e.toNode, (indeg.get(e.toNode) ?? 0) + 1);
  }

  const queue: string[] = [];
  for (const [id, d] of indeg) if (d === 0) queue.push(id);

  const ordered: WorkflowNode[] = [];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    ordered.push(byId.get(cur)!);
    for (const nxt of adj.get(cur) ?? []) {
      const nd = (indeg.get(nxt) ?? 1) - 1;
      indeg.set(nxt, nd);
      if (nd === 0) queue.push(nxt);
    }
  }
  if (ordered.length !== graph.nodes.length) {
    throw new Error('WorkflowGraph contains a cycle or invalid edge');
  }
  return ordered;
}

export interface RunResult {
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>;
  refsByNode: Record<string, Record<string, ArtifactRef>>;
  totalElapsedMs: number;
}

interface ActiveGpuSlot {
  adapterId: string;
  worker: Worker;
  abortActive?: (err: Error) => void;
}

export class ExecutionScheduler {
  private activeGpu: ActiveGpuSlot | null = null;
  private readonly memo = new ResultMemo<Record<string, CanonicalArtifact>>();
  private reqCounter = 1;
  private running = false;

  getActiveGpuAdapter(): string | null {
    return this.activeGpu?.adapterId ?? null;
  }

  clearMemo(): void {
    this.memo.clear();
  }

  /** Deterministically dispose & terminate the active GPU worker to reclaim WebGPU VRAM. */
  async releaseActiveGpu(): Promise<void> {
    const slot = this.activeGpu;
    if (!slot) return;
    this.activeGpu = null;

    if (slot.abortActive) {
      const abort = slot.abortActive;
      slot.abortActive = undefined;
      abort(new Error(`GPU worker "${slot.adapterId}" was released while running`));
    }

    await new Promise<void>((resolve) => {
      const id = this.reqCounter++;
      const finish = () => {
        clearTimeout(timer);
        slot.worker.removeEventListener('message', onMsg);
        try {
          slot.worker.terminate();
        } catch {
          /* ignore */
        }
        resolve();
      };
      const timer = setTimeout(finish, 500);
      const onMsg = (ev: MessageEvent) => {
        if (ev.data?.id === id && ev.data?.type === 'disposed') {
          finish();
        }
      };
      slot.worker.addEventListener('message', onMsg);
      try {
        slot.worker.postMessage({ id, type: 'dispose' });
      } catch {
        finish();
      }
    });
  }

  private spawnWorker(adapterId: string): Worker {
    switch (adapterId) {
      case 'web-protein-mpnn':
        return new MpnnWorker();
      case 'localfold-af2':
        return new LocalFoldWorker();
      case 'rfd4-proteina':
        return new Rfd4Worker();
      case 'genie3':
        return new Genie3Worker();
      default:
        throw new Error(`No GPU worker registered for adapter "${adapterId}"`);
    }
  }

  private async runGpuNode(
    node: WorkflowNode,
    inputs: Record<string, CanonicalArtifact>,
    onProgress: (ev: NodeProgressEvent) => void,
  ): Promise<{ outputs: Record<string, CanonicalArtifact>; elapsedMs: number }> {
    // Enforce Single-Active-GPU-Worker Mutex (warm-1-LRU)
    if (this.activeGpu && this.activeGpu.adapterId !== node.adapterId) {
      onProgress({
        nodeId: node.id,
        status: 'loading',
        phase: 'vram',
        progress: 0,
        detail: `Releasing ${this.activeGpu.adapterId} VRAM before starting ${node.label ?? node.adapterId}...`,
      });
      await this.releaseActiveGpu();
    }

    onProgress({
      nodeId: node.id,
      status: 'loading',
      phase: 'init',
      progress: 0.02,
      detail: `Starting ${node.label ?? node.adapterId} worker...`,
    });

    if (!this.activeGpu) {
      const worker = this.spawnWorker(node.adapterId);
      this.activeGpu = { adapterId: node.adapterId, worker };
    }

    const slot = this.activeGpu;
    const worker = slot.worker;
    const id = this.reqCounter++;

    return await new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        worker.removeEventListener('message', onMessage);
        worker.removeEventListener('error', onError);
        if (slot.abortActive === abort) {
          slot.abortActive = undefined;
        }
      };
      const abort = (err: Error) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(err);
      };
      const onMessage = (ev: MessageEvent) => {
        const m = ev.data;
        if (m?.id !== id) return;
        if (m.type === 'progress') {
          onProgress({
            nodeId: node.id,
            status: m.phase === 'weights' || m.phase === 'init' ? 'loading' : 'running',
            phase: m.phase,
            progress: m.progress,
            detail: m.detail,
          });
        } else if (m.type === 'result') {
          if (settled) return;
          settled = true;
          cleanup();
          resolve({ outputs: m.outputs, elapsedMs: m.elapsedMs ?? 0 });
        } else if (m.type === 'error') {
          if (settled) return;
          settled = true;
          cleanup();
          reject(new Error(m.message || `Worker ${node.adapterId} failed`));
        }
      };
      const onError = (err: ErrorEvent) => {
        if (settled) return;
        settled = true;
        cleanup();
        if (this.activeGpu === slot) {
          this.activeGpu = null;
        }
        try {
          worker.terminate();
        } catch {
          /* ignore */
        }
        reject(new Error(err.message || `Worker ${node.adapterId} crashed`));
      };

      slot.abortActive = abort;
      worker.addEventListener('message', onMessage);
      worker.addEventListener('error', onError);
      try {
        // Plain-object clone strips any reactive Proxy (e.g. Svelte 5 $state) before postMessage.
        const plainParams = JSON.parse(JSON.stringify(node.params ?? {}));
        worker.postMessage({ id, type: 'run', params: plainParams, inputs });
      } catch (err) {
        abort(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  private async runCpuNode(
    node: WorkflowNode,
    inputs: Record<string, CanonicalArtifact>,
    onProgress?: (ev: NodeProgressEvent) => void,
  ): Promise<Record<string, CanonicalArtifact>> {
    if (node.adapterId === 'pdb-input') {
      const art = await materializePdbInputNode(node, onProgress);
      return { structure: art };
    }

    if (node.adapterId === 'sequence-input') {
      const seqSet = materializeSequenceInputNode(node);
      return { sequences: seqSet };
    }

    if (node.adapterId === 'rfd4-conditioner') {
      const p = node.params ?? {};
      const template = inputs.template as StructureArtifact | undefined;
      const taskPreset = String(p.taskPreset ?? 'custom');
      const rawOverride = String(p.rawJsonOverride ?? '').trim();

      if (rawOverride) {
        const parsed = JSON.parse(rawOverride) as Record<string, unknown>;
        const art: ConditionSpecArtifact = {
          kind: 'ConditionSpec',
          id: `cond-raw-${fnv1a(JSON.stringify({ rawOverride, tpl: template?.id }))}`,
          label: 'Custom JSON Spec',
          contig: String(parsed.contig ?? p.contig ?? '70-70'),
          conditions: (parsed.conditions as Record<string, unknown>) ?? {},
          inputCif: template?.cifText ?? template?.pdbText,
          specMode: true,
          summary: `raw spec (${String(parsed.contig ?? 'custom')})`,
        };
        return { condition: art };
      }

      if (taskPreset !== 'custom') {
        const art: ConditionSpecArtifact = {
          kind: 'ConditionSpec',
          id: `cond-preset-${taskPreset}`,
          label: `RFD4 Preset: ${taskPreset}`,
          contig: String(p.contig ?? ''),
          conditions: {},
          cifUrl: `preset:${taskPreset}`,
          summary: `preset:${taskPreset}`,
        };
        return { condition: art };
      }

      const intent = buildCanonicalConditionIntent(p);
      const { contig, conditions, tags } = compileConditionToRfd4(intent);

      const art: ConditionSpecArtifact = {
        kind: 'ConditionSpec',
        id: `cond-${fnv1a(JSON.stringify({ intent, tpl: template?.id }))}`,
        label: tags.join(' · '),
        contig,
        conditions,
        intent,
        inputCif: template?.cifText ?? template?.pdbText,
        specMode: true,
        summary: tags.join(' · '),
      };
      return { condition: art };
    }

    if (node.adapterId === 'designability-scorer') {
      const structure = inputs.structure as StructureArtifact | undefined;
      const folds = inputs.folds as FoldedStructure | undefined;
      if (!structure || !folds) {
        throw new Error('designability-scorer requires both structure and folds inputs');
      }
      const scThresh = Number(node.params.scRmsdThreshold ?? 2.0);
      const plddtThresh = Number(node.params.plddtThreshold ?? 70);
      const motifThresh = Number(node.params.motifRmsdThreshold ?? 1.0);
      const ipaeThresh = Number(node.params.ipaeThreshold ?? 10.0);
      const targetAlignedBinderThresh = Number(
        node.params.targetAlignedBinderRmsdThreshold ?? Math.max(scThresh * 2, 4.0),
      );

      const structLen = Math.floor(structure.caCoords.length / 3);

      // Pre-extract valid native motif correspondence if present on the designed StructureArtifact
      const motifPairs: Array<{ idx: number; coord: [number, number, number] }> = [];
      if (
        Array.isArray(structure.motifIndices) &&
        Array.isArray(structure.nativeMotifCoords) &&
        structure.motifIndices.length > 0 &&
        structure.nativeMotifCoords.length > 0
      ) {
        const limit = Math.min(structure.motifIndices.length, structure.nativeMotifCoords.length);
        for (let k = 0; k < limit; k++) {
          const idx = structure.motifIndices[k];
          const nc = structure.nativeMotifCoords[k];
          if (
            Number.isInteger(idx) &&
            idx >= 0 &&
            idx < structLen &&
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
          genMotifCa[k * 3 + 0] = structure.caCoords[idx * 3 + 0];
          genMotifCa[k * 3 + 1] = structure.caCoords[idx * 3 + 1];
          genMotifCa[k * 3 + 2] = structure.caCoords[idx * 3 + 2];
        }
        const genAlign = kabschAlign(genMotifCa, nativeMotifCa);
        if (Number.isFinite(genAlign.rmsd)) {
          genMotifRmsd = genAlign.rmsd;
        }
      }

      // Preserve cifText through Scorer for holo/DNA structures so 3D overlays & Compare mode retain ligands/ions/DNA
      const hasNonProtein = Boolean(
        structure.diagnostics?.hasLigands || structure.diagnostics?.hasNucleicAcids,
      );
      const designedStructureText =
        structure.cifText && (hasNonProtein || !structure.pdbText)
          ? structure.cifText
          : structure.pdbText;

      const rows: DesignabilityRow[] = folds.folds.map((f) => {
        if (f.caCoords.length !== structure.caCoords.length) {
          const foldLen = Math.floor(f.caCoords.length / 3);
          throw new Error(
            `Length mismatch in designability-scorer: refolded "${f.sequenceId}" has ${foldLen} residues, ` +
              `but designed structure "${structure.label}" has ${structLen} residues.`,
          );
        }
        const align = kabschAlign(f.caCoords, structure.caCoords);
        const alignedFoldPdb = applyTransformToPdb(f.pdbText, align.rotation, align.translation);

        let foldResNums: number[] | undefined;
        let foldChainIds = f.chainIds;
        if (f.pdbText) {
          try {
            const parsedFold = parseStructureText(f.pdbText);
            if (parsedFold.nResidues === structLen) {
              foldResNums = parsedFold.resNums;
              if (!foldChainIds || foldChainIds.length !== structLen) {
                foldChainIds = parsedFold.chainIds;
              }
            }
          } catch {
            /* keep f.chainIds without resNums if pdbText is synthetic */
          }
        }
        const geom = computeGeometryStats(f.caCoords, foldChainIds, foldResNums);

        // Multi-chain binder & interface evaluation (#8)
        const chainRanges = resolveScorerChainRanges(
          structLen,
          structure.chainLengths ?? f.chainLengths,
          structure.chainIds ?? foldChainIds,
        );
        let ipae: number | undefined;
        let binderPlddt: number | undefined;
        let targetPlddt: number | undefined;
        let binderScRmsd: number | undefined;
        let targetAlignedBinderRmsd: number | undefined;

        if (chainRanges.length > 1) {
          const split = splitBinderAndTargetIndices(
            chainRanges,
            structure.chainRoles,
            structure.lockedChains,
          );
          const { binderIndices, targetIndices, explicitRoleSplit } = split;

          if (f.pae && f.pae.length === structLen * structLen) {
            ipae = computeInterfacePae(
              f.pae,
              structLen,
              chainRanges,
              explicitRoleSplit ? binderIndices : undefined,
              explicitRoleSplit ? targetIndices : undefined,
            );
          }

          if (f.plddt.length === structLen) {
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
            const structBinderCa = gatherCoordsByIndices(structure.caCoords, binderIndices);
            const bAlign = kabschAlign(foldBinderCa, structBinderCa);
            if (Number.isFinite(bAlign.rmsd)) {
              binderScRmsd = bAlign.rmsd;
            }
          }

          if (targetIndices.length >= 3 && binderIndices.length >= 1) {
            const foldTargetCa = gatherCoordsByIndices(f.caCoords, targetIndices);
            const structTargetCa = gatherCoordsByIndices(structure.caCoords, targetIndices);
            const tAlign = kabschAlign(foldTargetCa, structTargetCa);
            if (Number.isFinite(tAlign.rmsd)) {
              const { rotation: R, translation: t } = tAlign;
              let sqSum = 0;
              for (let k = 0; k < binderIndices.length; k++) {
                const off = binderIndices[k] * 3;
                const x = f.caCoords[off + 0];
                const y = f.caCoords[off + 1];
                const z = f.caCoords[off + 2];
                const dx = R[0] * x + R[1] * y + R[2] * z + t[0] - structure.caCoords[off + 0];
                const dy = R[3] * x + R[4] * y + R[5] * z + t[1] - structure.caCoords[off + 1];
                const dz = R[6] * x + R[7] * y + R[8] * z + t[2] - structure.caCoords[off + 2];
                sqSum += dx * dx + dy * dy + dz * dz;
              }
              const tabRmsd = Math.sqrt(sqSum / binderIndices.length);
              if (Number.isFinite(tabRmsd)) {
                targetAlignedBinderRmsd = tabRmsd;
              }
            }
          }
        }

        // Motif scaffolding RMSD evaluation (#10)
        let refoldMotifRmsd: number | undefined;
        if (nativeMotifCa && motifPairs.length >= 3) {
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

        let designable =
          Number.isFinite(align.rmsd) && align.rmsd < scThresh && f.meanPlddt >= plddtThresh;
        if (binderScRmsd !== undefined) {
          designable = designable && binderScRmsd < scThresh;
        }
        if (binderPlddt !== undefined) {
          designable = designable && binderPlddt >= plddtThresh;
        }
        if (targetAlignedBinderRmsd !== undefined) {
          designable = designable && targetAlignedBinderRmsd < targetAlignedBinderThresh;
        }
        if (ipae !== undefined) {
          designable = designable && ipae < ipaeThresh;
        }
        if (refoldMotifRmsd !== undefined) {
          designable = designable && refoldMotifRmsd < motifThresh;
        }

        return {
          structureId: structure.id,
          sequenceId: f.sequenceId,
          sequence: f.sequence,
          scRmsd: align.rmsd,
          scTm: align.tmScore,
          meanPlddt: f.meanPlddt,
          ptm: f.ptm,
          iptm: f.iptm ?? null,
          ...(ipae !== undefined ? { ipae } : {}),
          ...(binderPlddt !== undefined ? { binderPlddt } : {}),
          ...(targetPlddt !== undefined ? { targetPlddt } : {}),
          ...(binderScRmsd !== undefined ? { binderScRmsd } : {}),
          ...(targetAlignedBinderRmsd !== undefined ? { targetAlignedBinderRmsd } : {}),
          ...(genMotifRmsd !== undefined ? { genMotifRmsd } : {}),
          ...(refoldMotifRmsd !== undefined ? { refoldMotifRmsd } : {}),
          designable,
          alignedFoldPdb,
          designedPdb: designedStructureText,
          pae: f.pae,
          plddt: f.plddt,
          geometry: geom,
          foldElapsedMs: f.elapsedMs,
        };
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
      const metrics: DesignabilityMetrics = {
        kind: 'DesignabilityMetrics',
        id: `metrics-${structure.id}-${folds.id}`,
        rows,
        best: rows[0] ?? null,
        passRate: rows.length > 0 ? passed / rows.length : 0,
      };
      return { metrics };
    }

    throw new Error(`Unknown CPU adapter "${node.adapterId}"`);
  }

  async run(
    graph: WorkflowGraph,
    onProgress: (ev: NodeProgressEvent) => void,
    opts: {
      bypassCache?: boolean;
      onNodeOutput?: (nodeId: string, outputs: Record<string, CanonicalArtifact>) => void;
    } = {},
  ): Promise<RunResult> {
    if (this.running) throw new Error('A workflow run is already in progress');
    this.running = true;
    const tStart = performance.now();
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {};
    const refsByNode: Record<string, Record<string, ArtifactRef>> = {};

    try {
      // Apply the same guards to UI, console, and WebMCP callers before any work starts.
      graph = normalizeWorkflowGraph(graph, graph?.id, graph?.name, false);
      const errors = validateGraph(graph);
      if (errors.length) throw new Error(errors.join('\n'));
      const order = topoSort(graph);

      for (const node of order) {
        const manifest = ADAPTER_REGISTRY[node.adapterId];
        if (!manifest) throw new Error(`Unknown adapter "${node.adapterId}"`);

        // Gather upstream inputs
        const inputs: Record<string, CanonicalArtifact> = {};
        const inputIds: Record<string, string> = {};
        for (const edge of graph.edges) {
          if (edge.toNode !== node.id) continue;
          const up = outputsByNode[edge.fromNode]?.[edge.fromPort];
          if (up) {
            inputs[edge.toPort] = up;
            inputIds[edge.toPort] = up.id;
          }
        }

        // Content-addressed memoization key
        const cacheKey = fnv1a(
          JSON.stringify({ adapter: node.adapterId, params: node.params, inputIds }),
        );

        let nodeOutputs: Record<string, CanonicalArtifact>;
        let elapsedMs = 0;

        const cached = opts.bypassCache ? undefined : this.memo.get(cacheKey);
        if (cached) {
          nodeOutputs = cached;
          onProgress({
            nodeId: node.id,
            status: 'cached',
            progress: 1,
            detail: 'Memoized hit (skipped)',
            elapsedMs: 0,
          });
        } else {
          const nStart = performance.now();
          onProgress({ nodeId: node.id, status: 'running', progress: 0, detail: 'Starting...' });
          try {
            if (manifest.runtime === 'cpu-ts') {
              nodeOutputs = await this.runCpuNode(node, inputs, onProgress);
              elapsedMs = performance.now() - nStart;
            } else {
              // Layer 2 biological capability guard before launching GPU worker / downloading weights
              const guard = validateRuntimeNodeInputs(node, inputs);
              if (guard.warnings.length > 0) {
                console.warn(`[denovo.run] ${node.label ?? node.adapterId}:`, guard.warnings.join(' | '));
              }
              const res = await this.runGpuNode(node, inputs, onProgress);
              nodeOutputs = res.outputs;
              elapsedMs = res.elapsedMs || performance.now() - nStart;
            }
            this.memo.set(cacheKey, nodeOutputs);
            onProgress({
              nodeId: node.id,
              status: 'done',
              progress: 1,
              detail: `Done in ${(elapsedMs / 1000).toFixed(1)}s`,
              elapsedMs,
            });
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            onProgress({ nodeId: node.id, status: 'error', error: message, detail: message });
            throw err;
          }
        }

        outputsByNode[node.id] = nodeOutputs;
        opts.onNodeOutput?.(node.id, nodeOutputs);
        refsByNode[node.id] = {};
        for (const [port, art] of Object.entries(nodeOutputs)) {
          refsByNode[node.id][port] = summarizeArtifact(art);
        }
      }

      return {
        outputsByNode,
        refsByNode,
        totalElapsedMs: performance.now() - tStart,
      };
    } finally {
      this.running = false;
    }
  }
}

function summarizeArtifact(art: CanonicalArtifact): ArtifactRef {
  if (art.kind === 'StructureArtifact') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        label: art.label,
        length: art.length,
        sourceModelId: art.sourceModelId,
        rg: Number(art.geometry.rg.toFixed(2)),
        helixPct: Number(art.geometry.helixPct.toFixed(1)),
      },
    };
  }
  if (art.kind === 'SequenceSet') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        count: art.candidates.length,
        bestScore: art.candidates[0]?.score ?? null,
        topSequence: art.candidates[0]?.sequence ?? '',
      },
    };
  }
  if (art.kind === 'FoldedStructure') {
    const iptmVals = art.folds
      .map((f) => f.iptm)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        count: art.folds.length,
        bestPlddt: Number(Math.max(...art.folds.map((f) => f.meanPlddt), 0).toFixed(1)),
        ...(iptmVals.length > 0 ? { bestIptm: Number(Math.max(...iptmVals).toFixed(3)) } : {}),
      },
    };
  }
  if (art.kind === 'ConditionSpec') {
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        contig: art.contig,
        summary: art.summary,
      },
    };
  }
  const best = art.best;
  return {
    id: art.id,
    kind: art.kind,
    summary: {
      passRate: art.passRate,
      bestScRmsd: best && Number.isFinite(best.scRmsd) ? Number(best.scRmsd.toFixed(3)) : null,
      bestScTm: best && Number.isFinite(best.scTm) ? Number(best.scTm.toFixed(3)) : null,
      bestPlddt: best && Number.isFinite(best.meanPlddt) ? Number(best.meanPlddt.toFixed(1)) : null,
      ...(best && typeof best.iptm === 'number' && Number.isFinite(best.iptm)
        ? { bestIptm: Number(best.iptm.toFixed(3)) }
        : {}),
      ...(best && typeof best.ipae === 'number' && Number.isFinite(best.ipae)
        ? { bestIpae: Number(best.ipae.toFixed(2)) }
        : {}),
      ...(best && typeof best.binderScRmsd === 'number' && Number.isFinite(best.binderScRmsd)
        ? { bestBinderScRmsd: Number(best.binderScRmsd.toFixed(3)) }
        : {}),
      ...(best &&
      typeof best.targetAlignedBinderRmsd === 'number' &&
      Number.isFinite(best.targetAlignedBinderRmsd)
        ? { bestTargetAlignedBinderRmsd: Number(best.targetAlignedBinderRmsd.toFixed(3)) }
        : {}),
      ...(best && typeof best.genMotifRmsd === 'number' && Number.isFinite(best.genMotifRmsd)
        ? { bestGenMotifRmsd: Number(best.genMotifRmsd.toFixed(3)) }
        : {}),
      ...(best && typeof best.refoldMotifRmsd === 'number' && Number.isFinite(best.refoldMotifRmsd)
        ? { bestRefoldMotifRmsd: Number(best.refoldMotifRmsd.toFixed(3)) }
        : {}),
    },
  };
}

interface ScorerChainRange {
  chainId: string;
  start: number;
  length: number;
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
  L: number,
  ranges: readonly ScorerChainRange[],
  binderIndices?: readonly number[],
  targetIndices?: readonly number[],
): number | undefined {
  let sum = 0;
  let count = 0;
  if (binderIndices && targetIndices && binderIndices.length > 0 && targetIndices.length > 0) {
    for (const bi of binderIndices) {
      for (const ti of targetIndices) {
        const v1 = pae[bi * L + ti];
        if (Number.isFinite(v1)) {
          sum += v1;
          count++;
        }
        const v2 = pae[ti * L + bi];
        if (Number.isFinite(v2)) {
          sum += v2;
          count++;
        }
      }
    }
  } else {
    const chainOf = new Int32Array(L);
    for (let c = 0; c < ranges.length; c++) {
      const r = ranges[c];
      for (let i = r.start; i < r.start + r.length; i++) chainOf[i] = c;
    }
    for (let i = 0; i < L; i++) {
      const ci = chainOf[i];
      for (let j = 0; j < L; j++) {
        if (ci === chainOf[j]) continue;
        const v = pae[i * L + j];
        if (Number.isFinite(v)) {
          sum += v;
          count++;
        }
      }
    }
  }
  return count > 0 ? sum / count : undefined;
}

/**
 * Expose programmatic engine API on window.__denovo AND register progressive WebMCP
 * tools (document.modelContext / navigator.modelContext) when the browser supports it.
 */
export function attachProgrammaticApi(
  scheduler: ExecutionScheduler,
  onProgress?: (ev: NodeProgressEvent) => void,
): { webMcpRegistered: boolean } {
  if (typeof window === 'undefined') return { webMcpRegistered: false };

  const api = {
    listAdapters: () => Object.values(ADAPTER_REGISTRY),
    listPresets: () => WORKFLOW_PRESETS.map((p) => ({ id: p.id, name: p.name, summary: p.summary })),
    runWorkflow: async (graph: WorkflowGraph) => {
      const res = await scheduler.run(graph, (ev) => onProgress?.(ev));
      return { refs: res.refsByNode, totalElapsedMs: res.totalElapsedMs };
    },
    runPreset: async (presetId: string) => {
      const preset = WORKFLOW_PRESETS.find((p) => p.id === presetId);
      if (!preset) throw new Error(`Unknown preset "${presetId}"`);
      const res = await scheduler.run(structuredClone(preset.graph), (ev) => onProgress?.(ev));
      return { refs: res.refsByNode, totalElapsedMs: res.totalElapsedMs };
    },
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (window as any).__denovo = api;

  // Progressive WebMCP registration (Chrome/Edge Origin Trial)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mc = (document as any).modelContext ?? (navigator as any).modelContext;
  if (mc && typeof mc.registerTool === 'function') {
    try {
      mc.registerTool({
        name: 'denovo_list_presets',
        description: 'List available browser-native WebGPU protein design and refolding workflows on denovo.run.',
        inputSchema: { type: 'object', properties: {} },
        execute: async () => api.listPresets(),
      });
      mc.registerTool({
        name: 'denovo_run_preset',
        description: 'Execute a preset WebGPU protein design / refolding workflow locally in the browser and return scRMSD, scTM, and pLDDT metrics.',
        inputSchema: {
          type: 'object',
          properties: {
            presetId: {
              type: 'string',
              enum: WORKFLOW_PRESETS.map((p) => p.id),
            },
          },
          required: ['presetId'],
        },
        execute: async (args: { presetId: string }) => api.runPreset(args.presetId),
      });
      return { webMcpRegistered: true };
    } catch {
      return { webMcpRegistered: false };
    }
  }
  return { webMcpRegistered: false };
}
