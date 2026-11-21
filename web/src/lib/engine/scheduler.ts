// Headless DAG ExecutionScheduler (zero Svelte/DOM dependencies).
// Handles topological execution, content-addressed memoization, Single-Active-GPU-Worker
// VRAM mutex (dispose -> terminate), CPU nodes, and WebMCP / window.__denovo hooks.

import MpnnWorker from '$lib/workers/mpnn.worker?worker';
import LocalFoldWorker from '$lib/workers/localfold.worker?worker';
import Rfd4Worker from '$lib/workers/rfd4.worker?worker';
import Genie3Worker from '$lib/workers/genie3.worker?worker';
import { generateSampleHelixBundlePdb, parseStructureText } from '$lib/protein-ops/pdb';
import { applyTransformToPdb, computeGeometryStats, kabschAlign } from '$lib/protein-ops/kabsch';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from './registry';
import { splitMultiChainSequence, validateRuntimeNodeInputs } from './bio-capabilities';
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
  private memo = new Map<string, Record<string, CanonicalArtifact>>();
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

  private runCpuNode(
    node: WorkflowNode,
    inputs: Record<string, CanonicalArtifact>,
  ): Record<string, CanonicalArtifact> {
    if (node.adapterId === 'pdb-input') {
      const mode = String(node.params.mode ?? 'demo-bundle');
      const rawText =
        mode === 'custom' && String(node.params.pdbText ?? '').trim().length > 0
          ? String(node.params.pdbText)
          : generateSampleHelixBundlePdb();
      const filename = String(
        mode === 'custom' ? node.params.filename || 'uploaded.pdb' : 'demo_3helix_53aa.pdb',
      );
      const parsed = parseStructureText(rawText);
      if (parsed.nResidues < 3) {
        throw new Error('Input PDB/mmCIF must contain at least 3 residues with Cα atoms');
      }
      const geometry = computeGeometryStats(parsed.caCoords, parsed.chainIds);
      const art: StructureArtifact = {
        kind: 'StructureArtifact',
        id: `pdb-${fnv1a(rawText)}`,
        label: `${filename} (L=${parsed.nResidues})`,
        sourceModelId: 'pdb-input',
        length: parsed.nResidues,
        sequence: parsed.sequence,
        caCoords: parsed.caCoords,
        backbone4Coords: parsed.backbone4Coords,
        pdbText: parsed.pdbText,
        cifText: parsed.format === 'mmcif' ? rawText : undefined,
        chainIds: parsed.chainIds,
        chainLengths: parsed.diagnostics.chainLengths,
        diagnostics: parsed.diagnostics,
        geometry,
      };
      return { structure: art };
    }

    if (node.adapterId === 'sequence-input') {
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
        const seenIds = new Map<string, number>();
        blocks.forEach((b, idx) => {
          const lines = b.split('\n');
          const baseHeader = lines[0].trim().split(/\s+/)[0] || `seq-${idx + 1}`;
          const split = splitMultiChainSequence(lines.slice(1).join(''));
          if (split.concatenated.length >= 5) {
            const count = seenIds.get(baseHeader) ?? 0;
            seenIds.set(baseHeader, count + 1);
            const id = count === 0 ? baseHeader : `${baseHeader}_${count + 1}`;
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
        sequences: {
          kind: 'SequenceSet',
          id: `seqin-${fnv1a(raw)}`,
          parentStructureId: 'none',
          sourceModelId: 'sequence-input',
          candidates,
        },
      };
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

      const contig = String(p.contig ?? '70-70').trim() || '70-70';
      const conditions: Record<string, unknown> = {};
      const tags: string[] = [contig];

      const lockChain = String(p.lockChain ?? '').trim();
      if (lockChain) {
        const ch = lockChain[0].toUpperCase();
        conditions.C_CRD = { true: [{ select: `sel('${ch}/*/*/*')` }] };
        conditions.C_SEQ = { true: [{ select: `sel('${ch}/*/*/*')` }] };
        tags.push(`lock:${ch}`);
      }

      const hotspots = String(p.hotspots ?? '').trim();
      if (hotspots) {
        // Group residues by chain (e.g. "A54, A56, A66" -> sel('A/*/[54,56,66]/*'))
        const byChain = new Map<string, string[]>();
        for (const tok of hotspots.split(/[,\s]+/).filter(Boolean)) {
          const m = /^([A-Za-z])?:?(\d+(?:-\d+)?)$/.exec(tok);
          if (!m) continue;
          const ch = (m[1] || 'A').toUpperCase();
          const list = byChain.get(ch) ?? [];
          list.push(m[2]);
          byChain.set(ch, list);
        }
        const exprs = [...byChain.entries()].map(([ch, nums]) => `sel('${ch}/*/[${nums.join(',')}]/*')`);
        if (exprs.length > 0) {
          conditions.C_HOT = { true: [{ select: exprs.join(' | ') }] };
          tags.push(`hot:${hotspots}`);
        }
      }

      const motifSpan = String(p.motifSpan ?? '').trim();
      if (motifSpan) {
        const byChain = new Map<string, string[]>();
        for (const tok of motifSpan.split(/[,\s]+/).filter(Boolean)) {
          const m = /^([A-Za-z])?:?(\d+(?:-\d+)?)$/.exec(tok);
          if (!m) continue;
          const ch = (m[1] || 'A').toUpperCase();
          const list = byChain.get(ch) ?? [];
          list.push(m[2]);
          byChain.set(ch, list);
        }
        const exprs = [...byChain.entries()].map(([ch, nums]) => `sel('${ch}/*/[${nums.join(',')}]/*')`);
        if (exprs.length > 0) {
          const sel = exprs.join(' | ');
          conditions.C_CRDGRP = { 1: [{ select: sel }] };
          const prevSeqTrue = (conditions.C_SEQ as { true?: Array<{ select: string }> } | undefined)?.true ?? [];
          conditions.C_SEQ = { true: [...prevSeqTrue, { select: sel }] };
          if (p.unindexedMotif) conditions.C_IDX = { false: [{ select: sel, group: 'motif' }] };
          tags.push(p.unindexedMotif ? `unidx-motif:${motifSpan}` : `motif:${motifSpan}`);
        }
      }

      const hbondMode = String(p.hbondMode ?? 'none');
      const hbondSel = String(p.hbondSelection ?? '').trim();
      if (hbondMode !== 'none' && hbondSel) {
        const expr = hbondSel.includes('sel(') ? hbondSel : `sel('${hbondSel}')`;
        conditions.C_HB = { [hbondMode]: [{ select: expr }] };
        tags.push(`HB:${hbondMode}`);
      }

      const cathCode = String(p.cathCode ?? '').trim();
      if (cathCode) {
        conditions.C_CATH = { [cathCode]: [{ select: "sel('*/*/*/*')" }] };
        tags.push(`CATH:${cathCode}`);
      }

      const art: ConditionSpecArtifact = {
        kind: 'ConditionSpec',
        id: `cond-${fnv1a(JSON.stringify({ contig, conditions, tpl: template?.id }))}`,
        label: tags.join(' · '),
        contig,
        conditions,
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

      const rows: DesignabilityRow[] = folds.folds.map((f) => {
        if (f.caCoords.length !== structure.caCoords.length) {
          const foldLen = Math.floor(f.caCoords.length / 3);
          const structLen = Math.floor(structure.caCoords.length / 3);
          throw new Error(
            `Length mismatch in designability-scorer: refolded "${f.sequenceId}" has ${foldLen} residues, ` +
              `but designed structure "${structure.label}" has ${structLen} residues.`,
          );
        }
        const align = kabschAlign(f.caCoords, structure.caCoords);
        const alignedFoldPdb = applyTransformToPdb(f.pdbText, align.rotation, align.translation);
        const geom = computeGeometryStats(f.caCoords, f.chainIds);
        const designable = Number.isFinite(align.rmsd) && align.rmsd < scThresh && f.meanPlddt >= plddtThresh;
        return {
          structureId: structure.id,
          sequenceId: f.sequenceId,
          sequence: f.sequence,
          scRmsd: align.rmsd,
          scTm: align.tmScore,
          meanPlddt: f.meanPlddt,
          ptm: f.ptm,
          iptm: f.iptm ?? null,
          designable,
          alignedFoldPdb,
          designedPdb: structure.pdbText,
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

        if (!opts.bypassCache && this.memo.has(cacheKey)) {
          nodeOutputs = this.memo.get(cacheKey)!;
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
              nodeOutputs = this.runCpuNode(node, inputs);
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
    return {
      id: art.id,
      kind: art.kind,
      summary: {
        count: art.folds.length,
        bestPlddt: Number(Math.max(...art.folds.map((f) => f.meanPlddt), 0).toFixed(1)),
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
  return {
    id: art.id,
    kind: art.kind,
    summary: {
      passRate: art.passRate,
      bestScRmsd: art.best ? Number(art.best.scRmsd.toFixed(3)) : null,
      bestScTm: art.best ? Number(art.best.scTm.toFixed(3)) : null,
      bestPlddt: art.best ? Number(art.best.meanPlddt.toFixed(1)) : null,
    },
  };
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
