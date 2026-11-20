// Headless DAG ExecutionScheduler (zero Svelte/DOM dependencies).
// Handles topological execution, content-addressed memoization, Single-Active-GPU-Worker
// VRAM mutex (dispose -> terminate), CPU nodes, and WebMCP / window.__denovo hooks.

import legacy-modelWorker from '$lib/workers/legacy-model.worker?worker';
import MpnnWorker from '$lib/workers/mpnn.worker?worker';
import LocalFoldWorker from '$lib/workers/localfold.worker?worker';
import Rfd4Worker from '$lib/workers/rfd4.worker?worker';
import { generateSampleHelixBundlePdb, parseStructureText } from '$lib/protein-ops/pdb';
import { applyTransformToPdb, computeGeometryStats, kabschAlign } from '$lib/protein-ops/kabsch';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from './registry';
import type {
  ArtifactRef,
  CanonicalArtifact,
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

    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        slot.worker.terminate();
        resolve();
      }, 500);
      const id = this.reqCounter++;
      const onMsg = (ev: MessageEvent) => {
        if (ev.data?.id === id && ev.data?.type === 'disposed') {
          clearTimeout(timer);
          slot.worker.removeEventListener('message', onMsg);
          slot.worker.terminate();
          resolve();
        }
      };
      slot.worker.addEventListener('message', onMsg);
      try {
        slot.worker.postMessage({ id, type: 'dispose' });
      } catch {
        clearTimeout(timer);
        slot.worker.terminate();
        resolve();
      }
    });
  }

  private spawnWorker(adapterId: string): Worker {
    switch (adapterId) {
      case 'legacy-teacher':
        return new legacy-modelWorker();
      case 'web-protein-mpnn':
        return new MpnnWorker();
      case 'localfold-af2':
        return new LocalFoldWorker();
      case 'rfd4-proteina':
        return new Rfd4Worker();
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
        detail: `Releasing ${this.activeGpu.adapterId} VRAM before starting ${node.adapterId}...`,
      });
      await this.releaseActiveGpu();
    }

    if (!this.activeGpu) {
      const worker = this.spawnWorker(node.adapterId);
      this.activeGpu = { adapterId: node.adapterId, worker };
    }

    const worker = this.activeGpu.worker;
    const id = this.reqCounter++;

    return await new Promise((resolve, reject) => {
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
          cleanup();
          resolve({ outputs: m.outputs, elapsedMs: m.elapsedMs ?? 0 });
        } else if (m.type === 'error') {
          cleanup();
          reject(new Error(m.message || `Worker ${node.adapterId} failed`));
        }
      };
      const onError = (err: ErrorEvent) => {
        cleanup();
        reject(new Error(err.message || `Worker ${node.adapterId} crashed`));
      };
      const cleanup = () => {
        worker.removeEventListener('message', onMessage);
        worker.removeEventListener('error', onError);
      };

      worker.addEventListener('message', onMessage);
      worker.addEventListener('error', onError);
      // Plain-object clone strips any reactive Proxy (e.g. Svelte 5 $state) before postMessage.
      const plainParams = JSON.parse(JSON.stringify(node.params ?? {}));
      worker.postMessage({ id, type: 'run', params: plainParams, inputs });
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
        mode === 'custom' ? node.params.filename || 'uploaded.pdb' : 'demo_3helix_56aa.pdb',
      );
      const parsed = parseStructureText(rawText);
      if (parsed.nResidues < 3) {
        throw new Error('Input PDB/mmCIF must contain at least 3 residues with Cα atoms');
      }
      const geometry = computeGeometryStats(parsed.caCoords);
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
        geometry,
      };
      return { structure: art };
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
        const align = kabschAlign(f.caCoords, structure.caCoords);
        const alignedFoldPdb = applyTransformToPdb(f.pdbText, align.rotation, align.translation);
        const geom = computeGeometryStats(f.caCoords);
        const designable = align.rmsd < scThresh && f.meanPlddt >= plddtThresh;
        return {
          structureId: structure.id,
          sequenceId: f.sequenceId,
          sequence: f.sequence,
          scRmsd: align.rmsd,
          scTm: align.tmScore,
          meanPlddt: f.meanPlddt,
          ptm: f.ptm,
          designable,
          alignedFoldPdb,
          designedPdb: structure.pdbText,
          pae: f.pae,
          plddt: f.plddt,
          geometry: geom,
          foldElapsedMs: f.elapsedMs,
        };
      });

      rows.sort((a, b) => a.scRmsd - b.scRmsd);
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
    opts: { bypassCache?: boolean } = {},
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
