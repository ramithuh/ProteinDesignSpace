// Deterministic WorkflowPlan -> WorkflowGraph compiler + Cloudflare Workers AI client.
// Supports creating fresh multi-branch workflows (`de_novo`, `conditioned_design`, `pdb`)
// AND safely editing/appending onto the user's current canvas (`edit_current`).

import { getBenchmarkById } from './benchmarks';
import { autoLayoutGraph, defaultParamsForAdapter, validateGraph } from './graph-utils';
import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from './registry';
import type { WorkflowEdge, WorkflowGraph, WorkflowNode } from './types';

export type GeneratorId = 'rfd4-proteina' | 'genie3';
export type SequenceDesignerId = 'web-protein-mpnn' | 'ligandmpnn' | 'lasermpnn';
export type FoldOracleId =
  | 'monomer'
  | 'multimer'
  | 'rosettafold3'
  | 'boltz2'
  | 'protenix2'
  | 'openbind0'
  | 'opendde'
  | 'mol1'
  | 'intellifold2'
  | 'ef2-fast-300m'
  | 'ef2-fast-600m'
  | 'af3';

export type PdbTaskId =
  | 'thermompnn'
  | 'ligandmpnn'
  | 'lasermpnn'
  | 'antifold'
  | 'esmif1'
  | 'proteinmpnn';

export type EditActionId =
  | 'add_stability_scan'
  | 'add_fold_oracles'
  | 'add_sequence_designer'
  | 'insert_top_k';

export interface GraphEditOperation {
  action: EditActionId;
  attachToNodeId: string | null;
  model: PdbTaskId | null;
  foldModels: FoldOracleId[];
  topK: number | null;
  numSequences: number | null;
  targetChain: string | null;
  regions: string | null;
  score: boolean;
}

export interface DeNovoWorkflowPlan {
  kind: 'de_novo';
  generators: GeneratorId[];
  length: number;
  numDesigns: number;
  sequenceDesigners?: SequenceDesignerId[];
  sequenceDesigner?: SequenceDesignerId;
  numSequences: number;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  score: boolean;
  stabilityScan: boolean;
  unsupported?: string[];
  summary: string;
}

export interface ConditionedDesignWorkflowPlan {
  kind: 'conditioned_design';
  designTask: 'motif' | 'binder' | 'enzyme-ligand';
  benchmarkId: string | null;
  pdbId: string | null;
  crop: string | null;
  generators: GeneratorId[];
  length: number | null;
  numDesigns: number;
  sequenceDesigners?: SequenceDesignerId[];
  sequenceDesigner?: SequenceDesignerId;
  numSequences: number;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  score: boolean;
  stabilityScan: boolean;
  unsupported?: string[];
  summary: string;
}

export interface PdbWorkflowPlan {
  kind: 'pdb';
  pdbId: string;
  crop: string | null;
  tasks?: PdbTaskId[];
  task?: PdbTaskId;
  targetChain: string | null;
  regions: string | null;
  numSequences: number | null;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  stabilityScan: boolean;
  unsupported?: string[];
  summary: string;
}

export interface EditCurrentWorkflowPlan {
  kind: 'edit_current';
  operations: GraphEditOperation[];
  unsupported?: string[];
  summary: string;
}

export interface UnsupportedWorkflowPlan {
  kind: 'unsupported';
  unsupported: string[];
  reason: string;
  suggestions: string[];
  summary: string;
}

export type WorkflowPlan =
  | DeNovoWorkflowPlan
  | ConditionedDesignWorkflowPlan
  | PdbWorkflowPlan
  | EditCurrentWorkflowPlan
  | UnsupportedWorkflowPlan;

export class UnsupportedWorkflowError extends Error {
  readonly unsupported: string[];
  readonly suggestions: string[];
  readonly model: string;
  readonly latencyMs: number;

  constructor(opts: {
    reason: string;
    unsupported: string[];
    suggestions: string[];
    model: string;
    latencyMs: number;
  }) {
    super(opts.reason);
    this.name = 'UnsupportedWorkflowError';
    this.unsupported = opts.unsupported;
    this.suggestions = opts.suggestions;
    this.model = opts.model;
    this.latencyMs = opts.latencyMs;
  }
}

export interface PlannerApiResponse {
  ok: boolean;
  plan?: WorkflowPlan;
  model?: string;
  latencyMs?: number;
  error?: string;
  detail?: string;
}

export interface PlannedWorkflowResult {
  plan: Exclude<WorkflowPlan, UnsupportedWorkflowPlan>;
  graph: WorkflowGraph;
  isIncrementalEdit: boolean;
  addedNodeCount: number;
  unsupported: string[];
  model: string;
  latencyMs: number;
  notes: string[];
}

export const DEFAULT_PLANNER_ENDPOINT =
  'https://denovo-planner-worker.ramith.workers.dev/api/plan';

function oracleShortLabel(modelId: FoldOracleId): string {
  const found = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId);
  return found?.shortLabel ?? modelId;
}

function makeUniqueIdFactory(existingIds?: Iterable<string>) {
  const used = new Set<string>(existingIds ?? []);
  const counts = new Map<string, number>();
  return (prefix: string): string => {
    let n = (counts.get(prefix) ?? 0) + 1;
    let candidate = `${prefix}_${n}`;
    while (used.has(candidate)) {
      n++;
      candidate = `${prefix}_${n}`;
    }
    counts.set(prefix, n);
    used.add(candidate);
    return candidate;
  };
}

function attachDownstreamEval(opts: {
  nextId: (prefix: string) => string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  branchTag: string;
  backboneNodeId: string | null;
  seqSourceNodeId: string;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  score: boolean;
  stabilityScan: boolean;
}) {
  const {
    nextId,
    nodes,
    edges,
    branchTag,
    backboneNodeId,
    topK,
    refold,
    foldModels,
    score,
    stabilityScan,
  } = opts;
  let seqNodeId = opts.seqSourceNodeId;

  if (topK && topK > 0) {
    const tkId = nextId('topk');
    nodes.push({
      id: tkId,
      adapterId: 'top-k-filter',
      label: branchTag ? `Top-${topK} (${branchTag})` : `Top-${topK} Filter`,
      params: { ...defaultParamsForAdapter('top-k-filter'), topK },
    });
    edges.push({
      fromNode: seqNodeId,
      fromPort: 'sequences',
      toNode: tkId,
      toPort: 'sequences',
    });
    seqNodeId = tkId;
  }

  let primaryFoldId: string | null = null;
  if (refold) {
    const oracles: FoldOracleId[] = foldModels.length > 0 ? foldModels : ['monomer'];
    for (const oracle of oracles) {
      const foldId = nextId('fold');
      if (!primaryFoldId) primaryFoldId = foldId;
      const oLabel = oracleShortLabel(oracle);
      nodes.push({
        id: foldId,
        adapterId: 'localfold-af2',
        label: branchTag ? `${oLabel} (${branchTag})` : oLabel,
        params: {
          ...defaultParamsForAdapter('localfold-af2'),
          model: oracle,
        },
      });
      edges.push({
        fromNode: seqNodeId,
        fromPort: 'sequences',
        toNode: foldId,
        toPort: 'sequences',
      });

      if (score && backboneNodeId) {
        const scId = nextId('score');
        const scLabel =
          oracles.length > 1 || branchTag
            ? `scRMSD · ${oLabel}${branchTag ? ` (${branchTag})` : ''}`
            : 'scRMSD / scTM';
        nodes.push({
          id: scId,
          adapterId: 'designability-scorer',
          label: scLabel,
          params: { ...defaultParamsForAdapter('designability-scorer') },
        });
        edges.push({
          fromNode: backboneNodeId,
          fromPort: 'structure',
          toNode: scId,
          toPort: 'structure',
        });
        edges.push({
          fromNode: foldId,
          fromPort: 'folds',
          toNode: scId,
          toPort: 'folds',
        });
      }
    }
  }

  if (stabilityScan && primaryFoldId) {
    const thermoId = nextId('thermo');
    nodes.push({
      id: thermoId,
      adapterId: 'thermompnn',
      label: branchTag ? `ThermoMPNN (${branchTag})` : 'ThermoMPNN',
      params: { ...defaultParamsForAdapter('thermompnn') },
    });
    edges.push({
      fromNode: primaryFoldId,
      fromPort: 'structure',
      toNode: thermoId,
      toPort: 'structure',
    });
  }
}

function attachGeneratorBranch(opts: {
  nextId: (prefix: string) => string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  genId: GeneratorId;
  branchTag: string;
  sourcePdbNodeId: string | null;
  paramsOverride: Record<string, unknown>;
  sequenceDesigners: SequenceDesignerId[];
  numSequences: number;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  score: boolean;
  stabilityScan: boolean;
}) {
  const {
    nextId,
    nodes,
    edges,
    genId,
    branchTag,
    sourcePdbNodeId,
    paramsOverride,
    sequenceDesigners,
    numSequences,
    topK,
    refold,
    foldModels,
    score,
    stabilityScan,
  } = opts;

  const nodeId = nextId(genId === 'genie3' ? 'genie' : 'rfd4');
  const genLabel = genId === 'genie3' ? 'Genie 3' : 'RFD4-Proteina';
  nodes.push({
    id: nodeId,
    adapterId: genId,
    label: genLabel,
    params: {
      ...defaultParamsForAdapter(genId),
      ...paramsOverride,
    },
  });

  if (sourcePdbNodeId) {
    edges.push({
      fromNode: sourcePdbNodeId,
      fromPort: 'structure',
      toNode: nodeId,
      toPort: 'structure',
    });
  }

  // Decide sequence design branch(es):
  // - If user explicitly requested 1+ sequenceDesigners (e.g. compare ProteinMPNN vs LigandMPNN), fan out!
  // - Otherwise Genie 3 gets Cα-ProteinMPNN, and RFD4 uses its native `sequences` port directly.
  const effectiveDesigners: Array<SequenceDesignerId | null> =
    sequenceDesigners.length > 0
      ? sequenceDesigners
      : genId === 'genie3'
        ? ['web-protein-mpnn']
        : [null];

  const multiDesigner = effectiveDesigners.length > 1;
  for (const designerAdapter of effectiveDesigners) {
    let seqSourceId = nodeId;
    let subTag = branchTag;

    if (designerAdapter) {
      const seqId = nextId('seq');
      const isCa = designerAdapter === 'web-protein-mpnn' && genId === 'genie3';
      const designerParams: Record<string, unknown> = {
        ...defaultParamsForAdapter(designerAdapter),
        numSequences,
        ...(isCa ? { family: 'ca', modelName: 'v_48_020' } : {}),
      };
      const designerLabel =
        designerAdapter === 'ligandmpnn'
          ? 'LigandMPNN'
          : designerAdapter === 'lasermpnn'
            ? 'LASErMPNN'
            : isCa
              ? 'Cα-ProteinMPNN'
              : 'ProteinMPNN';
      nodes.push({
        id: seqId,
        adapterId: designerAdapter,
        label: branchTag ? `${designerLabel} (${branchTag})` : designerLabel,
        params: designerParams,
      });
      edges.push({
        fromNode: nodeId,
        fromPort: 'structure',
        toNode: seqId,
        toPort: 'structure',
      });
      seqSourceId = seqId;
      if (multiDesigner) {
        subTag = branchTag ? `${branchTag} · ${designerLabel}` : designerLabel;
      }
    }

    attachDownstreamEval({
      nextId,
      nodes,
      edges,
      branchTag: subTag,
      backboneNodeId: nodeId,
      seqSourceNodeId: seqSourceId,
      topK,
      refold,
      foldModels,
      score,
      stabilityScan,
    });
  }
}

function hasOutputPort(node: WorkflowNode, portName: string): boolean {
  const adapter = ADAPTER_REGISTRY[node.adapterId];
  return Boolean(adapter?.outputs?.[portName]);
}

function findUpstreamStructureForSeqNode(
  seqNodeId: string,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
): string | null {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const visited = new Set<string>();
  const queue = [seqNodeId];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    if (visited.has(cur)) continue;
    visited.add(cur);
    const curNode = nodeById.get(cur);
    if (curNode && hasOutputPort(curNode, 'structure') && curNode.adapterId !== 'localfold-af2') {
      return curNode.id;
    }
    for (const e of edges) {
      if (e.toNode === cur && !visited.has(e.fromNode)) {
        queue.push(e.fromNode);
      }
    }
  }
  return null;
}

function applyEditCurrentPlan(
  plan: EditCurrentWorkflowPlan,
  currentGraph: WorkflowGraph,
): { graph: WorkflowGraph; notes: string[] } {
  const nodes: WorkflowNode[] = structuredClone(currentGraph.nodes);
  const edges: WorkflowEdge[] = structuredClone(currentGraph.edges);
  const notes: string[] = [];
  const nextId = makeUniqueIdFactory(nodes.map((n) => n.id));

  const resolveNode = (id: string | null | undefined): WorkflowNode | undefined =>
    id ? nodes.find((n) => n.id === id) : undefined;

  for (const op of plan.operations) {
    if (op.action === 'add_stability_scan') {
      let target = resolveNode(op.attachToNodeId);
      if (!target || !hasOutputPort(target, 'structure')) {
        // Prefer full-backbone structures: LocalFold > LASErMPNN > RFD4 > PDB Input > any structure
        const priority = ['localfold-af2', 'lasermpnn', 'rfd4-proteina', 'pdb-input'];
        for (const aid of priority) {
          const matches = nodes.filter((n) => n.adapterId === aid && hasOutputPort(n, 'structure'));
          if (matches.length > 0) {
            target = matches[matches.length - 1];
            break;
          }
        }
        if (!target) {
          target = [...nodes].reverse().find((n) => hasOutputPort(n, 'structure'));
        }
      }
      if (!target) {
        notes.push('Skipped ThermoMPNN: no structure-producing block exists on the canvas yet.');
        continue;
      }
      if (target.adapterId === 'genie3') {
        notes.push(
          'Attached ThermoMPNN to Genie 3 (note: Genie 3 is Cα-only; add AF2 first for full-backbone ΔΔG scores).',
        );
      }
      const thermoId = nextId('thermo');
      nodes.push({
        id: thermoId,
        adapterId: 'thermompnn',
        label: 'ThermoMPNN',
        params: {
          ...defaultParamsForAdapter('thermompnn'),
          ...(op.targetChain ? { chain: op.targetChain } : {}),
        },
      });
      edges.push({
        fromNode: target.id,
        fromPort: 'structure',
        toNode: thermoId,
        toPort: 'structure',
      });
    } else if (op.action === 'add_fold_oracles') {
      let seqTarget = resolveNode(op.attachToNodeId);
      if (!seqTarget || !hasOutputPort(seqTarget, 'sequences')) {
        const seqPriority = [
          'top-k-filter',
          'web-protein-mpnn',
          'ligandmpnn',
          'lasermpnn',
          'antifold-esmif1',
          'rfd4-proteina',
          'sequence-input',
          'genie3',
        ];
        for (const aid of seqPriority) {
          const matches = nodes.filter((n) => n.adapterId === aid && hasOutputPort(n, 'sequences'));
          if (matches.length > 0) {
            seqTarget = matches[matches.length - 1];
            break;
          }
        }
      }
      if (!seqTarget) {
        notes.push('Skipped refolding oracle: no sequence-producing block exists on the canvas yet.');
        continue;
      }
      const backboneId = findUpstreamStructureForSeqNode(seqTarget.id, nodes, edges);
      const models: FoldOracleId[] =
        op.foldModels.length > 0 ? op.foldModels : ['rosettafold3'];

      attachDownstreamEval({
        nextId,
        nodes,
        edges,
        branchTag: '',
        backboneNodeId: backboneId,
        seqSourceNodeId: seqTarget.id,
        topK: null,
        refold: true,
        foldModels: models,
        score: op.score && Boolean(backboneId),
        stabilityScan: false,
      });
    } else if (op.action === 'add_sequence_designer') {
      let structTarget = resolveNode(op.attachToNodeId);
      if (!structTarget || !hasOutputPort(structTarget, 'structure')) {
        const structPriority = ['pdb-input', 'rfd4-proteina', 'genie3', 'localfold-af2'];
        for (const aid of structPriority) {
          const matches = nodes.filter((n) => n.adapterId === aid && hasOutputPort(n, 'structure'));
          if (matches.length > 0) {
            structTarget = matches[matches.length - 1];
            break;
          }
        }
      }
      if (!structTarget) {
        notes.push('Skipped sequence designer: no structure block exists on the canvas yet.');
        continue;
      }
      const m = op.model ?? 'proteinmpnn';
      const adapterId =
        m === 'ligandmpnn'
          ? 'ligandmpnn'
          : m === 'lasermpnn'
            ? 'lasermpnn'
            : m === 'antifold' || m === 'esmif1'
              ? 'antifold-esmif1'
              : m === 'thermompnn'
                ? 'thermompnn'
                : 'web-protein-mpnn';
      const isCa = adapterId === 'web-protein-mpnn' && structTarget.adapterId === 'genie3';
      const label =
        adapterId === 'ligandmpnn'
          ? 'LigandMPNN'
          : adapterId === 'lasermpnn'
            ? 'LASErMPNN'
            : adapterId === 'thermompnn'
              ? 'ThermoMPNN'
              : adapterId === 'antifold-esmif1'
                ? m === 'esmif1'
                  ? 'ESM-IF1'
                  : 'AntiFold'
                : isCa
                  ? 'Cα-ProteinMPNN'
                  : 'ProteinMPNN';
      const newId = nextId(m);
      const params: Record<string, unknown> = {
        ...defaultParamsForAdapter(adapterId),
        ...(isCa ? { family: 'ca', modelName: 'v_48_020' } : {}),
        ...(adapterId === 'antifold-esmif1' ? { variant: m } : {}),
        ...(op.numSequences && adapterId !== 'thermompnn' ? { numSequences: op.numSequences } : {}),
        ...(op.targetChain && adapterId === 'antifold-esmif1' ? { targetChain: op.targetChain } : {}),
        ...(op.targetChain && adapterId === 'thermompnn' ? { chain: op.targetChain } : {}),
        ...(op.regions && m === 'antifold' ? { regions: op.regions } : {}),
      };
      nodes.push({ id: newId, adapterId, label, params });
      edges.push({
        fromNode: structTarget.id,
        fromPort: 'structure',
        toNode: newId,
        toPort: 'structure',
      });

      if (op.foldModels.length > 0 && hasOutputPort(nodes[nodes.length - 1], 'sequences')) {
        attachDownstreamEval({
          nextId,
          nodes,
          edges,
          branchTag: label,
          backboneNodeId: structTarget.id,
          seqSourceNodeId: newId,
          topK: op.topK,
          refold: true,
          foldModels: op.foldModels,
          score: op.score,
          stabilityScan: false,
        });
      }
    } else if (op.action === 'insert_top_k') {
      const topK = op.topK ?? 2;
      // Find a `sequences` edge feeding a folding node (prefer attachToNodeId if it's a fold node or seq source)
      let targetEdgeIdx = -1;
      if (op.attachToNodeId) {
        targetEdgeIdx = edges.findIndex(
          (e) =>
            e.fromPort === 'sequences' &&
            e.toPort === 'sequences' &&
            (e.toNode === op.attachToNodeId || e.fromNode === op.attachToNodeId),
        );
      }
      if (targetEdgeIdx < 0) {
        targetEdgeIdx = edges.findIndex(
          (e) =>
            e.fromPort === 'sequences' &&
            e.toPort === 'sequences' &&
            nodes.find((n) => n.id === e.toNode)?.adapterId === 'localfold-af2',
        );
      }
      if (targetEdgeIdx >= 0) {
        const orig = edges[targetEdgeIdx];
        const tkId = nextId('topk');
        nodes.push({
          id: tkId,
          adapterId: 'top-k-filter',
          label: `Top-${topK} Filter`,
          params: { ...defaultParamsForAdapter('top-k-filter'), topK },
        });
        // Rewire orig edge: orig.fromNode -> tkId -> orig.toNode
        edges.splice(
          targetEdgeIdx,
          1,
          { fromNode: orig.fromNode, fromPort: 'sequences', toNode: tkId, toPort: 'sequences' },
          { fromNode: tkId, fromPort: 'sequences', toNode: orig.toNode, toPort: 'sequences' },
        );
      } else {
        // No existing fold wire — attach Top-K to the latest sequence-producing node
        const seqNode = [...nodes].reverse().find((n) => hasOutputPort(n, 'sequences'));
        if (!seqNode) {
          notes.push('Skipped Top-K Filter: no sequence-producing block exists on the canvas yet.');
          continue;
        }
        const tkId = nextId('topk');
        nodes.push({
          id: tkId,
          adapterId: 'top-k-filter',
          label: `Top-${topK} Filter`,
          params: { ...defaultParamsForAdapter('top-k-filter'), topK },
        });
        edges.push({
          fromNode: seqNode.id,
          fromPort: 'sequences',
          toNode: tkId,
          toPort: 'sequences',
        });
      }
    }
  }

  const nextGraph: WorkflowGraph = {
    ...currentGraph,
    nodes,
    edges,
  };
  const errors = validateGraph(nextGraph);
  if (errors.length > 0) {
    throw new Error(`Edited workflow failed validation: ${errors.join('; ')}`);
  }
  return {
    graph: autoLayoutGraph(nextGraph),
    notes,
  };
}

export function compileWorkflowPlanToGraph(
  plan: WorkflowPlan,
  currentGraph?: WorkflowGraph,
): {
  graph: WorkflowGraph;
  notes: string[];
  unsupported: string[];
  isIncrementalEdit: boolean;
  addedNodeCount: number;
} {
  if (plan.kind === 'unsupported') {
    throw new UnsupportedWorkflowError({
      reason: plan.reason,
      unsupported: plan.unsupported ?? [],
      suggestions: plan.suggestions ?? [],
      model: '',
      latencyMs: 0,
    });
  }

  const unsupported = Array.isArray(plan.unsupported)
    ? plan.unsupported.filter((s) => typeof s === 'string' && s.trim().length > 0)
    : [];

  if (plan.kind === 'edit_current') {
    if (!currentGraph || currentGraph.nodes.length === 0) {
      throw new Error('Cannot edit current graph because the canvas is empty.');
    }
    const { graph, notes } = applyEditCurrentPlan(plan, currentGraph);
    return {
      graph,
      notes,
      unsupported,
      isIncrementalEdit: true,
      addedNodeCount: Math.max(0, graph.nodes.length - currentGraph.nodes.length),
    };
  }

  const nextId = makeUniqueIdFactory();
  const nodes: WorkflowNode[] = [];
  const edges: WorkflowEdge[] = [];
  const notes: string[] = [];
  const stamp = Date.now().toString(36).slice(-5);

  if (plan.kind === 'de_novo') {
    const gens: GeneratorId[] = plan.generators.length > 0 ? plan.generators : ['genie3'];
    const multiBranch = gens.length > 1;
    const explicitDesigners: SequenceDesignerId[] =
      plan.sequenceDesigners && plan.sequenceDesigners.length > 0
        ? plan.sequenceDesigners
        : plan.sequenceDesigner
          ? [plan.sequenceDesigner]
          : [];

    for (const genId of gens) {
      const branchTag = multiBranch ? (genId === 'genie3' ? 'Genie 3' : 'RFD4') : '';
      const paramsOverride: Record<string, unknown> =
        genId === 'genie3'
          ? {
              taskMode: 'unconditional',
              length: plan.length,
              numDesigns: plan.numDesigns,
            }
          : {
              mode: 'unconditional',
              length: plan.length,
              numDesigns: plan.numDesigns,
            };

      attachGeneratorBranch({
        nextId,
        nodes,
        edges,
        genId,
        branchTag,
        sourcePdbNodeId: null,
        paramsOverride,
        sequenceDesigners:
          explicitDesigners.length > 0
            ? explicitDesigners
            : genId === 'rfd4-proteina'
              ? ['web-protein-mpnn']
              : [],
        numSequences: plan.numSequences,
        topK: plan.topK,
        refold: plan.refold,
        foldModels: plan.foldModels,
        score: plan.score,
        stabilityScan: plan.stabilityScan,
      });
    }
  } else if (plan.kind === 'conditioned_design') {
    const bench = plan.benchmarkId ? getBenchmarkById(plan.benchmarkId) : undefined;
    const pdbId = bench?.pdbId ?? plan.pdbId ?? '1BCF';
    const cropSegments = bench?.cropSegments ?? plan.crop ?? '';
    const pdbNodeId = nextId('pdb');

    nodes.push({
      id: pdbNodeId,
      adapterId: 'pdb-input',
      label: bench ? `PDB: ${bench.name} (${bench.pdbId})` : `PDB: ${pdbId}`,
      params: {
        ...defaultParamsForAdapter('pdb-input'),
        mode: 'pdb-id',
        pdbId,
        filename: `${pdbId}.cif`,
        ...(cropSegments ? { cropSegments } : {}),
        ...(bench ? { benchmarkId: bench.id } : {}),
      },
    });

    let gens: GeneratorId[] =
      plan.generators.length > 0 ? [...plan.generators] : ['rfd4-proteina'];
    if (bench && !bench.supportsGenie3 && gens.includes('genie3')) {
      gens = gens.filter((g) => g !== 'genie3');
      if (gens.length === 0) gens = ['rfd4-proteina'];
      notes.push(
        `${bench.shortLabel} requires all-atom ligand/DNA conditioning, so Genie 3 (Cα-only) was skipped in favor of RFD4-Proteina.`,
      );
    }

    const explicitDesigners: SequenceDesignerId[] =
      plan.sequenceDesigners && plan.sequenceDesigners.length > 0
        ? plan.sequenceDesigners
        : plan.sequenceDesigner
          ? [plan.sequenceDesigner]
          : [];
    const multiBranch = gens.length > 1;
    const cp = bench?.conditionParams;

    for (const genId of gens) {
      const branchTag = multiBranch ? (genId === 'genie3' ? 'Genie 3' : 'RFD4') : '';
      let paramsOverride: Record<string, unknown>;

      if (genId === 'genie3') {
        const taskMode =
          plan.designTask === 'binder' || bench?.category === 'binder' ? 'binder' : 'motif';
        paramsOverride = {
          taskMode,
          numDesigns: plan.numDesigns,
          ...(plan.length ? { length: plan.length } : {}),
          ...(cp?.contig ? { contig: cp.contig } : {}),
          ...(cp?.motifSpan ? { motifSpan: cp.motifSpan } : {}),
          ...(cp?.lockChain ? { lockChain: cp.lockChain } : {}),
          ...(cp?.hotspots ? { hotspots: cp.hotspots } : {}),
          ...(bench ? { autoSeededBenchmarkId: bench.id } : {}),
        };
      } else {
        const mode =
          plan.designTask === 'binder' || bench?.category === 'binder' ? 'binder' : 'motif';
        paramsOverride = {
          mode,
          numDesigns: plan.numDesigns,
          ...(plan.length ? { length: plan.length } : {}),
          ...(cp?.contig ? { contig: cp.contig } : {}),
          ...(cp?.motifSpan ? { motifSpan: cp.motifSpan } : {}),
          ...(cp?.lockChain ? { lockChain: cp.lockChain } : {}),
          ...(cp?.hotspots ? { hotspots: cp.hotspots } : {}),
          ...(cp?.unindexedMotif ? { unindexedMotif: true } : {}),
          ...(cp?.hbondMode ? { hbondMode: cp.hbondMode } : {}),
          ...(cp?.hbondSelection ? { hbondSelection: cp.hbondSelection } : {}),
          ...(bench ? { autoSeededBenchmarkId: bench.id } : {}),
        };
      }

      attachGeneratorBranch({
        nextId,
        nodes,
        edges,
        genId,
        branchTag,
        sourcePdbNodeId: pdbNodeId,
        paramsOverride,
        sequenceDesigners: explicitDesigners,
        numSequences: plan.numSequences,
        topK: plan.topK,
        refold: plan.refold,
        foldModels: plan.foldModels,
        score: plan.score,
        stabilityScan: plan.stabilityScan,
      });
    }
  } else {
    // kind === 'pdb' (supports 1 or multiple parallel tasks, e.g. LigandMPNN vs LASErMPNN!)
    const pdbNodeId = nextId('pdb');
    const isDemo = !plan.pdbId || plan.pdbId.toLowerCase() === 'demo';
    nodes.push({
      id: pdbNodeId,
      adapterId: 'pdb-input',
      label: isDemo ? 'Backbone Input (Demo)' : `RCSB ${plan.pdbId}`,
      params: {
        ...defaultParamsForAdapter('pdb-input'),
        mode: isDemo ? 'demo-bundle' : 'pdb-id',
        ...(isDemo
          ? { filename: 'demo_3helix_53aa.pdb' }
          : {
              pdbId: plan.pdbId,
              filename: `${plan.pdbId}.cif`,
              ...(plan.crop ? { cropSegments: plan.crop } : {}),
            }),
      },
    });

    const taskList: PdbTaskId[] =
      plan.tasks && plan.tasks.length > 0
        ? plan.tasks
        : plan.task
          ? [plan.task]
          : ['thermompnn'];
    const multiTask = taskList.length > 1;

    for (const task of taskList) {
      if (task === 'thermompnn') {
        const thermoId = nextId('thermo');
        nodes.push({
          id: thermoId,
          adapterId: 'thermompnn',
          label: 'ThermoMPNN',
          params: {
            ...defaultParamsForAdapter('thermompnn'),
            ...(plan.targetChain ? { chain: plan.targetChain } : {}),
          },
        });
        edges.push({
          fromNode: pdbNodeId,
          fromPort: 'structure',
          toNode: thermoId,
          toPort: 'structure',
        });
      } else if (task === 'antifold' || task === 'esmif1') {
        const afId = nextId(task);
        const isAnti = task === 'antifold';
        const label = isAnti ? 'AntiFold' : 'ESM-IF1';
        nodes.push({
          id: afId,
          adapterId: 'antifold-esmif1',
          label,
          params: {
            ...defaultParamsForAdapter('antifold-esmif1'),
            variant: task,
            ...(plan.targetChain ? { targetChain: plan.targetChain } : {}),
            ...(isAnti && plan.regions ? { regions: plan.regions } : {}),
            ...(plan.numSequences ? { numSequences: plan.numSequences } : {}),
          },
        });
        edges.push({
          fromNode: pdbNodeId,
          fromPort: 'structure',
          toNode: afId,
          toPort: 'structure',
        });
        if (plan.refold || plan.topK) {
          attachDownstreamEval({
            nextId,
            nodes,
            edges,
            branchTag: multiTask ? label : '',
            backboneNodeId: pdbNodeId,
            seqSourceNodeId: afId,
            topK: plan.topK,
            refold: plan.refold,
            foldModels: plan.foldModels,
            score: plan.refold,
            stabilityScan: plan.stabilityScan,
          });
        }
      } else {
        const adapterId =
          task === 'ligandmpnn'
            ? 'ligandmpnn'
            : task === 'lasermpnn'
              ? 'lasermpnn'
              : 'web-protein-mpnn';
        const label =
          adapterId === 'ligandmpnn'
            ? 'LigandMPNN'
            : adapterId === 'lasermpnn'
              ? 'LASErMPNN'
              : 'ProteinMPNN';
        const seqNodeId = nextId(task);
        nodes.push({
          id: seqNodeId,
          adapterId,
          label,
          params: {
            ...defaultParamsForAdapter(adapterId),
            ...(plan.numSequences ? { numSequences: plan.numSequences } : {}),
          },
        });
        edges.push({
          fromNode: pdbNodeId,
          fromPort: 'structure',
          toNode: seqNodeId,
          toPort: 'structure',
        });

        if (plan.refold || plan.topK) {
          attachDownstreamEval({
            nextId,
            nodes,
            edges,
            branchTag: multiTask ? label : '',
            backboneNodeId: pdbNodeId,
            seqSourceNodeId: seqNodeId,
            topK: plan.topK,
            refold: plan.refold,
            foldModels: plan.foldModels,
            score: plan.refold,
            stabilityScan: plan.stabilityScan,
          });
        } else if (plan.stabilityScan && adapterId === 'lasermpnn') {
          const thermoId = nextId('thermo');
          nodes.push({
            id: thermoId,
            adapterId: 'thermompnn',
            label: multiTask ? `ThermoMPNN (${label})` : 'ThermoMPNN',
            params: { ...defaultParamsForAdapter('thermompnn') },
          });
          edges.push({
            fromNode: seqNodeId,
            fromPort: 'structure',
            toNode: thermoId,
            toPort: 'structure',
          });
        }
      }
    }
  }

  const rawGraph: WorkflowGraph = {
    id: `planned-${plan.kind}-${stamp}`,
    name: plan.summary || 'AI Planned Workflow',
    nodes,
    edges,
  };

  const errors = validateGraph(rawGraph);
  if (errors.length > 0) {
    throw new Error(`Compiled workflow failed validation: ${errors.join('; ')}`);
  }

  return {
    graph: autoLayoutGraph(rawGraph),
    notes,
    unsupported,
    isIncrementalEdit: false,
    addedNodeCount: nodes.length,
  };
}

export async function requestWorkflowPlan(
  prompt: string,
  options?: { currentGraph?: WorkflowGraph; signal?: AbortSignal },
): Promise<PlannedWorkflowResult> {
  const endpoint =
    (import.meta.env.VITE_PLANNER_URL as string | undefined)?.trim() ||
    DEFAULT_PLANNER_ENDPOINT;

  const currentNodes =
    options?.currentGraph?.nodes.slice(0, 20).map((n) => ({
      id: n.id,
      adapterId: n.adapterId,
      label: n.label,
    })) ?? [];

  const resp = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: prompt.trim(),
      ...(currentNodes.length > 0 ? { currentNodes } : {}),
    }),
    signal: options?.signal,
  });

  let data: PlannerApiResponse | null = null;
  try {
    data = (await resp.json()) as PlannerApiResponse;
  } catch {
    data = null;
  }

  if (!resp.ok || !data?.ok || !data.plan) {
    const msg =
      data?.error ||
      (resp.status === 429
        ? 'Rate limit reached — please wait a moment and try again.'
        : `Planner request failed (HTTP ${resp.status}).`);
    throw new Error(msg);
  }

  const model = data.model || '@cf/google/gemma-4-26b-a4b-it';
  const latencyMs = data.latencyMs ?? 0;

  if (data.plan.kind === 'unsupported') {
    throw new UnsupportedWorkflowError({
      reason: data.plan.reason,
      unsupported: data.plan.unsupported ?? [],
      suggestions: data.plan.suggestions ?? [],
      model,
      latencyMs,
    });
  }

  const { graph, notes, unsupported, isIncrementalEdit, addedNodeCount } =
    compileWorkflowPlanToGraph(data.plan, options?.currentGraph);
  return {
    plan: data.plan,
    graph,
    isIncrementalEdit,
    addedNodeCount,
    unsupported,
    model,
    latencyMs,
    notes,
  };
}
