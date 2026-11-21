// Deterministic WorkflowPlan -> WorkflowGraph compiler + Cloudflare Workers AI client.
// The LLM outputs a small validated intent JSON; TypeScript deterministically hydrates
// curated benchmarks, adapter defaults, smart sequence routing, multi-oracle folding,
// scRMSD scoring, validation, and auto-layout so zero broken wires can ever be produced.

import { getBenchmarkById } from './benchmarks';
import { autoLayoutGraph, defaultParamsForAdapter, validateGraph } from './graph-utils';
import { LOCALFOLD_MODEL_OPTIONS } from './registry';
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
  | 'af3';

export type PdbTaskId =
  | 'thermompnn'
  | 'ligandmpnn'
  | 'lasermpnn'
  | 'antifold'
  | 'esmif1'
  | 'proteinmpnn';

export interface DeNovoWorkflowPlan {
  kind: 'de_novo';
  generators: GeneratorId[];
  length: number;
  numDesigns: number;
  sequenceDesigner?: SequenceDesignerId;
  numSequences: number;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  score: boolean;
  stabilityScan: boolean;
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
  sequenceDesigner?: SequenceDesignerId;
  numSequences: number;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  score: boolean;
  stabilityScan: boolean;
  summary: string;
}

export interface PdbWorkflowPlan {
  kind: 'pdb';
  pdbId: string;
  crop: string | null;
  task: PdbTaskId;
  targetChain: string | null;
  regions: string | null;
  numSequences: number | null;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  stabilityScan: boolean;
  summary: string;
}

export type WorkflowPlan =
  | DeNovoWorkflowPlan
  | ConditionedDesignWorkflowPlan
  | PdbWorkflowPlan;

export interface PlannerApiResponse {
  ok: boolean;
  plan?: WorkflowPlan;
  model?: string;
  latencyMs?: number;
  error?: string;
  detail?: string;
}

export interface PlannedWorkflowResult {
  plan: WorkflowPlan;
  graph: WorkflowGraph;
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

function makeUniqueIdFactory() {
  const counts = new Map<string, number>();
  return (prefix: string): string => {
    const next = (counts.get(prefix) ?? 0) + 1;
    counts.set(prefix, next);
    return `${prefix}_${next}`;
  };
}

function attachDownstreamEval(opts: {
  nextId: (prefix: string) => string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  branchTag: string;
  backboneNodeId: string;
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

      if (score) {
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
  sequenceDesigner?: SequenceDesignerId;
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
    sequenceDesigner,
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

  // Decide sequence design block:
  // - Genie 3 emits Cα-only backbones -> attach Cα-ProteinMPNN (or explicit sequenceDesigner)
  // - RFD4-Proteina co-designs sequences natively -> use its `sequences` port directly
  //   unless user explicitly asked for an external sequenceDesigner or unconditional MPNN pass
  let seqSourceId = nodeId;
  const needsExternalDesigner =
    genId === 'genie3' || Boolean(sequenceDesigner);

  if (needsExternalDesigner) {
    const designerAdapter: SequenceDesignerId =
      sequenceDesigner ?? 'web-protein-mpnn';
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
  }

  attachDownstreamEval({
    nextId,
    nodes,
    edges,
    branchTag,
    backboneNodeId: nodeId,
    seqSourceNodeId: seqSourceId,
    topK,
    refold,
    foldModels,
    score,
    stabilityScan,
  });
}

export function compileWorkflowPlanToGraph(plan: WorkflowPlan): {
  graph: WorkflowGraph;
  notes: string[];
} {
  const nextId = makeUniqueIdFactory();
  const nodes: WorkflowNode[] = [];
  const edges: WorkflowEdge[] = [];
  const notes: string[] = [];
  const stamp = Date.now().toString(36).slice(-5);

  if (plan.kind === 'de_novo') {
    const gens: GeneratorId[] = plan.generators.length > 0 ? plan.generators : ['genie3'];
    const multiBranch = gens.length > 1;

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
        sequenceDesigner:
          plan.sequenceDesigner ?? (genId === 'rfd4-proteina' ? 'web-protein-mpnn' : undefined),
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
          plan.designTask === 'binder' || bench?.category === 'binder'
            ? 'binder'
            : plan.designTask === 'enzyme-ligand' || bench?.category === 'enzyme-ligand'
              ? 'motif'
              : 'motif';
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
        sequenceDesigner: plan.sequenceDesigner,
        numSequences: plan.numSequences,
        topK: plan.topK,
        refold: plan.refold,
        foldModels: plan.foldModels,
        score: plan.score,
        stabilityScan: plan.stabilityScan,
      });
    }
  } else {
    // kind === 'pdb'
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

    if (plan.task === 'thermompnn') {
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
    } else if (plan.task === 'antifold' || plan.task === 'esmif1') {
      const afId = nextId(plan.task);
      const isAnti = plan.task === 'antifold';
      nodes.push({
        id: afId,
        adapterId: 'antifold-esmif1',
        label: isAnti ? 'AntiFold' : 'ESM-IF1',
        params: {
          ...defaultParamsForAdapter('antifold-esmif1'),
          variant: plan.task,
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
          branchTag: '',
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
        plan.task === 'ligandmpnn'
          ? 'ligandmpnn'
          : plan.task === 'lasermpnn'
            ? 'lasermpnn'
            : 'web-protein-mpnn';
      const label =
        adapterId === 'ligandmpnn'
          ? 'LigandMPNN'
          : adapterId === 'lasermpnn'
            ? 'LASErMPNN'
            : 'ProteinMPNN';
      const seqNodeId = nextId('seq');
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
          branchTag: '',
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
          label: 'ThermoMPNN',
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
  };
}

export async function requestWorkflowPlan(
  prompt: string,
  signal?: AbortSignal,
): Promise<PlannedWorkflowResult> {
  const endpoint =
    (import.meta.env.VITE_PLANNER_URL as string | undefined)?.trim() ||
    DEFAULT_PLANNER_ENDPOINT;

  const resp = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: prompt.trim() }),
    signal,
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

  const { graph, notes } = compileWorkflowPlanToGraph(data.plan);
  return {
    plan: data.plan,
    graph,
    model: data.model || '@cf/google/gemma-4-26b-a4b-it',
    latencyMs: data.latencyMs ?? 0,
    notes,
  };
}
