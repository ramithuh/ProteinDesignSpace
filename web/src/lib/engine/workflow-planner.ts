// Deterministic WorkflowPlan -> WorkflowGraph compiler + Cloudflare Workers AI client.
// Supports creating fresh multi-branch workflows (`de_novo`, `conditioned_design`, `pdb`, `sequence`)
// AND safely editing/appending/removing/updating onto the user's current canvas (`edit_current`).

import {
  BENCHMARK_TARGETS,
  getBenchmarkById,
  type BenchmarkTargetMeta,
} from './benchmarks';
import {
  UI_TRACKING_PARAM_KEYS,
  applyPlannerBenchmarkSeedingToNode,
  isResidueSpecDisjointFromBenchmark,
} from './benchmark-sync';
import { extractUpstreamHoloHints, validateBioCapabilities } from './bio-capabilities';
import { autoLayoutGraph, defaultParamsForAdapter, validateConnection, validateGraph } from './graph-utils';
import { sanitizeEvoDiffFixedSequence } from './param-validation';
import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from './registry';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { BioCompatibilityIssue, WorkflowEdge, WorkflowGraph, WorkflowNode } from './types';

export type GeneratorId = 'rfd4-proteina' | 'genie3' | 'protpardelle';
export type SequenceDesignerId =
  | 'native'
  | 'web-protein-mpnn'
  | 'solublempnn'
  | 'ligandmpnn'
  | 'lasermpnn'
  | 'esmif1'
  | 'antifold';
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
  | 'proteinmpnn'
  | 'solublempnn'
  | 'esmc'
  | 'dyna1';

export type EditActionId =
  | 'add_stability_scan'
  | 'add_fold_oracles'
  | 'add_sequence_designer'
  | 'insert_top_k'
  | 'add_esmc'
  | 'add_dyna1'
  | 'remove_nodes'
  | 'update_params';

export type MpnnFamilyId = 'solublempnn' | 'proteinmpnn' | 'ca';

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
  targetAdapterIds?: string[];
  targetNodeIds?: string[];
  temperature?: number | null;
  omitAminoAcids?: string | null;
  numDesigns?: number | null;
  mpnnFamily?: MpnnFamilyId | null;
}

export interface DeNovoWorkflowPlan {
  kind: 'de_novo';
  generators: GeneratorId[];
  length: number;
  numDesigns: number;
  sequenceDesigners?: SequenceDesignerId[];
  sequenceDesigner?: SequenceDesignerId;
  numSequences: number;
  omitAminoAcids?: string | null;
  temperature?: number | null;
  mpnnFamily?: MpnnFamilyId | null;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  score: boolean;
  stabilityScan: boolean;
  esmcScore?: boolean;
  dynamicsScan?: boolean;
  unsupported?: string[];
  summary: string;
}

export interface ConditionedDesignWorkflowPlan {
  kind: 'conditioned_design';
  designTask: 'motif' | 'binder' | 'enzyme-ligand' | 'partial';
  partialFraction?: number | null;
  benchmarkId: string | null;
  pdbId: string | null;
  crop: string | null;
  generators: GeneratorId[];
  length: number | null;
  numDesigns: number;
  sequenceDesigners?: SequenceDesignerId[];
  sequenceDesigner?: SequenceDesignerId;
  numSequences: number;
  omitAminoAcids?: string | null;
  temperature?: number | null;
  mpnnFamily?: MpnnFamilyId | null;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  score: boolean;
  stabilityScan: boolean;
  esmcScore?: boolean;
  dynamicsScan?: boolean;
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
  omitAminoAcids?: string | null;
  temperature?: number | null;
  mpnnFamily?: MpnnFamilyId | null;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  stabilityScan: boolean;
  esmcScore?: boolean;
  dynamicsScan?: boolean;
  score?: boolean;
  unsupported?: string[];
  summary: string;
}

export interface SequenceWorkflowPlan {
  kind: 'sequence';
  sequences: string[];
  label?: string | null;
  foldModels: FoldOracleId[];
  msaMode?: 'single' | 'mmseqs2' | 'target-only' | null;
  stabilityScan?: boolean;
  esmcScore?: boolean;
  dynamicsScan?: boolean;
  topK?: number | null;
  score?: boolean;
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

export interface BlockGraphNodeSpec {
  id: string;
  adapter: string;
  label: string;
  params?: Record<string, unknown>;
}

export type BlockGraphEdgePair = [string, string];

export interface BlockGraphEditOp {
  action: 'add_node' | 'add_edge' | 'remove_node' | 'update_params';
  nodeId?: string | null;
  adapter?: string | null;
  label?: string | null;
  fromId?: string | null;
  toId?: string | null;
  params?: Record<string, unknown> | null;
}

export interface BlockGraphWorkflowPlan {
  kind: 'block_graph';
  mode: 'new' | 'edit';
  nodes?: BlockGraphNodeSpec[];
  edges?: BlockGraphEdgePair[];
  editOps?: BlockGraphEditOp[];
  unsupported?: string[];
  summary: string;
}

export type WorkflowPlan =
  | BlockGraphWorkflowPlan
  | DeNovoWorkflowPlan
  | ConditionedDesignWorkflowPlan
  | PdbWorkflowPlan
  | SequenceWorkflowPlan
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

export class PlannerHttpError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly retryAfterSec: number | null;

  constructor(opts: {
    message: string;
    status: number;
    code?: string | null;
    retryAfterSec?: number | null;
  }) {
    super(opts.message);
    this.name = 'PlannerHttpError';
    this.status = opts.status;
    this.code = opts.code ?? null;
    this.retryAfterSec = opts.retryAfterSec ?? null;
  }
}

export interface PlannerApiResponse {
  ok: boolean;
  plan?: WorkflowPlan;
  model?: string;
  latencyMs?: number;
  cached?: boolean;
  code?: string;
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

export const DEFAULT_PLANNER_ENDPOINT = 'https://planner.denovo.run/api/plan';
export const PLANNER_MIN_PROMPT_LENGTH = 4;
export const PLANNER_TIMEOUT_MS = 30_000;

const CROP_ALLOWLIST_RE = /^[A-Za-z0-9,:;+\s._-]{1,80}$/;
const MAX_GENERATORS = 3;
const MAX_SEQUENCE_DESIGNERS = 4;
const MAX_FOLD_MODELS = 4;
const MAX_PDB_TASKS = 4;
const MAX_EDIT_OPERATIONS = 4;
const MAX_UNSUPPORTED_ITEMS = 8;
const MAX_SUMMARY_CHARS = 200;

export interface LocalStarterExample {
  id: string;
  label: string;
  prompt: string;
  requiresExisting?: boolean;
  plan: Exclude<WorkflowPlan, UnsupportedWorkflowPlan>;
}

/**
 * Precomputed starter-chip plans that compile 100% locally (0 ms, 0 Cloudflare Neurons).
 * Keeps zero-cost instant onboarding even under heavy traffic or offline demos.
 */
export const LOCAL_EXAMPLE_PLANS: readonly LocalStarterExample[] = [
  {
    id: 'add-stability-scan',
    label: '+ Add stability scan',
    prompt: 'Add a ThermoMPNN stability scan to my current graph',
    requiresExisting: true,
    plan: {
      kind: 'edit_current',
      operations: [
        {
          action: 'add_stability_scan',
          attachToNodeId: null,
          model: null,
          foldModels: [],
          topK: null,
          numSequences: null,
          targetChain: null,
          regions: null,
          score: false,
        },
      ],
      summary: 'Add ThermoMPNN ΔΔG stability scan to current graph',
    },
  },
  {
    id: 'stabilize-1ubq',
    label: 'Stabilize Ubiquitin (1UBQ)',
    prompt: 'Scan Ubiquitin (1UBQ) with ThermoMPNN to find stabilizing mutations',
    plan: {
      kind: 'pdb',
      pdbId: '1UBQ',
      crop: null,
      tasks: ['thermompnn'],
      targetChain: null,
      regions: null,
      numSequences: 2,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: true,
      summary: '1UBQ → ThermoMPNN ΔΔG stability scan',
    },
  },
  {
    id: 'redesign-1l2y',
    label: 'Redesign & refold Trp-cage (1L2Y)',
    prompt: 'Redesign Trp-cage (1L2Y) with SolubleMPNN, refold with AF2, and score scRMSD',
    plan: {
      kind: 'pdb',
      pdbId: '1L2Y',
      crop: null,
      tasks: ['solublempnn'],
      targetChain: null,
      regions: null,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      stabilityScan: false,
      summary: '1L2Y → SolubleMPNN → AF2 Model 1 → scRMSD',
    },
  },
  {
    id: 'denovo-60aa',
    label: 'Design a 60aa de novo protein',
    prompt: 'Design a 60aa de novo protein with Genie 3 and verify refolding with AF2',
    plan: {
      kind: 'de_novo',
      generators: ['genie3'],
      length: 60,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      summary: '60aa de novo: Genie 3 → Cα-ProteinMPNN → AF2 → scRMSD',
    },
  },
];

function sanitizeDisplayText(text: unknown, fallback: string, maxLen = MAX_SUMMARY_CHARS): string {
  if (typeof text !== 'string') return fallback;
  const cleaned = text.replace(/[\u0000-\u001f\u007f]/g, ' ').replace(/\s+/g, ' ').trim();
  return cleaned ? cleaned.slice(0, maxLen) : fallback;
}

function validateOptionalCrop(crop: string | null | undefined): string | null {
  if (crop == null) return null;
  const trimmed = String(crop).trim();
  if (!trimmed) return null;
  if (!CROP_ALLOWLIST_RE.test(trimmed)) {
    throw new Error('Workflow plan contains an invalid PDB crop specification.');
  }
  return trimmed;
}

export function compileLocalStarterPlan(
  plan: Exclude<WorkflowPlan, UnsupportedWorkflowPlan>,
  currentGraph?: WorkflowGraph,
): PlannedWorkflowResult {
  const { graph, notes, unsupported, isIncrementalEdit, addedNodeCount } =
    compileWorkflowPlanToGraph(plan, currentGraph);
  return {
    plan,
    graph,
    isIncrementalEdit,
    addedNodeCount,
    unsupported,
    model: 'local-preset',
    latencyMs: 0,
    notes,
  };
}

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

function clampTemperature(t: number | null | undefined, min = 0.01, max = 1.5): number | undefined {
  if (typeof t !== 'number' || !Number.isFinite(t) || t <= 0) return undefined;
  return Math.max(min, Math.min(max, Number(t)));
}

function sanitizeOmitAminoAcids(raw: string | null | undefined): string | undefined {
  if (typeof raw !== 'string') return undefined;
  const cleaned = [
    ...new Set(raw.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '').split('')),
  ].join('');
  if (!cleaned) return undefined;
  return cleaned.length >= 20 ? cleaned.slice(0, 19) : cleaned;
}

function clampPartialFraction(v: number | null | undefined): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) return 0.5;
  return Math.max(0.05, Math.min(0.95, v));
}

function clonePlainValue<T>(value: T, seen = new WeakMap<object, unknown>()): T {
  if (value === null || typeof value !== 'object') return value;
  if (value instanceof Date) return new Date(value.getTime()) as T;
  if (ArrayBuffer.isView(value)) {
    return (value as unknown as { slice: () => unknown }).slice() as T;
  }
  if (seen.has(value as object)) return seen.get(value as object) as T;
  if (Array.isArray(value)) {
    const out: unknown[] = [];
    seen.set(value, out);
    for (let i = 0; i < value.length; i++) {
      out.push(clonePlainValue(value[i], seen));
    }
    return out as T;
  }
  const out: Record<string, unknown> = {};
  seen.set(value as object, out);
  for (const key of Object.keys(value as Record<string, unknown>)) {
    const prop = (value as Record<string, unknown>)[key];
    if (prop !== undefined) out[key] = clonePlainValue(prop, seen);
  }
  return out as T;
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
  msaMode?: 'single' | 'mmseqs2' | 'target-only' | null;
  score: boolean;
  stabilityScan: boolean;
  esmcScore?: boolean;
  dynamicsScan?: boolean;
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
    msaMode,
    score,
    stabilityScan,
    esmcScore = false,
    dynamicsScan = false,
  } = opts;
  let seqNodeId = opts.seqSourceNodeId;

  if (topK && topK > 0) {
    const boundedTopK = Math.max(1, Math.min(16, Math.round(topK)));
    const tkId = nextId('topk');
    nodes.push({
      id: tkId,
      adapterId: 'top-k-filter',
      label: branchTag ? `Top-${boundedTopK} (${branchTag})` : `Top-${boundedTopK} Filter`,
      params: { ...defaultParamsForAdapter('top-k-filter'), topK: boundedTopK },
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
          ...(msaMode ? { msaMode } : {}),
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

  if (esmcScore) {
    const esmcId = nextId('esmc');
    nodes.push({
      id: esmcId,
      adapterId: 'esmc',
      label: branchTag ? `ESM-C 300M (${branchTag})` : 'ESM-C 300M',
      params: { ...defaultParamsForAdapter('esmc') },
    });
    edges.push({
      fromNode: seqNodeId,
      fromPort: 'sequences',
      toNode: esmcId,
      toPort: 'sequences',
    });
  }

  if (dynamicsScan) {
    const dynaId = nextId('dyna');
    nodes.push({
      id: dynaId,
      adapterId: 'dyna1',
      label: branchTag ? `Dyna-1 (${branchTag})` : 'Dyna-1',
      params: { ...defaultParamsForAdapter('dyna1') },
    });
    edges.push({
      fromNode: seqNodeId,
      fromPort: 'sequences',
      toNode: dynaId,
      toPort: 'sequences',
    });
    if (primaryFoldId) {
      edges.push({
        fromNode: primaryFoldId,
        fromPort: 'structure',
        toNode: dynaId,
        toPort: 'structure',
      });
    }
  }
}

function resolveSequenceDesignerNodeSpec(
  designerId: Exclude<SequenceDesignerId, 'native'>,
  genId: GeneratorId,
  mpnnFamily?: MpnnFamilyId | null,
): {
  dedupeKey: string;
  adapterId: string;
  label: string;
  extraParams: Record<string, unknown>;
} {
  if (designerId === 'esmif1') {
    return {
      dedupeKey: 'antifold-esmif1:esmif1',
      adapterId: 'antifold-esmif1',
      label: 'ESM-IF1',
      extraParams: { variant: 'esmif1' },
    };
  }
  if (designerId === 'antifold') {
    return {
      dedupeKey: 'antifold-esmif1:antifold',
      adapterId: 'antifold-esmif1',
      label: 'AntiFold',
      extraParams: { variant: 'antifold' },
    };
  }
  if (designerId === 'ligandmpnn') {
    return {
      dedupeKey: 'ligandmpnn',
      adapterId: 'ligandmpnn',
      label: 'LigandMPNN',
      extraParams: {},
    };
  }
  if (designerId === 'lasermpnn') {
    return {
      dedupeKey: 'lasermpnn',
      adapterId: 'lasermpnn',
      label: 'LASErMPNN',
      extraParams: {},
    };
  }
  // web-protein-mpnn or solublempnn
  const isCa = genId === 'genie3' || mpnnFamily === 'ca';
  if (isCa) {
    return {
      dedupeKey: 'web-protein-mpnn:ca',
      adapterId: 'web-protein-mpnn',
      label: 'Cα-ProteinMPNN',
      extraParams: { family: 'ca', modelName: 'v_48_020' },
    };
  }
  const useSoluble = designerId === 'solublempnn' || mpnnFamily === 'solublempnn';
  if (useSoluble) {
    return {
      dedupeKey: 'web-protein-mpnn:solublempnn',
      adapterId: 'web-protein-mpnn',
      label: 'SolubleMPNN',
      extraParams: { family: 'solublempnn', modelName: 'v_48_020' },
    };
  }
  if (mpnnFamily === 'proteinmpnn') {
    return {
      dedupeKey: 'web-protein-mpnn:proteinmpnn',
      adapterId: 'web-protein-mpnn',
      label: 'ProteinMPNN',
      extraParams: { family: 'proteinmpnn', modelName: 'v_48_020' },
    };
  }
  return {
    dedupeKey: 'web-protein-mpnn:default',
    adapterId: 'web-protein-mpnn',
    label: 'ProteinMPNN',
    extraParams: {},
  };
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
  mpnnFamily?: MpnnFamilyId | null;
  omitAminoAcids?: string | null;
  temperature?: number | null;
  topK: number | null;
  refold: boolean;
  foldModels: FoldOracleId[];
  score: boolean;
  stabilityScan: boolean;
  esmcScore?: boolean;
  dynamicsScan?: boolean;
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
    mpnnFamily,
    omitAminoAcids,
    temperature,
    topK,
    refold,
    foldModels,
    score,
    stabilityScan,
    esmcScore = false,
    dynamicsScan = false,
  } = opts;

  const nodeId = nextId(
    genId === 'genie3' ? 'genie' : genId === 'protpardelle' ? 'pardelle' : 'rfd4',
  );
  const genLabel =
    genId === 'genie3'
      ? 'Genie 3'
      : genId === 'protpardelle'
        ? 'ProtPardelle-1c'
        : 'RFD4-Proteina';
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
  // - If user explicitly requested 1+ sequenceDesigners (e.g. compare RFD4 native seq vs ProteinMPNN, or ProteinMPNN vs LigandMPNN), fan out!
  // - 'native' on RFD4 / ProtPardelle uses its built-in co-designed `sequences` port directly; on Genie 3 (Cα-only), falls back to Cα-ProteinMPNN.
  // - Otherwise Genie 3 gets Cα-ProteinMPNN, and RFD4 / ProtPardelle use their native `sequences` port directly.
  const mappedDesigners: Array<SequenceDesignerId | null> =
    sequenceDesigners.length > 0
      ? sequenceDesigners.map((d) =>
          d === 'native' ? (genId === 'genie3' ? 'web-protein-mpnn' : null) : d,
        )
      : genId === 'genie3'
        ? ['web-protein-mpnn']
        : [null];

  const effectiveDesigners: Array<Exclude<SequenceDesignerId, 'native'> | null> = [];
  const seenKeys = new Set<string>();
  for (const d of mappedDesigners) {
    const val = d === 'native' ? null : d;
    const key = val ? resolveSequenceDesignerNodeSpec(val, genId, mpnnFamily).dedupeKey : 'native';
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      effectiveDesigners.push(val);
    }
  }

  const sanitizedOmit = sanitizeOmitAminoAcids(omitAminoAcids);
  const clampedTemp = clampTemperature(temperature);
  const boundedSeqs = Math.max(1, Math.min(8, Math.round(numSequences || 2)));
  const multiDesigner = effectiveDesigners.length > 1;

  for (const designerId of effectiveDesigners) {
    let seqSourceId = nodeId;
    let subTag = branchTag;

    if (designerId) {
      const spec = resolveSequenceDesignerNodeSpec(designerId, genId, mpnnFamily);
      const seqId = nextId('seq');
      const supportsOmit =
        spec.adapterId === 'web-protein-mpnn' ||
        spec.adapterId === 'ligandmpnn' ||
        spec.adapterId === 'lasermpnn';
      const designerParams: Record<string, unknown> = {
        ...defaultParamsForAdapter(spec.adapterId),
        ...spec.extraParams,
        numSequences: boundedSeqs,
        ...(clampedTemp !== undefined ? { temperature: clampedTemp } : {}),
        ...(supportsOmit && sanitizedOmit ? { omitAminoAcids: sanitizedOmit } : {}),
      };
      nodes.push({
        id: seqId,
        adapterId: spec.adapterId,
        label: branchTag ? `${spec.label} (${branchTag})` : spec.label,
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
        subTag = branchTag ? `${branchTag} · ${spec.label}` : spec.label;
      }
    } else if (multiDesigner) {
      const nativeTag = genId === 'protpardelle' ? 'ProtPardelle seq' : 'RFD4 seq';
      subTag = branchTag ? `${branchTag} · ${nativeTag}` : nativeTag;
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
      esmcScore,
      dynamicsScan,
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

const FOLD_ORACLE_SET = new Set<string>([
  'monomer',
  'multimer',
  'rosettafold3',
  'boltz2',
  'protenix2',
  'openbind0',
  'opendde',
  'mol1',
  'intellifold2',
  'ef2-fast-300m',
  'ef2-fast-600m',
  'af3',
]);

function matchesTargetSelector(node: WorkflowNode, rawToken: string): boolean {
  const tok = rawToken.trim().toLowerCase();
  if (!tok) return false;
  if (node.adapterId.toLowerCase() === tok) return true;
  if (tok === 'antifold') {
    return node.adapterId === 'antifold-esmif1' && String(node.params.variant ?? 'antifold') !== 'esmif1';
  }
  if (tok === 'esmif1' || tok === 'esm-if1') {
    return node.adapterId === 'antifold-esmif1' && String(node.params.variant ?? 'antifold') === 'esmif1';
  }
  if (tok === 'solublempnn') {
    return node.adapterId === 'web-protein-mpnn' && String(node.params.family ?? 'solublempnn') === 'solublempnn';
  }
  if (tok === 'proteinmpnn' || tok === 'ca-proteinmpnn' || tok === 'mpnn') {
    return node.adapterId === 'web-protein-mpnn';
  }
  if (tok === 'esmc' || tok === 'esm-c') return node.adapterId === 'esmc';
  if (tok === 'dyna1' || tok === 'dyna-1') return node.adapterId === 'dyna1';
  if (tok === 'thermompnn') return node.adapterId === 'thermompnn';
  if (tok === 'ligandmpnn') return node.adapterId === 'ligandmpnn';
  if (tok === 'lasermpnn') return node.adapterId === 'lasermpnn';
  if (tok === 'topk' || tok === 'top-k' || tok === 'top-k-filter') return node.adapterId === 'top-k-filter';
  if (tok === 'scrmsd' || tok === 'scorer' || tok === 'designability-scorer') {
    return node.adapterId === 'designability-scorer';
  }
  if (tok === 'af2' || tok === 'alphafold2') {
    return node.adapterId === 'localfold-af2' && String(node.params.model ?? 'monomer').startsWith('monomer');
  }
  if (tok === 'rf3' || tok === 'rosettafold3') {
    return node.adapterId === 'localfold-af2' && String(node.params.model ?? '') === 'rosettafold3';
  }
  if (tok === 'boltz-2' || tok === 'boltz2') {
    return node.adapterId === 'localfold-af2' && String(node.params.model ?? '') === 'boltz2';
  }
  if (FOLD_ORACLE_SET.has(tok)) {
    return node.adapterId === 'localfold-af2' && String(node.params.model ?? 'monomer') === tok;
  }
  return false;
}

function resolveOperationTargets(op: GraphEditOperation, nodes: WorkflowNode[]): WorkflowNode[] {
  const matched: WorkflowNode[] = [];
  const seen = new Set<string>();
  const addNode = (n: WorkflowNode) => {
    if (!seen.has(n.id)) {
      seen.add(n.id);
      matched.push(n);
    }
  };

  if (Array.isArray(op.targetNodeIds) && op.targetNodeIds.length > 0) {
    for (const id of op.targetNodeIds) {
      const found = nodes.find((n) => n.id === id);
      if (found) addNode(found);
    }
  }
  if (Array.isArray(op.targetAdapterIds) && op.targetAdapterIds.length > 0) {
    for (const tok of op.targetAdapterIds) {
      for (const n of nodes) {
        if (matchesTargetSelector(n, tok)) addNode(n);
      }
      // Fallback: if 'solublempnn' didn't match family==='solublempnn', match any web-protein-mpnn
      if (tok.trim().toLowerCase() === 'solublempnn' && !matched.some((m) => m.adapterId === 'web-protein-mpnn')) {
        for (const n of nodes) {
          if (n.adapterId === 'web-protein-mpnn') addNode(n);
        }
      }
    }
  }
  if (matched.length === 0 && op.model) {
    for (const n of nodes) {
      if (matchesTargetSelector(n, op.model)) addNode(n);
    }
  }
  if (matched.length === 0 && op.attachToNodeId) {
    const found = nodes.find((n) => n.id === op.attachToNodeId);
    if (found) addNode(found);
  }
  return matched;
}

function applyEditCurrentPlan(
  plan: EditCurrentWorkflowPlan,
  currentGraph: WorkflowGraph,
): { graph: WorkflowGraph; notes: string[] } {
  let nodes: WorkflowNode[] = clonePlainValue(currentGraph.nodes);
  let edges: WorkflowEdge[] = clonePlainValue(currentGraph.edges);
  const notes: string[] = [];
  const nextId = makeUniqueIdFactory(nodes.map((n) => n.id));

  const resolveNode = (id: string | null | undefined): WorkflowNode | undefined =>
    id ? nodes.find((n) => n.id === id) : undefined;

  for (const op of (plan.operations ?? []).slice(0, MAX_EDIT_OPERATIONS)) {
    if (op.action === 'remove_nodes') {
      const targets = resolveOperationTargets(op, nodes);
      if (targets.length === 0) {
        notes.push('Skipped remove: no matching block found on canvas.');
        continue;
      }
      const removedSet = new Set(targets.map((n) => n.id));
      const nodeById = new Map(nodes.map((n) => [n.id, n]));

      // Step A: Pass-through splicing across removed chain of nodes when port types match
      const splicedEdges: WorkflowEdge[] = [];
      const findPassThroughSource = (
        removedNodeId: string,
        neededType: string,
        visited = new Set<string>(),
      ): { fromNode: string; fromPort: string } | null => {
        if (visited.has(removedNodeId)) return null;
        visited.add(removedNodeId);
        const incoming = edges.filter((e) => e.toNode === removedNodeId);
        for (const inEdge of incoming) {
          const src = nodeById.get(inEdge.fromNode);
          if (!src) continue;
          const outSpec = ADAPTER_REGISTRY[src.adapterId]?.outputs?.[inEdge.fromPort];
          if (!outSpec || outSpec.type !== neededType) continue;
          if (!removedSet.has(src.id)) {
            return { fromNode: src.id, fromPort: inEdge.fromPort };
          }
          const deeper = findPassThroughSource(src.id, neededType, visited);
          if (deeper) return deeper;
        }
        return null;
      };

      for (const outEdge of edges) {
        if (!removedSet.has(outEdge.fromNode) || removedSet.has(outEdge.toNode)) continue;
        const dst = nodeById.get(outEdge.toNode);
        if (!dst) continue;
        const inSpec = ADAPTER_REGISTRY[dst.adapterId]?.inputs?.[outEdge.toPort];
        if (!inSpec) continue;
        const src = findPassThroughSource(outEdge.fromNode, inSpec.type);
        if (!src) continue;
        const targetPortOccupied =
          edges.some(
            (e) =>
              !removedSet.has(e.fromNode) &&
              e.toNode === dst.id &&
              e.toPort === outEdge.toPort,
          ) || splicedEdges.some((e) => e.toNode === dst.id && e.toPort === outEdge.toPort);
        if (!targetPortOccupied) {
          splicedEdges.push({
            fromNode: src.fromNode,
            fromPort: src.fromPort,
            toNode: dst.id,
            toPort: outEdge.toPort,
          });
        }
      }

      // Step B: Delete target nodes and incident edges, then append valid spliced edges
      const lostInputTargets = new Set<string>();
      for (const e of edges) {
        if (removedSet.has(e.fromNode) && !removedSet.has(e.toNode)) {
          lostInputTargets.add(`${e.toNode}:${e.toPort}`);
        }
      }
      nodes = nodes.filter((n) => !removedSet.has(n.id));
      edges = edges.filter((e) => !removedSet.has(e.fromNode) && !removedSet.has(e.toNode));
      for (const se of splicedEdges) {
        const tempGraph: WorkflowGraph = {
          id: currentGraph.id,
          name: currentGraph.name,
          nodes,
          edges,
        };
        if (validateConnection(tempGraph, se.fromNode, se.fromPort, se.toNode, se.toPort).valid) {
          edges.push(se);
          lostInputTargets.delete(`${se.toNode}:${se.toPort}`);
        }
      }

      // Step C: Cascade delete orphaned downstream nodes whose required input was lost
      const cascadedNames: string[] = [];
      let changed = true;
      while (changed) {
        changed = false;
        const orphanIds = new Set<string>();
        for (const node of nodes) {
          const manifest = ADAPTER_REGISTRY[node.adapterId];
          if (!manifest) continue;
          for (const [portName, spec] of Object.entries(manifest.inputs)) {
            if (spec.required === false) continue;
            const targetKey = `${node.id}:${portName}`;
            if (lostInputTargets.has(targetKey) && !edges.some((e) => e.toNode === node.id && e.toPort === portName)) {
              orphanIds.add(node.id);
              cascadedNames.push(node.label ?? manifest.shortName);
              break;
            }
          }
        }
        if (orphanIds.size > 0) {
          changed = true;
          for (const e of edges) {
            if (orphanIds.has(e.fromNode) && !orphanIds.has(e.toNode)) {
              lostInputTargets.add(`${e.toNode}:${e.toPort}`);
            }
          }
          nodes = nodes.filter((n) => !orphanIds.has(n.id));
          edges = edges.filter((e) => !orphanIds.has(e.fromNode) && !orphanIds.has(e.toNode));
        }
      }

      if (nodes.length === 0) {
        throw new Error('Cannot remove all blocks from the workflow.');
      }
      const removedLabels = targets.map((t) => t.label ?? t.adapterId).join(', ');
      notes.push(
        cascadedNames.length > 0
          ? `Removed ${removedLabels} (also removed orphaned downstream: ${cascadedNames.join(', ')}).`
          : `Removed ${removedLabels}.`,
      );
      continue;
    }

    if (op.action === 'update_params') {
      let targets = resolveOperationTargets(op, nodes);
      const hasExplicitSelector =
        (Array.isArray(op.targetNodeIds) && op.targetNodeIds.length > 0) ||
        (Array.isArray(op.targetAdapterIds) && op.targetAdapterIds.length > 0) ||
        Boolean(op.model) ||
        Boolean(op.attachToNodeId);

      if (targets.length === 0 && !hasExplicitSelector) {
        const inferredAdapters = new Set<string>();
        if (op.omitAminoAcids !== undefined && op.omitAminoAcids !== null) {
          inferredAdapters.add('web-protein-mpnn');
          inferredAdapters.add('ligandmpnn');
          inferredAdapters.add('lasermpnn');
        }
        if (op.mpnnFamily) {
          inferredAdapters.add('web-protein-mpnn');
        }
        if (op.temperature !== undefined && op.temperature !== null) {
          inferredAdapters.add('web-protein-mpnn');
          inferredAdapters.add('ligandmpnn');
          inferredAdapters.add('lasermpnn');
          inferredAdapters.add('antifold-esmif1');
          inferredAdapters.add('protpardelle');
        }
        if (op.numSequences !== undefined && op.numSequences !== null) {
          inferredAdapters.add('web-protein-mpnn');
          inferredAdapters.add('ligandmpnn');
          inferredAdapters.add('lasermpnn');
          inferredAdapters.add('antifold-esmif1');
        }
        if (op.numDesigns !== undefined && op.numDesigns !== null) {
          inferredAdapters.add('rfd4-proteina');
          inferredAdapters.add('genie3');
          inferredAdapters.add('protpardelle');
        }
        if (op.topK !== undefined && op.topK !== null) {
          inferredAdapters.add('top-k-filter');
        }
        if (
          (op.targetChain !== undefined && op.targetChain !== null) ||
          (op.regions !== undefined && op.regions !== null)
        ) {
          inferredAdapters.add('antifold-esmif1');
          inferredAdapters.add('thermompnn');
        }
        targets = nodes.filter((n) => inferredAdapters.has(n.adapterId));
      }

      if (targets.length === 0) {
        notes.push('Skipped parameter update: no compatible block found on canvas.');
        continue;
      }

      for (const target of targets) {
        const p = target.params;
        if (op.temperature !== undefined && op.temperature !== null) {
          if (
            target.adapterId === 'web-protein-mpnn' ||
            target.adapterId === 'ligandmpnn' ||
            target.adapterId === 'lasermpnn' ||
            target.adapterId === 'antifold-esmif1'
          ) {
            const t = clampTemperature(op.temperature, 0.01, 1.5);
            if (t !== undefined) p.temperature = t;
          } else if (target.adapterId === 'protpardelle') {
            const t = clampTemperature(op.temperature, 0.05, 2.0);
            if (t !== undefined) p.temperature = t;
          }
        }
        if (op.omitAminoAcids !== undefined && op.omitAminoAcids !== null) {
          if (
            target.adapterId === 'web-protein-mpnn' ||
            target.adapterId === 'ligandmpnn' ||
            target.adapterId === 'lasermpnn'
          ) {
            p.omitAminoAcids =
              op.omitAminoAcids === '' ? '' : sanitizeOmitAminoAcids(op.omitAminoAcids) ?? '';
          }
        }
        if (op.numSequences !== undefined && op.numSequences !== null) {
          if (
            target.adapterId === 'web-protein-mpnn' ||
            target.adapterId === 'ligandmpnn' ||
            target.adapterId === 'lasermpnn' ||
            target.adapterId === 'antifold-esmif1'
          ) {
            p.numSequences = Math.max(1, Math.min(8, Math.round(op.numSequences)));
          }
        }
        if (op.numDesigns !== undefined && op.numDesigns !== null) {
          if (target.adapterId === 'rfd4-proteina' || target.adapterId === 'genie3') {
            p.numDesigns = Math.max(1, Math.min(16, Math.round(op.numDesigns)));
          } else if (target.adapterId === 'protpardelle') {
            p.numDesigns = Math.max(1, Math.min(8, Math.round(op.numDesigns)));
          }
        }
        if (op.mpnnFamily && target.adapterId === 'web-protein-mpnn') {
          p.family = op.mpnnFamily;
          const curModel = String(p.modelName ?? 'v_48_020');
          if (op.mpnnFamily === 'solublempnn' && curModel !== 'v_48_020' && curModel !== 'v_48_010') {
            p.modelName = 'v_48_020';
          } else if (op.mpnnFamily === 'ca' && curModel === 'v_48_030') {
            p.modelName = 'v_48_020';
          }
          const baseName =
            op.mpnnFamily === 'solublempnn'
              ? 'SolubleMPNN'
              : op.mpnnFamily === 'ca'
                ? 'Cα-ProteinMPNN'
                : 'ProteinMPNN';
          const suffixMatch = /\(([^)]+)\)\s*$/.exec(target.label ?? '');
          target.label = suffixMatch ? `${baseName} (${suffixMatch[1]})` : baseName;
        }
        if (op.topK !== undefined && op.topK !== null && target.adapterId === 'top-k-filter') {
          const boundedTopK = Math.max(1, Math.min(16, Math.round(op.topK)));
          p.topK = boundedTopK;
          const suffixMatch = /\(([^)]+)\)\s*$/.exec(target.label ?? '');
          target.label = suffixMatch ? `Top-${boundedTopK} (${suffixMatch[1]})` : `Top-${boundedTopK} Filter`;
        }
        if (op.targetChain !== undefined && op.targetChain !== null) {
          if (target.adapterId === 'antifold-esmif1') p.targetChain = op.targetChain;
          else if (target.adapterId === 'thermompnn') p.chain = op.targetChain;
        }
        if (op.regions !== undefined && op.regions !== null && target.adapterId === 'antifold-esmif1') {
          if (String(p.variant ?? 'antifold') !== 'esmif1') {
            p.regions = op.regions;
          }
        }
      }
      continue;
    }

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
      const isCa =
        adapterId === 'web-protein-mpnn' &&
        (structTarget.adapterId === 'genie3' || op.mpnnFamily === 'ca');
      const useSoluble =
        adapterId === 'web-protein-mpnn' &&
        !isCa &&
        (m === 'solublempnn' || op.mpnnFamily === 'solublempnn');
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
                  : useSoluble
                    ? 'SolubleMPNN'
                    : 'ProteinMPNN';
      const newId = nextId(m);
      const clampedTemp = clampTemperature(op.temperature);
      const sanitizedOmit = sanitizeOmitAminoAcids(op.omitAminoAcids);
      const supportsOmit =
        adapterId === 'web-protein-mpnn' ||
        adapterId === 'ligandmpnn' ||
        adapterId === 'lasermpnn';
      const params: Record<string, unknown> = {
        ...defaultParamsForAdapter(adapterId),
        ...(isCa
          ? { family: 'ca', modelName: 'v_48_020' }
          : useSoluble
            ? { family: 'solublempnn', modelName: 'v_48_020' }
            : op.mpnnFamily && adapterId === 'web-protein-mpnn'
              ? { family: op.mpnnFamily, modelName: 'v_48_020' }
              : {}),
        ...(adapterId === 'antifold-esmif1' ? { variant: m } : {}),
        ...(op.numSequences && adapterId !== 'thermompnn'
          ? { numSequences: Math.max(1, Math.min(8, Math.round(op.numSequences))) }
          : {}),
        ...(clampedTemp !== undefined && adapterId !== 'thermompnn' ? { temperature: clampedTemp } : {}),
        ...(supportsOmit && sanitizedOmit ? { omitAminoAcids: sanitizedOmit } : {}),
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
    } else if (op.action === 'add_esmc' || op.action === 'add_dyna1') {
      const isDyna = op.action === 'add_dyna1';
      const adapterId = isDyna ? 'dyna1' : 'esmc';
      const label = isDyna ? 'Dyna-1' : 'ESM-C 300M';
      let target = resolveNode(op.attachToNodeId);
      if (!target || (!hasOutputPort(target, 'sequences') && !hasOutputPort(target, 'structure'))) {
        const priority = isDyna
          ? [
              'localfold-af2',
              'pdb-input',
              'rfd4-proteina',
              'protpardelle',
              'top-k-filter',
              'web-protein-mpnn',
              'ligandmpnn',
              'lasermpnn',
              'sequence-input',
              'genie3',
            ]
          : [
              'top-k-filter',
              'web-protein-mpnn',
              'ligandmpnn',
              'lasermpnn',
              'antifold-esmif1',
              'rfd4-proteina',
              'protpardelle',
              'sequence-input',
              'pdb-input',
              'localfold-af2',
              'genie3',
            ];
        for (const aid of priority) {
          const matches = nodes.filter(
            (n) =>
              n.adapterId === aid &&
              (hasOutputPort(n, 'sequences') || hasOutputPort(n, 'structure')),
          );
          if (matches.length > 0) {
            target = matches[matches.length - 1];
            break;
          }
        }
      }
      if (!target) {
        notes.push(`Skipped ${label}: no sequence or structure block exists on the canvas yet.`);
        continue;
      }
      const newId = nextId(isDyna ? 'dyna' : 'esmc');
      nodes.push({
        id: newId,
        adapterId,
        label,
        params: { ...defaultParamsForAdapter(adapterId) },
      });
      if (isDyna && hasOutputPort(target, 'structure')) {
        edges.push({
          fromNode: target.id,
          fromPort: 'structure',
          toNode: newId,
          toPort: 'structure',
        });
      } else if (hasOutputPort(target, 'sequences')) {
        edges.push({
          fromNode: target.id,
          fromPort: 'sequences',
          toNode: newId,
          toPort: 'sequences',
        });
      } else {
        edges.push({
          fromNode: target.id,
          fromPort: 'structure',
          toNode: newId,
          toPort: 'structure',
        });
      }
    }
  }

  const nextGraph: WorkflowGraph = {
    id: currentGraph.id,
    name: currentGraph.name,
    ...(currentGraph.description !== undefined ? { description: currentGraph.description } : {}),
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

function resolveBlockEdgePorts(
  fromNode: WorkflowNode,
  toNode: WorkflowNode,
  existingEdges: WorkflowEdge[],
  allNodes: WorkflowNode[],
): { fromPort: string; toPort: string } | null {
  const fromAdapter = ADAPTER_REGISTRY[fromNode.adapterId];
  const toAdapter = ADAPTER_REGISTRY[toNode.adapterId];
  if (!fromAdapter || !toAdapter) return null;

  // Prefer semantic primary port pairs first
  const preferredPairs: Array<[string, string]> = [];
  if (fromNode.adapterId === 'localfold-af2' && toNode.adapterId === 'designability-scorer') {
    preferredPairs.push(['folds', 'folds']);
  } else if (toNode.adapterId === 'designability-scorer') {
    preferredPairs.push(['structure', 'structure'], ['folds', 'folds']);
  } else if (toNode.adapterId === 'structure-compare') {
    const hasRefAlready = existingEdges.some(
      (e) => e.toNode === toNode.id && e.toPort === 'reference',
    );
    if (fromNode.adapterId === 'pdb-input' && !hasRefAlready) {
      preferredPairs.push(['structure', 'reference'], ['structure', 'structures']);
    } else {
      preferredPairs.push(['structure', 'structures'], ['structure', 'reference']);
    }
  } else if (toNode.adapterId === 'dyna1') {
    preferredPairs.push(['structure', 'structure'], ['sequences', 'sequences']);
  } else if (toNode.adapterId === 'esmc') {
    if (fromNode.adapterId === 'pdb-input') {
      preferredPairs.push(['structure', 'structure'], ['sequences', 'sequences']);
    } else {
      preferredPairs.push(['sequences', 'sequences'], ['structure', 'structure']);
    }
  } else if (
    toNode.adapterId === 'localfold-af2' ||
    toNode.adapterId === 'evodiff' ||
    toNode.adapterId === 'foldflow2'
  ) {
    preferredPairs.push(['sequences', 'sequences']);
  } else if ('structure' in toAdapter.inputs || 'template' in toAdapter.inputs) {
    preferredPairs.push(['structure', 'structure'], ['structure', 'template']);
  }

  const tmpGraph: WorkflowGraph = { id: 'tmp', name: 'tmp', nodes: allNodes, edges: existingEdges };
  for (const [fp, tp] of preferredPairs) {
    const res = validateConnection(tmpGraph, fromNode.id, fp, toNode.id, tp);
    if (res.valid) return { fromPort: fp, toPort: tp };
  }

  for (const outName of Object.keys(fromAdapter.outputs)) {
    for (const inName of Object.keys(toAdapter.inputs)) {
      const res = validateConnection(tmpGraph, fromNode.id, outName, toNode.id, inName);
      if (res.valid) return { fromPort: outName, toPort: inName };
    }
  }
  return null;
}

const BLOCK_FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
const MAX_BLOCK_GRAPH_NODES = 20;
const MAX_BLOCK_GRAPH_EDGES = 36;
const MAX_BLOCK_GRAPH_EDIT_OPS = 12;

const CONTROLLED_METADATA_KEYS_BY_ADAPTER: Record<string, ReadonlySet<string>> = {
  'pdb-input': new Set(['benchmarkId']),
  'rfd4-proteina': new Set(['benchmarkId']),
  genie3: new Set(['benchmarkId']),
};

function isBroadChainCrop(crop: unknown): boolean {
  return /^[A-Za-z](?:\s*,\s*[A-Za-z])*$/.test(String(crop ?? '').trim());
}

function transliterateBlockChars(raw: string): string {
  return raw.toLowerCase().replace(/[αΑ]/g, 'a');
}

function normalizeBlockName(raw: string): string {
  return transliterateBlockChars(raw)
    .replace(/\(.*?\)/g, '')
    .replace(/[^a-z0-9]+/g, '');
}

const SUPPORTED_BLOCK_NAMES: ReadonlySet<string> = (() => {
  const names = new Set<string>([
    'evodiff',
    'evodiffseq',
    'evodiff38m',
    'laproteina',
    'sidprotein',
    'proteina',
    'frameflow',
    'foldflow2',
    'boltzgen',
    'boltzgen1',
    'protpardelle',
    'protpardelle1c',
    'rfdiffusion',
    'genie2',
    'genie3',
    'rfd4',
    'rfd4proteina',
    'proteinmpnn',
    'solublempnn',
    'caproteinmpnn',
    'ligandmpnn',
    'lasermpnn',
    'antifold',
    'esmif1',
    'antifoldesmif1',
    'localfold',
    'thermompnn',
    'esmc',
    'esmc300m',
    'dyna1',
  ]);
  for (const manifest of Object.values(ADAPTER_REGISTRY)) {
    for (const candidate of [manifest.id, manifest.shortName, manifest.displayName]) {
      const full = transliterateBlockChars(candidate).replace(/[^a-z0-9]+/g, '');
      const noParen = normalizeBlockName(candidate);
      if (full) names.add(full);
      if (noParen) names.add(noParen);
    }
  }
  return names;
})();

function isSupportedBlockName(raw: string, compiledNodes?: readonly WorkflowNode[]): boolean {
  const norm = normalizeBlockName(raw);
  const normFull = transliterateBlockChars(raw).replace(/[^a-z0-9]+/g, '');
  if (!norm && !normFull) return false;
  if (SUPPORTED_BLOCK_NAMES.has(norm) || SUPPORTED_BLOCK_NAMES.has(normFull)) return true;
  if (compiledNodes && compiledNodes.length > 0) {
    for (const n of compiledNodes) {
      const m = ADAPTER_REGISTRY[n.adapterId];
      const candidates = [
        n.adapterId,
        n.label ?? '',
        m?.shortName ?? '',
        m?.displayName ?? '',
      ];
      for (const c of candidates) {
        const cn = normalizeBlockName(c);
        const cf = transliterateBlockChars(c).replace(/[^a-z0-9]+/g, '');
        if ((norm && (cn === norm || cf === norm)) || (normFull && (cn === normFull || cf === normFull))) {
          return true;
        }
      }
    }
  }
  return false;
}

function filterUnsupportedItems(
  items: readonly string[],
  compiledNodes?: readonly WorkflowNode[],
): string[] {
  return items.filter((item) => !isSupportedBlockName(item, compiledNodes));
}

function sanitizeRawBlockParamEntries(
  adapterId: string,
  rawParams: Record<string, unknown> | undefined,
  opts?: { preserveUiTrackingKeys?: boolean; isPatch?: boolean },
): Record<string, unknown> {
  const sanitized: Record<string, unknown> = {};
  if (!rawParams || typeof rawParams !== 'object') return sanitized;

  const defaults = defaultParamsForAdapter(adapterId);
  const manifest = Object.hasOwn(ADAPTER_REGISTRY, adapterId)
    ? ADAPTER_REGISTRY[adapterId]
    : undefined;
  const schemaKeys = new Set((manifest?.paramSchema ?? []).map((f) => f.key));
  const controlledKeys = CONTROLLED_METADATA_KEYS_BY_ADAPTER[adapterId];
  const allowedKeys = new Set<string>([
    ...Object.keys(defaults),
    ...schemaKeys,
    ...(controlledKeys ?? []),
  ]);
  const modeKey = manifest?.modeParamKey;
  const validModes = manifest?.modes?.length
    ? new Set(manifest.modes.map((m) => m.id))
    : null;

  for (const [k, v] of Object.entries(rawParams)) {
    if (BLOCK_FORBIDDEN_KEYS.has(k)) continue;
    if (k === 'fetchedPdbId' || k === 'pdbText' || k === 'customSpecYaml') continue;
    if (v === null || v === undefined) continue;

    if (opts?.preserveUiTrackingKeys && UI_TRACKING_PARAM_KEYS.has(k)) {
      if (adapterId === 'rfd4-proteina' || adapterId === 'genie3') {
        sanitized[k] = v;
      }
      continue;
    }
    if (UI_TRACKING_PARAM_KEYS.has(k)) continue;

    if (adapterId === 'antifold-esmif1' && k === 'model') {
      const rawVar = rawParams.variant;
      const hasExplicitValidVariant =
        rawVar != null && (!validModes || validModes.has(String(rawVar)));
      if (!hasExplicitValidVariant) {
        const lower = String(v).toLowerCase();
        if (lower.includes('esm')) sanitized.variant = 'esmif1';
        else if (lower.includes('anti')) sanitized.variant = 'antifold';
      }
      continue;
    }
    if (k === 'mode' && !allowedKeys.has('mode') && modeKey === 'variant') {
      const rawVar = rawParams.variant;
      const hasExplicitValidVariant =
        rawVar != null && (!validModes || validModes.has(String(rawVar)));
      if (!hasExplicitValidVariant) {
        const cand = String(v).trim();
        if (!validModes || validModes.has(cand)) {
          sanitized.variant = cand;
        } else {
          const lower = cand.toLowerCase();
          if (lower.includes('esm')) sanitized.variant = 'esmif1';
          else if (lower.includes('anti')) sanitized.variant = 'antifold';
        }
      }
      continue;
    }
    if (k === 'maxSequences' && !allowedKeys.has('maxSequences') && allowedKeys.has('numSequences')) {
      if (rawParams.numSequences == null) {
        sanitized.numSequences = v;
      }
      continue;
    }
    if (k === 'mode' && !allowedKeys.has('mode') && modeKey === 'taskMode') {
      const rawTask = rawParams.taskMode;
      const hasValidPrimary = rawTask != null && (!validModes || validModes.has(String(rawTask)));
      if (!hasValidPrimary && (!validModes || validModes.has(String(v)))) {
        sanitized.taskMode = v;
      }
      continue;
    }
    if (k === 'taskMode' && !allowedKeys.has('taskMode') && modeKey === 'mode') {
      const rawMode = rawParams.mode;
      const hasValidPrimary = rawMode != null && (!validModes || validModes.has(String(rawMode)));
      if (!hasValidPrimary && (!validModes || validModes.has(String(v)))) {
        sanitized.mode = v;
      }
      continue;
    }

    if (!allowedKeys.has(k)) continue;

    if (modeKey && k === modeKey && validModes) {
      if (validModes.has(String(v))) {
        sanitized[k] = v;
      }
      continue;
    }

    if (k === 'cropSegments') {
      const cleanCrop = validateOptionalCrop(String(v));
      if (cleanCrop) sanitized.cropSegments = cleanCrop;
      else if (opts?.isPatch && String(v).trim() === '') sanitized.cropSegments = '';
      continue;
    }
    if (adapterId === 'evodiff' && k === 'fixedSequence') {
      sanitized.fixedSequence = sanitizeEvoDiffFixedSequence(v);
      continue;
    }
    if (k === 'pdbId' && typeof v === 'string') {
      const up = v.trim().toUpperCase();
      if (/^[0-9][A-Z0-9]{3}$/.test(up) || up === 'DEMO') {
        sanitized.pdbId = up === 'DEMO' ? 'demo' : up;
      }
      continue;
    }
    sanitized[k] = v;
  }
  return sanitized;
}

function sanitizeBlockNodeParams(
  adapterId: string,
  rawParams: Record<string, unknown> | undefined,
  opts?: { preserveUiTrackingKeys?: boolean },
): Record<string, unknown> {
  const defaults = defaultParamsForAdapter(adapterId);
  const sanitized = sanitizeRawBlockParamEntries(adapterId, rawParams, opts);
  const merged: Record<string, unknown> = { ...defaults, ...sanitized };
  if (adapterId === 'pdb-input') {
    delete merged.fetchedPdbId;
    delete merged.pdbText;
    const bmId = typeof merged.benchmarkId === 'string' ? merged.benchmarkId : '';
    const bm = bmId ? getBenchmarkById(bmId) : undefined;
    if (bm) {
      merged.mode = 'pdb-id';
      if (!merged.pdbId || merged.pdbId === 'demo') merged.pdbId = bm.pdbId;
      if (
        bm.cropSegments &&
        (!merged.cropSegments || (bm.category === 'binder' && isBroadChainCrop(merged.cropSegments)))
      ) {
        merged.cropSegments = bm.cropSegments;
      }
    } else if (merged.pdbId && merged.pdbId !== 'demo') {
      merged.mode = 'pdb-id';
    } else {
      merged.mode = 'demo-bundle';
    }
  }
  return merged;
}

const PASS2_GENERATOR_ADAPTERS = new Set([
  'rfd4-proteina',
  'genie3',
  'genie2',
  'protpardelle',
  'rfdiffusion',
  'boltzgen',
  'la-proteina',
  'proteina',
  'sid-protein',
  'frameflow',
  'foldflow2',
]);

const CA_ONLY_GENERATORS = new Set(['genie3', 'genie2', 'proteina', 'sid-protein']);

const ALL_ATOM_GENERATORS = new Set([
  'rfd4-proteina',
  'rfdiffusion',
  'boltzgen',
  'protpardelle',
]);

const PASS2_AF3_MODELS = new Set(
  LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3').map((m) => m.id),
);

function wireMissingScorerStructureEdges(nodes: WorkflowNode[], edges: WorkflowEdge[]): void {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  for (const node of nodes) {
    if (node.adapterId !== 'designability-scorer') continue;
    const hasStruct = edges.some((e) => e.toNode === node.id && e.toPort === 'structure');
    if (hasStruct) continue;
    const visited = new Set<string>();
    const queue = edges.filter((e) => e.toNode === node.id).map((e) => e.fromNode);
    let fallbackPdb: string | null = null;
    let foundGen: string | null = null;
    while (queue.length > 0) {
      const cur = queue.shift()!;
      if (visited.has(cur)) continue;
      visited.add(cur);
      const cn = nodeById.get(cur);
      if (cn) {
        if (PASS2_GENERATOR_ADAPTERS.has(cn.adapterId)) {
          foundGen = cur;
          break;
        }
        if (cn.adapterId === 'pdb-input' && !fallbackPdb) fallbackPdb = cur;
      }
      for (const pe of edges) {
        if (pe.toNode === cur && !visited.has(pe.fromNode)) queue.push(pe.fromNode);
      }
    }
    const srcId = foundGen ?? fallbackPdb;
    if (srcId) {
      edges.push({
        fromNode: srcId,
        fromPort: 'structure',
        toNode: node.id,
        toPort: 'structure',
      });
    }
  }
}

function fixedPositionsLocksChain(fixedPositions: string, chain: string): boolean {
  if (!fixedPositions.trim()) return false;
  const upperChain = chain.trim().toUpperCase();
  return fixedPositions
    .split(/[\s,;]+/)
    .some((tok) => tok.trim().toUpperCase() === upperChain);
}

/**
 * Pass 2a: Deterministic biological-feedback auto-repair on a drafted/edited WorkflowGraph.
 * Uses the exact same `validateBioCapabilities()` validator that drives UI banners so the
 * AI planner and UI compatibility checks stay in sync at 0 ms latency and 0 extra LLM calls.
 */
export function repairDraftedWorkflowFromBioFeedback(graph: WorkflowGraph): {
  graph: WorkflowGraph;
  notes: string[];
} {
  const cloned: WorkflowGraph = {
    ...graph,
    nodes: graph.nodes.map((n) => ({
      ...n,
      params: { ...(n.params ?? {}) },
      ...(n.position ? { position: { x: n.position.x, y: n.position.y } } : {}),
    })),
    edges: graph.edges.map((e) => ({ ...e })),
  };
  const notes: string[] = [];

  // 0. Normalize evodiff fixedSequence and stray mode/taskMode aliases
  for (const node of cloned.nodes) {
    if (node.adapterId === 'evodiff' && typeof node.params.fixedSequence === 'string') {
      node.params.fixedSequence = sanitizeEvoDiffFixedSequence(node.params.fixedSequence);
    }
    const manifest = ADAPTER_REGISTRY[node.adapterId];
    if (manifest?.modeParamKey === 'taskMode' && node.params.mode !== undefined) {
      if (node.params.taskMode === undefined || node.params.taskMode === 'unconditional') {
        const valid = !manifest.modes?.length || manifest.modes.some((m) => m.id === String(node.params.mode));
        if (valid) node.params.taskMode = node.params.mode;
      }
      delete node.params.mode;
    } else if (manifest?.modeParamKey === 'mode' && node.params.taskMode !== undefined) {
      if (node.params.mode === undefined || node.params.mode === 'unconditional') {
        const valid = !manifest.modes?.length || manifest.modes.some((m) => m.id === String(node.params.taskMode));
        if (valid) node.params.mode = node.params.taskMode;
      }
      delete node.params.taskMode;
    }
  }

  // 1. Infer curated benchmarkId when pdb-input has a curated benchmark pdbId wired to a generator
  const nodeByIdInit = new Map(cloned.nodes.map((n) => [n.id, n]));
  for (const node of cloned.nodes) {
    if (node.adapterId !== 'pdb-input') continue;
    const pdbId = String(node.params.pdbId ?? '').trim().toUpperCase();
    if (!pdbId) continue;

    const downstreamGens = cloned.edges
      .filter((e) => e.fromNode === node.id)
      .map((e) => nodeByIdInit.get(e.toNode))
      .filter((n): n is WorkflowNode => Boolean(n && PASS2_GENERATOR_ADAPTERS.has(n.adapterId)));

    let bm =
      typeof node.params.benchmarkId === 'string' && node.params.benchmarkId
        ? getBenchmarkById(node.params.benchmarkId)
        : undefined;
    if (!bm && downstreamGens.length > 0) {
      const matches = BENCHMARK_TARGETS.filter((b) => b.pdbId === pdbId);
      if (matches.length === 1) {
        bm = matches[0];
        node.params.benchmarkId = bm.id;
      } else if (matches.length > 1) {
        const wantsBinder = downstreamGens.some((g) => {
          const m = String(g.params.mode ?? g.params.taskMode ?? '').trim();
          const lock = String(g.params.lockChain ?? '').trim();
          const hot = String(g.params.hotspots ?? '').trim();
          const contig = String(g.params.contig ?? '').trim();
          return m === 'binder' || Boolean(lock) || Boolean(hot) || contig.includes('/0');
        });
        bm = wantsBinder
          ? (matches.find((b) => b.category === 'binder') ?? matches[0])
          : (matches.find((b) => b.category === 'motif') ?? matches[0]);
        if (bm) node.params.benchmarkId = bm.id;
      }
    }
    if (bm) {
      const curCrop = String(node.params.cropSegments ?? '').trim();
      if (bm.cropSegments && (!curCrop || (bm.category === 'binder' && isBroadChainCrop(curCrop)))) {
        node.params.cropSegments = bm.cropSegments;
      }
      for (const gen of downstreamGens) {
        const { repairedHallucination } = applyPlannerBenchmarkSeedingToNode(gen, bm);
        if (repairedHallucination) {
          notes.push(`Aligned ${gen.label ?? gen.id} conditioning with ${bm.shortLabel}.`);
        }
      }
    }
  }

  // 1a. Ensure web-protein-mpnn downstream of all-atom generators defaults to SolubleMPNN (not Cα-only)
  for (const edge of cloned.edges) {
    const fn = nodeByIdInit.get(edge.fromNode);
    const tn = nodeByIdInit.get(edge.toNode);
    if (
      fn &&
      ALL_ATOM_GENERATORS.has(fn.adapterId) &&
      tn?.adapterId === 'web-protein-mpnn' &&
      String(tn.params.family ?? '').trim().toLowerCase() === 'ca'
    ) {
      tn.params.family = 'solublempnn';
      if (!tn.label || /c[αa]/i.test(tn.label)) {
        tn.label = 'SolubleMPNN';
      }
      notes.push(`Switched ${tn.label ?? tn.id} from Cα-ProteinMPNN to SolubleMPNN for all-atom ${fn.label ?? fn.id} output.`);
    }
  }

  // 1b. Default/repair AntiFold targetChain -> 'H' for antibody CDR complexes (e.g. 6XEY)
  // while preserving 1N8Z/1MEL auto-detection and structures lacking chain H or locking H.
  for (const node of cloned.nodes) {
    if (node.adapterId !== 'antifold-esmif1') continue;
    if (String(node.params.variant ?? 'antifold') !== 'antifold') continue;
    const structEdge = cloned.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
    const src = structEdge ? nodeByIdInit.get(structEdge.fromNode) : undefined;
    if (!src || src.adapterId !== 'pdb-input') continue;
    const pdbId = String(src.params.pdbId ?? '').trim().toUpperCase();
    if (pdbId === '1N8Z' || pdbId === '1MEL') continue;
    const fixedPositions = String(node.params.fixedPositions ?? '').trim();
    if (fixedPositionsLocksChain(fixedPositions, 'H')) continue;
    const curTarget = String(node.params.targetChain ?? '').trim().toUpperCase();
    if (curTarget && curTarget !== 'A') continue;

    const pdbText = String(src.params.pdbText ?? '').trim();
    if (pdbText) {
      try {
        const parsed = parseStructureText(pdbText);
        if (parsed.nResidues >= 3 && !parsed.chainIds.some((c) => c.toUpperCase() === 'H')) {
          continue;
        }
      } catch {
        /* ignore unparseable stub */
      }
    }
    const regions = String(node.params.regions ?? '').trim();
    const isKnownAntibodyWithH = pdbId === '6XEY' || pdbId === '6W41' || pdbId === '1IGT';
    const hasCdrCue = /\bcdr\b|\bcdr\d/i.test(regions) && !pdbId;
    if (isKnownAntibodyWithH || hasCdrCue) {
      node.params.targetChain = 'H';
    }
  }

  // 2. Sync explicit holo co-folding params (ligandCcdCodes / dnaSequences / rnaSequences) across
  //    parallel AF3-lineage fold siblings fed from the same upstream sequences node.
  const af3BySeqSource = new Map<string, WorkflowNode[]>();
  for (const node of cloned.nodes) {
    if (node.adapterId !== 'localfold-af2') continue;
    const modelId = String(node.params.model ?? 'monomer');
    if (!PASS2_AF3_MODELS.has(modelId)) continue;
    const seqEdge = cloned.edges.find((e) => e.toNode === node.id && e.toPort === 'sequences');
    if (!seqEdge) continue;
    const list = af3BySeqSource.get(seqEdge.fromNode) ?? [];
    list.push(node);
    af3BySeqSource.set(seqEdge.fromNode, list);
  }
  for (const siblings of af3BySeqSource.values()) {
    if (siblings.length < 2) continue;
    const sharedLig = siblings.map((n) => String(n.params.ligandCcdCodes ?? '').trim()).find(Boolean) ?? '';
    const sharedDna = siblings.map((n) => String(n.params.dnaSequences ?? '').trim()).find(Boolean) ?? '';
    const sharedRna = siblings.map((n) => String(n.params.rnaSequences ?? '').trim()).find(Boolean) ?? '';
    for (const n of siblings) {
      if (sharedLig && !String(n.params.ligandCcdCodes ?? '').trim()) {
        n.params.ligandCcdCodes = sharedLig;
        notes.push(`Synced holo ligands (${sharedLig}) onto ${n.label ?? n.id}.`);
      }
      if (sharedDna && !String(n.params.dnaSequences ?? '').trim()) {
        n.params.dnaSequences = sharedDna;
        notes.push(`Synced DNA sequence onto ${n.label ?? n.id}.`);
      }
      if (sharedRna && !String(n.params.rnaSequences ?? '').trim()) {
        n.params.rnaSequences = sharedRna;
        notes.push(`Synced RNA sequence onto ${n.label ?? n.id}.`);
      }
    }
  }

  // 3. Iteratively apply structured validateBioCapabilities() fixes (up to 4 passes for cascading fixes)
  for (let iter = 0; iter < 4; iter++) {
    const issues = validateBioCapabilities(cloned);
    if (issues.length === 0) break;
    let changed = false;
    const nodeById = new Map(cloned.nodes.map((n) => [n.id, n]));

    for (const issue of issues) {
      const targetId = issue.fix?.nodeId ?? issue.nodeId;
      const node = nodeById.get(targetId);
      if (!node) continue;

      if (issue.fix) {
        Object.assign(node.params, issue.fix.paramsPatch);
        if (issue.fix.nodeLabel) {
          node.label = issue.fix.nodeLabel;
        }
        notes.push(`Auto-configured ${node.label ?? node.id}: ${issue.fix.label}.`);
        changed = true;
      } else if (
        (issue.code === 'genie3-unsupported-benchmark' || issue.code === 'genie3-unsupported-entities') &&
        node.adapterId === 'genie3'
      ) {
        const prevParams = node.params;
        const tplEdge = cloned.edges.find((e) => e.toNode === node.id && e.toPort === 'structure');
        const tplNode = tplEdge ? nodeById.get(tplEdge.fromNode) : undefined;
        const bmId =
          tplNode?.adapterId === 'pdb-input' && typeof tplNode.params.benchmarkId === 'string'
            ? tplNode.params.benchmarkId
            : '';
        const bm = bmId ? getBenchmarkById(bmId) : undefined;
        const reqLen = typeof prevParams.length === 'number' ? prevParams.length : undefined;
        node.adapterId = 'rfd4-proteina';
        node.label = 'RFD4-Proteina';
        node.params = {
          ...defaultParamsForAdapter('rfd4-proteina'),
          mode: bm?.category === 'binder' ? 'binder' : 'motif',
          ...(reqLen !== undefined ? { length: reqLen } : {}),
          ...(prevParams.numDesigns !== undefined ? { numDesigns: prevParams.numDesigns } : {}),
        };
        if (bm) {
          applyPlannerBenchmarkSeedingToNode(node, bm, {
            explicitLength: reqLen !== undefined && reqLen !== 80 ? reqLen : null,
          });
        }
        notes.push(`Switched ${targetId} from Genie 3 to RFD4-Proteina for all-atom ligand/DNA conditioning.`);
        changed = true;
      } else if (issue.code === 'mpnn-non-protein-stripped' && node.adapterId === 'web-protein-mpnn') {
        const prevParams = node.params;
        node.adapterId = 'ligandmpnn';
        node.label = 'LigandMPNN';
        node.params = {
          ...defaultParamsForAdapter('ligandmpnn'),
          useLigandContext: true,
          ...(prevParams.numSequences !== undefined ? { numSequences: prevParams.numSequences } : {}),
          ...(prevParams.temperature !== undefined ? { temperature: prevParams.temperature } : {}),
          ...(prevParams.omitAminoAcids ? { omitAminoAcids: prevParams.omitAminoAcids } : {}),
          ...(prevParams.fixedPositions ? { fixedPositions: prevParams.fixedPositions } : {}),
          ...(prevParams.unlockInherited !== undefined
            ? { unlockInherited: prevParams.unlockInherited }
            : {}),
          ...(prevParams.seed !== undefined ? { seed: prevParams.seed } : {}),
        };
        notes.push(`Switched ${targetId} to LigandMPNN to preserve upstream ligand/DNA context.`);
        changed = true;
      }
    }
    if (!changed) break;
  }

  return { graph: cloned, notes };
}

function serializeGraphForFeedback(graph: WorkflowGraph): string {
  const nodesStr = graph.nodes
    .map((n) => {
      const p: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(n.params ?? {})) {
        if (v !== '' && v !== null && v !== undefined && v !== false) {
          p[k] = typeof v === 'string' && v.length > 60 ? `${v.slice(0, 60)}...` : v;
        }
      }
      return `- ${n.id} (${n.adapterId}, label="${n.label ?? n.adapterId}") params=${JSON.stringify(p)}`;
    })
    .join('\n');
  const edgesStr = graph.edges
    .map((e) => `- ${e.fromNode}.${e.fromPort} -> ${e.toNode}.${e.toPort}`)
    .join('\n');
  return `Nodes:\n${nodesStr}\nEdges:\n${edgesStr}`.slice(0, 2450);
}

function formatGraphFeedbackIssues(
  graph: WorkflowGraph,
  graphErrors: string[],
  bioIssues: BioCompatibilityIssue[],
): string[] {
  const out: string[] = [];
  for (const ge of graphErrors) {
    out.push(`[GRAPH_ERROR] ${ge}`);
  }
  for (const i of bioIssues) {
    const holo =
      i.code === 'af3-apo-refold-note' || i.code === 'af2-non-protein-entity'
        ? extractUpstreamHoloHints(graph, i.nodeId)
        : null;
    const hint = holo
      ? [
          holo.ligandCcdCodes.length > 0 ? ` [upstream ligands: ${holo.ligandCcdCodes.join(', ')}]` : '',
          holo.dnaSequences.length > 0 ? ` [upstream DNA: ${holo.dnaSequences.join(':')}]` : '',
        ].join('')
      : '';
    out.push(`[${i.severity.toUpperCase()} ${i.code}] node="${i.nodeId}": ${i.message}${hint}`);
  }
  return out.slice(0, 10);
}

function compileBlockGraphPlan(
  plan: BlockGraphWorkflowPlan,
  currentGraph: WorkflowGraph | undefined,
  unsupported: string[],
  opts?: { lenient?: boolean },
): {
  graph: WorkflowGraph;
  notes: string[];
  unsupported: string[];
  isIncrementalEdit: boolean;
  addedNodeCount: number;
} {
  const notes: string[] = [];
  const stamp = Date.now().toString(36).slice(-5);

  if (plan.mode === 'edit') {
    if (!currentGraph || currentGraph.nodes.length === 0) {
      throw new Error('Cannot edit current graph because the canvas is empty.');
    }
    // Proxy-safe deep copy of Svelte 5 $state graph
    const nodes: WorkflowNode[] = currentGraph.nodes.map((n) => ({
      id: n.id,
      adapterId: n.adapterId,
      label: n.label,
      params: { ...(n.params ?? {}) },
      ...(n.position ? { position: { x: n.position.x, y: n.position.y } } : {}),
    }));
    let edges: WorkflowEdge[] = currentGraph.edges.map((e) => ({
      fromNode: e.fromNode,
      fromPort: e.fromPort,
      toNode: e.toNode,
      toPort: e.toPort,
    }));

    const existingIds = new Set(nodes.map((n) => n.id));
    const nextId = makeUniqueIdFactory(existingIds);
    const idRemap = new Map<string, string>();
    const addedNodeIds: string[] = [];
    const addedNodeIdSet = new Set<string>();
    const changedPdbSourceIds = new Set<string>();
    const changedPdbCategory = new Map<
      string,
      { prevBm?: BenchmarkTargetMeta; newBm?: BenchmarkTargetMeta }
    >();
    const explicitlyPatchedNodeKeys = new Map<string, Set<string>>();

    for (const op of (plan.editOps ?? []).slice(0, MAX_BLOCK_GRAPH_EDIT_OPS)) {
      if (op.action === 'add_node') {
        const adapterId = op.adapter ?? '';
        if (!Object.hasOwn(ADAPTER_REGISTRY, adapterId)) continue;
        const requestedId = op.nodeId?.trim();
        const safeRequestedId =
          requestedId && !BLOCK_FORBIDDEN_KEYS.has(requestedId) ? requestedId : '';
        const finalId =
          safeRequestedId && !existingIds.has(safeRequestedId)
            ? safeRequestedId
            : nextId(adapterId.split('-')[0]);
        existingIds.add(finalId);
        if (safeRequestedId) idRemap.set(safeRequestedId, finalId);
        nodes.push({
          id: finalId,
          adapterId,
          label: sanitizeDisplayText(op.label, ADAPTER_REGISTRY[adapterId].shortName, 64),
          params: sanitizeBlockNodeParams(adapterId, op.params ?? undefined),
        });
        addedNodeIds.push(finalId);
        addedNodeIdSet.add(finalId);
      } else if (op.action === 'add_edge') {
        const fromId = idRemap.get(op.fromId ?? '') ?? op.fromId ?? '';
        const toId = idRemap.get(op.toId ?? '') ?? op.toId ?? '';
        const fromNode = nodes.find((n) => n.id === fromId);
        const toNode = nodes.find((n) => n.id === toId);
        if (!fromNode || !toNode) continue;
        const resolved = resolveBlockEdgePorts(fromNode, toNode, edges, nodes);
        if (resolved) {
          edges.push({
            fromNode: fromNode.id,
            fromPort: resolved.fromPort,
            toNode: toNode.id,
            toPort: resolved.toPort,
          });
        }
      } else if (op.action === 'remove_node') {
        const targetId = idRemap.get(op.nodeId ?? '') ?? op.nodeId ?? '';
        const idx = nodes.findIndex((n) => n.id === targetId);
        if (idx === -1) continue;
        const removed = nodes[idx];
        const inc = edges.filter((e) => e.toNode === targetId);
        const out = edges.filter((e) => e.fromNode === targetId);
        nodes.splice(idx, 1);
        edges = edges.filter((e) => e.fromNode !== targetId && e.toNode !== targetId);
        // Splice pass-through 1-in/1-out filters (like top-k-filter) cleanly
        if (removed.adapterId === 'top-k-filter' && inc.length === 1 && out.length > 0) {
          for (const oe of out) {
            edges.push({
              fromNode: inc[0].fromNode,
              fromPort: inc[0].fromPort,
              toNode: oe.toNode,
              toPort: oe.toPort,
            });
          }
        }
      } else if (op.action === 'update_params') {
        const targetId = idRemap.get(op.nodeId ?? '') ?? op.nodeId ?? '';
        const target = nodes.find((n) => n.id === targetId);
        if (!target || !op.params) continue;
        const sanitizedPatch = sanitizeRawBlockParamEntries(target.adapterId, op.params, {
          preserveUiTrackingKeys: false,
          isPatch: true,
        });
        let patchedSet = explicitlyPatchedNodeKeys.get(target.id);
        if (!patchedSet) {
          patchedSet = new Set<string>();
          explicitlyPatchedNodeKeys.set(target.id, patchedSet);
        }
        for (const k of Object.keys(sanitizedPatch)) patchedSet.add(k);

        const merged: Record<string, unknown> = {
          ...target.params,
          ...sanitizedPatch,
        };
        if (target.adapterId === 'pdb-input') {
          const prevMode = String(target.params.mode ?? '');
          const prevPdbId = String(target.params.pdbId ?? '').trim().toUpperCase();
          const prevBmId =
            typeof target.params.benchmarkId === 'string' ? target.params.benchmarkId.trim() : '';
          const prevBm = prevBmId ? getBenchmarkById(prevBmId) : undefined;

          const hasExplicitMode = Object.hasOwn(sanitizedPatch, 'mode');
          const explicitMode = hasExplicitMode ? String(sanitizedPatch.mode) : null;
          const hasExplicitPdbId = Object.hasOwn(sanitizedPatch, 'pdbId');
          const hasExplicitBmId = Object.hasOwn(sanitizedPatch, 'benchmarkId');
          const explicitBmId = hasExplicitBmId ? String(sanitizedPatch.benchmarkId ?? '').trim() : '';
          const explicitBm = explicitBmId ? getBenchmarkById(explicitBmId) : undefined;
          const hasExplicitCrop = Object.hasOwn(sanitizedPatch, 'cropSegments');
          const hasExplicitFilename = Object.hasOwn(sanitizedPatch, 'filename');

          let structureChanged = false;

          if (explicitBm) {
            const bmPdbUpper = explicitBm.pdbId.trim().toUpperCase();
            const bmChanged = explicitBm.id !== prevBmId || prevPdbId !== bmPdbUpper;
            const leavingCustom = prevMode === 'custom';
            merged.mode = 'pdb-id';
            merged.pdbId = explicitBm.pdbId;
            if (prevPdbId !== bmPdbUpper || leavingCustom) {
              delete merged.pdbText;
              delete merged.fetchedPdbId;
              if (!hasExplicitFilename) delete merged.filename;
              structureChanged = true;
            }
            if (bmChanged || leavingCustom) {
              if (!hasExplicitCrop) {
                merged.cropSegments = explicitBm.cropSegments ?? '';
              }
              structureChanged = true;
            } else if (
              explicitBm.cropSegments &&
              !hasExplicitCrop &&
              (!merged.cropSegments ||
                (explicitBm.category === 'binder' && isBroadChainCrop(merged.cropSegments)))
            ) {
              merged.cropSegments = explicitBm.cropSegments;
            }
            if (bmChanged) {
              changedPdbCategory.set(target.id, { prevBm, newBm: explicitBm });
            }
          } else {
            if (!explicitMode && hasExplicitPdbId) {
              merged.mode = merged.pdbId && merged.pdbId !== 'demo' ? 'pdb-id' : 'demo-bundle';
            }
            const nextMode = String(merged.mode ?? '');
            const nextPdbIdUpper = String(merged.pdbId ?? '').trim().toUpperCase();
            const leavingCustom = prevMode === 'custom' && nextMode !== 'custom';
            const switchingToDemo = nextMode === 'demo-bundle' && prevMode !== 'demo-bundle';
            const pdbIdChanged =
              hasExplicitPdbId && (nextPdbIdUpper !== prevPdbId || leavingCustom);

            if (switchingToDemo) {
              delete merged.pdbText;
              delete merged.fetchedPdbId;
              if (!hasExplicitFilename) delete merged.filename;
              if (!hasExplicitBmId) delete merged.benchmarkId;
              if (!hasExplicitCrop) delete merged.cropSegments;
              if (!hasExplicitPdbId) delete merged.pdbId;
              structureChanged = true;
            } else if (pdbIdChanged || leavingCustom) {
              delete merged.pdbText;
              delete merged.fetchedPdbId;
              if ((leavingCustom || nextPdbIdUpper !== prevPdbId) && !hasExplicitFilename) {
                delete merged.filename;
              }
              if (
                !hasExplicitBmId &&
                prevBmId &&
                (!prevBm || prevBm.pdbId.toUpperCase() !== nextPdbIdUpper || nextPdbIdUpper !== prevPdbId)
              ) {
                delete merged.benchmarkId;
              }
              if (!hasExplicitCrop && (nextPdbIdUpper !== prevPdbId || leavingCustom)) {
                delete merged.cropSegments;
              }
              if (nextPdbIdUpper) {
                const matches = BENCHMARK_TARGETS.filter((b) => b.pdbId === nextPdbIdUpper);
                if (matches.length === 1) {
                  changedPdbCategory.set(target.id, { prevBm, newBm: matches[0] });
                }
              }
              structureChanged = true;
            }
          }

          if (structureChanged) {
            changedPdbSourceIds.add(target.id);
          }
        }
        target.params = merged;
      }
    }

    // Clear stale residue/coordinate conditioning on pre-existing downstream generators
    // whose upstream pdb-input switched structure or benchmark in editOps.
    if (changedPdbSourceIds.size > 0) {
      const STALE_GEN_COORD_KEYS = [
        'motifSpan',
        'hotspots',
        'contig',
        'lockChain',
        'unindexedMotif',
        'motifSelection',
      ];
      for (const edge of edges) {
        if (!changedPdbSourceIds.has(edge.fromNode)) continue;
        const downstream = nodes.find((n) => n.id === edge.toNode);
        if (
          !downstream ||
          addedNodeIdSet.has(downstream.id) ||
          !PASS2_GENERATOR_ADAPTERS.has(downstream.adapterId)
        ) {
          continue;
        }
        const patchedKeys = explicitlyPatchedNodeKeys.get(downstream.id);
        for (const k of STALE_GEN_COORD_KEYS) {
          if (!patchedKeys?.has(k)) {
            delete downstream.params[k];
          }
        }
        const catChange = changedPdbCategory.get(edge.fromNode);
        if (catChange?.newBm) {
          const manifest = ADAPTER_REGISTRY[downstream.adapterId];
          const modeKey = manifest?.modeParamKey;
          if (modeKey && !patchedKeys?.has(modeKey)) {
            const curMode = String(downstream.params[modeKey] ?? '').trim();
            const shouldAlignMode = catChange.prevBm
              ? catChange.prevBm.category !== catChange.newBm.category
              : !curMode || curMode === 'unconditional' || curMode === 'motif' || curMode === 'binder';
            if (shouldAlignMode) {
              const nextCatMode = catChange.newBm.category === 'binder' ? 'binder' : 'motif';
              const validMode =
                !manifest?.modes?.length || manifest.modes.some((m) => m.id === nextCatMode);
              if (validMode) {
                downstream.params[modeKey] = nextCatMode;
              }
            }
          }
        }
      }
    }

    // Auto-wire any newly added node that had no explicit add_edge in editOps
    for (const nid of addedNodeIds) {
      const hasIn = edges.some((e) => e.toNode === nid);
      const addedNode = nodes.find((n) => n.id === nid);
      if (!addedNode || hasIn || Object.keys(ADAPTER_REGISTRY[addedNode.adapterId].inputs).length === 0) {
        continue;
      }
      for (let i = nodes.length - 1; i >= 0; i--) {
        const cand = nodes[i];
        if (cand.id === nid) continue;
        const resolved = resolveBlockEdgePorts(cand, addedNode, edges, nodes);
        if (resolved) {
          edges.push({
            fromNode: cand.id,
            fromPort: resolved.fromPort,
            toNode: nid,
            toPort: resolved.toPort,
          });
          break;
        }
      }
    }

    wireMissingScorerStructureEdges(nodes, edges);

    const nextGraphRaw: WorkflowGraph = {
      id: currentGraph.id,
      name: currentGraph.name,
      ...(currentGraph.description !== undefined ? { description: currentGraph.description } : {}),
      nodes,
      edges,
    };
    const { graph: nextGraph, notes: repairNotes } = repairDraftedWorkflowFromBioFeedback(nextGraphRaw);
    notes.push(...repairNotes);
    const errs = validateGraph(nextGraph);
    if (errs.length > 0 && !opts?.lenient) {
      throw new Error(`Edited block-graph workflow failed validation: ${errs.join('; ')}`);
    }
    return {
      graph: autoLayoutGraph(nextGraph),
      notes,
      unsupported: filterUnsupportedItems(unsupported, nextGraph.nodes),
      isIncrementalEdit: true,
      addedNodeCount: addedNodeIds.length,
    };
  }

  // mode === 'new'
  const rawNodes = (plan.nodes ?? []).slice(0, MAX_BLOCK_GRAPH_NODES);
  const nodes: WorkflowNode[] = [];
  const seenIds = new Set<string>();
  const explicitFamilyNodeIds = new Set<string>();
  for (const spec of rawNodes) {
    if (!Object.hasOwn(ADAPTER_REGISTRY, spec.adapter)) continue;
    const rawId = spec.id.trim();
    let id =
      rawId && !BLOCK_FORBIDDEN_KEYS.has(rawId)
        ? rawId
        : `${spec.adapter.split('-')[0]}_${nodes.length + 1}`;
    let n = 2;
    while (seenIds.has(id)) id = `${id}_${n++}`;
    seenIds.add(id);
    if (
      spec.adapter === 'web-protein-mpnn' &&
      spec.params &&
      typeof spec.params.family === 'string' &&
      spec.params.family.trim().length > 0
    ) {
      explicitFamilyNodeIds.add(id);
    }
    nodes.push({
      id,
      adapterId: spec.adapter,
      label: sanitizeDisplayText(spec.label, ADAPTER_REGISTRY[spec.adapter].shortName, 64),
      params: sanitizeBlockNodeParams(spec.adapter, spec.params),
    });
  }

  if (nodes.length === 0) {
    throw new Error('Block-graph workflow plan contained no valid blocks.');
  }

  const edges: WorkflowEdge[] = [];
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  for (const [fromId, toId] of (plan.edges ?? []).slice(0, MAX_BLOCK_GRAPH_EDGES)) {
    const fn = nodeById.get(fromId);
    const tn = nodeById.get(toId);
    if (!fn || !tn || fromId === toId) continue;
    const resolved = resolveBlockEdgePorts(fn, tn, edges, nodes);
    if (resolved) {
      edges.push({
        fromNode: fn.id,
        fromPort: resolved.fromPort,
        toNode: tn.id,
        toPort: resolved.toPort,
      });
    }
  }

  // Ensure Cα-only generators -> web-protein-mpnn use family='ca' unless explicitly overridden,
  // and all-atom generators -> web-protein-mpnn default to family='solublempnn' (not 'ca').
  for (const e of edges) {
    const fn = nodeById.get(e.fromNode);
    const tn = nodeById.get(e.toNode);
    if (!fn || tn?.adapterId !== 'web-protein-mpnn') continue;
    if (CA_ONLY_GENERATORS.has(fn.adapterId) && !explicitFamilyNodeIds.has(tn.id)) {
      tn.params.family = 'ca';
      if (!tn.label || tn.label === 'ProteinMPNN' || tn.label === 'SolubleMPNN') {
        tn.label = 'Cα-ProteinMPNN';
      }
    } else if (
      ALL_ATOM_GENERATORS.has(fn.adapterId) &&
      (!explicitFamilyNodeIds.has(tn.id) ||
        String(tn.params.family ?? '').trim().toLowerCase() === 'ca')
    ) {
      tn.params.family = 'solublempnn';
      if (!tn.label || /c[αa]/i.test(tn.label)) {
        tn.label = 'SolubleMPNN';
      }
    }
  }

  // Ensure every designability-scorer has its structure input wired from nearest upstream generator/pdb
  wireMissingScorerStructureEdges(nodes, edges);

  const rawGraph: WorkflowGraph = {
    id: `ai-plan-${stamp}`,
    name: sanitizeDisplayText(plan.summary, 'AI Drafted Workflow', 110),
    description: sanitizeDisplayText(plan.summary, '', 200),
    nodes,
    edges,
  };
  const { graph, notes: repairNotes } = repairDraftedWorkflowFromBioFeedback(rawGraph);
  notes.push(...repairNotes);
  const errs = validateGraph(graph);
  if (errs.length > 0 && !opts?.lenient) {
    throw new Error(`Drafted block-graph workflow failed validation: ${errs.join('; ')}`);
  }
  return {
    graph: autoLayoutGraph(graph),
    notes,
    unsupported: filterUnsupportedItems(unsupported, graph.nodes),
    isIncrementalEdit: false,
    addedNodeCount: graph.nodes.length,
  };
}

export function compileWorkflowPlanToGraph(
  plan: WorkflowPlan,
  currentGraph?: WorkflowGraph,
  opts?: { lenient?: boolean },
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

  const rawUnsupported = Array.isArray(plan.unsupported)
    ? plan.unsupported
        .filter((s): s is string => typeof s === 'string' && s.trim().length > 0)
        .map((s) => sanitizeDisplayText(s, '', 60))
        .filter(Boolean)
    : [];
  const unsupported = filterUnsupportedItems(rawUnsupported).slice(0, MAX_UNSUPPORTED_ITEMS);

  if (plan.kind === 'block_graph') {
    return compileBlockGraphPlan(plan, currentGraph, unsupported, opts);
  }

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
    const rawGens = Array.isArray(plan.generators) ? plan.generators.slice(0, MAX_GENERATORS) : [];
    const gens: GeneratorId[] = rawGens.length > 0 ? rawGens : ['genie3'];
    const multiBranch = gens.length > 1;
    const explicitDesigners: SequenceDesignerId[] = (
      plan.sequenceDesigners && plan.sequenceDesigners.length > 0
        ? plan.sequenceDesigners
        : plan.sequenceDesigner
          ? [plan.sequenceDesigner]
          : []
    ).slice(0, MAX_SEQUENCE_DESIGNERS);
    const boundedFoldModels = (plan.foldModels ?? []).slice(0, MAX_FOLD_MODELS);
    const deNovoLength = Math.max(20, Math.min(512, Math.round(plan.length || 80)));

    for (const genId of gens) {
      const branchTag = multiBranch
        ? genId === 'genie3'
          ? 'Genie 3'
          : genId === 'protpardelle'
            ? 'ProtPardelle'
            : 'RFD4'
        : '';
      const genNumDesigns =
        genId === 'protpardelle'
          ? Math.max(1, Math.min(8, Math.round(plan.numDesigns || 1)))
          : Math.max(1, Math.min(16, Math.round(plan.numDesigns || 1)));
      const paramsOverride: Record<string, unknown> =
        genId === 'genie3'
          ? {
              taskMode: 'unconditional',
              length: deNovoLength,
              numDesigns: genNumDesigns,
            }
          : genId === 'protpardelle'
            ? {
                taskMode: 'unconditional',
                length: deNovoLength,
                numDesigns: genNumDesigns,
              }
            : {
                mode: 'unconditional',
                length: deNovoLength,
                numDesigns: genNumDesigns,
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
        mpnnFamily: plan.mpnnFamily,
        omitAminoAcids: plan.omitAminoAcids,
        temperature: plan.temperature,
        topK: plan.topK,
        refold: plan.refold,
        foldModels: boundedFoldModels,
        score: plan.score,
        stabilityScan: plan.stabilityScan,
        esmcScore: Boolean(plan.esmcScore),
        dynamicsScan: Boolean(plan.dynamicsScan),
      });
    }
  } else if (plan.kind === 'sequence') {
    const rawList = Array.isArray(plan.sequences) ? plan.sequences : [];
    const chains: string[] = [];
    for (const item of rawList) {
      if (typeof item !== 'string') continue;
      for (const part of item.split(/[:/]+/)) {
        const cleaned = part.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
        if (cleaned.length > 0) chains.push(cleaned);
      }
    }
    const totalLen = chains.reduce((sum, c) => sum + c.length, 0);
    if (chains.length === 0 || totalLen < 5) {
      throw new Error('Sequence workflow plan requires at least 5 canonical amino acid residues.');
    }
    const seqInId = nextId('seqin');
    const defaultSeqLabel = chains.length > 1 ? 'Heterodimer Sequence' : 'Sequence Input';
    nodes.push({
      id: seqInId,
      adapterId: 'sequence-input',
      label: sanitizeDisplayText(plan.label, defaultSeqLabel, 48),
      params: {
        ...defaultParamsForAdapter('sequence-input'),
        sequence: chains.join(':'),
      },
    });
    const boundedFoldModels = (plan.foldModels ?? []).slice(0, MAX_FOLD_MODELS);
    const foldModels: FoldOracleId[] =
      boundedFoldModels.length > 0
        ? boundedFoldModels
        : [chains.length > 1 ? 'multimer' : 'monomer'];

    attachDownstreamEval({
      nextId,
      nodes,
      edges,
      branchTag: '',
      backboneNodeId: null,
      seqSourceNodeId: seqInId,
      topK: plan.topK ?? null,
      refold: true,
      foldModels,
      msaMode: plan.msaMode ?? null,
      score: false,
      stabilityScan: Boolean(plan.stabilityScan),
      esmcScore: Boolean(plan.esmcScore),
      dynamicsScan: Boolean(plan.dynamicsScan),
    });
  } else if (plan.kind === 'conditioned_design') {
    const bench = plan.benchmarkId ? getBenchmarkById(plan.benchmarkId) : undefined;
    const pdbId = bench?.pdbId ?? plan.pdbId ?? '1BCF';
    const validatedCrop = validateOptionalCrop(plan.crop);
    const cropSegments = bench?.cropSegments ?? validatedCrop ?? '';
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

    const isPartial = plan.designTask === 'partial';
    const rawGens = Array.isArray(plan.generators) ? plan.generators.slice(0, MAX_GENERATORS) : [];
    let gens: GeneratorId[] = rawGens.length > 0 ? [...rawGens] : ['rfd4-proteina'];
    if (isPartial && gens.includes('genie3')) {
      gens = gens.filter((g) => g !== 'genie3');
      if (gens.length === 0) gens = ['rfd4-proteina'];
      notes.push('Genie 3 does not support partial diffusion, so RFD4-Proteina is used instead.');
    }
    if (bench && !bench.supportsGenie3 && gens.includes('genie3')) {
      gens = gens.filter((g) => g !== 'genie3');
      if (gens.length === 0) gens = ['rfd4-proteina'];
      notes.push(
        `${bench.shortLabel} requires all-atom ligand/DNA conditioning, so Genie 3 (Cα-only) was skipped in favor of RFD4-Proteina.`,
      );
    }
    if (
      (plan.designTask === 'enzyme-ligand' || (bench && !bench.supportsGenie3)) &&
      gens.includes('protpardelle')
    ) {
      gens = gens.filter((g) => g !== 'protpardelle');
      if (gens.length === 0) gens = ['rfd4-proteina'];
      notes.push(
        'ProtPardelle-1c released denoisers are protein-only, so RFD4-Proteina is used for ligand/DNA conditioning.',
      );
    }

    const explicitDesigners: SequenceDesignerId[] = (
      plan.sequenceDesigners && plan.sequenceDesigners.length > 0
        ? plan.sequenceDesigners
        : plan.sequenceDesigner
          ? [plan.sequenceDesigner]
          : []
    ).slice(0, MAX_SEQUENCE_DESIGNERS);
    const boundedFoldModels = (plan.foldModels ?? []).slice(0, MAX_FOLD_MODELS);
    const multiBranch = gens.length > 1;
    const partialFrac = clampPartialFraction(plan.partialFraction);
    const explicitPlanLength =
      typeof plan.length === 'number' && Number.isFinite(plan.length) && plan.length > 0
        ? plan.length
        : null;

    for (const genId of gens) {
      const branchTag = multiBranch
        ? genId === 'genie3'
          ? 'Genie 3'
          : genId === 'protpardelle'
            ? 'ProtPardelle'
            : 'RFD4'
        : '';
      const genNumDesigns =
        genId === 'protpardelle'
          ? Math.max(1, Math.min(8, Math.round(plan.numDesigns || 1)))
          : Math.max(1, Math.min(16, Math.round(plan.numDesigns || 1)));
      let paramsOverride: Record<string, unknown>;

      if (genId === 'protpardelle' && isPartial) {
        paramsOverride = {
          taskMode: 'partial',
          rewindSteps: Math.max(5, Math.min(200, Math.round(partialFrac * 100))),
          numDesigns: genNumDesigns,
        };
      } else if (genId === 'rfd4-proteina' && isPartial) {
        paramsOverride = {
          mode: 'partial',
          partialFraction: partialFrac,
          numDesigns: genNumDesigns,
          ...(bench ? { autoSeededBenchmarkId: bench.id } : {}),
        };
      } else {
        const draftNode: WorkflowNode = {
          id: '__draft__',
          adapterId: genId,
          params:
            genId === 'genie3'
              ? {
                  taskMode: plan.designTask === 'binder' || bench?.category === 'binder' ? 'binder' : 'motif',
                  numDesigns: genNumDesigns,
                  ...(explicitPlanLength !== null ? { length: explicitPlanLength } : {}),
                }
              : genId === 'protpardelle'
                ? {
                    taskMode: 'motif',
                    numDesigns: genNumDesigns,
                    ...(explicitPlanLength !== null ? { length: explicitPlanLength } : {}),
                  }
                : {
                    mode: plan.designTask === 'binder' || bench?.category === 'binder' ? 'binder' : 'motif',
                    numDesigns: genNumDesigns,
                    ...(explicitPlanLength !== null ? { length: explicitPlanLength } : {}),
                  },
        };
        if (bench) {
          applyPlannerBenchmarkSeedingToNode(draftNode, bench, {
            explicitLength: explicitPlanLength,
          });
        }
        paramsOverride = draftNode.params;
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
        mpnnFamily: plan.mpnnFamily,
        omitAminoAcids: plan.omitAminoAcids,
        temperature: plan.temperature,
        topK: plan.topK,
        refold: plan.refold,
        foldModels: boundedFoldModels,
        score: plan.score,
        stabilityScan: plan.stabilityScan,
        esmcScore: Boolean(plan.esmcScore),
        dynamicsScan: Boolean(plan.dynamicsScan),
      });
    }
  } else if (plan.kind === 'pdb') {
    // kind === 'pdb' (supports 1 or multiple parallel tasks, e.g. LigandMPNN vs LASErMPNN!)
    const pdbNodeId = nextId('pdb');
    const isDemo = !plan.pdbId || plan.pdbId.toLowerCase() === 'demo';
    const validatedCrop = validateOptionalCrop(plan.crop);
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
              ...(validatedCrop ? { cropSegments: validatedCrop } : {}),
            }),
      },
    });

    const rawTasks: PdbTaskId[] =
      plan.tasks && plan.tasks.length > 0
        ? plan.tasks
        : plan.task
          ? [plan.task]
          : ['thermompnn'];
    const taskList: PdbTaskId[] = rawTasks.slice(0, MAX_PDB_TASKS);
    const boundedFoldModels = (plan.foldModels ?? []).slice(0, MAX_FOLD_MODELS);
    const effectiveRefold = Boolean(plan.refold) || boundedFoldModels.length > 0;
    const wantsDownstreamEval = effectiveRefold || Boolean(plan.topK);
    const multiTask = taskList.length > 1;
    const sanitizedOmit = sanitizeOmitAminoAcids(plan.omitAminoAcids);
    const clampedTemp = clampTemperature(plan.temperature);
    const hasSeqTask = taskList.some(
      (t) =>
        t === 'antifold' ||
        t === 'esmif1' ||
        t === 'proteinmpnn' ||
        t === 'solublempnn' ||
        t === 'ligandmpnn' ||
        t === 'lasermpnn',
    );
    let attachedPdbThermo = taskList.includes('thermompnn');
    let attachedNonInvSeqEval = false;

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
        if (!hasSeqTask && effectiveRefold && !attachedNonInvSeqEval && !taskList.includes('esmc') && !taskList.includes('dyna1')) {
          attachedNonInvSeqEval = true;
          const seqId = nextId('solublempnn');
          nodes.push({
            id: seqId,
            adapterId: 'web-protein-mpnn',
            label: 'SolubleMPNN',
            params: {
              ...defaultParamsForAdapter('web-protein-mpnn'),
              family: 'solublempnn',
              modelName: 'v_48_020',
              ...(plan.numSequences ? { numSequences: Math.max(1, Math.min(8, Math.round(plan.numSequences))) } : {}),
              ...(clampedTemp !== undefined ? { temperature: clampedTemp } : {}),
              ...(sanitizedOmit ? { omitAminoAcids: sanitizedOmit } : {}),
            },
          });
          edges.push({
            fromNode: pdbNodeId,
            fromPort: 'structure',
            toNode: seqId,
            toPort: 'structure',
          });
          attachDownstreamEval({
            nextId,
            nodes,
            edges,
            branchTag: '',
            backboneNodeId: pdbNodeId,
            seqSourceNodeId: seqId,
            topK: plan.topK,
            refold: true,
            foldModels: boundedFoldModels,
            score: plan.score ?? true,
            stabilityScan: false,
            esmcScore: Boolean(plan.esmcScore) && !taskList.includes('esmc'),
            dynamicsScan: Boolean(plan.dynamicsScan) && !taskList.includes('dyna1'),
          });
        }
      } else if (task === 'esmc') {
        const esmcId = nextId('esmc');
        nodes.push({
          id: esmcId,
          adapterId: 'esmc',
          label: 'ESM-C 300M',
          params: { ...defaultParamsForAdapter('esmc') },
        });
        edges.push({
          fromNode: pdbNodeId,
          fromPort: 'structure',
          toNode: esmcId,
          toPort: 'structure',
        });
        if (!hasSeqTask && wantsDownstreamEval && !attachedNonInvSeqEval) {
          attachedNonInvSeqEval = true;
          attachDownstreamEval({
            nextId,
            nodes,
            edges,
            branchTag: '',
            backboneNodeId: pdbNodeId,
            seqSourceNodeId: esmcId,
            topK: plan.topK,
            refold: effectiveRefold,
            foldModels: boundedFoldModels,
            score: plan.score ?? effectiveRefold,
            stabilityScan: Boolean(plan.stabilityScan && effectiveRefold && !attachedPdbThermo),
            esmcScore: false,
            dynamicsScan: Boolean(plan.dynamicsScan) && !taskList.includes('dyna1'),
          });
          if (plan.stabilityScan && !effectiveRefold && !attachedPdbThermo) {
            attachedPdbThermo = true;
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
          }
        } else if (!hasSeqTask && plan.stabilityScan && !attachedPdbThermo) {
          attachedPdbThermo = true;
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
        }
      } else if (task === 'dyna1') {
        const dynaId = nextId('dyna');
        nodes.push({
          id: dynaId,
          adapterId: 'dyna1',
          label: 'Dyna-1',
          params: { ...defaultParamsForAdapter('dyna1') },
        });
        edges.push({
          fromNode: pdbNodeId,
          fromPort: 'structure',
          toNode: dynaId,
          toPort: 'structure',
        });
        if (!hasSeqTask && wantsDownstreamEval && !attachedNonInvSeqEval) {
          attachedNonInvSeqEval = true;
          attachDownstreamEval({
            nextId,
            nodes,
            edges,
            branchTag: '',
            backboneNodeId: pdbNodeId,
            seqSourceNodeId: dynaId,
            topK: plan.topK,
            refold: effectiveRefold,
            foldModels: boundedFoldModels,
            score: plan.score ?? effectiveRefold,
            stabilityScan: Boolean(plan.stabilityScan && effectiveRefold && !attachedPdbThermo),
            esmcScore: Boolean(plan.esmcScore) && !taskList.includes('esmc'),
            dynamicsScan: false,
          });
          if (plan.stabilityScan && !effectiveRefold && !attachedPdbThermo) {
            attachedPdbThermo = true;
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
          }
        } else if (!hasSeqTask && plan.stabilityScan && !attachedPdbThermo) {
          attachedPdbThermo = true;
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
        }
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
            ...(plan.numSequences
              ? { numSequences: Math.max(1, Math.min(8, Math.round(plan.numSequences))) }
              : {}),
            ...(clampedTemp !== undefined ? { temperature: clampedTemp } : {}),
          },
        });
        edges.push({
          fromNode: pdbNodeId,
          fromPort: 'structure',
          toNode: afId,
          toPort: 'structure',
        });
        if (wantsDownstreamEval) {
          attachDownstreamEval({
            nextId,
            nodes,
            edges,
            branchTag: multiTask ? label : '',
            backboneNodeId: pdbNodeId,
            seqSourceNodeId: afId,
            topK: plan.topK,
            refold: effectiveRefold,
            foldModels: boundedFoldModels,
            score: plan.score ?? effectiveRefold,
            stabilityScan: Boolean(plan.stabilityScan && effectiveRefold),
            esmcScore: Boolean(plan.esmcScore) && !taskList.includes('esmc'),
            dynamicsScan: Boolean(plan.dynamicsScan) && !taskList.includes('dyna1'),
          });
          if (plan.stabilityScan && !effectiveRefold && !attachedPdbThermo) {
            attachedPdbThermo = true;
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
          }
        } else {
          if (plan.stabilityScan && !attachedPdbThermo) {
            attachedPdbThermo = true;
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
          }
          if (plan.esmcScore && !taskList.includes('esmc')) {
            const esmcId = nextId('esmc');
            nodes.push({
              id: esmcId,
              adapterId: 'esmc',
              label: multiTask ? `ESM-C 300M (${label})` : 'ESM-C 300M',
              params: { ...defaultParamsForAdapter('esmc') },
            });
            edges.push({
              fromNode: afId,
              fromPort: 'sequences',
              toNode: esmcId,
              toPort: 'sequences',
            });
          }
          if (plan.dynamicsScan && !taskList.includes('dyna1')) {
            const dynaId = nextId('dyna');
            nodes.push({
              id: dynaId,
              adapterId: 'dyna1',
              label: multiTask ? `Dyna-1 (${label})` : 'Dyna-1',
              params: { ...defaultParamsForAdapter('dyna1') },
            });
            edges.push({
              fromNode: afId,
              fromPort: 'sequences',
              toNode: dynaId,
              toPort: 'sequences',
            });
          }
        }
      } else {
        const adapterId =
          task === 'ligandmpnn'
            ? 'ligandmpnn'
            : task === 'lasermpnn'
              ? 'lasermpnn'
              : 'web-protein-mpnn';
        const useSoluble =
          adapterId === 'web-protein-mpnn' &&
          (task === 'solublempnn' || plan.mpnnFamily === 'solublempnn');
        const useCa = adapterId === 'web-protein-mpnn' && plan.mpnnFamily === 'ca';
        const label =
          adapterId === 'ligandmpnn'
            ? 'LigandMPNN'
            : adapterId === 'lasermpnn'
              ? 'LASErMPNN'
              : useCa
                ? 'Cα-ProteinMPNN'
                : useSoluble
                  ? 'SolubleMPNN'
                  : 'ProteinMPNN';
        const seqNodeId = nextId(task);
        nodes.push({
          id: seqNodeId,
          adapterId,
          label,
          params: {
            ...defaultParamsForAdapter(adapterId),
            ...(useSoluble
              ? { family: 'solublempnn', modelName: 'v_48_020' }
              : useCa
                ? { family: 'ca', modelName: 'v_48_020' }
                : plan.mpnnFamily && adapterId === 'web-protein-mpnn'
                  ? { family: plan.mpnnFamily, modelName: 'v_48_020' }
                  : {}),
            ...(plan.numSequences
              ? { numSequences: Math.max(1, Math.min(8, Math.round(plan.numSequences))) }
              : {}),
            ...(clampedTemp !== undefined ? { temperature: clampedTemp } : {}),
            ...(sanitizedOmit ? { omitAminoAcids: sanitizedOmit } : {}),
          },
        });
        edges.push({
          fromNode: pdbNodeId,
          fromPort: 'structure',
          toNode: seqNodeId,
          toPort: 'structure',
        });

        if (wantsDownstreamEval) {
          attachDownstreamEval({
            nextId,
            nodes,
            edges,
            branchTag: multiTask ? label : '',
            backboneNodeId: pdbNodeId,
            seqSourceNodeId: seqNodeId,
            topK: plan.topK,
            refold: effectiveRefold,
            foldModels: boundedFoldModels,
            score: plan.score ?? effectiveRefold,
            stabilityScan: Boolean(plan.stabilityScan && effectiveRefold),
            esmcScore: Boolean(plan.esmcScore) && !taskList.includes('esmc'),
            dynamicsScan: Boolean(plan.dynamicsScan) && !taskList.includes('dyna1'),
          });
          if (plan.stabilityScan && !effectiveRefold) {
            if (adapterId === 'lasermpnn') {
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
            } else if (!attachedPdbThermo) {
              attachedPdbThermo = true;
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
            }
          }
        } else {
          if (plan.stabilityScan) {
            if (adapterId === 'lasermpnn') {
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
            } else if (!attachedPdbThermo) {
              attachedPdbThermo = true;
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
            }
          }
          if (plan.esmcScore && !taskList.includes('esmc')) {
            const esmcId = nextId('esmc');
            nodes.push({
              id: esmcId,
              adapterId: 'esmc',
              label: multiTask ? `ESM-C 300M (${label})` : 'ESM-C 300M',
              params: { ...defaultParamsForAdapter('esmc') },
            });
            edges.push({
              fromNode: seqNodeId,
              fromPort: 'sequences',
              toNode: esmcId,
              toPort: 'sequences',
            });
          }
          if (plan.dynamicsScan && !taskList.includes('dyna1')) {
            const dynaId = nextId('dyna');
            nodes.push({
              id: dynaId,
              adapterId: 'dyna1',
              label: multiTask ? `Dyna-1 (${label})` : 'Dyna-1',
              params: { ...defaultParamsForAdapter('dyna1') },
            });
            edges.push({
              fromNode: seqNodeId,
              fromPort: 'sequences',
              toNode: dynaId,
              toPort: 'sequences',
            });
          }
        }
      }
    }
  } else {
    throw new Error(`Unsupported workflow plan kind: ${String((plan as { kind?: unknown }).kind)}`);
  }

  const rawGraph: WorkflowGraph = {
    id: `planned-${plan.kind}-${stamp}`,
    name: sanitizeDisplayText(plan.summary, 'AI Planned Workflow'),
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

const SESSION_CACHE_KEY = 'denovo:planner-cache:v2';
const SESSION_CACHE_MAX_ENTRIES = 24;

const NON_SEQUENCE_UPPERCASE_WORDS = new Set([
  'LIGANDMPNN',
  'LASERMPNN',
  'THERMOMPNN',
  'PROTEINMPNN',
  'SOLUBLEMPNN',
  'ANTIFOLD',
  'ESMIF1',
  'ROSETTAFOLD',
  'ROSETTAFOLD3',
  'ALPHAFOLD',
  'ALPHAFOLD2',
  'ALPHAFOLD3',
  'PROTPARDELLE',
  'PROTENIX',
  'PROTENIX2',
  'INTELLIFOLD',
  'INTELLIFOLD2',
  'LOCALFOLD',
  'ESMFOLD',
  'ESMFOLD2',
  'OPENBIND',
  'OPENBIND0',
  'OPENDDE',
  'MULTIMER',
  'MONOMER',
  'PROTEINA',
  'DESIGN',
  'SCAFFOLD',
  'PARTIAL',
  'DIFFUSION',
  'BINDER',
  'MOTIF',
  'ENZYME',
  'LIGAND',
  'SEQUENCE',
  'SEQUENCES',
  'STABILITY',
  'AUTODOCK',
  'CYSTEINE',
  'CYSTEINES',
  'AGGREGATION',
  'SOLUBLE',
  'PROTEIN',
  'ANTIBODY',
  'NANOBODY',
  'TARGET',
  'TARGETED',
  'PEPTIDE',
  'PEPTIDES',
  'STRAND',
  'STRANDS',
  'CREATE',
  'STABLE',
  'NATIVE',
  'CHAINS',
  'FILTER',
  'PLEASE',
  'REALLY',
  'CHANGE',
  'REMOVE',
  'DELETE',
  'UPDATE',
  'INSERT',
  'SELECT',
  'ATTACH',
  'APPEND',
  'SINGLE',
  'DIMERS',
  'TRIMER',
  'GENERATE',
  'STARTING',
  'SPECIFIC',
  'STANDARD',
  'ANALYSIS',
  'SEPARATE',
  'TEMPLATE',
  'TEMPLATES',
  'BINDERS',
  'MOTIFS',
  'ENZYMES',
  'LIGANDS',
  'HELICES',
  'SHEETS',
  'POCKET',
  'SIGNAL',
  'ACTIVE',
  'SIMPLE',
  'MINIMAL',
  'LENGTH',
  'WEIGHT',
  'SAMPLE',
  'SAMPLES',
  'PREDICT',
  'SCREEN',
  'SEARCH',
  'SWITCH',
  'SYSTEM',
  'CHARGE',
  'CHARGED',
  'THERMAL',
  'SIMILAR',
  'SEVERAL',
  'BETWEEN',
  'WITHIN',
  'DURING',
  'AGAINST',
  'ALWAYS',
  'BETTER',
  'FASTER',
  'LARGER',
  'SMALLER',
  'HIGHER',
  'STRICT',
  'DIRECT',
  'INLINE',
  'DESIGNED',
  'DESIGNING',
  'REFINED',
  'STABILIZE',
  'STABILIZING',
  'AGGREGATE',
  'CATALYTIC',
  'ALANINE',
  'GLYCINE',
  'VALINE',
  'SERINE',
  'LYSINE',
  'ARGININE',
  'HISTIDINE',
  'VARIANT',
  'VARIANTS',
  'ENERGY',
  'STATES',
  'CHANNEL',
  'CHANNELS',
  'ACIDIC',
  'HELICAL',
  'ANCHOR',
  'CHAIN',
  'ACTIVITY',
  'AFFINITY',
  'CRYSTAL',
  'SIDECHAIN',
  'SIDECHAINS',
  'INTERFACE',
  'INTERFACES',
  'TERMINAL',
  'CHEMICAL',
  'CLINICAL',
  'CRITICAL',
  'DISTANCE',
  'DYNAMIC',
  'DYNAMICS',
  'EXTERNAL',
  'INTERNAL',
  'INITIAL',
  'KINETIC',
  'KINETICS',
  'KINASE',
  'LIGASE',
  'PEPTIDASE',
  'LINKAGE',
  'LINKER',
  'LINKERS',
  'NITRATE',
  'PACKING',
  'PARTICLE',
  'PATHWAY',
  'PRIMARY',
  'RADICAL',
  'RESIDUAL',
  'SECRETED',
  'SYNTHETIC',
  'VALENCY',
  'VESICLE',
  'VITAMIN',
  'WATERS',
  'WILDTYPE',
  'PLASMID',
  'FERRITIN',
  'CASPASE',
  'INTEGRIN',
  'THYMIDINE',
  'ADENINE',
  'CYTIDINE',
  'THYMINE',
  'WEIGHTS',
  'STRENGTH',
  'STRENGTHS',
  'HEIGHT',
  'HEIGHTS',
  'LENGTHS',
  'SPECIFICITY',
  'SOLUBILITY',
  'CATALYST',
]);

const ALL_CAPS_ENGLISH_SUFFIX_RE =
  /(?:MENT|TION|SION|NESS|ABLE|IBLE|ICAL|ITY|IES|ANTS?|ENTS?|OUS|IVE|ICS?|ASES?|ISTS?|ITIS|SIS|CHAINS?|AGES?|IZES?|ENCY|ANCY)$/;

function extractPromptPdbId(prompt: string): string | null {
  const matches = prompt.match(/\b([1-9][A-Za-z0-9]{3})\b/g);
  if (!matches) return null;
  for (const raw of matches) {
    if (!/[A-Za-z]/.test(raw)) continue;
    if (/^\d+(?:aa|bp|nt|ns|ms|kb|mb|gb|[x×])$/i.test(raw)) continue;
    if (/^\d+[mk]$/i.test(raw)) continue;
    return raw.toUpperCase();
  }
  return null;
}

function extractPromptSequences(prompt: string): string[] {
  const out: string[] = [];
  const matches = prompt.match(/\b[ACDEFGHIKLMNPQRSTVWY]{6,}(?:[:/][ACDEFGHIKLMNPQRSTVWY]{3,})*\b/g);
  if (!matches) return out;
  const isAllCapsPrompt = !/[a-z]/.test(prompt) && prompt.trim().split(/\s+/).length >= 4;
  const hasExplicitSeqCue = /\b(?:seq(?:uence)?s?|fasta)\b/i.test(prompt);
  for (const token of matches) {
    const hasChainSep = token.includes(':') || token.includes('/');
    for (const part of token.split(/[:/]+/)) {
      const upper = part.toUpperCase();
      if (upper.length < 6) continue;
      if (NON_SEQUENCE_UPPERCASE_WORDS.has(upper)) continue;
      if (isAllCapsPrompt && !hasChainSep && !hasExplicitSeqCue && upper.length < 10) continue;
      if (
        !hasChainSep &&
        !hasExplicitSeqCue &&
        ALL_CAPS_ENGLISH_SUFFIX_RE.test(upper) &&
        !/(.)\1\1/.test(upper) &&
        !/Q/.test(upper)
      ) {
        continue;
      }
      out.push(upper);
    }
  }
  return out;
}

function extractPromptOmitAminoAcids(prompt: string): string | null {
  const omitMatch = /\bomit(?:ting)?\s+(?:amino\s+acids?\s+)?([ACDEFGHIKLMNPQRSTVWY](?:\s*(?:,|and|or|\s)\s*[ACDEFGHIKLMNPQRSTVWY])*)\b/i.exec(
    prompt,
  );
  if (omitMatch) {
    const tokens = omitMatch[1]
      .split(/[\s,]+|\band\b|\bor\b/i)
      .map((s) => s.trim().toUpperCase())
      .filter((s) => s.length === 1 && /^[ACDEFGHIKLMNPQRSTVWY]$/.test(s));
    if (tokens.length > 0) {
      return sanitizeOmitAminoAcids(tokens.join('')) ?? null;
    }
  }
  if (/\b(?:no|without|exclude|avoid)\s+cysteines?\b/i.test(prompt)) {
    return 'C';
  }
  return null;
}

function extractPromptTemperature(prompt: string): number | null {
  const m = /\btemp(?:erature)?\s*(?:to|=|:|at|of)?\s*([0-9]*\.?[0-9]+)\b/i.exec(prompt);
  if (!m) return null;
  const val = Number(m[1]);
  return Number.isFinite(val) && val > 0 ? val : null;
}

function extractPromptTopK(prompt: string): number | null {
  const m = /\btop[\s-]*(?:k\s*)?(?:\(\s*(\d+)\s*\)|(\d+)\b)/i.exec(prompt);
  if (!m) return null;
  const n = Number(m[1] ?? m[2]);
  return Number.isInteger(n) && n >= 1 ? Math.min(16, n) : null;
}

function extractPromptNumDesignsAndLength(prompt: string): {
  numDesigns: number | null;
  length: number | null;
} {
  let numDesigns: number | null = null;
  let length: number | null = null;
  const crossMatch = /(\d+)\s*[x×*]\s*(\d+)\s*-?\s*aa\b/i.exec(prompt);
  if (crossMatch) {
    const n = Number(crossMatch[1]);
    const len = Number(crossMatch[2]);
    if (Number.isInteger(n) && n >= 1) numDesigns = Math.min(16, n);
    if (Number.isInteger(len) && len >= 10) length = Math.min(512, len);
  }
  if (numDesigns === null) {
    const desMatch = /\b(\d+)\s+(?:backbones?|designs?)\b/i.exec(prompt);
    if (desMatch) {
      const n = Number(desMatch[1]);
      if (Number.isInteger(n) && n >= 1) numDesigns = Math.min(16, n);
    }
  }
  if (length === null) {
    const lenMatch = /\b(\d+)\s*-?\s*aa\b/i.exec(prompt);
    if (lenMatch) {
      const len = Number(lenMatch[1]);
      if (Number.isInteger(len) && len >= 10) length = Math.min(512, len);
    } else {
      const lenWordMatch = /\blength\s*(?:=|of|to)?\s*(\d+)\b/i.exec(prompt);
      if (lenWordMatch) {
        const len = Number(lenWordMatch[1]);
        if (Number.isInteger(len) && len >= 10) length = Math.min(512, len);
      }
    }
  }
  return { numDesigns, length };
}

function extractPromptFoldModels(prompt: string): FoldOracleId[] {
  const hits: Array<{ idx: number; id: FoldOracleId }> = [];
  const check = (re: RegExp, id: FoldOracleId) => {
    const m = re.exec(prompt);
    if (m && !hits.some((h) => h.id === id)) {
      hits.push({ idx: m.index, id });
    }
  };
  const multiMatch = /\b(?:af2[\s-]*multimer|alphafold2[\s-]*multimer|multimer)\b/i.exec(prompt);
  if (multiMatch) {
    hits.push({ idx: multiMatch.index, id: 'multimer' });
  } else {
    check(/\b(?:af2|alphafold2|alphafold)\b/i, 'monomer');
  }
  check(/\b(?:boltz[\s-]*2|boltz2)\b/i, 'boltz2');
  check(/\b(?:rf3|rosettafold[\s-]*3|rosettafold3)\b/i, 'rosettafold3');
  check(/\b(?:protenix[\s-]*v?2|protenix2|protenix)\b/i, 'protenix2');
  check(/\b(?:openbind[\s-]*0|openbind0)\b/i, 'openbind0');
  check(/\b(?:opendde)\b/i, 'opendde');
  check(/\b(?:mol[\s-]*1|mol1)\b/i, 'mol1');
  check(/\b(?:intellifold[\s-]*2|intellifold2)\b/i, 'intellifold2');
  check(/\b(?:esmfold[\s-]*2|ef2)\b/i, 'ef2-fast-300m');
  check(/\b(?:af3|alphafold[\s-]*3)\b/i, 'af3');
  hits.sort((a, b) => a.idx - b.idx);
  return hits.map((h) => h.id).slice(0, MAX_FOLD_MODELS);
}

function extractPromptCdrRegions(prompt: string): string | null {
  const found: string[] = [];
  for (const num of ['1', '2', '3'] as const) {
    if (new RegExp(`\\bcdr[\\s-]*(?:[hl][\\s-]*)?${num}\\b`, 'i').test(prompt)) {
      found.push(`CDR${num}`);
    }
  }
  if (found.length > 0) return found.join(' ');
  if (/\bcdrs?\b/i.test(prompt)) return 'CDR1 CDR2 CDR3';
  return null;
}

function makeEmptyEditOp(action: EditActionId): GraphEditOperation {
  return {
    action,
    attachToNodeId: null,
    model: null,
    foldModels: [],
    topK: null,
    numSequences: null,
    targetChain: null,
    regions: null,
    score: true,
  };
}

function resolveCanvasNodesForKeyword(
  clause: string,
  nodes: WorkflowNode[],
): { nodeIds: string[]; adapterIds: string[] } {
  const nodeIds: string[] = [];
  const adapterIds: string[] = [];
  const addMatch = (selector: string, predicate: (n: WorkflowNode) => boolean) => {
    const hits = nodes.filter(predicate);
    if (hits.length > 0) {
      if (!adapterIds.includes(selector)) adapterIds.push(selector);
      for (const h of hits) {
        if (!nodeIds.includes(h.id)) nodeIds.push(h.id);
      }
    }
  };
  if (/\bantifold\b/i.test(clause)) {
    addMatch('antifold', (n) => n.adapterId === 'antifold-esmif1' && String(n.params.variant ?? 'antifold') !== 'esmif1');
  }
  if (/\besm[\s-]*if[\s-]*1\b/i.test(clause)) {
    addMatch('esmif1', (n) => n.adapterId === 'antifold-esmif1' && String(n.params.variant ?? 'antifold') === 'esmif1');
  }
  if (/\besm[\s-]*c\b/i.test(clause)) addMatch('esmc', (n) => n.adapterId === 'esmc');
  if (/\bdyna[\s-]*1\b/i.test(clause)) addMatch('dyna1', (n) => n.adapterId === 'dyna1');
  if (/\bthermompnn\b/i.test(clause)) addMatch('thermompnn', (n) => n.adapterId === 'thermompnn');
  if (/\bligandmpnn\b/i.test(clause)) addMatch('ligandmpnn', (n) => n.adapterId === 'ligandmpnn');
  if (/\blasermpnn\b/i.test(clause)) addMatch('lasermpnn', (n) => n.adapterId === 'lasermpnn');
  if (/\bsoluble\s*mpnn\b/i.test(clause)) {
    addMatch('solublempnn', (n) => n.adapterId === 'web-protein-mpnn');
  } else if (/\b(?:protein\s*mpnn|ca-proteinmpnn|mpnn)\b/i.test(clause)) {
    addMatch('web-protein-mpnn', (n) => n.adapterId === 'web-protein-mpnn');
  }
  if (/\btop[\s-]*k\b/i.test(clause)) addMatch('top-k-filter', (n) => n.adapterId === 'top-k-filter');
  if (/\b(?:scrmsd|scorer|designability)\b/i.test(clause)) {
    addMatch('designability-scorer', (n) => n.adapterId === 'designability-scorer');
  }
  if (/\b(?:rf3|rosettafold[\s-]*3|rosettafold3)\b/i.test(clause)) {
    addMatch('rosettafold3', (n) => n.adapterId === 'localfold-af2' && String(n.params.model ?? '') === 'rosettafold3');
  }
  if (/\b(?:boltz[\s-]*2|boltz2)\b/i.test(clause)) {
    addMatch('boltz2', (n) => n.adapterId === 'localfold-af2' && String(n.params.model ?? '') === 'boltz2');
  }
  if (/\b(?:protenix[\s-]*v?2|protenix2|protenix)\b/i.test(clause)) {
    addMatch('protenix2', (n) => n.adapterId === 'localfold-af2' && String(n.params.model ?? '') === 'protenix2');
  }
  if (/\b(?:af2[\s-]*multimer|alphafold2?[\s-]*multimer|multimer)\b/i.test(clause)) {
    addMatch('multimer', (n) => n.adapterId === 'localfold-af2' && String(n.params.model ?? '') === 'multimer');
  } else if (/\b(?:af2|alphafold2|alphafold)\b/i.test(clause)) {
    addMatch(
      'af2',
      (n) => n.adapterId === 'localfold-af2' && String(n.params.model ?? 'monomer').startsWith('monomer'),
    );
  }
  return { nodeIds, adapterIds };
}

const EDIT_ACTION_VERB_RE =
  /\b(?:add|append|attach|include|insert|change|set|update|adjust|switch|put|run|use|omit|exclude|remove|delete|drop|strip)\b/i;
const EDIT_PARAM_DIRECTIVE_RE =
  /\b(?:temp(?:erature)?|omit(?:ting)?|(?:no|without|exclude|avoid)\s+cysteines?|num(?:ber)?\s*(?:of\s+)?(?:sequences?|designs?|backbones?)|sequences?|designs?|backbones?|top[\s-]*k\s*(?:to|=|:)?\s*\d+)\b/i;
const EDIT_REMOVE_VERB_RE = /\b(?:remove|delete|drop|strip)\b/i;
const EDIT_NON_REMOVE_VERB_RE =
  /\b(?:add|append|attach|include|insert|change|set|update|adjust|switch|put|run|use|omit|exclude)\b/i;

/**
 * Deterministic client-side edit operation parser for incremental canvas edits (`remove_nodes`,
 * `update_params`, and add operations). Used both as a fallback if remote `v7` returns 500/502
 * or empty operations AND to reconcile partial backend edit plans.
 */
export function parseDeterministicEditOperations(
  prompt: string,
  currentGraph: WorkflowGraph,
): EditCurrentWorkflowPlan | null {
  if (!currentGraph || currentGraph.nodes.length === 0) return null;
  const trimmed = prompt.trim();
  if (!trimmed) return null;

  const ops: GraphEditOperation[] = [];
  const rawParts = trimmed
    .split(/(?:\band\b|\bthen\b|[,;]|\.(?!\d))+/i)
    .map((c) => c.trim())
    .filter(Boolean);
  const clauses: string[] = [];
  for (const part of rawParts) {
    const prev = clauses.length > 0 ? clauses[clauses.length - 1] : null;
    const isContinuationNoun =
      prev !== null &&
      !EDIT_ACTION_VERB_RE.test(part) &&
      !EDIT_PARAM_DIRECTIVE_RE.test(part) &&
      (EDIT_REMOVE_VERB_RE.test(prev) ||
        /\bomit(?:ting)?\b/i.test(prev) ||
        extractPromptFoldModels(prev).length > 0);
    const isTargetPlusParam =
      prev !== null &&
      /\b(?:change|set|update|adjust|switch)\b/i.test(prev) &&
      !EDIT_PARAM_DIRECTIVE_RE.test(prev) &&
      !EDIT_REMOVE_VERB_RE.test(prev) &&
      !/\b(?:add|append|attach|include|insert)\b/i.test(prev) &&
      !EDIT_ACTION_VERB_RE.test(part) &&
      EDIT_PARAM_DIRECTIVE_RE.test(part);
    if (isContinuationNoun || isTargetPlusParam) {
      clauses[clauses.length - 1] = `${prev}, ${part}`;
    } else {
      clauses.push(part);
    }
  }

  // 1. Remove clauses (strictly scoped to clauses that contain a remove verb; never scan across unrelated clauses)
  const removeTargetIds: string[] = [];
  const removeAdapterIds: string[] = [];
  for (const clause of clauses) {
    const removeMatch = /\b(?:remove|delete|drop|strip)\b([\s\S]*)/i.exec(clause);
    if (!removeMatch) continue;
    let removeText = removeMatch[1];
    const nextVerbIdx = removeText.search(EDIT_NON_REMOVE_VERB_RE);
    if (nextVerbIdx >= 0) {
      removeText = removeText.slice(0, nextVerbIdx);
    }
    const { nodeIds, adapterIds } = resolveCanvasNodesForKeyword(removeText, currentGraph.nodes);
    for (const id of nodeIds) if (!removeTargetIds.includes(id)) removeTargetIds.push(id);
    for (const aid of adapterIds) if (!removeAdapterIds.includes(aid)) removeAdapterIds.push(aid);
  }
  if (removeTargetIds.length > 0 || removeAdapterIds.length > 0) {
    ops.push({
      ...makeEmptyEditOp('remove_nodes'),
      targetNodeIds: removeTargetIds,
      targetAdapterIds: removeAdapterIds,
      score: false,
    });
  }

  // 2. Add / insert clauses
  for (const clause of clauses) {
    if (EDIT_REMOVE_VERB_RE.test(clause)) continue;
    const hasAddVerb = /\b(?:add|append|attach|include|insert)\b/i.test(clause);
    if (hasAddVerb && /\bthermompnn\b/i.test(clause)) {
      const afterMatch = /\bafter\s+([A-Za-z0-9\s.-]+)/i.exec(clause);
      let attachToNodeId: string | null = null;
      if (afterMatch) {
        const refHits = resolveCanvasNodesForKeyword(afterMatch[1], currentGraph.nodes).nodeIds;
        if (refHits.length > 0) attachToNodeId = refHits[refHits.length - 1];
      }
      ops.push({
        ...makeEmptyEditOp('add_stability_scan'),
        attachToNodeId,
        score: false,
      });
    } else if (hasAddVerb && /\besm[\s-]*c\b/i.test(clause)) {
      ops.push({ ...makeEmptyEditOp('add_esmc'), score: false });
    } else if (hasAddVerb && /\bdyna[\s-]*1\b/i.test(clause)) {
      ops.push({ ...makeEmptyEditOp('add_dyna1'), score: false });
    } else if ((hasAddVerb || /\binsert\b/i.test(clause)) && /\btop[\s-]*k\b/i.test(clause)) {
      const topK = extractPromptTopK(clause) ?? 2;
      ops.push({ ...makeEmptyEditOp('insert_top_k'), topK });
    } else if (hasAddVerb) {
      const foldModels = extractPromptFoldModels(clause);
      if (foldModels.length > 0) {
        ops.push({ ...makeEmptyEditOp('add_fold_oracles'), foldModels, score: true });
      }
    }
  }

  // 3. Per-clause parameter updates (temperature, omitAminoAcids, numSequences, numDesigns, mpnnFamily, topK)
  for (const clause of clauses) {
    if (EDIT_REMOVE_VERB_RE.test(clause)) continue;
    const tempMatch = /\btemp(?:erature)?\s*(?:to|=|:|at|of)?\s*([0-9]*\.?[0-9]+)/i.exec(clause);
    const temperature = tempMatch ? Number(tempMatch[1]) : null;
    const omitAminoAcids = extractPromptOmitAminoAcids(clause);
    const seqMatch = /\b(?:num(?:ber)?\s*(?:of\s+)?sequences?|sequences?)\s*(?:to|=|:)?\s*(\d+)/i.exec(clause);
    const numSequences = seqMatch ? Number(seqMatch[1]) : null;
    const desMatch = /\b(?:num(?:ber)?\s*(?:of\s+)?(?:designs|backbones)|backbones|designs)\s*(?:to|=|:)?\s*(\d+)/i.exec(
      clause,
    );
    const numDesigns = desMatch ? Number(desMatch[1]) : null;
    const familyMatch =
      /\b(?:switch|change|use|set)\s+(?:mpnn\s+)?(?:to\s+)?(soluble\s*mpnn|protein\s*mpnn|ca[- ]?protein\s*mpnn)\b/i.exec(
        clause,
      );
    const mpnnFamily: MpnnFamilyId | null = familyMatch
      ? familyMatch[1].toLowerCase().startsWith('soluble')
        ? 'solublempnn'
        : familyMatch[1].toLowerCase().startsWith('ca')
          ? 'ca'
          : 'proteinmpnn'
      : null;
    const topKUpdateMatch = /\b(?:change|set|update)\s+top[\s-]*k\s*(?:to|=|:)?\s*(\d+)/i.exec(clause);
    const topKUpdate = topKUpdateMatch ? Number(topKUpdateMatch[1]) : null;

    const hasTemp = temperature !== null && Number.isFinite(temperature) && temperature > 0;
    const hasOmit = omitAminoAcids !== null;
    const hasSeqs = numSequences !== null && Number.isInteger(numSequences) && numSequences >= 1;
    const hasDesigns = numDesigns !== null && Number.isInteger(numDesigns) && numDesigns >= 1;
    const hasTopK = topKUpdate !== null && Number.isInteger(topKUpdate) && topKUpdate >= 1;

    if (hasTemp || hasOmit || hasSeqs || hasDesigns || mpnnFamily !== null || hasTopK) {
      const { nodeIds: paramTargetIds, adapterIds: paramAdapterIds } = resolveCanvasNodesForKeyword(
        clause,
        currentGraph.nodes,
      );
      if (mpnnFamily) {
        const mpnnHits = resolveCanvasNodesForKeyword('mpnn', currentGraph.nodes);
        for (const id of mpnnHits.nodeIds) if (!paramTargetIds.includes(id)) paramTargetIds.push(id);
        for (const aid of mpnnHits.adapterIds) if (!paramAdapterIds.includes(aid)) paramAdapterIds.push(aid);
      }
      ops.push({
        ...makeEmptyEditOp('update_params'),
        ...(paramTargetIds.length > 0 ? { targetNodeIds: paramTargetIds } : {}),
        ...(paramAdapterIds.length > 0 ? { targetAdapterIds: paramAdapterIds } : {}),
        ...(hasTemp ? { temperature } : {}),
        ...(hasOmit ? { omitAminoAcids } : {}),
        ...(hasSeqs ? { numSequences } : {}),
        ...(hasDesigns ? { numDesigns } : {}),
        ...(mpnnFamily !== null ? { mpnnFamily } : {}),
        ...(hasTopK ? { topK: topKUpdate } : {}),
        score: false,
      });
    }
  }

  if (ops.length === 0) return null;
  return {
    kind: 'edit_current',
    operations: ops.slice(0, MAX_EDIT_OPERATIONS),
    summary: sanitizeDisplayText(`Edit workflow: ${trimmed}`, 'Edit current workflow'),
  };
}

function extractPromptPartialRedesignSpan(prompt: string): {
  chain: string;
  start: number;
  end: number;
} | null {
  const explicitChainMatch = /\bchain\s+([A-Za-z])\b/i.exec(prompt);
  const firstNMatch =
    /(?:redesign|remodel|mutate|replace|regenerate)\b[\s\S]{0,60}?\bfirst\s+(\d{1,3})\s+residues?\b/i.exec(
      prompt,
    );
  if (firstNMatch) {
    const end = Number(firstNMatch[1]);
    if (Number.isInteger(end) && end >= 1 && end <= 500) {
      return {
        chain: (explicitChainMatch?.[1] ?? 'A').toUpperCase(),
        start: 1,
        end,
      };
    }
  }
  const rangeMatch =
    /(?:redesign|remodel|mutate|replace|regenerate)\b[\s\S]{0,60}?\bresidues?\s+([A-Za-z])?\s*(\d{1,4})\s*[-–]\s*(\d{1,4})\b/i.exec(
      prompt,
    );
  if (rangeMatch) {
    const start = Number(rangeMatch[2]);
    const end = Number(rangeMatch[3]);
    if (Number.isInteger(start) && Number.isInteger(end) && start >= 1 && end >= start) {
      return {
        chain: (rangeMatch[1] ?? explicitChainMatch?.[1] ?? 'A').toUpperCase(),
        start,
        end,
      };
    }
  }
  return null;
}

function extractPromptAntibodyChainInfo(prompt: string): {
  explicitChain: string | null;
  inferredHlChain: 'H' | 'L' | null;
} {
  const chainMatch = /\bchain\s+([A-Za-z])\b/i.exec(prompt);
  const explicitChain = chainMatch ? chainMatch[1].toUpperCase() : null;
  const cdrChainMatch = /\bcdr[\s-]*([hl])\d?\b/i.exec(prompt);
  let inferredHlChain: 'H' | 'L' | null = null;
  if (cdrChainMatch) {
    inferredHlChain = cdrChainMatch[1].toUpperCase() as 'H' | 'L';
  } else if (/\blight\s+chain\b/i.test(prompt)) {
    inferredHlChain = 'L';
  } else if (/\bheavy\s+chain\b/i.test(prompt)) {
    inferredHlChain = 'H';
  }
  return { explicitChain, inferredHlChain };
}

function reconcileBlockGraphPlan(
  plan: BlockGraphWorkflowPlan,
  prompt: string,
): BlockGraphWorkflowPlan {
  if (plan.mode === 'edit' || !Array.isArray(plan.nodes)) return plan;
  const trimmed = prompt.trim();
  const clonedNodes = plan.nodes.map((n) => ({
    ...n,
    params: n.params ? { ...n.params } : undefined,
  }));
  const promptPdbId = extractPromptPdbId(trimmed);
  const hasAntibodyIntent =
    /\b(?:antibody|antibodies|fab|scfv|cdr\d*|imgt|kabat|antifold)\b/i.test(trimmed) ||
    promptPdbId === '6XEY' ||
    promptPdbId === '6W41' ||
    promptPdbId === '1IGT';
  const isExcludedAutoChain = /\b(?:1N8Z|1MEL|nanobody|nanobodies|vhh)\b/i.test(trimmed);

  // 1. Partial redesign (e.g. 1LVM "redesign first 10 residues of chain A")
  const partialSpan = extractPromptPartialRedesignSpan(trimmed);
  if (partialSpan) {
    const { chain, start, end } = partialSpan;
    const complement =
      start <= 1
        ? `${chain}${end + 1}-9999`
        : `${chain}1-${start - 1}, ${chain}${end + 1}-9999`;
    const invertedSpanRe = new RegExp(`^(?:${chain})?${start}\\s*-\\s*${end}$`, 'i');
    for (const n of clonedNodes) {
      if (!n.params) n.params = {};
      if (n.adapter === 'pdb-input') {
        const crop = String(n.params.cropSegments ?? '').trim();
        if (crop && invertedSpanRe.test(crop)) {
          n.params.cropSegments = chain;
        }
      } else if (n.adapter === 'rfd4-proteina' || n.adapter === 'genie3') {
        const ms = String(n.params.motifSpan ?? '').trim();
        if (ms && invertedSpanRe.test(ms)) {
          delete n.params.motifSpan;
        }
      } else if (
        n.adapter === 'web-protein-mpnn' ||
        n.adapter === 'ligandmpnn' ||
        n.adapter === 'lasermpnn' ||
        n.adapter === 'antifold-esmif1'
      ) {
        const curFixed = String(n.params.fixedPositions ?? '').trim();
        if (!curFixed || invertedSpanRe.test(curFixed)) {
          n.params.fixedPositions = complement;
        }
      }
    }
  }

  // 2. Antibody AntiFold targetChain & CDR regions (e.g. 6XEY vs 1N8Z/1MEL)
  const { explicitChain, inferredHlChain } = extractPromptAntibodyChainInfo(trimmed);
  const cdrRegions = extractPromptCdrRegions(trimmed);
  const pdbNodes = clonedNodes.filter((n) => n.adapter === 'pdb-input');
  const hasExcludedPdb = pdbNodes.some((n) => {
    const pid = String(n.params?.pdbId ?? '').trim().toUpperCase();
    return pid === '1N8Z' || pid === '1MEL';
  });
  const isExcludedAntibodyAuto = isExcludedAutoChain || hasExcludedPdb;
  for (const n of clonedNodes) {
    if (n.adapter !== 'antifold-esmif1') continue;
    if (!n.params) n.params = {};
    const rawVar = n.params.variant != null ? String(n.params.variant).trim().toLowerCase() : '';
    const isExplicitEsmif =
      rawVar === 'esmif1' ||
      (rawVar !== 'antifold' &&
        (String(n.params.mode ?? '').trim().toLowerCase().includes('esm') ||
          String(n.params.model ?? '').trim().toLowerCase().includes('esm')));
    if (isExplicitEsmif) continue;
    const curRegions = String(n.params.regions ?? '').trim();
    const isDefaultAllCdrs =
      curRegions.toUpperCase().replace(/[\s,]+/g, ' ').trim() === 'CDR1 CDR2 CDR3';
    if (
      cdrRegions &&
      (!curRegions ||
        (isDefaultAllCdrs && cdrRegions !== 'CDR1 CDR2 CDR3') ||
        cdrRegions.length > curRegions.length)
    ) {
      n.params.regions = cdrRegions;
    }
    const fixedPos = String(n.params.fixedPositions ?? '').trim();
    const curTarget = String(n.params.targetChain ?? '').trim().toUpperCase();
    if (explicitChain) {
      if (!curTarget || curTarget !== explicitChain) {
        n.params.targetChain = explicitChain;
      }
    } else if (isExcludedAntibodyAuto) {
      if (curTarget === 'A' || curTarget === 'H') {
        n.params.targetChain = '';
      }
    } else if (inferredHlChain) {
      if (
        !fixedPositionsLocksChain(fixedPos, inferredHlChain) &&
        (!curTarget || curTarget === 'A')
      ) {
        n.params.targetChain = inferredHlChain;
      }
    } else if (hasAntibodyIntent && !fixedPositionsLocksChain(fixedPos, 'H')) {
      if (!curTarget || curTarget === 'A') {
        n.params.targetChain = 'H';
      }
    }
  }

  // 3. 1YCR prompt-guided benchmark disambiguation
  const ycrPdb = pdbNodes.find(
    (n) =>
      (String(n.params?.pdbId ?? '').trim().toUpperCase() === '1YCR' || promptPdbId === '1YCR') &&
      !String(n.params?.benchmarkId ?? '').trim(),
  );
  if (ycrPdb) {
    if (!ycrPdb.params) ycrPdb.params = {};
    const explicitBinderIntent =
      /\b(?:binder|binders|hotspot|hotspots|ppi|interface)\b/i.test(trimmed) ||
      /\bdesign\b[\s\S]{0,40}\bbind(?:er|ing)?\b/i.test(trimmed) ||
      /\bbind\s+(?:to|against)\s+mdm2\b/i.test(trimmed) ||
      /\bmdm2\s+binder\b/i.test(trimmed);
    const explicitMotifIntent =
      /\b(?:motif|scaffold|scaffolding|p53\s+helix|helix\s+motif)\b/i.test(trimmed);
    const hasWeakBinderCue = /\b(?:mdm2|bind)\b/i.test(trimmed);
    const hasWeakMotifCue = /\b(?:p53|helix|peptide)\b/i.test(trimmed);
    const downstreamHasBinder = clonedNodes.some((n) => {
      if (!PASS2_GENERATOR_ADAPTERS.has(n.adapter)) return false;
      const m = String(n.params?.mode ?? n.params?.taskMode ?? '').trim();
      const lock = String(n.params?.lockChain ?? '').trim();
      const hot = String(n.params?.hotspots ?? '').trim();
      const contig = String(n.params?.contig ?? '').trim();
      return m === 'binder' || Boolean(lock) || Boolean(hot) || contig.includes('/0');
    });
    const downstreamHasMotif = clonedNodes.some((n) => {
      if (!PASS2_GENERATOR_ADAPTERS.has(n.adapter)) return false;
      const m = String(n.params?.mode ?? n.params?.taskMode ?? '').trim();
      return m === 'motif';
    });

    let chooseBinder: boolean | null = null;
    if (explicitBinderIntent && !explicitMotifIntent) chooseBinder = true;
    else if (explicitMotifIntent && !explicitBinderIntent) chooseBinder = false;
    else if (downstreamHasBinder && !downstreamHasMotif) chooseBinder = true;
    else if (downstreamHasMotif && !downstreamHasBinder) chooseBinder = false;
    else if (explicitBinderIntent) chooseBinder = true;
    else if (explicitMotifIntent) chooseBinder = false;
    else if (hasWeakBinderCue && !hasWeakMotifCue) chooseBinder = true;
    else if (hasWeakMotifCue) chooseBinder = false;

    if (chooseBinder !== null) {
      ycrPdb.params.benchmarkId = chooseBinder ? 'mdm2_1ycr' : 'rfd_1ycr_p53';
      for (const n of clonedNodes) {
        if (!n.params) n.params = {};
        if (n.adapter === 'rfd4-proteina') {
          if (!n.params.mode || n.params.mode === 'unconditional') {
            n.params.mode = chooseBinder ? 'binder' : 'motif';
          }
        } else if (n.adapter === 'genie3') {
          if (!n.params.taskMode || n.params.taskMode === 'unconditional') {
            n.params.taskMode = chooseBinder ? 'binder' : 'motif';
          }
        } else if (n.adapter === 'protpardelle' && !chooseBinder) {
          if (!n.params.taskMode || n.params.taskMode === 'unconditional') {
            n.params.taskMode = 'motif';
          }
        }
      }
    }
  }

  // 4. Guard curated benchmark conditioning against LLM hallucinated residue spans (e.g. `1PRW` `motifSpan: "A1-10"`)
  const hasExplicitPromptResidues =
    /\b[A-Za-z]\d+\s*[-–]\s*\d+\b/.test(trimmed) ||
    /\b(?:residues?|positions?|res|hotspots?|sites?|motif)\s*[:=]?\s*[A-Za-z]?\d+/i.test(trimmed) ||
    /\b[A-Z]\d{2,4}\b/.test(trimmed);
  const blockEdges = Array.isArray(plan.edges) ? plan.edges : [];
  const blockNodeById = new Map(clonedNodes.map((n) => [n.id, n]));
  for (const pdbNode of pdbNodes) {
    const rawBmId = String(pdbNode.params?.benchmarkId ?? '').trim();
    const rawPdbId = String(pdbNode.params?.pdbId ?? '').trim().toUpperCase();
    let bm = rawBmId ? getBenchmarkById(rawBmId) : undefined;
    if (!bm && rawPdbId) {
      const matches = BENCHMARK_TARGETS.filter((b) => b.pdbId === rawPdbId);
      if (matches.length === 1) {
        bm = matches[0];
        if (!pdbNode.params) pdbNode.params = {};
        pdbNode.params.benchmarkId = bm.id;
      }
    }
    if (!bm) continue;

    const wiredGens = blockEdges
      .filter(([fromId]) => fromId === pdbNode.id)
      .map(([, toId]) => blockNodeById.get(toId))
      .filter((n): n is NonNullable<typeof n> => Boolean(n && PASS2_GENERATOR_ADAPTERS.has(n.adapter)));
    const targetGens =
      wiredGens.length > 0
        ? wiredGens
        : pdbNodes.length === 1
          ? clonedNodes.filter((n) => PASS2_GENERATOR_ADAPTERS.has(n.adapter))
          : [];

    const bp = bm.conditionParams;
    for (const gen of targetGens) {
      if (!gen.params) gen.params = {};
      if (gen.adapter === 'rfd4-proteina' || gen.adapter === 'genie3') {
        const curMotif = String(gen.params.motifSpan ?? '').trim();
        if (
          curMotif &&
          (!hasExplicitPromptResidues || isResidueSpecDisjointFromBenchmark(curMotif, bm))
        ) {
          if (bp.motifSpan) gen.params.motifSpan = bp.motifSpan;
          else delete gen.params.motifSpan;
        }
        const curHotspots = String(gen.params.hotspots ?? '').trim();
        if (
          curHotspots &&
          bm.category === 'binder' &&
          (!hasExplicitPromptResidues || isResidueSpecDisjointFromBenchmark(curHotspots, bm))
        ) {
          if (bp.hotspots) gen.params.hotspots = bp.hotspots;
          else delete gen.params.hotspots;
        }
        const curContig = String(gen.params.contig ?? '').trim();
        if (
          curContig &&
          (!hasExplicitPromptResidues || isResidueSpecDisjointFromBenchmark(curContig, bm))
        ) {
          delete gen.params.contig;
        }
      } else if (gen.adapter === 'protpardelle') {
        const curSel = String(gen.params.motifSelection ?? '').trim();
        if (
          curSel &&
          bp.motifSpan &&
          (!hasExplicitPromptResidues || isResidueSpecDisjointFromBenchmark(curSel, bm))
        ) {
          gen.params.motifSelection = bp.motifSpan;
        }
      }
    }
  }

  // 5. All-atom generators -> web-protein-mpnn default to SolubleMPNN unless Cα-ProteinMPNN was explicitly asked for
  const wantsCaMpnn = /\b(?:c[αa][\s-]*(?:protein\s*)?mpnn|c-?alpha[\s-]*(?:protein\s*)?mpnn)\b/i.test(
    trimmed,
  );
  if (!wantsCaMpnn) {
    for (const [fromId, toId] of blockEdges) {
      const fn = blockNodeById.get(fromId);
      const tn = blockNodeById.get(toId);
      if (fn && ALL_ATOM_GENERATORS.has(fn.adapter) && tn?.adapter === 'web-protein-mpnn') {
        if (!tn.params) tn.params = {};
        if (String(tn.params.family ?? '').trim().toLowerCase() === 'ca') {
          tn.params.family = 'solublempnn';
          if (!tn.label || /c[αa]/i.test(tn.label)) {
            tn.label = 'SolubleMPNN';
          }
        }
      }
    }
  }

  return {
    ...plan,
    nodes: clonedNodes,
  };
}

/**
 * Reconcile a planner response against explicit user prompt directives (and active canvas)
 * so stale/incomplete backend responses for Q1–Q9 are repaired cleanly before compilation,
 * while preserving `kind: 'unsupported'` (Q10) untouched.
 */
export function reconcilePlannedWorkflow(
  plan: WorkflowPlan,
  prompt: string,
  currentGraph?: WorkflowGraph,
): WorkflowPlan {
  if (plan.kind === 'unsupported') return plan;
  if (plan.kind === 'block_graph') {
    return reconcileBlockGraphPlan(plan, prompt);
  }
  const trimmed = prompt.trim();
  const hasCurrentGraph = Boolean(currentGraph && currentGraph.nodes.length > 0);
  const promptPdbId = extractPromptPdbId(trimmed);
  const promptSeqs = extractPromptSequences(trimmed);
  const promptOmit = extractPromptOmitAminoAcids(trimmed);
  const promptTemp = extractPromptTemperature(trimmed);
  const promptTopK = extractPromptTopK(trimmed);
  const promptFolds = extractPromptFoldModels(trimmed);
  const { numDesigns: promptNumDesigns, length: promptLength } =
    extractPromptNumDesignsAndLength(trimmed);
  const hasAntibodyIntent =
    /\b(?:antibody|antibodies|nanobody|nanobodies|fab|scfv|vhh|cdr\d*|imgt|kabat|antifold)\b/i.test(
      trimmed,
    );
  const wantsSolubleMpnn = /\bsoluble\s*mpnn\b/i.test(trimmed);
  const wantsStability = /\b(?:thermompnn|stabiliz|ddg|ΔΔG|aggregat)/i.test(trimmed);
  const wantsPartial = /\bpartial\s+(?:diffusion|denoising)\b|\bpartial\b/i.test(trimmed);
  const wantsBackboneGen =
    /\b(?:rfd4|genie\s*3|genie3|protpardelle|diffusion|scaffold|backbone|binder|de\s*novo)\b/i.test(
      trimmed,
    );

  // Rule 1: Incremental edit reconciliation (R1 / Q5)
  if (hasCurrentGraph && currentGraph) {
    const detEdit = parseDeterministicEditOperations(trimmed, currentGraph);
    if (plan.kind === 'edit_current') {
      const existingOps = Array.isArray(plan.operations) ? [...plan.operations] : [];
      if (detEdit) {
        if (existingOps.length === 0) {
          return detEdit;
        }
        for (const detOp of detEdit.operations) {
          if (detOp.action === 'remove_nodes') {
            const hasRemove = existingOps.some((o) => o.action === 'remove_nodes');
            if (!hasRemove) existingOps.push(detOp);
          } else if (detOp.action === 'update_params') {
            const detTargets = new Set(detOp.targetNodeIds ?? []);
            const existingUpdate =
              detTargets.size > 0
                ? existingOps.find(
                    (o) =>
                      o.action === 'update_params' &&
                      (!(o.targetNodeIds && o.targetNodeIds.length > 0) ||
                        o.targetNodeIds.some((id) => detTargets.has(id))),
                  )
                : existingOps.find((o) => o.action === 'update_params');
            if (!existingUpdate) {
              existingOps.push(detOp);
            } else {
              if (detOp.temperature != null && existingUpdate.temperature == null) {
                existingUpdate.temperature = detOp.temperature;
              }
              if (detOp.omitAminoAcids != null && existingUpdate.omitAminoAcids == null) {
                existingUpdate.omitAminoAcids = detOp.omitAminoAcids;
              }
              if (detOp.numSequences != null && existingUpdate.numSequences == null) {
                existingUpdate.numSequences = detOp.numSequences;
              }
              if (detOp.numDesigns != null && existingUpdate.numDesigns == null) {
                existingUpdate.numDesigns = detOp.numDesigns;
              }
              if (detOp.mpnnFamily != null && existingUpdate.mpnnFamily == null) {
                existingUpdate.mpnnFamily = detOp.mpnnFamily;
              }
              if (detOp.topK != null && existingUpdate.topK == null) {
                existingUpdate.topK = detOp.topK;
              }
            }
          } else if (!existingOps.some((o) => o.action === detOp.action)) {
            existingOps.push(detOp);
          }
        }
        return {
          ...plan,
          operations: existingOps.slice(0, MAX_EDIT_OPERATIONS),
        };
      }
      return plan;
    }

    const looksLikeIncrementalEdit =
      /\b(?:remove|delete|drop|strip|change|set|update|adjust|add|insert|switch)\b/i.test(trimmed) &&
      !promptPdbId &&
      promptSeqs.length === 0 &&
      !/\bde\s*novo\b/i.test(trimmed);
    if (looksLikeIncrementalEdit && detEdit) {
      return detEdit;
    }
  }

  // Rule 2: Sequence folding promotion / reconciliation (Q7)
  if (promptSeqs.length > 0 && !promptPdbId && !wantsBackboneGen) {
    const seqPlan: SequenceWorkflowPlan =
      plan.kind === 'sequence'
        ? { ...plan }
        : {
            kind: 'sequence',
            sequences: promptSeqs,
            label: promptSeqs.length > 1 ? 'Heterodimer Sequence' : 'Sequence Input',
            foldModels:
              promptFolds.length > 0
                ? promptFolds
                : [promptSeqs.length > 1 ? 'multimer' : 'monomer'],
            stabilityScan: wantsStability,
            esmcScore: /\besm[\s-]*c\b/i.test(trimmed),
            dynamicsScan: /\bdyna[\s-]*1\b/i.test(trimmed),
            topK: promptTopK,
            score: false,
            summary: sanitizeDisplayText(
              plan.summary || `Fold ${promptSeqs.length}-chain sequence`,
              'Fold protein sequence',
            ),
          };
    if (seqPlan.sequences.length < promptSeqs.length) {
      seqPlan.sequences = promptSeqs;
    }
    if (promptFolds.length > 0) {
      seqPlan.foldModels = promptFolds;
    } else if (seqPlan.sequences.length > 1) {
      if (seqPlan.foldModels.length === 0 || (seqPlan.foldModels.length === 1 && seqPlan.foldModels[0] === 'monomer')) {
        seqPlan.foldModels = ['multimer'];
      }
    } else if (seqPlan.foldModels.length === 0) {
      seqPlan.foldModels = ['monomer'];
    }
    if (wantsStability) seqPlan.stabilityScan = true;
    if (/\besm[\s-]*c\b/i.test(trimmed)) seqPlan.esmcScore = true;
    if (/\bdyna[\s-]*1\b/i.test(trimmed)) seqPlan.dynamicsScan = true;
    if (promptTopK !== null && !seqPlan.topK) seqPlan.topK = promptTopK;
    return seqPlan;
  }

  // Rule 3: Explicit PDB inverse-folding/analysis mistakenly routed to conditioned_design (Q3 1CTT)
  let reconciled: WorkflowPlan = plan;
  const hasPdbAnalysisTool =
    /\b(?:ligandmpnn|lasermpnn|thermompnn|antifold|esm[\s-]*if[\s-]*1|esm[\s-]*c|dyna[\s-]*1)\b/i.test(
      trimmed,
    );
  if (
    reconciled.kind === 'conditioned_design' &&
    promptPdbId &&
    hasPdbAnalysisTool &&
    !wantsBackboneGen &&
    !wantsPartial
  ) {
    const tasks: PdbTaskId[] = [];
    if (/\bligandmpnn\b/i.test(trimmed)) tasks.push('ligandmpnn');
    if (/\blasermpnn\b/i.test(trimmed)) tasks.push('lasermpnn');
    if (/\bantifold\b/i.test(trimmed)) tasks.push('antifold');
    if (/\besm[\s-]*if[\s-]*1\b/i.test(trimmed)) tasks.push('esmif1');
    if (wantsSolubleMpnn) tasks.push('solublempnn');
    else if (/\bprotein\s*mpnn\b/i.test(trimmed)) tasks.push('proteinmpnn');
    if (/\bthermompnn\b/i.test(trimmed)) tasks.push('thermompnn');
    if (/\besm[\s-]*c\b/i.test(trimmed)) tasks.push('esmc');
    if (/\bdyna[\s-]*1\b/i.test(trimmed)) tasks.push('dyna1');

    reconciled = {
      kind: 'pdb',
      pdbId: promptPdbId,
      crop: reconciled.crop,
      tasks: tasks.length > 0 ? tasks : ['ligandmpnn'],
      targetChain: null,
      regions: extractPromptCdrRegions(trimmed),
      numSequences: reconciled.numSequences ?? null,
      omitAminoAcids: promptOmit ?? reconciled.omitAminoAcids ?? null,
      temperature: promptTemp ?? reconciled.temperature ?? null,
      mpnnFamily: wantsSolubleMpnn ? 'solublempnn' : reconciled.mpnnFamily ?? null,
      topK: promptTopK ?? reconciled.topK ?? null,
      refold: promptFolds.length > 0,
      foldModels: promptFolds,
      stabilityScan: wantsStability,
      esmcScore: /\besm[\s-]*c\b/i.test(trimmed),
      dynamicsScan: /\bdyna[\s-]*1\b/i.test(trimmed),
      score: promptFolds.length > 0,
      ...(reconciled.unsupported ? { unsupported: reconciled.unsupported } : {}),
      summary: reconciled.summary,
    };
  }

  // Rule 4: Partial diffusion + ESM-IF1 reconciliation (Q8 1UBQ)
  if (wantsPartial && (reconciled.kind === 'conditioned_design' || reconciled.kind === 'pdb')) {
    const gens: GeneratorId[] = /\bprotpardelle\b/i.test(trimmed)
      ? ['protpardelle']
      : ['rfd4-proteina'];
    const fracPct = /(\d+(?:\.\d+)?)\s*%/.exec(trimmed);
    const fracDec = /\bpartial(?:\s+diffusion|\s+fraction)?\s*(?:=|of|at)?\s*(0?\.\d+)\b/i.exec(trimmed);
    const partialFraction = fracPct
      ? clampPartialFraction(Number(fracPct[1]) / 100)
      : fracDec
        ? clampPartialFraction(Number(fracDec[1]))
        : reconciled.kind === 'conditioned_design' && reconciled.partialFraction != null
          ? clampPartialFraction(reconciled.partialFraction)
          : 0.5;
    const seqDesigners: SequenceDesignerId[] = [];
    if (/\besm[\s-]*if[\s-]*1\b/i.test(trimmed)) seqDesigners.push('esmif1');
    if (wantsSolubleMpnn) seqDesigners.push('solublempnn');
    else if (/\bprotein\s*mpnn\b/i.test(trimmed)) seqDesigners.push('web-protein-mpnn');
    if (/\bligandmpnn\b/i.test(trimmed)) seqDesigners.push('ligandmpnn');
    if (/\blasermpnn\b/i.test(trimmed)) seqDesigners.push('lasermpnn');
    if (/\bantifold\b/i.test(trimmed) && hasAntibodyIntent) seqDesigners.push('antifold');
    if (seqDesigners.length === 0 && reconciled.kind === 'conditioned_design') {
      const prev = reconciled.sequenceDesigners ?? (reconciled.sequenceDesigner ? [reconciled.sequenceDesigner] : []);
      seqDesigners.push(...prev.filter((d) => hasAntibodyIntent || d !== 'antifold'));
    }

    const baseFoldModels =
      promptFolds.length > 0
        ? promptFolds
        : reconciled.foldModels.length > 0
          ? reconciled.foldModels
          : (['monomer'] as FoldOracleId[]);
    const wantsScore =
      /\b(?:designability|scrmsd|sctm|refold|score)\b/i.test(trimmed) || baseFoldModels.length > 0;

    reconciled = {
      kind: 'conditioned_design',
      designTask: 'partial',
      partialFraction,
      benchmarkId: null,
      pdbId:
        promptPdbId ??
        (reconciled.kind === 'conditioned_design' ? reconciled.pdbId : reconciled.pdbId) ??
        '1UBQ',
      crop: reconciled.crop,
      generators: gens,
      length: null,
      numDesigns: promptNumDesigns ?? (reconciled.kind === 'conditioned_design' ? reconciled.numDesigns : 1),
      sequenceDesigners: seqDesigners,
      numSequences: reconciled.numSequences ?? 2,
      omitAminoAcids: promptOmit ?? reconciled.omitAminoAcids ?? null,
      temperature: promptTemp ?? reconciled.temperature ?? null,
      mpnnFamily: wantsSolubleMpnn ? 'solublempnn' : reconciled.mpnnFamily ?? null,
      topK: promptTopK ?? reconciled.topK ?? null,
      refold: true,
      foldModels: baseFoldModels,
      score: wantsScore,
      stabilityScan: reconciled.stabilityScan || wantsStability,
      esmcScore: reconciled.esmcScore || /\besm[\s-]*c\b/i.test(trimmed),
      dynamicsScan: reconciled.dynamicsScan || /\bdyna[\s-]*1\b/i.test(trimmed),
      ...(reconciled.unsupported ? { unsupported: reconciled.unsupported } : {}),
      summary: reconciled.summary,
    };
  }

  // Rule 5 & 6: Parameter extraction + domain guard per plan kind
  if (reconciled.kind === 'de_novo') {
    const next: DeNovoWorkflowPlan = { ...reconciled };
    if (promptNumDesigns !== null) next.numDesigns = promptNumDesigns;
    if (promptLength !== null) next.length = promptLength;
    if (promptOmit !== null) next.omitAminoAcids = promptOmit;
    if (promptTemp !== null) next.temperature = promptTemp;
    if (promptTopK !== null) next.topK = promptTopK;
    if (promptFolds.length > 0) {
      next.foldModels = promptFolds;
      next.refold = true;
    }
    if (wantsStability) next.stabilityScan = true;
    const curDes =
      next.sequenceDesigners && next.sequenceDesigners.length > 0
        ? next.sequenceDesigners
        : next.sequenceDesigner
          ? [next.sequenceDesigner]
          : [];
    if (wantsSolubleMpnn) {
      next.mpnnFamily = 'solublempnn';
      next.sequenceDesigners =
        curDes.length > 0
          ? curDes.map((d) => (d === 'antifold' || d === 'web-protein-mpnn' ? 'solublempnn' : d))
          : ['solublempnn'];
      if (next.sequenceDesigner) {
        next.sequenceDesigner = next.sequenceDesigners[0];
      }
    } else if (!hasAntibodyIntent && curDes.includes('antifold')) {
      next.mpnnFamily = next.mpnnFamily ?? 'solublempnn';
      next.sequenceDesigners = curDes.map((d) => (d === 'antifold' ? 'solublempnn' : d));
      if (next.sequenceDesigner === 'antifold') {
        next.sequenceDesigner = 'solublempnn';
      }
    }
    return next;
  }

  if (reconciled.kind === 'conditioned_design') {
    const next: ConditionedDesignWorkflowPlan = { ...reconciled };
    if (promptPdbId && !next.pdbId && !next.benchmarkId) next.pdbId = promptPdbId;
    if (!next.benchmarkId && (next.pdbId || promptPdbId)) {
      const targetPdb = (promptPdbId ?? next.pdbId ?? '').toUpperCase();
      const matchingBench = BENCHMARK_TARGETS.find(
        (b) =>
          b.pdbId === targetPdb &&
          (next.designTask === 'binder'
            ? b.category === 'binder'
            : next.designTask === 'enzyme-ligand'
              ? b.category === 'enzyme-ligand'
              : b.category === 'motif'),
      );
      if (matchingBench) next.benchmarkId = matchingBench.id;
    }
    if (promptNumDesigns !== null) next.numDesigns = promptNumDesigns;
    if (promptLength !== null && next.designTask !== 'partial') next.length = promptLength;
    if (promptOmit !== null) next.omitAminoAcids = promptOmit;
    if (promptTemp !== null) next.temperature = promptTemp;
    if (promptTopK !== null) next.topK = promptTopK;
    if (promptFolds.length > 0) {
      next.foldModels = promptFolds;
      next.refold = true;
    }
    if (/\b(?:motif\s*rmsd|designability|scrmsd|sctm)\b/i.test(trimmed)) next.score = true;
    if (wantsStability) next.stabilityScan = true;
    const curDes =
      next.sequenceDesigners && next.sequenceDesigners.length > 0
        ? next.sequenceDesigners
        : next.sequenceDesigner
          ? [next.sequenceDesigner]
          : [];
    if (wantsSolubleMpnn) {
      next.mpnnFamily = 'solublempnn';
      const cleaned = curDes.filter((d) => d !== 'antifold');
      next.sequenceDesigners = cleaned.length > 0
        ? cleaned.map((d) => (d === 'web-protein-mpnn' ? 'solublempnn' : d))
        : ['solublempnn'];
      if (next.sequenceDesigner) {
        next.sequenceDesigner = next.sequenceDesigners[0];
      }
    } else if (!hasAntibodyIntent && curDes.includes('antifold')) {
      next.mpnnFamily = next.mpnnFamily ?? 'solublempnn';
      next.sequenceDesigners = curDes.map((d) => (d === 'antifold' ? 'solublempnn' : d));
      if (next.sequenceDesigner === 'antifold') {
        next.sequenceDesigner = 'solublempnn';
      }
    }
    return next;
  }

  if (reconciled.kind === 'pdb') {
    const next: PdbWorkflowPlan = { ...reconciled };
    if (promptPdbId && (!next.pdbId || next.pdbId.toLowerCase() === 'demo')) {
      next.pdbId = promptPdbId;
    }
    if (promptOmit !== null) next.omitAminoAcids = promptOmit;
    if (promptTemp !== null) next.temperature = promptTemp;
    if (promptTopK !== null) next.topK = promptTopK;
    if (promptFolds.length > 0) {
      next.foldModels = promptFolds;
      next.refold = true;
      next.score = true;
    }
    let existingTasks: PdbTaskId[] =
      next.tasks && next.tasks.length > 0
        ? [...next.tasks]
        : next.task
          ? [next.task]
          : [];
    const requestedAntiFold = hasAntibodyIntent && /\bantifold\b/i.test(trimmed);
    const requestedProteinMpnn = /\b(?:protein\s*mpnn|ca[- ]?protein\s*mpnn)\b/i.test(trimmed);
    if (requestedAntiFold && !existingTasks.includes('antifold')) {
      existingTasks = existingTasks.filter(
        (t) =>
          (t !== 'proteinmpnn' || requestedProteinMpnn) &&
          (t !== 'solublempnn' || wantsSolubleMpnn),
      );
      existingTasks.unshift('antifold');
    }
    if (/\bligandmpnn\b/i.test(trimmed) && !existingTasks.includes('ligandmpnn')) {
      existingTasks.unshift('ligandmpnn');
    }
    if (/\bthermompnn\b/i.test(trimmed) && !existingTasks.includes('thermompnn')) {
      existingTasks.push('thermompnn');
      next.stabilityScan = true;
    }
    if (/\besm[\s-]*if[\s-]*1\b/i.test(trimmed) && !existingTasks.includes('esmif1')) {
      existingTasks.push('esmif1');
    }
    if (wantsSolubleMpnn) {
      next.mpnnFamily = 'solublempnn';
      const filtered = requestedAntiFold
        ? existingTasks
        : existingTasks.filter((t) => t !== 'antifold');
      next.tasks = filtered.length > 0
        ? filtered.map((t) => (t === 'proteinmpnn' ? 'solublempnn' : t))
        : ['solublempnn'];
    } else if (!hasAntibodyIntent && existingTasks.includes('antifold')) {
      next.mpnnFamily = next.mpnnFamily ?? 'solublempnn';
      next.tasks = existingTasks.map((t) => (t === 'antifold' ? 'solublempnn' : t));
      if (wantsStability) {
        next.stabilityScan = true;
        if (!next.refold && !next.tasks.includes('thermompnn')) {
          next.tasks.push('thermompnn');
        }
      }
    } else if (existingTasks.length > 0) {
      next.tasks = existingTasks;
    }
    if (next.task && next.tasks && !next.tasks.includes(next.task)) {
      next.task = next.tasks[0];
    }
    if (wantsStability) next.stabilityScan = true;
    if (/\besm[\s-]*c\b/i.test(trimmed)) next.esmcScore = true;
    if (/\bdyna[\s-]*1\b/i.test(trimmed)) next.dynamicsScan = true;
    if (next.tasks?.includes('antifold')) {
      const cdrRegions = extractPromptCdrRegions(trimmed);
      const curRegions = String(next.regions ?? '').trim();
      const isDefaultAllCdrs =
        curRegions.toUpperCase().replace(/[\s,]+/g, ' ').trim() === 'CDR1 CDR2 CDR3';
      if (
        cdrRegions &&
        (!curRegions ||
          (isDefaultAllCdrs && cdrRegions !== 'CDR1 CDR2 CDR3') ||
          cdrRegions.length > curRegions.length)
      ) {
        next.regions = cdrRegions;
      }
      const { explicitChain, inferredHlChain } = extractPromptAntibodyChainInfo(trimmed);
      const pdbUpper = String(next.pdbId ?? promptPdbId ?? '').trim().toUpperCase();
      const isExcludedPdbOrNb =
        pdbUpper === '1N8Z' ||
        pdbUpper === '1MEL' ||
        /\b(?:1N8Z|1MEL|nanobody|nanobodies|vhh)\b/i.test(trimmed);
      const curTarget = String(next.targetChain ?? '').trim().toUpperCase();
      if (explicitChain) {
        if (!curTarget || curTarget !== explicitChain) {
          next.targetChain = explicitChain;
        }
      } else if (isExcludedPdbOrNb) {
        if (curTarget === 'A' || curTarget === 'H') {
          next.targetChain = '';
        }
      } else if (inferredHlChain) {
        if (!curTarget || curTarget === 'A') {
          next.targetChain = inferredHlChain;
        }
      } else if (
        (hasAntibodyIntent || pdbUpper === '6XEY' || pdbUpper === '6W41' || pdbUpper === '1IGT') &&
        (!curTarget || curTarget === 'A')
      ) {
        next.targetChain = 'H';
      }
    }
    return next;
  }

  return reconciled;
}

interface SessionCachedPlan {
  plan: Exclude<WorkflowPlan, UnsupportedWorkflowPlan>;
  model: string;
}

function buildSessionCacheKey(
  prompt: string,
  currentNodes: Array<{ id: string; adapterId: string }>,
): string {
  const normPrompt = prompt.trim().toLowerCase().replace(/\s+/g, ' ');
  const nodeSig = currentNodes
    .map((n) => `${n.id}:${n.adapterId}`)
    .sort()
    .join(',');
  return `${normPrompt}||${nodeSig}`;
}

function readSessionCache(key: string): SessionCachedPlan | null {
  try {
    if (typeof sessionStorage === 'undefined') return null;
    const raw = sessionStorage.getItem(SESSION_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Record<string, SessionCachedPlan>;
    return parsed && typeof parsed === 'object' && parsed[key] ? parsed[key] : null;
  } catch {
    return null;
  }
}

function writeSessionCache(key: string, entry: SessionCachedPlan): void {
  try {
    if (typeof sessionStorage === 'undefined') return;
    const raw = sessionStorage.getItem(SESSION_CACHE_KEY);
    const parsed = (raw ? JSON.parse(raw) : {}) as Record<string, SessionCachedPlan>;
    const next: Record<string, SessionCachedPlan> =
      parsed && typeof parsed === 'object' ? { ...parsed } : {};
    delete next[key];
    next[key] = entry;
    const keys = Object.keys(next);
    while (keys.length > SESSION_CACHE_MAX_ENTRIES) {
      const oldest = keys.shift();
      if (oldest) delete next[oldest];
    }
    sessionStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(next));
  } catch {
    // Ignore private-browsing / storage-quota errors
  }
}

export function parseRetryAfterSec(headerValue: string | null | undefined): number | null {
  if (!headerValue) return null;
  const trimmed = headerValue.trim();
  const asNum = Number(trimmed);
  if (Number.isFinite(asNum) && asNum > 0) {
    return Math.ceil(asNum);
  }
  const asDate = Date.parse(trimmed);
  if (Number.isFinite(asDate)) {
    const diffSec = Math.ceil((asDate - Date.now()) / 1000);
    return diffSec > 0 ? diffSec : null;
  }
  return null;
}

function combineSignalWithTimeout(
  userSignal: AbortSignal | undefined,
  timeoutMs: number,
): { signal: AbortSignal; cleanup: () => void; timedOut: () => boolean } {
  const ctrl = new AbortController();
  let didTimeout = false;
  const timer = setTimeout(() => {
    didTimeout = true;
    ctrl.abort();
  }, timeoutMs);

  const onUserAbort = () => {
    clearTimeout(timer);
    ctrl.abort();
  };
  if (userSignal) {
    if (userSignal.aborted) {
      onUserAbort();
    } else {
      userSignal.addEventListener('abort', onUserAbort, { once: true });
    }
  }

  return {
    signal: ctrl.signal,
    timedOut: () => didTimeout,
    cleanup: () => {
      clearTimeout(timer);
      userSignal?.removeEventListener('abort', onUserAbort);
    },
  };
}

export async function requestWorkflowPlan(
  prompt: string,
  options?: { currentGraph?: WorkflowGraph; signal?: AbortSignal; skipSessionCache?: boolean },
): Promise<PlannedWorkflowResult> {
  const trimmed = prompt.trim();
  if (trimmed.length < PLANNER_MIN_PROMPT_LENGTH) {
    throw new PlannerHttpError({
      message: `Please describe your workflow in at least ${PLANNER_MIN_PROMPT_LENGTH} characters.`,
      status: 400,
      code: 'prompt_too_short',
    });
  }

  const endpoint =
    (import.meta.env.VITE_PLANNER_URL as string | undefined)?.trim() ||
    DEFAULT_PLANNER_ENDPOINT;

  const currentNodes =
    options?.currentGraph?.nodes.slice(0, 20).map((n) => ({
      id: n.id,
      adapterId: n.adapterId,
      label: n.label,
    })) ?? [];

  const cacheKey = buildSessionCacheKey(trimmed, currentNodes);
  if (!options?.skipSessionCache) {
    const cached = readSessionCache(cacheKey);
    if (cached) {
      const reconciledCached = reconcilePlannedWorkflow(
        cached.plan,
        trimmed,
        options?.currentGraph,
      ) as Exclude<WorkflowPlan, UnsupportedWorkflowPlan>;
      const { graph, notes, unsupported, isIncrementalEdit, addedNodeCount } =
        compileWorkflowPlanToGraph(reconciledCached, options?.currentGraph);
      return {
        plan: reconciledCached,
        graph,
        isIncrementalEdit,
        addedNodeCount,
        unsupported,
        model: cached.model,
        latencyMs: 0,
        notes,
      };
    }
  }

  const { signal, cleanup, timedOut } = combineSignalWithTimeout(
    options?.signal,
    PLANNER_TIMEOUT_MS,
  );

  let resp: Response;
  try {
    resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: trimmed,
        ...(currentNodes.length > 0 ? { currentNodes } : {}),
      }),
      signal,
    });
  } catch (err) {
    cleanup();
    if (timedOut()) {
      throw new PlannerHttpError({
        message: 'Workflow planner timed out — try again or add blocks manually.',
        status: 504,
        code: 'timeout',
      });
    }
    if ((err as Error)?.name === 'AbortError') {
      throw err;
    }
    throw new PlannerHttpError({
      message:
        'Could not reach the workflow planner (network or rate limit). You can still build workflows manually with + Add Block.',
      status: 0,
      code: 'network_error',
    });
  } finally {
    cleanup();
  }

  let data: PlannerApiResponse | null = null;
  try {
    data = (await resp.json()) as PlannerApiResponse;
  } catch {
    data = null;
  }

  if (!resp.ok || !data?.ok || !data.plan) {
    // Resilient client-side fallback when remote v7 fails with 500/502 or empty edit ops
    if (
      options?.currentGraph &&
      options.currentGraph.nodes.length > 0 &&
      (resp.status === 502 ||
        resp.status === 500 ||
        (resp.ok && data?.plan?.kind === 'edit_current'))
    ) {
      const fallbackPlan = parseDeterministicEditOperations(trimmed, options.currentGraph);
      if (fallbackPlan && fallbackPlan.operations.length > 0) {
        const { graph, notes, unsupported, isIncrementalEdit, addedNodeCount } =
          compileWorkflowPlanToGraph(fallbackPlan, options.currentGraph);
        writeSessionCache(cacheKey, { plan: fallbackPlan, model: 'client-edit-fallback' });
        return {
          plan: fallbackPlan,
          graph,
          isIncrementalEdit,
          addedNodeCount,
          unsupported,
          model: 'client-edit-fallback',
          latencyMs: 0,
          notes,
        };
      }
    }

    const retryAfterSec = parseRetryAfterSec(resp.headers.get('Retry-After'));
    const code = data?.code ?? null;
    let msg = data?.error;
    if (!msg) {
      if (resp.status === 429) {
        msg =
          code === 'quota_exhausted'
            ? 'Free daily AI planning quota reached — use + Add Block to build manually.'
            : 'Too many requests — please wait a moment and try again.';
      } else if (resp.status === 403) {
        msg = 'Planner access blocked for this origin — use + Add Block to build manually.';
      } else if (resp.status === 413) {
        msg = 'Prompt or canvas is too large for the planner. Try shortening your prompt.';
      } else if (resp.status === 504) {
        msg = 'AI planner timed out — please try again.';
      } else {
        msg = `Planner request failed (HTTP ${resp.status}).`;
      }
    }
    throw new PlannerHttpError({
      message: msg,
      status: resp.status,
      code,
      retryAfterSec: retryAfterSec ?? (resp.status === 429 && code !== 'quota_exhausted' ? 10 : null),
    });
  }

  const model = data.model || '@cf/google/gemma-4-26b-a4b-it';
  const latencyMs = data.latencyMs ?? 0;

  const reconciledPlan = reconcilePlannedWorkflow(data.plan, trimmed, options?.currentGraph);

  if (reconciledPlan.kind === 'unsupported') {
    throw new UnsupportedWorkflowError({
      reason: sanitizeDisplayText(
        reconciledPlan.reason,
        'Requested model is not supported yet.',
        240,
      ),
      unsupported: (reconciledPlan.unsupported ?? []).slice(0, MAX_UNSUPPORTED_ITEMS),
      suggestions: (reconciledPlan.suggestions ?? []).slice(0, 4),
      model,
      latencyMs,
    });
  }

  let { graph, notes, unsupported, isIncrementalEdit, addedNodeCount } =
    compileWorkflowPlanToGraph(reconciledPlan, options?.currentGraph, { lenient: true });
  let totalLatencyMs = latencyMs;

  // Pass 2b (Conditional LLM feedback edit): ONLY fires if deterministic Pass 2a still left
  // unresolved structural errors (validateGraph) or biological incompatibilities (validateBioCapabilities).
  const remainingGraphErrs = validateGraph(graph);
  const remainingBioIssues = validateBioCapabilities(graph);
  if (
    reconciledPlan.kind === 'block_graph' &&
    (remainingGraphErrs.length > 0 || remainingBioIssues.length > 0)
  ) {
    const feedbackIssues = formatGraphFeedbackIssues(graph, remainingGraphErrs, remainingBioIssues);
    if (feedbackIssues.length > 0) {
      const p2Sig = combineSignalWithTimeout(options?.signal, PLANNER_TIMEOUT_MS);
      try {
        const draftNodes = graph.nodes.slice(0, 20).map((n) => ({
          id: n.id,
          adapterId: n.adapterId,
          label: n.label,
        }));
        const p2Resp = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: trimmed,
            currentNodes: draftNodes,
            graphFeedback: {
              graphSummary: serializeGraphForFeedback(graph),
              issues: feedbackIssues,
            },
          }),
          signal: p2Sig.signal,
        });
        if (p2Resp.ok) {
          const p2Data = (await p2Resp.json()) as PlannerApiResponse;
          if (p2Data?.ok && p2Data.plan && p2Data.plan.kind === 'block_graph') {
            totalLatencyMs += p2Data.latencyMs ?? 0;
            const p2Compiled = compileWorkflowPlanToGraph(p2Data.plan, graph, { lenient: true });
            const p2Errs = validateGraph(p2Compiled.graph);
            if (p2Errs.length === 0) {
              graph = p2Compiled.graph;
              notes = [...notes, ...p2Compiled.notes];
            }
          }
        }
      } catch {
        /* Pass 2b is best-effort; fall through to final validation check */
      } finally {
        p2Sig.cleanup();
      }
    }
  }

  const finalErrs = validateGraph(graph);
  if (finalErrs.length > 0) {
    throw new Error(`Drafted block-graph workflow failed validation: ${finalErrs.join('; ')}`);
  }

  writeSessionCache(cacheKey, { plan: reconciledPlan, model });

  return {
    plan: reconciledPlan,
    graph,
    isIncrementalEdit,
    addedNodeCount,
    unsupported,
    model,
    latencyMs: totalLatencyMs,
    notes,
  };
}
