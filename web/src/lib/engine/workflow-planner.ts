// Deterministic WorkflowPlan -> WorkflowGraph compiler + Cloudflare Workers AI client.
// Supports creating fresh multi-branch workflows (`de_novo`, `conditioned_design`, `pdb`, `sequence`)
// AND safely editing/appending/removing/updating onto the user's current canvas (`edit_current`).

import { adaptBenchmarkContigForLength, BENCHMARK_TARGETS, getBenchmarkById } from './benchmarks';
import { autoLayoutGraph, defaultParamsForAdapter, validateConnection, validateGraph } from './graph-utils';
import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from './registry';
import type { WorkflowEdge, WorkflowGraph, WorkflowNode } from './types';

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

export type WorkflowPlan =
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
export const PLANNER_TIMEOUT_MS = 15_000;

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
    id: 'add-thermo-rf3',
    label: '+ ThermoMPNN & RF3',
    prompt: 'Add ThermoMPNN and a RoseTTAFold 3 refolding branch to my current graph',
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
        {
          action: 'add_fold_oracles',
          attachToNodeId: null,
          model: null,
          foldModels: ['rosettafold3'],
          topK: null,
          numSequences: null,
          targetChain: null,
          regions: null,
          score: true,
        },
      ],
      summary: 'Add ThermoMPNN stability scan & RoseTTAFold 3 refold branch',
    },
  },
  {
    id: 'diiron-motif',
    label: 'Di-iron motif: Genie 3 vs RFD4',
    prompt:
      'Evaluate Genie 3 and RFD4 on the di-iron 4-helix bundle motif using AF2, RF3, and Protenix',
    plan: {
      kind: 'conditioned_design',
      designTask: 'motif',
      benchmarkId: 'rfd_1bcf_diiron',
      pdbId: null,
      crop: null,
      generators: ['genie3', 'rfd4-proteina'],
      length: null,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer', 'rosettafold3', 'protenix2'],
      score: true,
      stabilityScan: false,
      summary: 'Di-Iron 4-Helix (1BCF): Genie 3 & RFD4 → AF2, RF3 & Protenix-v2',
    },
  },
  {
    id: '3ptb-compare',
    label: 'LigandMPNN vs LASErMPNN (3PTB)',
    prompt: 'Compare LigandMPNN vs LASErMPNN on 3PTB keeping BEN and CA',
    plan: {
      kind: 'pdb',
      pdbId: '3PTB',
      crop: 'A, BEN, CA',
      tasks: ['ligandmpnn', 'lasermpnn'],
      targetChain: null,
      regions: null,
      numSequences: 2,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: false,
      summary: '3PTB [A, BEN, CA]: Compare LigandMPNN vs LASErMPNN',
    },
  },
  {
    id: 'denovo-100aa',
    label: '100aa de novo: RFD4 + Genie 3',
    prompt: 'Try RFD4 and Genie 3 for 100 length protein design and check refolding',
    plan: {
      kind: 'de_novo',
      generators: ['rfd4-proteina', 'genie3'],
      length: 100,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      summary: '100aa de novo: RFD4-Proteina & Genie 3 → AF2 refolding',
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
    ? plan.unsupported
        .filter((s): s is string => typeof s === 'string' && s.trim().length > 0)
        .slice(0, MAX_UNSUPPORTED_ITEMS)
        .map((s) => sanitizeDisplayText(s, '', 60))
        .filter(Boolean)
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
    const cp = bench?.conditionParams;
    const effectiveContig =
      !isPartial && cp?.contig
        ? plan.length && bench
          ? adaptBenchmarkContigForLength(cp.contig, plan.length, bench.category)
          : cp.contig
        : undefined;
    const partialFrac = clampPartialFraction(plan.partialFraction);

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

      if (genId === 'genie3') {
        const taskMode =
          plan.designTask === 'binder' || bench?.category === 'binder' ? 'binder' : 'motif';
        paramsOverride = {
          taskMode,
          numDesigns: genNumDesigns,
          ...(plan.length ? { length: plan.length } : {}),
          ...(effectiveContig ? { contig: effectiveContig } : {}),
          ...(cp?.motifSpan ? { motifSpan: cp.motifSpan } : {}),
          ...(cp?.lockChain ? { lockChain: cp.lockChain } : {}),
          ...(cp?.hotspots ? { hotspots: cp.hotspots } : {}),
          ...(bench ? { autoSeededBenchmarkId: bench.id } : {}),
        };
      } else if (genId === 'protpardelle') {
        paramsOverride = isPartial
          ? {
              taskMode: 'partial',
              rewindSteps: Math.max(5, Math.min(200, Math.round(partialFrac * 100))),
              numDesigns: genNumDesigns,
            }
          : {
              taskMode: 'motif',
              numDesigns: genNumDesigns,
              ...(plan.length ? { length: plan.length } : {}),
              ...(cp?.motifSpan ? { motifSelection: cp.motifSpan } : {}),
            };
      } else if (isPartial) {
        paramsOverride = {
          mode: 'partial',
          partialFraction: partialFrac,
          numDesigns: genNumDesigns,
          ...(bench ? { autoSeededBenchmarkId: bench.id } : {}),
        };
      } else {
        const mode =
          plan.designTask === 'binder' || bench?.category === 'binder' ? 'binder' : 'motif';
        paramsOverride = {
          mode,
          numDesigns: genNumDesigns,
          ...(plan.length ? { length: plan.length } : {}),
          ...(effectiveContig ? { contig: effectiveContig } : {}),
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
]);

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
  for (const r of ['CDR1', 'CDR2', 'CDR3']) {
    if (new RegExp(`\\b${r}\\b`, 'i').test(prompt)) found.push(r);
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
  if (/\b(?:af2|alphafold2|alphafold)\b/i.test(clause)) {
    addMatch('localfold-af2', (n) => n.adapterId === 'localfold-af2');
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
      if (cdrRegions && (!next.regions || cdrRegions.length > next.regions.length)) {
        next.regions = cdrRegions;
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

  const { graph, notes, unsupported, isIncrementalEdit, addedNodeCount } =
    compileWorkflowPlanToGraph(reconciledPlan, options?.currentGraph);

  writeSessionCache(cacheKey, { plan: reconciledPlan, model });

  return {
    plan: reconciledPlan,
    graph,
    isIncrementalEdit,
    addedNodeCount,
    unsupported,
    model,
    latencyMs,
    notes,
  };
}
