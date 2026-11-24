// Pure batch-flow, sequence-cap resolution, and reactive PlannedFlowMap engine.
// Shared single source of truth across bio-capabilities, run-preflight, GPU workers,
// canvas cards, wires, and node inspectors.

import { ADAPTER_REGISTRY } from './registry';
import { validateSequenceText } from './param-validation';
import { MAX_DESIGNS, MAX_SEQUENCES_PER_DESIGN } from './work-plan';
import type {
  CanonicalArtifact,
  FoldedStructure,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
  WorkflowNode,
} from './types';

export type SequenceCapMode = 'follow-upstream' | 'fixed';

export type FlowDropKind =
  | 'none'
  | 'intentional-filter'
  | 'cap-per-backbone'
  | 'cap-campaign'
  | 'latency-budget'
  | 'single-item-port'
  | 'collection-collapse';

export const MAX_TOTAL_FOLDS_CAP = 64;
export const DEFAULT_MAX_TOTAL_FOLDS = 16;
export const ESMC_MASKED_PASS_BUDGET = 1000;

export const SEQUENCE_CAPPED_ADAPTERS = new Set(['localfold-af2', 'esmc', 'dyna1']);

const MULTI_DESIGN_GENERATORS = new Set([
  'genie3',
  'rfd4-proteina',
  'rfd3',
  'protpardelle',
  'rfdiffusion',
  'boltzgen',
  'la-proteina',
  'proteina',
  'sid-protein',
  'genie2',
  'frameflow',
  'foldflow2',
]);

const INVERSE_FOLD_PRODUCERS = new Set([
  'web-protein-mpnn',
  'ligandmpnn',
  'lasermpnn',
  'antifold-esmif1',
]);

function readClampedInt(value: unknown, fallback: number, min: number, max: number): number {
  let n: number;
  if (typeof value === 'number') {
    n = value;
  } else if (typeof value === 'string' && value.trim() !== '') {
    n = Number(value);
  } else {
    return fallback;
  }
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.round(n)));
}

export function legacyFixedMaxSequencesForAdapter(adapterId: string): number {
  return adapterId === 'localfold-af2' ? 2 : 1;
}

/**
 * Resolve whether a sequence-capped adapter (`localfold-af2`, `esmc`, `dyna1`)
 * should auto-follow upstream batches or hold a fixed per-backbone cap.
 *
 * Backwards compatibility rule:
 * - Explicit `'follow-upstream'` / `'fixed'` always wins.
 * - If `capMode` is omitted AND `maxSequences` is explicitly provided (legacy nodes,
 *   v1 share payloads, or programmatic unit tests), treat as `'fixed'` so legacy
 *   caps are never silently altered.
 * - If neither is specified, default to `'follow-upstream'`.
 */
export function resolveSequenceCapMode(
  adapterId: string,
  params?: Record<string, unknown>,
): SequenceCapMode {
  const raw = String(params?.capMode ?? '').trim().toLowerCase();
  if (raw === 'follow-upstream') return 'follow-upstream';
  if (raw === 'fixed') return 'fixed';
  if (params && params.maxSequences !== undefined && params.maxSequences !== null && params.maxSequences !== '') {
    return 'fixed';
  }
  if (SEQUENCE_CAPPED_ADAPTERS.has(adapterId)) return 'follow-upstream';
  return 'fixed';
}

export interface EffectiveSequenceCap {
  capMode: SequenceCapMode;
  limit: number;
  maxAllowed: number;
  isPassBudgetCapped: boolean;
  unknownLengthFallback: boolean;
}

/**
 * Resolve the effective per-backbone sequence cap for `localfold-af2`, `esmc`, or `dyna1`,
 * incorporating the `esmc` `masked` mode `N × L <= 1000` forward-pass latency guard.
 */
export function resolveEffectiveMaxSequences(
  adapterId: string,
  params?: Record<string, unknown>,
  opts?: { sequenceLength?: number | null },
): EffectiveSequenceCap {
  const capMode = resolveSequenceCapMode(adapterId, params);
  const maxAllowed = MAX_SEQUENCES_PER_DESIGN;

  if (capMode === 'fixed') {
    const fallback = legacyFixedMaxSequencesForAdapter(adapterId);
    const limit = readClampedInt(params?.maxSequences, fallback, 1, maxAllowed);
    return {
      capMode,
      limit,
      maxAllowed,
      isPassBudgetCapped: false,
      unknownLengthFallback: false,
    };
  }

  // capMode === 'follow-upstream'
  if (adapterId === 'esmc') {
    const mode = String(params?.mode ?? 'wildtype').trim().toLowerCase();
    if (mode === 'masked') {
      const seqLen = opts?.sequenceLength;
      if (typeof seqLen === 'number' && Number.isFinite(seqLen) && seqLen > 0) {
        const budgetLimit = Math.max(1, Math.min(maxAllowed, Math.floor(ESMC_MASKED_PASS_BUDGET / seqLen)));
        return {
          capMode,
          limit: budgetLimit,
          maxAllowed,
          isPassBudgetCapped: budgetLimit < maxAllowed,
          unknownLengthFallback: false,
        };
      }
      // Conservative pre-run fallback when upstream sequence length L is unknown
      return {
        capMode,
        limit: 1,
        maxAllowed,
        isPassBudgetCapped: true,
        unknownLengthFallback: true,
      };
    }
  }

  return {
    capMode,
    limit: maxAllowed,
    maxAllowed,
    isPassBudgetCapped: false,
    unknownLengthFallback: false,
  };
}

export interface NodeFlowInfo {
  nodeId: string;
  adapterId: string;
  incomingCount: number;
  incomingBackbones: number;
  processedCount: number;
  droppedCount: number;
  dropKind: FlowDropKind;
  isEstimate: boolean;
  effectiveCap: number | null;
  maxTotalFolds?: number;
  capMode?: SequenceCapMode;
  severity: 'warning' | 'info' | null;
  badgeText: string | null;
  detailText: string | null;
}

export interface EdgeFlowInfo {
  edgeId: string;
  fromNode: string;
  toNode: string;
  toPort: string;
  incomingCount: number;
  processedCount: number;
  droppedCount: number;
  dropKind: FlowDropKind;
  isEstimate: boolean;
  isTruncated: boolean;
  hoverLabel: string;
}

export interface DownstreamImpactNotice {
  targetNodeId: string;
  targetLabel: string;
  targetAdapterId: string;
  incomingCount: number;
  processedCount: number;
  droppedCount: number;
  dropKind: FlowDropKind;
  isEstimate: boolean;
  message: string;
}

export interface PlannedFlowMap {
  nodes: Map<string, NodeFlowInfo>;
  edges: Map<string, EdgeFlowInfo>;
  downstreamImpactsBySource: Map<string, DownstreamImpactNotice[]>;
}

function makeEdgeId(fromNode: string, fromPort: string, toNode: string, toPort: string): string {
  return `${fromNode}:${fromPort}->${toNode}:${toPort}`;
}

function countFastaSequences(raw: unknown): { count: number; firstLength: number | null } {
  if (typeof raw !== 'string') return { count: 1, firstLength: null };
  const trimmed = raw.trim();
  if (!trimmed) return { count: 1, firstLength: null };
  const check = validateSequenceText(trimmed);
  const count = Math.max(1, check.numSequences);
  const firstLength =
    check.numSequences > 0 && check.totalResidues > 0
      ? Math.max(1, Math.round(check.totalResidues / check.numSequences))
      : null;
  return { count, firstLength };
}

function countDistinctBackbones(seqSet: SequenceSet): number {
  const anchors = new Set<string>();
  for (const c of seqSet.candidates) {
    const anchor = (c.designKey ?? c.sourceStructureId ?? '').trim();
    if (anchor && anchor.toLowerCase() !== 'none') anchors.add(anchor);
  }
  return Math.max(1, anchors.size);
}

function getMaterializedStructureCount(
  outputsByNode: Record<string, Record<string, CanonicalArtifact>> | undefined,
  nodeId: string,
): { count: number; length: number | null } | null {
  const struct = outputsByNode?.[nodeId]?.structure as StructureArtifact | undefined;
  if (!struct || struct.kind !== 'StructureArtifact') return null;
  const colLen = struct.collection?.items?.length ?? 0;
  return {
    count: Math.max(1, colLen),
    length: struct.length > 0 ? struct.length : null,
  };
}

function getMaterializedSeqInfo(
  outputsByNode: Record<string, Record<string, CanonicalArtifact>> | undefined,
  nodeId: string,
): { backbones: number; sequences: number; length: number | null } | null {
  const seqSet = outputsByNode?.[nodeId]?.sequences as SequenceSet | undefined;
  if (seqSet && seqSet.kind === 'SequenceSet' && seqSet.candidates.length > 0) {
    const firstSeq = seqSet.candidates[0]?.sequence ?? '';
    return {
      backbones: countDistinctBackbones(seqSet),
      sequences: seqSet.candidates.length,
      length: firstSeq.length > 0 ? firstSeq.length : null,
    };
  }
  return null;
}

/**
 * Compute reactive per-node & per-edge planned/actual batch flow across the full DAG.
 * Uses materialized `outputsByNode` when available (`isEstimate: false`) and parameter
 * estimates pre-run (`isEstimate: true`).
 */
export function computePlannedFlowMap(
  graph: WorkflowGraph,
  outputsByNode?: Record<string, Record<string, CanonicalArtifact>>,
): PlannedFlowMap {
  const nodeById = new Map<string, WorkflowNode>();
  for (const n of graph.nodes) nodeById.set(n.id, n);

  const findEdge = (nodeId: string, toPort: string) =>
    graph.edges.find((e) => e.toNode === nodeId && e.toPort === toPort) ?? null;

  const backboneMemo = new Map<string, { count: number; length: number | null; isEstimate: boolean }>();
  const seqMemo = new Map<
    string,
    { backbones: number; sequences: number; length: number | null; isEstimate: boolean }
  >();

  const backbonesOut = (
    node: WorkflowNode,
    visiting = new Set<string>(),
  ): { count: number; length: number | null; isEstimate: boolean } => {
    const cached = backboneMemo.get(node.id);
    if (cached) return cached;
    if (visiting.has(node.id)) return { count: 1, length: null, isEstimate: true };
    visiting.add(node.id);

    const matStruct = getMaterializedStructureCount(outputsByNode, node.id);
    if (matStruct) {
      const res = { count: matStruct.count, length: matStruct.length, isEstimate: false };
      visiting.delete(node.id);
      backboneMemo.set(node.id, res);
      return res;
    }

    let res: { count: number; length: number | null; isEstimate: boolean };
    if (MULTI_DESIGN_GENERATORS.has(node.adapterId)) {
      const n = readClampedInt(node.params?.numDesigns, 1, 1, MAX_DESIGNS);
      const lenRaw = Number(node.params?.length ?? 0);
      const length = Number.isFinite(lenRaw) && lenRaw >= 5 ? Math.round(lenRaw) : null;
      res = { count: n, length, isEstimate: true };
    } else if (node.adapterId === 'pdb-input') {
      const mode = String(node.params?.mode ?? 'demo-bundle');
      res = { count: 1, length: mode === 'demo-bundle' ? 53 : null, isEstimate: true };
    } else if (node.adapterId === 'localfold-af2') {
      // LocalFold `structure` port emits a StructureArtifact.collection containing all folded candidates
      const upSeqEdge = findEdge(node.id, 'sequences');
      const upSeqNode = upSeqEdge ? nodeById.get(upSeqEdge.fromNode) : null;
      if (upSeqNode) {
        const upInfo = sequencesOut(upSeqNode, visiting);
        const cap = resolveEffectiveMaxSequences('localfold-af2', node.params);
        const maxTotalFolds = readClampedInt(node.params?.maxTotalFolds, 16, 1, 64);
        const perBbAllowed = Math.min(upInfo.sequences, upInfo.backbones * cap.limit);
        const foldedCount =
          upInfo.backbones > 1
            ? Math.max(1, Math.min(perBbAllowed, maxTotalFolds))
            : Math.max(1, Math.min(upInfo.sequences, cap.limit));
        res = { count: foldedCount, length: upInfo.length, isEstimate: upInfo.isEstimate };
      } else {
        res = { count: 1, length: null, isEstimate: true };
      }
    } else {
      const upEdge = findEdge(node.id, 'structure') ?? findEdge(node.id, 'template');
      const upNode = upEdge ? nodeById.get(upEdge.fromNode) : null;
      res = upNode ? backbonesOut(upNode, visiting) : { count: 1, length: null, isEstimate: true };
    }

    visiting.delete(node.id);
    backboneMemo.set(node.id, res);
    return res;
  };

  const sequencesOut = (
    node: WorkflowNode,
    visiting = new Set<string>(),
  ): { backbones: number; sequences: number; length: number | null; isEstimate: boolean } => {
    const cached = seqMemo.get(node.id);
    if (cached) return cached;
    if (visiting.has(node.id)) return { backbones: 1, sequences: 1, length: null, isEstimate: true };
    visiting.add(node.id);

    const matSeq = getMaterializedSeqInfo(outputsByNode, node.id);
    if (matSeq) {
      const res = { ...matSeq, isEstimate: false };
      visiting.delete(node.id);
      seqMemo.set(node.id, res);
      return res;
    }

    let res: { backbones: number; sequences: number; length: number | null; isEstimate: boolean };
    if (MULTI_DESIGN_GENERATORS.has(node.adapterId)) {
      const bb = backbonesOut(node, visiting);
      res = { backbones: bb.count, sequences: bb.count, length: bb.length, isEstimate: bb.isEstimate };
    } else if (INVERSE_FOLD_PRODUCERS.has(node.adapterId)) {
      const upEdge = findEdge(node.id, 'structure');
      const upNode = upEdge ? nodeById.get(upEdge.fromNode) : null;
      const bb = upNode ? backbonesOut(upNode, visiting) : { count: 1, length: null, isEstimate: true };
      const k = readClampedInt(node.params?.numSequences, 2, 1, MAX_SEQUENCES_PER_DESIGN);
      res = { backbones: bb.count, sequences: bb.count * k, length: bb.length, isEstimate: true };
    } else if (node.adapterId === 'top-k-filter') {
      const upEdge = findEdge(node.id, 'sequences');
      const upNode = upEdge ? nodeById.get(upEdge.fromNode) : null;
      const upInfo = upNode
        ? sequencesOut(upNode, visiting)
        : { backbones: 1, sequences: 1, length: null, isEstimate: true };
      const mode = String(node.params?.mode ?? 'per-backbone');
      const topK = readClampedInt(node.params?.topK, 2, 1, 16);
      const maxTotal = readClampedInt(node.params?.maxTotal, 8, 1, MAX_TOTAL_FOLDS_CAP);
      let kept: number;
      if (mode === 'global') {
        kept = Math.min(upInfo.sequences, topK, maxTotal);
      } else {
        const perBb =
          upInfo.backbones > 0 ? Math.ceil(upInfo.sequences / upInfo.backbones) : upInfo.sequences;
        kept = Math.min(maxTotal, upInfo.backbones * Math.min(perBb, topK), upInfo.sequences);
      }
      res = {
        backbones: upInfo.backbones,
        sequences: Math.max(0, kept),
        length: upInfo.length,
        isEstimate: upInfo.isEstimate,
      };
    } else if (node.adapterId === 'evodiff') {
      const k = readClampedInt(node.params?.numSequences, 2, 1, MAX_SEQUENCES_PER_DESIGN);
      const lenRaw = Number(node.params?.length ?? 0);
      res = {
        backbones: 1,
        sequences: k,
        length: Number.isFinite(lenRaw) && lenRaw >= 5 ? Math.round(lenRaw) : null,
        isEstimate: true,
      };
    } else if (node.adapterId === 'sequence-input') {
      const fasta = countFastaSequences(node.params?.sequence);
      res = { backbones: 1, sequences: fasta.count, length: fasta.firstLength, isEstimate: false };
    } else if (node.adapterId === 'pdb-input') {
      const bb = backbonesOut(node, visiting);
      res = { backbones: 1, sequences: 1, length: bb.length, isEstimate: bb.isEstimate };
    } else if (node.adapterId === 'esmc' || node.adapterId === 'dyna1') {
      const upSeqEdge = findEdge(node.id, 'sequences');
      const upStructEdge = findEdge(node.id, 'structure');
      const upSeqNode = upSeqEdge ? nodeById.get(upSeqEdge.fromNode) : null;
      const upStructNode = upStructEdge ? nodeById.get(upStructEdge.fromNode) : null;
      const upInfo = upSeqNode
        ? sequencesOut(upSeqNode, visiting)
        : upStructNode
          ? (() => {
              const bb = backbonesOut(upStructNode, visiting);
              return { backbones: bb.count, sequences: 1, length: bb.length, isEstimate: bb.isEstimate };
            })()
          : { backbones: 1, sequences: 1, length: null, isEstimate: true };
      const cap = resolveEffectiveMaxSequences(node.adapterId, node.params, {
        sequenceLength: upInfo.length,
      });
      res = {
        backbones: upInfo.backbones,
        sequences: Math.max(1, Math.min(cap.limit, upInfo.sequences)),
        length: upInfo.length,
        isEstimate: upInfo.isEstimate,
      };
    } else {
      const upEdge = findEdge(node.id, 'sequences');
      const upNode = upEdge ? nodeById.get(upEdge.fromNode) : null;
      res = upNode
        ? sequencesOut(upNode, visiting)
        : { backbones: 1, sequences: 1, length: null, isEstimate: true };
    }

    visiting.delete(node.id);
    seqMemo.set(node.id, res);
    return res;
  };

  const nodes = new Map<string, NodeFlowInfo>();
  const edges = new Map<string, EdgeFlowInfo>();
  const downstreamImpactsBySource = new Map<string, DownstreamImpactNotice[]>();

  const recordEdge = (
    edge: { fromNode: string; fromPort: string; toNode: string; toPort: string },
    incomingCount: number,
    processedCount: number,
    droppedCount: number,
    dropKind: FlowDropKind,
    isEstimate: boolean,
    unit: 'seq' | 'fold' | 'struct',
  ) => {
    const eid = makeEdgeId(edge.fromNode, edge.fromPort, edge.toNode, edge.toPort);
    const prefix = isEstimate ? '~' : '';
    const uPlural =
      unit === 'seq'
        ? incomingCount === 1
          ? 'seq'
          : 'seqs'
        : unit === 'fold'
          ? incomingCount === 1
            ? 'fold'
            : 'folds'
          : incomingCount === 1
            ? 'structure'
            : 'structures';
    const isTruncated = droppedCount > 0 && dropKind !== 'none' && dropKind !== 'intentional-filter';
    const hoverLabel =
      droppedCount > 0
        ? dropKind === 'intentional-filter'
          ? `${prefix}${incomingCount} ${uPlural} → top ${processedCount} kept`
          : `${prefix}${incomingCount} ${uPlural} → ${processedCount} kept (${droppedCount} skipped)`
        : `${prefix}${incomingCount} ${uPlural}`;

    edges.set(eid, {
      edgeId: eid,
      fromNode: edge.fromNode,
      toNode: edge.toNode,
      toPort: edge.toPort,
      incomingCount,
      processedCount,
      droppedCount,
      dropKind,
      isEstimate,
      isTruncated,
      hoverLabel,
    });
  };

  const pushDownstreamImpact = (
    fromNodeId: string,
    targetNode: WorkflowNode,
    info: NodeFlowInfo,
  ) => {
    if (
      info.droppedCount <= 0 ||
      info.dropKind === 'none' ||
      info.dropKind === 'intentional-filter'
    ) {
      return;
    }
    const targetLabel =
      targetNode.label?.trim() ||
      ADAPTER_REGISTRY[targetNode.adapterId]?.shortName ||
      targetNode.adapterId;
    const verb =
      targetNode.adapterId === 'localfold-af2'
        ? 'fold'
        : targetNode.adapterId === 'esmc' || targetNode.adapterId === 'dyna1'
          ? 'score'
          : 'use';
    const tilde = info.isEstimate ? '~' : '';
    const reasonTag =
      info.dropKind === 'cap-campaign'
        ? `campaign cap ${info.maxTotalFolds ?? DEFAULT_MAX_TOTAL_FOLDS}`
        : info.dropKind === 'latency-budget'
          ? 'masked pass budget'
          : info.dropKind === 'single-item-port'
            ? 'first sequence only'
            : info.dropKind === 'collection-collapse'
              ? 'representative structure only'
              : info.capMode === 'fixed'
                ? `fixed cap ${info.effectiveCap}`
                : `max ${info.effectiveCap}/backbone`;
    const list = downstreamImpactsBySource.get(fromNodeId) ?? [];
    list.push({
      targetNodeId: targetNode.id,
      targetLabel,
      targetAdapterId: targetNode.adapterId,
      incomingCount: info.incomingCount,
      processedCount: info.processedCount,
      droppedCount: info.droppedCount,
      dropKind: info.dropKind,
      isEstimate: info.isEstimate,
      message: `${targetLabel} will ${verb} ${tilde}${info.processedCount} of ${info.incomingCount} (${reasonTag})`,
    });
    downstreamImpactsBySource.set(fromNodeId, list);
  };

  for (const node of graph.nodes) {
    if (node.adapterId === 'localfold-af2') {
      const upEdge = findEdge(node.id, 'sequences');
      if (!upEdge) continue;
      const upNode = nodeById.get(upEdge.fromNode);
      if (!upNode) continue;
      const upInfo = sequencesOut(upNode);
      const cap = resolveEffectiveMaxSequences('localfold-af2', node.params);
      const maxTotalFolds = readClampedInt(
        node.params?.maxTotalFolds,
        DEFAULT_MAX_TOTAL_FOLDS,
        1,
        MAX_TOTAL_FOLDS_CAP,
      );

      // Prefer materialized FoldedStructure counts if this node already ran
      const foldOut = outputsByNode?.[node.id]?.folds as FoldedStructure | undefined;
      let processedCount: number;
      let droppedCount: number;
      let isEstimate = upInfo.isEstimate;
      let incomingCount = upInfo.sequences;
      let dropKind: FlowDropKind = 'none';

      if (foldOut && foldOut.kind === 'FoldedStructure' && foldOut.folds.length > 0 && !upInfo.isEstimate) {
        processedCount = foldOut.folds.length;
        const excludedLen = foldOut.excludedCandidates?.length ?? 0;
        if (excludedLen > 0) {
          incomingCount = Math.max(incomingCount, processedCount + excludedLen);
        }
        droppedCount = Math.max(0, incomingCount - processedCount);
        isEstimate = false;
      } else {
        if (upInfo.backbones > 1) {
          const perBb = Math.min(cap.limit, Math.ceil(upInfo.sequences / upInfo.backbones));
          processedCount = Math.min(maxTotalFolds, upInfo.backbones * perBb, upInfo.sequences);
        } else {
          processedCount = Math.min(cap.limit, upInfo.sequences);
        }
        processedCount = Math.max(1, processedCount);
        droppedCount = Math.max(0, incomingCount - processedCount);
      }

      if (droppedCount > 0) {
        if (upInfo.backbones > 1 && processedCount === maxTotalFolds && upInfo.backbones * cap.limit > maxTotalFolds) {
          dropKind = 'cap-campaign';
        } else {
          dropKind = 'cap-per-backbone';
        }
      }

      const tilde = isEstimate ? '~' : '';
      let badgeText: string | null = null;
      let detailText: string | null = null;
      let severity: 'warning' | 'info' | null = null;
      if (droppedCount > 0) {
        severity = 'warning';
        if (dropKind === 'cap-campaign') {
          badgeText = `⚠ ${tilde}${processedCount} of ${incomingCount} folds (cap ${maxTotalFolds})`;
          detailText = `Campaign cap (maxTotalFolds=${maxTotalFolds}) limits ${incomingCount} upstream sequences across ${upInfo.backbones} backbones to ${processedCount} folds.`;
        } else if (cap.capMode === 'fixed') {
          badgeText = `⚠ ${tilde}${processedCount} of ${incomingCount} seqs (cap ${cap.limit})`;
          detailText = `Fixed cap (maxSequences=${cap.limit}) folds ${processedCount} of ${incomingCount} upstream sequences (${droppedCount} skipped). Switch to Follow upstream to fold all.`;
        } else {
          badgeText = `⚠ ${tilde}${processedCount} of ${incomingCount} seqs (max ${cap.limit}/bb)`;
          detailText = `Per-backbone limit (${cap.limit}/backbone) caps ${incomingCount} incoming sequences to ${processedCount} folds.`;
        }
      }

      const info: NodeFlowInfo = {
        nodeId: node.id,
        adapterId: node.adapterId,
        incomingCount,
        incomingBackbones: upInfo.backbones,
        processedCount,
        droppedCount,
        dropKind,
        isEstimate,
        effectiveCap: cap.limit,
        maxTotalFolds,
        capMode: cap.capMode,
        severity,
        badgeText,
        detailText,
      };
      nodes.set(node.id, info);
      recordEdge(upEdge, incomingCount, processedCount, droppedCount, dropKind, isEstimate, 'seq');
      pushDownstreamImpact(upEdge.fromNode, node, info);
      continue;
    }

    if (node.adapterId === 'esmc' || node.adapterId === 'dyna1') {
      const upSeqEdge = findEdge(node.id, 'sequences');
      const upStructEdge = findEdge(node.id, 'structure');
      const structOnlyDyna =
        node.adapterId === 'dyna1' && String(node.params?.inputMode ?? 'auto') === 'struct-only';

      if (upSeqEdge && !structOnlyDyna) {
        const upNode = nodeById.get(upSeqEdge.fromNode);
        if (upNode) {
          const upInfo = sequencesOut(upNode);
          const cap = resolveEffectiveMaxSequences(node.adapterId, node.params, {
            sequenceLength: upInfo.length,
          });
          const incomingCount = upInfo.sequences;
          const processedCount = Math.max(1, Math.min(cap.limit, incomingCount));
          const droppedCount = Math.max(0, incomingCount - processedCount);
          const dropKind: FlowDropKind =
            droppedCount > 0
              ? cap.isPassBudgetCapped
                ? 'latency-budget'
                : 'cap-per-backbone'
              : 'none';
          const tilde = upInfo.isEstimate ? '~' : '';
          let badgeText: string | null = null;
          let detailText: string | null = null;
          let severity: 'warning' | 'info' | null = null;
          if (droppedCount > 0) {
            severity = 'warning';
            if (dropKind === 'latency-budget') {
              badgeText = `⚠ ${tilde}${processedCount} of ${incomingCount} seqs (masked budget)`;
              detailText = cap.unknownLengthFallback
                ? `Masked LLR requires L forward passes per sequence; auto-capped at 1 pre-run until length is known.`
                : `Masked LLR pass budget (${ESMC_MASKED_PASS_BUDGET} passes) caps ${incomingCount} sequences at ${processedCount} (L=${upInfo.length}).`;
            } else if (cap.capMode === 'fixed') {
              badgeText = `⚠ ${tilde}${processedCount} of ${incomingCount} seqs (cap ${cap.limit})`;
              detailText = `Fixed cap (maxSequences=${cap.limit}) scores ${processedCount} of ${incomingCount} upstream sequences (${droppedCount} skipped).`;
            } else {
              badgeText = `⚠ ${tilde}${processedCount} of ${incomingCount} seqs (max ${cap.limit})`;
              detailText = `Sequence limit (${cap.limit}) caps ${incomingCount} incoming sequences to ${processedCount}.`;
            }
          }
          const info: NodeFlowInfo = {
            nodeId: node.id,
            adapterId: node.adapterId,
            incomingCount,
            incomingBackbones: upInfo.backbones,
            processedCount,
            droppedCount,
            dropKind,
            isEstimate: upInfo.isEstimate,
            effectiveCap: cap.limit,
            capMode: cap.capMode,
            severity,
            badgeText,
            detailText,
          };
          nodes.set(node.id, info);
          recordEdge(upSeqEdge, incomingCount, processedCount, droppedCount, dropKind, upInfo.isEstimate, 'seq');
          pushDownstreamImpact(upSeqEdge.fromNode, node, info);
        }
      } else if (upStructEdge) {
        const upNode = nodeById.get(upStructEdge.fromNode);
        if (upNode) {
          const bb = backbonesOut(upNode);
          const cap = resolveEffectiveMaxSequences(node.adapterId, node.params, {
            sequenceLength: bb.length,
          });
          const incomingCount = bb.count;
          const processedCount = Math.max(1, Math.min(cap.limit, incomingCount));
          const droppedCount = Math.max(0, incomingCount - processedCount);
          const dropKind: FlowDropKind =
            droppedCount > 0
              ? cap.isPassBudgetCapped
                ? 'latency-budget'
                : 'cap-per-backbone'
              : 'none';
          const tilde = bb.isEstimate ? '~' : '';
          let badgeText: string | null = null;
          let detailText: string | null = null;
          let severity: 'warning' | 'info' | null = null;
          if (droppedCount > 0) {
            severity = 'warning';
            if (dropKind === 'latency-budget') {
              badgeText = `⚠ ${tilde}${processedCount} of ${incomingCount} structs (masked budget)`;
              detailText = cap.unknownLengthFallback
                ? `Masked LLR requires L forward passes per structure; auto-capped at 1 pre-run until length is known.`
                : `Masked LLR pass budget (${ESMC_MASKED_PASS_BUDGET} passes) caps ${incomingCount} structures at ${processedCount} (L=${bb.length}).`;
            } else if (cap.capMode === 'fixed') {
              badgeText = `⚠ ${tilde}${processedCount} of ${incomingCount} structs (cap ${cap.limit})`;
              detailText = `Fixed cap (maxSequences=${cap.limit}) scores ${processedCount} of ${incomingCount} upstream structures (${droppedCount} skipped).`;
            } else {
              badgeText = `⚠ ${tilde}${processedCount} of ${incomingCount} structs (max ${cap.limit})`;
              detailText = `Batch limit (${cap.limit}) caps ${incomingCount} incoming structures to ${processedCount}.`;
            }
          }
          const info: NodeFlowInfo = {
            nodeId: node.id,
            adapterId: node.adapterId,
            incomingCount,
            incomingBackbones: bb.count,
            processedCount,
            droppedCount,
            dropKind,
            isEstimate: bb.isEstimate,
            effectiveCap: cap.limit,
            capMode: cap.capMode,
            severity,
            badgeText,
            detailText,
          };
          nodes.set(node.id, info);
          recordEdge(upStructEdge, incomingCount, processedCount, droppedCount, dropKind, bb.isEstimate, 'struct');
          pushDownstreamImpact(upStructEdge.fromNode, node, info);
        }
      }
      continue;
    }

    if (node.adapterId === 'top-k-filter') {
      const upEdge = findEdge(node.id, 'sequences');
      if (!upEdge) continue;
      const upNode = nodeById.get(upEdge.fromNode);
      if (!upNode) continue;
      const upInfo = sequencesOut(upNode);
      const outInfo = sequencesOut(node);
      const droppedCount = Math.max(0, upInfo.sequences - outInfo.sequences);
      const dropKind: FlowDropKind = droppedCount > 0 ? 'intentional-filter' : 'none';
      nodes.set(node.id, {
        nodeId: node.id,
        adapterId: node.adapterId,
        incomingCount: upInfo.sequences,
        incomingBackbones: upInfo.backbones,
        processedCount: outInfo.sequences,
        droppedCount,
        dropKind,
        isEstimate: upInfo.isEstimate,
        effectiveCap: outInfo.sequences,
        severity: null,
        badgeText: null,
        detailText: null,
      });
      recordEdge(
        upEdge,
        upInfo.sequences,
        outInfo.sequences,
        droppedCount,
        dropKind,
        upInfo.isEstimate,
        'seq',
      );
      continue;
    }

    // Single-sequence consumers (`foldflow2`, `evodiff` infill)
    if (
      node.adapterId === 'foldflow2' ||
      (node.adapterId === 'evodiff' && String(node.params?.taskMode ?? 'unconditional') === 'infill')
    ) {
      const upEdge = findEdge(node.id, 'sequences');
      if (!upEdge) continue;
      const upNode = nodeById.get(upEdge.fromNode);
      if (!upNode) continue;
      const upInfo = sequencesOut(upNode);
      if (upInfo.sequences > 1) {
        const info: NodeFlowInfo = {
          nodeId: node.id,
          adapterId: node.adapterId,
          incomingCount: upInfo.sequences,
          incomingBackbones: upInfo.backbones,
          processedCount: 1,
          droppedCount: upInfo.sequences - 1,
          dropKind: 'single-item-port',
          isEstimate: upInfo.isEstimate,
          effectiveCap: 1,
          severity: 'info',
          badgeText: `ℹ Uses seq #1 of ${upInfo.sequences} only`,
          detailText: `${ADAPTER_REGISTRY[node.adapterId]?.shortName ?? node.adapterId} reads only candidate #1 from an incoming multi-sequence set (${upInfo.sequences - 1} ignored).`,
        };
        nodes.set(node.id, info);
        recordEdge(
          upEdge,
          upInfo.sequences,
          1,
          upInfo.sequences - 1,
          'single-item-port',
          upInfo.isEstimate,
          'seq',
        );
        pushDownstreamImpact(upEdge.fromNode, node, info);
      }
      continue;
    }

    // Representative-structure-only consumers (`thermompnn`)
    if (node.adapterId === 'thermompnn') {
      const upEdge = findEdge(node.id, 'structure');
      if (!upEdge) continue;
      const upNode = nodeById.get(upEdge.fromNode);
      if (!upNode) continue;
      const bb = backbonesOut(upNode);
      if (bb.count > 1) {
        const info: NodeFlowInfo = {
          nodeId: node.id,
          adapterId: node.adapterId,
          incomingCount: bb.count,
          incomingBackbones: bb.count,
          processedCount: 1,
          droppedCount: bb.count - 1,
          dropKind: 'collection-collapse',
          isEstimate: bb.isEstimate,
          effectiveCap: 1,
          severity: 'info',
          badgeText: `ℹ Scores rep #1 of ${bb.count}`,
          detailText: `${ADAPTER_REGISTRY[node.adapterId]?.shortName ?? node.adapterId} evaluates only the representative structure (#1 of ${bb.count}).`,
        };
        nodes.set(node.id, info);
        recordEdge(
          upEdge,
          bb.count,
          1,
          bb.count - 1,
          'collection-collapse',
          bb.isEstimate,
          'struct',
        );
        pushDownstreamImpact(upEdge.fromNode, node, info);
      }
      continue;
    }

    // Record non-truncating edge counts on standard sequence/structure wires for hover telemetry
    for (const edge of graph.edges) {
      if (edge.toNode !== node.id) continue;
      const eid = makeEdgeId(edge.fromNode, edge.fromPort, edge.toNode, edge.toPort);
      if (edges.has(eid)) continue;
      const upNode = nodeById.get(edge.fromNode);
      if (!upNode) continue;
      if (edge.toPort === 'sequences') {
        const upInfo = sequencesOut(upNode);
        if (upInfo.sequences > 1) {
          recordEdge(edge, upInfo.sequences, upInfo.sequences, 0, 'none', upInfo.isEstimate, 'seq');
        }
      } else if (edge.toPort === 'structure' || edge.toPort === 'template' || edge.toPort === 'structures') {
        const bb = backbonesOut(upNode);
        if (bb.count > 1) {
          const isFanoutConsumer =
            INVERSE_FOLD_PRODUCERS.has(node.adapterId) ||
            node.adapterId === 'designability-scorer' ||
            node.adapterId === 'gnina' ||
            node.adapterId === 'structure-compare';
          if (isFanoutConsumer) {
            recordEdge(edge, bb.count, bb.count, 0, 'none', bb.isEstimate, 'struct');
          }
        }
      }
    }
  }

  return { nodes, edges, downstreamImpactsBySource };
}
