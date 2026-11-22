// Preflight estimator for first-run confirmation dialog: calculates deduplicated
// model weight downloads, peak VRAM (sequential VRAM-mutex scheduling), WebGPU use,
// and honest multi-design campaign fan-out / sequential GPU work counts.

import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS, RFD4_BENCHMARK_OPTIONS } from './registry';
import { validateSequenceText } from './param-validation';
import type { AdapterManifest, WorkflowGraph, WorkflowNode } from './types';
import { MAX_DESIGNS, MAX_SEQUENCES_PER_DESIGN } from './work-plan';

export interface PreflightModelItem {
  key: string;
  label: string;
  approxMB: number;
  estimatedVramMB: number;
  runtime: AdapterManifest['runtime'];
  nonCommercial?: boolean;
  /** Number of sequential GPU work items planned for this model (weights downloaded & loaded once). */
  plannedItems?: number;
}

export interface RemoteMsaTransferNotice {
  nodeId: string;
  nodeLabel: string;
  mode: 'mmseqs2' | 'target-only';
  destination: string;
}

export interface RunPreflightSummary {
  models: PreflightModelItem[];
  totalDownloadMB: number;
  peakVramMB: number;
  usesWebGpu: boolean;
  hasNonCommercialModel: boolean;
  /** External sequence transfers required by remote ColabFold MMseqs2 MSA nodes in this run. */
  remoteMsaTransfers?: RemoteMsaTransferNotice[];
  /** Total planned backbones across generators. */
  plannedBackbones?: number;
  /** Total planned sequences across inverse-folding / native generator outputs. */
  plannedSequences?: number;
  /** Total planned refolds across LocalFold stages. */
  plannedFolds?: number;
  /** Total sequential GPU work items across GPU stages (weights are never multiplied by numDesigns). */
  totalGpuWorkItems?: number;
  /** Honest sequential compute multiplier vs a 1-design baseline (weights stay 1x). */
  estimatedComputeScale?: number;
  /** True when weights are downloaded & loaded once per stage across sequential B=1 items. */
  weightsLoadedOnce?: boolean;
  /** Human-readable multi-design fan-out summary when numDesigns > 1 (e.g. "4 backbones → 8 sequences → 4 refolds"). */
  fanoutSummary?: string;
}

const MAX_TOTAL_FOLDS_CAP = 64;
const DEFAULT_MAX_TOTAL_FOLDS = 16;

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

const MULTI_DESIGN_GENERATORS = new Set([
  'genie3',
  'rfd4-proteina',
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

function readNumDesigns(node: WorkflowNode, forceSingle = false): number {
  if (forceSingle) return 1;
  if (!MULTI_DESIGN_GENERATORS.has(node.adapterId)) return 1;
  return readClampedInt(node.params?.numDesigns, 1, 1, MAX_DESIGNS);
}

function countFastaSequences(raw: unknown): number {
  if (typeof raw !== 'string') return 1;
  const trimmed = raw.trim();
  if (!trimmed) return 1;
  const check = validateSequenceText(trimmed);
  return Math.max(1, check.numSequences);
}

interface GraphWorkloadEstimate {
  hasMultiDesign: boolean;
  plannedBackbones: number;
  plannedSequences: number;
  plannedSelectedSequences: number | null;
  plannedFolds: number;
  hasMpnn: boolean;
  hasTopKFilter: boolean;
  hasLocalFold: boolean;
  nodeWorkItems: Map<string, number>;
  totalGpuWorkItems: number;
}

function estimateGraphWorkload(
  graph: WorkflowGraph,
  forceSingleDesign = false,
): GraphWorkloadEstimate {
  const nodeById = new Map<string, WorkflowNode>();
  for (const n of graph.nodes) nodeById.set(n.id, n);

  const findUpstreamNode = (nodeId: string, toPort: string): WorkflowNode | null => {
    const edge = graph.edges.find((e) => e.toNode === nodeId && e.toPort === toPort);
    return edge ? (nodeById.get(edge.fromNode) ?? null) : null;
  };

  const backboneMemo = new Map<string, number>();
  const seqMemo = new Map<string, { backbones: number; sequences: number }>();

  const backbonesOut = (node: WorkflowNode, visiting = new Set<string>()): number => {
    const cached = backboneMemo.get(node.id);
    if (cached !== undefined) return cached;
    if (visiting.has(node.id)) return 1;
    visiting.add(node.id);

    let count = 1;
    if (MULTI_DESIGN_GENERATORS.has(node.adapterId)) {
      count = readNumDesigns(node, forceSingleDesign);
    } else if (node.adapterId === 'pdb-input' || node.adapterId === 'localfold-af2') {
      count = 1;
    } else {
      const up = findUpstreamNode(node.id, 'structure') ?? findUpstreamNode(node.id, 'template');
      count = up ? backbonesOut(up, visiting) : 1;
    }
    visiting.delete(node.id);
    backboneMemo.set(node.id, count);
    return count;
  };

  const sequencesOut = (
    node: WorkflowNode,
    visiting = new Set<string>(),
  ): { backbones: number; sequences: number } => {
    const cached = seqMemo.get(node.id);
    if (cached !== undefined) return cached;
    if (visiting.has(node.id)) return { backbones: 1, sequences: 1 };
    visiting.add(node.id);

    let res: { backbones: number; sequences: number };
    if (MULTI_DESIGN_GENERATORS.has(node.adapterId)) {
      const n = readNumDesigns(node, forceSingleDesign);
      res = { backbones: n, sequences: n };
    } else if (
      node.adapterId === 'web-protein-mpnn' ||
      node.adapterId === 'ligandmpnn' ||
      node.adapterId === 'lasermpnn' ||
      node.adapterId === 'antifold-esmif1'
    ) {
      const up = findUpstreamNode(node.id, 'structure');
      const n = up ? backbonesOut(up, visiting) : 1;
      const k = readClampedInt(node.params?.numSequences, 2, 1, MAX_SEQUENCES_PER_DESIGN);
      res = { backbones: n, sequences: n * k };
    } else if (node.adapterId === 'top-k-filter') {
      const up = findUpstreamNode(node.id, 'sequences');
      const upInfo = up ? sequencesOut(up, visiting) : { backbones: 1, sequences: 1 };
      const mode = String(node.params?.mode ?? 'per-backbone');
      const topK = readClampedInt(node.params?.topK, 2, 1, 16);
      const maxTotal = readClampedInt(node.params?.maxTotal, 8, 1, MAX_TOTAL_FOLDS_CAP);
      let kept: number;
      if (mode === 'global') {
        kept = Math.min(upInfo.sequences, topK, maxTotal);
      } else {
        const perBackboneAvail =
          upInfo.backbones > 0 ? Math.ceil(upInfo.sequences / upInfo.backbones) : upInfo.sequences;
        kept = Math.min(
          maxTotal,
          upInfo.backbones * Math.min(perBackboneAvail, topK),
          upInfo.sequences,
        );
      }
      res = { backbones: upInfo.backbones, sequences: Math.max(0, kept) };
    } else if (node.adapterId === 'sequence-input') {
      res = { backbones: 1, sequences: countFastaSequences(node.params?.sequence) };
    } else {
      const up = findUpstreamNode(node.id, 'sequences');
      res = up ? sequencesOut(up, visiting) : { backbones: 1, sequences: 1 };
    }

    visiting.delete(node.id);
    seqMemo.set(node.id, res);
    return res;
  };

  let hasMultiDesign = false;
  let generatorBackbones = 0;
  let generatorCount = 0;
  let mpnnSequences = 0;
  let hasMpnn = false;
  let hasTopKFilter = false;
  let topKSelectedSequences = 0;
  let hasLocalFold = false;
  let plannedFolds = 0;

  const nodeWorkItems = new Map<string, number>();
  let totalGpuWorkItems = 0;

  for (const node of graph.nodes) {
    const manifest = ADAPTER_REGISTRY[node.adapterId];
    const isGpu = Boolean(manifest && manifest.runtime !== 'cpu-ts');

    if (MULTI_DESIGN_GENERATORS.has(node.adapterId)) {
      const n = readNumDesigns(node, forceSingleDesign);
      generatorCount++;
      generatorBackbones += n;
      if (n > 1) hasMultiDesign = true;
      nodeWorkItems.set(node.id, n);
      if (isGpu) totalGpuWorkItems += n;
      continue;
    }

    if (
      node.adapterId === 'web-protein-mpnn' ||
      node.adapterId === 'ligandmpnn' ||
      node.adapterId === 'lasermpnn' ||
      node.adapterId === 'antifold-esmif1'
    ) {
      hasMpnn = true;
      const out = sequencesOut(node);
      mpnnSequences += out.sequences;
      nodeWorkItems.set(node.id, out.sequences);
      if (isGpu) totalGpuWorkItems += out.sequences;
      continue;
    }

    if (node.adapterId === 'top-k-filter') {
      hasTopKFilter = true;
      const out = sequencesOut(node);
      topKSelectedSequences += out.sequences;
      nodeWorkItems.set(node.id, out.sequences);
      continue;
    }

    if (node.adapterId === 'localfold-af2') {
      hasLocalFold = true;
      const up = findUpstreamNode(node.id, 'sequences');
      const upInfo = up ? sequencesOut(up) : { backbones: 1, sequences: 1 };
      const maxSeqs = readClampedInt(node.params?.maxSequences, 2, 1, 8);
      const maxTotalFolds = readClampedInt(
        node.params?.maxTotalFolds,
        DEFAULT_MAX_TOTAL_FOLDS,
        1,
        MAX_TOTAL_FOLDS_CAP,
      );
      let foldsForNode: number;
      if (upInfo.backbones > 1) {
        const perBackbone = Math.min(maxSeqs, Math.ceil(upInfo.sequences / upInfo.backbones));
        foldsForNode = Math.min(maxTotalFolds, upInfo.backbones * perBackbone, upInfo.sequences);
      } else {
        foldsForNode = Math.min(maxSeqs, upInfo.sequences);
      }
      foldsForNode = Math.max(1, foldsForNode);
      plannedFolds += foldsForNode;
      nodeWorkItems.set(node.id, foldsForNode);
      if (isGpu) totalGpuWorkItems += foldsForNode;
      continue;
    }

    if (isGpu) {
      const upStruct = findUpstreamNode(node.id, 'structure');
      const upSeq = findUpstreamNode(node.id, 'sequences');
      const items = upStruct
        ? backbonesOut(upStruct)
        : upSeq
          ? sequencesOut(upSeq).sequences
          : 1;
      nodeWorkItems.set(node.id, items);
      totalGpuWorkItems += items;
    }
  }

  const plannedBackbones = generatorCount > 0 ? generatorBackbones : 0;
  // When MPNN is present, report MPNN's planned sequence total; otherwise if a generator feeds
  // LocalFold directly (e.g. RFD4 native sequences -> LocalFold), report the native sequence count.
  const plannedSequences = hasMpnn
    ? mpnnSequences
    : generatorCount > 0
      ? generatorBackbones
      : 0;
  const plannedSelectedSequences =
    hasTopKFilter && topKSelectedSequences < plannedSequences ? topKSelectedSequences : null;

  return {
    hasMultiDesign,
    plannedBackbones,
    plannedSequences,
    plannedSelectedSequences,
    plannedFolds,
    hasMpnn,
    hasTopKFilter,
    hasLocalFold,
    nodeWorkItems,
    totalGpuWorkItems,
  };
}

function buildFanoutSummary(w: GraphWorkloadEstimate): string | undefined {
  if (!w.hasMultiDesign || w.plannedBackbones <= 1) return undefined;
  const parts: string[] = [
    `${w.plannedBackbones} ${w.plannedBackbones === 1 ? 'backbone' : 'backbones'}`,
  ];
  if (w.hasMpnn || (w.hasLocalFold && w.plannedSequences > 0)) {
    parts.push(`${w.plannedSequences} ${w.plannedSequences === 1 ? 'sequence' : 'sequences'}`);
  }
  if (w.plannedSelectedSequences !== null) {
    parts.push(`${w.plannedSelectedSequences} selected`);
  }
  if (w.hasLocalFold && w.plannedFolds > 0) {
    parts.push(`${w.plannedFolds} ${w.plannedFolds === 1 ? 'refold' : 'refolds'}`);
  }
  return parts.join(' → ');
}

function rfd4NeedsEncoder(graph: WorkflowGraph, nodeId: string): boolean {
  const node = graph.nodes.find((n) => n.id === nodeId);
  if (!node) return false;
  const mode = String(node.params.mode ?? '');
  if (mode === 'partial' || mode === 'inverse-folding') return true;
  if (mode === 'preset') {
    const presetId = String(node.params.presetId ?? '');
    if (RFD4_BENCHMARK_OPTIONS.find((b) => b.id === presetId)?.needsEncoder) {
      return true;
    }
  }

  // Check if a connected rfd4-conditioner node selects a benchmark that requires the encoder
  const condEdge = graph.edges.find((e) => e.toNode === nodeId && e.toPort === 'condition');
  if (condEdge) {
    const condNode = graph.nodes.find((n) => n.id === condEdge.fromNode);
    if (condNode?.adapterId === 'rfd4-conditioner') {
      const taskPreset = String(condNode.params.taskPreset ?? '');
      const bm = RFD4_BENCHMARK_OPTIONS.find((b) => b.id === taskPreset);
      if (bm?.needsEncoder) return true;
    }
  }
  return false;
}

const AF2_DELTA_BASE: Record<string, { baseId: string; baseMB: number }> = {
  'monomer-2': { baseId: 'monomer', baseMB: 97 },
  'monomer-3': { baseId: 'monomer', baseMB: 97 },
  'monomer-4': { baseId: 'monomer', baseMB: 97 },
  'monomer-5': { baseId: 'monomer', baseMB: 97 },
  'multimer-2': { baseId: 'multimer', baseMB: 74 },
  'multimer-3': { baseId: 'multimer', baseMB: 74 },
  'multimer-4': { baseId: 'multimer', baseMB: 74 },
  'multimer-5': { baseId: 'multimer', baseMB: 74 },
};

export function estimateRunRequirements(graph: WorkflowGraph): RunPreflightSummary {
  const byKey = new Map<string, PreflightModelItem>();
  const af2ModelsInGraph = new Set<string>();
  for (const node of graph.nodes) {
    if (node.adapterId === 'localfold-af2') {
      af2ModelsInGraph.add(String(node.params.model ?? 'monomer'));
    }
  }
  const countedAf2Bases = new Set<string>();

  const workload = estimateGraphWorkload(graph, false);
  const baselineWorkload = workload.hasMultiDesign
    ? estimateGraphWorkload(graph, true)
    : workload;

  const addPlannedItems = (key: string, items: number) => {
    const entry = byKey.get(key);
    if (!entry) return;
    entry.plannedItems = (entry.plannedItems ?? 0) + items;
  };

  for (const node of graph.nodes) {
    const manifest = ADAPTER_REGISTRY[node.adapterId];
    if (!manifest) continue;
    if (manifest.runtime === 'cpu-ts' && manifest.approxWeightsMB === 0) continue;

    const nodeItems = workload.nodeWorkItems.get(node.id) ?? 1;

    if (node.adapterId === 'localfold-af2') {
      const modelId = String(node.params.model ?? 'monomer');
      const meta =
        LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
      const key = `localfold:${meta.id}`;
      if (!byKey.has(key)) {
        let approxMB = meta.approxMB;
        const deltaInfo = AF2_DELTA_BASE[meta.id];
        if (deltaInfo) {
          // Delta checkpoints share their base weights (monomer = 97 MB, multimer = 74 MB).
          // If the base checkpoint itself is in the graph, or another delta already counted the base,
          // only count the marginal delta bytes here.
          if (af2ModelsInGraph.has(deltaInfo.baseId) || countedAf2Bases.has(deltaInfo.baseId)) {
            approxMB = Math.max(0, meta.approxMB - deltaInfo.baseMB);
          } else {
            countedAf2Bases.add(deltaInfo.baseId);
          }
        } else if (meta.id === 'monomer' || meta.id === 'multimer') {
          countedAf2Bases.add(meta.id);
        }
        byKey.set(key, {
          key,
          label: meta.label,
          approxMB,
          estimatedVramMB:
            meta.family === 'af3'
              ? 1800
              : meta.family === 'af2-multimer'
                ? 1500
                : manifest.estimatedVramMB,
          runtime: manifest.runtime,
          nonCommercial: meta.nonCommercial,
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    if (node.adapterId === 'rfd4-proteina') {
      const needsEnc = rfd4NeedsEncoder(graph, node.id);
      const baseKey = 'rfd4-proteina';
      if (!byKey.has(baseKey)) {
        byKey.set(baseKey, {
          key: baseKey,
          label: manifest.displayName,
          approxMB: manifest.approxWeightsMB,
          estimatedVramMB: manifest.estimatedVramMB,
          runtime: manifest.runtime,
          plannedItems: 0,
        });
      }
      addPlannedItems(baseKey, nodeItems);
      if (needsEnc) {
        const encKey = 'rfd4-proteina:encoder';
        if (!byKey.has(encKey)) {
          byKey.set(encKey, {
            key: encKey,
            label: 'RFD4-Proteina Structure Encoder (Partial / Inverse Folding)',
            approxMB: 138,
            estimatedVramMB: 400,
            runtime: manifest.runtime,
            plannedItems: 0,
          });
        }
        addPlannedItems(encKey, nodeItems);
      }
      continue;
    }

    if (node.adapterId === 'web-protein-mpnn') {
      const family = String(node.params.family ?? 'solublempnn');
      const modelName = String(node.params.modelName ?? 'v_48_020');
      const familyLabel =
        family === 'solublempnn'
          ? 'SolubleMPNN'
          : family === 'ca'
            ? 'Cα-only ProteinMPNN'
            : 'ProteinMPNN';
      const key = `mpnn:${family}:${modelName}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: `${familyLabel} (${modelName})`,
          approxMB: manifest.approxWeightsMB,
          estimatedVramMB: manifest.estimatedVramMB,
          runtime: manifest.runtime,
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    if (node.adapterId === 'thermompnn') {
      const precision = String(node.params.precision ?? 'fp16');
      const approxMB = precision === 'int8' ? 1.9 : precision === 'fp32' ? 7.4 : 3.7;
      const key = `thermompnn:${precision}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: `ThermoMPNN (${precision.toUpperCase()})`,
          approxMB,
          estimatedVramMB: manifest.estimatedVramMB,
          runtime: manifest.runtime,
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    if (node.adapterId === 'ligandmpnn') {
      const checkpoint = String(node.params.checkpoint ?? 'ligandmpnn_v_32_020_25');
      const precision = String(node.params.precision ?? 'fp16');
      const approxMB = precision === 'int8' ? 2.8 : precision === 'fp32' ? 10.5 : 5.3;
      const ckptShort = checkpoint.replace(/^ligandmpnn_/, '');
      const key = `ligandmpnn:${checkpoint}:${precision}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: `LigandMPNN (${ckptShort} · ${precision.toUpperCase()})`,
          approxMB,
          estimatedVramMB: manifest.estimatedVramMB,
          runtime: manifest.runtime,
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    if (node.adapterId === 'lasermpnn') {
      const precision = String(node.params.precision ?? 'fp16');
      const approxMB = precision === 'int8' ? 21.1 : precision === 'fp32' ? 82.5 : 41.5;
      const key = `lasermpnn:${precision}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: `LASErMPNN (${precision.toUpperCase()})`,
          approxMB,
          estimatedVramMB: manifest.estimatedVramMB,
          runtime: manifest.runtime,
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    if (node.adapterId === 'antifold-esmif1') {
      const variant = String(node.params.variant ?? 'antifold') === 'esmif1' ? 'esmif1' : 'antifold';
      const precision = String(node.params.precision ?? 'fp16');
      const approxMB = precision === 'int8' ? 137 : precision === 'fp32' ? 567 : 271;
      const estimatedVramMB = precision === 'int8' ? 520 : manifest.estimatedVramMB;
      const variantLabel = variant === 'esmif1' ? 'ESM-IF1' : 'AntiFold';
      const key = `antifold-esmif1:${variant}:${precision}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: `${variantLabel} (${precision.toUpperCase()})`,
          approxMB,
          estimatedVramMB,
          runtime: manifest.runtime,
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    if (node.adapterId === 'protpardelle') {
      const checkpoint = String(node.params.checkpoint ?? 'cc91_epoch383');
      const precision = String(node.params.precision ?? 'int8');
      const sidechain = Boolean(node.params.sidechainMode ?? true);
      const isBb = checkpoint.startsWith('bb');
      const baseMB = isBb ? (precision === 'fp16' ? 71 : 36) : precision === 'fp16' ? 51 : 26;
      const mpnnMB = sidechain && !isBb && checkpoint !== 'cc58_epoch416' && checkpoint !== 'cc95_epoch3490'
        ? (precision === 'fp16' ? 57 : 29)
        : 0;
      const key = `protpardelle:${checkpoint}:${precision}:${sidechain ? 'aa' : 'bb'}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: `ProtPardelle-1c (${checkpoint} · ${precision.toUpperCase()})`,
          approxMB: baseMB + mpnnMB,
          estimatedVramMB: manifest.estimatedVramMB,
          runtime: manifest.runtime,
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    if (node.adapterId === 'esmc') {
      const precision = String(node.params.precision ?? 'int8');
      const approxMB = precision === 'fp16' ? 667 : 335;
      const estimatedVramMB = precision === 'fp16' ? 1150 : manifest.estimatedVramMB;
      const key = `esmc:${precision}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: `ESM-C 300M (${precision.toUpperCase()})`,
          approxMB,
          estimatedVramMB,
          runtime: manifest.runtime,
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    if (node.adapterId === 'dyna1') {
      const precision = String(node.params.precision ?? 'int8');
      const inputMode = String(node.params.inputMode ?? 'auto');
      const hasStructWire = graph.edges.some((e) => e.toNode === node.id && e.toPort === 'structure');
      const needsStructEnc = hasStructWire && inputMode !== 'seq-only';
      const baseMB = precision === 'fp32' ? 3362 : precision === 'fp16' ? 1745 : 844;
      const approxMB = baseMB + (needsStructEnc ? 88 : 0);
      const estimatedVramMB =
        (precision === 'fp32' ? 3900 : precision === 'fp16' ? 2200 : manifest.estimatedVramMB) +
        (needsStructEnc ? 140 : 0);
      const key = `dyna1:${precision}:${needsStructEnc ? 'struct' : 'seq'}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: `Dyna-1 (${precision.toUpperCase()}${needsStructEnc ? ' + VQ-VAE' : ''})`,
          approxMB,
          estimatedVramMB,
          runtime: manifest.runtime,
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    const PRECISION_WEIGHTS: Record<
      string,
      { int8: number; fp16: number; vramFp16?: number; nonCommercial?: boolean }
    > = {
      rfdiffusion: { int8: 69, fp16: 125, vramFp16: 650 },
      boltzgen: { int8: 482, fp16: 950, vramFp16: 2200 },
      'la-proteina': { int8: 295, fp16: 548, vramFp16: 1200 },
      proteina: { int8: 193, fp16: 366, vramFp16: 900, nonCommercial: true },
      'sid-protein': { int8: 193, fp16: 366, vramFp16: 900, nonCommercial: true },
      genie2: { int8: 16, fp16: 31, vramFp16: 320 },
      frameflow: { int8: 15, fp16: 26, vramFp16: 320 },
      foldflow2: { int8: 644, fp16: 1340, vramFp16: 2400, nonCommercial: true },
      evodiff: { int8: 40, fp16: 74, vramFp16: 380 },
    };
    const precSpec = PRECISION_WEIGHTS[node.adapterId];
    if (precSpec) {
      const precision = String(node.params.precision ?? (node.adapterId === 'genie2' ? 'fp16' : 'int8')).toLowerCase();
      const isFp16 = precision === 'fp16';
      const approxMB = isFp16 ? precSpec.fp16 : precSpec.int8;
      const estimatedVramMB = isFp16 && precSpec.vramFp16 ? precSpec.vramFp16 : manifest.estimatedVramMB;
      const key = `${manifest.id}:${isFp16 ? 'fp16' : 'int8'}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: `${manifest.displayName} (${isFp16 ? 'FP16' : 'INT8'})`,
          approxMB,
          estimatedVramMB,
          runtime: manifest.runtime,
          ...(precSpec.nonCommercial ? { nonCommercial: true } : {}),
          plannedItems: 0,
        });
      }
      addPlannedItems(key, nodeItems);
      continue;
    }

    if (!byKey.has(manifest.id)) {
      byKey.set(manifest.id, {
        key: manifest.id,
        label: manifest.displayName,
        approxMB: manifest.approxWeightsMB,
        estimatedVramMB: manifest.estimatedVramMB,
        runtime: manifest.runtime,
        plannedItems: 0,
      });
    }
    addPlannedItems(manifest.id, nodeItems);
  }

  const models = [...byKey.values()];
  // Weights are downloaded & loaded ONCE per stage — never multiplied by numDesigns.
  const totalDownloadMB = models.reduce((sum, m) => sum + m.approxMB, 0);
  let peakVramMB = models.reduce((mx, m) => Math.max(mx, m.estimatedVramMB), 0);
  // RFD4 trunk/decoder (900 MB) and autoencoder encoder (400 MB) are co-resident in the same GPU worker.
  const rfd4Item = byKey.get('rfd4-proteina');
  const rfd4EncItem = byKey.get('rfd4-proteina:encoder');
  if (rfd4Item && rfd4EncItem) {
    peakVramMB = Math.max(peakVramMB, rfd4Item.estimatedVramMB + rfd4EncItem.estimatedVramMB);
  }
  const usesWebGpu = models.some((m) => m.runtime !== 'cpu-ts');
  const hasNonCommercialModel = models.some((m) => m.nonCommercial);
  const fanoutSummary = buildFanoutSummary(workload);
  const estimatedComputeScale =
    baselineWorkload.totalGpuWorkItems > 0
      ? Number((workload.totalGpuWorkItems / baselineWorkload.totalGpuWorkItems).toFixed(2))
      : 1;

  const remoteMsaTransfers: RemoteMsaTransferNotice[] = [];
  for (const node of graph.nodes) {
    if (node.adapterId === 'localfold-af2') {
      const msaMode = String(node.params?.msaMode ?? 'single').trim().toLowerCase();
      if (msaMode === 'mmseqs2' || msaMode === 'target-only') {
        remoteMsaTransfers.push({
          nodeId: node.id,
          nodeLabel: node.label || 'LocalFold',
          mode: msaMode,
          destination: 'api.colabfold.com',
        });
      }
    }
  }

  return {
    models,
    totalDownloadMB,
    peakVramMB,
    usesWebGpu,
    hasNonCommercialModel,
    ...(remoteMsaTransfers.length > 0 ? { remoteMsaTransfers } : {}),
    plannedBackbones: workload.plannedBackbones,
    plannedSequences: workload.plannedSequences,
    plannedFolds: workload.plannedFolds,
    totalGpuWorkItems: workload.totalGpuWorkItems,
    estimatedComputeScale,
    weightsLoadedOnce: true,
    ...(fanoutSummary ? { fanoutSummary } : {}),
  };
}

export function formatApproxSize(mb: number): string {
  if (mb >= 1000) {
    return `~${(mb / 1024).toFixed(1)} GB`;
  }
  if (mb > 0 && mb < 10 && !Number.isInteger(mb)) {
    return `~${mb.toFixed(1)} MB`;
  }
  return `~${Math.round(mb)} MB`;
}
