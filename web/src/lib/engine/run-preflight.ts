// Preflight estimator for first-run confirmation dialog: calculates deduplicated
// model weight downloads, peak VRAM (sequential VRAM-mutex scheduling), and WebGPU use.

import {
  ADAPTER_REGISTRY,
  LOCALFOLD_MODEL_OPTIONS,
  RFD4_BENCHMARK_OPTIONS,
} from './registry';
import type { AdapterManifest, WorkflowGraph } from './types';

export interface PreflightModelItem {
  key: string;
  label: string;
  approxMB: number;
  estimatedVramMB: number;
  runtime: AdapterManifest['runtime'];
  nonCommercial?: boolean;
}

export interface RunPreflightSummary {
  models: PreflightModelItem[];
  totalDownloadMB: number;
  peakVramMB: number;
  usesWebGpu: boolean;
  hasNonCommercialModel: boolean;
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

export function estimateRunRequirements(graph: WorkflowGraph): RunPreflightSummary {
  const byKey = new Map<string, PreflightModelItem>();

  for (const node of graph.nodes) {
    const manifest = ADAPTER_REGISTRY[node.adapterId];
    if (!manifest) continue;
    if (manifest.runtime === 'cpu-ts' && manifest.approxWeightsMB === 0) continue;

    if (node.adapterId === 'localfold-af2') {
      const modelId = String(node.params.model ?? 'monomer');
      const meta =
        LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
      const key = `localfold:${meta.id}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          label: meta.label,
          approxMB: meta.approxMB,
          estimatedVramMB: meta.family === 'af3' ? 1800 : manifest.estimatedVramMB,
          runtime: manifest.runtime,
          nonCommercial: meta.nonCommercial,
        });
      }
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
        });
      }
      if (needsEnc && !byKey.has('rfd4-proteina:encoder')) {
        byKey.set('rfd4-proteina:encoder', {
          key: 'rfd4-proteina:encoder',
          label: 'RFD4-Proteina Structure Encoder (Partial / Inverse Folding)',
          approxMB: 138,
          estimatedVramMB: 400,
          runtime: manifest.runtime,
        });
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
        });
      }
      continue;
    }

    if (!byKey.has(manifest.id)) {
      byKey.set(manifest.id, {
        key: manifest.id,
        label: manifest.displayName,
        approxMB: manifest.approxWeightsMB,
        estimatedVramMB: manifest.estimatedVramMB,
        runtime: manifest.runtime,
      });
    }
  }

  const models = [...byKey.values()];
  const totalDownloadMB = models.reduce((sum, m) => sum + m.approxMB, 0);
  const peakVramMB = models.reduce((mx, m) => Math.max(mx, m.estimatedVramMB), 0);
  const usesWebGpu = models.some((m) => m.runtime !== 'cpu-ts');
  const hasNonCommercialModel = models.some((m) => m.nonCommercial);

  return {
    models,
    totalDownloadMB,
    peakVramMB,
    usesWebGpu,
    hasNonCommercialModel,
  };
}

export function formatApproxSize(mb: number): string {
  if (mb >= 1000) {
    return `~${(mb / 1024).toFixed(1)} GB`;
  }
  return `~${Math.round(mb)} MB`;
}
