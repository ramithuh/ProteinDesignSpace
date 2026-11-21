import { describe, expect, it } from 'vitest';
import { estimateRunRequirements, formatApproxSize } from '$lib/engine/run-preflight';
import { WORKFLOW_PRESETS } from '$lib/engine/registry';
import { extractSubgraphUpTo } from '$lib/engine/peek';
import type { WorkflowGraph } from '$lib/engine/types';

describe('estimateRunRequirements (first-run preflight)', () => {
  it('sums RFD4-Proteina + LocalFold AF2 accurately for the default preset', () => {
    const preset = WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-refold-native')!;
    const summary = estimateRunRequirements(preset.graph);
    expect(summary.usesWebGpu).toBe(true);
    expect(summary.totalDownloadMB).toBe(330 + 97);
    expect(summary.peakVramMB).toBe(1400);
    expect(summary.models.map((m) => m.key)).toEqual(['rfd4-proteina', 'localfold:monomer']);
  });

  it('estimates Genie 3 + SolubleMPNN + LocalFold AF2 preset accurately', () => {
    const preset = WORKFLOW_PRESETS.find((p) => p.id === 'genie3-mpnn-af2')!;
    const summary = estimateRunRequirements(preset.graph);
    expect(summary.usesWebGpu).toBe(true);
    expect(summary.totalDownloadMB).toBe(45 + 10 + 97);
    expect(summary.models).toHaveLength(3);
  });

  it('deduplicates identical checkpoints and sums distinct LocalFold checkpoints', () => {
    const graph: WorkflowGraph = {
      id: 'compare',
      name: 'Compare',
      nodes: [
        { id: 's', adapterId: 'sequence-input', params: {} },
        { id: 'f1', adapterId: 'localfold-af2', params: { model: 'monomer' } },
        { id: 'f2', adapterId: 'localfold-af2', params: { model: 'monomer' } },
        { id: 'f3', adapterId: 'localfold-af2', params: { model: 'boltz2' } },
      ],
      edges: [],
    };
    const summary = estimateRunRequirements(graph);
    expect(summary.models.map((m) => m.key)).toEqual(['localfold:monomer', 'localfold:boltz2']);
    expect(summary.totalDownloadMB).toBe(97 + 364);
    expect(summary.peakVramMB).toBe(1800);
  });

  it('adds the 138 MB RFD4 encoder when partial diffusion mode is active', () => {
    const graph: WorkflowGraph = {
      id: 'partial',
      name: 'Partial',
      nodes: [
        { id: 'g', adapterId: 'rfd4-proteina', params: { mode: 'partial' } },
      ],
      edges: [],
    };
    const summary = estimateRunRequirements(graph);
    expect(summary.totalDownloadMB).toBe(330 + 138);
    expect(summary.models.map((m) => m.key)).toEqual(['rfd4-proteina', 'rfd4-proteina:encoder']);
  });

  it('reports zero download and no WebGPU for CPU-only partial subgraphs', () => {
    const preset = WORKFLOW_PRESETS.find((p) => p.id === 'pdb-refold-qc')!;
    const sub = extractSubgraphUpTo(preset.graph, 'src');
    const summary = estimateRunRequirements(sub);
    expect(summary.usesWebGpu).toBe(false);
    expect(summary.totalDownloadMB).toBe(0);
    expect(summary.models).toEqual([]);
  });

  it('formats MB and GB sizes cleanly', () => {
    expect(formatApproxSize(45)).toBe('~45 MB');
    expect(formatApproxSize(427)).toBe('~427 MB');
    expect(formatApproxSize(1126)).toBe('~1.1 GB');
  });
});
