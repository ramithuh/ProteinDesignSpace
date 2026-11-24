import { describe, expect, it } from 'vitest';
import { estimateRunRequirements, formatApproxSize } from '$lib/engine/run-preflight';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from '$lib/engine/registry';
import { CATALOG_CATEGORIES } from '$lib/engine/catalog';
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

  it('adds the 138 MB RFD4 encoder for partial diffusion and inverse-folding, and labels ProteinMPNN variants accurately', () => {
    const partialGraph: WorkflowGraph = {
      id: 'partial',
      name: 'Partial',
      nodes: [
        { id: 'g', adapterId: 'rfd4-proteina', params: { mode: 'partial' } },
      ],
      edges: [],
    };
    const partialSummary = estimateRunRequirements(partialGraph);
    expect(partialSummary.totalDownloadMB).toBe(330 + 138);
    expect(partialSummary.models.map((m) => m.key)).toEqual(['rfd4-proteina', 'rfd4-proteina:encoder']);

    const invFoldGraph: WorkflowGraph = {
      id: 'inv-fold',
      name: 'InvFold',
      nodes: [
        { id: 'g', adapterId: 'rfd4-proteina', params: { mode: 'inverse-folding' } },
        { id: 'm', adapterId: 'web-protein-mpnn', params: { family: 'ca', modelName: 'v_48_020' } },
      ],
      edges: [],
    };
    const invSummary = estimateRunRequirements(invFoldGraph);
    expect(invSummary.models.map((m) => m.key)).toContain('rfd4-proteina:encoder');
    expect(invSummary.models.find((m) => m.key === 'mpnn:ca:v_48_020')?.label).toBe(
      'Cα-only ProteinMPNN (v_48_020)',
    );
  });

  it('adds the 138 MB RFD4 encoder when rfd4-proteina uses mode=preset with partial_diffusion_1prw or inverse_folding_1prw', () => {
    for (const presetId of ['partial_diffusion_1prw', 'inverse_folding_1prw']) {
      const graph: WorkflowGraph = {
        id: `preset-${presetId}`,
        name: `Preset ${presetId}`,
        nodes: [
          { id: 'g', adapterId: 'rfd4-proteina', params: { mode: 'preset', presetId } },
        ],
        edges: [],
      };
      const summary = estimateRunRequirements(graph);
      expect(summary.totalDownloadMB).toBe(468);
      expect(summary.models.map((m) => m.key)).toEqual(['rfd4-proteina', 'rfd4-proteina:encoder']);
    }

    const noEncGraph: WorkflowGraph = {
      id: 'preset-cath',
      name: 'Preset CATH',
      nodes: [
        { id: 'g', adapterId: 'rfd4-proteina', params: { mode: 'preset', presetId: 'custom_cath_monomer' } },
      ],
      edges: [],
    };
    const noEncSummary = estimateRunRequirements(noEncGraph);
    expect(noEncSummary.totalDownloadMB).toBe(330);
    expect(noEncSummary.models.map((m) => m.key)).toEqual(['rfd4-proteina']);
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

  it('sums co-resident RFD4 + encoder VRAM (1300 MB) and deduplicates shared AF2 delta base weights', () => {
    const encGraph: WorkflowGraph = {
      id: 'enc-vram',
      name: 'Encoder VRAM',
      nodes: [{ id: 'g', adapterId: 'rfd4-proteina', params: { mode: 'partial' } }],
      edges: [],
    };
    const encSummary = estimateRunRequirements(encGraph);
    expect(encSummary.peakVramMB).toBe(1300);

    // monomer (97 MB) + monomer-2 (43 MB delta) = 140 MB (base not double-counted)
    const af2MonoCombo: WorkflowGraph = {
      id: 'af2-mono-combo',
      name: 'AF2 Mono Combo',
      nodes: [
        { id: 'f1', adapterId: 'localfold-af2', params: { model: 'monomer' } },
        { id: 'f2', adapterId: 'localfold-af2', params: { model: 'monomer-2' } },
        { id: 'f3', adapterId: 'localfold-af2', params: { model: 'monomer-3' } },
      ],
      edges: [],
    };
    expect(estimateRunRequirements(af2MonoCombo).totalDownloadMB).toBe(97 + 43 + 43);

    // monomer-2 + monomer-3 without monomer: counts 97 MB base once + two 43 MB deltas = 183 MB
    const af2DeltaOnly: WorkflowGraph = {
      id: 'af2-delta-only',
      name: 'AF2 Delta Only',
      nodes: [
        { id: 'f2', adapterId: 'localfold-af2', params: { model: 'monomer-2' } },
        { id: 'f3', adapterId: 'localfold-af2', params: { model: 'monomer-3' } },
      ],
      edges: [],
    };
    expect(estimateRunRequirements(af2DeltaOnly).totalDownloadMB).toBe(140 + 43);

    // multimer (74 MB) + multimer-2 (44 MB delta) = 118 MB
    const af2MultiCombo: WorkflowGraph = {
      id: 'af2-multi-combo',
      name: 'AF2 Multimer Combo',
      nodes: [
        { id: 'm2', adapterId: 'localfold-af2', params: { model: 'multimer-2' } },
        { id: 'm1', adapterId: 'localfold-af2', params: { model: 'multimer' } },
      ],
      edges: [],
    };
    expect(estimateRunRequirements(af2MultiCombo).totalDownloadMB).toBe(74 + 44);
  });

  it('scales planned items and emits fanoutSummary for numDesigns=4 while keeping weight download MB unchanged', () => {
    const basePreset = WORKFLOW_PRESETS.find((p) => p.id === 'genie3-mpnn-af2')!;
    const baseSummary = estimateRunRequirements(basePreset.graph);
    expect(baseSummary.fanoutSummary).toBeUndefined();
    expect(baseSummary.estimatedComputeScale).toBe(1);

    const multiGraph: WorkflowGraph = {
      ...basePreset.graph,
      id: 'genie3-mpnn-af2-batch4',
      nodes: basePreset.graph.nodes.map((n) => {
        if (n.adapterId === 'genie3') {
          return { ...n, params: { ...n.params, numDesigns: 4 } };
        }
        if (n.adapterId === 'web-protein-mpnn') {
          return { ...n, params: { ...n.params, numSequences: 2 } };
        }
        if (n.adapterId === 'localfold-af2') {
          return { ...n, params: { ...n.params, capMode: 'fixed', maxSequences: 1, maxTotalFolds: 16 } };
        }
        return n;
      }),
    };

    const multiSummary = estimateRunRequirements(multiGraph);
    // Weights are downloaded & loaded ONCE per stage — never multiplied by numDesigns.
    expect(multiSummary.totalDownloadMB).toBe(baseSummary.totalDownloadMB);
    expect(multiSummary.totalDownloadMB).toBe(45 + 10 + 97);
    expect(multiSummary.peakVramMB).toBe(baseSummary.peakVramMB);
    expect(multiSummary.weightsLoadedOnce).toBe(true);

    expect(multiSummary.plannedBackbones).toBe(4);
    expect(multiSummary.plannedSequences).toBe(8);
    expect(multiSummary.plannedFolds).toBe(4);
    expect(multiSummary.fanoutSummary).toBe('4 backbones → 8 sequences → 4 refolds');
    expect(multiSummary.models.find((m) => m.key === 'genie3')?.plannedItems).toBe(4);
    expect(multiSummary.models.find((m) => m.key.startsWith('mpnn:'))?.plannedItems).toBe(8);
    expect(multiSummary.models.find((m) => m.key === 'localfold:monomer')?.plannedItems).toBe(4);
    expect(multiSummary.totalGpuWorkItems).toBe(16);
    expect(multiSummary.estimatedComputeScale).toBeGreaterThan(1);
  });

  it('respects maxTotalFolds cap and RFD4 native multi-design fan-out without inflating weight MB', () => {
    const rfdPreset = WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-refold-native')!;
    const rfdBase = estimateRunRequirements(rfdPreset.graph);
    const rfdBatch: WorkflowGraph = {
      ...rfdPreset.graph,
      id: 'rfd4-batch4',
      nodes: rfdPreset.graph.nodes.map((n) =>
        n.adapterId === 'rfd4-proteina'
          ? { ...n, params: { ...n.params, numDesigns: 4 } }
          : n,
      ),
    };
    const rfdSummary = estimateRunRequirements(rfdBatch);
    expect(rfdSummary.totalDownloadMB).toBe(rfdBase.totalDownloadMB);
    expect(rfdSummary.peakVramMB).toBe(rfdBase.peakVramMB);
    expect(rfdSummary.plannedBackbones).toBe(4);
    expect(rfdSummary.plannedSequences).toBe(4);
    expect(rfdSummary.plannedFolds).toBe(4);
    expect(rfdSummary.fanoutSummary).toBe('4 backbones → 4 sequences → 4 refolds');
    expect(rfdSummary.estimatedComputeScale).toBe(4);

    // Campaign cap via maxTotalFolds (4 backbones x 4 seqs = 16 seqs, maxSequences=2/backbone -> 8, capped to 6)
    const cappedGraph: WorkflowGraph = {
      id: 'capped-batch',
      name: 'Capped Batch',
      nodes: [
        { id: 'g', adapterId: 'genie3', params: { numDesigns: 4 } },
        { id: 'm', adapterId: 'web-protein-mpnn', params: { numSequences: 4 } },
        { id: 'f', adapterId: 'localfold-af2', params: { model: 'monomer', maxSequences: 2, maxTotalFolds: 6 } },
      ],
      edges: [
        { fromNode: 'g', fromPort: 'structure', toNode: 'm', toPort: 'structure' },
        { fromNode: 'm', fromPort: 'sequences', toNode: 'f', toPort: 'sequences' },
      ],
    };
    const capped = estimateRunRequirements(cappedGraph);
    expect(capped.totalDownloadMB).toBe(45 + 10 + 97);
    expect(capped.plannedBackbones).toBe(4);
    expect(capped.plannedSequences).toBe(16);
    expect(capped.plannedFolds).toBe(6);
    expect(capped.fanoutSummary).toBe('4 backbones → 16 sequences → 6 refolds');
  });

  it('accounts for top-k-filter reduction between MPNN and LocalFold', () => {
    const filteredGraph: WorkflowGraph = {
      id: 'filtered-batch',
      name: 'Filtered Batch',
      nodes: [
        { id: 'g', adapterId: 'genie3', params: { numDesigns: 4 } },
        { id: 'm', adapterId: 'web-protein-mpnn', params: { numSequences: 4 } },
        { id: 'k', adapterId: 'top-k-filter', params: { mode: 'per-backbone', topK: 1, maxTotal: 8 } },
        { id: 'f', adapterId: 'localfold-af2', params: { model: 'monomer', maxSequences: 2, maxTotalFolds: 16 } },
      ],
      edges: [
        { fromNode: 'g', fromPort: 'structure', toNode: 'm', toPort: 'structure' },
        { fromNode: 'm', fromPort: 'sequences', toNode: 'k', toPort: 'sequences' },
        { fromNode: 'k', fromPort: 'sequences', toNode: 'f', toPort: 'sequences' },
      ],
    };
    const summary = estimateRunRequirements(filteredGraph);
    expect(summary.totalDownloadMB).toBe(45 + 10 + 97);
    expect(summary.plannedBackbones).toBe(4);
    expect(summary.plannedSequences).toBe(16);
    expect(summary.plannedFolds).toBe(4);
    expect(summary.fanoutSummary).toBe('4 backbones → 16 sequences → 4 selected → 4 refolds');
  });

  it('CT-09: drives precisionWeightsMB & nonCommercial from ADAPTER_REGISTRY, honors sequence-input/evodiff counts, and clamps esmc/dyna1 to maxSequences', () => {
    // 1. Precision weights match ADAPTER_REGISTRY.*.precisionWeightsMB across all 5 previously divergent adapters
    for (const [adapterId, int8MB, fp16MB] of [
      ['la-proteina', 277, 548],
      ['proteina', 186, 366],
      ['sid-protein', 185, 366],
      ['frameflow', 17, 33],
      ['evodiff', 37, 73],
    ] as const) {
      expect(ADAPTER_REGISTRY[adapterId].precisionWeightsMB).toEqual({ int8: int8MB, fp16: fp16MB });
      const int8Summary = estimateRunRequirements({
        id: `${adapterId}-int8`,
        name: adapterId,
        nodes: [{ id: 'n', adapterId, params: { precision: 'int8' } }],
        edges: [],
      });
      expect(int8Summary.totalDownloadMB).toBe(int8MB);
      const fp16Summary = estimateRunRequirements({
        id: `${adapterId}-fp16`,
        name: adapterId,
        nodes: [{ id: 'n', adapterId, params: { precision: 'fp16' } }],
        edges: [],
      });
      expect(fp16Summary.totalDownloadMB).toBe(fp16MB);
    }

    // 2. sid-protein nonCommercial parity between registry, preflight, and catalog
    expect(ADAPTER_REGISTRY['sid-protein'].nonCommercial).toBe(true);
    const sidSummary = estimateRunRequirements({
      id: 'sid-nc',
      name: 'SiD NC',
      nodes: [{ id: 's', adapterId: 'sid-protein', params: { precision: 'int8' } }],
      edges: [],
    });
    expect(sidSummary.hasNonCommercialModel).toBe(true);
    const genCat = CATALOG_CATEGORIES.find((c) => c.id === 'generation')!;
    const sidCatalogItem = genCat.subGroups.flatMap((g) => g.items).find((i) => i.key === 'sid-protein')!;
    expect(sidCatalogItem.nonCommercial).toBe(true);
    expect(sidCatalogItem.badge).toBe('Non-Commercial');

    // 3. Multi-FASTA sequence-input & evodiff populate plannedSequences
    const seqFeedSummary = estimateRunRequirements({
      id: 'seq-to-lf',
      name: 'Seq to LF',
      nodes: [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          params: { sequence: '>s1\nACDEFGHIK\n>s2\nLMNPQRSTV\n>s3\nWYACDEFGH' },
        },
        { id: 'lf', adapterId: 'localfold-af2', params: { model: 'monomer', maxSequences: 2 } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'lf', toPort: 'sequences' }],
    });
    expect(seqFeedSummary.plannedSequences).toBe(3);
    expect(seqFeedSummary.plannedFolds).toBe(2);

    const evoFeedSummary = estimateRunRequirements({
      id: 'evo-to-lf',
      name: 'EvoDiff to LF',
      nodes: [
        { id: 'evo', adapterId: 'evodiff', params: { taskMode: 'unconditional', numSequences: 4 } },
        { id: 'lf', adapterId: 'localfold-af2', params: { model: 'monomer', maxSequences: 2 } },
      ],
      edges: [{ fromNode: 'evo', fromPort: 'sequences', toNode: 'lf', toPort: 'sequences' }],
    });
    expect(evoFeedSummary.plannedSequences).toBe(4);
    expect(evoFeedSummary.plannedFolds).toBe(2);
    expect(evoFeedSummary.models.find((m) => m.key === 'evodiff:int8')?.plannedItems).toBe(4);

    // 4. esmc & dyna1 clamp GPU work items by maxSequences (default 1)
    const scorerClampSummary = estimateRunRequirements({
      id: 'clamp-scorers',
      name: 'Clamp Scorers',
      nodes: [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          params: {
            sequence:
              '>s1\nACDEFGHIK\n>s2\nACDEFGHIK\n>s3\nACDEFGHIK\n>s4\nACDEFGHIK\n>s5\nACDEFGHIK\n>s6\nACDEFGHIK',
          },
        },
        { id: 'ec', adapterId: 'esmc', params: { mode: 'wildtype', maxSequences: 2 } },
        { id: 'dy', adapterId: 'dyna1', params: { inputMode: 'seq-only', maxSequences: 1 } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'ec', toPort: 'sequences' },
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'dy', toPort: 'sequences' },
      ],
    });
    expect(scorerClampSummary.models.find((m) => m.key === 'esmc:int8')?.plannedItems).toBe(2);
    expect(scorerClampSummary.models.find((m) => m.key === 'dyna1:int8:seq')?.plannedItems).toBe(1);
    expect(scorerClampSummary.totalGpuWorkItems).toBe(3);
  });
});
