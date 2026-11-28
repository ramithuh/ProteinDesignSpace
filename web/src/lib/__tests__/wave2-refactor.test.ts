import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BENCHMARK_SYNC_ADAPTER_IDS,
  buildResetToBenchmarkParams,
  extractFirstContiguousMotifRange,
  formatRfdiffusionContig,
  reconcileBenchmarkSeeding,
  switchBenchmark,
} from '$lib/engine/benchmark-sync';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { validateNodeParams } from '$lib/engine/param-validation';
import type { WorkflowGraph } from '$lib/engine/types';

function makeGraph(nodes: WorkflowGraph['nodes'], edges: WorkflowGraph['edges']): WorkflowGraph {
  return {
    id: 'wave2-test',
    name: 'Wave 2 Test',
    nodes,
    edges,
  };
}

describe('Wave 2 — Multi-generator benchmark sync (WF-06)', () => {
  it('auto-seeds boltzgen, protpardelle, rfdiffusion, and genie2 from 1PRW motif benchmark and tears down symmetrically', () => {
    const graph = makeGraph(
      [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1PRW',
          params: { mode: 'pdb-id', pdbId: '1PRW', benchmarkId: 'rfd_1prw_efhand' },
        },
        { id: 'bg', adapterId: 'boltzgen', label: 'BoltzGen', params: { taskMode: 'unconditional', length: 48 } },
        { id: 'pp', adapterId: 'protpardelle', label: 'ProtPardelle', params: { taskMode: 'unconditional', length: 80 } },
        { id: 'rfd', adapterId: 'rfdiffusion', label: 'RFdiffusion', params: { taskMode: 'unconditional', length: 80 } },
        { id: 'g2', adapterId: 'genie2', label: 'Genie 2', params: { length: 80, motifRange: '' } },
      ],
      [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'bg', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'pp', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'g2', toPort: 'template' },
      ],
    );

    const seeded = reconcileBenchmarkSeeding(graph);
    expect(seeded.changedNodeIds.sort()).toEqual(['bg', 'g2', 'pp', 'rfd']);

    const bg = seeded.graph.nodes.find((n) => n.id === 'bg')!;
    const pp = seeded.graph.nodes.find((n) => n.id === 'pp')!;
    const rfd = seeded.graph.nodes.find((n) => n.id === 'rfd')!;
    const g2 = seeded.graph.nodes.find((n) => n.id === 'g2')!;

    expect(bg.params.taskMode).toBe('motif');
    expect(bg.params.autoSeededBenchmarkId).toBe('rfd_1prw_efhand');

    expect(pp.params.taskMode).toBe('motif');
    expect(String(pp.params.motifSelection)).toContain('A16-35');
    expect(pp.params.autoSeededBenchmarkId).toBe('rfd_1prw_efhand');

    expect(rfd.params.taskMode).toBe('motif');
    expect(String(rfd.params.contig)).toContain('/');
    expect(String(rfd.params.contig)).not.toContain(',');
    expect(rfd.params.autoSeededBenchmarkId).toBe('rfd_1prw_efhand');

    expect(g2.params.motifRange).toBe('A16-35, A52-71');
    expect(g2.params.autoSeededBenchmarkId).toBe('rfd_1prw_efhand');

    // Switch benchmark to 5O45 binder: BoltzGen flips to binder + hotspots, motif-only models reset cleanly
    const switched = switchBenchmark(seeded.graph, 'pdb', 'pdl1_5o45');
    const bg2 = switched.graph.nodes.find((n) => n.id === 'bg')!;
    const pp2 = switched.graph.nodes.find((n) => n.id === 'pp')!;
    const rfd2 = switched.graph.nodes.find((n) => n.id === 'rfd')!;
    const g2_2 = switched.graph.nodes.find((n) => n.id === 'g2')!;

    expect(bg2.params.taskMode).toBe('binder');
    expect(String(bg2.params.hotspots)).toContain('A56');
    expect(pp2.params.taskMode).toBe('unconditional');
    expect(pp2.params.motifSelection).toBe('');
    expect(rfd2.params.taskMode).toBe('unconditional');
    expect(rfd2.params.contig).toBe('');
    expect(g2_2.params.motifRange).toBe('');

    // Exit benchmark mode: symmetric teardown clears autoSeededBenchmarkId and restores defaults
    const cleared = switchBenchmark(switched.graph, 'pdb', null);
    for (const id of ['bg', 'pp', 'rfd', 'g2']) {
      const n = cleared.graph.nodes.find((x) => x.id === id)!;
      expect(n.params.autoSeededBenchmarkId).toBeUndefined();
    }
    expect(cleared.graph.nodes.find((x) => x.id === 'bg')!.params.length).toBe(48);
  });

  it('helper utilities normalize RFdiffusion contigs, Genie 2 ranges, and inspector resets cleanly', () => {
    expect(formatRfdiffusionContig('10-10, A16-35, 10-10')).toBe('10-10/A16-35/10-10');
    expect(extractFirstContiguousMotifRange('A16-35, A52-71')).toBe('A16-35');
    expect(BENCHMARK_SYNC_ADAPTER_IDS.size).toBe(8);

    const resetBg = buildResetToBenchmarkParams('boltzgen', {
      taskMode: 'binder',
      hotspots: 'A56',
      length: 90,
      userEditedTask: true,
      autoSeededBenchmarkId: 'pdl1_5o45',
    });
    expect(resetBg.taskMode).toBe('unconditional');
    expect(resetBg.hotspots).toBe('');
    expect(resetBg.length).toBe(48);
    expect(resetBg.userEditedTask).toBeUndefined();
    expect(resetBg.autoSeededBenchmarkId).toBeUndefined();
  });
});

describe('Wave 2 — CT-10 pre-run notices & schema guards', () => {
  it('warns and offers 1-click fix when RFdiffusion symmetry mode rounds length', () => {
    const graph = makeGraph(
      [
        {
          id: 'rfd',
          adapterId: 'rfdiffusion',
          label: 'RFdiffusion',
          params: { taskMode: 'symmetry', symmetry: 'd3', length: 16 },
        },
      ],
      [],
    );
    const issues = validateBioCapabilities(graph);
    const symWarn = issues.find((i) => i.code === 'rfdiffusion-symmetry-length-adjust');
    expect(symWarn).toBeDefined();
    expect(symWarn?.severity).toBe('warning');
    expect(symWarn?.fix?.paramsPatch).toEqual({ length: 48 });
  });

  it('validates Genie 2 motifRange syntax and missing template wire', () => {
    expect(
      validateNodeParams({ id: 'g2', adapterId: 'genie2', params: { motifRange: 'A16-35, A52-71' } }),
    ).toHaveLength(0);
    expect(
      validateNodeParams({ id: 'g2', adapterId: 'genie2', params: { motifRange: 'A35-16' } }).some((e) =>
        /motifRange/i.test(e),
      ),
    ).toBe(true);
    expect(
      validateNodeParams({ id: 'g2', adapterId: 'genie2', params: { motifRange: 'A16-35' } }),
    ).toHaveLength(0);

    const graph = makeGraph(
      [{ id: 'g2', adapterId: 'genie2', label: 'Genie 2', params: { motifRange: 'A16-35', length: 80 } }],
      [],
    );
    const issues = validateBioCapabilities(graph);
    expect(issues.some((i) => i.code === 'genie2-missing-template' && i.severity === 'error')).toBe(true);
  });

  it('warns on FoldFlow 2 multi-candidate and multi-chain sequence inputs', () => {
    const graph = makeGraph(
      [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          label: 'Seqs',
          params: { sequence: '>s1\nACDEFGHIK:LMNPQRSTV\n>s2\nACDEFGHIKLMNPQRSTV' },
        },
        { id: 'ff2', adapterId: 'foldflow2', label: 'FoldFlow 2', params: { length: 80 } },
      ],
      [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'ff2', toPort: 'sequences' }],
    );
    const issues = validateBioCapabilities(graph);
    expect(issues.some((i) => i.code === 'foldflow2-first-candidate-only')).toBe(true);
    expect(issues.some((i) => i.code === 'foldflow2-multichain-collapsed')).toBe(true);
  });

  it('errors pre-run when ESM-C or Dyna-1 has neither sequences nor structure wired', () => {
    const graph = makeGraph(
      [
        { id: 'ec', adapterId: 'esmc', label: 'ESM-C', params: { mode: 'wildtype' } },
        { id: 'dy', adapterId: 'dyna1', label: 'Dyna-1', params: { inputMode: 'auto' } },
      ],
      [],
    );
    const issues = validateBioCapabilities(graph);
    expect(issues.some((i) => i.code === 'esmc-missing-input' && i.severity === 'error')).toBe(true);
    expect(issues.some((i) => i.code === 'dyna1-missing-input' && i.severity === 'error')).toBe(true);
  });
});

describe('Wave 2 — UI hygiene & dead-code removal', () => {
  const srcRoot = path.resolve(process.cwd(), 'src/lib');

  it('removes dead CampaignViewer.svelte and wires shared modalFocusTrap across all 4 modals', () => {
    expect(fs.existsSync(path.join(srcRoot, 'ui/CampaignViewer.svelte'))).toBe(false);
    for (const modal of [
      'ui/FirstRunModal.svelte',
      'ui/CreditsModal.svelte',
      'ui/ExamplesGalleryModal.svelte',
      'ui/OnboardingModal.svelte',
    ]) {
      const content = fs.readFileSync(path.join(srcRoot, modal), 'utf8');
      expect(content).toContain('modalFocusTrap');
    }
  });

  it('syncs RecipeDrawer dirty state and removes horizontal overflow on Mol* style popover', () => {
    const recipe = fs.readFileSync(path.join(srcRoot, 'ui/RecipeDrawer.svelte'), 'utf8');
    expect(recipe).toContain('canvasChangedWhileDirty');
    expect(recipe).toContain('reloadFromCanvas');

    const mol = fs.readFileSync(path.join(srcRoot, 'viewer/MolSuperposeViewer.svelte'), 'utf8');
    expect(mol).toContain('overflow-x: hidden;');
  });
});
