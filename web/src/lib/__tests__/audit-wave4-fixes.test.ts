import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  compileWorkflowPlanToGraph,
  parseDeterministicEditOperations,
  reconcilePlannedWorkflow,
} from '$lib/engine/workflow-planner';
import type { WorkflowGraph } from '$lib/engine/types';

describe('Astra 6.2 Audit Wave 4: Precision Validation, Device-Loss Cleanup, Genie 3 Integrity & Planner Fixes', () => {
  it('Genie 3 loadWeightsFromManifestData verifies shard SHA-256 digests when present and rejects corrupted bytes', async () => {
    const url = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/genie3/shared/genie3/safetensors.js'),
    ).href;
    const mod = (await import(/* @vite-ignore */ url)) as {
      loadWeightsFromManifestData: (
        manifest: unknown,
        loadShard: (file: string) => Promise<Uint8Array>,
      ) => Promise<unknown>;
    };
    const manifest = {
      shards: [{ file: 'shard-0.bin', sha256: '0'.repeat(64) }],
      tensors: { w: { file: 'shard-0.bin', shape: [1], dtype: 'float32', byteOffset: 0 } },
    };
    await expect(
      mod.loadWeightsFromManifestData(manifest, async () => new Uint8Array([1, 2, 3, 4])),
    ).rejects.toThrow(/integrity mismatch/i);
  });

  it('compileWorkflowPlan preserves family="proteinmpnn" when tasks=["proteinmpnn"] (not silently solublempnn)', () => {
    const { graph } = compileWorkflowPlanToGraph({
      kind: 'pdb',
      pdbId: '1UBQ',
      crop: null,
      tasks: ['proteinmpnn'],
      targetChain: null,
      regions: null,
      numSequences: 2,
      omitAminoAcids: null,
      temperature: 0.1,
      mpnnFamily: null,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: false,
      score: false,
      summary: 'Design 1UBQ with ProteinMPNN',
    });
    const mpnn = graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn');
    expect(mpnn).toBeDefined();
    expect(mpnn?.params.family).toBe('proteinmpnn');
  });

  it('reconcilePlannedWorkflow resolves "ESMFold2 600M" to ef2-fast-600m and "AlphaFold 3" to af3 only (no monomer duplicate)', () => {
    const baseSeqPlan = {
      kind: 'sequence' as const,
      sequences: ['MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALE'],
      label: 'Test',
      foldModels: [] as Array<'monomer'>,
      stabilityScan: false,
      topK: null,
      score: false,
      summary: 'Fold sequence',
    };
    const ef2 = reconcilePlannedWorkflow(baseSeqPlan, 'Fold MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALE with ESMFold2 600M');
    expect(ef2.kind === 'sequence' && ef2.foldModels).toEqual(['ef2-fast-600m']);

    const af3 = reconcilePlannedWorkflow(baseSeqPlan, 'Fold MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALE with AlphaFold 3');
    expect(af3.kind === 'sequence' && af3.foldModels).toEqual(['af3']);
  });

  it('parseDeterministicEditOperations routes explicit GNINA requests to add_gnina', () => {
    const graph: WorkflowGraph = {
      id: 'g1',
      name: 'g1',
      nodes: [{ id: 'pdb_1', adapterId: 'pdb-input', label: '1STP', params: { pdbId: '1STP' } }],
      edges: [],
    };
    const edit = parseDeterministicEditOperations('add GNINA pocket scoring', graph);
    expect(edit).not.toBeNull();
    expect(edit?.operations.some((op) => op.action === 'add_gnina')).toBe(true);
  });

  it('reconcilePlannedWorkflow overrides default symmetry c3 with explicit prompt symmetry C6', () => {
    const plan = {
      kind: 'block_graph' as const,
      mode: 'new' as const,
      nodes: [
        {
          id: 'rfd_1',
          adapter: 'rfdiffusion',
          label: 'RFdiffusion',
          params: { taskMode: 'symmetry', symmetry: 'c3', length: 72 },
        },
      ],
      edges: [] as Array<[string, string]>,
      editOps: [],
      unsupported: [],
      summary: 'C6 symmetric oligomer',
    };
    const reconciled = reconcilePlannedWorkflow(plan, 'Generate a C6 symmetric 120aa oligomer with RFdiffusion');
    expect(reconciled.kind).toBe('block_graph');
    if (reconciled.kind === 'block_graph') {
      expect(reconciled.nodes?.[0]?.params?.symmetry).toBe('c6');
    }
  });

  it('compileWorkflowPlan does not silently compile ProtPardelle protein-binder request to monomer motif', () => {
    const { graph, notes } = compileWorkflowPlanToGraph({
      kind: 'conditioned_design',
      designTask: 'binder',
      benchmarkId: 'pdl1_5o45',
      pdbId: '5O45',
      crop: null,
      generators: ['protpardelle'],
      length: 70,
      numDesigns: 1,
      sequenceDesigners: [],
      numSequences: 2,
      omitAminoAcids: null,
      temperature: null,
      mpnnFamily: null,
      topK: null,
      refold: false,
      foldModels: [],
      score: false,
      stabilityScan: false,
      summary: 'Design a PD-L1 binder with ProtPardelle',
    });
    expect(graph.nodes.some((n) => n.adapterId === 'protpardelle')).toBe(false);
    expect(graph.nodes.some((n) => n.adapterId === 'rfd4-proteina')).toBe(true);
    expect(notes.join(' ')).toMatch(/binder/i);
  });

  it('add_sequence_designer attaches Cα-ProteinMPNN to Genie 2 and update_params.numDesigns updates RFdiffusion', () => {
    const graph: WorkflowGraph = {
      id: 'g2',
      name: 'g2',
      nodes: [
        { id: 'g2_1', adapterId: 'genie2', label: 'Genie 2', params: { length: 80, numDesigns: 1 } },
        { id: 'rfd_1', adapterId: 'rfdiffusion', label: 'RFdiffusion', params: { length: 80, numDesigns: 1 } },
      ],
      edges: [],
    };
    const { graph: next } = compileWorkflowPlanToGraph(
      {
        kind: 'edit_current',
        operations: [
          {
            action: 'add_sequence_designer',
            attachToNodeId: 'g2_1',
            model: 'proteinmpnn',
            foldModels: [],
            topK: null,
            numSequences: 2,
            targetChain: null,
            regions: null,
            score: false,
          },
          {
            action: 'update_params',
            attachToNodeId: null,
            targetNodeIds: ['rfd_1'],
            model: null,
            foldModels: [],
            topK: null,
            numSequences: null,
            numDesigns: 4,
            targetChain: null,
            regions: null,
            score: false,
          },
        ],
        summary: 'Attach MPNN to Genie 2 and set RFdiffusion numDesigns=4',
      },
      graph,
    );
    const mpnn = next.nodes.find((n) => n.adapterId === 'web-protein-mpnn');
    expect(mpnn?.params.family).toBe('ca');
    const rfd = next.nodes.find((n) => n.id === 'rfd_1');
    expect(rfd?.params.numDesigns).toBe(4);
  });

  it('GPU workers reject unsupported precision strings and attach device.lost invalidation', () => {
    const strictWorkers = [
      'src/lib/plugins/boltzgen/boltzgen.worker.ts',
      'src/lib/plugins/protpardelle/protpardelle.worker.ts',
      'src/lib/plugins/la-proteina/laproteina.worker.ts',
      'src/lib/plugins/proteina/proteina.worker.ts',
      'src/lib/plugins/sid-protein/sidprotein.worker.ts',
      'src/lib/plugins/foldflow2/foldflow2.worker.ts',
      'src/lib/plugins/evodiff/evodiff.worker.ts',
      'src/lib/plugins/genie2/genie2.worker.ts',
      'src/lib/plugins/frameflow/frameflow.worker.ts',
      'src/lib/plugins/rfdiffusion/rfdiffusion.worker.ts',
      'src/lib/plugins/esmc/esmc.worker.ts',
      'src/lib/plugins/dyna1/dyna1.worker.ts',
      'src/lib/plugins/gnina/gnina.worker.ts',
      'src/lib/plugins/antifold-esmif1/antifold.worker.ts',
    ];
    for (const rel of strictWorkers) {
      const src = readFileSync(resolve(process.cwd(), rel), 'utf8');
      expect(src, `${rel} should throw on unsupported precision`).toMatch(/Unsupported .* precision/);
    }
    const lostWorkers = [
      'src/lib/plugins/rfdiffusion/rfdiffusion.worker.ts',
      'src/lib/plugins/genie2/genie2.worker.ts',
      'src/lib/plugins/la-proteina/laproteina.worker.ts',
      'src/lib/plugins/proteina/proteina.worker.ts',
      'src/lib/plugins/sid-protein/sidprotein.worker.ts',
      'src/lib/plugins/esmc/esmc.worker.ts',
      'src/lib/plugins/dyna1/dyna1.worker.ts',
      'src/lib/plugins/gnina/gnina.worker.ts',
    ];
    for (const rel of lostWorkers) {
      const src = readFileSync(resolve(process.cwd(), rel), 'utf8');
      expect(src, `${rel} should handle device.lost`).toMatch(/\.lost\.then/);
    }
  });
});
