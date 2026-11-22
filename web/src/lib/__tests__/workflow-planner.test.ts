import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  compileLocalStarterPlan,
  compileWorkflowPlanToGraph,
  DEFAULT_PLANNER_ENDPOINT,
  LOCAL_EXAMPLE_PLANS,
  parseDeterministicEditOperations,
  parseRetryAfterSec,
  PlannerHttpError,
  reconcilePlannedWorkflow,
  requestWorkflowPlan,
  UnsupportedWorkflowError,
  type ConditionedDesignWorkflowPlan,
  type DeNovoWorkflowPlan,
  type EditCurrentWorkflowPlan,
  type PdbWorkflowPlan,
  type SequenceWorkflowPlan,
  type WorkflowPlan,
} from '$lib/engine/workflow-planner';
import { validateGraph } from '$lib/engine/graph-utils';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import type { WorkflowGraph } from '$lib/engine/types';

describe('WorkflowPlan deterministic compiler', () => {
  it('compiles Di-Iron 4-Helix Bundle motif + Genie 3 & RFD4 + AF2/RF3/Protenix-v2 into a valid 16-block multi-oracle graph', () => {
    const plan: ConditionedDesignWorkflowPlan = {
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
      summary: 'Di-Iron 4-Helix (1BCF): Genie 3 & RFD4 → AF2, RF3, & Protenix2 eval',
    };

    const { graph, notes } = compileWorkflowPlanToGraph(plan);
    expect(notes).toEqual([]);
    expect(validateGraph(graph)).toEqual([]);
    expect(validateBioCapabilities(graph, {}).filter((i) => i.severity === 'error')).toEqual([]);

    // 1 shared PDB (1BCF) + Genie3 + Cα-MPNN + RFD4 + 6 folds (3 per branch) + 6 scorers = 16 nodes
    expect(graph.nodes).toHaveLength(16);
    expect(graph.edges).toHaveLength(21);

    const pdb = graph.nodes.find((n) => n.adapterId === 'pdb-input')!;
    expect(pdb.params.pdbId).toBe('1BCF');
    expect(pdb.params.benchmarkId).toBe('rfd_1bcf_diiron');
    expect(String(pdb.params.cropSegments)).toContain('A18-25');

    const folds = graph.nodes.filter((n) => n.adapterId === 'localfold-af2');
    expect(folds).toHaveLength(6);
    expect(folds.map((f) => f.params.model).sort()).toEqual([
      'monomer',
      'monomer',
      'protenix2',
      'protenix2',
      'rosettafold3',
      'rosettafold3',
    ]);
  });

  it('auto-drops Genie 3 on all-atom enzyme/ligand benchmarks where supportsGenie3 is false', () => {
    const plan: ConditionedDesignWorkflowPlan = {
      kind: 'conditioned_design',
      designTask: 'enzyme-ligand',
      benchmarkId: 'ame_1ctt_cda',
      pdbId: null,
      crop: null,
      generators: ['genie3', 'rfd4-proteina'],
      length: null,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['rosettafold3'],
      score: true,
      stabilityScan: false,
      summary: 'AME CDA Zn enzyme scaffolding',
    };

    const { graph, notes } = compileWorkflowPlanToGraph(plan);
    expect(validateGraph(graph)).toEqual([]);
    expect(graph.nodes.some((n) => n.adapterId === 'genie3')).toBe(false);
    expect(graph.nodes.some((n) => n.adapterId === 'rfd4-proteina')).toBe(true);
    expect(notes.length).toBe(1);
  });

  it('compiles dual-generator de novo workflow with Top-K and ThermoMPNN stability scan', () => {
    const plan: DeNovoWorkflowPlan = {
      kind: 'de_novo',
      generators: ['rfd4-proteina', 'genie3'],
      length: 100,
      numDesigns: 2,
      numSequences: 4,
      topK: 2,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: true,
      summary: 'RFD4 & Genie 3 (100aa) → Top-2 → AF2 → ThermoMPNN',
    };

    const { graph } = compileWorkflowPlanToGraph(plan);
    expect(validateGraph(graph)).toEqual([]);
    expect(graph.nodes.filter((n) => n.adapterId === 'top-k-filter')).toHaveLength(2);
    expect(graph.nodes.filter((n) => n.adapterId === 'thermompnn')).toHaveLength(2);
    expect(graph.nodes.find((n) => n.adapterId === 'genie3')?.params.length).toBe(100);
    expect(graph.nodes.find((n) => n.adapterId === 'rfd4-proteina')?.params.length).toBe(100);
  });

  it('compiles PDB AntiFold CDR3 and holo LigandMPNN workflows cleanly', () => {
    const antiPlan: PdbWorkflowPlan = {
      kind: 'pdb',
      pdbId: '1FDL',
      crop: null,
      task: 'antifold',
      targetChain: 'H',
      regions: 'CDR3',
      numSequences: 3,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: false,
      summary: '1FDL H CDR3 → AntiFold',
    };
    const { graph: antiGraph } = compileWorkflowPlanToGraph(antiPlan);
    expect(validateGraph(antiGraph)).toEqual([]);
    const af = antiGraph.nodes.find((n) => n.adapterId === 'antifold-esmif1')!;
    expect(af.params.variant).toBe('antifold');
    expect(af.params.targetChain).toBe('H');
    expect(af.params.regions).toBe('CDR3');

    const ligPlan: PdbWorkflowPlan = {
      kind: 'pdb',
      pdbId: '3PTB',
      crop: 'A, BEN, CA',
      task: 'ligandmpnn',
      targetChain: null,
      regions: null,
      numSequences: 2,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: false,
      summary: '3PTB [A, BEN, CA] → LigandMPNN',
    };
    const { graph: ligGraph } = compileWorkflowPlanToGraph(ligPlan);
    expect(validateGraph(ligGraph)).toEqual([]);
    expect(ligGraph.nodes.find((n) => n.adapterId === 'pdb-input')?.params.cropSegments).toBe(
      'A, BEN, CA',
    );
  });

  it('compiles multi-task PDB comparison (LigandMPNN vs LASErMPNN) and incremental edit_current operations (ThermoMPNN, RF3, Top-K splice)', () => {
    const cmpPlan: PdbWorkflowPlan = {
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
    };
    const { graph: cmpGraph } = compileWorkflowPlanToGraph(cmpPlan);
    expect(validateGraph(cmpGraph)).toEqual([]);
    expect(cmpGraph.nodes.filter((n) => n.adapterId === 'pdb-input')).toHaveLength(1);
    expect(cmpGraph.nodes.some((n) => n.adapterId === 'ligandmpnn')).toBe(true);
    expect(cmpGraph.nodes.some((n) => n.adapterId === 'lasermpnn')).toBe(true);

    // Start with a Genie 3 -> Cα-MPNN -> AF2 -> scRMSD graph and incrementally edit it
    const base = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['genie3'],
      length: 80,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      summary: 'Genie 3 base',
    }).graph;

    const foldNode = base.nodes.find((n) => n.adapterId === 'localfold-af2')!;
    const mpnnNode = base.nodes.find((n) => n.adapterId === 'web-protein-mpnn')!;

    const editPlan: EditCurrentWorkflowPlan = {
      kind: 'edit_current',
      operations: [
        {
          action: 'insert_top_k',
          attachToNodeId: foldNode.id,
          model: null,
          foldModels: [],
          topK: 2,
          numSequences: null,
          targetChain: null,
          regions: null,
          score: true,
        },
        {
          action: 'add_stability_scan',
          attachToNodeId: foldNode.id,
          model: null,
          foldModels: [],
          topK: null,
          numSequences: null,
          targetChain: null,
          regions: null,
          score: true,
        },
        {
          action: 'add_fold_oracles',
          attachToNodeId: mpnnNode.id,
          model: null,
          foldModels: ['rosettafold3'],
          topK: null,
          numSequences: null,
          targetChain: null,
          regions: null,
          score: true,
        },
      ],
      summary: 'Splice Top-2 before AF2, add ThermoMPNN, and add RF3 refold branch',
    };

    const edited = compileWorkflowPlanToGraph(editPlan, base);
    expect(edited.isIncrementalEdit).toBe(true);
    expect(edited.addedNodeCount).toBe(4); // Top-K + ThermoMPNN + RF3 + scRMSD
    expect(validateGraph(edited.graph)).toEqual([]);
    expect(edited.graph.nodes.some((n) => n.adapterId === 'top-k-filter')).toBe(true);
    expect(edited.graph.nodes.some((n) => n.adapterId === 'thermompnn')).toBe(true);
    expect(
      edited.graph.nodes.some(
        (n) => n.adapterId === 'localfold-af2' && n.params.model === 'rosettafold3',
      ),
    ).toBe(true);
    // Original MPNN -> AF2 direct wire must be spliced through Top-K
    expect(
      edited.graph.edges.some((e) => e.fromNode === mpnnNode.id && e.toNode === foldNode.id),
    ).toBe(false);
  });

  it('propagates unsupported model notices on partial matches and throws UnsupportedWorkflowError on fully unsupported plans', () => {
    const partial = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['genie3'],
      length: 100,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['rosettafold3'],
      score: true,
      stabilityScan: false,
      unsupported: ['Chroma', 'Chai-1'],
      summary: '100aa de novo: Genie 3 → RoseTTAFold 3 (partial match)',
    });
    expect(validateGraph(partial.graph)).toEqual([]);
    expect(partial.unsupported).toEqual(['Chroma', 'Chai-1']);

    expect(() =>
      compileWorkflowPlanToGraph({
        kind: 'unsupported',
        unsupported: ['BindCraft', 'Chai-1'],
        reason: "BindCraft and Chai-1 aren't available in denovo.run yet.",
        suggestions: ['Use RFD4-Proteina or Genie 3'],
        summary: 'BindCraft and Chai-1 are not in denovo.run',
      }),
    ).toThrow(/BindCraft and Chai-1/);
  });

  it('points DEFAULT_PLANNER_ENDPOINT to the custom domain planner.denovo.run and compiles all 4 local starter chips at 0 Neurons', () => {
    expect(DEFAULT_PLANNER_ENDPOINT).toBe('https://planner.denovo.run/api/plan');

    const base = compileLocalStarterPlan(
      LOCAL_EXAMPLE_PLANS.find((e) => e.id === 'denovo-60aa')!.plan,
    );
    expect(base.model).toBe('local-preset');
    expect(base.latencyMs).toBe(0);
    expect(validateGraph(base.graph)).toEqual([]);

    for (const ex of LOCAL_EXAMPLE_PLANS) {
      const compiled = compileLocalStarterPlan(ex.plan, ex.requiresExisting ? base.graph : undefined);
      expect(compiled.model).toBe('local-preset');
      expect(compiled.latencyMs).toBe(0);
      expect(validateGraph(compiled.graph)).toEqual([]);
    }
  });

  it('rejects unknown plan.kind and invalid PDB crop syntax, and parses Retry-After headers', () => {
    expect(() =>
      compileWorkflowPlanToGraph({ kind: 'mystery_plan' } as unknown as WorkflowPlan),
    ).toThrow(/Unsupported workflow plan kind/);

    expect(() =>
      compileWorkflowPlanToGraph({
        kind: 'pdb',
        pdbId: '3PTB',
        crop: 'A; <script>alert(1)</script>',
        task: 'ligandmpnn',
        targetChain: null,
        regions: null,
        numSequences: 2,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: 'Invalid crop test',
      }),
    ).toThrow(/invalid PDB crop/i);

    expect(parseRetryAfterSec('42')).toBe(42);
    expect(parseRetryAfterSec('0')).toBeNull();
    expect(parseRetryAfterSec(null)).toBeNull();
  });

  it('compiles RFD4 native seq vs ProteinMPNN side-by-side refolding off the same RFD4 backbone', () => {
    const { graph, notes } = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['rfd4-proteina'],
      length: 100,
      numDesigns: 1,
      sequenceDesigners: ['native', 'web-protein-mpnn'],
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      summary: '100aa RFD4-Proteina: compare RFD4 native seq vs ProteinMPNN → AF2 refold',
    });

    expect(notes).toEqual([]);
    expect(validateGraph(graph)).toEqual([]);
    // 1 RFD4 + 1 ProteinMPNN + 2 AF2 folds + 2 scRMSD scorers = 6 nodes
    expect(graph.nodes).toHaveLength(6);
    const rfd4 = graph.nodes.find((n) => n.adapterId === 'rfd4-proteina')!;
    const mpnn = graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn')!;
    const folds = graph.nodes.filter((n) => n.adapterId === 'localfold-af2');
    expect(folds).toHaveLength(2);
    // One fold directly wired to rfd4.sequences, one wired to ProteinMPNN.sequences
    expect(graph.edges.some((e) => e.fromNode === rfd4.id && e.fromPort === 'sequences')).toBe(true);
    expect(graph.edges.some((e) => e.fromNode === mpnn.id && e.fromPort === 'sequences')).toBe(true);
  });

  it('compiles ProtPardelle-1c de novo + ESM-C + Dyna-1 and PDB ESM-C / Dyna-1 evaluation graphs', () => {
    const { graph: deNovoGraph } = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['protpardelle'],
      length: 90,
      numDesigns: 1,
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      esmcScore: true,
      dynamicsScan: true,
      summary: '90aa ProtPardelle-1c → AF2, ESM-C 300M & Dyna-1',
    });
    expect(validateGraph(deNovoGraph)).toEqual([]);
    expect(deNovoGraph.nodes.some((n) => n.adapterId === 'protpardelle')).toBe(true);
    expect(deNovoGraph.nodes.some((n) => n.adapterId === 'esmc')).toBe(true);
    expect(deNovoGraph.nodes.some((n) => n.adapterId === 'dyna1')).toBe(true);

    const { graph: pdbGraph } = compileWorkflowPlanToGraph({
      kind: 'pdb',
      pdbId: '1UBQ',
      crop: null,
      tasks: ['esmc', 'dyna1'],
      targetChain: null,
      regions: null,
      numSequences: null,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: false,
      summary: '1UBQ → ESM-C 300M & Dyna-1 dynamics',
    });
    expect(validateGraph(pdbGraph)).toEqual([]);
    expect(pdbGraph.nodes.map((n) => n.adapterId).sort()).toEqual(['dyna1', 'esmc', 'pdb-input']);
  });
});

function makeReactiveProxy<T>(value: T): T {
  if (value === null || typeof value !== 'object') return value;
  const target = Array.isArray(value)
    ? value.map((item) => makeReactiveProxy(item))
    : Object.fromEntries(
        Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, makeReactiveProxy(v)]),
      );
  return new Proxy(target as object, {
    get(t, prop, receiver) {
      return Reflect.get(t, prop, receiver);
    },
  }) as T;
}

describe('WorkflowPlan R1–R4 regressions & Q1–Q10 reconciliation', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('R1: clones Proxy-wrapped Svelte $state graphs safely without DataCloneError and applies remove_nodes & update_params', () => {
    const base = compileWorkflowPlanToGraph({
      kind: 'pdb',
      pdbId: '1IGT',
      crop: null,
      tasks: ['antifold', 'esmc'],
      targetChain: 'H',
      regions: 'CDR1 CDR3',
      numSequences: 2,
      topK: 2,
      refold: true,
      foldModels: ['monomer'],
      stabilityScan: false,
      summary: '1IGT base',
    }).graph;

    // Confirm structuredClone throws on our Proxy wrapper (proving we test the exact Svelte $state failure mode)
    const proxied = makeReactiveProxy(base);
    expect(() => structuredClone(proxied)).toThrow();

    // 1. Starter plan compiles cleanly against Proxy-wrapped currentGraph
    const starterRes = compileLocalStarterPlan(LOCAL_EXAMPLE_PLANS[0].plan, proxied);
    expect(validateGraph(starterRes.graph)).toEqual([]);
    expect(starterRes.graph.nodes.some((n) => n.adapterId === 'thermompnn')).toBe(true);

    // 2. remove_nodes splices pass-through Top-K cleanly without breaking AF2 -> scRMSD
    const spliceRes = compileWorkflowPlanToGraph(
      {
        kind: 'edit_current',
        operations: [
          {
            action: 'remove_nodes',
            attachToNodeId: null,
            model: null,
            foldModels: [],
            topK: null,
            numSequences: null,
            targetChain: null,
            regions: null,
            score: false,
            targetAdapterIds: ['top-k-filter'],
          },
        ],
        summary: 'Remove Top-K',
      },
      proxied,
    );
    expect(validateGraph(spliceRes.graph)).toEqual([]);
    expect(spliceRes.graph.nodes.some((n) => n.adapterId === 'top-k-filter')).toBe(false);
    expect(spliceRes.graph.nodes.some((n) => n.adapterId === 'localfold-af2')).toBe(true);
    expect(
      spliceRes.graph.edges.some(
        (e) =>
          spliceRes.graph.nodes.find((n) => n.id === e.fromNode)?.adapterId === 'antifold-esmif1' &&
          spliceRes.graph.nodes.find((n) => n.id === e.toNode)?.adapterId === 'localfold-af2',
      ),
    ).toBe(true);

    // 3. remove_nodes cascades orphaned downstream AF2 & scRMSD when AntiFold is removed alongside ESM-C
    const removeRes = compileWorkflowPlanToGraph(
      parseDeterministicEditOperations('Remove AntiFold and ESM-C', proxied)!,
      proxied,
    );
    expect(validateGraph(removeRes.graph)).toEqual([]);
    expect(removeRes.graph.nodes.map((n) => n.adapterId)).toEqual(['pdb-input']);

    // 4. Combined add_stability_scan + update_params (temperature, omitAminoAcids, mpnnFamily)
    const mpnnBase = makeReactiveProxy(
      compileWorkflowPlanToGraph({
        kind: 'de_novo',
        generators: ['rfd4-proteina'],
        length: 80,
        numDesigns: 1,
        sequenceDesigners: ['web-protein-mpnn'],
        numSequences: 2,
        topK: null,
        refold: true,
        foldModels: ['monomer'],
        score: true,
        stabilityScan: false,
        summary: 'RFD4 base',
      }).graph,
    );
    const editOps = parseDeterministicEditOperations(
      'Add ThermoMPNN after AlphaFold2 and change MPNN temperature to 0.2',
      mpnnBase,
    )!;
    const edited = compileWorkflowPlanToGraph(editOps, mpnnBase);
    expect(validateGraph(edited.graph)).toEqual([]);
    const mpnnNode = edited.graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn')!;
    expect(mpnnNode.params.temperature).toBe(0.2);
    const af2Node = edited.graph.nodes.find((n) => n.adapterId === 'localfold-af2')!;
    const thermoNode = edited.graph.nodes.find((n) => n.adapterId === 'thermompnn')!;
    expect(
      edited.graph.edges.some(
        (e) => e.fromNode === af2Node.id && e.toNode === thermoNode.id,
      ),
    ).toBe(true);

    // 5. update_params clamps temperature & updates omitAminoAcids & mpnnFamily -> SolubleMPNN
    const paramUpdated = compileWorkflowPlanToGraph(
      {
        kind: 'edit_current',
        operations: [
          {
            action: 'update_params',
            attachToNodeId: null,
            model: null,
            foldModels: [],
            topK: null,
            numSequences: 4,
            targetChain: null,
            regions: null,
            score: false,
            temperature: 9.0,
            omitAminoAcids: 'c, c',
            mpnnFamily: 'solublempnn',
          },
        ],
        summary: 'Update MPNN params',
      },
      mpnnBase,
    );
    expect(validateGraph(paramUpdated.graph)).toEqual([]);
    const updatedMpnn = paramUpdated.graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn')!;
    expect(updatedMpnn.params.temperature).toBe(1.5);
    expect(updatedMpnn.params.omitAminoAcids).toBe('C');
    expect(updatedMpnn.params.numSequences).toBe(4);
    expect(updatedMpnn.params.family).toBe('solublempnn');
    expect(updatedMpnn.label).toBe('SolubleMPNN');
  });

  it('R1 & R2: requestWorkflowPlan recovers from 502/empty-ops edit responses, uses v2 session cache, and preserves 429/502 non-edit errors', async () => {
    const store = new Map<string, string>();
    vi.stubGlobal('sessionStorage', {
      getItem: vi.fn((k: string) => store.get(k) ?? null),
      setItem: vi.fn((k: string, v: string) => {
        store.set(k, v);
      }),
    });

    const baseGraph = makeReactiveProxy(
      compileWorkflowPlanToGraph({
        kind: 'pdb',
        pdbId: '1IGT',
        crop: null,
        tasks: ['antifold', 'esmc'],
        targetChain: 'H',
        regions: 'CDR3',
        numSequences: 2,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: '1IGT',
      }).graph,
    );

    // 1. HTTP 502 fallback on deterministic edit instruction
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ ok: false, code: 'ai_error' }), { status: 502 })),
    );
    const recovered = await requestWorkflowPlan('Remove AntiFold and ESM-C', {
      currentGraph: baseGraph,
      skipSessionCache: true,
    });
    expect(recovered.model).toBe('client-edit-fallback');
    expect(recovered.graph.nodes.map((n) => n.adapterId)).toEqual(['pdb-input']);
    expect(store.has('denovo:planner-cache:v2')).toBe(true);
    expect(store.has('denovo:planner-cache:v1')).toBe(false);

    // 2. Empty edit_current ops repaired cleanly
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              ok: true,
              plan: { kind: 'edit_current', operations: [], summary: 'Empty ops' },
            }),
            { status: 200 },
          ),
      ),
    );
    const repaired = await requestWorkflowPlan('Remove AntiFold and ESM-C', {
      currentGraph: baseGraph,
      skipSessionCache: true,
    });
    expect(repaired.graph.nodes.map((n) => n.adapterId)).toEqual(['pdb-input']);

    // 3. HTTP 429 is NOT masked even with currentGraph
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ ok: false, error: 'Rate limited' }), { status: 429 })),
    );
    await expect(
      requestWorkflowPlan('Remove AntiFold and ESM-C', {
        currentGraph: baseGraph,
        skipSessionCache: true,
      }),
    ).rejects.toBeInstanceOf(PlannerHttpError);

    // 4. HTTP 502 on non-edit prompt without currentGraph still throws PlannerHttpError
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ ok: false }), { status: 502 })),
    );
    await expect(
      requestWorkflowPlan('Design a 100aa helical bundle', { skipSessionCache: true }),
    ).rejects.toBeInstanceOf(PlannerHttpError);
  });

  it('R2, R3, R4: compiles sequence, partial diffusion, solublempnn/esmif1/antifold, and pdb ThermoMPNN+foldModels graphs', () => {
    // 1. kind: 'sequence' (Q7 heterodimer)
    const seqPlan: SequenceWorkflowPlan = {
      kind: 'sequence',
      sequences: ['MKTEEELIKK', 'GGGGSVVVVV'],
      label: 'Heterodimer Sequence',
      foldModels: ['multimer'],
      stabilityScan: true,
      esmcScore: true,
      dynamicsScan: true,
      summary: 'Fold heterodimer + ThermoMPNN',
    };
    const seqCompiled = compileWorkflowPlanToGraph(seqPlan);
    expect(validateGraph(seqCompiled.graph)).toEqual([]);
    const seqIn = seqCompiled.graph.nodes.find((n) => n.adapterId === 'sequence-input')!;
    expect(seqIn.params.sequence).toBe('MKTEEELIKK:GGGGSVVVVV');
    expect(seqCompiled.graph.nodes.find((n) => n.adapterId === 'localfold-af2')?.params.model).toBe(
      'multimer',
    );
    expect(seqCompiled.graph.nodes.some((n) => n.adapterId === 'thermompnn')).toBe(true);
    expect(seqCompiled.graph.nodes.some((n) => n.adapterId === 'esmc')).toBe(true);
    expect(seqCompiled.graph.nodes.some((n) => n.adapterId === 'dyna1')).toBe(true);

    // 2. designTask: 'partial' (Q8 1UBQ) filters Genie 3, clamps partialFraction, wires ESM-IF1
    const partialCompiled = compileWorkflowPlanToGraph({
      kind: 'conditioned_design',
      designTask: 'partial',
      partialFraction: 0.99,
      benchmarkId: null,
      pdbId: '1UBQ',
      crop: null,
      generators: ['genie3', 'rfd4-proteina'],
      length: null,
      numDesigns: 2,
      sequenceDesigners: ['esmif1'],
      numSequences: 3,
      temperature: 0.25,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      summary: 'Partial diffusion 1UBQ + ESM-IF1',
    });
    expect(validateGraph(partialCompiled.graph)).toEqual([]);
    expect(partialCompiled.graph.nodes.some((n) => n.adapterId === 'genie3')).toBe(false);
    const rfd4 = partialCompiled.graph.nodes.find((n) => n.adapterId === 'rfd4-proteina')!;
    expect(rfd4.params.mode).toBe('partial');
    expect(rfd4.params.partialFraction).toBe(0.95);
    expect(rfd4.params.numDesigns).toBe(2);
    const esmIf1 = partialCompiled.graph.nodes.find((n) => n.adapterId === 'antifold-esmif1')!;
    expect(esmIf1.label).toBe('ESM-IF1');
    expect(esmIf1.params.variant).toBe('esmif1');
    expect(esmIf1.params.temperature).toBe(0.25);

    // 3. pdb downstream eval with foldModels + refold: false & non-refold stabilityScan (Q2, Q3)
    const pdbEvalCompiled = compileWorkflowPlanToGraph({
      kind: 'pdb',
      pdbId: '1CTT',
      crop: null,
      tasks: ['ligandmpnn'],
      targetChain: null,
      regions: null,
      numSequences: 2,
      omitAminoAcids: 'C',
      temperature: 0.2,
      topK: null,
      refold: false,
      foldModels: [],
      stabilityScan: true,
      esmcScore: true,
      dynamicsScan: true,
      summary: '1CTT LigandMPNN + ThermoMPNN',
    });
    expect(validateGraph(pdbEvalCompiled.graph)).toEqual([]);
    const lig = pdbEvalCompiled.graph.nodes.find((n) => n.adapterId === 'ligandmpnn')!;
    expect(lig.params.omitAminoAcids).toBe('C');
    expect(lig.params.temperature).toBe(0.2);
    expect(pdbEvalCompiled.graph.nodes.some((n) => n.adapterId === 'thermompnn')).toBe(true);
    expect(pdbEvalCompiled.graph.nodes.some((n) => n.adapterId === 'esmc')).toBe(true);
    expect(pdbEvalCompiled.graph.nodes.some((n) => n.adapterId === 'dyna1')).toBe(true);

    // 4. pdb thermompnn task with foldModels: ['monomer'] still wires downstream refold/score cleanly
    const thermoRefoldCompiled = compileWorkflowPlanToGraph({
      kind: 'pdb',
      pdbId: '1UBQ',
      crop: null,
      tasks: ['thermompnn'],
      targetChain: null,
      regions: null,
      numSequences: 2,
      topK: null,
      refold: false,
      foldModels: ['monomer'],
      stabilityScan: true,
      summary: '1UBQ ThermoMPNN + AF2',
    });
    expect(validateGraph(thermoRefoldCompiled.graph)).toEqual([]);
    expect(thermoRefoldCompiled.graph.nodes.some((n) => n.adapterId === 'thermompnn')).toBe(true);
    expect(thermoRefoldCompiled.graph.nodes.some((n) => n.adapterId === 'localfold-af2')).toBe(true);
    expect(thermoRefoldCompiled.graph.nodes.some((n) => n.adapterId === 'designability-scorer')).toBe(true);

    // 5. SolubleMPNN labeling on solublempnn / mpnnFamily: 'solublempnn'
    const solubleCompiled = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['rfd4-proteina'],
      length: 70,
      numDesigns: 1,
      sequenceDesigners: ['solublempnn'],
      numSequences: 2,
      omitAminoAcids: 'C, M',
      temperature: 0.15,
      topK: null,
      refold: true,
      foldModels: ['monomer'],
      score: true,
      stabilityScan: false,
      summary: 'SolubleMPNN de novo',
    });
    expect(validateGraph(solubleCompiled.graph)).toEqual([]);
    const solNode = solubleCompiled.graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn')!;
    expect(solNode.label).toBe('SolubleMPNN');
    expect(solNode.params.family).toBe('solublempnn');
    expect(solNode.params.omitAminoAcids).toBe('CM');
    expect(solNode.params.temperature).toBe(0.15);
  });

  it('R2–R4: reconcilePlannedWorkflow repairs Q1–Q9 v7 desyncs and leaves Q10 unsupported untouched', () => {
    // Q1: 5O45 65aa binder with SolubleMPNN omit C & AF2-Multimer
    const q1 = reconcilePlannedWorkflow(
      {
        kind: 'conditioned_design',
        designTask: 'binder',
        benchmarkId: 'pdl1_5o45',
        pdbId: '5O45',
        crop: null,
        generators: ['rfd4-proteina'],
        length: 65,
        numDesigns: 1,
        sequenceDesigners: ['antifold'],
        numSequences: 2,
        topK: null,
        refold: true,
        foldModels: ['multimer'],
        score: true,
        stabilityScan: false,
        summary: 'Q1',
      },
      'Design a 65aa binder to 5O45 with RFD4, SolubleMPNN omitting C, and AF2-Multimer',
    ) as ConditionedDesignWorkflowPlan;
    expect(q1.mpnnFamily).toBe('solublempnn');
    expect(q1.omitAminoAcids).toBe('C');
    expect(q1.sequenceDesigners).toEqual(['solublempnn']);
    const q1Graph = compileWorkflowPlanToGraph(q1).graph;
    expect(q1Graph.nodes.find((n) => n.adapterId === 'rfd4-proteina')?.params.contig).toBe(
      'A17-132,/0,65-65',
    );
    expect(q1Graph.nodes.find((n) => n.adapterId === 'web-protein-mpnn')?.params.omitAminoAcids).toBe('C');

    // Q2: 1IGT CDR1+CDR3 AntiFold + Top-K (3) + RF3
    const q2 = reconcilePlannedWorkflow(
      {
        kind: 'pdb',
        pdbId: '1IGT',
        crop: null,
        tasks: ['antifold'],
        targetChain: 'H',
        regions: 'CDR3',
        numSequences: 4,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: 'Q2',
      },
      'Redesign 1IGT CDR1 and CDR3 with AntiFold, Top-K (3), and RoseTTAFold 3',
    ) as PdbWorkflowPlan;
    expect(q2.topK).toBe(3);
    expect(q2.foldModels).toEqual(['rosettafold3']);
    expect(q2.refold).toBe(true);
    expect(q2.regions).toBe('CDR1 CDR3');
    const q2Graph = compileWorkflowPlanToGraph(q2).graph;
    expect(q2Graph.nodes.some((n) => n.adapterId === 'top-k-filter' && n.params.topK === 3)).toBe(true);
    expect(q2Graph.nodes.some((n) => n.adapterId === 'localfold-af2' && n.params.model === 'rosettafold3')).toBe(true);

    // Q3: 1CTT LigandMPNN + ThermoMPNN misrouted to conditioned_design -> pdb
    const q3 = reconcilePlannedWorkflow(
      {
        kind: 'conditioned_design',
        designTask: 'enzyme-ligand',
        benchmarkId: 'ame_1ctt_cda',
        pdbId: '1CTT',
        crop: null,
        generators: ['rfd4-proteina'],
        length: null,
        numDesigns: 1,
        numSequences: 2,
        topK: null,
        refold: true,
        foldModels: ['monomer'],
        score: true,
        stabilityScan: false,
        summary: 'Q3',
      },
      'Run LigandMPNN and ThermoMPNN on 1CTT',
    ) as PdbWorkflowPlan;
    expect(q3.kind).toBe('pdb');
    expect(q3.pdbId).toBe('1CTT');
    expect(q3.tasks).toEqual(['ligandmpnn', 'thermompnn']);
    const q3Graph = compileWorkflowPlanToGraph(q3).graph;
    expect(q3Graph.nodes.some((n) => n.adapterId === 'ligandmpnn')).toBe(true);
    expect(q3Graph.nodes.some((n) => n.adapterId === 'thermompnn')).toBe(true);
    expect(q3Graph.nodes.some((n) => n.adapterId === 'rfd4-proteina')).toBe(false);

    // Q4: 1YCR B19-27 80aa motif scaffold -> contig 35-35,B19-27,36-36
    const q4 = reconcilePlannedWorkflow(
      {
        kind: 'conditioned_design',
        designTask: 'motif',
        benchmarkId: 'rfd_1ycr_p53',
        pdbId: '1YCR',
        crop: null,
        generators: ['genie3'],
        length: null,
        numDesigns: 1,
        numSequences: 2,
        topK: null,
        refold: true,
        foldModels: ['monomer'],
        score: true,
        stabilityScan: false,
        summary: 'Q4',
      },
      'Scaffold 1YCR B19-27 into an 80aa protein with Genie 3, AF2, and motif RMSD',
    ) as ConditionedDesignWorkflowPlan;
    expect(q4.length).toBe(80);
    const q4Graph = compileWorkflowPlanToGraph(q4).graph;
    expect(q4Graph.nodes.find((n) => n.adapterId === 'genie3')?.params.contig).toBe('35-35,B19-27,36-36');

    // Q6: 2×90aa 3-way cross-folder fork
    const q6 = reconcilePlannedWorkflow(
      {
        kind: 'de_novo',
        generators: ['rfd4-proteina'],
        length: 90,
        numDesigns: 1,
        sequenceDesigners: ['web-protein-mpnn'],
        numSequences: 2,
        topK: null,
        refold: true,
        foldModels: ['monomer'],
        score: true,
        stabilityScan: false,
        summary: 'Q6',
      },
      'Generate 2×90aa de novo proteins with RFD4, ProteinMPNN, and compare AF2 vs Boltz-2 vs RF3',
    ) as DeNovoWorkflowPlan;
    expect(q6.numDesigns).toBe(2);
    expect(q6.length).toBe(90);
    expect(q6.foldModels).toEqual(['monomer', 'boltz2', 'rosettafold3']);
    const q6Graph = compileWorkflowPlanToGraph(q6).graph;
    expect(q6Graph.nodes.find((n) => n.adapterId === 'rfd4-proteina')?.params.numDesigns).toBe(2);
    expect(q6Graph.nodes.filter((n) => n.adapterId === 'localfold-af2')).toHaveLength(3);

    // Q7: Heterodimer sequence folding promoted from stale pdb demo fallback
    const q7 = reconcilePlannedWorkflow(
      {
        kind: 'pdb',
        pdbId: 'demo',
        crop: null,
        tasks: ['thermompnn'],
        targetChain: null,
        regions: null,
        numSequences: null,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: 'Q7',
      },
      'Fold heterodimer sequence MKTEEELIKK + GGGGSVVVVV with AF2-Multimer and ThermoMPNN',
    ) as SequenceWorkflowPlan;
    expect(q7.kind).toBe('sequence');
    expect(q7.sequences).toEqual(['MKTEEELIKK', 'GGGGSVVVVV']);
    expect(q7.foldModels).toEqual(['multimer']);
    expect(q7.stabilityScan).toBe(true);

    // Q8: Partial diffusion + ESM-IF1 on 1UBQ strips unrequested AntiFold
    const q8 = reconcilePlannedWorkflow(
      {
        kind: 'pdb',
        pdbId: '1UBQ',
        crop: null,
        tasks: ['antifold'],
        targetChain: null,
        regions: null,
        numSequences: 2,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: 'Q8',
      },
      'Run partial diffusion on 1UBQ with RFD4-Proteina, ESM-IF1, and check designability',
    ) as ConditionedDesignWorkflowPlan;
    expect(q8.kind).toBe('conditioned_design');
    expect(q8.designTask).toBe('partial');
    expect(q8.pdbId).toBe('1UBQ');
    expect(q8.sequenceDesigners).toEqual(['esmif1']);
    expect(q8.score).toBe(true);

    // Q9: Vague stabilization/anti-aggregation routes antifold -> SolubleMPNN + stabilityScan
    const q9 = reconcilePlannedWorkflow(
      {
        kind: 'pdb',
        pdbId: 'demo',
        crop: null,
        tasks: ['antifold'],
        targetChain: null,
        regions: null,
        numSequences: 2,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: 'Q9',
      },
      'Stabilize a small protein and avoid aggregation',
    ) as PdbWorkflowPlan;
    expect(q9.tasks).toContain('solublempnn');
    expect(q9.tasks).not.toContain('antifold');
    expect(q9.stabilityScan).toBe(true);

    // Q10: Unsupported plan stays completely unaltered and never mutates active currentGraph
    const activeSnapshot: WorkflowGraph = {
      id: 'active-g',
      name: 'Active',
      nodes: [{ id: 'pdb_1', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } }],
      edges: [],
    };
    const q10Input: WorkflowPlan = {
      kind: 'unsupported',
      unsupported: ['RDKit', 'AutoDock Vina'],
      reason: 'Small-molecule docking is not supported.',
      suggestions: ['Use LigandMPNN'],
      summary: 'Unsupported',
    };
    const q10Out = reconcilePlannedWorkflow(
      q10Input,
      'Run RDKit and AutoDock Vina docking on 3PTB',
      activeSnapshot,
    );
    expect(q10Out).toBe(q10Input);
    expect(() => compileWorkflowPlanToGraph(q10Out, activeSnapshot)).toThrow(UnsupportedWorkflowError);
    expect(activeSnapshot.nodes).toHaveLength(1);
  });

  it('Iteration 2 regressions: scoped remove/param edit parsing, de_novo singular sequenceDesigner, pdb AntiFold restore, pdb topK+stabilityScan, and ALL-CAPS English prompt guard', () => {
    // 1. Bug 1: Remove AntiFold and change MPNN temperature to 0.2 removes ONLY af_1 and updates mpnn_1 temp to 0.2
    const canvasBug1: WorkflowGraph = {
      id: 'canvas-bug1',
      name: 'Bug 1 Canvas',
      nodes: [
        { id: 'pdb_1', adapterId: 'pdb-input', label: 'PDB', params: { mode: 'demo-bundle' } },
        {
          id: 'mpnn_1',
          adapterId: 'web-protein-mpnn',
          label: 'ProteinMPNN',
          params: { family: 'proteinmpnn', temperature: 0.1, numSequences: 2 },
        },
        {
          id: 'af_1',
          adapterId: 'antifold-esmif1',
          label: 'AntiFold',
          params: { variant: 'antifold', temperature: 0.5, numSequences: 2 },
        },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' },
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'af_1', toPort: 'structure' },
      ],
    };
    const bug1Plan = parseDeterministicEditOperations(
      'Remove AntiFold and change MPNN temperature to 0.2',
      canvasBug1,
    )!;
    expect(bug1Plan).not.toBeNull();
    const bug1Compiled = compileWorkflowPlanToGraph(bug1Plan, canvasBug1).graph;
    expect(validateGraph(bug1Compiled)).toEqual([]);
    expect(bug1Compiled.nodes.find((n) => n.id === 'af_1')).toBeUndefined();
    const keptMpnn = bug1Compiled.nodes.find((n) => n.id === 'mpnn_1');
    expect(keptMpnn).toBeDefined();
    expect(keptMpnn!.params.temperature).toBe(0.2);

    // 2. Bug 2: Add ThermoMPNN after AntiFold and change MPNN temperature to 0.2 does NOT overwrite af_1 temperature
    const bug2Plan = parseDeterministicEditOperations(
      'Add ThermoMPNN after AntiFold and change MPNN temperature to 0.2',
      canvasBug1,
    )!;
    expect(bug2Plan).not.toBeNull();
    const bug2UpdateOp = bug2Plan.operations.find((o) => o.action === 'update_params')!;
    expect(bug2UpdateOp.targetNodeIds).toEqual(['mpnn_1']);
    const bug2Compiled = compileWorkflowPlanToGraph(bug2Plan, canvasBug1).graph;
    expect(validateGraph(bug2Compiled)).toEqual([]);
    expect(bug2Compiled.nodes.find((n) => n.id === 'mpnn_1')!.params.temperature).toBe(0.2);
    expect(bug2Compiled.nodes.find((n) => n.id === 'af_1')!.params.temperature).toBe(0.5);
    const thermoBug2 = bug2Compiled.nodes.find((n) => n.adapterId === 'thermompnn')!;
    expect(thermoBug2).toBeDefined();
    expect(
      bug2Compiled.edges.some((e) => e.fromNode === 'pdb_1' && e.toNode === thermoBug2.id),
    ).toBe(true);

    // 3. Bug 3: de_novo legacy singular sequenceDesigner: 'antifold' on Q9 stabilization replaces antifold -> solublempnn
    const q9DeNovo = reconcilePlannedWorkflow(
      {
        kind: 'de_novo',
        generators: ['rfd4-proteina'],
        length: 70,
        numDesigns: 1,
        sequenceDesigner: 'antifold',
        numSequences: 2,
        topK: null,
        refold: true,
        foldModels: ['monomer'],
        score: true,
        stabilityScan: false,
        summary: 'Stabilize small protein',
      },
      'Stabilize small protein and avoid aggregation',
    ) as DeNovoWorkflowPlan;
    expect(q9DeNovo.mpnnFamily).toBe('solublempnn');
    expect(q9DeNovo.sequenceDesigners).toEqual(['solublempnn']);
    expect(q9DeNovo.stabilityScan).toBe(true);
    const q9DeNovoGraph = compileWorkflowPlanToGraph(q9DeNovo).graph;
    expect(q9DeNovoGraph.nodes.some((n) => n.adapterId === 'antifold-esmif1')).toBe(false);
    expect(
      q9DeNovoGraph.nodes.some(
        (n) => n.adapterId === 'web-protein-mpnn' && n.label === 'SolubleMPNN',
      ),
    ).toBe(true);

    // 4. Bug 4: pdb AntiFold restoration when degraded v7 returns proteinmpnn for antibody AntiFold prompt
    const q2Restored = reconcilePlannedWorkflow(
      {
        kind: 'pdb',
        pdbId: '1IGT',
        crop: null,
        tasks: ['proteinmpnn'],
        targetChain: null,
        regions: null,
        numSequences: 2,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: 'Q2 degraded',
      },
      'Design CDR1 and CDR3 of antibody 1IGT using AntiFold, take Top-3, and refold with RF3',
    ) as PdbWorkflowPlan;
    expect(q2Restored.tasks).toContain('antifold');
    expect(q2Restored.tasks).not.toContain('proteinmpnn');
    expect(q2Restored.regions).toBe('CDR1 CDR3');
    expect(q2Restored.topK).toBe(3);
    expect(q2Restored.foldModels).toEqual(['rosettafold3']);
    const q2RestoredGraph = compileWorkflowPlanToGraph(q2Restored).graph;
    expect(validateGraph(q2RestoredGraph)).toEqual([]);
    expect(
      q2RestoredGraph.nodes.some(
        (n) => n.adapterId === 'antifold-esmif1' && n.params.variant === 'antifold',
      ),
    ).toBe(true);
    expect(q2RestoredGraph.nodes.some((n) => n.adapterId === 'web-protein-mpnn')).toBe(false);

    // 5. Reviewer caveat 7a: pdb with topK + stabilityScan + refold: false still attaches ThermoMPNN
    const pdbTopKThermo = compileWorkflowPlanToGraph({
      kind: 'pdb',
      pdbId: '1UBQ',
      crop: null,
      tasks: ['proteinmpnn'],
      targetChain: null,
      regions: null,
      numSequences: 4,
      topK: 2,
      refold: false,
      foldModels: [],
      stabilityScan: true,
      summary: '1UBQ MPNN Top-2 + ThermoMPNN',
    }).graph;
    expect(validateGraph(pdbTopKThermo)).toEqual([]);
    expect(pdbTopKThermo.nodes.some((n) => n.adapterId === 'top-k-filter')).toBe(true);
    expect(pdbTopKThermo.nodes.some((n) => n.adapterId === 'thermompnn')).toBe(true);

    // 6. Reviewer caveat 7b: ALL-CAPS English prompts do not falsely trigger kind: 'sequence'
    const allCapsReconciled = reconcilePlannedWorkflow(
      {
        kind: 'de_novo',
        generators: ['rfd4-proteina'],
        length: 80,
        numDesigns: 1,
        numSequences: 2,
        topK: null,
        refold: true,
        foldModels: ['monomer'],
        score: true,
        stabilityScan: false,
        summary: 'All caps de novo',
      },
      'DESIGN A STABLE PEPTIDE BINDER FOR TARGET STRAND WITH FILTER PLEASE CREATE NATIVE CHAINS',
    );
    expect(allCapsReconciled.kind).toBe('de_novo');
  });

  it('Iteration 2 remediation: AF2 selector targets only monomer on multi-oracle fork and uppercase English domain words do not hijack to sequence', () => {
    // 1. Multi-oracle 3-way fork (AF2 Model 1, Boltz-2, RoseTTAFold 3): AF2/AlphaFold2 targets ONLY monomer
    const forkGraph = compileWorkflowPlanToGraph({
      kind: 'de_novo',
      generators: ['rfd4-proteina'],
      length: 90,
      numDesigns: 1,
      sequenceDesigners: ['web-protein-mpnn'],
      numSequences: 2,
      topK: null,
      refold: true,
      foldModels: ['monomer', 'boltz2', 'rosettafold3'],
      score: true,
      stabilityScan: false,
      summary: '90aa 3-way fork',
    }).graph;
    const af2Node = forkGraph.nodes.find(
      (n) => n.adapterId === 'localfold-af2' && String(n.params.model ?? 'monomer').startsWith('monomer'),
    )!;
    const boltzNode = forkGraph.nodes.find(
      (n) => n.adapterId === 'localfold-af2' && n.params.model === 'boltz2',
    )!;
    const rf3Node = forkGraph.nodes.find(
      (n) => n.adapterId === 'localfold-af2' && n.params.model === 'rosettafold3',
    )!;
    expect(af2Node).toBeDefined();
    expect(boltzNode).toBeDefined();
    expect(rf3Node).toBeDefined();

    // Add ThermoMPNN after AlphaFold2 attaches specifically to AF2 Model 1, NOT RF3/Boltz-2
    const addAfterAf2 = parseDeterministicEditOperations('Add ThermoMPNN after AlphaFold2', forkGraph)!;
    expect(addAfterAf2).not.toBeNull();
    expect(addAfterAf2.operations[0].attachToNodeId).toBe(af2Node.id);
    const addedGraph = compileWorkflowPlanToGraph(addAfterAf2, forkGraph).graph;
    expect(validateGraph(addedGraph)).toEqual([]);
    const thermo = addedGraph.nodes.find((n) => n.adapterId === 'thermompnn')!;
    expect(addedGraph.edges.some((e) => e.fromNode === af2Node.id && e.toNode === thermo.id)).toBe(true);
    expect(addedGraph.edges.some((e) => e.fromNode === rf3Node.id && e.toNode === thermo.id)).toBe(false);

    // Remove AF2 removes ONLY AF2 Model 1, keeping Boltz-2 and RoseTTAFold 3 intact
    const removeAf2 = parseDeterministicEditOperations('Remove AF2', forkGraph)!;
    expect(removeAf2).not.toBeNull();
    expect(removeAf2.operations[0].targetNodeIds).toEqual([af2Node.id]);
    const removedGraph = compileWorkflowPlanToGraph(removeAf2, forkGraph).graph;
    expect(validateGraph(removedGraph)).toEqual([]);
    expect(removedGraph.nodes.some((n) => n.id === af2Node.id)).toBe(false);
    expect(removedGraph.nodes.some((n) => n.id === boltzNode.id)).toBe(true);
    expect(removedGraph.nodes.some((n) => n.id === rf3Node.id)).toBe(true);

    // 2. Uppercase English domain words (CHANNEL, ACIDIC, HELICAL, ACTIVITY, SPECIFICITY) do not trigger kind: 'sequence'
    for (const domainPrompt of [
      'Redesign this CHANNEL protein with SolubleMPNN and fold with AF2',
      'Redesign this ACIDIC HELICAL domain with SolubleMPNN and fold with AF2',
      'Optimize ACTIVITY and SPECIFICITY of this protein with SolubleMPNN and fold with AF2',
    ]) {
      const pdbFallback: PdbWorkflowPlan = {
        kind: 'pdb',
        pdbId: 'demo',
        crop: null,
        tasks: ['solublempnn'],
        targetChain: null,
        regions: null,
        numSequences: 2,
        topK: null,
        refold: true,
        foldModels: ['monomer'],
        stabilityScan: false,
        summary: 'demo -> SolubleMPNN -> AF2',
      };
      const reconciled = reconcilePlannedWorkflow(pdbFallback, domainPrompt);
      expect(reconciled.kind).toBe('pdb');
    }

    // Real uppercase AA sequences (including shorter delimited or sequence-cued chains) remain preserved
    const realSeq = reconcilePlannedWorkflow(
      {
        kind: 'pdb',
        pdbId: 'demo',
        crop: null,
        tasks: ['thermompnn'],
        targetChain: null,
        regions: null,
        numSequences: null,
        topK: null,
        refold: false,
        foldModels: [],
        stabilityScan: false,
        summary: 'demo -> ThermoMPNN',
      },
      'Fold sequence MKTEAL:GGGGSVVVVV with AF2-Multimer',
    ) as SequenceWorkflowPlan;
    expect(realSeq.kind).toBe('sequence');
    expect(realSeq.sequences).toEqual(['MKTEAL', 'GGGGSVVVVV']);
  });
});
