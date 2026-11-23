import { describe, expect, it } from 'vitest';
import {
  classifyParamUndoAction,
  computeBenchmarkAutoSeedPatches,
  computeChangedParamKeys,
  computePreviewChecks,
  isSharedWorkflowHash,
  parseLengthFromContig,
  shouldAllowNativeTextUndo,
  shouldSuppressOnboardingForHash,
  shouldSyncNumericDraft,
} from '$lib/ui/page-state-helpers';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { decodeHashToGraph } from '$lib/engine/share';
import type { CanonicalArtifact, StructureArtifact, WorkflowGraph } from '$lib/engine/types';
import {
  compileWorkflowPlanToGraph,
  type PlannedWorkflowResult,
} from '$lib/engine/workflow-planner';

function makeDummyStructure(id: string, sequence: string): StructureArtifact {
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'pdb-input',
    length: sequence.length,
    sequence,
    caCoords: new Float32Array(sequence.length * 3),
    backbone4Coords: new Float32Array(sequence.length * 12),
    pdbText: 'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n',
    geometry: {
      length: sequence.length,
      rg: 12,
      meanCaDist: 3.8,
      helixPct: 50,
      sheetPct: 0,
      coilPct: 50,
      chainBreaks: 0,
      clashes: 0,
    },
  };
}

describe('page-state-helpers (R5, R3 auto-seed, QA #1/#2, QA #5)', () => {
  it('R5: computePreviewChecks isolates new-graph previews from stale bioIssues & inputPreviewErrors while keeping incremental edits scoped', () => {
    // Committed Q1 graph: 5O45 binder + AF2-Multimer (with outputsByNode)
    const q1Graph: WorkflowGraph = {
      id: 'q1-5o45',
      name: 'Q1 5O45 Binder',
      nodes: [
        {
          id: 'pdb1',
          adapterId: 'pdb-input',
          label: '5O45',
          params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' },
        },
        {
          id: 'af2',
          adapterId: 'localfold-af2',
          label: 'AF2-Multimer',
          params: { ...defaultParamsForAdapter('localfold-af2'), model: 'multimer' },
        },
      ],
      edges: [{ fromNode: 'pdb1', fromPort: 'structure', toNode: 'af2', toPort: 'sequences' }],
    };

    // Unwired/invalid sequence + AF2-Multimer triggers bioIssue on monomer input -> AF2-Multimer
    const q1WithAntiFold: WorkflowGraph = {
      id: 'q1-bio',
      name: 'Q1',
      nodes: [
        {
          id: 'pdb1',
          adapterId: 'pdb-input',
          label: '1UBQ',
          params: { mode: 'pdb-id', pdbId: '1UBQ', benchmarkId: '' },
        },
        {
          id: 'af',
          adapterId: 'antifold-esmif1',
          label: 'AntiFold',
          params: { ...defaultParamsForAdapter('antifold-esmif1'), variant: 'antifold' },
        },
      ],
      edges: [{ fromNode: 'pdb1', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      pdb1: { structure: makeDummyStructure('ubq', 'M'.repeat(76)) },
    };
    const committedChecks = computePreviewChecks(q1WithAntiFold, null, outputsByNode, {
      pdb1: '1UBQ: simulated error',
    });
    expect(committedChecks.bioIssues.some((i) => i.code === 'antifold-non-antibody')).toBe(true);
    expect(committedChecks.displayedValidationErrors.some((e) => e.includes('simulated error'))).toBe(true);
    expect(committedChecks.hideBioFixes).toBe(false);
    expect(committedChecks.hideRunError).toBe(false);

    // Previewing a new Q2 graph (1IGT antibody + AntiFold) must NOT leak Q1's bioIssues or inputPreviewErrors
    const q2Graph: WorkflowGraph = {
      id: 'q2-1igt',
      name: 'Q2 1IGT',
      nodes: [
        {
          id: 'pdb1',
          adapterId: 'pdb-input',
          label: '1IGT',
          params: { mode: 'pdb-id', pdbId: '1IGT', fetchedPdbId: '1IGT', pdbText: 'ATOM...' },
        },
        {
          id: 'af',
          adapterId: 'antifold-esmif1',
          label: 'AntiFold',
          params: { ...defaultParamsForAdapter('antifold-esmif1'), variant: 'antifold' },
        },
      ],
      edges: [{ fromNode: 'pdb1', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    const newPreview: PlannedWorkflowResult = {
      graph: q2Graph,
      plan: { kind: 'pdb', summary: 'Q2' } as PlannedWorkflowResult['plan'],
      isIncrementalEdit: false,
      addedNodeCount: 0,
      unsupported: [],
      model: 'test',
      latencyMs: 1,
      notes: [],
    };
    const previewChecks = computePreviewChecks(q1WithAntiFold, newPreview, outputsByNode, {
      pdb1: '1UBQ: simulated error',
    });
    expect(previewChecks.bioIssues.some((i) => i.code === 'antifold-non-antibody')).toBe(false);
    expect(previewChecks.displayedValidationErrors.some((e) => e.includes('simulated error'))).toBe(false);
    expect(previewChecks.hideBioFixes).toBe(true);
    expect(previewChecks.hideRunError).toBe(true);

    // Incremental edit keeps matching node outputs/errors and keeps runError visible
    const incPreview: PlannedWorkflowResult = {
      graph: q1WithAntiFold,
      plan: { kind: 'edit_current', summary: 'Edit' } as PlannedWorkflowResult['plan'],
      isIncrementalEdit: true,
      addedNodeCount: 0,
      unsupported: [],
      model: 'test',
      latencyMs: 1,
      notes: [],
    };
    const incChecks = computePreviewChecks(q1WithAntiFold, incPreview, outputsByNode, {
      pdb1: '1UBQ: simulated error',
    });
    expect(incChecks.bioIssues.some((i) => i.code === 'antifold-non-antibody')).toBe(true);
    expect(incChecks.displayedValidationErrors.some((e) => e.includes('simulated error'))).toBe(true);
    expect(incChecks.hideBioFixes).toBe(true);
    expect(incChecks.hideRunError).toBe(false);
    expect(q1Graph.nodes).toHaveLength(2);
  });

  it('R3: parseLengthFromContig and computeBenchmarkAutoSeedPatches adapt benchmark contigs to custom/planned lengths', () => {
    expect(parseLengthFromContig('A17-132,/0,70-70')).toBe(70);
    expect(parseLengthFromContig('150-150')).toBe(150);
    expect(parseLengthFromContig('28-28,B19-27,28-28')).toBe(65);

    // 5O45 + RFD4 with planned length: 65 preserves 65 and produces A17-132,/0,65-65
    const g5o45: WorkflowGraph = {
      id: 'g-5o45',
      name: '5O45',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '5O45',
          params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' },
        },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          label: 'RFD4',
          params: { ...defaultParamsForAdapter('rfd4-proteina'), length: 65 },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' }],
    };
    const p5o45 = computeBenchmarkAutoSeedPatches(g5o45);
    expect(p5o45?.rfd?.length).toBe(65);
    expect(p5o45?.rfd?.contig).toBe('A17-132,/0,65-65');
    expect(p5o45?.rfd?.autoSeededBenchmarkId).toBe('pdl1_5o45');

    // 5O45 + Genie 3: default length (80) adopts benchmark 70; explicit 65 preserves 65
    const gGenieDefault: WorkflowGraph = {
      id: 'g-genie-def',
      name: 'Genie Default',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '5O45',
          params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' },
        },
        {
          id: 'g3',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: { ...defaultParamsForAdapter('genie3'), length: 80 },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    };
    const pGenieDef = computeBenchmarkAutoSeedPatches(gGenieDefault);
    expect(pGenieDef?.g3?.length).toBe(70);
    expect(pGenieDef?.g3?.contig).toBe('A17-132,/0,70-70');

    const gGenieCustom: WorkflowGraph = {
      ...gGenieDefault,
      nodes: [
        gGenieDefault.nodes[0],
        { ...gGenieDefault.nodes[1], params: { ...defaultParamsForAdapter('genie3'), length: 65 } },
      ],
    };
    const pGenieCustom = computeBenchmarkAutoSeedPatches(gGenieCustom);
    expect(pGenieCustom?.g3?.length).toBe(65);
    expect(pGenieCustom?.g3?.contig).toBe('A17-132,/0,65-65');

    // 1YCR motif (rfd_1ycr_p53) + RFD4 with custom length: 80 produces 35-35,B19-27,36-36 (total 80 aa)
    const g1ycr: WorkflowGraph = {
      id: 'g-1ycr',
      name: '1YCR',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1YCR',
          params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'rfd_1ycr_p53' },
        },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          label: 'RFD4',
          params: { ...defaultParamsForAdapter('rfd4-proteina'), length: 80 },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' }],
    };
    const p1ycr = computeBenchmarkAutoSeedPatches(g1ycr);
    expect(p1ycr?.rfd?.length).toBe(80);
    expect(p1ycr?.rfd?.contig).toBe('35-35,B19-27,36-36');

    // Switching pdl1_5o45 -> mdm2_1ycr (when previously auto-seeded at 70) updates to 60
    const gSwitch: WorkflowGraph = {
      id: 'g-switch',
      name: 'Switch',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1YCR',
          params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'mdm2_1ycr' },
        },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          label: 'RFD4',
          params: {
            ...defaultParamsForAdapter('rfd4-proteina'),
            length: 70,
            autoSeededBenchmarkId: 'pdl1_5o45',
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' }],
    };
    const pSwitch = computeBenchmarkAutoSeedPatches(gSwitch);
    expect(pSwitch?.rfd?.length).toBe(60);

    // Switching pdl1_5o45 -> ame_1ctt_cda resets Genie 3 to unconditional
    const gResetGenie: WorkflowGraph = {
      id: 'g-reset',
      name: 'Reset',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1CTT',
          params: { mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda' },
        },
        {
          id: 'g3',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: {
            ...defaultParamsForAdapter('genie3'),
            taskMode: 'binder',
            contig: 'A17-132,/0,70-70',
            autoSeededBenchmarkId: 'pdl1_5o45',
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    };
    const pReset = computeBenchmarkAutoSeedPatches(gResetGenie);
    expect(pReset?.g3?.taskMode).toBe('unconditional');
    expect(pReset?.g3?.contig).toBe('');

    // Skips when already seeded for same benchmark or userEditedTask is true
    expect(
      computeBenchmarkAutoSeedPatches({
        ...g5o45,
        nodes: [
          g5o45.nodes[0],
          { ...g5o45.nodes[1], params: { ...g5o45.nodes[1].params, autoSeededBenchmarkId: 'pdl1_5o45' } },
        ],
      }),
    ).toBeNull();
    expect(
      computeBenchmarkAutoSeedPatches({
        ...g5o45,
        nodes: [
          g5o45.nodes[0],
          { ...g5o45.nodes[1], params: { ...g5o45.nodes[1].params, userEditedTask: true } },
        ],
      }),
    ).toBeNull();

    // Preserves unseeded partial, inverse-folding, and preset modes (never clobbers to binder/motif)
    for (const explicitMode of ['partial', 'inverse-folding', 'preset'] as const) {
      expect(
        computeBenchmarkAutoSeedPatches({
          ...g5o45,
          nodes: [
            g5o45.nodes[0],
            {
              ...g5o45.nodes[1],
              params: {
                ...defaultParamsForAdapter('rfd4-proteina'),
                mode: explicitMode,
                partialFraction: 0.4,
              },
            },
          ],
        }),
      ).toBeNull();
    }

    // Preserves mode: 'partial' when switching benchmarks on a planner-seeded node (autoSeededBenchmarkId set, no userEditedTask)
    expect(
      computeBenchmarkAutoSeedPatches({
        id: 'g-partial-switch',
        name: 'Partial Switch',
        nodes: [
          {
            id: 'pdb',
            adapterId: 'pdb-input',
            label: '1YCR',
            params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'mdm2_1ycr' },
          },
          {
            id: 'rfd',
            adapterId: 'rfd4-proteina',
            label: 'RFD4',
            params: {
              ...defaultParamsForAdapter('rfd4-proteina'),
              mode: 'partial',
              partialFraction: 0.5,
              autoSeededBenchmarkId: 'pdl1_5o45',
            },
          },
        ],
        edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' }],
      }),
    ).toBeNull();

    // Genie 3 multi-hop switch: pdl1_5o45 (70) -> ame_1ctt_cda (unsupported, resets to 80) -> mdm2_1ycr (60)
    const gGenieHop1 = computeBenchmarkAutoSeedPatches(gGenieDefault);
    expect(gGenieHop1?.g3?.length).toBe(70);
    const gGenieAtCda: WorkflowGraph = {
      id: 'g-genie-cda',
      name: 'Genie CDA',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1CTT',
          params: { mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda' },
        },
        {
          ...gGenieDefault.nodes[1],
          params: gGenieHop1!.g3,
        },
      ],
      edges: gGenieDefault.edges,
    };
    const gGenieHop2 = computeBenchmarkAutoSeedPatches(gGenieAtCda);
    expect(gGenieHop2?.g3?.taskMode).toBe('unconditional');
    expect(gGenieHop2?.g3?.length).toBe(80);
    const gGenieAtMdm2: WorkflowGraph = {
      id: 'g-genie-mdm2',
      name: 'Genie MDM2',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1YCR',
          params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'mdm2_1ycr' },
        },
        {
          ...gGenieDefault.nodes[1],
          params: gGenieHop2!.g3,
        },
      ],
      edges: gGenieDefault.edges,
    };
    const gGenieHop3 = computeBenchmarkAutoSeedPatches(gGenieAtMdm2);
    expect(gGenieHop3?.g3?.length).toBe(60);
    expect(gGenieHop3?.g3?.contig).toBe('A25-109,/0,60-60');

    // userEditedLength: true preserves length even when equal to adapterDefaultLen (80 on Genie 3)
    const gGenieUserLen = computeBenchmarkAutoSeedPatches({
      id: 'g-genie-userlen',
      name: 'Genie UserLen',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1YCR',
          params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'mdm2_1ycr' },
        },
        {
          id: 'g3',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: {
            ...defaultParamsForAdapter('genie3'),
            length: 80,
            userEditedLength: true,
            autoSeededBenchmarkId: 'pdl1_5o45',
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    });
    expect(gGenieUserLen?.g3?.length).toBe(80);
    expect(gGenieUserLen?.g3?.contig).toBe('A25-109,/0,80-80');
  });

  it('R3: genie3 preserves custom/planned lengths across unsupported benchmark hops and guards non-numeric autoSeededLength', () => {
    const unsupportedCollisions: Array<{
      unsupportedBenchId: string;
      pdbId: string;
      length: number;
    }> = [
      { unsupportedBenchId: 'rfd4_7m5w_bdna', pdbId: '7M5W', length: 90 },
      { unsupportedBenchId: 'ame_1ctt_cda', pdbId: '1CTT', length: 180 },
      { unsupportedBenchId: 'sm_7c7m_sam', pdbId: '7C7M', length: 110 },
      { unsupportedBenchId: 'ame_1os7_taud', pdbId: '1OS7', length: 180 },
    ];

    for (const { unsupportedBenchId, pdbId, length } of unsupportedCollisions) {
      const atPdl1: WorkflowGraph = {
        id: `g-hop-pdl1-${length}`,
        name: 'Hop PDL1',
        nodes: [
          {
            id: 'pdb',
            adapterId: 'pdb-input',
            label: '5O45',
            params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' },
          },
          {
            id: 'g3',
            adapterId: 'genie3',
            label: 'Genie 3',
            params: { ...defaultParamsForAdapter('genie3'), length },
          },
        ],
        edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
      };
      const hop1 = computeBenchmarkAutoSeedPatches(atPdl1);
      expect(hop1?.g3?.length).toBe(length);
      expect(hop1?.g3?.contig).toBe(`A17-132,/0,${length}-${length}`);
      expect(hop1?.g3?.autoSeededLength).toBeUndefined();

      const atUnsupported: WorkflowGraph = {
        ...atPdl1,
        nodes: [
          {
            ...atPdl1.nodes[0],
            label: pdbId,
            params: { mode: 'pdb-id', pdbId, benchmarkId: unsupportedBenchId },
          },
          { ...atPdl1.nodes[1], params: hop1!.g3 },
        ],
      };
      const hop2 = computeBenchmarkAutoSeedPatches(atUnsupported);
      expect(hop2?.g3?.taskMode).toBe('unconditional');
      expect(hop2?.g3?.length).toBe(length);
      expect(hop2?.g3?.autoSeededBenchmarkId).toBe(unsupportedBenchId);
      expect(hop2?.g3?.autoSeededLength).toBeUndefined();

      const atMdm2: WorkflowGraph = {
        ...atPdl1,
        nodes: [
          {
            ...atPdl1.nodes[0],
            label: '1YCR',
            params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'mdm2_1ycr' },
          },
          { ...atPdl1.nodes[1], params: hop2!.g3 },
        ],
      };
      const hop3 = computeBenchmarkAutoSeedPatches(atMdm2);
      expect(hop3?.g3?.taskMode).toBe('binder');
      expect(hop3?.g3?.length).toBe(length);
      expect(hop3?.g3?.contig).toBe(`A25-109,/0,${length}-${length}`);
      expect(hop3?.g3?.autoSeededBenchmarkId).toBe('mdm2_1ycr');
    }

    // autoSeededLength: null or '' must not coerce to 0 (falls back to prevBenchLen when valid, or null when unsupported)
    for (const badAutoLen of [null, ''] as const) {
      const fromSupported = computeBenchmarkAutoSeedPatches({
        id: 'g-bad-autolen-supported',
        name: 'Bad AutoLen Supported',
        nodes: [
          {
            id: 'pdb',
            adapterId: 'pdb-input',
            label: '1YCR',
            params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'mdm2_1ycr' },
          },
          {
            id: 'g3',
            adapterId: 'genie3',
            label: 'Genie 3',
            params: {
              ...defaultParamsForAdapter('genie3'),
              length: 70,
              autoSeededBenchmarkId: 'pdl1_5o45',
              autoSeededLength: badAutoLen,
            },
          },
        ],
        edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
      });
      expect(fromSupported?.g3?.length).toBe(60);
      expect(fromSupported?.g3?.contig).toBe('A25-109,/0,60-60');

      const fromUnsupported = computeBenchmarkAutoSeedPatches({
        id: 'g-bad-autolen-unsupported',
        name: 'Bad AutoLen Unsupported',
        nodes: [
          {
            id: 'pdb',
            adapterId: 'pdb-input',
            label: '1YCR',
            params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'mdm2_1ycr' },
          },
          {
            id: 'g3',
            adapterId: 'genie3',
            label: 'Genie 3',
            params: {
              ...defaultParamsForAdapter('genie3'),
              length: 90,
              autoSeededBenchmarkId: 'rfd4_7m5w_bdna',
              autoSeededLength: badAutoLen,
            },
          },
        ],
        edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
      });
      expect(fromUnsupported?.g3?.length).toBe(90);
      expect(fromUnsupported?.g3?.contig).toBe('A25-109,/0,90-90');
    }
  });

  it('R3: preserves planned custom lengths across supported-benchmark multi-hop collisions and guards non-positive autoSeededLength / length', () => {
    const switchBenchmark = (
      graph: WorkflowGraph,
      adapterId: 'genie3' | 'rfd4-proteina',
      benchmarkId: string,
      pdbId: string,
      nextNodeParams: Record<string, unknown>,
    ): WorkflowGraph => ({
      ...graph,
      nodes: graph.nodes.map((n) => {
        if (n.adapterId === 'pdb-input') {
          return {
            ...n,
            label: pdbId,
            params: { ...n.params, mode: 'pdb-id', pdbId, benchmarkId },
          };
        }
        if (n.adapterId === adapterId) {
          return { ...n, params: nextNodeParams };
        }
        return n;
      }),
    });

    // (a) compileWorkflowPlanToGraph 60 aa genie3 across pdl1_5o45 -> mdm2_1ycr -> pdl1_5o45
    const geniePlan = compileWorkflowPlanToGraph({
      kind: 'conditioned_design',
      designTask: 'binder',
      benchmarkId: 'pdl1_5o45',
      pdbId: '5O45',
      crop: null,
      generators: ['genie3'],
      length: 60,
      numDesigns: 1,
      sequenceDesigners: ['web-protein-mpnn'],
      numSequences: 1,
      topK: null,
      refold: false,
      foldModels: [],
      score: false,
      stabilityScan: false,
      summary: '5O45 60aa genie3 binder',
    });
    const genieNode0 = geniePlan.graph.nodes.find((n) => n.adapterId === 'genie3')!;
    expect(genieNode0.params.length).toBe(60);
    expect(genieNode0.params.contig).toBe('A17-132,/0,60-60');

    const genieHop1 = computeBenchmarkAutoSeedPatches(
      switchBenchmark(geniePlan.graph, 'genie3', 'mdm2_1ycr', '1YCR', genieNode0.params),
    );
    expect(genieHop1?.[genieNode0.id]?.length).toBe(60);
    expect(genieHop1?.[genieNode0.id]?.contig).toBe('A25-109,/0,60-60');
    expect(genieHop1?.[genieNode0.id]?.userEditedLength).toBe(true);
    expect(genieHop1?.[genieNode0.id]?.autoSeededLength).toBeUndefined();

    const genieHop2 = computeBenchmarkAutoSeedPatches(
      switchBenchmark(geniePlan.graph, 'genie3', 'pdl1_5o45', '5O45', genieHop1![genieNode0.id]),
    );
    expect(genieHop2?.[genieNode0.id]?.length).toBe(60);
    expect(genieHop2?.[genieNode0.id]?.contig).toBe('A17-132,/0,60-60');
    expect(genieHop2?.[genieNode0.id]?.userEditedLength).toBe(true);

    // (b) compileWorkflowPlanToGraph 180 aa rfd4-proteina across pdl1_5o45 -> ame_1ctt_cda -> mdm2_1ycr
    const rfd4Plan = compileWorkflowPlanToGraph({
      kind: 'conditioned_design',
      designTask: 'binder',
      benchmarkId: 'pdl1_5o45',
      pdbId: '5O45',
      crop: null,
      generators: ['rfd4-proteina'],
      length: 180,
      numDesigns: 1,
      sequenceDesigners: ['web-protein-mpnn'],
      numSequences: 1,
      topK: null,
      refold: false,
      foldModels: [],
      score: false,
      stabilityScan: false,
      summary: '5O45 180aa rfd4 binder',
    });
    const rfd4Node0 = rfd4Plan.graph.nodes.find((n) => n.adapterId === 'rfd4-proteina')!;
    expect(rfd4Node0.params.length).toBe(180);
    expect(rfd4Node0.params.contig).toBe('A17-132,/0,180-180');

    const rfd4Hop1 = computeBenchmarkAutoSeedPatches(
      switchBenchmark(rfd4Plan.graph, 'rfd4-proteina', 'ame_1ctt_cda', '1CTT', rfd4Node0.params),
    );
    expect(rfd4Hop1?.[rfd4Node0.id]?.length).toBe(180);
    expect(rfd4Hop1?.[rfd4Node0.id]?.contig).toBe('180-180');
    expect(rfd4Hop1?.[rfd4Node0.id]?.userEditedLength).toBe(true);
    expect(rfd4Hop1?.[rfd4Node0.id]?.autoSeededLength).toBeUndefined();

    const rfd4Hop2 = computeBenchmarkAutoSeedPatches(
      switchBenchmark(rfd4Plan.graph, 'rfd4-proteina', 'mdm2_1ycr', '1YCR', rfd4Hop1![rfd4Node0.id]),
    );
    expect(rfd4Hop2?.[rfd4Node0.id]?.length).toBe(180);
    expect(rfd4Hop2?.[rfd4Node0.id]?.contig).toBe('A25-109,/0,180-180');
    expect(rfd4Hop2?.[rfd4Node0.id]?.userEditedLength).toBe(true);

    // (c) autoSeededLength: 0 on supported pdl1_5o45 (70 -> mdm2_1ycr 60) and unsupported rfd4_7m5w_bdna (90 -> mdm2_1ycr 90)
    const zeroAutoLenSupported = computeBenchmarkAutoSeedPatches({
      id: 'g-zero-autolen-supported',
      name: 'Zero AutoLen Supported',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1YCR',
          params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'mdm2_1ycr' },
        },
        {
          id: 'g3',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: {
            ...defaultParamsForAdapter('genie3'),
            length: 70,
            autoSeededBenchmarkId: 'pdl1_5o45',
            autoSeededLength: 0,
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    });
    expect(zeroAutoLenSupported?.g3?.length).toBe(60);
    expect(zeroAutoLenSupported?.g3?.contig).toBe('A25-109,/0,60-60');

    const zeroAutoLenUnsupported = computeBenchmarkAutoSeedPatches({
      id: 'g-zero-autolen-unsupported',
      name: 'Zero AutoLen Unsupported',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1YCR',
          params: { mode: 'pdb-id', pdbId: '1YCR', benchmarkId: 'mdm2_1ycr' },
        },
        {
          id: 'g3',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: {
            ...defaultParamsForAdapter('genie3'),
            length: 90,
            autoSeededBenchmarkId: 'rfd4_7m5w_bdna',
            autoSeededLength: 0,
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    });
    expect(zeroAutoLenUnsupported?.g3?.length).toBe(90);
    expect(zeroAutoLenUnsupported?.g3?.contig).toBe('A25-109,/0,90-90');
    expect(zeroAutoLenUnsupported?.g3?.userEditedLength).toBe(true);

    // Non-positive cp.length (e.g. 0) without userEditedLength resets to 80 on unsupported Genie 3 benchmark
    const zeroLenReset = computeBenchmarkAutoSeedPatches({
      id: 'g-zero-len-reset',
      name: 'Zero Len Reset',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1CTT',
          params: { mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda' },
        },
        {
          id: 'g3',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: {
            ...defaultParamsForAdapter('genie3'),
            length: 0,
            autoSeededBenchmarkId: 'pdl1_5o45',
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    });
    expect(zeroLenReset?.g3?.taskMode).toBe('unconditional');
    expect(zeroLenReset?.g3?.length).toBe(80);
    expect(zeroLenReset?.g3?.autoSeededLength).toBe(80);
  });

  it('QA #1 & #2: hash helpers detect shared links, return null on broken hashes, and suppress onboarding', async () => {
    expect(isSharedWorkflowHash('#g=broken123')).toBe(true);
    expect(isSharedWorkflowHash('#p=unknown')).toBe(true);
    expect(isSharedWorkflowHash('')).toBe(false);
    expect(isSharedWorkflowHash('#section')).toBe(false);

    expect(await decodeHashToGraph('#g=broken123')).toBeNull();
    expect(await decodeHashToGraph('#p=unknown-preset-xyz')).toBeNull();

    expect(shouldSuppressOnboardingForHash(false, true)).toBe(true);
    expect(shouldSuppressOnboardingForHash(true, false)).toBe(true);
    expect(shouldSuppressOnboardingForHash(false, false)).toBe(false);
  });

  it('QA #5: undo guard, NumericInput draft sync, and param/label coalescing behave consistently', () => {
    // Non-text controls allow app-level Cmd+Z undo; textareas & text inputs retain native text undo
    expect(shouldAllowNativeTextUndo({ tagName: 'SELECT' })).toBe(false);
    expect(shouldAllowNativeTextUndo({ tagName: 'INPUT', type: 'checkbox' })).toBe(false);
    expect(shouldAllowNativeTextUndo({ tagName: 'INPUT', type: 'range' })).toBe(false);
    expect(shouldAllowNativeTextUndo({ tagName: 'INPUT', type: 'file' })).toBe(false);
    expect(
      shouldAllowNativeTextUndo({
        tagName: 'INPUT',
        type: 'text',
        classList: { contains: (c: string) => c === 'num-input' },
      }),
    ).toBe(false);
    expect(shouldAllowNativeTextUndo({ tagName: 'TEXTAREA' })).toBe(true);
    expect(shouldAllowNativeTextUndo({ tagName: 'INPUT', type: 'text' })).toBe(true);
    expect(shouldAllowNativeTextUndo({ tagName: 'INPUT', type: 'text', disabled: true })).toBe(false);
    expect(shouldAllowNativeTextUndo({ tagName: 'DIV', isContentEditable: true })).toBe(true);

    // NumericInput syncs on external undo/redo even while focused, skipping only self-echoes
    expect(shouldSyncNumericDraft(true, true)).toBe(false);
    expect(shouldSyncNumericDraft(true, false)).toBe(true);
    expect(shouldSyncNumericDraft(false, true)).toBe(true);

    // Changed param key computation excludes userEditedTask and userEditedLength
    expect(
      computeChangedParamKeys(
        { temperature: 0.1, userEditedTask: false, userEditedLength: false },
        { temperature: 0.2, userEditedTask: true, userEditedLength: true },
      ),
    ).toEqual(['temperature']);

    // Coalescing & atomic label+param merge
    const labelMarker = { nodeId: 'n1', at: 1000 };
    expect(classifyParamUndoAction('n1', 'variant', 1010, null, labelMarker)).toBe('merge-with-label');
    const paramMarker = { nodeId: 'n1', paramKey: 'temperature', at: 1000 };
    expect(classifyParamUndoAction('n1', 'temperature', 1500, paramMarker, null)).toBe('coalesce');
    expect(classifyParamUndoAction('n1', 'temperature', 1900, paramMarker, null)).toBe('push');
    expect(classifyParamUndoAction('n1', 'numSequences', 1200, paramMarker, null)).toBe('push');
  });
});
