import { describe, expect, it } from 'vitest';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  buildCanonicalConditionIntent,
  checkGenie3ConditionCompatibility,
  compileConditionToGenie3Spec,
  compileConditionToRfd4,
} from '$lib/engine/conditioning';
import { BENCHMARK_TARGETS } from '$lib/engine/benchmarks';
import { isValidPdbId } from '$lib/protein-ops/rcsb';
import { generateSampleHelixBundlePdb, parseStructureText } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { validateConnection } from '$lib/engine/graph-utils';
import type { ConditionSpecArtifact, StructureArtifact, WorkflowGraph } from '$lib/engine/types';

function makeSampleTemplate(): StructureArtifact {
  const pdbText = generateSampleHelixBundlePdb();
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id: 'tpl-sample',
    label: 'sample_53aa.pdb',
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    diagnostics: parsed.diagnostics,
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds),
  };
}

describe('Model-Agnostic Conditioning & Curated Benchmark Targets', () => {
  it('compiles canonical intent cleanly into RFD4 sel(...) condition dicts', () => {
    const intent = buildCanonicalConditionIntent({
      contig: 'A17-132,/0,70-70',
      lockChain: 'A',
      hotspots: 'A54, A56, A115',
      motifSpan: 'A16-35',
      unindexedMotif: true,
      hbondMode: 'ACCEPTOR',
      hbondSelection: 'B/SAM/*/*',
      cathCode: '3.40.50',
    });

    const rfd4 = compileConditionToRfd4(intent);
    expect(rfd4.contig).toBe('A17-132,/0,70-70');
    expect(rfd4.conditions.C_CRD).toBeDefined();
    expect(rfd4.conditions.C_HOT).toEqual({
      true: [{ select: "sel('A/*/[54,56,115]/*')" }],
    });
    expect(rfd4.conditions.C_CRDGRP).toEqual({
      1: [{ select: "sel('A/*/[16-35]/*')" }],
    });
    expect(rfd4.conditions.C_IDX).toBeDefined();
    expect(rfd4.conditions.C_HB).toEqual({ ACCEPTOR: [{ select: "sel('B/SAM/*/*')" }] });
    expect(rfd4.conditions.C_CATH).toEqual({ '3.40.50': [{ select: "sel('*/*/*/*')" }] });
  });

  it('compiles PPI binder & motif scaffolding intents into valid Genie 3 createFeaturesFromSpec batches', async () => {
    const tpl = makeSampleTemplate();

    // 1. PPI Binder Design: 40-aa binder against target A5-34 (30 aa) with hotspots A10, A15
    const binderIntent = buildCanonicalConditionIntent({
      contig: 'A5-34,/0,40-40',
      lockChain: 'A',
      hotspots: 'A10, A15',
    });
    const binderCond: ConditionSpecArtifact = {
      kind: 'ConditionSpec',
      id: 'cond-binder',
      label: 'Binder',
      contig: binderIntent.contig,
      conditions: {},
      intent: binderIntent,
      summary: 'binder',
    };
    const binderSpec = compileConditionToGenie3Spec(binderCond, tpl, 60);
    expect(binderSpec.mode).toBe('binder');
    expect(binderSpec.length).toBe(70); // 40 binder + 30 target
    expect(binderSpec.chainLengths).toEqual([40, 30]);
    expect(binderSpec.asymId[0]).toBe(0);
    expect(binderSpec.asymId[39]).toBe(0);
    expect(binderSpec.asymId[40]).toBe(1);
    expect(binderSpec.condStructMask[39]).toBe(0);
    expect(binderSpec.condStructMask[40]).toBe(1);
    // Hotspot A10 is offset 5 into A5-34 => index 40 + 5 = 45; A15 => 40 + 10 = 50
    expect(binderSpec.condInterfaceMask[45]).toBe(1);
    expect(binderSpec.condInterfaceMask[50]).toBe(1);
    expect(binderSpec.condInterfaceMask[46]).toBe(0);

    // 2. Motif Scaffolding: 10-10,A10-20,10-10 (10 + 11 + 10 = 31 aa)
    const motifIntent = buildCanonicalConditionIntent({
      contig: '10-10,A10-20,10-10',
      motifSpan: 'A10-20',
    });
    const motifCond: ConditionSpecArtifact = {
      kind: 'ConditionSpec',
      id: 'cond-motif',
      label: 'Motif',
      contig: motifIntent.contig,
      conditions: {},
      intent: motifIntent,
      summary: 'motif',
    };
    const motifSpec = compileConditionToGenie3Spec(motifCond, tpl, 60);
    expect(motifSpec.mode).toBe('motif');
    expect(motifSpec.length).toBe(31);
    expect(motifSpec.condStructMask[9]).toBe(0);
    expect(motifSpec.condStructMask[10]).toBe(1);
    expect(motifSpec.condStructMask[20]).toBe(1);
    expect(motifSpec.condStructMask[21]).toBe(0);

    // 3. Feed both compiled specs straight into upstream Genie 3 `createFeaturesFromSpec`
    const featuresPath = resolve(__dirname, '../../../static/vendor/genie3/shared/genie3/features.js');
    const g3Features = await import(pathToFileURL(featuresPath).href);
    const binderBatch = g3Features.createFeaturesFromSpec(binderSpec);
    expect(binderBatch.length).toBe(70);
    expect(binderBatch.cond_struct_frame_mask[45]).toBe(1);
    expect(binderBatch.cond_interface_mask[45]).toBe(1);

    const motifBatch = g3Features.createFeaturesFromSpec(motifSpec);
    expect(motifBatch.length).toBe(31);
    expect(motifBatch.cond_struct_frame_mask[15]).toBe(1);
  });

  it('rejects RFD4-only constraints (H-bonds, CATH, unindexed motif, RFD4 preset) on Genie 3 and warns in Layer 1', () => {
    const hbondCond: ConditionSpecArtifact = {
      kind: 'ConditionSpec',
      id: 'hb',
      label: 'HB',
      contig: '80-80',
      conditions: {},
      intent: buildCanonicalConditionIntent({
        contig: '80-80',
        hbondMode: 'ACCEPTOR',
        hbondSelection: 'B/SAM/*/*',
      }),
      summary: 'HB',
    };
    expect(checkGenie3ConditionCompatibility(hbondCond).supported).toBe(false);

    const graph: WorkflowGraph = {
      id: 'g3-warn',
      name: 'Genie3 Warn',
      nodes: [
        {
          id: 'cond',
          adapterId: 'rfd4-conditioner',
          params: { taskPreset: 'custom', contig: '80-80', cathCode: '3.40.50' },
        },
        { id: 'g3', adapterId: 'genie3', params: { length: 80 } },
      ],
      edges: [{ fromNode: 'cond', fromPort: 'condition', toNode: 'g3', toPort: 'condition' }],
    };
    expect(validateConnection(graph, 'cond', 'condition', 'g3', 'condition').valid).toBe(true);
    const issues = validateBioCapabilities(graph);
    expect(issues.some((i) => i.code === 'genie3-unsupported-condition' && i.nodeId === 'g3')).toBe(true);
  });

  it('curated BENCHMARK_TARGETS all have unique IDs, valid RCSB PDB IDs, and valid WebGPU token budgets', () => {
    expect(BENCHMARK_TARGETS.length).toBeGreaterThanOrEqual(16);
    const ids = new Set(BENCHMARK_TARGETS.map((b) => b.id));
    expect(ids.size).toBe(BENCHMARK_TARGETS.length);
    for (const b of BENCHMARK_TARGETS) {
      expect(isValidPdbId(b.pdbId), `Invalid PDB ID on ${b.id}: ${b.pdbId}`).toBe(true);
      expect(b.approxTokens).toBeGreaterThan(20);
      expect(b.approxTokens).toBeLessThanOrEqual(300);
      expect(b.conditionParams.contig.length).toBeGreaterThan(0);
    }
  });
});
