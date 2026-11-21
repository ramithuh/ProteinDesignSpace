import { describe, expect, it } from 'vitest';
import {
  analyzeSequence,
  extractSubgraphUpTo,
  findContextStructure,
  foldModelSlug,
  formatFoldModelName,
  getNodePeekSlots,
  pickDefaultPeekSlot,
  previewTextLines,
} from '$lib/engine/peek';
import { LOCALFOLD_MODEL_OPTIONS, WORKFLOW_PRESETS } from '$lib/engine/registry';
import type { CanonicalArtifact, SequenceSet, StructureArtifact } from '$lib/engine/types';

function makeDummyStructure(id: string, sequence = 'MKTAYIAKQRQISFVKSHFSRQ'): StructureArtifact {
  return {
    kind: 'StructureArtifact',
    id,
    label: `struct-${id}`,
    sourceModelId: 'pdb-input',
    length: sequence.length,
    sequence,
    caCoords: new Float32Array(sequence.length * 3),
    backbone4Coords: new Float32Array(sequence.length * 12),
    pdbText: 'ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n',
    geometry: {
      length: sequence.length,
      rg: 10.5,
      meanCaDist: 3.8,
      helixPct: 60,
      sheetPct: 0,
      coilPct: 40,
      chainBreaks: 0,
      clashes: 0,
    },
  };
}

function makeDummySeqSet(id: string, seq = 'MKTAYIAKQRQISFVKSHFSRQ'): SequenceSet {
  return {
    kind: 'SequenceSet',
    id,
    parentStructureId: 's1',
    sourceModelId: 'web-protein-mpnn',
    candidates: [
      {
        id: 'cand-1',
        sequence: seq,
        score: 0.92,
        globalScore: 0.95,
        sourceModelId: 'web-protein-mpnn',
      },
    ],
  };
}

describe('peek helpers', () => {
  const mpnnPreset = WORKFLOW_PRESETS.find((p) => p.id === 'pdb-refold-qc')!.graph;
  const forkPreset = WORKFLOW_PRESETS.find((p) => p.id === 'rfd4-branch-compare')!.graph;

  it('resolves incoming and outgoing peek slots for an intermediate node', () => {
    const struct = makeDummyStructure('s1');
    const seqs = makeDummySeqSet('q1');
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      src: { structure: struct },
      mpnn: { sequences: seqs },
    };

    const mpnnSlots = getNodePeekSlots(mpnnPreset, 'mpnn', outputsByNode);
    expect(mpnnSlots).toHaveLength(2);
    expect(mpnnSlots[0]).toMatchObject({
      id: 'in:structure',
      direction: 'in',
      portName: 'structure',
      portType: 'StructureArtifact',
      wired: true,
      peerNodeId: 'src',
      peerNodeLabel: 'Backbone Input',
      artifact: struct,
    });
    expect(mpnnSlots[1]).toMatchObject({
      id: 'out:sequences',
      direction: 'out',
      portName: 'sequences',
      portType: 'SequenceSet',
      artifact: seqs,
    });

    // While LocalFold has not finished yet, its incoming sequences slot is already live!
    const foldSlots = getNodePeekSlots(mpnnPreset, 'fold', outputsByNode);
    expect(foldSlots.find((s) => s.id === 'in:sequences')?.artifact).toBe(seqs);
    expect(foldSlots.find((s) => s.id === 'out:folds')?.artifact).toBeNull();
    expect(pickDefaultPeekSlot(foldSlots)).toBe('in:sequences');
  });

  it('prefers outgoing live output over incoming when both are available', () => {
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      src: { structure: makeDummyStructure('s1') },
      mpnn: { sequences: makeDummySeqSet('q1') },
    };
    const slots = getNodePeekSlots(mpnnPreset, 'mpnn', outputsByNode);
    expect(pickDefaultPeekSlot(slots)).toBe('out:sequences');
  });

  it('extractSubgraphUpTo slices only upstream ancestors on branched DAGs', () => {
    const subNative = extractSubgraphUpTo(forkPreset, 'fold_native');
    expect(subNative.nodes.map((n) => n.id)).toEqual(['gen', 'fold_native']);
    expect(subNative.edges).toEqual([
      { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold_native', toPort: 'sequences' },
    ]);

    const subMpnn = extractSubgraphUpTo(forkPreset, 'mpnn');
    expect(subMpnn.nodes.map((n) => n.id)).toEqual(['gen', 'mpnn']);
    expect(subMpnn.edges).toHaveLength(1);

    const subSource = extractSubgraphUpTo(mpnnPreset, 'src');
    expect(subSource.nodes.map((n) => n.id)).toEqual(['src']);
    expect(subSource.edges).toEqual([]);
  });

  it('findContextStructure walks upstream to locate the closest backbone', () => {
    const struct = makeDummyStructure('s1');
    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      src: { structure: struct },
      mpnn: { sequences: makeDummySeqSet('q1') },
    };
    expect(findContextStructure(mpnnPreset, 'fold', outputsByNode)).toBe(struct);
    expect(findContextStructure(mpnnPreset, 'mpnn', outputsByNode)).toBe(struct);
  });

  it('analyzeSequence flags poly-ALA collapse & unknown X and computes sequence recovery', () => {
    const polyAla = analyzeSequence('A'.repeat(50));
    expect(polyAla.uniqueAaCount).toBe(1);
    expect(polyAla.topAa).toEqual({ aa: 'A', count: 50, pct: 100 });
    expect(polyAla.warnings.some((w) => w.includes('Low complexity'))).toBe(true);

    const withX = analyzeSequence('ACDEFGXXIK');
    expect(withX.unknownCount).toBe(2);
    expect(withX.warnings.some((w) => w.includes("unknown ('X')"))).toBe(true);

    const healthy = analyzeSequence('MKTAYIAKQR', 'MKTAYAAAAA');
    expect(healthy.warnings).toEqual([]);
    expect(healthy.recoveryPct).toBeCloseTo(60, 5);
  });

  it('previewTextLines bounds long PDB outputs cleanly', () => {
    const text = Array.from({ length: 60 }, (_, i) => `LINE ${i + 1}`).join('\n');
    const res = previewTextLines(text, 10);
    expect(res.totalLines).toBe(60);
    expect(res.truncated).toBe(true);
    expect(res.preview.split('\n')).toHaveLength(10);
  });

  it('foldModelSlug and formatFoldModelName format all 13 worker foldModelId values and bare localfold-af2', () => {
    expect(LOCALFOLD_MODEL_OPTIONS).toHaveLength(13);
    for (const opt of LOCALFOLD_MODEL_OPTIONS) {
      const workerId =
        opt.family === 'af2' ? `localfold-af2-${opt.id}-int5` : `localfold-${opt.id}-int5`;
      expect(foldModelSlug(workerId)).toBe(opt.id);
      expect(formatFoldModelName(workerId)).toBe(opt.label);
    }
    expect(foldModelSlug('localfold-af2')).toBe('af2');
    expect(formatFoldModelName('localfold-af2')).toBe('AlphaFold2');
    expect(foldModelSlug('')).toBe('folded');
    expect(formatFoldModelName('')).toBe('folded');
  });
});
