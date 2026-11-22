import { describe, expect, it } from 'vitest';
import {
  analyzeSequence,
  extractSubgraphUpTo,
  findContextStructure,
  foldModelSlug,
  formatCandidateSequence,
  formatFoldModelName,
  getNodePeekSlots,
  pickDefaultPeekSlot,
  previewTextLines,
  resolveActiveRowChainLengths,
  resolveFoldIpsae,
  selectPrimaryOutputPort,
} from '$lib/engine/peek';
import { LOCALFOLD_MODEL_OPTIONS, WORKFLOW_PRESETS } from '$lib/engine/registry';
import type {
  CanonicalArtifact,
  MutationMatrixArtifact,
  ResidueTrackArtifact,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

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

  it('findContextStructure prioritizes incoming structure edges over folds edges on auto-wired scorers', () => {
    const designedBackbone: StructureArtifact = {
      ...makeDummyStructure('gen-backbone'),
      sourceModelId: 'rfd4-proteina',
      trajectoryPdb: 'MODEL        1\nATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 90.00           C\nENDMDL\n',
    };
    const refoldedStruct: StructureArtifact = {
      ...makeDummyStructure('lf-refold'),
      sourceModelId: 'localfold-af2',
    };

    // Simulate auto-wired edge order where fold.folds -> score.folds precedes gen.structure -> score.structure
    const autoWiredGraph = {
      id: 'auto-wired-scorer',
      name: 'Auto-Wired Scorer',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4-Proteina', params: {} },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: {} },
        { id: 'score', adapterId: 'designability-scorer', label: 'scRMSD / scTM', params: {} },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
      ],
    };

    const outputsByNode: Record<string, Record<string, CanonicalArtifact>> = {
      gen: { structure: designedBackbone, sequences: makeDummySeqSet('q-gen') },
      fold: { structure: refoldedStruct },
    };

    const resolved = findContextStructure(autoWiredGraph, 'score', outputsByNode);
    expect(resolved).toBe(designedBackbone);
    expect(resolved?.trajectoryPdb).toBeDefined();
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

  it('previewTextLines bounds long PDB outputs cleanly (including CRLF and maxLines=0)', () => {
    const text = Array.from({ length: 60 }, (_, i) => `LINE ${i + 1}`).join('\n');
    const res = previewTextLines(text, 10);
    expect(res.totalLines).toBe(60);
    expect(res.truncated).toBe(true);
    expect(res.preview.split('\n')).toHaveLength(10);

    const crlf = 'LINE 1\r\nLINE 2\r\nLINE 3';
    const crlfRes = previewTextLines(crlf, 2);
    expect(crlfRes.totalLines).toBe(3);
    expect(crlfRes.truncated).toBe(true);
    expect(crlfRes.preview).toBe('LINE 1\nLINE 2');

    const zeroRes = previewTextLines(crlf, 0);
    expect(zeroRes.preview).toBe('');
    expect(zeroRes.totalLines).toBe(3);
    expect(zeroRes.truncated).toBe(true);
  });

  it('formats all registered worker foldModelId values and bare localfold-af2', () => {
    expect(LOCALFOLD_MODEL_OPTIONS.map((m) => m.id)).toEqual(expect.arrayContaining(['ef2-fast-300m', 'ef2-fast-600m']));
    for (const opt of LOCALFOLD_MODEL_OPTIONS) {
      const workerId =
        opt.family === 'af2' || opt.family === 'af2-multimer'
          ? `localfold-af2-${opt.id}-int5`
          : `localfold-${opt.id}-int5`;
      expect(foldModelSlug(workerId)).toBe(opt.id);
      expect(formatFoldModelName(workerId)).toBe(opt.label);
    }
    expect(foldModelSlug('localfold-af2')).toBe('af2');
    expect(formatFoldModelName('localfold-af2')).toBe('AlphaFold2');
    expect(foldModelSlug('')).toBe('folded');
    expect(formatFoldModelName('')).toBe('folded');
  });

  it('Compare/Peek #5: prioritizes MutationMatrix/ResidueTrack over StructureArtifact and keeps StructureArtifact above SequenceSet', () => {
    const struct = makeDummyStructure('s1');
    const seqs = makeDummySeqSet('q1');
    const mut: MutationMatrixArtifact = {
      kind: 'MutationMatrix',
      id: 'mut-1',
      sourceModelId: 'thermompnn',
      metric: 'ddg-kcal-mol',
      direction: 'lower-is-better',
      residues: 'MK',
      residueIds: ['A1', 'A2'],
      alphabet: 'ACDEFGHIKLMNPQRSTVWY',
      values: new Float32Array(40).fill(-0.5),
    };
    const track: ResidueTrackArtifact = {
      kind: 'ResidueTrack',
      id: 'trk-1',
      sourceModelId: 'dyna1',
      quantity: 'exchange-prob',
      residues: 'MK',
      residueIds: ['A1', 'A2'],
      values: new Float32Array([0.1, 0.2]),
      summary: { min: 0.1, max: 0.2, mean: 0.15 },
    };

    // Order-independent pickDefaultPeekSlot for ThermoMPNN (mutations vs structure)
    const thermoSlotsOrderA = [
      { id: 'in:structure', direction: 'in' as const, portName: 'structure', portType: 'StructureArtifact' as const, required: true, wired: true, artifact: struct },
      { id: 'out:mutations', direction: 'out' as const, portName: 'mutations', portType: 'MutationMatrix' as const, required: true, wired: true, artifact: mut },
      { id: 'out:structure', direction: 'out' as const, portName: 'structure', portType: 'StructureArtifact' as const, required: true, wired: true, artifact: struct },
    ];
    const thermoSlotsOrderB = [thermoSlotsOrderA[0], thermoSlotsOrderA[2], thermoSlotsOrderA[1]];
    expect(pickDefaultPeekSlot(thermoSlotsOrderA)).toBe('out:mutations');
    expect(pickDefaultPeekSlot(thermoSlotsOrderB)).toBe('out:mutations');

    // Dyna-1 prefers ResidueTrack over StructureArtifact and SequenceSet
    const dynaSlots = [
      { id: 'out:structure', direction: 'out' as const, portName: 'structure', portType: 'StructureArtifact' as const, required: true, wired: true, artifact: struct },
      { id: 'out:sequences', direction: 'out' as const, portName: 'sequences', portType: 'SequenceSet' as const, required: true, wired: true, artifact: seqs },
      { id: 'out:track', direction: 'out' as const, portName: 'track', portType: 'ResidueTrack' as const, required: true, wired: true, artifact: track },
    ];
    expect(pickDefaultPeekSlot(dynaSlots)).toBe('out:track');

    // De novo generators (RFD4 / Genie 3) preserve 3D StructureArtifact above SequenceSet
    const genSlots = [
      { id: 'out:sequences', direction: 'out' as const, portName: 'sequences', portType: 'SequenceSet' as const, required: true, wired: true, artifact: seqs },
      { id: 'out:structure', direction: 'out' as const, portName: 'structure', portType: 'StructureArtifact' as const, required: true, wired: true, artifact: struct },
    ];
    expect(pickDefaultPeekSlot(genSlots)).toBe('out:structure');

    // selectPrimaryOutputPort matches pickDefaultPeekSlot priority
    expect(selectPrimaryOutputPort({ structure: struct, mutations: mut })).toMatchObject({
      portName: 'mutations',
      is3D: false,
    });
    expect(selectPrimaryOutputPort({ structure: struct, track, sequences: seqs })).toMatchObject({
      portName: 'track',
      is3D: false,
    });
    expect(selectPrimaryOutputPort({ sequences: seqs, structure: struct })).toMatchObject({
      portName: 'structure',
      is3D: true,
    });
  });

  it('QA #7 & #8: resolveFoldIpsae falls back to PAE matrix and formatCandidateSequence inserts ":" chain delimiters', () => {
    // Explicit ipsae takes precedence
    expect(resolveFoldIpsae({ ipsae: 0.72, ipsaeMin: 0.68 })).toEqual({ ipsae: 0.72, ipsaeMin: 0.68 });

    // Monomer returns null
    const monoPae = new Float32Array(40 * 40).fill(15);
    expect(resolveFoldIpsae({ pae: monoPae, chainLengths: [40] })).toBeNull();

    // Multimer 20+20 with high interface PAE falls back to 0.00
    const highPae = new Float32Array(40 * 40).fill(17.5);
    expect(resolveFoldIpsae({ pae: highPae, chainLengths: [20, 20] })).toEqual({
      ipsae: 0,
      ipsaeMin: 0,
    });

    // Multimer 40+40 with low interface PAE (including AF3-style padded 85x85 PAE) computes > 0.6
    const dim = 85;
    const lowInterPae = new Float32Array(dim * dim).fill(1.0);
    const res = resolveFoldIpsae({ pae: lowInterPae, chainLengths: [40, 40] });
    expect(res).not.toBeNull();
    expect(res!.ipsae).toBeGreaterThan(0.6);

    // QA #8: resolveActiveRowChainLengths prioritizes row.chainLengths even without alignedFoldPdb
    const rowLens = resolveActiveRowChainLengths({ chainLengths: [20, 20], alignedFoldPdb: '' }, null);
    expect(rowLens).toEqual([20, 20]);
    const seq40 = 'NLYIQWLKDGGPSSGRPPPSNLYIQWLKDGGPSSGRPPPS';
    expect(seq40).toHaveLength(40);
    expect(formatCandidateSequence(seq40, rowLens)).toBe(
      'NLYIQWLKDGGPSSGRPPPS:NLYIQWLKDGGPSSGRPPPS',
    );
  });
});
