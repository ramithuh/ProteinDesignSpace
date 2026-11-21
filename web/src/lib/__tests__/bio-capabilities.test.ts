import { describe, expect, it } from 'vitest';
import {
  splitMultiChainSequence,
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from '$lib/engine/bio-capabilities';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import { validateConnection } from '$lib/engine/graph-utils';
import { atom37ToPdb, chainIdsFromLengths, parsePdbBackbone } from '$lib/protein-ops/pdb';
import type { SequenceSet, StructureArtifact, WorkflowGraph } from '$lib/engine/types';

describe('Two-Layer Biological Capability Schema & Multi-Chain I/O', () => {
  it('Layer 1 warns when a multi-chain binder preset (ppi_pdl1) feeds AF2 monomer and offers 1-click AF2-Multimer v3 fix', () => {
    const graph: WorkflowGraph = {
      id: 'ppi-test',
      name: 'PD-L1 Binder',
      nodes: [
        { id: 'cond', adapterId: 'rfd4-conditioner', label: 'Condition', params: { taskPreset: 'ppi_pdl1' } },
        { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4', params: { mode: 'conditioned' } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'LocalFold (AF2 m1)', params: { model: 'monomer' } },
      ],
      edges: [
        { fromNode: 'cond', fromPort: 'condition', toNode: 'gen', toPort: 'condition' },
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };

    const issues = validateBioCapabilities(graph);
    const multimerIssue = issues.find((i) => i.code === 'multimer-into-af2-monomer');
    expect(multimerIssue).toBeDefined();
    expect(multimerIssue!.severity).toBe('warning');
    expect(multimerIssue!.fix).toEqual({
      label: 'Switch to AF2-Multimer v3',
      nodeId: 'fold',
      paramsPatch: { model: 'multimer' },
      nodeLabel: 'LocalFold (AF2-Multimer 1)',
    });

    // Applying the 1-click quick fix resolves the warning
    const fixedGraph: WorkflowGraph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === multimerIssue!.fix!.nodeId
          ? { ...n, params: { ...n.params, ...multimerIssue!.fix!.paramsPatch }, label: multimerIssue!.fix!.nodeLabel }
          : n,
      ),
    };
    expect(validateBioCapabilities(fixedGraph).find((i) => i.code === 'multimer-into-af2-monomer')).toBeUndefined();
  });

  it('Layer 1 warns on non-protein ligands (sm_sam) and DNA (dna_7m5w) into ProteinMPNN and AF2, offering RF3 switch', () => {
    const graph: WorkflowGraph = {
      id: 'ligand-test',
      name: 'SAM Binder',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4', params: { mode: 'preset', presetId: 'sm_sam' } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN', params: { family: 'solublempnn' } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'LocalFold', params: { model: 'monomer' } },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };

    const issues = validateBioCapabilities(graph);
    expect(issues.some((i) => i.code === 'mpnn-non-protein-stripped' && i.nodeId === 'mpnn')).toBe(true);
    const af2Issue = issues.find((i) => i.code === 'af2-non-protein-entity' && i.nodeId === 'fold');
    expect(af2Issue).toBeDefined();
    expect(af2Issue!.fix?.paramsPatch).toEqual({ model: 'rosettafold3' });
  });

  it('Layer 1 flags pure A/C/G/T DNA strings pasted into Sequence Input', () => {
    const graph: WorkflowGraph = {
      id: 'dna-seq-test',
      name: 'DNA Accident',
      nodes: [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          label: 'Sequence Input',
          params: { sequence: 'ATGCGATCGATCGATCGATCGATC' },
        },
      ],
      edges: [],
    };
    const issues = validateBioCapabilities(graph);
    expect(issues.some((i) => i.code === 'dna-in-protein-seq')).toBe(true);
  });

  it('preserves multi-chain colon syntax in splitMultiChainSequence and sequence-input CPU execution', async () => {
    const split = splitMultiChainSequence('MKTAYIAKQR:QISFVKSHFSRQ');
    expect(split.chains).toEqual(['MKTAYIAKQR', 'QISFVKSHFSRQ']);
    expect(split.chainLengths).toEqual([10, 12]);
    expect(split.concatenated).toBe('MKTAYIAKQRQISFVKSHFSRQ');

    const scheduler = new ExecutionScheduler();
    const graph: WorkflowGraph = {
      id: 'multichain-seqin',
      name: 'Multi-Chain FASTA',
      nodes: [
        {
          id: 'seq',
          adapterId: 'sequence-input',
          label: 'Sequence Input',
          params: { sequence: '>heterodimer\nMKTAYIAKQR:QISFVKSHFSRQ\n' },
        },
      ],
      edges: [],
    };

    const res = await scheduler.run(graph, () => {});
    const seqSet = res.outputsByNode.seq.sequences as SequenceSet;
    expect(seqSet.candidates).toHaveLength(1);
    expect(seqSet.candidates[0].sequence).toBe('MKTAYIAKQRQISFVKSHFSRQ');
    expect(seqSet.candidates[0].chainLengths).toEqual([10, 12]);
  });

  it('atom37ToPdb emits distinct chain IDs and TER records for multi-chain complexes, and parsePdbBackbone extracts diagnostics', () => {
    const L = 6;
    const atom37 = new Float32Array(L * 37 * 3);
    const atom37Mask = new Float32Array(L * 37);
    const plddt = new Float32Array(L).fill(91.5);
    for (let i = 0; i < L; i++) {
      // N (0), CA (1), C (2), O (4)
      for (const a of [0, 1, 2, 4]) {
        atom37Mask[i * 37 + a] = 1;
        atom37[(i * 37 + a) * 3] = i * 3.8 + a * 0.4;
        atom37[(i * 37 + a) * 3 + 1] = a * 0.2;
        atom37[(i * 37 + a) * 3 + 2] = 0;
      }
    }
    const chainIds = chainIdsFromLengths([3, 3]);
    expect(chainIds).toEqual(['A', 'A', 'A', 'B', 'B', 'B']);

    const { pdbText } = atom37ToPdb(atom37, atom37Mask, 'ACDEFG', plddt, chainIds);
    expect(pdbText).toContain('TER');

    const withHet =
      pdbText.replace('END\n', '') +
      'HETATM   99  C1  SAM C   1       1.000   2.000   3.000  1.00 20.00           C\n' +
      'ATOM    100  P    DA D   1       4.000   5.000   6.000  1.00 20.00           P\n' +
      'END\n';
    const parsed = parsePdbBackbone(withHet);
    expect(parsed.nResidues).toBe(6);
    expect(parsed.diagnostics.chainCount).toBe(2);
    expect(parsed.diagnostics.chainLengths).toEqual([3, 3]);
    expect(parsed.diagnostics.hasLigands).toBe(true);
    expect(parsed.diagnostics.ligandNames).toEqual(['SAM']);
    expect(parsed.diagnostics.hasNucleicAcids).toBe(true);
    expect(parsed.diagnostics.nucleicAcidChains).toEqual(['D']);
  });

  it('Layer 2 runtime guard rejects sequences over WebGPU token ceiling and warns on multimer into AF2 monomer', () => {
    const tooLong: SequenceSet = {
      kind: 'SequenceSet',
      id: 'long-seq',
      parentStructureId: 'none',
      sourceModelId: 'sequence-input',
      candidates: [
        { id: 'huge', sequence: 'A'.repeat(600), score: null, globalScore: null, sourceModelId: 'sequence-input' },
      ],
    };

    expect(() =>
      validateRuntimeNodeInputs(
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'boltz2' } },
        { sequences: tooLong },
      ),
    ).toThrow(/WebGPU memory ceiling/);

    const complexSeq: SequenceSet = {
      kind: 'SequenceSet',
      id: 'complex-seq',
      parentStructureId: 'none',
      sourceModelId: 'sequence-input',
      candidates: [
        {
          id: 'dimer',
          sequence: 'MKTAYIAKQRQISFVKSHFS',
          chainLengths: [10, 10],
          score: null,
          globalScore: null,
          sourceModelId: 'sequence-input',
        },
      ],
    };
    const res = validateRuntimeNodeInputs(
      { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer' } },
      { sequences: complexSeq },
    );
    expect(res.warnings).toHaveLength(1);
    expect(res.warnings[0]).toContain('AF2-Multimer v3');

    const ligStruct: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 's1',
      label: 'SAM complex',
      sourceModelId: 'pdb-input',
      length: 10,
      sequence: 'MKTAYIAKQR',
      caCoords: new Float32Array(30),
      backbone4Coords: new Float32Array(120),
      pdbText: 'ATOM ...',
      geometry: {
        length: 10,
        rg: 10,
        meanCaDist: 3.8,
        helixPct: 50,
        sheetPct: 0,
        coilPct: 50,
        chainBreaks: 0,
        clashes: 0,
      },
      diagnostics: {
        chainCount: 1,
        chainLengths: [10],
        hasLigands: true,
        ligandNames: ['SAM'],
        hasNucleicAcids: false,
        nucleicAcidChains: [],
      },
    };
    const mpnnGuard = validateRuntimeNodeInputs(
      { id: 'mpnn', adapterId: 'web-protein-mpnn', params: {} },
      { structure: ligStruct },
    );
    expect(mpnnGuard.warnings[0]).toContain('SAM');
  });

  it('allows wiring LocalFold (structure) directly into ProteinMPNN (structure) for iterative redesign loops', () => {
    const graph: WorkflowGraph = {
      id: 'iter-redesign',
      name: 'Iterative Redesign',
      nodes: [
        { id: 'fold1', adapterId: 'localfold-af2', params: { model: 'monomer' } },
        { id: 'mpnn2', adapterId: 'web-protein-mpnn', params: { family: 'solublempnn' } },
      ],
      edges: [],
    };
    const check = validateConnection(graph, 'fold1', 'structure', 'mpnn2', 'structure');
    expect(check.valid).toBe(true);
  });
});
