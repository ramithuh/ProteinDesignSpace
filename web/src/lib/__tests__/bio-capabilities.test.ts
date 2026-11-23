import { describe, expect, it, vi } from 'vitest';
import {
  extractUpstreamHoloHints,
  selectFoldRepresentative,
  selectFoldWorkItems,
  splitMultiChainSequence,
  summarizeCropSegments,
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from '$lib/engine/bio-capabilities';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import { validateConnection } from '$lib/engine/graph-utils';
import { atom37ToPdb, chainIdsFromLengths, parsePdbBackbone, parseStructureText } from '$lib/protein-ops/pdb';
import { clearRcsbCache } from '$lib/protein-ops/rcsb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import type {
  FoldedCandidate,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

describe('Two-Layer Biological Capability Schema & Multi-Chain I/O', () => {
  it('Layer 1 warns when a multi-chain binder preset (ppi_pdl1) feeds AF2 monomer and offers 1-click AF2-Multimer v3 fix', () => {
    const graph: WorkflowGraph = {
      id: 'ppi-test',
      name: 'PD-L1 Binder',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', label: 'RFD4', params: { mode: 'preset', presetId: 'ppi_pdl1' } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'LocalFold (AF2 m1)', params: { model: 'monomer' } },
      ],
      edges: [
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
      nodeLabel: 'AF2-Multimer 1',
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
    expect(af2Issue!.fix?.nodeLabel).toBe('RoseTTAFold 3');
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

  it('validates ESMFold2 protein residues and chain boundaries before loading weights', () => {
    const node = { id: 'fold', adapterId: 'localfold-af2', params: { model: 'ef2-fast-300m' } };
    const check = (sequence: string, chainLengths?: number[]) => validateRuntimeNodeInputs(node, {
      sequences: {
        kind: 'SequenceSet', id: 'seqs', parentStructureId: 'none', sourceModelId: 'sequence-input',
        candidates: [{ id: 'example', sequence, chainLengths, sourceModelId: 'sequence-input', score: null, globalScore: null }],
      },
    });
    expect(check('ACDE:FGHI', [4, 4]).warnings).toEqual([]);
    expect(() => check('ACDXFGHI')).toThrow(/canonical protein/);
    expect(() => check('ACDEFGHI', [4, 3])).toThrow(/chain boundaries/);
    expect(() => check('A'.repeat(513))).toThrow(/512/);
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

  it('uses live outputsByNode StructureArtifact.diagnostics, propagates strippedNonProtein through MPNN, and catches preset/Genie3/MPNN edge cases', () => {
    // 1. Live outputsByNode detects ligands & multimer even when pdb-input node.params.pdbText is empty (auto-preview mode)
    const graphAutoPreview: WorkflowGraph = {
      id: 'auto-preview-live',
      name: 'Auto Preview',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '9XYZ' } },
        { id: 'g3', adapterId: 'genie3', params: { taskMode: 'motif', length: 80 } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    };
    const liveHolo: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'live-holo',
      label: '9XYZ.cif',
      sourceModelId: 'pdb-input',
      length: 40,
      sequence: 'A'.repeat(40),
      caCoords: new Float32Array(120),
      backbone4Coords: new Float32Array(480),
      pdbText: 'ATOM ...',
      diagnostics: {
        chainCount: 1,
        chainLengths: [40],
        hasLigands: true,
        ligandNames: ['ZN', 'SAM'],
        hasNucleicAcids: false,
        nucleicAcidChains: [],
      },
      geometry: {
        length: 40,
        rg: 10,
        meanCaDist: 3.8,
        helixPct: 50,
        sheetPct: 0,
        coilPct: 50,
        chainBreaks: 0,
        clashes: 0,
      },
    };
    const liveIssues = validateBioCapabilities(graphAutoPreview, { pdb: { structure: liveHolo } });
    const g3EntityIssue = liveIssues.find(
      (i) => i.code === 'genie3-unsupported-entities' && i.nodeId === 'g3',
    );
    expect(g3EntityIssue).toBeDefined();
    expect(g3EntityIssue?.fix?.label).toBe('Crop protein chain');

    // Unified Primitive 2 policy: >=3 protein residues + uncropped ligands produce a non-blocking
    // Layer-2 warning (NO runtime throw), whereas <3 protein residues or incompatible all-atom benchmarks hard-block.
    const g3Rt = validateRuntimeNodeInputs(
      { id: 'g3', adapterId: 'genie3', params: { taskMode: 'motif', motifSpan: 'A1-10' } },
      { structure: liveHolo },
    );
    expect(g3Rt.warnings.some((w) => /protein-only/i.test(w))).toBe(true);
    expect(() =>
      validateRuntimeNodeInputs(
        { id: 'g3', adapterId: 'genie3', params: { taskMode: 'motif', motifSpan: 'A1-10' } },
        { structure: { ...liveHolo, length: 2, sequence: 'AA' } },
      ),
    ).toThrow(/at least 3 protein/i);
    expect(() =>
      validateRuntimeNodeInputs(
        { id: 'g3', adapterId: 'genie3', params: { taskMode: 'motif', motifSpan: 'A1-10' } },
        { structure: { ...liveHolo, benchmarkId: 'ame_1ctt_cda' } },
      ),
    ).toThrow(/all-atom/i);

    // 2. strippedNonProtein propagates through MPNN so RFD4 (SAM) -> MPNN -> LocalFold still warns about apo refold!
    const mpnnApoGraph: WorkflowGraph = {
      id: 'sam-mpnn-af3',
      name: 'SAM -> MPNN -> RF3',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', params: { mode: 'preset', presetId: 'sm_sam' } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'solublempnn' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'rosettafold3' } },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const mpnnApoIssues = validateBioCapabilities(mpnnApoGraph);
    expect(mpnnApoIssues.some((i) => i.code === 'mpnn-non-protein-stripped' && i.nodeId === 'mpnn')).toBe(true);
    expect(mpnnApoIssues.some((i) => i.code === 'af3-apo-refold-note' && i.nodeId === 'fold')).toBe(true);

    // 3. Warn when a wired structure is ignored by RFD4 preset mode
    const rfd4PresetTplGraph: WorkflowGraph = {
      id: 'preset-ignores-tpl',
      name: 'Preset Ignores Template',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        { id: 'gen', adapterId: 'rfd4-proteina', params: { mode: 'preset', presetId: 'sm_sam' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'gen', toPort: 'structure' }],
    };
    const presetIssues = validateBioCapabilities(rfd4PresetTplGraph);
    expect(presetIssues.some((i) => i.code === 'rfd4-preset-ignores-template' && i.nodeId === 'gen')).toBe(true);

    // 4. Layer 2 rejects pure-ligand/DNA structure (<3 protein residues) into ProteinMPNN
    const pureLigStruct: StructureArtifact = { ...liveHolo, length: 0, sequence: '' };
    expect(() =>
      validateRuntimeNodeInputs(
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: {} },
        { structure: pureLigStruct },
      ),
    ).toThrow(/at least 3 protein residues/i);
  });

  it('Issue #4: selectFoldWorkItems and preflight guard resolve identical candidate slice and expose excluded candidates', () => {
    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'mixed-len-set',
      parentStructureId: 'none',
      sourceModelId: 'sequence-input',
      collectionKind: 'independent-candidates',
      candidates: [
        {
          id: 'cand-20aa',
          key: 'cand-20aa',
          sequence: 'MKTAYIAKQRQISFVKSHFS',
          score: null,
          globalScore: null,
          sourceModelId: 'sequence-input',
        },
        {
          id: 'cand-769aa',
          key: 'cand-769aa',
          sequence: 'A'.repeat(769),
          score: null,
          globalScore: null,
          sourceModelId: 'sequence-input',
        },
      ],
    };

    // With maxSequences=1, shared helper selects only candidate #1 and exposes candidate #2 as excluded
    const work = selectFoldWorkItems(seqSet, { maxSequences: 1 });
    expect(work.maxSequences).toBe(1);
    expect(work.selected.map((c) => c.id)).toEqual(['cand-20aa']);
    expect(work.excludedCandidates).toHaveLength(1);
    expect(work.excludedCandidates[0]).toMatchObject({
      key: 'cand-769aa',
      id: 'cand-769aa',
      index: 1,
    });

    // Preflight passes cleanly (does NOT reject excluded 769-aa candidate #2!) and exposes excludedCandidates
    const preflight = validateRuntimeNodeInputs(
      { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer', maxSequences: 1 } },
      { sequences: seqSet },
    );
    expect(preflight.selectedCandidateIds).toEqual(['cand-20aa']);
    expect(preflight.excludedCandidates).toEqual(work.excludedCandidates);
    expect(preflight.warnings.some((w) => w.includes('cand-769aa'))).toBe(true);

    // Increasing maxSequences to 2 includes the 769-aa candidate -> preflight rejects (>768 aa ceiling)
    expect(() =>
      validateRuntimeNodeInputs(
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer', maxSequences: 2 } },
        { sequences: seqSet },
      ),
    ).toThrow(/WebGPU memory ceiling/);
  });

  it('Issue #5: selectFoldRepresentative records selection criterion and supports confidence-less ensembles & ordered frames', () => {
    const makeFold = (
      id: string,
      meanPlddt: number,
      extra?: Partial<FoldedCandidate>,
    ): FoldedCandidate => ({
      sequenceId: id,
      key: id,
      sequence: 'ACDEFGHIKL',
      foldModelId: 'mock-folder',
      pdbText: 'END\n',
      caCoords: new Float32Array(30),
      plddt: new Float32Array(10).fill(Number.isFinite(meanPlddt) ? meanPlddt : 0),
      meanPlddt,
      ptm: null,
      elapsedMs: 10,
      ...extra,
    });

    // 1. Confidence-scored folds select highest mean pLDDT with explicit provenance
    const scoredRep = selectFoldRepresentative([
      makeFold('cand-a', 76.2, { confidence: 76.2, confidenceAvailable: true }),
      makeFold('cand-b', 91.4, { confidence: 91.4, confidenceAvailable: true }),
      makeFold('cand-c', 82.0, { confidence: 82.0, confidenceAvailable: true }),
    ]);
    expect(scoredRep).not.toBeNull();
    expect(scoredRep!.fold.sequenceId).toBe('cand-b');
    expect(scoredRep!.selection).toEqual({
      selectedKey: 'cand-b',
      criterion: 'highest-mean-plddt',
      totalMembers: 3,
      selectedIndex: 1,
      scoreValue: 91.4,
    });

    // 2. Confidence-less multi-sample ensemble falls back deterministically without fabricating a score
    const ensembleRep = selectFoldRepresentative(
      [
        makeFold('sample-0', Number.NaN, { confidence: null, confidenceAvailable: false }),
        makeFold('sample-1', Number.NaN, { confidence: null, confidenceAvailable: false }),
      ],
      { collectionKind: 'multi-sample' },
    );
    expect(ensembleRep).not.toBeNull();
    expect(ensembleRep!.fold.sequenceId).toBe('sample-0');
    expect(ensembleRep!.selection.selectedKey).toBe('sample-0');
    expect(ensembleRep!.selection.criterion).toContain('first-member-fallback');
    expect(ensembleRep!.selection.totalMembers).toBe(2);
    expect(ensembleRep!.selection.scoreValue).toBeNull();

    // 3. Confidence-less ordered frames falls back to final frame deterministically
    const trajRep = selectFoldRepresentative(
      [
        makeFold('frame-0', Number.NaN, { confidence: null, confidenceAvailable: false }),
        makeFold('frame-1', Number.NaN, { confidence: null, confidenceAvailable: false }),
        makeFold('frame-2', Number.NaN, { confidence: null, confidenceAvailable: false }),
      ],
      { collectionKind: 'ordered-frames' },
    );
    expect(trajRep!.fold.sequenceId).toBe('frame-2');
    expect(trajRep!.selection.criterion).toContain('final-frame-fallback');
    expect(trajRep!.selection.scoreValue).toBeNull();
  });

  it('Wave-2 P1/P2 & #123 regressions: missing structure edge blocking, duplicate model label fix, Genie 3 / Cα-only -> SolubleMPNN 1-click switch, and clean genie3-mpnn-af2 preset', () => {
    // 1. Missing required structure edge blocks RFD4 (motif/binder/partial/inverse-folding) and Genie 3 (motif/binder)
    for (const mode of ['motif', 'binder', 'partial', 'inverse-folding']) {
      const g: WorkflowGraph = {
        id: `rfd4-missing-${mode}`,
        name: 'Missing RFD4 Tpl',
        nodes: [{ id: 'rfd', adapterId: 'rfd4-proteina', params: { mode, length: 70 } }],
        edges: [],
      };
      const issues = validateBioCapabilities(g);
      const err = issues.find((i) => i.code === 'rfd4-missing-template');
      expect(err, mode).toBeDefined();
      expect(err!.severity).toBe('error');
    }

    for (const taskMode of ['motif', 'binder']) {
      const g: WorkflowGraph = {
        id: `g3-missing-${taskMode}`,
        name: 'Missing Genie3 Tpl',
        nodes: [
          {
            id: 'g3',
            adapterId: 'genie3',
            params: { taskMode, length: 80, ...(taskMode === 'motif' ? { motifSpan: 'A16-35' } : { lockChain: 'A' }) },
          },
        ],
        edges: [],
      };
      const issues = validateBioCapabilities(g);
      const err = issues.find((i) => i.code === 'genie3-missing-template');
      expect(err, taskMode).toBeDefined();
      expect(err!.severity).toBe('error');
    }

    // 2. Duplicate model label fix: "AF2 Model 1" does not repeat as "AF2 Model 1 (AF2 Model 1)"
    const dupLabelGraph: WorkflowGraph = {
      id: 'dup-label',
      name: 'Dup Label',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQR:QISFVKSHFS' } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { model: 'monomer' } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const dupIssues = validateBioCapabilities(dupLabelGraph);
    const multimerWarn = dupIssues.find((i) => i.code === 'multimer-into-af2-monomer')!;
    expect(multimerWarn.message).not.toContain('(AF2 Model 1)');
    expect(multimerWarn.message).toContain('AF2 Model 1');

    // 3. Genie 3 -> SolubleMPNN triggers Cα-only warning with 1-click Switch to Cα-ProteinMPNN fix
    const g3SolubleGraph: WorkflowGraph = {
      id: 'g3-soluble',
      name: 'Genie3 Soluble',
      nodes: [
        { id: 'g3', adapterId: 'genie3', params: { taskMode: 'unconditional', length: 80 } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'SolubleMPNN', params: { family: 'solublempnn' } },
      ],
      edges: [{ fromNode: 'g3', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' }],
    };
    const g3SolIssues = validateBioCapabilities(g3SolubleGraph);
    const caWarn = g3SolIssues.find((i) => i.code === 'mpnn-ca-only-backbone');
    expect(caWarn).toBeDefined();
    expect(caWarn!.severity).toBe('error');
    expect(caWarn!.fix).toEqual({
      label: 'Switch to Cα-ProteinMPNN',
      nodeId: 'mpnn',
      paramsPatch: { family: 'ca', modelName: 'v_48_020' },
      nodeLabel: 'Cα-ProteinMPNN',
    });

    // Applying the fix clears the error
    g3SolubleGraph.nodes[1].params.family = 'ca';
    expect(validateBioCapabilities(g3SolubleGraph).find((i) => i.code === 'mpnn-ca-only-backbone')).toBeUndefined();

    // 4. Cα-only custom PDB -> SolubleMPNN also triggers blocking Switch to Cα-ProteinMPNN error
    const caOnlyPdb = [
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  ',
      'ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C  ',
      'ATOM      3  CA  VAL A   3       7.600   0.000   0.000  1.00  0.00           C  ',
      'END',
    ].join('\n');
    const caPdbGraph: WorkflowGraph = {
      id: 'ca-pdb-mpnn',
      name: 'CA PDB MPNN',
      nodes: [
        { id: 'src', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: caOnlyPdb } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'SolubleMPNN', params: { family: 'solublempnn' } },
      ],
      edges: [{ fromNode: 'src', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' }],
    };
    const caPdbIssues = validateBioCapabilities(caPdbGraph);
    const caPdbErr = caPdbIssues.find((i) => i.code === 'mpnn-ca-only-backbone');
    expect(caPdbErr?.severity).toBe('error');
    expect(caPdbErr?.fix?.label).toBe('Switch to Cα-ProteinMPNN');
  });

  it('Wave-3 MSA preflight recommendations: Target-Only MSA for locked-target binders, MMseqs2 for natural proteins, and warning-free single-seq for de novo monomers', () => {
    // 1. Multi-chain binder with locked target chain under msaMode='single' -> Enable Target-Only MSA
    const binderGraph: WorkflowGraph = {
      id: 'msa-binder',
      name: 'Binder MSA',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '5O45', benchmarkId: 'pdl1_5o45' } },
        { id: 'gen', adapterId: 'rfd4-proteina', params: { mode: 'binder', lockChain: 'A', length: 70 } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AF2-Multimer 1', params: { model: 'multimer', msaMode: 'single' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'gen', toPort: 'structure' },
        { fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    const binderIssues = validateBioCapabilities(binderGraph);
    const targetMsaWarn = binderIssues.find((i) => i.code === 'binder-single-seq-msa');
    expect(targetMsaWarn).toBeDefined();
    expect(targetMsaWarn!.severity).toBe('warning');
    expect(targetMsaWarn!.fix).toEqual({
      label: 'Enable Target-Only MSA',
      nodeId: 'fold',
      paramsPatch: { msaMode: 'target-only' },
    });
    // Applying fix clears the warning
    binderGraph.nodes[2].params.msaMode = 'target-only';
    expect(validateBioCapabilities(binderGraph).find((i) => i.code === 'binder-single-seq-msa')).toBeUndefined();

    // 2. Natural protein from sequence-input under single-seq -> Enable MMseqs2 MSA
    const seqGraph: WorkflowGraph = {
      id: 'msa-natural-seq',
      name: 'Natural Seq MSA',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQD' } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AF2 Model 1', params: { model: 'monomer' } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const seqIssues = validateBioCapabilities(seqGraph);
    const natSeqWarn = seqIssues.find((i) => i.code === 'natural-single-seq-msa');
    expect(natSeqWarn).toBeDefined();
    expect(natSeqWarn!.severity).toBe('warning');
    expect(natSeqWarn!.fix).toEqual({
      label: 'Enable MMseqs2 MSA',
      nodeId: 'fold',
      paramsPatch: { msaMode: 'mmseqs2' },
    });
    seqGraph.nodes[1].params.msaMode = 'mmseqs2';
    expect(validateBioCapabilities(seqGraph).find((i) => i.code === 'natural-single-seq-msa')).toBeUndefined();

    // 3. Inverse-folded sequences from pdb-input -> SolubleMPNN -> LocalFold are designed sequences,
    //    NOT natural sequences, so single-sequence folding should NOT emit a natural-single-seq-msa warning.
    const pdbNaturalGraph: WorkflowGraph = {
      id: 'msa-natural-pdb',
      name: 'Redesigned PDB Backbone',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1PRW' } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'solublempnn' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer', msaMode: 'single' } },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    expect(validateBioCapabilities(pdbNaturalGraph).find((i) => i.code === 'natural-single-seq-msa')).toBeUndefined();

    // 4. De novo single-chain monomers (RFD4 -> AF2 and Genie 3 -> Cα-MPNN -> AF2) are warning-free under 'single'
    const rfd4MonomerGraph: WorkflowGraph = {
      id: 'denovo-rfd4-mono',
      name: 'De Novo RFD4 Monomer',
      nodes: [
        { id: 'gen', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 70 } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer', msaMode: 'single' } },
      ],
      edges: [{ fromNode: 'gen', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    expect(validateBioCapabilities(rfd4MonomerGraph)).toEqual([]);

    const genie3MonomerGraph: WorkflowGraph = {
      id: 'denovo-g3-mono',
      name: 'De Novo Genie3 Monomer',
      nodes: [
        { id: 'gen', adapterId: 'genie3', params: { taskMode: 'unconditional', length: 80 } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'ca', modelName: 'v_48_020' } },
        { id: 'fold', adapterId: 'localfold-af2', params: { model: 'monomer', msaMode: 'single' } },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      ],
    };
    expect(validateBioCapabilities(genie3MonomerGraph)).toEqual([]);
  });

  it('R4: warns when AntiFold is wired to non-antibody structures and offers 1-click Switch to ESM-IF1 fix without false positives on antibodies', () => {
    // 1. Demo bundle / RFD4 / 1UBQ -> AntiFold triggers antifold-non-antibody warning
    const demoAntiGraph: WorkflowGraph = {
      id: 'demo-anti',
      name: 'Demo AntiFold',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle', filename: 'demo_3helix_53aa.pdb' } },
        {
          id: 'af',
          adapterId: 'antifold-esmif1',
          label: 'AntiFold',
          params: { variant: 'antifold', regions: 'CDR1 CDR2 CDR3' },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    const demoIssues = validateBioCapabilities(demoAntiGraph);
    const antiWarn = demoIssues.find((i) => i.code === 'antifold-non-antibody');
    expect(antiWarn).toBeDefined();
    expect(antiWarn!.severity).toBe('warning');
    expect(antiWarn!.message).toMatch(/ESM-IF1|SolubleMPNN/i);
    expect(antiWarn!.fix).toEqual({
      label: 'Switch to ESM-IF1',
      nodeId: 'af',
      paramsPatch: { variant: 'esmif1' },
      nodeLabel: 'ESM-IF1',
    });

    // Applying the +page.svelte fix merge resolves the warning
    const fixedGraph: WorkflowGraph = {
      ...demoAntiGraph,
      nodes: demoAntiGraph.nodes.map((n) =>
        n.id === antiWarn!.fix!.nodeId
          ? {
              ...n,
              params: { ...n.params, ...antiWarn!.fix!.paramsPatch },
              ...(antiWarn!.fix!.nodeLabel ? { label: antiWarn!.fix!.nodeLabel } : {}),
            }
          : n,
      ),
    };
    expect(validateBioCapabilities(fixedGraph).find((i) => i.code === 'antifold-non-antibody')).toBeUndefined();

    // 2. RFD4 de novo backbone -> AntiFold triggers warning
    const rfdAntiGraph: WorkflowGraph = {
      id: 'rfd-anti',
      name: 'RFD4 AntiFold',
      nodes: [
        { id: 'rfd', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 80 } },
        { id: 'af', adapterId: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold' } },
      ],
      edges: [{ fromNode: 'rfd', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    expect(validateBioCapabilities(rfdAntiGraph).some((i) => i.code === 'antifold-non-antibody')).toBe(true);

    // 3. Negative control: antibody H/L chain / molecularContext antibody-H does NOT emit warning
    const abGraph: WorkflowGraph = {
      id: 'ab-anti',
      name: 'Antibody AntiFold',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1IGT' } },
        { id: 'af', adapterId: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold', targetChain: 'H', regions: 'CDR1 CDR3' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    expect(validateBioCapabilities(abGraph).find((i) => i.code === 'antifold-non-antibody')).toBeUndefined();

    const abStructOut = {
      pdb: {
        structure: {
          kind: 'StructureArtifact',
          id: 'ab-1igt',
          label: '1IGT',
          sourceModelId: 'pdb-input',
          length: 115,
          sequence: 'A'.repeat(115),
          caCoords: new Float32Array(115 * 3),
          backbone4Coords: new Float32Array(115 * 4 * 3),
          pdbText: '',
          geometry: computeGeometryStats(new Float32Array(115 * 3)),
          molecularContext: { chainTypes: { A: 'antibody-H' } },
        } satisfies StructureArtifact,
      },
    };
    const abCtxGraph: WorkflowGraph = {
      id: 'ab-ctx',
      name: 'Antibody Context AntiFold',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1IGT' } },
        { id: 'af', adapterId: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold', regions: 'CDR3' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    expect(validateBioCapabilities(abCtxGraph, abStructOut).find((i) => i.code === 'antifold-non-antibody')).toBeUndefined();
  });

  it('Iteration 2 Bug 5: does NOT suppress antifold-non-antibody warning via resNum >= 105 on >=105aa RFD4 or non-antibody PDBs (1CTT, 5O45)', () => {
    const makeSequentialPdb = (nRes: number, chain = 'A', startNum = 1) => {
      const lines: string[] = [];
      let serial = 1;
      for (let i = 0; i < nRes; i++) {
        const rn = String(startNum + i).padStart(4);
        const x = (i * 3.8).toFixed(3).padStart(8);
        lines.push(`ATOM  ${String(serial++).padStart(5)}  N   ALA ${chain}${rn}    ${x}   0.000   0.000  1.00 20.00           N  `);
        lines.push(`ATOM  ${String(serial++).padStart(5)}  CA  ALA ${chain}${rn}    ${x}   1.000   0.000  1.00 20.00           C  `);
        lines.push(`ATOM  ${String(serial++).padStart(5)}  C   ALA ${chain}${rn}    ${x}   2.000   0.000  1.00 20.00           C  `);
        lines.push(`ATOM  ${String(serial++).padStart(5)}  O   ALA ${chain}${rn}    ${x}   2.000   1.000  1.00 20.00           O  `);
      }
      lines.push('END');
      return lines.join('\n');
    };

    // Case A: RFD4 120aa with materialized liveStruct (sourceResidues 1..120 on chain A)
    const rfd4LiveStruct: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'rfd4-120',
      label: 'RFD4 120aa',
      sourceModelId: 'rfd4-proteina',
      length: 120,
      sequence: 'A'.repeat(120),
      caCoords: new Float32Array(120 * 3),
      backbone4Coords: new Float32Array(120 * 4 * 3),
      pdbText: makeSequentialPdb(120),
      chainIds: Array(120).fill('A'),
      sourceResidues: Array.from({ length: 120 }, (_, i) => ({
        index: i,
        authChainId: 'A',
        resNum: i + 1,
        insCode: '',
        resName: 'ALA',
        projectedChainId: 'A',
      })),
      geometry: computeGeometryStats(new Float32Array(120 * 3)),
    };
    const rfd4Graph: WorkflowGraph = {
      id: 'rfd4-120-antifold',
      name: 'RFD4 120aa AntiFold',
      nodes: [
        { id: 'rfd4_1', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 120 } },
        { id: 'af_1', adapterId: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold' } },
      ],
      edges: [{ fromNode: 'rfd4_1', fromPort: 'structure', toNode: 'af_1', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(rfd4Graph, { rfd4_1: { structure: rfd4LiveStruct } }).some(
        (i) => i.code === 'antifold-non-antibody',
      ),
    ).toBe(true);

    // Case B: 1CTT (294aa enzyme) with pdbText loaded
    const cttGraph: WorkflowGraph = {
      id: '1ctt-antifold',
      name: '1CTT AntiFold',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '1CTT', pdbText: makeSequentialPdb(294) },
        },
        { id: 'af_1', adapterId: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold' } },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'af_1', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(cttGraph, {}).some((i) => i.code === 'antifold-non-antibody'),
    ).toBe(true);

    // Case C: 5O45 benchmark target with pdbText >= 105aa
    const pdl1Graph: WorkflowGraph = {
      id: '5o45-antifold',
      name: '5O45 AntiFold',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '5O45', pdbText: makeSequentialPdb(116, 'A', 17) },
        },
        { id: 'af_1', adapterId: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold' } },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'af_1', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(pdl1Graph, {}).some((i) => i.code === 'antifold-non-antibody'),
    ).toBe(true);

    // Case D: Custom antibody PDB on chain A identified via PDB header keyword does NOT warn
    const customNanobodyGraph: WorkflowGraph = {
      id: 'nb-header-antifold',
      name: 'Nanobody Header AntiFold',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: {
            mode: 'custom',
            pdbText: `HEADER    NANOBODY VHH\n${makeSequentialPdb(120, 'A', 1)}`,
          },
        },
        { id: 'af_1', adapterId: 'antifold-esmif1', label: 'AntiFold', params: { variant: 'antifold' } },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'af_1', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(customNanobodyGraph, {}).some((i) => i.code === 'antifold-non-antibody'),
    ).toBe(false);
  });

  it('Issue #89: summarizeCropSegments & extractUpstreamHoloHints parse ligands, multi-char chains, and ignore insertion-code/water/MSE tokens', () => {
    const holoCrop = summarizeCropSegments('A, A:ZN, A:DHZ');
    expect(holoCrop.valid).toBe(true);
    expect([...holoCrop.proteinChains]).toEqual(['A']);
    expect(holoCrop.ligandNames).toEqual(['ZN', 'DHZ']);

    const multiChar = summarizeCropSegments('AA, AB, AA:sam, A:10A, A:HOH, A:MSE');
    expect(multiChar.valid).toBe(true);
    expect([...multiChar.proteinChains]).toEqual(['AA', 'AB', 'A']);
    expect(multiChar.ligandNames).toEqual(['SAM']);

    const invalid = summarizeCropSegments('A, ???');
    expect(invalid.valid).toBe(false);

    const hintGraph: WorkflowGraph = {
      id: 'holo-hints-multichar',
      name: 'Holo Hints Multi-Char',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '7C7M', cropSegments: 'AA, aa:sam, A:10A, A:HOH, A:MSE' },
        },
        {
          id: 'fold_1',
          adapterId: 'localfold-af2',
          params: { model: 'af3' },
        },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' }],
    };
    const hints = extractUpstreamHoloHints(hintGraph, 'fold_1');
    expect(hints.ligandCcdCodes).toEqual(['SAM']);
  });

  it('Issue #89: apo AF3 downstream of holo RFD4 does NOT leak ligand entities, while holo AF3 downstream of unfetched cropSegments passes GNINA', () => {
    // 1. Apo AF3 downstream of holo RFD4 (ame_1ctt_cda): no false mpnn-non-protein-stripped, catches gnina-no-organic-ligand, keeps af3-apo-refold-note
    const apoBranchGraph: WorkflowGraph = {
      id: 'apo-af3-rfd4-branch',
      name: 'Apo AF3 Downstream of Holo RFD4',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          label: 'AME (1CTT)',
          params: {
            mode: 'pdb-id',
            pdbId: '1CTT',
            cropSegments: 'A102, A104, A129, A132, A:ZN, A:DHZ',
            benchmarkId: 'ame_1ctt_cda',
          },
        },
        {
          id: 'rfd4_1',
          adapterId: 'rfd4-proteina',
          label: 'RFD4 (AME)',
          params: {
            mode: 'motif',
            length: 180,
            motifSpan: 'A102, A104, A129, A132',
            unindexedMotif: true,
            contig: '180-180',
            autoSeededBenchmarkId: 'ame_1ctt_cda',
          },
        },
        {
          id: 'af3_apo',
          adapterId: 'localfold-af2',
          label: 'AlphaFold 3 #2',
          params: { model: 'af3', ligandCcdCodes: '' },
        },
        {
          id: 'mpnn_1',
          adapterId: 'web-protein-mpnn',
          label: 'ProteinMPNN',
          params: { family: 'solublempnn' },
        },
        {
          id: 'gnina_apo',
          adapterId: 'gnina',
          label: 'GNINA 1.3',
          params: { taskMode: 'score' },
        },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' },
        { fromNode: 'rfd4_1', fromPort: 'sequences', toNode: 'af3_apo', toPort: 'sequences' },
        { fromNode: 'af3_apo', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' },
        { fromNode: 'af3_apo', fromPort: 'structure', toNode: 'gnina_apo', toPort: 'structure' },
      ],
    };

    const apoIssues = validateBioCapabilities(apoBranchGraph);
    expect(apoIssues.some((i) => i.nodeId === 'mpnn_1' && i.code === 'mpnn-non-protein-stripped')).toBe(false);
    expect(apoIssues.some((i) => i.nodeId === 'gnina_apo' && i.code === 'gnina-no-organic-ligand' && i.severity === 'error')).toBe(true);
    expect(apoIssues.some((i) => i.nodeId === 'af3_apo' && i.code === 'af3-apo-refold-note' && i.severity === 'warning')).toBe(true);

    // 2. Unfetched pdb-input with cropSegments: "A, A:ZN, A:DHZ" (no benchmarkId) -> holo AF3 -> GNINA passes cleanly
    const pdbCropAf3Graph: WorkflowGraph = {
      id: 'pdb-crop-af3-gnina',
      name: 'Unfetched Crop -> AF3 -> GNINA',
      nodes: [
        {
          id: 'pdb_input_1',
          adapterId: 'pdb-input',
          label: 'PDB Input #2',
          params: { mode: 'pdb-id', pdbId: '1CTT', cropSegments: 'A, A:ZN, A:DHZ', pdbText: '' },
        },
        {
          id: 'localfold_af2_2',
          adapterId: 'localfold-af2',
          label: 'AlphaFold 3 #3',
          params: { model: 'af3', ligandCcdCodes: 'ZN, DHZ', msaMode: 'single' },
        },
        {
          id: 'gnina_4',
          adapterId: 'gnina',
          label: 'GNINA 1.3 #4',
          params: { taskMode: 'score' },
        },
      ],
      edges: [
        { fromNode: 'pdb_input_1', fromPort: 'sequences', toNode: 'localfold_af2_2', toPort: 'sequences' },
        { fromNode: 'localfold_af2_2', fromPort: 'structure', toNode: 'gnina_4', toPort: 'structure' },
      ],
    };
    const holoIssues = validateBioCapabilities(pdbCropAf3Graph);
    expect(holoIssues.filter((i) => i.severity === 'error')).toEqual([]);
    expect(holoIssues.some((i) => i.nodeId === 'localfold_af2_2' && i.code === 'natural-single-seq-msa')).toBe(true);
    expect(holoIssues.some((i) => i.nodeId === 'localfold_af2_2' && i.code === 'af3-apo-refold-note')).toBe(false);

    // 3. Customized benchmark cropSegments removing organic ligand (keeping only ZN) -> GNINA catches metal-only on both pdb-input and rfd4-proteina
    const metalRfd4Graph: WorkflowGraph = {
      id: 'metal-rfd4-gnina',
      name: 'Metal RFD4 Template -> GNINA',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          label: 'AME (Metal Only)',
          params: {
            mode: 'pdb-id',
            pdbId: '1CTT',
            cropSegments: 'A102, A104, A129, A132, A:ZN',
            benchmarkId: 'ame_1ctt_cda',
          },
        },
        {
          id: 'rfd4_1',
          adapterId: 'rfd4-proteina',
          label: 'RFD4',
          params: { mode: 'motif', length: 180, motifSpan: 'A102, A104, A129, A132', unindexedMotif: true, contig: '180-180' },
        },
        {
          id: 'gnina_1',
          adapterId: 'gnina',
          label: 'GNINA 1.3',
          params: { taskMode: 'score' },
        },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' },
        { fromNode: 'rfd4_1', fromPort: 'structure', toNode: 'gnina_1', toPort: 'structure' },
      ],
    };
    const metalRfd4Err = validateBioCapabilities(metalRfd4Graph).find(
      (i) => i.nodeId === 'gnina_1' && i.code === 'gnina-no-organic-ligand',
    );
    expect(metalRfd4Err).toBeDefined();
    expect(metalRfd4Err!.message).toContain('ZN');
  });

  it('Issue #89: localfold-af2 honors live outputsByNode[id].structure diagnostics and scheduler.run validates resumeCheckpoint outputs', async () => {
    const graph: WorkflowGraph = {
      id: 'af3-live-diagnostics',
      name: 'AF3 Live Diagnostics',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFSRQ' } },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AlphaFold 3', params: { model: 'af3', ligandCcdCodes: '' } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN', params: { family: 'solublempnn' } },
        { id: 'gnina', adapterId: 'gnina', label: 'GNINA 1.3', params: { taskMode: 'score' } },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'fold', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'structure', toNode: 'gnina', toPort: 'structure' },
      ],
    };

    // Live StructureArtifact with DHZ overrides apo static params in inferNodeOutputProfile (triggering mpnn-non-protein-stripped and clearing gnina error)
    const liveFoldHolo: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'af3-live-holo',
      label: 'AF3 Holo',
      sourceModelId: 'localfold-af2',
      length: 22,
      sequence: 'MKTAYIAKQRQISFVKSHFSRQ',
      caCoords: new Float32Array(66),
      backbone4Coords: new Float32Array(264),
      pdbText: 'ATOM ...',
      geometry: computeGeometryStats(new Float32Array(66)),
      diagnostics: {
        chainCount: 1,
        chainLengths: [22],
        hasLigands: true,
        ligandNames: ['ZN', 'DHZ'],
        hasNucleicAcids: false,
        nucleicAcidChains: [],
      },
    };
    const liveIssues = validateBioCapabilities(graph, { fold: { structure: liveFoldHolo } });
    expect(liveIssues.some((i) => i.nodeId === 'mpnn' && i.code === 'mpnn-non-protein-stripped')).toBe(true);
    expect(liveIssues.some((i) => i.nodeId === 'gnina' && i.severity === 'error')).toBe(false);

    // ExecutionScheduler.run() resumeCheckpoint passes checkpointed outputs to pre-run bio validation
    const scheduler = new ExecutionScheduler();
    const pdbOnlyGraph: WorkflowGraph = {
      id: 'ckpt-pdb-demo',
      name: 'Checkpoint PDB Demo',
      nodes: [
        { id: 'pdb_1', adapterId: 'pdb-input', label: 'Demo PDB', params: { mode: 'demo-bundle' } },
      ],
      edges: [],
    };
    await scheduler.run(pdbOnlyGraph, () => {});
    const ckpt = scheduler.exportCheckpointObject()!;
    expect(ckpt).not.toBeNull();

    const resumedApoGraph: WorkflowGraph = {
      id: 'ckpt-pdb-demo-gnina',
      name: 'Checkpoint PDB Demo -> GNINA',
      nodes: [
        pdbOnlyGraph.nodes[0],
        { id: 'gnina_1', adapterId: 'gnina', label: 'GNINA 1.3', params: { taskMode: 'score' } },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'gnina_1', toPort: 'structure' }],
    };
    await expect(
      scheduler.run(resumedApoGraph, () => {}, {
        resumeCheckpoint: ckpt,
        signal: AbortSignal.abort(),
        returnPartialOnCancel: true,
      }),
    ).rejects.toThrow(/no organic small-molecule ligand/);
  });

  it('Iteration 2: honors custom cropSegments on benchmark templates, RNA in MPNN, punctuation-only AF3 params, and strengthened resumeCheckpoint', async () => {
    const MINI_HOLO_PDB = [
      'ATOM      1  N   HIS A  10      10.000  10.000  10.000  1.00 20.00           N  ',
      'ATOM      2  CA  HIS A  10      11.458  10.000  10.000  1.00 20.00           C  ',
      'ATOM      3  C   HIS A  10      12.000  11.420  10.000  1.00 20.00           C  ',
      'ATOM      4  O   HIS A  10      11.250  12.400  10.000  1.00 20.00           O  ',
      'ATOM      5  N   ASP A  11      13.320  11.520  10.000  1.00 20.00           N  ',
      'ATOM      6  CA  ASP A  11      13.980  12.820  10.000  1.00 20.00           C  ',
      'ATOM      7  C   ASP A  11      15.490  12.680  10.000  1.00 20.00           C  ',
      'ATOM      8  O   ASP A  11      16.050  11.580  10.000  1.00 20.00           O  ',
      'ATOM      9  N   ALA A  12      16.140  13.840  10.000  1.00 20.00           N  ',
      'ATOM     10  CA  ALA A  12      17.590  13.880  10.000  1.00 20.00           C  ',
      'ATOM     11  C   ALA A  12      18.120  15.300  10.000  1.00 20.00           C  ',
      'ATOM     12  O   ALA A  12      17.380  16.280  10.000  1.00 20.00           O  ',
      'HETATM   13 ZN    ZN A 200      12.500  11.000  12.500  1.00 15.00          ZN  ',
      'HETATM   14  C1  DHZ A 300      13.000  11.000  14.800  1.00 18.00           C  ',
      'HETATM   15  C2  DHZ A 300      14.390  11.000  14.800  1.00 18.00           C  ',
      'END',
    ].join('\n');
    const MINI_APO_PDB = MINI_HOLO_PDB.split('\n')
      .filter((l) => !l.includes('DHZ') && !l.includes('ZN'))
      .join('\n');

    // 1. Customized cropSegments: 'A' on ame_1ctt_cda (both unfetched and fetched pdbText) strips ligand from RFD4 -> MPNN / AF3
    for (const pdbText of ['', MINI_HOLO_PDB]) {
      const strippedBenchGraph: WorkflowGraph = {
        id: `stripped-bench-${pdbText ? 'fetched' : 'unfetched'}`,
        name: 'Stripped Benchmark Crop',
        nodes: [
          {
            id: 'pdb_1',
            adapterId: 'pdb-input',
            label: 'AME (Cropped)',
            params: { mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda', cropSegments: 'A', pdbText },
          },
          {
            id: 'rfd4_1',
            adapterId: 'rfd4-proteina',
            label: 'RFD4',
            params: { mode: 'motif', length: 180, motifSpan: 'A10-11', unindexedMotif: true, contig: '180-180' },
          },
          { id: 'mpnn_1', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN', params: { family: 'solublempnn' } },
          { id: 'af3_apo', adapterId: 'localfold-af2', label: 'AlphaFold 3', params: { model: 'af3', ligandCcdCodes: '' } },
        ],
        edges: [
          { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' },
          { fromNode: 'rfd4_1', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' },
          { fromNode: 'rfd4_1', fromPort: 'sequences', toNode: 'af3_apo', toPort: 'sequences' },
        ],
      };
      const issues = validateBioCapabilities(strippedBenchGraph);
      expect(issues.some((i) => i.code === 'mpnn-non-protein-stripped')).toBe(false);
      expect(issues.some((i) => i.code === 'af3-apo-refold-note')).toBe(false);
    }

    // 2. Live outputsByNode on pdb-input with customized cropSegments: 'A' does not re-inject bench.entities
    const parsedApo = parseStructureText(MINI_APO_PDB);
    const liveApoStruct: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'pdb-1-cropped-apo',
      label: '1CTT (A)',
      sourceModelId: 'pdb-input',
      length: parsedApo.nResidues,
      sequence: parsedApo.sequence,
      caCoords: parsedApo.caCoords,
      backbone4Coords: parsedApo.backbone4Coords,
      pdbText: MINI_APO_PDB,
      diagnostics: parsedApo.diagnostics,
      geometry: computeGeometryStats(parsedApo.caCoords),
    };
    const liveBenchGraph: WorkflowGraph = {
      id: 'live-stripped-bench',
      name: 'Live Stripped Bench',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          label: 'AME (Cropped)',
          params: { mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda', cropSegments: 'A' },
        },
        { id: 'mpnn_1', adapterId: 'web-protein-mpnn', label: 'ProteinMPNN', params: { family: 'solublempnn' } },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(liveBenchGraph, { pdb_1: { structure: liveApoStruct } }).some(
        (i) => i.code === 'mpnn-non-protein-stripped',
      ),
    ).toBe(false);

    // 3. Punctuation-only ligandCcdCodes (' , ; ') downstream of holo RFD4 preserves af3-apo-refold-note
    const punctAf3Graph: WorkflowGraph = {
      id: 'punct-af3-apo-note',
      name: 'Punctuation AF3 Apo Note',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda', cropSegments: 'A102, A:ZN, A:DHZ' },
        },
        {
          id: 'rfd4_1',
          adapterId: 'rfd4-proteina',
          params: { mode: 'motif', length: 180, motifSpan: 'A102', unindexedMotif: true, contig: '180-180' },
        },
        {
          id: 'af3_punct',
          adapterId: 'localfold-af2',
          params: { model: 'af3', ligandCcdCodes: ' , ; ' },
        },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' },
        { fromNode: 'rfd4_1', fromPort: 'sequences', toNode: 'af3_punct', toPort: 'sequences' },
      ],
    };
    expect(
      validateBioCapabilities(punctAf3Graph).some(
        (i) => i.nodeId === 'af3_punct' && i.code === 'af3-apo-refold-note',
      ),
    ).toBe(true);

    // 4. AF3 with rnaSequences triggers mpnn-non-protein-stripped on web-protein-mpnn and suppresses crop-dropped-ligands on ligandmpnn
    const rnaMpnnGraph: WorkflowGraph = {
      id: 'rna-mpnn-parity',
      name: 'RNA MPNN Parity',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'MKTAYIAKQRQISFVKSHFSRQ' } },
        { id: 'af3_rna', adapterId: 'localfold-af2', params: { model: 'af3', rnaSequences: 'AUGCAUGC' } },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', params: { family: 'solublempnn' } },
        { id: 'ligmpnn', adapterId: 'ligandmpnn', params: {} },
      ],
      edges: [
        { fromNode: 'seq', fromPort: 'sequences', toNode: 'af3_rna', toPort: 'sequences' },
        { fromNode: 'af3_rna', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'af3_rna', fromPort: 'structure', toNode: 'ligmpnn', toPort: 'structure' },
      ],
    };
    const rnaIssues = validateBioCapabilities(rnaMpnnGraph);
    const rnaMpnnWarn = rnaIssues.find((i) => i.nodeId === 'mpnn' && i.code === 'mpnn-non-protein-stripped');
    expect(rnaMpnnWarn).toBeDefined();
    expect(rnaMpnnWarn!.message).toContain('RNA');
    expect(rnaIssues.some((i) => i.nodeId === 'ligmpnn' && i.code === 'crop-dropped-ligands')).toBe(false);

    // 5. mode: 'custom' with empty pdbText and metal-only cropSegments ('A, A:ZN') -> GNINA rejects
    const customEmptyPdbMetal: WorkflowGraph = {
      id: 'custom-empty-pdb-metal',
      name: 'Custom Empty PDB Metal',
      nodes: [
        { id: 'pdb_1', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: '', cropSegments: 'A, A:ZN' } },
        { id: 'gnina_1', adapterId: 'gnina', params: { taskMode: 'score' } },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'gnina_1', toPort: 'structure' }],
    };
    expect(
      validateBioCapabilities(customEmptyPdbMetal).some(
        (i) => i.nodeId === 'gnina_1' && i.code === 'gnina-no-organic-ligand' && i.severity === 'error',
      ),
    ).toBe(true);

    // 6. Strengthened resumeCheckpoint test: unfetched pdb-id (1STP) passes statically without ckpt,
    //    but rejects when ckpt holds a checkpointed apo StructureArtifact (and passes if cropSegments invalidates ckpt).
    clearRcsbCache();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('.cif')) return new Response('Not Found', { status: 404 });
      return new Response(MINI_APO_PDB, { status: 200 });
    });
    vi.stubGlobal('fetch', fetchMock);
    const fetchApoGraph: WorkflowGraph = {
      id: 'ckpt-fetch-apo',
      name: 'Checkpoint Fetch Apo',
      nodes: [
        { id: 'pdb_1', adapterId: 'pdb-input', label: '1STP', params: { mode: 'pdb-id', pdbId: '1STP', cropSegments: '' } },
      ],
      edges: [],
    };
    const scheduler = new ExecutionScheduler();
    let apoPdbCkpt;
    try {
      await scheduler.run(fetchApoGraph, () => {});
      apoPdbCkpt = scheduler.exportCheckpointObject()!;
      expect(apoPdbCkpt).not.toBeNull();
    } finally {
      vi.unstubAllGlobals();
      clearRcsbCache();
    }

    const unfetchedPdbGnina: WorkflowGraph = {
      id: 'unfetched-1stp-gnina',
      name: 'Unfetched 1STP -> GNINA',
      nodes: [
        fetchApoGraph.nodes[0],
        { id: 'gnina_1', adapterId: 'gnina', label: 'GNINA 1.3', params: { taskMode: 'score' } },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'gnina_1', toPort: 'structure' }],
    };

    // Without resumeCheckpoint: unfetched pdb-id passes pre-run static validation
    const noCkptRes = await scheduler.run(unfetchedPdbGnina, () => {}, {
      signal: AbortSignal.abort(),
      returnPartialOnCancel: true,
    });
    expect(noCkptRes.status).toBe('cancelled');

    // With resumeCheckpoint containing apo StructureArtifact: pre-run rejects!
    await expect(
      scheduler.run(unfetchedPdbGnina, () => {}, {
        resumeCheckpoint: apoPdbCkpt,
        signal: AbortSignal.abort(),
        returnPartialOnCancel: true,
      }),
    ).rejects.toThrow(/no organic small-molecule ligand/);

    // Invalidating pdb_1 via cropSegments misses checkpoint -> passes pre-run again
    const invalidatedCropGraph: WorkflowGraph = {
      ...unfetchedPdbGnina,
      nodes: [
        { ...unfetchedPdbGnina.nodes[0], params: { mode: 'pdb-id', pdbId: '1STP', pdbText: '', cropSegments: 'A, A:BTN' } },
        unfetchedPdbGnina.nodes[1],
      ],
    };
    const invalidatedRes = await scheduler.run(invalidatedCropGraph, () => {}, {
      resumeCheckpoint: apoPdbCkpt,
      signal: AbortSignal.abort(),
      returnPartialOnCancel: true,
    });
    expect(invalidatedRes.status).toBe('cancelled');

    // 7. Optional-port / uncheckpointed new upstream guard:
    //    Checkpoint a 2-aa holo structure (valid in pdb-input due to DHZ, invalid for genie3 < 3 aa) -> cmp_1.
    //    Without new edge, cmp_1 matches cmpCkpt in resumeOutputs and genie3 rejects pre-run.
    //    Wiring uncheckpointed pdb_2 to cmp_1.reference MUST prevent cmp_1 from matching its stale checkpoint entry.
    const twoAaHoloPdb = MINI_HOLO_PDB.split('\n')
      .filter((l) => !l.includes('ALA A  12'))
      .join('\n');
    const cmpSeedGraph: WorkflowGraph = {
      id: 'ckpt-cmp-seed',
      name: 'Checkpoint Compare Seed',
      nodes: [
        { id: 'pdb_1', adapterId: 'pdb-input', label: '2aa Holo', params: { mode: 'custom', pdbText: twoAaHoloPdb } },
        { id: 'cmp_1', adapterId: 'structure-compare', label: 'Compare', params: {} },
      ],
      edges: [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'cmp_1', toPort: 'structures' }],
    };
    await scheduler.run(cmpSeedGraph, () => {});
    const cmpCkpt = scheduler.exportCheckpointObject()!;
    expect(cmpCkpt.nodes.cmp_1?.outputs?.structure).toBeDefined();

    const cmpMatchedRejectsGraph: WorkflowGraph = {
      id: 'ckpt-cmp-matched-rejects',
      name: 'Compare Matched Rejects Genie3',
      nodes: [
        ...cmpSeedGraph.nodes,
        { id: 'genie_1', adapterId: 'genie3', label: 'Genie 3', params: { taskMode: 'motif', motifSpan: 'A10-11' } },
      ],
      edges: [
        ...cmpSeedGraph.edges,
        { fromNode: 'cmp_1', fromPort: 'structure', toNode: 'genie_1', toPort: 'structure' },
      ],
    };
    await expect(
      scheduler.run(cmpMatchedRejectsGraph, () => {}, {
        resumeCheckpoint: cmpCkpt,
        signal: AbortSignal.abort(),
        returnPartialOnCancel: true,
      }),
    ).rejects.toThrow(/protein|residue/i);

    const cmpWithNewRefGraph: WorkflowGraph = {
      id: 'ckpt-cmp-new-ref',
      name: 'Compare With New Optional Ref',
      nodes: [
        ...cmpMatchedRejectsGraph.nodes,
        { id: 'pdb_2', adapterId: 'pdb-input', label: '1STP Ref', params: { mode: 'pdb-id', pdbId: '1STP' } },
      ],
      edges: [
        ...cmpMatchedRejectsGraph.edges,
        { fromNode: 'pdb_2', fromPort: 'structure', toNode: 'cmp_1', toPort: 'reference' },
      ],
    };
    const cmpRefRes = await scheduler.run(cmpWithNewRefGraph, () => {}, {
      resumeCheckpoint: cmpCkpt,
      signal: AbortSignal.abort(),
      returnPartialOnCancel: true,
    });
    expect(cmpRefRes.status).toBe('cancelled');
  });
});
