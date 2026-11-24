import { describe, expect, it } from 'vitest';
import {
  extractNucleicAcidSequencesFromStructureText,
  extractUpstreamHoloHints,
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from '$lib/engine/bio-capabilities';
import {
  extractProteinHeavyAtomsByResidueIndex,
  runDesignabilityScorerCpu,
} from '$lib/plugins/designability-scorer/cpu-runner';
import { computeGeometryStats, kabschAlignSubstructure } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  DesignabilityMetrics,
  DesignabilityRow,
  FoldedStructure,
  StructureArtifact,
  WorkflowGraph,
  WorkflowNode,
} from '$lib/engine/types';

const SEVEN_M5W_STRAND_B = 'GGTTATGAATGAAAAAGC';
const SEVEN_M5W_STRAND_C = 'GCTTTTTCATTCATAACC';

function buildDnaPdb(strands: Array<{ chain: string; seq: string }>): string {
  const baseToComp: Record<string, string> = { A: 'DA', C: 'DC', G: 'DG', T: 'DT', U: 'U' };
  const lines: string[] = ['MODEL        1'];
  let serial = 1;
  for (const { chain, seq } of strands) {
    for (let i = 0; i < seq.length; i++) {
      const comp = (baseToComp[seq[i]] ?? 'DA').padStart(3, ' ');
      const resSeq = String(i + 1).padStart(4, ' ');
      const x = (i * 3.4).toFixed(3).padStart(8, ' ');
      const y = (chain === 'B' ? 0 : 12).toFixed(3).padStart(8, ' ');
      const z = (0).toFixed(3).padStart(8, ' ');
      // Emit 2 atoms per nucleotide to verify per-residue deduplication
      lines.push(
        `ATOM  ${String(serial++).padStart(5, ' ')}  P   ${comp} ${chain}${resSeq}    ${x}${y}${z}  1.00 20.00           P  `,
      );
      lines.push(
        `ATOM  ${String(serial++).padStart(5, ' ')}  C1' ${comp} ${chain}${resSeq}    ${x}${y}${z}  1.00 20.00           C  `,
      );
    }
  }
  lines.push('ENDMDL');
  // Second model with decoy strand should be ignored
  lines.push('MODEL        2');
  lines.push(
    'ATOM    999  P    DA B   1       0.000   0.000   0.000  1.00 20.00           P  ',
  );
  lines.push('ENDMDL');
  lines.push('END');
  return lines.join('\n');
}

function buildDnaMmcif(strands: Array<{ chain: string; seq: string }>): string {
  const baseToComp: Record<string, string> = { A: 'DA', C: 'DC', G: 'DG', T: 'DT', U: 'U' };
  const lines = [
    'data_7M5W',
    'loop_',
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.label_atom_id',
    '_atom_site.label_comp_id',
    '_atom_site.auth_asym_id',
    '_atom_site.auth_seq_id',
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.pdbx_PDB_model_num',
  ];
  let id = 1;
  for (const { chain, seq } of strands) {
    for (let i = 0; i < seq.length; i++) {
      const comp = baseToComp[seq[i]] ?? 'DA';
      const x = (i * 3.4).toFixed(3);
      const y = chain === 'B' ? '0.000' : '12.000';
      lines.push(`ATOM ${id++} P ${comp} ${chain} ${i + 1} ? ${x} ${y} 0.000 1`);
      lines.push(`ATOM ${id++} "C1'" ${comp} ${chain} ${i + 1} ? ${x} ${y} 1.500 1`);
    }
  }
  // Model 2 row should be ignored
  lines.push(`ATOM ${id++} P DA B 99 ? 0.000 0.000 0.000 2`);
  return lines.join('\n');
}

function fmtAtom(
  serial: number,
  atomName: string,
  resName: string,
  chain: string,
  resSeq: number,
  x: number,
  y: number,
  z: number,
  elem: string,
  record: 'ATOM  ' | 'HETATM' = 'ATOM  ',
): string {
  const aName = atomName.length >= 4 ? atomName.slice(0, 4) : ` ${atomName.padEnd(3, ' ')}`;
  return (
    `${record}${String(serial).padStart(5, ' ')} ${aName} ${resName.padStart(3, ' ')} ` +
    `${chain}${String(resSeq).padStart(4, ' ')}    ` +
    `${x.toFixed(3).padStart(8, ' ')}${y.toFixed(3).padStart(8, ' ')}${z.toFixed(3).padStart(8, ' ')}` +
    `  1.00 90.00          ${elem.padStart(2, ' ')}  `
  );
}

function buildCatalyticTriadPdb(opts: { rotateSidechain?: boolean; includeLigand?: boolean } = {}): string {
  const lines: string[] = [];
  let serial = 1;
  // 4-residue protein (HIS-ASP-SER-ALA) where residues 0, 1, 2 carry catalytic sidechain tip atoms
  const residues: Array<{
    resName: string;
    ca: [number, number, number];
    tipName: string;
    tipElem: string;
    tipOffset: [number, number, number];
  }> = [
    {
      resName: 'HIS',
      ca: [0.0, 0.0, 0.0],
      tipName: 'NE2',
      tipElem: 'N',
      // Rotate only HIS NE2 + ASP OD1 rotamers while keeping SER OG & all Cα fixed
      // (non-rigid rotamer flip: 0 Å Cα RMSD, > 2 Å tip-atom RMSD!)
      tipOffset: opts.rotateSidechain ? [-2.5, -2.5, -1.8] : [0.0, 3.2, 2.0],
    },
    {
      resName: 'ASP',
      ca: [3.8, 0.5, 0.0],
      tipName: 'OD1',
      tipElem: 'O',
      tipOffset: opts.rotateSidechain ? [0.5, -2.8, 2.5] : [0.5, 2.8, -2.5],
    },
    {
      resName: 'SER',
      ca: [7.6, 0.0, 0.2],
      tipName: 'OG',
      tipElem: 'O',
      tipOffset: [1.0, 2.4, 1.8],
    },
    {
      resName: 'ALA',
      ca: [11.4, 0.3, 0.0],
      tipName: 'CB',
      tipElem: 'C',
      tipOffset: [0.0, 1.5, 0.0],
    },
  ];

  for (let i = 0; i < residues.length; i++) {
    const r = residues[i];
    const [cx, cy, cz] = r.ca;
    lines.push(fmtAtom(serial++, 'N', r.resName, 'A', i + 1, cx - 1.2, cy - 0.5, cz, 'N'));
    lines.push(fmtAtom(serial++, 'CA', r.resName, 'A', i + 1, cx, cy, cz, 'C'));
    lines.push(fmtAtom(serial++, 'C', r.resName, 'A', i + 1, cx + 1.2, cy - 0.3, cz, 'C'));
    lines.push(fmtAtom(serial++, 'O', r.resName, 'A', i + 1, cx + 1.5, cy - 1.4, cz, 'O'));
    lines.push(
      fmtAtom(
        serial++,
        r.tipName,
        r.resName,
        'A',
        i + 1,
        cx + r.tipOffset[0],
        cy + r.tipOffset[1],
        cz + r.tipOffset[2],
        r.tipElem,
      ),
    );
  }
  if (opts.includeLigand) {
    lines.push(fmtAtom(serial++, 'ZN', 'ZN', 'A', 200, 3.8, 4.0, 0.0, 'ZN', 'HETATM'));
    lines.push(fmtAtom(serial++, 'O', 'HOH', 'A', 300, 20.0, 20.0, 20.0, 'O', 'HETATM'));
  }
  lines.push('END');
  return lines.join('\n');
}

function makeStructureArtifactFromPdb(
  id: string,
  pdbText: string,
  extra: Partial<StructureArtifact> & {
    nativeMotifAtomCoords?: Array<{ resIdx: number; atomName: string; x: number; y: number; z: number }>;
  } = {},
): StructureArtifact {
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'rfd4-proteina',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    diagnostics: parsed.diagnostics,
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums),
    ...extra,
  } as StructureArtifact;
}

describe('Wave 1B: Holo Hints, BioGuard Mismatches, Tip-Atom Scorer, Holo iPAE & Unindexed Multi-Island Alignment', () => {
  it('P0-5: extractNucleicAcidSequencesFromStructureText extracts 18-bp 7M5W duplex from PDB and mmCIF', () => {
    const pdb = buildDnaPdb([
      { chain: 'B', seq: SEVEN_M5W_STRAND_B },
      { chain: 'C', seq: SEVEN_M5W_STRAND_C },
    ]);
    const cif = buildDnaMmcif([
      { chain: 'B', seq: SEVEN_M5W_STRAND_B },
      { chain: 'C', seq: SEVEN_M5W_STRAND_C },
    ]);

    expect(extractNucleicAcidSequencesFromStructureText(pdb)).toEqual([
      SEVEN_M5W_STRAND_B,
      SEVEN_M5W_STRAND_C,
    ]);
    expect(extractNucleicAcidSequencesFromStructureText(cif)).toEqual([
      SEVEN_M5W_STRAND_B,
      SEVEN_M5W_STRAND_C,
    ]);
  });

  it('P0-5: extractUpstreamHoloHints resolves 1CTT, 1OS7, 7C7M, 18-bp 7M5W, RFD4 presets, un-run pdbText, filters buffers, and exempts 1PRW CA', () => {
    const makeSimpleGraph = (
      pdbParams: Record<string, unknown>,
      rfd4Params?: Record<string, unknown>,
    ): WorkflowGraph => ({
      id: 'hints-test',
      name: 'Hints Test',
      nodes: [
        ...(pdbParams ? [{ id: 'pdb_1', adapterId: 'pdb-input', params: pdbParams }] : []),
        ...(rfd4Params ? [{ id: 'rfd4_1', adapterId: 'rfd4-proteina', params: rfd4Params }] : []),
        { id: 'fold_1', adapterId: 'localfold-af2', params: { model: 'rosettafold3' } },
      ],
      edges: rfd4Params
        ? [
            ...(pdbParams
              ? [{ fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' }]
              : []),
            { fromNode: 'rfd4_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' },
          ]
        : [{ fromNode: 'pdb_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' }],
    });

    // 1CTT, 1OS7, 7C7M benchmarks
    expect(
      extractUpstreamHoloHints(
        makeSimpleGraph({ mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda' }),
        'fold_1',
      ),
    ).toEqual({ ligandCcdCodes: ['ZN', 'DHZ'], dnaSequences: [] });

    expect(
      extractUpstreamHoloHints(
        makeSimpleGraph({ mode: 'pdb-id', pdbId: '1OS7', benchmarkId: 'ame_1os7_taud' }),
        'fold_1',
      ),
    ).toEqual({ ligandCcdCodes: ['FE2', 'AKG', 'TAU'], dnaSequences: [] });

    expect(
      extractUpstreamHoloHints(
        makeSimpleGraph({ mode: 'pdb-id', pdbId: '7C7M', benchmarkId: 'sm_7c7m_sam' }),
        'fold_1',
      ),
    ).toEqual({ ligandCcdCodes: ['SAM'], dnaSequences: [] });

    // 7M5W exact 18-bp fallback when unfetched
    expect(
      extractUpstreamHoloHints(
        makeSimpleGraph({ mode: 'pdb-id', pdbId: '7M5W', benchmarkId: 'rfd4_7m5w_bdna' }),
        'fold_1',
      ),
    ).toEqual({
      ligandCcdCodes: [],
      dnaSequences: [SEVEN_M5W_STRAND_B, SEVEN_M5W_STRAND_C],
    });

    // 7M5W live/un-run pdbText extracts exact duplex directly
    const customDuplexPdb = buildDnaPdb([
      { chain: 'B', seq: SEVEN_M5W_STRAND_B },
      { chain: 'C', seq: SEVEN_M5W_STRAND_C },
    ]);
    expect(
      extractUpstreamHoloHints(
        makeSimpleGraph({
          mode: 'custom',
          pdbText: customDuplexPdb,
        }),
        'fold_1',
      ),
    ).toEqual({
      ligandCcdCodes: [],
      dnaSequences: [SEVEN_M5W_STRAND_B, SEVEN_M5W_STRAND_C],
    });

    // Buffer filtering on live StructureArtifact + 1PRW CA exemption
    const live1prwWithCaAndBuffers: StructureArtifact = {
      ...makeStructureArtifactFromPdb('pdb_1prw', buildCatalyticTriadPdb()),
      benchmarkId: 'rfd_1prw_efhand',
      diagnostics: {
        chainCount: 1,
        chainLengths: [4],
        hasLigands: true,
        ligandNames: ['CA', 'SO4', 'GOL', 'EDO', 'HOH', 'CL', 'NA'],
        hasNucleicAcids: false,
        nucleicAcidChains: [],
      },
    };
    const prwGraph = makeSimpleGraph({
      mode: 'pdb-id',
      pdbId: '1PRW',
      benchmarkId: 'rfd_1prw_efhand',
    });
    expect(
      extractUpstreamHoloHints(prwGraph, 'fold_1', {
        pdb_1: { structure: live1prwWithCaAndBuffers },
      }),
    ).toEqual({ ligandCcdCodes: [], dnaSequences: [] });

    // RFD4 built-in presets (sm_sam -> SAM, dna_7m5w -> 18-bp DNA)
    const presetSamGraph: WorkflowGraph = {
      id: 'preset-sam',
      name: 'Preset SAM',
      nodes: [
        { id: 'rfd4_1', adapterId: 'rfd4-proteina', params: { mode: 'preset', presetId: 'sm_sam' } },
        { id: 'fold_1', adapterId: 'localfold-af2', params: { model: 'rosettafold3' } },
      ],
      edges: [{ fromNode: 'rfd4_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' }],
    };
    expect(extractUpstreamHoloHints(presetSamGraph, 'fold_1')).toEqual({
      ligandCcdCodes: ['SAM'],
      dnaSequences: [],
    });

    const presetDnaGraph: WorkflowGraph = {
      id: 'preset-dna',
      name: 'Preset DNA',
      nodes: [
        { id: 'rfd4_1', adapterId: 'rfd4-proteina', params: { mode: 'preset', presetId: 'dna_7m5w' } },
        { id: 'fold_1', adapterId: 'localfold-af2', params: { model: 'rosettafold3' } },
      ],
      edges: [{ fromNode: 'rfd4_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' }],
    };
    expect(extractUpstreamHoloHints(presetDnaGraph, 'fold_1')).toEqual({
      ligandCcdCodes: [],
      dnaSequences: [SEVEN_M5W_STRAND_B, SEVEN_M5W_STRAND_C],
    });
  });

  it('P0-6: BioGuard catches af3-holo-mismatch and af3-stale-holo-on-apo with 1-click sync/clear fixes', () => {
    // Case 1a: 7C7M (SAM) upstream while LocalFold still has stale ZN, DHZ from 1CTT
    const samStaleGraph: WorkflowGraph = {
      id: 'sam-stale-holo',
      name: 'SAM with stale ZN, DHZ',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '7C7M', benchmarkId: 'sm_7c7m_sam', cropSegments: 'A:SAM' },
        },
        {
          id: 'rfd4_1',
          adapterId: 'rfd4-proteina',
          params: { mode: 'motif', length: 110, contig: '110-110', hbondSelection: 'A/SAM/*/*' },
        },
        { id: 'mpnn_1', adapterId: 'ligandmpnn', params: {} },
        {
          id: 'fold_1',
          adapterId: 'localfold-af2',
          label: 'RoseTTAFold 3',
          params: { model: 'rosettafold3', ligandCcdCodes: 'ZN, DHZ' },
        },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' },
        { fromNode: 'rfd4_1', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' },
        { fromNode: 'mpnn_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' },
      ],
    };
    const samIssues = validateBioCapabilities(samStaleGraph);
    const mismatch = samIssues.find((i) => i.nodeId === 'fold_1' && i.code === 'af3-holo-mismatch');
    expect(mismatch).toBeDefined();
    expect(mismatch!.severity).toBe('warning');
    expect(mismatch!.fix?.paramsPatch).toEqual({ ligandCcdCodes: 'SAM', dnaSequences: '' });

    // Case 1b: 7M5W DNA upstream while LocalFold still has stale ZN, DHZ and empty dnaSequences
    const dnaStaleGraph: WorkflowGraph = {
      id: 'dna-stale-holo',
      name: '7M5W with stale ZN, DHZ',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '7M5W', benchmarkId: 'rfd4_7m5w_bdna', cropSegments: 'B, C' },
        },
        {
          id: 'rfd4_1',
          adapterId: 'rfd4-proteina',
          params: { mode: 'motif', length: 90, contig: '90-90' },
        },
        { id: 'mpnn_1', adapterId: 'ligandmpnn', params: {} },
        {
          id: 'fold_1',
          adapterId: 'localfold-af2',
          label: 'RoseTTAFold 3',
          params: { model: 'rosettafold3', ligandCcdCodes: 'ZN, DHZ', dnaSequences: '' },
        },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' },
        { fromNode: 'rfd4_1', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' },
        { fromNode: 'mpnn_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' },
      ],
    };
    const dnaMismatch = validateBioCapabilities(dnaStaleGraph).find(
      (i) => i.nodeId === 'fold_1' && i.code === 'af3-holo-mismatch',
    );
    expect(dnaMismatch).toBeDefined();
    expect(dnaMismatch!.fix?.paramsPatch).toEqual({
      ligandCcdCodes: '',
      dnaSequences: `${SEVEN_M5W_STRAND_B}:${SEVEN_M5W_STRAND_C}`,
    });

    // Case 1c: 1CTT missing one of the upstream ligands (ZN without DHZ)
    const partial1cttGraph: WorkflowGraph = {
      id: '1ctt-missing-dhz',
      name: '1CTT missing DHZ',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: {
            mode: 'pdb-id',
            pdbId: '1CTT',
            benchmarkId: 'ame_1ctt_cda',
            cropSegments: 'A102, A104, A129, A132, A:ZN, A:DHZ',
          },
        },
        {
          id: 'rfd4_1',
          adapterId: 'rfd4-proteina',
          params: { mode: 'motif', length: 180, contig: '180-180', motifSpan: 'A102, A104, A129, A132', unindexedMotif: true },
        },
        {
          id: 'fold_1',
          adapterId: 'localfold-af2',
          params: { model: 'rosettafold3', ligandCcdCodes: 'ZN' },
        },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' },
        { fromNode: 'rfd4_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' },
      ],
    };
    const partialMismatch = validateBioCapabilities(partial1cttGraph).find(
      (i) => i.nodeId === 'fold_1' && i.code === 'af3-holo-mismatch',
    );
    expect(partialMismatch).toBeDefined();
    expect(partialMismatch!.fix?.paramsPatch).toEqual({ ligandCcdCodes: 'ZN, DHZ', dnaSequences: '' });

    // Case 2: Apo protein-only upstream (1PRW) while LocalFold still has stale ZN, DHZ
    const apoStaleGraph: WorkflowGraph = {
      id: 'apo-stale-holo',
      name: '1PRW apo with stale ZN, DHZ',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: {
            mode: 'pdb-id',
            pdbId: '1PRW',
            benchmarkId: 'rfd_1prw_efhand',
            cropSegments: 'A16-35, A52-71',
          },
        },
        {
          id: 'rfd4_1',
          adapterId: 'rfd4-proteina',
          params: { mode: 'motif', length: 76, contig: '10-10,A16-35,16-16,A52-71,10-10', motifSpan: 'A16-35, A52-71' },
        },
        {
          id: 'fold_1',
          adapterId: 'localfold-af2',
          label: 'RoseTTAFold 3',
          params: { model: 'rosettafold3', ligandCcdCodes: 'ZN, DHZ' },
        },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' },
        { fromNode: 'rfd4_1', fromPort: 'sequences', toNode: 'fold_1', toPort: 'sequences' },
      ],
    };
    const staleApo = validateBioCapabilities(apoStaleGraph).find(
      (i) => i.nodeId === 'fold_1' && i.code === 'af3-stale-holo-on-apo',
    );
    expect(staleApo).toBeDefined();
    expect(staleApo!.severity).toBe('warning');
    expect(staleApo!.fix?.paramsPatch).toEqual({ ligandCcdCodes: '', dnaSequences: '' });
  });

  it('P1-8: 7M5W -> RFD4 -> LigandMPNN -> RF3 produces ZERO false binder-single-seq-msa or multimer warnings, and Genie 3 unconditional never throws protein-only error', () => {
    const dnaLivePdb = `${buildCatalyticTriadPdb()}\n${buildDnaPdb([
      { chain: 'B', seq: SEVEN_M5W_STRAND_B },
      { chain: 'C', seq: SEVEN_M5W_STRAND_C },
    ]).replace(/^MODEL.*\n|ENDMDL\n?|END\n?/gm, '')}\nEND`;
    const rfd4LiveDnaStruct = makeStructureArtifactFromPdb('rfd4-7m5w-live', dnaLivePdb);
    expect(rfd4LiveDnaStruct.diagnostics?.hasNucleicAcids).toBe(true);
    expect(rfd4LiveDnaStruct.diagnostics?.nucleicAcidChains).toEqual(['B', 'C']);

    const graph7m5w: WorkflowGraph = {
      id: '7m5w-rfd4-mpnn-rf3',
      name: '7M5W -> RFD4 -> LigandMPNN -> RF3',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '7M5W', benchmarkId: 'rfd4_7m5w_bdna', cropSegments: 'B, C' },
        },
        {
          id: 'rfd4_1',
          adapterId: 'rfd4-proteina',
          params: { mode: 'motif', length: 90, contig: '90-90' },
        },
        {
          id: 'mpnn_1',
          adapterId: 'ligandmpnn',
          params: { fixedPositions: 'A1' },
        },
        {
          id: 'rf3_1',
          adapterId: 'localfold-af2',
          label: 'RoseTTAFold 3',
          params: {
            model: 'rosettafold3',
            msaMode: 'single',
            dnaSequences: `${SEVEN_M5W_STRAND_B}:${SEVEN_M5W_STRAND_C}`,
          },
        },
      ],
      edges: [
        { fromNode: 'pdb_1', fromPort: 'structure', toNode: 'rfd4_1', toPort: 'structure' },
        { fromNode: 'rfd4_1', fromPort: 'structure', toNode: 'mpnn_1', toPort: 'structure' },
        { fromNode: 'mpnn_1', fromPort: 'sequences', toNode: 'rf3_1', toPort: 'sequences' },
      ],
    };

    for (const outputs of [undefined, { rfd4_1: { structure: rfd4LiveDnaStruct } }]) {
      const issues = validateBioCapabilities(graph7m5w, outputs);
      expect(issues.some((i) => i.code === 'binder-single-seq-msa')).toBe(false);
      expect(issues.some((i) => i.code === 'multimer-into-af2-monomer')).toBe(false);
      expect(issues.some((i) => i.code === 'af3-holo-mismatch')).toBe(false);
      expect(issues.some((i) => i.code === 'af3-stale-holo-on-apo')).toBe(false);
    }

    // Genie 3 in unconditional mode should NOT throw on a wired holo/DNA template
    const holo1cttStruct: StructureArtifact = {
      ...makeStructureArtifactFromPdb('pdb-1ctt', buildCatalyticTriadPdb({ includeLigand: true })),
      benchmarkId: 'ame_1ctt_cda',
    };
    const genieUncondNode: WorkflowNode = {
      id: 'g3_uncond',
      adapterId: 'genie3',
      params: { taskMode: 'unconditional', length: 80 },
    };
    expect(() => validateRuntimeNodeInputs(genieUncondNode, { structure: holo1cttStruct })).not.toThrow();
    // Sanity check: in motif mode with incompatible benchmark it still hard-blocks
    const genieMotifNode: WorkflowNode = {
      id: 'g3_motif',
      adapterId: 'genie3',
      params: { taskMode: 'motif', motifSpan: 'A1-3' },
    };
    expect(() => validateRuntimeNodeInputs(genieMotifNode, { structure: holo1cttStruct })).toThrow();
  });

  it('P0-3 & P1-8: designability-scorer computes exact genMotifAtomRmsd & refoldMotifAtomRmsd (catching rotated rotamer with 0 Å Cα RMSD) and holo protein-ligand iPAE', async () => {
    const nativePdb = buildCatalyticTriadPdb({ rotateSidechain: false, includeLigand: true });
    const rotatedSidechainPdb = buildCatalyticTriadPdb({ rotateSidechain: true, includeLigand: true });

    const nativeAtomsByRes = extractProteinHeavyAtomsByResidueIndex(nativePdb);
    expect(nativeAtomsByRes.length).toBe(4);
    expect(nativeAtomsByRes[0].get('NE2')).toBeDefined();
    expect(nativeAtomsByRes[1].get('OD1')).toBeDefined();
    expect(nativeAtomsByRes[2].get('OG')).toBeDefined();

    const nativeMotifAtomCoords = [
      { resIdx: 0, atomName: 'CA', x: nativeAtomsByRes[0].get('CA')![0], y: nativeAtomsByRes[0].get('CA')![1], z: nativeAtomsByRes[0].get('CA')![2] },
      { resIdx: 0, atomName: 'NE2', x: nativeAtomsByRes[0].get('NE2')![0], y: nativeAtomsByRes[0].get('NE2')![1], z: nativeAtomsByRes[0].get('NE2')![2] },
      { resIdx: 1, atomName: 'CA', x: nativeAtomsByRes[1].get('CA')![0], y: nativeAtomsByRes[1].get('CA')![1], z: nativeAtomsByRes[1].get('CA')![2] },
      { resIdx: 1, atomName: 'OD1', x: nativeAtomsByRes[1].get('OD1')![0], y: nativeAtomsByRes[1].get('OD1')![1], z: nativeAtomsByRes[1].get('OD1')![2] },
      { resIdx: 2, atomName: 'CA', x: nativeAtomsByRes[2].get('CA')![0], y: nativeAtomsByRes[2].get('CA')![1], z: nativeAtomsByRes[2].get('CA')![2] },
      { resIdx: 2, atomName: 'OG', x: nativeAtomsByRes[2].get('OG')![0], y: nativeAtomsByRes[2].get('OG')![1], z: nativeAtomsByRes[2].get('OG')![2] },
    ];

    const parsedNative = parseStructureText(nativePdb);
    const designedStruct = makeStructureArtifactFromPdb('design-holo-1', nativePdb, {
      motifIndices: [0, 1, 2],
      nativeMotifCoords: [
        [parsedNative.caCoords[0], parsedNative.caCoords[1], parsedNative.caCoords[2]],
        [parsedNative.caCoords[3], parsedNative.caCoords[4], parsedNative.caCoords[5]],
        [parsedNative.caCoords[6], parsedNative.caCoords[7], parsedNative.caCoords[8]],
      ],
      nativeMotifAtomCoords,
    });

    // Build 6x6 PAE matrix: 4 protein tokens [0..3] + 2 holo ligand/DNA tokens [4..5]
    // Protein-protein = 1.0, protein->nonprot = 3.0, nonprot->protein = 5.0 => mean holo iPAE = 4.0
    const paeDim = 6;
    const pae = new Float32Array(paeDim * paeDim);
    for (let i = 0; i < paeDim; i++) {
      for (let j = 0; j < paeDim; j++) {
        if (i < 4 && j < 4) pae[i * paeDim + j] = 1.0;
        else if (i < 4 && j >= 4) pae[i * paeDim + j] = 3.0;
        else if (i >= 4 && j < 4) pae[i * paeDim + j] = 5.0;
        else pae[i * paeDim + j] = 0.5;
      }
    }

    const parsedRot = parseStructureText(rotatedSidechainPdb);
    const folds: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'folds-1',
      parentStructureId: 'design-holo-1',
      sourceStructureId: 'design-holo-1',
      folds: [
        {
          sequenceId: 'seq-rotamer-flip',
          sequence: parsedRot.sequence,
          foldModelId: 'rosettafold3',
          sourceStructureId: 'design-holo-1',
          pdbText: rotatedSidechainPdb,
          caCoords: parsedRot.caCoords,
          plddt: new Float32Array([92, 91, 90, 89]),
          meanPlddt: 90.5,
          ptm: 0.88,
          iptm: 0.84,
          chainLengths: [4],
          chainIds: ['A', 'A', 'A', 'A'],
          resNums: [1, 2, 3, 4],
          pae,
          elapsedMs: 100,
        },
      ],
    };

    const scorerNode: WorkflowNode = {
      id: 'scorer_1',
      adapterId: 'designability-scorer',
      params: { scRmsdThreshold: 2.0, plddtThreshold: 70, motifRmsdThreshold: 1.0, ipaeThreshold: 10.0 },
    };

    const res = await runDesignabilityScorerCpu(scorerNode, {
      structure: designedStruct,
      folds,
    });
    const metrics = res.metrics as DesignabilityMetrics & {
      genMotifAtomRmsd?: number | null;
      refoldMotifAtomRmsd?: number | null;
    };
    const row = metrics.rows[0] as DesignabilityRow & {
      genMotifAtomRmsd?: number | null;
      refoldMotifAtomRmsd?: number | null;
    };

    // Backbone Cα RMSD is ~0 Å, catching the rotamer flip ONLY via tip-atom RMSD (> 2 Å)!
    expect(row.genMotifRmsd).toBeLessThan(1e-3);
    expect(row.refoldMotifRmsd).toBeLessThan(1e-3);
    expect(row.genMotifAtomRmsd).toBeDefined();
    expect(row.genMotifAtomRmsd!).toBeLessThan(1e-3);
    expect(row.refoldMotifAtomRmsd).toBeDefined();
    expect(row.refoldMotifAtomRmsd!).toBeGreaterThan(2.0);
    expect(metrics.genMotifAtomRmsd).toBeCloseTo(row.genMotifAtomRmsd!, 5);
    expect(metrics.refoldMotifAtomRmsd).toBeCloseTo(row.refoldMotifAtomRmsd!, 5);

    // Holo protein↔non-protein iPAE = mean(3.0, 5.0) = 4.0
    expect(row.ipae).toBeCloseTo(4.0, 5);
  });

  it('P1-5: kabschAlignSubstructure accurately aligns a 2-island 20-aa unindexed motif (m > 7) with swapped island order', () => {
    // Construct a 2-island asymmetric 20-aa motif (island 1: 10 aa helix-like, island 2: 10 aa curved loop)
    const island1: Array<[number, number, number]> = [];
    const island2: Array<[number, number, number]> = [];
    for (let i = 0; i < 10; i++) {
      const a = i * 1.1;
      island1.push([i * 1.5, 2.3 * Math.cos(a), 2.3 * Math.sin(a)]);
      const b = i * 0.65;
      island2.push([22.0 + i * 1.9, 8.0 + 3.5 * Math.sin(b), -6.0 + 1.2 * Math.cos(2 * b)]);
    }
    const shortCoords = new Float32Array(20 * 3);
    const shortResNums: number[] = [];
    for (let i = 0; i < 10; i++) {
      shortCoords[i * 3 + 0] = island1[i][0];
      shortCoords[i * 3 + 1] = island1[i][1];
      shortCoords[i * 3 + 2] = island1[i][2];
      shortResNums.push(16 + i);
    }
    for (let i = 0; i < 10; i++) {
      const idx = 10 + i;
      shortCoords[idx * 3 + 0] = island2[i][0];
      shortCoords[idx * 3 + 1] = island2[i][1];
      shortCoords[idx * 3 + 2] = island2[i][2];
      shortResNums.push(52 + i);
    }

    // Apply an arbitrary rigid rotation + translation to place swapped islands in a 60-aa scaffold:
    // Island 2 sits at scaffold indices [5..14], Island 1 sits at scaffold indices [35..44]
    const theta = 0.87;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const rotTranslate = (p: [number, number, number]): [number, number, number] => [
      c * p[0] - s * p[1] + 15.0,
      s * p[0] + c * p[1] - 9.0,
      p[2] + 4.5,
    ];

    const longCoords = new Float32Array(60 * 3);
    for (let i = 0; i < 60; i++) {
      longCoords[i * 3 + 0] = i * 3.8;
      longCoords[i * 3 + 1] = 40.0;
      longCoords[i * 3 + 2] = -30.0;
    }
    for (let i = 0; i < 10; i++) {
      const p2 = rotTranslate(island2[i]);
      longCoords[(5 + i) * 3 + 0] = p2[0];
      longCoords[(5 + i) * 3 + 1] = p2[1];
      longCoords[(5 + i) * 3 + 2] = p2[2];

      const p1 = rotTranslate(island1[i]);
      longCoords[(35 + i) * 3 + 0] = p1[0];
      longCoords[(35 + i) * 3 + 1] = p1[1];
      longCoords[(35 + i) * 3 + 2] = p1[2];
    }

    // Ascending knownIndices: [5..14, 35..44] — swapped relative to shortCoords [island1, island2]!
    const knownIndices = [
      ...Array.from({ length: 10 }, (_, i) => 5 + i),
      ...Array.from({ length: 10 }, (_, i) => 35 + i),
    ];

    // 1. Without knownNativeCoords (pure island-level Kabsch over longCoords)
    const alignedNoNative = kabschAlignSubstructure(shortCoords, longCoords, shortResNums, {
      knownIndices,
    });
    expect(alignedNoNative.alignedResidues).toBe(20);
    expect(alignedNoNative.rmsd).toBeLessThan(1e-3);

    // 2. With rotated/shifted knownNativeCoords (different frame so step 1a fails, step 1c resolves!)
    const knownNativeCoords: number[][] = [
      ...island2.map((p) => [p[0] + 100, p[1] - 50, p[2] + 25]),
      ...island1.map((p) => [p[0] + 100, p[1] - 50, p[2] + 25]),
    ];
    const alignedWithShiftedNative = kabschAlignSubstructure(shortCoords, longCoords, shortResNums, {
      knownIndices,
      knownNativeCoords,
    });
    expect(alignedWithShiftedNative.alignedResidues).toBe(20);
    expect(alignedWithShiftedNative.rmsd).toBeLessThan(1e-3);
  });
});
