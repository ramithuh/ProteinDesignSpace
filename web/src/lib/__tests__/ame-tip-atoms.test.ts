import { describe, expect, it } from 'vitest';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { getBenchmarkById } from '$lib/engine/benchmarks';
import {
  assertRfd4ConditionSelectorsMatch,
  buildCanonicalConditionIntent,
  compileConditionToRfd4,
  extractRfd4DesignMetadata,
  extractRfd4TemplateContext,
  parseMotifAtomsSpec,
  resolveRfd4CompiledInput,
} from '$lib/engine/conditioning';
import {
  buildResetToBenchmarkParams,
  parseLengthFromContig,
  reconcileBenchmarkSeeding,
  switchBenchmark,
} from '$lib/engine/benchmark-sync';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { materializePdbInputNode } from '$lib/plugins/pdb-input/cpu-runner';
import type { WorkflowGraph } from '$lib/engine/types';

const CIF_HEADER = [
  'data_test',
  'loop_',
  '_atom_site.group_PDB',
  '_atom_site.id',
  '_atom_site.type_symbol',
  '_atom_site.label_atom_id',
  '_atom_site.label_alt_id',
  '_atom_site.label_comp_id',
  '_atom_site.label_asym_id',
  '_atom_site.label_entity_id',
  '_atom_site.label_seq_id',
  '_atom_site.pdbx_PDB_ins_code',
  '_atom_site.Cartn_x',
  '_atom_site.Cartn_y',
  '_atom_site.Cartn_z',
  '_atom_site.occupancy',
  '_atom_site.B_iso_or_equiv',
  '_atom_site.auth_seq_id',
  '_atom_site.auth_comp_id',
  '_atom_site.auth_asym_id',
  '_atom_site.auth_atom_id',
  '_atom_site.pdbx_PDB_model_num',
];

function mkAllAtomResidueRows(
  startSerial: number,
  labelChain: string,
  authChain: string,
  entityId: number,
  residues: Array<{
    labelSeq: number;
    authSeq: number;
    resName: string;
    atoms: Array<[string, string]>;
  }>,
): { rows: string[]; nextSerial: number } {
  const rows: string[] = [];
  let s = startSerial;
  for (let i = 0; i < residues.length; i++) {
    const r = residues[i];
    const x0 = i * 5.0;
    for (let j = 0; j < r.atoms.length; j++) {
      const [nm, el] = r.atoms[j];
      const qNm = nm.includes("'") ? `"${nm}"` : nm;
      const x = (x0 + (j % 4) * 1.2).toFixed(3);
      const y = (Math.floor(j / 4) * 1.3).toFixed(3);
      const z = ((j % 3) * 0.7).toFixed(3);
      rows.push(
        `ATOM ${s++} ${el} ${qNm} . ${r.resName} ${labelChain} ${entityId} ${r.labelSeq} ? ${x} ${y} ${z} 1.00 10.00 ${r.authSeq} ${r.resName} ${authChain} ${qNm} 1`,
      );
    }
  }
  return { rows, nextSerial: s };
}

async function loadVendoredRfd4() {
  const base = resolve(__dirname, '../../../static/vendor/rfd4-proteina/shared/rfd4p');
  const [rfd4, cifMod, specMod, prepareMod] = await Promise.all([
    import(pathToFileURL(resolve(base, 'pipeline/index.js')).href),
    import(pathToFileURL(resolve(base, 'pipeline/cif.js')).href),
    import(pathToFileURL(resolve(base, 'pipeline/spec.js')).href),
    import(pathToFileURL(resolve(base, 'pipeline/prepare.js')).href),
  ]);
  return { rfd4, cifMod, specMod, prepareMod };
}

describe('Wave 1A: AME Tip-Atom Conditioning, Multi-Role H-Bonds, Unindexed Auto-Lock & Sync Fixes', () => {
  it('compiles 1CTT (ame_1ctt_cda) with exact 10 AME tip atoms in C_CRDGRP and 4 distinct C_IDX unindexed groups via vendored RFD4', async () => {
    const { rfd4, cifMod, specMod, prepareMod } = await loadVendoredRfd4();

    const cttRes = mkAllAtomResidueRows(1, 'A', 'A', 1, [
      {
        labelSeq: 98,
        authSeq: 102,
        resName: 'HIS',
        atoms: [
          ['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'],
          ['CB', 'C'], ['CG', 'C'], ['ND1', 'N'], ['CD2', 'C'], ['CE1', 'C'], ['NE2', 'N'],
        ],
      },
      {
        labelSeq: 100,
        authSeq: 104,
        resName: 'GLU',
        atoms: [
          ['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'],
          ['CB', 'C'], ['CG', 'C'], ['CD', 'C'], ['OE1', 'O'], ['OE2', 'O'],
        ],
      },
      {
        labelSeq: 125,
        authSeq: 129,
        resName: 'CYS',
        atoms: [
          ['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'],
          ['CB', 'C'], ['SG', 'S'],
        ],
      },
      {
        labelSeq: 128,
        authSeq: 132,
        resName: 'CYS',
        atoms: [
          ['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'],
          ['CB', 'C'], ['SG', 'S'],
        ],
      },
    ]);
    const cttCif = [
      ...CIF_HEADER,
      ...cttRes.rows,
      `HETATM ${cttRes.nextSerial} ZN ZN . ZN B 2 . ? 2.000 2.000 2.000 1.00 10.00 296 ZN A ZN 1`,
      `HETATM ${cttRes.nextSerial + 1} C C1 . DHZ C 3 . ? 3.500 2.000 2.000 1.00 10.00 295 DHZ A C1 1`,
      `HETATM ${cttRes.nextSerial + 2} O O1 . DHZ C 3 . ? 4.700 2.000 2.000 1.00 10.00 295 DHZ A O1 1`,
      '#',
    ].join('\n');

    const bench = getBenchmarkById('ame_1ctt_cda')!;
    expect(bench.biology).toMatch(/tip-atom/i);
    expect(bench.conditionParams.motifAtoms).toBe(
      'A102:ND1,CG,CE1; A104:OE1,CD; A129:SG,CB; A132:SG,CB,CA',
    );

    const ctx = extractRfd4TemplateContext(cttCif);
    const intent = buildCanonicalConditionIntent({ ...bench.conditionParams, contig: '24-24' });
    const compiled = compileConditionToRfd4(intent, ctx);

    expect(compiled.contig).toBe('24-24,A98,A100,A125,A128,/0,B296,/0,C295');
    expect(compiled.conditions.C_CRDGRP).toEqual({
      1: [
        {
          select:
            "sel('A/*/98/[ND1,CG,CE1]') | sel('A/*/100/[OE1,CD]') | sel('A/*/125/[SG,CB]') | sel('A/*/128/[SG,CB,CA]') | sel('B/*/*/*') | sel('C/*/*/*')",
        },
      ],
    });
    expect(compiled.conditions.C_SEQ).toEqual({
      true: [{ select: "sel('A/*/[98,100,125,128]/*')" }],
    });
    expect(compiled.conditions.C_IDX).toEqual({
      false: [
        { select: "sel('A/*/98/*')", group: 'motif_0' },
        { select: "sel('A/*/100/*')", group: 'motif_1' },
        { select: "sel('A/*/125/*')", group: 'motif_2' },
        { select: "sel('A/*/128/*')", group: 'motif_3' },
      ],
    });

    // Live vendored RFD4 spec evaluation + buildBatch verification
    const cttAtoms = cifMod.parseCifToAtomArray(cttCif, { fixArginines: true });
    prepareMod.assignIonCharges(cttAtoms);
    expect(() =>
      assertRfd4ConditionSelectorsMatch(cttAtoms, compiled.contig, compiled.conditions, specMod),
    ).not.toThrow();

    const annotated = specMod.buildAtomArrayFromConditionSpec(
      { contig: compiled.contig, conditions: compiled.conditions },
      { inputAtomArray: cttAtoms },
    );
    const crdGrp: number[] = annotated.annotations.condition_CoordinateGroup_1_atom;
    const proteinTipHits: string[] = [];
    for (let i = 0; i < annotated.nAtoms; i++) {
      if (annotated.is_polymer[i] && crdGrp[i] === 1) {
        proteinTipHits.push(`${annotated.res_name[i]}:${annotated.atom_name[i]}`);
      }
    }
    expect(proteinTipHits).toHaveLength(10);
    expect(proteinTipHits).toEqual([
      'HIS:CG', 'HIS:ND1', 'HIS:CE1',
      'GLU:CD', 'GLU:OE1',
      'CYS:CB', 'CYS:SG',
      'CYS:CA', 'CYS:CB', 'CYS:SG',
    ]);

    // Verify 4 distinct index-group IDs across the 4 unindexed catalytic residues
    const isIndexed: boolean[] = annotated.annotations.condition_index_1_residue;
    const idxGrp: number[] = annotated.annotations['condition_index-group_1_token'];
    const unidxGroups = new Set<number>();
    for (let i = 0; i < annotated.nAtoms; i++) {
      if (isIndexed[i] === false) unidxGroups.add(idxGrp[i]);
    }
    expect(unidxGroups.size).toBe(4);

    // Run full buildBatch
    const built = rfd4.buildBatch(
      { contig: compiled.contig, conditions: compiled.conditions },
      { inputAtomArray: cttAtoms, seed: 42 },
    );
    expect(built.n).toBe(31); // 24 scaffold + 4 unindexed catalytic + 1 ZN + 2 DHZ
    const inCtxCount = Array.from(built.batch.is_in_context_conditioning.data).filter(
      (v) => Number(v) > 0,
    ).length;
    expect(inCtxCount).toBe(4);
  });

  it('compiles 1OS7 (ame_1os7_taud) with exact 14 AME tip atoms in C_CRDGRP and 4 distinct C_IDX groups via vendored RFD4', async () => {
    const { rfd4, cifMod, specMod, prepareMod } = await loadVendoredRfd4();

    const os7Res = mkAllAtomResidueRows(1, 'A', 'A', 1, [
      {
        labelSeq: 95,
        authSeq: 99,
        resName: 'HIS',
        atoms: [
          ['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'],
          ['CB', 'C'], ['CG', 'C'], ['ND1', 'N'], ['CD2', 'C'], ['CE1', 'C'], ['NE2', 'N'],
        ],
      },
      {
        labelSeq: 97,
        authSeq: 101,
        resName: 'ASP',
        atoms: [
          ['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'],
          ['CB', 'C'], ['CG', 'C'], ['OD1', 'O'], ['OD2', 'O'],
        ],
      },
      {
        labelSeq: 251,
        authSeq: 255,
        resName: 'HIS',
        atoms: [
          ['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'],
          ['CB', 'C'], ['CG', 'C'], ['ND1', 'N'], ['CD2', 'C'], ['CE1', 'C'], ['NE2', 'N'],
        ],
      },
      {
        labelSeq: 266,
        authSeq: 270,
        resName: 'ARG',
        atoms: [
          ['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'],
          ['CB', 'C'], ['CG', 'C'], ['CD', 'C'], ['NE', 'N'], ['CZ', 'C'], ['NH1', 'N'], ['NH2', 'N'],
        ],
      },
    ]);
    const os7Cif = [
      ...CIF_HEADER,
      ...os7Res.rows,
      `HETATM ${os7Res.nextSerial} FE FE . FE2 D 2 . ? 1.000 1.000 1.000 1.00 10.00 301 FE2 A FE 1`,
      `HETATM ${os7Res.nextSerial + 1} C C1 . AKG E 3 . ? 2.500 1.000 1.000 1.00 10.00 302 AKG A C1 1`,
      `HETATM ${os7Res.nextSerial + 2} S S1 . TAU F 4 . ? 4.000 1.000 1.000 1.00 10.00 303 TAU A S1 1`,
      '#',
    ].join('\n');

    const bench = getBenchmarkById('ame_1os7_taud')!;
    expect(bench.biology).toMatch(/Arg270/);
    expect(bench.biology).toMatch(/taurine \(TAU\)/);
    expect(bench.biology).toMatch(/tip-atom/i);
    expect(bench.conditionParams.motifAtoms).toBe(
      'A99:NE2,CD2,CE1; A101:OD1,CG,CB,OD2; A255:NE2,CD2,CE1,CG,ND1; A270:NH1,CZ',
    );

    const ctx = extractRfd4TemplateContext(os7Cif);
    const compiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ ...bench.conditionParams, contig: '24-24' }),
      ctx,
    );
    expect(compiled.contig).toBe('24-24,A95,A97,A251,A266,/0,D301,/0,E302,/0,F303');
    expect((compiled.conditions.C_IDX as { false: Array<{ group: string }> }).false.map((e) => e.group)).toEqual([
      'motif_0',
      'motif_1',
      'motif_2',
      'motif_3',
    ]);

    const os7Atoms = cifMod.parseCifToAtomArray(os7Cif, { fixArginines: true });
    prepareMod.assignIonCharges(os7Atoms);
    const annotated = specMod.buildAtomArrayFromConditionSpec(
      { contig: compiled.contig, conditions: compiled.conditions },
      { inputAtomArray: os7Atoms },
    );
    const crdGrp: number[] = annotated.annotations.condition_CoordinateGroup_1_atom;
    const proteinTipHits: string[] = [];
    for (let i = 0; i < annotated.nAtoms; i++) {
      if (annotated.is_polymer[i] && crdGrp[i] === 1) {
        proteinTipHits.push(`${annotated.res_name[i]}:${annotated.atom_name[i]}`);
      }
    }
    expect(proteinTipHits).toHaveLength(14);

    const isIndexed: boolean[] = annotated.annotations.condition_index_1_residue;
    const idxGrp: number[] = annotated.annotations['condition_index-group_1_token'];
    const unidxGroups = new Set<number>();
    for (let i = 0; i < annotated.nAtoms; i++) {
      if (isIndexed[i] === false) unidxGroups.add(idxGrp[i]);
    }
    expect(unidxGroups.size).toBe(4);

    const built = rfd4.buildBatch(
      { contig: compiled.contig, conditions: compiled.conditions },
      { inputAtomArray: os7Atoms, seed: 13 },
    );
    expect(built.n).toBe(31);
  });

  it('extractRfd4DesignMetadata locks all unindexed catalytic residues and emits motifIndices + nativeMotifAtomCoords on tip-atom insertions', () => {
    // 6-residue scaffold where residues 1 (HIS), 2 (GLU), 4 (CYS), 5 (CYS) are unindexed tip-atom insertions.
    // On unindexed insertions, batch.condition_sequence_1_residue[scaffoldTid] is 0, and CA has coordGroup=-1
    // on residues 1, 2, 4 (only tip atoms carry coordGroup=1 + intCrd), while residue 5 (Cys132) includes CA.
    const parsed = {
      nResidues: 6,
      chainIds: ['A', 'A', 'A', 'A', 'A', 'A'],
      resNums: [1, 2, 3, 4, 5, 6],
    };
    const mkRes = (
      resId: number,
      tokenId: number,
      unidx: boolean,
      tipSpecs: Array<{ atomName: string; coord: [number, number, number] }>,
    ) => {
      const tipByName = new Map(tipSpecs.map((t) => [t.atomName, t.coord]));
      const baseAtoms = ['N', 'CA', 'C', 'O'];
      const names = [...new Set([...baseAtoms, ...tipSpecs.map((t) => t.atomName)])];
      return names.map((atomName) => {
        const tipCoord = tipByName.get(atomName);
        return {
          isHetero: false,
          chainId: 'A',
          resId,
          tokenId,
          atomName,
          element: atomName[0],
          isUnindexedInsertion: unidx,
          coordGroup: tipCoord ? 1 : -1,
          inCoord: null,
          intCrd: tipCoord ?? null,
          crd: null,
        };
      });
    };

    const structureAtoms = [
      ...mkRes(1, 0, false, []),
      ...mkRes(2, 1, true, [
        { atomName: 'ND1', coord: [1, 2, 3] },
        { atomName: 'CG', coord: [1.5, 2.5, 3.5] },
        { atomName: 'CE1', coord: [2, 2.2, 3.1] },
      ]),
      ...mkRes(3, 2, true, [
        { atomName: 'OE1', coord: [4, 5, 6] },
        { atomName: 'CD', coord: [4.4, 5.1, 6.2] },
      ]),
      ...mkRes(4, 3, false, []),
      ...mkRes(5, 4, true, [
        { atomName: 'SG', coord: [7, 8, 9] },
        { atomName: 'CB', coord: [7.2, 8.3, 9.1] },
      ]),
      ...mkRes(6, 5, true, [
        { atomName: 'SG', coord: [10, 11, 12] },
        { atomName: 'CB', coord: [10.3, 11.2, 12.1] },
        { atomName: 'CA', coord: [10.8, 11.5, 12.4] },
      ]),
      // Non-protein hetero ligand atoms should be ignored for protein residue indexing
      { isHetero: true, chainId: 'B', resId: 296, tokenId: 6, atomName: 'ZN', element: 'ZN', coordGroup: 1, inCoord: [5, 5, 5] },
    ];

    // Scaffold tokens have C_SEQ=0 (unconditioned sequence) — unindexed insertion flag must still lock them!
    const batch = { condition_sequence_1_residue: { data: new Int32Array([0, 0, 0, 0, 0, 0, 0]) } };

    const meta = extractRfd4DesignMetadata(parsed, structureAtoms, batch);
    expect(meta.fixedResidueIndices).toEqual([2, 3, 5, 6]); // 1-based!
    expect(meta.motifIndices).toEqual([1, 2, 4, 5]); // 0-based!
    expect(meta.nativeMotifAtomCoords).toHaveLength(10);
    expect(meta.nativeMotifAtomCoords?.map((a) => `${a.resIdx}:${a.atomName}`)).toEqual([
      '1:ND1', '1:CG', '1:CE1',
      '2:OE1', '2:CD',
      '4:SG', '4:CB',
      '5:CA', '5:SG', '5:CB',
    ]);
    // Because CA was only conditioned on 1 of 4 catalytic residues, nativeMotifCoords is omitted to prevent index misalignment
    expect(meta.nativeMotifCoords).toBeUndefined();
  });

  it('compiles 7C7M (sm_7c7m_sam) with multi-role polar-only C_HB (DONOR + ACCEPTOR, ZERO carbons tagged) via vendored RFD4', async () => {
    const { cifMod, specMod } = await loadVendoredRfd4();

    const samBench = getBenchmarkById('sm_7c7m_sam')!;
    expect(samBench.conditionParams.hbondMode).toBeUndefined();
    expect(samBench.conditionParams.hbondSelection).toBeUndefined();
    expect(samBench.conditionParams.hbondDonorSelection).toBe("A/SAM/*/[N,N6,O2',O3']");
    expect(samBench.conditionParams.hbondAcceptorSelection).toBe("A/SAM/*/[O,OXT,O4',N1,N3,N7]");

    // All 27 heavy atoms of SAM (S-adenosylmethionine): 15 carbons + SD + N9 + 4 donors + 6 acceptors
    const samAtomSpecs: Array<[string, string]> = [
      ['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'], ['OXT', 'O'],
      ['CB', 'C'], ['CG', 'C'], ['SD', 'S'], ['CE', 'C'],
      ["C5'", 'C'], ["C4'", 'C'], ["O4'", 'O'], ["C3'", 'C'], ["O3'", 'O'],
      ["C2'", 'C'], ["O2'", 'O'], ["C1'", 'C'],
      ['N9', 'N'], ['C8', 'C'], ['N7', 'N'], ['C5', 'C'], ['C6', 'C'],
      ['N6', 'N'], ['N1', 'N'], ['C2', 'C'], ['N3', 'N'], ['C4', 'C'],
    ];
    const samRows = samAtomSpecs.map(([nm, el], idx) => {
      const qNm = nm.includes("'") ? `"${nm}"` : nm;
      return `HETATM ${idx + 1} ${el} ${qNm} . SAM D 2 . ? ${(idx * 0.5).toFixed(3)} 1.000 2.000 1.00 10.00 301 SAM A ${qNm} 1`;
    });
    const samCif = [...CIF_HEADER, ...samRows, '#'].join('\n');

    const ctx = extractRfd4TemplateContext(samCif);
    const compiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ ...samBench.conditionParams, contig: '20-20' }),
      ctx,
    );
    expect(compiled.contig).toBe('20-20,/0,D301');
    expect(compiled.conditions.C_CRD).toEqual({ true: [{ select: "sel('D/*/*/*')" }] });
    expect(compiled.conditions.C_HB).toEqual({
      DONOR: [{ select: 'sel("D/SAM/*/[N,N6,O2\',O3\']")' }],
      ACCEPTOR: [{ select: 'sel("D/SAM/*/[O,OXT,O4\',N1,N3,N7]")' }],
    });

    const samAtoms = cifMod.parseCifToAtomArray(samCif, { fixArginines: true });
    expect(() =>
      assertRfd4ConditionSelectorsMatch(samAtoms, compiled.contig, compiled.conditions, specMod),
    ).not.toThrow();

    const annotated = specMod.buildAtomArrayFromConditionSpec(
      { contig: compiled.contig, conditions: compiled.conditions },
      { inputAtomArray: samAtoms },
    );
    const hbArr: number[] = annotated.annotations.condition_hbond_1_atom;
    const donors: string[] = [];
    const acceptors: string[] = [];
    const taggedCarbons: string[] = [];
    const taggedSulfurs: string[] = [];
    for (let i = 0; i < annotated.nAtoms; i++) {
      if (annotated.res_name[i] !== 'SAM') continue;
      if (hbArr[i] === specMod.CONDITION_ENUMS.HBondEnum.DONOR) donors.push(annotated.atom_name[i]);
      if (hbArr[i] === specMod.CONDITION_ENUMS.HBondEnum.ACCEPTOR) acceptors.push(annotated.atom_name[i]);
      if (hbArr[i] !== 0 && annotated.element[i] === 'C') taggedCarbons.push(annotated.atom_name[i]);
      if (hbArr[i] !== 0 && annotated.element[i] === 'S') taggedSulfurs.push(annotated.atom_name[i]);
    }
    expect(donors.sort()).toEqual(['N', 'N6', "O2'", "O3'"].sort());
    expect(acceptors.sort()).toEqual(['N1', 'N3', 'N7', 'O', "O4'", 'OXT'].sort());
    expect(taggedCarbons).toEqual([]);
    expect(taggedSulfurs).toEqual([]);

    // Backward-compatible legacy hbondMode + hbondSelection still works
    const legacy = compileConditionToRfd4(
      buildCanonicalConditionIntent({
        contig: '50-50',
        hbondMode: 'ACCEPTOR',
        hbondSelection: 'A/SAM/*/[O,N1]',
      }),
      ctx,
    );
    expect(legacy.conditions.C_HB).toEqual({
      ACCEPTOR: [{ select: "sel('D/SAM/*/[O,N1]')" }],
    });
  });

  it('supports motifAtoms shorthands (BKBN / ALL / TIP) and biochemical C_RAS / C_PISTACK selections', () => {
    expect(parseMotifAtomsSpec('BKBN')).toEqual({ shorthand: 'BKBN', entries: [] });
    expect(parseMotifAtomsSpec('ALL')).toEqual({ shorthand: 'ALL', entries: [] });
    expect(parseMotifAtomsSpec('TIP')).toEqual({ shorthand: 'TIP', entries: [] });

    const bkbn = compileConditionToRfd4(
      buildCanonicalConditionIntent({
        contig: '10-10,A16-20,10-10',
        motifSpan: 'A16-20',
        motifAtoms: 'BKBN',
        rasaBuriedSelection: 'A/SAM/*/*',
        piStackSelection: 'A/SAM/*/[C4,C5,C6,C8,N7,N9]',
      }),
    );
    expect(bkbn.conditions.C_CRDGRP).toEqual({
      1: [{ select: "sel('A/*/[16-20]/[N,CA,C,O]')" }],
    });
    expect(bkbn.conditions.C_SEQ).toEqual({
      true: [{ select: "sel('A/*/[16-20]/*')" }],
    });
    expect(bkbn.conditions.C_RAS).toEqual({
      BURIED: [{ select: "sel('A/SAM/*/*')" }],
    });
    expect(bkbn.conditions.C_PISTACK).toEqual({
      true: [{ select: "sel('A/SAM/*/[C4,C5,C6,C8,N7,N9]')" }],
    });

    // resolveRfd4CompiledInput forwards motifAtoms + multi-role H-bond params
    const rfdResolved = resolveRfd4CompiledInput({
      mode: 'conditioned',
      length: 110,
      contig: '110-110',
      hbondDonorSelection: "A/SAM/*/[N,N6,O2',O3']",
      hbondAcceptorSelection: "A/SAM/*/[O,OXT,O4',N1,N3,N7]",
    });
    expect(rfdResolved.conditions.C_HB).toEqual({
      DONOR: [{ select: 'sel("A/SAM/*/[N,N6,O2\',O3\']")' }],
      ACCEPTOR: [{ select: 'sel("A/SAM/*/[O,OXT,O4\',N1,N3,N7]")' }],
    });
  });

  it('aligns 1BCF, 4XOJ, 5YUI, 5TPN, and 7M5W benchmarks with published paper specs and exact token lengths', () => {
    const bcf = getBenchmarkById('rfd_1bcf_diiron')!;
    expect(bcf.conditionParams.contig).toBe(
      '12-12,A92-99,24-24,A123-130,24-24,A47-54,23-23,A18-25,10-10',
    );
    expect(parseLengthFromContig(bcf.conditionParams.contig)).toBe(125);
    expect(bcf.approxTokens).toBe(125);
    expect(bcf.cropSegments).toBe('A18-25, A47-54, A92-99, A123-130');
    expect(bcf.conditionParams.motifSpan).toBe('A18-25, A47-54, A92-99, A123-130');

    const xoj = getBenchmarkById('g3_4xoj_trypsin')!;
    expect(xoj.conditionParams.contig).toBe('15-15,A55-55,45-45,A99-99,70-70,A190-192,15-15');
    expect(parseLengthFromContig(xoj.conditionParams.contig)).toBe(150);
    expect(xoj.approxTokens).toBe(150);

    const yui = getBenchmarkById('g3_5yui_ca2')!;
    expect(yui.conditionParams.contig).toBe('14-14,A93-97,20-20,A118-120,20-20,A198-200,10-10');
    expect(parseLengthFromContig(yui.conditionParams.contig)).toBe(75);
    expect(yui.approxTokens).toBe(75);
    expect(yui.biology).toMatch(/Thr199/);

    const tpn = getBenchmarkById('rfd_5tpn_rsvf_v')!;
    expect(tpn.conditionParams.contig).toBe('25-25,A163-181,31-31');
    expect(parseLengthFromContig(tpn.conditionParams.contig)).toBe(75);
    expect(tpn.approxTokens).toBe(75);
    expect(tpn.sourceFamily).toBe('RFdiffusion 25 / Genie 3 RSV-F');

    const dna = getBenchmarkById('rfd4_7m5w_bdna')!;
    expect(dna.biology).toContain('18-bp dsDNA duplex (containing 8-bp TGAATGAA Capicua core)');
  });

  it('fixes benchmark-sync reset, parseLengthFromContig edge cases, Genie 2 multi-island motifRange, and length-slider contig sync', () => {
    // 1. Reset clears customSpecYaml, cathCode, motifAtoms, and multi-role H-bond fields
    const resetRfd = buildResetToBenchmarkParams('rfd4-proteina', {
      mode: 'motif',
      contig: '180-180',
      motifSpan: 'A102, A104',
      motifAtoms: 'A102:ND1,CG,CE1',
      hbondDonorSelection: "A/SAM/*/[N,N6,O2',O3']",
      hbondAcceptorSelection: "A/SAM/*/[O,OXT,O4',N1,N3,N7]",
      cathCode: '3.40.50',
      customSpecYaml: 'contig: 90-90\n',
      userEditedTask: true,
      autoSeededBenchmarkId: 'ame_1ctt_cda',
    });
    expect(resetRfd.customSpecYaml).toBe('');
    expect(resetRfd.cathCode).toBe('');
    expect(resetRfd.motifAtoms).toBe('');
    expect(resetRfd.hbondDonorSelection).toBe('');
    expect(resetRfd.hbondAcceptorSelection).toBe('');
    expect(resetRfd.userEditedTask).toBeUndefined();
    expect(resetRfd.autoSeededBenchmarkId).toBeUndefined();

    // 2. parseLengthFromContig supports single-residue tokens, negative residue numbers, and RFD4 chain suffixes
    expect(parseLengthFromContig('15-15,A55,45-45,A99,70-70,A190-192,15-15')).toBe(150);
    expect(parseLengthFromContig('10-10,A-5--2,10-10')).toBe(24);
    expect(parseLengthFromContig('180-180P')).toBe(180);
    expect(parseLengthFromContig('A17-132,/0,70-70P')).toBe(70);

    // 3. Genie 2 receives full multi-island motifRange + passes crossValidate; RFD4 clears stale cathCode and adapts contig on length edit
    const graph: WorkflowGraph = {
      id: 'sync-multi',
      name: 'Sync Multi',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '1PRW', benchmarkId: 'rfd_1prw_efhand' },
        },
        {
          id: 'g2',
          adapterId: 'genie2',
          params: { ...defaultParamsForAdapter('genie2') },
        },
        {
          id: 'rfd4',
          adapterId: 'rfd4-proteina',
          params: {
            ...defaultParamsForAdapter('rfd4-proteina'),
            autoSeededBenchmarkId: 'pdl1_5o45',
            cathCode: '3.40.50',
            customSpecYaml: 'stale: true',
          },
        },
        {
          id: 'rfdiff',
          adapterId: 'rfdiffusion',
          params: { ...defaultParamsForAdapter('rfdiffusion') },
        },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'g2', toPort: 'template' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd4', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfdiff', toPort: 'structure' },
      ],
    };

    const seeded = reconcileBenchmarkSeeding(graph);
    const g2Node = seeded.graph.nodes.find((n) => n.id === 'g2')!;
    const rfd4Node = seeded.graph.nodes.find((n) => n.id === 'rfd4')!;
    const rfdiffNode = seeded.graph.nodes.find((n) => n.id === 'rfdiff')!;

    expect(g2Node.params.motifRange).toBe('A16-35, A52-71');
    expect(ADAPTER_REGISTRY.genie2.crossValidate!(g2Node.params)).toEqual([]);
    expect(rfd4Node.params.cathCode).toBe('');
    expect(rfd4Node.params.customSpecYaml).toBe('');
    expect(rfd4Node.params.contig).toBe('10-10,A16-35,16-16,A52-71,10-10');
    expect(rfdiffNode.params.contig).toBe('10-10/A16-35/16-16/A52-71/10-10');

    // Changing length slider on seeded RFD4 & RFdiffusion adapts contig cleanly (and stays idempotent)
    const lengthEdited: WorkflowGraph = {
      ...seeded.graph,
      nodes: seeded.graph.nodes.map((n) =>
        n.id === 'rfd4' || n.id === 'rfdiff'
          ? { ...n, params: { ...n.params, length: 100, userEditedLength: true } }
          : n,
      ),
    };
    const adapted = reconcileBenchmarkSeeding(lengthEdited);
    expect(adapted.changedNodeIds.sort()).toEqual(['rfd4', 'rfdiff']);
    const rfd4Adapted = adapted.graph.nodes.find((n) => n.id === 'rfd4')!;
    const rfdiffAdapted = adapted.graph.nodes.find((n) => n.id === 'rfdiff')!;
    expect(parseLengthFromContig(String(rfd4Adapted.params.contig))).toBe(100);
    expect(String(rfd4Adapted.params.contig)).toContain('A16-35');
    expect(String(rfd4Adapted.params.contig)).toContain('A52-71');
    expect(parseLengthFromContig(String(rfdiffAdapted.params.contig).replace(/\//g, ','))).toBe(100);
    // Idempotent on second reconcile
    expect(reconcileBenchmarkSeeding(adapted.graph).changedNodeIds).toEqual([]);

    // Switching to 1CTT seeds motifAtoms; switching to 7C7M clears motifAtoms & seeds polar donor/acceptor
    const toCtt = switchBenchmark(seeded.graph, 'pdb', 'ame_1ctt_cda');
    const rfdCtt = toCtt.graph.nodes.find((n) => n.id === 'rfd4')!;
    expect(rfdCtt.params.motifAtoms).toBe(
      'A102:ND1,CG,CE1; A104:OE1,CD; A129:SG,CB; A132:SG,CB,CA',
    );
    const toSam = switchBenchmark(toCtt.graph, 'pdb', 'sm_7c7m_sam');
    const rfdSam = toSam.graph.nodes.find((n) => n.id === 'rfd4')!;
    expect(rfdSam.params.motifAtoms).toBe('');
    expect(rfdSam.params.hbondDonorSelection).toBe("A/SAM/*/[N,N6,O2',O3']");
    expect(rfdSam.params.hbondAcceptorSelection).toBe("A/SAM/*/[O,OXT,O4',N1,N3,N7]");
  });

  it('preserves non-empty parsed.pdbText for mmCIF non-protein crops (7C7M SAM & 7M5W B-DNA) in pdb-input', async () => {
    const samCif = [
      ...CIF_HEADER,
      'HETATM 1 N N . SAM D 2 . ? 1.000 2.000 3.000 1.00 10.00 301 SAM A N 1',
      'HETATM 2 O O . SAM D 2 . ? 2.200 2.000 3.000 1.00 10.00 301 SAM A O 1',
      '#',
    ].join('\n');

    const art = await materializePdbInputNode({
      id: 'pdb_sam',
      adapterId: 'pdb-input',
      params: {
        mode: 'custom',
        pdbText: samCif,
        filename: '7C7M.cif',
        cropSegments: 'A:SAM',
        benchmarkId: 'sm_7c7m_sam',
      },
    });
    expect(art.length).toBe(0);
    expect(art.cifText).toContain('_atom_site.');
    expect(art.pdbText.trim().length).toBeGreaterThan(0);
    expect(art.pdbText).toContain('HETATM');
    expect(art.pdbText).toContain('SAM');
  });
});
