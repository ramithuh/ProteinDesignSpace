import { describe, it, expect, beforeAll } from 'vitest';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parsePdbBackbone, parseStructureText } from '$lib/protein-ops/pdb';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import {
  assertRfd4ConditionSelectorsMatch,
  buildCanonicalConditionIntent,
  compileConditionToRfd4,
  extractRfd4DesignMetadata,
  extractRfd4TemplateContext,
  pdbTextToSplitCifText,
} from '$lib/engine/conditioning';
import { getBenchmarkById } from '$lib/engine/benchmarks';
import type { ConditionSpecArtifact } from '$lib/engine/types';

interface Rfd4UpstreamModule {
  TokenType: { PROTEIN: number; RNA: number; DNA: number; ATOMIZED: number };
  UNIFIED_ATOM37_TOKENS: readonly string[];
  UNIFIED_ATOM37_ENCODING: {
    n_tokens: number;
    token_to_idx: Record<string, number>;
    token_slot_map: Record<string, Record<string, number>>;
  };
  UNIFIED_RESTYPE_ATOM37_MASK: readonly Uint8Array[];
  getRfd4pPresetById: (id: string) => {
    id: string;
    contig: string;
    conditions: Record<string, unknown>;
    cifUrl: string | null;
  };
  buildBatch: (
    cifTextOrSpec: string | Uint8Array | Record<string, unknown>,
    opts?: Record<string, unknown>,
  ) => {
    batch: Record<string, { data: ArrayLike<number>; shape: number[]; dtype: string }>;
    n: number;
    atoms: Record<string, unknown>;
  };
  sequenceForDecoder: (
    batch: Record<string, unknown>,
    n: number,
    seqLogits: Float32Array,
    nTypes?: number,
  ) => Int32Array;
  formatSampleLocalLatents: (
    batch: Record<string, unknown>,
    args: {
      n: number;
      caNm: Float32Array;
      decoded: {
        coorsNm: Float32Array;
        residueType: Int32Array;
        emitMask: Uint8Array;
        atomMask: Uint8Array;
      };
    },
  ) => { coors: Float32Array; atomMask: Uint8Array; residueType: Int32Array };
  processSampleToAtoms: (args: {
    coors: Float32Array;
    atomMask: Uint8Array;
    residueType: Int32Array;
    paddingMask?: ArrayLike<number> | null;
    inputAtoms?: Record<string, unknown> | null;
    batch?: Record<string, unknown> | null;
    inputFrame?: boolean;
  }) => { atoms: Array<Record<string, unknown>> };
  toPdbString: (atomsOrResult: unknown, opts?: Record<string, unknown>) => string;
  toCifString: (atomsOrResult: unknown, opts?: Record<string, unknown>) => string;
}

let rfd4: Rfd4UpstreamModule;

beforeAll(async () => {
  const vendorPath = resolve(process.cwd(), 'static/vendor/rfd4-proteina/webgpu/index.js');
  rfd4 = (await import(pathToFileURL(vendorPath).href)) as Rfd4UpstreamModule;
});

function buildMinimalTemplateCif(nRes: number): string {
  const resNames = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE'];
  const lines = [
    'data_tpl',
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
  let serial = 1;
  for (let i = 0; i < nRes; i++) {
    const rn = resNames[i % resNames.length];
    const seqId = i + 1;
    const x0 = i * 3.8;
    const atoms: Array<[string, string, number, number, number]> = [
      ['N', 'N', x0 - 1.2, 0.5, 0.0],
      ['CA', 'C', x0, 0.0, 0.0],
      ['C', 'C', x0 + 1.25, 0.5, 0.0],
      ['O', 'O', x0 + 1.25, 1.73, 0.0],
    ];
    for (const [nm, elem, x, y, z] of atoms) {
      lines.push(
        `ATOM ${serial++} ${elem} ${nm} . ${rn} A 1 ${seqId} ? ${x.toFixed(3)} ${y.toFixed(3)} ${z.toFixed(3)} 1.00 0.00 ${seqId} ${rn} A ${nm} 1`,
      );
    }
  }
  lines.push('#');
  return lines.join('\n') + '\n';
}

describe('RFD4-Proteina upstream integration regressions', () => {
  it('buildBatch builds deterministic unconditional monomer batch (contig 70-70)', () => {
    const built1 = rfd4.buildBatch({ contig: '70-70', conditions: {} }, { seed: 42 });
    const built2 = rfd4.buildBatch({ contig: '70-70', conditions: {} }, { seed: 42 });

    expect(built1.n).toBe(70);
    expect(built2.n).toBe(70);

    const pad = built1.batch.padding_mask.data;
    const tokType = built1.batch.token_type.data;
    const condSeq = built1.batch.condition_sequence_1_residue.data;
    const supLat = built1.batch.supports_local_latent.data;
    const inCtx = built1.batch.is_in_context_conditioning.data;

    for (let i = 0; i < 70; i++) {
      expect(pad[i]).toBe(1);
      expect(tokType[i]).toBe(rfd4.TokenType.PROTEIN);
      expect(condSeq[i]).toBe(0); // <M> masked
      expect(supLat[i]).toBe(1);
      expect(inCtx[i]).toBe(0);
    }
    expect(Array.from(condSeq)).toEqual(Array.from(built2.batch.condition_sequence_1_residue.data));
  });

  it('buildBatch handles CATH 3.40.50 + explicit alpha/beta secondary structure preset', () => {
    const preset = rfd4.getRfd4pPresetById('custom_cath_monomer');
    expect(preset.id).toBe('custom_cath_monomer');
    expect(preset.cifUrl).toBeNull();

    const built = rfd4.buildBatch(
      { contig: preset.contig, conditions: preset.conditions },
      { seed: 7 },
    );

    // 16 + 4 + 8 + 4 + 14 + 8 = 54 residues
    expect(built.n).toBe(54);
    expect(built.batch.padding_mask.data.length).toBeGreaterThanOrEqual(54);

    // Check CATH & secondary-structure conditioning tensors are populated (non-all-zero)
    const condKeys = Object.keys(built.batch).filter((k) => k.startsWith('condition_'));
    const hasCath = condKeys.some((k) => k.toLowerCase().includes('cath'));
    const hasSec = condKeys.some((k) => k.toLowerCase().includes('secondary'));
    expect(hasCath).toBe(true);
    expect(hasSec).toBe(true);

    const secKey = condKeys.find((k) => k.toLowerCase().includes('secondary'))!;
    const secData = Array.from(built.batch[secKey].data);
    // Should contain at least 3 distinct SS classes (HELIX, LOOP, STRAND)
    expect(new Set(secData).size).toBeGreaterThanOrEqual(3);
  });

  it('buildBatch parses condition_spec DSL with template CIF (hotspots, motif, locked sequence)', () => {
    const tplCif = buildMinimalTemplateCif(6);
    const built = rfd4.buildBatch(
      {
        contig: 'A1-6,/0,10-10',
        conditions: {
          C_CRD: { true: [{ select: "sel('A/*/*/*')" }] },
          C_SEQ: { true: [{ select: "sel('A/*/*/*')" }] },
          C_HOT: { true: [{ select: "sel('A/*/[2,4]/*')" }] },
        },
      },
      { inputCif: tplCif, seed: 11 },
    );

    expect(built.n).toBe(16); // 6 template + 10 generated
    const condSeq = Array.from(built.batch.condition_sequence_1_residue.data);
    // First 6 tokens should have locked template residue types (> 0), remaining 10 masked (0)
    const lockedCount = condSeq.filter((v) => v > 0).length;
    const maskedCount = condSeq.filter((v) => v === 0).length;
    expect(lockedCount).toBe(6);
    expect(maskedCount).toBe(10);

    // Hotspot tensor [1, n, 37] should mark both A2 and A4 (2 residues, 4 atoms each)
    const hotKey = Object.keys(built.batch).find((k) => k.toLowerCase().includes('hotspot'))!;
    expect(hotKey).toBeDefined();
    const hotData = built.batch[hotKey].data;
    const hotResidues: number[] = [];
    for (let t = 0; t < built.n; t++) {
      let hasHot = false;
      for (let a = 0; a < 37; a++) {
        if (hotData[t * 37 + a] > 0) {
          hasHot = true;
          break;
        }
      }
      if (hasHot) hotResidues.push(t);
    }
    expect(hotResidues).toHaveLength(2);

    // Also verify YAML condition_spec string parsing through buildBatch
    const yamlSpec = [
      'contig: "12-12"',
      'conditions:',
      '  C_CATH:',
      '    "3.40.50":',
      '      - select: "sel(\'*/*/*/*\')"',
    ].join('\n');
    const builtYaml = rfd4.buildBatch(yamlSpec, { seed: 3 });
    expect(builtYaml.n).toBe(12);
  });

  it('regression: multi-hotspot, lockChain+motif C_SEQ merge, and unindexed C_IDX integrate cleanly with upstream buildBatch', async () => {
    const tplCif = buildMinimalTemplateCif(6);
    const { buildCanonicalConditionIntent, compileConditionToRfd4 } = await import('$lib/engine/conditioning');
    const intent = buildCanonicalConditionIntent({
      contig: 'A1-6,/0,10-10',
      lockChain: 'A',
      hotspots: 'A2, A4',
      motifSpan: 'A1-2, A5-6',
      unindexedMotif: true,
    });
    const compiled = compileConditionToRfd4(intent);
    const built = rfd4.buildBatch(
      { contig: compiled.contig, conditions: compiled.conditions },
      { inputCif: tplCif, seed: 11 },
    );
    // Both A2 and A4 must be marked in condition_hotspot_1_atom
    const hotData = built.batch.condition_hotspot_1_atom.data;
    const hotResidues: number[] = [];
    for (let t = 0; t < built.n; t++) {
      for (let a = 0; a < 37; a++) {
        if (hotData[t * 37 + a] > 0) {
          hotResidues.push(t);
          break;
        }
      }
    }
    expect(hotResidues).toHaveLength(2);
    // Unindexed motif tokens (A1-2, A5-6 = 4 residues) must be flagged in is_in_context_conditioning
    const ctxCount = Array.from(built.batch.is_in_context_conditioning.data).filter((v) => v > 0).length;
    expect(ctxCount).toBe(4);
  });

  it('sequenceForDecoder: valid logits vs all-zero logits (poly-ALA=1 collapse) vs NaN logits', () => {
    const n = 20;
    const nTypes = rfd4.UNIFIED_ATOM37_ENCODING.n_tokens; // 147
    expect(nTypes).toBe(147);
    expect(rfd4.UNIFIED_ATOM37_TOKENS[0]).toBe('<M>');
    expect(rfd4.UNIFIED_ATOM37_TOKENS[1]).toBe('ALA');

    const built = rfd4.buildBatch({ contig: `${n}-${n}`, conditions: {} }, { seed: 1 });

    // 1. Valid logits peaking across all 20 canonical AAs (indices 1..20), even with huge disallowed RNA/NCAA logits
    const validLogits = new Float32Array(n * nTypes).fill(-5.0);
    for (let i = 0; i < n; i++) {
      validLogits[i * nTypes + (i + 1)] = 10.0; // target canonical AA (1..20)
      validLogits[i * nTypes + 0] = 999.0; // disallowed <M>
      validLogits[i * nTypes + 22] = 999.0; // disallowed RNA 'A'
      validLogits[i * nTypes + 40] = 999.0; // disallowed NCAA
    }
    const seqValid = rfd4.sequenceForDecoder(built.batch, n, validLogits, nTypes);
    expect(Array.from(seqValid)).toEqual(Array.from({ length: n }, (_, i) => i + 1));

    // 2. All-zero logits (e.g. silent WebGPU validation failure): collapses EVERY token to ALA (index 1)!
    // Root cause: r=0 (<M>) is masked to -3.4e38, r=1 (ALA) is the first allowed protein class with logit 0 > -3.4e38,
    // and strict `v > bestV` never beats 0 on ties for r=2..21.
    const zeroLogits = new Float32Array(n * nTypes);
    const seqZero = rfd4.sequenceForDecoder(built.batch, n, zeroLogits, nTypes);
    expect(Array.from(seqZero)).toEqual(new Array(n).fill(1));
    expect(Array.from(seqZero).map((rt) => rfd4.UNIFIED_ATOM37_TOKENS[rt])).toEqual(
      new Array(n).fill('ALA'),
    );

    // 3. NaN logits: `NaN > -Infinity` is false for all allowed residues while disallowed r=0 (<M>) gets -3.4e38 > -Infinity,
    // so every unconditioned token collapses to 0 (<M>)!
    const nanLogits = new Float32Array(n * nTypes).fill(Number.NaN);
    const seqNan = rfd4.sequenceForDecoder(built.batch, n, nanLogits, nTypes);
    expect(Array.from(seqNan)).toEqual(new Array(n).fill(0));
    expect(Array.from(seqNan).map((rt) => rfd4.UNIFIED_ATOM37_TOKENS[rt])).toEqual(
      new Array(n).fill('<M>'),
    );

    // 4. Pre-conditioned sequence residues (C_SEQ !== 0) are preserved regardless of logits
    const lockedBatch = {
      ...built.batch,
      condition_sequence_1_residue: {
        ...built.batch.condition_sequence_1_residue,
        data: Int32Array.from(built.batch.condition_sequence_1_residue.data, (v, idx) =>
          idx === 0 ? 18 : idx === 5 ? 19 : v, // TRP(18) at 0, TYR(19) at 5
        ),
      },
    };
    const seqWithLock = rfd4.sequenceForDecoder(lockedBatch, n, zeroLogits, nTypes);
    expect(seqWithLock[0]).toBe(18);
    expect(seqWithLock[5]).toBe(19);
    expect(seqWithLock[1]).toBe(1); // unconditioned still falls to ALA=1 under zero logits
  });

  it('formatSampleLocalLatents -> processSampleToAtoms -> toPdbString -> parsePdbBackbone exact 20-AA round-trip', () => {
    const n = 20;
    // Indices 1..20 in UNIFIED_ATOM37_TOKENS are the 20 canonical AAs in alphabetical 3-letter order:
    // ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
    const expectedSeq = 'ARNDCQEGHILKMFPSTWYV';
    const built = rfd4.buildBatch({ contig: `${n}-${n}`, conditions: {} }, { seed: 99 });

    const caNm = new Float32Array(n * 3);
    const coorsNm = new Float32Array(n * 37 * 3);
    const atomMask = new Uint8Array(n * 37);
    const residueType = new Int32Array(n);
    const emitMask = new Uint8Array(n).fill(1);

    for (let i = 0; i < n; i++) {
      const rt = i + 1; // 1..20
      residueType[i] = rt;
      const legal = rfd4.UNIFIED_RESTYPE_ATOM37_MASK[rt];
      const caX = i * 0.38; // 3.8 A in nm
      const caY = (i % 2 === 0 ? 0.1 : -0.1);
      const caZ = 0.05 * Math.sin(i);
      caNm[i * 3 + 0] = caX;
      caNm[i * 3 + 1] = caY;
      caNm[i * 3 + 2] = caZ;

      for (let slot = 0; slot < 37; slot++) {
        if (!legal[slot]) continue;
        atomMask[i * 37 + slot] = 1;
        const base = (i * 37 + slot) * 3;
        // Distinct deterministic offsets in nm around CA for N(0), CA(1), C(2), CB(3), O(4), sidechains
        if (slot === 0) {
          coorsNm[base + 0] = caX - 0.12;
          coorsNm[base + 1] = caY + 0.05;
          coorsNm[base + 2] = caZ;
        } else if (slot === 1) {
          coorsNm[base + 0] = caX;
          coorsNm[base + 1] = caY;
          coorsNm[base + 2] = caZ;
        } else if (slot === 2) {
          coorsNm[base + 0] = caX + 0.125;
          coorsNm[base + 1] = caY + 0.05;
          coorsNm[base + 2] = caZ;
        } else if (slot === 4) {
          coorsNm[base + 0] = caX + 0.125;
          coorsNm[base + 1] = caY + 0.173;
          coorsNm[base + 2] = caZ;
        } else {
          coorsNm[base + 0] = caX + 0.01 * slot;
          coorsNm[base + 1] = caY - 0.15;
          coorsNm[base + 2] = caZ + 0.02 * slot;
        }
      }
    }

    const formatted = rfd4.formatSampleLocalLatents(built.batch, {
      n,
      caNm,
      decoded: { coorsNm, residueType, emitMask, atomMask },
    });

    // Verify nm -> Angstrom (*10) conversion on CA slot 1
    for (let i = 0; i < n; i++) {
      const caBase = (i * 37 + 1) * 3;
      expect(formatted.coors[caBase + 0]).toBeCloseTo(caNm[i * 3 + 0] * 10, 4);
      expect(formatted.coors[caBase + 1]).toBeCloseTo(caNm[i * 3 + 1] * 10, 4);
      expect(formatted.coors[caBase + 2]).toBeCloseTo(caNm[i * 3 + 2] * 10, 4);
    }

    const structure = rfd4.processSampleToAtoms({
      coors: formatted.coors,
      atomMask: formatted.atomMask,
      residueType: formatted.residueType,
      paddingMask: built.batch.padding_mask.data,
      inputAtoms: built.atoms,
      batch: built.batch,
    });
    expect(structure.atoms.length).toBeGreaterThan(140); // all heavy atoms of 20 distinct AAs

    // PDB round-trip
    const pdbText = rfd4.toPdbString(structure);
    const parsedPdb = parsePdbBackbone(pdbText);
    expect(parsedPdb.nResidues).toBe(20);
    expect(parsedPdb.sequence).toBe(expectedSeq);

    for (let i = 0; i < n; i++) {
      expect(parsedPdb.caCoords[i * 3 + 0]).toBeCloseTo(caNm[i * 3 + 0] * 10, 2);
      expect(parsedPdb.caCoords[i * 3 + 1]).toBeCloseTo(caNm[i * 3 + 1] * 10, 2);
      expect(parsedPdb.caCoords[i * 3 + 2]).toBeCloseTo(caNm[i * 3 + 2] * 10, 2);
    }

    // mmCIF round-trip via parseStructureText
    const cifText = rfd4.toCifString(structure);
    const parsedCif = parseStructureText(cifText);
    expect(parsedCif.format).toBe('mmcif');
    expect(parsedCif.nResidues).toBe(20);
    expect(parsedCif.sequence).toBe(expectedSeq);
  });

  it('AME enzymes (1CTT/1OS7), shifted CIFs (5TPN/5O45/1YCR), SAM (7C7M), B-DNA (7M5W), and holo PDBs build valid conditioned batches with >0 matched atoms', async () => {
    const cifMod = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/rfd4-proteina/shared/rfd4p/pipeline/cif.js')).href
    );
    const prepareMod = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/rfd4-proteina/shared/rfd4p/pipeline/prepare.js')).href
    );
    const specMod = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/rfd4-proteina/shared/rfd4p/pipeline/spec.js')).href
    );

    const cifHeader = [
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
    const mkBackboneRows = (
      startSerial: number,
      labelChain: string,
      authChain: string,
      entityId: number,
      entries: Array<{ labelSeq: number; authSeq: number; resName: string }>,
    ) => {
      const rows: string[] = [];
      let s = startSerial;
      for (let i = 0; i < entries.length; i++) {
        const { labelSeq, authSeq, resName } = entries[i];
        const x0 = i * 3.8;
        for (const [nm, el, dx, dy, dz] of [
          ['N', 'N', -1.2, 0.5, 0],
          ['CA', 'C', 0, 0, 0],
          ['C', 'C', 1.25, 0.5, 0],
          ['O', 'O', 1.25, 1.73, 0],
        ] as const) {
          rows.push(
            `ATOM ${s++} ${el} ${nm} . ${resName} ${labelChain} ${entityId} ${labelSeq} ? ${(x0 + dx).toFixed(3)} ${dy.toFixed(3)} ${dz.toFixed(3)} 1.00 10.00 ${authSeq} ${resName} ${authChain} ${nm} 1`,
          );
        }
      }
      return { rows, nextSerial: s };
    };

    // 1. 1CTT AME enzyme (shifted auth_seq_id 102,104,129,132 -> label_seq_id 98,100,125,128 + ZN + DHZ)
    const cttRes = mkBackboneRows(1, 'A', 'A', 1, [
      { labelSeq: 98, authSeq: 102, resName: 'HIS' },
      { labelSeq: 100, authSeq: 104, resName: 'GLU' },
      { labelSeq: 125, authSeq: 129, resName: 'CYS' },
      { labelSeq: 128, authSeq: 132, resName: 'CYS' },
    ]);
    const cttCif = [
      ...cifHeader,
      ...cttRes.rows,
      `HETATM ${cttRes.nextSerial} ZN ZN . ZN B 2 . ? 2.000 2.000 2.000 1.00 10.00 296 ZN A ZN 1`,
      `HETATM ${cttRes.nextSerial + 1} C C1 . DHZ C 3 . ? 3.500 2.000 2.000 1.00 10.00 295 DHZ A C1 1`,
      `HETATM ${cttRes.nextSerial + 2} O O1 . DHZ C 3 . ? 4.700 2.000 2.000 1.00 10.00 295 DHZ A O1 1`,
      '#',
    ].join('\n');

    const cttAtoms = cifMod.parseCifToAtomArray(cttCif, { fixArginines: true });
    prepareMod.assignIonCharges(cttAtoms);
    const znIdx = cttAtoms.res_name.indexOf('ZN');
    expect(cttAtoms.charge[znIdx]).toBe(2);

    const cttBench = getBenchmarkById('ame_1ctt_cda')!;
    const cttCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ ...cttBench.conditionParams, contig: '24-24' }),
      extractRfd4TemplateContext(cttCif),
    );
    expect(() =>
      assertRfd4ConditionSelectorsMatch(cttAtoms, cttCompiled.contig, cttCompiled.conditions, specMod),
    ).not.toThrow();
    const cttBuilt = rfd4.buildBatch(
      { contig: cttCompiled.contig, conditions: cttCompiled.conditions },
      { inputAtomArray: cttAtoms, seed: 42 },
    );
    // 24 scaffold + 4 unindexed catalytic residues + 1 ZN atom + 2 DHZ atoms = 31 tokens
    expect(cttBuilt.n).toBe(31);
    const inCtxCount = Array.from(cttBuilt.batch.is_in_context_conditioning.data).filter((v) => v > 0).length;
    expect(inCtxCount).toBe(4);
    const atomizedCount = Array.from(cttBuilt.batch.token_type.data).filter(
      (v) => v === rfd4.TokenType.ATOMIZED,
    ).length;
    expect(atomizedCount).toBe(3);

    // 2. Shifted-numbering binder (5O45 style: auth A17-22 -> label A1-6, hotspot A20 -> label A4)
    const pdl1Res = mkBackboneRows(1, 'A', 'A', 1, [
      { labelSeq: 1, authSeq: 17, resName: 'ALA' },
      { labelSeq: 2, authSeq: 18, resName: 'VAL' },
      { labelSeq: 3, authSeq: 19, resName: 'LEU' },
      { labelSeq: 4, authSeq: 20, resName: 'ILE' },
      { labelSeq: 5, authSeq: 21, resName: 'PHE' },
      { labelSeq: 6, authSeq: 22, resName: 'TRP' },
    ]);
    const pdl1Cif = [...cifHeader, ...pdl1Res.rows, '#'].join('\n');
    const pdl1Atoms = cifMod.parseCifToAtomArray(pdl1Cif, { fixArginines: true });
    const pdl1Compiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: 'A17-22,/0,10-10', lockChain: 'A', hotspots: 'A20' }),
      extractRfd4TemplateContext(pdl1Cif),
    );
    expect(pdl1Compiled.contig).toBe('A1-6,/0,10-10');
    assertRfd4ConditionSelectorsMatch(pdl1Atoms, pdl1Compiled.contig, pdl1Compiled.conditions, specMod);
    const pdl1Built = rfd4.buildBatch(
      { contig: pdl1Compiled.contig, conditions: pdl1Compiled.conditions },
      { inputAtomArray: pdl1Atoms, seed: 5 },
    );
    expect(pdl1Built.n).toBe(16);
    expect(Array.from(pdl1Built.batch.condition_sequence_1_residue.data).filter((v) => v > 0)).toHaveLength(6);

    // 2b. 1OS7 AME enzyme (FE2 + AKG + TAU ligands on distinct label_asym_id, shifted auth_seq_id)
    const os7Res = mkBackboneRows(1, 'A', 'A', 1, [
      { labelSeq: 95, authSeq: 99, resName: 'HIS' },
      { labelSeq: 97, authSeq: 101, resName: 'ASP' },
      { labelSeq: 251, authSeq: 255, resName: 'HIS' },
      { labelSeq: 266, authSeq: 270, resName: 'ARG' },
    ]);
    const os7Cif = [
      ...cifHeader,
      ...os7Res.rows,
      `HETATM ${os7Res.nextSerial} FE FE . FE2 D 2 . ? 1.000 1.000 1.000 1.00 10.00 301 FE2 A FE 1`,
      `HETATM ${os7Res.nextSerial + 1} C C1 . AKG E 3 . ? 2.500 1.000 1.000 1.00 10.00 302 AKG A C1 1`,
      `HETATM ${os7Res.nextSerial + 2} S S1 . TAU F 4 . ? 4.000 1.000 1.000 1.00 10.00 303 TAU A S1 1`,
      '#',
    ].join('\n');
    const os7Atoms = cifMod.parseCifToAtomArray(os7Cif, { fixArginines: true });
    prepareMod.assignIonCharges(os7Atoms);
    expect(os7Atoms.charge[os7Atoms.res_name.indexOf('FE2')]).toBe(2);
    const os7Bench = getBenchmarkById('ame_1os7_taud')!;
    const os7Compiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ ...os7Bench.conditionParams, contig: '24-24' }),
      extractRfd4TemplateContext(os7Cif),
    );
    expect(os7Compiled.contig).toBe('24-24,A95,A97,A251,A266,/0,D301,/0,E302,/0,F303');
    assertRfd4ConditionSelectorsMatch(os7Atoms, os7Compiled.contig, os7Compiled.conditions, specMod);
    const os7Built = rfd4.buildBatch(
      { contig: os7Compiled.contig, conditions: os7Compiled.conditions },
      { inputAtomArray: os7Atoms, seed: 13 },
    );
    expect(os7Built.n).toBe(31); // 24 scaffold + 4 unindexed + 3 ligand atoms

    // 2c. 5TPN & 1YCR shifted-numbering motifs (auth A163-166 -> label A140-143; auth B19-22 -> label B3-6)
    const tpnRes = mkBackboneRows(1, 'A', 'A', 1, [
      { labelSeq: 140, authSeq: 163, resName: 'SER' },
      { labelSeq: 141, authSeq: 164, resName: 'GLU' },
      { labelSeq: 142, authSeq: 165, resName: 'LEU' },
      { labelSeq: 143, authSeq: 166, resName: 'ASN' },
    ]);
    const tpnCif = [...cifHeader, ...tpnRes.rows, '#'].join('\n');
    const tpnAtoms = cifMod.parseCifToAtomArray(tpnCif, { fixArginines: true });
    const tpnCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: '8-8,A163-166,8-8', motifSpan: 'A163-166' }),
      extractRfd4TemplateContext(tpnCif),
    );
    expect(tpnCompiled.contig).toBe('8-8,A140-143,8-8');
    assertRfd4ConditionSelectorsMatch(tpnAtoms, tpnCompiled.contig, tpnCompiled.conditions, specMod);
    expect(
      rfd4.buildBatch({ contig: tpnCompiled.contig, conditions: tpnCompiled.conditions }, { inputAtomArray: tpnAtoms, seed: 2 }).n,
    ).toBe(20);

    const ycrRes = mkBackboneRows(1, 'B', 'B', 1, [
      { labelSeq: 3, authSeq: 19, resName: 'PHE' },
      { labelSeq: 4, authSeq: 20, resName: 'SER' },
      { labelSeq: 5, authSeq: 21, resName: 'ASP' },
      { labelSeq: 6, authSeq: 22, resName: 'LEU' },
    ]);
    const ycrCif = [...cifHeader, ...ycrRes.rows, '#'].join('\n');
    const ycrAtoms = cifMod.parseCifToAtomArray(ycrCif, { fixArginines: true });
    const ycrCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: '8-8,B19-22,8-8', motifSpan: 'B19-22' }),
      extractRfd4TemplateContext(ycrCif),
    );
    expect(ycrCompiled.contig).toBe('8-8,B3-6,8-8');
    assertRfd4ConditionSelectorsMatch(ycrAtoms, ycrCompiled.contig, ycrCompiled.conditions, specMod);
    expect(
      rfd4.buildBatch({ contig: ycrCompiled.contig, conditions: ycrCompiled.conditions }, { inputAtomArray: ycrAtoms, seed: 3 }).n,
    ).toBe(20);

    // 3. 7M5W wired B-DNA conditioning
    const dnaCif = [
      ...cifHeader,
      'ATOM 1 P P . DA B 1 1 ? 0.0 0.0 0.0 1.0 10.0 101 DA B P 1',
      "ATOM 2 C C4' . DA B 1 1 ? 1.5 0.0 0.0 1.0 10.0 101 DA B C4' 1",
      'ATOM 3 P P . DT C 2 1 ? 0.0 5.0 0.0 1.0 10.0 201 DT C P 1',
      "ATOM 4 C C4' . DT C 2 1 ? 1.5 5.0 0.0 1.0 10.0 201 DT C C4' 1",
      '#',
    ].join('\n');
    const dnaAtoms = cifMod.parseCifToAtomArray(dnaCif, { fixArginines: true });
    const dnaCompiled = compileConditionToRfd4(
      buildCanonicalConditionIntent({ contig: '12-12' }),
      extractRfd4TemplateContext(dnaCif),
    );
    expect(dnaCompiled.contig).toBe('12-12,/0,B1-1,/0,C1-1');
    assertRfd4ConditionSelectorsMatch(dnaAtoms, dnaCompiled.contig, dnaCompiled.conditions, specMod);
    const dnaBuilt = rfd4.buildBatch(
      { contig: dnaCompiled.contig, conditions: dnaCompiled.conditions },
      { inputAtomArray: dnaAtoms, seed: 9 },
    );
    expect(dnaBuilt.n).toBe(14);
    expect(Array.from(dnaBuilt.batch.token_type.data).filter((v) => v === rfd4.TokenType.DNA)).toHaveLength(2);

    // 4. Custom holo .pdb with HETATM ZN + HETATM SAM + HETATM MSE
    const holoPdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N',
      'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C',
      'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 10.00           C',
      'ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 10.00           O',
      'HETATM    5  N   MSE A   2       3.320   1.540   0.000  1.00 10.00           N',
      'HETATM    6  CA  MSE A   2       4.000   2.830   0.000  1.00 10.00           C',
      'HETATM    7  SE  MSE A   2       5.500   3.500   0.000  1.00 10.00          SE',
      'HETATM    8 ZN    ZN A 296       8.000   0.000   0.000  1.00 10.00          ZN',
      'HETATM    9  C1  SAM A 301      10.000   0.000   0.000  1.00 10.00           C',
      'END',
    ].join('\n');
    const splitCif = pdbTextToSplitCifText(holoPdb);
    const holoAtoms = cifMod.parseCifToAtomArray(splitCif, { fixArginines: true });
    prepareMod.convertMseToMet(holoAtoms);
    prepareMod.assignIonCharges(holoAtoms);
    expect(holoAtoms.res_name.includes('MSE')).toBe(false);
    expect(holoAtoms.res_name.includes('MET')).toBe(true);
    expect(holoAtoms.atom_name.includes('SD')).toBe(true);
    const znAtomIdx = holoAtoms.res_name.indexOf('ZN');
    const samAtomIdx = holoAtoms.res_name.indexOf('SAM');
    expect(holoAtoms.is_polymer[znAtomIdx]).toBe(false);
    expect(holoAtoms.charge[znAtomIdx]).toBe(2);
    expect(holoAtoms.is_polymer[samAtomIdx]).toBe(false);
    expect(holoAtoms.chain_id[znAtomIdx]).not.toBe(holoAtoms.chain_id[0]);
    expect(holoAtoms.chain_id[samAtomIdx]).not.toBe(holoAtoms.chain_id[znAtomIdx]);

    // 5. Selector assertion catches 0-atom match instead of silently falling back to unconditional
    expect(() =>
      assertRfd4ConditionSelectorsMatch(
        pdl1Atoms,
        '10-10',
        { C_HOT: { true: [{ select: "sel('A/*/[20]/*')" }] } },
        specMod,
      ),
    ).toThrow(/matched 0 atoms/i);

    // 6. extractRfd4DesignMetadata identifies lockedChains & chainRoles for binder complex
    const meta = extractRfd4DesignMetadata(
      {
        nResidues: 16,
        chainIds: [...Array(6).fill('A'), ...Array(10).fill('B')],
        resNums: [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      },
      Array.from({ length: 16 }, (_, i) => ({
        isHetero: false,
        atomName: 'CA',
        chainId: i < 6 ? 'A' : 'B',
        tokenId: i,
        coordGroup: i < 6 ? 0 : -1,
        crd: i < 6 ? [i * 3.8, 0, 0] : null,
      })),
      pdl1Built.batch,
    );
    expect(meta.lockedChains).toEqual(['A']);
    expect(meta.chainRoles).toEqual({ A: 'target', B: 'binder' });
  });
});
