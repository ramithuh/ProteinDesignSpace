import { describe, it, expect, beforeAll } from 'vitest';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parsePdbBackbone, parseStructureText } from '$lib/protein-ops/pdb';
import { ExecutionScheduler } from '$lib/engine/scheduler';
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

  // BUG DOCUMENTED (do not fix app code in test-builder):
  // scheduler.ts:333 & 351 (rfd4-conditioner) emit unbracketed `sel('A/*/2,4/*')`,
  // which upstream spec.js parses as residue 2 only (silently dropping residue 4)!
  // Upstream spec.js requires bracketed `sel('A/*/[2,4]/*')` or `sel('A/*/2/*') | sel('A/*/4/*')`.
  // Change `it.fails` -> `it` once scheduler.ts:333 & 351 are fixed.
  it.fails('regression: rfd4-conditioner multi-hotspot output selects all listed residues in upstream buildBatch', async () => {
    const tplCif = buildMinimalTemplateCif(6);
    const scheduler = new ExecutionScheduler();
    const res = await scheduler.run(
      {
        id: 'hot-e2e',
        name: 'Hotspot E2E',
        nodes: [
          {
            id: 'c',
            adapterId: 'rfd4-conditioner',
            params: { taskPreset: 'custom', contig: 'A1-6,/0,10-10', hotspots: 'A2, A4' },
          },
        ],
        edges: [],
      },
      () => {},
    );
    const cond = res.outputsByNode.c.condition as ConditionSpecArtifact;
    const built = rfd4.buildBatch(
      { contig: cond.contig, conditions: cond.conditions },
      { inputCif: tplCif, seed: 11 },
    );
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
});
