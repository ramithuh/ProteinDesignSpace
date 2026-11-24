import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  AF2_ATOM37_NAMES,
  AF3_DENSE_INTERNAL_ATOMS,
  GAP_AATYPE,
  LocalFoldTemplateError,
  MODIFIED_PARENTS,
  UNK_AATYPE,
  alignQueryToTemplate,
  assertInlineTemplateIntegrity,
  buildAf2TemplatePayload,
  buildAf3TemplateSlots,
  detectTemplateFormat,
  getLocalFoldTemplateDialectPolicy,
  isLocalFoldTemplateCapable,
  localFoldTemplateFamily,
  parseTemplateStructure,
  resolveLocalFoldTemplateSource,
  suggestTemplateCapableModel,
} from '$lib/engine/localfold-template';
import { resolveLocalFoldEffectiveConfig } from '$lib/engine/conditioning';
import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS } from '$lib/engine/registry';
import {
  defaultParamsForAdapter,
  insertBlockSmart,
  sanitizeParamsOnModeSwitch,
  validateGraph,
} from '$lib/engine/graph-utils';
import {
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from '$lib/engine/bio-capabilities';
import { computeUploadDigest } from '$lib/engine/param-validation';
import type { CanonicalArtifact, SequenceSet, StructureArtifact, WorkflowGraph } from '$lib/engine/types';

const VENDOR_ROOT = resolve(__dirname, '../../../static/vendor/localfold');
const importVendor = (relPath: string) =>
  import(pathToFileURL(resolve(VENDOR_ROOT, relPath)).href);

// ---------------------------------------------------------------------------
// Synthetic PDB / mmCIF Fixtures
// ---------------------------------------------------------------------------

function makePdbLine(opts: {
  record?: 'ATOM  ' | 'HETATM';
  serial: number;
  atomName: string;
  altLoc?: string;
  resName: string;
  chainId?: string;
  resSeq: number;
  insCode?: string;
  x: number;
  y: number;
  z: number;
  occ?: number;
  b?: number;
  element?: string;
}): string {
  const rec = opts.record ?? 'ATOM  ';
  const serial = String(opts.serial).padStart(5, ' ');
  const rawName = opts.atomName.trim();
  const atomField =
    rawName.length === 4 ? rawName : ` ${rawName.padEnd(3, ' ')}`;
  const alt = (opts.altLoc ?? ' ')[0];
  const resName = opts.resName.padStart(3, ' ');
  const chain = (opts.chainId ?? 'A')[0];
  const resSeq = String(opts.resSeq).padStart(4, ' ');
  const ins = (opts.insCode ?? ' ')[0];
  const x = opts.x.toFixed(3).padStart(8, ' ');
  const y = opts.y.toFixed(3).padStart(8, ' ');
  const z = opts.z.toFixed(3).padStart(8, ' ');
  const occ = (opts.occ ?? 1.0).toFixed(2).padStart(6, ' ');
  const b = (opts.b ?? 50.0).toFixed(2).padStart(6, ' ');
  const el = (opts.element ?? rawName[0] ?? 'C').padStart(2, ' ');
  return `${rec}${serial} ${atomField}${alt}${resName} ${chain}${resSeq}${ins}   ${x}${y}${z}${occ}${b}          ${el}`;
}

const THREE_LETTER: Record<string, string> = {
  A: 'ALA', R: 'ARG', N: 'ASN', D: 'ASP', C: 'CYS',
  Q: 'GLN', E: 'GLU', G: 'GLY', H: 'HIS', I: 'ILE',
  L: 'LEU', K: 'LYS', M: 'MET', F: 'PHE', P: 'PRO',
  S: 'SER', T: 'THR', W: 'TRP', Y: 'TYR', V: 'VAL',
};

function makeBackbonePdb(
  chains: Array<{
    chainId: string;
    sequence: string;
    startResSeq?: number;
    bFactor?: number;
    bFactors?: number[];
  }>,
): string {
  const lines: string[] = ['HEADER    SYNTHETIC TEST STRUCTURE'];
  let serial = 1;
  let chainOffsetZ = 0;
  for (const ch of chains) {
    const start = ch.startResSeq ?? 1;
    for (let i = 0; i < ch.sequence.length; i += 1) {
      const code = ch.sequence[i];
      const resName = THREE_LETTER[code] ?? 'ALA';
      const b = ch.bFactors?.[i] ?? ch.bFactor ?? 85.0;
      const x0 = i * 3.8;
      const y0 = Math.sin(i * 0.5) * 2.0;
      const z0 = chainOffsetZ;
      lines.push(makePdbLine({ serial: serial++, atomName: 'N', resName, chainId: ch.chainId, resSeq: start + i, x: x0 - 0.5, y: y0, z: z0, b, element: 'N' }));
      lines.push(makePdbLine({ serial: serial++, atomName: 'CA', resName, chainId: ch.chainId, resSeq: start + i, x: x0, y: y0, z: z0, b, element: 'C' }));
      lines.push(makePdbLine({ serial: serial++, atomName: 'C', resName, chainId: ch.chainId, resSeq: start + i, x: x0 + 1.2, y: y0 + 0.3, z: z0, b, element: 'C' }));
      lines.push(makePdbLine({ serial: serial++, atomName: 'O', resName, chainId: ch.chainId, resSeq: start + i, x: x0 + 1.5, y: y0 + 1.4, z: z0, b, element: 'O' }));
      if (code !== 'G') {
        lines.push(makePdbLine({ serial: serial++, atomName: 'CB', resName, chainId: ch.chainId, resSeq: start + i, x: x0, y: y0 - 1.5, z: z0 + 0.5, b, element: 'C' }));
      }
    }
    lines.push('TER');
    chainOffsetZ += 12.0;
  }
  lines.push('END');
  return lines.join('\n');
}

function makeBackboneMmcif(
  chains: Array<{ chainId: string; sequence: string; startResSeq?: number; bFactor?: number }>,
): string {
  const lines: string[] = [
    'data_test_template',
    'loop_',
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.type_symbol',
    '_atom_site.label_atom_id',
    '_atom_site.label_alt_id',
    '_atom_site.label_comp_id',
    '_atom_site.label_asym_id',
    '_atom_site.auth_asym_id',
    '_atom_site.label_seq_id',
    '_atom_site.auth_seq_id',
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.pdbx_PDB_model_num',
  ];
  let id = 1;
  let zOff = 0;
  for (const ch of chains) {
    const start = ch.startResSeq ?? 1;
    const b = (ch.bFactor ?? 88.0).toFixed(2);
    for (let i = 0; i < ch.sequence.length; i += 1) {
      const code = ch.sequence[i];
      const resName = THREE_LETTER[code] ?? 'ALA';
      const seq = start + i;
      const x0 = i * 3.8;
      const y0 = Math.cos(i * 0.4) * 1.8;
      const atoms: Array<[string, string, number, number, number]> = [
        ['N', 'N', x0 - 0.5, y0, zOff],
        ['CA', 'C', x0, y0, zOff],
        ['C', 'C', x0 + 1.2, y0 + 0.3, zOff],
        ['O', 'O', x0 + 1.5, y0 + 1.4, zOff],
      ];
      if (code !== 'G') atoms.push(['CB', 'C', x0, y0 - 1.5, zOff + 0.4]);
      for (const [atom, el, x, y, z] of atoms) {
        lines.push(
          `ATOM ${id++} ${el} ${atom} . ${resName} ${ch.chainId} ${ch.chainId} ${seq} ${seq} ? ${x.toFixed(3)} ${y.toFixed(3)} ${z.toFixed(3)} 1.00 ${b} 1`,
        );
      }
    }
    zOff += 10.0;
  }
  return lines.join('\n');
}

function makeStructureArtifact(opts: { pdbText?: string; cifText?: string }): StructureArtifact {
  return {
    kind: 'StructureArtifact',
    id: 'struct_tpl_1',
    label: 'Template Structure',
    sequence: 'MKTAYIAK',
    length: 8,
    caCoords: new Float32Array(24),
    backbone4Coords: new Float32Array(96),
    pdbText: opts.pdbText ?? '',
    cifText: opts.cifText,
    sourceModelId: 'pdb-input',
    geometry: {
      length: 8,
      rg: 10,
      meanCaDist: 3.8,
      chainBreaks: 0,
      clashes: 0,
      helixPct: 50,
      sheetPct: 0,
      coilPct: 50,
    },
  };
}

// ---------------------------------------------------------------------------
// 1. Model Capability Taxonomy & Dialect Policy
// ---------------------------------------------------------------------------

describe('LocalFold Template Support (M1)', () => {
  it('classifies all 20 LocalFold models accurately and matches vendored AF3 dialect templatePolicyFor', async () => {
    const [dialectMod, manifestsMod] = await Promise.all([
      importVendor('shared/af3/dialect.js'),
      importVendor('shared/bundles/manifests/index.js'),
    ]);

    expect(LOCALFOLD_MODEL_OPTIONS).toHaveLength(20);

    const expectedFamily: Record<string, ReturnType<typeof localFoldTemplateFamily>> = {
      monomer: 'af2-monomer',
      'monomer-2': 'af2-monomer',
      'monomer-3': 'none',
      'monomer-4': 'none',
      'monomer-5': 'none',
      multimer: 'af2-multimer',
      'multimer-2': 'af2-multimer',
      'multimer-3': 'af2-multimer',
      'multimer-4': 'af2-multimer',
      'multimer-5': 'af2-multimer',
      rosettafold3: 'af3',
      boltz2: 'af3',
      openbind0: 'af3',
      protenix2: 'af3',
      opendde: 'af3',
      mol1: 'af3',
      intellifold2: 'af3',
      af3: 'af3',
      'ef2-fast-300m': 'none',
      'ef2-fast-600m': 'none',
    };

    for (const opt of LOCALFOLD_MODEL_OPTIONS) {
      expect(localFoldTemplateFamily(opt.id)).toBe(expectedFamily[opt.id]);
      expect(isLocalFoldTemplateCapable(opt.id)).toBe(expectedFamily[opt.id] !== 'none');
    }
    expect(isLocalFoldTemplateCapable('unknown-model')).toBe(false);
    expect(localFoldTemplateFamily('unknown-model')).toBe('none');

    // Verify AF3 dialect policies against vendored templatePolicyFor(dialectFor(...))
    for (const opt of LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3')) {
      const bundle = manifestsMod.MODEL_BUNDLES[opt.id];
      const dialect = dialectMod.dialectFor(bundle.model);
      const vendorPolicy = dialectMod.templatePolicyFor(dialect);
      const tsPolicy = getLocalFoldTemplateDialectPolicy(opt.id);

      expect(tsPolicy.modified ?? undefined).toBe(vendorPolicy.modified);
      expect(tsPolicy.altlocFirst).toBe(vendorPolicy.altloc === 'first');
      expect(tsPolicy.gapWhereNoAtoms).toBe(Boolean(dialect.templateGapWhereNoAtoms));
    }

    expect(suggestTemplateCapableModel('monomer-3')).toEqual({ modelId: 'monomer', label: 'AF2 Model 1' });
    expect(suggestTemplateCapableModel('monomer-4', { isMultimer: true })).toEqual({
      modelId: 'multimer',
      label: 'AF2-Multimer 1',
    });
    expect(suggestTemplateCapableModel('ef2-fast-300m', { hasHolo: true })).toEqual({
      modelId: 'boltz2',
      label: 'Boltz-2',
    });
  });

  // -------------------------------------------------------------------------
  // 2. Source Resolution
  // -------------------------------------------------------------------------

  it('resolves template sources with port > inline precedence, cifText > pdbText, and FIPS digest parity', () => {
    expect(resolveLocalFoldTemplateSource({ params: {} })).toBeNull();

    const pdb = makeBackbonePdb([{ chainId: 'A', sequence: 'MKTAYIAKQR' }]);
    const cif = makeBackboneMmcif([{ chainId: 'A', sequence: 'MKTAYIAKQR' }]);

    const inlineOnly = resolveLocalFoldTemplateSource({
      params: { templatePdbText: pdb, templateFilename: 'inline.pdb', templateSpanChains: false },
    });
    expect(inlineOnly).not.toBeNull();
    expect(inlineOnly!.sourceKind).toBe('inline');
    expect(inlineOnly!.format).toBe('pdb');
    expect(inlineOnly!.filename).toBe('inline.pdb');
    expect(inlineOnly!.spanChains).toBe(false);
    expect(inlineOnly!.bothProvided).toBe(false);
    expect(inlineOnly!.digest).toBe(computeUploadDigest(pdb));

    const both = resolveLocalFoldTemplateSource({
      templateInput: makeStructureArtifact({ pdbText: pdb, cifText: cif }),
      params: { templatePdbText: pdb, templateFilename: 'ignored.pdb' },
    });
    expect(both!.sourceKind).toBe('port');
    expect(both!.format).toBe('mmcif');
    expect(both!.text).toBe(cif);
    expect(both!.bothProvided).toBe(true);
    expect(both!.spanChains).toBe(true);
    expect(both!.warnings.length).toBeGreaterThan(0);
    expect(both!.digest).toBe(computeUploadDigest(cif));

    expect(() =>
      resolveLocalFoldTemplateSource({
        templateInput: makeStructureArtifact({ pdbText: '   ', cifText: '' }),
      }),
    ).toThrowError(LocalFoldTemplateError);
  });

  // -------------------------------------------------------------------------
  // 3. Vendored Table Parity & PDB / mmCIF All-Atom Parser Edge Cases
  // -------------------------------------------------------------------------

  it('matches vendored MODIFIED_PARENTS, ATOM37, GAP_AATYPE, and REFERENCE_CONFORMERS bit-for-bit', async () => {
    const [modParentsMod, featMod, inputMod, confMod] = await Promise.all([
      importVendor('shared/af3/featurise/modified-parents.js'),
      importVendor('shared/af3/featurise/template-features.js'),
      importVendor('shared/af3/featurise/template-input.js'),
      importVendor('shared/af3/featurise/reference-conformers.js'),
    ]);

    expect(GAP_AATYPE).toBe(inputMod.GAP_AATYPE);
    expect(AF2_ATOM37_NAMES).toEqual(featMod.ATOM37);
    expect(MODIFIED_PARENTS).toEqual(modParentsMod.MODIFIED_PARENTS);

    for (const code of [...'ARNDCQEGHILKMFPSTWYVX']) {
      const vendorAtomOrder = confMod.conformerFor(code, false).map((entry: [number, string]) => entry[1]);
      expect(AF3_DENSE_INTERNAL_ATOMS[code]).toEqual(vendorAtomOrder);
    }
  });

  it('parses PDB and mmCIF all-atom structures honoring ENDMDL, multi-char chains, altLocs, MSE/SEP, ARG swap, and exclusions', async () => {
    const templateInputMod = await importVendor('shared/af3/featurise/template-input.js');

    // Multi-model PDB with ENDMDL, altLoc occupancy vs first, MSE SE->SD, SEP modified AA, ARG NH1/NH2 swap, waters, caps, hydrogens, DNA
    const complexPdb = [
      'MODEL        1',
      makePdbLine({ record: 'HETATM', serial: 1, atomName: 'N', resName: 'ACE', chainId: 'A', resSeq: 0, x: 0, y: 0, z: 0 }),
      // Residue 1: ALA with altLoc A (occ 0.40) and B (occ 0.60)
      makePdbLine({ serial: 2, atomName: 'N', altLoc: 'A', resName: 'ALA', chainId: 'A', resSeq: 1, x: 1, y: 0, z: 0, occ: 0.4, b: 91 }),
      makePdbLine({ serial: 3, atomName: 'N', altLoc: 'B', resName: 'ALA', chainId: 'A', resSeq: 1, x: 9, y: 0, z: 0, occ: 0.6, b: 92 }),
      makePdbLine({ serial: 4, atomName: 'CA', resName: 'ALA', chainId: 'A', resSeq: 1, x: 2, y: 0, z: 0, b: 92 }),
      makePdbLine({ serial: 5, atomName: 'C', resName: 'ALA', chainId: 'A', resSeq: 1, x: 3, y: 0, z: 0, b: 92 }),
      makePdbLine({ serial: 6, atomName: 'O', resName: 'ALA', chainId: 'A', resSeq: 1, x: 3, y: 1, z: 0, b: 92 }),
      makePdbLine({ serial: 7, atomName: 'CB', resName: 'ALA', chainId: 'A', resSeq: 1, x: 2, y: -1.5, z: 0, b: 92 }),
      makePdbLine({ serial: 8, atomName: 'HA', resName: 'ALA', chainId: 'A', resSeq: 1, x: 2, y: 0.9, z: 0, element: 'H' }),
      // Residue 2: MSE (HETATM selenomethionine, SE -> SD)
      makePdbLine({ record: 'HETATM', serial: 9, atomName: 'N', resName: 'MSE', chainId: 'A', resSeq: 2, x: 5, y: 0, z: 0, b: 80, element: 'N' }),
      makePdbLine({ record: 'HETATM', serial: 10, atomName: 'CA', resName: 'MSE', chainId: 'A', resSeq: 2, x: 6, y: 0, z: 0, b: 80, element: 'C' }),
      makePdbLine({ record: 'HETATM', serial: 11, atomName: 'C', resName: 'MSE', chainId: 'A', resSeq: 2, x: 7, y: 0, z: 0, b: 80, element: 'C' }),
      makePdbLine({ record: 'HETATM', serial: 12, atomName: 'CB', resName: 'MSE', chainId: 'A', resSeq: 2, x: 6, y: -1.5, z: 0, b: 80, element: 'C' }),
      makePdbLine({ record: 'HETATM', serial: 13, atomName: 'SE', resName: 'MSE', chainId: 'A', resSeq: 2, x: 6, y: -3.5, z: 0, b: 80, element: 'SE' }),
      // Residue 3: SEP (phosphoserine, modified parent SER)
      makePdbLine({ record: 'HETATM', serial: 14, atomName: 'N', resName: 'SEP', chainId: 'A', resSeq: 3, x: 9, y: 0, z: 0, b: 75, element: 'N' }),
      makePdbLine({ record: 'HETATM', serial: 15, atomName: 'CA', resName: 'SEP', chainId: 'A', resSeq: 3, x: 10, y: 0, z: 0, b: 75, element: 'C' }),
      makePdbLine({ record: 'HETATM', serial: 16, atomName: 'C', resName: 'SEP', chainId: 'A', resSeq: 3, x: 11, y: 0, z: 0, b: 75, element: 'C' }),
      makePdbLine({ record: 'HETATM', serial: 17, atomName: 'CB', resName: 'SEP', chainId: 'A', resSeq: 3, x: 10, y: -1.5, z: 0, b: 75, element: 'C' }),
      // Residue 4: ARG with NH1 farther from CD than NH2 (should swap when arginines=true)
      makePdbLine({ serial: 18, atomName: 'N', resName: 'ARG', chainId: 'A', resSeq: 4, x: 13, y: 0, z: 0, b: 70, element: 'N' }),
      makePdbLine({ serial: 19, atomName: 'CA', resName: 'ARG', chainId: 'A', resSeq: 4, x: 14, y: 0, z: 0, b: 70, element: 'C' }),
      makePdbLine({ serial: 20, atomName: 'C', resName: 'ARG', chainId: 'A', resSeq: 4, x: 15, y: 0, z: 0, b: 70, element: 'C' }),
      makePdbLine({ serial: 21, atomName: 'CD', resName: 'ARG', chainId: 'A', resSeq: 4, x: 14, y: -3, z: 0, b: 70, element: 'C' }),
      makePdbLine({ serial: 22, atomName: 'NH1', resName: 'ARG', chainId: 'A', resSeq: 4, x: 14, y: -8, z: 0, b: 70, element: 'N' }),
      makePdbLine({ serial: 23, atomName: 'NH2', resName: 'ARG', chainId: 'A', resSeq: 4, x: 14, y: -5, z: 0, b: 70, element: 'N' }),
      // Excluded: Water, Ligand, Cap, DNA
      makePdbLine({ record: 'HETATM', serial: 24, atomName: 'O', resName: 'HOH', chainId: 'A', resSeq: 101, x: 20, y: 0, z: 0, element: 'O' }),
      makePdbLine({ record: 'HETATM', serial: 25, atomName: 'ZN', resName: 'ZN', chainId: 'A', resSeq: 102, x: 21, y: 0, z: 0, element: 'ZN' }),
      makePdbLine({ record: 'HETATM', serial: 26, atomName: 'N', resName: 'NME', chainId: 'A', resSeq: 5, x: 16, y: 0, z: 0, element: 'N' }),
      makePdbLine({ serial: 27, atomName: "C1'", resName: 'DA', chainId: 'B', resSeq: 1, x: 30, y: 0, z: 0, element: 'C' }),
      'ENDMDL',
      'MODEL        2',
      makePdbLine({ serial: 28, atomName: 'CA', resName: 'GLY', chainId: 'A', resSeq: 99, x: 99, y: 99, z: 99 }),
      'ENDMDL',
    ].join('\n');

    const occParsed = parseTemplateStructure(complexPdb, { altlocFirst: false, arginines: true });
    expect(occParsed.chains).toHaveLength(1);
    expect(occParsed.chains[0].sequence).toBe('AMSR');
    expect(occParsed.totalResidues).toBe(4);
    expect(occParsed.hasAltLocs).toBe(true);
    // AltLoc B (occ 0.60, x=9) wins over A (occ 0.40, x=1)
    expect(occParsed.chains[0].residues[0].atoms.get('N')![0]).toBeCloseTo(9, 4);
    expect(occParsed.chains[0].residues[0].atoms.has('HA')).toBe(false);
    expect(occParsed.chains[0].residues[0].confidence).toBeCloseTo(92, 2);
    // MSE SE -> SD
    expect(occParsed.chains[0].residues[1].atoms.has('SD')).toBe(true);
    expect(occParsed.chains[0].residues[1].atoms.has('SE')).toBe(false);
    // SEP is marked non-standard with parent code 'S'
    expect(occParsed.chains[0].residues[2].resName).toBe('SEP');
    expect(occParsed.chains[0].residues[2].isStandard).toBe(false);
    expect(occParsed.chains[0].residues[2].code).toBe('S');
    // ARG NH1/NH2 swapped (NH1 now at y=-5, closer to CD at y=-3)
    expect(occParsed.chains[0].residues[3].atoms.get('NH1')![1]).toBeCloseTo(-5, 4);
    expect(occParsed.chains[0].residues[3].atoms.get('NH2')![1]).toBeCloseTo(-8, 4);

    // Compare directly against vendored chainResidues on Model 1 PDB slice
    const model1Only = complexPdb.split('ENDMDL')[0];
    const vendorRes = templateInputMod.chainResidues(model1Only, 'A', { altloc: undefined, arginines: true });
    expect(vendorRes.sequence).toBe('AMSR');
    expect(vendorRes.residues[0].atoms.get('N')![0]).toBeCloseTo(9, 4);

    // altlocFirst = true picks A (x=1)
    const firstAltParsed = parseTemplateStructure(complexPdb, { altlocFirst: true, arginines: false });
    expect(firstAltParsed.chains[0].residues[0].atoms.get('N')![0]).toBeCloseTo(1, 4);
    expect(firstAltParsed.chains[0].residues[3].atoms.get('NH1')![1]).toBeCloseTo(-8, 4);

    // Multi-char mmCIF chain IDs & model filtering
    const cifMultiChar =
      makeBackboneMmcif([
        { chainId: 'AA', sequence: 'MKTAYI' },
        { chainId: 'AB1', sequence: 'GSGSGS' },
      ]) +
      '\nATOM 999 C CA . ALA ZZ ZZ 1 1 ? 0.0 0.0 0.0 1.00 50.0 2\n';
    const cifParsed = parseTemplateStructure(cifMultiChar);
    expect(cifParsed.format).toBe('mmcif');
    expect(cifParsed.chains.map((c) => c.chainId)).toEqual(['AA', 'AB1']);
    expect(cifParsed.chains.map((c) => c.sequence)).toEqual(['MKTAYI', 'GSGSGS']);

    // Error handling: empty, invalid, non-polymer-only
    expect(() => parseTemplateStructure('   ')).toThrowError(LocalFoldTemplateError);
    try {
      parseTemplateStructure('');
    } catch (e) {
      expect((e as LocalFoldTemplateError).code).toBe('EMPTY_TEMPLATE');
    }
    try {
      parseTemplateStructure('not a pdb or cif file');
    } catch (e) {
      expect((e as LocalFoldTemplateError).code).toBe('INVALID_FORMAT');
    }
    const waterOnlyPdb = makePdbLine({
      record: 'HETATM',
      serial: 1,
      atomName: 'O',
      resName: 'HOH',
      chainId: 'A',
      resSeq: 1,
      x: 1,
      y: 2,
      z: 3,
      element: 'O',
    });
    try {
      parseTemplateStructure(waterOnlyPdb);
      expect.unreachable('should have thrown NO_POLYMER_RESIDUES');
    } catch (e) {
      expect((e as LocalFoldTemplateError).code).toBe('NO_POLYMER_RESIDUES');
    }
  });

  // -------------------------------------------------------------------------
  // 4. Chain Matching & Sequence-to-Template Alignment
  // -------------------------------------------------------------------------

  it('aligns designed sequences, mutations, gaps, tags, partial domains, chain permutations, and rejects mismatches', () => {
    const natSeq = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK';
    const tplParsed = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: natSeq }]));

    // Exact match
    const exact = alignQueryToTemplate([{ chainId: 'A', sequence: natSeq }], tplParsed);
    expect(exact.overallIdentity).toBe(1);
    expect(exact.overallCoverage).toBe(1);
    expect(exact.matchedQueryResidues).toBe(natSeq.length);

    // Designed sequence (~30% identity, equal length including mutated N/C termini)
    const designed =
      'AKTEYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK'
        .split('')
        .map((c, i) => (i % 3 === 0 ? c : 'L'))
        .join('');
    const mpnnAlign = alignQueryToTemplate([{ chainId: 'A', sequence: designed }], tplParsed);
    expect(mpnnAlign.matchedQueryResidues).toBe(natSeq.length);
    expect(mpnnAlign.chainAlignments[0].residueMap.get(0)).toBe(0);
    expect(mpnnAlign.chainAlignments[0].residueMap.get(natSeq.length - 1)).toBe(natSeq.length - 1);

    // N-terminal His-tag + internal missing loop in template
    const gappedTplSeq = natSeq.slice(0, 15) + natSeq.slice(25);
    const gappedTpl = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: gappedTplSeq }]));
    const taggedQuery = 'HHHHHH' + natSeq;
    const tagGapAlign = alignQueryToTemplate([{ chainId: 'A', sequence: taggedQuery }], gappedTpl);
    // Tag (0..5) stays unaligned; N-term core (6..20) -> (0..14); C-term core (31..58) -> (15..42)
    expect(tagGapAlign.chainAlignments[0].residueMap.has(0)).toBe(false);
    expect(tagGapAlign.chainAlignments[0].residueMap.get(6)).toBe(0);
    expect(tagGapAlign.chainAlignments[0].residueMap.get(31)).toBe(15);

    // Partial domain coverage
    const domain2 = 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY';
    const twoDomainQuery = domain2 + natSeq;
    const partialAlign = alignQueryToTemplate([{ chainId: 'A', sequence: twoDomainQuery }], tplParsed);
    expect(partialAlign.chainAlignments[0].residueMap.get(domain2.length)).toBe(0);
    expect(partialAlign.overallCoverage).toBeCloseTo(natSeq.length / twoDomainQuery.length, 3);

    // Multi-chain permutation (template has [B, A], query has [A, B])
    const seqA = 'MKTAYIAKQRQISFVKSHFSRQ';
    const seqB = 'ACDEFGHIKLMNPQRSTVWYAC';
    const permutedTpl = parseTemplateStructure(
      makeBackbonePdb([
        { chainId: 'X', sequence: seqB },
        { chainId: 'Y', sequence: seqA },
      ]),
    );
    const permAlign = alignQueryToTemplate(
      [
        { chainId: 'A', sequence: seqA },
        { chainId: 'B', sequence: seqB },
      ],
      permutedTpl,
    );
    expect(permAlign.matchedChains).toBe(2);
    expect(permAlign.chainAlignments[0].templateChainId).toBe('Y');
    expect(permAlign.chainAlignments[1].templateChainId).toBe('X');

    // Subset template (2-chain query, 1-chain template)
    const subsetAlign = alignQueryToTemplate(
      [
        { chainId: 'A', sequence: seqA },
        { chainId: 'B', sequence: seqB },
      ],
      parseTemplateStructure(makeBackbonePdb([{ chainId: 'B', sequence: seqB }])),
    );
    expect(subsetAlign.matchedChains).toBe(1);
    expect(subsetAlign.chainAlignments[0].templateChainId).toBeNull();
    expect(subsetAlign.chainAlignments[1].templateChainId).toBe('B');
    expect(subsetAlign.warnings.length).toBeGreaterThan(0);

    // Zero overlap rejection
    const allAla = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: 'A'.repeat(25) }]));
    try {
      alignQueryToTemplate([{ chainId: 'A', sequence: 'W'.repeat(25) }], allAla);
      expect.unreachable('expected ZERO_SEQUENCE_OVERLAP');
    } catch (e) {
      expect((e as LocalFoldTemplateError).code).toBe('ZERO_SEQUENCE_OVERLAP');
    }

    // Low identity rejection
    try {
      alignQueryToTemplate(
        [{ chainId: 'A', sequence: 'AA' + 'W'.repeat(28) }],
        parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: 'AA' + 'Y'.repeat(28) }])),
      );
      expect.unreachable('expected LOW_COVERAGE_OR_IDENTITY');
    } catch (e) {
      expect((e as LocalFoldTemplateError).code).toBe('LOW_COVERAGE_OR_IDENTITY');
    }
  });

  // -------------------------------------------------------------------------
  // 5. Bit-Exact Vendored Parity: AF2 Monomer/Multimer & All 8 AF3 Dialects
  // -------------------------------------------------------------------------

  it('builds AF2 Monomer & AF2-Multimer atom37 payloads with bit-exact vendored templateGeometry parity', async () => {
    const [templateInputMod, templateFeatMod] = await Promise.all([
      importVendor('shared/af3/featurise/template-input.js'),
      importVendor('shared/af3/featurise/template-features.js'),
    ]);

    const core = 'MKTAYIAKQRQISFVKSHFSRQ';
    const query = 'GGGG' + core; // 4-aa uncovered N-terminal overhang
    const pdb = makeBackbonePdb([{ chainId: 'A', sequence: core }]);
    const parsed = parseTemplateStructure(pdb);
    const alignment = alignQueryToTemplate([{ chainId: 'A', sequence: query }], parsed);

    // AF2 Monomer
    const monoPayload = buildAf2TemplatePayload([{ chainId: 'A', sequence: query }], alignment, 'af2-monomer');
    for (let i = 0; i < 4; i += 1) {
      expect(monoPayload.template.aatype[i]).toBe(GAP_AATYPE);
      expect(monoPayload.template.atomMask[i * 37 + 1]).toBe(0);
    }
    expect(monoPayload.coveredResidues).toBe(core.length);

    const vendorChain = templateInputMod.chainResidues(pdb, 'A');
    const vendorMap = new Map<number, number>();
    for (let i = 0; i < core.length; i += 1) vendorMap.set(i, i);
    const vendorSlot = templateInputMod.templateSlotAtom37({
      structure: vendorChain,
      tokens: query.length,
      map: vendorMap,
      offset: 4,
    });
    expect(monoPayload.template.aatype).toEqual(vendorSlot.aatype);
    expect(monoPayload.template.atomPositions).toEqual(vendorSlot.atomPositions);
    expect(monoPayload.template.atomMask).toEqual(vendorSlot.atomMask);

    const tsGeomMono = templateFeatMod.templateGeometry(
      monoPayload.template,
      monoPayload.multichainMask2d,
      query.length,
      templateFeatMod.AF2_ATOM37_MONOMER,
    );
    const vendorGeomMono = templateFeatMod.templateGeometry(
      vendorSlot,
      new Float32Array(query.length * query.length).fill(1),
      query.length,
      templateFeatMod.AF2_ATOM37_MONOMER,
    );
    expect(tsGeomMono.distogram).toEqual(vendorGeomMono.distogram);
    expect(tsGeomMono.unitVector).toEqual(vendorGeomMono.unitVector);

    // AF2-Multimer 1-chain asymId synthesis + 2-chain spanChains true vs false
    expect(monoPayload.asymId).toBeInstanceOf(Float32Array);
    expect(monoPayload.asymId.length).toBe(query.length);

    const seqA = 'MKTAYIAKQRQI';
    const seqB = 'SFVKSHFSRQLE';
    const dimPdb = makeBackbonePdb([
      { chainId: 'A', sequence: seqA },
      { chainId: 'B', sequence: seqB },
    ]);
    const dimParsed = parseTemplateStructure(dimPdb);
    const dimAlign = alignQueryToTemplate(
      [
        { chainId: 'A', sequence: seqA },
        { chainId: 'B', sequence: seqB },
      ],
      dimParsed,
    );
    const L = seqA.length + seqB.length;
    const spanned = buildAf2TemplatePayload(
      [
        { chainId: 'A', sequence: seqA },
        { chainId: 'B', sequence: seqB },
      ],
      dimAlign,
      'af2-multimer',
      { spanChains: true },
    );
    const unspanned = buildAf2TemplatePayload(
      [
        { chainId: 'A', sequence: seqA },
        { chainId: 'B', sequence: seqB },
      ],
      dimAlign,
      'af2-multimer',
      { spanChains: false },
    );
    // Cross-chain pair (0, seqA.length) is 1 when spanChains=true, 0 when spanChains=false
    expect(spanned.multichainMask2d[0 * L + seqA.length]).toBe(1);
    expect(unspanned.multichainMask2d[0 * L + seqA.length]).toBe(0);
    // Intra-chain pair (0, 1) is 1 in both
    expect(unspanned.multichainMask2d[0 * L + 1]).toBe(1);
    // asymId distinguishes chains
    expect(spanned.asymId[0]).toBe(0);
    expect(spanned.asymId[seqA.length]).toBe(1);
  });

  it('builds AF3 dense-24 templateSlots across all 8 AF3 dialects with modified/DNA/ligand batch parity', async () => {
    const [
      dialectMod,
      manifestsMod,
      batchMod,
      ccdMod,
      templateInputMod,
      templateFeatMod,
      fusedMod,
    ] = await Promise.all([
      importVendor('shared/af3/dialect.js'),
      importVendor('shared/bundles/manifests/index.js'),
      importVendor('shared/af3/featurise/batch.js'),
      importVendor('shared/af3/featurise/ccd-component.js'),
      importVendor('shared/af3/featurise/template-input.js'),
      importVendor('shared/af3/featurise/template-features.js'),
      importVendor('shared/af3/featurise/template-fused-features.js'),
    ]);

    const chainA = 'MKTAYIAKQR';
    const chainB = 'QISFVKSHFS';
    // Include SEP (modified parent SER), MSE (SE->SD), altLocs, DNA, and ZN ligand in PDB
    const tplPdb = [
      makePdbLine({ serial: 1, atomName: 'N', resName: 'MET', chainId: 'A', resSeq: 1, x: 0, y: 0, z: 0 }),
      makePdbLine({ serial: 2, atomName: 'CA', resName: 'MET', chainId: 'A', resSeq: 1, x: 1, y: 0, z: 0 }),
      makePdbLine({ serial: 3, atomName: 'C', resName: 'MET', chainId: 'A', resSeq: 1, x: 2, y: 0, z: 0 }),
      makePdbLine({ serial: 4, atomName: 'CB', resName: 'MET', chainId: 'A', resSeq: 1, x: 1, y: -1.5, z: 0 }),
      // SEP at residue 2 (representing K/S position)
      makePdbLine({ record: 'HETATM', serial: 5, atomName: 'N', resName: 'SEP', chainId: 'A', resSeq: 2, x: 3.5, y: 0, z: 0, element: 'N' }),
      makePdbLine({ record: 'HETATM', serial: 6, atomName: 'CA', resName: 'SEP', chainId: 'A', resSeq: 2, x: 4.5, y: 0, z: 0, element: 'C' }),
      makePdbLine({ record: 'HETATM', serial: 7, atomName: 'C', resName: 'SEP', chainId: 'A', resSeq: 2, x: 5.5, y: 0, z: 0, element: 'C' }),
      makePdbLine({ record: 'HETATM', serial: 8, atomName: 'CB', resName: 'SEP', chainId: 'A', resSeq: 2, x: 4.5, y: -1.5, z: 0, element: 'C' }),
      // Rest of chain A & chain B via backbone generator lines
      ...makeBackbonePdb([
        { chainId: 'A', sequence: chainA.slice(2), startResSeq: 3 },
        { chainId: 'B', sequence: chainB, startResSeq: 1 },
      ])
        .split('\n')
        .filter((l) => l.startsWith('ATOM  ')),
    ].join('\n');

    const af3Models = LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3');
    expect(af3Models).toHaveLength(8);

    const zn = ccdMod.parseCcdComponent([
      'data_ZN',
      '_chem_comp.id ZN',
      '_chem_comp.type non-polymer',
      '_chem_comp_atom.comp_id ZN',
      '_chem_comp_atom.atom_id ZN',
      '_chem_comp_atom.type_symbol ZN',
      '_chem_comp_atom.charge 2',
      '_chem_comp_atom.pdbx_model_Cartn_x_ideal 0.0',
      '_chem_comp_atom.pdbx_model_Cartn_y_ideal 0.0',
      '_chem_comp_atom.pdbx_model_Cartn_z_ideal 0.0',
      '#',
    ].join('\n'));
    const polymerSeq = `${chainA}:${chainB}:ACGT`; // protein A + protein B + DNA

    for (const opt of af3Models) {
      const bundle = manifestsMod.MODEL_BUNDLES[opt.id];
      const dialect = dialectMod.dialectFor(bundle.model);
      const featOpts = dialectMod.featuriserDialect(dialect);
      const vendorPolicy = dialectMod.templatePolicyFor(dialect);
      const tsPolicy = getLocalFoldTemplateDialectPolicy(opt.id);

      const { batch } = batchMod.af3BatchFromA3m(polymerSeq, null, {
        ...featOpts,
        chainKinds: ['protein', 'protein', 'dna'],
        ligands: [zn],
      });

      const parsed = parseTemplateStructure(tplPdb, {
        altlocFirst: tsPolicy.altlocFirst,
        arginines: true,
      });
      const alignment = alignQueryToTemplate(
        [
          { chainId: 'A', sequence: chainA },
          { chainId: 'B', sequence: chainB },
        ],
        parsed,
      );
      const slots = buildAf3TemplateSlots(
        { tokens: batch.tokens, residueOfToken: batch.residueOfToken },
        alignment,
        opt.id,
        { spanChains: true },
      );
      expect(slots).toHaveLength(1);
      const tsSlot = slots[0];
      expect(tsSlot.aatype.length).toBe(batch.tokens);
      expect(tsSlot.covered).toBe(chainA.length + chainB.length);
      // DNA & ZN ligand tokens must remain uncovered (GAP_AATYPE = 21)
      const firstNonProteinToken = chainA.length + chainB.length;
      for (let t = firstNonProteinToken; t < batch.tokens; t += 1) {
        expect(tsSlot.aatype[t]).toBe(GAP_AATYPE);
      }

      // Build identical vendor slots using chainResidues + templateSlot + mergeTemplateSlots
      const vChainA = templateInputMod.chainResidues(tplPdb, 'A', vendorPolicy);
      const vChainB = templateInputMod.chainResidues(tplPdb, 'B', vendorPolicy);
      const vSlotA = templateInputMod.templateSlot({
        structure: vChainA,
        tokens: batch.tokens,
        map: alignment.chainAlignments[0].residueMap,
        tokenOf: (r: number) => batch.residueOfToken.indexOf(r),
        policy: vendorPolicy,
      });
      const vSlotB = templateInputMod.templateSlot({
        structure: vChainB,
        tokens: batch.tokens,
        map: alignment.chainAlignments[1].residueMap,
        tokenOf: (r: number) => batch.residueOfToken.indexOf(chainA.length + r),
        policy: vendorPolicy,
      });
      const vMerged = templateInputMod.mergeTemplateSlots([vSlotA, vSlotB]);

      expect(tsSlot.aatype).toEqual(vMerged.aatype);
      expect(tsSlot.atomPositions).toEqual(vMerged.atomPositions);
      expect(tsSlot.atomMask).toEqual(vMerged.atomMask);

      // Verify downstream template feature computations match bit-for-bit across every dialect regime
      const mask = templateFeatMod.multichainMaskFor(batch.asymId, batch.tokens, {
        coverage: templateFeatMod.coverageOf(tsSlot, batch.tokens),
        spanChains: tsSlot.spanChains,
      });
      const vMask = templateFeatMod.multichainMaskFor(batch.asymId, batch.tokens, {
        coverage: templateFeatMod.coverageOf(vMerged, batch.tokens),
        spanChains: true,
      });
      expect(mask).toEqual(vMask);

      if (opt.id === 'boltz2') {
        const tsF = templateFeatMod.boltz2TemplateFeatures(tsSlot, mask, batch.tokens);
        const vF = templateFeatMod.boltz2TemplateFeatures(vMerged, vMask, batch.tokens);
        expect(tsF).toEqual(vF);
      } else if (opt.id === 'rosettafold3') {
        const tsF = templateFeatMod.rosettafold3TemplateFeatures(tsSlot, mask, batch.tokens);
        const vF = templateFeatMod.rosettafold3TemplateFeatures(vMerged, vMask, batch.tokens);
        expect(tsF).toEqual(vF);
      } else if (opt.id === 'protenix2' || opt.id === 'mol1') {
        const tsF = fusedMod.fusedTemplateFeaturesSparse(tsSlot, batch.tokens, 108, dialect, mask);
        const vF = fusedMod.fusedTemplateFeaturesSparse(vMerged, batch.tokens, 108, dialect, vMask);
        expect(tsF).toEqual(vF);
      } else {
        const tsG = templateFeatMod.templateGeometry(tsSlot, mask, batch.tokens, templateFeatMod.AF3_DENSE);
        const vG = templateFeatMod.templateGeometry(vMerged, vMask, batch.tokens, templateFeatMod.AF3_DENSE);
        expect(tsG.distogram).toEqual(vG.distogram);
        expect(tsG.unitVector).toEqual(vG.unitVector);
      }
    }

    // Verify SEP modified policy differences explicitly across dialects
    const sepToken = 1;
    const { batch: sepBatchA } = batchMod.af3BatchFromA3m(chainA, null);
    const openbindSlot = buildAf3TemplateSlots(
      sepBatchA,
      alignQueryToTemplate([{ chainId: 'A', sequence: chainA }], parseTemplateStructure(tplPdb)),
      'openbind0',
    )[0];
    expect(openbindSlot.aatype[sepToken]).toBe(UNK_AATYPE);

    const af3Slot = buildAf3TemplateSlots(
      sepBatchA,
      alignQueryToTemplate([{ chainId: 'A', sequence: chainA }], parseTemplateStructure(tplPdb)),
      'af3',
    )[0];
    expect(af3Slot.aatype[sepToken]).toBe(15); // SER = 15 in ARNDCQEGHILKMFPSTWYV
  });

  // -------------------------------------------------------------------------
  // 6. Engine Registry, Smart-Wiring, Conditioning, Bio-Compatibility & Runtime Guards
  // -------------------------------------------------------------------------

  it('preserves baseline defaults, keeps sequences at port 0, smart-wires sequences before template, and validates template contracts', () => {
    const lfManifest = ADAPTER_REGISTRY['localfold-af2'];
    expect(Object.keys(lfManifest.inputs)).toEqual(['sequences', 'template']);
    expect(lfManifest.inputs.template).toEqual({ type: 'StructureArtifact', required: false });

    // Baseline defaults unchanged
    expect(Object.keys(defaultParamsForAdapter('localfold-af2'))).toEqual([
      'model',
      'recycles',
      'diffusionSteps',
      'capMode',
      'maxSequences',
      'seed',
    ]);

    // Mode switch preserves template params
    const switched = sanitizeParamsOnModeSwitch(
      'localfold-af2',
      { model: 'monomer', templatePdbText: 'ATOM...', templateFilename: 'x.pdb', templateSpanChains: false },
      'boltz2',
    );
    expect(switched.templatePdbText).toBe('ATOM...');
    expect(switched.templateFilename).toBe('x.pdb');
    expect(switched.templateSpanChains).toBe(false);

    // insertBlockSmart prefers required `sequences` over optional `template` when connecting rfd4-proteina -> localfold-af2
    const rfdGraph: WorkflowGraph = {
      id: 'smart-wire',
      name: 'Smart Wire',
      nodes: [
        { id: 'rfd1', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 70 }, position: { x: 40, y: 100 } },
      ],
      edges: [],
    };
    const inserted = insertBlockSmart(rfdGraph, 'localfold-af2', 'rfd1');
    const edgeToLf = inserted.graph.edges.find((e) => e.toNode === inserted.nodeId);
    expect(edgeToLf).toEqual({
      fromNode: 'rfd1',
      fromPort: 'sequences',
      toNode: inserted.nodeId,
      toPort: 'sequences',
    });

    const seq = 'MKTAYIAKQRQISFVKSHFSRQ';
    const validPdb = makeBackbonePdb([{ chainId: 'A', sequence: seq }]);

    // resolveLocalFoldEffectiveConfig captures invalid template safely without throwing
    const badCfg = resolveLocalFoldEffectiveConfig({
      modelKey: 'monomer',
      params: { templatePdbText: 'not valid pdb' },
    });
    expect(badCfg.isTemplateCapable).toBe(true);
    expect(badCfg.templateFamily).toBe('af2-monomer');
    expect(badCfg.templateSource).toBeNull();
    expect(badCfg.templateSourceError).toMatch(/Unrecognized/);

    // crossValidate rejects template on monomer-3..5 and ef2-fast-*, and invalid inline template
    for (const freeModel of ['monomer-3', 'monomer-4', 'monomer-5', 'ef2-fast-300m', 'ef2-fast-600m']) {
      const g: WorkflowGraph = {
        id: 'g-free',
        name: 'Free',
        nodes: [
          { id: 'seq1', adapterId: 'sequence-input', params: { sequence: seq } },
          { id: 'lf1', adapterId: 'localfold-af2', params: { model: freeModel, templatePdbText: validPdb } },
        ],
        edges: [{ fromNode: 'seq1', fromPort: 'sequences', toNode: 'lf1', toPort: 'sequences' }],
      };
      const errs = validateGraph(g);
      expect(errs.length).toBeGreaterThan(0);
      expect(errs.some((e) => e.includes('template-free'))).toBe(true);

      // Also reject wired template port on template-free model via crossValidate
      const gWired: WorkflowGraph = {
        id: 'g-free-wired',
        name: 'Free Wired',
        nodes: [
          { id: 'pdb1', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
          { id: 'seq1', adapterId: 'sequence-input', params: { sequence: seq } },
          { id: 'lf1', adapterId: 'localfold-af2', params: { model: freeModel } },
        ],
        edges: [
          { fromNode: 'seq1', fromPort: 'sequences', toNode: 'lf1', toPort: 'sequences' },
          { fromNode: 'pdb1', fromPort: 'structure', toNode: 'lf1', toPort: 'template' },
        ],
      };
      const wiredErrs = validateGraph(gWired);
      expect(wiredErrs.some((e) => e.includes('template-free'))).toBe(true);

      const bioIssues = validateBioCapabilities(g);
      const unsupported = bioIssues.find((i) => i.code === 'localfold-template-unsupported-model');
      expect(unsupported).toBeDefined();
      expect(unsupported!.severity).toBe('error');
      expect(unsupported!.fix?.paramsPatch.model).toBe('monomer');
    }

    // Both port & inline emits warning with 1-click fix
    const bothGraph: WorkflowGraph = {
      id: 'g-both',
      name: 'Both',
      nodes: [
        { id: 'pdb1', adapterId: 'pdb-input', params: { mode: 'upload', pdbText: validPdb } },
        { id: 'seq1', adapterId: 'sequence-input', params: { sequence: seq } },
        { id: 'lf1', adapterId: 'localfold-af2', params: { model: 'monomer', templatePdbText: validPdb } },
      ],
      edges: [
        { fromNode: 'seq1', fromPort: 'sequences', toNode: 'lf1', toPort: 'sequences' },
        { fromNode: 'pdb1', fromPort: 'structure', toNode: 'lf1', toPort: 'template' },
      ],
    };
    const bothIssues = validateBioCapabilities(bothGraph);
    const bothWarn = bothIssues.find((i) => i.code === 'localfold-template-both-sources');
    expect(bothWarn).toBeDefined();
    expect(bothWarn!.severity).toBe('warning');
    expect(bothWarn!.fix?.paramsPatch).toEqual({ templatePdbText: '', templateFilename: '' });

    // Static mismatch detection in validateBiologicalCompatibility
    const mismatchGraph: WorkflowGraph = {
      id: 'g-mis',
      name: 'Mismatch',
      nodes: [
        { id: 'seq1', adapterId: 'sequence-input', params: { sequence: 'W'.repeat(25) } },
        {
          id: 'lf1',
          adapterId: 'localfold-af2',
          params: {
            model: 'monomer',
            templatePdbText: makeBackbonePdb([{ chainId: 'A', sequence: 'A'.repeat(25) }]),
          },
        },
      ],
      edges: [{ fromNode: 'seq1', fromPort: 'sequences', toNode: 'lf1', toPort: 'sequences' }],
    };
    const mismatchIssues = validateBioCapabilities(mismatchGraph);
    expect(mismatchIssues.some((i) => i.code === 'localfold-template-mismatch')).toBe(true);

    // Runtime input guard: template-free rejection, invalid/mismatched rejection, valid pass with warning
    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'seqset_1',
      parentStructureId: 'struct_src',
      sourceModelId: 'sequence-input',
      candidates: [{ id: 'c1', sequence: seq, sourceModelId: 'sequence-input', score: null, globalScore: null }],
    };
    const tplArtifact = makeStructureArtifact({ pdbText: validPdb });

    expect(() =>
      validateRuntimeNodeInputs(
        { id: 'lf1', adapterId: 'localfold-af2', params: { model: 'monomer-3' } },
        { sequences: seqSet, template: tplArtifact } as Record<string, CanonicalArtifact>,
      ),
    ).toThrow(/template-free/);

    expect(() =>
      validateRuntimeNodeInputs(
        { id: 'lf1', adapterId: 'localfold-af2', params: { model: 'ef2-fast-600m', templatePdbText: validPdb } },
        { sequences: seqSet } as Record<string, CanonicalArtifact>,
      ),
    ).toThrow(/template-free/);

    expect(() =>
      validateRuntimeNodeInputs(
        { id: 'lf1', adapterId: 'localfold-af2', params: { model: 'monomer' } },
        {
          sequences: {
            kind: 'SequenceSet',
            id: 's_bad',
            parentStructureId: 'struct_src',
            sourceModelId: 'sequence-input',
            candidates: [{ id: 'c1', sequence: 'W'.repeat(25), sourceModelId: 'sequence-input', score: null, globalScore: null }],
          } as SequenceSet,
          template: makeStructureArtifact({
            pdbText: makeBackbonePdb([{ chainId: 'A', sequence: 'A'.repeat(25) }]),
          }),
        } as Record<string, CanonicalArtifact>,
      ),
    ).toThrowError(LocalFoldTemplateError);

    const runtimeOk = validateRuntimeNodeInputs(
      { id: 'lf1', adapterId: 'localfold-af2', params: { model: 'boltz2', templatePdbText: validPdb } },
      { sequences: seqSet, template: tplArtifact } as Record<string, CanonicalArtifact>,
    );
    expect(runtimeOk.warnings.some((w) => w.includes('wired template port'))).toBe(true);

    // Verify residueOfToken multi-token shift (e.g. atomized modified query residue occupying multiple tokens)
    const shiftedBatchLayout = {
      tokens: 6,
      // residue 0 -> token 0, residue 1 -> tokens 1,2,3 (modified 3-atom residue), residue 2 -> token 4, ligand -> token 5 (-1)
      residueOfToken: new Int32Array([0, 1, 1, 1, 2, -1]),
    };
    const shiftMiniPdb = makeBackbonePdb([{ chainId: 'A', sequence: 'MKT' }]);
    const shiftParsed = parseTemplateStructure(shiftMiniPdb);
    const shiftAlign = alignQueryToTemplate([{ chainId: 'A', sequence: 'MKT' }], shiftParsed);
    const shiftSlot = buildAf3TemplateSlots(shiftedBatchLayout, shiftAlign, 'af3')[0];
    expect(shiftSlot.aatype[0]).toBe(12); // M
    expect(shiftSlot.aatype[1]).toBe(11); // K (first token of residue 1)
    expect(shiftSlot.aatype[2]).toBe(GAP_AATYPE); // remaining tokens of residue 1
    expect(shiftSlot.aatype[3]).toBe(GAP_AATYPE);
    expect(shiftSlot.aatype[4]).toBe(16); // T (shifted from residue 2 -> token 4!)
    expect(shiftSlot.aatype[5]).toBe(GAP_AATYPE); // ligand token (-1)
  });

  it('handles concatenated chainLengths, missingTemplateUpload, semi-global tagged/truncated cores, short peptides, and high-scale alignments', () => {
    // 0. Ensure unrelated adapters like top-k-filter were not polluted with a template port
    expect(Object.keys(ADAPTER_REGISTRY['top-k-filter'].inputs)).toEqual(['sequences']);

    // 1. Concatenated multi-chain SequenceSet (no `:`) with chainLengths + permuted 2-chain template [B, A]
    const chainA = 'MKTAYIAKQRQISFVKSHFSRQLEER';
    const chainB = 'ACDEFGHIKLMNPQRSTVWYACDEFG';
    const permutedPdb = makeBackbonePdb([
      { chainId: 'B', sequence: chainB },
      { chainId: 'A', sequence: chainA },
    ]);
    const concatSeqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'seqset_concat',
      parentStructureId: 'struct_src',
      sourceModelId: 'proteinmpnn',
      candidates: [
        {
          id: 'mpnn_1',
          sequence: chainA + chainB,
          chainLengths: [chainA.length, chainB.length],
          chainIds: ['A', 'B'],
          sourceModelId: 'proteinmpnn',
          score: 1.0,
          globalScore: 1.0,
        },
      ],
    };
    const concatTpl = makeStructureArtifact({ pdbText: permutedPdb });

    const concatRuntime = validateRuntimeNodeInputs(
      { id: 'lf1', adapterId: 'localfold-af2', params: { model: 'multimer' } },
      { sequences: concatSeqSet, template: concatTpl } as Record<string, CanonicalArtifact>,
    );
    expect(concatRuntime.warnings).toEqual([]);

    const concatGraph: WorkflowGraph = {
      id: 'g-concat',
      name: 'Concat Multichain',
      nodes: [
        { id: 'mpnn1', adapterId: 'proteinmpnn', params: {} },
        { id: 'pdb1', adapterId: 'pdb-input', params: { mode: 'upload', pdbText: permutedPdb } },
        { id: 'lf1', adapterId: 'localfold-af2', params: { model: 'multimer' } },
      ],
      edges: [
        { fromNode: 'mpnn1', fromPort: 'sequences', toNode: 'lf1', toPort: 'sequences' },
        { fromNode: 'pdb1', fromPort: 'structure', toNode: 'lf1', toPort: 'template' },
      ],
    };
    const concatBioIssues = validateBioCapabilities(concatGraph, {
      mpnn1: { sequences: concatSeqSet },
      pdb1: { structure: concatTpl },
    });
    expect(concatBioIssues.filter((i) => i.code === 'localfold-template-mismatch')).toEqual([]);

    // Mismatched concatenated candidate with chainLengths fails cleanly in both validators
    const badConcatSeqSet: SequenceSet = {
      ...concatSeqSet,
      candidates: [
        {
          id: 'mpnn_bad',
          sequence: 'W'.repeat(chainA.length) + 'W'.repeat(chainB.length),
          chainLengths: [chainA.length, chainB.length],
          chainIds: ['A', 'B'],
          sourceModelId: 'proteinmpnn',
          score: 1.0,
          globalScore: 1.0,
        },
      ],
    };
    const badConcatBioIssues = validateBioCapabilities(concatGraph, {
      mpnn1: { sequences: badConcatSeqSet },
      pdb1: { structure: concatTpl },
    });
    expect(badConcatBioIssues.some((i) => i.code === 'localfold-template-mismatch')).toBe(true);
    expect(() =>
      validateRuntimeNodeInputs(
        { id: 'lf1', adapterId: 'localfold-af2', params: { model: 'multimer' } },
        { sequences: badConcatSeqSet, template: concatTpl } as Record<string, CanonicalArtifact>,
      ),
    ).toThrowError(LocalFoldTemplateError);

    // 2. missingTemplateUpload: true on template-free models triggers 1-click switch fix in validateBioCapabilities
    for (const freeModel of ['monomer-3', 'ef2-fast-300m'] as const) {
      const sharedGraph: WorkflowGraph = {
        id: 'g-missing-tpl',
        name: 'Shared Missing Template',
        nodes: [
          { id: 'seq1', adapterId: 'sequence-input', params: { sequence: chainA } },
          {
            id: 'lf1',
            adapterId: 'localfold-af2',
            params: {
              model: freeModel,
              missingTemplateUpload: true,
              expectedTemplateFilename: 'scaffold.pdb',
            },
          },
        ],
        edges: [{ fromNode: 'seq1', fromPort: 'sequences', toNode: 'lf1', toPort: 'sequences' }],
      };
      const issues = validateBioCapabilities(sharedGraph);
      const unsupported = issues.find((i) => i.code === 'localfold-template-unsupported-model');
      expect(unsupported).toBeDefined();
      expect(unsupported!.severity).toBe('error');
      expect(unsupported!.fix?.paramsPatch.model).toBe('monomer');
    }

    // 3. True semi-global alignment: N-terminal tag + low-identity (20%) N-terminal core (qLen !== tLen) & 1-aa C-term truncation
    const tpl55 = 'A'.repeat(15) + 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV';
    const taggedDes55 = 'MGSSHHHHHH' + 'AWWWWWAWWWWWAWW' + 'MKTEYLPKQRQIDFAKSHFNRQLDDRLGVIEAQAPVLSRM';
    const parsed55 = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: tpl55 }]));
    const align55 = alignQueryToTemplate([{ chainId: 'A', sequence: taggedDes55 }], parsed55);
    expect(align55.matchedQueryResidues).toBe(55);
    expect(align55.chainAlignments[0].residueMap.has(0)).toBe(false); // tag stays uncovered
    expect(align55.chainAlignments[0].residueMap.has(9)).toBe(false);
    expect(align55.chainAlignments[0].residueMap.get(10)).toBe(0); // full N-term core preserved
    expect(align55.chainAlignments[0].residueMap.get(64)).toBe(54);

    const t60 = 'A'.repeat(15) + 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ';
    const q59 = 'AWWWWWAWWWWWAWW' + 'MKTEYLPKQRQIDFAKSHFNRQLDDRLGVIEAQAPVLSRMGDGT';
    const parsed60 = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: t60 }]));
    const align59 = alignQueryToTemplate([{ chainId: 'A', sequence: q59 }], parsed60);
    expect(align59.matchedQueryResidues).toBe(59);
    expect(align59.chainAlignments[0].residueMap.get(0)).toBe(0);
    expect(align59.chainAlignments[0].residueMap.get(58)).toBe(58);

    // 4. Short peptide (5 aa, 60% identity) and 15-aa low-identity (20%, dp.score == 0) succeed; unrelated still throw
    const pepTpl = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: 'MKTAY' }]));
    const pepAlign = alignQueryToTemplate([{ chainId: 'A', sequence: 'AKTEY' }], pepTpl);
    expect(pepAlign.matchedQueryResidues).toBe(5);
    expect(pepAlign.overallIdentity).toBeCloseTo(0.6, 4);

    const des15Tpl = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: 'A'.repeat(15) }]));
    const des15Align = alignQueryToTemplate([{ chainId: 'A', sequence: 'AWWWWWAWWWWWAWW' }], des15Tpl);
    expect(des15Align.matchedQueryResidues).toBe(15);
    expect(des15Align.overallIdentity).toBeCloseTo(0.2, 4);

    expect(() =>
      alignQueryToTemplate([{ chainId: 'A', sequence: 'WWWWW' }], pepTpl),
    ).toThrowError(LocalFoldTemplateError);
    expect(() =>
      alignQueryToTemplate([{ chainId: 'A', sequence: 'AKWWW' }], pepTpl),
    ).toThrowError(LocalFoldTemplateError);

    // 5. High-scale sanity checks: Q = 10 chains (> 8 greedy bipartite path) and >1000 aa sequence
    const alphabet20 = 'ACDEFGHIKLMNPQRSTVWY';
    const tenChains = Array.from({ length: 10 }, (_, idx) => {
      const seq = Array.from({ length: 18 }, (__, j) => alphabet20[(idx * 7 + j * 3) % 20]).join('');
      return { chainId: String.fromCharCode(65 + idx), sequence: seq };
    });
    const permutedTenTpl = parseTemplateStructure(
      makeBackboneMmcif([...tenChains].reverse().map((c, idx) => ({ chainId: `T${idx}`, sequence: c.sequence }))),
    );
    const tenAlign = alignQueryToTemplate(tenChains, permutedTenTpl);
    expect(tenAlign.matchedChains).toBe(10);
    expect(tenAlign.overallIdentity).toBe(1);
    expect(tenAlign.overallCoverage).toBe(1);

    const longSeq = Array.from({ length: 1100 }, (_, i) => alphabet20[(i * 7 + (i >> 3)) % 20]).join('');
    const longMut = longSeq.slice(0, 500) + 'A' + longSeq.slice(501);
    const longTpl = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: longSeq }]));
    const longAlign = alignQueryToTemplate([{ chainId: 'A', sequence: longMut }], longTpl);
    expect(longAlign.matchedQueryResidues).toBe(1100);
    expect(longAlign.overallCoverage).toBe(1);
  });

  it('rejects unrelated real PDB pairs and corner motifs while trimming dual-end tags and accepting valid low-identity/domain/indels', () => {
    // 1. All 10 pairwise checks of 5 unrelated real PDB proteins must throw LocalFoldTemplateError
    const realProteins: Array<{ name: string; seq: string }> = [
      {
        name: 'ubiquitin_76',
        seq: 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
      },
      {
        name: 'bpti_58',
        seq: 'RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA',
      },
      {
        name: 'barnase_110',
        seq: 'AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR',
      },
      {
        name: 'lysozyme_129',
        seq: 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL',
      },
      {
        name: 'streptavidin_125',
        seq: 'EAGITGTWYNQLGSTFIVTAGADGALTGTYESAVGNAESRYVLTGRYDSAPATDGSGTALGWTVAWKNNYRNAHSATTWSGQYVGGAEARINTQWLLTSGTTEANAWKSTLVGHDTFTKVKPSA',
      },
    ];
    for (let i = 0; i < realProteins.length; i += 1) {
      for (let j = i + 1; j < realProteins.length; j += 1) {
        const tplParsed = parseTemplateStructure(
          makeBackbonePdb([{ chainId: 'A', sequence: realProteins[i].seq }]),
        );
        expect(() =>
          alignQueryToTemplate([{ chainId: 'A', sequence: realProteins[j].seq }], tplParsed),
          `${realProteins[i].name} vs ${realProteins[j].name} should fail`,
        ).toThrowError(LocalFoldTemplateError);
      }
    }

    // 2. Opposite-corner motif exploits (HHHHHH on 60-aa, GGGGS on 40-aa) throw LocalFoldTemplateError
    const cornerT60 = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: 'HHHHHH' + 'ADEFGI'.repeat(9) }]),
    );
    expect(() =>
      alignQueryToTemplate([{ chainId: 'A', sequence: 'KLMNPQ'.repeat(9) + 'HHHHHH' }], cornerT60),
    ).toThrowError(LocalFoldTemplateError);

    const cornerT40 = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: 'GGGGS' + 'ADEFHI'.repeat(5) + 'ADEFH' }]),
    );
    expect(() =>
      alignQueryToTemplate(
        [{ chainId: 'A', sequence: 'KLMNPQ'.repeat(5) + 'KLMNP' + 'GGGGS' }],
        cornerT40,
      ),
    ).toThrowError(LocalFoldTemplateError);

    // 3. Dual-end non-homologous tags leave tags unmapped (GAP_AATYPE) and map the core cleanly
    const core40 = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV';

    // 3a. MGSSHHHHHH + core40 vs DYKDDDDKGG + core40
    const dualN10Tpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: 'DYKDDDDKGG' + core40 }]),
    );
    const dualN10Align = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'MGSSHHHHHH' + core40 }],
      dualN10Tpl,
    );
    expect(dualN10Align.matchedQueryResidues).toBe(40);
    expect(dualN10Align.chainAlignments[0].residueMap.get(10)).toBe(10);
    expect(dualN10Align.chainAlignments[0].residueMap.get(49)).toBe(49);
    const dualN10Af2 = buildAf2TemplatePayload(
      [{ chainId: 'A', sequence: 'MGSSHHHHHH' + core40 }],
      dualN10Align,
      'af2-monomer',
    );
    for (let k = 0; k < 10; k += 1) {
      expect(dualN10Af2.template.aatype[k]).toBe(GAP_AATYPE);
    }
    expect(dualN10Af2.template.aatype[10]).not.toBe(GAP_AATYPE);

    // 3b. MGSSHHHHHH + core40 vs MADYKDDDDK + core40 (shared leading M at tag start)
    const dualN10MTpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: 'MADYKDDDDK' + core40 }]),
    );
    const dualN10MAlign = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'MGSSHHHHHH' + core40 }],
      dualN10MTpl,
    );
    expect(dualN10MAlign.matchedQueryResidues).toBe(40);
    expect(dualN10MAlign.chainAlignments[0].residueMap.has(0)).toBe(false);
    expect(dualN10MAlign.chainAlignments[0].residueMap.get(10)).toBe(10);

    // 3c. 10+40+10 vs 12+40+12 dual N+C tags
    const dualNCTpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: 'WSHPQFEKKGGW' + core40 + 'GSDYKDDDDKGG' }]),
    );
    const dualNCAlign = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'MGSSHHHHHH' + core40 + 'LEHHHHHHAA' }],
      dualNCTpl,
    );
    expect(dualNCAlign.matchedQueryResidues).toBe(40);
    expect(dualNCAlign.chainAlignments[0].residueMap.get(10)).toBe(12);
    expect(dualNCAlign.chainAlignments[0].residueMap.get(49)).toBe(51);
    const dualNCAf2 = buildAf2TemplatePayload(
      [{ chainId: 'A', sequence: 'MGSSHHHHHH' + core40 + 'LEHHHHHHAA' }],
      dualNCAlign,
      'af2-monomer',
    );
    for (let k = 0; k < 10; k += 1) {
      expect(dualNCAf2.template.aatype[k]).toBe(GAP_AATYPE);
      expect(dualNCAf2.template.aatype[50 + k]).toBe(GAP_AATYPE);
    }

    // 3d. D*20 + 30-aa 20% core vs E*20 + A*30
    const dualLowTpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: 'E'.repeat(20) + 'A'.repeat(30) }]),
    );
    const dualLowAlign = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'D'.repeat(20) + 'ACCCC'.repeat(6) }],
      dualLowTpl,
    );
    expect(dualLowAlign.matchedQueryResidues).toBe(30);
    expect(dualLowAlign.overallIdentity).toBeCloseTo(0.2, 4);
    expect(dualLowAlign.chainAlignments[0].residueMap.get(20)).toBe(20);
    expect(dualLowAlign.chainAlignments[0].residueMap.get(49)).toBe(49);

    // 4. Valid 16% 100-aa (without and with small gap), 20-aa domain in 200-aa query, 50-aa in 100-aa, 3/5, 3/15, L=1500 indels
    const consSub: Record<string, string> = {
      A: 'S', R: 'K', N: 'D', D: 'E', C: 'S', Q: 'E', E: 'D', G: 'A', H: 'Y', I: 'L',
      L: 'I', K: 'R', M: 'L', F: 'Y', P: 'A', S: 'T', T: 'S', W: 'F', Y: 'F', V: 'I',
    };
    const base100 =
      'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKIADKLEKLKSVIGKVDQLEKQVQELREELKSVQELKSDQIKRLKEKA';
    expect(base100.length).toBe(100);
    // 16% identity (keep every 6th residue for indices 0..90 -> 16 matches, conservatively substitute rest)
    const des16 = base100
      .split('')
      .map((c, i) => (i <= 90 && i % 6 === 0 ? c : consSub[c] ?? 'A'))
      .join('');
    const tpl100 = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: base100 }]));
    const align16 = alignQueryToTemplate([{ chainId: 'A', sequence: des16 }], tpl100);
    expect(align16.matchedQueryResidues).toBe(100);
    expect(align16.overallIdentity).toBeCloseTo(0.16, 4);

    // 16%+ 100-aa with a 1-aa internal deletion in query (keep 17/99 matches so idn/(al+gc+6*go) = 17/106 >= 0.15)
    const des16Gap =
      des16.slice(0, 48) +
      base100[49] + // 17th match
      des16.slice(50);
    const align16Gap = alignQueryToTemplate([{ chainId: 'A', sequence: des16Gap }], tpl100);
    expect(align16Gap.matchedQueryResidues).toBe(99);
    expect(align16Gap.overallIdentity).toBeGreaterThanOrEqual(0.16);

    // 20-aa domain in 200-aa query & 50-aa template in 100-aa query
    const dom20 = 'MKTAYIAKQRQISFVKSHFS';
    const flank90 = 'ADEFGIKLMNPQRSTVWYCA'.repeat(4) + 'ADEFGIKLMN';
    const q200 = flank90 + dom20 + flank90;
    const align20in200 = alignQueryToTemplate(
      [{ chainId: 'A', sequence: q200 }],
      parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: dom20 }])),
    );
    expect(align20in200.matchedQueryResidues).toBe(20);
    expect(align20in200.chainAlignments[0].residueMap.get(90)).toBe(0);

    const dom50 = base100.slice(25, 75);
    const align50in100 = alignQueryToTemplate(
      [{ chainId: 'A', sequence: base100 }],
      parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: dom50 }])),
    );
    expect(align50in100.matchedQueryResidues).toBe(50);
    expect(align50in100.chainAlignments[0].residueMap.get(25)).toBe(0);

    // 3/5 and 3/15 pass
    const align3of5 = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'AKTEY' }],
      parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: 'MKTAY' }])),
    );
    expect(align3of5.matchedQueryResidues).toBe(5);
    expect(align3of5.overallIdentity).toBeCloseTo(0.6, 4);

    const align3of15 = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'AWWWWWAWWWWWAWW' }],
      parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: 'A'.repeat(15) }])),
    );
    expect(align3of15.matchedQueryResidues).toBe(15);
    expect(align3of15.overallIdentity).toBeCloseTo(0.2, 4);

    // L=1500 with N-tag, 2-aa deletion, 2-aa insertion, and point mutations
    const alphabet20 = 'ACDEFGHIKLMNPQRSTVWY';
    const tpl1500Seq = Array.from({ length: 1500 }, (_, i) => alphabet20[(i * 7 + (i >> 3)) % 20]).join('');
    const mutChars = tpl1500Seq.split('');
    for (let i = 30; i < 1470; i += 37) {
      mutChars[i] = alphabet20[(alphabet20.indexOf(mutChars[i]) + 1) % 20];
    }
    const q1500Mut =
      'MGSSHHHHHH' +
      mutChars.slice(0, 400).join('') +
      mutChars.slice(402, 900).join('') +
      'GG' +
      mutChars.slice(900).join('');
    const tpl1500 = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: tpl1500Seq }]));
    const align1500 = alignQueryToTemplate([{ chainId: 'A', sequence: q1500Mut }], tpl1500);
    expect(align1500.matchedQueryResidues).toBe(1498);
    expect(align1500.chainAlignments[0].residueMap.has(0)).toBe(false);
    expect(align1500.chainAlignments[0].residueMap.get(10)).toBe(0);
    expect(align1500.overallIdentity).toBeGreaterThan(0.95);
  });

  it('rejects shared 10-aa tags on unrelated PDBs, preserves dual-tagged 25-50% cores, excludes X/UNK from identity, and handles minConfidence filtering order', () => {
    const ubiquitin = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG';
    const bpti = 'RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA';
    const barnase =
      'AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR';

    // 1. Shared 10-aa N- and C-terminal tags/linkers on unrelated proteins must be rejected
    const sharedTagPairs: Array<[string, string]> = [
      ['MGSSHHHHHH' + bpti, 'MGSSHHHHHH' + ubiquitin],
      ['MGSSHHHHHH' + barnase, 'MGSSHHHHHH' + ubiquitin],
      ['GGGGSGGGGS' + bpti, 'GGGGSGGGGS' + ubiquitin],
      [bpti + 'MGSSHHHHHH', ubiquitin + 'MGSSHHHHHH'],
    ];
    for (const [qSeq, tSeq] of sharedTagPairs) {
      const tpl = parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: tSeq }]));
      expect(() =>
        alignQueryToTemplate([{ chainId: 'A', sequence: qSeq }], tpl),
      ).toThrowError(LocalFoldTemplateError);
    }

    // 2. Dual-tagged 50% conservative core (no lookahead over-trimming) and 25% conservative core (rescue pass)
    const core40 = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV';
    const consSub: Record<string, string> = {
      A: 'S', R: 'K', N: 'D', D: 'E', C: 'S', Q: 'E', E: 'D', G: 'A', H: 'Y', I: 'L',
      L: 'I', K: 'R', M: 'L', F: 'Y', P: 'A', S: 'T', T: 'S', W: 'F', Y: 'F', V: 'I',
    };
    const core40_50pct_ends = core40
      .split('')
      .map((c, i) => (i % 2 === 0 || i === core40.length - 1 ? c : consSub[c]))
      .join('');
    const core40_50pct_even = core40
      .split('')
      .map((c, i) => (i % 2 === 0 ? c : consSub[c]))
      .join('');
    const core40_25pct = core40
      .split('')
      .map((c, i) => (i % 4 === 0 ? c : consSub[c]))
      .join('');

    const dualTpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: 'WSHPQFEKKGGW' + core40 + 'GSDYKDDDDKGG' }]),
    );

    const align50Ends = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'MGSSHHHHHH' + core40_50pct_ends + 'LEHHHHHHAA' }],
      dualTpl,
    );
    expect(align50Ends.matchedQueryResidues).toBe(40);
    expect(align50Ends.chainAlignments[0].residueMap.get(10)).toBe(12);
    expect(align50Ends.chainAlignments[0].residueMap.get(49)).toBe(51);

    const align50Even = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'MGSSHHHHHH' + core40_50pct_even + 'LEHHHHHHAA' }],
      dualTpl,
    );
    expect(align50Even.matchedQueryResidues).toBeGreaterThanOrEqual(39);
    expect(align50Even.chainAlignments[0].residueMap.get(10)).toBe(12);
    expect(align50Even.chainAlignments[0].residueMap.get(48)).toBe(50);

    const align25Dual = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'MGSSHHHHHH' + core40_25pct + 'LEHHHHHHAA' }],
      dualTpl,
    );
    expect(align25Dual.matchedQueryResidues).toBeGreaterThanOrEqual(37);
    expect(align25Dual.chainAlignments[0].residueMap.get(10)).toBe(12);
    expect(align25Dual.overallIdentity).toBeGreaterThanOrEqual(0.25);

    // 3. 'X' / UNK wildcard residues must NOT count as exact identity matches
    const makeUnkPdb = (seq: string): string => {
      const lines: string[] = [];
      let serial = 1;
      for (let i = 0; i < seq.length; i += 1) {
        const ch = seq[i];
        const resName = THREE_LETTER[ch] ?? 'UNK';
        lines.push(makePdbLine({ serial: serial++, atomName: 'N', resName, chainId: 'A', resSeq: i + 1, x: i * 3.8, y: 0, z: 0, element: 'N' }));
        lines.push(makePdbLine({ serial: serial++, atomName: 'CA', resName, chainId: 'A', resSeq: i + 1, x: i * 3.8 + 1, y: 0, z: 0, element: 'C' }));
      }
      lines.push('END');
      return lines.join('\n');
    };

    const unk30Tpl = parseTemplateStructure(makeUnkPdb('X'.repeat(30)));
    expect(() =>
      alignQueryToTemplate([{ chainId: 'A', sequence: 'X'.repeat(30) }], unk30Tpl),
    ).toThrowError(LocalFoldTemplateError);

    const prefixUnkTpl = parseTemplateStructure(makeUnkPdb('ZZZZZ' + 'X'.repeat(25)));
    expect(() =>
      alignQueryToTemplate([{ chainId: 'A', sequence: 'MKTAY' + 'X'.repeat(25) }], prefixUnkTpl),
    ).toThrowError(LocalFoldTemplateError);

    // 4. minConfidence filtering executes prior to alignment metrics & preserves LOW_COVERAGE_OR_IDENTITY when all residues fail B-factor filter
    const lowConfTpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: core40 + 'MKTAYIAKQRQISFVKSHFS', bFactor: 10.0 }]),
    );
    try {
      alignQueryToTemplate(
        [{ chainId: 'A', sequence: core40 + 'MKTAYIAKQRQISFVKSHFS' }],
        lowConfTpl,
        { minConfidence: 50 },
      );
      expect.unreachable('expected LOW_COVERAGE_OR_IDENTITY');
    } catch (e) {
      expect(e).toBeInstanceOf(LocalFoldTemplateError);
      expect((e as LocalFoldTemplateError).code).toBe('LOW_COVERAGE_OR_IDENTITY');
    }
  });

  it('eliminates 4x36 and adversarial shared-tag false positives while preserving tagged homologs, 20%-id MPNN designs, and minConfidence scattered residues', () => {
    const proteins9: Array<{ name: string; seq: string }> = [
      {
        name: 'ubiquitin_76',
        seq: 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
      },
      {
        name: 'bpti_58',
        seq: 'RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA',
      },
      {
        name: 'barnase_110',
        seq: 'AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR',
      },
      {
        name: 'lysozyme_129',
        seq: 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL',
      },
      {
        name: 'streptavidin_125',
        seq: 'EAGITGTWYNQLGSTFIVTAGADGALTGTYESAVGNAESRYVLTGRYDSAPATDGSGTALGWTVAWKNNYRNAHSATTWSGQYVGGAEARINTQWLLTSGTTEANAWKSTLVGHDTFTKVKPSA',
      },
      {
        name: 'gb1_56',
        seq: 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE',
      },
      {
        name: 'crambin_46',
        seq: 'TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN',
      },
      {
        name: 'villin_35',
        seq: 'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
      },
      {
        name: 'trpcage_20',
        seq: 'NLYIQWLKDGGPSSGRPPPS',
      },
    ];

    // 1. Full 4 x 36 = 144 shared-tag matrix (N/C-MGSSHHHHHH & N/C-GGGGSGGGGS across all 36 pairs) -> 0/144 FPs
    const tagConfigs: Array<{ tag: string; side: 'N' | 'C' }> = [
      { tag: 'MGSSHHHHHH', side: 'N' },
      { tag: 'MGSSHHHHHH', side: 'C' },
      { tag: 'GGGGSGGGGS', side: 'N' },
      { tag: 'GGGGSGGGGS', side: 'C' },
    ];
    for (const { tag, side } of tagConfigs) {
      const tagged = proteins9.map((p) => ({
        name: p.name,
        seq: side === 'N' ? tag + p.seq : p.seq + tag,
      }));
      const parsedTemplates = tagged.map((p) =>
        parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: p.seq }])),
      );
      for (let i = 0; i < tagged.length; i += 1) {
        for (let j = i + 1; j < tagged.length; j += 1) {
          expect(() =>
            alignQueryToTemplate([{ chainId: 'A', sequence: tagged[j].seq }], parsedTemplates[i]),
            `${side}-${tag}: ${tagged[i].name} vs ${tagged[j].name} should throw`,
          ).toThrowError(LocalFoldTemplateError);
        }
      }
    }

    // 2. DYKDDDDKGG & WSHPQFEKKGGW shared-tag pairs across 4 core proteins (N & C termini)
    const proteins4 = proteins9.filter((p) =>
      ['bpti_58', 'barnase_110', 'ubiquitin_76', 'gb1_56'].includes(p.name),
    );
    for (const tag of ['DYKDDDDKGG', 'WSHPQFEKKGGW']) {
      for (const side of ['N', 'C'] as const) {
        const tagged = proteins4.map((p) => ({
          name: p.name,
          seq: side === 'N' ? tag + p.seq : p.seq + tag,
        }));
        const parsedTemplates = tagged.map((p) =>
          parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: p.seq }])),
        );
        for (let i = 0; i < tagged.length; i += 1) {
          for (let j = i + 1; j < tagged.length; j += 1) {
            expect(() =>
              alignQueryToTemplate([{ chainId: 'A', sequence: tagged[j].seq }], parsedTemplates[i]),
              `${side}-${tag}: ${tagged[i].name} vs ${tagged[j].name} should throw`,
            ).toThrowError(LocalFoldTemplateError);
          }
        }
      }
    }

    // 3. Adversarial tag variants: dual_shared_tags, MGSSHHHHHH + trpcage_20, and tag12HighBody20Low
    const ubqSeq = proteins9.find((p) => p.name === 'ubiquitin_76')!.seq;
    const bptiSeq = proteins9.find((p) => p.name === 'bpti_58')!.seq;
    const villinSeq = proteins9.find((p) => p.name === 'villin_35')!.seq;
    const trpcageSeq = proteins9.find((p) => p.name === 'trpcage_20')!.seq;

    const dualTagTpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: 'MGSSHHHHHH' + ubqSeq + 'DYKDDDDKGG' }]),
    );
    expect(() =>
      alignQueryToTemplate(
        [{ chainId: 'A', sequence: 'MGSSHHHHHH' + bptiSeq + 'DYKDDDDKGG' }],
        dualTagTpl,
      ),
    ).toThrowError(LocalFoldTemplateError);

    const trpTagQuery = 'MGSSHHHHHH' + trpcageSeq;
    for (const unrelated of [villinSeq, ubqSeq]) {
      const tpl = parseTemplateStructure(
        makeBackbonePdb([{ chainId: 'A', sequence: 'MGSSHHHHHH' + unrelated }]),
      );
      expect(() => alignQueryToTemplate([{ chainId: 'A', sequence: trpTagQuery }], tpl)).toThrowError(
        LocalFoldTemplateError,
      );
    }

    // tag12HighBody20Low: minConfidence=50 masks the 20-aa unrelated body leaving only the 12-aa shared tag
    const tag12 = 'WSHPQFEKKGGW';
    const maskedBodySeq = tag12 + villinSeq.slice(0, 20);
    const maskedBodyBFactors = [
      ...Array.from({ length: tag12.length }, () => 90.0),
      ...Array.from({ length: 20 }, () => 10.0),
    ];
    const tag12Tpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: maskedBodySeq, bFactors: maskedBodyBFactors }]),
    );
    expect(() =>
      alignQueryToTemplate([{ chainId: 'A', sequence: tag12 + trpcageSeq }], tag12Tpl, {
        minConfidence: 50,
      }),
    ).toThrowError(LocalFoldTemplateError);

    // 4. Valid positives preserved:
    // 4a. Differently tagged true homolog (MGSSHHHHHH + bpti_58 vs WSHPQFEKKGGW + bpti_58 -> 58/58)
    const bptiDiffTagTpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: 'WSHPQFEKKGGW' + bptiSeq }]),
    );
    const bptiDiffTagAlign = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'MGSSHHHHHH' + bptiSeq }],
      bptiDiffTagTpl,
    );
    expect(bptiDiffTagAlign.matchedQueryResidues).toBe(58);
    expect(bptiDiffTagAlign.overallIdentity).toBe(1);
    expect(bptiDiffTagAlign.chainAlignments[0].residueMap.get(10)).toBe(12);
    expect(bptiDiffTagAlign.chainAlignments[0].residueMap.get(67)).toBe(69);

    // 4b. Realistic 60-aa ~20%-id ProteinMPNN design (6/10 in 0..9 + 8/50 in 10..59)
    const consSub: Record<string, string> = {
      A: 'S', R: 'K', N: 'D', D: 'E', C: 'S', Q: 'E', E: 'D', G: 'A', H: 'Y', I: 'L',
      L: 'I', K: 'R', M: 'L', F: 'Y', P: 'A', S: 'T', T: 'S', W: 'F', Y: 'F', V: 'I',
    };
    const mpnnNat60 =
      'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKIADKLEK';
    expect(mpnnNat60.length).toBe(60);
    // 6/10 exact at indices 0..9 (0,1,3,5,7,9) + 8/50 exact at indices 10..59 (12,18,24,30,36,42,48,54) -> 14/60 = 23.3%
    const mpnnKeep = new Set([0, 1, 3, 5, 7, 9, 12, 18, 24, 30, 36, 42, 48, 54]);
    const mpnnDes60 = mpnnNat60
      .split('')
      .map((c, i) => (mpnnKeep.has(i) ? c : consSub[c]))
      .join('');
    const mpnnNatTpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: mpnnNat60 }]),
    );
    const mpnn60Align = alignQueryToTemplate([{ chainId: 'A', sequence: mpnnDes60 }], mpnnNatTpl);
    expect(mpnn60Align.matchedQueryResidues).toBe(60);
    expect(mpnn60Align.overallIdentity).toBeCloseTo(14 / 60, 4);
    expect(mpnn60Align.chainAlignments[0].residueMap.get(0)).toBe(0);
    expect(mpnn60Align.chainAlignments[0].residueMap.get(59)).toBe(59);

    // 4c. minConfidence=50 with scattered low-B-factor residues (no synthetic DP gaps!)
    for (const stride of [2, 3, 4, 5]) {
      const bFactors = Array.from({ length: 60 }, (_, idx) => (idx % stride === 0 ? 10.0 : 90.0));
      const scatteredTpl = parseTemplateStructure(
        makeBackbonePdb([{ chainId: 'A', sequence: mpnnNat60, bFactors }]),
      );
      const expectedSurviving = bFactors.filter((b) => b >= 50).length;
      const scatteredAlign = alignQueryToTemplate(
        [{ chainId: 'A', sequence: mpnnNat60 }],
        scatteredTpl,
        { minConfidence: 50 },
      );
      expect(scatteredAlign.matchedQueryResidues).toBe(expectedSurviving);
      expect(scatteredAlign.overallIdentity).toBe(1);
    }

    const seq30 = mpnnNat60.slice(0, 30);
    const bFactors30 = Array.from({ length: 30 }, (_, idx) => (idx % 5 === 0 ? 10.0 : 90.0));
    const scattered30Tpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: seq30, bFactors: bFactors30 }]),
    );
    const scattered30Align = alignQueryToTemplate(
      [{ chainId: 'A', sequence: seq30 }],
      scattered30Tpl,
      { minConfidence: 50 },
    );
    expect(scattered30Align.matchedQueryResidues).toBe(24);
    expect(scattered30Align.overallIdentity).toBe(1);

    // q60gap (real 2-aa deletion) + stride=4 low-B-factor template
    const q60gap = mpnnNat60.slice(0, 28) + mpnnNat60.slice(30);
    const bFactors60Stride4 = Array.from({ length: 60 }, (_, idx) => (idx % 4 === 0 ? 10.0 : 90.0));
    const stride4Tpl = parseTemplateStructure(
      makeBackbonePdb([{ chainId: 'A', sequence: mpnnNat60, bFactors: bFactors60Stride4 }]),
    );
    const q60gapAlign = alignQueryToTemplate([{ chainId: 'A', sequence: q60gap }], stride4Tpl, {
      minConfidence: 50,
    });
    expect(q60gapAlign.matchedQueryResidues).toBe(44);
    expect(q60gapAlign.overallIdentity).toBe(1);
    const q60gapPayload = buildAf2TemplatePayload(
      [{ chainId: 'A', sequence: q60gap }],
      q60gapAlign,
      'af2-monomer',
    );
    expect(q60gapPayload.coveredResidues).toBe(q60gapAlign.matchedQueryResidues);
  });

  it('rejects 14–24 aa shared purification tags across all 9 real PDBs while preserving 20-in-200, 50-in-100, 25+ aa domains, and tagged homologs', () => {
    const proteins9: Array<{ name: string; seq: string }> = [
      {
        name: 'ubiquitin_76',
        seq: 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
      },
      {
        name: 'bpti_58',
        seq: 'RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA',
      },
      {
        name: 'barnase_110',
        seq: 'AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR',
      },
      {
        name: 'lysozyme_129',
        seq: 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL',
      },
      {
        name: 'streptavidin_125',
        seq: 'EAGITGTWYNQLGSTFIVTAGADGALTGTYESAVGNAESRYVLTGRYDSAPATDGSGTALGWTVAWKNNYRNAHSATTWSGQYVGGAEARINTQWLLTSGTTEANAWKSTLVGHDTFTKVKPSA',
      },
      {
        name: 'gb1_56',
        seq: 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE',
      },
      {
        name: 'crambin_46',
        seq: 'TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN',
      },
      {
        name: 'villin_35',
        seq: 'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
      },
      {
        name: 'trpcage_20',
        seq: 'NLYIQWLKDGGPSSGRPPPS',
      },
    ];

    const longTags = [
      { name: '14aa_his_gs', seq: 'MGSSHHHHHHGGGG' },
      { name: '16aa_his_gs', seq: 'MGSSHHHHHHGGGGSG' },
      { name: '17aa_thrombin_his', seq: 'MGSSHHHHHHSSGLVPR' },
      { name: '18aa_flag_gs', seq: 'DYKDDDDKGGGGSGGGGS' },
      { name: '20aa_his_thrombin', seq: 'MGSSHHHHHHSSGLVPRGSH' },
      { name: '24aa_twin_strep_like', seq: 'WSHPQFEKGGGSGGGSGGSAWSHP' },
    ];

    // 1. All 14–24 aa shared tags across N & C termini and all 72 directed pairs of the 9 PDBs -> 0 FPs
    for (const tag of longTags) {
      for (const side of ['N', 'C'] as const) {
        const tagged = proteins9.map((p) => ({
          name: p.name,
          seq: side === 'N' ? tag.seq + p.seq : p.seq + tag.seq,
        }));
        const parsedTemplates = tagged.map((p) =>
          parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: p.seq }])),
        );
        for (let i = 0; i < tagged.length; i += 1) {
          for (let j = 0; j < tagged.length; j += 1) {
            if (i === j) continue;
            expect(() =>
              alignQueryToTemplate([{ chainId: 'A', sequence: tagged[j].seq }], parsedTemplates[i]),
              `${tag.name}-${side}: ${tagged[j].name} vs ${tagged[i].name} should throw`,
            ).toThrowError(LocalFoldTemplateError);
          }
        }
      }
    }

    // 2. Dual 20-aa + 18-aa shared tags & minConfidence=50 body masking on 20-aa tag
    const ubqSeq = proteins9.find((p) => p.name === 'ubiquitin_76')!.seq;
    const bptiSeq = proteins9.find((p) => p.name === 'bpti_58')!.seq;
    const villinSeq = proteins9.find((p) => p.name === 'villin_35')!.seq;
    const crambinSeq = proteins9.find((p) => p.name === 'crambin_46')!.seq;
    const trpcageSeq = proteins9.find((p) => p.name === 'trpcage_20')!.seq;

    const dualLongTpl = parseTemplateStructure(
      makeBackbonePdb([
        { chainId: 'A', sequence: 'MGSSHHHHHHSSGLVPRGSH' + ubqSeq + 'DYKDDDDKGGGGSGGGGS' },
      ]),
    );
    expect(() =>
      alignQueryToTemplate(
        [{ chainId: 'A', sequence: 'MGSSHHHHHHSSGLVPRGSH' + bptiSeq + 'DYKDDDDKGGGGSGGGGS' }],
        dualLongTpl,
      ),
    ).toThrowError(LocalFoldTemplateError);

    const tag20 = 'MGSSHHHHHHSSGLVPRGSH';
    const tag20LowTpl = parseTemplateStructure(
      makeBackbonePdb([
        {
          chainId: 'A',
          sequence: tag20 + villinSeq.slice(0, 20),
          bFactors: [...Array(20).fill(90.0), ...Array(20).fill(10.0)],
        },
      ]),
    );
    expect(() =>
      alignQueryToTemplate([{ chainId: 'A', sequence: tag20 + trpcageSeq }], tag20LowTpl, {
        minConfidence: 50,
      }),
    ).toThrowError(LocalFoldTemplateError);

    // 3. Valid positives preserved:
    // 3a. 20-in-200 (both directions) & 50-in-100 (both directions)
    const protein200 =
      ubqSeq +
      'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKIADKLEK' +
      bptiSeq +
      'ACDEFGHIKLMNPQRSTVWY'.slice(0, 6);
    expect(protein200.length).toBe(200);
    const domain20 = protein200.slice(90, 110);
    const r20in200 = alignQueryToTemplate(
      [{ chainId: 'A', sequence: protein200 }],
      parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: domain20 }])),
    );
    expect(r20in200.matchedQueryResidues).toBe(20);
    expect(r20in200.overallIdentity).toBe(1);

    const r200with20 = alignQueryToTemplate(
      [{ chainId: 'A', sequence: domain20 }],
      parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: protein200 }])),
    );
    expect(r200with20.matchedQueryResidues).toBe(20);
    expect(r200with20.overallIdentity).toBe(1);

    const protein100 = protein200.slice(0, 100);
    const domain50 = protein100.slice(25, 75);
    const r50in100 = alignQueryToTemplate(
      [{ chainId: 'A', sequence: protein100 }],
      parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: domain50 }])),
    );
    expect(r50in100.matchedQueryResidues).toBe(50);
    expect(r50in100.overallIdentity).toBe(1);

    // 3b. 25+ aa conserved domain shared across unrelated flanking domains stays intact
    const domain25 = bptiSeq.slice(0, 25);
    const rDom25 = alignQueryToTemplate(
      [{ chainId: 'A', sequence: domain25 + villinSeq }],
      parseTemplateStructure(makeBackbonePdb([{ chainId: 'A', sequence: domain25 + crambinSeq }])),
    );
    expect(rDom25.matchedQueryResidues).toBeGreaterThanOrEqual(25);
    expect(rDom25.overallIdentity).toBe(1);

    // 3c. Differently tagged true homolog (20-aa pET-28a tag vs 12-aa Strep tag on bpti_58)
    const pet28BptiAlign = alignQueryToTemplate(
      [{ chainId: 'A', sequence: 'MGSSHHHHHHSSGLVPRGSH' + bptiSeq }],
      parseTemplateStructure(
        makeBackbonePdb([{ chainId: 'A', sequence: 'WSHPQFEKKGGW' + bptiSeq }]),
      ),
    );
    expect(pet28BptiAlign.matchedQueryResidues).toBe(58);
    expect(pet28BptiAlign.overallIdentity).toBe(1);
    expect(pet28BptiAlign.chainAlignments[0].residueMap.get(20)).toBe(12);
    expect(pet28BptiAlign.chainAlignments[0].residueMap.get(77)).toBe(69);
  });

  it('Tier 5 adversarial (Gaps 1, 2, 3): preserves original queryChainIndex across empty chains, enforces STAR/CIF stop_/save_ & case-insensitivity, and checks inline template integrity', () => {
    const seqA = 'MKTAYIAKQRQISFVKSHFSRQ';
    const seqB = 'ACDEFGHIKLMNPQRSTVWYAA';

    // Gap 2: leading and embedded empty chains preserve queryChainIndex -> buildAf2TemplatePayload & buildAf3TemplateSlots
    const parsedAB = parseTemplateStructure(
      makeBackbonePdb([
        { chainId: 'A', sequence: seqA },
        { chainId: 'B', sequence: seqB },
      ]),
    );

    const leadEmptyAlign = alignQueryToTemplate(
      [
        { chainId: 'A', sequence: '' },
        { chainId: 'B', sequence: seqB },
      ],
      parsedAB,
    );
    expect(leadEmptyAlign.chainAlignments).toHaveLength(1);
    expect(leadEmptyAlign.chainAlignments[0].queryChainIndex).toBe(1);
    expect(leadEmptyAlign.chainAlignments[0].templateChainId).toBe('B');

    const leadAf2 = buildAf2TemplatePayload(['', seqB], leadEmptyAlign, 'af2-multimer', {
      spanChains: true,
    });
    expect(leadAf2.totalResidues).toBe(seqB.length);
    expect(leadAf2.coveredResidues).toBe(seqB.length);
    expect(leadAf2.template.aatype[0]).not.toBe(GAP_AATYPE);
    expect(leadAf2.template.atomMask[0 * 37 + 1]).toBe(1);

    const leadAf3 = buildAf3TemplateSlots(seqB.length, leadEmptyAlign, 'af3', { spanChains: true });
    expect(leadAf3[0].covered).toBe(seqB.length);
    expect(leadAf3[0].aatype[0]).not.toBe(GAP_AATYPE);

    const midEmptyAlign = alignQueryToTemplate(
      [
        { chainId: 'A', sequence: seqA },
        { chainId: 'B', sequence: '' },
        { chainId: 'C', sequence: seqB },
      ],
      parsedAB,
    );
    expect(midEmptyAlign.chainAlignments.map((c) => c.queryChainIndex)).toEqual([0, 2]);
    const midAf2 = buildAf2TemplatePayload([seqA, '', seqB], midEmptyAlign, 'af2-multimer', {
      spanChains: true,
    });
    expect(midAf2.totalResidues).toBe(seqA.length + seqB.length);
    expect(midAf2.coveredResidues).toBe(seqA.length + seqB.length);
    expect(midAf2.template.aatype[seqA.length]).not.toBe(GAP_AATYPE);
    expect(midAf2.asymId[0]).toBe(0);
    expect(midAf2.asymId[seqA.length]).toBe(2);

    const midAf3 = buildAf3TemplateSlots(seqA.length + seqB.length, midEmptyAlign, 'boltz2', {
      spanChains: true,
    });
    expect(midAf3[0].covered).toBe(seqA.length + seqB.length);
    expect(midAf3[0].aatype[seqA.length]).not.toBe(GAP_AATYPE);

    // Gap 3a: stop_ and save_ terminators break _atom_site loop (no phantom residue leak)
    const cifWithStop = [
      'data_stop_test',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 N N MET A 1 0.0 0.0 0.0',
      'ATOM 2 C CA MET A 1 1.4 0.0 0.0',
      'ATOM 3 C C MET A 1 2.0 1.3 0.0',
      'stop_',
      'X Y Z W P Q R S T ATOM 99 C CA TRP A 2 9.0 9.0 9.0',
      'ATOM 100 N N TRP A 2 8.0 9.0 9.0',
    ].join('\n');
    const parsedStop = parseTemplateStructure(cifWithStop);
    expect(parsedStop.chains.map((c) => `${c.chainId}:${c.sequence}`)).toEqual(['A:M']);

    const cifWithSave = [
      'data_save_test',
      'save_before_loop',
      '_dict.item foo',
      'save_',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 C CA ALA A 1 1.0 2.0 3.0',
      'SAVE_next_frame',
      'ATOM 2 C CA TRP A 2 4.0 5.0 6.0',
    ].join('\n');
    const parsedSave = parseTemplateStructure(cifWithSave);
    expect(parsedSave.chains.map((c) => `${c.chainId}:${c.sequence}`)).toEqual(['A:A']);

    // Gap 3b: Uppercase CIF keywords/data-names and quoted keyword-like values
    const cifUpperQuoted = [
      'DATA_UPPER',
      'LOOP_',
      '_ATOM_SITE.GROUP_PDB',
      '_ATOM_SITE.ID',
      '_ATOM_SITE.TYPE_SYMBOL',
      '_ATOM_SITE.LABEL_ATOM_ID',
      '_ATOM_SITE.LABEL_ALT_ID',
      '_ATOM_SITE.LABEL_COMP_ID',
      '_ATOM_SITE.AUTH_ASYM_ID',
      '_ATOM_SITE.AUTH_SEQ_ID',
      '_ATOM_SITE.CARTN_X',
      '_ATOM_SITE.CARTN_Y',
      '_ATOM_SITE.CARTN_Z',
      "ATOM 1 C CA 'stop_' MET A 1 1.0 2.0 3.0",
      'ATOM 2 C CA "save_v" ALA A 2 4.8 2.0 3.0',
      'STOP_',
      'ATOM 3 C CA . TRP A 3 8.6 2.0 3.0',
    ].join('\n');
    const parsedUpper = parseTemplateStructure(cifUpperQuoted);
    expect(parsedUpper.format).toBe('mmcif');
    expect(parsedUpper.chains.map((c) => `${c.chainId}:${c.sequence}`)).toEqual(['A:MA']);

    // Gap 1: assertInlineTemplateIntegrity unit contract
    const pdbOne = makeBackbonePdb([{ chainId: 'A', sequence: seqA }]);
    expect(() =>
      assertInlineTemplateIntegrity({ missingTemplateUpload: true }, false),
    ).toThrowError(LocalFoldTemplateError);
    expect(() =>
      assertInlineTemplateIntegrity(
        { templatePdbText: pdbOne, expectedTemplateSha256: '0'.repeat(64) },
        false,
      ),
    ).toThrowError(LocalFoldTemplateError);
    expect(() =>
      assertInlineTemplateIntegrity(
        { missingTemplateUpload: true, expectedTemplateSha256: 'bad-hex' },
        true,
      ),
    ).not.toThrow();
  });

  it('Tier 5 R2 adversarial (Findings 1 & 2): classifies PDB mentioning _atom_site. in REMARK/TITLE/HEADER as pdb and preserves mmCIF column alignment across semicolon text blocks', () => {
    // Finding 1: PDB mentioning _atom_site. and loop_ in REMARK/TITLE/HEADER must detect as 'pdb' and parse all atoms
    const pdbWithCifKeywords = [
      'HEADER    SYNTHETIC _atom_site.group_PDB TEST',
      'TITLE     CONVERTED FROM _atom_site.id',
      'REMARK   1 Generated from _atom_site.Cartn_x and loop_',
      makePdbLine({ serial: 1, atomName: 'N', resName: 'MET', chainId: 'A', resSeq: 1, x: 0.0, y: 0.0, z: 0.0, element: 'N' }),
      makePdbLine({ serial: 2, atomName: 'CA', resName: 'MET', chainId: 'A', resSeq: 1, x: 1.4, y: 0.0, z: 0.0, element: 'C' }),
      makePdbLine({ serial: 3, atomName: 'C', resName: 'MET', chainId: 'A', resSeq: 1, x: 2.0, y: 1.3, z: 0.0, element: 'C' }),
      makePdbLine({ serial: 4, atomName: 'O', resName: 'MET', chainId: 'A', resSeq: 1, x: 1.5, y: 2.4, z: 0.0, element: 'O' }),
      'END',
    ].join('\n');
    expect(detectTemplateFormat(pdbWithCifKeywords)).toBe('pdb');
    const parsedPdb = parseTemplateStructure(pdbWithCifKeywords);
    expect(parsedPdb.format).toBe('pdb');
    expect(parsedPdb.chains.map((c) => `${c.chainId}:${c.sequence}`)).toEqual(['A:M']);
    expect([...parsedPdb.chains[0].residues[0].atoms.keys()]).toEqual(['N', 'CA', 'C', 'O']);

    // Headless mmCIF & leading-comment mmCIF still classify as 'mmcif'
    const headlessCif = [
      '# leading comment',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      'ATOM 1 C CA ALA A 1 1.0 2.0 3.0',
    ].join('\n');
    expect(detectTemplateFormat(headlessCif)).toBe('mmcif');
    expect(detectTemplateFormat('loop_ _atom_site.group_PDB _atom_site.id')).toBe('mmcif');

    // Finding 2: mmCIF _atom_site loop with ;...; semicolon multiline text blocks keeps column alignment
    const cifSemicolonAtoms = [
      'data_semicolon_atom_loop',
      '_citation.title',
      ';Title mentioning _atom_site.Cartn_x and',
      'stop_',
      ';',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.details',
      'ATOM 1 N N ALA A 1 0.0 0.0 0.0',
      ';note line 1',
      ';',
      'ATOM 2 C CA ALA A 1 1.4 0.0 0.0',
      ';embedded stop_ and data_fake inside semicolon block',
      'second line',
      ';',
      'ATOM 3 C C ALA A 1 2.0 1.3 0.0 .',
      'ATOM 4 O O ALA A 1 1.5 2.4 0.0 ?',
      'ATOM 5 C CB ALA A 1 1.4 -1.2 0.5 "quoted"',
    ].join('\n');
    const parsedSemicolon = parseTemplateStructure(cifSemicolonAtoms);
    expect(parsedSemicolon.format).toBe('mmcif');
    expect(parsedSemicolon.chains).toHaveLength(1);
    expect(parsedSemicolon.chains[0].sequence).toBe('A');
    expect([...parsedSemicolon.chains[0].residues[0].atoms.keys()]).toEqual([
      'N',
      'CA',
      'C',
      'O',
      'CB',
    ]);
    expect(parsedSemicolon.chains[0].residues[0].atoms.get('CB')).toEqual([1.4, -1.2, 0.5]);
  });
});
