import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  NUMERIC_SPECS,
  commitNumericDraft,
  computeUploadDigest,
  isValidResidueToken,
  isValidSha256Hex,
  sanitizeNumericDraft,
  validateJsonOverride,
  validateNodeParams,
  validateNumericDraft,
  validateSequenceText,
  validateStructureText,
  validateTiedPositions,
} from '../engine/param-validation';
import { parseTiedPositions } from '../engine/mpnn-spec';
import { generateSampleHelixBundlePdb } from '../protein-ops/pdb';
import { defaultParamsForAdapter } from '../engine/graph-utils';

describe('parameter validation & numeric sanitization', () => {
  it('rejects fractional count parameters but preserves continuous parameters', () => {
    expect(validateNodeParams({ id: 'g', adapterId: 'genie3', params: { length: 80.5 } }).join(' '))
      .toMatch(/integer/);
    expect(validateNodeParams({ id: 'm', adapterId: 'web-protein-mpnn', params: { numSequences: 1.5 } }).join(' '))
      .toMatch(/integer/);
    expect(validateNodeParams({ id: 'm', adapterId: 'web-protein-mpnn', params: { temperature: 0.15 } }))
      .toEqual([]);
  });

  it('strips stray letters like "5r" or scientific notation "1e5" immediately', () => {
    expect(sanitizeNumericDraft('5r', true)).toBe('5');
    expect(sanitizeNumericDraft('70aa', true)).toBe('70');
    expect(sanitizeNumericDraft('1e5', true)).toBe('15');
    expect(sanitizeNumericDraft('0.15r', false)).toBe('0.15');
    expect(sanitizeNumericDraft('1.2.3', false)).toBe('1.23');
    expect(sanitizeNumericDraft('-42', true)).toBe('42');
  });

  it('flags out-of-range numeric drafts while typing and clamps on commit', () => {
    const lenSpec = NUMERIC_SPECS['rfd4-proteina'].length;
    expect(validateNumericDraft('5', lenSpec).status).toBe('error');
    expect(validateNumericDraft('500', lenSpec).status).toBe('error');
    expect(validateNumericDraft('', lenSpec).status).toBe('error');
    expect(validateNumericDraft('70', lenSpec).status).toBe('ok');

    expect(commitNumericDraft('5', lenSpec, 70)).toBe(30);
    expect(commitNumericDraft('500', lenSpec, 70)).toBe(256);
    expect(commitNumericDraft('', lenSpec, 70)).toBe(70);
  });

  it('warns on RFD4 nsteps below 58 due to vf_tsr instability', () => {
    const stepSpec = NUMERIC_SPECS['rfd4-proteina'].nsteps;
    expect(validateNumericDraft('30', stepSpec).status).toBe('warn');
    expect(validateNumericDraft('60', stepSpec).status).toBe('ok');
  });

  it('validates plain, multi-FASTA, multi-chain, and invalid amino acid sequences', () => {
    const ok = validateSequenceText('>seq1\nMKTAYIAKQRQISFVK');
    expect(ok.status).toBe('ok');
    expect(ok.numSequences).toBe(1);
    expect(ok.totalResidues).toBe(16);

    const multiChain = validateSequenceText('MKTAYIAKQR:QISFVKSHFS');
    expect(multiChain.status).toBe('ok');
    expect(multiChain.chainsInFirst).toBe(2);

    const bad = validateSequenceText('MKTA5X');
    expect(bad.status).toBe('error');
    expect(bad.illegalChars).toEqual(['5', 'X']);

    const short = validateSequenceText('MKTA');
    expect(short.status).toBe('error');

    const colonsOnly = validateSequenceText('::::');
    expect(colonsOnly.status).toBe('error');
  });

  it('validates residue tokens, JSON overrides, and PDB/mmCIF structure text', () => {
    expect(isValidResidueToken('A54')).toBe(true);
    expect(isValidResidueToken('A16-35')).toBe(true);
    expect(isValidResidueToken('10-14')).toBe(true);
    expect(isValidResidueToken('A100A')).toBe(true);
    expect(isValidResidueToken('A100A-100C')).toBe(true);
    expect(isValidResidueToken('A-2')).toBe(true);
    expect(isValidResidueToken('A-5--2')).toBe(true);
    expect(isValidResidueToken('A-2-5')).toBe(true);
    expect(isValidResidueToken('A999-1')).toBe(false);
    expect(isValidResidueToken('A10B-10A')).toBe(false);
    expect(isValidResidueToken('A5r')).toBe(false);
    expect(isValidResidueToken('bad!')).toBe(false);

    expect(validateJsonOverride('').status).toBe('ok');
    expect(validateJsonOverride('{"contig":"70-70"}').status).toBe('ok');
    expect(validateJsonOverride('{broken').status).toBe('error');
    expect(validateJsonOverride('[1,2]').status).toBe('error');

    expect(validateStructureText('').status).toBe('error');
    expect(validateStructureText('hello world').status).toBe('error');
    expect(validateStructureText('ATOM      1  CA  ALA A   1       0.000   0.000   0.000').status).toBe('error');
    expect(validateStructureText('loop_\n_atom_site.group_PDB').status).toBe('error');
    const threeCaPdb = [
      'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  ',
      'ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C  ',
      'ATOM      3  CA  VAL A   3       7.600   0.000   0.000  1.00  0.00           C  ',
      'END',
    ].join('\n');
    expect(validateStructureText(threeCaPdb).status).toBe('ok');
  });

  it('reports blocking parameter errors from validateNodeParams', () => {
    expect(
      validateNodeParams({
        id: 'n1',
        adapterId: 'rfd4-proteina',
        label: 'RFD4',
        params: { mode: 'unconditional', length: 5 },
      }),
    ).toHaveLength(1);

    expect(
      validateNodeParams({
        id: 'n2',
        adapterId: 'sequence-input',
        label: 'Seq',
        params: { sequence: '12345' },
      }),
    ).toHaveLength(1);

    expect(
      validateNodeParams({
        id: 'n3',
        adapterId: 'rfd4-proteina',
        label: 'RFD4',
        params: { mode: 'motif', customSpecYaml: '{bad' },
      }),
    ).toHaveLength(1);
  });

  it('validates RFD4 binder length and enum params across all model blocks', () => {
    // Finding #2: validate length in binder mode
    expect(
      validateNodeParams({
        id: 'b1',
        adapterId: 'rfd4-proteina',
        label: 'RFD4 Binder',
        params: { mode: 'binder', length: 5 },
      }),
    ).toHaveLength(1);
    expect(
      validateNodeParams({
        id: 'b2',
        adapterId: 'rfd4-proteina',
        label: 'RFD4 Binder',
        params: { mode: 'binder', length: 70 },
      }),
    ).toEqual([]);

    // Finding #29: enum parameter validation (now unified via validateParamsWithSchema + crossValidate)
    expect(
      validateNodeParams({
        id: 'e1',
        adapterId: 'rfd4-proteina',
        params: { mode: 'bogus_mode' },
      }).join(' '),
    ).toMatch(/Unknown mode "bogus_mode"/);
    expect(
      validateNodeParams({
        id: 'e2',
        adapterId: 'rfd4-proteina',
        params: { mode: 'preset', presetId: 'nonexistent_preset' },
      }).join(' '),
    ).toMatch(/Invalid presetId "nonexistent_preset"/);
    expect(
      validateNodeParams({
        id: 'e3',
        adapterId: 'genie3',
        params: { taskMode: 'ligand' },
      }).join(' '),
    ).toMatch(/Unknown taskMode "ligand"/);
    expect(
      validateNodeParams({
        id: 'e4',
        adapterId: 'web-protein-mpnn',
        params: { family: 'unknown_mpnn' },
      }).join(' '),
    ).toMatch(/Invalid family "unknown_mpnn"/);
    expect(
      validateNodeParams({
        id: 'e5',
        adapterId: 'web-protein-mpnn',
        params: { family: 'solublempnn', modelName: 'v_48_030' },
      }).join(' '),
    ).toMatch(/Unknown solublempnn modelName/);
    expect(
      validateNodeParams({
        id: 'e6',
        adapterId: 'localfold-af2',
        params: { model: 'af4_turbo' },
      }).join(' '),
    ).toMatch(/Invalid model "af4_turbo"/);
  });

  it.each(['ef2-fast-300m', 'ef2-fast-600m'])('validates the active inputs and schedule for %s', (model) => {
    const validate = (params: Record<string, unknown>) =>
      validateNodeParams({ id: 'fold', adapterId: 'localfold-af2', params: { model, ...params } });
    expect(validate({})).toEqual([]);
    expect(validate({ esmfold2Steps: '32', recycles: 19 })).toEqual([]);
    expect(validate({ esmfold2Steps: 25 }).join(' ')).toContain('diffusion schedule');
    expect(validate({ msaMode: 'mmseqs2' }).join(' ')).toContain('single-sequence');
    expect(validate({ ligandCcdCodes: 'ZN, DHZ' })).toEqual([]);
  });

  it('mpnn.worker.ts supports whole-chain locking, designChains, inherited locks, omitAminoAcids, tiedPositions, and aligned bounds', () => {
    const mpnnSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/web-protein-mpnn/mpnn.worker.ts'), 'utf8');
    expect(mpnnSrc).toContain('parseFixedPositionsAndChains');
    expect(mpnnSrc).toContain('structure.lockedChains');
    expect(mpnnSrc).toContain('structure.fixedResidueIndices');
    expect(mpnnSrc).toContain('designChains');
    expect(mpnnSrc).toContain('omitAminoAcids');
    expect(mpnnSrc).toContain('tiedPositions');
    expect(mpnnSrc).toContain('buildGapAwareResidueIndices');
    // Bounds aligned with NUMERIC_SPECS['web-protein-mpnn'] (1..8, 0.01..1.5)
    expect(mpnnSrc).toContain('Math.min(8,');
    expect(mpnnSrc).toContain('Math.min(1.5,');
  });

  it('computes exact FIPS 180-4 SHA-256 upload digests matching Node crypto', () => {
    const samples = [
      '',
      'abc',
      'The quick brown fox jumps over the lazy dog',
      'Ångström α/β motif — 1PRW',
      generateSampleHelixBundlePdb(),
    ];
    for (const s of samples) {
      const expected = createHash('sha256').update(s, 'utf8').digest('hex');
      const actual = computeUploadDigest(s);
      expect(actual).toBe(expected);
      expect(isValidSha256Hex(actual)).toBe(true);
    }
    expect(isValidSha256Hex('not-a-sha')).toBe(false);
  });

  it('blocks missing or mismatched custom PDB uploads until matching digest is uploaded or explicitly cleared (Issue #1)', () => {
    const pdbA = generateSampleHelixBundlePdb();
    const pdbB = pdbA + '\nREMARK altered\n';
    const shaA = computeUploadDigest(pdbA);

    // 1. Missing custom upload blocks with filename + digest details
    const missingErrs = validateNodeParams({
      id: 'src',
      adapterId: 'pdb-input',
      label: 'Custom Target',
      params: {
        mode: 'custom',
        pdbText: '',
        filename: 'target.pdb',
        expectedFilename: 'target.pdb',
        missingUpload: true,
        expectedPdbSha256: shaA,
      },
    });
    expect(missingErrs).toHaveLength(1);
    expect(missingErrs[0]).toMatch(/target\.pdb/);
    expect(missingErrs[0]).toContain(shaA.slice(0, 12));

    // 2. Wrong re-uploaded file blocks with digest mismatch
    const mismatchErrs = validateNodeParams({
      id: 'src',
      adapterId: 'pdb-input',
      label: 'Custom Target',
      params: {
        mode: 'custom',
        pdbText: pdbB,
        filename: 'other.pdb',
        expectedFilename: 'target.pdb',
        missingUpload: true,
        expectedPdbSha256: shaA,
      },
    });
    expect(mismatchErrs).toHaveLength(1);
    expect(mismatchErrs[0]).toMatch(/does not match expected custom upload/i);
    expect(mismatchErrs[0]).toMatch(/target\.pdb/);

    // 3. Matching re-uploaded file passes even while NodeInspector leaves missingUpload/expectedPdbSha256 attached
    expect(
      validateNodeParams({
        id: 'src',
        adapterId: 'pdb-input',
        label: 'Custom Target',
        params: {
          mode: 'custom',
          pdbText: pdbA,
          filename: 'target.pdb',
          expectedFilename: 'target.pdb',
          missingUpload: true,
          expectedPdbSha256: shaA,
        },
      }),
    ).toEqual([]);

    // 4. Explicitly clearing expectedPdbSha256 allows replacing with a different valid PDB
    expect(
      validateNodeParams({
        id: 'src',
        adapterId: 'pdb-input',
        label: 'Custom Target',
        params: {
          mode: 'custom',
          pdbText: pdbB,
          filename: 'other.pdb',
          missingUpload: false,
          expectedPdbSha256: '',
        },
      }),
    ).toEqual([]);

    // 5. Explicitly switching mode to demo-bundle or valid pdb-id unblocks validation
    expect(
      validateNodeParams({
        id: 'src',
        adapterId: 'pdb-input',
        params: {
          mode: 'demo-bundle',
          pdbText: '',
          missingUpload: true,
          expectedPdbSha256: shaA,
        },
      }),
    ).toEqual([]);

    // 6. Malformed expectedPdbSha256 or unknown pdb-input mode is rejected
    expect(
      validateNodeParams({
        id: 'src',
        adapterId: 'pdb-input',
        params: { mode: 'custom', pdbText: pdbA, expectedPdbSha256: 'bad-digest' },
      }).join(' '),
    ).toMatch(/Invalid expectedPdbSha256/i);
    expect(
      validateNodeParams({
        id: 'src',
        adapterId: 'pdb-input',
        params: { mode: 'bogus' },
      }).join(' '),
    ).toMatch(/Unknown mode "bogus"/i);
  });

  it('reports every unresolvedImportIssues entry and blocks execution until resolved (Issue #1)', () => {
    const errs = validateNodeParams({
      id: 'g3',
      adapterId: 'genie3',
      label: 'Genie 3',
      params: {
        taskMode: 'unconditional',
        unresolvedImportIssues: [
          'Legacy CATH fold conditioning (cathCode="3.40.50") is not supported by Genie 3.',
          'Legacy H-bond conditioning (hbondMode="ACCEPTOR") is not supported by Genie 3.',
        ],
      },
    });
    expect(errs).toHaveLength(2);
    expect(errs[0]).toMatch(/CATH.*3\.40\.50/);
    expect(errs[1]).toMatch(/H-bond.*ACCEPTOR/);

    // Non-array malformed unresolvedImportIssues (e.g. object) also blocks cleanly with JSON representation
    const objErrs = validateNodeParams({
      id: 'g3',
      adapterId: 'genie3',
      label: 'Genie 3',
      params: {
        taskMode: 'unconditional',
        unresolvedImportIssues: { field: 'cathCode' },
      },
    });
    expect(objErrs).toHaveLength(1);
    expect(objErrs[0]).toContain('{"field":"cathCode"}');

    // Clearing unresolvedImportIssues removes the block
    expect(
      validateNodeParams({
        id: 'g3',
        adapterId: 'genie3',
        label: 'Genie 3',
        params: {
          taskMode: 'unconditional',
          unresolvedImportIssues: [],
        },
      }),
    ).toEqual([]);
  });

  it('Wave-2 P1/P2 regressions: Genie 3 empty motif & directionScale 0.0, MPNN fixedPositions/tiedPositions, RFD4 motif fallback length & customSpecYaml', () => {
    // 1. Genie 3 directionScale allows 0.0 (unconditional guidance)
    expect(NUMERIC_SPECS.genie3.directionScale.min).toBe(0);
    expect(
      validateNodeParams({
        id: 'g3',
        adapterId: 'genie3',
        params: { taskMode: 'unconditional', length: 80, directionScale: 0 },
      }),
    ).toEqual([]);

    // 2. Genie 3 motif mode with BOTH motifSpan and contig empty is blocked
    expect(
      validateNodeParams({
        id: 'g3-empty-motif',
        adapterId: 'genie3',
        label: 'Genie 3',
        params: { taskMode: 'motif', length: 80, motifSpan: '', contig: '' },
      }).join(' '),
    ).toMatch(/motifSpan.*contig/i);

    // 3. ProteinMPNN fixedPositions & tiedPositions validation (accepts insCodes & negative resnums, rejects reversed/malformed)
    expect(
      validateNodeParams({
        id: 'mpnn-ok',
        adapterId: 'web-protein-mpnn',
        params: { fixedPositions: 'A100A, A-2, A10-20', tiedPositions: 'A1=B1; A2=B2' },
      }),
    ).toEqual([]);
    expect(
      validateNodeParams({
        id: 'mpnn-bad-fixed',
        adapterId: 'web-protein-mpnn',
        params: { fixedPositions: 'A999-1' },
      }).join(' '),
    ).toMatch(/fixedPositions/i);
    expect(
      validateNodeParams({
        id: 'mpnn-bad-tied',
        adapterId: 'web-protein-mpnn',
        params: { tiedPositions: 'A1, ???' },
      }).join(' '),
    ).toMatch(/tiedPositions/i);

    // 4. RFD4 motif mode validates fallback length when contig & customSpecYaml are empty, and validates customSpecYaml
    expect(
      validateNodeParams({
        id: 'rfd-motif-badlen',
        adapterId: 'rfd4-proteina',
        params: { mode: 'motif', length: 5, contig: '', customSpecYaml: '' },
      }).join(' '),
    ).toMatch(/length/i);
    expect(
      validateNodeParams({
        id: 'rfd-yaml-ok',
        adapterId: 'rfd4-proteina',
        params: { mode: 'motif', customSpecYaml: 'contig: 70-70\nconditions: {}' },
      }),
    ).toEqual([]);
    expect(
      validateNodeParams({
        id: 'rfd-yaml-bad',
        adapterId: 'rfd4-proteina',
        params: { mode: 'motif', customSpecYaml: 'not a valid yaml mapping' },
      }).join(' '),
    ).toMatch(/customSpecYaml/i);
  });

  it('Wave-3 validation: RFD4 high recycle & earlyStop default, LocalFold recycles 10/20 & MSA modes, scorer gate specs, and worker provenance contracts', () => {
    // 1. RFD4 recycleSchedule accepts 'high' and defaults earlyStop: false in graph-utils
    expect(defaultParamsForAdapter('rfd4-proteina').earlyStop).toBe(false);
    expect(
      validateNodeParams({
        id: 'rfd-hi',
        adapterId: 'rfd4-proteina',
        params: { recycleSchedule: 'high', nsteps: 60 },
      }),
    ).toEqual([]);
    expect(
      validateNodeParams({
        id: 'rfd-bad-rec',
        adapterId: 'rfd4-proteina',
        params: { recycleSchedule: 'ultra' },
      }).join(' '),
    ).toMatch(/recycleSchedule/);

    // 2. LocalFold recycles cap: <= 10 for standard models, <= 20 for mol1 & multimer*
    expect(NUMERIC_SPECS['localfold-af2'].recycles.max).toBe(20);
    expect(
      validateNodeParams({ id: 'lf1', adapterId: 'localfold-af2', params: { model: 'monomer', recycles: 10, tolerance: 0 } }),
    ).toEqual([]);
    expect(
      validateNodeParams({ id: 'lf2', adapterId: 'localfold-af2', params: { model: 'monomer', recycles: 11 } }).join(' '),
    ).toMatch(/recycles must be <= 10/);
    expect(
      validateNodeParams({ id: 'lf3', adapterId: 'localfold-af2', params: { model: 'mol1', recycles: 20 } }),
    ).toEqual([]);
    expect(
      validateNodeParams({ id: 'lf4', adapterId: 'localfold-af2', params: { model: 'multimer-2', recycles: 20 } }),
    ).toEqual([]);
    expect(
      validateNodeParams({ id: 'lf5', adapterId: 'localfold-af2', params: { model: 'mol1', recycles: 21 } }).join(' '),
    ).toMatch(/recycles must be between 0 and 20/);

    // 3. LocalFold msaMode enum & required msaA3mText when msaMode === 'a3m'
    for (const m of ['single', 'mmseqs2', 'target-only']) {
      expect(
        validateNodeParams({ id: `msa-${m}`, adapterId: 'localfold-af2', params: { msaMode: m } }),
      ).toEqual([]);
    }
    expect(
      validateNodeParams({ id: 'msa-bad', adapterId: 'localfold-af2', params: { msaMode: 'blast' } }).join(' '),
    ).toMatch(/Unknown msaMode "blast"/);
    expect(
      validateNodeParams({ id: 'msa-empty-a3m', adapterId: 'localfold-af2', params: { msaMode: 'a3m', msaA3mText: '   ' } }).join(' '),
    ).toMatch(/msaA3mText/);
    expect(
      validateNodeParams({
        id: 'msa-ok-a3m',
        adapterId: 'localfold-af2',
        params: { msaMode: 'a3m', msaA3mText: '>q\nACDEFGHIKL\n' },
      }),
    ).toEqual([]);

    // 4. Scorer numeric specs for motifRmsdThreshold, ipaeThreshold, ipsaeThreshold, targetAlignedBinderRmsdThreshold
    expect(NUMERIC_SPECS['designability-scorer'].motifRmsdThreshold.defaultValue).toBe(1.0);
    expect(NUMERIC_SPECS['designability-scorer'].ipaeThreshold.defaultValue).toBe(10.0);
    expect(NUMERIC_SPECS['designability-scorer'].ipsaeThreshold.defaultValue).toBe(0);
    expect(NUMERIC_SPECS['designability-scorer'].targetAlignedBinderRmsdThreshold.defaultValue).toBe(4.0);
    expect(
      validateNodeParams({
        id: 'sc-ok',
        adapterId: 'designability-scorer',
        params: { motifRmsdThreshold: 1.0, ipaeThreshold: 10.0, ipsaeThreshold: 0.6, targetAlignedBinderRmsdThreshold: 4.0 },
      }),
    ).toEqual([]);
    expect(
      validateNodeParams({
        id: 'sc-bad-ipsae',
        adapterId: 'designability-scorer',
        params: { ipsaeThreshold: 1.5 },
      }).join(' '),
    ).toMatch(/ipsaeThreshold/);

    // 5. Worker fidelity & metadata contracts (RFD4, Genie 3, MPNN)
    const rfd4Src = readFileSync(resolve(process.cwd(), 'src/lib/workers/rfd4.worker.ts'), 'utf8');
    expect(rfd4Src).toContain("mode === 'inverse-folding' ? false : requestedEarlyStop");
    expect(rfd4Src).toContain("bfactorSemantics: mode === 'partial' || mode === 'inverse-folding' ? 'crystal-b' : 'none'");
    expect(rfd4Src).toContain("atomGranularity: 'all-atom'");
    expect(rfd4Src).toContain("quantization: 'int8'");
    expect(rfd4Src).toContain('degradedFallback');
    expect(rfd4Src).toContain('designMeta.lockedChains');
    expect(rfd4Src).toContain('designMeta.chainRoles');

    const genie3Src = readFileSync(resolve(process.cwd(), 'src/lib/workers/genie3.worker.ts'), 'utf8');
    expect(genie3Src).toContain("bfactorSemantics: 'none'");
    expect(genie3Src).toContain("atomGranularity: 'ca-only'");
    expect(genie3Src).toContain('...(lockedChains ? { lockedChains } : {})');
    expect(genie3Src).toContain('...(chainRoles ? { chainRoles } : {})');

    const mpnnSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/web-protein-mpnn/mpnn.worker.ts'), 'utf8');
    expect(mpnnSrc).toContain('propagatedLockedChains');
    expect(mpnnSrc).toContain('structure.chainRoles');
  });

  it('Wave-4 validation: web-protein-mpnn backboneNoise & omitAminoAcids bounds and schema-driven UI wiring', () => {
    expect(NUMERIC_SPECS['web-protein-mpnn'].backboneNoise).toMatchObject({
      min: 0,
      max: 1,
      defaultValue: 0,
    });
    expect(
      validateNodeParams({
        id: 'mpnn-adv-ok',
        adapterId: 'web-protein-mpnn',
        params: {
          family: 'proteinmpnn',
          modelName: 'v_48_030',
          backboneNoise: 0.2,
          omitAminoAcids: 'C, M',
          designChains: 'A',
          tiedPositions: 'A10=B10',
        },
      }),
    ).toEqual([]);
    expect(
      validateNodeParams({
        id: 'mpnn-noise-high',
        adapterId: 'web-protein-mpnn',
        params: { backboneNoise: 1.5 },
      }).join(' '),
    ).toMatch(/backboneNoise/);
    expect(
      validateNodeParams({
        id: 'mpnn-bad-omit',
        adapterId: 'web-protein-mpnn',
        params: { omitAminoAcids: 'C1Z' },
      }).join(' '),
    ).toMatch(/omitAminoAcids/);

    const inspectorSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/NodeInspector.svelte'), 'utf8');
    expect(inspectorSrc).toContain('SchemaParamFields');
    expect(inspectorSrc).toContain('getVisibleParamFields');
    expect(inspectorSrc).toContain('getActiveModeSpec');
  });

  it('Chrome QA regressions: NumericInput untrack, Sequence Input invalid residue / empty FASTA, RecipeDrawer format switch, and all-20-AA omitAminoAcids', async () => {
    // 1. NumericInput must untrack module-level invalidNumericCount updates inside $effect to avoid effect_update_depth_exceeded
    const numSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/NumericInput.svelte'), 'utf8');
    expect(numSrc).toMatch(/untrack\(\(\)\s*=>\s*\{\s*invalidNumericCount\s*\+=\s*1;/);
    expect(numSrc).toMatch(/untrack\(\(\)\s*=>\s*\{\s*invalidNumericCount\s*=\s*Math\.max/);

    // 2. Sequence Input must NEVER silently strip illegal residues like 'X' during materialization
    const { materializeSequenceInputNode } = await import('$lib/engine/scheduler');
    expect(() =>
      materializeSequenceInputNode({
        id: 'seq-bad',
        adapterId: 'sequence-input',
        params: { sequence: 'ACDEFGXHIK' },
      }),
    ).toThrow(/Invalid residue character.*X/i);

    // 3. Trailing >empty FASTA header warns and counts only non-empty records
    const emptyCheck = validateSequenceText('>seq1\nACDEFGHIK\n>empty\n');
    expect(emptyCheck.status).toBe('warn');
    expect(emptyCheck.numSequences).toBe(1);
    expect(emptyCheck.emptyHeaders).toEqual(['empty']);
    expect(emptyCheck.message).toMatch(/>empty/);

    // 4. Excluding all 20 standard amino acids is rejected pre-run across MPNN adapters
    const all20 = 'ACDEFGHIKLMNPQRSTVWY';
    for (const adapterId of ['web-protein-mpnn', 'ligandmpnn', 'lasermpnn'] as const) {
      const errs = validateNodeParams({
        id: `omit20-${adapterId}`,
        adapterId,
        params: { omitAminoAcids: all20 },
      });
      expect(errs.join(' '), `Expected all-20 omit error for ${adapterId}`).toMatch(
        /excludes all 20 standard amino acids/i,
      );
    }
    // Positive control: excluding 19 AAs (leaving G allowed) passes
    expect(
      validateNodeParams({
        id: 'omit19-ok',
        adapterId: 'web-protein-mpnn',
        params: { omitAminoAcids: 'ACDEFHIKLMNPQRSTVWY' },
      }),
    ).toEqual([]);

    // 5. RecipeDrawer preserves valid unapplied edits across YAML <-> JSON and guards syntax-error discards
    const recipeSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/RecipeDrawer.svelte'), 'utf8');
    expect(recipeSrc).toContain('pendingSwitchFormat');
    expect(recipeSrc).toContain('confirmDiscardAndSwitch');
  });

  it('QA #6: validateTiedPositions & parseTiedPositions reject malformed, single-residue, missing-chain, and overlapping tie specs', () => {
    // Valid cases
    for (const valid of [
      '',
      '   ',
      'A10=B10',
      'A10=B10; A11=B11',
      'A10=B10, A11=B11',
      'A10=B10; A11=B11, A12=B12',
      'A10 B10; A11 B11',
      'A100A=H100a',
      'A-2=B-2',
      'A0=B0',
      'A10=B10=C10',
      'A:10=B:10',
    ]) {
      expect(validateTiedPositions(valid), `Expected valid for "${valid}"`).toBeNull();
    }

    // Invalid cases
    for (const bad of [
      'INVALID_TIED',
      'A10',
      'A10=B10; A11',
      'A10=B10, A11',
      '10=20',
      'A10=',
      '=B10',
      'A10==B10',
      'A10=A10',
      'A10=B10; A10=C10',
      'A10=B10X1',
      ';;;',
    ]) {
      expect(validateTiedPositions(bad), `Expected error for "${bad}"`).not.toBeNull();
    }

    // Mixed ; and , splits cleanly in parseTiedPositions and throws on malformed/single-token groups
    const parsedMixed = parseTiedPositions('A10=B10; A11=B11, A12=B12');
    expect(parsedMixed).toHaveLength(3);
    expect(() => parseTiedPositions('INVALID_TIED')).toThrow(/Invalid tied position group/i);
    expect(() => parseTiedPositions('A10=B10, A11')).toThrow(/Invalid tied position group/i);

    // Iteration 2 Challenger 2 QA #6 parity: parseTiedPositions rejects separator-only, missing-chain, self-tie, and overlapping groups
    expect(() => parseTiedPositions(';;;')).toThrow(/Invalid tied position group/i);
    expect(() => parseTiedPositions('10=20')).toThrow(/Invalid tied position token/i);
    expect(() => parseTiedPositions('A10=A10')).toThrow(/Duplicate or overlapping/i);
    expect(() => parseTiedPositions('A010=A10')).toThrow(/Duplicate or overlapping/i);
    expect(() => parseTiedPositions('A10=B10; A10=C10')).toThrow(/Duplicate or overlapping/i);

    // validateNodeParams checks tiedPositions on web-protein-mpnn and ligandmpnn
    expect(
      validateNodeParams({
        id: 'mpnn-invalid-tied',
        adapterId: 'web-protein-mpnn',
        params: { tiedPositions: 'INVALID_TIED' },
      }).join(' '),
    ).toMatch(/tiedPositions/i);
    expect(
      validateNodeParams({
        id: 'mpnn-dup-tied',
        adapterId: 'web-protein-mpnn',
        params: { tiedPositions: 'A10=B10; A10=C10' },
      }).join(' '),
    ).toMatch(/Duplicate or overlapping/i);
    expect(
      validateNodeParams({
        id: 'lig-invalid-tied',
        adapterId: 'ligandmpnn',
        params: { tiedPositions: 'A10' },
      }).join(' '),
    ).toMatch(/tiedPositions/i);
  });

  it('Wave 1B: eliminates BUILTIN_HAND_VALIDATED (CT-03), enforces crossValidate hooks (CT-02/CT-06/CT-10), unifies residue tokens (UX-04), and enforces #78 step guardrails without duplicate errors', () => {
    // 1. CT-03: RFD4 validates hbondMode, non-empty motif, and residue tokens via schema
    expect(
      validateNodeParams({
        id: 'rfd-hb',
        adapterId: 'rfd4-proteina',
        params: { mode: 'unconditional', hbondMode: 'INVALID_HB' },
      }).join(' '),
    ).toMatch(/hbondMode/i);
    expect(
      validateNodeParams({
        id: 'rfd-empty-motif',
        adapterId: 'rfd4-proteina',
        params: { mode: 'motif', motifSpan: '', contig: '', customSpecYaml: '' },
      }).join(' '),
    ).toMatch(/motifSpan.*contig/i);
    expect(
      validateNodeParams({
        id: 'rfd-bad-hot',
        adapterId: 'rfd4-proteina',
        params: { mode: 'binder', hotspots: 'A' },
      }).join(' '),
    ).toMatch(/Invalid hotspots token "A"/);
    expect(
      validateNodeParams({
        id: 'g3-bad-motif',
        adapterId: 'genie3',
        params: { taskMode: 'motif', motifSpan: 'A999-1' },
      }).join(' '),
    ).toMatch(/Invalid motifSpan token "A999-1"/);

    // 2. UX-04: multi-char bare chains (AA, HL) pass in MPNN fixedPositions & pdb-input cropSegments
    expect(
      validateNodeParams({
        id: 'mpnn-multichar',
        adapterId: 'web-protein-mpnn',
        params: { fixedPositions: 'AA, HL, A10-15' },
      }),
    ).toEqual([]);
    expect(
      validateNodeParams({
        id: 'pdb-crop-lig',
        adapterId: 'pdb-input',
        params: { mode: 'pdb-id', pdbId: '7C7M', cropSegments: 'A102, A:ZN, A:DHZ, AA' },
      }),
    ).toEqual([]);
    expect(
      validateNodeParams({
        id: 'pdb-crop-bad',
        adapterId: 'pdb-input',
        params: { mode: 'pdb-id', pdbId: '5O45', cropSegments: 'A999-1' },
      }).join(' '),
    ).toMatch(/Invalid cropSegments token/);

    // 3. CT-06: protpardelle enforces rewindSteps < numSteps in partial mode
    expect(
      validateNodeParams({
        id: 'pp-bad-rewind',
        adapterId: 'protpardelle',
        params: { taskMode: 'partial', numSteps: 30, rewindSteps: 40 },
      }).join(' '),
    ).toMatch(/rewindSteps \(40\) must be strictly less than numSteps \(30\)/);
    expect(
      validateNodeParams({
        id: 'pp-eq-rewind',
        adapterId: 'protpardelle',
        params: { taskMode: 'partial', numSteps: 40, rewindSteps: 40 },
      }).join(' '),
    ).toMatch(/rewindSteps/);
    expect(
      validateNodeParams({
        id: 'pp-ok-rewind',
        adapterId: 'protpardelle',
        params: { taskMode: 'partial', numSteps: 50, rewindSteps: 20 },
      }),
    ).toEqual([]);

    // 4. CT-10: esmc & dyna1 require at least one connected sequences or structure input
    expect(
      validateNodeParams(
        { id: 'esmc-disconn', adapterId: 'esmc', params: { mode: 'wildtype' } },
        { connectedInputs: [] },
      ).join(' '),
    ).toMatch(/sequences.*structure/i);
    expect(
      validateNodeParams(
        { id: 'esmc-ok', adapterId: 'esmc', params: { mode: 'wildtype' } },
        { connectedInputs: ['sequences'] },
      ),
    ).toEqual([]);
    expect(
      validateNodeParams(
        { id: 'dyna-disconn', adapterId: 'dyna1', params: { inputMode: 'auto' } },
        { connectedInputs: [] },
      ).join(' '),
    ).toMatch(/sequences.*structure/i);
    expect(
      validateNodeParams(
        { id: 'dyna-struct-only', adapterId: 'dyna1', params: { inputMode: 'struct-only' } },
        { connectedInputs: ['sequences'] },
      ).join(' '),
    ).toMatch(/requires a connected "structure" input/i);

    // 5. CT-02: designability-scorer validates residueMapping syntax before runtime
    expect(
      validateNodeParams({
        id: 'sc-bad-map',
        adapterId: 'designability-scorer',
        params: { residueMapping: 'bad:xyz' },
      }).join(' '),
    ).toMatch(/Invalid residueMapping/);
    expect(
      validateNodeParams({
        id: 'sc-ok-map',
        adapterId: 'designability-scorer',
        params: { residueMapping: '0:0, 1:1, 2:2' },
      }),
    ).toEqual([]);

    // 6. #78: step guardrails produce single non-duplicated errors
    const g2Errs = validateNodeParams({
      id: 'g2-low',
      adapterId: 'genie2',
      params: { steps: 500 },
    });
    expect(g2Errs).toHaveLength(1);
    expect(g2Errs[0]).toMatch(/between 1000 and 1000/);

    const laErrs = validateNodeParams({
      id: 'la-low',
      adapterId: 'la-proteina',
      params: { steps: 20, mode: 'sde' },
    });
    expect(laErrs).toHaveLength(1);
    expect(laErrs[0]).toMatch(/between 50 and 400/);

    const proErrs = validateNodeParams({
      id: 'pro-low',
      adapterId: 'proteina',
      params: { steps: 20, mode: 'sc' },
    });
    expect(proErrs).toHaveLength(1);
    expect(proErrs[0]).toMatch(/between 50 and 400/);
  });
});
