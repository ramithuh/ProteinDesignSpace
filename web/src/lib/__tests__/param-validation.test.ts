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
} from '../engine/param-validation';
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

    // Finding #29: enum parameter validation
    expect(
      validateNodeParams({
        id: 'e1',
        adapterId: 'rfd4-proteina',
        params: { mode: 'bogus_mode' },
      }).join(' '),
    ).toMatch(/Unknown RFD4 mode/);
    expect(
      validateNodeParams({
        id: 'e2',
        adapterId: 'rfd4-proteina',
        params: { mode: 'preset', presetId: 'nonexistent_preset' },
      }).join(' '),
    ).toMatch(/Unknown RFD4 preset/);
    expect(
      validateNodeParams({
        id: 'e3',
        adapterId: 'genie3',
        params: { taskMode: 'ligand' },
      }).join(' '),
    ).toMatch(/Unknown Genie 3 taskMode/);
    expect(
      validateNodeParams({
        id: 'e4',
        adapterId: 'web-protein-mpnn',
        params: { family: 'unknown_mpnn' },
      }).join(' '),
    ).toMatch(/Unknown ProteinMPNN family/);
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
    ).toMatch(/Unknown LocalFold model/);
  });

  it.each(['ef2-fast-300m', 'ef2-fast-600m'])('validates the active inputs and schedule for %s', (model) => {
    const validate = (params: Record<string, unknown>) =>
      validateNodeParams({ id: 'fold', adapterId: 'localfold-af2', params: { model, ...params } });
    expect(validate({})).toEqual([]);
    expect(validate({ esmfold2Steps: '32', recycles: 19 })).toEqual([]);
    expect(validate({ esmfold2Steps: 25 }).join(' ')).toContain('diffusion schedule');
    expect(validate({ msaMode: 'mmseqs2' }).join(' ')).toContain('single-sequence');
    expect(validate({ ligandCcdCodes: ['ZN'] }).join(' ')).toContain('protein chains only');
  });

  it('mpnn.worker.ts supports whole-chain locking, designChains, inherited locks, omitAminoAcids, tiedPositions, and aligned bounds', () => {
    const mpnnSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/mpnn.worker.ts'), 'utf8');
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
    ).toMatch(/Unknown PDB Input mode/i);
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
    ).toMatch(/Invalid msaMode/);
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
    expect(rfd4Src).toContain("bfactorSemantics: 'crystal-b'");
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

    const mpnnSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/mpnn.worker.ts'), 'utf8');
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
});
