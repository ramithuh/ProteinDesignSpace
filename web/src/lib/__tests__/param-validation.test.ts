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
    expect(short.status).toBe('warn');
  });

  it('validates residue tokens, JSON overrides, and PDB/mmCIF structure text', () => {
    expect(isValidResidueToken('A54')).toBe(true);
    expect(isValidResidueToken('A16-35')).toBe(true);
    expect(isValidResidueToken('10-14')).toBe(true);
    expect(isValidResidueToken('A5r')).toBe(false);
    expect(isValidResidueToken('bad!')).toBe(false);

    expect(validateJsonOverride('').status).toBe('ok');
    expect(validateJsonOverride('{"contig":"70-70"}').status).toBe('ok');
    expect(validateJsonOverride('{broken').status).toBe('error');
    expect(validateJsonOverride('[1,2]').status).toBe('error');

    expect(validateStructureText('').status).toBe('error');
    expect(validateStructureText('hello world').status).toBe('error');
    expect(validateStructureText('ATOM      1  CA  ALA A   1       0.000   0.000   0.000').status).toBe('ok');
    expect(validateStructureText('loop_\n_atom_site.group_PDB').status).toBe('ok');
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
        adapterId: 'rfd4-conditioner',
        label: 'Cond',
        params: { taskPreset: 'custom', contig: '70-70', rawJsonOverride: '{bad' },
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
});
