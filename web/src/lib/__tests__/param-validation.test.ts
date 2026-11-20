import { describe, expect, it } from 'vitest';
import {
  NUMERIC_SPECS,
  commitNumericDraft,
  isValidResidueToken,
  sanitizeNumericDraft,
  validateJsonOverride,
  validateNodeParams,
  validateNumericDraft,
  validateSequenceText,
  validateStructureText,
} from '../engine/param-validation';

describe('parameter validation & numeric sanitization', () => {
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
});
