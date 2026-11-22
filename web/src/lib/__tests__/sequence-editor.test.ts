import { describe, expect, it } from 'vitest';
import {
  cursorToResiduePosition,
  diffEntryResidues,
  mutateResidueInRaw,
  parseMutationCommand,
  parseSequenceEntries,
} from '$lib/engine/sequence-editor';

const GB98_SEQ = 'TTYKLILNLKQAKEEAIKELVDAGTAEKYFKLIANAKTVEGVWTYKDEIKTFTVTE';

describe('sequence-editor helpers', () => {
  it('parses plain, multi-FASTA, and multi-chain sequences with exact char offsets', () => {
    const plain = parseSequenceEntries(GB98_SEQ);
    expect(plain).toHaveLength(1);
    expect(plain[0].residues).toHaveLength(56);
    expect(plain[0].residues[44]).toMatchObject({ pos: 45, aa: 'Y', charIndex: 44 });

    const multi = parseSequenceEntries('>chain_ab\nACDE:FG\n>second\nMKT\n');
    expect(multi).toHaveLength(2);
    expect(multi[0].header).toBe('chain_ab');
    expect(multi[0].residues.map((r) => `${r.aa}@${r.chainIdx}:${r.chainPos}`)).toEqual([
      'A@0:1',
      'C@0:2',
      'D@0:3',
      'E@0:4',
      'F@1:1',
      'G@1:2',
    ]);
    expect(multi[0].residues[4].startsNewChain).toBe(true);
    expect(multi[1].header).toBe('second');
    expect(multi[1].residues.map((r) => r.aa).join('')).toBe('MKT');
  });

  it('mutates residues in place while preserving FASTA headers, newlines, and chain breaks', () => {
    const flipped = mutateResidueInRaw(GB98_SEQ, 0, 45, 'l');
    expect(flipped).toBe('TTYKLILNLKQAKEEAIKELVDAGTAEKYFKLIANAKTVEGVWTLKDEIKTFTVTE');
    expect(diffEntryResidues(GB98_SEQ, flipped!, 0)).toEqual([
      { pos: 45, from: 'Y', to: 'L', label: 'Y45L' },
    ]);

    const fasta = '>gb98\nTTYKLILNLK\nQAKEEAIKEL\n';
    const mutFasta = mutateResidueInRaw(fasta, 0, 12, 'R');
    expect(mutFasta).toBe('>gb98\nTTYKLILNLK\nQRKEEAIKEL\n');
  });

  it('parses biologist mutation & jump commands', () => {
    expect(parseMutationCommand('Y45L')).toEqual({
      kind: 'mutate',
      expectedWt: 'Y',
      pos: 45,
      targetAa: 'L',
    });
    expect(parseMutationCommand('y45→l')).toEqual({
      kind: 'mutate',
      expectedWt: 'Y',
      pos: 45,
      targetAa: 'L',
    });
    expect(parseMutationCommand('45L')).toEqual({
      kind: 'mutate',
      pos: 45,
      targetAa: 'L',
    });
    expect(parseMutationCommand('45')).toEqual({ kind: 'jump', pos: 45 });
    expect(parseMutationCommand('X45Z').kind).toBe('error');
  });

  it('maps raw textarea cursor offset to 1-indexed residue position', () => {
    const info = cursorToResiduePosition(GB98_SEQ, 44);
    expect(info).toEqual({ entryIdx: 0, pos: 45, aa: 'Y', total: 56 });
  });
});
