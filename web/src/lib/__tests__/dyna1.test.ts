import { describe, expect, it } from 'vitest';
import {
  extractDyna1Backbone3Coords,
  normalizeDyna1InputMode,
  resolveDyna1Mode,
} from '$lib/engine/dyna1-helpers';
import { parseStructureText } from '$lib/protein-ops/pdb';

const MINI_PDB = [
  'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N',
  'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C',
  'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C',
  'ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O',
  'ATOM      5  N   GLY A   2       3.332   1.550   0.000  1.00 20.00           N',
  'ATOM      6  CA  GLY A   2       3.972   2.856   0.000  1.00 20.00           C',
  'ATOM      7  C   GLY A   2       5.485   2.742   0.000  1.00 20.00           C',
  'ATOM      8  O   GLY A   2       6.051   1.653   0.000  1.00 20.00           O',
  'ATOM      9  N   VAL A   3       6.128   3.902   0.000  1.00 20.00           N',
  'ATOM     10  CA  VAL A   3       7.579   3.963   0.000  1.00 20.00           C',
  'ATOM     11  C   VAL A   3       8.095   5.394   0.000  1.00 20.00           C',
  'ATOM     12  O   VAL A   3       7.327   6.356   0.000  1.00 20.00           O',
  'END',
].join('\n');

const MISSING_NC_PDB = [
  'ATOM      1  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C',
  'ATOM      2  CA  GLY A   2       3.972   2.856   0.000  1.00 20.00           C',
  'ATOM      3  CA  VAL A   3       7.579   3.963   0.000  1.00 20.00           C',
  'END',
].join('\n');

describe('Dyna-1 structure conditioning helpers', () => {
  it('extracts [N, CA, C] (L * 9) coords from full-backbone structures and encodes missing atoms as NaN', () => {
    const parsed = parseStructureText(MINI_PDB);
    const bb = extractDyna1Backbone3Coords({
      length: parsed.nResidues,
      backbone4Coords: parsed.backbone4Coords,
      atomPresence: parsed.atomPresence,
      atomGranularity: 'all-atom',
    });
    expect(bb).not.toBeNull();
    expect(bb!.length).toBe(3);
    expect(bb!.validResidueCount).toBe(3);
    expect(bb!.coords.length).toBe(27);
    expect(bb!.coords[3]).toBeCloseTo(1.458, 3); // Res 0 CA.x
    expect(bb!.coords[6]).toBeCloseTo(2.009, 3); // Res 0 C.x
    expect(bb!.coords[18]).toBeCloseTo(6.128, 3); // Res 2 N.x

    // Missing N/C (Cα-only PDB): N and C become NaN and validResidueCount = 0
    const caOnly = parseStructureText(MISSING_NC_PDB);
    const bbCa = extractDyna1Backbone3Coords({
      length: caOnly.nResidues,
      backbone4Coords: caOnly.backbone4Coords,
      atomPresence: caOnly.atomPresence,
      atomGranularity: 'all-atom',
    });
    expect(bbCa).not.toBeNull();
    expect(bbCa!.validResidueCount).toBe(0);
    expect(Number.isNaN(bbCa!.coords[0])).toBe(true); // N.x is NaN
    expect(bbCa!.coords[3]).toBeCloseTo(1.458, 3); // CA.x preserved
    expect(Number.isNaN(bbCa!.coords[6])).toBe(true); // C.x is NaN

    // Explicit ca-only granularity returns null immediately
    expect(
      extractDyna1Backbone3Coords({
        length: parsed.nResidues,
        backbone4Coords: parsed.backbone4Coords,
        atomPresence: parsed.atomPresence,
        atomGranularity: 'ca-only',
      }),
    ).toBeNull();
  });

  it('resolves auto/seq+struct/seq-only/struct-only conditioning modes cleanly', () => {
    expect(normalizeDyna1InputMode(undefined)).toBe('auto');
    expect(normalizeDyna1InputMode('seq+struct')).toBe('seq+struct');
    expect(normalizeDyna1InputMode('struct-only')).toBe('struct-only');
    expect(normalizeDyna1InputMode('seq-only')).toBe('seq-only');

    const parsed = parseStructureText(MINI_PDB);
    const bb = extractDyna1Backbone3Coords({
      length: parsed.nResidues,
      backbone4Coords: parsed.backbone4Coords,
      atomPresence: parsed.atomPresence,
      atomGranularity: 'all-atom',
    });

    expect(resolveDyna1Mode('auto', 3, bb)).toBe('seq+struct');
    expect(resolveDyna1Mode('auto', 5, bb)).toBe('seq-only'); // length mismatch falls back safely
    expect(resolveDyna1Mode('auto', 3, null)).toBe('seq-only');
    expect(resolveDyna1Mode('seq-only', 3, bb)).toBe('seq-only');
    expect(resolveDyna1Mode('struct-only', 3, bb)).toBe('struct-only');
    expect(() => resolveDyna1Mode('seq+struct', 3, null)).toThrow(/full backbone/i);
    expect(() => resolveDyna1Mode('struct-only', 3, null)).toThrow(/full backbone/i);
  });
});
