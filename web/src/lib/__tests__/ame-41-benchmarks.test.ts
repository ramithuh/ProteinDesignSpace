import { describe, expect, it } from 'vitest';
import {
  AME_BENCHMARK_TARGETS,
  BENCHMARK_TARGETS,
  FEATURED_AME_BENCHMARK_IDS,
  adaptBenchmarkContigForLength,
  getBenchmarkById,
} from '$lib/engine/benchmarks';
import {
  buildBenchmarkPdbParams,
  isBenchmarkSupportedForAdapter,
} from '$lib/engine/benchmark-sync';
import { filterCatalogCategories } from '$lib/engine/catalog';
import { isValidPdbId } from '$lib/protein-ops/rcsb';

describe('AME 41 enzyme benchmark suite', () => {
  it('registers all 41 canonical AME enzyme tasks with unique IDs, M-CSA IDs, and valid PDB IDs', () => {
    expect(AME_BENCHMARK_TARGETS).toHaveLength(41);

    const ids = new Set(AME_BENCHMARK_TARGETS.map((b) => b.id));
    const mcsaIds = new Set(AME_BENCHMARK_TARGETS.map((b) => b.mcsaId));
    const pdbIds = new Set(AME_BENCHMARK_TARGETS.map((b) => b.pdbId));

    expect(ids.size).toBe(41);
    expect(mcsaIds.size).toBe(41);
    expect(pdbIds.size).toBe(41);

    for (const bench of AME_BENCHMARK_TARGETS) {
      expect(bench.id.startsWith('ame_'), `Expected ame_ prefix on ${bench.id}`).toBe(true);
      expect(bench.mcsaId, `Missing mcsaId on ${bench.id}`).toMatch(/^M\d{4}$/);
      expect(isValidPdbId(bench.pdbId), `Invalid PDB ID on ${bench.id}: ${bench.pdbId}`).toBe(true);
      expect(bench.category).toBe('enzyme-ligand');
      expect(bench.supportsGenie3).toBe(false);
      expect(bench.entities).toEqual(['protein', 'ligand']);
      expect(bench.targetLength).toBe(180);
      expect(bench.approxTokens).toBeGreaterThan(20);
      expect(bench.approxTokens).toBeLessThanOrEqual(300);

      // Lookup via getBenchmarkById and presence in BENCHMARK_TARGETS
      expect(getBenchmarkById(bench.id)).toBe(bench);
      expect(BENCHMARK_TARGETS).toContain(bench);
    }
  });

  it('validates non-empty tip atoms, motifSpan, ligands, and cropSegments consistency across all 41 AME entries', () => {
    for (const bench of AME_BENCHMARK_TARGETS) {
      const { contig, motifSpan, motifAtoms, unindexedMotif } = bench.conditionParams;
      expect(contig).toBe('180-180');
      expect(unindexedMotif).toBe(true);
      expect(motifSpan && motifSpan.trim().length > 0, `Missing motifSpan on ${bench.id}`).toBe(true);
      expect(motifAtoms && motifAtoms.trim().length > 0, `Missing motifAtoms on ${bench.id}`).toBe(true);
      expect(bench.cropSegments && bench.cropSegments.trim().length > 0, `Missing cropSegments on ${bench.id}`).toBe(true);

      const ligs = bench.ligands ?? [];
      expect(ligs.length, `Missing ligands on ${bench.id}`).toBeGreaterThanOrEqual(1);
      for (const lig of ligs) {
        expect(lig).toMatch(/^[A-Z0-9]{1,4}$/);
        expect(bench.cropSegments, `cropSegments missing <chain>:${lig} on ${bench.id}`).toMatch(
          new RegExp(`[A-Z]:${lig}\\b`),
        );
      }

      const atomEntries = motifAtoms!.split(';').map((s) => s.trim()).filter(Boolean);
      const spanResidues = motifSpan!.split(',').map((s) => s.trim()).filter(Boolean);
      expect(atomEntries.length).toBe(spanResidues.length);
      expect(bench.motifResidueCount).toBe(spanResidues.length);

      for (let i = 0; i < atomEntries.length; i++) {
        const [resToken, atomList] = atomEntries[i].split(':').map((s) => s.trim());
        expect(resToken).toBe(spanResidues[i]);
        expect(resToken).toMatch(/^[A-Z]\d+$/);
        const atoms = (atomList ?? '').split(',').map((a) => a.trim()).filter(Boolean);
        expect(atoms.length, `Empty tip atoms for ${resToken} in ${bench.id}`).toBeGreaterThanOrEqual(2);
        expect(bench.cropSegments, `cropSegments missing ${resToken} on ${bench.id}`).toContain(resToken);
      }
    }
  });

  it('preserves 100% backward compatibility for ame_1ctt_cda and ame_1os7_taud', () => {
    const ctt = getBenchmarkById('ame_1ctt_cda');
    expect(ctt).toBeDefined();
    expect(ctt!.pdbId).toBe('1CTT');
    expect(ctt!.mcsaId).toBe('M0097');
    expect(ctt!.cropSegments).toBe('A102, A104, A129, A132, A:ZN, A:DHZ');
    expect(ctt!.conditionParams.motifAtoms).toBe(
      'A102:ND1,CG,CE1; A104:OE1,CD; A129:SG,CB; A132:SG,CB,CA',
    );
    expect(FEATURED_AME_BENCHMARK_IDS.has('ame_1ctt_cda')).toBe(true);

    const os7 = getBenchmarkById('ame_1os7_taud');
    expect(os7).toBeDefined();
    expect(os7!.pdbId).toBe('1OS7');
    expect(os7!.mcsaId).toBe('M0129');
    expect(os7!.cropSegments).toBe('A99, A101, A255, A270, A:FE2, A:AKG, A:TAU');
    expect(os7!.conditionParams.motifAtoms).toBe(
      'A99:NE2,CD2,CE1; A101:OD1,CG,CB,OD2; A255:NE2,CD2,CE1,CG,ND1; A270:NH1,CZ',
    );
    expect(FEATURED_AME_BENCHMARK_IDS.has('ame_1os7_taud')).toBe(true);
  });

  it('integrates all 41 AME entries with conditioning, benchmark-sync, and catalog search', () => {
    for (const bench of AME_BENCHMARK_TARGETS) {
      expect(isBenchmarkSupportedForAdapter('rfd4-proteina', bench)).toBe(true);
      expect(isBenchmarkSupportedForAdapter('rfd3', bench)).toBe(true);
      expect(isBenchmarkSupportedForAdapter('genie3', bench)).toBe(false);

      const pdbParams = buildBenchmarkPdbParams({}, bench);
      expect(pdbParams.pdbId).toBe(bench.pdbId);
      expect(pdbParams.benchmarkId).toBe(bench.id);
      expect(pdbParams.cropSegments).toBe(bench.cropSegments);

      expect(adaptBenchmarkContigForLength(bench.conditionParams.contig, 150, bench.category)).toBe('150-150');
    }

    // Spot-check catalog search by M-CSA ID, PDB ID, and ligand across newly added AME tasks
    for (const query of ['M0024', '13PK', 'M0907', '1RBL', 'M0630', 'OXM']) {
      const hits = filterCatalogCategories(query);
      const inputsCat = hits.find((h) => h.category.id === 'inputs');
      expect(inputsCat, `Catalog search "${query}" should match Inputs`).toBeDefined();
      expect(inputsCat!.matchingSubItems.length).toBeGreaterThan(0);
    }
  });
});
