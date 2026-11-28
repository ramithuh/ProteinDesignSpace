import { describe, expect, it } from 'vitest';
import {
  ALL_ATOM_MOTIF_BENCHMARK_IDS,
  BENCHMARK_TARGETS,
  FEATURED_MOTIF_BENCHMARK_IDS,
  MOTIF_BENCHMARK_TARGETS,
  MOTIF_BENCHMARK_YAML_KEYS,
  adaptBenchmarkContigForLength,
  getBenchmarkById,
} from '$lib/engine/benchmarks';
import {
  buildBenchmarkPdbParams,
  formatRfdiffusionContig,
  isBenchmarkSupportedForAdapter,
} from '$lib/engine/benchmark-sync';
import { filterCatalogCategories } from '$lib/engine/catalog';
import { isValidPdbId } from '$lib/protein-ops/rcsb';

/** Total motif residues + fixed scaffold flanks encoded by a canonical benchmark contig. */
function contigLength(contig: string): { total: number; motif: number; tokens: number } {
  const tokens = contig
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean);
  let total = 0;
  let motif = 0;
  for (const tok of tokens) {
    const span = /^([A-Za-z])(\d+)-(\d+)$/.exec(tok);
    if (span) {
      const len = Number(span[3]) - Number(span[2]) + 1;
      total += len;
      motif += len;
      continue;
    }
    const flank = /^(\d+)-(\d+)$/.exec(tok);
    expect(flank, `Unrecognized contig token "${tok}" in "${contig}"`).not.toBeNull();
    expect(flank![1], `Flank token "${tok}" must be a fixed length`).toBe(flank![2]);
    total += Number(flank![1]);
  }
  return { total, motif, tokens: tokens.length };
}

function parseSpans(span: string): { chain: string; start: number; end: number }[] {
  return span
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      const m = /^([A-Za-z])(\d+)(?:-(\d+))?$/.exec(s);
      expect(m, `Unparseable motif span segment "${s}"`).not.toBeNull();
      const start = Number(m![2]);
      return { chain: m![1], start, end: m![3] !== undefined ? Number(m![3]) : start };
    });
}

describe('Motif scaffolding benchmark suite (motif_dict.yaml)', () => {
  it('registers all 31 motif targets with unique IDs and valid RCSB PDB IDs', () => {
    expect(MOTIF_BENCHMARK_TARGETS).toHaveLength(31);

    const ids = new Set(MOTIF_BENCHMARK_TARGETS.map((b) => b.id));
    expect(ids.size).toBe(31);
    // shortLabel drives both the catalog flyout and the PDB-input <select>, so it must be unique.
    expect(new Set(MOTIF_BENCHMARK_TARGETS.map((b) => b.shortLabel)).size).toBe(31);
    expect(new Set(MOTIF_BENCHMARK_TARGETS.map((b) => b.name)).size).toBe(31);

    for (const bench of MOTIF_BENCHMARK_TARGETS) {
      expect(isValidPdbId(bench.pdbId), `Invalid PDB ID on ${bench.id}: ${bench.pdbId}`).toBe(true);
      expect(bench.pdbId).toMatch(/^[0-9][A-Z0-9]{3}$/);
      expect(bench.category).toBe('motif');
      expect(bench.entities).toEqual(['protein']);
      expect(bench.supportsGenie3).toBe(true);
      expect(bench.biology.trim().length).toBeGreaterThan(10);
      expect(bench.sourceFamily.trim().length).toBeGreaterThan(0);
      expect(bench.approxTokens).toBeGreaterThan(20);
      expect(bench.approxTokens).toBeLessThanOrEqual(300);
      expect(bench.tier).toBe(bench.approxTokens <= 200 ? 'fast' : 'standard');

      // Lookup + registration in the single global registry
      expect(getBenchmarkById(bench.id)).toBe(bench);
      expect(BENCHMARK_TARGETS).toContain(bench);
    }

    // Every motif entry in the global registry comes from MOTIF_BENCHMARK_TARGETS, in order.
    expect(BENCHMARK_TARGETS.filter((b) => b.category === 'motif')).toEqual(
      MOTIF_BENCHMARK_TARGETS,
    );
  });

  it('defines consistent cropSegments, motifSpan and fixed-length contigs for every entry', () => {
    for (const bench of MOTIF_BENCHMARK_TARGETS) {
      const { contig, motifSpan, motifAtoms, unindexedMotif, lockChain, hotspots } =
        bench.conditionParams;
      expect(bench.conditionParams.taskPreset).toBe('custom');
      expect(contig.trim().length, `Missing contig on ${bench.id}`).toBeGreaterThan(0);
      expect(motifSpan, `Missing motifSpan on ${bench.id}`).toBeTruthy();
      expect(bench.cropSegments, `Missing cropSegments on ${bench.id}`).toBeTruthy();

      // Motif tasks are indexed CA/backbone problems: no unindexed/tip-atom or binder params.
      expect(motifAtoms).toBeUndefined();
      expect(unindexedMotif).toBeUndefined();
      expect(lockChain).toBeUndefined();
      expect(hotspots).toBeUndefined();
      expect(contig).not.toContain('/0');
      expect(contig.split(',')).not.toContain('0-0');

      // cropSegments mirrors motifSpan exactly, sorted ascending by (chain, start residue).
      expect(bench.cropSegments).toBe(motifSpan);
      const spans = parseSpans(motifSpan!);
      expect(spans.length).toBeGreaterThanOrEqual(1);
      const sorted = [...spans].sort((a, b) => a.chain.localeCompare(b.chain) || a.start - b.start);
      expect(spans).toEqual(sorted);
      for (const seg of spans) {
        expect(seg.end).toBeGreaterThanOrEqual(seg.start);
        expect(seg.start).toBeGreaterThan(0);
      }

      // The contig's fixed flanks + motif residues sum to exactly the canonical design length,
      // and it carries one contig segment per motifSpan island.
      const { total, motif, tokens } = contigLength(contig);
      expect(total, `Contig length mismatch on ${bench.id}`).toBe(bench.approxTokens);
      expect(motif).toBe(spans.reduce((s, seg) => s + (seg.end - seg.start + 1), 0));
      expect(tokens).toBeGreaterThan(spans.length);
      if (bench.targetLength !== undefined) {
        expect(bench.targetLength).toBe(bench.approxTokens);
      }

      // Contig islands reference the same chains/residues as motifSpan (order may differ).
      const contigSpans = contig
        .split(',')
        .map((t) => /^([A-Za-z])(\d+)-(\d+)$/.exec(t.trim()))
        .filter((m): m is RegExpExecArray => m !== null)
        .map((m) => `${m[1]}${m[2]}-${m[3]}`)
        .sort();
      const declaredSpans = spans.map((s) => `${s.chain}${s.start}-${s.end}`).sort();
      expect(contigSpans).toEqual(declaredSpans);
    }
  });

  it('preserves 100% backward compatibility for the original 6 curated motif benchmarks', () => {
    expect(FEATURED_MOTIF_BENCHMARK_IDS.size).toBe(6);
    // The curated 6 stay first, in their original registry order.
    expect(MOTIF_BENCHMARK_TARGETS.slice(0, 6).map((b) => b.id)).toEqual([
      'rfd_5tpn_rsvf_v',
      'rfd_1prw_efhand',
      'g3_5yui_ca2',
      'rfd_1ycr_p53',
      'rfd_1bcf_diiron',
      'g3_4xoj_trypsin',
    ]);

    const expected: Record<
      string,
      { pdbId: string; cropSegments: string; contig: string; motifSpan: string; approxTokens: number }
    > = {
      rfd_5tpn_rsvf_v: {
        pdbId: '5TPN',
        cropSegments: 'A163-181',
        contig: '25-25,A163-181,31-31',
        motifSpan: 'A163-181',
        approxTokens: 75,
      },
      rfd_1prw_efhand: {
        pdbId: '1PRW',
        cropSegments: 'A16-35, A52-71',
        contig: '10-10,A16-35,16-16,A52-71,10-10',
        motifSpan: 'A16-35, A52-71',
        approxTokens: 76,
      },
      g3_5yui_ca2: {
        pdbId: '5YUI',
        cropSegments: 'A93-97, A118-120, A198-200',
        contig: '14-14,A93-97,20-20,A118-120,20-20,A198-200,10-10',
        motifSpan: 'A93-97, A118-120, A198-200',
        approxTokens: 75,
      },
      rfd_1ycr_p53: {
        pdbId: '1YCR',
        cropSegments: 'B19-27',
        contig: '28-28,B19-27,28-28',
        motifSpan: 'B19-27',
        approxTokens: 65,
      },
      rfd_1bcf_diiron: {
        pdbId: '1BCF',
        cropSegments: 'A18-25, A47-54, A92-99, A123-130',
        contig: '12-12,A92-99,24-24,A123-130,24-24,A47-54,23-23,A18-25,10-10',
        motifSpan: 'A18-25, A47-54, A92-99, A123-130',
        approxTokens: 125,
      },
      g3_4xoj_trypsin: {
        pdbId: '4XOJ',
        cropSegments: 'A55, A99, A190-192',
        contig: '15-15,A55-55,45-45,A99-99,70-70,A190-192,15-15',
        motifSpan: 'A55, A99, A190-192',
        approxTokens: 150,
      },
    };

    for (const [id, want] of Object.entries(expected)) {
      const bench = getBenchmarkById(id);
      expect(bench, `Missing curated motif benchmark ${id}`).toBeDefined();
      expect(FEATURED_MOTIF_BENCHMARK_IDS.has(id)).toBe(true);
      expect(bench!.category).toBe('motif');
      expect(bench!.pdbId).toBe(want.pdbId);
      expect(bench!.cropSegments).toBe(want.cropSegments);
      expect(bench!.conditionParams.contig).toBe(want.contig);
      expect(bench!.conditionParams.motifSpan).toBe(want.motifSpan);
      expect(bench!.approxTokens).toBe(want.approxTokens);
      expect(bench!.tier).toBe('fast');
      expect(bench!.supportsGenie3).toBe(true);
      // The curated 6 predate the YAML import and carry no derived targetLength.
      expect(bench!.targetLength).toBeUndefined();
    }
  });

  it('maps every canonical motif_dict.yaml task (incl. deduped _AA/_TIP variants) to exactly one ID', () => {
    expect(MOTIF_BENCHMARK_YAML_KEYS.size).toBe(30);

    const seen = new Map<string, string>();
    for (const [benchId, keys] of MOTIF_BENCHMARK_YAML_KEYS) {
      expect(getBenchmarkById(benchId), `Unknown benchmark for YAML keys ${keys}`).toBeDefined();
      expect(keys.length).toBeGreaterThanOrEqual(1);
      for (const key of keys) {
        expect(seen.has(key), `YAML key ${key} claimed by both ${seen.get(key)} and ${benchId}`).toBe(
          false,
        );
        seen.set(key, benchId);
      }
    }

    // All 77 YAML entries collapse to 30 tasks, each claimed exactly once:
    // 26 `motif_data_aa/` tasks + their 26 `_TIP` twins + 25 plain `motif_data/` tasks.
    expect(seen.size).toBe(77);
    const keys = [...seen.keys()];
    expect(keys.filter((k) => k.endsWith('_TIP'))).toHaveLength(26);
    expect(keys.filter((k) => k.includes('_AA') && !k.endsWith('_TIP'))).toHaveLength(26);
    expect(keys.filter((k) => !k.includes('_AA'))).toHaveLength(25);
    const covered = new Set(MOTIF_BENCHMARK_YAML_KEYS.keys());
    expect(
      MOTIF_BENCHMARK_TARGETS.filter((b) => !covered.has(b.id)).map((b) => b.id),
    ).toEqual(['g3_4xoj_trypsin']);

    // Every benchmark whose YAML keys are all-atom-only is flagged as such.
    expect([...ALL_ATOM_MOTIF_BENCHMARK_IDS].sort()).toEqual([
      'rfdaa_1qjg_ksi_native',
      'rfdaa_5aou_ra95_tetrad',
      'rfdaa_6vw1_ace2',
      'rfdaa_7k4v_kemp',
      'rfdaa_5aou_ra95_triad',
    ].sort());
    for (const id of ALL_ATOM_MOTIF_BENCHMARK_IDS) {
      expect(MOTIF_BENCHMARK_YAML_KEYS.get(id)!.every((k) => k.includes('_AA'))).toBe(true);
    }
  });

  it('covers the full RFdiffusion 25 / design25 problem set with RCSB-resolved chains', () => {
    const byPdb = new Map<string, number>();
    for (const bench of MOTIF_BENCHMARK_TARGETS) {
      byPdb.set(bench.pdbId, (byPdb.get(bench.pdbId) ?? 0) + 1);
    }
    // Multi-length / multi-ordering problem families keep one entry per variant.
    expect(byPdb.get('7MRX')).toBe(3);
    expect(byPdb.get('5TRV')).toBe(3);
    expect(byPdb.get('6E6R')).toBe(3);
    expect(byPdb.get('6EXZ')).toBe(3);
    expect(byPdb.get('1QJG')).toBe(3);
    expect(byPdb.get('5AOU')).toBe(2);

    // Chain letters re-resolved from the preprocessed YAML PDBs to real RCSB auth chains.
    const chainFixes: Record<string, string> = {
      rfd_3ixt_rsvf_ii: 'P254-277',
      rfd_4jhw_rsvf_zero: 'F63-69, F196-212',
      rfd_5ius_pd1: 'A63-82, A119-140',
      rfd_7mrx_barstar_60: 'B25-46',
      rfd_1qjg_ksi: 'A14, A38, A99',
      rfd_2kl8_or15: 'A1-7, A28-79',
      // 6EXZ ships renumbered-from-1 in motif_data/; RCSB numbers MEX67's C-term 530–598.
      rfd_6exz_mex67_short: 'A557-571',
    };
    for (const [id, span] of Object.entries(chainFixes)) {
      expect(getBenchmarkById(id)!.conditionParams.motifSpan, `Wrong motif span on ${id}`).toBe(span);
    }

    // The three 1QJG orderings are genuinely distinct problems, not duplicates.
    const ksiContigs = ['rfd_1qjg_ksi', 'rfd_1qjg_ksi_indexed', 'rfdaa_1qjg_ksi_native'].map(
      (id) => getBenchmarkById(id)!.conditionParams.contig,
    );
    expect(new Set(ksiContigs).size).toBe(3);
    expect(ksiContigs[0]).toContain('A38-38,30-30,A14-14');
    expect(ksiContigs[1]).toContain('A14-14,30-30,A38-38');
    expect(getBenchmarkById('rfdaa_1qjg_ksi_native')!.approxTokens).toBe(135);
  });

  it('integrates every motif entry with benchmark-sync, contig adaptation, and catalog search', () => {
    for (const bench of MOTIF_BENCHMARK_TARGETS) {
      expect(isBenchmarkSupportedForAdapter('rfd4-proteina', bench)).toBe(true);
      expect(isBenchmarkSupportedForAdapter('rfd3', bench)).toBe(true);
      expect(isBenchmarkSupportedForAdapter('genie3', bench)).toBe(true);
      expect(isBenchmarkSupportedForAdapter('genie2', bench)).toBe(true);
      expect(isBenchmarkSupportedForAdapter('boltzgen', bench)).toBe(true);
      expect(isBenchmarkSupportedForAdapter('protpardelle', bench)).toBe(true);
      expect(isBenchmarkSupportedForAdapter('rfdiffusion', bench)).toBe(true);

      const pdbParams = buildBenchmarkPdbParams({}, bench);
      expect(pdbParams.pdbId).toBe(bench.pdbId);
      expect(pdbParams.benchmarkId).toBe(bench.id);
      expect(pdbParams.cropSegments).toBe(bench.cropSegments);

      // RFdiffusion `/`-delimited round-trip keeps every motif island intact.
      const rfdContig = formatRfdiffusionContig(bench.conditionParams.contig);
      for (const seg of parseSpans(bench.conditionParams.motifSpan!)) {
        expect(rfdContig, `RFdiffusion contig for ${bench.id} dropped ${seg.chain}${seg.start}`).toContain(
          `${seg.chain}${seg.start}-${seg.end}`,
        );
      }

      // Re-scaling to a user length preserves fixed motif spans and hits the requested total.
      for (const L of [60, 120, 200]) {
        const adapted = adaptBenchmarkContigForLength(
          bench.conditionParams.contig,
          L,
          bench.category,
        );
        const { motif } = contigLength(adapted);
        expect(motif, `Motif residues changed when adapting ${bench.id} to ${L}`).toBe(
          contigLength(bench.conditionParams.contig).motif,
        );
        for (const seg of parseSpans(bench.conditionParams.motifSpan!)) {
          expect(adapted).toContain(`${seg.chain}${seg.start}-${seg.end}`);
        }
      }
    }

    // Catalog search reaches the newly added targets by PDB ID, benchmark ID, and biology terms.
    for (const query of ['3IXT', '6EXZ', 'rfdaa_7k4v_kemp', 'barstar', 'ace2', 'ketosteroid']) {
      const hits = filterCatalogCategories(query);
      const inputsCat = hits.find((h) => h.category.id === 'inputs');
      expect(inputsCat, `Catalog search "${query}" should match Inputs`).toBeDefined();
      expect(inputsCat!.matchingSubItems.length).toBeGreaterThan(0);
    }
  });
});
