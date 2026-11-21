import { describe, expect, it, vi } from 'vitest';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';
import {
  clampLocalFoldRecycles,
  computeAf2MsaBudgets,
  computeTargetOnlyOffMask,
  createMsaSearchCache,
  maxRecyclesForLocalFoldModel,
  resolveAf2Tolerance,
  resolveMsaForCandidate,
  selectLocalFoldRepresentative,
  validateAndParseA3mText,
  type MmseqsApi,
} from '$lib/engine/localfold-msa';
import type { FoldedCandidate } from '$lib/engine/types';

const VENDOR_ROOT = resolve(__dirname, '../../../static/vendor/localfold');

async function importLocalFold(relPath: string) {
  return await import(pathToFileURL(resolve(VENDOR_ROOT, relPath)).href);
}

/** Build a valid POSIX tar archive in memory for testing real vendored mmseqs2-api.js! */
function buildTarArchive(entries: Array<{ name: string; content: string }>): Uint8Array {
  const enc = new TextEncoder();
  const chunks: Uint8Array[] = [];
  for (const { name, content } of entries) {
    const body = enc.encode(content);
    const header = new Uint8Array(512);
    header.set(enc.encode(name).subarray(0, 100), 0);
    header.set(enc.encode('0000644\0'), 100);
    header.set(enc.encode('0000000\0'), 108);
    header.set(enc.encode('0000000\0'), 116);
    const octalSize = `${body.byteLength.toString(8).padStart(11, '0')}\0`;
    header.set(enc.encode(octalSize), 124);
    header.set(enc.encode('00000000000\0'), 136);
    header[156] = 48; // '0' regular file
    // Checksum field filled with spaces during computation
    for (let i = 148; i < 156; i++) header[i] = 32;
    let sum = 0;
    for (let i = 0; i < 512; i++) sum += header[i];
    header.set(enc.encode(`${sum.toString(8).padStart(6, '0')}\0 `), 148);
    chunks.push(header);
    chunks.push(body);
    const pad = (512 - (body.byteLength % 512)) % 512;
    if (pad > 0) chunks.push(new Uint8Array(pad));
  }
  chunks.push(new Uint8Array(1024)); // EOF blocks
  const total = chunks.reduce((acc, c) => acc + c.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out;
}

function makeStubCandidate(overrides: Partial<FoldedCandidate>): FoldedCandidate {
  return {
    sequenceId: 'cand-1',
    sequence: 'MKTAYIAKQRQISFVKSHFS',
    foldModelId: 'localfold-af2-multimer-int5',
    pdbText: '',
    caCoords: new Float32Array(60),
    plddt: new Float32Array(20).fill(85),
    meanPlddt: 85,
    confidence: 85,
    confidenceAvailable: true,
    ptm: 0.75,
    iptm: 0.65,
    chainLengths: [10, 10],
    elapsedMs: 100,
    ...overrides,
  };
}

describe('localfold-msa: .a3m validation & parsing', () => {
  it('validates single-chain A3M with lowercase insertions and ColabFold # header', () => {
    const a3m = [
      '#10\t1',
      '>query',
      'MKTAYIAKQR',
      '>hit1',
      'MKTaaAYIAKQK',
      '>hit2',
      'MKT-YIAKQR',
    ].join('\n');
    const res = validateAndParseA3mText(a3m, ['MKTAYIAKQR']);
    expect(res.query).toBe('MKTAYIAKQR');
    expect(res.depth).toBe(3);
    expect(res.distinctDepth).toBe(3);
    expect(res.length).toBe(10);
    expect(res.wasPerChainBlocks).toBe(false);
    expect(res.a3m).toContain('MKTaaAYIAKQK');
  });

  it('validates concatenated multi-chain A3M', () => {
    const a3m = [
      '#10,8\t1,1',
      '>query',
      'MKTAYIAKQRACDEFGHI',
      '>h1',
      'MKTAYIAKQK--------',
      '>h2',
      '----------ACDEFGHK',
    ].join('\n');
    const res = validateAndParseA3mText(a3m, ['MKTAYIAKQR', 'ACDEFGHI']);
    expect(res.query).toBe('MKTAYIAKQRACDEFGHI');
    expect(res.depth).toBe(3);
    expect(res.wasPerChainBlocks).toBe(false);
  });

  it('validates per-chain A3M blocks separated by # or > headers', () => {
    const hashSeparated = [
      '#chain A',
      '>qA',
      'MKTAYIAKQR',
      '>hitA1',
      'MKTAYIAKQK',
      '#chain B',
      '>qB',
      'ACDEFGHI',
      '>hitB1',
      'ACDEFGHK',
      '>hitB2',
      'ACDEFGHL',
    ].join('\n');
    const resHash = validateAndParseA3mText(hashSeparated, ['MKTAYIAKQR', 'ACDEFGHI']);
    expect(resHash.wasPerChainBlocks).toBe(true);
    expect(resHash.perChainA3ms).toHaveLength(2);
    expect(resHash.depth).toBe(3);

    const fastaSeparated = [
      '>qA',
      'MKTAYIAKQR',
      '>hitA1',
      'MKTAYIAKQK',
      '>qB',
      'ACDEFGHI',
      '>hitB1',
      'ACDEFGHK',
    ].join('\n');
    const resFasta = validateAndParseA3mText(fastaSeparated, ['MKTAYIAKQR', 'ACDEFGHI']);
    expect(resFasta.wasPerChainBlocks).toBe(true);
    expect(resFasta.perChainA3ms).toHaveLength(2);
    expect(resFasta.query).toBe('MKTAYIAKQRACDEFGHI');
  });

  it('fails cleanly on empty text or query mismatch', () => {
    expect(() => validateAndParseA3mText('   \n  ', ['MKTAYIAKQR'])).toThrow(/empty/i);
    expect(() =>
      validateAndParseA3mText('>query\nGGGGGGGGGG\n', ['MKTAYIAKQR']),
    ).toThrow(/mismatch/i);
    expect(() =>
      validateAndParseA3mText('>query\nMKT-YIAKQR\n', ['MKTAYIAKQR']),
    ).toThrow(/ungapped/i);
  });
});

describe('localfold-msa: target-only chain-off masking', () => {
  it('searches target/locked chains and skips de novo binder/scaffold chains', () => {
    expect(computeTargetOnlyOffMask(['A', 'B'], ['A'], undefined)).toEqual([false, true]);
    expect(
      computeTargetOnlyOffMask(['A', 'B'], undefined, { A: 'target', B: 'binder' }),
    ).toEqual([false, true]);
    expect(
      computeTargetOnlyOffMask(['A', 'B'], undefined, { A: 'binder', B: 'target' }),
    ).toEqual([true, false]);
    expect(
      computeTargetOnlyOffMask(['A', 'B', 'C'], ['A'], { B: 'target', C: 'binder' }),
    ).toEqual([false, false, true]);
  });

  it('falls back to searching all chains when no locks or roles are present', () => {
    expect(computeTargetOnlyOffMask(['A', 'B'], undefined, undefined)).toEqual([false, false]);
    expect(computeTargetOnlyOffMask(['A', 'B'], [], {})).toEqual([false, false]);
    expect(computeTargetOnlyOffMask(['A'], undefined, undefined)).toEqual([false]);
  });

  it('skips explicit single-chain de novo scaffold in target-only mode', () => {
    expect(computeTargetOnlyOffMask(['A'], undefined, { A: 'scaffold' })).toEqual([true]);
  });
});

describe('localfold-msa: mock MMseqs2 fetch + target-only cache reuse across 4 binder candidates', () => {
  it('queries api.colabfold.com ONCE for the shared target across 4 MPNN binder designs using real vendored mmseqs2-api.js', async () => {
    const [mmseqsMod, a3mMod, manifestsMod] = await Promise.all([
      importLocalFold('shared/input/mmseqs2-api.js'),
      importLocalFold('shared/input/a3m.js'),
      importLocalFold('shared/bundles/manifests/index.js'),
    ]);

    const TARGET = 'MKTAYIAKQRQISFVKSHFS';
    const BINDERS = [
      'ACDEFGHIKLMNPQRSTVWY',
      'ACDEFGHIKLMNPQRSTVWA',
      'ACDEFGHIKLMNPQRSTVWC',
      'ACDEFGHIKLMNPQRSTVWD',
    ];

    const unirefA3m = [
      '>101',
      TARGET,
      '>UniRef100_1',
      'MKTAYIAKQRQISFVKSHFA',
      '>UniRef100_2',
      'MKTAYIAKQRQISFVKSHFG',
      '>UniRef100_3',
      'MKTAYIAKQRQISFVKSHFL',
    ].join('\n');
    const envA3m = [
      '>101',
      TARGET,
      '>Env_1',
      'MKTAYIAKQRQISFVKSHFV',
    ].join('\n');
    const tarGz = gzipSync(
      buildTarArchive([
        { name: 'uniref.a3m', content: `${unirefA3m}\n` },
        { name: 'bfd.mgnify30.metaeuk30.smag30.a3m', content: `${envA3m}\n` },
      ]),
    );

    const submittedUrls: string[] = [];
    const mockFetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      submittedUrls.push(url);
      if (url.includes('/ticket/msa')) {
        return new Response(JSON.stringify({ id: 'job-target-1', status: 'COMPLETE' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.includes('/result/download/job-target-1')) {
        return new Response(tarGz, { status: 200 });
      }
      throw new Error(`Unexpected mock fetch URL: ${url}`);
    }) as unknown as typeof fetch;

    const mmseqsApi: MmseqsApi = {
      generateMmseqs2Msa: mmseqsMod.generateMmseqs2Msa,
      generateMmseqs2ComplexMsa: mmseqsMod.generateMmseqs2ComplexMsa,
      expandSearchedChains: mmseqsMod.expandSearchedChains,
      mergeSearchedChains: mmseqsMod.mergeSearchedChains,
      singleSequenceA3m: mmseqsMod.singleSequenceA3m,
      parseA3m: a3mMod.parseA3m,
      graphFamily: manifestsMod.graphFamily,
    };

    const cache = createMsaSearchCache();
    const progressLog: string[] = [];

    const results = [];
    for (const binder of BINDERS) {
      const res = await resolveMsaForCandidate({
        msaMode: 'target-only',
        modelKey: 'boltz2',
        chainSeqs: [TARGET, binder],
        chainIds: ['A', 'B'],
        lockedChains: ['A'],
        chainRoles: { A: 'target', B: 'binder' },
        mmseqsApi,
        cache,
        onProgress: (ev) => progressLog.push(ev.detail),
        requestOptions: {
          fetchImplementation: mockFetch,
          wait: async () => {},
        },
      });
      results.push(res);
    }

    // ONLY ONE MMseqs2 submission + ONE result download across all 4 binder candidates!
    expect(submittedUrls.filter((u) => u.includes('/ticket/msa'))).toHaveLength(1);
    expect(submittedUrls.filter((u) => u.includes('/result/download/'))).toHaveLength(1);
    expect(results[0].cacheHit).toBe(false);
    expect(results[1].cacheHit).toBe(true);
    expect(results[2].cacheHit).toBe(true);
    expect(results[3].cacheHit).toBe(true);

    // Every candidate preserves its own binder query while sharing the target's homolog rows
    for (let i = 0; i < BINDERS.length; i++) {
      const parsed = a3mMod.parseA3m(results[i].a3m);
      expect(parsed.query).toBe(TARGET + BINDERS[i]);
      expect(results[i].msaDepth).toBeGreaterThan(1);
      expect(results[i].offMask).toEqual([false, true]);
      // Binder columns on homolog rows must be gapped (query-only!)
      expect(parsed.sequences[1].slice(TARGET.length)).toBe('-'.repeat(BINDERS[i].length));
      // Target columns on homolog rows carry the retrieved homologs
      expect(parsed.sequences[1].slice(0, TARGET.length)).toBe('MKTAYIAKQRQISFVKSHFA');
    }

    expect(progressLog.some((msg) => msg.includes('Fetching MMseqs2 MSA'))).toBe(true);
    expect(progressLog.some((msg) => msg.includes('MSA depth:'))).toBe(true);
    expect(progressLog.some((msg) => msg.includes('(cached)'))).toBe(true);
  });

  it('runs full mmseqs2 complex search with pairing on AF3/Multimer and skips pairing on AF2 monomer', async () => {
    const [mmseqsMod, a3mMod, manifestsMod] = await Promise.all([
      importLocalFold('shared/input/mmseqs2-api.js'),
      importLocalFold('shared/input/a3m.js'),
      importLocalFold('shared/bundles/manifests/index.js'),
    ]);

    const SEQ_A = 'MKTAYIAKQR';
    const SEQ_B = 'ACDEFGHIKL';

    const makeMsaTar = (q: string, hit: string) =>
      gzipSync(
        buildTarArchive([
          { name: 'uniref.a3m', content: `>101\n${q}\n>h1\n${hit}\n` },
          { name: 'bfd.mgnify30.metaeuk30.smag30.a3m', content: `>101\n${q}\n` },
        ]),
      );
    const pairTar = gzipSync(
      buildTarArchive([
        {
          name: 'pair.a3m',
          content: `>101\n${SEQ_A}\n>sp1\nMKTAYIAKQK\n\0>102\n${SEQ_B}\n>sp1\nACDEFGHIKM\n`,
        },
      ]),
    );

    const calledEndpoints: string[] = [];
    const mockFetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/ticket/msa')) {
        calledEndpoints.push('msa');
        const bodyStr = String(init?.body ?? '');
        const isA = bodyStr.includes(SEQ_A);
        return new Response(JSON.stringify({ id: isA ? 'tA' : 'tB', status: 'COMPLETE' }), { status: 200 });
      }
      if (url.includes('/ticket/pair')) {
        calledEndpoints.push('pair');
        return new Response(JSON.stringify({ id: 'tPair', status: 'COMPLETE' }), { status: 200 });
      }
      if (url.includes('/result/download/tA')) return new Response(makeMsaTar(SEQ_A, 'MKTAYIAKQV'), { status: 200 });
      if (url.includes('/result/download/tB')) return new Response(makeMsaTar(SEQ_B, 'ACDEFGHIKV'), { status: 200 });
      if (url.includes('/result/download/tPair')) return new Response(pairTar, { status: 200 });
      throw new Error(`Unexpected URL ${url}`);
    }) as unknown as typeof fetch;

    const mmseqsApi: MmseqsApi = {
      generateMmseqs2Msa: mmseqsMod.generateMmseqs2Msa,
      generateMmseqs2ComplexMsa: mmseqsMod.generateMmseqs2ComplexMsa,
      expandSearchedChains: mmseqsMod.expandSearchedChains,
      mergeSearchedChains: mmseqsMod.mergeSearchedChains,
      singleSequenceA3m: mmseqsMod.singleSequenceA3m,
      parseA3m: a3mMod.parseA3m,
      graphFamily: manifestsMod.graphFamily,
    };

    const cache = createMsaSearchCache();
    const resMultimer = await resolveMsaForCandidate({
      msaMode: 'mmseqs2',
      modelKey: 'multimer-2',
      chainSeqs: [SEQ_A, SEQ_B],
      chainIds: ['A', 'B'],
      mmseqsApi,
      cache,
      requestOptions: { fetchImplementation: mockFetch, wait: async () => {} },
    });

    expect(calledEndpoints.filter((e) => e === 'msa')).toHaveLength(2);
    expect(calledEndpoints.filter((e) => e === 'pair')).toHaveLength(1);
    expect(resMultimer.pairedDepth).toBe(2);
    expect(resMultimer.msaDepth).toBeGreaterThanOrEqual(3);
    expect(resMultimer.af3Alignment).toBeTruthy();

    // Reusing the same cache for AF2 monomer hits the pair:1 cache with ZERO extra fetches and drops paired rows
    const resMono = await resolveMsaForCandidate({
      msaMode: 'mmseqs2',
      modelKey: 'monomer-3',
      chainSeqs: [SEQ_A, SEQ_B],
      chainIds: ['A', 'B'],
      mmseqsApi,
      cache,
      requestOptions: { fetchImplementation: mockFetch, wait: async () => {} },
    });
    expect(resMono.cacheHit).toBe(true);
    expect(resMono.pairedDepth).toBe(0);
    expect(calledEndpoints).toHaveLength(3);
  });

  it('single mode makes zero external calls and returns af3Alignment=null', async () => {
    const [mmseqsMod, a3mMod] = await Promise.all([
      importLocalFold('shared/input/mmseqs2-api.js'),
      importLocalFold('shared/input/a3m.js'),
    ]);
    const spySingle = vi.fn();
    const spyComplex = vi.fn();
    const mmseqsApi: MmseqsApi = {
      generateMmseqs2Msa: spySingle,
      generateMmseqs2ComplexMsa: spyComplex,
      expandSearchedChains: mmseqsMod.expandSearchedChains,
      mergeSearchedChains: mmseqsMod.mergeSearchedChains,
      singleSequenceA3m: mmseqsMod.singleSequenceA3m,
      parseA3m: a3mMod.parseA3m,
    };

    const res = await resolveMsaForCandidate({
      msaMode: 'single',
      modelKey: 'openbind0',
      chainSeqs: ['MKTAYIAKQR'],
      chainIds: ['A'],
      mmseqsApi,
    });
    expect(spySingle).not.toHaveBeenCalled();
    expect(spyComplex).not.toHaveBeenCalled();
    expect(res.af3Alignment).toBeNull();
    expect(res.msaDepth).toBe(1);
  });
});

describe('localfold-msa: AF3 / OpenBind duplicateQueryRow, af3BatchFromA3m & conformerSeed parity', () => {
  it('af3BatchFromA3m gives AF3 and OpenBind-0 exact upstream single-sequence MSA depth 2 (duplicateQueryRow) and depth 1 for Boltz-2 / RF3 / Protenix-2', async () => {
    const [dialectMod, batchMod] = await Promise.all([
      importLocalFold('shared/af3/dialect.js'),
      importLocalFold('shared/af3/featurise/batch.js'),
    ]);
    const seq = 'MKTAYIAKQRQISFVKSHFS';

    for (const model of ['af3', 'openbind0']) {
      const featOpts = dialectMod.featuriserDialect(dialectMod.dialectFor(model));
      expect(featOpts.dedupeSelfMsa).toBe(false);
      const { batch } = batchMod.af3BatchFromA3m(seq, null, { ...featOpts, seed: 42 });
      const msaRows = batch.msa.length / batch.tokens;
      expect(msaRows, `${model} single-sequence MSA depth`).toBe(2);
    }

    for (const model of ['boltz2', 'rosettafold3', 'protenix2', 'opendde']) {
      const featOpts = dialectMod.featuriserDialect(dialectMod.dialectFor(model));
      expect(featOpts.dedupeSelfMsa).toBe(true);
      const { batch } = batchMod.af3BatchFromA3m(seq, null, { ...featOpts, seed: 42 });
      const msaRows = batch.msa.length / batch.tokens;
      expect(msaRows, `${model} single-sequence MSA depth`).toBe(1);
    }
  });

  it('af3BatchFromA3m feeds real MSA rows into AF3/OpenBind/Boltz-2 and propagates seed -> conformerSeed on Protenix-2', async () => {
    const [dialectMod, batchMod] = await Promise.all([
      importLocalFold('shared/af3/dialect.js'),
      importLocalFold('shared/af3/featurise/batch.js'),
    ]);
    const seq = 'MKTAYIAKQR';
    const a3m = [
      '>query',
      'MKTAYIAKQR',
      '>h1',
      'MKTAYIAKQK',
      '>h2',
      'MKTAYIAKQL',
      '>h3',
      'MKTAYIAKQV',
    ].join('\n');

    const obOpts = dialectMod.featuriserDialect(dialectMod.dialectFor('openbind0'));
    const { batch: obBatch } = batchMod.af3BatchFromA3m(seq, a3m, { ...obOpts, seed: 7 });
    // AF3/OpenBind appends the 4 unpaired rows after query row 0 -> 5 rows total
    expect(obBatch.msa.length / obBatch.tokens).toBeGreaterThanOrEqual(4);

    // Protenix-2 has translateConformers=true -> different seeds move refPos via conformerSeed
    const pxOpts = dialectMod.featuriserDialect(dialectMod.dialectFor('protenix2'));
    expect(pxOpts.translateConformers).toBe(true);
    const { batch: b0 } = batchMod.af3BatchFromA3m(seq, null, { ...pxOpts, seed: 0 });
    const { batch: b1 } = batchMod.af3BatchFromA3m(seq, null, { ...pxOpts, seed: 1 });
    let diff = 0;
    for (let k = 0; k < b0.refPos.length; k++) {
      diff += Math.abs(b0.refPos[k] - b1.refPos[k]);
    }
    expect(diff).toBeGreaterThan(0);
  });
});

describe('localfold-msa: AF2/AF3 fidelity helpers (recycles, tolerance, MSA budgets, multimer ranking)', () => {
  it('enforces model-specific recycle ceilings (20 for mol1 & multimer*, 10 for others)', () => {
    expect(maxRecyclesForLocalFoldModel('mol1')).toBe(20);
    expect(maxRecyclesForLocalFoldModel('multimer')).toBe(20);
    expect(maxRecyclesForLocalFoldModel('multimer-3')).toBe(20);
    expect(maxRecyclesForLocalFoldModel('monomer')).toBe(10);
    expect(maxRecyclesForLocalFoldModel('monomer-5')).toBe(10);
    expect(maxRecyclesForLocalFoldModel('boltz2')).toBe(10);
    expect(maxRecyclesForLocalFoldModel('openbind0')).toBe(10);

    expect(clampLocalFoldRecycles('mol1', 25)).toBe(20);
    expect(clampLocalFoldRecycles('multimer-2', 18)).toBe(18);
    expect(clampLocalFoldRecycles('monomer', 15)).toBe(10);
    expect(clampLocalFoldRecycles('rosettafold3', 8)).toBe(8);
    expect(clampLocalFoldRecycles('monomer', -2)).toBe(0);
  });

  it('defaults tolerance to 0 for AF2 Monomer and 0.5 for AF2-Multimer, honoring explicit overrides', () => {
    expect(resolveAf2Tolerance('monomer', undefined)).toBe(0);
    expect(resolveAf2Tolerance('monomer-4', '')).toBe(0);
    expect(resolveAf2Tolerance('multimer', undefined)).toBe(0.5);
    expect(resolveAf2Tolerance('multimer-3', null)).toBe(0.5);
    expect(resolveAf2Tolerance('monomer', 0.25)).toBe(0.25);
    expect(resolveAf2Tolerance('multimer', 0)).toBe(0);
    expect(resolveAf2Tolerance('multimer', 99)).toBe(5);
  });

  it('unlocks AF2 cluster & extra MSA budgets when msaDepth > 1', () => {
    expect(computeAf2MsaBudgets(1, 2)).toEqual({
      maxMsaSequences: 3,
      maxExtraSequences: 1,
      maskFraction: 0,
    });
    expect(computeAf2MsaBudgets(250, 2, 240)).toEqual({
      maxMsaSequences: 250,
      maxExtraSequences: 250,
      maskFraction: 0.15,
    });
    expect(computeAf2MsaBudgets(5000, 2, 4800)).toEqual({
      maxMsaSequences: 508,
      maxExtraSequences: 1024,
      maskFraction: 0.15,
    });
    // Zero-homolog MMseqs2 result (depth 2 duplicate query, distinct 1) keeps maskFraction=0
    expect(computeAf2MsaBudgets(2, 1, 1).maskFraction).toBe(0);
  });

  it('ranks multi-chain folds by multimerScore (0.8 * ipTM + 0.2 * pTM) instead of meanPlddt', () => {
    const highPlddtWeakInterface = makeStubCandidate({
      sequenceId: 'seq-high-plddt',
      meanPlddt: 92.5,
      confidence: 92.5,
      ptm: 0.72,
      iptm: 0.35,
      multimerScore: 0.8 * 0.35 + 0.2 * 0.72, // 0.424
    });
    const strongInterface = makeStubCandidate({
      sequenceId: 'seq-strong-interface',
      meanPlddt: 84.0,
      confidence: 84.0,
      ptm: 0.85,
      iptm: 0.82,
      multimerScore: 0.8 * 0.82 + 0.2 * 0.85, // 0.826
    });

    const rep = selectLocalFoldRepresentative([highPlddtWeakInterface, strongInterface]);
    expect(rep).not.toBeNull();
    expect(rep!.fold.sequenceId).toBe('seq-strong-interface');
    expect(rep!.selection.criterion).toBe('highest-multimer-score');
    expect(rep!.selection.scoreValue).toBeCloseTo(0.826, 5);

    // Single-chain folds still rank by meanPlddt via selectFoldRepresentative
    const monoA = makeStubCandidate({ sequenceId: 'm1', chainLengths: [20], meanPlddt: 78, confidence: 78, iptm: null, multimerScore: undefined });
    const monoB = makeStubCandidate({ sequenceId: 'm2', chainLengths: [20], meanPlddt: 91, confidence: 91, iptm: null, multimerScore: undefined });
    const monoRep = selectLocalFoldRepresentative([monoA, monoB]);
    expect(monoRep!.fold.sequenceId).toBe('m2');
    expect(monoRep!.selection.criterion).toBe('highest-mean-plddt');
  });
});
