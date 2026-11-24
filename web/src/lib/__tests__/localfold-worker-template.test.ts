import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import {
  existsSync,
  mkdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { FoldedStructure, SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';

const WEB_ROOT = resolve(__dirname, '../../..');
const VENDOR_ROOT = resolve(WEB_ROOT, 'static/vendor/localfold');

const SEQ_A = 'MKTAYIAKQRQISFVKSHFSRQ';
const SEQ_B = 'ACDEFGHIKLMNPQRSTVWYAA';

const AA1_TO_3: Record<string, string> = {
  A: 'ALA',
  C: 'CYS',
  D: 'ASP',
  E: 'GLU',
  F: 'PHE',
  G: 'GLY',
  H: 'HIS',
  I: 'ILE',
  K: 'LYS',
  L: 'LEU',
  M: 'MET',
  N: 'ASN',
  P: 'PRO',
  Q: 'GLN',
  R: 'ARG',
  S: 'SER',
  T: 'THR',
  V: 'VAL',
  W: 'TRP',
  Y: 'TYR',
};

function makeBackbonePdb(
  chains: Array<{ chainId: string; sequence: string; shiftX?: number }>,
): string {
  const lines: string[] = [];
  let serial = 1;
  for (let c = 0; c < chains.length; c++) {
    const ch = chains[c];
    const shiftX = ch.shiftX ?? 0;
    for (let i = 0; i < ch.sequence.length; i++) {
      const res3 = AA1_TO_3[ch.sequence[i]] ?? 'ALA';
      const resSeq = i + 1;
      const baseX = i * 3.8 + shiftX;
      const baseY = c * 14.0;
      const atoms: Array<[string, number, number, number]> = [
        ['N', baseX - 1.2, baseY - 0.7, 0],
        ['CA', baseX, baseY, 0],
        ['C', baseX + 1.3, baseY - 0.7, 0],
        ['O', baseX + 1.4, baseY - 1.9, 0],
        ...(ch.sequence[i] !== 'G' ? [['CB', baseX + 0.1, baseY + 1.4, 0.5] as [string, number, number, number]] : []),
      ];
      for (const [name, x, y, z] of atoms) {
        lines.push(
          `ATOM  ${String(serial++).padStart(5, ' ')}  ${name.padEnd(3, ' ')} ${res3} ${ch.chainId[0]}${String(resSeq).padStart(4, ' ')}    ${x.toFixed(3).padStart(8, ' ')}${y.toFixed(3).padStart(8, ' ')}${z.toFixed(3).padStart(8, ' ')}  1.00 85.00           ${name[0]}`,
        );
      }
    }
    lines.push(`TER   ${String(serial++).padStart(5, ' ')}      ${AA1_TO_3[ch.sequence.at(-1) ?? 'A'] ?? 'ALA'} ${ch.chainId[0]}${String(ch.sequence.length).padStart(4, ' ')}`);
  }
  lines.push('END');
  return `${lines.join('\n')}\n`;
}

function makeBackboneMmcif(
  chains: Array<{ chainId: string; sequence: string }>,
): string {
  const lines: string[] = [
    'data_tpl',
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
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.pdbx_PDB_model_num',
  ];
  let serial = 1;
  for (let c = 0; c < chains.length; c++) {
    const ch = chains[c];
    for (let i = 0; i < ch.sequence.length; i++) {
      const res3 = AA1_TO_3[ch.sequence[i]] ?? 'ALA';
      const resSeq = i + 1;
      const baseX = i * 3.8;
      const baseY = c * 14.0;
      for (const [name, dx, dy] of [
        ['N', -1.2, -0.7],
        ['CA', 0, 0],
        ['C', 1.3, -0.7],
        ['O', 1.4, -1.9],
      ] as const) {
        lines.push(
          `ATOM ${serial++} ${name[0]} ${name} . ${res3} ${ch.chainId} ${ch.chainId} ${resSeq} ${resSeq} ${(baseX + dx).toFixed(3)} ${(baseY + dy).toFixed(3)} 0.000 1.00 88.00 1`,
        );
      }
    }
  }
  return `${lines.join('\n')}\n`;
}

function makeStructureArtifact(id: string, pdbText: string, cifText?: string): StructureArtifact {
  const parsed = parseStructureText(cifText ?? pdbText);
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText,
    ...(cifText ? { cifText } : {}),
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    chainIdMap: parsed.chainIdMap,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    geometry: computeGeometryStats(
      parsed.caCoords,
      parsed.chainIds,
      parsed.resNums,
      parsed.insCodes,
    ),
  };
}

function makeSeqSet(id: string, candidates: SequenceCandidate[]): SequenceSet {
  return {
    kind: 'SequenceSet',
    id,
    parentStructureId: 'seq',
    sourceModelId: 'sequence-input',
    candidates,
  };
}

interface WorkerCapture {
  family: 'af2-monomer' | 'af2-multimer' | 'af3';
  method?: 'predictA3m' | 'predict';
  options?: Record<string, unknown>;
  featuresByRecycle?: Array<{ asymId?: ArrayLike<number> }>;
  batch?: { tokens: number; residueOfToken?: ArrayLike<number>; asymId?: ArrayLike<number> };
}

describe('LocalFold worker template integration (AF2, AF2-Multimer, AF3-family, runHash & provenance)', () => {
  let stubOriginDir = '';
  let stubOriginUrl = '';

  beforeAll(() => {
    stubOriginDir = resolve(WEB_ROOT, `.vitest-lf-worker-origin-${process.pid}`);
    rmSync(stubOriginDir, { recursive: true, force: true });
    const lfRoot = resolve(stubOriginDir, 'vendor/localfold');
    mkdirSync(lfRoot, { recursive: true });
    writeFileSync(resolve(stubOriginDir, 'package.json'), '{"type":"module"}\n', 'utf8');

    const safeLink = (target: string, linkPath: string) => {
      mkdirSync(dirname(linkPath), { recursive: true });
      if (!existsSync(linkPath)) symlinkSync(target, linkPath);
    };
    for (const sub of ['af2', 'bioemu2', 'chem', 'esmfold2', 'heads', 'input', 'weights']) {
      safeLink(resolve(VENDOR_ROOT, 'shared', sub), resolve(lfRoot, 'shared', sub));
    }
    for (const item of [
      'chain-geometry.js',
      'dialect.js',
      'diffusion',
      'feature-convergence.js',
      'featurise',
      'msa-draw.js',
      'random.js',
      'structure',
      'trunk',
    ]) {
      safeLink(resolve(VENDOR_ROOT, 'shared/af3', item), resolve(lfRoot, 'shared/af3', item));
    }

    const writeStub = (relPath: string, code: string) => {
      const p = resolve(lfRoot, relPath);
      mkdirSync(dirname(p), { recursive: true });
      writeFileSync(p, code, 'utf8');
    };

    writeStub(
      'webgpu/runtime/device.js',
      `export async function requestAlphaFoldDevice(adapter) {
        return { adapter, limits: { maxStorageBufferBindingSize: 1024 * 1024 * 1024 }, lost: new Promise(() => {}), destroy() {} };
      }\n`,
    );
    writeStub(
      'webgpu/runtime/resident.js',
      `export function releaseAllWeights() {}
      export function releaseResidentWeights() {}\n`,
    );
    writeStub(
      'shared/bundles/manifests/index.js',
      `export const MODEL_BUNDLES = {
        'monomer-2': { delta: { base: 'monomer' } },
        'monomer-3': { delta: { base: 'monomer' } },
        'multimer-2': { delta: { base: 'multimer' } },
      };
      export function graphFamily(m) { return String(m || '').startsWith('multimer') ? 'multimer' : 'monomer'; }
      export function bundleBaseUrl(m) { return 'https://stub.invalid/bundles/' + m; }
      export async function loadManifest(m) { return { model: m, tensors: {} }; }\n`,
    );
    writeStub(
      'shared/bundles/http-tensor-store.js',
      `export class HttpTensorStore {
        static async fromManifest(base, manifest) { return new HttpTensorStore(); }
        prefetch() {}
        releaseDecoded() {}
      }\n`,
    );
    writeStub(
      'shared/bundles/delta-tensor-store.js',
      `export class DeltaTensorStore {
        constructor(baseStore, deltaStore) {}
        releaseDecoded() {}
      }\n`,
    );
    writeStub(
      'shared/bundles/alphafold-fixture.js',
      `export class AlphaFoldFixture {
        static fromStore() { return new AlphaFoldFixture(); }
        async embeddingWeights() { return {}; }
        async templateWeights() { return {}; }
        async templateEmbeddingWeights() { return {}; }
        async extraStackWeights() { return {}; }
        async mainStackWeights() { return {}; }
        async structureWeights() { return {}; }
        async confidenceWeights() { return { lddt: {}, pae: {} }; }
        async geometryTables() { return {}; }
        async queryOnlyFeatureTables() { return {}; }
        async tensor() { return new Float32Array(64); }
      }\n`,
    );
    writeStub(
      'shared/af3/weights/weights.js',
      `export function releaseWeights() {}\n`,
    );
    writeStub(
      'shared/af3/weights/diffusion-weights.js',
      `import { featuriserDialect } from '../dialect.js';
      export async function foldWeights(store) {
        const model = globalThis.__LF_TEST_MODEL__ || 'af3';
        const d = featuriserDialect(model);
        return { trunk: { dialect: { ...d, modelName: model } } };
      }\n`,
    );
    writeStub(
      'webgpu/input/a3m-features.js',
      `export async function makeA3mFeaturesFor(device, a3mText, tables, options = {}) {
        const lines = String(a3mText).split(/\\r?\\n/).filter((l) => l && !l.startsWith('>'));
        const len = (lines[0] || '').replace(/[^A-Za-z]/g, '').length || 10;
        const chainLengths = Array.isArray(options.chainLengths) ? options.chainLengths : [len];
        const recycles = Number(options.recycles ?? 1);
        const out = [];
        for (let r = 0; r <= recycles; r++) {
          const entry = { length: len };
          if (options.chainAware && chainLengths.length > 1) {
            const asym = new Float32Array(len);
            let off = 0;
            for (let c = 0; c < chainLengths.length; c++) {
              for (let k = 0; k < chainLengths[c]; k++) asym[off++] = c;
            }
            entry.asymId = asym;
          }
          out.push(entry);
        }
        return out;
      }\n`,
    );
    const af2OutHelper = `function synthAf2Out(len, tpl, chainLengths) {
      const atom37 = new Float32Array(len * 37 * 3);
      const atom37Mask = new Float32Array(len * 37);
      for (let i = 0; i < len; i++) {
        const base = (i * 37 + 1) * 3;
        atom37[base] = i * 3.8;
        atom37[base + 1] = tpl ? 1.0 : 0.0;
        atom37Mask[i * 37 + 1] = 1;
      }
      const plddt = new Float32Array(len).fill(tpl ? 92.0 : 80.0);
      const pae = new Float32Array(len * len).fill(2.0);
      return {
        final: {
          structure: { atom37, atom37Mask },
          confidence: { plddt, ptm: tpl ? 0.9 : 0.78, iptm: chainLengths.length > 1 ? 0.85 : undefined, predictedAlignedError: pae },
        },
        recycles: [{}],
      };
    }`;
    writeStub(
      'webgpu/af2/model/monomer.js',
      `${af2OutHelper}
      export class AlphaFoldMonomerGpu {
        constructor(device) { this.device = device; }
        async predictA3m(a3mText, weights, tables, options = {}) {
          globalThis.__LF_TEST_CAPTURES__?.push({ family: 'af2-monomer', method: 'predictA3m', options });
          const len = (options.chainLengths || [22]).reduce((a, b) => a + b, 0);
          return synthAf2Out(len, options.template, options.chainLengths || [len]);
        }
      }\n`,
    );
    writeStub(
      'webgpu/af2/multimer/model.js',
      `${af2OutHelper}
      export class AlphaFoldUnifiedGpu {
        constructor(device) { this.device = device; }
        async predictA3m(a3mText, weights, tables, options = {}) {
          globalThis.__LF_TEST_CAPTURES__?.push({ family: 'af2-multimer', method: 'predictA3m', options });
          if (options.template) throw new Error('predictA3m does not forward 1-chain asymId');
          const len = (options.chainLengths || [22]).reduce((a, b) => a + b, 0);
          return synthAf2Out(len, options.template, options.chainLengths || [len]);
        }
        async predict(featuresByRecycle, weights, paeBreaks, onRecycle, onProgress, options = {}) {
          globalThis.__LF_TEST_CAPTURES__?.push({
            family: 'af2-multimer',
            method: 'predict',
            options,
            featuresByRecycle,
          });
          if (options.template && (!featuresByRecycle?.[0] || featuresByRecycle[0].asymId === undefined)) {
            throw new Error('a template needs asymId (or multichainMask2d)');
          }
          const len = featuresByRecycle?.[0]?.length || 22;
          return synthAf2Out(len, options.template, options.chainLengths || [len]);
        }
      }\n`,
    );
    writeStub(
      'webgpu/af3/fold.js',
      `export async function foldBatch(device, batch, weights, options) {
        globalThis.__LF_TEST_CAPTURES__?.push({ family: 'af3', batch, options });
        const resOf = batch?.residueOfToken || [];
        const resCount = resOf.length ? Math.max(...Array.from(resOf)) + 1 : (batch?.tokens || 10);
        const lines = [];
        let ser = 1;
        for (let r = 0; r < resCount; r++) {
          for (const [nm, dx, dy] of [['N', -1.2, -0.7], ['CA', 0, 0], ['C', 1.3, -0.7], ['O', 1.4, -1.9]]) {
            lines.push('ATOM  ' + String(ser++).padStart(5, ' ') + '  ' + nm.padEnd(3, ' ') + ' ALA A' + String(r + 1).padStart(4, ' ') + '    ' + (r * 3.8 + dx).toFixed(3).padStart(8, ' ') + dy.toFixed(3).padStart(8, ' ') + '   0.000  1.00 88.00           ' + nm[0]);
          }
        }
        lines.push('END');
        const paeDim = batch?.asymId?.length || resCount;
        return {
          pdb: lines.join('\\n') + '\\n',
          perResiduePlddt: Array.from({ length: resCount }, () => 89.0),
          meanPlddt: 89.0,
          ptm: 0.87,
          iptm: 0.81,
          scores: { pae: new Float32Array(paeDim * paeDim).fill(2.0) },
          reusable: { recycles: options?.recycles ?? 1 },
        };
      }\n`,
    );

    stubOriginUrl = pathToFileURL(stubOriginDir).href.replace(/\/+$/, '');
  });

  afterAll(() => {
    vi.unstubAllGlobals();
    if (stubOriginDir && existsSync(stubOriginDir)) {
      rmSync(stubOriginDir, { recursive: true, force: true });
    }
  });

  async function runWorker(
    params: Record<string, unknown>,
    inputs: Record<string, unknown>,
    extra?: { maxFoldResidues?: number },
  ): Promise<{
    status: 'ok' | 'error';
    outputs?: Record<string, unknown>;
    error?: string;
    captures: WorkerCapture[];
    fetchUrls: string[];
  }> {
    const captures: WorkerCapture[] = [];
    const fetchUrls: string[] = [];
    const messages: Array<Record<string, unknown>> = [];
    (globalThis as Record<string, unknown>).__LF_TEST_CAPTURES__ = captures;
    (globalThis as Record<string, unknown>).__LF_TEST_MODEL__ = String(params.model ?? 'monomer');

    const scope = {
      postMessage: (msg: Record<string, unknown>) => {
        messages.push(msg);
      },
      location: { origin: stubOriginUrl },
      onmessage: async (_ev: { data: unknown }) => {},
    };
    const mockAdapter = {
      features: new Set<string>(),
      limits: { maxStorageBufferBindingSize: 1024 * 1024 * 1024 },
      info: { vendor: 'stub', architecture: '', device: 'stub', description: 'stub' },
    };
    vi.stubGlobal('self', scope);
    vi.stubGlobal('navigator', {
      userAgent: 'Linux',
      hardwareConcurrency: 8,
      gpu: { requestAdapter: vi.fn(async () => mockAdapter) },
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string | URL) => {
        const url = String(input);
        fetchUrls.push(url);
        if (url.includes('/ccd/') || url.toLowerCase().includes('zn.cif')) {
          return new Response(
            [
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
            ].join('\n'),
            { status: 200 },
          );
        }
        return new Response('', { status: 404 });
      }),
    );

    vi.resetModules();
    await import('$lib/plugins/localfold-af2/localfold.worker');
    await scope.onmessage({
      data: {
        id: 1,
        type: 'run',
        params,
        inputs,
        ...(extra?.maxFoldResidues !== undefined
          ? { maxFoldResidues: extra.maxFoldResidues }
          : {}),
      },
    });

    const errMsg = messages.find((m) => m.type === 'error');
    if (errMsg) {
      return {
        status: 'error',
        error: String(errMsg.message ?? 'unknown error'),
        captures,
        fetchUrls,
      };
    }
    const doneMsg = messages.find((m) => m.type === 'result');
    return {
      status: 'ok',
      outputs: doneMsg?.outputs as Record<string, unknown> | undefined,
      captures,
      fetchUrls,
    };
  }

  it('wires AF2 Monomer (monomer & monomer-2) with PDB and mmCIF templates, GAP_AATYPE=21 for uncovered flanks, and full provenance', async () => {
    const pdb = makeBackbonePdb([{ chainId: 'A', sequence: SEQ_A }]);
    const cif = makeBackboneMmcif([{ chainId: 'A', sequence: SEQ_A }]);
    const seqWithTag = makeSeqSet('seq-tagged', [
      {
        id: 'c1',
        sequence: `HHHHHH${SEQ_A}`,
        sourceModelId: 'sequence-input',
        score: null,
        globalScore: null,
      },
    ]);

    const r1 = await runWorker(
      { model: 'monomer', msaMode: 'single', recycles: 1, templatePdbText: pdb, templateFilename: 'tpl.pdb' },
      { sequences: seqWithTag },
    );
    expect(r1.status).toBe('ok');
    expect(r1.captures).toHaveLength(1);
    const tpl1 = r1.captures[0].options?.template as { aatype: Int32Array; atomMask: Float32Array };
    expect(tpl1.aatype.length).toBe(6 + SEQ_A.length);
    for (let k = 0; k < 6; k++) expect(tpl1.aatype[k]).toBe(21); // GAP_AATYPE for uncovered N-term tag
    for (let k = 6; k < tpl1.aatype.length; k++) expect(tpl1.aatype[k]).toBeLessThan(20);

    const folds1 = r1.outputs?.folds as FoldedStructure;
    const struct1 = r1.outputs?.structure as StructureArtifact;
    expect(folds1.folds[0].provenance).toMatchObject({
      templateUsed: true,
      templateFormat: 'pdb',
      templateMatchedResidues: SEQ_A.length,
      templateTotalResidues: 6 + SEQ_A.length,
      templateSpanChains: true,
    });
    expect(struct1.provenance?.templateDigest).toBeTruthy();

    const r2 = await runWorker(
      { model: 'monomer-2', msaMode: 'single', recycles: 1 },
      { sequences: seqWithTag, template: makeStructureArtifact('tpl-cif', pdb, cif) },
    );
    expect(r2.status).toBe('ok');
    expect((r2.outputs?.folds as FoldedStructure).folds[0].provenance?.templateFormat).toBe('mmcif');
  });

  it('fixes AF2-Multimer 1-chain asymId via makeA3mFeaturesFor + runner.predict and distinguishes spanChains on 2-chain runs', async () => {
    const pdbMono = makeBackbonePdb([{ chainId: 'A', sequence: SEQ_A }]);
    const seqMono = makeSeqSet('s1', [
      { id: 'c1', sequence: SEQ_A, sourceModelId: 'sequence-input', score: null, globalScore: null },
    ]);

    const rMonoTpl = await runWorker(
      { model: 'multimer', msaMode: 'single', recycles: 1, templatePdbText: pdbMono },
      { sequences: seqMono },
    );
    expect(rMonoTpl.status).toBe('ok');
    expect(rMonoTpl.captures[0].method).toBe('predict');
    expect(rMonoTpl.captures[0].featuresByRecycle?.[0]?.asymId).toBeDefined();
    expect(rMonoTpl.captures[0].featuresByRecycle?.[0]?.asymId?.length).toBe(SEQ_A.length);

    // Untemplated AF2-Multimer remains on predictA3m
    const rMonoUntpl = await runWorker(
      { model: 'multimer', msaMode: 'single', recycles: 1 },
      { sequences: seqMono },
    );
    expect(rMonoUntpl.status).toBe('ok');
    expect(rMonoUntpl.captures[0].method).toBe('predictA3m');

    // 2-chain multimer spanChains true vs false
    const pdbComplex = makeBackbonePdb([
      { chainId: 'A', sequence: SEQ_A },
      { chainId: 'B', sequence: SEQ_B },
    ]);
    const seqComplex = makeSeqSet('s2', [
      {
        id: 'c2',
        sequence: `${SEQ_A}:${SEQ_B}`,
        chainLengths: [SEQ_A.length, SEQ_B.length],
        sourceModelId: 'sequence-input',
        score: null,
        globalScore: null,
      },
    ]);
    const rSpanT = await runWorker(
      { model: 'multimer-2', msaMode: 'single', recycles: 1, templatePdbText: pdbComplex, templateSpanChains: true },
      { sequences: seqComplex },
    );
    const rSpanF = await runWorker(
      { model: 'multimer-2', msaMode: 'single', recycles: 1, templatePdbText: pdbComplex, templateSpanChains: false },
      { sequences: seqComplex },
    );
    expect(rSpanT.status).toBe('ok');
    expect(rSpanF.status).toBe('ok');
    expect((rSpanT.captures[0].options?.template as { spanChains: boolean }).spanChains).toBe(true);
    expect((rSpanF.captures[0].options?.template as { spanChains: boolean }).spanChains).toBe(false);
    expect((rSpanT.outputs?.folds as FoldedStructure).id).not.toBe(
      (rSpanF.outputs?.folds as FoldedStructure).id,
    );
  });

  it('builds token-aligned dense-24 templateSlots across all 8 AF3 dialects including DNA + ligand co-folding', async () => {
    const pdb = makeBackbonePdb([{ chainId: 'A', sequence: SEQ_A }]);
    const seqSet = makeSeqSet('af3-s', [
      { id: 'c1', sequence: SEQ_A, sourceModelId: 'sequence-input', score: null, globalScore: null },
    ]);

    for (const model of [
      'af3',
      'intellifold2',
      'protenix2',
      'mol1',
      'boltz2',
      'rosettafold3',
      'openbind0',
      'opendde',
    ]) {
      const res = await runWorker(
        {
          model,
          msaMode: 'single',
          recycles: 1,
          diffusionSteps: 10,
          templatePdbText: pdb,
          dnaSequences: 'ACGT',
          ligandCcdCodes: 'ZN',
        },
        { sequences: seqSet },
      );
      expect(res.status, `model ${model} failed: ${res.error}`).toBe('ok');
      const cap = res.captures[0];
      expect(cap.family).toBe('af3');
      const slots = cap.options?.templateSlots as Array<{
        aatype: Int32Array;
        covered: number;
        atoms: number;
      }>;
      expect(slots).toHaveLength(1);
      expect(slots[0].aatype.length).toBe(cap.batch!.tokens);
      expect(slots[0].covered).toBe(SEQ_A.length);
      expect(slots[0].atoms).toBeGreaterThan(SEQ_A.length * 3);
      // Trailing DNA/ligand tokens remain GAP_AATYPE=21
      expect(slots[0].aatype[cap.batch!.tokens - 1]).toBe(21);
    }
  });

  it('handles multi-candidate SequenceSet loops with distinct sequences and records provenance per fold', async () => {
    const pdb = makeBackbonePdb([
      { chainId: 'A', sequence: SEQ_A },
      { chainId: 'B', sequence: SEQ_B },
    ]);
    const seqSet = makeSeqSet('multi-cand', [
      { id: 'candA', sequence: SEQ_A, sourceModelId: 'sequence-input', score: null, globalScore: null },
      { id: 'candB', sequence: SEQ_B, sourceModelId: 'sequence-input', score: null, globalScore: null },
    ]);
    const res = await runWorker(
      { model: 'monomer', msaMode: 'single', recycles: 1, maxSequences: 2, templatePdbText: pdb },
      { sequences: seqSet },
    );
    expect(res.status).toBe('ok');
    const folds = res.outputs?.folds as FoldedStructure;
    expect(folds.folds).toHaveLength(2);
    expect(folds.folds[0].provenance?.templateUsed).toBe(true);
    expect(folds.folds[1].provenance?.templateUsed).toBe(true);
    expect(folds.folds[0].provenance?.templateMatchedResidues).toBe(SEQ_A.length);
    expect(folds.folds[1].provenance?.templateMatchedResidues).toBe(SEQ_B.length);
  });

  it('fails pre-fetch (0 fetch calls) on template-free models, invalid templates, sequence mismatches, and maxFoldResidues ceiling precedence', async () => {
    const pdb = makeBackbonePdb([{ chainId: 'A', sequence: SEQ_A }]);
    const seqSet = makeSeqSet('s1', [
      { id: 'c1', sequence: SEQ_A, sourceModelId: 'sequence-input', score: null, globalScore: null },
    ]);

    for (const model of ['monomer-3', 'monomer-5', 'ef2-fast-300m']) {
      const r = await runWorker(
        { model, msaMode: 'single', templatePdbText: pdb },
        { sequences: seqSet },
      );
      expect(r.status).toBe('error');
      expect(r.error).toMatch(/template/i);
      expect(r.fetchUrls).toEqual([]);
    }

    const rInvalid = await runWorker(
      { model: 'monomer', msaMode: 'single', templatePdbText: 'not valid pdb' },
      { sequences: seqSet },
    );
    expect(rInvalid.status).toBe('error');
    expect(rInvalid.fetchUrls).toEqual([]);

    const rMismatch = await runWorker(
      { model: 'monomer', msaMode: 'single', templatePdbText: pdb },
      {
        sequences: makeSeqSet('smis', [
          { id: 'c1', sequence: 'WWWWWWWWWWWWWWWWWWWWWW', sourceModelId: 'sequence-input', score: null, globalScore: null },
        ]),
      },
    );
    expect(rMismatch.status).toBe('error');
    expect(rMismatch.error).toMatch(/overlap|identity|mismatch/i);
    expect(rMismatch.fetchUrls).toEqual([]);

    const rCeiling = await runWorker(
      { model: 'monomer', msaMode: 'single', templatePdbText: pdb },
      {
        sequences: makeSeqSet('slong', [
          { id: 'c1', sequence: `M${'A'.repeat(99)}`, sourceModelId: 'sequence-input', score: null, globalScore: null },
        ]),
      },
      { maxFoldResidues: 64 },
    );
    expect(rCeiling.status).toBe('error');
    expect(rCeiling.error).toMatch(/ceiling/i);
    expect(rCeiling.fetchUrls).toEqual([]);
  });

  it('preserves untemplated runHash parity, enforces port precedence, and invalidates runHash on coordinate changes', async () => {
    const pdb = makeBackbonePdb([{ chainId: 'A', sequence: SEQ_A }]);
    const pdbMoved = makeBackbonePdb([{ chainId: 'A', sequence: SEQ_A, shiftX: 5.0 }]);
    const pdbInlineOther = makeBackbonePdb([{ chainId: 'A', sequence: SEQ_A, shiftX: 12.0 }]);
    const seqSet = makeSeqSet('hash-s', [
      { id: 'c1', sequence: SEQ_A, sourceModelId: 'sequence-input', score: null, globalScore: null },
    ]);

    const rUntpl = await runWorker({ model: 'monomer', msaMode: 'single', recycles: 1 }, { sequences: seqSet });
    const rEmptyTpl = await runWorker(
      { model: 'monomer', msaMode: 'single', recycles: 1, templatePdbText: '' },
      { sequences: seqSet },
    );
    expect((rUntpl.outputs?.folds as FoldedStructure).id).toBe(
      (rEmptyTpl.outputs?.folds as FoldedStructure).id,
    );

    const rPortOnly = await runWorker(
      { model: 'monomer', msaMode: 'single', recycles: 1 },
      { sequences: seqSet, template: makeStructureArtifact('tpl-port', pdb) },
    );
    const rBoth = await runWorker(
      { model: 'monomer', msaMode: 'single', recycles: 1, templatePdbText: pdbInlineOther },
      { sequences: seqSet, template: makeStructureArtifact('tpl-port', pdb) },
    );
    expect((rBoth.outputs?.folds as FoldedStructure).id).toBe(
      (rPortOnly.outputs?.folds as FoldedStructure).id,
    );
    expect((rBoth.outputs?.folds as FoldedStructure).folds[0].provenance?.warnings?.length).toBeGreaterThan(0);

    const rMoved = await runWorker(
      { model: 'monomer', msaMode: 'single', recycles: 1, templatePdbText: pdbMoved },
      { sequences: seqSet },
    );
    expect((rMoved.outputs?.folds as FoldedStructure).id).not.toBe(
      (rPortOnly.outputs?.folds as FoldedStructure).id,
    );
    expect((rPortOnly.outputs?.folds as FoldedStructure).id).not.toBe(
      (rUntpl.outputs?.folds as FoldedStructure).id,
    );
  });

  it('Tier 5 adversarial (Gaps 1, 4, 7): worker enforces missingTemplateUpload/SHA-256 pre-fetch, trims modelKey, and keeps af2-monomer runHash invariant to templateSpanChains', async () => {
    const pdb = makeBackbonePdb([{ chainId: 'A', sequence: SEQ_A }]);
    const seqSet = makeSeqSet('t5-worker', [
      { id: 'c1', sequence: SEQ_A, sourceModelId: 'sequence-input', score: null, globalScore: null },
    ]);

    // Gap 1: missingTemplateUpload & mismatched expectedTemplateSha256 fail pre-fetch (0 fetch calls)
    const rMissing = await runWorker(
      {
        model: 'monomer',
        msaMode: 'single',
        missingTemplateUpload: true,
        expectedTemplateFilename: 'omitted.pdb',
        expectedTemplateSha256: 'a'.repeat(64),
      },
      { sequences: seqSet },
    );
    expect(rMissing.status).toBe('error');
    expect(rMissing.error).toMatch(/omitted/i);
    expect(rMissing.fetchUrls).toEqual([]);

    const rBadSha = await runWorker(
      {
        model: 'monomer',
        msaMode: 'single',
        templatePdbText: pdb,
        expectedTemplateSha256: '0'.repeat(64),
      },
      { sequences: seqSet },
    );
    expect(rBadSha.status).toBe('error');
    expect(rBadSha.error).toMatch(/does not match/i);
    expect(rBadSha.fetchUrls).toEqual([]);

    // Live template port overrides stale inline upload guards
    const rPortOverride = await runWorker(
      {
        model: 'monomer',
        msaMode: 'single',
        missingTemplateUpload: true,
        templatePdbText: pdb,
        expectedTemplateSha256: '0'.repeat(64),
      },
      { sequences: seqSet, template: makeStructureArtifact('tpl-override', pdb) },
    );
    expect(rPortOverride.status).toBe('ok');

    // Gap 4: modelKey.trim() parity across monomer and boltz2
    const rCanonicalMono = await runWorker(
      { model: 'monomer', msaMode: 'single', recycles: 1, templatePdbText: pdb },
      { sequences: seqSet },
    );
    const rTrimmedMono = await runWorker(
      { model: ' monomer ', msaMode: 'single', recycles: 1, templatePdbText: pdb },
      { sequences: seqSet },
    );
    expect(rTrimmedMono.status).toBe('ok');
    expect((rTrimmedMono.outputs?.folds as FoldedStructure).id).toBe(
      (rCanonicalMono.outputs?.folds as FoldedStructure).id,
    );

    const rTrimmedBoltz = await runWorker(
      { model: ' boltz2 ', msaMode: 'single', recycles: 1, templatePdbText: pdb },
      { sequences: seqSet },
    );
    expect(rTrimmedBoltz.status).toBe('ok');

    // Gap 7: af2-monomer ignores templateSpanChains (all-ones mask in monomer.js), so runHash & provenance stay invariant
    const pdbComplex = makeBackbonePdb([
      { chainId: 'A', sequence: SEQ_A },
      { chainId: 'B', sequence: SEQ_B },
    ]);
    const dimerSeqSet = makeSeqSet('t5-dimer', [
      {
        id: 'dimer',
        sequence: `${SEQ_A}:${SEQ_B}`,
        chainLengths: [SEQ_A.length, SEQ_B.length],
        chainIds: ['A', 'B'],
        sourceModelId: 'sequence-input',
        score: null,
        globalScore: null,
      },
    ]);
    const rMonoSpanT = await runWorker(
      { model: 'monomer', msaMode: 'single', recycles: 1, templatePdbText: pdbComplex, templateSpanChains: true },
      { sequences: dimerSeqSet },
    );
    const rMonoSpanF = await runWorker(
      { model: 'monomer', msaMode: 'single', recycles: 1, templatePdbText: pdbComplex, templateSpanChains: false },
      { sequences: dimerSeqSet },
    );
    expect(rMonoSpanT.status).toBe('ok');
    expect(rMonoSpanF.status).toBe('ok');
    expect((rMonoSpanT.outputs?.folds as FoldedStructure).id).toBe(
      (rMonoSpanF.outputs?.folds as FoldedStructure).id,
    );
    expect((rMonoSpanF.outputs?.folds as FoldedStructure).folds[0].provenance?.templateSpanChains).toBe(true);
  });
});
