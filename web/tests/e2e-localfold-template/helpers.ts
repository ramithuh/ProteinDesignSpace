import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { vi } from 'vitest';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import type {
  CanonicalArtifact,
  FoldedStructure,
  NodeProgressEvent,
  SequenceSet,
  StructureArtifact,
  WorkflowGraph,
} from '$lib/engine/types';
import { AA1_TO_AA3, STANDARD_MONOMER_PDB } from './fixtures';

const VENDOR_ROOT = resolve(__dirname, '../../static/vendor/localfold');

export type LocalFoldTemplateFamily = 'af2-monomer' | 'af2-multimer' | 'af3' | 'none';

export interface ParsedTemplateResidue {
  resName: string;
  code: string;
  resSeq?: number;
  insCode?: string;
  confidence?: number;
  atoms: Map<string, [number, number, number] | number[]>;
}

export interface ParsedTemplateChain {
  chainId: string;
  sequence: string;
  residues: ParsedTemplateResidue[];
}

export interface ParsedTemplateStructure {
  format: 'pdb' | 'mmcif';
  chains: ParsedTemplateChain[];
  totalResidues?: number;
}

export interface TemplateChainAlignment {
  queryChainIndex: number;
  queryChainId: string;
  templateChainId: string | null;
  templateChainIndex?: number | null;
  identity: number;
  coverage: number;
  alignedCount?: number;
  residueMap: Map<number, number>;
}

export interface TemplateAlignmentResult {
  parsed?: ParsedTemplateStructure;
  chainAlignments: TemplateChainAlignment[];
  matchedQueryResidues: number;
  totalQueryResidues: number;
  overallCoverage?: number;
  overallIdentity?: number;
}

export interface Af2TemplateObject {
  aatype: Int32Array;
  atomPositions: Float32Array;
  atomMask: Float32Array;
  spanChains?: boolean;
  covered?: number;
  atoms?: number;
}

export interface Af3TemplateSlot {
  aatype: Int32Array;
  atomPositions: Float32Array;
  atomMask: Float32Array;
  spanChains?: boolean;
  covered?: number;
  atoms?: number;
}

export interface ResolvedLocalFoldTemplateSource {
  sourceKind: 'port' | 'inline';
  text: string;
  format: 'pdb' | 'mmcif';
  filename?: string;
  digest: string;
  spanChains: boolean;
}

export interface LocalFoldTemplateContract {
  isLocalFoldTemplateCapable: (modelId: string) => boolean;
  localFoldTemplateFamily: (modelId: string) => LocalFoldTemplateFamily;
  resolveLocalFoldTemplateSource?: (opts: {
    params?: Record<string, unknown>;
    templateInput?: StructureArtifact | null;
  }) => ResolvedLocalFoldTemplateSource | null;
  parseTemplateStructure: (
    text: string,
    opts?: {
      altlocFirst?: boolean;
      arginines?: boolean;
    },
  ) => ParsedTemplateStructure;
  alignQueryToTemplate: (
    queryChains: Array<{ chainId?: string; sequence: string }>,
    parsed: ParsedTemplateStructure,
    opts?: { minCoverage?: number; minIdentity?: number },
  ) => TemplateAlignmentResult;
  buildAf2TemplatePayload: (
    queryChains: string[],
    alignment: TemplateAlignmentResult,
    family: 'af2-monomer' | 'af2-multimer',
    opts: { spanChains: boolean },
  ) => {
    template: Af2TemplateObject;
    multichainMask2d: Float32Array;
    asymId: Float32Array;
  };
  buildAf3TemplateSlots: (...args: unknown[]) => Af3TemplateSlot[];
}

const templateModules = import.meta.glob('/src/lib/engine/localfold-template.ts');

export async function loadLocalFoldTemplateModule(): Promise<LocalFoldTemplateContract> {
  const loader = templateModules['/src/lib/engine/localfold-template.ts'];
  if (!loader) {
    throw new Error(
      'M1 module $lib/engine/localfold-template.ts does not exist yet — awaiting M1 implementation',
    );
  }
  return (await loader()) as LocalFoldTemplateContract;
}

export async function importLocalFoldVendor<T = Record<string, any>>(relPath: string): Promise<T> {
  const fullPath = resolve(VENDOR_ROOT, relPath);
  return (await import(pathToFileURL(fullPath).href)) as T;
}

export interface VendorTemplateOracles {
  dialectFor: (model: string) => Record<string, any>;
  templatePolicyFor: (dialect: Record<string, any>) => Record<string, any>;
  featuriserDialect: (dialect: Record<string, any>) => Record<string, any>;
  af3BatchFromA3m: (
    sequence: string,
    alignment: unknown,
    options?: Record<string, unknown>,
  ) => { batch: Record<string, any>; rows: Record<string, any> };
  featuriseProtein: (sequence: string, options?: Record<string, unknown>) => Record<string, any>;
  GAP_AATYPE: number;
  templateSlot: (args: Record<string, unknown>) => Af3TemplateSlot;
  templateSlotAtom37: (args: Record<string, unknown>) => Af2TemplateObject;
  mergeTemplateSlots: (slots: Af3TemplateSlot[]) => Af3TemplateSlot;
  chainResidues: (text: string, chain: string, policy?: Record<string, unknown>) => Record<string, any>;
  chosenAltLocs: (atoms: unknown[], first?: boolean) => unknown[];
  finishResidue: (residue: Record<string, any>, opts?: { arginines?: boolean }) => void;
  AF2_ATOM37: Record<string, unknown>;
  AF2_ATOM37_MONOMER: Record<string, unknown>;
  AF3_DENSE: Record<string, unknown>;
  ATOM37: string[];
  NUM_DENSE: number;
  templateGeometry: (
    template: Record<string, any>,
    multichainMask2d: ArrayLike<number>,
    tokens: number,
    layout?: Record<string, unknown>,
    options?: Record<string, unknown>,
  ) => {
    distogram: Float32Array;
    pseudoBetaMask2d: Float32Array;
    unitVector: Float32Array;
    backboneMask2d: Float32Array;
  };
  packTemplateGeometry: (geometry: Record<string, any>, tokens: number) => Float32Array;
  multichainMaskFor: (
    asymId: ArrayLike<number>,
    tokens: number,
    options?: { coverage?: ArrayLike<number>; spanChains?: boolean },
  ) => Float32Array;
  coverageOf: (template: Record<string, any>, tokens: number, layout?: Record<string, unknown>) => Float32Array;
  boltz2TemplateFeatures: (
    template: Record<string, any>,
    asymMask2d: ArrayLike<number>,
    tokens: number,
  ) => Float32Array;
  rosettafold3TemplateFeatures: (
    template: Record<string, any>,
    asymMask2d: ArrayLike<number>,
    tokens: number,
  ) => Float32Array;
  fusedTemplateFeatures: (
    template: Record<string, any> | null | undefined,
    tokens: number,
    width: number,
    dialect: Record<string, any>,
    multichainMask2d?: ArrayLike<number>,
    useGap?: boolean,
  ) => Float32Array;
  fusedTemplateFeaturesSparse: (
    template: Record<string, any> | null | undefined,
    tokens: number,
    width: number,
    dialect: Record<string, any>,
    multichainMask2d?: ArrayLike<number>,
    useGap?: boolean,
  ) => Uint32Array;
  templateEmbedding: (
    input: Record<string, any>,
    weights: Record<string, any>,
    dialect: Record<string, any>,
  ) => Float32Array;
  fusedTemplateEmbedding: (
    input: Record<string, any>,
    weights: Record<string, any>,
    dialect: Record<string, any>,
  ) => Float32Array;
}

let cachedOracles: VendorTemplateOracles | null = null;

export async function loadVendorTemplateOracles(): Promise<VendorTemplateOracles> {
  if (cachedOracles) return cachedOracles;
  const [dialectMod, batchMod, featMod, tiMod, tfMod, tffMod, cpuMod] = await Promise.all([
    importLocalFoldVendor('shared/af3/dialect.js'),
    importLocalFoldVendor('shared/af3/featurise/batch.js'),
    importLocalFoldVendor('shared/af3/featurise/featurise.js'),
    importLocalFoldVendor('shared/af3/featurise/template-input.js'),
    importLocalFoldVendor('shared/af3/featurise/template-features.js'),
    importLocalFoldVendor('shared/af3/featurise/template-fused-features.js'),
    importLocalFoldVendor('cpu/af3/trunk/template.js'),
  ]);
  cachedOracles = {
    dialectFor: dialectMod.dialectFor,
    templatePolicyFor: dialectMod.templatePolicyFor,
    featuriserDialect: dialectMod.featuriserDialect,
    af3BatchFromA3m: batchMod.af3BatchFromA3m,
    featuriseProtein: featMod.featuriseProtein,
    GAP_AATYPE: tiMod.GAP_AATYPE,
    templateSlot: tiMod.templateSlot,
    templateSlotAtom37: tiMod.templateSlotAtom37,
    mergeTemplateSlots: tiMod.mergeTemplateSlots,
    chainResidues: tiMod.chainResidues,
    chosenAltLocs: tiMod.chosenAltLocs,
    finishResidue: tiMod.finishResidue,
    AF2_ATOM37: tfMod.AF2_ATOM37,
    AF2_ATOM37_MONOMER: tfMod.AF2_ATOM37_MONOMER,
    AF3_DENSE: tfMod.AF3_DENSE,
    ATOM37: tfMod.ATOM37,
    NUM_DENSE: tfMod.NUM_DENSE,
    templateGeometry: tfMod.templateGeometry,
    packTemplateGeometry: tfMod.packTemplateGeometry,
    multichainMaskFor: tfMod.multichainMaskFor,
    coverageOf: tfMod.coverageOf,
    boltz2TemplateFeatures: tfMod.boltz2TemplateFeatures,
    rosettafold3TemplateFeatures: tfMod.rosettafold3TemplateFeatures,
    fusedTemplateFeatures: tffMod.fusedTemplateFeatures,
    fusedTemplateFeaturesSparse: tffMod.fusedTemplateFeaturesSparse,
    templateEmbedding: cpuMod.templateEmbedding,
    fusedTemplateEmbedding: cpuMod.fusedTemplateEmbedding,
  };
  return cachedOracles;
}

function deterministicWeightArray(len: number, seedVal: number, scale = 0.15): Float32Array {
  const out = new Float32Array(len);
  let state = seedVal >>> 0;
  for (let i = 0; i < len; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    out[i] = ((state / 0xffffffff) * 2 - 1) * scale;
  }
  return out;
}

export async function runAf3TemplateCpuOracle(opts: {
  modelKey: string;
  tokens: number;
  asymId: ArrayLike<number>;
  slots: Array<Af3TemplateSlot | undefined>;
  multichainMask2d?: Float32Array;
}): Promise<{ output: Float32Array; featureWidth: number; isFused: boolean }> {
  const oracles = await loadVendorTemplateOracles();
  const dialect = oracles.dialectFor(opts.modelKey);
  const queryChannels = 16;
  const channels = 8;
  const pairs = opts.tokens * opts.tokens;
  const pair = deterministicWeightArray(pairs * queryChannels, 101, 0.5);
  const pairMask = new Float32Array(pairs).fill(1);

  const isFused = dialect.fusedTemplateEmbedder === true;
  if (!isFused) {
    const weights = {
      channels,
      queryChannels,
      blocks: [],
      queryEmbeddingNormScale: new Float32Array(queryChannels).fill(1),
      queryEmbeddingNormOffset: new Float32Array(queryChannels).fill(0),
      templatePairEmbedding0: deterministicWeightArray(39 * channels, 11),
      templatePairEmbedding1: deterministicWeightArray(channels, 12),
      templatePairEmbedding2: deterministicWeightArray(31 * channels, 13),
      templatePairEmbedding3: deterministicWeightArray(31 * channels, 14),
      templatePairEmbedding4: deterministicWeightArray(channels, 15),
      templatePairEmbedding5: deterministicWeightArray(channels, 16),
      templatePairEmbedding6: deterministicWeightArray(channels, 17),
      templatePairEmbedding7: deterministicWeightArray(channels, 18),
      templatePairEmbedding8: deterministicWeightArray(queryChannels * channels, 19),
      outputLayerNormScale: new Float32Array(channels).fill(1),
      outputLayerNormOffset: new Float32Array(channels).fill(0),
      outputLinear: deterministicWeightArray(channels * queryChannels, 20, 0.25),
    };
    const out = oracles.templateEmbedding(
      {
        pair,
        tokens: opts.tokens,
        pairMask,
        templates: Math.max(4, opts.slots.length),
        slots: opts.slots,
        asymId: opts.asymId,
        ...(opts.multichainMask2d ? { multichainMask2d: opts.multichainMask2d } : {}),
      },
      weights,
      dialect,
    );
    return { output: out, featureWidth: 88, isFused: false };
  }

  const featureWidth = dialect.boltz2TemplateFeatures
    ? 109
    : dialect.rosettafold3TemplateFeatures
      ? 66
      : 108;
  const mask2dFor = (slot: Af3TemplateSlot | null): Float32Array =>
    opts.multichainMask2d ??
    oracles.multichainMaskFor(opts.asymId, opts.tokens, {
      coverage: slot ? oracles.coverageOf(slot, opts.tokens) : undefined,
      spanChains: slot?.spanChains === true,
    });

  let templateFeatures: Float32Array;
  if (dialect.templateFeatureMeanOnePass === true) {
    // Match webgpu/af3/trunk/template.js:670-709: RoseTTAFold3 filters present slots
    // by atomMask > 0, passes useGap = false (all-zero empty features), and averages across present slots.
    const present = opts.slots.filter(
      (here): here is Af3TemplateSlot =>
        here !== undefined &&
        here !== null &&
        here.atomMask != null &&
        (typeof here.atomMask.some === 'function'
          ? here.atomMask.some((v) => v > 0)
          : Boolean(Array.prototype.some.call(here.atomMask, (v: number) => v > 0))),
    );
    if (present.length === 0) {
      templateFeatures = oracles.fusedTemplateFeatures(
        undefined,
        opts.tokens,
        featureWidth,
        dialect,
        undefined,
        false,
      );
    } else {
      templateFeatures = oracles.fusedTemplateFeatures(
        present[0],
        opts.tokens,
        featureWidth,
        dialect,
        mask2dFor(present[0]),
        false,
      );
      for (let extra = 1; extra < present.length; extra += 1) {
        const more = oracles.fusedTemplateFeatures(
          present[extra],
          opts.tokens,
          featureWidth,
          dialect,
          mask2dFor(present[extra]),
          false,
        );
        for (let idx = 0; idx < templateFeatures.length; idx += 1) {
          templateFeatures[idx] += more[idx];
        }
      }
      if (present.length > 1) {
        for (let idx = 0; idx < templateFeatures.length; idx += 1) {
          templateFeatures[idx] /= present.length;
        }
      }
    }
  } else {
    const firstSlot = opts.slots[0] ?? null;
    templateFeatures = oracles.fusedTemplateFeatures(
      firstSlot,
      opts.tokens,
      featureWidth,
      dialect,
      mask2dFor(firstSlot),
    );
  }
  const weights = {
    channels,
    queryChannels,
    featureWidth,
    blocks: [],
    queryEmbeddingNormScale: new Float32Array(queryChannels).fill(1),
    queryEmbeddingNormOffset: new Float32Array(queryChannels).fill(0),
    zProjection: deterministicWeightArray(queryChannels * channels, 31),
    aProjection: deterministicWeightArray(featureWidth * channels, 32),
    outputLayerNormScale: new Float32Array(channels).fill(1),
    outputLayerNormOffset: new Float32Array(channels).fill(0),
    outputLinear: deterministicWeightArray(channels * queryChannels, 33, 0.25),
  };
  const out = oracles.fusedTemplateEmbedding(
    {
      pair,
      tokens: opts.tokens,
      pairMask,
      templates: Math.max(4, opts.slots.length),
      slots: opts.slots,
      templateFeatures,
    },
    weights,
    dialect,
  );
  return { output: out, featureWidth, isFused: true };
}

export function diffFloat32Arrays(a: Float32Array, b: Float32Array): {
  l2: number;
  maxAbs: number;
  hasNaN: boolean;
  nonZeroA: number;
  nonZeroB: number;
} {
  let sumSq = 0;
  let maxAbs = 0;
  let hasNaN = false;
  let nonZeroA = 0;
  let nonZeroB = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const va = a[i];
    const vb = b[i];
    if (!Number.isFinite(va) || !Number.isFinite(vb)) hasNaN = true;
    if (va !== 0) nonZeroA++;
    if (vb !== 0) nonZeroB++;
    const d = Math.abs(va - vb);
    if (d > maxAbs) maxAbs = d;
    sumSq += d * d;
  }
  return { l2: Math.sqrt(sumSq), maxAbs, hasNaN, nonZeroA, nonZeroB };
}

export interface WorkerCapturedCall {
  family: 'af2-monomer' | 'af2-multimer' | 'af3' | 'esmfold2';
  mode?: 'predict' | 'predictA3m';
  featuresByRecycle?: Array<Record<string, unknown>>;
  predictOpts?: Record<string, unknown>;
  effectiveMultichainMask2d?: Float32Array;
  a3m?: unknown;
  batch?: Record<string, unknown>;
  af3Options?: Record<string, unknown>;
  ef2Args?: Record<string, unknown>;
}

const SYNTHETIC_ZN_CCD_CIF = `data_ZN
_chem_comp.id ZN
_chem_comp.name "ZINC ION"
_chem_comp_atom.comp_id ZN
_chem_comp_atom.atom_id ZN
_chem_comp_atom.type_symbol ZN
_chem_comp_atom.charge 2
_chem_comp_atom.pdbx_leaving_atom_flag N
_chem_comp_atom.model_Cartn_x 0.0
_chem_comp_atom.model_Cartn_y 0.0
_chem_comp_atom.model_Cartn_z 0.0
#
`;

const SYNTHETIC_ATP_CCD_CIF = `data_ATP
_chem_comp.id ATP
_chem_comp.name "ADENOSINE-5'-TRIPHOSPHATE"
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
ATP PG P 0 N 0.0 0.0 0.0
ATP O1G O 0 N 1.5 0.0 0.0
ATP O2G O -1 N 0.0 1.5 0.0
ATP C4' C 0 N 3.0 1.0 0.5
#
`;

let activeOriginDir: string | null = null;

function safeSymlink(target: string, linkPath: string): void {
  mkdirSync(dirname(linkPath), { recursive: true });
  if (!existsSync(linkPath)) {
    symlinkSync(target, linkPath);
  }
}

export function ensureWorkerVendorOrigin(): string {
  if (activeOriginDir && existsSync(activeOriginDir)) {
    return activeOriginDir;
  }
  const root = mkdtempSync(resolve(tmpdir(), 'localfold-e2e-'));
  const lfRoot = resolve(root, 'vendor/localfold');
  mkdirSync(lfRoot, { recursive: true });
  writeFileSync(resolve(root, 'package.json'), '{"type":"module"}\n', 'utf8');

  // Symlink untouched real vendor directories needed for dialect, batching, MSA, CPU features
  for (const sub of ['cpu']) {
    safeSymlink(resolve(VENDOR_ROOT, sub), resolve(lfRoot, sub));
  }
  for (const sub of ['af2', 'bioemu2', 'chem', 'esmfold2', 'heads', 'input', 'weights']) {
    safeSymlink(resolve(VENDOR_ROOT, 'shared', sub), resolve(lfRoot, 'shared', sub));
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
    safeSymlink(resolve(VENDOR_ROOT, 'shared/af3', item), resolve(lfRoot, 'shared/af3', item));
  }

  // Thin leaf stubs for WebGPU / weight-fetch modules only
  const writeStub = (relPath: string, code: string) => {
    const p = resolve(lfRoot, relPath);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, code, 'utf8');
  };

  writeStub(
    'webgpu/runtime/device.js',
    `export async function requestAlphaFoldDevice(adapter) {
      return {
        adapter,
        limits: { maxStorageBufferBindingSize: 1024 * 1024 * 1024 },
        lost: new Promise(() => {}),
        destroy() {},
      };
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
      'monomer-4': { delta: { base: 'monomer' } },
      'monomer-5': { delta: { base: 'monomer' } },
      'multimer-2': { delta: { base: 'multimer' } },
      'multimer-3': { delta: { base: 'multimer' } },
      'multimer-4': { delta: { base: 'multimer' } },
      'multimer-5': { delta: { base: 'multimer' } },
    };
    export function graphFamily(k) { return String(k).startsWith('multimer') ? 'multimer' : 'monomer'; }
    export function bundleBaseUrl(k) { return 'https://stub.local/bundles/' + k; }
    export async function loadManifest(k) { return { model: k, tensors: {} }; }\n`,
  );

  writeStub(
    'shared/bundles/http-tensor-store.js',
    `export class HttpTensorStore {
      static async fromManifest() { return new HttpTensorStore(); }
      prefetch() {}
      releaseDecoded() {}
    }\n`,
  );

  writeStub(
    'shared/bundles/delta-tensor-store.js',
    `export class DeltaTensorStore {
      constructor(base, delta) { this.base = base; this.delta = delta; }
      prefetch() {}
      releaseDecoded() {}
    }\n`,
  );

  writeStub(
    'shared/bundles/alphafold-fixture.js',
    `export class AlphaFoldFixture {
      static fromStore() { return new AlphaFoldFixture(); }
      async embeddingWeights() { return {}; }
      async templateWeights() { return { stub: 'monomer-template' }; }
      async templateEmbeddingWeights() { return { stub: 'multimer-template' }; }
      async extraStackWeights() { return [{}]; }
      async mainStackWeights() { return [{}]; }
      async structureWeights() { return {}; }
      async confidenceWeights() { return { lddt: {}, pae: {} }; }
      async geometryTables() { return {}; }
      async queryOnlyFeatureTables() {
        return {
          atom37ToAtom14: new Float32Array(21 * 37),
          atom37Mask: new Float32Array(21 * 37),
        };
      }
      async tensor() { return new Float32Array([0]); }
    }\n`,
  );

  writeStub(
    'webgpu/input/a3m-features.js',
    `import { makeA3mFeatures } from '../../shared/input/a3m-features.js';
    export async function makeA3mFeaturesFor(_device, a3mText, tables, options = {}) {
      return makeA3mFeatures(a3mText, tables, options);
    }\n`,
  );

  writeStub(
    'shared/af3/weights/weights.js',
    `export function releaseWeights() {}\n`,
  );

  writeStub(
    'shared/af3/weights/diffusion-weights.js',
    `import { dialectFor } from '../dialect.js';
    export async function foldWeights(store) {
      const modelKey = globalThis.__LF_WORKER_CURRENT_MODEL__ || 'af3';
      return { trunk: { dialect: dialectFor(modelKey) }, store };
    }\n`,
  );

  writeStub(
    'webgpu/af2/model/monomer.js',
    `function synthAf2Out(a3mText, predictOpts) {
      const firstLine = String(a3mText || '').split(/\\r?\\n/).find((l) => l && !l.startsWith('>') && !l.startsWith('#')) || 'MKWVTFISLL';
      const L = firstLine.replace(/[\\s/-]/g, '').length || 10;
      const atom37 = new Float32Array(L * 37 * 3);
      const atom37Mask = new Float32Array(L * 37);
      for (let i = 0; i < L; i++) {
        for (const s of [0, 1, 2, 4]) {
          atom37Mask[i * 37 + s] = 1;
          atom37[(i * 37 + s) * 3] = i * 3.8 + s * 0.6;
          atom37[(i * 37 + s) * 3 + 1] = s === 2 ? 1.2 : 0;
          atom37[(i * 37 + s) * 3 + 2] = s === 0 ? 0.5 : 0;
        }
      }
      const plddt = new Float32Array(L).fill(predictOpts?.template ? 91.5 : 82.0);
      return {
        final: {
          structure: { atom37, atom37Mask },
          confidence: { plddt, ptm: predictOpts?.template ? 0.89 : 0.78, predictedAlignedError: new Float32Array(L * L).fill(2.5) },
        },
        recycles: [{}, {}],
      };
    }
    export class AlphaFoldMonomerGpu {
      async predictA3m(a3m, weights, featureTables, predictOpts) {
        globalThis.__LF_WORKER_CAPTURES__?.push({ family: 'af2-monomer', a3m, predictOpts });
        return synthAf2Out(a3m, predictOpts);
      }
    }\n`,
  );

  writeStub(
    'webgpu/af2/multimer/model.js',
    `import { makeA3mFeatures } from '../../../shared/input/a3m-features.js';
    import { multichainMaskFor, coverageOf, AF2_ATOM37 } from '../../../shared/af3/featurise/template-features.js';
    function synthAf2MultiOut(featuresByRecycle, predictOpts) {
      const L = featuresByRecycle?.[0]?.aatype?.length
        || (Array.isArray(predictOpts?.chainLengths) ? predictOpts.chainLengths.reduce((a, b) => a + b, 0) : 10)
        || 10;
      const atom37 = new Float32Array(L * 37 * 3);
      const atom37Mask = new Float32Array(L * 37);
      for (let i = 0; i < L; i++) {
        for (const s of [0, 1, 2, 4]) {
          atom37Mask[i * 37 + s] = 1;
          atom37[(i * 37 + s) * 3] = i * 3.8 + s * 0.6;
          atom37[(i * 37 + s) * 3 + 1] = s === 2 ? 1.2 : 0;
        }
      }
      const plddt = new Float32Array(L).fill(predictOpts?.template ? 92.0 : 81.0);
      return {
        final: {
          structure: { atom37, atom37Mask },
          confidence: {
            plddt,
            ptm: predictOpts?.template ? 0.90 : 0.77,
            iptm: predictOpts?.template ? 0.87 : 0.71,
            predictedAlignedError: new Float32Array(L * L).fill(2.2),
          },
        },
        recycles: [{}, {}],
      };
    }
    export class AlphaFoldUnifiedGpu {
      #fromPredictA3m = false;
      #a3m = undefined;
      async predictA3m(a3m, weights, featureTables, options = {}, paeBreaks, onRecycle, onProgress) {
        const featuresByRecycle = makeA3mFeatures(a3m, featureTables, {
          ...options,
          multimerClustering: options.multimerClustering ?? options.outerProductMeanFirst === true,
        });
        this.#fromPredictA3m = true;
        this.#a3m = a3m;
        try {
          return await this.predict(featuresByRecycle, weights, paeBreaks, onRecycle, onProgress, options);
        } finally {
          this.#fromPredictA3m = false;
          this.#a3m = undefined;
        }
      }
      async predict(featuresByRecycle, weights, paeBreaks, onRecycle, onProgress, recycleOptions = {}) {
        const mode = this.#fromPredictA3m ? 'predictA3m' : 'predict';
        const a3m = this.#a3m;
        const options = recycleOptions ?? {};
        const asymId = featuresByRecycle?.[0]?.asymId;
        if (options?.template !== undefined && !asymId) {
          throw new Error(
            "a template needs asymId (or multichainMask2d): AF2-multimer masks the template's geometry across chains, and assuming one chain lets it speak about pairs it has never seen in one coordinate frame",
          );
        }
        let effectiveMultichainMask2d;
        if (options?.template !== undefined && asymId) {
          const length = featuresByRecycle?.[0]?.aatype?.length ?? asymId.length;
          effectiveMultichainMask2d = multichainMaskFor(asymId, length, {
            coverage: coverageOf(options.template, length, AF2_ATOM37),
            spanChains: options.template.spanChains === true,
          });
        }
        globalThis.__LF_WORKER_CAPTURES__?.push({
          family: 'af2-multimer',
          mode,
          a3m,
          featuresByRecycle,
          predictOpts: options,
          ...(effectiveMultichainMask2d ? { effectiveMultichainMask2d } : {}),
        });
        return synthAf2MultiOut(featuresByRecycle, options);
      }
    }\n`,
  );

  writeStub(
    'webgpu/af3/fold.js',
    `export async function foldBatch(device, batch, weights, options) {
      globalThis.__LF_WORKER_CAPTURES__?.push({ family: 'af3', batch, af3Options: options });
      const asym = batch?.asymId || [];
      const resOf = batch?.residueOfToken || [];
      const resCount = resOf.length ? (Math.max(...Array.from(resOf)) + 1) : (batch?.tokens || 10);
      const lines = [];
      let ser = 1;
      for (let r = 0; r < resCount; r++) {
        const ch = 'A';
        const x = (r * 3.8).toFixed(3).padStart(8, ' ');
        for (const [nm, dx, dy] of [['N', -1.2, -0.7], ['CA', 0, 0], ['C', 1.3, -0.7], ['O', 1.4, -1.9]]) {
          const ax = (r * 3.8 + dx).toFixed(3).padStart(8, ' ');
          const ay = dy.toFixed(3).padStart(8, ' ');
          lines.push('ATOM  ' + String(ser++).padStart(5, ' ') + '  ' + nm.padEnd(3, ' ') + ' ALA ' + ch + String(r + 1).padStart(4, ' ') + '    ' + ax + ay + '   0.000  1.00 88.00           ' + nm[0]);
        }
      }
      lines.push('END');
      const hasTpl = Array.isArray(options?.templateSlots) && options.templateSlots.some(Boolean);
      const perResiduePlddt = Array.from({ length: resCount }, () => hasTpl ? 90.0 : 81.0);
      const paeDim = asym.length || resCount;
      return {
        pdb: lines.join('\\n') + '\\n',
        perResiduePlddt,
        meanPlddt: hasTpl ? 90.0 : 81.0,
        ptm: hasTpl ? 0.88 : 0.76,
        iptm: 0.75,
        scores: { pae: new Float32Array(paeDim * paeDim).fill(2.5) },
        reusable: { recycles: options?.recycles ?? 1 },
      };
    }\n`,
  );

  writeStub(
    'web/esmfold2-model.js',
    `export async function loadEsmfold2Weights() {
      return { shape: { loops: 2, pairChannels: 256 }, language: { megabytes: 130 }, weights: {}, confidenceWeights: {} };
    }
    export function languageModelRunner() { return () => {}; }
    export function esmfold2StepsFor(steps) { return steps; }\n`,
  );

  writeStub(
    'webgpu/esmfold2/fold.js',
    `import { featuriseForEsmfold2 } from '../../shared/esmfold2/featurise.js';
    export async function foldEsmfold2(_dev, args) {
      globalThis.__LF_WORKER_CAPTURES__?.push({ family: 'esmfold2', ef2Args: args });
      const f = featuriseForEsmfold2(args.sequence);
      return {
        features: f,
        coordinates: f.refPos,
        steps: 15,
        confidence: {
          plddt: new Float32Array(f.tokens).fill(0.8),
          plddtPerAtom: new Float32Array(f.atoms).fill(0.8),
          pae: new Float32Array(f.tokens ** 2).fill(3.0),
          ptm: 0.7,
        },
      };
    }\n`,
  );

  activeOriginDir = root;
  return root;
}

export function cleanupWorkerVendorOrigin(): void {
  if (activeOriginDir && existsSync(activeOriginDir)) {
    rmSync(activeOriginDir, { recursive: true, force: true });
    activeOriginDir = null;
  }
}

let workerRunQueue: Promise<void> = Promise.resolve();

export async function runLocalFoldWorkerInNode(opts: {
  params: Record<string, unknown>;
  inputs: Record<string, unknown>;
  maxFoldResidues?: number;
}): Promise<{
  status: 'result' | 'error';
  outputs?: { folds: FoldedStructure; structure: StructureArtifact };
  errorMessage?: string;
  progressEvents: Array<{ phase?: string; progress?: number; detail?: string }>;
  captures: WorkerCapturedCall[];
  fetchUrls: string[];
}> {
  const prev = workerRunQueue;
  let release!: () => void;
  workerRunQueue = new Promise<void>((res) => {
    release = res;
  });
  await prev;

  const originDir = ensureWorkerVendorOrigin();
  const originUrl = `/@fs${originDir}`;
  const captures: WorkerCapturedCall[] = [];
  const fetchUrls: string[] = [];
  const progressEvents: Array<{ phase?: string; progress?: number; detail?: string }> = [];
  const posted: Array<Record<string, unknown>> = [];

  const scope = {
    location: { origin: originUrl },
    postMessage: (msg: Record<string, unknown>) => {
      posted.push(msg);
      if (msg.type === 'progress') {
        progressEvents.push({
          phase: msg.phase as string | undefined,
          progress: msg.progress as number | undefined,
          detail: msg.detail as string | undefined,
        });
      }
    },
    onmessage: async (_ev: { data: unknown }) => {},
  };

  try {
    (globalThis as Record<string, unknown>).__LF_WORKER_CAPTURES__ = captures;
    (globalThis as Record<string, unknown>).__LF_WORKER_CURRENT_MODEL__ = String(
      opts.params.model ?? 'monomer',
    );

    vi.stubGlobal('self', scope);
    vi.stubGlobal('navigator', {
      gpu: {
        requestAdapter: async () => ({ name: 'MockWebGPUAdapter' }),
      },
    });
    vi.stubGlobal('fetch', async (url: string | URL) => {
      const u = String(url);
      fetchUrls.push(u);
      if (u.endsWith('/ZN.cif')) {
        return new Response(SYNTHETIC_ZN_CCD_CIF, { status: 200 });
      }
      if (u.endsWith('/ATP.cif')) {
        return new Response(SYNTHETIC_ATP_CCD_CIF, { status: 200 });
      }
      return new Response('', { status: 404 });
    });

    vi.resetModules();
    await import('$lib/plugins/localfold-af2/localfold.worker');
    await scope.onmessage({
      data: {
        id: 1,
        type: 'run',
        params: opts.params,
        inputs: opts.inputs,
        ...(opts.maxFoldResidues !== undefined ? { maxFoldResidues: opts.maxFoldResidues } : {}),
      },
    });
    const errorMsg = posted.find((m) => m.type === 'error');
    if (errorMsg) {
      return {
        status: 'error',
        errorMessage: String(errorMsg.message ?? 'Unknown worker error'),
        progressEvents,
        captures,
        fetchUrls,
      };
    }
    const resMsg = posted.find((m) => m.type === 'result');
    return {
      status: 'result',
      outputs: resMsg?.outputs as { folds: FoldedStructure; structure: StructureArtifact } | undefined,
      progressEvents,
      captures,
      fetchUrls,
    };
  } finally {
    vi.unstubAllGlobals();
    delete (globalThis as Record<string, unknown>).__LF_WORKER_CAPTURES__;
    delete (globalThis as Record<string, unknown>).__LF_WORKER_CURRENT_MODEL__;
    release();
  }
}

export interface SchedulerMockCall {
  nodeParams: Record<string, unknown>;
  inputs: Record<string, CanonicalArtifact>;
}

export async function runSchedulerWithCapturedLocalFold(
  graph: WorkflowGraph,
  opts: { bypassCache?: boolean } = {},
): Promise<{
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>;
  workerCalls: SchedulerMockCall[];
  progress: NodeProgressEvent[];
}> {
  const workerCalls: SchedulerMockCall[] = [];

  class CapturingMockWorker {
    private listeners = new Map<string, Set<(ev: { data: unknown }) => void>>();
    addEventListener(type: string, fn: (ev: { data: unknown }) => void) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type)!.add(fn);
    }
    removeEventListener(type: string, fn: (ev: { data: unknown }) => void) {
      this.listeners.get(type)?.delete(fn);
    }
    private emit(msg: unknown) {
      for (const fn of this.listeners.get('message') ?? []) {
        fn({ data: msg });
      }
    }
    postMessage(msg: {
      id: number;
      type: string;
      params?: Record<string, unknown>;
      inputs?: Record<string, CanonicalArtifact>;
    }) {
      if (msg.type === 'dispose') {
        this.emit({ id: msg.id, type: 'disposed' });
        return;
      }
      if (msg.type !== 'run') return;
      const params = msg.params ?? {};
      const inputs = msg.inputs ?? {};
      workerCalls.push({ nodeParams: params, inputs });

      // Detect if inputs contain a SequenceSet (localfold-af2) vs StructureArtifact (mpnn)
      const seqSet = inputs.sequences as SequenceSet | undefined;
      if (seqSet?.candidates?.length) {
        const cand = seqSet.candidates[0];
        const parts = cand.sequence.split(/[:/]+/).map((s) => s.replace(/\s+/g, '')).filter(Boolean);
        const chainLengths =
          cand.chainLengths && cand.chainLengths.length > 0
            ? cand.chainLengths
            : parts.length > 1
              ? parts.map((p) => p.length)
              : [parts.join('').length];
        const cleanSeq = parts.join('');
        const L = cleanSeq.length;
        const chainIds: string[] = [];
        const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
        chainLengths.forEach((len, chIdx) => {
          const cid = alphabet[chIdx % alphabet.length];
          for (let k = 0; k < len; k++) chainIds.push(cid);
        });
        const modelKey = String(params.model ?? 'monomer');
        const tplInput = inputs.template as StructureArtifact | undefined;
        const hasTpl = Boolean(tplInput) || Boolean(String(params.templatePdbText ?? '').trim());
        const hashSuffix = tplInput ? `tpl-${tplInput.id}` : hasTpl ? 'tpl-inline' : 'notpl';
        const caCoords = new Float32Array(L * 3);
        const backbone4Coords = new Float32Array(L * 4 * 3);
        for (let i = 0; i < L; i++) {
          caCoords[i * 3] = i * 3.8;
          backbone4Coords[i * 12 + 3] = i * 3.8;
          backbone4Coords[i * 12] = i * 3.8 - 1.2;
          backbone4Coords[i * 12 + 6] = i * 3.8 + 1.3;
          backbone4Coords[i * 12 + 9] = i * 3.8 + 1.4;
        }
        const pdbRows: string[] = [];
        let ser = 1;
        let resOffset = 0;
        for (let chIdx = 0; chIdx < chainLengths.length; chIdx++) {
          const chId = alphabet[chIdx % alphabet.length];
          for (let r = 0; r < chainLengths[chIdx]; r++) {
            const idx = resOffset + r;
            const x = (idx * 3.8).toFixed(3).padStart(8, ' ');
            const xm = (idx * 3.8 - 1.2).toFixed(3).padStart(8, ' ');
            const xp = (idx * 3.8 + 1.3).toFixed(3).padStart(8, ' ');
            const rSeq = String(r + 1).padStart(4, ' ');
            const res3 = AA1_TO_AA3[cleanSeq[idx]] ?? 'ALA';
            pdbRows.push(
              `ATOM  ${String(ser++).padStart(5, ' ')}  N   ${res3} ${chId}${rSeq}    ${xm}  -0.700   0.000  1.00 88.00           N`,
              `ATOM  ${String(ser++).padStart(5, ' ')}  CA  ${res3} ${chId}${rSeq}    ${x}   0.000   0.000  1.00 88.00           C`,
              `ATOM  ${String(ser++).padStart(5, ' ')}  C   ${res3} ${chId}${rSeq}    ${xp}  -0.700   0.000  1.00 88.00           C`,
              `ATOM  ${String(ser++).padStart(5, ' ')}  O   ${res3} ${chId}${rSeq}    ${xp}  -1.900   0.000  1.00 88.00           O`,
            );
          }
          resOffset += chainLengths[chIdx];
        }
        pdbRows.push('END');
        const pdbLines = pdbRows.join('\n') + '\n';
        const foldStruct: FoldedStructure = {
          kind: 'FoldedStructure',
          id: `folds-${modelKey}-${seqSet.id}-${hashSuffix}-${workerCalls.length}`,
          parentStructureId: seqSet.parentStructureId,
          sourceStructureId: seqSet.sourceStructureId,
          folds: [
            {
              sequenceId: cand.id,
              sequence: cleanSeq,
              foldModelId: `localfold-af2-${modelKey}-int5`,
              pdbText: pdbLines,
              caCoords,
              plddt: new Float32Array(L).fill(hasTpl ? 91 : 82),
              meanPlddt: hasTpl ? 91 : 82,
              ptm: hasTpl ? 0.89 : 0.77,
              chainLengths,
              chainIds,
              elapsedMs: 12,
            },
          ],
        };
        const outStruct: StructureArtifact = {
          kind: 'StructureArtifact',
          id: `struct-${modelKey}-${hashSuffix}-${workerCalls.length}`,
          label: `${modelKey} fold`,
          sourceModelId: `localfold-af2-${modelKey}-int5`,
          length: L,
          sequence: cleanSeq,
          caCoords,
          backbone4Coords,
          pdbText: pdbLines,
          geometry: computeGeometryStats(caCoords),
        };
        this.emit({ id: msg.id, type: 'result', outputs: { folds: foldStruct, structure: outStruct }, elapsedMs: 12 });
        return;
      }

      // Fallback for MPNN / other GPU nodes in multi-stage workflows
      const inStruct = inputs.structure as StructureArtifact | undefined;
      if (inStruct) {
        const designedSeq: SequenceSet = {
          kind: 'SequenceSet',
          id: `mpnn-seq-${inStruct.id}-${workerCalls.length}`,
          parentStructureId: inStruct.id,
          sourceStructureId: inStruct.id,
          sourceModelId: 'web-protein-mpnn',
          candidates: [
            {
              id: 'mpnn-cand-1',
              sequence: inStruct.sequence || 'MKWVTFISLLFLFSSAYSRG',
              score: -1.42,
              globalScore: -1.42,
              sourceModelId: 'web-protein-mpnn',
              sourceStructureId: inStruct.id,
            },
          ],
        };
        this.emit({ id: msg.id, type: 'result', outputs: { sequences: designedSeq }, elapsedMs: 8 });
      }
    }
    terminate() {}
  }

  vi.stubGlobal('Worker', CapturingMockWorker);
  const scheduler = new ExecutionScheduler();
  const progress: NodeProgressEvent[] = [];
  try {
    const res = await scheduler.run(graph, (ev) => progress.push(ev), opts);
    return { outputsByNode: res.outputsByNode, workerCalls, progress };
  } finally {
    await scheduler.releaseActiveGpu();
    vi.unstubAllGlobals();
  }
}

export function makeDefaultLocalFoldParams(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    ...defaultParamsForAdapter('localfold-af2'),
    ...overrides,
  };
}

export function hasLiveWebGpu(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    typeof navigator === 'object' &&
    'gpu' in navigator &&
    Boolean((navigator as Navigator & { gpu?: unknown }).gpu)
  );
}
