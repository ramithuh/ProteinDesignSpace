// cg2all (plugin #29): Cα graph/features, e3nn SH + Clebsch–Gordan equivariance, rigid-group all-atom assembly,
// WebGPU engine lifecycle (device.lost fail-fast, bounded pipelines/pacing), plugin registration & DSL round-trip,
// and pinned goldens from the PyTorch reference (huhlim/cg2all a00b881, CalphaBasedModel / -FIX, FP32).
// Full-network parity against the real checkpoints runs when exported bundles are present
// (CG2ALL_WEIGHTS_DIR, default /tmp/cg2all-weights; exporter and golden tooling in tools/cg2all/).

import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { ADAPTER_TO_SHORT, SHORT_TO_ADAPTER, decodeFlowQuery, encodeGraphToFlow, parseFlowDsl } from '$lib/engine/flow-dsl';
import { CATALOG_ITEMS } from '$lib/engine/catalog';
import { defaultParamsForAdapter, validateGraph } from '$lib/engine/graph-utils';
import { getModelCardForAdapter } from '$lib/engine/model-cards';
import { generateLlmsTxt } from '$lib/engine/llms-txt';
import { isWebGpuDeviceLostError } from '$lib/engine/webgpu-device';
import type { SequenceSet, StructureArtifact, WorkflowGraph } from '$lib/engine/types';
import { createDefaultPluginRegistry } from '$lib/plugins';
import { cg2allPlugin } from '$lib/plugins/cg2all';
import {
  buildCg2allArtifact,
  cg2allOptionsFromParams,
  planCg2allJobs,
  structureItems,
  wrapCg2allOutputs,
} from '$lib/plugins/cg2all/jobs';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { formatNodeParamSummary } from '$lib/ui/node-display-helpers';

/* eslint-disable @typescript-eslint/no-explicit-any */
const VENDOR = resolve(process.cwd(), 'static/vendor/cg2all');
const load = async (rel: string): Promise<any> => import(/* @vite-ignore */ pathToFileURL(join(VENDOR, rel)).href);
const shared = await load('shared/index.js');
const features = await load('shared/features.js');
const model = await load('shared/model.js');
const constants = await load('shared/constants.js');
const gpuEngineMod = await load('webgpu/engine.js');
const shaders = await load('webgpu/shaders.js');

interface FixtureVariant {
  s: number[];
  v: number[];
  ss: number[];
  R: number[];
}
interface FixtureCase {
  pdb: string;
  n: number;
  e: number;
  src?: number[];
  dst?: number[];
  edgeClass?: string;
  f0?: number[];
  f1?: number[];
  residueType: number[];
  mask: string;
  ca: FixtureVariant;
  'ca-fix': FixtureVariant;
}
const FIXTURE = JSON.parse(readFileSync(resolve(process.cwd(), 'src/lib/__tests__/fixtures/cg2all-parity.json'), 'utf8')) as {
  cases: Record<'helix16' | 'crambin46' | 'twochain', FixtureCase>;
  tensorShapes: [string, number[]][];
  shProbe: { vec: [number, number, number]; sh: number[] };
};

const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

/** All-atom RMSD in Å over cg2all's output-atom mask (no superposition: same frame). */
function maskedRmsd(R: Float64Array, ref: number[], c: FixtureCase): number {
  let ss = 0;
  let n = 0;
  for (let i = 0; i < c.residueType.length; i++) {
    if (constants.AMINO_ACIDS[c.residueType[i]] === 'UNK') continue;
    for (let a = 0; a < 24; a++) {
      if (c.mask[i * 24 + a] !== '1') continue;
      for (let k = 0; k < 3; k++) ss += (10 * (R[(i * 24 + a) * 3 + k] - ref[(i * 24 + a) * 3 + k])) ** 2;
      n++;
    }
  }
  return Math.sqrt(ss / n);
}

/** Deterministic random weights with the checkpoint's exact tensor names and shapes. */
function syntheticTensors(seed = 7): Map<string, { shape: number[]; data: Float32Array }> {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
  const out = new Map<string, { shape: number[]; data: Float32Array }>();
  for (const [name, shape] of FIXTURE.tensorShapes) {
    const count = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(count);
    const isNormAffine = /(layer_norms\.\d|group_norm|net\.[14])\.weight$/.test(name);
    const fanIn = shape.length === 2 ? shape[1] : 1;
    for (let i = 0; i < count; i++) {
      data[i] = isNormAffine ? 1 + 0.2 * rand() : shape.length === 2 ? (2 * rand() * Math.sqrt(3)) / Math.sqrt(fanIn) : 0.2 * rand();
    }
    out.set(name, { shape, data });
  }
  return out;
}

function rotation(ax: number, ay: number, az: number): number[] {
  const [ca, sa, cb, sb, cc, sc] = [Math.cos(ax), Math.sin(ax), Math.cos(ay), Math.sin(ay), Math.cos(az), Math.sin(az)];
  const X = [1, 0, 0, 0, ca, -sa, 0, sa, ca];
  const Y = [cb, 0, sb, 0, 1, 0, -sb, 0, cb];
  const Z = [cc, -sc, 0, sc, cc, 0, 0, 0, 1];
  const mul = (A: number[], B: number[]) =>
    Array.from({ length: 9 }, (_, k) => A[3 * Math.floor(k / 3)] * B[k % 3] + A[3 * Math.floor(k / 3) + 1] * B[3 + (k % 3)] + A[3 * Math.floor(k / 3) + 2] * B[6 + (k % 3)]);
  return mul(Z, mul(Y, X));
}
const rot = (Q: number[], v: ArrayLike<number>, o = 0) => [
  Q[0] * v[o] + Q[1] * v[o + 1] + Q[2] * v[o + 2],
  Q[3] * v[o] + Q[4] * v[o + 1] + Q[5] * v[o + 2],
  Q[6] * v[o] + Q[7] * v[o + 1] + Q[8] * v[o + 2],
];

describe('cg2all graph & feature construction (pinned PyTorch goldens)', () => {
  for (const name of ['helix16', 'crambin46'] as const) {
    it(`reproduces the radius graph, edge classes and 17+4 node features for ${name}`, () => {
      const c = FIXTURE.cases[name];
      const p = shared.prepareInput(c.pdb);
      expect(p.n).toBe(c.n);
      expect(p.e).toBe(c.e);
      expect(Array.from(p.src)).toEqual(c.src);
      expect(Array.from(p.dst)).toEqual(c.dst);
      expect(Array.from(p.residueType)).toEqual(c.residueType);
      const cls = Array.from({ length: p.e }, (_, k) => {
        const [b, s] = [p.edgeFeat[4 * k], p.edgeFeat[4 * k + 1]];
        return b && s ? 'x' : b ? 'b' : s ? 's' : '.';
      }).join('');
      expect(cls).toBe(c.edgeClass);
      expect(maxAbsDiff(p.f0, c.f0!)).toBeLessThan(5e-5);
      expect(maxAbsDiff(p.f1, c.f1!)).toBeLessThan(5e-6);
      for (let k = 0; k < p.e; k++) {
        const r = Math.hypot(p.relPos[3 * k], p.relPos[3 * k + 1], p.relPos[3 * k + 2]);
        expect(p.edgeFeat[4 * k + 3]).toBeCloseTo(r, 12);
        expect(r).toBeLessThanOrEqual(features.GRAPH_RADIUS);
      }
    });
  }

  it('detects SSBOND disulfides (crambin 3–40, 4–32, 16–26) and drops their HG1 from the output', () => {
    const p = shared.prepareInput(FIXTURE.cases.crambin46.pdb);
    expect(p.trace.ssbonds).toEqual([[2, 39], [3, 31], [15, 25]]);
    const ss = Array.from({ length: p.e }, (_, k) => p.edgeFeat[4 * k + 1]).reduce((a: number, b: number) => a + b, 0);
    expect(ss).toBe(6); // three disulfides, both directions, all within 10 Å
    const res = shared.assemble(p, { s: FIXTURE.cases.crambin46.ca.s, v: FIXTURE.cases.crambin46.ca.v }, { hydrogens: true });
    const cysLines = res.pdbText.split('\n').filter((l: string) => l.startsWith('ATOM') && l.slice(17, 20) === 'CYS');
    expect(cysLines.some((l: string) => l.slice(12, 16).trim() === 'HG')).toBe(false);
    expect(res.pdbText.split('\n').filter((l: string) => l.startsWith('SSBOND'))).toHaveLength(3);
  });

  it('marks chain ends and >10 Å gaps as discontinuous and keeps multi-chain topology', () => {
    const p = shared.prepareInput(FIXTURE.cases.twochain.pdb);
    expect(p.n).toBe(42);
    expect(p.e).toBe(FIXTURE.cases.twochain.e);
    expect(new Set(p.chainIndex).size).toBe(2);
    expect(shared.countChainBreaks(p)).toBe(1); // residues 33–36 removed: 14.6 Å gap inside chain B
    expect(p.trace.ligandLines.some((l: string) => l.includes(' ZN '))).toBe(true);
    const contPrev = Array.from(p.continuity.prev);
    expect(contPrev[0]).toBe(0);
    expect(contPrev[22]).toBe(0); // first residue of chain B
    expect(contPrev.filter((v) => v === 0)).toHaveLength(3);
  });

  it('threads a sequence onto the trace with PULCHRA semantics', () => {
    const trace = shared.parseCaTrace(FIXTURE.cases.helix16.pdb);
    shared.threadSequence(trace, 'AAAAAAAAGGGGWWWW');
    expect(trace.residues.map((r: any) => r.resName).join(',')).toBe(
      'ALA,ALA,ALA,ALA,ALA,ALA,ALA,ALA,GLY,GLY,GLY,GLY,TRP,TRP,TRP,TRP',
    );
    expect(() => shared.threadSequence(shared.parseCaTrace(FIXTURE.cases.helix16.pdb), 'AAA')).toThrow(/3 residues but the trace has 16/);
    expect(() => shared.splitSequence('AB/C', [2, 1])).toThrow(/unsupported residue code 'B'/);
    expect(shared.splitSequence('AC:D', [2, 1])).toEqual(['AC', 'D']);
  });
});

describe('cg2all spherical harmonics & Clebsch–Gordan basis', () => {
  it('matches e3nn 0.5.1 real spherical harmonics (integral normalization) to 1e-12', () => {
    const { vec, sh } = FIXTURE.shProbe;
    const out = features.sphericalHarmonics(vec[0], vec[1], vec[2]);
    expect(maxAbsDiff(out, sh)).toBeLessThan(1e-11);
  });

  it('satisfies the addition theorem Σ_m Y_lm² = (2l+1)/4π for every degree', () => {
    for (const v of [[1, 0, 0], [0.2, -0.7, 0.4], [-3, 2, 5]]) {
      const sh = features.sphericalHarmonics(v[0], v[1], v[2]);
      for (let l = 0; l <= 4; l++) {
        let s = 0;
        for (let m = 0; m < 2 * l + 1; m++) s += sh[l * l + m] ** 2;
        expect(s).toBeCloseTo((2 * l + 1) / (4 * Math.PI), 12);
      }
    }
  });

  it('builds TFN basis blocks of the right sizes and rotates them consistently (B(Qr) contracted with Q-rotated vectors)', () => {
    const rel = new Float64Array([0.31, -0.42, 0.55, -0.2, 0.1, 0.7]);
    const basis = features.computeBasis(rel);
    expect(basis.pairs.map((p: any) => `${p.dIn},${p.dOut}:${p.size}`)).toEqual([
      '0,1:3', '0,2:5', '1,0:3', '1,1:27', '1,2:45', '2,0:5', '2,1:45', '2,2:125',
    ]);
    // (1 -> 0) block is a scaled projection onto r̂: scalar output of x·B is rotation invariant
    const Q = rotation(0.4, -1.1, 2.0);
    const relQ = new Float64Array([...rot(Q, rel, 0), ...rot(Q, rel, 3)]);
    const basisQ = features.computeBasis(relQ);
    const p10 = basis.pairs.find((p: any) => p.dIn === 1 && p.dOut === 0);
    const x = [0.3, 0.9, -0.5];
    for (let e = 0; e < 2; e++) {
      const b = basis.data.subarray(p10.offset + e * 3, p10.offset + e * 3 + 3);
      const bq = basisQ.data.subarray(p10.offset + e * 3, p10.offset + e * 3 + 3);
      const xq = rot(Q, x);
      // e3nn l=1 components are (x, y, z): Cartesian rotation applies directly
      expect(b[0] * x[0] + b[1] * x[1] + b[2] * x[2]).toBeCloseTo(bq[0] * xq[0] + bq[1] * xq[1] + bq[2] * xq[2], 12);
    }
  });

  // Wigner-3j tables are float32 (as in SE3Transformer's `.float()` basis), so equivariance holds to ~1e-6.
  it('keeps the whole SE(3)-Transformer equivariant (random weights): scalars invariant, vectors and atoms rotate', () => {
    const net = model.buildNetwork(syntheticTensors());
    const trace = shared.parseCaTrace(FIXTURE.cases.helix16.pdb);
    const p1 = shared.prepareFromTrace(trace);
    const Q = rotation(0.7, -0.3, 1.9);
    const t = [12.5, -3.25, 40];
    const traceQ = { ...trace, residues: trace.residues.map((r: any) => ({ ...r, ca: rot(Q, r.ca).map((v, k) => v + t[k]) })) };
    const p2 = shared.prepareFromTrace(traceQ);
    expect(p2.e).toBe(p1.e);
    const o1 = model.forwardCpu(net, p1);
    const o2 = model.forwardCpu(net, p2);
    const scale = Math.max(...Array.from(o1.s as Float64Array, Math.abs));
    expect(maxAbsDiff(o1.s, o2.s) / scale).toBeLessThan(1e-5);
    const vScale = Math.max(...Array.from(o1.v as Float64Array, Math.abs));
    let vErr = 0;
    for (let i = 0; i < p1.n * 3; i++) {
      const q = rot(Q, o1.v, 3 * i);
      for (let k = 0; k < 3; k++) vErr = Math.max(vErr, Math.abs(o2.v[3 * i + k] - q[k]));
    }
    expect(vErr / vScale).toBeLessThan(1e-5);
    const a1 = shared.assemble(p1, o1);
    const a2 = shared.assemble(p2, o2);
    // Compare emitted atoms only (padding slots of the 24-atom layout sit at the origin by construction).
    let rErr = 0;
    a1.atoms.forEach((atoms: { name: string; xyz: number[] }[] | null, i: number) => {
      atoms?.forEach((atom, j) => {
        const q = rot(Q, atom.xyz);
        for (let k = 0; k < 3; k++) rErr = Math.max(rErr, Math.abs(a2.atoms[i][j].xyz[k] - q[k] - t[k] / 10));
      });
    });
    expect(rErr).toBeLessThan(1e-4); // nm
  }, 30_000);
});

describe('cg2all rigid-group all-atom assembly (golden structure-module outputs → atoms)', () => {
  for (const name of ['helix16', 'crambin46', 'twochain'] as const) {
    for (const variant of ['ca', 'ca-fix'] as const) {
      it(`rebuilds ${name} (${variant}) atoms within 5e-3 Å of PyTorch build_structure`, () => {
        const c = FIXTURE.cases[name];
        const p = shared.prepareInput(c.pdb);
        const res = shared.assemble(p, { s: c[variant].s, v: c[variant].v }, { fixAtom: variant === 'ca-fix', hydrogens: true });
        expect(Array.from(res.ss)).toEqual(c[variant].ss);
        expect(maskedRmsd(res.R, c[variant].R, c)).toBeLessThan(5e-3);
        expect(Array.from(res.R as Float64Array).every(Number.isFinite)).toBe(true);
        if (variant === 'ca-fix') {
          for (let i = 0; i < p.n; i++) {
            for (let k = 0; k < 3; k++) expect(res.R[(i * 24 + 1) * 3 + k]).toBeCloseTo(p.pos[3 * i + k], 9);
          }
        }
      });
    }
  }

  it('writes a standard-named heavy-atom PDB with termini, valid bond lengths and no chain breaks', () => {
    const c = FIXTURE.cases.crambin46;
    const p = shared.prepareInput(c.pdb);
    const res = shared.assemble(p, { s: c.ca.s, v: c.ca.v });
    const atoms = res.pdbText.split('\n').filter((l: string) => l.startsWith('ATOM'));
    expect(atoms.every((l: string) => l.slice(76, 78).trim() !== 'H')).toBe(true);
    const names = atoms.map((l: string) => `${l.slice(17, 20)}${Number(l.slice(22, 26))}:${l.slice(12, 16).trim()}`);
    expect(names).toContain('ASN46:OXT');
    expect(names).toContain('ASN46:O');
    expect(names).not.toContain('ASN46:OT1');
    expect(names).toContain('ILE7:CD1'); // CHARMM CD → PDB CD1
    expect(res.atomCount).toBe(atoms.length);
    const parsed = parseStructureText(res.pdbText);
    expect(parsed.nResidues).toBe(46);
    expect(parsed.sequence).toBe(parseStructureText(c.pdb).sequence);
    const geom = computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes);
    expect(geom.chainBreaks).toBe(0);
    const xyz = new Map<string, number[]>(
      atoms.map((l: string) => [`${Number(l.slice(22, 26))}:${l.slice(12, 16).trim()}`, [Number(l.slice(30, 38)), Number(l.slice(38, 46)), Number(l.slice(46, 54))]]),
    );
    const dist = (a: string, b: string) => Math.hypot(...xyz.get(a)!.map((v, k) => v - xyz.get(b)![k]));
    for (let r = 1; r <= 46; r++) {
      expect(dist(`${r}:N`, `${r}:CA`)).toBeGreaterThan(1.35);
      expect(dist(`${r}:N`, `${r}:CA`)).toBeLessThan(1.58);
      expect(dist(`${r}:CA`, `${r}:C`)).toBeGreaterThan(1.42);
      expect(dist(`${r}:CA`, `${r}:C`)).toBeLessThan(1.62);
      if (r < 46) {
        expect(dist(`${r}:C`, `${r + 1}:N`)).toBeGreaterThan(1.2);
        expect(dist(`${r}:C`, `${r + 1}:N`)).toBeLessThan(1.45);
      }
    }
  });

  it('adds CHARMM-predicted hydrogens with PDB names and N-terminal H/H2/H3 on request', () => {
    const c = FIXTURE.cases.helix16;
    const p = shared.prepareInput(c.pdb);
    const res = shared.assemble(p, { s: c.ca.s, v: c.ca.v }, { hydrogens: true });
    const first = res.pdbText.split('\n').filter((l: string) => l.startsWith('ATOM') && Number(l.slice(22, 26)) === 1);
    expect(first.map((l: string) => l.slice(12, 16).trim()).slice(0, 5)).toEqual(['N', 'H', 'H2', 'H3', 'CA']);
    expect(res.pdbText).toContain(' HIS A   3'); // HSD written as HIS
    expect(res.pdbText).not.toContain('HSD');
  });

  it('keeps ligands, chain IDs and TER records for the two-chain trace', () => {
    const c = FIXTURE.cases.twochain;
    const p = shared.prepareInput(c.pdb);
    const keep = shared.assemble(p, { s: c.ca.s, v: c.ca.v });
    const drop = shared.assemble(p, { s: c.ca.s, v: c.ca.v }, { ligands: false });
    expect(keep.pdbText).toMatch(/^HETATM.{6}ZN/m);
    expect(drop.pdbText).not.toMatch(/^HETATM/m);
    expect(keep.pdbText.split('\n').filter((l: string) => l.startsWith('TER'))).toHaveLength(2);
    const parsed = parseStructureText(keep.pdbText);
    expect([...new Set(parsed.chainIds)]).toEqual(['A', 'B']);
    expect(parsed.nResidues).toBe(42);
  });
});

describe('cg2all weight bundles', () => {
  const manifestFor = (dtype: 'float16' | 'int8' | 'float32') => {
    const bytes = new Uint8Array(64);
    const dv = new DataView(bytes.buffer);
    if (dtype === 'float32') [1.5, -2, 0.25, 8].forEach((v, i) => dv.setFloat32(i * 4, v, true));
    if (dtype === 'float16') [0x3e00, 0xc000, 0x3400, 0x4800].forEach((v, i) => dv.setUint16(i * 2, v, true));
    if (dtype === 'int8') {
      [127, -64, 0, 32].forEach((v, i) => dv.setInt8(i, v));
      dv.setUint16(16, 0x3c00, true); // scale 1.0 for the single 4-wide group
    }
    const tensors = [{ name: 't', shape: [1, 4], dtype, offset: 0, bytes: 16, ...(dtype === 'int8' ? { scale: { dtype: 'float16', offset: 16, bytes: 2, groupSize: 4 } } : {}) }];
    return { manifest: { format: 'cg2all-webgpu/v1', tensors }, bytes };
  };

  it('decodes fp32, fp16 and int8 (+fp16 group scale) tensors', () => {
    for (const [dtype, want] of [['float32', [1.5, -2, 0.25, 8]], ['float16', [1.5, -2, 0.25, 8]], ['int8', [127, -64, 0, 32]]] as const) {
      const { manifest, bytes } = manifestFor(dtype);
      expect(Array.from(shared.decodeTensors(manifest, bytes).get('t').data)).toEqual(want);
    }
  });

  it('rejects truncated bundles and SHA-256 mismatches so the worker can evict a corrupt cache entry', async () => {
    const { manifest, bytes } = manifestFor('float32');
    const full = { ...manifest, files: [{ name: 'weights.bin', bytes: bytes.length, sha256: '0'.repeat(64) }] };
    const reader = (payload: Uint8Array) => async (name: string) =>
      name === 'manifest.json' ? new TextEncoder().encode(JSON.stringify(full)) : payload;
    await expect(shared.loadWeights('mem://', { read: reader(bytes) })).rejects.toThrow(/checksum mismatch/);
    await expect(shared.loadWeights('mem://', { read: reader(bytes.slice(0, 10)) })).rejects.toThrow(/truncated/);
    const ok = await shared.loadWeights('mem://', { read: reader(bytes), verify: false });
    expect(ok.tensors.get('t').data[0]).toBe(1.5);
  });
});

// ── WebGPU engine lifecycle against a recording mock device ──
const GPU_GLOBALS = {
  GPUBufferUsage: { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128 },
  GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
  GPUMapMode: { READ: 1, WRITE: 2 },
};
function mockDevice(opts: { loseOnSubmit?: number } = {}) {
  let resolveLost: (info: unknown) => void = () => {};
  const stats = { pipelines: 0, dispatches: 0, submits: 0, maxQueued: 0, queued: 0, destroyed: 0 };
  const device: any = {
    limits: { maxStorageBufferBindingSize: 1 << 30 },
    lost: new Promise((r) => (resolveLost = r)),
    pushErrorScope() {},
    popErrorScope: async () => null,
    createBuffer: (d: { size: number }) => ({
      size: d.size,
      destroy: () => stats.destroyed++,
      mapAsync: async () => {},
      getMappedRange: () => new ArrayBuffer(d.size),
      unmap() {},
    }),
    createShaderModule: (d: unknown) => d,
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipelineAsync: async (d: unknown) => {
      stats.pipelines++;
      return d;
    },
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups: () => stats.dispatches++, end() {} }),
      copyBufferToBuffer() {},
      finish: () => ({}),
    }),
    queue: {
      writeBuffer() {},
      submit: () => {
        stats.submits++;
        stats.queued++;
        stats.maxQueued = Math.max(stats.maxQueued, stats.queued);
        if (opts.loseOnSubmit === stats.submits) resolveLost({ reason: 'unknown', message: 'GPU hang' });
      },
      onSubmittedWorkDone: () => new Promise<void>((r) => setTimeout(() => {
        stats.queued--;
        r();
      }, 0)),
    },
  };
  return { device, stats, lose: (info: unknown) => resolveLost(info) };
}

describe('cg2all WebGPU engine lifecycle', () => {
  // WebGPU enum objects the engine reads; Node has none (each vitest file runs in its own context).
  for (const [k, v] of Object.entries(GPU_GLOBALS)) (globalThis as any)[k] ??= v;
  const net = model.buildNetwork(syntheticTensors(3));
  const prepared = shared.prepareInput(FIXTURE.cases.helix16.pdb);

  it('compiles exactly 6 shape-independent pipelines in parallel and paces ~8 bounded submissions per forward', async () => {
    const { device, stats } = mockDevice();
    const engine = await gpuEngineMod.createGpuEngine({ device, net });
    expect(stats.pipelines).toBe(Object.keys(shaders.KERNELS).length);
    expect(stats.pipelines).toBe(6);
    const out = await engine.predict(prepared);
    expect(out.s).toHaveLength(prepared.n * 20);
    expect(out.v).toHaveLength(prepared.n * 9);
    expect(stats.submits).toBe(8); // init · 6 attention layers · final conv + residual + structure head
    expect(stats.maxQueued).toBeLessThanOrEqual(3);
    expect(stats.dispatches).toBeGreaterThan(100);
    expect(stats.dispatches).toBeLessThan(200);
    const before = stats.pipelines;
    await engine.predict(shared.prepareInput(FIXTURE.cases.crambin46.pdb));
    expect(stats.pipelines).toBe(before); // a new shape reuses the same pipelines
    engine.destroy();
    await expect(engine.predict(prepared)).rejects.toThrow(/destroyed/);
  });

  it('fails fast with a device-lost error when the device is lost before or during a forward pass', async () => {
    const before = mockDevice();
    const engine = await gpuEngineMod.createGpuEngine({ device: before.device, net });
    before.lose({ reason: 'unknown', message: 'GPU hang' });
    await new Promise((r) => setTimeout(r, 0));
    const err = await engine.predict(prepared).catch((e: unknown) => e);
    expect(String(err)).toMatch(/WebGPU device lost \(unknown\): GPU hang/);
    expect(isWebGpuDeviceLostError(err)).toBe(true);
    expect(before.stats.submits).toBe(0);

    const during = mockDevice({ loseOnSubmit: 2 });
    const engine2 = await gpuEngineMod.createGpuEngine({ device: during.device, net });
    const err2 = await engine2.predict(prepared).catch((e: unknown) => e);
    expect(isWebGpuDeviceLostError(err2)).toBe(true);
    expect(during.stats.submits).toBeLessThan(8); // stopped at the next stage boundary
  });

  it('wires the worker to the resilient device helper, device-lost guard and weight-cache eviction', () => {
    const src = readFileSync(resolve(process.cwd(), 'src/lib/plugins/cg2all/cg2all.worker.ts'), 'utf8');
    expect(src).toContain("requestResilientWebGpuDevice({ optionalFeatures: ['shader-f16'] })");
    expect(src).toContain('createDeviceLostGuard()');
    expect(src).toContain('deviceLostGuard.watch(device');
    expect(src).toContain('deviceLostGuard.destroy(gpu?.device)');
    expect(src).toContain('isWebGpuDeviceLostError(err)');
    expect(src).toContain('evictCachedUrls(fetchedUrls)');
    expect(src).toContain("const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/cg2all-webgpu/resolve/main'");
    expect(src).toContain('VITE_CG2ALL_WEIGHTS_URL');
    for (const code of Object.values(shaders.KERNELS) as string[]) {
      expect(code).not.toMatch(/-\s+-\d/); // Naga: no unparenthesized double negatives
      expect((code.match(/var<storage/g) ?? []).length).toBeLessThanOrEqual(8);
    }
  });
});

describe('cg2all plugin #29 registration', () => {
  it('registers manifest, short code, catalog, model card, GPU allowlists and llms.txt', () => {
    const reg = createDefaultPluginRegistry();
    expect(reg.getPlugin('cg2all')).toBe(cg2allPlugin);
    expect(reg.listPlugins()).toHaveLength(29);
    expect(reg.resolveShortCode('cg')).toBe('cg2all');
    expect(ADAPTER_TO_SHORT.cg2all).toBe('cg');
    expect(SHORT_TO_ADAPTER.cg).toBe('cg2all');
    expect(Object.values(ADAPTER_TO_SHORT).filter((v) => v === 'cg')).toHaveLength(1);
    const m = ADAPTER_REGISTRY.cg2all;
    expect(m.runtime).toBe('wgsl-custom');
    expect(m.stage).toBe('score');
    expect(Object.keys(m.inputs)).toEqual(['structure', 'sequences']);
    expect(m.inputs.sequences.required).toBe(false);
    expect(Object.keys(m.outputs)).toEqual(['structure']);
    expect(defaultParamsForAdapter('cg2all')).toEqual({ variant: 'ca', precision: 'fp16', outputHydrogens: false, preserveLigands: true });
    expect(cg2allPlugin.planner?.plannerRole).toBe('optimizer');
    expect(cg2allPlugin.cachePolicy.usesSharedOpfsStore).toBe(true);
    expect(CATALOG_ITEMS.find((c) => c.id === 'cg2all')?.group).toBe('Refinement / Backmapping');
    expect(cg2allPlugin.catalog.categoryId).toBe('refinement');
    const card = getModelCardForAdapter('cg2all', { precision: 'int8' })!;
    expect(card.adapterId).toBe('cg2all');
    expect(card.activePrecisionId).toBe('int8');
    expect(card.weightsRepo).toBe('https://huggingface.co/ramithuh/cg2all-webgpu');
    expect(card.weightsRevision).toBe('ab35079b1fab667ec00a23f6a6cf3fd863cb01e1');
    expect(card.license).toContain('Apache-2.0 (code)');
    expect(card.license).toContain('CC-BY-4.0 (weights, Zenodo 8393343)');
    expect(formatNodeParamSummary('cg2all', { variant: 'ca-fix', precision: 'int8', outputHydrogens: true })).toBe('Fixed Cα · INT8 · +H');
    expect(cg2allPlugin.display?.formatCanvasSummary?.({ precision: 'fp16' })).toBe('Cα refined · FP16');
    const serverMod = createRequire(import.meta.url)(resolve(process.cwd(), '../server/selfhost-server.mjs')) as {
      ALLOWED_GPU_ADAPTER_IDS: Set<string>;
    };
    expect(serverMod.ALLOWED_GPU_ADAPTER_IDS.has('cg2all')).toBe(true);
    expect(generateLlmsTxt()).toMatch(/`cg2all`/);
    expect(readFileSync(resolve(process.cwd(), 'static/vendor/cg2all/LICENSE'), 'utf8')).toContain('Apache License');
  });

  it('round-trips through the flow DSL with params, aliases and a structure edge', () => {
    const flow = parseFlowDsl('src:p:pdbId=1crn~cg:variant=ca-fix,outputHydrogens=true');
    expect(flow.ok).toBe(true);
    const node = flow.graph!.nodes.find((n) => n.adapterId === 'cg2all')!;
    expect(node.params.variant).toBe('ca-fix');
    expect(node.params.outputHydrogens).toBe(true);
    expect(flow.graph!.edges).toContainEqual(expect.objectContaining({ toNode: node.id, toPort: 'structure', fromPort: 'structure' }));
    expect(validateGraph(flow.graph as WorkflowGraph)).toEqual([]);
    const encoded = encodeGraphToFlow(flow.graph as WorkflowGraph);
    const again = decodeFlowQuery(`https://denovo.run/?${encoded.query}`);
    expect(again.ok).toBe(true);
    const node2 = again.graph!.nodes.find((n) => n.adapterId === 'cg2all')!;
    expect(node2.params).toMatchObject({ variant: 'ca-fix', outputHydrogens: true });
    expect(parseFlowDsl('src:p:pdbId=1crn~backmap').graph!.nodes.map((n) => n.adapterId)).toContain('cg2all');
    expect(parseFlowDsl('src:p:pdbId=1crn~cg2all-fix').graph!.nodes.find((n) => n.adapterId === 'cg2all')!.params.variant).toBe('ca-fix');
  });

  it('validates params, pairs sequences with parents and wraps collections', () => {
    expect(cg2allOptionsFromParams({})).toEqual({ variant: 'ca', precision: 'fp16', outputHydrogens: false, preserveLigands: true });
    expect(() => cg2allOptionsFromParams({ variant: 'martini' })).toThrow(/Unsupported cg2all variant/);
    expect(() => cg2allOptionsFromParams({ precision: 'fp32' })).toThrow(/Unsupported cg2all precision/);
    const c = FIXTURE.cases.crambin46;
    const prepared = shared.prepareInput(c.pdb);
    const conv = { ...shared.assemble(prepared, { s: c.ca.s, v: c.ca.v }), prepared };
    const parentParsed = parseStructureText(c.pdb);
    const parent = {
      kind: 'StructureArtifact',
      id: 'crn',
      label: '1CRN Cα',
      sourceModelId: 'pdb-input',
      length: 46,
      caCoords: parentParsed.caCoords,
      backbone4Coords: parentParsed.backbone4Coords,
      pdbText: c.pdb,
      geometry: computeGeometryStats(parentParsed.caCoords),
    } as unknown as StructureArtifact;
    const opts = cg2allOptionsFromParams({});
    const art = buildCg2allArtifact({ parent }, conv, opts, 'webgpu');
    expect(art.sourceModelId).toBe('cg2all');
    expect(art.parentStructureId).toBe('crn');
    expect(art.atomGranularity).toBe('all-atom');
    expect(art.length).toBe(46);
    expect(art.geometry.chainBreaks).toBe(0);
    expect(art.provenance?.quantization).toBe('fp16');
    expect(Array.from(art.caCoords).every(Number.isFinite)).toBe(true);
    const cpuArt = buildCg2allArtifact({ parent }, conv, opts, 'cpu');
    expect(cpuArt.provenance?.degradedFallback).toBe('cpu-js');

    const seqs = {
      kind: 'SequenceSet',
      id: 's',
      candidates: [
        { id: 'd1', sequence: 'A'.repeat(46), sourceStructureId: 'crn' },
        { id: 'd2', sequence: 'G'.repeat(46) },
      ],
    } as unknown as SequenceSet;
    const jobs = planCg2allJobs(structureItems(parent), seqs);
    expect(jobs.map((j) => [j.parent.id, j.candidate?.id])).toEqual([['crn', 'd1'], ['crn', 'd2']]);
    expect(planCg2allJobs([parent], undefined)).toEqual([{ parent }]);
    const wrapped = wrapCg2allOutputs([art, cpuArt], parent);
    expect(wrapped.collection?.items).toHaveLength(2);
    expect(wrapped.id).toMatch(/^cg2all-batch-2-/);
  });

  it('estimates one model per structure, or per sequence when threading, with the precision download size', () => {
    const base = { params: { precision: 'int8' }, upstreamProfiles: { structure: { itemCount: 3 }, sequences: { itemCount: 9 } } };
    const est = (connected: string[]) =>
      cg2allPlugin.preflight!({ ...base, connectedInputs: new Set(connected) } as unknown as Parameters<NonNullable<typeof cg2allPlugin.preflight>>[0]);
    expect(est(['structure']).outputCardinality).toEqual({ structure: 3 });
    expect(est(['structure', 'sequences']).outputCardinality).toEqual({ structure: 9 });
    expect(est(['structure']).estimatedDownloadMB).toBe(4.8);
  });
});

describe('cg2all QA regressions (P0 / P1 / P2)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.doUnmock('http://cg2all-test.local/vendor/cg2all/webgpu/index.js?v=1');
    vi.doUnmock('http://cg2all-test.local/vendor/cg2all/cpu/index.js?v=1');
    vi.resetModules();
  });

  const makeCaLine = (serial: number, resName: string, chain: string, resNum: number, x: number, y: number, z: number, altLoc = ' ') =>
    `ATOM  ${String(serial).padStart(5)}  CA ${altLoc}${resName.padStart(3)} ${chain}${String(resNum).padStart(4)}    ` +
    `${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00  0.00           C`;

  it('P0-2: keeps B-only, C-only, and first-observed altloc residues while letting priority-0 (space/A/1) supersede', () => {
    const pdb = [
      makeCaLine(1, 'ALA', 'A', 1, 0.0, 0.0, 0.0, 'B'),
      makeCaLine(2, 'GLY', 'A', 2, 3.8, 0.0, 0.0, 'C'),
      makeCaLine(3, 'SER', 'A', 3, 7.6, 1.0, 0.0, 'B'),
      makeCaLine(4, 'SER', 'A', 3, 99.0, 99.0, 99.0, 'C'), // second non-primary altloc ignored
      makeCaLine(5, 'VAL', 'A', 4, 50.0, 50.0, 50.0, 'B'),
      makeCaLine(6, 'VAL', 'A', 4, 11.4, 0.0, 1.0, 'A'), // out-of-order A supersedes B
      makeCaLine(7, 'PTR', 'A', 5, 15.2, 1.0, 1.0, 'B'), // UNK residue with altLoc B only
      'TER',
      'END',
    ].join('\n');

    const trace = shared.parseCaTrace(pdb);
    expect(trace.residues).toHaveLength(5);
    expect(trace.residues.map((r: any) => r.resName)).toEqual(['ALA', 'GLY', 'SER', 'VAL', 'PTR']);
    expect(trace.residues[0].ca).toEqual([0, 0, 0]);
    expect(trace.residues[1].ca).toEqual([3.8, 0, 0]);
    expect(trace.residues[2].ca).toEqual([7.6, 1, 0]); // kept first observed 'B', not 'C'
    expect(trace.residues[3].ca).toEqual([11.4, 0, 1]); // 'A' superseded 'B'
    expect(trace.residues[4].lines).toHaveLength(1);
  });

  it('P0-3: filters SSBOND pairs for CYS-CYS both on parseCaTrace and after sequence threading', () => {
    const c = FIXTURE.cases.crambin46;
    // 1. Mutate residue 3 in the input PDB to ALA while leaving stale SSBOND 3-40
    const nonCysInput = c.pdb.replace(/^(ATOM\s+\d+\s+\S+\s+)CYS(\s+A\s+3\b)/gm, '$1ALA$2');
    const nonCysTrace = shared.parseCaTrace(nonCysInput);
    expect(nonCysTrace.ssbonds).toEqual([[3, 31], [15, 25]]);
    expect(nonCysTrace.warnings.some((w: string) => /Ignored SSBOND.*requires CYS/.test(w))).toBe(true);

    // 2. Thread residue 3 (index 2) from CYS -> ALA on valid crambin46 trace
    const seqMut = parseStructureText(c.pdb).sequence.split('');
    seqMut[2] = 'A';
    const threaded = shared.prepareInput(c.pdb, { sequence: seqMut.join('') });
    expect(threaded.trace.ssbonds).toEqual([[3, 31], [15, 25]]);
    expect(threaded.ssbonds).toEqual([[3, 31], [15, 25]]);
    expect(threaded.warnings.some((w: string) => /Ignored SSBOND.*after sequence threading/.test(w))).toBe(true);
    const ssEdges = Array.from({ length: threaded.e }, (_, k) => threaded.edgeFeat[4 * k + 1]).reduce(
      (a: number, b: number) => a + b,
      0,
    );
    expect(ssEdges).toBe(4); // only 2 disulfides × 2 directions

    const res = shared.assemble(threaded, { s: c.ca.s, v: c.ca.v }, { hydrogens: true });
    expect(res.pdbText.split('\n').filter((l: string) => l.startsWith('SSBOND'))).toHaveLength(2);
    // Former partner CYS 40 is now unpaired and keeps its thiol hydrogen (HG in PDB naming), while bonded CYS 4 lacks HG
    const res40Atoms = res.pdbText
      .split('\n')
      .filter((l: string) => l.startsWith('ATOM') && Number(l.slice(22, 26)) === 40)
      .map((l: string) => l.slice(12, 16).trim());
    const res4Atoms = res.pdbText
      .split('\n')
      .filter((l: string) => l.startsWith('ATOM') && Number(l.slice(22, 26)) === 4)
      .map((l: string) => l.slice(12, 16).trim());
    expect(res40Atoms).toContain('HG');
    expect(res4Atoms).not.toContain('HG');

    // 3. Direct build.residueAtoms / build.toPdb guard if trace.residues was mutated manually
    const staleTrace = shared.parseCaTrace(c.pdb);
    staleTrace.residues[2].resName = 'ALA';
    const stalePrep = shared.prepareFromTrace(staleTrace);
    const directRes = shared.assemble(stalePrep, { s: c.ca.s, v: c.ca.v }, { hydrogens: true });
    expect(directRes.pdbText.split('\n').filter((l: string) => l.startsWith('SSBOND'))).toHaveLength(2);
  });

  it('P1-4: rejects blank, NaN, or non-finite Cα coordinates instead of coercing to 0.0', () => {
    const blankCoords = 'ATOM      1  CA  ALA A   1\nEND\n';
    const nanCoords = 'ATOM      1  CA  ALA A   1         NaN   1.000   2.000  1.00  0.00           C\nEND\n';
    const infCoords = 'ATOM      1  CA  ALA A   1       1.000   2.000Infinity  1.00  0.00           C\nEND\n';
    for (const bad of [blankCoords, nanCoords, infCoords]) {
      expect(() => shared.parseCaTrace(bad)).toThrow(/^cg2all: Cα record/);
    }
    // Missing optional B-factor / occupancy columns still succeed cleanly
    const noBf = 'ATOM      1  CA  ALA A   1       1.000   2.000   3.000\nEND\n';
    expect(shared.parseCaTrace(noBf).residues[0].ca).toEqual([1, 2, 3]);
    expect(shared.parseCaTrace(noBf).residues[0].bfactor).toBe(0);
  });

  it('P1-5: emits an axis-rotation warning for L === 2 and collinear Cα chains', () => {
    const twoResPdb = [
      makeCaLine(1, 'ALA', 'A', 1, 0, 0, 0),
      makeCaLine(2, 'GLY', 'A', 2, 3.8, 0, 0),
      'TER\nEND\n',
    ].join('\n');
    const prep2 = shared.prepareInput(twoResPdb);
    expect(prep2.warnings.some((w: string) => /L === 2.*unconstrained/i.test(w))).toBe(true);

    const collinear4Pdb = [
      makeCaLine(1, 'ALA', 'B', 1, 0, 0, 0),
      makeCaLine(2, 'GLY', 'B', 2, 3.8, 0, 0),
      makeCaLine(3, 'VAL', 'B', 3, 7.6, 0, 0),
      makeCaLine(4, 'LEU', 'B', 4, 11.4, 0, 0),
      'TER\nEND\n',
    ].join('\n');
    const prep4 = shared.prepareInput(collinear4Pdb);
    expect(prep4.warnings.some((w: string) => /collinear.*unconstrained/i.test(w))).toBe(true);

    const net = model.buildNetwork(syntheticTensors(5));
    const out = model.forwardCpu(net, prep4);
    const conv = { ...shared.assemble(prep4, out), prepared: prep4 };
    const parent = {
      kind: 'StructureArtifact',
      id: 'col4',
      label: 'col4',
      sourceModelId: 'pdb-input',
      length: 4,
      caCoords: new Float32Array(12),
      backbone4Coords: new Float32Array(48),
      pdbText: collinear4Pdb,
      geometry: computeGeometryStats(new Float32Array(12)),
    } as unknown as StructureArtifact;
    const art = buildCg2allArtifact({ parent }, conv, cg2allOptionsFromParams({}), 'webgpu');
    expect(art.provenance?.warnings?.some((w) => /collinear.*unconstrained/i.test(w))).toBe(true);
  });

  it('P1-6: populates molecularContext (ligands, nucleicAcidChains, hasExplicitHydrogens, chainTypes) accurately', () => {
    const c = FIXTURE.cases.twochain;
    const holoPdb = [
      ...c.pdb.split('\n').filter((l) => !l.startsWith('END')),
      "ATOM    501  P    DG D   1      20.000  20.000  20.000  1.00  0.00           P",
      "ATOM    502  O5'  DG D   1      21.000  20.500  20.000  1.00  0.00           O",
      'TER',
      'END',
    ].join('\n');
    const prep = shared.prepareInput(holoPdb);
    const keepConv = { ...shared.assemble(prep, { s: c.ca.s, v: c.ca.v }, { hydrogens: true, ligands: true }), prepared: prep };
    const dropConv = { ...shared.assemble(prep, { s: c.ca.s, v: c.ca.v }, { hydrogens: false, ligands: false }), prepared: prep };

    const parsedParent = parseStructureText(holoPdb);
    const parent = {
      kind: 'StructureArtifact',
      id: 'holo-twochain',
      label: 'Holo TwoChain',
      sourceModelId: 'pdb-input',
      length: parsedParent.nResidues,
      caCoords: parsedParent.caCoords,
      backbone4Coords: parsedParent.backbone4Coords,
      pdbText: holoPdb,
      molecularContext: {
        chainTypes: { A: 'protein', B: 'protein', D: 'dna', C: 'ligand' },
        ligands: ['ZN'],
        nucleicAcidChains: ['D'],
        hasExplicitHydrogens: true,
      },
      geometry: computeGeometryStats(parsedParent.caCoords, parsedParent.chainIds),
    } as unknown as StructureArtifact;

    const keptArt = buildCg2allArtifact(
      { parent },
      keepConv,
      cg2allOptionsFromParams({ outputHydrogens: true, preserveLigands: true }),
      'webgpu',
    );
    expect(keptArt.molecularContext?.ligands).toEqual(['ZN']);
    expect(keptArt.molecularContext?.nucleicAcidChains).toEqual(['D']);
    expect(keptArt.molecularContext?.hasExplicitHydrogens).toBe(true);
    expect(keptArt.molecularContext?.chainTypes).toEqual({ A: 'protein', B: 'protein', D: 'dna', C: 'ligand' });

    const droppedArt = buildCg2allArtifact(
      { parent },
      dropConv,
      cg2allOptionsFromParams({ outputHydrogens: false, preserveLigands: false }),
      'webgpu',
    );
    expect(droppedArt.molecularContext?.ligands).toBeUndefined();
    expect(droppedArt.molecularContext?.nucleicAcidChains).toBeUndefined();
    expect(droppedArt.molecularContext?.hasExplicitHydrogens).toBeUndefined();
    expect(droppedArt.molecularContext?.chainTypes).toEqual({ A: 'protein', B: 'protein' });
  });

  it('P1-7: emits a provenance warning for intra-chain Cα–Cα gaps in (4.5, 10.0] Å', () => {
    const gapPdb = [
      makeCaLine(1, 'ALA', 'A', 1, 0.0, 0.0, 0.0),
      makeCaLine(2, 'GLY', 'A', 2, 3.8, 0.0, 0.0),
      makeCaLine(3, 'VAL', 'A', 3, 6.2, 2.9, 0.0),
      // 7.5 Å intra-chain gap to residue 6 (within cg2all's 10 Å cutoff)
      makeCaLine(4, 'LEU', 'A', 6, 13.7, 2.9, 0.0),
      makeCaLine(5, 'SER', 'A', 7, 17.5, 2.9, 0.0),
      'TER\nEND\n',
    ].join('\n');
    const prep = shared.prepareInput(gapPdb);
    expect(prep.warnings.some((w: string) => /\(4\.5, 10\.0\] Å.*7\.5 Å.*10 Å \(1\.0 nm\) cutoff/.test(w))).toBe(true);

    const net = model.buildNetwork(syntheticTensors(9));
    const conv = { ...shared.assemble(prep, model.forwardCpu(net, prep)), prepared: prep };
    const parent = {
      kind: 'StructureArtifact',
      id: 'gap-test',
      label: 'gap-test',
      sourceModelId: 'pdb-input',
      length: 5,
      caCoords: new Float32Array(15),
      backbone4Coords: new Float32Array(60),
      pdbText: gapPdb,
      geometry: computeGeometryStats(new Float32Array(15)),
    } as unknown as StructureArtifact;
    const art = buildCg2allArtifact({ parent }, conv, cg2allOptionsFromParams({}), 'webgpu');
    expect(art.provenance?.warnings?.some((w) => /\(4\.5, 10\.0\] Å/.test(w))).toBe(true);
  });

  it('P2-9 & P2-10: evicts cached HTML manifest errors and recomputes cpu-js REMARK + device-loss warning on GPU→CPU fallback', async () => {
    const c = FIXTURE.cases.helix16;
    const prepared = shared.prepareInput(c.pdb);
    const validConv = { ...shared.assemble(prepared, { s: c.ca.s, v: c.ca.v }), prepared };

    const validManifest = new TextEncoder().encode(
      JSON.stringify({ format: 'cg2all-webgpu/v1', fixAtom: false, files: [{ name: 'weights.bin', bytes: 4, sha256: '' }], tensors: [] }),
    );
    const htmlManifest = new TextEncoder().encode('<!DOCTYPE html><html>Captive Portal</html>');
    const cacheStore = new Map<string, Response>([
      ['https://huggingface.co/ramithuh/cg2all-webgpu/resolve/main/ca/fp16/manifest.json', new Response(htmlManifest, { status: 200 })],
    ]);
    const deletedUrls: string[] = [];
    const fakeCache = {
      match: async (url: string) => cacheStore.get(url)?.clone(),
      put: async (url: string, resp: Response) => {
        cacheStore.set(url, resp);
      },
      delete: async (url: string) => {
        deletedUrls.push(url);
        return cacheStore.delete(url);
      },
    };
    vi.stubGlobal('caches', { open: async () => fakeCache });
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (String(url).endsWith('manifest.json')) return new Response(validManifest, { status: 200 });
      return new Response(new Uint8Array(4), { status: 200 });
    }));

    let resolveLost!: (info: unknown) => void;
    const fakeDevice = {
      lost: new Promise((r) => (resolveLost = r)),
      destroy() {},
    };
    vi.stubGlobal('navigator', {
      gpu: {
        requestAdapter: async () => ({
          features: new Set<string>(['shader-f16']),
          limits: {
            maxBufferSize: 1 << 30,
            maxStorageBufferBindingSize: 1 << 30,
            maxStorageBuffersPerShaderStage: 8,
            maxComputeWorkgroupStorageSize: 32768,
            maxComputeInvocationsPerWorkgroup: 1024,
          },
          requestDevice: async () => fakeDevice,
        }),
      },
    });

    const gpuConvertCalls: string[][] = [];
    const cpuConvertCalls: string[][] = [];
    vi.doMock('http://cg2all-test.local/vendor/cg2all/webgpu/index.js?v=1', () => ({
      async loadWeights(_baseUrl: string, opts: { read: (name: string) => Promise<Uint8Array> }) {
        const raw = await opts.read('manifest.json');
        const manifest = JSON.parse(new TextDecoder().decode(raw));
        return { manifest, tensors: new Map() };
      },
      async createCg2allEngine() {
        return { backend: 'webgpu', predict: async () => ({}), destroy() {} };
      },
      async convert(_engine: unknown, _pdb: string, opts: { remarks?: string[] }) {
        gpuConvertCalls.push(opts.remarks ?? []);
        resolveLost({ reason: 'destroyed', message: 'GPU device lost' });
        throw new Error('WebGPU device lost (destroyed): GPU device lost');
      },
    }));
    vi.doMock('http://cg2all-test.local/vendor/cg2all/cpu/index.js?v=1', () => ({
      async loadWeights(_baseUrl: string, opts: { read: (name: string) => Promise<Uint8Array> }) {
        const raw = await opts.read('manifest.json');
        const manifest = JSON.parse(new TextDecoder().decode(raw));
        return { manifest, tensors: new Map() };
      },
      async createCg2allCpuEngine() {
        return { backend: 'cpu', predict: async () => ({}), destroy() {} };
      },
      async convert(_engine: unknown, _pdb: string, opts: { remarks?: string[] }) {
        cpuConvertCalls.push(opts.remarks ?? []);
        const remarkLines = (opts.remarks ?? []).map((r) => `REMARK   1 ${r}`).join('\n');
        return { ...validConv, pdbText: `${remarkLines}\n${validConv.pdbText}` };
      },
    }));

    const messages: Array<Record<string, unknown>> = [];
    const scope = {
      postMessage: (m: Record<string, unknown>) => messages.push(m),
      location: { origin: 'http://cg2all-test.local' },
      onmessage: null as null | ((ev: { data: unknown }) => Promise<void>),
    };
    vi.stubGlobal('self', scope);
    vi.resetModules();
    await import('$lib/plugins/cg2all/cg2all.worker');

    const parsedParent = parseStructureText(c.pdb);
    const parent = {
      kind: 'StructureArtifact',
      id: 'hlx-worker',
      label: 'Helix 16',
      sourceModelId: 'pdb-input',
      length: parsedParent.nResidues,
      caCoords: parsedParent.caCoords,
      backbone4Coords: parsedParent.backbone4Coords,
      pdbText: c.pdb,
      geometry: computeGeometryStats(parsedParent.caCoords),
    } as unknown as StructureArtifact;

    await scope.onmessage!({
      data: {
        id: 1,
        type: 'run',
        params: { variant: 'ca', precision: 'fp16', outputHydrogens: false, preserveLigands: true },
        inputs: { structure: parent },
      },
    });

    const reply = messages.find((m) => m.id === 1 && (m.type === 'result' || m.type === 'error')) as any;
    expect(reply?.type, String(reply?.message ?? '')).toBe('result');
    expect(deletedUrls.some((u) => u.endsWith('manifest.json'))).toBe(true);
    expect(gpuConvertCalls).toHaveLength(2);
    expect(gpuConvertCalls[0][0]).toContain('webgpu)');
    expect(cpuConvertCalls).toHaveLength(1);
    expect(cpuConvertCalls[0][0]).toContain('cpu-js)');

    const outArt = reply.outputs.structure as StructureArtifact;
    expect(outArt.pdbText).toContain('REMARK   1 cg2all CalphaBasedModel (fp16, cpu-js)');
    expect(outArt.provenance?.degradedFallback).toBe('cpu-js');
    expect(outArt.provenance?.warnings?.join(' ')).toMatch(/device loss/i);
    expect(outArt.provenance?.warnings?.join(' ')).not.toMatch(/WebGPU unavailable/i);
  });
});

const WEIGHTS_DIR = process.env.CG2ALL_WEIGHTS_DIR ?? '/tmp/cg2all-weights';
const HAVE_WEIGHTS = existsSync(join(WEIGHTS_DIR, 'ca/fp16/manifest.json')) && existsSync(join(WEIGHTS_DIR, 'ca-fix/int8/manifest.json'));

describe.skipIf(!HAVE_WEIGHTS)('cg2all full-network parity vs PyTorch (exported checkpoints present)', () => {
  const bundles = new Map<string, any>();
  const net = async (variant: string, precision: string) => {
    const key = `${variant}/${precision}`;
    if (!bundles.has(key)) {
      const dir = join(WEIGHTS_DIR, variant, precision);
      const weights = await shared.loadWeights(dir, { read: async (name: string) => new Uint8Array(readFileSync(join(dir, name))) });
      bundles.set(key, { weights, net: model.buildNetwork(weights.tensors) });
    }
    return bundles.get(key);
  };
  for (const [precision, tol] of [['fp16', 0.05], ['int8', 0.15]] as const) {
    for (const variant of ['ca', 'ca-fix'] as const) {
      it(`${variant}/${precision}: all-atom RMSD < ${tol} Å on helix16 / crambin46 / twochain`, async () => {
        const { weights, net: network } = await net(variant, precision);
        expect(weights.manifest.fixAtom).toBe(variant === 'ca-fix');
        for (const name of ['helix16', 'crambin46', 'twochain'] as const) {
          const c = FIXTURE.cases[name];
          const p = shared.prepareInput(c.pdb);
          const out = model.forwardCpu(network, p);
          const res = shared.assemble(p, out, { fixAtom: weights.manifest.fixAtom, hydrogens: true });
          // INT8 noise may flip one near-tied secondary-structure logit (crambin, CA checkpoint)
          const ssFlips = Array.from(res.ss as Int32Array).filter((v, i) => v !== c[variant].ss[i]).length;
          expect(ssFlips).toBeLessThanOrEqual(precision === 'fp16' ? 0 : 1);
          expect(maskedRmsd(res.R, c[variant].R, c)).toBeLessThan(tol);
        }
      }, 120_000);
    }
  }
});
