// cg2all (plugin #29): Cα graph/features, e3nn SH + Clebsch–Gordan equivariance, rigid-group all-atom assembly,
// WebGPU engine lifecycle (device.lost fail-fast, bounded pipelines/pacing), plugin registration & DSL round-trip,
// and pinned goldens from the PyTorch reference (huhlim/cg2all a00b881, CalphaBasedModel / -FIX, FP32).
// Full-network parity against the real checkpoints runs when exported bundles are present
// (CG2ALL_WEIGHTS_DIR, default /tmp/cg2all-weights; exporter and golden tooling in tools/cg2all/).

import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
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
  });
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
    expect(CATALOG_ITEMS.find((c) => c.id === 'cg2all')?.group).toBe('Evaluation');
    const card = getModelCardForAdapter('cg2all', { precision: 'int8' })!;
    expect(card.adapterId).toBe('cg2all');
    expect(card.activePrecisionId).toBe('int8');
    expect(card.license).toBe('Apache-2.0');
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
