import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { beforeAll, describe, expect, it } from 'vitest';
import { BENCHMARK_TARGETS } from '$lib/engine/benchmarks';
import { resolveGenie3CompiledInput } from '$lib/engine/conditioning';
import type { StructureArtifact } from '$lib/engine/types';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import {
  buildBoltzGenMotifPlan,
  buildGenie2OptionalMotifs,
  computeGeometricMinGap,
  planMultiIslandMotifPlacement,
  resolveRfdiffusionMotifContig,
  type MotifResidueSpec,
} from '$lib/protein-ops/motif-placement';
import { parseStructureText } from '$lib/protein-ops/pdb';

// Load real vendored plain-JS upstream parsers & feature builders dynamically so tsc does not
// typecheck untyped vendored JS under static/vendor/.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let boltzgenCreateFeatures: (opts: Record<string, unknown>) => Map<string, any>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let boltzgenValidateFeatures: (feat: Map<string, any>) => { N: number; A: number };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let genie2BuildFeatures: (opts: Record<string, unknown>) => Record<string, any>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let protpardelleSelectPositions: (spec: string, structure: any) => number[];
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let protpardelleParseStructure: (text: string, opts?: Record<string, unknown>) => any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfdParseContig: (contig: string, opts?: Record<string, unknown>) => any[];
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfdParsePDB: (text: string) => any;

beforeAll(async () => {
  const vendorUrl = (rel: string) =>
    pathToFileURL(resolve(__dirname, '../../../static/vendor', rel)).href;
  const [bgInput, g2Feat, ppInputs, ppStruct, rfdStruct] = await Promise.all([
    import(/* @vite-ignore */ vendorUrl('boltzgen/shared/input.js')),
    import(/* @vite-ignore */ vendorUrl('genie2/shared/features.js')),
    import(/* @vite-ignore */ vendorUrl('protpardelle/shared/inputs.js')),
    import(/* @vite-ignore */ vendorUrl('protpardelle/shared/structure.js')),
    import(/* @vite-ignore */ vendorUrl('rfdiffusion/shared/structure.js')),
  ]);
  boltzgenCreateFeatures = bgInput.createFeatures;
  boltzgenValidateFeatures = bgInput.validateFeatures;
  genie2BuildFeatures = g2Feat.buildFeatures;
  protpardelleSelectPositions = ppInputs.selectPositions;
  protpardelleParseStructure = ppStruct.parseStructure;
  rfdParseContig = rfdStruct.parseContig;
  rfdParsePDB = rfdStruct.parsePDB;
});

interface SyntheticIsland {
  chain: string;
  startRes: number;
  endRes: number;
  /** Starting Cα [x, y, z] coordinate for the first residue in this island. */
  origin: [number, number, number];
  /** Per-residue Cα step vector (~3.8 Å norm). */
  step?: [number, number, number];
  /** Optional 3-letter residue names cycling along the island. */
  resNames?: string[];
}

function fmtCoord(v: number): string {
  return v.toFixed(3).padStart(8, ' ');
}

/**
 * Generate a valid N/CA/C/O backbone PDB for arbitrary multi-island residue ranges
 * with exact controllable 3D Cα endpoint distances between islands.
 */
function buildSyntheticMotifPdb(islands: SyntheticIsland[]): string {
  const lines: string[] = [];
  let serial = 1;
  for (const isl of islands) {
    const step = isl.step ?? [3.8, 0, 0];
    const names = isl.resNames ?? ['ALA', 'HIS', 'ASP', 'SER', 'GLY', 'LEU'];
    const count = isl.endRes - isl.startRes + 1;
    for (let k = 0; k < count; k++) {
      const resNum = isl.startRes + k;
      const resName = names[k % names.length];
      const cx = isl.origin[0] + k * step[0];
      const cy = isl.origin[1] + k * step[1];
      const cz = isl.origin[2] + k * step[2];
      const atoms: Array<[string, number, number, number, string]> = [
        [' N  ', cx - 1.458, cy + 0.0, cz, 'N'],
        [' CA ', cx, cy, cz, 'C'],
        [' C  ', cx + 0.55, cy + 1.42, cz, 'C'],
        [' O  ', cx + 0.55, cy + 2.65, cz, 'O'],
      ];
      for (const [atomName, x, y, z, elem] of atoms) {
        lines.push(
          `ATOM  ${String(serial++).padStart(5, ' ')} ${atomName} ${resName} ${isl.chain}${String(resNum).padStart(4, ' ')}    ` +
            `${fmtCoord(x)}${fmtCoord(y)}${fmtCoord(z)}  1.00 20.00          ${elem.padStart(2, ' ')}`,
        );
      }
    }
  }
  lines.push('END');
  return `${lines.join('\n')}\n`;
}

function makeStructureArtifactFromPdb(pdbText: string, label = 'synthetic'): StructureArtifact {
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id: `struct-${label}`,
    label,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes),
    atomGranularity: 'backbone4',
    bfactorSemantics: 'none',
  };
}

interface BenchmarkFixture {
  id: string;
  pdbId: string;
  motifSpec: string;
  targetLength: number;
  expectedResidues: number;
  expectedIslands: number;
  pdbText: string;
  artifact: StructureArtifact;
}

function buildAllSixMotifBenchmarkFixtures(): Record<string, BenchmarkFixture> {
  const fixtures: Record<string, { islands: SyntheticIsland[] }> = {
    '5TPN': {
      islands: [{ chain: 'A', startRes: 163, endRes: 181, origin: [0, 0, 0] }],
    },
    '1PRW': {
      // A16-35 (20 res) -> A52-71 (20 res): delta = 17, ~10.5 Å jump
      islands: [
        { chain: 'A', startRes: 16, endRes: 35, origin: [0, 0, 0], step: [1.5, 0, 0] },
        { chain: 'A', startRes: 52, endRes: 71, origin: [28.5, 10.5, 0], step: [1.5, 0, 0] },
      ],
    },
    '5YUI': {
      // A93-97 (5 res), A118-120 (3 res), A198-200 (3 res)
      // A97 Cα at [6, 0, 0]; A118 Cα at [6, 9.0, 0] (dist = 9.0 Å)
      // A120 Cα at [9.0, 9.0, 0]; A198 Cα at [9.0, 9.0, 11.93] (exact 11.93 Å native distance!)
      islands: [
        { chain: 'A', startRes: 93, endRes: 97, origin: [0, 0, 0], step: [1.5, 0, 0] },
        { chain: 'A', startRes: 118, endRes: 120, origin: [6, 9.0, 0], step: [1.5, 0, 0] },
        { chain: 'A', startRes: 198, endRes: 200, origin: [9.0, 9.0, 11.93], step: [1.5, 0, 0] },
      ],
    },
    '1YCR': {
      islands: [{ chain: 'B', startRes: 19, endRes: 27, origin: [0, 0, 0], step: [1.5, 0, 0] }],
    },
    '1BCF': {
      // A18-25, A47-54, A92-99, A123-130 (32 res, 4 islands)
      islands: [
        { chain: 'A', startRes: 18, endRes: 25, origin: [0, 0, 0], step: [1.5, 0, 0] },
        { chain: 'A', startRes: 47, endRes: 54, origin: [10.5, 10.0, 0], step: [1.5, 0, 0] },
        { chain: 'A', startRes: 92, endRes: 99, origin: [21.0, 10.0, 11.5], step: [1.5, 0, 0] },
        { chain: 'A', startRes: 123, endRes: 130, origin: [31.5, 0, 11.5], step: [1.5, 0, 0] },
      ],
    },
    '4XOJ': {
      // A55 (1 res), A99 (1 res), A190-192 (3 res)
      // A55 at [0, 0, 0], A99 at [10.0, 0, 0], A190 at [10.0, 15.41, 0] (exact 15.41 Å native distance!)
      islands: [
        { chain: 'A', startRes: 55, endRes: 55, origin: [0, 0, 0], resNames: ['HIS'] },
        { chain: 'A', startRes: 99, endRes: 99, origin: [10.0, 0, 0], resNames: ['ASP'] },
        {
          chain: 'A',
          startRes: 190,
          endRes: 192,
          origin: [10.0, 15.41, 0],
          step: [3.8, 0, 0],
          resNames: ['GLY', 'ASP', 'SER'],
        },
      ],
    },
  };

  const out: Record<string, BenchmarkFixture> = {};
  for (const bench of BENCHMARK_TARGETS.filter((b) => b.category === 'motif')) {
    const spec = fixtures[bench.pdbId];
    if (!spec) throw new Error(`Missing test fixture for motif benchmark ${bench.pdbId}`);
    const pdbText = buildSyntheticMotifPdb(spec.islands);
    const artifact = makeStructureArtifactFromPdb(pdbText, bench.pdbId);
    const expectedResidues = spec.islands.reduce((s, isl) => s + (isl.endRes - isl.startRes + 1), 0);
    out[bench.pdbId] = {
      id: bench.id,
      pdbId: bench.pdbId,
      motifSpec: bench.conditionParams.motifSpan ?? bench.cropSegments ?? '',
      targetLength: bench.approxTokens,
      expectedResidues,
      expectedIslands: spec.islands.length,
      pdbText,
      artifact,
    };
  }
  return out;
}

const MOTIF_FIXTURES = buildAllSixMotifBenchmarkFixtures();
const SIX_MOTIF_IDS = ['5TPN', '1PRW', '5YUI', '1YCR', '1BCF', '4XOJ'] as const;

describe('P0-7: planMultiIslandMotifPlacement & geometric linker lower bounds', () => {
  it('computes strict 3D geometric minGap = max(4, ceil(dist / 3.0) + 2)', () => {
    expect(computeGeometricMinGap([0, 0, 0], [4.0, 0, 0])).toBe(4);
    expect(computeGeometricMinGap([0, 0, 0], [11.93, 0, 0])).toBe(6); // 5YUI A120->A198
    expect(computeGeometricMinGap([0, 0, 0], [15.41, 0, 0])).toBe(8); // 4XOJ A99->A190
    expect(computeGeometricMinGap(null, null)).toBe(4);
  });

  it('never collapses delta > 65 to gap = 0 on 5YUI (delta=78, 11.93 Å) or 4XOJ (delta=91, 15.41 Å)', () => {
    const f5yui = MOTIF_FIXTURES['5YUI'];
    const p5yui = parseStructureText(f5yui.pdbText);
    const specs5yui: MotifResidueSpec[] = p5yui.sourceResidues.map((sr, i) => ({
      chain: sr.authChainId,
      resNum: sr.resNum,
      insCode: sr.insCode,
      caCoord: [p5yui.caCoords[i * 3], p5yui.caCoords[i * 3 + 1], p5yui.caCoords[i * 3 + 2]],
    }));

    const res5yui = planMultiIslandMotifPlacement(specs5yui, { targetLength: 75 });
    expect(res5yui.effectiveLength).toBe(75);
    expect(res5yui.islands).toHaveLength(3);
    expect(res5yui.minInterIslandGaps).toEqual([5, 6]); // ceil(9.0/3)+2=5, ceil(11.93/3)+2=6
    expect(res5yui.interIslandGaps[0]).toBeGreaterThanOrEqual(5);
    expect(res5yui.interIslandGaps[1]).toBeGreaterThanOrEqual(6);
    // Proportional scaling preserves larger linker for the delta=78 jump than delta=21 jump
    expect(res5yui.interIslandGaps[1]).toBeGreaterThan(res5yui.interIslandGaps[0]);
    expect(res5yui.targetIndices).toHaveLength(11);
    expect(res5yui.nFlank + 11 + res5yui.interIslandGaps[0] + res5yui.interIslandGaps[1] + res5yui.cFlank).toBe(75);

    // Even under aggressive targetLength compression (L=40), geometric lower bounds still hold strictly!
    const res5yuiTight = planMultiIslandMotifPlacement(specs5yui, { targetLength: 40 });
    expect(res5yuiTight.effectiveLength).toBe(40);
    expect(res5yuiTight.interIslandGaps[0]).toBeGreaterThanOrEqual(5);
    expect(res5yuiTight.interIslandGaps[1]).toBeGreaterThanOrEqual(6);

    const f4xoj = MOTIF_FIXTURES['4XOJ'];
    const p4xoj = parseStructureText(f4xoj.pdbText);
    const specs4xoj: MotifResidueSpec[] = p4xoj.sourceResidues.map((sr, i) => ({
      chain: sr.authChainId,
      resNum: sr.resNum,
      insCode: sr.insCode,
      caCoord: [p4xoj.caCoords[i * 3], p4xoj.caCoords[i * 3 + 1], p4xoj.caCoords[i * 3 + 2]],
    }));

    const res4xoj = planMultiIslandMotifPlacement(specs4xoj, { targetLength: 150 });
    expect(res4xoj.effectiveLength).toBe(150);
    expect(res4xoj.islands).toHaveLength(3);
    expect(res4xoj.minInterIslandGaps).toEqual([6, 8]); // ceil(10.0/3)+2=6, ceil(15.41/3)+2=8
    expect(res4xoj.interIslandGaps).toEqual([43, 90]); // exact native gaps fit inside 150!
    expect(res4xoj.nFlank).toBe(6);
    expect(res4xoj.cFlank).toBe(6);

    // Under tight L=60, 4XOJ scales proportionally while strictly preserving minGap >= 8 on A99->A190
    const res4xojTight = planMultiIslandMotifPlacement(specs4xoj, { targetLength: 60 });
    expect(res4xojTight.effectiveLength).toBe(60);
    expect(res4xojTight.interIslandGaps[0]).toBeGreaterThanOrEqual(6);
    expect(res4xojTight.interIslandGaps[1]).toBeGreaterThanOrEqual(8);
  });

  it('never collapses chain breaks or non-monotonic jumps, and keeps insertion codes contiguous', () => {
    const specs: MotifResidueSpec[] = [
      { chain: 'A', resNum: 100, insCode: '', caCoord: [0, 0, 0] },
      { chain: 'A', resNum: 100, insCode: 'A', caCoord: [3.8, 0, 0] },
      { chain: 'A', resNum: 101, insCode: '', caCoord: [7.6, 0, 0] },
      // Chain break to B
      { chain: 'B', resNum: 20, insCode: '', caCoord: [7.6, 12.0, 0] },
      { chain: 'B', resNum: 21, insCode: '', caCoord: [11.4, 12.0, 0] },
      // Non-monotonic jump back to B5
      { chain: 'B', resNum: 5, insCode: '', caCoord: [11.4, 12.0, 9.0] },
    ];
    const res = planMultiIslandMotifPlacement(specs, { targetLength: 60 });
    expect(res.islands).toHaveLength(3);
    expect(res.islands[0].length).toBe(3); // 100, 100A, 101 stay contiguous!
    expect(res.interIslandGaps[0]).toBeGreaterThanOrEqual(6); // 12 Å -> ceil(4)+2=6
    expect(res.interIslandGaps[1]).toBeGreaterThanOrEqual(5); // 9 Å -> ceil(3)+2=5
    expect(res.effectiveLength).toBe(60);
  });
});

describe('P0-7: BoltzGen motif placement, 1BCF length adherence, nativeMotifCoords & vendored createFeatures', () => {
  it.each(SIX_MOTIF_IDS)(
    'places %s cleanly and validates against real vendored BoltzGen createFeatures',
    (pdbId) => {
      const f = MOTIF_FIXTURES[pdbId];
      const plan = buildBoltzGenMotifPlan(f.artifact, {
        targetLength: f.targetLength,
        motifOffset: 10,
      });

      // Exact target length honored on ALL 6 motif benchmarks (including 1BCF = 125, NOT 127!)
      expect(plan.effectiveLength).toBe(f.targetLength);
      expect(plan.motifIndices).toHaveLength(f.expectedResidues);
      expect(plan.fixedResidueIndices).toEqual(plan.motifIndices.map((i) => i + 1));
      expect(plan.nativeMotifCoords).toHaveLength(f.expectedResidues);
      expect(plan.placement.islands).toHaveLength(f.expectedIslands);

      // Every inter-island gap is strictly non-zero and >= its geometric lower bound
      for (let g = 0; g < plan.placement.interIslandGaps.length; g++) {
        expect(plan.placement.interIslandGaps[g]).toBeGreaterThanOrEqual(
          plan.placement.minInterIslandGaps[g],
        );
        expect(plan.placement.interIslandGaps[g]).toBeGreaterThanOrEqual(4);
      }

      // Pass directly into REAL vendored BoltzGen createFeatures() + validateFeatures()
      const feat = boltzgenCreateFeatures({
        mode: 'motif',
        L: plan.effectiveLength,
        targetPDB: f.pdbText,
        motifIndices: plan.motifIndices,
        seed: 42,
      });
      const validated = boltzgenValidateFeatures(feat);
      expect(validated.N).toBe(plan.effectiveLength);
      const designMask = feat.get('f:design_mask').data as Float32Array;
      const motifSet = new Set(plan.motifIndices);
      for (let i = 0; i < plan.effectiveLength; i++) {
        expect(designMask[i]).toBe(motifSet.has(i) ? 0 : 1);
      }
    },
  );

  it('honors 1BCF targetLength=125 with motifOffset=10 instead of silently expanding to 127', () => {
    const f1bcf = MOTIF_FIXTURES['1BCF'];
    const plan = buildBoltzGenMotifPlan(f1bcf.artifact, { targetLength: 125, motifOffset: 10 });
    expect(plan.effectiveLength).toBe(125);
    expect(plan.placement.interIslandGaps).toEqual([21, 37, 23]);
    expect(plan.placement.nFlank).toBe(8);
    expect(plan.placement.cFlank).toBe(4);
    expect(plan.motifIndices[0]).toBe(8);
    expect(plan.motifIndices[plan.motifIndices.length - 1]).toBe(120);
  });

  it('extracts nativeMotifCoords from actual selected targetPDB residues even when upstreamStruct.caCoords is uncropped', () => {
    const f5yui = MOTIF_FIXTURES['5YUI'];
    const parsedCropped = parseStructureText(f5yui.pdbText);

    // Simulate an upstreamStruct whose caCoords holds an uncropped 250-aa parent array
    // with completely different coordinates than the cropped 11-aa motif in pdbText!
    const uncroppedCa = new Float32Array(250 * 3);
    for (let i = 0; i < 250 * 3; i++) uncroppedCa[i] = 777.0 + i;

    const mismatchedUpstream: StructureArtifact = {
      ...f5yui.artifact,
      caCoords: uncroppedCa,
    };

    const plan = buildBoltzGenMotifPlan(mismatchedUpstream, {
      targetLength: 75,
      motifOffset: 10,
    });
    expect(plan.nativeMotifCoords).toHaveLength(11);
    for (let k = 0; k < 11; k++) {
      expect(plan.nativeMotifCoords[k][0]).toBeCloseTo(parsedCropped.caCoords[k * 3], 3);
      expect(plan.nativeMotifCoords[k][1]).toBeCloseTo(parsedCropped.caCoords[k * 3 + 1], 3);
      expect(plan.nativeMotifCoords[k][2]).toBeCloseTo(parsedCropped.caCoords[k * 3 + 2], 3);
    }
    // Verify 3D Cα distance across A120 (index 7) -> A198 (index 8) is preserved as 11.93 Å
    const c7 = plan.nativeMotifCoords[7];
    const c8 = plan.nativeMotifCoords[8];
    const dist = Math.hypot(c8[0] - c7[0], c8[1] - c7[1], c8[2] - c7[2]);
    expect(dist).toBeCloseTo(11.93, 2);
  });

  it('rejects 0-protein-residue templates cleanly before model loading', () => {
    const emptyArtifact = makeStructureArtifactFromPdb('HETATM    1 ZN    ZN A 301       0.000   0.000   0.000  1.00  0.00          ZN\nEND\n', 'zn');
    expect(() =>
      buildBoltzGenMotifPlan(emptyArtifact, { targetLength: 75, motifOffset: 10 }),
    ).toThrow(/at least 1 protein residue/i);
  });
});

describe('P0-7: ProtPardelle multi-island placement parity with vendored parseStructure', () => {
  it.each(SIX_MOTIF_IDS)(
    'places %s cleanly using vendored ProtPardelle parseStructure + selectPositions without gap collapse',
    (pdbId) => {
      const f = MOTIF_FIXTURES[pdbId];
      const parsedTemplate = protpardelleParseStructure(f.pdbText, {
        format: 'pdb',
        chainResidueGap: 200,
      });
      const sourceIndices = protpardelleSelectPositions(f.motifSpec, parsedTemplate) as number[];
      expect(sourceIndices).toHaveLength(f.expectedResidues);

      const tplRes = parsedTemplate.residues as Array<{ chain?: string; resid?: number; icode?: string }>;
      const tplCoords = parsedTemplate.coords as Float32Array;
      const tplAtomMask = parsedTemplate.atomMask as Float32Array;

      const motifSpecs: MotifResidueSpec[] = sourceIndices.map((srcIdx) => {
        const r = tplRes[srcIdx];
        const caCoord: [number, number, number] | null =
          tplAtomMask[srcIdx * 37 + 1] > 0.5
            ? [tplCoords[srcIdx * 111 + 3], tplCoords[srcIdx * 111 + 4], tplCoords[srcIdx * 111 + 5]]
            : null;
        return {
          chain: r.chain ?? 'A',
          resNum: r.resid ?? srcIdx + 1,
          insCode: r.icode ?? '',
          caCoord,
        };
      });

      const placement = planMultiIslandMotifPlacement(motifSpecs, {
        targetLength: f.targetLength,
        minNFlank: 4,
        minCFlank: 4,
        allowExpandLength: true,
      });

      expect(placement.effectiveLength).toBe(f.targetLength);
      expect(placement.targetIndices).toHaveLength(f.expectedResidues);
      expect(placement.islands).toHaveLength(f.expectedIslands);
      for (let i = 1; i < placement.targetIndices.length; i++) {
        expect(placement.targetIndices[i]).toBeGreaterThan(placement.targetIndices[i - 1]);
      }
      expect(placement.targetIndices[placement.targetIndices.length - 1]).toBeLessThan(
        placement.effectiveLength,
      );
      for (let g = 0; g < placement.interIslandGaps.length; g++) {
        expect(placement.interIslandGaps[g]).toBeGreaterThanOrEqual(
          placement.minInterIslandGaps[g],
        );
      }
    },
  );

  it('removes legacy delta <= 65 collapse heuristic from both boltzgen.worker.ts and protpardelle.worker.ts', () => {
    const bgSrc = readFileSync(
      resolve(__dirname, '../plugins/boltzgen/boltzgen.worker.ts'),
      'utf-8',
    );
    const ppSrc = readFileSync(
      resolve(__dirname, '../plugins/protpardelle/protpardelle.worker.ts'),
      'utf-8',
    );
    expect(bgSrc).not.toMatch(/delta\s*<=\s*65/);
    expect(ppSrc).not.toMatch(/delta\s*<=\s*65/);
  });
});

describe('P0-8: Genie 2 multi-island motif parser & vendored buildFeatures parity', () => {
  it.each(SIX_MOTIF_IDS)(
    'parses %s (%s) and builds valid vendored Genie 2 features with shared rigid-body group=1',
    (pdbId) => {
      const f = MOTIF_FIXTURES[pdbId];
      const built = buildGenie2OptionalMotifs(f.artifact, f.motifSpec, f.targetLength);
      expect(built).toBeDefined();
      expect(built!.motifs).toHaveLength(f.expectedIslands);
      expect(built!.motifIndices).toHaveLength(f.expectedResidues);
      expect(built!.fixedResidueIndices).toEqual(built!.motifIndices.map((i) => i + 1));
      expect(built!.nativeMotifCoords).toHaveLength(f.expectedResidues);

      // Every island shares group: 1 so inter-island catalytic 3D geometry is rigidly fixed
      for (const m of built!.motifs) {
        expect(m.group).toBe(1);
      }

      // Pass directly into REAL vendored Genie 2 buildFeatures()
      const features = genie2BuildFeatures({
        length: f.targetLength,
        motifs: built!.motifs,
      });

      const fixedCount = Array.from(features.fixed_sequence_mask as Float32Array).reduce(
        (s, v) => s + v,
        0,
      );
      expect(fixedCount).toBe(f.expectedResidues);

      // Pairwise fixed_structure_mask must fix ALL motif residue pairs across all islands
      const structPairCount = Array.from(features.fixed_structure_mask as Float32Array).reduce(
        (s, v) => s + v,
        0,
      );
      expect(structPairCount).toBe(f.expectedResidues * f.expectedResidues);

      // Verify inter-island 3D Cα pairwise distance is preserved after shared-centroid centering
      if (f.expectedResidues >= 2) {
        const firstIdx = built!.motifIndices[0];
        const lastIdx = built!.motifIndices[built!.motifIndices.length - 1];
        const pos = features.atom_positions as Float32Array;
        const dFeat = Math.hypot(
          pos[lastIdx * 3] - pos[firstIdx * 3],
          pos[lastIdx * 3 + 1] - pos[firstIdx * 3 + 1],
          pos[lastIdx * 3 + 2] - pos[firstIdx * 3 + 2],
        );
        const c0 = built!.nativeMotifCoords[0];
        const cN = built!.nativeMotifCoords[built!.nativeMotifCoords.length - 1];
        const dNative = Math.hypot(cN[0] - c0[0], cN[1] - c0[1], cN[2] - c0[2]);
        expect(dFeat).toBeCloseTo(dNative, 3);
      }
    },
  );

  it('accepts comma-, semicolon-, and space-separated multi-island specs and single-residue tokens (4XOJ)', () => {
    const f4xoj = MOTIF_FIXTURES['4XOJ'];
    const byComma = buildGenie2OptionalMotifs(f4xoj.artifact, 'A55, A99, A190-192', 150)!;
    const bySemicolon = buildGenie2OptionalMotifs(f4xoj.artifact, 'A55; A99; A190-192', 150)!;
    const bySpace = buildGenie2OptionalMotifs(f4xoj.artifact, 'A55 A99 A190-192', 150)!;

    expect(byComma.motifIndices).toEqual(bySemicolon.motifIndices);
    expect(byComma.motifIndices).toEqual(bySpace.motifIndices);
    expect(byComma.motifs.map((m) => m.sequence)).toEqual(['H', 'D', 'GDS']);
    // Gaps between A55->A99 and A99->A190 are strictly positive (never 0!)
    expect(byComma.placement.interIslandGaps).toEqual([43, 90]);
  });

  it('enforces minimum 3 residues across ALL islands combined (rejects 2 total residues)', () => {
    const f4xoj = MOTIF_FIXTURES['4XOJ'];
    expect(() => buildGenie2OptionalMotifs(f4xoj.artifact, 'A55, A99', 80)).toThrow(
      /minimum 3 required/i,
    );
  });
});

describe('P1-7: Genie 3 unconditional guard bypasses ligand/DNA template policy', () => {
  it('allows taskMode="unconditional" even when a ligand-only or DNA-only template is wired', () => {
    const ligandOnlyTemplate: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'sm-7c7m',
      label: 'SAM Pocket (7C7M)',
      benchmarkId: 'sm_7c7m_sam',
      sourceModelId: 'pdb-input',
      length: 0,
      sequence: '',
      caCoords: new Float32Array(0),
      backbone4Coords: new Float32Array(0),
      pdbText: 'HETATM    1  N1  SAM A 401       1.000   2.000   3.000  1.00 20.00           N\nEND\n',
      chainIds: [],
      diagnostics: {
        hasLigands: true,
        ligandNames: ['SAM'],
        chainLengths: [],
      } as unknown as StructureArtifact['diagnostics'],
      geometry: computeGeometryStats(new Float32Array(0)),
      atomGranularity: 'all-atom',
      bfactorSemantics: 'none',
    };

    // Unconditional ignores template and must NOT throw a hard-block error!
    const uncond = resolveGenie3CompiledInput(
      { taskMode: 'unconditional', length: 80, seed: 42 },
      ligandOnlyTemplate,
      undefined,
    );
    expect(uncond.length).toBe(80);
    expect(uncond.compiledSpec).toBeUndefined();

    // Motif/binder modes MUST still hard-block incompatible ligand-only template
    expect(() =>
      resolveGenie3CompiledInput(
        { taskMode: 'motif', length: 80, motifSpan: 'A1-10', seed: 42 },
        ligandOnlyTemplate,
        undefined,
      ),
    ).toThrow(/Genie 3/i);
  });

  it('genie3.worker.ts gates evaluateProteinOnlyStructurePolicy on taskMode !== "unconditional" (not || !condArtifact)', () => {
    const g3Src = readFileSync(
      resolve(__dirname, '../plugins/genie3/genie3.worker.ts'),
      'utf-8',
    );
    expect(g3Src).not.toContain("taskMode !== 'unconditional' || !condArtifact");
    expect(g3Src).toContain("if (template && taskMode !== 'unconditional')");
  });
});

describe('P1-7: RFdiffusion 0-protein-residue guard & empty-contig multi-island synthesis', () => {
  it('throws a clear actionable error on 0-protein-residue (ligand/DNA-only) templates', () => {
    const dnaOnly = makeStructureArtifactFromPdb(
      'HETATM    1  P    DA B   1       1.000   2.000   3.000  1.00 20.00           P\nEND\n',
      '7M5W-DNA',
    );
    expect(() => resolveRfdiffusionMotifContig(dnaOnly, '', 90)).toThrow(
      /0 protein ATOM residues/i,
    );
    expect(() => resolveRfdiffusionMotifContig(dnaOnly, '10-10/B1-10/10-10', 90)).toThrow(
      /0 protein ATOM residues/i,
    );
  });

  it.each(SIX_MOTIF_IDS)(
    'synthesizes a valid empty-contig multi-island specification for %s that parses cleanly in vendored RFdiffusion parseContig',
    (pdbId) => {
      const f = MOTIF_FIXTURES[pdbId];
      const contig = resolveRfdiffusionMotifContig(f.artifact, '', f.targetLength);

      // Parse with REAL vendored RFdiffusion parsePDB + parseContig
      const rfdPdb = rfdParsePDB(f.pdbText);
      const residues = rfdParseContig(contig, { pdb: rfdPdb, random: () => 0.5 });

      expect(residues).toHaveLength(f.targetLength);
      const motifRes = residues.filter((r: { source?: unknown }) => Boolean(r.source));
      expect(motifRes).toHaveLength(f.expectedResidues);

      // Verify specific multi-island tokens on 5YUI and 4XOJ (never invalid A93-103!)
      if (pdbId === '5YUI') {
        expect(contig).toContain('A93-97');
        expect(contig).toContain('A118-120');
        expect(contig).toContain('A198-200');
        expect(contig).not.toContain('A93-103');
      } else if (pdbId === '4XOJ') {
        expect(contig).toContain('A55-55');
        expect(contig).toContain('A99-99');
        expect(contig).toContain('A190-192');
      } else if (pdbId === '1BCF') {
        expect(contig).toContain('A18-25');
        expect(contig).toContain('A47-54');
        expect(contig).toContain('A92-99');
        expect(contig).toContain('A123-130');
      } else if (pdbId === '1PRW') {
        expect(contig).toContain('A16-35');
        expect(contig).toContain('A52-71');
      }
    },
  );

  it('caps a large uncropped single-chain protein to 20 motif residues when contig is empty', () => {
    const uncroppedPdb = buildSyntheticMotifPdb([
      { chain: 'A', startRes: 1, endRes: 120, origin: [0, 0, 0] },
    ]);
    const uncroppedArt = makeStructureArtifactFromPdb(uncroppedPdb, 'uncropped-120');
    const contig = resolveRfdiffusionMotifContig(uncroppedArt, '', 80);
    const rfdPdb = rfdParsePDB(uncroppedPdb);
    const residues = rfdParseContig(contig, { pdb: rfdPdb, random: () => 0.5 });
    expect(residues).toHaveLength(80);
    expect(contig).toContain('A1-20');
  });

  it('formats explicit comma-separated contigs to slash-delimited RFdiffusion syntax', () => {
    const f5yui = MOTIF_FIXTURES['5YUI'];
    const contig = resolveRfdiffusionMotifContig(
      f5yui.artifact,
      '15-15, A93-97, 20-20, A118-120, 20-20, A198-200, 10-10',
      75,
    );
    expect(contig).toBe('15-15/A93-97/20-20/A118-120/20-20/A198-200/10-10');
    const residues = rfdParseContig(contig, { pdb: rfdParsePDB(f5yui.pdbText), random: () => 0.5 });
    expect(residues).toHaveLength(76); // 15 + 5 + 20 + 3 + 20 + 3 + 10 = 76
  });
});
