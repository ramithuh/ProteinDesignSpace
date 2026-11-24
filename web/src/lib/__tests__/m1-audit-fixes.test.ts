import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { ADAPTER_REGISTRY } from '../engine/registry';
import { defaultParamsForAdapter } from '../engine/graph-utils';
import { validateParamsWithSchema } from '../engine/param-schema';
import {
  compileMpnnInputSpec,
  parseFixedPositionsAndChains,
  type OrderedStructureResidue,
} from '../engine/mpnn-spec';
import { planMpnnFanout } from '../engine/work-plan';
import { compileNodeInputPreview } from '../engine/input-preview';
import { validateAdapterOutputs } from '../engine/scheduler';
import { sanitizeEvoDiffFixedSequence } from '../engine/param-validation';
import { parseStructureText } from '../protein-ops/pdb';
import { computeGeometryStats } from '../protein-ops/kabsch';
import type { StructureArtifact, WorkflowGraph } from '../engine/types';

const vendorUrl = (rel: string) => pathToFileURL(resolve(process.cwd(), rel)).href;

function makeLinearPdb(
  chains: Array<{ chain: string; resNums: number[]; aas?: string[]; insCodes?: string[]; shift?: [number, number, number] }>,
): string {
  const lines: string[] = [];
  let serial = 1;
  const fmt = (n: number) => n.toFixed(3).padStart(8);
  const aaMap: Record<string, string> = {
    A: 'ALA', C: 'CYS', D: 'ASP', E: 'GLU', F: 'PHE',
    G: 'GLY', H: 'HIS', I: 'ILE', K: 'LYS', L: 'LEU',
    M: 'MET', N: 'ASN', P: 'PRO', Q: 'GLN', R: 'ARG',
    S: 'SER', T: 'THR', V: 'VAL', W: 'TRP', Y: 'TYR',
  };
  for (let cIdx = 0; cIdx < chains.length; cIdx++) {
    const ch = chains[cIdx];
    const [sx, sy, sz] = ch.shift ?? [0, 0, 0];
    for (let i = 0; i < ch.resNums.length; i++) {
      const rn = ch.resNums[i];
      const aa = aaMap[ch.aas?.[i] ?? 'A'] ?? 'ALA';
      const ins = (ch.insCodes?.[i] ?? '').padEnd(1).slice(0, 1);
      const x = sx + i * 3.8;
      const y = sy + cIdx * 12.0;
      const z = sz;
      const rStr = String(rn).padStart(4);
      lines.push(`ATOM  ${String(serial++).padStart(5)}  N   ${aa} ${ch.chain}${rStr}${ins}   ${fmt(x - 1.2)}${fmt(y)}${fmt(z)}  1.00 20.00           N  `);
      lines.push(`ATOM  ${String(serial++).padStart(5)}  CA  ${aa} ${ch.chain}${rStr}${ins}   ${fmt(x)}${fmt(y)}${fmt(z)}  1.00 20.00           C  `);
      lines.push(`ATOM  ${String(serial++).padStart(5)}  C   ${aa} ${ch.chain}${rStr}${ins}   ${fmt(x + 1.2)}${fmt(y + 0.5)}${fmt(z)}  1.00 20.00           C  `);
      lines.push(`ATOM  ${String(serial++).padStart(5)}  O   ${aa} ${ch.chain}${rStr}${ins}   ${fmt(x + 1.2)}${fmt(y + 1.7)}${fmt(z)}  1.00 20.00           O  `);
    }
    lines.push('TER');
  }
  lines.push('END');
  return lines.join('\n') + '\n';
}

function makeArtifactFromPdb(id: string, pdbText: string, extra: Partial<StructureArtifact> = {}): StructureArtifact {
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'test',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes),
    ...extra,
  };
}

describe('M1 RTX 5090 audit fixes', () => {
  it('§3.1a-d & §3.5b: updates trained step defaults across registry, graph-utils, and workers, and caps Genie 2 scale at 1.0', () => {
    expect(ADAPTER_REGISTRY.genie2.paramSchema!.find((f) => f.key === 'steps')?.defaultValue).toBe(1000);
    expect(ADAPTER_REGISTRY.genie2.paramSchema!.find((f) => f.key === 'scale')?.max).toBe(1.0);
    expect(ADAPTER_REGISTRY.proteina.paramSchema!.find((f) => f.key === 'steps')?.defaultValue).toBe(400);
    expect(ADAPTER_REGISTRY['la-proteina'].paramSchema!.find((f) => f.key === 'steps')?.defaultValue).toBe(400);
    expect(ADAPTER_REGISTRY.rfdiffusion.paramSchema!.find((f) => f.key === 'numSteps')?.defaultValue).toBe(50);

    expect(defaultParamsForAdapter('genie2')).toMatchObject({ steps: 1000, scale: 0.6 });
    expect(defaultParamsForAdapter('proteina')).toMatchObject({ steps: 400 });
    expect(defaultParamsForAdapter('la-proteina')).toMatchObject({ steps: 400 });
    expect(defaultParamsForAdapter('rfdiffusion')).toMatchObject({ numSteps: 50 });

    const scaleErrs = validateParamsWithSchema(ADAPTER_REGISTRY.genie2, {
      ...defaultParamsForAdapter('genie2'),
      scale: 1.2,
    });
    expect(scaleErrs.some((e) => /1\.0|1/.test(e))).toBe(true);
  });

  it('§3.1d & §3.3: RFdiffusion sample() spans 50..1 for numSteps<50, fixes step/40 progress, centers motif coords, and keeps nativeMotifCoords uncentered', async () => {
    const { CPUAlgebra } = await import(vendorUrl('static/vendor/rfdiffusion/cpu/algebra.js'));
    const { sample } = await import(vendorUrl('static/vendor/rfdiffusion/shared/sampler.js'));
    const { reverseStep } = await import(vendorUrl('static/vendor/rfdiffusion/shared/diffusion.js'));
    const o = new CPUAlgebra();

    // Strided posterior works for nextT < t - 1 and validates nextT bounds
    const xt = o.tensor([2, 4, 3], new Float32Array(24).fill(1));
    const px0 = o.tensor([2, 4, 3], new Float32Array(24).fill(2));
    const strided = reverseStep(o, {
      xt,
      px0,
      t: 50,
      nextT: 48,
      translationNoise: o.tensor([2, 3], new Float32Array(6)),
      rotationNoise: o.tensor([2, 3], new Float32Array(6)),
      mask: [0, 0],
    });
    expect(strided.coords.shape).toEqual([2, 4, 3]);
    expect(() =>
      reverseStep(o, {
        xt,
        px0,
        t: 10,
        nextT: 10,
        translationNoise: o.tensor([2, 3], new Float32Array(6)),
        rotationNoise: o.tensor([2, 3], new Float32Array(6)),
        mask: [0, 0],
      }),
    ).toThrow(/nextT/);

    const shift: [number, number, number] = [100, -50, 80];
    const motifPdb = makeLinearPdb([{ chain: 'A', resNums: [1, 2, 3], shift }]);
    const observedT: number[] = [];
    const progressEvents: Array<{ step?: number; total?: number; blockTotal?: number }> = [];
    let initialInputCoords: Float32Array | null = null;

    const res = await sample(
      o,
      async (input: { length: number; t1d: Float32Array; xyz: Float32Array }, { onProgress }: { onProgress?: (p: unknown) => void }) => {
        const L = input.length;
        onProgress?.({ phase: 'denoise', block: 1, total: 40 });
        if (!initialInputCoords) initialInputCoords = Float32Array.from(input.xyz);
        // Residue 0 is scaffold (mask=0), so t1d[21] === 1 - t / 50
        observedT.push(Math.round((1 - input.t1d[21]) * 50));
        return {
          px0: new Float32Array(L * 9).fill(0.5),
          coords: new Float32Array(L * 12).fill(0.5),
          alpha: new Float32Array(L * 20),
        };
      },
      {
        contig: '2-2/A1-3/2-2',
        inputPdb: motifPdb,
        numSteps: 25,
        seed: 7,
        onProgress: (p: { step?: number; total?: number; blockTotal?: number }) => progressEvents.push(p),
      },
    );

    expect(observedT).toHaveLength(25);
    expect(observedT[0]).toBe(50);
    expect(observedT[observedT.length - 1]).toBe(1);
    for (let i = 1; i < observedT.length; i++) {
      expect(observedT[i]).toBeLessThan(observedT[i - 1]);
    }
    expect(progressEvents.every((p) => p.total === 25)).toBe(true);
    expect(progressEvents.some((p) => p.blockTotal === 40)).toBe(true);
    expect(Array.from(res.diffusionMask)).toEqual([0, 0, 1, 1, 1, 0, 0]);
    expect(res.motifIndices).toEqual([2, 3, 4]);

    // Motif CA coordinates fed into model (input.xyz has stride 9 = N/CA/C) are centered around (0, 0, 0)
    const motifCom = [0, 0, 0];
    for (const idx of res.motifIndices) {
      motifCom[0] += initialInputCoords![idx * 9 + 3];
      motifCom[1] += initialInputCoords![idx * 9 + 4];
      motifCom[2] += initialInputCoords![idx * 9 + 5];
    }
    expect(Math.hypot(motifCom[0] / 3, motifCom[1] / 3, motifCom[2] / 3)).toBeLessThan(1e-3);

    // Returned nativeMotifCoords remain uncentered at original crystal coordinates
    expect(res.nativeMotifCoords).toHaveLength(3);
    expect(res.nativeMotifCoords[0][0]).toBeCloseTo(100, 2);
    expect(res.nativeMotifCoords[0][1]).toBeCloseTo(-50, 2);
    expect(res.nativeMotifCoords[0][2]).toBeCloseTo(80, 2);
  });

  it('§3.4a: RFdiffusion symmetry tiedPositions (global 1..L numbering) resolves cleanly in compileMpnnInputSpec & planMpnnFanout', () => {
    const c3Pdb = makeLinearPdb([
      { chain: 'A', resNums: [1, 2, 3] },
      { chain: 'B', resNums: [4, 5, 6] },
      { chain: 'C', resNums: [7, 8, 9] },
    ]);
    const art = makeArtifactFromPdb('rfd-c3', c3Pdb, {
      tiedPositions: 'A1=B4=C7; A2=B5=C8; A3=B6=C9',
      chainRoles: { A: 'scaffold', B: 'scaffold', C: 'scaffold' },
    });

    const compiled = compileMpnnInputSpec(art, {});
    expect(compiled.errors).toEqual([]);
    expect(compiled.tiedPositions).toHaveLength(3);
    expect(compiled.residues.filter((r) => r.tieGroupId !== undefined)).toHaveLength(9);

    const unlocked = compileMpnnInputSpec(art, { unlockInherited: true });
    expect(unlocked.tiedPositions).toEqual([]);

    const fanout = planMpnnFanout([art], { modelId: 'proteinmpnn', sequencesPerBackbone: 2, baseSeed: 42 });
    expect(fanout.perBackbonePlans[0].tiedPositions).toBe('A1=B4=C7; A2=B5=C8; A3=B6=C9');
  });

  it('§3.2a-c: ProtPardelle-1c worker strategy, partial diffusion schedule/stateT/absStep/gamma, and mmCIF auth_seq_id numbering', async () => {
    const workerSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/protpardelle/protpardelle.worker.ts'), 'utf8');
    expect(workerSrc).not.toContain("'allatom'");
    expect(workerSrc).toContain("'sidechain-tip'");
    expect(workerSrc).toContain("'backbone-sidechain'");
    expect(workerSrc).toContain('sChurn:');

    // Partial diffusion sampler test with mock CPU ops
    const { sample } = await import(vendorUrl('static/vendor/protpardelle/shared/sampler.js'));
    const { parseStructure } = await import(vendorUrl('static/vendor/protpardelle/shared/structure.js'));
    const ops = {
      fromArray(data: ArrayLike<number>, rows: number, cols: number) {
        return { data: Float32Array.from(data), rows, cols };
      },
      zeros(rows: number, cols: number) {
        return { data: new Float32Array(rows * cols), rows, cols };
      },
      mask(t: { data: Float32Array; rows: number; cols: number }) {
        return t;
      },
      noise(input: { seqMask: ArrayLike<number>; noiseLevel: Float32Array }, d: number) {
        ops.noiseLevels.push(input.noiseLevel[0]);
        return ops.zeros(input.seqMask.length, d);
      },
      features(input: { seqMask: ArrayLike<number> }) {
        return ops.zeros(input.seqMask.length, 4);
      },
      linear(x: { rows: number; cols: number }, name: string) {
        const cols = name.endsWith('proj_out') ? 21 : x.cols;
        return ops.zeros(x.rows, cols);
      },
      silu(x: unknown) {
        return x;
      },
      norm(x: unknown) {
        return x;
      },
      output(_patch: unknown, input: { seqMask: ArrayLike<number> }) {
        return ops.zeros(input.seqMask.length, 111);
      },
      graphFeatures(input: { seqMask: ArrayLike<number> }) {
        ops.mpnnCalledSteps.push(input.seqMask.length);
        return { features: ops.zeros(1, 4), indices: [] };
      },
      neighbors() {
        return ops.zeros(1, 4);
      },
      repeatRows(x: { data: Float32Array }, rows: number) {
        return ops.zeros(rows, x.data.length);
      },
      logSoftmax(x: { rows: number; cols: number }) {
        return ops.zeros(x.rows, x.cols);
      },
      kernel(
        { rows, cols, inputs, cpu }: {
          rows: number;
          cols: number;
          inputs: Array<{ data: Float32Array }>;
          cpu: (i: number, r: number, c: number, arrays: Float32Array[]) => number;
        },
      ) {
        const out = new Float32Array(rows * cols);
        const arrs = inputs.map((x) => x.data);
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            out[r * cols + c] = cpu(r * cols + c, r, c, arrs);
          }
        }
        return { data: out, rows, cols };
      },
      noiseLevels: [] as number[],
      mpnnCalledSteps: [] as number[],
    };
    const config = {
      data: { sigma_data: 10, dummy_fill_mode: 'zero' },
      train: { self_cond_train_prob: 0 },
      model: {
        task: 'ai-allatom',
        struct_model: {
          n_channel: 4,
          n_atoms: 37,
          uvit: { patch_size: 1, n_filt_per_layer: [], n_layers: 0, position_embedding_type: '' },
        },
        mpnn_model: {
          use_self_conditioning: false,
          n_channel: 4,
          n_layers: 0,
          n_decoder_layers: 0,
        },
        crop_conditional_guidance: {},
      },
    };
    const n = 4;
    const partialOut = await sample(
      ops,
      config,
      {
        aatype: new Float32Array(n).fill(0),
        seqMask: new Float32Array(n).fill(1),
        residueIndex: Float32Array.from({ length: n }, (_, i) => i),
        chainIndex: new Float32Array(n).fill(0),
        numSteps: 10,
        partial: {
          structure: { coords: new Float32Array(n * 111), aatype: new Float32Array(n).fill(0) },
          rewindSteps: 3,
          repack: true,
        },
        jumpSteps: true,
        sidechainMode: true,
        sChurn: 0,
      },
      { sequenceOps: ops },
    );
    // 3 rewind steps -> absStep = 7, 8, 9 -> initial noiseLevel = schedule[7] << 800, miniMPNN fires at absStep 8, 9
    // (noise() is called once per denoise step + once per miniMPNN call -> 3 denoise + 2 miniMPNN = 5 calls; denoise levels at 0, 1, 3)
    expect(ops.noiseLevels[0]).toBeLessThan(100);
    expect(ops.mpnnCalledSteps).toHaveLength(2);
    expect(partialOut.coords.data.every(Number.isFinite)).toBe(true);

    // mmCIF auth_seq_id vs label_seq_id
    const cif1ycr = [
      'data_1YCR',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.label_asym_id',
      '_atom_site.auth_seq_id',
      '_atom_site.label_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.occupancy',
      '_atom_site.B_iso_or_equiv',
      '_atom_site.pdbx_PDB_model_num',
      ...[19, 20, 21].flatMap((authId, idx) => {
        const lbl = idx + 1;
        const x = idx * 3.8;
        return [
          `ATOM N . ALA B B ${authId} ${lbl} ? ${(x - 1).toFixed(2)} 0.00 0.00 1.00 20.0 1`,
          `ATOM CA . ALA B B ${authId} ${lbl} ? ${x.toFixed(2)} 0.00 0.00 1.00 20.0 1`,
          `ATOM C . ALA B B ${authId} ${lbl} ? ${(x + 1).toFixed(2)} 1.00 0.00 1.00 20.0 1`,
          `ATOM O . ALA B B ${authId} ${lbl} ? ${(x + 1).toFixed(2)} 2.00 0.00 1.00 20.0 1`,
        ];
      }),
    ].join('\n');
    const authParsed = parseStructure(cif1ycr, { format: 'cif' });
    expect(Array.from(authParsed.originalIndex)).toEqual([19, 20, 21]);
    const lblParsed = parseStructure(cif1ycr, { format: 'cif', seqIdField: 'label' });
    expect(Array.from(lblParsed.originalIndex)).toEqual([1, 2, 3]);
  });

  it('§3.3 Genie 2 worker centers motif coordinates, preserves nativeMotifCoords, and avoids per-step GPU readback', async () => {
    const workerSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/genie2/genie2.worker.ts'), 'utf8');
    expect(workerSrc).toContain('readStepCoordinates: false');
    expect(workerSrc).toContain('nativeMotifCoords');
    expect(workerSrc).toContain("bfactorSemantics: 'none'");

    const { CPUAlgebra } = await import(vendorUrl('static/vendor/genie2/cpu/algebra.js'));
    const { initializeEngine } = await import(vendorUrl('static/vendor/genie2/shared/engine.js'));

    // Build a tiny valid safetensors shard with zero weights for n_pair_transform_layer=0, n_structure_block=0
    const tensorDefs: Array<[string, number[]]> = [
      ['single_feature_net.linear.weight', [4, 35]],
      ['single_feature_net.linear.bias', [4]],
      ['pair_feature_net.linear_s_p_i.weight', [4, 4]],
      ['pair_feature_net.linear_s_p_i.bias', [4]],
      ['pair_feature_net.linear_s_p_j.weight', [4, 4]],
      ['pair_feature_net.linear_s_p_j.bias', [4]],
      ['pair_feature_net.linear_relpos.weight', [4, 5]],
      ['pair_feature_net.linear_relpos.bias', [4]],
      ['pair_feature_net.linear_template.weight', [4, 15]],
      ['pair_feature_net.linear_template.bias', [4]],
      ['pair_feature_net.linear_motif_template.weight', [4, 6]],
      ['pair_feature_net.linear_motif_template.bias', [4]],
    ];
    let byteOffset = 0;
    let totalParams = 0;
    const headerObj: Record<string, unknown> = {};
    const manifestTensors: Record<string, unknown> = {};
    for (const [name, shape] of tensorDefs) {
      const count = shape.reduce((a, b) => a * b, 1);
      totalParams += count;
      const lenBytes = count * 4;
      headerObj[name] = { dtype: 'F32', shape, data_offsets: [byteOffset, byteOffset + lenBytes] };
      manifestTensors[name] = {
        dtype: 'F32',
        shape,
        shard: 'shard-0.safetensors',
        row_start: 0,
      };
      byteOffset += lenBytes;
    }
    const headerBytes = new TextEncoder().encode(JSON.stringify(headerObj));
    const shardBuf = new ArrayBuffer(8 + headerBytes.byteLength + byteOffset);
    new DataView(shardBuf).setBigUint64(0, BigInt(headerBytes.byteLength), true);
    new Uint8Array(shardBuf, 8, headerBytes.byteLength).set(headerBytes);
    const shardSha = createHash('sha256').update(new Uint8Array(shardBuf)).digest('hex');
    const manifestBytes = new TextEncoder().encode(
      JSON.stringify({
        format: 'genie2-safetensors-v1',
        precision: 'fp32',
        num_tensors: tensorDefs.length,
        total_bytes: shardBuf.byteLength,
        total_params: totalParams,
        config: {
          c_s: 4, c_p: 4, rescale: 1, c_pos_emb: 4, max_n_res: 32,
          c_chain_emb: 4, max_n_chain: 8, c_timestep_emb: 4, relpos_k: 1,
          template_dist_n_bin: 4, template_dist_min: 2, template_dist_step: 2,
          n_pair_transform_layer: 0, n_structure_block: 0, n_structure_layer: 0,
          n_structure_transition_layer: 0, n_head_ipa: 1, c_hidden_ipa: 4,
          n_qk_point: 1, n_v_point: 1,
        },
        shards: [
          {
            file: 'shard-0.safetensors',
            byteLength: shardBuf.byteLength,
            bytes: shardBuf.byteLength,
            sha256: shardSha,
          },
        ],
        tensors: manifestTensors,
      }),
    );

    const alg = new CPUAlgebra();
    let readCalls = 0;
    const origRead = alg.read.bind(alg);
    alg.read = async (t: unknown) => {
      readCalls++;
      return origRead(t);
    };
    const eng = await initializeEngine(alg, {
      baseUrl: 'mock://genie2',
      precision: 'fp32',
      read: async (url: string) => (url.endsWith('manifest.json') ? manifestBytes.buffer : shardBuf),
    });

    const stepEvents: Array<{ step: number; coordinates?: Float32Array }> = [];
    const progressEvents: Array<{ step: number; totalSteps: number }> = [];
    await eng.sample({
      length: 4,
      steps: 4,
      scale: 0.6,
      readStepCoordinates: false,
      onStep: (ev: { step: number; coordinates?: Float32Array }) => stepEvents.push(ev),
      onProgress: (ev: { step: number; totalSteps: number }) => progressEvents.push(ev),
    });
    // Only final coordinates + rotations readbacks (2 calls), NOT 4 per-step readbacks!
    expect(readCalls).toBe(2);
    expect(stepEvents).toHaveLength(4);
    expect(stepEvents.every((e) => e.coordinates === undefined)).toBe(true);
    expect(progressEvents.map((e) => e.step)).toEqual([1, 2, 3, 4]);
    eng.destroy();

    const netSrc = readFileSync(resolve(process.cwd(), 'static/vendor/genie2/shared/network.js'), 'utf8');
    expect(netSrc).toContain('{ sync: false }');
  });

  it('§3.4b: BoltzGen-1 createFeatures sets tok.binding_type = 1 on matched target hotspots (including insCodes) & rejects unknown hotspots', async () => {
    const { createFeatures } = await import(vendorUrl('static/vendor/boltzgen/shared/input.js'));
    const targetPDB = makeLinearPdb([
      { chain: 'B', resNums: [54, 55, 100], insCodes: ['', '', 'A'] },
    ]);
    const f = createFeatures({
      mode: 'binder',
      L: 4,
      targetPDB,
      hotspots: ['B:54', 'B:100A'],
    });
    const bt = Array.from(f.get('f:binding_type').data);
    // Tokens 0..3 = designed binder (0), tokens 4,5,6 = target B:54, B:55, B:100A
    expect(bt).toEqual([0, 0, 0, 0, 1, 0, 1]);
    expect(f.metadata.hotspotCount).toBe(2);
    expect(() => createFeatures({ mode: 'binder', L: 4, targetPDB, hotspots: ['B:999'] })).toThrow(/not found/);

    const bgWorkerSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/boltzgen/boltzgen.worker.ts'), 'utf8');
    expect(bgWorkerSrc).toContain('lockedChains');
    expect(bgWorkerSrc).toContain('chainRoles');
    expect(bgWorkerSrc).toContain("atomGranularity: 'backbone4'");
    expect(bgWorkerSrc).toContain("bfactorSemantics: 'none'");
  });

  it('§3.5a: AntiFold/MPNN Kabat insertion codes (H82A/B in H39-94), Kabat CDR-H3 (95-102), and 1N8Z chain auto-detection', async () => {
    // 1. Backwards-compatible 2-arg behavior
    expect(parseFixedPositionsAndChains('H100A, A-2-0, H', 'A')).toEqual({
      lockedChains: ['H'],
      fixedPositions: [
        { chain: 'H', number: 100, insertionCode: 'A' },
        { chain: 'A', number: -2 },
        { chain: 'A', number: -1 },
        { chain: 'A', number: 0 },
      ],
    });

    // 2. Structure-aware 3-arg expansion captures insertion codes inside H39-94 & H100A-100C
    const ordered: OrderedStructureResidue[] = [
      { chain: 'H', number: 81, insertionCode: '' },
      { chain: 'H', number: 82, insertionCode: '' },
      { chain: 'H', number: 82, insertionCode: 'A' },
      { chain: 'H', number: 82, insertionCode: 'B' },
      { chain: 'H', number: 83, insertionCode: '' },
      { chain: 'H', number: 100, insertionCode: 'A' },
      { chain: 'H', number: 100, insertionCode: 'B' },
      { chain: 'H', number: 100, insertionCode: 'C' },
    ];
    const expanded = parseFixedPositionsAndChains('H39-94, H100A-100C', 'H', ordered);
    expect(expanded.fixedPositions).toEqual([
      { chain: 'H', number: 81 },
      { chain: 'H', number: 82 },
      { chain: 'H', number: 82, insertionCode: 'A' },
      { chain: 'H', number: 82, insertionCode: 'B' },
      { chain: 'H', number: 83 },
      { chain: 'H', number: 100, insertionCode: 'A' },
      { chain: 'H', number: 100, insertionCode: 'B' },
      { chain: 'H', number: 100, insertionCode: 'C' },
    ]);

    // 3. Kabat 6W41-like & 1N8Z-like antibody fixtures in compileNodeInputPreview
    const kabatVhNums = [22, 36, 82, 82, 82, 92, 95, 96, 100, 100, 102, 103];
    const kabatVhAas = ['C', 'W', 'L', 'S', 'T', 'C', 'A', 'R', 'G', 'Y', 'D', 'W'];
    const kabatVhIns = ['', '', '', 'A', 'B', '', '', '', '', 'A', '', ''];
    const kabatVlNums = [23, 35, 88, 89, 98];
    const kabatVlAas = ['C', 'W', 'C', 'Q', 'F'];
    const pdb6w41 = makeLinearPdb([
      { chain: 'H', resNums: kabatVhNums, aas: kabatVhAas, insCodes: kabatVhIns },
      { chain: 'L', resNums: kabatVlNums, aas: kabatVlAas },
      { chain: 'C', resNums: [1, 2, 3] },
    ]);
    const graph6w41: WorkflowGraph = {
      id: 'g-6w41',
      name: '6W41 Kabat',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: '6W41', params: { mode: 'custom', pdbText: pdb6w41 } },
        {
          id: 'af',
          adapterId: 'antifold-esmif1',
          label: 'AntiFold',
          params: {
            ...defaultParamsForAdapter('antifold-esmif1'),
            targetChain: 'H',
            regions: 'CDR3',
            fixedPositions: 'H39-94',
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    const prev6w41 = compileNodeInputPreview(graph6w41, 'af', {});
    const hChain = prev6w41.chains.find((c) => c.chainId === 'H')!;
    expect(
      hChain.residues.filter((r) => r.roles.includes('designable')).map((r) => `${r.authSeqId}${r.insCode}`),
    ).toEqual(['95', '96', '100', '100A', '102']);
    expect(
      hChain.residues
        .filter((r) => r.authSeqId === 82)
        .every((r) => r.roles.includes('fixed-sequence')),
    ).toBe(true);

    // 1N8Z-style complex: A=VL, B=VH, C=HER2 with blank targetChain auto-picks B (VH)
    const pdb1n8z = makeLinearPdb([
      { chain: 'A', resNums: kabatVlNums, aas: kabatVlAas },
      { chain: 'B', resNums: kabatVhNums, aas: kabatVhAas, insCodes: kabatVhIns },
      { chain: 'C', resNums: [1, 2, 3] },
    ]);
    const graph1n8z: WorkflowGraph = {
      id: 'g-1n8z',
      name: '1N8Z Auto VH',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: '1N8Z', params: { mode: 'custom', pdbText: pdb1n8z } },
        {
          id: 'af',
          adapterId: 'antifold-esmif1',
          label: 'AntiFold',
          params: { ...defaultParamsForAdapter('antifold-esmif1'), regions: 'CDR3' },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    const prev1n8z = compileNodeInputPreview(graph1n8z, 'af', {});
    expect(prev1n8z.payload.compiledSummary.targetChain).toBe('B');
    expect(prev1n8z.payload.compiledSummary.numbering).toBe('kabat');

    // Vendor AntiFold regionMask respects numbering: 'kabat'
    const antiStruct = await import(vendorUrl('static/vendor/antifold-esmif1/shared/structure.js'));
    const parsed6w41 = antiStruct.parseStructure(pdb6w41);
    const prepKabat = antiStruct.prepareInput(parsed6w41, {
      targetChain: 'H',
      heavyChain: 'H',
      lightChain: 'L',
      numbering: 'kabat',
      regions: ['CDR3'],
    });
    const cdrMask = antiStruct.regionMask(prepKabat.residues, {
      targetChain: 'H',
      heavyChain: 'H',
      lightChain: 'L',
      numbering: 'kabat',
      regions: ['CDRH3'],
    });
    const designedInVendor = prepKabat.residues
      .filter((_: unknown, idx: number) => cdrMask[idx] === 1)
      .map((r: { number: number; insertion: string }) => `${r.number}${r.insertion}`);
    expect(designedInVendor).toEqual(['95', '96', '100', '100A', '102']);
  });

  it('§3.5c LASErMPNN: 7C7M MSE->MET + SE->SD + altloc + SAM preservation and GPU-resident decode parity (1 readMany vs 9*N)', async () => {
    const { parseStructure, featurize, writePdb } = await import(
      vendorUrl('static/vendor/lasermpnn/shared/lasermpnn/structure.js')
    );
    const { makeGeometry } = await import(vendorUrl('static/vendor/lasermpnn/shared/lasermpnn/geometry.js'));
    const { LaserModel } = await import(vendorUrl('static/vendor/lasermpnn/shared/lasermpnn/model.js'));

    // 7C7M-like structure: MSE with SE atom, altloc '1' / 'B', microheterogeneity A=SER/B=THR, and SAM ligand with altloc '1'
    const pdb7c7m = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  ',
      'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  ',
      'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C  ',
      'ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O  ',
      'HETATM    5  N   MSE A   2       3.320   1.510   0.000  1.00 20.00           N  ',
      'HETATM    6  CA  MSE A   2       3.960   2.810   0.000  1.00 20.00           C  ',
      'HETATM    7  C   MSE A   2       5.470   2.700   0.000  1.00 20.00           C  ',
      'HETATM    8  O   MSE A   2       6.050   1.620   0.000  1.00 20.00           O  ',
      'HETATM    9  CB  MSE A   2       3.500   3.800   1.100  1.00 20.00           C  ',
      'HETATM   10  CG  MSE A   2       4.200   5.100   1.100  1.00 20.00           C  ',
      'HETATM   11  SE  MSE A   2       3.800   6.300   2.400  1.00 20.00          SE  ',
      'HETATM   12  CE  MSE A   2       4.900   7.600   2.000  1.00 20.00           C  ',
      'ATOM     13  N  ASER A   3       6.100   3.850   0.000  0.60 20.00           N  ',
      'ATOM     14  CA ASER A   3       7.500   3.950   0.000  0.60 20.00           C  ',
      'ATOM     15  C  ASER A   3       8.200   2.650   0.000  0.60 20.00           C  ',
      'ATOM     16  O  ASER A   3       7.700   1.550   0.000  0.60 20.00           O  ',
      'ATOM     17  N  BTHR A   3       6.150   3.850   0.000  0.40 20.00           N  ',
      'ATOM     18  CA BTHR A   3       7.550   3.950   0.000  0.40 20.00           C  ',
      'ATOM     19  C  BTHR A   3       8.250   2.650   0.000  0.40 20.00           C  ',
      'ATOM     20  O  BTHR A   3       7.750   1.550   0.000  0.40 20.00           O  ',
      'HETATM   21  N1 1SAM A 501       4.000   4.000   4.000  1.00 18.00           N  ',
      'HETATM   22  C2 BSAM A 501       5.000   4.000   4.000  1.00 18.00           C  ',
      'END',
      '',
    ].join('\n');

    const parsed = parseStructure(pdb7c7m);
    expect(parsed.residues).toHaveLength(3);
    expect(parsed.residues[2].resname).toBe('SER'); // microheterogeneity resolved A over B
    expect(parsed.ligands).toHaveLength(2); // SAM with altloc '1' and 'B' retained!

    const idealProtCoords = new Float32Array(21 * 5 * 3);
    idealProtCoords.set([0, 0, 0, 1.458, 0, 0, 2.009, 1.42, 0, 1.251, 2.39, 0, 1.9, -0.7, 1.2], 0);
    const mockWeights = {
      'rotamer_builder.aa_to_chi_angle_atom_index': { data: new Float32Array(21 * 16).fill(0), shape: [21, 16] },
      'rotamer_builder.aa_to_chi_angle_mask': { data: new Float32Array(21 * 4).fill(1), shape: [21, 4] },
      'rotamer_builder.ideal_prot_aa_coords': { data: idealProtCoords, shape: [21, 5, 3] },
      'rotamer_builder.ideal_aa_coords': { data: new Float32Array(21 * 14 * 3), shape: [21, 14, 3] },
      'rotamer_builder.ideal_aa_mask': { data: new Float32Array(21 * 14), shape: [21, 14] },
      'rotamer_builder.chi_angle_rotations': { data: new Float32Array(21 * 4 * 2), shape: [21, 4, 2] },
      'ligand_featurizer.atomic_number_idx_to_period_idx': { data: new Float32Array(120).fill(1), shape: [120] },
      'ligand_featurizer.atomic_number_idx_to_group_idx': { data: new Float32Array(120).fill(1), shape: [120] },
    };
    // Fill MET (index 12) N/CA/CB/CG/SD/CE -> heavyAtomNames[12] = ['N','CA','C','O','CB','CG','SD','CE']
    mockWeights['rotamer_builder.aa_to_chi_angle_atom_index'].data.set(
      [0, 1, 4, 5, 1, 4, 5, 6, 4, 5, 6, 7, 0, 0, 0, 0],
      12 * 16,
    );
    const feat = featurize(parsed, mockWeights, {
      model: {
        graph_structure: {
          pr_pr_knn_graph_k: 4,
          lig_lig_knn_graph_k: 4,
          lig_pr_knn_graph_k: 4,
          lig_pr_distance_cutoff: 10,
        },
      },
    });
    expect(feat.sequence[1]).toBe(12); // MET (not 20 / XAA!)
    expect(Number.isFinite(feat.chi[1 * 4 + 2])).toBe(true); // chi3 computed via SE->SD!

    const outPdb = writePdb(
      { structure: parsed },
      { sequence: feat.sequence },
      { coords: new Float32Array(3 * 14 * 3), atomsPerResidue: 14 },
    );
    const reParsed = parseStructureText(outPdb);
    expect(reParsed.sequence).toBe('AMS');
    expect(reParsed.diagnostics.hasLigands).toBe(true);
    expect(reParsed.diagnostics.ligandNames).toEqual(['SAM']);

    // Verify gpuPipelined vs legacy CPU path equivalence & 1 readMany call
    const makeFakeBackend = (withGpuOps: boolean) => {
      let readCount = 0;
      let readManyCount = 0;
      const weights: Record<string, { data: Float32Array; shape: number[] }> = {
        'sequence_label_embedding.weight': { data: Float32Array.from({ length: 22 * 4 }, (_, i) => ((i % 7) - 3) * 0.1), shape: [22, 4] },
        'rotamer_builder.aa_to_chi_angle_mask': { data: new Float32Array(21 * 4).fill(1), shape: [21, 4] },
        'sequence_output_layer.weight': { data: Float32Array.from({ length: 21 * 4 }, (_, i) => ((i % 5) - 2) * 0.2), shape: [21, 4] },
      };
      for (let i = 0; i < 4; i++) {
        weights[`chi_prediction_layers.${i}.0.weight`] = { data: new Float32Array(72 * (8 + i * 72)).fill(0.01), shape: [72, 8 + i * 72] };
        weights[`chi_offset_prediction_layers.${i}.0.weight`] = { data: new Float32Array(144).fill(0.02), shape: [1, 144] };
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const b: Record<string, any> = {
        weights,
        readCount: () => readCount,
        readManyCount: () => readManyCount,
        mark: () => 0,
        releaseSince: () => {},
        from: (data: ArrayLike<number>, r: number, c: number) => ({ data: Float32Array.from(data), r, c }),
        zeros: (r: number, c: number) => ({ data: new Float32Array(r * c), r, c }),
        weight: (k: string) => weights[k],
        gather: (x: { data: Float32Array; r?: number; shape?: number[]; c?: number }, idx: number[] | { data: Float32Array; r: number }) => {
          const arr = Array.isArray(idx) ? idx : Array.from(idx.data, (v) => Math.round(v));
          const cols = x.c ?? x.shape![1];
          const out = new Float32Array(arr.length * cols);
          for (let r = 0; r < arr.length; r++) out.set(x.data.subarray(arr[r] * cols, (arr[r] + 1) * cols), r * cols);
          return { data: out, r: arr.length, c: cols };
        },
        cat: (xs: Array<{ data: Float32Array; r: number; c: number }>) => {
          const rows = xs[0]?.r ?? 0;
          const totalC = xs.reduce((s, x) => s + x.c, 0);
          const out = new Float32Array(rows * totalC);
          for (let r = 0; r < rows; r++) {
            let off = 0;
            for (const x of xs) {
              out.set(x.data.subarray(r * x.c, (r + 1) * x.c), r * totalC + off);
              off += x.c;
            }
          }
          return { data: out, r: rows, c: totalC };
        },
        rows: (xs: Array<{ data: Float32Array; r: number; c: number }>) => {
          const cols = xs[0].c;
          const totalR = xs.reduce((s, x) => s + x.r, 0);
          const out = new Float32Array(totalR * cols);
          let off = 0;
          for (const x of xs) { out.set(x.data, off); off += x.data.length; }
          return { data: out, r: totalR, c: cols };
        },
        assign: (x: { data: Float32Array; c: number }, row: number, y: { data: Float32Array }) => {
          x.data.set(y.data, row * x.c);
        },
        linear: (x: { data: Float32Array; r: number; c: number }, key: string) => {
          const w = weights[key + '.weight'];
          const out = new Float32Array(x.r * w.shape[0]);
          for (let r = 0; r < x.r; r++)
            for (let oc = 0; oc < w.shape[0]; oc++) {
              let s = 0;
              for (let k = 0; k < x.c; k++) s += x.data[r * x.c + k] * w.data[oc * x.c + k];
              out[r * w.shape[0] + oc] = s;
            }
          return { data: out, r: x.r, c: w.shape[0] };
        },
        read: async (x: { data: Float32Array }) => {
          readCount++;
          return Float32Array.from(x.data);
        },
      };
      if (withGpuOps) {
        b.sampleIndex = (logits: { data: Float32Array; c: number }, temp: number, u: number, mask?: { data: Float32Array } | null) => {
          const vals = Array.from(logits.data, (v, j) => (mask && mask.data[j] < 0.5 ? -3.402823466e38 : v));
          const mx = Math.max(...vals);
          const p = vals.map((v) => Math.exp((v - mx) / temp));
          const sum = p.reduce((a, c) => a + c, 0);
          let cdf = 0;
          let idx = p.length - 1;
          for (let j = 0; j < p.length; j++) { cdf += p[j] / sum; if (u <= cdf) { idx = j; break; } }
          return { data: new Float32Array([idx]), r: 1, c: 1 };
        };
        b.catOneHot = (logits: { data: Float32Array; c: number }, choice: { data: Float32Array }) => {
          const out = new Float32Array(logits.c + 72);
          out.set(logits.data, 0);
          out[logits.c + Math.round(choice.data[0])] = 1;
          return { data: out, r: 1, c: logits.c + 72 };
        };
        b.chiAngle = (choice: { data: Float32Array }, offset: { data: Float32Array }) => {
          const raw = -180 + Math.round(choice.data[0]) * 5 + offset.data[0];
          const s = raw + 180;
          const m = ((s % 360) + 360) % 360;
          return { data: new Float32Array([Math.fround(m - 180)]), r: 1, c: 1 };
        };
        b.chiRbf = (ang: { data: Float32Array }) => {
          const a = Array.from({ length: 72 }, (_, j) => {
            const c = -180 + j * 5;
            const d1 = (((c - ang.data[0]) % 360) + 360) % 360;
            const d2 = (((ang.data[0] - c) % 360) + 360) % 360;
            return Math.exp(-0.5 * (Math.min(d1, d2) / 2.5) ** 2);
          });
          const s = a.reduce((x, y) => x + y, 0);
          return { data: Float32Array.from(a, (v) => v / s), r: 1, c: 72 };
        };
        b.maskRow = (row: { data: Float32Array; r: number; c: number }, mask: { data: Float32Array }, i: number) => ({
          data: row.data.map((v) => v * mask.data[i]), r: row.r, c: row.c,
        });
        b.readMany = async (xs: Array<{ data: Float32Array }>) => {
          readManyCount++;
          return xs.map((x) => Float32Array.from(x.data));
        };
        b.sync = async () => {};
      }
      return b;
    };

    const makeMockModel = (b: ReturnType<typeof makeFakeBackend>) => {
      const m = new LaserModel(b, {
        alphabet: 'ACDEFGHIKLMNPQRSTVWYX',
        model: { node_embedding_dim: 4, num_laser_vectors: 4 },
      });
      m.encode = async (batch: { length: number }) => ({
        lig: { s: b.zeros(0, 4) },
        prot: { s: b.zeros(batch.length, 4), v: b.zeros(batch.length, 12) },
        pe: b.zeros(0, 4),
        ple: b.zeros(0, 4),
        traces: {},
      });
      m.hetero = () => ({ s: b.zeros(1, 4), v: b.zeros(1, 12) });
      m.mlp = (x: { data: Float32Array; r: number; c: number }, prefix: string) => b.linear(x, prefix + '.0');
      m.gvp = () => ({ s: b.zeros(1, 4), v: b.zeros(1, 12) });
      m.eqnorm = (x: unknown) => x;
      return m;
    };

    const batch = {
      length: 4,
      pp: [[], []],
      lp: [[], []],
      fixed: new Int32Array([1, 0, 0, 0]),
      sequence: new Int32Array([0, 1, 2, 3]),
      chi: new Float32Array(16).fill(NaN),
    };
    const legacyB = makeFakeBackend(false);
    const gpuB = makeFakeBackend(true);
    const rLegacy = await makeMockModel(legacyB).run(batch, { seed: 11, omitAAs: 'C', trace: true });
    const rGpu = await makeMockModel(gpuB).run(batch, { seed: 11, omitAAs: 'C', trace: true });

    expect((legacyB.readCount as () => number)()).toBe(36); // 9 * 4 residues
    expect((gpuB.readCount as () => number)()).toBe(0);
    expect((gpuB.readManyCount as () => number)()).toBe(1);
    expect(Array.from(rGpu.sequence)).toEqual(Array.from(rLegacy.sequence));
    expect(Array.from(rGpu.chiBins)).toEqual(Array.from(rLegacy.chiBins));
    expect(Array.from(rGpu.sequenceLogits)).toEqual(Array.from(rLegacy.sequenceLogits));
    expect(Array.from(rGpu.traces.sequence_output_layer)).toEqual(Array.from(rLegacy.traces.sequence_output_layer));

    void makeGeometry;
  });

  it('§3.5d & M2 contracts: ESM-C optional structure port, rowOnly sliced readback parity, pdb-input sequences port, and EvoDiff #MASK# guard', async () => {
    expect(ADAPTER_REGISTRY['pdb-input'].outputs.sequences).toEqual({
      type: 'SequenceSet',
      required: false,
    });
    expect(ADAPTER_REGISTRY.esmc.outputs.structure).toEqual({
      type: 'StructureArtifact',
      required: false,
    });
    expect(() =>
      validateAdapterOutputs(
        { id: 'p', adapterId: 'pdb-input', label: 'PDB', params: {} },
        { structure: { kind: 'StructureArtifact', id: 's1' } },
      ),
    ).not.toThrow();
    expect(() =>
      validateAdapterOutputs(
        { id: 'e', adapterId: 'esmc', label: 'ESMC', params: {} },
        {
          mutations: { kind: 'MutationMatrix', id: 'm1' },
          track: { kind: 'ResidueTrack', id: 't1' },
          sequences: { kind: 'SequenceSet', id: 'q1' },
        },
      ),
    ).not.toThrow();

    const { runTask } = await import(vendorUrl('static/vendor/esmc/shared/tasks.js'));
    const forwardCalls: Array<{ rowOnly?: boolean; traceRow?: number }> = [];
    const mockEngine = {
      config: { max_sequence_length: 1024 },
      async forward(tokens: number[], opts: { rowOnly?: boolean; traceRow?: number }) {
        forwardCalls.push({ rowOnly: opts.rowOnly, traceRow: opts.traceRow });
        const L = tokens.length;
        const fullLlr = Float32Array.from({ length: (L - 2) * 20 }, (_, i) => (i % 7) * 0.1 - 0.3);
        const fullWt = Float32Array.from({ length: L - 2 }, (_, i) => -1.5 - i * 0.1);
        const fullLogits = Float32Array.from({ length: L * 64 }, (_, i) => (i % 13) * 0.05);
        if (opts.rowOnly && opts.traceRow !== undefined) {
          const p = opts.traceRow - 1;
          return {
            llr: fullLlr.subarray(p * 20, (p + 1) * 20),
            wtLogProbs: fullWt.subarray(p, p + 1),
            logits: fullLogits.subarray(opts.traceRow * 64, (opts.traceRow + 1) * 64),
          };
        }
        return { llr: fullLlr, wtLogProbs: fullWt, logits: fullLogits };
      },
      async summarize(wt: Float32Array) {
        const mean = Array.from(wt).reduce((a, b) => a + b, 0) / wt.length;
        return { naturalness: mean, pseudoPerplexity: Math.exp(-mean) };
      },
    };
    const resRow = await runTask(mockEngine, 'ACDEF', { mode: 'masked' });
    expect(forwardCalls).toHaveLength(5);
    expect(forwardCalls.every((c, i) => c.rowOnly === true && c.traceRow === i + 1)).toBe(true);
    expect(resRow.llr).toHaveLength(5 * 20);
    expect(resRow.wtLogProbs).toHaveLength(5);
    expect(resRow.maskedLogits).toHaveLength(5 * 64);

    // EvoDiff #MASK# / [MASK] / <mask> sanitization contract
    const evoSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/evodiff/evodiff.worker.ts'), 'utf8');
    expect(evoSrc).toContain('sanitizeEvoDiffFixedSequence');
    expect(sanitizeEvoDiffFixedSequence('acdefg #MASK# hiklm')).toBe('ACDEFG#HIKLM');
    expect(sanitizeEvoDiffFixedSequence('ACDEFG[MASK]HIKLM')).toBe('ACDEFG#HIKLM');
    expect(sanitizeEvoDiffFixedSequence('ACDEFG<mask>HIKLM')).toBe('ACDEFG#HIKLM');
  });

  it('Iteration-2 regressions: AntiFold 1N8Z chainless parity & locked-H fallback, LASErMPNN CA-only ligand leak & opts.choices+fixed parity & sampleIndex lastValid, BoltzGen authChainId hotspots, regionMask end inscode', async () => {
    // 1. AntiFold 1N8Z chainless fixedPositions parity + locked-H target fallback
    const kabatVhNums = [22, 36, 82, 82, 82, 92, 95, 96, 100, 100, 102, 103];
    const kabatVhAas = ['C', 'W', 'L', 'S', 'T', 'C', 'A', 'R', 'G', 'Y', 'D', 'W'];
    const kabatVhIns = ['', '', '', 'A', 'B', '', '', '', '', 'A', '', ''];
    const kabatVlNums = [23, 35, 88, 95, 98];
    const kabatVlAas = ['C', 'W', 'C', 'Q', 'F'];
    const pdb1n8z = makeLinearPdb([
      { chain: 'A', resNums: kabatVlNums, aas: kabatVlAas },
      { chain: 'B', resNums: kabatVhNums, aas: kabatVhAas, insCodes: kabatVhIns },
      { chain: 'C', resNums: [1, 2, 3] },
    ]);
    const graph1n8zChainless: WorkflowGraph = {
      id: 'g-1n8z-chainless',
      name: '1N8Z Chainless Fixed',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: '1N8Z', params: { mode: 'custom', pdbText: pdb1n8z } },
        {
          id: 'af',
          adapterId: 'antifold-esmif1',
          label: 'AntiFold',
          params: {
            ...defaultParamsForAdapter('antifold-esmif1'),
            samplingMode: 'oneshot',
            targetChain: '',
            fixedPositions: '95',
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    const prev1n8z = compileNodeInputPreview(graph1n8zChainless, 'af', {});
    expect(prev1n8z.payload.compiledSummary.targetChain).toBe('B');
    const chainA = prev1n8z.chains.find((c) => c.chainId === 'A')!;
    const chainB = prev1n8z.chains.find((c) => c.chainId === 'B')!;
    // Chainless '95' must lock B:95 (targetChain B), NOT A:95
    expect(chainB.residues.find((r) => r.authSeqId === 95)!.roles).toContain('fixed-sequence');
    expect(chainA.residues.find((r) => r.authSeqId === 95)!.roles).toEqual(['designable']);

    // Worker uses the same two-step resolvedTargetChain parsing & skips locked chains when picking effectiveTargetChain
    const afWorkerSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/antifold-esmif1/antifold.worker.ts'), 'utf8');
    expect(afWorkerSrc).toContain('parseFixedPositionsAndChains(fixedPositionsSpec, resolvedTargetChain, orderedResidues)');
    expect(afWorkerSrc).toContain('unlockedHeavy ?? unlockedLight ?? firstUnlocked');

    // Locking H on 6W41 with blank targetChain targets L (preserving heavyChain/lightChain for CDRL1)
    const pdb6w41 = makeLinearPdb([
      { chain: 'H', resNums: kabatVhNums, aas: kabatVhAas, insCodes: kabatVhIns },
      { chain: 'L', resNums: [23, 24, 34, 35, 88, 98], aas: ['C', 'R', 'S', 'W', 'C', 'F'] },
      { chain: 'C', resNums: [1, 2, 3] },
    ]);
    const graphLockH: WorkflowGraph = {
      id: 'g-6w41-lock-h',
      name: '6W41 Lock H',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', label: '6W41', params: { mode: 'custom', pdbText: pdb6w41 } },
        {
          id: 'af',
          adapterId: 'antifold-esmif1',
          label: 'AntiFold',
          params: {
            ...defaultParamsForAdapter('antifold-esmif1'),
            targetChain: '',
            fixedPositions: 'H',
            regions: 'CDRL1',
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    const prevLockH = compileNodeInputPreview(graphLockH, 'af', {});
    expect(prevLockH.payload.compiledSummary.targetChain).toBe('L');
    expect(prevLockH.issues.some((i) => /Invalid AntiFold/i.test(i.message))).toBe(false);
    expect(
      prevLockH.chains
        .find((c) => c.chainId === 'L')!
        .residues.filter((r) => r.roles.includes('designable'))
        .map((r) => r.authSeqId),
    ).toEqual([24, 34]);

    // 2. LASErMPNN CA-only / incomplete protein residue must NOT leak into ligands
    const { parseStructure: parseLaser } = await import(
      vendorUrl('static/vendor/lasermpnn/shared/lasermpnn/structure.js')
    );
    const mixedCaPdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  ',
      'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  ',
      'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C  ',
      'ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O  ',
      'ATOM      5  N   ALA A   2       3.320   1.510   0.000  1.00 20.00           N  ',
      'ATOM      6  CA  ALA A   2       3.960   2.810   0.000  1.00 20.00           C  ',
      'ATOM      7  C   ALA A   2       5.470   2.700   0.000  1.00 20.00           C  ',
      'ATOM      8  O   ALA A   2       6.050   1.620   0.000  1.00 20.00           O  ',
      'ATOM      9  N   ALA A   3       6.100   3.850   0.000  1.00 20.00           N  ',
      'ATOM     10  CA  ALA A   3       7.500   3.950   0.000  1.00 20.00           C  ',
      'ATOM     11  C   ALA A   3       8.200   2.650   0.000  1.00 20.00           C  ',
      'ATOM     12  O   ALA A   3       7.700   1.550   0.000  1.00 20.00           O  ',
      'ATOM     13  CA  GLY A   4      10.000   2.650   0.000  1.00 20.00           C  ',
      'HETATM   14  CA  MSE A   5      12.000   2.650   0.000  1.00 20.00           C  ',
      'HETATM   15  N1  SAM A 501       4.000   4.000   4.000  1.00 18.00           N  ',
      'END',
      '',
    ].join('\n');
    const laserParsed = parseLaser(mixedCaPdb);
    expect(laserParsed.residues).toHaveLength(3);
    expect(laserParsed.ligands.map((l: { resname: string; name: string }) => `${l.resname}:${l.name}`)).toEqual([
      'SAM:N1',
    ]);
    const laserWorkerSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/lasermpnn/lasermpnn.worker.ts'), 'utf8');
    expect(laserWorkerSrc).toContain('completeChainList');
    expect(laserWorkerSrc).toContain('!entry.candidate.chainLengths');

    // 3. LASErMPNN gpuPipelined vs legacy parity when opts.choices is supplied with fixed residues
    const { LaserModel } = await import(vendorUrl('static/vendor/lasermpnn/shared/lasermpnn/model.js'));
    const makeChoiceBackend = (withGpuOps: boolean) => {
      const weights: Record<string, { data: Float32Array; shape: number[] }> = {
        'sequence_label_embedding.weight': {
          data: Float32Array.from({ length: 22 * 4 }, (_, i) => ((i % 7) - 3) * 0.1),
          shape: [22, 4],
        },
        'rotamer_builder.aa_to_chi_angle_mask': {
          data: new Float32Array(21 * 4).fill(1),
          shape: [21, 4],
        },
        'sequence_output_layer.weight': {
          data: Float32Array.from({ length: 21 * 4 }, (_, i) => ((i % 5) - 2) * 0.2),
          shape: [21, 4],
        },
      };
      for (let i = 0; i < 4; i++) {
        weights[`chi_prediction_layers.${i}.0.weight`] = {
          data: new Float32Array(72 * (8 + i * 72)).fill(0.01),
          shape: [72, 8 + i * 72],
        };
        weights[`chi_offset_prediction_layers.${i}.0.weight`] = {
          data: new Float32Array(144).fill(0.02),
          shape: [1, 144],
        };
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const b: Record<string, any> = {
        weights,
        mark: () => 0,
        releaseSince: () => {},
        from: (data: ArrayLike<number>, r: number, c: number) => ({ data: Float32Array.from(data), r, c }),
        zeros: (r: number, c: number) => ({ data: new Float32Array(r * c), r, c }),
        weight: (k: string) => weights[k],
        gather: (x: { data: Float32Array; c?: number; shape?: number[] }, idx: number[] | { data: Float32Array }) => {
          const arr = Array.isArray(idx) ? idx : Array.from(idx.data, (v) => Math.round(v));
          const cols = x.c ?? x.shape![1];
          const out = new Float32Array(arr.length * cols);
          for (let r = 0; r < arr.length; r++) out.set(x.data.subarray(arr[r] * cols, (arr[r] + 1) * cols), r * cols);
          return { data: out, r: arr.length, c: cols };
        },
        cat: (xs: Array<{ data: Float32Array; r: number; c: number }>) => {
          const rows = xs[0]?.r ?? 0;
          const totalC = xs.reduce((s, x) => s + x.c, 0);
          const out = new Float32Array(rows * totalC);
          for (let r = 0; r < rows; r++) {
            let off = 0;
            for (const x of xs) {
              out.set(x.data.subarray(r * x.c, (r + 1) * x.c), r * totalC + off);
              off += x.c;
            }
          }
          return { data: out, r: rows, c: totalC };
        },
        rows: (xs: Array<{ data: Float32Array; r: number; c: number }>) => {
          const cols = xs[0].c;
          const totalR = xs.reduce((s, x) => s + x.r, 0);
          const out = new Float32Array(totalR * cols);
          let off = 0;
          for (const x of xs) {
            out.set(x.data, off);
            off += x.data.length;
          }
          return { data: out, r: totalR, c: cols };
        },
        assign: (x: { data: Float32Array; c: number }, row: number, y: { data: Float32Array }) => {
          x.data.set(y.data, row * x.c);
        },
        linear: (x: { data: Float32Array; r: number; c: number }, key: string) => {
          const w = weights[key + '.weight'];
          const out = new Float32Array(x.r * w.shape[0]);
          for (let r = 0; r < x.r; r++)
            for (let oc = 0; oc < w.shape[0]; oc++) {
              let s = 0;
              for (let k = 0; k < x.c; k++) s += x.data[r * x.c + k] * w.data[oc * x.c + k];
              out[r * w.shape[0] + oc] = s;
            }
          return { data: out, r: x.r, c: w.shape[0] };
        },
        read: async (x: { data: Float32Array }) => Float32Array.from(x.data),
      };
      if (withGpuOps) {
        b.sampleIndex = () => {
          throw new Error('sampleIndex should not run when opts.choices is supplied');
        };
        b.catOneHot = (logits: { data: Float32Array; c: number }, choice: { data: Float32Array }) => {
          const out = new Float32Array(logits.c + 72);
          out.set(logits.data, 0);
          out[logits.c + Math.round(choice.data[0])] = 1;
          return { data: out, r: 1, c: logits.c + 72 };
        };
        b.chiAngle = (choice: { data: Float32Array }, offset: { data: Float32Array }) => {
          const raw = -180 + Math.round(choice.data[0]) * 5 + offset.data[0];
          const s = raw + 180;
          const m = ((s % 360) + 360) % 360;
          return { data: new Float32Array([Math.fround(m - 180)]), r: 1, c: 1 };
        };
        b.chiRbf = (ang: { data: Float32Array }) => {
          const a = Array.from({ length: 72 }, (_, j) => {
            const c = -180 + j * 5;
            const d1 = (((c - ang.data[0]) % 360) + 360) % 360;
            const d2 = (((ang.data[0] - c) % 360) + 360) % 360;
            return Math.exp(-0.5 * (Math.min(d1, d2) / 2.5) ** 2);
          });
          const s = a.reduce((x, y) => x + y, 0);
          return { data: Float32Array.from(a, (v) => v / s), r: 1, c: 72 };
        };
        b.maskRow = (row: { data: Float32Array; r: number; c: number }, mask: { data: Float32Array }, i: number) => ({
          data: row.data.map((v) => v * mask.data[i]),
          r: row.r,
          c: row.c,
        });
        b.readMany = async (xs: Array<{ data: Float32Array }>) => xs.map((x) => Float32Array.from(x.data));
        b.sync = async () => {};
      }
      return b;
    };
    const buildModel = (b: ReturnType<typeof makeChoiceBackend>) => {
      const m = new LaserModel(b, {
        alphabet: 'ACDEFGHIKLMNPQRSTVWYX',
        model: { node_embedding_dim: 4, num_laser_vectors: 4 },
      });
      m.encode = async (batch: { length: number }) => ({
        lig: { s: b.zeros(0, 4) },
        prot: { s: b.zeros(batch.length, 4), v: b.zeros(batch.length, 12) },
        pe: b.zeros(0, 4),
        ple: b.zeros(0, 4),
        traces: {},
      });
      m.hetero = () => ({ s: b.zeros(1, 4), v: b.zeros(1, 12) });
      m.mlp = (x: { data: Float32Array; r: number; c: number }, prefix: string) => b.linear(x, prefix + '.0');
      m.gvp = () => ({ s: b.zeros(1, 4), v: b.zeros(1, 12) });
      m.eqnorm = (x: unknown) => x;
      return m;
    };
    const choiceBatch = {
      length: 6,
      pp: [[], []],
      lp: [[], []],
      fixed: new Int32Array([1, 1, 0, 0, 0, 1]),
      sequence: new Int32Array([1, 7, 2, 3, 4, 0]),
      chi: new Float32Array(24).fill(NaN),
    };
    const choices = [
      9, 0, 1, 2, 3,
      10, 4, 5, 6, 7,
      15, 8, 9, 10, 11,
      16, 12, 13, 14, 15,
      17, 16, 17, 18, 19,
      18, 20, 21, 22, 23,
    ];
    const decOrder = [0, 1, 2, 3, 4, 5];
    const rChoiceLegacy = await buildModel(makeChoiceBackend(false)).run(choiceBatch, {
      choices,
      decodingOrder: decOrder,
    });
    const rChoiceGpu = await buildModel(makeChoiceBackend(true)).run(choiceBatch, {
      choices,
      decodingOrder: decOrder,
    });
    expect(Array.from(rChoiceGpu.sequence)).toEqual(Array.from(rChoiceLegacy.sequence));
    expect(Array.from(rChoiceGpu.chiBins)).toEqual(Array.from(rChoiceLegacy.chiBins));

    // 4. LASErMPNN WGSL sampleIndex tracks lastValid instead of k - 1u
    const laserShaderSrc = readFileSync(
      resolve(process.cwd(), 'static/vendor/lasermpnn/webgpu/lasermpnn/shaders.js'),
      'utf8',
    );
    expect(laserShaderSrc).toContain('var choice = lastValid;');
    expect(laserShaderSrc).not.toContain('var choice = k - 1u;');

    // 5. BoltzGen multi-char authChainId range hotspots expansion
    const orderedMultiChar: OrderedStructureResidue[] = [
      { chain: 'HL', number: 54, insertionCode: '' },
      { chain: 'HL', number: 55, insertionCode: '' },
      { chain: 'HL', number: 56, insertionCode: '' },
      { chain: 'HL', number: 100, insertionCode: 'A' },
      { chain: 'HL', number: 100, insertionCode: 'B' },
      { chain: 'HL', number: 100, insertionCode: 'C' },
    ];
    const multiHotspots = parseFixedPositionsAndChains('HL:54-56, HL:100A-100C', 'HL', orderedMultiChar);
    expect(multiHotspots.fixedPositions).toEqual([
      { chain: 'HL', number: 54 },
      { chain: 'HL', number: 55 },
      { chain: 'HL', number: 56 },
      { chain: 'HL', number: 100, insertionCode: 'A' },
      { chain: 'HL', number: 100, insertionCode: 'B' },
      { chain: 'HL', number: 100, insertionCode: 'C' },
    ]);
    const bgWorkerSrc = readFileSync(resolve(process.cwd(), 'src/lib/plugins/boltzgen/boltzgen.worker.ts'), 'utf8');
    expect(bgWorkerSrc).toContain('chain: sr.authChainId');

    // 6. AntiFold regionMask & input-preview support end-bound insertion codes ('H:100A-100C')
    const antiStruct = await import(vendorUrl('static/vendor/antifold-esmif1/shared/structure.js'));
    const insResidues = [
      { chain: 'H', number: 99, insertion: '' },
      { chain: 'H', number: 100, insertion: '' },
      { chain: 'H', number: 100, insertion: 'A' },
      { chain: 'H', number: 100, insertion: 'B' },
      { chain: 'H', number: 100, insertion: 'C' },
      { chain: 'H', number: 100, insertion: 'D' },
    ];
    expect([
      ...antiStruct.regionMask(insResidues, { heavyChain: 'H', regions: ['H:100A-100C'] }),
    ]).toEqual([0, 0, 1, 1, 1, 0]);
    expect([
      ...antiStruct.regionMask(insResidues, { heavyChain: 'H', regions: ['H:99-100B'] }),
    ]).toEqual([1, 1, 1, 1, 0, 0]);
    expect(() =>
      antiStruct.regionMask(insResidues, { heavyChain: 'H', regions: ['H:100C-100A'] }),
    ).toThrow(/Reversed/);

    const pdbInsSpan = makeLinearPdb([
      { chain: 'H', resNums: [22, 36, 92, 100, 100, 100, 100, 103], insCodes: ['', '', '', '', 'A', 'B', 'C', ''] },
      { chain: 'L', resNums: [23, 35, 88, 98] },
    ]);
    const prevInsSpan = compileNodeInputPreview(
      {
        id: 'g-ins-span',
        name: 'Ins Span',
        nodes: [
          { id: 'pdb', adapterId: 'pdb-input', label: 'Ab', params: { mode: 'custom', pdbText: pdbInsSpan } },
          {
            id: 'af',
            adapterId: 'antifold-esmif1',
            label: 'AntiFold',
            params: { ...defaultParamsForAdapter('antifold-esmif1'), targetChain: 'H', regions: 'H:100A-100B' },
          },
        ],
        edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
      },
      'af',
      {},
    );
    expect(prevInsSpan.issues.some((i) => /Invalid AntiFold/i.test(i.message))).toBe(false);
    expect(
      prevInsSpan.chains
        .find((c) => c.chainId === 'H')!
        .residues.filter((r) => r.roles.includes('designable'))
        .map((r) => `${r.authSeqId}${r.insCode}`),
    ).toEqual(['100A', '100B']);
  });
});
