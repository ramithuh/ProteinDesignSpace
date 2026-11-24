import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { getBenchmarkById } from '$lib/engine/benchmarks';
import {
  BENCHMARK_SYNC_ADAPTER_IDS,
  applyPlannerBenchmarkSeedingToNode,
  buildResetToBenchmarkParams,
  isBenchmarkSupportedForAdapter,
  reconcileBenchmarkSeeding,
  switchBenchmark,
} from '$lib/engine/benchmark-sync';
import { extractUpstreamHoloHints, validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import type { SequenceSet, StructureArtifact, WorkflowGraph, WorkflowNode } from '$lib/engine/types';
import { PluginRegistry } from '$lib/kernel';
import { proteinBiopolymerDomainPack } from '$lib/domains/protein-biopolymer';
import { rfd3Plugin } from '$lib/plugins/rfd3';
import {
  buildAutoFlankContigFromResidues,
  buildFullProteinContigFromResidues,
  computeRfd3MotifMeta,
  drainGpuDeviceQueue,
  isReusableRfd3Engine,
  normalizeRfd3Contig,
  resolveRfd3UpstreamStructure,
  trimRfd3EngineSafely,
} from '$lib/plugins/rfd3/lifecycle';
import { materializePdbInputNode } from '$lib/plugins/pdb-input';

const MINI_1CTT_CIF = [
  'data_1CTT',
  '#',
  'loop_',
  '_chem_comp_bond.comp_id',
  '_chem_comp_bond.atom_id_1',
  '_chem_comp_bond.atom_id_2',
  '_chem_comp_bond.value_order',
  'DHZ C1 N2 SING',
  'DHZ N2 C3 DOUB',
  '#',
  'loop_',
  '_atom_site.group_PDB',
  '_atom_site.id',
  '_atom_site.type_symbol',
  '_atom_site.label_atom_id',
  '_atom_site.label_alt_id',
  '_atom_site.label_comp_id',
  '_atom_site.label_asym_id',
  '_atom_site.label_entity_id',
  '_atom_site.label_seq_id',
  '_atom_site.pdbx_PDB_ins_code',
  '_atom_site.Cartn_x',
  '_atom_site.Cartn_y',
  '_atom_site.Cartn_z',
  '_atom_site.occupancy',
  '_atom_site.B_iso_or_equiv',
  '_atom_site.pdbx_formal_charge',
  '_atom_site.auth_seq_id',
  '_atom_site.auth_comp_id',
  '_atom_site.auth_asym_id',
  '_atom_site.auth_atom_id',
  '_atom_site.pdbx_PDB_model_num',
  'ATOM   1  N  N   . HIS A 1 1   ? 0.000  0.000  0.000  1.00 20.00 ? 102 HIS A N   1',
  'ATOM   2  C  CA  . HIS A 1 1   ? 1.458  0.000  0.000  1.00 20.00 ? 102 HIS A CA  1',
  'ATOM   3  C  C   . HIS A 1 1   ? 2.009  1.420  0.000  1.00 20.00 ? 102 HIS A C   1',
  'ATOM   4  O  O   . HIS A 1 1   ? 1.251  2.390  0.000  1.00 20.00 ? 102 HIS A O   1',
  'ATOM   5  C  CB  . HIS A 1 1   ? 1.988 -0.773 -1.208  1.00 20.00 ? 102 HIS A CB  1',
  'ATOM   6  C  CG  . HIS A 1 1   ? 3.450 -0.950 -1.250  1.00 20.00 ? 102 HIS A CG  1',
  'ATOM   7  N  ND1 . HIS A 1 1   ? 4.250 -0.100 -1.950  1.00 20.00 ? 102 HIS A ND1 1',
  'ATOM   8  C  CD2 . HIS A 1 1   ? 4.220 -1.880 -0.650  1.00 20.00 ? 102 HIS A CD2 1',
  'ATOM   9  C  CE1 . HIS A 1 1   ? 5.500 -0.500 -1.780  1.00 20.00 ? 102 HIS A CE1 1',
  'ATOM   10 N  NE2 . HIS A 1 1   ? 5.520 -1.580 -0.990  1.00 20.00 ? 102 HIS A NE2 1',
  'ATOM   11 N  N   . GLU A 1 2   ? 3.300  1.500  0.000  1.00 20.00 ? 103 GLU A N   1',
  'ATOM   12 C  CA  . GLU A 1 2   ? 4.000  2.800  0.000  1.00 20.00 ? 103 GLU A CA  1',
  'ATOM   13 C  C   . GLU A 1 2   ? 5.500  2.600  0.000  1.00 20.00 ? 103 GLU A C   1',
  'ATOM   14 O  O   . GLU A 1 2   ? 6.000  1.500  0.000  1.00 20.00 ? 103 GLU A O   1',
  'ATOM   15 N  N   . CYS A 1 3   ? 6.100  3.800  0.000  1.00 20.00 ? 104 CYS A N   1',
  'ATOM   16 C  CA  . CYS A 1 3   ? 7.550  3.950  0.000  1.00 20.00 ? 104 CYS A CA  1',
  'ATOM   17 C  C   . CYS A 1 3   ? 8.100  5.380  0.000  1.00 20.00 ? 104 CYS A C   1',
  'ATOM   18 O  O   . CYS A 1 3   ? 7.340  6.350  0.000  1.00 20.00 ? 104 CYS A O   1',
  'HETATM 19 ZN ZN  . ZN  B 2 .   ? 5.000  0.500 -2.800  1.00 15.00 2 296 ZN  A ZN  1',
  'HETATM 20 C  C1  . DHZ C 3 .   ? 6.500  0.500 -3.500  1.00 15.00 ? 295 DHZ A C1  1',
  'HETATM 21 N  N2  . DHZ C 3 .   ? 9.800  0.500 -3.500  1.00 15.00 ? 295 DHZ A N2  1',
  'HETATM 22 C  C3  . DHZ C 3 .   ? 9.800  3.900 -3.500  1.00 15.00 ? 295 DHZ A C3  1',
  '#',
  '',
].join('\n');

describe('Issue #98 — RFD3 GPU lifecycle, R1 mmCIF-first + motifMeta, R2 benchmark sync, R3 holo hints & exclusive VRAM lease', () => {
  it('guards stale closed engines, trims scratch pools safely, and awaits GPU queue drain before device destroy', async () => {
    const liveEngine = { resources: { closed: false }, trim: vi.fn() };
    const closedEngine = { resources: { closed: true }, trim: vi.fn() };

    expect(isReusableRfd3Engine(liveEngine, 'fp16', 'fp16')).toBe(true);
    expect(isReusableRfd3Engine(liveEngine, 'fp16', 'int8')).toBe(false);
    expect(isReusableRfd3Engine(closedEngine, 'fp16', 'fp16')).toBe(false);
    expect(isReusableRfd3Engine(null, 'fp16', 'fp16')).toBe(false);

    expect(trimRfd3EngineSafely(liveEngine)).toBe(true);
    expect(liveEngine.trim).toHaveBeenCalledTimes(1);
    expect(trimRfd3EngineSafely(closedEngine)).toBe(false);
    expect(closedEngine.trim).not.toHaveBeenCalled();

    let settled = false;
    await drainGpuDeviceQueue({
      queue: {
        onSubmittedWorkDone: async () => {
          await new Promise((r) => setTimeout(r, 15));
          settled = true;
        },
      },
    });
    expect(settled).toBe(true);

    // Bounded timeout prevents hanging if a lost GPUDevice never resolves onSubmittedWorkDone
    const t0 = Date.now();
    await drainGpuDeviceQueue(
      {
        queue: {
          onSubmittedWorkDone: () => new Promise(() => {}),
        },
      },
      25,
    );
    expect(Date.now() - t0).toBeLessThan(250);

    // Verify rfd3.worker.ts wires all three guards in production source
    const workerSrc = readFileSync(
      resolve(process.cwd(), 'src/lib/plugins/rfd3/rfd3.worker.ts'),
      'utf8',
    );
    expect(workerSrc).toContain('!engine.resources?.closed');
    expect(workerSrc).toContain('trimRfd3EngineSafely(eng)');
    expect(workerSrc).toContain('await drainGpuDeviceQueue(ownedDevice');
  });

  it('surfaces GPUDeviceLostInfo reason & message, caps GPUOps scratch pool, chunks encoders, and skips intermediate sequence readbacks', async () => {
    const vendorDir = resolve(process.cwd(), 'static/vendor/rfd3');
    const { createGpuResources } = await import(
      pathToFileURL(resolve(vendorDir, 'webgpu/resources.js')).href
    );
    const { GPUOps } = await import(pathToFileURL(resolve(vendorDir, 'webgpu/ops.js')).href);
    const { sampleCPU } = await import(
      pathToFileURL(resolve(vendorDir, 'shared/sampler.js')).href
    );
    const { createMonomerFeatures } = await import(
      pathToFileURL(resolve(vendorDir, 'shared/features.js')).href
    );

    let loseDevice!: (info: { reason: string; message: string }) => void;
    const lostPromise = new Promise<{ reason: string; message: string }>((res) => {
      loseDevice = res;
    });
    const submittedBuffers: unknown[] = [];
    let workDoneCount = 0;
    const mockDevice = {
      adapterInfo: { vendor: 'apple', architecture: 'metal-3', isFallbackAdapter: false },
      lost: lostPromise,
      limits: {
        maxBufferSize: 512 * 1024 * 1024,
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        minStorageBufferOffsetAlignment: 16,
      },
      createBuffer: (desc: { size: number; usage: number; label?: string }) => ({
        ...desc,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      }),
      createShaderModule: ({ code }: { code: string }) => ({ code }),
      createComputePipeline: ({ compute }: { compute: { module: { code: string } } }) => ({
        code: compute.module.code,
        getBindGroupLayout: () => ({}),
      }),
      createBindGroup: () => ({}),
      createCommandEncoder: () => {
        const passes: number[] = [];
        return {
          beginComputePass: () => ({
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {
              passes.push(1);
            },
            end() {},
          }),
          finish: () => ({ dispatches: passes.length }),
        };
      },
      queue: {
        writeBuffer() {},
        submit(cmds: unknown[]) {
          submittedBuffers.push(...cmds);
        },
        onSubmittedWorkDone: async () => {
          workDoneCount++;
        },
      },
      destroy() {},
    };

    const resources = await createGpuResources({ device: mockDevice });
    const ops = new GPUOps(resources, { maxPoolBytes: 1024 });

    // Pool cap & encoder chunking via ops.flush() without calling onSubmittedWorkDone()
    ops.begin();
    const tA = ops.tensor(128); // 512 B
    const tB = ops.tensor(128); // 512 B
    const tC = ops.tensor(128); // 512 B (exceeds 1024 B pool cap when all 3 freed)
    const kOut = ops.kernel([tA, tB], 128, 'out[idx]=a0[idx]+a1[idx];');
    ops.flush();
    expect(submittedBuffers.length).toBe(1);
    expect(workDoneCount).toBe(0);
    ops.free(tA);
    ops.free(tB);
    ops.free(tC);
    ops.free(kOut);
    await ops.submit();
    expect(workDoneCount).toBe(1);
    expect(ops.poolBytes).toBeLessThanOrEqual(1024);
    ops.trim();
    expect(resources.memoryUsage()).toBe(0);

    // Device loss surfaces both reason and driver message instead of generic string
    loseDevice({ reason: 'destroyed', message: 'Command buffer execution failed: Timeout' });
    await resources.lost;
    expect(() => resources.allocate(64, 128)).toThrow(
      /GPU resources destroyed or device lost \(reason=destroyed: Command buffer execution failed: Timeout\)/,
    );

    // Sampler passes coordinatesOnly: true on intermediate steps and false on final step
    const feat = createMonomerFeatures({ length: 2 });
    const coordinatesOnlyFlags: boolean[] = [];
    const mockEngine = {
      initialize: () => ({}),
      run: ({ X_noisy_L, coordinatesOnly }: { X_noisy_L: Float32Array; coordinatesOnly?: boolean }) => {
        coordinatesOnlyFlags.push(Boolean(coordinatesOnly));
        if (coordinatesOnly) {
          return { X_L: { shape: [1, 28, 3], data: new Float32Array(X_noisy_L) } };
        }
        return {
          X_L: { shape: [1, 28, 3], data: new Float32Array(X_noisy_L) },
          sequence_logits_I: { shape: [1, 2, 32], data: new Float32Array(64) },
          sequence_indices_I: { shape: [1, 2], data: new Int32Array([0, 1]) },
        };
      },
    };
    const sampled = sampleCPU(mockEngine, { features: feat, steps: 5, seed: 7, retainTrajectory: false });
    expect(coordinatesOnlyFlags).toEqual([true, true, true, false]);
    expect(sampled.sequence_indices_I.data).toEqual(new Int32Array([0, 1]));
    expect(sampled.sequence_entropy_traj).toHaveLength(1);
  });

  it('R1: prefers upstream cifText over synthesized pdbText (preserving ZN +2 charge & DHZ _chem_comp_bond) and populates motifMeta for both mmCIF and PDB', async () => {
    const cifArtifact = await materializePdbInputNode({
      id: 'pdb1',
      adapterId: 'pdb-input',
      params: { mode: 'custom', pdbText: MINI_1CTT_CIF, filename: '1CTT.cif' },
    });
    expect(cifArtifact.cifText).toBeTruthy();
    expect(cifArtifact.pdbText).toBeTruthy(); // synthesized PDB lacks formal charge & _chem_comp_bond

    const resolved = resolveRfd3UpstreamStructure(cifArtifact, true, '');
    expect(resolved.structureFormat).toBe('cif');
    expect(resolved.structureText).toBe(cifArtifact.cifText);
    expect(resolved.fallbackResKeys).toEqual(['A102', 'A103', 'A104']);
    expect(resolved.autoLigandNames).toEqual(expect.arrayContaining(['ZN', 'DHZ']));

    // Verify vendored buildInput preserves ZN +2 formal charge and DHZ _chem_comp_bond ONLY when fed cifText!
    // Note: C1-N2 and N2-C3 in MINI_1CTT_CIF are 3.3-3.4 Å apart (> covalent cutoff), so bonds come strictly from _chem_comp_bond!
    const { buildInput } = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/rfd3/shared/input.js')).href
    );
    const builtFromCif = buildInput({
      mode: 'motif',
      length: 16,
      structure: resolved.structureText,
      format: resolved.structureFormat,
      unindex: 'A102,A104',
      fixedAtoms: { A102: 'ND1,CG,CE1', A104: 'CA,C' },
      ligand: 'ZN,DHZ',
    });
    const znAtomIdx = builtFromCif.atoms.findIndex((a: { resname?: string }) => a.resname === 'ZN');
    expect(znAtomIdx).toBeGreaterThanOrEqual(0);
    expect(builtFromCif.features.ref_charge[znAtomIdx]).toBe(2);

    const dhzTokens = builtFromCif.tokens
      .map((t: { resname?: string }, idx: number) => ({ resname: t.resname, idx }))
      .filter((t: { resname?: string }) => t.resname === 'DHZ');
    expect(dhzTokens).toHaveLength(3);
    const nTok = builtFromCif.tokens.length;
    expect(builtFromCif.features.token_bonds[dhzTokens[0].idx * nTok + dhzTokens[1].idx]).toBe(1);
    expect(builtFromCif.features.token_bonds[dhzTokens[1].idx * nTok + dhzTokens[2].idx]).toBe(1);
    expect(builtFromCif.features.token_bonds[dhzTokens[0].idx * nTok + dhzTokens[2].idx]).toBe(0);

    // Auto-contigs & motifMeta work identically on mmCIF and PDB inputs
    expect(buildAutoFlankContigFromResidues(resolved.caResidues, 23)).toBe('10,A102-104,10');
    expect(buildFullProteinContigFromResidues(resolved.caResidues)).toBe('A102-104');
    expect(normalizeRfd3Contig('25-25, A102-104, 25-25')).toBe('25,A102-104,25');

    const unindexedMeta = computeRfd3MotifMeta({
      mode: 'motif',
      caResidues: resolved.caResidues,
      resolvedLen: 20,
      motifAssignments: [
        { source: 'A102', token: 4 },
        { source: 'A104', token: 11 },
      ],
    });
    expect(unindexedMeta?.motifIndices).toEqual([4, 11]);
    expect(unindexedMeta?.fixedResidueIndices).toEqual([5, 12]);
    expect(unindexedMeta?.nativeMotifCoords).toHaveLength(2);
    expect(unindexedMeta?.nativeMotifCoords?.[0][0]).toBeCloseTo(1.458, 5);
    expect(unindexedMeta?.nativeMotifCoords?.[1][0]).toBeCloseTo(7.55, 5);
    expect(unindexedMeta?.nativeMotifCoords?.[1][1]).toBeCloseTo(3.95, 5);

    const indexedFeatures = {
      is_protein: new Float32Array([1, 1, 1, 1, 1]),
      ref_motif_token_type: new Float32Array([
        1, 0, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        1, 0, 0,
      ]),
    };
    const indexedMeta = computeRfd3MotifMeta({
      mode: 'motif',
      caResidues: resolved.caResidues,
      resolvedLen: 5,
      effectiveContig: '1,A102-104,1',
      features: indexedFeatures,
    });
    expect(indexedMeta?.motifIndices).toEqual([1, 2, 3]);
    expect(indexedMeta?.fixedResidueIndices).toEqual([2, 3, 4]);
    expect(indexedMeta?.nativeMotifCoords).toHaveLength(3);
    expect(indexedMeta?.nativeMotifCoords?.[0][0]).toBeCloseTo(1.458, 5);
    expect(indexedMeta?.nativeMotifCoords?.[1][0]).toBeCloseTo(4.0, 5);
    expect(indexedMeta?.nativeMotifCoords?.[1][1]).toBeCloseTo(2.8, 5);
    expect(indexedMeta?.nativeMotifCoords?.[2][0]).toBeCloseTo(7.55, 5);
  });

  it('R2: auto-seeds, adapts length, resets, and symmetrically tears down rfd3 across enzyme-ligand, motif, and binder benchmarks', () => {
    expect(BENCHMARK_SYNC_ADAPTER_IDS.has('rfd3')).toBe(true);
    const ctt = getBenchmarkById('ame_1ctt_cda')!;
    const prw = getBenchmarkById('rfd_1prw_efhand')!;
    const pdl1 = getBenchmarkById('pdl1_5o45')!;
    expect(isBenchmarkSupportedForAdapter('rfd3', ctt)).toBe(true);
    expect(isBenchmarkSupportedForAdapter('rfd3', prw)).toBe(true);
    expect(isBenchmarkSupportedForAdapter('rfd3', pdl1)).toBe(true);

    const baseGraph: WorkflowGraph = {
      id: 'rfd3-bench-sync',
      name: 'RFD3 Benchmark Sync',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '1CTT', benchmarkId: 'ame_1ctt_cda', cropSegments: ctt.cropSegments },
        },
        {
          id: 'rfd3',
          adapterId: 'rfd3',
          params: { ...defaultParamsForAdapter('rfd3') },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd3', toPort: 'structure' }],
    };

    // 1. Seed 1CTT enzyme-ligand unindexed theozyme
    const seededCtt = reconcileBenchmarkSeeding(baseGraph);
    const r3Ctt = seededCtt.graph.nodes.find((n) => n.id === 'rfd3')!;
    expect(r3Ctt.params.mode).toBe('motif');
    expect(r3Ctt.params.length).toBe(180);
    expect(r3Ctt.params.unindex).toBe('A102, A104, A129, A132');
    expect(r3Ctt.params.fixedAtoms).toBe(
      'A102:ND1,CG,CE1; A104:OE1,CD; A129:SG,CB; A132:SG,CB,CA',
    );
    expect(r3Ctt.params.ligand).toBe('ZN, DHZ');
    expect(r3Ctt.params.autoSeededBenchmarkId).toBe('ame_1ctt_cda');

    // 2. Switch to 1PRW indexed motif -> concrete RFD3 contig, adapts on length slider edit
    const switchedPrw = switchBenchmark(seededCtt.graph, 'pdb', 'rfd_1prw_efhand');
    const r3Prw = switchedPrw.graph.nodes.find((n) => n.id === 'rfd3')!;
    expect(r3Prw.params.mode).toBe('motif');
    expect(r3Prw.params.length).toBe(76);
    expect(r3Prw.params.contig).toBe('10,A16-35,16,A52-71,10');
    expect(r3Prw.params.unindex).toBe('');
    expect(r3Prw.params.ligand).toBe('');

    const resizedGraph: WorkflowGraph = {
      ...switchedPrw.graph,
      nodes: switchedPrw.graph.nodes.map((n) =>
        n.id === 'rfd3' ? { ...n, params: { ...n.params, length: 90, userEditedLength: true } } : n,
      ),
    };
    const adaptedPrw = reconcileBenchmarkSeeding(resizedGraph);
    expect(adaptedPrw.graph.nodes.find((n) => n.id === 'rfd3')!.params.contig).toBe(
      '13,A16-35,23,A52-71,14',
    );

    // 3. Switch to 5O45 binder -> mode: 'binder', target: 'A17-132', hotspots
    const switchedPdl1 = switchBenchmark(seededCtt.graph, 'pdb', 'pdl1_5o45');
    const r3Pdl1 = switchedPdl1.graph.nodes.find((n) => n.id === 'rfd3')!;
    expect(r3Pdl1.params.mode).toBe('binder');
    expect(r3Pdl1.params.length).toBe(70);
    expect(r3Pdl1.params.target).toBe('A17-132');
    expect(r3Pdl1.params.hotspots).toBe('A54, A56, A66, A115');

    // 4. Reset helper & symmetric teardown
    const reset = buildResetToBenchmarkParams('rfd3', {
      ...r3Ctt.params,
      userEditedTask: true,
      userEditedLength: true,
    });
    expect(reset.mode).toBe('unconditional');
    expect(reset.length).toBe(80);
    expect(reset.unindex).toBe('');
    expect(reset.fixedAtoms).toBe('');
    expect(reset.ligand).toBe('');
    expect(reset.userEditedTask).toBeUndefined();

    const cleared = switchBenchmark(switchedPdl1.graph, 'pdb', null);
    const r3Cleared = cleared.graph.nodes.find((n) => n.id === 'rfd3')!;
    expect(r3Cleared.params.mode).toBe('unconditional');
    expect(r3Cleared.params.length).toBe(80);
    expect(r3Cleared.params.autoSeededBenchmarkId).toBeUndefined();

    // 5. Planner hydrator repairs hallucinated unindex on rfd3
    const plannedNode: WorkflowNode = {
      id: 'gen',
      adapterId: 'rfd3',
      params: { mode: 'motif', unindex: 'A1-5' },
    };
    const plannerRes = applyPlannerBenchmarkSeedingToNode(plannedNode, ctt);
    expect(plannerRes.repairedHallucination).toBe(true);
    expect(plannedNode.params.unindex).toBe('A102, A104, A129, A132');
    expect(plannedNode.params.ligand).toBe('ZN, DHZ');
  });

  it('R3: honors rfd3 in extractUpstreamHoloHints, isMotifNode disjoint crop check, and full-backbone error guidance', () => {
    const baseNodes: WorkflowNode[] = [
      {
        id: 'pdb',
        adapterId: 'pdb-input',
        params: {
          mode: 'pdb-id',
          pdbId: '1CTT',
          benchmarkId: 'ame_1ctt_cda',
          cropSegments: 'A102, A104, A129, A132, A:ZN, A:DHZ',
        },
      },
      {
        id: 'rfd3',
        adapterId: 'rfd3',
        params: { ...defaultParamsForAdapter('rfd3'), mode: 'unconditional' },
      },
      {
        id: 'af3',
        adapterId: 'localfold-af2',
        params: { ...defaultParamsForAdapter('localfold-af2'), model: 'af3' },
      },
    ];
    const edges = [
      { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd3', toPort: 'structure' },
      { fromNode: 'rfd3', fromPort: 'sequences', toNode: 'af3', toPort: 'sequences' },
    ];

    // Unconditional rfd3 blocks upstream template ligands from leaking downstream
    const uncondHints = extractUpstreamHoloHints(
      { id: 'h1', name: 'H1', nodes: baseNodes, edges },
      'af3',
    );
    expect(uncondHints.ligandCcdCodes).toEqual([]);

    // Conditioned rfd3 propagates explicit params.ligand + upstream/live holo ligands
    const condNodes = baseNodes.map((n) =>
      n.id === 'rfd3'
        ? { ...n, params: { ...n.params, mode: 'motif', ligand: 'ATP' } }
        : n,
    );
    const condHints = extractUpstreamHoloHints(
      { id: 'h2', name: 'H2', nodes: condNodes, edges },
      'af3',
      {
        rfd3: {
          structure: {
            kind: 'StructureArtifact',
            id: 'live-rfd3',
            label: 'RFD3',
            sourceModelId: 'rfd3/fp16',
            length: 40,
            sequence: 'A'.repeat(40),
            caCoords: new Float32Array(120),
            backbone4Coords: new Float32Array(480),
            pdbText: 'END\n',
            diagnostics: {
              chainCount: 1,
              chainLengths: [40],
              hasLigands: true,
              ligandNames: ['MG'],
              hasNucleicAcids: false,
              nucleicAcidChains: [],
            },
            geometry: {
              length: 40,
              rg: 10,
              meanCaDist: 3.8,
              chainBreaks: 0,
              clashes: 0,
              helixPct: 50,
              sheetPct: 0,
              coilPct: 50,
            },
          } satisfies StructureArtifact,
        },
      },
    );
    expect(condHints.ligandCcdCodes).toEqual(expect.arrayContaining(['ATP', 'MG', 'ZN', 'DHZ']));

    // Disjoint unindex on rfd3 motif mode emits motif-span-disjoint-from-crop with 1-click fix
    const disjointGraph: WorkflowGraph = {
      id: 'disjoint-rfd3',
      name: 'Disjoint RFD3',
      nodes: [
        baseNodes[0],
        {
          id: 'rfd3',
          adapterId: 'rfd3',
          label: 'RFdiffusion3',
          params: { ...defaultParamsForAdapter('rfd3'), mode: 'motif', unindex: 'A1-10' },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd3', toPort: 'structure' }],
    };
    const disjointIssues = validateBioCapabilities(disjointGraph);
    const cropWarn = disjointIssues.find((i) => i.code === 'motif-span-disjoint-from-crop');
    expect(cropWarn).toBeDefined();
    expect(cropWarn?.fix?.paramsPatch).toEqual({ unindex: 'A102, A104, A129, A132' });

    // Full-backbone guidance mentions RFdiffusion3
    const caOnlyIssues = validateBioCapabilities({
      id: 'ca-check',
      name: 'CA Check',
      nodes: [
        { id: 'g3', adapterId: 'genie3', params: defaultParamsForAdapter('genie3') },
        { id: 'lmpnn', adapterId: 'ligandmpnn', params: defaultParamsForAdapter('ligandmpnn') },
      ],
      edges: [{ fromNode: 'g3', fromPort: 'structure', toNode: 'lmpnn', toPort: 'structure' }],
    });
    const caErr = caOnlyIssues.find((i) => i.code === 'mpnn-ca-only-backbone');
    expect(caErr?.message).toContain('RFdiffusion3');
  });

  it('immediately disposes exclusive-lease GPU workers upon node completion and honors 2000 ms dispose queue-drain grace window', async () => {
    let liveWorkers = 0;
    let disposeMessages = 0;
    let disposedCleanlyBeforeTerminate = false;

    const reg = new PluginRegistry();
    reg.registerDomainPack(proteinBiopolymerDomainPack);
    reg.registerPlugin({
      ...rfd3Plugin,
      execution: {
        ...rfd3Plugin.execution,
        vramLeaseMode: 'exclusive',
        createWorker: () => {
          liveWorkers++;
          let terminated = false;
          const listeners = new Map<string, Set<(ev: MessageEvent) => void>>();
          return {
            addEventListener(type: string, cb: (ev: MessageEvent) => void) {
              if (!listeners.has(type)) listeners.set(type, new Set());
              listeners.get(type)!.add(cb);
            },
            removeEventListener(type: string, cb: (ev: MessageEvent) => void) {
              listeners.get(type)?.delete(cb);
            },
            postMessage(msg: { id: number; type: string }) {
              if (msg.type === 'run') {
                const structure: StructureArtifact = {
                  kind: 'StructureArtifact',
                  id: 'rfd3-test-out',
                  label: 'RFD3 Out',
                  sourceModelId: 'rfd3/fp16',
                  length: 16,
                  sequence: 'A'.repeat(16),
                  caCoords: new Float32Array(48),
                  backbone4Coords: new Float32Array(192),
                  pdbText: 'END\n',
                  geometry: {
                    length: 16,
                    rg: 8,
                    meanCaDist: 3.8,
                    chainBreaks: 0,
                    clashes: 0,
                    helixPct: 50,
                    sheetPct: 0,
                    coilPct: 50,
                  },
                };
                const sequences: SequenceSet = {
                  kind: 'SequenceSet',
                  id: 'rfd3-test-seq',
                  parentStructureId: structure.id,
                  sourceStructureId: structure.id,
                  sourceModelId: 'rfd3/fp16',
                  candidates: [
                    {
                      id: 's1',
                      key: 'd1-native',
                      sequence: 'A'.repeat(16),
                      score: null,
                      globalScore: null,
                      sourceModelId: 'rfd3/fp16',
                    },
                  ],
                };
                queueMicrotask(() => {
                  for (const cb of listeners.get('message') ?? []) {
                    cb({ data: { id: msg.id, type: 'result', outputs: { structure, sequences }, elapsedMs: 5 } } as MessageEvent);
                  }
                });
              } else if (msg.type === 'dispose') {
                disposeMessages++;
                // Simulate 650 ms GPU queue drain (> old 500 ms timeout, < new 2000 ms timeout)
                setTimeout(() => {
                  if (!terminated) {
                    disposedCleanlyBeforeTerminate = true;
                  }
                  for (const cb of listeners.get('message') ?? []) {
                    cb({ data: { id: msg.id, type: 'disposed' } } as MessageEvent);
                  }
                }, 650);
              }
            },
            terminate() {
              if (!terminated) {
                terminated = true;
                liveWorkers--;
              }
            },
          } as unknown as Worker;
        },
      },
    });
    reg.freeze();

    // Disable idle timer (gpuIdleReleaseMs: 0) to prove release comes from vramLeaseMode === 'exclusive'
    const scheduler = new ExecutionScheduler({ pluginRegistry: reg, gpuIdleReleaseMs: 0 });
    const res = await scheduler.run(
      {
        id: 'excl-lease-test',
        name: 'Exclusive Lease',
        nodes: [{ id: 'g', adapterId: 'rfd3', params: { ...defaultParamsForAdapter('rfd3'), length: 16 } }],
        edges: [],
      },
      () => {},
    );

    expect(res.status).toBe('completed');
    expect(disposeMessages).toBe(1);
    expect(disposedCleanlyBeforeTerminate).toBe(true);
    expect(liveWorkers).toBe(0);
    expect(scheduler.getActiveGpuAdapter()).toBeNull();
  });
});
