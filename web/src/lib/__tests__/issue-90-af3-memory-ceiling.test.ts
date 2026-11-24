import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it, vi } from 'vitest';
import { validateBioCapabilities, validateRuntimeNodeInputs } from '$lib/engine/bio-capabilities';
import {
  computeSafeBrowserFoldCeiling,
  defaultComputeBackendConfig,
  resolveNodeFoldLimitOptions,
} from '$lib/engine/compute-backend';
import { compileNodeInputPreview } from '$lib/engine/input-preview';
import { LOCALFOLD_MODEL_OPTIONS } from '$lib/engine/registry';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import type { SequenceSet, WorkflowGraph, WorkflowNode } from '$lib/engine/types';
import {
  extractAdapterLimits,
  probeBrowserHardwareCapabilities,
  requestResilientWebGpuAdapter,
  type WebGpuHardwareCapabilities,
} from '$lib/engine/webgpu-device';
import { repairDraftedWorkflowFromBioFeedback } from '$lib/engine/workflow-planner';

const MIB = 1024 * 1024;
const GIB = 1024 * 1024 * 1024;

function make1CttDimerFixture(opts: { ligandCcdCodes?: string; model?: string } = {}) {
  const dimerSeq = 'A'.repeat(294) + ':' + 'A'.repeat(294);
  const seqNode: WorkflowNode = {
    id: 'seq1',
    adapterId: 'sequence-input',
    label: 'WT 1CTT Dimer Seq',
    params: { sequence: `>1CTT_WT_homodimer\n${dimerSeq}` },
  };
  const af3Node: WorkflowNode = {
    id: 'af3_1',
    adapterId: 'localfold-af2',
    label: 'AF3: WT Dimer',
    params: {
      model: opts.model ?? 'af3',
      msaMode: 'mmseqs2',
      ...(opts.ligandCcdCodes ? { ligandCcdCodes: opts.ligandCcdCodes } : {}),
    },
  };
  const graph: WorkflowGraph = {
    id: 'issue-90-1ctt-dimer',
    name: 'Issue 90 1CTT Dimer',
    description: '',
    nodes: [seqNode, af3Node],
    edges: [{ fromNode: 'seq1', fromPort: 'sequences', toNode: 'af3_1', toPort: 'sequences' }],
  };
  const seqSet: SequenceSet = {
    kind: 'SequenceSet',
    id: 'seqset-1ctt',
    parentStructureId: 'none',
    sourceModelId: 'sequence-input',
    candidates: [
      {
        id: '1CTT_WT_homodimer',
        sequence: 'A'.repeat(588),
        chainLengths: [294, 294],
        score: null,
        globalScore: null,
        sourceModelId: 'sequence-input',
      },
    ],
  };
  return { seqNode, af3Node, graph, seqSet };
}

function makeCaps(overrides: Partial<WebGpuHardwareCapabilities> & { maxBindingMiB?: number } = {}): WebGpuHardwareCapabilities {
  const maxStorageBufferBindingSize = (overrides.maxBindingMiB ?? 256) * MIB;
  return {
    isSoftwareFallback: overrides.isSoftwareFallback ?? false,
    limits:
      overrides.limits !== undefined
        ? overrides.limits
        : {
            maxStorageBufferBindingSize,
            maxBufferSize: Math.max(maxStorageBufferBindingSize * 2, 1024 * MIB),
            maxComputeWorkgroupStorageSize: 32768,
            maxComputeInvocationsPerWorkgroup: 1024,
            maxComputeWorkgroupSizeX: 1024,
          },
    deviceMemoryGiB: overrides.deviceMemoryGiB !== undefined ? overrides.deviceMemoryGiB : 8,
    estimatedMemoryBudgetBytes:
      overrides.estimatedMemoryBudgetBytes !== undefined ? overrides.estimatedMemoryBudgetBytes : null,
  };
}

describe('Issue #90: AF3 browser WebGPU memory ceiling & capability scaling', () => {
  it('safely extracts adapter.limits and probes browser hardware capabilities across partial/missing mocks', async () => {
    expect(extractAdapterLimits(null)).toBeNull();
    expect(extractAdapterLimits({ limits: {} } as unknown as GPUAdapter)).toBeNull();

    const partialLimits = extractAdapterLimits({
      limits: { maxStorageBufferBindingSize: 256 * MIB, maxBufferSize: 1024 * MIB },
    } as unknown as GPUAdapter);
    expect(partialLimits).toEqual({
      maxStorageBufferBindingSize: 256 * MIB,
      maxBufferSize: 1024 * MIB,
      maxComputeWorkgroupStorageSize: 0,
      maxComputeInvocationsPerWorkgroup: 0,
      maxComputeWorkgroupSizeX: 0,
    });

    // Safari macOS desktop omits deviceMemory -> estimatedMemoryBudgetBytes fallback (>= 2 GiB)
    const safariCaps = probeBrowserHardwareCapabilities(
      { isSoftwareFallback: false, limits: partialLimits },
      { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15', hardwareConcurrency: 10 },
    );
    expect(safariCaps.deviceMemoryGiB).toBeNull();
    expect(safariCaps.estimatedMemoryBudgetBytes).toBeGreaterThanOrEqual(2 * GIB);

    // requestResilientWebGpuAdapter populates limits & hardwareCapabilities
    const mockAdapter = {
      features: new Set(['shader-f16']),
      limits: {
        maxStorageBufferBindingSize: 512 * MIB,
        maxBufferSize: 2 * GIB,
        maxComputeWorkgroupStorageSize: 32768,
        maxComputeInvocationsPerWorkgroup: 1024,
        maxComputeWorkgroupSizeX: 1024,
      },
      info: { vendor: 'apple', architecture: 'metal-3', device: 'Apple M3', description: 'Apple M3' },
    } as unknown as GPUAdapter;
    vi.stubGlobal('navigator', {
      deviceMemory: 8,
      gpu: { requestAdapter: vi.fn(async () => mockAdapter) },
    });
    try {
      const res = await requestResilientWebGpuAdapter({ retries: 1, retryDelayMs: 1 });
      expect(res.limits?.maxStorageBufferBindingSize).toBe(512 * MIB);
      expect(res.hardwareCapabilities).toMatchObject({
        isSoftwareFallback: false,
        deviceMemoryGiB: 8,
        limits: expect.objectContaining({ maxStorageBufferBindingSize: 512 * MIB }),
      });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('unlocks 588-aa 1CTT_WT_homodimer AF3 (apo & ZN/DHZ holo) in Browser Auto on capable desktop WebGPU', () => {
    const caps256 = makeCaps({ maxBindingMiB: 256, deviceMemoryGiB: 8 });
    const caps512 = makeCaps({ maxBindingMiB: 512, deviceMemoryGiB: 8 });
    const capsSafariMac = probeBrowserHardwareCapabilities(
      {
        isSoftwareFallback: false,
        limits: {
          maxStorageBufferBindingSize: 256 * MIB,
          maxBufferSize: 1024 * MIB,
          maxComputeWorkgroupStorageSize: 32768,
          maxComputeInvocationsPerWorkgroup: 1024,
          maxComputeWorkgroupSizeX: 1024,
        },
      },
      { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15', hardwareConcurrency: 8 },
    );

    expect(computeSafeBrowserFoldCeiling('af3', caps256)).toBe(724);
    expect(computeSafeBrowserFoldCeiling('af3', caps512)).toBe(768);
    expect(computeSafeBrowserFoldCeiling('af3', capsSafariMac)).toBe(724);

    for (const ligandCcdCodes of [undefined, 'ZN, ZN, DHZ, DHZ']) {
      const { af3Node, graph, seqSet } = make1CttDimerFixture({ ligandCcdCodes });
      const browserCfg = defaultComputeBackendConfig();

      for (const caps of [caps256, caps512, capsSafariMac]) {
        const opts = resolveNodeFoldLimitOptions(af3Node, browserCfg, null, {
          localWebGpuAvailable: true,
          localWebGpuFallback: false,
          localHardwareCaps: caps,
        });
        expect(opts.isHostGpu).toBe(false);
        expect(opts.maxFoldResidues).toBeGreaterThanOrEqual(724);

        expect(() => validateRuntimeNodeInputs(af3Node, { sequences: seqSet }, opts)).not.toThrow();
        const bioIssues = validateBioCapabilities(graph, {}, opts);
        expect(bioIssues.some((i) => i.code === 'webgpu-token-ceiling')).toBe(false);

        const preview = compileNodeInputPreview(graph, 'af3_1', { seq1: { sequences: seqSet } }, opts);
        expect(preview.availability).toBe('ready');
        expect(preview.issues.some((i) => i.severity === 'error')).toBe(false);
        expect(preview.payload.compiledSummary.maxFoldResidues).toBe(opts.maxFoldResidues);
      }
    }
  });

  it('rejects 588-aa 1CTT_WT_homodimer on 128 MiB spec-floor, low-RAM, unknown-RAM, software fallback, and unprobed defaults', () => {
    const { af3Node, graph, seqSet } = make1CttDimerFixture();
    const browserCfg = defaultComputeBackendConfig();

    const constrainedCases: Array<{ name: string; caps?: WebGpuHardwareCapabilities | null; expectedCeiling: number }> = [
      { name: 'unprobed', caps: undefined, expectedCeiling: 512 },
      { name: '128 MiB spec floor', caps: makeCaps({ maxBindingMiB: 128, deviceMemoryGiB: 8 }), expectedCeiling: 512 },
      { name: 'low RAM 4 GiB', caps: makeCaps({ maxBindingMiB: 512, deviceMemoryGiB: 4 }), expectedCeiling: 512 },
      {
        name: 'unknown RAM',
        caps: makeCaps({ maxBindingMiB: 512, deviceMemoryGiB: null, estimatedMemoryBudgetBytes: null }),
        expectedCeiling: 512,
      },
      {
        name: 'software fallback',
        caps: makeCaps({ maxBindingMiB: 512, deviceMemoryGiB: 8, isSoftwareFallback: true }),
        expectedCeiling: 512,
      },
    ];

    for (const tc of constrainedCases) {
      expect(computeSafeBrowserFoldCeiling('af3', tc.caps), tc.name).toBe(tc.expectedCeiling);
      const opts = resolveNodeFoldLimitOptions(af3Node, browserCfg, null, {
        localWebGpuAvailable: true,
        localWebGpuFallback: tc.caps?.isSoftwareFallback ?? false,
        localHardwareCaps: tc.caps ?? null,
      });
      expect(() => validateRuntimeNodeInputs(af3Node, { sequences: seqSet }, opts), tc.name).toThrow(
        /512 residues/,
      );
      const bio = validateBioCapabilities(graph, {}, opts);
      const ceilingIssue = bio.find((i) => i.code === 'webgpu-token-ceiling');
      expect(ceilingIssue, tc.name).toBeDefined();
      expect(ceilingIssue?.fix?.computeBackendPatch).toEqual({ maxFoldResidues: 768 });
    }
  });

  it('keeps wide-pair AF3 dialects bounded by pairChannels & tokenExpansionFactor without masking warnings', () => {
    const expectedMeta: Record<string, { pairChannels: number; tokenExpansionFactor?: number }> = {
      af3: { pairChannels: 128 },
      rosettafold3: { pairChannels: 128 },
      boltz2: { pairChannels: 128 },
      openbind0: { pairChannels: 128 },
      protenix2: { pairChannels: 256 },
      opendde: { pairChannels: 384, tokenExpansionFactor: 2 },
      mol1: { pairChannels: 448 },
      intellifold2: { pairChannels: 512 },
      'ef2-fast-300m': { pairChannels: 256 },
      'ef2-fast-600m': { pairChannels: 256 },
      monomer: { pairChannels: 128 },
      multimer: { pairChannels: 128 },
    };
    for (const [id, spec] of Object.entries(expectedMeta)) {
      const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === id);
      expect(meta?.pairChannels).toBe(spec.pairChannels);
      expect(meta?.tokenExpansionFactor).toBe(spec.tokenExpansionFactor);
    }

    const caps128 = makeCaps({ maxBindingMiB: 128, deviceMemoryGiB: 8 });
    const caps256 = makeCaps({ maxBindingMiB: 256, deviceMemoryGiB: 8 });

    expect(computeSafeBrowserFoldCeiling('protenix2', caps128)).toBe(362);
    expect(computeSafeBrowserFoldCeiling('protenix2', caps256)).toBe(512);
    expect(computeSafeBrowserFoldCeiling('opendde', caps128)).toBe(147);
    expect(computeSafeBrowserFoldCeiling('opendde', caps256)).toBe(209);
    expect(computeSafeBrowserFoldCeiling('mol1', caps128)).toBe(273);
    expect(computeSafeBrowserFoldCeiling('mol1', caps256)).toBe(387);
    expect(computeSafeBrowserFoldCeiling('intellifold2', caps128)).toBe(256);
    expect(computeSafeBrowserFoldCeiling('intellifold2', caps256)).toBe(362);
    expect(computeSafeBrowserFoldCeiling('ef2-fast-300m', caps256)).toBe(512);

    // customMax < defaultTokenLimit (384) must not be masked upward above hardLimit (362)
    const { graph } = make1CttDimerFixture({ model: 'intellifold2' });
    graph.nodes[0].params.sequence = `>mid\n${'A'.repeat(370)}`;
    const issues = validateBioCapabilities(graph, {}, { maxFoldResidues: 362, isHostGpu: false });
    expect(issues.some((i) => i.code === 'webgpu-token-ceiling')).toBe(true);
  });

  it('enforces maxFoldResidues across all LocalFold families & hardware maxStorageBufferBindingSize pre-download', async () => {
    const workerSrc = readFileSync(
      resolve(__dirname, '../plugins/localfold-af2/localfold.worker.ts'),
      'utf8',
    );
    expect(workerSrc).toContain('resolveLocalFoldMemoryBudgetBytes');

    const postMessage = vi.fn();
    const fetchSpy = vi.fn(async () => {
      throw new Error('fetch must not be called before pre-download checks pass');
    });
    const scope = {
      postMessage,
      location: { origin: pathToFileURL(resolve(__dirname, '../../../static')).href },
      onmessage: async (_event: { data: unknown }) => {},
    };
    vi.stubGlobal('self', scope);
    vi.stubGlobal('fetch', fetchSpy);
    try {
      vi.resetModules();
      await import('$lib/plugins/localfold-af2/localfold.worker');

      // 1. Pre-download maxFoldResidues enforcement across AF3, AF2 monomer, AF2 multimer, and ESMFold2 (+ DNA/ligands)
      const cases = [
        { model: 'af3', seqLen: 588, maxFoldResidues: 512, expectSubstr: '512 residues' },
        { model: 'af3', seqLen: 500, maxFoldResidues: 512, extra: { dnaSequences: 'ACGT'.repeat(5) }, expectSubstr: '520 tokens' },
        { model: 'af3', seqLen: 500, maxFoldResidues: 512, extra: { ligandCcdCodes: 'ZN, DHZ, ATP, SAM, FAD, HEM, NAD, GTP, GDP, MG, CA, MN, FE' }, expectSubstr: '513 tokens' },
        { model: 'monomer', seqLen: 800, maxFoldResidues: undefined, expectSubstr: '768 residues' },
        { model: 'multimer', seqLen: 400, maxFoldResidues: 362, expectSubstr: '362 residues' },
        { model: 'intellifold2', seqLen: 400, maxFoldResidues: 362, expectSubstr: '362 residues' },
        { model: 'ef2-fast-300m', seqLen: 450, maxFoldResidues: 512, extra: { dnaSequences: 'A'.repeat(100) }, expectSubstr: '550 tokens' },
      ];

      for (const tc of cases) {
        postMessage.mockClear();
        await scope.onmessage({
          data: {
            id: 1,
            type: 'run',
            params: { model: tc.model, msaMode: 'single', ...(tc.extra ?? {}) },
            inputs: {
              sequences: {
                id: 'in',
                candidates: [
                  {
                    id: 'cand1',
                    sequence: 'A'.repeat(tc.seqLen),
                    sourceModelId: 'sequence-input',
                    score: null,
                    globalScore: null,
                  },
                ],
              },
            },
            ...(tc.maxFoldResidues !== undefined ? { maxFoldResidues: tc.maxFoldResidues } : {}),
          },
        });
        expect(postMessage).toHaveBeenCalledTimes(1);
        expect(postMessage).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'error',
            message: expect.stringContaining(tc.expectSubstr),
          }),
        );
      }
      expect(fetchSpy).not.toHaveBeenCalled();

      // 2. Manual maxFoldResidues override (e.g. 2048) still fails fast on hardware maxStorageBufferBindingSize
      // inside ensureModelLoaded BEFORE fetching manifest/weight shards
      const mockLimits = {
        maxStorageBufferBindingSize: 128 * MIB,
        maxBufferSize: 512 * MIB,
        maxComputeWorkgroupStorageSize: 32768,
        maxComputeInvocationsPerWorkgroup: 1024,
        maxComputeWorkgroupSizeX: 1024,
        maxComputeWorkgroupsPerDimension: 65535,
      };
      const mockDevice = {
        features: new Set<string>(),
        limits: mockLimits,
        lost: new Promise(() => {}),
        destroy: vi.fn(),
      };
      const mockAdapter = {
        features: new Set<string>(),
        limits: mockLimits,
        info: { vendor: 'test', architecture: 'test', device: 'test', description: 'test' },
        requestDevice: vi.fn(async () => mockDevice),
      };
      vi.stubGlobal('navigator', {
        deviceMemory: 8,
        gpu: { requestAdapter: vi.fn(async () => mockAdapter) },
      });

      postMessage.mockClear();
      await scope.onmessage({
        data: {
          id: 2,
          type: 'run',
          params: { model: 'af3', msaMode: 'single' },
          inputs: {
            sequences: {
              id: 'in',
              candidates: [
                {
                  id: '1CTT_WT_homodimer',
                  sequence: 'A'.repeat(588),
                  chainLengths: [294, 294],
                  sourceModelId: 'sequence-input',
                  score: null,
                  globalScore: null,
                },
              ],
            },
          },
          maxFoldResidues: 2048,
        },
      });
      expect(postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'error',
          message: expect.stringContaining('maxStorageBufferBindingSize'),
        }),
      );
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('provides 1-click computeBackendPatch without causing workflow-planner auto-repair loops and aligns scheduler/preview', async () => {
    const { af3Node, graph, seqSet } = make1CttDimerFixture();

    // workflow-planner must NOT loop or emit bogus notes on computeBackendPatch-only webgpu-token-ceiling fixes
    const repaired = repairDraftedWorkflowFromBioFeedback(graph);
    expect(repaired.notes.some((n) => n.includes('Raise Max fold len'))).toBe(false);

    // Scheduler forwards foldLimitResolver into validateBioCapabilities (Layer 1) and validateRuntimeNodeInputs (Layer 2)
    const caps128 = makeCaps({ maxBindingMiB: 128, deviceMemoryGiB: 8 });
    const caps256 = makeCaps({ maxBindingMiB: 256, deviceMemoryGiB: 8 });
    const scheduler = new ExecutionScheduler({ gpuIdleReleaseMs: 0 });
    scheduler.setRemoteGpuExecutor((node) => {
      if (node.adapterId !== 'localfold-af2') return null;
      const pdbText = 'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C  \nEND\n';
      return Promise.resolve({
        outputs: {
          folds: {
            kind: 'FoldedStructure',
            id: 'folds-1',
            parentStructureId: 'none',
            sourceStructureId: 'none',
            folds: [
              {
                sequenceId: '1CTT_WT_homodimer',
                seed: 0,
                sequence: 'A'.repeat(588),
                foldModelId: 'localfold-af3-int5',
                pdbText,
                caCoords: new Float32Array([0, 0, 0]),
                plddt: new Float32Array([90]),
                meanPlddt: 90,
                confidence: 90,
                confidenceAvailable: true,
                ptm: 0.85,
                iptm: 0.82,
                chainLengths: [294, 294],
                chainIds: Array.from({ length: 588 }, (_, i) => (i < 294 ? 'A' : 'B')),
                elapsedMs: 1,
              },
            ],
          },
          structure: {
            kind: 'StructureArtifact',
            id: 'struct-1',
            label: '1CTT_WT_homodimer',
            sourceModelId: 'localfold-af2',
            length: 1,
            sequence: 'A',
            caCoords: new Float32Array([0, 0, 0]),
            backbone4Coords: new Float32Array(12),
            pdbText,
            chainIds: ['A'],
            chainLengths: [1],
            atomGranularity: 'all-atom',
            bfactorSemantics: 'plddt-0-100',
            geometry: computeGeometryStats(new Float32Array([0, 0, 0]), ['A']),
          },
        },
        elapsedMs: 1,
      });
    });

    scheduler.setFoldLimitResolver((n) =>
      resolveNodeFoldLimitOptions(n, defaultComputeBackendConfig(), null, {
        localWebGpuAvailable: true,
        localWebGpuFallback: false,
        localHardwareCaps: caps128,
      }),
    );
    await expect(
      scheduler.run(graph, () => {}, { runId: 'test-sched-caps-128' }),
    ).rejects.toThrow(/512 residues/);

    scheduler.setFoldLimitResolver((n) =>
      resolveNodeFoldLimitOptions(n, defaultComputeBackendConfig(), null, {
        localWebGpuAvailable: true,
        localWebGpuFallback: false,
        localHardwareCaps: caps256,
      }),
    );
    const res = await scheduler.run(graph, () => {}, { runId: 'test-sched-caps-256' });
    expect(res.outputsByNode.af3_1.folds).toBeDefined();

    // Unprobed input-preview reports ceiling error; probed 256 MiB desktop preview is ready with 724
    const unprobedPrev = compileNodeInputPreview(graph, 'af3_1', { seq1: { sequences: seqSet } });
    expect(unprobedPrev.availability).toBe('error');
    expect(unprobedPrev.payload.compiledSummary.maxFoldResidues).toBe(512);

    const probedOpts = resolveNodeFoldLimitOptions(af3Node, defaultComputeBackendConfig(), null, {
      localWebGpuAvailable: true,
      localWebGpuFallback: false,
      localHardwareCaps: caps256,
    });
    const probedPrev = compileNodeInputPreview(graph, 'af3_1', { seq1: { sequences: seqSet } }, probedOpts);
    expect(probedPrev.availability).toBe('ready');
    expect(probedPrev.payload.compiledSummary.maxFoldResidues).toBe(724);
  });
});
