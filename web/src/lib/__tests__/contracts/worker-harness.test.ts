import { describe, expect, it, vi } from 'vitest';
import {
  PluginRegistry,
  aggregatePluginCachePolicies,
  collectOwnedTransferables,
  createWorkerHarness,
  deriveKernelSeed,
  estimateSafeBrowserMemoryBudgetBytes,
  hydrateLegacyArtifactToEnvelope,
  isArtifactEnvelope,
  isManagedCacheStorageName,
  requestResilientWebGpuDevice,
  resolveBatchCap,
  serializeEnvelopeForDnv1,
  type DomainPack,
  type ModelPlugin,
  type WorkerScopeLike,
} from '$lib/kernel';
import { decodePayloadBytes, encodePayloadBytes } from '$lib/engine/artifact-store';
import { ExecutionScheduler, RunCancellationError } from '$lib/engine/scheduler';
import type { WorkflowGraph } from '$lib/engine/types';

function makeTestDomainPack(id = 'test-domain', portType = 'TestArtifact'): DomainPack {
  return {
    id,
    displayName: `Domain ${id}`,
    stages: [
      { id: 'source', label: 'Input', order: 0 },
      { id: 'predict', label: 'Predict', order: 1 },
    ],
    portTypes: [
      {
        typeId: portType,
        domainId: id,
        label: portType,
        color: { bg: '#111', fg: '#eee', border: '#333' },
        peekPriority: 10,
        summarize: (art) => ({ id: art.id, kind: art.kind }),
        viewerId: `${id}-viewer`,
      },
    ],
    artifactViewers: [
      {
        id: `${id}-viewer`,
        domainId: id,
        label: 'Test Viewer',
        componentLoader: async () => ({ default: {} }),
      },
    ],
  };
}

function makeFakeWorkerFromRunner(
  runner: Parameters<typeof createWorkerHarness>[0],
): { worker: Worker; terminated: () => boolean; postedToWorker: unknown[] } {
  let isTerminated = false;
  const listeners = new Map<string, Set<(ev: MessageEvent) => void>>();
  const postedToWorker: unknown[] = [];

  const scope: WorkerScopeLike = {
    onmessage: null,
    postMessage: (data: unknown) => {
      if (isTerminated) return;
      const ev = { data } as MessageEvent;
      for (const cb of [...(listeners.get('message') ?? [])]) {
        cb(ev);
      }
    },
  };

  createWorkerHarness(runner, { scope });

  const worker = {
    addEventListener(type: string, cb: (ev: MessageEvent) => void) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)!.add(cb);
    },
    removeEventListener(type: string, cb: (ev: MessageEvent) => void) {
      listeners.get(type)?.delete(cb);
    },
    postMessage(data: unknown) {
      postedToWorker.push(data);
      if (isTerminated) return;
      void scope.onmessage?.({ data } as MessageEvent);
    },
    terminate() {
      isTerminated = true;
    },
  } as unknown as Worker;

  return { worker, terminated: () => isTerminated, postedToWorker };
}

describe('Phase 1 contract: Layer 1 Core Kernel, PluginRegistry, ArtifactEnvelope, & WorkerHarness', () => {
  it('enforces PluginRegistry startup invariants including shortCode uniqueness and soft-wired cross-domain ports', () => {
    const reg = new PluginRegistry();
    reg.registerDomainPack(makeTestDomainPack('protein-biopolymer', 'StructureArtifact'));

    const validPlugin: ModelPlugin = {
      id: 'mock-fold',
      domainId: 'protein-biopolymer',
      manifest: {
        id: 'mock-fold',
        displayName: 'Mock Fold',
        shortName: 'MockFold',
        stage: 'predict',
        runtime: 'wgsl-custom',
        estimatedVramMB: 256,
        approxWeightsMB: 50,
        licenseNote: 'MIT',
        description: 'Mock GPU plugin',
        inputs: {
          structure: { type: 'StructureArtifact', required: true },
          ligands: {
            type: [],
            required: false,
            acceptsOptionalPortTypes: ['MoleculeSet'],
          },
        },
        outputs: {
          out: { type: 'StructureArtifact' },
        },
      },
      execution: {
        runtime: 'wgsl-custom',
        fallback: { kind: 'none', reason: 'WebGPU only' },
        createWorker: () => ({}) as Worker,
      },
      cachePolicy: { opfsDirs: ['mock-fold-v1'], cacheStorageNames: ['mock-fold-v1'] },
      share: { shortCode: 'mf' },
      catalog: { categoryId: 'predict', conciseRole: 'Mock folding' },
      display: { formatCanvasSummary: () => 'MockFold' },
      preflight: () => ({ estimatedVramMB: 256, estimatedDownloadMB: 50, workItems: 1 }),
    };

    // Boots cleanly even though optional cross-domain PortType 'MoleculeSet' is not loaded yet!
    reg.registerPlugin(validPlugin);
    expect(reg.resolveAcceptedPortTypes(validPlugin.manifest.inputs.ligands)).toEqual([]);

    // Once @domain/small-molecule loads 'MoleculeSet', the soft-wired port resolves automatically
    reg.registerDomainPack(makeTestDomainPack('small-molecule', 'MoleculeSet'));
    expect(reg.resolveAcceptedPortTypes(validPlugin.manifest.inputs.ligands)).toEqual(['MoleculeSet']);
    expect(
      reg.arePortsCompatible({ type: 'MoleculeSet' }, validPlugin.manifest.inputs.ligands),
    ).toBe(true);

    // Duplicate shortCode must throw synchronously
    expect(() =>
      reg.registerPlugin({
        ...validPlugin,
        id: 'mock-fold-2',
        manifest: { ...validPlugin.manifest, id: 'mock-fold-2' },
        share: { shortCode: 'mf' },
      }),
    ).toThrow(/shortCode "mf" collision/i);

    // GPU plugin missing static cachePolicy must throw
    expect(() =>
      reg.registerPlugin({
        ...validPlugin,
        id: 'mock-fold-nocache',
        manifest: { ...validPlugin.manifest, id: 'mock-fold-nocache' },
        share: { shortCode: 'nc' },
        cachePolicy: {},
      }),
    ).toThrow(/static cachePolicy/i);

    // Freeze prevents further mutations
    reg.freeze();
    expect(reg.isFrozen()).toBe(true);
    expect(() => reg.registerDomainPack(makeTestDomainPack('late', 'LatePort'))).toThrow(/frozen/i);
  });

  it('losslessly hydrates legacy flat DNV1 artifacts into ArtifactEnvelope and preserves TypedArrays', () => {
    const legacyFlat = {
      kind: 'StructureArtifact',
      id: 'struct-1ctt',
      label: '1CTT',
      sourceModelId: 'pdb-input',
      sourceStructureId: 'parent-1',
      designKey: 'design:0',
      length: 2,
      sequence: 'AC',
      caCoords: new Float32Array([1.5, -2.25, 3.0, 4.5, 5.0, -6.75]),
      observedBackboneMask: new Uint8Array([0x0f, 0x07]),
      pdbText: 'END\n',
    };

    const env = hydrateLegacyArtifactToEnvelope(legacyFlat);
    expect(isArtifactEnvelope(env)).toBe(true);
    expect(env.kind).toBe('StructureArtifact');
    expect(env.lineage?.groupKey).toBe('design:0');
    expect(env.lineage?.parentArtifactIds).toEqual(['parent-1']);

    // Round-trip canonical ArtifactEnvelope through real DNV1 binary codec
    const bytes = encodePayloadBytes(serializeEnvelopeForDnv1(env));
    const decoded = decodePayloadBytes<Record<string, unknown>>(bytes);
    expect(isArtifactEnvelope(decoded)).toBe(true);
    const payload = (decoded as unknown as typeof env).payload;
    expect(payload.caCoords).toBeInstanceOf(Float32Array);
    expect(Array.from(payload.caCoords as Float32Array)).toEqual([1.5, -2.25, 3.0, 4.5, 5.0, -6.75]);
    expect(payload.observedBackboneMask).toBeInstanceOf(Uint8Array);

    // Legacy flat projection preserves flat field access for Phase 1-3 compatibility
    const flatAgain = serializeEnvelopeForDnv1(env, { legacyFlat: true });
    expect(flatAgain.pdbText).toBe('END\n');
    expect(flatAgain.caCoords).toBeInstanceOf(Float32Array);
  });

  it('WorkerHarness awaits inFlightRunPromise settlement before posting aborted and rejects concurrent runs', async () => {
    const events: string[] = [];
    const posted: Array<Record<string, unknown>> = [];
    let releaseRunStep!: () => void;
    const stepGate = new Promise<void>((r) => {
      releaseRunStep = r;
    });

    const scope: WorkerScopeLike = {
      onmessage: null,
      postMessage: (m) => {
        const msg = m as Record<string, unknown>;
        posted.push(msg);
        events.push(`post:${String(msg.type)}:${String(msg.id)}`);
      },
    };

    createWorkerHarness(
      {
        async run(ctx) {
          events.push('run:start');
          ctx.progress('infer', 1.4, 'clamped');
          try {
            await stepGate;
            return { outputs: { ok: true, buf: new Float32Array([1, 2, 3]) } };
          } finally {
            events.push('run:finally');
          }
        },
        abort() {
          events.push('abort:called');
        },
        dispose() {
          events.push('dispose:called');
        },
      },
      { scope },
    );

    // Start run #1 (does not finish until stepGate resolves)
    const run1 = scope.onmessage!({ data: { id: 1, type: 'run', params: {}, inputs: {} } } as MessageEvent);
    await Promise.resolve();
    expect(posted.some((m) => m.type === 'progress' && m.progress === 1)).toBe(true);

    // Overlapping run #2 must be rejected immediately by the run-lease guard
    await scope.onmessage!({ data: { id: 2, type: 'run', params: {}, inputs: {} } } as MessageEvent);
    expect(posted.some((m) => m.type === 'error' && m.id === 2)).toBe(true);

    // Trigger abort #3 while run #1 is still in flight
    const abort3 = scope.onmessage!({ data: { id: 3, type: 'abort' } } as MessageEvent);
    await Promise.resolve();
    // MUST NOT have posted 'aborted' yet because run #1's finally block hasn't settled!
    expect(posted.some((m) => m.type === 'aborted')).toBe(false);

    // Now let run #1 unwind cleanly
    releaseRunStep();
    await Promise.all([run1, abort3]);

    const finallyIdx = events.indexOf('run:finally');
    const abortedIdx = events.indexOf('post:aborted:3');
    expect(finallyIdx).toBeGreaterThan(-1);
    expect(abortedIdx).toBeGreaterThan(finallyIdx);
    // Late 'result' must be suppressed once aborted
    expect(posted.some((m) => m.type === 'result')).toBe(false);
  });

  it('ExecutionScheduler keeps warm worker alive on timely cooperative abort and escalates to dispose+terminate on timeout', async () => {
    let workerSpawnCount = 0;
    let releaseSlowStep!: () => void;
    let activeFakeWorker!: ReturnType<typeof makeFakeWorkerFromRunner>;

    const reg = new PluginRegistry();
    reg.registerDomainPack(makeTestDomainPack('protein-biopolymer', 'StructureArtifact'));
    reg.registerPlugin({
      id: 'evodiff',
      domainId: 'protein-biopolymer',
      manifest: {
        id: 'evodiff',
        displayName: 'EvoDiff',
        shortName: 'EvoDiff',
        stage: 'predict',
        runtime: 'wgsl-custom',
        estimatedVramMB: 128,
        approxWeightsMB: 40,
        licenseNote: 'MIT',
        description: 'Cooperative abort test plugin',
        inputs: {},
        outputs: { sequences: { type: 'StructureArtifact' } },
      },
      execution: {
        runtime: 'wgsl-custom',
        fallback: { kind: 'none', reason: 'test' },
        supportsCooperativeAbort: true,
        abortGraceMs: 1500,
        createWorker: () => {
          workerSpawnCount++;
          let runCount = 0;
          activeFakeWorker = makeFakeWorkerFromRunner({
            async run() {
              runCount++;
              if (runCount === 1) {
                await new Promise<void>((r) => {
                  releaseSlowStep = r;
                });
              }
              return {
                outputs: {
                  sequences: {
                    kind: 'SequenceSet',
                    id: `seq-${runCount}`,
                    parentStructureId: 'none',
                    sourceModelId: 'evodiff',
                    candidates: [
                      {
                        id: 'c1',
                        sequence: 'ACDEFGHIKLMN',
                        score: -1.2,
                        globalScore: -1.2,
                        sourceModelId: 'evodiff',
                      },
                    ],
                  },
                },
              };
            },
            abort() {
              releaseSlowStep?.();
            },
            dispose() {},
          });
          return activeFakeWorker.worker;
        },
      },
      cachePolicy: { opfsDirs: ['evodiff-v1'] },
      share: { shortCode: 'e' },
      catalog: { categoryId: 'predict', conciseRole: 'test' },
      display: { formatCanvasSummary: () => 'EvoDiff' },
      preflight: () => ({ estimatedVramMB: 128, estimatedDownloadMB: 40, workItems: 1 }),
    });

    const scheduler = new ExecutionScheduler({ pluginRegistry: reg, gpuIdleReleaseMs: 0 });
    const graph: WorkflowGraph = {
      id: 'coop-abort-test',
      name: 'Coop Abort',
      description: '',
      nodes: [{ id: 'n1', adapterId: 'evodiff', params: { length: 12, numSequences: 1, seed: 1 } }],
      edges: [],
    };

    // 1. Cancel run #1: worker acks `{ type: 'aborted' }` cleanly -> worker stays WARM (not terminated!)
    const ac = new AbortController();
    const run1Promise = scheduler.run(graph, () => {}, { signal: ac.signal });
    await new Promise((r) => setTimeout(r, 15));
    ac.abort();
    await expect(run1Promise).rejects.toBeInstanceOf(RunCancellationError);
    expect(scheduler.getActiveRunState().status).toBe('cancelled');
    expect(scheduler.getActiveRunState().leaseHeld).toBe(false);
    expect(activeFakeWorker.terminated()).toBe(false);
    expect(scheduler.getActiveGpuAdapter()).toBe('evodiff');
    expect(workerSpawnCount).toBe(1);

    // 2. Run #2 immediately reuses the exact same warm worker without respawning!
    const res2 = await scheduler.run(graph, () => {});
    expect(res2.status).toBe('completed');
    expect(workerSpawnCount).toBe(1);

    // 3. Now test timeout escalation when a hung worker never replies `{ type: 'aborted' }`
    vi.useFakeTimers();
    try {
      let hungWorkerTerminated = false;
      const hungReg = new PluginRegistry();
      hungReg.registerDomainPack(makeTestDomainPack('protein-biopolymer', 'StructureArtifact'));
      hungReg.registerPlugin({
        ...reg.getPlugin('evodiff')!,
        execution: {
          ...reg.getPlugin('evodiff')!.execution,
          createWorker: () =>
            ({
              addEventListener() {},
              removeEventListener() {},
              postMessage() {},
              terminate() {
                hungWorkerTerminated = true;
              },
            }) as unknown as Worker,
        },
      });
      const hungScheduler = new ExecutionScheduler({ pluginRegistry: hungReg, gpuIdleReleaseMs: 0 });
      const ac2 = new AbortController();
      const hungRun = hungScheduler.run(graph, () => {}, { signal: ac2.signal });
      const hungAssertion = expect(hungRun).rejects.toBeInstanceOf(RunCancellationError);
      await vi.advanceTimersByTimeAsync(10);
      ac2.abort();
      // Grace window (1500 ms) + dispose timeout (500 ms) -> escalates to terminate()
      await vi.advanceTimersByTimeAsync(2100);
      await hungAssertion;
      expect(hungScheduler.getActiveRunState().status).toBe('cancelled');
      expect(hungScheduler.getActiveRunState().leaseHeld).toBe(false);
      expect(hungWorkerTerminated).toBe(true);
      expect(hungScheduler.getActiveGpuAdapter()).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('supports Safari memory budget helper, Dawn subgroup retry, deterministic v3 seeds, and static cache aggregation', async () => {
    // Safari macOS desktop (no deviceMemory) -> non-null safe desktop budget
    expect(
      estimateSafeBrowserMemoryBudgetBytes({
        userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15',
        hardwareConcurrency: 12,
      }),
    ).toBeGreaterThan(4 * 1024 * 1024 * 1024);
    // Chromium with deviceMemory present -> null (defer to browser)
    expect(
      estimateSafeBrowserMemoryBudgetBytes({ deviceMemory: 8, userAgent: 'Macintosh Chrome' }),
    ).toBeNull();
    // iPhone mobile -> null (keep conservative mobile cap)
    expect(
      estimateSafeBrowserMemoryBudgetBytes({ userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0)' }),
    ).toBeNull();

    // Deterministic domain-separated v3 seeds
    const s1 = deriveKernelSeed({ baseSeed: 42, pluginId: 'admet-chemprop', stage: 'property-pred', itemKey: 'mol:0' });
    const s2 = deriveKernelSeed({ baseSeed: 42, pluginId: 'admet-chemprop', stage: 'property-pred', itemKey: 'mol:0' });
    const s3 = deriveKernelSeed({ baseSeed: 42, pluginId: 'admet-chemprop', stage: 'property-pred', itemKey: 'mol:1' });
    expect(s1).toBe(s2);
    expect(s1).not.toBe(s3);
    expect(s1).toBeGreaterThanOrEqual(0);
    expect(s1).toBeLessThanOrEqual(0x7fffffff);
    expect(resolveBatchCap(50, 16)).toBe(16);

    // Owned transferable ArrayBuffer collection deduplicates cleanly
    const f32 = new Float32Array([1, 2, 3]);
    const transfers = collectOwnedTransferables({ a: f32, b: { c: f32 } });
    expect(transfers).toHaveLength(1);
    expect(transfers[0]).toBe(f32.buffer);

    // Static cache policy aggregation
    const reg = new PluginRegistry();
    reg.registerDomainPack(makeTestDomainPack('property-prediction', 'PropertyTable'));
    reg.registerPlugin({
      id: 'admet-chemprop',
      domainId: 'property-prediction',
      manifest: {
        id: 'admet-chemprop',
        displayName: 'ADMET',
        shortName: 'ADMET',
        stage: 'predict',
        runtime: 'wgsl-custom',
        estimatedVramMB: 64,
        approxWeightsMB: 12,
        licenseNote: 'MIT',
        description: 'ADMET predictor',
        inputs: {},
        outputs: { table: { type: 'PropertyTable' } },
      },
      execution: {
        runtime: 'wgsl-custom',
        fallback: { kind: 'none', reason: 'WebGPU' },
        createWorker: () => ({}) as Worker,
      },
      cachePolicy: {
        cacheStorageNames: ['admet-v1'],
        cacheStoragePrefixes: ['admet-shard-'],
        opfsDirs: ['admet-v1'],
      },
      share: { shortCode: 'ac' },
      catalog: { categoryId: 'predict', conciseRole: 'ADMET' },
      display: { formatCanvasSummary: () => 'ADMET' },
      preflight: () => ({ estimatedVramMB: 64, estimatedDownloadMB: 12, workItems: 1 }),
    });
    const ns = aggregatePluginCachePolicies(reg);
    expect(isManagedCacheStorageName('admet-v1', ns)).toBe(true);
    expect(isManagedCacheStorageName('admet-shard-001', ns)).toBe(true);
    expect(isManagedCacheStorageName('unrelated-cache', ns)).toBe(false);
    expect(ns.opfsDirs).toContain('admet-v1');

    // Dawn experimental subgroup exclusion retry
    let attempts = 0;
    const fakeAdapter = {
      features: new Set(['shader-f16', 'chromium-experimental-subgroup-matrix']),
      limits: { maxBufferSize: 1024, maxStorageBufferBindingSize: 1024 },
      info: { vendor: 'nvidia', architecture: 'blackwell', device: '5090', description: 'RTX 5090' },
      async requestDevice(desc: { requiredFeatures?: string[] }) {
        attempts++;
        if (desc.requiredFeatures?.includes('chromium-experimental-subgroup-matrix')) {
          throw new Error('subgroup-matrix rejected by driver');
        }
        return { label: 'recovered-device' } as unknown as GPUDevice;
      },
    } as unknown as GPUAdapter;

    const origNav = globalThis.navigator;
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { gpu: { requestAdapter: async () => fakeAdapter } },
    });
    try {
      const devRes = await requestResilientWebGpuDevice({
        useF16: true,
        optionalFeatures: ['chromium-experimental-subgroup-matrix'],
      });
      expect(attempts).toBe(2);
      expect(devRes.device.label).toBe('recovered-device');
    } finally {
      Object.defineProperty(globalThis, 'navigator', { configurable: true, value: origNav });
    }
  });
});
