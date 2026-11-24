import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'svelte/compiler';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExecutionScheduler, RunCancellationError, type ActiveRunState } from '$lib/engine/scheduler';
import type { FoldedStructure, StructureArtifact, WorkflowGraph } from '$lib/engine/types';
import { resolveViewerSpinState } from '$lib/viewer/viewer-spin';

function readSrc(relPath: string): string {
  return readFileSync(resolve(process.cwd(), relPath), 'utf8');
}

function makePlaceholderStructure(id: string, seq = 'ACDEFGHIKL'): StructureArtifact {
  const L = Math.max(3, seq.length);
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'mock-worker',
    length: L,
    sequence: seq.slice(0, L),
    caCoords: new Float32Array(L * 3),
    backbone4Coords: new Float32Array(L * 4 * 3),
    pdbText: 'END\n',
    geometry: {
      length: L,
      rg: 10,
      meanCaDist: 3.8,
      chainBreaks: 0,
      clashes: 0,
      helixPct: 50,
      sheetPct: 0,
      coilPct: 50,
    },
  };
}

let workerMode: 'complete' | 'hang' | 'fail' = 'complete';
let workerRunCalls = 0;

class MockSpinTestWorker {
  private listeners = new Map<string, Set<(ev: MessageEvent) => void>>();

  addEventListener(type: string, cb: (ev: MessageEvent) => void) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(cb);
  }

  removeEventListener(type: string, cb: (ev: MessageEvent) => void) {
    this.listeners.get(type)?.delete(cb);
  }

  postMessage(msg: { id: number; type: string }) {
    if (msg.type === 'dispose') {
      this.emit('message', { data: { id: msg.id, type: 'disposed' } } as MessageEvent);
      return;
    }
    if (msg.type === 'abort') {
      this.emit('message', { data: { id: msg.id, type: 'aborted' } } as MessageEvent);
      return;
    }
    if (msg.type === 'run') {
      workerRunCalls += 1;
      if (workerMode === 'hang') return;
      if (workerMode === 'fail') {
        queueMicrotask(() => {
          this.emit('message', {
            data: { id: msg.id, type: 'error', message: 'Simulated WebGPU failure' },
          } as MessageEvent);
        });
        return;
      }
      const folds: FoldedStructure = {
        kind: 'FoldedStructure',
        id: 'mock-folds',
        parentStructureId: 'none',
        folds: [],
      };
      const structure = makePlaceholderStructure('mock-foldstruct', 'ACDEFGHIKL');
      queueMicrotask(() => {
        this.emit('message', {
          data: { id: msg.id, type: 'result', outputs: { folds, structure }, elapsedMs: 4 },
        } as MessageEvent);
      });
    }
  }

  terminate() {
    // no-op
  }

  private emit(type: string, ev: MessageEvent) {
    for (const cb of this.listeners.get(type) ?? []) cb(ev);
  }
}

function makeGpuTestGraph(id = 'spin-pause-graph'): WorkflowGraph {
  return {
    id,
    name: 'Spin Pause Test Graph',
    nodes: [
      { id: 'seq', adapterId: 'sequence-input', params: { sequence: 'ACDEFGHIKLMNPQRSTVWY' } },
      { id: 'fold', adapterId: 'localfold-af2', params: { recycles: 1, maxSequences: 1 } },
    ],
    edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
  };
}

describe('Issue #94 — resolveViewerSpinState pure resolver', () => {
  it('enables spin and disables occlusion only when spin is on, structure is loaded, not canonical, and not paused', () => {
    expect(
      resolveViewerSpinState({
        spin: true,
        hasAnyStructure: true,
        canonicalCamera: false,
        pauseSpin: false,
      }),
    ).toEqual({
      effectiveSpin: true,
      disableOcclusion: true,
      spinConfigured: true,
    });
  });

  it('pauses spin while keeping disableOcclusion=true and spinConfigured=true when pauseSpin=true', () => {
    expect(
      resolveViewerSpinState({
        spin: true,
        hasAnyStructure: true,
        canonicalCamera: false,
        pauseSpin: true,
      }),
    ).toEqual({
      effectiveSpin: false,
      disableOcclusion: true,
      spinConfigured: true,
    });
  });

  it('disables spin when hasAnyStructure=false or canonicalCamera=true and keeps occlusion off only during active pauseSpin', () => {
    expect(
      resolveViewerSpinState({
        spin: true,
        hasAnyStructure: false,
        canonicalCamera: false,
        pauseSpin: false,
      }),
    ).toEqual({
      effectiveSpin: false,
      disableOcclusion: false,
      spinConfigured: false,
    });

    expect(
      resolveViewerSpinState({
        spin: true,
        hasAnyStructure: false,
        canonicalCamera: false,
        pauseSpin: true,
      }),
    ).toEqual({
      effectiveSpin: false,
      disableOcclusion: true,
      spinConfigured: false,
    });

    expect(
      resolveViewerSpinState({
        spin: true,
        hasAnyStructure: true,
        canonicalCamera: true,
        pauseSpin: false,
      }),
    ).toEqual({
      effectiveSpin: false,
      disableOcclusion: false,
      spinConfigured: false,
    });

    expect(
      resolveViewerSpinState({
        spin: true,
        hasAnyStructure: true,
        canonicalCamera: true,
        pauseSpin: true,
      }),
    ).toEqual({
      effectiveSpin: false,
      disableOcclusion: true,
      spinConfigured: false,
    });
  });

  it('supports mid-run user spin toggling and restores exact preference when pauseSpin clears', () => {
    let viewerSpin = true;

    // 1. Run starts -> spin pauses without mutating viewerSpin
    expect(
      resolveViewerSpinState({
        spin: viewerSpin,
        hasAnyStructure: true,
        canonicalCamera: false,
        pauseSpin: true,
      }),
    ).toEqual({ effectiveSpin: false, disableOcclusion: true, spinConfigured: true });

    // 2. User unchecks Spin mid-run -> stays paused, spinConfigured updates
    viewerSpin = false;
    expect(
      resolveViewerSpinState({
        spin: viewerSpin,
        hasAnyStructure: true,
        canonicalCamera: false,
        pauseSpin: true,
      }),
    ).toEqual({ effectiveSpin: false, disableOcclusion: true, spinConfigured: false });

    // 3. Run ends -> respects user's mid-run uncheck (stays off)
    expect(
      resolveViewerSpinState({
        spin: viewerSpin,
        hasAnyStructure: true,
        canonicalCamera: false,
        pauseSpin: false,
      }),
    ).toEqual({ effectiveSpin: false, disableOcclusion: false, spinConfigured: false });

    // 4. User checks Spin mid-next-run -> stays paused until run finishes, then resumes
    viewerSpin = true;
    expect(
      resolveViewerSpinState({
        spin: viewerSpin,
        hasAnyStructure: true,
        canonicalCamera: false,
        pauseSpin: true,
      }),
    ).toEqual({ effectiveSpin: false, disableOcclusion: true, spinConfigured: true });

    expect(
      resolveViewerSpinState({
        spin: viewerSpin,
        hasAnyStructure: true,
        canonicalCamera: false,
        pauseSpin: false,
      }),
    ).toEqual({ effectiveSpin: true, disableOcclusion: true, spinConfigured: true });
  });
});

describe('Issue #94 — ExecutionScheduler ActiveRunState transitions driving pauseSpin', () => {
  beforeEach(() => {
    workerMode = 'complete';
    workerRunCalls = 0;
    vi.stubGlobal('Worker', MockSpinTestWorker);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function deriveRunSpinSnapshot(state: ActiveRunState, viewerSpin: boolean, canonicalCamera = false) {
    const isRunning = state.leaseHeld || state.status === 'running';
    return {
      status: state.status,
      isRunning,
      viewerSpin,
      ...resolveViewerSpinState({
        spin: viewerSpin,
        hasAnyStructure: true,
        canonicalCamera,
        pauseSpin: isRunning,
      }),
    };
  }

  it('pauses spin during idle -> running -> completed and restores viewerSpin immediately on completion', async () => {
    const scheduler = new ExecutionScheduler();
    const viewerSpin = true;
    const snapshots: ReturnType<typeof deriveRunSpinSnapshot>[] = [];
    const unsubscribe = scheduler.subscribeRunState((state) => {
      snapshots.push(deriveRunSpinSnapshot(state, viewerSpin));
    });

    await scheduler.run(makeGpuTestGraph('complete-graph'), () => {}, { source: 'canvas', bypassCache: true });
    unsubscribe();

    expect(workerRunCalls).toBe(1);
    expect(snapshots[0]).toMatchObject({ status: 'idle', isRunning: false, effectiveSpin: true, disableOcclusion: true });
    const runningSnapshots = snapshots.filter((s) => s.status === 'running');
    expect(runningSnapshots.length).toBeGreaterThanOrEqual(1);
    for (const s of runningSnapshots) {
      expect(s.isRunning).toBe(true);
      expect(s.effectiveSpin).toBe(false);
      expect(s.disableOcclusion).toBe(true);
      expect(s.viewerSpin).toBe(true);
    }
    expect(snapshots[snapshots.length - 1]).toMatchObject({
      status: 'completed',
      isRunning: false,
      viewerSpin: true,
      effectiveSpin: true,
      disableOcclusion: true,
    });

    await scheduler.releaseActiveGpu();
  });

  it('pauses spin during idle -> running -> cancelled via cancelActiveRun and restores immediately', async () => {
    workerMode = 'hang';
    const scheduler = new ExecutionScheduler();
    const viewerSpin = true;
    const snapshots: ReturnType<typeof deriveRunSpinSnapshot>[] = [];
    const unsubscribe = scheduler.subscribeRunState((state) => {
      snapshots.push(deriveRunSpinSnapshot(state, viewerSpin));
    });

    const runPromise = scheduler.run(makeGpuTestGraph('cancel-graph'), () => {}, {
      source: 'canvas',
      bypassCache: true,
    });
    await Promise.resolve();
    await Promise.resolve();

    expect(scheduler.getActiveRunState().status).toBe('running');
    expect(deriveRunSpinSnapshot(scheduler.getActiveRunState(), viewerSpin).effectiveSpin).toBe(false);
    expect(deriveRunSpinSnapshot(scheduler.getActiveRunState(), viewerSpin).disableOcclusion).toBe(true);

    expect(scheduler.cancelActiveRun('User cancelled')).toBe(true);
    await expect(runPromise).rejects.toBeInstanceOf(RunCancellationError);
    unsubscribe();

    expect(snapshots[snapshots.length - 1]).toMatchObject({
      status: 'cancelled',
      isRunning: false,
      viewerSpin: true,
      effectiveSpin: true,
      disableOcclusion: true,
    });

    await scheduler.releaseActiveGpu();
  });

  it('pauses spin during idle -> running -> failed and restores immediately on failure', async () => {
    workerMode = 'fail';
    const scheduler = new ExecutionScheduler();
    const viewerSpin = true;
    const snapshots: ReturnType<typeof deriveRunSpinSnapshot>[] = [];
    const unsubscribe = scheduler.subscribeRunState((state) => {
      snapshots.push(deriveRunSpinSnapshot(state, viewerSpin));
    });

    await expect(
      scheduler.run(makeGpuTestGraph('fail-graph'), () => {}, { source: 'canvas', bypassCache: true }),
    ).rejects.toThrow(/Simulated WebGPU failure/);
    unsubscribe();

    expect(snapshots.some((s) => s.status === 'running' && !s.effectiveSpin && s.disableOcclusion)).toBe(true);
    expect(snapshots[snapshots.length - 1]).toMatchObject({
      status: 'failed',
      isRunning: false,
      viewerSpin: true,
      effectiveSpin: true,
      disableOcclusion: true,
    });

    await scheduler.releaseActiveGpu();
  });

  it('pauses spin for programmatic API runs (runWorkflowApi, source: "api") and preserves mid-run user toggle', async () => {
    workerMode = 'hang';
    const scheduler = new ExecutionScheduler();
    let viewerSpin = true;
    const snapshots: ReturnType<typeof deriveRunSpinSnapshot>[] = [];
    const unsubscribe = scheduler.subscribeRunState((state) => {
      snapshots.push(deriveRunSpinSnapshot(state, viewerSpin));
    });

    const apiAbort = new AbortController();
    const apiPromise = scheduler.runWorkflowApi(makeGpuTestGraph('api-graph'), () => {}, {
      signal: apiAbort.signal,
      bypassCache: true,
    });
    await Promise.resolve();
    await Promise.resolve();

    const live = scheduler.getActiveRunState();
    expect(live.source).toBe('api');
    expect(live.status).toBe('running');
    expect(live.leaseHeld).toBe(true);
    expect(deriveRunSpinSnapshot(live, viewerSpin)).toMatchObject({
      isRunning: true,
      effectiveSpin: false,
      disableOcclusion: true,
    });

    // Simulate user unchecking Spin mid-run: viewerSpin becomes false, spin remains paused
    viewerSpin = false;
    expect(deriveRunSpinSnapshot(scheduler.getActiveRunState(), viewerSpin)).toMatchObject({
      isRunning: true,
      viewerSpin: false,
      effectiveSpin: false,
      disableOcclusion: true,
      spinConfigured: false,
    });

    apiAbort.abort();
    await expect(apiPromise).rejects.toBeInstanceOf(RunCancellationError);
    unsubscribe();

    expect(snapshots[snapshots.length - 1]).toMatchObject({
      status: 'cancelled',
      isRunning: false,
      viewerSpin: false,
      effectiveSpin: false,
      disableOcclusion: false,
    });

    await scheduler.releaseActiveGpu();
  });
});

describe('Issue #94 — Svelte 5 AST & source wiring guards', () => {
  it('MolSuperposeViewer.svelte and +page.svelte parse cleanly into Svelte 5 ASTs and wire pauseSpin properly', () => {
    const molSrc = readSrc('src/lib/viewer/MolSuperposeViewer.svelte');
    const pageSrc = readSrc('src/routes/+page.svelte');

    expect(parse(molSrc, { modern: true }).fragment).toBeDefined();
    expect(parse(pageSrc, { modern: true }).fragment).toBeDefined();

    // MolSuperposeViewer guards
    expect(molSrc).toContain("import { resolveViewerSpinState } from './viewer-spin';");
    expect(molSrc).toContain('pauseSpin?: boolean;');
    expect(molSrc).toContain('pauseSpin = false,');
    expect(molSrc).toContain('resolveViewerSpinState({');
    expect(molSrc).toContain('let effectiveSpin = $derived(spinState.effectiveSpin);');
    expect(molSrc).toContain('buildPostprocessingProps(illustrative, spinState.disableOcclusion)');
    expect(molSrc).toContain('applyPostprocessing(illustrative, spinState.disableOcclusion);');
    expect(molSrc).not.toContain('applyPostprocessing(illustrative, spin && hasAnyStructure)');
    expect(molSrc).not.toContain('applyPostprocessing(illustrative, spin && hasAnyStructure && !styleOpts.canonicalCamera)');
    // Single consolidated spin/postprocessing effect (not two back-to-back effects calling setProps)
    expect(molSrc).toContain('if (plugin) applySpin(on, speed, illus, disableOcclusion);');
    expect(molSrc).not.toMatch(/\$effect\(\(\) => \{\s*const on = illustrative;\s*const spinning = effectiveSpin;/);
    expect(molSrc).toContain('{#if spinState.spinConfigured}');

    // +page.svelte guards
    expect(pageSrc).toContain('let isSpinPausedForRun = $derived(isRunning && viewerSpin && !isCanonicalViewerLayout);');
    expect(pageSrc).toContain('spin={viewerSpin}');
    expect(pageSrc).toContain('pauseSpin={isRunning}');
    expect(pageSrc).toContain('canonicalCamera={isCanonicalViewerLayout}');
    expect(pageSrc).not.toContain('spin={viewerSpin && !isRunning && !isCanonicalViewerLayout}');
    const pausedHints = pageSrc.match(/class="spin-paused-hint"\>\(paused\)/g) ?? [];
    expect(pausedHints.length).toBe(2);
    expect(pageSrc).not.toMatch(/viewerSpin\s*=\s*(?:false|true)/);
  });
});
