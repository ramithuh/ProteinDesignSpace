// Layer 1 Core Kernel — Standardized Browser-Native WebGPU Worker Harness.
// Eliminates per-worker boilerplate across 21 GPU workers:
//   - Binds `self.onmessage = async (ev) => ...` (directly awaitable in Node unit tests)
//   - Unified resilient WebGPU adapter/device acquisition + Safari desktop memory budget helper
//   - Reference-counted serialized control chain (`inFlightRunPromise` + `activeControlOps` + `controlChain`) preventing abort/dispose/run GPU races:
//     `{ id, type: 'aborted' }` is ONLY posted after `inFlightRunPromise` settles cleanly.
//   - Owned transferable ArrayBuffer collection (`collectOwnedTransferables`).

import {
  estimateSafeBrowserMemoryBudgetBytes,
  requestResilientWebGpuAdapter,
  requestResilientWebGpuDevice,
  type ResilientAdapterResult,
  type ResilientDeviceResult,
} from './webgpu-device';

export interface WorkerProgressPayload {
  phase: 'init' | 'weights' | 'infer' | (string & {});
  progress: number;
  detail: string;
}

export interface WorkerCacheContext {
  fetchBytes: (
    cacheKey: string,
    url: string,
    onProgress?: (p: { loaded: number; total: number; fromCache: boolean }) => void,
    signal?: AbortSignal,
  ) => Promise<Uint8Array>;
  evictUrls: (urls: Iterable<string>) => Promise<void>;
}

export interface WorkerRuntimeContext {
  acquireWebGpuAdapter: typeof requestResilientWebGpuAdapter;
  acquireWebGpuDevice: typeof requestResilientWebGpuDevice;
  estimateMemoryBudgetBytes: typeof estimateSafeBrowserMemoryBudgetBytes;
  cache?: WorkerCacheContext;
  postProgress: (id: number, payload: WorkerProgressPayload) => void;
  /** Optional AbortSignal active during `init()` so weight downloads can cancel cleanly. */
  signal?: AbortSignal;
}

export interface WorkerRunContext<
  TParams extends Record<string, unknown> = Record<string, unknown>,
  TInputs extends Record<string, unknown> = Record<string, unknown>,
> {
  id: number;
  params: TParams;
  inputs: TInputs;
  /** Domain-agnostic runtime resource caps (e.g. `{ maxResidues: 1024 }`). */
  resourceLimits?: Record<string, number>;
  /** @deprecated Backwards-compatible alias for protein fold workers; prefer `resourceLimits`. */
  maxFoldResidues?: number;
  signal: AbortSignal;
  runtime: WorkerRuntimeContext;
  progress: (phase: WorkerProgressPayload['phase'], progress: number, detail: string) => void;
}

export interface ModelRunner<
  TParams extends Record<string, unknown> = Record<string, unknown>,
  TInputs extends Record<string, unknown> = Record<string, unknown>,
  TOutputs extends Record<string, unknown> = Record<string, unknown>,
> {
  init?(ctx: WorkerRuntimeContext): Promise<void>;
  run(
    ctx: WorkerRunContext<TParams, TInputs>,
  ): Promise<{ outputs: TOutputs; transferables?: Transferable[] }>;
  /** Cooperative abort: stop iteration loop cleanly while keeping GPU pipelines & weights warm. */
  abort?(runId: number): Promise<void> | void;
  /** Full VRAM release prior to worker termination. */
  dispose(): Promise<void> | void;
}

export interface WorkerScopeLike {
  onmessage: ((ev: MessageEvent) => Promise<void> | void) | null;
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
}

export interface WorkerHarnessOptions {
  scope?: WorkerScopeLike;
  cache?: WorkerCacheContext;
  acquireWebGpuAdapter?: typeof requestResilientWebGpuAdapter;
  acquireWebGpuDevice?: typeof requestResilientWebGpuDevice;
}

/**
 * Collect unique, non-shared ArrayBuffers backing TypedArrays inside a result payload
 * so workers can zero-copy transfer large coordinate/score tensors without aliasing errors.
 */
export function collectOwnedTransferables(
  value: unknown,
  seenObjects = new Set<unknown>(),
  buffers = new Set<ArrayBuffer>(),
): Transferable[] {
  if (!value || typeof value !== 'object') return [...buffers];
  if (seenObjects.has(value)) return [...buffers];
  seenObjects.add(value);

  if (ArrayBuffer.isView(value)) {
    const buf = value.buffer;
    if (
      typeof ArrayBuffer !== 'undefined' &&
      buf instanceof ArrayBuffer &&
      value.byteOffset === 0 &&
      value.byteLength === buf.byteLength
    ) {
      buffers.add(buf);
    }
    return [...buffers];
  }
  if (typeof ArrayBuffer !== 'undefined' && value instanceof ArrayBuffer) {
    buffers.add(value);
    return [...buffers];
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      collectOwnedTransferables(item, seenObjects, buffers);
    }
    return [...buffers];
  }
  for (const v of Object.values(value as Record<string, unknown>)) {
    collectOwnedTransferables(v, seenObjects, buffers);
  }
  return [...buffers];
}

function clampProgress(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.max(0, Math.min(1, p));
}

export interface WorkerHarnessHandle {
  handleMessage: (ev: MessageEvent) => Promise<void>;
  isRunning: () => boolean;
  isAborting: () => boolean;
}

/**
 * Bind a `ModelRunner` onto a Web Worker scope (`self.onmessage = async (ev) => ...`).
 * Enforces single-flight execution, cooperative abort settling (`inFlightRunPromise`),
 * structured progress clamping, and safe `dispose` teardown.
 */
export function createWorkerHarness<
  TParams extends Record<string, unknown> = Record<string, unknown>,
  TInputs extends Record<string, unknown> = Record<string, unknown>,
  TOutputs extends Record<string, unknown> = Record<string, unknown>,
>(
  runner: ModelRunner<TParams, TInputs, TOutputs>,
  opts?: WorkerHarnessOptions,
): WorkerHarnessHandle {
  const scope: WorkerScopeLike =
    opts?.scope ?? (globalThis.self as unknown as WorkerScopeLike);

  let initialized = false;
  let inFlightRunPromise: Promise<void> | null = null;
  let activeRunId: number | null = null;
  let activeAbortController: AbortController | null = null;
  // Reference-counted control barrier + serialized promise chain so overlapping
  // `abort` and `dispose` messages can never clear each other's guard early.
  let activeControlOps = 0;
  let controlChain: Promise<void> = Promise.resolve();

  const enqueueControlOp = (op: () => Promise<void>): Promise<void> => {
    activeControlOps++;
    const next = controlChain.then(op, op).finally(() => {
      activeControlOps--;
    });
    controlChain = next.catch(() => {});
    return next;
  };

  const post = (msg: unknown, transfer?: Transferable[]) => {
    if (transfer && transfer.length > 0) {
      scope.postMessage(msg, transfer);
    } else {
      scope.postMessage(msg);
    }
  };

  const runtimeCtx: WorkerRuntimeContext = {
    acquireWebGpuAdapter: opts?.acquireWebGpuAdapter ?? requestResilientWebGpuAdapter,
    acquireWebGpuDevice: opts?.acquireWebGpuDevice ?? requestResilientWebGpuDevice,
    estimateMemoryBudgetBytes: estimateSafeBrowserMemoryBudgetBytes,
    cache: opts?.cache,
    postProgress: (id, payload) => {
      post({
        id,
        type: 'progress',
        phase: payload.phase,
        progress: clampProgress(payload.progress),
        detail: payload.detail,
      });
    },
  };

  const handleMessage = async (ev: MessageEvent): Promise<void> => {
    const msg = (ev?.data ?? {}) as {
      id?: number;
      type?: string;
      params?: TParams;
      inputs?: TInputs;
      maxFoldResidues?: number;
    };
    const msgId = typeof msg.id === 'number' ? msg.id : 0;

    if (msg.type === 'abort') {
      await enqueueControlOp(async () => {
        const runIdToAbort = activeRunId ?? msgId;
        activeAbortController?.abort(new Error('Run aborted'));
        await runner.abort?.(runIdToAbort);
        // Critical invariant: NEVER post 'aborted' until in-flight run's finally block settles!
        if (inFlightRunPromise) {
          await inFlightRunPromise.catch(() => {});
        }
        post({ id: msgId, type: 'aborted' });
      });
      return;
    }

    if (msg.type === 'dispose') {
      await enqueueControlOp(async () => {
        activeAbortController?.abort(new Error('Worker disposed'));
        if (inFlightRunPromise) {
          await runner.abort?.(activeRunId ?? msgId);
          await inFlightRunPromise.catch(() => {});
        }
        await runner.dispose();
        initialized = false;
        post({ id: msgId, type: 'disposed' });
      });
      return;
    }

    if (msg.type === 'run') {
      // In-flight run-lease guard: reject overlapping runs while busy or aborting/disposing
      if (inFlightRunPromise !== null || activeControlOps > 0) {
        post({
          id: msgId,
          type: 'error',
          message:
            activeControlOps > 0
              ? 'Worker is aborting or disposing a previous run; wait for settlement before starting a new run.'
              : 'Worker already has an active run in flight.',
        });
        return;
      }

      const ctrl = new AbortController();
      activeRunId = msgId;
      activeAbortController = ctrl;
      const t0 = performance.now();

      const runPromise = (async () => {
        try {
          if (!initialized && runner.init) {
            await runner.init({ ...runtimeCtx, signal: ctrl.signal });
            initialized = true;
          }
          if (ctrl.signal.aborted) {
            throw new Error('Workflow run cancelled');
          }
          const res = await runner.run({
            id: msgId,
            params: (msg.params ?? {}) as TParams,
            inputs: (msg.inputs ?? {}) as TInputs,
            ...(typeof msg.maxFoldResidues === 'number'
              ? { maxFoldResidues: msg.maxFoldResidues }
              : {}),
            signal: ctrl.signal,
            runtime: runtimeCtx,
            progress: (phase, progress, detail) => {
              runtimeCtx.postProgress(msgId, { phase, progress, detail });
            },
          });
          if (ctrl.signal.aborted || activeControlOps > 0) {
            // Suppress late 'result' if an abort/dispose took ownership of this run
            return;
          }
          // Opt-in zero-copy transfer only when runner explicitly supplies `transferables`
          // so cached ArrayBuffers retained by a runner across runs are never detached.
          const transfer = res.transferables ?? [];
          post(
            {
              id: msgId,
              type: 'result',
              outputs: res.outputs,
              elapsedMs: performance.now() - t0,
            },
            transfer,
          );
        } catch (err) {
          if (ctrl.signal.aborted || activeControlOps > 0) {
            // Abort path will emit `{ id: abortId, type: 'aborted' }` once inFlightRunPromise settles
            return;
          }
          post({
            id: msgId,
            type: 'error',
            message: err instanceof Error ? err.message : String(err),
          });
        } finally {
          if (activeRunId === msgId) {
            activeRunId = null;
            activeAbortController = null;
          }
        }
      })();

      inFlightRunPromise = runPromise;
      try {
        await runPromise;
      } finally {
        if (inFlightRunPromise === runPromise) {
          inFlightRunPromise = null;
        }
      }
    }
  };

  if (scope) {
    scope.onmessage = handleMessage;
  }

  return {
    handleMessage,
    isRunning: () => inFlightRunPromise !== null,
    isAborting: () => activeControlOps > 0,
  };
}

export type { ResilientAdapterResult, ResilientDeviceResult };
