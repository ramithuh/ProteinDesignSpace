// Pure helpers for resolving GNINA Monte Carlo search worker count and
// safely falling back to single-threaded execution (`workers: 1`) if nested
// Web Worker spawning fails before the first search chain completes.

export interface GninaDockProgressEvent {
  phase: string;
  completed: number;
  total: number;
}

export interface GninaDockFallbackOptions {
  workers: number;
  signal?: AbortSignal;
  onProgress?: (ev: GninaDockProgressEvent) => void;
  onFallback?: (err: unknown) => void;
}

export interface GninaDockFallbackResult<T> {
  result: T;
  usedWorkers: number;
  fellBack: boolean;
}

/**
 * Resolve the effective worker count (`>= 1`, `<= 8`, `<= exhaustiveness`) for
 * `vendor/gnina/shared/docking.js`'s `dockPose(..., { workers })`.
 *
 * - Returns `1` whenever `Worker` is unavailable (`!hasWorkerGlobal`).
 * - Maps Auto (`0`, omitted, negative, or non-finite) to
 *   `Math.max(1, Math.min(8, safeExh, safeHw))`.
 * - Clamps explicit values to `Math.max(1, Math.min(8, safeExh, Math.round(parsed)))`.
 */
export function resolveGninaDockWorkers(
  rawWorkers: unknown,
  exhaustiveness: number,
  hasWorkerGlobal = typeof Worker === 'function',
  hardwareConcurrency = globalThis.navigator?.hardwareConcurrency,
): number {
  const safeExh = Number.isFinite(exhaustiveness) ? Math.max(1, Math.round(exhaustiveness)) : 1;
  if (!hasWorkerGlobal) return 1;

  const parsed =
    rawWorkers === undefined || rawWorkers === null || rawWorkers === ''
      ? 0
      : Number(rawWorkers);

  if (!Number.isFinite(parsed) || parsed <= 0) {
    const safeHw =
      typeof hardwareConcurrency === 'number' &&
      Number.isFinite(hardwareConcurrency) &&
      hardwareConcurrency >= 1
        ? Math.floor(hardwareConcurrency)
        : 1;
    return Math.max(1, Math.min(8, safeExh, safeHw));
  }

  return Math.max(1, Math.min(8, safeExh, Math.round(parsed)));
}

function isAbortOrDestroyedError(err: unknown): boolean {
  if (!(err instanceof Error)) {
    const msg = String(err ?? '');
    return /engine has been destroyed|workflow run cancelled|\babort/i.test(msg);
  }
  if (err.name === 'AbortError') return true;
  return /engine has been destroyed|workflow run cancelled|\babort/i.test(err.message);
}

/**
 * Execute `runDockPose(opts.workers, ...)` and transparently retry once with
 * `workers: 1` iff nested worker startup failed during `search` before any
 * chain completed (`searchStarted && searchCompleted === 0`), while never
 * masking pre-search validation errors, mid-search failures (`searchCompleted > 0`),
 * cancellation signals, or destroyed-engine errors.
 */
export async function dockPoseWithWorkerFallback<T>(
  runDockPose: (
    workers: number,
    onProgress: (ev: GninaDockProgressEvent) => void,
  ) => Promise<T>,
  opts: GninaDockFallbackOptions,
): Promise<GninaDockFallbackResult<T>> {
  let searchStarted = false;
  let searchCompleted = 0;

  const wrappedProgress = (ev: GninaDockProgressEvent) => {
    if (ev.phase === 'search') {
      searchStarted = true;
      if (Number.isFinite(ev.completed)) {
        searchCompleted = Math.max(searchCompleted, ev.completed);
      }
    }
    opts.onProgress?.(ev);
  };

  try {
    const result = await runDockPose(opts.workers, wrappedProgress);
    return { result, usedWorkers: opts.workers, fellBack: false };
  } catch (err) {
    const canFallback =
      opts.workers > 1 &&
      searchStarted &&
      searchCompleted === 0 &&
      !opts.signal?.aborted &&
      !isAbortOrDestroyedError(err);

    if (!canFallback) {
      throw err;
    }

    opts.onFallback?.(err);
    const result = await runDockPose(1, (ev) => {
      opts.onProgress?.(ev);
    });
    return { result, usedWorkers: 1, fellBack: true };
  }
}
