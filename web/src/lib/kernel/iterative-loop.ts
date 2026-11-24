// Layer 1 Core Kernel — Bounded Closed-Loop Iterative Optimization / Active-Learning Runner.
// Supports BindCraft / BoltzDesign / EvoBind / Bayesian optimization / wet-lab feedback loops
// without introducing unconstrained cycles into the DAG ExecutionScheduler.
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

import type { ArtifactEnvelope } from './artifacts';
import { deriveKernelSeed } from './work-plan';

export const MAX_ITERATIVE_LOOP_ROUNDS = 100;

export interface IterativeLoopSpec {
  readonly loopId: string;
  readonly maxRounds: number;
  readonly baseSeed: number;
  /** Optional convergence goal direction when selecting the best round automatically. */
  readonly objectiveDirection?: 'maximize' | 'minimize';
}

export interface IterativeRoundContext<TState = unknown> {
  readonly roundIndex: number;
  readonly maxRounds: number;
  readonly seed: number;
  readonly prevState: TState | null;
  readonly prevArtifacts: Readonly<Record<string, ArtifactEnvelope>> | null;
  readonly signal?: AbortSignal;
}

export interface IterativeRoundOutput<
  TState = unknown,
  TMetrics extends Record<string, number | string | boolean | null> = Record<
    string,
    number | string | boolean | null
  >,
> {
  state: TState;
  artifacts: Record<string, ArtifactEnvelope>;
  /** Scalar optimization objective for best-round tracking & convergence checks. */
  objectiveScore?: number | null;
  metrics?: TMetrics;
  converged?: boolean;
}

export interface IterativeRoundRecord<
  TState = unknown,
  TMetrics extends Record<string, number | string | boolean | null> = Record<
    string,
    number | string | boolean | null
  >,
> {
  readonly roundIndex: number;
  readonly seed: number;
  readonly state: TState;
  readonly artifacts: Record<string, ArtifactEnvelope>;
  readonly objectiveScore: number | null;
  readonly metrics: TMetrics;
  readonly elapsedMs: number;
}

export interface IterativeLoopResult<
  TState = unknown,
  TMetrics extends Record<string, number | string | boolean | null> = Record<
    string,
    number | string | boolean | null
  >,
> {
  readonly loopId: string;
  readonly status: 'completed' | 'converged' | 'cancelled';
  readonly rounds: readonly IterativeRoundRecord<TState, TMetrics>[];
  readonly bestRound: IterativeRoundRecord<TState, TMetrics> | null;
  readonly totalElapsedMs: number;
}

function stampRoundLineage(
  artifacts: Record<string, ArtifactEnvelope>,
  roundIndex: number,
): Record<string, ArtifactEnvelope> {
  const stamped: Record<string, ArtifactEnvelope> = {};
  for (const [port, env] of Object.entries(artifacts)) {
    stamped[port] = {
      ...env,
      lineage: {
        ...(env.lineage ?? {}),
        roundIndex,
        ...(roundIndex > 0 ? { parentRoundIndex: roundIndex - 1 } : {}),
      },
    };
  }
  return stamped;
}

/**
 * Execute a deterministic, strictly bounded iterative optimization loop (rounds `0..maxRounds-1`).
 * Stamps `lineage.roundIndex` / `parentRoundIndex` onto every produced `ArtifactEnvelope`,
 * derives per-round `denovo-seed/v3` seeds, and handles early convergence or cancellation cleanly.
 */
export async function runIterativeLoop<
  TState = unknown,
  TMetrics extends Record<string, number | string | boolean | null> = Record<
    string,
    number | string | boolean | null
  >,
>(opts: {
  spec: IterativeLoopSpec;
  executeRound: (
    ctx: IterativeRoundContext<TState>,
  ) => Promise<IterativeRoundOutput<TState, TMetrics>>;
  stopWhen?: (round: IterativeRoundRecord<TState, TMetrics>) => boolean;
  onRoundComplete?: (round: IterativeRoundRecord<TState, TMetrics>) => void;
  signal?: AbortSignal;
}): Promise<IterativeLoopResult<TState, TMetrics>> {
  const { spec, executeRound, stopWhen, onRoundComplete, signal } = opts;
  const maxRounds = Math.max(
    1,
    Math.min(MAX_ITERATIVE_LOOP_ROUNDS, Math.round(spec.maxRounds || 1)),
  );
  const direction = spec.objectiveDirection ?? 'maximize';
  const tStart = performance.now();

  const rounds: IterativeRoundRecord<TState, TMetrics>[] = [];
  let bestRound: IterativeRoundRecord<TState, TMetrics> | null = null;
  let prevState: TState | null = null;
  let prevArtifacts: Record<string, ArtifactEnvelope> | null = null;
  let status: IterativeLoopResult['status'] = 'completed';

  for (let roundIndex = 0; roundIndex < maxRounds; roundIndex++) {
    if (signal?.aborted) {
      status = 'cancelled';
      break;
    }

    const rStart = performance.now();
    const seed = deriveKernelSeed({
      baseSeed: spec.baseSeed,
      pluginId: spec.loopId,
      stage: 'iterative-loop',
      itemKey: `round:${roundIndex}`,
    });

    const out = await executeRound({
      roundIndex,
      maxRounds,
      seed,
      prevState,
      prevArtifacts,
      signal,
    });

    const stampedArtifacts = stampRoundLineage(out.artifacts, roundIndex);
    const score =
      typeof out.objectiveScore === 'number' && Number.isFinite(out.objectiveScore)
        ? out.objectiveScore
        : null;

    const record: IterativeRoundRecord<TState, TMetrics> = {
      roundIndex,
      seed,
      state: out.state,
      artifacts: stampedArtifacts,
      objectiveScore: score,
      metrics: (out.metrics ?? {}) as TMetrics,
      elapsedMs: performance.now() - rStart,
    };

    rounds.push(record);
    prevState = out.state;
    prevArtifacts = stampedArtifacts;

    if (score !== null) {
      if (
        !bestRound ||
        bestRound.objectiveScore === null ||
        (direction === 'maximize'
          ? score > bestRound.objectiveScore
          : score < bestRound.objectiveScore)
      ) {
        bestRound = record;
      }
    } else if (!bestRound) {
      bestRound = record;
    }

    onRoundComplete?.(record);

    if (out.converged || stopWhen?.(record)) {
      status = 'converged';
      break;
    }
  }

  return {
    loopId: spec.loopId,
    status,
    rounds,
    bestRound,
    totalElapsedMs: performance.now() - tStart,
  };
}
