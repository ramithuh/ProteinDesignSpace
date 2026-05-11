// Messages exchanged between the main thread and the inference worker.
// Each request carries an `id`; responses echo it so callers can correlate.

import type { ScheduleKind } from '$lib/protein-ops/schedules';
import type { GtMode } from '$lib/samplers/sde-proteina-endpoint';

export type SamplerKind = 'euler-fm' | 'sde-proteina';

export interface SDEParams {
  gamma: number; // Proteina γ (default 0.1)
  gtMode: GtMode;
  gtEps: number; // default 1e-2
  gtMax: number | null; // default 3.0 in production; null = no cap
  tLimOde: number; // default 0.99
  noiseSeed: number; // for per-step noise (separate from initial-noise seed)
}

export type WorkerRequest =
  | { id: number; type: 'load'; modelUrl: string }
  | {
      id: number;
      type: 'sample';
      length: number;
      nSamples: number;
      nfe: number;
      schedule: ScheduleKind;
      schedulP: number;
      seed: number;
      sampler: SamplerKind;
      sde?: SDEParams; // required iff sampler === 'sde-proteina'
    };

export type WorkerResponse =
  | { id: number; type: 'progress'; loaded: number; total: number }
  | {
      id: number;
      type: 'loaded';
      provider: string;
      inputNames: string[];
      outputNames: string[];
      modelSizeBytes: number;
      initLatencyMs: number;
    }
  | {
      id: number;
      type: 'sampleStep';
      step: number; // 0..numSteps
      t: number;
      latencyMs: number;
    }
  | {
      id: number;
      type: 'sampleDone';
      finalSamples: Float32Array[];
      trajectories: Float32Array[][];
      length: number;
      totalLatencyMs: number;
      sampler: SamplerKind;
    }
  | { id: number; type: 'error'; message: string };
