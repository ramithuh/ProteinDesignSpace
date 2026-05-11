// Messages exchanged between the main thread and the inference worker.
// Each request carries an `id`; responses echo it so callers can correlate.

import type { ScheduleKind } from '$lib/protein-ops/schedules';

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
      step: number; // 0..numSteps (frame index in the trajectory)
      t: number;
      latencyMs: number; // per-step latency
    }
  | {
      id: number;
      type: 'sampleDone';
      /** length = nSamples; each entry is a Float32Array of [length*3, 3] flat coords for the final frame. */
      finalSamples: Float32Array[];
      /** length = nSamples; each entry is an array of (numSteps+1) Float32Array frames of [length*3, 3] flat. */
      trajectories: Float32Array[][];
      length: number; // residue count
      totalLatencyMs: number;
    }
  | { id: number; type: 'error'; message: string };
