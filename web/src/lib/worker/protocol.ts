// Messages exchanged between the main thread and the inference worker.
// Each request carries an `id`; responses echo it so callers can correlate.

export type WorkerRequest =
  | { id: number; type: 'load'; modelUrl: string }
  | { id: number; type: 'forward'; length: number; seed: number; t: number };

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
      type: 'forwardDone';
      outputShape: readonly number[];
      latencyMs: number;
      sample: number[];
    }
  | { id: number; type: 'error'; message: string };
