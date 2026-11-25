// Queue pacing for RFD4-Proteina's WebGPU engines, installed on the upstream engine instances from the worker.
//
// `Rfd4pEngine.run` / `runAndRead` (static/vendor/rfd4-proteina/webgpu/rfd4p/engine.js) record one command buffer per
// program (two trunk layers) but hand a whole trunk step — every recycle, 2.8 s to 13.6 s at n=280 by upstream's own
// note — to ONE `queue.submit`. While it runs, macOS gives WindowServer no GPU time and the desktop freezes. The vendor
// tree is re-synced verbatim from upstream on every build (tools/sync-upstream.mjs), so the fix lives here: the same
// recorded dispatches are replayed in the same order, cut into submissions of about SUBMISSION_MS each, with about
// QUEUED_WORK_MS of work queued — LocalFold's rule (vendor/localfold-*/webgpu/runtime/queue-pacing.js). LocalFold's
// QueuePacer learns the cuts from a stack of identical blocks, which an RFD4 step is not, so here each
// (pipeline, grid) is timed once, alone on an idle queue, and submissions are filled by summing those times.

import type { Rfd4GeneratorLike } from './lifecycle';

/** The longest a submission should run: Windows reset its driver at 35 ms in LocalFold's measurements, never at 26. */
export const SUBMISSION_MS = 15;
/** About a second queued: Stop and closing the tab wait for whatever is already committed. */
export const QUEUED_WORK_MS = 1000;

const PACED = Symbol.for('pds.rfd4.queuePaced');

/** One recorded dispatch, as `Rfd4pEngine.step` returns it. */
export interface Rfd4Dispatch {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  x: number;
  y?: number;
}

/** A recorded program (`Rfd4pEngine.program`): its flat dispatches, and the replay upstream encodes them with. */
export interface Rfd4Program {
  steps?: Rfd4Dispatch[];
  run(pass: GPUComputePassEncoder): void;
}

export type Rfd4Programs = Rfd4Program | readonly Rfd4Programs[];

type Piece = Rfd4Dispatch | Rfd4Program;

interface Rfd4EngineLike {
  device: GPUDevice;
  run(...programs: Rfd4Programs[]): void;
  runAndRead(
    programs: Rfd4Programs,
    buffer: GPUBuffer,
    floats?: number,
    offsetFloats?: number,
    Type?: unknown,
  ): Promise<unknown>;
  [PACED]?: true;
}

function isDispatch(piece: Piece): piece is Rfd4Dispatch {
  return (piece as Rfd4Dispatch).pipeline !== undefined;
}

/** Programs flattened to dispatches, in order; a program without `steps` stays whole. */
function piecesOf(programs: Rfd4Programs, out: Piece[] = []): Piece[] {
  if (Array.isArray(programs)) {
    for (const p of programs as readonly Rfd4Programs[]) piecesOf(p, out);
  } else {
    const p = programs as Rfd4Program;
    if (Array.isArray(p.steps)) out.push(...p.steps);
    else out.push(p);
  }
  return out;
}

const gridKey = (d: Rfd4Dispatch) => `${d.x},${d.y ?? 1}`;

/** One submission being filled. */
class Submission {
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private ms = 0;

  constructor(private readonly device: GPUDevice) {}

  fits(ms: number): boolean {
    return this.pass === null || this.ms + ms <= SUBMISSION_MS;
  }

  add(piece: Piece, ms: number): void {
    if (!this.pass) {
      this.encoder = this.device.createCommandEncoder({ label: 'rfd4p' });
      this.pass = this.encoder.beginComputePass({ label: 'rfd4p' });
    }
    if (isDispatch(piece)) {
      this.pass.setPipeline(piece.pipeline);
      this.pass.setBindGroup(0, piece.bindGroup);
      this.pass.dispatchWorkgroups(piece.x, piece.y);
    } else {
      piece.run(this.pass);
    }
    this.ms += ms;
  }

  /** Submits what is open and returns its expected time, or null when nothing was open. */
  flush(): number | null {
    if (!this.pass || !this.encoder) return null;
    this.pass.end();
    this.device.queue.submit([this.encoder.finish()]);
    const ms = this.ms;
    this.encoder = null;
    this.pass = null;
    this.ms = 0;
    return ms;
  }
}

/** How long the page takes to hear a fence on an idle queue, which a timed dispatch's fence also includes. */
async function idleFenceMs(device: GPUDevice): Promise<number> {
  let best = Infinity;
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    await device.queue.onSubmittedWorkDone();
    best = Math.min(best, performance.now() - t0);
  }
  return best;
}

export class Rfd4QueuePacer {
  /** Measured ms per pipeline and grid. Pipelines outlive `releaseActivations`, so a length is timed once per worker. */
  private readonly costs = new WeakMap<GPUComputePipeline, Map<string, number>>();
  private fenceMs: number | undefined;
  private inflight: Array<{ done: Promise<unknown>; ms: number }> = [];
  private queuedMs = 0;

  constructor(readonly device: GPUDevice) {}

  costOf(piece: Piece): number | undefined {
    return isDispatch(piece) ? this.costs.get(piece.pipeline)?.get(gridKey(piece)) : undefined;
  }

  /**
   * Submits `programs` in order, cut by what is already known (anything unknown goes alone), without waiting: for
   * upstream's synchronous `run`, whose caller goes on to write buffers the queue must see after this work.
   */
  submitNow(programs: Rfd4Programs): void {
    const sub = new Submission(this.device);
    for (const piece of piecesOf(programs)) {
      const ms = this.costOf(piece) ?? SUBMISSION_MS;
      if (!sub.fits(ms)) sub.flush();
      sub.add(piece, ms);
    }
    sub.flush();
  }

  /** Submits `programs` in order, timing dispatches not seen before and keeping about QUEUED_WORK_MS queued. */
  async submitPaced(programs: Rfd4Programs): Promise<void> {
    const sub = new Submission(this.device);
    for (const piece of piecesOf(programs)) {
      const known = this.costOf(piece);
      if (known === undefined && isDispatch(piece)) {
        this.queued(sub.flush());
        await this.time(piece);
        continue;
      }
      const ms = known ?? SUBMISSION_MS;
      if (!sub.fits(ms)) {
        this.queued(sub.flush());
        await this.room();
      }
      sub.add(piece, ms);
    }
    this.queued(sub.flush());
  }

  /** Everything submitted so far has finished (a readback returned). */
  settled(): void {
    this.inflight = [];
    this.queuedMs = 0;
  }

  private queued(ms: number | null): void {
    if (ms === null) return;
    const done = this.device.queue.onSubmittedWorkDone();
    done.catch(() => {});
    this.inflight.push({ done, ms });
    this.queuedMs += ms;
  }

  private async room(): Promise<void> {
    while (this.inflight.length > 1 && this.queuedMs > QUEUED_WORK_MS) {
      const oldest = this.inflight.shift()!;
      this.queuedMs -= oldest.ms;
      await oldest.done;
    }
  }

  /** Runs `piece` alone on an idle queue and records how long it took. */
  private async time(piece: Rfd4Dispatch): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
    this.settled();
    this.fenceMs ??= await idleFenceMs(this.device);
    const sub = new Submission(this.device);
    const t0 = performance.now();
    sub.add(piece, 0);
    sub.flush();
    await this.device.queue.onSubmittedWorkDone();
    const ms = Math.max(performance.now() - t0 - this.fenceMs, 0);
    let grids = this.costs.get(piece.pipeline);
    if (!grids) this.costs.set(piece.pipeline, (grids = new Map()));
    grids.set(gridKey(piece), ms);
  }
}

const pacers = new WeakMap<GPUDevice, Rfd4QueuePacer>();

function pacerFor(device: GPUDevice): Rfd4QueuePacer {
  let pacer = pacers.get(device);
  if (!pacer) pacers.set(device, (pacer = new Rfd4QueuePacer(device)));
  return pacer;
}

function isEngine(engine: unknown): engine is Rfd4EngineLike {
  const e = engine as Partial<Rfd4EngineLike> | null | undefined;
  return Boolean(e?.device) && typeof e?.run === 'function' && typeof e?.runAndRead === 'function';
}

/** Routes one engine's `run` / `runAndRead` through the device's pacer. Upstream's readback is kept: it gets no programs. */
export function paceRfd4Engine(engine: unknown): void {
  if (!isEngine(engine) || engine[PACED]) return;
  const pacer = pacerFor(engine.device);
  const runAndRead = engine.runAndRead;
  engine.run = (...programs: Rfd4Programs[]) => pacer.submitNow(programs);
  engine.runAndRead = async (programs, buffer, floats, offsetFloats, Type) => {
    await pacer.submitPaced(programs);
    try {
      return await runAndRead.call(engine, [], buffer, floats, offsetFloats, Type);
    } finally {
      pacer.settled();
    }
  };
  engine[PACED] = true;
}

type PacedGenerator = Rfd4GeneratorLike & {
  getEncoder?: () => { engine?: unknown } | null | undefined;
  [PACED]?: true;
};

/**
 * Paces every engine `generator` has, and the encoder it creates later (`getEncoder`, on the first partial-diffusion
 * or inverse-folding run). Safe to call before every `generate()`.
 */
export function installRfd4QueuePacing(generator: PacedGenerator | null | undefined): void {
  if (!generator) return;
  paceRfd4Engine(generator.trunk?.engine);
  paceRfd4Engine(generator.decoder?.engine);
  paceRfd4Engine(generator.encoder?.engine);
  if (generator[PACED] || typeof generator.getEncoder !== 'function') return;
  const getEncoder = generator.getEncoder;
  generator.getEncoder = function (this: unknown) {
    const encoder = getEncoder.call(this);
    paceRfd4Engine(encoder?.engine);
    return encoder;
  };
  generator[PACED] = true;
}
