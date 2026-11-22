/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/dyna1-webgpu (MIT · Dyna-1 + ESM-3 small + VQ-VAE Structure Encoder)
// Predicts per-residue millisecond conformational exchange probabilities (0..1) using
// sequence + 3D structure tokens (when a full-backbone structure is connected), structure-only,
// or sequence-only mode, and paints dynamics probabilities onto 3D B-factors.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  extractDyna1Backbone3Coords,
  normalizeDyna1InputMode,
  resolveDyna1Mode,
  type Dyna1ResolvedMode,
} from '$lib/engine/dyna1-helpers';
import { paintStructureBfactors } from '$lib/protein-ops/bfactor-paint';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  ResidueTrackArtifact,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/dyna1-webgpu/resolve/main';

type Precision = 'int8' | 'fp16' | 'fp32';

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function collectOwnedTransferables(views: Array<ArrayBufferView | undefined>): Transferable[] {
  const seen = new Set<ArrayBuffer>();
  const out: Transferable[] = [];
  for (const v of views) {
    if (!v) continue;
    const buf = v.buffer;
    if (!(buf instanceof ArrayBuffer)) continue;
    if (v.byteOffset !== 0 || v.byteLength !== buf.byteLength) continue;
    if (seen.has(buf)) continue;
    seen.add(buf);
    out.push(buf);
  }
  return out;
}

function normalizePrecision(raw: unknown): Precision {
  const val = String(raw ?? 'int8').trim().toLowerCase();
  if (val === 'fp16' || val === 'fp32') return val;
  return 'int8';
}

function cleanCanonicalSequence(raw: string, label: string): string {
  const seq = raw.replace(/[\s:/,-]+/g, '').toUpperCase();
  if (seq.length < 1 || seq.length > 1024) {
    throw new Error(
      `Dyna-1 supports 1–1024 canonical amino acids ("${label}" has ${seq.length}).`,
    );
  }
  if (!/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(seq)) {
    throw new Error(
      `Dyna-1 requires canonical 20-AA protein sequences in "${label}" (non-canonical letters found).`,
    );
  }
  return seq;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let dynaMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let ownedDevice: GPUDevice | null = null;
let loadedPrecision: Precision | null = null;
let activeAbort: AbortController | null = null;

async function acquireWebGpuDevice(): Promise<GPUDevice> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU is not available in this browser worker.');
  }
  let adapter: GPUAdapter | null = null;
  for (let attempt = 0; attempt < 3 && !adapter; attempt++) {
    if (attempt > 0) await new Promise((r) => setTimeout(r, attempt * 250));
    adapter =
      (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
      (await navigator.gpu.requestAdapter());
  }
  if (!adapter) {
    throw new Error('No WebGPU adapter found for Dyna-1.');
  }
  return await adapter.requestDevice();
}

async function ensureEngine(id: number, precision: Precision) {
  if (engine && loadedPrecision === precision) return engine;

  if (engine) {
    try {
      engine.destroy();
    } catch {
      /* ignore */
    }
    engine = null;
    loadedPrecision = null;
  }
  if (ownedDevice) {
    try {
      ownedDevice.destroy();
    } catch {
      /* ignore */
    }
    ownedDevice = null;
  }

  if (!dynaMod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading Dyna-1 WebGPU modules...',
    });
    const moduleUrl = new URL('/vendor/dyna1/webgpu/index.js', self.location.origin).href;
    dynaMod = await import(/* @vite-ignore */ moduleUrl);
  }

  const device = await acquireWebGpuDevice();
  ownedDevice = device;

  const envBaseUrl = (import.meta.env.VITE_DYNA1_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.05,
      detail: `Fetching Dyna-1 (${precision.toUpperCase()}) weights...`,
    });

    try {
      return await dynaMod.createDyna1Engine({
        baseUrl,
        precision,
        device,
        cache: 'none',
        useF16: false,
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { phase?: string; loaded?: number; total?: number }) => {
          if (prog.phase !== 'weights' && prog.phase !== 'structure-encoder') return;
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac = total > 0 ? loaded / total : 0.4;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          const isStructEnc = prog.phase === 'structure-encoder';
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: isStructEnc
              ? Math.min(0.84, 0.72 + frac * 0.12)
              : Math.min(0.72, 0.05 + frac * 0.67),
            detail: isStructEnc
              ? `Dyna-1 ESM-3 structure encoder: ${mb} / ${totMb} MB`
              : `Dyna-1 weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
          });
        },
      });
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        /integrity|sha-?256|manifest|mismatch/i.test(errMsg)
      ) {
        await evictCachedUrls(fetchedUrls);
        return await loadOnce(true);
      }
      throw err;
    }
  };

  try {
    engine = await loadOnce(false);
    loadedPrecision = precision;
    return engine;
  } catch (err) {
    if (ownedDevice) {
      try {
        ownedDevice.destroy();
      } catch {
        /* ignore */
      }
      ownedDevice = null;
    }
    throw err;
  }
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'dispose') {
      activeAbort?.abort();
      activeAbort = null;
      try {
        engine?.destroy();
      } catch {
        /* ignore */
      }
      try {
        ownedDevice?.destroy();
      } catch {
        /* ignore */
      }
      engine = null;
      ownedDevice = null;
      loadedPrecision = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const p = msg.params ?? {};
      const precision = normalizePrecision(p.precision);
      const requestedInputMode = normalizeDyna1InputMode(p.inputMode);
      const maxSequences = Math.max(1, Math.min(8, Math.round(Number(p.maxSequences ?? 1))));

      const seqSet = msg.inputs?.sequences as SequenceSet | undefined;
      const struct = msg.inputs?.structure as StructureArtifact | undefined;
      const backbone3 = extractDyna1Backbone3Coords(struct);

      let sourceCandidates: SequenceCandidate[] = [];
      const parentStructureId: string | undefined = struct?.id ?? seqSet?.parentStructureId;

      if (requestedInputMode === 'struct-only' && struct) {
        sourceCandidates = [
          {
            id: struct.id,
            key: struct.designKey ?? struct.id,
            sequence: struct.sequence || 'A'.repeat(struct.length),
            chainLengths: struct.chainLengths,
            score: null,
            globalScore: null,
            sourceModelId: struct.sourceModelId,
            sourceStructureId: struct.id,
          },
        ];
      } else if (seqSet && seqSet.candidates.length > 0) {
        sourceCandidates = seqSet.candidates.slice(0, maxSequences);
      } else if (struct && struct.sequence) {
        sourceCandidates = [
          {
            id: struct.id,
            key: struct.designKey ?? struct.id,
            sequence: struct.sequence,
            chainLengths: struct.chainLengths,
            score: null,
            globalScore: null,
            sourceModelId: struct.sourceModelId,
            sourceStructureId: struct.id,
          },
        ];
      } else {
        throw new Error(
          'Dyna-1 requires an upstream StructureArtifact or SequenceSet.',
        );
      }

      if (requestedInputMode === 'struct-only' && backbone3 && (backbone3.length < 1 || backbone3.length > 1024)) {
        throw new Error(
          `Dyna-1 supports 1–1024 residues (structure "${struct?.label ?? 'input'}" has ${backbone3.length}).`,
        );
      }
      const cleanedSeqs = sourceCandidates.map((c) =>
        requestedInputMode === 'struct-only'
          ? c.sequence.replace(/[\s:/,-]+/g, '').toUpperCase() || 'X'.repeat(backbone3?.length ?? 0)
          : cleanCanonicalSequence(c.sequence, c.id),
      );
      const resolvedModes: Dyna1ResolvedMode[] = cleanedSeqs.map((seq) =>
        resolveDyna1Mode(requestedInputMode, seq.length, backbone3),
      );

      const activeEngine = await ensureEngine(msg.id, precision);
      const controller = new AbortController();
      activeAbort = controller;

      const scoredCandidates: SequenceCandidate[] = [];
      let primaryProbs = new Float32Array(0);
      let primarySeq = cleanedSeqs[0];
      let primaryMode: Dyna1ResolvedMode = resolvedModes[0];
      let primaryMean = 0;
      let primaryMin = 0;
      let primaryMax = 0;
      let primaryFlexFrac = 0;

      const needsStructEncoding =
        backbone3 !== null &&
        resolvedModes.some((m) => m === 'seq+struct' || m === 'struct-only');
      let cachedStructureTokens: Int32Array | null = null;

      try {
        if (needsStructEncoding && backbone3 && typeof activeEngine.encodeStructure === 'function') {
          const encRes = await activeEngine.encodeStructure(backbone3.coords, {
            residueIndex: backbone3.residueIndex,
            signal: controller.signal,
            onProgress: (prog: { phase?: string; loaded?: number; total?: number }) => {
              if (prog.phase !== 'structure-encoder') return;
              const loaded = prog.loaded ?? 0;
              const total = prog.total ?? 0;
              const frac = total > 0 ? loaded / total : 0.4;
              const mb = (loaded / 1024 / 1024).toFixed(1);
              const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'weights',
                progress: Math.min(0.84, 0.72 + frac * 0.12),
                detail: `Dyna-1 ESM-3 structure encoder: ${mb} / ${totMb} MB`,
              });
            },
          });
          cachedStructureTokens =
            encRes?.structureTokens instanceof Int32Array
              ? encRes.structureTokens
              : encRes instanceof Int32Array
                ? encRes
                : null;
        }

        for (let i = 0; i < sourceCandidates.length; i++) {
          const cand = sourceCandidates[i];
          const seq = cleanedSeqs[i];
          const mode = resolvedModes[i];
          const useStruct = (mode === 'seq+struct' || mode === 'struct-only') && backbone3 !== null;

          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'score',
            progress: Math.min(0.95, 0.84 + (i / sourceCandidates.length) * 0.11),
            detail: `Dyna-1 (${mode}) on ${cand.id} (${useStruct ? backbone3!.length : seq.length} aa)...`,
          });

          const res = await activeEngine.predict(
            mode === 'struct-only' ? null : seq,
            {
              mode,
              ...(useStruct
                ? cachedStructureTokens
                  ? { structureTokens: cachedStructureTokens }
                  : { structureCoords: backbone3!.coords, residueIndex: backbone3!.residueIndex }
                : {}),
              signal: controller.signal,
              onProgress: (prog: { phase?: string; loaded?: number; total?: number }) => {
                if (prog.phase !== 'structure-encoder') return;
                const loaded = prog.loaded ?? 0;
                const total = prog.total ?? 0;
                const frac = total > 0 ? loaded / total : 0.4;
                const mb = (loaded / 1024 / 1024).toFixed(1);
                const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
                self.postMessage({
                  id: msg.id,
                  type: 'progress',
                  phase: 'weights',
                  progress: Math.min(0.84, 0.72 + frac * 0.12),
                  detail: `Dyna-1 ESM-3 structure encoder: ${mb} / ${totMb} MB`,
                });
              },
            },
          );
          const probs = new Float32Array(res.probs as Float32Array);

          let sum = 0;
          let mn = Infinity;
          let mx = -Infinity;
          let flexCount = 0;
          for (let r = 0; r < probs.length; r++) {
            const v = probs[r];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            if (v >= 0.5) flexCount++;
          }
          const mean = sum / Math.max(1, probs.length);
          const minV = Number.isFinite(mn) ? mn : mean;
          const maxV = Number.isFinite(mx) ? mx : mean;
          const flexFrac = flexCount / Math.max(1, probs.length);

          if (i === 0) {
            primaryProbs = probs;
            primarySeq = seq.length === probs.length ? seq : 'X'.repeat(probs.length);
            primaryMode = mode;
            primaryMean = mean;
            primaryMin = minV;
            primaryMax = maxV;
            primaryFlexFrac = flexFrac;
          }

          scoredCandidates.push({
            ...cand,
            sequence: seq,
            // Lower mean exchange probability = more rigid (aligns with Top-K min-sorting)
            score: mean,
            globalScore: maxV,
            sourceModelId: 'dyna1',
          });
        }
      } finally {
        if (activeAbort === controller) activeAbort = null;
      }

      const L = primaryProbs.length;
      const residueIds =
        struct?.sourceResidues && struct.sourceResidues.length === L
          ? struct.sourceResidues.map((r) => `${r.authChainId}${r.resNum}${r.insCode ?? ''}`)
          : Array.from({ length: L }, (_, i) => `A${i + 1}`);

      const runHash = fnv1a(
        JSON.stringify({
          seq: primaryMode === 'struct-only' ? '' : primarySeq,
          structId: primaryMode !== 'seq-only' ? struct?.id ?? '' : '',
          mode: primaryMode,
          count: scoredCandidates.length,
          precision,
        }),
      );

      const modeTag =
        primaryMode === 'seq+struct'
          ? 'seq+struct'
          : primaryMode === 'struct-only'
            ? 'struct-only'
            : 'seq-only';

      const track: ResidueTrackArtifact = {
        kind: 'ResidueTrack',
        id: `dyna1-track-${precision}-${runHash}`,
        label: `Dyna-1 Exchange Prob (mean ${primaryMean.toFixed(2)} · ${Math.round(primaryFlexFrac * 100)}% flex · ${modeTag} · ${precision.toUpperCase()})`,
        sourceModelId: 'dyna1',
        parentStructureId,
        quantity: 'dynamics-prob',
        unit: 'p(exch)',
        range: [0, 1],
        threshold: 0.5,
        residues: primarySeq,
        residueIds,
        values: primaryProbs,
        summary: {
          mean: primaryMean,
          min: primaryMin,
          max: primaryMax,
          fractionAboveThreshold: primaryFlexFrac,
        },
      };

      const sequences: SequenceSet = {
        kind: 'SequenceSet',
        id: `dyna1-seqs-${precision}-${runHash}`,
        parentStructureId: parentStructureId ?? 'none',
        sourceStructureId: parentStructureId ?? 'none',
        sourceModelId: 'dyna1',
        candidates: scoredCandidates,
      };

      const outputs: Record<string, unknown> = { track, sequences };
      const transferableViews: Array<ArrayBufferView | undefined> = [track.values];

      // When an upstream 3D structure is connected and length-matched, paint dynamics probabilities onto its B-factors
      if (struct && (struct.pdbText || struct.cifText) && struct.length === L) {
        const structText = struct.cifText || struct.pdbText;
        const sourceResidues =
          struct.sourceResidues && struct.sourceResidues.length > 0
            ? struct.sourceResidues
            : structText
              ? parseStructureText(structText).sourceResidues
              : [];
        const bfactors = new Float32Array(Math.max(sourceResidues.length, L)).fill(Number.NaN);
        const nPaint = Math.min(bfactors.length, primaryProbs.length);
        for (let i = 0; i < nPaint; i++) {
          bfactors[i] = primaryProbs[i];
        }
        const paintedPdb = struct.pdbText
          ? paintStructureBfactors(struct.pdbText, bfactors.subarray(0, sourceResidues.length || L))
          : struct.pdbText;
        const paintedCif = struct.cifText
          ? paintStructureBfactors(struct.cifText, bfactors.subarray(0, sourceResidues.length || L))
          : undefined;
        const caCopy = new Float32Array(struct.caCoords);
        const bbCopy = new Float32Array(struct.backbone4Coords);
        const obsCopy = struct.observedBackboneMask
          ? new Uint8Array(struct.observedBackboneMask)
          : undefined;

        const paintedStructure: StructureArtifact = {
          ...struct,
          id: `dyna1-struct-${struct.id}-${precision}-${runHash}`,
          label: `${struct.label} (Dyna-1 · mean ${primaryMean.toFixed(2)})`,
          parentStructureId: struct.id,
          caCoords: caCopy,
          backbone4Coords: bbCopy,
          ...(obsCopy ? { observedBackboneMask: obsCopy } : {}),
          pdbText: paintedPdb,
          ...(paintedCif !== undefined ? { cifText: paintedCif } : {}),
          bfactorSemantics: 'dynamics-prob',
          provenance: {
            ...struct.provenance,
            quantization: precision,
          },
        };
        outputs.structure = paintedStructure;
        transferableViews.push(caCopy, bbCopy, obsCopy);
      }

      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          outputs,
          elapsedMs: performance.now() - t0,
        },
        collectOwnedTransferables(transferableViews),
      );
    }
  } catch (err) {
    self.postMessage({
      id: msg.id,
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
    });
  }
};

export {};
