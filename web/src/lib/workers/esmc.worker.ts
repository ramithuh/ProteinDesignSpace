/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/esmc-webgpu (MIT · ESM-C 300M)
// Computes zero-shot L×20 log-likelihood ratios (wildtype or masked-marginal),
// naturalness / pseudo-perplexity residue tracks, and masked-position sequence fill.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import type {
  MutationMatrixArtifact,
  ResidueTrackArtifact,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/esmc-webgpu/resolve/main/esmc-300m';
const CANONICAL_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY';

type Precision = 'int8' | 'fp16';
type EsmcMode = 'wildtype' | 'masked' | 'fill';

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
  return val === 'fp16' ? 'fp16' : 'int8';
}

function normalizeMode(raw: unknown): EsmcMode {
  const val = String(raw ?? 'wildtype').trim().toLowerCase();
  if (val === 'masked' || val === 'fill') return val;
  return 'wildtype';
}

function parseMaskPositions(raw: unknown, seqLen: number): number[] {
  const text = String(raw ?? '').trim();
  if (!text) return [];
  const positions = new Set<number>();
  for (const tok of text.split(/[,\s]+/).filter(Boolean)) {
    const m = /^(\d+)(?:-(\d+))?$/.exec(tok);
    if (!m) {
      throw new Error(`Invalid mask position "${tok}". Use 1-based numbers or ranges like "5, 12-15".`);
    }
    const lo = Number(m[1]);
    const hi = m[2] ? Number(m[2]) : lo;
    if (lo < 1 || hi < lo || hi > seqLen) {
      throw new Error(`Mask position "${tok}" is out of range for a ${seqLen}-aa sequence.`);
    }
    for (let pos = lo; pos <= hi; pos++) {
      positions.add(pos - 1); // 0-based for ESM-C
    }
  }
  return [...positions].sort((a, b) => a - b);
}

function cleanCanonicalSequence(raw: string, label: string): string {
  const seq = raw.replace(/[\s:/,-]+/g, '').toUpperCase();
  if (seq.length < 5 || seq.length > 1022) {
    throw new Error(
      `ESM-C 300M supports 5–1022 canonical amino acids ("${label}" has ${seq.length}).`,
    );
  }
  if (!/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(seq)) {
    throw new Error(
      `ESM-C 300M requires canonical 20-AA protein sequences in "${label}" (non-canonical letters found).`,
    );
  }
  return seq;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let esmcMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let loadedPrecision: Precision | null = null;
let activeAbort: AbortController | null = null;

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

  if (!esmcMod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading ESM-C 300M WebGPU modules...',
    });
    const moduleUrl = new URL('/vendor/esmc/webgpu/index.js', self.location.origin).href;
    esmcMod = await import(/* @vite-ignore */ moduleUrl);
  }

  const envBaseUrl = (import.meta.env.VITE_ESMC_WEIGHTS_URL as string | undefined)?.trim();
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
      detail: `Fetching ESM-C 300M (${precision.toUpperCase()}) weights...`,
    });

    try {
      return await esmcMod.createEsmcEngine({
        baseUrl,
        precision,
        cache: 'none',
        useF16: false,
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { loaded?: number; total?: number }) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac = total > 0 ? loaded / total : 0.4;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.75, 0.05 + frac * 0.7),
            detail: `ESM-C 300M weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
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

  engine = await loadOnce(false);
  loadedPrecision = precision;
  return engine;
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
      engine = null;
      loadedPrecision = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const p = msg.params ?? {};
      const mode = normalizeMode(p.mode);
      const precision = normalizePrecision(p.precision);
      const maskPositionsRaw = String(p.maskPositions ?? '').trim();
      const maxSequences = Math.max(1, Math.min(8, Math.round(Number(p.maxSequences ?? 1))));

      const seqSet = msg.inputs?.sequences as SequenceSet | undefined;
      const struct = msg.inputs?.structure as StructureArtifact | undefined;

      let sourceCandidates: SequenceCandidate[] = [];
      let parentStructureId: string | undefined = struct?.id ?? seqSet?.parentStructureId;

      if (seqSet && seqSet.candidates.length > 0) {
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
          'ESM-C 300M requires an upstream SequenceSet (or StructureArtifact to extract its sequence).',
        );
      }

      const cleanedSeqs = sourceCandidates.map((c) => cleanCanonicalSequence(c.sequence, c.id));
      const fillPositions =
        mode === 'fill' ? parseMaskPositions(maskPositionsRaw, cleanedSeqs[0].length) : [];
      if (mode === 'fill' && fillPositions.length === 0) {
        throw new Error(
          'ESM-C fill mode requires 1-based mask positions (e.g. "5, 12-15").',
        );
      }

      const activeEngine = await ensureEngine(msg.id, precision);
      const controller = new AbortController();
      activeAbort = controller;

      const scoredCandidates: SequenceCandidate[] = [];
      let primaryLlr = new Float32Array(0);
      let primaryWtLogProbs = new Float32Array(0);
      let primarySeq = cleanedSeqs[0];
      let primaryNaturalness = 0;
      let primaryPpl = 0;

      try {
        for (let i = 0; i < sourceCandidates.length; i++) {
          const cand = sourceCandidates[i];
          const seq = cleanedSeqs[i];
          const positions = mode === 'fill' ? parseMaskPositions(maskPositionsRaw, seq.length) : undefined;

          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'score',
            progress: Math.min(0.95, 0.78 + (i / sourceCandidates.length) * 0.18),
            detail: `ESM-C 300M (${mode}) on ${cand.id} (${seq.length} aa)...`,
          });

          const res = await activeEngine.run(seq, {
            mode,
            ...(positions ? { positions } : {}),
            signal: controller.signal,
            onProgress: (prog: { loaded?: number; total?: number }) => {
              const frac = prog.total ? (prog.loaded ?? 0) / prog.total : 0.5;
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: 'score',
                progress: Math.min(0.96, 0.78 + ((i + frac) / sourceCandidates.length) * 0.18),
                detail: `ESM-C 300M masked sweep (${cand.id}): ${prog.loaded ?? 0}/${prog.total ?? seq.length}`,
              });
            },
          });

          const wtLogProbs = new Float32Array(res.wtLogProbs as Float32Array);
          const llr = new Float32Array(res.llr as Float32Array);
          let sumLogP = 0;
          for (let r = 0; r < wtLogProbs.length; r++) sumLogP += wtLogProbs[r];
          const naturalness =
            typeof res.naturalness === 'number' && Number.isFinite(res.naturalness)
              ? res.naturalness
              : sumLogP / Math.max(1, wtLogProbs.length);
          const pseudoPerplexity =
            typeof res.pseudoPerplexity === 'number' && Number.isFinite(res.pseudoPerplexity)
              ? res.pseudoPerplexity
              : Math.exp(-naturalness);

          if (i === 0) {
            primaryLlr = llr;
            primaryWtLogProbs = wtLogProbs;
            primarySeq = seq;
            primaryNaturalness = naturalness;
            primaryPpl = pseudoPerplexity;
          }

          const outSeq = mode === 'fill' && typeof res.filledSequence === 'string'
            ? res.filledSequence
            : seq;

          scoredCandidates.push({
            ...cand,
            id: mode === 'fill' ? `${cand.id}-esmc-fill` : cand.id,
            sequence: outSeq,
            // Store negative naturalness (= mean NLL) so Top-K min-sorting picks the most natural sequence
            score: -naturalness,
            globalScore: pseudoPerplexity,
            sourceModelId: 'esmc',
          });
        }
      } finally {
        if (activeAbort === controller) activeAbort = null;
      }

      const L = primarySeq.length;
      const K = CANONICAL_ALPHABET.length;
      const residueIds = Array.from({ length: L }, (_, i) => `A${i + 1}`);
      const meanPerPosition = new Float32Array(L);
      for (let r = 0; r < L; r++) {
        let sum = 0;
        let cnt = 0;
        const wt = primarySeq[r];
        for (let a = 0; a < K; a++) {
          if (CANONICAL_ALPHABET[a] === wt) continue;
          const v = primaryLlr[r * K + a];
          if (Number.isFinite(v)) {
            sum += v;
            cnt++;
          }
        }
        meanPerPosition[r] = cnt > 0 ? sum / cnt : Number.NaN;
      }

      let minWt = Infinity;
      let maxWt = -Infinity;
      for (let r = 0; r < primaryWtLogProbs.length; r++) {
        const v = primaryWtLogProbs[r];
        if (v < minWt) minWt = v;
        if (v > maxWt) maxWt = v;
      }

      const runHash = fnv1a(
        JSON.stringify({
          seq: primarySeq,
          count: scoredCandidates.length,
          mode,
          precision,
          maskPositionsRaw,
        }),
      );

      const mutations: MutationMatrixArtifact = {
        kind: 'MutationMatrix',
        id: `esmc-llr-${mode}-${precision}-${runHash}`,
        label: `ESM-C 300M ${mode === 'masked' ? 'Masked LLR' : mode === 'fill' ? 'Fill LLR' : 'WT LLR'} (${precision.toUpperCase()} · PPPL ${primaryPpl.toFixed(2)})`,
        sourceModelId: 'esmc',
        parentStructureId,
        metric: 'llr-fitness',
        unit: 'LLR',
        direction: 'higher-is-better',
        residues: primarySeq,
        residueIds,
        alphabet: CANONICAL_ALPHABET,
        values: primaryLlr,
        meanPerPosition,
        provenance: {
          quantization: precision,
        },
      };

      const track: ResidueTrackArtifact = {
        kind: 'ResidueTrack',
        id: `esmc-track-${mode}-${precision}-${runHash}`,
        label: `ESM-C WT Log-Prob (Naturalness ${primaryNaturalness.toFixed(3)} · PPPL ${primaryPpl.toFixed(2)})`,
        sourceModelId: 'esmc',
        parentStructureId,
        quantity: 'wt-log-prob',
        unit: 'log p',
        residues: primarySeq,
        residueIds,
        values: primaryWtLogProbs,
        summary: {
          mean: primaryNaturalness,
          min: Number.isFinite(minWt) ? minWt : primaryNaturalness,
          max: Number.isFinite(maxWt) ? maxWt : primaryNaturalness,
          pseudoPerplexity: primaryPpl,
        },
      };

      const sequences: SequenceSet = {
        kind: 'SequenceSet',
        id: `esmc-seqs-${mode}-${runHash}`,
        parentStructureId: parentStructureId ?? 'none',
        sourceStructureId: parentStructureId ?? 'none',
        sourceModelId: 'esmc',
        candidates: scoredCandidates,
      };

      const transferables = collectOwnedTransferables([
        mutations.values,
        mutations.meanPerPosition,
        track.values,
      ]);

      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          outputs: { mutations, track, sequences },
          elapsedMs: performance.now() - t0,
        },
        transferables,
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
