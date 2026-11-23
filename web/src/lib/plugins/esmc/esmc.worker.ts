/// <reference lib="webworker" />
// Layer 3 Plugin Worker — esmc (ramithuh/esmc-webgpu, MIT · ESM-C 300M)
// Computes zero-shot L×20 log-likelihood ratios (wildtype or masked-marginal),
// naturalness / pseudo-perplexity residue tracks, and masked-position sequence fill.
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress, and disposal.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  collectOwnedTransferables,
  createWorkerHarness,
  fnv1a,
  type WorkerRunContext,
} from '$lib/kernel';
import { paintStructureBfactors } from '$lib/protein-ops/bfactor-paint';
import { parseStructureText } from '$lib/protein-ops/pdb';
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

function formatResidueId(chain: string, resNum: number, ins: string): string {
  if (!chain) return `${resNum}${ins}`;
  return chain.length > 1 || !/^[A-Za-z]$/.test(chain)
    ? `${chain}:${resNum}${ins}`
    : `${chain}${resNum}${ins}`;
}

function resolveCandidateChainSplits(
  rawSeq: string,
  chainLengths: readonly number[] | undefined,
  cleanSeq: string,
): { chainSeqs: string[]; chainLengths: number[] } {
  const parts = rawSeq
    .split(/[:/]+/)
    .map((part) => part.replace(/[\s,-]+/g, '').toUpperCase())
    .filter(Boolean);
  if (parts.length > 1 && parts.reduce((s, part) => s + part.length, 0) === cleanSeq.length) {
    return { chainSeqs: parts, chainLengths: parts.map((p) => p.length) };
  }
  if (
    chainLengths &&
    chainLengths.length > 1 &&
    chainLengths.every((n) => Number.isInteger(n) && n > 0) &&
    chainLengths.reduce((s, n) => s + n, 0) === cleanSeq.length
  ) {
    const seqs: string[] = [];
    let off = 0;
    for (const len of chainLengths) {
      seqs.push(cleanSeq.slice(off, off + len));
      off += len;
    }
    return { chainSeqs: seqs, chainLengths: [...chainLengths] };
  }
  return { chainSeqs: [cleanSeq], chainLengths: [cleanSeq.length] };
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

async function ensureEngine(ctx: WorkerRunContext, precision: Precision) {
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
    ctx.progress('init', 0.02, 'Loading ESM-C 300M WebGPU modules...');
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

    ctx.progress('weights', 0.05, `Fetching ESM-C 300M (${precision.toUpperCase()}) weights...`);

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
          ctx.progress(
            'weights',
            Math.min(0.75, 0.05 + frac * 0.7),
            `ESM-C 300M weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
          );
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

createWorkerHarness({
  async run(ctx) {
    const p = ctx.params ?? {};
    const mode = normalizeMode(p.mode);
    const precision = normalizePrecision(p.precision);
    const maskPositionsRaw = String(p.maskPositions ?? '').trim();
    const maxSequences = Math.max(1, Math.min(8, Math.round(Number(p.maxSequences ?? 1))));

    const seqSet = ctx.inputs?.sequences as SequenceSet | undefined;
    const struct = ctx.inputs?.structure as StructureArtifact | undefined;

    let sourceCandidates: SequenceCandidate[] = [];
    const parentStructureId: string | undefined = struct?.id ?? seqSet?.parentStructureId;

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

    const activeEngine = await ensureEngine(ctx, precision);

    const scoredCandidates: SequenceCandidate[] = [];
    let primaryLlr = new Float32Array(0);
    let primaryWtLogProbs = new Float32Array(0);
    let primarySeq = cleanedSeqs[0];
    let primaryNaturalness = 0;
    let primaryPpl = 0;

    for (let i = 0; i < sourceCandidates.length; i++) {
      const cand = sourceCandidates[i];
      const seq = cleanedSeqs[i];
      const positions = mode === 'fill' ? parseMaskPositions(maskPositionsRaw, seq.length) : undefined;
      const { chainSeqs, chainLengths: effChainLens } = resolveCandidateChainSplits(
        cand.sequence,
        cand.chainLengths,
        seq,
      );

      ctx.progress(
        'score',
        Math.min(0.95, 0.78 + (i / sourceCandidates.length) * 0.18),
        `ESM-C 300M (${mode}) on ${cand.id} (${seq.length} aa)...`,
      );

      // Score multi-chain inputs per chain so EVERY chain gets proper BOS + EOS token context
      // (avoids middle chains lacking BOS/EOS boundaries).
      const runChainAware = async (
        targetSeq: string,
        targetChains: string[],
        runMode: EsmcMode,
        maskPos?: number[],
      ): Promise<{
        wtLogProbs: Float32Array<ArrayBuffer>;
        llr: Float32Array<ArrayBuffer>;
        naturalness: number;
        pseudoPerplexity: number;
        filledSequence?: string;
      }> => {
        const K20 = CANONICAL_ALPHABET.length;
        if (targetChains.length <= 1) {
          const res = await activeEngine.run(targetSeq, {
            mode: runMode,
            ...(maskPos ? { positions: maskPos } : {}),
            signal: ctx.signal,
            onProgress: (prog: { loaded?: number; total?: number }) => {
              const frac = prog.total ? (prog.loaded ?? 0) / prog.total : 0.5;
              ctx.progress(
                'score',
                Math.min(0.96, 0.78 + ((i + frac) / sourceCandidates.length) * 0.18),
                `ESM-C 300M masked sweep (${cand.id}): ${prog.loaded ?? 0}/${prog.total ?? targetSeq.length}`,
              );
            },
          });
          const wtLp = new Float32Array(res.wtLogProbs as Float32Array);
          const llrArr = new Float32Array(res.llr as Float32Array);
          let sumLp = 0;
          for (let r = 0; r < wtLp.length; r++) sumLp += wtLp[r];
          const nat =
            typeof res.naturalness === 'number' && Number.isFinite(res.naturalness)
              ? res.naturalness
              : sumLp / Math.max(1, wtLp.length);
          const ppl =
            typeof res.pseudoPerplexity === 'number' && Number.isFinite(res.pseudoPerplexity)
              ? res.pseudoPerplexity
              : Math.exp(-nat);
          return {
            wtLogProbs: wtLp,
            llr: llrArr,
            naturalness: nat,
            pseudoPerplexity: ppl,
            ...(typeof res.filledSequence === 'string' ? { filledSequence: res.filledSequence } : {}),
          };
        }

        const mergedWt = new Float32Array(targetSeq.length);
        const mergedLlr = new Float32Array(targetSeq.length * K20);
        const filledParts: string[] = [];
        let offset = 0;
        for (let c = 0; c < targetChains.length; c++) {
          const cSeq = targetChains[c];
          const cMask = maskPos
            ? maskPos
                .filter((idx) => idx >= offset && idx < offset + cSeq.length)
                .map((idx) => idx - offset)
            : undefined;
          const chainMode: EsmcMode =
            runMode === 'fill' && (!cMask || cMask.length === 0) ? 'wildtype' : runMode;
          const res = await activeEngine.run(cSeq, {
            mode: chainMode,
            ...(chainMode === 'fill' && cMask ? { positions: cMask } : {}),
            signal: ctx.signal,
          });
          const wtLp = new Float32Array(res.wtLogProbs as Float32Array);
          const llrArr = new Float32Array(res.llr as Float32Array);
          mergedWt.set(wtLp.subarray(0, cSeq.length), offset);
          mergedLlr.set(llrArr.subarray(0, cSeq.length * K20), offset * K20);
          filledParts.push(
            chainMode === 'fill' && typeof res.filledSequence === 'string'
              ? res.filledSequence
              : cSeq,
          );
          offset += cSeq.length;
        }
        let sumLp = 0;
        for (let r = 0; r < mergedWt.length; r++) sumLp += mergedWt[r];
        const nat = sumLp / Math.max(1, mergedWt.length);
        return {
          wtLogProbs: mergedWt,
          llr: mergedLlr,
          naturalness: nat,
          pseudoPerplexity: Math.exp(-nat),
          ...(runMode === 'fill' ? { filledSequence: filledParts.join('') } : {}),
        };
      };

      const initialRes = await runChainAware(seq, chainSeqs, mode, positions);
      const outSeq =
        mode === 'fill' && typeof initialRes.filledSequence === 'string'
          ? initialRes.filledSequence
          : seq;
      const seqChanged = mode === 'fill' && outSeq !== seq;

      // In fill mode, rescore the newly filled sequence so candidate ranking reflects the filled residues
      // rather than the pre-fill input tokens, and assign a fresh derived sequenceKey when mutated.
      let rankNaturalness = initialRes.naturalness;
      let rankPpl = initialRes.pseudoPerplexity;
      if (seqChanged) {
        const filledChains: string[] = [];
        let off = 0;
        for (const len of effChainLens) {
          filledChains.push(outSeq.slice(off, off + len));
          off += len;
        }
        const rescored = await runChainAware(outSeq, filledChains, 'wildtype');
        rankNaturalness = rescored.naturalness;
        rankPpl = rescored.pseudoPerplexity;
      }

      if (i === 0) {
        primaryLlr = initialRes.llr;
        primaryWtLogProbs = initialRes.wtLogProbs;
        primarySeq = seq;
        primaryNaturalness = initialRes.naturalness;
        primaryPpl = initialRes.pseudoPerplexity;
      }

      const derivedSeqKey = seqChanged
        ? `${cand.sequenceKey ?? cand.key ?? cand.id}:esmc-fill:${fnv1a(outSeq)}`
        : cand.sequenceKey;

      scoredCandidates.push({
        ...cand,
        id: mode === 'fill' ? `${cand.id}-esmc-fill` : cand.id,
        ...(seqChanged ? { key: derivedSeqKey, sequenceKey: derivedSeqKey } : {}),
        ...(effChainLens.length > 1 ? { chainLengths: effChainLens } : {}),
        sequence: outSeq,
        // Store negative naturalness (= mean NLL) so Top-K min-sorting picks the most natural sequence
        score: -rankNaturalness,
        globalScore: rankPpl,
        sourceModelId: 'esmc',
      });
    }

    const L = primarySeq.length;
    const K = CANONICAL_ALPHABET.length;
    const structText = struct?.cifText || struct?.pdbText;
    const sourceResidues =
      struct?.sourceResidues && struct.sourceResidues.length > 0
        ? struct.sourceResidues
        : structText
          ? parseStructureText(structText).sourceResidues
          : [];
    let residueIds: string[];
    if (sourceResidues.length === L) {
      residueIds = sourceResidues.map((r) =>
        formatResidueId(r.authChainId, r.resNum, r.insCode ?? ''),
      );
    } else {
      const cLens = sourceCandidates[0]?.chainLengths;
      if (cLens && cLens.length > 1 && cLens.reduce((a, b) => a + b, 0) === L) {
        residueIds = [];
        for (let cIdx = 0; cIdx < cLens.length; cIdx++) {
          const ch = String.fromCharCode(65 + cIdx);
          for (let k = 0; k < cLens[cIdx]; k++) {
            residueIds.push(`${ch}${k + 1}`);
          }
        }
      } else {
        residueIds = Array.from({ length: L }, (_, i) => `A${i + 1}`);
      }
    }
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
        candidates: scoredCandidates.map((c) => ({
          id: c.id,
          key: c.sequenceKey ?? c.key ?? c.id,
          seq: c.sequence,
          chains: c.chainLengths ?? null,
        })),
        mode,
        precision,
        maskPositionsRaw,
        structureId: struct?.id ?? null,
      }),
    );

    const isTrueMaskedPpl = mode === 'masked';
    const mutations: MutationMatrixArtifact = {
      kind: 'MutationMatrix',
      id: `esmc-llr-${mode}-${precision}-${runHash}`,
      label: isTrueMaskedPpl
        ? `ESM-C 300M Masked LLR (${precision.toUpperCase()} · PPPL ${primaryPpl.toFixed(2)})`
        : mode === 'fill'
          ? `ESM-C 300M Fill LLR (${precision.toUpperCase()})`
          : `ESM-C 300M WT LLR (${precision.toUpperCase()} · 1-pass approx)`,
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
      label: isTrueMaskedPpl
        ? `ESM-C Masked Log-Prob (Naturalness ${primaryNaturalness.toFixed(3)} · PPPL ${primaryPpl.toFixed(2)})`
        : `ESM-C WT Log-Prob (Mean ${primaryNaturalness.toFixed(3)} · 1-pass WT approx)`,
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
        ...(isTrueMaskedPpl ? { pseudoPerplexity: primaryPpl } : {}),
      },
    };

    const sequences: SequenceSet = {
      kind: 'SequenceSet',
      id: `esmc-seqs-${mode}-${runHash}`,
      parentStructureId: parentStructureId ?? 'none',
      sourceStructureId: parentStructureId ?? 'none',
      ...(seqSet?.sourceStructureIds ? { sourceStructureIds: seqSet.sourceStructureIds } : {}),
      ...(seqSet?.collectionKind ? { collectionKind: seqSet.collectionKind } : {}),
      ...(seqSet?.lockedChains?.length ? { lockedChains: seqSet.lockedChains } : {}),
      ...(seqSet?.chainRoles ? { chainRoles: seqSet.chainRoles } : {}),
      sourceModelId: 'esmc',
      candidates: scoredCandidates,
    };

    const outputs: Record<string, unknown> = { mutations, track, sequences };
    const transferableViews: Array<ArrayBufferView | undefined> = [
      mutations.values,
      mutations.meanPerPosition,
      track.values,
    ];

    if (struct && (struct.pdbText || struct.cifText) && sourceResidues.length === L) {
      const paintValues = new Float32Array(L);
      for (let i = 0; i < L; i++) {
        paintValues[i] = Number.isFinite(meanPerPosition[i])
          ? meanPerPosition[i]
          : primaryWtLogProbs[i];
      }
      const paintedPdb = struct.pdbText
        ? paintStructureBfactors(struct.pdbText, paintValues)
        : struct.pdbText;
      const paintedCif = struct.cifText
        ? paintStructureBfactors(struct.cifText, paintValues)
        : undefined;
      const caCopy = new Float32Array(struct.caCoords);
      const bbCopy = new Float32Array(struct.backbone4Coords);
      const obsCopy = struct.observedBackboneMask
        ? new Uint8Array(struct.observedBackboneMask)
        : undefined;

      const paintedStructure: StructureArtifact = {
        ...struct,
        id: `esmc-struct-${struct.id}-${mode}-${precision}-${runHash}`,
        label: `${struct.label} (ESM-C ${mode === 'masked' ? 'Masked' : mode === 'fill' ? 'Fill' : 'WT'} LLR)`,
        parentStructureId: struct.id,
        caCoords: caCopy,
        backbone4Coords: bbCopy,
        ...(obsCopy ? { observedBackboneMask: obsCopy } : {}),
        pdbText: paintedPdb,
        ...(paintedCif !== undefined ? { cifText: paintedCif } : {}),
        bfactorSemantics: 'llr-fitness',
        provenance: {
          ...struct.provenance,
          quantization: precision,
        },
      };
      outputs.structure = paintedStructure;
      transferableViews.push(caCopy, bbCopy, obsCopy);
    }

    const transferables = collectOwnedTransferables(transferableViews);
    return { outputs, transferables };
  },

  async dispose() {
    try {
      engine?.destroy();
    } catch {
      /* ignore */
    }
    engine = null;
    loadedPrecision = null;
  },
});

export {};
