/// <reference lib="webworker" />
// Layer 3 Plugin Worker — esmc (ramithuh/esmc-webgpu, MIT · ESM-C 300M)
// Computes zero-shot L×20 log-likelihood ratios (wildtype or masked-marginal),
// naturalness / pseudo-perplexity residue tracks, and masked-position sequence fill.
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress, and disposal.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { createDeviceLostGuard, requestResilientWebGpuDevice } from '$lib/engine/webgpu-device';
import {
  collectOwnedTransferables,
  createWorkerHarness,
  fnv1a,
  type WorkerRunContext,
} from '$lib/kernel';
import { paintStructureBfactors } from '$lib/protein-ops/bfactor-paint';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { resolveEffectiveMaxSequences } from '$lib/engine/batch-flow';
import type {
  ExcludedCandidateRecord,
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
  if (raw == null || String(raw).trim() === '') return 'int8';
  const val = String(raw).trim().toLowerCase();
  if (val === 'fp16' || val === 'int8') return val;
  throw new Error(`Unsupported ESM-C precision "${String(raw)}" (supported: int8, fp16).`);
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
// Resilient GPUDevice injected into the engine: kept across precision switches, destroyed only on dispose.
let gpu: { device: GPUDevice; hasF16: boolean } | null = null;
const deviceLostGuard = createDeviceLostGuard();

async function ensureGpu(): Promise<{ device: GPUDevice; hasF16: boolean }> {
  if (gpu) return gpu;
  const { device, hasF16 } = await requestResilientWebGpuDevice({ optionalFeatures: ['shader-f16'] });
  const acquired = { device, hasF16 };
  gpu = acquired;
  deviceLostGuard.watch(device, () => {
    if (gpu !== acquired) return;
    gpu = null;
    // The engine's buffers and pipelines died with the device: the next run rebuilds it on a fresh one.
    loadedPrecision = null;
  });
  return acquired;
}

async function ensureEngine(ctx: WorkerRunContext, precision: Precision) {
  // `gpu` is cleared when its device is lost, even mid-load, so a cache hit always runs on a live device.
  if (engine && loadedPrecision === precision && gpu) return engine;

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
    const moduleUrl = new URL('/vendor/esmc/webgpu/index.js?v=2', self.location.origin).href;
    esmcMod = await import(/* @vite-ignore */ moduleUrl);
  }

  const envBaseUrl = (import.meta.env.VITE_ESMC_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;
  const { device, hasF16 } = await ensureGpu();

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
        device,
        // Native f16 weight reads on FP16 bundles when the device has shader-f16 (activations stay f32).
        useF16: hasF16,
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
    const capMode = p.capMode;
    const seqSet = ctx.inputs?.sequences as SequenceSet | undefined;
    const struct = ctx.inputs?.structure as StructureArtifact | undefined;

    const structItems: StructureArtifact[] =
      struct?.collection?.items && struct.collection.items.length > 1
        ? struct.collection.items
        : struct
          ? [struct]
          : [];
    const firstSeqLen =
      seqSet?.candidates?.[0]?.sequence?.length ?? structItems[0]?.sequence?.length ?? null;
    const capInfo = resolveEffectiveMaxSequences(
      'esmc',
      { ...p, capMode: p.capMode, maxSequences: p.maxSequences },
      { sequenceLength: firstSeqLen },
    );
    const maxSequences = capInfo.limit;

    let sourceCandidates: SequenceCandidate[] = [];
    let excludedCandidates: ExcludedCandidateRecord[] = [];
    let pairedStructures: StructureArtifact[] = [];
    const parentStructureId: string | undefined = struct?.id ?? seqSet?.parentStructureId;

    if (seqSet && seqSet.candidates.length > 0) {
      sourceCandidates = seqSet.candidates.slice(0, maxSequences);
      if (seqSet.candidates.length > maxSequences) {
        const reasonPrefix = capInfo.isPassBudgetCapped
          ? `Excluded by masked LLR pass budget (${maxSequences} of ${seqSet.candidates.length})`
          : `Excluded by maxSequences=${maxSequences}`;
        excludedCandidates = seqSet.candidates.slice(maxSequences).map((cand, offset) => {
          const index = maxSequences + offset;
          return {
            key: cand.key ?? cand.id,
            id: cand.id,
            index,
            reason: `${reasonPrefix} (candidate #${index + 1} of ${seqSet.candidates.length})`,
          };
        });
      }
    } else if (structItems.length > 0 && structItems.some((s) => Boolean(s.sequence))) {
      const validStructs = structItems.filter((s) => Boolean(s.sequence));
      pairedStructures = validStructs.slice(0, maxSequences);
      sourceCandidates = pairedStructures.map((s, idx) => ({
        id: s.id,
        key: s.designKey ?? s.id,
        ...(s.designKey ? { designKey: s.designKey } : {}),
        designIndex: s.designIndex ?? idx,
        sequence: s.sequence,
        chainLengths: s.chainLengths,
        score: null,
        globalScore: null,
        sourceModelId: s.sourceModelId,
        sourceStructureId: s.id,
      }));
      if (validStructs.length > maxSequences) {
        const reasonPrefix = capInfo.isPassBudgetCapped
          ? `Excluded by masked LLR pass budget (${maxSequences} of ${validStructs.length})`
          : `Excluded by maxSequences=${maxSequences}`;
        excludedCandidates = validStructs.slice(maxSequences).map((s, offset) => {
          const index = maxSequences + offset;
          return {
            key: s.designKey ?? s.id,
            id: s.id,
            index,
            reason: `${reasonPrefix} (structure #${index + 1} of ${validStructs.length})`,
          };
        });
      }
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
    const perCandidateRes: Array<{
      seq: string;
      llr: Float32Array<ArrayBuffer>;
      wtLogProbs: Float32Array<ArrayBuffer>;
    }> = [];
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
      perCandidateRes.push({ seq, llr: initialRes.llr, wtLogProbs: initialRes.wtLogProbs });

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

    const emittedSourceStructureIds =
      seqSet?.sourceStructureIds ??
      (pairedStructures.length > 1 ? pairedStructures.map((s) => s.id) : undefined);
    const emittedCollectionKind =
      seqSet?.collectionKind ??
      (pairedStructures.length > 1
        ? (struct?.collection?.collectionKind ?? 'independent-candidates')
        : undefined);

    const sequences: SequenceSet = {
      kind: 'SequenceSet',
      id: `esmc-seqs-${mode}-${runHash}`,
      parentStructureId: parentStructureId ?? 'none',
      sourceStructureId: parentStructureId ?? 'none',
      ...(emittedSourceStructureIds ? { sourceStructureIds: emittedSourceStructureIds } : {}),
      ...(emittedCollectionKind ? { collectionKind: emittedCollectionKind } : {}),
      ...(seqSet?.lockedChains?.length ? { lockedChains: seqSet.lockedChains } : {}),
      ...(seqSet?.chainRoles ? { chainRoles: seqSet.chainRoles } : {}),
      sourceModelId: 'esmc',
      candidates: scoredCandidates,
      ...(excludedCandidates.length > 0 ? { excludedCandidates } : {}),
    };

    const outputs: Record<string, unknown> = { mutations, track, sequences };
    const transferableViews: Array<ArrayBufferView | undefined> = [
      mutations.values,
      mutations.meanPerPosition,
      track.values,
    ];

    const buildPaintedStruct = (
      targetStruct: StructureArtifact,
      targetSeq: string,
      targetLlr: Float32Array,
      targetWt: Float32Array,
    ): StructureArtifact | null => {
      if (!targetStruct.pdbText && !targetStruct.cifText) return null;
      const tText = targetStruct.cifText || targetStruct.pdbText;
      const tRes =
        targetStruct.sourceResidues && targetStruct.sourceResidues.length > 0
          ? targetStruct.sourceResidues
          : tText
            ? parseStructureText(tText).sourceResidues
            : [];
      const tL = targetSeq.length;
      if (tRes.length !== tL) return null;
      const paintValues = new Float32Array(tL);
      for (let r = 0; r < tL; r++) {
        let s = 0;
        let cnt = 0;
        const wt = targetSeq[r];
        for (let a = 0; a < K; a++) {
          if (CANONICAL_ALPHABET[a] === wt) continue;
          const v = targetLlr[r * K + a];
          if (Number.isFinite(v)) {
            s += v;
            cnt++;
          }
        }
        paintValues[r] = cnt > 0 ? s / cnt : targetWt[r];
      }
      const paintedPdb = targetStruct.pdbText
        ? paintStructureBfactors(targetStruct.pdbText, paintValues)
        : targetStruct.pdbText;
      const paintedCif = targetStruct.cifText
        ? paintStructureBfactors(targetStruct.cifText, paintValues)
        : undefined;
      const caCopy = new Float32Array(targetStruct.caCoords);
      const bbCopy = new Float32Array(targetStruct.backbone4Coords);
      const obsCopy = targetStruct.observedBackboneMask
        ? new Uint8Array(targetStruct.observedBackboneMask)
        : undefined;
      transferableViews.push(caCopy, bbCopy, obsCopy);
      const baseCopy = { ...targetStruct };
      delete baseCopy.collection;
      return {
        ...baseCopy,
        id: `esmc-struct-${targetStruct.id}-${mode}-${precision}-${runHash}`,
        label: `${targetStruct.label} (ESM-C ${mode === 'masked' ? 'Masked' : mode === 'fill' ? 'Fill' : 'WT'} LLR)`,
        parentStructureId: targetStruct.id,
        caCoords: caCopy,
        backbone4Coords: bbCopy,
        ...(obsCopy ? { observedBackboneMask: obsCopy } : {}),
        pdbText: paintedPdb,
        ...(paintedCif !== undefined ? { cifText: paintedCif } : {}),
        bfactorSemantics: 'llr-fitness',
        provenance: {
          ...targetStruct.provenance,
          quantization: precision,
        },
      };
    };

    if (pairedStructures.length > 1) {
      const paintedItems: StructureArtifact[] = [];
      for (let idx = 0; idx < pairedStructures.length; idx++) {
        const resItem = perCandidateRes[idx];
        if (!resItem) continue;
        const painted = buildPaintedStruct(
          pairedStructures[idx],
          resItem.seq,
          resItem.llr,
          resItem.wtLogProbs,
        );
        if (painted) paintedItems.push(painted);
      }
      if (paintedItems.length === 1) {
        outputs.structure = paintedItems[0];
      } else if (paintedItems.length > 1) {
        const primaryPainted = paintedItems[0];
        outputs.structure = {
          ...primaryPainted,
          id: `esmc-batch-${paintedItems.length}-${mode}-${precision}-${runHash}`,
          collection: {
            kind: 'Collection',
            schema: 'denovo.collection/v1',
            id: `esmc-col-${mode}-${runHash}`,
            familyKey: `esmc-${mode}-${runHash}`,
            collectionKind: emittedCollectionKind ?? 'independent-candidates',
            requestedCount: paintedItems.length + excludedCandidates.length,
            complete: excludedCandidates.length === 0,
            items: paintedItems,
            keys: paintedItems.map((it, idx) => it.designKey ?? it.id ?? `item-${idx}`),
          },
        };
      }
    } else if (struct && (struct.pdbText || struct.cifText) && sourceResidues.length === L) {
      const painted = buildPaintedStruct(struct, primarySeq, primaryLlr, primaryWtLogProbs);
      if (painted) outputs.structure = painted;
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
    deviceLostGuard.destroy(gpu?.device);
    gpu = null;
  },
});

export {};
