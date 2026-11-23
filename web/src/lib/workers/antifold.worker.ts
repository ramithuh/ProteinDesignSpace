/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/antifold-esmif1-webgpu (BSD-3 / MIT)
// Supports AntiFold (antibody CDR-focused inverse folding) & ESM-IF1 (general GVP-Transformer).

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  computeChainLengths,
  detectAntibodyHeavyLightChains,
  normalizeAntiFoldNumberingParam,
  parseFixedPositionsAndChains,
  remapSequentialAntibodyRegions,
  resolveAntibodyNumbering,
  type AntiFoldNumberingParam,
  type NumberedChainInfo,
  type OrderedStructureResidue,
} from '$lib/engine/mpnn-spec';
import {
  computeStructurePayloadDigest,
  type CollectionMember,
} from '$lib/engine/collections';
import { deriveStableItemSeed, planMpnnFanout } from '$lib/engine/work-plan';
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/antifold-esmif1-webgpu/resolve/main';

type Variant = 'antifold' | 'esmif1';
type Precision = 'fp16' | 'int8';
type SamplingMode = 'independent' | 'autoregressive';

interface ParsedChainResidue {
  chain: string;
  number: number;
  insertion: string;
  name?: string;
}

interface ParsedChain {
  id: string;
  residues: ParsedChainResidue[];
  sequence: string;
  coords: Float32Array;
}

interface ParsedAntiFoldStructure {
  chains: ParsedChain[];
}

interface PreparedAntiFoldInput {
  L: number;
  N: number;
  tokens: Int32Array;
  sequence: string;
  residues: Array<ParsedChainResidue | null>;
  targetChain?: string;
  heavyChain?: string;
  lightChain?: string;
}

interface AntiFoldScoreResult {
  logLikelihood: number;
  perplexity: number;
  coordinateLogLikelihood: number;
  coordinatePerplexity: number;
  sequence: string;
  timingMs: number;
}

interface AntiFoldSampleResult {
  tokens: Int32Array;
  sequence: string;
  timingMs: number;
}

interface GvpEngine {
  predict(
    input: ParsedAntiFoldStructure,
    options?: Record<string, unknown>,
  ): Promise<AntiFoldScoreResult>;
  score(
    input: ParsedAntiFoldStructure,
    options?: Record<string, unknown>,
  ): Promise<AntiFoldScoreResult>;
  sample(
    input: ParsedAntiFoldStructure,
    options?: Record<string, unknown>,
  ): Promise<AntiFoldSampleResult>;
  sampleIndependent(
    input: ParsedAntiFoldStructure,
    options?: Record<string, unknown>,
  ): Promise<AntiFoldSampleResult>;
  destroy(): void;
}

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateUniforms(length: number, seed: number): Float32Array {
  const rng = mulberry32(seed);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const u = rng();
    // Strictly clamp to [0, 1) as validated by AntiFold/ESM-IF1 samplers
    out[i] = u >= 1 ? 0.9999999 : u < 0 ? 0 : u;
  }
  return out;
}

function normalizeVariant(raw: unknown): Variant {
  const val = String(raw ?? 'antifold').trim().toLowerCase();
  return val === 'esmif1' ? 'esmif1' : 'antifold';
}

function normalizePrecision(raw: unknown): Precision {
  const val = String(raw ?? 'fp16').trim().toLowerCase();
  return val === 'int8' ? 'int8' : 'fp16';
}

function normalizeSamplingMode(raw: unknown): SamplingMode {
  const val = String(raw ?? 'independent').trim().toLowerCase();
  return val === 'autoregressive' ? 'autoregressive' : 'independent';
}

function residueKey(chain: string, num: number, ins: string): string {
  return `${chain}:${num}:${ins}`;
}

interface ResolvedAntiFoldPlan {
  parsed: ParsedAntiFoldStructure;
  prepared: PreparedAntiFoldInput;
  combineOpts: Record<string, unknown>;
  fixed: Record<number, number>;
  sampleLength: number;
  propagatedLockedChains: string[];
  originalChainLengths: number[];
  originalOrderKeys: string[];
  nativeAaByKey: Map<string, string>;
}

function resolveAntiFoldPlan(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sharedMod: any,
  backbone: StructureArtifact,
  structureText: string,
  variant: Variant,
  samplingMode: SamplingMode,
  targetChainSpec: string,
  numberingParam: AntiFoldNumberingParam,
  regionsSpec: string,
  fixedPositionsSpec: string,
  unlockInherited: boolean,
): ResolvedAntiFoldPlan {
  const parsed = sharedMod.parseStructure(structureText) as ParsedAntiFoldStructure;
  const availableChains = parsed.chains.map((c) => c.id);
  const chainSet = new Set(availableChains);
  const upperToChain = new Map<string, string>();
  for (const ch of availableChains) {
    if (!upperToChain.has(ch.toUpperCase())) upperToChain.set(ch.toUpperCase(), ch);
  }

  const resolveChain = (raw: string): string | undefined => {
    if (!raw) return undefined;
    if (chainSet.has(raw)) return raw;
    return upperToChain.get(raw.toUpperCase());
  };

  // Preserve original structure residue ordering & native AAs for clean downstream alignment
  const originalOrderKeys: string[] = [];
  const originalChainList: string[] = [];
  const nativeAaByKey = new Map<string, string>();
  const orderedResidues: OrderedStructureResidue[] = [];
  const chainInfos: NumberedChainInfo[] = [];
  for (const ch of parsed.chains) {
    const infoResidues: Array<{ number: number; insertion: string; aa1: string }> = [];
    for (let i = 0; i < ch.residues.length; i++) {
      const r = ch.residues[i];
      const k = residueKey(r.chain, r.number, r.insertion);
      const aa1 = ch.sequence[i] ?? 'X';
      originalOrderKeys.push(k);
      originalChainList.push(r.chain);
      nativeAaByKey.set(k, aa1);
      orderedResidues.push({
        chain: r.chain,
        number: r.number,
        insertionCode: r.insertion ?? '',
      });
      infoResidues.push({
        number: r.number,
        insertion: r.insertion ?? '',
        aa1,
      });
    }
    chainInfos.push({ id: ch.id, residues: infoResidues });
  }
  const originalChainLengths = computeChainLengths(originalChainList);

  // Resolve inherited & user locked chains (pass 1: collect whole-chain locks before target resolution)
  const defaultChain = availableChains[0] ?? 'A';
  const { lockedChains: preUserLockedChains } =
    parseFixedPositionsAndChains(fixedPositionsSpec, defaultChain, orderedResidues);

  const lockedChainSet = new Set<string>();
  for (const ch of preUserLockedChains) {
    const resolved = resolveChain(ch);
    if (resolved) lockedChainSet.add(resolved);
  }
  if (!unlockInherited && Array.isArray(backbone.lockedChains)) {
    for (const rawCh of backbone.lockedChains) {
      const resolved = resolveChain(String(rawCh).trim());
      if (resolved) lockedChainSet.add(resolved);
    }
  }

  // Resolve targetChain, heavyChain, lightChain
  const explicitTarget = targetChainSpec ? resolveChain(targetChainSpec) : undefined;
  if (targetChainSpec && !explicitTarget) {
    throw new Error(
      `Target chain "${targetChainSpec}" not found in structure (available: ${availableChains.join(', ')}).`,
    );
  }

  const chainTypes = backbone.molecularContext?.chainTypes ?? {};
  const { heavyChain: detectedHeavy, lightChain: detectedLight } = detectAntibodyHeavyLightChains(
    chainInfos,
    chainTypes,
    lockedChainSet,
    explicitTarget,
  );

  const unlockedHeavy =
    detectedHeavy && !lockedChainSet.has(detectedHeavy) ? detectedHeavy : undefined;
  const unlockedLight =
    detectedLight && !lockedChainSet.has(detectedLight) ? detectedLight : undefined;
  const firstUnlocked = availableChains.find((c) => !lockedChainSet.has(c)) ?? availableChains[0];
  // Only set `targetChain` when the user explicitly specified one: leaving it unset for
  // H+L antibodies lets score() include both H & L CDRs and prevents AR mode from truncating at H.
  const effectiveTargetChain = explicitTarget;
  const resolvedTargetChain =
    explicitTarget ??
    (variant === 'antifold' ? unlockedHeavy ?? unlockedLight ?? firstUnlocked : undefined) ??
    defaultChain;

  // Pass 2: parse chainless fixedPositions against resolvedTargetChain (matching input-preview.ts)
  const { fixedPositions: userRefs, lockedChains: userLockedChains } =
    parseFixedPositionsAndChains(fixedPositionsSpec, resolvedTargetChain, orderedResidues);
  for (const ch of userLockedChains) {
    const resolved = resolveChain(ch);
    if (resolved) lockedChainSet.add(resolved);
  }

  const regionList =
    variant === 'antifold'
      ? regionsSpec
          .split(/[,\s]+/)
          .map((t) => t.trim())
          .filter(Boolean)
      : [];

  const firstNonLight = availableChains.find((c) => c !== detectedLight && !lockedChainSet.has(c))
    ?? availableChains.find((c) => c !== detectedLight);
  const heavyChain =
    variant === 'antifold'
      ? detectedHeavy ?? explicitTarget ?? (!detectedLight ? firstUnlocked : firstNonLight ?? firstUnlocked)
      : undefined;
  const lightChain =
    variant === 'antifold'
      ? detectedLight && detectedLight !== heavyChain
        ? detectedLight
        : explicitTarget && explicitTarget !== heavyChain
          ? explicitTarget
          : undefined
      : undefined;

  const resolvedNumbering =
    variant === 'antifold'
      ? resolveAntibodyNumbering(
          numberingParam,
          chainInfos,
          detectedHeavy ?? (heavyChain !== detectedLight ? heavyChain : undefined),
          detectedLight ?? lightChain,
        )
      : 'imgt';

  const remapped =
    variant === 'antifold' && regionList.length > 0
      ? remapSequentialAntibodyRegions(
          regionList,
          chainInfos,
          detectedHeavy ?? (heavyChain !== detectedLight ? heavyChain : undefined),
          detectedLight ?? lightChain,
          numberingParam,
        )
      : { regions: regionList, remappedChains: [] };

  const combineOpts: Record<string, unknown> = {
    ...(effectiveTargetChain ? { targetChain: effectiveTargetChain } : {}),
    ...(heavyChain ? { heavyChain } : {}),
    ...(lightChain ? { lightChain } : {}),
    ...(variant === 'antifold' ? { numbering: resolvedNumbering } : {}),
    ...(remapped.regions.length > 0 ? { regions: remapped.regions } : {}),
  };

  const prepared = sharedMod.prepareInput(parsed, combineOpts) as PreparedAntiFoldInput;

  // Determine active sampling length
  let sampleLength = prepared.L;
  if (samplingMode === 'autoregressive' && prepared.targetChain) {
    const boundary = prepared.residues.findIndex((r) => !r || r.chain !== prepared.targetChain);
    sampleLength = boundary >= 0 ? boundary : prepared.L;
  }

  // Collect fixed residue keys (user fixedPositions + inherited fixedResidueIndices)
  const fixedKeySet = new Set<string>();
  for (const ref of userRefs) {
    if (typeof ref === 'number') {
      const k = originalOrderKeys[ref];
      if (k) fixedKeySet.add(k);
    } else {
      const ch = resolveChain(ref.chain) ?? ref.chain;
      fixedKeySet.add(residueKey(ch, ref.number, ref.insertionCode ?? ''));
    }
  }
  if (!unlockInherited && Array.isArray(backbone.fixedResidueIndices)) {
    for (const oneBased of backbone.fixedResidueIndices) {
      const idx = Number(oneBased) - 1;
      if (Number.isInteger(idx) && idx >= 0 && idx < originalOrderKeys.length) {
        fixedKeySet.add(originalOrderKeys[idx]);
      }
    }
  }

  // Build index-keyed fixed map against `prepared.residues` (bounded by `sampleLength`).
  // Whenever the user explicitly specifies `targetChain`, ALWAYS fix every residue on
  // non-target chains — even when `regions` (e.g. `CDR3`) is also set, so selecting
  // heavy chain `H` + `CDR3` never mutates `CDRL3` on the light chain.
  const fixed: Record<number, number> = {};
  for (let i = 0; i < sampleLength; i++) {
    const r = prepared.residues[i];
    if (!r) {
      // Preserve null chain-break separator tokens (<cath>/<pad>) during multi-chain AR sampling
      fixed[i] = prepared.tokens[i];
      continue;
    }
    const k = residueKey(r.chain, r.number, r.insertion);
    const fixedByTarget = Boolean(explicitTarget) && r.chain !== explicitTarget;
    if (lockedChainSet.has(r.chain) || fixedKeySet.has(k) || fixedByTarget) {
      fixed[i] = prepared.tokens[i];
    }
  }

  return {
    parsed,
    prepared,
    combineOpts,
    fixed,
    sampleLength,
    propagatedLockedChains: [...lockedChainSet],
    originalChainLengths,
    originalOrderKeys,
    nativeAaByKey,
  };
}

function reconstructOriginalSequence(
  plan: ResolvedAntiFoldPlan,
  sampledSeq: string,
): string {
  const sampledByKey = new Map<string, string>();
  for (let i = 0; i < sampledSeq.length && i < plan.prepared.residues.length; i++) {
    const r = plan.prepared.residues[i];
    if (!r) continue;
    const ch = sampledSeq[i];
    if (ch && ch !== '-') {
      sampledByKey.set(residueKey(r.chain, r.number, r.insertion), ch);
    }
  }
  return plan.originalOrderKeys
    .map((k) => sampledByKey.get(k) ?? plan.nativeAaByKey.get(k) ?? 'X')
    .join('');
}

function buildCombinedSequenceForScore(
  plan: ResolvedAntiFoldPlan,
  sampledSeq: string,
): string {
  if (sampledSeq.length === plan.prepared.L) return sampledSeq;
  // Autoregressive mode sampled only prefix `sampleLength`; pad remainder with native combined sequence
  return sampledSeq + plan.prepared.sequence.slice(sampledSeq.length);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let antiFoldMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let sharedMod: any = null;
let engine: GvpEngine | null = null;
let loadedKey: string | null = null;

async function ensureEngine(
  id: number,
  variant: Variant,
  precision: Precision,
): Promise<GvpEngine> {
  const key = `${variant}:${precision}`;
  if (engine && loadedKey === key) return engine;

  if (engine) {
    try {
      engine.destroy();
    } catch {
      /* ignore */
    }
    engine = null;
    loadedKey = null;
  }

  if (!antiFoldMod || !sharedMod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: `Loading ${variant === 'antifold' ? 'AntiFold' : 'ESM-IF1'} WebGPU modules...`,
    });
    const webgpuUrl = new URL('/vendor/antifold-esmif1/webgpu/index.js?v=2', self.location.origin).href;
    const sharedUrl = new URL('/vendor/antifold-esmif1/shared/index.js', self.location.origin).href;
    [antiFoldMod, sharedMod] = await Promise.all([
      import(/* @vite-ignore */ webgpuUrl),
      import(/* @vite-ignore */ sharedUrl),
    ]);
  }

  const envBaseUrl = (import.meta.env.VITE_ANTIFOLD_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;
  const displayName = variant === 'antifold' ? 'AntiFold' : 'ESM-IF1';

  const loadOnce = async (retried: boolean): Promise<GvpEngine> => {
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
      detail: `Fetching ${displayName} (${precision.toUpperCase()}) weights...`,
    });

    try {
      return (await antiFoldMod.createGVPTransformerEngine({
        baseUrl,
        variant,
        precision,
        cache: 'none',
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { phase?: string; loaded?: number; total?: number }) => {
          if (prog.phase !== 'weights') return;
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac = total > 0 ? loaded / total : 0.5;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.92, 0.06 + frac * 0.86),
            detail: `${displayName} weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
          });
        },
      })) as GvpEngine;
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        /sha-?256|mismatch|manifest|incomplete|invalid safetensors/i.test(errMsg)
      ) {
        await evictCachedUrls(fetchedUrls);
        return await loadOnce(true);
      }
      throw err;
    }
  };

  engine = await loadOnce(false);
  loadedKey = key;
  return engine;
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'dispose') {
      try {
        engine?.destroy();
      } catch {
        /* ignore */
      }
      engine = null;
      loadedKey = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const structure = msg.inputs?.structure as StructureArtifact | undefined;
      if (!structure || (!structure.cifText && !structure.pdbText)) {
        throw new Error('AntiFold / ESM-IF1 requires an upstream StructureArtifact.');
      }

      const p = msg.params ?? {};
      const variant = normalizeVariant(p.variant);
      const precision = normalizePrecision(p.precision);
      const rawNum = Number(p.numSequences ?? 2);
      const numSequences = Number.isFinite(rawNum)
        ? Math.max(1, Math.min(8, Math.round(rawNum)))
        : 2;
      const rawTemp = Number(p.temperature ?? 0.2);
      const temperature = Number.isFinite(rawTemp)
        ? Math.max(0.01, Math.min(1.5, rawTemp))
        : 0.2;
      const samplingMode = normalizeSamplingMode(p.samplingMode);
      const targetChainSpec = String(p.targetChain ?? '').trim();
      const numberingParam = normalizeAntiFoldNumberingParam(p.numbering);
      const regionsSpec = String(p.regions ?? 'CDR1 CDR2 CDR3').trim();
      const fixedPositionsSpec = String(p.fixedPositions ?? '').trim();
      const unlockInherited = Boolean(p.unlockInherited);
      const rawSeed = Number(p.seed ?? 42);
      const seed = Number.isFinite(rawSeed) ? Math.round(rawSeed) : 42;

      const activeEngine = await ensureEngine(msg.id, variant, precision);
      const modelLabel = variant === 'antifold' ? 'antifold' : 'esmif1';
      const displayName = variant === 'antifold' ? 'AntiFold' : 'ESM-IF1';

      const sampleOne = async (
        plan: ResolvedAntiFoldPlan,
        itemSeed: number,
      ): Promise<{ sequence: string; score: number | null; globalScore: number | null }> => {
        const uniforms = generateUniforms(plan.sampleLength, itemSeed);
        const sampleRes =
          samplingMode === 'autoregressive'
            ? await activeEngine.sample(plan.parsed, {
                ...plan.combineOpts,
                temperature,
                uniforms,
                fixed: plan.fixed,
                length: plan.sampleLength,
              })
            : await activeEngine.sampleIndependent(plan.parsed, {
                ...plan.combineOpts,
                temperature,
                uniforms,
                fixed: plan.fixed,
              });

        const fullSequence = reconstructOriginalSequence(plan, sampleRes.sequence);
        const combinedForScore = buildCombinedSequenceForScore(plan, sampleRes.sequence);
        const scored = await activeEngine.score(plan.parsed, {
          ...plan.combineOpts,
          sequence: combinedForScore,
        });
        // Negative log-likelihood (lower is better, matching MPNN / Top-K conventions)
        const nll = Number.isFinite(scored.logLikelihood) ? -scored.logLikelihood : null;
        const coordNll = Number.isFinite(scored.coordinateLogLikelihood)
          ? -scored.coordinateLogLikelihood
          : nll;
        return { sequence: fullSequence, score: nll, globalScore: coordNll };
      };

      const collectionItems =
        structure.collection?.items && structure.collection.items.length > 1
          ? structure.collection.items
          : null;

      if (!collectionItems) {
        const structureText = structure.cifText ?? structure.pdbText;
        const plan = resolveAntiFoldPlan(
          sharedMod,
          structure,
          structureText,
          variant,
          samplingMode,
          targetChainSpec,
          numberingParam,
          regionsSpec,
          fixedPositionsSpec,
          unlockInherited,
        );

        const candidates: SequenceCandidate[] = [];
        for (let i = 0; i < numSequences; i++) {
          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'sample',
            progress: i / numSequences,
            detail: `${displayName} (${samplingMode}) designing seq ${i + 1}/${numSequences}...`,
          });

          const res = await sampleOne(plan, seed + i);
          const candChainLengths =
            plan.originalChainLengths.length > 0 &&
            plan.originalChainLengths.reduce((a, b) => a + b, 0) === res.sequence.length
              ? plan.originalChainLengths
              : undefined;

          candidates.push({
            id: `${modelLabel}-${i + 1}`,
            sourceStructureId: structure.id,
            sampleIndex: i,
            sequence: res.sequence,
            score: res.score,
            globalScore: res.globalScore,
            sourceModelId: modelLabel,
            temperature,
            seed: seed + i,
            ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
          });
        }

        candidates.sort((a, b) => (a.score ?? 99) - (b.score ?? 99));

        const runHash = fnv1a(
          JSON.stringify({
            structureId: structure.id,
            variant,
            precision,
            numSequences,
            temperature,
            samplingMode,
            targetChainSpec,
            numberingParam,
            regionsSpec,
            fixedPositionsSpec,
            unlockInherited,
            seed,
          }),
        );

        const seqSet: SequenceSet = {
          kind: 'SequenceSet',
          id: `${modelLabel}-seq-${structure.id}-${seed}-${runHash}`,
          parentStructureId: structure.id,
          sourceStructureId: structure.id,
          sourceModelId: modelLabel,
          ...(plan.propagatedLockedChains.length > 0
            ? { lockedChains: plan.propagatedLockedChains }
            : {}),
          ...(structure.chainRoles ? { chainRoles: structure.chainRoles } : {}),
          candidates,
        };

        self.postMessage({
          id: msg.id,
          type: 'result',
          outputs: { sequences: seqSet },
          elapsedMs: performance.now() - t0,
        });
        return;
      }

      // Multi-backbone collection fanout
      const structureMembers: CollectionMember<StructureArtifact>[] = collectionItems.map(
        (item, idx) => {
          const designKey = item.designKey ?? structure.collection?.keys?.[idx] ?? item.id;
          const designIndex = typeof item.designIndex === 'number' ? item.designIndex : idx;
          return {
            key: designKey,
            ordinal: designIndex,
            memberKind: 'StructureArtifact',
            lineage: {
              designKey,
              designIndex,
              sourceStructure: {
                artifactId: item.id,
                payloadDigest: computeStructurePayloadDigest(item),
                designKey,
              },
              parentKeys: [],
            },
            payload: item,
            summary: {
              label: item.label || item.id,
              residueCount: item.length,
            },
          };
        },
      );

      const fanoutPlan = planMpnnFanout(structureMembers, {
        modelId: modelLabel,
        sequencesPerBackbone: numSequences,
        baseSeed: seed,
        temperature,
        scientificParams: {
          variant,
          precision,
          samplingMode,
          targetChain: targetChainSpec,
          numbering: numberingParam,
          regions: regionsSpec,
          fixedPositions: fixedPositionsSpec,
          unlockInherited,
        },
      });

      const backboneById = new Map(collectionItems.map((b) => [b.id, b]));
      const allCandidates: SequenceCandidate[] = [];
      const allPropagatedLocked = new Set<string>();
      const totalPlanned = Math.max(1, fanoutPlan.totalPlannedSequences);
      let completedSeqs = 0;

      for (let bIdx = 0; bIdx < fanoutPlan.perBackbonePlans.length; bIdx++) {
        const subplan = fanoutPlan.perBackbonePlans[bIdx];
        const backbone = backboneById.get(subplan.sourceStructure.artifactId) ?? collectionItems[bIdx];
        const text = backbone?.cifText ?? backbone?.pdbText;
        if (!backbone || !text) {
          throw new Error(`Backbone "${subplan.sourceStructure.artifactId}" is missing structure text for ${displayName}.`);
        }

        const plan = resolveAntiFoldPlan(
          sharedMod,
          backbone,
          text,
          variant,
          samplingMode,
          targetChainSpec,
          numberingParam,
          regionsSpec,
          fixedPositionsSpec,
          unlockInherited,
        );
        for (const ch of plan.propagatedLockedChains) allPropagatedLocked.add(ch);
        const backboneCandidates: SequenceCandidate[] = [];

        for (let q = 0; q < subplan.items.length; q++) {
          const item = subplan.items[q];
          const itemSeed = deriveStableItemSeed(seed, 'mpnn', item.key);
          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'sample',
            progress: completedSeqs / totalPlanned,
            detail: `Backbone ${bIdx + 1}/${fanoutPlan.backboneCount} · ${displayName} seq ${q + 1}/${subplan.items.length}`,
          });

          const res = await sampleOne(plan, itemSeed);
          completedSeqs++;

          const candChainLengths =
            plan.originalChainLengths.length > 0 &&
            plan.originalChainLengths.reduce((a, b) => a + b, 0) === res.sequence.length
              ? plan.originalChainLengths
              : undefined;

          backboneCandidates.push({
            id: `d${subplan.designIndex + 1}-${modelLabel}-${item.sequenceIndex + 1}`,
            key: item.key,
            sequenceKey: item.key,
            designKey: subplan.designKey,
            designIndex: subplan.designIndex,
            sourceStructureId: backbone.id,
            sampleIndex: item.sequenceIndex,
            sequence: res.sequence,
            score: res.score,
            globalScore: res.globalScore,
            sourceModelId: modelLabel,
            temperature,
            seed: itemSeed,
            ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
          });
        }

        backboneCandidates.sort((a, b) => {
          const diff = (a.score ?? 99) - (b.score ?? 99);
          return diff !== 0 ? diff : (a.sampleIndex ?? 0) - (b.sampleIndex ?? 0);
        });
        allCandidates.push(...backboneCandidates);
      }

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `${modelLabel}-seq-col-${fanoutPlan.planDigest.slice(0, 16)}`,
        parentStructureId: structure.id,
        sourceStructureId: structure.id,
        sourceStructureIds: collectionItems.map((b) => b.id),
        sourceModelId: modelLabel,
        collectionKind: 'independent-candidates',
        ...(allPropagatedLocked.size > 0 ? { lockedChains: [...allPropagatedLocked] } : {}),
        ...(structure.chainRoles ? { chainRoles: { ...structure.chainRoles } } : {}),
        candidates: allCandidates,
      };

      self.postMessage({
        id: msg.id,
        type: 'result',
        outputs: { sequences: seqSet },
        elapsedMs: performance.now() - t0,
      });
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
