/// <reference lib="webworker" />
// Layer 3 Plugin Worker — lasermpnn (ramithuh/lasermpnn-webgpu, MIT)
// Joint sequence design + all-atom sidechain packing (with ligand context & explicit hydrogens).
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress, and disposal.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  computeChainLengths,
  parseFixedPositionsAndChains,
  type OrderedStructureResidue,
} from '$lib/engine/mpnn-spec';
import {
  computeStructurePayloadDigest,
  type CollectionMember,
} from '$lib/engine/collections';
import { deriveStableItemSeed, planMpnnFanout } from '$lib/engine/work-plan';
import {
  collectOwnedTransferables,
  createWorkerHarness,
  fnv1a,
  type WorkerRunContext,
} from '$lib/kernel';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/lasermpnn-webgpu/resolve/main';

type Precision = 'fp16' | 'int8' | 'fp32';

interface LaserScore {
  sequenceLogLikelihood: number;
  chiLogLikelihood: number;
  totalLogLikelihood: number;
}

interface LaserDesignResult {
  sequences: string[];
  scores: LaserScore[];
  chiAngles: Float32Array[];
  pdbTexts: string[];
  timingMs: number;
}

interface LaserEngine {
  design(
    structureText: string,
    opts?: {
      numSequences?: number;
      temperature?: number;
      chiTemperature?: number;
      seed?: number;
      fixedPositions?: string[];
      designChains?: string[];
      omitAAs?: string;
    },
  ): Promise<LaserDesignResult>;
  destroy(): void;
}

function normalizePrecision(raw: unknown): Precision {
  const val = String(raw ?? 'fp16').trim().toLowerCase();
  if (val === 'fp32' || val === 'int8' || val === 'fp16') return val;
  return 'fp16';
}

function sanitizeOmitAAs(raw: unknown): string | undefined {
  const clean = String(raw ?? '')
    .trim()
    .toUpperCase()
    .replace(/[^ACDEFGHIKLMNPQRSTVWYX]/g, '');
  if (!clean) return undefined;
  return Array.from(new Set(clean.split(''))).join('');
}

function buildLaserConstraints(
  structure: StructureArtifact,
  structureText: string,
  fixedPositionsSpec: string,
  unlockInherited: boolean,
): {
  fixedIds: string[];
  designChains: string[] | undefined;
  propagatedLockedChains: string[];
  chainLengths: number[];
  matchedUserFixedKeys: Set<string>;
} {
  const useCif = Boolean(structure.cifText);
  const parsedIn =
    structure.sourceResidues && structure.sourceResidues.length > 0
      ? null
      : parseStructureText(structureText);
  const sourceResidues = parsedIn ? parsedIn.sourceResidues : structure.sourceResidues!;
  const atomPresence = structure.atomPresence ?? parsedIn?.atomPresence;

  const chainList: string[] = [];
  const completeChainList: string[] = [];
  const presentChains = new Set<string>();
  const upperToChain = new Map<string, string>();
  const orderedResidues: OrderedStructureResidue[] = [];
  const completeBackboneKeys = new Set<string>();
  const hasAnyAtomPresence = Boolean(atomPresence && atomPresence.length === sourceResidues.length);

  for (let i = 0; i < sourceResidues.length; i++) {
    const sr = sourceResidues[i];
    const ch = useCif ? sr.authChainId : sr.projectedChainId;
    chainList.push(ch);
    presentChains.add(ch);
    if (!upperToChain.has(ch.toUpperCase())) upperToChain.set(ch.toUpperCase(), ch);
    orderedResidues.push({
      chain: ch,
      number: sr.resNum,
      insertionCode: sr.insCode ?? '',
    });
    const ap = atomPresence?.[i];
    const complete =
      !hasAnyAtomPresence ||
      !ap ||
      (ap.hasN && ap.hasCA && ap.hasC && (ap.hasO || ap.oReconstructed));
    if (complete) {
      completeBackboneKeys.add(`${ch}:${sr.resNum}${sr.insCode ?? ''}`);
      completeChainList.push(ch);
    }
  }
  const allChains = [...presentChains];
  const defaultChain = allChains[0] ?? 'A';
  const chainLengths = computeChainLengths(
    completeChainList.length > 0 ? completeChainList : chainList,
  );

  const { fixedPositions: userRefs, lockedChains: userLockedChains } =
    parseFixedPositionsAndChains(fixedPositionsSpec, defaultChain, orderedResidues);

  const fixedSet = new Set<string>();
  const matchedUserFixedKeys = new Set<string>();
  const lockedChainSet = new Set<string>();

  for (const ch of userLockedChains) {
    const resolved = presentChains.has(ch) ? ch : upperToChain.get(ch.toUpperCase());
    if (resolved) lockedChainSet.add(resolved);
  }

  // LASErMPNN residue IDs use `${chain}:${resnum}${icode}` format; guard incomplete backbones
  for (const ref of userRefs) {
    if (typeof ref === 'number') {
      const sr = sourceResidues[ref];
      if (sr) {
        const ch = useCif ? sr.authChainId : sr.projectedChainId;
        const laserId = `${ch}:${sr.resNum}${sr.insCode ?? ''}`;
        if (completeBackboneKeys.has(laserId)) fixedSet.add(laserId);
        matchedUserFixedKeys.add(`${sr.authChainId}:${sr.resNum}:${sr.insCode ?? ''}`);
        matchedUserFixedKeys.add(`${sr.projectedChainId}:${sr.resNum}:${sr.insCode ?? ''}`);
      }
    } else {
      const ch = presentChains.has(ref.chain)
        ? ref.chain
        : upperToChain.get(ref.chain.toUpperCase()) ?? ref.chain;
      const ins = ref.insertionCode ?? '';
      const laserId = `${ch}:${ref.number}${ins}`;
      if (completeBackboneKeys.has(laserId)) fixedSet.add(laserId);
      matchedUserFixedKeys.add(`${ch}:${ref.number}:${ins}`);
    }
  }

  if (!unlockInherited) {
    if (Array.isArray(structure.lockedChains)) {
      for (const rawCh of structure.lockedChains) {
        const ch = String(rawCh).trim();
        const resolved = presentChains.has(ch) ? ch : upperToChain.get(ch.toUpperCase());
        if (resolved) lockedChainSet.add(resolved);
      }
    }
    if (Array.isArray(structure.fixedResidueIndices)) {
      for (const oneBased of structure.fixedResidueIndices) {
        const idx = Number(oneBased) - 1;
        if (Number.isInteger(idx) && idx >= 0 && idx < sourceResidues.length) {
          const sr = sourceResidues[idx];
          const ch = useCif ? sr.authChainId : sr.projectedChainId;
          const laserId = `${ch}:${sr.resNum}${sr.insCode ?? ''}`;
          if (completeBackboneKeys.has(laserId)) fixedSet.add(laserId);
        }
      }
    }
  }

  const designChains =
    lockedChainSet.size > 0
      ? allChains.filter((ch) => !lockedChainSet.has(ch))
      : undefined;

  return {
    fixedIds: [...fixedSet],
    designChains,
    propagatedLockedChains: [...lockedChainSet],
    chainLengths,
    matchedUserFixedKeys,
  };
}

function buildPropagatedDesignMetadata(
  upstream: StructureArtifact,
  parsedOut: ReturnType<typeof parseStructureText>,
  unlockInherited: boolean,
  matchedUserFixedKeys: ReadonlySet<string>,
): Pick<StructureArtifact, 'fixedResidueIndices' | 'motifIndices' | 'nativeMotifCoords'> {
  const upRes = upstream.sourceResidues;
  const outRes = parsedOut.sourceResidues;
  const sameLen = parsedOut.nResidues === upstream.length;

  const outIdxByKey = new Map<string, number>();
  if (!sameLen) {
    for (let i = 0; i < outRes.length; i++) {
      const sr = outRes[i];
      outIdxByKey.set(`${sr.authChainId}:${sr.resNum}:${sr.insCode ?? ''}`, i);
      outIdxByKey.set(`${sr.projectedChainId}:${sr.resNum}:${sr.insCode ?? ''}`, i);
    }
  }
  const mapOldIdx = (oldZeroIdx: number): number => {
    if (sameLen) return oldZeroIdx >= 0 && oldZeroIdx < outRes.length ? oldZeroIdx : -1;
    const sr = upRes?.[oldZeroIdx];
    if (!sr) return -1;
    return (
      outIdxByKey.get(`${sr.authChainId}:${sr.resNum}:${sr.insCode ?? ''}`) ??
      outIdxByKey.get(`${sr.projectedChainId}:${sr.resNum}:${sr.insCode ?? ''}`) ??
      -1
    );
  };

  const fixedSet = new Set<number>();
  if (!unlockInherited && Array.isArray(upstream.fixedResidueIndices)) {
    for (const oneBased of upstream.fixedResidueIndices) {
      const mapped = mapOldIdx(Number(oneBased) - 1);
      if (mapped >= 0) fixedSet.add(mapped + 1);
    }
  }
  if (matchedUserFixedKeys.size > 0) {
    for (let i = 0; i < outRes.length; i++) {
      const sr = outRes[i];
      if (
        matchedUserFixedKeys.has(`${sr.authChainId}:${sr.resNum}:${sr.insCode ?? ''}`) ||
        matchedUserFixedKeys.has(`${sr.projectedChainId}:${sr.resNum}:${sr.insCode ?? ''}`)
      ) {
        fixedSet.add(i + 1);
      }
    }
  }

  let motifIndices: number[] | undefined;
  let nativeMotifCoords: number[][] | undefined;
  if (
    Array.isArray(upstream.motifIndices) &&
    Array.isArray(upstream.nativeMotifCoords) &&
    upstream.motifIndices.length > 0 &&
    upstream.nativeMotifCoords.length > 0
  ) {
    const limit = Math.min(upstream.motifIndices.length, upstream.nativeMotifCoords.length);
    const mIdx: number[] = [];
    const mCoords: number[][] = [];
    for (let k = 0; k < limit; k++) {
      const mapped = mapOldIdx(upstream.motifIndices[k]);
      const coord = upstream.nativeMotifCoords[k];
      if (mapped >= 0 && Array.isArray(coord) && coord.length >= 3) {
        mIdx.push(mapped);
        mCoords.push([coord[0], coord[1], coord[2]]);
      }
    }
    if (mIdx.length > 0) {
      motifIndices = mIdx;
      nativeMotifCoords = mCoords;
    }
  }

  const fixedResidueIndices = fixedSet.size > 0 ? [...fixedSet].sort((a, b) => a - b) : undefined;
  return {
    ...(fixedResidueIndices ? { fixedResidueIndices } : {}),
    ...(motifIndices ? { motifIndices, nativeMotifCoords } : {}),
  };
}

function buildPackedStructureArtifact(
  upstream: StructureArtifact,
  pdbText: string,
  precision: Precision,
  id: string,
  label: string,
  propagatedLockedChains: string[],
  unlockInherited: boolean,
  matchedUserFixedKeys: ReadonlySet<string>,
  extra: Partial<StructureArtifact> = {},
): StructureArtifact {
  const parsedOut = parseStructureText(pdbText);
  const geometry = computeGeometryStats(
    parsedOut.caCoords,
    parsedOut.chainIds,
    parsedOut.resNums,
    parsedOut.insCodes,
  );
  const designMeta = buildPropagatedDesignMetadata(
    upstream,
    parsedOut,
    unlockInherited,
    matchedUserFixedKeys,
  );
  return {
    kind: 'StructureArtifact',
    id,
    label,
    sourceModelId: 'lasermpnn',
    parentStructureId: upstream.id,
    length: parsedOut.nResidues,
    sequence: parsedOut.sequence,
    caCoords: parsedOut.caCoords,
    backbone4Coords: parsedOut.backbone4Coords,
    pdbText,
    chainIds: parsedOut.chainIds,
    chainLengths: parsedOut.diagnostics.chainLengths,
    insCodes: parsedOut.insCodes,
    sourceResidues: parsedOut.sourceResidues,
    chainIdMap: parsedOut.chainIdMap,
    atomPresence: parsedOut.atomPresence,
    observedBackboneMask: parsedOut.observedBackboneMask,
    diagnostics: parsedOut.diagnostics,
    bfactorSemantics: 'none',
    atomGranularity: 'all-atom',
    molecularContext: {
      ...upstream.molecularContext,
      ...(parsedOut.diagnostics.hasLigands
        ? { ligands: parsedOut.diagnostics.ligandNames }
        : {}),
      ...(parsedOut.diagnostics.hasNucleicAcids
        ? { nucleicAcidChains: parsedOut.diagnostics.nucleicAcidChains }
        : {}),
      hasExplicitHydrogens: true,
    },
    provenance: {
      quantization: precision,
    },
    ...(propagatedLockedChains.length > 0 ? { lockedChains: propagatedLockedChains } : {}),
    ...(upstream.chainRoles ? { chainRoles: { ...upstream.chainRoles } } : {}),
    ...designMeta,
    geometry,
    ...extra,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let laserMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let weightsMod: any = null;
let engine: LaserEngine | null = null;
let loadedPrecision: Precision | null = null;

async function ensureEngine(ctx: WorkerRunContext, precision: Precision): Promise<LaserEngine> {
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

  if (!laserMod || !weightsMod) {
    ctx.progress('init', 0.02, 'Loading LASErMPNN WebGPU modules...');
    const entryUrl = new URL('/vendor/lasermpnn/webgpu/index.js', self.location.origin).href;
    const weightsUrl = new URL(
      '/vendor/lasermpnn/shared/lasermpnn/weights.js',
      self.location.origin,
    ).href;
    [laserMod, weightsMod] = await Promise.all([
      import(/* @vite-ignore */ entryUrl),
      import(/* @vite-ignore */ weightsUrl),
    ]);
  }

  const envBaseUrl = (import.meta.env.VITE_LASERMPNN_WEIGHTS_URL as string | undefined)?.trim();
  const rootUrl = (envBaseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
  const baseUrl = /(fp32|fp16|int8)$/.test(rootUrl) ? rootUrl : `${rootUrl}/${precision}`;

  const loadOnce = async (retried: boolean): Promise<LaserEngine> => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.06, `Fetching LASErMPNN (${precision.toUpperCase()}) weights...`);

    try {
      const bundle = await weightsMod.loadWeights(baseUrl, {
        cache: false,
        read: async (name: string): Promise<Uint8Array> => {
          const fileUrl = `${baseUrl}/${name}`;
          const resp = await cachedFetch(fileUrl);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${fileUrl}`);
          return new Uint8Array(await resp.arrayBuffer());
        },
        onProgress: (prog: { loaded?: number; total?: number }) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac = total > 0 ? loaded / total : 0.5;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          ctx.progress(
            'weights',
            Math.min(0.92, 0.08 + frac * 0.84),
            `LASErMPNN weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
          );
        },
      });

      ctx.progress('init', 0.94, 'Uploading LASErMPNN weights to GPU...');

      return (await laserMod.createLaserMpnnEngine({
        baseUrl,
        precision,
        weights: bundle,
        cache: false,
      })) as LaserEngine;
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        /checksum|truncated|manifest|mismatch/i.test(errMsg)
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
    const structure = ctx.inputs?.structure as StructureArtifact | undefined;
    const structureText = structure?.cifText ?? structure?.pdbText;
    if (!structure || !structureText) {
      throw new Error('LASErMPNN requires an upstream StructureArtifact.');
    }

    const p = ctx.params ?? {};
    const precision = normalizePrecision(p.precision);
    const rawNum = Number(p.numSequences ?? 2);
    const numSequences = Number.isFinite(rawNum)
      ? Math.max(1, Math.min(8, Math.round(rawNum)))
      : 2;
    const rawTemp = Number(p.temperature ?? 0.1);
    const temperature = Number.isFinite(rawTemp)
      ? Math.max(0.01, Math.min(1.5, rawTemp))
      : 0.1;
    const fixedPositionsSpec = String(p.fixedPositions ?? '').trim();
    const omitAAs = sanitizeOmitAAs(p.omitAminoAcids);
    const unlockInherited = Boolean(p.unlockInherited);
    const rawSeed = Number(p.seed ?? 42);
    const seed = (Number.isFinite(rawSeed) ? Math.round(rawSeed) : 42) >>> 0;

    const activeEngine = await ensureEngine(ctx, precision);

    const collectionItems =
      structure.collection?.items && structure.collection.items.length > 1
        ? structure.collection.items
        : null;

    if (!collectionItems) {
      const { fixedIds, designChains, propagatedLockedChains, chainLengths, matchedUserFixedKeys } =
        buildLaserConstraints(structure, structureText, fixedPositionsSpec, unlockInherited);

      const entries: Array<{ candidate: SequenceCandidate; pdbText: string }> = [];
      for (let i = 0; i < numSequences; i++) {
        if (ctx.signal.aborted) throw new Error('Workflow run cancelled');
        const itemSeed = (seed + i) >>> 0;
        ctx.progress(
          'sample',
          i / numSequences,
          `LASErMPNN designing & packing sidechains ${i + 1}/${numSequences}...`,
        );

        const res = await activeEngine.design(structureText, {
          numSequences: 1,
          temperature,
          chiTemperature: temperature,
          seed: itemSeed,
          ...(fixedIds.length > 0 ? { fixedPositions: fixedIds } : {}),
          ...(designChains ? { designChains } : {}),
          ...(omitAAs ? { omitAAs } : {}),
        });

        const seq = res.sequences[0];
        const sc = res.scores[0];
        const pdbText = res.pdbTexts[0];
        const len = Math.max(1, seq.length);
        // Convert sum log-likelihood into mean NLL (lower is better, matching ProteinMPNN/Top-K conventions)
        const seqNll = Number.isFinite(sc?.sequenceLogLikelihood)
          ? -sc.sequenceLogLikelihood / len
          : null;
        const totalNll = Number.isFinite(sc?.totalLogLikelihood)
          ? -sc.totalLogLikelihood / len
          : seqNll;

        const candChainLengths =
          chainLengths.length > 0 &&
          chainLengths.reduce((a, b) => a + b, 0) === seq.length
            ? chainLengths
            : undefined;

        entries.push({
          pdbText,
          candidate: {
            id: `lasermpnn-${i + 1}`,
            sourceStructureId: structure.id,
            sampleIndex: i,
            sequence: seq,
            score: seqNll,
            globalScore: totalNll,
            sourceModelId: 'lasermpnn',
            temperature,
            seed: itemSeed,
            ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
          },
        });
      }

      entries.sort((a, b) => (a.candidate.score ?? 99) - (b.candidate.score ?? 99));

      const runHash = fnv1a(
        JSON.stringify({
          structureId: structure.id,
          precision,
          numSequences,
          temperature,
          fixedIds,
          designChains: designChains ?? null,
          omitAAs: omitAAs ?? null,
          unlockInherited,
          seed,
        }),
      );

      const packedItems: StructureArtifact[] = entries.map((entry, idx) => {
        const packed = buildPackedStructureArtifact(
          structure,
          entry.pdbText,
          precision,
          idx === 0
            ? `lasermpnn-struct-${structure.id}-${seed}-${runHash}`
            : `lasermpnn-struct-${structure.id}-${seed}-${runHash}-s${idx + 1}`,
          entries.length > 1
            ? `LASErMPNN #${idx + 1} (${precision.toUpperCase()} · L=${entry.candidate.sequence.length})`
            : `LASErMPNN packed (${precision.toUpperCase()} · L=${entry.candidate.sequence.length})`,
          propagatedLockedChains,
          unlockInherited,
          matchedUserFixedKeys,
          {
            designKey: `lasermpnn-seq-${idx + 1}`,
            designIndex: entry.candidate.sampleIndex ?? idx,
            seed: entry.candidate.seed ?? seed,
          },
        );
        if (
          !entry.candidate.chainLengths &&
          packed.chainLengths &&
          packed.chainLengths.reduce((a, b) => a + b, 0) === entry.candidate.sequence.length
        ) {
          entry.candidate.chainLengths = packed.chainLengths;
        }
        return packed;
      });

      const bestStruct = packedItems[0];
      const packedStructure: StructureArtifact =
        packedItems.length > 1
          ? {
              ...bestStruct,
              collection: {
                kind: 'Collection',
                schema: 'denovo.collection/v1',
                id: `lasermpnn-col-${structure.id}-${seed}-${runHash}`,
                familyKey: `lasermpnn-${structure.id}`,
                collectionKind: 'independent-candidates',
                requestedCount: numSequences,
                complete: true,
                items: packedItems,
                keys: packedItems.map((it) => it.designKey ?? it.id),
                representativeSelection: {
                  selectedKey: bestStruct.designKey ?? bestStruct.id,
                  criterion: 'lowest-seq-nll',
                  totalMembers: packedItems.length,
                  selectedIndex: 0,
                  scoreValue: entries[0].candidate.score ?? null,
                },
              },
              representativeSelection: {
                selectedKey: bestStruct.designKey ?? bestStruct.id,
                criterion: 'lowest-seq-nll',
                totalMembers: packedItems.length,
                selectedIndex: 0,
                scoreValue: entries[0].candidate.score ?? null,
              },
            }
          : bestStruct;

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `lasermpnn-seq-${structure.id}-${seed}-${runHash}`,
        parentStructureId: structure.id,
        sourceStructureId: structure.id,
        sourceModelId: 'lasermpnn',
        ...(propagatedLockedChains.length > 0 ? { lockedChains: propagatedLockedChains } : {}),
        ...(structure.chainRoles ? { chainRoles: structure.chainRoles } : {}),
        candidates: entries.map((e) => e.candidate),
      };

      const transferableViews: Array<ArrayBufferView | undefined> = [];
      for (const item of packedItems) {
        transferableViews.push(item.caCoords, item.backbone4Coords, item.observedBackboneMask);
      }
      const transferables = collectOwnedTransferables(transferableViews);

      return {
        outputs: { sequences: seqSet, structure: packedStructure },
        transferables,
      };
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
      modelId: 'lasermpnn',
      sequencesPerBackbone: numSequences,
      baseSeed: seed,
      temperature,
      scientificParams: {
        precision,
        fixedPositions: fixedPositionsSpec,
        omitAminoAcids: omitAAs ?? '',
        unlockInherited,
      },
    });

    const backboneById = new Map(collectionItems.map((b) => [b.id, b]));
    const allCandidates: SequenceCandidate[] = [];
    const allPropagatedLocked = new Set<string>();
    const allPackedEntries: Array<{ score: number | null; structure: StructureArtifact }> = [];
    const totalPlanned = Math.max(1, fanoutPlan.totalPlannedSequences);
    let completedSeqs = 0;

    for (let bIdx = 0; bIdx < fanoutPlan.perBackbonePlans.length; bIdx++) {
      const subplan = fanoutPlan.perBackbonePlans[bIdx];
      const backbone = backboneById.get(subplan.sourceStructure.artifactId) ?? collectionItems[bIdx];
      const text = backbone?.cifText ?? backbone?.pdbText;
      if (!backbone || !text) {
        throw new Error(
          `Backbone "${subplan.sourceStructure.artifactId}" is missing structure text for LASErMPNN.`,
        );
      }

      const {
        fixedIds,
        designChains,
        propagatedLockedChains,
        chainLengths,
        matchedUserFixedKeys,
      } = buildLaserConstraints(backbone, text, fixedPositionsSpec, unlockInherited);
      for (const ch of propagatedLockedChains) allPropagatedLocked.add(ch);
      const backboneCandidates: SequenceCandidate[] = [];
      const backbonePacked: Array<{ score: number | null; structure: StructureArtifact }> = [];

      for (let q = 0; q < subplan.items.length; q++) {
        if (ctx.signal.aborted) throw new Error('Workflow run cancelled');
        const item = subplan.items[q];
        const itemSeed = deriveStableItemSeed(seed, 'mpnn', item.key);
        ctx.progress(
          'sample',
          completedSeqs / totalPlanned,
          `Backbone ${bIdx + 1}/${fanoutPlan.backboneCount} · LASErMPNN seq ${q + 1}/${subplan.items.length}`,
        );

        const res = await activeEngine.design(text, {
          numSequences: 1,
          temperature,
          chiTemperature: temperature,
          seed: itemSeed,
          ...(fixedIds.length > 0 ? { fixedPositions: fixedIds } : {}),
          ...(designChains ? { designChains } : {}),
          ...(omitAAs ? { omitAAs } : {}),
        });
        completedSeqs++;

        const seq = res.sequences[0];
        const sc = res.scores[0];
        const pdbText = res.pdbTexts[0];
        const len = Math.max(1, seq.length);
        const seqNll = Number.isFinite(sc?.sequenceLogLikelihood)
          ? -sc.sequenceLogLikelihood / len
          : null;
        const totalNll = Number.isFinite(sc?.totalLogLikelihood)
          ? -sc.totalLogLikelihood / len
          : seqNll;

        const candChainLengths =
          chainLengths.length > 0 &&
          chainLengths.reduce((a, b) => a + b, 0) === seq.length
            ? chainLengths
            : undefined;

        backboneCandidates.push({
          id: `d${subplan.designIndex + 1}-lasermpnn-${item.sequenceIndex + 1}`,
          key: item.key,
          sequenceKey: item.key,
          designKey: subplan.designKey,
          designIndex: subplan.designIndex,
          sourceStructureId: backbone.id,
          sampleIndex: item.sequenceIndex,
          sequence: seq,
          score: seqNll,
          globalScore: totalNll,
          sourceModelId: 'lasermpnn',
          temperature,
          seed: itemSeed,
          ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
        });

        const packedStruct = buildPackedStructureArtifact(
          backbone,
          pdbText,
          precision,
          `lasermpnn-struct-${backbone.id}-${item.key}`,
          `LASErMPNN d${subplan.designIndex + 1}#${item.sequenceIndex + 1} (${precision.toUpperCase()} · L=${len})`,
          propagatedLockedChains,
          unlockInherited,
          matchedUserFixedKeys,
          {
            designKey: item.key,
            designIndex: subplan.designIndex,
            seed: itemSeed,
          },
        );
        const curCand = backboneCandidates[backboneCandidates.length - 1];
        if (
          !curCand.chainLengths &&
          packedStruct.chainLengths &&
          packedStruct.chainLengths.reduce((a, b) => a + b, 0) === curCand.sequence.length
        ) {
          curCand.chainLengths = packedStruct.chainLengths;
        }
        backbonePacked.push({
          score: seqNll,
          structure: packedStruct,
        });
      }

      backboneCandidates.sort((a, b) => {
        const diff = (a.score ?? 99) - (b.score ?? 99);
        return diff !== 0 ? diff : (a.sampleIndex ?? 0) - (b.sampleIndex ?? 0);
      });
      backbonePacked.sort((a, b) => (a.score ?? 99) - (b.score ?? 99));
      allCandidates.push(...backboneCandidates);
      allPackedEntries.push(...backbonePacked);
    }

    const sortedPacked = [...allPackedEntries].sort((a, b) => (a.score ?? 99) - (b.score ?? 99));
    const allPackedStructures = sortedPacked.map((e) => e.structure);
    const bestOverall = allPackedStructures[0];
    const packedStructure: StructureArtifact =
      allPackedStructures.length > 1
        ? {
            ...bestOverall,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `lasermpnn-struct-col-${fanoutPlan.planDigest.slice(0, 16)}`,
              familyKey: fanoutPlan.familyKey,
              collectionKind: 'independent-candidates',
              requestedCount: allPackedStructures.length,
              complete: true,
              items: allPackedStructures,
              keys: allPackedStructures.map((it) => it.designKey ?? it.id),
              representativeSelection: {
                selectedKey: bestOverall.designKey ?? bestOverall.id,
                criterion: 'lowest-seq-nll',
                totalMembers: allPackedStructures.length,
                selectedIndex: 0,
                scoreValue: sortedPacked[0].score ?? null,
              },
            },
            representativeSelection: {
              selectedKey: bestOverall.designKey ?? bestOverall.id,
              criterion: 'lowest-seq-nll',
              totalMembers: allPackedStructures.length,
              selectedIndex: 0,
              scoreValue: sortedPacked[0].score ?? null,
            },
          }
        : bestOverall;

    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: `lasermpnn-seq-col-${fanoutPlan.planDigest.slice(0, 16)}`,
      parentStructureId: structure.id,
      sourceStructureId: structure.id,
      sourceStructureIds: collectionItems.map((b) => b.id),
      sourceModelId: 'lasermpnn',
      collectionKind: 'independent-candidates',
      ...(allPropagatedLocked.size > 0 ? { lockedChains: [...allPropagatedLocked] } : {}),
      ...(structure.chainRoles ? { chainRoles: { ...structure.chainRoles } } : {}),
      candidates: allCandidates,
    };

    const transferableViews: Array<ArrayBufferView | undefined> = [];
    for (const item of allPackedStructures) {
      transferableViews.push(item.caCoords, item.backbone4Coords, item.observedBackboneMask);
    }
    const transferables = collectOwnedTransferables(transferableViews);

    return {
      outputs: { sequences: seqSet, structure: packedStructure },
      transferables,
    };
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
