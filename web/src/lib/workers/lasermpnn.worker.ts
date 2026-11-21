/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/lasermpnn-webgpu (MIT)
// Joint sequence design + all-atom sidechain packing (with ligand context & explicit hydrogens).

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  computeChainLengths,
  parseFixedPositionsAndChains,
} from '$lib/engine/mpnn-spec';
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
      seed?: number;
      fixedPositions?: string[];
      designChains?: string[];
      omitAAs?: string;
    },
  ): Promise<LaserDesignResult>;
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
} {
  const useCif = Boolean(structure.cifText);
  const sourceResidues =
    structure.sourceResidues && structure.sourceResidues.length > 0
      ? structure.sourceResidues
      : parseStructureText(structureText).sourceResidues;

  const chainList: string[] = [];
  const presentChains = new Set<string>();
  const upperToChain = new Map<string, string>();
  for (const sr of sourceResidues) {
    const ch = useCif ? sr.authChainId : sr.projectedChainId;
    chainList.push(ch);
    presentChains.add(ch);
    if (!upperToChain.has(ch.toUpperCase())) upperToChain.set(ch.toUpperCase(), ch);
  }
  const allChains = [...presentChains];
  const defaultChain = allChains[0] ?? 'A';
  const chainLengths = computeChainLengths(chainList);

  const { fixedPositions: userRefs, lockedChains: userLockedChains } =
    parseFixedPositionsAndChains(fixedPositionsSpec, defaultChain);

  const fixedSet = new Set<string>();
  const lockedChainSet = new Set<string>();

  for (const ch of userLockedChains) {
    const resolved = presentChains.has(ch) ? ch : upperToChain.get(ch.toUpperCase());
    if (resolved) lockedChainSet.add(resolved);
  }

  // LASErMPNN residue IDs use `${chain}:${resnum}${icode}` format
  for (const ref of userRefs) {
    if (typeof ref === 'number') {
      const sr = sourceResidues[ref];
      if (sr) {
        const ch = useCif ? sr.authChainId : sr.projectedChainId;
        fixedSet.add(`${ch}:${sr.resNum}${sr.insCode ?? ''}`);
      }
    } else {
      const ch = presentChains.has(ref.chain)
        ? ref.chain
        : upperToChain.get(ref.chain.toUpperCase()) ?? ref.chain;
      fixedSet.add(`${ch}:${ref.number}${ref.insertionCode ?? ''}`);
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
          fixedSet.add(`${ch}:${sr.resNum}${sr.insCode ?? ''}`);
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
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let laserMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let weightsMod: any = null;
let engine: LaserEngine | null = null;
let loadedPrecision: Precision | null = null;

async function ensureEngine(id: number, precision: Precision): Promise<LaserEngine> {
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
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading LASErMPNN WebGPU modules...',
    });
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

    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.06,
      detail: `Fetching LASErMPNN (${precision.toUpperCase()}) weights...`,
    });

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
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.92, 0.08 + frac * 0.84),
            detail: `LASErMPNN weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
          });
        },
      });

      self.postMessage({
        id,
        type: 'progress',
        phase: 'init',
        progress: 0.94,
        detail: 'Uploading LASErMPNN weights to GPU...',
      });

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
      loadedPrecision = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const structure = msg.inputs?.structure as StructureArtifact | undefined;
      const structureText = structure?.cifText ?? structure?.pdbText;
      if (!structure || !structureText) {
        throw new Error('LASErMPNN requires an upstream StructureArtifact.');
      }

      const p = msg.params ?? {};
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

      const { fixedIds, designChains, propagatedLockedChains, chainLengths } =
        buildLaserConstraints(structure, structureText, fixedPositionsSpec, unlockInherited);

      const activeEngine = await ensureEngine(msg.id, precision);

      const entries: Array<{ candidate: SequenceCandidate; pdbText: string }> = [];
      for (let i = 0; i < numSequences; i++) {
        const itemSeed = (seed + i) >>> 0;
        self.postMessage({
          id: msg.id,
          type: 'progress',
          phase: 'sample',
          progress: i / numSequences,
          detail: `LASErMPNN designing & packing sidechains ${i + 1}/${numSequences}...`,
        });

        const res = await activeEngine.design(structureText, {
          numSequences: 1,
          temperature,
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
      const best = entries[0];
      const parsedBest = parseStructureText(best.pdbText);
      const geometry = computeGeometryStats(
        parsedBest.caCoords,
        parsedBest.chainIds,
        parsedBest.resNums,
        parsedBest.insCodes,
      );

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

      const packedStructure: StructureArtifact = {
        kind: 'StructureArtifact',
        id: `lasermpnn-struct-${structure.id}-${seed}-${runHash}`,
        label: `LASErMPNN packed (${precision.toUpperCase()} · L=${parsedBest.nResidues})`,
        sourceModelId: 'lasermpnn',
        parentStructureId: structure.id,
        length: parsedBest.nResidues,
        sequence: parsedBest.sequence,
        caCoords: parsedBest.caCoords,
        backbone4Coords: parsedBest.backbone4Coords,
        pdbText: best.pdbText,
        chainIds: parsedBest.chainIds,
        chainLengths: parsedBest.diagnostics.chainLengths,
        insCodes: parsedBest.insCodes,
        sourceResidues: parsedBest.sourceResidues,
        chainIdMap: parsedBest.chainIdMap,
        atomPresence: parsedBest.atomPresence,
        observedBackboneMask: parsedBest.observedBackboneMask,
        diagnostics: parsedBest.diagnostics,
        bfactorSemantics: 'none',
        atomGranularity: 'all-atom',
        molecularContext: {
          ...structure.molecularContext,
          ...(parsedBest.diagnostics.hasLigands
            ? { ligands: parsedBest.diagnostics.ligandNames }
            : {}),
          ...(parsedBest.diagnostics.hasNucleicAcids
            ? { nucleicAcidChains: parsedBest.diagnostics.nucleicAcidChains }
            : {}),
          hasExplicitHydrogens: true,
        },
        provenance: {
          quantization: precision,
        },
        ...(propagatedLockedChains.length > 0 ? { lockedChains: propagatedLockedChains } : {}),
        ...(structure.chainRoles ? { chainRoles: structure.chainRoles } : {}),
        geometry,
      };

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

      const transferables = collectOwnedTransferables([
        packedStructure.caCoords,
        packedStructure.backbone4Coords,
        packedStructure.observedBackboneMask,
      ]);

      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          outputs: { sequences: seqSet, structure: packedStructure },
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
