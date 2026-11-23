/// <reference lib="webworker" />
// Layer 3 Plugin Worker — ligandmpnn (ramithuh/ligandmpnn-webgpu, MIT)
// Context-aware inverse folding with non-protein ligands, metals, cofactors & nucleic acids
// plus pure-JS CPU fallback when WebGPU is unavailable or loses device.
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress, and disposal.

import {
  computeChainLengths,
  parseFixedPositionsAndChains,
  parseTiedPositions,
} from '$lib/engine/mpnn-spec';
import {
  computeStructurePayloadDigest,
  type CollectionMember,
} from '$lib/engine/collections';
import { deriveStableItemSeed, planMpnnFanout } from '$lib/engine/work-plan';
import { isWebGpuDeviceLostError } from '$lib/engine/webgpu-device';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/ligandmpnn-webgpu/resolve/main';

type Checkpoint = 'ligandmpnn_v_32_020_25' | 'ligandmpnn_v_32_010_25';
type Precision = 'fp16' | 'int8' | 'fp32';

interface LigandDesignResult {
  sequences: string[];
  scores: Array<{ global: number; ligand: number }>;
  residues: Array<{
    chain: string;
    number: number;
    insertionCode: string;
    token: string;
  }>;
  timingMs: number;
}

interface LigandEngine {
  design(
    structureText: string,
    opts?: {
      numSequences?: number;
      temperature?: number;
      seed?: number;
      fixedPositions?: string[];
      tiedPositions?: string[][];
      omitAAs?: string;
      useAtomContext?: boolean;
    },
  ): Promise<LigandDesignResult>;
  destroy(): void;
}

function normalizeCheckpoint(raw: unknown): Checkpoint {
  const val = String(raw ?? 'ligandmpnn_v_32_020_25').trim();
  if (val === 'ligandmpnn_v_32_010_25' || val === 'ligandmpnn_v_32_020_25') return val;
  return 'ligandmpnn_v_32_020_25';
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

function buildLigandConstraints(
  backbone: StructureArtifact,
  structureText: string,
  fixedPositionsSpec: string,
  tiedPositionsSpec: string,
  unlockInherited: boolean,
): { fixedTokens: string[]; tiedGroups: string[][]; propagatedLockedChains: string[] } {
  const useCif = Boolean(backbone.cifText);
  const sourceResidues =
    backbone.sourceResidues && backbone.sourceResidues.length > 0
      ? backbone.sourceResidues
      : parseStructureText(structureText).sourceResidues;

  const presentChains = new Set<string>();
  const upperToChain = new Map<string, string>();
  for (const sr of sourceResidues) {
    const ch = useCif ? sr.authChainId : sr.projectedChainId;
    presentChains.add(ch);
    if (!upperToChain.has(ch.toUpperCase())) upperToChain.set(ch.toUpperCase(), ch);
  }

  const defaultChain = sourceResidues[0]
    ? useCif
      ? sourceResidues[0].authChainId
      : sourceResidues[0].projectedChainId
    : 'A';

  const orderedResidues = sourceResidues.map((sr) => ({
    chain: useCif ? sr.authChainId : sr.projectedChainId,
    number: sr.resNum,
    insertionCode: sr.insCode ?? '',
  }));

  const { fixedPositions: userRefs, lockedChains: userLockedChains } =
    parseFixedPositionsAndChains(fixedPositionsSpec, defaultChain, orderedResidues);

  const fixedSet = new Set<string>();
  const lockedChainSet = new Set<string>();

  for (const ch of userLockedChains) {
    const resolved = presentChains.has(ch) ? ch : upperToChain.get(ch.toUpperCase()) ?? ch;
    fixedSet.add(resolved);
    if (presentChains.has(resolved)) lockedChainSet.add(resolved);
  }

  for (const ref of userRefs) {
    if (typeof ref === 'number') {
      const sr = sourceResidues[ref];
      if (sr) {
        const ch = useCif ? sr.authChainId : sr.projectedChainId;
        fixedSet.add(`${ch}${sr.resNum}${sr.insCode ?? ''}`);
      }
    } else {
      const ch = presentChains.has(ref.chain)
        ? ref.chain
        : upperToChain.get(ref.chain.toUpperCase()) ?? ref.chain;
      fixedSet.add(`${ch}${ref.number}${ref.insertionCode ?? ''}`);
    }
  }

  if (!unlockInherited) {
    if (Array.isArray(backbone.lockedChains)) {
      for (const rawCh of backbone.lockedChains) {
        const ch = String(rawCh).trim();
        const resolved = presentChains.has(ch) ? ch : upperToChain.get(ch.toUpperCase());
        if (resolved) {
          fixedSet.add(resolved);
          lockedChainSet.add(resolved);
        }
      }
    }
    if (Array.isArray(backbone.fixedResidueIndices)) {
      for (const oneBased of backbone.fixedResidueIndices) {
        const idx = Number(oneBased) - 1;
        if (Number.isInteger(idx) && idx >= 0 && idx < sourceResidues.length) {
          const sr = sourceResidues[idx];
          const ch = useCif ? sr.authChainId : sr.projectedChainId;
          fixedSet.add(`${ch}${sr.resNum}${sr.insCode ?? ''}`);
        }
      }
    }
  }

  const rawTiedGroups = parseTiedPositions(tiedPositionsSpec);
  const tiedGroups: string[][] = [];
  for (const grp of rawTiedGroups) {
    const tokens: string[] = [];
    for (const ref of grp) {
      if (typeof ref === 'number') {
        const sr = sourceResidues[ref];
        if (sr) {
          const ch = useCif ? sr.authChainId : sr.projectedChainId;
          tokens.push(`${ch}${sr.resNum}${sr.insCode ?? ''}`);
        }
      } else {
        const ch = presentChains.has(ref.chain)
          ? ref.chain
          : upperToChain.get(ref.chain.toUpperCase()) ?? ref.chain;
        tokens.push(`${ch}${ref.number}${ref.insertionCode ?? ''}`);
      }
    }
    if (tokens.length >= 2) tiedGroups.push(tokens);
  }

  return {
    fixedTokens: [...fixedSet],
    tiedGroups,
    propagatedLockedChains: [...lockedChainSet],
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ligandMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ligandCpuMod: any = null;
let engine: LigandEngine | null = null;
let loadedKey: string | null = null;
let loadedBackend: 'webgpu' | 'cpu' | null = null;

async function ensureEngine(
  ctx: WorkerRunContext,
  checkpoint: Checkpoint,
  precision: Precision,
  forceCpu = false,
): Promise<{ engine: LigandEngine; backend: 'webgpu' | 'cpu' }> {
  const key = `${checkpoint}:${precision}`;
  if (engine && loadedKey === key && (!forceCpu || loadedBackend === 'cpu')) {
    return { engine, backend: loadedBackend ?? 'webgpu' };
  }

  if (engine) {
    try {
      engine.destroy();
    } catch {
      /* ignore */
    }
    engine = null;
    loadedKey = null;
    loadedBackend = null;
  }

  const envBaseUrl = (import.meta.env.VITE_LIGANDMPNN_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const makeProgressCb = (backendTag: string) => (prog: { loaded?: number; total?: number }) => {
    const loaded = prog.loaded ?? 0;
    const total = prog.total ?? 0;
    const frac = total > 0 ? loaded / total : 0.5;
    const mb = (loaded / 1024 / 1024).toFixed(1);
    const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
    ctx.progress(
      'weights',
      Math.min(0.92, 0.08 + frac * 0.84),
      `LigandMPNN weights (${precision.toUpperCase()}${backendTag}): ${mb} / ${totMb} MB`,
    );
  };

  if (!forceCpu && typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu) {
    try {
      if (!ligandMod) {
        ctx.progress('init', 0.02, 'Loading LigandMPNN WebGPU modules...');
        const moduleUrl = new URL('/vendor/ligandmpnn/webgpu/index.js?v=2', self.location.origin).href;
        ligandMod = await import(/* @vite-ignore */ moduleUrl);
      }

      ctx.progress(
        'weights',
        0.06,
        `Fetching LigandMPNN (${checkpoint} · ${precision.toUpperCase()}) weights...`,
      );

      engine = (await ligandMod.createLigandMpnnEngine({
        baseUrl,
        variant: checkpoint,
        precision,
        cache: true,
        onProgress: makeProgressCb(''),
      })) as LigandEngine;
      loadedKey = key;
      loadedBackend = 'webgpu';
      return { engine, backend: 'webgpu' };
    } catch (gpuErr) {
      const msg = gpuErr instanceof Error ? gpuErr.message : String(gpuErr);
      if (/HTTP \d+|fetch|network|integrity|sha-?256|manifest/i.test(msg)) {
        throw gpuErr;
      }
      // Fall through cleanly to pure-JS CPU fallback when WebGPU adapter/device fails
    }
  }

  if (!ligandCpuMod) {
    ctx.progress('init', 0.02, 'WebGPU unavailable — loading LigandMPNN CPU fallback...');
    const cpuUrl = new URL('/vendor/ligandmpnn/cpu/index.js?v=1', self.location.origin).href;
    ligandCpuMod = await import(/* @vite-ignore */ cpuUrl);
  }

  ctx.progress(
    'weights',
    0.06,
    `Fetching LigandMPNN (${checkpoint} · ${precision.toUpperCase()} · CPU fallback) weights...`,
  );

  engine = (await ligandCpuMod.createLigandMpnnCpuEngine({
    baseUrl,
    variant: checkpoint,
    precision,
    onProgress: makeProgressCb(' · CPU fallback'),
  })) as LigandEngine;
  loadedKey = key;
  loadedBackend = 'cpu';
  return { engine, backend: 'cpu' };
}

createWorkerHarness({
  async run(ctx) {
    const structure = ctx.inputs?.structure as StructureArtifact | undefined;
    if (!structure || (!structure.cifText && !structure.pdbText)) {
      throw new Error('LigandMPNN requires an upstream StructureArtifact.');
    }

    const p = ctx.params ?? {};
    const checkpoint = normalizeCheckpoint(p.checkpoint);
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
    const tiedPositionsSpec = String(p.tiedPositions ?? '').trim();
    const omitAAs = sanitizeOmitAAs(p.omitAminoAcids);
    const useLigandContext = p.useLigandContext !== false;
    const unlockInherited = Boolean(p.unlockInherited);
    const rawSeed = Number(p.seed ?? 42);
    const seed = Number.isFinite(rawSeed) ? Math.round(rawSeed) : 42;

    let { engine: activeEngine, backend } = await ensureEngine(ctx, checkpoint, precision);
    const designWithFallback = async (
      structureText: string,
      opts: Parameters<LigandEngine['design']>[1],
    ): Promise<LigandDesignResult> => {
      try {
        return await activeEngine.design(structureText, opts);
      } catch (designErr) {
        if (backend === 'webgpu' && isWebGpuDeviceLostError(designErr)) {
          ({ engine: activeEngine, backend } = await ensureEngine(ctx, checkpoint, precision, true));
          return await activeEngine.design(structureText, opts);
        }
        throw designErr;
      }
    };
    const modelLabel = `ligandmpnn/${checkpoint}`;

    const collectionItems =
      structure.collection?.items && structure.collection.items.length > 1
        ? structure.collection.items
        : null;

    if (!collectionItems) {
      const structureText = structure.cifText ?? structure.pdbText;
      const { fixedTokens, tiedGroups, propagatedLockedChains } = buildLigandConstraints(
        structure,
        structureText,
        fixedPositionsSpec,
        tiedPositionsSpec,
        unlockInherited,
      );

      const hasNonProteinContext = Boolean(
        structure.diagnostics?.hasLigands ||
          (structure.diagnostics?.ligandNames?.length ?? 0) > 0 ||
          structure.diagnostics?.hasNucleicAcids ||
          (structure.diagnostics?.nucleicAcidChains?.length ?? 0) > 0,
      );
      const contextTag = !useLigandContext
        ? ' (protein-only)'
        : hasNonProteinContext
          ? ' (holo context)'
          : ' (apo — 0 ligands in input)';

      const candidates: SequenceCandidate[] = [];
      for (let i = 0; i < numSequences; i++) {
        if (ctx.signal.aborted) throw new Error('Workflow run cancelled');
        ctx.progress(
          'sample',
          i / numSequences,
          `LigandMPNN designing seq ${i + 1}/${numSequences}${contextTag}${backend === 'cpu' ? ' · CPU fallback' : ''}...`,
        );

        const res = await designWithFallback(structureText, {
          numSequences: 1,
          temperature,
          seed: seed + i,
          useAtomContext: useLigandContext,
          ...(fixedTokens.length > 0 ? { fixedPositions: fixedTokens } : {}),
          ...(tiedGroups.length > 0 ? { tiedPositions: tiedGroups } : {}),
          ...(omitAAs ? { omitAAs } : {}),
        });

        const seq = res.sequences[0];
        const sc = res.scores[0];
        const chainLengths = computeChainLengths(res.residues.map((r) => r.chain));
        const candChainLengths =
          chainLengths.length > 0 &&
          chainLengths.reduce((a, b) => a + b, 0) === seq.length
            ? chainLengths
            : undefined;

        candidates.push({
          id: `ligandmpnn-${i + 1}`,
          sourceStructureId: structure.id,
          sampleIndex: i,
          sequence: seq,
          score: Number.isFinite(sc?.ligand) ? sc.ligand : Number.isFinite(sc?.global) ? sc.global : null,
          globalScore: Number.isFinite(sc?.global) ? sc.global : null,
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
          checkpoint,
          precision,
          numSequences,
          temperature,
          fixedTokens,
          tiedGroups,
          omitAAs: omitAAs ?? null,
          useLigandContext,
          unlockInherited,
          seed,
        }),
      );

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `ligandmpnn-seq-${structure.id}-${seed}-${runHash}`,
        parentStructureId: structure.id,
        sourceStructureId: structure.id,
        sourceModelId: modelLabel,
        ...(propagatedLockedChains.length > 0 ? { lockedChains: propagatedLockedChains } : {}),
        ...(structure.chainRoles ? { chainRoles: structure.chainRoles } : {}),
        candidates,
      };

      return { outputs: { sequences: seqSet } };
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
        checkpoint,
        precision,
        fixedPositions: fixedPositionsSpec,
        omitAminoAcids: omitAAs ?? '',
        useLigandContext,
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
        throw new Error(`Backbone "${subplan.sourceStructure.artifactId}" is missing structure text for LigandMPNN.`);
      }

      const { fixedTokens, tiedGroups, propagatedLockedChains } = buildLigandConstraints(
        backbone,
        text,
        fixedPositionsSpec,
        tiedPositionsSpec,
        unlockInherited,
      );
      for (const ch of propagatedLockedChains) allPropagatedLocked.add(ch);
      const backboneCandidates: SequenceCandidate[] = [];

      for (let q = 0; q < subplan.items.length; q++) {
        if (ctx.signal.aborted) throw new Error('Workflow run cancelled');
        const item = subplan.items[q];
        const itemSeed = deriveStableItemSeed(seed, 'mpnn', item.key);
        ctx.progress(
          'sample',
          completedSeqs / totalPlanned,
          `Backbone ${bIdx + 1}/${fanoutPlan.backboneCount} · LigandMPNN seq ${q + 1}/${subplan.items.length}${backend === 'cpu' ? ' · CPU fallback' : ''}`,
        );

        const res = await designWithFallback(text, {
          numSequences: 1,
          temperature,
          seed: itemSeed,
          useAtomContext: useLigandContext,
          ...(fixedTokens.length > 0 ? { fixedPositions: fixedTokens } : {}),
          ...(tiedGroups.length > 0 ? { tiedPositions: tiedGroups } : {}),
          ...(omitAAs ? { omitAAs } : {}),
        });
        completedSeqs++;

        const seq = res.sequences[0];
        const sc = res.scores[0];
        const chainLengths = computeChainLengths(res.residues.map((r) => r.chain));
        const candChainLengths =
          chainLengths.length > 0 &&
          chainLengths.reduce((a, b) => a + b, 0) === seq.length
            ? chainLengths
            : undefined;

        backboneCandidates.push({
          id: `d${subplan.designIndex + 1}-ligandmpnn-${item.sequenceIndex + 1}`,
          key: item.key,
          sequenceKey: item.key,
          designKey: subplan.designKey,
          designIndex: subplan.designIndex,
          sourceStructureId: backbone.id,
          sampleIndex: item.sequenceIndex,
          sequence: seq,
          score: Number.isFinite(sc?.ligand) ? sc.ligand : Number.isFinite(sc?.global) ? sc.global : null,
          globalScore: Number.isFinite(sc?.global) ? sc.global : null,
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
      id: `ligandmpnn-seq-col-${fanoutPlan.planDigest.slice(0, 16)}`,
      parentStructureId: structure.id,
      sourceStructureId: structure.id,
      sourceStructureIds: collectionItems.map((b) => b.id),
      sourceModelId: modelLabel,
      collectionKind: 'independent-candidates',
      ...(allPropagatedLocked.size > 0 ? { lockedChains: [...allPropagatedLocked] } : {}),
      ...(structure.chainRoles ? { chainRoles: { ...structure.chainRoles } } : {}),
      candidates: allCandidates,
    };

    return { outputs: { sequences: seqSet } };
  },

  async dispose() {
    try {
      engine?.destroy();
    } catch {
      /* ignore */
    }
    engine = null;
    loadedKey = null;
    loadedBackend = null;
  },
});

export {};
