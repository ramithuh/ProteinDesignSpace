/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/ligandmpnn-webgpu (MIT)
// Context-aware inverse folding with non-protein ligands, metals, cofactors & nucleic acids.

import {
  computeChainLengths,
  parseFixedPositionsAndChains,
} from '$lib/engine/mpnn-spec';
import {
  computeStructurePayloadDigest,
  type CollectionMember,
} from '$lib/engine/collections';
import { deriveStableItemSeed, planMpnnFanout } from '$lib/engine/work-plan';
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
      omitAAs?: string;
      useAtomContext?: boolean;
    },
  ): Promise<LigandDesignResult>;
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
  unlockInherited: boolean,
): { fixedTokens: string[]; propagatedLockedChains: string[] } {
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

  const { fixedPositions: userRefs, lockedChains: userLockedChains } =
    parseFixedPositionsAndChains(fixedPositionsSpec, defaultChain);

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

  return {
    fixedTokens: [...fixedSet],
    propagatedLockedChains: [...lockedChainSet],
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ligandMod: any = null;
let engine: LigandEngine | null = null;
let loadedKey: string | null = null;

async function ensureEngine(
  id: number,
  checkpoint: Checkpoint,
  precision: Precision,
): Promise<LigandEngine> {
  const key = `${checkpoint}:${precision}`;
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

  if (!ligandMod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading LigandMPNN WebGPU modules...',
    });
    const moduleUrl = new URL('/vendor/ligandmpnn/webgpu/index.js?v=2', self.location.origin).href;
    ligandMod = await import(/* @vite-ignore */ moduleUrl);
  }

  const envBaseUrl = (import.meta.env.VITE_LIGANDMPNN_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  self.postMessage({
    id,
    type: 'progress',
    phase: 'weights',
    progress: 0.06,
    detail: `Fetching LigandMPNN (${checkpoint} · ${precision.toUpperCase()}) weights...`,
  });

  engine = (await ligandMod.createLigandMpnnEngine({
    baseUrl,
    variant: checkpoint,
    precision,
    cache: true,
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
        detail: `LigandMPNN weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
      });
    },
  })) as LigandEngine;
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
        throw new Error('LigandMPNN requires an upstream StructureArtifact.');
      }

      const p = msg.params ?? {};
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
      const omitAAs = sanitizeOmitAAs(p.omitAminoAcids);
      const useLigandContext = p.useLigandContext !== false;
      const unlockInherited = Boolean(p.unlockInherited);
      const rawSeed = Number(p.seed ?? 42);
      const seed = Number.isFinite(rawSeed) ? Math.round(rawSeed) : 42;

      const activeEngine = await ensureEngine(msg.id, checkpoint, precision);
      const modelLabel = `ligandmpnn/${checkpoint}`;

      const collectionItems =
        structure.collection?.items && structure.collection.items.length > 1
          ? structure.collection.items
          : null;

      if (!collectionItems) {
        const structureText = structure.cifText ?? structure.pdbText;
        const { fixedTokens, propagatedLockedChains } = buildLigandConstraints(
          structure,
          structureText,
          fixedPositionsSpec,
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
          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'sample',
            progress: i / numSequences,
            detail: `LigandMPNN designing seq ${i + 1}/${numSequences}${contextTag}...`,
          });

          const res = await activeEngine.design(structureText, {
            numSequences: 1,
            temperature,
            seed: seed + i,
            useAtomContext: useLigandContext,
            ...(fixedTokens.length > 0 ? { fixedPositions: fixedTokens } : {}),
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
            score: Number.isFinite(sc?.global) ? sc.global : null,
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
      const totalPlanned = Math.max(1, fanoutPlan.totalPlannedSequences);
      let completedSeqs = 0;

      for (let bIdx = 0; bIdx < fanoutPlan.perBackbonePlans.length; bIdx++) {
        const subplan = fanoutPlan.perBackbonePlans[bIdx];
        const backbone = backboneById.get(subplan.sourceStructure.artifactId) ?? collectionItems[bIdx];
        const text = backbone?.cifText ?? backbone?.pdbText;
        if (!backbone || !text) {
          throw new Error(`Backbone "${subplan.sourceStructure.artifactId}" is missing structure text for LigandMPNN.`);
        }

        const { fixedTokens } = buildLigandConstraints(
          backbone,
          text,
          fixedPositionsSpec,
          unlockInherited,
        );
        const backboneCandidates: SequenceCandidate[] = [];

        for (let q = 0; q < subplan.items.length; q++) {
          const item = subplan.items[q];
          const itemSeed = deriveStableItemSeed(seed, 'mpnn', item.key);
          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'sample',
            progress: completedSeqs / totalPlanned,
            detail: `Backbone ${bIdx + 1}/${fanoutPlan.backboneCount} · LigandMPNN seq ${q + 1}/${subplan.items.length}`,
          });

          const res = await activeEngine.design(text, {
            numSequences: 1,
            temperature,
            seed: itemSeed,
            useAtomContext: useLigandContext,
            ...(fixedTokens.length > 0 ? { fixedPositions: fixedTokens } : {}),
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
            score: Number.isFinite(sc?.global) ? sc.global : null,
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
        ...(!unlockInherited && structure.lockedChains && structure.lockedChains.length > 0
          ? { lockedChains: [...structure.lockedChains] }
          : {}),
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
