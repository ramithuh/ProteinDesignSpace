/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/boltzgen-webgpu (MIT · HannesStark/boltzgen)
// Runs dynamic BoltzGen-1 all-atom diffusion for unconditional, binder, and motif design.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  parseFixedPositionsAndChains,
  type OrderedStructureResidue,
} from '$lib/engine/mpnn-spec';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { StructureArtifact, TrajectoryFramesArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/boltzgen-webgpu/resolve/main';

type Precision = 'int8' | 'fp16';
type Variant = 'diverse' | 'adherence';
type TaskMode = 'unconditional' | 'binder' | 'motif';

interface BoltzAtomMeta {
  token: number;
  name: string;
  residue: string;
  chain: string;
  sourceChain?: string;
  residueId: string;
  designed: boolean;
}

const BB4_SET = new Set(['N', 'CA', 'C', 'O']);

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
  return String(raw ?? 'int8').trim().toLowerCase() === 'fp16' ? 'fp16' : 'int8';
}

function normalizeVariant(raw: unknown): Variant {
  return String(raw ?? 'diverse').trim().toLowerCase() === 'adherence' ? 'adherence' : 'diverse';
}

function normalizeTaskMode(raw: unknown): TaskMode {
  const val = String(raw ?? 'unconditional').trim().toLowerCase();
  if (val === 'binder' || val === 'motif') return val;
  return 'unconditional';
}

function formatAtomName(name: string): string {
  if (name.length >= 4) return name.slice(0, 4);
  return ` ${name.padEnd(3, ' ')}`;
}

function boltzAtomsToPdb(coords: Float32Array, atoms: BoltzAtomMeta[]): string {
  const lines: string[] = [];
  let serial = 1;
  let prevChain = '';
  for (let i = 0; i < atoms.length; i++) {
    const a = atoms[i];
    // For designed (UNK) residues, emit canonical backbone N/CA/C/O as GLY so downstream
    // Mol* cartoon & ProteinMPNN/LocalFold parsers consume valid backbone geometry cleanly.
    if (a.designed && !BB4_SET.has(a.name)) continue;
    const x = coords[i * 3];
    const y = coords[i * 3 + 1];
    const z = coords[i * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    const chain = (a.chain || 'A')[0];
    if (prevChain && chain !== prevChain) {
      lines.push('TER');
    }
    prevChain = chain;
    const resName = a.designed ? 'GLY' : (a.residue || 'ALA').slice(0, 3).padStart(3, ' ');
    const resNumMatch = /^-?\d+/.exec(String(a.residueId || a.token + 1));
    const resSeq = (resNumMatch ? resNumMatch[0] : String(a.token + 1)).slice(0, 4).padStart(4, ' ');
    const ins = String(a.residueId || '').replace(/^-?\d+/, '').slice(0, 1) || ' ';
    const elem = a.name.replace(/[^A-Za-z]/g, '').slice(0, 1).toUpperCase() || 'C';
    const line =
      `ATOM  ${String(serial).padStart(5, ' ')} ${formatAtomName(a.name)} ${resName} ${chain}${resSeq}${ins}   ` +
      `${x.toFixed(3).padStart(8, ' ')}${y.toFixed(3).padStart(8, ' ')}${z.toFixed(3).padStart(8, ' ')}` +
      `  1.00  0.00          ${elem.padStart(2, ' ')}`;
    lines.push(line);
    serial++;
  }
  lines.push('END');
  return `${lines.join('\n')}\n`;
}

function extractCaFrame(coords: Float32Array, atoms: BoltzAtomMeta[]): Float32Array {
  const caIdx: number[] = [];
  for (let i = 0; i < atoms.length; i++) {
    if (atoms[i].name === 'CA') caIdx.push(i);
  }
  const out = new Float32Array(caIdx.length * 3);
  for (let i = 0; i < caIdx.length; i++) {
    const idx = caIdx[i] * 3;
    out[i * 3] = coords[idx];
    out[i * 3 + 1] = coords[idx + 1];
    out[i * 3 + 2] = coords[idx + 2];
  }
  return out;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let bgMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let ownedDevice: GPUDevice | null = null;
let loadedKey: string | null = null;
let activeAbort: AbortController | null = null;

function disposeEngine() {
  try {
    engine?.destroy();
  } catch {
    /* ignore */
  }
  engine = null;
  try {
    ownedDevice?.destroy();
  } catch {
    /* ignore */
  }
  ownedDevice = null;
  loadedKey = null;
}

async function ensureEngine(id: number, variant: Variant, precision: Precision) {
  const key = `${variant}:${precision}`;
  if (engine && loadedKey === key) return engine;
  disposeEngine();

  if (!bgMod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading BoltzGen-1 WebGPU modules...',
    });
    const base = `${self.location.origin}/vendor/boltzgen`;
    bgMod = await import(/* @vite-ignore */ `${base}/webgpu/index.js`);
  }

  const { adapter } = await requestResilientWebGpuAdapter({ allowSoftwareFallback: true });
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  ownedDevice = device;

  const envBaseUrl = (import.meta.env.VITE_BOLTZGEN_WEIGHTS_URL as string | undefined)?.trim();
  const baseRoot = (envBaseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
  const bundleUrl = `${baseRoot}/${variant}/${precision}/`;

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
      progress: 0.06,
      detail: `Fetching BoltzGen-1 (${variant} · ${precision.toUpperCase()}) weights...`,
    });

    try {
      return await bgMod.createBoltzGenEngine({
        baseUrl: bundleUrl,
        precision,
        device,
        cache: 'none',
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { loaded?: number; total?: number; shard?: number; shards?: number }) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac =
            total > 0
              ? loaded / total
              : prog.shards && prog.shard !== undefined
                ? prog.shard / prog.shards
                : 0.4;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.45, 0.06 + frac * 0.39),
            detail:
              total > 0
                ? `BoltzGen weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
                : `Loading BoltzGen weights (${variant} · ${precision.toUpperCase()})...`,
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
    loadedKey = key;
    return engine;
  } catch (err) {
    disposeEngine();
    throw err;
  }
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'dispose') {
      activeAbort?.abort();
      activeAbort = null;
      disposeEngine();
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const p = msg.params ?? {};
      const taskMode = normalizeTaskMode(p.taskMode);
      const variant = normalizeVariant(p.variant);
      const precision = normalizePrecision(p.precision);
      const length = Math.max(12, Math.min(160, Math.round(Number(p.length ?? 48))));
      const numDesigns = Math.max(1, Math.min(4, Math.round(Number(p.numDesigns ?? 1))));
      const numSteps = Math.max(3, Math.min(100, Math.round(Number(p.numSteps ?? 20))));
      const recyclingSteps = Math.max(0, Math.min(3, Math.round(Number(p.recyclingSteps ?? 0))));
      const hotspotsSpec = String(p.hotspots ?? '').trim();
      const motifOffset = Math.max(0, Math.round(Number(p.motifOffset ?? 10)));
      const seed = Math.max(0, Math.round(Number(p.seed ?? 42)));

      const upstreamStruct = msg.inputs?.structure as StructureArtifact | undefined;
      if ((taskMode === 'binder' || taskMode === 'motif') && !upstreamStruct) {
        throw new Error(`BoltzGen-1 in "${taskMode}" mode requires a connected target structure.`);
      }

      let resolvedHotspots: string[] = [];
      if (taskMode === 'binder' && upstreamStruct && hotspotsSpec) {
        const targetResidues =
          upstreamStruct.sourceResidues && upstreamStruct.sourceResidues.length > 0
            ? upstreamStruct.sourceResidues
            : parseStructureText(upstreamStruct.pdbText).sourceResidues;
        const orderedTargetResidues: OrderedStructureResidue[] = targetResidues.map((sr) => ({
          chain: sr.authChainId,
          number: sr.resNum,
          insertionCode: sr.insCode ?? '',
        }));
        const authToProjected = new Map<string, string>();
        for (const sr of targetResidues) {
          if (!authToProjected.has(sr.authChainId)) authToProjected.set(sr.authChainId, sr.projectedChainId);
          if (!authToProjected.has(sr.authChainId.toUpperCase())) {
            authToProjected.set(sr.authChainId.toUpperCase(), sr.projectedChainId);
          }
        }
        const defaultTargetChain = orderedTargetResidues[0]?.chain ?? 'A';
        const { fixedPositions: hotspotRefs, lockedChains: hotspotChains } =
          parseFixedPositionsAndChains(hotspotsSpec, defaultTargetChain, orderedTargetResidues);
        const keySet = new Set<string>();
        for (const ch of hotspotChains) {
          const proj = authToProjected.get(ch) ?? authToProjected.get(ch.toUpperCase()) ?? ch;
          for (const sr of targetResidues) {
            if (sr.projectedChainId.toUpperCase() === proj.toUpperCase()) {
              keySet.add(`${sr.projectedChainId}:${sr.resNum}${sr.insCode ?? ''}`);
            }
          }
        }
        for (const ref of hotspotRefs) {
          if (typeof ref === 'number') continue;
          const proj =
            authToProjected.get(ref.chain) ?? authToProjected.get(ref.chain.toUpperCase()) ?? ref.chain;
          keySet.add(`${proj}:${ref.number}${ref.insertionCode ?? ''}`);
        }
        resolvedHotspots = [...keySet];
        if (resolvedHotspots.length === 0) {
          throw new Error(`BoltzGen hotspots "${hotspotsSpec}" matched 0 target residues.`);
        }
      }

      let effectiveLength = length;
      let motifIndices: number[] | undefined;
      if (taskMode === 'motif' && upstreamStruct) {
        const targetLen = upstreamStruct.length;
        effectiveLength = Math.max(length, targetLen + motifOffset + 4);
        const startIdx = Math.min(motifOffset, effectiveLength - targetLen);
        motifIndices = Array.from({ length: targetLen }, (_, i) => startIdx + i);
      }

      const activeEngine = await ensureEngine(msg.id, variant, precision);
      const controller = new AbortController();
      activeAbort = controller;

      const items: StructureArtifact[] = [];
      let primaryTrajFrames: Float32Array[] = [];
      let primaryTrajLabels: string[] = [];

      const runHash = fnv1a(
        JSON.stringify({
          taskMode,
          variant,
          precision,
          effectiveLength,
          numDesigns,
          numSteps,
          recyclingSteps,
          hotspots: taskMode === 'binder' ? hotspotsSpec : '',
          motifOffset,
          seed,
          upstreamId: upstreamStruct?.id ?? null,
        }),
      );

      try {
        for (let d = 0; d < numDesigns; d++) {
          const genRes = await activeEngine.generate({
            mode: taskMode,
            L: effectiveLength,
            numSteps,
            recyclingSteps,
            seed: seed + d,
            ...(taskMode !== 'unconditional' && upstreamStruct
              ? { targetPDB: upstreamStruct.pdbText }
              : {}),
            ...(resolvedHotspots.length > 0 ? { hotspots: resolvedHotspots } : {}),
            ...(motifIndices ? { motifIndices } : {}),
            signal: controller.signal,
            onProgress: (prog: { phase?: string; step?: number; total?: number; block?: number }) => {
              const step = prog.step ?? 0;
              const tot = prog.total ?? numSteps;
              const frac = tot > 0 ? Math.min(1, step / tot) : 0.2;
              const overall = 0.48 + ((d + frac) / numDesigns) * 0.5;
              self.postMessage({
                id: msg.id,
                type: 'progress',
                phase: prog.phase === 'trunk' ? 'trunk' : 'diffuse',
                progress: Math.min(0.98, overall),
                detail:
                  prog.phase === 'trunk'
                    ? `BoltzGen design ${d + 1}/${numDesigns}: pairformer trunk (${step}/${tot})`
                    : `BoltzGen design ${d + 1}/${numDesigns}: step ${step}/${tot}`,
              });
            },
          });

          const atoms = genRes.atoms as BoltzAtomMeta[];
          const pdbText = boltzAtomsToPdb(genRes.atom_coords as Float32Array, atoms);
          const parsed = parseStructureText(pdbText);
          const geometry = computeGeometryStats(
            parsed.caCoords,
            parsed.chainIds,
            parsed.resNums,
            parsed.insCodes,
          );
          const designKey = `design-${d + 1}`;
          const structId = `bg-${variant}-${taskMode}-${parsed.nResidues}aa-d${d + 1}-${runHash}`;

          let lockedChains: string[] | undefined;
          let chainRoles: Record<string, 'target' | 'binder' | 'scaffold'> | undefined;
          let fixedResidueIndices: number[] | undefined;
          let outMotifIndices: number[] | undefined;
          let nativeMotifCoords: number[][] | undefined;

          if (taskMode === 'binder') {
            const designedByChain = new Map<string, boolean>();
            for (const a of atoms) {
              const ch = (a.chain || 'A')[0];
              designedByChain.set(ch, (designedByChain.get(ch) ?? false) || Boolean(a.designed));
            }
            const locked: string[] = [];
            const roles: Record<string, 'target' | 'binder' | 'scaffold'> = {};
            for (const ch of [...new Set(parsed.chainIds)]) {
              if (designedByChain.get(ch)) {
                roles[ch] = 'binder';
              } else {
                roles[ch] = 'target';
                locked.push(ch);
              }
            }
            if (locked.length > 0) lockedChains = locked;
            chainRoles = roles;
          } else if (taskMode === 'motif' && motifIndices && upstreamStruct) {
            chainRoles = { A: 'scaffold' };
            fixedResidueIndices = motifIndices.map((idx) => idx + 1);
            outMotifIndices = [...motifIndices];
            nativeMotifCoords = motifIndices.map((_, k) => {
              const base = k * 3;
              if (base + 2 < upstreamStruct.caCoords.length) {
                return [
                  upstreamStruct.caCoords[base],
                  upstreamStruct.caCoords[base + 1],
                  upstreamStruct.caCoords[base + 2],
                ];
              }
              return [0, 0, 0];
            });
          } else {
            const roles: Record<string, 'target' | 'binder' | 'scaffold'> = {};
            for (const ch of [...new Set(parsed.chainIds)]) roles[ch] = 'scaffold';
            chainRoles = roles;
          }

          const artifact: StructureArtifact = {
            kind: 'StructureArtifact',
            id: structId,
            label: `BoltzGen (${variant} · ${parsed.nResidues} aa${numDesigns > 1 ? ` · #${d + 1}` : ''})`,
            sourceModelId: 'boltzgen',
            parentStructureId: upstreamStruct?.id,
            length: parsed.nResidues,
            sequence: parsed.sequence,
            caCoords: parsed.caCoords,
            backbone4Coords: parsed.backbone4Coords,
            pdbText: parsed.pdbText,
            chainIds: parsed.chainIds,
            chainLengths: parsed.diagnostics.chainLengths,
            insCodes: parsed.insCodes,
            sourceResidues: parsed.sourceResidues,
            observedBackboneMask: parsed.observedBackboneMask,
            diagnostics: parsed.diagnostics,
            geometry,
            atomGranularity: 'backbone4',
            bfactorSemantics: 'none',
            ...(lockedChains ? { lockedChains } : {}),
            ...(fixedResidueIndices ? { fixedResidueIndices } : {}),
            ...(chainRoles ? { chainRoles } : {}),
            ...(outMotifIndices && nativeMotifCoords
              ? { motifIndices: outMotifIndices, nativeMotifCoords }
              : {}),
            designKey,
            designIndex: d,
            seed: seed + d,
            provenance: {
              quantization: precision,
              diffusionStepsScheduled: numSteps,
              diffusionStepsRun: numSteps,
              recyclesRun: recyclingSteps,
            },
          };
          items.push(artifact);

          if (d === 0 && Array.isArray(genRes.denoised)) {
            primaryTrajFrames = (genRes.denoised as Float32Array[]).map((arr) =>
              extractCaFrame(arr, atoms),
            );
            primaryTrajLabels = primaryTrajFrames.map((_, idx) => `step ${idx + 1}/${numSteps}`);
          }
        }
      } finally {
        if (activeAbort === controller) activeAbort = null;
      }

      const primary = items[0];
      const repSelection = {
        selectedKey: primary.designKey ?? primary.id,
        criterion: 'lowest-creation-ordinal',
        totalMembers: items.length,
        selectedIndex: 0,
        scoreValue: null,
      };
      const structureOut: StructureArtifact =
        items.length > 1
          ? {
              ...primary,
              label: `BoltzGen ${variant} (${items.length} designs · L=${primary.length})`,
              representativeSelection: repSelection,
              collection: {
                kind: 'Collection',
                schema: 'denovo.collection/v1',
                id: `bg-col-${variant}-${runHash}`,
                familyKey: `boltzgen-${variant}`,
                collectionKind: 'independent-candidates',
                requestedCount: numDesigns,
                complete: true,
                items,
                keys: items.map((it) => it.designKey ?? it.id),
                representativeSelection: repSelection,
              },
            }
          : primary;

      const trajFrames = (primaryTrajFrames.length > 0 ? primaryTrajFrames : [primary.caCoords.slice()]).map(
        (caCoords, idx) => ({
          key: `frame-${idx}`,
          frameIndex: idx,
          caCoords,
          timeStep: idx + 1,
        }),
      );

      const trajectoryOut: TrajectoryFramesArtifact = {
        kind: 'TrajectoryFrames',
        id: `bg-traj-${runHash}`,
        label: `BoltzGen trajectory (${primaryTrajLabels.length || trajFrames.length} steps)`,
        parentStructureId: primary.id,
        sourceModelId: 'boltzgen',
        frameCount: trajFrames.length,
        ordering: 'ordered-frames',
        collectionKind: 'ordered-frames',
        frames: trajFrames,
      };

      const transferables = collectOwnedTransferables([
        ...items.flatMap((it) => [it.caCoords, it.backbone4Coords, it.observedBackboneMask]),
        ...trajFrames.map((f) => f.caCoords),
      ]);

      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          outputs: { structure: structureOut, trajectory: trajectoryOut },
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
