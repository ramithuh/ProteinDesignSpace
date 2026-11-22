/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/thermompnn-webgpu (MIT)
// Predicts full site-saturation ΔΔG stability landscapes (L × 20 kcal/mol)
// and paints per-residue stability aggregates onto the 3D structure B-factors.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { isWebGpuDeviceLostError } from '$lib/engine/webgpu-device';
import { paintStructureBfactors } from '$lib/protein-ops/bfactor-paint';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { MutationMatrixArtifact, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/thermompnn-webgpu/resolve/main/default';

type Precision = 'fp16' | 'int8' | 'fp32';
type BfactorAggregate = 'best-stabilizing' | 'mean-ddg' | 'worst-destabilizing';

interface ThermoResidue {
  index: number;
  chain: string;
  number: number;
  ins: string;
  aa: string;
  missing: boolean;
  structuralValid: boolean;
}

interface ThermoPredictResult {
  ddg: Float32Array;
  shape: [number, number];
  alphabet: string;
  sequence: string;
  residues: ThermoResidue[];
  mask: Float32Array;
  valid: Uint8Array;
  timingMs: number;
}

interface ThermoEngine {
  predict(
    text: string,
    options?: { chains?: string[]; maxLength?: number },
  ): Promise<ThermoPredictResult>;
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

function normalizeAggregate(raw: unknown): BfactorAggregate {
  const val = String(raw ?? 'best-stabilizing').trim();
  if (val === 'mean-ddg' || val === 'worst-destabilizing' || val === 'best-stabilizing') {
    return val;
  }
  return 'best-stabilizing';
}

function parseChainFilter(raw: unknown, availableChains?: readonly string[]): string[] | undefined {
  const text = String(raw ?? '').trim();
  if (!text) return undefined;
  const tokens = text
    .split(/[,\s]+/)
    .map((t) => t.trim())
    .filter(Boolean);
  if (tokens.length === 0) return undefined;

  const availSet = availableChains ? new Set(availableChains) : null;
  const upperMap = new Map<string, string>();
  if (availableChains) {
    for (const ch of availableChains) {
      const up = ch.toUpperCase();
      if (!upperMap.has(up)) upperMap.set(up, ch);
    }
  }

  const out: string[] = [];
  const seen = new Set<string>();
  for (const tok of tokens) {
    const resolved =
      availSet && availSet.has(tok)
        ? tok
        : upperMap.get(tok.toUpperCase()) ?? tok;
    if (!seen.has(resolved)) {
      seen.add(resolved);
      out.push(resolved);
    }
  }
  return out.length > 0 ? out : undefined;
}

function formatResidueId(chain: string, resNum: number, ins: string): string {
  if (!chain) return `${resNum}${ins}`;
  return chain.length > 1 ? `${chain}:${resNum}${ins}` : `${chain}${resNum}${ins}`;
}

function aggregateRow(
  ddg: Float32Array,
  rowOffset: number,
  alphabet: string,
  wt: string,
  mode: BfactorAggregate,
): number {
  let minVal = Infinity;
  let maxVal = -Infinity;
  let sum = 0;
  let count = 0;
  for (let a = 0; a < alphabet.length; a++) {
    if (alphabet[a] === wt) continue;
    const v = ddg[rowOffset + a];
    if (!Number.isFinite(v)) continue;
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
    sum += v;
    count++;
  }
  if (count === 0) return Number.NaN;
  if (mode === 'mean-ddg') return sum / count;
  if (mode === 'worst-destabilizing') return maxVal;
  return minVal;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let thermoMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let thermoCpuMod: any = null;
let engine: ThermoEngine | null = null;
let loadedPrecision: Precision | null = null;
let loadedBackend: 'webgpu' | 'cpu' | null = null;

async function ensureEngine(
  id: number,
  precision: Precision,
  forceCpu = false,
): Promise<{ engine: ThermoEngine; backend: 'webgpu' | 'cpu' }> {
  if (engine && loadedPrecision === precision && (!forceCpu || loadedBackend === 'cpu')) {
    return { engine, backend: loadedBackend ?? 'webgpu' };
  }

  if (engine) {
    try {
      engine.destroy();
    } catch {
      /* ignore */
    }
    engine = null;
    loadedPrecision = null;
    loadedBackend = null;
  }

  const envBaseUrl = (import.meta.env.VITE_THERMOMPNN_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const loadWithBackend = async (
    backend: 'webgpu' | 'cpu',
    retried: boolean,
  ): Promise<ThermoEngine> => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    if (backend === 'webgpu') {
      if (!thermoMod) {
        self.postMessage({
          id,
          type: 'progress',
          phase: 'init',
          progress: 0.02,
          detail: 'Loading ThermoMPNN WebGPU modules...',
        });
        const moduleUrl = new URL('/vendor/thermompnn/webgpu/index.js?v=3', self.location.origin).href;
        thermoMod = await import(/* @vite-ignore */ moduleUrl);
      }
    } else if (!thermoCpuMod) {
      self.postMessage({
        id,
        type: 'progress',
        phase: 'init',
        progress: 0.02,
        detail: 'WebGPU unavailable — loading ThermoMPNN CPU fallback...',
      });
      const cpuUrl = new URL('/vendor/thermompnn/cpu/index.js?v=1', self.location.origin).href;
      thermoCpuMod = await import(/* @vite-ignore */ cpuUrl);
    }

    const backendTag = backend === 'cpu' ? ' · CPU fallback' : '';
    self.postMessage({
      id,
      type: 'progress',
      phase: 'weights',
      progress: 0.06,
      detail: `Fetching ThermoMPNN (${precision.toUpperCase()}${backendTag}) weights...`,
    });

    try {
      const factory =
        backend === 'webgpu'
          ? thermoMod.createThermoMPNNEngine
          : thermoCpuMod.createThermoMPNNCPU;
      return (await factory({
        baseUrl,
        precision,
        cache: 'none',
        allowSoftwareFallback: true,
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
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
            detail: `ThermoMPNN weights (${precision.toUpperCase()}${backendTag}): ${mb} / ${totMb} MB`,
          });
        },
      })) as ThermoEngine;
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        /integrity|sha-?256|manifest|mismatch/i.test(errMsg)
      ) {
        await evictCachedUrls(fetchedUrls);
        return await loadWithBackend(backend, true);
      }
      throw err;
    }
  };

  if (!forceCpu && typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu) {
    try {
      engine = await loadWithBackend('webgpu', false);
      loadedPrecision = precision;
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

  engine = await loadWithBackend('cpu', false);
  loadedPrecision = precision;
  loadedBackend = 'cpu';
  return { engine, backend: 'cpu' };
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
      loadedBackend = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const structure = msg.inputs?.structure as StructureArtifact | undefined;
      const structureText = structure?.cifText ?? structure?.pdbText;
      if (!structure || !structureText) {
        throw new Error('ThermoMPNN requires an upstream StructureArtifact with PDB or mmCIF text.');
      }

      const p = msg.params ?? {};
      const precision = normalizePrecision(p.precision);
      const chainSpec = String(p.chain ?? '').trim();
      const bfactorAggregate = normalizeAggregate(p.bfactorAggregate);

      const availableChains = structure.sourceResidues?.length
        ? [...new Set(structure.sourceResidues.map((r) => (structure.cifText ? r.authChainId : r.projectedChainId)))]
        : structure.chainIds
          ? [...new Set(structure.chainIds)]
          : undefined;
      const chains = parseChainFilter(chainSpec, availableChains);

      let { engine: activeEngine, backend } = await ensureEngine(msg.id, precision);

      const runPredict = async (eng: ThermoEngine, bk: 'webgpu' | 'cpu') => {
        self.postMessage({
          id: msg.id,
          type: 'progress',
          phase: 'score',
          progress: 0.94,
          detail: `Running ThermoMPNN ΔΔG scan (${precision.toUpperCase()}${bk === 'cpu' ? ', CPU fallback' : ''}${chains ? `, chain ${chains.join(',')}` : ''})...`,
        });
        return await eng.predict(structureText, {
          ...(chains ? { chains } : {}),
        });
      };

      let res: ThermoPredictResult;
      try {
        res = await runPredict(activeEngine, backend);
      } catch (predictErr) {
        if (backend === 'webgpu' && isWebGpuDeviceLostError(predictErr)) {
          ({ engine: activeEngine, backend } = await ensureEngine(msg.id, precision, true));
          res = await runPredict(activeEngine, backend);
        } else {
          throw predictErr;
        }
      }

      const alphabet = (res.alphabet || 'ACDEFGHIKLMNPQRSTVWY').slice(0, 20).toUpperCase();
      const K = alphabet.length;

      // Filter out gap-padded residues so MutationMatrix aligns cleanly to observed polymer residues
      const observedIndices: number[] = [];
      for (let i = 0; i < res.residues.length; i++) {
        if (!res.residues[i].missing) observedIndices.push(i);
      }
      const L = observedIndices.length;
      const values = new Float32Array(L * K);
      const meanPerPosition = new Float32Array(L);
      const residueIds: string[] = new Array(L);
      const wtChars: string[] = new Array(L);
      const aggByKey = new Map<string, number>();

      for (let outIdx = 0; outIdx < L; outIdx++) {
        const srcIdx = observedIndices[outIdx];
        const r = res.residues[srcIdx];
        const wt = (r.aa || 'X').toUpperCase();
        wtChars[outIdx] = wt;
        residueIds[outIdx] = formatResidueId(r.chain, r.number, r.ins);

        const isValid = Boolean(res.valid?.[srcIdx]) && wt !== 'X';
        const srcOff = srcIdx * K;
        const dstOff = outIdx * K;
        if (isValid) {
          values.set(res.ddg.subarray(srcOff, srcOff + K), dstOff);
          meanPerPosition[outIdx] = aggregateRow(res.ddg, srcOff, alphabet, wt, 'mean-ddg');
          const paintAgg = aggregateRow(res.ddg, srcOff, alphabet, wt, bfactorAggregate);
          aggByKey.set(`${r.chain}:${r.number}:${r.ins}`, paintAgg);
        } else {
          values.fill(Number.NaN, dstOff, dstOff + K);
          meanPerPosition[outIdx] = Number.NaN;
        }
      }

      const wtSequence = wtChars.join('');
      const runHash = fnv1a(
        JSON.stringify({
          structureId: structure.id,
          precision,
          chains: chains ?? null,
          bfactorAggregate,
        }),
      );
      const chainSuffix = chains ? ` · chain ${chains.join(',')}` : '';

      const mutations: MutationMatrixArtifact = {
        kind: 'MutationMatrix',
        id: `thermompnn-mut-${structure.id}-${precision}-${runHash}`,
        label: `ThermoMPNN ΔΔG (${precision.toUpperCase()}${chainSuffix})`,
        sourceModelId: 'thermompnn',
        parentStructureId: structure.id,
        metric: 'ddg-kcal-mol',
        unit: 'kcal/mol',
        direction: 'lower-is-better',
        residues: wtSequence,
        residueIds,
        alphabet,
        values,
        meanPerPosition,
        provenance: {
          quantization: precision,
        },
      };

      // Map per-residue aggregates onto full input structure residues (handles gaps & chain filters safely)
      const sourceResidues =
        structure.sourceResidues && structure.sourceResidues.length > 0
          ? structure.sourceResidues
          : parseStructureText(structureText).sourceResidues;

      const pdbBfactors = new Float32Array(sourceResidues.length).fill(Number.NaN);
      const cifBfactors = new Float32Array(sourceResidues.length).fill(Number.NaN);
      for (let i = 0; i < sourceResidues.length; i++) {
        const sr = sourceResidues[i];
        const ins = sr.insCode ?? '';
        const authKey = `${sr.authChainId}:${sr.resNum}:${ins}`;
        const projKey = `${sr.projectedChainId}:${sr.resNum}:${ins}`;
        const val = aggByKey.has(authKey)
          ? aggByKey.get(authKey)!
          : aggByKey.has(projKey)
            ? aggByKey.get(projKey)!
            : Number.NaN;
        pdbBfactors[i] = val;
        cifBfactors[i] = val;
      }

      const paintedPdb = structure.pdbText
        ? paintStructureBfactors(structure.pdbText, pdbBfactors)
        : structure.pdbText;
      const paintedCif = structure.cifText
        ? paintStructureBfactors(structure.cifText, cifBfactors)
        : undefined;

      const caCoordsCopy = new Float32Array(structure.caCoords);
      const backbone4Copy = new Float32Array(structure.backbone4Coords);
      const obsMaskCopy = structure.observedBackboneMask
        ? new Uint8Array(structure.observedBackboneMask)
        : undefined;

      const paintedStructure: StructureArtifact = {
        ...structure,
        id: `thermompnn-struct-${structure.id}-${precision}-${runHash}`,
        label: `${structure.label} (ThermoMPNN ΔΔG · ${bfactorAggregate})`,
        parentStructureId: structure.id,
        caCoords: caCoordsCopy,
        backbone4Coords: backbone4Copy,
        ...(obsMaskCopy ? { observedBackboneMask: obsMaskCopy } : {}),
        pdbText: paintedPdb,
        ...(paintedCif !== undefined ? { cifText: paintedCif } : {}),
        bfactorSemantics: 'ddg-stability',
        provenance: {
          ...structure.provenance,
          quantization: precision,
        },
      };

      const transferables = collectOwnedTransferables([
        mutations.values,
        mutations.meanPerPosition,
        paintedStructure.caCoords,
        paintedStructure.backbone4Coords,
        paintedStructure.observedBackboneMask,
      ]);

      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          outputs: { mutations, structure: paintedStructure },
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
