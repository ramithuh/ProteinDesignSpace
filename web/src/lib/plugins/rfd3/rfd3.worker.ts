/// <reference lib="webworker" />
// Layer 3 Plugin Worker — rfd3 (RFdiffusion3 All-Atom Diffusion · Baker Lab / Foundry · ramithuh/rfd3-webgpu)
// Runs RFdiffusion3 (168M params) natively on WebGPU for all-atom backbone + sidechain + sequence co-design.
// Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { assertSafeStepCount, formatLowStepGeometryWarning } from '$lib/engine/step-policy';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';
import {
  buildAutoFlankContigFromResidues,
  buildFullProteinContigFromResidues,
  computeRfd3MotifMeta,
  drainGpuDeviceQueue,
  isReusableRfd3Engine,
  normalizeRfd3Contig,
  resolveRfd3UpstreamStructure,
  trimRfd3EngineSafely,
} from './lifecycle';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/rfd3-webgpu/resolve/main';

type Precision = 'fp16' | 'int8';
type Rfd3Mode = 'unconditional' | 'motif' | 'binder' | 'partial';

const NEGOTIATED_LIMIT_KEYS = [
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxComputeWorkgroupStorageSize',
  'maxStorageBuffersPerShaderStage',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
] as const;

function normalizePrecision(raw: unknown): Precision {
  if (raw == null || String(raw).trim() === '') return 'fp16';
  const val = String(raw).trim().toLowerCase();
  if (val === 'fp16' || val === 'int8') return val;
  throw new Error(`Unsupported RFdiffusion3 precision "${String(raw)}" (supported: fp16, int8).`);
}

function normalizeMode(raw: unknown): Rfd3Mode {
  const val = String(raw ?? 'unconditional').trim().toLowerCase();
  if (val === 'motif' || val === 'binder' || val === 'partial') return val;
  return 'unconditional';
}

function normalizeSymmetry(raw: unknown): string | null {
  const val = String(raw ?? 'none').trim().toUpperCase();
  if (!val || val === 'NONE') return null;
  if (/^C([2-9]|[1-9][0-9]+)$/.test(val)) return val;
  throw new Error(`Unsupported RFdiffusion3 symmetry "${String(raw)}" (expected none or C2–C6).`);
}

/**
 * Parse flexible user atom/residue selections into RFD3's `selected()` input shape:
 * - `'true'` / `'ALL'` -> `true`
 * - `'false'` / `'NONE'` -> `false`
 * - JSON object (`{"A102":"ND1,CG"}`) -> parsed object
 * - Semicolon/newline map (`A102:ND1,CG,CE1; A104:BKBN`) -> `{ A102: 'ND1,CG,CE1', A104: 'BKBN' }`
 * - Bare `'BKBN'` -> map across all input residues `{ A1: 'BKBN', ... }`
 * - Residue list (`A16-35, A52-71`) -> comma-separated string
 */
function parseAtomOrResidueSelection(
  raw: unknown,
  fallbackResidueKeys?: string[],
): boolean | string | Record<string, string | boolean> | undefined {
  if (typeof raw === 'boolean') return raw;
  const text = String(raw ?? '').trim();
  if (!text) return undefined;
  const upper = text.toUpperCase();
  if (upper === 'TRUE' || upper === 'ALL') return true;
  if (upper === 'FALSE' || upper === 'NONE') return false;
  if (upper === 'BKBN') {
    if (fallbackResidueKeys && fallbackResidueKeys.length > 0) {
      const map: Record<string, string> = {};
      for (const key of fallbackResidueKeys) map[key] = 'BKBN';
      return map;
    }
    return undefined;
  }
  if (text.startsWith('{')) {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, string | boolean>;
    }
    throw new Error('Atom selection JSON must be a residue-to-atoms object.');
  }
  const entries = text
    .split(/[;\n]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const looksLikeAtomMap = entries.some((entry) => {
    const idx = entry.indexOf(':');
    if (idx <= 0) return false;
    const lhs = entry.slice(0, idx).trim();
    const rhs = entry.slice(idx + 1).trim();
    return /^[A-Za-z0-9]+-?\d+(?:--?\d+)?$/.test(lhs) && /[A-Za-z]/.test(rhs);
  });
  if (looksLikeAtomMap) {
    const map: Record<string, string | boolean> = {};
    for (const entry of entries) {
      const idx = entry.indexOf(':');
      if (idx <= 0) {
        map[entry] = true;
      } else {
        const lhs = entry.slice(0, idx).trim();
        const rhs = entry.slice(idx + 1).trim();
        map[lhs] = rhs || true;
      }
    }
    return map;
  }
  return text;
}

function buildSymmetryTiedPositions(copies: number, unitLength: number): string {
  const chains = Array.from({ length: copies }, (_, idx) => String.fromCharCode(65 + idx));
  const groups: string[] = [];
  for (let pos = 1; pos <= unitLength; pos++) {
    groups.push(chains.map((ch) => `${ch}${pos}`).join('='));
  }
  return groups.join('; ');
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let mod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let cachedArchitecture: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let ownedDevice: GPUDevice | null = null;
let loadedPrecision: Precision | null = null;
let activeAbort: AbortController | null = null;
let abortRequested = false;

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
  loadedPrecision = null;
}

async function ensureEngine(ctx: WorkerRunContext, precision: Precision, signal: AbortSignal) {
  if (engine && !engine.resources?.closed && isReusableRfd3Engine(engine, loadedPrecision, precision)) {
    return engine;
  }
  disposeEngine();

  const vendorBase = `${self.location.origin}/vendor/rfd3`;
  if (!mod) {
    ctx.progress('init', 0.02, 'Loading RFdiffusion3 WebGPU modules...');
    mod = await import(/* @vite-ignore */ `${vendorBase}/webgpu/index.js?v=2`);
  }
  if (!cachedArchitecture) {
    ctx.progress('init', 0.04, 'Loading RFdiffusion3 architecture metadata...');
    const archResp = await fetch(`${vendorBase}/fixtures/architecture.json`, { signal });
    if (!archResp.ok) {
      throw new Error(`Failed to fetch RFdiffusion3 architecture.json (HTTP ${archResp.status})`);
    }
    cachedArchitecture = await archResp.json();
  }

  const { adapter } = await requestResilientWebGpuAdapter({ allowSoftwareFallback: true });
  const buildLimits = (a: GPUAdapter) => {
    const out: Record<string, number> = {};
    for (const key of NEGOTIATED_LIMIT_KEYS) {
      const val = (a.limits as unknown as Record<string, unknown>)[key];
      if (typeof val === 'number' && Number.isFinite(val)) out[key] = val;
    }
    return out;
  };
  const wantF16 = adapter.features.has('shader-f16');
  const requiredFeatures: GPUFeatureName[] = wantF16 ? ['shader-f16'] : [];
  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: buildLimits(adapter),
    });
  } catch (err) {
    if (requiredFeatures.length === 0) throw err;
    const { adapter: retryAdapter } = await requestResilientWebGpuAdapter({
      allowSoftwareFallback: true,
    });
    device = await retryAdapter.requestDevice({
      requiredLimits: buildLimits(retryAdapter),
    });
  }
  ownedDevice = device;
  void device.lost.then(() => {
    if (ownedDevice === device) {
      engine = null;
      ownedDevice = null;
      loadedPrecision = null;
    }
  });

  const envBaseUrl = (import.meta.env.VITE_RFD3_WEIGHTS_URL as string | undefined)?.trim();
  const baseURL = envBaseUrl || DEFAULT_BASE_URL;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.05, `Fetching RFdiffusion3 (${precision.toUpperCase()}) weights...`);

    try {
      return await mod.createRfd3Engine({
        architecture: cachedArchitecture,
        baseURL,
        precision,
        device,
        allowSoftwareFallback: true,
        cache: 'none',
        signal,
        read: async (url: string, opts?: { signal?: AbortSignal }): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url, { signal: opts?.signal ?? signal });
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
          return await resp.arrayBuffer();
        },
        onProgress: (prog: { loaded?: number; total?: number; shard?: string }) => {
          const loaded = prog.loaded ?? 0;
          const total = prog.total ?? 0;
          const frac = total > 0 ? Math.min(1, loaded / total) : 0.4;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(1) : '?';
          ctx.progress(
            'weights',
            Math.min(0.45, 0.05 + frac * 0.4),
            total > 0
              ? `RFdiffusion3 (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
              : `Loading RFdiffusion3 (${precision.toUpperCase()})...`,
          );
        },
      });
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      if (
        !retried &&
        anyFromCache &&
        fetchedUrls.length > 0 &&
        /integrity|sha-?256|manifest|mismatch|truncated|exceeds/i.test(errMsg)
      ) {
        await evictCachedUrls(fetchedUrls);
        return await loadOnce(true);
      }
      throw err;
    }
  };

  try {
    engine = await loadOnce(false);
    loadedPrecision = precision;
    return engine;
  } catch (err) {
    disposeEngine();
    throw err;
  }
}

createWorkerHarness({
  async run(ctx) {
    abortRequested = false;
    const p = ctx.params ?? {};

    const mode = normalizeMode(p.mode);
    const precision = normalizePrecision(p.precision);
    const length = Math.max(16, Math.min(200, Math.round(Number(p.length ?? 80))));
    const steps = assertSafeStepCount('rfd3', { ...p, steps: p.steps }, 50);
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));
    const seed = Math.max(0, Math.min(0xffffffff, Math.round(Number(p.seed ?? 42))));

    const rawContig = String(p.contig ?? '').trim();
    const rawUnindex = String(p.unindex ?? '').trim();
    const rawFixedAtoms = p.fixedAtoms;
    const rawUnfixedSequence = p.unfixedSequence;
    const rawTarget = String(p.target ?? '').trim();
    const rawHotspots = p.hotspots;
    const rawLigand = String(p.ligand ?? '').trim();
    const partialT = Math.max(0.5, Math.min(160, Number(p.partialT ?? 5.0)));
    const cyclic = Boolean(p.cyclic);
    const symmetryId = mode === 'unconditional' ? normalizeSymmetry(p.symmetry) : null;
    const isNonLoopy = Boolean(p.isNonLoopy);
    const rawHbondDonor = p.hbondDonor;
    const rawHbondAcceptor = p.hbondAcceptor;
    const cfgScale = Math.max(0, Math.min(4, Number(p.cfgScale ?? 1.0)));
    const gamma0 = Math.max(0, Math.min(1.5, Number(p.gamma0 ?? 0.6)));
    const noiseScale = Math.max(0, Math.min(2.0, Number(p.noiseScale ?? 1.003)));
    const stepScale = Math.max(0.2, Math.min(3.0, Number(p.stepScale ?? 1.5)));

    const upstreamStruct = ctx.inputs?.structure as StructureArtifact | undefined;
    const needsStructure = mode === 'motif' || mode === 'binder' || mode === 'partial';
    if (needsStructure && !upstreamStruct?.pdbText && !upstreamStruct?.cifText) {
      throw new Error(
        `RFdiffusion3 (${mode} mode) requires a connected input Structure (wire a PDB Input node to the "structure" port).`,
      );
    }

    const {
      structureText,
      structureFormat,
      caResidues: inputCaResidues,
      fallbackResKeys,
      autoLigandNames,
    } = resolveRfd3UpstreamStructure(upstreamStruct, needsStructure, rawLigand);

    const effectiveLigand = rawLigand || (autoLigandNames.length > 0 ? autoLigandNames.join(',') : '');
    const parsedFixedAtoms = parseAtomOrResidueSelection(rawFixedAtoms, fallbackResKeys);
    const parsedUnfixedSeq = parseAtomOrResidueSelection(rawUnfixedSequence, fallbackResKeys);
    const parsedHotspots = parseAtomOrResidueSelection(rawHotspots, fallbackResKeys);
    const parsedHbondDonor = parseAtomOrResidueSelection(rawHbondDonor, fallbackResKeys);
    const parsedHbondAcceptor = parseAtomOrResidueSelection(rawHbondAcceptor, fallbackResKeys);

    let rfd3Mode: 'unconditional' | 'motif' | 'binder' | 'conditioned' = 'unconditional';
    let effectiveContig: string | undefined;
    let effectivePartialT: number | undefined;
    let effectiveFixedAtoms = parsedFixedAtoms;

    if (mode === 'motif') {
      rfd3Mode = 'motif';
      if (rawContig) {
        effectiveContig = normalizeRfd3Contig(rawContig);
      } else if (!rawUnindex && !effectiveLigand && inputCaResidues.length > 0) {
        effectiveContig = buildAutoFlankContigFromResidues(inputCaResidues, length);
      }
    } else if (mode === 'binder') {
      rfd3Mode = 'binder';
      if (rawContig) {
        effectiveContig = normalizeRfd3Contig(rawContig);
      }
    } else if (mode === 'partial') {
      rfd3Mode = 'conditioned';
      effectivePartialT = partialT;
      if (rawContig) {
        effectiveContig = normalizeRfd3Contig(rawContig);
      } else if (inputCaResidues.length > 0) {
        effectiveContig = buildFullProteinContigFromResidues(inputCaResidues);
      }
      // In partial diffusion, default fixedAtoms to false so the backbone actually diffuses
      if (effectiveFixedAtoms === undefined) {
        effectiveFixedAtoms = false;
      }
    }

    if (abortRequested || ctx.signal.aborted) {
      throw new Error('Run aborted');
    }

    const abortCtrl = new AbortController();
    activeAbort = abortCtrl;
    const onCtxAbort = () => abortCtrl.abort();
    if (ctx.signal.aborted || abortRequested) {
      abortCtrl.abort();
    } else {
      ctx.signal.addEventListener('abort', onCtxAbort, { once: true });
    }

    const eng = await ensureEngine(ctx, precision, abortCtrl.signal);
    if (abortRequested || ctx.signal.aborted) {
      throw new Error('Run aborted');
    }

    const items: StructureArtifact[] = [];
    const seqCandidates: SequenceCandidate[] = [];
    const sourceModelId = `rfd3/${precision}`;

    try {
      for (let d = 0; d < numDesigns; d++) {
        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }
        if (d > 0 && !eng.resources?.closed) {
          eng.trim();
        }

        const designSeed = (seed + d) >>> 0;
        const designKey = `d${d + 1}`;
        const progressBase = 0.45 + (d / numDesigns) * 0.52;
        const progressSpan = 0.52 / numDesigns;

        ctx.progress(
          'sample',
          progressBase,
          numDesigns > 1
            ? `RFdiffusion3 design ${d + 1}/${numDesigns} (${mode}, ${steps} steps)...`
            : `RFdiffusion3: ${mode === 'unconditional' ? `${length} aa` : mode} all-atom diffusion (${steps} steps)...`,
        );

        const generateOpts: Record<string, unknown> = {
          mode: rfd3Mode,
          length,
          steps: steps + 1,
          cfgScale,
          gamma0,
          noiseScale,
          stepScale,
          retainTrajectory: false,
          seed: designSeed,
          signal: abortCtrl.signal,
          onStep: ({ step, total }: { step: number; total: number }) => {
            if (abortRequested || ctx.signal.aborted) return;
            const frac = total > 0 ? Math.min(1, (step + 1) / total) : 0.5;
            ctx.progress(
              'sample',
              progressBase + frac * progressSpan,
              numDesigns > 1
                ? `RFdiffusion3 ${d + 1}/${numDesigns} · step ${step + 1}/${total}`
                : `RFdiffusion3 · step ${step + 1}/${total}`,
            );
          },
        };

        if (isNonLoopy) generateOpts.isNonLoopy = true;
        if (mode === 'unconditional') {
          if (symmetryId) {
            generateOpts.symmetry = { id: symmetryId };
          } else if (cyclic) {
            generateOpts.cyclic = true;
          }
        } else {
          generateOpts.structure = structureText;
          if (structureFormat) generateOpts.format = structureFormat;
          if (effectiveContig) generateOpts.contig = effectiveContig;
          if (mode === 'motif' && rawUnindex) {
            generateOpts.unindex = rawUnindex;
            generateOpts.cleanupGuideposts = true;
          }
          if (effectiveFixedAtoms !== undefined) generateOpts.fixedAtoms = effectiveFixedAtoms;
          if (parsedUnfixedSeq !== undefined) generateOpts.unfixedSequence = parsedUnfixedSeq;
          if (mode === 'binder') {
            if (rawTarget) generateOpts.target = normalizeRfd3Contig(rawTarget);
            if (parsedHotspots !== undefined) {
              generateOpts.hotspots = parsedHotspots;
              generateOpts.infer_ori_strategy = 'hotspots';
            }
          }
          if (effectiveLigand) generateOpts.ligand = effectiveLigand;
          if (parsedHbondDonor !== undefined) generateOpts.hbondDonor = parsedHbondDonor;
          if (parsedHbondAcceptor !== undefined) generateOpts.hbondAcceptor = parsedHbondAcceptor;
          if (effectivePartialT !== undefined) generateOpts.partialT = effectivePartialT;
        }

        const res = await eng.generate(generateOpts);

        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }

        const pdbText = String(res.pdb ?? '');
        if (!pdbText) throw new Error('RFdiffusion3 did not return PDB coordinates.');
        const mmcifText = typeof res.mmcif === 'string' && res.mmcif.trim() ? res.mmcif : undefined;

        const parsed = parseStructureText(pdbText);
        const geom = computeGeometryStats(
          parsed.caCoords,
          parsed.chainIds,
          parsed.resNums,
          parsed.insCodes,
        );
        const stepGeomWarn = formatLowStepGeometryWarning('rfd3', p, geom);
        if (stepGeomWarn) console.warn(stepGeomWarn);
        const sequence = parsed.sequence;
        const resolvedLen = parsed.nResidues || length;

        const runHash = fnv1a(
          JSON.stringify({
            mode,
            length,
            steps,
            contig: effectiveContig ?? null,
            unindex: rawUnindex || null,
            target: rawTarget || null,
            ligand: effectiveLigand || null,
            symmetry: symmetryId,
            cyclic,
            partialT: effectivePartialT ?? null,
            templateId: upstreamStruct?.id ?? null,
            seed: designSeed,
            precision,
          }),
        );

        // Motif metadata for downstream RMSD / MPNN auto-locking (supports both mmCIF and PDB inputs)
        const motifMeta = computeRfd3MotifMeta({
          mode,
          caResidues: inputCaResidues,
          resolvedLen,
          effectiveContig,
          motifAssignments: Array.isArray(res.motifAssignments) ? res.motifAssignments : undefined,
          features: res.features,
        });

        // Binder locked-chain & symmetry tied-position metadata for downstream inverse folding
        const uniqueChains = [...new Set(parsed.chainIds)];
        let chainMeta: Partial<StructureArtifact> = {};
        if (mode === 'binder' && uniqueChains.length >= 2) {
          const binderChain = uniqueChains[0];
          const targetChains = uniqueChains.slice(1);
          const chainRoles: Record<string, 'binder' | 'target'> = { [binderChain]: 'binder' };
          for (const tc of targetChains) chainRoles[tc] = 'target';
          chainMeta = { lockedChains: targetChains, chainRoles };
        } else if (symmetryId) {
          const copies = Number(symmetryId.slice(1));
          if (Number.isInteger(copies) && copies >= 2) {
            const unitLen = Math.floor(resolvedLen / copies);
            if (unitLen >= 1) {
              chainMeta = { tiedPositions: buildSymmetryTiedPositions(copies, unitLen) };
            }
          }
        }

        const modeTag = symmetryId
          ? ` (${symmetryId}, ${resolvedLen} aa)`
          : cyclic
            ? ` (Cyclic, ${resolvedLen} aa)`
            : mode !== 'unconditional'
              ? ` (${mode}, ${resolvedLen} aa)`
              : ` (${resolvedLen} aa, ${steps} steps)`;
        const baseLabel = `RFdiffusion3${modeTag}`;

        const artifact: StructureArtifact = {
          kind: 'StructureArtifact',
          id: `rfd3-${mode}-${resolvedLen}aa-s${designSeed}-${runHash}`,
          label: numDesigns > 1 ? `${baseLabel} #${d + 1}` : baseLabel,
          sourceModelId,
          ...(needsStructure && upstreamStruct?.id ? { parentStructureId: upstreamStruct.id } : {}),
          pdbText,
          ...(mmcifText ? { cifText: mmcifText } : {}),
          length: resolvedLen,
          chainIds:
            parsed.chainIds.length === resolvedLen
              ? parsed.chainIds
              : Array<string>(resolvedLen).fill('A'),
          chainLengths: parsed.diagnostics.chainLengths,
          sequence,
          caCoords: parsed.caCoords,
          backbone4Coords: parsed.backbone4Coords,
          atomGranularity: 'all-atom',
          bfactorSemantics: 'none',
          atomPresence: parsed.atomPresence,
          diagnostics: parsed.diagnostics,
          geometry: geom,
          provenance: {
            quantization: precision,
            diffusionStepsScheduled: steps,
            diffusionStepsRun: steps,
            ...(stepGeomWarn ? { warnings: [stepGeomWarn] } : {}),
          },
          designKey,
          designIndex: d,
          ...(motifMeta ?? {}),
          ...chainMeta,
        };
        items.push(artifact);

        if (sequence.length > 0) {
          seqCandidates.push({
            id: `rfd3-seq-${d + 1}`,
            key: `${designKey}-native`,
            sequenceKey: `${designKey}-native`,
            designKey,
            designIndex: d,
            sourceStructureId: artifact.id,
            sampleIndex: 0,
            sequence,
            ...(parsed.diagnostics.chainLengths.length > 1
              ? { chainLengths: parsed.diagnostics.chainLengths }
              : {}),
            score: null,
            globalScore: null,
            sourceModelId,
            seed: designSeed,
          });
        }
      }
    } finally {
      ctx.signal.removeEventListener('abort', onCtxAbort);
      if (activeAbort === abortCtrl) activeAbort = null;
      if (!eng.resources?.closed) {
        trimRfd3EngineSafely(eng);
      }
    }

    const primary = items[0];
    const batchHash = fnv1a(`${items.map((it) => it.id).join('|')}:${items.length}`);
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
            id: `rfd3-batch-${items.length}-${batchHash}`,
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `rfd3-col-${batchHash}`,
              familyKey: `rfd3-${batchHash}`,
              collectionKind: 'independent-candidates',
              requestedCount: numDesigns,
              complete: true,
              items,
              keys: items.map((it) => it.designKey ?? it.id),
              representativeSelection: repSelection,
            },
          }
        : primary;

    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: items.length > 1 ? `seqset-rfd3-batch-${items.length}-${batchHash}` : `seqset-${primary.id}`,
      parentStructureId: primary.id,
      sourceStructureId: primary.id,
      sourceStructureIds: items.map((it) => it.id),
      sourceModelId,
      candidates: seqCandidates,
    };

    return { outputs: { structure: structureOut, sequences: seqSet } };
  },

  abort() {
    abortRequested = true;
    activeAbort?.abort();
  },

  async dispose() {
    abortRequested = true;
    activeAbort?.abort();
    activeAbort = null;
    await drainGpuDeviceQueue(ownedDevice, 1500);
    disposeEngine();
  },
});

export {};
