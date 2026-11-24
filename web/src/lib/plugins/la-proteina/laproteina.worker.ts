/// <reference lib="webworker" />
// Layer 3 Plugin Worker — la-proteina (ramithuh/la-proteina-webgpu · Apache-2.0 code / NVIDIA Open Model License weights)
// Runs La-Proteina LD1 latent diffusion + AE1 all-atom & sequence decoder natively on WebGPU.
// Wrapped in Layer 1 `createWorkerHarness()`.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import { requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { createWorkerHarness, fnv1a, type WorkerRunContext } from '$lib/kernel';
import type { SequenceCandidate, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/la-proteina-webgpu/resolve/main';

type Precision = 'int8' | 'fp16';
type SamplerMode = 'sde' | 'ode';
type LaProteinaVariant = 'ld1' | 'ld4' | 'ld5' | 'ld6' | 'ld7';

const VARIANT_LABELS: Record<LaProteinaVariant, string> = {
  ld1: 'LD1 + AE1',
  ld4: 'LD4 + AE3 (Indexed All-Atom)',
  ld5: 'LD5 + AE3 (Indexed Tip-Atom)',
  ld6: 'LD6 + AE3 (Unindexed All-Atom)',
  ld7: 'LD7 + AE3 (Unindexed Tip-Atom)',
};

function normalizePrecision(raw: unknown): Precision {
  if (raw == null || String(raw).trim() === '') return 'int8';
  const val = String(raw).trim().toLowerCase();
  if (val === 'fp16' || val === 'int8') return val;
  throw new Error(`Unsupported La-Proteina precision "${String(raw)}" (supported: int8, fp16).`);
}

function normalizeMode(raw: unknown): SamplerMode {
  return String(raw ?? 'sde').trim().toLowerCase() === 'ode' ? 'ode' : 'sde';
}

function normalizeVariant(raw: unknown): LaProteinaVariant {
  if (raw == null || String(raw).trim() === '') return 'ld1';
  const val = String(raw).trim().toLowerCase();
  if (val === 'ld1' || val === 'unconditional') return 'ld1';
  if (val === 'ld4' || val === 'motif_idx_aa') return 'ld4';
  if (val === 'ld5' || val === 'motif_idx_tip') return 'ld5';
  if (val === 'ld6' || val === 'motif_uidx_aa') return 'ld6';
  if (val === 'ld7' || val === 'motif_uidx_tip') return 'ld7';
  throw new Error(
    `Unsupported La-Proteina variant "${String(raw)}" (supported: ld1, ld4, ld5, ld6, ld7).`,
  );
}

/**
 * Strip ATOM lines carrying non-blank PDB insertion codes (col 27) because
 * La-Proteina's upstream PDB parser rejects insertion codes.
 */
function sanitizeMotifPdb(pdbText: string): string {
  const lines = pdbText.split(/\r?\n/);
  let stripped = false;
  const out: string[] = [];
  for (const line of lines) {
    if (line.startsWith('ATOM  ') && (line[26] ?? ' ') !== ' ') {
      stripped = true;
      continue;
    }
    out.push(line);
  }
  return stripped ? out.join('\n') : pdbText;
}

interface ParsedPdbCaResidue {
  chainId: string;
  resId: number;
  ca: [number, number, number];
}

function extractPdbCaResidues(pdbText: string): ParsedPdbCaResidue[] {
  const out: ParsedPdbCaResidue[] = [];
  const seen = new Set<string>();
  let sawModel = false;
  let inModel = false;
  for (const line of pdbText.split(/\r?\n/)) {
    if (line.startsWith('MODEL')) {
      if (sawModel) break;
      sawModel = true;
      inModel = true;
      continue;
    }
    if (line.startsWith('ENDMDL') && inModel) break;
    if (!line.startsWith('ATOM  ')) continue;
    const altloc = line[16] ?? ' ';
    if (altloc !== ' ' && altloc !== 'A' && altloc !== '1') continue;
    if ((line[26] ?? ' ') !== ' ') continue;
    if (line.slice(12, 16).trim() !== 'CA') continue;
    const chainId = (line[21] ?? 'A').trim() || 'A';
    const resId = Number.parseInt(line.slice(22, 26).trim(), 10);
    const x = Number.parseFloat(line.slice(30, 38));
    const y = Number.parseFloat(line.slice(38, 46));
    const z = Number.parseFloat(line.slice(46, 54));
    if (!Number.isFinite(resId) || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    const key = `${chainId}:${resId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ chainId, resId, ca: [x, y, z] });
  }
  return out;
}

/**
 * Auto-build a clean flanking contig from the first contiguous chain span when the user
 * leaves `contig` blank for LD4–LD7 motif scaffolding.
 */
function buildAutoFlankContig(pdbText: string, targetLength: number): string {
  const residues = extractPdbCaResidues(pdbText);
  if (residues.length === 0) {
    throw new Error('Input structure has no valid ATOM Cα residues for La-Proteina motif scaffolding.');
  }
  const primaryChain = residues[0].chainId;
  const chainRes = residues.filter((r) => r.chainId === primaryChain);
  // Find longest contiguous (by resId + 1) run on the primary chain
  let bestStart = 0;
  let bestLen = 1;
  let curStart = 0;
  let curLen = 1;
  for (let i = 1; i < chainRes.length; i++) {
    if (chainRes[i].resId === chainRes[i - 1].resId + 1) {
      curLen++;
      if (curLen > bestLen) {
        bestStart = curStart;
        bestLen = curLen;
      }
    } else {
      curStart = i;
      curLen = 1;
    }
  }
  let spanStartIdx = bestStart;
  let spanLen = bestLen;
  // If the entire input structure is larger than targetLength, take a central motif window
  if (spanLen >= targetLength) {
    const maxMotif = Math.max(8, Math.min(24, Math.floor(targetLength * 0.4)));
    spanStartIdx = bestStart + Math.max(0, Math.floor((spanLen - maxMotif) / 2));
    spanLen = Math.min(spanLen, maxMotif);
  }
  const startRes = chainRes[spanStartIdx].resId;
  const endRes = chainRes[spanStartIdx + spanLen - 1].resId;
  const remaining = Math.max(0, targetLength - spanLen);
  const flankN = Math.floor(remaining / 2);
  const flankC = remaining - flankN;
  const parts: string[] = [];
  if (flankN > 0) parts.push(String(flankN));
  parts.push(startRes === endRes ? `${primaryChain}${startRes}` : `${primaryChain}${startRes}-${endRes}`);
  if (flankC > 0) parts.push(String(flankC));
  return parts.join('/');
}

/**
 * Extract native template Cα coordinates matching the PDB chain/range segments in `contig`,
 * in exact `extractMotif()` selection order, for downstream motif RMSD verification.
 */
function extractNativeMotifCaCoords(pdbText: string, contig: string): number[][] {
  const residues = extractPdbCaResidues(pdbText);
  const selected: number[][] = [];
  const seen = new Set<string>();
  for (const part of contig.split('/').map((s) => s.trim()).filter(Boolean)) {
    if (!/^[A-Za-z]/.test(part)) continue;
    const chainId = part[0];
    const body = part.slice(1);
    const [start, end] = body.includes('-') ? body.split('-').map(Number) : [Number(body), Number(body)];
    for (const r of residues) {
      if (r.chainId !== chainId || r.resId < start || r.resId > end) continue;
      const key = `${r.chainId}:${r.resId}`;
      if (seen.has(key)) continue;
      seen.add(key);
      selected.push(r.ca);
    }
  }
  return selected;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let mod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let ownedDevice: GPUDevice | null = null;
let loadedPrecision: Precision | null = null;
let loadedVariant: LaProteinaVariant | null = null;
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
  loadedVariant = null;
}

async function ensureEngine(
  ctx: WorkerRunContext,
  precision: Precision,
  variant: LaProteinaVariant,
) {
  if (engine && loadedPrecision === precision && loadedVariant === variant) return engine;
  disposeEngine();

  if (!mod) {
    ctx.progress('init', 0.02, 'Loading La-Proteina WebGPU modules...');
    const base = `${self.location.origin}/vendor/la-proteina`;
    mod = await import(/* @vite-ignore */ `${base}/webgpu/index.js`);
  }

  const { adapter } = await requestResilientWebGpuAdapter({ allowSoftwareFallback: true });
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  ownedDevice = device;
  void device.lost.then(() => {
    if (ownedDevice === device) {
      engine = null;
      ownedDevice = null;
      loadedPrecision = null;
      loadedVariant = null;
    }
  });

  const envBaseUrl = (import.meta.env.VITE_LAPROTEINA_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;
  const variantLabel = VARIANT_LABELS[variant];

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.05, `Fetching La-Proteina ${variantLabel} (${precision.toUpperCase()}) weights...`);

    try {
      return await mod.createLaProteinaEngine({
        baseUrl,
        precision,
        variant,
        device,
        adapter,
        allowSoftwareFallback: true,
        cache: 'none',
        read: async (url: string): Promise<ArrayBuffer> => {
          const resp = await cachedFetch(url);
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
              ? `La-Proteina ${variant.toUpperCase()} (${precision.toUpperCase()}): ${mb} / ${totMb} MB`
              : `Loading La-Proteina ${variantLabel} (${precision.toUpperCase()})...`,
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

  try {
    engine = await loadOnce(false);
    loadedPrecision = precision;
    loadedVariant = variant;
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
    const variant = normalizeVariant(p.variant);
    const isMotif = variant !== 'ld1';
    const rawContig = String(p.contig ?? '').trim();
    const length = Math.max(20, Math.min(300, Math.round(Number(p.length ?? 80))));
    const steps = Math.max(10, Math.min(400, Math.round(Number(p.steps ?? 400))));
    const mode = normalizeMode(p.mode);
    const seed = Math.round(Number(p.seed ?? 42));
    const precision = normalizePrecision(p.precision);
    const numDesigns = Math.max(1, Math.min(8, Math.round(Number(p.numDesigns ?? 1))));

    const upstreamStruct = ctx.inputs?.structure as StructureArtifact | undefined;
    let motifPdb: string | null = null;
    let effectiveContig: string | null = null;
    let nativeMotifCoords: number[][] | null = null;

    if (isMotif) {
      if (!upstreamStruct || !upstreamStruct.pdbText) {
        throw new Error(
          `La-Proteina ${variant.toUpperCase()} motif scaffolding requires an input Structure (connect a PDB Input node to the "structure" port).`,
        );
      }
      motifPdb = sanitizeMotifPdb(upstreamStruct.pdbText);
      effectiveContig = rawContig || buildAutoFlankContig(motifPdb, length);
      nativeMotifCoords = extractNativeMotifCaCoords(motifPdb, effectiveContig);
    }

    if (abortRequested || ctx.signal.aborted) {
      throw new Error('Run aborted');
    }

    const eng = await ensureEngine(ctx, precision, variant);
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

    const items: StructureArtifact[] = [];
    const seqCandidates: SequenceCandidate[] = [];
    const ldTag = variant.toUpperCase();
    const aeTag = isMotif ? 'AE3' : 'AE1';
    const sourceModelId = isMotif ? `la-proteina/${variant}/${precision}` : `la-proteina/${precision}`;

    try {
      for (let d = 0; d < numDesigns; d++) {
        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }

        const designSeed = seed + d;
        const designKey = `d${d + 1}`;
        const progressBase = 0.45 + (d / numDesigns) * 0.52;
        const progressSpan = 0.52 / numDesigns;

        ctx.progress(
          'sample',
          progressBase,
          numDesigns > 1
            ? `La-Proteina design ${d + 1}/${numDesigns}: ${ldTag} sampling (${steps} steps${isMotif && effectiveContig ? `, ${effectiveContig}` : `, ${length} aa`})...`
            : `La-Proteina: ${ldTag} latent diffusion (${steps} steps${isMotif && effectiveContig ? `, ${effectiveContig}` : `, ${length} aa`}) + ${aeTag} decode...`,
        );

        const res = await eng.sample({
          variant,
          length,
          steps,
          seed: designSeed,
          mode,
          ...(isMotif && motifPdb && effectiveContig
            ? { motif: { pdb: motifPdb, contig: effectiveContig, centerMotif: true } }
            : {}),
          signal: abortCtrl.signal,
          onProgress: (info: {
            step?: number;
            steps?: number;
            layer?: number;
            total?: number;
            phase?: string;
          }) => {
            if (abortRequested || ctx.signal.aborted) return;
            const isAe =
              info.phase === 'ae1' || info.phase === 'ae3_decoder' || info.phase === 'decode';
            const isSampling = info.phase === 'sampling' || (!info.phase && info.step !== undefined);
            // Ignore inner LD layer progress ticks so the overall sampling progress bar never jumps backward
            if (!isAe && !isSampling) return;
            const step = isAe ? (info.layer ?? info.step ?? 0) : (info.step ?? 0);
            const totalSteps = isAe
              ? (info.total ?? info.steps ?? 12)
              : (info.steps ?? info.total ?? steps);
            const frac = totalSteps > 0 ? Math.min(1, step / totalSteps) : 0.5;
            const scaledFrac = isAe ? 0.9 + frac * 0.1 : frac * 0.9;
            ctx.progress(
              'sample',
              progressBase + scaledFrac * progressSpan,
              isAe
                ? `La-Proteina ${aeTag} decode · layer ${step}/${totalSteps}`
                : numDesigns > 1
                  ? `La-Proteina ${ldTag} ${d + 1}/${numDesigns} · step ${step}/${totalSteps}`
                  : `La-Proteina ${ldTag} · step ${step}/${totalSteps} (${mode.toUpperCase()})`,
            );
          },
        });

        if (abortRequested || ctx.signal.aborted) {
          throw new Error('Run aborted');
        }

        const pdbText = String(res.pdb ?? '');
        if (!pdbText) throw new Error('La-Proteina did not return PDB coordinates.');
        const parsed = parseStructureText(pdbText);
        const geom = computeGeometryStats(parsed.caCoords);
        const sequence = String(res.sequence ?? parsed.sequence ?? '');

        const runHash = fnv1a(
          JSON.stringify({
            variant,
            contig: effectiveContig,
            templateId: isMotif ? (upstreamStruct?.id ?? null) : null,
            length,
            steps,
            mode,
            seed: designSeed,
            precision,
          }),
        );
        const resolvedLen = parsed.nResidues || length;
        const matchedIndicesRaw: number[] | undefined = Array.isArray(res.motifMetrics?.matchedIndices)
          ? res.motifMetrics.matchedIndices
          : Array.isArray(res.motifIndices)
            ? res.motifIndices
            : undefined;
        const motifMeta =
          isMotif &&
          matchedIndicesRaw &&
          matchedIndicesRaw.length > 0 &&
          nativeMotifCoords &&
          nativeMotifCoords.length === matchedIndicesRaw.length
            ? {
                fixedResidueIndices: matchedIndicesRaw.map((idx) => idx + 1),
                motifIndices: matchedIndicesRaw,
                nativeMotifCoords,
              }
            : null;
        const motifRmsdVal =
          typeof res.motifMetrics?.motifRmsdAlignedAngstrom === 'number' &&
          Number.isFinite(res.motifMetrics.motifRmsdAlignedAngstrom)
            ? res.motifMetrics.motifRmsdAlignedAngstrom
            : null;

        const baseLabel = isMotif
          ? `La-Proteina ${ldTag}${motifRmsdVal != null ? ` (${resolvedLen} aa · motif ${motifRmsdVal.toFixed(2)} Å)` : ` (${resolvedLen} aa)`}`
          : `La-Proteina (${resolvedLen} aa, ${steps} steps)`;

        const artifact: StructureArtifact = {
          kind: 'StructureArtifact',
          id: `laproteina-${variant}-${resolvedLen}aa-s${designSeed}-${runHash}`,
          label: numDesigns > 1 ? `${baseLabel} #${d + 1}` : baseLabel,
          sourceModelId,
          ...(isMotif && upstreamStruct?.id ? { parentStructureId: upstreamStruct.id } : {}),
          pdbText,
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
          designKey,
          designIndex: d,
          ...(motifMeta ?? {}),
        };
        items.push(artifact);

        if (sequence.length > 0) {
          seqCandidates.push({
            id: `laproteina-seq-${d + 1}`,
            key: `${designKey}-native`,
            sequenceKey: `${designKey}-native`,
            designKey,
            designIndex: d,
            sourceStructureId: artifact.id,
            sampleIndex: 0,
            sequence,
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
            id: `laproteina-batch-${items.length}-${batchHash}`,
            representativeSelection: repSelection,
            collection: {
              kind: 'Collection',
              schema: 'denovo.collection/v1',
              id: `laproteina-col-${batchHash}`,
              familyKey: `la-proteina-${batchHash}`,
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
      id: items.length > 1 ? `seqset-laproteina-batch-${items.length}-${batchHash}` : `seqset-${primary.id}`,
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
    disposeEngine();
  },
});

export {};
