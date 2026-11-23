/// <reference lib="webworker" />
// Layer 3 Plugin Worker — gnina (ramithuh/gnina-webgpu, Apache-2.0 / GPLv2 · GNINA 1.3 default 3-model 3D CNN ensemble)
// Scores protein–ligand poses (CNNscore, CNNaffinity pK, CNN_VS, Vina kcal/mol),
// runs pocket residue CNN ablation scans (ΔCNN_VS -> ResidueTrack + B-factor painting),
// performs flexible-ligand docking, and performs rigid-body pose minimization under Vina - cnnWeight * log(CNNscore).
// Wrapped in Layer 1 `createWorkerHarness()` for standardized lifecycle, progress, and disposal.

import { evictCachedUrls, makeCachedFetch } from '$lib/engine/opfs-cache';
import {
  collectOwnedTransferables,
  createWorkerHarness,
  fnv1a,
  type WorkerRunContext,
} from '$lib/kernel';
import { paintStructureBfactors } from '$lib/protein-ops/bfactor-paint';
import {
  appendExternalLigandToReceptorPdb,
  applyMinimizedHoloLigandCoords,
  buildMultiPoseDockedPdb,
  computePoseHeavyRmsd,
  detectExternalLigandFormat,
  extractHoloReceptorAndLigands,
  findPocketResidues,
  parsePocketResidueTokensStrict,
} from '$lib/protein-ops/gnina-prep';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type {
  DockingScoreSummary,
  GninaPoseScoreRow,
  ResidueTrackArtifact,
  StructureArtifact,
} from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

const DEFAULT_BASE_URL = 'https://huggingface.co/ramithuh/gnina-webgpu/resolve/main/default';

type Precision = 'int8' | 'fp16' | 'fp32';
type TaskMode = 'score' | 'pocket-scan' | 'minimize' | 'dock';

function normalizePrecision(raw: unknown): Precision {
  const val = String(raw ?? 'fp16').trim().toLowerCase();
  if (val === 'int8' || val === 'fp32') return val;
  return 'fp16';
}

function normalizeTaskMode(raw: unknown): TaskMode {
  const val = String(raw ?? 'score').trim().toLowerCase();
  if (val === 'pocket-scan' || val === 'minimize' || val === 'dock') return val;
  return 'score';
}

async function fetchRcsbCcdIdealSdf(ccdRaw: string, signal?: AbortSignal): Promise<string> {
  const ccd = ccdRaw.trim().toUpperCase();
  if (!/^[A-Z0-9]{1,5}$/.test(ccd)) {
    throw new Error(`Invalid RCSB CCD ligand code "${ccdRaw}" (expected 1–5 alphanumeric characters, e.g. BTN, ATP, I4B).`);
  }
  const url = `https://files.rcsb.org/ligands/download/${encodeURIComponent(ccd)}_ideal.sdf`;
  const resp = await fetch(url, { signal });
  if (!resp.ok) {
    throw new Error(`Could not fetch ideal 3D SDF for CCD "${ccd}" from RCSB (HTTP ${resp.status}).`);
  }
  const text = await resp.text();
  if (!text.includes('V2000') && !text.includes('V3000')) {
    throw new Error(`RCSB response for CCD "${ccd}" was not a valid SDF conformer.`);
  }
  return text;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let gninaMod: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any = null;
let ownedDevice: GPUDevice | null = null;
let loadedPrecision: Precision | null = null;

async function acquireWebGpuDevice(): Promise<GPUDevice> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU is not available in this browser worker.');
  }
  let adapter: GPUAdapter | null = null;
  for (let attempt = 0; attempt < 3 && !adapter; attempt++) {
    if (attempt > 0) await new Promise((r) => setTimeout(r, attempt * 250));
    adapter =
      (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
      (await navigator.gpu.requestAdapter());
  }
  if (!adapter) {
    throw new Error('No WebGPU adapter found for GNINA 1.3.');
  }
  return await adapter.requestDevice();
}

async function ensureEngine(ctx: WorkerRunContext, precision: Precision) {
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
  if (ownedDevice) {
    try {
      ownedDevice.destroy();
    } catch {
      /* ignore */
    }
    ownedDevice = null;
  }

  if (!gninaMod) {
    ctx.progress('init', 0.02, 'Loading GNINA 1.3 WebGPU modules...');
    const moduleUrl = new URL('/vendor/gnina/webgpu/index.js', self.location.origin).href;
    gninaMod = await import(/* @vite-ignore */ moduleUrl);
  }

  const device = await acquireWebGpuDevice();
  ownedDevice = device;

  const envBaseUrl = (import.meta.env.VITE_GNINA_WEIGHTS_URL as string | undefined)?.trim();
  const baseUrl = envBaseUrl || DEFAULT_BASE_URL;

  const loadOnce = async (retried: boolean) => {
    const fetchedUrls: string[] = [];
    let anyFromCache = false;
    const cachedFetch = makeCachedFetch((u, fromCache) => {
      fetchedUrls.push(u);
      if (fromCache) anyFromCache = true;
    });

    ctx.progress('weights', 0.06, `Fetching GNINA 1.3 (${precision.toUpperCase()}) 3-model ensemble...`);

    try {
      return await gninaMod.createGninaEngine({
        baseUrl,
        precision,
        device,
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
          const mb = (loaded / 1024 / 1024).toFixed(2);
          const totMb = total > 0 ? (total / 1024 / 1024).toFixed(2) : '?';
          ctx.progress(
            'weights',
            Math.min(0.45, 0.08 + frac * 0.37),
            `GNINA 1.3 weights (${precision.toUpperCase()}): ${mb} / ${totMb} MB`,
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
    return engine;
  } catch (err) {
    if (ownedDevice) {
      try {
        ownedDevice.destroy();
      } catch {
        /* ignore */
      }
      ownedDevice = null;
    }
    throw err;
  }
}

function formatVinaTerms(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  contributions: any,
  rotors: number,
): GninaPoseScoreRow['vinaTerms'] | undefined {
  const t = contributions?.terms;
  if (!t || typeof t !== 'object') return undefined;
  return {
    gauss1: Number(t.gauss1 ?? 0),
    gauss2: Number(t.gauss2 ?? 0),
    repulsion: Number(t.repulsion ?? 0),
    hydrophobic: Number(t.hydrophobic ?? 0),
    hydrogen: Number(t.hbond ?? 0),
    rotors,
  };
}

function formatPerModel(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelsObj: any,
): Record<string, { cnnScore: number; cnnAffinity: number }> {
  const out: Record<string, { cnnScore: number; cnnAffinity: number }> = {};
  if (!modelsObj || typeof modelsObj !== 'object') return out;
  for (const [k, v] of Object.entries(modelsObj as Record<string, { cnnScore?: number; cnnAffinity?: number }>)) {
    out[k] = {
      cnnScore: Number(v?.cnnScore ?? 0),
      cnnAffinity: Number(v?.cnnAffinity ?? 0),
    };
  }
  return out;
}

createWorkerHarness({
  async run(ctx) {
    const p = ctx.params ?? {};
    const precision = normalizePrecision(p.precision);
    const taskMode = normalizeTaskMode(p.taskMode);
    const ligandCode = String(p.ligandCode ?? '').trim();
    const ligandSdf = String(p.ligandSdf ?? '').trim();
    const dockLigandCcd = String(p.dockLigandCcd ?? '').trim().toUpperCase();
    const pocketResiduesSpec = String(p.pocketResidues ?? '').trim();
    const autoboxAdd = Math.max(2.0, Math.min(12.0, Number(p.autoboxAdd ?? 4.0)));
    const exhaustiveness = Math.max(1, Math.min(16, Math.round(Number(p.exhaustiveness ?? 8))));
    const numModes = Math.max(1, Math.min(9, Math.round(Number(p.numModes ?? 5))));
    const sortByRaw = String(p.sortBy ?? 'cnnScore').trim();
    const sortBy = (['cnnScore', 'cnnVS', 'cnnAffinity', 'vina'].includes(sortByRaw)
      ? sortByRaw
      : 'cnnScore') as 'cnnScore' | 'cnnVS' | 'cnnAffinity' | 'vina';
    const seed = Math.max(0, Math.min(999999, Math.round(Number(p.seed ?? 42))));
    const includeMetals = p.includeMetals !== false;
    const pocketRadius = Math.max(3.0, Math.min(8.0, Number(p.pocketRadius ?? 5.0)));
    const maxPocketResidues = Math.max(4, Math.min(48, Math.round(Number(p.maxPocketResidues ?? 24))));
    const minimizeIterations = Math.max(1, Math.min(12, Math.round(Number(p.minimizeIterations ?? 4))));
    const cnnWeight = Math.max(0, Math.min(5, Number(p.cnnWeight ?? 1.0)));

    const struct = ctx.inputs?.structure as StructureArtifact | undefined;
    if (!struct || (!struct.pdbText && !struct.cifText)) {
      throw new Error('GNINA 1.3 requires an upstream all-atom StructureArtifact (holo protein–ligand complex or receptor + ligand SDF).');
    }
    if (struct.atomGranularity === 'ca-only' || struct.atomGranularity === 'backbone4') {
      throw new Error(
        `GNINA 1.3 requires all-atom receptor coordinates (including side chains), but "${struct.label}" is ${struct.atomGranularity}.`,
      );
    }

    const rawStructText = struct.pdbText?.trim() ? struct.pdbText : struct.cifText ?? '';
    const prep = extractHoloReceptorAndLigands(rawStructText, {
      ligandCode: ligandCode || undefined,
      includeMetals,
    });

    if (!prep.receptorPdb.trim()) {
      throw new Error(`GNINA 1.3 found no protein receptor atoms in "${struct.label}".`);
    }

    const useExternal = ligandSdf.length > 0;
    const useRcsbCcd = taskMode === 'dock' && !useExternal && dockLigandCcd.length > 0;
    let rcsbCcdSdf: string | null = null;
    if (useRcsbCcd) {
      ctx.progress('init', 0.04, `Fetching 3D ideal SDF for CCD "${dockLigandCcd}" from RCSB...`);
      rcsbCcdSdf = await fetchRcsbCcdIdealSdf(dockLigandCcd, ctx.signal);
    }

    if (!useExternal && !useRcsbCcd && !prep.selectedLigandSdf) {
      if (ligandCode && prep.detectedLigandCodes.length > 0) {
        throw new Error(
          `Ligand "${ligandCode}" not found in "${struct.label}". Detected organic ligands: ${prep.detectedLigandCodes.join(', ')}.`,
        );
      }
      throw new Error(
        taskMode === 'dock'
          ? `No ligand specified for docking into "${struct.label}". Provide a CCD code (e.g. BTN, ATP, I4B), paste a ligand SDF/MOL2, or connect a holo complex.`
          : `No organic small-molecule ligand found in "${struct.label}". Wire a holo protein–ligand structure (e.g. 1STP, 1CTT, 184L) or paste a ligand SDF/MOL2 in Advanced parameters.`,
      );
    }

    const activeEngine = await ensureEngine(ctx, precision);

    const typedReceptor = gninaMod.parseReceptor(prep.receptorPdb, 'pdb');
    const externalFmt = useExternal ? detectExternalLigandFormat(ligandSdf) : 'sdf';
    const typedPoses = useExternal
      ? gninaMod.parseLigands(ligandSdf, externalFmt)
      : useRcsbCcd
        ? gninaMod.parseLigands(rcsbCcdSdf!, 'sdf')
        : gninaMod.parseLigands(prep.selectedLigandSdf!, 'sdf');

    if (!typedPoses || typedPoses.length === 0) {
      throw new Error('No valid ligand poses could be parsed for GNINA 1.3.');
    }

    const selectedLigandName = useExternal
      ? externalFmt === 'mol2'
        ? 'External MOL2'
        : 'External SDF'
      : useRcsbCcd
        ? dockLigandCcd
        : prep.selectedLigand!.compId;

    const rows: GninaPoseScoreRow[] = [];
    let outputPdbText = prep.normalizedPdb;
    let outputCifText: string | undefined = struct.cifText;
    let outputTrajectoryPdb: string | undefined;
    let outputTrajectoryCount: number | undefined;
    let outputBfactorSemantics = struct.bfactorSemantics;
    let trackArtifact: ResidueTrackArtifact | undefined;
    let pocketResiduesScanned: number | undefined;
    let dockBox: DockingScoreSummary['box'] | undefined;
    let dockStats: DockingScoreSummary['dockStats'] | undefined;

    if (taskMode === 'score') {
      for (let i = 0; i < typedPoses.length; i++) {
        const pose = typedPoses[i];
        ctx.progress(
          'score',
          Math.min(0.95, 0.5 + (i / typedPoses.length) * 0.45),
          `Scoring ${selectedLigandName}${typedPoses.length > 1 ? ` pose ${i + 1}/${typedPoses.length}` : ''} with GNINA 1.3...`,
        );
        const res = await activeEngine.scorePose(typedReceptor, pose, {
          contributions: true,
          signal: ctx.signal,
        });
        rows.push({
          poseId: typedPoses.length > 1 ? `${selectedLigandName}#${i + 1}` : selectedLigandName,
          ligandCode: selectedLigandName,
          ligandSource: useExternal
            ? externalFmt === 'mol2'
              ? 'external-mol2'
              : 'external-sdf'
            : 'holo-hetatm',
          heavyAtoms: pose.atoms.length,
          rotors: pose.rotors ?? 0,
          cnnScore: Number(res.cnnScore),
          cnnAffinity: Number(res.cnnAffinity),
          cnnVS: Number(res.cnnVS),
          vinaScore: Number(res.vinaScore),
          models: formatPerModel(res.models),
          vinaTerms: formatVinaTerms(res.contributions, pose.rotors ?? 0),
        });
      }
      if (useExternal) {
        outputPdbText = appendExternalLigandToReceptorPdb(prep.receptorPdb, typedPoses[0].atoms);
        outputCifText = undefined;
      }
    } else if (taskMode === 'minimize') {
      const primaryPose = typedPoses[0];
      ctx.progress(
        'minimize',
        0.52,
        `Scoring initial ${selectedLigandName} pose before rigid-body minimization...`,
      );
      const initial = await activeEngine.scorePose(typedReceptor, primaryPose, {
        contributions: true,
        signal: ctx.signal,
      });
      ctx.progress(
        'minimize',
        0.65,
        `Minimizing ${selectedLigandName} (${minimizeIterations} rigid-body iterations, w=${cnnWeight})...`,
      );
      const minRes = await activeEngine.minimizePose(typedReceptor, primaryPose, {
        iterations: minimizeIterations,
        cnnWeight,
        signal: ctx.signal,
      });
      const finalScored = await activeEngine.scorePose(typedReceptor, minRes.pose, {
        contributions: true,
        signal: ctx.signal,
      });
      const poseRmsd = computePoseHeavyRmsd(primaryPose.atoms, minRes.pose.atoms);

      rows.push({
        poseId: `${selectedLigandName} (min)`,
        ligandCode: selectedLigandName,
        ligandSource: useExternal
          ? externalFmt === 'mol2'
            ? 'external-mol2'
            : 'external-sdf'
          : 'holo-hetatm',
        heavyAtoms: minRes.pose.atoms.length,
        rotors: minRes.pose.rotors ?? primaryPose.rotors ?? 0,
        cnnScore: Number(finalScored.cnnScore),
        cnnAffinity: Number(finalScored.cnnAffinity),
        cnnVS: Number(finalScored.cnnVS),
        vinaScore: Number(finalScored.vinaScore),
        initialCnnScore: Number(initial.cnnScore),
        initialCnnAffinity: Number(initial.cnnAffinity),
        initialCnnVS: Number(initial.cnnVS),
        initialVinaScore: Number(initial.vinaScore),
        poseRmsd,
        models: formatPerModel(finalScored.models),
        vinaTerms: formatVinaTerms(finalScored.contributions, minRes.pose.rotors ?? 0),
      });

      if (!useExternal && prep.selectedLigand) {
        outputPdbText = applyMinimizedHoloLigandCoords(
          prep.normalizedPdb,
          prep.selectedLigand,
          minRes.pose.atoms,
        );
        outputCifText = undefined;
      } else {
        outputPdbText = appendExternalLigandToReceptorPdb(prep.receptorPdb, minRes.pose.atoms);
        outputCifText = undefined;
      }
    } else if (taskMode === 'dock') {
      const dockLigand = typedPoses[0];
      const parsedPocket = pocketResiduesSpec
        ? parsePocketResidueTokensStrict(pocketResiduesSpec, prep.receptorPdb)
        : { keys: [], unmatched: [] };
      const explicitPocketKeys = parsedPocket.keys;
      if (parsedPocket.unmatched.length > 0) {
        throw new Error(
          `Pocket residue selector(s) [${parsedPocket.unmatched.join(', ')}] did not match any residues in "${struct.label}".`,
        );
      }
      if (pocketResiduesSpec && explicitPocketKeys.length === 0) {
        throw new Error(
          `Pocket residues "${pocketResiduesSpec}" did not match any residues in "${struct.label}". Use format like "A45, A79, A90".`,
        );
      }
      const autoboxLig =
        explicitPocketKeys.length === 0 && prep.selectedLigandSdf
          ? gninaMod.parseLigands(prep.selectedLigandSdf, 'sdf')[0]
          : undefined;
      if (explicitPocketKeys.length === 0 && !autoboxLig) {
        throw new Error(
          `Docking into apo receptor "${struct.label}" requires Pocket Residues (e.g. "A45, A79, A90") to define the binding box.`,
        );
      }

      ctx.progress(
        'dock',
        0.48,
        `Building Vina pocket grid & docking ${selectedLigandName} (exhaustiveness=${exhaustiveness})...`,
      );

      const dockRes = await activeEngine.dockPose(typedReceptor, dockLigand, {
        ...(explicitPocketKeys.length > 0
          ? { pocketResidues: explicitPocketKeys }
          : { autoboxLigand: autoboxLig }),
        autoboxAdd,
        seed,
        exhaustiveness,
        numModes,
        minRmsd: 1.0,
        cnnScoring: 'rescore',
        sortBy,
        workers: 1,
        signal: ctx.signal,
        onProgress: (ev: { phase: string; completed: number; total: number }) => {
          const frac = ev.total > 0 ? ev.completed / ev.total : 0;
          if (ev.phase === 'grid') {
            ctx.progress(
              'dock',
              0.48 + frac * 0.1,
              `Building Vina affinity grids (${ev.completed}/${ev.total})...`,
            );
          } else if (ev.phase === 'search') {
            ctx.progress(
              'dock',
              0.58 + frac * 0.22,
              `MC + BFGS flexible search (${ev.completed}/${ev.total} chains)...`,
            );
          } else if (ev.phase === 'rescore') {
            ctx.progress(
              'score',
              0.8 + frac * 0.18,
              `GNINA 1.3 CNN rescoring (${ev.completed}/${ev.total} poses)...`,
            );
          }
        },
      });

      if (!dockRes.poses || dockRes.poses.length === 0) {
        throw new Error(`GNINA 1.3 docking produced 0 valid poses for ${selectedLigandName}.`);
      }

      dockBox = {
        center: [
          Number(dockRes.box.center[0]),
          Number(dockRes.box.center[1]),
          Number(dockRes.box.center[2]),
        ],
        size: [
          Number(dockRes.box.size[0]),
          Number(dockRes.box.size[1]),
          Number(dockRes.box.size[2]),
        ],
      };
      dockStats = {
        gridMs: Number(dockRes.stats.gridMs),
        searchMs: Number(dockRes.stats.searchMs),
        rescoreMs: Number(dockRes.stats.rescoreMs),
        evaluatedCandidates: Number(dockRes.stats.evaluatedCandidates),
        rotors: Number(dockRes.stats.rotors),
      };

      const ligSource: GninaPoseScoreRow['ligandSource'] = useExternal
        ? externalFmt === 'mol2'
          ? 'external-mol2'
          : 'external-sdf'
        : useRcsbCcd
          ? 'rcsb-ccd'
          : 'holo-hetatm';

      for (const dp of dockRes.poses) {
        rows.push({
          poseId: `${selectedLigandName} #${dp.rank}`,
          rank: Number(dp.rank),
          ligandCode: selectedLigandName,
          ligandSource: ligSource,
          heavyAtoms: dp.pose.atoms.length,
          rotors: dp.pose.rotors ?? dockStats.rotors,
          cnnScore: Number(dp.cnnScore),
          cnnAffinity: Number(dp.cnnAffinity),
          cnnVS: Number(dp.cnnVS),
          vinaScore: Number(dp.vinaScore),
          rmsdToBest: Number(dp.rmsdToBest ?? 0),
          models: formatPerModel(dp.models),
          vinaTerms: formatVinaTerms(dp.contributions, dp.pose.rotors ?? dockStats.rotors),
        });
      }

      const topPose = dockRes.poses[0].pose;
      const resCode3 = (useRcsbCcd ? dockLigandCcd : !useExternal && prep.selectedLigand ? prep.selectedLigand.compId : 'LIG').slice(0, 3);
      outputPdbText = appendExternalLigandToReceptorPdb(prep.receptorPdb, topPose.atoms, resCode3, 'L');
      outputCifText = undefined;
      if (dockRes.poses.length > 1) {
        outputTrajectoryPdb = buildMultiPoseDockedPdb(
          prep.receptorPdb,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          dockRes.poses.map((dp: any) => dp.pose.atoms),
          resCode3,
          'L',
        );
        outputTrajectoryCount = dockRes.poses.length;
      }
    } else {
      // pocket-scan: baseline score + per-residue CNN ablation over pocket residues within pocketRadius Å
      const primaryPose = typedPoses[0];
      const pocketResidues = findPocketResidues(
        prep.receptorPdb,
        primaryPose.atoms,
        pocketRadius,
        maxPocketResidues,
      );
      // Only keep keys actually present in GNINA's typed receptor
      const validReceptorKeys = new Set<string>(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        typedReceptor.atoms.map((a: any) => String(a.residue)),
      );
      const scanEntries = pocketResidues.filter((r) => validReceptorKeys.has(r.gninaKey));
      pocketResiduesScanned = scanEntries.length;

      ctx.progress(
        'score',
        0.5,
        `Running GNINA 1.3 pocket scan (${scanEntries.length} residues within ${pocketRadius.toFixed(1)} Å)...`,
      );

      const baseline = await activeEngine.scorePose(typedReceptor, primaryPose, {
        contributions: true,
        signal: ctx.signal,
      });
      const maskRes =
        scanEntries.length > 0
          ? await activeEngine.maskContributions(typedReceptor, primaryPose, {
              residues: scanEntries.map((r) => r.gninaKey),
              signal: ctx.signal,
            })
          : { baseline, atoms: [], residues: [] };

      rows.push({
        poseId: selectedLigandName,
        ligandCode: selectedLigandName,
        ligandSource: useExternal
          ? externalFmt === 'mol2'
            ? 'external-mol2'
            : 'external-sdf'
          : 'holo-hetatm',
        heavyAtoms: primaryPose.atoms.length,
        rotors: primaryPose.rotors ?? 0,
        cnnScore: Number(baseline.cnnScore),
        cnnAffinity: Number(baseline.cnnAffinity),
        cnnVS: Number(baseline.cnnVS),
        vinaScore: Number(baseline.vinaScore),
        models: formatPerModel(baseline.models),
        vinaTerms: formatVinaTerms(baseline.contributions, primaryPose.rotors ?? 0),
      });

      // Map residue ablation ΔCNN_VS onto receptor polymer sequence + B-factors
      const deltaByResId = new Map<string, number>();
      const deltaByKey = new Map<string, number>();
      for (const item of maskRes.residues ?? []) {
        const key = String(item.residue);
        const dv = Number(item.cnnVS ?? 0);
        deltaByKey.set(key, dv);
      }
      for (const entry of scanEntries) {
        const dv = deltaByKey.get(entry.gninaKey);
        if (dv !== undefined && Number.isFinite(dv)) {
          deltaByResId.set(entry.residueId, dv);
        }
      }

      const sourceResidues =
        struct.sourceResidues && struct.sourceResidues.length > 0
          ? struct.sourceResidues
          : parseStructureText(prep.normalizedPdb).sourceResidues;
      const L = sourceResidues.length || struct.length;
      const trackValues = new Float32Array(L);
      const residueIds: string[] = new Array(L);

      let sum = 0;
      let minV = Infinity;
      let maxV = -Infinity;
      let posCount = 0;
      let scannedCount = 0;

      for (let i = 0; i < L; i++) {
        const sr = sourceResidues[i];
        const authRid = sr ? `${sr.authChainId}${sr.resNum}${sr.insCode ?? ''}` : `A${i + 1}`;
        const projRid = sr ? `${sr.projectedChainId}${sr.resNum}${sr.insCode ?? ''}` : authRid;
        residueIds[i] = authRid;
        const matchedKey = deltaByResId.has(authRid)
          ? authRid
          : deltaByResId.has(projRid)
            ? projRid
            : null;
        const dv = matchedKey ? deltaByResId.get(matchedKey)! : 0;
        trackValues[i] = dv;
        if (matchedKey) {
          scannedCount++;
          sum += dv;
          if (dv < minV) minV = dv;
          if (dv > maxV) maxV = dv;
          if (dv >= 0.05) posCount++;
        }
      }

      const baseForPaint = useExternal
        ? appendExternalLigandToReceptorPdb(prep.receptorPdb, primaryPose.atoms)
        : prep.normalizedPdb;
      outputPdbText = paintStructureBfactors(baseForPaint, trackValues);
      outputCifText = undefined;
      outputBfactorSemantics = 'gnina-contrib';

      const runHashTrack = fnv1a(
        JSON.stringify({
          structId: struct.id,
          lig: selectedLigandName,
          pocketRadius,
          maxPocketResidues,
          precision,
        }),
      );
      const effMin = Number.isFinite(minV) ? Math.min(minV, 0) : 0;
      const effMax = Number.isFinite(maxV) ? Math.max(maxV, 0.1) : 0.1;
      trackArtifact = {
        kind: 'ResidueTrack',
        id: `gnina-pocket-${precision}-${runHashTrack}`,
        label: `GNINA 1.3 Pocket ΔCNN_VS (${selectedLigandName} · ${scannedCount} pocket res · max +${(Number.isFinite(maxV) ? maxV : 0).toFixed(2)})`,
        sourceModelId: 'gnina',
        parentStructureId: struct.id,
        quantity: 'gnina-contrib',
        unit: 'ΔCNN_VS',
        range: [effMin, effMax],
        threshold: 0.05,
        residues: struct.sequence.length === L ? struct.sequence : struct.sequence.slice(0, L).padEnd(L, 'X'),
        residueIds,
        values: trackValues,
        summary: {
          mean: scannedCount > 0 ? sum / scannedCount : 0,
          min: Number.isFinite(minV) ? minV : 0,
          max: Number.isFinite(maxV) ? maxV : 0,
          fractionAboveThreshold: L > 0 ? posCount / L : 0,
        },
      };
    }

    const primaryRow = rows[0];
    const dockingMetrics: DockingScoreSummary = {
      engine: 'gnina-1.3',
      taskMode,
      precision,
      ensemble: Array.isArray(activeEngine.ensemble)
        ? [...activeEngine.ensemble]
        : ['dense_1.3', 'dense_1.3_PT_KD_3', 'crossdock_default2018_KD_4'],
      receptorName: struct.label,
      includeMetals,
      detectedLigands: prep.detectedLigandCodes,
      selectedLigand: selectedLigandName,
      ...(pocketResiduesScanned !== undefined ? { pocketResiduesScanned, pocketRadius } : {}),
      ...(dockBox ? { box: dockBox } : {}),
      ...(dockStats ? { dockStats } : {}),
      rows,
    };

    const externalLigandHash = useExternal ? fnv1a(ligandSdf) : '';
    const runHash = fnv1a(
      JSON.stringify({
        structId: struct.id,
        taskMode,
        precision,
        selectedLigandName,
        externalLigandHash,
        includeMetals,
        minimizeIterations: taskMode === 'minimize' ? minimizeIterations : 0,
        cnnWeight: taskMode === 'minimize' ? cnnWeight : 0,
        pocketRadius: taskMode === 'pocket-scan' ? pocketRadius : 0,
        maxPocketResidues: taskMode === 'pocket-scan' ? maxPocketResidues : 0,
        dockKey: taskMode === 'dock' ? `${pocketResiduesSpec}|${exhaustiveness}|${numModes}|${sortBy}|${seed}|${autoboxAdd}` : '',
      }),
    );

    const caCopy = new Float32Array(struct.caCoords);
    const bbCopy = new Float32Array(struct.backbone4Coords);
    const obsCopy = struct.observedBackboneMask
      ? new Uint8Array(struct.observedBackboneMask)
      : undefined;

    const modeTag =
      taskMode === 'dock'
        ? 'dock'
        : taskMode === 'minimize'
          ? 'min'
          : taskMode === 'pocket-scan'
            ? 'pocket'
            : 'score';

    const emittedDiagnostics =
      useExternal || taskMode === 'dock' || taskMode === 'minimize'
        ? parseStructureText(outputPdbText).diagnostics
        : struct.diagnostics;

    const outStructure: StructureArtifact = {
      ...struct,
      id: `gnina-${modeTag}-${struct.id}-${precision}-${runHash}`,
      label: `${struct.label} · GNINA ${selectedLigandName} (CNN ${primaryRow.cnnScore.toFixed(2)} · pK ${primaryRow.cnnAffinity.toFixed(2)})`,
      sourceModelId: 'gnina',
      parentStructureId: struct.id,
      caCoords: caCopy,
      backbone4Coords: bbCopy,
      ...(obsCopy ? { observedBackboneMask: obsCopy } : {}),
      pdbText: outputPdbText,
      cifText: outputCifText,
      trajectoryPdb: outputTrajectoryPdb,
      trajectoryFrameCount: outputTrajectoryCount,
      diagnostics: emittedDiagnostics,
      bfactorSemantics: outputBfactorSemantics,
      dockingMetrics,
      provenance: {
        ...struct.provenance,
        quantization: precision,
      },
    };

    const outputs: Record<string, unknown> = { structure: outStructure };
    const transferViews: Array<ArrayBufferView | undefined> = [caCopy, bbCopy, obsCopy];
    if (trackArtifact) {
      outputs.track = trackArtifact;
      transferViews.push(trackArtifact.values);
    }

    return {
      outputs,
      transferables: collectOwnedTransferables(transferViews),
    };
  },

  async dispose() {
    try {
      engine?.destroy();
    } catch {
      /* ignore */
    }
    try {
      ownedDevice?.destroy();
    } catch {
      /* ignore */
    }
    engine = null;
    ownedDevice = null;
    loadedPrecision = null;
  },
});

export {};
