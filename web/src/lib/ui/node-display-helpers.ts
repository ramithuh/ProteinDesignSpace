import { getBenchmarkById } from '$lib/engine/benchmarks';
import { parseFixedPositionsAndChains } from '$lib/engine/mpnn-spec';
import { LOCALFOLD_MODEL_OPTIONS, RFD4_BENCHMARK_OPTIONS } from '$lib/engine/registry';
import { validateSequenceText } from '$lib/engine/param-validation';
import type {
  CanonicalArtifact,
  MutationMatrixArtifact,
  StructureArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

const INVERSE_FOLD_ADAPTERS = new Set([
  'web-protein-mpnn',
  'ligandmpnn',
  'lasermpnn',
  'antifold-esmif1',
]);

const PASSTHROUGH_STRUCTURE_ADAPTERS = new Set([
  'lasermpnn',
  'thermompnn',
  'dyna1',
]);

export interface UpstreamLockSummary {
  status: 'active' | 'unlocked' | 'overridden' | 'warning';
  sourceAdapterId: string;
  sourceLabel: string;
  lockedChains: string[];
  designingChains: string[];
  fixedResidueCount: number;
  hasFixedMotif: boolean;
  chainRoles: Record<string, string>;
  chainLengths: Record<string, number>;
  nodeBadge: string;
  detailText: string;
  locksAllDesignChainsWarning: string | null;
}

/**
 * Format the canvas output badge for a `MutationMatrixArtifact`, excluding wild-type
 * self-mutations (`alphabet[aaIdx] === residues[posIdx]`) so WT `0.00` never masks
 * negative `max LLR` or positive `min ΔΔG`.
 */
export function formatMutationMatrixBadge(art: MutationMatrixArtifact): string {
  const residues = (art.residues ?? '').toUpperCase();
  const alphabet = (art.alphabet || 'ACDEFGHIKLMNPQRSTVWY').toUpperCase();
  const K = alphabet.length;
  const nPos =
    art.residueIds && art.residueIds.length > 0 ? art.residueIds.length : residues.length;
  const lowerBetter = art.direction === 'lower-is-better';

  let bestVal: number | null = null;
  if (K > 0 && art.values && art.values.length > 0) {
    const numRows = Math.ceil(art.values.length / K);
    for (let posIdx = 0; posIdx < numRows; posIdx++) {
      const wt = posIdx < residues.length ? residues[posIdx] : '';
      const rowOff = posIdx * K;
      for (let aaIdx = 0; aaIdx < K; aaIdx++) {
        const idx = rowOff + aaIdx;
        if (idx >= art.values.length) break;
        const mut = alphabet[aaIdx];
        if (!mut || (wt && mut === wt)) continue;
        const v = art.values[idx];
        if (!Number.isFinite(v)) continue;
        if (bestVal === null || (lowerBetter ? v < bestVal : v > bestVal)) {
          bestVal = v;
        }
      }
    }
  }

  if (bestVal === null && art.targetedMutations && art.targetedMutations.length > 0) {
    for (const tm of art.targetedMutations) {
      if (!tm.mutant) continue;
      if (tm.mutant.toUpperCase() === (tm.wildType ?? '').toUpperCase()) continue;
      if (!Number.isFinite(tm.score)) continue;
      if (bestVal === null || (lowerBetter ? tm.score < bestVal : tm.score > bestVal)) {
        bestVal = tm.score;
      }
    }
  }

  const fmt = (v: number): string => {
    const s = v.toFixed(2);
    return s === '-0.00' ? '0.00' : s;
  };

  if (art.metric === 'ddg-kcal-mol') {
    return bestVal !== null
      ? `${nPos} pos · min ΔΔG ${fmt(bestVal)}`
      : `${nPos}×${K || 20} ΔΔG`;
  }
  const statLbl = lowerBetter ? 'min LLR' : 'max LLR';
  return bestVal !== null
    ? `${nPos} pos · ${statLbl} ${fmt(bestVal)}`
    : `${nPos}×${K || 20} LLR`;
}

function num(val: unknown, fallback: number): number {
  if (val === undefined || val === null || typeof val === 'boolean') return fallback;
  if (typeof val === 'string' && val.trim() === '') return fallback;
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Format the compact 1-line parameter chip displayed on a node card for all 25 active
 * registry adapters.
 */
export function formatNodeParamSummary(
  adapterId: string,
  params?: Record<string, unknown>,
  opts?: { autoModeHint?: string },
): string {
  const p = params ?? {};
  switch (adapterId) {
    case 'pdb-input': {
      const crop = String(p.cropSegments ?? '').trim();
      const cropTag = crop ? ` · ${crop}` : '';
      if (p.mode === 'pdb-id') {
        const id = String(p.pdbId ?? '').trim().toUpperCase();
        return id ? `RCSB ${id}${cropTag}` : 'Fetch by PDB ID';
      }
      if (p.mode === 'custom') {
        const rawFn = String(p.filename ?? '').trim();
        const fname = rawFn && rawFn !== 'demo_3helix_53aa.pdb' ? rawFn : '';
        const hasText = String(p.pdbText ?? '').trim().length > 0;
        if (!hasText) {
          return fname ? `${fname} (empty)` : 'No PDB loaded';
        }
        return `${fname || 'custom PDB'}${cropTag}`;
      }
      return `53 aa 3-helix demo${cropTag}`;
    }
    case 'sequence-input': {
      const raw = String(p.sequence ?? '');
      const check = validateSequenceText(raw);
      const emptyCount = check.emptyHeaders?.length ?? 0;
      const emptyNote = emptyCount > 0 ? ` · ${emptyCount} empty` : '';
      if (check.numSequences > 1) {
        return `${check.numSequences} seqs · ${check.totalResidues} aa${emptyNote}`;
      }
      return `${check.totalResidues || 0} aa FASTA${emptyNote}`;
    }
    case 'rfd4-proteina': {
      const mode = String(p.mode ?? 'unconditional');
      const steps = num(p.nsteps, 60);
      if (mode === 'binder') {
        const hotCount = String(p.hotspots ?? '').split(/[,\s]+/).filter(Boolean).length;
        const hotTag = hotCount > 0 ? ` · ${hotCount} hotspots` : '';
        return `Binder${hotTag} · ${steps} steps`;
      }
      if (mode === 'motif') {
        const tag = p.unindexedMotif ? 'Unindexed Motif' : 'Motif';
        const motif = String(p.motifSpan ?? '').trim();
        return `${tag}${motif ? ` (${motif})` : ''} · ${steps} steps`;
      }
      if (mode === 'conditioned') {
        const contig = String(p.contig ?? '').trim();
        return `Conditioned${contig ? ` · ${contig}` : ''} · ${steps} steps`;
      }
      if (mode === 'preset') {
        const presetId = String(p.presetId ?? '');
        const opt = RFD4_BENCHMARK_OPTIONS.find((o) => o.id === presetId);
        const presetLabel =
          opt?.label ?? (presetId === 'yaml_5o45_binder' ? '5O45 PD-L1 Binder (YAML)' : 'Preset task');
        return `${presetLabel} · ${steps} steps`;
      }
      if (mode === 'partial') return `Partial (${num(p.partialFraction, 0.5)}) · ${steps} steps`;
      if (mode === 'inverse-folding') return `Inverse folding · ${steps} steps`;
      const cath = String(p.cathCode ?? '').trim();
      return `${cath ? `CATH ${cath} · ` : ''}${num(p.length, 70)} aa · ${steps} steps`;
    }
    case 'la-proteina': {
      const len = num(p.length, 80);
      const steps = num(p.steps, 400);
      const mode = String(p.mode ?? 'sde').toUpperCase();
      const n = num(p.numDesigns, 1);
      return `${len} aa · ${steps} steps · ${mode}${n > 1 ? ` · ×${n}` : ''}`;
    }
    case 'proteina': {
      const len = num(p.length, 80);
      const steps = num(p.steps, 400);
      const modeTag = String(p.mode ?? 'sc') === 'vf' ? 'ODE' : 'SDE';
      const n = num(p.numDesigns, 1);
      return `${len} aa · ${steps} steps · ${modeTag}${n > 1 ? ` · ×${n}` : ''}`;
    }
    case 'sid-protein': {
      const len = num(p.length, 80);
      const steps = num(p.steps, 16);
      const prec = String(p.precision ?? 'int8').toUpperCase();
      const n = num(p.numDesigns, 1);
      return `${len} aa · ${steps} steps · ${prec}${n > 1 ? ` · ×${n}` : ''}`;
    }
    case 'genie3': {
      const len = num(p.length, 80);
      const steps = num(p.nSampleStep, 100);
      const tm = String(p.taskMode ?? 'unconditional');
      if (tm === 'binder') {
        const hotCount = String(p.hotspots ?? '').split(/[,\s]+/).filter(Boolean).length;
        return `Binder · ${len} aa${hotCount > 0 ? ` · ${hotCount} hot` : ''} · ${steps} steps`;
      }
      if (tm === 'motif') {
        const motif = String(p.motifSpan ?? '').trim();
        return `Motif${motif ? ` (${motif})` : ''} · ${steps} steps`;
      }
      if (opts?.autoModeHint) return `${opts.autoModeHint} · ${len} aa · ${steps} steps`;
      return `${len} aa · ${steps} steps`;
    }
    case 'genie2': {
      const len = num(p.length, 80);
      const steps = num(p.steps, 1000);
      const motif = String(p.motifRange ?? '').trim();
      const prec = String(p.precision ?? 'fp16').toUpperCase();
      const n = num(p.numDesigns, 1);
      const nTag = n > 1 ? ` · ×${n}` : '';
      if (motif) {
        return `Motif (${motif}) · ${len} aa · ${steps} steps${nTag}`;
      }
      return `${len} aa · ${steps} steps · ${prec}${nTag}`;
    }
    case 'rfdiffusion': {
      const tm = String(p.taskMode ?? 'unconditional');
      const len = num(p.length, 80);
      const steps = num(p.numSteps, 50);
      const prec = String(p.precision ?? 'int8').toUpperCase();
      const n = num(p.numDesigns, 1);
      const nTag = n > 1 ? ` · ×${n}` : '';
      if (tm === 'motif') {
        const contig = String(p.contig ?? '').trim();
        return `Motif${contig ? ` (${contig})` : ''} · ${steps} steps${nTag}`;
      }
      if (tm === 'symmetry') {
        const sym = String(p.symmetry ?? 'c3').toUpperCase();
        return `${sym} · ${len} aa · ${steps} steps${nTag}`;
      }
      return `${len} aa · ${steps} steps · ${prec}${nTag}`;
    }
    case 'frameflow': {
      const len = num(p.length, 80);
      const steps = num(p.steps, 100);
      const prec = String(p.precision ?? 'fp16').toUpperCase();
      const n = num(p.numDesigns, 1);
      return `${len} aa · ${steps} steps · ${prec}${n > 1 ? ` · ×${n}` : ''}`;
    }
    case 'foldflow2': {
      const len = num(p.length, 80);
      const steps = num(p.steps, 50);
      const prec = String(p.precision ?? 'int8').toUpperCase();
      const n = num(p.numDesigns, 1);
      return `${len} aa · ${steps} steps · ${prec}${n > 1 ? ` · ×${n}` : ''}`;
    }
    case 'boltzgen': {
      const tm = String(p.taskMode ?? 'unconditional');
      const variant = String(p.variant ?? 'diverse');
      const len = num(p.length, 48);
      const steps = num(p.numSteps, 20);
      const n = num(p.numDesigns, 1);
      const nTag = n > 1 ? ` · ×${n}` : '';
      if (tm === 'binder') return `Binder (${variant}) · ${len} aa · ${steps} steps${nTag}`;
      if (tm === 'motif') return `Motif (${variant}) · ${len} aa · ${steps} steps${nTag}`;
      return `${variant} · ${len} aa · ${steps} steps${nTag}`;
    }
    case 'protpardelle': {
      const ckpt = String(p.checkpoint ?? 'cc91_epoch383').replace(/_epoch\d+$/, '');
      const mode = String(p.taskMode ?? 'unconditional');
      const steps = num(p.numSteps, 100);
      if (mode === 'motif') return `${ckpt} · Motif · ${steps} steps`;
      if (mode === 'partial') return `${ckpt} · Partial (${num(p.rewindSteps, 40)})`;
      return `${ckpt} · ${num(p.length, 80)} aa · ${steps} steps`;
    }
    case 'evodiff': {
      const tm = String(p.taskMode ?? 'unconditional');
      const len = num(p.length, 64);
      const n = num(p.numSequences, 2);
      const prec = String(p.precision ?? 'int8').toUpperCase();
      if (tm === 'infill') return `Infill · ${n} seq${n === 1 ? '' : 's'} · ${prec}`;
      return `${len} aa · ${n} seq${n === 1 ? '' : 's'} · ${prec}`;
    }
    case 'esmc': {
      const mode = String(p.mode ?? 'wildtype');
      const prec = String(p.precision ?? 'int8').toUpperCase();
      const tag = mode === 'masked' ? 'Masked LLR' : mode === 'fill' ? 'Fill' : 'WT LLR';
      return `${tag} · ${prec}`;
    }
    case 'dyna1': {
      const prec = String(p.precision ?? 'int8').toUpperCase();
      const mode = String(p.inputMode ?? 'auto');
      const modeLbl =
        mode === 'seq+struct'
          ? 'Seq+Struct'
          : mode === 'seq-only'
            ? 'Seq-only'
            : mode === 'struct-only'
              ? 'Struct-only'
              : 'Auto (Seq+Struct)';
      return `${modeLbl} · ${prec}`;
    }
    case 'thermompnn': {
      const chain = String(p.chain ?? '').trim();
      const agg = String(p.bfactorAggregate ?? 'best-stabilizing');
      const aggLbl =
        agg === 'mean-ddg'
          ? 'mean ΔΔG'
          : agg === 'worst-destabilizing'
            ? 'max ΔΔG'
            : 'min ΔΔG';
      const prec = String(p.precision ?? 'fp16').toUpperCase();
      return `${chain ? `Chain ${chain} · ` : ''}${aggLbl} · ${prec}`;
    }
    case 'web-protein-mpnn': {
      const fam =
        p.family === 'proteinmpnn'
          ? 'ProteinMPNN'
          : p.family === 'ca'
            ? 'Cα-ProteinMPNN'
            : 'SolubleMPNN';
      const ckpt = String(p.modelName ?? 'v_48_020');
      const n = num(p.numSequences, 2);
      const t = num(p.temperature, 0.1);
      return `${fam} ${ckpt} · ${n} seq${n === 1 ? '' : 's'} · T=${t}`;
    }
    case 'ligandmpnn': {
      const ckptShort = String(p.checkpoint ?? 'ligandmpnn_v_32_020_25').replace(
        /^ligandmpnn_/,
        '',
      );
      const n = num(p.numSequences, 2);
      const t = num(p.temperature, 0.1);
      const useLig = p.useLigandContext !== false;
      return `${ckptShort} · ${n} seq${n === 1 ? '' : 's'} · T=${t} · ${useLig ? 'holo' : 'apo'}`;
    }
    case 'lasermpnn': {
      const n = num(p.numSequences, 2);
      const t = num(p.temperature, 0.1);
      const prec = String(p.precision ?? 'fp16').toUpperCase();
      return `All-atom · ${n} seq${n === 1 ? '' : 's'} · T=${t} · ${prec}`;
    }
    case 'antifold-esmif1': {
      const variant = String(p.variant ?? 'antifold');
      const n = num(p.numSequences, 2);
      const t = num(p.temperature, 0.2);
      if (variant === 'esmif1') {
        return `ESM-IF1 · ${n} seq${n === 1 ? '' : 's'} · T=${t}`;
      }
      const regions = String(p.regions ?? 'CDR1 CDR2 CDR3').trim();
      return `AntiFold${regions ? ` (${regions})` : ''} · ${n} seq${n === 1 ? '' : 's'} · T=${t}`;
    }
    case 'localfold-af2': {
      const modelId = String(p.model ?? 'monomer');
      const meta =
        LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
      const rec = num(p.recycles, 1);
      const msaMode = String(p.msaMode ?? 'single');
      const msaTag =
        msaMode === 'mmseqs2'
          ? 'MMseqs2'
          : msaMode === 'target-only'
            ? 'target MSA'
            : msaMode === 'a3m'
              ? '.a3m'
              : 'single-seq';
      if (meta.family === 'esmfold2') {
        return `${meta.shortLabel} · single-seq · ${num(p.esmfold2Steps, 15)} scheduled steps`;
      }
      if (meta.family === 'af3') {
        const steps = num(p.diffusionSteps, 25);
        return `LocalFold · ${msaTag} · ${steps} steps · ${rec} rec`;
      }
      return `LocalFold · ${msaTag} · ${rec} rec`;
    }
    case 'top-k-filter': {
      const topK = num(p.topK, 2);
      const mode = String(p.mode ?? 'per-backbone');
      const maxTotal = num(p.maxTotal, 8);
      return `Top-${topK} (${mode === 'global' ? 'Global' : 'Per-backbone'}) · max ${maxTotal}`;
    }
    case 'structure-compare': {
      const layout = String(p.layout ?? 'overlay');
      const align = p.alignKabsch !== false;
      const color = String(p.colorMode ?? 'by-model');
      return `${layout === 'side-by-side' ? 'Side-by-side' : 'Overlay'}${align ? ' · Kabsch' : ''}${color === 'plddt' ? ' · pLDDT' : ''}`;
    }
    case 'designability-scorer': {
      return `scRMSD < ${num(p.scRmsdThreshold, 2.0).toFixed(1)} Å · pLDDT ≥ ${num(p.plddtThreshold, 70)}`;
    }
    default:
      return '';
  }
}

/**
 * Compute pre-run and post-run inherited lock awareness for inverse-folding nodes
 * (`web-protein-mpnn`, `ligandmpnn`, `lasermpnn`, `antifold-esmif1`), including warning when
 * user `fixedPositions` or `targetChain` locks all remaining designable chains (Bio P0-4).
 */
export function computeUpstreamLockSummary(
  graph: WorkflowGraph,
  nodeId: string,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
): UpstreamLockSummary | null {
  const node = graph.nodes.find((n) => n.id === nodeId);
  if (!node || !INVERSE_FOLD_ADAPTERS.has(node.adapterId)) return null;

  // Walk upstream along `structure` edges (stepping through pass-through blocks when unrun)
  const visited = new Set<string>([nodeId]);
  let curId = nodeId;
  let sourceNode: (typeof graph.nodes)[number] | null = null;
  let materializedStruct: StructureArtifact | null = null;

  while (true) {
    const edge = graph.edges.find((e) => e.toNode === curId && e.toPort === 'structure');
    if (!edge || visited.has(edge.fromNode)) break;
    visited.add(edge.fromNode);
    const upstream = graph.nodes.find((n) => n.id === edge.fromNode);
    if (!upstream) break;

    const outArt = outputsByNode[upstream.id]?.[edge.fromPort];
    if (outArt?.kind === 'StructureArtifact') {
      sourceNode = upstream;
      materializedStruct = outArt;
      break;
    }

    sourceNode = upstream;
    if (PASSTHROUGH_STRUCTURE_ADAPTERS.has(upstream.adapterId)) {
      curId = upstream.id;
      continue;
    }
    break;
  }

  if (!sourceNode) return null;

  const sourceLabel = sourceNode.label || sourceNode.adapterId;
  let lockedChains: string[] = [];
  let designingChains: string[] = [];
  let fixedResidueCount = 0;
  let hasFixedMotif = false;
  let chainRoles: Record<string, string> = {};
  const chainLengths: Record<string, number> = {};

  if (materializedStruct) {
    lockedChains = [...new Set((materializedStruct.lockedChains ?? []).map((c) => c.trim()).filter(Boolean))];
    fixedResidueCount = materializedStruct.fixedResidueIndices?.length ?? 0;
    hasFixedMotif = fixedResidueCount > 0;
    chainRoles = { ...(materializedStruct.chainRoles ?? {}) };

    const uniqueChains: string[] = [];
    const seenC = new Set<string>();
    for (const c of materializedStruct.chainIds ?? []) {
      const t = c.trim();
      if (!t) continue;
      chainLengths[t] = (chainLengths[t] ?? 0) + 1;
      if (!seenC.has(t)) {
        seenC.add(t);
        uniqueChains.push(t);
      }
    }
    for (const c of Object.keys(chainRoles)) {
      if (!seenC.has(c)) {
        seenC.add(c);
        uniqueChains.push(c);
      }
    }
    designingChains = uniqueChains.filter((c) => !lockedChains.includes(c));
    if (lockedChains.length === 0 && !hasFixedMotif) {
      return null;
    }
  } else if (sourceNode.adapterId === 'genie3') {
    const gp = sourceNode.params ?? {};
    const tm = String(gp.taskMode ?? 'unconditional');
    const lockChainParam = String(gp.lockChain ?? '').trim().toUpperCase();
    const motifSpan = String(gp.motifSpan ?? '').trim();
    const contig = String(gp.contig ?? '').trim();
    const binderLen = Number(gp.length ?? 80);

    const tplEdge = graph.edges.find(
      (e) => e.toNode === sourceNode!.id && (e.toPort === 'structure' || e.toPort === 'template'),
    );
    const tplNode = tplEdge ? graph.nodes.find((x) => x.id === tplEdge.fromNode) : null;
    const tplOut = tplEdge ? outputsByNode[tplEdge.fromNode]?.[tplEdge.fromPort] : undefined;
    const benchId =
      tplNode?.adapterId === 'pdb-input' ? String(tplNode.params?.benchmarkId ?? '').trim() : '';
    const bench = benchId ? getBenchmarkById(benchId) : undefined;
    const targetLen =
      tplOut?.kind === 'StructureArtifact'
        ? tplOut.length
        : bench && bench.category !== 'motif'
          ? Math.max(0, bench.approxTokens - (Number.isFinite(binderLen) ? binderLen : 80))
          : undefined;

    const autoMotif = tm === 'unconditional' && Boolean(tplEdge) && bench?.category === 'motif';
    const autoBinder =
      tm === 'unconditional' && Boolean(tplEdge) && (!bench || bench.category !== 'motif');

    if (tm === 'binder' || Boolean(lockChainParam) || contig.includes('/0') || autoBinder) {
      const lockCh = lockChainParam || 'B';
      const desCh = lockCh === 'A' ? 'B' : 'A';
      lockedChains = [lockCh];
      designingChains = [desCh];
      chainRoles = { [desCh]: 'binder', [lockCh]: 'target' };
      if (Number.isFinite(binderLen) && binderLen > 0) chainLengths[desCh] = binderLen;
      if (targetLen && targetLen > 0) chainLengths[lockCh] = targetLen;
    } else if (tm === 'motif' || Boolean(motifSpan) || autoMotif) {
      hasFixedMotif = true;
      designingChains = ['A'];
      chainRoles = { A: 'scaffold' };
    } else {
      return null;
    }
  } else if (sourceNode.adapterId === 'rfd4-proteina') {
    const rp = sourceNode.params ?? {};
    const mode = String(rp.mode ?? 'unconditional');
    const presetId = String(rp.presetId ?? '');
    const lockChainParam = String(rp.lockChain ?? '').trim().toUpperCase();
    const contig = String(rp.contig ?? '').trim();
    const binderLen = Number(rp.length ?? 70);

    const isBinder =
      mode === 'binder' ||
      (mode === 'preset' && (presetId === 'ppi_pdl1' || presetId === 'yaml_5o45_binder')) ||
      Boolean(lockChainParam) ||
      (mode === 'conditioned' && contig.includes('/0'));
    const isMotif =
      mode === 'motif' ||
      (mode === 'preset' && (presetId === 'motif_idx_1prw' || presetId === 'motif_unidx_1prw'));

    if (isBinder) {
      let lockCh = lockChainParam || 'A';
      let desCh = lockCh === 'B' ? 'A' : 'B';
      if (contig.includes('/0')) {
        const parts = contig.split('/0');
        const beforeToks = (parts[0] ?? '').split(/[,\s]+/).filter(Boolean);
        const afterStr = parts.slice(1).join('/0');
        const beforeBareLengths =
          beforeToks.length > 0 && beforeToks.every((t) => /^\d+(?:-\d+)?[PRDN]?$/i.test(t));
        const afterHasTemplateSpan = /[A-Za-z]\d/.test(afterStr);
        if (beforeBareLengths && afterHasTemplateSpan && !lockChainParam) {
          desCh = 'A';
          lockCh = 'B';
        }
      }
      lockedChains = [lockCh];
      designingChains = [desCh];
      chainRoles = { [lockCh]: 'target', [desCh]: 'binder' };
      if (Number.isFinite(binderLen) && binderLen > 0) chainLengths[desCh] = binderLen;
    } else if (isMotif) {
      hasFixedMotif = true;
      designingChains = ['A'];
      chainRoles = { A: 'scaffold' };
    } else {
      return null;
    }
  } else if (sourceNode.adapterId === 'rfdiffusion') {
    const gp = sourceNode.params ?? {};
    const tm = String(gp.taskMode ?? 'unconditional');
    const contig = String(gp.contig ?? '').trim();
    if (contig.includes('/0')) {
      lockedChains = ['B'];
      designingChains = ['A'];
      chainRoles = { A: 'binder', B: 'target' };
    } else if (tm === 'motif') {
      hasFixedMotif = true;
      designingChains = ['A'];
      chainRoles = { A: 'scaffold' };
    } else {
      return null;
    }
  } else if (sourceNode.adapterId === 'boltzgen') {
    const bp = sourceNode.params ?? {};
    const tm = String(bp.taskMode ?? 'unconditional');
    if (tm === 'binder') {
      lockedChains = ['B'];
      designingChains = ['A'];
      chainRoles = { A: 'binder', B: 'target' };
    } else if (tm === 'motif') {
      hasFixedMotif = true;
      designingChains = ['A'];
      chainRoles = { A: 'scaffold' };
    } else {
      return null;
    }
  } else {
    return null;
  }

  const np = node.params ?? {};
  const unlockInherited = Boolean(np.unlockInherited);
  let userLockedChains: string[] = [];
  try {
    const parsed = parseFixedPositionsAndChains(String(np.fixedPositions ?? ''));
    userLockedChains = parsed.lockedChains;
  } catch {
    userLockedChains = [];
  }

  const explicitDesignChains =
    node.adapterId === 'web-protein-mpnn'
      ? String(np.designChains ?? '')
          .split(/[,\s]+/)
          .map((s) => s.trim().toUpperCase())
          .filter(Boolean)
      : [];

  const targetChain =
    node.adapterId === 'antifold-esmif1'
      ? String(np.targetChain ?? '').trim().toUpperCase()
      : '';

  let locksAllDesignChainsWarning: string | null = null;
  if (!unlockInherited && explicitDesignChains.length === 0 && lockedChains.length > 0) {
    if (designingChains.length > 0 && designingChains.every((c) => userLockedChains.includes(c))) {
      const allLocked = [...new Set([...lockedChains, ...userLockedChains])];
      locksAllDesignChainsWarning = `0 designable residues: ${sourceLabel} already locks ${lockedChains.join(', ')} and Fixed positions locks ${userLockedChains.join(', ')} (all chains ${allLocked.join('+')} locked). Clear "${designingChains.join(', ')}" from Fixed positions or enable "Unlock inherited".`;
    } else if (targetChain && lockedChains.includes(targetChain)) {
      locksAllDesignChainsWarning = `0 designable residues: targetChain "${targetChain}" is auto-locked by ${sourceLabel} (designable: ${designingChains.join(', ') || 'none'}). Clear targetChain or set it to ${designingChains[0] ?? 'the binder chain'}.`;
    }
  }

  const formatChainDetail = (c: string, roleFallback: string): string => {
    const role = chainRoles[c] ?? roleFallback;
    const len = chainLengths[c];
    return `${c} (${role}${len ? `, ${len} aa` : ''})`;
  };

  let status: UpstreamLockSummary['status'] = 'active';
  let nodeBadge = '';
  let detailText = '';

  if (locksAllDesignChainsWarning) {
    status = 'warning';
    const allLocked = [...new Set([...lockedChains, ...userLockedChains, ...(targetChain ? [targetChain] : [])])];
    nodeBadge = `⚠ 0 designable aa — ${allLocked.join('+')} locked`;
    detailText = locksAllDesignChainsWarning;
  } else if (unlockInherited) {
    status = 'unlocked';
    nodeBadge = '🔓 Inherited locks ignored';
    detailText = `Inherited locks from ${sourceLabel} are ignored (Unlock inherited enabled).`;
  } else if (explicitDesignChains.length > 0) {
    status = 'overridden';
    nodeBadge = `Design ${explicitDesignChains.join(',')}`;
    detailText = `Explicit designChains (${explicitDesignChains.join(', ')}) overrides inherited locks from ${sourceLabel}.`;
  } else if (lockedChains.length > 0) {
    status = 'active';
    nodeBadge = `🔒 Lock ${lockedChains.join(',')} · Design ${designingChains.join(',') || 'none'}`;
    const lockDesc = lockedChains.map((c) => formatChainDetail(c, 'target')).join(', ');
    const desDesc =
      designingChains.length > 0
        ? designingChains.map((c) => formatChainDetail(c, 'binder')).join(', ')
        : 'none';
    detailText = `Inherited from ${sourceLabel}: Chain ${lockDesc} locked · Chain ${desDesc} designed.`;
  } else {
    status = 'active';
    nodeBadge = `🔒 ${fixedResidueCount > 0 ? `${fixedResidueCount} motif aa` : 'Motif'} fixed`;
    detailText = `Inherited from ${sourceLabel}: ${fixedResidueCount > 0 ? `${fixedResidueCount} motif residue(s)` : 'Motif residues'} automatically fixed; scaffold residues designed.`;
  }

  return {
    status,
    sourceAdapterId: sourceNode.adapterId,
    sourceLabel,
    lockedChains,
    designingChains,
    fixedResidueCount,
    hasFixedMotif,
    chainRoles,
    chainLengths,
    nodeBadge,
    detailText,
    locksAllDesignChainsWarning,
  };
}
