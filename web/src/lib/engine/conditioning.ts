// Model-agnostic conditioning compiler:
// 1. Builds a canonical CanonicalConditionIntent from block parameters.
// 2. Compiles canonical intent -> RFD4-Proteina `sel(...)` / `C_*` dict (decoupled from scheduler).
// 3. Compiles canonical intent + template structure -> Genie 3 `createFeaturesFromSpec(...)` batch
//    for both protein motif scaffolding and PPI binder design!

import { parseStructureText } from '$lib/protein-ops/pdb';
import { getBenchmarkById } from './benchmarks';
import type {
  CanonicalConditionIntent,
  ConditionSpecArtifact,
  StructureArtifact,
} from './types';

export interface ResidueRangeToken {
  chain: string;
  start: number;
  end: number;
}

const GENIE3_RESTYPE_1 = 'ARNDCQEGHILKMFPSTWYV';

/** Parse comma/whitespace-separated tokens like "A54, A56, A16-35" into structured ranges. */
export function parseResidueRangeTokens(raw: string): ResidueRangeToken[] {
  const out: ResidueRangeToken[] = [];
  for (const tok of raw.split(/[,\s]+/).filter(Boolean)) {
    const m = /^([A-Za-z])?:?(\d+)(?:-(\d+))?$/.exec(tok.trim());
    if (!m) continue;
    const chain = (m[1] || 'A').toUpperCase();
    const start = parseInt(m[2], 10);
    const end = m[3] ? parseInt(m[3], 10) : start;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) continue;
    out.push({ chain, start, end });
  }
  return out;
}

/** Extract a clean, model-agnostic CanonicalConditionIntent from node params. */
export function buildCanonicalConditionIntent(
  params: Record<string, unknown>,
): CanonicalConditionIntent {
  const contig = String(params.contig ?? '70-70').trim() || '70-70';
  const lockChain = String(params.lockChain ?? '').trim().toUpperCase().slice(0, 1) || undefined;
  const hotspots = String(params.hotspots ?? '')
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const motifSpans = String(params.motifSpan ?? '')
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const unindexedMotif = Boolean(params.unindexedMotif);
  const hbondMode = (String(params.hbondMode ?? 'none') as CanonicalConditionIntent['hbondMode']) || 'none';
  const hbondSelection = String(params.hbondSelection ?? '').trim() || undefined;
  const cathCode = String(params.cathCode ?? '').trim() || undefined;
  const benchmarkId = String(params.benchmarkId ?? '').trim() || undefined;
  const pdbId = String(params.pdbId ?? '').trim().toUpperCase() || undefined;

  return {
    contig,
    ...(lockChain ? { lockChain } : {}),
    ...(hotspots.length > 0 ? { hotspots } : {}),
    ...(motifSpans.length > 0 ? { motifSpans } : {}),
    ...(unindexedMotif ? { unindexedMotif: true } : {}),
    ...(hbondMode && hbondMode !== 'none' ? { hbondMode } : {}),
    ...(hbondSelection ? { hbondSelection } : {}),
    ...(cathCode ? { cathCode } : {}),
    ...(benchmarkId ? { benchmarkId } : {}),
    ...(pdbId ? { pdbId } : {}),
  };
}

/**
 * Compile model-agnostic CanonicalConditionIntent -> RFD4-Proteina `conditions` dictionary.
 * Keeps all RFD4-specific `sel(...)` and `C_*` syntax cleanly isolated from the core DAG engine.
 */
export function compileConditionToRfd4(intent: CanonicalConditionIntent): {
  contig: string;
  conditions: Record<string, unknown>;
  tags: string[];
} {
  const contig = intent.contig.trim() || '70-70';
  const conditions: Record<string, unknown> = {};
  const tags: string[] = [contig];

  if (intent.lockChain) {
    const ch = intent.lockChain[0].toUpperCase();
    conditions.C_CRD = { true: [{ select: `sel('${ch}/*/*/*')` }] };
    conditions.C_SEQ = { true: [{ select: `sel('${ch}/*/*/*')` }] };
    tags.push(`lock:${ch}`);
  }

  if (intent.hotspots && intent.hotspots.length > 0) {
    const byChain = new Map<string, string[]>();
    for (const r of parseResidueRangeTokens(intent.hotspots.join(', '))) {
      const list = byChain.get(r.chain) ?? [];
      list.push(r.start === r.end ? `${r.start}` : `${r.start}-${r.end}`);
      byChain.set(r.chain, list);
    }
    const exprs = [...byChain.entries()].map(([ch, nums]) => `sel('${ch}/*/[${nums.join(',')}]/*')`);
    if (exprs.length > 0) {
      conditions.C_HOT = { true: [{ select: exprs.join(' | ') }] };
      tags.push(`hot:${intent.hotspots.join(', ')}`);
    }
  }

  if (intent.motifSpans && intent.motifSpans.length > 0) {
    const byChain = new Map<string, string[]>();
    for (const r of parseResidueRangeTokens(intent.motifSpans.join(', '))) {
      const list = byChain.get(r.chain) ?? [];
      list.push(r.start === r.end ? `${r.start}` : `${r.start}-${r.end}`);
      byChain.set(r.chain, list);
    }
    const exprs = [...byChain.entries()].map(([ch, nums]) => `sel('${ch}/*/[${nums.join(',')}]/*')`);
    if (exprs.length > 0) {
      const sel = exprs.join(' | ');
      conditions.C_CRDGRP = { 1: [{ select: sel }] };
      const prevSeqTrue = (conditions.C_SEQ as { true?: Array<{ select: string }> } | undefined)?.true ?? [];
      conditions.C_SEQ = { true: [...prevSeqTrue, { select: sel }] };
      if (intent.unindexedMotif) {
        conditions.C_IDX = { false: [{ select: sel, group: 'motif' }] };
      }
      tags.push(
        intent.unindexedMotif
          ? `unidx-motif:${intent.motifSpans.join(', ')}`
          : `motif:${intent.motifSpans.join(', ')}`,
      );
    }
  }

  if (intent.hbondMode && intent.hbondMode !== 'none' && intent.hbondSelection) {
    const expr = intent.hbondSelection.includes('sel(')
      ? intent.hbondSelection
      : `sel('${intent.hbondSelection}')`;
    conditions.C_HB = { [intent.hbondMode]: [{ select: expr }] };
    tags.push(`HB:${intent.hbondMode}`);
  }

  if (intent.cathCode) {
    conditions.C_CATH = { [intent.cathCode]: [{ select: "sel('*/*/*/*')" }] };
    tags.push(`CATH:${intent.cathCode}`);
  }

  return { contig, conditions, tags };
}

/** Check whether a ConditionSpec can be compiled and executed by Genie 3 (protein motif / PPI binder). */
export function checkGenie3ConditionCompatibility(
  cond: ConditionSpecArtifact | undefined,
): { supported: boolean; reason?: string } {
  if (!cond) return { supported: true };
  if (cond.cifUrl?.startsWith('preset:')) {
    return {
      supported: false,
      reason: `RFD4 built-in preset (${cond.cifUrl.slice(7)}) uses RFD4-specific context archives — switch to Custom/Benchmark conditioning with a PDB Input template for Genie 3, or run with RFD4-Proteina.`,
    };
  }
  const intent = cond.intent;
  if (!intent) {
    return {
      supported: false,
      reason: 'Raw RFD4 JSON overrides cannot run on Genie 3 — use RFD4-Proteina or standard Conditioning fields.',
    };
  }
  if (intent.hbondMode && intent.hbondMode !== 'none') {
    return {
      supported: false,
      reason: 'Hydrogen-bond atom constraints require RFD4-Proteina (Genie 3 is Cα backbone diffusion).',
    };
  }
  if (intent.cathCode) {
    return {
      supported: false,
      reason: 'CATH fold-code conditioning is supported by RFD4-Proteina only.',
    };
  }
  if (intent.unindexedMotif) {
    return {
      supported: false,
      reason: 'Unindexed motif placement requires RFD4-Proteina — uncheck "Unindexed Motif" (indexed motif scaffolding) to run on Genie 3.',
    };
  }
  return { supported: true };
}

/**
 * Build an automatic ConditionSpec when a user wires a template PDB directly into Genie 3
 * without an explicit Conditioning block:
 * - If the template came from a curated benchmark, applies its curated hotspots / motif spec
 *   (adapting binder length to Genie 3's Length slider).
 * - Otherwise, designs a de novo binder of `defaultLength` aa against the template's primary chain.
 */
export function buildDefaultGenie3ConditionFromTemplate(
  template: StructureArtifact,
  defaultLength: number,
): ConditionSpecArtifact {
  const clampedLen = Math.max(20, Math.min(200, Math.round(defaultLength || 80)));

  if (template.benchmarkId) {
    const bench = getBenchmarkById(template.benchmarkId);
    if (bench) {
      if (!bench.supportsGenie3) {
        throw new Error(
          `${bench.shortLabel} requires all-atom enzyme/ligand/DNA conditioning, which is supported by RFD4-Proteina rather than Genie 3.`,
        );
      }
      const cp = bench.conditionParams;
      let contig = cp.contig;
      if (bench.category === 'binder' && contig.includes('/0')) {
        // Respect Genie 3's Length slider for the designed binder chain while keeping the curated target crop
        const parts = contig.split(/[,\s]+/).filter(Boolean);
        const updated = parts.map((tok) => (/^\d+(?:-\d+)?$/.test(tok) ? `${clampedLen}-${clampedLen}` : tok));
        contig = updated.join(',');
      }
      const intent = buildCanonicalConditionIntent({
        contig,
        lockChain: cp.lockChain ?? '',
        hotspots: cp.hotspots ?? '',
        motifSpan: cp.motifSpan ?? '',
        unindexedMotif: false,
        benchmarkId: bench.id,
        pdbId: bench.pdbId,
      });
      return {
        kind: 'ConditionSpec',
        id: `auto-g3-${bench.id}-L${clampedLen}`,
        label: bench.shortLabel,
        contig: intent.contig,
        conditions: {},
        intent,
        inputCif: template.cifText ?? template.pdbText,
        specMode: true,
        summary: `${bench.shortLabel} (auto)`,
      };
    }
  }

  // Generic template PDB: design a de novo binder against the template's first protein chain
  const firstChain = (template.chainIds?.[0] || 'A').toUpperCase();
  const intent = buildCanonicalConditionIntent({
    contig: `/0,${clampedLen}-${clampedLen}`,
    lockChain: firstChain,
  });
  return {
    kind: 'ConditionSpec',
    id: `auto-g3-binder-${template.id}-L${clampedLen}`,
    label: `Binder vs ${template.label}`,
    contig: intent.contig,
    conditions: {},
    intent,
    inputCif: template.cifText ?? template.pdbText,
    specMode: true,
    summary: `binder ${clampedLen} aa vs chain ${firstChain}`,
  };
}

export interface Genie3CompiledSpec {
  mode: 'unconditional' | 'binder' | 'motif';
  length: number;
  asymId: Int32Array;
  symId: Int32Array;
  entityId: Int32Array;
  residueIndex: Int32Array;
  condGroup: Int32Array;
  condSeqMask: Float32Array;
  condStructMask: Float32Array;
  condInterfaceMask: Float32Array;
  gtRestype: Float32Array;
  gtAtomPositions: Float32Array;
  chainLengths: number[];
  summary: string;
}

function parseLengthRangeMid(tok: string): number | null {
  const m = /^(\d+)(?:-(\d+))?$/.exec(tok.trim());
  if (!m) return null;
  const lo = parseInt(m[1], 10);
  const hi = m[2] ? parseInt(m[2], 10) : lo;
  return Math.max(1, Math.round((lo + hi) / 2));
}

function writeRestypeOneHot(gtRestype: Float32Array, tokenIdx: number, aa1: string): void {
  const idx = GENIE3_RESTYPE_1.indexOf(aa1.toUpperCase());
  if (idx >= 0) {
    gtRestype[tokenIdx * 20 + idx] = 1;
  }
}

/**
 * Compile a canonical ConditionSpecArtifact + upstream template PDB/mmCIF into the exact
 * typed arrays expected by Genie 3's `createFeaturesFromSpec(...)`.
 */
export function compileConditionToGenie3Spec(
  cond: ConditionSpecArtifact,
  template: StructureArtifact | undefined,
  defaultLength: number,
): Genie3CompiledSpec {
  const compat = checkGenie3ConditionCompatibility(cond);
  if (!compat.supported) {
    throw new Error(compat.reason ?? 'Conditioning spec is not supported by Genie 3');
  }

  const intent = cond.intent!;
  const rawStructure = template?.cifText ?? template?.pdbText ?? cond.inputCif ?? '';

  // Pure length contig (no lockChain, no motifSpans): unconditional monomer of requested length
  const hasLock = Boolean(intent.lockChain);
  const hasMotif = Boolean(intent.motifSpans && intent.motifSpans.length > 0);
  if (!hasLock && !hasMotif) {
    const len = parseLengthRangeMid(intent.contig) ?? defaultLength;
    const clamped = Math.max(30, Math.min(256, len));
    const residueIndex = new Int32Array(clamped);
    for (let i = 0; i < clamped; i++) residueIndex[i] = i + 1;
    return {
      mode: 'unconditional',
      length: clamped,
      asymId: new Int32Array(clamped),
      symId: new Int32Array(clamped),
      entityId: new Int32Array(clamped),
      residueIndex,
      condGroup: new Int32Array(clamped),
      condSeqMask: new Float32Array(clamped),
      condStructMask: new Float32Array(clamped),
      condInterfaceMask: new Float32Array(clamped),
      gtRestype: new Float32Array(clamped * 20),
      gtAtomPositions: new Float32Array(clamped * 3),
      chainLengths: [clamped],
      summary: `unconditional L=${clamped}`,
    };
  }

  if (!rawStructure.trim()) {
    throw new Error(
      'Genie 3 motif/binder conditioning requires a template 3D structure — wire a PDB Input block into Conditioning (template) or Genie 3 (template).',
    );
  }

  const parsed = parseStructureText(rawStructure);
  if (parsed.nResidues < 3) {
    throw new Error('Template structure must contain at least 3 protein residues with Cα atoms.');
  }

  // Index template residues by chain + residue number for exact canonical RCSB lookup
  interface TplRes {
    chain: string;
    resNum: number;
    aa1: string;
    x: number;
    y: number;
    z: number;
  }
  const tplList: TplRes[] = [];
  const tplMap = new Map<string, TplRes>();
  for (let i = 0; i < parsed.nResidues; i++) {
    const entry: TplRes = {
      chain: parsed.chainIds[i] || 'A',
      resNum: parsed.resNums[i] ?? i + 1,
      aa1: parsed.sequence[i] || 'A',
      x: parsed.caCoords[i * 3 + 0],
      y: parsed.caCoords[i * 3 + 1],
      z: parsed.caCoords[i * 3 + 2],
    };
    tplList.push(entry);
    tplMap.set(`${entry.chain}:${entry.resNum}`, entry);
  }

  // ── Mode 1: PPI Binder Design (Genie 3 binder-first convention: chain 0 = binder, chain 1 = fixed target) ──
  if (hasLock) {
    const lockChain = intent.lockChain!.toUpperCase();
    const parts = intent.contig.split(/[,\s]+/).filter(Boolean);
    let binderLen = defaultLength;
    const targetRanges: ResidueRangeToken[] = [];

    for (const part of parts) {
      if (part === '/0') continue;
      const rangeTokens = parseResidueRangeTokens(part);
      if (rangeTokens.length > 0 && /^[A-Za-z]/.test(part)) {
        targetRanges.push(...rangeTokens);
      } else {
        const mid = parseLengthRangeMid(part);
        if (mid !== null) binderLen = mid;
      }
    }

    let targetRes: TplRes[] = [];
    if (targetRanges.length > 0) {
      for (const r of targetRanges) {
        for (const t of tplList) {
          if (t.chain === r.chain && t.resNum >= r.start && t.resNum <= r.end) {
            targetRes.push(t);
          }
        }
      }
    } else {
      targetRes = tplList.filter((t) => t.chain === lockChain);
    }

    if (targetRes.length < 3) {
      throw new Error(
        `Could not find target residues for chain "${lockChain}" in the wired template structure.`,
      );
    }

    const hotspotSet = new Set<string>();
    for (const h of parseResidueRangeTokens((intent.hotspots ?? []).join(', '))) {
      for (let rn = h.start; rn <= h.end; rn++) {
        hotspotSet.add(`${h.chain}:${rn}`);
      }
    }

    const clampedBinder = Math.max(20, Math.min(200, binderLen));
    const totalLen = clampedBinder + targetRes.length;
    if (totalLen > 384) {
      throw new Error(
        `Total binder + target length (${totalLen} aa) exceeds Genie 3 browser WebGPU limit (384 aa). Crop the target range in the contig.`,
      );
    }

    const asymId = new Int32Array(totalLen);
    const symId = new Int32Array(totalLen);
    const entityId = new Int32Array(totalLen);
    const residueIndex = new Int32Array(totalLen);
    const condGroup = new Int32Array(totalLen);
    const condSeqMask = new Float32Array(totalLen);
    const condStructMask = new Float32Array(totalLen);
    const condInterfaceMask = new Float32Array(totalLen);
    const gtRestype = new Float32Array(totalLen * 20);
    const gtAtomPositions = new Float32Array(totalLen * 3);

    // Compute target centroid so fixed coordinates sit cleanly at the origin
    let cx = 0, cy = 0, cz = 0;
    for (const t of targetRes) {
      cx += t.x; cy += t.y; cz += t.z;
    }
    cx /= targetRes.length; cy /= targetRes.length; cz /= targetRes.length;

    // Chain 0: designed binder (unconditioned)
    for (let i = 0; i < clampedBinder; i++) {
      asymId[i] = 0;
      entityId[i] = 0;
      residueIndex[i] = i + 1;
    }

    // Chain 1: fixed target with hotspots
    let hotspotMatches = 0;
    for (let j = 0; j < targetRes.length; j++) {
      const idx = clampedBinder + j;
      const t = targetRes[j];
      asymId[idx] = 1;
      entityId[idx] = 1;
      residueIndex[idx] = j + 1;
      condGroup[idx] = 1;
      condStructMask[idx] = 1;
      condSeqMask[idx] = 1;
      if (hotspotSet.has(`${t.chain}:${t.resNum}`)) {
        condInterfaceMask[idx] = 1;
        hotspotMatches++;
      }
      writeRestypeOneHot(gtRestype, idx, t.aa1);
      gtAtomPositions[idx * 3 + 0] = t.x - cx;
      gtAtomPositions[idx * 3 + 1] = t.y - cy;
      gtAtomPositions[idx * 3 + 2] = t.z - cz;
    }

    return {
      mode: 'binder',
      length: totalLen,
      asymId,
      symId,
      entityId,
      residueIndex,
      condGroup,
      condSeqMask,
      condStructMask,
      condInterfaceMask,
      gtRestype,
      gtAtomPositions,
      chainLengths: [clampedBinder, targetRes.length],
      summary: `binder ${clampedBinder} aa + target ${targetRes.length} aa (${hotspotMatches} hotspots)`,
    };
  }

  // ── Mode 2: Indexed Motif Scaffolding (single chain with fixed motif segments) ──
  const contigParts = intent.contig.split(/[,\s]+/).filter((s) => s && s !== '/0');
  type Segment = { kind: 'scaffold'; len: number } | { kind: 'motif'; residues: TplRes[] };
  const segments: Segment[] = [];
  let hasExplicitMotifInContig = false;

  for (const part of contigParts) {
    if (/^[A-Za-z]/.test(part)) {
      const ranges = parseResidueRangeTokens(part);
      const res: TplRes[] = [];
      for (const r of ranges) {
        for (let rn = r.start; rn <= r.end; rn++) {
          const hit = tplMap.get(`${r.chain}:${rn}`);
          if (hit) res.push(hit);
        }
      }
      if (res.length > 0) {
        hasExplicitMotifInContig = true;
        segments.push({ kind: 'motif', residues: res });
      }
    } else {
      const mid = parseLengthRangeMid(part);
      if (mid !== null) segments.push({ kind: 'scaffold', len: mid });
    }
  }

  // If contig didn't explicitly include the motif spans (e.g. contig="80-80" + motifSpan="A16-35"),
  // build balanced scaffold flanking segments around the motif spans!
  if (!hasExplicitMotifInContig) {
    const motifGroups = parseResidueRangeTokens((intent.motifSpans ?? []).join(', '));
    const motifBlocks: TplRes[][] = [];
    for (const r of motifGroups) {
      const res: TplRes[] = [];
      for (let rn = r.start; rn <= r.end; rn++) {
        const hit = tplMap.get(`${r.chain}:${rn}`);
        if (hit) res.push(hit);
      }
      if (res.length > 0) motifBlocks.push(res);
    }
    if (motifBlocks.length === 0) {
      throw new Error(
        `Could not locate motif residues (${(intent.motifSpans ?? []).join(', ')}) in the wired template structure.`,
      );
    }
    const targetTotal = parseLengthRangeMid(intent.contig) ?? defaultLength;
    const motifTotal = motifBlocks.reduce((s, b) => s + b.length, 0);
    const remaining = Math.max(motifBlocks.length + 1, targetTotal - motifTotal);
    const gapSlots = motifBlocks.length + 1;
    const baseGap = Math.max(4, Math.floor(remaining / gapSlots));
    segments.length = 0;
    for (let i = 0; i < motifBlocks.length; i++) {
      segments.push({ kind: 'scaffold', len: baseGap });
      segments.push({ kind: 'motif', residues: motifBlocks[i] });
    }
    segments.push({ kind: 'scaffold', len: Math.max(4, remaining - baseGap * motifBlocks.length) });
  }

  const totalLen = segments.reduce(
    (s, seg) => s + (seg.kind === 'scaffold' ? seg.len : seg.residues.length),
    0,
  );
  if (totalLen < 10 || totalLen > 384) {
    throw new Error(`Motif scaffold length (${totalLen} aa) must be between 10 and 384 aa for Genie 3.`);
  }

  const motifResAll = segments.flatMap((s) => (s.kind === 'motif' ? s.residues : []));
  if (motifResAll.length === 0) {
    throw new Error('No matching motif residues found in the wired template structure.');
  }

  let cx = 0, cy = 0, cz = 0;
  for (const m of motifResAll) {
    cx += m.x; cy += m.y; cz += m.z;
  }
  cx /= motifResAll.length; cy /= motifResAll.length; cz /= motifResAll.length;

  const asymId = new Int32Array(totalLen);
  const symId = new Int32Array(totalLen);
  const entityId = new Int32Array(totalLen);
  const residueIndex = new Int32Array(totalLen);
  const condGroup = new Int32Array(totalLen);
  const condSeqMask = new Float32Array(totalLen);
  const condStructMask = new Float32Array(totalLen);
  const condInterfaceMask = new Float32Array(totalLen);
  const gtRestype = new Float32Array(totalLen * 20);
  const gtAtomPositions = new Float32Array(totalLen * 3);

  let cursor = 0;
  for (const seg of segments) {
    if (seg.kind === 'scaffold') {
      for (let k = 0; k < seg.len; k++) {
        residueIndex[cursor] = cursor + 1;
        cursor++;
      }
    } else {
      for (const m of seg.residues) {
        residueIndex[cursor] = cursor + 1;
        condGroup[cursor] = 1;
        condStructMask[cursor] = 1;
        condSeqMask[cursor] = 1;
        writeRestypeOneHot(gtRestype, cursor, m.aa1);
        gtAtomPositions[cursor * 3 + 0] = m.x - cx;
        gtAtomPositions[cursor * 3 + 1] = m.y - cy;
        gtAtomPositions[cursor * 3 + 2] = m.z - cz;
        cursor++;
      }
    }
  }

  return {
    mode: 'motif',
    length: totalLen,
    asymId,
    symId,
    entityId,
    residueIndex,
    condGroup,
    condSeqMask,
    condStructMask,
    condInterfaceMask,
    gtRestype,
    gtAtomPositions,
    chainLengths: [totalLen],
    summary: `motif scaffold L=${totalLen} (${motifResAll.length} fixed motif aa)`,
  };
}
