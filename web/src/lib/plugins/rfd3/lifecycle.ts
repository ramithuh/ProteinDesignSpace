import { isCrystallizationBuffer } from '$lib/engine/conditioning';
import type { StructureArtifact } from '$lib/engine/types';
import { parseStructureText } from '$lib/protein-ops/pdb';

export interface ParsedCaResidue {
  chainId: string;
  resId: number;
  ca: [number, number, number];
  labelChainId?: string;
  labelSeqId?: number;
}

export interface Rfd3EngineLike {
  resources?: {
    closed?: boolean;
    loss?: { reason?: string; message?: string };
  };
  trim?: () => void;
  destroy?: () => void;
}

export function isReusableRfd3Engine(
  eng: Rfd3EngineLike | null | undefined,
  loadedPrecision: string | null,
  requestedPrecision: string,
): boolean {
  return Boolean(eng && !eng.resources?.closed && loadedPrecision === requestedPrecision);
}

export function trimRfd3EngineSafely(eng: Rfd3EngineLike | null | undefined): boolean {
  if (!eng || eng.resources?.closed || typeof eng.trim !== 'function') return false;
  try {
    eng.trim();
    return true;
  } catch {
    return false;
  }
}

export async function drainGpuDeviceQueue(
  device: { queue?: { onSubmittedWorkDone?: () => Promise<unknown> } } | null | undefined,
  timeoutMs = 1500,
): Promise<void> {
  const done = device?.queue?.onSubmittedWorkDone?.();
  if (!done || typeof done.then !== 'function') return;
  let timer: ReturnType<typeof setTimeout> | null = null;
  try {
    await Promise.race([
      done.catch(() => {}),
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== null) clearTimeout(timer);
  }
}

function normalizeContigPart(part: string): string {
  if (part === '0' || part === '/0') return '/0';
  const bareRange = /^(\d+)-(\d+)[PRDN]?$/i.exec(part);
  if (bareRange) {
    const lo = Number(bareRange[1]);
    const hi = Number(bareRange[2]);
    return String(Math.max(1, Math.round((lo + hi) / 2)));
  }
  const bareLen = /^(\d+)[PRDN]$/i.exec(part);
  if (bareLen) return bareLen[1];
  return part;
}

/**
 * Normalize contig strings so comma-separated RFD3 syntax (`25,A16-35,25`),
 * benchmark range syntax (`25-25,A16-35,25-25`), and slash-separated RFdiffusion1 syntax
 * (`25/A16-35/25`) all resolve to concrete RFD3 tokens.
 */
export function normalizeRfd3Contig(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return '';
  if (trimmed.includes(',')) {
    return trimmed
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      .map(normalizeContigPart)
      .join(',');
  }
  if (trimmed.includes('/')) {
    return trimmed
      .split('/')
      .map((s) => s.trim())
      .filter(Boolean)
      .map(normalizeContigPart)
      .join(',');
  }
  return normalizeContigPart(trimmed);
}

/**
 * Sanitize PDB text for RFD3's upstream parser:
 * - Keep only the first MODEL if multi-model
 * - Strip ATOM/HETATM lines carrying non-blank insertion codes (col 27)
 */
export function sanitizePdbForRfd3(pdbText: string): string {
  const lines = pdbText.split(/\r?\n/);
  const out: string[] = [];
  let sawModel = false;
  let inModel = false;
  for (const line of lines) {
    if (line.startsWith('MODEL')) {
      if (sawModel) break;
      sawModel = true;
      inModel = true;
      continue;
    }
    if (line.startsWith('ENDMDL') && inModel) break;
    if ((line.startsWith('ATOM  ') || line.startsWith('HETATM')) && (line[26] ?? ' ') !== ' ') {
      continue;
    }
    out.push(line);
  }
  return out.join('\n');
}

export function extractPdbCaResidues(pdbText: string): ParsedCaResidue[] {
  const out: ParsedCaResidue[] = [];
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
    if (!Number.isFinite(resId) || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      continue;
    }
    const key = `${chainId}:${resId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ chainId, resId, ca: [x, y, z] });
  }
  return out;
}

export function extractInputCaResidues(
  structureText: string,
  format: 'pdb' | 'cif',
  upstreamStruct?: StructureArtifact,
): ParsedCaResidue[] {
  if (format === 'pdb') {
    return extractPdbCaResidues(structureText);
  }
  const out: ParsedCaResidue[] = [];
  const seen = new Set<string>();
  try {
    const parsed = parseStructureText(structureText);
    for (let i = 0; i < parsed.nResidues; i++) {
      const ins = parsed.insCodes?.[i] ?? '';
      if (ins && ins.trim() !== '') continue;
      const src = parsed.sourceResidues?.[i];
      const chainId = (src?.authChainId || parsed.chainIds[i] || 'A').trim() || 'A';
      const resId = src?.resNum ?? parsed.resNums[i];
      const x = parsed.caCoords[i * 3];
      const y = parsed.caCoords[i * 3 + 1];
      const z = parsed.caCoords[i * 3 + 2];
      if (!Number.isFinite(resId) || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        continue;
      }
      const key = `${chainId}:${resId}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({
        chainId,
        resId,
        ca: [x, y, z],
        ...(src?.labelChainId ? { labelChainId: src.labelChainId } : {}),
        ...(typeof src?.labelSeqId === 'number' ? { labelSeqId: src.labelSeqId } : {}),
      });
    }
  } catch {
    /* fall through to artifact arrays / synthesized pdbText fallback */
  }
  if (out.length === 0 && upstreamStruct) {
    if (upstreamStruct.sourceResidues?.length && upstreamStruct.caCoords?.length) {
      for (const src of upstreamStruct.sourceResidues) {
        if (src.insCode && src.insCode.trim() !== '') continue;
        const i = src.index;
        if (i < 0 || (i + 1) * 3 > upstreamStruct.caCoords.length) continue;
        const chainId = (src.authChainId || src.projectedChainId || 'A').trim() || 'A';
        const resId = src.resNum;
        const x = upstreamStruct.caCoords[i * 3];
        const y = upstreamStruct.caCoords[i * 3 + 1];
        const z = upstreamStruct.caCoords[i * 3 + 2];
        const key = `${chainId}:${resId}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({
          chainId,
          resId,
          ca: [x, y, z],
          ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
          ...(typeof src.labelSeqId === 'number' ? { labelSeqId: src.labelSeqId } : {}),
        });
      }
    } else if (upstreamStruct.pdbText?.trim()) {
      return extractPdbCaResidues(sanitizePdbForRfd3(upstreamStruct.pdbText));
    }
  }
  return out;
}

export interface ResolvedRfd3UpstreamStructure {
  structureText?: string;
  structureFormat?: 'pdb' | 'cif';
  caResidues: ParsedCaResidue[];
  fallbackResKeys: string[];
  autoLigandNames: string[];
}

/**
 * R1: Prefer `upstreamStruct.cifText` over synthesized `pdbText` so formal charges (e.g. `ZN` +2)
 * and `_chem_comp_bond` tables (e.g. `DHZ`) are preserved, while extracting `caResidues` and
 * `fallbackResKeys` identically across BOTH mmCIF and PDB inputs.
 */
export function resolveRfd3UpstreamStructure(
  upstreamStruct: StructureArtifact | undefined,
  needsStructure: boolean,
  rawLigand = '',
): ResolvedRfd3UpstreamStructure {
  if (!needsStructure || !upstreamStruct) {
    return { caResidues: [], fallbackResKeys: [], autoLigandNames: [] };
  }
  let structureText: string | undefined;
  let structureFormat: 'pdb' | 'cif' | undefined;
  if (upstreamStruct.cifText?.trim()) {
    structureText = upstreamStruct.cifText;
    structureFormat = 'cif';
  } else if (upstreamStruct.pdbText?.trim()) {
    structureText = sanitizePdbForRfd3(upstreamStruct.pdbText);
    structureFormat = 'pdb';
  }
  const caResidues =
    structureText && structureFormat
      ? extractInputCaResidues(structureText, structureFormat, upstreamStruct)
      : [];
  const fallbackResKeys = caResidues.map((r) => `${r.chainId}${r.resId}`);
  const autoLigandNames =
    !rawLigand && upstreamStruct.diagnostics?.ligandNames?.length
      ? upstreamStruct.diagnostics.ligandNames.filter((name) => !isCrystallizationBuffer(name))
      : [];
  return {
    structureText,
    structureFormat,
    caResidues,
    fallbackResKeys,
    autoLigandNames,
  };
}

export function buildAutoFlankContigFromResidues(
  residues: readonly ParsedCaResidue[],
  targetLength: number,
): string {
  if (residues.length === 0) {
    throw new Error('Input structure has no valid ATOM Cα residues for RFdiffusion3 motif scaffolding.');
  }
  const primaryChain = residues[0].chainId;
  const chainRes = residues.filter((r) => r.chainId === primaryChain);
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
  return parts.join(',');
}

export function buildFullProteinContigFromResidues(residues: readonly ParsedCaResidue[]): string {
  if (residues.length === 0) {
    throw new Error('Input structure has no valid ATOM Cα residues for RFdiffusion3 partial diffusion.');
  }
  const parts: string[] = [];
  let prevChain: string | null = null;
  let runStart: number | null = null;
  let runEnd: number | null = null;
  let runChain: string | null = null;

  const flush = () => {
    if (runChain === null || runStart === null || runEnd === null) return;
    parts.push(runStart === runEnd ? `${runChain}${runStart}` : `${runChain}${runStart}-${runEnd}`);
    runStart = null;
    runEnd = null;
    runChain = null;
  };

  for (const r of residues) {
    if (prevChain !== null && r.chainId !== prevChain) {
      flush();
      parts.push('/0');
    }
    if (runChain === r.chainId && runEnd !== null && r.resId === runEnd + 1) {
      runEnd = r.resId;
    } else {
      flush();
      runChain = r.chainId;
      runStart = r.resId;
      runEnd = r.resId;
    }
    prevChain = r.chainId;
  }
  flush();
  return parts.join(',');
}

export function buildNativeCaMapFromResidues(
  residues: readonly ParsedCaResidue[],
): Map<string, [number, number, number]> {
  const map = new Map<string, [number, number, number]>();
  for (const r of residues) {
    map.set(`${r.chainId}${r.resId}`, r.ca);
    map.set(`${r.chainId}:${r.resId}`, r.ca);
    if (r.labelChainId && typeof r.labelSeqId === 'number') {
      const lKey1 = `${r.labelChainId}${r.labelSeqId}`;
      const lKey2 = `${r.labelChainId}:${r.labelSeqId}`;
      if (!map.has(lKey1)) map.set(lKey1, r.ca);
      if (!map.has(lKey2)) map.set(lKey2, r.ca);
    }
  }
  return map;
}

export function extractNativeMotifCaCoordsFromResidues(
  residues: readonly ParsedCaResidue[],
  contig: string,
): number[][] {
  const selected: number[][] = [];
  const seen = new Set<string>();
  for (const part of contig.split(',').map((s) => s.trim()).filter(Boolean)) {
    if (part === '/0' || /^\d+$/.test(part)) continue;
    const m =
      /^([^:]+):(-?\d+)(?:-(-?\d+))?$/.exec(part) ??
      /^([A-Za-z0-9]+?)(-?\d+)(?:-(-?\d+))?$/.exec(part);
    if (!m) continue;
    const chainId = m[1];
    const start = Number(m[2]);
    const end = m[3] != null ? Number(m[3]) : start;
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

export interface Rfd3MotifMeta {
  fixedResidueIndices?: number[];
  motifIndices?: number[];
  nativeMotifCoords?: number[][];
}

export function computeRfd3MotifMeta(opts: {
  mode: string;
  caResidues: readonly ParsedCaResidue[];
  resolvedLen: number;
  effectiveContig?: string;
  motifAssignments?: ReadonlyArray<{ source?: string; token?: number }>;
  features?: {
    is_protein?: ArrayLike<number>;
    ref_motif_token_type?: ArrayLike<number>;
  };
}): Rfd3MotifMeta | null {
  const { mode, caResidues, resolvedLen, effectiveContig, motifAssignments, features } = opts;
  if (mode !== 'motif' || caResidues.length === 0) return null;

  if (Array.isArray(motifAssignments) && motifAssignments.length > 0) {
    const nativeMap = buildNativeCaMapFromResidues(caResidues);
    const matchedIndices: number[] = [];
    const nativeCoords: number[][] = [];
    for (const asgn of motifAssignments) {
      if (typeof asgn.token === 'number' && asgn.token >= 0 && asgn.token < resolvedLen) {
        const ca = asgn.source ? nativeMap.get(asgn.source) : undefined;
        if (ca) {
          matchedIndices.push(asgn.token);
          nativeCoords.push(ca);
        }
      }
    }
    if (matchedIndices.length > 0) {
      return {
        motifIndices: matchedIndices,
        fixedResidueIndices: matchedIndices.map((idx) => idx + 1),
        nativeMotifCoords: nativeCoords,
      };
    }
    return null;
  }

  if (effectiveContig && features?.is_protein && features?.ref_motif_token_type) {
    const isProt = features.is_protein;
    const motifType = features.ref_motif_token_type;
    const matchedIndices: number[] = [];
    let protIdx = 0;
    for (let t = 0; t < isProt.length; t++) {
      if (isProt[t] !== 1) continue;
      if (motifType[t * 3 + 1] === 1) {
        matchedIndices.push(protIdx);
      }
      protIdx++;
    }
    const nativeCoords = extractNativeMotifCaCoordsFromResidues(caResidues, effectiveContig);
    if (matchedIndices.length > 0) {
      return {
        motifIndices: matchedIndices,
        fixedResidueIndices: matchedIndices.map((idx) => idx + 1),
        ...(nativeCoords.length === matchedIndices.length
          ? { nativeMotifCoords: nativeCoords }
          : {}),
      };
    }
  }
  return null;
}
